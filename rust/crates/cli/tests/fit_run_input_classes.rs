//! A case for every input of `camdl fit run`, checked as its class predicts
//! (`rust/crates/cli/src/input_class.rs`, `FIT_RUN`; proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests").
//!
//! The runner, [`fit_into`], runs the release `camdl fit run` with its
//! working directory a fresh temporary store, in a cleared environment with an
//! explicit allowlist, and returns every stored leaf: its kind, its run id, its
//! level hashes (`fit`, `method`, `seed`), and the bytes its manifest checksums
//! (each file's SHA-256, checked against the digest `run.json` records). Each
//! case is one test, named `<class>_<input>` — `keyed_seed`,
//! `execution_parallel`, `expands_sweep`, `provenance_label`,
//! `removed_stage`; an environment variable's case is `<class>_env_<field>` —
//! and `input_class::tests::every_fit_run_input_has_a_case` fails if an input
//! has none.
//!
//! What each class asserts:
//!
//! - `keyed(level)`: changed alone, every fit leaf's run id changes, and so
//!   does its hash at the level the class table names ([`assert_rekeys`]). An
//!   environment variable the table keys at a simulation's `model` level enters
//!   a fit at its `fit` level, which holds the fit's model and engine.
//! - `execution`: the run ids and the stored bytes stay the same
//!   ([`assert_same_result`]).
//! - `expands`: the leaves common to both values keep their run ids
//!   ([`assert_shared_leaves_keep_ids`]).
//! - `presentation`, `provenance`: the leaves are unaffected — same run ids,
//!   same stored bytes ([`assert_leaf_unaffected`]).
//! - `removed`: the old spelling is refused, naming it, and nothing is stored
//!   ([`assert_removed`]).
//!
//! These are regression evidence on small fixtures, not proof: another model
//! could reach a path these do not (the proposal, "What tests can and cannot
//! show"). Each fit is tiny (a few iterations of a few particles), so the
//! file runs in seconds.
//!
//! The fixtures: `sir.camdl`, an SIR model with a parameter-only rate factor
//! that loop-invariant code motion hoists, observed through an incidence and
//! a prevalence stream; `patch.camdl`, a two-patch model whose coupling matrix
//! has structural zeros, which the constant fold drops. The fits run on the
//! compiled IR except where the compile itself is the input under test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use sha2::{Digest, Sha256};

// ─── Fixtures ────────────────────────────────────────────────────────────────

const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate        in [0.05, 2.0]  ~ log_normal(mu = -1.0, sigma = 0.3)
  gamma : rate        in [0.01, 1.0]  ~ log_normal(mu = -1.5, sigma = 0.3)
  rho   : probability in [0.05, 0.95]
  k     : probability in [0.01, 0.99]
}

let N = S + I + R

transitions {
  infection : S --> I @ beta * exp(-k) * S * I / N
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected)
  }
  prevalent {
    columns       { time : time, prevalent : count }
    projected     = I
    emit_schedule = every 1 'days
    prevalent     ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 20 'days }
"#;

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters {
  beta  : rate        in [0.01, 2.0]
  gamma : rate        in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
}
tables {
  W : patch × patch = [[0.0, 0.5],
                       [0.5, 0.0]]
}
let N[l in patch] = S[l] + I[l] + R[l]
transitions {
  infection[l in patch] : S[l] --> I[l] @ beta * S[l] * sum(q in patch, W[l, q] * I[q] / N[q])
  recovery[l in patch]  : I[l] --> R[l] @ gamma * I[l]
}
init {
  S[p0] = 990  I[p0] = 10
  S[p1] = 1000  I[p1] = 5
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = sum(l in patch, incidence(infection[l]))
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected)
  }
}
simulate { from = 0 'days  to = 10 'days }
"#;

/// The `[method]` tables the fits run, by name: each a few iterations of a few
/// particles.
const METHODS: &[(&str, &str)] = &[
    ("if2", "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 30\niterations = 3\ncooling = 0.9\n"),
    ("if2_one", "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 30\niterations = 3\ncooling = 0.9\n"),
    ("pgas", "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\nsweeps = 6\nburn_in = 2\n"),
    ("pmmh", "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\niterations = 20\nburn_in = 5\n"),
    ("pmmh_long", "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\niterations = 40\nburn_in = 5\n"),
    // `record_prequential` defaults on; off here so `--record-prequential`
    // turns it on.
    ("pfilter", "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 30\nreplicates = 1\nrecord_prequential = false\n"),
];

/// The fixture files, written under the target directory's scratch space
/// (`CARGO_TARGET_TMPDIR`) and removed when the last test holding them is
/// done, with the two stored fits the `--starts` and `--resume` cases read.
struct Fixture {
    dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
    /// A completed one-chain IF2 leaf: a `from_mle` source with no verdict
    /// to refuse.
    mle_source: PathBuf,
    /// A completed two-chain PMMH leaf of the `pmmh` config: a
    /// `from_posterior` source, and the base the `pmmh_long` config resumes.
    pmmh_base: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Fixture {
    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }

    fn create() -> Fixture {
        static N: AtomicUsize = AtomicUsize::new(0);
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        // The binary under test is built from the source this test was built
        // from: `--version` names the source fingerprint both builds embed.
        let version = Command::new(&camdl).arg("--version").output().unwrap();
        let version = String::from_utf8_lossy(&version.stdout);
        assert!(
            version.contains(&format!("+{} ", env!("CAMDL_SOURCE_LABEL"))),
            "{} is not built from this source ({}): {version} — run `make build`",
            camdl.display(),
            env!("CAMDL_SOURCE_LABEL")
        );
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "fit_run_input_classes-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).unwrap();
        write("sir.camdl", SIR);
        write("patch.camdl", PATCH);
        for stem in ["sir", "patch"] {
            let out = Command::new(&camdlc).env_clear().arg(dir.join(format!("{stem}.camdl"))).output().unwrap();
            assert!(out.status.success(), "camdlc {stem}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(format!("{stem}.ir.json")), out.stdout).unwrap();
        }
        let mut cases = String::from("time\tcases\n");
        let mut prevalent = String::from("time\tprevalent\n");
        for t in 0..=20 {
            if t > 0 {
                cases.push_str(&format!("{t}\t{}\n", 3 + t));
            }
            prevalent.push_str(&format!("{t}\t{}\n", 10 + 2 * t));
        }
        write("cases.tsv", &cases);
        write("prevalent.tsv", &prevalent);
        write("patch_cases.tsv", &(1..=10).fold(String::from("time\tcases\n"), |s, t| s + &format!("{t}\t{}\n", 2 + t)));
        write("truth.toml", "beta = 0.4\ngamma = 0.2\nrho = 0.5\nk = 0.3\n");
        let d = |name: &str| dir.join(name).to_string_lossy().into_owned();
        let problem = |model: &str, rho: f64| {
            format!(
                "[model]\ncamdl = \"{model}\"\n[data.observations]\ncases = \"{}\"\nprevalent = \"{}\"\n\
                 [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\ngamma = {{ bounds = [0.01, 1.0], start = 0.2 }}\n\
                 [fixed]\nrho = {rho}\nk = 0.3\n",
                d("cases.tsv"),
                d("prevalent.tsv"),
            )
        };
        for (name, method) in METHODS {
            write(&format!("{name}.toml"), &format!("{}[method]\n{method}", problem(&d("sir.ir.json"), 0.5)));
        }
        let if2 = METHODS[0].1;
        write("if2_rho.toml", &format!("{}[method]\n{if2}", problem(&d("sir.ir.json"), 0.6)));
        write("if2_source.toml", &format!("{}[method]\n{if2}", problem(&d("sir.camdl"), 0.5)));
        write(
            "if2_patch.toml",
            &format!(
                "[model]\ncamdl = \"{}\"\n[data.observations]\ncases = \"{}\"\n\
                 [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\n[fixed]\ngamma = 0.2\nrho = 0.5\n\
                 [method]\n{if2}",
                d("patch.camdl"),
                d("patch_cases.tsv"),
            ),
        );
        write(
            "if2_synthetic.toml",
            &format!(
                "[model]\ncamdl = \"{}\"\n[synthetic]\ntrue_params = \"{}\"\nsim_seeds = [1]\n\
                 [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\n[fixed]\ngamma = 0.2\nrho = 0.5\nk = 0.3\n\
                 [method]\n{if2}",
                d("sir.ir.json"),
                d("truth.toml"),
            ),
        );
        let mut fx = Fixture { dir: dir.clone(), camdl, camdlc, mle_source: PathBuf::new(), pmmh_base: PathBuf::new() };
        let sources = dir.join("sources");
        std::fs::create_dir_all(&sources).unwrap();
        // The one fit leaf a fixture fit adds to the source store.
        let source_fit = |config: &str| -> PathBuf {
            let before: BTreeSet<PathBuf> = leaves(&sources).into_iter().map(|l| l.dir).collect();
            let out = fx_command(&fx.camdl, &fx.camdlc, &fx.dir, &sources, &[], &[])
                .args(["fit", "run", &d(config), "--progress", "none"])
                .output()
                .unwrap();
            assert!(out.status.success(), "fixture fit {config}: {}", String::from_utf8_lossy(&out.stderr));
            let mut added: Vec<PathBuf> = leaves(&sources)
                .into_iter()
                .filter(|l| l.kind == "fit_method" && !before.contains(&l.dir))
                .map(|l| l.dir)
                .collect();
            assert_eq!(added.len(), 1, "fixture fit {config} adds one fit leaf: {added:?}");
            added.pop().unwrap()
        };
        fx.mle_source = source_fit("if2_one.toml");
        fx.pmmh_base = source_fit("pmmh.toml");
        fx
    }
}

/// The fixture, shared by the tests that hold it at once and created again
/// when none does.
fn fx() -> Arc<Fixture> {
    static LIVE: Mutex<Weak<Fixture>> = Mutex::new(Weak::new());
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(fx) = live.upgrade() {
        return fx;
    }
    let fx = Arc::new(Fixture::create());
    *live = Arc::downgrade(&fx);
    fx
}

// ─── The runner ──────────────────────────────────────────────────────────────

/// One invocation: the fit config, the arguments after it, and the variables
/// it sets in the child's otherwise cleared environment.
#[derive(Clone)]
struct Invocation {
    fx: Arc<Fixture>,
    args: Vec<String>,
    set: Vec<(String, String)>,
    /// Allowlisted variables this invocation drops.
    drop: Vec<String>,
}

impl Invocation {
    fn env(mut self, k: &str, v: &str) -> Self {
        self.set.push((k.into(), v.into()));
        self
    }
    fn without(mut self, k: &str) -> Self {
        self.drop.push(k.into());
        self
    }
}

/// `camdl fit run <fixture config> <extra>`.
fn fit(fx: &Arc<Fixture>, config: &str, extra: &[&str]) -> Invocation {
    let mut args = vec![fx.path(config)];
    args.extend(extra.iter().map(|s| s.to_string()));
    Invocation { fx: fx.clone(), args, set: Vec::new(), drop: Vec::new() }
}

/// The variables a run may inherit from the test process: what starting a
/// process needs, and nothing that configures camdl.
const ALLOWED: &[&str] = &["PATH", "TMPDIR"];

/// A `camdl` command in `cwd`, in a cleared environment holding [`ALLOWED`],
/// a private `HOME` and compiled-IR cache, and the compiler under test.
fn fx_command(camdl: &Path, camdlc: &Path, fixture: &Path, cwd: &Path, set: &[(String, String)], drop: &[String]) -> Command {
    let mut cmd = Command::new(camdl);
    cmd.current_dir(cwd).env_clear();
    for k in ALLOWED.iter().filter(|k| !drop.iter().any(|d| d == *k)) {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    cmd.env("HOME", fixture.join("home"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc)
        // A private compiled-IR cache: never the user's, never shared with
        // another test process.
        .env("CAMDL_IR_CACHE_DIR", fixture.join("ir-cache"));
    for k in drop {
        cmd.env_remove(k);
    }
    for (k, v) in set {
        cmd.env(k, v);
    }
    cmd
}

/// Run `camdl fit run` with `inv` in the store `store` (its working
/// directory, under which `results/` is the default store root) and return
/// the output.
fn run_in(store: &Path, inv: &Invocation) -> Output {
    let fx = &inv.fx;
    let mut cmd = fx_command(&fx.camdl, &fx.camdlc, &fx.dir, store, &inv.set, &inv.drop);
    cmd.args(["fit", "run"]).args(&inv.args);
    if !inv.args.iter().any(|a| a == "--progress" || a == "--no-progress") {
        cmd.args(["--progress", "none"]);
    }
    cmd.output().expect("spawn camdl fit run")
}

/// [`run_in`], requiring success, and every leaf the store holds afterwards.
fn fit_into(store: &Path, inv: &Invocation) -> Stored {
    let out = run_in(store, inv);
    assert!(
        out.status.success(),
        "camdl fit run {:?} (env {:?}) failed:\n{}",
        inv.args,
        inv.set,
        String::from_utf8_lossy(&out.stderr)
    );
    Stored { leaves: leaves(store) }
}

/// [`fit_into`] a fresh store.
fn run(inv: &Invocation) -> Stored {
    let store = tempfile::tempdir().unwrap();
    fit_into(store.path(), inv)
}

/// A stored leaf.
#[derive(Debug, Clone)]
struct Leaf {
    kind: String,
    run_id: String,
    /// Each level's name and hash, in path order.
    levels: BTreeMap<String, String>,
    /// Each file the manifest checksums, with the SHA-256 of its bytes.
    files: BTreeMap<String, String>,
    /// The leaf's `run.json`, parsed.
    record: serde_json::Value,
    dir: PathBuf,
}

#[derive(Debug)]
struct Stored {
    leaves: Vec<Leaf>,
}

impl Stored {
    fn fits(&self) -> BTreeMap<String, &Leaf> {
        self.leaves.iter().filter(|l| l.kind == "fit_method").map(|l| (l.run_id.clone(), l)).collect()
    }
    fn fit_ids(&self) -> BTreeSet<String> {
        self.fits().into_keys().collect()
    }
    /// The hashes the fit leaves hold at `level`.
    fn level_hashes(&self, level: &str) -> BTreeSet<String> {
        self.fits().values().map(|l| l.levels[level].clone()).collect()
    }
}

/// Every leaf under `root`: a directory holding a `run.json`, outside a
/// quarantined or earlier attempt.
fn leaves(root: &Path) -> Vec<Leaf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() || p.file_name().is_some_and(|n| n == ".quarantine" || n == ".staging" || n == ".attempts") {
                continue;
            }
            let record = p.join("run.json");
            if record.is_file() {
                out.push(leaf(&p, &record));
            }
            stack.push(p);
        }
    }
    out.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    out
}

fn leaf(dir: &Path, record: &Path) -> Leaf {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(record).unwrap()).unwrap();
    let mut files = BTreeMap::new();
    for (name, entry) in v["artifacts"].as_object().expect("artifacts") {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let digest = hex(&Sha256::digest(&bytes));
        assert_eq!(Some(digest.as_str()), entry["digest"].as_str(), "{}: {name} is the manifest's bytes", dir.display());
        files.insert(name.clone(), digest);
    }
    let levels = v["levels"]
        .as_array()
        .expect("levels")
        .iter()
        .map(|l| (l["name"].as_str().unwrap().to_string(), l["hash"].as_str().unwrap().to_string()))
        .collect();
    Leaf {
        kind: v["kind"].as_str().unwrap().to_string(),
        run_id: v["run_id"].as_str().unwrap().to_string(),
        levels,
        files,
        record: v,
        dir: dir.to_path_buf(),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The level the class table declares for a keyed input: `input` is a
/// `camdl fit run` argument id, or `env:<field>` for an environment variable.
/// Read from the tables' source (`input_class::FIT_RUN`, `env::VARS`), which
/// this integration test cannot import. An environment variable keyed at a
/// simulation's `model` level enters a fit at its `fit` level.
fn declared_level(input: &str) -> String {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let (file, needle, after) = match input.strip_prefix("env:") {
        Some(f) => ("env.rs", format!("\"{f}\", Keyed("), None),
        None => ("input_class.rs", format!("(\"{input}\", Keyed("), Some("pub const FIT_RUN")),
    };
    let text = std::fs::read_to_string(src.join(file)).unwrap();
    let text = match after {
        Some(marker) => &text[text.find(marker).expect("the FIT_RUN table")..],
        None => text.as_str(),
    };
    let line = text
        .lines()
        .find(|l| l.contains(&needle))
        .unwrap_or_else(|| panic!("`{input}` is not keyed in {file}"));
    let level = line.split(&needle).nth(1).unwrap().split(')').next().unwrap().to_ascii_lowercase();
    match level.as_str() {
        "model" if input.starts_with("env:") => "fit".to_string(),
        _ => level,
    }
}

// ─── The assertions, one per class ──────────────────────────────────────────

/// `keyed(level)`: every fit leaf of `with` has a run id `base` does not, and
/// a hash at the declared level `base` does not.
fn assert_rekeys(input: &str, base: &Invocation, with: &Invocation) {
    assert_rekeyed(input, &run(base), &run(with));
}

fn assert_rekeyed(input: &str, a: &Stored, b: &Stored) {
    let level = declared_level(input);
    let (ia, ib) = (a.fit_ids(), b.fit_ids());
    assert!(!ia.is_empty() && !ib.is_empty(), "both runs store a fit leaf");
    assert!(ia.is_disjoint(&ib), "{input} is keyed, so it re-keys every leaf: {ia:?} vs {ib:?}");
    let (la, lb) = (a.level_hashes(&level), b.level_hashes(&level));
    assert!(
        la.is_disjoint(&lb),
        "{input} is classified keyed({level}), so it re-keys the `{level}` level: {la:?} vs {lb:?}"
    );
}

/// The kind, run id and stored bytes of every leaf.
fn result(s: &Stored) -> Vec<(String, String, BTreeMap<String, String>)> {
    s.leaves.iter().map(|l| (l.kind.clone(), l.run_id.clone(), l.files.clone())).collect()
}

/// `execution`: the same run ids and the same stored bytes.
fn assert_same_result(a: &Stored, b: &Stored) {
    assert!(!a.fits().is_empty(), "the run stores a fit leaf");
    assert_eq!(result(a), result(b), "an execution input leaves the run ids and the stored bytes unchanged");
}

fn assert_same(a: &Invocation, b: &Invocation) {
    assert_same_result(&run(a), &run(b));
}

/// `expands`: every leaf of the smaller set is a leaf of the larger one, under
/// the same run id, and the larger set has more.
fn assert_shared_leaves_keep_ids(small: &Invocation, large: &Invocation) {
    let (a, b) = (run(small).fit_ids(), run(large).fit_ids());
    assert!(!a.is_empty(), "the smaller set stores a leaf");
    assert!(a.is_subset(&b), "the shared leaves keep their run ids: {a:?} not within {b:?}");
    assert!(b.len() > a.len(), "the larger set has more leaves");
}

/// `presentation` / `provenance`: the same leaves, run ids and stored bytes.
fn assert_leaf_unaffected(input: &str, base: &Invocation, with: &Invocation) {
    let (a, b) = (run(base), run(with));
    assert!(!a.fits().is_empty(), "the run stores a fit leaf");
    assert_eq!(result(&a), result(&b), "{input} leaves the run ids and the stored bytes unchanged");
}

/// `removed`: the run is refused, the message names the old spelling, and
/// the store holds nothing.
fn assert_removed(spelling: &str, with: &Invocation) {
    let store = tempfile::tempdir().unwrap();
    let out = run_in(store.path(), with);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{spelling} is refused:\n{err}");
    assert!(err.contains(spelling), "the refusal names {spelling}:\n{err}");
    assert!(leaves(store.path()).is_empty(), "a refused run stores nothing");
}

// ─── keyed ───────────────────────────────────────────────────────────────────

#[test]
fn keyed_config() {
    let fx = fx();
    assert_rekeys("config", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2_rho.toml", &[]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    let fx = fx();
    assert_rekeys("allow_forcing_extrapolation", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_seed() {
    let fx = fx();
    assert_rekeys("seed", &fit(&fx, "if2.toml", &["--seed", "1"]), &fit(&fx, "if2.toml", &["--seed", "2"]));
}

#[test]
fn keyed_starts() {
    let fx = fx();
    assert_rekeys("starts", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--starts", "single"]));
}

#[test]
fn keyed_starts_chains() {
    let fx = fx();
    let source = format!("from_posterior={}", fx.pmmh_base.display());
    assert_rekeys(
        "starts_chains",
        // The subset is judged by its own R̂, which a fixture this short need
        // not pass: the verdict is lifted on both sides.
        &fit(&fx, "if2.toml", &["--starts", &source, "--allow-nonconverged-source"]),
        &fit(&fx, "if2.toml", &["--starts", &source, "--starts-chains", "0,1", "--allow-nonconverged-source"]),
    );
}

#[test]
fn keyed_resume() {
    let fx = fx();
    let base = fx.pmmh_base.to_string_lossy().into_owned();
    assert_rekeys("resume", &fit(&fx, "pmmh_long.toml", &[]), &fit(&fx, "pmmh_long.toml", &["--resume", &base]));
}

#[test]
fn keyed_emit_every() {
    let fx = fx();
    assert_rekeys(
        "emit_every",
        &fit(&fx, "if2_synthetic.toml", &[]),
        &fit(&fx, "if2_synthetic.toml", &["--emit-every", "prevalent=2"]),
    );
}

#[test]
fn keyed_decibans_thresh() {
    let fx = fx();
    assert_rekeys("decibans_thresh", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--decibans-thresh", "5"]));
}

#[test]
fn keyed_no_dt_check() {
    let fx = fx();
    assert_rekeys("no_dt_check", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--no-dt-check"]));
}

#[test]
fn keyed_dt_check_strict() {
    let fx = fx();
    assert_rekeys("dt_check_strict", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--dt-check-strict"]));
}

#[test]
fn keyed_dt_check_halvings() {
    let fx = fx();
    assert_rekeys("dt_check_halvings", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--dt-check-halvings", "3"]));
}

#[test]
fn keyed_binomial() {
    let fx = fx();
    assert_rekeys("binomial", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--binomial", "btrs"]));
}

#[test]
fn keyed_cooling_target_iters() {
    let fx = fx();
    assert_rekeys("cooling_target_iters", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--cooling-target-iters", "2"]));
}

#[test]
fn keyed_max_tree_depth() {
    let fx = fx();
    assert_rekeys("max_tree_depth", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--max-tree-depth", "3"]));
}

#[test]
fn keyed_trajectory_warmup() {
    let fx = fx();
    assert_rekeys("trajectory_warmup", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--trajectory-warmup", "1"]));
}

#[test]
fn keyed_csmc_sweeps_per_nuts() {
    let fx = fx();
    assert_rekeys("csmc_sweeps_per_nuts", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--csmc-sweeps-per-nuts", "2"]));
}

#[test]
fn keyed_n_trajectories() {
    let fx = fx();
    assert_rekeys("n_trajectories", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--n-trajectories", "2"]));
}

#[test]
fn keyed_diagonal_mass() {
    let fx = fx();
    assert_rekeys("diagonal_mass", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--diagonal-mass"]));
}

#[test]
fn keyed_no_nuts() {
    let fx = fx();
    assert_rekeys("no_nuts", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--no-nuts"]));
}

#[test]
fn keyed_no_ancestor_sampling() {
    let fx = fx();
    assert_rekeys("no_ancestor_sampling", &fit(&fx, "pgas.toml", &[]), &fit(&fx, "pgas.toml", &["--no-ancestor-sampling"]));
}

#[test]
fn keyed_no_adapt() {
    let fx = fx();
    assert_rekeys("no_adapt", &fit(&fx, "pmmh.toml", &[]), &fit(&fx, "pmmh.toml", &["--no-adapt"]));
}

#[test]
fn keyed_adapt_start() {
    let fx = fx();
    assert_rekeys("adapt_start", &fit(&fx, "pmmh.toml", &[]), &fit(&fx, "pmmh.toml", &["--adapt-start", "3"]));
}

#[test]
fn keyed_record_ancestry() {
    let fx = fx();
    assert_rekeys("record_ancestry", &fit(&fx, "pfilter.toml", &[]), &fit(&fx, "pfilter.toml", &["--record-ancestry"]));
}

#[test]
fn keyed_record_prequential() {
    let fx = fx();
    assert_rekeys("record_prequential", &fit(&fx, "pfilter.toml", &[]), &fit(&fx, "pfilter.toml", &["--record-prequential"]));
}

/// `--no-licm` changes the IR the compiler emits for a model with a
/// parameter-only rate factor, and every compile the fit makes is given it:
/// the IR cache, private to the run, holds no entry compiled with LICM.
#[test]
fn keyed_no_licm() {
    let fx = fx();
    let cache = tempfile::tempdir().unwrap();
    let with = fit(&fx, "if2_source.toml", &["--no-licm"]).env("CAMDL_IR_CACHE_DIR", &cache.path().to_string_lossy());
    let (a, b) = (run(&fit(&fx, "if2_source.toml", &[])), run(&with));
    assert_rekeyed("no_licm", &a, &b);
    let entries: Vec<PathBuf> = std::fs::read_dir(cache.path())
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json") && !p.to_string_lossy().ends_with(".deps.json"))
        .collect();
    assert!(!entries.is_empty(), "the run compiled into its cache");
    for p in entries {
        let ir = std::fs::read_to_string(&p).unwrap();
        assert!(!ir.contains("PerEvalRef"), "{} was compiled with LICM under --no-licm", p.display());
    }
}

#[test]
fn keyed_env_no_licm() {
    let fx = fx();
    assert_rekeys("env:no_licm", &fit(&fx, "if2_source.toml", &[]), &fit(&fx, "if2_source.toml", &[]).env("CAMDL_NO_LICM", "1"));
}

#[test]
fn keyed_env_no_constant_fold() {
    let fx = fx();
    assert_rekeys(
        "env:no_constant_fold",
        &fit(&fx, "if2_patch.toml", &[]),
        &fit(&fx, "if2_patch.toml", &[]).env("CAMDL_NO_CONSTANT_FOLD", "1"),
    );
}

/// An engine switch keys the fit, and the leaf's record names it beside the
/// build: `engine_version` is the same as a default run's.
#[test]
fn keyed_env_eval_unresolved() {
    let fx = fx();
    let (a, b) = (run(&fit(&fx, "if2.toml", &[])), run(&fit(&fx, "if2.toml", &[]).env("CAMDL_EVAL_UNRESOLVED", "1")));
    assert_rekeyed("env:eval_unresolved", &a, &b);
    let (plain, switched) = (a.fits().into_values().next().unwrap().clone(), b.fits().into_values().next().unwrap().clone());
    assert_eq!(switched.record["engine_version"], plain.record["engine_version"], "engine_version is the build");
    assert_eq!(switched.record["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
    assert!(plain.record.get("engine_switches").is_none(), "a default run's record has no switch list");
    // The stored result is the default run's: the switch is an alternative
    // evaluation path, keyed until its equivalence is better established.
    // `mle_params.toml` records the fit level's hash, which the switch moves.
    let without_hash = |l: &Leaf| -> BTreeMap<String, String> {
        l.files.iter().filter(|(k, _)| k.as_str() != "mle_params.toml").map(|(k, v)| (k.clone(), v.clone())).collect()
    };
    assert_eq!(without_hash(&switched), without_hash(&plain), "the same stored bytes under the alternative path");
}

#[test]
fn keyed_env_no_binding_cache() {
    let fx = fx();
    assert_rekeys("env:no_binding_cache", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let fx = fx();
    let with = fit(&fx, "if2.toml", &[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1");
    let (a, b) = (run(&fit(&fx, "if2.toml", &[])), run(&with));
    assert_rekeyed("env:experimental", &a, &b);
    let leaf = b.fits().into_values().next().unwrap().clone();
    assert_eq!(leaf.record["provenance"]["experimental"]["CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST"], "1");
    assert_eq!(
        leaf.record["engine_switches"],
        serde_json::json!([{ "name": "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "value": "1" }])
    );
}

// ─── execution ───────────────────────────────────────────────────────────────

#[test]
fn execution_parallel() {
    // On each method that runs chains or particles in parallel: IF2, PGAS
    // and PMMH.
    let fx = fx();
    for config in ["if2.toml", "pgas.toml", "pmmh.toml"] {
        assert_same(&fit(&fx, config, &["--parallel", "1"]), &fit(&fx, config, &["--parallel", "4"]));
    }
}

#[test]
fn execution_force() {
    // A forced recompute into the store that holds the run stores the same
    // leaf; the incumbent is quarantined.
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let first = fit_into(store.path(), &fit(&fx, "if2.toml", &[]));
    let forced = fit_into(store.path(), &fit(&fx, "if2.toml", &["--force"]));
    assert_same_result(&first, &forced);
    assert!(store.path().join("results/.quarantine").is_dir(), "--force displaced the incumbent");
}

#[test]
fn execution_allow_nonconverged_source() {
    // A source with no verdict to refuse runs either way, and the flag
    // changes nothing it computes.
    let fx = fx();
    let source = format!("from_mle={}", fx.mle_source.display());
    assert_same(
        &fit(&fx, "if2.toml", &["--starts", &source]),
        &fit(&fx, "if2.toml", &["--starts", &source, "--allow-nonconverged-source"]),
    );
}

#[test]
fn execution_no_ir_cache() {
    let fx = fx();
    assert_same(&fit(&fx, "if2_source.toml", &[]), &fit(&fx, "if2_source.toml", &["--no-ir-cache"]));
}

#[test]
fn execution_env_no_ir_cache() {
    let fx = fx();
    assert_same(&fit(&fx, "if2_source.toml", &[]), &fit(&fx, "if2_source.toml", &[]).env("CAMDL_NO_IR_CACHE", "1"));
}

#[test]
fn execution_env_ir_cache_dir() {
    let fx = fx();
    let elsewhere = tempfile::tempdir().unwrap();
    let moved = fit(&fx, "if2_source.toml", &[]).env("CAMDL_IR_CACHE_DIR", &elsewhere.path().to_string_lossy());
    assert_same(&fit(&fx, "if2_source.toml", &[]), &moved);
}

#[test]
fn execution_env_xdg_cache_home() {
    // With no `CAMDL_IR_CACHE_DIR`, the cache lives under `XDG_CACHE_HOME`.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| fit(&fx, "if2_source.toml", &[]).without("CAMDL_IR_CACHE_DIR").env("XDG_CACHE_HOME", &d.to_string_lossy());
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_home() {
    // With neither `CAMDL_IR_CACHE_DIR` nor `XDG_CACHE_HOME`, under `HOME`.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| fit(&fx, "if2_source.toml", &[]).without("CAMDL_IR_CACHE_DIR").env("HOME", &d.to_string_lossy());
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_camdlc_path() {
    // The same compiler named by `CAMDLC_PATH` (which outranks `CAMDLC`).
    let fx = fx();
    let camdlc = fx.camdlc.to_string_lossy().into_owned();
    assert_same(
        &fit(&fx, "if2_source.toml", &["--no-ir-cache"]),
        &fit(&fx, "if2_source.toml", &["--no-ir-cache"]).env("CAMDLC_PATH", &camdlc),
    );
}

#[test]
fn execution_env_camdlc() {
    // The same compiler named by `CAMDLC`, against the one found on `PATH`.
    let fx = fx();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.camdlc, bin.path().join("camdlc")).unwrap();
    let path = format!("{}:{}", bin.path().display(), std::env::var("PATH").unwrap_or_default());
    let on_path = fit(&fx, "if2_source.toml", &["--no-ir-cache"]).without("CAMDLC").env("PATH", &path);
    assert_same(&fit(&fx, "if2_source.toml", &["--no-ir-cache"]), &on_path);
}

#[test]
fn execution_env_output_dir() {
    // The same fit into the default store root and into the one
    // `CAMDL_OUTPUT_DIR` names: the stored leaves are the same.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let elsewhere = b.path().join("elsewhere");
    let moved = fit(&fx, "if2.toml", &[]).env("CAMDL_OUTPUT_DIR", &elsewhere.to_string_lossy());
    let (x, y) = (fit_into(a.path(), &fit(&fx, "if2.toml", &[])), fit_into(b.path(), &moved));
    assert!(y.leaves.iter().all(|l| l.dir.starts_with(&elsewhere)), "the store root moved");
    assert_same_result(&x, &y);
}

// ─── expands ─────────────────────────────────────────────────────────────────

#[test]
fn expands_sweep() {
    let fx = fx();
    assert_shared_leaves_keep_ids(&fit(&fx, "if2.toml", &["--sweep", "rho=0.5"]), &fit(&fx, "if2.toml", &["--sweep", "rho=0.5,0.6"]));
}

// ─── presentation and provenance ────────────────────────────────────────────

#[test]
fn provenance_label() {
    let fx = fx();
    assert_leaf_unaffected("--label", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--label", "a-label"]));
}

#[test]
fn presentation_progress() {
    let fx = fx();
    assert_leaf_unaffected("--progress", &fit(&fx, "if2.toml", &["--progress", "none"]), &fit(&fx, "if2.toml", &["--progress", "plain"]));
}

#[test]
fn presentation_no_progress() {
    let fx = fx();
    assert_leaf_unaffected("--no-progress", &fit(&fx, "if2.toml", &["--progress", "plain"]), &fit(&fx, "if2.toml", &["--no-progress"]));
}

#[test]
fn presentation_verbosity() {
    let fx = fx();
    assert_leaf_unaffected("--verbosity", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &["--verbosity", "debug"]));
}

#[test]
fn presentation_env_trace_steps() {
    let fx = fx();
    // On the one-pass particle filter: the trace writes a row per substep.
    assert_leaf_unaffected("CAMDL_TRACE_STEPS", &fit(&fx, "pfilter.toml", &[]), &fit(&fx, "pfilter.toml", &[]).env("CAMDL_TRACE_STEPS", "1"));
}

#[test]
fn presentation_env_no_color() {
    let fx = fx();
    assert_leaf_unaffected("NO_COLOR", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &[]).env("NO_COLOR", "1"));
}

#[test]
fn presentation_env_clicolor_force() {
    let fx = fx();
    assert_leaf_unaffected("CLICOLOR_FORCE", &fit(&fx, "if2.toml", &[]), &fit(&fx, "if2.toml", &[]).env("CLICOLOR_FORCE", "1"));
}

// ─── the fit file is read once ──────────────────────────────────────────────

/// `fit run` reads its fit file once and stores that text in every fit folder
/// it writes, whatever the file says later. The file here is a named pipe that
/// serves the config on its first read and an edited config on every later
/// one, under a two-point sweep (two fit folders): every archive is the first
/// text, and every folder's `fit_toml_hash` is that text's.
#[test]
fn the_fit_file_is_read_once() {
    use std::io::Write;
    let fx = fx();
    let dir = tempfile::tempdir().unwrap();
    let fifo = dir.path().join("fit.toml");
    let made = Command::new("mkfifo").arg(&fifo).status().expect("mkfifo");
    assert!(made.success(), "mkfifo {}", fifo.display());
    let first = std::fs::read_to_string(fx.dir.join("if2.toml")).unwrap();
    let edited = format!("{first}\n# edited while the fit ran\n");
    {
        let (fifo, first, edited) = (fifo.clone(), first.clone(), edited.clone());
        // Each open of the pipe for writing waits for a reader; the first
        // reader gets the config, every later one the edit. The thread is left
        // waiting on an open no reader makes.
        std::thread::spawn(move || {
            let mut text = &first;
            while let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(&fifo) {
                let _ = f.write_all(text.as_bytes());
                text = &edited;
            }
        });
    }
    let inv = Invocation {
        fx: fx.clone(),
        args: vec![fifo.to_string_lossy().into_owned(), "--sweep".into(), "k=0.3,0.4".into()],
        set: Vec::new(),
        drop: Vec::new(),
    };
    let store = tempfile::tempdir().unwrap();
    let stored = fit_into(store.path(), &inv);
    let segments: BTreeSet<PathBuf> =
        stored.fits().values().map(|l| l.dir.parent().unwrap().parent().unwrap().to_path_buf()).collect();
    assert_eq!(segments.len(), 2, "one fit folder per sweep point");
    let hash = hex(&Sha256::digest(first.as_bytes()));
    for seg in segments {
        let archived = std::fs::read_to_string(seg.join("fit.toml.original")).unwrap();
        assert_eq!(archived, first, "{}: the archive is the text resolved", seg.display());
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(seg.join("fit.meta.json")).unwrap()).unwrap();
        assert_eq!(meta["fit_toml_hash"], serde_json::json!(hash), "{}: the hash is the text resolved's", seg.display());
    }
}

/// `camdl show` on a fit run names the engine that computed it: the build,
/// and the engine switches its identity keyed, in the text and the JSON.
#[test]
fn show_names_a_fit_runs_engine_and_its_switches() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let stored = fit_into(store.path(), &fit(&fx, "if2.toml", &[]).env("CAMDL_EVAL_UNRESOLVED", "1"));
    let leaf = stored.fits().into_values().next().unwrap().clone();
    let show = |extra: &[&str]| -> String {
        let out = fx_command(&fx.camdl, &fx.camdlc, &fx.dir, store.path(), &[], &[])
            .arg("show")
            .arg(&leaf.dir)
            .args(extra)
            .output()
            .unwrap();
        assert!(out.status.success(), "camdl show: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let build = leaf.record["engine_version"].as_str().unwrap().to_string();
    let text = show(&[]);
    let line = text.lines().find(|l| l.trim_start().starts_with("engine")).unwrap_or_else(|| panic!("an engine line:\n{text}"));
    assert!(line.contains(&build) && line.contains("CAMDL_EVAL_UNRESOLVED"), "{line}");
    let json: serde_json::Value = serde_json::from_str(&show(&["--format", "json"])).unwrap();
    assert_eq!(json["engine"]["build"], serde_json::json!(build));
    assert_eq!(json["engine"]["switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
}

// ─── removed ─────────────────────────────────────────────────────────────────

#[test]
fn removed_stage() {
    let fx = fx();
    assert_removed("--stage", &fit(&fx, "if2.toml", &["--stage", "posterior"]));
}

#[test]
fn removed_init() {
    let fx = fx();
    assert_removed("--init", &fit(&fx, "if2.toml", &["--init", "from_prior"]));
}

#[test]
fn removed_posterior() {
    let fx = fx();
    assert_removed("--posterior", &fit(&fx, "if2.toml", &["--posterior", "draws.tsv"]));
}

#[test]
fn removed_mle() {
    let fx = fx();
    assert_removed("--mle", &fit(&fx, "if2.toml", &["--mle", "mle.toml"]));
}

#[test]
fn removed_params() {
    let fx = fx();
    assert_removed("--params", &fit(&fx, "if2.toml", &["--params", "p.toml"]));
}

#[test]
fn removed_survey_path() {
    let fx = fx();
    assert_removed("--survey-path", &fit(&fx, "if2.toml", &["--survey-path", "survey"]));
}

#[test]
fn removed_survey_top_k() {
    let fx = fx();
    assert_removed("--survey-top-k", &fit(&fx, "if2.toml", &["--survey-top-k", "3"]));
}

#[test]
fn removed_allow_nonconverged_scout() {
    let fx = fx();
    assert_removed("--allow-nonconverged-scout", &fit(&fx, "if2.toml", &["--allow-nonconverged-scout"]));
}

#[test]
fn removed_init_method() {
    let fx = fx();
    assert_removed("--init-method", &fit(&fx, "if2.toml", &["--init-method", "lhs"]));
}

#[test]
fn removed_starts_from() {
    let fx = fx();
    assert_removed("--starts-from", &fit(&fx, "if2.toml", &["--starts-from", "fit"]));
}

#[test]
fn removed_tempering() {
    let fx = fx();
    assert_removed("--tempering", &fit(&fx, "pgas.toml", &["--tempering", "1,0.5"]));
}

#[test]
fn removed_rho() {
    let fx = fx();
    assert_removed("--rho", &fit(&fx, "pmmh.toml", &["--rho", "0.9"]));
}
