//! A case for every input of `camdl simulate`, checked as its class predicts
//! (`rust/crates/cli/src/input_class.rs`, proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests").
//!
//! The runner, [`simulate`], runs the release `camdl` into a store in a fresh
//! temporary directory (never the crate directory, gh#1252), in a cleared
//! environment with an explicit allowlist, and returns every stored leaf: its
//! kind, its run id, its level hashes, and the bytes its manifest checksums
//! (each file's SHA-256, checked against the digest `run.json` records). Each
//! case is one test, named `<class>_<input>` — `keyed_dt`,
//! `execution_parallel`, `expands_seeds`, `presentation_dates`,
//! `provenance_label`; an environment variable's case is
//! `<class>_env_<field>` — and `input_class::tests::every_simulate_input_has_a_case`
//! fails if an input has none.
//!
//! What each class asserts:
//!
//! - `keyed(level)`: changed alone, every cell's run id changes, and so does
//!   its hash at the level the class table names ([`assert_rekeys`], which
//!   reads the level from the table). `keyed(obs)`: the trajectory keeps its
//!   run id and the observation subtree under it re-keys.
//! - `execution`: the run ids and the stored bytes stay the same
//!   ([`assert_same_result`]).
//! - `expands`: the cells common to both values keep their run ids
//!   ([`assert_shared_cells_keep_ids`]).
//! - `presentation`, `provenance`: on a fixture where the input does something,
//!   the leaves are unaffected — same run ids, same stored bytes
//!   ([`assert_leaf_unaffected`]) — or, for an input that leaves the store,
//!   none is written ([`assert_writes_no_leaf`]).
//! - `child`: run after a run without it in the same store, every leaf keeps
//!   its run id and its manifest byte for byte and gains the child at its
//!   address; a store first written with it holds the same leaves
//!   ([`assert_child_written`]).
//!
//! These are regression evidence on small fixtures, not proof: another model
//! could reach a path these do not (the proposal, "What tests can and cannot
//! show").
//!
//! The fixtures: `sir.camdl`, an SIR model with a vaccination intervention,
//! two scenarios, a prior, an observation stream, a `quantities {}` block, a
//! lineage-annotated transition and a parameter-only rate factor that LICM
//! hoists; `sir_dated.camdl`, the same with a calendar origin; `patch.camdl`, a
//! two-patch model with an `external` table and a coupling matrix with
//! structural zeros, which the constant fold drops; and the repository's
//! reactive-policy fixture.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use sha2::{Digest, Sha256};

// ─── Fixtures ────────────────────────────────────────────────────────────────

const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta_a : rate        in [0.01, 5.0]  ~ log_normal(mu = -1.0, sigma = 0.3)
  beta_b : rate        in [0.01, 5.0]
  gamma  : rate        in [0.01, 5.0]
  rho    : probability in [0.001, 1.0]
}

transitions {
  #[lineage]
  infection_a : S --> I  @ beta_a * exp(-rho) * S * I / N
  infection_b : S --> I  @ beta_b * S * I / N
  recovery    : I --> R  @ gamma * I
}

init { S = 990  I = 10  R = 0  V = 0 }

interventions {
  sia : transfer(fraction = 0.5, from = S, to = V) at [5 'days]
}

scenarios {
  fast     { set = { beta_b = 0.3 } }
  with_sia { enable = [sia] }
}

simulate { from = 0 'days  to = 20 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection_a)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}

quantities {
  peak = max(I / N)
}
"#;

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0] }
tables {
  w : patch = external("w")
  W : patch × patch = [[0.0, 0.5],
                       [0.5, 0.0]]
}
let N[l in patch] = S[l] + I[l] + R[l]
transitions {
  infection[l in patch] : S[l] --> I[l] @ beta * w[l] * S[l] * sum(q in patch, W[l, q] * I[q] / N[q])
  recovery[l in patch]  : I[l] --> R[l] @ gamma * I[l]
}
init {
  S[p0] = 990  I[p0] = 10
  S[p1] = 1000  I[p1] = 5
}
simulate { from = 0 'days  to = 10 'days }
"#;

/// A reporting vocabulary for `--quantities`.
const VOCABULARY: &str = "quantities {\n  peak_i = max(I)\n}\n";

/// The fit config `--fit` names for `--draws prior` (a uniform prior on
/// `beta_a`, where the model declares a log-normal one) and `--design-from`
/// reads (the `cases` stream bound to `cases.tsv`).
const FIT_TOML: &str = r#"
[model]
camdl = "sir.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
beta_a = { bounds = [0.01, 1.0], start = 0.3, prior = { uniform = {} } }
[fixed]
beta_b = 0.2
gamma = 0.2
rho = 0.5
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 3
cooling = 0.7
"#;

/// The parameters of the repository's reactive-policy fixture.
const REACTIVE_PARAMS: &[&str] = &[
    "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "rho=0.2", "--param", "trigger_threshold=2",
    "--param", "sia_cov=0.7", "--param", "N0=1000", "--param", "I0=10",
];

/// The fixture files, written under the target directory's scratch space
/// (`CARGO_TARGET_TMPDIR`) and removed when the last test holding them is
/// done.
struct Fixture {
    dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
    reactive: PathBuf,
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
        let reactive = manifest.join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json");
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
            "simulate_input_classes-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("post")).unwrap();
        let fx = Fixture { dir, camdl, camdlc, reactive };
        let write = |name: &str, text: &str| std::fs::write(fx.dir.join(name), text).unwrap();
        write("sir.camdl", SIR);
        write("sir_long.camdl", &SIR.replace("to = 20 'days", "to = 25 'days"));
        write("sir_dated.camdl", &SIR.replace("time_unit = 'days\n", "time_unit = 'days\norigin = date(\"2025-01-01\")\n"));
        write("patch.camdl", PATCH);
        for stem in ["sir", "sir_long", "sir_dated", "patch"] {
            let out = Command::new(&fx.camdlc).env_clear().arg(fx.dir.join(format!("{stem}.camdl"))).output().unwrap();
            assert!(out.status.success(), "camdlc {stem}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(fx.dir.join(format!("{stem}.ir.json")), out.stdout).unwrap();
        }
        write("vocabulary.camdl", VOCABULARY);
        write("p.toml", "beta_a = 0.4\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n");
        write("p2.toml", "beta_a = 0.5\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n");
        write("patch.toml", "beta = 0.4\ngamma = 0.2\n");
        write("w1.tsv", "1.0\n0.5\n");
        write("w2.tsv", "1.0\n0.7\n");
        write("beta_vec.tsv", "a\t0.45\nb\t0.25\n");
        write("cases.tsv", "time\tcases\n1\t3\n2\t4\n3\t5\n");
        write("fit.toml", FIT_TOML);
        write("state.tsv", "# camdl-final-state v1\tt=10\nparticle\tS\tI\tR\tV\n0\t800\t50\t150\t0\n1\t810\t40\t150\t0\n");
        write("draws2.tsv", "beta_a\tbeta_b\n0.3\t0.2\n0.4\t0.2\n");
        write("draws3.tsv", "beta_a\tbeta_b\n0.3\t0.2\n0.4\t0.2\n0.5\t0.2\n");
        write("sweep2.tsv", "beta_a\n0.3\n0.4\n");
        write("sweep3.tsv", "beta_a\n0.3\n0.4\n0.5\n");
        let mut post = String::from("chain\tdraw\tbeta_a\tbeta_b\tgamma\trho\n");
        for chain in 0..3 {
            for draw in 0..2 {
                post.push_str(&format!("{chain}\t{draw}\t{}\t0.2\t0.2\t0.5\n", 0.3 + 0.05 * (2 * chain + draw) as f64));
            }
        }
        write("post/draws.tsv", &post);
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

/// One invocation: the arguments after `simulate`, and the variables it sets
/// in the child's otherwise cleared environment.
#[derive(Clone)]
struct Invocation {
    fx: Arc<Fixture>,
    args: Vec<String>,
    set: Vec<(String, String)>,
    /// Allowlisted variables this invocation drops.
    drop: Vec<String>,
}

impl Invocation {
    fn new(fx: &Arc<Fixture>, args: Vec<String>, extra: &[&str]) -> Self {
        let mut args = args;
        args.extend(extra.iter().map(|s| s.to_string()));
        Invocation { fx: fx.clone(), args, set: Vec::new(), drop: Vec::new() }
    }
    fn env(mut self, k: &str, v: &str) -> Self {
        self.set.push((k.into(), v.into()));
        self
    }
    fn without(mut self, k: &str) -> Self {
        self.drop.push(k.into());
        self
    }
}

/// The SIR model, from its compiled IR.
fn sir(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    Invocation::new(fx, vec![fx.path("sir.ir.json"), "--params".into(), fx.path("p.toml")], extra)
}

/// The SIR model with a calendar origin.
fn sir_dated(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    Invocation::new(fx, vec![fx.path("sir_dated.ir.json"), "--params".into(), fx.path("p.toml")], extra)
}

/// The SIR model compiled from source on each run, so that the compile
/// inputs (`--no-licm`, the IR cache) take part.
fn sir_source(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    Invocation::new(fx, vec![fx.path("sir.camdl"), "--params".into(), fx.path("p.toml")], extra)
}

/// The patch model compiled from source on each run, with its table.
fn patch_source(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    let args = vec![
        fx.path("patch.camdl"), "--params".into(), fx.path("patch.toml"),
        "--table".into(), format!("w={}", fx.path("w1.tsv")),
    ];
    Invocation::new(fx, args, extra)
}

/// The reactive-policy fixture.
fn reactive(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    let mut args = vec![fx.reactive.to_string_lossy().into_owned()];
    args.extend(REACTIVE_PARAMS.iter().map(|s| s.to_string()));
    Invocation::new(fx, args, extra)
}

/// A stored leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Leaf {
    kind: String,
    run_id: String,
    /// Each level's name and hash, in path order.
    levels: BTreeMap<String, String>,
    /// Each file the manifest checksums, with the SHA-256 of its bytes.
    files: BTreeMap<String, String>,
    /// The names of the observation subtrees under the leaf (`obs/<name>/`).
    obs: BTreeSet<String>,
    /// The manifest as stored: `run.json`'s `artifacts`, serialized.
    manifest: String,
    /// Every file under the leaf's declared children (`run.json`'s
    /// `children`), by its path relative to the leaf.
    child_files: BTreeSet<String>,
    /// The ids `run.json` declares under each child name.
    declared: BTreeMap<String, BTreeSet<String>>,
    /// The leaf's directory.
    dir: PathBuf,
}

impl Leaf {
    /// Whether a declared child holds a file at or under `address`.
    fn has_child(&self, address: &str) -> bool {
        self.child_files.iter().any(|f| f.starts_with(address))
    }
}

#[derive(Debug)]
struct Stored {
    leaves: Vec<Leaf>,
}

impl Stored {
    fn sims(&self) -> BTreeMap<String, &Leaf> {
        self.leaves.iter().filter(|l| l.kind == "sim").map(|l| (l.run_id.clone(), l)).collect()
    }
    fn sim_ids(&self) -> BTreeSet<String> {
        self.sims().into_keys().collect()
    }
    /// The hashes the cells hold at `level`.
    fn level_hashes(&self, level: &str) -> BTreeSet<String> {
        self.sims().values().map(|l| l.levels[level].clone()).collect()
    }
}

/// The variables a run may inherit from the test process: what starting a
/// process needs, and nothing that configures camdl.
const ALLOWED: &[&str] = &["PATH", "TMPDIR"];

/// Run `camdl simulate` with `inv` into the store at `store`, require success,
/// and return every leaf the store holds afterwards.
fn simulate_into(store: &Path, inv: &Invocation) -> Stored {
    let fx = &inv.fx;
    let mut cmd = Command::new(&fx.camdl);
    cmd.current_dir(store).env_clear();
    for k in ALLOWED.iter().filter(|k| !inv.drop.iter().any(|d| d == *k)) {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    cmd.env("HOME", fx.dir.join("home"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &fx.camdlc)
        // A private compiled-IR cache: never the user's, never shared with
        // another test process.
        .env("CAMDL_IR_CACHE_DIR", fx.dir.join("ir-cache"));
    for k in &inv.drop {
        cmd.env_remove(k);
    }
    for (k, v) in &inv.set {
        cmd.env(k, v);
    }
    cmd.arg("simulate").args(&inv.args);
    if !inv.args.iter().any(|a| a == "--progress" || a == "--no-progress") {
        cmd.args(["--progress", "none"]);
    }
    if !inv.args.iter().any(|a| a == "--output-dir") {
        cmd.arg("--output-dir").arg(store.join("results"));
    }
    let out = cmd.output().expect("spawn camdl simulate");
    assert!(
        out.status.success(),
        "camdl simulate {:?} (env {:?}) failed:\n{}",
        inv.args,
        inv.set,
        String::from_utf8_lossy(&out.stderr)
    );
    Stored { leaves: leaves(store) }
}

/// [`simulate_into`] a fresh store.
fn simulate(inv: &Invocation) -> Stored {
    let store = tempfile::tempdir().unwrap();
    simulate_into(store.path(), inv)
}

/// Every leaf under `root`: a directory holding a `run.json`.
fn leaves(root: &Path) -> Vec<Leaf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() || p.file_name().is_some_and(|n| n == ".quarantine" || n == ".staging") {
                continue;
            }
            let record = p.join("run.json");
            if record.is_file() {
                out.push(leaf(&p, &record));
            } else {
                stack.push(p);
            }
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
    let obs = std::fs::read_dir(dir.join("obs"))
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    let mut child_files = BTreeSet::new();
    for name in v["children"].as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()).unwrap_or_default() {
        let mut stack = vec![dir.join(&name)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    child_files.insert(p.strip_prefix(dir).unwrap().to_string_lossy().into_owned());
                }
            }
        }
    }
    Leaf {
        kind: v["kind"].as_str().unwrap().to_string(),
        run_id: v["run_id"].as_str().unwrap().to_string(),
        levels,
        files,
        obs,
        manifest: v["artifacts"].to_string(),
        child_files,
        declared: v["children"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, ids)| {
                        let ids = ids.as_array().unwrap().iter().map(|i| i.as_str().unwrap().to_string()).collect();
                        (k.clone(), ids)
                    })
                    .collect()
            })
            .unwrap_or_default(),
        dir: dir.to_path_buf(),
    }
}

/// Every child under `leaf` is at the address its declared id names: an
/// observation subtree `obs/<hash8>-<obs_seed>/` whose `obs.json` gives the
/// `obs_hash` and `obs_seed` of a declared id SHA-256(`"{run_id}:{obs_seed}:{obs_hash}"`),
/// and the event log `event_log/` declared as SHA-256(`"{run_id}:event_log"`).
fn assert_children_addressed(leaf: &Leaf) {
    let sha = |s: String| hex(&Sha256::digest(s.as_bytes()));
    for entry in std::fs::read_dir(leaf.dir.join("obs")).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(entry.path().join("obs.json")).unwrap()).unwrap();
        let (obs_hash, obs_seed) = (meta["obs_hash"].as_str().unwrap(), meta["obs_seed"].as_u64().unwrap());
        assert_eq!(name, format!("{}-{obs_seed}", &obs_hash[..8]), "obs/{name} is named by its obs.json");
        let id = sha(format!("{}:{obs_seed}:{obs_hash}", leaf.run_id));
        assert!(leaf.declared.get("obs").is_some_and(|ids| ids.contains(&id)),
            "obs/{name} is a declared child: {id} not in {:?}", leaf.declared.get("obs"));
    }
    if leaf.dir.join("event_log").exists() {
        let id = sha(format!("{}:event_log", leaf.run_id));
        assert_eq!(leaf.declared.get("event_log"), Some(&BTreeSet::from([id])), "event_log/ is a declared child");
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The level the class table declares for a keyed input: `field` is a
/// `camdl simulate` argument id, or `env:<field>` for an environment variable.
/// Read from the tables' source (`input_class::SIMULATE`, `env::VARS`), which
/// this integration test cannot import.
fn declared_level(input: &str) -> String {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let (file, needle) = match input.strip_prefix("env:") {
        Some(f) => ("env.rs", format!("\"{f}\", Keyed(")),
        None => ("input_class.rs", format!("(\"{input}\", Keyed(")),
    };
    let text = std::fs::read_to_string(src.join(file)).unwrap();
    let line = text
        .lines()
        .find(|l| l.contains(&needle))
        .unwrap_or_else(|| panic!("`{input}` is not keyed in {file}"));
    let level = line.split(&needle).nth(1).unwrap().split(')').next().unwrap();
    level.to_ascii_lowercase()
}

// ─── The assertions, one per class ──────────────────────────────────────────

/// `keyed(level)`: every cell of `with` has a run id `base` does not, and a
/// hash at the declared level `base` does not. For `keyed(obs)`, the
/// trajectory keeps its run id and the observation subtree re-keys.
fn assert_rekeys(input: &str, base: &Invocation, with: &Invocation) {
    let level = declared_level(input);
    let (a, b) = (simulate(base), simulate(with));
    if level == "obs" {
        assert_eq!(a.sim_ids(), b.sim_ids(), "{input}: the trajectory's identity is unchanged");
        for (id, leaf) in a.sims() {
            let other = &b.sims()[&id];
            assert!(!leaf.obs.is_empty(), "the run stores observations");
            assert!(leaf.obs.is_disjoint(&other.obs), "{input}: the observation subtree re-keys");
        }
        return;
    }
    let (ia, ib) = (a.sim_ids(), b.sim_ids());
    assert!(!ia.is_empty() && !ib.is_empty(), "both runs store a cell");
    assert!(ia.is_disjoint(&ib), "{input} is keyed, so it re-keys every cell: {ia:?} vs {ib:?}");
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
    assert!(!a.leaves.is_empty(), "the run stores a leaf");
    assert_eq!(result(a), result(b), "an execution input leaves the run ids and the stored bytes unchanged");
}

fn assert_same(a: &Invocation, b: &Invocation) {
    assert_same_result(&simulate(a), &simulate(b));
}

/// `expands`: every cell of the smaller set is a cell of the larger one, under
/// the same run id, and the larger set has more.
fn assert_shared_cells_keep_ids(small: &Invocation, large: &Invocation) {
    let (a, b) = (simulate(small).sim_ids(), simulate(large).sim_ids());
    assert!(!a.is_empty(), "the smaller set stores a cell");
    assert!(a.is_subset(&b), "the shared cells keep their run ids: {a:?} not within {b:?}");
    assert!(b.len() > a.len(), "the larger set has more cells");
}

/// `presentation` / `provenance`: the same leaves, run ids and stored bytes,
/// and `live` — the input's own output — exists after the run with it.
fn assert_leaf_unaffected(input: &str, base: &Invocation, with: &Invocation, live: Option<&Path>) {
    let (a, b) = (simulate(base), simulate(with));
    assert!(!a.leaves.is_empty(), "the run stores a leaf");
    assert_eq!(result(&a), result(&b), "{input} leaves the run ids and the stored bytes unchanged");
    if let Some(p) = live {
        assert!(p.exists(), "{input} wrote {}: the fixture exercises it", p.display());
    }
}

/// `child`: in one store, a run with the input after a run without it leaves
/// every leaf's run id and manifest (`run.json`'s `artifacts`, byte for byte)
/// as they were, and puts the child at `address` under each sim leaf, where
/// the run without it wrote nothing, at the address its declared id names
/// ([`assert_children_addressed`]); and a fresh store written with the input
/// holds the same run ids and stored bytes as one written without it, so what
/// a run id holds does not depend on which invocation wrote it first. `live`
/// — the input's own output outside the store — exists afterwards.
fn assert_child_written(input: &str, base: &Invocation, with: &Invocation, address: &str, live: Option<&Path>) {
    let store = tempfile::tempdir().unwrap();
    let a = simulate_into(store.path(), base);
    assert!(!a.leaves.is_empty(), "the run stores a leaf");
    assert!(a.leaves.iter().all(|l| !l.has_child(address)), "without {input}, no child at {address}");
    let b = simulate_into(store.path(), with);
    let manifests = |s: &Stored| -> Vec<(String, String)> {
        s.leaves.iter().map(|l| (l.run_id.clone(), l.manifest.clone())).collect()
    };
    assert_eq!(manifests(&a), manifests(&b), "{input} leaves every leaf's run id and manifest unchanged");
    assert!(b.sims().values().all(|l| l.has_child(address)), "{input} writes the child at {address}");
    b.sims().values().for_each(|l| assert_children_addressed(l));
    assert_eq!(result(&a), result(&simulate(with)), "{input}: a store first written with it holds the same leaves");
    if let Some(p) = live {
        assert!(p.exists(), "{input} wrote {}: the fixture exercises it", p.display());
    }
}

/// `presentation`, for an input that takes the run out of the store: it
/// succeeds and writes no leaf, and `live` exists afterwards.
fn assert_writes_no_leaf(input: &str, with: &Invocation, live: Option<&Path>) {
    let store = tempfile::tempdir().unwrap();
    let s = simulate_into(store.path(), with);
    assert!(s.leaves.is_empty(), "{input} writes no leaf: {:?}", s.leaves);
    if let Some(p) = live {
        assert!(p.exists(), "{input} wrote {}: the fixture exercises it", p.display());
    }
}

// ─── keyed ───────────────────────────────────────────────────────────────────

#[test]
fn keyed_model() {
    let fx = fx();
    let long = Invocation::new(&fx, vec![fx.path("sir_long.ir.json"), "--params".into(), fx.path("p.toml")], &[]);
    assert_rekeys("model", &sir(&fx, &[]), &long);
}

#[test]
fn keyed_to() {
    let fx = fx();
    assert_rekeys("to", &sir(&fx, &[]), &sir(&fx, &["--to", "15"]));
}

#[test]
fn keyed_init_state() {
    let fx = fx();
    let state = fx.path("state.tsv");
    assert_rekeys("init_state", &sir(&fx, &["--replicates", "2"]), &sir(&fx, &["--replicates", "2", "--init-state", &state]));
}

#[test]
fn keyed_allow_degenerate_rates() {
    let fx = fx();
    assert_rekeys("allow_degenerate_rates", &sir(&fx, &[]), &sir(&fx, &["--allow-degenerate-rates"]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    let fx = fx();
    assert_rekeys("allow_forcing_extrapolation", &sir(&fx, &[]), &sir(&fx, &["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_params() {
    let fx = fx();
    let mut other = sir(&fx, &[]);
    other.args[2] = fx.path("p2.toml");
    assert_rekeys("params", &sir(&fx, &[]), &other);
}

#[test]
fn keyed_param() {
    let fx = fx();
    assert_rekeys("param", &sir(&fx, &[]), &sir(&fx, &["--param", "beta_a=0.45"]));
}

#[test]
fn keyed_table() {
    let fx = fx();
    let mut other = patch_source(&fx, &[]);
    other.args[4] = format!("w={}", fx.path("w2.tsv"));
    assert_rekeys("table", &patch_source(&fx, &[]), &other);
}

#[test]
fn keyed_param_vec() {
    let fx = fx();
    let v = format!("beta={}", fx.path("beta_vec.tsv"));
    assert_rekeys("param_vec", &sir(&fx, &[]), &sir(&fx, &["--param-vec", &v]));
}

#[test]
fn keyed_fit() {
    let fx = fx();
    let fit = fx.path("fit.toml");
    assert_rekeys("fit", &sir(&fx, &["--draws", "prior", "-n", "2"]), &sir(&fx, &["--draws", "prior", "-n", "2", "--fit", &fit]));
}

#[test]
fn keyed_enable() {
    let fx = fx();
    assert_rekeys("enable", &sir(&fx, &[]), &sir(&fx, &["--enable", "sia"]));
}

#[test]
fn keyed_disable() {
    let fx = fx();
    assert_rekeys("disable", &sir(&fx, &["--enable", "sia"]), &sir(&fx, &["--disable", "sia"]));
}

#[test]
fn keyed_backend() {
    let fx = fx();
    assert_rekeys("backend", &sir(&fx, &[]), &sir(&fx, &["--backend", "gillespie"]));
}

#[test]
fn keyed_dt() {
    let fx = fx();
    assert_rekeys("dt", &sir(&fx, &[]), &sir(&fx, &["--dt", "0.5"]));
}

#[test]
fn keyed_integrator() {
    let fx = fx();
    assert_rekeys("integrator", &sir(&fx, &["--backend", "ode"]), &sir(&fx, &["--backend", "ode", "--integrator", "rk45"]));
}

#[test]
fn keyed_every() {
    let fx = fx();
    assert_rekeys("every", &sir(&fx, &[]), &sir(&fx, &["--output-every", "2"]));
}

#[test]
fn keyed_no_flows() {
    let fx = fx();
    assert_rekeys("no_flows", &sir(&fx, &[]), &sir(&fx, &["--no-flows"]));
}

#[test]
fn keyed_columns() {
    let fx = fx();
    assert_rekeys("columns", &sir(&fx, &[]), &sir(&fx, &["--columns", "S,I"]));
}

#[test]
fn keyed_seed() {
    let fx = fx();
    assert_rekeys("seed", &sir(&fx, &[]), &sir(&fx, &["--seed", "2"]));
}

#[test]
fn keyed_no_licm() {
    let fx = fx();
    assert_rekeys("no_licm", &sir_source(&fx, &[]), &sir_source(&fx, &["--no-licm"]));
}

#[test]
fn keyed_emit_every() {
    let fx = fx();
    let mirror = tempfile::tempdir().unwrap();
    let obs = mirror.path().join("obs").to_string_lossy().into_owned();
    assert_rekeys("emit_every", &sir(&fx, &["--obs-dir", &obs]), &sir(&fx, &["--obs-dir", &obs, "--emit-every", "2"]));
}

#[test]
fn keyed_env_eval_unresolved() {
    let fx = fx();
    let with = sir(&fx, &[]).env("CAMDL_EVAL_UNRESOLVED", "1");
    assert_rekeys("env:eval_unresolved", &sir(&fx, &[]), &with);
    // The record names the switch beside the build, and `engine_version` is
    // the build alone: the same as a default run's.
    let (plain, switched) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let _ = simulate_into(plain.path(), &sir(&fx, &[]));
    let _ = simulate_into(switched.path(), &with);
    let (plain, switched) = (records(plain.path()).pop().expect("a record"), records(switched.path()).pop().expect("a record"));
    assert_eq!(switched["engine_version"], plain["engine_version"], "engine_version is the build");
    assert_eq!(switched["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
    assert!(plain.get("engine_switches").is_none(), "a default run's record has no switch list: {plain}");
}

#[test]
fn keyed_env_no_binding_cache() {
    let fx = fx();
    assert_rekeys("env:no_binding_cache", &sir(&fx, &[]), &sir(&fx, &[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let fx = fx();
    let with = sir(&fx, &[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1");
    assert_rekeys("env:experimental", &sir(&fx, &[]), &with);
    // The record lists it.
    let store = tempfile::tempdir().unwrap();
    let _ = simulate_into(store.path(), &with);
    let record = records(store.path()).pop().expect("a record");
    assert_eq!(record["provenance"]["experimental"]["CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST"], "1");
}

#[test]
fn keyed_env_no_constant_fold() {
    let fx = fx();
    assert_rekeys("env:no_constant_fold", &patch_source(&fx, &[]), &patch_source(&fx, &[]).env("CAMDL_NO_CONSTANT_FOLD", "1"));
}

#[test]
fn keyed_env_no_licm() {
    let fx = fx();
    assert_rekeys("env:no_licm", &sir_source(&fx, &[]), &sir_source(&fx, &[]).env("CAMDL_NO_LICM", "1"));
}

/// Every record under `root`, parsed.
fn records(root: &Path) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                out.push(serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap());
            }
        }
    }
    out
}

// ─── expands ─────────────────────────────────────────────────────────────────

#[test]
fn expands_scenarios() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--scenario", "fast"]), &sir(&fx, &["--scenario", "fast", "--scenario", "with_sia"]));
}

#[test]
fn expands_seeds() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--seeds", "1:2"]), &sir(&fx, &["--seeds", "1:3"]));
}

#[test]
fn expands_replicates() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--replicates", "2"]), &sir(&fx, &["--replicates", "3"]));
}

#[test]
fn expands_draws() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--draws", &fx.path("draws2.tsv")]), &sir(&fx, &["--draws", &fx.path("draws3.tsv")]));
}

#[test]
fn expands_n_draws() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--draws", "prior", "-n", "2"]), &sir(&fx, &["--draws", "prior", "-n", "3"]));
}

#[test]
fn expands_sweep() {
    let fx = fx();
    assert_shared_cells_keep_ids(&sir(&fx, &["--sweep", "beta_a=0.3,0.4"]), &sir(&fx, &["--sweep", "beta_a=0.3,0.4,0.5"]));
}

#[test]
fn expands_sweep_file() {
    let fx = fx();
    assert_shared_cells_keep_ids(
        &sir(&fx, &["--sweep-file", &fx.path("sweep2.tsv")]),
        &sir(&fx, &["--sweep-file", &fx.path("sweep3.tsv")]),
    );
}

#[test]
fn expands_chains() {
    let fx = fx();
    let post = fx.path("post");
    let draws = ["--draws", "posterior", "--fit", &post];
    let one = [&draws[..], &["--chains", "0"]].concat();
    let two = [&draws[..], &["--chains", "0,1"]].concat();
    assert_shared_cells_keep_ids(&sir(&fx, &one), &sir(&fx, &two));
}

#[test]
fn expands_exclude_chains() {
    let fx = fx();
    let post = fx.path("post");
    let draws = ["--draws", "posterior", "--fit", &post];
    let fewer = [&draws[..], &["--exclude-chains", "1,2"]].concat();
    let more = [&draws[..], &["--exclude-chains", "2"]].concat();
    assert_shared_cells_keep_ids(&sir(&fx, &fewer), &sir(&fx, &more));
}

// ─── execution ───────────────────────────────────────────────────────────────

#[test]
fn execution_parallel() {
    let fx = fx();
    assert_same(&sir(&fx, &["--seeds", "1:4", "--parallel", "1"]), &sir(&fx, &["--seeds", "1:4", "--parallel", "4"]));
}

#[test]
fn execution_output_dir() {
    // The same run into two store roots: the stored leaves are the same.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let at = |d: &Path| sir(&fx, &["--output-dir", &d.join("elsewhere").to_string_lossy()]);
    assert_same_result(&simulate_into(a.path(), &at(a.path())), &simulate_into(b.path(), &at(b.path())));
}

#[test]
fn execution_force() {
    // A forced recompute into the store that holds the run stores the same
    // leaf; the incumbent is quarantined.
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let first = simulate_into(store.path(), &sir(&fx, &[]));
    let forced = simulate_into(store.path(), &sir(&fx, &["--force"]));
    assert_same_result(&first, &forced);
    assert!(store.path().join("results/.quarantine").is_dir(), "--force displaced the incumbent");
}

#[test]
fn execution_no_ir_cache() {
    let fx = fx();
    assert_same(&sir_source(&fx, &[]), &sir_source(&fx, &["--no-ir-cache"]));
}

#[test]
fn execution_env_no_ir_cache() {
    let fx = fx();
    assert_same(&sir_source(&fx, &[]), &sir_source(&fx, &[]).env("CAMDL_NO_IR_CACHE", "1"));
}

#[test]
fn execution_env_ir_cache_dir() {
    let fx = fx();
    let elsewhere = tempfile::tempdir().unwrap();
    let moved = sir_source(&fx, &[]).env("CAMDL_IR_CACHE_DIR", &elsewhere.path().to_string_lossy());
    assert_same(&sir_source(&fx, &[]), &moved);
}

#[test]
fn execution_env_xdg_cache_home() {
    // With no `CAMDL_IR_CACHE_DIR`, the cache lives under `XDG_CACHE_HOME`.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| sir_source(&fx, &[]).without("CAMDL_IR_CACHE_DIR").env("XDG_CACHE_HOME", &d.to_string_lossy());
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_home() {
    // With neither `CAMDL_IR_CACHE_DIR` nor `XDG_CACHE_HOME`, under `HOME`.
    let fx = fx();
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| sir_source(&fx, &[]).without("CAMDL_IR_CACHE_DIR").env("HOME", &d.to_string_lossy());
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_camdlc_path() {
    // The same compiler named by `CAMDLC_PATH` (which outranks `CAMDLC`).
    let fx = fx();
    let camdlc = fx.camdlc.to_string_lossy().into_owned();
    assert_same(&sir_source(&fx, &["--no-ir-cache"]), &sir_source(&fx, &["--no-ir-cache"]).env("CAMDLC_PATH", &camdlc));
}

#[test]
fn execution_env_camdlc() {
    // The same compiler named by `CAMDLC`, against the one found on `PATH`.
    let fx = fx();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.camdlc, bin.path().join("camdlc")).unwrap();
    let path = format!("{}:{}", bin.path().display(), std::env::var("PATH").unwrap_or_default());
    let on_path = sir_source(&fx, &["--no-ir-cache"]).without("CAMDLC").env("PATH", &path);
    assert_same(&sir_source(&fx, &["--no-ir-cache"]), &on_path);
}

// ─── presentation and provenance ────────────────────────────────────────────

#[test]
fn presentation_output() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let f = out.path().join("traj.tsv");
    assert_leaf_unaffected("--output", &sir(&fx, &[]), &sir(&fx, &["-o", &f.to_string_lossy()]), Some(&f));
}

// ─── child ───────────────────────────────────────────────────────────────────

// Each `--obs*` flag writes, besides its own mirror, the observation subtree
// `obs/<hash8>-<obs_seed>/` under the leaf.

#[test]
fn child_obs() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let f = out.path().join("obs.tsv");
    assert_child_written("--obs", &sir(&fx, &[]), &sir(&fx, &["--obs", &f.to_string_lossy()]), "obs/", Some(&f));
}

#[test]
fn child_obs_dir() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let d = out.path().join("obs");
    assert_child_written("--obs-dir", &sir(&fx, &[]), &sir(&fx, &["--obs-dir", &d.to_string_lossy()]), "obs/", Some(&d.join("cases.tsv")));
}

#[test]
fn child_obs_only() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let f = out.path().join("obs.tsv");
    assert_child_written("--obs-only", &sir(&fx, &[]), &sir(&fx, &["--obs-only", &f.to_string_lossy()]), "obs/", Some(&f));
}

#[test]
fn child_obs_only_dir() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let d = out.path().join("obs");
    let with = sir(&fx, &["--obs-only-dir", &d.to_string_lossy()]);
    assert_child_written("--obs-only-dir", &sir(&fx, &[]), &with, "obs/", Some(&d.join("cases.tsv")));
}

#[test]
fn child_event_log() {
    // The recorder is passive, so the trajectory and the run id are the same;
    // the log is the leaf's `event_log/` child, and the leaf's own files are
    // what a plain run stores.
    let fx = fx();
    assert_child_written("--event-log", &sir(&fx, &[]), &sir(&fx, &["--event-log"]), "event_log/event_log.tsv", None);
}

// ─── presentation and provenance (continued) ────────────────────────────────

#[test]
fn presentation_dates() {
    // On a model with a calendar origin, `--dates` adds a date column to the
    // mirror; the stored leaf is undated either way.
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let (a, b) = (out.path().join("a.tsv"), out.path().join("b.tsv"));
    let base = sir_dated(&fx, &["-o", &a.to_string_lossy()]);
    let with = sir_dated(&fx, &["-o", &b.to_string_lossy(), "--dates"]);
    assert_leaf_unaffected("--dates", &base, &with, None);
    let header = |p: &Path| -> String {
        let text = std::fs::read_to_string(p).unwrap();
        text.lines().find(|l| !l.starts_with('#')).unwrap_or_default().to_string()
    };
    assert!(!header(&a).contains("date") && header(&b).contains("\tdate\t"), "--dates dated the mirror");
}

#[test]
fn presentation_draws_out() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let f = out.path().join("draws.tsv");
    let base = sir(&fx, &["--draws", "prior", "-n", "2"]);
    let with = sir(&fx, &["--draws", "prior", "-n", "2", "--draws-out", &f.to_string_lossy()]);
    assert_leaf_unaffected("--draws-out", &base, &with, Some(&f));
}

#[test]
fn presentation_quantities_out() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let d = out.path().join("q");
    assert_leaf_unaffected("--quantities-out", &sir(&fx, &[]), &sir(&fx, &["--quantities-out", &d.to_string_lossy()]), Some(&d.join("quantities.json")));
}

#[test]
fn presentation_quantities() {
    // A vocabulary in place of the model's `quantities {}` block changes the
    // sidecar, never the leaf.
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let (a, b) = (out.path().join("a"), out.path().join("b"));
    let base = sir_source(&fx, &["--quantities-out", &a.to_string_lossy()]);
    let with = sir_source(&fx, &["--quantities-out", &b.to_string_lossy(), "--quantities", &fx.path("vocabulary.camdl")]);
    assert_leaf_unaffected("--quantities", &base, &with, None);
    assert!(std::fs::read_dir(&b).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("quantities-")), "--quantities wrote its keyed table");
}

#[test]
fn presentation_format() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let (a, b) = (out.path().join("a.log"), out.path().join("b.log"));
    let base = sir(&fx, &["--event-log", &a.to_string_lossy()]);
    let with = sir(&fx, &["--event-log", &b.to_string_lossy(), "--format", "tsv"]);
    assert_leaf_unaffected("--format", &base, &with, Some(&b));
}

#[test]
fn presentation_tsv() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let (a, b) = (out.path().join("a.log"), out.path().join("b.log"));
    let base = sir(&fx, &["--event-log", &a.to_string_lossy()]);
    let with = sir(&fx, &["--event-log", &b.to_string_lossy(), "--tsv"]);
    assert_leaf_unaffected("--tsv", &base, &with, Some(&b));
}

#[test]
fn presentation_reactive_log() {
    // On a model with a reactive policy, the leaf holds the firing log either
    // way; `--reactive-log` mirrors it.
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let f = out.path().join("reactive.tsv");
    assert_leaf_unaffected(
        "--reactive-log",
        &reactive(&fx, &["--enable", "sia"]),
        &reactive(&fx, &["--enable", "sia", "--reactive-log", &f.to_string_lossy()]),
        Some(&f),
    );
}

#[test]
fn presentation_progress() {
    let fx = fx();
    assert_leaf_unaffected("--progress", &sir(&fx, &["--progress", "none"]), &sir(&fx, &["--progress", "plain"]), None);
}

#[test]
fn presentation_no_progress() {
    let fx = fx();
    assert_leaf_unaffected("--no-progress", &sir(&fx, &["--progress", "plain"]), &sir(&fx, &["--no-progress"]), None);
}

#[test]
fn presentation_verbosity() {
    let fx = fx();
    assert_leaf_unaffected("--verbosity", &sir(&fx, &[]), &sir(&fx, &["--verbosity", "debug"]), None);
}

#[test]
fn presentation_env_trace_steps() {
    let fx = fx();
    assert_leaf_unaffected("CAMDL_TRACE_STEPS", &sir(&fx, &[]), &sir(&fx, &[]).env("CAMDL_TRACE_STEPS", "1"), None);
}

#[test]
fn presentation_env_no_color() {
    let fx = fx();
    assert_leaf_unaffected("NO_COLOR", &sir(&fx, &[]), &sir(&fx, &[]).env("NO_COLOR", "1"), None);
}

#[test]
fn presentation_env_clicolor_force() {
    let fx = fx();
    assert_leaf_unaffected("CLICOLOR_FORCE", &sir(&fx, &[]), &sir(&fx, &[]).env("CLICOLOR_FORCE", "1"), None);
}

#[test]
fn provenance_label() {
    let fx = fx();
    assert_leaf_unaffected("--label", &sir(&fx, &[]), &sir(&fx, &["--label", "a label"]), None);
}

#[test]
fn presentation_stdout() {
    let fx = fx();
    assert_writes_no_leaf("--stdout", &sir(&fx, &["--stdout"]), None);
}

#[test]
fn presentation_dry_run() {
    let fx = fx();
    assert_writes_no_leaf("--dry-run", &sir(&fx, &["--seeds", "1:2", "--dry-run"]), None);
}

#[test]
fn presentation_design_from() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let d = out.path().join("design");
    let with = sir(&fx, &["--design-from", &fx.path("fit.toml"), "--obs-only-dir", &d.to_string_lossy()]);
    assert_writes_no_leaf("--design-from", &with, Some(&d.join("cases.tsv")));
}

#[test]
fn presentation_projected_out() {
    let fx = fx();
    let out = tempfile::tempdir().unwrap();
    let d = out.path().join("projected");
    let with = sir(&fx, &["--design-from", &fx.path("fit.toml"), "--projected-out", &d.to_string_lossy()]);
    assert_writes_no_leaf("--projected-out", &with, Some(&d.join("cases.tsv")));
}
