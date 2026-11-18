//! Differential cases for every input of `camdl simulate` whose class is
//! checked by running camdl (`keyed`, `execution`, `expands`;
//! `rust/crates/cli/src/input_class.rs`, proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests").
//!
//! The runner, [`simulate`], runs the release `camdl` into a store in a fresh
//! temporary directory (never the crate directory, gh#1252) and returns every
//! stored leaf: its kind, its run id, and the bytes its manifest checksums (each
//! file's SHA-256, checked against the digest `run.json` records). Each case is
//! one test, named `<class>_<input>` — `keyed_dt`, `execution_parallel`,
//! `expands_seeds`; an environment variable's case is `<class>_env_<field>` —
//! and `input_class::tests::every_differential_simulate_input_has_a_case`
//! fails if a `keyed`, `execution` or `expands` input has none.
//!
//! What each class asserts:
//!
//! - `keyed`: changed alone, every cell's run id changes ([`assert_rekeys`]). A
//!   `keyed(obs)` input changes the address of the observation subtree under
//!   the leaf and leaves the trajectory's run id alone ([`assert_obs_rekeys`]).
//! - `execution`: changed alone, the run ids and the stored bytes stay the
//!   same ([`assert_same_result`]).
//! - `expands`: the cells common to both values keep their run ids
//!   ([`assert_shared_cells_keep_ids`]).
//!
//! These are regression evidence on two small fixtures, not proof: another
//! model could reach a path these do not (the proposal, "What tests can and
//! cannot show").
//!
//! The fixtures: `sir.camdl`, an SIR model with a vaccination intervention,
//! two scenarios, a prior, an observation stream and a parameter-only rate
//! factor that LICM hoists; `patch.camdl`, a two-patch model with an
//! `external` table and a coupling matrix with structural zeros, which the
//! constant fold drops.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

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

/// The fit config `--fit` names for `--draws prior`: a uniform prior on
/// `beta_a`, where the model declares a log-normal one.
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

struct Fixture {
    dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
}

impl Fixture {
    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }
}

/// Write the fixture files once per test process, under the target directory's
/// scratch space (`CARGO_TARGET_TMPDIR`), and compile the two models.
fn fixture() -> &'static Fixture {
    static FX: OnceLock<Fixture> = OnceLock::new();
    FX.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("simulate_input_classes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("post")).unwrap();
        let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).unwrap();
        write("sir.camdl", SIR);
        write("patch.camdl", PATCH);
        for stem in ["sir", "patch"] {
            let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(dir.join(format!("{stem}.camdl"))).output().unwrap();
            assert!(out.status.success(), "camdlc {stem}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(format!("{stem}.ir.json")), out.stdout).unwrap();
        }
        // A second model: the first with a longer horizon.
        write("sir_long.camdl", &SIR.replace("to = 20 'days", "to = 25 'days"));
        let out = Command::new(&camdlc).arg(dir.join("sir_long.camdl")).output().unwrap();
        assert!(out.status.success());
        std::fs::write(dir.join("sir_long.ir.json"), out.stdout).unwrap();

        write("p.toml", "beta_a = 0.4\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n");
        write("p2.toml", "beta_a = 0.5\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n");
        write("patch.toml", "beta = 0.4\ngamma = 0.2\n");
        write("w1.tsv", "1.0\n0.5\n");
        write("w2.tsv", "1.0\n0.7\n");
        write("beta_vec.tsv", "a\t0.45\nb\t0.25\n");
        write("cases.tsv", "time\tcases\n1\t3\n2\t4\n3\t5\n");
        write("fit.toml", FIT_TOML);
        let states = "# camdl-final-state v1\tt=10\nparticle\tS\tI\tR\tV\n0\t800\t50\t150\t0\n1\t810\t40\t150\t0\n";
        write("state.tsv", states);
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
        std::fs::write(dir.join("post/draws.tsv"), post).unwrap();
        Fixture { dir, camdl, camdlc }
    })
}

// ─── The runner ──────────────────────────────────────────────────────────────

/// One invocation: the arguments after `simulate`, and the environment it
/// changes from the runner's default.
#[derive(Clone, Default)]
struct Invocation {
    args: Vec<String>,
    set: Vec<(String, String)>,
    unset: Vec<String>,
}

fn sir(extra: &[&str]) -> Invocation {
    let fx = fixture();
    let mut args = vec![fx.path("sir.ir.json"), "--params".into(), fx.path("p.toml")];
    args.extend(extra.iter().map(|s| s.to_string()));
    Invocation { args, ..Default::default() }
}

/// The patch model, compiled from source on each run, so that the compile
/// inputs (`--no-licm`, `CAMDL_NO_CONSTANT_FOLD`, the IR cache) take part.
fn patch_source(extra: &[&str]) -> Invocation {
    let fx = fixture();
    let mut args = vec![
        fx.path("patch.camdl"), "--params".into(), fx.path("patch.toml"),
        "--table".into(), format!("w={}", fx.path("w1.tsv")),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    Invocation { args, ..Default::default() }
}

/// The SIR model compiled from source on each run.
fn sir_source(extra: &[&str]) -> Invocation {
    let fx = fixture();
    let mut args = vec![fx.path("sir.camdl"), "--params".into(), fx.path("p.toml")];
    args.extend(extra.iter().map(|s| s.to_string()));
    Invocation { args, ..Default::default() }
}

impl Invocation {
    fn env(mut self, k: &str, v: &str) -> Self {
        self.set.push((k.into(), v.into()));
        self
    }
    fn unset(mut self, k: &str) -> Self {
        self.unset.push(k.into());
        self
    }
}

/// A stored leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Leaf {
    kind: String,
    run_id: String,
    /// Each file the manifest checksums, with the SHA-256 of its bytes.
    files: BTreeMap<String, String>,
    /// The names of the observation subtrees under the leaf (`obs/<name>/`).
    obs: BTreeSet<String>,
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
}

/// Run `camdl simulate` with `inv` into the store at `store`, require success,
/// and return every leaf the store holds afterwards.
fn simulate_into(store: &Path, inv: &Invocation) -> Stored {
    let fx = fixture();
    let mut cmd = Command::new(&fx.camdl);
    cmd.current_dir(store)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &fx.camdlc)
        // A private compiled-IR cache: never the user's, never shared with
        // another test process.
        .env("CAMDL_IR_CACHE_DIR", fx.dir.join("ir-cache"))
        .arg("simulate")
        .args(&inv.args)
        .args(["--progress", "none"]);
    if !inv.args.iter().any(|a| a == "--output-dir") {
        cmd.arg("--output-dir").arg(store.join("results"));
    }
    for k in &inv.unset {
        cmd.env_remove(k);
    }
    for (k, v) in &inv.set {
        cmd.env(k, v);
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
    let obs = std::fs::read_dir(dir.join("obs"))
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    Leaf {
        kind: v["kind"].as_str().unwrap().to_string(),
        run_id: v["run_id"].as_str().unwrap().to_string(),
        files,
        obs,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ─── The assertions, one per class ──────────────────────────────────────────

/// `keyed`: every cell of `with` has a run id `base` does not.
fn assert_rekeys(base: &Invocation, with: &Invocation) {
    let (a, b) = (simulate(base).sim_ids(), simulate(with).sim_ids());
    assert!(!a.is_empty() && !b.is_empty(), "both runs store a cell");
    assert!(a.is_disjoint(&b), "the input is keyed, so it re-keys every cell: {a:?} vs {b:?}");
}

/// `keyed(obs)`: the trajectory keeps its run id and the observation subtree
/// under it is another address.
fn assert_obs_rekeys(base: &Invocation, with: &Invocation) {
    let (a, b) = (simulate(base), simulate(with));
    assert_eq!(a.sim_ids(), b.sim_ids(), "the trajectory's identity is unchanged");
    for (id, leaf) in a.sims() {
        let other = &b.sims()[&id];
        assert!(!leaf.obs.is_empty(), "the run stores observations");
        assert!(leaf.obs.is_disjoint(&other.obs), "{id}: the observation subtree re-keys: {:?} vs {:?}", leaf.obs, other.obs);
    }
}

/// `execution`: the same run ids and the same stored bytes.
fn assert_same_result(a: &Stored, b: &Stored) {
    assert!(!a.leaves.is_empty(), "the run stores a leaf");
    let strip = |s: &Stored| -> Vec<(String, String, BTreeMap<String, String>)> {
        s.leaves.iter().map(|l| (l.kind.clone(), l.run_id.clone(), l.files.clone())).collect()
    };
    assert_eq!(strip(a), strip(b), "an execution input leaves the run ids and the stored bytes unchanged");
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

// ─── keyed ───────────────────────────────────────────────────────────────────

#[test]
fn keyed_model() {
    let fx = fixture();
    let long = Invocation {
        args: vec![fx.path("sir_long.ir.json"), "--params".into(), fx.path("p.toml")],
        ..Default::default()
    };
    assert_rekeys(&sir(&[]), &long);
}

#[test]
fn keyed_to() {
    assert_rekeys(&sir(&[]), &sir(&["--to", "15"]));
}

#[test]
fn keyed_init_state() {
    let state = fixture().path("state.tsv");
    assert_rekeys(&sir(&["--replicates", "2"]), &sir(&["--replicates", "2", "--init-state", &state]));
}

#[test]
fn keyed_allow_degenerate_rates() {
    assert_rekeys(&sir(&[]), &sir(&["--allow-degenerate-rates"]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    assert_rekeys(&sir(&[]), &sir(&["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_params() {
    let p2 = fixture().path("p2.toml");
    let mut other = sir(&[]);
    other.args[2] = p2;
    assert_rekeys(&sir(&[]), &other);
}

#[test]
fn keyed_param() {
    assert_rekeys(&sir(&[]), &sir(&["--param", "beta_a=0.45"]));
}

#[test]
fn keyed_table() {
    let w2 = format!("w={}", fixture().path("w2.tsv"));
    let mut other = patch_source(&[]);
    other.args[4] = w2;
    assert_rekeys(&patch_source(&[]), &other);
}

#[test]
fn keyed_param_vec() {
    let v = format!("beta={}", fixture().path("beta_vec.tsv"));
    assert_rekeys(&sir(&[]), &sir(&["--param-vec", &v]));
}

#[test]
fn keyed_fit() {
    let fit = fixture().path("fit.toml");
    assert_rekeys(&sir(&["--draws", "prior", "-n", "2"]), &sir(&["--draws", "prior", "-n", "2", "--fit", &fit]));
}

#[test]
fn keyed_enable() {
    assert_rekeys(&sir(&[]), &sir(&["--enable", "sia"]));
}

#[test]
fn keyed_disable() {
    assert_rekeys(&sir(&["--enable", "sia"]), &sir(&["--disable", "sia"]));
}

#[test]
fn keyed_backend() {
    assert_rekeys(&sir(&[]), &sir(&["--backend", "gillespie"]));
}

#[test]
fn keyed_dt() {
    assert_rekeys(&sir(&[]), &sir(&["--dt", "0.5"]));
}

#[test]
fn keyed_integrator() {
    assert_rekeys(&sir(&["--backend", "ode"]), &sir(&["--backend", "ode", "--integrator", "rk45"]));
}

#[test]
fn keyed_every() {
    assert_rekeys(&sir(&[]), &sir(&["--output-every", "2"]));
}

#[test]
fn keyed_no_flows() {
    assert_rekeys(&sir(&[]), &sir(&["--no-flows"]));
}

#[test]
fn keyed_columns() {
    assert_rekeys(&sir(&[]), &sir(&["--columns", "S,I"]));
}

#[test]
fn keyed_seed() {
    assert_rekeys(&sir(&[]), &sir(&["--seed", "2"]));
}

#[test]
fn keyed_no_licm() {
    assert_rekeys(&sir_source(&[]), &sir_source(&["--no-licm"]));
}

#[test]
fn keyed_emit_every() {
    let mirror = tempfile::tempdir().unwrap();
    let obs = mirror.path().join("obs").to_string_lossy().into_owned();
    let base = sir(&["--obs-dir", &obs]);
    assert_obs_rekeys(&base, &sir(&["--obs-dir", &obs, "--emit-every", "2"]));
}

#[test]
fn keyed_env_eval_unresolved() {
    assert_rekeys(&sir(&[]), &sir(&[]).env("CAMDL_EVAL_UNRESOLVED", "1"));
}

#[test]
fn keyed_env_no_binding_cache() {
    assert_rekeys(&sir(&[]), &sir(&[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let with = sir(&[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1");
    assert_rekeys(&sir(&[]), &with);
    // The record lists it.
    let store = tempfile::tempdir().unwrap();
    let _ = simulate_into(store.path(), &with);
    let record = leaves_records(store.path()).pop().expect("a record");
    assert_eq!(record["provenance"]["experimental"]["CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST"], "1");
}

#[test]
fn keyed_env_no_constant_fold() {
    assert_rekeys(&patch_source(&[]), &patch_source(&[]).env("CAMDL_NO_CONSTANT_FOLD", "1"));
}

#[test]
fn keyed_env_no_licm() {
    assert_rekeys(&sir_source(&[]), &sir_source(&[]).env("CAMDL_NO_LICM", "1"));
}

/// Every sim record under `root`, parsed.
fn leaves_records(root: &Path) -> Vec<serde_json::Value> {
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
    assert_shared_cells_keep_ids(&sir(&["--scenario", "fast"]), &sir(&["--scenario", "fast", "--scenario", "with_sia"]));
}

#[test]
fn expands_seeds() {
    assert_shared_cells_keep_ids(&sir(&["--seeds", "1:2"]), &sir(&["--seeds", "1:3"]));
}

#[test]
fn expands_replicates() {
    assert_shared_cells_keep_ids(&sir(&["--replicates", "2"]), &sir(&["--replicates", "3"]));
}

#[test]
fn expands_draws() {
    let fx = fixture();
    assert_shared_cells_keep_ids(
        &sir(&["--draws", &fx.path("draws2.tsv")]),
        &sir(&["--draws", &fx.path("draws3.tsv")]),
    );
}

#[test]
fn expands_n_draws() {
    assert_shared_cells_keep_ids(&sir(&["--draws", "prior", "-n", "2"]), &sir(&["--draws", "prior", "-n", "3"]));
}

#[test]
fn expands_sweep() {
    assert_shared_cells_keep_ids(&sir(&["--sweep", "beta_a=0.3,0.4"]), &sir(&["--sweep", "beta_a=0.3,0.4,0.5"]));
}

#[test]
fn expands_sweep_file() {
    let fx = fixture();
    assert_shared_cells_keep_ids(
        &sir(&["--sweep-file", &fx.path("sweep2.tsv")]),
        &sir(&["--sweep-file", &fx.path("sweep3.tsv")]),
    );
}

#[test]
fn expands_chains() {
    let post = fixture().path("post");
    let draws = ["--draws", "posterior", "--fit", &post];
    let one = [&draws[..], &["--chains", "0"]].concat();
    let two = [&draws[..], &["--chains", "0,1"]].concat();
    assert_shared_cells_keep_ids(&sir(&one), &sir(&two));
}

#[test]
fn expands_exclude_chains() {
    let post = fixture().path("post");
    let draws = ["--draws", "posterior", "--fit", &post];
    let fewer = [&draws[..], &["--exclude-chains", "1,2"]].concat();
    let more = [&draws[..], &["--exclude-chains", "2"]].concat();
    assert_shared_cells_keep_ids(&sir(&fewer), &sir(&more));
}

// ─── execution ───────────────────────────────────────────────────────────────

#[test]
fn execution_parallel() {
    assert_same(&sir(&["--seeds", "1:4", "--parallel", "1"]), &sir(&["--seeds", "1:4", "--parallel", "4"]));
}

#[test]
fn execution_output_dir() {
    // The same run into two store roots: the stored leaves are the same.
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let at = |d: &Path| {
        let mut inv = sir(&[]);
        inv.args.extend(["--output-dir".to_string(), d.join("elsewhere").to_string_lossy().into_owned()]);
        inv
    };
    assert_same_result(&simulate_into(a.path(), &at(a.path())), &simulate_into(b.path(), &at(b.path())));
}

#[test]
fn execution_force() {
    // A forced recompute into the store that holds the run stores the same
    // leaf; the incumbent is quarantined.
    let store = tempfile::tempdir().unwrap();
    let first = simulate_into(store.path(), &sir(&[]));
    let forced = simulate_into(store.path(), &sir(&["--force"]));
    assert_same_result(&first, &forced);
    assert!(store.path().join("results/.quarantine").is_dir(), "--force displaced the incumbent");
}

#[test]
fn execution_no_ir_cache() {
    assert_same(&sir_source(&[]), &sir_source(&["--no-ir-cache"]));
}

#[test]
fn execution_env_no_ir_cache() {
    assert_same(&sir_source(&[]), &sir_source(&[]).env("CAMDL_NO_IR_CACHE", "1"));
}

#[test]
fn execution_env_ir_cache_dir() {
    let elsewhere = tempfile::tempdir().unwrap();
    let moved = sir_source(&[]).env("CAMDL_IR_CACHE_DIR", &elsewhere.path().to_string_lossy());
    assert_same(&sir_source(&[]), &moved);
}

#[test]
fn execution_env_xdg_cache_home() {
    // With no `CAMDL_IR_CACHE_DIR`, the cache lives under `XDG_CACHE_HOME`.
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| sir_source(&[]).unset("CAMDL_IR_CACHE_DIR").env("XDG_CACHE_HOME", &d.to_string_lossy());
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_home() {
    // With neither `CAMDL_IR_CACHE_DIR` nor `XDG_CACHE_HOME`, under `HOME`.
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let under = |d: &Path| {
        sir_source(&[]).unset("CAMDL_IR_CACHE_DIR").unset("XDG_CACHE_HOME").env("HOME", &d.to_string_lossy())
    };
    assert_same(&under(a.path()), &under(b.path()));
}

#[test]
fn execution_env_camdlc_path() {
    // The same compiler named by `CAMDLC_PATH` (which outranks `CAMDLC`).
    let camdlc = fixture().camdlc.to_string_lossy().into_owned();
    assert_same(&sir_source(&["--no-ir-cache"]), &sir_source(&["--no-ir-cache"]).env("CAMDLC_PATH", &camdlc));
}

#[test]
fn execution_env_camdlc() {
    // The same compiler named by `CAMDLC`, against the one beside it on `PATH`.
    let fx = fixture();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.camdlc, bin.path().join("camdlc")).unwrap();
    let path = format!("{}:{}", bin.path().display(), std::env::var("PATH").unwrap_or_default());
    let on_path = sir_source(&["--no-ir-cache"]).unset("CAMDLC").env("PATH", &path);
    assert_same(&sir_source(&["--no-ir-cache"]), &on_path);
}

// ─── presentation and provenance ────────────────────────────────────────────

/// A presentation or provenance input leaves the stored result alone: the
/// same cells, under the same run ids, holding the same bytes. (That they are
/// not fields of the resolved configuration is checked structurally, by
/// `sim_config::tests::presentation_and_provenance_are_not_configuration`.)
///
/// `--event-log` is the exception: the recorder is passive, so the trajectory
/// and the run id are the same, but the leaf gains an `event_log.tsv` artifact
/// in its manifest, so the same run id holds different stored bytes depending
/// on whether any invocation asked for the log. Its check is the trajectory
/// alone, and the gain is asserted, so a change to that behaviour is seen here.
#[test]
fn presentation_and_provenance_inputs_leave_the_leaf_unaffected() {
    let scratch = tempfile::tempdir().unwrap();
    let at = |name: &str| scratch.path().join(name).to_string_lossy().into_owned();
    let base = simulate(&sir(&[]));
    let cases: Vec<(&str, Vec<String>)> = vec![
        ("--output", vec!["-o".into(), at("traj.tsv")]),
        ("--obs", vec!["--obs".into(), at("obs.tsv")]),
        ("--obs-dir", vec!["--obs-dir".into(), at("obs_dir")]),
        ("--obs-only", vec!["--obs-only".into(), at("obs_only.tsv")]),
        ("--obs-only-dir", vec!["--obs-only-dir".into(), at("obs_only_dir")]),
        ("--label", vec!["--label".into(), "a label".into()]),
        ("--cas", vec!["--cas".into()]),
        ("--reactive-log", vec!["--reactive-log".into(), at("reactive.tsv")]),
        ("--quantities-out", vec!["--quantities-out".into(), at("q")]),
        ("--verbosity", vec!["--verbosity".into(), "debug".into()]),
        ("--no-progress", vec!["--no-progress".into()]),
    ];
    let traj = |s: &Stored| -> Vec<(String, Option<String>)> {
        s.sims().into_iter().map(|(id, l)| (id, l.files.get("traj.tsv").cloned())).collect()
    };
    let files = |s: &Stored| -> Vec<BTreeMap<String, String>> { s.leaves.iter().map(|l| l.files.clone()).collect() };
    for (flag, extra) in &cases {
        let extra: Vec<&str> = extra.iter().map(String::as_str).collect();
        let with = simulate(&sir(&extra));
        assert_eq!(traj(&base), traj(&with), "{flag}: the cells and their trajectories are unchanged");
        assert_eq!(files(&base), files(&with), "{flag}: the stored bytes are unchanged");
    }
    let logged = simulate(&sir(&["--event-log"]));
    assert_eq!(traj(&base), traj(&logged), "--event-log: the cell and its trajectory are unchanged");
    let gained: Vec<&String> = logged.leaves[0].files.keys().filter(|k| !base.leaves[0].files.contains_key(*k)).collect();
    assert_eq!(gained, ["event_log.tsv"], "--event-log adds its log to the leaf, under the same run id");
}
