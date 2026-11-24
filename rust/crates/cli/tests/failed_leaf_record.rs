//! The record a stored fit run keeps when the method fails: the run is
//! `failed` in its `run.json`, with the operation that was running and the
//! error that stopped it, its event log ends on the failed `method_report`,
//! and the next `fit run` computes it again rather than reusing it.
//!
//! The failing fixture is a PMMH fit whose every start is refused at chain
//! init (see [`FIT`]): the filter degenerates at every start, each chain is
//! refused, and the method fails through `abandon_method`.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    camdl
}

fn camdlc() -> PathBuf {
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    camdlc
}

/// The fit, over `tests/fixtures/mre`'s two-patch SIR and its weekly cases.
/// The bounds hold transmission far below recovery, so the epidemic dies out
/// within a week and a later week's positive count has a projected mean of
/// zero: no start can be scored.
const FIT: &str = r#"
[model]
camdl = "model/sir_patches.camdl"
[data]
file = "cases.tsv"
[estimate.beta]
bounds = [0.0011, 0.0012]
start  = 0.00115
prior  = { flat = {} }
[estimate.gamma]
bounds = [0.9, 1.0]
start  = 0.95
prior  = { flat = {} }
[fixed]
rho = 0.6
k   = 20
I0  = 1
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 2
particles  = 50
iterations = 20
burn_in    = 10
starts     = "from_prior"
"#;

/// A camdl command in `dir`, against the compiler under test.
fn camdl_in(dir: &Path) -> Command {
    let mut c = Command::new(camdl());
    c.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
        .env("CAMDL_STORE", dir.join("camdl-store"))
        .current_dir(dir);
    c
}

/// A fresh directory holding `tests/fixtures/mre`'s model and data, and
/// `fit` as `fit.toml`.
fn project(fit: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fixture = manifest().join("../../../tests/fixtures/mre");
    std::fs::create_dir(dir.join("model")).unwrap();
    for f in ["model/sir_patches.camdl", "model/pop.tsv", "cases.tsv"] {
        std::fs::copy(fixture.join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fit.toml"), fit).unwrap();
    tmp
}

/// `camdl fit run fit.toml` in `dir`: its exit status and stderr.
fn fit_run(dir: &Path) -> (Option<i32>, String) {
    let out = camdl_in(dir)
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"])
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// The one fit run in `dir`'s store.
fn only_leaf(dir: &Path) -> PathBuf {
    fit_store_support::the_run(&dir.join("camdl-store"))
}

/// Run the failing fit; return its stderr, the one run it claimed, and the
/// directory holding both.
fn failed_run() -> (String, PathBuf, tempfile::TempDir) {
    let tmp = project(FIT);
    let (code, stderr) = fit_run(tmp.path());
    assert_eq!(code, Some(1), "the run fails with status 1:\n{stderr}");
    assert!(stderr.contains("all 2 chains failed init-eval"),
        "the run fails at chain init, every chain refused:\n{stderr}");
    let leaf = only_leaf(tmp.path());
    (stderr, leaf, tmp)
}

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The events of a run's stored event log, each record's `event`.
fn events(leaf: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(leaf.join("events.jsonl")).unwrap()
        .lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"].clone()).collect()
}

/// The run's `run.json` reaches `failed`, as its event log and
/// `progress.json` do, with a `failure` record holding the error that
/// stopped the run and what was running. Its identity was written with the
/// claim, so the failed run records what it was (method, backend, chains,
/// seed); only the values a finished run computes are absent.
#[test]
fn a_leaf_that_fails_at_chain_init_records_failed_and_why_in_its_run_json() {
    let (_stderr, leaf, _tmp) = failed_run();
    let rec = json(&leaf.join("run.json"));
    assert_eq!(rec["status"], "failed", "run.json: {rec}");
    let last = events(&leaf).pop().unwrap();
    assert_eq!((&last["event"], &last["outcome"]), (&"method_report".into(), &"failed".into()));
    assert_eq!(rec["failure"], serde_json::json!({"reason": last["error"], "operation": "pmmh"}),
        "run.json records why, the error the run reported, and what was running: {rec}");
    let progress = json(&leaf.join("progress.json"));
    assert!(progress["snapshot"]["state"].get("failed").is_some(), "progress.json: {progress}");
    let ident = json(&leaf.join("identity.json"));
    let chain = &ident["identity"]["chain"];
    let pmmh = &chain["method"]["Pmmh"];
    assert_eq!((&pmmh["backend"], &pmmh["chains"], &chain["seed"]["seed"]),
        (&"ChainBinomial".into(), &2.into(), &1.into()), "{ident}");
    assert!(!leaf.join("run_inputs.json").exists(),
        "a run that did not finish records nothing it would have computed (best log-likelihood, wall time)");
}

/// The IF2 runner's exit when every chain's filter degenerated (the fixture
/// of `gh1016_runner_exits_record_findings.rs`) records the same.
#[test]
fn an_if2_runner_exit_records_failed_and_why_in_its_run_json() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 3.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#).unwrap();
    std::fs::write(dir.join("cases.tsv"),
        "time\tcases\n1\t12\n2\t15\n3\t19\n4\t24\n5\t30\n6\t36\n7\t42\n8\t48\n9\t52\n10\t55\n").unwrap();
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"m.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = { bounds = [0.01, 3.0], start = 0.3 }\n\
         gamma = { bounds = [0.01, 1.0], start = 0.1 }\n[fixed]\nN0 = 1000\n[method]\n\
         algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 2\n\
         iterations = 5\ncooling = 0.7\nstarts = \"lhs\"\n").unwrap();
    let (code, stderr) = fit_run(dir);
    assert_eq!(code, Some(1), "{stderr}");
    let leaf = only_leaf(dir);
    let rec = json(&leaf.join("run.json"));
    assert_eq!(rec["status"], "failed", "{rec}");
    assert_eq!(rec["failure"]["operation"], "if2");
    let message = rec["failure"]["reason"].as_str().unwrap();
    assert!(message.contains("bailed via the PF degeneracy watchdog"), "{message}");
    assert_eq!(events(&leaf).pop().unwrap()["error"].as_str(), Some(message));
}

/// `fit summary` of a failed run exits 1 and says the run failed, with the
/// reason its `run.json` records.
#[test]
fn fit_summary_of_a_failed_run_names_it_failed_with_its_reason() {
    let (_stderr, leaf, tmp) = failed_run();
    let out = camdl_in(tmp.path()).args(["fit", "summary", &leaf.to_string_lossy()]).output().unwrap();
    let said = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{said}");
    assert!(said.contains("failed") && said.contains("all 2 chains failed init-eval"), "{said}");
}

/// A failed run is not reused: the next `fit run` of the same file computes
/// it again, and the store still holds the one run, failed again.
#[test]
fn a_failed_run_is_computed_again_by_the_next_fit_run() {
    let (_stderr, leaf, tmp) = failed_run();
    let (code, stderr) = fit_run(tmp.path());
    assert_eq!(code, Some(1), "{stderr}");
    assert!(!stderr.contains("cache hit"), "a failed run is not reused:\n{stderr}");
    assert!(stderr.contains("all 2 chains failed init-eval"), "the method ran again:\n{stderr}");
    assert_eq!(only_leaf(tmp.path()), leaf, "the same run, computed again");
    assert_eq!(json(&leaf.join("run.json"))["status"], "failed");
}
