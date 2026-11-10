//! The lifecycle record of a streamed fit leaf: a run camdl stops is
//! committed `failed` with its reason; an interrupted one stays `running`
//! and reads as interrupted; a retry keeps the prior attempt's record and
//! completes clean.
//!
//! The failing fixture is a PMMH fit whose every start is refused at chain
//! init (see [`FIT`]): the filter degenerates at every start, each chain is
//! refused, and the method fails through `abandon_method`. The completing
//! fixture is a short IF2 fit of the same model ([`FIT_OK`]).

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

/// A short IF2 fit of the same model and data, which completes.
const FIT_OK: &str = r#"
[model]
camdl = "model/sir_patches.camdl"
[data]
file = "cases.tsv"
[estimate.beta]
bounds = [0.15, 0.6]
start  = 0.3
[estimate.gamma]
bounds = [0.08, 0.3]
start  = 0.15
[fixed]
rho = 0.6
k   = 20
I0  = 5
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 100
iterations = 2
cooling    = 0.7
starts     = "single"
"#;

/// A camdl command in `dir`, against the compiler under test.
fn camdl_in(dir: &Path) -> Command {
    let mut c = Command::new(camdl());
    c.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
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

/// The one fit leaf under `dir`.
fn only_leaf(dir: &Path) -> PathBuf {
    let subdirs = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d).unwrap().flatten().map(|e| e.path())
            .filter(|p| p.is_dir() && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
            .collect()
    };
    let leaves: Vec<PathBuf> = subdirs(&dir.join("results/fits")).iter()
        .flat_map(|fit| subdirs(fit))
        .flat_map(|m| subdirs(&m))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("seed_"))
        .collect();
    assert_eq!(leaves.len(), 1, "one claimed leaf: {leaves:?}");
    leaves[0].clone()
}

/// Run the failing fit; return its stderr, the one leaf it claimed, and the
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

fn write_json(path: &Path, v: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}

fn events(leaf: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(leaf.join("events.jsonl")).unwrap()
        .lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// `fit summary <leaf>`'s stderr: with no completed leaf it exits 1, naming
/// each leaf in its state as `run.json` and `.lock` give it.
fn summary_stderr(dir: &Path, leaf: &Path) -> String {
    let out = camdl_in(dir).args(["fit", "summary", &leaf.to_string_lossy()]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The leaf's `run.json` reaches `failed`, as its event log and
/// `progress.json` do, with a `failure` record holding the error that
/// stopped the run and what was running, and the claim's lock is released.
/// The failing exit runs no destructor, so the claim's own guard never saw
/// it, and the leaf stayed `running` beside a log that said it had failed.
/// Its `inputs` were written with the claim, so the failed leaf is a complete
/// record; only the values a finished run computes are absent.
#[test]
fn a_leaf_that_fails_at_chain_init_records_failed_and_why_in_its_run_json() {
    let (_stderr, leaf, _tmp) = failed_run();
    let rec = json(&leaf.join("run.json"));
    assert_eq!(rec["status"], "failed", "run.json: {rec}");
    assert!(!leaf.join(".lock").exists(), "the claim's lock is released");
    let last = events(&leaf).pop().unwrap();
    assert_eq!((&last["event"], &last["outcome"]), (&"method_report".into(), &"failed".into()));
    assert_eq!(rec["failure"], serde_json::json!({"message": last["error"], "operation": "pmmh"}),
        "run.json records why, the error the run reported, and what was running: {rec}");
    let progress = json(&leaf.join("progress.json"));
    assert!(progress["state"].get("failed").is_some(), "progress.json: {progress}");
    let inputs = &rec["inputs"];
    assert_eq!((&inputs["method"], &inputs["backend"], &inputs["n_chains"], &inputs["seed"]),
        (&"pmmh".into(), &"chain_binomial".into(), &2.into(), &1.into()), "{inputs}");
    assert!(inputs.get("best_loglik").is_none() && inputs.get("wall_time_seconds").is_none(),
        "a run that did not finish records nothing it would have computed: {inputs}");
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
    let message = rec["failure"]["message"].as_str().unwrap();
    assert!(message.contains("bailed via the PF degeneracy watchdog"), "{message}");
    assert_eq!(events(&leaf).pop().unwrap()["error"].as_str(), Some(message));
}

/// A run killed before it recorded an outcome leaves a `running` record. It
/// is interrupted, not failed: `fit summary` reads that from the absent (or
/// dead-pid) `.lock` (`runid::leaf_state`), and a retry preserves the record as it stood rather
/// than rewriting it `failed`.
#[test]
fn an_interrupted_run_is_not_reported_as_failed() {
    let (_stderr, leaf, tmp) = failed_run();
    // What a killed run leaves: its record still `running`, no outcome, and
    // its `.lock` gone (or naming a process that has exited).
    let mut rec = json(&leaf.join("run.json"));
    rec["status"] = "running".into();
    rec.as_object_mut().unwrap().remove("failure");
    write_json(&leaf.join("run.json"), &rec);
    let mut child = Command::new("true").spawn().unwrap();
    let dead_pid = child.id();
    child.wait().unwrap();
    for lock in [None, Some(dead_pid)] {
        match lock {
            Some(pid) => std::fs::write(leaf.join(".lock"), pid.to_string()).unwrap(),
            None => { let _ = std::fs::remove_file(leaf.join(".lock")); }
        }
        let said = summary_stderr(tmp.path(), &leaf);
        assert!(said.contains("pmmh interrupted: no live process holds it"), "{said}");
        assert!(!said.contains("failed"), "an interrupted run is not called failed: {said}");
        assert_eq!(json(&leaf.join("run.json"))["status"], "running", "reading rewrites nothing");
    }
    // The retry (the same fit again) keeps the interrupted record unchanged.
    let (code, _) = fit_run(tmp.path());
    assert_eq!(code, Some(1));
    let kept = json(&leaf.join(".attempts/1/attempt.json"));
    assert_eq!(kept["status"], "running", "the retry preserved it as it stood: {kept}");
    assert!(kept.get("failure").is_none());
}

/// A `failed` record without a `failure`, as a camdl before the field wrote,
/// reads: `fit summary` lists it failed, its reason unknown.
#[test]
fn an_old_failed_record_without_a_reason_reads() {
    let (_stderr, leaf, tmp) = failed_run();
    let mut rec = json(&leaf.join("run.json"));
    rec.as_object_mut().unwrap().remove("failure");
    write_json(&leaf.join("run.json"), &rec);
    let said = summary_stderr(tmp.path(), &leaf);
    assert!(said.contains("pmmh failed (no reason recorded): "), "{said}");
    assert!(!said.contains("error: error running"), "no reason is invented: {said}");
}

/// Mark the completed leaf as a failed attempt, as a run that failed at this
/// identity would have left it: `failed`, with a reason, and its own records.
fn mark_failed_attempt(leaf: &Path, n: usize) {
    let mut rec = json(&leaf.join("run.json"));
    rec["status"] = "failed".into();
    rec["failure"] = serde_json::json!({"message": format!("attempt {n} failed"), "operation": "if2"});
    write_json(&leaf.join("run.json"), &rec);
    std::fs::write(leaf.join("diagnostics.json"), format!("[\"attempt {n}\"]")).unwrap();
}

/// A retry after a failure keeps the failed attempt's record under
/// `.attempts/1/` and completes as a clean leaf: a cache hit on the next run,
/// listed once by a walk of the store (the preserved record is
/// `attempt.json`, not a second `run.json` of the same run id).
#[test]
fn a_retry_after_a_failure_preserves_its_evidence_and_completes_clean() {
    let tmp = project(FIT_OK);
    let dir = tmp.path();
    let (code, stderr) = fit_run(dir);
    assert_eq!(code, Some(0), "{stderr}");
    let leaf = only_leaf(dir);
    let run_id = json(&leaf.join("run.json"))["run_id"].clone();
    mark_failed_attempt(&leaf, 1);

    let (code, stderr) = fit_run(dir);
    assert_eq!(code, Some(0), "the retry completes:\n{stderr}");
    assert!(!stderr.contains("cache hit"), "a failed leaf is not reused:\n{stderr}");
    let rec = json(&leaf.join("run.json"));
    assert_eq!(rec["status"], "completed");
    assert!(rec.get("failure").is_none(), "{rec}");
    assert!(rec["artifacts"].as_object().unwrap().keys().all(|k| !k.starts_with(".attempts")),
        "the manifest excludes the preserved attempt: {}", rec["artifacts"]);
    let kept = json(&leaf.join(".attempts/1/attempt.json"));
    assert_eq!(kept["failure"]["message"], "attempt 1 failed");
    assert_eq!(std::fs::read_to_string(leaf.join(".attempts/1/diagnostics.json")).unwrap(),
        "[\"attempt 1\"]");
    assert!(leaf.join(".attempts/1/events.jsonl").is_file(), "its event log is kept");
    assert!(!leaf.join(".attempts/1/chain_0").exists(), "its bulk outputs are not");

    let (code, stderr) = fit_run(dir);
    assert_eq!(code, Some(0));
    assert!(stderr.contains("cache hit"), "the retried leaf is a clean hit:\n{stderr}");

    let out = camdl_in(dir).args(["list", "--kind", "fit", "--format", "json"]).output().unwrap();
    let rows: Vec<serde_json::Value> = String::from_utf8_lossy(&out.stdout).lines()
        .map(|l| serde_json::from_str(l).unwrap()).collect();
    let listed: Vec<&serde_json::Value> = rows.iter()
        .flat_map(|r| r["methods"].as_array().into_iter().flatten())
        .filter(|m| m["run_id"] == run_id)
        .collect();
    assert_eq!(listed.len(), 1, "one leaf for the run id: {rows:?}");
}

/// Retries keep the newest three attempts (`MAX_PRESERVED_ATTEMPTS`): after
/// a fourth failed attempt is retried, the first is gone.
#[test]
fn retries_keep_at_most_three_attempts() {
    let tmp = project(FIT_OK);
    let dir = tmp.path();
    assert_eq!(fit_run(dir).0, Some(0));
    let leaf = only_leaf(dir);
    for n in 1..=4 {
        mark_failed_attempt(&leaf, n);
        let (code, stderr) = fit_run(dir);
        assert_eq!(code, Some(0), "{stderr}");
    }
    let mut kept: Vec<String> = std::fs::read_dir(leaf.join(".attempts")).unwrap().flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    kept.sort();
    assert_eq!(kept, ["2", "3", "4"], "the oldest attempt is dropped");
    for n in 2..=4 {
        let rec = json(&leaf.join(format!(".attempts/{n}/attempt.json")));
        assert_eq!(rec["failure"]["message"], format!("attempt {n} failed"));
    }
}

/// A leaf held from another host is not reclaimed and not called
/// interrupted: `fit run` refuses it naming the host, the pid and `--force`,
/// `fit summary` says its liveness cannot be checked, and `fit run --force`
/// moves it, records included, into the store's quarantine and runs again.
#[test]
fn a_leaf_held_from_another_host_is_refused_until_forced() {
    let (_stderr, leaf, tmp) = failed_run();
    let dir = tmp.path();
    let mut rec = json(&leaf.join("run.json"));
    rec["status"] = "running".into();
    rec.as_object_mut().unwrap().remove("failure");
    write_json(&leaf.join("run.json"), &rec);
    std::fs::write(leaf.join(".lock"), "4242\nsome-other-host\n").unwrap();

    let said = summary_stderr(dir, &leaf);
    assert!(said.contains("running on host some-other-host (pid 4242)"), "{said}");
    assert!(!said.contains("interrupted"), "{said}");

    let (code, stderr) = fit_run(dir);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("pid 4242 on host some-other-host") && stderr.contains("--force"),
        "the refusal names the owner and the way to clear it:\n{stderr}");
    assert_eq!(json(&leaf.join("run.json"))["status"], "running", "the leaf is untouched");
    assert!(leaf.join("diagnostics.json").is_file());

    let out = camdl_in(dir)
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none", "--force"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("all 2 chains failed init-eval"), "--force runs again:\n{stderr}");
    assert_eq!(json(&leaf.join("run.json"))["status"], "failed");
    let quarantined: Vec<PathBuf> = std::fs::read_dir(dir.join("results/.quarantine")).unwrap()
        .flatten().map(|e| e.path()).collect();
    assert_eq!(quarantined.len(), 1, "{quarantined:?}");
    assert!(quarantined[0].join("diagnostics.json").is_file(), "its records are kept");
}
