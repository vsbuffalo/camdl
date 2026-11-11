//! A command that reads a run's results — its posterior draws (`fit predict`,
//! `simulate --draws posterior`), its point estimate (`starts = { from_mle }`)
//! — reads only a completed run, and refuses any other with the state its
//! `run.json` and `.lock` give: failed with its reason, failed with none
//! recorded, still running, interrupted, held from another host.
//!
//! A sampler writes `draws.tsv` before the steps that end its run (its
//! diagnostics, the `--resume` lineage, the leaf's finalize), so a run that
//! fails at one of them, or is killed there, leaves a `draws.tsv` that looks
//! complete beside a record that is not. The fixtures here are completed runs
//! whose `run.json` is then rewritten to the state under test, the shape such
//! a run leaves.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

/// A short PMMH fit of `tests/fixtures/mre`'s two-patch SIR, which completes
/// and writes a `draws.tsv`.
const PMMH: &str = r#"
[model]
camdl = "model/sir_patches.camdl"
[data]
file = "cases.tsv"
[estimate.beta]
bounds = [0.15, 0.6]
start  = 0.3
prior  = { flat = {} }
[estimate.gamma]
bounds = [0.08, 0.3]
start  = 0.15
prior  = { flat = {} }
[fixed]
rho = 0.6
k   = 20
I0  = 5
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 2
particles  = 30
iterations = 60
burn_in    = 20
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

fn run(dir: &Path, args: &[&str]) -> Output {
    camdl_in(dir).args(args).output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
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

/// `camdl fit run <toml>` in `dir`, which must succeed; its stderr.
fn fit_run(dir: &Path, toml: &str) -> String {
    let out = run(dir, &["fit", "run", toml, "--seed", "1", "--progress", "none"]);
    let said = stderr(&out);
    assert!(out.status.success(), "fit run {toml}:\n{said}");
    said
}

/// The fit leaves under `dir`'s store whose method directory starts with
/// `method`.
fn leaves(dir: &Path, method: &str) -> Vec<PathBuf> {
    let subdirs = |d: &Path| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path())
            .filter(|p| p.is_dir() && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
            .collect();
        v.sort();
        v
    };
    subdirs(&dir.join("results/fits")).iter()
        .flat_map(|fit| subdirs(fit))
        .filter(|m| m.file_name().unwrap().to_string_lossy().starts_with(method))
        .flat_map(|m| subdirs(&m))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("seed_"))
        .collect()
}

/// A completed PMMH fit; its project and its one leaf, which holds a
/// `draws.tsv`.
fn completed_pmmh() -> (tempfile::TempDir, PathBuf) {
    let tmp = project(PMMH);
    fit_run(tmp.path(), "fit.toml");
    let found = leaves(tmp.path(), "pmmh");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].join("draws.tsv").is_file(), "the sampler wrote its draws");
    (tmp, found[0].clone())
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The states a leaf that did not complete can be in, each as the rewrite of
/// a completed leaf's `run.json` and `.lock` that leaves it so, with what a
/// refusal must say about it.
#[derive(Debug, Clone, Copy)]
enum NotCompleted {
    Failed,
    FailedNoReason,
    Running,
    Interrupted,
}

impl NotCompleted {
    const ALL: [NotCompleted; 4] =
        [NotCompleted::Failed, NotCompleted::FailedNoReason, NotCompleted::Running, NotCompleted::Interrupted];

    fn apply(self, leaf: &Path) {
        let mut rec = json(&leaf.join("run.json"));
        let fields = rec.as_object_mut().unwrap();
        fields.remove("failure");
        match self {
            NotCompleted::Failed => {
                fields.insert("status".into(), "failed".into());
                fields.insert("failure".into(), serde_json::json!({
                    "message": "error: could not finalize fit method leaf: disk full",
                    "operation": "pmmh",
                }));
            }
            NotCompleted::FailedNoReason => {
                fields.insert("status".into(), "failed".into());
            }
            NotCompleted::Running | NotCompleted::Interrupted => {
                fields.insert("status".into(), "running".into());
            }
        }
        std::fs::write(leaf.join("run.json"), serde_json::to_vec_pretty(&rec).unwrap()).unwrap();
        match self {
            // This test process is alive on this host: the run is in flight.
            NotCompleted::Running => {
                std::fs::write(leaf.join(".lock"), std::process::id().to_string()).unwrap()
            }
            _ => {
                let _ = std::fs::remove_file(leaf.join(".lock"));
            }
        }
    }

    fn says(self) -> String {
        match self {
            NotCompleted::Failed => "it failed: could not finalize fit method leaf: disk full".into(),
            NotCompleted::FailedNoReason => "it failed, and recorded no reason".into(),
            NotCompleted::Running => format!("it is still running (pid {})", std::process::id()),
            NotCompleted::Interrupted => {
                "it was interrupted: no live process holds it, and it recorded no outcome".into()
            }
        }
    }
}

/// The refusal a result consumer gives `leaf` in `state`.
fn assert_refused(out: &Output, leaf: &Path, state: NotCompleted, what: &str) {
    let said = stderr(out);
    assert!(!out.status.success(), "{what} reads a {state:?} leaf:\n{said}");
    assert!(said.contains(&format!("{} is not a completed run: {}", leaf.display(), state.says())),
        "{what}: the refusal names the leaf and its state ({state:?}):\n{said}");
}

/// `fit predict <leaf>` refuses a sampler leaf that did not complete, in each
/// state, and writes no predictive for it.
#[test]
fn fit_predict_refuses_a_leaf_that_did_not_complete() {
    let (tmp, leaf) = completed_pmmh();
    let dir = tmp.path();
    let completed = json(&leaf.join("run.json"));
    for state in NotCompleted::ALL {
        state.apply(&leaf);
        let out = run(dir, &["fit", "predict", &leaf.to_string_lossy(), "--n-draws", "4"]);
        assert_refused(&out, &leaf, state, "fit predict");
        let segment = leaf.parent().unwrap().parent().unwrap();
        assert!(!segment.join("predict").exists(),
            "nothing is written for a {state:?} leaf: the run is refused before predict starts");
    }
    // A run-id prefix names the run at any state; it is refused the same way.
    NotCompleted::Failed.apply(&leaf);
    let prefix = completed["run_id"].as_str().unwrap()[..12].to_string();
    let out = run(dir, &["fit", "predict", &prefix, "--n-draws", "4"]);
    let said = stderr(&out);
    assert!(!out.status.success() && said.contains("is not a completed run: it failed"),
        "a run-id prefix naming a failed run:\n{said}");
    // Restored, the completed leaf predicts.
    std::fs::write(leaf.join("run.json"), serde_json::to_vec_pretty(&completed).unwrap()).unwrap();
    let _ = std::fs::remove_file(leaf.join(".lock"));
    let out = run(dir, &["fit", "predict", &leaf.to_string_lossy(), "--n-draws", "4"]);
    assert!(out.status.success(), "a completed leaf predicts:\n{}", stderr(&out));
}

/// `simulate --draws posterior --fit <leaf>` refuses the same leaves: the
/// leaf directory holds a `draws.tsv`, and the draws authority reads it only
/// from a completed run.
#[test]
fn simulate_from_the_posterior_of_a_leaf_that_did_not_complete_is_refused() {
    let (tmp, leaf) = completed_pmmh();
    let dir = tmp.path();
    for state in [NotCompleted::Failed, NotCompleted::Running] {
        state.apply(&leaf);
        let out = run(dir, &["simulate", "model/sir_patches.camdl", "--draws", "posterior",
            "--fit", &leaf.to_string_lossy(), "-n", "2", "--seed", "1", "--output-dir", "sims"]);
        assert_refused(&out, &leaf, state, "simulate --draws posterior");
    }
}

/// A leaf handle whose `run.json` is not a run record is an error naming the
/// leaf and why, from a reader of results (`fit predict`) and from a reader
/// that shows any run (`fit summary`) alike, not a problem folder that
/// "holds no completed fit run".
#[test]
fn a_handle_whose_run_json_is_corrupt_is_an_error_naming_it() {
    let (tmp, leaf) = completed_pmmh();
    let dir = tmp.path();
    std::fs::write(leaf.join("run.json"), "{ \"format_version\": 1, \"kind\": ").unwrap();
    for args in [vec!["fit", "predict"], vec!["fit", "summary"]] {
        let mut args = args;
        let handle = leaf.to_string_lossy().into_owned();
        args.push(&handle);
        let out = run(dir, &args);
        let said = stderr(&out);
        assert!(!out.status.success(), "{args:?}:\n{said}");
        assert!(said.contains(&format!("{}: its run.json is not a run record", leaf.display())),
            "{args:?} names the record and why:\n{said}");
        assert!(!said.contains("holds no completed fit run"), "{args:?}:\n{said}");
    }
}
