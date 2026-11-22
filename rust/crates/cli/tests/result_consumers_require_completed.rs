//! A command that reads a run's results — its posterior draws (`fit predict`,
//! `simulate --draws posterior`), its point estimate (`starts = { from_mle }`)
//! — reads only a completed run, and refuses any other with the state its
//! `run.json` gives: failed with its reason, failed with none recorded,
//! still running (its owner process alive on this host), interrupted (its
//! owner gone).
//!
//! A sampler writes `draws.tsv` before the steps that end its run (its
//! diagnostics, the `--resume` lineage, the leaf's finalize), so a run that
//! fails at one of them, or is killed there, leaves a `draws.tsv` that looks
//! complete beside a record that is not. The fixtures here are completed runs
//! in a run store whose `run.json` is then rewritten to the state under test,
//! the shape such a run leaves.

mod fit_store_support;

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

/// A short IF2 fit of the same model and data, which completes: the source a
/// `from_mle` start reads.
const IF2: &str = r#"
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

/// [`IF2`] as a fit that starts every chain from `source`'s point estimate.
fn from_mle_of(source: &Path) -> String {
    IF2.replace("starts     = \"single\"", &format!("starts     = {{ from_mle = \"{}\" }}", source.display()))
}

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

/// The fit runs in `dir`'s run store whose method is `method`.
fn leaves(dir: &Path, method: &str) -> Vec<PathBuf> {
    fit_store_support::runs_of_method(&dir.join("camdl-store"), method)
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

/// The states a run that did not complete can be in, each as the rewrite of
/// a completed run's `run.json` that leaves it so (its `status`, `failure`
/// and the `owner` process that holds it), with what a refusal must say
/// about it.
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
                    "operation": "pmmh",
                    "reason": "error: could not finalize fit method leaf: disk full",
                }));
            }
            NotCompleted::FailedNoReason => {
                fields.insert("status".into(), "failed".into());
            }
            NotCompleted::Running | NotCompleted::Interrupted => {
                fields.insert("status".into(), "running".into());
            }
        }
        // The owner keeps the host the fit ran on, this one. A store reads an
        // owner as alive when a process with its pid exists and started at
        // its start time: this test process, so a run it owns is in flight.
        // No process has the largest pid: a run it owns was interrupted.
        match self {
            NotCompleted::Running => {
                fields["owner"]["pid"] = std::process::id().into();
                fields["owner"]["started"] = this_process_started().0.into();
            }
            _ => fields["owner"]["pid"] = (i32::MAX as u32).into(),
        }
        std::fs::write(leaf.join("run.json"), serde_json::to_vec_pretty(&rec).unwrap()).unwrap();
    }

    fn says(self) -> &'static str {
        match self {
            NotCompleted::Failed => "failed while pmmh: error: could not finalize fit method leaf: disk full",
            NotCompleted::FailedNoReason => "failed",
            NotCompleted::Running => "is still running",
            NotCompleted::Interrupted => "was interrupted: its process ended before it completed",
        }
    }
}

/// This process's start time as the OS reports it, to the millisecond, in the
/// form a run's `owner.started` records it.
#[cfg(target_os = "macos")]
fn this_process_started() -> store::Timestamp {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: `info` is a valid, writable proc_bsdinfo of `size` bytes.
    let n = unsafe {
        libc::proc_pidinfo(std::process::id() as libc::c_int, libc::PROC_PIDTBSDINFO, 0,
            (&mut info as *mut libc::proc_bsdinfo).cast(), size)
    };
    assert_eq!(n, size, "proc_pidinfo of this process");
    let ms = info.pbi_start_tvsec * 1000 + info.pbi_start_tvusec / 1000;
    store::Timestamp::from_system(std::time::UNIX_EPOCH + std::time::Duration::from_millis(ms))
}

/// This process's start time as the OS reports it: field 22 of
/// `/proc/self/stat`, in clock ticks since boot, added to the boot time.
#[cfg(target_os = "linux")]
fn this_process_started() -> store::Timestamp {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let after = &stat[stat.rfind(')').unwrap() + 1..];
    let ticks: u64 = after.split_whitespace().nth(19).unwrap().parse().unwrap();
    let btime: u64 = std::fs::read_to_string("/proc/stat").unwrap().lines()
        .find_map(|l| l.strip_prefix("btime ")).unwrap().trim().parse().unwrap();
    // SAFETY: sysconf has no preconditions.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
    store::Timestamp::from_system(std::time::UNIX_EPOCH + std::time::Duration::from_millis(btime * 1000 + ticks * 1000 / hz))
}

/// The refusal a result consumer gives the handle `handle` naming `leaf` in
/// `state`: one line naming the handle, then the run's folder and its state.
fn assert_refused_as(out: &Output, handle: &str, leaf: &Path, state: NotCompleted, what: &str) {
    let said = stderr(out);
    assert!(!out.status.success(), "{what} reads a {state:?} leaf:\n{said}");
    let opening = format!("{handle} names a fit run that did not complete and is not read: ");
    let line = said.lines().find(|l| l.contains(&opening))
        .unwrap_or_else(|| panic!("{what}: the refusal names the handle ({state:?}):\n{said}"));
    let folder = leaf.file_name().unwrap().to_string_lossy();
    assert!(line.ends_with(&format!("{folder} {}", state.says())),
        "{what}: the refusal names the run's folder and its state ({state:?}):\n{said}");
}

/// [`assert_refused_as`] for a handle that is the leaf's own path.
fn assert_refused(out: &Output, leaf: &Path, state: NotCompleted, what: &str) {
    assert_refused_as(out, &leaf.display().to_string(), leaf, state, what);
}

/// The refusal of a handle naming `leaf`, whose `run.json` is not a run
/// record: the run is damaged, and the refusal names the record.
fn assert_damaged_record(said: &str, leaf: &Path, what: &str) {
    let line = said.lines()
        .find(|l| l.contains(&format!("{} names a fit run that did not complete and is not read: ", leaf.display())))
        .unwrap_or_else(|| panic!("{what}: the refusal names the handle:\n{said}"));
    assert!(line.contains("is damaged: cannot read") && line.contains("run.json"),
        "{what} names the record and why:\n{said}");
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
        assert!(!leaf.join("predict").exists(),
            "nothing is written for a {state:?} leaf: the run is refused before predict starts");
    }
    // A run-id prefix names the run at any state; it is refused the same way.
    NotCompleted::Failed.apply(&leaf);
    let prefix = fit_store_support::run_id(&leaf)[..12].to_string();
    let out = run(dir, &["fit", "predict", &prefix, "--n-draws", "4"]);
    assert_refused_as(&out, &prefix, &leaf, NotCompleted::Failed, "fit predict <run-id prefix>");
    // Restored, the completed leaf predicts.
    std::fs::write(leaf.join("run.json"), serde_json::to_vec_pretty(&completed).unwrap()).unwrap();
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
            "--fit", &leaf.to_string_lossy(), "-n", "2", "--seed", "1", "--store", "camdl-store"]);
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
    std::fs::write(leaf.join("run.json"), "{ \"format\": 1, \"status\": ").unwrap();
    for args in [vec!["fit", "predict"], vec!["fit", "summary"]] {
        let mut args = args;
        let handle = leaf.to_string_lossy().into_owned();
        args.push(&handle);
        let out = run(dir, &args);
        let said = stderr(&out);
        assert!(!out.status.success(), "{args:?}:\n{said}");
        assert_damaged_record(&said, &leaf, &format!("{args:?}"));
        assert!(!said.contains("holds no completed fit run"), "{args:?}:\n{said}");
    }
}

/// A `from_mle` start reads its source's point estimate only from a
/// completed run: a source leaf that failed, or is running, is refused with
/// its state before the consuming fit claims a leaf. Lifting the convergence
/// gate (`--allow-nonconverged-source`) does not lift this.
#[test]
fn a_from_mle_start_from_a_leaf_that_did_not_complete_is_refused() {
    let tmp = project(IF2);
    let dir = tmp.path();
    fit_run(dir, "fit.toml");
    let source = leaves(dir, "if2");
    assert_eq!(source.len(), 1, "{source:?}");
    let source = source[0].clone();
    assert!(source.join("fit_state.toml").is_file(), "the source wrote its point estimate");
    std::fs::write(dir.join("next.toml"), from_mle_of(&source)).unwrap();
    for state in [NotCompleted::Failed, NotCompleted::Running, NotCompleted::Interrupted] {
        state.apply(&source);
        // The convergence gate is lifted, so the refusal is the state's alone
        // (this short IF2 source is not converged).
        let out = run(dir, &["fit", "run", "next.toml", "--seed", "2", "--progress", "none",
            "--allow-nonconverged-source"]);
        assert_refused(&out, &source, state, "starts = from_mle");
        assert!(stderr(&out).contains("from_mle"), "the refusal says which start it is:\n{}", stderr(&out));
        assert_eq!(fit_store_support::runs(&dir.join("camdl-store")), vec![source.clone()],
            "no run is claimed for the consuming fit ({state:?})");
    }
}

/// A source whose `run.json` cannot be read keys nothing: a `--resume` base
/// and a `from_posterior` source with a damaged record are each refused,
/// naming the record and why, before the consuming fit claims a leaf. A
/// readable record keys the consumer by the source's `run_id`; keying it on
/// a file digest instead would give the consumer another identity.
#[test]
fn a_source_with_a_corrupt_run_json_is_an_error_not_a_digest_key() {
    let (tmp, leaf) = completed_pmmh();
    let dir = tmp.path();
    std::fs::write(dir.join("resumed.toml"), PMMH.replace("iterations = 60", "iterations = 90")).unwrap();
    std::fs::write(dir.join("post.toml"), IF2.replace("starts     = \"single\"",
        &format!("starts     = {{ from_posterior = \"{}\" }}", leaf.display()))).unwrap();
    std::fs::write(leaf.join("run.json"), "{ \"format\": 1, \"status\": ").unwrap();
    let leaf_s = leaf.to_string_lossy().into_owned();
    for (what, args) in [
        ("--resume", vec!["fit", "run", "resumed.toml", "--seed", "1", "--progress", "none", "--resume", &leaf_s]),
        ("from_posterior", vec!["fit", "run", "post.toml", "--seed", "2", "--progress", "none",
            "--allow-nonconverged-source"]),
    ] {
        let out = run(dir, &args);
        let said = stderr(&out);
        assert!(!out.status.success(), "{what}:\n{said}");
        assert_damaged_record(&said, &leaf, what);
        let consumers: Vec<PathBuf> =
            fit_store_support::runs(&dir.join("camdl-store")).into_iter().filter(|r| *r != leaf).collect();
        assert!(consumers.is_empty(), "{what}: no consuming run is claimed: {consumers:?}");
    }
}
