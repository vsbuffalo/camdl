//! gh#1084, gh#1138: `camdl pfilter --fit <fit.toml> --save-final-state FILE`
//! under the fit's `holdout_after`.
//!
//! gh#1084: the filter ran through every bound row, held-out ones included,
//! and saved the swarm at the last of them, a forecast origin that already
//! conditions on the rows a forecast from it is scored against. That fix
//! refused the run. gh#1138 (maintainer decision D67 a) replaces the refusal:
//! the run binds only the training rows, cut by the rule the fit's partition
//! uses, saves the state at their end and says so. The saved state is the one
//! a run bound to a training-only file saves.
//!
//! Without `--save-final-state` the filter still scores the full series (its
//! log-likelihood, trace and prequential outputs are unchanged, and `compare`
//! relies on them).
//!
//! End-to-end via the release `camdl` binary and the built `camdlc.exe`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let bin = Path::new(manifest).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

fn camdlc() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let p = Path::new(manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

/// A SIR model with one daily incidence stream on days 1 to 10.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

const FULL: &str = "time\tcases\n1\t3\n2\t4\n3\t5\n4\t6\n5\t8\n6\t9\n7\t11\n8\t13\n9\t15\n10\t17\n";
const TRAINING: &str = "time\tcases\n1\t3\n2\t4\n3\t5\n4\t6\n5\t8\n6\t9\n";

/// Write the compiled model, both data files and a fit config whose `[data]`
/// carries `holdout_line`.
fn fixture(dir: &Path, holdout_line: &str) {
    std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
    let out = Command::new(camdlc()).env_remove("CLICOLOR_FORCE")
        .arg(dir.join("sir.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), FULL).unwrap();
    std::fs::write(dir.join("training.tsv"), TRAINING).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(r#"
output_dir = "results"

[model]
camdl = "sir.ir.json"

[data]
{holdout_line}
[data.observations]
cases = "cases.tsv"

[estimate.beta]
bounds = [0.05, 5.0]
start  = 0.5

[fixed]
gamma = 0.2

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5
"#)).unwrap();
}

fn pfilter(dir: &Path, extra: &[&str]) -> Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .arg("pfilter").arg("sir.ir.json")
        .args(["--fit", "fit.toml", "--param", "beta=0.5", "--param", "gamma=0.2"])
        .args(["--particles", "50", "--seed", "3"])
        .args(extra)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The time in a final-state file's header (`# camdl-final-state v1\tt=…`).
fn origin_of(path: &Path) -> f64 {
    let text = std::fs::read_to_string(path).unwrap();
    let header = text.lines().next().unwrap();
    header.split("t=").nth(1).unwrap().trim().parse().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// The full series bound under `holdout_after = 6` binds the training rows
/// alone: the state is saved at t = 6, the run says so, and the state and the
/// log-likelihood are the ones a run bound to the training-only file gives
/// (same seed, same rows, so byte-identical).
#[test]
fn save_final_state_under_a_holdout_binds_only_the_training_rows() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let cut = pfilter(tmp.path(), &["--save-final-state", "cut.tsv", "--no-store"]);
    assert!(cut.status.success(),
        "--save-final-state under holdout_after must cut to the training rows, \
         not refuse; stderr:\n{}", stderr(&cut));
    assert_eq!(origin_of(&tmp.path().join("cut.tsv")), 6.0);
    let err = stderr(&cut);
    assert!(err.contains("state saved at t = 6, the end of the training rows (holdout_after)")
            && err.contains("the log-likelihood shown covers training rows only"),
        "the run must say where the state is saved and what the loglik covers; \
         stderr:\n{err}");
    assert!(err.contains("6 observation boundaries"),
        "only the training rows may be bound; stderr:\n{err}");

    let file = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "file.tsv", "--no-store"]);
    assert!(file.status.success(), "{}", stderr(&file));
    assert_eq!(read(&tmp.path().join("cut.tsv")), read(&tmp.path().join("file.tsv")),
        "the cut run's state must equal the state from binding the training-only file");
    assert_eq!(stdout(&cut), stdout(&file),
        "the cut run's log-likelihood must equal the training-only file's");
}

/// The store path of the run's pfilter leaf, from its `stored` line or, on a
/// cache hit, its `cached` line. The path is one segment per identity level,
/// each carrying that level's hash, so two runs share a path exactly when
/// they share a run_id.
fn leaf(o: &Output) -> String {
    let err = stderr(o);
    err.lines().chain(stdout(o).lines())
        .find_map(|l| {
            let l = l.trim();
            l.strip_prefix("stored ").or_else(|| l.strip_prefix("cached "))
                .filter(|p| p.contains("pfilters/"))
                .map(|p| p.trim().to_string())
        })
        .unwrap_or_else(|| panic!("no stored leaf named; stderr:\n{err}"))
}

/// A cut that drops no row stores what the same evaluation stores without
/// `--save-final-state`, so it keeps that run_id: the training rows enter the
/// identity only when they change the stored log-likelihood. Both ways a cut
/// can drop nothing are checked: a run bound to the training-only file, and a
/// cutoff at or after every row.
#[test]
fn a_cut_that_drops_no_row_keeps_the_run_id() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let plain = pfilter(tmp.path(), &["--data", "cases=training.tsv"]);
    assert!(plain.status.success(), "{}", stderr(&plain));
    let saved = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs.tsv"]);
    assert!(saved.status.success(), "{}", stderr(&saved));
    assert_eq!(leaf(&plain), leaf(&saved),
        "a training-only file loses no row to the cut and must keep its run_id");

    let late = tempfile::tempdir().unwrap();
    fixture(late.path(), "holdout_after = 10");
    let plain = pfilter(late.path(), &[]);
    assert!(plain.status.success(), "{}", stderr(&plain));
    let saved = pfilter(late.path(), &["--save-final-state", "fs.tsv"]);
    assert!(saved.status.success(), "{}", stderr(&saved));
    assert_eq!(origin_of(&late.path().join("fs.tsv")), 10.0);
    assert_eq!(leaf(&plain), leaf(&saved),
        "a cutoff after every row drops nothing and must keep the run_id");
}

/// The cut is stored under its own identity: the training rows are keyed, so
/// the cut run's stored log-likelihood does not share a leaf with the
/// full-series run over the same file.
#[test]
fn the_cut_run_is_not_stored_under_the_full_series_identity() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let full = pfilter(tmp.path(), &[]);
    assert!(full.status.success(), "{}", stderr(&full));
    let cut = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(cut.status.success(), "{}", stderr(&cut));
    assert_ne!(leaf(&full), leaf(&cut),
        "a log-likelihood over the training rows must not share a run_id with one \
         over the full series");
}

/// An anchored cutoff resolves over the fit's own files, where `fit run`
/// resolved it (`last_obs` = 10, so t = 6), not over the `--data` files: over
/// the training rows alone `last_obs - 4 days` would resolve to t = 2 and cut
/// rows that trained.
#[test]
fn an_anchored_cutoff_resolves_over_the_fit_files() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = \"last_obs - 4 days\"");
    let cut = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(cut.status.success(), "{}", stderr(&cut));
    assert!(stderr(&cut).contains("(t = 6)"), "{}", stderr(&cut));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 6.0);
    let out = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs2.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs2.tsv")), 6.0);
}

/// `NA` rows past the cutoff are not training rows: the fit's partition drops
/// them, and so does the cut, so the state is saved at the training extent
/// rather than propagated through the unobserved tail. (Before gh#1138 the
/// state was saved at the last `NA` row, t = 10.)
#[test]
fn trailing_na_rows_past_the_cutoff_are_cut() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    std::fs::write(tmp.path().join("na_tail.tsv"),
        format!("{TRAINING}7\tNA\n8\tNA\n9\tNA\n10\tNA\n")).unwrap();
    let out = pfilter(tmp.path(),
        &["--data", "cases=na_tail.tsv", "--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 6.0);
}

/// A cutoff that leaves a stream no training row with a value is refused, as
/// `fit run` refuses it, and writes no state.
#[test]
fn a_cutoff_before_every_row_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 0");
    let out = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(!out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("no training data would remain"), "{}", stderr(&out));
    assert!(!tmp.path().join("fs.tsv").exists(), "no state file may be written");
}

/// A bare-time cutoff needs no rows, so the fit config's own files are not
/// read when `--data` replaces them: here the fit's file is gone, and only an
/// anchored cutoff (which must read it) fails on that.
#[test]
fn a_bare_time_cutoff_does_not_read_the_fit_files() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    std::fs::remove_file(tmp.path().join("cases.tsv")).unwrap();
    let out = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 6.0);

    let anchored = tempfile::tempdir().unwrap();
    fixture(anchored.path(), "holdout_after = \"last_obs - 4 days\"");
    std::fs::remove_file(anchored.path().join("cases.tsv")).unwrap();
    let out = pfilter(anchored.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs.tsv"]);
    assert!(!out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("read to resolve holdout_after"), "{}", stderr(&out));
}

/// Control: with no holdout the state is saved at the last row, as before.
#[test]
fn without_a_holdout_the_state_is_saved_at_the_last_row() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "");
    let out = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 10.0);
    assert!(!stderr(&out).contains("the end of the training rows"), "{}", stderr(&out));
}

/// Control: the cut is scoped to the saved state. A plain filter under the
/// same holdout still scores every row, and its log-likelihood is the one the
/// same filter gives with no holdout declared.
#[test]
fn the_filter_without_a_saved_state_still_runs_under_a_holdout() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let out = pfilter(tmp.path(), &["--no-store"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("10 observation boundaries"),
        "the filter must still score every row; stderr:\n{}", stderr(&out));
    assert!(!stderr(&out).contains("the end of the training rows"), "{}", stderr(&out));

    let plain = tempfile::tempdir().unwrap();
    fixture(plain.path(), "");
    let control = pfilter(plain.path(), &["--no-store"]);
    assert!(control.status.success(), "{}", stderr(&control));
    assert_eq!(stdout(&out), stdout(&control),
        "without --save-final-state the holdout must not change the log-likelihood");
}

/// gh#1138's case: a stream labelled by the start of its day (`covers =
/// day(time)`). The row labelled 6 measures [6, 7), so `holdout_after = 6`
/// holds it out (the fit's rule, on what a row measures, not on its label):
/// the cut keeps days 1 to 5, saves the state at t = 6, and equals a run bound
/// to a file of those five rows.
#[test]
fn a_start_labelled_row_is_cut_by_what_it_measures() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let day_model = MODEL.replace("closing_at(time, 1 'days)", "day(time)");
    assert_ne!(day_model, MODEL, "the fixture's covers clause moved");
    std::fs::write(tmp.path().join("sir.camdl"), day_model).unwrap();
    let out = Command::new(camdlc()).env_remove("CLICOLOR_FORCE")
        .arg(tmp.path().join("sir.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(tmp.path().join("sir.ir.json"), &out.stdout).unwrap();
    std::fs::write(tmp.path().join("days_1_5.tsv"),
        "time\tcases\n1\t3\n2\t4\n3\t5\n4\t6\n5\t8\n").unwrap();

    let cut = pfilter(tmp.path(), &["--save-final-state", "cut.tsv", "--no-store"]);
    assert!(cut.status.success(), "{}", stderr(&cut));
    assert_eq!(origin_of(&tmp.path().join("cut.tsv")), 6.0);
    let file = pfilter(tmp.path(),
        &["--data", "cases=days_1_5.tsv", "--save-final-state", "file.tsv", "--no-store"]);
    assert!(file.status.success(), "{}", stderr(&file));
    assert_eq!(read(&tmp.path().join("cut.tsv")), read(&tmp.path().join("file.tsv")),
        "the day-6 row measures [6, 7) and must be cut");
    assert_eq!(stdout(&cut), stdout(&file));
}
