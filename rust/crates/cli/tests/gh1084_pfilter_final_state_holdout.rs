//! gh#1084: `camdl pfilter --fit <fit.toml> --save-final-state FILE` ignored
//! the fit's `holdout_after`. The filter ran through every bound row, held-out
//! ones included, and saved the swarm at the last of them — a forecast origin
//! that already conditions on the rows a forecast from it is scored against.
//!
//! The filter itself still scores the full series (its log-likelihood, trace
//! and prequential outputs are unchanged, and `compare` relies on them); only
//! the saved state is refused while the bound rows run past the cutoff. Binding
//! training rows alone with `--data` saves the state at the training extent.
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

/// The defect: a held-out fit's rows past the cutoff reach the saved state.
#[test]
fn save_final_state_past_the_cutoff_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let out = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    let err = stderr(&out);
    assert!(!out.status.success(),
        "--save-final-state under holdout_after must not save a state filtered \
         through the held-out rows; stderr:\n{err}");
    assert!(err.contains("--save-final-state") && err.contains("holdout_after"),
        "the refusal must name the flag and the declaration; stderr:\n{err}");
    assert!(!tmp.path().join("fs.tsv").exists(), "no state file may be written");
}

/// The remedy the refusal names: bind the training rows alone and the state is
/// saved at the training extent.
#[test]
fn training_rows_alone_save_at_the_training_extent() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let out = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 6.0);
}

/// An anchored cutoff resolves over the fit's own files, where `fit run`
/// resolved it (`last_obs` = 10, so t = 6), not over the `--data` files: over
/// the training rows alone `last_obs - 4 days` would resolve to t = 2 and
/// refuse the rows that trained.
#[test]
fn an_anchored_cutoff_resolves_over_the_fit_files() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = \"last_obs - 4 days\"");
    let refused = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(!refused.status.success(), "{}", stderr(&refused));
    assert!(stderr(&refused).contains("(t = 6)"), "{}", stderr(&refused));
    let out = pfilter(tmp.path(),
        &["--data", "cases=training.tsv", "--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 6.0);
}

/// Control: with no holdout the state is saved at the last row, as before.
#[test]
fn without_a_holdout_the_state_is_saved_at_the_last_row() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "");
    let out = pfilter(tmp.path(), &["--save-final-state", "fs.tsv"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(origin_of(&tmp.path().join("fs.tsv")), 10.0);
}

/// Control: the refusal is scoped to the saved state. A plain filter under the
/// same holdout still runs over the full series.
#[test]
fn the_filter_without_a_saved_state_still_runs_under_a_holdout() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path(), "holdout_after = 6");
    let out = pfilter(tmp.path(), &[]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("10 observation boundaries"),
        "the filter must still score every row; stderr:\n{}", stderr(&out));
}
