//! Under `--progress plain`, the last progress line of `pfilter --replicates
//! N` names the mean log-likelihood the run reports: the running mean of every
//! finite replicate. A replicate's value was added to the running mean after
//! its count was ticked, so the line printed at the last count could leave the
//! last replicate out. One thread runs the replicates in order, so it always
//! did here.

use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.01, 1.0]
  gamma : rate in [0.01, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected + 0.5)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

fn bin(rel: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    assert!(p.exists(), "{} missing — run `make build`", p.display());
    p
}

#[test]
fn the_last_plain_progress_line_names_the_reported_mean() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let out = Command::new(bin("../../../ocaml/_build/default/bin/camdlc.exe"))
        .arg({
            std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
            dir.join("m.camdl")
        })
        .output()
        .unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();
    std::fs::write(dir.join("p.toml"), "beta = 0.3\ngamma = 0.1\n").unwrap();
    let out = Command::new(bin("../../target/release/camdl"))
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env_remove("CAMDL_STORE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["--progress", "plain", "pfilter", "m.ir.json", "--params", "p.toml", "--data", "cases.tsv"])
        .args(["--particles", "5", "--replicates", "3", "--parallel", "1", "--no-store"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "pfilter failed:\n{err}");
    let shown = err.lines().rev()
        .find_map(|l| l.split(" ll=").nth(1))
        .map(|v| v.split_whitespace().next().unwrap_or_default().to_string())
        .unwrap_or_else(|| panic!("a progress line names the mean:\n{err}"));
    let reported = err.lines()
        .find_map(|l| l.strip_prefix("loglik = "))
        .map(|v| v.split_whitespace().next().unwrap_or_default().to_string())
        .unwrap_or_else(|| panic!("the run reports its mean:\n{err}"));
    assert_eq!(shown, reported, "the last progress line names the reported mean:\n{err}");
}

/// Under `--progress plain` the last progress line of a run that ticks one
/// bar from several threads is its total, `pfilter N/N`, printed once. The
/// plain line is throttled, so a run shorter than the interval printed only
/// its first line, and a log ended on a count that suggested the run stopped
/// there.
#[test]
fn the_last_plain_progress_line_is_the_total() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    let out = Command::new(bin("../../../ocaml/_build/default/bin/camdlc.exe"))
        .arg(dir.join("m.camdl"))
        .output()
        .unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();
    std::fs::write(dir.join("p.toml"), "beta = 0.3\ngamma = 0.1\n").unwrap();
    let out = Command::new(bin("../../target/release/camdl"))
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["--progress", "plain", "pfilter", "m.ir.json", "--params", "p.toml", "--data", "cases.tsv"])
        .args(["--particles", "5", "--replicates", "40", "--no-store"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "pfilter failed:\n{err}");
    let counts: Vec<&str> = err.lines().map(str::trim)
        .filter(|l| l.starts_with("pfilter ") && l.as_bytes().get(8).is_some_and(u8::is_ascii_digit))
        .collect();
    assert_eq!(counts.last().map(|l| l.split_whitespace().nth(1)), Some(Some("40/40")),
        "the last progress line must be the total; the progress lines were {counts:?}");
    assert_eq!(counts.iter().filter(|l| l.contains("40/40")).count(), 1,
        "and the total is printed once: {counts:?}");
}
