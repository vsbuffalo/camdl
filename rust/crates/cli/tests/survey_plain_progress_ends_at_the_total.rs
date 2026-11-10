//! Under `--progress plain` the last progress line a survey prints is its
//! total, `survey N/N`, printed once.
//!
//! The plain line is throttled (one per 30 s per bar), so a survey that runs
//! in less than the interval printed only its first line — `survey 1/40` — and
//! a log ended on a count that suggested the survey stopped there.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 1.0]
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
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

#[test]
fn the_last_plain_progress_line_is_the_total() {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let tmp = Tmp(std::env::temp_dir()
        .join(format!("camdl_survey_progress_{}_{ns}", std::process::id())));
    let dir = &tmp.0;
    std::fs::create_dir_all(dir).unwrap();
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"),
        "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();

    let out = Command::new(camdl_bin())
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["--progress", "plain", "survey", "m.ir.json", "--data", "cases.tsv",
               "--estimate", "beta=0.1:0.9", "--fixed", "gamma=0.2", "--fixed", "N0=1000",
               "--n-points", "40", "--eval", "simulate"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "survey failed:\n{err}");
    let counts: Vec<&str> = err.lines().map(str::trim)
        .filter(|l| l.starts_with("survey ") && l.as_bytes().get(7).is_some_and(u8::is_ascii_digit))
        .collect();
    assert_eq!(counts.last().map(|l| l.split_whitespace().nth(1)), Some(Some("40/40")),
        "the last progress line must be the total; the progress lines were {counts:?}");
    assert_eq!(counts.iter().filter(|l| l.contains("40/40")).count(), 1,
        "and the total is printed once: {counts:?}");
}
