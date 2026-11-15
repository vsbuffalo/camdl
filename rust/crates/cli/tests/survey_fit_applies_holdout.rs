//! `camdl survey MODEL --fit FIT` scores the rows `fit run FIT` trains on
//! (gh#1059). A config's `[data] holdout_after` removes every row after the
//! cutoff from training; a survey of that config must leave them out of every
//! point's likelihood too, or it ranks start points on data the fit never
//! sees.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir()
        .join(format!("camdl_survey_holdout_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// A closed SIR seeded with `i0` infected, compiled to `name.ir.json` with the
/// in-tree compiler (so nothing here reads an installed `camdlc`).
fn compile(dir: &Path, name: &str, i0: u32) -> PathBuf {
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.01, 1.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = {s0}  I = {i0} }}
simulate {{ from = 0 'days  to = 6 'days }}
"#, s0 = 1000 - i0);
    let src_path = dir.join(format!("{name}.camdl"));
    std::fs::write(&src_path, src).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{name}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// A fit config estimating `beta`; `extra_data` goes into `[data]`, `cases`
/// names the data file.
fn fit_toml(extra_data: &str, cases: &str) -> String {
    format!(r#"
[model]
camdl = "fitted.ir.json"
[data]
{extra_data}
[data.observations]
cases = "{cases}"
[estimate]
beta = {{ bounds = [0.1, 0.9] }}
[fixed]
gamma = 0.2
N0    = 1000
"#)
}

fn setup(tag: &str, fit: &str) -> Tmp {
    let tmp = tempdir(tag);
    compile(&tmp.0, "fitted", 10);
    std::fs::write(tmp.0.join("cases.tsv"),
        "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();
    std::fs::write(tmp.0.join("train.tsv"),
        "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n").unwrap();
    std::fs::write(tmp.0.join("fit.toml"), fit).unwrap();
    tmp
}

fn survey(dir: &Path, model: &str, out_dir: &str) -> Output {
    Command::new(camdl_bin())
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["survey", model, "--fit", "fit.toml", "--n-points", "4", "--eval", "simulate",
               "--output", out_dir])
        .output()
        .unwrap()
}

/// The `loglik` column of the one `landscape.tsv` under `root`.
fn logliks(root: &Path) -> Vec<String> {
    fn find(d: &Path) -> Option<PathBuf> {
        for e in std::fs::read_dir(d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Some(f) = find(&p) { return Some(f); }
            } else if p.file_name().is_some_and(|n| n == "landscape.tsv") {
                return Some(p);
            }
        }
        None
    }
    let path = find(root).unwrap_or_else(|| panic!("no landscape.tsv under {}", root.display()));
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = header.iter().position(|&c| c == "loglik")
        .unwrap_or_else(|| panic!("no loglik column in {header:?}"));
    lines.map(|l| l.split('\t').nth(col).unwrap().to_string()).collect()
}

/// With `holdout_after = 4` the fit trains on rows t <= 4. Every survey point
/// must score exactly what a survey of the same config without the holdout,
/// bound to a file of those rows, scores.
#[test]
fn a_holdout_after_is_left_out_of_every_point() {
    let reference = {
        let tmp = setup("reference", &fit_toml("", "train.tsv"));
        let out = survey(&tmp.0, "fitted.ir.json", "res");
        assert!(out.status.success(), "survey of the training rows failed:\n{}",
            String::from_utf8_lossy(&out.stderr));
        logliks(&tmp.0.join("res"))
    };
    assert_eq!(reference.len(), 4);
    let tmp = setup("held", &fit_toml("holdout_after = 4", "cases.tsv"));
    let out = survey(&tmp.0, "fitted.ir.json", "res");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "survey of a holdout_after config failed:\n{err}");
    assert_eq!(logliks(&tmp.0.join("res")), reference,
        "every point must score the training rows only");
    assert!(err.contains("2 observation row(s) at t > 4 withheld from training"),
        "the run must say what the holdout withheld:\n{err}");
}

/// `[data.holdout]` files leave the training files whole, so a survey scores
/// what the fit trains on; the files are checked as `fit run` checks them:
/// every held-out row must lie after the latest training row. A holdout that
/// overlaps the training data is refused.
#[test]
fn explicit_holdout_files_are_checked_against_the_training_data() {
    let holdout = "[data.holdout]\ncases = \"held.tsv\"";
    for (what, training, ok) in [("tail", "train.tsv", true), ("overlapping", "cases.tsv", false)] {
        let tmp = setup(what, &fit_toml(holdout, training));
        std::fs::write(tmp.0.join("held.tsv"), "time\tcases\n5\t22\n6\t24\n").unwrap();
        let out = survey(&tmp.0, "fitted.ir.json", "res");
        let err = String::from_utf8_lossy(&out.stderr);
        if ok {
            assert!(out.status.success(), "{what}: {err}");
            assert!(err.contains("2 held-out observation row(s) bound as the tail"),
                "{what}: the run must say the holdout was checked:\n{err}");
        } else {
            assert!(!out.status.success(), "{what}: must be refused:\n{err}");
            assert!(err.contains("[data.holdout]") && err.contains("held-out rows"),
                "{what}: the refusal names the holdout:\n{err}");
        }
    }
}
