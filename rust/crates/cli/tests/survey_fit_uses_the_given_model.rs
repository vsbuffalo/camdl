//! `camdl survey MODEL --fit FIT` scores MODEL, the model on the command line.
//!
//! `--fit` supplies the `[estimate]` bounds, the `[data]` and the `[fixed]`
//! values (`survey --help`, run spec "`camdl survey MODEL`"), as it does for
//! `pfilter --fit` and `profile --fit`. The fit-aware branch once compiled the
//! config's own `[model]` instead and never read MODEL at all: a positional
//! path that did not exist ran to exit 0, and a different model scored the
//! config's model while the store leaf and `run.json` were named after the
//! positional one.

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
        .join(format!("camdl_survey_model_{tag}_{}_{ns}", std::process::id()));
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

fn setup(tag: &str) -> Tmp {
    let tmp = tempdir(tag);
    compile(&tmp.0, "fitted", 10);
    compile(&tmp.0, "other", 50);
    std::fs::write(tmp.0.join("cases.tsv"),
        "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();
    std::fs::write(tmp.0.join("fit.toml"), r#"
[model]
camdl = "fitted.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
beta = { bounds = [0.1, 0.9] }
[fixed]
gamma = 0.2
N0    = 1000
"#).unwrap();
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

#[test]
fn a_model_path_that_does_not_exist_is_refused() {
    let tmp = setup("missing");
    let out = survey(&tmp.0, "nope.ir.json", "res");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a missing MODEL must be refused, not ignored:\n{err}");
    assert!(err.contains("nope.ir.json"), "the refusal names the missing model:\n{err}");
}

/// The two models differ only in their initial state, so with the same box,
/// data and seed the landscapes differ exactly when the positional model is
/// the one scored.
#[test]
fn the_positional_model_is_the_one_scored() {
    let tmp = setup("scored");
    for (model, out_dir) in [("fitted.ir.json", "res_fitted"), ("other.ir.json", "res_other")] {
        let out = survey(&tmp.0, model, out_dir);
        assert!(out.status.success(), "survey {model} failed:\n{}",
            String::from_utf8_lossy(&out.stderr));
    }
    let fitted = logliks(&tmp.0.join("res_fitted"));
    let other = logliks(&tmp.0.join("res_other"));
    assert_eq!(fitted.len(), 4);
    assert_ne!(fitted, other,
        "surveying other.ir.json must score other.ir.json, not the config's fitted.ir.json");
}
