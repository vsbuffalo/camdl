//! `--starts from_params=<file>` accepts the `mle_params.toml` a fit writes.
//!
//! An IF2 fit writes its maximum-likelihood estimate to `mle_params.toml`:
//! one numeric key per parameter, then a `[provenance]` table of fit metadata
//! (backend, dt, camdl version, log-likelihood). Every `--params` consumer
//! skips that table. `from_params` refused it ("key `provenance` has
//! non-numeric value"), so a PGAS or pfilter run could not start from an
//! IF2 estimate written by camdl itself.
//!
//! End-to-end via the built `camdl` binary. A missing release binary or
//! `camdlc.exe` fails the test (run `make build`) rather than passing it
//! without running.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test"))
}

fn camdl_bin() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

const MODEL: &str = r#"
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
"#;

const CASES_TSV: &str =
    "time\tcases\n1\t12\n2\t15\n3\t19\n4\t24\n5\t30\n6\t36\n7\t42\n8\t48\n9\t52\n10\t55\n";

/// `<name>.toml`: a fit of beta and gamma with the `[method]` body given.
fn fit_toml(dir: &Path, name: &str, method: &str) {
    std::fs::write(dir.join(format!("{name}.toml")), format!(
        "store = \"res_{name}\"\n[model]\ncamdl = \"m.ir.json\"\n\
         [data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.01, 3.0] }}\ngamma = {{ bounds = [0.01, 1.0] }}\n\
         [fixed]\nN0 = 1000\n\
         [method]\n{method}"
    )).unwrap();
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

/// Every file named `name` under `root`.
fn find(root: &Path, name: &str) -> Vec<PathBuf> {
    fn walk(d: &Path, name: &str, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, name, out);
            } else if p.file_name().is_some_and(|n| n == name) {
                out.push(p);
            }
        }
    }
    let mut found = Vec::new();
    walk(root, name, &mut found);
    found
}

#[test]
fn from_params_starts_from_an_if2_fits_own_mle_params_toml() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES_TSV).unwrap();

    fit_toml(dir, "mle",
        "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
         particles = 100\niterations = 3\ncooling = 0.7\nstarts = \"lhs\"\n");
    let out = camdl(dir, &["fit", "run", "mle.toml", "--seed", "1", "--progress", "none"]);
    assert!(out.status.success(), "the IF2 fit runs: {}", String::from_utf8_lossy(&out.stderr));
    let mut mles = find(&dir.join("res_mle"), "mle_params.toml");
    assert_eq!(mles.len(), 1, "one mle_params.toml: {mles:?}");
    let mle = mles.pop().unwrap();
    let text = std::fs::read_to_string(&mle).unwrap();
    assert!(text.contains("[provenance]"),
        "fixture precondition: the file the fit wrote carries a [provenance] table:\n{text}");

    // A pfilter run scores one θ, so it is the cheapest consumer of a point.
    fit_toml(dir, "score",
        "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 100\n");
    let flag = format!("from_params={}", mle.display());
    let out = camdl(dir, &["fit", "run", "score.toml", "--starts", &flag, "--progress", "none"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "`--starts from_params=<mle_params.toml>` must accept the fit's own file:\n{err}");
    assert!(!err.contains("non-numeric"), "{err}");
}
