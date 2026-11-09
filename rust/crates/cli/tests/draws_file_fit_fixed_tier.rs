//! The fit config's `[fixed]` is tier 2 of the run spec §1.3 chain, below
//! `--params` (tier 3) and a draw row (tier 3.5), on both `--fit` draws paths.
//!
//! `simulate --draws prior --fit F --params P` already resolved it there (the
//! config's `[fixed]` rides on `SimRun::fit_fixed`). `--draws FILE --fit F`
//! used to copy `[fixed]` into every row of the file for each column the file
//! lacks, so the values resolved at tier 3.5 and discarded `--params`: with
//! `[fixed] N0 = 1000` and `--params` giving `N0 = 2000`, the prior path ran at
//! 2000 and the file path at 1000.

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
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

const FIT: &str = r#"output_dir = "res"
[model]
camdl = "m.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
beta = { bounds = [0.01, 1.0], start = 0.3 }
[fixed]
gamma = 0.1
N0 = 1000
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 3
cooling = 0.7
"#;

/// The same config with a prior on beta, which `--draws prior --fit` needs.
fn fit_with_prior() -> String {
    FIT.replace("start = 0.3 }", "start = 0.3, prior = { uniform = {} } }")
}

fn setup(dir: &Path) {
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("m.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t10\n2\t11\n3\t10\n").unwrap();
    std::fs::write(dir.join("fit.toml"), FIT).unwrap();
    std::fs::write(dir.join("fit_prior.toml"), fit_with_prior()).unwrap();
    std::fs::write(dir.join("p2000.toml"), "N0 = 2000\n").unwrap();
    std::fs::write(dir.join("rows.tsv"), "beta\tgamma\n0.2\t0.1\n0.4\t0.1\n").unwrap();
    std::fs::write(dir.join("rows_n0.tsv"), "beta\tN0\n0.2\t1500\n0.4\t1500\n").unwrap();
}

/// `camdl simulate m.ir.json <args>` in `dir/<cwd>`; returns the N0 every cell
/// resolved to (one value, asserted), from each leaf's run.json.
fn simulate_n0(dir: &Path, cwd: &str, args: &[&str]) -> f64 {
    let work = dir.join(cwd);
    std::fs::create_dir_all(&work).unwrap();
    let out: Output = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(&work)
        .args(["simulate", &dir.join("m.ir.json").to_string_lossy()])
        .args(args)
        .arg("--seed").arg("1")
        .output()
        .expect("spawn camdl simulate");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "simulate {args:?} failed: {err}");
    let mut found = Vec::new();
    collect_n0(&work.join("results/sims"), &mut found);
    assert!(!found.is_empty(), "no leaf run.json under {}/results/sims; stderr:\n{err}", cwd);
    assert!(found.iter().all(|v| *v == found[0]), "cells disagree on N0: {found:?}");
    found[0]
}

fn collect_n0(dir: &Path, out: &mut Vec<f64>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_n0(&p, out);
        } else if p.file_name().is_some_and(|n| n == "run.json") {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            if let Some(n0) = v["inputs"]["parameters_provenance"]["N0"]["value"].as_f64() {
                out.push(n0);
            }
        }
    }
}

/// `--params` outranks the fit config's `[fixed]` on both `--fit` draws paths.
#[test]
fn params_outranks_the_fit_fixed_on_both_draws_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let prior = simulate_n0(dir, "prior", &["--draws", "prior", "--fit", "../fit_prior.toml",
        "--params", "../p2000.toml", "-n", "2"]);
    let file = simulate_n0(dir, "file", &["--draws", "../rows.tsv", "--fit", "../fit.toml",
        "--params", "../p2000.toml"]);
    assert_eq!(prior, 2000.0, "--draws prior: --params (tier 3) over [fixed] (tier 2)");
    assert_eq!(file, 2000.0,
        "--draws FILE: --params (tier 3) over [fixed] (tier 2), as the prior path");
}

/// What the file path's backfill was for: a draws file that lacks a fixed
/// parameter's column still gets the fit's `[fixed]` value rather than nothing,
/// and a column the file does carry outranks both `[fixed]` and `--params`.
#[test]
fn the_fit_fixed_still_reaches_a_column_the_draws_file_lacks() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let lacking = simulate_n0(dir, "lacking", &["--draws", "../rows.tsv", "--fit", "../fit.toml"]);
    assert_eq!(lacking, 1000.0, "[fixed] fills the column the file lacks");
    let carried = simulate_n0(dir, "carried", &["--draws", "../rows_n0.tsv", "--fit", "../fit.toml",
        "--params", "../p2000.toml"]);
    assert_eq!(carried, 1500.0, "a draw row (tier 3.5) outranks --params and [fixed]");
}
