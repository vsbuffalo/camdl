//! Observation periods whose starts and ends lie off the integration grid
//! (`docs/dev/temporal-architecture.md` §4.5, decision D43).
//!
//! A method that steps exactly to every period boundary (if2, pfilter, pmmh)
//! fits such data as it is. PGAS steps only on the grid `t_start + k·dt`, so a
//! period boundary off that grid would be scored on a different period; it is
//! refused, naming the row and the two grid times either side of the boundary.
//!
//! Fixture F1 of the temporal-architecture migration
//! (`docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`,
//! Part 6): two-and-a-half-day periods closing at 2.5, 5, …, 15 under
//! `dt = 1`. Its PGAS variant starts the run at 0.5 with two-day periods
//! closing at 3, 5, 7, 9: every label is a whole number of days, but every
//! period boundary lies half a step off the grid 0.5, 1.5, 2.5, ….

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} - run `make build-ocaml` or `make test`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_offgrid_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// F1's model with its run start, period width and step given, compiled to IR.
fn write_model(dir: &Path, from: f64, width: f64, dt: f64) -> PathBuf {
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{ beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0]  rho : probability in [0.01, 1.0] }}
transitions {{ infection : S --> I @ beta * S * I / (S + I + R)
              recovery  : I --> R @ gamma * I }}
observations {{ cases {{ columns {{ time : time, cases : count }}
  covers = closing_at(time, {width} 'days)  projected = incidence(infection)
  cases ~ poisson(rate = rho * projected + 0.1) }} }}
init {{ S = 990  I = 10  R = 0 }}
simulate {{ from = {from} 'days  to = 30 'days  dt = {dt} 'days }}
"#);
    let model = dir.join("f1.camdl");
    std::fs::write(&model, src).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store()).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("f1.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

fn write_data(dir: &Path, rows: &[(f64, u32)]) -> PathBuf {
    let mut body = String::from("time\tcases\n");
    for (t, n) in rows {
        body.push_str(&format!("{t}\t{n}\n"));
    }
    let p = dir.join("cases.tsv");
    std::fs::write(&p, body).unwrap();
    p
}

/// F1's data: periods closing at 2.5, 5, …, 15.
fn f1_rows() -> Vec<(f64, u32)> {
    [4, 7, 11, 16, 21, 24].iter().enumerate().map(|(i, &n)| (2.5 * (i + 1) as f64, n)).collect()
}

fn method_block(algorithm: &str) -> &'static str {
    match algorithm {
        "if2" => "chains = 1\nparticles = 60\niterations = 2\ncooling = 0.7\n",
        "pfilter" => "particles = 60\nreplicates = 1\nrecord_prequential = false\n",
        "pmmh" => "chains = 1\nparticles = 30\niterations = 6\nburn_in = 2\nthin = 1\n",
        "pgas" => "chains = 1\nparticles = 20\nsweeps = 4\nburn_in = 1\nthin = 1\n",
        other => panic!("no method block for {other}"),
    }
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, algorithm: &str) -> PathBuf {
    let toml = format!(r#"
store = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 2.0], prior = {{ log_normal = {{ mu = -1.0, sigma = 0.5 }} }}, start = 0.4 }}
[fixed]
gamma = 0.2
rho = 0.5
[method]
algorithm = "{algorithm}"
backend = "chain_binomial"
{block}"#,
        out = dir.join("results").display(), ir = ir.display(), data = data.display(),
        block = method_block(algorithm),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

fn fit_run(toml: &Path) -> std::process::Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
        .args(["fit", "run"]).arg(toml).args(["--seed", "7", "--no-progress"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn camdl")
}

/// if2, pfilter and pmmh step exactly to every period boundary, so F1 fits
/// under each of them as it is.
#[test]
fn f1_off_grid_periods_fit_under_every_exact_stepping_method() {
    for algorithm in ["if2", "pfilter", "pmmh"] {
        let tmp = tempdir(algorithm);
        let ir = write_model(tmp.path(), 0.0, 2.5, 1.0);
        let data = write_data(tmp.path(), &f1_rows());
        let toml = write_fit_toml(tmp.path(), &ir, &data, algorithm);
        let o = fit_run(&toml);
        assert!(o.status.success(),
            "[{algorithm}] periods closing at 2.5, 5, … under dt = 1 must fit:\n{}",
            String::from_utf8_lossy(&o.stderr));
    }
}

/// PGAS cannot step to a boundary off its grid; F1's PGAS variant is refused,
/// naming the row and the grid times either side of its boundary, instead of
/// being scored on the period one half-step later.
#[test]
fn f1_pgas_variant_is_refused_naming_the_row_and_the_grid_times() {
    let tmp = tempdir("pgas");
    let ir = write_model(tmp.path(), 0.5, 2.0, 1.0);
    let data = write_data(tmp.path(), &[(3.0, 4), (5.0, 7), (7.0, 11), (9.0, 16)]);
    let toml = write_fit_toml(tmp.path(), &ir, &data, "pgas");
    let o = fit_run(&toml);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "PGAS must refuse periods off its grid; it ran:\n{stderr}");
    // The first row covers [1, 3): its start 1 lies between grid times 0.5 and
    // 1.5.
    for needle in ["pgas", "`cases`", "row 1", "time = 3", "[1, 3)", "0.5", "1.5", "dt = 1"] {
        assert!(stderr.contains(needle), "the refusal must name `{needle}`:\n{stderr}");
    }
}

/// The same variant with its periods on the grid (two-day periods closing at
/// 2.5, 4.5, …) runs under PGAS: the refusal is about the boundaries, not the
/// run's start.
#[test]
fn pgas_runs_periods_on_its_grid_from_an_off_integer_start() {
    let tmp = tempdir("pgas_on_grid");
    let ir = write_model(tmp.path(), 0.5, 2.0, 1.0);
    let data = write_data(tmp.path(), &[(2.5, 4), (4.5, 7), (6.5, 11), (8.5, 16)]);
    let toml = write_fit_toml(tmp.path(), &ir, &data, "pgas");
    let o = fit_run(&toml);
    assert!(o.status.success(), "periods on PGAS's grid must run:\n{}", String::from_utf8_lossy(&o.stderr));
}

/// A run store of its own for one command, so tests that run the same filter
/// at once do not meet in a shared root (a filter run another live process
/// holds is refused).
fn isolated_store() -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "camdl-store-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}
