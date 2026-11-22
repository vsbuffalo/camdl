//! gh#161: `dt` is a model knob. The model's `simulate { dt = … }` is the
//! default simulation step; an explicit `--dt` overrides it. The effective dt
//! is recorded in the stored cell's identity record (`config.dt`), so
//! we read it back from the store rather than parsing dynamics.

use std::path::{Path, PathBuf};
use std::process::Command;

mod sim_store_cells;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing() -> PathBuf {
    let b = binary();
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        b.display()
    );
    b
}

/// `ocaml/golden/sir_dt.ir.json` declares `simulate { dt = 0.5 }`.
fn golden_sir_dt() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/sir_dt.ir.json")
}

/// Read the effective (backend, dt) from the single stored sim cell's
/// identity record.
fn read_sim_config(store: &Path) -> (String, String) {
    let cell = sim_store_cells::only_sim_cell(store);
    let config = cell.config();
    let backend = config["backend"].as_str().expect("config.backend").to_string();
    let dt = config["dt"].as_f64().expect("config.dt").to_string();
    (backend, dt)
}

fn run_simulate(bin: &Path, tmp: &Path, extra: &[&str]) -> PathBuf {
    let output = tmp.join("camdl-store");
    let mut args: Vec<String> = vec![
        "simulate".into(), golden_sir_dt().to_string_lossy().to_string(),
        // The baseline scenario supplies parameter values; dt is orthogonal to
        // the scenario, so this isolates the dt-knob behavior under test.
        "--scenario".into(), "baseline".into(),
        "--seed".into(), "1".into(),
        "--store".into(), output.to_string_lossy().to_string(),
        "-o".into(), tmp.join("t.tsv").to_string_lossy().to_string(),
    ];
    for a in extra { args.push(a.to_string()); }
    let out = Command::new(bin).env_remove("CLICOLOR_FORCE").args(&args).output().expect("spawn");
    assert!(out.status.success(), "simulate should succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    output
}

#[test]
fn model_dt_is_the_default_step() {
    // No --dt passed → the model's `simulate { dt = 0.5 }` is the effective
    // step. Pre-fix the CLI ignored the model knob and used dt=1.0.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let output = run_simulate(&bin, tmp.path(), &[]);
    let (_backend, dt) = read_sim_config(&output);
    assert_eq!(dt, "0.5",
        "model `simulate {{ dt = 0.5 }}` must be the default step when --dt \
         is not passed (got dt={dt})");
}

#[test]
fn explicit_dt_overrides_model_dt() {
    // --dt 0.25 must win over the model's dt = 0.5 (the override is for
    // sensitivity sweeps / Richardson extrapolation).
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let output = run_simulate(&bin, tmp.path(), &["--dt", "0.25"]);
    let (_backend, dt) = read_sim_config(&output);
    assert_eq!(dt, "0.25",
        "explicit --dt must override the model's dt = 0.5 (got dt={dt})");
}
