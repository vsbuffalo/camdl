//! gh#161: `dt` is a model knob. The model's `simulate { dt = … }` is the
//! default simulation step; an explicit `--dt` overrides it. The effective dt
//! is recorded in the stored `run.json` config label (`<backend>-dt<dt>`), so
//! we read it back from CAS rather than parsing dynamics.

use std::path::{Path, PathBuf};
use std::process::Command;

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

fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p.clone()); }
                else { out.push(p); }
            }
        }
    }
    out
}

/// Read the effective (backend, dt) from the single stored sim run.json.
fn read_sim_config(output_root: &Path) -> (String, String) {
    let run_jsons: Vec<_> = walkdir(&output_root.join("sims")).into_iter()
        .filter(|p| p.file_name().map(|s| s == "run.json").unwrap_or(false))
        .collect();
    assert_eq!(run_jsons.len(), 1,
        "expected exactly one run.json under {}, got {:?}",
        output_root.display(), run_jsons);
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&run_jsons[0]).unwrap()).unwrap();
    let label = v["levels"].as_array().unwrap().iter()
        .find(|l| l["name"] == "config").expect("config level")
        ["label"].as_str().unwrap().to_string();
    let (backend, dt) = label.rsplit_once("-dt")
        .unwrap_or_else(|| panic!("config label should be <backend>-dt<dt>, got {label}"));
    (backend.to_string(), dt.to_string())
}

fn run_simulate(bin: &Path, tmp: &Path, extra: &[&str]) -> PathBuf {
    let output = tmp.join("out");
    let mut args: Vec<String> = vec![
        "simulate".into(), golden_sir_dt().to_string_lossy().to_string(),
        // The baseline scenario supplies parameter values; dt is orthogonal to
        // the scenario, so this isolates the dt-knob behavior under test.
        "--scenario".into(), "baseline".into(),
        "--seed".into(), "1".into(),
        "--output-dir".into(), output.to_string_lossy().to_string(),
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
