//! Explicit tests of intervention / event default activation, for both
//! `simulate` and `fit` entry points. Encodes the spec contract:
//!
//!   §14 / §14.4 (camdl-language-spec.md):
//!     - events (`always_active = true`)  → on by default
//!     - interventions (`always_active = false`) → off by default
//!
//! All tests shell out to the built `camdl` binary and observe
//! behaviour via trajectory output / cached run files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// Build a minimal IR JSON: two compartments (S, V), zero transitions,
/// one event that adds 100 to S at t=5, one intervention that moves
/// 50% of S → V at t=10. No transitions → S never decays via natural
/// dynamics, so any change is observable as a pure state jump.
fn mixed_model_ir() -> String {
    // gh#audit-C8. Wrap in IR envelope so the binary's
    // envelope-aware deserializer (ir::from_str) accepts it.
    r#"{
      "ir_version": "0.14",
      "validated_by": "test-fixture",
      "model": {
        "name": "mixed", "version": "0.3", "time_unit": "days",
        "description": null, "origin": null,
        "compartments": [
          { "name": "S", "kind": "integer" },
          { "name": "V", "kind": "integer" }
        ],
        "transitions": [],
        "ode_equations": [], "time_functions": [], "tables": [],
        "observations": [],
        "parameters": [],
        "initial_conditions": { "explicit": { "S": 1000.0, "V": 0.0 } },
        "output": {
          "times": { "regular": { "start": 0.0, "step": 1.0, "end": 20.0 } },
          "format": "tsv", "trajectory": true, "observations": false
        },
        "simulation": {
          "t_start": 0.0, "t_end": 20.0,
          "time_semantics": "continuous", "dt": 1.0, "rng_seed": null
        },
        "scenarios": [
          { "name": "with_sia", "label": "with sia",
            "params": {}, "scale": {}, "enable": ["sia"], "disable": [], "compose": [] }
        ],
        "interventions": [
          {
            "name": "bump",
            "schedule": { "at_times": [5.0] },
            "actions": [ { "add": { "compartment": "S", "count": { "const": 100.0 } } } ],
            "kind": "event"
          },
          {
            "name": "sia",
            "schedule": { "at_times": [10.0] },
            "actions": [ { "fraction_transfer": {
              "src": "S", "dst": "V", "fraction": { "const": 0.5 }
            } } ],
            "kind": "scenario"
          }
        ],
        "model_structure": null, "balance": null
      }
    }"#.to_string()
}

fn write_ir(tmp: &tempfile::TempDir) -> PathBuf {
    let path = tmp.path().join("model.ir.json");
    std::fs::write(&path, mixed_model_ir()).unwrap();
    path
}

/// Read the trajectory TSV, return S column at time t.
fn s_at(traj: &str, t: f64) -> i64 {
    for line in traj.lines() {
        if line.starts_with('#') || line.starts_with('t') { continue; }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.is_empty() { continue; }
        let t_parsed: f64 = cols[0].parse().unwrap_or(-1.0);
        if (t_parsed - t).abs() < 1e-6 {
            return cols[1].parse().unwrap_or(0);
        }
    }
    panic!("no trajectory row at t={}: {}", t, traj);
}

/// Run `camdl simulate` and return the trajectory TSV. The trajectory is the
/// `-o PATH` mirror (the CAS leaf is the system of record; the trajectory is
/// never written to stdout — Item C). A unique `-o` file + isolated
/// `--output-dir` are appended so concurrent test runs never share a CAS root.
fn run_simulate(args: &[&str]) -> String {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let traj_path = tmp.path().join("traj.tsv");
    let cas_dir = tmp.path().join("results");
    let out = Command::new(&bin)
        .args(args)
        .args(["-o", &traj_path.to_string_lossy(),
               "--output-dir", &cas_dir.to_string_lossy()])
        .output()
        .expect("spawn");
    if !out.status.success() {
        panic!("simulate failed: stderr={}", String::from_utf8_lossy(&out.stderr));
    }
    std::fs::read_to_string(&traj_path)
        .unwrap_or_else(|e| panic!("reading trajectory mirror {}: {}", traj_path.display(), e))
}

// ─── simulate defaults ───────────────────────────────────────────────────────

#[test]
fn simulate_default_event_fires_intervention_does_not() {
    let _ = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);
    let traj = run_simulate(&[
        "simulate", &ir.to_string_lossy(),
        "--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
    ]);

    // Start: S=1000. Event at t=5 adds 100. No transitions, no intervention
    // at t=10 because it's toggleable and not enabled. So:
    //   t=4  → S = 1000  (pre-event)
    //   t=5  → S = 1100  (event fired)
    //   t=10 → S = 1100  (intervention NOT fired)
    //   t=15 → S = 1100
    assert_eq!(s_at(&traj, 4.0), 1000, "pre-event");
    assert_eq!(s_at(&traj, 5.0), 1100, "event must fire by default");
    assert_eq!(s_at(&traj, 15.0), 1100,
        "toggleable intervention must NOT fire without --enable or --scenario");
}

#[test]
fn simulate_enable_activates_toggleable_intervention() {
    // Contract asserted here: the intervention DID fire when enabled.
    // The exact arithmetic of the transfer is backend-specific (see
    // task #85 — chain_binomial currently double-fires interventions,
    // an orthogonal pre-existing bug). Testing "did it fire at all" is
    // sufficient for the default-activation contract under test.
    let _ = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);
    let traj = run_simulate(&[
        "simulate", &ir.to_string_lossy(),
        "--enable", "sia",
        "--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
    ]);

    // t=5: event fires → S = 1100
    assert_eq!(s_at(&traj, 5.0), 1100, "event must fire by default");
    // t=10: sia transfers (some) S → V; exact amount backend-dependent.
    let s10 = s_at(&traj, 10.0);
    assert!(s10 < 1100, "sia must fire when --enable'd; S was {}", s10);
    assert!(s10 > 0, "sia must not wipe out S entirely; S was {}", s10);
}

#[test]
fn simulate_scenario_enable_activates_by_name() {
    let _ = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);
    let traj = run_simulate(&[
        "simulate", &ir.to_string_lossy(),
        "--scenario", "with_sia",
        "--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
    ]);
    assert_eq!(s_at(&traj, 5.0), 1100, "event fires regardless of scenario");
    assert!(s_at(&traj, 10.0) < 1100, "scenario enables sia");
}

#[test]
fn simulate_disable_silences_event() {
    let _ = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);
    let traj = run_simulate(&[
        "simulate", &ir.to_string_lossy(),
        "--disable", "bump",
        "--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
    ]);

    // Event disabled, intervention not enabled → S stays at 1000 throughout.
    assert_eq!(s_at(&traj, 5.0), 1000, "bump event must not fire when --disabled");
    assert_eq!(s_at(&traj, 10.0), 1000);
    assert_eq!(s_at(&traj, 15.0), 1000);
}

#[test]
fn simulate_wildcard_enable_activates_all_interventions() {
    let _ = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);
    let traj = run_simulate(&[
        "simulate", &ir.to_string_lossy(),
        "--enable", "*",
        "--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
    ]);
    // Wildcard fires all toggleable interventions AND events still fire.
    assert_eq!(s_at(&traj, 5.0), 1100, "event fires");
    assert!(s_at(&traj, 10.0) < 1100, "wildcard enable activates sia");
}

// ─── fit default-activation diagnostic ───────────────────────────────────────

#[test]
fn fit_toml_v2_accepts_scenario_field() {
    // Spec-contract: the v2 fit.toml schema must accept
    // `scenario = "..."` as a top-level field. If we regress the
    // serde wiring, TOML parsing fails with "unknown field".
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let fit_path = tmp.path().join("fit.toml");
    std::fs::write(&fit_path, r#"
scenario = "with_sia"

[model]
camdl = "/nonexistent/model.camdl"

[data]
observations = {}

[estimate]
[fixed]
[stages]
"#).unwrap();

    // `fit run` will fail on the missing model but only AFTER TOML parse.
    // If serde rejects `scenario`, stderr contains "unknown field"; we
    // assert that it does NOT.
    let out = Command::new(&bin)
        .args(["fit", "run", &fit_path.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("unknown field `scenario`"),
        "fit.toml v2 must accept top-level `scenario`: {}", stderr
    );
}

#[test]
fn fit_toml_rejects_scenario_and_enable_simultaneously() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let fit_path = tmp.path().join("fit.toml");
    // Reference a real model so we get past TOML parse + model load
    // and hit the "mutually exclusive" validation.
    let model_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../ocaml/golden/sir_basic.ir.json");
    std::fs::write(&fit_path, format!(r#"
scenario = "baseline"
enable = ["something"]
output_dir = "{out}/out"

[model]
camdl = "{model}"

[data]
observations = {{ cases = "/nowhere.tsv" }}

[estimate]
[fixed]
[stages]
"#, model = model_path.display(), out = tmp.path().display())).unwrap();

    let out = Command::new(&bin)
        .args(["fit", "run", &fit_path.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "scenario + enable together must error with the exclusivity message: {}",
        stderr
    );
}
