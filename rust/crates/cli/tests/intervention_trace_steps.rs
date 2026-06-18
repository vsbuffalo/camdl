//! gh#106: `CAMDL_TRACE_STEPS=1` must trace *every* intervention/event
//! action arm, not just `Action::Add`. Before the fix only the Add arm
//! emitted a trace line; FractionTransfer / AbsoluteTransfer / Set (in
//! `inject_event_deltas`) and the whole `apply_intervention` path were
//! silent.
//!
//! These tests shell out to the built `camdl` binary and capture stderr
//! (the trace channel). The control case asserts that without the env
//! var no trace lines are emitted at all.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// A model exercising two previously-silent arms:
///   - an `always_active` event `pin` that `set`s S to 900 at t=3
///     (routes through `inject_event_deltas` → Action::Set)
///   - a toggleable intervention `sia` that `fraction_transfer`s 50%
///     of S → V at t=10 (routes through `apply_intervention` →
///     Action::FractionTransfer; fires only when `--enable`'d)
/// No transitions, so state moves only via these actions.
fn trace_model_ir() -> String {
    // `__IR_VERSION__` → the build's IR_VERSION (envelope-checked on load), so a
    // schema bump never staleness-breaks this fixture.
    r#"{
      "ir_version": "__IR_VERSION__",
      "validated_by": "test-fixture",
      "model": {
        "name": "trace_arms", "version": "0.3", "time_unit": "days",
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
        "scenarios": [],
        "interventions": [
          {
            "name": "pin",
            "fire": { "scheduled": { "at_times": [3.0] } },
            "actions": [ { "set": {
              "compartment": "S", "value": { "const": 900.0 }
            } } ],
            "kind": "event"
          },
          {
            "name": "sia",
            "fire": { "scheduled": { "at_times": [10.0] } },
            "actions": [ { "fraction_transfer": {
              "src": "S", "dst": "V", "fraction": { "const": 0.5 }
            } } ],
            "kind": "scenario"
          }
        ],
        "model_structure": null, "balance": null
      }
    }"#
    .replace("__IR_VERSION__", ir::IR_VERSION.trim())
}

fn write_ir(tmp: &tempfile::TempDir) -> PathBuf {
    let path = tmp.path().join("model.ir.json");
    std::fs::write(&path, trace_model_ir()).unwrap();
    path
}

/// Run `camdl simulate`, returning captured stderr. Trace output goes to
/// stderr; stdout carries nothing relevant here (trajectory is written to
/// the -o mirror / CAS store). `trace` toggles `CAMDL_TRACE_STEPS=1`.
fn run_capture_stderr(args: &[&str], trace: bool) -> String {
    let bin = binary();
    assert!(
        bin.exists(),
        "release binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    let tmp = tempfile::tempdir().unwrap();
    let traj_path = tmp.path().join("traj.tsv");
    let cas_dir = tmp.path().join("results");
    let mut cmd = Command::new(&bin);
    cmd.args(args)
        .args([
            "-o",
            &traj_path.to_string_lossy(),
            "--output-dir",
            &cas_dir.to_string_lossy(),
        ]);
    if trace {
        cmd.env("CAMDL_TRACE_STEPS", "1");
    } else {
        cmd.env_remove("CAMDL_TRACE_STEPS");
    }
    let out = cmd.output().expect("spawn");
    if !out.status.success() {
        panic!(
            "simulate failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn trace_steps_emits_set_event_and_fraction_transfer_intervention() {
    let bin = binary();
    if !bin.exists() {
        panic!(
            "release binary missing: {} — run `make build-rust` or `make test`",
            bin.display()
        );
    }
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);

    let stderr = run_capture_stderr(
        &[
            "simulate",
            &ir.to_string_lossy(),
            "--enable",
            "sia",
            "--seed",
            "1",
            "--backend",
            "chain_binomial",
            "--dt",
            "1.0",
        ],
        true,
    );

    // Set arm (inject_event_deltas), always_active event `pin` at t=3.
    assert!(
        stderr.contains("EVENT 'pin'") && stderr.contains("set S ="),
        "Set arm must emit a trace line under CAMDL_TRACE_STEPS=1; stderr:\n{}",
        stderr
    );
    // FractionTransfer arm (apply_intervention), toggleable `sia` at t=10.
    assert!(
        stderr.contains("INTERVENTION 'sia'") && stderr.contains("transfer S -> V"),
        "FractionTransfer arm must emit a trace line under CAMDL_TRACE_STEPS=1; stderr:\n{}",
        stderr
    );
}

#[test]
fn no_trace_steps_emits_no_intervention_trace() {
    let bin = binary();
    if !bin.exists() {
        panic!(
            "release binary missing: {} — run `make build-rust` or `make test`",
            bin.display()
        );
    }
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_ir(&tmp);

    // Negative control: same run, no CAMDL_TRACE_STEPS — no trace lines.
    let stderr = run_capture_stderr(
        &[
            "simulate",
            &ir.to_string_lossy(),
            "--enable",
            "sia",
            "--seed",
            "1",
            "--backend",
            "chain_binomial",
            "--dt",
            "1.0",
        ],
        false,
    );
    assert!(
        !stderr.contains("EVENT 'pin'"),
        "no Set trace without the env var; stderr:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("INTERVENTION 'sia'"),
        "no FractionTransfer trace without the env var; stderr:\n{}",
        stderr
    );
}
