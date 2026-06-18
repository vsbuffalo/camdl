//! gh#204 PR1 runtime contract: a reactive intervention is a policy
//! (`kind = Scenario`), inactive by default. The runtime accepts a run where the
//! reactive policy is NOT enabled (it is dropped, like any toggleable
//! intervention), and rejects a run where it IS active with the
//! `REACTIVE_INTERVENTIONS` capability error — no backend executes the agenda
//! yet.
//!
//! Uses the COMMITTED compiler-generated golden
//! (`tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json`, kept
//! in sync by `make check-reactive-golden`) rather than a hand-written IR
//! literal, so the rejection exercises the real IR shape. Params are supplied on
//! the CLI (the golden leaves them estimated).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn golden_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// The golden's estimated parameters, pinned on the CLI so the model resolves.
const PARAMS: &[&str] = &[
    "--param", "beta=0.3",
    "--param", "gamma=0.1",
    "--param", "rho=0.2",
    "--param", "trigger_threshold=2",
    "--param", "sia_cov=0.7",
    "--param", "N0=1000",
    "--param", "I0=10",
];

fn run(extra: &[&str]) -> std::process::Output {
    let bin = skip_if_missing_binary();
    let ir = golden_ir();
    let tmp = tempfile::tempdir().unwrap();
    let traj = tmp.path().join("traj.tsv");
    let cas_dir = tmp.path().join("results");
    let mut args: Vec<String> = vec!["simulate".into(), ir.to_string_lossy().into_owned()];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    args.extend(
        ["--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
         "-o", &traj.to_string_lossy(), "--output-dir", &cas_dir.to_string_lossy()]
            .iter()
            .map(|s| s.to_string()),
    );
    Command::new(&bin).args(&args).output().expect("spawn camdl")
}

/// Dormant reactive policy (not enabled) → the run is accepted; the policy is
/// dropped like any toggleable intervention.
#[test]
fn reactive_inactive_run_is_accepted() {
    let _ = skip_if_missing_binary();
    let out = run(&[]);
    assert!(
        out.status.success(),
        "a run that does not enable the reactive policy must be accepted; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--enable sia` activates the reactive policy → rejected at dispatch with the
/// capability error (no backend runs the agenda yet).
#[test]
fn reactive_active_via_enable_is_rejected() {
    let _ = skip_if_missing_binary();
    let out = run(&["--enable", "sia"]);
    assert!(!out.status.success(), "an enabled reactive policy must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("REACTIVE_INTERVENTIONS"),
        "rejection must name the capability; stderr={stderr}"
    );
}
