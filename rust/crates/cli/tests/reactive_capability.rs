//! gh#204 runtime contract for reactive interventions.
//!
//! PR1 represented + rejected them everywhere. PR2 makes **forward
//! chain-binomial** run the reactive agenda, so:
//!   - inactive (not enabled) → runs the baseline (policy dropped, like any
//!     toggleable intervention);
//!   - active on chain-binomial → RUNS (the agenda fires the policy);
//!   - active on gillespie/ode (forward) → still rejected with the
//!     `REACTIVE_INTERVENTIONS` capability error (PR3);
//!   - the inference path is covered separately (fit/methods.rs) — still
//!     withheld there (PR4).
//!
//! Uses the COMMITTED compiler-generated golden (kept in sync by
//! `make check-reactive-golden`) with params supplied on the CLI.

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

fn run(backend: &str, extra: &[&str]) -> std::process::Output {
    let bin = skip_if_missing_binary();
    let ir = golden_ir();
    let tmp = tempfile::tempdir().unwrap();
    let traj = tmp.path().join("traj.tsv");
    let cas_dir = tmp.path().join("results");
    let mut args: Vec<String> = vec!["simulate".into(), ir.to_string_lossy().into_owned()];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    args.extend(
        ["--seed", "1", "--backend", backend, "--dt", "1.0",
         "-o", &traj.to_string_lossy(), "--output-dir", &cas_dir.to_string_lossy()]
            .iter()
            .map(|s| s.to_string()),
    );
    Command::new(&bin).args(&args).output().expect("spawn camdl")
}

/// Dormant reactive policy (not enabled) → accepted; dropped like any
/// toggleable intervention.
#[test]
fn reactive_inactive_run_is_accepted() {
    let _ = skip_if_missing_binary();
    let out = run("chain_binomial", &[]);
    assert!(
        out.status.success(),
        "a run that does not enable the reactive policy must be accepted; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--enable sia` on chain-binomial → the reactive agenda RUNS the policy (PR2).
#[test]
fn reactive_active_on_chain_binomial_runs() {
    let _ = skip_if_missing_binary();
    let out = run("chain_binomial", &["--enable", "sia"]);
    assert!(
        out.status.success(),
        "forward chain-binomial runs the reactive agenda; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--enable sia` on gillespie → still rejected (no reactive agenda there yet).
#[test]
fn reactive_active_on_gillespie_is_rejected() {
    let _ = skip_if_missing_binary();
    let out = run("gillespie", &["--enable", "sia"]);
    assert!(!out.status.success(), "gillespie does not run reactive policies yet");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("REACTIVE_INTERVENTIONS"),
        "rejection must name the capability; stderr={stderr}"
    );
}
