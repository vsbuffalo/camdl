//! gh#191 — the chain_binomial INFERENCE producer carries no real state and
//! never advances a reservoir, so a real-coupled (REAL_COMPARTMENTS) model
//! would be fit with its reservoir frozen at init: silently mis-fit. The
//! capability gate (`fit::methods::check_model_capabilities`) rejects it, and
//! `camdl fit`/`profile` route through that gate — but the standalone
//! `camdl pfilter` command bypassed it. This test pins that `pfilter` now gates
//! the same way (and does NOT over-reject a real-compartment-free model).
//!
//! The gate fires right after the model compiles, BEFORE `--data`/observation
//! resolution, so a params-only invocation reaches it. Shells out to the
//! release binary to exercise the full CLI path.

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
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        b.display()
    );
    b
}

fn golden(name: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join(format!("../../../ocaml/golden/{name}.ir.json"))
}

/// `sir_reservoir_mixed` has real compartments (W1..W5); `pfilter` must reject
/// it with the gh#191 REAL_COMPARTMENTS message rather than silently filtering
/// with the reservoir frozen.
#[test]
fn pfilter_rejects_real_coupled_model() {
    let bin = skip_if_missing();
    let out = Command::new(&bin)
        .args([
            "pfilter", &golden("sir_reservoir_mixed").to_string_lossy(),
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "xi=0.05",
            "--param", "delta=0.1", "--param", "N0=1000", "--param", "I0=10",
            "--particles", "100", "--seed", "1",
        ])
        .output()
        .expect("spawn camdl pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "pfilter must reject a real-coupled model; stdout/stderr={}{}",
        String::from_utf8_lossy(&out.stdout), stderr,
    );
    assert!(
        stderr.contains("gh#191"),
        "rejection must cite the tracking issue; stderr={stderr}"
    );
    assert!(
        stderr.contains("REAL_COMPARTMENTS"),
        "rejection must name the capability; stderr={stderr}"
    );
}

/// Control: `sir_basic` has no real compartments, so the gh#191 gate must NOT
/// fire. (The invocation still errors later — no `--data`/observations — but
/// that error must not be the REAL_COMPARTMENTS rejection.)
#[test]
fn pfilter_does_not_over_reject_real_free_model() {
    let bin = skip_if_missing();
    let out = Command::new(&bin)
        .args([
            "pfilter", &golden("sir_basic").to_string_lossy(),
            "--param", "beta=0.3", "--param", "gamma=0.1",
            "--param", "N0=1000", "--param", "I0=10",
            "--particles", "100", "--seed", "1",
        ])
        .output()
        .expect("spawn camdl pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("gh#191"),
        "a real-compartment-free model must not trip the gh#191 gate; stderr={stderr}"
    );
}
