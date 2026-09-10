//! gh#67 — events { add(C, n) at [t] } must fire under every backend, not just
//! chain_binomial. Before this commit: ODE and gillespie silently dropped
//! events because they call `apply_interventions_at` (which skips
//! `always_active` items per the `if iv.always_active { continue }` guard in
//! `sim/intervention.rs:99`) but never call the sister `inject_event_deltas`.
//! chain_binomial alone wires `inject_event_deltas` into its per-substep loop
//! (chain_binomial.rs:417), so events fire correctly there.
//!
//! This test runs a small SIR seeded with I=1 (non-absorbing from t=0) plus a
//! single event `add(I, 100) at [10]`. It samples I just before and just after
//! the scheduled event and asserts that the post-event count jumped by at least
//! the event payload. Pre-fix the three broken backends never apply the +100, so
//! the jump is purely the transition dynamics and far below 100; the test fails
//! on those.
//!
//! (The absorbing-from-t=0 + scheduled-event case this test used to step around
//! — gh#70, gillespie back-filling the event into pre-event output rows — is now
//! fixed and asserted directly in
//! `sim/tests/cross_backend_lifecycle_agreement.rs::full_trajectory_no_pre_event_leak_or_time_reversal`.)

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn tempdir() -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_evt_parity_{}_{}", std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

const MODEL_SRC: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta : rate in [1e-6, 5.0]
  gamma : rate in [1e-6, 1.0]
  N0 : count in [100, 10000000]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
events { booster : add(I, 100) at [10] }
init { S = N0 - 1  I = 1 }
simulate { from = 0  to = 12 }
"#;

/// Read column `col_name` at the row whose `t` equals `target_t` (within 1e-6).
fn col_at(traj_path: &Path, col_name: &str, target_t: f64) -> Option<f64> {
    let text = std::fs::read_to_string(traj_path).ok()?;
    let mut header: Vec<&str> = vec![];
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() { continue; }
        let cols: Vec<&str> = line.split('\t').collect();
        if header.is_empty() { header = cols; continue; }
        let t: f64 = cols[0].parse().ok()?;
        if (t - target_t).abs() < 1e-6 {
            let idx = header.iter().position(|h| *h == col_name)?;
            return cols[idx].parse().ok();
        }
    }
    None
}

fn run_simulate(camdl: &Path, model: &Path, backend: &str, traj: &Path) {
    let out = Command::new(camdl)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", model.to_str().unwrap(),
            "--backend", backend, "--dt", "1", "--seed", "0",
            // Very slow dynamics: per-day infection rate ≈ 0.0001 per
            // infected, so the I count drifts by O(1) over t∈[0,12].
            // The event payload (+100) is then unmistakable.
            "--param", "beta=0.0001", "--param", "gamma=0.0001", "--param", "N0=10000000",
            "-o", traj.to_str().unwrap(),
        ])
        .output().expect("camdl must invoke");
    assert!(out.status.success(),
        "simulate ({backend}) failed: {}", String::from_utf8_lossy(&out.stderr));
}

/// gh#67: every backend must fire the `add(I, 100) at [10]` event. Pre-fix,
/// ode/gillespie never apply the +100 so I stays in the single
/// digits (slow dynamics) across the entire run.
#[test]
fn events_fire_under_every_backend() {
    let camdl = camdl_bin();
    let tmp = tempdir();
    let model = tmp.join("model.camdl");
    std::fs::write(&model, MODEL_SRC).unwrap();

    for backend in &["chain_binomial", "ode", "gillespie"] {
        let traj = tmp.join(format!("{}.tsv", backend));
        run_simulate(&camdl, &model, backend, &traj);

        let i_pre  = col_at(&traj, "I",  9.0).expect("row t=9 has I");
        let i_post = col_at(&traj, "I", 11.0).expect("row t=11 has I");
        let jump = i_post - i_pre;

        // Pre-fix jump for ode/gillespie is ≈ 0 (slow dynamics,
        // event silently dropped). Post-fix jump ≥ 100 (the event payload).
        assert!(jump >= 90.0,
            "{backend}: event `add(I, 100) at [10]` should jump I by ≥100 \
             between t=9 and t=11 (saw {pre} → {post}, jump={j}) — \
             gh#67: event silently dropped",
            pre = i_pre, post = i_post, j = jump);
    }
}
