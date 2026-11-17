//! T4 — evolving real-reservoir coupling on chain_binomial and gillespie (the
//! stochastic backends with behavioral oracle coverage for the RK4-real path),
//! with ODE as the deterministic reference.
//!
//! Model: `cholera_siwr` golden IR. W is a REAL compartment integrated by
//! `dW/dt = xi·I − omega_W·W`, and W feeds back into the infection rate via the
//! environmental term `beta_W·W / (W + kappa)`:
//!
//!   infection rate = S · ( beta_I·I/(S+I+R) + beta_W·W/(W+kappa) )
//!
//! The two facts we pin per stochastic backend:
//!   (1) W EVOLVES — its peak rises far above the init (W_init = 0) and its final
//!       value is well above 0 — proving the RK4 real-reservoir integration
//!       actually ran on that backend (not held at init), and
//!   (2) S DEPLETES substantially — the W-coupled infection fired.
//!
//! NEGATIVE-CONTROL REASONING (why (1)∧(2) is a discriminator, not two unrelated
//! facts): if the W-coupling were broken (W stuck at 0), the environmental term
//! beta_W·W/(W+kappa) would be 0, and S's depletion would depend ONLY on the weak
//! direct term beta_I·I/(S+I+R) with I=10 init — the outbreak would barely move.
//! Observing W rise into the hundreds AND S collapse from 990 to a few dozen is
//! the joint signature that the real reservoir both evolved and drove the rate.
//!
//! Thresholds are grounded in measured values at seed 42 (W peaks ~735–899 across
//! backends; W_end ~124–172; S 990 → ~46–52; ODE settles W_end ≈ 140, S ≈ 52).
//! They are loose multiples of those so the gate is robust to RNG, not a hash.

use std::path::PathBuf;
use std::sync::Arc;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;
const S0: i64 = 990; // cholera_siwr init S

fn load() -> CompiledModel {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ir/golden/cholera_siwr.ir.json");
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    let model: ir::Model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse cholera_siwr: {}", e));
    CompiledModel::new(model).unwrap()
}

/// (W_max over the trajectory, W at the final snapshot, S at the final snapshot).
fn summarize(compiled: &CompiledModel, traj: &Trajectory) -> (f64, f64, i64) {
    let w_idx = compiled.global_to_real[compiled.comp_index["W"]].expect("W is real");
    let s_idx = compiled.global_to_int[compiled.comp_index["S"]].expect("S is integer");
    let w_max = traj.snapshots.iter()
        .map(|s| s.real_state.values[w_idx])
        .fold(0.0_f64, f64::max);
    let last = traj.snapshots.last().expect("non-empty trajectory");
    (w_max, last.real_state.values[w_idx], last.int_state.counts[s_idx])
}

/// Assert the evolving-real-coupling signature on one stochastic backend.
fn assert_real_coupling_evolved(backend: &str, cfg: SimConfig) {
    let compiled = Arc::new(load());
    let t_end = compiled.model.simulation.t_end;
    assert_eq!(t_end, 365.0, "fixture t_end drifted; thresholds assume 365");

    let sim: &dyn Simulate = match backend {
        "gillespie" => &GillespieSim,
        "chain_binomial" => &ChainBinomialSim,
        other => panic!("unexpected backend {other}"),
    };
    let traj = sim.run(&compiled, &compiled.default_params, SEED, &cfg).unwrap();
    let (w_max, w_end, s_end) = summarize(&compiled, &traj);

    // (1) W evolved: it rose far above its init (0) — RK4 real integration ran.
    //     Measured W_max ≥ 735 across backends; require > 300 (loose, RNG-robust).
    assert!(
        w_max > 300.0,
        "{backend}: W must evolve well above its init 0 (RK4 real integration); W_max={w_max:.2}"
    );
    //     And it is still well above 0 at the final snapshot (not snapped back to
    //     init). Measured W_end ~124–172; require > 50.
    assert!(
        w_end > 50.0,
        "{backend}: W_end must remain well above 0 (W evolves, not held at init); W_end={w_end:.2}"
    );

    // (2) S depleted substantially — the W-coupled infection fired. Measured
    //     S_end ~46–52 (from 990); require < S0/3 (≈330), a loose bound that the
    //     weak direct-only pathway (W≡0) could not produce.
    assert!(
        s_end < S0 / 3,
        "{backend}: S must deplete substantially from {S0} (W-coupled infection fired); S_end={s_end}"
    );
}

#[test]
fn chain_binomial_evolves_real_reservoir_and_couples() {
    assert_real_coupling_evolved(
        "chain_binomial",
        SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 365.0, dt: 0.5, binomial: sim::rng::BinomialAlgorithm::Btpe }),
    );
}

#[test]
fn gillespie_evolves_real_reservoir_and_couples() {
    assert_real_coupling_evolved(
        "gillespie",
        SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: 365.0, output_dt: None }),
    );
}

/// ODE is the deterministic reference (no RNG): W and S settle to known
/// ballparks (measured W_end ≈ 140.4, S_end ≈ 52 at this config). Cross-checking
/// the deterministic backend anchors the loose stochastic thresholds above to a
/// hand-verified target — the stochastic runs must land in the same regime, not
/// a different attractor.
#[test]
fn ode_reference_settles_in_expected_ballpark() {
    let compiled = Arc::new(load());
    let traj = OdeSim
        .run(&compiled, &compiled.default_params, SEED,
             &SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 365.0, dt: 0.5 }))
        .unwrap();
    let (w_max, w_end, s_end) = summarize(&compiled, &traj);

    // W peaked into the hundreds and settled around ~140 (measured 140.39).
    assert!(w_max > 300.0, "ODE W_max should peak into the hundreds; got {w_max:.2}");
    assert!(
        (w_end - 140.0).abs() < 40.0,
        "ODE W_end should settle ≈ 140 (deterministic reference); got {w_end:.2}"
    );
    // S collapsed from 990 to ~52 (deterministic), the regime the stochastic
    // backends must also reach.
    assert!(
        s_end < S0 / 3,
        "ODE S should collapse from {S0} (W-coupled outbreak); got {s_end}"
    );
}
