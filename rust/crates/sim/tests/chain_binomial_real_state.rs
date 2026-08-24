//! Regression test for the chain-binomial stale-real-state incident
//! (docs/dev/incidents/2026-06-07-chain-binomial-stale-real-state.md).
//!
//! Bug: `step_one` synced `scratch.int_s` from the run's integer counts but
//! never synced `scratch.real_s` from the run's real compartment state. Rates
//! are evaluated against `scratch.real_s`, which stayed at its zero init, so
//! any integer transition whose rate couples to a real compartment computed
//! that rate with the real value identically 0 — silently wrong on the
//! chain-binomial backend (the only inference kernel).
//!
//! The fixture `real_coupled_rate.ir.json` has an integer transition
//! `S --> I` with rate `beta * (W/(W+kappa)) * S`, where `W` is a real
//! compartment held constant by `dW/dt = 0`. With `W` large the saturation
//! term `W/(W+kappa)` is ~1 (rate ~ beta*S, infections happen); with `W = 0`
//! the rate is exactly 0 (no infections). A correct backend produces
//! different integer dynamics for `W>0` vs `W=0`. A backend that ignores the
//! real state evaluates `W == 0` regardless and produces *identical* output —
//! that is the bug this test pins.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

fn fixture_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("tests/fixtures")
}

fn load_fixture(name: &str) -> ir::Model {
    let path = fixture_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e))
}

/// Override the initial value of compartment `comp` (real-valued) in the model.
///
/// The fixture must seed `comp` with a literal: overwriting an expression that
/// reads a parameter or another compartment would change what the test compares,
/// so refuse rather than silently rewrite it.
fn set_w_init(model: &mut ir::Model, comp: &str, value: f64) {
    use ir::expr::Expr;
    use ir::model::InitSpec;
    match model.initial_conditions.0.get(comp) {
        Some(InitSpec::Deterministic(Expr::Const(_))) => {}
        other => panic!("fixture must seed `{comp}` with a constant, got {other:?}"),
    }
    model
        .initial_conditions
        .0
        .insert(comp.to_string(), InitSpec::Deterministic(Expr::const_(value)));
}

fn cb_run(model: &ir::Model, w_init: f64) -> sim::state::Trajectory {
    let mut model = model.clone();
    set_w_init(&mut model, "W", w_init);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 1.0,
    });
    ChainBinomialSim.run(&compiled, &params, 42, &config).unwrap()
}

/// Total infections fired over the run = N0_S - S(end), since the only
/// transition that removes S is `infection`.
fn total_infections(traj: &sim::state::Trajectory) -> i64 {
    let s0 = traj.snapshots.first().unwrap().int_state.counts[0];
    let s_end = traj.snapshots.last().unwrap().int_state.counts[0];
    s0 - s_end
}

/// RED-FIRST: chain-binomial dynamics MUST depend on the real compartment W
/// that feeds the infection rate. With W=0 the rate is identically 0 (no
/// infections); with W large the rate is ~beta*S (many infections). A correct
/// backend gives different trajectories. The buggy backend reads W==0 always,
/// so both runs are identical — this assertion fails on the buggy code.
#[test]
fn chain_binomial_rate_couples_to_real_compartment() {
    let model = load_fixture("real_coupled_rate");

    // W held at 0: rate = beta * 0/(0+kappa) * S = 0 → zero infections.
    let traj_w0 = cb_run(&model, 0.0);
    // W held large: rate ≈ beta * S → many infections.
    let traj_wbig = cb_run(&model, 1.0e6);

    let inf_w0 = total_infections(&traj_w0);
    let inf_wbig = total_infections(&traj_wbig);

    // Hand-checkable anchor: with W=0 the rate is exactly 0, so chain-binomial
    // must fire ZERO infections regardless of the bug — this holds either way.
    assert_eq!(
        inf_w0, 0,
        "W=0 forces rate 0 → expected 0 infections, got {inf_w0}"
    );

    // The diagnostic assertion: with W large the saturation term ~1, so the
    // rate is ~beta*S and a CORRECT backend fires many infections. The buggy
    // backend reads W==0 in the rate → still 0 infections → identical to the
    // W=0 run. This is the line that fails RED on the stale-real-state bug.
    assert!(
        inf_wbig > 0,
        "chain-binomial ignored the real compartment W in the infection rate: \
         W=1e6 should drive ~beta*S infections, but fired {inf_wbig} \
         (identical to the W=0 run, inf_w0={inf_w0}). This is the \
         stale-real-state bug: step_one never synced scratch.real_s."
    );
}

/// Cross-backend check (approach b): with W held at a constant nonzero value,
/// the per-capita infection rate is constant (W/(W+kappa) is fixed), so all
/// three backends model the same constant-hazard S→I process. They should
/// agree on the *direction and rough magnitude* of depletion. The buggy
/// chain-binomial is the outlier: it fires zero infections while ODE and
/// Gillespie deplete S substantially.
#[test]
fn chain_binomial_agrees_with_ode_and_gillespie_on_real_coupling() {
    let mut model = load_fixture("real_coupled_rate");
    set_w_init(&mut model, "W", 1.0e6); // saturation term ~1, rate ~beta*S
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end = model.simulation.t_end;

    // ODE: deterministic reference. dS/dt = -beta*(W/(W+kappa))*S, so S decays
    // exponentially with rate ~beta=0.5. After 20 days S → ~1000*exp(-10) ≈ 0.
    let ode = OdeSim
        .run(
            &compiled,
            &params,
            0,
            &SimConfig::Ode(OdeConfig { t_start, t_end, dt: 0.5 }),
        )
        .unwrap();
    let ode_s_end = ode.snapshots.last().unwrap().int_state.counts[0];

    // Gillespie: exact stochastic. Also depletes S nearly fully.
    let gil = GillespieSim
        .run(
            &compiled,
            &params,
            42,
            &SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        )
        .unwrap();
    let gil_s_end = gil.snapshots.last().unwrap().int_state.counts[0];

    let cb = ChainBinomialSim
        .run(
            &compiled,
            &params,
            42,
            &SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 }),
        )
        .unwrap();
    let cb_s_end = cb.snapshots.last().unwrap().int_state.counts[0];

    // ODE and Gillespie both deplete S strongly (S_end well below 100 of 1000).
    assert!(ode_s_end < 100, "ODE should deplete S (got S_end={ode_s_end})");
    assert!(gil_s_end < 100, "Gillespie should deplete S (got S_end={gil_s_end})");

    // The bug makes chain-binomial the outlier — it leaves S at the full 1000
    // because it reads W==0 (rate 0). A correct backend also depletes S.
    assert!(
        cb_s_end < 100,
        "chain-binomial did NOT deplete S (S_end={cb_s_end}) while ODE \
         (S_end={ode_s_end}) and Gillespie (S_end={gil_s_end}) did. \
         chain-binomial ignored the real compartment W coupling in the rate."
    );
}
