//! Wiring tests for the two intervention-value guards (Tier 1):
//!
//! - **D-finite**: an action whose resolved value is non-finite must error
//!   before the silent `f64 as i64` cast (`NaN -> 0`, `+inf -> i64::MAX`).
//!   The rate (propensity) path is finite-guarded; the intervention-amount
//!   path was not. `intervention_nonfinite` drives a `set V = c/z` (z=0 ->
//!   NaN); on the pre-guard code V silently becomes 0 and the run succeeds.
//!
//! - **D-negative**: a `set` to a value below zero must be caught by the
//!   centralized post-INTERVENE/BALANCE scan. The pre-advance negativity scan
//!   runs *before* INTERVENE and misses it. `intervention_set_negative` drives
//!   a `set V = -5`; V is in no rate, so on the pre-guard code it silently
//!   sits at -5 and the run succeeds.
//!
//! Both fixtures keep V out of every transition rate so that, before the
//! guards, the corrupted value cannot trip an *unrelated* error (negative
//! propensity / overflow) — the run genuinely succeeds, so the `is_err()`
//! assertion is a true red against the pre-guard code, not a symptom match.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

fn load_fixture(name: &str) -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = PathBuf::from(&manifest).join(format!("tests/fixtures/{name}.ir.json"));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));
    ir::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"))
}

fn run_chain(model: &ir::Model) -> Result<sim::state::Trajectory, sim::SimError> {
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 1.0,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    ChainBinomialSim.run(&compiled, &params, 42, &config)
}

/// D-finite: `set V = c/z` (z=0) resolves to NaN. The finite guard must turn
/// this into an error before the cast. Pre-guard, the run succeeds (V casts to
/// 0). The message names the cause and the offending action.
#[test]
fn scheduled_set_to_non_finite_value_errors() {
    let model = load_fixture("intervention_nonfinite");
    let err = run_chain(&model)
        .expect_err("a non-finite intervention value must error, not cast silently to 0");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "error should name the non-finite cause; got: {msg}"
    );
    assert!(
        msg.contains("bad_set") && msg.contains("set V"),
        "error should name the intervention and action; got: {msg}"
    );
}

/// D-negative: `set V = -5` leaves V negative after INTERVENE. The centralized
/// post-INTERVENE scan must reject it. Pre-guard, the run succeeds with V=-5.
#[test]
fn scheduled_set_to_negative_value_errors() {
    let model = load_fixture("intervention_set_negative");
    let err = run_chain(&model)
        .expect_err("a set leaving a non-balance compartment negative must error");
    match err {
        sim::SimError::NegativeCount { compartment, attempted_value, cause, .. } => {
            assert_eq!(compartment, "V", "should point at the compartment set negative");
            assert_eq!(attempted_value, -5);
            assert_eq!(cause, sim::NegativeCountCause::InterventionNegative);
        }
        other => panic!("expected NegativeCount{{InterventionNegative}}, got: {other}"),
    }
}
