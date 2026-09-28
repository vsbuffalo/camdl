//! gh#692: a negative propensity is refused by EVERY forward backend with the
//! same typed error, `SimError::NegativePropensity` — the backend × property
//! cell the ODE backend used to leave open (it integrated the negative rate as a
//! negative flow, and the trajectory became a likelihood).
//!
//! Two shapes of "the rate left the model's support":
//!
//! - **At θ, from t = 0**: `pure_death` (`N → ∅` at rate `mu·N`) with `mu < 0`.
//!   Every backend evaluates the rate at the initial state, so every backend —
//!   chain-binomial, Gillespie, ODE with fixed RK4 and with adaptive DOPRI5 —
//!   must refuse the same θ.
//! - **Mid-trajectory**: the gh#208 fixture (SIR + waning; `recover` rate
//!   `gamma·I·(cap − I)/N`, which crosses zero once `I > cap = 9`). The rate is
//!   positive at t = 0 and goes negative only as the epidemic grows, so a
//!   boundary-only check would miss it. The deterministic ODE trajectory crosses
//!   on both integrators. (The stochastic backends' crossing is seed-dependent;
//!   Gillespie's is pinned in `gillespie_sparse_negative_rate.rs`.)
//!
//! Each has a negative control on the same backend set (a θ whose rates stay
//! non-negative must complete), so the refusal cannot pass vacuously.

use std::path::PathBuf;

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    error::SimError,
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 7;

fn load(rel: &str) -> ir::Model {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..").join(rel);
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

fn with_rk45(mut model: ir::Model) -> ir::Model {
    model.simulation.integrator = ir::model::Integrator::Rk45 { atol: None, rtol: None };
    model
}

fn set_param(model: &mut ir::Model, name: &str, v: f64) {
    let p = model
        .parameters
        .iter_mut()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no parameter {name}"));
    p.value = p.value.with_value(v);
}

/// Every forward backend (and both ODE integrators), each paired with the
/// model it runs — the rk45 cell is the same model with the integrator swapped.
fn every_backend(model: &ir::Model) -> Vec<(&'static str, CompiledModel, Box<dyn Simulate>, SimConfig)> {
    let (t_start, t_end) = (model.simulation.t_start, model.simulation.t_end);
    let compile = |m: ir::Model| CompiledModel::new(m).expect("compile");
    vec![
        (
            "chain_binomial",
            compile(model.clone()),
            Box::new(ChainBinomialSim),
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 0.1 }),
        ),
        (
            "gillespie",
            compile(model.clone()),
            Box::new(GillespieSim),
            SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        ),
        (
            "ode rk4",
            compile(model.clone()),
            Box::new(OdeSim),
            SimConfig::Ode(OdeConfig { t_start, t_end, dt: 0.1 }),
        ),
        (
            "ode rk45",
            compile(with_rk45(model.clone())),
            Box::new(OdeSim),
            SimConfig::Ode(OdeConfig { t_start, t_end, dt: 0.1 }),
        ),
    ]
}

fn expect_negative_propensity(
    label: &str,
    res: Result<sim::Trajectory, SimError>,
    transition: &str,
) -> f64 {
    match res {
        Err(SimError::NegativePropensity { transition: tr, value, t }) => {
            assert_eq!(tr, transition, "{label}: names the offending transition");
            assert!(value < 0.0, "{label}: carries the negative value, got {value}");
            t
        }
        Err(other) => panic!("{label}: expected NegativePropensity, got {other:?}"),
        Ok(_) => panic!(
            "{label}: completed silently — the `{transition}` rate went negative and was \
             used as a propensity instead of raising NegativePropensity (gh#692)"
        ),
    }
}

#[test]
fn negative_rate_at_theta_is_refused_by_every_backend() {
    let mut model = load("ir/golden/pure_death.ir.json");
    set_param(&mut model, "mu", -0.1);
    for (label, compiled, sim, cfg) in every_backend(&model) {
        let params = compiled.default_params.clone();
        expect_negative_propensity(label, sim.run(&compiled, &params, SEED, &cfg), "death");
    }

    // Control: the same model at a θ with a non-negative rate completes on
    // every backend.
    set_param(&mut model, "mu", 0.1);
    for (label, compiled, sim, cfg) in every_backend(&model) {
        let params = compiled.default_params.clone();
        let res = sim.run(&compiled, &params, SEED, &cfg);
        assert!(res.is_ok(), "{label}: control mu = 0.1 must complete, got {:?}", res.err());
    }
}

#[test]
fn ode_refuses_a_rate_that_goes_negative_mid_trajectory() {
    let fixture = "tests/fixtures/regression/ir/gh208_sparse_negative_rate.ir.json";
    let model = load(fixture);
    for (label, compiled, sim, cfg) in every_backend(&model) {
        if !label.starts_with("ode") {
            continue; // stochastic crossing is seed-dependent; see module doc
        }
        let params = compiled.default_params.clone();
        let t = expect_negative_propensity(label, sim.run(&compiled, &params, SEED, &cfg), "recover");
        assert!(t > model.simulation.t_start, "{label}: the crossing is mid-trajectory, got t = {t}");
    }

    // Control: with `cap` large enough that `cap − I` never goes negative, both
    // ODE integrators complete.
    let mut control = load(fixture);
    set_param(&mut control, "cap", 1000.0);
    for (label, compiled, sim, cfg) in every_backend(&control) {
        if !label.starts_with("ode") {
            continue;
        }
        let params = compiled.default_params.clone();
        let res = sim.run(&compiled, &params, SEED, &cfg);
        assert!(res.is_ok(), "{label}: control cap = 1000 must complete, got {:?}", res.err());
    }
}
