//! Tests for intervention application, including BUG-5: FractionTransfer uses floor not round.

use std::collections::HashMap;
use ir::{
    expr::{ConstExpr, Expr},
    intervention::{Action, FractionTransfer, Intervention, InterventionSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    Model,
    parameter::Parameter,
};
use sim::{
    compiled_model::CompiledModel,
    effects::due_effects,
    intervention::apply_effect_batch,
    schedule::EffectBatch,
    state::{IntState, RealState},
};

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

/// Test adapter preserving the old `apply_interventions_at` call shape: derive
/// the due batch at `t` (the seam `due_effects` now owns) and apply the
/// scheduled interventions. Byte-identical to the retired entry point.
#[allow(clippy::too_many_arguments)]
fn apply_interventions_at(
    t: f64,
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    dt: f64,
    grid_dt: f64,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
    _tolerance: f64,
) -> Result<bool, sim::error::SimError> {
    let mut batch = EffectBatch::default();
    due_effects(model, fire_steps, t, grid_dt, &mut batch);
    apply_effect_batch(t, model, &batch.intervention_idx, dt, int_s, real_s, params)
}

fn minimal_model_with_interventions(
    compartments: Vec<Compartment>,
    params: Vec<Parameter>,
    interventions: Vec<Intervention>,
) -> Model {
    Model {
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments,
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions,
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: params,
        initial_conditions: InitialConditions::Parameterized(HashMap::new()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![],
    }
}

/// BUG-5: FractionTransfer should use floor, not round.
/// source=1, fraction=0.6: floor(0.6) = 0, round(0.6) = 1.
/// This test ensures we transfer 0 (floor), not 1 (round).
#[test]
fn test_fraction_transfer_uses_floor_not_round() {
    let intervention = Intervention {
        name: "test_iv".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![30.0])),
        kind: ir::intervention::InterventionKind::Scenario, actions: vec![
            Action::FractionTransfer(FractionTransfer {
                src: "S".into(),
                dst: "V".into(),
                fraction: Expr::Const(ConstExpr { value: 0.6 }),
            }),
        ],
    };

    let model = CompiledModel::new(minimal_model_with_interventions(
        vec![int_comp("S"), int_comp("V")],
        vec![],
        vec![intervention],
    )).unwrap();

    let mut int_s = IntState::from_vec(vec![1, 0]); // S=1, V=0
    let mut real_s = RealState::new(0);

    let fire_steps = model.resolve_fire_steps(1.0, &[]);
    apply_interventions_at(30.0, &model, &fire_steps, 1.0, 1.0, &mut int_s, &mut real_s, &[], 1e-10).unwrap();

    // floor(1 * 0.6) = floor(0.6) = 0: no transfer should happen
    assert_eq!(int_s.counts[0], 1, "S should remain 1 (no transfer)");
    assert_eq!(int_s.counts[1], 0, "V should remain 0 (no transfer)");
}

/// Sanity check: fraction=0.8, source=5 → floor(5*0.8) = floor(4.0) = 4 transferred.
#[test]
fn test_fraction_transfer_floor_larger() {
    let intervention = Intervention {
        name: "test_iv".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![30.0])),
        kind: ir::intervention::InterventionKind::Scenario, actions: vec![
            Action::FractionTransfer(FractionTransfer {
                src: "S".into(),
                dst: "V".into(),
                fraction: Expr::Const(ConstExpr { value: 0.8 }),
            }),
        ],
    };

    let model = CompiledModel::new(minimal_model_with_interventions(
        vec![int_comp("S"), int_comp("V")],
        vec![],
        vec![intervention],
    )).unwrap();

    let mut int_s = IntState::from_vec(vec![5, 0]); // S=5, V=0
    let mut real_s = RealState::new(0);

    let fire_steps = model.resolve_fire_steps(1.0, &[]);
    apply_interventions_at(30.0, &model, &fire_steps, 1.0, 1.0, &mut int_s, &mut real_s, &[], 1e-10).unwrap();

    // floor(5 * 0.8) = floor(4.0) = 4
    assert_eq!(int_s.counts[0], 1, "S should be 1 after 4 transferred");
    assert_eq!(int_s.counts[1], 4, "V should be 4 after transfer");
}

/// Regression: chain_binomial used to fire scheduled interventions twice
/// per firing time — once in `step_one` at t+dt, once in
/// `run_chain_binomial` after `t += dt`. Two fires means retain fraction
/// is (1-f)², so a 50% transfer on S=1000 left S=250 instead of 500.
/// See docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md.
#[test]
fn chain_binomial_fires_scheduled_intervention_exactly_once() {
    use sim::{
        config::{ChainBinomialConfig, SimConfig},
        simulate::Simulate,
        ChainBinomialSim,
    };
    use ir::intervention::InterventionSchedule;

    let intervention = Intervention {
        name: "sia".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![10.0])),
        kind: ir::intervention::InterventionKind::Scenario,
        actions: vec![Action::FractionTransfer(FractionTransfer {
            src: "S".into(),
            dst: "V".into(),
            fraction: Expr::Const(ConstExpr { value: 0.5 }),
        })],
    };

    // Build a model with a known starting state (S=1000, V=0) and
    // outputs at every integer t so we can observe the post-firing state.
    let mut init = HashMap::new();
    init.insert("S".to_string(), 1000.0);
    init.insert("V".to_string(), 0.0);
    let model = Model {
        name: "double_fire_regression".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None, origin: None, origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("V")],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![],
        initial_conditions: InitialConditions::Explicit(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes((0..=20).map(|t| t as f64).collect()),
            format: "tsv".into(),
            trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 20.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
            integrator: Default::default(),
        },
        interventions: vec![intervention],
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![],
    };
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: 0.0, t_end: 20.0, dt: 1.0,
    });
    let traj = ChainBinomialSim.run(&compiled, &params, 42, &cfg).unwrap();

    let s_after = traj.snapshots.iter()
        .find(|s| s.t >= 15.0)
        .expect("trajectory should reach t=15")
        .int_state.counts[0];
    assert_eq!(s_after, 500,
        "one 50% fractional transfer must leave S=500; double-firing would \
         give S=250 (got S={})", s_after);
}
