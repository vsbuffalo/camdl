//! Guard for the real-compartment INTERVENTION branch through the effect-seam
//! refactor (step 2). `apply_intervention` historically handled real (f64)
//! compartments (`global_to_real`), but no golden or corner-case fixture ever
//! exercised it — so routing interventions through the shared `resolve_effects`
//! seam had no coverage of the real arena. This pins it: a `set` on a real
//! reservoir lands exactly (no rounding), end-to-end through the ODE backend.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    simulate::Simulate,
    OdeSim,
};
use ir::{
    expr::Expr,
    intervention::{Action, Intervention, InterventionSchedule, SetAction},
};

fn load_real_coupled() -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = PathBuf::from(&manifest).join("tests/fixtures/real_coupled_rate.ir.json");
    let contents = std::fs::read_to_string(&path).unwrap();
    ir::from_str(&contents).unwrap()
}

/// `set W = 42.5` (W is a real compartment, local real index 0) at t=2. The
/// real arena applies the exact f64 — a correct backend reports W = 42.5, not a
/// rounded value. Guards the resolver's real-Set branch + the apply wiring.
#[test]
fn intervention_set_on_real_compartment_is_exact() {
    let mut model = load_real_coupled();
    model.interventions.push(Intervention {
        name: "set_w".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![2.0])),
        actions: vec![Action::Set(SetAction {
            compartment: "W".into(),
            value: Expr::const_(42.5),
            value_grad: Default::default(),
        })],
        kind: ir::intervention::InterventionKind::Scenario,
    });

    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let traj = OdeSim
        .run(
            &compiled,
            &params,
            0,
            &SimConfig::Ode(OdeConfig {
                t_start: model.simulation.t_start,
                t_end: model.simulation.t_end,
                dt: 0.5,
            }),
        )
        .unwrap();

    // W is held constant by dW/dt = 0 except for the intervention. Before t=2 it
    // is its init (0.0); from t=2 on it is exactly 42.5.
    let w_end = traj.snapshots.last().unwrap().real_state.values[0];
    assert_eq!(
        w_end, 42.5,
        "real-compartment `set` must apply the exact f64 (got {w_end})"
    );
    // And it actually changed from the init — the intervention fired.
    let w0 = traj.snapshots.first().unwrap().real_state.values[0];
    assert_eq!(w0, 0.0, "W starts at its init 0.0 (got {w0})");
}

/// gh#196: `set W = -5` (W is a real compartment) is a config bug — symmetric
/// with `set(int, <0)` and `add(<0)`, both of which already error. The real
/// arena's `set` had no negativity guard, so a negative real-set was silently
/// accepted and -5 flowed into the reservoir. End-to-end through the ODE backend
/// this must now error.
#[test]
fn intervention_set_negative_on_real_compartment_errors() {
    let mut model = load_real_coupled();
    model.interventions.push(Intervention {
        name: "set_w_neg".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![2.0])),
        actions: vec![Action::Set(SetAction {
            compartment: "W".into(),
            value: Expr::const_(-5.0),
            value_grad: Default::default(),
        })],
        kind: ir::intervention::InterventionKind::Scenario,
    });

    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let err = OdeSim
        .run(
            &compiled,
            &params,
            0,
            &SimConfig::Ode(OdeConfig {
                t_start: model.simulation.t_start,
                t_end: model.simulation.t_end,
                dt: 0.5,
            }),
        )
        .expect_err("a `set` driving a real compartment negative must error, not write -5");
    match err {
        sim::SimError::NegativeCount { compartment, cause, .. } => {
            assert_eq!(compartment, "W", "should point at the real compartment set negative");
            assert_eq!(cause, sim::NegativeCountCause::InterventionNegative);
        }
        other => panic!("expected NegativeCount{{InterventionNegative}}, got: {other}"),
    }
}
