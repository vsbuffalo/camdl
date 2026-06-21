//! Red→green for the events-on-real fix (step 3 of the effect-purity seam).
//!
//! Historically `inject_event_deltas` only resolved the integer arena
//! (`global_to_int`), so an always-active event targeting a REAL compartment
//! was silently dropped on every backend — a latent correctness bug with zero
//! fixture coverage. The pure resolver handles both arenas, and the kernels now
//! apply the real-event deltas to the reservoir.
//!
//! Model: a real reservoir `W` (init 0, dW/dt = 0) plus an always-active event
//! `topup` that adds 2.5 to W at steps 2 and 4. A correct backend ends with
//! W = 5.0 (two firings, exact f64 — the real path must NOT round). On the
//! pre-fix code the event was dropped and W stayed 0.0.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    intervention::{AddAction, Action, Intervention, InterventionSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, OdeSim,
};

/// S, I integer (a rate-`gamma` decay so the run is well-defined) + W real held
/// constant by dW/dt = 0 except for the always-active event `topup`.
fn model_with_real_event() -> CompiledModel {
    let model = Model {
        name: "event_on_real".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "W".into(), kind: CompartmentKind::Real },
        ],
        transitions: vec![Transition {
            name: "decay".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
        }],
        ode_equations: vec![ir::ode_equation::OdeEquation {
            compartment: "W".into(),
            derivative: Expr::const_(0.0),
        }],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "topup".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![2.0, 4.0])),
            actions: vec![Action::Add(AddAction {
                compartment: "W".into(),
                count: Expr::const_(2.5),
            })],
            kind: ir::intervention::InterventionKind::Event,
        }],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Fixed { value: 0.05 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("S".into(), 100.0);
            m.insert("I".into(), 0.0);
            m.insert("W".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 5.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 5.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(7),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    CompiledModel::new(model).unwrap()
}

fn w_end(traj: &sim::state::Trajectory) -> f64 {
    traj.snapshots.last().unwrap().real_state.values[0]
}

/// ODE backend: the event applies to the real reservoir, exact f64.
#[test]
fn event_on_real_applies_under_ode() {
    let compiled = model_with_real_event();
    let params = compiled.default_params.clone();
    let traj = OdeSim
        .run(&compiled, &params, 0, &SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 5.0, dt: 1.0 }))
        .unwrap();
    assert_eq!(
        w_end(&traj), 5.0,
        "always-active event adding 2.5 to a real reservoir at steps 2 and 4 must \
         leave W = 5.0 (got {}); pre-fix the real-targeted event was dropped (W = 0)",
        w_end(&traj)
    );
}

/// Chain-binomial (the inference kernel): same — real-event deltas apply to the
/// reservoir and survive the write-back.
#[test]
fn event_on_real_applies_under_chain_binomial() {
    let compiled = Arc::new(model_with_real_event());
    let params = compiled.default_params.clone();
    let traj = ChainBinomialSim
        .run(&compiled, &params, 7, &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 5.0, dt: 1.0 }))
        .unwrap();
    assert_eq!(
        w_end(&traj), 5.0,
        "chain-binomial must apply the real-targeted event (W = 5.0, got {})",
        w_end(&traj)
    );
}
