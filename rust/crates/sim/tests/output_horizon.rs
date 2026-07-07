//! gh#143 — `simulation.t_end` is the sole output-horizon authority.
//!
//! The output schedule no longer carries its own `end` (a redundant mirror of
//! `simulation.t_end` that had drifted apart post-gh#233). The runtime derives
//! output times from `[start, simulation.t_end]` at emission. This pins the two
//! failure modes the drift produced:
//!
//!   - UPWARD: a longer `t_end` must extend emission all the way to `t_end`
//!     (the bug: emission stopped at the schedule's stale `end`, ignoring a
//!     raised horizon).
//!   - NO PADDING: emission must stop AT `t_end` — never emit frozen-state rows
//!     past the dynamics horizon (the bug: a shorter `t_end` stopped the loop
//!     early, then padded frozen rows out to the stale `end`).
//!
//! Both are asserted by running a real backend and checking the emitted times,
//! for both discrete backends (each routes through `OutputTimes::from_model`).

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        RegularOutputSchedule, SimulationConfig,
    },
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

fn mul(left: Expr, right: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(left), right: Box::new(right) } })
}
fn param(p: &str) -> Expr { Expr::Param(ParamExpr { param: p.into() }) }
fn pop(p: &str) -> Expr { Expr::Pop(PopExpr { pop: p.into() }) }

/// A minimal S --> I model. `t_end` and the output `step` are parameters so a
/// test can pick the horizon; the schedule carries only `start` + `step`.
fn model(t_end: f64, step: f64) -> CompiledModel {
    let m = Model {
        ic_grad: Default::default(),
        name: "output_horizon".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infect".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: mul(param("beta"), pop("S")),
            metadata: None,
            draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: 0.05 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut ic = HashMap::new();
            ic.insert("S".into(), 100.0);
            ic.insert("I".into(), 0.0);
            ic
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(7),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    CompiledModel::new(m).unwrap()
}

fn run(compiled: &Arc<CompiledModel>, cfg: SimConfig) -> Vec<f64> {
    let traj = match &cfg {
        SimConfig::ChainBinomial(_) => ChainBinomialSim.run(compiled, &compiled.default_params, 7, &cfg).unwrap(),
        SimConfig::Gillespie(_) => GillespieSim.run(compiled, &compiled.default_params, 7, &cfg).unwrap(),
        _ => unreachable!(),
    };
    traj.snapshots.iter().map(|s| s.t).collect()
}

/// Emission reaches `t_end` (upward) and never exceeds it (no frozen padding),
/// for a horizon both above and below the model's dynamics, on both discrete
/// backends. The emitted grid is exactly `0, step, …, t_end`.
#[test]
fn emission_confined_to_t_end() {
    for &t_end in &[40.0_f64, 160.0_f64] {
        let compiled = Arc::new(model(t_end, 1.0));

        let cb = run(&compiled, SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end, dt: 1.0 }));
        let gil = run(&compiled, SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end, output_dt: None }));

        for (name, times) in [("chain_binomial", &cb), ("gillespie", &gil)] {
            assert_eq!(
                *times.last().unwrap(), t_end,
                "{name}: last emitted time must equal t_end={t_end} (upward horizon must not be ignored)"
            );
            assert!(
                times.iter().all(|&t| t <= t_end),
                "{name}: no output time may exceed t_end={t_end} (no frozen-state padding); got {times:?}"
            );
            let expected: Vec<f64> = (0..=(t_end as i64)).map(|i| i as f64).collect();
            assert_eq!(times, &expected, "{name}: emitted grid must be 0,1,…,{t_end}");
        }
    }
}
