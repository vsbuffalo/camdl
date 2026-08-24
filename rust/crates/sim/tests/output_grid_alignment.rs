//! gh#125 — chain_binomial (Snap) must reject output times off the dt grid.
//!
//! The chain-binomial backend steps a full `dt` per substep and records outputs
//! at grid times (`t_start + k*dt`); it never lands on a sub-`dt` output time.
//! An off-grid output time would therefore be stamped with the POST-step state
//! under an earlier label — a silent-wrong result (the snapshot labelled `t=0.5`
//! would carry the state at `t=1.0`). This backend rejects a misaligned output
//! time with a located error.
//!
//! ODE/Gillespie use the EXACT policy: they clip exactly to each output time and
//! record the true state there, so they accept off-grid output times. The guard
//! must be Snap-specific, NOT a blanket rejection.

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

/// Minimal S --> I model whose output schedule is `Regular { start: 0, step }`.
/// With a runtime `dt` that `step` does not divide, the output grid is off the
/// substep grid.
fn model(t_end: f64, step: f64) -> CompiledModel {
    let m = Model {
        ic_grad: Default::default(),
        name: "output_grid_alignment".into(),
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
        initial_conditions: InitialConditions::constants({
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
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    CompiledModel::new(m).unwrap()
}

#[test]
fn chain_binomial_rejects_off_grid_output_time() {
    // dt = 1, output step = 0.5 → output times 0, 0.5, 1.0, … The 0.5 output is
    // off the `t_start + k*dt` (k*1) grid; the Snap backend cannot represent the
    // sub-dt state, so it must reject rather than silently stamp the t=1.0 state.
    let compiled = Arc::new(model(5.0, 0.5));
    let err = ChainBinomialSim
        .run(&compiled, &compiled.default_params, 7,
             &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 5.0, dt: 1.0 }))
        .expect_err("chain_binomial must REJECT a sub-dt (off-grid) output time (gh#125)");
    let msg = err.to_string();
    assert!(msg.contains("0.5"), "error must name the offending output time 0.5: {msg}");
    assert!(msg.to_lowercase().contains("dt") || msg.to_lowercase().contains("grid"),
        "error must explain the dt-grid requirement: {msg}");
}

#[test]
fn gillespie_accepts_off_grid_output_time() {
    // Exact clips exactly to each output time, so a sub-dt output is fine and
    // records the TRUE state there — the guard must be Snap-specific.
    let compiled = Arc::new(model(5.0, 0.5));
    let traj = GillespieSim
        .run(&compiled, &compiled.default_params, 7,
             &SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: 5.0, output_dt: None }))
        .expect("gillespie (Exact) must accept off-grid output times");
    assert!(traj.snapshots.iter().any(|s| (s.t - 0.5).abs() < 1e-9),
        "gillespie must record the t=0.5 snapshot (Exact clips to it)");
}

#[test]
fn chain_binomial_accepts_grid_aligned_output() {
    // step = 1 divides dt = 1 → every output time is on the grid; no regression.
    let compiled = Arc::new(model(5.0, 1.0));
    let traj = ChainBinomialSim
        .run(&compiled, &compiled.default_params, 7,
             &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 5.0, dt: 1.0 }))
        .expect("chain_binomial must accept grid-aligned output times");
    assert_eq!(traj.snapshots.last().unwrap().t, 5.0);
}

#[test]
fn chain_binomial_accepts_aligned_fractional_dt_over_long_horizon() {
    // gh#125 review regression: output-every == dt == 0.1 (perfectly on-grid) over
    // a long horizon. `output_times` accumulates `t += 0.1`, so the accumulated
    // value drifts from the freshly-computed grid by an amount that GROWS with t;
    // an absolute 1e-12 tolerance false-rejected this legitimate model near t≈93.
    // The magnitude-scaled tolerance must accept it.
    let compiled = Arc::new(model(100.0, 0.1));
    let traj = ChainBinomialSim
        .run(&compiled, &compiled.default_params, 7,
             &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 100.0, dt: 0.1 }))
        .expect("chain_binomial must accept output-every==dt=0.1 on a long horizon (gh#125 review)");
    assert!((traj.snapshots.last().unwrap().t - 100.0).abs() < 1e-6,
        "last snapshot should be ~100.0, got {}", traj.snapshots.last().unwrap().t);
}
