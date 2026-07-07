//! T2 — `balance {}` conservation through flow AND through a coincident
//! intervention substep (chain-binomial; balance is chain-only).
//!
//! Model: S, I, R integer. Live transition `infect : S --> I @ beta*S` (β>0, so
//! S/I carry real flow but `infect` conserves S+I). A balance constraint pins the
//! residual into R: `balance { target: R, expr: N0 - S - I }` — every substep,
//! after transitions/interventions, R is overwritten with N0 − S − I. A scheduled
//! intervention `cull : fraction_transfer(S -> R, fraction=0.2) at t=5` moves 20%
//! of S out; balance then re-derives R so the total is preserved across the cull
//! substep too.
//!
//! Assertions:
//!   - S + I + R == N0 (= 1000) at EVERY output snapshot (balance conserves the
//!     total through flow and through the cull-then-rebalance substep), and
//!   - DISCRIMINATING CONTROL: R > 0 by the end. `infect` alone keeps S+I = N0, so
//!     N0 − S − I stays 0; R can only become positive because the cull actually
//!     moved mass out of S. R > 0 is therefore proof the cull fired and balance
//!     absorbed it — not a no-op conservation that would hold even with no effect.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    intervention::{Action, FractionTransfer, Intervention, InterventionSchedule},
    model::{
        BalanceSpec, Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, RegularOutputSchedule, SimulationConfig,
    },
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

const N0: i64 = 1000;
const BETA: f64 = 0.1;
const T_END: f64 = 10.0;

/// `mul(left, right)`.
fn mul(left: Expr, right: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(left), right: Box::new(right) } })
}
/// `sub(left, right)`.
fn sub(left: Expr, right: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Sub, left: Box::new(left), right: Box::new(right) } })
}
fn param(p: &str) -> Expr { Expr::Param(ParamExpr { param: p.into() }) }
fn pop(p: &str) -> Expr { Expr::Pop(PopExpr { pop: p.into() }) }

/// S, I, R integer (indices 0, 1, 2). `infect : S --> I @ beta*S`; balance pins
/// R = N0 − S − I; a scheduled cull moves 20% of S into R at t=5.
fn balance_model() -> CompiledModel {
    let model = Model {
        ic_grad: Default::default(),
        name: "balance_conservation".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infect".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: mul(param("beta"), pop("S")), // beta * S
            metadata: None,
            draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "cull".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![5.0])),
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(),
                dst: "R".into(),
                fraction: Expr::const_(0.2),
            })],
            kind: ir::intervention::InterventionKind::Scenario,
        }],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: BETA }, param_kind: None, param_dim: None },
            Parameter { name: "N0".into(), value: ir::parameter::ParamValue::Fixed { value: N0 as f64 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("S".into(), N0 as f64);
            m.insert("I".into(), 0.0);
            m.insert("R".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: T_END, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(11),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        // balance { target: R, expr: N0 - S - I }
        balance: Some(BalanceSpec { target: "R".into(), expr: sub(sub(param("N0"), pop("S")), pop("I")) }),
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    CompiledModel::new(model).unwrap()
}

#[test]
fn balance_conserves_total_through_flow_and_cull() {
    let compiled = Arc::new(balance_model());
    let traj = ChainBinomialSim
        .run(&compiled, &compiled.default_params, 11,
             &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 }))
        .unwrap();

    // S, I, R all integer → counts = [S, I, R].
    let s_idx = compiled.global_to_int[compiled.comp_index["S"]].unwrap();
    let i_idx = compiled.global_to_int[compiled.comp_index["I"]].unwrap();
    let r_idx = compiled.global_to_int[compiled.comp_index["R"]].unwrap();

    assert!(!traj.snapshots.is_empty(), "expected output snapshots");

    // S + I + R == N0 at EVERY output snapshot (conservation through flow and
    // through the cull-then-rebalance substep at t=5).
    for snap in &traj.snapshots {
        let s = snap.int_state.counts[s_idx];
        let i = snap.int_state.counts[i_idx];
        let r = snap.int_state.counts[r_idx];
        assert_eq!(
            s + i + r, N0,
            "balance must conserve S+I+R = N0 at t={} (got S={s} I={i} R={r}, sum={})",
            snap.t, s + i + r
        );
    }

    // DISCRIMINATING CONTROL: R > 0 by the end. `infect` conserves S+I, so
    // N0 − S − I (= R under balance) would stay 0 with no cull; R > 0 proves the
    // cull moved mass into R and balance absorbed it (not a vacuous no-op).
    let last = traj.snapshots.last().unwrap();
    let r_end = last.int_state.counts[r_idx];
    assert!(
        r_end > 0,
        "cull (fraction_transfer S->R @5) must have moved mass into R; R_end={r_end} (≤0 ⇒ cull was a no-op)"
    );

    // And S must have depleted from N0 (flow + cull both removed S), so the
    // conservation above held under genuine movement, not a frozen state.
    let s_end = last.int_state.counts[s_idx];
    assert!(s_end < N0, "S must have depleted from {N0} (infect + cull), got {s_end}");
}
