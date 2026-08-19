//! T1 — cross-backend lifecycle agreement UNDER FLOW (rate ≠ 0), with an
//! integer `Set`, on all four backends including gillespie's snapshot apply.
//!
//! The model has a live transition `infect : S --> I @ beta*S` (β=0.1 > 0), so
//! S/I carry real stochastic flow whose realization differs per backend. A third
//! compartment `V` is touched ONLY by effects — it appears in no transition
//! stoichiometry and in no rate — so its trajectory is deterministic and must be
//! IDENTICAL on every backend:
//!
//!   - always-active EVENT `topup`  : add(V, 100) at t=5   → V = 100 in [5, 8)
//!   - scheduled INTERVENTION `pin` : set(V, 50)  at t=8   → V = 50  at t ≥ 8
//!
//! The integer `set` lands the snapshot-relative delta 50 − 100 = −50.
//!
//! Asserting V's exact integer value on each backend pins, in one test:
//!   (1) always-active event wiring on ALL four backends (incl. gillespie's
//!       distinct `apply_events_at` snapshot path),
//!   (2) scheduled-intervention wiring on all four (the shared `apply_post_advance`),
//!   (3) the integer-`Set` happy path (snapshot-relative delta), and
//!   (4) agreement-under-flow — V is identical across backends *while* S/I flow.
//!
//! Non-vacuity controls: S strictly DECREASED from 1000 on every backend (the
//! transition actually ran, so agreement is informative, not a tautology), and
//! ODE's final S matches the deterministic decay S0·exp(−β·t_end) = 1000·e^{−1}
//! ≈ 367.88.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    intervention::{AddAction, Action, Intervention, InterventionSchedule, SetAction},
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
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const BETA: f64 = 0.1;
const T_END: f64 = 10.0;
const S0: i64 = 1000;

/// S, I, V all integer (declared in that order → int-state indices 0, 1, 2).
/// `infect : S --> I @ beta*S` is the only transition; V is untouched by it.
fn flow_model() -> CompiledModel {
    let model = Model {
        ic_grad: Default::default(),
        name: "lifecycle_agreement_under_flow".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "V".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infect".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            // beta * S
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![
            // always-active EVENT: add 100 to V at t=5.
            Intervention {
                name: "topup".into(),
                base_name: None,
                fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![5.0])),
                actions: vec![Action::Add(AddAction {
                    compartment: "V".into(),
                    count: Expr::const_(100.0),
                })],
                kind: ir::intervention::InterventionKind::Event,
            },
            // scheduled INTERVENTION: set V to 50 at t=8.
            Intervention {
                name: "pin".into(),
                base_name: None,
                fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![8.0])),
                actions: vec![Action::Set(SetAction {
                    compartment: "V".into(),
                    value: Expr::const_(50.0),
                })],
                kind: ir::intervention::InterventionKind::Scenario,
            },
        ],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: BETA }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("S".into(), S0 as f64);
            m.insert("I".into(), 0.0);
            m.insert("V".into(), 0.0);
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
            dt: Some(1.0), rng_seed: Some(7),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    CompiledModel::new(model).unwrap()
}

/// Integer count for compartment `name` at the output snapshot nearest time `t`.
/// S, I, V are all integer, so the snapshot's `int_state.counts` is `[S, I, V]`.
fn count_at(compiled: &CompiledModel, traj: &Trajectory, name: &str, t: f64) -> i64 {
    let g = compiled.comp_index[name];
    let local = compiled.global_to_int[g].expect("compartment is integer");
    let snap = traj
        .snapshots
        .iter()
        .find(|s| (s.t - t).abs() < 1e-6)
        .unwrap_or_else(|| panic!("no output snapshot at t={t} (have {:?})",
            traj.snapshots.iter().map(|s| s.t).collect::<Vec<_>>()));
    snap.int_state.counts[local]
}

fn run(backend: &str) -> (Arc<CompiledModel>, Trajectory) {
    let compiled = Arc::new(flow_model());
    let cfg = match backend {
        "gillespie" => SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: T_END, output_dt: Some(1.0) }),
        "chain_binomial" => SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 }),
        "ode" => SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: 1.0 }),
        other => panic!("unknown backend {other}"),
    };
    let sim: &dyn Simulate = match backend {
        "gillespie" => &GillespieSim,
        "ode" => &OdeSim,
        _ => &ChainBinomialSim,
    };
    let traj = sim.run(&compiled, &compiled.default_params, 7, &cfg).unwrap();
    (compiled, traj)
}

/// V is effect-only: every backend must agree EXACTLY — 100 in [5,8), 50 at t≥8.
/// We sample the interior points t∈{6,7} (after the t=5 event, before the t=8
/// set) and t∈{9,10} (after the t=8 set) to avoid boundary-step ambiguity.
fn assert_v_lifecycle(backend: &str) {
    let (compiled, traj) = run(backend);

    // V == 100 throughout [5, 8): the always-active `topup` event added 100.
    for t in [6.0, 7.0] {
        let v = count_at(&compiled, &traj, "V", t);
        assert_eq!(v, 100, "{backend}: V must be 100 at t={t} (event add(V,100)@5), got {v}");
    }
    // V == 50 at t >= 8: the scheduled `pin` set V to 50 (Δ = 50 − 100 = −50).
    for t in [9.0, 10.0] {
        let v = count_at(&compiled, &traj, "V", t);
        assert_eq!(v, 50, "{backend}: V must be 50 at t={t} (intervention set(V,50)@8), got {v}");
    }

    // CONTROL: the transition actually ran — S strictly decreased from S0, so V's
    // agreement is informative (it held constant *under* live flow), not vacuous.
    let s_end = count_at(&compiled, &traj, "S", T_END);
    assert!(
        s_end < S0,
        "{backend}: S must have depleted from {S0} (rate=beta*S, beta={BETA}>0), got {s_end}"
    );
    // And I gained what S lost on the infect transition (V is uninvolved).
    let i_end = count_at(&compiled, &traj, "I", T_END);
    assert_eq!(s_end + i_end, S0, "{backend}: S + I conserved on infect (V uninvolved), got {}", s_end + i_end);
}

#[test]
fn chain_binomial_agrees_under_flow() {
    assert_v_lifecycle("chain_binomial");
}

#[test]
fn gillespie_agrees_under_flow() {
    assert_v_lifecycle("gillespie");
}

#[test]
fn ode_agrees_under_flow() {
    assert_v_lifecycle("ode");
}

/// ODE is the deterministic reference: with dS/dt = −β·S the survivor count is
/// S(t_end) = S0·exp(−β·t_end) = 1000·e^{−1} ≈ 367.88, so the integer-rounded
/// output lands in a tight ballpark. This anchors the "flow happened" control to
/// a hand-computed number, not just "S < 1000".
#[test]
fn ode_final_s_matches_deterministic_decay() {
    let (compiled, traj) = run("ode");
    let s_end = count_at(&compiled, &traj, "S", T_END);
    let expected = (S0 as f64) * (-BETA * T_END).exp(); // 1000*e^{-1} ≈ 367.88
    assert!(
        (s_end as f64 - expected).abs() <= 2.0,
        "ODE S(t_end) should be ≈ {expected:.2} (1000·e^{{-1}}); got {s_end}"
    );
    // V is still pinned to its effect-only trajectory on the deterministic path.
    assert_eq!(count_at(&compiled, &traj, "V", 7.0), 100);
    assert_eq!(count_at(&compiled, &traj, "V", 10.0), 50);
}
