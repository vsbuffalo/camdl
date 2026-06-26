//! gh#217 — co-drain: a compartment drained by BOTH a transition AND a draining
//! `events {}` action in the same step.
//!
//! Model (the gh#217 minimal repro):
//!   compartments { A, B, I }
//!   transitions  { drain : A --> I @ r * A }
//!   events       { dump  : transfer(fraction = 1.0, from = A, to = B) at [5] }
//!   init { A = 1000 },  r = 0.1
//!
//! The `drain` hazard acts over the interval; the `dump` is a point operation AT
//! t=5. The continuous-time ground truth (ODE / Gillespie): drain removes its
//! flow over the interval, then `dump` moves ALL remaining A → B. So A = 0 at t=5
//! and mass is conserved (A + B + I = 1000).
//!
//! Before the fix, chain_binomial resolved the draining event against the
//! START-OF-STEP snapshot and FUSED its delta into the same atomic apply as the
//! transition draws, so both the transition and `dump` subtracted from the full
//! snapshot A → A = −(drain flow) → `NegativeCount{BinomialOvershoot}`.
//!
//! The fix: draining event actions (the `from` side of a transfer; `Set`) read
//! and act on the POST-TRANSITION residual; inflow `Add` actions keep
//! start-of-step snapshot semantics.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    intervention::{
        AddAction, Action, FractionTransfer, Intervention, InterventionKind, InterventionSchedule,
    },
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        RegularOutputSchedule, SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const R: f64 = 0.1;
const A0: i64 = 1000;
const FIRE_TIME: f64 = 5.0;
const T_END: f64 = 6.0;

/// The gh#217 co-drain model: A, B, I integer (int-state indices 0, 1, 2).
/// `drain : A --> I @ r*A` is the only transition; `dump` is an always-active
/// EVENT that transfers ALL of A → B at t=5.
fn codrain_model() -> CompiledModel {
    let model = Model {
        name: "event_transition_codrain".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "A".into(), kind: CompartmentKind::Integer },
            Compartment { name: "B".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            name: "drain".into(),
            stoichiometry: vec![StoichiometryEntry("A".into(), -1), StoichiometryEntry("I".into(), 1)],
            // r * A
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "r".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "A".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "dump".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![FIRE_TIME])),
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "A".into(),
                dst: "B".into(),
                fraction: Expr::const_(1.0),
            })],
            kind: InterventionKind::Event,
        }],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "r".into(),
            value: ir::parameter::ParamValue::Fixed { value: R },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("A".into(), A0 as f64);
            m.insert("B".into(), 0.0);
            m.insert("I".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: T_END }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![],
    };
    CompiledModel::new(model).unwrap()
}

/// Integer count for compartment `name` at the output snapshot nearest time `t`.
fn count_at(compiled: &CompiledModel, traj: &Trajectory, name: &str, t: f64) -> i64 {
    let g = compiled.comp_index[name];
    let local = compiled.global_to_int[g].expect("compartment is integer");
    let snap = traj
        .snapshots
        .iter()
        .find(|s| (s.t - t).abs() < 1e-6)
        .unwrap_or_else(|| {
            panic!(
                "no output snapshot at t={t} (have {:?})",
                traj.snapshots.iter().map(|s| s.t).collect::<Vec<_>>()
            )
        });
    snap.int_state.counts[local]
}

fn cfg_for(backend: &str) -> SimConfig {
    match backend {
        "gillespie" => SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: T_END, output_dt: Some(1.0) }),
        "chain_binomial" => SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 }),
        "ode" => SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: 1.0 }),
        other => panic!("unknown backend {other}"),
    }
}

fn sim_for(backend: &str) -> &'static dyn Simulate {
    match backend {
        "gillespie" => &GillespieSim,
        "ode" => &OdeSim,
        "chain_binomial" => &ChainBinomialSim,
        other => panic!("unknown backend {other}"),
    }
}

/// Run one backend at `seed`, returning the trajectory (or the SimError).
fn run_backend(backend: &str, seed: u64) -> Result<(Arc<CompiledModel>, Trajectory), sim::SimError> {
    let compiled = Arc::new(codrain_model());
    let cfg = cfg_for(backend);
    let traj = sim_for(backend).run(&compiled, &compiled.default_params, seed, &cfg)?;
    Ok((compiled, traj))
}

// ════════════════════════════════════════════════════════════════════════════
// RED test: chain_binomial must run the co-drain model without overshooting,
// and A must be 0 at/after t=5.
// ════════════════════════════════════════════════════════════════════════════

/// gh#217: chain_binomial must NOT abort with NegativeCount/BinomialOvershoot
/// when A is drained by both `drain` (transition) and `dump` (event) at t=5.
/// A must end at 0 (the `dump` moved all survivors to B).
#[test]
fn chain_binomial_codrain_succeeds_and_drains_a_to_zero() {
    let (compiled, traj) = run_backend("chain_binomial", 1)
        .expect("chain_binomial must simulate the co-drain model (gh#217)");
    for t in [5.0, 6.0] {
        let a = count_at(&compiled, &traj, "A", t);
        assert_eq!(a, 0, "A must be 0 at t={t} (the t=5 dump moves all survivors A → B), got {a}");
    }
    // Mass conserved.
    let a = count_at(&compiled, &traj, "A", T_END);
    let b = count_at(&compiled, &traj, "B", T_END);
    let i = count_at(&compiled, &traj, "I", T_END);
    assert_eq!(a + b + i, A0, "mass must be conserved (A+B+I=A0), got {}", a + b + i);
}

// ════════════════════════════════════════════════════════════════════════════
// Cross-backend agreement: A=0, conservation on all three; chain_binomial E[B],
// E[I] over many seeds match ODE's deterministic B/I at t=5.
// ════════════════════════════════════════════════════════════════════════════

/// ODE / Gillespie already read post-advance/live state, so they handle the
/// co-drain correctly. After the fix chain_binomial joins them: A=0 and mass
/// conserved on all three; and chain_binomial's E[B], E[I] over many seeds match
/// ODE's deterministic split (computed from the ODE run, not hardcoded).
#[test]
fn codrain_cross_backend_agreement() {
    // ODE is the deterministic reference. A → 0; B = survivors moved by dump;
    // I = drained over the interval. Compute the expected split from the ODE run.
    let (ode_c, ode_traj) = run_backend("ode", 1).expect("ode must simulate the co-drain model");
    let ode_a = count_at(&ode_c, &ode_traj, "A", T_END);
    let ode_b = count_at(&ode_c, &ode_traj, "B", T_END);
    let ode_i = count_at(&ode_c, &ode_traj, "I", T_END);
    assert_eq!(ode_a, 0, "ODE: A must be 0 at t_end");
    assert_eq!(ode_a + ode_b + ode_i, A0, "ODE: mass conserved");
    assert!(ode_b > 0 && ode_i > 0, "ODE: both B and I must be populated (got B={ode_b}, I={ode_i})");

    // Gillespie: A=0 and conservation (its own stochastic split).
    let (gil_c, gil_traj) = run_backend("gillespie", 1).expect("gillespie must simulate the co-drain model");
    let gil_a = count_at(&gil_c, &gil_traj, "A", T_END);
    let gil_b = count_at(&gil_c, &gil_traj, "B", T_END);
    let gil_i = count_at(&gil_c, &gil_traj, "I", T_END);
    assert_eq!(gil_a, 0, "gillespie: A must be 0 at t_end");
    assert_eq!(gil_a + gil_b + gil_i, A0, "gillespie: mass conserved");

    // chain_binomial: A=0 and conservation on every seed; E[B], E[I] match ODE.
    let n_seeds = 50u64;
    let mut sum_b = 0.0_f64;
    let mut sum_i = 0.0_f64;
    for seed in 1..=n_seeds {
        let (cb_c, cb_traj) = run_backend("chain_binomial", seed)
            .unwrap_or_else(|e| panic!("chain_binomial seed {seed} must simulate: {e}"));
        let a = count_at(&cb_c, &cb_traj, "A", T_END);
        let b = count_at(&cb_c, &cb_traj, "B", T_END);
        let i = count_at(&cb_c, &cb_traj, "I", T_END);
        assert_eq!(a, 0, "chain_binomial seed {seed}: A must be 0 at t_end, got {a}");
        assert_eq!(a + b + i, A0, "chain_binomial seed {seed}: mass conserved, got {}", a + b + i);
        sum_b += b as f64;
        sum_i += i as f64;
    }
    let mean_b = sum_b / n_seeds as f64;
    let mean_i = sum_i / n_seeds as f64;

    // E[B], E[I] within a few % of ODE's deterministic split.
    let tol_frac = 0.05;
    assert!(
        (mean_b - ode_b as f64).abs() <= tol_frac * ode_b as f64,
        "chain_binomial E[B]={mean_b:.1} must be within {}% of ODE B={ode_b}",
        tol_frac * 100.0
    );
    assert!(
        (mean_i - ode_i as f64).abs() <= tol_frac * ode_i as f64,
        "chain_binomial E[I]={mean_i:.1} must be within {}% of ODE I={ode_i}",
        tol_frac * 100.0
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Inflow non-regression: an `add()` inflow event coincident with a transition
// on a DIFFERENT compartment must keep snapshot semantics (unchanged amount).
// ════════════════════════════════════════════════════════════════════════════

/// An inflow `add(V, 100)` event at t=5 on a compartment NOT touched by any
/// transition, while `drain : A --> I @ r*A` flows on A. The inflow amount is a
/// constant (100), so the snapshot-vs-residual distinction is not exercised by
/// the amount itself; this test guards that the inflow path still fires (the fix
/// must not drop or alter `Add` events) and lands exactly 100 in V on every
/// backend — i.e. the change is NOT over-broad (does not route `Add` through the
/// residual phase, which would still be 100 here but the test pins the wiring).
fn inflow_model() -> CompiledModel {
    let model = Model {
        name: "event_transition_inflow".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "A".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "V".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            name: "drain".into(),
            stoichiometry: vec![StoichiometryEntry("A".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "r".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "A".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "topup".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![FIRE_TIME])),
            actions: vec![Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(100.0) })],
            kind: InterventionKind::Event,
        }],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "r".into(),
            value: ir::parameter::ParamValue::Fixed { value: R },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("A".into(), A0 as f64);
            m.insert("I".into(), 0.0);
            m.insert("V".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: T_END }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![],
    };
    CompiledModel::new(model).unwrap()
}

/// The inflow `add(V, 100)` event fires and lands exactly 100 in V — on every
/// backend, unchanged by the co-drain fix (snapshot-based inflow path intact).
#[test]
fn inflow_event_coincident_with_transition_unchanged() {
    for backend in ["chain_binomial", "ode", "gillespie"] {
        let compiled = Arc::new(inflow_model());
        let cfg = cfg_for(backend);
        let traj = sim_for(backend)
            .run(&compiled, &compiled.default_params, 1, &cfg)
            .unwrap_or_else(|e| panic!("{backend}: inflow model must simulate: {e}"));
        // V is 0 before t=5, exactly 100 from t=5 on.
        let v_before = count_at(&compiled, &traj, "V", 4.0);
        let v_after = count_at(&compiled, &traj, "V", 5.0);
        let v_end = count_at(&compiled, &traj, "V", T_END);
        assert_eq!(v_before, 0, "{backend}: V must be 0 before the t=5 inflow, got {v_before}");
        assert_eq!(v_after, 100, "{backend}: inflow add(V,100) must land exactly 100 at t=5, got {v_after}");
        assert_eq!(v_end, 100, "{backend}: V must stay 100 after the inflow, got {v_end}");
    }
}
