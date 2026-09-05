//! gh#833: a declared period OPENS at its own start, not at `t_start`.
//!
//! The defect this pins: an incidence stream's accumulator is zeroed at
//! `t_start` and thereafter only where the stream SCORES, so the first declared
//! period `[start_0, stop_0)` was scored against flow accumulated over
//! `[t_start, stop_0)` — however far `start_0` sat inside it. With a warm-up
//! before the first observation, which is the normal shape of a fit, a row
//! declaring one day was scored against the whole warm-up.
//!
//! The oracle is exact and read off the trajectory itself. With a single
//! outflow `recovery: I -> R @ gamma * I` on the deterministic ODE backend, the
//! flow through `recovery` over a window is exactly the per-interval flow the
//! integrator reports for that window. So the expected count under the
//! declared window is the reported flow over `(5,6]` alone, and under the
//! defect it is the flow summed over `(0,6]`. Poisson-scoring the same
//! observation against each gives two log-likelihoods that differ by many
//! nats; `compute_ode_loglik` must produce the first.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        RegularOutputSchedule, SimulationConfig,
    },
    observation::{
        ColumnRole, Likelihood, ObsColumn, ObservationModel as IrObs, ObservationSchedule,
        PoissonLikelihood, Projection,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    inference::{
        compute_ode_loglik, dense_cells,
        multi_stream_obs::{StreamProjection, StreamSpec},
        obs_loglik::poisson_logpmf,
        BoundObs, MultiStreamObsModel, Period,
    },
    simulate::Simulate,
    OdeSim,
};

const GAMMA: f64 = 0.1;
const I0: f64 = 40.0;
const T_END: f64 = 10.0;

/// S,I,R with ONE transition, `recovery: I -> R @ gamma * I`, observed as
/// `incidence(recovery)` under a Poisson. Regular unit output so the ODE path
/// has a snapshot at every integer boundary, including the declared start.
fn model() -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "covers_leading_reset".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "recovery".into(),
            stoichiometry: vec![
                StoichiometryEntry("I".into(), -1),
                StoichiometryEntry("R".into(), 1),
            ],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
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
        interventions: vec![],
        observations: vec![IrObs {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn {
                    name: "cases".into(),
                    role: ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "cases".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            covers: None,
            projection: Projection::CumulativeFlow("recovery".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "gamma".into(),
            value: ir::parameter::ParamValue::Fixed { value: GAMMA },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), 900.0);
            h.insert("I".into(), I0);
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
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
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

/// The ODE backend's own per-interval `recovery` flow at each unit boundary,
/// indexed by the boundary's time `t` (flow over `(t-1, t]`).
fn reported_recovery_flow(compiled: &CompiledModel, params: &[f64]) -> HashMap<u32, f64> {
    let recovery = compiled
        .model
        .transitions
        .iter()
        .position(|t| t.name == "recovery")
        .expect("recovery transition");
    let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: 1.0 });
    let traj = OdeSim.run(compiled, params, 1, &cfg).expect("ode run");
    traj.snapshots
        .iter()
        .map(|s| (s.t.round() as u32, s.flows.as_real()[recovery]))
        .collect()
}

/// Proposal Testing item 4: a declared gap discards the flow in the uncovered
/// span, checked against a run where that span carries known flow. Periods
/// [2,3) and [5,6): row 0 scores against the flow over (2,3] alone, row 1
/// against (5,6] alone, and the flow over (3,5] — real, and large enough to
/// matter — appears in neither. The alternative reading, where the second bin
/// silently widens to swallow the gap (what deleting a row does to an
/// undeclared stream), is the number this must NOT produce.
#[test]
fn a_declared_gap_discards_the_flow_in_the_uncovered_span() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let recovery = compiled.model.transitions.iter().position(|t| t.name == "recovery").unwrap();

    let flows = reported_recovery_flow(&compiled, &params);
    let expect0 = flows[&3];
    let expect1 = flows[&6];
    let uncovered: f64 = flows[&4] + flows[&5];
    assert!(uncovered > 1.0,
        "vacuous fixture: the gap must carry real flow to discard (got {uncovered})");

    let y0 = expect0.round().max(1.0);
    let y1 = expect1.round().max(1.0);
    let spec = StreamSpec::dense_covering(
        StreamProjection::FlowSum(vec![recovery]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![y0, y1]),
        vec![Period::new(2.0, 3.0).unwrap(), Period::new(5.0, 6.0).unwrap()],
    );
    let (bound, report) = BoundObs::bind(vec![spec]).expect("a per-row gap binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    let union = bound.times().to_vec();
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone()).unwrap();

    let ll = compute_ode_loglik(&compiled, &obs_model, &union, 1.0, &params, 1.0)
        .expect("ode loglik");

    let ll_correct = poisson_logpmf(y0, expect0) + poisson_logpmf(y1, expect1);
    let ll_swallowed = poisson_logpmf(y0, expect0) + poisson_logpmf(y1, uncovered + expect1);
    assert!(
        (ll - ll_correct).abs() < 1e-9,
        "each row must be scored against its own window only: got {ll}, expected \
         {ll_correct}; a second bin widened to swallow the gap would give {ll_swallowed}"
    );
    assert!((ll - ll_swallowed).abs() > 1e-6, "non-vacuous: the two readings must differ");
}

#[test]
fn a_declared_first_period_is_scored_against_its_own_window_only() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let recovery = compiled.model.transitions.iter().position(|t| t.name == "recovery").unwrap();

    // Oracle straight from the integrator: what the first declared window
    // [5,6) should be scored against, and what the defect scored it against.
    let flows = reported_recovery_flow(&compiled, &params);
    let expected_declared = flows[&6];
    let expected_buggy: f64 = (1..=6).map(|t| flows[&t]).sum();
    assert!(
        (expected_buggy - expected_declared).abs() > 1.0,
        "vacuous fixture: the one-day and six-day windows must separate clearly \
         (declared={expected_declared}, buggy={expected_buggy})"
    );

    // One observation, declared to cover [5,6). The count is the declared
    // expectation rounded, so the correct hypothesis is the likely one.
    let y = expected_declared.round().max(1.0);
    let spec = StreamSpec::dense_covering(
        StreamProjection::FlowSum(vec![recovery]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![y]),
        vec![Period::new(5.0, 6.0).unwrap()],
    );
    let (bound, report) = BoundObs::bind(vec![spec]).expect("a declared stream binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    let union = bound.times().to_vec();
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone()).unwrap();

    let ll = compute_ode_loglik(&compiled, &obs_model, &union, 1.0, &params, 1.0)
        .expect("ode loglik");

    let ll_declared = poisson_logpmf(y, expected_declared);
    let ll_buggy = poisson_logpmf(y, expected_buggy);
    assert!(
        (ll - ll_declared).abs() < 1e-9,
        "a declared [5,6) period must be scored against the flow over (5,6] only: \
         got {ll}, expected {ll_declared} (the whole-warm-up reading would give {ll_buggy})"
    );
    assert!(
        (ll - ll_buggy).abs() > 1e-6,
        "non-vacuous: the correct and defective readings must be distinguishable"
    );
}
