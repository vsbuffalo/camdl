//! Regression: a time-varying observation likelihood must be evaluated at the
//! observation's actual time, not a frozen `t = 0`.
//!
//! The bug: `eval_likelihood_resolved` (and the projection eval) built their
//! `EvalCtx` with a hardcoded `t: 0.0`, so any time-dependent term in an
//! observation expression — e.g. a reporting ramp
//! `rho_t = rho_max / (1 + exp(-(t - t_rep) / w_rep))` — was frozen at
//! `rho_t(0)` for every observation. That silently corrupts the likelihood of
//! any fit whose reporting/observation process varies in calendar time, and
//! breaks origin-invariance (shifting `origin`/`from` changed the loglik).
//!
//! This test isolates the mechanism with the simplest possible time-dependent
//! likelihood: a Poisson whose `rate = time + 1`. With the bug, both
//! observations evaluate `rate = 0 + 1 = 1` and the two logliks are identical;
//! correct behavior evaluates `rate = 11` at t=10 and `rate = 21` at t=20.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr, ConstExpr, TimeExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        BoundObs, ParticleState, ObservationModel, MultiStreamObsModel, dense_cells,
        multi_stream_obs::{StreamSpec, StreamProjection},
        obs_loglik::poisson_logpmf,
    },
};

/// Pure-death model with a single transition, used only as a vehicle for the
/// observation block — the dynamics are irrelevant here.
fn model() -> Arc<CompiledModel> {
    let m = Model {
        name: "obs_time_dependence".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "death".into(),
                stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
                rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                }}),
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new(); h.insert("N".into(), 100.0); h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 50.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 50.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

/// Poisson likelihood with `rate = time + 1` — purely time-dependent, so the
/// projected/state value is irrelevant and the only thing under test is which
/// `t` the evaluator uses.
fn time_varying_obs(compiled: &Arc<CompiledModel>, obs_times: Vec<f64>, observations: Vec<f64>) -> MultiStreamObsModel {
    let rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
        op: BinOp::Add,
        left: Box::new(Expr::Time(TimeExpr { time: () })),
        right: Box::new(Expr::Const(ConstExpr { value: 1.0 })),
    }});
    MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: ir::observation::ObservationModel {
                name: "cases".into(),
                source: "cases".into(),
                columns: vec![
                    ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                    ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ],
                scored: "cases".into(),
                emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: ir::observation::Projection::CumulativeFlow("death".into()),
                likelihood: ir::observation::Likelihood::Poisson(
                    ir::observation::PoissonLikelihood { rate: ir::Diffable::new(rate) },
                ),
            },
            observations: dense_cells(observations),
            obs_times,
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap()
}

#[test]
fn obs_likelihood_uses_observation_time_not_zero() {
    let compiled = model();
    let params = compiled.default_params.clone();

    // Two observations at t = 10 and t = 20, same observed value.
    let obs_m = time_varying_obs(&compiled, vec![10.0, 20.0], vec![11.0, 11.0]);

    // State is identical for both; projection value is unused by `rate = t + 1`.
    // One `FlowSum` stream ⇒ one `acc` slot; the trait scoring reads `acc[0]`.
    // (The rate ignores `projected`, so the value is immaterial — but `acc` must
    // be sized for the single Interval stream.)
    let mut state = ParticleState::new(compiled.int_local_to_global.len(), 1, 1);
    state.counts[0] = 100;
    state.flow_accumulators[0] = 5;
    state.acc[0] = 5;

    let ll0 = obs_m.log_likelihood(&state, 0, &params); // t = 10 → rate = 11
    let ll1 = obs_m.log_likelihood(&state, 1, &params); // t = 20 → rate = 21

    // Correct evaluation uses the observation time, so the two rates differ.
    let expected0 = poisson_logpmf(11.0, 11.0);
    let expected1 = poisson_logpmf(11.0, 21.0);

    assert!(
        (ll0 - expected0).abs() < 1e-9,
        "obs at t=10 should evaluate rate=11 (logpmf {expected0}); got {ll0} \
         (frozen-t bug evaluates rate=1: {})",
        poisson_logpmf(11.0, 1.0),
    );
    assert!(
        (ll1 - expected1).abs() < 1e-9,
        "obs at t=20 should evaluate rate=21 (logpmf {expected1}); got {ll1}",
    );
    assert!(
        (ll0 - ll1).abs() > 1.0,
        "time-varying likelihood must differ between t=10 and t=20; \
         identical values ({ll0} == {ll1}) mean t is frozen at 0",
    );
}
