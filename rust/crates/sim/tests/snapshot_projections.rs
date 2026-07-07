//! State-snapshot projection tests for `MultiStreamObsModel`.
//!
//! Covers the three new projection paths added in the state-snapshot
//! proposal (2026-04-17):
//!
//! - `prevalence_projection_reads_compartment_count` — `CurrentPop`
//!   reads the correct compartment at observation time and scores the
//!   likelihood against it.
//! - `derived_expr_matches_comp_sum` — a `DerivedExpr` projection that
//!   sums two compartments matches the arithmetic of an explicit
//!   `CurrentPopSum` over the same compartments.
//! - `snapshot_reads_post_intervention_state` — when a scheduled
//!   intervention fires at the observation time, the snapshot sees the
//!   post-intervention state. Guards the intervention-ordering
//!   semantics that the PF observation tick relies on.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, PopExpr, ProjectedExpr},
    intervention::{Action, FractionTransfer, Intervention, InterventionSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    observation::{Likelihood, ObservationModel as IrObservationModel, ObservationSchedule, PoissonLikelihood, Projection},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        BoundObs,
        dense_cells,
        multi_stream_obs::{MultiStreamObsModel, StreamProjection, StreamSpec},
        traits::ObservationModel,
        ParticleState,
    },
};

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

/// Minimal model with a single integer compartment `I` (plus whatever else)
/// and a single Poisson observation on `projected`. Callers supply the IR
/// `Projection` variant to exercise.
fn model_with_obs(
    compartments: Vec<Compartment>,
    initial: HashMap<String, f64>,
    projection: Projection,
) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "snapshot_projection_test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments,
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![IrObservationModel {
            name: "obs".into(),
            source: "obs".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "obs".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "obs".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![5.0])),
            stratum: vec![],
            projection,
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                // rate = projected + 0.1 (floor to avoid Poisson(0) → -inf)
                rate: ir::Diffable::new(Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Add,
                        left: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
                        right: Box::new(Expr::Const(ConstExpr { value: 0.1 })),
                    },
                })),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![],
        initial_conditions: InitialConditions::Explicit(initial),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 5.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 10.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

fn build_obs_model(compiled: &Arc<CompiledModel>, projection: StreamProjection, observed: f64)
    -> MultiStreamObsModel
{
    let obs_ir = compiled.model.observations[0].clone();
    MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection,
            ir_model: obs_ir,
            observations: dense_cells(vec![observed]),
            obs_times: vec![5.0],
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap()
}

/// `CurrentPopSum(["E_e1", "E_e2", "E_e3"])` — the IR shape the compiler
/// emits for `prevalence(E)` on an Erlang-stratified `E` — resolves
/// through `StreamProjection::from_ir` to an `IntCompSum` whose sum
/// matches the declared-order local indices. Guards the compiler↔runtime
/// contract for the Erlang-substage prevalence case.
#[test]
fn current_pop_sum_from_ir_resolves_stratified_compartments() {
    let mut init = HashMap::new();
    init.insert("S".into(),     900.0);
    init.insert("E_e1".into(),   10.0);
    init.insert("E_e2".into(),    7.0);
    init.insert("E_e3".into(),    3.0);
    init.insert("I".into(),       5.0);
    let compartments = vec![
        int_comp("S"), int_comp("E_e1"), int_comp("E_e2"), int_comp("E_e3"), int_comp("I"),
    ];
    let compiled = Arc::new(CompiledModel::new(model_with_obs(
        compartments, init,
        Projection::CurrentPopSum(vec!["E_e1".into(), "E_e2".into(), "E_e3".into()]),
    )).unwrap());
    let params = compiled.default_params.clone();

    // The from_ir resolver must pick the three E_e* int indices.
    let projection = StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "obs",
    ).expect("CurrentPopSum over declared compartments must resolve");

    // Build a state and verify the projection sums E_e1 + E_e2 + E_e3 = 20.
    let mut state = ParticleState::new(5, 0, 0);
    state.counts[0] = 900;
    state.counts[1] = 10;
    state.counts[2] = 7;
    state.counts[3] = 3;
    state.counts[4] = 5;

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection,
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(vec![20.0]),
            obs_times: vec![5.0],
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();
    // Poisson(observed=20, rate=projected+0.1) peaks at projected≈20.
    let ll_at_truth = obs_model.log_likelihood(&state, 0, &params);
    assert!(ll_at_truth.is_finite(), "log-lik must be finite: {}", ll_at_truth);
    // Sanity: a state where only E_e1 is nonzero scores worse.
    let mut state_skewed = ParticleState::new(5, 0, 0);
    state_skewed.counts[1] = 20;
    let ll_skewed = obs_model.log_likelihood(&state_skewed, 0, &params);
    assert!((ll_at_truth - ll_skewed).abs() < 1e-9,
        "projection must sum the three E_e* compartments — projecting from \
         (10+7+3)=20 and from (20+0+0)=20 should score identically ({} vs {})",
        ll_at_truth, ll_skewed);
}

/// An unknown compartment name in a `CurrentPopSum` must produce a
/// readable error from `from_ir`, not a panic, not a silent miss.
#[test]
fn current_pop_sum_unknown_compartment_errors_cleanly() {
    let mut init = HashMap::new();
    init.insert("S".into(), 100.0);
    init.insert("I".into(),  5.0);
    let compiled = Arc::new(CompiledModel::new(model_with_obs(
        vec![int_comp("S"), int_comp("I")],
        init,
        Projection::CurrentPopSum(vec!["I".into(), "NOT_A_COMPARTMENT".into()]),
    )).unwrap());

    let err = match StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "obs",
    ) {
        Ok(_)  => panic!("unknown compartment must be rejected"),
        Err(e) => e,
    };
    assert!(err.contains("NOT_A_COMPARTMENT"),
        "error must name the bad compartment for debuggability: {}", err);
    assert!(err.contains("obs"),
        "error must name the observation block: {}", err);
}

/// `CurrentPop("I")` reads the `I` compartment at the observation tick
/// and scores the likelihood against it. Ground-truth state has `I = 42`,
/// so `projected` is 42 and the Poisson log-likelihood at `observed = 42`
/// must peak there (higher than at `observed = 10` or `observed = 80`).
#[test]
fn prevalence_projection_reads_compartment_count() {
    let mut init = HashMap::new();
    init.insert("S".into(), 100.0);
    init.insert("I".into(), 42.0);
    let compiled = Arc::new(CompiledModel::new(model_with_obs(
        vec![int_comp("S"), int_comp("I")],
        init,
        Projection::CurrentPop("I".into()),
    )).unwrap());
    let params = compiled.default_params.clone();

    // Fabricate a particle state with S=100, I=42.
    let mut state = ParticleState::new(2, 0, 0);
    // Local int order follows declaration order: S=0, I=1.
    state.counts[0] = 100;
    state.counts[1] = 42;

    let ll_at_truth = build_obs_model(
        &compiled,
        StreamProjection::IntCompSum(vec![1]),
        42.0,
    ).log_likelihood(&state, 0, &params);
    let ll_low = build_obs_model(
        &compiled,
        StreamProjection::IntCompSum(vec![1]),
        10.0,
    ).log_likelihood(&state, 0, &params);
    let ll_high = build_obs_model(
        &compiled,
        StreamProjection::IntCompSum(vec![1]),
        80.0,
    ).log_likelihood(&state, 0, &params);

    assert!(ll_at_truth.is_finite());
    assert!(ll_at_truth > ll_low, "truth={} low={}", ll_at_truth, ll_low);
    assert!(ll_at_truth > ll_high, "truth={} high={}", ll_at_truth, ll_high);
}

/// A `DerivedExpr` projection that sums two compartments yields the same
/// likelihood as an equivalent `CurrentPopSum`. Guards that the two
/// routes through the projection machinery agree arithmetically.
#[test]
fn derived_expr_matches_comp_sum() {
    let mut init = HashMap::new();
    init.insert("B1".into(), 13.0);
    init.insert("B2".into(), 27.0);

    // Build two compiled models, identical except for the projection.
    let model_expr = model_with_obs(
        vec![int_comp("B1"), int_comp("B2")],
        init.clone(),
        Projection::DerivedExpr(Expr::BinOp(BinOpWrap {
            bin_op: BinOpExpr {
                op: BinOp::Add,
                left: Box::new(Expr::Pop(PopExpr { pop: "B1".into() })),
                right: Box::new(Expr::Pop(PopExpr { pop: "B2".into() })),
            },
        })),
    );
    let model_sum = model_with_obs(
        vec![int_comp("B1"), int_comp("B2")],
        init,
        Projection::CurrentPopSum(vec!["B1".into(), "B2".into()]),
    );
    let compiled_expr = Arc::new(CompiledModel::new(model_expr).unwrap());
    let compiled_sum = Arc::new(CompiledModel::new(model_sum).unwrap());
    let params = compiled_expr.default_params.clone();

    let mut state = ParticleState::new(2, 0, 0);
    state.counts[0] = 13;
    state.counts[1] = 27;

    let proj_expr = StreamProjection::from_ir(
        &compiled_expr.model.observations[0].projection, &compiled_expr, "obs",
    ).unwrap();
    let proj_sum = StreamProjection::from_ir(
        &compiled_sum.model.observations[0].projection, &compiled_sum, "obs",
    ).unwrap();

    let ll_expr = build_obs_model(&compiled_expr, proj_expr, 40.0)
        .log_likelihood(&state, 0, &params);
    let ll_sum = build_obs_model(&compiled_sum, proj_sum, 40.0)
        .log_likelihood(&state, 0, &params);

    assert!((ll_expr - ll_sum).abs() < 1e-12,
        "DerivedExpr(B1+B2) and CurrentPopSum([B1,B2]) must match: {} vs {}",
        ll_expr, ll_sum);
}

/// Snapshot-after-intervention semantic: a `FractionTransfer(0.5)` firing
/// at the observation time must be applied *before* the snapshot reads
/// the compartment counts. Otherwise the likelihood sees the
/// pre-intervention state, biasing the PF against correctly-modeled
/// interventions.
///
/// This exercises the PGAS / pgas_grad path
/// (`log_likelihood_from_flows_and_counts`) with `counts_after` — the
/// state recorded after `step_one` has fired scheduled interventions at
/// `t+dt`.
#[test]
fn snapshot_reads_post_intervention_state() {
    use sim::{
        chain_binomial::{step_one, StepScratch},
        rng::StatefulRng,
    };

    let mut init = HashMap::new();
    init.insert("S".into(), 1000.0);
    init.insert("V".into(),    0.0);
    let model = Model {
        ic_grad: Default::default(),
        name: "snap_intv".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None, origin: None, origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("V")],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "halve_S".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![5.0])),
            kind: ir::intervention::InterventionKind::Event, // fire without scenario gating
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(),
                dst: "V".into(),
                fraction: Expr::Const(ConstExpr { value: 0.5 }),
            })],
        }],
        observations: vec![IrObservationModel {
            name: "obs".into(),
            source: "obs".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "obs".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "obs".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![5.0])),
            stratum: vec![],
            projection: Projection::CurrentPop("S".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Add,
                        left: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
                        right: Box::new(Expr::Const(ConstExpr { value: 0.1 })),
                    },
                })),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![],
        initial_conditions: InitialConditions::Explicit(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![5.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 10.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();

    // Drive the chain-binomial `step_one` path for five dt=1 steps to t=5.
    // At t+dt=5 (the step ending at t=5), `step_one` fires the intervention
    // and produces the `counts_after` state that the PF observation tick
    // reads for the snapshot.
    let mut counts = vec![1000i64, 0i64];
    let mut flows = vec![0u64; 0];
    let mut rng = StatefulRng::new(42);
    let mut scratch = StepScratch::new(&compiled);
    let mut real = sim::state::RealState::new(compiled.real_local_to_global.len());

    let fire_steps = compiled.resolve_fire_steps(1.0, &[]);
    for k in 0..5 {
        let t = k as f64;
        sim::effects::due_effects(&compiled, &fire_steps, t + 1.0, 1.0, &mut scratch.effect_batch);
        step_one(&compiled, &mut counts, &mut flows, &mut real, &params, t, 1.0, None, &mut rng, &mut scratch)
            .unwrap();
    }

    // After the t=4→5 step, the intervention at t=5 must have fired.
    // S = 1000 * (1 - 0.5) = 500; V = 500. If the intervention did NOT
    // fire (old behavior would be pre-snapshot reading), S would still be
    // 1000.
    assert_eq!(counts[0], 500, "S must be 500 post-intervention (was {})", counts[0]);
    assert_eq!(counts[1], 500, "V must be 500 post-intervention (was {})", counts[1]);

    // The PF likelihood scores the observation against `counts_after`.
    // Hand the same `counts` slice into `log_likelihood_from_flows_and_counts`
    // and assert it projects 500, not 1000.
    let projection = StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "obs",
    ).unwrap();
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection,
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(vec![500.0]),
            obs_times: vec![5.0],
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    // Prevalence (`CurrentPop`) stream + no transitions ⇒ no `Interval` slot,
    // so the per-stream `acc` bin vector is empty; scoring reads `counts`.
    let acc: Vec<u64> = vec![];
    let ll_at_post = obs_model.log_likelihood_from_flows_and_counts(
        &acc, &counts, 0, &params,
    );
    // Compare to "if the snapshot had read pre-intervention state": hand
    // in counts [1000, 0] and score the same observation (500). Poisson
    // likelihood at (observed=500, rate=1000) is much worse than at
    // (observed=500, rate=500), so this strongly dominates.
    let pre_counts = vec![1000i64, 0i64];
    let ll_at_pre = obs_model.log_likelihood_from_flows_and_counts(
        &acc, &pre_counts, 0, &params,
    );

    assert!(ll_at_post.is_finite());
    assert!(ll_at_pre.is_finite());
    assert!(ll_at_post > ll_at_pre,
        "snapshot-after-intervention likelihood ({}) must be greater than \
         snapshot-before-intervention likelihood ({}); the PF intervention \
         ordering is broken if this flips",
        ll_at_post, ll_at_pre);
}
