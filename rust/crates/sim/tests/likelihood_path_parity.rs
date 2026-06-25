//! gh#139: `MultiStreamObsModel` has two independent likelihood
//! summation loops — the trait `log_likelihood` (PF/IF2/PMMH) and the
//! inherent `log_likelihood_from_flows_and_counts` (PGAS). They must
//! agree on the same `(flows, counts, data)`, since `ParticleState` is
//! exactly `{ counts, flow_accumulators }`.
//!
//! This is a CHARACTERIZATION test. After the gh#139 unification (the
//! trait method delegates to the flat method) the agreement is
//! structural, but the test stays as a guard so a future re-split — the
//! GH#6 / incident-2026-04-22 class of bug, which has produced a ~100×
//! log-likelihood divergence twice — fails loudly here. Verified
//! non-vacuous: injecting a `*2.0` divergence into the trait delegate
//! makes this test fail.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ConstExpr, ParamExpr, PopExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    observation::{
        ObservationModel as IrObs, ObservationSchedule, Projection,
        Likelihood, NegBinomialLikelihood,
    },
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        BoundObs,
        ParticleState,
        dense_cells,
        traits::ObservationModel,
        multi_stream_obs::{MultiStreamObsModel, StreamSpec, StreamProjection},
    },
};

/// S,I,R with a *state-dependent* observation likelihood:
/// `neg_binomial(mean = rho * I, r = k)`. The mean references compartment
/// `I` via a `Pop` node, so the likelihood eval reads `counts` — exactly
/// the GH#6 case where a zero scratch silently broke one path. Prevalence
/// projection (`IntCompSum` over `I`) reads counts too.
fn model() -> Arc<CompiledModel> {
    let m = Model {
        name: "likelihood_path_parity".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "recovery".into(),
                stoichiometry: vec![
                    StoichiometryEntry("I".into(), -1),
                    StoichiometryEntry("R".into(), 1),
                ],
                rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                }}),
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![
            IrObs {
                name: "cases".into(),
                source: "cases".into(),
                columns: vec![
                    ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                    ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ],
                scored: "cases".into(),
                emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: Projection::CurrentPop("I".into()),
                likelihood: Likelihood::NegBinomial(NegBinomialLikelihood {
                    // mean = rho * I  (Pop ref → reads counts)
                    mean: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "rho".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                    }}),
                    dispersion: Expr::Const(ConstExpr { value: 5.0 }),
                }),
            },
        ],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None, doc: None },
            Parameter { name: "rho".into(), value: ir::parameter::ParamValue::Fixed { value: 0.3 }, param_kind: None, param_dim: None, doc: None },
            Parameter { name: "k".into(), value: ir::parameter::ParamValue::Fixed { value: 5.0 }, param_kind: None, param_dim: None, doc: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("S".into(), 950.0); h.insert("I".into(), 40.0); h.insert("R".into(), 10.0); h
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
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

#[test]
fn pf_and_pgas_likelihood_paths_agree() {
    let compiled = model();
    let i_idx = compiled.global_to_int[compiled.comp_index["I"]]
        .expect("I is an integer compartment");

    let spec = StreamSpec {
        ir_model: compiled.model.observations[0].clone(),
        projection: StreamProjection::IntCompSum(vec![i_idx]), // prevalence of I
        observations: dense_cells(vec![12.0, 30.0]),
        obs_times: vec![1.0, 5.0],
        aux: vec![],
    };
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![spec]).unwrap().0, compiled.clone()).unwrap();

    // Non-zero in BOTH fields so the identity exercises the full
    // ParticleState, not a degenerate all-zero case.
    let counts = vec![950i64, 40, 10];  // S, I, R
    let flows  = vec![7u64];            // a non-empty flow vector (recovery)
    let params = compiled.default_params.clone();

    // Prevalence stream (`IntCompSum` over I) ⇒ no `Interval` slot, so `acc` is
    // empty; both paths score from `counts`. `flow_accumulators` is kept
    // populated to prove neither path reads it for a prevalence projection.
    let state = ParticleState {
        counts: counts.clone(),
        flow_accumulators: flows.clone(),
        acc: vec![],
    };

    for obs_idx in 0..2 {
        // PF/IF2/PMMH path (trait), reads `state.acc`:
        let via_state = obs_model.log_likelihood(&state, obs_idx, &params);
        // PGAS path (flat arrays), takes the per-stream `acc` (empty here):
        let via_flat =
            obs_model.log_likelihood_from_flows_and_counts(&[], &counts, obs_idx, &params);

        assert!(via_state.is_finite(),
            "obs {obs_idx}: trait path must be finite, got {via_state}");
        assert!(via_flat.is_finite(),
            "obs {obs_idx}: flat path must be finite, got {via_flat}");
        // The load-bearing invariant: the two seams agree exactly.
        assert_eq!(via_state, via_flat,
            "obs {obs_idx}: PF/IF2 (trait) and PGAS (flat) likelihood paths \
             diverged — gh#139 / the GH#6 dual-loop class. state={via_state} flat={via_flat}");
    }

    // Negative control: the two observation indices have different data
    // (12 vs 30) against the same projected mean, so the likelihood must
    // actually differ — proves the test isn't passing on a trivial constant.
    let ll0 = obs_model.log_likelihood(&state, 0, &params);
    let ll1 = obs_model.log_likelihood(&state, 1, &params);
    assert!((ll0 - ll1).abs() > 1e-9,
        "different observed data must score differently (non-vacuous guard): {ll0} vs {ll1}");
}
