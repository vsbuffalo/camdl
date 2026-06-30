//! Tests for observation-level parameter support.
//!
//! Verifies that obs model parameters (e.g., NegBinomial dispersion k,
//! reporting rate rho) are correctly evaluated with current params — not
//! baked in at construction time. This was the key bug that the trait
//! refactor fixed: PGAS/PMMH now pass params at call time via
//! ObservationModel::log_likelihood(state, obs_idx, params).

use std::sync::Arc;
use std::collections::HashMap;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, ProjectedExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
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
        dense_cells,
        multi_stream_obs::{MultiStreamObsModel, StreamSpec},
        traits::ObservationModel,
        ParticleState,
    },
};

/// Build a pure death model with obs-level dispersion parameter `k`.
///
/// The model has one transition (death at rate mu*N) and one observation
/// stream (NegBinomial with mean=projected, dispersion=k). The key point
/// is that `k` is a PARAMETER (index into the params array), not a constant.
fn model_with_obs_param() -> (Arc<CompiledModel>, Vec<f64>) {
    let model = Model {
        name: "obs_param_test".into(),
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
                rate: Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                    },
                }),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(), lineage: None,
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
                projection: Projection::CumulativeFlow("death".into()),
                likelihood: Likelihood::NegBinomial(NegBinomialLikelihood {
                    // mean = projected (cumulative deaths)
                    mean: Expr::Projected(ProjectedExpr { projected: () }),
                    // dispersion = k (a PARAMETER, not a constant)
                    dispersion: Expr::Param(ParamExpr { param: "k".into() }),
                }),
            },
        ],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
            Parameter { name: "k".into(), value: ir::parameter::ParamValue::Fixed { value: 10.0 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("N".into(), 100.0); m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 50.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 50.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// T1: Changing obs-level parameter `k` changes the log-likelihood.
///
/// With the old ObsStreamSpec closure API, changing k in the params array
/// had NO EFFECT because k was baked into the closure at construction time.
/// With MultiStreamObsModel, k is evaluated from params at call time.
#[test]
fn test_obs_param_changes_loglik() {
    let (compiled, params) = model_with_obs_param();
    let k_idx = *compiled.param_index.get("k").unwrap();

    // Build obs model with NegBinomial(projected, k) where k is a param
    let obs_times = vec![10.0, 20.0, 30.0];
    let obs_values = vec![8.0, 7.0, 5.0]; // some observed death counts
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs_values),
            obs_times,
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    // Create a particle state with some flow accumulator values. One
    // `FlowSum(vec![0])` stream ⇒ one `acc` slot; the trait scoring reads
    // `acc[0]` (the already-folded bin), so set it to the intended bin (10).
    let mut state = ParticleState::new(1, 1, 1);
    state.flow_accumulators[0] = 10;
    state.acc[0] = 10; // projected = 10

    // Evaluate obs loglik with k=10
    let mut params_k10 = params.clone();
    params_k10[k_idx] = 10.0;
    let ll_k10 = obs_model.log_likelihood(&state, 0, &params_k10);

    // Evaluate obs loglik with k=1 (more overdispersed)
    let mut params_k1 = params.clone();
    params_k1[k_idx] = 1.0;
    let ll_k1 = obs_model.log_likelihood(&state, 0, &params_k1);

    // Evaluate obs loglik with k=100 (nearly Poisson)
    let mut params_k100 = params.clone();
    params_k100[k_idx] = 100.0;
    let ll_k100 = obs_model.log_likelihood(&state, 0, &params_k100);

    // All should be finite
    assert!(ll_k10.is_finite(), "k=10 loglik should be finite, got {}", ll_k10);
    assert!(ll_k1.is_finite(), "k=1 loglik should be finite, got {}", ll_k1);
    assert!(ll_k100.is_finite(), "k=100 loglik should be finite, got {}", ll_k100);

    // They should all be DIFFERENT — the obs model responds to k changes
    assert!((ll_k10 - ll_k1).abs() > 0.01,
        "k=10 and k=1 should give different logliks: {} vs {}", ll_k10, ll_k1);
    assert!((ll_k10 - ll_k100).abs() > 0.001,
        "k=10 and k=100 should give different logliks: {} vs {}", ll_k10, ll_k100);

    eprintln!("obs-level param test: ll(k=1)={:.4}, ll(k=10)={:.4}, ll(k=100)={:.4}",
        ll_k1, ll_k10, ll_k100);
}

/// T2: log_likelihood_from_flows also responds to obs-level params.
///
/// This is the code path used by PGAS (which passes cumulative flows
/// directly, not ParticleState).
#[test]
fn test_obs_param_from_flows() {
    let (compiled, params) = model_with_obs_param();
    let k_idx = *compiled.param_index.get("k").unwrap();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(vec![8.0]),
            obs_times: vec![10.0],
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    // Phase 2a: the flat helper's first arg is now the per-stream `acc` bin.
    // One `FlowSum(vec![0])` stream ⇒ one slot, so `acc[0]` is the bin (10).
    let acc: Vec<u64> = vec![10];

    let mut params_k10 = params.clone();
    params_k10[k_idx] = 10.0;
    let ll_k10 = obs_model.log_likelihood_from_flows(&acc, 0, &params_k10);

    let mut params_k1 = params.clone();
    params_k1[k_idx] = 1.0;
    let ll_k1 = obs_model.log_likelihood_from_flows(&acc, 0, &params_k1);

    assert!(ll_k10.is_finite());
    assert!(ll_k1.is_finite());
    assert!((ll_k10 - ll_k1).abs() > 0.01,
        "PGAS flow-based obs should respond to k: ll(k=10)={} vs ll(k=1)={}", ll_k10, ll_k1);
}

/// T3: Consistency between ParticleState and flow-based evaluation.
///
/// log_likelihood(state, obs_idx, params) should equal
/// log_likelihood_from_flows(state.acc, obs_idx, params) — both read the
/// per-stream `acc` bin (Phase 2a). The trait path unpacks `state.acc`; the
/// flat helper takes `acc` directly.
#[test]
fn test_obs_model_consistency() {
    let (compiled, params) = model_with_obs_param();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(vec![8.0, 7.0]),
            obs_times: vec![10.0, 20.0],
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    // One `FlowSum` stream ⇒ one `acc` slot. Set `acc[0]` to the bin (15); the
    // trait path reads `state.acc`, the flat path takes the same `acc`.
    let mut state = ParticleState::new(1, 1, 1);
    state.flow_accumulators[0] = 15;
    state.acc[0] = 15;

    for obs_idx in 0..2 {
        let ll_state = obs_model.log_likelihood(&state, obs_idx, &params);
        let ll_flows = obs_model.log_likelihood_from_flows(&state.acc, obs_idx, &params);
        assert!((ll_state - ll_flows).abs() < 1e-12,
            "state-based and flow-based logliks must match at obs_idx={}: {} vs {}",
            obs_idx, ll_state, ll_flows);
    }
}
