//! Byte-identical-loglik property of the prequential recorder (gh#269).
//!
//! The one-step-ahead posterior predictive (`camdl fit predict --horizon
//! one_step`) runs a bootstrap filter per posterior draw with
//! `record_prequential = true`, reading the per-particle predictive samples it
//! captures. That recording MUST NOT perturb the filter's log-likelihood — the
//! predictive draw shares the one RNG-consuming `obs_model.sample(...)` call
//! with the score, and the recorder only copies pre-existing weights. This test
//! pins the property: the same filter, same fixture, same seed, with
//! `record_prequential` off vs on, returns a BIT-IDENTICAL `log_likelihood`.
//!
//! If this ever fails, the one-step predictive path is silently altering
//! inference — exactly the failure mode the design typed out.
//!
//! Fixture mirrors `per_stream_reset.rs`: a single deterministic
//! `inflow @ deterministic(K)` transition observed as `incidence`, so the flow
//! is reproducible and the Normal likelihood is benign. (Even with stochastic
//! draws the property holds — the recorder is pure copy — but a deterministic
//! fixture makes the loglik a fixed value, so a mismatch is unambiguous.)

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{ConstExpr, Expr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::{
        Likelihood, ObservationModel as IrObs, ObservationSchedule,
        NormalLikelihood, Projection,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        bootstrap_filter, dense_cells,
        ChainBinomialProcess, BoundObs, MultiStreamObsModel,
        traits::SMCConfig,
        multi_stream_obs::{StreamProjection, StreamSpec},
    },
};

/// `--> R @ deterministic(K)`, one `inflow` transition observed as incidence
/// (a `Normal(mean = projected)` likelihood). Deterministic so the per-substep
/// flow is exactly `nearbyint(K·dt)`.
fn model(k_per_unit: f64) -> Arc<CompiledModel> {
    let m = Model {
        name: "prequential_invariant".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "inflow".into(),
                stoichiometry: vec![StoichiometryEntry("R".into(), 1)],
                rate: Expr::Const(ConstExpr { value: k_per_unit }),
                metadata: None,
                draw_method: DrawMethod::Deterministic,
                rate_grad: Default::default(),
                lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![ir_incidence_obs("cases")],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "dummy".into(), value: ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0); h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 90.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 90.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

fn ir_incidence_obs(name: &str) -> IrObs {
    IrObs {
        name: name.into(),
        source: name.into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: name.into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("inflow".into()),
        likelihood: Likelihood::Normal(NormalLikelihood {
            mean: Expr::Projected(ProjectedExpr { projected: () }),
            sd: Expr::Const(ConstExpr { value: 50.0 }),
        }),
    }
}

/// Run the bootstrap filter once with the given `record_prequential` flag,
/// everything else held fixed, and return the log-likelihood.
fn run_loglik(record_prequential: bool) -> f64 {
    let k = 10.0;
    let compiled = model(k);
    let times = vec![7.0, 14.0, 21.0, 28.0, 35.0];
    // The true 7-day window flow is 70; use it as the observed value so the
    // Normal likelihood is finite and benign.
    let inflow = compiled.model.transitions.iter()
        .position(|t| t.name == "inflow").unwrap();
    let cases = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![70.0; times.len()]),
        times,
    );
    let obs = MultiStreamObsModel::new(
        BoundObs::bind(vec![cases]).expect("bind").0, compiled.clone()).unwrap();

    let process = ChainBinomialProcess::new(compiled.clone());
    let cfg = SMCConfig {
        n_particles: 64, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let params = compiled.default_params.clone();
    bootstrap_filter(&process, &obs, &params, &cfg, 7)
        .expect("pfilter must run")
        .log_likelihood
}

/// The load-bearing invariant: recording the one-step predictive samples does
/// NOT perturb the filter's log-likelihood. Off vs on must be BIT-IDENTICAL.
#[test]
fn record_prequential_does_not_perturb_loglik() {
    let off = run_loglik(false);
    let on = run_loglik(true);
    assert!(off.is_finite(), "baseline loglik must be finite, got {off}");
    assert_eq!(
        off.to_bits(),
        on.to_bits(),
        "record_prequential must not change the log-likelihood: \
         off = {off} (bits {:#x}), on = {on} (bits {:#x}). The one-step \
         predictive recorder is silently altering inference.",
        off.to_bits(), on.to_bits(),
    );

    // And the recording side actually produced samples (so the test is not
    // vacuously comparing two no-op runs).
    let k = 10.0;
    let compiled = model(k);
    let times = vec![7.0, 14.0, 21.0, 28.0, 35.0];
    let inflow = compiled.model.transitions.iter()
        .position(|t| t.name == "inflow").unwrap();
    let cases = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![70.0; times.len()]),
        times.clone(),
    );
    let obs = MultiStreamObsModel::new(
        BoundObs::bind(vec![cases]).expect("bind").0, compiled.clone()).unwrap();
    let process = ChainBinomialProcess::new(compiled.clone());
    let cfg = SMCConfig {
        n_particles: 64, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: true,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let params = compiled.default_params.clone();
    let res = bootstrap_filter(&process, &obs, &params, &cfg, 7).unwrap();
    let preq = res.prequential.expect("record_prequential=true populates prequential");
    assert_eq!(preq.obs_times.len(), times.len(), "one recorded step per observation");
    assert_eq!(preq.stream_names, vec!["cases".to_string()], "the single bound stream");
    // [obs_idx][stream][particle]: every step holds 64 per-particle samples.
    assert_eq!(preq.per_stream_samples.len(), times.len());
    assert_eq!(preq.per_stream_samples[0][0].len(), 64,
        "each step records one predictive sample per particle");
}
