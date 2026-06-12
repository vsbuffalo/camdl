//! Regression tests for the audit-C1 preflight gate after the gh#76 residual
//! (BetaBinomial obs-density gradient) landed.
//!
//! History: the C1 gate fenced estimation of any parameter whose reachability
//! graph traversed an *uncovered* observation-gradient arm. There were two:
//!
//!   1. `BetaBinomial`: `eval_likelihood_resolved_grad` was a no-op.
//!   2. Parametric `DerivedExpr` projections: the chain-rule term
//!      ∂L/∂(projected) · ∂(projected)/∂θ is omitted.
//!
//! The gh#76 residual wired the BetaBinomial gradient
//! (`beta_binomial_logpmf_grad` + the `eval_likelihood_resolved_grad` arm),
//! so arm 1 is now covered and the gate no longer fences it. Arm 2 is still
//! a documented no-op, so the gate must still fire for it.
//!
//! These two tests pin both halves of that transition:
//!
//!   * `gh76_pgas_runs_betabinomial_routed_param_with_nuts` — a fit that
//!     estimates a BetaBinomial-bound param via NUTS now RUNS (no gate
//!     error). This is the *inversion* of the original assertion.
//!   * `gh76_pgas_refuses_parametric_derived_projection_param` — the gate
//!     STILL refuses a param routed through a parametric `DerivedExpr`
//!     projection (the remaining uncovered arm). This preserves coverage of
//!     the live fence.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::error::SimError;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::BoundObs;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamSpec, StreamProjection};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

fn set_param_defaults(model: &mut ir::Model, defaults: &[(&str, f64)]) {
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            } else {
                p.value = p.value.with_value(0.5);
            }
        }
    }
}

/// Build a BetaBinomial obs stream over the infection flow with `alpha` that
/// references a new parameter `alpha_obs`. The literal `n = projected`
/// (the cum-flow over the interval) is irrelevant for the gradient path —
/// what matters is that `alpha` depends on a parameter.
fn build_betabinomial_obs_block(alpha_param: &str) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;

    let n = Expr::Projected(ProjectedExpr { projected: () });
    let alpha = Expr::Param(ParamExpr { param: alpha_param.into() });
    let beta = Expr::Const(ConstExpr { value: 1.0 });

    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        projection: Projection::CumulativeFlow("infection".into()),
        likelihood: Likelihood::BetaBinomial(BetaBinomialLikelihood { n, alpha, beta }),
    }
}

/// Build an obs block whose *projection* is a parametric `DerivedExpr`:
/// `proj = scale * <flow>`. The likelihood is an ordinary Poisson over the
/// projected value, so the only uncovered-gradient path is the projection
/// itself (the `scale` parameter). This is the live C1 fence.
fn build_parametric_derived_proj_block(scale_param: &str) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;

    // projection = scale_param * projected — a DerivedExpr that depends on a
    // parameter, so ∂(projected)/∂(scale) ≠ 0 but is not propagated.
    let projection_expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Mul,
            left: Box::new(Expr::Param(ParamExpr { param: scale_param.into() })),
            right: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
        },
    });

    // likelihood: Poisson(rate = projected). The projected value already
    // carries the scale factor, so the param reaches the likelihood only
    // through the projection.
    let rate = Expr::Projected(ProjectedExpr { projected: () });

    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        projection: Projection::DerivedExpr(projection_expr),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate }),
    }
}

/// Shared host-model setup: SIR-overdispersion fixture with sane defaults.
fn host_model() -> ir::Model {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", 0.1),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    model
}

fn params_from_compiled(compiled: &CompiledModel) -> Vec<f64> {
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    params
}

#[test]
fn gh76_pgas_runs_betabinomial_routed_param_with_nuts() {
    // The gh#76 residual wired the BetaBinomial obs-density gradient, so the
    // C1 gate no longer fences a BetaBinomial-routed parameter. This fit must
    // now RUN (return Ok) — the inversion of the original gate assertion.
    let mut model = host_model();
    model.parameters.push(ir::parameter::Parameter { name: "alpha_obs".into(), value: ir::parameter::ParamValue::Estimated { init: Some(2.0), bounds: Some((0.01, 100.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: Some(ir::parameter::ParamKind::Positive), param_dim: None });
    model.observations = vec![build_betabinomial_obs_block("alpha_obs")];

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = params_from_compiled(&compiled);

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Synthetic weekly obs by summing the infection flow into windows, halved
    // so the BetaBinomial constraint k ≤ n holds.
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: (cum_infection as f64 * 0.5).round() });
            cum_infection = 0;
        }
    }

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![
        EstimatedParam {
            name: "alpha_obs".into(),
            index: compiled.param_index["alpha_obs"],
            initial: 2.0,
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.01, hi: 100.0 },
            lower: 0.01,
            upper: 100.0,
            rw_sd_auto: false,
            ivp: false,
        },
    ];
    let priors = vec![Prior::Flat];

    let config = PGASConfig {
        n_particles: 50,
        n_sweeps: 3,
        burn_in: 1,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 4,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model, 12345, None, None, "gate_betabinomial".into(),
    );

    // Must NOT be refused by the C1 gate. The BetaBinomial gradient arm is
    // covered, so the run proceeds. (We assert Ok rather than just
    // "not-a-gate-error" because the fit is well-posed and small.)
    match result {
        Ok(_) => {}
        Err(SimError::Validation(msg)) if msg.contains("BetaBinomial")
            || msg.contains("does not cover") => {
            panic!(
                "C1 gate must NOT fence a BetaBinomial-routed param now that the \
                 gradient is wired (gh#76 residual). Got gate error: {}", msg
            );
        }
        Err(e) => panic!("run_pgas failed for an unrelated reason: {:?}", e),
    }
}

#[test]
fn gh76_pgas_refuses_parametric_derived_projection_param() {
    // The remaining uncovered obs-gradient arm: a parametric `DerivedExpr`
    // projection. The chain-rule term ∂L/∂(projected)·∂(projected)/∂θ is
    // omitted, so estimating the projection's `scale` param via NUTS would
    // be a silent-zero gradient. The C1 gate must STILL fire here.
    let mut model = host_model();
    model.parameters.push(ir::parameter::Parameter { name: "scale_obs".into(), value: ir::parameter::ParamValue::Estimated { init: Some(1.0), bounds: Some((0.1, 10.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: Some(ir::parameter::ParamKind::Positive), param_dim: None });
    model.observations = vec![build_parametric_derived_proj_block("scale_obs")];

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = params_from_compiled(&compiled);

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum_infection as f64 });
            cum_infection = 0;
        }
    }

    // The stream projection is the resolved DerivedExpr (scale * flow).
    let stream_proj = StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "weekly_cases",
    ).unwrap();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: stream_proj,
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![
        EstimatedParam {
            name: "scale_obs".into(),
            index: compiled.param_index["scale_obs"],
            initial: 1.0,
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.1, hi: 10.0 },
            lower: 0.1,
            upper: 10.0,
            rw_sd_auto: false,
            ivp: false,
        },
    ];
    let priors = vec![Prior::Flat];

    let config = PGASConfig {
        n_particles: 50,
        n_sweeps: 5,
        burn_in: 2,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 4,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model, 12345, None, None, "gate_derived_proj".into(),
    );

    match result {
        Err(SimError::Validation(msg)) => {
            // The gate must (a) name the blocked parameter and (b) point at
            // the uncovered parametric-projection arm.
            assert!(msg.contains("scale_obs"),
                "error must name the blocked parameter; got: {}", msg);
            assert!(msg.contains("DerivedExpr") || msg.contains("projection"),
                "error must name the uncovered parametric-projection arm; got: {}", msg);
            eprintln!("[gate test] saw expected validation error:\n  {}", msg);
        }
        Err(e) => panic!("expected SimError::Validation, got: {:?}", e),
        Ok(_) => panic!(
            "run_pgas must refuse to estimate a param routed through a parametric \
             DerivedExpr projection via NUTS — the projection chain-rule term is a \
             documented no-op (silent-zero gradient)."
        ),
    }
}
