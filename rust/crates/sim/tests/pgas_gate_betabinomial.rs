//! Regression tests for the observation-gradient preflight (proposal §4.4) at
//! the `run_pgas` boundary. Both cases route a parameter into an observation
//! and assert the fit is ADMITTED — the compiler now emits a real
//! `DerivEntry::Grad` for each, so neither is refused:
//!
//!   * `gh76_pgas_runs_betabinomial_routed_param_with_nuts` — a BetaBinomial
//!     `alpha` param estimated via NUTS RUNS (the BetaBinomial obs-density
//!     gradient is wired).
//!   * `gh180_pgas_admits_parametric_derived_projection_param` — a param
//!     driving a parametric `DerivedExpr` projection RUNS. This is the
//!     inversion of the old C1 fence: the projection is inlined into the
//!     likelihood argument and differentiated (tier-1), so the preflight
//!     admits it instead of refusing every projection param.
//!
//! The complementary refusals — an estimated param reaching an `Unsupported`
//! obs/σ² gradient, or a parametric Binomial/BetaBinomial `n` — live in
//! `pgas_gate_obs_unsupported.rs`.

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
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::BetaBinomial(BetaBinomialLikelihood {
            n, alpha: ir::Diffable::new(alpha), beta: ir::Diffable::new(beta),
        }),
    }
}

/// Build an obs block whose *projection* is a parametric `DerivedExpr`:
/// `proj = scale * <flow>`. The likelihood is an ordinary Poisson over the
/// projected value. Post-gh#180, this is a **tier-1 differentiable** case: the
/// compiler inlines the projection into `rate` (`rate = scale · projected`) and
/// emits `∂rate/∂scale = projected` as a `DerivEntry::Grad`, so the preflight
/// ADMITS the `scale` parameter. The `rate_grad` below mirrors that emitted
/// gradient (the FD checks in `gradient_check_obs.rs` pin the numeric value).
fn build_parametric_derived_proj_block(scale_param: &str) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;

    // projection = scale_param * I — a DerivedExpr that depends on a parameter;
    // ∂(projected)/∂(scale) ≠ 0 and IS propagated (the fix).
    //
    // The state term is a compartment reference, NOT `Expr::Projected`. A
    // projection has nothing to project from — `ResolveCtx` supplies
    // `projected` only inside a LIKELIHOOD — so `ResolvedExpr::Projected`
    // floors to 0.0 there (`resolved_expr.rs`), and a `scale · projected`
    // projection is identically zero. That made `Poisson(rate = 0)` score
    // `-inf` against every positive observation: the fit this fixture claims
    // to run never had a finite likelihood. It is also IR the compiler cannot
    // emit — `projected = scale * prevalence(I)` inlines the compartment,
    // which is what this now mirrors.
    let projection_expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Mul,
            left: Box::new(Expr::Param(ParamExpr { param: scale_param.into() })),
            right: Box::new(Expr::pop("I")),
        },
    });

    // likelihood: Poisson(rate = projected). The projected value already
    // carries the scale factor, so the param reaches the likelihood only
    // through the projection.
    let rate = Expr::Projected(ProjectedExpr { projected: () });

    // Compiler-mirroring emitted gradient: inlining the projection gives
    // `rate = scale · I`, so `∂rate/∂scale = I` — a real `Grad`, NOT an
    // `Unsupported`. The preflight admits a `Grad`. (`projected` would be
    // `scale · I`, an off-by-`scale` gradient, and it is the LIKELIHOOD's
    // `projected` — the very value being differentiated.)
    let rate_grad = std::collections::HashMap::from([(
        scale_param.to_string(),
        ir::deriv::DerivEntry::Grad(Expr::pop("I")),
    )]);

    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::DerivedExpr(projection_expr),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable { expr: rate, grad: rate_grad, proj_grad: None } }),
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
            aux: vec![],
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
            perturb_only_at_t0: false,
        },
    ];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        ancestor_sampling: true,
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
fn gh180_pgas_admits_parametric_derived_projection_param() {
    // Post-gh#180: a parametric `DerivedExpr` projection is a tier-1
    // differentiable case. The compiler inlines the projection into the
    // likelihood argument and emits `∂rate/∂scale` as a `DerivEntry::Grad`, so
    // the preflight ADMITS the `scale` param (the inversion of the old C1
    // fence, which refused every param in a parametric projection). This fit
    // must now RUN.
    let mut model = host_model();
    model.parameters.push(ir::parameter::Parameter { name: "scale_obs".into(), value: ir::parameter::ParamValue::Estimated { init: Some(1.0), bounds: Some((0.1, 10.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: Some(ir::parameter::ParamKind::Positive), param_dim: None });
    model.observations = vec![build_parametric_derived_proj_block("scale_obs")];

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = params_from_compiled(&compiled);

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // The projection is `scale · I` — a state snapshot, so the observations are
    // weekly PREVALENCE, read off the truth trajectory at the observation
    // instant. (A flow sum would be scored against a quantity the projection
    // never computes.) The window stops at day 42, while the epidemic is still
    // large: past the peak `I` decays toward 0, and a week where the reference
    // draw reaches I = 0 against a positive observation is `Poisson(rate = 0)`
    // — a `-inf` no trajectory at these parameters can repair.
    let i_global = compiled.model.compartments.iter().position(|c| c.name == "I")
        .expect("host model has an I compartment");
    let i_local = compiled.global_to_int[i_global]
        .expect("I is an integer compartment");
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 && t <= 42.0 {
            obs.push(Observation { time: t, value: rec.counts_after[i_local] as f64 });
        }
    }

    // The stream projection is the resolved DerivedExpr (scale * I).
    let stream_proj = StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "weekly_cases",
    ).unwrap();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: stream_proj,
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
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
            perturb_only_at_t0: false,
        },
    ];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        ancestor_sampling: true,
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

    // Must NOT be refused: the parametric projection now carries a real emitted
    // `DerivEntry::Grad`, so the preflight admits it (the inversion of the old
    // C1 fence). A `Validation` error naming the projection would be the old
    // refusal leaking back.
    match result {
        Ok(_) => {}
        Err(SimError::Validation(msg))
            if msg.contains("scale_obs")
                || msg.contains("DerivedExpr")
                || msg.contains("projection")
                || msg.contains("could not emit") =>
        {
            panic!(
                "preflight must NOT fence a parametric DerivedExpr projection param \
                 now that it is differentiated (gh#180); got refusal: {}", msg
            );
        }
        Err(e) => panic!("run_pgas failed for an unrelated reason: {:?}", e),
    }
}
