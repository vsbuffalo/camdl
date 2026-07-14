//! Finite-difference regression tests for the observation-density gradient
//! term in `complete_data_loglik_grad` (gh#76).
//!
//! Each test:
//!   1. Loads a small SEIR/SIR model with an observation block that uses
//!      one of {NegBin, Poisson, discretized-Normal}.
//!   2. Simulates a reference trajectory at fixed params.
//!   3. Synthesizes obs values via the obs-model mean (so observations land
//!      near the likelihood's mode — out of the helpers' tail-precision floor).
//!   4. Compares the analytic gradient (via the new
//!      `MultiStreamObsModel::log_likelihood_grad_from_flows_and_counts`
//!      method) against a central finite difference of `complete_data_loglik`.
//!
//! Tolerance: 1e-4 relative. Observed agreement is 1e-7 to 1e-11 on
//! near-mode observations.
//!
//! These tests bypass `pgas::run_pgas` (now ungated for obs params) by
//! calling `complete_data_loglik_grad` directly.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{IVPMapping, simulate_reference, complete_data_loglik, build_obs_at_substep};
use sim::inference::pgas_grad::complete_data_loglik_grad;
use sim::inference::{BoundObs, MultiStreamObsModel, dense_cells};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, eval_stream_projection};
use sim::inference::particle_filter::Observation;
use sim::rng::StatefulRng;
use sim::state::{IntState, RealState};

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

/// A single-entry compiler-mirroring gradient map `{param: Grad(e)}` for the
/// programmatic obs-model builders below. Under P4 the runtime consumes the
/// emitted `*_grad` map (not a runtime differentiator), so these hand-built
/// likelihoods must carry the same `∂arg/∂θ` the OCaml autodiff would emit for
/// the same argument — the FD check is the ground-truth validator of that
/// mirror (a wrong entry makes the FD fail).
fn grad1(param: &str, e: ir::expr::Expr) -> std::collections::HashMap<String, ir::deriv::DerivEntry> {
    std::collections::HashMap::from([(param.to_string(), ir::deriv::DerivEntry::Grad(e))])
}

/// `∂arg/∂θ = 1` for an argument that is exactly `Param(θ)` (e.g. `dispersion = k`,
/// `p = rho`, `alpha = a_obs`) — mirrors the compiler, which folds the
/// derivative of a bare parameter to `Const 1.0`.
fn const1() -> ir::expr::Expr {
    ir::expr::Expr::Const(ir::expr::ConstExpr { value: 1.0 })
}

fn build_obs_model(
    compiled: Arc<CompiledModel>,
    obs_times: &[f64],
    per_stream_data: Vec<Vec<f64>>,
) -> MultiStreamObsModel {
    let model = compiled.model.clone();
    let specs: Vec<StreamSpec> = model.observations.iter().enumerate().map(|(si, om)| {
        let projection = StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap();
        StreamSpec {
            projection,
            ir_model: om.clone(),
            observations: dense_cells(per_stream_data[si].clone()),
            obs_times: obs_times.to_vec(),
            aux: vec![],
        }
    }).collect();
    MultiStreamObsModel::new(BoundObs::bind(specs).unwrap().0, compiled).unwrap()
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

fn build_params_and_names(compiled: &CompiledModel) -> (Vec<f64>, Vec<String>) {
    let n = compiled.param_index.len();
    let mut params = vec![0.0; n];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    let mut names = vec![String::new(); n];
    for p in &compiled.model.parameters {
        names[compiled.param_index[p.name.as_str()]] = p.name.clone();
    }
    (params, names)
}

/// Walk a recorded trajectory and produce one synthetic observed value
/// per (obs_time, stream). For each stream, the observed value is the
/// obs-model *mean* evaluated at the current params (e.g. rho * projected
/// for a NegBin/Normal model with rho-scaled mean). This places observations
/// near the likelihood's mode — the regime where each per-distribution
/// gradient helper is most accurate, so FD vs analytic comparisons aren't
/// limited by the helper's tail-precision floor.
fn project_trajectory_to_obs(
    compiled: &Arc<CompiledModel>,
    trajectory: &sim::inference::pgas::PGASTrajectory,
    obs_substep_indices: &[usize],
    params: &[f64],
    dt: f64,
) -> Vec<Vec<f64>> {
    let n_streams = compiled.model.observations.len();
    let n_tr = compiled.model.transitions.len();
    let mut per_stream: Vec<Vec<f64>> = (0..n_streams).map(|_| Vec::new()).collect();
    let t_start = compiled.model.simulation.t_start;
    let real_s = RealState::new(compiled.real_local_to_global.len());

    let projections: Vec<StreamProjection> = compiled.model.observations.iter()
        .map(|om| StreamProjection::from_ir(&om.projection, compiled, &om.name).unwrap())
        .collect();

    let resolved_lhs: Vec<sim::resolved_expr::ResolvedLikelihood> = compiled.model.observations
        .iter()
        .map(|om| {
            use sim::resolved_expr::{resolve_likelihood, ResolveCtx};
            use ir::table::OobPolicy;
            let table_meta: Vec<(OobPolicy, usize)> = compiled.model.tables.iter()
                .zip(&compiled.table_values_cache)
                .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
                .collect();
            let ctx = ResolveCtx {
                comp_index: &compiled.comp_index,
                param_index: &compiled.param_index,
                time_func_index: &compiled.time_func_index,
                table_index: &compiled.table_index,
                binding_index: &compiled.binding_index,
                per_eval_index: &compiled.per_eval_index,
                global_to_int: &compiled.global_to_int,
                global_to_real: &compiled.global_to_real,
                table_meta: &table_meta,
            };
            resolve_likelihood(&om.likelihood, &ctx).unwrap()
        })
        .collect();

    let mut cum_flows: Vec<u64> = vec![0; n_tr];
    let mut next_obs = 0;

    for (s, rec) in trajectory.substeps.iter().enumerate() {
        for (i, &f) in rec.flows.iter().enumerate() { cum_flows[i] += f; }

        if next_obs < obs_substep_indices.len() && obs_substep_indices[next_obs] == s {
            let t_obs = t_start + ((s + 1) as f64) * dt;
            let int_s = IntState::from_vec(rec.counts_after.clone());
            for si in 0..n_streams {
                let v = eval_stream_projection(
                    &projections[si], &cum_flows, &rec.counts_after,
                    params, compiled, &real_s, t_obs,
                );
                let mean = obs_mean_for_likelihood(
                    &resolved_lhs[si], t_obs, v, params, compiled, &int_s, &real_s,
                );
                per_stream[si].push(mean.max(0.0).round());
            }
            cum_flows.fill(0);
            next_obs += 1;
        }
    }
    per_stream
}

/// Like `project_trajectory_to_obs` but shifts each synthetic obs by
/// `tail_sigmas · sd` (using a likelihood-specific sd) — places observations
/// in the tail of the obs likelihood instead of at the mode.
///
/// gh#76 cleanup: this exercises the regime where the prior φ-difference
/// gradient denominator on `discretized_normal_logpmf_grad` collapsed to
/// noise; after the audit-H2 port the gradient should match FD at 1e-4.
fn project_trajectory_to_obs_shifted(
    compiled: &Arc<CompiledModel>,
    trajectory: &sim::inference::pgas::PGASTrajectory,
    obs_substep_indices: &[usize],
    params: &[f64],
    dt: f64,
    tail_sigmas: f64,
) -> Vec<Vec<f64>> {
    let n_streams = compiled.model.observations.len();
    let n_tr = compiled.model.transitions.len();
    let mut per_stream: Vec<Vec<f64>> = (0..n_streams).map(|_| Vec::new()).collect();
    let t_start = compiled.model.simulation.t_start;
    let real_s = RealState::new(compiled.real_local_to_global.len());

    let projections: Vec<StreamProjection> = compiled.model.observations.iter()
        .map(|om| StreamProjection::from_ir(&om.projection, compiled, &om.name).unwrap())
        .collect();

    let resolved_lhs: Vec<sim::resolved_expr::ResolvedLikelihood> = compiled.model.observations
        .iter()
        .map(|om| {
            use sim::resolved_expr::{resolve_likelihood, ResolveCtx};
            use ir::table::OobPolicy;
            let table_meta: Vec<(OobPolicy, usize)> = compiled.model.tables.iter()
                .zip(&compiled.table_values_cache)
                .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
                .collect();
            let ctx = ResolveCtx {
                comp_index: &compiled.comp_index,
                param_index: &compiled.param_index,
                time_func_index: &compiled.time_func_index,
                table_index: &compiled.table_index,
                binding_index: &compiled.binding_index,
                per_eval_index: &compiled.per_eval_index,
                global_to_int: &compiled.global_to_int,
                global_to_real: &compiled.global_to_real,
                table_meta: &table_meta,
            };
            resolve_likelihood(&om.likelihood, &ctx).unwrap()
        })
        .collect();

    let mut cum_flows: Vec<u64> = vec![0; n_tr];
    let mut next_obs = 0;

    for (s, rec) in trajectory.substeps.iter().enumerate() {
        for (i, &f) in rec.flows.iter().enumerate() { cum_flows[i] += f; }

        if next_obs < obs_substep_indices.len() && obs_substep_indices[next_obs] == s {
            let t_obs = t_start + ((s + 1) as f64) * dt;
            let int_s = IntState::from_vec(rec.counts_after.clone());
            for si in 0..n_streams {
                let v = eval_stream_projection(
                    &projections[si], &cum_flows, &rec.counts_after,
                    params, compiled, &real_s, t_obs,
                );
                let mean = obs_mean_for_likelihood(
                    &resolved_lhs[si], t_obs, v, params, compiled, &int_s, &real_s,
                );
                let sd = obs_sd_for_likelihood(
                    &resolved_lhs[si], t_obs, v, params, compiled, &int_s, &real_s,
                );
                let shifted = (mean + tail_sigmas * sd).max(0.0).round();
                per_stream[si].push(shifted);
            }
            cum_flows.fill(0);
            next_obs += 1;
        }
    }
    per_stream
}

/// Per-likelihood standard deviation at the obs mean. Used to shift obs into
/// the tail by k · sd in `project_trajectory_to_obs_shifted`. The numbers
/// don't need to be exact — they just need to put the obs in a regime that
/// exercises the gradient's tail precision.
fn obs_sd_for_likelihood(
    lh: &sim::resolved_expr::ResolvedLikelihood,
    t: f64,
    projected: f64,
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
) -> f64 {
    use sim::propensity::EvalCtx;
    use sim::resolved_expr::{ResolvedLikelihood, eval_resolved};
    let ctx = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0,
        projected: Some(proj), aux: None, int_float_override: None, per_eval: None,
    };
    match lh {
        ResolvedLikelihood::NegBinomial { mean, dispersion, .. } => {
            // var = μ + μ²/k → sd = √(μ + μ²/k)
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected)).max(1e-30);
            (m + m * m / k).max(0.0).sqrt()
        }
        ResolvedLikelihood::Normal { sd, .. } => eval_resolved(sd, &ctx(projected)),
        ResolvedLikelihood::Poisson { rate, .. } => eval_resolved(rate, &ctx(projected)).max(0.0).sqrt(),
        ResolvedLikelihood::Binomial { n, p, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected)).clamp(0.0, 1.0);
            (n_val * p_val * (1.0 - p_val)).max(0.0).sqrt()
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let a = eval_resolved(alpha, &ctx(projected)).max(1e-30);
            let b = eval_resolved(beta, &ctx(projected)).max(1e-30);
            let denom = (a + b).max(1e-30);
            let p = a / denom;
            let var = n_val * p * (1.0 - p) * (a + b + n_val) / (a + b + 1.0);
            var.max(0.0).sqrt()
        }
        ResolvedLikelihood::Bernoulli { p, .. } => {
            let p_val = eval_resolved(p, &ctx(projected)).clamp(0.0, 1.0);
            (p_val * (1.0 - p_val)).max(0.0).sqrt()
        }
        ResolvedLikelihood::ZeroInflatedNegBinomial { .. } => {
            unreachable!("zero-inflated NB is non-differentiable; the obs gradient check does not cover it")
        }
    }
}

/// Local re-implementation of `eval_obs_mean_resolved` — the obs_model
/// helper is `pub(crate)` and not exposed for external test code.
fn obs_mean_for_likelihood(
    lh: &sim::resolved_expr::ResolvedLikelihood,
    t: f64,
    projected: f64,
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
) -> f64 {
    use sim::propensity::EvalCtx;
    use sim::resolved_expr::{ResolvedLikelihood, eval_resolved};
    let ctx = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0,
        projected: Some(proj), aux: None, int_float_override: None, per_eval: None,
    };
    match lh {
        ResolvedLikelihood::NegBinomial { mean, .. } => eval_resolved(mean, &ctx(projected)),
        ResolvedLikelihood::Normal { mean, .. } => eval_resolved(mean, &ctx(projected)),
        ResolvedLikelihood::Poisson { rate, .. } => eval_resolved(rate, &ctx(projected)),
        ResolvedLikelihood::Binomial { n, p, .. } => {
            eval_resolved(n, &ctx(projected)) * eval_resolved(p, &ctx(projected))
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let a = eval_resolved(alpha, &ctx(projected));
            let b = eval_resolved(beta, &ctx(projected));
            let denom = (a + b).max(1e-300);
            n_val * (a / denom)
        }
        ResolvedLikelihood::Bernoulli { p, .. } => eval_resolved(p, &ctx(projected)),
        ResolvedLikelihood::ZeroInflatedNegBinomial { .. } => {
            unreachable!("zero-inflated NB is non-differentiable; the obs gradient check does not cover it")
        }
    }
}

fn fd_check(
    compiled: &Arc<CompiledModel>,
    trajectory: &sim::inference::pgas::PGASTrajectory,
    observations: &[Observation],
    obs_model: &MultiStreamObsModel,
    params: &[f64],
    param_names: &[String],
    estimated_indices: &[usize],
    params_to_check: &[usize],
    dt: f64,
    rel_tol: f64,
    name: &str,
) {
    let d = estimated_indices.len();
    let n_model_params = compiled.model.parameters.len();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; n_model_params];
    for (est_idx, &model_idx) in estimated_indices.iter().enumerate() {
        model_to_estimated[model_idx] = Some(est_idx);
    }
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(observations, compiled.model.simulation.t_start, dt).unwrap();
    let estimated_to_model: Vec<usize> = estimated_indices.to_vec();

    let (ll, grad) = complete_data_loglik_grad(
        compiled, trajectory, params, observations, dt,
        obs_model, &ivp_mappings,
        d, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "[{}] log-likelihood must be finite, got {}", name, ll);
    eprintln!("[{}] LL = {:.4}", name, ll);

    for &est_idx in params_to_check {
        let model_idx = estimated_indices[est_idx];
        let p_val = params[model_idx];
        let eps = (1e-5 * p_val.abs()).max(1e-8);

        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[model_idx] += eps;
        p_minus[model_idx] -= eps;

        let ll_plus = complete_data_loglik(
            compiled, trajectory, &p_plus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            compiled, trajectory, &p_minus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let analytic = grad[est_idx];
        let rel_err = if fd.abs() > 1e-10 {
            (analytic - fd).abs() / fd.abs()
        } else {
            (analytic - fd).abs()
        };

        eprintln!(
            "[{}] d(ll)/d({:12}) = {:14.6e} (analytic) vs {:14.6e} (fd), rel_err = {:.2e}",
            name, param_names[model_idx], analytic, fd, rel_err
        );

        assert!(
            rel_err < rel_tol,
            "[{}] gradient mismatch for {}: analytic={:.6e}, fd={:.6e}, rel_err={:.2e} (tol={:.0e})",
            name, param_names[model_idx], analytic, fd, rel_err, rel_tol
        );
    }
}

fn obs_substep_indices_regular(
    t_start: f64, dt: f64, n_substeps: usize,
    obs_start: f64, obs_step: f64, obs_end: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut indices = Vec::new();
    let mut times = Vec::new();
    let mut t = obs_start;
    while t <= obs_end + 1e-9 {
        let s_f = (t - t_start) / dt - 1.0;
        let s = s_f.round() as i64;
        if s >= 0 && (s as usize) < n_substeps {
            indices.push(s as usize);
            times.push(t);
        }
        t += obs_step;
    }
    (indices, times)
}

#[test]
fn gh76_negbin_obs_grad_matches_fd() {
    // SEIR with NegBinomial(rho * incidence, k) + Bernoulli(p_detect).
    // Estimate rho, k, p_detect plus a rate param (beta) as regression
    // check that the rate gradient is unaffected.
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    // multi-stream constraint: all streams share obs_times. weekly_cases is
    // every 7 days, detection every 14 days → intersection every 14 days.
    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 14.0, 14.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let rho_idx = compiled.param_index["rho"];
    let k_idx = compiled.param_index["k"];
    let p_detect_idx = compiled.param_index["p_detect"];
    let beta_idx = compiled.param_index["beta"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        &[rho_idx, k_idx, p_detect_idx, beta_idx],
        dt, 1e-4,
        "gh76_negbin_obs",
    );
}

#[test]
fn licm_hoisting_kernel_grad_matches_fd() {
    // gh#284 coverage: the DIRECT LICM-on gradient check. The existing FD tests
    // load `seir_observations` (linear rate, 0 per-eval bindings), so LICM-on
    // gradient correctness rested on COMPOSING two suites — the A/B gate
    // (on == off) and these FD checks (off == truth). This test closes the gap:
    // a fixture whose infection rate carries a param-only transcendental kernel
    // `beta * exp(-kappa)` that LICM actually hoists, checked FD-vs-analytic on
    // `kappa` itself — whose gradient routes through a `per_eval_binding`
    // (`__licm_1 = beta * exp(-kappa) * -1`). on == truth, directly.
    let mut model = load_model("tests/fixtures/licm_grad_fd.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1), ("kappa", 0.4),
        ("rho", 0.5), ("k", 5.0), ("N0", 10000.0), ("I0", 10.0),
    ]);
    // (t_end = 60 days comes from the fixture; no override needed.)
    let compiled = Arc::new(CompiledModel::new(model).unwrap());

    // Non-vacuity: the pass actually fired. Without this, a future change that
    // stopped hoisting this kernel would turn the test into a no-op LICM check.
    assert!(
        !compiled.model.per_eval_bindings.is_empty(),
        "fixture must hoist (per_eval_bindings non-empty) or this is not a LICM-on check"
    );

    let (params, param_names) = build_params_and_names(&compiled);
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(43);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let kappa_idx = compiled.param_index["kappa"];
    let beta_idx = compiled.param_index["beta"];
    let rho_idx = compiled.param_index["rho"];
    let k_idx = compiled.param_index["k"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // kappa is the load-bearing index — its gradient runs through the
        // hoisted per-eval binding; beta/rho/k guard the rest of the surface.
        &[kappa_idx, beta_idx, rho_idx, k_idx],
        dt, 1e-4,
        "licm_hoisting_kernel",
    );
}

/// Build a Poisson-only obs version of seir_observations programmatically.
fn build_poisson_seir() -> ir::Model {
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    model.observations.retain(|o| o.name == "weekly_cases");
    for om in &mut model.observations {
        use ir::observation::{Likelihood, PoissonLikelihood};
        if let Likelihood::NegBinomial(nb) = &om.likelihood {
            om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                // rate = rho * projected → ∂rate/∂rho = projected. Reuse the
                // compiler-emitted mean_grad from the NegBin source verbatim.
                rate: ir::Diffable { expr: nb.mean.expr.clone(), grad: nb.mean.grad.clone(), proj_grad: nb.mean.proj_grad.clone() },
            });
        }
    }
    model
}

#[test]
fn gh76_poisson_obs_grad_matches_fd() {
    let mut model = build_poisson_seir();
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(45);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let rho_idx = compiled.param_index["rho"];
    let beta_idx = compiled.param_index["beta"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        &[rho_idx, beta_idx],
        dt, 1e-4,
        "gh76_poisson_obs",
    );
}

/// Build a discretized-normal obs version of seir_observations programmatically.
/// Mean = rho * projected, sd = sigma_obs (a new parameter).
fn build_discretized_normal_seir() -> ir::Model {
    use ir::expr::{Expr, ParamExpr};
    use ir::parameter::Parameter;
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    model.observations.retain(|o| o.name == "weekly_cases");

    model.parameters.push(Parameter { name: "sigma_obs".to_string(), value: ir::parameter::ParamValue::Estimated { init: Some(8.0), bounds: Some((0.1, 1000.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: Some(ir::parameter::ParamKind::Positive), param_dim: None });

    for om in &mut model.observations {
        use ir::observation::{Likelihood, NormalLikelihood};
        if let Likelihood::NegBinomial(nb) = &om.likelihood {
            om.likelihood = Likelihood::Normal(NormalLikelihood {
                // mean = rho * projected → ∂mean/∂rho = projected (reuse source).
                mean: ir::Diffable { expr: nb.mean.expr.clone(), grad: nb.mean.grad.clone(), proj_grad: nb.mean.proj_grad.clone() },
                // sd = sigma_obs → ∂sd/∂sigma_obs = 1.
                sd: ir::Diffable { expr: Expr::Param(ParamExpr { param: "sigma_obs".to_string() }), grad: grad1("sigma_obs", const1()), proj_grad: None },
            });
        }
    }
    model
}

#[test]
fn gh76_discretized_normal_obs_grad_matches_fd() {
    // Synthetic obs are projected via the obs-model mean — observations
    // sit near the likelihood's peak. The tail-shifted variants below
    // (`_tail_3sigma`, `_tail_5sigma`, `_tail_8sigma`) exercise the
    // tail-precision regime where the prior φ-difference denominator
    // collapsed to noise; after the gh#76-cleanup erfc port both regimes
    // agree at 1e-4.
    run_discretized_normal_grad_fd(0.0, "gh76_discretized_normal_obs");
}

/// gh#76 cleanup: tail FD points. After the audit-H2 port into
/// `discretized_normal_logpmf_grad`, the gradient and the value share the
/// same erfc-stable `prob`, so FD-vs-analytic must hold at 1e-4 in the
/// regime where the prior implementation degraded to ~1e-3 (or worse).
#[test]
fn gh76_discretized_normal_obs_grad_matches_fd_tail_3sigma() {
    run_discretized_normal_grad_fd(3.0, "gh76_discretized_normal_obs_tail_3σ");
}

#[test]
fn gh76_discretized_normal_obs_grad_matches_fd_tail_5sigma() {
    run_discretized_normal_grad_fd(5.0, "gh76_discretized_normal_obs_tail_5σ");
}

#[test]
fn gh76_discretized_normal_obs_grad_matches_fd_tail_8sigma() {
    run_discretized_normal_grad_fd(8.0, "gh76_discretized_normal_obs_tail_8σ");
}

/// Build a Binomial-obs version of seir_observations programmatically.
///
/// Models reported cases as `k ~ Binomial(n = incidence, p = rho)` —
/// `rho` is the per-case reporting probability. Used by gh#76 cleanup
/// concern (E) to exercise the Binomial dispatch arm in
/// `eval_likelihood_resolved_grad` (which had no FD coverage in the
/// gh#76 commit).
fn build_binomial_seir() -> ir::Model {
    use ir::expr::*;
    use ir::observation::*;
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    model.observations.retain(|o| o.name == "weekly_cases");
    for om in &mut model.observations {
        if let Likelihood::NegBinomial(_) = &om.likelihood {
            // n = projected (weekly incidence — the FlowSum). p = rho.
            // Mean is n·p = projected·rho, matching the NegBin source's
            // mean of `rho * projected`.
            om.likelihood = Likelihood::Binomial(BinomialLikelihood {
                n: Expr::Projected(ProjectedExpr { projected: () }),
                // p = rho → ∂p/∂rho = 1. (n = projected is θ-independent — no grad.)
                p: ir::Diffable { expr: Expr::Param(ParamExpr { param: "rho".to_string() }), grad: grad1("rho", const1()), proj_grad: None },
            });
        }
    }
    model
}

#[test]
fn gh76_binomial_obs_grad_matches_fd() {
    // gh#76 cleanup concern (E). The Binomial dispatch arm at
    // `obs_model.rs:204-218` was wired in the gh#76 commit but had no
    // FD test exercising it end-to-end through
    // `complete_data_loglik_grad`. The existing NegBin/Poisson/Normal
    // tests don't cover Binomial's chain-rule form
    //   d/dp [log C(n,k) + k·log(p) + (n−k)·log(1−p)] = k/p − (n−k)/(1−p)
    // multiplied by d(p)/d(θ_k) through the emitted `p_grad` map.
    let mut model = build_binomial_seir();
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(47);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let rho_idx = compiled.param_index["rho"];
    let beta_idx = compiled.param_index["beta"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // rho is the new term (Binomial's `p` arg). beta is the
        // regression check that the rate-density gradient still works.
        &[rho_idx, beta_idx],
        dt, 1e-4,
        "gh76_binomial_obs",
    );
}

/// Build a BetaBinomial-obs version of seir_observations programmatically.
///
/// Models reported cases as `k ~ BetaBinomial(n = incidence, α, β)` — the
/// overdispersed-reporting analogue of the Binomial model, where the
/// per-case reporting probability itself varies (cluster/household
/// heterogeneity). `α` and `β` are exposed as estimated parameters
/// (`a_obs`, `b_obs`) so the gradient's chain-rule path through
/// the emitted `alpha_grad`/`beta_grad` maps are exercised on each in turn. Mean is
/// `n·α/(α+β)` (see `obs_mean_for_likelihood`), which stays well below `n`
/// for the chosen α, β, so synthetic obs satisfy `k ≤ n`.
fn build_beta_binomial_seir() -> ir::Model {
    use ir::expr::*;
    use ir::observation::*;
    use ir::parameter::Parameter;
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    model.observations.retain(|o| o.name == "weekly_cases");

    for (name, val) in [("a_obs", 2.0), ("b_obs", 8.0)] {
        model.parameters.push(Parameter { name: name.to_string(), value: ir::parameter::ParamValue::Estimated { init: Some(val), bounds: Some((0.01, 1000.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: Some(ir::parameter::ParamKind::Positive), param_dim: None });
    }

    for om in &mut model.observations {
        if let Likelihood::NegBinomial(_) = &om.likelihood {
            // n = projected (weekly incidence — the FlowSum).
            // α = a_obs, β = b_obs. Mean = n·a_obs/(a_obs+b_obs) = n·0.2.
            om.likelihood = Likelihood::BetaBinomial(BetaBinomialLikelihood {
                n: Expr::Projected(ProjectedExpr { projected: () }),
                // alpha = a_obs → ∂alpha/∂a_obs = 1.
                alpha: ir::Diffable { expr: Expr::Param(ParamExpr { param: "a_obs".to_string() }), grad: grad1("a_obs", const1()), proj_grad: None },
                // beta = b_obs → ∂beta/∂b_obs = 1.
                beta: ir::Diffable { expr: Expr::Param(ParamExpr { param: "b_obs".to_string() }), grad: grad1("b_obs", const1()), proj_grad: None },
            });
        }
    }
    model
}

#[test]
fn gh76_beta_binomial_obs_grad_matches_fd() {
    // gh#76 residual. The BetaBinomial dispatch arm in
    // `eval_likelihood_resolved_grad` was a documented no-op; this test
    // pins it end-to-end through `complete_data_loglik_grad`.
    //
    // RED→GREEN evidence: against the pre-wiring no-op, the analytic
    // gradient on a_obs / b_obs is identically 0 while the FD is nonzero,
    // so rel_err ≈ 1 and `fd_check` fails. After wiring, both arms agree.
    //
    // The chain-rule form differs from the other arms:
    //   d/dα [log C + lgamma(k+α) + lgamma(n−k+β) + lgamma(α+β)
    //         − lgamma(n+α+β) − lgamma(α) − lgamma(β)]
    //   = ψ(k+α) − ψ(α) − ψ(n+α+β) + ψ(α+β)   (and the β-mirror)
    // multiplied by d(α)/d(θ_k) [resp. d(β)/d(θ_k)] via the emitted grad maps.
    let mut model = build_beta_binomial_seir();
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
        ("a_obs", 2.0), ("b_obs", 8.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(48);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let a_obs_idx = compiled.param_index["a_obs"];
    let b_obs_idx = compiled.param_index["b_obs"];
    let beta_idx = compiled.param_index["beta"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // a_obs (BetaBinomial α arg) and b_obs (β arg) are the new terms;
        // beta is the regression check that the rate-density gradient is
        // unaffected.
        &[a_obs_idx, b_obs_idx, beta_idx],
        dt, 1e-4,
        "gh76_beta_binomial_obs",
    );
}

/// Build the gh#180 headline model: a Poisson obs with a PARAMETRIC
/// `DerivedExpr` projection `projected = qgam * I`. The scaling parameter `qgam`
/// reaches the observation ONLY through the projection; the rate `rho * projected`
/// then depends on qgam via the chain rule.
///
/// The emitted `rate_grad` inlines the projection into the argument exactly as
/// the OCaml autodiff does for a `DerivedExpr` projection (verified against the
/// `surveillance_likelihoods` golden's inlined `alpha_grad[kappa] = R/N`):
/// with `rate = rho · (qgam · I)`,
///   `∂rate/∂rho  = qgam · I`  (the projection, inlined — the FlowSum case would
///                              instead leave a bare `projected` node), and
///   `∂rate/∂qgam = rho · I`   (the gh#180 chain-rule term — silently zero when
///                              the obs path differentiated the argument alone).
fn build_parametric_projection_poisson_seir() -> ir::Model {
    use ir::expr::*;
    use ir::observation::*;
    use ir::parameter::Parameter;
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    model.observations.retain(|o| o.name == "weekly_cases");

    // qgam: the estimated projection-scaling parameter (positive).
    model.parameters.push(Parameter {
        name: "qgam".to_string(),
        value: ir::parameter::ParamValue::Estimated {
            init: Some(0.1),
            bounds: Some((0.001, 10.0)),
            prior: ir::parameter::PriorSpec::Flat,
            transform: ir::parameter::Transform::Identity,
        },
        param_kind: Some(ir::parameter::ParamKind::Positive),
        param_dim: None,
    });

    // qgam · I  — infectious prevalence scaled by qgam (the parametric DerivedExpr
    // projection) — and rho · I, the qgam chain-rule term after inlining.
    let qgam_times_i = Expr::bin_op(BinOp::Mul, Expr::param("qgam"), Expr::pop("I"));
    let rho_times_i  = Expr::bin_op(BinOp::Mul, Expr::param("rho"),  Expr::pop("I"));

    for om in &mut model.observations {
        om.projection = Projection::DerivedExpr(qgam_times_i.clone());
        let mut rate_grad = std::collections::HashMap::new();
        rate_grad.insert("rho".to_string(),
            ir::deriv::DerivEntry::Grad(qgam_times_i.clone()));
        rate_grad.insert("qgam".to_string(),
            ir::deriv::DerivEntry::Grad(rho_times_i.clone()));
        om.likelihood = Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable {
                expr: Expr::bin_op(
                    BinOp::Mul,
                    Expr::param("rho"),
                    Expr::Projected(ProjectedExpr { projected: () }),
                ),
                grad: rate_grad,
                proj_grad: None,
            },
        });
    }
    model
}

#[test]
fn gh180_parametric_projection_grad_matches_fd() {
    // gh#180 headline (the proof this whole arc exists for). A parameter `qgam`
    // reaches the observation ONLY through a parametric `DerivedExpr` projection
    // `projected = qgam · I`. Before the unified obs-gradient autodiff, the obs
    // gradient ran a runtime forward-mode differentiator over the likelihood
    // ARGUMENT alone (treating `projected` as constant), so ∂L/∂qgam was
    // SILENTLY ZERO. Now the emitted `rate_grad` carries the projection-inlined
    // chain-rule term and the runtime consumes it.
    //
    // Asserts the qgam gradient is (a) NON-zero and (b) central-difference
    // matching. Calls `complete_data_loglik_grad` DIRECTLY — `run_pgas`'s C1
    // preflight still refuses a parametric DerivedExpr projection until P5.
    let mut model = build_parametric_projection_poisson_seir();
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0), ("qgam", 0.1),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(49);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt);
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let qgam_idx = compiled.param_index["qgam"];
    let rho_idx = compiled.param_index["rho"];
    let beta_idx = compiled.param_index["beta"];

    // Non-zero check: compute the analytic gradient directly and assert the qgam
    // coordinate is non-zero — the exact quantity that was silently zero before.
    let d = estimated_indices.len();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; compiled.model.parameters.len()];
    for (est_idx, &model_idx) in estimated_indices.iter().enumerate() {
        model_to_estimated[model_idx] = Some(est_idx);
    }
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed, &model_to_estimated,
    );
    let oas = build_obs_at_substep(&observations, t_start, dt).unwrap();
    let (_ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &[], d, &rate_grads_for_run, &oas, &estimated_indices,
    ).unwrap();
    let qgam_est = estimated_indices.iter().position(|&i| i == qgam_idx).unwrap();
    eprintln!("[gh180_parametric_projection] d(ll)/d(qgam) = {:.6e} (analytic)", grad[qgam_est]);
    assert!(
        grad[qgam_est].abs() > 1e-6,
        "gh#180: the qgam gradient must be NON-zero — it reaches the observation \
         through the parametric projection and was silently zero before the \
         chain-rule term was captured; got {:.3e}", grad[qgam_est]
    );

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // qgam is the headline (the projection chain-rule term); rho shares the
        // projection; beta guards the rate-density gradient.
        &[qgam_idx, rho_idx, beta_idx],
        dt, 1e-4,
        "gh180_parametric_projection",
    );
}

fn run_discretized_normal_grad_fd(tail_sigmas: f64, name: &str) {
    let mut model = build_discretized_normal_seir();
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
        ("sigma_obs", 8.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(46);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let n_substeps = trajectory.substeps.len();

    let (substep_idx, obs_times) = obs_substep_indices_regular(
        t_start, dt, n_substeps, 7.0, 7.0, t_end,
    );

    let per_stream = if tail_sigmas == 0.0 {
        project_trajectory_to_obs(&compiled, &trajectory, &substep_idx, &params, dt)
    } else {
        project_trajectory_to_obs_shifted(
            &compiled, &trajectory, &substep_idx, &params, dt, tail_sigmas,
        )
    };
    let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
    let observations: Vec<Observation> = obs_times.iter()
        .map(|&t| Observation { time: t, value: 0.0 }).collect();

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let rho_idx = compiled.param_index["rho"];
    let sigma_obs_idx = compiled.param_index["sigma_obs"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        &[rho_idx, sigma_obs_idx],
        dt, 1e-4,
        name,
    );
}
