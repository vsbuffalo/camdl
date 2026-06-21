//! Compile observation model likelihoods from the IR into dmeasure closures.
//!
//! Evaluates the Expr fields in the IR's Likelihood using the expression
//! evaluator with `projected` set to the projected observation value.

use std::sync::Arc;
use crate::compiled_model::CompiledModel;
use crate::rng::StatefulRng;
use crate::propensity::EvalCtx;
use crate::resolved_expr::{
    ResolvedLikelihood, ResolveCtx, resolve_likelihood, eval_resolved, eval_resolved_deriv,
};
use crate::state::{IntState, RealState};
use crate::inference::obs_loglik::{
    negbin_logpmf, discretized_normal_logpmf_tol, poisson_logpmf, DEFAULT_TOL,
    negbin_logpmf_grad, discretized_normal_logpmf_grad, poisson_logpmf_grad,
    beta_binomial_logpmf_grad,
};
use crate::inference::types::LOG_PROB_FLOOR;
use ir::observation::ObservationModel;
use rand::prelude::Distribution;
use rand_distr::{Gamma, Normal};

/// Resolve a Likelihood using the compiled model's index maps.
///
/// IM3 in 2026-04-19 inference review: previously used `.expect`,
/// which panicked the whole process on construction-time resolve
/// failures (unknown parameter / compartment / table name inside a
/// likelihood expression). Now returns the underlying `SimError` so
/// the CLI can surface a proper diagnostic.
pub(crate) fn resolve_likelihood_from_model(
    likelihood: &ir::observation::Likelihood,
    compiled: &CompiledModel,
) -> Result<ResolvedLikelihood, crate::error::SimError> {
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
        global_to_int: &compiled.global_to_int,
        global_to_real: &compiled.global_to_real,
        table_meta: &table_meta,
        binding_index: &compiled.binding_index,
        per_eval_index: &compiled.per_eval_index,
    };
    resolve_likelihood(likelihood, &ctx)
}

/// Evaluate a resolved likelihood at (projected, observed, params).
///
/// `t` is the observation time. Likelihood expressions may reference
/// `time` (e.g. a reporting ramp `rho_t = rho_max/(1+exp(-(t-t_rep)/w_rep))`);
/// passing a frozen 0.0 silently corrupts any time-varying observation
/// model. Callers thread the actual observation time
/// (`MultiStreamObsModel::obs_times[obs_idx]`).
pub(crate) fn eval_likelihood_resolved(
    likelihood: &ResolvedLikelihood,
    t: f64,
    projected: f64,
    observed: f64,
    // Per-observation auxiliary data (e.g. a binomial denominator `n = tested`),
    // keyed by declared column name. `Expr::ObsColumnRef` reads from this. Empty
    // when the likelihood references no aux column. (§3.)
    aux: &[(String, f64)],
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
) -> f64 {
    let ctx = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0, projected: Some(proj),
        aux: Some(aux), int_float_override: None, per_eval: None,
    };

    match likelihood {
        ResolvedLikelihood::NegBinomial { mean, dispersion } => {
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            negbin_logpmf(observed, m, k)
        }
        ResolvedLikelihood::Normal { mean, sd } => {
            // IC2 in the 2026-04-19 inference review: `Normal` is
            // pomp/He-et-al.'s discretized-Normal *count* likelihood.
            // A clearly-fractional observation probably means the user
            // intended a continuous Normal PDF; log once so the
            // silent-coercion failure mode is visible.
            if observed.is_finite() && (observed - observed.round()).abs() > 1e-6 {
                use std::sync::atomic::{AtomicBool, Ordering};
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "Normal observation likelihood: observed value {:.6} is \
                         not integer-valued. The `normal(...)` likelihood is a \
                         discretized-count distribution (pomp/He et al.); your \
                         data will be rounded to the nearest non-negative integer \
                         before scoring. If you need a continuous Normal PDF, \
                         file a request for a ContinuousNormal variant.",
                        observed);
                }
            }
            let m = eval_resolved(mean, &ctx(projected));
            let s = eval_resolved(sd, &ctx(projected));
            discretized_normal_logpmf_tol(observed, m, s * s, DEFAULT_TOL)
        }
        ResolvedLikelihood::Poisson { rate } => {
            let r = eval_resolved(rate, &ctx(projected));
            poisson_logpmf(observed, r)
        }
        ResolvedLikelihood::Binomial { n, p } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            let k = observed.round().max(0.0) as u64;
            let n_int = n_val.round().max(0.0) as u64;
            crate::inference::obs_loglik::binom_logpmf(k, n_int, p_val)
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_val = eval_resolved(alpha, &ctx(projected));
            let beta_val = eval_resolved(beta, &ctx(projected));
            let k = observed.round().max(0.0) as u64;
            let n_int = n_val.round().max(0.0) as u64;
            crate::inference::obs_loglik::beta_binomial_logpmf(k, n_int, alpha_val, beta_val)
        }
        ResolvedLikelihood::Bernoulli { p } => {
            // Clamp p to [0, 1] before forming the log-probability.
            // Without the clamp, an out-of-range p (e.g. PGAS proposes
            // p_detect > 1 before the posterior concentrates) produces
            // a positive log-probability — invalid as an SMC weight,
            // and silently inflates posterior mass on the bad region.
            // Mirrors the Binomial / BetaBinomial clamping at lines
            // 154 and 162. See docs/dev/reviews/2026-04-30-correctness.md C1.
            let p_val = eval_resolved(p, &ctx(projected)).clamp(0.0, 1.0);
            if observed > 0.5 { p_val.max(LOG_PROB_FLOOR).ln() }
            else              { (1.0 - p_val).max(LOG_PROB_FLOOR).ln() }
        }
    }
}

/// Gradient of `eval_likelihood_resolved` w.r.t. estimated parameters.
///
/// Accumulates into `grad[i] += d(log L)/d(θ_i)` for each estimated parameter,
/// chain-ruling through every likelihood-argument expression that may depend
/// on θ. Mirrors `eval_likelihood_resolved` exactly so the gradient and the
/// scalar match. (gh#76)
///
/// Caveat — projection dependence. If a stream's `projection` is a
/// `DerivedExpr` that depends on parameters, the resulting d(projected)/dθ is
/// **not** propagated here — the likelihood args are differentiated assuming
/// `projected` is a constant. For `FlowSum` and `IntCompSum` projections (the
/// common case), `projected` does not depend on θ, so this is exact. For
/// `DerivedExpr` projections referencing θ, the gradient is missing a term;
/// see docs/dev/notes/2026-05-25-pgas-obs-grad-derivation.md. Issue gh#76's
/// reproducer (rho, k_obs, σ_obs) does not exercise that case.
pub(crate) fn eval_likelihood_resolved_grad(
    likelihood: &ResolvedLikelihood,
    t: f64,
    projected: f64,
    observed: f64,
    aux: &[(String, f64)],
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
    estimated_to_model: &[usize],
    grad: &mut [f64],
) {
    let ctx_at = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0,
        projected: Some(proj), aux: Some(aux), int_float_override: None, per_eval: None,
    };
    let ctx = ctx_at(projected);

    match likelihood {
        ResolvedLikelihood::NegBinomial { mean, dispersion } => {
            let m = eval_resolved(mean, &ctx);
            let k = eval_resolved(dispersion, &ctx);
            let (d_mu, d_k) = negbin_logpmf_grad(observed, m, k);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_resolved_deriv(mean, model_idx, &ctx);
                let dk = eval_resolved_deriv(dispersion, model_idx, &ctx);
                grad[i] += d_mu * dm + d_k * dk;
            }
        }
        ResolvedLikelihood::Normal { mean, sd } => {
            // Discretized-normal in the implementation; gradients w.r.t.
            // (mean, variance) come from `discretized_normal_logpmf_grad`,
            // then chain-ruled to (mean, sd) via d(var)/d(sd) = 2·sd.
            let m = eval_resolved(mean, &ctx);
            let s = eval_resolved(sd, &ctx);
            let var = s * s;
            let (d_mu, d_var) = discretized_normal_logpmf_grad(observed, m, var, DEFAULT_TOL);
            // d(log L)/d(sd) = d(log L)/d(var) · d(var)/d(sd) = d_var · 2·sd
            let d_sd = d_var * 2.0 * s;
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_resolved_deriv(mean, model_idx, &ctx);
                let ds = eval_resolved_deriv(sd, model_idx, &ctx);
                grad[i] += d_mu * dm + d_sd * ds;
            }
        }
        ResolvedLikelihood::Poisson { rate } => {
            let r = eval_resolved(rate, &ctx);
            let d_rate = poisson_logpmf_grad(observed, r);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dr = eval_resolved_deriv(rate, model_idx, &ctx);
                grad[i] += d_rate * dr;
            }
        }
        ResolvedLikelihood::Binomial { n, p } => {
            // log p(k|n,p) = log C(n,k) + k·log(p) + (n-k)·log(1-p)
            // n is integer-valued (rounded); treat n as constant w.r.t. θ.
            // d/dp = k/p - (n-k)/(1-p)
            let n_val = eval_resolved(n, &ctx);
            let p_val = eval_resolved(p, &ctx);
            let n_int = n_val.round().max(0.0) as u64;
            let k_obs = observed.round().max(0.0) as u64;
            if p_val > 0.0 && p_val < 1.0 && k_obs <= n_int {
                let d_p = k_obs as f64 / p_val - (n_int - k_obs) as f64 / (1.0 - p_val);
                for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                    let dp = eval_resolved_deriv(p, model_idx, &ctx);
                    grad[i] += d_p * dp;
                }
            }
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta } => {
            // log L = log C(n,k) + lgamma(k+α) + lgamma(n−k+β) + lgamma(α+β)
            //         − lgamma(n+α+β) − lgamma(α) − lgamma(β)
            // n is integer-valued (rounded); treat it as constant w.r.t. θ,
            // exactly as the Binomial arm treats its `n`. The combinatorial
            // log C(n,k) term carries no α/β dependence. The remaining
            // gradient w.r.t. (α, β) comes from `beta_binomial_logpmf_grad`,
            // then chain-rules to each estimated param via the α/β arg
            // expressions.
            let n_val = eval_resolved(n, &ctx);
            let alpha_val = eval_resolved(alpha, &ctx);
            let beta_val = eval_resolved(beta, &ctx);
            let n_round = n_val.round().max(0.0);
            let (d_alpha, d_beta) =
                beta_binomial_logpmf_grad(observed, n_round, alpha_val, beta_val);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let da = eval_resolved_deriv(alpha, model_idx, &ctx);
                let db = eval_resolved_deriv(beta, model_idx, &ctx);
                grad[i] += d_alpha * da + d_beta * db;
            }
        }
        ResolvedLikelihood::Bernoulli { p } => {
            // log L = log(p)         if observed > 0.5
            //       = log(1 - p)     otherwise
            // Outside [0,1] the clamp in eval_likelihood_resolved fires,
            // so the in-domain gradient is exact only when 0 < p < 1.
            let p_val = eval_resolved(p, &ctx);
            if p_val > 0.0 && p_val < 1.0 {
                let d_log = if observed > 0.5 { 1.0 / p_val } else { -1.0 / (1.0 - p_val) };
                for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                    let dp = eval_resolved_deriv(p, model_idx, &ctx);
                    grad[i] += d_log * dp;
                }
            }
        }
    }
}

// ── rmeasure: observation model sampler ─────────────────────────────────────

/// Build an rmeasure closure for pfilter (fixed params).
/// Takes (projected, t, counts, rng) → observation draw.
///
/// `t` is the observation time (threaded into likelihood arg
/// expressions so time-varying reporting / observation processes
/// evaluate at the right instant; a frozen 0.0 silently corrupts any
/// time-varying obs model). `counts` must be the integer-compartment
/// state (local-indexed) at the observation time. The sampler
/// evaluates likelihood argument expressions (e.g. `p = projected /
/// PopSum([S, I, R])`) against this state. Passing a zero-filled
/// slice silently corrupts state-dependent likelihoods — see
/// `docs/dev/incidents/2026-04-22-observation-sampler-scratch-state.md`.
pub fn compile_obs_sample_pf(
    obs_model: &ObservationModel,
    compiled: Arc<CompiledModel>,
    params: &[f64],
) -> Box<dyn Fn(f64, f64, &[i64], &mut crate::rng::StatefulRng) -> f64> {
    let resolved = resolve_likelihood_from_model(&obs_model.likelihood, &compiled)
        .unwrap_or_else(|e| panic!("observation likelihood resolution failed: {:?}", e));
    let params = params.to_vec();
    let real_s = RealState::new(compiled.real_local_to_global.len());
    let n_int  = compiled.int_local_to_global.len();

    Box::new(move |projected: f64, t: f64, counts: &[i64], rng: &mut StatefulRng| {
        // GH #6 fix: evaluate likelihood args against the real state,
        // not a zero-filled scratch. Caller is responsible for passing
        // the compartment snapshot at the obs time.
        assert_eq!(counts.len(), n_int,
            "compile_obs_sample_pf: counts length {} != expected {}", counts.len(), n_int);
        let int_s = IntState::from_vec(counts.to_vec());
        // No per-observation aux at emission time (simulate --obs has no data
        // file). A likelihood referencing an aux column (binomial `n = tested`)
        // can't be sampled without a denominator — its `ObsColumnRef` would
        // error at eval, the honest behaviour.
        sample_obs_resolved(&resolved, t, projected, &[], &params, &compiled, &int_s, &real_s, rng)
    })
}

/// Draw one sample from the resolved observation model at observation
/// time `t`. Likelihood expressions referencing `time` (e.g. a reporting
/// ramp) are evaluated at `t`; a frozen 0.0 silently corrupts any
/// time-varying observation model.
pub(crate) fn sample_obs_resolved(
    likelihood: &ResolvedLikelihood,
    t: f64,
    projected: f64,
    aux: &[(String, f64)],
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
    rng: &mut StatefulRng,
) -> f64 {
    let ctx = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0, projected: Some(proj),
        aux: Some(aux), int_float_override: None, per_eval: None,
    };

    match likelihood {
        ResolvedLikelihood::NegBinomial { mean, dispersion } => {
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            if m <= 0.0 || k <= 0.0 { return 0.0; }
            let g = Gamma::new(k, m / k).unwrap().sample(rng.inner_mut());
            rng.poisson(g) as f64
        }
        ResolvedLikelihood::Normal { mean, sd } => {
            let m = eval_resolved(mean, &ctx(projected));
            let s = eval_resolved(sd, &ctx(projected));
            let draw = Normal::new(m, s.max(1e-10)).unwrap().sample(rng.inner_mut());
            draw.round().max(0.0)
        }
        ResolvedLikelihood::Poisson { rate } => {
            let r = eval_resolved(rate, &ctx(projected));
            rng.poisson(r) as f64
        }
        ResolvedLikelihood::Binomial { n, p } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            rng.binomial(n_val.round().max(0.0) as u64, p_val.clamp(0.0, 1.0)) as f64
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta } => {
            // Draw BetaBinomial(n, alpha, beta): p ~ Beta(alpha, beta),
            // then k ~ Binomial(n, p). Uses the inner RNG directly for
            // the Beta draw (Gamma(a,1)/(Gamma(a,1)+Gamma(b,1))).
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_val = eval_resolved(alpha, &ctx(projected)).max(LOG_PROB_FLOOR);
            let beta_val  = eval_resolved(beta,  &ctx(projected)).max(LOG_PROB_FLOOR);
            let n_int = n_val.round().max(0.0) as u64;
            use rand_distr::{Gamma, Distribution};
            let inner = rng.inner_mut();
            let a = Gamma::new(alpha_val, 1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            let b = Gamma::new(beta_val,  1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            let p = a / (a + b);
            rng.binomial(n_int, p.clamp(0.0, 1.0)) as f64
        }
        ResolvedLikelihood::Bernoulli { p } => {
            // Clamp before sampling — an out-of-range p would
            // otherwise always-1 (p > 1) or always-0 (p < 0).
            let p_val = eval_resolved(p, &ctx(projected)).clamp(0.0, 1.0);
            if rng.uniform() < p_val { 1.0 } else { 0.0 }
        }
    }
}

/// Compute E[y | projected, params] — the observation model mean, no sampling.
/// `t` is the observation time, threaded for time-varying likelihoods.
pub(crate) fn eval_obs_mean_resolved(
    likelihood: &ResolvedLikelihood,
    t: f64,
    projected: f64,
    aux: &[(String, f64)],
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
) -> f64 {
    let ctx = |proj: f64| EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0, projected: Some(proj),
        aux: Some(aux), int_float_override: None, per_eval: None,
    };

    match likelihood {
        ResolvedLikelihood::NegBinomial { mean, .. } => {
            eval_resolved(mean, &ctx(projected))
        }
        ResolvedLikelihood::Normal { mean, .. } => {
            eval_resolved(mean, &ctx(projected))
        }
        ResolvedLikelihood::Poisson { rate } => {
            eval_resolved(rate, &ctx(projected))
        }
        ResolvedLikelihood::Binomial { n, p } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            n_val * p_val
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta } => {
            // E[BetaBinomial(n, α, β)] = n · α / (α + β)
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_val = eval_resolved(alpha, &ctx(projected));
            let beta_val  = eval_resolved(beta,  &ctx(projected));
            let denom = (alpha_val + beta_val).max(LOG_PROB_FLOOR);
            n_val * (alpha_val / denom)
        }
        ResolvedLikelihood::Bernoulli { p } => {
            eval_resolved(p, &ctx(projected))
        }
    }
}

// NOTE: The old compile_joint_obs_loglik was replaced by
// types::joint_obs_weight + types::ObsStreamSpec. The join now happens
// in ONE shared function used by PF, PGAS, CSMC, and gradient evaluation.

#[cfg(test)]
mod tests {
    use super::*;

    /// Bernoulli log-pmf computation, isolated from the IR/CompiledModel
    /// scaffolding so we can assert on edge cases directly. Mirrors
    /// the inline expression at `eval_likelihood_resolved`'s
    /// `ResolvedLikelihood::Bernoulli` arm.
    fn bernoulli_logpmf_clamped(p_val: f64, observed: f64) -> f64 {
        let p = p_val.clamp(0.0, 1.0);
        if observed > 0.5 { p.max(LOG_PROB_FLOOR).ln() }
        else              { (1.0 - p).max(LOG_PROB_FLOOR).ln() }
    }

    #[test]
    fn bernoulli_logpmf_clamps_p_above_one() {
        // p_val = 1.5 must produce log(1.0) = 0.0, not log(1.5) ≈ 0.405.
        // The unclamped version was a critical bug: a positive log-prob
        // inflates SMC weights for any particle that sampled an
        // out-of-range p (e.g. PGAS proposals before the posterior
        // concentrates).
        let log_p = bernoulli_logpmf_clamped(1.5, 1.0);
        assert!(log_p <= 0.0, "log-prob must be ≤ 0, got {}", log_p);
        // Specifically, p=1 with observation=1 → log(1) = 0.
        assert!(log_p.abs() < 1e-12, "expected 0.0, got {}", log_p);
    }

    #[test]
    fn bernoulli_logpmf_clamps_p_below_zero() {
        // p_val = -0.3 must produce log(1.0) = 0.0 for observation=0,
        // not log(1.3) ≈ 0.262.
        let log_p = bernoulli_logpmf_clamped(-0.3, 0.0);
        assert!(log_p <= 0.0, "log-prob must be ≤ 0, got {}", log_p);
        assert!(log_p.abs() < 1e-12, "expected 0.0, got {}", log_p);

        // Observation = 1 with p = -0.3 → log(0) → LOG_PROB_FLOOR.ln().
        let log_p_obs1 = bernoulli_logpmf_clamped(-0.3, 1.0);
        assert!(log_p_obs1 <= LOG_PROB_FLOOR.ln() + 1e-12,
            "p<0 with obs=1 must floor: got {}", log_p_obs1);
    }

    #[test]
    fn bernoulli_logpmf_in_range_unchanged() {
        // p=0.7, obs=1 → log(0.7) ≈ -0.357
        let log_p = bernoulli_logpmf_clamped(0.7, 1.0);
        assert!((log_p - 0.7_f64.ln()).abs() < 1e-12);
        // p=0.7, obs=0 → log(0.3) ≈ -1.204
        let log_p = bernoulli_logpmf_clamped(0.7, 0.0);
        assert!((log_p - 0.3_f64.ln()).abs() < 1e-12);
    }
}
