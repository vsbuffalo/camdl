//! Compile observation model likelihoods from the IR into dmeasure closures.
//!
//! Evaluates the Expr fields in the IR's Likelihood using the expression
//! evaluator with `projected` set to the projected observation value.

use std::sync::Arc;
use crate::compiled_model::CompiledModel;
use crate::rng::StatefulRng;
use crate::propensity::EvalCtx;
use crate::resolved_expr::{
    ResolvedDerivEntry, ResolvedProjGrad,
    ResolvedLikelihood, ResolveCtx, resolve_likelihood, eval_resolved, eval_emitted_grad,
};
use crate::state::{IntState, RealState};
use crate::inference::obs_loglik::{
    negbin_logpmf, zi_negbin_logpmf, zi_negbin_logpmf_grad, discretized_normal_logpmf_tol,
    poisson_logpmf, DEFAULT_TOL,
    negbin_logpmf_grad, discretized_normal_logpmf_grad, poisson_logpmf_grad,
    beta_binomial_logpmf_grad, beta_logpdf, beta_logpdf_grad,
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
        ResolvedLikelihood::NegBinomial { mean, dispersion, .. } => {
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            negbin_logpmf(observed, m, k)
        }
        ResolvedLikelihood::Normal { mean, sd, .. } => {
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
        ResolvedLikelihood::Poisson { rate, .. } => {
            let r = eval_resolved(rate, &ctx(projected));
            poisson_logpmf(observed, r)
        }
        ResolvedLikelihood::Binomial { n, p, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            let k = observed.round().max(0.0) as u64;
            let n_int = n_val.round().max(0.0) as u64;
            crate::inference::obs_loglik::binom_logpmf(k, n_int, p_val)
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_val = eval_resolved(alpha, &ctx(projected));
            let beta_val = eval_resolved(beta, &ctx(projected));
            let k = observed.round().max(0.0) as u64;
            let n_int = n_val.round().max(0.0) as u64;
            crate::inference::obs_loglik::beta_binomial_logpmf(k, n_int, alpha_val, beta_val)
        }
        ResolvedLikelihood::Beta { mean, concentration, .. } => {
            let m = eval_resolved(mean, &ctx(projected));
            let c = eval_resolved(concentration, &ctx(projected));
            beta_logpdf(observed, m, c)
        }
        ResolvedLikelihood::Bernoulli { p, .. } => {
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
        ResolvedLikelihood::ZeroInflatedNegBinomial { mean, dispersion, pi, .. } => {
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            let p = eval_resolved(pi, &ctx(projected));
            zi_negbin_logpmf(observed, m, k, p)
        }
    }
}

/// Gradient of `eval_likelihood_resolved` w.r.t. estimated parameters.
///
/// Accumulates into `grad[i] += d(log L)/d(θ_i)` for each estimated parameter,
/// chain-ruling through every likelihood-argument expression that may depend
/// on θ. Mirrors `eval_likelihood_resolved` exactly so the gradient and the
/// scalar match. (gh#76, gh#180)
///
/// The `∂arg/∂θ` factor now comes from the compiler-emitted gradient map
/// (`*_grad`, resolved into each arm's `*_grad` carrier), evaluated through the
/// shared [`eval_emitted_grad`](crate::resolved_expr::eval_emitted_grad) seam —
/// the same authority that feeds `rate_grad`. Because the OCaml autodiff inlines
/// a `DerivedExpr` projection into the argument before differentiating, a
/// parameter that reaches an observation THROUGH the projection
/// (`projected = qgam · prevalence`) now contributes its chain-rule term
/// `∂L/∂projected · ∂projected/∂θ` — the gh#180 term that was silently zero when
/// this path ran a runtime forward-mode differentiator over the argument alone.
/// The per-distribution `∂logpmf/∂arg` factors (`negbin_logpmf_grad`, …) are the
/// irreducible runtime piece and are unchanged.
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
        ResolvedLikelihood::NegBinomial { mean, mean_grad, dispersion, dispersion_grad, .. } => {
            let m = eval_resolved(mean, &ctx);
            let k = eval_resolved(dispersion, &ctx);
            let (d_mu, d_k) = negbin_logpmf_grad(observed, m, k);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_emitted_grad(mean_grad, model_idx, &ctx);
                let dk = eval_emitted_grad(dispersion_grad, model_idx, &ctx);
                grad[i] += d_mu * dm + d_k * dk;
            }
        }
        ResolvedLikelihood::Normal { mean, mean_grad, sd, sd_grad, .. } => {
            // Discretized-normal in the implementation; gradients w.r.t.
            // (mean, variance) come from `discretized_normal_logpmf_grad`,
            // then chain-ruled to (mean, sd) via d(var)/d(sd) = 2·sd.
            let m = eval_resolved(mean, &ctx);
            let s = eval_resolved(sd, &ctx);
            let var = s * s;
            let (d_mu, d_var) = discretized_normal_logpmf_grad(observed, m, var, DEFAULT_TOL);
            // d(log L)/d(sd) = d(log L)/d(var) · d(var)/d(sd) = d_var · 2·sd.
            // The `2·sd` Jacobian stays in this runtime factor — the emitted
            // `sd_grad` is ∂sd/∂θ, NOT ∂var/∂θ (proposal §4).
            let d_sd = d_var * 2.0 * s;
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_emitted_grad(mean_grad, model_idx, &ctx);
                let ds = eval_emitted_grad(sd_grad, model_idx, &ctx);
                grad[i] += d_mu * dm + d_sd * ds;
            }
        }
        ResolvedLikelihood::Poisson { rate, rate_grad, .. } => {
            let r = eval_resolved(rate, &ctx);
            let d_rate = poisson_logpmf_grad(observed, r);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dr = eval_emitted_grad(rate_grad, model_idx, &ctx);
                grad[i] += d_rate * dr;
            }
        }
        ResolvedLikelihood::Binomial { n, p, p_grad, .. } => {
            // log p(k|n,p) = log C(n,k) + k·log(p) + (n-k)·log(1-p)
            // n is integer-valued (rounded) and θ-independent (gated by P5);
            // it carries no gradient. d/dp = k/p - (n-k)/(1-p)
            let n_val = eval_resolved(n, &ctx);
            let p_val = eval_resolved(p, &ctx);
            let n_int = n_val.round().max(0.0) as u64;
            let k_obs = observed.round().max(0.0) as u64;
            if p_val > 0.0 && p_val < 1.0 && k_obs <= n_int {
                let d_p = k_obs as f64 / p_val - (n_int - k_obs) as f64 / (1.0 - p_val);
                for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                    let dp = eval_emitted_grad(p_grad, model_idx, &ctx);
                    grad[i] += d_p * dp;
                }
            }
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, alpha_grad, beta, beta_grad, .. } => {
            // log L = log C(n,k) + lgamma(k+α) + lgamma(n−k+β) + lgamma(α+β)
            //         − lgamma(n+α+β) − lgamma(α) − lgamma(β)
            // n is integer-valued (rounded) and θ-independent (gated by P5); it
            // carries no gradient, exactly as the Binomial arm treats its `n`.
            // The combinatorial log C(n,k) term carries no α/β dependence. The
            // remaining gradient w.r.t. (α, β) comes from
            // `beta_binomial_logpmf_grad`, then chain-rules to each estimated
            // param via the emitted α/β gradient maps.
            let n_val = eval_resolved(n, &ctx);
            let alpha_val = eval_resolved(alpha, &ctx);
            let beta_val = eval_resolved(beta, &ctx);
            let n_round = n_val.round().max(0.0);
            let (d_alpha, d_beta) =
                beta_binomial_logpmf_grad(observed, n_round, alpha_val, beta_val);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let da = eval_emitted_grad(alpha_grad, model_idx, &ctx);
                let db = eval_emitted_grad(beta_grad, model_idx, &ctx);
                grad[i] += d_alpha * da + d_beta * db;
            }
        }
        ResolvedLikelihood::Beta { mean, mean_grad, concentration, concentration_grad, .. } => {
            // log f = (a−1)ln x + (b−1)ln(1−x) + lgamma(φ) − lgamma(a) − lgamma(b),
            // a = mean·φ, b = (1−mean)·φ. The (mean, φ) partials from
            // `beta_logpdf_grad` chain-rule to each estimated param via the
            // emitted mean/concentration gradient maps.
            let m = eval_resolved(mean, &ctx);
            let c = eval_resolved(concentration, &ctx);
            let (d_mean, d_conc) = beta_logpdf_grad(observed, m, c);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_emitted_grad(mean_grad, model_idx, &ctx);
                let dc = eval_emitted_grad(concentration_grad, model_idx, &ctx);
                grad[i] += d_mean * dm + d_conc * dc;
            }
        }
        ResolvedLikelihood::Bernoulli { p, p_grad, .. } => {
            // log L = log(p)         if observed > 0.5
            //       = log(1 - p)     otherwise
            // Outside [0,1] the clamp in eval_likelihood_resolved fires,
            // so the in-domain gradient is exact only when 0 < p < 1.
            let p_val = eval_resolved(p, &ctx);
            if p_val > 0.0 && p_val < 1.0 {
                let d_log = if observed > 0.5 { 1.0 / p_val } else { -1.0 / (1.0 - p_val) };
                for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                    let dp = eval_emitted_grad(p_grad, model_idx, &ctx);
                    grad[i] += d_log * dp;
                }
            }
        }
        ResolvedLikelihood::ZeroInflatedNegBinomial {
            mean, mean_grad, dispersion, dispersion_grad, pi, pi_grad, ..
        } => {
            // The mixture's (mu, k, pi) partials chain-rule to each estimated
            // parameter through the emitted gradient maps, exactly as the
            // NegBinomial arm above does for its two.
            let m = eval_resolved(mean, &ctx);
            let k = eval_resolved(dispersion, &ctx);
            let p = eval_resolved(pi, &ctx);
            let (d_mean, d_disp, d_pi) = zi_negbin_logpmf_grad(observed, m, k, p);
            for (i, &model_idx) in estimated_to_model.iter().enumerate() {
                let dm = eval_emitted_grad(mean_grad, model_idx, &ctx);
                let dk = eval_emitted_grad(dispersion_grad, model_idx, &ctx);
                let dp = eval_emitted_grad(pi_grad, model_idx, &ctx);
                grad[i] += d_mean * dm + d_disp * dk + d_pi * dp;
            }
        }
    }
}

/// Evaluate a resolved `∂arg/∂projected` ([`ResolvedProjGrad`]) to a number:
/// `None` → 0 (the argument does not read the projection output), `Grad` → the
/// value evaluator, `Unsupported` → **unreachable on a gated path** (the §1h gate
/// refused any nonsmooth-of-projection argument before a gradient was taken), so a
/// `debug_assert` surfaces a regression and release falls back to 0.
#[inline]
fn eval_proj_grad(proj: &ResolvedProjGrad, ctx: &EvalCtx<'_>) -> f64 {
    match proj {
        None => 0.0,
        Some(ResolvedDerivEntry::Grad(e)) => eval_resolved(e, ctx),
        Some(ResolvedDerivEntry::Unsupported { code }) => {
            debug_assert!(
                false,
                "ungated Unsupported proj_grad ({code:?}) reached dlogp_dprojected — the \
                 §1h gate invariant was violated"
            );
            0.0
        }
    }
}

/// `∂ log p(y | ·)/∂projected` — the observation score with respect to the projected
/// value, summed over every distribution argument that depends on `projected`
/// (gh#275 §"chain rule"): `Σ_arg (∂logp/∂arg)·(∂arg/∂projected)`. The
/// per-distribution `∂logp/∂arg` partials are the SAME irreducible runtime factors
/// [`eval_likelihood_resolved_grad`] uses (`negbin_logpmf_grad`, …); the chained
/// `∂arg/∂projected` factor is now the compiler-emitted `proj_grad` (via the
/// `WrtProjected` autodiff), so a reporting rate `rho·projected` and the He
/// mean-linked variance are supported — not just `arg = projected`.
///
/// This is the ODE-gradient FACTOR 2 (the trajectory chain); FACTOR 1 (θ entering
/// the distribution directly, `projected` held fixed) stays with
/// `eval_likelihood_resolved_grad`. The two are orthogonal and summed by the caller.
pub(crate) fn dlogp_dprojected(
    likelihood: &ResolvedLikelihood,
    t: f64,
    projected: f64,
    observed: f64,
    aux: &[(String, f64)],
    params: &[f64],
    compiled: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
) -> f64 {
    let ctx = EvalCtx {
        model: compiled, int_s, real_s, params, t, dt: 0.0,
        projected: Some(projected), aux: Some(aux), int_float_override: None, per_eval: None,
    };
    match likelihood {
        ResolvedLikelihood::NegBinomial { mean, dispersion, mean_proj, dispersion_proj, .. } => {
            let m = eval_resolved(mean, &ctx);
            let k = eval_resolved(dispersion, &ctx);
            let (d_mu, d_k) = negbin_logpmf_grad(observed, m, k);
            d_mu * eval_proj_grad(mean_proj, &ctx) + d_k * eval_proj_grad(dispersion_proj, &ctx)
        }
        ResolvedLikelihood::Normal { mean, sd, mean_proj, sd_proj, .. } => {
            let m = eval_resolved(mean, &ctx);
            let s = eval_resolved(sd, &ctx);
            let var = s * s;
            let (d_mu, d_var) = discretized_normal_logpmf_grad(observed, m, var, DEFAULT_TOL);
            let d_sd = d_var * 2.0 * s;
            d_mu * eval_proj_grad(mean_proj, &ctx) + d_sd * eval_proj_grad(sd_proj, &ctx)
        }
        ResolvedLikelihood::Poisson { rate, rate_proj, .. } => {
            let r = eval_resolved(rate, &ctx);
            poisson_logpmf_grad(observed, r) * eval_proj_grad(rate_proj, &ctx)
        }
        ResolvedLikelihood::Binomial { n, p, p_proj, .. } => {
            // `n` is θ- AND projection-independent (the §1h gate refuses a `Projected`
            // in `n`), so only `p`'s projection derivative contributes to factor 2.
            let n_val = eval_resolved(n, &ctx);
            let p_val = eval_resolved(p, &ctx);
            let n_int = n_val.round().max(0.0) as u64;
            let k_obs = observed.round().max(0.0) as u64;
            if p_val > 0.0 && p_val < 1.0 && k_obs <= n_int {
                let d_p = k_obs as f64 / p_val - (n_int - k_obs) as f64 / (1.0 - p_val);
                d_p * eval_proj_grad(p_proj, &ctx)
            } else {
                0.0
            }
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, alpha_proj, beta_proj, .. } => {
            let n_val = eval_resolved(n, &ctx);
            let alpha_val = eval_resolved(alpha, &ctx);
            let beta_val = eval_resolved(beta, &ctx);
            let n_round = n_val.round().max(0.0);
            let (d_alpha, d_beta) =
                beta_binomial_logpmf_grad(observed, n_round, alpha_val, beta_val);
            d_alpha * eval_proj_grad(alpha_proj, &ctx) + d_beta * eval_proj_grad(beta_proj, &ctx)
        }
        ResolvedLikelihood::Beta { mean, concentration, mean_proj, concentration_proj, .. } => {
            let m = eval_resolved(mean, &ctx);
            let c = eval_resolved(concentration, &ctx);
            let (d_mean, d_conc) = beta_logpdf_grad(observed, m, c);
            d_mean * eval_proj_grad(mean_proj, &ctx)
                + d_conc * eval_proj_grad(concentration_proj, &ctx)
        }
        ResolvedLikelihood::Bernoulli { p, p_proj, .. } => {
            let p_val = eval_resolved(p, &ctx);
            if p_val > 0.0 && p_val < 1.0 {
                let d_log = if observed > 0.5 { 1.0 / p_val } else { -1.0 / (1.0 - p_val) };
                d_log * eval_proj_grad(p_proj, &ctx)
            } else {
                0.0
            }
        }
        ResolvedLikelihood::ZeroInflatedNegBinomial {
            mean, dispersion, pi, mean_proj, dispersion_proj, pi_proj, ..
        } => {
            let m = eval_resolved(mean, &ctx);
            let k = eval_resolved(dispersion, &ctx);
            let p = eval_resolved(pi, &ctx);
            let (d_mean, d_disp, d_pi) = zi_negbin_logpmf_grad(observed, m, k, p);
            d_mean * eval_proj_grad(mean_proj, &ctx)
                + d_disp * eval_proj_grad(dispersion_proj, &ctx)
                + d_pi * eval_proj_grad(pi_proj, &ctx)
        }
    }
}

// ── rmeasure: observation model sampler ─────────────────────────────────────

/// A compiled observation emitter: `(projected, t, counts, aux, rng) → y`, one
/// draw from `p(y | x_t, θ)`. Named so the sampler and its mean companion below
/// read as a pair.
pub type ObsSampleFn =
    Box<dyn Fn(f64, f64, &[i64], &[(String, f64)], &mut StatefulRng) -> f64>;

/// The mean companion: `(projected, t, counts, aux) → E[y | x_t, θ]`. The same
/// arguments minus the RNG — the type is what proves it draws nothing.
pub type ObsMeanFn = Box<dyn Fn(f64, f64, &[i64], &[(String, f64)]) -> f64>;

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
) -> ObsSampleFn {
    let resolved = resolve_likelihood_from_model(&obs_model.likelihood, &compiled)
        .unwrap_or_else(|e| panic!("observation likelihood resolution failed: {:?}", e));
    let params = params.to_vec();
    let real_s = RealState::new(compiled.real_local_to_global.len());
    let n_int  = compiled.int_local_to_global.len();

    Box::new(move |projected: f64, t: f64, counts: &[i64], aux: &[(String, f64)], rng: &mut StatefulRng| {
        // GH #6 fix: evaluate likelihood args against the real state,
        // not a zero-filled scratch. Caller is responsible for passing
        // the compartment snapshot at the obs time.
        assert_eq!(counts.len(), n_int,
            "compile_obs_sample_pf: counts length {} != expected {}", counts.len(), n_int);
        let int_s = IntState::from_vec(counts.to_vec());
        // Per-observation aux (e.g. a binomial denominator `n = tested`) is the
        // CALLER's to supply. `fit predict` forwards the OBSERVED aux at each obs
        // time, so the posterior-predictive draws `y_rep ~ binomial(n_observed,
        // p̂)` — the exogenous survey sample size carried forward. Data-free
        // emitters (`simulate --obs`, synthetic generation) pass `&[]`; a
        // likelihood that references an unavailable aux column then evaluates its
        // denominator to 0 and draws 0, the honest data-free behaviour.
        sample_obs_resolved(&resolved, t, projected, aux, &params, &compiled, &int_s, &real_s, rng)
    })
}

/// Build the mean companion of [`compile_obs_sample_pf`] (fixed params).
/// Takes (projected, t, counts, aux) → `E[y | x_t, θ]`, the value the
/// observation distribution is centred on **before** observation noise.
///
/// Same arguments, same resolution, same state contract as the sampler — the
/// two differ only in whether the noise is drawn, so this consumes no RNG and
/// is safe to call beside a sampler without perturbing a paired-seed replay.
/// It is a sibling rather than an extra field on the sampler's closure because
/// the two have different signatures (one needs an RNG, one must be provably
/// unable to consume one) and eight call sites want only the sampler.
///
/// The reason it is public: a diagnostic that reduces the predictive over
/// chains has to reduce the *mean*, not the draw. Observation noise lands in the
/// within-chain variance and drags a draw-based R̂ toward 1 however much the
/// chains disagree about the trajectory (gh#794).
pub fn compile_obs_mean_pf(
    obs_model: &ObservationModel,
    compiled: Arc<CompiledModel>,
    params: &[f64],
) -> ObsMeanFn {
    let resolved = resolve_likelihood_from_model(&obs_model.likelihood, &compiled)
        .unwrap_or_else(|e| panic!("observation likelihood resolution failed: {:?}", e));
    let params = params.to_vec();
    let real_s = RealState::new(compiled.real_local_to_global.len());
    let n_int = compiled.int_local_to_global.len();

    Box::new(move |projected: f64, t: f64, counts: &[i64], aux: &[(String, f64)]| {
        assert_eq!(counts.len(), n_int,
            "compile_obs_mean_pf: counts length {} != expected {}", counts.len(), n_int);
        let int_s = IntState::from_vec(counts.to_vec());
        eval_obs_mean_resolved(
            &resolved, t, projected, aux, &params, &compiled, &int_s, &real_s,
        )
    })
}

/// True iff any likelihood argument is NaN — typically `0/0` from a
/// collapsed-compartment denominator in a degenerate prior draw (gh#619).
/// Such a draw has no defined value; every arm of [`sample_obs_resolved`]
/// emits 0 (the honest dead-epidemic value, consistent with the documented
/// missing-aux behaviour) and counts the event so the end-of-run eval-stats
/// summary surfaces it. Counting here, not silently: a NaN that reached a
/// sampler used to abort the whole run (neg_binomial), draw ~1e15
/// (`rng.poisson`'s NaN.min(1e15) cap), or produce in-range garbage
/// (Beta-family shapes floored to epsilon).
#[inline]
fn obs_args_nan(vals: &[f64]) -> bool {
    if vals.iter().any(|v| v.is_nan()) {
        crate::eval_stats::inc_obs_sample_nan();
        true
    } else {
        false
    }
}

/// Draw `NegBinomial(mean, dispersion)` via the Gamma–Poisson mixture:
/// `g ~ Gamma(k, mean/k)`, `y ~ Poisson(g)`. Total over degenerate inputs
/// (gh#619) — callers guard NaN via [`obs_args_nan`] first:
///
/// - `mean <= 0` or `dispersion <= 0`: mass at zero → 0 (pre-existing
///   behaviour);
/// - `mean/k` under- or overflows even though both are positive (e.g. the
///   smallest subnormal mean over k = 500, or k = inf): the Gamma mixing
///   density is degenerate and `Gamma::new` rejects the scale — previously
///   an `unwrap()` that aborted the run mid-`--obs`-file. The exact
///   `k → ∞` limit is `Poisson(mean)`, the same fallback `rng::neg_binomial`
///   documents for its degenerate-shape regime, counted the same way.
fn draw_neg_binomial(mean: f64, dispersion: f64, rng: &mut StatefulRng) -> f64 {
    if mean <= 0.0 || dispersion <= 0.0 {
        return 0.0;
    }
    let scale = mean / dispersion;
    if scale <= 0.0 || !scale.is_finite() {
        crate::eval_stats::inc_neg_binomial_pois();
        return rng.poisson(mean) as f64;
    }
    let g = Gamma::new(dispersion, scale)
        .expect("guarded: shape and scale are positive and finite")
        .sample(rng.inner_mut());
    rng.poisson(g) as f64
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
        ResolvedLikelihood::NegBinomial { mean, dispersion, .. } => {
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            if obs_args_nan(&[m, k]) { return 0.0; }
            draw_neg_binomial(m, k, rng)
        }
        ResolvedLikelihood::Normal { mean, sd, .. } => {
            let m = eval_resolved(mean, &ctx(projected));
            let s = eval_resolved(sd, &ctx(projected));
            if obs_args_nan(&[m, s]) { return 0.0; }
            let draw = Normal::new(m, s.max(1e-10)).unwrap().sample(rng.inner_mut());
            draw.round().max(0.0)
        }
        ResolvedLikelihood::Poisson { rate, .. } => {
            let r = eval_resolved(rate, &ctx(projected));
            if obs_args_nan(&[r]) { return 0.0; }
            rng.poisson(r) as f64
        }
        ResolvedLikelihood::Binomial { n, p, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            if obs_args_nan(&[n_val, p_val]) { return 0.0; }
            rng.binomial(n_val.round().max(0.0) as u64, p_val.clamp(0.0, 1.0)) as f64
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, .. } => {
            // Draw BetaBinomial(n, alpha, beta): p ~ Beta(alpha, beta),
            // then k ~ Binomial(n, p). Uses the inner RNG directly for
            // the Beta draw (Gamma(a,1)/(Gamma(a,1)+Gamma(b,1))).
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_raw = eval_resolved(alpha, &ctx(projected));
            let beta_raw  = eval_resolved(beta,  &ctx(projected));
            if obs_args_nan(&[n_val, alpha_raw, beta_raw]) { return 0.0; }
            let alpha_val = alpha_raw.max(LOG_PROB_FLOOR);
            let beta_val  = beta_raw.max(LOG_PROB_FLOOR);
            let n_int = n_val.round().max(0.0) as u64;
            use rand_distr::{Gamma, Distribution};
            let inner = rng.inner_mut();
            let a = Gamma::new(alpha_val, 1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            let b = Gamma::new(beta_val,  1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            let p = a / (a + b);
            rng.binomial(n_int, p.clamp(0.0, 1.0)) as f64
        }
        ResolvedLikelihood::Beta { mean, concentration, .. } => {
            // Draw x ~ Beta(a, b), a = mean·φ, b = (1−mean)·φ, via the
            // Gamma(a,1)/(Gamma(a,1)+Gamma(b,1)) construction (same as the
            // BetaBinomial p-draw) — the continuous proportion, not a count.
            let m = eval_resolved(mean, &ctx(projected));
            let c = eval_resolved(concentration, &ctx(projected));
            if obs_args_nan(&[m, c]) { return 0.0; }
            let a = (m * c).max(LOG_PROB_FLOOR);
            let b = ((1.0 - m) * c).max(LOG_PROB_FLOOR);
            use rand_distr::{Distribution, Gamma};
            let inner = rng.inner_mut();
            let ga = Gamma::new(a, 1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            let gb = Gamma::new(b, 1.0).map(|d| d.sample(inner)).unwrap_or(1.0);
            ga / (ga + gb)
        }
        ResolvedLikelihood::Bernoulli { p, .. } => {
            // Clamp before sampling — an out-of-range p would
            // otherwise always-1 (p > 1) or always-0 (p < 0).
            let p_raw = eval_resolved(p, &ctx(projected));
            if obs_args_nan(&[p_raw]) { return 0.0; }
            let p_val = p_raw.clamp(0.0, 1.0);
            if rng.uniform() < p_val { 1.0 } else { 0.0 }
        }
        ResolvedLikelihood::ZeroInflatedNegBinomial { mean, dispersion, pi, .. } => {
            // With prob pi draw a structural zero; otherwise draw from the
            // NegBinomial base (Gamma-Poisson mixture, mirroring the NB arm).
            let pi_raw = eval_resolved(pi, &ctx(projected));
            if obs_args_nan(&[pi_raw]) { return 0.0; }
            let p = pi_raw.clamp(0.0, 1.0);
            if rng.uniform() < p { return 0.0; }
            let m = eval_resolved(mean, &ctx(projected));
            let k = eval_resolved(dispersion, &ctx(projected));
            if obs_args_nan(&[m, k]) { return 0.0; }
            draw_neg_binomial(m, k, rng)
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
        ResolvedLikelihood::Poisson { rate, .. } => {
            eval_resolved(rate, &ctx(projected))
        }
        ResolvedLikelihood::Binomial { n, p, .. } => {
            let n_val = eval_resolved(n, &ctx(projected));
            let p_val = eval_resolved(p, &ctx(projected));
            n_val * p_val
        }
        ResolvedLikelihood::BetaBinomial { n, alpha, beta, .. } => {
            // E[BetaBinomial(n, α, β)] = n · α / (α + β)
            let n_val = eval_resolved(n, &ctx(projected));
            let alpha_val = eval_resolved(alpha, &ctx(projected));
            let beta_val  = eval_resolved(beta,  &ctx(projected));
            let denom = (alpha_val + beta_val).max(LOG_PROB_FLOOR);
            n_val * (alpha_val / denom)
        }
        ResolvedLikelihood::Beta { mean, .. } => {
            // E[Beta(mean·φ, (1−mean)·φ)] = mean.
            eval_resolved(mean, &ctx(projected))
        }
        ResolvedLikelihood::Bernoulli { p, .. } => {
            eval_resolved(p, &ctx(projected))
        }
        ResolvedLikelihood::ZeroInflatedNegBinomial { mean, pi, .. } => {
            // E[Y] = (1 - pi)·E[NB] = (1 - pi)·mean.
            let m = eval_resolved(mean, &ctx(projected));
            let p = eval_resolved(pi, &ctx(projected)).clamp(0.0, 1.0);
            (1.0 - p) * m
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
