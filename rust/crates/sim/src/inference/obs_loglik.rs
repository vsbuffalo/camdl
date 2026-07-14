//! Observation log-likelihood functions.
//!
//! These evaluate log p(y | projected, θ) for a single observation.
//! No external dependencies — lgamma implemented inline for stability.

use std::f64::consts::PI;

// `lgamma` / `digamma` / `log_gamma_density` / `normal_cdf` / `normal_quantile`
// now live in the shared `numerics` crate (deduplicated with the standalone
// `external-harness`, whose probit copy had drifted). Re-exported here so
// existing `sim::inference::obs_loglik::*` / `sim::inference::*` call sites are
// unchanged.
pub use numerics::{digamma, lgamma, log_gamma_density, normal_cdf, normal_quantile};

/// Log-density of one overdispersion gamma multiplier `g ~ Γ(shape = dt/σ²,
/// scale = σ²/dt)`, the form recorded per overdispersed transition in a PGAS
/// trajectory.
///
/// THE single source of truth for both the PGAS value path
/// (`pgas::complete_data_loglik`) and the gradient-path energy
/// (`pgas_grad::complete_data_loglik_grad`'s returned `log_p`). Routing both
/// through one function makes `complete_data_loglik_grad(θ).0 ==
/// complete_data_loglik(θ).total` hold f64-exactly for the gamma term — the
/// gh#197 fix, pinned by the spine oracle in `gradient_check_overdisp.rs`.
///
/// `g.max(LOG_PROB_FLOOR)` floors ONLY the `ln(g)` term: a gamma draw is
/// strictly positive in practice, and the floor guards `ln(0)` for a degenerate
/// draw without the `-∞` cliff that [`log_gamma_density`] takes (a deliberate
/// difference — this is the overdispersion-multiplier convention, drift site #4
/// in the value/grad density proposal, resolved to the value side's floor).
pub(crate) fn gamma_multiplier_log_density(shape: f64, scale: f64, g: f64) -> f64 {
    (shape - 1.0) * g.max(crate::inference::types::LOG_PROB_FLOOR).ln()
        - g / scale
        - shape * scale.ln()
        - lgamma(shape)
}

/// Gradient of negbin_logpmf w.r.t. (mu, k).
///
/// d/d(mu) = y/mu - (y+k)/(mu+k)
/// d/d(k) = ψ(y+k) - ψ(k) + ln(k/(k+mu)) + 1 - (y+k)/(k+mu)
pub fn negbin_logpmf_grad(y: f64, mu: f64, k: f64) -> (f64, f64) {
    if mu <= 0.0 || k <= 0.0 { return (0.0, 0.0); }
    let y = y.round().max(0.0);
    let d_mu = y / mu - (y + k) / (mu + k);
    let d_k = digamma(y + k) - digamma(k)
            + (k / (k + mu)).ln() + 1.0 - (y + k) / (k + mu);
    (d_mu, d_k)
}

/// Gradient of `beta_binomial_logpmf` w.r.t. (alpha, beta).
///
/// With ψ = digamma and log P = logC(n,k) + lgamma(k+α) + lgamma(n−k+β)
/// + lgamma(α+β) − lgamma(n+α+β) − lgamma(α) − lgamma(β):
///
///   ∂logP/∂α = ψ(k+α) − ψ(α) − ψ(n+α+β) + ψ(α+β)
///   ∂logP/∂β = ψ(n−k+β) − ψ(β) − ψ(n+α+β) + ψ(α+β)
///
/// `n` is the (integer) number of trials; only `α`, `β` are continuous
/// likelihood parameters, so the gradient is taken w.r.t. them. `k` is the
/// observed count. The combinatorial `logC(n,k)` term has no α/β dependence
/// and drops out. Mirrors the `(d_mu, d_k)` shape of `negbin_logpmf_grad`.
///
/// Returns `(0.0, 0.0)` outside the domain (`α ≤ 0`, `β ≤ 0`, or `k > n`),
/// matching the value function's `NEG_INFINITY` clamp — the gradient of a
/// constant `-inf` floor is zero, consistent with the other helpers.
pub fn beta_binomial_logpmf_grad(k: f64, n: f64, alpha: f64, beta: f64) -> (f64, f64) {
    if alpha <= 0.0 || beta <= 0.0 || k > n { return (0.0, 0.0); }
    let k = k.round().max(0.0);
    let psi_apb = digamma(alpha + beta);
    let psi_napb = digamma(n + alpha + beta);
    let d_alpha = digamma(k + alpha) - digamma(alpha) - psi_napb + psi_apb;
    let d_beta = digamma(n - k + beta) - digamma(beta) - psi_napb + psi_apb;
    (d_alpha, d_beta)
}

/// Gradient of normal_logpdf w.r.t. (mu, sigma).
pub fn normal_logpdf_grad(y: f64, mu: f64, sigma: f64) -> (f64, f64) {
    if sigma <= 0.0 { return (0.0, 0.0); }
    let d_mu = (y - mu) / (sigma * sigma);
    let d_sigma = ((y - mu).powi(2) - sigma * sigma)
                / (sigma * sigma * sigma);
    (d_mu, d_sigma)
}

/// Gradient of `discretized_normal_logpmf_tol` w.r.t. (mean, variance).
///
/// gh#76 cleanup: this helper now uses the *same* erfc-stable `prob`
/// expression as `discretized_normal_logpmf_tol`. The two functions are
/// numerically symmetric: a NUTS Hamiltonian sees the same potential and
/// gradient regardless of which routine evaluates which term. The prior
/// asymmetry (value used erfc-difference, gradient used Φ-difference) was
/// a classical recipe for energy non-conservation and spurious divergences
/// in the tails.
///
/// Symbolically: d/dθ log P = (1/P) · dP/dθ. The numerator `dP/dθ` stays
/// in the φ-difference form (φ is well-conditioned across the real line,
/// and the gradient is *correctly* small when both φ values are tiny —
/// no rewrite needed). The denominator P uses the corrected erfc-stable
/// branches from `discretized_normal_logpmf_tol`.
///
/// Tol-floor behaviour: when `prob` clamps to `tol`, `log(prob) = log(tol)`
/// is a constant in θ, so the gradient is 0. This matches what a clean
/// FD against the value function would compute, restoring symmetry in the
/// regime where the value is at the floor.
pub fn discretized_normal_logpmf_grad(
    y: f64, mu: f64, variance: f64, tol: f64,
) -> (f64, f64) {
    let sigma = variance.sqrt().max(1e-10);
    let y = y.round().max(0.0);
    let z_hi = (y + 0.5 - mu) / sigma;
    let z_lo = (y - 0.5 - mu) / sigma;

    let npdf = |z: f64| (-0.5 * z * z).exp() / (2.0 * PI).sqrt();

    // erfc-stable prob — matches `discretized_normal_logpmf_tol` (q.v.
    // for the derivation of the corrected branches).
    let prob_raw = if y <= 0.5 {
        normal_cdf(z_hi)
    } else if z_lo + z_hi >= 0.0 {
        // Upper tail — Q-form: 0.5·(erfc(z_lo/√2) − erfc(z_hi/√2)).
        0.5 * (libm::erfc(z_lo / std::f64::consts::SQRT_2)
             - libm::erfc(z_hi / std::f64::consts::SQRT_2))
    } else {
        // Lower tail — P-form: 0.5·(erfc(−z_hi/√2) − erfc(−z_lo/√2)).
        0.5 * (libm::erfc(-z_hi / std::f64::consts::SQRT_2)
             - libm::erfc(-z_lo / std::f64::consts::SQRT_2))
    };

    // Tol-floor symmetry with the value function: when `prob` floors, the
    // value is the constant log(tol) in θ and the gradient is zero.
    if prob_raw <= tol {
        return (0.0, 0.0);
    }
    let prob = prob_raw;

    let dp_dmu = if y <= 0.5 {
        -npdf(z_hi) / sigma
    } else {
        (npdf(z_lo) - npdf(z_hi)) / sigma
    };
    let d_mu = dp_dmu / prob;

    // dz/d(var) = -z / (2·var), so dΦ(z)/d(var) = -φ(z)·z / (2·var)
    let dp_dvar = if y <= 0.5 {
        -npdf(z_hi) * z_hi / (2.0 * variance)
    } else {
        (-npdf(z_hi) * z_hi + npdf(z_lo) * z_lo) / (2.0 * variance)
    };
    let d_var = dp_dvar / prob;

    (d_mu, d_var)
}

/// Gradient of poisson_logpmf w.r.t. rate.
pub fn poisson_logpmf_grad(k: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 { return 0.0; }
    k / lambda - 1.0
}

/// Negative binomial log-PMF.
///
/// Parameterization: mean = mu, size = k (dispersion parameter).
/// As k → ∞, NegBin(mu, k) → Poisson(mu).
///
/// log p(y | mu, k) = lgamma(y+k) - lgamma(y+1) - lgamma(k)
///                   + k·log(k/(k+mu)) + y·log(mu/(k+mu))
pub fn negbin_logpmf(y: f64, mu: f64, k: f64) -> f64 {
    if mu <= 0.0 {
        return if y.round() == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if k <= 0.0 { return f64::NEG_INFINITY; }

    let y = y.round().max(0.0);
    let p = k / (k + mu);

    lgamma(y + k) - lgamma(y + 1.0) - lgamma(k)
        + k * p.ln()
        + y * (1.0 - p).ln()
}

/// Stable two-term log-sum-exp of already-logged terms:
/// `log(exp(la) + exp(lb))`, with `-inf` inputs handled.
fn log_add_exp(la: f64, lb: f64) -> f64 {
    if la == f64::NEG_INFINITY { return lb; }
    if lb == f64::NEG_INFINITY { return la; }
    let m = la.max(lb);
    m + ((la - m).exp() + (lb - m).exp()).ln()
}

/// Zero-inflated Negative-Binomial log-PMF.
///
/// A structural-zero mass `pi ∈ [0, 1]` mixed with `NegBinomial(mu, k)`:
///   `P(Y = 0)   = pi + (1 - pi)·f_NB(0)`
///   `P(Y = j>0) = (1 - pi)·f_NB(j)`
/// where `f_NB(·; mu, k)` is [`negbin_logpmf`]'s distribution. `pi = 0` recovers
/// the plain NegBinomial exactly. `pi` is clamped to `[0, 1]` so a numerical
/// excursion (e.g. an MH proposal stepping slightly out of range before the
/// chain concentrates) cannot produce a NaN log-probability.
pub fn zi_negbin_logpmf(y: f64, mu: f64, k: f64, pi: f64) -> f64 {
    let pi = pi.clamp(0.0, 1.0);
    let base = negbin_logpmf(y, mu, k); // log f_NB(y)
    if y.round() == 0.0 {
        let log_pi = if pi > 0.0 { pi.ln() } else { f64::NEG_INFINITY };
        let log_rest = if pi < 1.0 { (1.0 - pi).ln() + base } else { f64::NEG_INFINITY };
        log_add_exp(log_pi, log_rest)
    } else if pi < 1.0 {
        (1.0 - pi).ln() + base
    } else {
        // pi == 1: all mass at zero, so any positive count is impossible.
        f64::NEG_INFINITY
    }
}

/// Normal log-PDF.
///
/// log p(y | mu, sigma) = -0.5·((y-mu)/sigma)² - log(sigma) - 0.5·log(2π)
pub fn normal_logpdf(y: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 { return f64::NEG_INFINITY; }
    -0.5 * ((y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * (2.0 * PI).ln()
}

/// Discretized Normal log-PMF (He et al. 2010 observation model).
///
/// P(y | mean, variance) = Φ((y+0.5-μ)/σ) - Φ((y-0.5-μ)/σ)  for y > 0
///                        = Φ((0.5-μ)/σ)                       for y = 0
///
/// The ±0.5 continuity correction discretizes a continuous Normal
/// onto integer case counts. The variance is typically heteroscedastic:
///
///   variance = ρ·C·(1 - ρ + ψ²·ρ·C)
///
/// where C is the true incidence projection, ρ is reporting probability,
/// and ψ is the overdispersion coefficient. This gives tight observations
/// during inter-epidemic troughs (binomial sampling dominates) and loose
/// observations during peaks (correlated reporting noise dominates).
/// Default likelihood tolerance — matches pomp's `tol` parameter.
/// Exposed as `--tol` on the CLI for models where a different floor is needed.
pub const DEFAULT_TOL: f64 = 1e-18;

pub fn discretized_normal_logpmf(y: f64, mean: f64, variance: f64) -> f64 {
    discretized_normal_logpmf_tol(y, mean, variance, DEFAULT_TOL)
}

/// Discretized Normal log-PMF with configurable tolerance floor.
///
/// `tol` is the minimum probability before taking log. At 1e-18 (pomp's
/// default), particles that predict ~0 when data shows 80 get log-weight
/// ≈ -41 regardless of exactly how wrong they are. At 1e-300, the gap
/// between "zero" and "nearly zero" is 650 log-units, which collapses ESS.
///
/// For large-population models (London measles), 1e-18 is correct.
/// For small-population models where observing 3 vs 0 is informative,
/// a tighter tolerance (e.g., 1e-30) preserves that signal.
pub fn discretized_normal_logpmf_tol(y: f64, mean: f64, variance: f64, tol: f64) -> f64 {
    let sd = variance.max(1e-30).sqrt();
    let y = y.round().max(0.0);

    // gh#audit-H2 + gh#76 cleanup. erfc-based interval that subtracts
    // two *small* values rather than two *near-1* values:
    //
    //   For z ≥ 0:  Φ(z) = 1 − 0.5·erfc(z/√2),  Q(z) := 0.5·erfc(z/√2) is small.
    //   For z ≤ 0:  Φ(z) = 0.5·erfc(−z/√2),    P(z) := 0.5·erfc(−z/√2) is small.
    //
    //   * Upper tail (z_lo + z_hi ≥ 0): Φ(z_hi) − Φ(z_lo)
    //     = (1 − Q(z_hi)) − (1 − Q(z_lo)) = Q(z_lo) − Q(z_hi)
    //     = 0.5·(erfc(z_lo/√2) − erfc(z_hi/√2))
    //   * Lower tail (z_lo + z_hi < 0): Φ(z_hi) − Φ(z_lo) = P(z_hi) − P(z_lo)
    //     = 0.5·(erfc(−z_hi/√2) − erfc(−z_lo/√2))
    //
    // Both forms feed erfc with *positive* arguments — erfc(large positive)
    // is tiny, no cancellation against 2.0. Verified against an mpmath-
    // precise reference: at z_lo=9.5, z_hi=10 the prior form returned 0
    // (cancellation against 2.0); this form returns 1.04e-21 (correct).
    //
    // The prior audit-H2 (b981d60^) branches had the formulas swapped —
    // both branches called erfc with negative args, which gives erfc ≈
    // 2 − tiny → cancellation when subtracted, defeating the audit-H2
    // intent. See docs/dev/notes/2026-05-25-pgas-obs-grad-derivation.md
    // for the diagnostic that surfaced this during the gh#76 cleanup.
    let prob = if y > 0.0 {
        let z_lo = (y - 0.5 - mean) / sd;
        let z_hi = (y + 0.5 - mean) / sd;
        let p = if z_lo + z_hi >= 0.0 {
            // Upper tail — both Φ values near 1. Use Q-form: erfc of
            // positive args (since z_lo + z_hi ≥ 0 ⇒ z_hi > 0; z_lo
            // may be ≤ 0 in the straddle case, where Q(z_lo) = 1 − tiny,
            // still no cancellation against the small Q(z_hi)).
            0.5 * (libm::erfc(z_lo / std::f64::consts::SQRT_2)
                 - libm::erfc(z_hi / std::f64::consts::SQRT_2))
        } else {
            // Lower tail — both Φ values near 0. Use P-form: erfc of
            // negated args (since z_lo + z_hi < 0 ⇒ z_lo < 0; the −z's
            // are positive, erfc returns tiny values).
            0.5 * (libm::erfc(-z_hi / std::f64::consts::SQRT_2)
                 - libm::erfc(-z_lo / std::f64::consts::SQRT_2))
        };
        p.max(tol)
    } else {
        normal_cdf((0.5 - mean) / sd).max(tol)
    };

    prob.ln()
}

/// Binomial log-PMF: log P(X = k) where X ~ Binom(n, p).
///
/// log p(k | n, p) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
///                  + k·log(p) + (n-k)·log(1-p)
///
/// Used by PGAS for transition density evaluation.
pub fn binom_logpmf(k: u64, n: u64, p: f64) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    if p <= 0.0 { return if k == 0 { 0.0 } else { f64::NEG_INFINITY }; }
    if p >= 1.0 { return if k == n { 0.0 } else { f64::NEG_INFINITY }; }
    lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0)
        + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()
}

/// Beta-Binomial log-PMF.
///
/// log p(k | n, alpha, beta) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
///                            + log B(k+alpha, n-k+beta) - log B(alpha, beta)
///
/// where log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b).
///
/// Models overdispersed count observations when the per-trial
/// success probability itself varies (e.g., household- or
/// cluster-level variation in reporting probability).
///
/// IC1 in the 2026-04-19 inference review: previously this was
/// a `log::warn!` + `-inf` stub that made every BetaBinomial
/// observation corrupt the fit.
pub fn beta_binomial_logpmf(k: u64, n: u64, alpha: f64, beta: f64) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    if alpha <= 0.0 || beta <= 0.0 { return f64::NEG_INFINITY; }
    let lbeta = |a: f64, b: f64| lgamma(a) + lgamma(b) - lgamma(a + b);
    lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0)
        + lbeta(k as f64 + alpha, (n - k) as f64 + beta)
        - lbeta(alpha, beta)
}

/// Poisson log-PMF.
///
/// log p(y | lambda) = y·log(lambda) - lambda - lgamma(y+1)
pub fn poisson_logpmf(y: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if y.round() == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    let y = y.round().max(0.0);
    y * lambda.ln() - lambda - lgamma(y + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_inverts_normal_cdf() {
        // Φ⁻¹(Φ(x)) ≈ x across the central range and into the tails.
        for &x in &[-3.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let round = normal_quantile(normal_cdf(x));
            assert!((round - x).abs() < 1e-4, "Φ⁻¹(Φ({})) = {}", x, round);
        }
        // Known quantiles.
        assert!((normal_quantile(0.5)).abs() < 1e-9, "median is 0");
        assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-4, "97.5% ≈ 1.96");
    }

    #[test]
    fn test_lgamma_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(5) = log(24) = 3.178...
        assert!((lgamma(1.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(2.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        assert!((lgamma(0.5) - (PI.sqrt().ln())).abs() < 1e-10);
    }

    #[test]
    fn test_negbin_logpmf_known_values() {
        // Reference: Python math.lgamma-based computation
        // negbin(10, mu=20, k=5) = -3.369870
        let ll = negbin_logpmf(10.0, 20.0, 5.0);
        assert!((ll - (-3.369870)).abs() < 1e-4,
            "negbin_logpmf(10, 20, 5) = {}, expected -3.370", ll);

        // negbin(0, mu=5, k=2): p = 2/7, ll = lgamma(2)-lgamma(1)-lgamma(2) + 2*ln(2/7)
        let ll = negbin_logpmf(0.0, 5.0, 2.0);
        let expected = 2.0 * (2.0_f64 / 7.0).ln();
        assert!((ll - expected).abs() < 1e-4,
            "negbin_logpmf(0, 5, 2) = {}, expected {}", ll, expected);
    }

    #[test]
    fn test_normal_logpdf_known() {
        // N(0, 1): log p(0) = -0.5*log(2π) = -0.9189
        let ll = normal_logpdf(0.0, 0.0, 1.0);
        assert!((ll - (-0.9189385)).abs() < 1e-5);

        // N(5, 2): log p(5) = -log(2) - 0.5*log(2π) = -1.612
        let ll = normal_logpdf(5.0, 5.0, 2.0);
        assert!((ll - (-1.612086)).abs() < 1e-4);
    }

    #[test]
    fn test_poisson_logpmf_known() {
        // poisson(5, lambda=3): 5*ln(3) - 3 - lgamma(6) = -2.2944
        let ll = poisson_logpmf(5.0, 3.0);
        assert!((ll - (-2.2944)).abs() < 1e-3,
            "poisson_logpmf(5, 3) = {}, expected -2.294", ll);
    }

    #[test]
    fn test_negbin_mu_zero_y_zero() {
        assert_eq!(negbin_logpmf(0.0, 0.0, 5.0), 0.0);
    }

    #[test]
    fn test_negbin_mu_zero_y_nonzero() {
        assert_eq!(negbin_logpmf(10.0, 0.0, 5.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_negbin_mu_zero_k_zero_y_zero() {
        // In4 in 2026-04-19 inference review: NegBin(μ=0, k=0) is
        // ill-defined (Gamma(0,·) is degenerate), but for y=0 the
        // μ==0 branch short-circuits to 0. Document the contract so
        // callers know "no cases expected, none observed" is a
        // degeneracy-safe log-prob of 0 regardless of k.
        assert_eq!(negbin_logpmf(0.0, 0.0, 0.0), 0.0);
        // y > 0 with μ=0 is still impossible, k irrelevant.
        assert_eq!(negbin_logpmf(1.0, 0.0, 0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_zi_negbin_reduces_to_negbin_at_pi_zero() {
        // pi = 0 must recover the plain NegBinomial exactly.
        for &(y, mu, k) in &[(0.0, 5.0, 2.0), (3.0, 5.0, 2.0), (10.0, 5.0, 2.0)] {
            let zi = zi_negbin_logpmf(y, mu, k, 0.0);
            let nb = negbin_logpmf(y, mu, k);
            assert!((zi - nb).abs() < 1e-12, "y={y}: zi={zi} nb={nb}");
        }
    }

    #[test]
    fn test_zi_negbin_mixture_math_and_normalization() {
        let (mu, k, pi) = (5.0, 2.0, 0.3);
        // P(Y=0) = pi + (1-pi)·f_NB(0).
        let f0 = negbin_logpmf(0.0, mu, k).exp();
        let expected0 = (pi + (1.0 - pi) * f0).ln();
        let got0 = zi_negbin_logpmf(0.0, mu, k, pi);
        assert!((got0 - expected0).abs() < 1e-12, "P(0): got {got0} expected {expected0}");
        // Negative control: zero-inflation must RAISE P(0) above the base NB —
        // catches a `pi` silently dropped from the zero branch.
        assert!(got0 > negbin_logpmf(0.0, mu, k), "zero-inflation must raise P(0)");
        // P(Y=2) = (1-pi)·f_NB(2): base shifted down by log(1-pi), and finite.
        let got2 = zi_negbin_logpmf(2.0, mu, k, pi);
        let expected2 = (1.0 - pi).ln() + negbin_logpmf(2.0, mu, k);
        assert!((got2 - expected2).abs() < 1e-12, "P(2): got {got2} expected {expected2}");
        assert!(got2.is_finite());
        // Boundary: pi = 1 puts all mass at 0.
        assert_eq!(zi_negbin_logpmf(0.0, mu, k, 1.0), 0.0);
        assert_eq!(zi_negbin_logpmf(1.0, mu, k, 1.0), f64::NEG_INFINITY);
        // The whole PMF must sum to 1 over a wide support — the strongest check
        // that the mixture is a proper distribution.
        let total: f64 = (0..200).map(|j| zi_negbin_logpmf(j as f64, mu, k, pi).exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "ZI-NB pmf must sum to 1; got {total}");
    }

    #[test]
    fn test_beta_binomial_known_values() {
        // IC1 regression: BetaBinomial(k=5 | n=10, α=2, β=3).
        //   log C(10,5) = log 252 ≈ 5.5294
        //   log B(7, 8) = lgamma(7)+lgamma(8)-lgamma(15) ≈ -10.087
        //   log B(2, 3) = lgamma(2)+lgamma(3)-lgamma(5)   ≈ -2.485
        //   ll = 5.5294 - 10.087 + 2.485 ≈ -2.073
        let ll = beta_binomial_logpmf(5, 10, 2.0, 3.0);
        assert!((ll - (-2.072473)).abs() < 1e-5,
            "beta_binomial_logpmf(5, 10, 2, 3) = {}, expected -2.072", ll);

        // k > n is -inf.
        assert_eq!(beta_binomial_logpmf(11, 10, 2.0, 3.0), f64::NEG_INFINITY);
        // α <= 0 or β <= 0 is -inf.
        assert_eq!(beta_binomial_logpmf(5, 10, 0.0, 3.0), f64::NEG_INFINITY);
        assert_eq!(beta_binomial_logpmf(5, 10, 2.0, -1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_normal_cdf_known() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
        assert!(normal_cdf(10.0) > 0.9999);
        assert!(normal_cdf(-10.0) < 0.0001);
    }

    #[test]
    fn test_discretized_normal_matches_scipy() {
        // Reference: scipy.stats.norm.cdf with He et al. variance formula
        // variance = rho * C * (1 - rho + psi^2 * rho * C)
        // rho=0.488, psi=0.116

        // y=100, C=200: mean=97.6, var=178.1 (trough — tight observation)
        assert!((discretized_normal_logpmf(100.0, 97.6, 178.1) - (-3.526643)).abs() < 1e-2,
            "y=100, C=200");

        // y=0, C=5: mean=2.4, var=1.3 (near-zero — very tight, tail of CDF)
        assert!((discretized_normal_logpmf(0.0, 2.4, 1.3) - (-3.074161)).abs() < 0.05,
            "y=0, C=5: got {}", discretized_normal_logpmf(0.0, 2.4, 1.3));

        // y=500, C=1000: mean=488.0, var=3454.3 (moderate incidence)
        assert!((discretized_normal_logpmf(500.0, 488.0, 3454.3) - (-5.013484)).abs() < 1e-2,
            "y=500, C=1000");

        // y=10, C=20: mean=9.8, var=6.3 (low count, binomial regime)
        assert!((discretized_normal_logpmf(10.0, 9.8, 6.3) - (-1.848681)).abs() < 1e-2,
            "y=10, C=20");

        // y=2000, C=4000: mean=1952.0, var=52270.9 (peak — loose observation)
        assert!((discretized_normal_logpmf(2000.0, 1952.0, 52270.9) - (-6.373076)).abs() < 1e-2,
            "y=2000, C=4000");
    }

    #[test]
    fn test_discretized_normal_zero_variance_safe() {
        // Should not panic or return NaN
        let ll = discretized_normal_logpmf(5.0, 5.0, 0.0);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_binom_logpmf_known() {
        // Binom(5, 10, 0.3): lgamma-based = -2.2738
        let ll = binom_logpmf(5, 10, 0.3);
        assert!((ll - (-2.2738)).abs() < 1e-3,
            "binom_logpmf(5, 10, 0.3) = {}, expected -2.274", ll);
    }

    #[test]
    fn test_binom_logpmf_boundaries() {
        assert_eq!(binom_logpmf(0, 10, 0.0), 0.0);
        assert_eq!(binom_logpmf(5, 10, 0.0), f64::NEG_INFINITY);
        assert_eq!(binom_logpmf(10, 10, 1.0), 0.0);
        assert_eq!(binom_logpmf(5, 10, 1.0), f64::NEG_INFINITY);
        assert_eq!(binom_logpmf(11, 10, 0.5), f64::NEG_INFINITY);
        // Binom(0, 0, p) = 1 for any p (within floating point tolerance)
        assert!((binom_logpmf(0, 0, 0.5)).abs() < 1e-14);
    }

    #[test]
    fn test_digamma_known_values() {
        assert!((digamma(1.0) - (-0.5772156649)).abs() < 1e-9);
        assert!((digamma(2.0) - 0.4227843351).abs() < 1e-9);
        assert!((digamma(0.5) - (-1.9635100260)).abs() < 1e-9);
        assert!((digamma(10.0) - 2.2517525890).abs() < 1e-8);
    }

    #[test]
    fn test_digamma_recurrence() {
        for x in [0.5, 1.0, 2.5, 7.3, 15.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!((lhs - rhs).abs() < 1e-10,
                "recurrence failed at x={}: {} vs {}", x, lhs, rhs);
        }
    }

    #[test]
    fn test_log_gamma_density() {
        // Gamma(shape=2, scale=3): p(3|2,3) = 3*exp(-1)/9 = exp(-1)/3
        let ld = log_gamma_density(3.0, 2.0, 3.0);
        let expected = (-1.0_f64).exp() / 3.0;
        assert!((ld.exp() - expected).abs() < 1e-6);
    }

    /// gh#20 defense-in-depth: scipy-anchored VALUE check for the gamma
    /// multiplier-density formula. The PGAS σ² (process-noise) gradient
    /// test in `gradient_check_obs.rs` validates grad-vs-value consistency
    /// (FD of `log_gamma_density`); it canNOT catch a wrong *value* formula
    /// — a constant offset or a shape/scale swap differentiates correctly
    /// and the FD test stays green. This pins the value against scipy.
    ///
    /// `log_gamma_density(x, shape, scale)` is the standalone form of the
    /// gamma multiplier density. The PGAS substep at `pgas.rs:649` computes
    /// the same density inline for the overdispersion gamma `g`
    /// (`shape = dt/σ²`, `scale = σ²/dt`). The two forms are identical
    /// algebra today (verified by reading both at this commit), so pinning
    /// this helper's value pins the formula they share; this test does not,
    /// by itself, prevent the inline copy from drifting independently.
    ///
    /// Reference values:
    /// ```python
    /// from scipy.stats import gamma
    /// gamma.logpdf(x, a=shape, scale=scale)
    /// ```
    #[test]
    fn test_log_gamma_density_matches_scipy() {
        // (x, shape, scale, scipy gamma.logpdf)
        let cases: [(f64, f64, f64, f64); 4] = [
            (3.0, 2.0, 3.0, -2.09861229),
            (1.0, 0.5, 2.0, -1.41893853),
            (5.0, 4.0, 0.5, -4.19085701),
            (0.8, 2.5, 1.2, -1.74186876),
        ];
        for (x, shape, scale, ref_lp) in cases {
            let lp = log_gamma_density(x, shape, scale);
            assert!((lp - ref_lp).abs() <= 1e-6,
                "log_gamma_density({x}, {shape}, {scale}) = {lp}, scipy ref {ref_lp}");
        }
    }

    #[test]
    fn test_negbin_grad_vs_fd() {
        let (y, mu, k) = (5.0, 10.0, 3.0);
        let eps = 1e-6;
        let (d_mu, d_k) = negbin_logpmf_grad(y, mu, k);
        let fd_mu = (negbin_logpmf(y, mu + eps, k) - negbin_logpmf(y, mu - eps, k)) / (2.0 * eps);
        let fd_k = (negbin_logpmf(y, mu, k + eps) - negbin_logpmf(y, mu, k - eps)) / (2.0 * eps);
        assert!((d_mu - fd_mu).abs() < 1e-5, "d_mu: {} vs fd {}", d_mu, fd_mu);
        assert!((d_k - fd_k).abs() < 1e-5, "d_k: {} vs fd {}", d_k, fd_k);
    }

    #[test]
    fn test_normal_grad_vs_fd() {
        let (y, mu, sigma) = (3.5, 2.0, 1.5);
        let eps = 1e-6;
        let (d_mu, d_sigma) = normal_logpdf_grad(y, mu, sigma);
        let fd_mu = (normal_logpdf(y, mu + eps, sigma) - normal_logpdf(y, mu - eps, sigma)) / (2.0 * eps);
        let fd_sigma = (normal_logpdf(y, mu, sigma + eps) - normal_logpdf(y, mu, sigma - eps)) / (2.0 * eps);
        assert!((d_mu - fd_mu).abs() < 1e-5);
        assert!((d_sigma - fd_sigma).abs() < 1e-5);
    }

    #[test]
    fn test_discretized_normal_grad_vs_fd() {
        let (y, mu, var) = (15.0, 12.0, 25.0);
        let eps = 1e-5;
        let tol = 1e-17;
        let (d_mu, d_var) = discretized_normal_logpmf_grad(y, mu, var, tol);
        let fd_mu = (discretized_normal_logpmf_tol(y, mu + eps, var, tol)
                   - discretized_normal_logpmf_tol(y, mu - eps, var, tol)) / (2.0 * eps);
        let fd_var = (discretized_normal_logpmf_tol(y, mu, var + eps, tol)
                    - discretized_normal_logpmf_tol(y, mu, var - eps, tol)) / (2.0 * eps);
        assert!((d_mu - fd_mu).abs() / fd_mu.abs().max(1e-10) < 1e-3,
            "d_mu: {} vs fd {}", d_mu, fd_mu);
        assert!((d_var - fd_var).abs() / fd_var.abs().max(1e-10) < 1e-3,
            "d_var: {} vs fd {}", d_var, fd_var);
    }

    /// gh#76 cleanup: tail FD points exercise the regime where the prior
    /// Φ-difference denominator collapsed to numerical noise (audit-H2 on
    /// the value side; ported here on the gradient side). After the port,
    /// the value and the gradient share the same erfc-stable `prob`, so
    /// FD-vs-analytic must agree at the same 1e-4 bar that the near-mode
    /// test holds at.
    ///
    /// μ = 50, σ² = 4 (σ = 2). Tail points at 3σ, 5σ, 8σ on the upper
    /// side; the lower-tail mirror is exercised by `_lower_tail` below.
    #[test]
    fn test_discretized_normal_grad_vs_fd_upper_tail_3sigma() {
        check_discretized_normal_grad_tail(50.0 + 3.0 * 2.0, 50.0, 4.0, 1e-4);
    }

    #[test]
    fn test_discretized_normal_grad_vs_fd_upper_tail_5sigma() {
        check_discretized_normal_grad_tail(50.0 + 5.0 * 2.0, 50.0, 4.0, 1e-4);
    }

    #[test]
    fn test_discretized_normal_grad_vs_fd_upper_tail_8sigma() {
        check_discretized_normal_grad_tail(50.0 + 8.0 * 2.0, 50.0, 4.0, 1e-4);
    }

    #[test]
    fn test_discretized_normal_grad_vs_fd_lower_tail_5sigma() {
        // y > 0 lower-tail branch (z_lo + z_hi < 0). μ large enough that
        // y = μ − 5σ is still positive (and rounds to a positive integer).
        check_discretized_normal_grad_tail(50.0 - 5.0 * 2.0, 50.0, 4.0, 1e-4);
    }

    fn check_discretized_normal_grad_tail(y: f64, mu: f64, var: f64, rel_tol: f64) {
        let tol = 1e-30;
        // FD step chosen relative to the parameter being varied, not the
        // value being observed.
        let eps_mu  = 1e-5 * mu.abs().max(1.0);
        let eps_var = 1e-5 * var.abs().max(1.0);

        let (d_mu, d_var) = discretized_normal_logpmf_grad(y, mu, var, tol);
        let fd_mu = (discretized_normal_logpmf_tol(y, mu + eps_mu, var, tol)
                   - discretized_normal_logpmf_tol(y, mu - eps_mu, var, tol)) / (2.0 * eps_mu);
        let fd_var = (discretized_normal_logpmf_tol(y, mu, var + eps_var, tol)
                    - discretized_normal_logpmf_tol(y, mu, var - eps_var, tol)) / (2.0 * eps_var);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-10);
        let r_mu  = rel(d_mu, fd_mu);
        let r_var = rel(d_var, fd_var);
        eprintln!("[tail-grad] y={:.2}, μ={:.2}, σ²={:.2}", y, mu, var);
        eprintln!("  d/dμ:   analytic={:.6e}  fd={:.6e}  rel={:.2e}", d_mu, fd_mu, r_mu);
        eprintln!("  d/dσ²:  analytic={:.6e}  fd={:.6e}  rel={:.2e}", d_var, fd_var, r_var);
        assert!(r_mu  < rel_tol, "d/dμ at y={}: rel_err {:.2e} > tol {:.0e}",  y, r_mu,  rel_tol);
        assert!(r_var < rel_tol, "d/dσ² at y={}: rel_err {:.2e} > tol {:.0e}", y, r_var, rel_tol);
    }

    #[test]
    fn test_poisson_grad_vs_fd() {
        let (k, lambda) = (7.0, 5.0);
        let eps = 1e-6;
        let d = poisson_logpmf_grad(k, lambda);
        let fd = (poisson_logpmf(k, lambda + eps) - poisson_logpmf(k, lambda - eps)) / (2.0 * eps);
        assert!((d - fd).abs() < 1e-5);
    }

    /// EXTERNAL (absolute-correctness) validation of the BetaBinomial
    /// gradient helper against scipy. Reference values computed with:
    ///
    /// ```python
    /// from scipy.special import digamma
    /// def grad(n, k, a, b):
    ///     da = digamma(k+a) - digamma(a) - digamma(n+a+b) + digamma(a+b)
    ///     db = digamma(n-k+b) - digamma(b) - digamma(n+a+b) + digamma(a+b)
    ///     return da, db
    /// ```
    ///
    /// Each (∂α, ∂β) below was further cross-checked against a central
    /// finite difference of `scipy.stats.betabinom.logpmf(k, n, a, b)`
    /// (varying a, b) — analytic-vs-scipy-FD agreement was ≤ 1e-7 relative,
    /// so these constants pin the helper to scipy's PMF, not just to the
    /// digamma identity.
    #[test]
    fn test_beta_binomial_grad_matches_scipy() {
        // (n, k, alpha, beta, d_alpha, d_beta)
        let cases: [(f64, f64, f64, f64, f64, f64); 4] = [
            (20.0,  7.0, 2.0, 3.0,  0.02523230,  0.12560415),
            (50.0,  5.0, 0.8, 4.0,  0.10049902,  0.09177322),
            (10.0,  9.0, 5.0, 1.5,  0.11696038, -0.31317337),
            (100.0, 50.0, 2.0, 2.0, 0.13535535,  0.13535535),
        ];
        for (n, k, alpha, beta, ref_da, ref_db) in cases {
            let (da, db) = beta_binomial_logpmf_grad(k, n, alpha, beta);
            assert!((da - ref_da).abs() <= 1e-7,
                "d_alpha(n={n}, k={k}, α={alpha}, β={beta}) = {da}, scipy ref {ref_da}");
            assert!((db - ref_db).abs() <= 1e-7,
                "d_beta(n={n}, k={k}, α={alpha}, β={beta}) = {db}, scipy ref {ref_db}");
        }
    }

    /// INTERNAL (self-consistency) check: the analytic helper matches a
    /// central finite difference of `beta_binomial_logpmf`, confirming the
    /// gradient differentiates the *implemented* value function (the unit
    /// uses integer n/k via u64; FD over α, β with k, n held fixed).
    #[test]
    fn test_beta_binomial_grad_vs_fd() {
        let eps = 1e-6;
        for &(n, k, alpha, beta) in &[
            (20u64, 7u64, 2.0, 3.0),
            (50, 5, 0.8, 4.0),
            (10, 9, 5.0, 1.5),
        ] {
            let (d_a, d_b) = beta_binomial_logpmf_grad(k as f64, n as f64, alpha, beta);
            let fd_a = (beta_binomial_logpmf(k, n, alpha + eps, beta)
                      - beta_binomial_logpmf(k, n, alpha - eps, beta)) / (2.0 * eps);
            let fd_b = (beta_binomial_logpmf(k, n, alpha, beta + eps)
                      - beta_binomial_logpmf(k, n, alpha, beta - eps)) / (2.0 * eps);
            assert!((d_a - fd_a).abs() < 1e-5, "d_alpha: {d_a} vs fd {fd_a}");
            assert!((d_b - fd_b).abs() < 1e-5, "d_beta: {d_b} vs fd {fd_b}");
        }
    }

    /// Out-of-domain returns (0, 0) — the gradient of the constant -inf
    /// floor that `beta_binomial_logpmf` produces there.
    #[test]
    fn test_beta_binomial_grad_out_of_domain() {
        assert_eq!(beta_binomial_logpmf_grad(5.0, 10.0, 0.0, 3.0), (0.0, 0.0)); // α ≤ 0
        assert_eq!(beta_binomial_logpmf_grad(5.0, 10.0, 2.0, -1.0), (0.0, 0.0)); // β ≤ 0
        assert_eq!(beta_binomial_logpmf_grad(11.0, 10.0, 2.0, 3.0), (0.0, 0.0)); // k > n
    }
}
