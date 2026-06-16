//! Prior distributions for Bayesian inference.
//!
//! The `Prior` enum carries distribution parameters; `log_density` evaluates
//! log-density at a parameter value; `from_ir` converts from the IR's
//! serialized form (populated from DSL `~` syntax or fit.toml).
//!
//! # Parameterization conventions
//!
//! - `log_normal(mu, sigma)`: mu and sigma on the **log scale**.
//!   `log(X) ~ Normal(mu, sigma)`. Median of X is `exp(mu)`.
//! - `half_normal(sigma)`: sigma is the SD of the underlying (unfolded) normal.
//! - `gamma(shape, rate)`: rate parameterization. `E[X] = shape/rate`.
//! - `exponential(rate)`: `E[X] = 1/rate`.
//! - `beta(alpha, beta)`: shape parameters on [0, 1].
//! - `normal(mean, sd)`: natural scale.
//! - `uniform(lower, upper)`: uniform density on [lower, upper].

use crate::inference::obs_loglik::{lgamma, normal_cdf};

/// 0.5 · ln(2π), used in Gaussian log-densities.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// Scale tag for `log_density` and the hierarchical evaluator.
///
/// The phantom distinction makes it an explicit function argument whether
/// callers are passing a natural (θ) or transformed (z = f(θ)) value.
/// Primary payoff: IC3 Jacobian double-count was a class of bug where
/// `TransformedNormal` returned a z-scale density while callers also
/// added `log|dθ/dz|` — this tag documents the contract at every call
/// site so future contributors can't silently repeat the mistake.
#[derive(Debug, Clone, Copy)]
pub enum Scale {
    /// θ — the natural parameter value (rate, probability, etc.).
    Natural,
    /// z = f(θ) — the unconstrained-transform value (e.g. z = ln θ).
    Transformed,
}

/// Prior distribution for one estimated parameter.
#[derive(Clone, Debug)]
pub enum Prior {
    /// Flat (improper) prior — log-density = 0 everywhere within transform bounds.
    Flat,
    /// Uniform(lower, upper) on natural scale. Flat within bounds, -inf outside.
    Uniform { lower: f64, upper: f64 },
    /// Normal(mean, sd) on the natural scale.
    Normal { mean: f64, sd: f64 },
    /// Normal(mean, sd) on the transformed (log/logit) scale.
    /// This is the "log_normal" when the param uses log transform.
    TransformedNormal { mean: f64, sd: f64 },
    /// Half-Normal(sigma): folded normal supported on [0, inf).
    HalfNormal { sigma: f64 },
    /// Beta(alpha, beta) on [0, 1]. For probability parameters.
    Beta { alpha: f64, beta: f64 },
    /// Gamma(shape, rate). Supported on (0, inf).
    Gamma { shape: f64, rate: f64 },
    /// Exponential(rate). Supported on [0, inf).
    Exponential { rate: f64 },
    /// Log-Uniform(lower, upper): uniform on the log scale, supported on
    /// [lower, upper] with `lower, upper > 0`. Density on the natural scale
    /// is `1 / (θ · (ln upper − ln lower))`. Like `TransformedNormal`, it
    /// pairs with the `Log` transform; on the z = ln θ scale it is flat.
    LogUniform { lower: f64, upper: f64 },
    /// Normal(mean, sd) truncated to [lower, upper]. Natural-scale density.
    TruncatedNormal { mean: f64, sd: f64, lower: f64, upper: f64 },
    /// Hierarchical prior: args are expressions that reference other
    /// parameters (hyperparents). Wave 2 / malaria #3 Gate 3. At each
    /// log-density call, args are resolved against a
    /// [`crate::inference::hierarchical::ParamEnv`]. The density
    /// function lives in the `hierarchical` module; this variant is
    /// a thin carrier for the IR data.
    Hierarchical(ir::parameter::HierarchicalPrior),
}

impl Prior {
    /// Log-density of the prior on the **natural** scale, `log p(θ)`.
    /// `transformed` is the unconstrained-scale value z where θ = f(z)
    /// — used by `TransformedNormal`, which evaluates
    /// `log N(z; mu, sd)` then subtracts `z` (for the Log transform
    /// Jacobian) so the return is the natural-scale log-density.
    ///
    /// IC3 fix (2026-04-19 inference review batch 2/3): previously
    /// `TransformedNormal` returned the z-scale density
    /// `log N(z; μ, σ)` while callers (PMMH `pmmh.rs:419-420`,
    /// PGAS `pgas.rs:1533-1534`) also added `log_jacobian(z) = z`
    /// on top, double-counting the Jacobian and producing a +σ²
    /// systematic bias on log-scale posteriors. The fix returns the
    /// natural-scale density here; callers continue to add
    /// `log_jacobian(z)` unconditionally to get the z-scale
    /// density, now correctly.
    ///
    /// Precondition: `TransformedNormal` is only meaningful when the
    /// parameter uses `Transform::Log`. IC4's validator
    /// (`fit/config.rs::validate_prior_transform_compat`) enforces
    /// this at fit-config load time.
    pub fn log_density(&self, natural: f64, transformed: f64) -> f64 {
        match self {
            Prior::Flat => 0.0,
            Prior::Uniform { lower, upper } => {
                if natural < *lower || natural > *upper {
                    f64::NEG_INFINITY
                } else {
                    -((upper - lower).ln())
                }
            }
            Prior::Normal { mean, sd } => {
                let z = (natural - mean) / sd;
                // Full normal log-density: -0.5 ln(2π) - ln(σ) - 0.5 z²
                -HALF_LN_2PI - sd.ln() - 0.5 * z * z
            }
            Prior::TransformedNormal { mean, sd } => {
                // Log-normal on natural scale:
                //   log p(θ) = log N(log θ; μ, σ) − log θ
                // With z = log θ (Log transform) this is
                //   log N(z; μ, σ) − z
                // The −z compensates for the Jacobian that the
                // caller will add back when evaluating on z-scale
                // (log_jacobian(z) = z for Log transform), recovering
                // the correct z-scale density log N(z; μ, σ).
                if natural <= 0.0 { return f64::NEG_INFINITY; }
                let z_score = (transformed - mean) / sd;
                -transformed - HALF_LN_2PI - sd.ln() - 0.5 * z_score * z_score
            }
            Prior::HalfNormal { sigma } => {
                if natural < 0.0 { return f64::NEG_INFINITY; }
                let z = natural / sigma;
                // log(2/(sigma * sqrt(2π))) - 0.5 z²
                // = ln(2) - ln(sigma) - 0.5 * ln(2π) - 0.5 z²
                std::f64::consts::LN_2 - sigma.ln() - HALF_LN_2PI - 0.5 * z * z
            }
            Prior::Beta { alpha, beta } => {
                if natural <= 0.0 || natural >= 1.0 { return f64::NEG_INFINITY; }
                (alpha - 1.0) * natural.ln() + (beta - 1.0) * (1.0 - natural).ln()
                    - (lgamma(*alpha) + lgamma(*beta) - lgamma(alpha + beta))
            }
            Prior::Gamma { shape, rate } => {
                if natural <= 0.0 { return f64::NEG_INFINITY; }
                // log Gamma(x; k, r) = k*ln(r) + (k-1)*ln(x) - r*x - lgamma(k)
                shape * rate.ln() + (shape - 1.0) * natural.ln() - rate * natural - lgamma(*shape)
            }
            Prior::Exponential { rate } => {
                if natural < 0.0 { return f64::NEG_INFINITY; }
                rate.ln() - rate * natural
            }
            Prior::LogUniform { lower, upper } => {
                // Natural-scale density 1/(θ·(ln U − ln L)) on [L, U]:
                //   log p(θ) = −ln θ − ln(ln U − ln L).
                // With z = ln θ (Log transform) the caller adds the Jacobian
                // +z, giving the flat z-scale density −ln(ln U − ln L).
                if natural < *lower || natural > *upper { return f64::NEG_INFINITY; }
                -natural.ln() - (upper.ln() - lower.ln()).ln()
            }
            Prior::TruncatedNormal { mean, sd, lower, upper } => {
                // Truncated normal on the natural scale:
                //   log p(θ) = log N(θ; μ, σ) − log Z,   θ ∈ [L, U]
                //   Z = Φ((U−μ)/σ) − Φ((L−μ)/σ)   (constant in θ).
                if natural < *lower || natural > *upper { return f64::NEG_INFINITY; }
                let z = (natural - mean) / sd;
                let log_z = (normal_cdf((upper - mean) / sd)
                    - normal_cdf((lower - mean) / sd)).ln();
                -HALF_LN_2PI - sd.ln() - 0.5 * z * z - log_z
            }
            // Hierarchical priors require a `ParamEnv` to resolve their
            // expression-valued args. Callers without an env (older code
            // paths: adaptive-proposal bookkeeping, trace display) see
            // an Hierarchical variant as f64::NEG_INFINITY here — safe
            // fallback. New callers should use `log_density_env`.
            Prior::Hierarchical(_) => f64::NEG_INFINITY,
        }
    }

    /// Env-aware log-density. Semantically identical to `log_density`
    /// for non-hierarchical variants (env is ignored). For
    /// `Prior::Hierarchical`, resolves arg expressions against `env`.
    /// Wave 2 / malaria #3 Gate 3.
    pub fn log_density_env<E: crate::inference::hierarchical::ParamEnv>(
        &self, natural: f64, transformed: f64, env: &E,
    ) -> f64 {
        match self {
            Prior::Hierarchical(hp) =>
                crate::inference::hierarchical::hierarchical_log_density(
                    hp, natural, transformed, env, Scale::Natural),
            _ => self.log_density(natural, transformed),
        }
    }

    /// Convert from the IR's `PriorDist` representation (serialized from DSL
    /// `~` syntax or fit.toml structured form).
    ///
    /// The IR uses `LogNormal` as a distribution name; in our runtime it maps
    /// to `TransformedNormal` (Normal on the log-transformed scale), because
    /// parameters with log_normal priors use log transforms for inference.
    pub fn from_ir(pd: &ir::parameter::PriorDist) -> Self {
        use ir::parameter::PriorDist;
        match pd {
            PriorDist::Uniform(u) => Prior::Uniform { lower: u.lower, upper: u.upper },
            PriorDist::Normal(p) => Prior::Normal { mean: p.mean, sd: p.sd },
            PriorDist::LogNormal(p) => Prior::TransformedNormal { mean: p.mu, sd: p.sigma },
            PriorDist::HalfNormal(p) => Prior::HalfNormal { sigma: p.sigma },
            PriorDist::Beta(p) => Prior::Beta { alpha: p.alpha, beta: p.beta },
            PriorDist::Gamma(p) => Prior::Gamma { shape: p.shape, rate: p.rate },
            PriorDist::Exponential(p) => Prior::Exponential { rate: p.rate },
            PriorDist::LogUniform(p) => Prior::LogUniform { lower: p.lower, upper: p.upper },
            PriorDist::TruncatedNormal(p) => Prior::TruncatedNormal {
                mean: p.mean, sd: p.sd, lower: p.lower, upper: p.upper },
            // Fixed is not really a prior — it means the param has a known value.
            // In inference contexts this parameter should be in [fixed], not
            // [estimate]. Treat as Flat if we see it in a prior slot.
            PriorDist::Fixed(_) => Prior::Flat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn flat_is_zero() {
        assert_eq!(Prior::Flat.log_density(0.5, 0.5), 0.0);
        assert_eq!(Prior::Flat.log_density(100.0, -100.0), 0.0);
    }

    #[test]
    fn uniform_within_bounds() {
        let p = Prior::Uniform { lower: 0.0, upper: 1.0 };
        // Inside: log(1/1) = 0
        assert!(approx_eq(p.log_density(0.5, 0.5), 0.0, 1e-10));
        // Outside
        assert_eq!(p.log_density(-0.1, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(1.1, 0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn transformed_normal_natural_scale_integrates_to_one() {
        // IC3 regression: TransformedNormal returns the natural-scale
        // log-density of a log-normal. Numerically integrate on the
        // natural axis and check the density integrates to ~1.
        // log_normal(mu=0, sigma=1) has density
        //   p(θ) = 1/(θ·√(2π)) · exp(−(log θ)² / 2) for θ > 0.
        let p = Prior::TransformedNormal { mean: 0.0, sd: 1.0 };
        let dx = 0.001;
        let total: f64 = (1..50_000).map(|i| {
            let theta = i as f64 * dx;
            let z = theta.ln();
            p.log_density(theta, z).exp() * dx
        }).sum();
        assert!((total - 1.0).abs() < 1e-3,
            "log-normal density should integrate to ~1, got {}", total);
    }

    #[test]
    fn transformed_normal_plus_jacobian_equals_z_scale_normal() {
        // IC3 regression: for transformed-space MH the density is
        //   log p̃(z) = log p(θ(z)) + log|dθ/dz|
        // For a log-normal(μ, σ) with Log transform:
        //   log|dθ/dz| = z, so log p̃(z) = log N(z; μ, σ).
        // Verify this identity holds with the fixed log_density.
        let p = Prior::TransformedNormal { mean: 1.0, sd: 0.5 };
        for &z in &[-1.0_f64, 0.0, 0.5, 1.0, 2.0] {
            let theta = z.exp();
            let log_natural = p.log_density(theta, z);
            let log_z_scale_expected = {
                let z_score = (z - 1.0) / 0.5;
                -HALF_LN_2PI - 0.5_f64.ln() - 0.5 * z_score * z_score
            };
            let caller_added_jacobian = z; // log_jacobian for Log transform
            let log_z_scale_actual = log_natural + caller_added_jacobian;
            assert!((log_z_scale_actual - log_z_scale_expected).abs() < 1e-10,
                "at z={}: natural+jacobian={} != z-scale normal={}",
                z, log_z_scale_actual, log_z_scale_expected);
        }
    }

    #[test]
    fn normal_peak_at_mean() {
        let p = Prior::Normal { mean: 1.0, sd: 0.5 };
        let at_mean = p.log_density(1.0, 0.0);
        let off = p.log_density(1.5, 0.0);
        assert!(at_mean > off);
    }

    #[test]
    fn normal_log_density_is_normalized() {
        // N(0, 1) at x=0: -0.5 ln(2π) ≈ -0.9189385
        // N(0, 1) at x=1: -0.5 ln(2π) - 0.5 ≈ -1.4189385
        let p = Prior::Normal { mean: 0.0, sd: 1.0 };
        assert!(approx_eq(p.log_density(0.0, 0.0), -HALF_LN_2PI, 1e-10));
        assert!(approx_eq(p.log_density(1.0, 0.0), -HALF_LN_2PI - 0.5, 1e-10));
        // Unit integral check via trapezoidal quadrature on the density.
        let dx = 0.001;
        let total: f64 = (-5000..=5000).map(|i| {
            let x = i as f64 * dx;
            p.log_density(x, 0.0).exp() * dx
        }).sum();
        assert!((total - 1.0).abs() < 1e-4, "density should integrate to ~1, got {}", total);
    }

    #[test]
    fn half_normal_nonnegative() {
        let p = Prior::HalfNormal { sigma: 1.0 };
        assert_eq!(p.log_density(-0.5, 0.0), f64::NEG_INFINITY);
        assert!(p.log_density(0.5, 0.0).is_finite());
    }

    #[test]
    fn gamma_positive() {
        let p = Prior::Gamma { shape: 2.0, rate: 1.0 };
        assert_eq!(p.log_density(-0.1, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(0.0, 0.0), f64::NEG_INFINITY);
        assert!(p.log_density(1.0, 0.0).is_finite());
        // Gamma(2, 1) mode at (k-1)/r = 1. Density higher at 1 than far from it.
        assert!(p.log_density(1.0, 0.0) > p.log_density(5.0, 0.0));
    }

    #[test]
    fn exponential_decays() {
        let p = Prior::Exponential { rate: 1.0 };
        assert!(p.log_density(0.0, 0.0) > p.log_density(1.0, 0.0));
        assert!(p.log_density(1.0, 0.0) > p.log_density(10.0, 0.0));
    }

    #[test]
    fn beta_on_unit_interval() {
        let p = Prior::Beta { alpha: 2.0, beta: 2.0 };
        assert_eq!(p.log_density(0.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(1.0, 0.0), f64::NEG_INFINITY);
        // Symmetric Beta(2,2) peak at 0.5
        assert!(p.log_density(0.5, 0.0) > p.log_density(0.3, 0.0));
    }

    #[test]
    fn log_uniform_support_and_density() {
        let p = Prior::LogUniform { lower: 1e-3, upper: 1e0 };
        // Outside support → -inf.
        assert_eq!(p.log_density(5e-4, (5e-4_f64).ln()), f64::NEG_INFINITY);
        assert_eq!(p.log_density(2.0, 2.0_f64.ln()), f64::NEG_INFINITY);
        // Natural-scale density integrates to ~1 over [lower, upper].
        let (lo, hi) = (1e-3_f64, 1e0_f64);
        let n = 200_000;
        let dx = (hi - lo) / n as f64;
        let total: f64 = (0..n).map(|i| {
            let theta = lo + (i as f64 + 0.5) * dx;
            p.log_density(theta, theta.ln()).exp() * dx
        }).sum();
        assert!((total - 1.0).abs() < 1e-3, "log_uniform density should integrate to ~1, got {}", total);
    }

    #[test]
    fn log_uniform_flat_on_log_scale() {
        // On the z = ln θ scale the (Jacobian-adjusted) density is constant.
        // log p(θ) + z must be equal for any two interior points.
        let p = Prior::LogUniform { lower: 1e-4, upper: 1e2 };
        let pts = [1e-3_f64, 1e-1, 1.0, 50.0];
        let zdens: Vec<f64> = pts.iter().map(|&t| p.log_density(t, t.ln()) + t.ln()).collect();
        for w in zdens.windows(2) {
            assert!((w[0] - w[1]).abs() < 1e-12, "log_uniform not flat in z: {:?}", zdens);
        }
    }

    #[test]
    fn truncated_normal_support_and_density() {
        let p = Prior::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 };
        // Outside [lower, upper] → -inf.
        assert_eq!(p.log_density(0.2, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(1.1, 0.0), f64::NEG_INFINITY);
        // Density integrates to ~1 over [lower, upper].
        let (lo, hi) = (0.3_f64, 1.0_f64);
        let n = 200_000;
        let dx = (hi - lo) / n as f64;
        let total: f64 = (0..n).map(|i| {
            let x = lo + (i as f64 + 0.5) * dx;
            p.log_density(x, 0.0).exp() * dx
        }).sum();
        assert!((total - 1.0).abs() < 1e-3, "truncated_normal density should integrate to ~1, got {}", total);
    }

    #[test]
    fn truncated_normal_renormalizes_above_untruncated() {
        // Truncation increases the in-support density relative to a plain
        // Normal (mass outside [lo,hi] is redistributed inward).
        let tn = Prior::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 };
        let n  = Prior::Normal { mean: 0.7, sd: 0.2 };
        assert!(tn.log_density(0.7, 0.0) > n.log_density(0.7, 0.0));
    }

    #[test]
    fn from_ir_roundtrip() {
        use ir::parameter::*;
        let ir_prior = PriorDist::LogNormal(LogNormalPrior { mu: -1.0, sigma: 0.5 });
        match Prior::from_ir(&ir_prior) {
            Prior::TransformedNormal { mean, sd } => {
                assert_eq!(mean, -1.0);
                assert_eq!(sd, 0.5);
            }
            _ => panic!("expected TransformedNormal"),
        }

        let ir_beta = PriorDist::Beta(BetaPrior { alpha: 2.0, beta: 5.0 });
        match Prior::from_ir(&ir_beta) {
            Prior::Beta { alpha, beta } => {
                assert_eq!(alpha, 2.0);
                assert_eq!(beta, 5.0);
            }
            _ => panic!("expected Beta"),
        }

        let ir_fixed = PriorDist::Fixed(0.5);
        assert!(matches!(Prior::from_ir(&ir_fixed), Prior::Flat));

        let ir_lu = PriorDist::LogUniform(LogUniformPrior { lower: 1e-5, upper: 1e-2 });
        match Prior::from_ir(&ir_lu) {
            Prior::LogUniform { lower, upper } => {
                assert_eq!(lower, 1e-5);
                assert_eq!(upper, 1e-2);
            }
            _ => panic!("expected LogUniform"),
        }

        let ir_tn = PriorDist::TruncatedNormal(
            TruncatedNormalPrior { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 });
        match Prior::from_ir(&ir_tn) {
            Prior::TruncatedNormal { mean, sd, lower, upper } => {
                assert_eq!(mean, 0.7);
                assert_eq!(sd, 0.2);
                assert_eq!(lower, 0.3);
                assert_eq!(upper, 1.0);
            }
            _ => panic!("expected TruncatedNormal"),
        }
    }
}
