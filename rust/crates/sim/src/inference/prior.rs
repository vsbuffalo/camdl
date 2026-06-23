//! Prior distributions for Bayesian inference.
//!
//! The density math lives in **one** place: [`Density::log_density_env`],
//! generic over how each distribution parameter is sourced (`ResolveArg`).
//! A [`Prior`] is that density plus a one-bit capability tag:
//!
//! - [`Prior::Fixed`] — parameters are constants known at config time; the
//!   density evaluates without an environment.
//! - [`Prior::Hierarchical`] — one or more parameters are expressions over
//!   *other* parameters (hyperparents), resolved against the current
//!   parameter values at each evaluation.
//!
//! The tag is the type-level marker the inference stack branches on (PGAS
//! refuses hierarchical leaves until the NUTS gradient lands — gh#175; PMMH
//! builds a [`ParamEnv`] only when a hierarchical prior is present). It
//! carries **no** density math of its own — that is the single source of
//! truth in `Density`, so a hierarchical leaf and an equivalent fixed prior
//! can never score the same value differently.
//!
//! # Parameterization conventions
//!
//! - `log_normal(mu, sigma)` (→ `TransformedNormal`): mu and sigma on the
//!   **log scale**. `log(X) ~ Normal(mu, sigma)`. Median of X is `exp(mu)`.
//! - `half_normal(sigma)`: sigma is the SD of the underlying (unfolded) normal.
//! - `gamma(shape, rate)`: rate parameterization. `E[X] = shape/rate`.
//! - `exponential(rate)`: `E[X] = 1/rate`.
//! - `beta(alpha, beta)`: shape parameters on [0, 1].
//! - `normal(mean, sd)`: natural scale.
//! - `uniform(lower, upper)`: uniform density on [lower, upper].

use crate::inference::hierarchical::{eval_prior_arg, ParamEnv};
use crate::inference::obs_loglik::{lgamma, normal_cdf};

/// 0.5 · ln(2π), used in Gaussian log-densities.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// How a distribution parameter is supplied.
///
/// `Const` is a parameter known at config time; `Expr` is an expression over
/// hyperparameters resolved against a [`ParamEnv`] at each evaluation. Used by
/// hierarchical priors; fixed priors use plain `f64` parameters.
#[derive(Clone, Debug)]
pub enum ParamArg {
    /// A constant parameter value.
    Const(f64),
    /// An expression over hyperparameters, resolved against the env.
    Expr(ir::expr::Expr),
}

/// A distribution parameter resolvable to an `f64`. The single seam that lets
/// [`Density`]'s formula be written once, generic over its parameter source:
/// `f64` for fixed priors (identity — ignores the env) and [`ParamArg`] for
/// hierarchical priors (`Expr` resolves against the env).
pub trait ResolveArg {
    /// Resolve to a concrete value against `env`. May return a non-finite
    /// value (e.g. an unbound hyperparent → `NaN`); the density formula's
    /// finiteness guards collapse those to `-∞`.
    fn resolve<E: ParamEnv>(&self, env: &E) -> f64;
}

impl ResolveArg for f64 {
    #[inline]
    fn resolve<E: ParamEnv>(&self, _env: &E) -> f64 {
        *self
    }
}

impl ResolveArg for ParamArg {
    #[inline]
    fn resolve<E: ParamEnv>(&self, env: &E) -> f64 {
        match self {
            ParamArg::Const(c) => *c,
            ParamArg::Expr(e) => eval_prior_arg(e, env),
        }
    }
}

/// The unconstrained transform a prior family requires for correct inference.
/// Used by the fit-config validator; the rule is per-family, so it is shared
/// by fixed and hierarchical priors alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformReq {
    /// Positive-support families (log_normal, half_normal, gamma, exponential,
    /// log_uniform) require a `Log` transform.
    Log,
    /// The Beta family requires a `Logit` transform.
    Logit,
    /// Compatible with any transform (flat, uniform, normal, truncated_normal).
    Any,
}

/// Distribution families and the **single source of truth** for prior
/// log-densities. Generic over the parameter source `P` (`f64` for fixed
/// priors, [`ParamArg`] for hierarchical ones).
#[derive(Clone, Debug)]
pub enum Density<P> {
    /// Flat (improper) prior — log-density 0 everywhere within transform bounds.
    Flat,
    /// Uniform(lower, upper) on the natural scale. Flat within bounds, -inf outside.
    Uniform { lower: P, upper: P },
    /// Normal(mean, sd) on the natural scale.
    Normal { mean: P, sd: P },
    /// Normal(mean, sd) on the transformed (log) scale — the "log_normal" when
    /// the parameter uses the Log transform.
    TransformedNormal { mean: P, sd: P },
    /// Half-Normal(sigma): folded normal supported on [0, inf).
    HalfNormal { sigma: P },
    /// Beta(alpha, beta) on [0, 1]. For probability parameters.
    Beta { alpha: P, beta: P },
    /// Gamma(shape, rate). Supported on (0, inf).
    Gamma { shape: P, rate: P },
    /// Exponential(rate). Supported on [0, inf).
    Exponential { rate: P },
    /// Log-Uniform(lower, upper): uniform on the log scale, supported on
    /// [lower, upper] with `lower, upper > 0`.
    LogUniform { lower: P, upper: P },
    /// Normal(mean, sd) truncated to [lower, upper]. Natural-scale density.
    TruncatedNormal { mean: P, sd: P, lower: P, upper: P },
}

impl<P> Density<P> {
    /// Distribution-family name for diagnostics and transform-compat errors.
    /// Matches the IR `HierarchicalKind::as_str` / `PriorDist` names.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Density::Flat => "flat",
            Density::Uniform { .. } => "uniform",
            Density::Normal { .. } => "normal",
            Density::TransformedNormal { .. } => "log_normal",
            Density::HalfNormal { .. } => "half_normal",
            Density::Beta { .. } => "beta",
            Density::Gamma { .. } => "gamma",
            Density::Exponential { .. } => "exponential",
            Density::LogUniform { .. } => "log_uniform",
            Density::TruncatedNormal { .. } => "truncated_normal",
        }
    }

    /// Which unconstrained transform this family requires (see [`TransformReq`]).
    /// `truncated_normal` is `Any` — its bounds-vs-support invariant is checked
    /// separately by the validator, not via the transform.
    pub fn transform_req(&self) -> TransformReq {
        match self {
            Density::TransformedNormal { .. }
            | Density::HalfNormal { .. }
            | Density::Gamma { .. }
            | Density::Exponential { .. }
            | Density::LogUniform { .. } => TransformReq::Log,
            Density::Beta { .. } => TransformReq::Logit,
            Density::Flat
            | Density::Uniform { .. }
            | Density::Normal { .. }
            | Density::TruncatedNormal { .. } => TransformReq::Any,
        }
    }
}

impl<P: ResolveArg> Density<P> {
    /// Log-density on the **natural** scale, `log p(θ)`, resolving any
    /// expression-valued parameters against `env` (constant `f64` parameters
    /// ignore it).
    ///
    /// `transformed` is the unconstrained-scale value z where θ = f(z), used
    /// by `TransformedNormal` and `LogUniform`: each returns the natural-scale
    /// density (pre-subtracting the Log-transform Jacobian z) so the caller can
    /// add `log_jacobian(z)` unconditionally to recover the z-scale density.
    ///
    /// IC3 fix (2026-04-19 inference review): `TransformedNormal` returns the
    /// natural-scale density (`log N(z; μ, σ) − z`), not the z-scale density,
    /// so callers adding `log_jacobian(z) = z` do not double-count the
    /// Jacobian. Precondition: `TransformedNormal` is only meaningful under
    /// `Transform::Log` (enforced by `validate_prior_transform_compat`).
    ///
    /// Degenerate resolved parameters (non-finite, σ ≤ 0, lower ≥ upper) and
    /// any non-finite formula result collapse to `-∞` rather than propagating
    /// `NaN` — defence-in-depth for hierarchical priors whose hyperparents may
    /// resolve to invalid values at some sampler states.
    pub fn log_density_env<E: ParamEnv>(&self, natural: f64, transformed: f64, env: &E) -> f64 {
        // Collapse a NaN / non-finite formula result to -inf (NaN isolation).
        let finite = |v: f64| if v.is_finite() { v } else { f64::NEG_INFINITY };
        match self {
            Density::Flat => 0.0,
            Density::Uniform { lower, upper } => {
                let (lo, hi) = (lower.resolve(env), upper.resolve(env));
                if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                    return f64::NEG_INFINITY;
                }
                if natural < lo || natural > hi {
                    f64::NEG_INFINITY
                } else {
                    -((hi - lo).ln())
                }
            }
            Density::Normal { mean, sd } => {
                let (mu, s) = (mean.resolve(env), sd.resolve(env));
                if !mu.is_finite() || !s.is_finite() || s <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                // Full normal log-density: -0.5 ln(2π) - ln(σ) - 0.5 z²
                let z = (natural - mu) / s;
                finite(-HALF_LN_2PI - s.ln() - 0.5 * z * z)
            }
            Density::TransformedNormal { mean, sd } => {
                // Log-normal on natural scale:
                //   log p(θ) = log N(log θ; μ, σ) − log θ
                // With z = log θ (Log transform) this is log N(z; μ, σ) − z;
                // the −z compensates for the Jacobian the caller adds back on
                // the z-scale (log_jacobian(z) = z for the Log transform).
                let (mu, s) = (mean.resolve(env), sd.resolve(env));
                if !mu.is_finite() || !s.is_finite() || s <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if natural <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let z_score = (transformed - mu) / s;
                finite(-transformed - HALF_LN_2PI - s.ln() - 0.5 * z_score * z_score)
            }
            Density::HalfNormal { sigma } => {
                let sg = sigma.resolve(env);
                if !sg.is_finite() || sg <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if natural < 0.0 {
                    return f64::NEG_INFINITY;
                }
                // log(2/(σ√(2π))) − 0.5 z² = ln 2 − ln σ − 0.5 ln(2π) − 0.5 z²
                let z = natural / sg;
                finite(std::f64::consts::LN_2 - sg.ln() - HALF_LN_2PI - 0.5 * z * z)
            }
            Density::Beta { alpha, beta } => {
                let (a, b) = (alpha.resolve(env), beta.resolve(env));
                if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if natural <= 0.0 || natural >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                finite(
                    (a - 1.0) * natural.ln() + (b - 1.0) * (1.0 - natural).ln()
                        - (lgamma(a) + lgamma(b) - lgamma(a + b)),
                )
            }
            Density::Gamma { shape, rate } => {
                let (k, r) = (shape.resolve(env), rate.resolve(env));
                if !k.is_finite() || !r.is_finite() || k <= 0.0 || r <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if natural <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                // log Gamma(x; k, r) = k·ln r + (k−1)·ln x − r·x − lgamma(k)
                finite(k * r.ln() + (k - 1.0) * natural.ln() - r * natural - lgamma(k))
            }
            Density::Exponential { rate } => {
                let r = rate.resolve(env);
                if !r.is_finite() || r <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                if natural < 0.0 {
                    return f64::NEG_INFINITY;
                }
                finite(r.ln() - r * natural)
            }
            Density::LogUniform { lower, upper } => {
                // Natural-scale density 1/(θ·(ln U − ln L)) on [L, U]:
                //   log p(θ) = −ln θ − ln(ln U − ln L).
                // With z = ln θ (Log transform) the caller adds the Jacobian
                // +z, giving the flat z-scale density −ln(ln U − ln L).
                let (lo, hi) = (lower.resolve(env), upper.resolve(env));
                if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= lo {
                    return f64::NEG_INFINITY;
                }
                if natural < lo || natural > hi {
                    return f64::NEG_INFINITY;
                }
                finite(-natural.ln() - (hi.ln() - lo.ln()).ln())
            }
            Density::TruncatedNormal { mean, sd, lower, upper } => {
                // Truncated normal on the natural scale:
                //   log p(θ) = log N(θ; μ, σ) − log Z,   θ ∈ [L, U]
                //   Z = Φ((U−μ)/σ) − Φ((L−μ)/σ)   (constant in θ).
                let (mu, s, lo, hi) =
                    (mean.resolve(env), sd.resolve(env), lower.resolve(env), upper.resolve(env));
                if !mu.is_finite() || !s.is_finite() || !lo.is_finite() || !hi.is_finite()
                    || s <= 0.0 || lo >= hi
                {
                    return f64::NEG_INFINITY;
                }
                if natural < lo || natural > hi {
                    return f64::NEG_INFINITY;
                }
                let z = (natural - mu) / s;
                let log_z = (normal_cdf((hi - mu) / s) - normal_cdf((lo - mu) / s)).ln();
                finite(-HALF_LN_2PI - s.ln() - 0.5 * z * z - log_z)
            }
        }
    }
}

/// Prior distribution for one estimated parameter: a [`Density`] plus a
/// one-bit capability tag. See the module docs.
#[derive(Clone, Debug)]
pub enum Prior {
    /// Constant-parameter prior; evaluates without an environment.
    Fixed(Density<f64>),
    /// Hierarchical prior; parameters are expressions over hyperparents,
    /// resolved against a [`ParamEnv`]. The inference stack branches on this
    /// variant (NUTS-gradient eligibility, env construction, chain-init).
    Hierarchical(Density<ParamArg>),
}

impl Prior {
    /// Log-density on the natural scale **without** an environment. Fixed
    /// priors evaluate directly; a hierarchical prior's hyperparent references
    /// resolve to `NaN` → `-∞` (the documented env-free fallback). Callers with
    /// hyperparameter values use [`Prior::log_density_env`].
    pub fn log_density(&self, natural: f64, transformed: f64) -> f64 {
        match self {
            Prior::Fixed(d) => d.log_density_env(natural, transformed, &()),
            Prior::Hierarchical(d) => d.log_density_env(natural, transformed, &()),
        }
    }

    /// Env-aware log-density on the natural scale. For fixed priors the env is
    /// ignored; for hierarchical priors it resolves the hyperparent references.
    pub fn log_density_env<E: ParamEnv>(&self, natural: f64, transformed: f64, env: &E) -> f64 {
        match self {
            Prior::Fixed(d) => d.log_density_env(natural, transformed, env),
            Prior::Hierarchical(d) => d.log_density_env(natural, transformed, env),
        }
    }

    /// Distribution-family name (for diagnostics / transform-compat errors).
    pub fn kind_str(&self) -> &'static str {
        match self {
            Prior::Fixed(d) => d.kind_str(),
            Prior::Hierarchical(d) => d.kind_str(),
        }
    }

    /// True if this prior depends on other parameters (a hierarchical leaf).
    pub fn is_hierarchical(&self) -> bool {
        matches!(self, Prior::Hierarchical(_))
    }

    /// Which unconstrained transform this prior's family requires.
    pub fn transform_req(&self) -> TransformReq {
        match self {
            Prior::Fixed(d) => d.transform_req(),
            Prior::Hierarchical(d) => d.transform_req(),
        }
    }

    /// Convert from the IR's `PriorDist` (fixed, constant-parameter priors).
    ///
    /// The IR uses `LogNormal` as a distribution name; it maps to
    /// `TransformedNormal` (Normal on the log-transformed scale) because
    /// log_normal parameters use the Log transform for inference. `Fixed`
    /// (a known value) is not a prior — treated as `Flat` if seen here.
    pub fn from_ir(pd: &ir::parameter::PriorDist) -> Self {
        use ir::parameter::PriorDist;
        Prior::Fixed(match pd {
            PriorDist::Uniform(u) => Density::Uniform { lower: u.lower, upper: u.upper },
            PriorDist::Normal(p) => Density::Normal { mean: p.mean, sd: p.sd },
            PriorDist::LogNormal(p) => Density::TransformedNormal { mean: p.mu, sd: p.sigma },
            PriorDist::HalfNormal(p) => Density::HalfNormal { sigma: p.sigma },
            PriorDist::Beta(p) => Density::Beta { alpha: p.alpha, beta: p.beta },
            PriorDist::Gamma(p) => Density::Gamma { shape: p.shape, rate: p.rate },
            PriorDist::Exponential(p) => Density::Exponential { rate: p.rate },
            PriorDist::LogUniform(p) => Density::LogUniform { lower: p.lower, upper: p.upper },
            PriorDist::TruncatedNormal(p) => Density::TruncatedNormal {
                mean: p.mean,
                sd: p.sd,
                lower: p.lower,
                upper: p.upper,
            },
            PriorDist::Fixed(_) => Density::Flat,
        })
    }

    /// Convert from the IR's `HierarchicalPrior` (expression-valued parameters
    /// over hyperparents). A missing required arg becomes a `NaN`-producing
    /// constant so the density guards collapse to `-∞` (defence-in-depth — the
    /// compiler validates arg presence; this is the backstop).
    pub fn from_hierarchical_ir(hp: &ir::parameter::HierarchicalPrior) -> Self {
        use ir::parameter::HierarchicalKind as K;
        let arg = |k: &str| -> ParamArg {
            match hp.args.get(k) {
                Some(e) => ParamArg::Expr(e.clone()),
                None => ParamArg::Const(f64::NAN),
            }
        };
        Prior::Hierarchical(match hp.kind {
            K::Uniform => Density::Uniform { lower: arg("lower"), upper: arg("upper") },
            K::Normal => Density::Normal { mean: arg("mu"), sd: arg("sigma") },
            K::LogNormal => Density::TransformedNormal { mean: arg("mu"), sd: arg("sigma") },
            K::HalfNormal => Density::HalfNormal { sigma: arg("sigma") },
            K::Beta => Density::Beta { alpha: arg("alpha"), beta: arg("beta") },
            K::Gamma => Density::Gamma { shape: arg("shape"), rate: arg("rate") },
            K::Exponential => Density::Exponential { rate: arg("rate") },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // Convenience constructors keep the fixed-prior tests readable.
    fn flat() -> Prior {
        Prior::Fixed(Density::Flat)
    }
    fn normal(mean: f64, sd: f64) -> Prior {
        Prior::Fixed(Density::Normal { mean, sd })
    }

    #[test]
    fn flat_is_zero() {
        assert_eq!(flat().log_density(0.5, 0.5), 0.0);
        assert_eq!(flat().log_density(100.0, -100.0), 0.0);
    }

    #[test]
    fn uniform_within_bounds() {
        let p = Prior::Fixed(Density::Uniform { lower: 0.0, upper: 1.0 });
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
        let p = Prior::Fixed(Density::TransformedNormal { mean: 0.0, sd: 1.0 });
        let dx = 0.001;
        let total: f64 = (1..50_000)
            .map(|i| {
                let theta = i as f64 * dx;
                let z = theta.ln();
                p.log_density(theta, z).exp() * dx
            })
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "log-normal density should integrate to ~1, got {}",
            total
        );
    }

    #[test]
    fn transformed_normal_plus_jacobian_equals_z_scale_normal() {
        // IC3 regression: for transformed-space MH the density is
        //   log p̃(z) = log p(θ(z)) + log|dθ/dz|
        // For a log-normal(μ, σ) with Log transform log|dθ/dz| = z, so
        //   log p̃(z) = log N(z; μ, σ). Verify the identity holds.
        let p = Prior::Fixed(Density::TransformedNormal { mean: 1.0, sd: 0.5 });
        for &z in &[-1.0_f64, 0.0, 0.5, 1.0, 2.0] {
            let theta = z.exp();
            let log_natural = p.log_density(theta, z);
            let log_z_scale_expected = {
                let z_score = (z - 1.0) / 0.5;
                -HALF_LN_2PI - 0.5_f64.ln() - 0.5 * z_score * z_score
            };
            let caller_added_jacobian = z; // log_jacobian for Log transform
            let log_z_scale_actual = log_natural + caller_added_jacobian;
            assert!(
                (log_z_scale_actual - log_z_scale_expected).abs() < 1e-10,
                "at z={}: natural+jacobian={} != z-scale normal={}",
                z,
                log_z_scale_actual,
                log_z_scale_expected
            );
        }
    }

    #[test]
    fn normal_peak_at_mean() {
        let p = normal(1.0, 0.5);
        let at_mean = p.log_density(1.0, 0.0);
        let off = p.log_density(1.5, 0.0);
        assert!(at_mean > off);
    }

    #[test]
    fn normal_log_density_is_normalized() {
        // N(0, 1) at x=0: -0.5 ln(2π) ≈ -0.9189385
        // N(0, 1) at x=1: -0.5 ln(2π) - 0.5 ≈ -1.4189385
        let p = normal(0.0, 1.0);
        assert!(approx_eq(p.log_density(0.0, 0.0), -HALF_LN_2PI, 1e-10));
        assert!(approx_eq(p.log_density(1.0, 0.0), -HALF_LN_2PI - 0.5, 1e-10));
        // Unit integral check via trapezoidal quadrature on the density.
        let dx = 0.001;
        let total: f64 = (-5000..=5000)
            .map(|i| {
                let x = i as f64 * dx;
                p.log_density(x, 0.0).exp() * dx
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-4, "density should integrate to ~1, got {}", total);
    }

    #[test]
    fn half_normal_nonnegative() {
        let p = Prior::Fixed(Density::HalfNormal { sigma: 1.0 });
        assert_eq!(p.log_density(-0.5, 0.0), f64::NEG_INFINITY);
        assert!(p.log_density(0.5, 0.0).is_finite());
    }

    #[test]
    fn gamma_positive() {
        let p = Prior::Fixed(Density::Gamma { shape: 2.0, rate: 1.0 });
        assert_eq!(p.log_density(-0.1, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(0.0, 0.0), f64::NEG_INFINITY);
        assert!(p.log_density(1.0, 0.0).is_finite());
        // Gamma(2, 1) mode at (k-1)/r = 1. Density higher at 1 than far from it.
        assert!(p.log_density(1.0, 0.0) > p.log_density(5.0, 0.0));
    }

    #[test]
    fn exponential_decays() {
        let p = Prior::Fixed(Density::Exponential { rate: 1.0 });
        assert!(p.log_density(0.0, 0.0) > p.log_density(1.0, 0.0));
        assert!(p.log_density(1.0, 0.0) > p.log_density(10.0, 0.0));
    }

    #[test]
    fn beta_on_unit_interval() {
        let p = Prior::Fixed(Density::Beta { alpha: 2.0, beta: 2.0 });
        assert_eq!(p.log_density(0.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(1.0, 0.0), f64::NEG_INFINITY);
        // Symmetric Beta(2,2) peak at 0.5
        assert!(p.log_density(0.5, 0.0) > p.log_density(0.3, 0.0));
    }

    #[test]
    fn log_uniform_support_and_density() {
        let p = Prior::Fixed(Density::LogUniform { lower: 1e-3, upper: 1e0 });
        // Outside support → -inf.
        assert_eq!(p.log_density(5e-4, (5e-4_f64).ln()), f64::NEG_INFINITY);
        assert_eq!(p.log_density(2.0, 2.0_f64.ln()), f64::NEG_INFINITY);
        // Natural-scale density integrates to ~1 over [lower, upper].
        let (lo, hi) = (1e-3_f64, 1e0_f64);
        let n = 200_000;
        let dx = (hi - lo) / n as f64;
        let total: f64 = (0..n)
            .map(|i| {
                let theta = lo + (i as f64 + 0.5) * dx;
                p.log_density(theta, theta.ln()).exp() * dx
            })
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "log_uniform density should integrate to ~1, got {}",
            total
        );
    }

    #[test]
    fn log_uniform_flat_on_log_scale() {
        // On the z = ln θ scale the (Jacobian-adjusted) density is constant.
        let p = Prior::Fixed(Density::LogUniform { lower: 1e-4, upper: 1e2 });
        let pts = [1e-3_f64, 1e-1, 1.0, 50.0];
        let zdens: Vec<f64> = pts.iter().map(|&t| p.log_density(t, t.ln()) + t.ln()).collect();
        for w in zdens.windows(2) {
            assert!((w[0] - w[1]).abs() < 1e-12, "log_uniform not flat in z: {:?}", zdens);
        }
    }

    #[test]
    fn truncated_normal_support_and_density() {
        let p = Prior::Fixed(Density::TruncatedNormal {
            mean: 0.7,
            sd: 0.2,
            lower: 0.3,
            upper: 1.0,
        });
        // Outside [lower, upper] → -inf.
        assert_eq!(p.log_density(0.2, 0.0), f64::NEG_INFINITY);
        assert_eq!(p.log_density(1.1, 0.0), f64::NEG_INFINITY);
        // Density integrates to ~1 over [lower, upper].
        let (lo, hi) = (0.3_f64, 1.0_f64);
        let n = 200_000;
        let dx = (hi - lo) / n as f64;
        let total: f64 = (0..n)
            .map(|i| {
                let x = lo + (i as f64 + 0.5) * dx;
                p.log_density(x, 0.0).exp() * dx
            })
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "truncated_normal density should integrate to ~1, got {}",
            total
        );
    }

    #[test]
    fn truncated_normal_renormalizes_above_untruncated() {
        // Truncation increases the in-support density relative to a plain
        // Normal (mass outside [lo,hi] is redistributed inward).
        let tn = Prior::Fixed(Density::TruncatedNormal {
            mean: 0.7,
            sd: 0.2,
            lower: 0.3,
            upper: 1.0,
        });
        let n = normal(0.7, 0.2);
        assert!(tn.log_density(0.7, 0.0) > n.log_density(0.7, 0.0));
    }

    #[test]
    fn from_ir_roundtrip() {
        use ir::parameter::*;
        let ir_prior = PriorDist::LogNormal(LogNormalPrior { mu: -1.0, sigma: 0.5 });
        match Prior::from_ir(&ir_prior) {
            Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
                assert_eq!(mean, -1.0);
                assert_eq!(sd, 0.5);
            }
            _ => panic!("expected Fixed(TransformedNormal)"),
        }

        let ir_beta = PriorDist::Beta(BetaPrior { alpha: 2.0, beta: 5.0 });
        match Prior::from_ir(&ir_beta) {
            Prior::Fixed(Density::Beta { alpha, beta }) => {
                assert_eq!(alpha, 2.0);
                assert_eq!(beta, 5.0);
            }
            _ => panic!("expected Fixed(Beta)"),
        }

        let ir_fixed = PriorDist::Fixed(0.5);
        assert!(matches!(Prior::from_ir(&ir_fixed), Prior::Fixed(Density::Flat)));

        let ir_lu = PriorDist::LogUniform(LogUniformPrior { lower: 1e-5, upper: 1e-2 });
        match Prior::from_ir(&ir_lu) {
            Prior::Fixed(Density::LogUniform { lower, upper }) => {
                assert_eq!(lower, 1e-5);
                assert_eq!(upper, 1e-2);
            }
            _ => panic!("expected Fixed(LogUniform)"),
        }

        let ir_tn = PriorDist::TruncatedNormal(TruncatedNormalPrior {
            mean: 0.7,
            sd: 0.2,
            lower: 0.3,
            upper: 1.0,
        });
        match Prior::from_ir(&ir_tn) {
            Prior::Fixed(Density::TruncatedNormal { mean, sd, lower, upper }) => {
                assert_eq!(mean, 0.7);
                assert_eq!(sd, 0.2);
                assert_eq!(lower, 0.3);
                assert_eq!(upper, 1.0);
            }
            _ => panic!("expected Fixed(TruncatedNormal)"),
        }
    }
}
