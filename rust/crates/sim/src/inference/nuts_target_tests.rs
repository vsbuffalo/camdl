//! The per-parameter target PGAS hands NUTS, against analytic densities.
//!
//! `run_pgas`'s NUTS closure (`log_prob_and_grad`) adds, for each estimated
//! parameter at its unconstrained coordinate z,
//!
//! ```text
//!   value(z)    = prior.log_density(&param.prior_point(z)) + param.log_jacobian(z)
//!   gradient(z) = prior_grad_z + param.jacobian_grad(z),
//!   (_, prior_grad_z) = prior.log_density_and_grad_z(&param.prior_point(z),
//!                                                    param.transform_deriv(z)),
//! ```
//!
//! to the complete-data log-likelihood. With no likelihood that sum *is* the
//! target, and for each prior family × transform the fit infrastructure pairs
//! it can be written down exactly as a density of z. These tests evaluate the
//! assembled value and gradient at fixed points and compare them with those
//! exact densities — the normalising constant included, so a dropped or
//! doubled Jacobian (a constant offset for `Logit`, a linear one for `Log`)
//! cannot hide in "equal up to a constant".
//!
//! The priors are built with `Prior::from_ir` from the IR's `PriorDist`, the
//! path a fit takes, and each transform is the one
//! `cli::fit::runner::derive_transform_with_bounds` gives the parameter's
//! kind and box: a `positive` with no box → `Log { 0, ∞ }`, a boxed
//! `positive` → `Log { lo, hi }`, a `real` with no box → `None`, a `real`
//! with a finite box → `Logit { lo, hi }`.
//!
//! The finite-difference consistency of these gradients is checked by
//! `pgas::prior_grad_tests`; this file checks the value and the gradient
//! against the analytic answer, which a finite difference of a wrong value
//! would not catch.

use super::prior::Prior;
use super::types::{EstimatedParam, Transform};
use ir::parameter::{ExponentialPrior, LogNormalPrior, NormalPrior, PriorDist, UniformPrior};

/// The assembled value and gradient have a handful of floating-point
/// operations between them and the analytic expression; 1e-12 relative is
/// ~5000 ulps, far above rounding and far below any modelling error.
const TOL: f64 = 1e-12;
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

fn param(transform: Transform, lower: f64, upper: f64) -> EstimatedParam {
    EstimatedParam {
        name: "p".into(), index: 0, initial: 0.0, rw_sd: 0.1, transform,
        lower, upper, rw_sd_auto: false, perturb_only_at_t0: false,
    }
}

/// (value, gradient) exactly as the PGAS NUTS closure assembles them for one
/// parameter with a zero likelihood.
fn assembled(prior: &Prior, p: &EstimatedParam, z: f64) -> (f64, f64) {
    let (prior_val, prior_grad_z) =
        prior.log_density_and_grad_z(&p.prior_point(z), p.transform_deriv(z));
    (prior_val + p.log_jacobian(z), prior_grad_z + p.jacobian_grad(z))
}

fn check(label: &str, prior: &Prior, p: &EstimatedParam, exact: &dyn Fn(f64) -> (f64, f64), zs: &[f64]) {
    for &z in zs {
        let (v, g) = assembled(prior, p, z);
        let (ev, eg) = exact(z);
        let close = |a: f64, b: f64| (a - b).abs() <= TOL * a.abs().max(b.abs()).max(1.0);
        assert!(close(v, ev), "{label} @ z={z}: target value {v} vs analytic {ev}");
        assert!(close(g, eg), "{label} @ z={z}: target gradient {g} vs analytic {eg}");
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `real ~ normal(−1, 0.6)`, no box → identity: z = θ, so the target is the
/// normal log density itself and there is no Jacobian.
#[test]
fn normal_on_the_identity_scale() {
    let prior = Prior::from_ir(&PriorDist::Normal(NormalPrior { mean: -1.0, sd: 0.6 }));
    let inf = f64::INFINITY;
    let p = param(Transform::None, -inf, inf);
    let exact = |z: f64| {
        let u = (z + 1.0) / 0.6;
        (-HALF_LN_2PI - 0.6_f64.ln() - 0.5 * u * u, -(z + 1.0) / 0.36)
    };
    check("normal(-1, 0.6), identity", &prior, &p, &exact, &[-3.0, -1.0, 0.0, 0.4, 2.0]);
}

/// `positive ~ exponential(λ)`, no box → log: z = ln τ has density
/// λ·eᶻ·exp(−λeᶻ), so log p(z) = ln λ + z − λeᶻ and d/dz = 1 − λeᶻ.
#[test]
fn exponential_on_the_log_scale() {
    let lambda = 5.99;
    let prior = Prior::from_ir(&PriorDist::Exponential(ExponentialPrior { rate: lambda }));
    let p = param(Transform::Log { lo: 0.0, hi: f64::INFINITY }, 0.0, f64::INFINITY);
    let exact = |z: f64| (lambda.ln() + z - lambda * z.exp(), 1.0 - lambda * z.exp());
    check("exponential(5.99), log", &prior, &p, &exact, &[-12.0, -4.0, -2.0, 0.0, 1.5]);
}

/// The same with the box [1e-6, 2.5]: identical inside, and outside it the
/// target is −∞ with a zero gradient (the box truncates the support — the
/// current behaviour `tests/nuts_validity.rs::boxed_log_wall_produces_divergences_under_nuts`
/// records the cost of under NUTS).
#[test]
fn exponential_on_a_boxed_log_scale() {
    let (lambda, lo, hi) = (5.99, 1e-6_f64, 2.5_f64);
    let prior = Prior::from_ir(&PriorDist::Exponential(ExponentialPrior { rate: lambda }));
    let p = param(Transform::Log { lo, hi }, lo, hi);
    let exact = |z: f64| (lambda.ln() + z - lambda * z.exp(), 1.0 - lambda * z.exp());
    check("exponential(5.99), log box", &prior, &p, &exact,
        &[lo.ln() + 1e-3, -4.0, -2.0, 0.0, hi.ln() - 1e-3]);
    for z in [lo.ln() - 1e-3, lo.ln() - 2.0, hi.ln() + 1e-3, hi.ln() + 2.0] {
        let (v, g) = assembled(&prior, &p, z);
        assert_eq!(v, f64::NEG_INFINITY, "outside the box at z={z}");
        assert_eq!(g, 0.0, "outside the box at z={z}");
    }
}

/// `real in [a, b] ~ uniform(a, b)` → scaled logit: θ = a + (b − a)σ(z), so
/// z has the logistic density σ(z)(1 − σ(z)) — log p = ln σ(z) + ln(1 − σ(z)),
/// d/dz = 1 − 2σ(z). The prior's −ln(b − a) and the Jacobian's +ln(b − a)
/// must cancel exactly.
#[test]
fn uniform_on_a_scaled_logit_box() {
    let (a, b) = (-3.0, 5.0);
    let prior = Prior::from_ir(&PriorDist::Uniform(UniformPrior { lower: a, upper: b }));
    let p = param(Transform::Logit { lo: a, hi: b }, a, b);
    let exact = |z: f64| {
        let s = sigmoid(z);
        // ln σ(z) = −ln(1 + e^{−z}), ln(1 − σ(z)) = −ln(1 + eᶻ): no cancellation.
        (-(-z).exp().ln_1p() - z.exp().ln_1p(), 1.0 - 2.0 * s)
    };
    check("uniform(-3, 5), scaled logit", &prior, &p, &exact, &[-6.0, -1.0, 0.0, 0.8, 6.0]);
}

/// `positive ~ log_normal(μ, σ)`, no box → log: z = ln θ ~ N(μ, σ), so the
/// target is exactly log N(z; μ, σ). The prior's natural-scale density
/// carries −z and the Jacobian +z; this is the case where the two must
/// cancel in both value and gradient.
#[test]
fn log_normal_on_the_log_scale() {
    let (mu, sigma) = (1.0, 0.5);
    let prior = Prior::from_ir(&PriorDist::LogNormal(LogNormalPrior { mu, sigma }));
    let p = param(Transform::Log { lo: 0.0, hi: f64::INFINITY }, 0.0, f64::INFINITY);
    let exact = |z: f64| {
        let u = (z - mu) / sigma;
        (-HALF_LN_2PI - sigma.ln() - 0.5 * u * u, -(z - mu) / (sigma * sigma))
    };
    check("log_normal(1, 0.5), log", &prior, &p, &exact, &[-1.0, 0.0, 1.0, 2.3]);
}
