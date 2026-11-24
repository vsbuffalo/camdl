//! Stress tests for StatefulRng samplers with extreme inputs.
//!
//! Every sampler must return a finite value for any input — no panics,
//! no NaN, no infinity (except Exp with rate=0 which returns infinity
//! by convention). IF2 random starts can push parameters to extreme
//! values; the samplers must handle this gracefully.

use sim::rng::StatefulRng;

fn rng() -> StatefulRng { StatefulRng::new(42) }

// ── Poisson ─────────────────────────────────────────────────────────────

#[test]
fn poisson_zero_lambda() {
    assert_eq!(rng().poisson(0.0), 0);
}

#[test]
fn poisson_negative_lambda() {
    assert_eq!(rng().poisson(-1.0), 0);
}

#[test]
fn poisson_huge_lambda() {
    // Should not panic; lambda is clamped to 1e15 internally
    let v = rng().poisson(1e20);
    assert!(v > 0);
}

#[test]
fn poisson_tiny_lambda() {
    // Very small lambda — most draws are 0, some are 1
    let v = rng().poisson(1e-10);
    assert!(v <= 1);
}

#[test]
fn poisson_nan_lambda() {
    // NaN should not panic
    let v = rng().poisson(f64::NAN);
    // NaN <= 0.0 is false, so it falls through to Poisson::new
    // which may or may not handle NaN — but must not panic
    let _ = v;
}

// ── Binomial ────────────────────────────────────────────────────────────

#[test]
fn binomial_zero_n() {
    assert_eq!(rng().binomial(0, 0.5), 0);
}

#[test]
fn binomial_zero_p() {
    assert_eq!(rng().binomial(100, 0.0), 0);
}

#[test]
fn binomial_negative_p() {
    assert_eq!(rng().binomial(100, -0.5), 0);
}

#[test]
fn binomial_p_one() {
    assert_eq!(rng().binomial(100, 1.0), 100);
}

#[test]
fn binomial_p_greater_than_one() {
    assert_eq!(rng().binomial(100, 1.5), 100);
}

#[test]
fn binomial_huge_n() {
    // n = 10 billion, p = 0.5 — should not panic
    let v = rng().binomial(10_000_000_000, 0.5);
    assert!(v > 0);
}

#[test]
fn binomial_tiny_p() {
    let v = rng().binomial(1000, 1e-15);
    assert!(v <= 1000);
}

// ── Gamma multiplier ────────────────────────────────────────────────────

#[test]
fn gamma_multiplier_zero_sigma() {
    assert_eq!(rng().gamma_multiplier(0.0, 1.0), 1.0);
}

#[test]
fn gamma_multiplier_negative_sigma() {
    assert_eq!(rng().gamma_multiplier(-1.0, 1.0), 1.0);
}

#[test]
fn gamma_multiplier_huge_sigma() {
    // sigma_sq = 1000, dt = 1 → shape = 0.001 < 1e-6 → returns 1.0
    let v = rng().gamma_multiplier(1000.0, 1.0);
    assert!(v.is_finite());
}

#[test]
fn gamma_multiplier_extreme_sigma() {
    // sigma_sq = 1e10 → shape = 1e-10 → degenerate guard
    let v = rng().gamma_multiplier(1e10, 1.0);
    assert_eq!(v, 1.0);
}

#[test]
fn gamma_multiplier_tiny_sigma() {
    // sigma_sq = 1e-10 → shape = 1e10 → very concentrated around 1.0
    let v = rng().gamma_multiplier(1e-10, 1.0);
    assert!(v.is_finite());
    assert!(v > 0.0);
}

// ── Neg-binomial ────────────────────────────────────────────────────────

#[test]
fn neg_binomial_zero_mean() {
    assert_eq!(rng().neg_binomial(0.0, 1.0, 1.0).0, 0);
}

#[test]
fn neg_binomial_negative_mean() {
    assert_eq!(rng().neg_binomial(-5.0, 1.0, 1.0).0, 0);
}

#[test]
fn neg_binomial_huge_sigma() {
    // sigma_sq >> dt → shape tiny → degenerate guard → Poisson fallback
    let v = rng().neg_binomial(100.0, 1e6, 1.0).0;
    assert!(v < 10000); // finite, reasonable
}

#[test]
fn neg_binomial_zero_sigma() {
    // sigma_sq = 0 → plain Poisson
    let v = rng().neg_binomial(100.0, 0.0, 1.0).0;
    assert!(v > 0);
}

// ── Exp ─────────────────────────────────────────────────────────────────

#[test]
fn exp_zero_rate() {
    assert_eq!(rng().exp(0.0), f64::INFINITY);
}

#[test]
fn exp_negative_rate() {
    assert_eq!(rng().exp(-1.0), f64::INFINITY);
}

#[test]
fn exp_huge_rate() {
    let v = rng().exp(1e15);
    assert!(v.is_finite());
    assert!(v >= 0.0);
}

// ── Normal ──────────────────────────────────────────────────────────────

#[test]
fn normal_produces_finite() {
    let v = rng().normal();
    assert!(v.is_finite());
}

// ── Uniform ─────────────────────────────────────────────────────────────

#[test]
fn uniform_in_range() {
    let v = rng().uniform();
    assert!((0.0..1.0).contains(&v));
}
