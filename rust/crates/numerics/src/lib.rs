//! Special functions shared across the camdl workspace.
//!
//! Pure, dependency-free numeric primitives (only `libm` for `erfc`), with no
//! camdl-domain types — so both the inference engine (`sim`, which re-exports
//! these from `inference::obs_loglik` for API stability) and the standalone
//! `external-harness` validation tool can share one copy. Previously
//! `normal_quantile` was duplicated (and had drifted) between the two.

use std::f64::consts::{PI, SQRT_2};

/// Log-gamma function via Stirling's approximation with Lanczos correction.
/// Accurate to ~15 significant digits for x > 0.5.
pub fn lgamma(x: f64) -> f64 {
    // Lanczos approximation (g=7, n=9) — same coefficients as Numerical Recipes.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for i in 1..9 {
        sum += COEFFS[i] / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Asymptotic expansion for x > 6, recurrence ψ(x+1) = ψ(x) + 1/x
/// for smaller x. Accurate to ~14 digits.
pub fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    result + x.ln() - 0.5 / x
        - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0
        - x2 * (1.0/240.0 - x2 * 1.0/132.0))))
}

/// Log-density of Gamma(x; shape, scale).
///
/// log p(x | a, b) = (a-1)·ln(x) - x/b - a·ln(b) - lgamma(a)
pub fn log_gamma_density(x: f64, shape: f64, scale: f64) -> f64 {
    if x <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - lgamma(shape)
}

/// Standard normal CDF Φ(x) = 0.5·erfc(-x/√2).
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// Inverse standard normal CDF (probit), Φ⁻¹(p) for p ∈ (0, 1).
///
/// Beasley–Springer–Moro rational approximation (accurate to ~1e-9 in the
/// central region, ~1e-6 in the tails). Used for exact inverse-CDF sampling
/// of truncated distributions (`log_uniform`, `truncated_normal` prior
/// draws) so the draw lands inside the support without rejection. `libm`
/// 0.2 has no `erfinv`, hence the explicit polynomial here.
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    // Clamp away from the open-interval endpoints so callers passing 0 or 1
    // (e.g. a rounded uniform draw) get a finite, monotone result.
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
            / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0]*r + A[1])*r + A[2])*r + A[3])*r + A[4])*r + A[5]) * q
            / (((((B[0]*r + B[1])*r + B[2])*r + B[3])*r + B[4])*r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
            / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn lgamma_known_values() {
        assert!(approx(lgamma(1.0), 0.0, 1e-12));          // Γ(1) = 1
        assert!(approx(lgamma(2.0), 0.0, 1e-12));          // Γ(2) = 1
        assert!(approx(lgamma(5.0), (24.0_f64).ln(), 1e-10)); // Γ(5) = 24
        assert!(approx(lgamma(0.5), PI.sqrt().ln(), 1e-10));  // Γ(½) = √π
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ (Euler–Mascheroni); ψ(2) = 1 - γ.
        let gamma = 0.577_215_664_901_532_9;
        assert!(approx(digamma(1.0), -gamma, 1e-10));
        assert!(approx(digamma(2.0), 1.0 - gamma, 1e-10));
        // Recurrence ψ(x+1) = ψ(x) + 1/x.
        assert!(approx(digamma(6.5), digamma(5.5) + 1.0 / 5.5, 1e-12));
    }

    #[test]
    fn normal_cdf_anchors() {
        assert!(approx(normal_cdf(0.0), 0.5, 1e-12));
        assert!(approx(normal_cdf(1.959_963_984_540_054), 0.975, 1e-9));
    }

    #[test]
    fn normal_quantile_inverts_cdf() {
        assert!(approx(normal_quantile(0.975), 1.959_963_984_540_054, 1e-6));
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            assert!(approx(normal_cdf(normal_quantile(p)), p, 1e-7), "round-trip p={p}");
        }
        // Endpoint inputs are clamped to a finite, monotone result (the
        // property whose absence was the drifted external-harness copy's bug).
        assert!(normal_quantile(0.0).is_finite());
        assert!(normal_quantile(1.0).is_finite());
        assert!(normal_quantile(0.0) < normal_quantile(1.0));
    }

    #[test]
    fn log_gamma_density_matches_manual() {
        // Gamma(x=2; shape=2, scale=1): (2-1)·ln2 - 2 - 2·ln1 - lgamma(2)
        let expect = (2.0_f64).ln() - 2.0 - 0.0 - lgamma(2.0);
        assert!(approx(log_gamma_density(2.0, 2.0, 1.0), expect, 1e-12));
        assert_eq!(log_gamma_density(-1.0, 2.0, 1.0), f64::NEG_INFINITY);
    }
}
