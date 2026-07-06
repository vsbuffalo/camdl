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
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
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
         1.38357751867269e+02, -3.066479806614716e+01,  2.506628277459239e+00,
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

/// Regularized lower incomplete gamma `P(a, x) = γ(a,x)/Γ(a)`, for `a > 0`,
/// `x ≥ 0`. Power series for `x < a+1`, Lentz continued fraction (for the upper
/// tail `Q = 1−P`) otherwise — the standard split (DiDonato & Morris 1986,
/// ACM TOMS Alg. 654; Cephes; Numerical Recipes §6.2). Returns a value in
/// `[0, 1]`; `NaN` for `a ≤ 0` or `x < 0`.
pub fn gammp(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let lg_a = lgamma(a);
    if x < a + 1.0 {
        // Series: P(a,x) = e^{a ln x − x − lnΓ(a)} · Σ_{n≥0} x^n / (a(a+1)…(a+n)).
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        (sum.ln() + a * x.ln() - x - lg_a).exp().min(1.0)
    } else {
        // Lentz continued fraction for Q(a,x); P = 1 − Q.
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (a * x.ln() - x - lg_a + h.ln()).exp()
    }
}

/// Density of `Gamma(a, 1)` at `x`: `x^{a−1} e^{−x} / Γ(a)` — this is `dP/dx` for
/// [`gammp`], the Newton derivative used by [`gammp_inv`].
fn gamma_pdf_unit(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - lgamma(a)).exp()
}

/// Inverse regularized lower incomplete gamma: the `p`-quantile of `Gamma(a, 1)`
/// — returns `x` with `P(a, x) = p`, for `a > 0`, `p ∈ (0, 1)`.
///
/// A closed-form initial guess (Wilson–Hilferty for `a > 1`, the small-`x` series
/// inversion `x ≈ (p·Γ(a+1))^{1/a}` for `a ≤ 1`) refined by Newton's method on
/// [`gammp`], bracketed by bisection for global safety. Converges to full-`f64`
/// precision for all shapes — including `a < 1`, where the bare Wilson–Hilferty
/// guess (the old correlated-PF overdispersion draw) was biased and clamped a
/// growing fraction of draws to zero (gh#372). This is the exact Gamma quantile
/// that inverse-CDF sampling needs.
pub fn gammp_inv(a: f64, p: f64) -> f64 {
    if a <= 0.0 {
        return f64::NAN;
    }
    let p = p.clamp(f64::MIN_POSITIVE, 1.0 - 1e-16);
    // Closed-form initial guess.
    let mut x = if a > 1.0 {
        let t = normal_quantile(p);
        let c = 1.0 / (9.0 * a);
        let g = 1.0 - c + t * c.sqrt();
        a * g * g * g
    } else {
        // P(a,x) ≈ x^a / Γ(a+1) as x→0  ⇒  x ≈ (p·Γ(a+1))^(1/a).
        ((p.ln() + lgamma(a + 1.0)) / a).exp()
    };
    if !x.is_finite() || x <= 0.0 {
        x = 1e-8;
    }
    // Safeguarded Newton: maintain a [lo, hi] bracket; bisect when a Newton step
    // would leave it or the density underflows.
    let mut lo = 0.0f64;
    let mut hi = f64::INFINITY;
    for _ in 0..100 {
        let err = gammp(a, x) - p;
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let pdf = gamma_pdf_unit(a, x);
        let mut xnew = if pdf > 0.0 { x - err / pdf } else { f64::NAN };
        if !(xnew.is_finite() && xnew > lo && xnew < hi) {
            xnew = if hi.is_finite() { 0.5 * (lo + hi) } else { (2.0 * x).max(1e-8) };
        }
        if (xnew - x).abs() <= x.abs() * 1e-13 + 1e-300 {
            return xnew;
        }
        x = xnew;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    fn rel_close(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want.abs().max(1e-30)
    }

    #[test]
    fn gammp_matches_scipy() {
        // scipy.special.gammainc oracles.
        let cases = [
            (0.1, 0.05, 0.775538635451031),
            (0.1, 0.5, 0.941402445890133),
            (0.355, 0.355, 0.711868215208794),
            (0.355, 2.0, 0.972243225728695),
            (0.5, 0.5, 0.682689492137086),
            (1.0, 1.0, 0.632120558828558),
            (2.0, 2.0, 0.593994150290162),
            (5.0, 2.0, 0.052653017343711),
            (5.0, 5.0, 0.559506714934788),
        ];
        for (a, x, want) in cases {
            let got = gammp(a, x);
            assert!(approx(got, want, 1e-12), "gammp({a},{x})={got}, want {want}");
        }
        assert_eq!(gammp(1.0, 0.0), 0.0);
    }

    #[test]
    fn gammp_inv_matches_scipy() {
        // scipy.special.gammaincinv oracles — the Gamma(a,1) quantile. Critically
        // covers a < 1, where Wilson-Hilferty alone was wrong and clamped to 0.
        let cases = [
            (0.355, 0.01, 0.000001677009944),
            (0.355, 0.1, 0.001101005107799),
            (0.355, 0.5, 0.111002149998881),
            (0.355, 0.9, 1.022888552316707),
            (0.355, 0.99, 2.844489838346098),
            (0.1, 0.5, 0.000593391104460),
            (0.1, 0.99, 1.588477817929504),
            (0.5, 0.5, 0.227468211559786),
            (1.0, 0.5, 0.693147180559946),
            (2.0, 0.5, 1.678346990016661),
            (5.0, 0.99, 11.604625579477178),
        ];
        for (a, p, want) in cases {
            let got = gammp_inv(a, p);
            assert!(rel_close(got, want, 1e-6), "gammp_inv({a},{p})={got}, want {want}");
        }
    }

    #[test]
    fn gammp_inv_round_trips_and_has_no_atom_at_zero() {
        // gammp(a, gammp_inv(a, p)) == p across shapes incl. a<1, and the quantile
        // is strictly positive — the atom-at-0 the WH clamp introduced is gone.
        for &a in &[0.1, 0.355, 0.5, 1.0, 2.0, 5.0] {
            for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
                let x = gammp_inv(a, p);
                assert!(x > 0.0, "quantile must be > 0 (no atom): a={a}, p={p} -> {x}");
                assert!(approx(gammp(a, x), p, 1e-9), "round-trip a={a} p={p}");
            }
        }
    }

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
