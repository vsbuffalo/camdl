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
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx). We return ln|Γ| (Γ is negative on
        // (-1,0),(-3,-2),… where sin(πx)<0), so take ln|sin(πx)| — a bare
        // `ln(π/sin(πx))` is `ln(negative) = NaN` there. Callers only pass x>0;
        // this keeps the `pub` fn correct for negative args too (gh#373).
        return PI.ln() - (PI * x).sin().abs().ln() - lgamma(1.0 - x);
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

/// Numerically stable log-sum-exp.
///
/// Im2 in the 2026-04-19 inference review batch 1: distinguish
/// +∞ vs −∞. If `max = +∞`, at least one entry is +∞ and the
/// result is also +∞ (not −∞ as the old bulk-check produced).
/// If `max = −∞`, every entry is −∞ and the result is −∞.
pub fn log_sum_exp(log_values: &[f64]) -> f64 {
    let max = log_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    if max == f64::INFINITY { return f64::INFINITY; }
    max + log_values.iter().map(|&lv| (lv - max).exp()).sum::<f64>().ln()
}

/// Sample standard deviation (N-1 denominator). Returns 0.0 for len<2.
pub fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// `logmeanexp` plus a delta-method standard error of the combined
/// estimate. Returns `(log(mean(exp(xs))), SE)`.
///
/// Derivation: with `Y = log(M⁻¹ Σ exp(X_i))` and `L_i = exp(X_i)`,
/// `Y = log(L̄)`. By the delta method (d/dL log L = 1/L),
/// `Var(Y) ≈ Var(L̄) / L̄² = Var(L_i) / (M · L̄²)`, so
/// `SE(Y) = SD(L_i) / (L̄ · √M)`. Numerically stable form (max-shift):
/// the `exp(m)` factor cancels between SD and L̄.
///
/// This is the correct SE for log-mean-exp combining of replicate
/// particle-filter log-likelihoods (matches pomp's `pfilter` output;
/// see Ionides et al. 2015 PNAS supplement and the pomp manual on
/// `logmeanexp`). The naïve `sd(xs)/√M` is only correct for the
/// *arithmetic* mean of the logs, not for log-of-arithmetic-mean of
/// likelihoods, and underestimates `Var(Y)` whenever per-replicate
/// variance is non-trivial.
///
/// Returns:
/// - `(NEG_INFINITY, NaN)` for empty input.
/// - `(value, 0.0)` for a single-element input.
/// - `(value, NaN)` if the input contains NaN.
pub fn logmeanexp_with_se(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NEG_INFINITY, f64::NAN);
    }
    if xs.len() == 1 {
        return (xs[0], 0.0);
    }
    if xs.iter().any(|x| x.is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        // All -inf, or +inf in input: SE is undefined.
        return (m, f64::NAN);
    }
    let shifted: Vec<f64> = xs.iter().map(|x| (x - m).exp()).collect();
    let mean_shifted: f64 = shifted.iter().sum::<f64>() / xs.len() as f64;
    let logmeanexp_val = m + mean_shifted.ln();
    // Delta-method SE on the log scale: SD(L_i) / (L̄ · √M). The
    // exp(m) factor cancels between SD and L̄, so we compute SD on
    // the shifted values directly.
    let sd_shifted = sample_sd(&shifted);
    let se = sd_shifted / (mean_shifted * (xs.len() as f64).sqrt());
    (logmeanexp_val, se)
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
    fn lgamma_negative_args() {
        // Reflection must handle x<0 (returns ln|Γ|); a bare ln(π/sin(πx)) is
        // NaN where sin(πx)<0, i.e. x in (-1,0),(-3,-2),… (gh#373).
        // scipy.special.gammaln oracles (covers both sin>0 and sin<0 regions).
        let cases = [
            (-0.5, 1.265512123484645),
            (-0.1, 2.368961332728789),
            (-1.5, 0.860047015376481),
            (-2.5, -0.056243716497674),
            (-2.7, -0.071407085315646),
            (-3.5, -1.309006684993042),
        ];
        for (x, want) in cases {
            let got = lgamma(x);
            assert!(got.is_finite(), "lgamma({x}) must be finite, got {got}");
            assert!(approx(got, want, 1e-10), "lgamma({x})={got}, want {want}");
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

    #[test]
    fn sample_sd_basic() {
        assert_eq!(sample_sd(&[]), 0.0);
        assert_eq!(sample_sd(&[5.0]), 0.0);
        // sd of [1, 2, 3, 4, 5] is sqrt(2.5) ≈ 1.5811 with N-1 denom.
        let got = sample_sd(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((got - 2.5f64.sqrt()).abs() < 1e-9, "got {}", got);
    }

    #[test]
    fn logmeanexp_with_se_constant_input_zero_se() {
        // Identical replicates → SE = 0 (no per-rep spread).
        let (val, se) = logmeanexp_with_se(&[-7.3, -7.3, -7.3, -7.3]);
        assert!((val - (-7.3)).abs() < 1e-12);
        assert!(se.abs() < 1e-12, "constant input should give SE = 0, got {}", se);
    }

    #[test]
    fn logmeanexp_with_se_is_numerically_stable() {
        // Naive exp of -1e6 underflows to 0; stable form must not.
        // log((e^-1e6 + e^0)/2) = log(1/2) ≈ -0.6931 (the -1e6 term
        // is negligible). Naive implementation would return -inf.
        let (val, _) = logmeanexp_with_se(&[-1e6, 0.0]);
        assert!((val - (-(2.0f64).ln())).abs() < 1e-9, "got {}", val);
    }

    #[test]
    fn logmeanexp_with_se_matches_pf_bias_example() {
        // Two PF reps ℓ = [-100, -98]. logmeanexp > arithmetic mean
        // (Jensen): log((e^-100 + e^-98)/2) = -98 + log((e^-2+1)/2).
        let xs = [-100.0, -98.0];
        let (val, _) = logmeanexp_with_se(&xs);
        let arith = (xs[0] + xs[1]) / 2.0;
        assert!(val > arith,
            "logmeanexp {} should exceed arithmetic mean {}", val, arith);
        let expected = -98.0 + ((1.0 + (-2f64).exp()) / 2.0).ln();
        assert!((val - expected).abs() < 1e-9, "got {}, expected {}", val, expected);
    }

    #[test]
    fn logmeanexp_with_se_known_two_replicate_case() {
        // Two replicates ℓ = [-100, -98]. After max-shift (m = -98):
        //   shifted = [exp(-2), 1] ≈ [0.13534, 1.0]
        //   mean_shifted = 0.56767
        //   logmeanexp = -98 + ln(0.56767) ≈ -98.5662
        // SE = sd(shifted) / (mean_shifted * sqrt(M)).
        let xs = [-100.0, -98.0];
        let (val, se) = logmeanexp_with_se(&xs);
        let expected_val = -98.0 + ((1.0 + (-2f64).exp()) / 2.0).ln();
        assert!((val - expected_val).abs() < 1e-9);

        let m = -98.0_f64;
        let shifted = [(-100.0_f64 - m).exp(), (-98.0_f64 - m).exp()];
        let mean_shifted: f64 = (shifted[0] + shifted[1]) / 2.0;
        let sd_shifted = sample_sd(&shifted);
        let expected_se = sd_shifted / (mean_shifted * 2f64.sqrt());
        assert!((se - expected_se).abs() < 1e-9, "got {}, expected {}", se, expected_se);
    }

    #[test]
    fn logmeanexp_with_se_matches_naive_in_small_variance_limit() {
        // When per-rep variance is small (σ ≪ 1 nat), the delta-method
        // SE of logmeanexp converges to sd(x)/√M to first order. This is
        // the regime users typically have post-IF2 (per-rep PF noise of
        // a fraction of a nat). The two SEs should agree closely.
        let xs = [-100.0, -100.05, -99.95, -100.0, -100.0];
        let (_, se) = logmeanexp_with_se(&xs);
        let naive = sample_sd(&xs) / (xs.len() as f64).sqrt();
        let rel_diff = (se - naive).abs() / naive;
        assert!(rel_diff < 0.05,
            "delta-method SE ({}) and naïve SE ({}) should agree within 5% \
             in small-variance regime (rel diff {})",
            se, naive, rel_diff);
    }

    #[test]
    fn logmeanexp_with_se_diverges_from_naive_when_variance_large() {
        // When per-rep variance is non-trivial (≥ 1 nat), the delta-
        // method SE diverges from sd(x)/√M. Direction is regime-
        // dependent (smaller for samples where a few replicates dominate
        // L̄; larger in symmetric-lognormal regimes). The point is that
        // the two formulas are NOT interchangeable — using sd(x)/√M as
        // the SE of logmeanexp is wrong outside the small-variance
        // limit. Just assert they disagree by > 5% so a future regression
        // would catch a fall-back to the naive formula.
        let xs = [-105.0, -100.0, -95.0];
        let (_, se) = logmeanexp_with_se(&xs);
        let naive = sample_sd(&xs) / (xs.len() as f64).sqrt();
        let rel_diff = (se - naive).abs() / naive;
        assert!(rel_diff > 0.05,
            "delta-method SE ({}) and naïve SE ({}) should disagree noticeably \
             outside the small-variance regime (rel diff {})",
            se, naive, rel_diff);
    }

    #[test]
    fn logmeanexp_with_se_edge_cases() {
        // Empty input: value is -inf, SE is NaN.
        let (v, s) = logmeanexp_with_se(&[]);
        assert!(v.is_infinite() && v.is_sign_negative());
        assert!(s.is_nan());
        // Single-element input: SE is exactly 0.
        let (v, s) = logmeanexp_with_se(&[5.0]);
        assert!((v - 5.0).abs() < 1e-12);
        assert!(s.abs() < 1e-12);
        // NaN propagates to both fields.
        let (v, s) = logmeanexp_with_se(&[f64::NAN, 0.0]);
        assert!(v.is_nan() && s.is_nan());
        // All -inf: value is -inf, SE is NaN (undefined).
        let (v, s) = logmeanexp_with_se(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert!(v.is_infinite() && v.is_sign_negative());
        assert!(s.is_nan());
    }
}
