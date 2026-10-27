//! Validate periodic / Fourier / periodic-spline forcing math against
//! hand-computed and numpy-checked values. An off-by-one in the bin index would
//! shift the school calendar by one bin width.
//!
//! These exercise the pure per-kind math directly (proposal
//! `2026-06-09-const-parametric-forcing.md` §3): a forcing's scalar
//! coefficients are now live `ResolvedExpr`, and the closed-form math is a pure
//! function of the already-evaluated scalars, tested here without building a
//! `CompiledModel`.

use sim::periodic_bspline::PeriodicBasis;
use sim::propensity::{fourier_value, periodic_value};

#[test]
fn test_periodic_4_bins() {
    // Period=100, 4 bins of width 25: values [1, 2, 3, 4]
    // Bin 0: t ∈ [0, 25)  → 1
    // Bin 1: t ∈ [25, 50) → 2
    // Bin 2: t ∈ [50, 75) → 3
    // Bin 3: t ∈ [75, 100) → 4
    let values = [1.0, 2.0, 3.0, 4.0];

    let cases: Vec<(f64, f64)> = vec![
        (0.0, 1.0), (12.5, 1.0), (24.9, 1.0),
        (25.0, 2.0), (37.5, 2.0), (49.9, 2.0),
        (50.0, 3.0), (62.5, 3.0), (74.9, 3.0),
        (75.0, 4.0), (87.5, 4.0), (99.9, 4.0),
        // Wrapping: t=100 → phase=0 → bin 0
        (100.0, 1.0), (125.0, 2.0),
        // Negative time: t=-10 wraps to phase=90 → bin 3
        (-10.0, 4.0),
    ];

    for (t, expected) in &cases {
        let actual = periodic_value(100.0, &values, *t);
        assert_eq!(
            actual, *expected,
            "periodic(t={}) = {}, expected {}", t, actual, expected
        );
    }
}

#[test]
fn test_periodic_school_calendar_52_weeks() {
    // He et al. school calendar: 52 bins over 365.25 days (7.024 days/bin)
    // Known values: bin 0 (day 0) = holiday, bin 2 (day ~14) = term, bin 15 (day ~105) = holiday
    let mut values = vec![0.0; 52];
    // Term weeks (1-indexed in He et al.): 2-14, 17-28, 37-43, 45-51
    // 0-indexed: 1-13, 16-27, 36-42, 44-50
    for i in 1..=13 { values[i] = 1.0; }
    for i in 16..=27 { values[i] = 1.0; }
    for i in 36..=42 { values[i] = 1.0; }
    for i in 44..=50 { values[i] = 1.0; }

    let bin_width = 365.25 / 52.0;

    // Check each bin's midpoint
    for (i, &expected) in values.iter().enumerate() {
        let t = (i as f64 + 0.5) * bin_width; // midpoint of bin i
        let actual = periodic_value(365.25, &values, t);
        assert_eq!(
            actual, expected,
            "school(t={:.1}, bin={}) = {}, expected {}", t, i, actual, expected
        );
    }

    // Check that day 0 is holiday (bin 0)
    assert_eq!(periodic_value(365.25, &values, 0.0), 0.0, "day 0 should be holiday");

    // Check wrapping: year 2, same pattern
    let t_year2 = 365.25 + 2.0 * bin_width; // bin 2 of year 2
    assert_eq!(periodic_value(365.25, &values, t_year2), 1.0, "year 2 bin 2 should be term");
}

#[test]
fn test_periodic_boundary_no_out_of_bounds() {
    // Edge case: t exactly at period boundary
    let values = [1.0, 2.0];
    // t=10.0 → phase=0.0 → bin 0
    assert_eq!(periodic_value(10.0, &values, 10.0), 1.0);
    // t=5.0 → phase=5.0 → bin 1
    assert_eq!(periodic_value(10.0, &values, 5.0), 2.0);
    // t=4.99999 → bin 0
    assert_eq!(periodic_value(10.0, &values, 4.999), 1.0);
}

// ── gh#59: Fourier + PeriodicSpline ──────────────────────────────────────────

#[test]
fn test_fourier_pure_cos_first_harmonic() {
    // (a_1, b_1) = (1, 0), period = 1.0:
    // f(t) = cos(2π t).
    //   t = 0    → 1
    //   t = 0.25 → 0
    //   t = 0.5  → -1
    //   t = 0.75 → 0
    //   t = 1.0  → 1   (periodicity)
    let harmonics = [(1.0, 0.0)];
    let cases = [(0.0_f64, 1.0), (0.25, 0.0), (0.5, -1.0), (0.75, 0.0), (1.0, 1.0)];
    for (t, expected) in cases {
        let actual = fourier_value(1.0, &harmonics, t);
        assert!(
            (actual - expected).abs() < 1e-9,
            "fourier cos(2π·{}) = {}, expected {}", t, actual, expected
        );
    }
}

#[test]
fn test_fourier_pure_sin_second_harmonic() {
    // (a_1, b_1) = (0, 0), (a_2, b_2) = (0, 1), period = 1.0:
    // f(t) = sin(4π t).
    //   t = 0.125 → sin(π/2) = 1
    //   t = 0.25  → sin(π)   = 0
    //   t = 0.375 → sin(3π/2)= -1
    let harmonics = [(0.0, 0.0), (0.0, 1.0)];
    assert!((fourier_value(1.0, &harmonics, 0.125) - 1.0).abs() < 1e-9);
    assert!((fourier_value(1.0, &harmonics, 0.25)  - 0.0).abs() < 1e-9);
    assert!((fourier_value(1.0, &harmonics, 0.375) + 1.0).abs() < 1e-9);
}

#[test]
fn test_fourier_zero_harmonics_returns_zero() {
    let harmonics: [(f64, f64); 0] = [];
    for &t in &[0.0_f64, 1.0, 100.0, -50.0] {
        assert_eq!(fourier_value(1.0 / 365.25, &harmonics, t), 0.0);
    }
}

#[test]
fn test_periodic_spline_partition_of_unity() {
    // gh#59 v2: with all coefs = 1, the B-spline sums to 1 everywhere
    // (partition of unity is preserved by the periodic wrap-fold).
    let coefs = [1.0; 6];
    for t in [0.0_f64, 0.5, 1.0, 1.7, 2.0, 3.3, 3.99] {
        let v = PeriodicBasis::new(4.0, 6, 3).unwrap().eval(t, |k| coefs[k]);
        assert!(
            (v - 1.0).abs() < 1e-12,
            "partition-of-unity violated at t={}: {}", t, v
        );
    }
}

#[test]
fn test_periodic_spline_wraps() {
    // gh#59 v2: f(t) == f(t + period).
    let coefs = [0.7, 1.2, 0.9, 0.5, 1.1, 0.8];
    for t in [0.123_f64, 1.5, 2.9, 3.6] {
        let a = PeriodicBasis::new(4.0, 6, 3).unwrap().eval(t, |k| coefs[k]);
        let b = PeriodicBasis::new(4.0, 6, 3).unwrap().eval(t + 4.0, |k| coefs[k]);
        let c = PeriodicBasis::new(4.0, 6, 3).unwrap().eval(t - 4.0, |k| coefs[k]);
        assert!((a - b).abs() < 1e-12, "periodicity at t={}: a={} t+P={}", t, a, b);
        assert!((a - c).abs() < 1e-12, "periodicity at t={}: a={} t-P={}", t, a, c);
    }
}
