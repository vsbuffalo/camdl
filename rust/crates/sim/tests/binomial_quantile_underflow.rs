//! Regression test for `binomial_quantile` underflow (gh#362).
//!
//! The inverse binomial CDF walked the CDF from k=0 seeded at `(1-p)^n`. For p
//! near 1 with large n that start underflows to 0, the walk never accumulates,
//! and the function returned `n` — the whole source compartment — for every `u`.
//! On the correlated-PF / correlated-PMMH path that silently over-drains a fast
//! compartment. The fix walks the lighter tail so the seed never underflows.

use sim::inference::correlated_pf::binomial_quantile;

/// Brute-force inverse CDF by a direct walk from 0. Correct for SMALL n, where
/// `(1-p)^n` does not underflow; used as the reference for exact cross-checks.
fn ref_quantile(n: u64, p: f64, u: f64) -> u64 {
    let q = 1.0 - p;
    let mut cdf = 0.0;
    let mut pmf = q.powi(n as i32);
    for k in 0..=n {
        cdf += pmf;
        if cdf >= u {
            return k;
        }
        pmf *= (n - k) as f64 / (k + 1) as f64 * p / q;
    }
    n
}

#[test]
fn matches_reference_across_p_and_u() {
    // Exact agreement with the brute-force CDF for small n (no underflow),
    // including p > 0.5 where the fix walks the complement (failure) tail. This
    // pins the complement mapping — a wrong ≥/> boundary would show up here.
    for &n in &[4u64, 7, 10, 20, 30] {
        for &p in &[0.1, 0.25, 0.5, 0.7, 0.9, 0.95] {
            for i in 1..20 {
                let u = i as f64 / 20.0;
                let got = binomial_quantile(n, p, u);
                let want = ref_quantile(n, p, u);
                assert_eq!(got, want, "binomial_quantile({n}, {p}, {u}) = {got}, ref = {want}");
            }
        }
    }
}

#[test]
fn underflow_regime_returns_sensible_value_not_full_compartment() {
    // Binomial(1000, 0.99): (0.01)^1000 underflows; median is ~990, NOT 1000.
    let k = binomial_quantile(1000, 0.99, 0.5);
    assert!(k < 1000, "u=0.5 must not return the full compartment (got {k})");
    assert!((k as f64 - 990.0).abs() <= 4.0, "expected ~990 (the mean), got {k}");

    // Deeper underflow: (0.001)^1000 == 0.0 exactly. Median ~999.
    let k2 = binomial_quantile(1000, 0.999, 0.5);
    assert!(
        k2 < 1000 && (k2 as f64 - 999.0).abs() <= 3.0,
        "expected ~999 (< 1000), got {k2}"
    );

    // The distribution has spread: different u must give different k. The bug
    // collapsed every u to n (1000).
    let lo = binomial_quantile(1000, 0.99, 0.05);
    let hi = binomial_quantile(1000, 0.99, 0.95);
    assert!(lo < hi, "quantile must increase with u (got lo={lo}, hi={hi})");
    assert!(lo < 1000 && hi <= 1000, "both must be bracketed by n (lo={lo}, hi={hi})");
}

#[test]
fn monotone_and_bracketed() {
    // Non-decreasing in u and within [0, n] across the whole range.
    let (n, p) = (1000u64, 0.98);
    let mut prev = 0u64;
    for i in 1..=99 {
        let u = i as f64 / 100.0;
        let k = binomial_quantile(n, p, u);
        assert!(k <= n, "out of range at u={u}: {k} > {n}");
        assert!(k >= prev, "non-monotone at u={u}: {k} < prev {prev}");
        prev = k;
    }
}
