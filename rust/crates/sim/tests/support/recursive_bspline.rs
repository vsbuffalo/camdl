//! The recursive, allocating periodic B-spline evaluator camdl used before
//! `sim::bspline` (proposal 2026-10-27-estimable-bsplines §3, §9 Stage 1),
//! kept verbatim as a test oracle and as the "before" arm of
//! `benches/spline_eval.rs`. Not part of the library.
//!
//! `bspline_value` is the naive Cox–de Boor recursion over a full knot vector,
//! evaluating one basis function at a time; `eval_periodic_bspline_recursive`
//! builds the periodic knot and basis vectors per call, evaluates every basis
//! function, folds and applies the centring shift.

/// Evaluate the de Boor B-spline recurrence for a single basis
/// function index `i` at point `x`, on a knot vector `knots` of
/// length `n_basis + degree + 1`.
///
/// Returns `B_i^degree(x)` where
///   B_i^0(x)  = 1 if knots[i] ≤ x < knots[i+1], else 0
///   B_i^k(x)  = ((x − knots[i])     / (knots[i+k]   − knots[i]))   · B_i^{k-1}(x)
///             + ((knots[i+k+1] − x) / (knots[i+k+1] − knots[i+1])) · B_{i+1}^{k-1}(x)
///
/// Pure function; no allocation. Recursion depth = degree (≤ 3 for
/// the typical cubic case, so no stack concerns).
pub fn bspline_value(x: f64, i: usize, degree: u32, knots: &[f64]) -> f64 {
    if degree == 0 {
        let lo = knots[i];
        let hi = knots[i + 1];
        if x >= lo && x < hi { 1.0 } else { 0.0 }
    } else {
        let p = degree as usize;
        let denom_a = knots[i + p] - knots[i];
        let denom_b = knots[i + p + 1] - knots[i + 1];
        let mut sum = 0.0;
        if denom_a > 0.0 {
            sum += (x - knots[i]) / denom_a
                 * bspline_value(x, i, degree - 1, knots);
        }
        if denom_b > 0.0 {
            sum += (knots[i + p + 1] - x) / denom_b
                 * bspline_value(x, i + 1, degree - 1, knots);
        }
        sum
    }
}

/// Evaluate `f(t) = Σ_i coefs[i] · B_i(t)` for a periodic B-spline
/// basis with uniform knots over `[0, period)`.
///
/// Algorithm matches pomp's `periodic_bspline_basis_eval_deriv`
/// numerically (we verify against the pomp oracle TSV) but is a
/// clean-room implementation from primary sources — see module-level
/// comment for citations.
///
/// Returns 0.0 for `period <= 0`, empty coefs, or `n_basis <= degree`.
pub fn eval_periodic_bspline_recursive(
    t: f64,
    period: f64,
    n_basis: u32,
    degree: u32,
    coefs: &[f64],
) -> f64 {
    if period <= 0.0 || coefs.is_empty() {
        return 0.0;
    }
    let nb = n_basis as usize;
    let deg = degree as usize;
    if nb <= deg || coefs.len() != nb {
        return 0.0;
    }

    // Uniform knots over [-degree·dx, (n_basis + degree)·dx].
    // Total length: n_basis + 2*degree + 1 basis-eval positions need
    // n_basis + 2*degree + degree + 1 = n_basis + 3*degree + 1 knot
    // entries (basis function i = nb + deg - 1 uses knots[i..i+deg+1]).
    let dx = period / nb as f64;
    let n_eval = nb + 2 * deg + 1;
    let n_knots = n_eval + deg + 1;
    let mut knots = Vec::with_capacity(n_knots);
    for k in 0..n_knots {
        // k=0 corresponds to index -degree.
        let kk = (k as i64) - (deg as i64);
        knots.push(kk as f64 * dx);
    }

    // Wrap t into [0, period). Use rem_euclid for negative-t correctness.
    let mut x = t.rem_euclid(period);
    // Floating-point rounding: rem_euclid can land exactly on `period`
    // when t is a tiny negative multiple. Clamp to [0, period).
    if x >= period {
        x -= period;
    }

    // Standard (non-periodic) basis evaluation for all n_eval functions
    // at x. yy[k] is the k-th basis value over the extended knot vector.
    let mut yy = vec![0.0_f64; n_eval];
    for k in 0..n_eval {
        yy[k] = bspline_value(x, k, degree, &knots);
    }

    // Periodic wrap-fold: the last `degree` basis functions (indices
    // nb..nb+degree) overlap with the first `degree` once we identify
    // t = 0 with t = period. Fold them onto each other so the basis
    // is genuinely periodic and partition-of-unity is preserved.
    for k in 0..deg {
        yy[k] += yy[nb + k];
    }

    // Centering shift: rotate output so basis 0 is centered on t = 0
    // rather than at t = -(degree+1)/2 · dx. For cubic (degree=3),
    // shift = 1. This is a labeling choice that matches pomp's
    // convention (Wand & Ormerod 2008 §3 documents the rationale).
    // scipy's BSpline does not apply this shift; we account for it
    // in the scipy oracle by rolling the input coefficient vector
    // before constructing the spline.
    let shift = (deg.saturating_sub(1)) / 2;

    // Linear combination yy[(shift + k) % nb] · coefs[k].
    let mut sum = 0.0;
    for k in 0..nb {
        sum += coefs[k] * yy[(shift + k) % nb];
    }
    sum
}
