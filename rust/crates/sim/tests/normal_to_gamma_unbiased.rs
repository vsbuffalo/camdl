//! gh#372: the correlated-PF overdispersion multiplier must be the *exact* Gamma
//! inverse-CDF — unit-mean with no atom at 0 — not the Wilson-Hilferty
//! approximation, which biased the mean low and clamped a growing fraction of
//! draws to exactly 0 for `shape < ~1` (silently biasing correlated PMMH on
//! overdispersed models, e.g. the He-2010 measles benchmark at shape=0.355).

use sim::inference::correlated_pf::normal_to_gamma;
use sim::inference::obs_loglik::normal_quantile;

/// The OLD Wilson-Hilferty map, retained here only to demonstrate the bias the
/// fix removed (it is no longer in the codebase).
fn wh_reference(z: f64, shape: f64, scale: f64) -> f64 {
    if shape < 1e-6 {
        return 1.0;
    }
    let c = 1.0 / (9.0 * shape);
    let cube = 1.0 - c + z * c.sqrt();
    let x = if cube > 0.0 { shape * cube * cube * cube } else { 0.0 };
    x * scale
}

/// Mean and atom-at-0 count of `f` over a uniform-`u` grid (`u → z = Φ⁻¹(u)`);
/// `E[f(Φ⁻¹(U))]` is the mean of the multiplier under `z ~ N(0,1)`.
fn mean_and_zeros(f: impl Fn(f64) -> f64, n: usize) -> (f64, usize) {
    let mut sum = 0.0;
    let mut zeros = 0usize;
    for i in 0..n {
        let u = (i as f64 + 0.5) / n as f64;
        let g = f(normal_quantile(u));
        sum += g;
        if g <= 0.0 {
            zeros += 1;
        }
    }
    (sum / n as f64, zeros)
}

#[test]
fn multiplier_is_unit_mean_with_no_atom_at_zero() {
    let n = 200_000;
    // The multiplier is Gamma(shape, 1/shape): unit mean by construction.
    for &shape in &[2.0, 0.355, 0.1] {
        let scale = 1.0 / shape;
        let (mean, zeros) = mean_and_zeros(|z| normal_to_gamma(z, shape, scale), n);
        assert!(
            (mean - 1.0).abs() < 1e-2,
            "fixed E[G] must be ~1 at shape={shape}, got {mean}"
        );
        assert_eq!(
            zeros, 0,
            "fixed multiplier must never be 0 (no atom) at shape={shape}, got {zeros} zeros"
        );
    }

    // Document what the fix removed: at the flagship shape=0.355, WH is biased
    // low and clamps a chunk of draws to exactly 0.
    let (mean_wh, zeros_wh) = mean_and_zeros(|z| wh_reference(z, 0.355, 1.0 / 0.355), n);
    assert!(mean_wh < 0.99, "WH reference should be biased low, got E={mean_wh}");
    assert!(zeros_wh > n / 20, "WH reference should clamp many draws to 0, got {zeros_wh}");
}

#[test]
fn multiplier_is_monotone_in_z_preserving_crn() {
    // CRN coupling requires the z → g map be monotone increasing.
    let (shape, scale) = (0.355, 1.0 / 0.355);
    let mut prev = f64::NEG_INFINITY;
    for i in 0..2000 {
        let z = -5.0 + 10.0 * (i as f64) / 2000.0;
        let g = normal_to_gamma(z, shape, scale);
        assert!(g > prev, "must be strictly increasing in z at z={z}: {g} <= {prev}");
        prev = g;
    }
}
