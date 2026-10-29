//! Regression test for the IF2 cooling schedule (gh#363).
//!
//! pomp's geometric `cooling.fraction.50` reaches the fraction after exactly
//! `cooling_target_iters` iterations (pomp's fixed "50") and keeps cooling past
//! it: `SD(m) = cooling_fraction^(m / cooling_target_iters)` — exponent 1
//! (`pomp:::mif2_cooling` returns `alpha`, and `mif2_pfilter` perturbs with
//! `alpha · rw.sd`; it does NOT use the squared `gamma`). The earlier code used
//! exponent 2.0, which reaches the fraction at the midpoint and `fraction²` at
//! the endpoint — twice as fast as pomp.

use sim::inference::if2::CoolingSchedule;

#[test]
fn fraction_reached_at_target_iters_not_midpoint() {
    // The load-bearing property: at iter = cooling_target_iters, the SD multiplier
    // equals cooling_fraction (pomp). The bug (exponent 2.0) gives cooling_fraction²
    // here, so this fails against it.
    for &frac in &[0.05, 0.5, 0.7, 0.95] {
        for &target in &[25usize, 50, 80] {
            for &n_obs in &[1usize, 10, 52] {
                let at_target = CoolingSchedule { fraction: frac, target_iters: target }.multiplier_at_iter(n_obs, target);
                assert!(
                    (at_target - frac).abs() < 1e-12,
                    "at iter=target={target}, multiplier must equal cooling_fraction={frac}, \
                     got {at_target} (n_obs={n_obs}) — exponent-2.0 would give {}",
                    frac * frac
                );
            }
        }
    }
}

#[test]
fn matches_pomp_geometric_schedule() {
    // pomp: alpha(m) = cooling.fraction.50^(m/50). With cooling_target_iters=50,
    // camdl's per-iteration multiplier must equal frac^(iter/50) to machine
    // precision, independent of n_obs (the (1+n_obs) granularity cancels).
    let frac = 0.5;
    let target = 50;
    for &n_obs in &[1usize, 10, 100] {
        for &iter in &[1usize, 25, 40, 50, 80] {
            let camdl = CoolingSchedule { fraction: frac, target_iters: target }.multiplier_at_iter(n_obs, iter);
            let pomp = frac.powf(iter as f64 / target as f64);
            assert!(
                (camdl - pomp).abs() < 1e-12,
                "iter={iter}: camdl {camdl} vs pomp frac^(m/50) {pomp} (n_obs={n_obs})"
            );
        }
    }
}

#[test]
fn per_step_factor_consistent_with_at_iter() {
    // The at-iter multiplier is exactly the per-step factor raised to the
    // iteration's global-step count (iter * (1 + n_obs)).
    let (frac, target, n_obs) = (0.05, 50usize, 10usize);
    let per_step = CoolingSchedule { fraction: frac, target_iters: target }.per_step_factor(n_obs);
    for &iter in &[1usize, 10, 25, 50] {
        let a = CoolingSchedule { fraction: frac, target_iters: target }.multiplier_at_iter(n_obs, iter);
        let b = per_step.powf((iter * (1 + n_obs)) as f64);
        assert!((a - b).abs() < 1e-12, "iter={iter}: {a} vs {b}");
    }
}
