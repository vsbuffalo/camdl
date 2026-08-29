//! gh#226 — degenerate (all-`-inf`) fits must be caught, not completed with
//! exit 0.
//!
//! `run_pmmh` intentionally returns `Ok(-inf)` for a ruled-out θ (the MH ratio
//! rejects it), and a whole run of `Ok(-inf)` is a legitimate *raw* sampler
//! result. The bug (gh#226) is that the DRIVER used to accept that degenerate
//! result — acceptance 0, MAP loglik `-inf` — and write a degenerate posterior
//! with exit 0. The fix adds a whole-fit backstop keyed on the shared
//! `sim::inference::no_finite_anchor` predicate.
//!
//! These sim-level tests pin (a) the degenerate raw result `run_pmmh` produces
//! for an all-`-inf` surface, (b) that the backstop predicate flags it, and
//! (c) that it does NOT flag a healthy finite fit (the false-positive guard the
//! driver-level backstop relies on to keep legitimate mixed-init fits running).
//! `run_pmmh` takes a black-box loglik closure, so the all-`-inf` surface is
//! injected directly — no pathological model needed.

use sim::error::SimError;
use sim::inference::{
    if2::{EstimatedParam, Transform},
    no_finite_anchor,
    pmmh::{run_pmmh, PMMHConfig, Prior},
    prior::Density,
};

/// μ death-rate spec at param index 0 (log-transformed, positive).
fn mu_param() -> EstimatedParam {
    EstimatedParam {
        name: "mu".into(),
        index: 0,
        initial: 0.01,
        rw_sd: 0.002,
        transform: Transform::Log { lo: 1e-6, hi: 1.0 },
        lower: 1e-6,
        upper: 1.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

/// A small PMMH config — we only need the chain to run its MH loop; the loglik
/// surface is supplied by the closure, so particle count / step count are tiny.
fn small_config() -> PMMHConfig {
    PMMHConfig {
        t_start: 0.0,
        n_steps: 20,
        n_particles: 10,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 0,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0,
        rho: None,
        n_source_groups: 0,
    }
}

/// (a)+(b): an all-`-inf` likelihood surface produces the degenerate raw result
/// (Ok, acceptance 0, MAP loglik non-finite) AND is flagged by the backstop.
#[test]
fn all_inf_surface_is_degenerate_and_flagged() {
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(Density::Normal { mean: 0.01, sd: 0.01 })];
    let base_params = vec![0.01_f64];
    let cfg = small_config();

    // Every θ scores -inf — a recoverable error firing at every θ, or a
    // genuinely impossible-data region (gh#226).
    let eval_inf = |_: &[f64], _: u64| -> Result<f64, SimError> { Ok(f64::NEG_INFINITY) };

    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &cfg, &[],
        &eval_inf, None, 42, None, None, String::new(),
    )
    .expect("run_pmmh still returns Ok for a degenerate all-(-inf) surface (raw sampler result)");

    // The degenerate raw result the driver used to silently accept.
    assert_eq!(result.acceptance_rate, 0.0,
        "MH chain is stuck (-inf - (-inf) = NaN never accepts): acceptance must be 0");
    assert!(!result.map_loglik.is_finite(),
        "MAP loglik must be non-finite for an all-(-inf) surface; got {}", result.map_loglik);

    // gh#226 backstop: the whole-fit predicate MUST flag this degenerate result.
    assert!(no_finite_anchor(result.map_loglik),
        "no_finite_anchor must flag an all-(-inf) fit (map_loglik={})", result.map_loglik);
}

/// (c): a healthy finite surface reaches a finite anchor and must NOT be
/// flagged — a spurious fire would break legitimate fits, which is worse than
/// the bug.
#[test]
fn finite_surface_reaches_a_finite_anchor() {
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(Density::Normal { mean: 0.01, sd: 0.01 })];
    let base_params = vec![0.01_f64];
    let cfg = small_config();

    // Every θ scores a finite loglik — a normal fit.
    let eval_finite = |_: &[f64], _: u64| -> Result<f64, SimError> { Ok(-42.0) };

    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &cfg, &[],
        &eval_finite, None, 42, None, None, String::new(),
    )
    .expect("run_pmmh returns Ok for a finite surface");

    assert!(result.map_loglik.is_finite(),
        "a finite surface must yield a finite MAP loglik; got {}", result.map_loglik);
    assert!(!no_finite_anchor(result.map_loglik),
        "no_finite_anchor must NOT flag a healthy finite fit (map_loglik={})", result.map_loglik);
}

/// The predicate itself: `-inf`, `+inf`, and `NaN` are all "no finite anchor";
/// ordinary finite logliks are not.
#[test]
fn no_finite_anchor_predicate_semantics() {
    assert!(no_finite_anchor(f64::NEG_INFINITY));
    assert!(no_finite_anchor(f64::INFINITY));
    assert!(no_finite_anchor(f64::NAN));
    assert!(!no_finite_anchor(-42.0));
    assert!(!no_finite_anchor(0.0));
    assert!(!no_finite_anchor(-1234.5));
}
