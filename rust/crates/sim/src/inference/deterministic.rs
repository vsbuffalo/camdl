//! Deterministic-likelihood optimization (Phase 1 of the ODE-inference proposal).
//!
//! `optimize_det()` runs a local NLopt algorithm — Sbplx (default, robust to
//! boundary non-smoothness) or BOBYQA (faster on smooth interior objectives) —
//! on a user-supplied loglik closure with parameter bounds. Used by
//! `cli::fit::nlopt_stage::run_stage` for ODE-backed MLE; see
//! `docs/dev/proposals/2026-05-04-ode-inference-three-phase.md` §Phase 1.
//!
//! This module is pure: it knows nothing about ODE solves, observation
//! models, or fit.toml schemas. The caller wires those into the closure.

use nlopt::{Algorithm, Nlopt, Target};

// Re-export so downstream crates (`cli::fit::nlopt_stage`) can pattern-
// match on the `OptStatus::Converged(SuccessState)` variant without
// taking a direct dep on the nlopt crate. The variant already leaks
// `SuccessState` as part of its public API; the re-export just makes
// that fact callable.
pub use nlopt::SuccessState;

/// Which NLopt algorithm to run. Phase 1 surfaces two — `Sbplx` is the default
/// for compartmental likelihoods (smooth interior, possibly non-smooth at
/// parameter-bound boundaries); `Bobyqa` is faster on smooth objectives but
/// fails at boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NloptAlgorithm {
    Sbplx,
    Bobyqa,
}

impl NloptAlgorithm {
    fn as_nlopt(self) -> Algorithm {
        match self {
            Self::Sbplx => Algorithm::Sbplx,
            Self::Bobyqa => Algorithm::Bobyqa,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sbplx => "nl-sbplx",
            Self::Bobyqa => "nl-bobyqa",
        }
    }
}

/// Outcome classification for a single optimization run. The proposal calls
/// out the distinction between a true convergence (parameter / function
/// tolerance reached) and a soft failure (`MaxEvalReached`) so the per-stage
/// runner can surface them separately rather than lumping under a single
/// "status" string.
#[derive(Debug, Clone, Copy)]
pub enum OptStatus {
    /// `Success` / `XtolReached` / `FtolReached` / `StopValReached`.
    Converged(SuccessState),
    /// Hit the `max_evals` budget without converging. Soft failure; the
    /// returned params are the best seen so far but the optimizer wasn't
    /// done with them.
    MaxEvalReached,
    /// Wall-clock budget exhausted. Phase 1 doesn't set `maxtime`, so this
    /// shouldn't fire — kept for completeness.
    MaxTimeReached,
    /// NLopt returned a hard error (RoundoffLimited, ForcedStop, ...).
    /// `params` is the last evaluated point; `loglik` is its score.
    Failed,
}

impl PartialEq for OptStatus {
    fn eq(&self, other: &Self) -> bool {
        // SuccessState has no PartialEq impl in nlopt 0.8 — match it via
        // the variant discriminant by way of `as_str()`.
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && match (self, other) {
                (Self::Converged(a), Self::Converged(b)) => {
                    successstate_as_str(*a) == successstate_as_str(*b)
                }
                _ => true,
            }
    }
}

impl Eq for OptStatus {}

fn successstate_as_str(s: SuccessState) -> &'static str {
    match s {
        SuccessState::Success => "success",
        SuccessState::StopValReached => "stopval_reached",
        SuccessState::FtolReached => "ftol_reached",
        SuccessState::XtolReached => "xtol_reached",
        SuccessState::MaxEvalReached => "maxeval_reached",
        SuccessState::MaxTimeReached => "maxtime_reached",
    }
}

impl OptStatus {
    pub fn is_converged(self) -> bool {
        matches!(self, OptStatus::Converged(_))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Converged(s) => successstate_as_str(s),
            Self::MaxEvalReached => "maxeval_reached",
            Self::MaxTimeReached => "maxtime_reached",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptResult {
    pub params: Vec<f64>,
    pub loglik: f64,
    pub status: OptStatus,
    pub n_evals: usize,
}

/// Maximize `objective(params)` subject to `bounds` using the given local
/// NLopt algorithm.
///
/// `tolerance` is `xtol_rel` — relative parameter tolerance for convergence.
/// `max_evals` caps the per-call objective-evaluation count; hitting it is
/// reported as `OptStatus::MaxEvalReached` (soft failure).
///
/// Errors: dimension mismatch between `initial` and `bounds`, or any NLopt
/// configuration error (e.g. invalid bounds). Numeric failures inside the
/// optimizer come back through `OptStatus::Failed` rather than `Err`.
pub fn optimize_det<F>(
    algorithm: NloptAlgorithm,
    initial: &[f64],
    bounds: &[(f64, f64)],
    tolerance: f64,
    max_evals: usize,
    objective: F,
) -> Result<OptResult, String>
where
    F: FnMut(&[f64]) -> f64,
{
    let dim = initial.len();
    if dim != bounds.len() {
        return Err(format!(
            "optimize_det: initial has dim {} but bounds has len {}",
            dim,
            bounds.len()
        ));
    }
    let lower: Vec<f64> = bounds.iter().map(|&(lo, _)| lo).collect();
    let upper: Vec<f64> = bounds.iter().map(|&(_, hi)| hi).collect();
    for (i, (l, u)) in lower.iter().zip(&upper).enumerate() {
        // NaN arm explicit: a NaN bound is not a valid interval, and
        // `l >= u` alone is false for NaN.
        if l.is_nan() || u.is_nan() || l >= u {
            return Err(format!(
                "optimize_det: bounds[{}] has lower {} >= upper {}",
                i, l, u
            ));
        }
    }

    // NLopt's `ObjFn<T>` trait requires an `Fn`-callable objective. We
    // smuggle our `FnMut` user closure through `user_data`: the framework
    // passes it as `&mut UserData<F>` to the static callback, which in
    // turn calls `(ud.f)(params)` (legal because we have `&mut F`).
    struct UserData<F: FnMut(&[f64]) -> f64> {
        f: F,
        n_evals: usize,
    }

    fn callback<F: FnMut(&[f64]) -> f64>(
        params: &[f64],
        _grad: Option<&mut [f64]>,
        ud: &mut UserData<F>,
    ) -> f64 {
        ud.n_evals += 1;
        let v = (ud.f)(params);
        if v.is_finite() {
            v
        } else {
            // NLopt is undefined on NaN. Map non-finite logliks (model
            // blew up at this θ) to a large negative value so the
            // optimizer steers away from this region.
            -1e100
        }
    }

    let user_data = UserData { f: objective, n_evals: 0 };
    let mut opt = Nlopt::new(
        algorithm.as_nlopt(),
        dim,
        callback::<F>,
        Target::Maximize,
        user_data,
    );
    opt.set_lower_bounds(&lower)
        .map_err(|e| format!("nlopt set_lower_bounds: {:?}", e))?;
    opt.set_upper_bounds(&upper)
        .map_err(|e| format!("nlopt set_upper_bounds: {:?}", e))?;
    opt.set_xtol_rel(tolerance)
        .map_err(|e| format!("nlopt set_xtol_rel: {:?}", e))?;
    opt.set_maxeval(max_evals as u32)
        .map_err(|e| format!("nlopt set_maxeval: {:?}", e))?;

    // Clamp `initial` into the bounds box so a slightly out-of-bounds
    // starting point (rounding error in LHS, prior-state value at the
    // boundary) doesn't trip NLopt's invalid-args check.
    let mut x: Vec<f64> = initial
        .iter()
        .zip(&lower)
        .zip(&upper)
        .map(|((&v, &l), &u)| v.clamp(l, u))
        .collect();

    let outcome = opt.optimize(&mut x);
    let n_evals = opt.recover_user_data().n_evals;

    let (loglik, status) = match outcome {
        Ok((s, ll)) => {
            let st = match s {
                SuccessState::MaxEvalReached => OptStatus::MaxEvalReached,
                SuccessState::MaxTimeReached => OptStatus::MaxTimeReached,
                _ => OptStatus::Converged(s),
            };
            (ll, st)
        }
        Err((_e, ll)) => (ll, OptStatus::Failed),
    };
    Ok(OptResult {
        params: x,
        loglik,
        status,
        n_evals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadratic in 2D: f(x, y) = -((x-3)^2 + (y+1)^2). Maximum at (3, -1)
    /// with value 0. Both Sbplx and BOBYQA should find this within a few
    /// dozen evaluations.
    fn quadratic(p: &[f64]) -> f64 {
        let dx = p[0] - 3.0;
        let dy = p[1] + 1.0;
        -(dx * dx + dy * dy)
    }

    /// A NaN bound is not a valid interval. `l >= u` alone is false for NaN,
    /// so without the explicit NaN arm a NaN bound would pass validation and
    /// reach NLopt, which is not a diagnosable failure for the caller.
    #[test]
    fn nan_bounds_are_rejected() {
        for bad in [(f64::NAN, 10.0), (-10.0, f64::NAN), (f64::NAN, f64::NAN)] {
            let r = optimize_det(
                NloptAlgorithm::Sbplx,
                &[0.0, 0.0],
                &[(-10.0, 10.0), bad],
                1e-8,
                500,
                quadratic,
            );
            assert!(r.is_err(), "bounds {bad:?} must be rejected, got Ok");
        }
        // Negative control: the same call with finite bounds still runs.
        assert!(optimize_det(
            NloptAlgorithm::Sbplx,
            &[0.0, 0.0],
            &[(-10.0, 10.0), (-10.0, 10.0)],
            1e-8,
            500,
            quadratic,
        )
        .is_ok());
    }

    #[test]
    fn sbplx_finds_quadratic_maximum() {
        let result = optimize_det(
            NloptAlgorithm::Sbplx,
            &[0.0, 0.0],
            &[(-10.0, 10.0), (-10.0, 10.0)],
            1e-8,
            500,
            quadratic,
        )
        .unwrap();
        assert!(result.status.is_converged(), "status: {:?}", result.status);
        assert!((result.params[0] - 3.0).abs() < 1e-3);
        assert!((result.params[1] - (-1.0)).abs() < 1e-3);
        assert!(result.loglik > -1e-4);
        assert!(result.n_evals > 0);
    }

    #[test]
    fn bobyqa_finds_quadratic_maximum() {
        let result = optimize_det(
            NloptAlgorithm::Bobyqa,
            &[0.0, 0.0],
            &[(-10.0, 10.0), (-10.0, 10.0)],
            1e-8,
            500,
            quadratic,
        )
        .unwrap();
        assert!(result.status.is_converged(), "status: {:?}", result.status);
        assert!((result.params[0] - 3.0).abs() < 1e-3);
        assert!((result.params[1] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn maxeval_reported_as_soft_failure() {
        // Severely starve the budget: with max_evals=2 and a fresh start,
        // the optimizer can't converge. Should report MaxEvalReached, not
        // Converged or Failed.
        let result = optimize_det(
            NloptAlgorithm::Sbplx,
            &[0.0, 0.0],
            &[(-10.0, 10.0), (-10.0, 10.0)],
            1e-12,
            2,
            quadratic,
        )
        .unwrap();
        assert_eq!(result.status, OptStatus::MaxEvalReached);
        assert!(!result.status.is_converged());
        assert!(result.n_evals <= 3); // budget + maybe 1 final eval
    }

    #[test]
    fn dim_mismatch_errors_cleanly() {
        let err = optimize_det(
            NloptAlgorithm::Sbplx,
            &[0.0, 0.0],
            &[(-1.0, 1.0)],
            1e-6,
            10,
            quadratic,
        )
        .unwrap_err();
        assert!(err.contains("initial"));
    }

    #[test]
    fn empty_bounds_errors_cleanly() {
        let err = optimize_det(
            NloptAlgorithm::Sbplx,
            &[0.0],
            &[(1.0, 1.0)],
            1e-6,
            10,
            |_| 0.0,
        )
        .unwrap_err();
        assert!(err.contains("lower"));
    }

    #[test]
    fn nan_objective_steered_away() {
        // Objective returns NaN at x[0] < 0; valid quadratic for x[0] >= 0.
        // Optimizer should still find the maximum at x = (3, -1).
        let result = optimize_det(
            NloptAlgorithm::Sbplx,
            &[5.0, 0.0],
            &[(0.0, 10.0), (-10.0, 10.0)],
            1e-6,
            500,
            |p| {
                if p[0] < 0.0 {
                    f64::NAN
                } else {
                    quadratic(p)
                }
            },
        )
        .unwrap();
        assert!(result.status.is_converged());
        assert!((result.params[0] - 3.0).abs() < 1e-3);
    }
}
