//! Deterministic-likelihood optimization (Phase 1 of the ODE-inference proposal).
//!
//! `optimize_det()` runs a local NLopt algorithm — Sbplx (default, robust to
//! boundary non-smoothness), BOBYQA (faster on smooth interior objectives), or
//! L-BFGS (quasi-Newton, needs a gradient) — on a user-supplied loglik closure
//! with parameter bounds. Used by `cli::fit::nlopt_stage::run_stage` for
//! ODE-backed MLE; see
//! `docs/dev/proposals/2026-05-04-ode-inference-three-phase.md` §Phase 1.
//!
//! This module is pure: it knows nothing about ODE solves, observation
//! models, or fit.toml schemas. The caller wires those into the closure —
//! including the gradient, which arrives through NLopt's own slot on the
//! objective rather than through a second callback, so one objective type
//! serves the derivative-free and gradient algorithms alike.

use nlopt::{Algorithm, Nlopt, Target};

// Re-export so downstream crates (`cli::fit::nlopt_stage`) can pattern-
// match on the `OptStatus::Converged(SuccessState)` variant without
// taking a direct dep on the nlopt crate. The variant already leaks
// `SuccessState` as part of its public API; the re-export just makes
// that fact callable.
pub use nlopt::SuccessState;

/// Which NLopt algorithm to run. `Sbplx` is the default for compartmental
/// likelihoods (smooth interior, possibly non-smooth at parameter-bound
/// boundaries); `Bobyqa` is faster on smooth objectives but fails at
/// boundaries. Both are derivative-free. `Lbfgs` is the gradient method: it
/// asks for `∇` at every point, which the caller supplies through the
/// objective's gradient slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NloptAlgorithm {
    Sbplx,
    Bobyqa,
    Lbfgs,
}

impl NloptAlgorithm {
    fn as_nlopt(self) -> Algorithm {
        match self {
            Self::Sbplx => Algorithm::Sbplx,
            Self::Bobyqa => Algorithm::Bobyqa,
            Self::Lbfgs => Algorithm::Lbfgs,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sbplx => "nl-sbplx",
            Self::Bobyqa => "nl-bobyqa",
            Self::Lbfgs => "nl-lbfgs",
        }
    }

    /// Whether NLopt will ask this algorithm's objective for a gradient — the
    /// `NLOPT_{G,L}D_*` half of the naming convention, where `D` "denotes
    /// derivative-free/gradient-based algorithms" (NLopt Algorithms reference,
    /// "Local gradient-based optimization"). A derivative-free algorithm passes
    /// `None` in the gradient slot and the caller must not compute one.
    pub fn uses_gradient(self) -> bool {
        matches!(self, Self::Lbfgs)
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

/// The value reported to NLopt in place of a non-finite objective. Large
/// enough that any scorable point beats it, finite so the optimizer's own
/// arithmetic stays defined.
const NONFINITE_FLOOR: f64 = -1e100;

#[derive(Debug, Clone)]
pub struct OptResult {
    pub params: Vec<f64>,
    pub loglik: f64,
    pub status: OptStatus,
    pub n_evals: usize,
}

/// Maximize `objective(params, grad)` subject to `bounds` using the given
/// local NLopt algorithm.
///
/// `grad` is NLopt's own gradient slot, passed through verbatim: `Some(g)` for
/// a gradient algorithm (`NloptAlgorithm::uses_gradient`), in which case the
/// objective must write `∇` of the value it returns into `g`; `None` for a
/// derivative-free one, which ignores it. Nothing here computes a gradient —
/// the caller owns that, the same way it owns the value.
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
    F: FnMut(&[f64], Option<&mut [f64]>) -> f64,
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
        // NaN arm explicit: a NaN bound is not a valid interval, and `l >= u`
        // alone is false for NaN.
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
    // turn calls `(ud.f)(params, grad)` (legal because we have `&mut F`).
    struct UserData<F: FnMut(&[f64], Option<&mut [f64]>) -> f64> {
        f: F,
        n_evals: usize,
    }

    fn callback<F: FnMut(&[f64], Option<&mut [f64]>) -> f64>(
        params: &[f64],
        grad: Option<&mut [f64]>,
        ud: &mut UserData<F>,
    ) -> f64 {
        ud.n_evals += 1;
        // The gradient slot goes through UNTOUCHED. It is tempting to sanitize
        // a non-finite gradient to zeros the way the value is floored below,
        // and it is wrong: measured against nlopt 0.8.1, `Lbfgs` answers a
        // non-finite gradient with `NLOPT_FAILURE`, ending the optimization at
        // the last point — which `optimize_det` reports as `OptStatus::Failed`,
        // the loud outcome. Zeroing the slot instead turns that same point into
        // a stationary one: the run returns `Ok((Success, …))` sitting exactly
        // where the gradient could not be computed, i.e. a reported optimum
        // whose only property is that the gradient failed there. A caller that
        // wants a specific message latches its own reason (see
        // `cli::fit::nlopt_stage::optimize_cell`); correctness does not rest on
        // NLopt's handling, only the promptness of the stop does.
        let raw = (ud.f)(params, grad);
        // NLopt is undefined on NaN. Map non-finite logliks (model blew up at
        // this θ) to a large negative value so the optimizer steers away from
        // this region.
        if raw.is_finite() { raw } else { NONFINITE_FLOOR }
    }

    let user_data = UserData { f: objective, n_evals: 0 };
    let mut opt = Nlopt::new(
        algorithm.as_nlopt(),
        dim,
        callback::<F>,
        Target::Maximize,
        user_data,
    );
    // Box bounds are set for every algorithm here, gradient ones included.
    // `NLOPT_LD_LBFGS` ("Low-storage BFGS") is one of the local gradient-based
    // algorithms, and the NLopt Algorithms reference says of that family: "Of
    // these algorithms, only MMA and SLSQP support arbitrary nonlinear
    // inequality constraints, and only SLSQP supports nonlinear equality
    // constraints; the rest support bound-constrained or unconstrained problems
    // only." (NLopt Algorithms, "Local gradient-based optimization".) So the
    // box is honoured — which this caller depends on, since every estimated
    // parameter carries a natural-scale `[lower, upper]`.
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
    fn quadratic_value(p: &[f64]) -> f64 {
        let dx = p[0] - 3.0;
        let dy = p[1] + 1.0;
        -(dx * dx + dy * dy)
    }

    /// [`quadratic_value`] in the objective shape `optimize_det` takes. The
    /// derivative-free algorithms get `None` and this ignores it, which is
    /// the whole point of the widened signature: one type, two consumers.
    fn quadratic(p: &[f64], _grad: Option<&mut [f64]>) -> f64 {
        quadratic_value(p)
    }

    /// The same quadratic WITH its analytic gradient
    /// `∇ = (-2(x-3), -2(y+1))`, written into NLopt's slot when one is asked
    /// for.
    fn quadratic_with_grad(p: &[f64], grad: Option<&mut [f64]>) -> f64 {
        if let Some(g) = grad {
            g[0] = -2.0 * (p[0] - 3.0);
            g[1] = -2.0 * (p[1] + 1.0);
        }
        quadratic_value(p)
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

    /// The widened objective, from the gradient side: NLopt hands `Some(slot)`
    /// to an `LD_` algorithm, the closure writes `∇` into it, and the box
    /// bounds are honoured (the NLopt Algorithms reference: the local
    /// gradient-based family "support[s] bound-constrained or unconstrained
    /// problems only" — bounds are in, nonlinear constraints are not).
    #[test]
    fn lbfgs_finds_quadratic_maximum_from_the_gradient_slot() {
        let result = optimize_det(
            NloptAlgorithm::Lbfgs,
            &[0.0, 0.0],
            &[(-10.0, 10.0), (-10.0, 10.0)],
            1e-8,
            500,
            quadratic_with_grad,
        )
        .unwrap();
        assert!(result.status.is_converged(), "status: {:?}", result.status);
        assert!((result.params[0] - 3.0).abs() < 1e-3);
        assert!((result.params[1] - (-1.0)).abs() < 1e-3);
        assert!(result.loglik > -1e-4);
    }

    /// The gradient slot really is `Some` for `Lbfgs` and `None` for the
    /// derivative-free algorithms — the fact the caller's "compute a gradient
    /// only when asked" branch rests on. Without it a derivative-free fit
    /// would pay for a gradient nothing reads, and a gradient fit would hand
    /// NLopt an unwritten slot.
    #[test]
    fn the_slot_is_some_only_for_the_gradient_algorithm() {
        for (algorithm, want_slot) in [
            (NloptAlgorithm::Sbplx, false),
            (NloptAlgorithm::Bobyqa, false),
            (NloptAlgorithm::Lbfgs, true),
        ] {
            let mut saw_slot: Option<bool> = None;
            let _ = optimize_det(
                algorithm,
                &[0.0, 0.0],
                &[(-10.0, 10.0), (-10.0, 10.0)],
                1e-8,
                5,
                |p: &[f64], grad: Option<&mut [f64]>| {
                    if saw_slot.is_none() {
                        saw_slot = Some(grad.is_some());
                    }
                    if let Some(g) = grad {
                        g[0] = -2.0 * (p[0] - 3.0);
                        g[1] = -2.0 * (p[1] + 1.0);
                    }
                    quadratic_value(p)
                },
            )
            .unwrap();
            assert_eq!(
                saw_slot,
                Some(want_slot),
                "{}: NLopt's gradient slot presence must match \
                 `uses_gradient()` = {}",
                algorithm.as_str(),
                algorithm.uses_gradient()
            );
        }
    }

    /// A gradient the caller could not compute reaches NLopt unsanitized, and
    /// NLopt ends the run: `OptStatus::Failed` at the last point, never a
    /// converged optimum. This is the contract `optimize_cell`'s latched
    /// reason rides on — the status says "this is not an answer" and the
    /// caller's message says why.
    ///
    /// The negative control is the same objective with a real gradient at the
    /// same start: it converges. Without it the assertion would pass for a
    /// run that failed for any other reason.
    #[test]
    fn a_non_finite_gradient_fails_the_run_rather_than_converging() {
        let bad = optimize_det(
            NloptAlgorithm::Lbfgs,
            &[9.0, 9.0],
            &[(0.0, 10.0), (0.0, 10.0)],
            1e-6,
            50,
            |p: &[f64], grad: Option<&mut [f64]>| {
                if let Some(g) = grad {
                    g.fill(f64::NAN);
                }
                // Finite value throughout: only the gradient is missing, so
                // nothing but the gradient can be what stopped the run.
                quadratic_value(p)
            },
        )
        .unwrap();
        assert_eq!(
            bad.status,
            OptStatus::Failed,
            "a non-finite gradient must end the run as Failed, not as a \
             converged optimum sitting where the gradient could not be taken; \
             got {:?} at {:?}",
            bad.status,
            bad.params
        );

        let good = optimize_det(
            NloptAlgorithm::Lbfgs,
            &[9.0, 9.0],
            &[(0.0, 10.0), (0.0, 10.0)],
            1e-6,
            50,
            quadratic_with_grad,
        )
        .unwrap();
        assert!(
            good.status.is_converged(),
            "negative control: the same start with a real gradient must converge"
        );
        assert!((good.params[0] - 3.0).abs() < 1e-3);
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
        assert!(
            optimize_det(
                NloptAlgorithm::Sbplx,
                &[0.0, 0.0],
                &[(-10.0, 10.0), (-10.0, 10.0)],
                1e-8,
                500,
                quadratic,
            )
            .is_ok()
        );
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
            |_: &[f64], _: Option<&mut [f64]>| 0.0,
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
            |p: &[f64], _grad: Option<&mut [f64]>| {
                if p[0] < 0.0 {
                    f64::NAN
                } else {
                    quadratic_value(p)
                }
            },
        )
        .unwrap();
        assert!(result.status.is_converged());
        assert!((result.params[0] - 3.0).abs() < 1e-3);
    }
}
