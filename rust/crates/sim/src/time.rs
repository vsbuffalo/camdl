//! Time-axis validation and substep counting.
//!
//! An effect's firing step is not computed here: it is
//! [`crate::effect_plan::fire_step`], on the run's own grid from `t_start` at
//! the run's `dt` — one authority for every path. (gh#53 was a fire step baked
//! at compile time from `model.simulation.dt` while the integrator ran another
//! `dt`; the plan is built per run from the dt-invariant fire times.)

use crate::SimError;

/// Validate an integrator step `dt` BEFORE it is used to map times to
/// step indices or to drive a substep loop. Returns a named
/// [`SimError::Validation`] for a non-finite or non-positive `dt`.
///
/// This is the RELEASE-build guard for gh#126: a bad (or
/// parameter-proposed) `dt` of `0`, a negative, or `NaN`/`±∞` would
/// otherwise put the integrator in an infinite loop (`dt <= 0` never
/// advances time) or silently corrupt every step index (`NaN as i64 ==
/// 0`). The `debug_assert!`s in [`interval_steps`] catch the same thing in
/// dev builds, but they are
/// compiled out of `--release`; this check runs unconditionally and is
/// called once at each backend's entry point so the failure is a
/// controlled setup error, not a stalled worker or a silent wrong
/// answer. See `docs/dev/notes/2026-06-08-static-typing-as-bug-prevention.md`
/// §6 (parse-don't-validate: relocate the check from a panic/debug-only
/// assert to an always-on validation-time error at the boundary).
pub fn validate_dt(dt: f64) -> Result<(), SimError> {
    if !dt.is_finite() {
        return Err(SimError::Validation(format!(
            "integrator step dt must be finite, got {dt}"
        )));
    }
    if dt <= 0.0 {
        return Err(SimError::Validation(format!(
            "integrator step dt must be positive, got {dt}"
        )));
    }
    Ok(())
}

/// Validate a `Regular` output schedule's step BEFORE it drives the
/// `output_times` enumeration loop. Returns a named
/// [`SimError::Validation`] for a non-finite or non-positive step.
///
/// gh#257: `output_times` enumerates `start, start+step, …` with a
/// `while t <= t_end { t += step }` loop. A `step` of `0` (or a
/// negative, or `NaN`/`±∞`) never advances `t` past the bound, so the
/// loop runs forever and the worker stalls with a half-written
/// trajectory. Mirrors [`validate_dt`]: an always-on release check at
/// the schedule boundary, so a bad (or parameter-proposed) step is a
/// controlled setup error, not a hung process.
pub fn validate_output_step(step: f64) -> Result<(), SimError> {
    if !step.is_finite() {
        return Err(SimError::Validation(format!(
            "output schedule step must be finite, got {step}"
        )));
    }
    if step <= 0.0 {
        return Err(SimError::Validation(format!(
            "output schedule step must be positive, got {step}; \
             an every-{step} cadence never advances the output cursor"
        )));
    }
    Ok(())
}

/// Validate a `Recurring` intervention schedule's period BEFORE it
/// drives the fire-time enumeration loop. Returns a named
/// [`SimError::Validation`] for a non-finite or non-positive period.
///
/// gh#257: `intervention_fire_times` enumerates
/// `start, start+period, …` with a `while t <= end { t += period }`
/// loop. A `period` of `0` (or a negative, or `NaN`/`±∞`) never
/// advances `t` past `end`, so the loop runs forever. Mirrors
/// [`validate_dt`]: an always-on release check at the schedule
/// boundary so a bad (or parameter-proposed) period is a controlled
/// setup error, not a hung process.
pub fn validate_recurrence_period(period: f64) -> Result<(), SimError> {
    if !period.is_finite() {
        return Err(SimError::Validation(format!(
            "recurring schedule period must be finite, got {period}"
        )));
    }
    if period <= 0.0 {
        return Err(SimError::Validation(format!(
            "recurring schedule period must be positive, got {period}; \
             a period of {period} never advances to the next fire time"
        )));
    }
    Ok(())
}

/// Validate a list of scheduled fire times: every time must be finite.
/// A non-finite fire time (`NaN`/`±∞`, e.g. from a parametric `at [...]`
/// schedule expression that went through zero) would map to a garbage
/// step index (`NaN as i64 == 0`) and silently fire an intervention at
/// the wrong step. gh#126: reject it at schedule resolution with a named
/// error instead.
pub fn validate_fire_times(times: &[f64]) -> Result<(), SimError> {
    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(SimError::Validation(format!(
                "scheduled fire time #{i} must be finite, got {t}"
            )));
        }
    }
    Ok(())
}

/// Number of `dt`-sized substeps spanning the interval `[t0, t1]`.
/// This is a *count* (relative substep span), not a step index; an
/// effect's firing step is [`crate::effect_plan::fire_step`]. Used for
/// substep-loop bounds and observation-spacing arithmetic in the
/// PGAS / PMMH inner loops.
///
/// `t1 ≥ t0` and `dt > 0` are caller responsibilities (debug_asserts).
/// Returns `usize` because every consumer wants a loop bound.
#[inline]
pub fn interval_steps(t0: f64, t1: f64, dt: f64) -> usize {
    debug_assert!(t1.is_finite() && t0.is_finite(),
        "interval_steps: non-finite t (t0={}, t1={})", t0, t1);
    debug_assert!(dt > 0.0, "interval_steps: non-positive dt = {}", dt);
    debug_assert!(t1 >= t0,
        "interval_steps: t1 ({}) < t0 ({})", t1, t0);
    ((t1 - t0) / dt).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_steps_basic() {
        assert_eq!(interval_steps(0.0, 7.0, 1.0), 7);
        assert_eq!(interval_steps(0.0, 7.0, 0.5), 14);
        assert_eq!(interval_steps(0.0, 7.0, 0.25), 28);
        // t_start ≠ 0 — relative count, not absolute index.
        assert_eq!(interval_steps(100.0, 107.0, 1.0), 7);
        assert_eq!(interval_steps(100.0, 107.0, 0.5), 14);
    }

    #[test]
    fn interval_steps_at_zero_t0_t1_returns_zero() {
        assert_eq!(interval_steps(5.0, 5.0, 1.0), 0);
    }

    #[test]
    fn interval_steps_rounds_to_nearest() {
        // Rounds to the nearest whole count.
        assert_eq!(interval_steps(0.0, 7.4, 1.0), 7);
        assert_eq!(interval_steps(0.0, 7.6, 1.0), 8);
    }

    #[test]
    fn interval_steps_panics_on_inverted_interval() {
        let result = std::panic::catch_unwind(|| interval_steps(7.0, 0.0, 1.0));
        assert!(result.is_err());
    }

    // ── validate_dt / validate_fire_times (gh#126) ──────────────────
    //
    // These checks must fire in RELEASE builds too — a bad (or
    // parameter-proposed) dt/schedule must be REJECTED with a named
    // error at construction, not pass silently because the only guard
    // was a `debug_assert!` compiled out of `--release`. We assert the
    // rejection unconditionally (no `cfg!(debug_assertions)` gate), so a
    // regression to debug-only would fail this test in `cargo test
    // --release`.

    #[test]
    fn validate_dt_rejects_nonpositive_in_release() {
        // dt = 0 would put the integrator in an infinite loop; dt < 0 is
        // nonsense. Both must be a named Validation error, regardless of
        // build profile.
        let err = validate_dt(0.0).expect_err("dt = 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
        assert!(format!("{err}").contains("dt"), "error must name dt: {err}");

        let err = validate_dt(-0.5).expect_err("dt < 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
    }

    #[test]
    fn validate_dt_rejects_non_finite_in_release() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = validate_dt(bad).expect_err("non-finite dt must be rejected");
            assert!(matches!(err, crate::SimError::Validation(_)),
                "expected Validation for dt={bad}, got {err:?}");
            assert!(format!("{err}").contains("dt"), "error must name dt: {err}");
        }
    }

    #[test]
    fn validate_dt_accepts_positive_finite() {
        assert!(validate_dt(1.0).is_ok());
        assert!(validate_dt(0.5).is_ok());
        assert!(validate_dt(1e-6).is_ok());
    }

    #[test]
    fn validate_fire_times_rejects_non_finite_in_release() {
        let err = validate_fire_times(&[1.0, f64::NAN, 3.0])
            .expect_err("non-finite fire time must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");

        let err = validate_fire_times(&[f64::INFINITY])
            .expect_err("infinite fire time must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
    }

    #[test]
    fn validate_fire_times_accepts_finite() {
        assert!(validate_fire_times(&[0.0, 1.0, 258.0, 988.5]).is_ok());
    }

    // gh#257: the same infinite-loop hazard as `dt <= 0`, at two other
    // schedule-driving loops — `output_times` (`t += step`) and the
    // `Recurring` fire-time enumeration (`t += period`). A non-positive
    // output step or recurrence period must be rejected with a named
    // Validation error at the boundary, not silently loop forever.

    #[test]
    fn validate_output_step_rejects_nonpositive() {
        let err = validate_output_step(0.0).expect_err("output step = 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
        assert!(format!("{err}").contains("step"), "error must name the step: {err}");

        let err = validate_output_step(-1.0).expect_err("output step < 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
    }

    #[test]
    fn validate_output_step_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = validate_output_step(bad).expect_err("non-finite output step must be rejected");
            assert!(matches!(err, crate::SimError::Validation(_)),
                "expected Validation for step={bad}, got {err:?}");
        }
    }

    #[test]
    fn validate_output_step_accepts_positive_finite() {
        assert!(validate_output_step(1.0).is_ok());
        assert!(validate_output_step(0.5).is_ok());
        assert!(validate_output_step(1e-6).is_ok());
    }

    #[test]
    fn validate_recurrence_period_rejects_nonpositive() {
        let err = validate_recurrence_period(0.0).expect_err("period = 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
        assert!(format!("{err}").contains("period"), "error must name the period: {err}");

        let err = validate_recurrence_period(-7.0).expect_err("period < 0 must be rejected");
        assert!(matches!(err, crate::SimError::Validation(_)), "expected Validation, got {err:?}");
    }

    #[test]
    fn validate_recurrence_period_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = validate_recurrence_period(bad).expect_err("non-finite period must be rejected");
            assert!(matches!(err, crate::SimError::Validation(_)),
                "expected Validation for period={bad}, got {err:?}");
        }
    }

    #[test]
    fn validate_recurrence_period_accepts_positive_finite() {
        assert!(validate_recurrence_period(7.0).is_ok());
        assert!(validate_recurrence_period(1.0).is_ok());
    }

}
