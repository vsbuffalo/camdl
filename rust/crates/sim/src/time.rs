//! Time-to-step conversion — the single entrypoint for mapping
//! continuous time to integer step indices.
//!
//! This module exists because the alternative — inlining
//! `(t / dt).round() as i64` at every site that needs the conversion
//! — produced gh#53: `compiled_model.rs:561` baked fire-step indices
//! at compile time using `model.simulation.dt`, but the runtime
//! integrator's dt could differ (every `camdl pfilter --dt 0.5` run on
//! a model declared at `dt = 1.0`). The result was a sub-day-step bug
//! invisible to synth-recovery and single-dt benchmarks but visible
//! against pomp on He et al. 2010 measles (5862 nat divergence at the
//! literature MLE; gh#52 Richardson ladder caught it).
//!
//! Funnel every continuous-time → step-index conversion through
//! [`time_to_step`]. The conversion is trivial; consolidating it
//! gives one place to invariant-test, one place to fix if the
//! semantics ever change, and one place agents and reviewers know to
//! audit.

use crate::SimError;

/// Validate an integrator step `dt` BEFORE it is used to map times to
/// step indices or to drive a substep loop. Returns a named
/// [`SimError::Validation`] for a non-finite or non-positive `dt`.
///
/// This is the RELEASE-build guard for gh#126: a bad (or
/// parameter-proposed) `dt` of `0`, a negative, or `NaN`/`±∞` would
/// otherwise put the integrator in an infinite loop (`dt <= 0` never
/// advances time) or silently corrupt every step index (`NaN as i64 ==
/// 0`). The per-conversion `debug_assert!`s in [`time_to_step`] /
/// [`interval_steps`] catch the same thing in dev builds, but they are
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
/// error instead. This is the RELEASE-build sibling of the
/// `t.is_finite()` `debug_assert!` in [`time_to_step`].
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

/// Map continuous time `t` (in the model's `time_unit`, typically days
/// or years) to the integer step index for an integrator running at
/// step size `dt` (same unit). Rounds to the nearest step — interventions
/// fire in whichever step contains them.
///
/// Non-finite `t` or non-positive `dt` are caller bugs that must be
/// rejected at the boundary with [`validate_dt`] / [`validate_fire_times`]
/// (which run in release); the `debug_assert!`s here are a dev-build
/// backstop, not the load-bearing guard (gh#126).
#[inline]
pub fn time_to_step(t: f64, dt: f64) -> i64 {
    debug_assert!(t.is_finite(), "time_to_step: non-finite t = {}", t);
    debug_assert!(dt > 0.0, "time_to_step: non-positive dt = {}", dt);
    (t / dt).round() as i64
}

/// Map a list of continuous fire times to a sorted, deduplicated
/// `BTreeSet` of step indices for integrator step `dt`. Used by
/// [`crate::compiled_model::CompiledModel::resolve_fire_steps`] to
/// derive the runtime view of a (compile-time, dt-invariant) fire-time
/// schedule.
pub fn fire_times_to_steps(times: &[f64], dt: f64) -> std::collections::BTreeSet<i64> {
    times.iter().map(|&t| time_to_step(t, dt)).collect()
}

/// Number of `dt`-sized substeps spanning the interval `[t0, t1]`.
/// Distinct operation from [`time_to_step`]: this is a *count*
/// (relative substep span), not an absolute step index. Used for
/// substep-loop bounds and observation-spacing arithmetic in the
/// PGAS / PMMH / correlated-PF inner loops.
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
    fn time_to_step_at_dt_1_is_identity_on_integers() {
        assert_eq!(time_to_step(0.0, 1.0), 0);
        assert_eq!(time_to_step(1.0, 1.0), 1);
        assert_eq!(time_to_step(258.0, 1.0), 258);
    }

    #[test]
    fn time_to_step_at_sub_day_dt_scales_correctly() {
        // The bug-fingerprint test: at dt=0.5, day 258 should map to
        // step 516, NOT step 258 (the gh#53 bug). The compile-time
        // pre-baked fire_steps had 258 as a step index and the
        // runtime walked at dt=0.5, so the impulse fired at wall
        // time 129 (= 258 * 0.5). With this helper used uniformly,
        // that confusion can't happen.
        assert_eq!(time_to_step(258.0, 0.5), 516);
        assert_eq!(time_to_step(258.0, 0.25), 1032);
        assert_eq!(time_to_step(258.0, 0.125), 2064);
    }

    #[test]
    fn time_to_step_rounds_to_nearest() {
        // 0.5*dt below a step boundary rounds up; at-or-above a step
        // boundary stays. The choice (round vs floor) matches pomp's
        // convention (`fabs(t - target) < 0.5*dt`) where the firing
        // step is the one that *contains* the target.
        assert_eq!(time_to_step(7.4, 1.0), 7);
        assert_eq!(time_to_step(7.5, 1.0), 8);  // banker's rounding ties to even? rust f64::round → 8
        assert_eq!(time_to_step(7.6, 1.0), 8);
    }

    #[test]
    fn time_to_step_at_zero_dt_panics_in_debug() {
        // Defensive: dt = 0 would put the integrator in an infinite
        // loop. Catching at the conversion point gives a clearer
        // panic site than a stuck simulator.
        let result = std::panic::catch_unwind(|| time_to_step(1.0, 0.0));
        assert!(result.is_err());
    }

    #[test]
    fn time_to_step_at_negative_dt_panics_in_debug() {
        let result = std::panic::catch_unwind(|| time_to_step(1.0, -0.5));
        assert!(result.is_err());
    }

    #[test]
    fn time_to_step_at_nan_t_panics_in_debug() {
        // Rm4 in 2026-04-19 engine review: NaN as i64 is 0 on
        // current rustc, which would silently match step 0 in any
        // fire-step-checking code. The debug_assert catches it.
        let result = std::panic::catch_unwind(|| time_to_step(f64::NAN, 1.0));
        assert!(result.is_err());
    }

    // ── fire_times_to_steps ─────────────────────────────────────────

    #[test]
    fn fire_times_to_steps_resolves_periodic_schedule_dt_invariantly() {
        // Cohort entry at day 258 every 365.25 days. Three fires:
        // days 258, 623.25, 988.5. At any dt that divides into 258
        // cleanly, all three fires should map to distinct step
        // indices.
        let fire_times = vec![258.0, 623.25, 988.5];

        let steps_dt1   = fire_times_to_steps(&fire_times, 1.0);
        let steps_dt05  = fire_times_to_steps(&fire_times, 0.5);
        let steps_dt025 = fire_times_to_steps(&fire_times, 0.25);

        // Each ladder produces the same number of distinct fires —
        // exactly 3 — regardless of dt. (The gh#53 bug was that
        // these would alias incorrectly under finer dt.)
        assert_eq!(steps_dt1.len(), 3);
        assert_eq!(steps_dt05.len(), 3);
        assert_eq!(steps_dt025.len(), 3);

        // The wall times recovered from the steps must equal the
        // input times (within rounding) at every dt.
        for (steps, dt) in [(&steps_dt1, 1.0), (&steps_dt05, 0.5), (&steps_dt025, 0.25)] {
            let recovered: Vec<f64> = steps.iter().map(|&s| s as f64 * dt).collect();
            for (orig, rec) in fire_times.iter().zip(&recovered) {
                assert!((orig - rec).abs() <= 0.5 * dt,
                    "dt={}: fire time {} → step → wall {} drifted >0.5*dt",
                    dt, orig, rec);
            }
        }
    }

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
        // Same banker's-rounding semantics as time_to_step.
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

    #[test]
    fn fire_times_to_steps_dedups_collisions() {
        // Two fire times that round to the same step at coarse dt
        // collapse to a single entry in the BTreeSet. Documented
        // behaviour: the set semantics inherently dedup. No fire
        // is "lost" because BTreeSet membership is what the runtime
        // checks, not a count — one fire per step.
        let fire_times = vec![100.0, 100.3];  // both round to step 100 at dt=1
        let steps = fire_times_to_steps(&fire_times, 1.0);
        assert_eq!(steps.len(), 1);
        assert!(steps.contains(&100));
    }
}
