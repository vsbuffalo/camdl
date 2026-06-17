//! Role-typed, validated boundary-time lists for schedule construction (gh#233
//! Layer 2.5).
//!
//! The three boundary axes are all `Vec<f64>` but mean different things — output
//! (record a snapshot), effect (fire an intervention/event), observation (score
//! the likelihood + reset the accumulator). Passing the wrong one to a `Schedule`
//! constructor compiles and silently produces a plausible-but-wrong trajectory;
//! wrapping each axis in its own type makes that swap a COMPILE error. Named
//! constructor parameters do not help here — Rust has no argument labels.
//!
//! All three share one validated substrate, [`SortedFiniteTimes`] (finite +
//! ascending), upgrading `Schedule::new`'s former `debug_assert!`-only sort check
//! to a real one. Deduplication is deliberately NOT baked in: it is role-specific
//! (coincident effects fire as a batch; coincident observations are rejected
//! upstream by the loader's collision check), so it stays with each axis's own
//! semantics rather than a single generic policy.
//!
//! The wrappers live only at the construction boundary: the mode constructors on
//! [`crate::schedule::Schedule`] unwrap them to `Vec<f64>` immediately, so nothing
//! threads through the hot loop or the per-particle cursor.

use crate::compiled_model::CompiledModel;
use crate::error::SimError;

/// A finite, ascending list of times — the shared substrate for the role
/// wrappers below. Construction rejects NaN / ±∞ and sorts by total order; it
/// does NOT deduplicate (that is role-specific).
#[derive(Clone, Debug)]
pub struct SortedFiniteTimes(Vec<f64>);

impl SortedFiniteTimes {
    /// Validate (finite) and sort. The only way to build one — so holding a
    /// `SortedFiniteTimes` is proof the invariant holds.
    pub fn new(mut times: Vec<f64>) -> Result<Self, SimError> {
        if let Some(bad) = times.iter().copied().find(|t| !t.is_finite()) {
            return Err(SimError::Validation(format!(
                "boundary time list contains a non-finite value ({bad}); \
                 all boundary times must be finite"
            )));
        }
        times.sort_by(f64::total_cmp);
        Ok(SortedFiniteTimes(times))
    }

    /// Consume into the raw sorted vector — for the schedule constructors only.
    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

/// Trajectory-snapshot times (where a backend records a [`crate::state::Snapshot`]).
#[derive(Clone, Debug)]
pub struct OutputTimes(SortedFiniteTimes);

impl OutputTimes {
    /// Produce + validate + role-tag in one place, from the model's output schedule.
    pub fn from_model(model: &CompiledModel) -> Result<Self, SimError> {
        Ok(OutputTimes(SortedFiniteTimes::new(crate::output::output_times(
            &model.model.output.times,
        ))?))
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0.into_vec()
    }
}

/// Scheduled-effect boundary times (interventions + always-active events).
#[derive(Clone, Debug)]
pub struct EffectTimes(SortedFiniteTimes);

impl EffectTimes {
    /// Produce + validate + role-tag from the model's interventions/events.
    pub fn from_model(model: &CompiledModel, params: &[f64]) -> Result<Self, SimError> {
        Ok(EffectTimes(SortedFiniteTimes::new(
            crate::intervention::all_intervention_times(model, params),
        )?))
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0.into_vec()
    }
}
