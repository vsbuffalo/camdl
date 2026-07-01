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
//! Validation is **per-axis**, not a single generic policy, because ORDER matters
//! differently per axis:
//!
//! - **Independent producer lists** (output times, the forward effect set) carry
//!   no parallel index data, so they may be freely SORTED into canonical order —
//!   [`SortedFiniteTimes`].
//! - **Index-aligned / ordered lists** (the inference effect timeline, whose
//!   `times[i]` aligns with `TimelineEffects::batches[i]`; and observation times,
//!   indexed by obs position) must NOT be sorted — reordering would desync the
//!   parallel `batches` (firing the wrong batch) or the obs index. These are
//!   validated **order-preserving**: a non-monotone input is REJECTED, never
//!   silently reordered.
//!
//! The wrappers live only at the construction boundary: the mode constructors on
//! [`crate::schedule::Schedule`] unwrap them to `Vec<f64>` immediately, so nothing
//! threads through the hot loop or the per-particle cursor.

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::intervention::TimelineEffects;

fn reject_non_finite(times: &[f64], axis: &str) -> Result<(), SimError> {
    if let Some(bad) = times.iter().copied().find(|t| !t.is_finite()) {
        return Err(SimError::Validation(format!(
            "{axis} time list contains a non-finite value ({bad}); all times must be finite"
        )));
    }
    Ok(())
}

fn reject_non_increasing(times: &[f64], axis: &str) -> Result<(), SimError> {
    for w in times.windows(2) {
        if !(w[0] < w[1]) {
            return Err(SimError::Validation(format!(
                "{axis} times must be strictly increasing; found {} followed by {}",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

fn reject_non_monotone(times: &[f64], axis: &str) -> Result<(), SimError> {
    for w in times.windows(2) {
        if !(w[0] <= w[1]) {
            return Err(SimError::Validation(format!(
                "{axis} times must be non-decreasing; found {} followed by {}",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

/// A finite, ascending list of times produced by SORTING an INDEPENDENT list (one
/// with no parallel index data). Use ONLY for such lists; never for the inference
/// effect timeline or observation times (see the module header).
#[derive(Clone, Debug)]
pub struct SortedFiniteTimes(Vec<f64>);

impl SortedFiniteTimes {
    pub fn new(mut times: Vec<f64>) -> Result<Self, SimError> {
        reject_non_finite(&times, "boundary")?;
        times.sort_by(f64::total_cmp);
        Ok(SortedFiniteTimes(times))
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

/// Trajectory-snapshot times (where a backend records a [`crate::state::Snapshot`]).
/// An independent list — sorted into canonical order.
#[derive(Clone, Debug)]
pub struct OutputTimes(Vec<f64>);

impl OutputTimes {
    /// Produce + validate + role-tag from the model's output schedule.
    pub fn from_model(model: &CompiledModel) -> Result<Self, SimError> {
        Ok(OutputTimes(
            SortedFiniteTimes::new(crate::output::output_times(
                &model.model.output.times,
                model.model.simulation.t_end,
            ))?
            .into_vec(),
        ))
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

/// Scheduled-effect boundary times.
#[derive(Clone, Debug)]
pub struct EffectTimes(Vec<f64>);

impl EffectTimes {
    /// The FORWARD effect set (interventions + always-active events) — an
    /// independent list with no parallel index, so it is sorted into canonical
    /// order. The forward backends fire via `round(t/dt)`, not by an index into
    /// this list.
    pub fn from_model(model: &CompiledModel, params: &[f64]) -> Result<Self, SimError> {
        Ok(EffectTimes(
            SortedFiniteTimes::new(crate::intervention::all_intervention_times(model, params))?
                .into_vec(),
        ))
    }

    /// The INFERENCE effect timeline: `timeline.times[i]` is the boundary the
    /// producer fires `timeline.batches[i]` at — so the order is index-aligned
    /// with the batches and must be PRESERVED. Validate finite + non-decreasing
    /// (the producer's invariant); never sort (that would desync the batches).
    pub fn from_timeline(timeline: &TimelineEffects) -> Result<Self, SimError> {
        reject_non_finite(&timeline.times, "effect")?;
        reject_non_monotone(&timeline.times, "effect")?;
        Ok(EffectTimes(timeline.times.clone()))
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

/// Observation boundary times (where the inference filters score the likelihood
/// and reset the accumulator). Indexed by observation position, so the order is
/// PRESERVED: a non-strictly-increasing input is rejected, never reordered
/// (sorting would break the obs-index alignment, and coincident obs are an
/// upstream error the loader already rejects).
#[derive(Clone, Debug)]
pub struct ObsTimes(Vec<f64>);

impl ObsTimes {
    pub fn new(times: Vec<f64>) -> Result<Self, SimError> {
        reject_non_finite(&times, "observation")?;
        reject_non_increasing(&times, "observation")?;
        Ok(ObsTimes(times))
    }

    /// The last (largest) observation time, if any — the natural `t_end` for an
    /// inference window. `None` for an empty list.
    pub(crate) fn last(&self) -> Option<f64> {
        self.0.last().copied()
    }

    pub(crate) fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_finite_rejects_non_finite_and_sorts() {
        assert!(SortedFiniteTimes::new(vec![1.0, f64::NAN, 2.0]).is_err(), "NaN rejected");
        assert!(SortedFiniteTimes::new(vec![1.0, f64::INFINITY]).is_err(), "inf rejected");
        // Sorts (independent list); does NOT dedup.
        let s = SortedFiniteTimes::new(vec![3.0, 1.0, 2.0, 2.0]).unwrap();
        assert_eq!(s.into_vec(), vec![1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn obs_times_preserve_order_and_reject_non_increasing() {
        // Order preserved (already increasing).
        assert_eq!(ObsTimes::new(vec![1.0, 2.5, 7.3]).unwrap().into_vec(), vec![1.0, 2.5, 7.3]);
        // Out-of-order is REJECTED, not silently sorted (that would desync the
        // obs index).
        assert!(ObsTimes::new(vec![3.0, 1.0, 2.0]).is_err(), "non-increasing rejected, not sorted");
        // Coincident obs rejected (strictly increasing); non-finite rejected.
        assert!(ObsTimes::new(vec![1.0, 1.0]).is_err(), "duplicate obs rejected");
        assert!(ObsTimes::new(vec![f64::NAN]).is_err());
        // Empty / single are fine.
        assert!(ObsTimes::new(vec![]).is_ok());
        assert!(ObsTimes::new(vec![4.0]).is_ok());
    }

    #[test]
    fn effect_from_timeline_preserves_order_aligned_with_batches() {
        let tl = TimelineEffects { times: vec![2.0, 5.0, 5.0], batches: vec![vec![0], vec![1], vec![2]] };
        // Non-decreasing (coincident effect boundaries allowed) + preserved.
        assert_eq!(EffectTimes::from_timeline(&tl).unwrap().into_vec(), vec![2.0, 5.0, 5.0]);
        // A non-monotone timeline is rejected (would desync batches if sorted).
        let bad = TimelineEffects { times: vec![5.0, 2.0], batches: vec![vec![0], vec![1]] };
        assert!(EffectTimes::from_timeline(&bad).is_err(), "non-monotone timeline rejected, not sorted");
    }
}
