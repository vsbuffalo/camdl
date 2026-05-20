//! Offline line-list projections beyond the transmission tree (Phase 3).
//!
//! Like [`super::tree`], these are **pure functions over a parsed line list** —
//! no model, no simulation re-run. The line list speaks compartment / transition
//! *ids* (not names), so these projections take ids; the CLI documents the id
//! convention (the global compartment index, matching `camdl simulate` column
//! order). Keeping them model-free is what makes them cacheable, re-runnable,
//! and independently testable.
//!
//! Two projections:
//!   - [`sojourn`] — dwell-time distribution in a compartment: for each tracked
//!     individual, `(time it left the compartment) − (time it entered)`.
//!   - [`cohort`] — per-time-window event summary: new lineage events
//!     (infections) binned into fixed windows, as incidence + cumulative.

use std::collections::HashMap;

use super::writer::LineListEntry;
use super::{CompartmentId, ParentRef};

/// One completed sojourn: an individual entered `compartment` at `entry` and
/// left it at `exit`. `dwell = exit − entry`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sojourn {
    pub individual: u64,
    pub entry: f64,
    pub exit: f64,
    pub dwell: f64,
}

/// Result of the sojourn projection: completed sojourns plus the count of
/// right-censored individuals (entered the compartment but never left within
/// the simulated horizon).
#[derive(Debug, Clone)]
pub struct SojournResult {
    pub compartment: CompartmentId,
    pub completed: Vec<Sojourn>,
    /// Individuals that entered `compartment` but had no exit event in the line
    /// list (still resident at the end of the run). Excluded from `completed`
    /// because their dwell time is unknown (right-censored).
    pub censored: usize,
}

impl SojournResult {
    /// Mean completed dwell time (NaN if no completed sojourns).
    pub fn mean_dwell(&self) -> f64 {
        if self.completed.is_empty() {
            f64::NAN
        } else {
            self.completed.iter().map(|s| s.dwell).sum::<f64>() / self.completed.len() as f64
        }
    }

    /// Quantile of the completed dwell distribution (`q` in [0, 1]), via the
    /// nearest-rank method on the sorted dwell times. NaN if empty.
    pub fn dwell_quantile(&self, q: f64) -> f64 {
        if self.completed.is_empty() {
            return f64::NAN;
        }
        let mut d: Vec<f64> = self.completed.iter().map(|s| s.dwell).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = q.clamp(0.0, 1.0);
        // Nearest-rank: index = ceil(q * n) - 1, clamped.
        let n = d.len();
        let rank = ((q * n as f64).ceil() as usize).max(1);
        d[(rank - 1).min(n - 1)]
    }
}

/// Compute the dwell-time distribution for `compartment`.
///
/// An individual *enters* `compartment` at the time of the first line-list
/// event in which it is the focal individual with `destination == compartment`;
/// it *leaves* at the first subsequent event in which it is focal with
/// `source == compartment`. Each (entry, exit) pair is one completed sojourn.
/// An individual with an entry but no exit is right-censored (counted, not
/// timed). Entries are processed in time order so an individual that cycles
/// through the compartment more than once contributes one sojourn per visit.
///
/// Note: an individual seeded into `compartment` at t=0 has no entry event in
/// the line list (seeds appear only as parents / sources later), so its first
/// observed *exit* has no matching entry and is ignored — its initial sojourn
/// is left-censored and not reported. This matches the dwell-time semantics
/// (we report fully-observed sojourns only).
pub fn sojourn(entries: &[LineListEntry], compartment: CompartmentId) -> SojournResult {
    // Process events in time order. Ties broken by the entry's position in the
    // file (stable sort) so a same-time enter-then-leave is handled in record
    // order.
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| entries[a].time.partial_cmp(&entries[b].time).unwrap());

    // Open entry time per individual currently resident in `compartment`.
    let mut open: HashMap<u64, f64> = HashMap::new();
    let mut completed: Vec<Sojourn> = Vec::new();

    for &i in &order {
        let e = &entries[i];
        let ind = e.individual.0;
        // A move out of the compartment closes an open sojourn.
        if e.source == Some(compartment) {
            if let Some(entry) = open.remove(&ind) {
                completed.push(Sojourn {
                    individual: ind,
                    entry,
                    exit: e.time,
                    dwell: (e.time - entry).max(0.0),
                });
            }
        }
        // A move into the compartment opens a sojourn (if not already open —
        // the same focal event can both leave a source and enter a destination,
        // and an individual is in exactly one compartment at a time).
        if e.destination == Some(compartment) {
            open.entry(ind).or_insert(e.time);
        }
    }

    SojournResult {
        compartment,
        completed,
        censored: open.len(),
    }
}

/// Which events the cohort projection counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortEvent {
    /// New transmission (lineage) events — `parent_kind == individual`. Model-
    /// independent: identifiable purely from the line list. This is the
    /// proposal's `--event infection`.
    Infection,
    /// Events of a specific transition id.
    Transition(usize),
}

/// One time window of the cohort summary.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortBin {
    /// Window start (inclusive).
    pub start: f64,
    /// Window end (exclusive).
    pub end: f64,
    /// New events in `[start, end)` — incidence.
    pub incidence: u64,
    /// Cumulative events up to and including this window.
    pub cumulative: u64,
}

/// Bin events into fixed-width time windows of width `window`, starting at the
/// earliest matching event time (or 0 if `align_zero`). Returns one [`CohortBin`]
/// per window from the first to the last matching event, with running
/// cumulative counts. An empty / no-match line list yields an empty vector.
pub fn cohort(entries: &[LineListEntry], event: CohortEvent, window: f64, align_zero: bool) -> Vec<CohortBin> {
    assert!(window > 0.0, "cohort window must be positive");

    let matches = |e: &LineListEntry| match event {
        CohortEvent::Infection => matches!(e.parent, ParentRef::Individual(_)),
        CohortEvent::Transition(t) => e.transition == t,
    };

    let times: Vec<f64> = entries.iter().filter(|e| matches(e)).map(|e| e.time).collect();
    if times.is_empty() {
        return Vec::new();
    }
    let min_t = if align_zero {
        0.0
    } else {
        times.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let max_t = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Number of windows needed to cover [min_t, max_t]. The last event must
    // land in the final window: index = floor((t - min_t)/window).
    let n_windows = ((max_t - min_t) / window).floor() as usize + 1;
    let mut counts = vec![0u64; n_windows];
    for &t in &times {
        let idx = (((t - min_t) / window).floor() as usize).min(n_windows - 1);
        counts[idx] += 1;
    }

    let mut bins = Vec::with_capacity(n_windows);
    let mut cumulative = 0u64;
    for (k, &c) in counts.iter().enumerate() {
        cumulative += c;
        let start = min_t + k as f64 * window;
        bins.push(CohortBin {
            start,
            end: start + window,
            incidence: c,
            cumulative,
        });
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::IndividualId;

    fn entry(t: f64, ind: u64, src: Option<usize>, dst: Option<usize>, lineage_parent: Option<u64>) -> LineListEntry {
        LineListEntry {
            time: t,
            transition: 0,
            individual: IndividualId(ind),
            source: src,
            destination: dst,
            deme: 0,
            parent: match lineage_parent {
                Some(p) => ParentRef::Individual(IndividualId(p)),
                None => ParentRef::None,
            },
            parent_deme: lineage_parent.map(|_| 0),
        }
    }

    #[test]
    fn sojourn_single_visit() {
        // Individual 1 enters compartment 1 at t=1 (S->I), leaves at t=4 (I->R).
        let entries = vec![
            entry(1.0, 1, Some(0), Some(1), Some(0)),
            entry(4.0, 1, Some(1), Some(2), None),
        ];
        let r = sojourn(&entries, 1);
        assert_eq!(r.completed.len(), 1);
        assert_eq!(r.censored, 0);
        assert_eq!(r.completed[0].dwell, 3.0);
        assert_eq!(r.mean_dwell(), 3.0);
    }

    #[test]
    fn sojourn_censored_individual_counted_not_timed() {
        // Enters compartment 1 at t=2 but never leaves.
        let entries = vec![entry(2.0, 5, Some(0), Some(1), Some(0))];
        let r = sojourn(&entries, 1);
        assert_eq!(r.completed.len(), 0);
        assert_eq!(r.censored, 1);
        assert!(r.mean_dwell().is_nan());
    }

    #[test]
    fn sojourn_quantile_nearest_rank() {
        // Dwell times 1,2,3,4 → median (q=0.5) nearest-rank index ceil(2)-1=1 → 2.
        let entries = vec![
            entry(0.0, 1, None, Some(1), None),
            entry(1.0, 1, Some(1), Some(2), None),
            entry(0.0, 2, None, Some(1), None),
            entry(2.0, 2, Some(1), Some(2), None),
            entry(0.0, 3, None, Some(1), None),
            entry(3.0, 3, Some(1), Some(2), None),
            entry(0.0, 4, None, Some(1), None),
            entry(4.0, 4, Some(1), Some(2), None),
        ];
        let r = sojourn(&entries, 1);
        assert_eq!(r.completed.len(), 4);
        assert_eq!(r.dwell_quantile(0.5), 2.0);
        assert_eq!(r.dwell_quantile(1.0), 4.0);
        assert_eq!(r.dwell_quantile(0.0), 1.0);
    }

    #[test]
    fn cohort_infection_windows() {
        // Lineage events at t = 0.5, 1.5, 1.9, 3.2. Window 1.0, aligned to zero.
        let entries = vec![
            entry(0.5, 1, Some(0), Some(1), Some(0)),
            entry(1.5, 2, Some(0), Some(1), Some(1)),
            entry(1.9, 3, Some(0), Some(1), Some(1)),
            entry(3.2, 4, Some(0), Some(1), Some(2)),
        ];
        let bins = cohort(&entries, CohortEvent::Infection, 1.0, true);
        // Windows [0,1): 1, [1,2): 2, [2,3): 0, [3,4): 1.
        assert_eq!(bins.len(), 4);
        assert_eq!(bins[0].incidence, 1);
        assert_eq!(bins[1].incidence, 2);
        assert_eq!(bins[2].incidence, 0);
        assert_eq!(bins[3].incidence, 1);
        assert_eq!(bins[3].cumulative, 4);
    }

    #[test]
    fn cohort_empty_when_no_matches() {
        let entries = vec![entry(1.0, 1, Some(1), Some(2), None)];
        // No lineage (infection) events.
        let bins = cohort(&entries, CohortEvent::Infection, 1.0, true);
        assert!(bins.is_empty());
    }

    #[test]
    fn cohort_transition_id_filter() {
        let mut a = entry(0.5, 1, Some(0), Some(1), Some(0));
        a.transition = 7;
        let b = entry(0.7, 2, Some(1), Some(2), None); // transition 0
        let bins = cohort(&[a, b], CohortEvent::Transition(7), 1.0, true);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].incidence, 1);
    }
}
