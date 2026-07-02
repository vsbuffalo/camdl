use ir::model::{OutputSchedule, RegularOutputSchedule};

/// Convert an `OutputSchedule` to a sorted, deduplicated list of output times,
/// confined to the horizon `[start, t_end]`.
///
/// `t_end` is the sole horizon authority (`simulation.t_end`, gh#143): the
/// output schedule no longer carries its own end. A `Regular` schedule
/// enumerates `start, start+step, …` up to and including `t_end` (already
/// ascending and unique by construction); an `AtTimes` list is filtered to
/// entries `≤ t_end`, so an explicitly-listed time beyond the dynamics horizon
/// is dropped, not emitted against a frozen state.
///
/// gh#257: an author-supplied `at = [...]` list may arrive out of order or with
/// repeats. The `AtTimes` branch is sorted and deduplicated here so the result
/// honours the "sorted list" contract and a repeated time does not emit a
/// duplicate snapshot row. Non-finite entries are not this function's concern —
/// they are rejected at the `OutputTimes`/`SortedFiniteTimes` boundary the
/// backends build their schedule through (`NaN`/`+∞` never satisfy `≤ t_end`
/// and drop out here; a surviving `-∞` is rejected there).
pub fn output_times(sched: &OutputSchedule, t_end: f64) -> Vec<f64> {
    match sched {
        OutputSchedule::Regular(RegularOutputSchedule { start, step }) => {
            let mut times = Vec::new();
            let mut t = *start;
            while t <= t_end + step * 1e-9 {
                times.push(t);
                t += step;
            }
            times
        }
        OutputSchedule::AtTimes(ts) => {
            let mut times: Vec<f64> = ts.iter().copied().filter(|&t| t <= t_end).collect();
            times.sort_by(f64::total_cmp);
            times.dedup();
            times
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(start: f64, step: f64) -> OutputSchedule {
        OutputSchedule::Regular(RegularOutputSchedule { start, step })
    }

    /// gh#143 upward: `simulation.t_end` is the sole horizon authority, so a
    /// Regular schedule enumerates all the way to `t_end` — the schedule no
    /// longer carries an `end` that could cap emission short.
    #[test]
    fn regular_enumerates_up_to_t_end() {
        let times = output_times(&reg(0.0, 1.0), 160.0);
        assert_eq!(*times.last().unwrap(), 160.0);
        assert_eq!(times.len(), 161); // 0,1,…,160
    }

    /// gh#143 downward: a shorter horizon confines emission to `[start, t_end]`
    /// with no frozen-padding rows past it.
    #[test]
    fn regular_confines_to_t_end_no_padding() {
        let times = output_times(&reg(0.0, 1.0), 40.0);
        assert_eq!(*times.last().unwrap(), 40.0);
        assert_eq!(times.len(), 41); // 0,1,…,40 — nothing past 40
        assert!(times.iter().all(|&t| t <= 40.0));
    }

    /// An explicit `at = [...]` time beyond the horizon is dropped, not emitted
    /// against a frozen state.
    #[test]
    fn at_times_drops_entries_beyond_t_end() {
        let sched = OutputSchedule::AtTimes(vec![0.0, 10.0, 40.0, 50.0, 100.0]);
        let times = output_times(&sched, 40.0);
        assert_eq!(times, vec![0.0, 10.0, 40.0]);
    }

    /// gh#257: an `at = [...]` list may arrive unsorted and with duplicates.
    /// `output_times` is documented to return a *sorted* list; it must also
    /// deduplicate, so a repeated output time does not emit a duplicate
    /// snapshot row. (Non-finite entries are rejected downstream at the
    /// `OutputTimes`/`SortedFiniteTimes` boundary; here we pin the sort + dedup
    /// normalization at the producer so every consumer sees a canonical axis.)
    #[test]
    fn at_times_are_sorted_and_deduped() {
        let sched = OutputSchedule::AtTimes(vec![10.0, 5.0, 5.0, 0.0, 10.0]);
        let times = output_times(&sched, 40.0);
        assert_eq!(times, vec![0.0, 5.0, 10.0]);
    }
}
