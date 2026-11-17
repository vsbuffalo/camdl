//! `execution.json` — the record of one attempt at a fit method leaf's
//! computation.
//!
//! A leaf's stored bytes (the files its manifest checksums) are a function of
//! its computation: the same inputs write the same bytes. Everything about a
//! particular attempt to compute it — when it started and finished, how long
//! it took, which process on which host ran it, how many worker threads it
//! had, what its sampling cost in wall time — differs between two attempts,
//! so it is kept here, beside the leaf and outside its manifest
//! ([`runid::EXECUTION_RECORD`], one of [`runid::PER_ATTEMPT_RECORDS`]).
//!
//! The run writes the record when it claims the leaf, so an attempt that dies
//! still leaves when and where it ran, and rewrites it with the finish when
//! the method returns, before the leaf commits. A reclaim moves a dead
//! attempt's record into `.attempts/<n>/` with its `run.json`, so a leaf
//! keeps one record per attempt; the one at the leaf root is the attempt that
//! wrote the leaf's result.
//!
//! Read by `fit summary`, `fit table` and the end-of-run report
//! (`method_result.rs`, `fit_summary.rs`): the wall time behind ESS per second,
//! the `stored … · <time>` line and the sampling-cost line.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fit::report::SamplingCost;

/// The tag a reader keys on to know the record's layout.
pub const SCHEMA: &str = "camdl.execution/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Execution {
    pub schema: String,
    /// When this attempt claimed the leaf, ISO 8601 UTC to the second.
    pub started_at: String,
    /// When the method returned; `None` for an attempt still running, or one
    /// that died before it finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    /// The method's wall time in this attempt, from the dispatch to the
    /// commit; `None` until it finished. A resumed leaf's draws span earlier
    /// segments too: [`chain_wall_time_seconds`](Self::chain_wall_time_seconds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_time_seconds: Option<f64>,
    pub pid: u32,
    /// The host's name as `gethostname` gives it; empty when it cannot be read.
    pub host: String,
    /// Worker threads available to the method (`--parallel`, or every core).
    pub threads: usize,
    /// What sampling cost a PMMH or `mh` method: the chains' wall time, and
    /// per likelihood evaluation. `None` for every other method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<SamplingCost>,
    /// The chains' wall time of a PGAS method, the figure its `pgas complete
    /// in …` line prints. `None` for every other method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_seconds: Option<f64>,
    /// A resumed leaf only: the earlier segments of its chain, oldest first,
    /// each with the method wall time its own execution record gave. The
    /// segment boundaries themselves are lineage, a function of the
    /// computation, and are in `run.json` `inputs.resume`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub earlier_segments: Vec<SegmentTime>,
}

/// One earlier segment of a resumed chain, as the resumed leaf's execution
/// record keeps it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentTime {
    /// The run that sampled the segment.
    pub run_id: String,
    /// Its method wall time; `None` when its record had none.
    pub wall_time_seconds: Option<f64>,
}

/// What a sampler measured of its own cost, handed to the run for the
/// execution record (`Out::record_sampling`).
#[derive(Debug, Clone, PartialEq)]
pub enum Sampling {
    /// PMMH and `mh`: wall time, and per likelihood evaluation.
    PerEvaluation(SamplingCost),
    /// PGAS: the chains' wall time.
    Chains { seconds: f64 },
}

impl Execution {
    /// Record what the sampler measured of its cost.
    pub fn record(&mut self, sampling: Sampling) {
        match sampling {
            Sampling::PerEvaluation(cost) => self.cost = Some(cost),
            Sampling::Chains { seconds } => self.sampling_seconds = Some(seconds),
        }
    }

    /// The record of an attempt starting now, in this process.
    pub fn begin() -> Self {
        Execution {
            schema: SCHEMA.to_string(),
            started_at: crate::cas::iso8601_utc(std::time::SystemTime::now()),
            finished_at: None,
            wall_time_seconds: None,
            pid: std::process::id(),
            host: runid::this_host().to_string(),
            threads: rayon::current_num_threads(),
            cost: None,
            sampling_seconds: None,
            earlier_segments: Vec::new(),
        }
    }

    /// The attempt finished its method `wall_time_seconds` after it started.
    pub fn finish(&mut self, wall_time_seconds: f64) {
        self.finished_at = Some(crate::cas::iso8601_utc(std::time::SystemTime::now()));
        self.wall_time_seconds = Some(wall_time_seconds);
    }

    /// Write the record into `leaf`, atomically.
    pub fn write(&self, leaf: &Path) -> Result<(), String> {
        let path = leaf.join(runid::EXECUTION_RECORD);
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("cannot encode {}: {e}", path.display()))?;
        runid::write_file_atomic(&path, &json).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// The record of the attempt that wrote `leaf`; `Ok(None)` when the leaf
    /// has none. A record that exists and cannot be read or parsed is an
    /// error, not an absence.
    pub fn read(leaf: &Path) -> Result<Option<Self>, String> {
        let path = leaf.join(runid::EXECUTION_RECORD);
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| format!("cannot parse {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("cannot read {}: {e}", path.display())),
        }
    }

    /// The method wall time behind every draw the leaf holds — the
    /// denominator of ESS per second: this attempt's, plus every earlier
    /// segment's for a resumed leaf. `None` when any of them is unknown,
    /// rather than one segment's time set against the whole chain's ESS.
    pub fn chain_wall_time_seconds(&self) -> Option<f64> {
        self.earlier_segments
            .iter()
            .map(|s| s.wall_time_seconds)
            .chain(std::iter::once(self.wall_time_seconds))
            .sum()
    }

    /// The earlier segments a leaf resuming `parent` inherits: the parent's
    /// own earlier segments and the parent itself, timed by its execution
    /// record. A parent with no readable record contributes a segment of
    /// unknown time.
    pub fn segments_resuming(parent: &Path, parent_run_id: &str) -> Vec<SegmentTime> {
        match Self::read(parent) {
            Ok(Some(p)) => {
                let mut segs = p.earlier_segments;
                segs.push(SegmentTime { run_id: parent_run_id.to_string(), wall_time_seconds: p.wall_time_seconds });
                segs
            }
            _ => vec![SegmentTime { run_id: parent_run_id.to_string(), wall_time_seconds: None }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(w: Option<f64>) -> SegmentTime {
        SegmentTime { run_id: "r".into(), wall_time_seconds: w }
    }

    #[test]
    fn a_resumed_chains_wall_time_is_the_sum_over_its_segments() {
        let mut e = Execution::begin();
        assert_eq!(e.chain_wall_time_seconds(), None, "an attempt not yet finished has no wall time");
        e.finish(3.0);
        assert_eq!(e.chain_wall_time_seconds(), Some(3.0));
        e.earlier_segments = vec![seg(Some(1.5)), seg(Some(0.5))];
        assert_eq!(e.chain_wall_time_seconds(), Some(5.0));
        e.earlier_segments.push(seg(None));
        assert_eq!(e.chain_wall_time_seconds(), None, "an unknown segment leaves it unknown");
    }

    #[test]
    fn the_record_round_trips_and_its_absence_is_not_an_error() {
        let d = crate::test_support::unique_temp_dir("execution_record");
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(Execution::read(&d).unwrap(), None);
        let mut e = Execution::begin();
        e.finish(2.0);
        e.sampling_seconds = Some(1.25);
        e.write(&d).unwrap();
        assert_eq!(Execution::read(&d).unwrap(), Some(e.clone()));

        let segs = Execution::segments_resuming(&d, "parent");
        assert_eq!(segs, vec![SegmentTime { run_id: "parent".into(), wall_time_seconds: Some(2.0) }]);
        std::fs::write(d.join(runid::EXECUTION_RECORD), b"not json").unwrap();
        assert!(Execution::read(&d).is_err(), "an unparseable record is an error");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The heartbeat file is one of the store's per-attempt records by the
    /// name the heartbeat writes it under: the two crates name it apart.
    #[test]
    fn the_heartbeat_is_a_per_attempt_record() {
        assert!(runid::PER_ATTEMPT_RECORDS.contains(&io::PROGRESS_FILE));
    }
}
