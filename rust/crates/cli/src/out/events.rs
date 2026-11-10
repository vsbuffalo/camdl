//! A fit method's event log, `events.jsonl`: what the run reports, one JSON
//! line per event, as it happens.
//!
//! The terminal shows a run's checks, progress, findings and end-of-run
//! report as text; this records the same values for a reader that is not the
//! terminal — camdl-scope, a terminal UI — so it never parses the text
//! (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`, "Live
//! consumers"). The layout is specified in `docs/camdl-run-spec.md` §10.11.
//!
//! [`Out`](super::Out) opens the log when a method begins and owns it; the
//! chain workers write to it through the handle's collector and progress
//! tasks, so every line goes through one [`EventLog`] behind one lock. Each
//! line is written whole with one `write`, so a reader tailing the file sees
//! whole lines. The log is a display record: it enters no hash, and the
//! store's exact set passes over it ([`runid::EVENT_LOG`]), because the
//! end-of-run report is appended after the leaf commits.
//!
//! A write that fails is the run's failure, as `diagnostics.json`'s is: the
//! first failure is kept, later events are not written, and the run reports
//! it before it calls the method done ([`Out::events_result`]) and after the
//! report ([`Out::close_events`]).
//!
//! [`Out::events_result`]: super::Out::events_result
//! [`Out::close_events`]: super::Out::close_events

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use crate::fit::loglik::LoglikType;
use crate::fit::report::build::MethodReport;
use crate::fit::report::{Check, RunHeader};
use sim::inference::diagnostic::Diagnostic;

/// The schema every line names. A change to what an existing field means,
/// or a removed field, is a new version; a new event or a new field is not.
pub const SCHEMA: &str = "camdl.events/v1";

/// One event a running method reports.
#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event<'a> {
    /// The method started: which one, and the run's opening block, which no
    /// file in the leaf holds until it commits.
    Begin { method: &'a str, header: &'a RunHeader },
    /// A check the run showed before sampling (`Out::check`).
    Check { check: &'a Check },
    /// A chain's progress, at the plain progress line's cadence and at its
    /// last step.
    Progress(Progress),
    /// A finding, as `diagnostics.json` records it, when it is pushed.
    Finding { finding: &'a Diagnostic },
    /// How the method ended: the end-of-run report it showed, or the error
    /// that stopped it. The last line of a run that camdl ended; a log with
    /// none was cut off (the process was killed), or the run stopped where
    /// the handle cannot reach (an IF2 or NLopt runner's own exit).
    MethodReport(Outcome<'a>),
}

/// The method's outcome, as `Out::method_report` shows it.
#[derive(Debug, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome<'a> {
    /// The report `fit summary` rebuilds from the leaf, as the run built it.
    Completed { report: &'a MethodReport },
    /// The error as the command reported it. The findings it recorded are
    /// the log's `finding` lines.
    Failed { error: &'a str },
}

/// One chain's progress: the step it has reached and its current figures.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Progress {
    /// The chain's number, as its `chain_N/` directory carries it.
    pub chain: usize,
    /// Steps taken in this run, of `of`. One step is what `progress.json`'s
    /// step counts for the method (§10.10).
    pub step: u64,
    pub of: u64,
    /// The chain's current log-likelihood and its class; absent until the
    /// chain has one.
    pub loglik: Option<Loglik>,
    /// The chain's acceptance fraction so far, in `[0, 1]`, for a method that
    /// accepts or rejects proposals (PMMH, `mh`); absent otherwise.
    pub accept: Option<f64>,
}

/// A log-likelihood and its class: a marginal `log p(y | θ)` or PGAS's
/// complete-data value, which are not comparable (gh#280).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Loglik {
    pub value: f64,
    pub kind: LoglikType,
}

impl Progress {
    /// A chain's progress from what its bar shows: `metric` is the figure
    /// last set on it.
    pub fn of_chain(chain: usize, step: u64, of: u64, metric: Option<super::progress::Metric>) -> Self {
        use super::progress::Metric as M;
        let (loglik, accept) = match metric {
            Some(M::Loglik { value, kind }) => (Some(Loglik { value, kind }), None),
            Some(M::Mcmc { loglik, kind, accept }) => (Some(Loglik { value: loglik, kind }), Some(accept)),
            // A best-so-far figure is a search's, not a chain's current one.
            Some(M::BestLoglik(_)) | None => (None, None),
        };
        Progress { chain, step, of, loglik, accept }
    }
}

/// A line as written: the schema, the line's place in the log, the seconds
/// since the method began, and the event's own fields.
#[derive(Serialize)]
struct Line<'a> {
    schema: &'static str,
    seq: u64,
    elapsed: f64,
    #[serde(flatten)]
    event: &'a Event<'a>,
}

/// The open log of one method's events. Cloned into every writer; all share
/// one file behind one lock.
#[derive(Clone)]
pub struct EventLog(Arc<Inner>);

struct Inner {
    path: PathBuf,
    began: Instant,
    state: Mutex<State>,
}

struct State {
    /// `None` once a write has failed.
    file: Option<std::fs::File>,
    seq: u64,
    failure: Option<String>,
}

impl EventLog {
    /// Create `dir/events.jsonl`, replacing any log a previous run left.
    pub fn create(dir: &Path) -> Result<Self, String> {
        let path = dir.join(runid::EVENT_LOG);
        let file = std::fs::File::create(&path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(EventLog(Arc::new(Inner {
            path,
            began: Instant::now(),
            state: Mutex::new(State { file: Some(file), seq: 0, failure: None }),
        })))
    }

    /// Append `event` as one line. Numbers are kept exact: a non-finite one
    /// is the string `"NaN"`, `"inf"` or `"-inf"`, never `null`.
    pub fn record(&self, event: &Event<'_>) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let State { file, seq, failure } = &mut *state;
        let Some(f) = file.as_mut() else { return };
        let line = Line { schema: SCHEMA, seq: *seq, elapsed: self.0.began.elapsed().as_secs_f64(), event };
        let written = crate::fit::report::exact::exact_json(&line)
            .map_err(|e| e.to_string())
            .and_then(|v| {
                let mut text = v.to_string();
                text.push('\n');
                f.write_all(text.as_bytes()).map_err(|e| e.to_string())
            });
        match written {
            Ok(()) => *seq += 1,
            Err(e) => {
                *failure = Some(format!("cannot write {}: {e}", self.0.path.display()));
                *file = None;
            }
        }
    }

    /// The first write that failed, if one has.
    pub fn result(&self) -> Result<(), String> {
        let state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        match &state.failure {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind};

    fn header() -> RunHeader {
        RunHeader {
            config: "fit.toml".into(), method: "pmmh".into(), model: "sir.camdl".into(),
            estimate: vec!["beta".into()], fixed: vec!["N0".into()], starts: "single".into(),
            output: "results/fits/fit-aaaaaaaa".into(),
        }
    }

    /// The JSON of `event` as a line carries it, without the fields that
    /// differ between runs (`elapsed`).
    fn line(event: &Event<'_>) -> serde_json::Value {
        let mut v = crate::fit::report::exact::exact_json(&Line { schema: SCHEMA, seq: 7, elapsed: 0.0, event })
            .unwrap();
        v.as_object_mut().unwrap().remove("elapsed");
        v
    }

    /// The schema, pinned: the fields every line carries, and each event's
    /// own. A consumer (camdl-scope, a terminal UI) reads these names; a
    /// change here is a change to `docs/camdl-run-spec.md` §10.11, and a
    /// change of meaning is a new [`SCHEMA`].
    #[test]
    fn the_event_schema() {
        use serde_json::json;
        let h = header();
        assert_eq!(line(&Event::Begin { method: "pmmh", header: &h }), json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "begin", "method": "pmmh",
            "header": {
                "config": "fit.toml", "method": "pmmh", "model": "sir.camdl", "estimate": ["beta"],
                "fixed": ["N0"], "starts": "single", "output": "results/fits/fit-aaaaaaaa",
            },
        }));

        let c = Check::MethodCaveat { method: "pmmh".into(), note: "n".into(), series_len: Some(14) };
        assert_eq!(line(&Event::Check { check: &c }), json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "check",
            "check": { "kind": "method_caveat", "method": "pmmh", "note": "n", "series_len": 14 },
        }));

        let p = Progress::of_chain(2, 1, 300, Some(super::super::progress::mcmc(f64::NEG_INFINITY, LoglikType::Marginal, 0.25)));
        assert_eq!(line(&Event::Progress(p)), json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "progress", "chain": 2, "step": 1, "of": 300,
            "loglik": { "value": "-inf", "kind": "marginal" }, "accept": 0.25,
        }));
        let p = Progress::of_chain(0, 300, 300,
            Some(super::super::progress::ll_kind(-58.5, LoglikType::CompleteData)));
        assert_eq!(line(&Event::Progress(p)), json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "progress", "chain": 0, "step": 300, "of": 300,
            "loglik": { "value": -58.5, "kind": "complete_data" }, "accept": null,
        }));

        let collector = DiagnosticCollector::new("pmmh");
        collector.push(DiagnosticKind::ScaleFarFromOptimum { chain_id: 1, final_scale: f64::NAN });
        let mut d = collector.snapshot().remove(0);
        d.timestamp = "2026-11-09T00:00:00Z".into();
        let mut finding = line(&Event::Finding { finding: &d });
        let message = finding["finding"]["message"].take();
        assert!(message.as_str().is_some_and(|m| !m.is_empty()));
        assert_eq!(finding, json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "finding",
            "finding": {
                "kind": { "type": "scale_far_from_optimum", "chain_id": 1, "final_scale": "NaN" },
                "severity": "warning", "message": null, "method": "pmmh",
                "timestamp": "2026-11-09T00:00:00Z",
            },
        }));

        assert_eq!(line(&Event::MethodReport(Outcome::Failed { error: "error: no chain started" })), json!({
            "schema": "camdl.events/v1", "seq": 7, "event": "method_report", "outcome": "failed",
            "error": "error: no chain started",
        }));
    }

    /// A write that fails is kept as the log's failure, and nothing is
    /// written after it: the run reports the first failure, once. A log that
    /// cannot be created is an error at once.
    #[test]
    fn a_failed_write_is_the_logs_failure() {
        let dir = std::env::temp_dir().join(format!("camdl_events_fail_{}_{:?}", std::process::id(),
            std::time::SystemTime::now()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.jsonl");
        std::fs::write(&path, "").unwrap();
        // Open for reading only, so every write fails.
        let file = std::fs::File::open(&path).unwrap();
        let log = EventLog(Arc::new(Inner {
            path: path.clone(), began: Instant::now(),
            state: Mutex::new(State { file: Some(file), seq: 0, failure: None }),
        }));
        let h = header();
        log.record(&Event::Begin { method: "pmmh", header: &h });
        let e = log.result().expect_err("a write to a read-only file failed");
        assert!(e.starts_with(&format!("cannot write {}: ", path.display())), "{e}");
        log.record(&Event::Begin { method: "pmmh", header: &h });
        assert_eq!(log.result(), Err(e), "the first failure is the one reported");

        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        let e = EventLog::create(&dir).err().expect("a directory in the log's place");
        assert!(e.starts_with("cannot write "), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Lines are numbered in the order they are written, each is whole, and
    /// a run's log starts empty however much a previous one left.
    #[test]
    fn lines_are_whole_numbered_and_start_afresh() {
        let dir = std::env::temp_dir().join(format!("camdl_events_{}_{:?}", std::process::id(),
            std::time::SystemTime::now()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("events.jsonl"), "left by a previous run\n").unwrap();
        let log = EventLog::create(&dir).unwrap();
        let h = header();
        std::thread::scope(|s| {
            for chain in 0..4 {
                let log = log.clone();
                s.spawn(move || log.record(&Event::Progress(Progress::of_chain(chain, 1, 2, None))));
            }
        });
        log.record(&Event::Begin { method: "pmmh", header: &h });
        assert_eq!(log.result(), Ok(()));
        let text = std::fs::read_to_string(dir.join("events.jsonl")).unwrap();
        let seqs: Vec<u64> = text.lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("a whole JSON line")["seq"].as_u64().unwrap())
            .collect();
        assert_eq!(seqs, [0, 1, 2, 3, 4]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
