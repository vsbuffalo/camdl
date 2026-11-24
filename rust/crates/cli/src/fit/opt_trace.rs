//! `opt_trace.tsv`: each NLopt search's running best log-likelihood, written
//! while the searches run (`docs/camdl-run-spec.md` §10.13).
//!
//! An NLopt search reports nothing until it finishes, and on a model whose
//! evaluation costs ~0.4 s an 8,000-evaluation search is close to an hour of
//! silence; a search that stopped at `max_evals` cannot then be told apart
//! from one that had levelled off. The trace is the record that tells them
//! apart: one file per leaf, beside `chain_results.tsv`, with a row per search
//! every so often giving its evaluation count, the value it just scored, and
//! the best finite value and point it has seen.
//!
//! The trace observes the objective and never feeds back into it: a
//! [`SearchTrace`] reads the value the objective returned to NLopt, after it
//! was returned, and the optimiser never sees the trace.
//!
//! It is a record of one execution, not of the computation: which
//! evaluations get a row depends partly on wall-clock time, so the file is
//! a provenance file of the run, not a result file ([`OPT_TRACE`]).

/// The trace's file name in a method directory.
pub(crate) const OPT_TRACE: &str = "opt_trace.tsv";

use std::path::PathBuf;
use std::time::{Duration, Instant};

use score::convergence::ChainIndex;

/// A row at least every this many evaluations of one search. A fast search
/// (milliseconds per evaluation, thousands of evaluations) then writes a few
/// hundred rows, enough to draw its climb without the trace outgrowing the
/// rest of the leaf.
const ROW_EVERY_EVALS: u64 = 25;

/// And at least every this long. A slow search (0.4 s per evaluation) would
/// otherwise go ten seconds between rows; two keeps a live plot of it moving
/// with every few evaluations.
const ROW_EVERY: Duration = Duration::from_secs(2);

/// `opt_trace.tsv`'s header line, newline included. The parameter columns
/// are `chain_results.tsv`'s, in its order and spelling, so the two files
/// join on `chain` and line up column for column.
fn header(est_names: &[String]) -> String {
    let mut line = String::from("chain\teval\tloglik\tbest_loglik");
    for name in est_names {
        line.push('\t');
        line.push_str(name);
    }
    line.push('\n');
    line
}

/// A float the way the trace writes it: Rust's shortest representation that
/// parses back to the same `f64`, and `-inf`, `inf` or `nan` for the
/// non-finite values (`Display` would write `NaN`).
fn push_f64(line: &mut String, v: f64) {
    use std::fmt::Write as _;
    if v.is_nan() {
        line.push_str("nan");
    } else {
        write!(line, "{v}").expect("writing to a String cannot fail");
    }
}

/// The leaf's `opt_trace.tsv` while the searches run: created with its
/// header before any search starts, then appended to by every search.
///
/// Each row is formatted in full and handed over in a single unbuffered
/// append, so a reader tailing the file sees whole rows only, and rows from
/// searches on different threads never interleave mid-line.
pub(super) struct OptTraceFile {
    /// Where the file can be read while it is written.
    pub(super) path: PathBuf,
    /// How many parameter columns the header has, so a row with no best
    /// point yet still has one (empty) field per column.
    n_params: usize,
    file: super::output::OutFile,
}

impl OptTraceFile {
    pub(super) fn create(files: &dyn super::output::RunWriter, est_names: &[String]) -> Result<Self, String> {
        let file = files.create(OPT_TRACE)?;
        file.write_all(header(est_names).as_bytes())?;
        Ok(OptTraceFile { path: files.folder().join(OPT_TRACE), n_params: est_names.len(), file })
    }

    fn append(&self, row: &str) -> std::io::Result<()> {
        self.file.write_all(row.as_bytes()).map_err(std::io::Error::other)
    }
}

/// One search's rows: what it has seen so far, and when it last wrote.
pub(super) struct SearchTrace<'a> {
    file: &'a OptTraceFile,
    chain: usize,
    /// The latest evaluation: its count (from 1) and the value it scored.
    latest: Option<(u64, f64)>,
    /// The best finite value seen so far, and the point that scored it.
    /// An equal later value does not replace it, as NLopt keeps the first
    /// point that reached its best.
    best_loglik: Option<f64>,
    best_point: Vec<f64>,
    /// The evaluation count and time of the last row written.
    last_row: Option<(u64, Instant)>,
    /// The first failed write. Rows stop after it rather than retrying a
    /// failing disk on every evaluation; the caller reports it once.
    failed: Option<std::io::Error>,
}

impl<'a> SearchTrace<'a> {
    pub(super) fn new(file: &'a OptTraceFile, chain: usize) -> Self {
        SearchTrace {
            file,
            chain,
            latest: None,
            best_loglik: None,
            best_point: Vec::new(),
            last_row: None,
            failed: None,
        }
    }

    /// The best finite log-likelihood this search has scored, if any.
    pub(super) fn best_loglik(&self) -> Option<f64> {
        self.best_loglik
    }

    /// How many evaluations this search has made, if any.
    pub(super) fn evals(&self) -> Option<u64> {
        self.latest.map(|(evals, _)| evals)
    }

    /// Record evaluation `evals` (from 1) of the point `est` (estimated slots,
    /// natural scale), which scored `loglik`, and write a row if one is due.
    /// Says whether a row was due.
    pub(super) fn observe(&mut self, evals: u64, est: &[f64], loglik: f64) -> bool {
        self.observe_at(evals, est, loglik, Instant::now())
    }

    /// [`observe`](Self::observe) at a stated time, so the cadence can be
    /// tested without waiting on the clock.
    fn observe_at(&mut self, evals: u64, est: &[f64], loglik: f64, now: Instant) -> bool {
        self.latest = Some((evals, loglik));
        if loglik.is_finite() && self.best_loglik.is_none_or(|b| loglik > b) {
            self.best_loglik = Some(loglik);
            self.best_point.clear();
            self.best_point.extend_from_slice(est);
        }
        let due = match self.last_row {
            None => true,
            Some((at_eval, at_time)) => {
                evals.saturating_sub(at_eval) >= ROW_EVERY_EVALS
                    || now.saturating_duration_since(at_time) >= ROW_EVERY
            }
        };
        if due {
            self.write_row(evals, loglik, now);
        }
        due
    }

    /// Write the search's last row, at its final evaluation count, unless
    /// that evaluation already has one, and hand back the first write error,
    /// if any. A search that made no evaluation writes nothing.
    pub(super) fn finish(mut self) -> Option<std::io::Error> {
        if let Some((evals, loglik)) = self.latest {
            if self.last_row.is_none_or(|(at_eval, _)| at_eval != evals) {
                self.write_row(evals, loglik, Instant::now());
            }
        }
        self.failed
    }

    fn write_row(&mut self, evals: u64, loglik: f64, now: Instant) {
        self.last_row = Some((evals, now));
        if self.failed.is_some() {
            return;
        }
        let row = self.row(evals, loglik);
        if let Err(e) = self.file.append(&row) {
            self.failed = Some(e);
        }
    }

    /// One row, newline included: chain, evaluation, this evaluation's value,
    /// then the best value and its point, the last two empty until the search
    /// has scored a finite value.
    fn row(&self, evals: u64, loglik: f64) -> String {
        let mut line = format!("{}\t{evals}\t", ChainIndex(self.chain));
        push_f64(&mut line, loglik);
        line.push('\t');
        match self.best_loglik {
            Some(best) => {
                push_f64(&mut line, best);
                for &v in &self.best_point {
                    line.push('\t');
                    push_f64(&mut line, v);
                }
            }
            None => {
                // One empty field per parameter column.
                for _ in 0..self.file.n_params {
                    line.push('\t');
                }
            }
        }
        line.push('\n');
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("p{i}")).collect()
    }

    fn rows(path: &std::path::Path) -> Vec<Vec<String>> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(1)
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect()
    }

    #[test]
    fn the_header_is_on_disk_before_any_search_writes() {
        let dir = tempfile::tempdir().unwrap();
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(2)).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file.path).unwrap(),
            "chain\teval\tloglik\tbest_loglik\tp0\tp1\n"
        );
    }

    /// A row at the first evaluation, then every 25, and one at the end; the
    /// rows are on disk while the search is still going, as a reader tailing
    /// the file would see them.
    #[test]
    fn rows_reach_disk_while_the_search_runs() {
        let dir = tempfile::tempdir().unwrap();
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(1)).unwrap();
        let t0 = Instant::now();
        let mut trace = SearchTrace::new(&file, 3);
        for e in 1..=60u64 {
            trace.observe_at(e, &[e as f64], -100.0 + e as f64, t0);
        }
        let mid: Vec<String> = rows(&file.path).iter().map(|r| r[1].clone()).collect();
        assert_eq!(mid, ["1", "26", "51"], "rows written before the search ends");
        assert!(trace.finish().is_none());
        let end = rows(&file.path);
        let evals: Vec<&str> = end.iter().map(|r| r[1].as_str()).collect();
        assert_eq!(evals, ["1", "26", "51", "60"], "and a final row at the last evaluation");
        assert_eq!(end[3], ["3", "60", "-40", "-40", "60"]);
    }

    /// The final row is not repeated when the last evaluation already has one.
    #[test]
    fn a_search_ending_on_a_row_writes_it_once() {
        let dir = tempfile::tempdir().unwrap();
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(1)).unwrap();
        let t0 = Instant::now();
        let mut trace = SearchTrace::new(&file, 0);
        for e in 1..=26u64 {
            trace.observe_at(e, &[0.5], -1.0, t0);
        }
        trace.finish();
        let evals: Vec<String> = rows(&file.path).iter().map(|r| r[1].clone()).collect();
        assert_eq!(evals, ["1", "26"]);
    }

    /// A slow search writes a row once two seconds have passed, however few
    /// evaluations that was.
    #[test]
    fn two_seconds_bring_a_row_before_25_evaluations() {
        let dir = tempfile::tempdir().unwrap();
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(1)).unwrap();
        let t0 = Instant::now();
        let mut trace = SearchTrace::new(&file, 0);
        let s = |x: f64| t0 + Duration::from_secs_f64(x);
        trace.observe_at(1, &[0.1], -9.0, s(0.0));
        trace.observe_at(2, &[0.2], -8.0, s(0.4));
        trace.observe_at(3, &[0.3], -7.0, s(1.9));
        trace.observe_at(4, &[0.4], -6.0, s(2.0));
        trace.observe_at(5, &[0.5], -5.0, s(2.4));
        let evals: Vec<String> = rows(&file.path).iter().map(|r| r[1].clone()).collect();
        assert_eq!(evals, ["1", "4"]);
    }

    /// A search whose start is not scorable: its rows carry the non-finite
    /// value, the best and the point stay empty until a finite value arrives,
    /// and the best never decreases after it.
    #[test]
    fn a_non_finite_start_leaves_the_best_empty_until_a_finite_value() {
        let dir = tempfile::tempdir().unwrap();
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(2)).unwrap();
        let t0 = Instant::now();
        let mut trace = SearchTrace::new(&file, 1);
        let values = [f64::NEG_INFINITY, f64::NAN, f64::INFINITY, -12.5, -20.0, -3.25];
        for (i, &v) in values.iter().enumerate() {
            let e = i as u64 + 1;
            // A row at every evaluation, to see each one.
            trace.observe_at(e, &[e as f64, -(e as f64)], v, t0 + ROW_EVERY * e as u32);
        }
        assert!(trace.finish().is_none());
        let got = rows(&file.path);
        assert_eq!(
            got,
            [
                vec!["1", "1", "-inf", "", "", ""],
                vec!["1", "2", "nan", "", "", ""],
                vec!["1", "3", "inf", "", "", ""],
                vec!["1", "4", "-12.5", "-12.5", "4", "-4"],
                vec!["1", "5", "-20", "-12.5", "4", "-4"],
                vec!["1", "6", "-3.25", "-3.25", "6", "-6"],
            ]
        );
    }

    /// Searches run on different threads: every row from every search is
    /// whole, and each search's best never decreases.
    #[test]
    fn concurrent_searches_keep_every_row_whole() {
        let dir = tempfile::tempdir().unwrap();
        let est = names(5);
        let file = OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &est).unwrap();
        const SEARCHES: usize = 8;
        const EVALS: u64 = 400;
        std::thread::scope(|s| {
            for chain in 0..SEARCHES {
                let file = &file;
                s.spawn(move || {
                    let mut trace = SearchTrace::new(file, chain);
                    for e in 1..=EVALS {
                        let v = -(((e * 7919 + chain as u64) % 101) as f64) - 1.0 / 3.0;
                        trace.observe(e, &[v / 7.0; 5], v);
                    }
                    assert!(trace.finish().is_none());
                });
            }
        });
        let got = rows(&file.path);
        let mut last_best = [f64::NEG_INFINITY; SEARCHES];
        let mut n = [0usize; SEARCHES];
        for r in &got {
            assert_eq!(r.len(), 4 + est.len(), "malformed row {r:?}");
            let chain: usize = r[0].parse().unwrap();
            let best: f64 = r[3].parse().unwrap();
            assert!(best >= last_best[chain], "best decreased on chain {chain}: {r:?}");
            last_best[chain] = best;
            n[chain] += 1;
        }
        // 1, 26, …, 376 and the final 400.
        assert!(n.iter().all(|&k| k >= 17), "rows per search: {n:?}");
    }

    #[test]
    fn floats_round_trip_and_name_the_non_finite_values() {
        for v in [0.1, -47.81234567890123, 1e-7, 123456789.123, -0.0] {
            let mut s = String::new();
            push_f64(&mut s, v);
            assert_eq!(s.parse::<f64>().unwrap().to_bits(), v.to_bits(), "{s}");
        }
        for (v, want) in [(f64::NAN, "nan"), (f64::INFINITY, "inf"), (f64::NEG_INFINITY, "-inf")] {
            let mut s = String::new();
            push_f64(&mut s, v);
            assert_eq!(s, want);
        }
    }
}
