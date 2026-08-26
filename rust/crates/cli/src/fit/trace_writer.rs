//! Shared streaming TSV trace writer for MCMC methods (PGAS, PMMH).
//!
//! Handles header construction, append mode for `--resume`, periodic
//! flushing, and thread-safe writing via Mutex.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Flush at least this often, even when fewer than `flush_interval` rows have
/// accumulated. On a slow sampler (seconds per sweep) this bounds how long a
/// live `trace.tsv` tail waits to see a row to ~this window — instead of the
/// file stalling for the whole 50-row batch — while a fast sampler still
/// batches by the row-count ceiling.
const FLUSH_AFTER: Duration = Duration::from_millis(250);

/// Streaming TSV trace writer for MCMC traces.
///
/// Shared columns: `{index_col}`, `{loglik_col}`, `log_posterior`. The
/// loglik column is named by the caller because its *meaning* differs by
/// method — PMMH writes the marginal/PF estimate (`log_likelihood`), PGAS
/// writes the complete-data conditional value (`log_complete_data_ll`), which
/// is many orders of magnitude more negative; a shared bare header would
/// invite comparing the two (gh#261). Method-specific columns (e.g.,
/// `trajectory_renewal`, `accepted`) are passed as `extra_columns` at
/// construction and `extra_values` at each write.
pub struct TraceWriter {
    inner: Mutex<Inner>,
    flush_interval: usize,
    row_count: AtomicUsize,
}

/// The parts that must stay consistent under one lock: the buffered writer and
/// the wall-clock of its last flush (the time-based flush trigger). Keeping
/// them behind the same `Mutex` means a flush and its timestamp update can never
/// race across the chains sharing the writer.
struct Inner {
    writer: BufWriter<File>,
    last_flush: Instant,
}

impl TraceWriter {
    /// Create a new trace writer.
    ///
    /// - `append = false`: creates file and writes header.
    /// - `append = true`: opens in append mode (header already exists).
    pub fn new(
        path: &str,
        index_col: &str,
        loglik_col: &str,
        extra_columns: &[&str],
        param_names: &[String],
        append: bool,
    ) -> Self {
        let writer = if append && std::path::Path::new(path).exists() {
            BufWriter::new(
                OpenOptions::new().append(true).open(path)
                    .unwrap_or_else(|e| panic!("cannot open {} for append: {}", path, e))
            )
        } else {
            let mut f = BufWriter::new(
                File::create(path)
                    .unwrap_or_else(|e| panic!("cannot create {}: {}", path, e))
            );
            // Write header
            write!(f, "{}\t{}\tlog_posterior", index_col, loglik_col).unwrap();
            for col in extra_columns {
                write!(f, "\t{}", col).unwrap();
            }
            for name in param_names {
                write!(f, "\t{}", name).unwrap();
            }
            writeln!(f).unwrap();
            // Flush the header immediately so the file is non-empty — a watcher
            // can discover it and read its schema the instant the writer exists,
            // rather than only after the first 50-row batch drains.
            f.flush().ok();
            f
        };

        TraceWriter {
            inner: Mutex::new(Inner { writer, last_flush: Instant::now() }),
            flush_interval: 50,
            row_count: AtomicUsize::new(0),
        }
    }

    /// Write one trace row.
    ///
    /// `extra_values` must match the `extra_columns` passed to `new()`.
    /// Values are pre-formatted by the caller (e.g., `"0.9571"` or `"1"`).
    pub fn write_row(
        &self,
        index: usize,
        log_likelihood: f64,
        log_posterior: f64,
        extra_values: &[&str],
        param_values: &[f64],
    ) {
        if let Ok(mut inner) = self.inner.lock() {
            write!(inner.writer, "{}\t{:.4}\t{:.4}", index, log_likelihood, log_posterior).unwrap();
            for val in extra_values {
                write!(inner.writer, "\t{}", val).unwrap();
            }
            for &v in param_values {
                // Shortest round-trippable Display — fixed `{:.6}` zeroed any
                // parameter below ~5e-7 (importation/spark rates), faking a
                // frozen chain and corrupting R̂/ESS (gh#266).
                write!(inner.writer, "\t{}", v).unwrap();
            }
            writeln!(inner.writer).unwrap();

            // Flush when EITHER the row-count ceiling is reached OR the time
            // window has elapsed — whichever comes first. The count ceiling
            // bounds buffering on a fast sampler; the time trigger bounds
            // staleness on a slow one (so a live tail sees ~every row).
            let n = self.row_count.fetch_add(1, Ordering::Relaxed);
            let count_due = (n + 1).is_multiple_of(self.flush_interval);
            let time_due = inner.last_flush.elapsed() >= FLUSH_AFTER;
            if (count_due || time_due)
                && inner.writer.flush().is_ok() {
                    inner.last_flush = Instant::now();
                }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#266: a small-magnitude parameter (e.g. an importation rate ~1e-7)
    /// must round-trip through the trace — not be truncated to "0.000000" and
    /// read back as a frozen 0.0 that corrupts every mixing diagnostic.
    #[test]
    fn small_magnitude_param_round_trips_in_trace() {
        let dir = std::env::temp_dir().join(format!("camdl_tw_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.tsv");
        let ps = path.to_string_lossy().into_owned();
        {
            let tw = TraceWriter::new(&ps, "step", "log_likelihood", &[], &["iota".to_string()], false);
            tw.write_row(0, -10.0, -11.0, &[], &[1e-7]);
            tw.write_row(1, -10.0, -11.0, &[], &[9.9e-8]);
        } // drop flushes the BufWriter

        let txt = std::fs::read_to_string(&path).unwrap();
        let mut lines = txt.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let col = header.iter().position(|h| *h == "iota").unwrap();
        let parse = |line: &str| -> f64 { line.split('\t').nth(col).unwrap().parse().unwrap() };
        assert_eq!(parse(lines.next().unwrap()), 1e-7, "1e-7 must round-trip, not truncate to 0");
        assert_eq!(parse(lines.next().unwrap()), 9.9e-8, "9.9e-8 must round-trip");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A live `trace.tsv` tail must not stay 0 bytes until the 50-row batch
    /// drains. The header must be on disk the instant `new()` returns
    /// (discoverable + schema-visible), and a single written row must become
    /// visible via the time-based flush — without accumulating `flush_interval`
    /// (50) rows first.
    #[test]
    fn header_flushed_on_new_and_row_flushed_before_full_batch() {
        let dir = std::env::temp_dir().join(format!("camdl_tw_flush_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.tsv");
        let ps = path.to_string_lossy().into_owned();

        let tw = TraceWriter::new(
            &ps, "step", "log_likelihood", &["accepted"], &["beta".to_string()], false,
        );

        // Header on disk immediately — before any row is written.
        let after_new = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            after_new, "step\tlog_likelihood\tlog_posterior\taccepted\tbeta\n",
            "new() must flush the header so the file is non-empty from creation",
        );

        // One row, far under the 50-row ceiling, must be flushed once the time
        // window elapses. Sleeping past FLUSH_AFTER makes the time trigger fire
        // on the next write; over-sleeping only strengthens the guarantee.
        std::thread::sleep(FLUSH_AFTER + Duration::from_millis(100));
        tw.write_row(0, -10.0, -11.0, &["1"], &[0.5]);

        let txt = std::fs::read_to_string(&path).unwrap();
        let data_lines: Vec<&str> = txt.lines().skip(1).filter(|l| !l.is_empty()).collect();
        assert_eq!(
            data_lines.len(), 1,
            "the single row must be visible via the time-based flush (not batched to 50): {txt:?}",
        );
        assert_eq!(
            data_lines[0], "0\t-10.0000\t-11.0000\t1\t0.5",
            "row content must be intact",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
