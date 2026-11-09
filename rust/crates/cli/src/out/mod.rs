//! The stream layer: how anything reaches the terminal. Knows nothing about
//! fits.
//!
//! It carries both streams, not only stdout: stdout holds data and a report
//! command's report (`fit summary`, `fit diff`), and stderr holds everything
//! else — progress, milestones, notices, warnings, errors and `fit run`'s
//! end-of-run block. "out" names the layer, not the stream.

pub mod notice;
pub mod paint;
pub mod progress;
pub mod status;

/// Writes live lines to stderr from any thread, one whole line at a time.
///
/// The chain workers and the progress tasks reach the terminal only through
/// this handle, so their lines never interleave mid-line and never pass
/// through the `log` crate's filter, timestamp and module path (GH #14).
#[derive(Clone, Copy)]
pub struct LiveHandle {
    lock: &'static std::sync::Mutex<()>,
}

/// The process's live handle.
pub fn live() -> LiveHandle {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LiveHandle { lock: &LOCK }
}

impl LiveHandle {
    /// A chain that finished with something the end of the run will report
    /// in full (`chain 0: proposal scale at its floor …`), so a long run does
    /// not hide it until it ends.
    pub fn chain_finished(&self, line: &str) {
        self.progress(line);
    }

    /// A plain progress line (`chain 1 1/3000  ll=-319.5  acc=0%`), written
    /// whole.
    pub fn progress(&self, line: &str) {
        use std::io::Write;
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        // stderr is unbuffered, and a failed write to it has nowhere to be
        // reported.
        let _ = writeln!(std::io::stderr(), "{line}");
    }
}
