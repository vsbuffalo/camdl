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

use std::io::Write;
use std::sync::Arc;

use paint::{Line, Painter, Stream};
use sim::inference::diagnostic::DiagnosticCollector;

/// A command's exit status, as the process reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitCode(pub i32);

impl ExitCode {
    /// The command failed: 1, as every error site has always exited.
    pub const FAILURE: ExitCode = ExitCode(1);

    /// End the process with this status.
    pub fn exit(self) -> ! {
        std::process::exit(self.0)
    }
}

/// An error a command reports and exits on.
///
/// `message` is the whole first line, as the command has always printed it
/// (`error: …`, `error running pmmh: …`): the sites word their lead
/// differently, and this type moves them without rewording them. `hint`, when
/// present, is printed indented on the lines below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandError {
    pub code: ExitCode,
    pub message: String,
    pub hint: Option<String>,
}

impl CommandError {
    /// An error exiting 1, with no hint.
    pub fn new(message: impl Into<String>) -> Self {
        CommandError { code: ExitCode::FAILURE, message: message.into(), hint: None }
    }
}

/// The one handle a command writes through, to either stream, and the record
/// of the method a `fit run` is running.
///
/// stdout carries a report command's report ([`report`](Self::report)) and a
/// data command's data ([`data`](Self::data)); everything else goes to
/// stderr. Each stream has its own painter, so neither carries the other's
/// colour decision. Before writing to stderr, stdout is flushed, so the two
/// interleave in call order when they share a terminal. A closed stdout
/// (`… | head`) ends the command with status 0.
pub struct Out {
    stdout: Painter,
    stderr: Painter,
    notices: Vec<notice::Notice>,
    /// The method running now, from [`begin_method`](Self::begin_method) to
    /// [`end_method`](Self::end_method).
    method: Option<RunningMethod>,
}

/// What `Out` keeps for the method that is running: the run's opening block,
/// the checks it has shown, the collector its findings go to, and its
/// heartbeat, which an error marks failed.
struct RunningMethod {
    header: crate::fit::report::RunHeader,
    checks: Vec<crate::fit::report::Check>,
    collector: Arc<DiagnosticCollector>,
    heartbeat: Arc<io::HeartbeatGuard>,
}

/// A method's record as it ended: what a failed run's report is built from,
/// and what a completed run's report adds to its leaf's typed result.
pub struct MethodRecord {
    pub header: crate::fit::report::RunHeader,
    pub checks: Vec<crate::fit::report::Check>,
    pub findings: Vec<sim::inference::diagnostic::Diagnostic>,
}

impl Out {
    /// A handle whose painters honour `no_color` (a command's `--no-color`),
    /// then the environment, then whether each stream is a terminal.
    pub fn new(no_color: bool) -> Self {
        Out {
            stdout: Painter::for_stream(Stream::Stdout, no_color),
            stderr: Painter::for_stream(Stream::Stderr, no_color),
            notices: Vec::new(),
            method: None,
        }
    }

    /// The stderr painter, for call sites that still build their own strings.
    pub fn stderr_painter(&self) -> Painter {
        self.stderr
    }

    /// Write `lines` to stderr, after flushing stdout.
    pub fn stderr_lines(&self, lines: &[Line]) {
        let _ = std::io::stdout().flush();
        let _guard = live().lock.lock().unwrap_or_else(|e| e.into_inner());
        // stderr is unbuffered, and a failed write to it has nowhere to be
        // reported.
        let _ = self.stderr.write(&mut std::io::stderr().lock(), lines);
    }

    /// A milestone (`   stored pmmh · results/fits/… · 1.0s`).
    pub fn milestone(&self, verb: &str, detail: impl std::fmt::Display) {
        self.say(status::step_text(self.stderr, verb, detail));
    }

    /// A dimmed continuation under a milestone, aligned under its detail
    /// (`           camdl cat <id>`).
    pub fn hint(&self, detail: impl std::fmt::Display) {
        self.say(status::hint_text(self.stderr, detail));
    }

    /// One line of the command's own account of what it is doing, on stderr
    /// (`pfilter: 14 observation boundaries × 1 streams, …`). `text` may hold
    /// several lines; a newline is added after the last.
    ///
    /// The lines a command narrates as it goes are not yet typed values; this
    /// is where they reach stderr, so that the command writes through `Out`
    /// and through nothing else.
    pub fn say(&self, text: impl std::fmt::Display) {
        self.say_raw(format_args!("{text}\n"));
    }

    /// [`say`](Self::say) without the newline, for text that already ends in
    /// one (a pre-rendered table) or is completed by a later write.
    pub fn say_raw(&self, text: impl std::fmt::Display) {
        let _ = std::io::stdout().flush();
        let _guard = live().lock.lock().unwrap_or_else(|e| e.into_inner());
        // stderr is unbuffered, and a failed write to it has nowhere to be
        // reported.
        let _ = write!(std::io::stderr().lock(), "{text}");
    }

    /// Start recording a method: its opening block and its heartbeat. Returns
    /// the collector its findings go to, for the chain workers that push from
    /// their own threads; the command's own thread pushes through
    /// [`finding`](Self::finding).
    pub fn begin_method(
        &mut self,
        header: crate::fit::report::RunHeader,
        method: &str,
        heartbeat: Arc<io::HeartbeatGuard>,
    ) -> Arc<DiagnosticCollector> {
        let collector = Arc::new(DiagnosticCollector::new(method));
        self.method = Some(RunningMethod {
            header, checks: Vec::new(), collector: collector.clone(), heartbeat,
        });
        collector
    }

    /// Record a check the run has just shown. The check's own printer writes
    /// its lines today (`pf_noise::report`, `util::print_observations_summary`,
    /// the caveat banner, the starts note); this keeps the value, so the
    /// run's report carries what it showed.
    pub fn check(&mut self, c: crate::fit::report::Check) {
        if let Some(m) = self.method.as_mut() {
            m.checks.push(c);
        }
    }

    /// The running method's collector, for code that pushes findings from the
    /// chain workers' threads.
    ///
    /// # Panics
    /// When no method is running: a method's findings have nowhere else to
    /// go, and a runner called outside one is a wiring bug.
    pub fn collector(&self) -> Arc<DiagnosticCollector> {
        self.method.as_ref().map(|m| m.collector.clone())
            .expect("a method's collector is asked for only while it runs (`Out::begin_method`)")
    }

    /// Record a finding, from the command's thread.
    pub fn finding(&self, kind: sim::inference::diagnostic::DiagnosticKind) {
        if let Some(m) = &self.method {
            m.collector.push(kind);
        }
    }

    /// The method ran to its end: its heartbeat's terminal record is `done`.
    pub fn method_done(&self) {
        if let Some(m) = &self.method {
            m.heartbeat.done();
        }
    }

    /// Stop recording the method and hand back what it recorded.
    pub fn end_method(&mut self) -> Option<MethodRecord> {
        self.method.take().map(|m| MethodRecord {
            header: m.header,
            checks: m.checks,
            findings: m.collector.snapshot(),
        })
    }

    /// The running method's record so far, without ending it.
    pub fn method_record(&self) -> Option<MethodRecord> {
        self.method.as_ref().map(|m| MethodRecord {
            header: m.header.clone(),
            checks: m.checks.clone(),
            findings: m.collector.snapshot(),
        })
    }

    /// The end-of-run block, drawn once from the method's outcome
    /// (`View::RunEnd`), on stderr.
    pub fn method_report(&mut self, o: &crate::fit::report::build::MethodOutcome) {
        use crate::fit::report::render::{render_report, View};
        self.stderr_lines(&render_report(o, View::RunEnd));
        if let crate::fit::report::build::MethodOutcome::Completed(report) = o {
            crate::fit::report::exact::capture("run", report);
        }
    }

    /// A notice, printed once and kept with the command's others.
    #[cfg_attr(not(test), expect(dead_code, reason = "foundation for proposal step 9; wired when \
        the commands that collect notices (profile, survey, the parameter resolver) move onto Out"))]
    pub fn notice(&mut self, n: notice::Notice) {
        let _ = std::io::stdout().flush();
        eprintln!("{}", n.render());
        self.notices.push(n);
    }

    /// The notices this command has printed, in order.
    #[cfg_attr(not(test), expect(dead_code, reason = "read with `notice`, proposal step 9"))]
    pub fn notices(&self) -> &[notice::Notice] {
        &self.notices
    }

    /// Report `e` on stderr and return the status the command exits with.
    pub fn error(&self, e: CommandError) -> ExitCode {
        if let Some(m) = &self.method {
            // The same sentence a reader of stderr gets, for a reader holding
            // only `progress.json`.
            m.heartbeat.failed(e.message.as_str());
        }
        self.say(&e.message);
        if let Some(hint) = &e.hint {
            for line in hint.lines() {
                self.say(format_args!("  {line}"));
            }
        }
        e.code
    }

    /// Report `message` as an error exiting 1, and exit: the
    /// `eprintln!(…); process::exit(1)` pair every command's error sites
    /// were written as.
    pub fn fail(&self, message: impl Into<String>) -> ! {
        self.error(CommandError::new(message)).exit()
    }

    /// A report command's report, on stdout.
    pub fn report(&mut self, lines: &[Line]) {
        let written = self.stdout.write(&mut std::io::stdout().lock(), lines);
        stdout_written(written);
    }

    /// One line of a report command's report, on stdout, as text already
    /// painted for stdout where it is coloured (`camdl show`'s fields).
    pub fn report_line(&self, text: impl std::fmt::Display) {
        self.report_raw(format_args!("{text}\n"));
    }

    /// Text of a report command's report, on stdout, written as given: a
    /// rendered table or page that ends in its own newline.
    pub fn report_raw(&self, text: impl std::fmt::Display) {
        let written = write!(std::io::stdout().lock(), "{text}");
        stdout_written(written);
    }

    /// The writer a command's data goes to: stdout, unpainted.
    pub fn data(&self) -> Data {
        Data(std::io::stdout().lock())
    }

    /// One line of a data command's data, on stdout.
    pub fn data_line(&self, text: impl std::fmt::Display) {
        self.data_raw(format_args!("{text}\n"));
    }

    /// [`data_line`](Self::data_line) without the newline.
    pub fn data_raw(&self, text: impl std::fmt::Display) {
        let written = write!(std::io::stdout().lock(), "{text}");
        stdout_written(written);
    }
}

/// An `Out` for code that holds no handle: a helper below every command's
/// handle that warns or exits where it stands (an observation projecting a
/// compartment the model lacks, a sink's failed side file). It writes what
/// the command's own `Out` would, painted by the environment's policy; a
/// function that can be handed the command's `Out` takes it instead.
pub fn detached() -> Out {
    Out::new(false)
}

/// stdout for data. A closed pipe ends the command with status 0; any other
/// failed write is fatal, as it is for `print!`.
pub struct Data(std::io::StdoutLock<'static>);

impl Write for Data {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.0.write(buf);
        if let Err(e) = &n {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }
        n
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let r = self.0.flush();
        stdout_written(r);
        Ok(())
    }
}

/// The outcome of a write to stdout: a closed pipe is a normal exit, anything
/// else is fatal.
fn stdout_written(r: std::io::Result<()>) {
    match r {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => std::process::exit(0),
        Err(e) => panic!("failed printing to stdout: {e}"),
    }
}

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

    /// A notice said while the command runs, rendered (`warning: …`) and
    /// written whole.
    pub fn notice(&self, n: &notice::Notice) {
        self.progress(&n.render());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_error_exits_one_unless_it_says_otherwise() {
        let e = CommandError::new("error: no completed method leaf");
        assert_eq!(e.code, ExitCode::FAILURE);
        assert_eq!(e.code.0, 1);
        assert_eq!(e.hint, None);
        let out = Out::new(true);
        assert_eq!(out.error(CommandError { code: ExitCode(2), ..e }), ExitCode(2));
    }

    /// What a method records on `Out` — the checks it showed, the findings
    /// pushed from the command's thread and from a worker's — comes back
    /// whole when it ends, and nothing is recorded once it has.
    #[test]
    fn a_method_hands_back_what_it_recorded() {
        use crate::fit::report::{Check, RunHeader};
        use sim::inference::diagnostic::DiagnosticKind;
        let header = RunHeader {
            config: "f.toml".into(), method: "pmmh".into(), model: "m.camdl".into(),
            estimate: vec!["beta".into()], fixed: Vec::new(), starts: "single".into(),
            output: "results/fits/f".into(),
        };
        let mut out = Out::new(true);
        let worker = out.begin_method(header.clone(), "pmmh", Arc::new(io::HeartbeatGuard::inert()));
        out.check(Check::Starts { mode: "lhs".into(), params: vec!["beta".into()], note: "n".into() });
        out.finding(DiagnosticKind::InitialLoglikInfinite);
        std::thread::scope(|s| {
            s.spawn(|| worker.push(DiagnosticKind::InitialLoglikInfinite));
        });
        let record = out.end_method().expect("a method was begun");
        assert_eq!(record.header, header);
        assert_eq!(record.checks.len(), 1);
        assert_eq!(record.findings.len(), 2);
        assert!(out.end_method().is_none());
        out.finding(DiagnosticKind::InitialLoglikInfinite);
        assert_eq!(worker.snapshot().len(), 2, "a finding after the method ended went nowhere");
    }

    #[test]
    fn a_notice_is_kept_after_it_is_printed() {
        let mut out = Out::new(true);
        out.notice(notice::Notice::info("param-override", "x"));
        assert_eq!(out.notices().len(), 1);
        assert_eq!(out.notices()[0].code, "param-override");
    }
}
