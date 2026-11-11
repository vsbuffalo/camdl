//! The stream layer: how anything reaches the terminal. Knows nothing about
//! fits.
//!
//! It carries both streams, not only stdout: stdout holds data and a report
//! command's report (`fit summary`, `fit diff`), and stderr holds everything
//! else — progress, milestones, notices, warnings, errors and `fit run`'s
//! end-of-run block. "out" names the layer, not the stream.

pub mod events;
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
    /// The method running now, from [`begin_method`](Self::begin_method) to
    /// [`end_method`](Self::end_method).
    method: Option<RunningMethod>,
    /// The running method's event log, from `begin_method` to
    /// [`close_events`](Self::close_events): it outlives `end_method`, since
    /// the end-of-run report, its last line, is built after the method's
    /// record is handed back.
    events: Option<events::EventLog>,
    /// The store claim on the running method's leaf, from
    /// [`hold_claim`](Self::hold_claim) to [`take_claim`](Self::take_claim).
    /// An error reported while it is held fails it, so the leaf's
    /// `run.json` reaches `failed`, recording the error, on every error
    /// path, `process::exit` (which runs no destructor) included. Behind a
    /// lock because the chain workers report errors through a shared `&Out`.
    claim: std::sync::Mutex<Option<HeldClaim>>,
}

/// The operation context of a streamed run: the store claim on its leaf, and
/// what is running in it, which a failure records (`runid::Failure`). Owned
/// by the command's `Out` for the run's span — one per `fit run` cell, or per
/// survey — and never by anything process-wide.
pub struct HeldClaim {
    claim: crate::resolve::ResolvedClaim,
    /// The fit method (`pmmh`, `if2`, …) or command (`survey`).
    operation: String,
}

/// What `Out` keeps for the method that is running: the run's opening block,
/// the checks it has shown, the collector its findings go to, its heartbeat,
/// which an error marks failed, and its leaf, where its records are written.
struct RunningMethod {
    header: crate::fit::report::RunHeader,
    checks: Vec<crate::fit::report::Check>,
    collector: Arc<DiagnosticCollector>,
    heartbeat: Arc<io::HeartbeatGuard>,
    leaf: std::path::PathBuf,
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
            method: None,
            events: None,
            claim: std::sync::Mutex::new(None),
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

    /// Start recording a method: its opening block and its heartbeat, and
    /// its event log, `leaf/events.jsonl`, which starts with a `begin` line
    /// carrying the healthy band of `acceptance`, the kernel behind the
    /// method's acceptance diagnostics (`None` for a method without them).
    /// Returns the collector its findings go to, for the chain workers that
    /// push from their own threads; the command's own thread pushes through
    /// [`finding`](Self::finding). Every finding pushed is also an event.
    ///
    /// # Errors
    /// The event log cannot be created or its first line written.
    pub fn begin_method(
        &mut self,
        header: crate::fit::report::RunHeader,
        method: &str,
        acceptance: Option<sim::inference::diagnostic::AcceptanceKernel>,
        heartbeat: Arc<io::HeartbeatGuard>,
        leaf: &std::path::Path,
    ) -> Result<Arc<DiagnosticCollector>, String> {
        let log = events::EventLog::create(leaf)?;
        log.record(&events::Event::Begin {
            method, header: &header, acceptance: acceptance.map(|k| k.band()),
        });
        log.result()?;
        let collector = Arc::new({
            let log = log.clone();
            DiagnosticCollector::with_observer(method, move |d| log.record(&events::Event::Finding { finding: d }))
        });
        self.method = Some(RunningMethod {
            header, checks: Vec::new(), collector: collector.clone(), heartbeat,
            leaf: leaf.to_path_buf(),
        });
        self.events = Some(log);
        Ok(collector)
    }

    /// Hold the store claim on the method's leaf until the run finalizes it
    /// ([`take_claim`](Self::take_claim)). Until then, an error reported
    /// through this handle marks the leaf `failed` and releases its lock.
    pub fn hold_claim(&mut self, claim: crate::resolve::ResolvedClaim, operation: impl Into<String>) {
        *self.claim.get_mut().unwrap_or_else(|e| e.into_inner()) =
            Some(HeldClaim { claim, operation: operation.into() });
    }

    /// Hand back the claim [`hold_claim`](Self::hold_claim) took, for the
    /// run to finalize.
    pub fn take_claim(&mut self) -> Option<crate::resolve::ResolvedClaim> {
        self.claim.get_mut().unwrap_or_else(|e| e.into_inner()).take().map(|h| h.claim)
    }

    /// The running method's event log, for the chain tasks that record their
    /// progress in it ([`progress::Task::recording`]); `None` when no method
    /// is running.
    pub fn events(&self) -> Option<events::EventLog> {
        self.events.clone()
    }

    /// Whether every event so far was written: the first failure, if one
    /// was not. A run checks this before it calls its method done.
    pub fn events_result(&self) -> Result<(), String> {
        self.events.as_ref().map_or(Ok(()), events::EventLog::result)
    }

    /// Close the method's event log after its last line, and say whether
    /// every event was written.
    pub fn close_events(&mut self) -> Result<(), String> {
        self.events.take().map_or(Ok(()), |log| log.result())
    }

    /// Record a check the run has just shown. The check's own renderer draws
    /// its lines (`pf_noise::report`, `util::print_observations_summary`, the
    /// caveat banner, the starts note), from context the `Check` does not
    /// carry, and the run writes them through this `Out`; this keeps the
    /// value, so the run's report carries what it showed.
    pub fn check(&mut self, c: crate::fit::report::Check) {
        if let Some(m) = self.method.as_mut() {
            if let Some(log) = &self.events {
                log.record(&events::Event::Check { check: &c });
            }
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

    /// The running method's leaf, where its `diagnostics.json` and event
    /// log go; `None` when no method is running.
    pub fn method_leaf(&self) -> Option<&std::path::Path> {
        self.method.as_ref().map(|m| m.leaf.as_path())
    }

    /// Record a finding, from the command's thread.
    pub fn finding(&self, kind: sim::inference::diagnostic::DiagnosticKind) {
        if let Some(m) = &self.method {
            m.collector.push(kind);
        }
    }

    /// The method ran to its end: its heartbeat's terminal record is `done`,
    /// provisionally. An error reported before
    /// [`method_committed`](Self::method_committed) (the leaf's finalize
    /// failing, say) replaces it with `failed`, so `progress.json` never says
    /// `done` for a run whose `run.json` says `failed`.
    pub fn method_done(&self) {
        if let Some(m) = &self.method {
            m.heartbeat.done();
        }
    }

    /// The method's leaf is committed `completed`: the `done` in
    /// `progress.json` is final. The leaf's manifest has hashed that file, so
    /// no later error rewrites it.
    pub fn method_committed(&self) {
        if let Some(m) = &self.method {
            m.heartbeat.commit();
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
    /// (`View::RunEnd`), on stderr, and recorded as the event log's
    /// `method_report` line.
    pub fn method_report(&self, o: &crate::fit::report::build::MethodOutcome) {
        use crate::fit::report::build::MethodOutcome;
        use crate::fit::report::render::{render_report, View};
        self.stderr_lines(&render_report(o, View::RunEnd));
        if let Some(log) = &self.events {
            let outcome = match o {
                MethodOutcome::Completed(report) => events::Outcome::Completed { report },
                MethodOutcome::Failed { error, .. } => events::Outcome::Failed { error },
            };
            log.record(&events::Event::MethodReport(outcome));
        }
    }

    /// A notice (`info: …`, `warning: …`), on stderr. The run's record of
    /// its notices is `run.json`'s display inputs, which the store's sink
    /// builds from the same resolution (`params_resolver::notices`,
    /// `obs_anchor::spline_hold_notices`), so `Out` writes them and does not
    /// keep them.
    pub fn notice(&self, n: &notice::Notice) {
        self.say(n.render());
    }

    /// Each of `notices`, in order, through [`notice`](Self::notice).
    pub fn notices(&self, notices: &notice::Notices) {
        for n in notices.iter() {
            self.notice(n);
        }
    }

    /// Report `e` on stderr and return the status the command exits with.
    ///
    /// A held store claim is failed first: the leaf's `run.json` becomes
    /// `failed` and records the message as its `failure`, the terminal
    /// status its event log and `progress.json` are about to record. A
    /// record that could not be written is added to the message.
    pub fn error(&self, mut e: CommandError) -> ExitCode {
        let held = self.claim.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(held) = held {
            fail_claim(held.claim, &held.operation, &mut e.message);
        }
        if let Some(m) = &self.method {
            // The same sentence a reader of stderr gets, for a reader holding
            // only `progress.json`.
            m.heartbeat.failed(e.message.as_str());
        }
        self.say(one_head(&e.message));
        if let Some(hint) = &e.hint {
            for line in hint.lines() {
                self.say(format_args!("  {line}"));
            }
        }
        e.code
    }

    /// [`fail`](Self::fail) for a command that holds `claim` itself rather
    /// than through [`hold_claim`](Self::hold_claim): the claim is failed
    /// with `message` first, so its leaf's `run.json` records `failed` and
    /// why before the process exits (`process::exit` runs no destructor).
    pub fn fail_with_claim(
        &self,
        claim: crate::resolve::ResolvedClaim,
        operation: &str,
        message: impl Into<String>,
    ) -> ! {
        let mut message = message.into();
        fail_claim(claim, operation, &mut message);
        self.fail(message)
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

/// Fail `claim` with `message` (`run.json` → `failed`, recording it; lock
/// released), adding to `message` when that record could not be written.
fn fail_claim(claim: crate::resolve::ResolvedClaim, operation: &str, message: &mut String) {
    let dir = claim.dir().to_path_buf();
    let failure = runid::Failure { message: message.clone(), operation: operation.to_string() };
    if let Err(err) = claim.fail(failure) {
        message.push_str(&format!(
            "\n  and {} could not be marked failed: {err}", dir.join("run.json").display()));
    }
}

/// `message` without its leading `error: ` when what follows is itself a
/// headed error report, so it is headed once.
///
/// Commands head the cause their helpers return (`error: {e}`), and every
/// helper returns an unheaded cause except one: a compile that camdlc refused
/// returns camdlc's own report, which opens with `error: …` or `error[E310]: …`
/// (painted, when camdl passes its colour decision on). Collapsing it here,
/// where every command's error is written, covers every site that relays a
/// compile error, rather than each of them sniffing the cause.
fn one_head(message: &str) -> &str {
    let Some(rest) = message.strip_prefix("error: ") else { return message };
    // Past any colour camdlc painted its head with.
    let unpainted = paint::without_leading_sgr(rest);
    if unpainted.starts_with("error:") || unpainted.starts_with("error[") {
        rest
    } else {
        message
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

    /// A claim `Out` holds is failed by the error it reports: the leaf's
    /// `run.json` is `failed`, recording the error, and its lock released
    /// before the command exits, since `process::exit` runs no destructor. A
    /// claim taken back is the caller's again, and an error no longer
    /// touches it.
    #[test]
    fn an_error_reported_while_a_claim_is_held_marks_its_leaf_failed() {
        let root = tempfile::tempdir().unwrap();
        let store = runid::FsCasStore::new(root.path());
        let h = |b: u8| runid::ContentHash::from_bytes([b; 32]);
        let level = |name: &str, label: &str, b: u8| runid::LevelId {
            name: name.into(), label: label.into(), hash: h(b), schema_version: 1,
        };
        let claim = |run: u8| {
            let artifact = crate::resolve::ResolvedArtifact {
                kind: runid::ArtifactKind::FitMethod,
                levels: vec![level("fit", "f", 1), level("method", "pmmh", 2),
                    level("seed", &format!("seed_{run}"), run)],
                run_id: h(run),
                display_inputs: serde_json::Value::Null,
            };
            let meta = crate::resolve::RecordMeta::new("0.0", "m.camdl", None);
            match crate::resolve::begin_resolved_write(&store, root.path(), &artifact, &meta,
                crate::resolve::WriteMode::Streaming).unwrap()
            {
                crate::resolve::ResolvedWrite::Streaming(c) => c,
                crate::resolve::ResolvedWrite::Committed(_) => unreachable!(),
            }
        };
        let status = |dir: &std::path::Path| -> String {
            let rec: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
            rec["status"].as_str().unwrap().to_string()
        };

        let mut out = Out::new(true);
        let held = claim(3);
        let dir = held.dir().to_path_buf();
        out.hold_claim(held, "pmmh");
        assert_eq!(status(&dir), "running");
        assert_eq!(out.error(CommandError::new("error: the run failed")), ExitCode::FAILURE);
        assert_eq!(status(&dir), "failed");
        let rec: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(rec["failure"], serde_json::json!({"message": "error: the run failed", "operation": "pmmh"}),
            "run.json records why, and what was running");
        assert!(!dir.join(".lock").exists(), "the lock is released");

        let taken = claim(4);
        let dir = taken.dir().to_path_buf();
        out.hold_claim(taken, "pmmh");
        let taken = out.take_claim().expect("the held claim is handed back");
        out.error(CommandError::new("error: after the claim was taken back"));
        assert_eq!(status(&dir), "running", "a taken-back claim is the caller's");
        drop(taken);
    }

    /// `progress.json` says `done` before the leaf is finalized (it is one of
    /// the files the manifest hashes), so a finalize that then fails must turn
    /// it into `failed`, with the reason, to agree with the leaf's `run.json`.
    /// Once the leaf is committed `completed`, a later error leaves the
    /// hashed file alone.
    #[test]
    fn a_failure_after_done_but_before_commit_makes_progress_json_say_failed() {
        use crate::fit::report::RunHeader;
        let header = RunHeader {
            config: "f.toml".into(), method: "pmmh".into(), model: "m.camdl".into(),
            estimate: vec!["beta".into()], fixed: Vec::new(), starts: "single".into(),
            output: "results/fits/f".into(),
        };
        let state_after = |commit: bool| {
            let leaf = std::env::temp_dir()
                .join(format!("camdl_out_done_{}_{commit}", std::process::id()));
            std::fs::create_dir_all(&leaf).unwrap();
            let guard = io::HeartbeatGuard::new(io::Heartbeat::mcmc(
                leaf.clone(), 1, 4, std::time::Duration::from_millis(20), 1));
            let mut out = Out::new(true);
            out.begin_method(header.clone(), "pmmh", None, Arc::new(guard), &leaf)
                .expect("the event log is created");
            out.method_done();
            assert_eq!(io::read_progress(&leaf).unwrap().state, io::RunState::Done);
            if commit {
                out.method_committed();
            }
            out.error(CommandError::new("error: could not finalize fit method leaf: disk full"));
            let state = io::read_progress(&leaf).unwrap().state;
            std::fs::remove_dir_all(&leaf).ok();
            state
        };
        assert_eq!(state_after(false), io::RunState::Failed {
            reason: "error: could not finalize fit method leaf: disk full".into() });
        assert_eq!(state_after(true), io::RunState::Done);
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
        let leaf = std::env::temp_dir().join(format!("camdl_out_record_{}", std::process::id()));
        std::fs::create_dir_all(&leaf).unwrap();
        let worker = out.begin_method(header.clone(), "pmmh", None, Arc::new(io::HeartbeatGuard::inert()), &leaf)
            .expect("the event log is created");
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
        // The log holds the opening, the check and both findings, in order.
        assert_eq!(out.close_events(), Ok(()));
        let events: Vec<String> = std::fs::read_to_string(leaf.join("events.jsonl")).unwrap().lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(events, ["begin", "check", "finding", "finding"]);
        std::fs::remove_dir_all(&leaf).ok();
    }
}

#[cfg(test)]
mod one_head_tests {
    use super::one_head;

    #[test]
    fn a_relayed_report_keeps_only_its_own_head() {
        assert_eq!(one_head("error: error: nope.camdl: No such file or directory"),
            "error: nope.camdl: No such file or directory");
        assert_eq!(one_head("error: error[E310]: transition 'x' has no net effect"),
            "error[E310]: transition 'x' has no net effect");
        // Two escapes before the head, as camdlc paints it (red, then bold).
        use super::paint::{Painter, Style};
        let p = Painter::coloured();
        let painted = format!("{}: x", p.paint(&p.paint("error[E310]", Style::Bold), Style::Err));
        assert_eq!(one_head(&format!("error: {painted}")), painted);
    }

    #[test]
    fn any_other_message_is_unchanged() {
        for m in ["error: cannot read nope.toml", "error running pmmh: error: x",
                  "error: errors in fit.toml", "warning: error: x", "error loading trace"] {
            assert_eq!(one_head(m), m);
        }
    }
}
