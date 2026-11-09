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

use paint::{Line, Painter, Stream};

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

/// The one handle a command writes through, to either stream.
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
}

impl Out {
    /// A handle whose painters honour `no_color` (a command's `--no-color`),
    /// then the environment, then whether each stream is a terminal.
    pub fn new(no_color: bool) -> Self {
        Out {
            stdout: Painter::for_stream(Stream::Stdout, no_color),
            stderr: Painter::for_stream(Stream::Stderr, no_color),
            notices: Vec::new(),
        }
    }

    /// The stderr painter, for call sites that still build their own strings.
    pub fn stderr_painter(&self) -> Painter {
        self.stderr
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
    pub fn error(&mut self, e: CommandError) -> ExitCode {
        let _ = std::io::stdout().flush();
        eprintln!("{}", e.message);
        if let Some(hint) = &e.hint {
            for line in hint.lines() {
                eprintln!("  {line}");
            }
        }
        e.code
    }

    /// A report command's report, on stdout.
    pub fn report(&mut self, lines: &[Line]) {
        let written = self.stdout.write(&mut std::io::stdout().lock(), lines);
        stdout_written(written);
    }

    /// The writer a command's data goes to: stdout, unpainted.
    pub fn data(&mut self) -> Data {
        Data(std::io::stdout().lock())
    }
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
        let mut out = Out::new(true);
        assert_eq!(out.error(CommandError { code: ExitCode(2), ..e }), ExitCode(2));
    }

    #[test]
    fn a_notice_is_kept_after_it_is_printed() {
        let mut out = Out::new(true);
        out.notice(notice::Notice::info("param-override", "x"));
        assert_eq!(out.notices().len(), 1);
        assert_eq!(out.notices()[0].code, "param-override");
    }
}
