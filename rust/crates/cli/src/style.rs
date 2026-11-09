//! Centralized terminal styling.
//!
//! Three surfaces:
//!
//!   0. [`Painter`] — the one colour policy ([`Painter::for_stream`]) and the
//!      only place an escape code is written. Text that may be coloured is
//!      built as [`Line`]s of [`Span`]s, each carrying an optional semantic
//!      [`Style`], and the painter for the stream it goes to applies the codes
//!      when it writes them. Call sites that still build strings use
//!      [`Painter::paint`], or [`stderr`] for the stderr painter.
//!
//!   1. `bold / dim` — small text-coloring helpers, through the stderr
//!      painter.
//!
//!   2. `colored_help!` macro + `colorize_after_help` — turn the
//!      `#[command(after_help = "...")]` literal into a styled string
//!      according to four conventions our help blocks already follow:
//!
//!         Rule 1 (section heading → bold):
//!             A line whose first character is at column 0, contains
//!             only alphanumerics+spaces, and ends with `:`.
//!             Examples: `Examples:`, `Notes:`, `Common workflows:`.
//!
//!         Rule 2 (comment → dim, with cyan inline-code spans):
//!             A line whose first non-whitespace character is `#`.
//!             Plain segments dim; backtick-delimited spans cyan.
//!
//!         Rule 3 (inline code spans → cyan):
//!             Backtick-delimited spans render cyan; backticks
//!             stripped from output. Applies inside both Rule 2
//!             (comments) and Rule 4 (default lines).
//!
//!         Rule 4 (default):
//!             Anything else stays the original color. Rule 3 still
//!             highlights backtick spans within.
//!
//! The macro is `#[macro_export]` at the crate root, with a per-call-site
//! `OnceLock<String>` so the colorized form is built at most once per
//! distinct after_help value, on first access (typically `Cli::parse()`).

use std::io::{IsTerminal, Write};
use std::sync::OnceLock;

// ─── Styled text and the colour policy ────────────────────────────────────────

/// What a span of text means, not what colour it is. The painter maps each to
/// one escape code; nothing else in the crate names a code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// A passing check or a good value (✓).
    Ok,
    /// A warning, or a value outside its healthy range but not an error (~, ⚠).
    Warn,
    /// An error or a failing check (✗).
    Err,
    /// A labelled aside (`holdout:`, `ic-free inference:`).
    Info,
    Bold,
    Dim,
    /// Inline code: a flag, a key, a command.
    Code,
}

impl Style {
    /// The SGR parameter for this style.
    fn sgr(self) -> &'static str {
        match self {
            Style::Ok => "32",
            Style::Warn => "33",
            Style::Err => "31",
            Style::Info | Style::Code => "36",
            Style::Bold => "1",
            Style::Dim => "2",
        }
    }
}

/// A run of text with one optional style.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub style: Option<Style>,
}

impl Span {
    pub fn plain(text: impl Into<String>) -> Self {
        Span { text: text.into(), style: None }
    }
    pub fn styled(text: impl Into<String>, style: Style) -> Self {
        Span { text: text.into(), style: Some(style) }
    }
}

/// One line of output, without its newline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Line(pub Vec<Span>);

impl Line {
    /// The line's text with no styling, as it reads with colour off.
    pub fn text(&self) -> String {
        self.0.iter().map(|s| s.text.as_str()).collect()
    }
}

/// The two streams a command writes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

/// Applies colour when text is written, by one policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Painter {
    color: bool,
}

impl Painter {
    /// The colour policy, highest precedence first: the `--no-color` flag
    /// (off), then `NO_COLOR` (off; no-color.org), then `CLICOLOR_FORCE`, any
    /// value (on, the convention `ls`, `grep` and `git` use when piped through
    /// `less -R`), then whether `stream` is a terminal.
    pub fn for_stream(stream: Stream, no_color_flag: bool) -> Self {
        let color = if no_color_flag || std::env::var_os("NO_COLOR").is_some() {
            false
        } else if std::env::var_os("CLICOLOR_FORCE").is_some() {
            true
        } else {
            match stream {
                Stream::Stdout => std::io::stdout().is_terminal(),
                Stream::Stderr => std::io::stderr().is_terminal(),
            }
        };
        Painter { color }
    }

    /// A painter that writes no escape codes, for text bound for a file or a
    /// test.
    pub fn plain() -> Self {
        Painter { color: false }
    }

    pub fn color(&self) -> bool {
        self.color
    }

    /// `s` in `style`, or `s` unchanged when colour is off.
    pub fn paint(&self, s: &str, style: Style) -> String {
        if self.color {
            format!("\x1b[{}m{s}\x1b[0m", style.sgr())
        } else {
            s.to_string()
        }
    }

    /// One line as text, styled by this painter, without its newline.
    pub fn render(&self, line: &Line) -> String {
        line.0.iter()
            .map(|span| match span.style {
                Some(style) => self.paint(&span.text, style),
                None => span.text.clone(),
            })
            .collect()
    }

    /// Write `lines`, each followed by a newline.
    pub fn write(&self, w: &mut impl Write, lines: &[Line]) -> std::io::Result<()> {
        for line in lines {
            writeln!(w, "{}", self.render(line))?;
        }
        Ok(())
    }
}

/// The painter for stderr when the command has no `--no-color` flag. Sampled
/// once per process.
pub fn stderr() -> Painter {
    static CACHED: OnceLock<Painter> = OnceLock::new();
    *CACHED.get_or_init(|| Painter::for_stream(Stream::Stderr, false))
}

/// The painter for stdout when the command has no `--no-color` flag. Sampled
/// once per process.
pub fn stdout() -> Painter {
    static CACHED: OnceLock<Painter> = OnceLock::new();
    *CACHED.get_or_init(|| Painter::for_stream(Stream::Stdout, false))
}

/// `warning:` as a stderr message's leading tag.
pub fn warning_tag() -> String { stderr().paint("warning:", Style::Warn) }

/// `note:` as a stderr message's leading tag.
pub fn note_tag() -> String { stderr().paint("note:", Style::Warn) }

/// True when the stderr painter colours.
pub fn enabled() -> bool {
    stderr().color()
}

// ─── Single-string helpers ────────────────────────────────────────────────────

pub fn bold(s: &str) -> String { stderr().paint(s, Style::Bold) }
pub fn dim (s: &str) -> String { stderr().paint(s, Style::Dim) }

// ─── Help-block colorizer ─────────────────────────────────────────────────────

/// Apply the four-rule colorization to a multi-line help block.
/// Returns `raw.to_string()` unchanged when color is disabled.
pub fn colorize_after_help(raw: &str) -> String {
    // clap writes help to stdout, so stdout's painter decides.
    let painter = stdout();
    if !painter.color() {
        // Backticks must still be stripped on the no-color path so the
        // user-visible help text is the same shape regardless of color.
        return raw.lines().map(strip_backticks).collect::<Vec<_>>().join("\n");
    }
    raw.lines().map(|l| style_line(&painter, l)).collect::<Vec<_>>().join("\n")
}

fn is_section_heading(line: &str) -> bool {
    !line.is_empty()
        && !line.starts_with(char::is_whitespace)
        && line.ends_with(':')
        && line[..line.len() - 1].chars().all(|c| c.is_alphanumeric() || c == ' ')
}

fn style_line(painter: &Painter, line: &str) -> String {
    if is_section_heading(line) {
        return painter.paint(line, Style::Bold);
    }
    let stripped = line.trim_start();
    if stripped.starts_with('#') {
        let indent = &line[..line.len() - stripped.len()];
        return format!("{indent}{}", backticks_dim(painter, stripped));
    }
    backticks_default(painter, line)
}

/// Plain text + cyan backtick spans (backticks stripped).
fn backticks_default(painter: &Painter, line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_code = false;
    for chunk in line.split('`') {
        if in_code { out.push_str(&painter.paint(chunk, Style::Code)); }
        else { out.push_str(chunk); }
        in_code = !in_code;
    }
    out
}

/// Dim text + cyan backtick spans (cyan beats dim; backticks stripped).
fn backticks_dim(painter: &Painter, line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_code = false;
    for chunk in line.split('`') {
        if in_code { out.push_str(&painter.paint(chunk, Style::Code)); }
        else if !chunk.is_empty() { out.push_str(&painter.paint(chunk, Style::Dim)); }
        in_code = !in_code;
    }
    out
}

/// No-color companion: same backticks-stripped output without any styling.
fn strip_backticks(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for (i, chunk) in line.split('`').enumerate() {
        if i > 0 { /* boundary; backtick consumed */ }
        out.push_str(chunk);
    }
    out
}

/// Wrap an `after_help` literal so it renders styled in TTY contexts and
/// plain otherwise. The styled form is built at most once per call site
/// (per-site `OnceLock<String>`).
///
/// Usage:
///
/// ```ignore
/// #[command(after_help = colored_help!("\
/// Examples:
///   # Run a thing
///   camdl thing --flag
/// "))]
/// pub struct Args { ... }
/// ```
#[macro_export]
macro_rules! colored_help {
    ($raw:expr) => {{
        static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        CACHED.get_or_init(|| $crate::style::colorize_after_help($raw)).as_str()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Colour off writes the text and nothing else; colour on wraps each
    /// styled span, and only the styled ones, in its code and a reset.
    #[test]
    fn the_painter_styles_only_styled_spans_and_only_when_on() {
        let line = Line(vec![
            Span::plain("chain 0: "),
            Span::styled("✗ too low", Style::Err),
        ]);
        assert_eq!(Painter::plain().render(&line), "chain 0: ✗ too low");
        assert_eq!(line.text(), "chain 0: ✗ too low");
        let on = Painter { color: true };
        assert_eq!(on.render(&line), "chain 0: \x1b[31m✗ too low\x1b[0m");
        assert_eq!(on.paint("ok", Style::Ok), "\x1b[32mok\x1b[0m");
        assert_eq!(Painter::plain().paint("ok", Style::Ok), "ok");

        let mut buf = Vec::new();
        Painter::plain().write(&mut buf, &[line.clone(), Line::default()]).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "chain 0: ✗ too low\n\n");
    }

    /// The flag wins over every environment variable, so `--no-color` is off
    /// even under `CLICOLOR_FORCE`. (The environment arms are not exercised
    /// here: setting a process-wide variable races the other tests.)
    #[test]
    fn the_no_color_flag_is_off_on_both_streams() {
        assert!(!Painter::for_stream(Stream::Stdout, true).color());
        assert!(!Painter::for_stream(Stream::Stderr, true).color());
    }

    #[test]
    fn section_heading_matches() {
        assert!(is_section_heading("Examples:"));
        assert!(is_section_heading("Common workflows:"));
        assert!(is_section_heading("Notes:"));
        assert!(!is_section_heading("  Examples:"));      // indented
        assert!(!is_section_heading("Source: github"));    // text after colon
        assert!(!is_section_heading("Examples"));          // no colon
        assert!(!is_section_heading(""));
    }

    #[test]
    fn strip_backticks_drops_them() {
        assert_eq!(strip_backticks("see `--resume`."), "see --resume.");
        assert_eq!(strip_backticks("plain text"), "plain text");
    }

    #[test]
    fn no_color_path_strips_backticks_only() {
        // Forcing the no-color branch via the public API would require
        // mocking a global; instead exercise the shape-stable branch
        // directly. The public guarantee: color-off output has zero
        // ANSI escape bytes.
        let raw = "Examples:\n  # see `--resume`\n  camdl X";
        let out: String = raw.lines().map(strip_backticks).collect::<Vec<_>>().join("\n");
        assert!(!out.contains('\x1b'));
        assert!(!out.contains('`'));
    }
}
