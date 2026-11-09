//! Styled text and the one colour policy.
//!
//! [`Painter`] is the one place an escape code is written. Text that may be
//! coloured is built as [`Line`]s of [`Span`]s, each carrying an optional
//! semantic [`Style`], and the painter for the stream it goes to applies the
//! codes when it writes them ([`Painter::for_stream`]). Call sites that still
//! build strings use [`Painter::paint`], or [`stderr`] for the stderr painter.

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
    #[cfg(test)]
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
    #[cfg(test)]
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
}
