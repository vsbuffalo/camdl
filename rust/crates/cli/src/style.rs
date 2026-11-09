//! `--help` colouring: the `colored_help!` macro and `colorize_after_help`
//! turn a `#[command(after_help = "...")]` literal into a styled string
//! according to four conventions our help blocks already follow:
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

use crate::out::paint::{stdout, Painter, Style};

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
