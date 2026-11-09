//! Colour is applied in one place in the command-line crate.
//!
//! `out::paint::Painter` decides whether a stream gets colour (`--no-color`, then
//! `NO_COLOR`, then `CLICOLOR_FORCE`, then whether the stream is a terminal)
//! and is the only code that writes an escape sequence. A literal escape code
//! anywhere else in `cli/src/` writes colour into a pipe or a file whatever the
//! policy says, and a direct `owo_colors` call does the same through a
//! library. This test fails on either, so a new call site has to go through
//! the painter.
//!
//! The scan is textual and covers tests inside `cli/src/` too: a unit test
//! that needs an escape code builds its expectation through the painter.

use std::path::{Path, PathBuf};

/// The one file allowed to spell an escape code: it owns the policy.
const ALLOWED: &[&str] = &["out/paint.rs"];

/// Spellings of ESC in Rust source, and the colour library.
const FORBIDDEN: &[&str] = &["\\x1b", "\\x1B", "\\u{1b}", "\\u{1B}", "\\u{001b}", "\\u{001B}", "\\033", "owo_colors"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_escape_code_outside_the_painter() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(files.len() > 50, "the scan found {} files under {}; the walk is broken", files.len(), src.display());

    let mut hits = Vec::new();
    for path in &files {
        let rel = path.strip_prefix(&src).unwrap().to_string_lossy().into_owned();
        if ALLOWED.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap();
        for (i, line) in text.lines().enumerate() {
            for pat in FORBIDDEN {
                if line.contains(pat) {
                    hits.push(format!("{rel}:{}: `{pat}` in: {}", i + 1, line.trim()));
                }
            }
        }
    }
    assert!(hits.is_empty(),
        "colour must go through `style::Painter` (`paint`, `write`), not a literal escape code \
         or a direct colour-library call:\n{}", hits.join("\n"));
}

/// The scan is not vacuous: the painter's own file does spell the code it
/// writes, so a pattern list that could never match would fail here.
#[test]
fn the_patterns_match_the_painter() {
    let style = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/out/paint.rs")).unwrap();
    assert!(FORBIDDEN.iter().any(|p| style.contains(p)),
        "out/paint.rs writes the escape code, so at least one pattern must match it");
}
