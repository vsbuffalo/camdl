//! Commands write through `Out` and through nothing else.
//!
//! `out::Out` is the one handle a command writes through (proposal
//! `docs/dev/proposals/2026-11-08-cli-reporting.md`, "The reporting
//! interface"): stdout carries data or a report command's report, everything
//! else goes to stderr, and colour is applied when a stream is written. A
//! print macro anywhere else in `cli/src/` writes past it: to a stream the
//! rule does not allow, unpainted or painted by another policy, interleaved
//! with a chain worker's progress line, and recorded nowhere. Taking a
//! process stream directly (`std::io::stdout()`, `std::io::stderr()`) does the
//! same with more words, and `dbg!` is a print macro. This test fails on any
//! of them, so a new message has to go through `Out`.
//!
//! The scan is textual and covers the tests inside `cli/src/` too. The one
//! directory allowed is `out/`, which owns the streams. `style.rs`, the
//! `--help` colouring, writes nothing: clap prints the help.

use std::path::{Path, PathBuf};

/// The directory that owns the process's streams.
const ALLOWED_DIR: &str = "out/";

/// A print macro, `dbg!`, or a process stream taken directly. Each pattern
/// is matched where the preceding character cannot continue an identifier,
/// so `eprintln!` does not also count as `println!` and `paint::stderr()`
/// (the stderr painter) is not `io::stderr()`.
const FORBIDDEN: &[&str] = &["println!", "print!", "eprintln!", "eprint!", "dbg!", "io::stdout()", "io::stderr()"];

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

/// The places `pat` occurs in `line` as a macro or call of its own: not
/// preceded by an identifier character (so `eprintln!` holds no `println!`)
/// or, for a stream, by anything but `std::`/`::`'s colon or a space.
fn hits(line: &str, pat: &str) -> bool {
    line.match_indices(pat).any(|(i, _)| {
        let before = line[..i].chars().next_back();
        !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

#[test]
fn no_print_outside_out() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(files.len() > 50, "the scan found {} files under {}; the walk is broken", files.len(), src.display());

    let mut found = Vec::new();
    for path in &files {
        let rel = path.strip_prefix(&src).unwrap().to_string_lossy().replace('\\', "/");
        if rel.starts_with(ALLOWED_DIR) {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap();
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for pat in FORBIDDEN {
                if hits(code, pat) {
                    found.push(format!("{rel}:{}: `{pat}` in: {}", i + 1, line.trim()));
                }
            }
        }
    }
    assert!(found.is_empty(),
        "a command writes through `out::Out` (`say`, `fail`, `milestone`, `report_line`, `data_line`, \
         …), not a print macro or a process stream taken directly:\n{}", found.join("\n"));
}

/// The scan is not vacuous: `out/`, which owns the streams, does print, so a
/// pattern list or matcher that could never fire would fail here.
#[test]
fn the_patterns_match_out() {
    let out = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/out/mod.rs")).unwrap();
    assert!(out.lines().any(|l| FORBIDDEN.iter().any(|p| hits(l, p))),
        "out/mod.rs writes to the streams, so at least one pattern must match it");
    assert!(!hits("    eprintln!(\"x\");", "println!"), "`eprintln!` is not `println!`");
    assert!(!hits("let p = paint::stderr();", "io::stderr()"), "the stderr painter is not the stream");
    assert!(hits("std::io::stderr().lock()", "io::stderr()"));
}
