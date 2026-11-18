//! The simulation library prints nothing.
//!
//! `sim` returns values and records findings; the command-line crate decides
//! what a user sees (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`,
//! "The reporting interface"). This test reads every source file under
//! `sim/src/` and fails naming each print macro (`println!`, `print!`,
//! `eprintln!`, `eprint!`, `dbg!`) and each `log::warn!` / `log::info!` in
//! non-test code. The one exemption is `sim::trace` (`src/trace.rs`), the
//! environment-gated developer traces that must reach stderr from inside the
//! engine. `log::debug!` and `log::trace!` stay, for `RUST_LOG` debugging.
//!
//! Comments are not code, and a `#[cfg(test)]` item (a test module, a
//! test-only helper) is not the library, so both are removed before the
//! search.

use std::path::Path;

mod source_scan;
use source_scan::{rust_files, without_comments, without_test_items};

/// The module allowed to write to stderr.
const EXEMPT: &str = "trace.rs";

/// The macros the library may not call.
const FORBIDDEN: &[&str] = &["println!", "print!", "eprintln!", "eprint!", "dbg!", "warn!", "info!"];

/// Each forbidden macro call in `code`, as (line, macro). A macro name counts
/// when it is not the tail of a longer identifier: `eprintln!` does not also
/// count as `println!`, and `log::debug!` is not `debug!`'s concern here.
fn forbidden_calls(code: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    for (n, line) in code.lines().enumerate() {
        for m in FORBIDDEN {
            let mut from = 0;
            while let Some(off) = line[from..].find(m) {
                let at = from + off;
                let before = line[..at].chars().next_back();
                let is_tail = before.is_some_and(|c| c.is_alphanumeric() || c == '_');
                // `warn!` / `info!` count as `log::warn!` / `log::info!` or
                // bare (through a `use log::…`), not under another path.
                let log_only = matches!(*m, "warn!" | "info!");
                let qualified = line[..at].ends_with("log::");
                if !is_tail && (!log_only || qualified || before != Some(':')) {
                    hits.push((n + 1, if qualified { format!("log::{m}") } else { m.to_string() }));
                }
                from = at + m.len();
            }
        }
    }
    hits
}

#[test]
fn sim_prints_nothing_outside_trace() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    assert!(files.len() > 50, "found {} source files under {}; the walk is broken", files.len(), src.display());
    let mut found = Vec::new();
    for path in &files {
        if path.strip_prefix(&src).unwrap() == Path::new(EXEMPT) {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap();
        let code = without_test_items(&without_comments(&text));
        for (line, m) in forbidden_calls(&code) {
            found.push(format!("{}:{line}: {m}", path.strip_prefix(&src).unwrap().display()));
        }
    }
    assert!(found.is_empty(),
        "the sim library prints only through sim::trace (src/{EXEMPT}); found:\n  {}",
        found.join("\n  "));
}

/// The search finds what it is for and skips what it is not.
#[test]
fn the_search_rule() {
    let src = "fn f() {\n    eprintln!(\"x\");\n    log::warn!(\"y\");\n    log::debug!(\"z\");\n    \
               // println!(\"comment\")\n    let s = \"// not a comment\"; print!(\"{s}\");\n}\n\
               #[cfg(test)]\nmod tests {\n    fn g() { println!(\"test\"); }\n}\n";
    let code = without_test_items(&without_comments(src));
    let hits = forbidden_calls(&code);
    assert_eq!(hits, vec![
        (2, "eprintln!".to_string()),
        (3, "log::warn!".to_string()),
        (6, "print!".to_string()),
    ]);
}
