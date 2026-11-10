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

use std::path::{Path, PathBuf};

/// The module allowed to write to stderr.
const EXEMPT: &str = "trace.rs";

/// The macros the library may not call.
const FORBIDDEN: &[&str] = &["println!", "print!", "eprintln!", "eprint!", "dbg!", "warn!", "info!"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten() {
        let p = entry.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `src` with comments blanked to spaces (newlines kept, so line numbers
/// hold). String and char literals are kept as they are, and are skipped
/// over so a `//` inside one is not read as a comment.
fn without_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while i < b.len() {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    let c = src[i..].chars().next().unwrap();
                    out.push(blank(c));
                    i += c.len_utf8();
                }
            }
        } else if b[i] == b'"' {
            // A string literal, escapes honoured; raw strings with `#`
            // delimiters do not occur in sim's print sites.
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    out.push('\\');
                    i += 1;
                }
                let c = src[i..].chars().next().unwrap();
                out.push(c);
                i += c.len_utf8();
            }
            if i < b.len() {
                out.push('"');
                i += 1;
            }
        } else if b[i] == b'\'' && b.get(i + 2) == Some(&b'\'') {
            // A one-byte char literal such as '"', which would otherwise open
            // a string.
            out.push_str(&src[i..i + 3]);
            i += 3;
        } else if b[i] == b'\'' && b.get(i + 1) == Some(&b'\\') && b.get(i + 3) == Some(&b'\'') {
            out.push_str(&src[i..i + 4]);
            i += 4;
        } else {
            let c = src[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// `code` (already without comments) with every `#[cfg(test)]` item blanked:
/// from the attribute to the end of the item it annotates — the matching `}`
/// of its first `{`, or its `;` when it has no body.
fn without_test_items(code: &str) -> String {
    let mut out: Vec<char> = code.chars().collect();
    let text: String = code.to_string();
    let mut search = 0;
    while let Some(off) = text[search..].find("#[cfg(test)]") {
        let start = search + off;
        let mut depth = 0usize;
        let mut end = text.len();
        for (j, c) in text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + j + 1;
                        break;
                    }
                }
                ';' if depth == 0 => {
                    end = start + j + 1;
                    break;
                }
                _ => {}
            }
        }
        let (cs, ce) = (text[..start].chars().count(), text[..end].chars().count());
        for c in &mut out[cs..ce] {
            if *c != '\n' {
                *c = ' ';
            }
        }
        search = end;
    }
    out.into_iter().collect()
}

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
