//! No file under `score/src` names the parts of the standard library that
//! reach outside the process's own memory: files and other I/O, processes,
//! threads, environment variables, sockets and the clock.
//!
//! A lint cannot say this: clippy's `disallowed_methods` and
//! `disallowed_types` take single functions and types, and refuse a module
//! path. So this is a plain text search, and it is deliberately blunt:
//!
//! - It does not exempt comments or doc comments. Prose in `src/` must not
//!   spell these paths either; telling code from prose would need a Rust
//!   parser, and a test that can be fooled by a comment marker is weaker
//!   than one that asks for slightly different wording.
//! - It refuses `use std::{`, which would let a module be named without its
//!   `std::` prefix (`use std::{fs, …}` then `fs::read`).
//!
//! It does not catch an alias such as `extern crate std as s;`, or a path
//! through `core` or `alloc`; neither has the modules listed here, except
//! `core::time`, whose `Duration` reads no clock.

use std::path::{Path, PathBuf};

const FORBIDDEN: [&str; 8] = [
    "std::fs",
    "std::process",
    "std::thread",
    "std::env",
    "std::net",
    "std::time",
    "std::io",
    "use std::{",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_source_file_names_io_processes_threads_the_environment_or_the_clock() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(
        files.iter().any(|f| f.ends_with("lib.rs")),
        "found no lib.rs under {}; the search would pass vacuously",
        src.display()
    );

    let mut hits = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
        for (n, line) in text.lines().enumerate() {
            for pattern in FORBIDDEN {
                if line.contains(pattern) {
                    let shown = file.strip_prefix(&src).unwrap_or(file);
                    hits.push(format!("src/{}:{}: `{pattern}`: {}", shown.display(), n + 1, line.trim()));
                }
            }
        }
    }
    assert!(hits.is_empty(), "score/src names a forbidden path:\n{}", hits.join("\n"));
}
