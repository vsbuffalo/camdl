//! The environment is read in one module.
//!
//! Every environment variable camdl reads is read once, in `cli::env`, where it
//! is classified by its effect on results (proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`, part E).
//! This test reads the source of every library crate in the `camdl` binary and
//! fails naming each `env::var`, `env::var_os`, `env::vars` or `env::vars_os`
//! call outside that module, in non-test code. `sim` reads none at all: its
//! switches arrive as values (`sim::trace::configure`, and the
//! `sim::EvalSettings` each compiled model carries).
//!
//! One read is exempt by name, with its reason: `runid`'s host name on a
//! non-Unix platform (`COMPUTERNAME`), recorded on an execution record and in
//! no identity.
//!
//! Comments and `#[cfg(test)]` items are removed before the search, as in
//! `sim/tests/sim_prints_nothing.rs`, whose scanner this shares.

use std::path::Path;

#[path = "../../sim/tests/source_scan/mod.rs"]
mod source_scan;
use source_scan::{rust_files, without_comments, without_test_items};

/// The crates whose `src/` is compiled into the binary.
const CRATES: &[&str] = &["cli", "sim", "io", "ir", "runid", "score", "numerics", "declared", "diff"];

/// The module allowed to read the environment.
const THE_MODULE: &str = "cli/src/env.rs";

/// Reads allowed elsewhere: (file, the variable it names).
const EXEMPT: &[(&str, &str)] = &[("runid/src/store.rs", "COMPUTERNAME")];

const READS: &[&str] = &["env::var(", "env::var_os(", "env::vars(", "env::vars_os("];

/// Each environment read in `code`, as (line, the line's text).
fn reads(code: &str) -> Vec<(usize, String)> {
    code.lines()
        .enumerate()
        .filter(|(_, l)| READS.iter().any(|r| l.contains(r)))
        .map(|(n, l)| (n + 1, l.trim().to_string()))
        .collect()
}

#[test]
fn the_environment_is_read_only_in_cli_env() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut found = Vec::new();
    let mut n_files = 0;
    for name in CRATES {
        let src = crates.join(name).join("src");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        files.sort();
        n_files += files.len();
        for path in &files {
            let rel = format!("{name}/src/{}", path.strip_prefix(&src).unwrap().display());
            // A `tests.rs` is a test module (`#[cfg(test)] mod tests;` in its
            // parent, which `engine_fingerprint` checks for every one under a
            // `src/`), so it is not the library.
            if rel == THE_MODULE || path.file_name().is_some_and(|n| n == "tests.rs") {
                continue;
            }
            let code = without_test_items(&without_comments(&std::fs::read_to_string(path).unwrap()));
            for (line, text) in reads(&code) {
                let exempt = EXEMPT.iter().any(|(f, var)| *f == rel && text.contains(var));
                if !exempt {
                    found.push(format!("{rel}:{line}: {text}"));
                }
            }
        }
    }
    assert!(n_files > 200, "found {n_files} source files; the walk is broken");
    assert!(
        found.is_empty(),
        "an environment variable is read outside {THE_MODULE}; read it there, classify it, \
         and pass the value down:\n  {}",
        found.join("\n  ")
    );
}

/// The search finds a read in code, and not in a comment or a test item.
#[test]
fn the_search_rule() {
    let src = "fn f() {\n    let a = std::env::var_os(\"X\");\n    // std::env::var(\"Y\")\n    \
               let s = env::vars();\n}\n#[cfg(test)]\nmod tests {\n    fn g() { std::env::var(\"Z\"); }\n}\n";
    let hits: Vec<usize> = reads(&without_test_items(&without_comments(src))).into_iter().map(|(n, _)| n).collect();
    assert_eq!(hits, vec![2, 4]);
}
