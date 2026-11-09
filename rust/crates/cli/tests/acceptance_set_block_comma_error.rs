//! Acceptance test for finding minor-1 (CLI review): a comma-separated
//! `set = { a = 1, b = 2 }` block produces a bare `E001` syntax error that
//! points at the second key with no hint about the separator.
//!
//! Proposal: docs/dev/proposals/2026-05-28-simulate-batch-coherence-and-obs-ensembles.md
//!
//! Verified cause (at time of writing): `parser.mly:847` parses the block as
//! `list(scenario_kv_item)`; menhir `list(...)` has no separator and
//! `scenario_kv_item` (`parser.mly:880-883`) is `IDENT [ [idx,…] ] EQ expr`.
//! Entries are newline-separated by convention; a COMMA matches no production.
//!
//! Decision (proposal): keep newline separation (commas are reserved for
//! `[...]` lists and `(...)` arg lists), but emit a hint. This is the
//! "error messages are a feature" mandate.
//!
//! CLI-level: compile the model through the binary and assert the improved
//! error surfaces. The implementing agent should ALSO add the OCaml unit test
//! `compile_expect_error_code ~code:"E001" ~contains:"newline"` in
//! `ocaml/test/test_compiler.ml` (red→green) per the TDD discipline.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// Model with a one-line, comma-separated `set { }` block — the exact shape
/// the user tripped on.
fn write_model_with_comma_set(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S }

parameters {
  mu  : rate in [0.001, 10.0]
  nu  : rate in [0.001, 10.0]
}

init { S = 1000 }

transitions {
  death : S -->   @ mu * S
}

simulate { from = 0 'days  to = 20 'days }

scenarios {
  baseline { set = { mu = 0.1, nu = 0.2 } }
}
"#;
    std::fs::write(path, src).unwrap();
}

#[test]
fn comma_in_set_block_errors_with_separator_hint() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("comma.camdl");
    write_model_with_comma_set(&model);

    // simulate compiles the model first; the parse error surfaces before any
    // simulation. (-o to a temp path so we never reach output writing.)
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate", &model.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");

    assert!(!out.status.success(),
        "comma-separated set{{}} entries must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);

    // It is (and should remain) an E001 syntax error...
    assert!(stderr.contains("E001"),
        "expected E001 syntax error, got: {}", stderr);
    // ...but it must now explain the separator. Accept either phrasing of the
    // hint so the implementing agent has latitude on wording, as long as it
    // mentions newline separation.
    let lc = stderr.to_lowercase();
    assert!(lc.contains("newline") && lc.contains("comma"),
        "the E001 error for a comma in a set{{}} block must hint that entries \
         are separated by newlines, not commas. Got:\n{}", stderr);
}
