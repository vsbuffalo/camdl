//! `camdl compare --explain` serves the model-comparison methods guide from
//! the comparison surface itself (gh#806).
//!
//! The footer under a comparison table names concepts — the Jeffreys tiers,
//! the within-noise gate, se(Δ) and its small-T caveat — without room to
//! define them. `--explain` is where those definitions live at the terminal,
//! so the flag has to work for a reader who has not assembled a comparison
//! yet: no model paths, no config, no fits on disk.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

#[test]
fn explain_prints_the_methods_topic_without_any_model_arguments() {
    let out = Command::new(binary())
        .args(["compare", "--explain"])
        .output()
        .expect("spawn camdl compare --explain");

    assert_eq!(
        out.status.code(),
        Some(0),
        "`compare --explain` must exit 0 with no models given; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Jeffreys"),
        "the guide defines the tiers the evidence column prints:\n{stdout}"
    );
    assert!(
        stdout.contains("prequential"),
        "and the score the table is built on:\n{stdout}"
    );
}

/// The same text `camdl docs model-comparison` prints — one embedded copy,
/// two surfaces. A byte difference here means a second `include_str!` crept
/// in and the two can drift.
#[test]
fn explain_and_the_docs_topic_serve_the_same_bytes() {
    let via_compare = Command::new(binary())
        .args(["compare", "--explain"])
        .output()
        .expect("spawn camdl compare --explain");
    let via_docs = Command::new(binary())
        .args(["docs", "model-comparison"])
        .output()
        .expect("spawn camdl docs model-comparison");
    assert_eq!(via_compare.stdout, via_docs.stdout);
}

/// Model paths alongside `--explain` print the guide and run nothing: the
/// flag is a documentation surface, not a modifier on the comparison. The
/// paths here do not exist, so a run would have failed loudly.
#[test]
fn explain_with_model_paths_prints_and_runs_no_comparison() {
    let out = Command::new(binary())
        .args(["compare", "--explain", "/no/such/a", "/no/such/b"])
        .output()
        .expect("spawn camdl compare --explain with paths");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Jeffreys"), "{stdout}");
    // "Scored steps:" opens the rendered table's footer and appears nowhere
    // in the guide, so its absence says no comparison was rendered.
    assert!(
        !stdout.contains("Scored steps:"),
        "no comparison table is rendered:\n{stdout}"
    );
}
