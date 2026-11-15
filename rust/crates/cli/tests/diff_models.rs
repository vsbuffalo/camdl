//! Integration test: `camdl diff A B` on two model files.
//!
//! Proposal: docs/dev/proposals/2026-11-13-model-structure-diff.md
//!
//! Shells out to the built `camdl` binary. Which models: the fixture pairs
//! under `tests/fixtures/diff_pairs` (see `rust/crates/diff/tests/pairs.rs`,
//! which states what each diff must contain). This test checks what the
//! command adds to the library: that it compiles or reads each side, prints the
//! library's report unchanged, and exits 0 for identical, 1 for different and
//! 2 when it cannot report.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", bin.display());
    bin
}

fn pairs() -> PathBuf {
    repo_root().join("tests/fixtures/diff_pairs")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .current_dir(pairs())
        .output()
        .expect("run camdl diff")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is UTF-8")
}

/// Two compiled files that differ: the report is the committed text for that
/// pair, and the exit status is 1.
#[test]
fn two_compiled_models_that_differ_print_the_report_and_exit_1() {
    let out = run(&["diff", "ir/base.ir.json", "ir/let_body_changed.ir.json"]);
    assert_eq!(out.status.code(), Some(1), "stderr: {}", stderr(&out));
    // The committed text names the files by file name; the command names them
    // as typed.
    let expected = std::fs::read_to_string(pairs().join("expected/let_body_changed.txt"))
        .expect("the committed report")
        .replace("left   base.ir.json", "left   ir/base.ir.json")
        .replace("right  let_body_changed.ir.json", "right  ir/let_body_changed.ir.json");
    assert_eq!(stdout(&out), expected);
}

/// Two sources compiled by the command itself. The reformatted file is the
/// reference model written differently: identical, exit 0.
#[test]
fn two_sources_of_one_model_are_identical_and_exit_0() {
    let out = run(&["diff", "base.camdl", "reformatted.camdl"]);
    assert_eq!(out.status.code(), Some(0), "stdout: {}\nstderr: {}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains("Model: identical."), "{}", stdout(&out));
}

/// A source against a compiled file of the same model: the command compiles
/// the one and reads the other, and they are one model.
#[test]
fn a_source_against_its_own_compiled_file_is_identical() {
    let out = run(&["diff", "base.camdl", "ir/base.ir.json"]);
    assert_eq!(out.status.code(), Some(0), "stdout: {}\nstderr: {}", stdout(&out), stderr(&out));
}

/// A model whose reported quantity alone differs: the model is identical, the
/// reporting layer is not, and the exit status says the two differ.
#[test]
fn a_reporting_only_difference_exits_1_over_an_identical_model() {
    let out = run(&["diff", "ir/base.ir.json", "ir/quantity_changed.ir.json"]);
    assert_eq!(out.status.code(), Some(1), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("Model: identical."), "{text}");
    assert!(text.contains("Reported quantities and contrasts: different."), "{text}");
}

/// `--format json` is the library's report, readable back through its types.
#[test]
fn json_is_the_report_itself() {
    let out = run(&["diff", "ir/base.ir.json", "ir/let_renamed.ir.json", "--format", "json"]);
    assert_eq!(out.status.code(), Some(1), "stderr: {}", stderr(&out));
    let report: diff::DiffReport = serde_json::from_str(&stdout(&out)).expect("stdout is a DiffReport");
    assert_eq!(report.schema_version, diff::SCHEMA_VERSION);
    assert_eq!(report.model_outcome, diff::ModelOutcome::WrittenDifferently);
    assert!(report.model_identical);
    assert_eq!(report.left.name, "ir/base.ir.json");
}

/// A compiled file with no declared section cannot be diffed. The command says
/// which side and exits 2; it does not fall back to comparing hashes.
#[test]
fn a_model_without_a_declared_section_is_refused_with_exit_2() {
    let dir = std::env::temp_dir().join(format!("camdl-diff-bare-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create a temporary directory");
    let text = std::fs::read_to_string(pairs().join("ir/base.ir.json")).expect("the compiled reference model");
    // The declared section is the envelope's last member; cut it off as text,
    // so the model's bytes are untouched.
    let member = ",\"declared\":";
    assert_eq!(text.matches(member).count(), 1, "the fixture has one declared member");
    let cut = text.find(member).expect("the fixture has a declared member");
    let bare = dir.join("bare.ir.json");
    std::fs::write(&bare, format!("{}\n}}\n", &text[..cut])).expect("write the bare model");

    let out = run(&["diff", "ir/base.ir.json", bare.to_str().expect("a UTF-8 path")]);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(out.status.code(), Some(2), "stdout: {}\nstderr: {}", stdout(&out), stderr(&out));
    assert_eq!(stdout(&out), "", "a refusal prints no report");
    let err = stderr(&out);
    assert!(err.contains("unavailable for the right model"), "{err}");
    assert!(err.contains("no declared model is recorded"), "{err}");
}

/// A fit configuration is not a model file; the command says so, points at the
/// command that compares configurations, and exits 2.
#[test]
fn a_fit_configuration_is_refused_with_a_pointer() {
    let out = run(&["diff", "base.camdl", "fit.toml"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("fit.toml is not a .camdl or a compiled .ir.json"), "{err}");
    assert!(err.contains("camdl fit diff"), "{err}");
}
