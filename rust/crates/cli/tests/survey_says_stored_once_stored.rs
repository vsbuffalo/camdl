//! `camdl survey` says `stored <leaf>` only once the leaf is stored: after
//! every point is evaluated, the landscape written and the leaf finalized.
//! It printed the line right after claiming the leaf, before evaluating any
//! point, so a survey that failed part-way had already told the reader its
//! result was stored.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn binary() -> PathBuf {
    manifest().join("../../target/release/camdl")
}

/// The line index of the first stderr line containing `needle`.
fn line_of(stderr: &str, needle: &str) -> usize {
    stderr
        .lines()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing {needle:?}:\n{stderr}"))
}

#[test]
fn stored_follows_the_evaluation_and_the_landscape() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let tmp = std::env::temp_dir().join(format!("camdl_survey_stored_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let fixture = manifest().join("tests/fixtures/reporting");
    for f in ["sir.camdl", "if2.toml", "cases.tsv"] {
        std::fs::copy(fixture.join(f), tmp.join(f)).unwrap();
    }
    let out = Command::new(&bin)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.join(".ir-cache"))
        .current_dir(&tmp)
        .args([
            "survey", "sir.camdl", "--fit", "if2.toml", "--n-points", "8",
            "--eval-particles", "50", "--seed", "1", "--progress", "plain",
        ])
        .output()
        .expect("spawn camdl");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "survey failed:\n{stderr}");

    let stored = line_of(&stderr, "stored ");
    // The plain progress line is throttled, so which counts print varies;
    // any of them is a point being evaluated.
    let progress = stderr
        .lines()
        .position(|l| l.starts_with("survey ") && l.ends_with("/8"))
        .unwrap_or_else(|| panic!("no progress line:\n{stderr}"));
    let landscape = line_of(&stderr, "landscape.tsv (8 rows)");
    assert!(
        stored > progress && stored > landscape,
        "`stored` (line {stored}) must follow the evaluation progress (line {progress}) \
         and the landscape (line {landscape}):\n{stderr}"
    );
    assert!(
        Path::new(&tmp).join("results/surveys").is_dir(),
        "the leaf the line names exists"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
