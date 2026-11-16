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

/// The line index of the first plain progress line of the survey bar:
/// `survey <pos>/8`, optionally followed by the best-loglik metric.
fn progress_line_of(stderr: &str) -> Option<usize> {
    stderr.lines().position(|l| {
        l.strip_prefix("survey ")
            .and_then(|rest| rest.split_whitespace().next())
            .is_some_and(|count| {
                count.strip_suffix("/8").is_some_and(|n| n.parse::<u32>().is_ok())
            })
    })
}

/// A plain progress line carries the best loglik so far when a worker has
/// set one before the line is rendered, so the count is not always the last
/// token. The closing `8/8` line is always printed, and always carries it
/// here: that is the stderr this test must still recognise.
#[test]
fn a_progress_line_is_recognised_with_or_without_its_metric() {
    let with_metric = "survey 1/8  best ll=-286.8\nsurvey 8/8  best ll=-286.8\nstored x\n";
    assert_eq!(progress_line_of(with_metric), Some(0));
    let bare_first = "survey 1/8\nsurvey 8/8  best ll=-286.8\n";
    assert_eq!(progress_line_of(bare_first), Some(0));
    assert_eq!(progress_line_of("survey: wrote results/x/8\n"), None);
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
    let progress = progress_line_of(&stderr)
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
