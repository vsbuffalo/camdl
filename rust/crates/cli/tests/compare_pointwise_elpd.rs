//! End-to-end acceptance for `camdl compare --pointwise` (gh#706): the
//! per-observation `Δelpd` vector the comparison already computes in order to
//! form `se(Δelpd)`, written out instead of discarded.
//!
//! `Δelpd = 12 nats` says a model won. The pointwise vector says it won
//! entirely on three weeks around an intervention, or entirely on one district,
//! or on a single reporting batch — the difference between a model comparison
//! and a model diagnosis. Gelman, Vehtari & McElreath (2026) Fig. 8.10 is this
//! vector plotted two ways; its payoff in their worked example is that the whole
//! gap comes from about ten particular observations.
//!
//! The fixtures here are hand-written `prequential.json` traces rather than
//! real fits: `compare` reads an explicit trace as-is, and the quantity under
//! test is a projection of that trace, not the filter that produced it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// A trace with three scored steps, two streams each. `log_scores` is the joint
/// score per step; `north`/`south` are the per-stream breakdown.
fn trace_json(joint: [f64; 3], north: [f64; 3], south: [f64; 3]) -> String {
    let steps: Vec<String> = (0..3)
        .map(|i| {
            format!(
                r#"{{"t": {t}, "y_obs": 10.0, "y_pred_samples": [],
                     "log_score": {j}, "crps": 1.0, "pit": 0.5, "ess": 900.0,
                     "per_stream": [
                       {{"stream": "north", "y_obs": 6.0, "y_pred_samples": [],
                         "log_score": {n}, "crps": 0.5, "pit": 0.4}},
                       {{"stream": "south", "y_obs": 4.0, "y_pred_samples": [],
                         "log_score": {s}, "crps": 0.5, "pit": 0.6}}
                     ]}}"#,
                t = 7.0 * (i + 1) as f64,
                j = joint[i],
                n = north[i],
                s = south[i],
            )
        })
        .collect();
    format!(
        r#"{{"schema_version": 1, "t0": 1, "provenance": "plug_in",
             "conditioning": "in_sample", "warnings": [],
             "steps": [{}]}}"#,
        steps.join(",")
    )
}

fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("a header row")
        .split('\t').map(String::from).collect();
    let rows = lines.map(|l| l.split('\t').map(String::from).collect()).collect();
    (header, rows)
}

/// The whole point: the candidate beats the baseline by 3 nats in total, and
/// ALL of it comes from the middle step — which the scalar cannot say and the
/// vector says immediately. The per-stream rows localize it further, to
/// `north`.
#[test]
fn pointwise_tsv_localizes_the_elpd_difference_to_one_step_and_one_stream() {
    let bin = binary();
    assert!(bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display());

    let dir = std::env::temp_dir().join("camdl_compare_pointwise_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // base: joint -6 per step. cand: identical except step 2, which is 3 nats
    // better, and the gain sits entirely in `north`.
    let base = dir.join("base.json");
    let cand = dir.join("cand.json");
    std::fs::write(&base,
        trace_json([-6.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();
    std::fs::write(&cand,
        trace_json([-6.0, -3.0, -6.0], [-3.0, 0.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();

    let out = dir.join("pointwise.tsv");
    let run = Command::new(&bin)
        .arg("compare")
        .arg(&base)
        .arg(&cand)
        .args(["--baseline", "base.json"])
        .arg("--pointwise")
        .arg(&out)
        .output()
        .expect("running camdl compare");
    assert!(run.status.success(),
        "compare --pointwise must succeed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));

    let (header, rows) = read_tsv(&out);
    for col in ["model", "baseline", "t", "scope", "stream", "log_score",
                "baseline_log_score", "delta_log_score"] {
        assert!(header.iter().any(|h| h == col),
            "the pointwise TSV must carry a `{col}` column; header = {header:?}");
    }
    let col = |name: &str| header.iter().position(|h| h == name).unwrap();

    // The baseline is not differenced against itself — only the candidate has
    // rows, and it is named alongside the baseline it was scored against.
    assert!(rows.iter().all(|r| r[col("model")] == "cand.json"),
        "only the compared candidates get rows: {rows:?}");
    assert!(rows.iter().all(|r| r[col("baseline")] == "base.json"),
        "every row names the baseline it was differenced against: {rows:?}");

    let joint: Vec<&Vec<String>> = rows.iter().filter(|r| r[col("scope")] == "joint").collect();
    assert_eq!(joint.len(), 3, "one joint row per scored step: {rows:?}");
    let deltas: Vec<f64> = joint.iter()
        .map(|r| r[col("delta_log_score")].parse().unwrap())
        .collect();
    assert!((deltas[0]).abs() < 1e-9 && (deltas[2]).abs() < 1e-9,
        "steps 1 and 3 are ties: {deltas:?}");
    assert!((deltas[1] - 3.0).abs() < 1e-9,
        "the entire 3-nat gap sits on step 2: {deltas:?}");
    // The rows join back to the observation axis by time, not by row index.
    assert_eq!(joint[1][col("t")], "14", "the differing step is t = 14: {joint:?}");

    // Per-stream: the gain is `north`'s alone.
    let stream_row = |name: &str, t: &str| -> &Vec<String> {
        rows.iter()
            .find(|r| r[col("scope")] == "stream" && r[col("stream")] == name
                && r[col("t")] == t)
            .unwrap_or_else(|| panic!("no {name} row at t={t}: {rows:?}"))
    };
    let north: f64 = stream_row("north", "14")[col("delta_log_score")].parse().unwrap();
    let south: f64 = stream_row("south", "14")[col("delta_log_score")].parse().unwrap();
    assert!((north - 3.0).abs() < 1e-9, "north carries the whole gain: {north}");
    assert!(south.abs() < 1e-9, "south is a tie: {south}");
}

/// Review blocker 3. `paired_delta` pairs steps BY INDEX and the preflight
/// guards only `n_scored()`, so two traces scoring the same NUMBER of
/// observations at different TIMES were differenced anyway — Δelpd, se(Δ), the
/// likelihood ratio and the deciban verdict all computed across two different
/// observation axes, and rendered as a confident answer.
///
/// The `--pointwise` path already refuses this. Guarding the opt-in path and
/// leaving the DEFAULT path to render is the wrong half: the scalar table is
/// what people read.
///
/// The refusal must name which times differ. This turns a comparison that
/// previously rendered into an error, and "these traces are not comparable"
/// with no further detail is not actionable on a real outbreak.
#[test]
fn compare_refuses_traces_on_different_observation_axes() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());

    let dir = std::env::temp_dir().join("camdl_compare_axis_mismatch_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Same T_score (3 scored steps), different observation times: the third
    // step is t=21 on one side and t=28 on the other.
    let base = dir.join("base.json");
    let cand = dir.join("cand.json");
    std::fs::write(&base,
        trace_json([-6.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();
    std::fs::write(&cand,
        trace_json([-5.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])
            .replace("\"t\": 21", "\"t\": 28")).unwrap();

    let run = Command::new(&bin)
        .arg("compare").arg(&base).arg(&cand)
        .args(["--baseline", "base.json"])
        .output().expect("running camdl compare");

    assert!(!run.status.success(),
        "a comparison across two different observation axes must be refused, \
         not rendered.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));

    let err = String::from_utf8_lossy(&run.stderr);
    assert!(err.contains("21") && err.contains("28"),
        "the refusal must name the times that differ so it is actionable: {err}");
    assert!(err.contains("base.json") && err.contains("cand.json"),
        "and which two traces disagree: {err}");

    // The scalar verdict must not appear on stdout — no ratio, no decibans.
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(!out.contains("dB"),
        "no evidence verdict may be rendered for an uncomparable pair: {out}");
}

/// Negative control: aligned traces of equal length still compare. The new
/// refusal must not fire on the ordinary case.
#[test]
fn compare_still_renders_when_the_observation_axes_agree() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());

    let dir = std::env::temp_dir().join("camdl_compare_axis_ok_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base.json");
    let cand = dir.join("cand.json");
    std::fs::write(&base,
        trace_json([-6.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();
    std::fs::write(&cand,
        trace_json([-5.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();

    let run = Command::new(&bin)
        .arg("compare").arg(&base).arg(&cand)
        .args(["--baseline", "base.json"])
        .output().expect("running camdl compare");
    assert!(run.status.success(),
        "aligned traces must still compare.\nstderr:\n{}",
        String::from_utf8_lossy(&run.stderr));
}

/// gh#570: two models scoring different stream sets produce an elpd difference
/// that is not a like-for-like comparison, and the scalar hides it. In the
/// pointwise view the stream that only one side scored has an empty cell on the
/// other, and an empty difference — visible rather than silently summed.
#[test]
fn a_stream_only_one_model_scored_shows_as_an_empty_difference() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());

    let dir = std::env::temp_dir().join("camdl_compare_pointwise_streams_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // The candidate scores a third stream the baseline never had.
    let base = dir.join("base.json");
    std::fs::write(&base,
        trace_json([-6.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])).unwrap();
    let cand_text = trace_json([-6.0, -6.0, -6.0], [-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0])
        .replace(
            r#"{"stream": "south""#,
            r#"{"stream": "east", "y_obs": 1.0, "y_pred_samples": [],
                "log_score": -1.0, "crps": 0.2, "pit": 0.5},
               {"stream": "south""#,
        );
    let cand = dir.join("cand.json");
    std::fs::write(&cand, cand_text).unwrap();

    let out = dir.join("pointwise.tsv");
    let run = Command::new(&bin)
        .arg("compare").arg(&base).arg(&cand)
        .args(["--baseline", "base.json"])
        .arg("--pointwise").arg(&out)
        .output().expect("running camdl compare");
    assert!(run.status.success(), "stderr:\n{}", String::from_utf8_lossy(&run.stderr));

    let (header, rows) = read_tsv(&out);
    let col = |name: &str| header.iter().position(|h| h == name).unwrap();
    let east: Vec<&Vec<String>> = rows.iter()
        .filter(|r| r[col("stream")] == "east").collect();
    assert_eq!(east.len(), 3, "the unmatched stream still gets its rows: {rows:?}");
    for r in &east {
        assert_eq!(r[col("baseline_log_score")], "",
            "the baseline never scored `east`, so its cell is empty: {r:?}");
        assert_eq!(r[col("delta_log_score")], "",
            "a difference against a stream the baseline never scored is not a \
             number: {r:?}");
        assert_ne!(r[col("log_score")], "",
            "the candidate's own score is still reported: {r:?}");
    }
}
