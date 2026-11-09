//! gh#269: `camdl pfilter --save-prequential` must emit a PER-STREAM
//! (per-district) score breakdown alongside the joint summary.
//!
//! Before gh#269 the `{STEM}.tsv` carried only a joint, summed-across-streams
//! score per step (no `stream` column). The filter already computes the
//! per-stream predictive but discarded the breakdown with `.sum()`. gh#269
//! stops summing: the joint stays as the summary (`stream="joint"`) and each
//! bound stream gets its own row.
//!
//! These tests pin:
//!  1. The TSV is tidy/long with a `stream` column: a `joint` row plus one row
//!     per bound stream (`cases_p1`, `cases_p2`) at every step.
//!  2. The per-stream `y_obs` sum to the `joint` `y_obs` at each step (the
//!     joint == sum-of-streams invariant for the observed values, the directly
//!     checkable consequence of the structural `score_streams` seam).

use std::collections::BTreeMap;
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
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn multi_block_model() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/seir_spatial_5_inference.ir.json")
}

fn synth_obs(bin: &Path, model: &Path, tmp: &Path, extra_args: &[&str]) -> PathBuf {
    let obs_path = tmp.join("obs.tsv");
    let mut cmd = Command::new(bin);
    cmd.env_remove("CLICOLOR_FORCE");
    cmd.env("CAMDL_SKIP_VERSION_CHECK", "1").args([
        "simulate",
        &model.to_string_lossy(),
        "--backend",
        "chain_binomial",
        "--dt",
        "1",
        "--seed",
        "42",
        "--obs-only",
        &obs_path.to_string_lossy(),
    ]);
    cmd.args(extra_args);
    let status = cmd.status().expect("spawn simulate");
    assert!(status.success(), "synthetic obs generation failed");
    obs_path
}

/// One TSV row keyed by (time, stream): the score columns for that cell.
#[derive(Debug, Clone)]
struct Row {
    y_obs: f64,
    log_score: f64,
    crps: f64,
    pit: f64,
}

/// Parse the tidy prequential TSV. Returns `time -> (stream -> Row)`.
fn read_preq_tsv(path: &Path) -> BTreeMap<i64, BTreeMap<String, Row>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |name: &str| -> usize {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no '{}' column in {:?}", name, header))
    };
    let (t_i, s_i, y_i, ls_i, c_i, p_i) = (
        col("t"),
        col("stream"),
        col("y_obs"),
        col("log_score"),
        col("crps"),
        col("pit"),
    );
    let mut out: BTreeMap<i64, BTreeMap<String, Row>> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let t: f64 = f[t_i].parse().expect("time parse");
        let stream = f[s_i].to_string();
        let row = Row {
            y_obs: f[y_i].parse().expect("y_obs parse"),
            log_score: f[ls_i].parse().expect("log_score parse"),
            crps: f[c_i].parse().expect("crps parse"),
            pit: f[p_i].parse().expect("pit parse"),
        };
        out.entry(t.round() as i64).or_default().insert(stream, row);
    }
    out
}

#[test]
fn prequential_per_stream_tsv_has_stream_column_and_joint_equals_sum() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(), &["--scenario", "true_params"]);
    let stem = tmp.path().join("preq");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter",
            &multi_block_model().to_string_lossy(),
            "--scenario",
            "true_params",
            "--data",
            &format!("cases_p1={}", obs.to_string_lossy()),
            "--data",
            &format!("cases_p2={}", obs.to_string_lossy()),
            "--particles",
            "200",
            "--dt",
            "1",
            "--seed",
            "1",
            "--save-prequential",
            &stem.to_string_lossy(),
        ])
        .output()
        .expect("spawn pfilter");
    assert!(
        out.status.success(),
        "multi-stream pfilter --save-prequential failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let preq = read_preq_tsv(&stem.with_extension("tsv"));
    assert!(!preq.is_empty(), "prequential trace empty");

    // The observations must be non-trivial, else the sum check could pass
    // vacuously on all-zero data.
    let any_nonzero = preq
        .values()
        .any(|streams| streams.get("joint").map(|r| r.y_obs != 0.0).unwrap_or(false));
    assert!(any_nonzero, "synthetic obs all zero — test would not detect the bug");

    for (t, streams) in &preq {
        // 1. The `joint` row exists at every step.
        let joint = streams
            .get("joint")
            .unwrap_or_else(|| panic!("no joint row at t={}", t));
        // 2. Both bound streams have their own rows.
        let p1 = streams
            .get("cases_p1")
            .unwrap_or_else(|| panic!("no cases_p1 row at t={} (no stream column?)", t));
        let p2 = streams
            .get("cases_p2")
            .unwrap_or_else(|| panic!("no cases_p2 row at t={} (no stream column?)", t));

        // 3. joint y_obs == sum of per-stream y_obs (the structural invariant).
        let sum = p1.y_obs + p2.y_obs;
        assert_eq!(
            joint.y_obs, sum,
            "at t={}: joint y_obs = {} but cases_p1+cases_p2 = {}",
            t, joint.y_obs, sum
        );

        // Sanity: per-stream scores are finite (they were computed, not absent).
        for (name, r) in [("cases_p1", p1), ("cases_p2", p2)] {
            assert!(
                r.log_score.is_finite() && r.crps.is_finite() && r.pit.is_finite(),
                "per-stream scores for {} at t={} must be finite: {:?}",
                name, t, r
            );
        }
    }
}
