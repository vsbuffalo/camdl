//! gh#269: `camdl pfilter --save-prequential` must emit a PER-STREAM
//! (per-district) score breakdown alongside the joint score.
//!
//! `{STEM}.tsv` holds the joint score per step (`t`, `log_score`, `ess`) and
//! `{STEM}.streams.tsv` one row per step × scored stream with that stream's
//! own observed value, predictive band and scores. CRPS and PIT exist only per
//! stream: across streams there is no one observed quantity to score them
//! against.
//!
//! These tests pin:
//!  1. The joint file has one row per scored step and no stream column.
//!  2. The streams file has a row for each bound stream (`cases_p1`,
//!     `cases_p2`) at every joint step, with finite scores, and no `joint`
//!     pseudo-stream.

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

/// One per-stream TSV row keyed by (time, stream): the score columns for that cell.
#[derive(Debug, Clone)]
struct Row {
    y_obs: f64,
    log_score: f64,
    crps: f64,
    pit: f64,
}

/// A TSV's header and rows.
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("header").split('\t').map(String::from).collect();
    let rows = lines.map(|l| l.split('\t').map(String::from).collect()).collect();
    (header, rows)
}

/// Parse the per-stream prequential TSV. Returns `time -> (stream -> Row)`.
fn read_streams_tsv(path: &Path) -> BTreeMap<i64, BTreeMap<String, Row>> {
    let (header, rows) = read_tsv(path);
    let col = |name: &str| -> usize {
        header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no '{}' column in {:?}", name, header))
    };
    let mut out: BTreeMap<i64, BTreeMap<String, Row>> = BTreeMap::new();
    for f in rows {
        let t: f64 = f[col("t")].parse().expect("time parse");
        let row = Row {
            y_obs: f[col("y_obs")].parse().expect("y_obs parse"),
            log_score: f[col("log_score")].parse().expect("log_score parse"),
            crps: f[col("crps")].parse().expect("crps parse"),
            pit: f[col("pit")].parse().expect("pit parse"),
        };
        out.entry(t.round() as i64).or_default().insert(f[col("stream")].clone(), row);
    }
    out
}

#[test]
fn prequential_tsv_writes_joint_scores_and_per_stream_scores_to_two_files() {
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

    let (joint_header, joint) = read_tsv(&stem.with_extension("tsv"));
    assert_eq!(joint_header, ["t", "log_score", "ess"],
        "the joint file carries the step's joint score and ESS, no stream column");
    assert!(!joint.is_empty(), "prequential trace empty");
    let preq = read_streams_tsv(&tmp.path().join("preq.streams.tsv"));

    // The observations must be non-trivial, else the finiteness checks could
    // pass on all-zero data.
    let any_nonzero = preq.values().flat_map(|s| s.values()).any(|r| r.y_obs != 0.0);
    assert!(any_nonzero, "synthetic obs all zero — test would not detect the bug");

    for row in &joint {
        let t: f64 = row[0].parse().unwrap();
        let ls: f64 = row[1].parse().unwrap();
        assert!(ls.is_finite(), "joint log score at t={t} must be finite");
        let streams = preq.get(&(t.round() as i64))
            .unwrap_or_else(|| panic!("no per-stream rows at t={t}"));
        assert!(!streams.contains_key("joint"), "no `joint` pseudo-stream at t={t}");
        for name in ["cases_p1", "cases_p2"] {
            let r = streams.get(name)
                .unwrap_or_else(|| panic!("no {name} row at t={t}"));
            assert!(
                r.log_score.is_finite() && r.crps.is_finite() && r.pit.is_finite(),
                "per-stream scores for {name} at t={t} must be finite: {r:?}"
            );
        }
    }
}
