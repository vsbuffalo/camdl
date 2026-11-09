//! End-to-end tests for `camdl pfilter --save-paths` and
//! `--save-filtering`.
//!
//! Covers the two-commit PF-latent-trajectories feature from
//! `docs/dev/proposals/2026-04-19-pf-latent-trajectories.md`. Shells
//! out to the release binary so we exercise the full CLI → PF → TSV
//! writer path.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing() -> PathBuf {
    let b = binary();
    assert!(
        b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        b.display()
    );
    b
}

/// A pure-death model with a Poisson observation on N. Small fixture
/// that finishes in ~tens of ms with 500 particles × 5 obs.
fn golden_pure_death() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/pure_death.ir.json")
}

fn write_fixture(tmp: &Path) -> (PathBuf, PathBuf) {
    let data = tmp.join("cases.tsv");
    std::fs::write(&data,
        "time\tpopulation\n1\t950\n2\t900\n3\t850\n4\t800\n5\t750\n").unwrap();
    let params = tmp.join("params.toml");
    std::fs::write(&params, "mu=0.05\n").unwrap();
    (data, params)
}

#[test]
fn save_paths_writes_n_times_t_rows() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let (data, params) = write_fixture(tmp.path());
    let paths = tmp.path().join("paths.tsv");

    let status = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "500", "--seed", "1",
               "--n-paths", "10", "--save-paths", &paths.to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success(), "pfilter should succeed");

    let out = std::fs::read_to_string(&paths).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    // header + 10 paths × 5 obs = 51 lines
    assert_eq!(lines.len(), 51, "expected 51 lines, got {}: {}",
        lines.len(), out);
    assert!(lines[0].starts_with("path\ttime\t"),
        "header should start with 'path\\ttime\\t': {}", lines[0]);

    // Every path_id 1..=10 should appear exactly 5 times.
    let mut counts = std::collections::HashMap::new();
    for line in &lines[1..] {
        let path_id: &str = line.split('\t').next().unwrap();
        *counts.entry(path_id.to_string()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 10, "should have 10 distinct paths");
    for (k, v) in &counts {
        assert_eq!(*v, 5, "path {} should have 5 timesteps, got {}", k, v);
    }
}

#[test]
fn save_filtering_writes_n_times_t_particles() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let (data, params) = write_fixture(tmp.path());
    let filter = tmp.path().join("filter.tsv");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "200", "--seed", "1",
               "--save-filtering", &filter.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "pfilter should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr));

    // 200 particles × 5 obs = 1000 rows + header
    let contents = std::fs::read_to_string(&filter).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 1001,
        "expected 200 × 5 + header = 1001 lines, got {}", lines.len());
    assert!(lines[0].ends_with("\tlog_weight"),
        "filtering header should end with log_weight column: {}", lines[0]);
}

#[test]
fn save_filtering_emits_mandatory_info_log() {
    // The info log explaining "this is not a smoothing path" fires
    // unconditionally. Load-bearing because the alternative — users
    // treating filtering marginals as sample paths — is silent.
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let (data, params) = write_fixture(tmp.path());
    let filter = tmp.path().join("f.tsv");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "100", "--seed", "1",
               "--save-filtering", &filter.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("filtering marginals"),
        "info log should mention 'filtering marginals': {}", stderr);
    assert!(stderr.contains("not smoothing paths") ||
            stderr.contains("NOT yield trajectory"),
        "info log should clarify these aren't sample paths: {}", stderr);
}

#[test]
fn save_paths_does_not_emit_filtering_caveat() {
    // Sanity-check the inverse: --save-paths without --save-filtering
    // should NOT print the filtering-marginals info log (it's a
    // --save-filtering-specific warning).
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let (data, params) = write_fixture(tmp.path());
    let paths = tmp.path().join("p.tsv");

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "100", "--seed", "1",
               "--n-paths", "5", "--save-paths", &paths.to_string_lossy()])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("filtering marginals"),
        "the filtering caveat should not fire for --save-paths alone: {}",
        stderr);
}

#[test]
fn save_paths_values_are_monotone_nonincreasing_for_pure_death() {
    // Scientific sanity: in a pure-death model, N can only decrease.
    // Every sampled path should have N[t+1] ≤ N[t].
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let (data, params) = write_fixture(tmp.path());
    let paths = tmp.path().join("paths.tsv");

    Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "500", "--seed", "1",
               "--n-paths", "20", "--save-paths", &paths.to_string_lossy()])
        .status().expect("spawn");

    let contents = std::fs::read_to_string(&paths).unwrap();
    let mut by_path: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for line in contents.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        let path_id = cols[0].to_string();
        let n: f64 = cols[2].parse().unwrap();
        by_path.entry(path_id).or_default().push(n);
    }
    assert!(!by_path.is_empty(), "should have at least one path");
    for (path_id, values) in &by_path {
        for pair in values.windows(2) {
            assert!(pair[0] >= pair[1],
                "path {}: pure-death N should be monotone nonincreasing, \
                 got {} → {}", path_id, pair[0], pair[1]);
        }
    }
}
