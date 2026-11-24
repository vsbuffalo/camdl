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

    let status = Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "500", "--seed", "1",
               "--n-paths", "10", "--save-paths", &paths.to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success(), "pfilter should succeed");

    let out = std::fs::read_to_string(&paths).unwrap();
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    // header + 10 paths × 5 obs = 51 lines
    assert_eq!(lines.len(), 51, "expected 51 lines, got {}: {}",
        lines.len(), out);
    assert!(lines[0].starts_with("path\ttime\tt_start\tt_stop\t"),
        "header should start with 'path\\ttime\\tt_start\\tt_stop\\t': {}", lines[0]);

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

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
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

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
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

    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
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

    Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
        .args(["pfilter", &golden_pure_death().to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--data",   &data.to_string_lossy(),
               "--particles", "500", "--seed", "1",
               "--n-paths", "20", "--save-paths", &paths.to_string_lossy()])
        .status().expect("spawn");

    let table = PathsTable::read(&paths);
    let mut by_path: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for row in &table.rows {
        by_path.entry(row[0].clone()).or_default().push(table.num(row, "N"));
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

/// A run store of its own for one command, so tests that run the same filter
/// at once do not meet in a shared root (a filter run another live process
/// holds is refused).
fn isolated_store() -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "camdl-store-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

// ── Per-row flows and rate noise on saved paths ─────────────────────────────

/// A saved-paths file: its comment lines, its header and its rows.
struct PathsTable {
    comments: Vec<String>,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl PathsTable {
    fn read(path: &Path) -> Self {
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines();
        let mut comments = Vec::new();
        let header = loop {
            let line = lines.next().expect("a header line");
            match line.strip_prefix("# ") {
                Some(c) => comments.push(c.to_string()),
                None => break line.split('\t').map(str::to_string).collect::<Vec<_>>(),
            }
        };
        let rows = lines.map(|l| l.split('\t').map(str::to_string).collect()).collect();
        PathsTable { comments, header, rows }
    }

    fn col(&self, name: &str) -> usize {
        self.header.iter().position(|h| h == name)
            .unwrap_or_else(|| panic!("no column `{name}` in {:?}", self.header))
    }

    fn num(&self, row: &[String], name: &str) -> f64 {
        row[self.col(name)].parse().unwrap()
    }

    /// The rows of each path, in time order.
    fn by_path(&self) -> Vec<Vec<&Vec<String>>> {
        let mut paths: std::collections::BTreeMap<&str, Vec<&Vec<String>>> = Default::default();
        for r in &self.rows {
            paths.entry(r[0].as_str()).or_default().push(r);
        }
        paths.into_values().collect()
    }
}

/// An SEIR model whose infection carries gamma rate noise, observed through
/// weekly incidence and a prevalence count every 10 days, so the filter's
/// stops (the union of both streams' times) cut the weeks into pieces. Only
/// the transitions change S, E and I: there is no event, intervention or
/// balance.
const NOISY_SEIR: &str = r#"
time_unit = 'days

compartments { S, E, I, R }

parameters {
  beta     : rate in [0.001, 5.0]
  sigma    : rate in [0.01, 1.0]
  gamma    : rate in [0.01, 1.0]
  sigma_se : positive dim T in [0.0, 5.0]
  rho      : probability in [0.0, 1.0]
}

init { S = 990  E = 0  I = 10  R = 0 }

transitions {
  infection : S --> E  @ overdispersed(beta * S * I / (S + E + I + R), sigma_se)
  latency   : E --> I  @ sigma * E
  recovery  : I --> R  @ gamma * I
}

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(recovery)
    cases         ~ poisson(rate = rho * projected)
    emit_schedule = every 7 'days
  }
  prevalence {
    columns       { time : time, prevalence : count }
    projected     = I
    prevalence    ~ poisson(rate = rho * projected)
    emit_schedule = every 10 'days
  }
}

simulate { from = 0 'days  to = 70 'days }
"#;

/// Runs `pfilter --save-paths` on [`NOISY_SEIR`] with `extra` arguments
/// appended, and returns the command's output and the paths file's path.
fn noisy_seir_paths(tmp: &Path, extra: &[&str]) -> (std::process::Output, PathBuf) {
    let bin = skip_if_missing();
    let model = tmp.join("m.camdl");
    std::fs::write(&model, NOISY_SEIR).unwrap();
    let params = tmp.join("p.toml");
    std::fs::write(&params, "beta = 0.6\nsigma = 0.3\ngamma = 0.2\nsigma_se = 0.5\nrho = 0.5\n").unwrap();
    let cases = tmp.join("cases.tsv");
    std::fs::write(&cases, "time\tcases\n7\t5\n14\t20\n21\t40\n28\t50\n35\t40\n42\t30\n49\t20\n56\t10\n63\t5\n70\t2\n").unwrap();
    let prev = tmp.join("prev.tsv");
    std::fs::write(&prev, "time\tprevalence\n5\t5\n15\t15\n25\t40\n35\t30\n45\t20\n55\t10\n65\t4\n").unwrap();
    let paths = tmp.join("paths.tsv");
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["pfilter", &model.to_string_lossy(), "--params", &params.to_string_lossy(),
               "--data", &format!("cases={}", cases.display()),
               "--data", &format!("prevalence={}", prev.display()),
               "--particles", "200", "--seed", "3", "--n-paths", "8",
               "--save-paths", &paths.to_string_lossy()])
        .args(extra)
        .output().expect("spawn");
    (out, paths)
}

#[test]
fn save_paths_flows_sum_to_the_state_change_over_any_range_of_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &[]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let t = PathsTable::read(&paths);

    // The rows are the union of both streams' times; each row's period opens
    // at the previous row, and the first at the run's start.
    let times: Vec<f64> = vec![5., 7., 14., 15., 21., 25., 28., 35., 42., 45., 49., 55., 56., 63., 65., 70.];
    let at_start = [("S", 990.0), ("E", 0.0), ("I", 10.0)];
    let net = |row: &[String], comp: &str| -> f64 {
        let f = |n: &str| t.num(row, n);
        match comp {
            "S" => -f("flow_infection"),
            "E" => f("flow_infection") - f("flow_latency"),
            "I" => f("flow_latency") - f("flow_recovery"),
            _ => unreachable!(),
        }
    };
    let paths = t.by_path();
    assert_eq!(paths.len(), 8);
    for rows in &paths {
        let row_times: Vec<f64> = rows.iter().map(|r| t.num(r, "time")).collect();
        assert_eq!(row_times, times);
        for (k, r) in rows.iter().enumerate() {
            let opens = if k == 0 { 0.0 } else { times[k - 1] };
            assert_eq!(t.num(r, "t_start"), opens, "row {k}'s period opens at the previous row");
            assert_eq!(t.num(r, "t_stop"), times[k]);
        }
        // Over every range of rows (a, b], including from the run's start,
        // the summed flows equal the change in each compartment.
        for (comp, initial) in at_start {
            let state = |k: usize| t.num(rows[k], comp);
            for b in 0..rows.len() {
                let mut sum = 0.0;
                for a in (0..=b).rev() {
                    sum += net(rows[a], comp);
                    let before = if a == 0 { initial } else { state(a - 1) };
                    assert_eq!(state(b) - before, sum,
                        "{comp}: the flows over rows {a}..={b} must sum to its change");
                }
            }
        }
    }
}

#[test]
fn save_paths_noise_is_one_when_the_noise_variance_is_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &["--param", "sigma_se=0"]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let t = PathsTable::read(&paths);
    let mut drawn = 0;
    for r in &t.rows {
        if t.num(r, "noise_dt_infection") > 0.0 {
            assert_eq!(t.num(r, "noise_infection"), 1.0, "σ² = 0 draws a multiplier of exactly 1: {r:?}");
            drawn += 1;
        }
    }
    assert!(drawn > 0, "some row must have drawn the multiplier");

    // With σ² > 0 the same rows carry the realised noise, which varies.
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &[]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let t = PathsTable::read(&paths);
    let noise: Vec<f64> = t.rows.iter().map(|r| t.num(r, "noise_infection")).filter(|v| v.is_finite()).collect();
    assert!(noise.iter().any(|&v| v != 1.0) && noise.iter().all(|&v| v >= 0.0),
        "σ² = 0.5 realises a non-negative noise that is not identically 1: {noise:?}");
}

#[test]
fn save_paths_noise_exposure_is_the_time_the_source_could_fire() {
    // At beta = 5 the epidemic empties S part-way through the run. A substep
    // that starts with S = 0 draws no multiplier, so it adds no exposure, and
    // a row with no exposure has no noise value.
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &["--param", "beta=5"]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let t = PathsTable::read(&paths);
    let mut partial = 0;
    for rows in t.by_path() {
        for (k, r) in rows.iter().enumerate() {
            let width = t.num(r, "t_stop") - t.num(r, "t_start");
            let exposure = t.num(r, "noise_dt_infection");
            let noise = t.num(r, "noise_infection");
            let before = |c: &str, initial: f64| if k == 0 { initial } else { t.num(rows[k - 1], c) };
            assert!((0.0..=width).contains(&exposure), "exposure within the row: {r:?}");
            if before("S", 990.0) == 0.0 {
                assert_eq!(exposure, 0.0, "an empty source draws nothing: {r:?}");
                assert!(noise.is_nan(), "no exposure, no noise value: {r:?}");
            } else if before("I", 10.0) > 0.0 {
                assert!(exposure > 0.0 && noise.is_finite() && noise >= 0.0,
                    "a source with members and a positive rate at the row's start \
                     draws on its first substep: {r:?}");
            }
            if exposure > 0.0 && exposure < width {
                partial += 1;
            }
        }
    }
    assert!(partial > 0, "some row must see S empty part-way through it");
}

#[test]
fn save_paths_header_names_every_stream_of_the_row_grid() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &[]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let t = PathsTable::read(&paths);
    assert_eq!(t.comments, vec![
        "camdl-paths v2".to_string(),
        "rows: the observation times of streams cases, prevalence; flow_* and noise_* cover [t_start, t_stop)".to_string(),
    ]);
    assert_eq!(t.header, [
        "path", "time", "t_start", "t_stop", "S", "E", "I", "R",
        "flow_infection", "flow_latency", "flow_recovery",
        "noise_infection", "noise_dt_infection", "cases", "prevalence",
    ]);
}

#[test]
fn save_paths_column_selection_drops_flows_and_keeps_noise() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &["--no-flows"]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(PathsTable::read(&paths).header, [
        "path", "time", "t_start", "t_stop", "S", "E", "I", "R",
        "noise_infection", "noise_dt_infection", "cases", "prevalence",
    ]);

    let tmp = tempfile::tempdir().unwrap();
    let (out, paths) = noisy_seir_paths(tmp.path(), &["--columns", "noise_infection,S,flow_recovery"]);
    assert!(out.status.success(), "pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(PathsTable::read(&paths).header, [
        "path", "time", "t_start", "t_stop", "S", "flow_recovery",
        "noise_infection", "noise_dt_infection", "cases", "prevalence",
    ]);
}

#[test]
fn save_paths_refuses_a_column_the_file_cannot_hold() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, _) = noisy_seir_paths(tmp.path(), &["--columns", "noise_latency"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a transition without rate noise has no noise column");
    assert!(stderr.contains("--columns: unknown column `noise_latency`") && stderr.contains("noise_infection"),
        "the refusal names the column and the valid ones: {stderr}");
}
