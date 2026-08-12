//! `camdl simulate --quantities-out <dir>` — generated quantities (proposal
//! 2026-06-25) emitted from the simulate path, reusing the predict machinery.
//!
//! A single fixed-params run writes POINT values (a bare `value` column); a
//! multi-cell `--draws` run writes BANDED quantiles. Without `--quantities-out`,
//! a model that declares quantities still simulates fine — the quantities are
//! skipped with a one-line note (not a hard error). Quantities are a regenerated
//! sidecar, never part of the content-addressed run identity.

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
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        b.display()
    );
    b
}

/// A closed SIR with a `quantities {}` block: a series, a value scalar, and a
/// time scalar (right-censorable). The baseline scenario supplies fixed params
/// (R0 = 5, so prevalence climbs past the onset threshold).
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate     in [0.001, 2.0]
  gamma : rate     in [0.001, 1.0]
  N0    : count    in [100, 100000]
  I0    : count    in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

quantities {
  prevalence = I / N                   # series
  peak       = max(I / N)              # value scalar (no censoring)
  onset      = first_above(I / N, 0.05)   # time scalar (right-censorable)
}

simulate {
  from = 0 'days
  to   = 80 'days
}

scenarios {
  baseline {
    set = {
      beta  = 0.5
      gamma = 0.1
      N0    = 1000
      I0    = 10
    }
  }
  ctrl {
    set = {
      beta  = 0.2
      gamma = 0.1
      N0    = 1000
      I0    = 10
    }
  }
}
"#;

fn write_model(dir: &Path) -> PathBuf {
    let p = dir.join("sir_q.camdl");
    std::fs::write(&p, MODEL).unwrap();
    p
}

fn run(bin: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent; the worktree camdlc is found via the CAMDLC env or
        // PATH). Mirrors the runbook's ad-hoc guidance.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

#[test]
fn simulate_point_run_writes_quantities_sidecar() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    let qdir = tmp.path().join("q");

    let out = run(
        &bin,
        &[
            "simulate",
            model.to_str().unwrap(),
            "--scenario",
            "baseline",
            "--seed",
            "1",
            "--output-dir",
            tmp.path().join("results").to_str().unwrap(),
            "--quantities-out",
            qdir.to_str().unwrap(),
        ],
    );
    assert!(
        out.status.success(),
        "simulate --quantities-out should succeed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // ── prevalence.tsv: a series → `time\tvalue` (point: no banding) ──
    let prev = qdir.join("quantities").join("prevalence.tsv");
    let prev_txt = std::fs::read_to_string(&prev)
        .unwrap_or_else(|_| panic!("missing {}", prev.display()));
    let mut plines = prev_txt.lines();
    // This run passes `--scenario baseline`, so it HAS a scenario axis and the
    // design coordinate leads every row (gh#562).
    assert_eq!(
        plines.next().unwrap(),
        "scenario\ttime\tvalue",
        "point series header carries the scenario coordinate, then `time value`"
    );
    let prow: Vec<&str> = plines.next().expect("at least one prevalence row").split('\t').collect();
    assert_eq!(prow.len(), 3, "point series row is `scenario time value`");
    assert_eq!(prow[0], "baseline", "the scenario that was actually applied");
    assert!(prow[1].parse::<f64>().is_ok(), "time column numeric, got {:?}", prow[1]);
    assert!(prow[2].parse::<f64>().is_ok(), "value column numeric, got {:?}", prow[2]);

    // ── peak.tsv: a value scalar → bare `value` ──
    let peak = qdir.join("quantities").join("peak.tsv");
    let peak_txt = std::fs::read_to_string(&peak)
        .unwrap_or_else(|_| panic!("missing {}", peak.display()));
    let mut klines = peak_txt.lines();
    assert_eq!(
        klines.next().unwrap(), "scenario\tvalue",
        "point scalar header is `scenario value`"
    );
    let krow: Vec<&str> = klines.next().expect("a peak row").split('\t').collect();
    assert_eq!(krow[0], "baseline");
    assert!(krow[1].parse::<f64>().is_ok(), "peak value numeric, got {:?}", krow[1]);

    // ── onset.tsv: a time scalar → bare `value`; a finite time OR `NA` ──
    let onset = qdir.join("quantities").join("onset.tsv");
    let onset_txt = std::fs::read_to_string(&onset)
        .unwrap_or_else(|_| panic!("missing {}", onset.display()));
    let mut olines = onset_txt.lines();
    assert_eq!(
        olines.next().unwrap(), "scenario\tvalue",
        "point time-scalar header is `scenario value`"
    );
    let orow: Vec<&str> = olines.next().expect("an onset row").split('\t').collect();
    assert_eq!(orow[0], "baseline");
    assert!(
        orow[1] == "NA" || orow[1].parse::<f64>().is_ok(),
        "onset is a finite time or NA (right-censored), got {:?}",
        orow[1]
    );

    // ── quantities.json: the manifest, one entry per logical quantity ──
    let manifest = qdir.join("quantities.json");
    let mtxt = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|_| panic!("missing {}", manifest.display()));
    let mjson: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    assert_eq!(mjson["schema"], "camdl.quantities/v1", "manifest schema tag");
    // Calendar semantics travel with the artifact. This fixture is unanchored
    // (no `origin`), so origin is null — the consumer's numeric-fallback path.
    assert_eq!(mjson["calendar"]["time_unit"], "days", "calendar time_unit travels");
    assert!(mjson["calendar"]["origin"].is_null(), "unanchored model → null origin");
    let qs = mjson["quantities"].as_array().expect("quantities array");
    let lookup = |n: &str| qs.iter().find(|q| q["name"] == n).unwrap_or_else(|| panic!("manifest missing {n}"));
    assert_eq!(lookup("prevalence")["shape"], "series");
    assert_eq!(lookup("peak")["shape"], "scalar");
    assert_eq!(lookup("peak")["reduce"], "max");
    assert!(lookup("peak")["censoring"].is_null(), "a value reduction is not censorable");
    assert_eq!(lookup("onset")["reduce"], "first_above");
    assert!(lookup("onset")["censoring"].is_object(), "a time reduction records right-censoring");
}

#[test]
fn simulate_draws_run_writes_banded_quantities() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    let qdir = tmp.path().join("q");

    // A `--draws` source bands even when each cell is one realization. Uniform
    // draws over the declared bounds resolve every parameter, so no scenario is
    // needed.
    let out = run(
        &bin,
        &[
            "simulate",
            model.to_str().unwrap(),
            "--draws",
            "uniform",
            "-n",
            "5",
            "--seed",
            "7",
            "--output-dir",
            tmp.path().join("results").to_str().unwrap(),
            "--quantities-out",
            qdir.to_str().unwrap(),
        ],
    );
    assert!(
        out.status.success(),
        "simulate --draws --quantities-out should succeed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Series quantity → banded header (n_draws + the quantile band), NOT a point
    // `value`.
    let prev = qdir.join("quantities").join("prevalence.tsv");
    let prev_txt = std::fs::read_to_string(&prev)
        .unwrap_or_else(|_| panic!("missing {}", prev.display()));
    assert_eq!(
        prev_txt.lines().next().unwrap(),
        "time\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "a --draws run bands the series (time + n_draws + quantiles)"
    );

    // A value scalar → banded header with no time, no censoring trio.
    let peak = qdir.join("quantities").join("peak.tsv");
    let peak_txt = std::fs::read_to_string(&peak)
        .unwrap_or_else(|_| panic!("missing {}", peak.display()));
    assert_eq!(
        peak_txt.lines().next().unwrap(),
        "n_draws\tq05\tq25\tq50\tq75\tq95",
        "value-scalar banded header"
    );

    // A time scalar → banded header carrying the censoring trio.
    let onset = qdir.join("quantities").join("onset.tsv");
    let onset_txt = std::fs::read_to_string(&onset)
        .unwrap_or_else(|_| panic!("missing {}", onset.display()));
    assert_eq!(
        onset_txt.lines().next().unwrap(),
        "n_draws\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95",
        "time-scalar banded header carries the censoring trio"
    );
}

/// gh#562: each scenario gets its OWN band.
///
/// Scenario is a design coordinate — it says which world was simulated — while
/// draws are sampling coordinates within one world. A quantile summarises the
/// second kind only. Pooling them averaged a baseline and its counterfactual
/// into one ribbon describing neither, in a file whose shape was
/// indistinguishable from a correct posterior band.
///
/// The load-bearing assertion is `n_draws == 5` per scenario, NOT 10. A test
/// that only checked for the `scenario` column would pass against a version
/// that emits the column and still pools.
#[test]
fn simulate_bands_each_scenario_separately() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    let qdir = tmp.path().join("q");
    let results_dir = tmp.path().join("results");

    let out = run(
        &bin,
        &[
            "simulate",
            model.to_str().unwrap(),
            "--draws", "uniform",
            "-n", "5",
            "--seed", "7",
            "--scenario", "baseline",
            "--scenario", "ctrl",
            "--output-dir", results_dir.to_str().unwrap(),
            "--quantities-out", qdir.to_str().unwrap(),
        ],
    );
    assert!(
        out.status.success(),
        "multi-scenario --quantities-out must succeed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let prev = qdir.join("quantities").join("prevalence.tsv");
    let txt = std::fs::read_to_string(&prev)
        .unwrap_or_else(|_| panic!("missing {}", prev.display()));
    let mut lines = txt.lines();

    let header = lines.next().expect("header");
    assert_eq!(
        header, "scenario\ttime\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "the scenario column leads the banded series header"
    );

    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert!(!rows.is_empty(), "at least one band row");

    // Both arms present, and nothing else.
    let scenarios: std::collections::BTreeSet<&str> =
        rows.iter().map(|r| r[0]).collect();
    assert_eq!(
        scenarios,
        ["baseline", "ctrl"].into_iter().collect(),
        "one group per scenario"
    );

    // THE assertion: 5 draws per scenario, not 5 x 2 pooled.
    for r in &rows {
        assert_eq!(
            r[2], "5",
            "n_draws must be the PER-SCENARIO draw count; 10 means the arms were \
             pooled into one band (gh#562). Row: {r:?}"
        );
    }

    // Each arm contributes the same number of time rows.
    let n_base = rows.iter().filter(|r| r[0] == "baseline").count();
    let n_ctrl = rows.iter().filter(|r| r[0] == "ctrl").count();
    assert_eq!(n_base, n_ctrl, "both arms band over the same time grid");

    // The manifest tags one entry per (quantity, scenario).
    let mjson: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(qdir.join("quantities.json")).unwrap(),
    )
    .unwrap();
    let prev_entries: Vec<&serde_json::Value> = mjson["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|q| q["name"] == "prevalence")
        .collect();
    assert_eq!(prev_entries.len(), 2, "one manifest entry per (quantity, scenario)");
    let tagged: std::collections::BTreeSet<&str> = prev_entries
        .iter()
        .map(|q| q["scenario"].as_str().expect("scenario tag"))
        .collect();
    assert_eq!(tagged, ["baseline", "ctrl"].into_iter().collect());
    assert_eq!(
        mjson["calendar"]["time_unit"], "days",
        "calendar semantics travel with the artifact"
    );
}

/// With no `--scenario` there is no scenario axis, so no coordinate is emitted.
///
/// The runtime synthesizes a scenario named `baseline` internally to give the
/// grid a non-empty axis (`main.rs`), and `baseline` is an ordinary name a model
/// may declare — this fixture declares one. Emitting it would stamp
/// `scenario = baseline` on rows where the `baseline` preset was NOT applied: a
/// design coordinate naming a world the run did not simulate.
#[test]
fn simulate_without_scenario_emits_no_scenario_column() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    let qdir = tmp.path().join("q");
    let results_dir = tmp.path().join("results");

    let out = run(
        &bin,
        &[
            "simulate",
            model.to_str().unwrap(),
            "--draws", "uniform",
            "-n", "3",
            "--seed", "7",
            "--output-dir", results_dir.to_str().unwrap(),
            "--quantities-out", qdir.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));

    let txt = std::fs::read_to_string(qdir.join("quantities").join("prevalence.tsv")).unwrap();
    let header = txt.lines().next().unwrap();
    assert_eq!(
        header, "time\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "no --scenario means no scenario axis and no fabricated coordinate"
    );
}

#[test]
fn simulate_without_quantities_out_skips_with_a_note() {
    let bin = skip_if_missing();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());

    let out = run(
        &bin,
        &[
            "simulate",
            model.to_str().unwrap(),
            "--scenario",
            "baseline",
            "--seed",
            "1",
            "--output-dir",
            tmp.path().join("results").to_str().unwrap(),
        ],
    );
    // The run still succeeds — declaring quantities without emitting them is not
    // an error.
    assert!(
        out.status.success(),
        "simulate without --quantities-out should still succeed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--quantities-out"),
        "a note must point the user at --quantities-out, got:\n{stderr}"
    );
    // And NO quantities directory was written.
    assert!(
        !tmp.path().join("q").exists(),
        "no quantities sidecar without the flag"
    );
}
