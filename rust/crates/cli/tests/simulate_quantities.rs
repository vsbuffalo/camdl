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
    assert_eq!(
        plines.next().unwrap(),
        "time\tvalue",
        "point series header is `time\\tvalue`, no n_draws/quantiles"
    );
    let prow: Vec<&str> = plines.next().expect("at least one prevalence row").split('\t').collect();
    assert_eq!(prow.len(), 2, "point series row is `time value`");
    assert!(prow[0].parse::<f64>().is_ok(), "time column numeric, got {:?}", prow[0]);
    assert!(prow[1].parse::<f64>().is_ok(), "value column numeric, got {:?}", prow[1]);

    // ── peak.tsv: a value scalar → bare `value` ──
    let peak = qdir.join("quantities").join("peak.tsv");
    let peak_txt = std::fs::read_to_string(&peak)
        .unwrap_or_else(|_| panic!("missing {}", peak.display()));
    let mut klines = peak_txt.lines();
    assert_eq!(klines.next().unwrap(), "value", "point scalar header is bare `value`");
    let krow = klines.next().expect("a peak row");
    assert!(krow.parse::<f64>().is_ok(), "peak value numeric, got {:?}", krow);

    // ── onset.tsv: a time scalar → bare `value`; a finite time OR `NA` ──
    let onset = qdir.join("quantities").join("onset.tsv");
    let onset_txt = std::fs::read_to_string(&onset)
        .unwrap_or_else(|_| panic!("missing {}", onset.display()));
    let mut olines = onset_txt.lines();
    assert_eq!(olines.next().unwrap(), "value", "point time-scalar header is bare `value`");
    let orow = olines.next().expect("an onset row");
    assert!(
        orow == "NA" || orow.parse::<f64>().is_ok(),
        "onset is a finite time or NA (right-censored), got {:?}",
        orow
    );

    // ── quantities.json: the manifest, one entry per logical quantity ──
    let manifest = qdir.join("quantities.json");
    let mtxt = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|_| panic!("missing {}", manifest.display()));
    let mjson: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    assert_eq!(mjson["schema"], "camdl.quantities/v1", "manifest schema tag");
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
