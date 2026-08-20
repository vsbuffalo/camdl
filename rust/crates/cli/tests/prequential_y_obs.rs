//! gh#268: `camdl pfilter --save-prequential` must record the real observed
//! value as each step's `y_obs`, not a hardcoded `0.0`.
//!
//! The bug: the prequential time axis was built as `Observation { time, value:
//! 0.0 }` (a never-scored placeholder) and that zero was read into the
//! prequential trace's `y_obs`. The predictive samples are then scored against
//! zeros — silent garbage `log_score`/`crps`/`pit`, inherited by `camdl
//! compare`. Regressed by PR #218 (the sparse/multi-cadence union axis), so it
//! affects single-stream AND multi-stream models.
//!
//! These tests pin: `y_obs[step]` equals the bound observed value(s) — the
//! per-stream sum across all bound streams on the union axis (matching the
//! joint, cross-stream predictive sample the score is computed against).
//!
//! gh#648 is the same bug at the OTHER call site. `fit run`'s PFilter stage
//! built its `y_obs` from `FitRunConfig.observations` — the canonical union
//! TIME axis, whose `value` is documented as a never-scored placeholder 0.0
//! (`fit/runner.rs`) — while `camdl pfilter --save-prequential` went through
//! `obs_model.joint_observed()`. The gh#268 fix landed at one site and missed
//! the other. `fit_run_prequential_equals_pfilter_prequential` below pins the
//! two against each other.

use sim::inference::prequential::PrequentialTrace;
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

fn single_block_model() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/sir_vaccination.ir.json")
}

fn synth_obs(bin: &Path, model: &Path, tmp: &Path, extra_args: &[&str]) -> PathBuf {
    let obs_path = tmp.join("obs.tsv");
    let mut cmd = Command::new(bin);
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

/// Parse a TSV with a header row into time -> value maps for the named columns.
/// Returns a map keyed by the (rounded) time column for robust alignment.
///
/// gh#269: the prequential TSV is now tidy/long with a `stream` column (a
/// `joint` summary row plus per-stream rows). This reader's intent is the JOINT
/// score per step, so when a `stream` column is present it keeps only the
/// `stream == "joint"` rows. Files without a `stream` column (the source obs
/// TSV) are read unfiltered.
fn read_tsv_columns(path: &Path, time_col: &str, value_cols: &[&str]) -> BTreeMap<i64, Vec<f64>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let tidx = header
        .iter()
        .position(|h| *h == time_col)
        .unwrap_or_else(|| panic!("no '{}' column in {:?}", time_col, header));
    let stream_idx = header.iter().position(|h| *h == "stream");
    let vidxs: Vec<usize> = value_cols
        .iter()
        .map(|c| {
            header
                .iter()
                .position(|h| h == c)
                .unwrap_or_else(|| panic!("no '{}' column in {:?}", c, header))
        })
        .collect();
    let mut out = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        // Keep only the joint summary row when this is a tidy prequential TSV.
        if let Some(si) = stream_idx {
            if f[si] != "joint" {
                continue;
            }
        }
        let t: f64 = f[tidx].parse().expect("time parse");
        let vals: Vec<f64> = vidxs.iter().map(|&i| f[i].parse().expect("value parse")).collect();
        out.insert(t.round() as i64, vals);
    }
    out
}

#[test]
fn prequential_y_obs_single_stream_equals_data() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &single_block_model(), tmp.path(), &[]);
    let stem = tmp.path().join("preq");

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter",
            &single_block_model().to_string_lossy(),
            "--data",
            &obs.to_string_lossy(),
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
        "pfilter --save-prequential failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let preq = read_tsv_columns(&stem.with_extension("tsv"), "t", &["y_obs"]);
    let data = read_tsv_columns(&obs, "time", &["reported_cases"]);
    assert!(!preq.is_empty(), "prequential trace empty");
    // The data must contain some nonzero observation, else the all-zero bug
    // could pass vacuously.
    assert!(
        data.values().any(|v| v[0] != 0.0),
        "synthetic obs are all zero — test would not detect the bug"
    );
    for (t, y) in &preq {
        let expected = data.get(t).unwrap_or_else(|| panic!("no data at t={}", t))[0];
        assert_eq!(
            y[0], expected,
            "prequential y_obs at t={} = {} but bound data = {} (the hardcoded-0 bug)",
            t, y[0], expected
        );
    }
}

#[test]
fn prequential_y_obs_multi_stream_is_cross_stream_sum() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(), &["--scenario", "true_params"]);
    let stem = tmp.path().join("preq");

    // Bind two streams (their own distinct columns of the same wide obs file).
    let out = Command::new(&bin)
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

    let preq = read_tsv_columns(&stem.with_extension("tsv"), "t", &["y_obs"]);
    let data = read_tsv_columns(&obs, "time", &["cases_p1", "cases_p2"]);
    assert!(!preq.is_empty(), "prequential trace empty");
    assert!(
        data.values().any(|v| v[0] + v[1] != 0.0),
        "synthetic obs are all zero — test would not detect the bug"
    );
    for (t, y) in &preq {
        let d = data.get(t).unwrap_or_else(|| panic!("no data at t={}", t));
        let expected = d[0] + d[1];
        assert_eq!(
            y[0], expected,
            "joint prequential y_obs at t={} = {} but cases_p1+cases_p2 = {} \
             (the hardcoded-0 bug)",
            t, y[0], expected
        );
    }
}

// ── gh#648: the same bug at `fit run`'s PFilter stage ──────────────────────

/// A closed SIR with a weekly NegBinomial observation. Every parameter is
/// pinned by the fit toml below, so the PFilter stage runs at a θ we can
/// reproduce exactly on the `camdl pfilter` command line.
const SIR_MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95]
  k     : positive     in [1.0, 100.0]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const SIR_DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// A single PFilter stage — no estimation. `start` values plus `[fixed]` fully
/// determine `FitRunConfig::base_params`, which is the θ the stage filters at,
/// so `SIR_THETA` below reproduces it verbatim for the manual `camdl pfilter`.
const SIR_FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[config]
dt = 1.0

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.5
k   = 10.0

[stages.score]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 300
"#;

const SIR_THETA: &str = "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 10\nrho = 0.5\nk = 10.0\n";

/// The lone `prequential.json` under a `fit run` output tree.
fn find_prequential_json(dir: &Path) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = find_prequential_json(&p) {
                return Some(found);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some("prequential.json") {
            return Some(p);
        }
    }
    None
}

fn read_trace(path: &Path) -> PrequentialTrace {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{} is not a PrequentialTrace: {e}", path.display()))
}

/// gh#648: `fit run`'s PFilter stage and `camdl pfilter --save-prequential`
/// must produce the SAME prequential trace at the same model / data / θ /
/// particles / seed. They call the same `bootstrap_filter` and the same
/// `build_trace`; only the `y_obs` they score against differed, because
/// `fit run` read the union time axis's placeholder value (always 0.0)
/// instead of `obs_model.joint_observed()`.
///
/// Symptom before the fix, on this fixture: `fit run` reported
/// `mean_crps=474.243` and `PIT 90% cov=0.00` where `camdl pfilter` reported
/// `93.321` and `0.88` — CRPS reading the forecast LEVEL rather than the
/// forecast ERROR, and PIT pinned at 0 so a calibrated model read as totally
/// miscalibrated. `elpd` was identical (-49.06) in both, because `elpd` never
/// reads `y_obs` — so one file carried a correct elpd beside a wrong CRPS with
/// nothing to signal that they disagreed.
#[test]
fn fit_run_prequential_equals_pfilter_prequential() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), SIR_MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), SIR_DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), SIR_FIT_TOML).unwrap();
    std::fs::write(dir.join("theta.toml"), SIR_THETA).unwrap();

    let run = |args: &[&str]| {
        Command::new(&bin)
            .args(args)
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    };

    let out = run(&["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run with a pfilter stage failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(&[
        "pfilter",
        "model.camdl",
        "--data",
        "weekly_cases=weekly_cases.tsv",
        "--params",
        "theta.toml",
        "--particles",
        "300",
        "--dt",
        "1",
        "--seed",
        "1",
        "--save-prequential",
        "manual",
    ]);
    assert!(
        out.status.success(),
        "pfilter --save-prequential failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let fit_json = find_prequential_json(&dir.join("results"))
        .expect("the pfilter stage wrote a prequential.json under results/");
    let from_fit = read_trace(&fit_json);
    let from_pfilter = read_trace(&dir.join("manual.json"));

    // Non-vacuity: the data must carry nonzero observations, or an all-zero
    // y_obs would match one that is merely absent.
    assert!(
        from_pfilter.steps.iter().any(|s| s.y_obs != 0.0),
        "fixture observations are all zero — this test could not detect the bug"
    );
    assert_eq!(
        from_fit.steps.len(),
        from_pfilter.steps.len(),
        "the two paths scored a different number of steps"
    );

    for (a, b) in from_fit.steps.iter().zip(from_pfilter.steps.iter()) {
        assert_eq!(a.t, b.t, "step times diverge");
        assert_eq!(
            a.y_obs, b.y_obs,
            "at t={}: fit run scored against y_obs={} but pfilter scored against \
             y_obs={} (fit run read the union axis's never-scored 0.0 placeholder)",
            a.t, a.y_obs, b.y_obs
        );
        assert_eq!(a.log_score, b.log_score, "log_score diverges at t={}", a.t);
        assert_eq!(
            a.crps, b.crps,
            "at t={}: crps {} vs {} — CRPS reads the forecast LEVEL, not the \
             forecast ERROR, when y_obs is zero",
            a.t, a.crps, b.crps
        );
        assert_eq!(
            a.pit, b.pit,
            "at t={}: pit {} vs {} — a PIT pinned at 0 reads as total \
             miscalibration",
            a.t, a.pit, b.pit
        );
        assert_eq!(a.ess, b.ess, "ess diverges at t={}", a.t);
    }

    assert_eq!(from_fit.elpd(), from_pfilter.elpd(), "elpd diverges");
    assert_eq!(from_fit.mean_crps(), from_pfilter.mean_crps(), "mean_crps diverges");
    assert_eq!(
        from_fit.pit_coverage(0.90),
        from_pfilter.pit_coverage(0.90),
        "PIT 90% coverage diverges"
    );

    // Catch-all: nothing else in the trace may differ either. Same filter, same
    // seed, same θ — the two files must be identical.
    let a: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fit_json).unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manual.json")).unwrap()).unwrap();
    assert_eq!(a, b, "the two prequential traces must be identical");
}
