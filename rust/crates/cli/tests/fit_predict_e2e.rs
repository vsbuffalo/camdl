//! End-to-end acceptance for `camdl fit predict` — the free-forward posterior
//! predictive verb. Runs a real (tiny) PGAS fit, then `fit predict`, and checks
//! the two tidy artifacts have the typed-axis columns the proposal specifies.
//! Also checks the safety property: an optimizer (IF2) fit is refused, never
//! silently turned into a band.
//!
//! Proposal: docs/dev/proposals/2026-06-22-predictive-ergonomics.md

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
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation — small and well-behaved.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
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

quantities {
  prevalence = I / N                  # series  (one value per snapshot)
  peak       = max(I / N)             # value scalar (no censoring)
  onset      = first_above(I / N, 0.01)   # time scalar (right-censorable)
  onset2     = first_above(I / N, 0.02)   # time scalar
  spread     = onset2 - onset             # Derived over Time scalars; censorable
  peak_obs   = max(observations.weekly_cases)   # v1.1: reduce the per-draw y_sim
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

/// A short observed weekly series (rise-and-fall), times on the weekly grid.
const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(algorithm_block: &str, output_dir: &str) -> String {
    format!(
        r#"output_dir = "{output_dir}"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

{algorithm_block}
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent). Mirrors the runbook's ad-hoc guidance.
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    // results/fits/<stem>-<hash>/<sub>/<stream>.tsv
    let fits = root.join("fits");
    let entries = std::fs::read_dir(&fits).ok()?;
    for e in entries.flatten() {
        let p = e.path().join(sub).join(format!("{stream}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Find a file written directly into the fit segment (e.g. `quantities.json`).
fn find_segment_file(root: &Path, file: &str) -> Option<PathBuf> {
    let fits = root.join("fits");
    for e in std::fs::read_dir(&fits).ok()?.flatten() {
        let p = e.path().join(file);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
fn fit_predict_writes_posterior_predictive_and_observed_artifacts() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    let pgas = r#"[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(pgas, "results")).unwrap();

    // Run the fit.
    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Predict.
    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let results = tmp.join("results");

    // ── predictive/weekly_cases.tsv: typed-axis columns + quantile band ──
    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let mut lines = pred_txt.lines();
    let header = lines.next().unwrap();
    assert_eq!(
        header,
        "time\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "predictive header carries both axes + convergence (rhat_max, ess_min) + n_draws + band"
    );
    let first = lines.next().expect("at least one predictive row");
    let cols: Vec<&str> = first.split('\t').collect();
    assert_eq!(cols.len(), 11, "row shape matches header");
    assert_eq!(cols[1], "free_forward", "horizon axis is explicit");
    assert_eq!(cols[2], "posterior", "treatment axis is explicit (not a plug-in)");
    // rhat_max is carried (a finite number), never silently blank for a PGAS fit.
    assert!(
        cols[3].parse::<f64>().is_ok(),
        "rhat_max carried on the band, got {:?}",
        cols[3]
    );
    // n_draws is a positive count of the cloud the band was reduced over.
    assert!(
        cols[5].parse::<usize>().map(|n| n > 0).unwrap_or(false),
        "n_draws carried and positive, got {:?}",
        cols[5]
    );
    // The quantile band is monotone non-decreasing q05 ≤ q25 ≤ … ≤ q95.
    let qs: Vec<f64> = cols[6..11].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in qs.windows(2) {
        assert!(w[0] <= w[1], "quantiles must be ordered: {qs:?}");
    }

    // ── default emits BOTH horizons for a chain-binomial fit: the same file
    // also carries one_step rows (typed `horizon` column distinguishes them).
    let one_step_rows: Vec<&str> = pred_txt
        .lines()
        .filter(|l| l.split('\t').nth(1) == Some("one_step"))
        .collect();
    assert!(
        !one_step_rows.is_empty(),
        "default predict on a chain-binomial fit must also emit one_step rows; \
         got only:\n{pred_txt}"
    );
    // A one-step row is well-formed: posterior treatment, positive n_draws,
    // ordered quantile band.
    let osr: Vec<&str> = one_step_rows[0].split('\t').collect();
    assert_eq!(osr.len(), 11, "one_step row shape matches header");
    assert_eq!(osr[1], "one_step", "horizon axis");
    assert_eq!(osr[2], "posterior", "one-step is a posterior-treatment band");
    assert!(
        osr[5].parse::<usize>().map(|n| n > 0).unwrap_or(false),
        "one_step n_draws carried and positive (the subsample used), got {:?}",
        osr[5]
    );
    let osq: Vec<f64> = osr[6..11].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in osq.windows(2) {
        assert!(w[0] <= w[1], "one_step quantiles must be ordered: {osq:?}");
    }

    // ── observed/weekly_cases.tsv: the observed half, same time keys ──
    let obs = find_artifact(&results, "observed", "weekly_cases")
        .expect("observed/weekly_cases.tsv must be written");
    let obs_txt = std::fs::read_to_string(&obs).unwrap();
    let mut olines = obs_txt.lines();
    assert_eq!(olines.next().unwrap(), "time\tvalue", "observed header");
    // The observed value at t=28 is the planted peak, 1303.
    let peak = obs_txt.lines().find(|l| l.starts_with("28\t"));
    assert_eq!(peak, Some("28\t1303"), "observed series is the recorded data");

    // ── quantities/prevalence.tsv: a series (time + banded columns, no dims) ──
    let prev = find_artifact(&results, "quantities", "prevalence")
        .expect("quantities/prevalence.tsv must be written");
    let prev_txt = std::fs::read_to_string(&prev).unwrap();
    let mut plines = prev_txt.lines();
    assert_eq!(
        plines.next().unwrap(),
        "time\tn_draws\tq05\tq25\tq50\tq75\tq95",
        "series quantity header: time + banded columns"
    );
    let prow: Vec<&str> = plines.next().expect("at least one prevalence row").split('\t').collect();
    assert_eq!(prow.len(), 7, "series row shape matches header");
    let pq: Vec<f64> = prow[2..7].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in pq.windows(2) {
        assert!(w[0] <= w[1], "prevalence quantiles ordered: {pq:?}");
    }

    // ── quantities/peak.tsv: a value scalar (banded, NO censoring trio) ──
    let peakf = find_artifact(&results, "quantities", "peak")
        .expect("quantities/peak.tsv must be written");
    let peak_txt = std::fs::read_to_string(&peakf).unwrap();
    let mut klines = peak_txt.lines();
    assert_eq!(
        klines.next().unwrap(),
        "n_draws\tq05\tq25\tq50\tq75\tq95",
        "value-scalar header: banded columns, no time, no censoring"
    );
    let krow: Vec<&str> = klines.next().expect("a peak row").split('\t').collect();
    assert_eq!(krow.len(), 6, "value-scalar row shape matches header");

    // ── quantities/peak_obs.tsv: an observation-source value scalar ──────────
    // `max(observations.weekly_cases)` reduces the per-draw y_sim — same banded
    // value-scalar shape as a state reduction (a Value reduction never censors),
    // and the band must be finite (the obs series was materialized, not empty).
    let peakobsf = find_artifact(&results, "quantities", "peak_obs")
        .expect("quantities/peak_obs.tsv must be written");
    let po_txt = std::fs::read_to_string(&peakobsf).unwrap();
    let mut polines = po_txt.lines();
    assert_eq!(
        polines.next().unwrap(),
        "n_draws\tq05\tq25\tq50\tq75\tq95",
        "obs value-scalar header: banded columns, no censoring trio"
    );
    let porow: Vec<&str> = polines.next().expect("a peak_obs row").split('\t').collect();
    assert_eq!(porow.len(), 6, "obs value-scalar row shape matches header");
    let q50: f64 = porow[3].parse().expect("peak_obs q50 parses");
    assert!(
        q50.is_finite() && q50 > 0.0,
        "peak_obs median must be a finite positive count (y_sim materialized), got {q50}"
    );

    // ── quantities/onset.tsv: a time scalar (censorable → the censoring trio) ──
    let onsetf = find_artifact(&results, "quantities", "onset")
        .expect("quantities/onset.tsv must be written");
    let onset_txt = std::fs::read_to_string(&onsetf).unwrap();
    let mut olines2 = onset_txt.lines();
    assert_eq!(
        olines2.next().unwrap(),
        "n_draws\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95",
        "censorable scalar header carries the censoring trio"
    );
    let orow: Vec<&str> = olines2.next().expect("an onset row").split('\t').collect();
    assert_eq!(orow.len(), 9, "censorable row shape matches header");

    // ── quantities/spread.tsv: a Derived over Time scalars inherits censoring ──
    // `spread = onset2 - onset` propagates a censored endpoint, so it must carry
    // the censoring trio (not silently drop censored draws under a plain header).
    let spreadf = find_artifact(&results, "quantities", "spread")
        .expect("quantities/spread.tsv must be written");
    let spread_txt = std::fs::read_to_string(&spreadf).unwrap();
    assert_eq!(
        spread_txt.lines().next().unwrap(),
        "n_draws\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95",
        "a Derived transitively referencing a Time scalar inherits the censoring trio"
    );

    // ── quantities.json: lists all three logical quantities, typed ──
    let manifest = find_segment_file(&results, "quantities.json")
        .expect("quantities.json manifest must be written");
    let mtxt = std::fs::read_to_string(&manifest).unwrap();
    let mjson: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    assert_eq!(mjson["schema"], "camdl.quantities/v1", "manifest schema tag");
    let qs = mjson["quantities"].as_array().expect("quantities array");
    let lookup = |n: &str| qs.iter().find(|q| q["name"] == n).unwrap_or_else(|| panic!("manifest missing {n}"));
    assert_eq!(lookup("prevalence")["shape"], "series");
    assert_eq!(lookup("peak")["shape"], "scalar");
    assert_eq!(lookup("peak")["reduce"], "max");
    assert!(lookup("peak")["censoring"].is_null(), "a value reduction is not censorable");
    assert_eq!(lookup("onset")["shape"], "scalar");
    assert_eq!(lookup("onset")["reduce"], "first_above");
    assert!(lookup("onset")["censoring"].is_object(), "a time reduction records right-censoring");
    assert_eq!(lookup("peak_obs")["shape"], "scalar");
    assert_eq!(lookup("peak_obs")["reduce"], "max");
    assert!(lookup("peak_obs")["censoring"].is_null(), "an obs value reduction is not censorable");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fit_predict_refuses_an_optimizer_fit() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_predict_refuse_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    let if2 = r#"[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 25
cooling = 0.7
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(if2, "results")).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "if2 fit run failed");

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml"]);
    assert!(
        !out.status.success(),
        "predict must refuse an optimizer fit (no posterior cloud), not exit 0"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("optimizer fit") && stderr.contains("--params-only"),
        "refusal must be actionable, got: {stderr}"
    );
    // And it must NOT have written a band.
    assert!(
        find_artifact(&tmp.join("results"), "predictive", "weekly_cases").is_none(),
        "no predictive artifact for a point-estimate fit"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
