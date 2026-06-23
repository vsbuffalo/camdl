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
        header, "time\thorizon\ttreatment\trhat_max\tq05\tq25\tq50\tq75\tq95",
        "predictive header carries both axes + convergence + the quantile band"
    );
    let first = lines.next().expect("at least one predictive row");
    let cols: Vec<&str> = first.split('\t').collect();
    assert_eq!(cols.len(), 9, "row shape matches header");
    assert_eq!(cols[1], "free_forward", "horizon axis is explicit");
    assert_eq!(cols[2], "posterior", "treatment axis is explicit (not a plug-in)");
    // rhat_max is carried (a finite number), never silently blank for a PGAS fit.
    assert!(
        cols[3].parse::<f64>().is_ok(),
        "rhat_max carried on the band, got {:?}",
        cols[3]
    );
    // The quantile band is monotone non-decreasing q05 ≤ q25 ≤ … ≤ q95.
    let qs: Vec<f64> = cols[4..9].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    for w in qs.windows(2) {
        assert!(w[0] <= w[1], "quantiles must be ordered: {qs:?}");
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
