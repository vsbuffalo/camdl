//! `camdl fit predict --horizon DURATION` (`--mode forecast`): the
//! fixed-origin forecast from the end of a fit's training rows (migration
//! plan, Stage 5; `docs/dev/temporal-architecture.md` §2.5, §4.3, §4.4).
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily and weekly, fitted by PMMH with
//! `holdout_after = 24`: the training rows end at day 24, inside the week
//! [21, 28).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    bin
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forecast").join(name)
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A PMMH fit of the fixture in `dir`, training rows through day 24. The
/// weekly file also carries rows past the cutoff, which are held out.
fn fit(dir: &Path, method: &str, extra_data: &str) {
    std::fs::copy(fixture("two_cadence_sir.camdl"), dir.join("model.camdl")).unwrap();
    std::fs::copy(fixture("daily_cases.tsv"), dir.join("daily_cases.tsv")).unwrap();
    std::fs::copy(fixture("weekly_cases.tsv"), dir.join("weekly_cases.tsv")).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data]
holdout_after = 24.0
{extra_data}
[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = {{ bounds = [0.2, 1.0], start = 0.5 }}

[fixed]
gamma = 0.25
rho   = 0.5
k     = 10.0
N0    = 1000
I0    = 5

{method}
"#)).unwrap();
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run: {}", stderr(&out));
}

const PMMH: &str = r#"[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 60
burn_in = 20
thin = 1"#;

/// The directory the latest predict wrote: `<segment>/predict/<run id>-<key>/`.
fn forecast_dir(dir: &Path) -> PathBuf {
    let mut found = Vec::new();
    for seg in std::fs::read_dir(dir.join("results/fits")).unwrap().flatten() {
        for d in std::fs::read_dir(seg.path().join("predict")).into_iter().flatten().flatten() {
            if d.path().join("forecast.json").is_file() {
                found.push(d.path());
            }
        }
    }
    assert_eq!(found.len(), 1, "one forecast directory: {found:?}");
    found.remove(0)
}

/// Rows of a TSV as header-keyed maps.
fn rows(path: &Path) -> Vec<std::collections::HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split('\t').map(String::from).collect();
    lines.map(|l| header.iter().cloned().zip(l.split('\t').map(String::from)).collect()).collect()
}

/// The headline: from the training extent (day 24, inside a week),
/// `--horizon "4 weeks"` forecasts the week under way as weekly horizon 0
/// and daily horizons 0–27; the targets are the declared periods, not the
/// most common gap; nothing after the origin is filtered, and the held-out
/// weekly rows name their targets.
#[test]
fn a_forecast_starts_at_the_training_extent_and_counts_each_streams_own_periods() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "4 weeks", "--n-draws", "8"]);
    assert!(out.status.success(), "fit predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "camdl.forecast/v1");
    assert_eq!(manifest["mode"], "fixed_origin");
    assert_eq!(manifest["origin"]["t"], 24.0, "the origin is the training extent");
    assert_eq!(manifest["horizon"]["duration"], 28.0);
    assert_eq!(manifest["state_set"]["through"]["t"], 24.0);
    let state_rows: Vec<(String, u64)> = manifest["state_set"]["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap().to_string(), s["rows"].as_u64().unwrap()))
        .collect();
    assert_eq!(state_rows, vec![("daily_cases".into(), 24), ("weekly_cases".into(), 3)],
        "the state set is the training rows: no row ending after the origin is filtered");

    let weekly = rows(&fdir.join("forecast/weekly_cases.tsv"));
    let spans: Vec<(&str, &str, &str, &str, &str)> = weekly.iter()
        .map(|r| (r["t_start"].as_str(), r["t_stop"].as_str(), r["horizon"].as_str(),
                  r["lead"].as_str(), r["held_out_row"].as_str()))
        .collect();
    assert_eq!(spans, vec![
        ("21", "28", "0", "4", "3"),
        ("28", "35", "1", "11", "4"),
        ("35", "42", "2", "18", "5"),
        ("42", "49", "3", "25", "6"),
    ], "the week under way is horizon 0; horizons count weeks; held-out rows name their targets");
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    assert_eq!(daily.len(), 28, "daily horizons 0–27");
    assert_eq!((daily[0]["t_start"].as_str(), daily[0]["t_stop"].as_str()), ("24", "25"));
    assert_eq!(daily[27]["horizon"], "27");
    for r in daily.iter().chain(&weekly) {
        assert_eq!(r["n_samples"], (8 * 500).to_string());
    }

    // Joint samples: one value per target per sample.
    let samples = rows(&fdir.join("forecast-samples.tsv"));
    assert_eq!(samples.len(), 8 * 5 * 32);
}

/// Flag rules (D41): `--horizon` alone is the forecast, `--mode forecast`
/// needs a reach, and a reach means nothing to another mode.
#[test]
fn the_mode_and_horizon_flags_are_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run(tmp.path(), &["fit", "predict", "x", "--mode", "forecast"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("--mode forecast needs --horizon"), "{}", stderr(&out));
    let out = run(tmp.path(), &["fit", "predict", "x", "--mode", "one_step", "--horizon", "4 weeks"]);
    assert!(stderr(&out).contains("sets the reach of a forecast"), "{}", stderr(&out));
}

/// The capability matrix's refused cells for a fit that exists here: an
/// optimizer (IF2) has no posterior, and scenarios are not arms yet.
#[test]
fn single_estimate_fits_and_scenarios_are_refused_a_forecast() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 3
cooling = 0.7"#, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "2 weeks"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("single"), "IF2 is a single estimate: {}", stderr(&out));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "2 weeks",
                         "--sweep", "rho=0.4,0.6"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("cannot be combined with --mode forecast"), "{}", stderr(&out));
}
