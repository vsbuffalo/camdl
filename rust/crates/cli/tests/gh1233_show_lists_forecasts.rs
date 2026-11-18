//! gh#1233: `camdl show` lists the fixed-origin forecasts that
//! `fit predict --mode forecast` writes under `predict/<run12>-<key>/`, and
//! states each one's origin, horizon, chain selection and particle count.
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily and weekly, fitted by PMMH (two chains,
//! 100 particles) with `holdout_after = 24`, so a forecast's origin is day 24.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    bin
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("NO_COLOR", "1")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn ok(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert!(out.status.success(), "camdl {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A PMMH fit of the forecast fixture in `dir`; returns its problem folder.
fn fit(dir: &Path) -> PathBuf {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forecast");
    std::fs::copy(fixtures.join("two_cadence_sir.camdl"), dir.join("model.camdl")).unwrap();
    for f in ["daily_cases.tsv", "weekly_cases.tsv"] {
        std::fs::copy(fixtures.join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fit.toml"), r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data]
holdout_after = 24.0

[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = { bounds = [0.2, 1.0], start = 0.5 }

[fixed]
gamma = 0.25
rho   = 0.5
k     = 10.0
N0    = 1000
I0    = 5

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 60
burn_in = 20
thin = 1
"#).unwrap();
    ok(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    let segs: Vec<PathBuf> = std::fs::read_dir(dir.join("results/fits")).unwrap().flatten().map(|e| e.path()).collect();
    assert_eq!(segs.len(), 1, "{segs:?}");
    segs[0].clone()
}

/// The forecast directories under `segment`, segment-relative with a
/// trailing `/` as `show` lists them, keyed by whether the forecast left a
/// chain out.
fn forecast_dirs(segment: &Path) -> (String, String) {
    let mut full = None;
    let mut subset = None;
    for d in std::fs::read_dir(segment.join("predict")).unwrap().flatten() {
        let m: serde_json::Value = match std::fs::read_to_string(d.path().join("forecast.json")) {
            Ok(t) => serde_json::from_str(&t).unwrap(),
            Err(_) => continue,
        };
        let rel = format!("predict/{}/", d.file_name().to_string_lossy());
        if m["options"]["excluded_chains"].as_array().unwrap().is_empty() { full = Some(rel) } else { subset = Some(rel) }
    }
    (full.expect("the full-cloud forecast"), subset.expect("the chain-subset forecast"))
}

/// The block of `show`'s output that follows `heading` (a line with no
/// indent), up to the next unindented line.
fn block<'a>(stdout: &'a str, heading: &str) -> Vec<&'a str> {
    stdout
        .lines()
        .skip_while(|l| *l != heading)
        .skip(1)
        .take_while(|l| l.starts_with(' '))
        .collect()
}

/// The lines under `dir` in a `forecasts` block, up to the next directory.
fn forecast_entry<'a>(forecasts: &[&'a str], dir: &str) -> Vec<&'a str> {
    let head = format!("  {dir}");
    let at = forecasts.iter().position(|l| *l == head)
        .unwrap_or_else(|| panic!("{dir} is in the forecasts block:\n{}", forecasts.join("\n")));
    forecasts[at + 1..].iter().take_while(|l| l.starts_with("    ")).copied().collect()
}

#[test]
fn show_lists_each_forecast_with_its_origin_horizon_chains_and_particles() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let segment = fit(dir);
    let seg = segment.to_string_lossy().into_owned();
    ok(dir, &["fit", "predict", &seg, "--mode", "forecast", "--horizon", "4 weeks"]);
    ok(dir, &["fit", "predict", &seg, "--mode", "forecast", "--horizon", "2 weeks",
              "--exclude-chains", "1", "--particles", "50"]);
    let (full, subset) = forecast_dirs(&segment);

    // The folder view: every forecast file is a listed output `cat --stream`
    // takes, and each forecast has its own entry.
    let shown = ok(dir, &["show", &seg]);
    let outputs = block(&shown, "outputs");
    for d in [&full, &subset] {
        for f in ["forecast.json", "forecast-samples.tsv", "forecast/daily_cases.tsv", "forecast/weekly_cases.tsv"] {
            let want = format!("  {d}{f}");
            assert!(outputs.contains(&want.as_str()), "{want} listed under outputs:\n{shown}");
        }
    }
    let cat = ok(dir, &["cat", &seg, "--stream", &format!("{full}forecast/weekly_cases.tsv")]);
    assert!(cat.starts_with("scenario\t"), "a listed forecast path is one cat takes:\n{cat}");

    let forecasts = block(&shown, "forecasts");
    let e = forecast_entry(&forecasts, &full);
    assert_eq!(e, vec![
        "    origin     t = 24",
        "    horizon    4 weeks (28 days)",
        "    chains     all (80 posterior draws)",
        "    particles  100 per draw (the PMMH fit's filter)",
    ], "{shown}");
    let e = forecast_entry(&forecasts, &subset);
    assert_eq!(e, vec![
        "    origin     t = 24",
        "    horizon    2 weeks (14 days)",
        "    chains     kept 0, left out 1, of 2 (40 posterior draws)",
        "    particles  50 per draw (--particles)",
    ], "{shown}");

    // The run view lists the same forecasts.
    let m: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(segment.join(&full).join("forecast.json")).unwrap()).unwrap();
    let run12 = &m["run_id"].as_str().unwrap()[..12];
    let shown = ok(dir, &["show", run12, "--root", "results"]);
    let forecasts = block(&shown, "forecasts");
    assert_eq!(forecast_entry(&forecasts, &full).len(), 4, "{shown}");
    assert_eq!(forecast_entry(&forecasts, &subset).len(), 4, "{shown}");
}
