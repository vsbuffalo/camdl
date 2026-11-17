//! `fit predict` writes under `<segment>/predict/<run id>-<key>/`, and readers
//! find a prediction by that address. The key must therefore cover every input
//! that changes what predict writes: two predictions computed from different
//! inputs never share a directory (proposal 2026-11-17, phase 1e).
//!
//! The case pinned end to end is the chain selection in forecast mode. The
//! forecast's `forecast.json` and `forecast-samples.tsv` carry no chain-subset
//! suffix, so before the fix `--exclude-chains 1` wrote them over the pooled
//! forecast's, in the pooled forecast's directory: whoever read that directory
//! afterwards read a one-chain forecast as the pooled one.
//!
//! Fixture: `tests/fixtures/forecast/two_cadence_sir.camdl`, fitted by a
//! two-chain PMMH with `holdout_after = 24`.

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
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn assert_ok(o: &std::process::Output, what: &str) {
    assert!(o.status.success(), "{what}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
}

fn fit(dir: &Path) {
    for f in ["two_cadence_sir.camdl", "daily_cases.tsv", "weekly_cases.tsv"] {
        let to = if f.ends_with(".camdl") { "model.camdl" } else { f };
        std::fs::copy(fixture(f), dir.join(to)).unwrap();
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
    assert_ok(&run(dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
}

/// Every directory under the fit's `predict/` holding a `forecast.json`.
fn forecast_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for seg in std::fs::read_dir(dir.join("results/fits")).unwrap().flatten() {
        for d in std::fs::read_dir(seg.path().join("predict")).into_iter().flatten().flatten() {
            if d.path().join("forecast.json").is_file() {
                found.push(d.path());
            }
        }
    }
    found.sort();
    found
}

/// The chains the forecast in `d` drew from, per its manifest.
fn chains_drawn(d: &Path) -> std::collections::BTreeSet<u64> {
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("forecast.json")).unwrap()).unwrap();
    j["draws"].as_array().expect("draws").iter()
        .map(|r| r["chain"].as_u64().expect("a chain key")).collect()
}

#[test]
fn two_chain_selections_are_two_forecast_addresses() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir);

    let base = ["fit", "predict", "--fit", "fit.toml", "--horizon", "2 weeks", "--n-draws", "8"];
    assert_ok(&run(dir, &base), "pooled forecast");
    let pooled = forecast_dirs(dir);
    assert_eq!(pooled.len(), 1, "{pooled:?}");
    let pooled = pooled[0].clone();
    assert_eq!(chains_drawn(&pooled), [0, 1].into(), "the control: the pooled forecast draws both chains");
    let pooled_samples = std::fs::read(pooled.join("forecast-samples.tsv")).unwrap();

    let mut excl = base.to_vec();
    excl.extend(["--exclude-chains", "1"]);
    assert_ok(&run(dir, &excl), "forecast --exclude-chains 1");

    // A reader that found the pooled forecast at its address must still find
    // the pooled forecast there.
    assert_eq!(chains_drawn(&pooled), [0, 1].into(),
        "the pooled forecast's directory now holds a chain-subset forecast: \
         `--exclude-chains 1` wrote into the pooled forecast's address");
    assert_eq!(std::fs::read(pooled.join("forecast-samples.tsv")).unwrap(), pooled_samples,
        "the pooled forecast's samples were overwritten");

    let dirs = forecast_dirs(dir);
    assert_eq!(dirs.len(), 2, "two selections, two addresses: {dirs:?}");
    let other = dirs.iter().find(|d| **d != pooled).unwrap();
    assert_eq!(chains_drawn(other), [0].into(), "the subset forecast draws chain 0 only");

    // Each manifest records the key of the directory it sits in, in full.
    for d in &dirs {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("forecast.json")).unwrap()).unwrap();
        let key = j["address_key"].as_str().expect("forecast.json records its address key");
        assert!(key.len() >= 32 && key.chars().all(|c| c.is_ascii_hexdigit()),
            "the key is at least 128 bits of hex: {key}");
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with(&format!("-{key}")), "{name} is addressed by {key}");
    }
}
