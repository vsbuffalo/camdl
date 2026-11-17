//! D78: every `fit predict` mode draws with the binomial sampler its fit ran
//! (BTPE or BTRS, read from the fit leaf's `run.json`), and records it.
//!
//! The guarantee is the same sampler and recorded configuration, not the fit's
//! realised random numbers: nothing here compares a predict's draws with the
//! fit's.
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily and weekly, fitted by PMMH.

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

/// A PMMH fit of the fixture in `dir`, with `extra` appended to `fit run`.
/// Returns the fit leaf (the directory holding its `run.json`).
fn fit(dir: &Path, extra: &[&str]) -> PathBuf {
    for f in ["two_cadence_sir.camdl", "daily_cases.tsv", "weekly_cases.tsv"] {
        std::fs::copy(fixture(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fit.toml"), r#"output_dir = "results"

[model]
camdl = "two_cadence_sir.camdl"

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
    let mut args = vec!["fit", "run", "fit.toml", "--seed", "1"];
    args.extend_from_slice(extra);
    let out = run(dir, &args);
    assert!(out.status.success(), "fit run: {}", stderr(&out));
    let leaves: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "run.json"))
        .map(|p| p.parent().unwrap().to_path_buf())
        .collect();
    assert_eq!(leaves.len(), 1, "{leaves:?}");
    leaves[0].clone()
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

/// `fit predict <leaf> --mode <mode>`; returns `predictive.json` and the
/// daily stream's predictive table.
fn predict(dir: &Path, leaf: &Path, mode: &str) -> (serde_json::Value, String) {
    let out = run(dir, &["fit", "predict", leaf.to_str().unwrap(), "--mode", mode, "--n-draws", "4"]);
    assert!(out.status.success(), "fit predict --mode {mode}: {}", stderr(&out));
    let manifests: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "predictive.json"))
        .filter(|p| {
            let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            m["options"]["mode"] == mode
        })
        .collect();
    assert_eq!(manifests.len(), 1, "{manifests:?}");
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifests[0]).unwrap()).unwrap();
    let tsv = std::fs::read_to_string(manifests[0].parent().unwrap().join("predictive/daily_cases.tsv")).unwrap();
    (m, tsv)
}

/// A fit run with `--binomial btrs` records `btrs` in its leaf; the one-step
/// band and the free-forward replay of that leaf record `btrs` too.
#[test]
fn d78_a_btrs_fits_one_step_and_free_forward_record_btrs() {
    let tmp = tempfile::tempdir().unwrap();
    let leaf = fit(tmp.path(), &["--binomial", "btrs"]);
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    assert_eq!(record["inputs"]["binomial"], "btrs");
    for mode in ["one_step", "free_forward"] {
        let (m, _) = predict(tmp.path(), &leaf, mode);
        assert_eq!(m["binomial"], "btrs", "--mode {mode}");
    }
}

/// The same posterior predicted twice, with the leaf's recorded sampler set to
/// BTPE and then to BTRS: each mode records the leaf's sampler, and its draws
/// change with it, so the record is the sampler the mode drew with. Only the
/// record differs between the two runs (same draws, same seed), which is what
/// makes a difference in the tables attributable to the sampler.
#[test]
fn d78_one_step_and_free_forward_draw_with_the_fits_recorded_sampler() {
    let tmp = tempfile::tempdir().unwrap();
    let leaf = fit(tmp.path(), &[]);
    let run_json = leaf.join("run.json");
    let mut record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&run_json).unwrap()).unwrap();
    assert_eq!(record["inputs"]["binomial"], "btpe");
    let btpe: Vec<(serde_json::Value, String)> =
        ["one_step", "free_forward"].iter().map(|m| predict(tmp.path(), &leaf, m)).collect();

    record["inputs"]["binomial"] = serde_json::json!("btrs");
    std::fs::write(&run_json, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    let btrs: Vec<(serde_json::Value, String)> =
        ["one_step", "free_forward"].iter().map(|m| predict(tmp.path(), &leaf, m)).collect();

    for (mode, (a, b)) in ["one_step", "free_forward"].iter().zip(btpe.iter().zip(&btrs)) {
        assert_eq!((a.0["binomial"].as_str(), b.0["binomial"].as_str()), (Some("btpe"), Some("btrs")),
            "--mode {mode} records the leaf's sampler");
        assert!(a.1 != b.1, "--mode {mode}: the predictive draws with the recorded sampler");
    }
}
