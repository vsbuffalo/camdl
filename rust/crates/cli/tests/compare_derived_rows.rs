//! What `camdl compare` does around a row it predicts and scores itself: what
//! the row records of how it was computed, and which data it scores.
//!
//! Every test fits two small IF2 fits of the committed SEIR golden (no
//! `camdlc` needed), stores them in the store `fits/` labelled `a` and `b`,
//! and compares them by label, so each row is a prediction of the fit's
//! winning point, scored.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

fn golden_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/seir_observations.ir.json")
        .canonicalize()
        .unwrap()
}

const OBS: &str = "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n";

/// Write and run one IF2 fit in `dir/name`, training on `dir/name/obs.tsv`,
/// stored in `dir/fits` under the label `name`.
fn fit(dir: &Path, name: &str, beta: f64) -> PathBuf {
    let sub = dir.join(name);
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("obs.tsv"), OBS).unwrap();
    std::fs::write(sub.join("fit.toml"), format!(r#"
store = "../fits"

[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "obs.tsv"

[estimate.I0]
bounds = [1, 1000]
start  = 5

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = {beta}

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().display())).unwrap();
    let out = Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_STORE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", "fit.toml", "--seed", "1", "--label", name])
        .current_dir(&sub)
        .output().expect("spawn camdl fit run");
    assert!(out.status.success(), "fit run in {} failed:\n{}",
        sub.display(), String::from_utf8_lossy(&out.stderr));
    sub
}

/// `camdl compare @a @b` run from `dir`, reading the fits from the store
/// `dir/fits`.
fn compare(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["compare", "@a", "@b",
        "--particles", "100", "--draws", "1", "--replicates", "2"];
    args.extend_from_slice(extra);
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", dir.join("fits"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(&args)
        .current_dir(dir)
        .output().expect("spawn camdl compare")
}

/// The folder of the run labelled `label` in the store `dir/fits`: the run
/// whose `fit.toml` is the one `dir/<label>/fit.toml` holds.
fn run_of(dir: &Path, label: &str) -> PathBuf {
    let given = std::fs::read(dir.join(label).join("fit.toml")).unwrap();
    let runs: Vec<PathBuf> = fit_store_support::runs(&dir.join("fits")).into_iter()
        .filter(|r| std::fs::read(r.join("fit.toml")).ok().as_deref() == Some(&given[..]))
        .collect();
    assert_eq!(runs.len(), 1, "one run of {label}/fit.toml: {runs:?}");
    runs.into_iter().next().unwrap()
}

/// `compare` scores each fit against the training data its stored problem
/// holds (`inputs/<stream>.tsv`), not the file the fit file names. Editing
/// that file after the fit ran changes nothing `compare` reports: every row
/// scores as before, and no data change is stated.
#[test]
fn a_training_file_edited_after_the_fit_does_not_change_what_compare_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fit_a = fit(dir, "a", 0.10);
    fit(dir, "b", 0.15);

    let scores = |out: &Output| -> Vec<(String, f64)> {
        assert!(out.status.success(), "compare failed:\n{}", String::from_utf8_lossy(&out.stderr));
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
        v["rows"].as_array().unwrap().iter()
            .map(|r| (r["name"].as_str().unwrap().to_string(), r["elpd"].as_f64().unwrap()))
            .collect()
    };
    let before = scores(&compare(dir, &["--format", "json"]));

    let stored = std::fs::read_to_string(
        fit_store_support::problem_of(&run_of(dir, "a")).join("inputs/weekly_cases.tsv")).unwrap();
    assert_eq!(stored, OBS, "the problem stores the training data it was fitted to");
    std::fs::write(fit_a.join("obs.tsv"), OBS.replace("35\t5", "35\t9")).unwrap();

    let after = scores(&compare(dir, &["--format", "json"]));
    assert_eq!(before, after, "the edit must not change what compare scores");
}

/// A trace records how it was computed (schema 5): `pfilter
/// --save-prequential` writes its own single pass, and a row `compare`
/// predicts records what it did — the replicates it combined and where θ came
/// from. Two rows scored at different particle or replicate
/// counts are refused (exit 2) and render, stating it, under
/// `--allow-mismatched-filter-settings`.
#[test]
fn the_trace_records_its_filter_settings_and_compare_checks_them() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fit_a = fit(dir, "a", 0.10);
    fit(dir, "b", 0.15);

    std::fs::write(dir.join("theta.toml"),
        "sigma = 0.25\ngamma = 0.3\nrho = 0.5\nk = 10.0\np_detect = 0.5\n\
         N0 = 1000\nbeta = 0.1\nI0 = 5\n").unwrap();
    let out = Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_STORE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["pfilter", &golden_ir().to_string_lossy(),
               "--data", &format!("weekly_cases={}", fit_a.join("obs.tsv").display()),
               "--params", "theta.toml", "--particles", "200", "--dt", "1",
               "--seed", "3", "--save-prequential", "manual", "--no-store"])
        .current_dir(dir)
        .output().expect("spawn camdl pfilter");
    assert!(out.status.success(), "pfilter failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let manual: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manual.json")).unwrap()).unwrap();
    assert_eq!(manual["schema_version"], 7);
    assert_eq!(manual["basis"], "filtered", "a pfilter trace is a filter's estimate");
    // One pass: its own elpd and per-stream sum as one-element vectors, no
    // filter-noise SE.
    let elpd: f64 = manual["steps"].as_array().unwrap().iter()
        .map(|s| s["log_score"].as_f64().unwrap()).sum();
    let stream_elpd: f64 = manual["steps"].as_array().unwrap().iter()
        .flat_map(|s| s["per_stream"].as_array().unwrap().iter())
        .map(|s| s["log_score"].as_f64().unwrap()).sum();
    assert_eq!(manual["derivation"], serde_json::json!({
        "particles": 200, "replicates": 1, "dt": 1.0, "seed": 3,
        "theta": "supplied", "excluded_chains": [], "enabled": [], "disabled": [],
        "replicate_elpd": [elpd],
        "replicate_elpd_by_stream": {"weekly_cases": [stream_elpd]},
    }), "pfilter records its single pass: {manual}");

    // Derived rows: two replicates of an optimizer fit's winner.
    let out = compare(dir, &["--format", "json"]);
    assert!(out.status.success(), "compare failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    for row in v["rows"].as_array().unwrap() {
        let d = &row["derivation"];
        assert_eq!(d["particles"], 100, "{row}");
        assert_eq!(d["replicates"], 2, "the derive's replicate count, not the pass's: {row}");
        assert_eq!(d["seed"], 1, "the filter seed replicate 0 ran at: {row}");
        assert_eq!(d["theta"], "optimizer_winner", "an IF2 fit's θ is its winner: {row}");
        assert_eq!(d["excluded_chains"], serde_json::json!([]), "{row}");
        // The replicate totals and the filter-noise SE live in the trace,
        // and the row's SE is read from there.
        assert_eq!(d["replicate_elpd"].as_array().map(|v| v.len()), Some(2), "{row}");
        assert_eq!(d["replicate_elpd_by_stream"]["weekly_cases"].as_array().map(|v| v.len()),
            Some(2), "{row}");
        assert!(d["mc_se_elpd"].as_f64().is_some_and(|se| se.is_finite() && se >= 0.0),
            "two replicates give an SE: {row}");
        assert_eq!(row["mc_se_elpd"], d["mc_se_elpd"], "{row}");
    }
    assert!(v["filter_settings_mismatch"].is_null() && v["filter_settings_unchecked"].is_null(),
        "{v}");

    // The pfilter trace (200 particles, 1 replicate) against a derived row
    // (100 particles, 2 replicates).
    let run = |extra: &[&str]| {
        let mut args = vec!["compare", "manual.json", "@b",
            "--particles", "100", "--draws", "1", "--replicates", "2"];
        args.extend_from_slice(extra);
        Command::new(binary()).env_remove("CLICOLOR_FORCE")
            .env("CAMDL_STORE", dir.join("fits"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .args(&args).current_dir(dir)
            .output().expect("spawn camdl compare")
    };
    let out = run(&[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2),
        "different filter settings are a commensurability refusal:\n{stderr}");
    for needle in ["'manual.json'", "'@b'", "particles 200", "particles 100",
                   "replicates 1", "replicates 2", "biased downward",
                   "--allow-mismatched-filter-settings"] {
        assert!(stderr.contains(needle), "the refusal names {needle:?}:\n{stderr}");
    }

    let out = run(&["--allow-mismatched-filter-settings"]);
    assert!(out.status.success(), "the override renders:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let table = String::from_utf8_lossy(&out.stdout);
    assert!(table.contains("different particle-filter settings")
            && table.contains("--allow-mismatched-filter-settings was set"),
        "the table states the mismatch:\n{table}");
    let out = run(&["--allow-mismatched-filter-settings", "--format", "json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert!(v["filter_settings_mismatch"].as_str().is_some_and(|s| s.contains("particles 200")),
        "{v}");
}
