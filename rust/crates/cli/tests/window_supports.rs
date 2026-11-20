//! D106: a window-column stream's data state observation supports, exact
//! `[start, stop)` windows that may leave gaps and vary in width, not a
//! cadence. The D82 comparison of a stream's data with its `emit_schedule` no
//! longer applies to such a stream, at `fit predict --horizon` and at
//! `pfilter --save-final-state` alike.
//!
//! The fixture is `fixtures/real_shaped/surveillance.camdl` with its bulletin:
//! a dated `cases` stream of daily windows with a three-day and a five-day
//! window among them, declared `emit_schedule = every 1 'days`. Both commands
//! refused it before, citing a data window against the schedule's day (for
//! example [13, 16) against [13, 14)).
//!
//! End to end via the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> PathBuf {
    let b = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(b.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`", b.display());
    b
}

fn real_shaped(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/real_shaped").join(name)
}

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("store"))
        .output().expect("spawn camdl")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn fixture() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for f in ["surveillance.camdl", "bulletin.tsv"] {
        std::fs::copy(real_shaped(f), tmp.path().join(f)).unwrap();
    }
    tmp
}

/// `pfilter --save-final-state` with the irregular `cases` windows bound
/// saves: its windows are not checked against the daily schedule.
#[test]
fn the_surveillance_bulletin_saves_a_final_state() {
    let tmp = fixture();
    let out = run(tmp.path(), &[
        "pfilter", "surveillance.camdl",
        "--data", "cases=bulletin.tsv", "--data", "in_care=bulletin.tsv",
        "--param", "beta=0.35", "--param", "gamma=0.2", "--param", "rho=0.5",
        "--param", "psi=0.3", "--param", "k=20",
        "--particles", "20", "--seed", "1", "--save-final-state", "state.tsv",
    ]);
    let err = stderr(&out);
    assert!(out.status.success(), "the save must not be refused: {err}");
    assert!(!err.contains("refused --save-final-state"), "{err}");
    assert!(tmp.path().join("state.tsv").exists());
}

/// Every `forecast.json` under `root`.
fn forecast_manifests(root: &Path) -> Vec<PathBuf> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.file_name().is_some_and(|n| n == "forecast.json") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

/// `fit predict --horizon` forecasts `cases`: its targets are the schedule's
/// days after the last bulletin window, and it is not refused over a data
/// window that the schedule's days cut across.
#[test]
fn the_surveillance_bulletin_forecasts_cases() {
    let tmp = fixture();
    std::fs::write(tmp.path().join("fit.toml"), r#"output_dir = "results"
[model]
camdl = "surveillance.camdl"
[data.observations]
cases = "bulletin.tsv"
in_care = "bulletin.tsv"
[estimate]
beta = { bounds = [0.25, 0.5], start = 0.35 }
[fixed]
gamma = 0.2
rho = 0.5
psi = 0.3
k = 20
[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 30
iterations = 20
burn_in = 5
thin = 1
"#).unwrap();
    let fit = run(tmp.path(), &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(fit.status.success(), "fit run: {}", stderr(&fit));
    let out = run(tmp.path(), &["fit", "predict", "--fit", "fit.toml", "--horizon", "7 days", "--seed", "1"]);
    let err = stderr(&out);
    assert!(out.status.success(), "the forecast must succeed: {err}");
    assert!(!err.contains("stream 'cases' is refused"), "{err}");
    let manifests = forecast_manifests(&tmp.path().join("results"));
    assert_eq!(manifests.len(), 1, "{manifests:?}");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifests[0]).unwrap()).unwrap();
    let sources = v["target_sources"].as_array().expect("target_sources");
    assert!(sources.iter().any(|s| s["stream"] == "cases" && s["source"] == "emit_schedule"),
        "cases is forecast from its emit_schedule: {sources:?}");
}
