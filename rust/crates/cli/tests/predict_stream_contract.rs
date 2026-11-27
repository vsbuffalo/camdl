//! D76: which of the model's observation streams `camdl predict` covers, and
//! what its exit status says about them, in every mode.
//!
//! | situation                                                    | exit |
//! | ------------------------------------------------------------ | ---- |
//! | every eligible stream produced                               | 0    |
//! | every eligible stream produced, streams with no data skipped | 0    |
//! | a stream with no bound data asked for by `--stream`          | 1    |
//! | several streams asked for and any one cannot be produced     | 1    |
//! | no stream eligible                                           | 1    |
//! | an eligible stream fails while it is produced                | 1    |
//!
//! A stream is eligible when the fit bound data to it. The "no stream
//! eligible" row is `predict_streams::tests::d76_row5_no_eligible_stream_is_refused`:
//! a fit cannot run with no stream bound, so the CLI never reaches it except
//! through `--stream`, which is the third row.
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily (`daily_cases`) and weekly
//! (`weekly_cases`), fitted by PMMH with `holdout_after = 24`. A fit that
//! binds only the daily file leaves `weekly_cases` with no bound data.

mod fit_store_support;

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
        .env("CAMDL_STORE", dir.join("store"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const BOTH: &str = "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"";
const DAILY_ONLY: &str = "daily_cases = \"daily_cases.tsv\"";

/// Fit `model` (the fixture's text, possibly edited) in `dir` with the
/// `[data.observations]` bindings `bound`, labelled `fit`.
fn fit_model(dir: &Path, model: &str, bound: &str) {
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    std::fs::copy(fixture("daily_cases.tsv"), dir.join("daily_cases.tsv")).unwrap();
    std::fs::copy(fixture("weekly_cases.tsv"), dir.join("weekly_cases.tsv")).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(r#"store = "store"

[model]
camdl = "model.camdl"

[data]
holdout_after = 24.0

[data.observations]
{bound}

[estimate]
beta = {{ bounds = [0.2, 1.0], start = 0.5 }}

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
"#)).unwrap();
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run: {}", stderr(&out));
}

fn fit(dir: &Path, bound: &str) {
    fit_model(dir, &std::fs::read_to_string(fixture("two_cadence_sir.camdl")).unwrap(), bound);
}

/// The one prediction folder in the store under `dir` holding `file`.
fn predict_dir(dir: &Path, file: &str) -> PathBuf {
    let mut found: Vec<PathBuf> =
        fit_store_support::predictions(&dir.join("store")).into_iter().filter(|d| d.join(file).is_file()).collect();
    assert_eq!(found.len(), 1, "one predict directory holding {file}: {found:?}");
    found.remove(0)
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn names(v: &serde_json::Value) -> Vec<String> {
    v.as_array().unwrap_or_else(|| panic!("not a list: {v}")).iter()
        .map(|s| s.as_str().map(String::from).unwrap_or_else(|| s["stream"].as_str().unwrap().to_string()))
        .collect()
}

fn forecast(dir: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "2"];
    args.extend_from_slice(extra);
    run(dir, &args)
}

fn free_forward(dir: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["predict", "@fit", "--mode", "free_forward", "--n-draws", "2"];
    args.extend_from_slice(extra);
    run(dir, &args)
}

// ── Forecast mode ───────────────────────────────────────────────────────────

/// Row 1: both streams bound, none asked for: both forecast, exit 0, and
/// `forecast.json` says so.
#[test]
fn d76_row1_every_eligible_stream_is_forecast_and_the_exit_status_is_0() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), BOTH);
    let out = forecast(tmp.path(), &[]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let m = json(&predict_dir(tmp.path(), "forecast.json").join("forecast.json"));
    assert_eq!(m["schema"], "camdl.forecast/v6");
    assert_eq!(names(&m["produced"]), ["daily_cases", "weekly_cases"]);
    assert_eq!(m["skipped"], serde_json::json!([]));
    assert_eq!(m["failed"], serde_json::json!([]));
}

/// Row 2: the weekly stream has no bound data and nobody asked for it. It is
/// skipped with `no_bound_data`, named on stderr, in `report.json` and in
/// `forecast.json`; the daily forecast is written and the exit status is 0.
#[test]
fn d76_row2_an_unbound_stream_not_asked_for_is_skipped_and_the_exit_status_is_0() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = forecast(tmp.path(), &[]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(stderr(&out).contains("skipped stream `weekly_cases` (no_bound_data)"), "{}", stderr(&out));
    let fdir = predict_dir(tmp.path(), "forecast.json");
    assert!(fdir.join("forecast/daily_cases.tsv").is_file());
    assert!(!fdir.join("forecast/weekly_cases.tsv").exists());
    let m = json(&fdir.join("forecast.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases"]);
    assert_eq!(m["skipped"], serde_json::json!([{ "stream": "weekly_cases", "reason": "no_bound_data" }]));
    assert_eq!(m["failed"], serde_json::json!([]));
    let report = json(&fdir.join("report.json"));
    assert_eq!(report["failures"], serde_json::json!([]), "a skipped stream is not a failure");
    assert_eq!(report["skipped"], m["skipped"]);
}

/// Row 3: `--stream weekly_cases`, which has no bound data: exit 1, naming it.
#[test]
fn d76_row3_an_unbound_stream_asked_for_makes_the_exit_status_1() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = forecast(tmp.path(), &["--stream", "weekly_cases"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("weekly_cases") && err.contains("bound no data"), "{err}");
}

/// Row 4: both streams asked for, the weekly one with no bound data. The
/// daily forecast is still written; `forecast.json` lists the weekly stream
/// under `failed`, and the exit status is 1.
#[test]
fn d76_row4_several_streams_asked_for_and_one_cannot_be_produced_makes_the_exit_status_1() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = forecast(tmp.path(), &["--stream", "daily_cases", "--stream", "weekly_cases"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let fdir = predict_dir(tmp.path(), "forecast.json");
    assert!(fdir.join("forecast/daily_cases.tsv").is_file(), "the stream that could be produced is written");
    let m = json(&fdir.join("forecast.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases"]);
    assert_eq!(m["skipped"], serde_json::json!([]), "a stream asked for is never skipped");
    assert_eq!(names(&m["failed"]), ["weekly_cases"]);
    let report = json(&fdir.join("report.json"));
    assert_eq!(report["failures"][0]["at"]["forecast_stream"], "weekly_cases", "{report}");
}

/// Row 6: both streams bound, but the weekly one is a state read with no
/// `emit_schedule`, so it declares no rhythm to forecast: it fails, the daily
/// forecast is written, and the exit status is 1.
#[test]
fn d76_row6_an_eligible_stream_that_fails_makes_the_exit_status_1() {
    let tmp = tempfile::tempdir().unwrap();
    let model = std::fs::read_to_string(fixture("two_cadence_sir.camdl")).unwrap();
    let weekly = "    covers        = closing_at(time, 7 'days)\n    projected     = incidence(infection)\n    emit_schedule = every 7 'days\n    weekly_cases ~ neg_binomial(mean = rho * projected, r = k)";
    assert!(model.contains(weekly), "the fixture's weekly block moved");
    let model = model.replace(weekly, "    projected     = I\n    weekly_cases ~ poisson(rate = rho * projected)");
    fit_model(tmp.path(), &model, BOTH);
    let out = forecast(tmp.path(), &[]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let fdir = predict_dir(tmp.path(), "forecast.json");
    assert!(fdir.join("forecast/daily_cases.tsv").is_file());
    let m = json(&fdir.join("forecast.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases"]);
    assert_eq!(names(&m["failed"]), ["weekly_cases"]);
    assert!(m["failed"][0]["reason"].as_str().unwrap().contains("emit_schedule"), "{}", m["failed"]);
}

// ── Free-forward and one-step modes ─────────────────────────────────────────

/// Row 1 in the default mode (free-forward and one-step): both streams
/// produced, exit 0, and `predictive.json` says so.
#[test]
fn d76_row1_the_default_mode_produces_every_eligible_stream_and_exits_0() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), BOTH);
    let out = run(tmp.path(), &["predict", "@fit", "--n-draws", "2"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    let m = json(&predict_dir(tmp.path(), "predictive.json").join("predictive.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases", "weekly_cases"]);
    assert_eq!(m["skipped"], serde_json::json!([]));
    assert_eq!(m["failed"], serde_json::json!([]));
}

/// Row 2 under `--mode free-forward`: the unbound weekly stream is skipped by
/// name, not dropped silently, and the exit status is 0.
#[test]
fn d76_row2_free_forward_names_the_unbound_stream_it_skips_and_exits_0() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = free_forward(tmp.path(), &[]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(stderr(&out).contains("skipped stream `weekly_cases` (no_bound_data)"), "{}", stderr(&out));
    let pdir = predict_dir(tmp.path(), "predictive.json");
    let m = json(&pdir.join("predictive.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases"]);
    assert_eq!(m["skipped"], serde_json::json!([{ "stream": "weekly_cases", "reason": "no_bound_data" }]));
    assert_eq!(m["failed"], serde_json::json!([]));
    assert_eq!(json(&pdir.join("report.json"))["skipped"], m["skipped"]);
}

/// Row 3 under `--mode free-forward`: `--stream weekly_cases` alone.
#[test]
fn d76_row3_free_forward_an_unbound_stream_asked_for_makes_the_exit_status_1() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = free_forward(tmp.path(), &["--stream", "weekly_cases"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("stream `weekly_cases`") && stderr(&out).contains("bound no data"),
        "{}", stderr(&out));
}

/// Row 4 under `--mode free-forward`: the daily predictive is written, the
/// weekly stream is under `failed`, and the exit status is 1.
#[test]
fn d76_row4_free_forward_writes_the_streams_it_can_and_fails_the_one_it_cannot() {
    let tmp = tempfile::tempdir().unwrap();
    fit(tmp.path(), DAILY_ONLY);
    let out = free_forward(tmp.path(), &["--stream", "weekly_cases", "--stream", "daily_cases"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let pdir = predict_dir(tmp.path(), "predictive.json");
    assert!(pdir.join("predictive/daily_cases.tsv").is_file());
    let m = json(&pdir.join("predictive.json"));
    assert_eq!(names(&m["produced"]), ["daily_cases"]);
    assert_eq!(names(&m["failed"]), ["weekly_cases"]);
    assert_eq!(m["options"]["stream"], serde_json::json!(["daily_cases", "weekly_cases"]),
        "several streams are recorded as a set");
    assert_eq!(json(&pdir.join("report.json"))["failures"][0]["at"]["stream"], "weekly_cases");
}
