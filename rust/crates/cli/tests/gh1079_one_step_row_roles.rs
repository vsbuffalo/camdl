//! gh#1079: `camdl fit predict` marks each predictive row with the part it
//! played in the fit, and `predictive.json` names both information sets
//! (`docs/dev/temporal-architecture.md` §2.4, §5.3).
//!
//! The one-step band filters through every row of the bound files, held-out
//! rows included, while the parameters came from the training rows alone.
//! Each row of `predictive/<stream>.tsv` therefore carries a `role`:
//! `trained` (in the fit's training prefix of its stream), `held_out` (a data
//! row after that prefix) or `target` (a free-forward row past the last data
//! row). Membership is the row's position in its stream, read from the fit's
//! recorded training partition.
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`: a
//! chain-binomial SIR reported daily (63 rows) and weekly (9 rows), fitted by
//! PMMH.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;

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

/// Everything the command said, on either channel.
fn said(o: &std::process::Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

/// Copy the fixture's data into `dir`, keeping the rows of `file` that `keep`
/// accepts (by their `time`).
fn copy_rows(dir: &Path, file: &str, to: &str, keep: &dyn Fn(f64) -> bool) {
    let text = std::fs::read_to_string(fixture(file)).unwrap();
    let mut lines = text.lines();
    let header = lines.next().unwrap().to_string();
    let body = lines.filter(|l| keep(l.split('\t').next().unwrap().parse().unwrap())).map(String::from);
    let out: String = std::iter::once(header).chain(body).map(|l| l + "\n").collect();
    std::fs::write(dir.join(to), out).unwrap();
}

/// Write `fit.toml` (`[data]` body `data`, then `tail`, e.g. a
/// `[data.holdout]` block) and run a small PMMH fit of both streams.
fn fit(dir: &Path, data: &str, tail: &str) {
    std::fs::copy(fixture("two_cadence_sir.camdl"), dir.join("model.camdl")).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data]
{data}
[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"
{tail}
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
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run: {}", said(&out));
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            if e.path().is_dir() { stack.push(e.path()) } else { found.push(e.path()) }
        }
    }
    found
}

/// Run `fit predict` (both modes unless `extra` says otherwise) and return
/// what it said and the directory it wrote `predictive.json` to.
fn predict(dir: &Path, extra: &[&str]) -> (String, PathBuf) {
    let mut args = vec!["fit", "predict", "--fit", "fit.toml", "--n-draws", "3", "--particles", "40"];
    args.extend_from_slice(extra);
    let out = run(dir, &args);
    assert!(out.status.success(), "fit predict: {}", said(&out));
    let manifests: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "predictive.json"))
        .collect();
    assert_eq!(manifests.len(), 1, "one predictive.json: {manifests:?}");
    (said(&out), manifests[0].parent().unwrap().to_path_buf())
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Rows of a TSV as header-keyed maps.
fn rows(path: &Path) -> Vec<HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split('\t').map(String::from).collect();
    lines.map(|l| header.iter().cloned().zip(l.split('\t').map(String::from)).collect()).collect()
}

/// `(time, role)` of the rows of `stream` on `horizon`, in file order.
fn roles(dir: &Path, stream: &str, horizon: &str) -> Vec<(f64, String)> {
    rows(&dir.join(format!("predictive/{stream}.tsv"))).into_iter()
        .filter(|r| r["horizon"] == horizon)
        .map(|r| (r["time"].parse().unwrap(), r.get("role").cloned().unwrap_or_else(|| "<no role column>".into())))
        .collect()
}

/// The roles a split at `cutoff` gives rows at `times`.
fn split_at(times: impl Iterator<Item = u32>, cutoff: u32) -> Vec<(f64, String)> {
    times.map(|t| (t as f64, if t <= cutoff { "trained" } else { "held_out" }.to_string())).collect()
}

fn header(dir: &Path, stream: &str) -> Vec<String> {
    let text = std::fs::read_to_string(dir.join(format!("predictive/{stream}.tsv"))).unwrap();
    text.lines().next().unwrap().split('\t').map(String::from).collect()
}

/// The headline. `holdout_after = 24`: the weekly rows ending 7, 14, 21 and
/// the daily rows ending 1 to 24 trained; every later row is held out, and
/// the one-step band still covers it. The manifest names the parameter set
/// (the fit's recorded partition) and each section's state set.
#[test]
fn rows_past_the_training_prefix_are_held_out_and_the_manifest_names_both_sets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|_| true);
    copy_rows(dir, "weekly_cases.tsv", "weekly_cases.tsv", &|_| true);
    fit(dir, "holdout_after = 24.0\n", "");
    let (text, out) = predict(dir, &[]);

    // The one-step rows: every data row, the held-out ones marked.
    assert_eq!(roles(&out, "weekly_cases", "one_step"), split_at((1..=9).map(|w| 7 * w), 24));
    assert_eq!(roles(&out, "daily_cases", "one_step"), split_at(1..=63, 24));
    // The free-forward rows sit at the same data rows and take the same roles.
    assert_eq!(roles(&out, "weekly_cases", "free_forward"), split_at((1..=9).map(|w| 7 * w), 24));
    assert_eq!(roles(&out, "daily_cases", "free_forward"), split_at(1..=63, 24));

    // `role` sits with the other coordinates, after `treatment`.
    let h = header(&out, "weekly_cases");
    let at = |c: &str| h.iter().position(|x| x == c).unwrap_or_else(|| panic!("no `{c}` column in {h:?}"));
    assert_eq!(at("role"), at("treatment") + 1, "{h:?}");

    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["schema"], "camdl.predictive/v4");
    for s in m["streams"].as_array().unwrap() {
        let coords: Vec<&str> = s["coordinates"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
        assert_eq!(&coords[coords.len() - 3..], ["horizon", "treatment", "role"], "{s}");
    }
    // The parameter set: the partition the fit recorded.
    let p = &m["parameter_set"];
    assert_eq!(p["rule"], json!({"kind": "cutoff", "at": {"t": 24.0}}));
    assert_eq!(p["extent"], json!({"t": 24.0}));
    let trained: Vec<(&str, u64)> = p["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap(), s["rows"].as_u64().unwrap())).collect();
    assert_eq!(trained, [("daily_cases", 24), ("weekly_cases", 3)]);
    // The state sets, one per section written.
    let os = &m["state_sets"]["one_step"];
    assert_eq!(os["conditioning"], "filtered");
    let filtered: Vec<(&str, u64, u64)> = os["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap(), s["rows"].as_u64().unwrap(), s["held_out_rows"].as_u64().unwrap()))
        .collect();
    assert_eq!(filtered, [("daily_cases", 63, 39), ("weekly_cases", 9, 6)]);
    assert_eq!(m["state_sets"]["free_forward"], json!({"conditioning": "none", "streams": []}));

    // Said once, in one sentence.
    let line = "fit predict: the one_step rows include held-out rows (39 of 63 in 'daily_cases', \
                6 of 9 in 'weekly_cases')";
    assert_eq!(text.matches(line).count(), 1, "{text}");
    assert!(text.contains("parameters fitted without them"), "{text}");
}

/// The control, and the forecast tail. No holdout, data through day 42 of a
/// model that runs to day 63: every data row trained, and the free-forward
/// rows past the last data row are targets. Nothing is said about held-out
/// rows.
#[test]
fn without_a_holdout_every_data_row_trained_and_the_tail_rows_are_targets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|t| t <= 42.0);
    copy_rows(dir, "weekly_cases.tsv", "weekly_cases.tsv", &|t| t <= 42.0);
    fit(dir, "", "");
    let (text, out) = predict(dir, &[]);

    assert_eq!(roles(&out, "weekly_cases", "one_step"), split_at((1..=6).map(|w| 7 * w), 42));
    assert_eq!(roles(&out, "daily_cases", "one_step"), split_at(1..=42, 42));
    let tail = |times: Vec<u32>| -> Vec<(f64, String)> {
        times.into_iter().map(|t| (t as f64, if t <= 42 { "trained" } else { "target" }.to_string())).collect()
    };
    assert_eq!(roles(&out, "weekly_cases", "free_forward"), tail((1..=9).map(|w| 7 * w).collect()));
    assert_eq!(roles(&out, "daily_cases", "free_forward"), tail((1..=63).collect()));

    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["parameter_set"]["rule"], json!({"kind": "all"}));
    assert_eq!(m["parameter_set"]["extent"], json!({"t": 42.0}));
    let held: Vec<u64> = m["state_sets"]["one_step"]["streams"].as_array().unwrap().iter()
        .map(|s| s["held_out_rows"].as_u64().unwrap()).collect();
    assert_eq!(held, [0, 0]);
    assert!(!text.contains("held-out rows"), "{text}");
}

/// `[data.holdout]` puts the held-out rows in their own file, which the
/// predictive does not read: its rows are the training files' rows, all
/// `trained`, and the manifest says the holdout files were not read.
#[test]
fn a_holdout_file_is_not_read_and_the_manifest_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|t| t <= 24.0);
    copy_rows(dir, "weekly_cases.tsv", "weekly_cases.tsv", &|t| t <= 24.0);
    copy_rows(dir, "weekly_cases.tsv", "weekly_hold.tsv", &|t| t > 24.0);
    fit(dir, "", "\n[data.holdout]\nweekly_cases = \"weekly_hold.tsv\"\n");
    let (text, out) = predict(dir, &["--mode", "one_step"]);

    assert_eq!(roles(&out, "weekly_cases", "one_step"), split_at((1..=3).map(|w| 7 * w), 24));
    assert_eq!(roles(&out, "daily_cases", "one_step"), split_at(1..=24, 24));
    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["parameter_set"]["rule"], json!({"kind": "separate_files"}));
    let os = &m["state_sets"]["one_step"];
    assert_eq!(os["holdout_files"], json!({"read": false, "sources": ["weekly_cases"]}));
    let filtered: Vec<(u64, u64)> = os["streams"].as_array().unwrap().iter()
        .map(|s| (s["rows"].as_u64().unwrap(), s["held_out_rows"].as_u64().unwrap())).collect();
    assert_eq!(filtered, [(24, 0), (3, 0)]);
    assert!(m["state_sets"].get("free_forward").is_none(), "no free-forward section was written");
    assert!(!text.contains("include held-out rows"), "{text}");
}

/// A fit that records no training partition is refused, with the sentence the
/// forecast mode uses: the roles are not guessed.
#[test]
fn a_fit_with_no_recorded_partition_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|_| true);
    copy_rows(dir, "weekly_cases.tsv", "weekly_cases.tsv", &|_| true);
    fit(dir, "holdout_after = 24.0\n", "");
    let meta = walk(&dir.join("results")).into_iter()
        .find(|p| p.file_name().is_some_and(|n| n == "fit.meta.json"))
        .expect("the fit's sidecar");
    let mut sidecar = read_json(&meta);
    assert!(sidecar.as_object_mut().unwrap().remove("training_partition").is_some(), "{sidecar}");
    std::fs::write(&meta, serde_json::to_string_pretty(&sidecar).unwrap()).unwrap();

    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--mode", "one_step", "--n-draws", "3"]);
    assert!(!out.status.success(), "{}", said(&out));
    assert!(said(&out).contains("records no training partition, so which rows its parameters were fitted to cannot be read"),
        "{}", said(&out));
    assert!(!walk(&dir.join("results")).iter().any(|p| p.file_name().is_some_and(|n| n == "predictive.json")),
        "nothing is written");
}
