//! gh#1079: `camdl predict` marks each predictive row with the part it
//! played in the fit, and `predictive.json` names both information sets
//! (`docs/engineering/forecasting.md` §1.4, §1.5).
//!
//! The one-step band filters through every row of the bound files, held-out
//! rows included, while the parameters came from the training rows alone.
//! Each row of `predictive/<stream>.tsv` therefore carries a `role`:
//! `trained` (in the fit's training prefix of its stream), `held_out` (a row
//! of the bound file after that prefix) or `future` (a free-forward row past
//! the last row of the bound file: the prediction read no data row for that
//! period). Membership is the row's position in its stream, read from the
//! fit's recorded training partition.
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
        .env("CAMDL_STORE", dir.join("results"))
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
    fit_model(dir, &std::fs::read_to_string(fixture("two_cadence_sir.camdl")).unwrap(), data, tail);
}

/// [`fit`], of the model source `model`.
fn fit_model(dir: &Path, model: &str, data: &str, tail: &str) {
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    std::fs::write(dir.join("fit.toml"), format!(r#"store = "results"

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
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
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

/// Run `camdl predict` (both modes unless `extra` says otherwise) and return
/// what it said and the directory it wrote `predictive.json` to.
fn predict(dir: &Path, extra: &[&str]) -> (String, PathBuf) {
    // Each call's options key their own directory; keep only this call's.
    for p in walk(&dir.join("results")).into_iter().filter(|p| p.file_name().is_some_and(|n| n == "predictive.json")) {
        std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
    }
    let mut args = vec!["predict", "@fit", "--n-draws", "3"];
    // `--particles` is refused when no filter runs.
    if !extra.contains(&"free_forward") {
        args.extend_from_slice(&["--particles", "40"]);
    }
    args.extend_from_slice(extra);
    let out = run(dir, &args);
    assert!(out.status.success(), "predict: {}", said(&out));
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
        .filter(|r| r["mode"] == horizon)
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

    // The one-step rows: the training rows. Filtering through a held-out row
    // needs its value, which a stored fit does not keep, so the band stops
    // at the training prefix.
    let trained_only = |rows: Vec<(f64, String)>| -> Vec<(f64, String)> { rows.into_iter().filter(|(_, r)| r == "trained").collect() };
    assert_eq!(roles(&out, "weekly_cases", "one_step"), trained_only(split_at((1..=9).map(|w| 7 * w), 24)));
    assert_eq!(roles(&out, "daily_cases", "one_step"), trained_only(split_at(1..=63, 24)));
    // The free-forward rows sit at the same data rows and take the same roles.
    assert_eq!(roles(&out, "weekly_cases", "free_forward"), split_at((1..=9).map(|w| 7 * w), 24));
    assert_eq!(roles(&out, "daily_cases", "free_forward"), split_at(1..=63, 24));

    // `role` sits with the other coordinates, after `treatment`.
    let h = header(&out, "weekly_cases");
    let at = |c: &str| h.iter().position(|x| x == c).unwrap_or_else(|| panic!("no `{c}` column in {h:?}"));
    assert_eq!(at("role"), at("treatment") + 1, "{h:?}");

    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["schema"], "camdl.predictive/v6");
    for s in m["streams"].as_array().unwrap() {
        let coords: Vec<&str> = s["coordinates"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
        assert_eq!(&coords[coords.len() - 5..], ["mode", "periods_ahead", "treatment", "role", "quantity"], "{s}");
    }
    // The parameter set: the partition the fit recorded.
    let p = &m["training_rows"];
    assert_eq!(p["rule"], json!({"kind": "cutoff", "at": {"t": 24.0}}));
    assert_eq!(p["extent"], json!({"t": 24.0}));
    let trained: Vec<(&str, u64)> = p["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap(), s["rows"].as_u64().unwrap())).collect();
    assert_eq!(trained, [("daily_cases", 24), ("weekly_cases", 3)]);
    // The state sets, one per section written.
    let os = &m["conditioning_rows"]["one_step"];
    assert_eq!(os["conditioning"], "filtered");
    let filtered: Vec<(&str, u64, u64)> = os["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap(), s["rows"].as_u64().unwrap(), s["held_out_rows"].as_u64().unwrap()))
        .collect();
    // The filter walks the stored training rows; the held-out ones are not
    // stored with their values.
    assert_eq!(filtered, [("daily_cases", 24, 0), ("weekly_cases", 3, 0)]);
    assert!(os.get("through").is_none(), "the counts say what was filtered through: {os}");
    assert_eq!(m["conditioning_rows"]["free_forward"], json!({"conditioning": "none", "streams": []}));
    // Written, as `null`, without a `[data.holdout]`: the key is not missing.
    assert_eq!(m.as_object().unwrap().get("holdout_files"), Some(&serde_json::Value::Null));

    // The one-step band holds no held-out row, so nothing is said of one.
    assert!(!text.contains("the one_step rows include held-out rows"), "{text}");

    // Under `--stream`, the state set names what was filtered.
    let (text, out) = predict(dir, &["--mode", "one_step", "--stream", "weekly_cases"]);
    assert!(!text.contains("held-out rows"), "{text}");
    let m = read_json(&out.join("predictive.json"));
    let filtered: Vec<&str> = m["conditioning_rows"]["one_step"]["streams"].as_array().unwrap().iter()
        .map(|s| s["stream"].as_str().unwrap()).collect();
    assert_eq!(filtered, ["weekly_cases"]);
    assert_eq!(roles(&out, "weekly_cases", "one_step"), trained_only(split_at((1..=9).map(|w| 7 * w), 24)));
}

/// The control, and the rows past the data. No holdout, data through day 42
/// of a model that runs to day 63: every data row trained, and the
/// free-forward rows past the last data row are `future` (the prediction read
/// no data row there). Nothing is said about held-out rows.
#[test]
fn without_a_holdout_every_data_row_trained_and_the_rows_past_the_data_are_future() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|t| t <= 42.0);
    copy_rows(dir, "weekly_cases.tsv", "weekly_cases.tsv", &|t| t <= 42.0);
    fit(dir, "", "");
    let (text, out) = predict(dir, &[]);

    assert_eq!(roles(&out, "weekly_cases", "one_step"), split_at((1..=6).map(|w| 7 * w), 42));
    assert_eq!(roles(&out, "daily_cases", "one_step"), split_at(1..=42, 42));
    let tail = |times: Vec<u32>| -> Vec<(f64, String)> {
        times.into_iter().map(|t| (t as f64, if t <= 42 { "trained" } else { "future" }.to_string())).collect()
    };
    assert_eq!(roles(&out, "weekly_cases", "free_forward"), tail((1..=9).map(|w| 7 * w).collect()));
    assert_eq!(roles(&out, "daily_cases", "free_forward"), tail((1..=63).collect()));

    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["training_rows"]["rule"], json!({"kind": "all"}));
    assert_eq!(m["training_rows"]["extent"], json!({"t": 42.0}));
    let held: Vec<u64> = m["conditioning_rows"]["one_step"]["streams"].as_array().unwrap().iter()
        .map(|s| s["held_out_rows"].as_u64().unwrap()).collect();
    assert_eq!(held, [0, 0]);
    assert!(!text.contains("held-out rows"), "{text}");
}

/// `[data.holdout]` puts the held-out rows in their own file, which the
/// predictive does not read (gh#1309): the one-step rows are the training
/// files' rows, all `trained`; the free-forward rows in the periods the
/// holdout file covers are `future`, not `held_out`; and the manifest says,
/// in either mode, that the holdout files were not read.
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
    assert_eq!(m["training_rows"]["rule"], json!({"kind": "separate_files"}));
    let not_read = json!({"read": false, "sources": ["weekly_cases"]});
    assert_eq!(m["holdout_files"], not_read);
    let os = &m["conditioning_rows"]["one_step"];
    let filtered: Vec<(u64, u64)> = os["streams"].as_array().unwrap().iter()
        .map(|s| (s["rows"].as_u64().unwrap(), s["held_out_rows"].as_u64().unwrap())).collect();
    assert_eq!(filtered, [(24, 0), (3, 0)]);
    assert!(m["conditioning_rows"].get("free_forward").is_none(), "no free-forward section was written");
    assert!(!text.contains("include held-out rows"), "{text}");

    // Free-forward alone: the training rows, then the periods no data row was
    // read for, the weeks the holdout file covers among them.
    let (_, out) = predict(dir, &["--mode", "free_forward"]);
    let past = |times: Vec<u32>| -> Vec<(f64, String)> {
        times.into_iter().map(|t| (t as f64, if t <= 24 { "trained" } else { "future" }.to_string())).collect()
    };
    assert_eq!(roles(&out, "weekly_cases", "free_forward"), past((1..=9).map(|w| 7 * w).collect()));
    assert_eq!(roles(&out, "daily_cases", "free_forward"), past((1..=63).collect()));
    let m = read_json(&out.join("predictive.json"));
    assert_eq!(m["holdout_files"], not_read, "a free-forward manifest says so too");
    assert!(m["conditioning_rows"].get("one_step").is_none());
}

/// Position, not label against the cutoff. The weekly stream is labelled by
/// each week's start (`starting_on`), and `holdout_after = 24` falls inside
/// the week labelled 21, which runs to day 28: its label is before the cutoff
/// and its end after it. The fit holds that row out (a row trains if what it
/// measures ends by the cutoff), so it is `held_out` in both sections. The
/// expected roles are written out per row, by position in the file (labels 0,
/// 7, 14 trained; 21 to 56 held out), not derived from any time.
#[test]
fn a_row_whose_label_precedes_the_cutoff_and_whose_period_ends_after_it_is_held_out() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_rows(dir, "daily_cases.tsv", "daily_cases.tsv", &|_| true);
    // The same nine weekly counts, labelled by the week's start: 0, 7, …, 56.
    let weekly = std::fs::read_to_string(fixture("weekly_cases.tsv")).unwrap();
    let mut lines = weekly.lines();
    let mut relabelled = lines.next().unwrap().to_string() + "\n";
    for l in lines {
        let (t, v) = l.split_once('\t').unwrap();
        relabelled += &format!("{}\t{v}\n", t.parse::<u32>().unwrap() - 7);
    }
    std::fs::write(dir.join("weekly_cases.tsv"), relabelled).unwrap();
    let closing = "covers        = closing_at(time, 7 'days)";
    let model = std::fs::read_to_string(fixture("two_cadence_sir.camdl")).unwrap();
    assert_eq!(model.matches(closing).count(), 1);
    fit_model(dir, &model.replace(closing, "covers        = starting_on(time, 7 'days)"),
        "holdout_after = 24.0\n", "");
    let (_, out) = predict(dir, &[]);

    let m = read_json(&out.join("predictive.json"));
    let weekly_prefix = m["training_rows"]["streams"].as_array().unwrap().iter()
        .find(|s| s["stream"] == "weekly_cases").unwrap()["rows"].as_u64().unwrap();
    assert_eq!(weekly_prefix, 3, "the fit trained on the weeks labelled 0, 7, 14");

    let (t, h) = ("trained", "held_out");
    // Free-forward rows carry the label. The data rows are the first nine.
    let ff = roles(&out, "weekly_cases", "free_forward");
    let expect_ff = [(0.0, t), (7.0, t), (14.0, t), (21.0, h), (28.0, h), (35.0, h), (42.0, h), (49.0, h), (56.0, h)]
        .map(|(time, role)| (time, role.to_string()));
    assert_eq!(&ff[..9], &expect_ff[..], "the week labelled 21 ends on day 28, after the cutoff");
    // One-step rows carry the period's end (gh#1152), so the training rows
    // are timed 7, 14, 21; the week labelled 21, timed 28, is held out, and
    // the band stops before it (its value is not stored).
    let expect_os = [(7.0, t), (14.0, t), (21.0, t)].map(|(time, role)| (time, role.to_string()));
    assert_eq!(roles(&out, "weekly_cases", "one_step"), expect_os);
}

/// A stored run whose training partition has been removed from its
/// `fit.meta.json` is refused in both modes: the file no longer matches the
/// checksum its run recorded, so the roles are not guessed from it, and
/// nothing is written.
#[test]
fn a_run_whose_recorded_partition_was_removed_is_refused_in_both_modes() {
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

    for mode in ["free_forward", "one_step"] {
        let out = run(dir, &["predict", "@fit", "--mode", mode, "--n-draws", "3"]);
        assert_eq!(out.status.code(), Some(1), "--mode {mode}: {}", said(&out));
        assert!(said(&out).contains("@fit"), "--mode {mode}: the refusal names the run: {}", said(&out));
        assert!(!walk(&dir.join("results")).iter().any(|p| p.file_name().is_some_and(|n| n == "predictive.json")),
            "--mode {mode}: nothing is written");
    }
}
