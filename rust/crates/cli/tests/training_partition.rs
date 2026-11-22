//! Which rows a fit trains on, and what it records (D38, D40).
//!
//! A row trains if the end of what it measures is at or before the
//! `holdout_after` cutoff; under `[data.holdout]` every row of the training
//! files trains. The fit run records the decision in its `fit.meta.json` as
//! `training_partition`: per stream, how many leading rows trained and the
//! row digest of exactly those rows, and the training extent, the latest end
//! among them. `last_obs` in a model anchor is the latest end of an observed
//! training row, so rows appended after the cutoff move no anchor.
//!
//! Spec: docs/dev/temporal-architecture.md §2.3, §3.2, §4.1, §4.7; proposal
//! docs/dev/proposals/2026-11-15-temporal-architecture-migration.md Stage 2,
//! fixtures F2 and F3.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`", bin.display());
    bin
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        // The store a reader of a stored fit (`compare`) opens; a fit file's
        // `store` key names its own.
        .env("CAMDL_STORE", dir.join("results"))
        .output()
        .expect("spawn camdl")
}

fn ok(out: Output, what: &str) -> Output {
    assert!(out.status.success(), "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

/// One observed stream of the closed SIR below: its name (also its value
/// column), its `covers` form and the flow it counts.
struct Stream {
    name: &'static str,
    covers: &'static str,
    flow: &'static str,
}

/// A closed SIR observed by `streams`, run from day 0 to `to`.
fn model_src(streams: &[Stream], to: &str) -> String {
    let obs: String = streams.iter().map(|s| format!(
        "  {n} {{\n    columns   {{ time : time, {n} : count }}\n    covers    = {c}\n    \
         projected = incidence({f})\n    {n} ~ neg_binomial(mean = rho * projected + 0.1, r = 2.0)\n  }}\n",
        n = s.name, c = s.covers, f = s.flow)).collect();
    format!(r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  rho   : probability  in [0.05, 0.95]
}}

let N = S + I + R

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

init {{
  S = 990
  I = 10
}}

observations {{
{obs}}}

simulate {{
  from = 0 'days
  to   = {to}
}}
"#)
}

/// A file with one row per label, values distinct.
fn rows_tsv(column: &str, labels: impl IntoIterator<Item = u32>) -> String {
    let mut s = format!("time\t{column}\n");
    for (i, l) in labels.into_iter().enumerate() {
        s.push_str(&format!("{l}\t{}\n", 3 + i));
    }
    s
}

/// A fit config binding each stream to `<name>.tsv` under `data` (the
/// `[data]` table body), fitted by a short IF2 run into the run store
/// `store` (relative to the config).
fn fit_toml(streams: &[Stream], data: &str, rho: f64, store: &str) -> String {
    let mut observations = String::from("[data.observations]\n");
    for s in streams {
        observations.push_str(&format!("{0} = \"{0}.tsv\"\n", s.name));
    }
    format!(r#"store = "{store}"

[model]
camdl = "model.camdl"

[data]
{data}
{observations}
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}

[fixed]
gamma = 0.15
rho   = {rho}

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 100
iterations = 2
cooling    = 0.7
"#)
}

fn find_fit_meta(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                return p;
            }
        }
    }
    panic!("no fit.meta.json under {}", root.display());
}

/// Fit `toml` in `dir` into the run store `dir/store` and return its
/// `training_partition` record and the fit's stderr.
fn fit_partition(dir: &Path, toml: &str, store: &str) -> (serde_json::Value, String) {
    let out = ok(camdl(dir, &["fit", "run", toml, "--seed", "1"]), "fit run");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(find_fit_meta(&dir.join(store))).unwrap()).unwrap();
    (meta["training_partition"].clone(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// `stream → rows` of a recorded partition.
fn rows_of(partition: &serde_json::Value) -> Vec<(String, u64)> {
    partition["streams"].as_array()
        .unwrap_or_else(|| panic!("no training_partition.streams: {partition}"))
        .iter()
        .map(|s| (s["stream"].as_str().unwrap().to_string(), s["rows"].as_u64().unwrap()))
        .collect()
}

fn extent(partition: &serde_json::Value) -> f64 {
    partition["extent"]["t"].as_f64()
        .unwrap_or_else(|| panic!("no training_partition.extent.t: {partition}"))
}

/// Fixture F3: a `starting_on(time, 7 'days)` stream labelled 0, 7, …, 63
/// under `holdout_after = 24`. The row labelled 21 measures days 21–27 and
/// ends at 28, after the cutoff, so it does not train; the training rows are
/// those labelled 0, 7 and 14, and the extent is 21, the end of row 14's
/// period.
#[test]
fn f3_the_row_straddling_the_cutoff_does_not_train() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let streams = [Stream { name: "cases", covers: "starting_on(time, 7 'days)", flow: "infection" }];
    std::fs::write(dir.join("model.camdl"), model_src(&streams, "70 'days")).unwrap();
    std::fs::write(dir.join("cases.tsv"), rows_tsv("cases", (0..=63).step_by(7))).unwrap();
    std::fs::write(dir.join("fit.toml"),
        fit_toml(&streams, "holdout_after = 24.0", 0.5, "results")).unwrap();
    let (p, stderr) = fit_partition(dir, "fit.toml", "results");
    assert_eq!(rows_of(&p), vec![("cases".to_string(), 3)], "rows 0, 7 and 14 train: {p}");
    assert_eq!(extent(&p), 21.0, "the extent is the end of row 14's period: {p}");
    assert_eq!(p["rule"]["kind"], "cutoff", "{p}");
    assert_eq!(p["rule"]["at"]["t"], 24.0, "{p}");
    assert!(stderr.contains("first held-out row") && stderr.contains("21"),
        "the fit names the first held-out row of each stream:\n{stderr}");
}

/// A `day(time)` stream labelled 1–16 under `holdout_after = 10`: the row
/// labelled D covers day D and ends at D + 1, so rows 1–9 train and the
/// extent is 10.
#[test]
fn a_day_stream_trains_the_rows_ending_by_the_cutoff() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let streams = [Stream { name: "cases", covers: "day(time)", flow: "infection" }];
    std::fs::write(dir.join("model.camdl"), model_src(&streams, "20 'days")).unwrap();
    std::fs::write(dir.join("cases.tsv"), rows_tsv("cases", 1..=16)).unwrap();
    std::fs::write(dir.join("fit.toml"),
        fit_toml(&streams, "holdout_after = 10.0", 0.5, "results")).unwrap();
    let (p, _) = fit_partition(dir, "fit.toml", "results");
    assert_eq!(rows_of(&p), vec![("cases".to_string(), 9)], "{p}");
    assert_eq!(extent(&p), 10.0, "{p}");
}

/// Fixture F2: a weekly `closing_at` stream labelled 7–70 and a daily
/// `day(time)` stream labelled 0–69 under `holdout_after = 24`. Daily rows
/// 0–23 and weekly rows 7, 14 and 21 train; the extent is 24. The weekly row
/// labelled 28 (days 21–27) did not train, although its period overlaps daily
/// training days, so it is among the rows `compare` counts as held out.
#[test]
fn f2_two_streams_at_different_rates() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let streams = [
        Stream { name: "weekly", covers: "closing_at(time, 7 'days)", flow: "infection" },
        Stream { name: "daily", covers: "day(time)", flow: "recovery" },
    ];
    std::fs::write(dir.join("model.camdl"), model_src(&streams, "70 'days")).unwrap();
    std::fs::write(dir.join("weekly.tsv"), rows_tsv("weekly", (7..=70).step_by(7))).unwrap();
    std::fs::write(dir.join("daily.tsv"), rows_tsv("daily", 0..=69)).unwrap();
    std::fs::write(dir.join("a.toml"),
        fit_toml(&streams, "holdout_after = 24.0", 0.5, "results")).unwrap();
    std::fs::write(dir.join("b.toml"),
        fit_toml(&streams, "holdout_after = 24.0", 0.6, "results")).unwrap();
    // Both fits in one run store, so `compare` names each by its label.
    ok(camdl(dir, &["fit", "run", "a.toml", "--seed", "1", "--label", "a"]), "fit run a");
    let p = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(find_fit_meta(&dir.join("results"))).unwrap()).unwrap()["training_partition"].clone();
    assert_eq!(rows_of(&p),
        vec![("daily".to_string(), 24), ("weekly".to_string(), 3)], "{p}");
    assert_eq!(extent(&p), 24.0, "{p}");
    ok(camdl(dir, &["fit", "run", "b.toml", "--seed", "1", "--label", "b"]), "fit run b");

    // `compare` refuses to score a stored fit's held-out rows, and counts
    // them per stream: the weekly rows labelled 28–70 (7 rows, the row
    // labelled 28 among them) and the daily rows 24–69 (46 rows).
    let out = camdl(dir, &["compare", "@a", "@b", "--baseline", "@b",
        "--format", "json", "--particles", "50", "--draws", "1", "--replicates", "1",
        "--seed", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "held-out rows are not scored:\n{stderr}");
    assert!(stderr.contains("held rows out of its training data (daily: 46, weekly: 7)")
        && stderr.contains("step 7") && stderr.contains("--in-sample"),
        "the refusal counts each stream's held-out rows:\n{stderr}");
}

/// `[data.holdout]` and `holdout_after` describing the same split record the
/// same training rows and extent: a `day(time)` stream labelled 1–16 cut at
/// 10, against a training file of rows 1–9 and a holdout file of rows 10–16.
#[test]
fn holdout_files_and_holdout_after_record_the_same_partition() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let streams = [Stream { name: "cases", covers: "day(time)", flow: "infection" }];
    std::fs::write(dir.join("model.camdl"), model_src(&streams, "20 'days")).unwrap();
    let full = rows_tsv("cases", 1..=16);
    std::fs::write(dir.join("cases.tsv"), &full).unwrap();
    let mut lines = full.lines();
    let header = lines.next().unwrap();
    let body: Vec<&str> = lines.collect();
    let sub = dir.join("files");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("model.camdl"), model_src(&streams, "20 'days")).unwrap();
    std::fs::write(sub.join("cases.tsv"), format!("{header}\n{}\n", body[..9].join("\n"))).unwrap();
    std::fs::write(sub.join("cases_holdout.tsv"),
        format!("{header}\n{}\n", body[9..].join("\n"))).unwrap();

    std::fs::write(dir.join("fit.toml"),
        fit_toml(&streams, "holdout_after = 10.0", 0.5, "results")).unwrap();
    std::fs::write(sub.join("fit.toml"),
        format!("{}\n[data.holdout]\ncases = \"cases_holdout.tsv\"\n",
            fit_toml(&streams, "", 0.5, "results"))).unwrap();
    let (cut, _) = fit_partition(dir, "fit.toml", "results");
    let (files, _) = fit_partition(&sub, "fit.toml", "results");
    assert_eq!(cut["streams"], files["streams"],
        "the same training rows, with the same digest:\n{cut}\n{files}");
    assert_eq!(extent(&cut), extent(&files), "{cut}\n{files}");
    assert_eq!(extent(&cut), 10.0, "{cut}");
    assert_eq!(files["rule"]["kind"], "separate_files", "{files}");
}

/// The model's `simulate { to = last_obs + 7 'days }` resolves over the
/// training rows: rows appended after the `holdout_after` cutoff leave the
/// fit's end time where it was. With rows 1–9 of a `day(time)` stream
/// training (the latest ends at day 10), the end is 17 whether the file runs
/// to day 16 or to day 20.
#[test]
fn appending_held_out_rows_leaves_a_last_obs_end_time_unchanged() {
    let resolved_end = |last: u32| -> String {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let streams = [Stream { name: "cases", covers: "day(time)", flow: "infection" }];
        std::fs::write(dir.join("model.camdl"), model_src(&streams, "last_obs + 7 'days")).unwrap();
        std::fs::write(dir.join("cases.tsv"), rows_tsv("cases", 1..=last)).unwrap();
        std::fs::write(dir.join("fit.toml"),
            fit_toml(&streams, "holdout_after = 10.0", 0.5, "results")).unwrap();
        let (_, stderr) = fit_partition(dir, "fit.toml", "results");
        stderr.lines()
            .find(|l| l.contains("simulate { to }") && l.contains("→ t ="))
            .unwrap_or_else(|| panic!("no resolved end time on stderr:\n{stderr}"))
            .trim().to_string()
    };
    let short = resolved_end(16);
    let long = resolved_end(20);
    assert!(short.ends_with("→ t = 17"), "last_obs is the latest training end, 10: {short}");
    assert_eq!(short, long, "appending held-out rows moved the end time");
}

/// A stream that stopped reporting well before the others gets one note,
/// naming it, per fit, and each `holdout:` startup line is printed once,
/// although the fit decides its partition twice (once to record it, once in
/// the method's runner). `deaths` ends at day 5; `cases` trains to day 12.
#[test]
fn a_stopped_stream_is_noted_once_per_fit() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let streams = [
        Stream { name: "cases", covers: "day(time)", flow: "infection" },
        Stream { name: "deaths", covers: "day(time)", flow: "recovery" },
    ];
    std::fs::write(dir.join("model.camdl"), model_src(&streams, "20 'days")).unwrap();
    std::fs::write(dir.join("cases.tsv"), rows_tsv("cases", 0..=15)).unwrap();
    std::fs::write(dir.join("deaths.tsv"), rows_tsv("deaths", 0..=4)).unwrap();
    std::fs::write(dir.join("fit.toml"),
        fit_toml(&streams, "holdout_after = 12.0", 0.5, "results")).unwrap();
    let (_, stderr) = fit_partition(dir, "fit.toml", "results");
    let count = |needle: &str| stderr.lines().filter(|l| l.contains(needle)).count();
    assert_eq!(count("note: stream 'deaths' has its last observation at t = 5"), 1,
        "one note for the stopped stream:\n{stderr}");
    assert_eq!(count("note: stream 'cases'"), 0, "cases did not stop:\n{stderr}");
    assert_eq!(count("holdout: trained on"), 1, "the holdout line once:\n{stderr}");
    assert_eq!(count("holdout: first held-out row of 'cases'"), 1,
        "the first held-out row once:\n{stderr}");
}
