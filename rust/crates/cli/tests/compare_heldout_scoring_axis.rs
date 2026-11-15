//! gh#1062: a held-out `compare` scores no row the fit trained on, whatever
//! the stream's `covers` form.
//!
//! The fit cuts training by a row's LABEL (`holdout_after = τ` keeps the rows
//! labelled `≤ τ`; `[data.holdout]` keeps the training file). A stream that
//! accumulates over a period is SCORED at the period's close, which for a
//! period-start label (`day(time)`, `starting_on`, `ending_on`) is after the
//! label: the row labelled 10 under `day(time)` covers [10, 11) and is scored
//! at t = 11. A scoring cut at the label boundary (`t > 10`) therefore scored
//! that training row as held out.
//!
//! Each test fits a one- or two-stream closed SIR to daily rows labelled 1 to
//! 16 with distinct observed values, holds out the rows labelled 11 to 16, and
//! reads which rows `compare` scored: the scored times come from
//! `compare --pointwise`, and the observed value at each scored time from a
//! full-series `camdl pfilter --save-prequential` of the same data (the
//! observed value at a step is the data, independent of θ).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// Daily observed values for rows labelled 1..=16: distinct, so a scored
/// value names the row it came from.
const CASES: [u64; 16] = [3, 4, 5, 7, 9, 12, 16, 21, 27, 35, 44, 55, 67, 80, 93, 105];
const DEATHS: [u64; 16] = [1, 2, 6, 8, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24];

/// The last training label: `holdout_after = 10`, and the `[data.holdout]`
/// training file's last row.
const TAU: u64 = 10;

/// One observed stream: its name (also its column and data file), its
/// `covers` declaration, the flow it counts, and its values for labels 1..=16.
struct Stream {
    name: &'static str,
    covers: &'static str,
    flow: &'static str,
    values: &'static [u64; 16],
}

#[derive(Clone, Copy, PartialEq)]
enum Form {
    /// `[data] holdout_after = 10` over the full file.
    HoldoutAfter,
    /// `[data.holdout]`: a training file (labels 1..=10) and a holdout file
    /// (labels 11..=16).
    HoldoutFiles,
}

/// The `covers` value that stands for per-row window columns
/// (`window_start`/`window_stop`, no `covers =`): row D covers [D − 1, D),
/// labelled by its stop.
const WINDOW_COLUMNS: &str = "window columns";

fn model_src(streams: &[Stream]) -> String {
    let obs: String = streams.iter().map(|s| {
        let temporal = match s.covers {
            WINDOW_COLUMNS => format!(
                "columns       {{ window_start : window_start, window_stop : window_stop, \
                 {} : count }}", s.name),
            c => format!("columns       {{ time : time, {} : count }}\n    covers        = {c}",
                s.name),
        };
        format!(
            "  {n} {{\n    {temporal}\n    projected     = incidence({f})\n    \
             {n}{pad} ~ neg_binomial(mean = rho * projected, r = k)\n  }}\n",
            n = s.name, f = s.flow,
            pad = " ".repeat(13usize.saturating_sub(s.name.len())))
    }).collect();
    format!(r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}}

let N = S + I + R

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

init {{
  S = N0 - I0
  I = I0
}}

observations {{
{obs}}}

simulate {{
  from = 0 'days
  to   = 20 'days
}}
"#)
}

fn rows_tsv(stream: &Stream, labels: std::ops::RangeInclusive<u64>) -> String {
    let windowed = stream.covers == WINDOW_COLUMNS;
    let mut s = match windowed {
        true => format!("window_start\twindow_stop\t{}\n", stream.name),
        false => format!("time\t{}\n", stream.name),
    };
    for l in labels {
        let v = stream.values[(l - 1) as usize];
        match windowed {
            true => s.push_str(&format!("{}\t{l}\t{v}\n", l - 1)),
            false => s.push_str(&format!("{l}\t{v}\n")),
        }
    }
    s
}

fn fit_toml(streams: &[Stream], form: Form, rho: f64, output_dir: &str) -> String {
    let mut data = String::new();
    if form == Form::HoldoutAfter {
        data.push_str(&format!("[data]\nholdout_after = {TAU}.0\n\n"));
    }
    data.push_str("[data.observations]\n");
    for s in streams {
        match form {
            Form::HoldoutAfter => data.push_str(&format!("{0} = \"{0}.tsv\"\n", s.name)),
            Form::HoldoutFiles => data.push_str(&format!("{0} = \"{0}_train.tsv\"\n", s.name)),
        }
    }
    if form == Form::HoldoutFiles {
        data.push_str("\n[data.holdout]\n");
        for s in streams {
            data.push_str(&format!("{0} = \"{0}_holdout.tsv\"\n", s.name));
        }
    }
    format!(r#"output_dir = "{output_dir}"

[model]
camdl = "model.camdl"

{data}
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = {rho}
k   = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
"#)
}

fn camdl(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    let out = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl");
    assert!(out.status.success(), "camdl {args:?} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

/// What `compare` scored for the first fit: its `t_score`, and per stream the
/// observed values it scored, in time order.
struct Scored {
    t_score: u64,
    values: BTreeMap<String, Vec<u64>>,
}

/// Fit two configs (differing only in a fixed `rho`) under `form`, compare
/// them held-out, and read back what the first one scored.
fn heldout_scored(slug: &str, streams: &[Stream], form: Form) -> Scored {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::Builder::new().prefix(&format!("camdl_gh1062_{slug}_")).tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), model_src(streams)).unwrap();
    for s in streams {
        std::fs::write(dir.join(format!("{}.tsv", s.name)), rows_tsv(s, 1..=16)).unwrap();
        std::fs::write(dir.join(format!("{}_train.tsv", s.name)),
            rows_tsv(s, 1..=TAU)).unwrap();
        std::fs::write(dir.join(format!("{}_holdout.tsv", s.name)),
            rows_tsv(s, TAU + 1..=16)).unwrap();
    }
    std::fs::write(dir.join("a.toml"), fit_toml(streams, form, 0.5, "results_a")).unwrap();
    std::fs::write(dir.join("b.toml"), fit_toml(streams, form, 0.6, "results_b")).unwrap();
    for cfg in ["a.toml", "b.toml"] {
        camdl(&bin, dir, &["fit", "run", cfg, "--seed", "1"]);
    }

    let out = camdl(&bin, dir, &["compare", "a.toml", "b.toml", "--baseline", "b.toml",
        "--format", "json", "--particles", "100", "--draws", "1", "--replicates", "1",
        "--seed", "1", "--pointwise", "pw.tsv"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("compare json");
    let row = v["rows"].as_array().unwrap().iter()
        .find(|r| r["name"] == "a.toml")
        .unwrap_or_else(|| panic!("no a.toml row:\n{v}"))
        .clone();
    assert_eq!(row["conditioning"]["hold_out_tail"]["train_end"], serde_json::json!(TAU as f64),
        "the row is stamped held-out at the fit's train_end: {row}");
    let t_score = row["t_score"].as_u64().expect("t_score");

    // Observed value at each (stream, scoring time), from a full-series
    // filter pass over the same data.
    std::fs::write(dir.join("theta.toml"),
        "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 10\nrho = 0.5\nk = 10.0\n").unwrap();
    let mut args: Vec<String> = ["pfilter", "model.camdl", "--params", "theta.toml",
        "--particles", "50", "--seed", "1", "--save-prequential", "full", "--no-store"]
        .iter().map(|s| s.to_string()).collect();
    for s in streams {
        args.push("--data".into());
        args.push(format!("{0}={0}.tsv", s.name));
    }
    camdl(&bin, dir, &args.iter().map(String::as_str).collect::<Vec<_>>());
    let full: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("full.json")).unwrap()).unwrap();
    let mut y_at: BTreeMap<(String, String), u64> = BTreeMap::new();
    for step in full["steps"].as_array().unwrap() {
        for ss in step["per_stream"].as_array().unwrap() {
            y_at.insert((ss["stream"].as_str().unwrap().to_string(), step["t"].to_string()),
                ss["y_obs"].as_f64().unwrap() as u64);
        }
    }

    // The scored (time, stream) pairs of fit a, from --pointwise.
    let pw = std::fs::read_to_string(dir.join("pw.streams.tsv")).unwrap();
    let mut lines = pw.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (c_model, c_t, c_stream, c_score) = (col("model"), col("t"), col("stream"), col("log_score"));
    let mut values: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        if f[c_model] != "a.toml" || f[c_score].is_empty() {
            continue;
        }
        let t: f64 = f[c_t].parse().unwrap();
        let key = (f[c_stream].to_string(), serde_json::json!(t).to_string());
        let y = *y_at.get(&key)
            .unwrap_or_else(|| panic!("no observed value for {key:?} in the full trace"));
        values.entry(f[c_stream].to_string()).or_default().push(y);
    }
    Scored { t_score, values }
}

/// The one-stream assertions: exactly the withheld rows are scored, starting
/// at the first one, and the last training row is not among them.
fn assert_scores_only_the_withheld_rows(covers: &'static str, form: Form, slug: &str) {
    let streams = [Stream { name: "cases", covers, flow: "infection", values: &CASES }];
    let got = heldout_scored(slug, &streams, form);
    let scored = &got.values["cases"];
    let withheld: Vec<u64> = CASES[TAU as usize..].to_vec();
    assert_eq!(got.t_score, withheld.len() as u64,
        "covers = {covers}: one scored step per withheld row; scored values {scored:?}");
    assert_eq!(scored.first(), withheld.first(),
        "covers = {covers}: the first scored value is the first withheld row's; \
         scored values {scored:?}");
    let last_training = CASES[TAU as usize - 1];
    assert!(!scored.contains(&last_training),
        "covers = {covers}: the last training row (value {last_training}) is scored \
         as held out; scored values {scored:?}");
    assert_eq!(scored, &withheld, "covers = {covers}");
}

#[test]
fn day_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("day(time)", Form::HoldoutAfter, "day");
}

#[test]
fn starting_on_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("starting_on(time, 1 'days)", Form::HoldoutAfter, "starting");
}

#[test]
fn ending_on_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("ending_on(time, 1 'days)", Form::HoldoutAfter, "ending");
}

#[test]
fn day_holdout_files_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("day(time)", Form::HoldoutFiles, "dayfiles");
}

/// Control: a `closing_at` row is scored at its label, so the label cut and
/// the scoring cut coincide; its scored rows are the withheld ones, before
/// and after gh#1062.
#[test]
fn closing_at_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("closing_at(time, 1 'days)", Form::HoldoutAfter, "closing");
}

/// Control: a per-row window stream is labelled by its stop, which is its
/// scoring time, so its scored rows are the withheld ones.
#[test]
fn window_columns_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows(WINDOW_COLUMNS, Form::HoldoutAfter, "window");
}

/// Two streams under one `holdout_after = 10`: `cases` by `day(time)` (row
/// labelled D scored at D + 1) and `deaths` by `closing_at` (row labelled D
/// scored at D). The training rows close at t = 11 (`cases`) and t = 10
/// (`deaths`), so scoring starts after t = 11. No training row of either
/// stream is scored. `cases` scores all six withheld rows (labels 11–16);
/// `deaths` scores labels 12–16: its first withheld row, labelled 11, closes
/// at t = 11 and is assimilated without being scored — the conservative
/// side of a single scoring boundary.
#[test]
fn two_streams_score_no_training_row_of_either() {
    let streams = [
        Stream { name: "cases", covers: "day(time)", flow: "infection", values: &CASES },
        Stream { name: "deaths", covers: "closing_at(time, 1 'days)", flow: "recovery",
                 values: &DEATHS },
    ];
    let got = heldout_scored("two", &streams, Form::HoldoutAfter);
    for (name, values) in [("cases", &CASES), ("deaths", &DEATHS)] {
        let scored = &got.values[name];
        for training in &values[..TAU as usize] {
            assert!(!scored.contains(training),
                "{name}: training value {training} scored as held out; scored {scored:?}");
        }
    }
    assert_eq!(got.values["cases"], CASES[TAU as usize..].to_vec(),
        "cases scores its six withheld rows");
    assert_eq!(got.values["deaths"], DEATHS[TAU as usize + 1..].to_vec(),
        "deaths scores its withheld rows after the first, which closes at the boundary");
}
