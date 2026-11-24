//! gh#1062, D38: a fit holding rows out trains on exactly the rows that end
//! at or before the cutoff, whatever the stream's `covers` form, and
//! `compare` scores a stored fit on exactly those rows.
//!
//! A row trains if the END of what it measures is at or before the cutoff
//! (`holdout_after = τ`); under `[data.holdout]` every row of the training
//! file trains. A stream that accumulates over a period is scored at the
//! period's close, which for a period-start label (`day(time)`,
//! `starting_on`, `ending_on`) is after the label: the row labelled 10 under
//! `day(time)` covers [10, 11), ends at 11 and so does not train under
//! `holdout_after = 10`.
//!
//! Each test fits a one- or two-stream closed SIR to daily rows labelled 1 to
//! 16 with distinct observed values, cut at 10 (the rows labelled 11 to 16
//! held out where the label is the period's end, 10 to 16 where the period
//! opens at the label). It reads the partition the stored problem holds: the
//! training rows in `inputs/<stream>.tsv`, the held-out rows' periods in
//! `held_out/<stream>.tsv` and the training extent in `partition.json`. A
//! stored fit keeps its held-out rows' periods and not their values, so
//! `compare` scores them from the outcome files: the full file under
//! `holdout_after` (`--data`), the holdout file under `[data.holdout]`
//! (`--data-held-out`). The test reads which rows `compare` scored held out,
//! and which `compare --in-sample` scored: the scored times come from
//! `compare --pointwise`, and the observed value at each scored time from a
//! full-series `camdl pfilter --save-prequential` of the same data (the
//! observed value at a step is the data, independent of θ). Most fits run
//! IF2 on the chain-binomial backend, predicted one-step at their winner; the
//! `ode_` tests run MH on the `ode` backend, predicted free-forward.

mod fit_store_support;

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

/// How a fit is run: on the chain-binomial backend, which `compare` predicts
/// one-step through a particle filter, or on the ODE backend, predicted
/// free-forward.
#[derive(Clone, Copy, PartialEq)]
enum Backend {
    /// IF2 on the chain-binomial backend.
    Filter,
    /// MH on the `ode` backend.
    Ode,
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

fn fit_toml(streams: &[Stream], form: Form, backend: Backend, rho: f64)
    -> String {
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
    let method = match backend {
        Backend::Filter => "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
                            particles = 100\niterations = 5\ncooling = 0.7\n",
        Backend::Ode => "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 1\n\
                         iterations = 200\nburn_in = 50\nstarts = \"single\"\n",
    };
    format!(r#"store = "results"

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
{method}"#)
}

fn run_camdl(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        // `@label` handles resolve in the store the fit files name.
        .env("CAMDL_STORE", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn camdl(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    let out = run_camdl(bin, dir, args);
    assert!(out.status.success(), "camdl {args:?} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

/// What the first fit stored and what `compare` did with it: the training
/// rows and held-out rows its stored problem holds, the training extent its
/// `partition.json` records, and what `compare` scored held out and
/// `--in-sample`.
struct Scored {
    /// Per stream, the observed values of the rows in `inputs/<stream>.tsv`.
    training: BTreeMap<String, Vec<u64>>,
    /// Per stream, the labels of the rows in `held_out/<stream>.tsv`.
    held_out: BTreeMap<String, Vec<u64>>,
    /// `partition.json`'s `extent.t`.
    extent: f64,
    /// The held-out row's `t_score`, and its stamped training end.
    held_t_score: u64,
    held_train_end: f64,
    /// Per stream, the observed values `compare` scored held out.
    held_values: BTreeMap<String, Vec<u64>>,
    /// The `--in-sample` row's `t_score`.
    t_score: u64,
    /// Per stream, the observed values `compare --in-sample` scored, in time
    /// order.
    values: BTreeMap<String, Vec<u64>>,
}

/// Whether a row of this `covers` form measures a period that opens at its
/// label (and so ends one day after it), rather than one that ends there.
fn opens_at_label(covers: &str) -> bool {
    ["day(", "starting_on(", "ending_on("].iter().any(|f| covers.starts_with(f))
}

/// The index into the values of the first row that does not train: under
/// `holdout_after = 10` the row labelled 10 when its period opens at the
/// label (it ends at 11), the row labelled 11 otherwise; under
/// `[data.holdout]` the first row of the holdout file, labelled 11.
fn first_withheld(covers: &str, form: Form) -> usize {
    match (form, opens_at_label(covers)) {
        (Form::HoldoutAfter, true) => TAU as usize - 1,
        _ => TAU as usize,
    }
}

/// The training extent, the latest end among the training rows: 10 under
/// `holdout_after = 10` whatever the form, and under `[data.holdout]` the end
/// of the training file's last row, labelled 10.
fn extent(covers: &str, form: Form) -> f64 {
    match (form, opens_at_label(covers)) {
        (Form::HoldoutFiles, true) => TAU as f64 + 1.0,
        _ => TAU as f64,
    }
}

/// The values of column `col` of a stored TSV, parsed as whole numbers.
fn column(path: &Path, col: &str) -> Vec<u64> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let c = header.iter().position(|h| *h == col)
        .unwrap_or_else(|| panic!("{} has no column {col}: {header:?}", path.display()));
    lines.filter(|l| !l.is_empty())
        .map(|l| l.split('\t').nth(c).unwrap().parse::<f64>().unwrap() as u64)
        .collect()
}

/// Fit two configs (differing only in a fixed `rho`) under `form`, read the
/// partition the first one stored, ask `compare` for a held-out score, and
/// read back what `compare --in-sample` scored for the first one.
fn heldout_scored(slug: &str, streams: &[Stream], form: Form, backend: Backend) -> Scored {
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
    std::fs::write(dir.join("a.toml"), fit_toml(streams, form, backend, 0.5))
        .unwrap();
    std::fs::write(dir.join("b.toml"), fit_toml(streams, form, backend, 0.6))
        .unwrap();
    for (cfg, label) in [("a.toml", "a"), ("b.toml", "b")] {
        camdl(&bin, dir, &["fit", "run", cfg, "--seed", "1", "--label", label]);
    }

    // The problem `a.toml` stored: the run whose fit file fixes rho at 0.5.
    let run_a = fit_store_support::runs(&dir.join("results")).into_iter()
        .find(|r| std::fs::read_to_string(r.join("fit.toml")).unwrap().contains("rho = 0.5"))
        .expect("the run of a.toml");
    let problem = fit_store_support::problem_of(&run_a);
    let mut training = BTreeMap::new();
    let mut held_out = BTreeMap::new();
    for s in streams {
        training.insert(s.name.to_string(),
            column(&problem.join("inputs").join(format!("{}.tsv", s.name)), s.name));
        held_out.insert(s.name.to_string(),
            column(&problem.join("held_out").join(format!("{}.tsv", s.name)), "label"));
    }
    let extent = fit_store_support::json(&problem.join("partition.json"))["extent"]["t"]
        .as_f64().expect("partition.json extent.t");

    // An ODE fit is predicted free-forward, which runs no filter.
    let mut compare: Vec<String> = ["compare", "@a", "@b", "--baseline", "@b",
        "--format", "json", "--draws", "1", "--replicates", "1", "--seed", "1"]
        .iter().map(|s| s.to_string()).collect();
    if backend == Backend::Filter {
        compare.extend(["--particles".to_string(), "100".to_string()]);
    }
    let row_of = |extra: &[String]| -> serde_json::Value {
        let line: Vec<&str> = compare.iter().chain(extra).map(String::as_str).collect();
        let out = camdl(&bin, dir, &line);
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("compare json");
        v["rows"].as_array().unwrap().iter()
            .find(|r| r["name"] == "@a")
            .unwrap_or_else(|| panic!("no @a row:\n{v}"))
            .clone()
    };
    // The held-out rows' outcomes: the full files, or the holdout files alone.
    let mut held: Vec<String> = Vec::new();
    for st in streams {
        match form {
            Form::HoldoutAfter => held.extend(["--data".to_string(), format!("{0}={0}.tsv", st.name)]),
            Form::HoldoutFiles => held.extend(["--data-held-out".to_string(), format!("{0}={0}_holdout.tsv", st.name)]),
        }
    }
    held.extend(["--pointwise".to_string(), "pw_held.tsv".to_string()]);
    let row = row_of(&held);
    assert_eq!(row["role"], "held_out", "{row}");
    let held_t_score = row["t_score"].as_u64().expect("t_score");
    let held_train_end = row["conditioning"]["hold_out_tail"]["train_end"].as_f64()
        .unwrap_or_else(|| panic!("a held-out row is stamped hold_out_tail: {row}"));

    let row = row_of(&["--in-sample".to_string(), "--pointwise".to_string(), "pw.tsv".to_string()]);
    assert_eq!(row["conditioning"], "in_sample", "{row}");
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
    let scored_values = |file: &str| -> BTreeMap<String, Vec<u64>> {
        let pw = std::fs::read_to_string(dir.join(file)).unwrap();
        let mut lines = pw.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
        let (c_model, c_t, c_stream, c_score) = (col("model"), col("t"), col("stream"), col("log_score"));
        let mut values: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for line in lines {
            let f: Vec<&str> = line.split('\t').collect();
            if f[c_model] != "@a" || f[c_score].is_empty() {
                continue;
            }
            let t: f64 = f[c_t].parse().unwrap();
            let key = (f[c_stream].to_string(), serde_json::json!(t).to_string());
            let y = *y_at.get(&key)
                .unwrap_or_else(|| panic!("no observed value for {key:?} in the full trace"));
            values.entry(f[c_stream].to_string()).or_default().push(y);
        }
        values
    };
    let values = scored_values("pw.streams.tsv");
    let held_values = scored_values("pw_held.streams.tsv");
    Scored { training, held_out, extent, held_t_score, held_train_end, held_values, t_score, values }
}

/// The one-stream assertions: the stored problem trains on exactly the rows
/// before the first withheld one and holds out the rest; `compare` scores
/// exactly the held-out rows, stamped at the training extent; `compare
/// --in-sample` scores exactly the training rows.
fn assert_scores_only_the_withheld_rows(covers: &'static str, form: Form, slug: &str) {
    assert_backend_scores_only_the_withheld_rows(covers, form, Backend::Filter, slug);
}

/// [`assert_scores_only_the_withheld_rows`] for fits run on `backend`.
fn assert_backend_scores_only_the_withheld_rows(
    covers: &'static str, form: Form, backend: Backend, slug: &str,
) {
    let streams = [Stream { name: "cases", covers, flow: "infection", values: &CASES }];
    let got = heldout_scored(slug, &streams, form, backend);
    let first = first_withheld(covers, form);
    let trained: Vec<u64> = CASES[..first].to_vec();
    let withheld_labels: Vec<u64> = (first as u64 + 1..=16).collect();
    assert_eq!(got.extent, extent(covers, form),
        "covers = {covers}: partition.json records the training extent");
    assert_eq!(got.training["cases"], trained,
        "covers = {covers}: the stored training rows end at the last training row");
    assert_eq!(got.held_out["cases"], withheld_labels,
        "covers = {covers}: the stored held-out rows start at the first withheld row");
    assert_eq!(got.held_values["cases"], CASES[first..].to_vec(),
        "covers = {covers}: the held-out score scores every held-out row and no training row");
    assert_eq!(got.held_t_score, withheld_labels.len() as u64, "covers = {covers}");
    assert_eq!(got.held_train_end, got.extent, "covers = {covers}: stamped at the training extent");
    let scored = &got.values["cases"];
    assert_eq!(got.t_score, trained.len() as u64,
        "covers = {covers}: one in-sample step per training row; scored values {scored:?}");
    assert!(!scored.contains(&CASES[first]),
        "covers = {covers}: the first withheld row (value {}) is scored in sample; \
         scored values {scored:?}", CASES[first]);
    assert_eq!(scored, &trained, "covers = {covers}");
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
/// the scoring cut coincide.
#[test]
fn closing_at_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows("closing_at(time, 1 'days)", Form::HoldoutAfter, "closing");
}

/// Control: a per-row window stream is labelled by its stop, which is its
/// scoring time, so the label cut and the scoring cut coincide.
#[test]
fn window_columns_holdout_after_scores_only_withheld_rows() {
    assert_scores_only_the_withheld_rows(WINDOW_COLUMNS, Form::HoldoutAfter, "window");
}

/// Two streams under one `holdout_after = 10`: `cases` by `day(time)` (row
/// labelled D ends at D + 1) and `deaths` by `closing_at` (row labelled D
/// ends at D). `cases` trains rows 1–9 and `deaths` rows 1–10; both end by
/// t = 10, the training extent. The held-out score scores each stream's own
/// held-out rows (`cases` labels 10–16, `deaths` labels 11–16), and
/// `compare --in-sample` scores each stream's training rows and no held-out
/// row of either.
#[test]
fn two_streams_score_no_training_row_of_either() {
    let streams = [
        Stream { name: "cases", covers: "day(time)", flow: "infection", values: &CASES },
        Stream { name: "deaths", covers: "closing_at(time, 1 'days)", flow: "recovery",
                 values: &DEATHS },
    ];
    let got = heldout_scored("two", &streams, Form::HoldoutAfter, Backend::Filter);
    assert_eq!(got.extent, TAU as f64, "partition.json records the training extent");
    for (name, values, first) in [("cases", &CASES, TAU as usize - 1), ("deaths", &DEATHS, TAU as usize)] {
        assert_eq!(got.training[name], values[..first].to_vec(),
            "{name}: the stored training rows");
        assert_eq!(got.held_out[name], (first as u64 + 1..=16).collect::<Vec<_>>(),
            "{name}: the stored held-out rows");
        assert_eq!(got.held_values[name], values[first..].to_vec(), "{name}: the held-out rows scored");
        let scored = &got.values[name];
        for withheld in &values[first..] {
            assert!(!scored.contains(withheld),
                "{name}: held-out value {withheld} scored in sample; scored {scored:?}");
        }
        assert_eq!(scored, &values[..first].to_vec(), "{name} scores every training row");
    }
}

/// The ODE path cuts on the same boundary: two ODE+MH fits of a `day(time)`
/// stream hold out the rows labelled 10 to 16, which end after the cutoff,
/// and train on the row labelled 9, which ends at t = 10.
#[test]
fn ode_day_holdout_after_scores_only_withheld_rows() {
    assert_backend_scores_only_the_withheld_rows("day(time)", Form::HoldoutAfter, Backend::Ode,
        "ode_day");
}

/// Control for the ODE path: a `closing_at` row is scored at its label, so
/// the boundary is the last training label.
#[test]
fn ode_closing_at_holdout_after_scores_only_withheld_rows() {
    assert_backend_scores_only_the_withheld_rows("closing_at(time, 1 'days)",
        Form::HoldoutAfter, Backend::Ode, "ode_closing");
}
