//! gh#1138: `camdl data split --at-time T` put a row in the training file when
//! its time label was at or before `T`. A fit's `holdout_after = T` trains a
//! row when the end of what it measures is at or before `T` (D38). Under
//! `covers = day(time)` the row labelled `D` covers `[D, D + 1)`, so the two
//! rules disagreed about the row labelled `T`: the split trained it, the fit
//! held it out.
//!
//! Given the model and the stream (`--model`, `--stream`), the split reads the
//! file through the fit's row reader and judges each row by the fit's rule, so
//! its training file holds exactly the rows the fit trains on. Without a model
//! it still splits on the label, and says so.
//!
//! End-to-end via the release `camdl` binary.

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
        .output()
        .expect("spawn camdl")
}

fn ok(out: Output, what: &str) -> Output {
    assert!(out.status.success(), "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

/// A closed SIR with a daily `day(time)` incidence stream and a weekly
/// `closing_at` one, so `--stream` has a choice to make.
const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  rho   : probability  in [0.05, 0.95]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = 990
  I = 10
}

observations {
  daily_cases {
    columns   { time : time, daily_cases : count }
    covers    = day(time)
    projected = incidence(infection)
    daily_cases ~ neg_binomial(mean = rho * projected + 0.1, r = 2.0)
  }
  weekly_cases {
    columns   { time : time, weekly_cases : count }
    covers    = closing_at(time, 7 'days)
    projected = incidence(recovery)
    weekly_cases ~ neg_binomial(mean = rho * projected + 0.1, r = 2.0)
  }
}

simulate {
  from = 0 'days
  to   = 45 'days
}
"#;

/// A fit of both streams under `holdout_after = 23`, a short IF2 run.
const FIT: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data]
holdout_after = 23.0

[data.observations]
daily_cases  = "daily.tsv"
weekly_cases = "weekly.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }

[fixed]
gamma = 0.15
rho   = 0.5

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.7
"#;

fn rows_tsv(column: &str, labels: impl IntoIterator<Item = u32>) -> String {
    let mut s = format!("time\t{column}\n");
    for (i, l) in labels.into_iter().enumerate() {
        s.push_str(&format!("{l}\t{}\n", 3 + i));
    }
    s
}

fn fixture(dir: &Path) {
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("daily.tsv"), rows_tsv("daily_cases", 1..=40)).unwrap();
    std::fs::write(dir.join("weekly.tsv"), rows_tsv("weekly_cases", (7..=42).step_by(7))).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT).unwrap();
}

/// The first column of each data row of a written split file.
fn labels(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path).unwrap().lines().skip(1)
        .map(|l| l.split('\t').next().unwrap().parse().unwrap())
        .collect()
}

fn find_fit_meta(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
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

/// The defect: split at 23, the `day(time)` row labelled 23 covers [23, 24),
/// ends after the cut, and belongs in the held-out file, where the fit's
/// `holdout_after = 23` puts it. The training file holds as many rows as the
/// fit records as trained for the stream.
#[test]
fn a_day_stream_split_holds_out_the_row_ending_after_the_cut_as_the_fit_does() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    let out = ok(camdl(dir, &["data", "split", "daily.tsv", "--at-time", "23",
        "--model", "model.camdl", "--stream", "daily_cases",
        "--train", "d-tr.tsv", "--holdout", "d-ho.tsv"]), "data split");
    let stderr = String::from_utf8_lossy(&out.stderr);

    let train = labels(&dir.join("d-tr.tsv"));
    let held = labels(&dir.join("d-ho.tsv"));
    assert_eq!(train, (1..=22).collect::<Vec<_>>(),
        "rows ending by 23 train (the row labelled 22 ends at 23):\n{stderr}");
    assert_eq!(held, (23..=40).collect::<Vec<_>>(),
        "the row labelled 23 covers [23, 24) and is held out:\n{stderr}");

    ok(camdl(dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(find_fit_meta(&dir.join("results"))).unwrap()).unwrap();
    let fitted = meta["training_partition"]["streams"].as_array()
        .unwrap_or_else(|| panic!("no training_partition.streams: {meta}"))
        .iter()
        .find(|s| s["stream"] == "daily_cases")
        .and_then(|s| s["rows"].as_u64())
        .unwrap_or_else(|| panic!("no daily_cases prefix: {}", meta["training_partition"]));
    assert_eq!(train.len() as u64, fitted,
        "the split trains the rows the fit trains: {}", meta["training_partition"]);
}

/// A `closing_at` stream is labelled by the end of each row's period, so the
/// label and the period end agree and the row labelled 21 trains under a cut
/// at 23, the row labelled 28 does not.
#[test]
fn a_closing_at_stream_split_by_the_model_trains_rows_labelled_by_the_cut() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    ok(camdl(dir, &["data", "split", "weekly.tsv", "--at-time", "23",
        "--model", "model.camdl", "--stream", "weekly_cases",
        "--train", "w-tr.tsv", "--holdout", "w-ho.tsv"]), "data split");
    assert_eq!(labels(&dir.join("w-tr.tsv")), vec![7, 14, 21]);
    assert_eq!(labels(&dir.join("w-ho.tsv")), vec![28, 35, 42]);
}

/// A model with several streams needs `--stream`; the refusal names them.
#[test]
fn a_model_with_several_streams_needs_stream() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    let out = camdl(dir, &["data", "split", "daily.tsv", "--at-time", "23",
        "--model", "model.camdl"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "refused:\n{stderr}");
    assert!(stderr.contains("--stream") && stderr.contains("daily_cases")
        && stderr.contains("weekly_cases"), "names the streams:\n{stderr}");
}

/// Without a model the split still reads the label, and says so.
#[test]
fn without_a_model_the_split_says_it_reads_the_label() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    let out = ok(camdl(dir, &["data", "split", "daily.tsv", "--at-time", "23",
        "--train", "d-tr.tsv", "--holdout", "d-ho.tsv"]), "data split");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(labels(&dir.join("d-tr.tsv")).last(), Some(&23));
    assert!(stderr.contains("time label") && stderr.contains("--model"),
        "the summary says the split read the label:\n{stderr}");
}
