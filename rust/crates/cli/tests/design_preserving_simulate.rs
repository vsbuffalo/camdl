//! gh#831: simulating on a real observation design, end to end.
//!
//! Two surfaces share one primitive (`obs_emit::simulate_dataset`):
//!
//! - `camdl simulate --design-from <fit.toml>` writes a dataset on the rows the
//!   fit's bound data occupies — its labels, each row's own window, its `NA`
//!   holes — so a self-consistency test can be run on the design the fit will
//!   actually see rather than on a regular grid.
//! - `[synthetic]` writes its replicates through the same emitter, one file per
//!   stream under the declared column names, which is what lets a windowed
//!   model run a parameter-recovery study at all: the single wide `ds_NN.tsv`
//!   this replaced could not carry two temporal boundaries.
//!
//! The load-bearing assertion in both is the round trip: the generated file
//! re-loads under the model that produced it, with the periods it was written
//! with.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        p.display());
    p
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir()
        .join(format!("camdl_design_e2e_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    TempDir(p)
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(camdl_bin())
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// Compile `src` to IR beside it, returning the IR path.
fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let camdlc = camdlc().expect("camdlc.exe");
    let model = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// An SIR whose incidence stream declares its windows per row, so no uniform
/// `covers` form could state the irregular data below.
const WINDOWED_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { win_start : window_start, win_stop : window_stop,
                    cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 20 'days }
"#;

/// Irregular observation windows — one-day rows, a three-day row, a two-day
/// row — and one `NA` hole. This is the design shape gh#831 exists for: no
/// `covers` form can state it, so simulating on the model's own schedule would
/// give a synthetic fit more information than the real fit has.
const IRREGULAR_DATA: &str = "win_start\twin_stop\tcases\n\
     0\t1\t2\n\
     1\t2\t4\n\
     2\t5\t19\n\
     5\t6\t7\n\
     6\t7\tNA\n\
     7\t9\t14\n\
     9\t10\t9\n\
     10\t11\t8\n";

const STAGES: &str = r#"
[estimate]
beta  = { bounds = [0.01, 5.0], start = 1.0 }
gamma = { bounds = [0.01, 1.0], start = 0.3 }

[fixed]
N0 = 1000

[stages.mle]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 50
iterations = 3
cooling    = 0.7
"#;

fn truth_toml(dir: &Path) -> PathBuf {
    let p = dir.join("truth.toml");
    std::fs::write(&p, "beta = 0.8\ngamma = 0.3\nN0 = 1000\n").unwrap();
    p
}

/// The `(win_start, win_stop, value)` rows of a windowed stream's file, the
/// value as written (`NA` for a hole).
fn window_rows(path: &Path) -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "win_start\twin_stop\tcases",
        "the declared columns, so the loader reads the file back (gh#830/gh#833)");
    lines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[0].to_string(), f[1].to_string(), f[2].to_string())
    }).collect()
}

/// `camdl simulate --design-from` reproduces the bound design exactly: the same
/// windows, in the same order, with the hole still a hole. Only the values
/// differ — they are drawn from the model.
#[test]
fn design_from_writes_the_bound_rows_windows_and_holes() {
    if camdlc().is_none() { return; }
    let tmp = tempdir("design_from");
    let ir = compile(tmp.path(), "windowed", WINDOWED_MODEL);
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, IRREGULAR_DATA).unwrap();
    let truth = truth_toml(tmp.path());

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "[model]\ncamdl = \"{}\"\n\n[data.observations]\ncases = \"{}\"\n{}",
        ir.display(), data.display(), STAGES)).unwrap();

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "11",
    ]);
    assert!(output.status.success(),
        "simulate --design-from must run:\n{}", String::from_utf8_lossy(&output.stderr));

    let observed = window_rows(&data);
    let simulated = window_rows(&out_dir.join("cases.tsv"));
    let windows = |rows: &[(String, String, String)]| -> Vec<(String, String)> {
        rows.iter().map(|(a, b, _)| (a.clone(), b.clone())).collect()
    };
    assert_eq!(windows(&simulated), windows(&observed),
        "every row's own window is reproduced — including the three-day and \
         two-day rows no `covers` form can state");
    assert_eq!(
        simulated.iter().map(|(_, _, v)| v == "NA").collect::<Vec<_>>(),
        observed.iter().map(|(_, _, v)| v == "NA").collect::<Vec<_>>(),
        "the hole stays a hole: a row with its period and no value");
    assert!(simulated.iter().filter(|(_, _, v)| v != "NA").any(|(_, _, v)| v != "0"),
        "the observed rows are drawn from the model, not written as zeros: {simulated:?}");

    // The round trip: the generated file re-loads under the model that wrote
    // it, through the same loader the real data uses.
    let stem = tmp.path().join("preq");
    let output = run(&[
        "pfilter", ir.to_str().unwrap(),
        "--particles", "100", "--dt", "1", "--seed", "1",
        "--data", out_dir.join("cases.tsv").to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--save-prequential", stem.to_str().unwrap(),
    ]);
    assert!(output.status.success(),
        "the generated file must re-load under its own model:\n{}",
        String::from_utf8_lossy(&output.stderr));
    let trace: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}.json", stem.display())).unwrap()).unwrap();
    let scored: Vec<(String, String)> = trace["steps"].as_array().unwrap().iter()
        .map(|s| {
            let c = &s["per_stream"][0]["coverage"];
            assert_eq!(c["kind"], "interval");
            (format!("{}", c["start"].as_f64().unwrap()),
             format!("{}", c["stop"].as_f64().unwrap()))
        })
        .collect();
    // A hole carries no likelihood term, so the prequential trace has no step
    // for it (gh#636); every other period is scored exactly as written.
    let scorable: Vec<(String, String)> = observed.iter()
        .filter(|(_, _, v)| v != "NA")
        .map(|(a, b, _)| (a.clone(), b.clone()))
        .collect();
    assert_eq!(scored, scorable,
        "the periods scored on the generated file are the periods the fit binds");
}

/// A `[synthetic]` fit on a windowed model runs end to end. Before the
/// per-stream emitter this was refused outright — the single wide `ds_NN.tsv`
/// has one shared time column and cannot carry `window_start`/`window_stop` —
/// so the self-consistency test the docs call the highest-return step could not
/// be run on such a model at all.
#[test]
fn a_synthetic_fit_on_a_windowed_model_completes() {
    if camdlc().is_none() { return; }
    let tmp = tempdir("synthetic_windowed");
    let ir = compile(tmp.path(), "windowed", WINDOWED_MODEL);
    let truth = truth_toml(tmp.path());
    let out = tmp.path().join("out");

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "output_dir = \"{}\"\n\n[model]\ncamdl = \"{}\"\n\n\
         [synthetic]\ntrue_params = \"{}\"\nsim_seeds = [3]\n{}",
        out.display(), ir.display(), truth.display(), STAGES)).unwrap();

    let output = run(&["fit", "run", fit_toml.to_str().unwrap()]);
    assert!(output.status.success(),
        "a [synthetic] fit on a windowed model must complete:\n{}",
        String::from_utf8_lossy(&output.stderr));

    // The generated dataset: one file per stream, under the declared window
    // columns, contiguous over the run.
    let mut generated: Option<PathBuf> = None;
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        let candidate = d.join("cases.tsv");
        if candidate.is_file() && d.file_name().map(|n| n == "ds_01").unwrap_or(false) {
            generated = Some(candidate);
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() { stack.push(e.path()); }
            }
        }
    }
    let generated = generated.unwrap_or_else(
        || panic!("no synthetic/data/ds_01/cases.tsv under {}", out.display()));
    let rows = window_rows(&generated);
    assert_eq!(rows.first().map(|(a, b, _)| (a.as_str(), b.as_str())), Some(("0", "1")),
        "windows open at the run's start: {rows:?}");
    assert!(rows.windows(2).all(|w| w[0].1 == w[1].0),
        "the declared emission is contiguous: {rows:?}");
    assert!(rows.iter().all(|(_, _, v)| v != "NA"),
        "a declared emission has no holes: {rows:?}");
}

/// gh#829's own reproduction: a binomial denominator the data file supplies and
/// the model has no term to generate.
const COVARIATE_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  survey {
    columns       { time : time, positives : count, tested : count }
    projected     = I / N0
    emit_schedule = every 2 'days
    positives     ~ binomial(n = tested, p = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 10 'days }
"#;

/// gh#829, at the CLI. A stream whose likelihood reads a data column has no
/// source for that column when the data is what is being generated. It is
/// refused by name — a file of zeros would assert "we tested and found
/// nothing", which is then scored as a real observation.
#[test]
fn design_from_refuses_a_covariate_stream_by_name_and_writes_nothing() {
    if camdlc().is_none() { return; }
    let tmp = tempdir("covariate");
    let ir = compile(tmp.path(), "survey", COVARIATE_MODEL);
    let data = tmp.path().join("survey.tsv");
    std::fs::write(&data,
        "time\tpositives\ttested\n2\t3\t100\n4\t7\t120\n6\t9\t90\n").unwrap();
    let truth = truth_toml(tmp.path());

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "[model]\ncamdl = \"{}\"\n\n[data.observations]\nsurvey = \"{}\"\n{}",
        ir.display(), data.display(), STAGES)).unwrap();

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "11",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(),
        "a covariate stream must be refused, not written:\n{stderr}");
    assert!(stderr.contains("'survey'") && stderr.contains("`tested`")
            && stderr.contains("gh#829"),
        "the refusal names the stream, the column and the issue:\n{stderr}");
    assert!(!out_dir.join("survey.tsv").exists(),
        "nothing is written for a stream that cannot be drawn honestly");
}

/// The same refusal on the `[synthetic]` path, which reaches it through the
/// declared design rather than a bound one. Before this it drew the covariate
/// as 0 from an empty aux slice — the gh#829 shape, on the very workflow that
/// exists to test whether a model is identified.
#[test]
fn a_synthetic_fit_refuses_a_covariate_stream_by_name() {
    if camdlc().is_none() { return; }
    let tmp = tempdir("synthetic_covariate");
    let ir = compile(tmp.path(), "survey", COVARIATE_MODEL);
    let truth = truth_toml(tmp.path());
    let out = tmp.path().join("out");

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "output_dir = \"{}\"\n\n[model]\ncamdl = \"{}\"\n\n\
         [synthetic]\ntrue_params = \"{}\"\nsim_seeds = [3]\n{}",
        out.display(), ir.display(), truth.display(), STAGES)).unwrap();

    let output = run(&["fit", "run", fit_toml.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(),
        "a [synthetic] fit on a covariate stream must be refused:\n{stderr}");
    assert!(stderr.contains("'survey'") && stderr.contains("`tested`")
            && stderr.contains("gh#829"),
        "the refusal names the stream, the column and the issue:\n{stderr}");
    let mut written: Vec<PathBuf> = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.extension().map(|x| x == "tsv").unwrap_or(false) {
                    written.push(p);
                }
            }
        }
    }
    assert!(written.is_empty(),
        "no dataset is written for a stream that cannot be drawn: {written:?}");
}
