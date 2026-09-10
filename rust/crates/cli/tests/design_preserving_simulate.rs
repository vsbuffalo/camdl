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

[method]
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

// ── The real-shaped fixture (gh#878) ────────────────────────────────────────
//
// `tests/fixtures/real_shaped/` is the same two surfaces on input shaped the
// way published surveillance is: ISO dates in both window boundaries, one-day
// rows with a three-day and a five-day row among them, a scheduled row with no
// count, a stratified stream in long form, and a denominator the file supplies.

fn real_shaped(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/real_shaped").join(name)
}

/// The committed fixture model, compiled into `dir` through the same helper the
/// inline models above use.
fn compile_real_shaped(dir: &Path) -> PathBuf {
    let src = std::fs::read_to_string(real_shaped("surveillance.camdl")).unwrap();
    compile(dir, "surveillance", &src)
}

/// The fixture model's `origin`.
const REAL_SHAPED_ORIGIN: &str = "2026-06-29";

/// A temporal cell as model time, whichever representation the file states it
/// in — an ISO date read through the fixture's origin, or a bare day offset.
/// The assertions that compare periods are about the instants, not the cells;
/// the cells are asserted separately (gh#882).
fn as_time(cell: &str) -> f64 {
    if cell.contains('-') {
        ir::caltime::date_to_internal(REAL_SHAPED_ORIGIN, cell, "days")
            .unwrap_or_else(|e| panic!("cannot read '{cell}' through the origin: {e:?}"))
    } else {
        cell.parse().unwrap_or_else(|e| panic!("cannot read '{cell}' as a time: {e}"))
    }
}

const REAL_SHAPED_STAGES: &str = r#"
[estimate]
beta = { bounds = [0.25, 0.5], start = 0.35 }

[fixed]
gamma = 0.2
rho   = 0.5
psi   = 0.3
k     = 20

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 40
iterations = 3
cooling    = 0.7
"#;

fn real_shaped_truth(dir: &Path) -> PathBuf {
    let p = dir.join("truth.toml");
    std::fs::write(&p, "beta = 0.35\ngamma = 0.2\nrho = 0.5\npsi = 0.3\nk = 20\n").unwrap();
    p
}

/// A fit config binding the named streams to the named committed files.
fn real_shaped_fit(dir: &Path, ir: &Path, binds: &[(&str, &str)]) -> PathBuf {
    let paths: Vec<(&str, PathBuf)> =
        binds.iter().map(|(s, f)| (*s, real_shaped(f))).collect();
    real_shaped_fit_paths(dir, ir, &paths)
}

/// The same, binding streams to arbitrary paths — for re-binding a dataset the
/// emitter just wrote.
fn real_shaped_fit_paths(dir: &Path, ir: &Path, binds: &[(&str, PathBuf)]) -> PathBuf {
    let mut s = format!("[model]\ncamdl = \"{}\"\n\n[data.observations]\n", ir.display());
    for (stream, file) in binds {
        s.push_str(&format!("{stream} = \"{}\"\n", file.display()));
    }
    s.push_str(REAL_SHAPED_STAGES);
    let p = dir.join("fit.toml");
    std::fs::write(&p, s).unwrap();
    p
}

/// `(start, stop, value)` of a windowed stream's file, the boundaries as model
/// time and the value as written (`NA` for a hole).
fn window_periods(path: &Path) -> Vec<(f64, f64, String)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[..3], ["window_start", "window_stop", "cases"],
        "the declared columns, so the loader reads the file back");
    lines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (as_time(f[0]), as_time(f[1]), f[2].to_string())
    }).collect()
}

/// `simulate --design-from` on the real-shaped bulletin reproduces every row's
/// own period and its hole. The bound file's boundaries are ISO dates of uneven
/// width; the generated file's are day offsets naming the same instants — the
/// representation gap is gh#882, and this asserts the periods, not the cells.
#[test]
fn design_from_reproduces_the_real_shaped_windows_and_hole() {
    let tmp = tempdir("real_shaped_design");
    let ir = compile_real_shaped(tmp.path());
    let truth = real_shaped_truth(tmp.path());
    let fit_toml = real_shaped_fit(
        tmp.path(), &ir, &[("cases", "bulletin.tsv"), ("in_care", "bulletin.tsv")]);

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "0.5", "--seed", "11",
    ]);
    assert!(output.status.success(),
        "simulate --design-from must run on the real-shaped design:\n{}",
        String::from_utf8_lossy(&output.stderr));

    let observed = window_periods(&real_shaped("bulletin.tsv"));
    let simulated = window_periods(&out_dir.join("cases.tsv"));

    // Non-vacuous: the bound design is the one no uniform `covers` form states.
    assert!(observed.iter().any(|(a, b, _)| b - a > 2.0),
        "the bulletin carries a row wider than two days");
    assert_eq!(observed.iter().filter(|(_, _, v)| v == "NA").count(), 1,
        "and exactly one scheduled row with no count");

    let periods = |rows: &[(f64, f64, String)]| -> Vec<(f64, f64)> {
        rows.iter().map(|(a, b, _)| (*a, *b)).collect()
    };
    assert_eq!(periods(&simulated), periods(&observed),
        "every row's own period is reproduced — the three-day and five-day rows \
         among them");
    assert_eq!(
        simulated.iter().map(|(_, _, v)| v == "NA").collect::<Vec<_>>(),
        observed.iter().map(|(_, _, v)| v == "NA").collect::<Vec<_>>(),
        "the row with no count stays a row with no count");
    assert!(simulated.iter().filter(|(_, _, v)| v != "NA").any(|(_, _, v)| v != "0"),
        "the values are drawn from the model, not written as zeros: {simulated:?}");
}

/// The raw temporal cells of a file, as the file spells them: the first `n`
/// tab-separated fields of every row after the header.
fn temporal_cells(path: &Path, n: usize) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().skip(1)
        .map(|l| l.split('\t').take(n).map(str::to_string).collect())
        .collect()
}

/// gh#882: a design bound from an ISO-dated file is re-emitted dated. The
/// bulletin's two window boundaries and the stock stream's single label column
/// are all written as ISO dates in the fixture, so the design-preserving
/// writer renders them back through the model's `origin` and `time_unit` —
/// cell for cell the strings that went in, not the day offsets they convert
/// to. That is the round-trip property gh#831 asked for, extended from column
/// names to column representation.
#[test]
fn design_from_writes_a_dated_design_back_as_dates() {
    let tmp = tempdir("real_shaped_dates");
    let ir = compile_real_shaped(tmp.path());
    let truth = real_shaped_truth(tmp.path());
    let fit_toml = real_shaped_fit(
        tmp.path(), &ir, &[("cases", "bulletin.tsv"), ("in_care", "bulletin.tsv")]);

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "0.5", "--seed", "11",
    ]);
    assert!(output.status.success(),
        "simulate --design-from must run on the real-shaped design:\n{}",
        String::from_utf8_lossy(&output.stderr));

    // Non-vacuous: the bound file really is dated.
    let observed = temporal_cells(&real_shaped("bulletin.tsv"), 2);
    assert!(observed.iter().all(|r| r[0].contains('-') && r[1].contains('-')),
        "the fixture's window boundaries are ISO dates: {observed:?}");

    // A per-row window pair: both declared boundary columns come back dated.
    assert_eq!(temporal_cells(&out_dir.join("cases.tsv"), 2), observed,
        "every window boundary is written as the date the bound file stated");

    // A single `: time` label column on the same dated table.
    let stops: Vec<Vec<String>> = observed.iter().map(|r| vec![r[1].clone()]).collect();
    assert_eq!(temporal_cells(&out_dir.join("in_care.tsv"), 1), stops,
        "the stock stream's label column is written dated too");

    // And the dated file re-loads: the loader binds it with the same periods.
    let reload_toml = real_shaped_fit_paths(tmp.path(), &ir, &[
        ("cases", out_dir.join("cases.tsv")),
        ("in_care", out_dir.join("in_care.tsv")),
    ]);
    let reload = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", reload_toml.to_str().unwrap(),
        "--obs-only-dir", tmp.path().join("synth2").to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "0.5", "--seed", "11",
    ]);
    assert!(reload.status.success(),
        "the emitted dated file must re-load under its own model:\n{}",
        String::from_utf8_lossy(&reload.stderr));
    assert_eq!(temporal_cells(&tmp.path().join("synth2").join("cases.tsv"), 2), observed,
        "and re-emits the same dates, so the representation is a fixed point");
}

/// gh#829 on real-shaped input. The fixture's survey stream reads `tested` from
/// its file; bound into a design, it is refused by name and nothing is written.
#[test]
fn design_from_refuses_the_real_shaped_survey_by_name() {
    let tmp = tempdir("real_shaped_covariate");
    let ir = compile_real_shaped(tmp.path());
    let truth = real_shaped_truth(tmp.path());
    let fit_toml = real_shaped_fit(
        tmp.path(), &ir, &[("cases", "bulletin.tsv"), ("survey", "survey.tsv")]);

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "0.5", "--seed", "11",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(),
        "a covariate stream must be refused, not written:\n{stderr}");
    assert!(stderr.contains("'survey'") && stderr.contains("`tested`")
            && stderr.contains("gh#829"),
        "the refusal names the stream, the column and the issue:\n{stderr}");
    assert!(!out_dir.join("survey.tsv").exists() && !out_dir.join("cases.tsv").exists(),
        "nothing is written once one stream of the design cannot be drawn");
}

/// A `[synthetic]` fit on the whole real-shaped model is refused by name, and
/// before anything is written. The declared design covers every stream the
/// model has, and this model has two the writer cannot produce a loadable file
/// for: the stratified family (one long-format file per source, which a
/// one-file-per-stream writer cannot make) and the survey (gh#829). The
/// stratified one is reached first.
#[test]
fn a_synthetic_fit_on_the_real_shaped_model_is_refused_by_name() {
    let tmp = tempdir("real_shaped_synthetic");
    let ir = compile_real_shaped(tmp.path());
    let truth = real_shaped_truth(tmp.path());
    let out = tmp.path().join("out");

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "output_dir = \"{}\"\n\n[model]\ncamdl = \"{}\"\n\n\
         [synthetic]\ntrue_params = \"{}\"\nsim_seeds = [3]\n{}",
        out.display(), ir.display(), truth.display(), REAL_SHAPED_STAGES)).unwrap();

    let output = run(&["fit", "run", fit_toml.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(),
        "a [synthetic] fit that cannot write a loadable dataset must be refused:\n{stderr}");
    assert!(stderr.contains("'province_cases_") && stderr.contains("stratified"),
        "the refusal names the stream and why its file would not re-load:\n{stderr}");
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
        "no dataset is written for a model one of whose streams cannot be drawn: {written:?}");
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

// ── A ratio stream's denominator comes from the model (proposal 2026-09-09) ──

/// `(label, kivu_cases, cases_split)` rows of the share file, as written.
fn share_rows(path: &Path) -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "week_ending\tkivu_cases\tcases_split",
        "the declared columns — the denominator among them — so the loader reads the file back");
    lines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[0].to_string(), f[1].to_string(), f[2].to_string())
    }).collect()
}

/// `simulate --design-from` on the real-shaped fixture's `kivu_share` stream —
/// of the week's cases whose province was recorded, how many were in Kivu,
/// scored `binomial(n = cases_split, p = projected)` over a ratio of flows —
/// writes the denominator column from the model: the ratio's own denominator
/// flow over each row, an integer, with `k` drawn against it. The hole row
/// keeps its hole in the count and still carries a denominator. The file
/// re-loads under the model that wrote it.
///
/// Before the emitter wrote the column this stream was refused by name, like
/// the survey's `tested` — which stays refused: surveillance effort is not a
/// quantity the model generates.
#[test]
fn design_from_writes_a_ratio_streams_denominator_from_the_model() {
    let tmp = tempdir("real_shaped_ratio");
    let ir = compile_real_shaped(tmp.path());
    let truth = real_shaped_truth(tmp.path());
    let fit_toml = real_shaped_fit(
        tmp.path(), &ir, &[("cases", "bulletin.tsv"), ("kivu_share", "province_share.tsv")]);

    let out_dir = tmp.path().join("synth");
    let output = run(&[
        "simulate", ir.to_str().unwrap(),
        "--params", truth.to_str().unwrap(),
        "--design-from", fit_toml.to_str().unwrap(),
        "--obs-only-dir", out_dir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "0.5", "--seed", "11",
    ]);
    assert!(output.status.success(),
        "a ratio stream's denominator is the model's to write:\n{}",
        String::from_utf8_lossy(&output.stderr));

    let observed = share_rows(&real_shaped("province_share.tsv"));
    let simulated = share_rows(&out_dir.join("kivu_share.tsv"));
    assert_eq!(simulated.len(), observed.len(), "one row per bound row");
    assert_eq!(
        simulated.iter().map(|(_, k, _)| k == "NA").collect::<Vec<_>>(),
        observed.iter().map(|(_, k, _)| k == "NA").collect::<Vec<_>>(),
        "the week with no recorded split stays a hole in the count"
    );
    let mut any_positive = false;
    for (label, k, n) in &simulated {
        let n: i64 = n.parse().unwrap_or_else(|_| panic!(
            "row {label}: the denominator is written from the model as an integer, got {n:?}"));
        assert!(n >= 0, "row {label}: a count");
        any_positive |= n > 0;
        if k != "NA" {
            let k: i64 = k.parse().unwrap();
            assert!(k <= n, "row {label}: k = {k} of n = {n} — drawn against the model's own denominator");
        }
    }
    assert!(any_positive, "the epidemic produced cases, so some week has a denominator: {simulated:?}");

    // The file re-loads under the model, with the model's denominators bound.
    let reload = run(&[
        "pfilter", ir.to_str().unwrap(), "--particles", "50", "--dt", "0.5", "--seed", "1",
        "--params", truth.to_str().unwrap(),
        "--data", &format!("kivu_share={}", out_dir.join("kivu_share.tsv").display()),
    ]);
    assert!(reload.status.success(),
        "the emitted share file must re-load under the model that wrote it:\n{}",
        String::from_utf8_lossy(&reload.stderr));
}

/// A proportion-of-flows stream with a data-column denominator, in the smallest
/// model that has one.
const RATIO_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, H, Dc, Df }
let N = S + I + H
parameters {
  beta : rate in [0.01, 2.0]
  eta  : rate in [0.01, 1.0]
  mu_c : rate in [0.001, 0.5]
  mu_f : rate in [0.001, 0.5]
}
transitions {
  infection   : S --> I  @ beta * S * I / N
  hospitalise : I --> H  @ eta * I
  die_comm    : I --> Dc @ mu_c * I
  die_fac     : H --> Df @ mu_f * H
}
observations {
  comm_frac {
    columns       { time : time, comm_deaths : count, n_deaths : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))
    emit_schedule = every 7 'days
    comm_deaths   ~ binomial(n = n_deaths, p = projected)
  }
}
init { S = 4990  I = 10 }
simulate { from = 0 'days  to = 42 'days }
"#;

const RATIO_STAGES: &str = r#"
[estimate]
mu_c = { bounds = [0.001, 0.5], start = 0.05 }

[fixed]
beta = 0.5
eta  = 0.15
mu_f = 0.1

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 50
iterations = 3
cooling    = 0.7
"#;

/// A `[synthetic]` fit on a ratio stream with a data-column denominator
/// completes: the dataset it writes carries the denominator from the model, and
/// the fit reads it back. Before the emitter wrote the column this was refused
/// by name (gh#829), on the very workflow that exists to test whether a
/// community share is identified.
#[test]
fn a_synthetic_fit_on_a_ratio_stream_completes() {
    let tmp = tempdir("synthetic_ratio");
    let ir = compile(tmp.path(), "ratio", RATIO_MODEL);
    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, "beta = 0.5\neta = 0.15\nmu_c = 0.05\nmu_f = 0.1\n").unwrap();
    let out = tmp.path().join("out");

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(
        "output_dir = \"{}\"\n\n[model]\ncamdl = \"{}\"\n\n\
         [synthetic]\ntrue_params = \"{}\"\nsim_seeds = [3]\n{}",
        out.display(), ir.display(), truth.display(), RATIO_STAGES)).unwrap();

    let output = run(&["fit", "run", fit_toml.to_str().unwrap()]);
    assert!(output.status.success(),
        "a [synthetic] fit on a ratio stream must complete:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));

    // The dataset carries the model's denominator under the declared column.
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().map(|f| f == "comm_frac.tsv").unwrap_or(false) {
                    files.push(p);
                }
            }
        }
    }
    assert_eq!(files.len(), 1, "one dataset for one sim seed: {files:?}");
    let text = std::fs::read_to_string(&files[0]).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "time\tcomm_deaths\tn_deaths");
    let rows: Vec<(i64, i64)> = lines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[1].parse().unwrap(), f[2].parse().unwrap())
    }).collect();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|&(k, n)| 0 <= k && k <= n), "k of n, drawn against the model's denominator: {rows:?}");
    assert!(rows.iter().any(|&(_, n)| n > 0), "the epidemic produced deaths: {rows:?}");
}
