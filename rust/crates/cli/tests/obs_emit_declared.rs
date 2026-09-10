//! gh#833, ruling 3: `simulate --obs-dir` writes what the loader reads.
//!
//! The emitter follows the stream's declaration: the value goes under the
//! scored column's name (gh#830), the time under the declared `: time` column,
//! a windowed stream under its `window_start`/`window_stop` names, and a row
//! whose period falls outside the run is not written. Under `closing_at` with
//! a schedule starting at `t_start` that drops the leading row: its period
//! `[t_start − Δ, t_start)` was never simulated.
//!
//! Proposal Testing item 7 is the round trip: a windowed model's emitted file
//! re-loads under the same model with exactly the periods that were written,
//! read back off the prequential trace's per-stream `coverage`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn seed_timing_ir() -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::fs::read_to_string(Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json")).unwrap()
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_emit_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl).args(args).env("CAMDL_SKIP_VERSION_CHECK", "1").output().expect("camdl must invoke")
}

fn write_model(dir: &Path, name: &str, src: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, src).unwrap();
    p
}

/// The fixture with its stream's `covers` replaced (`None` removes it — an IR
/// the compiler cannot produce, which the runtime refuses). The committed
/// fixture declares `closing_at(time, 1 'days)`; edit it as JSON rather than by
/// text injection.
fn with_covers(src: &str, covers: Option<serde_json::Value>) -> String {
    let mut v: serde_json::Value = serde_json::from_str(src).unwrap();
    let obs = v["model"]["observations"][0].as_object_mut().expect("one stream");
    match covers {
        Some(c) => { obs.insert("covers".into(), c); }
        None => { obs.remove("covers"); }
    }
    serde_json::to_string_pretty(&v).unwrap()
}

/// The fixture with its `: time` column replaced by a window pair.
fn with_window_columns(src: &str) -> String {
    let with_cov = with_covers(src, Some(serde_json::json!({ "kind": "window_columns" })));
    let mut v: serde_json::Value = serde_json::from_str(&with_cov).unwrap();
    let cols = v["model"]["observations"][0]["columns"].as_array_mut().expect("columns");
    let time_idx = cols.iter().position(|c| c["role"] == "time").expect("a time column");
    cols.splice(time_idx..=time_idx, [
        serde_json::json!({ "name": "win_start", "role": "window_start" }),
        serde_json::json!({ "name": "win_stop", "role": "window_stop" }),
    ]);
    serde_json::to_string_pretty(&v).unwrap()
}

const PARAMS: &[&str] = &[
    "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "lambda=2.0", "--param", "w=3.0",
    "--param", "N0=5000", "--param", "rho=0.5", "--param", "k=20", "--param", "tau=2",
];

fn emit(camdl: &Path, model: &Path, out_dir: &Path) -> String {
    let mut args = vec![
        "simulate", model.to_str().unwrap(), "--backend", "chain_binomial", "--dt", "1",
        "--seed", "3", "--obs-only-dir", out_dir.to_str().unwrap(),
    ];
    args.extend_from_slice(PARAMS);
    let out = run(camdl, &args);
    assert!(out.status.success(), "simulate --obs-only-dir failed:\n{}", String::from_utf8_lossy(&out.stderr));
    std::fs::read_to_string(out_dir.join("cases.tsv")).expect("cases.tsv written")
}

fn first_column(line: &str) -> &str {
    line.split('\t').next().unwrap()
}

/// The fixture's schedule is `regular { start 0, step 1 }` over a run
/// `[0, 120]` under `closing_at(time, 1 'days)`: the emit time 0 would cover
/// `[−1, 0)`, which the run never simulated, and is not written; every other
/// emit time covers a period inside the run and is.
#[test]
fn closing_at_writes_every_row_the_run_covers_and_no_leading_zero_width_row() {
    let camdl = camdl_bin();
    let tmp = tempdir("closing");
    let closing = write_model(&tmp, "closing.ir.json", &seed_timing_ir());

    let c = emit(&camdl, &closing, &tmp.join("c"));
    let c_lines: Vec<&str> = c.lines().collect();
    assert_eq!(c_lines[0], "time\tcases", "the time under the declared column's name");
    let labels: Vec<&str> = c_lines[1..].iter().map(|l| first_column(l)).collect();
    let expected: Vec<String> = (1..=120).map(|t| t.to_string()).collect();
    assert_eq!(labels, expected, "labels 1..=120: the row at t_start has no simulated period");
}

/// An accumulating stream whose IR carries no `covers` is an IR the compiler
/// cannot have produced (E350); the emitter has no reading to write under and
/// refuses, naming the stream and the rule.
#[test]
fn an_incidence_stream_with_no_declaration_is_refused_by_the_emitter() {
    let camdl = camdl_bin();
    let tmp = tempdir("undeclared");
    let undeclared = write_model(&tmp, "undeclared.ir.json", &with_covers(&seed_timing_ir(), None));
    let out_dir = tmp.join("u");
    let mut args = vec![
        "simulate", undeclared.to_str().unwrap(), "--backend", "chain_binomial", "--dt", "1",
        "--seed", "3", "--obs-only-dir", out_dir.to_str().unwrap(),
    ];
    args.extend_from_slice(PARAMS);
    let out = run(&camdl, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "must refuse; stderr:\n{stderr}");
    assert!(stderr.contains("'cases'") && stderr.contains("E350"),
        "names the stream and the rule:\n{stderr}");
    assert!(!out_dir.join("cases.tsv").exists(), "nothing is written under an unstated reading");
}

#[test]
fn the_value_goes_under_the_scored_column_not_the_stream_name() {
    // gh#830: a stream named `reported` whose scored column is `cases`.
    let camdl = camdl_bin();
    let tmp = tempdir("scored");
    let mut v: serde_json::Value = serde_json::from_str(&seed_timing_ir()).unwrap();
    v["model"]["observations"][0]["name"] = serde_json::json!("reported");
    let src = serde_json::to_string_pretty(&v).unwrap();
    let model = write_model(&tmp, "renamed.ir.json", &src);
    let out_dir = tmp.join("o");
    let mut args = vec![
        "simulate", model.to_str().unwrap(), "--backend", "chain_binomial", "--dt", "1",
        "--seed", "3", "--obs-only-dir", out_dir.to_str().unwrap(),
    ];
    args.extend_from_slice(PARAMS);
    let out = run(&camdl, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = std::fs::read_to_string(out_dir.join("reported.tsv")).expect("one file per STREAM");
    assert_eq!(text.lines().next().unwrap(), "time\tcases",
        "the header names the scored column, which is what the loader reads");
}

/// Proposal Testing item 7. A windowed model emits `win_start`/`win_stop`
/// under its declared names; `pfilter` reads the file back under the same
/// model, and the periods it scores — recorded per step as `coverage` in the
/// prequential trace — are exactly the windows that were written.
#[test]
fn a_windowed_stream_round_trips_through_its_own_emitted_file() {
    let camdl = camdl_bin();
    let tmp = tempdir("roundtrip");
    let model = write_model(&tmp, "windowed.ir.json", &with_window_columns(&seed_timing_ir()));
    let out_dir = tmp.join("o");
    let text = emit(&camdl, &model, &out_dir);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "win_start\twin_stop\tcases", "the declared window columns, then the scored column");
    let written: Vec<(f64, f64)> = lines[1..].iter().map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[0].parse().unwrap(), f[1].parse().unwrap())
    }).collect();
    assert_eq!(written[0], (0.0, 1.0), "contiguous windows from the run's start; no zero-width row");
    assert!(written.windows(2).all(|w| w[0].1 == w[1].0), "contiguous");

    let stem = tmp.join("preq");
    let data = out_dir.join("cases.tsv");
    let mut args = vec![
        "pfilter", model.to_str().unwrap(), "--particles", "100", "--dt", "1", "--seed", "1",
        "--data", data.to_str().unwrap(),
        "--save-prequential", stem.to_str().unwrap(),
    ];
    args.extend_from_slice(PARAMS);
    let out = run(&camdl, &args);
    assert!(out.status.success(), "the emitted file must re-load under its own model:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}.json", stem.display())).unwrap()).unwrap();
    let scored: Vec<(f64, f64)> = trace["steps"].as_array().unwrap().iter().map(|s| {
        let c = &s["per_stream"][0]["coverage"];
        assert_eq!(c["kind"], "interval");
        (c["start"].as_f64().unwrap(), c["stop"].as_f64().unwrap())
    }).collect();
    assert_eq!(scored, written, "the periods scored are the periods written — the round trip closes");
}

// ── The real-shaped fixture (gh#878) ────────────────────────────────────────
//
// `tests/fixtures/real_shaped/` is a model with an `origin` and four streams
// shaped the way published surveillance is. Two of them the emitter writes
// readably; two it does not, and those are pinned below as ignored tests
// against the issues that own them.

/// The compiler, built by `build-ocaml`, which both `make test-rust` and
/// `make test-fast` depend on.
fn camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(
        p.exists(),
        "camdlc missing: {} - run `make build-ocaml`, or gate with `make test-rust`",
        p.display()
    );
    p
}

fn real_shaped(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/real_shaped").join(name)
}

fn compile_real_shaped(dir: &Path) -> PathBuf {
    let ir = dir.join("surveillance.ir.json");
    let out = std::process::Command::new(camdlc())
        .arg(real_shaped("surveillance.camdl"))
        .arg("-o")
        .arg(&ir)
        .output()
        .expect("camdlc must invoke");
    assert!(out.status.success(), "camdlc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    ir
}

const REAL_SHAPED_PARAMS: &[&str] = &[
    "--param", "beta=0.35", "--param", "gamma=0.2", "--param", "rho=0.5",
    "--param", "psi=0.3", "--param", "k=20",
];

/// Write the real-shaped model's declared dataset into `out_dir`.
fn emit_real_shaped(camdl: &Path, ir: &Path, out_dir: &Path) {
    let mut args = vec![
        "simulate", ir.to_str().unwrap(), "--backend", "chain_binomial", "--dt", "0.5",
        "--seed", "3", "--obs-only-dir", out_dir.to_str().unwrap(),
    ];
    args.extend_from_slice(REAL_SHAPED_PARAMS);
    let out = run(camdl, &args);
    assert!(out.status.success(), "simulate --obs-only-dir failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

/// Bind the named emitted files back under the model that wrote them.
fn reload(camdl: &Path, ir: &Path, out_dir: &Path, streams: &[&str]) -> std::process::Output {
    let binds: Vec<String> = streams.iter()
        .map(|s| format!("{s}={}", out_dir.join(format!("{s}.tsv")).display()))
        .collect();
    let mut args = vec![
        "pfilter", ir.to_str().unwrap(), "--particles", "50", "--dt", "0.5", "--seed", "1",
    ];
    for b in &binds {
        args.push("--data");
        args.push(b);
    }
    args.extend_from_slice(REAL_SHAPED_PARAMS);
    run(camdl, &args)
}

/// The two real-shaped streams the emitter writes readably: a windowed
/// incidence stream, and a prevalence stream whose `: time` column is the same
/// bulletin table's closing boundary. Both come back under their declared
/// column names, and both re-load under the model that wrote them.
#[test]
fn the_real_shaped_windowed_and_instant_streams_round_trip() {
    let camdl = camdl_bin();
    let tmp = tempdir("real_shaped");
    let ir = compile_real_shaped(&tmp);
    let out_dir = tmp.join("o");
    emit_real_shaped(&camdl, &ir, &out_dir);

    let cases = std::fs::read_to_string(out_dir.join("cases.tsv")).expect("cases.tsv");
    let lines: Vec<&str> = cases.lines().collect();
    assert_eq!(lines[0], "window_start\twindow_stop\tcases",
        "the declared window columns, then the scored column");
    let written: Vec<(f64, f64)> = lines[1..].iter().map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[0].parse().unwrap(), f[1].parse().unwrap())
    }).collect();
    assert!(written.windows(2).all(|w| w[0].1 == w[1].0), "contiguous: {written:?}");

    let care = std::fs::read_to_string(out_dir.join("in_care.tsv")).expect("in_care.tsv");
    assert_eq!(care.lines().next().unwrap(), "window_stop\tin_care",
        "a stream whose `: time` column is named for a window boundary keeps that name");

    let out = reload(&camdl, &ir, &out_dir, &["cases", "in_care"]);
    assert!(out.status.success(),
        "the emitted files must re-load under the model that wrote them:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

/// gh#884. A stratified family shares one `source` and one long-format file,
/// and the loader routes each row to its stratum leaf by the `: dim` column's
/// value — so the writer emits the family as that one file, dim column
/// included, rather than as one column-less file per leaf that nothing can
/// read back.
#[test]
fn the_real_shaped_stratified_family_round_trips() {
    let camdl = camdl_bin();
    let tmp = tempdir("real_shaped_family");
    let ir = compile_real_shaped(&tmp);
    let out_dir = tmp.join("o");
    emit_real_shaped(&camdl, &ir, &out_dir);

    let text = std::fs::read_to_string(out_dir.join("province_cases.tsv"))
        .expect("one long-format file for the family's source");
    assert_eq!(text.lines().next().unwrap(), "week_ending\tprovince\tprovince_cases",
        "the declared columns, the dim column among them");
    let out = reload(&camdl, &ir, &out_dir, &["province_cases"]);
    assert!(out.status.success(),
        "the emitted family file must re-load under the model that wrote it:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

/// gh#830. The survey stream declares `tested`, which its likelihood reads;
/// the writer drops it, so the file it writes has no denominator to be scored
/// against and does not re-load.
#[test]
#[ignore = "gh#830 — --obs-dir drops the covariate column the likelihood reads"]
fn the_real_shaped_covariate_stream_round_trips() {
    let camdl = camdl_bin();
    let tmp = tempdir("real_shaped_covariate");
    let ir = compile_real_shaped(&tmp);
    let out_dir = tmp.join("o");
    emit_real_shaped(&camdl, &ir, &out_dir);

    let text = std::fs::read_to_string(out_dir.join("survey.tsv")).expect("survey.tsv");
    assert_eq!(text.lines().next().unwrap(), "time\tpositives\ttested",
        "the declared columns, the denominator among them");
    let out = reload(&camdl, &ir, &out_dir, &["survey"]);
    assert!(out.status.success(),
        "the emitted survey file must re-load under the model that wrote it:\n{}",
        String::from_utf8_lossy(&out.stderr));
}

#[test]
fn a_wide_obs_file_refuses_a_windowed_stream_and_names_the_escape() {
    let camdl = camdl_bin();
    let tmp = tempdir("wide");
    let model = write_model(&tmp, "windowed.ir.json", &with_window_columns(&seed_timing_ir()));
    let wide = tmp.join("wide.tsv");
    let mut args = vec![
        "simulate", model.to_str().unwrap(), "--backend", "chain_binomial", "--dt", "1",
        "--seed", "3", "--obs-only", wide.to_str().unwrap(),
    ];
    args.extend_from_slice(PARAMS);
    let out = run(&camdl, &args);
    assert!(!out.status.success(), "a single wide file cannot carry two boundaries");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("window_start") && err.contains("--obs-dir"), "{err}");
}
