//! `camdl pfilter --sweep-file FILE` scores every row of a design file and
//! writes one row per design point (`docs/camdl-run-spec.md`, "Scoring a
//! design"). Each point is scored exactly as the single call at its values
//! would score it, on the same seeds, and stored as that call's `Pfilter`
//! leaf, so a row equals the single call, a stored point is reused, and the
//! table does not depend on the thread schedule.
//!
//! End-to-end via the release `camdl` binary, with `camdlc.exe` on PATH.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_exe() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

/// A two-parameter SIR. `gamma` reaches 60 so that a design can hold a point
/// at which every particle's `I` is 0 by the end of the series.
const MODEL: &str = r#"time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 5.0]
  gamma : rate in [0.01, 60.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
scenarios {
  slow_recovery { set = { gamma = 0.5 } }
}
"#;

const DATA: &str = "time\tcases\n2\t3\n4\t1\n6\t0\n8\t0\n10\t0\n";

/// `beta` is estimated and `gamma` is set by `[fixed]`.
const FIT: &str = r#"[model]
camdl = "model.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
beta = { bounds = [0.1, 3.0], start = 0.8 }

[fixed]
gamma = 1.5

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 200
"#;

/// Three design points, none of which collapses.
const DESIGN: &str = "beta\tgamma\n0.8\t1.5\n1.0\t1.2\n0.6\t1.8\n";

fn fixture(dir: &Path) {
    assert!(camdl_bin().exists(), "release camdl binary missing — run `make build-rust`");
    assert!(camdlc_exe().exists(), "camdlc.exe missing — run `make build-ocaml`");
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT).unwrap();
    std::fs::write(dir.join("design.tsv"), DESIGN).unwrap();
    let shim = dir.join("bin");
    std::fs::create_dir_all(&shim).unwrap();
    // `pfilter` compiles the `.camdl` with the `camdlc` first on PATH.
    std::os::unix::fs::symlink(camdlc_exe(), shim.join("camdlc")).unwrap();
}

/// `camdl pfilter model.camdl ARGS` in `dir`, storing under `dir/store`.
fn pfilter(dir: &Path, args: &[&str]) -> Output {
    let path = std::env::var("PATH").unwrap_or_default();
    Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("store"))
        .env("PATH", format!("{}:{path}", dir.join("bin").display()))
        .current_dir(dir)
        .arg("pfilter")
        .arg("model.camdl")
        .args(args)
        .output()
        .unwrap()
}

/// The text without ANSI escape sequences (the status lines are coloured).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() { break; }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn stderr(out: &Output) -> String {
    strip_ansi(&String::from_utf8_lossy(&out.stderr))
}

/// One row of the design table, by column name.
type Row = std::collections::BTreeMap<String, String>;

fn table(text: &str) -> Vec<Row> {
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("a header").split('\t').collect();
    lines
        .map(|l| {
            let cells: Vec<&str> = l.split('\t').collect();
            assert_eq!(cells.len(), header.len(), "row {l:?} against header {header:?}");
            header.iter().zip(cells).map(|(h, c)| (h.to_string(), c.to_string())).collect()
        })
        .collect()
}

/// The design run `ARGS --sweep-file design.tsv -o landscape.tsv` in `dir`,
/// which must succeed, and its table.
fn design(dir: &Path, args: &[&str]) -> (Vec<Row>, String) {
    let mut all = vec!["--sweep-file", "design.tsv", "-o", "landscape.tsv"];
    all.extend_from_slice(args);
    let out = pfilter(dir, &all);
    let err = stderr(&out);
    assert!(out.status.success(), "design run failed:\n{err}");
    let text = std::fs::read_to_string(dir.join("landscape.tsv")).unwrap();
    (table(&text), err)
}

/// What the single call stored: its `run_id`, and the `loglik` and
/// `loglik_sd` of its `loglik.toml`, as written.
struct Single {
    run_id: String,
    loglik: String,
    loglik_sd: Option<String>,
}

fn single(dir: &Path, args: &[&str]) -> Single {
    let out = pfilter(dir, args);
    let err = stderr(&out);
    assert!(out.status.success(), "single call failed:\n{err}");
    let run_id = err.lines()
        .find_map(|l| l.trim_start().strip_prefix("camdl cat "))
        .unwrap_or_else(|| panic!("no `camdl cat` hint:\n{err}"))
        .trim().to_string();
    let leaf = err.lines()
        .find_map(|l| l.trim_start().strip_prefix("stored "))
        .unwrap_or_else(|| panic!("no `stored` line:\n{err}"))
        .trim().to_string();
    let toml = std::fs::read_to_string(dir.join(leaf).join("loglik.toml")).unwrap();
    let field = |k: &str| toml.lines().find_map(|l| l.strip_prefix(k)).map(str::to_string);
    Single {
        run_id,
        loglik: field("loglik = ").expect("loglik"),
        loglik_sd: field("loglik_sd = "),
    }
}

const COMMON: [&str; 5] = ["--fit", "fit.toml", "--particles", "200", "--seed"];

/// Test 2: every row's `loglik`, `loglik_sd` and `run_id` equal the single
/// call at that row's values, for one replicate and for three.
#[test]
fn each_row_equals_the_single_call_at_its_values() {
    for replicates in ["1", "3"] {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        let mut args: Vec<&str> = COMMON.to_vec();
        args.extend(["1", "--replicates", replicates]);
        let (rows, _) = design(dir.path(), &args);
        assert_eq!(rows.len(), 3);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row["point_id"], i.to_string());
            let beta = format!("beta={}", row["sweep:beta"]);
            let gamma = format!("gamma={}", row["sweep:gamma"]);
            // A store of its own, so the call computes and stores rather than
            // finding the design's leaf.
            let own = tempfile::tempdir().unwrap();
            fixture(own.path());
            let mut one = args.clone();
            one.extend(["--param", &beta, "--param", &gamma]);
            let s = single(own.path(), &one);
            eprintln!(
                "replicates {replicates}, point {i} ({beta}, {gamma}): design loglik {} sd {:?} \
                 run_id {}; single loglik {} sd {:?} run_id {}",
                row["loglik"], row["loglik_sd"], row["run_id"], s.loglik, s.loglik_sd, s.run_id);
            let design_ll: f64 = row["loglik"].parse().unwrap();
            let single_ll: f64 = s.loglik.parse().unwrap();
            assert!(design_ll.is_finite(), "point {i} must score finite: {design_ll}");
            assert_eq!(design_ll.to_bits(), single_ll.to_bits(),
                "point {i}: the design row and the single call must score the same number");
            assert_eq!(row["run_id"], s.run_id, "point {i}: one stored leaf");
            match replicates {
                "1" => assert_eq!(row["loglik_sd"], "", "no spread for one replicate"),
                _ => assert_eq!(
                    row["loglik_sd"].parse::<f64>().unwrap().to_bits(),
                    s.loglik_sd.as_deref().expect("a spread").parse::<f64>().unwrap().to_bits(),
                    "point {i}: the spread over replicates"),
            }
            assert_eq!(row["n_failed"], "0");
            assert_eq!(row["failure"], "");
        }
    }
}

/// Test 3: a design run again finds every point stored; a design whose first
/// row was scored by a single call computes only the other rows.
#[test]
fn stored_points_are_reused() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let mut args: Vec<&str> = COMMON.to_vec();
    args.push("1");
    let (first, err) = design(dir.path(), &args);
    assert_eq!(err.matches("stored point ").count(), 3, "{err}");
    let (again, err) = design(dir.path(), &args);
    assert_eq!(err.matches("cached point ").count(), 3, "every point is found stored:\n{err}");
    assert_eq!(err.matches("stored point ").count(), 0, "{err}");
    assert_eq!(first, again, "the rerun's table is the first run's");

    let fresh = tempfile::tempdir().unwrap();
    fixture(fresh.path());
    let mut one = args.clone();
    one.extend(["--param", "beta=0.8", "--param", "gamma=1.5"]);
    single(fresh.path(), &one);
    let (rows, err) = design(fresh.path(), &args);
    assert_eq!(err.matches("cached point 0 ").count(), 1, "point 0 was stored by the single call:\n{err}");
    assert_eq!(err.matches("stored point ").count(), 2, "only points 1 and 2 are computed:\n{err}");
    assert_eq!(rows, first);
}

/// Test 4: the table does not depend on the thread schedule.
#[test]
fn the_table_is_byte_identical_across_thread_counts() {
    let mut tables = Vec::new();
    for parallel in ["1", "4"] {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        let mut args: Vec<&str> = COMMON.to_vec();
        args.extend(["1", "--replicates", "2", "--parallel", parallel]);
        design(dir.path(), &args);
        tables.push(std::fs::read(dir.path().join("landscape.tsv")).unwrap());
    }
    assert_eq!(tables[0], tables[1], "--parallel 1 and --parallel 4 must write the same bytes");
}

/// The design run of `design` on observations `cases`, two replicates per
/// point: the exit status, the table (when one was written) and stderr.
fn failing(cases: &str, design: &str) -> (Option<i32>, Vec<Row>, String) {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    std::fs::write(dir.path().join("cases.tsv"), cases).unwrap();
    std::fs::write(dir.path().join("design.tsv"), design).unwrap();
    let mut args = vec!["--sweep-file", "design.tsv", "-o", "landscape.tsv"];
    args.extend(COMMON);
    args.extend(["1", "--replicates", "2"]);
    let out = pfilter(dir.path(), &args);
    let rows = std::fs::read_to_string(dir.path().join("landscape.tsv"))
        .map(|t| table(&t))
        .unwrap_or_default();
    (out.status.code(), rows, stderr(&out))
}

/// Point 1 failed in every replicate and points 0 and 2 were scored.
fn assert_point_1_failed(rows: &[Row], err: &str, cause: &str) {
    assert_eq!(rows.len(), 3, "{err}");
    let row = &rows[1];
    assert_eq!(row["loglik"], "-inf", "{err}");
    assert_eq!(row["loglik_sd"], "", "no spread over failed replicates");
    assert_eq!(row["n_failed"], "2");
    assert_eq!(row["run_id"], "", "a failed point writes no leaf");
    assert!(row["failure"].contains(cause), "the failure names its cause: {:?}", row["failure"]);
    assert!(!row["failure"].starts_with("error: "), "{:?}", row["failure"]);
    for i in [0, 2] {
        assert!(rows[i]["loglik"].parse::<f64>().unwrap().is_finite(), "point {i} is scored");
        assert!(rows[i]["loglik_sd"].parse::<f64>().unwrap().is_finite());
        assert_eq!(rows[i]["n_failed"], "0");
        assert_eq!(rows[i]["failure"], "");
        assert_eq!(rows[i]["run_id"].len(), 64);
    }
    assert!(err.contains("1 of 3 design points failed"), "{err}");
}

/// Test 5, first way of failing: every replicate scores `-inf`. At
/// `gamma = 50` every particle's `I` is 0 by the last observation, of 2
/// cases, which a Poisson with rate 0 scores `-inf`. The collapse falls on the
/// last observation, so the filter returns `-inf` rather than an error.
#[test]
fn a_point_that_scores_minus_inf_is_a_row() {
    let (code, rows, err) = failing(
        "time\tcases\n2\t3\n4\t1\n6\t0\n8\t0\n10\t2\n",
        "beta\tgamma\n1.0\t1.0\n0.1\t50\n1.1\t1.0\n",
    );
    assert_eq!(code, Some(0), "{err}");
    assert_point_1_failed(&rows, &err, "the log-likelihood is -inf at these parameters");
}

/// Test 5, second way of failing: the filter returns an error. With the
/// first observation at t = 6 the swarm at `gamma = 50` is empty for every
/// window, and the filter stops with `PFDegenerate { EssCollapsed }`.
#[test]
fn a_point_where_the_filter_stops_with_an_error_is_a_row() {
    let cases = "time\tcases\n6\t8\n8\t9\n10\t8\n";
    let (code, rows, err) = failing(cases, "beta\tgamma\n1.0\t1.0\n0.1\t50\n1.2\t1.0\n");
    assert_eq!(code, Some(0), "{err}");
    assert_point_1_failed(&rows, &err, "EssCollapsed");
    // The message depends only on the point, so the table does too.
    let (_, again, _) = failing(cases, "beta\tgamma\n1.0\t1.0\n0.1\t50\n1.2\t1.0\n");
    assert_eq!(rows, again, "a rerun writes the same table");

    // Every point failed: the table is written and the run fails.
    let (code, rows, err) = failing(cases, "beta\tgamma\n0.1\t50\n0.12\t45\n");
    assert_eq!(code, Some(3), "every point failed:\n{err}");
    assert_eq!(rows.len(), 2, "the table is still written");
    assert!(rows.iter().all(|r| r["loglik"] == "-inf" && r["run_id"].is_empty() && r["n_failed"] == "2"));
    assert!(err.contains("every one of the 2 design points failed"), "{err}");
}

/// Test 6: the refusals beside `--sweep-file`.
#[test]
fn what_is_refused_beside_a_design() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let refused = |extra: &[&str], needle: &str| {
        let mut args = vec!["--sweep-file", "design.tsv", "--fit", "fit.toml", "--particles", "50"];
        args.extend_from_slice(extra);
        let out = pfilter(dir.path(), &args);
        let err = stderr(&out);
        assert_eq!(out.status.code(), Some(1), "{extra:?} must be refused:\n{err}");
        assert!(err.contains(needle), "{extra:?}: expected {needle:?} in:\n{err}");
        assert!(!dir.path().join("store").exists(), "{extra:?}: nothing may be scored");
    };
    refused(&["--param", "gamma=1.5"],
        "parameter `gamma` is both a column of --sweep-file design.tsv and pinned by `--param gamma=1.5`");
    refused(&["--scenario", "slow_recovery"],
        "--sweep-file design.tsv and --scenario slow_recovery: parameter `gamma` is controlled by both the sweep and scenario `slow_recovery`");
    for (flag, value) in [
        ("--trace", "trace.tsv"),
        ("--pf-health", "health.tsv"),
        ("--save-final-state", "final.tsv"),
        ("--save-paths", "paths.tsv"),
        ("--save-filtering", "filtering.tsv"),
        ("--save-filtered-streams", "streams.tsv"),
        ("--save-prequential", "preq"),
    ] {
        refused(&[flag, value], &format!("--sweep-file design.tsv cannot be combined with {flag}"));
        assert!(!dir.path().join(value).exists(), "{flag}: no file may be written");
    }
    // An `[estimate]` parameter the file has no column for has no value.
    std::fs::write(dir.path().join("design.tsv"), "gamma\n1.5\n1.2\n").unwrap();
    refused(&[], "parameter 'beta' has no value");
}

/// A design file's own `point_id` column names the rows of the table, in file
/// order.
#[test]
fn rows_are_written_under_the_files_point_ids() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    std::fs::write(
        dir.path().join("design.tsv"),
        "point_id\tbeta\tgamma\n4\t0.71\t1.43\n7\t1.12\t1.61\n9\t0.93\t1.27\n",
    ).unwrap();
    let mut args: Vec<&str> = COMMON.to_vec();
    args.push("1");
    let (rows, _) = design(dir.path(), &args);
    let ids: Vec<&str> = rows.iter().map(|r| r["point_id"].as_str()).collect();
    assert_eq!(ids, ["4", "7", "9"], "each row is written under its own id, in file order");
    assert_eq!(rows[2]["sweep:beta"], "0.93");
}

/// Test 7: `camdl design lhs` writes a design over the fit's `[estimate]`
/// parameter, and `pfilter --sweep-file` scores it with the fit's `[fixed]`
/// value for the other. Every design point is a row, under the design's id and
/// at the design's value, and a scored row equals the single call there.
#[test]
fn a_design_from_design_lhs_is_scored() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let path = std::env::var("PATH").unwrap_or_default();
    let out = Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("PATH", format!("{}:{path}", dir.path().join("bin").display()))
        .current_dir(dir.path())
        .args(["design", "lhs", "model.camdl", "--fit", "fit.toml", "--n", "6", "--seed", "1",
               "-o", "design.tsv"])
        .output()
        .unwrap();
    assert!(out.status.success(), "design lhs failed:\n{}", stderr(&out));
    let lhs = table(&std::fs::read_to_string(dir.path().join("design.tsv")).unwrap());
    assert_eq!(lhs.len(), 6);
    assert!(lhs.iter().all(|r| r.len() == 2 && r.contains_key("beta")),
        "the design varies the one [estimate] parameter: {lhs:?}");

    let mut args: Vec<&str> = COMMON.to_vec();
    args.extend(["1", "--replicates", "2"]);
    let (rows, err) = design(dir.path(), &args);
    assert_eq!(rows.len(), 6, "{err}");
    for (row, point) in rows.iter().zip(&lhs) {
        assert_eq!(row["point_id"], point["point_id"]);
        assert_eq!(row["sweep:beta"], point["beta"], "the value scored is the design's");
        match row["n_failed"].as_str() {
            "0" => assert_eq!(row["run_id"].len(), 64, "{row:?}"),
            _ => assert!(!row["failure"].is_empty(), "{row:?}"),
        }
    }
    let scored = rows.iter().find(|r| r["n_failed"] == "0").expect("a scored point");
    let own = tempfile::tempdir().unwrap();
    fixture(own.path());
    let beta = format!("beta={}", scored["sweep:beta"]);
    let mut one = args.clone();
    one.extend(["--param", &beta]);
    let s = single(own.path(), &one);
    assert_eq!(scored["loglik"].parse::<f64>().unwrap().to_bits(), s.loglik.parse::<f64>().unwrap().to_bits());
    assert_eq!(scored["run_id"], s.run_id);
}
