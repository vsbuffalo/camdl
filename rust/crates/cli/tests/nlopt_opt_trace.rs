//! `opt_trace.tsv`, end to end through the built `camdl` binary: an NLopt
//! multistart fit leaves one trace beside `chain_results.tsv`, whose rows per
//! search climb monotonically to the result that search reported, and which
//! the heartbeat's per-search block agrees with; a search that never scores a
//! finite value traces an empty best; a profile, whose cells run the same
//! optimiser, writes no trace (`docs/camdl-run-spec.md` §10.13).

mod fit_store_support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  N0    : count        in [100, 100000]
  I0    : count        in [0, 1000]
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  I = I0
  S = N0 - I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 56 'days
}
"#;

/// Simulated from the model at beta = 0.4, gamma = 0.15, N0 = 10000, I0 = 10,
/// rho = 0.6, k = 10 (chain_binomial, dt = 1, seed 7).
const DATA: &str = "time\tweekly_cases\n7\t47\n14\t111\n21\t474\n28\t1066\n\
                    35\t1471\n42\t533\n49\t300\n56\t40\n";

/// An `nl-sbplx` fit of `beta` and `gamma` with `chains` starts drawn by
/// `starts`, the epidemic seeded with `i0` infecteds.
fn fit_toml(chains: usize, starts: &str, i0: u32) -> String {
    format!(
        r#"store = "results"

[model]
camdl = "sir.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0 = 10000
I0 = {i0}
rho = 0.6
k = 10.0

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = {chains}
starts = "{starts}"
"#
    )
}

fn workdir(fit: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), fit).unwrap();
    tmp
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

/// Every file named `name` under `root`, recursively.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == name) {
                found.push(p);
            }
        }
    }
    found
}

/// A TSV as its header and its rows, each row a list of fields.
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.ends_with('\n'), "{}: every row is a whole line", path.display());
    let mut lines = text.lines().map(|l| l.split('\t').map(str::to_string).collect::<Vec<_>>());
    let header = lines.next().unwrap();
    (header, lines.collect())
}

/// The one fit run in the store `results` under `dir`.
fn the_leaf(dir: &Path) -> PathBuf {
    fit_store_support::the_run(&dir.join("results"))
}

/// Three searches from spread starts: the trace's columns are
/// `chain_results.tsv`'s, each search's best never decreases, and its last
/// row is the result `chain_results.tsv` reports for it — the same value, the
/// same point, at the same evaluation count. The heartbeat's per-search block
/// ends on the same counts, and the run's record files the trace as
/// provenance, a record of the attempt, not as part of the result.
#[test]
fn every_search_is_traced_up_to_the_result_it_reports() {
    let tmp = workdir(&fit_toml(3, "lhs", 10));
    let o = camdl(tmp.path(), &["fit", "run", "fit.toml", "--no-dt-check"]);
    assert!(o.status.success(), "fit run: {}", String::from_utf8_lossy(&o.stderr));
    let leaf = the_leaf(tmp.path());

    let (results_header, results) = read_tsv(&leaf.join("chain_results.tsv"));
    let (trace_header, trace) = read_tsv(&leaf.join("opt_trace.tsv"));
    assert_eq!(trace_header[..4], ["chain", "eval", "loglik", "best_loglik"]);
    assert_eq!(trace_header[4..], results_header[4..], "the parameter columns are chain_results.tsv's");
    assert_eq!(results.len(), 3);

    let mut by_chain: BTreeMap<String, Vec<&Vec<String>>> = BTreeMap::new();
    for row in &trace {
        assert_eq!(row.len(), trace_header.len(), "a whole row: {row:?}");
        by_chain.entry(row[0].clone()).or_default().push(row);
    }
    assert_eq!(by_chain.len(), 3, "one series per search: {:?}", by_chain.keys());

    for result in &results {
        let rows = &by_chain[&result[0]];
        assert_eq!(rows[0][1], "1", "chain {}: a row at the first evaluation", result[0]);
        let mut best = f64::NEG_INFINITY;
        let mut eval = 0u64;
        for r in rows {
            let e: u64 = r[1].parse().unwrap();
            assert!(e > eval, "chain {}: evaluations increase: {r:?}", result[0]);
            eval = e;
            let b: f64 = r[3].parse().unwrap();
            assert!(b >= best, "chain {}: best decreased: {r:?}", result[0]);
            best = b;
        }
        let last = rows.last().unwrap();
        assert_eq!(last[1], result[3], "chain {}: the last row is at n_evals", result[0]);
        assert_eq!(format!("{best:.6}"), result[1], "chain {}: the last best is the chain's loglik", result[0]);
        for (col, want) in last[4..].iter().zip(&result[4..]) {
            let v: f64 = col.parse().unwrap();
            assert_eq!(&format!("{v:.10}"), want, "chain {}: the best point is the chain's", result[0]);
        }
    }

    let progress: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("progress.json")).unwrap()).unwrap();
    assert_eq!(progress["snapshot"]["state"], "done");
    let searches = progress["snapshot"]["searches"].as_array().expect("a searches block");
    assert_eq!(searches.len(), 3);
    for (s, result) in searches.iter().zip(&results) {
        assert_eq!(s["chain"].to_string(), result[0]);
        assert_eq!(s["evals"].to_string(), result[3], "the heartbeat's count is the search's");
        let b = s["best_loglik"].as_f64().expect("a finite best");
        assert_eq!(format!("{b:.6}"), result[1]);
    }

    let record = fit_store_support::json(&leaf.join("run.json"));
    assert_eq!(record["files"]["opt_trace.tsv"]["role"], "provenance",
        "the trace is a record of the attempt, not part of the result: {}", record["files"]);
}

/// A search that never scores a finite value — the epidemic is seeded with no
/// infecteds, so every θ predicts zero cases against positive counts — traces
/// each evaluation's `-inf` beside an empty best and empty parameter fields,
/// and the heartbeat's best is `null`.
#[test]
fn a_search_with_no_finite_value_traces_an_empty_best() {
    let tmp = workdir(&fit_toml(1, "single", 0));
    let o = camdl(tmp.path(), &["fit", "run", "fit.toml", "--no-dt-check"]);
    assert!(!o.status.success(), "a fit with no finite value is refused (gh#916)");
    let leaf = the_leaf(tmp.path());
    let (header, rows) = read_tsv(&leaf.join("opt_trace.tsv"));
    assert!(rows.len() >= 2, "rows were written: {rows:?}");
    for r in &rows {
        assert_eq!(r.len(), header.len(), "a whole row: {r:?}");
        assert_eq!(r[2], "-inf", "{r:?}");
        assert!(r[3..].iter().all(String::is_empty), "no best and no point: {r:?}");
    }
    let progress: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("progress.json")).unwrap()).unwrap();
    assert_eq!(progress["snapshot"]["searches"][0]["best_loglik"], serde_json::Value::Null);
}

/// A profile's cells run the same NLopt search, and write no trace.
#[test]
fn a_profile_writes_no_opt_trace() {
    let tmp = workdir(&fit_toml(1, "single", 10));
    let o = camdl(tmp.path(), &["profile", "sir.camdl", "--fit", "fit.toml",
        "--sweep", "gamma=0.12,0.15", "--n-starts", "1", "--starts", "single",
        "--algorithm", "nl-sbplx", "--backend", "ode"]);
    assert!(o.status.success(), "profile: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(find_all(&tmp.path().join("results"), "mle.toml").len(), 2, "premise: two cells ran");
    assert_eq!(find_all(tmp.path(), "opt_trace.tsv"), Vec::<PathBuf>::new());
}
