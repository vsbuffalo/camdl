//! gh#204 PR2 slice 5: `reactive_log.tsv` is a declared CAS artifact.
//!
//! A run with an active reactive policy writes its firing log into the run's
//! Sim leaf, declared in the `run.json` exact-set (never an optional-on-cache-
//! hit artifact), and `--reactive-log PATH` mirrors that canonical file. A run
//! with no active reactive policy writes no log artifact — the `Option` on the
//! trajectory makes "active, never crossed" and "no policy" distinct.
//!
//! Uses the COMMITTED compiler-generated golden with params on the CLI.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn golden_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json")
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

const PARAMS: &[&str] = &[
    "--param", "beta=0.3",
    "--param", "gamma=0.1",
    "--param", "rho=0.2",
    "--param", "trigger_threshold=2",
    "--param", "sia_cov=0.7",
    "--param", "N0=1000",
    "--param", "I0=10",
];

/// Recursively collect files named `name` under `root`.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()) == Some(name) {
                out.push(p);
            }
        }
    }
    out
}

struct Run {
    // Both tempdirs are kept alive for the lifetime of the assertions — drop
    // deletes them, and `mirror` lives under `_tmp`.
    _tmp: tempfile::TempDir,
    cas: tempfile::TempDir,
    mirror: PathBuf,
    out: std::process::Output,
}

fn run(extra: &[&str]) -> Run {
    let bin = skip_if_missing_binary();
    let ir = golden_ir();
    let tmp = tempfile::tempdir().unwrap();
    let traj = tmp.path().join("traj.tsv");
    let mirror = tmp.path().join("rx_mirror.tsv");
    let cas = tempfile::tempdir().unwrap();
    let mut args: Vec<String> = vec!["simulate".into(), ir.to_string_lossy().into_owned()];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.extend(PARAMS.iter().map(|s| s.to_string()));
    args.extend(
        ["--seed", "1", "--backend", "chain_binomial", "--dt", "1.0",
         "-o", &traj.to_string_lossy(),
         "--reactive-log", &mirror.to_string_lossy(),
         "--output-dir", &cas.path().to_string_lossy()]
            .iter()
            .map(|s| s.to_string()),
    );
    let out = Command::new(&bin).args(&args).output().expect("spawn camdl");
    Run { _tmp: tmp, cas, mirror, out }
}

/// Active reactive policy → the log is a declared leaf artifact, recorded in
/// `run.json`'s exact-set, and `--reactive-log` mirrors it byte-for-byte.
#[test]
fn reactive_log_is_a_declared_artifact_and_mirrors() {
    let _ = skip_if_missing_binary();
    let r = run(&["--enable", "sia"]);
    assert!(
        r.out.status.success(),
        "active reactive run must succeed; stderr={}",
        String::from_utf8_lossy(&r.out.stderr)
    );

    // The leaf carries the log file...
    let leaf_logs = find_all(r.cas.path(), "reactive_log.tsv");
    assert_eq!(leaf_logs.len(), 1, "exactly one Sim leaf holds reactive_log.tsv");
    let leaf_bytes = std::fs::read(&leaf_logs[0]).unwrap();

    // ...and it is DECLARED in the run.json exact-set (not optional-on-cache-hit).
    let run_jsons = find_all(r.cas.path(), "run.json");
    assert!(
        run_jsons.iter().any(|p| {
            std::fs::read_to_string(p)
                .map(|s| s.contains("reactive_log.tsv"))
                .unwrap_or(false)
        }),
        "run.json must declare reactive_log.tsv in its artifacts map"
    );

    // The mirror is a byte-for-byte copy of the canonical leaf file.
    let mirror_bytes = std::fs::read(&r.mirror).expect("--reactive-log mirror written");
    assert_eq!(mirror_bytes, leaf_bytes, "the mirror must equal the leaf artifact");

    // The header is the proposal's 6-column schema, and the policy fired.
    let text = String::from_utf8(leaf_bytes).unwrap();
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap(),
        "trigger_time\tpolicy\ttrigger_value\tthreshold\tfire_time\taction"
    );
    let row = lines.next().expect("the SIR golden crosses the threshold and fires once");
    let cols: Vec<&str> = row.split('\t').collect();
    assert_eq!(cols.len(), 6, "a firing row has 6 columns");
    assert_eq!(cols[1], "sia", "the policy name");
    assert_eq!(cols[5], "transfer", "the action verb");
    // fire_time = trigger_time + 21 (the `after` lag).
    let trig: f64 = cols[0].parse().unwrap();
    let fire: f64 = cols[4].parse().unwrap();
    assert_eq!(fire, trig + 21.0, "fire_time lags the trigger by `after=21`");
}

/// No active reactive policy → no log artifact at all (the absence is exact:
/// `run.json` does not declare it, and there is no loose file).
#[test]
fn non_reactive_run_writes_no_reactive_log() {
    let _ = skip_if_missing_binary();
    let r = run(&[]); // no --enable: the toggleable reactive policy is dropped
    assert!(
        r.out.status.success(),
        "a run that does not enable the policy must succeed; stderr={}",
        String::from_utf8_lossy(&r.out.stderr)
    );
    assert!(
        find_all(r.cas.path(), "reactive_log.tsv").is_empty(),
        "a non-reactive run declares no reactive_log.tsv artifact"
    );
    let run_jsons = find_all(r.cas.path(), "run.json");
    assert!(!run_jsons.is_empty(), "the run still wrote a Sim leaf");
    assert!(
        run_jsons.iter().all(|p| {
            std::fs::read_to_string(p)
                .map(|s| !s.contains("reactive_log.tsv"))
                .unwrap_or(true)
        }),
        "no run.json may declare reactive_log.tsv for a non-reactive run"
    );
    // `--reactive-log` on a non-reactive run has nothing to mirror; the flag
    // must not fabricate an empty file.
    assert!(!r.mirror.exists(), "no mirror file when there is no leaf log");
}
