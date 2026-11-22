//! A fit run's stored bytes are a function of its computation.
//!
//! The store is input-addressed: a run folder's address is computed from the
//! run's inputs, and the folder holds the result. Byte reproducibility — two executions
//! of one computation writing exactly the same bytes — is the strongest of the
//! store's three reproducibility claims, and camdl promises it only where it is
//! tested. This file is that test, for one method of each kind: PMMH and PGAS
//! (samplers), IF2 (iterated filtering) and `nl-sbplx` (an optimiser).
//!
//! Each method runs twice, uninterrupted, in two fresh working directories with
//! fresh stores, and the two runs' manifests — every file the run's `run.json`
//! checksums, by path, length and SHA-256 — must be equal, for the run folder
//! and every result folder inside it. Separately, each method runs once
//! uninterrupted and once killed with SIGKILL mid-method and then re-run, which
//! reclaims the dead attempt's folder and recomputes it; the two manifests must
//! again be equal.
//!
//! What may differ is everything about an execution rather than its result —
//! when it ran, for how long, under which process. It lives in the attempt's
//! records (`progress.json`, `events.jsonl`, which the manifest lists without a
//! checksum) and in two provenance files the manifest checksums but which
//! record the execution itself: `execution.json`, and an optimiser's
//! `opt_trace.tsv`, whose rows from parallel chains are in the order the
//! chains' evaluations completed. Both are left out of the comparison, and
//! only while their `run.json` entry gives them the role `provenance`. A
//! manifest also records each file's mtime, for the store's cheap integrity
//! check; that is a property of the write, not of the bytes, and is not
//! compared.
//!
//! An equal manifest on these fixtures is regression evidence, not proof: a
//! different model could reach a parallel reduction or a scheduling path these
//! do not. Fixture: `tests/fixtures/reporting/sir.camdl` with `cases.tsv`.

mod fit_store_support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let p = manifest_dir().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust`", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = manifest_dir().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

struct Work(PathBuf);
impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fresh working directory holding the model, its data and `fit.toml`; the
/// store is `results/` inside it. Every work directory has the same depth, so
/// nothing a leaf records relative to its store differs between two of them.
fn work(tag: &str, fit_toml: &str) -> Work {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("camdl_byte_repro_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["sir.camdl", "cases.tsv"] {
        std::fs::copy(manifest_dir().join("tests/fixtures/reporting").join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fit.toml"), fit_toml).unwrap();
    Work(dir)
}

/// The store every run here writes to, relative to its working directory.
const STORE: &str = "results";

fn fit_command(dir: &Path, store: &str) -> Command {
    let mut c = Command::new(camdl());
    c.env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_STORE", store)
        .current_dir(dir)
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"]);
    c
}

fn run_fit(dir: &Path) {
    run_fit_into(dir, STORE);
}

fn run_fit_into(dir: &Path, store: &str) {
    let out = fit_command(dir, store).output().unwrap();
    assert!(out.status.success(), "camdl fit run in {}: {}", dir.display(), String::from_utf8_lossy(&out.stderr));
}

/// The fit run folder in the store `dir/results`: the one result folder of
/// kind `fit_run`.
fn method_leaf(dir: &Path) -> PathBuf {
    method_leaf_in(dir, STORE)
}

fn method_leaf_in(dir: &Path, store: &str) -> PathBuf {
    fit_store_support::the_run(&dir.join(store))
}

fn walk(dir: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.file_name().is_some_and(|n| n == ".attempts") {
            continue;
        }
        if p.is_dir() {
            walk(&p, f);
        } else {
            f(&p);
        }
    }
}

/// Every result folder at or under the run folder `leaf` (the run and any
/// result folder inside it), by its path relative to `leaf`: its run id,
/// status, and the files its `run.json` checksums other than the execution
/// records [`EXECUTION_RECORDS`], as `file -> (bytes, sha256)`.
/// The provenance files that record an execution rather than its result, and
/// so may differ between two executions of one computation.
const EXECUTION_RECORDS: [&str; 2] = ["execution.json", "opt_trace.tsv"];

type Leaves = BTreeMap<String, (String, String, BTreeMap<String, (u64, String)>)>;

fn leaves(leaf: &Path) -> Leaves {
    let mut out = Leaves::new();
    walk(leaf, &mut |p| {
        if p.file_name().is_none_or(|n| n != "run.json") {
            return;
        }
        let rec: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let folder = p.parent().unwrap();
        let files = rec["files"].as_object().unwrap().iter()
            .filter(|(k, v)| v.get("sha256").is_some()
                && !(EXECUTION_RECORDS.contains(&k.as_str()) && v["role"] == "provenance"))
            .map(|(k, v)| (k.clone(), (v["bytes"].as_u64().unwrap(), v["sha256"].as_str().unwrap().to_string())))
            .collect();
        let id = fit_store_support::json(&folder.join("identity.json"))["run_id"].as_str().unwrap().to_string();
        let rel = folder.strip_prefix(leaf).unwrap().display().to_string();
        out.insert(rel, (id, rec["status"].as_str().unwrap().to_string(), files));
    });
    out
}

/// The files whose recorded digests differ between two sets of leaves, for a
/// failure message that names them.
fn differing(a: &Leaves, b: &Leaves) -> Vec<String> {
    let mut d = Vec::new();
    for (rel, (ra, sa, fa)) in a {
        let Some((rb, sb, fb)) = b.get(rel) else { d.push(format!("{rel}: leaf missing from the second run")); continue };
        if ra != rb { d.push(format!("{rel}: run_id {ra} vs {rb}")); }
        if sa != sb { d.push(format!("{rel}: status {sa} vs {sb}")); }
        for k in fa.keys().chain(fb.keys()).collect::<std::collections::BTreeSet<_>>() {
            if fa.get(k) != fb.get(k) {
                d.push(format!("{rel}/{k}"));
            }
        }
    }
    d.extend(b.keys().filter(|k| !a.contains_key(*k)).map(|k| format!("{k}: leaf missing from the first run")));
    d
}

/// Run `fit_toml` twice, uninterrupted, in two fresh stores, a second apart
/// (so a whole-second timestamp in a checksummed file shows), and return both
/// leaf sets.
fn twice_uninterrupted(tag: &str, fit_toml: &str) -> (Leaves, Leaves) {
    let a = work(&format!("{tag}_a"), fit_toml);
    run_fit(&a.0);
    std::thread::sleep(Duration::from_millis(1100));
    let b = work(&format!("{tag}_b"), fit_toml);
    run_fit(&b.0);
    assert_no_build_date(tag, &method_leaf(&a.0));
    (leaves(&method_leaf(&a.0)), leaves(&method_leaf(&b.0)))
}

/// The build date `camdl --version` prints, `(2026-11-17)`. A rebuild of the
/// same source with the same engine identity has another, so no stored file
/// may carry it.
fn build_date() -> String {
    let out = Command::new(camdl()).arg("--version").output().unwrap();
    let v = String::from_utf8_lossy(&out.stdout).into_owned();
    let open = v.find('(').unwrap_or_else(|| panic!("no build date in `camdl --version`: {v}"));
    v[open..=v.find(')').unwrap()].to_string()
}

/// No manifested file of the leaf, or of a child leaf, holds the build date.
fn assert_no_build_date(tag: &str, leaf: &Path) {
    let date = build_date();
    let mut hits = Vec::new();
    for (rel, (_, _, files)) in leaves(leaf) {
        for f in files.keys() {
            let p = leaf.join(&rel).join(f);
            if String::from_utf8_lossy(&std::fs::read(&p).unwrap()).contains(&date) {
                hits.push(format!("{rel}/{f}"));
            }
        }
    }
    assert!(hits.is_empty(), "{tag}: stored files carry the build date {date}: {hits:?}");
}

fn assert_same(tag: &str, what: &str, a: &Leaves, b: &Leaves) {
    assert!(!a.is_empty(), "{tag}: no leaf found");
    assert!(a.values().all(|(_, s, _)| s == "completed"), "{tag}: a leaf is not completed: {a:?}");
    let d = differing(a, b);
    assert!(d.is_empty(), "{tag}: {what} wrote different stored bytes in: {d:#?}");
}

const HEAD: &str = r#"[model]
camdl = "sir.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
beta  = { start = 0.5 }
gamma = { start = 0.2 }

[fixed]
rho = 0.5
N0  = 2000
"#;

fn pmmh(iterations: usize) -> String {
    format!("{HEAD}\n[method]\nalgorithm  = \"pmmh\"\nbackend    = \"chain_binomial\"\nchains     = 2\n\
             particles  = 50\niterations = {iterations}\nburn_in    = 100\nthin       = 10\n")
}

fn pgas(sweeps: usize) -> String {
    format!("{HEAD}\n[method]\nalgorithm = \"pgas\"\nbackend   = \"chain_binomial\"\nchains    = 2\n\
             particles = 50\nsweeps    = {sweeps}\nburn_in   = 100\n")
}

fn if2(iterations: usize, particles: usize) -> String {
    format!("{HEAD}\n[method]\nalgorithm  = \"if2\"\nbackend    = \"chain_binomial\"\nchains     = 2\n\
             particles  = {particles}\niterations = {iterations}\ncooling    = 0.9\n")
}

/// `dt` sets the ODE step, and so how long the search takes.
fn nl_sbplx(dt: f64) -> String {
    format!("{HEAD}\n[method]\nalgorithm = \"nl-sbplx\"\nbackend   = \"ode\"\nchains    = 2\n\n[config]\ndt = {dt}\n")
}

#[test]
fn pmmh_twice_uninterrupted_writes_the_same_bytes() {
    let (a, b) = twice_uninterrupted("pmmh", &pmmh(300));
    assert_same("pmmh", "two uninterrupted runs", &a, &b);
}

#[test]
fn pgas_twice_uninterrupted_writes_the_same_bytes() {
    let (a, b) = twice_uninterrupted("pgas", &pgas(200));
    assert_same("pgas", "two uninterrupted runs", &a, &b);
}

#[test]
fn if2_twice_uninterrupted_writes_the_same_bytes() {
    let (a, b) = twice_uninterrupted("if2", &if2(5, 50));
    assert_same("if2", "two uninterrupted runs", &a, &b);
}

#[test]
fn nl_sbplx_twice_uninterrupted_writes_the_same_bytes() {
    let (a, b) = twice_uninterrupted("nlsbplx", &nl_sbplx(1.0));
    assert_same("nl-sbplx", "two uninterrupted runs", &a, &b);
}

/// Where the store lives is not part of the computation: a run written into a
/// store two directories deeper, relative to the model and data it read, holds
/// the same bytes. A path recorded relative to the store would differ.
#[test]
fn a_leaf_holds_the_same_bytes_wherever_its_store_is() {
    let toml = if2(5, 50);
    let a = work("store_here", &toml);
    run_fit(&a.0);
    let b = work("store_deeper", &toml);
    run_fit_into(&b.0, "deeper/still/results");
    let (la, lb) = (leaves(&method_leaf(&a.0)), leaves(&method_leaf_in(&b.0, "deeper/still/results")));
    assert_same("if2", "a store beside the model and one deeper", &la, &lb);
}

/// Start the fit, wait until its method has begun writing into the run folder
/// (the event log's first line), let it run `into` longer, and SIGKILL it.
/// The run must still be `running`: the kill landed before the commit. Then
/// run the same command again, which finds the run claimed by a dead process,
/// preserves the dead attempt under `.attempts/` and recomputes.
fn killed_then_rerun(dir: &Path, into: Duration) {
    let mut child = fit_command(dir, STORE).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let began = loop {
        let mut log = None;
        walk(&dir.join(STORE), &mut |p| {
            if p.file_name().is_some_and(|n| n == "events.jsonl") {
                log = Some(p.to_path_buf());
            }
        });
        if let Some(l) = log.filter(|l| std::fs::metadata(l).is_ok_and(|m| m.len() > 0)) {
            break l;
        }
        assert!(child.try_wait().unwrap().is_none(), "the fit exited before its method began");
        assert!(Instant::now() < deadline, "the fit's method never began");
        std::thread::sleep(Duration::from_millis(5));
    };
    std::thread::sleep(into);
    child.kill().unwrap();
    child.wait().unwrap();
    let rec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(began.parent().unwrap().join("run.json")).unwrap()).unwrap();
    assert_eq!(rec["status"], "running",
        "the fit finished before it was killed, so it was not interrupted: enlarge the fixture");
    run_fit(dir);
    let leaf = method_leaf(dir);
    assert!(leaf.join(".attempts/1").is_dir(), "the re-run did not reclaim the killed attempt");
}

fn uninterrupted_vs_killed_and_rerun(tag: &str, fit_toml: &str, into: Duration) -> (Leaves, Leaves) {
    let a = work(&format!("{tag}_whole"), fit_toml);
    run_fit(&a.0);
    let b = work(&format!("{tag}_killed"), fit_toml);
    killed_then_rerun(&b.0, into);
    (leaves(&method_leaf(&a.0)), leaves(&method_leaf(&b.0)))
}

#[test]
fn pmmh_killed_and_rerun_writes_the_uninterrupted_bytes() {
    let (a, b) = uninterrupted_vs_killed_and_rerun("pmmh", &pmmh(2000), Duration::from_millis(500));
    assert_same("pmmh", "an uninterrupted and a killed-and-rerun run", &a, &b);
}

#[test]
fn pgas_killed_and_rerun_writes_the_uninterrupted_bytes() {
    let (a, b) = uninterrupted_vs_killed_and_rerun("pgas", &pgas(1000), Duration::from_millis(500));
    assert_same("pgas", "an uninterrupted and a killed-and-rerun run", &a, &b);
}

#[test]
fn if2_killed_and_rerun_writes_the_uninterrupted_bytes() {
    let (a, b) = uninterrupted_vs_killed_and_rerun("if2", &if2(150, 400), Duration::from_millis(300));
    assert_same("if2", "an uninterrupted and a killed-and-rerun run", &a, &b);
}

#[test]
fn nl_sbplx_killed_and_rerun_writes_the_uninterrupted_bytes() {
    let (a, b) = uninterrupted_vs_killed_and_rerun("nlsbplx", &nl_sbplx(0.002), Duration::from_millis(500));
    assert_same("nl-sbplx", "an uninterrupted and a killed-and-rerun run", &a, &b);
}


/// A damaged execution record is said, never read silently as no record
/// (which would report the run's wall time as unknown without saying why).
/// The record is one of the files the run's `run.json` checksums, so a record
/// that does not parse is named whether the command warns or refuses.
#[test]
fn an_unreadable_execution_record_is_named() {
    let w = work("bad_execution", &pmmh(150));
    run_fit(&w.0);
    let leaf = method_leaf(&w.0);
    std::fs::write(leaf.join("execution.json"), b"{ not json").unwrap();
    let out = Command::new(camdl())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_STORE", w.0.join(STORE))
        .current_dir(&w.0)
        .args(["fit", "summary", &leaf.to_string_lossy()])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("execution.json"),
        "an unparseable execution record was not named:\n{stderr}");
    if out.status.success() {
        assert!(stderr.contains("warning:"), "read without a warning:\n{stderr}");
    }
}
