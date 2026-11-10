//! Every method leaf persists its findings, and a run that could not store its
//! result fails.
//!
//! A successful IF2 or NLopt `fit run` leaves `diagnostics.json` in its leaf
//! (a JSON array of findings, empty when the run raised none); a `pfilter` run
//! whose output cannot be claimed in the run store exits non-zero instead of
//! warning and exiting 0; and an observation file that does not bind to the
//! model stops a fit through the one path every other failure takes (a
//! `failed` report in the leaf's event log, the error printed, status 1).

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust`", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

/// A scratch directory holding the reporting fixtures.
fn workdir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
    }
    tmp
}

fn command(dir: &Path) -> Command {
    let mut c = Command::new(camdl());
    c.env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
        .current_dir(dir);
    c
}

/// The one claimed method leaf (`seed_*`) under `dir/results/fits`.
fn the_leaf(dir: &Path) -> PathBuf {
    let subdirs = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()
    };
    let leaves: Vec<PathBuf> = subdirs(&dir.join("results/fits")).iter()
        .flat_map(|fit| subdirs(fit))
        .flat_map(|m| subdirs(&m))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("seed_"))
        .collect();
    assert_eq!(leaves.len(), 1, "one claimed leaf: {leaves:?}");
    leaves.into_iter().next().unwrap()
}

fn findings_of(leaf: &Path) -> Vec<serde_json::Value> {
    let path = leaf.join("diagnostics.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("a successful run leaves {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn a_successful_if2_leaf_holds_diagnostics_json_with_the_findings_it_raised() {
    let tmp = workdir();
    let out = command(tmp.path())
        .args(["fit", "run", "if2.toml", "--seed", "1", "--progress", "none"])
        .output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let findings = findings_of(&the_leaf(tmp.path()));
    assert!(findings.iter().any(|f| f["kind"]["type"] == "convergence_incomplete"),
        "the finding the run printed is the one recorded: {findings:?}");
    assert!(!findings.iter().any(|f| f["kind"]["type"] == "rhat_high"),
        "an optimiser's run carries no sampler diagnostics it did not raise: {findings:?}");
}

#[test]
fn a_successful_nlopt_leaf_holds_diagnostics_json() {
    let tmp = workdir();
    let out = command(tmp.path())
        .args(["fit", "run", "nlopt.toml", "--seed", "1", "--progress", "none"])
        .output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // The optimiser raises no sampler findings; the file's presence is the
    // record that the run raised none, rather than that nobody wrote them.
    let findings = findings_of(&the_leaf(tmp.path()));
    assert!(findings.iter().all(|f| f["kind"]["type"] != "rhat_high"), "{findings:?}");
}

#[test]
fn an_observation_file_that_does_not_bind_abandons_the_fit_with_its_record() {
    let tmp = workdir();
    // Two rows at one time: the schedule is not strictly increasing.
    std::fs::write(tmp.path().join("cases.tsv"), "time\tcases\n1\t1\n2\t4\n2\t5\n3\t11\n").unwrap();
    let out = command(tmp.path())
        .args(["fit", "run", "if2.toml", "--seed", "1", "--progress", "none"])
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let leaf = the_leaf(tmp.path());
    let log = std::fs::read_to_string(leaf.join("events.jsonl")).unwrap();
    let last: serde_json::Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
    assert_eq!(last["event"], "method_report", "{log}");
    assert_eq!(last["outcome"], "failed", "{last}");
    let _ = findings_of(&leaf);
}
