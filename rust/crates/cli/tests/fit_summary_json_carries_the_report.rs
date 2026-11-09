//! `fit summary --format json` carries the report for every method.
//!
//! Each method's entry holds the verdict camdl decides (a consumer reads it
//! rather than recomputing it from R̂), the checks the run made before
//! sampling, the per-chain warm-up and the findings `diagnostics.json`
//! records — and every method has an entry: an NLopt or pfilter fit used to
//! fall through `build_summary_doc`'s match and be omitted.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust` (gh#105)", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fit_dir(work: &Path, fit: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(work.join("results/fits")).unwrap().flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(&format!("{fit}-")))
        .collect();
    assert_eq!(hits.len(), 1, "{hits:?}");
    hits.into_iter().next().unwrap()
}

#[test]
fn every_method_has_a_report_with_its_verdict_checks_warmup_and_findings() {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_summary_json_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for e in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        std::fs::copy(e.path(), work.join(e.file_name())).unwrap();
    }
    let run = |args: &[&str]| -> String {
        let out = Command::new(camdl())
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc())
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let summary = |fit: &str| -> serde_json::Value {
        let dir = fit_dir(&work, fit);
        let json = run(&["fit", "summary", dir.to_str().unwrap(), "--format", "json"]);
        let doc: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(doc["schema"]["version"], 4);
        assert_eq!(doc["methods"].as_array().unwrap().len(), 1, "{fit}: one method report:\n{json}");
        doc["methods"][0].clone()
    };

    run(&["fit", "run", "pmmh_stuck.toml", "--seed", "8", "--progress", "none"]);
    let m = summary("pmmh_stuck");
    assert_eq!(m["verdict"]["kind"], "not_converged");
    assert_eq!(m["verdict"]["band"], "severe");
    assert_eq!(m["verdict"]["param"], "beta");
    let kinds: Vec<&str> = m["checks"].as_array().unwrap().iter().map(|c| c["kind"].as_str().unwrap()).collect();
    // In the order the run shows them: PMMH says its caveat after the starts
    // note, once its data are bound.
    assert_eq!(kinds, ["starts", "method_caveat", "filter_noise", "obs_stream"]);
    assert_eq!(m["warmup"].as_array().unwrap().len(), 4);
    let floors: Vec<u64> = m["findings"].as_array().unwrap().iter()
        .filter(|f| f["kind"]["type"] == "scale_at_floor")
        .map(|f| f["kind"]["chain_id"].as_u64().unwrap())
        .collect();
    assert_eq!(floors, [1, 2], "the two stuck chains' floor findings, by chain");
    assert_eq!(m["header"]["config"], "pmmh_stuck.toml");

    run(&["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    assert_eq!(summary("pmmh_converged")["verdict"]["kind"], "converged");

    run(&["fit", "run", "if2.toml", "--seed", "1", "--progress", "none"]);
    let m = summary("if2");
    assert_eq!(m["verdict"]["kind"], "not_applicable");
    assert!(m["findings"].as_array().unwrap().iter().any(|f| f["kind"]["type"] == "convergence_incomplete"),
        "an IF2 run records the findings it printed: {}", m["findings"]);

    run(&["fit", "run", "nlopt.toml", "--seed", "1", "--progress", "none"]);
    let m = summary("nlopt");
    assert_eq!(m["method"], "nl-sbplx");
    assert_eq!(m["verdict"]["kind"], "not_applicable");
    assert!(m["best_loglik"].is_number());

    run(&["fit", "run", "pfilter.toml", "--seed", "1", "--progress", "none"]);
    let m = summary("pfilter");
    assert_eq!(m["method"], "pfilter");
    assert_eq!(m["method_result"]["n_replicates"], 3);
    assert_eq!(m["best_loglik"], m["method_result"]["ll_mean"]);
}
