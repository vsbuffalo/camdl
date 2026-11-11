//! The per-chain log-likelihood comparison (gh#406, gh#1008) is computed once,
//! at the end of the run, and recorded in the leaf; every reader takes the
//! recorded one.
//!
//! A sampler records it in its summary under `chain_loglik_agreement`; it
//! reaches `fit summary --format json` (and so `events.jsonl`'s
//! `method_report`) as `method_result.diagnostics.chain_logliks`, and the text
//! table prints its numbers. A leaf whose summary does not record it says so
//! and is not recomputed. IF2 records its clean-evaluation comparison in
//! `fit_state.toml`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn camdl(work: &Path, args: &[&str]) -> String {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let out = Command::new(&bin)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .current_dir(work)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "camdl {args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The single method leaf under `results/fits/<stem>-*/`.
fn leaf(work: &Path, stem: &str) -> PathBuf {
    let only = |dir: &Path, prefix: &str| -> PathBuf {
        let hits: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().starts_with(prefix))
            .collect();
        assert_eq!(hits.len(), 1, "one `{prefix}*` under {}: {hits:?}", dir.display());
        hits.into_iter().next().unwrap()
    };
    let fit = only(&work.join("results/fits"), &format!("{stem}-"));
    let method = only(&fit, "");
    only(&method, "seed_")
}

fn workdir(tag: &str) -> (Tmp, PathBuf) {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_agreement_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    for e in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::copy(e.path(), work.join(e.file_name())).unwrap();
        }
    }
    (Tmp(work.clone()), work)
}

#[test]
fn a_sampler_records_its_chain_comparison_and_the_summary_reads_it() {
    let (_guard, work) = workdir("pmmh");
    camdl(&work, &["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    let leaf = leaf(&work, "pmmh_converged");
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("pmmh_summary.json")).unwrap()).unwrap();
    let recorded = &summary["chain_loglik_agreement"];
    assert_eq!(recorded["column"], "log_likelihood", "{recorded}");
    assert_eq!(recorded["family_alpha"], 0.001);
    assert_eq!(recorded["min_ess"], 10.0);
    let chains = recorded["chains"].as_array().unwrap();
    assert_eq!(chains.len(), 4, "{recorded}");
    for (c, row) in chains.iter().enumerate() {
        assert_eq!(row["chain"], c, "{row}");
        assert!(row["mean"].is_number(), "{row}");
        assert!(matches!(row["outcome"]["status"].as_str(), Some("assessed" | "not_assessed")), "{row}");
    }

    // The JSON report carries the recorded comparison as it was written, and
    // the run's own end-of-run report (events.jsonl) does too.
    let leaf_s = leaf.to_str().unwrap();
    let doc: serde_json::Value =
        serde_json::from_str(&camdl(&work, &["fit", "summary", leaf_s, "--format", "json"])).unwrap();
    assert_eq!(&doc["methods"][0]["method_result"]["diagnostics"]["chain_logliks"], recorded);
    let events = std::fs::read_to_string(leaf.join("events.jsonl")).unwrap();
    let last: serde_json::Value = serde_json::from_str(events.lines().last().unwrap()).unwrap();
    assert_eq!(&last["report"]["method_result"]["diagnostics"]["chain_logliks"], recorded);

    // The text table prints the recorded Z of each assessed chain.
    let text = camdl(&work, &["fit", "summary", leaf_s]);
    for row in chains {
        if let Some(z) = row["outcome"]["z"].as_f64() {
            let cell = format!("{z:.2}");
            assert!(text.lines().any(|l| l.trim_start().starts_with(&row["chain"].to_string())
                && l.contains(&cell)), "chain {} Z {cell} in:\n{text}", row["chain"]);
        }
    }

    // A summary that does not record the comparison is said to, and nothing
    // is recomputed from the traces that are still there.
    let mut old = summary.clone();
    old.as_object_mut().unwrap().remove("chain_loglik_agreement");
    std::fs::write(leaf.join("pmmh_summary.json"), serde_json::to_vec_pretty(&old).unwrap()).unwrap();
    let text = camdl(&work, &["fit", "summary", leaf_s]);
    assert!(text.contains("chains' log-likelihood comparison not recorded"), "{text}");
    assert!(text.contains("(no per-chain log-likelihood comparison recorded in this leaf)"), "{text}");
    assert!(!text.contains("Δ others"), "no table is recomputed:\n{text}");
    let doc: serde_json::Value =
        serde_json::from_str(&camdl(&work, &["fit", "summary", leaf_s, "--format", "json"])).unwrap();
    assert!(doc["methods"][0]["method_result"]["diagnostics"]["chain_logliks"].is_null());
}

#[test]
fn if2_records_its_clean_evaluation_comparison_in_fit_state() {
    let (_guard, work) = workdir("if2");
    camdl(&work, &["fit", "run", "if2.toml", "--seed", "1", "--progress", "none"]);
    let leaf = leaf(&work, "if2");
    let state: toml::Value =
        toml::from_str(&std::fs::read_to_string(leaf.join("fit_state.toml")).unwrap()).unwrap();
    let a = &state["chain_loglik_agreement"];
    assert_eq!(a["column"].as_str(), Some("clean_loglik"), "{a}");
    let ids: Vec<i64> = state["chain_eval_ids"].as_array().unwrap().iter()
        .map(|v| v.as_integer().unwrap()).collect();
    let rows: Vec<i64> = a["chains"].as_array().unwrap().iter()
        .map(|r| r["chain"].as_integer().unwrap()).collect();
    assert_eq!(rows, ids, "one row per evaluated chain, by its number");
    let logliks: Vec<f64> = state["chain_eval_logliks"].as_array().unwrap().iter()
        .map(|v| v.as_float().unwrap()).collect();
    let means: Vec<f64> = a["chains"].as_array().unwrap().iter()
        .map(|r| r["mean"].as_float().unwrap()).collect();
    assert_eq!(means, logliks, "each row's mean is that chain's clean-eval loglik");
    // Recorded, not raised: an IF2 spread is judged by the source gate's
    // decibans leg, whose floor is in log-likelihood units.
    let diags = std::fs::read_to_string(leaf.join("diagnostics.json")).unwrap();
    assert!(!diags.contains("chain_loglik_differs"), "{diags}");
}
