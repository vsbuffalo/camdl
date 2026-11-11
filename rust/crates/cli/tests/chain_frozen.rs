//! gh#1032: a chain that takes no step after burn-in is reported as such.
//!
//! A PGAS fit whose NUTS warm-up is short can end adaptation on a step size at
//! which the sampler does not move: every retained draw of the chain is then
//! the parameter vector it stood at when warm-up ended. The run raises one
//! `chain_frozen` finding per such chain, in place of the chain's acceptance
//! and divergence findings; records the chains in `pgas_summary.json`
//! `frozen_chains`; and the verdict `fit summary` reads from the leaf is
//! `chains_frozen`, not converged. A run whose chains move raises none of it.
//!
//! The fixture is gh#1032's: the reporting SIR model, two chains, `burn_in =
//! 10` with 20 particles (frozen on both chains at seed 1), and `burn_in =
//! 100` with 50 particles (moving).

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

/// A work directory holding the reporting fixture's SIR model compiled to IR
/// and its case counts.
fn workdir(tag: &str) -> (Tmp, PathBuf) {
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_chain_frozen_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let fixtures = manifest().join("tests/fixtures/reporting");
    std::fs::copy(fixtures.join("cases.tsv"), work.join("cases.tsv")).unwrap();
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(fixtures.join("sir.camdl"))
        .output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(work.join("sir.ir.json"), &out.stdout).unwrap();
    (Tmp(work.clone()), work)
}

/// Run `camdl`, returning its stdout and stderr.
fn camdl(work: &Path, args: &[&str]) -> (String, String) {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let out = Command::new(&bin)
        .env_remove("CLICOLOR_FORCE")
        .current_dir(work)
        .args(args)
        .args(["--progress", "none"])
        .output()
        .unwrap();
    assert!(out.status.success(), "camdl {args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
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

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The findings of a leaf's `diagnostics.json` whose `kind.type` is `ty`.
fn findings_of(leaf: &Path, ty: &str) -> Vec<serde_json::Value> {
    json(&leaf.join("diagnostics.json")).as_array().unwrap().iter()
        .filter(|d| d["kind"]["type"] == ty)
        .cloned()
        .collect()
}

/// Run one PGAS fit of the reporting model at seed 1 and return its leaf and
/// the run's stderr.
fn pgas(work: &Path, stem: &str, particles: usize, sweeps: usize, burn_in: usize) -> (PathBuf, String) {
    std::fs::write(work.join(format!("{stem}.toml")), format!(r#"
[model]
camdl = "sir.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
beta  = {{ start = 0.5 }}
gamma = {{ start = 0.2 }}
[fixed]
rho = 0.5
N0  = 2000
[method]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 2
particles = {particles}
sweeps    = {sweeps}
burn_in   = {burn_in}
thin      = 1
n_trajectories = 4
"#)).unwrap();
    let (_, stderr) = camdl(work, &["fit", "run", &format!("{stem}.toml"), "--seed", "1"]);
    (leaf(work, stem), stderr)
}

#[test]
fn a_frozen_pgas_chain_is_its_own_finding_and_is_not_converged() {
    let (_guard, work) = workdir("frozen");
    let (leaf, stderr) = pgas(&work, "frozen", 20, 60, 10);

    // Premise: the draws of both chains are one vector each.
    let draws = std::fs::read_to_string(leaf.join("draws.tsv")).unwrap();
    for c in ["0", "1"] {
        let rows: std::collections::BTreeSet<Vec<&str>> = draws.lines().skip(1)
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|f| f[0] == c)
            .map(|f| f[2..].to_vec())
            .collect();
        assert_eq!(rows.len(), 1, "fixture premise: chain {c} never moved after burn-in");
    }

    let frozen = findings_of(&leaf, "chain_frozen");
    assert_eq!(frozen.len(), 2, "one finding per frozen chain:\n{frozen:#?}");
    for (c, f) in frozen.iter().enumerate() {
        assert_eq!(f["severity"], "error", "{f}");
        let k = &f["kind"];
        assert_eq!(k["chain_id"], c, "{f}");
        assert_eq!(k["n_draws"], 50, "{f}");
        assert_eq!(k["burn_in"], 10, "{f}");
        assert_eq!(k["n_steps"], 50, "{f}");
        assert_eq!(k["sampler"]["kernel"], "nuts", "{f}");
        let h = k["sampler"]["step_size"].as_f64().expect("the frozen step size is recorded");
        assert!(h.is_finite() && h > 0.0, "{f}");
        let n_div = k["sampler"]["n_divergent"].as_u64().unwrap();
        assert!(n_div > 0 && n_div <= 50, "{f}");
        let msg = f["message"].as_str().unwrap();
        assert!(msg.starts_with("took no step after burn-in: all 50 retained draws are one \
            parameter vector"), "{msg}");
        assert!(msg.contains(&format!("{n_div} of its 50 post-burn-in NUTS steps diverged"))
            || (n_div == 50 && msg.contains("Every one of its 50")), "{msg}");
        assert!(msg.contains("after burn_in = 10 sweeps"), "{msg}");
    }
    // Not reported twice: the chains' acceptance and divergence findings are
    // in the frozen ones.
    assert!(findings_of(&leaf, "acceptance_rate_unhealthy").is_empty());
    assert!(findings_of(&leaf, "divergent_transitions").is_empty());
    assert!(stderr.contains("chain 0: took no step after burn-in"), "{stderr}");

    let summary = json(&leaf.join("pgas_summary.json"));
    assert_eq!(summary["frozen_chains"], serde_json::json!([0, 1]), "{summary}");

    // The verdict every reader takes from the leaf.
    let leaf_s = leaf.to_str().unwrap();
    let (doc, _) = camdl(&work, &["fit", "summary", leaf_s, "--format", "json"]);
    let doc: serde_json::Value = serde_json::from_str(&doc).unwrap();
    let method = &doc["methods"][0];
    assert_eq!(method["verdict"], serde_json::json!({"kind": "chains_frozen", "chains": [0, 1]}),
        "{}", method["verdict"]);
    let (text, _) = camdl(&work, &["fit", "summary", leaf_s]);
    assert!(text.contains("NOT converged")
        && text.contains("chains 0 and 1 took no step after burn-in"), "{text}");

    // Excluding one frozen chain leaves the other: still not converged.
    let (doc, _) = camdl(&work, &["fit", "summary", leaf_s, "--format", "json", "--exclude-chains", "0"]);
    let doc: serde_json::Value = serde_json::from_str(&doc).unwrap();
    assert_eq!(doc["methods"][0]["verdict"]["kind"], "chains_frozen", "{doc}");
    assert_eq!(doc["methods"][0]["verdict"]["chains"], serde_json::json!([1]), "{doc}");
}

#[test]
fn a_moving_pgas_fit_raises_no_frozen_chain() {
    let (_guard, work) = workdir("moving");
    let (leaf, stderr) = pgas(&work, "moving", 50, 150, 100);
    assert!(findings_of(&leaf, "chain_frozen").is_empty(), "{stderr}");
    assert!(!stderr.contains("took no step"), "{stderr}");
    let summary = json(&leaf.join("pgas_summary.json"));
    assert_eq!(summary["frozen_chains"], serde_json::json!([]),
        "recorded as none frozen, not left unrecorded: {summary}");
    let (doc, _) = camdl(&work, &["fit", "summary", leaf.to_str().unwrap(), "--format", "json"]);
    let doc: serde_json::Value = serde_json::from_str(&doc).unwrap();
    assert_ne!(doc["methods"][0]["verdict"]["kind"], "chains_frozen", "{doc}");
}
