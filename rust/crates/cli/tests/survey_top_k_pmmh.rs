//! Integration test for `init = "survey_top_k"` on a PMMH stage
//! (gh#51 v2). Mirrors `survey_top_k_pgas.rs`.
//!
//! Verifies that PMMH dispatch accepts `init = "survey_top_k"`
//! and that the top-K landscape rows reach per-chain starts plus the
//! `chain_starts.tsv` audit sidecar. v1 (the IF2 path) shipped with
//! gh#51; this test pins the v2 extension to PMMH.
//!
//! Assertions:
//!
//! 1. The fit run succeeds (exit 0) with `init = "survey_top_k"`
//!    on a PMMH stage.
//! 2. The stage dir carries a `chain_starts.tsv` file whose two chain
//!    rows have `source` columns of the form
//!    `"survey:<full-hash>:rank-1"` / `"survey:<full-hash>:rank-2"`.
//! 3. The per-chain starts in `chain_starts.tsv` match the landscape's
//!    top-2 rows (by `loglik` descending).
//!
//! Skipped when the release binary or camdlc isn't present.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../target/release/camdl");
    if p.exists() { Some(p) } else { None }
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_survey_top_k_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Replicate `crate::hashing::model_hash` for the integration test.
/// The structural-hash algorithm hashes a fixed list of fields in
/// order: each `key\x00<json-value>\x00`, plus the `version` field
/// (without a trailing NUL) at the end.
fn model_hash_for_test(ir_json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(ir_json)
        .expect("model_hash_for_test: invalid JSON");
    let envelope = v.as_object().expect("model_hash_for_test: expected object");
    // gh#135: descend into the `model` envelope key (mirror of the
    // production fix in hashing.rs); tolerate a bare inner model.
    let obj = envelope.get("model").and_then(|m| m.as_object()).unwrap_or(envelope);
    let mut h = Sha256::new();
    let structural_keys = [
        "compartments", "transitions", "parameters", "tables",
        "time_functions", "interventions", "observations",
        "ode_equations", "initial_conditions",
    ];
    for key in &structural_keys {
        if let Some(val) = obj.get(*key) {
            h.update(key.as_bytes());
            h.update(b"\x00");
            h.update(serde_json::to_string(val).unwrap().as_bytes());
            h.update(b"\x00");
        }
    }
    if let Some(val) = obj.get("version") {
        h.update(b"version\x00");
        h.update(serde_json::to_string(val).unwrap().as_bytes());
    }
    hex::encode(h.finalize())
}

fn sha256_hex_of_file(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    let mut h = Sha256::new();
    h.update(&bytes);
    hex::encode(h.finalize())
}

/// Build a tiny SIR-with-Poisson-cases fixture that PMMH can fit in
/// seconds. The bounds on `beta` and `gamma` must be wide enough that
/// the landscape rows we'll seed from fall inside.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
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
  cases : {
    projected  = prevalence(I)
    every      = 1 'days
    likelihood = poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    // Tiny dataset — 6 days, low counts.
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

/// Write a 4-row landscape.tsv + a matching run.json into `survey_dir`.
/// `model_hash`, `data_hashes`, `fixed` and `estimated` must match what
/// the fit will compute when it cross-checks the survey. Returns the
/// full content hash embedded into run.json (the value PMMH will
/// surface as `chain_init_source` and as the `source` column of
/// `chain_starts.tsv`).
fn write_survey_artifact(
    survey_dir: &Path,
    model_hash: &str,
    data_hash_cases: &str,
) -> String {
    std::fs::create_dir_all(survey_dir).unwrap();

    // The full content hash is arbitrary as far as the cross-check is
    // concerned — only model_hash / data_hashes / fixed / estimated
    // get verified against the fit. Use a fully synthetic 64-char hex
    // string so the assertion on the `source` column has a stable
    // expected value.
    let survey_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    // run.json — RunKind::Survey shape. The Run struct is
    // `#[serde(tag = "kind", rename_all = "kebab-case")]` so the
    // discriminator goes inside `kind`.
    let run_json = serde_json::json!({
        "hash": survey_hash,
        "version": "test-fixture",
        "created_at": "2026-05-24T00:00:00Z",
        "argv": ["camdl", "survey", "<test-fixture>"],
        "status": { "completed": { "wall_time_seconds": 0.0 } },
        "kind": {
            "kind": "survey",
            "model": "sir.camdl",
            "model_hash": model_hash,
            "data_hashes": { "cases": data_hash_cases },
            "bounds": {
                "beta":  [0.01, 5.0],
                "gamma": [0.01, 1.0],
            },
            "n_points": 4,
            "eval_method": "pfilter",
            "eval_particles": 100,
            "eval_replicates": 1,
            "seed": 1,
            // The fit pins N0 at 1000; the survey must agree (it's a
            // superset of the fit's [fixed] block per the gh#51
            // cross-check).
            "fixed": { "N0": 1000.0 },
            "estimated": ["beta", "gamma"],
        }
    });
    std::fs::write(
        survey_dir.join("run.json"),
        serde_json::to_string_pretty(&run_json).unwrap(),
    ).unwrap();

    // landscape.tsv — 4 rows. PMMH will pick rank-1 + rank-2 by
    // loglik desc. Param columns in the order matched by run.json's
    // `estimated`. Same column-set the survey writer produces
    // (`<param>... loglik loglik_se mean_ess n_replicates point_id`).
    //
    // Row layout (loglik desc → ranks):
    //   beta=0.30, gamma=0.10 → loglik=-50.0 → rank-1 (BEST)
    //   beta=0.50, gamma=0.20 → loglik=-55.0 → rank-2
    //   beta=0.10, gamma=0.30 → loglik=-60.0 → rank-3
    //   beta=0.80, gamma=0.40 → loglik=-65.0 → rank-4
    //
    // (The loglik values are diagnostic-only — PMMH doesn't recompute
    //  them; the cross-check only walks run.json metadata.)
    let landscape = "\
# camdl survey landscape (test fixture)\n\
beta\tgamma\tloglik\tloglik_se\tmean_ess\tn_replicates\tpoint_id\n\
0.30\t0.10\t-50.0\t1.0\t0.8\t1\t0\n\
0.50\t0.20\t-55.0\t1.0\t0.8\t1\t1\n\
0.10\t0.30\t-60.0\t1.0\t0.8\t1\t2\n\
0.80\t0.40\t-65.0\t1.0\t0.8\t1\t3\n";
    std::fs::write(survey_dir.join("landscape.tsv"), landscape).unwrap();

    survey_hash.to_string()
}

fn write_fit_toml(
    dir: &Path,
    ir: &Path,
    data: &Path,
    survey_dir: &Path,
) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
backend = "chain_binomial"
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0],  prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.post]
algorithm      = "pmmh"
backend        = "chain_binomial"
chains         = 2
particles      = 30
iterations     = 50
burn_in        = 5
thin           = 1
init           = "survey_top_k"
survey_path    = "{survey}"
"#,
        out    = dir.join("results").display(),
        ir     = ir.display(),
        data   = data.display(),
        survey = survey_dir.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Parse `chain_starts.tsv` body into (header_cols, data_rows).
/// Comments (`#`-prefixed) are stripped; rows are split on tab.
fn parse_chain_starts_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut lines = raw.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header = lines.next().expect("header row");
    let cols: Vec<String> = header.split('\t').map(String::from).collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(String::from).collect())
        .collect();
    (cols, rows)
}

#[test]
#[ignore = "survey/top-k init not yet migrated to CAS — M3.3 (gh#151)"]
fn pmmh_survey_top_k_writes_chain_starts_with_survey_ranks() {
    let Some(bin) = camdl_bin() else { return };
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("pmmh");
    let (ir, data) = write_fixture(tmp.path());

    // Compute the model_hash + data_hashes the *fit* will see when it
    // cross-checks the survey. Both sides must agree byte-for-byte
    // (gh#51 §"Validation: the run.json cross-check").
    let ir_json = std::fs::read_to_string(&ir).unwrap();
    let mh = model_hash_for_test(&ir_json);
    let dh = sha256_hex_of_file(&data);

    // Write the synthetic survey CAS dir.
    let survey_dir = tmp.path().join("survey_dir");
    let survey_hash = write_survey_artifact(&survey_dir, &mh, &dh);

    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &survey_dir);
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", "1"])
        .output().expect("spawn camdl fit run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(),
        "pmmh fit must succeed with init=survey_top_k.\n\
         stdout:\n{}\nstderr:\n{}", stdout, stderr);

    // Locate the stage dir (output_dir/fits/<fit-stem>-<hash>/real/fit_1/post).
    let fits_dir = tmp.path().join("results/fits");
    let fit_dir = std::fs::read_dir(&fits_dir).unwrap()
        .flatten().map(|e| e.path()).next().expect("one fit dir");
    let stage_dir = fit_dir.join("real/fit_1/post");
    assert!(stage_dir.exists(),
        "stage dir missing: {}", stage_dir.display());

    let starts_tsv = stage_dir.join("chain_starts.tsv");
    assert!(starts_tsv.exists(),
        "chain_starts.tsv must be written under {}", stage_dir.display());

    let (cols, rows) = parse_chain_starts_tsv(&starts_tsv);
    assert_eq!(rows.len(), 2,
        "expected 2 chain rows (chains=2), got: {:?}", rows);

    let chain_id_idx = cols.iter().position(|c| c == "chain_id")
        .expect("chain_id column");
    let source_idx = cols.iter().position(|c| c == "source")
        .expect("source column");
    let beta_idx = cols.iter().position(|c| c == "beta")
        .expect("beta column");
    let gamma_idx = cols.iter().position(|c| c == "gamma")
        .expect("gamma column");

    // Rows are written chain_id-ordered (chain 0, chain 1, ...).
    // rank-N is 1-indexed; chain 0 = rank-1 (best), chain 1 = rank-2.
    let expected_sources = [
        format!("survey:{}:rank-1", survey_hash),
        format!("survey:{}:rank-2", survey_hash),
    ];
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row[chain_id_idx], i.to_string(),
            "row {} chain_id mismatch: row = {:?}", i, row);
        assert_eq!(row[source_idx], expected_sources[i],
            "row {} source mismatch: row = {:?}", i, row);
    }

    // Per-chain starts: rank-1 (loglik = -50.0) is beta=0.30, gamma=0.10;
    // rank-2 (loglik = -55.0) is beta=0.50, gamma=0.20. Both lie within
    // the fit's bounds (beta ∈ [0.01, 5.0], gamma ∈ [0.01, 1.0]).
    let row0_beta: f64 = rows[0][beta_idx].parse().unwrap();
    let row0_gamma: f64 = rows[0][gamma_idx].parse().unwrap();
    let row1_beta: f64 = rows[1][beta_idx].parse().unwrap();
    let row1_gamma: f64 = rows[1][gamma_idx].parse().unwrap();
    assert!((row0_beta - 0.30).abs() < 1e-9,
        "chain 0 beta should = 0.30 (rank-1), got {}", row0_beta);
    assert!((row0_gamma - 0.10).abs() < 1e-9,
        "chain 0 gamma should = 0.10 (rank-1), got {}", row0_gamma);
    assert!((row1_beta - 0.50).abs() < 1e-9,
        "chain 1 beta should = 0.50 (rank-2), got {}", row1_beta);
    assert!((row1_gamma - 0.20).abs() < 1e-9,
        "chain 1 gamma should = 0.20 (rank-2), got {}", row1_gamma);
}
