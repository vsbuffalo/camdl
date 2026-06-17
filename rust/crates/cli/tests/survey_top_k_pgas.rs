//! Integration test for `init = "survey_top_k"` on a PGAS stage
//! (gh#51 v2). Mirrors `survey_top_k_pmmh.rs`.
//!
//! Verifies that PGAS dispatch accepts `init = "survey_top_k"`
//! and that the top-K landscape rows reach per-chain starts plus the
//! `chain_starts.tsv` audit sidecar. v1 (the IF2 path) shipped with
//! gh#51; this test pins the v2 extension to PGAS.
//!
//! Assertions:
//!
//! 1. The fit run succeeds (exit 0) with `init = "survey_top_k"`
//!    on a PGAS stage.
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

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
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
/// Same algorithm; kept inline so the test binary doesn't depend on
/// the cli crate's private modules.
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
        // gh#147: calendar/time-axis context.
        "origin", "origin_rata_die", "time_unit",
    ];
    for key in &structural_keys {
        if let Some(val) = obj.get(*key) {
            h.update(key.as_bytes());
            h.update(b"\x00");
            h.update(serde_json::to_string(val).unwrap().as_bytes());
            h.update(b"\x00");
        }
    }
    // gh#147: output cadence (`output.times` only — format/flags are presentation).
    if let Some(times) = obj.get("output").and_then(|o| o.as_object()).and_then(|o| o.get("times")) {
        h.update(b"output.times\x00");
        h.update(serde_json::to_string(times).unwrap().as_bytes());
        h.update(b"\x00");
    }
    // gh#147: simulation horizon (`t_start`/`t_end` only — dt/seed/time_semantics excluded).
    if let Some(sim) = obj.get("simulation").and_then(|s| s.as_object()) {
        for key in ["t_start", "t_end"] {
            if let Some(val) = sim.get(key) {
                h.update(b"simulation.");
                h.update(key.as_bytes());
                h.update(b"\x00");
                h.update(serde_json::to_string(val).unwrap().as_bytes());
                h.update(b"\x00");
            }
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

/// Tiny 2-compartment SI fixture — PGAS is expensive per iteration, so
/// keep the obs schedule short and the iteration count low.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
time_unit = 'days
compartments { S, I }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> S @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 4 'days }
"#;
    let model_path = dir.join("si.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("si.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n").unwrap();

    (ir_path, data_path)
}

fn write_survey_artifact(
    survey_dir: &Path,
    model_hash: &str,
    data_hash_cases: &str,
) -> String {
    std::fs::create_dir_all(survey_dir).unwrap();

    let survey_hash = "c0ffee0123456789c0ffee0123456789c0ffee0123456789c0ffee0123456789";

    // New-format survey leaf (`runid::RunRecord`). The cross-check provenance
    // the `survey_top_k` consumer reads (model_hash / data_hashes / fixed /
    // estimated) lives in `inputs`; identity is `run_id`. The fit pins N0 at
    // 1000, so the survey [fixed] must be a superset per the gh#51 cross-check.
    let record = runid::RunRecord {
        format_version: runid::FORMAT_VERSION,
        kind: runid::ArtifactKind::Survey,
        run_id: runid::ContentHash::from_hex(survey_hash).unwrap(),
        hash_version: runid::HASH_VERSION,
        ir_version: "0.7".into(),
        engine_version: "test-fixture".into(),
        levels: Vec::new(),
        deps: Vec::new(),
        status: runid::RunStatus::Completed,
        artifacts: Default::default(),
        children: Default::default(),
        inputs: serde_json::json!({
            "model_hash": model_hash,
            "data_hashes": { "cases": data_hash_cases },
            "fixed": { "N0": 1000.0 },
            "estimated": ["beta", "gamma"],
            "eval_method": "pfilter",
            "eval_particles": 100,
            "eval_replicates": 1,
            "n_points": 4,
        }),
        provenance: Default::default(),
    };
    std::fs::write(
        survey_dir.join("run.json"),
        serde_json::to_string_pretty(&record).unwrap(),
    ).unwrap();

    // landscape.tsv — 4 rows ranked by loglik desc:
    //   beta=0.30, gamma=0.10 → loglik=-50.0 → rank-1
    //   beta=0.50, gamma=0.20 → loglik=-55.0 → rank-2
    //   beta=0.10, gamma=0.30 → loglik=-60.0 → rank-3
    //   beta=0.80, gamma=0.40 → loglik=-65.0 → rank-4
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
    // sweeps = 20: PGAS is expensive per iteration. burn_in = 2 keeps
    // post-burn-in non-empty. Priors are required (PGAS refuses to run
    // with flat priors on estimated params; see gh#audit-C4).
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0],  prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.post]
algorithm      = "pgas"
backend        = "chain_binomial"
chains         = 2
particles      = 30
sweeps         = 20
burn_in        = 2
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

/// The CAS stage leaf for `stage_substr` under `fits_root` —
/// `<fit>/<NN>-<stage>-<h8>/seed_<N>-<h8>/` holding a `fit_stage` run.json.
fn cas_stage_leaf(fits_root: &Path, stage_substr: &str) -> PathBuf {
    let mut stack = vec![fits_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("run.json").is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(d.join("run.json")).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"].as_array().into_iter().flatten()
                        .find(|l| l["name"].as_str() == Some("stage"))
                        .and_then(|l| l["label"].as_str()).unwrap_or("");
                    if stage.contains(stage_substr) { return d; }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no CAS '{}' stage leaf under {}", stage_substr, fits_root.display());
}

#[test]
fn pgas_survey_top_k_writes_chain_starts_with_survey_ranks() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("pgas");
    let (ir, data) = write_fixture(tmp.path());

    let ir_json = std::fs::read_to_string(&ir).unwrap();
    let mh = model_hash_for_test(&ir_json);
    let dh = sha256_hex_of_file(&data);

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
        "pgas fit must succeed with init=survey_top_k.\n\
         stdout:\n{}\nstderr:\n{}", stdout, stderr);

    let fits_dir = tmp.path().join("results/fits");
    let stage_dir = cas_stage_leaf(&fits_dir, "post");
    assert!(stage_dir.join("run.json").is_file(),
        "post stage leaf missing run.json: {}", stage_dir.display());

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
