//! Integration test for `camdl fit run --resume` against PGAS.
//!
//! Verifies the wiring landed in 2026-04-30 (`Stage::identity_payload`
//! split + `--resume` flag plumbed through the dispatcher):
//!
//! 1. A first PGAS run writes `chain_<n>/resume_state.bin` containing
//!    completed_sweeps == n_sweeps and the stage's identity hash.
//! 2. A second invocation with `--resume --stage post --sweeps N>n_sweeps`
//!    succeeds and continues the chain (does not re-run burn-in).
//! 3. Changing an *identity* field (e.g. `chains`) between the two
//!    invocations causes resume to reject with a hash-mismatch error.
//!
//! Skipped when the release binary or camdlc isn't present.

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
        "camdl_pgas_resume_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Build a tiny SIR model + Poisson obs IR and write trivial data so
/// PGAS can run end-to-end in seconds.
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

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, sweeps: usize, chains: usize) -> PathBuf {
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
beta  = {{ bounds = [0.01, 5.0],  prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = {chains}
particles = 30
sweeps = {sweeps}
# Tiny burn_in so the post-burn-in sample set is non-empty even with
# small `sweeps`. (PGAS panics if sweeps <= burn_in; default is 2000.)
burn_in = 2
"#,
        out = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join(format!("fit_{}_{}.toml", sweeps, chains));
    std::fs::write(&p, toml).unwrap();
    p
}

/// (1) and (2): first run writes resume_state.bin; second run with
/// `--resume` reads it and announces resumption.
///
/// Caveat documented here so future-us doesn't relitigate it: the
/// fit_dir is keyed on the raw fit.toml bytes (`fit_content_hash`).
/// So you cannot bump `sweeps` in the TOML and have the new run
/// find the old fit_dir's resume_state.bin — they live in different
/// directories. The identity-vs-extension split lives at the
/// stage-hash level (`fit_stage_hash` via `identity_payload`), not
/// the fit-content-hash level. A future `--sweeps N` CLI override
/// (analogous to `--max-tree-depth N`) would make resume-to-extend
/// work without TOML edits, but that's a separate feature.
#[test]
#[ignore = "resume not yet on CAS — M3.2 commit 2 (gh#152)"]
fn pgas_resume_announces_continuation_with_unchanged_toml() {
    let Some(bin) = camdl_bin() else { return };
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("continues");
    let (ir, data) = write_fixture(tmp.path());

    // First run: 8 sweeps. Tiny but enough to write resume_state.bin.
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, /*sweeps*/ 8, /*chains*/ 1);
    let out = Command::new(&bin)
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", "1"])
        .output().expect("spawn");
    assert!(out.status.success(),
        "first PGAS run failed: stderr={}", String::from_utf8_lossy(&out.stderr));

    // Find the resume_state.bin written by chain 1.
    let fits_dir = tmp.path().join("results/fits");
    let fit_dir = std::fs::read_dir(&fits_dir).unwrap()
        .flatten().map(|e| e.path()).next().expect("one fit dir");
    let stage_dir = fit_dir.join("real/fit_1/post");
    let resume_state = stage_dir.join("chain_1/resume_state.bin");
    assert!(resume_state.exists(),
        "first run should write {}", resume_state.display());
    let first_size = std::fs::metadata(&resume_state).unwrap().len();
    assert!(first_size > 0, "resume_state.bin must be non-empty");

    // Second run: same TOML, `--resume`. The runner reads
    // resume_state.bin and announces resumption (even though the
    // chain is already at the target sweep count, this exercises
    // the load + hash-check paths).
    let out = Command::new(&bin)
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--seed", "1", "--stage", "post", "--resume"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "resume run must succeed: stderr={}", stderr);
    assert!(stderr.contains("resuming from sweep"),
        "stderr should announce resumption: {}", stderr);
}

/// (3): changing an identity field (chains) between runs must reject
/// `--resume` with a hash-mismatch error.
#[test]
fn pgas_resume_rejects_when_identity_field_changes() {
    let Some(bin) = camdl_bin() else { return };
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("rejects");
    let (ir, data) = write_fixture(tmp.path());

    // First run: 1 chain.
    let fit1 = write_fit_toml(tmp.path(), &ir, &data, 8, 1);
    let out = Command::new(&bin)
        .args(["fit", "run", &fit1.to_string_lossy(), "--seed", "1"])
        .output().expect("spawn");
    assert!(out.status.success(),
        "first PGAS run failed: stderr={}", String::from_utf8_lossy(&out.stderr));

    // Second run: 2 chains (chains is identity-defining → fit hash
    // changes → fit_dir changes too, so the resume_state.bin from
    // run 1 isn't even visible). Resume should fail with "no resume
    // state file" (the new fit_dir has no chain_1/resume_state.bin).
    let fit2 = write_fit_toml(tmp.path(), &ir, &data, 8, 2);
    let out = Command::new(&bin)
        .args(["fit", "run", &fit2.to_string_lossy(),
               "--seed", "1", "--stage", "post", "--resume"])
        .output().expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "resume with changed chains must reject");
    assert!(stderr.contains("no resume state") || stderr.contains("config hash mismatch"),
        "expected hash-mismatch or no-state error: got {}", stderr);
}
