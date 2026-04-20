//! End-to-end tests for the backend-provenance guardrail.
//!
//! See `docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`
//! and the originating incident
//! `docs/dev/incidents/2026-04-19-backend-default-mismatch.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl-sim")
}

fn skip_if_missing() -> Option<PathBuf> {
    let b = binary();
    if !b.exists() {
        eprintln!("skipping: binary not built at {}", b.display());
        return None;
    }
    Some(b)
}

fn golden_pure_death() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/pure_death.ir.json")
}

/// Write a minimal mle_params.toml with a `[provenance]` block claiming
/// the fit used `backend` at `dt=1.0`. Top-level params come first
/// (required by TOML so they stay at the file's top level, not scoped
/// under the [provenance] table) — matches what `write_mle_params`
/// produces in production.
fn write_fake_mle(path: &Path, backend: &str, dt: f64) {
    let contents = format!(r#"mu = 0.05

[provenance]
camdl_version = "test"
timestamp = "2026-04-19T00:00:00Z"
content_hash = "deadbeef"
backend = "{}"
dt = {}
model = "pure_death.ir.json"
model_hash = "f00d"
seed = 1
stage = "refine"
chain = 1
log_likelihood = -22.0
loglik_sd = 0.0
n_particles = 500
"#, backend, dt);
    std::fs::write(path, contents).unwrap();
}

#[test]
fn simulate_auto_matches_fit_backend_when_backend_not_passed() {
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin)
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(), "expected success; stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("backend auto-matched"),
        "expected auto-match info log, got: {}", stderr);
    assert!(stderr.contains("chain_binomial"),
        "info log should name the fit's backend: {}", stderr);
}

#[test]
fn simulate_warns_on_explicit_backend_mismatch() {
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin)
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--backend", "gillespie",
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(),
        "simulate should proceed despite mismatch warning; stderr: {}",
        String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("backend mismatch"),
        "expected warning on mismatch, got: {}", stderr);
    assert!(stderr.contains("chain_binomial") && stderr.contains("gillespie"),
        "warning should name both backends: {}", stderr);
    assert!(stderr.contains("incidents/2026-04-19-backend-default-mismatch"),
        "warning should cite the incident file: {}", stderr);
}

#[test]
fn simulate_silent_on_matching_explicit_backend() {
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);

    let out = Command::new(&bin)
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "--backend", "chain_binomial", "--dt", "1.0",
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("backend auto-matched"),
        "matching explicit backend should not emit auto-match log: {}", stderr);
    assert!(!stderr.contains("backend mismatch"),
        "matching explicit backend should not emit warning: {}", stderr);
}

#[test]
fn simulate_unchanged_on_standalone_params_file() {
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let standalone = tmp.path().join("p.toml");
    // No [provenance] block — should behave as pre-guardrail.
    std::fs::write(&standalone, "mu = 0.05\n").unwrap();

    let out = Command::new(&bin)
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &standalone.to_string_lossy(),
               "--replicates", "2", "--seed", "1",
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("backend auto-matched"),
        "standalone params should not trigger auto-match log");
    assert!(!stderr.contains("backend mismatch"),
        "standalone params should not trigger warning");
}

#[test]
fn simulate_records_from_fit_hash_in_run_json() {
    // When simulate runs with --params pointing at a fit MLE, and --cas
    // is active, the resulting run.json should record `from_fit_hash`
    // matching the fit's hash. Closes the sim → fit provenance edge.
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    // Use a distinctive fit_hash so we can grep for it.
    let fit_hash = "abc12345deadbeef00000000000000000000000000000000000000000000abcd";
    let contents = format!(r#"mu = 0.05

[provenance]
camdl_version = "test"
timestamp = "2026-04-19T00:00:00Z"
content_hash = "deadbeef"
fit_hash = "{}"
backend = "chain_binomial"
dt = 1.0
model = "pure_death.ir.json"
model_hash = "f00d"
seed = 1
stage = "refine"
chain = 1
log_likelihood = -22.0
loglik_sd = 0.0
n_particles = 500
"#, fit_hash);
    std::fs::write(&mle, contents).unwrap();

    let output = tmp.path().join("out");
    let status = Command::new(&bin)
        .args(["simulate", &golden_pure_death().to_string_lossy(),
               "--params", &mle.to_string_lossy(),
               "--seed", "1", "--cas",
               "--output-dir", &output.to_string_lossy(),
               "-o", &tmp.path().join("t.tsv").to_string_lossy()])
        .status().expect("spawn");
    assert!(status.success());

    // Locate the run.json under output/sims/...
    let run_jsons: Vec<_> = walkdir(&output.join("sims")).into_iter()
        .filter(|p| p.file_name().map(|s| s == "run.json").unwrap_or(false))
        .collect();
    assert_eq!(run_jsons.len(), 1, "expected one sim run, got {}",
        run_jsons.len());
    let body = std::fs::read_to_string(&run_jsons[0]).unwrap();
    assert!(body.contains(fit_hash),
        "run.json must record from_fit_hash; got: {}", body);
}

// ── Behavior tests: read run.json and assert the *resolved* backend ──
//
// The tests above (auto-match, warn, silent) check the stderr text of
// each code path, which is useful as a UX regression guard but not
// sufficient: a bug where the info log says "auto-matched to
// chain_binomial" but the code silently proceeds with Gillespie would
// pass those tests. The tests below fix that by running `simulate
// --cas`, reading the resulting run.json, and asserting the
// `kind.backend` (+ `kind.dt`) fields match what we expect under the
// three-way matching rule.
//
// Two fixtures (one chain_binomial, one gillespie in provenance) so a
// bug hard-coding chain_binomial in the auto-match path is caught —
// checking only one backend direction is insufficient.

/// Helper: find the single run.json under `output/sims/**` and parse
/// its `kind` table.
fn read_sim_kind(output_root: &Path) -> serde_json::Value {
    let run_jsons: Vec<_> = walkdir(&output_root.join("sims")).into_iter()
        .filter(|p| p.file_name().map(|s| s == "run.json").unwrap_or(false))
        .collect();
    assert_eq!(run_jsons.len(), 1,
        "expected exactly one run.json under {}, got {:?}",
        output_root.display(), run_jsons);
    let body = std::fs::read_to_string(&run_jsons[0]).unwrap();
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    v["kind"].clone()
}

/// Run `simulate --cas --params <mle> [extra args...]`, return the
/// output root so the caller can inspect run.json.
fn run_simulate_cas(
    bin: &Path, tmp: &Path, mle: &Path, extra: &[&str],
) -> PathBuf {
    let output = tmp.join("out");
    let mut args: Vec<String> = vec![
        "simulate".into(), golden_pure_death().to_string_lossy().to_string(),
        "--params".into(), mle.to_string_lossy().to_string(),
        "--seed".into(), "1".into(), "--cas".into(),
        "--output-dir".into(), output.to_string_lossy().to_string(),
        "-o".into(), tmp.join("t.tsv").to_string_lossy().to_string(),
    ];
    for a in extra { args.push(a.to_string()); }
    let status = Command::new(bin).args(&args)
        .status().expect("spawn");
    assert!(status.success(), "simulate --cas should succeed");
    output
}

#[test]
fn run_json_records_chain_binomial_when_provenance_says_chain_binomial() {
    // Fixture A: provenance says chain_binomial. No --backend passed.
    // Expected: run.json records backend=chain_binomial (auto-match
    // actually took effect, not just the log text).
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_cbin.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["backend"], "chain_binomial",
        "run.json.backend must match provenance; got: {}", kind);
    assert_eq!(kind["dt"], 1.0);
}

#[test]
fn run_json_records_gillespie_when_provenance_says_gillespie() {
    // Fixture B: provenance says gillespie (tests the opposite
    // direction — guards against a bug where we hardcoded
    // chain_binomial in the auto-match path).
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_gill.toml");
    write_fake_mle(&mle, "gillespie", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["backend"], "gillespie",
        "run.json.backend must match provenance; got: {}", kind);
}

#[test]
fn run_json_records_auto_matched_dt_not_just_backend() {
    // Provenance has dt=0.5 (non-default). Regression guard: a bug
    // where we auto-match backend but NOT dt would make run.json
    // record dt=1.0 (CLI default) despite the fit having used 0.5.
    // Subtly wrong behavior, silent, hard to spot without this check.
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle_dt05.toml");
    write_fake_mle(&mle, "chain_binomial", 0.5);
    let output = run_simulate_cas(&bin, tmp.path(), &mle, &[]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["dt"], 0.5,
        "run.json.dt must auto-match provenance dt (0.5), not default \
         to 1.0. Got kind: {}", kind);
}

#[test]
fn explicit_backend_overrides_provenance_in_run_json() {
    // Provenance says chain_binomial; user explicitly passes
    // --backend gillespie. The user's choice must win (the warning
    // fires, but the run proceeds with gillespie). Regression
    // guard: a bug where the auto-match path always overrides
    // --backend would silently ignore the user's explicit choice.
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 1.0);
    let output = run_simulate_cas(&bin, tmp.path(), &mle,
        &["--backend", "gillespie"]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["backend"], "gillespie",
        "explicit --backend must override provenance; got: {}", kind);
}

#[test]
fn standalone_params_use_chain_binomial_default_in_run_json() {
    // Standalone params file (no [provenance] block) + no --backend:
    // run.json must record chain_binomial (the CLI default, matching
    // `camdl fit`'s default — see the 2026-04-19 incident). Regression
    // guard on the "no leakage from prior runs" invariant.
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let standalone = tmp.path().join("p.toml");
    std::fs::write(&standalone, "mu = 0.05\n").unwrap();
    let output = run_simulate_cas(&bin, tmp.path(), &standalone, &[]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["backend"], "chain_binomial",
        "standalone params must use chain_binomial default; got: {}", kind);
}

#[test]
fn explicit_dt_overrides_provenance_in_run_json() {
    // Provenance has dt=0.5; user explicitly passes --dt 0.25.
    // User's choice wins. Companion to the backend-override test —
    // dt and backend are paired fields and the override semantics
    // should be symmetric.
    let Some(bin) = skip_if_missing() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let mle = tmp.path().join("mle.toml");
    write_fake_mle(&mle, "chain_binomial", 0.5);
    let output = run_simulate_cas(&bin, tmp.path(), &mle,
        &["--dt", "0.25"]);
    let kind = read_sim_kind(&output);
    assert_eq!(kind["dt"], 0.25,
        "explicit --dt must override provenance dt; got: {}", kind);
    // And backend still auto-matches (we didn't pass --backend).
    assert_eq!(kind["backend"], "chain_binomial");
}

fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p.clone()); }
                else { out.push(p); }
            }
        }
    }
    out
}
