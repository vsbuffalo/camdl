//! Integration smoke test for `camdl profile --algorithm pmmh`.
//!
//! Mirrors the profile_multi_stream.rs harness shape: drives the
//! release binary, scrapes the per-cell artifacts.
//!
//! Assertions:
//!
//! 1. The profile run completes successfully on a 2-cell sweep of a
//!    small SEIR with ~52 weekly observations.
//! 2. Every cell writes an `mle.toml` containing `final_loglik`.
//! 3. The per-start `run.json` records `method = pmmh`,
//!    `backend = chain_binomial`, and an `algorithm` block matching
//!    the PMMH serialization (steps / particles / rho / dt).
//! 4. Passing `--algorithm pmmh --backend ode` is rejected with an
//!    actionable error pointing at `--backend chain_binomial`.
//!
//! 5. gh#97 (`profile_pmmh_reported_loglik_matches_saved_mle_params`):
//!    under a sharply informative nuisance prior the per-cell MAP θ
//!    sits OFF the likelihood ridge, so the per-sample max loglik
//!    (`best_ll`) exceeds the loglik at the MAP θ (`map_loglik`). The
//!    cell's `mle.toml` must report the loglik that belongs to the
//!    saved `[mle]` params — i.e. `map_loglik` — not `best_ll` from a
//!    different θ. This test independently re-evaluates the loglik at
//!    the saved params with `camdl pfilter` and asserts agreement
//!    within the particle-filter standard error. Pre-fix (final_ll =
//!    `result.map_loglik.max(best_ll)`) it FAILS by ~13 nats; the fix
//!    (report `result.map_loglik` directly) makes it pass.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// Locate the freshly-built `camdlc.exe`. The gh#97 test compiles its
/// own fixture (a small SIR with a sharp `~` prior on `gamma`), so it
/// needs the compiler; skip silently when absent (mirrors the rest of
/// the integration suite).
fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

fn seir_observations_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/seir_observations.ir.json")
}

/// Synthesize per-stream observation TSVs at the baseline preset.
/// `seir_observations.ir.json` has two streams with different
/// schedules (weekly_cases at 7d, detection at 14d), so `--obs-only`
/// can't unify them into a single TSV. We use `--obs-dir` and then
/// pick the `weekly_cases.tsv`.
fn synth_weekly_cases_tsv(bin: &Path, tmp: &Path) -> PathBuf {
    let obs_dir = tmp.join("obs_streams");
    std::fs::create_dir_all(&obs_dir).unwrap();
    let status = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", &seir_observations_ir().to_string_lossy(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "42",
            "--scenario", "baseline",
            "--obs-dir", &obs_dir.to_string_lossy(),
        ])
        .status()
        .expect("spawn camdl simulate");
    assert!(status.success(), "synthetic obs generation failed");
    let obs_path = obs_dir.join("weekly_cases.tsv");
    assert!(obs_path.exists(),
        "weekly_cases.tsv not written under {}", obs_dir.display());
    obs_path
}

#[test]
fn profile_pmmh_smoke_writes_mle_and_algorithm_block() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data_path = synth_weekly_cases_tsv(&bin, tmp.path());

    let out_root = tmp.path().join("camdl_out");
    let out_tsv  = tmp.path().join("profile_pmmh.tsv");

    // 2-cell sweep over beta with PMMH (very short chain, small PF —
    // smoke test only). One start so we have exactly two MLE files.
    let status = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_observations_ir().to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data_path.to_string_lossy(),
            "--obs", "weekly_cases",
            "--sweep", "beta=lin(0.25,0.35,2)",
            "--particles", "100",
            "--algorithm", "pmmh",
            // Must exceed the fixed per-cell burn-in (100); 150 leaves 50
            // post-burn-in samples — still a fast smoke run (gh#102).
            "--pmmh-steps", "150",
            "--pmmh-particles", "100",
            "--pmmh-rho", "0.99",
            "--starts", "1",
            "--rw-sd", "auto",
            "--fixed", "sigma=0.2", "--fixed", "gamma=0.1",
            "--fixed", "rho=0.5", "--fixed", "k=5.0",
            "--fixed", "p_detect=0.8", "--fixed", "N0=100000.0",
            "--fixed", "I0=10.0",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .status()
        .expect("spawn camdl profile pmmh");
    assert!(status.success(), "pmmh profile run failed");

    // Collect the new-format ProfilePoint leaves:
    // profiles/<base>/<point>/<stage>/<seed>/<start>/{mle.toml, run.json}.
    // Two grid points × 1 seed × 1 start ⇒ two leaves.
    fn collect_leaves(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join("run.json").is_file() {
            if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                    if v.get("kind").and_then(|k| k.as_str()) == Some("profile_point") {
                        out.push(dir.to_path_buf());
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { collect_leaves(&e.path(), out); } }
        }
    }
    let profiles_dir = out_root.join("profiles");
    let mut leaves = Vec::new();
    collect_leaves(&profiles_dir, &mut leaves);
    assert_eq!(leaves.len(), 2,
        "expected 2 ProfilePoint leaves (2 grid points × 1 seed × 1 start), got {:?}",
        leaves);

    for leaf in &leaves {
        let mle_toml = leaf.join("mle.toml");
        assert!(mle_toml.exists(), "missing mle.toml under {}", leaf.display());
        let body = std::fs::read_to_string(&mle_toml).unwrap();
        assert!(body.contains("final_loglik = "),
            "mle.toml missing final_loglik:\n{}", body);

        let run: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(leaf.join("run.json")).unwrap())
            .expect("parse run.json");
        assert_eq!(run.get("kind").and_then(|v| v.as_str()), Some("profile_point"),
            "expected kind = profile_point, got: {:?}", run.get("kind"));
        // The PMMH method + algorithm hyperparams live in the recorded
        // (display-only) `inputs` of the leaf record.
        let inputs = run.get("inputs").expect("inputs block");
        assert_eq!(inputs.get("method").and_then(|v| v.as_str()), Some("pmmh"),
            "method should be pmmh, got: {:?}", inputs.get("method"));
        let alg = inputs.get("algorithm").expect("algorithm block");
        assert_eq!(alg.get("steps").and_then(|v| v.as_u64()), Some(150),
            "algorithm.steps mismatch: {:?}", alg);
        assert_eq!(alg.get("particles").and_then(|v| v.as_u64()), Some(100),
            "algorithm.particles mismatch: {:?}", alg);
        let rho_v = alg.get("rho").and_then(|v| v.as_f64())
            .expect("algorithm.rho should be a finite float");
        assert!((rho_v - 0.99).abs() < 1e-12,
            "algorithm.rho expected 0.99, got: {}", rho_v);
    }
}

#[test]
fn profile_pmmh_rejects_ode_backend() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data_path = synth_weekly_cases_tsv(&bin, tmp.path());

    let out_root = tmp.path().join("camdl_out");
    let out_tsv  = tmp.path().join("profile_pmmh.tsv");

    let output = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_observations_ir().to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data_path.to_string_lossy(),
            "--obs", "weekly_cases",
            "--sweep", "beta=lin(0.25,0.35,2)",
            "--particles", "100",
            "--algorithm", "pmmh",
            "--backend", "ode",
            "--pmmh-steps", "50",
            "--pmmh-particles", "50",
            "--starts", "1",
            "--rw-sd", "auto",
            "--fixed", "sigma=0.2", "--fixed", "gamma=0.1",
            "--fixed", "rho=0.5", "--fixed", "k=5.0",
            "--fixed", "p_detect=0.8", "--fixed", "N0=100000.0",
            "--fixed", "I0=10.0",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .output()
        .expect("spawn camdl profile");
    assert!(!output.status.success(),
        "expected non-zero exit when combining pmmh with --backend ode");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Either the upstream methods-matrix rejection or the profile
    // PMMH-specific guard should fire; both name chain_binomial as
    // the right answer.
    assert!(stderr.contains("chain_binomial"),
        "error must guide user to --backend chain_binomial. stderr:\n{}",
        stderr);
}

// ── gh#97: reported loglik must belong to the saved MLE params ──────
//
// The PMMH per-cell branch sets `mle_params = result.map_params` (the
// MAP θ) but, pre-fix, reported `final_ll = result.map_loglik.max(
// best_ll)` where `best_ll = max(s.log_likelihood)` over all chain
// steps. Under a non-flat prior the loglik-maximizing step is NOT the
// posterior-maximizing (MAP) step, so `best_ll > map_loglik` and the
// reported number comes from a θ that is not `map_params`. The cell's
// `mle.toml` then claims "loglik at MLE" using a loglik from one θ and
// params from another.
//
// This test builds that exact regime: a SIR whose `gamma` carries a
// sharply informative `~ log_normal` prior centred (≈0.18) well below
// the data-generating value (0.30). The PMMH MAP `gamma` is pulled
// toward the prior, off the likelihood ridge, while the chain still
// visits higher-loglik (larger-gamma) steps — so `best_ll` exceeds the
// loglik at the saved MAP `gamma`. We parse the saved `[mle]` params +
// `final_loglik` from `mle.toml`, independently re-evaluate the loglik
// at those params with `camdl pfilter --replicates`, and assert the two
// agree within the PF standard error. The disagreement pre-fix is ~13
// nats — far outside any PF SE.

/// Compile a small SIR fixture with a sharp `~` prior on `gamma`.
/// The prior is centred at ≈0.18 (mu = ln 0.18) with sigma = 0.05, so
/// it pulls the MAP estimate of `gamma` well below the data-generating
/// value (0.30) yet stays in a region where the particle filter is
/// non-degenerate. Returns the compiled IR path.
fn write_sharp_prior_sir_ir(dir: &Path, camdlc: &Path) -> PathBuf {
    // mu = ln(0.18) = -1.7148; sigma = 0.05 makes the prior sharp.
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0] ~ log_normal(mu = -1.7148, sigma = 0.05)
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
scenarios {
  baseline {
    set = {
      beta  = 0.8
      gamma = 0.3
      N0    = 1000
    }
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 30 'days }
"#;
    let model_path = dir.join("sharp_prior_sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let out = Command::new(camdlc).arg(&model_path).output()
        .expect("spawn camdlc");
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir_path = dir.join("sharp_prior_sir.ir.json");
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Synthesize a `time,cases` TSV at the baseline preset, then strip the
/// `t = t_start` (t=0) row: the profile-PMMH bootstrap filter cannot
/// score an observation that coincides with `t_start`, which makes the
/// whole chain degenerate (loglik = -inf). Dropping the leading row
/// keeps the regime finite without touching the bug under test.
fn synth_cases_tsv_no_t0(bin: &Path, ir: &Path, tmp: &Path) -> PathBuf {
    let full = tmp.join("cases_full.tsv");
    let status = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", &ir.to_string_lossy(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "42",
            "--scenario", "baseline",
            "--obs-only", &full.to_string_lossy(),
        ])
        .status()
        .expect("spawn camdl simulate");
    assert!(status.success(), "synthetic cases generation failed");
    let body = std::fs::read_to_string(&full).unwrap();
    let mut kept = String::new();
    for (i, line) in body.lines().enumerate() {
        // Keep the header, and every data row whose time column != 0.
        if i == 0 {
            kept.push_str(line);
            kept.push('\n');
            continue;
        }
        let t = line.split('\t').next().unwrap_or("");
        if t != "0" {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    let out = tmp.join("cases.tsv");
    std::fs::write(&out, kept).unwrap();
    out
}

/// Parse `final_loglik` and the `[mle]` parameter block from an
/// `mle.toml`. Returns (final_loglik, [(name, value)]). Only the flat
/// `key = value` lines under `[mle]` are collected.
fn parse_mle_toml(path: &Path) -> (f64, Vec<(String, f64)>) {
    let body = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut final_ll: Option<f64> = None;
    let mut in_mle = false;
    let mut params = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_mle = t == "[mle]";
            continue;
        }
        if let Some(rest) = t.strip_prefix("final_loglik = ") {
            final_ll = Some(parse_toml_float(rest));
            continue;
        }
        if in_mle {
            if let Some((k, v)) = t.split_once('=') {
                params.push((k.trim().to_string(), parse_toml_float(v.trim())));
            }
        }
    }
    (
        final_ll.unwrap_or_else(|| panic!("final_loglik missing in {}", path.display())),
        params,
    )
}

/// Parse a TOML float, tolerating the `inf` / `-inf` the renderer emits
/// for non-finite logliks.
fn parse_toml_float(s: &str) -> f64 {
    match s.trim() {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        other => other.parse::<f64>()
            .unwrap_or_else(|e| panic!("parse float {:?}: {}", other, e)),
    }
}

/// Independently re-evaluate the loglik at `params` via
/// `camdl pfilter --replicates`, returning each replicate's loglik.
/// The replicate logliks are written to a TSV (`seed\tloglik`); we
/// parse the loglik column. Caller computes mean / SE.
fn pfilter_replicate_logliks(
    bin: &Path,
    ir: &Path,
    data: &Path,
    params: &[(String, f64)],
    particles: u32,
    replicates: u32,
    tmp: &Path,
) -> Vec<f64> {
    let out = tmp.join("reeval_logliks.tsv");
    let mut cmd = Command::new(bin);
    cmd.env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &ir.to_string_lossy(),
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--particles", &particles.to_string(),
            "--replicates", &replicates.to_string(),
            "--seed", "7",
            "--output", &out.to_string_lossy(),
        ]);
    for (name, value) in params {
        cmd.args(["--param", &format!("{}={}", name, value)]);
    }
    let output = cmd.output().expect("spawn camdl pfilter");
    assert!(output.status.success(),
        "pfilter re-eval failed; stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    let body = std::fs::read_to_string(&out)
        .unwrap_or_else(|e| panic!("read {}: {}", out.display(), e));
    let mut logliks = Vec::new();
    for (i, line) in body.lines().enumerate() {
        if i == 0 { continue; } // header: seed\tloglik
        if let Some(ll) = line.split('\t').nth(1) {
            logliks.push(parse_toml_float(ll));
        }
    }
    assert!(!logliks.is_empty(),
        "no replicate logliks parsed from {}", out.display());
    logliks
}

#[test]
fn profile_pmmh_reported_loglik_matches_saved_mle_params() {
    let bin = skip_if_missing_binary();
    let camdlc = match camdlc_bin() {
        Some(c) => c,
        None => return, // compiler absent — skip (CI builds it)
    };
    let tmp = tempfile::tempdir().unwrap();
    let ir = write_sharp_prior_sir_ir(tmp.path(), &camdlc);
    let data = synth_cases_tsv_no_t0(&bin, &ir, tmp.path());

    let out_root = tmp.path().join("camdl_out");
    let out_tsv = tmp.path().join("profile_pmmh.tsv");

    // 2-cell sweep over the focal `beta`, `gamma` estimated by PMMH
    // under its sharp prior. `N0` pinned. A long-ish chain + generous
    // particle count so the MAP is well-resolved and the per-cell
    // loglik trace stays finite.
    let status = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &ir.to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--sweep", "beta=lin(0.78,0.82,2)",
            "--particles", "1500",
            "--algorithm", "pmmh",
            "--pmmh-steps", "1000",
            "--pmmh-particles", "1500",
            "--pmmh-rho", "0.99",
            "--starts", "1",
            "--rw-sd", "auto",
            "--fixed", "N0=1000.0",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .status()
        .expect("spawn camdl profile pmmh");
    assert!(status.success(), "pmmh profile run failed");

    // Collect the ProfilePoint leaves (2 cells × 1 seed × 1 start).
    fn collect_leaves(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join("run.json").is_file() {
            if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                    if v.get("kind").and_then(|k| k.as_str()) == Some("profile_point") {
                        out.push(dir.to_path_buf());
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { collect_leaves(&e.path(), out); } }
        }
    }
    let mut leaves = Vec::new();
    collect_leaves(&out_root.join("profiles"), &mut leaves);
    assert_eq!(leaves.len(), 2,
        "expected 2 ProfilePoint leaves, got {:?}", leaves);

    // At least one cell must exercise the bug: a finite reported loglik
    // whose saved MAP θ sits off the ridge. We assert agreement on every
    // finite-loglik cell, and require that at least one cell was finite
    // (otherwise the regime degenerated and the test proved nothing).
    let mut checked_finite = 0usize;
    for leaf in &leaves {
        let mle_toml = leaf.join("mle.toml");
        assert!(mle_toml.exists(), "missing mle.toml under {}", leaf.display());
        let (reported_ll, mut mle_params) = parse_mle_toml(&mle_toml);

        if !reported_ll.is_finite() {
            // Degenerate cell — nothing to compare. (Should not happen
            // in this regime, but don't fail the whole test on it.)
            continue;
        }
        checked_finite += 1;

        // Re-evaluate the loglik at the FULL saved parameter vector:
        // the `[mle]` block carries `gamma`; the focal `beta` and the
        // fixed `N0` are not in `[mle]`, so add them from the leaf's
        // focal value / the pinned fixed value.
        let (_, focal_beta) = parse_focal_beta(&mle_toml);
        mle_params.push(("beta".to_string(), focal_beta));
        mle_params.push(("N0".to_string(), 1000.0));

        // Independent PF re-eval at the saved params. More particles +
        // replicates than the single PMMH-recorded estimate so the SE
        // is small relative to the bug gap (~13 nats).
        let logliks = pfilter_replicate_logliks(
            &bin, &ir, &data, &mle_params,
            /*particles=*/ 2000, /*replicates=*/ 12, tmp.path(),
        );
        let n = logliks.len() as f64;
        let mean = logliks.iter().sum::<f64>() / n;
        let var = logliks.iter().map(|&l| (l - mean).powi(2)).sum::<f64>()
            / (n - 1.0).max(1.0);
        let se = (var / n).sqrt();

        // The reported loglik must be the loglik at the SAVED params
        // (= `map_loglik`), so it agrees with this independent estimate
        // within PF noise. Tolerance: a generous multiple of the SE,
        // floored at a few nats to absorb the single-PMMH-run PF noise
        // and the param rounding in mle.toml — still far below the ~13-
        // nat pre-fix gap (which reports `best_ll` from a different θ).
        let tol = (6.0 * se).max(5.0);
        assert!(
            (reported_ll - mean).abs() <= tol,
            "mle.toml final_loglik ({:.4}) for params {:?} disagrees with \
             an independent PF re-eval at those same params ({:.4} ± {:.4}, \
             {} reps): |Δ| = {:.4} > tol {:.4}. This is gh#97 — the reported \
             loglik belongs to a different θ than the saved MLE params.",
            reported_ll, mle_params, mean, se, logliks.len(),
            (reported_ll - mean).abs(), tol,
        );
    }
    assert!(checked_finite >= 1,
        "no cell produced a finite loglik — the regime degenerated and the \
         test exercised nothing");
}

/// Extract the focal `beta` value from the `[focal]` block of an
/// `mle.toml`. Returns (name, value); the value is the pinned grid
/// point for this cell.
fn parse_focal_beta(path: &Path) -> (String, f64) {
    let body = std::fs::read_to_string(path).unwrap();
    let mut in_focal = false;
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with('[') { in_focal = t == "[focal]"; continue; }
        if in_focal {
            if let Some((k, v)) = t.split_once('=') {
                if k.trim() == "beta" {
                    return ("beta".to_string(), parse_toml_float(v.trim()));
                }
            }
        }
    }
    panic!("focal beta not found in {}", path.display());
}
