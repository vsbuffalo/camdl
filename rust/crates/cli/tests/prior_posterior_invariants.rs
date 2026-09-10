//! Cross-method `log_posterior` invariants.
//!
//! Locks in the bug class that gh#73, gh#118, and the 2026-05-26
//! week-audit C2 are all instances of: the `log_posterior` column
//! emitted by `camdl profile --algorithm pmmh` silently dropping
//! prior contributions. gh#73 dropped all priors (silent
//! MLE-with-flat-priors). gh#118 dropped focal-parameter priors
//! (silent off-by-Σ-log_prior(focal) per cell). Both bug classes
//! evaded existing tests because the column was *present* and the
//! value *finite*; only the formula behind it was wrong.
//!
//! The invariant pinned here:
//!
//! > `final_log_posterior` reported in `mle.toml` equals
//! > `final_loglik + Σ log_prior(every estimated parameter,
//! >  including focal swept params, evaluated at its post-PMMH
//! >  value)`, matching `camdl fit run`'s definition.
//!
//! Tests use only Uniform priors so the prior log-density is a
//! constant `-log(upper - lower)` everywhere inside the support.
//! That makes the expected `log_posterior - log_loglik` value
//! a constant determined entirely by the bounds — no need to
//! parse PMMH-sampled MLE values back out of the artifact, and
//! no PF-noise sensitivity. Any path that drops a prior
//! contribution produces a different (typically smaller)
//! observed gap; the assertion fails on the magnitude.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
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
/// Mirrors `profile_pmmh.rs::synth_weekly_cases_tsv` so the harness
/// has the same shape across test files.
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
    obs_dir.join("weekly_cases.tsv")
}

/// Parse a `key = float` line from a TOML body. Tolerant of leading
/// whitespace and `nan`/`-inf` sentinels (returns None for those).
fn parse_toml_finite_f64(body: &str, key: &str) -> Option<f64> {
    body.lines()
        .map(|l| l.trim_start())
        .find_map(|l| {
            let stripped = l.strip_prefix(key)?
                .trim_start()
                .strip_prefix("=")?
                .trim_start();
            // Strip any trailing comment.
            let value_str = stripped.split('#').next()?.trim_end();
            value_str.parse::<f64>().ok()
        })
        .filter(|v| v.is_finite())
}

/// gh#118 / gh#73 invariant: `final_log_posterior` in profile
/// mle.toml must equal `final_loglik + Σ log_prior(all estimated
/// params)`. With Uniform[lo, hi] priors and any MLE inside
/// support, the prior sum is `−log(hi_beta−lo_beta) −
/// log(hi_sigma−lo_sigma)` exactly.
///
/// Pre-gh#118 (commit 7c419a7's wiring) the focal contribution
/// was dropped, so this test would have observed only the
/// nuisance offset `−log(hi_sigma−lo_sigma)` and failed.
/// Pre-gh#73 (commit 5f658a16) the full prior set was Flat, so
/// this test would have observed offset = 0 and failed.
#[test]
fn profile_pmmh_log_posterior_includes_focal_and_nuisance_uniform_priors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data_path = synth_weekly_cases_tsv(&bin, tmp.path());

    // fit.toml declaring Uniform priors on the focal (beta) and one
    // nuisance (sigma) estimated parameter. All other params are
    // --fixed at the CLI so they don't contribute log_prior to the
    // sum.
    let fit_toml = tmp.path().join("fit.toml");
    let model_path = seir_observations_ir().to_string_lossy().to_string();
    // profile reads the problem half — `[model]`, `[estimate]`, `[fixed]` —
    // and ignores the `[method]`; a file with none would load too.
    // Asymmetric widths: focal width = 0.20, nuisance width = 0.25.
    // The two contributions to the prior sum are then distinct
    // (-ln(0.20) ≈ 1.6094 vs -ln(0.25) ≈ 1.3863) so a bug that
    // double-counted EITHER side (focal twice or nuisance twice)
    // produces an observed gap different from the expected
    // 1.6094 + 1.3863 ≈ 2.9957. A symmetric-widths version of
    // this test (the original) could not distinguish "added one
    // side twice" from "added focal + nuisance" — caught by an
    // independent code-review audit on 2026-05-27.
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{}"
[estimate]
beta  = {{ bounds = [0.20, 0.40], rw_sd = 0.02, prior = {{ uniform = {{ lower = 0.20, upper = 0.40 }} }} }}
sigma = {{ bounds = [0.10, 0.35], rw_sd = 0.02, prior = {{ uniform = {{ lower = 0.10, upper = 0.35 }} }} }}
[fixed]
gamma     = 0.1
rho       = 0.5
k         = 5.0
p_detect  = 0.8
N0        = 100000.0
I0        = 10.0
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#, model_path)).unwrap();

    let out_root = tmp.path().join("camdl_out");
    let out_tsv  = tmp.path().join("profile.tsv");

    // 2-cell sweep over beta with PMMH, sigma left as the nuisance
    // estimated parameter. Both focal and nuisance carry Uniform
    // priors so the prior contribution at any inside-support MLE is
    // a known constant.
    let status = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_observations_ir().to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data_path.to_string_lossy(),
            "--obs", "weekly_cases",
            "--fit", &fit_toml.to_string_lossy(),
            "--sweep", "beta=lin(0.25,0.35,2)",
            "--particles", "100",
            "--algorithm", "pmmh",
            "--pmmh-steps", "200",
            "--pmmh-particles", "100",
            // rho = 0.0 = the PLAIN bootstrap PF. The correlated PF (rho > 0)
            // returns -inf on this sharp, high-count NegBinomial likelihood — a
            // real bug filed in docs/dev/notes/2026-06-08-correlated-pf-degenerate-loglik.md
            // (localized: at rho=0.0 final_loglik is finite, at rho>=0.5 it is
            // -inf, everything else equal). This test pins the *prior-sum*
            // invariant, which is filter-agnostic (the gap = log_posterior -
            // loglik is exactly the prior contribution regardless of the filter),
            // so it runs on the working plain PF. Do NOT raise rho here to "test
            // correlation" — that re-buries the correlated-PF bug behind -inf.
            "--pmmh-rho", "0.0",
            "--starts", "1",
            "--rw-sd", "auto",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .status()
        .expect("spawn camdl profile pmmh");
    assert!(status.success(), "pmmh profile run failed");

    // Expected gap: focal (beta on [0.20, 0.40], width 0.20) +
    // nuisance (sigma on [0.10, 0.35], width 0.25). Asymmetric widths
    // so each contribution is distinct.
    let focal_width    = 0.40_f64 - 0.20_f64;   // 0.20
    let nuisance_width = 0.35_f64 - 0.10_f64;   // 0.25
    let expected_gap   = -(focal_width.ln()) + -(nuisance_width.ln());
    // -ln(0.20) + -ln(0.25) ≈ 1.6094 + 1.3863 = 2.9957

    // Walk the seed_1 / points / start_0 / mle.toml tree and assert
    // the invariant on every cell.
    // Collect the new-format ProfilePoint leaf dirs (each holds an
    // `mle.toml`): profiles/<base>/<point>/<stage>/<seed>/<start>/.
    fn collect_leaf_dirs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if dir.join("mle.toml").is_file() {
            if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
                if b.contains("\"profile_point\"") { out.push(dir.to_path_buf()); }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { collect_leaf_dirs(&e.path(), out); } }
        }
    }
    let profiles_dir = out_root.join("profiles");
    let mut leaf_dirs = Vec::new();
    collect_leaf_dirs(&profiles_dir, &mut leaf_dirs);
    leaf_dirs.sort();
    assert_eq!(leaf_dirs.len(), 2,
        "expected 2 grid points, got {}", leaf_dirs.len());

    for leaf in &leaf_dirs {
        let mle_toml = leaf.join("mle.toml");
        let body = std::fs::read_to_string(&mle_toml)
            .unwrap_or_else(|e| panic!("read {}: {}", mle_toml.display(), e));

        let final_loglik = parse_toml_finite_f64(&body, "final_loglik")
            .unwrap_or_else(|| panic!(
                "missing/non-finite final_loglik in {}:\n{}",
                mle_toml.display(), body));
        let final_log_posterior = parse_toml_finite_f64(&body, "final_log_posterior")
            .unwrap_or_else(|| panic!(
                "missing/non-finite final_log_posterior in {}:\n{}",
                mle_toml.display(), body));

        let observed_gap = final_log_posterior - final_loglik;
        // Tolerance: 1e-9. The wiring is exact-arithmetic addition
        // (loglik + nuisance_log_prior + focal_log_prior_offset) and
        // mle.toml writes f64 with `{}` formatter at full precision
        // (profile.rs:2067, 2072). PF noise does NOT enter — both
        // quantities come from the same PMMH step, so the delta is
        // exactly the prior sum by construction. The 2026-05-27
        // code-review audit flagged the previous 1e-3 tolerance as
        // looser than needed; tightening to 1e-9 catches off-by-
        // tiny-bias bugs that 1e-3 would hide.
        assert!((observed_gap - expected_gap).abs() < 1e-9,
            "log_posterior invariant broken at {}:\n  \
             final_loglik          = {:.10}\n  \
             final_log_posterior   = {:.10}\n  \
             observed gap          = {:.10}\n  \
             expected gap          = {:.10} \
             (focal Uniform[0.20,0.40] contributes -ln(0.20) ≈ 1.6094; \
              nuisance Uniform[0.10,0.35] contributes -ln(0.25) ≈ 1.3863)\n  \
             body:\n{}",
            mle_toml.display(),
            final_loglik, final_log_posterior, observed_gap, expected_gap,
            body);
    }
}

/// Sanity-check the parse helper independently so a future change
/// to mle.toml formatting fails this test first (not the
/// invariant test, which would report a confusing "missing key").
#[test]
fn parse_toml_finite_f64_handles_basic_and_nan_lines() {
    let body = "\
final_loglik = -123.456
final_log_posterior = nan
other_key = -inf
trailing_comment = 1.5  # hello
indented_key =    2.5
";
    assert_eq!(parse_toml_finite_f64(body, "final_loglik"), Some(-123.456));
    assert_eq!(parse_toml_finite_f64(body, "final_log_posterior"), None);
    assert_eq!(parse_toml_finite_f64(body, "other_key"), None);
    assert_eq!(parse_toml_finite_f64(body, "trailing_comment"), Some(1.5));
    assert_eq!(parse_toml_finite_f64(body, "indented_key"), Some(2.5));
    assert_eq!(parse_toml_finite_f64(body, "missing_key"), None);
}
