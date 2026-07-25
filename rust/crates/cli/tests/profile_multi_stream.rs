//! gh#38: integration tests for `camdl profile` against indexed
//! observation families.
//!
//! Symptom the fix targets: previously, `camdl profile --obs cases`
//! on a model whose IR had 5 expanded `cases_p1`...`cases_p5`
//! streams scored only the first stream and reported a loglik ~5
//! orders of magnitude smaller (in absolute value) than the joint
//! likelihood `camdl fit run` was optimising. Profile-likelihood
//! plots derived from this output were not commensurate with fit
//! summaries.
//!
//! These tests assert the post-fix behaviour:
//!
//! 1. Family-name resolution: `--obs <root>` against a multi-stream
//!    IR produces a loglik whose magnitude is the sum across all
//!    expanded streams (single-stream `--obs <leaf>` produces ~1/N
//!    of the magnitude when N streams expand from the family).
//! 2. Single-stream profile (one IR observation, no family) keeps
//!    its prior behaviour.
//! 3. Default behaviour for a multi-stream model with `--obs`
//!    omitted is a hard error listing the available stream names —
//!    no silent fall-back to the first IR observation.

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

/// 5-patch SEIR with five neg_binomial obs streams `cases_p1`...
/// `cases_p5` sharing family root `cases`. Used here as a stand-in
/// for an indexed `cases[s,a]` family — the expander emits the same
/// `<family>_<index>` naming convention, so the resolution path is
/// identical.
fn seir_spatial_5_inference() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/seir_spatial_5_inference.ir.json")
}

/// Generate a synthetic multi-stream observations TSV from the model
/// at known parameter values. Returns the path; caller cleans up via
/// the surrounding tempdir.
fn synth_obs_tsv(bin: &Path, tmp: &Path) -> PathBuf {
    let obs_path = tmp.join("seir_obs.tsv");
    let status = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", &seir_spatial_5_inference().to_string_lossy(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "42",
            "--scenario", "true_params",
            "--obs-only", &obs_path.to_string_lossy(),
        ])
        .status()
        .expect("spawn camdl simulate");
    assert!(status.success(), "synthetic obs generation failed");
    assert!(obs_path.exists(), "obs TSV not written");
    obs_path
}

/// Collect the per-grid-point best loglik from the new-format
/// `ProfilePoint` leaves under `<out_root>/profiles/`, ordered by the
/// `point`-level label (so the returned Vec is grid-traversal order). The
/// cross-point rollup TSV is the deferred M4 derived view (gh#154); the
/// per-point loglik lives on each leaf's recorded `inputs.best_loglik`.
fn collect_logliks(out_root: &Path) -> Vec<f64> {
    fn walk(dir: &Path, out: &mut Vec<(String, f64)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("profile_point") {
                    let point = v.get("levels").and_then(|ls| ls.as_array())
                        .and_then(|a| a.iter().find(|l|
                            l.get("name").and_then(|n| n.as_str()) == Some("point")))
                        .and_then(|l| l.get("label").and_then(|x| x.as_str()))
                        .unwrap_or("").to_string();
                    if let Some(ll) = v.get("inputs")
                        .and_then(|i| i.get("best_loglik")).and_then(|x| x.as_f64()) {
                        out.push((point, ll));
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), out); } }
        }
    }
    let mut pairs = Vec::new();
    walk(&out_root.join("profiles"), &mut pairs);
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs.into_iter().map(|(_, ll)| ll).collect()
}

/// Run `camdl profile` once and return the parsed loglik values.
fn run_profile(
    bin: &Path,
    output_root: &Path,
    obs_arg: &str,
    data_path: &Path,
    out_tsv: &Path,
) -> Vec<f64> {
    let status = Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", output_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_spatial_5_inference().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &data_path.to_string_lossy(),
            "--obs", obs_arg,
            "--sweep", "R0=lin(15,25,2)",
            // 500, not 100. The sweep straddles the truth (the `true_params`
            // scenario has R0 = 20), so the R0 = 15 cell is scored well off the
            // optimum, and a 100-particle bootstrap filter on this 5-patch
            // spatial SEIR loses the whole swarm there and returns −inf. That
            // serialises as `"best_loglik": null` (JSON has no −inf), so the
            // grid point vanishes from `collect_logliks` and the test fails
            // with "expected 2 grid points" rather than on anything it means
            // to check.
            //
            // The cell is NOT genuinely ruled out — it is under-sampled.
            // Measured on this fixture, R0 = 15's loglik by particle count:
            //     100 → null  (−inf; the swarm dies after ~8 ms)
            //     500 → −1876.13
            //    2000 → −1610.70
            // 500 is the smallest of those that scores every cell. This gives
            // the fixture enough particles to answer the question; every
            // assertion below is unchanged.
            "--particles", "500", "--iterations", "1", "--starts", "1",
            "--rw-sd", "auto",
            "--fixed", "sigma=0.125", "--fixed", "gamma=0.2",
            "--fixed", "kappa=0.05", "--fixed", "amplitude=0.3",
            "--fixed", "iota=1e-06", "--fixed", "rho=0.4",
            "--fixed", "sigma_se=0.05", "--fixed", "k=10.0",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .status()
        .expect("spawn camdl profile");
    assert!(status.success(), "profile run failed for --obs {}", obs_arg);
    let _ = out_tsv; // --output is accepted; the rollup TSV is the deferred M4 view.
    collect_logliks(output_root)
}

#[test]
fn profile_family_root_sums_all_expanded_streams() {
    // Core gh#38 regression test: `--obs cases` resolves to all 5
    // expanded streams (`cases_p1`...`cases_p5`) and the reported
    // loglik must be the joint sum, not the first-stream-only value.
    //
    // Concretely we expect the magnitude (|loglik|) under
    // `--obs cases` to be substantially larger than under
    // `--obs cases_p1`. With a uniform-ish 5-stream split we'd
    // expect a factor of ~5; we assert a much weaker lower bound
    // (≥3×) to stay robust to per-stream variation under stochastic
    // IF2 with iterations=1.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data_path = synth_obs_tsv(&bin, tmp.path());

    let cases_dir = tmp.path().join("out_cases");
    let p1_dir    = tmp.path().join("out_cases_p1");
    let cases_tsv = tmp.path().join("profile_cases.tsv");
    let p1_tsv    = tmp.path().join("profile_p1.tsv");

    let ll_family = run_profile(&bin, &cases_dir, "cases", &data_path, &cases_tsv);
    let ll_single = run_profile(&bin, &p1_dir,    "cases_p1", &data_path, &p1_tsv);

    assert_eq!(ll_family.len(), 2, "expected 2 grid points, got {:?}", ll_family);
    assert_eq!(ll_single.len(), 2, "expected 2 grid points, got {:?}", ll_single);

    // Both families should produce finite, negative logliks.
    for (i, ll) in ll_family.iter().enumerate() {
        assert!(ll.is_finite() && *ll < 0.0,
            "multi-stream loglik at grid {} not a finite negative: {}", i, ll);
    }
    for (i, ll) in ll_single.iter().enumerate() {
        assert!(ll.is_finite() && *ll < 0.0,
            "single-stream loglik at grid {} not a finite negative: {}", i, ll);
    }

    // Magnitude check: |ll_family| should be ≥3× |ll_single|. Before
    // the fix, family resolution silently scored only the first IR
    // observation (cases_p1), so the family loglik equalled the
    // single-stream loglik (ratio ≈ 1×).
    for (i, (lf, ls)) in ll_family.iter().zip(ll_single.iter()).enumerate() {
        let ratio = lf.abs() / ls.abs();
        assert!(ratio >= 3.0,
            "grid {}: |loglik(family)| = {} should be ≥ 3× |loglik(single)| = {} \
             (ratio = {:.2}x). Pre-fix: ratio ≈ 1× (silent first-stream-only).",
            i, lf.abs(), ls.abs(), ratio);
    }
}

#[test]
fn profile_multi_stream_model_requires_explicit_obs() {
    // `--obs` omitted on a multi-stream IR must hard-error, not
    // silently default to the first stream. The error message must
    // list the available streams so the user knows the family root
    // (or one specific stream) to pass.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data_path = synth_obs_tsv(&bin, tmp.path());
    let out_dir = tmp.path().join("out_no_obs");
    let out_tsv = tmp.path().join("profile_no_obs.tsv");

    let output = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_spatial_5_inference().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &data_path.to_string_lossy(),
            "--sweep", "R0=lin(15,25,2)",
            "--particles", "100", "--iterations", "1", "--starts", "1",
            "--rw-sd", "auto",
            "--fixed", "sigma=0.125", "--fixed", "gamma=0.2",
            "--fixed", "kappa=0.05", "--fixed", "amplitude=0.3",
            "--fixed", "iota=1e-06", "--fixed", "rho=0.4",
            "--fixed", "sigma_se=0.05", "--fixed", "k=10.0",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .output()
        .expect("spawn camdl profile");
    assert!(!output.status.success(),
        "profile must fail without --obs on multi-stream model");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // gh#90: error now actionable — names both the --data NAME=PATH
    // and --data PATH --obs NAME paths the user can take.
    assert!(stderr.contains("--obs"),
        "error must guide the user to pass --obs: {}", stderr);
    assert!(stderr.contains("--data NAME=PATH"),
        "error must suggest --data NAME=PATH multi-stream form: {}", stderr);
    assert!(stderr.contains("cases_p1"),
        "error must list at least one available stream: {}", stderr);
}
