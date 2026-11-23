//! gh#38: integration tests for `camdl profile` against indexed
//! observation families.
//!
//! A fit file whose `[data.observations]` binds every stream of an indexed
//! family (`cases_p1`...`cases_p5`) makes a profile score the joint likelihood
//! `camdl fit run` optimises. Binding one stream scores that stream alone, and
//! about 1/N of the magnitude when N streams expand from the family.

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
    let status = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", tmp.join("camdl-store"))
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
    // The `true_params` values of the parameters the profile estimates, as the
    // base point they start from. A scenario that set them would be refused:
    // a parameter is estimated or given a value.
    std::fs::write(
        start_file(&obs_path),
        "N0_p1 = 100000\nN0_p2 = 80000\nN0_p3 = 60000\nN0_p4 = 50000\nN0_p5 = 150000\n",
    ).unwrap();
    obs_path
}

/// `--starts from_params=<start.toml>` beside the observations TSV.
fn starts_arg(data_path: &Path) -> String {
    format!("from_params={}", start_file(data_path).to_string_lossy())
}

fn start_file(data_path: &Path) -> PathBuf {
    data_path.with_file_name("start.toml")
}

/// The best loglik of each profile point stored under `store`, ordered by the
/// focal `R0` value (read from each point's `identity.json`), so the returned
/// Vec is grid-traversal order. The per-point loglik is `best_loglik` in the
/// point's `provenance.json`; a point whose filter returned -inf serialises it
/// as null and is omitted.
fn collect_logliks(store: &Path) -> Vec<f64> {
    fn walk(dir: &Path, out: &mut Vec<(f64, f64)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("identity.json")) {
            let v: serde_json::Value = serde_json::from_str(&b).unwrap();
            if v["kind"] == "profile_point" {
                let focal = v["identity"]["focal"]["R0"].as_f64().expect("focal R0");
                let prov: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("provenance.json")).unwrap()).unwrap();
                if let Some(ll) = prov["best_loglik"].as_f64() {
                    out.push((focal, ll));
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() {
                if e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                    walk(&e.path(), out);
                }
            }
        }
    }
    let mut pairs = Vec::new();
    walk(store, &mut pairs);
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs.into_iter().map(|(_, ll)| ll).collect()
}

/// Write the fit problem beside `data_path` that binds the observations file to
/// each of `streams`, estimates `R0` (swept by the profile)
/// and the five `N0_p*`, and holds the other parameters. Returns its path.
fn write_fit_file(data_path: &Path, label: &str, streams: &[&str]) -> PathBuf {
    let path = data_path.with_file_name(format!("fit_{label}.toml"));
    let bound: String = streams.iter()
        .map(|s| format!("{s} = \"{}\"\n", data_path.file_name().unwrap().to_string_lossy()))
        .collect();
    let model = seir_spatial_5_inference();
    std::fs::write(&path, format!(
        "[model]\ncamdl = \"{model}\"\n\n[data.observations]\n{bound}\n\
         [estimate]\nR0 = {{ bounds = [5.0, 80.0] }}\n\
         N0_p1 = {{ bounds = [1000.0, 500000.0] }}\n\
         N0_p2 = {{ bounds = [1000.0, 500000.0] }}\n\
         N0_p3 = {{ bounds = [1000.0, 500000.0] }}\n\
         N0_p4 = {{ bounds = [1000.0, 500000.0] }}\n\
         N0_p5 = {{ bounds = [1000.0, 500000.0] }}\n\n\
         [fixed]\nsigma = 0.125\ngamma = 0.2\nkappa = 0.05\namplitude = 0.3\n\
         iota = 1e-06\nrho = 0.4\nsigma_se = 0.05\nk = 10.0\n",
        model = model.display(),
    )).unwrap();
    path
}

/// Run `camdl profile` once and return the parsed loglik values.
fn run_profile(
    bin: &Path,
    output_root: &Path,
    streams: &[&str],
    data_path: &Path,
    out_tsv: &Path,
) -> Vec<f64> {
    let status = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", output_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &seir_spatial_5_inference().to_string_lossy(),
            "--starts", &starts_arg(data_path),
            "--fit", &write_fit_file(data_path, &streams.join("_"), streams).to_string_lossy(),
            "--sweep", "R0=lin(15,25,2)",
            // 500, not 100. The sweep straddles the truth (R0 = 20 in the
            // `true_params` scenario that generated the data), so the R0 = 15 cell is scored well off the
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
            "--particles", "500", "--iterations", "1", "--n-starts", "1",
            "--rw-sd", "auto",
            "--output", &out_tsv.to_string_lossy(),
            "--seed", "1",
        ])
        .status()
        .expect("spawn camdl profile");
    assert!(status.success(), "profile run failed binding {:?}", streams);
    let _ = out_tsv; // --output is accepted; the rollup TSV is the deferred M4 view.
    collect_logliks(output_root)
}

#[test]
fn profile_family_root_sums_all_expanded_streams() {
    // Core gh#38 regression test: the `cases` key binds all 5
    // expanded streams (`cases_p1`...`cases_p5`) and the reported
    // loglik must be the joint sum, not the first-stream-only value.
    //
    // Concretely we expect the magnitude (|loglik|) under the
    // `cases` key to be substantially larger than under the
    // `cases_p1` key. With a uniform-ish 5-stream split we'd
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

    let ll_family = run_profile(&bin, &cases_dir, &["cases_p1", "cases_p2", "cases_p3", "cases_p4", "cases_p5"], &data_path, &cases_tsv);
    let ll_single = run_profile(&bin, &p1_dir, &["cases_p1"], &data_path, &p1_tsv);

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
