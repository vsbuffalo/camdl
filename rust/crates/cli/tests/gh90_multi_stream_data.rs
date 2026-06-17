//! gh#90: integration tests for the polymorphic `--data` flag on
//! `camdl pfilter` and `camdl profile`.
//!
//! The symptom: pre-gh#90, `camdl pfilter` / `camdl profile` against
//! a multi-block model silently scored exactly one observation block
//! (the one named by `--obs` or — for single-block models — the only
//! one). Other blocks' parameters fell back to priors and contributed
//! zero to the likelihood. Users got a plausible-looking but
//! methodologically wrong result.
//!
//! These tests pin the post-gh#90 behaviour:
//!
//! 1. Single-stream `--data PATH` regression on a single-block model
//!    keeps the legacy behaviour (the polymorphic flag accepts the
//!    single-PATH form without an `=`).
//! 2. Multi-stream `--data NAME=PATH` (repeatable) on a multi-block
//!    model: every named stream is bound, the joint loglik sums
//!    across all bound streams, and NO unbound-streams warning fires.
//! 3. `--data PATH --obs NAME` on a multi-block model: the unbound-
//!    streams warning fires naming each silently-zero block.
//! 4. CLI `--data` overrides the `--fit` toml's `[data.observations]`
//!    map: emit an info line that the CLI flags won.

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

fn multi_block_model() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/seir_spatial_5_inference.ir.json")
}

fn single_block_model() -> PathBuf {
    // sir_vaccination has one obs block `reported_cases`.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/sir_vaccination.ir.json")
}

fn synth_obs(bin: &Path, model: &Path, tmp: &Path, extra_args: &[&str]) -> PathBuf {
    let obs_path = tmp.join("obs.tsv");
    let mut cmd = Command::new(bin);
    cmd.env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", &model.to_string_lossy(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "42",
            "--obs-only", &obs_path.to_string_lossy(),
        ])
        .args(extra_args);
    let status = cmd.status().expect("spawn simulate");
    assert!(status.success(), "synthetic obs generation failed");
    obs_path
}

#[test]
fn pfilter_single_stream_data_path_regression_on_single_block_model() {
    // Pre-gh#90 form `--data PATH` (no NAME=) must still work on a
    // single-block model. This is the legacy compatibility path —
    // any breakage here would silently break every existing
    // pfilter invocation in the wild.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &single_block_model(), tmp.path(), &[]);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &single_block_model().to_string_lossy(),
            "--data", &obs.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(out.status.success(), "single-stream pfilter failed: \nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr));
    // stdout should be a finite negative loglik.
    let ll: f64 = String::from_utf8_lossy(&out.stdout).trim().parse()
        .expect("loglik parse");
    assert!(ll.is_finite() && ll < 0.0, "expected finite negative loglik, got {}", ll);
    // No unbound-streams warning: model has 1 obs block.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("silently zero"),
        "single-block model must NOT emit gh#90 warning: {}", stderr);
}

#[test]
fn pfilter_multi_stream_named_pairs_joint_scoring() {
    // gh#90 primary trap. Multi-block model, 5 streams; `--data
    // NAME=PATH` repeats bind every stream. The joint loglik should
    // be substantially more negative than a single-stream cell, and
    // NO unbound-streams warning should fire.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &format!("cases_p1={}", obs.to_string_lossy()),
            "--data", &format!("cases_p2={}", obs.to_string_lossy()),
            "--data", &format!("cases_p3={}", obs.to_string_lossy()),
            "--data", &format!("cases_p4={}", obs.to_string_lossy()),
            "--data", &format!("cases_p5={}", obs.to_string_lossy()),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(out.status.success(),
        "multi-stream pfilter failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr));
    let ll_joint: f64 = String::from_utf8_lossy(&out.stdout).trim().parse()
        .expect("joint loglik parse");
    let stderr = String::from_utf8_lossy(&out.stderr);

    // No silent-zero warning: every block bound.
    assert!(!stderr.contains("silently zero"),
        "all-streams-bound case must NOT emit gh#90 warning: {}", stderr);
    // Banner names every stream.
    for s in &["cases_p1", "cases_p2", "cases_p3", "cases_p4", "cases_p5"] {
        assert!(stderr.contains(s), "stderr should name stream {}: {}", s, stderr);
    }

    // Compare against the single-stream variant: joint loglik should
    // be at least ~3× (in magnitude) the single-stream loglik. This
    // is the same shape as the gh#38 family-root test in
    // profile_multi_stream.rs.
    let single_out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &obs.to_string_lossy(), "--obs", "cases_p1",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn single pfilter");
    let ll_single: f64 = String::from_utf8_lossy(&single_out.stdout).trim().parse()
        .expect("single loglik parse");
    let ratio = ll_joint.abs() / ll_single.abs();
    assert!(ratio >= 3.0,
        "|loglik(5-stream joint)| = {} should be ≥ 3× |loglik(single)| = {} \
         (ratio = {:.2}x). Pre-gh#90: ratio ≈ 1× (silently scored 1 stream).",
        ll_joint.abs(), ll_single.abs(), ratio);
}

#[test]
fn pfilter_single_stream_with_obs_on_multi_block_emits_warning() {
    // The intentional single-stream subset of a multi-block model.
    // Must succeed (the user explicitly named --obs) but must also
    // surface the warning so the user knows the other blocks are
    // silently zero in the likelihood.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &obs.to_string_lossy(), "--obs", "cases_p1",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(out.status.success(),
        "single-stream-on-multi-block must succeed: \n{}",
        String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("silently zero"),
        "unbound-streams warning must fire when M<N blocks bound: {}", stderr);
    // Warning names every unbound stream.
    for s in &["cases_p2", "cases_p3", "cases_p4", "cases_p5"] {
        assert!(stderr.contains(s),
            "warning should name unbound stream {}: {}", s, stderr);
    }
    // And the suggested fix.
    assert!(stderr.contains("--data NAME=PATH"),
        "warning should suggest --data NAME=PATH: {}", stderr);
}

#[test]
fn pfilter_mixed_data_forms_errors() {
    // --data PATH and --data NAME=PATH in one invocation is a hard
    // error — mixing single-stream and multi-stream forms is a
    // user-confusion smell.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &obs.to_string_lossy(),
            "--data", &format!("cases_p2={}", obs.to_string_lossy()),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(!out.status.success(), "mixed forms must hard-error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mutually exclusive"),
        "error must explain that the forms are mutually exclusive: {}", stderr);
}

#[test]
fn pfilter_multi_block_no_data_with_no_obs_errors_actionable() {
    // --data PATH (no NAME) on a multi-block model without --obs
    // is ambiguous: refuses to silently score one of N streams.
    // Error must name both --data NAME=PATH and --data PATH --obs
    // NAME as fixes.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &obs.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(!out.status.success(), "single-PATH + no --obs on multi-block must error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--data NAME=PATH"),
        "error must suggest --data NAME=PATH multi-stream form: {}", stderr);
    assert!(stderr.contains("--obs"),
        "error must mention --obs NAME single-stream form: {}", stderr);
    assert!(stderr.contains("ambiguous") || stderr.contains("refusing"),
        "error must explain the silent-failure-mode it's refusing: {}", stderr);
}

#[test]
fn pfilter_cli_data_overrides_fit_toml_when_both_supplied() {
    // CLI `--data` ALWAYS wins over a `--fit` toml's `[data.observations]`
    // section. An info line should announce the precedence so the
    // user knows which path took effect.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    // Write a minimal fit.toml with a bogus [data.observations] that
    // would fail if loaded (paths don't exist). The CLI --data should
    // override this — pfilter must still succeed, and the info line
    // about override-precedence must fire.
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{model}"

[config]
dt = 1.0

[estimate]

[fixed]

[stages]

[data.observations]
cases_p1 = "/nonexistent/should_not_be_read.tsv"
cases_p2 = "/nonexistent/should_not_be_read.tsv"
cases_p3 = "/nonexistent/should_not_be_read.tsv"
cases_p4 = "/nonexistent/should_not_be_read.tsv"
cases_p5 = "/nonexistent/should_not_be_read.tsv"
"#, model = multi_block_model().to_string_lossy())).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--data", &format!("cases_p1={}", obs.to_string_lossy()),
            "--data", &format!("cases_p2={}", obs.to_string_lossy()),
            "--data", &format!("cases_p3={}", obs.to_string_lossy()),
            "--data", &format!("cases_p4={}", obs.to_string_lossy()),
            "--data", &format!("cases_p5={}", obs.to_string_lossy()),
            "--fit", &fit_toml.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "CLI --data must win over --fit toml (the toml's paths don't \
         exist; if the fallback was consulted we'd fail at file open): \
         {}", stderr);
    // Info line announces the precedence.
    assert!(stderr.contains("--data on CLI overrides")
        || stderr.contains("overrides --fit toml"),
        "stderr should announce CLI precedence over fit toml: {}", stderr);
}

#[test]
fn pfilter_fit_toml_fallback_when_no_cli_data() {
    // The fit-toml fallback: --fit fit.toml + no CLI --data flags →
    // read multi-stream binding from [data.observations]. Same
    // joint-scoring as the named-pairs path.
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, &multi_block_model(), tmp.path(),
        &["--scenario", "true_params"]);

    let fit_toml = tmp.path().join("fit.toml");
    let obs_lossy = obs.to_string_lossy();
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{model}"

[config]
dt = 1.0

[estimate]

[fixed]

[stages]

[data.observations]
cases_p1 = "{p}"
cases_p2 = "{p}"
cases_p3 = "{p}"
cases_p4 = "{p}"
cases_p5 = "{p}"
"#, model = multi_block_model().to_string_lossy(), p = obs_lossy)).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &multi_block_model().to_string_lossy(),
            "--scenario", "true_params",
            "--fit", &fit_toml.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "fit-toml fallback should succeed:\n{}", stderr);
    // No silent-zero warning: all 5 streams bound via the toml.
    assert!(!stderr.contains("silently zero"),
        "all-streams-bound (via fit toml) must NOT emit warning: {}", stderr);
}
