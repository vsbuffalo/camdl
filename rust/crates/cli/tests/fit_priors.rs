//! Integration tests for the `camdl fit run` prior-resolution surface
//! (gh#75 — extending the gh#73 precedence chain to fit run).
//!
//! Each test drives the release binary end-to-end and asserts
//! observable behaviour:
//!
//!   1. `fit_run_with_model_ir_priors_only_succeeds` — load-bearing.
//!      Model file declares `~` priors on every estimable parameter;
//!      fit TOML declares none. `camdl fit run` exits 0 and writes
//!      a fit dir. Before gh#75 this was rejected at validation; this
//!      test pins the fix.
//!
//!   2. `fit_run_with_fit_toml_priors_only_succeeds` — regression
//!      guard for the pre-existing behaviour where the fit TOML
//!      supplies every prior.
//!
//!   3. `fit_run_with_mixed_priors_succeeds_and_sources_correctly` —
//!      one param from fit TOML, one from model IR, both apply; the
//!      run's `FitMeta.resolved_priors` array names each parameter's
//!      source.
//!
//!   4. `fit_run_with_no_priors_anywhere_fails_at_config_load` —
//!      neither source has priors; the run exits non-zero before any
//!      fit work happens and the error message names every offending
//!      parameter and lists three remedies (model `~`, fit-toml
//!      `prior`, explicit `prior = { flat = {} }`).
//!
//!   5. `fit_run_with_explicit_flat_prior_succeeds_without_warning`
//!      — fit TOML supplies `prior = { flat = {} }` for every
//!      estimated parameter; the fit succeeds; provenance records
//!      `flat_explicit` for each parameter; no warning emitted.
//!
//!   6. `fit_run_priors_cache_invalidates_on_model_ir_prior_change` —
//!      run fit twice against a model whose `~` prior changes between
//!      runs; assert the produced fit dir (CAS hash) differs.
//!
//! Tests skip silently when the release binary or the `camdlc`
//! compiler isn't present (mirrors the rest of the integration
//! suite). The fixture is a tiny SIR with Poisson observations; the
//! fits use PMMH with a small particle / sweep count so they finish
//! in seconds.

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
        "camdl_fit_priors_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Build a tiny SIR-with-Poisson-cases fixture. The DSL source has a
/// configurable `~` prior block — pass `with_tilde_priors = true` to
/// emit `~ log_normal(...)` declarations for every estimable param,
/// `false` for the priors-only-in-fit-toml case, and use
/// `tilde_priors_variant` to produce a *different* set of `~` priors
/// to exercise the CAS-hash-invalidation test.
fn write_fixture(dir: &Path, kind: TildeMode) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let prior_lines = match kind {
        TildeMode::None => (String::new(), String::new()),
        TildeMode::Default => (
            "~ log_normal(mu = -0.3, sigma = 0.5)".to_string(),
            "~ log_normal(mu = -1.2, sigma = 0.5)".to_string(),
        ),
        TildeMode::Variant => (
            // Same shape, different params → IR JSON differs →
            // model identity differs → fit-level digest differs.
            "~ log_normal(mu = -1.0, sigma = 0.3)".to_string(),
            "~ log_normal(mu = -1.5, sigma = 0.3)".to_string(),
        ),
    };
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0] {beta_prior}
  gamma : rate  in [0.01, 1.0]  {gamma_prior}
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = projected)
  }}
}}
scenarios {{
  baseline {{
    set = {{
      beta  = 0.3
      gamma = 0.1
      N0    = 1000
    }}
  }}
}}
init {{ S = 999  I = 1 }}
simulate {{ from = 0 'days  to = 6 'days }}
"#,
        beta_prior  = prior_lines.0,
        gamma_prior = prior_lines.1,
    );
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

#[derive(Copy, Clone)]
enum TildeMode {
    /// No `~` priors declared in the model — fit TOML is the only source.
    None,
    /// `~ log_normal(mu = -0.3, sigma = 0.5)` etc. — the standard test fixture.
    Default,
    /// A different set of `~` log_normal parameters; same families.
    /// Used for the hash-invalidation test.
    Variant,
}

/// Variants on the fit TOML's `[estimate]` block. Same `[stages.posterior]`
/// (PMMH, chain_binomial, tiny particle count) across all of them.
#[derive(Copy, Clone)]
enum FitTomlMode {
    /// `[estimate.<param>]` has no `prior = ...`; relies entirely on
    /// the model's IR fallback. Both estimated params (beta + gamma)
    /// covered by this mode.
    NoPriors,
    /// Both estimated params declare a `prior = { log_normal = ... }` field.
    BothPriors,
    /// `beta` has a fit-TOML prior; `gamma` does not (so `gamma` falls
    /// through to the IR if the model declares it, else falls through to
    /// the missing-prior error).
    BetaOnly,
    /// Both estimated params declare an explicit `prior = { flat = {} }`.
    BothFlat,
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, mode: FitTomlMode, tag: &str) -> PathBuf {
    let (beta_field, gamma_field) = match mode {
        FitTomlMode::NoPriors => (
            "{ bounds = [0.01, 5.0], start = 0.4 }".to_string(),
            "{ bounds = [0.01, 1.0], start = 0.15 }".to_string(),
        ),
        FitTomlMode::BothPriors => (
            "{ bounds = [0.01, 5.0], start = 0.4, \
              prior = { log_normal = { mu = -0.3, sigma = 0.5 } } }".to_string(),
            "{ bounds = [0.01, 1.0], start = 0.15, \
              prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }".to_string(),
        ),
        FitTomlMode::BetaOnly => (
            "{ bounds = [0.01, 5.0], start = 0.4, \
              prior = { log_normal = { mu = -0.3, sigma = 0.5 } } }".to_string(),
            "{ bounds = [0.01, 1.0], start = 0.15 }".to_string(),
        ),
        FitTomlMode::BothFlat => (
            "{ bounds = [0.01, 5.0], start = 0.4, prior = { flat = {} } }".to_string(),
            "{ bounds = [0.01, 1.0], start = 0.15, prior = { flat = {} } }".to_string(),
        ),
    };
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {beta_field}
gamma = {gamma_field}
[fixed]
N0 = 1000
[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 20
iterations = 20
# Tiny burn_in so the post-burn-in sample set is non-empty.
burn_in = 2
"#,
        out  = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join(format!("fit_{}.toml", tag));
    std::fs::write(&p, toml).unwrap();
    p
}

/// Locate the single fit dir under `<out_root>/fits/` produced by a
/// successful `camdl fit run`.
fn find_fit_dir(out_root: &Path) -> PathBuf {
    let fits = out_root.join("fits");
    let entries: Vec<_> = std::fs::read_dir(&fits)
        .unwrap_or_else(|e| panic!("read_dir {}: {}", fits.display(), e))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    assert_eq!(entries.len(), 1,
        "expected exactly one fit dir under {}, found {:?}",
        fits.display(), entries);
    entries.into_iter().next().unwrap()
}

/// Read the `resolved_priors` array from a fit's provenance sidecar
/// (`fit.meta.json`). gh#147 (M3.2): a CAS fit has no fit-wide `run.json`; the
/// gh#75 per-parameter prior-source provenance lives in the fit-level sidecar
/// (`run_meta::FitSidecar`), at the top level (flattened FitMeta provenance).
fn read_resolved_priors(fit_dir: &Path) -> Vec<serde_json::Value> {
    let body = std::fs::read_to_string(fit_dir.join("fit.meta.json"))
        .unwrap_or_else(|e| panic!("read_to_string fit.meta.json: {}", e));
    let v: serde_json::Value = serde_json::from_str(&body)
        .unwrap_or_else(|e| panic!("parse fit.meta.json: {}", e));
    v.get("resolved_priors")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_else(|| panic!("resolved_priors array missing in fit.meta.json: {}", v))
}

fn run_fit(bin: &Path, fit_toml: &Path) -> std::process::Output {
    Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", "1"])
        .output()
        .expect("spawn camdl fit run")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Load-bearing test for gh#75: priors live ONLY in the model file.
/// Before gh#75 the validator at config_v2.rs rejected this with
/// "stage 'posterior' (method=pmmh) requires priors, but missing for:
/// beta, gamma". After gh#75 the fit succeeds and the resolver pulls
/// the priors from the model IR.
#[test]
fn fit_run_with_model_ir_priors_only_succeeds() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("ir_only");
    let (ir, data) = write_fixture(tmp.path(), TildeMode::Default);
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, FitTomlMode::NoPriors, "ir_only");

    let out = run_fit(&bin, &fit_toml);
    assert!(out.status.success(),
        "fit run with model-IR-only priors must succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    let out_root = tmp.path().join("results");
    let fit_dir = find_fit_dir(&out_root);

    // Provenance: every estimated param recorded as `model_ir`.
    let resolved = read_resolved_priors(&fit_dir);
    let lookup = |param: &str| -> &str {
        resolved.iter().find(|e| {
            e.get("param").and_then(|p| p.as_str()) == Some(param)
        }).unwrap_or_else(|| panic!("param {} in resolved_priors", param))
          .get("source").and_then(|s| s.as_str()).unwrap_or("")
    };
    assert_eq!(lookup("beta"),  "model_ir",
        "beta must be sourced from model_ir, got resolved_priors={}", resolved.iter().collect::<Vec<_>>().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
    assert_eq!(lookup("gamma"), "model_ir",
        "gamma must be sourced from model_ir, got resolved_priors={}", resolved.iter().collect::<Vec<_>>().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
}

/// Regression guard: when the fit TOML supplies a prior for every
/// estimated parameter, the run succeeds and `resolved_priors`
/// records `fit_toml` as the source for each. This was the
/// pre-gh#75 behaviour and stays unchanged.
#[test]
fn fit_run_with_fit_toml_priors_only_succeeds() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("toml_only");
    // No `~` priors in the model — only fit-toml priors.
    let (ir, data) = write_fixture(tmp.path(), TildeMode::None);
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, FitTomlMode::BothPriors, "toml_only");

    let out = run_fit(&bin, &fit_toml);
    assert!(out.status.success(),
        "fit run with fit-toml-only priors must succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    let fit_dir = find_fit_dir(&tmp.path().join("results"));
    let resolved = read_resolved_priors(&fit_dir);
    let lookup = |param: &str| -> &str {
        resolved.iter().find(|e| {
            e.get("param").and_then(|p| p.as_str()) == Some(param)
        }).unwrap_or_else(|| panic!("param {}", param))
          .get("source").and_then(|s| s.as_str()).unwrap_or("")
    };
    assert_eq!(lookup("beta"),  "fit_toml");
    assert_eq!(lookup("gamma"), "fit_toml");
}

/// Mixed sources: `beta` from fit TOML, `gamma` from model IR. Both
/// applied; both surfaced in `resolved_priors` with the correct
/// source labels.
#[test]
fn fit_run_with_mixed_priors_succeeds_and_sources_correctly() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("mixed");
    let (ir, data) = write_fixture(tmp.path(), TildeMode::Default);
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, FitTomlMode::BetaOnly, "mixed");

    let out = run_fit(&bin, &fit_toml);
    assert!(out.status.success(),
        "mixed-source fit run must succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    let fit_dir = find_fit_dir(&tmp.path().join("results"));
    let resolved = read_resolved_priors(&fit_dir);
    let lookup = |param: &str| -> &str {
        resolved.iter().find(|e| {
            e.get("param").and_then(|p| p.as_str()) == Some(param)
        }).unwrap_or_else(|| panic!("param {}", param))
          .get("source").and_then(|s| s.as_str()).unwrap_or("")
    };
    assert_eq!(lookup("beta"),  "fit_toml",
        "beta should be sourced from fit_toml (declared inline)");
    assert_eq!(lookup("gamma"), "model_ir",
        "gamma should be sourced from model_ir (`~` syntax)");
}

/// Validation error: neither source declares priors. Must exit
/// non-zero BEFORE the fit dir is populated, and the error must
/// (a) name every offending parameter and (b) list the three
/// remedies: model `~`, fit-toml `prior`, explicit `prior = { flat = {} }`.
#[test]
fn fit_run_with_no_priors_anywhere_fails_at_config_load() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("nothing");
    // No `~` priors in the model AND no prior= in the fit TOML.
    let (ir, data) = write_fixture(tmp.path(), TildeMode::None);
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, FitTomlMode::NoPriors, "nothing");

    let out = run_fit(&bin, &fit_toml);
    assert!(!out.status.success(),
        "fit run with no priors anywhere must fail; got success with stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    let stderr = String::from_utf8_lossy(&out.stderr);
    // Error names every offending parameter.
    assert!(stderr.contains("beta"),
        "error must name beta as missing-prior; got:\n{}", stderr);
    assert!(stderr.contains("gamma"),
        "error must name gamma as missing-prior; got:\n{}", stderr);
    // Three remedies surfaced.
    assert!(stderr.contains("model file") || stderr.contains("`~`"),
        "error must mention model-file `~` syntax as remedy (i); got:\n{}", stderr);
    assert!(stderr.contains("prior = {") || stderr.contains("[estimate."),
        "error must mention fit-toml `prior = ...` as remedy (ii); got:\n{}", stderr);
    assert!(stderr.contains("flat") || stderr.contains("Flat"),
        "error must mention explicit flat priors as remedy (iii); got:\n{}", stderr);

    // No fit dir produced (or, if a dir was created, it must NOT have
    // a populated run.json — we want the failure to land at config
    // validation before any fit work happens).
    let fits_dir = tmp.path().join("results/fits");
    if fits_dir.exists() {
        let entries: Vec<_> = std::fs::read_dir(&fits_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path()).collect();
        for d in &entries {
            // A "running" run.json that flips status by an error counts
            // as work-having-happened; assert we never even got that far.
            assert!(!d.join("run.json").exists(),
                "validation failure must surface BEFORE any fit dir is populated, \
                 but {} exists", d.join("run.json").display());
        }
    }
}

/// Explicit opt-in path: `prior = { flat = {} }` in the fit TOML is
/// accepted, the fit proceeds, and `resolved_priors` records the
/// source as `flat_explicit`. No warning emitted.
#[test]
fn fit_run_with_explicit_flat_prior_succeeds_without_warning() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("flat_explicit");
    // Model has no `~` priors; fit TOML opts in explicitly to flat.
    let (ir, data) = write_fixture(tmp.path(), TildeMode::None);
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, FitTomlMode::BothFlat, "flat_explicit");

    let out = run_fit(&bin, &fit_toml);
    assert!(out.status.success(),
        "explicit `prior = {{ flat = {{}} }}` must succeed; stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    let stderr = String::from_utf8_lossy(&out.stderr);
    // No "flat priors" warning when the user opted in explicitly.
    // The pre-gh#73 surface emitted `warning: ... flat priors ...` from
    // `format_flat_fallback_warning`; that text exact-matches "flat priors"
    // and is what we must NOT see. (We grep for the precise phrase rather
    // than "flat" alone because unrelated diagnostics — fit-temp-dir paths
    // containing `fit_priors`, post-fit warnings about `mle_params.toml` —
    // legitimately exist in stderr without indicating a prior-warning fire.)
    assert!(!stderr.contains("flat priors"),
        "explicit flat opt-in must not trigger the gh#73 \
         flat-priors warning text; got stderr:\n{}", stderr);
    // Also defensively check the warning's standard prefix isn't paired
    // with the "flat" keyword on the same line.
    for line in stderr.lines() {
        let is_warning = line.contains("warning:") || line.contains("⚠");
        let mentions_flat_prior = line.contains("flat prior") || line.contains("flat-prior");
        assert!(!(is_warning && mentions_flat_prior),
            "warning line mentioning flat priors must not fire on the \
             explicit opt-in path; got line:\n  {}", line);
    }

    let fit_dir = find_fit_dir(&tmp.path().join("results"));
    let resolved = read_resolved_priors(&fit_dir);
    let lookup = |param: &str| -> &str {
        resolved.iter().find(|e| {
            e.get("param").and_then(|p| p.as_str()) == Some(param)
        }).unwrap_or_else(|| panic!("param {}", param))
          .get("source").and_then(|s| s.as_str()).unwrap_or("")
    };
    assert_eq!(lookup("beta"),  "flat_explicit",
        "beta source must be flat_explicit");
    assert_eq!(lookup("gamma"), "flat_explicit",
        "gamma source must be flat_explicit");
}

/// CAS-hash invalidation: changing the model IR's `~` prior must
/// produce a different fit dir. The fit-level digest keys on the model
/// content (via `FitDigest.model`), so an IR-prior edit re-keys the fit;
/// this test pins that the same chain continues to hold once the
/// resolver wires the IR-prior into the fit cache key path.
#[test]
fn fit_run_priors_cache_invalidates_on_model_ir_prior_change() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("cas_invalidate");

    // Run A: model declares `~ log_normal(mu = -0.3, sigma = 0.5)`.
    let dir_a = tmp.path().join("a");
    std::fs::create_dir_all(&dir_a).unwrap();
    let (ir_a, data_a) = write_fixture(&dir_a, TildeMode::Default);
    let fit_a = write_fit_toml(&dir_a, &ir_a, &data_a, FitTomlMode::NoPriors, "a");
    let out_a = run_fit(&bin, &fit_a);
    assert!(out_a.status.success(),
        "run A must succeed; stderr=\n{}", String::from_utf8_lossy(&out_a.stderr));
    let fit_dir_a = find_fit_dir(&dir_a.join("results"));
    let name_a = fit_dir_a.file_name().unwrap().to_owned();

    // Run B: same fit TOML, but the model file's `~` priors are
    // changed (different mu/sigma values).
    let dir_b = tmp.path().join("b");
    std::fs::create_dir_all(&dir_b).unwrap();
    let (ir_b, data_b) = write_fixture(&dir_b, TildeMode::Variant);
    let fit_b = write_fit_toml(&dir_b, &ir_b, &data_b, FitTomlMode::NoPriors, "b");
    let out_b = run_fit(&bin, &fit_b);
    assert!(out_b.status.success(),
        "run B must succeed; stderr=\n{}", String::from_utf8_lossy(&out_b.stderr));
    let fit_dir_b = find_fit_dir(&dir_b.join("results"));
    let name_b = fit_dir_b.file_name().unwrap().to_owned();

    assert_ne!(name_a, name_b,
        "changing the model's `~` prior must produce a different fit dir \
         (CAS cache must invalidate). A={:?} B={:?}", name_a, name_b);
}
