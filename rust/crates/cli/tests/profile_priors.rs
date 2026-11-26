//! Integration tests for the `camdl profile` prior-provenance surface
//! (gh#73). Profile cells are maximizations that read no prior, but the
//! per-parameter prior source is recorded in each stored point's
//! `provenance.json`, and the priors are part of the problem the point's run
//! id is computed from. Each test drives the release binary end-to-end.
//!
//!   1. `profile_without_priors_records_flat_fallback`: neither the model
//!      nor the fit file declares a `~` prior; no prior warning on stderr;
//!      the record gives the estimated param the source `flat_fallback`.
//!   2. `profile_with_fit_toml_records_fit_toml_priors`: a fit toml
//!      supplies priors; the record gives source = `fit_toml`.
//!   3. `same_model_different_fit_toml_different_cas_dir`: two distinct fit
//!      tomls produce disjoint run ids.
//!   4. `same_model_flat_fit_vs_prior_fit_different_cas_dir`: the same with a
//!      fit file that declares no priors as one of the variants.
//!
//! Skipped when the `camdlc` compiler isn't present.

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
        "camdl_profile_priors_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Build a tiny SIR-with-Poisson-cases fixture. Two estimated params
/// (`beta`, `gamma`); `N0` is fixed via the toml or CLI as needed.
/// No `~` priors in the model file so the resolver's flat-fallback
/// case fires when `--fit` is absent.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    // `start.toml` gives the parameters values for the test that passes
    // `--starts from_params`.
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
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    std::fs::write(dir.join("start.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 1000\n").unwrap();

    (ir_path, data_path)
}

/// Write a fit toml with [estimate] containing log_normal priors for
/// every estimated param. Profile reads the problem half through
/// `Problem::load` — `[estimate]` and `[fixed]` for prior / bounds
/// resolution — and never runs the `[method]`, which is kept here so the
/// same file also serves `fit run`.
fn write_fit_toml_with_priors(dir: &Path, ir: &Path, data: &Path, name: &str) -> PathBuf {
    let toml = format!(r#"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join(format!("{}.toml", name));
    std::fs::write(&p, toml).unwrap();
    p
}

/// Write a fit toml that declares no prior: `beta` and `gamma` are estimated
/// at the model's bounds, `N0` is held.
fn write_fit_toml_flat(dir: &Path, ir: &Path, data: &Path) -> PathBuf {
    let toml = format!(r#"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.001, 5.0] }}
gamma = {{ bounds = [0.01, 1.0] }}
[fixed]
N0 = 1000
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join("fit_flat.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Variant: distinct prior parameters from the baseline fit toml.
/// Used by the CAS-hash test — same model + data + bounds, different
/// priors must produce a different CAS dir.
fn write_fit_toml_with_priors_variant(dir: &Path, ir: &Path, data: &Path) -> PathBuf {
    let toml = format!(r#"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
# Same shape, different mu/sigma.
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -1.0, sigma = 0.3 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.5, sigma = 0.3 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join("fit_variant.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Every stored profile point under `store`: its folder and `identity.json`.
/// Directories whose own name starts with `.` (the invocation log) are skipped.
fn profile_points(store: &Path) -> Vec<(PathBuf, serde_json::Value)> {
    let mut found = Vec::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(body) = std::fs::read_to_string(d.join("identity.json")) {
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            if v["kind"] == "profile_point" { found.push((d.clone(), v)); continue; }
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let hidden = e.file_name().to_string_lossy().starts_with('.');
            if p.is_dir() && !hidden { stack.push(p); }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// The run ids of every profile point under `store`, sorted.
fn point_run_ids(store: &Path) -> Vec<String> {
    let mut ids: Vec<String> = profile_points(store).iter()
        .map(|(_, v)| v["run_id"].as_str().unwrap().to_string()).collect();
    ids.sort();
    ids
}

/// The first point's `provenance.json` (the record of how the point's inputs
/// were resolved: priors, parameter provenance, init provenance).
fn read_sidecar(store: &Path) -> serde_json::Value {
    let points = profile_points(store);
    assert!(!points.is_empty(), "no profile point under {}", store.display());
    let body = std::fs::read_to_string(points[0].0.join("provenance.json"))
        .unwrap_or_else(|e| panic!("read provenance.json: {}", e));
    serde_json::from_str::<serde_json::Value>(&body).unwrap()
}

fn run_profile(
    bin: &Path,
    out_root: &Path,
    ir: &Path,
    extra_args: &[&str],
) -> std::process::Output {
    std::fs::create_dir_all(out_root).unwrap();
    let out_tsv = out_root.join("profile.tsv");
    let mut args: Vec<String> = vec![
        "profile".into(), ir.to_string_lossy().into_owned(),
        "--sweep".into(), "beta=lin(0.2,0.4,2)".into(),
        "--algorithm".into(), "if2".into(),
        "--particles".into(), "30".into(),
        "--iterations".into(), "5".into(),
        "--n-starts".into(), "1".into(),
        "--rw-sd".into(), "auto".into(),
        "--output".into(), out_tsv.to_string_lossy().into_owned(),
        "--seed".into(), "1".into(),
    ];
    for a in extra_args { args.push((*a).into()); }
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", out_root.join("store"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args.iter().map(|s| s.as_str()))
        .output()
        .expect("spawn camdl profile")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn profile_without_priors_records_flat_fallback() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("flat_warning");
    let (ir, data) = write_fixture(tmp.path());
    let flat = write_fit_toml_flat(tmp.path(), &ir, &data);
    let out_root = tmp.path().join("out_flat");

    let output = run_profile(&bin, &out_root, &ir, &["--fit", &flat.to_string_lossy()]);
    assert!(output.status.success(),
        "profile run failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);

    // The cells maximize the likelihood and read no prior, so a flat
    // fallback is not something to warn about.
    assert!(!stderr.contains("flat priors"),
        "profile must not warn about flat priors, got:\n{}", stderr);

    // The profile-base sidecar must record sources = "flat_fallback"
    // for the estimated params (gamma; beta is the swept focal).
    let side = read_sidecar(&out_root.join("store"));
    let resolved = side["provenance"].get("resolved_priors").expect("resolved_priors");
    let arr = resolved.as_array().expect("resolved_priors array");
    assert!(!arr.is_empty(), "resolved_priors must have at least one entry");
    let gamma_entry = arr.iter().find(|e| {
        e.get("param").and_then(|p| p.as_str()) == Some("gamma")
    }).expect("gamma must appear in resolved_priors");
    assert_eq!(
        gamma_entry.get("source").and_then(|s| s.as_str()),
        Some("flat_fallback"),
        "gamma should be flat_fallback when neither model-IR nor --fit \
         declares a prior. Got: {}", gamma_entry);
}

#[test]
fn profile_with_fit_toml_records_fit_toml_priors() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("fit_priors");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml_with_priors(tmp.path(), &ir, &data, "fit");
    let out_root = tmp.path().join("out_fit");

    let output = run_profile(&bin, &out_root, &ir,
        &["--fit", &fit_toml.to_string_lossy()]);
    assert!(output.status.success(),
        "profile run with --fit failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    // Sidecar: gamma resolved from fit_toml (beta is the swept focal).
    let side = read_sidecar(&out_root.join("store"));
    let resolved = side["provenance"].get("resolved_priors").expect("resolved_priors");
    let gamma_entry = resolved.as_array().unwrap().iter().find(|e| {
        e.get("param").and_then(|p| p.as_str()) == Some("gamma")
    }).expect("gamma in resolved_priors");
    assert_eq!(
        gamma_entry.get("source").and_then(|s| s.as_str()),
        Some("fit_toml"),
        "gamma must be sourced from fit_toml. Got: {}", gamma_entry);
    // The prior with its arguments and the bounds the cells search, as a
    // fit's `fit.meta.json` records them.
    assert_eq!(gamma_entry["prior"], serde_json::json!({ "log_normal": { "mu": -1.2, "sigma": 0.5 } }), "{gamma_entry}");
    assert_eq!(gamma_entry["bounds"], serde_json::json!([0.01, 1.0]), "{gamma_entry}");
}

#[test]
fn same_model_different_fit_toml_different_cas_dir() {
    // Hash provenance: two profile runs with the same model + data
    // but different fit tomls (different priors) must produce
    // different CAS dirs.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("hash_two_fits");
    let (ir, data) = write_fixture(tmp.path());
    let fit_a = write_fit_toml_with_priors(tmp.path(), &ir, &data, "fit_a");
    let fit_b = write_fit_toml_with_priors_variant(tmp.path(), &ir, &data);

    let out_a = tmp.path().join("out_a");
    let out_b = tmp.path().join("out_b");

    let a = run_profile(&bin, &out_a, &ir,
        &["--fit", &fit_a.to_string_lossy()]);
    let b = run_profile(&bin, &out_b, &ir,
        &["--fit", &fit_b.to_string_lossy()]);
    assert!(a.status.success(), "run A failed:\n{}",
        String::from_utf8_lossy(&a.stderr));
    assert!(b.status.success(), "run B failed:\n{}",
        String::from_utf8_lossy(&b.stderr));

    let ids_a = point_run_ids(&out_a.join("store"));
    let ids_b = point_run_ids(&out_b.join("store"));
    assert!(!ids_a.is_empty() && !ids_b.is_empty(), "both runs store points");
    assert!(ids_a.iter().all(|id| !ids_b.contains(id)),
        "two distinct fit tomls must produce disjoint run ids. A={:?}, B={:?}",
        ids_a, ids_b);
}

#[test]
fn same_model_flat_fit_vs_prior_fit_different_cas_dir() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("hash_flat_vs_prior");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml_with_priors(tmp.path(), &ir, &data, "fit");

    let flat = write_fit_toml_flat(tmp.path(), &ir, &data);
    let out_no  = tmp.path().join("out_flat");
    let out_yes = tmp.path().join("out_prior");

    let no  = run_profile(&bin, &out_no,  &ir, &["--fit", &flat.to_string_lossy()]);
    let yes = run_profile(&bin, &out_yes, &ir,
        &["--fit", &fit_toml.to_string_lossy()]);
    assert!(no.status.success(),
        "flat-fit run failed:\n{}", String::from_utf8_lossy(&no.stderr));
    assert!(yes.status.success(),
        "prior-fit run failed:\n{}", String::from_utf8_lossy(&yes.stderr));

    let ids_no  = point_run_ids(&out_no.join("store"));
    let ids_yes = point_run_ids(&out_yes.join("store"));
    assert!(!ids_no.is_empty() && !ids_yes.is_empty(), "both runs store points");
    assert!(ids_no.iter().all(|id| !ids_yes.contains(id)),
        "a fit with priors and a fit without must produce disjoint run ids");
}

#[test]
fn focal_param_in_fixed_errors_clearly() {
    // Spec §2 rule: a parameter cannot simultaneously be the sweep
    // axis and in --fixed. The error must name the conflict source.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("focal_conflict");
    let (ir, data) = write_fixture(tmp.path());
    let flat = write_fit_toml_flat(tmp.path(), &ir, &data);
    let out_root = tmp.path().join("out");

    let output = run_profile(&bin, &out_root, &ir,
        // beta is both swept and fixed.
        &["--fit", &flat.to_string_lossy(), "--fixed", "beta=0.3", "--fixed", "gamma=0.1"]);
    assert!(!output.status.success(),
        "swept+fixed conflict must be a hard error");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--sweep") && stderr.contains("--fixed"),
        "error must name both --sweep and --fixed in the conflict \
         message. Got:\n{}", stderr);
}

/// A profile run of a fit that declares no prior (flat-prior fallback) and
/// `--starts from_params` (per-chain init provenance) records both provenance kinds in
/// each point's `provenance.json`, and records its `--label` on the
/// invocation.
#[test]
fn provenance_is_recorded_on_the_point_and_the_invocation() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("prov_roundtrip");
    let (ir, data) = write_fixture(tmp.path());
    let flat = write_fit_toml_flat(tmp.path(), &ir, &data);
    let start = tmp.path().join("start.toml");
    let out_root = tmp.path().join("out");

    let output = run_profile(&bin, &out_root, &ir, &[
        "--fit", &flat.to_string_lossy(),
        "--starts", &format!("from_params={}", start.display()),
        "--label", "round-trip prov",
    ]);
    assert!(output.status.success(),
        "profile run failed: stderr=\n{}", String::from_utf8_lossy(&output.stderr));

    let store = out_root.join("store");

    // Write side: the point's provenance record carries both provenance
    // kinds, and the invocation record carries the label.
        let prov = read_sidecar(&store);
    let prov = prov.get("provenance").expect("provenance.provenance present");
    assert!(prov.get("parameters_provenance").and_then(|p| p.as_object())
        .is_some_and(|o| !o.is_empty()),
        "parameters_provenance (gh#83/85) must be recorded per point");
    assert!(prov.get("init_provenance").map(|v| !v.is_null()).unwrap_or(false),
        "init_provenance must be non-null with --starts from_params. Got: {}", prov);
    let invocations: Vec<serde_json::Value> =
        std::fs::read_dir(store.join(".invocations")).unwrap().flatten()
            .map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).unwrap()).unwrap())
            .collect();
    assert_eq!(invocations.len(), 1, "{invocations:?}");
    assert_eq!(invocations[0]["label"], "round-trip prov",
        "the label is recorded on the invocation");
}
