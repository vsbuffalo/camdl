//! Integration tests for the `camdl profile` prior-resolution surface
//! (gh#73). Each test drives the release binary end-to-end and asserts
//! observable behaviour: warning text, `run.json` provenance fields,
//! CAS-dir hashes. The unit-level counterpart in
//! `crate::profile_priors::tests` exercises the precedence helper on
//! synthetic `ir::Model` fixtures.
//!
//! Assertions covered:
//!
//!   1. `profile_pmmh_with_neither_warns_and_uses_flat`: model has no
//!      `~` priors, no `--fit`; warning fires on stderr naming every
//!      estimated parameter; `run.json` records every estimated param
//!      as `flat_fallback`.
//!   2. `profile_pmmh_with_fit_toml_silences_flat_warning`: same
//!      model, but a fit toml supplies priors for every estimated
//!      param; no warning; `run.json` records source = `fit_toml` per
//!      param.
//!   3. `run_json_records_resolved_prior_sources`: explicit shape
//!      assertion on the `resolved_priors` array.
//!   4. `same_model_different_fit_toml_different_cas_dir`: hash
//!      provenance — two distinct fit tomls produce two distinct CAS
//!      dirs (different `fit_toml_hash` → different inner_hash).
//!   5. `same_model_no_fit_vs_with_fit_different_cas_dir`: same
//!      provenance with the "no fit" baseline as one of the variants.
//!
//! Skipped when the release binary or the `camdlc` compiler isn't
//! present (mirrors the rest of the integration suite).

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
/// case fires when `--fit` is absent. Matches the `survey_top_k_pmmh`
/// fixture shape so the fit toml schema is well-trodden.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    // Defaults supplied via a `baseline` preset block so the
    // `no --fit / no --params` test path has values to start from
    // (the simulator validates that every parameter has a value).
    // The `--fit toml` path overrides via [estimate].start; the
    // resolver's prior-source assertion is what each test cares
    // about, not the starting value.
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
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
scenarios {
  baseline {
    set = {
      beta  = 0.3
      gamma = 0.1
      N0    = 1000
    }
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

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

/// Write a fit toml with [estimate] containing log_normal priors for
/// every estimated param. The `[stages.dummy]` block satisfies
/// `FitConfigV2::load`'s schema check — profile never *runs* the
/// stages, it only reads `[estimate]` and `[fixed]` for prior /
/// bounds resolution (the v2 schema requires at least one stage to
/// be declared; we treat that as a fixable schema burden rather than
/// an excuse to fork the loader).
fn write_fit_toml_with_priors(dir: &Path, ir: &Path, data: &Path, name: &str) -> PathBuf {
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
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.dummy]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        out  = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join(format!("{}.toml", name));
    std::fs::write(&p, toml).unwrap();
    p
}

/// Variant: distinct prior parameters from the baseline fit toml.
/// Used by the CAS-hash test — same model + data + bounds, different
/// priors must produce a different CAS dir.
fn write_fit_toml_with_priors_variant(dir: &Path, ir: &Path, data: &Path) -> PathBuf {
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
# Same shape, different mu/sigma → different fit_toml_hash.
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -1.0, sigma = 0.3 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.5, sigma = 0.3 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.dummy]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        out  = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join("fit_variant.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Find the profile-base segment under `<out_root>/profiles/`. Each run
/// writes one factored tree `profiles/<base>/<point>/<stage>/<seed>/<start>/`
/// with the provenance sidecar at the base; this returns that base dir.
fn find_profile_base(out_root: &Path) -> PathBuf {
    let profiles = out_root.join("profiles");
    let entries: Vec<_> = std::fs::read_dir(&profiles)
        .unwrap_or_else(|e| panic!("read_dir {}: {}", profiles.display(), e))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    assert_eq!(entries.len(), 1,
        "expected exactly one profile-base dir under {}, found {:?}",
        profiles.display(), entries);
    entries.into_iter().next().unwrap()
}

/// Read the profile-base `fit.meta.json` provenance sidecar — the single
/// authoritative home for the fit-wide attributes (`--label`,
/// `resolved_priors`, `fit_toml_hash`, `estimated`, `data_hashes`).
fn read_sidecar(base: &Path) -> serde_json::Value {
    let body = std::fs::read_to_string(base.join("fit.meta.json"))
        .unwrap_or_else(|e| panic!("read {}/fit.meta.json: {}", base.display(), e));
    serde_json::from_str::<serde_json::Value>(&body).unwrap()
}

fn run_profile(
    bin: &Path,
    out_root: &Path,
    ir: &Path,
    data: &Path,
    extra_args: &[&str],
) -> std::process::Output {
    let out_tsv = out_root.join("profile.tsv");
    let mut args: Vec<String> = vec![
        "profile".into(), ir.to_string_lossy().into_owned(),
        // Pulls defaults for beta/gamma/N0 from the baseline preset
        // so the validate_parameter_values step doesn't reject the
        // run with "no value for 'beta'". This is the same way
        // `camdl profile` is invoked everywhere else in the test
        // suite (see profile_pmmh.rs).
        "--scenario".into(), "baseline".into(),
        "--data".into(), data.to_string_lossy().into_owned(),
        "--obs".into(), "cases".into(),
        "--sweep".into(), "beta=lin(0.2,0.4,2)".into(),
        "--algorithm".into(), "pmmh".into(),
        // Must exceed the fixed per-cell burn-in (100); 120 leaves 20
        // post-burn-in samples — still a fast smoke run (gh#102).
        "--pmmh-steps".into(), "120".into(),
        "--pmmh-particles".into(), "30".into(),
        "--pmmh-rho".into(), "0.99".into(),
        "--particles".into(), "30".into(),
        "--iterations".into(), "5".into(),
        "--starts".into(), "1".into(),
        "--rw-sd".into(), "auto".into(),
        "--output".into(), out_tsv.to_string_lossy().into_owned(),
        "--seed".into(), "1".into(),
    ];
    for a in extra_args { args.push((*a).into()); }
    Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args.iter().map(|s| s.as_str()))
        .output()
        .expect("spawn camdl profile")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn profile_pmmh_with_neither_warns_and_uses_flat() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("flat_warning");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out_flat");

    let output = run_profile(&bin, &out_root, &ir, &data, &["--fixed", "N0=1000"]);
    assert!(output.status.success(),
        "profile run failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Warning must fire.
    assert!(stderr.contains("flat priors"),
        "expected 'flat priors' wording in stderr, got:\n{}", stderr);
    // Warning must name every estimated parameter (gamma in this
    // fixture; beta is the focal sweep param so it's excluded).
    assert!(stderr.contains("gamma"),
        "expected 'gamma' named in warning, got:\n{}", stderr);
    // Remediation lines must surface --fit + model-file paths.
    assert!(stderr.contains("--fit"),
        "expected --fit remedy in warning, got:\n{}", stderr);
    assert!(stderr.contains("model file"),
        "expected 'model file' remedy in warning, got:\n{}", stderr);

    // The profile-base sidecar must record sources = "flat_fallback"
    // for the estimated params.
    let side = read_sidecar(&find_profile_base(&out_root));
    let resolved = side.get("resolved_priors").expect("resolved_priors");
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
    // No --fit → the sidecar's fit_toml_hash is empty (defaults to "",
    // never a populated 64-hex digest).
    let fth = side.get("fit_toml_hash").and_then(|h| h.as_str()).unwrap_or("");
    assert!(fth.is_empty(),
        "fit_toml_hash must be empty without --fit. Got: {:?}", fth);
}

#[test]
fn profile_pmmh_with_fit_toml_silences_flat_warning() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("fit_priors");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml_with_priors(tmp.path(), &ir, &data, "fit");
    let out_root = tmp.path().join("out_fit");

    let output = run_profile(&bin, &out_root, &ir, &data,
        &["--fit", &fit_toml.to_string_lossy()]);
    assert!(output.status.success(),
        "profile run with --fit failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);

    // No flat-priors warning when every param has a prior.
    assert!(!stderr.contains("flat priors"),
        "warning must NOT fire when fit toml supplies priors. \
         Got stderr:\n{}", stderr);

    // Sidecar: gamma resolved from fit_toml (beta is the swept focal).
    let side = read_sidecar(&find_profile_base(&out_root));
    let resolved = side.get("resolved_priors").expect("resolved_priors");
    let gamma_entry = resolved.as_array().unwrap().iter().find(|e| {
        e.get("param").and_then(|p| p.as_str()) == Some("gamma")
    }).expect("gamma in resolved_priors");
    assert_eq!(
        gamma_entry.get("source").and_then(|s| s.as_str()),
        Some("fit_toml"),
        "gamma must be sourced from fit_toml. Got: {}", gamma_entry);

    // fit_toml_hash must be present and a 64-char hex string.
    let hash = side.get("fit_toml_hash").and_then(|h| h.as_str())
        .filter(|h| !h.is_empty())
        .expect("fit_toml_hash must be present when --fit is supplied");
    assert_eq!(hash.len(), 64, "fit_toml_hash must be SHA-256 hex (64 chars), got: {}", hash);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()),
        "fit_toml_hash must be hex, got: {}", hash);
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

    let a = run_profile(&bin, &out_a, &ir, &data,
        &["--fit", &fit_a.to_string_lossy()]);
    let b = run_profile(&bin, &out_b, &ir, &data,
        &["--fit", &fit_b.to_string_lossy()]);
    assert!(a.status.success(), "run A failed:\n{}",
        String::from_utf8_lossy(&a.stderr));
    assert!(b.status.success(), "run B failed:\n{}",
        String::from_utf8_lossy(&b.stderr));

    let base_a = find_profile_base(&out_a);
    let base_b = find_profile_base(&out_b);
    assert_ne!(base_a.file_name().unwrap(),
               base_b.file_name().unwrap(),
        "two distinct fit tomls must produce two distinct CAS dirs. \
         A={}, B={}", base_a.display(), base_b.display());
}

#[test]
fn same_model_no_fit_vs_with_fit_different_cas_dir() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("hash_fit_vs_none");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml_with_priors(tmp.path(), &ir, &data, "fit");

    let out_no  = tmp.path().join("out_nofit");
    let out_yes = tmp.path().join("out_yesfit");

    let no  = run_profile(&bin, &out_no,  &ir, &data, &["--fixed", "N0=1000"]);
    let yes = run_profile(&bin, &out_yes, &ir, &data,
        &["--fit", &fit_toml.to_string_lossy()]);
    assert!(no.status.success(),
        "no-fit run failed:\n{}", String::from_utf8_lossy(&no.stderr));
    assert!(yes.status.success(),
        "with-fit run failed:\n{}", String::from_utf8_lossy(&yes.stderr));

    let base_no  = find_profile_base(&out_no);
    let base_yes = find_profile_base(&out_yes);
    assert_ne!(base_no.file_name().unwrap(),
               base_yes.file_name().unwrap(),
        "no-fit and with-fit runs must produce distinct CAS dirs");
}

#[test]
fn focal_param_in_fixed_errors_clearly() {
    // Spec §2 rule: a parameter cannot simultaneously be the sweep
    // axis and in --fixed. The error must name the conflict source.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("focal_conflict");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile(&bin, &out_root, &ir, &data,
        // beta is both swept and fixed.
        &["--fixed", "beta=0.3", "--fixed", "gamma=0.1", "--fixed", "N0=1000"]);
    assert!(!output.status.success(),
        "swept+fixed conflict must be a hard error");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--sweep") && stderr.contains("--fixed"),
        "error must name both --sweep and --fixed in the conflict \
         message. Got:\n{}", stderr);
}

/// Run a `camdl <subcmd>` reader command against the CAS root, returning its
/// captured stdout (the reader display) as a String.
fn camdl_read(bin: &Path, out_root: &Path, args: &[&str]) -> String {
    let out = Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .output()
        .expect("spawn camdl reader");
    // `list` prints its table on stdout and section headers on stderr; show/cat
    // print on stdout. Concatenate so a single helper covers all three.
    format!("{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr))
}

/// The `run_id` (hex) of the first `ProfilePoint` leaf under a profile base.
fn first_leaf_run_id(base: &Path) -> String {
    fn walk(dir: &Path, out: &mut Option<String>) {
        if out.is_some() { return; }
        let rj = dir.join("run.json");
        if rj.is_file() {
            if let Ok(body) = std::fs::read_to_string(&rj) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                    if v.get("kind").and_then(|k| k.as_str()) == Some("profile_point") {
                        if let Some(rid) = v.get("run_id").and_then(|r| r.as_str()) {
                            *out = Some(rid.to_string());
                            return;
                        }
                    }
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { walk(&p, out); }
            }
        }
    }
    let mut found = None;
    walk(base, &mut found);
    found.unwrap_or_else(|| panic!("no ProfilePoint leaf under {}", base.display()))
}

/// P1 provenance discipline (write → read → visible): the audit data the old
/// per-run `ProfileMeta` carried must be *surfaced by the new reader*, not just
/// present on disk. A profile run with no `--fit` (flat-prior fallback),
/// `--suppress-warnings` (waiver trail), and `--init from_prior` (per-chain
/// init provenance) populates all three provenance kinds; this asserts each is
/// visible through `camdl show`/`list`/`cat`. The fit migration once dropped
/// provenance silently — the round-trip is what catches that.
#[test]
fn provenance_round_trips_through_reader() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("prov_roundtrip");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile(&bin, &out_root, &ir, &data, &[
        "--fixed", "N0=1000",
        "--suppress-warnings",
        "--init", "from_prior",
        "--label", "round-trip prov",
    ]);
    assert!(output.status.success(),
        "profile run failed: stderr=\n{}", String::from_utf8_lossy(&output.stderr));

    let base = find_profile_base(&out_root);

    // Write side: the leaf record carries all three provenance kinds, and the
    // base sidecar carries the label (its single authoritative home).
    let rid = first_leaf_run_id(&base);
    let leaf_dir = {
        // Re-find the leaf dir holding `rid` for a disk-side sanity check.
        fn walk(dir: &Path, rid: &str, out: &mut Option<PathBuf>) {
            if out.is_some() { return; }
            if let Ok(body) = std::fs::read_to_string(dir.join("run.json")) {
                if body.contains(rid) { *out = Some(dir.to_path_buf()); return; }
            }
            if let Ok(es) = std::fs::read_dir(dir) {
                for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), rid, out); } }
            }
        }
        let mut f = None; walk(&base, &rid, &mut f);
        f.expect("leaf dir for run_id")
    };
    let leaf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(leaf_dir.join("run.json")).unwrap()).unwrap();
    let prov = leaf.get("inputs").and_then(|i| i.get("provenance"))
        .expect("leaf inputs.provenance present");
    assert!(prov.get("parameters_provenance").and_then(|p| p.as_object())
        .is_some_and(|o| !o.is_empty()),
        "parameters_provenance (gh#83/85) must be recorded per leaf");
    assert!(prov.get("init_provenance").map(|v| !v.is_null()).unwrap_or(false),
        "init_provenance must be non-null with --init from_prior. Got: {}", prov);
    assert!(prov.get("suppressed_warnings").and_then(|s| s.as_array())
        .is_some_and(|a| a.iter().any(|w| w.as_str() == Some("profile_flat_prior_fallback"))),
        "suppressed-warnings waiver must be recorded per leaf");
    assert_eq!(read_sidecar(&base).get("label").and_then(|l| l.as_str()),
        Some("round-trip prov"),
        "label must live on the base sidecar (its single authoritative home)");

    // Read side: `camdl show <leaf>` must SURFACE each provenance kind.
    let shown = camdl_read(&bin, &out_root, &["show", &rid[..12]]);
    assert!(shown.contains("round-trip prov"),
        "show must surface the --label (from sidecar). Got:\n{}", shown);
    assert!(shown.contains("parameter provenance"),
        "show must surface the gh#83/85 parameter-resolution provenance. Got:\n{}", shown);
    assert!(shown.contains("init provenance"),
        "show must surface the per-chain init provenance. Got:\n{}", shown);
    assert!(shown.contains("profile_flat_prior_fallback"),
        "show must surface the suppressed-warnings waiver. Got:\n{}", shown);

    // `camdl list` surfaces the profile with its label.
    let listed = camdl_read(&bin, &out_root, &["list"]);
    assert!(listed.contains("round-trip prov"),
        "list must surface the profile label. Got:\n{}", listed);

    // `camdl cat <leaf>` returns the per-cell mle.toml.
    let catted = camdl_read(&bin, &out_root, &["cat", &rid[..12]]);
    assert!(catted.contains("final_loglik"),
        "cat must return the leaf's mle.toml. Got:\n{}", catted);
}

/// `camdl label` on a profile must rewrite the profile-base sidecar (the
/// label's single authoritative home, guardrail 5) — NOT a per-leaf copy.
/// Relabeling touches one file regardless of how many cells the profile has.
#[test]
fn label_command_relabels_profile_sidecar() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("label_profile");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    // Profile with no `--label`.
    let output = run_profile(&bin, &out_root, &ir, &data, &["--fixed", "N0=1000"]);
    assert!(output.status.success(), "profile run failed:\n{}",
        String::from_utf8_lossy(&output.stderr));

    let base = find_profile_base(&out_root);
    assert!(read_sidecar(&base).get("label").and_then(|l| l.as_str()).is_none(),
        "a fresh profile (no --label) must have no sidecar label");

    // The profile-base hash prefix is the `hash8` suffix of the base dir name.
    let dir_name = base.file_name().unwrap().to_string_lossy().into_owned();
    let hash8 = dir_name.rsplit('-').next().unwrap().to_string();

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["label", &hash8, "relabelled profile",
               "--root", &out_root.to_string_lossy()])
        .output().expect("spawn camdl label");
    assert!(out.status.success(),
        "label on profile must succeed: stderr=\n{}",
        String::from_utf8_lossy(&out.stderr));

    // Read back from the sidecar — the one place the label lives.
    assert_eq!(read_sidecar(&base).get("label").and_then(|l| l.as_str()),
        Some("relabelled profile"),
        "camdl label must write the profile-base sidecar (the label's home)");

    // And `camdl show <leaf>` surfaces the relabelled value.
    let rid = first_leaf_run_id(&base);
    let shown = camdl_read(&bin, &out_root, &["show", &rid[..12]]);
    assert!(shown.contains("relabelled profile"),
        "show must surface the relabelled profile label. Got:\n{}", shown);
}
