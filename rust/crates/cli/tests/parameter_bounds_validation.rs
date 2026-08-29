//! End-to-end CLI tests for parameter bounds + finite-value validation
//! (gh#31).
//!
//! Before this fix, `camdl simulate` silently accepted parameter values
//! outside the `[bounds: lo, hi]` declared in the model — only the
//! prior-sampling path bounds-checked. The simulation ran, the output
//! looked plausible, and the user got no signal that they violated the
//! model author's declared validity range. For policy-informing
//! software, that's a silent-wrong-answer class bug.
//!
//! This test file covers the validation matrix from the issue:
//!
//! - `--param NAME=VAL` above hi  → error
//! - `--param NAME=VAL` below lo  → error
//! - `--param NAME=VAL` exactly on either bound → accepted
//! - `--param NAME=NaN`           → error (separate finite-value check)
//! - `--param NAME=Infinity`      → error
//! - Param without declared bounds + any finite value → accepted
//! - `--params my.toml` with an OOB value → error
//! - Scenario `scale = { x = factor }` taking value out of bounds → error

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

/// Write a small SIR-ish model with bounds on `beta` and `gamma`,
/// no bounds on `vacc_eff` (free), and a `scaled_up` scenario that
/// scales beta out of its declared bounds. The death model is
/// nonsense epidemiologically — this fixture exists only to exercise
/// the validator.
fn write_bounded_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta     : rate         in [0.01, 2.0]
  gamma    : rate         in [0.01, 1.0]
  vacc_eff : real
  N0       : count
}

init { S = 990  I = 10  R = 0 }

transitions {
  infect  : S --> I  @ beta * S * I / N0
  recover : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 5 'days }

scenarios {
  scaled_up { scale = { beta = 100.0 } }
}
"#;
    std::fs::write(path, src).unwrap();
}

fn write_baseline_params(path: &Path) {
    std::fs::write(path,
        "beta = 0.3\ngamma = 0.1\nvacc_eff = 0.5\nN0 = 1000\n").unwrap();
}

/// Run `camdl simulate` with the supplied extra args. Returns
/// (status, stderr) so tests can introspect both exit code and the
/// diagnostic message body.
fn run_simulate(
    bin: &Path, model: &Path, params: &Path, extra: &[&str],
) -> (std::process::ExitStatus, String) {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("traj.tsv");
    let mut cmd = Command::new(bin);
    cmd.args([
        "simulate", &model.to_string_lossy(),
        "--params", &params.to_string_lossy(),
        "--backend", "ode",
        "--seed", "1",
        "-o", &out_path.to_string_lossy(),
    ]);
    cmd.args(extra);
    let out = cmd.output().expect("spawn camdl simulate");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status, stderr)
}

// ─── --param OOB: high side ──────────────────────────────────────────────────

#[test]
fn param_above_upper_bound_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "beta=5.0"]);
    assert!(!status.success(), "OOB --param must fail; stderr was:\n{}", stderr);
    assert!(stderr.contains("beta"),
        "error must name the parameter; got:\n{}", stderr);
    assert!(stderr.contains("5"),
        "error must show the supplied value; got:\n{}", stderr);
    assert!(stderr.contains("0.01") && stderr.contains("2"),
        "error must show declared bounds [0.01, 2.0]; got:\n{}", stderr);
    assert!(stderr.contains("outside") || stderr.contains("bounds"),
        "error must indicate bounds violation; got:\n{}", stderr);
}

// ─── --param OOB: low side ───────────────────────────────────────────────────

#[test]
fn param_below_lower_bound_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "beta=0.001"]);
    assert!(!status.success(),
        "value below lower bound must fail; stderr was:\n{}", stderr);
    assert!(stderr.contains("beta") && stderr.contains("0.01"),
        "error must name parameter and show lower bound; got:\n{}", stderr);
}

// ─── --param exactly on bound: accepted ──────────────────────────────────────

#[test]
fn param_on_lower_bound_is_accepted() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "beta=0.01"]);
    assert!(status.success(),
        "beta exactly on the lower bound (0.01) must be accepted \
         (bounds are inclusive). stderr was:\n{}", stderr);
}

#[test]
fn param_on_upper_bound_is_accepted() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "beta=2.0"]);
    assert!(status.success(),
        "beta exactly on the upper bound (2.0) must be accepted \
         (bounds are inclusive). stderr was:\n{}", stderr);
}

// ─── --param NaN ─────────────────────────────────────────────────────────────

#[test]
fn param_nan_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    // Test on `vacc_eff` which has no bounds — to prove the
    // finite-value check fires independently of bounds. Rust's
    // f64::from_str accepts "NaN" (case-insensitive).
    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "vacc_eff=NaN"]);
    assert!(!status.success(),
        "NaN parameter must error; stderr was:\n{}", stderr);
    assert!(stderr.contains("vacc_eff"),
        "error must name the parameter; got:\n{}", stderr);
    assert!(stderr.contains("not finite") || stderr.contains("NaN"),
        "error must indicate non-finite value; got:\n{}", stderr);
    // The bounds-violation hint mentions "bounds" — the finite-value
    // path must not say "outside bounds" for a NaN value, since the
    // problem is finiteness, not range.
    assert!(!stderr.contains("outside declared bounds"),
        "NaN should produce a finite-value error, not a bounds error; got:\n{}",
        stderr);
}

// ─── --param Infinity ────────────────────────────────────────────────────────

#[test]
fn param_positive_infinity_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "vacc_eff=inf"]);
    assert!(!status.success(),
        "+∞ parameter must error; stderr was:\n{}", stderr);
    assert!(stderr.contains("vacc_eff") && stderr.contains("not finite"),
        "error must name parameter and indicate non-finite; got:\n{}", stderr);
}

// ─── Free parameter (no declared bounds): any finite value accepted ─────────

#[test]
fn unbounded_param_with_finite_value_is_accepted() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    // vacc_eff is declared `: real` with no `in [...]` — bounds = None.
    // The validator must not produce a false positive on any finite
    // value, including extreme ones. -1e9 and +1e9 cover both signs.
    for v in &["-1e9", "0", "1e9"] {
        let (status, stderr) = run_simulate(
            &bin, &model, &params, &["--param", &format!("vacc_eff={}", v)]);
        assert!(status.success(),
            "finite vacc_eff={} on an unbounded parameter must be \
             accepted; stderr was:\n{}", v, stderr);
    }
}

// ─── --params TOML file with OOB value ───────────────────────────────────────

#[test]
fn params_file_with_oob_value_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("bad.toml");
    write_bounded_model(&model);
    // gamma's upper bound is 1.0; 5.0 is far outside.
    std::fs::write(&params,
        "beta = 0.3\ngamma = 5.0\nvacc_eff = 0.5\nN0 = 1000\n").unwrap();

    let (status, stderr) = run_simulate(&bin, &model, &params, &[]);
    assert!(!status.success(),
        "params.toml with OOB gamma=5.0 must error; stderr was:\n{}", stderr);
    assert!(stderr.contains("gamma"),
        "error must name the violating parameter; got:\n{}", stderr);
    assert!(stderr.contains("5"),
        "error must show the OOB value; got:\n{}", stderr);
    assert!(stderr.contains("0.01") && stderr.contains("1"),
        "error must show declared bounds [0.01, 1.0]; got:\n{}", stderr);
}

// ─── Scenario `scale = {...}` pushing a parameter out of bounds ──────────────

#[test]
fn scenario_scale_taking_value_out_of_bounds_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    // scenarios.scaled_up multiplies beta by 100 → 0.3 * 100 = 30.0,
    // outside [0.01, 2.0].
    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--scenario", "scaled_up"]);
    assert!(!status.success(),
        "scenario scale taking beta out of [0.01, 2.0] must error; \
         stderr was:\n{}", stderr);
    assert!(stderr.contains("beta"),
        "error must name beta; got:\n{}", stderr);
    assert!(stderr.contains("30") || stderr.contains("outside"),
        "error must show the scaled value or indicate bounds violation; \
         got:\n{}", stderr);
}

// ─── Multiple violations reported together ───────────────────────────────────

#[test]
fn multiple_oob_params_reported_in_one_error() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_bounded_model(&model);
    write_baseline_params(&params);

    // Both beta and gamma OOB. The validator collects all
    // violations and reports them together rather than failing on
    // the first — saves the user a fix-recompile-fix loop.
    let (status, stderr) = run_simulate(
        &bin, &model, &params,
        &["--param", "beta=5.0", "--param", "gamma=10.0"]);
    assert!(!status.success(),
        "multi-violation case must error; stderr was:\n{}", stderr);
    assert!(stderr.contains("beta") && stderr.contains("gamma"),
        "both violating parameters must appear in the error; got:\n{}", stderr);
}

// ─── `probability` is bounded by its type, not only by `in [...]` (gh#763) ───
//
// The value check and the fit's search box must be the SAME interval. While
// they disagreed, a `probability` outside [0, 1] passed validation and was
// then silently clamped by the logit — a plausible-looking wrong answer
// rather than an error. `real` has no such support and must stay unbounded,
// which `finite_value_on_unbounded_param_accepted` above pins on `vacc_eff`.

/// Same shape as `write_bounded_model`, but `report : probability` declares
/// no `in [...]` — its only constraint is the type.
fn write_untyped_probability_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta   : rate         in [0.01, 2.0]
  gamma  : rate         in [0.01, 1.0]
  report : probability
  N0     : count
}

init { S = 990  I = 10  R = 0 }

transitions {
  infect  : S --> I  @ beta * report * S * I / N0
  recover : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 5 'days }
"#;
    std::fs::write(path, src).unwrap();
}

fn write_probability_params(path: &Path) {
    std::fs::write(path,
        "beta = 0.3\ngamma = 0.1\nreport = 0.5\nN0 = 1000\n").unwrap();
}

#[test]
fn probability_above_one_errors_without_a_declared_range() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_untyped_probability_model(&model);
    write_probability_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "report=1.5"]);
    assert!(!status.success(),
        "a `probability` of 1.5 must be rejected on the type alone (gh#763); \
         stderr was:\n{}", stderr);
    assert!(stderr.contains("report"),
        "error must name the parameter; got:\n{}", stderr);
    assert!(stderr.contains("0") && stderr.contains("1"),
        "error must show the [0, 1] support; got:\n{}", stderr);
}

#[test]
fn probability_below_zero_errors_without_a_declared_range() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_untyped_probability_model(&model);
    write_probability_params(&params);

    let (status, stderr) = run_simulate(
        &bin, &model, &params, &["--param", "report=-0.2"]);
    assert!(!status.success(),
        "a negative `probability` must be rejected on the type alone (gh#763); \
         stderr was:\n{}", stderr);
    assert!(stderr.contains("report"),
        "error must name the parameter; got:\n{}", stderr);
}

#[test]
fn probability_inside_the_unit_interval_is_accepted() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    let params = tmp.path().join("p.toml");
    write_untyped_probability_model(&model);
    write_probability_params(&params);

    // Non-vacuity: the new rejection must not swallow legitimate values,
    // including both endpoints.
    for v in ["0", "0.25", "1"] {
        let (status, stderr) = run_simulate(
            &bin, &model, &params, &["--param", &format!("report={}", v)]);
        assert!(status.success(),
            "report={} is inside [0, 1] and must be accepted; stderr:\n{}",
            v, stderr);
    }
}
