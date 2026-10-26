//! A parameter is either estimated or given a value, never both.
//!
//! Two contradictions in a fit config are refused:
//!
//! - **a scenario that sets or scales a parameter `[estimate]` lists** — the
//!   fit config's `scenario =`, or `--scenario` on `survey --fit` — on `fit run`
//!   and `survey --fit`, composed presets included. Before this, the estimate
//!   silently overwrote the scenario's value.
//! - **a name in both `[fixed]` and `[estimate]`** of one fit config. `fit run`
//!   refused it already; `profile --fit` and `survey --fit` used to warn and let
//!   `[fixed]` win.
//!
//! `[fixed] from_scenario` is not the fit's scenario: it imports a scenario's
//! values minus the estimated ones (gh#37), and stays accepted.
//!
//! `--fixed` over a name in `[estimate]` is a layered override: not refused,
//! reported as an `info:` line.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test"))
}

fn camdl_bin() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 1.0]
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
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
scenarios {
  hi    { set = { beta = 0.8 } }
  half  { scale = { beta = 0.5 } }
  comp  { compose = [hi] }
  fastg { set = { gamma = 0.9 } }
  bigN  { set = { N0 = 5000 } }
  base  {
    set = {
      beta  = 0.3
      gamma = 0.2
      N0    = 1000
    }
  }
}
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

const METHOD: &str = r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 3
cooling = 0.7
"#;

/// The compiled model and data in `dir`; returns the IR path.
fn setup(dir: &Path) -> PathBuf {
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), CASES_TSV).unwrap();
    ir
}

/// Write `<name>.toml` in `dir`: `top` above `[model]`, then `[estimate]`
/// and `[fixed]` bodies verbatim, then the method.
fn fit_toml(dir: &Path, name: &str, top: &str, estimate: &str, fixed: &str) -> PathBuf {
    let p = dir.join(format!("{name}.toml"));
    std::fs::write(&p, format!(
        "output_dir = \"res_{name}\"\n{top}\n[model]\ncamdl = \"m.ir.json\"\n\
         [data.observations]\ncases = \"cases.tsv\"\n[estimate]\n{estimate}\n[fixed]\n{fixed}\n{METHOD}"
    )).unwrap();
    p
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Non-zero exit, every `line` in the diagnostic, and nothing under `written`.
fn assert_refused(what: &str, out: &Output, lines: &[&str], written: &Path) {
    let err = stderr(out);
    assert!(!out.status.success(), "{what}: must be refused, but exited 0; stderr:\n{err}");
    for line in lines {
        assert!(err.contains(line), "{what}: diagnostic lacks {line:?}; stderr:\n{err}");
    }
    assert!(!written.exists(), "{what}: a refused run must write nothing ({}); stderr:\n{err}",
        written.display());
}

const BETA: &str = "beta = { bounds = [0.01, 1.0], start = 0.3 }";
const BETA_GAMMA: &str =
    "beta = { bounds = [0.01, 1.0], start = 0.3 }\ngamma = { bounds = [0.01, 1.0], start = 0.1 }";

// ─── fit run ──────────────────────────────────────────────────────────────────

#[test]
fn fit_run_refuses_a_scenario_that_sets_an_estimated_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "hi", "scenario = \"hi\"", BETA, "gamma = 0.15\nN0 = 1000");
    let out = camdl(dir, &["fit", "run", "hi.toml", "--seed", "1"]);
    assert_refused("fit run hi", &out, &[
        "error: parameter `beta` is both estimated and set by scenario `hi`",
        "  [estimate]:  beta    (in `hi.toml`)",
        "  scenario:    beta = 0.8    (set beta = 0.8)",
        "Fix: drop `beta` from [estimate] in `hi.toml` to fit under the scenario's value \
         (give [fixed] the scenario's value: `beta = 0.8`), or name a scenario that does not \
         touch it.",
    ], &dir.join("res_hi"));
}

#[test]
fn fit_run_refuses_a_composed_scenario_and_a_scaling_scenario() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "comp", "scenario = \"comp\"", BETA, "gamma = 0.15\nN0 = 1000");
    let out = camdl(dir, &["fit", "run", "comp.toml", "--seed", "1"]);
    assert_refused("fit run comp", &out, &[
        "error: parameter `beta` is both estimated and set by scenario `comp`",
        "  scenario:    beta = 0.8    (set beta = 0.8 from composed preset `hi`)",
    ], &dir.join("res_comp"));

    fit_toml(dir, "half", "scenario = \"half\"", BETA, "gamma = 0.15\nN0 = 1000");
    let out = camdl(dir, &["fit", "run", "half.toml", "--seed", "1"]);
    assert_refused("fit run half", &out, &[
        "error: parameter `beta` is both estimated and scaled by scenario `half`",
        "  scenario:    scale beta \u{d7} 0.5",
        "Fix: name a scenario that does not touch it.",
    ], &dir.join("res_half"));
}

/// A scenario that touches only parameters the fit does not estimate still
/// runs: the refusal is about the estimated set, not any scenario.
#[test]
fn fit_run_accepts_a_scenario_disjoint_from_the_estimate() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "fastg", "scenario = \"fastg\"", BETA, "gamma = 0.9\nN0 = 1000");
    let out = camdl(dir, &["fit", "run", "fastg.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run fastg: {}", stderr(&out));
}

/// `[fixed] from_scenario` imports `base`'s values minus the estimated `beta`
/// (gh#37). That import is the `[fixed]` source, not the fit's scenario, so it
/// is not refused. Pinned as-is while the carve-out awaits its own decision.
#[test]
fn fit_run_from_scenario_carve_out_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "carve", "", BETA, "from_scenario = \"base\"");
    let out = camdl(dir, &["fit", "run", "carve.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run carve: {}", stderr(&out));
    assert!(!stderr(&out).contains("both estimated and set by scenario"), "{}", stderr(&out));
}

#[test]
fn fit_run_refuses_a_name_in_both_fixed_and_estimate() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "both", "", BETA, "beta = 0.4\ngamma = 0.15\nN0 = 1000");
    let out = camdl(dir, &["fit", "run", "both.toml", "--seed", "1"]);
    assert_refused("fit run both", &out, &[
        "error: parameter `beta` is in both [estimate] and [fixed]",
        "Fix: remove `beta` from [fixed] to estimate it, or from [estimate] to hold it at its \
         [fixed] value.",
    ], &dir.join("res_both"));
}

// ─── profile ──────────────────────────────────────────────────────────────────

fn profile(dir: &Path, cwd: &str, fit: &str, extra: &[&str]) -> Output {
    let cwd = dir.join(cwd);
    std::fs::create_dir_all(&cwd).unwrap();
    let ir = dir.join("m.ir.json");
    let fit = dir.join(fit);
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(&cwd)
        .args(["profile", &ir.to_string_lossy(), "--fit", &fit.to_string_lossy()])
        .args(["--sweep", "gamma=0.1,0.5", "--particles", "50", "--iterations", "3",
               "--n-starts", "1", "--seed", "1"])
        .args(extra)
        .output()
        .expect("spawn camdl profile")
}

#[test]
fn profile_refuses_a_name_in_both_fixed_and_estimate() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "pf_both", "", BETA_GAMMA, "beta = 0.4\nN0 = 1000");
    let out = profile(dir, "both", "pf_both.toml", &["--rw-sd", "beta=0.02"]);
    assert_refused("profile both", &out, &[
        "error: parameter `beta` is in both [estimate] and [fixed]",
    ], &dir.join("both").join("results"));
}

// ─── survey ───────────────────────────────────────────────────────────────────

fn survey(dir: &Path, root: &str, args: &[&str]) -> Output {
    let ir = dir.join("m.ir.json");
    Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(["survey", &ir.to_string_lossy()])
        .args(args)
        .args(["--eval", "simulate", "--n-points", "10", "--seed", "1", "--output", root])
        .output()
        .expect("spawn camdl survey")
}

#[test]
fn survey_refuses_a_scenario_that_sets_an_estimated_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "sv", "", BETA, "gamma = 0.15\nN0 = 1000");
    fit_toml(dir, "sv_hi", "scenario = \"hi\"", BETA, "gamma = 0.15\nN0 = 1000");
    for (what, args) in [("config scenario", &["--fit", "sv_hi.toml"][..]),
                         ("--scenario", &["--fit", "sv.toml", "--scenario", "hi"][..])] {
        let root = what.trim_start_matches('-').replace(' ', "_");
        let out = survey(dir, &root, args);
        assert_refused(what, &out, &[
            "error: parameter `beta` is both estimated and set by scenario `hi`",
            "  [estimate]:  beta    (in `",
        ], &dir.join(&root));
    }
}

#[test]
fn survey_refuses_a_name_in_both_fixed_and_estimate() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "sv_both", "", BETA, "beta = 0.4\ngamma = 0.15\nN0 = 1000");
    let out = survey(dir, "both", &["--fit", "sv_both.toml"]);
    assert_refused("survey both", &out, &[
        "error: parameter `beta` is in both [estimate] and [fixed]",
    ], &dir.join("both"));
}

/// The carve-out on the `survey --fit` path, as on `fit run`.
#[test]
fn survey_from_scenario_carve_out_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "sv_carve", "", BETA, "from_scenario = \"base\"");
    let out = survey(dir, "carve", &["--fit", "sv_carve.toml"]);
    assert!(out.status.success(), "survey carve: {}", stderr(&out));
}

// ─── layered overrides reported at info ───────────────────────────────────────

/// `--fixed N0=1000` over a fit config listing N0 in `[estimate]` pins it:
/// one `info:` line, not a warning, and the profile runs.
#[test]
fn profile_fixed_over_estimate_is_an_info_notice() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let est = format!("{BETA_GAMMA}\nN0 = {{ bounds = [100, 10000], start = 1000 }}");
    fit_toml(dir, "pf_kick", "", &est, "");
    let out = profile(dir, "kick", "pf_kick.toml", &["--fixed", "N0=1000", "--rw-sd", "beta=0.02"]);
    let err = stderr(&out);
    assert!(out.status.success(), "profile --fixed: {err}");
    let line = "info: N0: `--fixed N0=1000` pins it, so it is not estimated (the fit \
                config lists it in [estimate])";
    assert_eq!(err.matches(line).count(), 1, "want {line:?} once; stderr:\n{err}");
    assert!(!err.contains("warning: --fixed"), "no warning for a layered --fixed: {err}");
}
