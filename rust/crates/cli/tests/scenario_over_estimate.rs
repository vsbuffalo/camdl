//! A parameter is either estimated or given a value, never both.
//!
//! Two contradictions in a fit config are refused:
//!
//! - **a scenario that sets or scales a parameter `[estimate]` lists** — the
//!   fit config's `scenario =`, or `--scenario` on `profile --fit` — on `fit run`
//!   and `profile --fit`, composed presets included. Before this, the estimate
//!   silently overwrote the scenario's value.
//! - **a name in both `[fixed]` and `[estimate]`** of one fit config. `fit run`
//!   refused it already; `profile --fit` used to warn and let
//!   `[fixed]` win.
//!
//! - **`[fixed] from_scenario` naming a scenario that sets or scales a
//!   parameter `[estimate]` lists** — on every command that reads the fit
//!   config. The import used to skip the estimated parameters (gh#37).
//!
//! Two layered overrides are not refused: `--fixed` over a name in
//! `[estimate]`, which holds that parameter in the stored problem, and a
//! profile's scenario setting the focal parameter the profile steps across its
//! grid, which is reported as an `info:` line.

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
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
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
        "store = \"res_{name}\"\n{top}\n[model]\ncamdl = \"m.ir.json\"\n\
         [data.observations]\ncases = \"cases.tsv\"\n[estimate]\n{estimate}\n[fixed]\n{fixed}\n{METHOD}"
    )).unwrap();
    p
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
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

/// `[fixed] from_scenario = "base"` with `beta` in `[estimate]`: `base` sets
/// beta = 0.3, so beta would be both fixed (imported) and estimated.
#[test]
fn fit_run_refuses_from_scenario_that_sets_an_estimated_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "carve", "", BETA, "from_scenario = \"base\"");
    let out = camdl(dir, &["fit", "run", "carve.toml", "--seed", "1"]);
    assert_refused("fit run from_scenario base", &out, &[
        "error: parameter `beta` is both estimated and set by scenario `base`, which \
         [fixed] from_scenario imports",
        "  [estimate]:  beta",
        "  scenario:    beta = 0.3    (set beta = 0.3)",
        "A parameter cannot be both fixed (imported from the scenario) and estimated.",
        "Fix: define a scenario that does not set `beta` and name it in from_scenario, or \
         drop `beta` from [estimate] to hold it at the scenario's value.",
    ], &dir.join("res_carve"));
}

/// Composed presets and `scale` are caught as on the fit's own scenario. A
/// scale-only collision has no "drop it from [estimate]" fix: the import
/// would then have no value for the scale to multiply.
#[test]
fn fit_run_refuses_from_scenario_composed_and_scaling() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "fs_comp", "", BETA, "from_scenario = \"comp\"");
    let out = camdl(dir, &["fit", "run", "fs_comp.toml", "--seed", "1"]);
    assert_refused("fit run from_scenario comp", &out, &[
        "error: parameter `beta` is both estimated and set by scenario `comp`, which \
         [fixed] from_scenario imports",
        "  scenario:    beta = 0.8    (set beta = 0.8 from composed preset `hi`)",
    ], &dir.join("res_fs_comp"));

    fit_toml(dir, "fs_half", "", BETA, "from_scenario = \"half\"");
    let out = camdl(dir, &["fit", "run", "fs_half.toml", "--seed", "1"]);
    assert_refused("fit run from_scenario half", &out, &[
        "error: parameter `beta` is both estimated and scaled by scenario `half`, which \
         [fixed] from_scenario imports",
        "  scenario:    scale beta \u{d7} 0.5",
        "Fix: define a scenario that does not scale `beta` and name it in from_scenario.",
    ], &dir.join("res_fs_half"));
}

/// A `from_scenario` disjoint from `[estimate]` still imports and runs.
#[test]
fn fit_run_accepts_from_scenario_disjoint_from_the_estimate() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "fs_ok", "", BETA_GAMMA, "from_scenario = \"bigN\"");
    let out = camdl(dir, &["fit", "run", "fs_ok.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run fs_ok: {}", stderr(&out));
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
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(&cwd)
        .args(["profile", &ir.to_string_lossy(), "--fit", &fit.to_string_lossy()])
        .args(["--sweep", "gamma=0.1,0.5", "--particles", "50", "--iterations", "3",
               "--n-starts", "1", "--seed", "1"])
        // The store in the working directory, over the fit file's `store`.
        .args(["--store", "camdl-store"])
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

/// `profile --fit` refuses a scenario (`--scenario` or the fit config's own)
/// that sets or scales a nuisance parameter `[estimate]` lists, as `fit run`
/// does. Before, the scenario only moved the estimate's starting value and
/// run.json recorded the parameter as estimated with `source: scenario`.
#[test]
fn profile_refuses_a_scenario_that_sets_an_estimated_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "pf_sv", "", BETA, "N0 = 1000");
    fit_toml(dir, "pf_sv_hi", "scenario = \"hi\"", BETA, "N0 = 1000");
    for (what, cwd, fit, extra) in [
        ("profile config scenario", "pf_cfg", "pf_sv_hi.toml", &["--rw-sd", "beta=0.02"][..]),
        ("profile --scenario", "pf_cli", "pf_sv.toml",
         &["--rw-sd", "beta=0.02", "--scenario", "hi"][..]),
    ] {
        let out = profile(dir, cwd, fit, extra);
        assert_refused(what, &out, &[
            "error: parameter `beta` is both estimated and set by scenario `hi`",
            "  [estimate]:  beta    (in `",
        ], &dir.join(cwd).join("results"));
    }
    fit_toml(dir, "pf_sv_half", "scenario = \"half\"", BETA, "N0 = 1000");
    let out = profile(dir, "pf_half", "pf_sv_half.toml", &["--rw-sd", "beta=0.02"]);
    assert_refused("profile scaling scenario", &out, &[
        "error: parameter `beta` is both estimated and scaled by scenario `half`",
    ], &dir.join("pf_half").join("results"));
}

/// `camdl profile` of the fit `fit` (a file in `dir`) with the given
/// arguments, in `dir/cwd`.
fn profile_of(dir: &Path, cwd: &str, fit: &str, args: &[&str]) -> Output {
    let cwd = dir.join(cwd);
    std::fs::create_dir_all(&cwd).unwrap();
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(&cwd)
        .args(["profile", &dir.join("m.ir.json").to_string_lossy(),
               "--fit", &dir.join(fit).to_string_lossy()])
        .args(["--particles", "30", "--iterations", "2", "--n-starts", "1", "--seed", "1"])
        .args(["--store", "camdl-store"])
        .args(args)
        .output()
        .expect("spawn camdl profile")
}

/// The parameters a profile estimates are the ones the fit's `[estimate]`
/// lists, less the swept one; `--rw-sd auto` and `--rw-sd gamma=...` both take
/// them from it. A scenario that sets one of them is refused, and a scenario
/// that sets a parameter the fit holds in `[fixed]` is refused as two values
/// for one parameter.
#[test]
fn profile_refuses_a_scenario_that_sets_a_parameter_rw_sd_estimates() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "pf_est", "", BETA_GAMMA, "N0 = 1000");
    fit_toml(dir, "pf_held", "", BETA, "N0 = 1000\ngamma = 0.1");
    let sweep = ["--sweep", "beta=0.2,0.4", "--scenario", "fastg"];
    for (what, cwd, rw_sd) in [("auto", "pn_auto", "auto"), ("explicit", "pn_map", "gamma=0.02")] {
        let out = profile_of(dir, cwd, "pf_est.toml", &[&sweep[..], &["--rw-sd", rw_sd]].concat());
        assert_refused(what, &out, &[
            "error: parameter `gamma` is both estimated and set by scenario `fastg`",
            "  [estimate]:  gamma",
            "  scenario:    gamma = 0.9    (set gamma = 0.9)",
        ], &dir.join(cwd).join("results"));
    }
    let out = profile_of(dir, "pn_held", "pf_held.toml", &sweep);
    assert_refused("held in [fixed]", &out, &[
        "error: parameter `gamma` is given two values: by scenario `fastg` and by [fixed] in `",
    ], &dir.join("pn_held").join("results"));
}

/// The refusal is about the estimated set: pinning the scenario's parameter
/// out of it (`--fixed-file` holds `gamma`; the scenario's value is the one
/// used) lets the profile run.
#[test]
fn profile_accepts_a_scenario_whose_parameters_are_pinned() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "pf_nuis", "", BETA, "N0 = 1000");
    let sweep = ["--sweep", "beta=0.2,0.4", "--scenario", "fastg", "--rw-sd", "auto"];
    std::fs::write(dir.join("g.toml"), "gamma = 0.5\n").unwrap();
    let g = dir.join("g.toml").to_string_lossy().into_owned();
    let out = profile_of(dir, "pn_pinned", "pf_nuis.toml", &[&sweep[..], &["--fixed-file", g.as_str()]].concat());
    assert!(out.status.success(), "pinned by --fixed-file: {}", stderr(&out));
    // A file value equal to the scenario's leaves no replacement on record,
    // and still pins: the name is read from the file.
    std::fs::write(dir.join("g_eq.toml"), "gamma = 0.9\n").unwrap();
    let g_eq = dir.join("g_eq.toml").to_string_lossy().into_owned();
    let out = profile_of(dir, "pn_pinned_eq", "pf_nuis.toml", &[&sweep[..], &["--fixed-file", g_eq.as_str()]].concat());
    assert!(out.status.success(), "pinned at the scenario's own value: {}", stderr(&out));
}

/// The `from_scenario` refusal on the `profile --fit` path.
#[test]
fn profile_refuses_from_scenario_that_sets_an_estimated_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "sv_carve", "", BETA, "from_scenario = \"base\"");
    let line = "parameter `beta` is both estimated and set by scenario `base`, which \
                [fixed] from_scenario imports";
    let out = profile(dir, "pf_carve", "sv_carve.toml", &["--rw-sd", "beta=0.02"]);
    assert_refused("profile from_scenario", &out, &[line],
        &dir.join("pf_carve").join("results"));
}

// ─── layered overrides reported at info ───────────────────────────────────────

/// The `parameters_provenance` and the stored problem's run id of the first
/// profile point under `store`.
fn first_point(store: &Path) -> (serde_json::Value, String) {
    let mut stack = vec![store.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(body) = std::fs::read_to_string(d.join("identity.json")) {
            let id: serde_json::Value = serde_json::from_str(&body).unwrap();
            if id["kind"] == "profile_point" {
                let prov: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(d.join("provenance.json")).unwrap()).unwrap();
                return (prov["provenance"]["parameters_provenance"].clone(),
                    id["identity"]["problem"]["run_id"].as_str().expect("problem run_id").to_string());
            }
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let hidden = e.file_name().to_string_lossy().starts_with('.');
            if e.path().is_dir() && !hidden { stack.push(e.path()); }
        }
    }
    panic!("no profile point under {}", store.display());
}

/// `--fixed N0=1000` over a fit config listing N0 in `[estimate]` holds N0 in
/// the stored problem: the profile runs, every point records N0 as fixed at
/// 1000, and the problem it profiles differs from the one profiled without
/// `--fixed`.
#[test]
fn profile_fixed_over_estimate_holds_the_parameter() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let est = format!("{BETA_GAMMA}\nN0 = {{ bounds = [100, 10000], start = 1000 }}");
    fit_toml(dir, "pf_kick", "", &est, "");
    let held = profile(dir, "kick", "pf_kick.toml", &["--fixed", "N0=1000", "--rw-sd", "beta=0.02"]);
    assert!(held.status.success(), "profile --fixed over [estimate]: {}", stderr(&held));
    let free = profile(dir, "free", "pf_kick.toml", &["--rw-sd", "auto"]);
    assert!(free.status.success(), "profile without --fixed: {}", stderr(&free));

    let (prov, held_problem) = first_point(&dir.join("kick").join("camdl-store"));
    assert_eq!(prov["N0"]["role"], "fixed", "{prov}");
    assert_eq!(prov["N0"]["value"], 1000.0, "{prov}");
    let (free_prov, free_problem) = first_point(&dir.join("free").join("camdl-store"));
    assert_eq!(free_prov["N0"]["role"], "estimated", "{free_prov}");
    assert_ne!(held_problem, free_problem,
        "holding N0 changes the problem the profile is of");
}

/// A profile's scenario that sets the focal parameter is replaced by the grid,
/// and says so once per run (not once per grid point).
#[test]
fn profile_scenario_setting_the_focal_parameter_is_an_info_notice() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "pf_focal", "", "beta = { bounds = [0.01, 1.0], start = 0.3 }", "N0 = 1000");
    let cwd = dir.join("focal");
    std::fs::create_dir_all(&cwd).unwrap();
    let out = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(&cwd)
        .args(["profile", &dir.join("m.ir.json").to_string_lossy(),
               "--fit", &dir.join("pf_focal.toml").to_string_lossy()])
        .args(["--sweep", "gamma=0.1,0.5", "--scenario", "fastg", "--particles", "50",
               "--iterations", "3", "--n-starts", "1", "--seed", "1", "--rw-sd", "beta=0.02"])
        .output()
        .expect("spawn camdl profile");
    let err = stderr(&out);
    assert!(out.status.success(), "profile --scenario fastg: {err}");
    let line = "info: scenario `fastg` sets gamma = 0.9; gamma is profiled, so the grid values \
                replace it";
    assert_eq!(err.matches(line).count(), 1, "want {line:?} once; stderr:\n{err}");
}
