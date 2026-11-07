//! gh#571: a tier-5 value — `simulate --param NAME=VALUE`, or `--fixed
//! NAME=VALUE` on `survey` / `profile` — on a parameter a selected scenario
//! `set`s or `scale`s is refused before anything runs.
//!
//! Precedence puts tier 5 above the scenario (run spec §1.3), so the flag
//! would replace the scenario's value in every cell: three scenarios that set
//! `ce` to three values would all run the flag's value and come out
//! bit-identical, each still labelled as a different scenario. The negative
//! controls: a scenario that does not touch the flagged parameter runs, a
//! scenario whose `set` equals the flag's value runs, and a `--params` file
//! (tier 3) under a scenario runs, because a scenario overriding a parameter
//! file is the intended layering.
//!
//! CLI-level on purpose: the defect class is a guard some command path does
//! not reach, which a test of the guard function alone cannot see.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

/// An SIR whose transmission is multiplied by `ce`, with scenarios touching
/// `ce` in each way the guard has to see:
///
/// - `baseline`: touches nothing (needs `ce` from somewhere else).
/// - `c25`, `c50`: `set ce` directly.
/// - `halve`: `scale ce` directly.
/// - `combo`: sets only `gamma` itself, and gets `ce` from the composed `c25`.
/// - `slow`: touches only `gamma`.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]
  gamma : rate        in [0.01, 0.5]
  ce    : probability in [0, 1]
  rho   : probability in [0.1, 0.9]
}
transitions {
  infection : S --> I  @ ce * beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}
init { S = 990  I = 10 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ poisson(rate = rho * projected)
  }
}
simulate { from = 0 'days  to = 28 'days }
scenarios {
  baseline { label = "no change" }
  c25      { set = { ce = 0.75 } }
  c50      { set = { ce = 0.5 } }
  halve    { scale = { ce = 0.5 } }
  slow     { set = { gamma = 0.2 } }
  combo    { compose = [c25]  set = { gamma = 0.2 } }
}
"#;

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("m.camdl"), MODEL).unwrap();
        std::fs::write(p.join("p.toml"), "beta = 0.4\ngamma = 0.15\nrho = 0.5\n").unwrap();
        std::fs::write(p.join("pce.toml"), "beta = 0.4\ngamma = 0.15\nrho = 0.5\nce = 1.0\n")
            .unwrap();
        std::fs::write(p.join("cases.tsv"), "time\tweekly_cases\n7\t20\n14\t40\n21\t60\n28\t50\n")
            .unwrap();
        Fixture { dir }
    }

    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(skip_if_missing_binary())
            .args(args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_OUTPUT_DIR", self.dir.path().join("res"))
            .output()
            .expect("spawn camdl")
    }

    /// `simulate m.camdl --params <params> --backend ode --stdout --scenario
    /// <scenarios> <flags…>`.
    fn simulate(&self, params: &str, scenarios: &str, flags: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", params, "--backend", "ode", "--stdout",
            "--scenario", scenarios,
        ];
        args.extend_from_slice(flags);
        self.camdl(&args)
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Assert `out` failed and its stderr contains `expected` verbatim.
fn assert_refused_with(out: &Output, what: &str, expected: &str) {
    let err = stderr(out);
    assert!(!out.status.success(), "{what}: must be refused, but exited 0; stderr:\n{err}");
    assert!(err.contains(expected), "{what}: diagnostic differs.\nexpected:\n{expected}\nstderr:\n{err}");
}

fn assert_runs(out: &Output, what: &str) {
    let err = stderr(out);
    assert!(out.status.success(), "{what}: must run, but failed; stderr:\n{err}");
    assert!(!err.contains("would replace the value"), "{what}: refused-looking stderr:\n{err}");
}

/// The reported case: two scenarios set the parameter, a third (`baseline`)
/// does not and is the one that needed the flag's value.
#[test]
fn param_over_two_setting_scenarios_is_refused() {
    let f = Fixture::new();
    let out = f.simulate("p.toml", "baseline,c25,c50", &["--param", "ce=1.0"]);
    assert_refused_with(&out, "--param over two sets", "\
error: --param ce=1.0 would replace the value two selected scenarios set
  scenario 'c25' sets ce = 0.75
  scenario 'c50' sets ce = 0.5
  A --param value overrides every scenario, so these scenarios would all run
  with ce = 1.0 instead of their own values. The output would still label them
  as different scenarios, with nothing to show the override.
hint: put the value in the model's scenarios {} block instead:
  - if only the scenario that doesn't set it needs it (here 'baseline'):
        baseline { set = { ce = 1.0 } }
  - if you want a variant of 'c25' with ce = 1.0:
        c25_variant { compose = [c25], set = { ce = 1.0 } }
  or drop --param ce to run each scenario with its own value.
");
}

#[test]
fn param_over_one_setting_scenario_is_refused() {
    let f = Fixture::new();
    let out = f.simulate("p.toml", "c25", &["--param", "ce=1.0"]);
    assert_refused_with(&out, "--param over one set", "\
error: --param ce=1.0 would replace the value scenario 'c25' sets
  scenario 'c25' sets ce = 0.75
  A --param value overrides every scenario, so scenario 'c25' would run with
  ce = 1.0 instead of its own value. The output would still label it as
  scenario 'c25', with nothing to show the override.
hint: put the value in the model's scenarios {} block instead:
  - if you want a variant of 'c25' with ce = 1.0:
        c25_variant { compose = [c25], set = { ce = 1.0 } }
  or drop --param ce to run scenario 'c25' with its own value.
");
}

#[test]
fn param_over_a_scaling_scenario_is_refused() {
    let f = Fixture::new();
    let out = f.simulate("p.toml", "halve", &["--param", "ce=1.0"]);
    assert_refused_with(&out, "--param over a scale", "\
error: --param ce=1.0 would replace the value scenario 'halve' scales
  scenario 'halve' scales ce by 0.5
  A --param value overrides every scenario, so scenario 'halve' would run with
  ce = 1.0 instead of its own value. The output would still label it as
  scenario 'halve', with nothing to show the override.
hint: put the value in a --params file instead; a scenario scales a file's value
  rather than being overridden by it:
        ce = 1.0
  or drop --param ce to run scenario 'halve' with its own value.
");
}

/// A `set` reached through `compose`, beside a `scale`: both are named, the
/// set attributed to the composed scenario that carries it.
#[test]
fn param_over_a_composed_set_and_a_scale_is_refused() {
    let f = Fixture::new();
    let out = f.simulate("p.toml", "baseline,halve,combo", &["--param", "ce=1.0"]);
    assert_refused_with(&out, "--param over compose + scale", "\
error: --param ce=1.0 would replace the value two selected scenarios set or scale
  scenario 'halve' scales ce by 0.5
  scenario 'combo' sets ce = 0.75 (through composed scenario 'c25')
");
}

/// `--dry-run` plans without reaching the engine; it refuses the same run.
#[test]
fn dry_run_refuses_the_same_collision() {
    let f = Fixture::new();
    let out = f.simulate("p.toml", "c25", &["--param", "ce=1.0", "--dry-run"]);
    assert_refused_with(&out, "--dry-run", "error: --param ce=1.0 would replace the value scenario 'c25' sets");
}

#[test]
fn param_on_a_parameter_no_scenario_touches_runs() {
    let f = Fixture::new();
    assert_runs(&f.simulate("p.toml", "baseline,slow", &["--param", "ce=1.0"]), "untouched");
}

/// A `set` to the flag's own value: the two agree, nothing is discarded.
#[test]
fn param_equal_to_the_scenario_value_runs() {
    let f = Fixture::new();
    assert_runs(&f.simulate("p.toml", "baseline,c25", &["--param", "ce=0.75"]), "equal value");
}

/// Tier 3: a scenario overriding a `--params` file is the intended layering.
#[test]
fn params_file_under_scenarios_runs_and_each_scenario_applies() {
    let f = Fixture::new();
    let out = f.simulate("pce.toml", "baseline,c25,c50", &[]);
    assert_runs(&out, "--params file");
    let err = stderr(&out);
    assert!(err.contains("scenario `c25` replaced ce = 1 (from pce.toml) with 0.75"), "{err}");
    assert!(err.contains("scenario `c50` replaced ce = 1 (from pce.toml) with 0.5"), "{err}");
}

#[test]
fn survey_fixed_over_a_setting_scenario_is_refused() {
    let f = Fixture::new();
    let out = f.camdl(&[
        "survey", "m.camdl", "--data", "cases.tsv", "--estimate", "beta=0.1:0.9",
        "--fixed", "gamma=0.15", "--fixed", "rho=0.5", "--fixed", "ce=1.0",
        "--scenario", "c25", "--n-points", "4",
    ]);
    assert_refused_with(&out, "survey --fixed", "\
error: --fixed ce=1.0 would replace the value scenario 'c25' sets
  scenario 'c25' sets ce = 0.75
  A --fixed value overrides every scenario, so scenario 'c25' would run with
  ce = 1.0 instead of its own value.");
}

#[test]
fn profile_fixed_over_a_scaling_scenario_is_refused() {
    let f = Fixture::new();
    let out = f.camdl(&[
        "profile", "m.camdl", "--data", "cases.tsv", "--sweep", "beta=0.3,0.4",
        "--rw-sd", "gamma=auto", "--particles", "50", "--n-starts", "1",
        "--fixed", "rho=0.5", "--fixed", "ce=1.0", "--fixed", "gamma=0.15",
        "--scenario", "halve",
    ]);
    assert_refused_with(&out, "profile --fixed", "\
error: --fixed ce=1.0 would replace the value scenario 'halve' scales
  scenario 'halve' scales ce by 0.5");
    assert!(stderr(&out).contains("put the value in a --fixed-file file instead"), "{}", stderr(&out));
}
