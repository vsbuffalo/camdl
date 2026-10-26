//! `fit run` refuses a fit.toml whose scenario disagrees with its own
//! `[fixed]` block, or with a `--sweep`, on a parameter.
//!
//! The resolver puts the scenario (tier 4) above `[fixed]` (tier 2) and a sweep
//! point (tier 3.5), docs/camdl-run-spec.md §1.3. On the same parameter that
//! would silently discard a value the user wrote, so the combination is refused
//! before anything is written. Where the two agree, or the scenario touches
//! nothing the user wrote, the fit runs at the resolver's value, which is the
//! scenario's.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95]
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = 10.0)
  }
}
scenarios {
  fastrec   { set = { gamma = 0.3 } }
  doublerec { scale = { gamma = 2.0 } }
}
simulate { from = 0 'days  to = 56 'days }
"#;

fn fit_toml(out: &str, gamma: f64, scenario: &str) -> String {
    format!(r#"output_dir = "{out}"
{scenario}
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
[fixed]
gamma = {gamma}
N0  = 10000
I0  = 10
rho = 0.6
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 5
cooling = 0.7
"#)
}

fn fixture() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"),
        "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n")
        .unwrap();
    tmp
}

/// Write `<name>.toml` (output under `res_<name>`) and run `fit run` on it.
fn run(dir: &Path, name: &str, gamma: f64, scenario: &str, extra: &[&str]) -> Output {
    std::fs::write(dir.join(format!("{name}.toml")), fit_toml(&format!("res_{name}"), gamma, scenario))
        .unwrap();
    Command::new(binary())
        .args(["fit", "run", &format!("{name}.toml"), "--seed", "1"])
        .args(extra)
        .current_dir(dir).output().expect("spawn")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Run the fit, require success, and return its `mle_params.toml` values (the
/// `[provenance]` block dropped).
fn fit(dir: &Path, name: &str, gamma: f64, scenario: &str) -> toml::Table {
    let out = run(dir, name, gamma, scenario, &[]);
    assert!(out.status.success(), "fit {name}: {}", stderr(&out));
    let mut stack = vec![dir.join(format!("res_{name}"))];
    while let Some(d) = stack.pop() {
        let f = d.join("mle_params.toml");
        if f.is_file() {
            let mut t: toml::Table = toml::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
            t.remove("provenance");
            return t;
        }
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            if e.path().is_dir() { stack.push(e.path()); }
        }
    }
    panic!("no mle_params.toml for {name}");
}

/// The run was refused: non-zero exit, every `line` in the diagnostic, and no
/// output directory.
fn assert_refused(dir: &Path, name: &str, out: &Output, lines: &[&str]) {
    let err = stderr(out);
    assert!(!out.status.success(), "{name}: must be refused, but exited 0; stderr:\n{err}");
    for line in lines {
        assert!(err.contains(line), "{name}: diagnostic lacks {line:?}; stderr:\n{err}");
    }
    assert!(!dir.join(format!("res_{name}")).exists(),
        "{name}: a refused fit must write nothing; stderr:\n{err}");
}

#[test]
fn a_scenario_setting_a_fixed_parameter_to_another_value_is_refused() {
    let tmp = fixture();
    let out = run(tmp.path(), "clash", 0.2, r#"scenario = "fastrec""#, &[]);
    assert_refused(tmp.path(), "clash", &out, &[
        "error: parameter `gamma` is given two values: by scenario `fastrec` and by [fixed] in `clash.toml`",
        "[fixed]:   gamma = 0.2",
        "scenario:  gamma = 0.3    (set gamma = 0.3)",
        "Fix: remove `gamma` from scenario `fastrec`",
        "give [fixed] in `clash.toml` the scenario's value: `gamma = 0.3`",
    ]);
}

#[test]
fn a_scenario_scaling_a_fixed_parameter_is_refused() {
    let tmp = fixture();
    let out = run(tmp.path(), "scaled", 0.2, r#"scenario = "doublerec""#, &[]);
    assert_refused(tmp.path(), "scaled", &out, &[
        "error: parameter `gamma` is given two values: by scenario `doublerec`",
        "[fixed]:   gamma = 0.2",
        "scenario:  gamma = 0.4    (scale gamma \u{d7} 2)",
    ]);
}

/// The scenario and `[fixed]` agree on gamma = 0.3: the fit runs at 0.3, and
/// its estimate is the one `[fixed] gamma = 0.3` gives with no scenario, and
/// differs from the estimate at gamma = 0.2 (so the 0.3 is used, not just
/// recorded).
#[test]
fn a_scenario_agreeing_with_fixed_fits_at_that_value() {
    let tmp = fixture();
    let dir = tmp.path();
    let with_scenario = fit(dir, "agree", 0.3, r#"scenario = "fastrec""#);
    let fixed_03 = fit(dir, "fixed03", 0.3, "");
    let fixed_02 = fit(dir, "fixed02", 0.2, "");

    assert_eq!(with_scenario["gamma"].as_float(), Some(0.3));
    assert_eq!(with_scenario, fixed_03,
        "a scenario agreeing with [fixed] gamma = 0.3 must fit exactly as [fixed] alone");
    assert_ne!(with_scenario["beta"], fixed_02["beta"],
        "beta at gamma = 0.3 must differ from beta at gamma = 0.2");
}

/// A scenario touching only a parameter the sweep does not vary runs every
/// sweep point; one touching the swept parameter is refused, even where the
/// scenario agrees with `[fixed]`.
#[test]
fn fit_run_sweep_over_a_parameter_the_scenario_sets_is_refused() {
    let tmp = fixture();
    let dir = tmp.path();
    let out = run(dir, "swept", 0.3, r#"scenario = "fastrec""#, &["--sweep", "gamma=0.1,0.2"]);
    assert_refused(dir, "swept", &out, &[
        "error: parameter `gamma` is controlled by both the sweep and scenario `fastrec`",
        "sweep:     gamma = [0.1, 0.2]",
        "Fix: remove `gamma` from the sweep",
    ]);
}
