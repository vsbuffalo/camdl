//! `[fixed] from_scenario = "s"` imports exactly the parameter values `camdl
//! simulate --scenario s` runs at: the scenario's `set` values with its
//! `scale` factors applied, through composed presets.
//!
//! The import used to copy the `set` values only, so a fit under a scenario
//! that scales a parameter it sets held that parameter at the unscaled value
//! (gamma = 0.15 where simulate ran at 0.3), with nothing said. A scenario
//! that scales a parameter it does not set has no value for the factor to
//! multiply on the import side, and is refused.

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
  beta  : rate     in [0.01, 2.0]
  gamma : rate     in [0.01, 1.0]
  k     : positive in [1.0, 100.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  prev {
    columns   { time : time, prev : count }
    projected = prevalence(I)
    prev      ~ neg_binomial(mean = projected, r = k)
  }
}
init { S = 990  I = 10  R = 0 }
scenarios {
  s      { set = { gamma = 0.15  k = 10.0 }  scale = { gamma = 2.0 } }
  base   { set = { gamma = 0.15  k = 10.0 } }
  dbl    { compose = [base]  scale = { gamma = 2.0 } }
  sc     { scale = { gamma = 2.0 } }
  sub    { compose = [sc]  set = { gamma = 0.15  k = 10.0 } }
  noset  { set = { k = 10.0 }  scale = { gamma = 2.0 } }
}
simulate { from = 0 'days  to = 10 'days }
"#;

const PREV_TSV: &str = "time\tprev\n1\t10\n2\t12\n3\t13\n4\t15\n5\t16\n6\t18\n7\t19\n8\t20\n9\t21\n10\t22\n";

fn setup(dir: &Path) {
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("prev.tsv"), PREV_TSV).unwrap();
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

/// Write `<name>.toml` estimating `beta` with `[fixed] from_scenario = scen`.
fn fit_toml(dir: &Path, name: &str, scen: &str) -> PathBuf {
    let p = dir.join(format!("{name}.toml"));
    std::fs::write(&p, format!(
        "output_dir = \"res_{name}\"\n[model]\ncamdl = \"m.ir.json\"\n\
         [data.observations]\nprev = \"prev.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.01, 2.0], start = 0.5 }}\n\
         [fixed]\nfrom_scenario = \"{scen}\"\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
         particles = 50\niterations = 3\ncooling = 0.9\n"
    )).unwrap();
    p
}

/// Every file named `file` under `root`.
fn find(root: &Path, file: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() { stack.push(p) } else if p.file_name().is_some_and(|n| n == file) { out.push(p) }
        }
    }
    out
}

/// The value `simulate --scenario scen` runs `param` at (run.json `inputs.params`).
fn simulate_value(dir: &Path, scen: &str, param: &str) -> f64 {
    let out = camdl(dir, &["simulate", "m.ir.json", "--scenario", scen, "--param", "beta=0.3",
                           "--output-dir", &format!("sim_{scen}")]);
    assert!(out.status.success(), "simulate --scenario {scen}: {}", stderr(&out));
    let runs = find(&dir.join(format!("sim_{scen}")), "run.json");
    assert_eq!(runs.len(), 1, "one run.json: {runs:?}");
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&runs[0]).unwrap()).unwrap();
    j["inputs"]["params"][param].as_f64().expect("param in run.json inputs.params")
}

/// The value a `from_scenario` fit holds `param` at: `mle_params.toml` and
/// `fit.meta.json` `parameters_provenance` (role fixed, source fit_toml_fixed).
fn fit_fixed_value(dir: &Path, name: &str, scen: &str, param: &str) -> f64 {
    fit_toml(dir, name, scen);
    let out = camdl(dir, &["fit", "run", &format!("{name}.toml"), "--seed", "1"]);
    assert!(out.status.success(), "fit run {name}: {}", stderr(&out));
    let root = dir.join(format!("res_{name}"));
    let mle = find(&root, "mle_params.toml");
    assert_eq!(mle.len(), 1, "one mle_params.toml: {mle:?}");
    let t: toml::Table = std::fs::read_to_string(&mle[0]).unwrap().parse().unwrap();
    let from_mle = t[param].as_float().expect("param in mle_params.toml");
    let meta = find(&root, "fit.meta.json");
    assert_eq!(meta.len(), 1, "one fit.meta.json: {meta:?}");
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta[0]).unwrap()).unwrap();
    let prov = &j["parameters_provenance"][param];
    assert_eq!(prov["source"], "fit_toml_fixed", "{prov}");
    assert_eq!(prov["role"], "fixed", "{prov}");
    assert_eq!(prov["value"].as_f64(), Some(from_mle), "meta and mle_params agree: {prov}");
    from_mle
}

#[test]
fn from_scenario_applies_the_scenarios_scale() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let sim = simulate_value(dir, "s", "gamma");
    assert!((sim - 0.3).abs() < 1e-12, "simulate --scenario s runs gamma at 0.3, got {sim}");
    let fit = fit_fixed_value(dir, "fs_s", "s", "gamma");
    assert_eq!(fit, sim, "the from_scenario fit holds gamma at simulate's value");
}

/// A scale from the composing parent over a composed `set`, and a composed
/// sub-preset's scale under the parent's own `set`, both land on 0.3.
#[test]
fn from_scenario_applies_scale_through_composed_presets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    for scen in ["dbl", "sub"] {
        let sim = simulate_value(dir, scen, "gamma");
        assert!((sim - 0.3).abs() < 1e-12, "simulate --scenario {scen}: gamma 0.3, got {sim}");
        let fit = fit_fixed_value(dir, &format!("fs_{scen}"), scen, "gamma");
        assert_eq!(fit, sim, "from_scenario {scen}: fit gamma equals simulate's");
    }
}

#[test]
fn from_scenario_refuses_a_scale_on_a_parameter_the_scenario_does_not_set() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    fit_toml(dir, "fs_noset", "noset");
    let out = camdl(dir, &["fit", "run", "fs_noset.toml", "--seed", "1"]);
    let err = stderr(&out);
    assert!(!out.status.success(), "must be refused; stderr:\n{err}");
    for line in [
        "error: scenario `noset` scales parameter `gamma` but does not set it, and [fixed] \
         from_scenario imports only the values a scenario sets: the scale has no value to \
         multiply",
        "  scenario `noset`:  scale `gamma` by 2",
        "Fix: set `gamma` in scenario `noset` as well, so the scale applies to a value the \
         scenario gives.",
    ] {
        assert!(err.contains(line), "diagnostic lacks {line:?}; stderr:\n{err}");
    }
    assert!(!dir.join("res_fs_noset").exists(), "a refused run writes nothing");
}
