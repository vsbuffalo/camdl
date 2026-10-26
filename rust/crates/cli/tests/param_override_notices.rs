//! `simulate` reports every replacement of one parameter tier's value by
//! another's — on stderr, once per run, and in each leaf's `run.json`.
//!
//! The rule (docs/camdl-run-spec.md §1.3): no tier may replace a value a lower
//! tier supplied, with a different value, without the user being told. The
//! case that motivated it: a params file gives `I0 = 5`, a scenario sets
//! `I0 = 3`, and the run used 3 with no message.

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

/// Infection with a parameterised initial infected count. `lockdown` sets
/// `I0 = 3`.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I }
parameters {
  beta : rate in [0.0, 5.0]
  I0 : count in [0, 100]
}
transitions { infection : S --> I @ beta * S * I / (S + I) }
init { S = 1000 - I0  I = I0 }
simulate { from = 0 'days  to = 5 'days }

scenarios {
  lockdown { set = { I0 = 3 } }
}
"#;

struct Run {
    stderr: String,
    store: PathBuf,
}

/// Run `simulate` on [`MODEL`] in `dir` with `args`, chain-binomial at dt 1,
/// seed 1, into a fresh store under `dir/out`.
fn simulate(dir: &Path, args: &[&str]) -> Run {
    let bin = skip_if_missing_binary();
    let model = dir.join("m.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let store = dir.join("out");
    let out = Command::new(&bin)
        .arg("simulate")
        .arg(&model)
        .args(args)
        .args(["--backend", "chain_binomial", "--dt", "1", "--seed", "1"])
        .arg("--output-dir")
        .arg(&store)
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "simulate {args:?} failed: {stderr}");
    Run { stderr, store }
}

fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, contents).unwrap();
    p
}

/// Every leaf `run.json` under `root`, parsed.
fn run_jsons(root: &Path) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut stack = vec![root.join("sims")];
    while let Some(dir) = stack.pop() {
        let rj = dir.join("run.json");
        if rj.is_file() {
            out.push(serde_json::from_str(&std::fs::read_to_string(rj).unwrap()).unwrap());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    out
}

/// Lines of `stderr` that report a parameter override.
fn override_lines(stderr: &str) -> Vec<&str> {
    stderr.lines().filter(|l| l.contains(" replaced ")).collect()
}

#[test]
fn a_scenario_over_a_params_file_is_reported_and_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(tmp.path(), "p.toml", "beta = 0.3\nI0 = 5\n");
    let run = simulate(tmp.path(), &["--params", &p.to_string_lossy(), "--scenario", "lockdown"]);

    let line = format!("info: scenario `lockdown` replaced I0 = 5 (from {}) with 3", p.display());
    assert_eq!(override_lines(&run.stderr), vec![line.as_str()], "stderr: {}", run.stderr);

    let leaves = run_jsons(&run.store);
    assert_eq!(leaves.len(), 1);
    let inputs = &leaves[0]["inputs"];
    // The run used the scenario's value.
    assert_eq!(inputs["params"]["I0"], 3.0);
    let i0 = &inputs["parameters_provenance"]["I0"];
    assert_eq!(i0["source"], "scenario");
    assert_eq!(
        i0["overrides"],
        serde_json::json!([{
            "replaced": { "source": "fixed_file", "value": 5.0, "path": p.display().to_string() },
            "by": { "source": "scenario", "value": 3.0, "scenario": "lockdown" },
        }])
    );
    assert!(inputs["parameters_provenance"]["beta"].get("overrides").is_none());
    assert_eq!(inputs["notices"][0]["code"], "param-override");
    assert_eq!(inputs["notices"][0]["severity"], "info");
}

#[test]
fn a_later_params_file_over_an_earlier_one_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let a = write(tmp.path(), "a.toml", "beta = 0.3\nI0 = 5\n");
    let b = write(tmp.path(), "b.toml", "I0 = 4\n");
    let run = simulate(
        tmp.path(),
        &["--params", &a.to_string_lossy(), "--params", &b.to_string_lossy()],
    );
    let line = format!(
        "info: params file {} replaced I0 = 5 (from {}) with 4",
        b.display(),
        a.display()
    );
    assert_eq!(override_lines(&run.stderr), vec![line.as_str()], "stderr: {}", run.stderr);
}

#[test]
fn param_over_a_scenario_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(tmp.path(), "p.toml", "beta = 0.3\n");
    let run = simulate(
        tmp.path(),
        &["--params", &p.to_string_lossy(), "--scenario", "lockdown", "--param", "I0=7"],
    );
    assert_eq!(
        override_lines(&run.stderr),
        vec!["info: --param/--fixed replaced I0 = 3 (from scenario `lockdown`) with 7"],
        "stderr: {}",
        run.stderr
    );
}

/// The tamper hash a fit writes into `[provenance] content_hash`
/// (`fit::provenance::mle_params_tamper_hash`), so the fixture is a fit's
/// unedited file, which `simulate --params` checks before reading the block.
fn tamper_hash(params: &[(&str, f64)]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted = params.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = Sha256::new();
    for (name, value) in sorted {
        h.update(format!("{}={:.12}\x00", name, value).as_bytes());
    }
    hex::encode(&h.finalize()[..4])
}

#[test]
fn a_fits_estimate_replaced_by_a_scenario_is_a_warning() {
    let tmp = tempfile::tempdir().unwrap();
    let mle = write(
        tmp.path(),
        "mle_params.toml",
        &format!(
            r#"beta = 0.3
I0 = 5

[provenance]
camdl_version = "test"
timestamp = "2026-10-25T00:00:00Z"
content_hash = "{hash}"
backend = "chain_binomial"
dt = 1.0
model = "m.camdl"
model_identity = "f00d"
seed = 1
method = "if2"
chain = 1
log_likelihood = -10.0
loglik_sd = 0.0
n_particles = 100
"#,
            hash = tamper_hash(&[("beta", 0.3), ("I0", 5.0)])
        ),
    );
    let run = simulate(tmp.path(), &["--params", &mle.to_string_lossy(), "--scenario", "lockdown"]);
    let line = format!(
        "warning: scenario `lockdown` replaced I0 = 5 (from {}, a fit's estimate) with 3",
        mle.display()
    );
    assert_eq!(override_lines(&run.stderr), vec![line.as_str()], "stderr: {}", run.stderr);
}

#[test]
fn an_equal_value_replacement_is_not_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(tmp.path(), "p.toml", "beta = 0.3\nI0 = 3\n");
    let run = simulate(tmp.path(), &["--params", &p.to_string_lossy(), "--scenario", "lockdown"]);
    assert!(override_lines(&run.stderr).is_empty(), "stderr: {}", run.stderr);
    let leaves = run_jsons(&run.store);
    assert!(leaves[0]["inputs"]["parameters_provenance"]["I0"].get("overrides").is_none());
}

#[test]
fn a_multi_cell_sweep_reports_each_override_once() {
    let tmp = tempfile::tempdir().unwrap();
    let p = write(tmp.path(), "p.toml", "beta = 0.3\nI0 = 5\n");
    let run = simulate(
        tmp.path(),
        &[
            "--params", &p.to_string_lossy(),
            "--scenario", "lockdown",
            "--sweep", "beta=lin(0.2,0.4,3)",
        ],
    );
    let beta = format!(
        "info: the draw or sweep point replaced beta = 0.3 (from {}) with 0.2 to 0.4, across 3 cells",
        p.display()
    );
    let i0 = format!("info: scenario `lockdown` replaced I0 = 5 (from {}) with 3", p.display());
    let mut lines = override_lines(&run.stderr);
    lines.sort();
    let mut want = vec![beta.as_str(), i0.as_str()];
    want.sort();
    assert_eq!(lines, want, "stderr: {}", run.stderr);

    // Every leaf records its own overrides.
    let leaves = run_jsons(&run.store);
    assert_eq!(leaves.len(), 3);
    for leaf in &leaves {
        let pp = &leaf["inputs"]["parameters_provenance"];
        assert_eq!(pp["I0"]["overrides"][0]["replaced"]["value"], 5.0);
        assert_eq!(pp["beta"]["overrides"][0]["by"]["source"], "sweep_point");
    }
}
