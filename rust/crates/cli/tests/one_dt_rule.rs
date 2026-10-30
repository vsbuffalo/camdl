//! The one integration-step rule (`dt_rule`), end to end: the model's
//! declared `simulate { dt = … }` is the step every scoring command runs at; a
//! written `[config] dt` or `--dt` that differs is refused; an agreeing one
//! runs; 1.0 is the default only when the model declares no step.
//!
//! A missing release binary or `camdlc.exe` fails the test (run `make build`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe is missing: run `make build-ocaml`");
    assert!(camdl_bin().exists(), "the release camdl is missing: run `make build-rust`");
    p
}

pub struct TempDir(pub PathBuf);
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
pub fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_one_dt_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// A SIR model (with `dt = <model_dt>` in `simulate {}` when given), its
/// compiled IR and a prevalence series, in `dir`. Returns the IR path.
pub fn write_model(dir: &Path, model_dt: Option<f64>) -> PathBuf {
    let sim_dt = model_dt.map(|d| format!(" dt = {d} 'days ")).unwrap_or_default();
    let model = dir.join("sir.camdl");
    std::fs::write(&model, format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 9990  I = 10 }}
simulate {{ from = 0 'days  to = 20 'days {sim_dt}}}
"#)).unwrap();
    let out = Command::new(camdlc()).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("sir.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let mut tsv = String::from("time\tcases\n");
    for (k, c) in [12, 15, 19, 24, 30, 37, 45, 54, 63, 71].iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", 2 * (k + 1), c));
    }
    std::fs::write(dir.join("cases.tsv"), tsv).unwrap();
    ir
}

fn write_fit(dir: &Path, ir: &Path, config: &str, method: &str) -> PathBuf {
    let fit = dir.join("fit.toml");
    std::fs::write(&fit, format!(r#"
output_dir = "{out}"
{config}
[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = 0.4, prior = {{ log_normal = {{ mu = -1.0, sigma = 0.5 }} }} }}

[fixed]
gamma = 0.3
N0 = 10000

[method]
{method}
"#, out = dir.join("out").display(), ir = ir.display(),
        data = dir.join("cases.tsv").display())).unwrap();
    fit
}

fn fit_run(fit: &Path) -> (bool, String) {
    let out = Command::new(camdl_bin())
        .args(["fit", "run"]).arg(fit).args(["--no-progress"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
    particles = 50\niterations = 2\ncooling = 0.7\nstarts = \"single\"\n\
    [method.dt_check]\nenabled = false\n";
const PFILTER: &str = "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 50\n\
    starts = \"single\"\n";

/// A chain-binomial IF2 fit runs at the model's declared step, not the
/// `[config] dt` default of 1.0 it used to.
#[test]
fn a_chain_binomial_fit_runs_at_the_models_dt() {
    let tmp = tempdir("if2_model_dt");
    let ir = write_model(&tmp.0, Some(0.5));
    let (ok, err) = fit_run(&write_fit(&tmp.0, &ir, "", IF2));
    assert!(ok, "{err}");
    assert!(err.contains("dt=0.5"), "IF2 runs at the model's dt = 0.5:\n{err}");
}

/// A written `[config] dt` that differs from the model's is refused, for a
/// chain-binomial method as for the ODE ones; an agreeing one runs.
#[test]
fn a_differing_config_dt_is_refused_and_an_agreeing_one_runs() {
    for method in [IF2, PFILTER] {
        let tmp = tempdir("cfg_dt");
        let ir = write_model(&tmp.0, Some(0.5));
        let (ok, err) = fit_run(&write_fit(&tmp.0, &ir, "[config]\ndt = 0.25\n", method));
        assert!(!ok, "[config] dt = 0.25 beside the model's 0.5 must be refused:\n{err}");
        assert!(err.contains("[config] dt = 0.25") && err.contains("dt = 0.5"), "{err}");
        assert!(!tmp.0.join("out").exists(), "refused before any work");

        let tmp = tempdir("cfg_dt_same");
        let ir = write_model(&tmp.0, Some(0.5));
        let (ok, err) = fit_run(&write_fit(&tmp.0, &ir, "[config]\ndt = 0.5\n", method));
        assert!(ok, "an agreeing [config] dt runs:\n{err}");
    }
}

/// With no step in the model, the written `[config] dt` is the step.
#[test]
fn without_a_model_dt_the_config_dt_is_the_step() {
    let tmp = tempdir("no_model_dt");
    let ir = write_model(&tmp.0, None);
    let (ok, err) = fit_run(&write_fit(&tmp.0, &ir, "[config]\ndt = 0.5\n", IF2));
    assert!(ok, "{err}");
    assert!(err.contains("dt=0.5"), "{err}");
}

/// `camdl profile` over a model, with a `--fit` file supplying the base
/// point; `fit_config` lands in that file, `extra` on the command line.
fn profile(dir: &Path, ir: &Path, algo: &[&str], fit_config: &str, extra: &[&str]) -> (bool, String, String) {
    let fit = dir.join("pf.toml");
    std::fs::write(&fit, format!(r#"
{fit_config}
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
beta = {{ bounds = [0.05, 5.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}
[fixed]
N0 = 10000
"#, ir = ir.display(), data = dir.join("cases.tsv").display())).unwrap();
    let out = Command::new(camdl_bin())
        .args(["profile"]).arg(ir).arg("--fit").arg(&fit)
        .arg("--data").arg(dir.join("cases.tsv"))
        .args(["--sweep", "beta=0.4,0.5", "--rw-sd", "gamma=auto", "--n-starts", "1"])
        .args(algo).args(extra)
        .env("CAMDL_OUTPUT_DIR", dir.join("res"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    // The step each stored cell records.
    let mut dts = String::new();
    let mut stack = vec![dir.join("res")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p) }
            else if p.file_name().and_then(|n| n.to_str()) == Some("run.json") {
                let v: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                if let Some(dt) = v.pointer("/inputs/algorithm/dt") { dts.push_str(&format!("{dt} ")); }
            }
        }
    }
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned(), dts)
}

const PROFILE_IF2: &[&str] = &["--particles", "50", "--iterations", "2"];
const PROFILE_NL: &[&str] = &["--algorithm", "nl-sbplx", "--backend", "ode"];

/// Profile cells, IF2 and nl-*, step at the model's dt; a differing --dt or
/// --fit `[config] dt` is refused; an agreeing one runs.
#[test]
fn profile_cells_step_at_the_models_dt() {
    for algo in [PROFILE_IF2, PROFILE_NL] {
        let tmp = tempdir("profile_dt");
        let ir = write_model(&tmp.0, Some(0.5));
        let (ok, err, dts) = profile(&tmp.0, &ir, algo, "", &[]);
        assert!(ok, "{err}");
        assert!(!dts.is_empty() && dts.split_whitespace().all(|d| d == "0.5"),
            "every cell at the model's 0.5: {dts:?}\n{err}");

        let (ok, err, _) = profile(&tmp.0, &ir, algo, "", &["--dt", "0.25"]);
        assert!(!ok && err.contains("--dt = 0.25"), "{err}");
        let (ok, err, _) = profile(&tmp.0, &ir, algo, "[config]\ndt = 0.25\n", &[]);
        assert!(!ok && err.contains("[config] dt = 0.25"), "{err}");
        let (ok, err, _) = profile(&tmp.0, &ir, algo, "[config]\ndt = 0.5\n", &["--dt", "0.5"]);
        assert!(ok, "agreeing values run:\n{err}");
    }
}

fn pfilter(dir: &Path, ir: &Path, extra: &[&str]) -> (bool, String) {
    let out = Command::new(camdl_bin())
        .args(["pfilter"]).arg(ir).arg("--data").arg(dir.join("cases.tsv"))
        .args(["--particles", "50", "--param", "beta=0.4", "--param", "gamma=0.3",
               "--param", "N0=10000"])
        .args(extra)
        .env("CAMDL_OUTPUT_DIR", dir.join("res"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// `camdl pfilter` scores at the model's dt; a differing --dt is refused; an
/// agreeing one, or --dt on a model with no step, runs at it.
#[test]
fn pfilter_runs_at_the_models_dt() {
    let tmp = tempdir("pfilter_dt");
    let ir = write_model(&tmp.0, Some(0.5));
    let (ok, err) = pfilter(&tmp.0, &ir, &[]);
    assert!(ok && err.contains("dt=0.5"), "{err}");
    let (ok, err) = pfilter(&tmp.0, &ir, &["--dt", "0.25"]);
    assert!(!ok && err.contains("--dt = 0.25"), "{err}");
    let (ok, err) = pfilter(&tmp.0, &ir, &["--dt", "0.5"]);
    assert!(ok, "{err}");

    let tmp = tempdir("pfilter_no_model_dt");
    let ir = write_model(&tmp.0, None);
    let (ok, err) = pfilter(&tmp.0, &ir, &["--dt", "0.25"]);
    assert!(ok && err.contains("dt=0.25"), "{err}");
}
