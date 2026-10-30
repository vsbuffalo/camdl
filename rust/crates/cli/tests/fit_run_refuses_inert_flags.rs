//! `fit run` refuses an explicit option that the fit would read nowhere,
//! rather than run, re-key, and leave the user to infer that it did nothing.
//!
//! The per-method override table is unit-tested in `config_v2`; this file
//! covers the CLI wiring and the two refusals that live in the driver:
//! `--seed` beside a file's `fit_seeds`, and `--allow-nonconverged-source`
//! with a starts rule that has no stored verdict to lift.
//!
//! End-to-end via the built `camdl` binary. A missing release binary or
//! `camdlc.exe` fails the test (run `make build`, or `make test-fast`, which
//! builds both) rather than passing it without running.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_inert_flags_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// Write a small SIR model and an `nl-sbplx` fit of it; `extra` is appended
/// to the top level of the fit file. Returns the fit file's path.
fn write_fit(tmp: &TempDir, extra: &str) -> PathBuf {
    write_fit_with_model_dt(tmp, extra, None)
}

/// [`write_fit`] with the model's `simulate {}` declaring `dt = <model_dt>`.
fn write_fit_with_model_dt(tmp: &TempDir, extra: &str, model_dt: Option<f64>) -> PathBuf {
    let camdlc = camdlc().expect(
        "camdlc.exe is missing: run `make build-ocaml` (this test must not pass without running)");
    assert!(camdl_bin().exists(),
        "the release camdl binary is missing: run `make build-rust` (this test must not pass \
         without running)");
    let dir = &tmp.0;
    let model = dir.join("sir.camdl");
    std::fs::write(&model, r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 20 'days SIMDT}
"#.replace("SIMDT", &model_dt.map(|d| format!(" dt = {d} 'days ")).unwrap_or_default())).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("sir.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let mut tsv = String::from("time\tcases\n");
    for (k, c) in [12, 15, 19, 24, 30, 37, 45, 54, 63, 71].iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", 2 * (k + 1), c));
    }
    let data = dir.join("cases.tsv");
    std::fs::write(&data, tsv).unwrap();
    let fit = dir.join("fit.toml");
    std::fs::write(&fit, format!(r#"
output_dir = "{out}"
{extra}

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = 1.5 }}

[fixed]
gamma = 0.3
N0 = 10000

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
starts = "single"
"#, out = dir.join("out").display(), ir = ir.display(), data = data.display())).unwrap();
    fit
}

fn run(fit: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(camdl_bin())
        .args(["fit", "run"]).arg(fit).args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn a_flag_the_method_does_not_read_is_refused() {
    let tmp = tempdir("method");
    let fit = write_fit(&tmp, "");
    let (ok, err) = run(&fit, &["--cooling-target-iters", "20"]);
    assert!(!ok, "--cooling-target-iters on nl-sbplx must be refused:\n{err}");
    assert!(err.contains("--cooling-target-iters") && err.contains("`nl-sbplx`")
        && err.contains("`if2`"), "names the flag, the method and who reads it:\n{err}");
    assert!(!tmp.0.join("out").exists(), "refused before any work");
}

#[test]
fn seed_beside_fit_seeds_is_refused() {
    let tmp = tempdir("seed");
    let fit = write_fit(&tmp, "fit_seeds = [3]");
    let (ok, err) = run(&fit, &["--seed", "7"]);
    assert!(!ok, "--seed beside fit_seeds must be refused:\n{err}");
    assert!(err.contains("--seed") && err.contains("fit_seeds"), "{err}");
}

#[test]
fn allow_nonconverged_source_without_a_fit_source_is_refused() {
    let tmp = tempdir("source");
    let fit = write_fit(&tmp, "");
    let (ok, err) = run(&fit, &["--allow-nonconverged-source"]);
    assert!(!ok, "--allow-nonconverged-source with `starts = \"single\"` must be refused:\n{err}");
    assert!(err.contains("--allow-nonconverged-source") && err.contains("`single`"), "{err}");
}

/// An nl-* leaf records its optimizer's own convergence verdict, which the
/// warm-start source check and `fit table` read (its `tail_chain_agreement`
/// holds relative ranges, not IF2's Â, and cannot be judged as Â).
#[test]
fn an_nlopt_fit_records_its_convergence_verdict() {
    let tmp = tempdir("nl_verdict");
    let fit = write_fit(&tmp, "");
    let (ok, err) = run(&fit, &["--no-dt-check"]);
    assert!(ok, "the fit runs:\n{err}");
    let mut stack = vec![tmp.0.join("out")];
    let mut found = None;
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p) }
            else if p.file_name().and_then(|n| n.to_str()) == Some("fit_state.toml") { found = Some(p) }
        }
    }
    let state = std::fs::read_to_string(found.expect("a fit_state.toml")).unwrap();
    assert!(state.contains("[optimizer_convergence]") && state.contains("chain_agreement_failed"),
        "{state}");
}

/// The inert-key refusal runs at `fit run` (not in the shared parser, which
/// also reads stored fits' archived configs): an `mh` method with
/// `adapt = false` beside `adapt_start` is refused before any work.
#[test]
fn an_inert_method_key_is_refused_at_fit_run() {
    let tmp = tempdir("inert_key");
    let fit = write_fit(&tmp, "");
    let src = std::fs::read_to_string(&fit).unwrap()
        .replace("beta = { bounds = [0.05, 5.0], start = 1.5 }",
                 "beta = { bounds = [0.05, 5.0], start = 1.5, prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }")
        .replace("algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\n",
                 "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 10\nadapt = false\nadapt_start = 5\n");
    assert!(src.contains("adapt_start = 5"), "fixture rewrite applied:\n{src}");
    std::fs::write(&fit, src).unwrap();
    let (ok, err) = run(&fit, &[]);
    assert!(!ok, "adapt_start beside adapt = false must be refused:\n{err}");
    assert!(err.contains("adapt_start") && err.contains("no effect"), "{err}");
    assert!(!tmp.0.join("out").exists(), "refused before any work");
}

/// An ODE method integrates at the model's declared dt (`runner::ode_step_dt`),
/// so a `[config] dt` written in the file that differs would be ignored:
/// refused. One that agrees, or a model that declares none, runs.
#[test]
fn an_ode_fit_refuses_a_config_dt_the_model_overrides() {
    let tmp = tempdir("cfg_dt");
    let fit = write_fit_with_model_dt(&tmp, "[config]\ndt = 0.25\n", Some(0.5));
    let (ok, err) = run(&fit, &["--no-dt-check"]);
    assert!(!ok, "[config] dt = 0.25 beside the model's dt = 0.5 must be refused:\n{err}");
    assert!(err.contains("dt = 0.25") && err.contains("0.5"), "{err}");

    let tmp = tempdir("cfg_dt_same");
    let fit = write_fit_with_model_dt(&tmp, "[config]\ndt = 0.5\n", Some(0.5));
    let (ok, err) = run(&fit, &["--no-dt-check"]);
    assert!(ok, "an agreeing [config] dt runs:\n{err}");
}
