//! `fit run` refuses an explicit option that the fit would read nowhere,
//! rather than run, re-key, and leave the user to infer that it did nothing.
//!
//! The per-method override table is unit-tested in `config_v2`; this file
//! covers the CLI wiring and the two refusals that live in the driver:
//! `--seed` beside a file's `fit_seeds`, and `--allow-nonconverged-source`
//! with a starts rule that has no stored verdict to lift.
//!
//! End-to-end via the built `camdl` binary; skipped when the release binary or
//! `camdlc.exe` is missing.

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
fn write_fit(tmp: &TempDir, extra: &str) -> Option<PathBuf> {
    let camdlc = camdlc()?;
    if !camdl_bin().exists() {
        return None;
    }
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
simulate { from = 0 'days  to = 20 'days }
"#).unwrap();
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
    Some(fit)
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
    let Some(fit) = write_fit(&tmp, "") else { return };
    let (ok, err) = run(&fit, &["--cooling-target-iters", "20"]);
    assert!(!ok, "--cooling-target-iters on nl-sbplx must be refused:\n{err}");
    assert!(err.contains("--cooling-target-iters") && err.contains("`nl-sbplx`")
        && err.contains("`if2`"), "names the flag, the method and who reads it:\n{err}");
    assert!(!tmp.0.join("out").exists(), "refused before any work");
}

#[test]
fn seed_beside_fit_seeds_is_refused() {
    let tmp = tempdir("seed");
    let Some(fit) = write_fit(&tmp, "fit_seeds = [3]") else { return };
    let (ok, err) = run(&fit, &["--seed", "7"]);
    assert!(!ok, "--seed beside fit_seeds must be refused:\n{err}");
    assert!(err.contains("--seed") && err.contains("fit_seeds"), "{err}");
}

#[test]
fn allow_nonconverged_source_without_a_fit_source_is_refused() {
    let tmp = tempdir("source");
    let Some(fit) = write_fit(&tmp, "") else { return };
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
    let Some(fit) = write_fit(&tmp, "") else { return };
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
