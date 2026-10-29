//! A `pfilter` method scores one θ, so a `from_params` / `from_mle` point
//! that leaves an estimated parameter out is refused: the draw would fill it
//! from a bounds-uniform draw, a random θ.
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
        "camdl_pfilter_point_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}


fn run_pfilter(tmp: &TempDir, params_toml: &str) -> Option<(bool, String)> {
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
    std::fs::write(dir.join("cases.tsv"), tsv).unwrap();
    std::fs::write(dir.join("theta.toml"), params_toml).unwrap();
    let fit = dir.join("fit.toml");
    std::fs::write(&fit, format!(r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0] }}
gamma = {{ bounds = [0.01, 1.0] }}

[fixed]
N0 = 10000

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 300
starts = {{ from_params = "{theta}" }}
"#, out = dir.join("out").display(), ir = ir.display(),
        data = dir.join("cases.tsv").display(), theta = dir.join("theta.toml").display())).unwrap();
    let out = Command::new(camdl_bin())
        .args(["fit", "run"]).arg(&fit)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    Some((out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned()))
}

#[test]
fn a_pfilter_point_missing_a_parameter_is_refused() {
    let tmp = tempdir("missing");
    let Some((ok, err)) = run_pfilter(&tmp, "beta = 1.2\n") else { return };
    assert!(!ok, "a from_params point without gamma must be refused:\n{err}");
    assert!(err.contains("gives no value for gamma") && err.contains("scores the log-likelihood at one θ"),
        "refused for that reason, not by a later filter failure:\n{err}");
}

#[test]
fn a_pfilter_point_naming_every_parameter_runs() {
    let tmp = tempdir("complete");
    let Some((ok, err)) = run_pfilter(&tmp, "beta = 0.4\ngamma = 0.3\n") else { return };
    assert!(ok, "every θ given:\n{err}");
}
