//! `camdl dev eval` evaluates a forcing or a parameter by name, and refuses a
//! `let` binding by saying what it is.
//!
//! The compiler inlines a `let` into the expressions that use it, so the IR
//! has no expression for one by name. `dev eval` used to answer a request for
//! a let with "unknown expression", as if the name were a typo, while its own
//! module doc claimed it evaluated let bindings. It now names the let, prints
//! its source line, and says why it cannot be evaluated.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE").args(args).current_dir(dir).output().expect("spawn camdl")
}

const MODEL: &str = r#"
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
forcing {
  seasonal : sinusoidal 'ratio {
    baseline  = 1.0
    amplitude = 0.2
    period    = 365 'days
    phase     = 0 'days
  }
}
let N = S + I + R
let doy = mod(t, 365.0 'days) / (1.0 'days)
transitions {
  infection : S --> I @ beta * seasonal(t) * S * I / N
  recovery  : I --> R @ gamma * I
}
init {
  S = 990
  I = 10
}
simulate {
  from = 0 'days
  to   = 10 'days
  dt   = 1 'days
}
"#;

fn fixture(dir: &Path) {
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("p.toml"), "beta = 0.3\ngamma = 0.1\n").unwrap();
}

#[test]
fn a_forcing_evaluates_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path());
    let out = camdl(tmp.path(), &["dev", "eval", "m.camdl", "--params", "p.toml", "--expr", "seasonal", "--at", "0"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.lines().nth(1), Some("0\t1.000000"), "stdout: {stdout}");
}

#[test]
fn a_let_binding_is_refused_as_a_let_with_its_source_line() {
    let tmp = tempfile::tempdir().unwrap();
    fixture(tmp.path());
    let out = camdl(tmp.path(), &["dev", "eval", "m.camdl", "--params", "p.toml", "--expr", "doy", "--at", "0"]);
    assert!(!out.status.success(), "a let must be refused, not evaluated as something else");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("'doy' is a `let` binding"), "stderr: {stderr}");
    assert!(stderr.contains("let doy = mod(t, 365.0 'days) / (1.0 'days)"), "stderr: {stderr}");
    assert!(!stderr.contains("unknown expression"), "a declared let is not unknown: {stderr}");
}
