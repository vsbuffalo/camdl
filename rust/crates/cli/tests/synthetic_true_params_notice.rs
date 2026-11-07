//! `[synthetic] true_params` is tier 5 of the run spec §1.3 chain for the
//! forward run that draws a synthetic dataset, so it replaces a value the
//! `[synthetic] scenario` sets. The notice for that replacement names the
//! source that did it. It used to say `--param/--fixed replaced beta = 0.8
//! (from scenario ...)`, naming flags `fit run` does not take and the user
//! never gave.

use std::path::{Path, PathBuf};
use std::process::Command;

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
  hi { set = { beta = 0.8 } }
}
"#;

const FIT: &str = r#"output_dir = "res"
[model]
camdl = "m.ir.json"
[synthetic]
true_params = "truth.toml"
sim_seeds = [1]
scenario = "hi"
[estimate]
beta = { bounds = [0.01, 1.0], start = 0.3 }
[fixed]
gamma = 0.1
N0 = 1000
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 3
cooling = 0.7
"#;

fn setup(dir: &Path) {
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    let out = Command::new(camdlc_bin()).arg(dir.join("m.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("truth.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 1000\n").unwrap();
    std::fs::write(dir.join("syn.toml"), FIT).unwrap();
}

/// The replacement is reported once, naming `[synthetic] true_params` and the
/// truth file, and no flag.
#[test]
fn true_params_replacing_the_synthetic_scenario_is_named_as_the_source() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    setup(dir);
    let out = Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(["fit", "run", "syn.toml", "--seed", "1"])
        .output()
        .expect("spawn camdl fit run");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fit run: {err}");
    let line = "info: [synthetic] true_params truth.toml replaced beta = 0.8 \
                (from scenario `hi`) with 0.3";
    assert_eq!(err.matches(line).count(), 1, "want {line:?} once; stderr:\n{err}");
    assert!(!err.contains("--param/--fixed"), "no flag was given; stderr:\n{err}");
}
