//! `simulate --draws <file>|prior --fit fit.toml` takes every parameter the
//! draws lack from the config's `[fixed]` block. A block written as
//! `from_scenario = "held"` names a model scenario whose `set` values are the
//! fixed values; it needs the model to resolve. Both paths resolved `[fixed]`
//! without it, so the scenario's values were dropped and the run was refused
//! with "parameter 'N0' has no value … nothing supplies it" — for a parameter
//! the config does supply.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate  in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days  to = 20 'days }
scenarios {
  held { set = { N0 = 5000  I0 = 7 } }
}
"#;

const FIT_TOML: &str = r#"[model]
camdl = "model.camdl"
[estimate]
beta  = { bounds = [0.05, 1.0] }
gamma = { bounds = [0.01, 0.5] }
[fixed]
from_scenario = "held"
"#;

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The `S` value of the first trajectory row: `N0 - I0` under the scenario.
fn first_s(traj: &str) -> f64 {
    let mut lines = traj.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = header.iter().position(|h| *h == "S").expect("S column");
    let row: Vec<&str> = lines.next().expect("first row").split('\t').collect();
    row[col].parse().expect("numeric S")
}

#[test]
fn fixed_from_scenario_supplies_the_parameters_the_draws_lack() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let dir = std::env::temp_dir().join(format!("camdl_fixed_from_scenario_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();
    std::fs::write(dir.join("d.tsv"), "beta\tgamma\n0.3\t0.1\n").unwrap();

    for (draws, out) in [
        (&["--draws", "d.tsv"][..], "file.tsv"),
        (&["--draws", "prior", "-n", "1"][..], "prior.tsv"),
    ] {
        let mut args = vec!["simulate", "model.camdl"];
        args.extend_from_slice(draws);
        args.extend_from_slice(&["--fit", "fit.toml", "--seed", "1", "--output", out]);
        let o = run(&dir, &args);
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "{draws:?}: [fixed] from_scenario must supply N0 and I0:\n{err}");
        let traj = std::fs::read_to_string(dir.join(out)).unwrap();
        assert_eq!(first_s(&traj), 5000.0 - 7.0, "{draws:?}: S(0) = N0 - I0 from scenario `held`");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
