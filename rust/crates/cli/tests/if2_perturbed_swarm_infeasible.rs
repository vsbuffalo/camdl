//! IF2's "no finite log-likelihood" backstop runs on each chain's *final*
//! estimate, after the perturbations, not on the start. When the start is
//! feasible and the perturbed swarm is not, the refusal must say so — the
//! perturbation is too large — and name each parameter's step, instead of
//! blaming the start ("Initial log-likelihood is -inf at starting
//! parameters", which a `pfilter` at the start contradicts).
//!
//! The model makes infeasibility a hard −∞: a binomial count observed out of
//! the prevalence, so a trajectory with fewer infecteds than an observed
//! count scores −∞. Infection is at rate a²·S·I/N, symmetric in `a`, and the
//! start a = 0.9 is the truth. A step of about 0.7 in `a`'s transformed
//! coordinate spreads the swarm over both signs, and its mean lands near
//! a = 0, where
//! the epidemic dies: every chain's final estimate scores −∞ (measured at
//! seeds 1, 2 and 3), while the start is finite (log-likelihood −101.9).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust`", p.display());
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  a     : real  in [-3.0, 3.0]
  gamma : rate  in [0.01, 1.0]
}
transitions {
  infection : S --> I @ a * a * 1 'per_day * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ binomial(n = projected, p = 0.5)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 30 'days }
"#;

fn fit_toml(rw_sd: f64) -> String {
    format!(r#"output_dir = "out"
[model]
camdl = "m.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
a = {{ start = 0.9, rw_sd = {rw_sd} }}
[fixed]
gamma = 0.3
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 10
cooling = 0.9
starts = "single"
"#)
}

/// The step that spreads the swarm across both signs of `a`: a natural-scale
/// rw_sd of 1.0, converted once at the midpoint of `a`'s box [−3, 3], where
/// dθ/dz = 6/4, to 1.0 / 1.5 = 0.667 in `a`'s transformed coordinate.
const LARGE_RW_SD: f64 = 1.0;

#[test]
fn a_feasible_start_with_too_large_a_step_names_the_perturbed_swarm() {
    let Some(camdlc) = camdlc_bin() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    let out = Command::new(&camdlc).arg(dir.join("m.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("truth.toml"), "a = 0.9\ngamma = 0.3\n").unwrap();
    let sim = Command::new(camdl_bin())
        .args(["simulate", "m.ir.json", "--params", "truth.toml", "--backend", "chain_binomial",
               "--seed", "3", "--obs-only", "cases.tsv"])
        .current_dir(dir).env("CAMDL_SKIP_VERSION_CHECK", "1").output().unwrap();
    assert!(sim.status.success(), "simulate: {}", String::from_utf8_lossy(&sim.stderr));

    let run = |rw_sd: f64| {
        std::fs::write(dir.join("fit.toml"), fit_toml(rw_sd)).unwrap();
        Command::new(camdl_bin())
            .args(["fit", "run", "fit.toml", "--seed", "1", "--force", "--progress", "none"])
            .current_dir(dir).env("CAMDL_SKIP_VERSION_CHECK", "1").output().unwrap()
    };

    // Control: a small step from the same start fits.
    let ok = run(0.05);
    assert!(ok.status.success(), "a small step must fit: {}", String::from_utf8_lossy(&ok.stderr));

    let bad = run(LARGE_RW_SD);
    let err = String::from_utf8_lossy(&bad.stderr);
    assert!(!bad.status.success(), "every final estimate is -inf, so the fit refuses:\n{err}");
    assert!(err.contains("the start is feasible and the perturbed swarm is not"),
        "the refusal must say the perturbed swarm, not the start, is infeasible:\n{err}");
    assert!(err.contains("a=0.667"), "the refusal must name the step in z:\n{err}");
    assert!(!err.contains("Initial log-likelihood is -inf at starting parameters"),
        "the start is feasible; it must not be blamed:\n{err}");
}
