//! Auto `rw_sd` is a fraction of the search box. A box with an infinite end
//! gives it no scale: an unbounded `real` parameter got 166666.67 (its
//! `[0, ∞)` fall-back box, `Transform::None`, read as ±1e6) and an unbounded
//! `rate` got ~1e-299 (`Transform::Log` from a lower bound floored at
//! 1e-300). Either makes the IF2 random walk meaningless, and PMMH's and
//! MH's first proposals are five times it. Such a fit is refused, naming the
//! parameter and the two remedies; an explicit `rw_sd` runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

/// SIR with two parameters the dynamics ignore: `delta : real` and
/// `eta : rate`, neither with bounds.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate        in [0.01, 0.5]
  N0    : count
  I0    : count
  delta : real
  eta   : rate
}
let N = S + I + R
transitions {
  infection : S --> I  @ (beta + 0.0 * delta + 0.0 * eta) * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = 0.6 * projected, r = 10.0)
  }
}
simulate { from = 0 'days  to = 56 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// Fit `free` (with `extra` in its `[estimate]` entry) under `method`; the
/// other unbounded parameter is fixed.
fn fit(free: &str, extra: &str, method: &str) -> Output {
    run(free, extra, method, &["fit", "run", "fit.toml", "--seed", "1"])
}

fn run(free: &str, extra: &str, method: &str, args: &[&str]) -> Output {
    let held = if free == "delta" { "eta = 0.1" } else { "delta = 0.0" };
    let toml = format!(r#"[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta = {{ bounds = [0.05, 1.0], start = 0.25 }}
{free} = {{ start = 0.5{extra} }}
[fixed]
gamma = 0.15
N0 = 10000
I0 = 10
{held}
[method]
{method}
"#);
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), toml).unwrap();
    Command::new(binary())
        .args(args)
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("res"))
        .current_dir(tmp.path())
        .output()
        .expect("spawn")
}

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
                   particles = 50\niterations = 2\ncooling = 0.7";
const PMMH: &str = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 1\n\
                    particles = 50\niterations = 20\nburn_in = 5\nstarts = \"single\"";

fn refused(out: &Output, name: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "expected a refusal for {name}:\n{err}");
    assert!(err.contains(&format!("`{name}`")) && err.contains("rw_sd") && err.contains("bounds"),
        "the refusal names the parameter and both remedies:\n{err}");
}

#[test]
fn an_unbounded_real_parameter_has_no_auto_rw_sd() {
    refused(&fit("delta", "", IF2), "delta");
}

#[test]
fn an_unbounded_rate_parameter_has_no_auto_rw_sd() {
    refused(&fit("eta", "", IF2), "eta");
}

#[test]
fn pmmh_needs_the_same_scale_for_its_first_proposal() {
    refused(&fit("delta", ", prior = { normal = { mean = 0.0, sd = 1.0 } }", PMMH), "delta");
}

#[test]
fn an_explicit_rw_sd_or_finite_bounds_run() {
    for (free, extra) in [("delta", ", rw_sd = 0.1"), ("eta", ", bounds = [0.001, 1.0]")] {
        let out = fit(free, extra, IF2);
        assert!(out.status.success(), "{free}{extra}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn a_profile_cell_needs_the_same_scale() {
    let out = run("delta", "", IF2, &["profile", "model.camdl", "--fit", "fit.toml",
        "--sweep", "beta=0.3", "--rw-sd", "auto", "--n-starts", "1", "--particles", "20",
        "--iterations", "2"]);
    refused(&out, "delta");
}
