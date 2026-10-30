//! Auto `rw_sd` is a fraction of the search box. A box with an infinite end
//! gives it no scale: an unbounded `real` parameter got 166666.67 (an
//! infinite end of its identity box read as ±1e6) and an unbounded `rate`
//! got ~1e-299 (a lower bound floored at 1e-300). Either makes the IF2
//! random walk meaningless, and PMMH's and MH's first proposals are five
//! times it. Such a fit is refused, naming the parameter and the two
//! remedies; an explicit `rw_sd` runs.
//!
//! A positive kind boxed `[0, hi]` (`zeta : rate in [0, 2]` below) takes the
//! interval map, whose automatic step is a fraction of the width; under the
//! walled log map it was about 1e-148 and IF2 never moved it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

/// SIR with three parameters the dynamics ignore: `delta : real` and
/// `eta : rate`, neither with bounds, and `zeta : rate in [0, 2]`.
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
  zeta  : rate        in [0.0, 2.0]
}
let N = S + I + R
transitions {
  infection : S --> I  @ (beta + 0.0 * delta + 0.0 * eta + 0.0 * zeta) * S * I / N
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
/// other ignored parameters are fixed.
fn fit(free: &str, extra: &str, method: &str) -> Output {
    run(free, extra, method, &["fit", "run", "fit.toml", "--seed", "1"])
}

fn run(free: &str, extra: &str, method: &str, args: &[&str]) -> Output {
    let tmp = tempfile::tempdir().unwrap();
    run_in(tmp.path(), free, extra, method, args)
}

/// As [`run`], in `dir`, which the caller keeps to read the outputs.
fn run_in(dir: &Path, free: &str, extra: &str, method: &str, args: &[&str]) -> Output {
    let held = [("delta", "delta = 0.0"), ("eta", "eta = 0.1"), ("zeta", "zeta = 0.5")]
        .iter().filter(|(n, _)| *n != free).map(|(_, h)| *h).collect::<Vec<_>>().join("\n");
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
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), toml).unwrap();
    Command::new(binary())
        .args(args)
        .env("CAMDL_OUTPUT_DIR", dir.join("res"))
        .current_dir(dir)
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

/// A positive kind boxed `[0, 2]` has an automatic step: IF2, PMMH and MH
/// run, profile's IF2 cell runs, and IF2 moves the parameter.
#[test]
fn a_positive_kind_boxed_from_zero_has_an_auto_rw_sd_that_moves_it() {
    let mh = "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 1\n\
              iterations = 20\nburn_in = 5\nstarts = \"single\"";
    let prior = ", prior = { uniform = {} }";
    for (method, extra, tag) in [(IF2, "", "if2"), (PMMH, prior, "pmmh"), (mh, prior, "mh")] {
        let out = fit("zeta", extra, method);
        assert!(out.status.success(), "{tag}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let out = run("zeta", "", IF2, &["profile", "model.camdl", "--fit", "fit.toml",
        "--sweep", "beta=0.3", "--rw-sd", "auto", "--n-starts", "1", "--particles", "20",
        "--iterations", "2"]);
    assert!(out.status.success(), "profile: {}", String::from_utf8_lossy(&out.stderr));
    let tmp = tempfile::tempdir().unwrap();
    let out = run_in(tmp.path(), "zeta", "", IF2, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let vals = zeta_trace(tmp.path());
    assert!(vals.iter().any(|v| (v - vals[0]).abs() > 1e-6),
        "zeta never moved from {}: {vals:?}", vals[0]);
}

fn zeta_trace(dir: &Path) -> Vec<f64> {
    let traces = find(&dir.join("res"), "parameter_traces.tsv").expect("parameter_traces.tsv");
    let text = std::fs::read_to_string(traces).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let col = lines.next().unwrap().split('\t').position(|h| h == "zeta").expect("zeta column");
    lines.map(|l| l.split('\t').nth(col).unwrap().parse().unwrap()).collect()
}

/// With an explicit `rw_sd` the same box runs, and IF2 moves the parameter.
#[test]
fn an_explicit_rw_sd_on_a_box_from_zero_moves_it() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_in(tmp.path(), "zeta", ", rw_sd = 0.1", IF2, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let vals = zeta_trace(tmp.path());
    assert!(vals.iter().any(|v| (v - vals[0]).abs() > 1e-6),
        "zeta never moved from {}: {vals:?}", vals[0]);
}

fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find(&p, name) { return Some(f); }
        } else if p.file_name().is_some_and(|n| n == name) {
            return Some(p);
        }
    }
    None
}
