//! A replay of a fit's parameters runs on the simulator they were estimated
//! on: the fit's backend, its integration step, and (ODE) its integrator.
//!
//! Before this, `simulate --draws posterior --fit <fit>` ran on the
//! command's own defaults: a chain-binomial fit at `[config] dt = 0.5` was
//! replayed at dt 1, and an ODE fit on chain_binomial, both at exit 0 with no
//! notice. The store path names the simulator a run used
//! (`<backend>-dt<dt>-…`), so it is what these tests read.
//!
//! The reader is `fit::fitted_model::stored_fit_simulator` (and
//! `fit_config_simulator` for a fit config); the flags-over-fit rule is
//! `replay_simulator`, unit-tested beside it.

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
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const PROBLEM: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
"#;

/// A chain-binomial PMMH fit integrating at dt 0.5 (the model declares none).
const CB_METHOD: &str = r#"[config]
dt = 0.5
[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 30
iterations = 40
burn_in = 10
"#;

/// An ODE MH fit.
const ODE_METHOD: &str = r#"[method]
algorithm = "mh"
backend = "ode"
chains = 1
iterations = 60
burn_in = 20
starts = "single"
"#;

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

struct Fixture {
    dir: PathBuf,
    segment: PathBuf,
}

fn fixture(tag: &str, method: &str) -> Fixture {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let dir = std::env::temp_dir().join(format!("camdl_replay_sim_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), format!("{PROBLEM}{method}")).unwrap();
    std::fs::write(dir.join("d.tsv"), "beta\tgamma\n0.3\t0.1\n0.5\t0.2\n").unwrap();
    let out = run(&dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let segment = std::fs::read_dir(dir.join("results").join("fits"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("fit segment dir");
    Fixture { dir, segment }
}

/// `simulate model.camdl <rest>`: success, stderr, and the `<backend>-dt<dt>`
/// component of the store path the run landed at.
fn simulate(f: &Fixture, rest: &[&str]) -> (bool, String, Option<String>) {
    let mut args = vec!["simulate", "model.camdl"];
    args.extend_from_slice(rest);
    args.extend_from_slice(&["--seed", "3", "--output-dir", "out"]);
    let out = run(&f.dir, &args);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let config = stderr
        .split(|c: char| c.is_whitespace() || c == '/')
        .find(|s| s.starts_with("chain_binomial-dt") || s.starts_with("ode-dt") || s.starts_with("gillespie-dt"))
        .map(|s| s.rsplit_once('-').map(|(head, _hash)| head.to_string()).unwrap_or_default());
    (out.status.success(), stderr, config)
}

fn meta(f: &Fixture) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(f.segment.join("fit.meta.json")).unwrap()).unwrap()
}

#[test]
fn a_chain_binomial_fit_replays_at_its_own_dt_and_records_it() {
    let f = fixture("cb", CB_METHOD);
    let seg = f.segment.to_string_lossy().into_owned();

    // The fit states the step it ran at.
    let m = meta(&f);
    assert_eq!(m["dt"].as_f64(), Some(0.5), "fit.meta.json must record dt: {m}");
    assert_eq!(m["integrator"]["method"].as_str(), Some("rk4"), "{m}");

    // Every replay of its parameters runs at it.
    for rest in [
        &["--draws", "posterior", "--fit", seg.as_str(), "-n", "2"][..],
        &["--draws", "prior", "--fit", "fit.toml", "-n", "2"][..],
        &["--draws", "d.tsv", "--fit", seg.as_str()][..],
        &["--draws", "d.tsv", "--fit", "fit.toml"][..],
    ] {
        let (ok, err, config) = simulate(&f, rest);
        assert!(ok, "{rest:?} failed:\n{err}");
        assert_eq!(config.as_deref(), Some("chain_binomial-dt0.5"), "{rest:?} ran at the wrong step:\n{err}");
        assert!(!err.contains("was fitted on"), "no flag, no notice:\n{err}");
    }

    // A step from the command line still runs, saying both.
    let (ok, err, config) = simulate(&f, &["--draws", "posterior", "--fit", &seg, "-n", "2", "--dt", "1"]);
    assert!(ok, "{err}");
    assert_eq!(config.as_deref(), Some("chain_binomial-dt1"), "{err}");
    assert!(
        err.contains("fitted on chain_binomial at dt = 0.5")
            && err.contains("runs on chain_binomial at dt = 1"),
        "the notice must name both simulators:\n{err}"
    );

    // `compare`'s derived prequential re-filters the data at the fit's step.
    // The child `pfilter` names its step on stderr, which `compare` relays
    // only when the child fails, so it is made to: one particle has an
    // effective sample size of 1 and collapses at the first observation.
    let out = run(&f.dir, &["compare", &seg, &seg, "--draws", "1", "--replicates", "1", "--particles", "1"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("1 particles, dt=0.5,"), "compare must filter at the fit's dt:\n{err}");

    // A recorded step wins over the archived config, and an older fit with no
    // record recovers it from the archived config.
    let path = f.segment.join("fit.meta.json");
    let mut m = meta(&f);
    m.as_object_mut().unwrap().remove("dt");
    std::fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let (ok, err, config) = simulate(&f, &["--draws", "posterior", "--fit", &seg, "-n", "2"]);
    assert!(ok, "{err}");
    assert_eq!(config.as_deref(), Some("chain_binomial-dt0.5"), "a fit without a recorded dt:\n{err}");
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn an_ode_fit_replays_on_ode_and_a_differing_backend_runs_with_a_notice() {
    let f = fixture("ode", ODE_METHOD);
    let seg = f.segment.to_string_lossy().into_owned();

    let (ok, err, config) = simulate(&f, &["--draws", "posterior", "--fit", &seg, "-n", "2"]);
    assert!(ok, "{err}");
    assert_eq!(config.as_deref(), Some("ode-dt1"), "an ODE fit must replay on ODE:\n{err}");

    let (ok, err, config) =
        simulate(&f, &["--draws", "posterior", "--fit", &seg, "-n", "2", "--backend", "chain_binomial"]);
    assert!(ok, "{err}");
    assert_eq!(config.as_deref(), Some("chain_binomial-dt1"), "{err}");
    assert_eq!(err.matches("was fitted on").count(), 1, "exactly one notice:\n{err}");
    assert!(
        err.contains("fitted on ode (rk4) at dt = 1") && err.contains("runs on chain_binomial at dt = 1"),
        "{err}"
    );

    // The integrator belongs to ODE: carrying one onto chain_binomial is refused.
    let (ok, err, _) = simulate(
        &f,
        &["--draws", "posterior", "--fit", &seg, "-n", "2", "--backend", "chain_binomial", "--integrator", "rk45"],
    );
    assert!(!ok && err.contains("--integrator selects the ODE integrator"), "{err}");

    // On ODE it is a differing integrator: runs, with the notice.
    let (ok, err, config) = simulate(&f, &["--draws", "posterior", "--fit", &seg, "-n", "2", "--integrator", "rk45"]);
    assert!(ok, "{err}");
    assert_eq!(config.as_deref(), Some("ode-dt1"), "{err}");
    assert!(err.contains("runs on ode (rk45) at dt = 1"), "{err}");
    let _ = std::fs::remove_dir_all(&f.dir);
}
