//! `fit run` refuses an estimated parameter the method cannot use as the
//! file asks, from where the parameter enters the compiled model: IF2's
//! perturbation timing against the initial-state property, and the
//! use-class policy (proposal `2026-10-29-method-scoped-settings.md` §4.6).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin(rel: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    assert!(p.exists(), "{} missing: run `make build`", p.display());
    p
}

fn camdl_bin() -> PathBuf { bin("../../target/release/camdl") }
fn camdlc_bin() -> PathBuf { bin("../../../ocaml/_build/default/bin/camdlc.exe") }

/// An SIR whose initial state reads `I0` and nothing after t = 0 does, with
/// a hyperparameter `mu_g` of `gamma`, a prior-only `tau` and a flat `nu`
/// that nothing reads.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  mu_g  : real        in [-5.0, 5.0]  ~ normal(mu = -2.0, sigma = 1.0)
  gamma : rate                        ~ log_normal(mu = mu_g, sigma = 0.1)
  I0    : count       in [1, 100]     ~ log_normal(mu = 2.0, sigma = 0.5)
  tau   : positive    in [0.1, 10.0]  ~ half_normal(sigma = 1.0)
  nu    : positive    in [0.1, 10.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 1000 - I0  I = I0 }
simulate { from = 0 'days  to = 6 'days }
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

struct Fixture {
    dir: tempfile::TempDir,
    ir: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.path().join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(dir.path().join("cases.tsv"), CASES_TSV).unwrap();
    Fixture { dir, ir }
}

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\n\
                   iterations = 1\ncooling = 0.5\n";
const PMMH: &str = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\n\
                    iterations = 6\nburn_in = 2\n";
const SBPLX: &str = "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\nmax_evals = 5\n";

/// Runs `fit run` on a file that estimates `estimate` (one `[estimate]` line
/// each), holds the model's other parameters, and runs `method` (the
/// `[method]` body, its subtables included).
fn fit(fx: &Fixture, tag: &str, estimate: &[&str], method: &str) -> Output {
    let all = ["beta", "mu_g", "gamma", "I0", "tau", "nu"];
    let held = [("beta", "0.3"), ("mu_g", "-2.0"), ("gamma", "0.15"), ("I0", "10"), ("tau", "1.0"), ("nu", "1.0")];
    let named = |line: &&str| line.split_whitespace().next().unwrap().to_string();
    let estimated: Vec<String> = estimate.iter().map(named).collect();
    let fixed: String = held.iter().filter(|(n, _)| !estimated.iter().any(|e| e == n))
        .map(|(n, v)| format!("{n} = {v}\n")).collect();
    assert!(estimated.iter().all(|e| all.contains(&e.as_str())));
    let file = fx.dir.path().join(format!("{tag}.toml"));
    std::fs::write(&file, format!(
        "[model]\ncamdl = \"{ir}\"\n[data.observations]\ncases = \"{data}\"\n[estimate]\n{est}\n[fixed]\n{fixed}\n[method]\n{method}",
        ir = fx.ir.display(), data = fx.dir.path().join("cases.tsv").display(), est = estimate.join("\n"),
    )).unwrap();
    let cwd = fx.dir.path().join(tag);
    std::fs::create_dir_all(&cwd).unwrap();
    Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .env("CAMDL_STORE", cwd.join("store"))
        .current_dir(&cwd)
        .args(["fit", "run", file.to_str().unwrap(), "--seed", "1"])
        .output()
        .unwrap()
}

fn stderr(o: &Output) -> String { String::from_utf8_lossy(&o.stderr).into_owned() }

fn refused(o: &Output, what: &str) -> String {
    let err = stderr(o);
    assert!(!o.status.success(), "{what} must be refused:\n{err}");
    err
}

fn ran(o: &Output, what: &str) -> String {
    let err = stderr(o);
    assert!(o.status.success(), "{what} must run:\n{err}");
    err
}

#[test]
fn if2_refuses_an_initial_state_parameter_perturbed_at_every_observation() {
    let fx = fixture();
    let err = refused(&fit(&fx, "i0", &["beta = { start = 0.3 }", "I0 = { start = 10 }"], IF2), "unmarked I0");
    assert!(err.contains("`I0` enters only the initial state (the initial condition of `S`)")
        && err.contains("I0 = { initial_state = true }"), "{err}");
    assert!(!err.contains("`beta`"), "beta enters the dynamics and is not refused:\n{err}");

    let marked = format!("{IF2}[method.perturbation]\nI0 = {{ initial_state = true }}\n");
    ran(&fit(&fx, "i0_marked", &["beta = { start = 0.3 }", "I0 = { start = 10 }"], &marked), "I0 marked initial_state");
}

#[test]
fn if2_refuses_initial_state_on_a_parameter_of_the_dynamics() {
    let fx = fixture();
    let method = format!("{IF2}[method.perturbation]\nbeta = {{ sd = 0.05, initial_state = true }}\n");
    let err = refused(&fit(&fx, "beta_t0", &["beta = { start = 0.3 }"], &method), "initial_state on beta");
    assert!(err.contains("`beta` has `initial_state = true`")
        && err.contains("the rate of transition `infection`"), "{err}");
}

#[test]
fn no_other_method_checks_the_initial_state_timing() {
    let fx = fixture();
    let err = ran(&fit(&fx, "pmmh_i0", &["beta = { start = 0.3 }", "I0 = { start = 10 }"], PMMH), "PMMH over I0");
    assert!(!err.contains("initial state"), "PMMH has no perturbation timing:\n{err}");
}

#[test]
fn a_hyperparameter_is_refused_where_no_prior_is_read_and_accepted_where_one_is() {
    let fx = fixture();
    let est = ["beta = { start = 0.3 }", "mu_g = { start = -2.0 }", "gamma = { start = 0.15 }"];
    let err = refused(&fit(&fx, "hier_if2", &est, IF2), "a hyperparameter under IF2");
    assert!(err.contains("`mu_g` is estimated") && err.contains("only in the prior of `gamma`")
        && err.contains("`if2` reads no prior"), "{err}");
    // `gamma` keeps its kind's box, the log_normal's whole support, which
    // gives a box-drawing start rule no region: the chains start at `start`.
    let pmmh = format!("{PMMH}starts = \"single\"\n[method.proposal]\ngamma = 0.5\n");
    let err = ran(&fit(&fx, "hier_pmmh", &est, &pmmh), "a hyperparameter under PMMH");
    assert!(!err.contains("`mu_g` is estimated"), "{err}");
}

#[test]
fn a_prior_only_parameter_is_refused_by_the_likelihood_methods_and_warned_of_by_the_samplers() {
    let fx = fixture();
    let est = ["beta = { start = 0.3 }", "tau = { start = 1.0 }"];
    for (tag, method, name) in [("tau_if2", IF2, "if2"), ("tau_sbplx", SBPLX, "nl-sbplx")] {
        let err = refused(&fit(&fx, tag, &est, method), "a prior-only parameter");
        assert!(err.contains("`tau` is estimated") && err.contains(&format!("`{name}` reads only the likelihood")),
            "{tag}: {err}");
    }
    let err = ran(&fit(&fx, "tau_pmmh", &est, PMMH), "a prior-only parameter under PMMH");
    assert!(err.contains("`tau`'s marginal posterior equals its prior"), "{err}");
}

#[test]
fn an_unused_parameter_is_refused_by_every_method() {
    let fx = fixture();
    let est = ["beta = { start = 0.3 }", "nu = { start = 1.0, prior = { flat = {} } }"];
    let err = refused(&fit(&fx, "nu_pmmh", &est, PMMH), "an unused parameter under PMMH");
    assert!(err.contains("`nu` is estimated") && err.contains("its own prior is flat"), "{err}");
    let err = refused(&fit(&fx, "nu_if2", &est, IF2), "an unused parameter under IF2");
    assert!(err.contains("`nu` is estimated"), "{err}");
}

#[test]
fn prior_only_refuse_refuses_a_prior_only_parameter_and_warn_runs_it() {
    let fx = fixture();
    let est = ["beta = { start = 0.3 }", "tau = { start = 1.0 }"];
    let err = refused(&fit(&fx, "tau_refuse", &est, &format!("{PMMH}prior_only = \"refuse\"\n")),
        "a prior-only parameter under prior_only = \"refuse\"");
    assert!(err.contains("`tau`'s marginal posterior equals its prior")
        && err.contains("prior_only = \"refuse\""), "{err}");
    let err = ran(&fit(&fx, "tau_warn", &est, &format!("{PMMH}prior_only = \"warn\"\n")),
        "a prior-only parameter under prior_only = \"warn\"");
    assert!(err.contains("`tau`'s marginal posterior equals its prior"), "{err}");
}

#[test]
fn prior_only_is_refused_under_a_method_that_reads_no_prior() {
    let fx = fixture();
    let method = format!("{IF2}prior_only = \"warn\"\n");
    let err = refused(&fit(&fx, "prior_only_if2", &["beta = { start = 0.3 }"], &method), "prior_only under if2");
    assert!(err.contains("`prior_only`") && err.contains("pmmh, mh and pgas") && err.contains("`if2`"), "{err}");
}
