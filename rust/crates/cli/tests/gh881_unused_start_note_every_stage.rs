//! gh#881: a declared `[estimate].start` that the stage's `init` mode discards
//! is announced for every stage kind, not just IF2.
//!
//! `init::ignores_base_point` had one production call site, inside the
//! `Stage::IF2 { … }` arm of the `fit run` dispatch. PMMH, PGAS, MH, NUTS and
//! the NLopt stages take their own arm and never reached it, so they threw the
//! declared start away in silence. In gh#876 the modeller declared
//! `start = 0.0001` — the true value — ran a `pmmh` stage under the default
//! `starts = "uniform_unconstrained"`, fit from starts 33-41% away from it, and
//! had every chain refused; the note would have said why.
//!
//! The defect is at the call site, not inside `ignores_base_point`, which was
//! already right about every mode (`fit::init::tests`). Only running the
//! dispatch can tell whether the check is reached, so this is end to end.
//!
//! Deliberately cheap: 2 chains, a few hundred particles at most, a handful of
//! iterations, 5 observations — under a second for all three. The note is
//! printed before the stage runs, so nothing here depends on the fit
//! converging or on it being a good fit.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The half of the note that identifies it, ANSI-free and independent of which
/// init mode drew the starts.
const NOTE: &str = "draws every chain's start, so `[estimate].start` is unused here for";

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn golden_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/seir_observations.ir.json")
}

struct Run {
    ok: bool,
    stderr: String,
    stdout: String,
}

/// One single-stage fit over the SEIR golden IR. `stage_body` is spliced in
/// under `[method]`, so each caller varies exactly the algorithm and the
/// `init` mode and nothing else. `beta` declares both a `start` and a prior
/// (the Bayesian stages require one).
fn run_fit(stage_body: &str) -> Run {
    run_fit_with(
        "[estimate.beta]\nbounds = [0.01, 0.5]\nstart  = 0.123\n\
         prior  = { log_normal = { mu = -2.0, sigma = 0.5 } }\n",
        stage_body,
    )
}

/// `run_fit` with the `[estimate]` block supplied too, for the tests that
/// vary whether `beta` declares a prior.
fn run_fit_with(estimate_block: &str, stage_body: &str) -> Run {
    let bin = binary();
    assert!(bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display());
    let camdlc = camdlc_bin();
    assert!(camdlc.exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`", camdlc.display());

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let data = dir.join("obs.tsv");
    std::fs::write(&data,
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

    let fit_toml = dir.join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

{estimate_block}
[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[method]
{stage_body}

[config]
dt = 1.0
"#, ir = golden_ir().display(), data = data.display(), estimate_block = estimate_block,
     stage_body = stage_body)).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", "1"])
        .output()
        .expect("spawn fit run");
    Run {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
    }
}

fn note_count(stderr: &str) -> usize {
    stderr.matches(NOTE).count()
}

/// The gh#881 regression proper. A `pmmh` stage under the default `init`
/// discards `[estimate].start`, and must say so — it said nothing.
#[test]
fn a_pmmh_stage_says_that_it_discarded_the_declared_start() {
    let run = run_fit(
        "algorithm = \"pmmh\"\n\
         backend = \"chain_binomial\"\n\
         chains = 2\n\
         particles = 200\n\
         iterations = 6\n\
         burn_in = 2\n");
    assert!(run.ok, "fit run failed:\nstdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert_eq!(note_count(&run.stderr), 1,
        "a pmmh stage whose `init` draws every start must say that \
         `[estimate].start` was unused, exactly once.\nstderr:\n{}", run.stderr);
    assert!(run.stderr.contains("`starts = \"from_prior\"` draws"),
        "and must name the rule that discarded it — the default here, since beta \
         declares a prior.\nstderr:\n{}", run.stderr);
    assert!(run.stderr.contains("beta"),
        "and the parameter whose start was discarded.\nstderr:\n{}", run.stderr);
}

/// The path the check used to live on must keep printing it — and print it
/// once, not twice, now that the check is hoisted above the dispatch.
#[test]
fn the_if2_path_still_says_it_exactly_once() {
    let run = run_fit(
        "algorithm = \"if2\"\n\
         backend = \"chain_binomial\"\n\
         chains = 2\n\
         particles = 20\n\
         iterations = 1\n\
         cooling = 0.5\n");
    assert!(run.ok, "fit run failed:\nstdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert_eq!(note_count(&run.stderr), 1,
        "the IF2 note must survive the hoist and must not double up.\n\
         stderr:\n{}", run.stderr);
}

/// The negative control: `starts = "single"` is the mode whose whole contract is
/// the declared start, so there is nothing to announce. Without this the test
/// above would pass on a note that fires unconditionally.
#[test]
fn init_single_keeps_the_declared_start_and_says_nothing() {
    let run = run_fit(
        "algorithm = \"pmmh\"\n\
         backend = \"chain_binomial\"\n\
         chains = 2\n\
         particles = 200\n\
         iterations = 6\n\
         burn_in = 2\n\
         starts = \"single\"\n");
    assert!(run.ok, "fit run failed:\nstdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert_eq!(note_count(&run.stderr), 0,
        "`starts = \"single\"` uses the declared start, so the note must be \
         silent.\nstderr:\n{}", run.stderr);
}

/// The default `starts` rule (proposal §8, item 20) is said at startup with
/// its reason: `from_prior` when every estimated parameter has a prior the
/// chains can be drawn from, `uniform_unconstrained` naming the parameters
/// without one otherwise. The two fixtures differ only in `beta`'s prior.
#[test]
fn the_startup_block_says_which_default_starts_rule_ran_and_why() {
    let if2 = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
               particles = 20\niterations = 1\ncooling = 0.5\n";
    let with_prior = run_fit(if2);
    assert!(with_prior.ok, "fit run failed:\nstderr:\n{}", with_prior.stderr);
    assert!(with_prior.stderr.contains(
            "starts:   from_prior (one independent draw per chain) — default: every \
             estimated parameter declares a prior"),
        "with a prior on every parameter the default is from_prior, and the run says so:\n{}",
        with_prior.stderr);

    let without = run_fit_with("[estimate.beta]\nbounds = [0.01, 0.5]\nstart = 0.123\n", if2);
    assert!(without.ok, "fit run failed:\nstderr:\n{}", without.stderr);
    assert!(without.stderr.contains(
            "starts:   uniform_unconstrained (one independent draw per chain) — default: \
             no sampleable prior on beta"),
        "with no prior on beta the default is uniform_unconstrained, naming beta:\n{}",
        without.stderr);

    // A spelled rule is reported as spelled, not as a default.
    let spelled = run_fit(&format!("{if2}starts = \"lhs\"\n"));
    assert!(spelled.ok, "fit run failed:\nstderr:\n{}", spelled.stderr);
    assert!(spelled.stderr.contains("starts:   lhs (one independent draw per chain)")
            && !spelled.stderr.contains("— default:"),
        "a declared rule is reported without a default reason:\n{}", spelled.stderr);
}

/// A file with no `[method]` is a complete problem for the non-fit readers
/// and is refused by `fit run` by name.
#[test]
fn fit_run_refuses_a_problem_only_file_and_simulate_loads_it() {
    let bin = binary();
    let camdlc = camdlc_bin();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = dir.join("obs.tsv");
    std::fs::write(&data, "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n").unwrap();
    let problem = dir.join("problem.toml");
    std::fs::write(&problem, format!(r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

[estimate.beta]
bounds = [0.01, 0.5]
prior  = {{ log_normal = {{ mu = -2.0, sigma = 0.5 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[config]
dt = 1.0
"#, ir = golden_ir().display(), data = data.display())).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["fit", "run", &problem.to_string_lossy(), "--seed", "1"])
        .output()
        .expect("spawn fit run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "fit run must refuse a file with no [method]:\n{stderr}");
    assert!(stderr.contains("declares no `[method]` table, so there is nothing to run")
            && stderr.contains("[method]\n  algorithm ="),
        "the refusal must name the table and show its shape:\n{stderr}");

    // The same file is a complete problem for `simulate --draws prior --fit`.
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["simulate", &golden_ir().to_string_lossy(),
               "--draws", "prior", "--fit", &problem.to_string_lossy(),
               "-n", "2", "--seed", "1", "--obs-dir", &dir.join("ppc").to_string_lossy(),
               "--output-dir", &dir.join("sims").to_string_lossy()])
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "simulate --draws prior --fit must load a [method]-less problem:\n{stderr}");
}
