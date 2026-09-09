//! gh#881: a declared `[estimate].start` that the stage's `init` mode discards
//! is announced for every stage kind, not just IF2.
//!
//! `init::ignores_base_point` had one production call site, inside the
//! `Stage::IF2 { … }` arm of the `fit run` dispatch. PMMH, PGAS, MH, NUTS and
//! the NLopt stages take their own arm and never reached it, so they threw the
//! declared start away in silence. In gh#876 the modeller declared
//! `start = 0.0001` — the true value — ran a `pmmh` stage under the default
//! `init = "uniform_unconstrained"`, fit from starts 33-41% away from it, and
//! had every chain refused; the note would have said why.
//!
//! The defect is at the CALLSITE, not inside `ignores_base_point`, which was
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
/// under `[stages.only]`, so each caller varies exactly the algorithm and the
/// `init` mode and nothing else. `beta` declares both a `start` and a prior
/// (the Bayesian stages require one).
fn run_fit(stage_body: &str) -> Run {
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

[estimate.beta]
bounds = [0.01, 0.5]
start  = 0.123
prior  = {{ log_normal = {{ mu = -2.0, sigma = 0.5 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[stages.only]
{stage_body}

[config]
dt = 1.0
"#, ir = golden_ir().display(), data = data.display(), stage_body = stage_body)).unwrap();

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
    assert!(run.stderr.contains("uniform_unconstrained"),
        "and must name the init mode that discarded it.\nstderr:\n{}", run.stderr);
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

/// The negative control: `init = "single"` is the mode whose whole contract is
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
         init = \"single\"\n");
    assert!(run.ok, "fit run failed:\nstdout:\n{}\nstderr:\n{}", run.stdout, run.stderr);
    assert_eq!(note_count(&run.stderr), 0,
        "`init = \"single\"` uses the declared start, so the note must be \
         silent.\nstderr:\n{}", run.stderr);
}
