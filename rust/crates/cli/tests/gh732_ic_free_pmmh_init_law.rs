//! gh#732 — `ic_free` under plain PMMH is decided by the MODEL, end to end.
//!
//! `ic_free = true` conditions the initial state on y₁ by weighting and
//! resampling at the first observation. That only means anything if the
//! particles DIFFER in x₀; with one shared initial state the first reweight
//! scores every particle identically and the run silently drops y₁ instead of
//! conditioning on it.
//!
//! The bootstrap particle filter now draws x₀ per particle, so the property
//! holds exactly when the model's `init { }` declares a law. Two configs that
//! differ in nothing else pin both sides:
//!
//! * `init { I ~ poisson(rate = I0); S = N0 - I }` — the fit RUNS.
//! * `init { I = I0;                 S = N0 - I }` — the fit is REFUSED, at
//!   config load, naming the model as the reason and the law as the fix.
//!
//! This has to be end to end. Three separate checks decide the cell —
//! `methods::validate_ic_free` (unit-tested per cell), the per-particle-spread
//! precondition in `FitRunConfig::build`, and the filter itself — and a
//! disagreement between them shows up as an unsatisfiable config or a run that
//! should have been refused, neither of which any one unit test can see.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

/// A closed SIR with weekly reported cases. `init_i` is spliced in as the
/// `init { }` entry for `I`, and is the ONLY difference between the two
/// variants under test.
fn model_src(init_i: &str) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count        in [100, 100000]
  I0    : count        in [1, 1000]
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}}

let N = S + I + R

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

init {{
  {init_i}
  S = N0 - I
}}

observations {{
  weekly_cases {{
    columns       {{ time : time, weekly_cases : count }}
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }}
}}

simulate {{
  from = 0 'days
  to   = 56 'days
}}

scenarios {{
  baseline {{
    set = {{
      beta  = 0.4
      gamma = 0.15
      N0    = 10000
      I0    = 10
      rho   = 0.6
      k     = 10.0
    }}
  }}
}}
"#
    )
}

/// A minimal `ic_free = true` PMMH fit over `<variant>.camdl`, scoring
/// `weekly_cases.tsv`. Deliberately tiny — the question is whether the cell is
/// admitted and runs, not whether it converges. All paths are relative to the
/// fit toml (which is what camdl asks for; absolute ones draw a portability
/// warning that would be noise here).
fn fit_src(variant: &str) -> String {
    format!(
        r#"ic_free = true
output_dir = "results-{variant}"

[model]
camdl = "{variant}.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = {{ bounds = [0.05, 1.0], start = 0.4 }}

[fixed]
gamma = 0.15
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 60
iterations = 12
burn_in = 2
init = "single"
"#
    )
}

fn camdl(bin: &Path) -> Command {
    let mut c = Command::new(bin);
    c.env("CAMDL_SKIP_VERSION_CHECK", "1");
    c.env("CAMDLC", camdlc_bin());
    c
}

struct Case {
    dir: tempfile::TempDir,
    bin: PathBuf,
}

impl Case {
    /// Write both model variants, simulate a weekly case series off the
    /// law-bearing one, and return the shared scaffolding.
    fn new() -> Case {
        let bin = binary();
        assert!(
            bin.exists(),
            "release camdl binary missing: {} — run `make build-rust` or `make test`",
            bin.display()
        );
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("law.camdl"), model_src("I ~ poisson(rate = I0)"))
            .expect("write law.camdl");
        std::fs::write(dir.path().join("det.camdl"), model_src("I = I0"))
            .expect("write det.camdl");

        let out = camdl(&bin)
            .args([
                "simulate", "law.camdl",
                "--scenario", "baseline",
                "--backend", "chain_binomial",
                "--dt", "1",
                "--seed", "7",
                "--obs-only", "weekly_cases.tsv",
            ])
            .current_dir(dir.path())
            .output()
            .expect("spawn simulate");
        assert!(
            out.status.success(),
            "synthetic data generation failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        Case { dir, bin }
    }

    fn run_fit(&self, variant: &str) -> std::process::Output {
        let fit = format!("{variant}.toml");
        std::fs::write(self.dir.path().join(&fit), fit_src(variant))
            .expect("write fit toml");
        camdl(&self.bin)
            .args(["fit", "run", &fit, "--seed", "3", "--no-progress"])
            .current_dir(self.dir.path())
            .output()
            .expect("spawn fit run")
    }
}

/// The cell this change OPENS: a declared `init { }` law gives the bootstrap
/// swarm spread at t=0, so `ic_free` + plain `pmmh` is admitted and runs to
/// completion.
#[test]
fn ic_free_pmmh_runs_when_the_model_draws_its_initial_state() {
    let case = Case::new();
    let out = case.run_fit("law");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "ic_free + pmmh on a model whose init {{ }} declares a law must run \
         (gh#732). stderr:\n{stderr}"
    );
    // Non-vacuity: the run really conditioned rather than quietly turning
    // ic_free off, and the startup banner names the SOURCE of the t=0 spread
    // it is relying on. That line used to report only `perturb_only_at_t0`
    // parameters, so a fit whose spread came from the model read
    // `spread from perturb_only_at_t0 params: []` — the opposite of the truth.
    assert!(
        stderr.contains("ic-free inference"),
        "the run should report that it conditioned on y₁; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("initial state spread from: `init { }` laws on [I]"),
        "the banner must attribute the spread to the model's init law; stderr:\n{stderr}"
    );
}

/// The cell that stays SHUT, and the reason it is a model question: the same
/// fit against a deterministic `init { }` is refused at config load — before
/// any particle-filter time is spent — with a message that names the model as
/// the cause and the law as the fix.
#[test]
fn ic_free_pmmh_is_refused_when_the_model_computes_its_initial_state() {
    let case = Case::new();
    let out = case.run_fit("det");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "ic_free + pmmh on a deterministic init {{ }} must be refused (gh#732). \
         stderr:\n{stderr}"
    );
    assert!(stderr.contains("ic_free"), "must name ic_free:\n{stderr}");
    assert!(stderr.contains("gh#732"), "must cite the issue:\n{stderr}");
    assert!(
        stderr.contains("poisson(rate = I0)"),
        "must show the init-law form that fixes it:\n{stderr}"
    );
    // It must not have run a filter first.
    assert!(
        !stderr.contains("MAP loglik"),
        "the refusal must land at config load, not after the fit:\n{stderr}"
    );
}
