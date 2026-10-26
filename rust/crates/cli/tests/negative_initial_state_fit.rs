//! A negative starting count under `fit` and `pfilter`, end to end.
//!
//! Two models that differ only in how `I` starts, with `R = N0 - S - I`:
//!
//! * `I = I0` — the start is a function of the parameters alone. Where
//!   `I0 > N0 - S0` every particle starts with `R < 0`, the filter fails at
//!   every particle, and PMMH rejects that θ (log-likelihood −∞) and carries
//!   on. The fit completes and no retained draw has `I0` above the feasible
//!   limit.
//! * `I ~ poisson(rate = I0)` with `N0 - S0 = I0 = 5` — some particles draw
//!   `I > 5` and start with `R < 0`. That is a structural error: the fit and
//!   `pfilter` stop, and the message names the compartment, its value, the
//!   random initial law, and the fix.
//!
//! `R` is in no rate, so nothing but the negative-count checks can catch it.

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

fn model_src(init_i: &str) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count        in [1, 100000]
  S0    : count        in [1, 100000]
  I0    : count        in [1, 100000]
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}}

let N = S + I + R

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

init {{
  S = S0
  {init_i}
  R = N0 - S - I
}}

observations {{
  weekly_cases {{
    columns       {{ time : time, weekly_cases : count }}
    covers        = closing_at(time, 7 'days)
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
    set = {{ beta = 0.4  gamma = 0.15  N0 = 1000  S0 = 990  I0 = 5  rho = 0.6  k = 10.0 }}
  }}
}}
"#
    )
}

/// PMMH over `<variant>.camdl`. `estimate` is the `[estimate]` body.
fn fit_src(variant: &str, estimate: &str, fixed: &str) -> String {
    format!(
        r#"output_dir = "results-{variant}"

[model]
camdl = "{variant}.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
{estimate}

[fixed]
{fixed}

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 60
iterations = 200
burn_in = 20
starts = "single"
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
    /// Write both variants and simulate a weekly case series off the
    /// deterministic one at a feasible start (`R = 1000 - 990 - 5 = 5`).
    fn new() -> Case {
        let bin = binary();
        assert!(
            bin.exists(),
            "release camdl binary missing: {} — run `make build-rust` or `make test`",
            bin.display()
        );
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("det.camdl"), model_src("I = I0")).expect("write det");
        std::fs::write(dir.path().join("law.camdl"), model_src("I ~ poisson(rate = I0)"))
            .expect("write law");
        let out = camdl(&bin)
            .args([
                "simulate", "det.camdl", "--scenario", "baseline", "--backend", "chain_binomial",
                "--dt", "1", "--seed", "7", "--obs-only", "weekly_cases.tsv",
            ])
            .current_dir(dir.path())
            .output()
            .expect("spawn simulate");
        assert!(out.status.success(), "data generation failed:\n{}",
            String::from_utf8_lossy(&out.stderr));
        Case { dir, bin }
    }

    fn fit(&self, variant: &str, estimate: &str, fixed: &str) -> std::process::Output {
        let toml = format!("{variant}.toml");
        std::fs::write(self.dir.path().join(&toml), fit_src(variant, estimate, fixed))
            .expect("write fit toml");
        camdl(&self.bin)
            .args(["fit", "run", &toml, "--seed", "3", "--no-progress"])
            .current_dir(self.dir.path())
            .output()
            .expect("spawn fit run")
    }
}

/// Case (b): a random initial law that can make `R` negative stops `fit` and
/// `pfilter` with the structural message.
#[test]
fn a_drawn_negative_start_stops_fit_and_pfilter() {
    let case = Case::new();
    let fixed = "I0 = 5\ngamma = 0.15\nN0 = 1000\nS0 = 995\nrho = 0.6\nk = 10.0";
    let out = case.fit("law", "beta = { bounds = [0.05, 1.0], start = 0.4 }", fixed);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a drawn negative start must stop the fit:\n{stderr}");
    // Stopped as a structural error, not by every θ scoring −∞: no evaluation
    // may report the θ as ruled out.
    assert!(stderr.contains("structural error"), "fit must stop on a structural error:\n{stderr}");
    assert!(!stderr.contains("ruled out"), "a drawn negative start must not rule θ out:\n{stderr}");
    for needle in [
        "compartment 'R' starts at -",
        "cause: InitialStateDrawn",
        "depends on a random initial law",
        "which the model does not define",
        "Fix: constrain the init expression or the law so it cannot go negative",
    ] {
        assert!(stderr.contains(needle), "fit stderr lacks {needle:?}:\n{stderr}");
    }

    std::fs::write(
        case.dir.path().join("law_params.toml"),
        format!("beta = 0.4\n{fixed}\n"),
    )
    .expect("write params");
    let out = camdl(&case.bin)
        .args([
            "pfilter", "law.camdl", "--params", "law_params.toml",
            "--data", "weekly_cases=weekly_cases.tsv", "--particles", "60", "--seed", "1",
        ])
        .current_dir(case.dir.path())
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "pfilter must stop:\n{stderr}");
    assert!(stderr.contains("cause: InitialStateDrawn")
            && stderr.contains("depends on a random initial law"),
        "pfilter must print the structural message:\n{stderr}");
}
