//! gh#653 — a completed fit is found by what its `fit.toml` MEANS, not by the
//! bytes it was written in.
//!
//! Reflowing a comment across a set of configs left every completed fit
//! unreachable: the run store held hours of sampling, and every verb handed the
//! config reported "no completed fit found". The parsed TOML was identical.
//!
//! The two halves of this test are equally load-bearing. A meaning-preserving
//! edit (comment reflow, table reorder, `10.0` written `10.00`) must still
//! resolve — and resolve to the SAME run, checked by comparing the θ̂ the fit
//! reports. A meaning-CHANGING edit (a different particle count, a different
//! data file) must NOT resolve: canonicalising too aggressively would hand back
//! the posterior of a different fit, which is worse than the bug being fixed.

use std::path::{Path, PathBuf};
use std::process::Command;

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
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The config as first written and run.
const ORIGINAL: &str = r#"# Scout fit for the closed SIR. Reporting rate is fixed at 0.5 because the
# surveillance review has not landed yet.
output_dir = "results"

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
rho = 0.5
k   = 10.0

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 15
cooling = 0.7
"#;

/// The same config, edited the way a config actually gets edited: the comment
/// rewrapped, the tables reordered, `10.0` written `10.00` and `0.5` as `0.50`,
/// the assignments realigned. Not one of these changes what camdl reads.
const REFLOWED: &str = r#"# Scout fit for the closed SIR.
# Reporting rate is fixed at 0.5 because the surveillance review has not
# landed yet.
output_dir = "results"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.50
k   = 10.00

[data.observations]
weekly_cases = "weekly_cases.tsv"

[model]
camdl = "model.camdl"

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 150
iterations = 15
cooling    = 0.70
"#;

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// `camdl fit summary a.toml --params-only` — succeeds only if the config
/// resolves to a completed fit, and its stdout is that fit's θ̂.
fn resolve_theta(bin: &Path, dir: &Path) -> Result<String, String> {
    let out = run(bin, dir, &["fit", "summary", "a.toml", "--params-only"]);
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn a_reflowed_config_still_finds_its_completed_fit() {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );

    let tmp = std::env::temp_dir().join(format!("camdl_gh653_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    // A second copy of the same data, for the changed-data-path control: the
    // file exists, so a failure to resolve is about identity, not a missing file.
    std::fs::write(tmp.join("weekly_cases_v2.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("a.toml"), ORIGINAL).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "a.toml", "--label", "a", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let theta = resolve_theta(&bin, &tmp).expect("the config it was run from resolves");
    assert!(
        theta.contains("beta"),
        "θ̂ carries the estimated parameters:\n{theta}"
    );

    // ── The fix: meaning-preserving edits keep the fit reachable ──
    std::fs::write(tmp.join("a.toml"), REFLOWED).unwrap();
    let after = resolve_theta(&bin, &tmp).unwrap_or_else(|e| {
        panic!("a reflowed comment must not orphan a completed fit (gh#653): {e}")
    });
    assert_eq!(
        after, theta,
        "the reflowed config resolves to the SAME run, so θ̂ is unchanged"
    );

    // ── The control: meaning-changing edits must NOT resolve ──
    // A different particle count is a different fit; resolving to the old run
    // would hand back a posterior nobody asked for.
    let more_particles = REFLOWED.replace("particles  = 150", "particles  = 151");
    assert_ne!(more_particles, REFLOWED);
    std::fs::write(tmp.join("a.toml"), &more_particles).unwrap();
    let err = resolve_theta(&bin, &tmp)
        .err()
        .expect("a changed particle count must not resolve to the old fit");
    assert!(
        err.contains("no completed fit found"),
        "the miss is reported plainly; got: {err}"
    );

    // A different data file is a different fit, even with identical contents.
    let other_data = REFLOWED.replace("weekly_cases.tsv", "weekly_cases_v2.tsv");
    assert_ne!(other_data, REFLOWED);
    std::fs::write(tmp.join("a.toml"), &other_data).unwrap();
    let err = resolve_theta(&bin, &tmp)
        .err()
        .expect("a changed data path must not resolve to the old fit");
    assert!(
        err.contains("no completed fit found"),
        "the miss is reported plainly; got: {err}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
