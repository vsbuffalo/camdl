//! gh#653 — a fit run's identity is what its `fit.toml` means, not the bytes
//! it was written in.
//!
//! Reflowing a comment across a set of configs must not orphan the completed
//! runs the store holds: the parsed TOML is identical, so `fit run` on the
//! edited config is a cache hit on the same run.
//!
//! The two halves of this test are equally load-bearing. A meaning-preserving
//! edit (comment reflow, table reorder, `10.0` written `10.00`) must reuse the
//! stored run. A meaning-changing edit (a different particle count, different
//! data) must compute a new run: canonicalising too aggressively would hand
//! back the posterior of a different fit, which is worse than the bug being
//! fixed. Data files are identified by their content, so the same lines at a
//! different path are the same problem, and changed lines are a new one.

mod fit_store_support;

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
    covers        = closing_at(time, 7 'days)
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
store = "results"

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

[method]
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
store = "results"

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

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 150
iterations = 15
cooling    = 0.70
"#;

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_STORE", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// `camdl fit summary <handle> --params-only`: the stored run's θ̂.
fn theta(bin: &Path, dir: &Path, handle: &str) -> String {
    let out = run(bin, dir, &["fit", "summary", handle, "--params-only"]);
    assert!(out.status.success(), "fit summary {handle}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `camdl fit run a.toml --seed 1`, which must succeed; its stderr.
fn fit_run(bin: &Path, dir: &Path, extra: &[&str]) -> String {
    let mut args = vec!["fit", "run", "a.toml", "--seed", "1"];
    args.extend_from_slice(extra);
    let out = run(bin, dir, &args);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    stderr
}

#[test]
fn a_reflowed_config_reuses_its_completed_run() {
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
    // A second copy of the same data at another path, and a copy with one
    // count changed, for the data controls.
    std::fs::write(tmp.join("weekly_cases_v2.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("weekly_cases_v3.tsv"), DATA.replace("\t1303\n", "\t1304\n")).unwrap();
    std::fs::write(tmp.join("a.toml"), ORIGINAL).unwrap();
    let store = tmp.join("results");

    fit_run(&bin, &tmp, &["--label", "a"]);
    let first = fit_store_support::the_run(&store);
    let theta_a = theta(&bin, &tmp, "@a");
    assert!(theta_a.contains("beta"), "θ̂ carries the estimated parameters:\n{theta_a}");

    // ── The fix: a meaning-preserving edit reuses the stored run ──
    std::fs::write(tmp.join("a.toml"), REFLOWED).unwrap();
    let stderr = fit_run(&bin, &tmp, &[]);
    assert!(
        stderr.contains("cache hit"),
        "a reflowed comment must not orphan a completed fit (gh#653):\n{stderr}"
    );
    assert_eq!(
        fit_store_support::runs(&store),
        vec![first.clone()],
        "the reflowed config names the same run"
    );
    // Named by its folder relative to the working directory, so the summary's
    // `# source:` line spells the path as `@a` resolves it.
    assert_eq!(
        theta(&bin, &tmp, &first.strip_prefix(&tmp).unwrap().to_string_lossy()),
        theta_a,
        "and that run's θ̂ is unchanged"
    );

    // ── The control: meaning-changing edits compute a new run ──
    // A different particle count is a different fit; reusing the old run
    // would hand back a posterior nobody asked for.
    let more_particles = REFLOWED.replace("particles  = 150", "particles  = 151");
    assert_ne!(more_particles, REFLOWED);
    std::fs::write(tmp.join("a.toml"), &more_particles).unwrap();
    let stderr = fit_run(&bin, &tmp, &[]);
    assert!(!stderr.contains("cache hit"), "a changed particle count must not reuse the old run:\n{stderr}");
    assert_eq!(fit_store_support::runs(&store).len(), 2, "a second run for the changed particle count");

    // The same lines at another path are the same problem and the same run.
    let same_lines = REFLOWED.replace("weekly_cases.tsv", "weekly_cases_v2.tsv");
    assert_ne!(same_lines, REFLOWED);
    std::fs::write(tmp.join("a.toml"), &same_lines).unwrap();
    let stderr = fit_run(&bin, &tmp, &[]);
    assert!(stderr.contains("cache hit"), "identical data at another path is the same fit:\n{stderr}");
    assert_eq!(fit_store_support::runs(&store).len(), 2);

    // Changed data is a different problem, and a new run.
    let other_lines = REFLOWED.replace("weekly_cases.tsv", "weekly_cases_v3.tsv");
    std::fs::write(tmp.join("a.toml"), &other_lines).unwrap();
    let stderr = fit_run(&bin, &tmp, &[]);
    assert!(!stderr.contains("cache hit"), "changed data must not reuse the old run:\n{stderr}");
    assert_eq!(fit_store_support::runs(&store).len(), 3, "a third run for the changed data");
    assert_eq!(fit_store_support::problems(&store).len(), 2, "on a second problem");

    // A configuration is not a handle for a stored run.
    let out = run(&bin, &tmp, &["fit", "summary", "a.toml", "--params-only"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && err.contains("a configuration never names a run"),
        "a fit file is refused as a run handle; got: {err}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// `ORIGINAL`'s fit, with a `[provenance]` note.
fn with_note(reason: &str) -> String {
    format!(r#"store = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.5
k   = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 15
cooling = 0.7

[provenance]
derived_from = "scout.toml"
reason = "{reason}"
"#)
}

/// gh#583 item F. `[provenance]` is a note to the reader — where a config came
/// from and why — and nothing reads it but the reader: no runner, no
/// likelihood, no stored artifact is a function of it. So it is not part of a
/// fit run's identity, and a reworded note is a cache hit on the same run.
#[test]
fn a_provenance_note_is_not_fit_identity() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    let tmp = std::env::temp_dir().join(format!("camdl_gh583f_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    std::fs::write(tmp.join("a.toml"), with_note("first scout")).unwrap();
    fit_run(&bin, &tmp, &[]);
    let store = tmp.join("results");
    let first = fit_store_support::the_run(&store);
    let theta_first = theta(&bin, &tmp, &first.to_string_lossy());

    std::fs::write(tmp.join("a.toml"), with_note("reworded: rho pinned pending review"))
        .unwrap();
    let stderr = fit_run(&bin, &tmp, &[]);
    let leaves = fit_store_support::runs(&store);
    assert_eq!(leaves, vec![first.clone()],
        "a reworded provenance note must be a cache hit, not a second fit: {leaves:?}\n{stderr}");
    assert!(stderr.contains("cache hit"), "the rerun reuses the stored run:\n{stderr}");
    assert_eq!(theta(&bin, &tmp, &first.to_string_lossy()), theta_first,
        "and it is the same fit");

    let _ = std::fs::remove_dir_all(&tmp);
}
