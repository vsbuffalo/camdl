//! gh#715: one non-finite draw in one `quantities` entry must not destroy the
//! whole predictive artifact.
//!
//! The reported case: `growth = value_at(I, last_obs) / value_at(I, last_obs -
//! 2 'weeks)` divides by an empty compartment on 3 of 200 draws — an ordinary
//! member of that posterior, not a corrupt run — and `camdl fit predict` wrote
//! **nothing at all**: no `predictive/`, no `observed/`, no bands for any of
//! four observation streams, and none of the other 21 quantities. That cost
//! four of five arms of a model comparison their entire posterior predictive.
//!
//! Refusing to quantile a `NaN`/±∞ is right; taking the artifact down with it
//! is not. Same rule as the free-forward tail (proposal
//! `docs/dev/proposals/2026-09-08-workflow-first-fit-config.md` §3.5): the entry
//! that produced the non-finite value is recorded as a failure with the draw
//! index and the quantity name, every other quantity and every stream artifact
//! is written, and the exit status is 1.
//!
//! The fixture divides by `count_above(I / N, 1.0)`, which is exactly 0 on
//! every draw because the infected fraction of a closed population is never
//! above 1. That makes the non-finite value deterministic — no dependence on
//! which corner of the posterior a seed lands in.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
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

quantities {
  prevalence = I / N                     # series — must still be written
  peak       = max(I / N)                # scalar — must still be written
  never      = count_above(I / N, 1.0)   # exactly 0: a closed population's
                                         # infected fraction never exceeds 1
  ratio      = peak / never              # +inf on every draw
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const FIT_TOML: &str = r#"output_dir = "results"

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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#;

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn segment(results: &Path) -> PathBuf {
    let fits = results.join("fits");
    for e in std::fs::read_dir(&fits).expect("results/fits exists").flatten() {
        let p = e.path();
        if p.join("fit.meta.json").is_file() {
            return p;
        }
    }
    panic!("no fit segment under {}", fits.display());
}

#[test]
fn one_non_finite_quantity_drops_out_and_everything_else_is_written() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_gh715_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(&bin, &tmp, &["fit", "predict", "--fit", "fit.toml"]);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    // 1. Exit 1: a non-finite value is a defect and the run says so.
    assert_eq!(
        out.status.code(),
        Some(1),
        "a non-finite quantity is a deterministic failure:\nstdout={stdout}\nstderr={stderr}"
    );

    let seg = segment(&tmp.join("results"));

    // 2. The observation stream's bands are written — the thing the issue is
    //    about. One reporting-only expression must not cost a stream its
    //    posterior predictive.
    assert!(
        seg.join("predictive").join("weekly_cases.tsv").is_file(),
        "the stream's predictive must survive a broken quantity:\n{stdout}\n{stderr}"
    );
    assert!(seg.join("observed").join("weekly_cases.tsv").is_file());

    // 3. The other quantities are written; only the offending one is missing.
    let qdir = seg.join("quantities");
    for name in ["prevalence", "peak", "never"] {
        assert!(
            qdir.join(format!("{name}.tsv")).is_file(),
            "quantity `{name}` is unaffected and must still be written"
        );
    }
    assert!(
        !qdir.join("ratio.tsv").is_file(),
        "the refused quantity writes no file — a band over a NaN must never reach disk"
    );

    // 4. …and the manifest does not advertise the file nobody wrote.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(seg.join("quantities.json")).unwrap(),
    )
    .unwrap();
    let listed: Vec<&str> = manifest["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(listed.contains(&"peak"), "listed: {listed:?}");
    assert!(
        !listed.contains(&"ratio"),
        "a manifest entry for an unwritten file sends a consumer to a missing \
         path: {listed:?}"
    );

    // 5. The failure is recorded with the quantity name and a draw index.
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(seg.join("report.json")).expect("report.json is written"),
    )
    .unwrap();
    let failures = report["failures"].as_array().unwrap();
    assert_eq!(failures.len(), 1, "one entry refused: {report:#}");
    assert_eq!(failures[0]["kind"], "non_finite");
    assert_eq!(failures[0]["at"]["quantity"], "ratio");
    assert!(
        failures[0]["draw"].is_number(),
        "the draw that produced it is named: {report:#}"
    );
    assert!(
        failures[0]["reason"].as_str().unwrap().contains("non-finite"),
        "{report:#}"
    );

    // 6. The run names the quantity on stderr, not just in the file.
    assert!(
        stderr.contains("quantity `ratio`"),
        "stderr names the offending entry:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
