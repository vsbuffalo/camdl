//! Unit invariance of the dynamics (proposal
//! docs/dev/proposals/2026-11-06-dimension-checking-as-one-constraint-system.md
//! §7.4): one model written with unit literals, compiled under
//! `time_unit = 'days` and under `'weeks`, is the same epidemic. Simulated
//! deterministically (ODE backend) with the same physical step, the flows over
//! each week agree to the precision the TSV prints; and with a weekly
//! reported-case stream, the particle filter at a fixed seed gives the same
//! log-likelihood under both.
//!
//! A unit literal whose scale were dropped (gh#1006: a `'per_month` parameter
//! read per `time_unit`) makes the weeks model a different epidemic, and this
//! test fails. The prior half of §7.4, that log-priors differ by exactly the
//! Jacobian of the change of unit, is `test_prior_unit_invariance` in
//! ocaml/test/test_compiler.ml.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn model(time_unit: &str) -> String {
    format!(
        r#"time_unit = '{time_unit}

compartments {{ S, I, R }}

let N = S + I + R
let beta  : rate = 0.4 'per_day
let gamma : rate = 1 / (5 'days)
let rho   : rate = 0.2 'per_week

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  waning    : R --> S  @ rho * R
}}

init {{ S = 990  I = 10 }}

simulate {{ from = 0 'days  to = 70 'days }}
"#
    )
}

/// The flow columns of every emitted row, after the header.
fn flows(tsv: &str) -> Vec<Vec<f64>> {
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let cols: Vec<usize> = header
        .iter()
        .enumerate()
        .filter(|(_, h)| h.starts_with("flow_"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(cols.len(), 3, "three flow columns in {header:?}");
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            cols.iter().map(|&i| f[i].parse::<f64>().expect("number")).collect()
        })
        .collect()
}

fn simulate(dir: &Path, time_unit: &str, dt: &str, every: &str) -> Vec<Vec<f64>> {
    let src = dir.join(format!("sir_{time_unit}.camdl"));
    std::fs::write(&src, model(time_unit)).unwrap();
    let out = dir.join(format!("{time_unit}.tsv"));
    let o = Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("cas"))
        .args([
            "simulate",
            src.to_str().unwrap(),
            "--backend", "ode",
            "--dt", dt,
            "--output-every", every,
            "--seed", "1",
            "-o", out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn simulate");
    assert!(
        o.status.success(),
        "simulate under {time_unit} must succeed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    flows(&std::fs::read_to_string(out).unwrap())
}

#[test]
fn days_and_weeks_are_one_epidemic() {
    let tmp = tempfile::tempdir().unwrap();
    // One step of 0.07 days = 0.01 weeks, one row per week.
    let days = simulate(tmp.path(), "days", "0.07", "7");
    let weeks = simulate(tmp.path(), "weeks", "0.01", "1");
    assert_eq!(days.len(), weeks.len(), "one row per week in both");
    assert!(days.len() >= 10, "ten weeks of rows");
    let peak = days.iter().flatten().cloned().fold(0.0, f64::max);
    assert!(peak > 10.0, "an epidemic happened (peak weekly flow {peak})");
    for (k, (d, w)) in days.iter().zip(&weeks).enumerate() {
        for (j, (a, b)) in d.iter().zip(w).enumerate() {
            // The TSV prints four decimals.
            assert!(
                (a - b).abs() <= 2e-4,
                "week {k}, flow {j}: {a} under days, {b} under weeks"
            );
        }
    }
}

/// The same SIR with a weekly reported-case stream, under `'days` and
/// `'weeks`, for the particle-filter half of §7.4: the observation window
/// (`covers`) and the likelihood see the unit change too.
fn observed_model(time_unit: &str) -> String {
    format!(
        r#"time_unit = '{time_unit}

compartments {{ S, I, R }}

let N = S + I + R
let beta  : rate = 0.4 'per_day
let gamma : rate = 1 / (5 'days)
let rho   : probability = 0.5

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

observations {{
  cases {{
    columns {{ time : time, y : count }}
    covers = closing_at(time, 7 'days)
    projected = incidence(infection)
    y ~ poisson(rate = rho * projected)
  }}
}}

init {{ S = 990  I = 10 }}

simulate {{ from = 0 'days  to = 70 'days }}
"#
    )
}

const WEEKLY_CASES: [u32; 10] = [9, 17, 30, 45, 52, 40, 25, 14, 7, 3];

/// The log-likelihood `camdl pfilter` prints, at a fixed seed.
fn pfilter_loglik(dir: &Path, time_unit: &str, week: u32, dt: &str) -> f64 {
    let src = dir.join(format!("obs_{time_unit}.camdl"));
    std::fs::write(&src, observed_model(time_unit)).unwrap();
    let data = dir.join(format!("cases_{time_unit}.tsv"));
    let rows: String = WEEKLY_CASES
        .iter()
        .enumerate()
        .map(|(k, y)| format!("{}\t{y}\n", (k as u32 + 1) * week))
        .collect();
    std::fs::write(&data, format!("time\ty\n{rows}")).unwrap();
    let o = Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("cas"))
        .args([
            "pfilter",
            src.to_str().unwrap(),
            "--data", data.to_str().unwrap(),
            "--particles", "200",
            "--dt", dt,
            "--seed", "3",
            "--no-progress",
        ])
        .output()
        .expect("spawn pfilter");
    assert!(
        o.status.success(),
        "pfilter under {time_unit} must succeed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let out = String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr);
    out.lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no log-likelihood line in pfilter output:\n{out}"))
}

#[test]
fn days_and_weeks_give_one_likelihood() {
    let tmp = tempfile::tempdir().unwrap();
    // One step of one day in both: 1 under days, 1/7 under weeks. The same
    // seed and the same number of steps draw the same random numbers, so the
    // two filters agree to round-off unless a unit is dropped somewhere.
    let days = pfilter_loglik(tmp.path(), "days", 7, "1");
    let weeks = pfilter_loglik(tmp.path(), "weeks", 1, "0.14285714285714285");
    assert!(days.is_finite() && days < 0.0, "a log-likelihood ({days})");
    // pfilter prints four decimals.
    assert!(
        (days - weeks).abs() <= 2e-4,
        "log-likelihood {days} under days, {weeks} under weeks"
    );
}
