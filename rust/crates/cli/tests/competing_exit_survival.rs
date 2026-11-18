//! The fraction that completes an Erlang dwell beside a competing exit
//! (proposal docs/dev/proposals/2026-11-17-unified-staged-residence.md,
//! "Competing exits").
//!
//! A cohort starts in `E`, which has an Erlang-`k` dwell of stage rate
//! `k * sigma` into `I` and a death at constant hazard `mu` per person into
//! `D`. Nothing enters `E`. A person advances from each stage with probability
//! `k*sigma / (k*sigma + mu)`, so once `E` has emptied the fraction of the
//! cohort in `I` is `(k*sigma / (k*sigma + mu))^k`.
//!
//! Simulated deterministically (ODE backend), so the comparison is exact up
//! to the integrator's error. The model text is the same for every `k`,
//! including `k = 1`, where `E` is not staged.

use std::path::{Path, PathBuf};
use std::process::Command;

const SIGMA: f64 = 0.25;
const MU: f64 = 0.1;
/// Large, so the three decimals the trajectory file prints resolve the
/// fraction to one part in 1e12.
const COHORT: f64 = 1.0e9;

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

fn model(stages: u32) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ E, I, D }}

let sigma : rate = {SIGMA} 'per_day
let mu    : rate = {MU} 'per_day

transitions {{
  latency : E --> I via erlang(stages = {stages}, rate = sigma)
  die     : E --> D @ mu * E
}}

init {{ E = {COHORT:.0} }}

simulate {{ from = 0 'days  to = 400 'days }}
"#
    )
}

/// The last row of the trajectory, as (column name, value) pairs.
fn final_row(dir: &Path, stages: u32) -> Vec<(String, f64)> {
    let src = dir.join(format!("cohort_{stages}.camdl"));
    std::fs::write(&src, model(stages)).unwrap();
    let out = dir.join(format!("cohort_{stages}.tsv"));
    let o = Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", dir.join("cas"))
        .args([
            "simulate",
            src.to_str().unwrap(),
            "--backend", "ode",
            "--dt", "0.01",
            "--output-every", "400",
            "--seed", "1",
            "-o", out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn simulate");
    assert!(
        o.status.success(),
        "simulate at stages = {stages} must succeed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let tsv = std::fs::read_to_string(out).unwrap();
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<String> =
        lines.next().expect("header").split('\t').map(str::to_string).collect();
    let last: Vec<f64> = lines
        .next_back()
        .expect("a final row")
        .split('\t')
        .map(|f| f.parse::<f64>().expect("number"))
        .collect();
    header.into_iter().zip(last).collect()
}

fn column(row: &[(String, f64)], name: &str) -> f64 {
    row.iter()
        .find(|(h, _)| h == name)
        .unwrap_or_else(|| panic!("no column '{name}' in {:?}", row.iter().map(|r| &r.0).collect::<Vec<_>>()))
        .1
}

#[test]
fn fraction_completing_the_dwell_matches_the_closed_form() {
    let tmp = tempfile::tempdir().unwrap();
    for stages in [1u32, 2, 3, 8] {
        let row = final_row(tmp.path(), stages);
        let k = f64::from(stages);
        let expected = (k * SIGMA / (k * SIGMA + MU)).powi(stages as i32);
        let completed = column(&row, "I") / COHORT;
        let died = column(&row, "D") / COHORT;
        // The cohort has left E: what remains after 400 days is below the
        // tolerance of the comparison.
        let in_e: f64 = row
            .iter()
            .filter(|(h, _)| h == "E" || h.starts_with("E_"))
            .map(|(_, v)| v)
            .sum();
        assert!(in_e / COHORT < 1e-9, "stages = {stages}: {in_e} still in E");
        assert!(
            (completed - expected).abs() < 1e-6,
            "stages = {stages}: fraction completing the dwell is {completed}, \
             expected (k*sigma/(k*sigma+mu))^k = {expected}"
        );
        assert!(
            (completed + died - 1.0).abs() < 1e-9,
            "stages = {stages}: the cohort is conserved ({completed} + {died})"
        );
    }
}
