//! `camdl simulate --backend ode` refuses a fixed RK4 step that drives a
//! compartment below zero, instead of setting it back to zero (gh#1263).
//!
//! The model is a closed cohort of 1e9 in an 8-stage `via erlang` latent chain
//! with rate 0.2 per day, so the product `dt * stages * rate` is 1.6 at
//! `dt = 1` and 0.16 at `dt = 0.1`. One RK4 step of a linear chain starting in
//! stage 1 places the degree-4 Taylor terms of `exp(-1.6)` in stages 1 to 5,
//! and two of them are negative; setting those stages to zero kept the flow
//! already credited downstream, and the cohort grew to 3.2e9 by day 8. At
//! `dt = 1` the run must now fail and say why; at `dt = 0.1` it must run and
//! conserve the cohort.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_camdl"))
}

const MODEL: &str = r#"
time_unit = 'days
compartments { E, I }
let sigma : rate = 0.2 'per_day
transitions { latency : E --> I via erlang(stages = 8, rate = sigma) }
init { E = 1000000000 }
simulate { from = 0 'days  to = 8 'days }
"#;

const COHORT: f64 = 1e9;

fn simulate(root: &Path, dt: &str) -> (std::process::Output, PathBuf) {
    let model = root.join("ode8.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let out = root.join(format!("ode8_dt{dt}.tsv"));
    let output = Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(root)
        .args([
            "simulate",
            model.to_str().unwrap(),
            "--backend",
            "ode",
            "--dt",
            dt,
            "--seed",
            "1",
            "--store",
            root.join("store").to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("spawn camdl");
    (output, out)
}

#[test]
fn coarse_step_that_drives_a_stage_negative_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (output, _) = simulate(tmp.path(), "1");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "dt = 1 drives E_s2 and E_s4 negative on the first step; the run must fail.\n\
         stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("compartment 'E_s2'")
            && stderr.contains("dt = 1")
            && stderr.contains("t = 0")
            && stderr.contains("smaller dt"),
        "the refusal must name the compartment, the step and the time, and suggest a \
         smaller dt.\nstderr:\n{stderr}"
    );
}

#[test]
fn fine_step_runs_and_conserves_the_cohort() {
    let tmp = tempfile::tempdir().unwrap();
    let (output, tsv) = simulate(tmp.path(), "0.1");
    assert!(
        output.status.success(),
        "dt = 0.1 keeps dt * stages * rate at 0.16; the run must succeed.\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&tsv).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let compartments: Vec<usize> = header
        .iter()
        .enumerate()
        .filter(|(_, h)| h.starts_with("E_s") || **h == "I")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(compartments.len(), 9, "8 latent stages and I, header: {header:?}");
    let mut rows = 0;
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        let total: f64 = compartments.iter().map(|&i| cols[i].parse::<f64>().unwrap()).sum();
        // Snapshots round each compartment to an integer count, so the total
        // can be off by at most half a person per compartment.
        assert!(
            (total - COHORT).abs() <= 0.5 * compartments.len() as f64,
            "the closed cohort must be conserved, row t = {}: total {total}",
            cols[0]
        );
        rows += 1;
    }
    assert_eq!(rows, 9, "one row per day from 0 to 8");
}
