//! A competing exit whose rate is not linear in the staged compartment, run
//! on the ODE and chain-binomial backends (proposal
//! docs/dev/proposals/2026-11-17-unified-staged-residence.md, "Competing
//! exits": the guarded quotient `R_s = R * E_s / E`, zero when `E = 0`).
//!
//! `treat : E --> T @ cmax * E * K / (K + E)` (a hazard per person that falls
//! as the compartment fills) is written beside an Erlang dwell on `E`. The
//! compiler expands it to one transition per stage. What the runtime must
//! then deliver is the written rate: summed over the stages, the flow of
//! `treat` is the flow of `cmax * E * K / (K + E)` with `E` the compartment's
//! total, however the people are spread over the stages, and
//! an empty compartment gives zero flow, not 0/0.

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

const STAGES: usize = 3;
const SIGMA: f64 = 0.3;

/// `S` feeds `E` at `beta * S`; `E` leaves by the Erlang dwell into `I` and by
/// the saturating treatment into `T`.
fn model(cmax: f64, k: f64, beta: f64, init: &str, days: u32) -> String {
    format!(
        r#"time_unit = 'days

compartments {{ S, E, I, T }}

let beta  : rate  = {beta} 'per_day
let sigma : rate  = {SIGMA} 'per_day
let cmax  : rate  = {cmax} 'per_day
let K     : count = {k}

transitions {{
  infection : S --> E @ beta * S
  latency   : E --> I via erlang(stages = {STAGES}, rate = sigma)
  treat     : E --> T @ cmax * E * K / (K + E)
}}

init {{ {init} }}

simulate {{ from = 0 'days  to = {days} 'days }}
"#
    )
}

/// A trajectory: column names and rows of numbers.
struct Table {
    header: Vec<String>,
    rows: Vec<Vec<f64>>,
}

impl Table {
    fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column '{name}' in {:?}", self.header))
    }
    fn e_total(&self, row: &[f64]) -> f64 {
        (1..=STAGES).map(|s| row[self.col(&format!("E_s{s}"))]).sum()
    }
    fn treat_flow(&self, row: &[f64]) -> f64 {
        (1..=STAGES).map(|s| row[self.col(&format!("flow_treat_s{s}"))]).sum()
    }
}

fn simulate(dir: &Path, tag: &str, src: &str, backend: &str, dt: &str, seed: u64) -> Table {
    let path = dir.join(format!("{tag}.camdl"));
    std::fs::write(&path, src).unwrap();
    let out = dir.join(format!("{tag}.tsv"));
    let o = Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("camdl-store"))
        .args(["simulate", path.to_str().unwrap(), "--backend", backend, "--dt", dt])
        .args(["--output-every", dt, "--seed", &seed.to_string(), "-o", out.to_str().unwrap()])
        .output()
        .expect("spawn simulate");
    assert!(
        o.status.success(),
        "{tag}: simulate on {backend} must succeed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let tsv = std::fs::read_to_string(out).unwrap();
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    // The `-o` table opens with the cell's coordinate columns (`replicate`,
    // `scenario`, `seed`); the trajectory starts at `t`.
    let full_header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let t_col = full_header.iter().position(|h| *h == "t").expect("a `t` column");
    let header: Vec<String> = full_header[t_col..].iter().map(|h| h.to_string()).collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split('\t').skip(t_col).map(|f| f.parse::<f64>().expect("number")).collect())
        .collect();
    for (i, row) in rows.iter().enumerate() {
        for (h, v) in header.iter().zip(row) {
            assert!(v.is_finite(), "{tag}: row {i}, column {h} is {v}");
        }
    }
    Table { header, rows }
}

/// On the ODE the flow of `treat` over a step, summed over its stage copies,
/// is the integral of the written rate at the compartment total over that
/// step, while the cohort moves through the stages.
#[test]
fn ode_stage_flows_sum_to_the_flow_of_the_written_rate() {
    let tmp = tempfile::tempdir().unwrap();
    // Large numbers, so the whole numbers the trajectory prints for the
    // compartments resolve the rate to one part in 1e9.
    let (cmax, k, dt) = (0.5, 2.0e8, 0.01);
    let t = simulate(
        tmp.path(),
        "ode_flow",
        &model(cmax, k, 0.0, "E = 1000000000", 6),
        "ode",
        "0.01",
        1,
    );
    let written = |e: f64| cmax * e * k / (k + e);
    let mut checked = 0;
    let mut spread = false;
    for w in t.rows.windows(2) {
        let (e0, e1) = (t.e_total(&w[0]), t.e_total(&w[1]));
        // Trapezoid over the step; its error is O(dt^3) of a smooth rate.
        let expected = dt * (written(e0) + written(e1)) / 2.0;
        let flow = t.treat_flow(&w[1]);
        assert!(
            (flow - expected).abs() <= 1e-5 * expected,
            "step ending t = {}: stage flows sum to {flow}, the written rate gives {expected}",
            w[1][0]
        );
        // Each stage's flow is its occupancy share of that total.
        for s in 1..=STAGES {
            let mid = (w[0][t.col(&format!("E_s{s}"))] + w[1][t.col(&format!("E_s{s}"))]) / 2.0;
            let share = w[1][t.col(&format!("flow_treat_s{s}"))];
            let expected_share = expected * mid / ((e0 + e1) / 2.0);
            assert!(
                (share - expected_share).abs() <= 1e-3 * expected + 1.0,
                "step ending t = {}: stage {s} flow {share}, occupancy share {expected_share}",
                w[1][0]
            );
        }
        spread |= (1..=STAGES).all(|s| w[1][t.col(&format!("E_s{s}"))] > 1.0e6);
        checked += 1;
    }
    assert_eq!(checked, 600);
    assert!(spread, "the cohort is spread over every stage at some step");
    // Everyone who left E by treatment is in T.
    let last = t.rows.last().unwrap();
    let total_flow: f64 = t.rows.iter().map(|r| t.treat_flow(r)).sum();
    assert!((last[t.col("T")] - total_flow).abs() <= 1e-6 * total_flow, "T = sum of the flows");
}

/// On chain-binomial, with the people spread over the stages, the expected
/// flow of `treat` in one step is what the written rate gives for the
/// compartment total: every individual, in whatever stage, leaves by treatment
/// with probability `(1 - exp(-h dt)) * (R / E) / h`, where `R / E` is the
/// written hazard per person and `h` adds the stage's progression rate.
#[test]
fn chain_binomial_stage_flows_sum_to_the_flow_of_the_written_rate() {
    let tmp = tempfile::tempdir().unwrap();
    let (cmax, k) = (0.4, 20000.0);
    // Unequal stages: an allocation that ignored occupancy would show.
    let stages = [15000.0, 9000.0, 6000.0];
    let e: f64 = stages.iter().sum();
    let per_person = cmax * k / (k + e);
    let h = STAGES as f64 * SIGMA + per_person;
    let p_treat = (1.0 - (-h).exp()) * per_person / h;
    let src = model(cmax, k, 0.0, "stage(E, 1) = 15000  stage(E, 2) = 9000  stage(E, 3) = 6000", 1);
    let n_seeds = 60;
    let mut totals = Vec::new();
    let mut by_stage = [0.0f64; STAGES];
    for seed in 1..=n_seeds {
        let t = simulate(tmp.path(), &format!("cb_{seed}"), &src, "chain_binomial", "1", seed);
        let row = &t.rows[1];
        totals.push(t.treat_flow(row));
        for (s, acc) in by_stage.iter_mut().enumerate() {
            *acc += row[t.col(&format!("flow_treat_s{}", s + 1))];
        }
    }
    let n = totals.len() as f64;
    let mean = totals.iter().sum::<f64>() / n;
    let var = totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    let expected = e * p_treat;
    assert!(
        (mean - expected).abs() < 4.0 * se,
        "mean total flow {mean} over {n_seeds} seeds, the written rate gives {expected} (se {se})"
    );
    // And each stage carries its occupancy share of it.
    for (s, total) in by_stage.iter().enumerate() {
        let mean_s = total / n;
        let expected_s = stages[s] * p_treat;
        // Binomial sd of one draw, over n seeds.
        let se_s = (stages[s] * p_treat * (1.0 - p_treat) / n).sqrt();
        assert!(
            (mean_s - expected_s).abs() < 4.0 * se_s,
            "stage {}: mean flow {mean_s}, occupancy share {expected_s} (se {se_s})",
            s + 1
        );
    }
}

/// An empty compartment has a rate of zero, not 0/0: the run proceeds, every
/// printed value is a number, and nothing flows through `treat` until `E`
/// holds someone.
#[test]
fn an_empty_staged_compartment_has_zero_flow_on_both_backends() {
    let tmp = tempfile::tempdir().unwrap();
    for (backend, dt) in [("ode", "0.05"), ("chain_binomial", "1")] {
        // Nobody anywhere: E stays empty for the whole run.
        let t = simulate(
            tmp.path(),
            &format!("empty_{backend}"),
            &model(0.5, 20.0, 0.0, "S = 0", 5),
            backend,
            dt,
            1,
        );
        for row in &t.rows {
            assert_eq!(t.e_total(row), 0.0, "{backend}: E stays empty");
            assert_eq!(t.treat_flow(row), 0.0, "{backend}: no flow from an empty compartment");
        }
        // E starts empty and fills from S: the first state has E = 0, and the
        // exit then carries people once there are some.
        let t = simulate(
            tmp.path(),
            &format!("fills_{backend}"),
            &model(0.5, 20.0, 0.5, "S = 1000", 12),
            backend,
            dt,
            1,
        );
        assert_eq!(t.e_total(&t.rows[0]), 0.0, "{backend}: E starts empty");
        if backend == "chain_binomial" {
            // The first step is drawn from the state with E = 0.
            assert_eq!(t.treat_flow(&t.rows[1]), 0.0, "{backend}: first step");
        }
        let last = t.rows.last().unwrap();
        let treated = last[t.col("T")];
        assert!(treated > 0.0, "{backend}: treatment carried people once E held some");
        let flows: f64 = t.rows.iter().map(|r| t.treat_flow(r)).sum();
        let tol = if backend == "ode" { 1.0 } else { 0.0 };
        assert!((treated - flows).abs() <= tol, "{backend}: T = {treated}, flows sum to {flows}");
        let people = last[t.col("S")] + t.e_total(last) + last[t.col("I")] + treated;
        let tol = if backend == "ode" { 4.0 } else { 0.0 };
        assert!((people - 1000.0).abs() <= tol, "{backend}: {people} people of 1000");
    }
}
