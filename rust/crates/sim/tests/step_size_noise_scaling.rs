//! The chain-binomial backend's gamma environmental noise keeps its intensity
//! per unit of time when the step size changes.
//!
//! `StatefulRng::gamma_multiplier(σ², dt)` draws a per-step multiplier
//! `G ~ Gamma(shape = dt/σ², scale = σ²/dt)`, so `E[G] = 1` and
//! `Var[G] = σ²/dt`. Summed over a span `T`, the integrated noise
//! `W(T) = Σ G·dt` is a sum of `T/dt` independent `Gamma(dt/σ², σ²)` variables,
//! which is `Gamma(T/σ², σ²)` at every `dt`: mean `T`, variance `σ²·T`. That is
//! the integrated gamma white noise of Bretó, He, Ionides & King (2009, Ann.
//! Appl. Stat. 3(1):319-348), whose increments have `E[ΔW] = Δt` and
//! `Var[ΔW] = σ²Δt`, as pomp's `rgammawn(sigma, dt)` draws them (pomp
//! reference manual, `?rgammawn`); the multiplier is `ΔW/Δt`.
//!
//! The tests pin both levels. At the sampler, the per-step multiplier has mean 1
//! and variance `σ²/dt`, and its integral over one day has mean 1 and variance
//! `σ²`. At a model run, a pure-death process `X --> D` at per-capita rate
//! `μ·G` leaves `X(T) = X(0)·exp(−μ·W(T))` up to binomial noise, so with
//! `X(0) = 10⁹` the run's own `−log(X(T)/X(0))/μ` reads `W(T)` off the
//! trajectory, and its mean and variance are checked against `T` and `σ²·T` at
//! `dt = 1, 0.25, 0.1`.
//!
//! What a wrong scaling would look like: a per-step variance of `σ²` (not
//! `σ²/dt`) gives an integrated variance of `σ²·dt`, ten times too small at
//! `dt = 0.1`; a per-step variance of `σ²/dt²` gives one ten times too large.
//! Either is far outside the bands below.
//!
//! Tolerances are 4 Monte Carlo standard errors, computed from the sample
//! size and the known gamma moments (`Var[s²] ≈ v²·(2/(n−1) + κ/n)` with excess
//! kurtosis `κ = 6/shape`), not tuned to today's draws. Seeds are fixed, so the
//! test is deterministic.

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    rng::StatefulRng,
    simulate::Simulate,
    ChainBinomialSim,
};
use std::path::PathBuf;

const SIGMA_SQ: f64 = 0.25;
const DTS: [f64; 3] = [1.0, 0.25, 0.1];

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, v)
}

/// The 4-SE half-widths for the sample mean and the sample variance of `n`
/// draws from a gamma distribution with variance `v` and shape `shape`.
fn bands(n: usize, v: f64, shape: f64) -> (f64, f64) {
    let n = n as f64;
    let kurt = 6.0 / shape;
    let se_mean = (v / n).sqrt();
    let se_var = v * (2.0 / (n - 1.0) + kurt / n).sqrt();
    (4.0 * se_mean, 4.0 * se_var)
}

/// The per-step multiplier has mean 1 and variance σ²/dt.
#[test]
fn per_step_multiplier_has_unit_mean_and_variance_sigma_sq_over_dt() {
    let n = 200_000;
    for (k, &dt) in DTS.iter().enumerate() {
        let mut rng = StatefulRng::new(1000 + k as u64);
        let g: Vec<f64> = (0..n).map(|_| rng.gamma_multiplier(SIGMA_SQ, dt)).collect();
        let (m, v) = mean_var(&g);
        let want_v = SIGMA_SQ / dt;
        let (bm, bv) = bands(n, want_v, dt / SIGMA_SQ);
        println!("dt = {dt}: per-step mean {m:.4} (want 1 ± {bm:.4}), var {v:.4} (want {want_v:.4} ± {bv:.4})");
        assert!((m - 1.0).abs() < bm, "dt = {dt}: per-step multiplier mean {m} is not 1 (± {bm})");
        assert!((v - want_v).abs() < bv,
            "dt = {dt}: per-step multiplier variance {v} is not σ²/dt = {want_v} (± {bv})");
    }
}

/// Integrated over one unit of time (1/dt steps), the multiplier has mean 1
/// and variance σ², whatever the step.
#[test]
fn multiplier_integrated_over_one_day_has_variance_sigma_sq_at_every_dt() {
    let n = 50_000;
    for (k, &dt) in DTS.iter().enumerate() {
        let steps = (1.0 / dt).round() as usize;
        let mut rng = StatefulRng::new(2000 + k as u64);
        let w: Vec<f64> = (0..n)
            .map(|_| (0..steps).map(|_| rng.gamma_multiplier(SIGMA_SQ, dt) * dt).sum())
            .collect();
        let (m, v) = mean_var(&w);
        let (bm, bv) = bands(n, SIGMA_SQ, 1.0 / SIGMA_SQ);
        println!("dt = {dt}: integrated mean {m:.4} (want 1 ± {bm:.4}), var {v:.4} (want {SIGMA_SQ} ± {bv:.4})");
        assert!((m - 1.0).abs() < bm, "dt = {dt}: integrated multiplier mean {m} is not 1 (± {bm})");
        assert!((v - SIGMA_SQ).abs() < bv,
            "dt = {dt}: integrated multiplier variance {v} is not σ² = {SIGMA_SQ} (± {bv})");
    }
}

fn load_death_model() -> ir::Model {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/step_size/ir/death_overdispersed.ir.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));
    ir::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"))
}

fn fixed_param(model: &ir::Model, name: &str) -> f64 {
    match model.parameters.iter().find(|p| p.name == name).map(|p| &p.value) {
        Some(ir::parameter::ParamValue::Fixed { value }) => *value,
        other => panic!("parameter {name} is not a baked fixed value: {other:?}"),
    }
}

/// A model run at dt = 1, 0.25 and 0.1: the integrated noise read off a
/// pure-death trajectory over one day has mean 1 and variance σ².
#[test]
fn pure_death_run_reads_integrated_noise_with_variance_sigma_sq_at_every_dt() {
    let model = load_death_model();
    let mu = fixed_param(&model, "mu");
    let sigma_sq = fixed_param(&model, "sigma_sq");
    let x0 = fixed_param(&model, "X0");
    assert_eq!(sigma_sq, SIGMA_SQ, "fixture's sigma_sq differs from the test's");
    assert!(x0 >= 1e9, "X0 must be large for the binomial noise to be negligible");
    let ix = model.compartments.iter().position(|c| c.name == "X").expect("compartment X");
    let (t_start, t_end) = (model.simulation.t_start, model.simulation.t_end);
    assert_eq!(t_end - t_start, 1.0, "the fixture's span is one day");

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    let n = 4000;
    for &dt in &DTS {
        let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt });
        let w: Vec<f64> = (0..n as u64)
            .map(|seed| {
                let traj = ChainBinomialSim.run(&compiled, &params, seed, &cfg).unwrap();
                let first = &traj.snapshots.first().unwrap().int_state.counts;
                let last = &traj.snapshots.last().unwrap().int_state.counts;
                assert_eq!(first[ix] as f64, x0);
                -((last[ix] as f64) / x0).ln() / mu
            })
            .collect();
        let (m, v) = mean_var(&w);
        let (bm, bv) = bands(n, SIGMA_SQ, 1.0 / SIGMA_SQ);
        println!("dt = {dt}: run-level W(1) mean {m:.4} (want 1 ± {bm:.4}), var {v:.4} (want {SIGMA_SQ} ± {bv:.4})");
        assert!((m - 1.0).abs() < bm, "dt = {dt}: integrated noise from the run has mean {m}, not 1 (± {bm})");
        assert!((v - SIGMA_SQ).abs() < bv,
            "dt = {dt}: integrated noise from the run has variance {v}, not σ²·T = {SIGMA_SQ} (± {bv})");
    }
}
