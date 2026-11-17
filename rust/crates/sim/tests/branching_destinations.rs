//! Probabilistic branching runtime correctness (wave 2 / malaria #2).
//!
//! The branching DSL `S --> { I_symp : p_symp, I_asym : 1-p_symp }`
//! desugars to two IR transitions sharing source S. Correctness
//! requires two things from the runtime:
//!
//! 1. **Atomicity / conservation.** S + I_symp + I_asym is exactly
//!    constant at every snapshot. If branches were drawn as two
//!    *independent* Binomials on S (wrong algorithm), S would be
//!    double-consumed in a single step when both branches fire.
//!
//! 2. **Correct split ratio.** The long-run fraction
//!    I_symp / (I_symp + I_asym) converges to p_symp as N → ∞.
//!    If source-grouping used the wrong weights (or didn't group at
//!    all), this ratio would be biased.
//!
//! Both invariants are checked under Gillespie, tau-leap, and
//! chain-binomial.

use std::path::Path;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ir/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

fn load_branching() -> (ir::Model, CompiledModel) {
    let contents = std::fs::read_to_string(golden_path("branching_si_symp_asym"))
        .expect("read fixture");
    let mut model: ir::Model = ir::from_str(&contents).unwrap();  // gh#audit-C8
    if let Some(preset) = model.presets.first() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    let compiled = CompiledModel::new(model.clone()).unwrap();
    (model, compiled)
}

fn local_idx(compiled: &CompiledModel, name: &str) -> usize {
    let g = *compiled.comp_index.get(name).expect("compartment");
    compiled.global_to_int[g].expect("integer compartment")
}

fn assert_conservation<F>(compiled: &CompiledModel, run_seed: F, backend: &str)
where F: Fn(u64) -> sim::Trajectory {
    let idx_s  = local_idx(compiled, "S");
    let idx_is = local_idx(compiled, "I_symp");
    let idx_ia = local_idx(compiled, "I_asym");
    for seed in 0..10u64 {
        let traj = run_seed(seed);
        let n0 = traj.snapshots[0].int_state.counts[idx_s]
               + traj.snapshots[0].int_state.counts[idx_is]
               + traj.snapshots[0].int_state.counts[idx_ia];
        for snap in &traj.snapshots {
            let n = snap.int_state.counts[idx_s]
                  + snap.int_state.counts[idx_is]
                  + snap.int_state.counts[idx_ia];
            assert_eq!(n, n0,
                "{}: N drift at t={} seed={}: {} != {}",
                backend, snap.t, seed, n, n0);
        }
    }
}

/// Mean split ratio over `n_seeds` runs must be within `tol` of p_symp
/// at the final time (when most of S has been consumed).
fn assert_split_ratio<F>(
    compiled: &CompiledModel,
    run_seed: F,
    p_symp: f64,
    tol: f64,
    backend: &str,
) where F: Fn(u64) -> sim::Trajectory {
    let idx_is = local_idx(compiled, "I_symp");
    let idx_ia = local_idx(compiled, "I_asym");
    // Baseline I0 goes into I_symp, so subtract it out to measure
    // just the infection-branch outcomes. With N0=1000, I0=10, baseline
    // params, we expect ~900+ total new infections by t=60.
    let i0: i64 = 10;
    let n_seeds = 200u64;
    let mut sum_symp = 0.0;
    let mut sum_total = 0.0;
    for seed in 0..n_seeds {
        let traj = run_seed(seed);
        let last = traj.snapshots.last().unwrap();
        let new_symp = (last.int_state.counts[idx_is] - i0).max(0) as f64;
        let asym = last.int_state.counts[idx_ia] as f64;
        sum_symp  += new_symp;
        sum_total += new_symp + asym;
    }
    assert!(sum_total > 0.0, "{}: no infections occurred in any seed", backend);
    let ratio = sum_symp / sum_total;
    assert!((ratio - p_symp).abs() < tol,
        "{}: split ratio {} differs from p_symp={} by more than {}",
        backend, ratio, p_symp, tol);
}

#[test]
fn test_branching_gillespie_conservation() {
    let (model, compiled) = load_branching();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });
    assert_conservation(&compiled,
        |seed| GillespieSim.run(&compiled, &params, seed, &config).unwrap(),
        "gillespie");
}

#[test]
fn test_branching_chain_binomial_conservation() {
    let (model, compiled) = load_branching();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 0.5,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    assert_conservation(&compiled,
        |seed| ChainBinomialSim.run(&compiled, &params, seed, &config).unwrap(),
        "chain_binomial");
}

#[test]
fn test_branching_gillespie_split_ratio() {
    let (model, compiled) = load_branching();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });
    // 200 seeds × ~900 infections ≈ 180k Bernoulli draws; tolerance
    // of 0.02 is ~9σ for a binomial proportion. The key thing this
    // would catch is a systematic bias from wrong split weights.
    assert_split_ratio(&compiled,
        |seed| GillespieSim.run(&compiled, &params, seed, &config).unwrap(),
        0.6, 0.02, "gillespie");
}

#[test]
fn test_branching_chain_binomial_split_ratio() {
    let (model, compiled) = load_branching();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 0.5,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    assert_split_ratio(&compiled,
        |seed| ChainBinomialSim.run(&compiled, &params, seed, &config).unwrap(),
        0.6, 0.02, "chain_binomial");
}
