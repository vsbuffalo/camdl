//! Invariant tests for the chain-binomial backend.
//! Mirrors gillespie_invariants.rs — population conservation, non-negativity,
//! determinism. The population conservation test is the one that would have
//! caught the total-propensity-as-per-capita bug (adc611e).

use std::path::Path;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

fn load_model(path: &str) -> ir::Model {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("could not read {}", path));
    ir::from_str(&contents)  // gh#audit-C8
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

fn apply_baseline(model: &mut ir::Model) {
    if let Some(preset) = model.presets.first() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
}

fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

fn chain_binomial_config(model: &ir::Model, dt: f64) -> SimConfig {
    SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end.min(30.0),
        dt,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    })
}

#[test]
fn test_sir_basic_population_conservation() {
    // SIR with no births/deaths: S + I + R must be constant.
    // This is the test that catches the propensity-as-per-capita bug:
    // with N=1000, mu*R would give p ≈ 1.0, killing everyone in one step.
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = chain_binomial_config(&model, 1.0);

    for seed in 0..20u64 {
        let traj = ChainBinomialSim.run(&compiled, &params, seed, &config).unwrap();
        if traj.snapshots.is_empty() { continue; }
        let n0: i64 = traj.snapshots[0].int_state.total();
        for snap in &traj.snapshots {
            let n = snap.int_state.total();
            assert_eq!(n, n0,
                "chain-binomial: population not conserved at t={} seed={}: expected {}, got {} (delta={})",
                snap.t, seed, n0, n, n - n0);
        }
    }
}

#[test]
fn test_sir_basic_non_negativity() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = chain_binomial_config(&model, 1.0);

    for seed in 0..20u64 {
        let traj = ChainBinomialSim.run(&compiled, &params, seed, &config).unwrap();
        for snap in &traj.snapshots {
            for &c in &snap.int_state.counts {
                assert!(c >= 0,
                    "chain-binomial: negative compartment at t={} seed={}", snap.t, seed);
            }
        }
    }
}

#[test]
fn test_determinism_same_seed() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = chain_binomial_config(&model, 0.5);

    let traj1 = ChainBinomialSim.run(&compiled, &params, 42, &config).unwrap();
    let traj2 = ChainBinomialSim.run(&compiled, &params, 42, &config).unwrap();

    assert_eq!(traj1.snapshots.len(), traj2.snapshots.len());
    for (s1, s2) in traj1.snapshots.iter().zip(&traj2.snapshots) {
        assert_eq!(s1.int_state, s2.int_state,
            "chain-binomial: trajectories differ at t={}", s1.t);
    }
}

#[test]
fn test_different_seeds_different_trajectories() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = chain_binomial_config(&model, 0.5);

    let traj1 = ChainBinomialSim.run(&compiled, &params, 1, &config).unwrap();
    let traj2 = ChainBinomialSim.run(&compiled, &params, 2, &config).unwrap();

    let identical = traj1.snapshots.iter().zip(&traj2.snapshots)
        .all(|(s1, s2)| s1.int_state == s2.int_state);
    assert!(!identical, "different seeds produced identical chain-binomial trajectories");
}

#[test]
fn test_large_compartment_no_blowup() {
    // Regression test: with R = 2.4M and mu*R = 131/day as total propensity,
    // the old code computed p = 1 - exp(-131 * dt) ≈ 1.0, killing everyone.
    // The fix divides by n_src first: p = 1 - exp(-mu * dt) ≈ 0.0000274.
    let mut model = load_model(&golden_path("sir_demography"));
    apply_baseline(&mut model);
    // Set large population to stress-test the per-capita conversion
    for p in &mut model.parameters {
        if p.name == "N0" { p.value = p.value.with_value(1_000_000.0); }
        if p.name == "I0" { p.value = p.value.with_value(100.0); }
    }
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = chain_binomial_config(&model, 0.5);

    let traj = ChainBinomialSim.run(&compiled, &params, 42, &config).unwrap();
    for snap in &traj.snapshots {
        let total: i64 = snap.int_state.total();
        // With births and deaths, population should stay within 50% of initial
        let n0: i64 = traj.snapshots[0].int_state.total();
        assert!(total > n0 / 2 && total < n0 * 2,
            "chain-binomial: population blowup/collapse at t={}: N={} (N0={})",
            snap.t, total, n0);
    }
}
