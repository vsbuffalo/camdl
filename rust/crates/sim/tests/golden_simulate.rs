//! Golden IR deserialization + simulation smoke tests.
//! Loads each golden IR file, runs all forward backends, checks basic invariants.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig, OdeConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

fn golden_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("../../../ir/golden")
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

fn load_golden(name: &str) -> ir::Model {
    let path = golden_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("could not read {:?}", path));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e))
}

fn check_trajectory_invariants(
    name: &str,
    traj: &sim::state::Trajectory,
    expect_conservation: bool,
) {
    assert!(!traj.snapshots.is_empty(), "{}: trajectory is empty", name);

    let n0: i64 = if expect_conservation {
        traj.snapshots[0].int_state.total()
    } else {
        0
    };

    for snap in &traj.snapshots {
        // Non-negativity: integer compartments
        for &c in &snap.int_state.counts {
            assert!(c >= 0, "{}: negative integer compartment at t={}", name, snap.t);
        }
        // Non-negativity: real compartments
        for &v in &snap.real_state.values {
            assert!(v >= 0.0, "{}: negative real compartment at t={}", name, snap.t);
        }
        // Population conservation for closed models
        if expect_conservation {
            let n = snap.int_state.total();
            assert_eq!(n, n0, "{}: population not conserved at t={}", name, snap.t);
        }
    }
}

#[test]
fn test_deserialize_all_golden() {
    for name in &["sir_basic", "pure_death", "two_state", "birth_death",
                  "sir_vaccination", "cholera_siwr", "sir_placebo_ekrng"]
    {
        let _model = load_golden(name);
        // Just check it deserializes without panicking
    }
}

#[test]
fn test_sir_basic_all_backends() {
    let mut model = load_golden("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();

    // Gillespie
    let g_config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });
    let traj = GillespieSim.run(&compiled, &params, 42, &g_config).unwrap();
    check_trajectory_invariants("sir_basic/gillespie", &traj, true);

    // Chain-binomial
    let cb_config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 1.0,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    let traj = ChainBinomialSim.run(&compiled, &params, 42, &cb_config).unwrap();
    check_trajectory_invariants("sir_basic/chain_binomial", &traj, false);
}

#[test]
fn test_pure_death_gillespie() {
    let model = load_golden("pure_death");
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });
    let traj = GillespieSim.run(&compiled, &params, 42, &config).unwrap();
    check_trajectory_invariants("pure_death", &traj, false);

    // Population should be non-increasing
    let counts: Vec<i64> = traj.snapshots.iter().map(|s| s.int_state.counts[0]).collect();
    for w in counts.windows(2) {
        assert!(w[1] <= w[0], "pure death: population increased from {} to {}", w[0], w[1]);
    }
}

#[test]
fn test_cholera_siwr_gillespie() {
    let model = load_golden("cholera_siwr");
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: 30.0, // shorter for test speed
        output_dt: None,
    });
    let traj = GillespieSim.run(&compiled, &params, 42, &config).unwrap();
    check_trajectory_invariants("cholera_siwr", &traj, false);
}

#[test]
fn test_config_mismatch_returns_err() {
    let mut model = load_golden("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    // Pass a non-Gillespie config to GillespieSim — should be ConfigMismatch error
    let wrong_config = SimConfig::Ode(OdeConfig {
        t_start: 0.0, t_end: 10.0, dt: 1.0,
    });
    let result = GillespieSim.run(&compiled, &params, 42, &wrong_config);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), sim::SimError::ConfigMismatch { .. }));
}
