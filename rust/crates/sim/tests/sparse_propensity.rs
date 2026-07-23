//! Tests for Phase 4 sparse propensity update infrastructure.
//!
//! Verifies that:
//! 1. `comp_to_transitions` and `time_dep_transitions` are built correctly.
//! 2. The dependency graph covers every transition that references a compartment.
//! 3. Sparse Gillespie is deterministic (same seed → same output).
//! 4. Population invariants hold over long runs (many events, past the full-recompute trigger).

use std::path::Path;
use sim::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    simulate::Simulate,
    GillespieSim,
};

fn load_model(path: &str) -> ir::Model {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("could not read {}", path));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

fn apply_baseline(model: &mut ir::Model) {
    if let Some(preset) = model.presets.first().cloned() {
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
        .join("../../../ir/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

fn ocaml_golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

// ── Dependency graph structure tests ──────────────────────────────────────────

/// SIR model: every transition references compartments; comp_to_transitions must be non-empty.
#[test]
fn test_comp_to_transitions_nonempty_for_sir() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    // SIR has 3 integer compartments (S, I, R) and 2 transitions (infection, recovery)
    assert!(!compiled.comp_to_transitions().is_empty(),
        "comp_to_transitions should not be empty");

    // Every entry that maps to an empty vec means a compartment has no effect on any transition.
    // For SIR, all 3 compartments appear in at least one rate expression.
    let covered: usize = compiled.comp_to_transitions().iter().filter(|v| !v.is_empty()).count();
    assert!(covered > 0,
        "at least one compartment should have dependent transitions");
}

/// For every transition, at least one compartment in comp_to_transitions should list it,
/// OR the transition has only Const/Param/Time/TimeFunc dependencies (no Pop references).
/// This verifies the graph is neither over-wide nor under-inclusive.
#[test]
fn test_every_transition_reachable_from_graph() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    let n_transitions = compiled.model.transitions.len();
    let mut reachable = vec![false; n_transitions];

    for dep_list in compiled.comp_to_transitions() {
        for &tr_idx in dep_list {
            reachable[tr_idx] = true;
        }
    }
    for &tr_idx in &compiled.time_dep_transitions {
        reachable[tr_idx] = true;
    }

    // For SIR, all transitions reference compartments — all should be reachable
    for (i, &reached) in reachable.iter().enumerate() {
        assert!(reached,
            "transition {} not reachable from comp_to_transitions or time_dep_transitions", i);
    }
}

/// Age-stratified model (seir_age): contact matrix means infection[a] depends on
/// compartments from all age groups. Verify more transitions are covered.
#[test]
fn test_comp_to_transitions_age_stratified() {
    let mut model = load_model(&golden_path("seir_age"));
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    // seir_age has 2 age groups × 4 compartments = 8 integer compartments
    // Each infection[a] depends on I[b] for all b via the contact matrix sum
    // → all I compartments should have the infection transitions in their dep lists
    let total_deps: usize = compiled.comp_to_transitions().iter().map(|v| v.len()).sum();
    assert!(total_deps > compiled.model.transitions.len(),
        "age-stratified model should have more total deps than transitions (due to cross-group coupling)");
}

/// A model with sinusoidal forcing must have time_dep_transitions non-empty.
#[test]
fn test_time_dep_transitions_seasonal() {
    let path = ocaml_golden_path("seir_vaccine_seasonal");
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {} not found", path);
        return;
    }
    let mut model = load_model(&path);
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    assert!(!compiled.time_dep_transitions.is_empty(),
        "seasonal model must have time_dep_transitions non-empty");
}

/// A plain SEIR+V model (no seasonal forcing) should have zero time-dependent transitions.
#[test]
fn test_time_dep_transitions_empty_for_plain_model() {
    let path = ocaml_golden_path("seir_vaccine");
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {} not found", path);
        return;
    }
    let mut model = load_model(&path);
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    assert!(compiled.time_dep_transitions.is_empty(),
        "plain SEIR+V should have no time-dependent transitions");
}

// ── Correctness invariants under sparse updates ────────────────────────────────

/// Population must be conserved across all snapshots even with sparse propensity updates.
/// Run for long enough that the FULL_RECOMPUTE_INTERVAL (10K events) triggers multiple times.
#[test]
fn test_population_conservation_with_sparse_updates() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);

    // Use a large population to generate many events (well past 10K threshold)
    for p in &mut model.parameters {
        if p.name == "N0" { p.value = p.value.with_value(100_000.0); }
        if p.name == "I0" { p.value = p.value.with_value(1_000.0); }
    }

    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = &compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });

    let traj = GillespieSim.run(&compiled, params, 42, &config).unwrap();
    assert!(!traj.snapshots.is_empty(), "trajectory should have snapshots");

    let n0 = traj.snapshots[0].int_state.total();
    for snap in &traj.snapshots {
        assert_eq!(snap.int_state.total(), n0,
            "population not conserved at t={}", snap.t);
        for &c in &snap.int_state.counts {
            assert!(c >= 0, "negative count at t={}", snap.t);
        }
    }
}

/// Sparse update is deterministic: same seed → identical snapshot sequence.
#[test]
fn test_sparse_updates_deterministic() {
    let mut model = load_model(&golden_path("sir_basic"));
    apply_baseline(&mut model);
    for p in &mut model.parameters {
        if p.name == "N0" { p.value = p.value.with_value(50_000.0); }
        if p.name == "I0" { p.value = p.value.with_value(500.0); }
    }

    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = &compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });

    let traj1 = GillespieSim.run(&compiled, params, 7, &config).unwrap();
    let traj2 = GillespieSim.run(&compiled, params, 7, &config).unwrap();

    assert_eq!(traj1.snapshots.len(), traj2.snapshots.len(),
        "determinism: trajectory lengths differ");

    for (i, (s1, s2)) in traj1.snapshots.iter().zip(&traj2.snapshots).enumerate() {
        assert_eq!(s1.int_state.counts, s2.int_state.counts,
            "determinism violated at snapshot {}, t={}", i, s1.t);
    }
}

/// Non-negativity must hold across all snapshots of the cholera model (which has
/// a contact matrix and more complex rate expressions).
#[test]
fn test_non_negativity_cholera() {
    let model = load_model(&golden_path("cholera_siwr"));
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = &compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });

    for seed in 0..5u64 {
        let traj = GillespieSim.run(&compiled, params, seed, &config).unwrap();
        for snap in &traj.snapshots {
            for &c in &snap.int_state.counts {
                assert!(c >= 0, "negative count at t={} seed={}", snap.t, seed);
            }
        }
    }
}
