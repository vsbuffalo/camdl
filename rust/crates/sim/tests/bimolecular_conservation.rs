//! Multi-source transition tests (wave 1 / malaria #1).
//!
//! For the bimolecular reaction `A + B --> C @ k * A * B / N`, a correct
//! firing atomically decrements A and B together and increments C. Gillespie
//! applies each firing as one atomic CTMC event, so it holds the invariants
//!
//!   A(t) + C(t) = A(0)   (every A consumed became a C)
//!   B(t) + C(t) = B(0)   (every B consumed became a C)
//!   A(0) - A(t) = B(0) - B(t)   (co-decrement — the atomicity invariant)
//!
//! chain_binomial bounds a transition's drawn flow by only its FIRST source, so
//! a multi-source transition can drive a secondary source negative — it is
//! rejected up front (gh#121; see the rejection test below and
//! `multi_source_transition.rs`).

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

fn load_bimolecular() -> (ir::Model, CompiledModel) {
    let contents = std::fs::read_to_string(golden_path("bimolecular"))
        .expect("read bimolecular.ir.json");
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

fn assert_bimolecular_invariants<F>(
    compiled: &CompiledModel,
    params: &[f64],
    run_seed: F,
    backend: &str,
) where F: Fn(u64) -> sim::Trajectory {
    let idx_a = local_idx(compiled, "A");
    let idx_b = local_idx(compiled, "B");
    let idx_c = local_idx(compiled, "C");
    let _ = params;

    for seed in 0..10u64 {
        let traj = run_seed(seed);
        let a0 = traj.snapshots[0].int_state.counts[idx_a];
        let b0 = traj.snapshots[0].int_state.counts[idx_b];
        for snap in &traj.snapshots {
            let a = snap.int_state.counts[idx_a];
            let b = snap.int_state.counts[idx_b];
            let c = snap.int_state.counts[idx_c];
            assert_eq!(a + c, a0,
                "{}: A + C drift at t={} seed={}: {} != {}",
                backend, snap.t, seed, a + c, a0);
            assert_eq!(b + c, b0,
                "{}: B + C drift at t={} seed={}: {} != {}",
                backend, snap.t, seed, b + c, b0);
            assert_eq!(a0 - a, b0 - b,
                "{}: A and B not co-decremented at t={} seed={}: ΔA={} ΔB={}",
                backend, snap.t, seed, a0 - a, b0 - b);
        }
    }
}

#[test]
fn test_bimolecular_gillespie_conservation() {
    let (model, compiled) = load_bimolecular();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        output_dt: None,
    });
    assert_bimolecular_invariants(&compiled, &params,
        |seed| GillespieSim.run(&compiled, &params, seed, &config).unwrap(),
        "gillespie");
}

/// gh#121: the bimolecular reaction `A + B --> C` is a MULTI-SOURCE stochastic
/// transition — chain_binomial bounds the drawn flow by only the first source
/// (`A`), so the secondary source `B` can be driven negative (silently in a mild
/// regime; here as a runtime `NegativeCount` once the flow exceeds `B`). It is
/// therefore rejected up front on chain_binomial with a located gh#121 error,
/// while gillespie/ode (above) run it correctly with atomic co-decrement. The
/// dedicated rejection is also asserted in `multi_source_transition.rs`.
#[test]
fn test_bimolecular_chain_binomial_rejected() {
    let (model, compiled) = load_bimolecular();
    let params = compiled.default_params.clone();

    // Structural validation rejects the multi-source transition with a located
    // gh#121 message naming the transition and its two sources.
    let err = compiled
        .validate_single_source_transitions()
        .expect_err("bimolecular is multi-source and must be rejected on chain_binomial");
    let msg = err.to_string();
    assert!(msg.contains("gh#121"), "message must cite the issue: {msg}");

    // Forward chain_binomial hard-errors with that same gh#121 message rather
    // than over-drawing `B` into a cryptic NegativeCount.
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 1.0,
    });
    let res = ChainBinomialSim.run(&compiled, &params, 0, &config);
    assert!(res.is_err(), "chain_binomial must reject the multi-source bimolecular model");
    assert!(res.unwrap_err().to_string().contains("gh#121"));
}
