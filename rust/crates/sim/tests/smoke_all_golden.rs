//! Smoke test: load every *.ir.json in ocaml/golden/, run all forward backends,
//! assert basic invariants. No camdlc dependency — Rust-only.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

fn ocaml_golden_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("../../../ocaml/golden")
}

fn discover_models() -> Vec<String> {
    let dir = ocaml_golden_dir();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {:?}: {}", dir, e))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            name.strip_suffix(".ir.json").map(|s| s.to_owned())
        })
        .collect();
    names.sort();
    names
}

fn load_and_apply_baseline(name: &str) -> ir::Model {
    let path = ocaml_golden_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    let mut model: ir::Model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e));

    if let Some(preset) = model.presets.first().cloned() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    model
}

fn check_invariants(label: &str, traj: &sim::state::Trajectory) {
    assert!(!traj.snapshots.is_empty(), "{}: empty trajectory", label);
    for snap in &traj.snapshots {
        for &c in &snap.int_state.counts {
            assert!(c >= 0, "{}: negative int compartment at t={}", label, snap.t);
        }
        for &v in &snap.real_state.values {
            assert!(v >= 0.0, "{}: negative real compartment at t={}", label, snap.t);
        }
    }
}

#[test]
fn test_smoke_all_ocaml_golden() {
    // gh#audit-C6 / S1: existing goldens (e.g. sir_five_age) include
    // rate expressions that divide by an empty stratum's population
    // at t=0, relying on the legacy silent-zero. Smoke test asserts
    // simulator round-trip / invariants, not numerical-collapse
    // semantics — opt into legacy mode here. Models that want the
    // new strict-mode behaviour should add explicit Cond guards
    // (e.g. `cond(N > 0, I/N, 0)`); the audit's S2 cleanup will
    // sweep production goldens for these patterns over time.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let models = discover_models();
    assert!(!models.is_empty(), "no *.ir.json files found in ocaml/golden/");

    for name in &models {
        let mut model = load_and_apply_baseline(name);
        model.interventions.clear(); // baseline: no interventions

        let compiled = CompiledModel::new(model.clone())
            .unwrap_or_else(|e| panic!("{}: compile error: {:?}", name, e));
        let params = compiled.default_params.clone();
        let t_start = model.simulation.t_start;
        let t_end = model.simulation.t_end.min(30.0);

        let backends: &[(&str, SimConfig)] = &[
            (
                "gillespie",
                SimConfig::Gillespie(GillespieConfig {
                    t_start,
                    t_end,
                    output_dt: None,
                }),
            ),
            (
                "chain_binomial",
                SimConfig::ChainBinomial(ChainBinomialConfig {
                    t_start,
                    t_end,
                    dt: 1.0,
                }),
            ),
        ];

        let required = compiled.required_capabilities();
        for (backend, config) in backends {
            let label = format!("{}/{}", name, backend);
            let sim: &dyn Simulate = match *backend {
                "gillespie" => &GillespieSim,
                _ => &ChainBinomialSim,
            };
            // Skip backends that don't support the model's required capabilities
            if !(required - sim.capabilities()).is_empty() {
                continue;
            }
            // gh#121: a multi-source stochastic transition (`A + B --> C`) is
            // rejected on chain_binomial (bounded by only the first source).
            // That is a structural model property, not a Capabilities bitflag,
            // so skip it explicitly here (the dedicated rejection is asserted in
            // multi_source_transition.rs). gillespie applies it correctly.
            if *backend == "chain_binomial"
                && compiled.validate_single_source_transitions().is_err()
            {
                continue;
            }
            let traj = sim.run(&compiled, &params, 42, config)
                .unwrap_or_else(|e| panic!("{}: sim error: {:?}", label, e));

            check_invariants(&label, &traj);
        }
    }
}
