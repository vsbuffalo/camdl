//! A forcing coefficient that references a parameter must be evaluated LIVE
//! against the params slice — exactly as a rate's `Expr::Param` is.
//!
//! The inference inner loop builds the `CompiledModel` ONCE and varies a
//! borrowed params slice; a coefficient frozen at construction makes the
//! likelihood flat in that parameter (silent garbage posterior). See
//! `docs/dev/incidents/2026-06-09-forcing-coefficient-param-frozen-at-construction.md`
//! and `docs/dev/proposals/2026-06-09-const-parametric-forcing.md` §5.
//!
//! Each test builds the model once, then varies a single forcing-coefficient
//! parameter in the live slice and asserts the trajectory responds. Against the
//! frozen (pre-fix) code these FAIL (byte-identical trajectory). A live-rate
//! parameter is varied as a control to prove the harness produces dynamics that
//! are actually sensitive — so a passing freeze assertion is non-vacuous.

use std::path::Path;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim,
};

fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ir/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

fn load_model(name: &str) -> ir::Model {
    let path = golden_path(name);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("could not read {}", path));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

/// Estimated/Required params carry no concrete value, so the model cannot build
/// without the fit/CLI layer. Seed every parameter with a concrete value here.
fn seed_params(model: &mut ir::Model, values: &[(&str, f64)]) {
    for (name, v) in values {
        let p = model.parameters.iter_mut()
            .find(|p| p.name == *name)
            .unwrap_or_else(|| panic!("model has no parameter '{}'", name));
        p.value = p.value.with_value(*v);
    }
    // Every parameter must resolve to a value or CompiledModel::new errors.
    for p in &model.parameters {
        assert!(p.value.resolved_value().is_some(),
            "parameter '{}' was not seeded a value", p.name);
    }
}

/// Flatten every integer compartment count across every snapshot — a strong
/// whole-trajectory signal (not just final counts).
fn all_counts(traj: &Trajectory) -> Vec<i64> {
    traj.snapshots.iter()
        .flat_map(|s| s.int_state.counts.iter().copied())
        .collect()
}

fn chain_binomial(compiled: &CompiledModel, params: &[f64], t_end: f64) -> Trajectory {
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end, dt: 1.0, binomial: sim::rng::BinomialAlgorithm::Btpe });
    ChainBinomialSim.run(compiled, params, 7, &cfg)
        .expect("chain-binomial run failed")
}

/// Sinusoidal amplitude is an **estimated** parameter (`seir_seasonal_patch`,
/// `amp_urban`). This is the exact incident reproduction.
#[test]
fn estimated_sinusoidal_amplitude_is_live() {
    let mut model = load_model("seir_seasonal_patch");
    seed_params(&mut model, &[
        ("sigma", 0.3), ("gamma", 0.2), ("baseline", 0.4),
        ("amp_urban", 0.4), ("amp_rural", 0.4),
        ("N0_urban", 100_000.0), ("N0_rural", 50_000.0), ("I0", 10.0),
    ]);
    let compiled = CompiledModel::new(model).unwrap();
    let t_end = 120.0;

    // Build once; vary ONLY the live slice — exactly what inference does.
    let p_lo = compiled.default_params.clone();
    let mut p_hi = p_lo.clone();
    p_hi[compiled.param_index["amp_urban"]] = 0.9;

    let lo = all_counts(&chain_binomial(&compiled, &p_lo, t_end));
    let hi = all_counts(&chain_binomial(&compiled, &p_hi, t_end));

    // Control: a live RATE parameter (sigma) must change the trajectory —
    // proves the model's dynamics are sensitive, so the assertion below is
    // testing the freeze and not a degenerate (always-identical) model.
    let mut p_ctrl = p_lo.clone();
    p_ctrl[compiled.param_index["sigma"]] = 0.6;
    let ctrl = all_counts(&chain_binomial(&compiled, &p_ctrl, t_end));
    assert_ne!(lo, ctrl, "control: varying live rate param `sigma` must change the trajectory");

    assert_ne!(lo, hi,
        "varying forcing-coefficient param `amp_urban` (0.4 → 0.9) in the live \
         slice must change the trajectory; identical means the coefficient is \
         frozen at construction");
}

/// Sinusoidal amplitude is a **required** parameter (`seir_vaccine_seasonal`,
/// `alpha`). §5 asks for a Required case, not only the Estimated golden.
#[test]
fn required_sinusoidal_amplitude_is_live() {
    let mut model = load_model("seir_vaccine_seasonal");
    seed_params(&mut model, &[
        ("beta", 0.5), ("sigma", 0.3), ("gamma", 0.2), ("omega", 0.01),
        ("reversion_rate", 0.01), ("alpha", 0.3), ("phi_season", 0.0),
        ("vacc_frac", 0.0), ("N0", 100_000.0), ("I0", 10.0),
    ]);
    let compiled = CompiledModel::new(model).unwrap();
    let t_end = 120.0;

    let p_lo = compiled.default_params.clone();
    let mut p_hi = p_lo.clone();
    p_hi[compiled.param_index["alpha"]] = 0.8;

    let lo = all_counts(&chain_binomial(&compiled, &p_lo, t_end));
    let hi = all_counts(&chain_binomial(&compiled, &p_hi, t_end));

    // Control: live rate param `beta`.
    let mut p_ctrl = p_lo.clone();
    p_ctrl[compiled.param_index["beta"]] = 0.7;
    let ctrl = all_counts(&chain_binomial(&compiled, &p_ctrl, t_end));
    assert_ne!(lo, ctrl, "control: varying live rate param `beta` must change the trajectory");

    assert_ne!(lo, hi,
        "varying forcing-coefficient param `alpha` (0.3 → 0.8) in the live \
         slice must change the trajectory; identical means the coefficient is \
         frozen at construction");
}
