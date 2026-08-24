//! Statistical validation that `consecutive()` Erlang-staging in camdl
//! actually produces an Erlang-distributed latent period.
//!
//! Closes audit gap P1.3 from
//! `docs/dev/reviews/2026-04-21-spec-claims-vs-tests.md`. Spec §14
//! describes `consecutive((s, s_next) in consecutive(dim))` as
//! producing an Erlang-k latent distribution when combined with rate
//! `k·σ` on each sub-transition. The compiler-level test
//! (test_compiler.ml::test_consecutive_pair_count) verifies the IR
//! has k-1 transitions. This file verifies the *runtime dynamics*
//! actually match the analytical Erlang survival function.
//!
//! Design pattern (useful for other regression tests of this class):
//!   1. Load an existing golden with the structure we want to test.
//!   2. Mutate `model.initial_conditions` + `model.parameters` in
//!      Rust to zero out coupled dynamics (here: infection + recovery)
//!      so the measurement isolates the phenomenon (E-chain decay).
//!   3. Run N seeds with Gillespie (exact CTMC → ground truth).
//!   4. At selected time points, assert the mean across seeds matches
//!      the analytical prediction within a tolerance scaled by N.
//!
//! Why this is slow (#[ignore]): 5000 seeds × 15-day simulation ≈ 10 s
//! locally. Opt-in for nightly / manual verification.
//!
//! Reference values generated with scipy.stats.gamma:
//!   from scipy.stats import gamma
//!   # P(T > t) where T ~ Erlang(k, rate = k·σ)
//!   surv = 1.0 - gamma.cdf(t, a=k, scale=1.0/(k*sigma))

use ir::model::InitialConditions;
use sim::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    simulate::Simulate,
    GillespieSim,
};
use std::collections::HashMap;

fn load_golden(name: &str) -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = std::path::PathBuf::from(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("could not read {:?}", path));
    ir::from_str(&contents).unwrap()
}

/// Set up the seir_erlang golden for a clean Erlang-decay measurement:
///   - Zero out S and I so infection and recovery can't fire.
///   - Load E[e1] with n0 individuals; e2, e3 empty.
///   - Set σ = 1/5 = 0.2 day⁻¹ → mean latent period = 1/σ = 5 days.
///   - Keep β = 0 (no transmission) and γ = 0 (no recovery).
fn setup_pure_erlang_decay(n0: i64) -> ir::Model {
    let mut model = load_golden("seir_erlang");

    // Force explicit integer initial conditions.
    let mut init = HashMap::new();
    init.insert("S".to_string(), 0.0);
    init.insert("E_e1".to_string(), n0 as f64);
    init.insert("E_e2".to_string(), 0.0);
    init.insert("E_e3".to_string(), 0.0);
    init.insert("I".to_string(), 0.0);
    init.insert("R".to_string(), 0.0);
    model.initial_conditions = InitialConditions::constants(init);

    // Override params: β = 0 (no infections possible from S=0 anyway,
    // but makes the setup self-evident), σ = 0.2 (5-day mean latent),
    // γ = 0 (no one leaves I ever; we only care about E-side dynamics).
    for p in &mut model.parameters {
        match p.name.as_str() {
            "beta"  => p.value = p.value.with_value(0.0),
            "sigma" => p.value = p.value.with_value(0.2),
            "gamma" => p.value = p.value.with_value(0.0),
            _ => {}
        }
    }

    model
}

/// Scipy-generated reference values: P(T > t) for T ~ Erlang(k=3, rate=3·σ)
/// with σ = 0.2 → mean latent = 5 days.
///
/// Generated via:
///   from scipy.stats import gamma
///   sigma, k = 0.2, 3
///   lam = k * sigma  # = 0.6
///   surv = 1.0 - gamma.cdf(t, a=k, scale=1.0/lam)
fn erlang_3_survival_reference() -> Vec<(f64, f64)> {
    vec![
        ( 2.0, 0.8794870988),
        ( 4.0, 0.5697087467),
        ( 5.0, 0.4231900811),   // at mean
        ( 6.0, 0.3027468447),
        ( 8.0, 0.1425392189),
        (10.0, 0.0619688044),
    ]
}

/// Exponential survival for the same σ — Erlang-3 and exponential
/// share mean but have visibly different shapes. Including this as a
/// sanity check so a regression that collapses k=3 to k=1 would fail.
///   P(T > t) = exp(-σ·t)
fn exponential_survival(t: f64, sigma: f64) -> f64 {
    (-sigma * t).exp()
}

/// Mean of E_total across seeds at simulation time `t_target`. Picks
/// the snapshot closest to t_target. Gillespie output grid is
/// irregular; caller supplies `output_dt` to control density.
fn mean_e_total_at(
    compiled: &CompiledModel,
    params: &[f64],
    t_target: f64,
    t_end: f64,
    n_seeds: u64,
    output_dt: f64,
) -> f64 {
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: 0.0,
        t_end,
        output_dt: Some(output_dt),
    });

    // Compartment indices for E_e1, E_e2, E_e3 — look up by name.
    let e_indices: Vec<usize> = (0..compiled.model.compartments.len())
        .filter(|&i| {
            let name = &compiled.model.compartments[i].name;
            name == "E_e1" || name == "E_e2" || name == "E_e3"
        })
        .collect();
    assert_eq!(e_indices.len(), 3, "expected E_e1, E_e2, E_e3 compartments");

    let mut sum = 0.0;
    for seed in 0..n_seeds {
        let traj = GillespieSim.run(compiled, params, seed, &config).unwrap();
        // Find snapshot closest to t_target.
        let snap = traj.snapshots.iter()
            .min_by(|a, b| {
                (a.t - t_target).abs().partial_cmp(&(b.t - t_target).abs()).unwrap()
            })
            .unwrap();
        let e_total: i64 = e_indices.iter()
            .map(|&i| snap.int_state.counts[i])
            .sum();
        sum += e_total as f64;
    }
    sum / n_seeds as f64
}

#[test]
#[ignore = "statistical test: run with --ignored (takes ~10 s)"]
fn erlang_3_latent_matches_analytical_survival() {
    let n0 = 10_000_i64;
    let n_seeds = 200_u64;   // 200 × 10000 individuals = 2M Bernoulli-like observations
    let model = setup_pure_erlang_decay(n0);
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    // SE of mean is σ_T / √(n0·n_seeds) ≈ sqrt(variance) per seed, avged.
    // var(indicator 1_{T>t}) = p·(1-p); for p~0.5, σ ≈ 0.5.
    // SE per seed's E_total(t) ≈ sqrt(n0 · 0.25) = 50
    // SE of mean across n_seeds = 50 / sqrt(200) ≈ 3.5
    // Use 3σ band: tolerance ≈ 10-15 particles. n0=10000 scales the
    // expected survival by 10000; tolerance in absolute count is ~15.

    for (t, surv) in erlang_3_survival_reference() {
        let expected = surv * n0 as f64;
        let actual   = mean_e_total_at(&compiled, &params, t, 20.0, n_seeds, 0.25);
        let tol = 3.0 * (n0 as f64 * surv * (1.0 - surv) / n_seeds as f64).sqrt() + 5.0;
        assert!(
            (actual - expected).abs() < tol,
            "E_total(t={}): got {:.1}, expected {:.1} (Erlang-3 survival × {}), \
             tolerance {:.1}. If these are roughly 2× off, the model is behaving \
             like exponential (k=1) — `consecutive()` is likely broken.",
            t, actual, expected, n0, tol
        );
    }
}

#[test]
#[ignore = "statistical test: sanity-check that Erlang is distinguishable from Exp"]
fn erlang_3_distinguishably_tighter_than_exponential() {
    // This is the "if consecutive() silently collapsed to exponential,
    // this test would catch it" check. At t = 2 days:
    //   Erlang-3 survival ≈ 0.879
    //   Exponential survival ≈ 0.670
    // These differ by ≈ 21% of the initial count — impossible to
    // explain by Monte Carlo noise at n0=10000, n_seeds=200.
    let n0 = 10_000_i64;
    let n_seeds = 200_u64;
    let model = setup_pure_erlang_decay(n0);
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    let actual = mean_e_total_at(&compiled, &params, 2.0, 10.0, n_seeds, 0.25);
    let erlang_pred = 0.8794870988 * n0 as f64;   // scipy reference
    let exp_pred    = exponential_survival(2.0, 0.2) * n0 as f64;
    let midpoint    = (erlang_pred + exp_pred) / 2.0;

    assert!(
        actual > midpoint,
        "E_total(t=2) = {:.1} is closer to exponential survival ({:.1}) \
         than to Erlang-3 survival ({:.1}). If `consecutive()` is \
         emitting a single transition instead of k sub-transitions, \
         the latent period degenerates to exponential.",
        actual, exp_pred, erlang_pred
    );
}
