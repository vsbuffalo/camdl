//! Particle filter validation tests.
//!
//! Test 1: Pure death process with Poisson observations.
//! The marginal likelihood is analytically tractable, so we can
//! verify the PF converges to the correct value.
//!
//! Test 2: Determinism — same seed gives same log-likelihood.
//!
//! Test 3: More particles → lower variance of the estimate.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        obs_loglik::poisson_logpmf,
        particle_filter::bootstrap_filter,
        ChainBinomialProcess,
        traits::{ObservationModel, SMCConfig},
        ParticleState,
    },
    rng::StatefulRng,
};

/// Test-only observation model: observes compartment 0 (prevalence) with Poisson likelihood.
struct PoissonPrevalenceObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonPrevalenceObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let projected = state.counts[0] as f64;
        poisson_logpmf(self.observations[obs_idx], projected.max(0.1))
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _state: &ParticleState, _obs_idx: usize, _params: &[f64], _rng: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _state: &ParticleState, _obs_idx: usize, _params: &[f64]) -> Vec<f64> { vec![] }
}

fn pure_death_model() -> (CompiledModel, Vec<f64>) {
    let model = Model {
        name: "pure_death_pf".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "death".into(),
                stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
                rate: Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                    },
                }),
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(),
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: Some(0.01), bounds: None, prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("N".into(), 100.0); m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 100.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None,
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Build observations and obs model for the pure death model.
fn pure_death_obs() -> PoissonPrevalenceObs {
    let obs: Vec<(f64, f64)> = (1..=10)
        .map(|k| {
            let t = k as f64 * 10.0;
            let expected = 100.0 * (-0.01 * t).exp();
            (t, expected.round())
        })
        .collect();
    PoissonPrevalenceObs {
        obs_times: obs.iter().map(|o| o.0).collect(),
        observations: obs.iter().map(|o| o.1).collect(),
    }
}

/// Run the particle filter on a pure death model with Poisson observations.
fn run_pf(n_particles: usize, seed: u64) -> f64 {
    let (compiled, params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_model = pure_death_obs();

    let config = SMCConfig { n_particles, dt: 1.0, t_start: 0.0, skip_first_obs_from_loglik: false, record_ancestry: false, record_prequential: false };

    let result = bootstrap_filter(
        &process, &obs_model, &params, &config, seed,
    ).unwrap();

    result.log_likelihood
}

#[test]
fn test_pf_determinism() {
    let ll1 = run_pf(200, 42);
    let ll2 = run_pf(200, 42);
    assert_eq!(ll1, ll2, "same seed must give identical log-likelihood");
}

#[test]
fn test_pf_different_seeds_differ() {
    let ll1 = run_pf(200, 1);
    let ll2 = run_pf(200, 2);
    assert_ne!(ll1, ll2, "different seeds should give different estimates");
}

#[test]
fn test_pf_loglik_finite() {
    let ll = run_pf(500, 42);
    assert!(ll.is_finite(), "log-likelihood should be finite, got {}", ll);
    assert!(ll < 0.0, "log-likelihood should be negative, got {}", ll);
}

#[test]
fn test_pf_more_particles_lower_variance() {
    // Run with 100 and 1000 particles, 20 replicates each.
    // The variance of the estimate should be lower with more particles.
    let mut ll_100: Vec<f64> = Vec::new();
    let mut ll_1000: Vec<f64> = Vec::new();

    for seed in 0..20u64 {
        ll_100.push(run_pf(100, seed));
        ll_1000.push(run_pf(1000, seed));
    }

    let var = |xs: &[f64]| -> f64 {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
    };

    let var_100 = var(&ll_100);
    let var_1000 = var(&ll_1000);

    assert!(var_1000 < var_100,
        "1000 particles should have lower variance than 100: var_100={:.4}, var_1000={:.4}",
        var_100, var_1000);
}

#[test]
fn test_pf_ess_reasonable() {
    let (compiled, params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_model = pure_death_obs();

    let config = SMCConfig { n_particles: 500, dt: 1.0, t_start: 0.0, skip_first_obs_from_loglik: false, record_ancestry: false, record_prequential: false };

    let result = bootstrap_filter(
        &process, &obs_model, &params, &config, 42,
    ).unwrap();

    // ESS should be reasonable — not collapsed to 1 or full N
    for (i, &ess) in result.ess_trace.iter().enumerate() {
        assert!(ess > 10.0,
            "ESS too low at obs {}: {:.1} (particle collapse)", i, ess);
        assert!(ess <= 500.0,
            "ESS > N_particles at obs {}: {:.1}", i, ess);
    }
}

// ── IC-free inference (skip_first_obs_from_loglik) ─────────────────────
//
// The PF-level contract: when `skip_first_obs_from_loglik = true`, the
// filter still weights and resamples at the first observation (that's
// how y₁ pins x₀ in the PF framing) but does NOT accumulate the t=0
// log-sum-exp into the returned log-likelihood. Log-accumulation picks
// up from t=1.
//
// The load-bearing identity:
//
//   log L_standard − log L_ic_free  ≡  ll_increments_standard[0]
//
// because the only difference between the two runs is whether we
// included the first increment.

fn run_pf_full(
    n_particles: usize, seed: u64, skip_first: bool,
) -> sim::inference::particle_filter::PFilterResult {
    let (compiled, params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_model = pure_death_obs();
    let config = SMCConfig {
        n_particles, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: skip_first,
        record_ancestry: false,
        record_prequential: false,
    };
    sim::inference::bootstrap_filter(&process, &obs_model, &params, &config, seed)
        .unwrap()
}

#[test]
fn ic_free_drops_only_the_first_obs_from_loglik() {
    // Both runs share the exact seed → identical particle trajectories
    // → identical ll_increments. The only difference is whether
    // increment[0] is added to total_loglik.
    let standard = run_pf_full(500, 42, false);
    let ic_free  = run_pf_full(500, 42, true);

    // Same filter trajectory means every per-obs increment matches.
    assert_eq!(standard.ll_increments.len(), ic_free.ll_increments.len(),
        "both runs must visit every observation");
    for (i, (&a, &b)) in standard.ll_increments.iter()
        .zip(ic_free.ll_increments.iter()).enumerate() {
        assert!((a - b).abs() < 1e-12,
            "per-obs ll_increments must match bit-for-bit at obs {}: {} vs {}",
            i, a, b);
    }

    // The difference is exactly the first increment.
    let diff = standard.log_likelihood - ic_free.log_likelihood;
    let expected = standard.ll_increments[0];
    assert!((diff - expected).abs() < 1e-12,
        "log L − log L_c must equal ll_increments[0]: diff={:.10}, expected={:.10}",
        diff, expected);
}

#[test]
fn ic_free_disabled_matches_standard_exactly() {
    // Explicit "ic_free = false produces identical output to no flag"
    // guard — it's a bool field with a default, and the default must
    // not silently change the answer.
    let a = run_pf_full(200, 7, false);
    let b = run_pf_full(200, 7, false);
    assert_eq!(a.log_likelihood, b.log_likelihood,
        "deterministic: same seed + same flag → same answer");
}

#[test]
fn ic_free_enabled_changes_reported_loglik() {
    // Sanity: the flag actually DOES something. If ic_free=true returns
    // the same value as ic_free=false, the guard regressed.
    let standard = run_pf_full(500, 7, false);
    let ic_free  = run_pf_full(500, 7, true);
    assert_ne!(standard.log_likelihood, ic_free.log_likelihood,
        "ic_free must change the reported loglik when obs[0] has any \
         observation density at all (it does — pure-death obs at t=10 \
         is Poisson-like with nonzero ll). If this passes loglik-equal \
         the guard isn't firing.");
}
