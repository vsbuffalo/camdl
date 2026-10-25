//! PMMH validation tests.
//!
//! Test 1: Pure death model — posterior covers true parameter value.
//! Test 2: Determinism — same seed gives same chain.
//! Test 3: Acceptance rate in reasonable range.
//! Test 4: Flat prior recovers near-MLE.
//! Test 5: Adaptive proposal improves acceptance from bad initial proposal.
//! Test 6: ESS computation sanity.

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
    error::SimError,
    inference::{
        obs_loglik::poisson_logpmf,
        particle_filter::bootstrap_filter,
        if2::{EstimatedParam, Transform},
        pmmh::{run_pmmh, Prior, PMMHConfig, mcmc_ess},
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

/// Build a pure death model: N → ∅ at rate μ*N.
/// Same model used in particle_filter.rs tests.
fn pure_death_model() -> (CompiledModel, Vec<f64>) {
    let model = Model {
        ic_grad: Default::default(),
        name: "pure_death_pmmh".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
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
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::constants({
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
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Synthetic observations from the pure death model at true μ=0.01.
fn pure_death_observations() -> PoissonPrevalenceObs {
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

/// Build the PF-based loglik evaluator for the pure death model.
/// Returns a closure: (full_params, seed) → log L̂(θ).
fn make_eval_loglik(
    compiled: Arc<CompiledModel>,
    n_particles: usize,
) -> impl Fn(&[f64], u64) -> Result<f64, SimError> {
    move |params: &[f64], pf_seed: u64| -> Result<f64, SimError> {
        let process = ChainBinomialProcess::new(compiled.clone());
        let obs_model = pure_death_observations();
        let config = SMCConfig { n_particles, dt: 1.0, t_start: 0.0, skip_first_obs_from_loglik: false, record_ancestry: false, record_prequential: false, record_predictions: false, max_substeps: sim::inference::degeneracy::ITER_BUDGET };

        // gh#224 classification: a per-θ excursion or a degenerate filter
        // is a legitimate "θ ruled out" (-∞); only a structural error
        // (model/config can't run) surfaces.
        match bootstrap_filter(&process, &obs_model, params, &config, pf_seed) {
            Ok(r) => Ok(r.log_likelihood),
            Err(e) if e.is_structural() => Err(e),
            Err(_) => Ok(f64::NEG_INFINITY),
        }
    }
}

/// EstimatedParam spec for the death rate μ (log-transformed, positive).
fn mu_param() -> EstimatedParam {
    EstimatedParam {
        name: "mu".into(),
        index: 0, // μ is the only parameter
        initial: 0.01,
        rw_sd: 0.002,
        transform: Transform::Log { lo: 1e-6, hi: 1.0 },
        lower: 1e-6,
        upper: 1.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

#[test]
fn test_pmmh_posterior_covers_truth() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 200;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Normal { mean: 0.01, sd: 0.01 })];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 3000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2], // on log scale
        adapt: true,
        adapt_start: 200,
        // The whole run is warm-up: these tests are about the adaptation
        // itself, so freezing it partway would only shorten what they measure.
        adapt_stop: 3000,
        thin: 1,
        burn_in: 500,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();

    // Extract μ samples (index 0 in param vector)
    let mu_samples: Vec<f64> = result.steps.iter().map(|s| s.params[0]).collect();
    assert!(!mu_samples.is_empty(), "should have post-burn-in samples");

    let mean_mu = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let mut sorted = mu_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q025 = sorted[sorted.len() * 25 / 1000];
    let q975 = sorted[sorted.len() * 975 / 1000];

    // True μ = 0.01 should be within the 95% CI
    assert!(q025 < 0.01 && 0.01 < q975,
        "true μ=0.01 should be in 95% CI [{:.5}, {:.5}], mean={:.5}",
        q025, q975, mean_mu);

    // Mean should be within 50% of truth
    assert!((mean_mu - 0.01).abs() < 0.005,
        "posterior mean {:.5} should be close to true μ=0.01", mean_mu);
}

#[test]
fn test_pmmh_determinism() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 100;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 100,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0,
    };

    let r1 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();
    let r2 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();

    assert_eq!(r1.steps.len(), r2.steps.len());
    for (s1, s2) in r1.steps.iter().zip(r2.steps.iter()) {
        assert_eq!(s1.accepted, s2.accepted, "acceptance mismatch at step {}", s1.step);
        assert_eq!(s1.log_likelihood, s2.log_likelihood, "loglik mismatch at step {}", s1.step);
        assert_eq!(s1.params, s2.params, "params mismatch at step {}", s1.step);
    }
}

#[test]
fn test_pmmh_acceptance_rate() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 200;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 1000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: true,
        adapt_start: 200,
        // The whole run is warm-up: these tests are about the adaptation
        // itself, so freezing it partway would only shorten what they measure.
        adapt_stop: 1000,
        thin: 1,
        burn_in: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();

    assert!(result.acceptance_rate > 0.05,
        "acceptance rate {:.3} too low (chain stuck)", result.acceptance_rate);
    assert!(result.acceptance_rate < 0.80,
        "acceptance rate {:.3} too high (proposals too narrow)", result.acceptance_rate);
}

#[test]
fn test_pmmh_flat_prior_finds_near_mle() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 200;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 2000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: true,
        adapt_start: 200,
        // The whole run is warm-up: these tests are about the adaptation
        // itself, so freezing it partway would only shorten what they measure.
        adapt_stop: 2000,
        thin: 1,
        burn_in: 500,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();

    // MAP should be close to true μ=0.01
    let map_mu = result.map_params[0];
    assert!((map_mu - 0.01).abs() < 0.005,
        "MAP estimate {:.5} should be close to true μ=0.01", map_mu);
}

#[test]
fn test_pmmh_adaptive_improves_acceptance() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 200;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base_params = compiled.default_params.clone();

    // Deliberately bad initial proposal: 10× too wide
    let config = PMMHConfig {
        n_steps: 1500,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![2.0], // way too wide on log scale
        adapt: true,
        adapt_start: 200,
        // The whole run is warm-up: these tests are about the adaptation
        // itself, so freezing it partway would only shorten what they measure.
        adapt_stop: 1500,
        thin: 1,
        burn_in: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 42, None, None, String::new()).unwrap();

    // Compute acceptance rate in the second half (after adaptation kicks in)
    let half = result.steps.len() / 2;
    let late_accepted = result.steps[half..].iter().filter(|s| s.accepted).count();
    let late_rate = late_accepted as f64 / (result.steps.len() - half) as f64;

    // Early acceptance rate should be terrible, late should be better
    let early_accepted = result.steps[..half].iter().filter(|s| s.accepted).count();
    let early_rate = early_accepted as f64 / half as f64;

    // With adaptation, late rate should improve (or at least not be zero)
    assert!(late_rate > 0.05,
        "late acceptance rate {:.3} should recover with adaptation (early: {:.3})",
        late_rate, early_rate);
}

#[test]
fn test_mcmc_ess_sanity() {
    // Constant chain: ESS should be 1
    let constant = vec![5.0; 100];
    let ess_const = mcmc_ess(&constant);
    assert!((ess_const - 1.0).abs() < 0.1,
        "ESS of constant chain should be ~1, got {:.1}", ess_const);

    // IID chain: ESS should be close to N
    // Use a simple deterministic "IID-like" sequence
    let n = 1000;
    let iid: Vec<f64> = (0..n).map(|i| {
        // Deterministic but uncorrelated-looking sequence
        ((i as f64 * 0.618033988749895) % 1.0) * 10.0
    }).collect();
    let ess_iid = mcmc_ess(&iid);
    assert!(ess_iid > n as f64 * 0.5,
        "ESS of IID-like chain should be > N/2, got {:.0} (N={})", ess_iid, n);
}

#[test]
fn test_pmmh_different_seeds_differ() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 100;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 50,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0,
    };

    let r1 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 1, None, None, String::new()).unwrap();
    let r2 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &eval_loglik, 2, None, None, String::new()).unwrap();

    // At least some steps should differ
    let any_differ = r1.steps.iter().zip(r2.steps.iter())
        .any(|(s1, s2)| s1.params != s2.params || s1.accepted != s2.accepted);
    assert!(any_differ, "different seeds should produce different chains");
}

/// gh#224. A structural (non-recoverable) error from the likelihood
/// evaluator must propagate out of `run_pmmh` as `Err`, not be silently
/// mistaken for a ruled-out θ. Before the typed-channel fix the eval
/// closures collapsed *every* error to −∞, so a model that cannot run
/// produced a degenerate posterior with a successful (exit-0) return.
#[test]
fn pmmh_surfaces_structural_eval_error() {
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Normal { mean: 0.01, sd: 0.01 })];
    let base_params = vec![0.01];
    let config = PMMHConfig {
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, adapt_stop: 0, thin: 1, burn_in: 0,
    };
    // Evaluator that always fails with a non-recoverable error.
    let eval = |_params: &[f64], _seed: u64| -> Result<f64, SimError> {
        Err(SimError::Validation("structural: model cannot run".into()))
    };
    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &config,
        &eval, 42, None, None, String::new(),
    );
    assert!(
        matches!(result, Err(SimError::Validation(_))),
        "a structural eval error must surface as Err, got {:?}",
        result.map(|_| "Ok"),
    );
}

/// gh#224 companion: `Ok(−∞)` is a *legitimate* loglik ("θ ruled out")
/// that PMMH handles via the MH reject path — it must NOT be promoted to
/// an error. The chain runs to completion and returns `Ok`.
#[test]
fn pmmh_tolerates_ruled_out_neg_inf() {
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Normal { mean: 0.01, sd: 0.01 })];
    let base_params = vec![0.01];
    let config = PMMHConfig {
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, adapt_stop: 0, thin: 1, burn_in: 0,
    };
    // A recoverable per-particle excursion legitimately yields −∞; the
    // closure has already classified it as Ok. PMMH must accept it.
    let eval = |_params: &[f64], _seed: u64| -> Result<f64, SimError> {
        Ok(f64::NEG_INFINITY)
    };
    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &config,
        &eval, 42, None, None, String::new(),
    );
    assert!(result.is_ok(), "Ok(−∞) is a valid ruled-out loglik, must not error");
}
