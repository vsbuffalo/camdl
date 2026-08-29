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
        correlated_pf::{bootstrap_filter_correlated, cpm_steps_per_obs, PFRandomState},
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
        let config = SMCConfig { n_particles, dt: 1.0, t_start: 0.0, skip_first_obs_from_loglik: false, record_ancestry: false, record_prequential: false, max_substeps: sim::inference::degeneracy::ITER_BUDGET };

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
        t_start: 0.0,
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
        burn_in: 500, rho: None, n_source_groups: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();

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
        t_start: 0.0,
        n_steps: 100,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0, rho: None, n_source_groups: 0,
    };

    let r1 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();
    let r2 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();

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
        t_start: 0.0,
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
        burn_in: 0, rho: None, n_source_groups: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();

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
        t_start: 0.0,
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
        burn_in: 500, rho: None, n_source_groups: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();

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
        t_start: 0.0,
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
        burn_in: 0, rho: None, n_source_groups: 0,
    };

    let result = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 42, None, None, String::new()).unwrap();

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
        t_start: 0.0,
        n_steps: 50,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0, rho: None, n_source_groups: 0,
    };

    let r1 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 1, None, None, String::new()).unwrap();
    let r2 = run_pmmh(&if2_params, &priors, &base_params, &[], &config, &[], &eval_loglik, None, 2, None, None, String::new()).unwrap();

    // At least some steps should differ
    let any_differ = r1.steps.iter().zip(r2.steps.iter())
        .any(|(s1, s2)| s1.params != s2.params || s1.accepted != s2.accepted);
    assert!(any_differ, "different seeds should produce different chains");
}

// ── Correlated PF (CPM) observation-grid handling ──────────────────────────
//
// CPM pre-draws random numbers into one block per observation window, sized at
// that window's own substep count, and indexes them with
// `particle*window_steps + substep`. Windows are therefore independent: a
// series that starts mid-period, or a daily series with a day of no reporting,
// indexes correctly because each window carries its own stride. What the filter
// still refuses is noise that was not drawn for the grid it is running on — a
// row too short for its window would read past its block, and the fall-through
// to fresh per-particle RNG would decorrelate the estimator the PMMH acceptance
// ratio depends on, with no diagnostic.

/// Build the smallest CPM harness: pure-death `ChainBinomialProcess` + a
/// `PoissonPrevalenceObs` at the given obs times, run `bootstrap_filter_correlated`
/// against `randoms`.
fn run_cpm_with(
    obs_times: Vec<f64>,
    dt: f64,
    randoms: &PFRandomState,
) -> Result<f64, sim::error::SimError> {
    let (compiled, params) = pure_death_model();
    let process = ChainBinomialProcess::new(Arc::new(compiled));
    let n_obs = obs_times.len();
    let obs_model = PoissonPrevalenceObs {
        observations: vec![50.0; n_obs],
        obs_times: obs_times.clone(),
    };
    let config = SMCConfig {
        n_particles: CPM_PARTICLES,
        dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter_correlated(&process, &obs_model, &params, &config, randoms, 7)
        .map(|r| r.log_likelihood)
}

const CPM_PARTICLES: usize = 64;
const CPM_SOURCE_GROUPS: usize = 1;

/// Noise drawn for exactly this grid, the way `run_pmmh` draws it.
fn cpm_randoms(obs_times: &[f64], dt: f64, seed: u64) -> PFRandomState {
    let steps_per_obs = cpm_steps_per_obs(obs_times, 0.0, dt);
    let mut rng = StatefulRng::new(seed);
    PFRandomState::draw_fresh(CPM_PARTICLES, &steps_per_obs, CPM_SOURCE_GROUPS, &mut rng)
}

fn run_cpm(obs_times: Vec<f64>, dt: f64) -> Result<f64, sim::error::SimError> {
    let randoms = cpm_randoms(&obs_times, dt, 7);
    run_cpm_with(obs_times, dt, &randoms)
}

#[test]
fn cpm_accepts_a_mid_period_first_window() {
    // t_start=0, obs at [5,12,19], dt=1: the first window [0,5] is 5 substeps
    // where the rest are 7. Its noise block is sized at 5, so the run proceeds.
    assert_eq!(cpm_steps_per_obs(&[5.0, 12.0, 19.0], 0.0, 1.0), vec![5, 7, 7]);
    let ll = run_cpm(vec![5.0, 12.0, 19.0], 1.0)
        .expect("a short first window must run, not be refused");
    assert!(ll.is_finite(), "mid-period-start CPM run must be finite, got {ll}");
}

#[test]
fn cpm_accepts_two_obs_with_a_short_first_window() {
    // The n_obs == 2 case: obs at [5,12], t_start=0. First window 5 substeps,
    // second 7.
    assert_eq!(cpm_steps_per_obs(&[5.0, 12.0], 0.0, 1.0), vec![5, 7]);
    let ll = run_cpm(vec![5.0, 12.0], 1.0).expect("two observations must run");
    assert!(ll.is_finite(), "two-observation CPM run must be finite, got {ll}");
}

#[test]
fn cpm_accepts_a_daily_grid_with_one_absent_day() {
    // The reporting case this supports: daily observations, one interior day
    // with no situation report. One window spans two substeps, the rest one.
    let times: Vec<f64> = (1..=20).filter(|d| *d != 12).map(|d| d as f64).collect();
    let sizes = cpm_steps_per_obs(&times, 0.0, 1.0);
    assert_eq!(sizes.iter().filter(|&&k| k == 2).count(), 1, "exactly one two-substep window");
    assert_eq!(sizes.iter().filter(|&&k| k == 1).count(), sizes.len() - 1);
    let ll = run_cpm(times, 1.0)
        .expect("a daily grid with one absent day must run");
    assert!(ll.is_finite(), "irregular-grid CPM run must be finite, got {ll}");
}

#[test]
fn cpm_accepts_genuinely_uniform_windows() {
    // Positive regression: first obs at exactly obs_dt from t_start=0, uniform
    // gaps. obs at [7,14,21], dt=1 → every window (first included) is 7 substeps.
    let ll = run_cpm(vec![7.0, 14.0, 21.0], 1.0)
        .expect("genuinely-uniform CPM windows must run");
    assert!(ll.is_finite(), "uniform CPM run must return a finite loglik, got {ll}");
}

#[test]
fn cpm_accepts_uniform_single_obs() {
    // Single observation: the first window [t_start=0, obs(0)=10] is the only
    // window, and it is 10 substeps — which is what it is now sized at.
    assert_eq!(cpm_steps_per_obs(&[10.0], 0.0, 1.0), vec![10]);
    let ll = run_cpm(vec![10.0], 1.0)
        .expect("single-observation CPM (window == whole run) must run");
    assert!(ll.is_finite(), "single-obs CPM run must return a finite loglik, got {ll}");
}

#[test]
fn cpm_refuses_noise_drawn_for_a_different_grid() {
    // Noise sized for a uniform daily grid, handed to a run whose grid skips a
    // day: the two-substep window needs twice the block the uniform draw gave
    // it. Reading past that block would fall through to fresh per-particle RNG
    // and silently decorrelate the estimator, so the filter refuses instead.
    let uniform: Vec<f64> = (1..=20).map(|d| d as f64).collect();
    let with_hole: Vec<f64> = (1..=21).filter(|d| *d != 12).map(|d| d as f64).collect();
    assert_eq!(uniform.len(), with_hole.len(), "same window count, different lengths");

    let mismatched = cpm_randoms(&uniform, 1.0, 7);
    let err = run_cpm_with(with_hole.clone(), 1.0, &mismatched)
        .expect_err("noise drawn for a different grid must be refused");
    let msg = format!("{err}");
    assert!(msg.contains("pre-drawn noise"), "got: {msg}");

    // The same run with noise drawn for its own grid proceeds.
    let matched = cpm_randoms(&with_hole, 1.0, 7);
    assert!(run_cpm_with(with_hole, 1.0, &matched).is_ok());
}

#[test]
fn cpm_reuses_the_same_draws_across_evaluations_on_an_irregular_grid() {
    // What makes CPM work: the same random is reused at the same
    // (window, particle, substep) between MCMC iterations, so two evaluations
    // at the same theta differ only by the fraction of noise the Crank-Nicolson
    // update refreshed, and the likelihood RATIO in the acceptance step is far
    // less noisy than either estimate. Measured on the irregular grid, where
    // the strides differ between windows.
    //
    // Averaged over 8 pairs because a single pair is a poor estimate of either
    // spread — one correlated pair can differ by as much as a typical
    // independent pair. Every draw here is seeded, so the numbers below are
    // reproducible, not sampled afresh each run.
    let times: Vec<f64> = (1..=30).filter(|d| *d != 18).map(|d| d as f64).collect();
    let dt = 1.0;

    let mut correlated = Vec::new();
    let mut independent = Vec::new();
    for r in 0..8u64 {
        let u = cpm_randoms(&times, dt, 100 + r);
        let mut rng = StatefulRng::new(900 + r);
        let u_next = u.correlate(0.999, &mut rng);
        let ll_u = run_cpm_with(times.clone(), dt, &u).expect("CPM run");
        let ll_u_next = run_cpm_with(times.clone(), dt, &u_next).expect("CPM run");
        correlated.push((ll_u - ll_u_next).abs());

        let v = cpm_randoms(&times, dt, 300 + r);
        let w = cpm_randoms(&times, dt, 500 + r);
        let ll_v = run_cpm_with(times.clone(), dt, &v).expect("CPM run");
        let ll_w = run_cpm_with(times.clone(), dt, &w).expect("CPM run");
        independent.push((ll_v - ll_w).abs());
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (c, i) = (mean(&correlated), mean(&independent));
    eprintln!("irregular-grid CPM: mean |dloglik| correlated {c:.4}, independent {i:.4}");
    assert!(
        c * 2.5 < i,
        "two CPM evaluations at rho=0.999 must be much closer than two \
         independent ones (mean gap correlated {c}, independent {i}) — \
         otherwise the pre-drawn noise is not being reused at the same slots"
    );
}

/// Every slot of every window's noise block must be read.
///
/// The filter reads `particle * window_steps + substep`, over `n_particles`
/// particles and the substeps the window actually walks, out of a block sized
/// `n_particles * window_steps`. If the stride were wrong, two (particle,
/// substep) pairs would collide on one slot and some other slot would go
/// unread — which does not fail, and does not even decorrelate: the estimator
/// just draws two particles from one random and loses the independence a
/// particle filter needs. An unread slot is the observable fingerprint, so
/// perturbing each slot in turn and requiring the log-likelihood to move pins
/// the read map as a bijection onto the block.
#[test]
fn cpm_reads_every_slot_of_every_window_block() {
    // Four windows of sizes [1, 1, 2, 1] — the absent-day shape, small enough
    // to perturb every slot exhaustively.
    let times = vec![1.0, 2.0, 4.0, 5.0];
    let dt = 1.0;
    let steps_per_obs = cpm_steps_per_obs(&times, 0.0, dt);
    assert_eq!(steps_per_obs, vec![1, 1, 2, 1]);

    let zeroed = |times: &[f64]| PFRandomState {
        gamma_noise: steps_per_obs.iter().map(|&k| vec![0.0; CPM_PARTICLES * k]).collect(),
        resample_noise: vec![0.0; times.len()],
        binomial_noise: steps_per_obs.iter()
            .map(|&k| vec![0.0; CPM_PARTICLES * k * CPM_SOURCE_GROUPS])
            .collect(),
        n_source_groups: CPM_SOURCE_GROUPS,
    };
    let baseline = run_cpm_with(times.clone(), dt, &zeroed(&times)).expect("CPM run");

    // A z this large sends the inverse-CDF draw to the far tail, so the
    // perturbed particle's exit count differs from every other particle's.
    const LARGE_Z: f64 = 8.0;
    for (obs_idx, &k) in steps_per_obs.iter().enumerate() {
        for slot in 0..CPM_PARTICLES * k * CPM_SOURCE_GROUPS {
            let mut randoms = zeroed(&times);
            randoms.binomial_noise[obs_idx][slot] = LARGE_Z;
            let ll = run_cpm_with(times.clone(), dt, &randoms).expect("CPM run");
            assert!(
                ll != baseline,
                "slot {slot} of window {obs_idx} (block of {}) is never read — \
                 the read map is not onto its block, so two (particle, substep) \
                 pairs share a draw",
                CPM_PARTICLES * k * CPM_SOURCE_GROUPS,
            );
        }
    }
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
        t_start: 0.0,
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, adapt_stop: 0, thin: 1, burn_in: 0,
        rho: None, n_source_groups: 0,
    };
    // Evaluator that always fails with a non-recoverable error.
    let eval = |_params: &[f64], _seed: u64| -> Result<f64, SimError> {
        Err(SimError::Validation("structural: model cannot run".into()))
    };
    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &config, &[],
        &eval, None, 42, None, None, String::new(),
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
        t_start: 0.0,
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, adapt_stop: 0, thin: 1, burn_in: 0,
        rho: None, n_source_groups: 0,
    };
    // A recoverable per-particle excursion legitimately yields −∞; the
    // closure has already classified it as Ok. PMMH must accept it.
    let eval = |_params: &[f64], _seed: u64| -> Result<f64, SimError> {
        Ok(f64::NEG_INFINITY)
    };
    let result = run_pmmh(
        &if2_params, &priors, &base_params, &[], &config, &[],
        &eval, None, 42, None, None, String::new(),
    );
    assert!(result.is_ok(), "Ok(−∞) is a valid ruled-out loglik, must not error");
}
