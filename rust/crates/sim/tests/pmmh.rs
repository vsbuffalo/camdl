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
        burn_in: 500, rho: None, n_source_groups: 0, init_noise_width: 0,
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
        burn_in: 0, rho: None, n_source_groups: 0, init_noise_width: 0,
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
        burn_in: 0, rho: None, n_source_groups: 0, init_noise_width: 0,
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
        burn_in: 500, rho: None, n_source_groups: 0, init_noise_width: 0,
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
        burn_in: 0, rho: None, n_source_groups: 0, init_noise_width: 0,
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
        burn_in: 0, rho: None, n_source_groups: 0, init_noise_width: 0,
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

/// Run `bootstrap_filter_correlated` on a built model and observation model —
/// the seam the pure-death harness and the law-bearing one below share.
fn run_cpm_core(
    compiled: CompiledModel,
    params: &[f64],
    obs_model: &dyn ObservationModel<ParticleState>,
    dt: f64,
    randoms: &PFRandomState,
) -> Result<f64, sim::error::SimError> {
    let process = ChainBinomialProcess::new(Arc::new(compiled));
    let config = SMCConfig {
        n_particles: CPM_PARTICLES,
        dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter_correlated(&process, obs_model, params, &config, randoms, 7)
        .map(|r| r.log_likelihood)
}

/// Build the smallest CPM harness: pure-death `ChainBinomialProcess` + a
/// `PoissonPrevalenceObs` at the given obs times, run `bootstrap_filter_correlated`
/// against `randoms`.
fn run_cpm_with(
    obs_times: Vec<f64>,
    dt: f64,
    randoms: &PFRandomState,
) -> Result<f64, sim::error::SimError> {
    let (compiled, params) = pure_death_model();
    let n_obs = obs_times.len();
    let obs_model = PoissonPrevalenceObs {
        observations: vec![50.0; n_obs],
        obs_times,
    };
    run_cpm_core(compiled, &params, &obs_model, dt, randoms)
}

const CPM_PARTICLES: usize = 64;
const CPM_SOURCE_GROUPS: usize = 1;

/// Noise drawn for exactly this grid and this model, the way `run_pmmh` draws
/// it. `init_width` is the model's `CompiledModel::init_noise_width` — zero for
/// every deterministic-`init { }` model, which is all of them but the law
/// fixture below.
fn cpm_randoms_for(
    obs_times: &[f64], dt: f64, n_groups: usize, init_width: usize, seed: u64,
) -> PFRandomState {
    let steps_per_obs = cpm_steps_per_obs(obs_times, 0.0, dt);
    let mut rng = StatefulRng::new(seed);
    PFRandomState::draw_fresh(CPM_PARTICLES, &steps_per_obs, n_groups, init_width, &mut rng)
}

fn cpm_randoms(obs_times: &[f64], dt: f64, seed: u64) -> PFRandomState {
    cpm_randoms_for(obs_times, dt, CPM_SOURCE_GROUPS, 0, seed)
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
        // The pure-death fixture's `init { }` is deterministic.
        init_noise: Vec::new(),
        init_width: 0,
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

// ── Correlated PF with a DRAWN initial state (gh#772) ──────────────────────
//
// `init { I ~ poisson(rate = I0) }` makes x₀ a random variable the filter has
// to integrate over. Correlated PMMH used to refuse such a model outright,
// because its pre-drawn correlated vector covered the transition kernel only
// and an x₀ drawn from a ChaCha stream would have been uncorrelated noise added
// to the one quantity the method needs correlated between the current and
// proposed θ. The vector now carries an initial-state block too, one row of
// `init_noise_width` normals per particle, so x₀ is correlated on the same
// terms as everything else — and each particle draws its own, which is what
// makes the swarm integrate over p(x₀ | θ) rather than condition on one
// realization of it.
//
// The two properties `cpm_reads_every_slot_of_every_window_block` and
// `cpm_reuses_the_same_draws_across_evaluations_on_an_irregular_grid` pin for
// the window blocks are repeated below for the init block.

/// Observes the TOTAL count across every compartment, so every initial-state
/// law's slot is visible in the log-likelihood. (`PoissonPrevalenceObs` reads
/// compartment 0 only, which would make two thirds of the init block
/// unobservable and the slot-bijection test below vacuous for them.)
struct PoissonTotalObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonTotalObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let total: i64 = state.counts.iter().sum();
        poisson_logpmf(self.observations[obs_idx], (total as f64).max(0.1))
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _s: &ParticleState, _i: usize, _p: &[f64], _r: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> Vec<f64> { vec![] }
}

/// Three decaying compartments whose initial counts are DRAWN, one from each
/// of the three count laws `init { }` admits.
///
/// NegBinomial is in there on purpose: it is generated as a Gamma multiplier
/// mixed into a Poisson, so it consumes TWO of the pre-drawn normals where the
/// others consume one. The block is therefore `[A, B, C_gamma, C_poisson]` —
/// four slots per particle, not three — which is exactly the case a slot map
/// derived from "the j-th law drawn so far" would get wrong.
///
/// Every compartment is integer. A real-valued `init { }` law would consume a
/// slot the correlated filter cannot show in a log-likelihood, because no
/// particle filter advances a real compartment
/// (`docs/dev/incidents/2026-06-07-chain-binomial-stale-real-state.md`); that
/// case is covered at the producer instead, in
/// `tests/gh772_cpm_initial_state_laws.rs`.
fn three_law_decay_model() -> (CompiledModel, Vec<f64>) {
    use ir::deriv::Diffable;
    use ir::model::{InitCountLaw, InitSpec};
    use ir::observation::{BinomialLikelihood, NegBinomialLikelihood, PoissonLikelihood};

    let decay = |comp: &str| Transition {
        rate_state_grad: Default::default(),
        name: format!("decay_{comp}"),
        stoichiometry: vec![StoichiometryEntry(comp.into(), -1)],
        rate: Expr::BinOp(BinOpWrap {
            bin_op: BinOpExpr {
                op: BinOp::Mul,
                left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                right: Box::new(Expr::Pop(PopExpr { pop: comp.into() })),
            },
        }),
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    };
    let param = |name: &str, value: f64| Parameter {
        name: name.into(),
        value: ir::parameter::ParamValue::Fixed { value },
        param_kind: None,
        param_dim: None,
    };
    let p = |name: &str| Expr::Param(ParamExpr { param: name.into() });

    let mut initial_conditions = InitialConditions::constants([
        ("A".to_string(), 0.0), ("B".to_string(), 0.0), ("C".to_string(), 0.0),
    ]);
    // `insert` on an existing key keeps its position, so declaration order is
    // A, B, C — and with no cross-references, that is the evaluation order too.
    initial_conditions.0.insert(
        "A".into(),
        InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood {
            rate: Diffable::new(p("a0")),
        })),
    );
    initial_conditions.0.insert(
        "B".into(),
        InitSpec::Count(InitCountLaw::Binomial(BinomialLikelihood {
            n: p("n0"),
            p: Diffable::new(p("p0")),
        })),
    );
    initial_conditions.0.insert(
        "C".into(),
        InitSpec::Count(InitCountLaw::NegBinomial(NegBinomialLikelihood {
            mean: Diffable::new(p("c0")),
            dispersion: Diffable::new(p("k0")),
        })),
    );

    let model = Model {
        ic_grad: Default::default(),
        name: "three_law_decay".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "A".into(), kind: CompartmentKind::Integer },
            Compartment { name: "B".into(), kind: CompartmentKind::Integer },
            Compartment { name: "C".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![decay("A"), decay("B"), decay("C")],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            param("mu", 0.02),
            // Poisson(20); Binomial(40, 0.5) — np = nq = 20, so B goes through
            // the exact inverse CDF rather than the normal approximation;
            // NegBinomial(mean 20, k 5).
            param("a0", 20.0), param("n0", 40.0), param("p0", 0.5),
            param("c0", 20.0), param("k0", 5.0),
        ],
        initial_conditions,
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 10.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 10.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
        quantities: vec![], contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).expect("the law fixture must compile");
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Observation times and the noise shape the law fixture runs at.
const LAW_OBS: [f64; 2] = [1.0, 2.0];
const LAW_SOURCE_GROUPS: usize = 3; // A, B and C each source one transition
const LAW_INIT_WIDTH: usize = 4;    // Poisson 1 + Binomial 1 + NegBinomial 2

fn run_cpm_law_with(randoms: &PFRandomState) -> Result<f64, sim::error::SimError> {
    let (compiled, params) = three_law_decay_model();
    let obs_model = PoissonTotalObs {
        observations: vec![55.0; LAW_OBS.len()],
        obs_times: LAW_OBS.to_vec(),
    };
    run_cpm_core(compiled, &params, &obs_model, 1.0, randoms)
}

fn law_randoms(seed: u64) -> PFRandomState {
    cpm_randoms_for(&LAW_OBS, 1.0, LAW_SOURCE_GROUPS, LAW_INIT_WIDTH, seed)
}

/// The refusal is gone: a model whose `init { }` draws from a law runs under
/// the correlated filter and returns a finite log-likelihood.
#[test]
fn cpm_accepts_a_declared_init_law() {
    let (compiled, _) = three_law_decay_model();
    assert!(compiled.has_init_law, "the fixture must declare at least one law");
    assert_eq!(
        compiled.init_noise_width, LAW_INIT_WIDTH,
        "Poisson + Binomial + NegBinomial is 1 + 1 + 2 normals per particle"
    );
    let ll = run_cpm_law_with(&law_randoms(11))
        .expect("a declared `init { }` law must no longer be refused (gh#772)");
    assert!(ll.is_finite(), "law-bearing CPM run must be finite, got {ll}");
}

/// Every slot of every particle's initial-state block must be read, exactly
/// once — the init-block twin of `cpm_reads_every_slot_of_every_window_block`.
///
/// The filter reads `particle * init_width + slot`. A wrong stride does not
/// error and does not decorrelate; it just has two particles share a draw,
/// which costs the swarm the independence a particle filter needs, and leaves
/// some other slot unread. The unread slot is the observable fingerprint:
/// perturb each in turn and require the log-likelihood to move.
///
/// This is also what pins x₀ as PER PARTICLE. If the filter took one draw and
/// copied it, only the first particle's block would be read and the other
/// `(64 - 1) * 4` perturbations would leave the log-likelihood untouched.
#[test]
fn cpm_reads_every_slot_of_every_init_block() {
    let steps_per_obs = cpm_steps_per_obs(&LAW_OBS, 0.0, 1.0);
    let zeroed = || PFRandomState {
        gamma_noise: steps_per_obs.iter().map(|&k| vec![0.0; CPM_PARTICLES * k]).collect(),
        resample_noise: vec![0.0; LAW_OBS.len()],
        binomial_noise: steps_per_obs.iter()
            .map(|&k| vec![0.0; CPM_PARTICLES * k * LAW_SOURCE_GROUPS])
            .collect(),
        n_source_groups: LAW_SOURCE_GROUPS,
        init_noise: vec![0.0; CPM_PARTICLES * LAW_INIT_WIDTH],
        init_width: LAW_INIT_WIDTH,
    };
    let baseline = run_cpm_law_with(&zeroed()).expect("CPM run");

    // Far enough into the tail that the perturbed particle's x₀ differs from
    // every other particle's on whichever law the slot belongs to.
    const LARGE_Z: f64 = 8.0;
    for slot in 0..CPM_PARTICLES * LAW_INIT_WIDTH {
        let mut randoms = zeroed();
        randoms.init_noise[slot] = LARGE_Z;
        let ll = run_cpm_law_with(&randoms).expect("CPM run");
        assert!(
            ll != baseline,
            "init-noise slot {slot} (particle {}, offset {}) is never read — the \
             read map is not onto the block, so two particles share an x0 draw",
            slot / LAW_INIT_WIDTH, slot % LAW_INIT_WIDTH,
        );
    }
}

/// The initial state is part of the correlated vector, not fresh noise added
/// to it — the init-block twin of
/// `cpm_reuses_the_same_draws_across_evaluations_on_an_irregular_grid`.
///
/// Isolated to the init block on purpose: both arms hold the window blocks
/// FIXED and identical, so the only thing that differs between the two
/// evaluations is where x₀ came from. With `rho` near 1 the two initial states
/// are nearly the same and the log-likelihoods track each other; with
/// independent init noise they do not. If x₀ were still drawn from a ChaCha
/// stream (or from noise the Crank-Nicolson update skipped), the two arms
/// would be indistinguishable.
#[test]
fn cpm_reuses_the_same_init_draws_across_evaluations() {
    let mut correlated = Vec::new();
    let mut independent = Vec::new();
    for r in 0..16u64 {
        // One draw of the window blocks, shared by every evaluation in this
        // replicate: whatever differs below is the initial state.
        let base = law_randoms(100 + r);

        let mut rng = StatefulRng::new(900 + r);
        let scale = (1.0f64 - 0.999 * 0.999).sqrt();
        let mut nudged = base.clone();
        for x in &mut nudged.init_noise {
            *x = 0.999 * *x + scale * rng.normal();
        }
        let ll_a = run_cpm_law_with(&base).expect("CPM run");
        let ll_b = run_cpm_law_with(&nudged).expect("CPM run");
        correlated.push((ll_a - ll_b).abs());

        let mut fresh = base.clone();
        fresh.init_noise = law_randoms(300 + r).init_noise;
        let ll_c = run_cpm_law_with(&fresh).expect("CPM run");
        let mut fresh2 = base.clone();
        fresh2.init_noise = law_randoms(500 + r).init_noise;
        let ll_d = run_cpm_law_with(&fresh2).expect("CPM run");
        independent.push((ll_c - ll_d).abs());
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (c, i) = (mean(&correlated), mean(&independent));
    eprintln!("init-block CPM: mean |dloglik| correlated {c:.4}, independent {i:.4}");
    assert!(
        c * 2.5 < i,
        "two CPM evaluations whose init noise differs only by a rho=0.999 \
         Crank-Nicolson step must be much closer than two with independent \
         init noise (mean gap correlated {c}, independent {i}) — otherwise x0 \
         is not being drawn from the correlated vector"
    );
}

/// The Crank-Nicolson update carries the init block, at the same `rho` as
/// everything else.
///
/// Asserted on `correlate` directly, and by the correlation it induces rather
/// than by "the numbers changed": a `correlate` that copied the init block
/// through unchanged would make two successive evaluations identical, which
/// looks like perfect correlation from the outside and is in fact no
/// refreshment at all. Measured over 16,384 slots, where the sampling error of
/// a correlation is under 0.01.
#[test]
fn correlate_carries_the_init_block_at_rho() {
    let mut rng = StatefulRng::new(4242);
    let base = PFRandomState::draw_fresh(4096, &[1], 1, 4, &mut rng);
    assert_eq!(base.init_noise.len(), 4096 * 4, "one block of 4 per particle");
    assert_eq!(base.init_width, 4);

    let corr = |a: &[f64], b: &[f64]| -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / n;
        let sa = (a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / n).sqrt();
        let sb = (b.iter().map(|y| (y - mb).powi(2)).sum::<f64>() / n).sqrt();
        cov / (sa * sb)
    };

    // rho = 1: the whole vector is reused verbatim.
    let same = base.correlate(1.0, &mut StatefulRng::new(1));
    assert_eq!(same.init_noise, base.init_noise, "rho = 1 must reuse the block exactly");
    assert_eq!(same.init_width, base.init_width);

    for &rho in &[0.0, 0.5, 0.9, 0.999] {
        let next = base.correlate(rho, &mut StatefulRng::new(7));
        assert_eq!(next.init_noise.len(), base.init_noise.len());
        let c = corr(&base.init_noise, &next.init_noise);
        eprintln!("init block: rho = {rho}, measured correlation {c:.4}");
        assert!(
            (c - rho).abs() < 0.02,
            "the init block must be refreshed at rho = {rho}, measured {c} — \
             a block left untouched reads as correlation 1 and never refreshes"
        );
    }
}

/// Init noise drawn for a different model is refused rather than strided
/// wrongly. A block sized for one law fed to a model with three would read
/// each particle at the wrong offset — valid floats from the wrong
/// compartment's slot, with no error to see.
#[test]
fn cpm_refuses_init_noise_drawn_for_a_different_model() {
    let mismatched = cpm_randoms_for(&LAW_OBS, 1.0, LAW_SOURCE_GROUPS, 1, 11);
    let err = run_cpm_law_with(&mismatched)
        .expect_err("init noise drawn for a different model must be refused");
    let msg = format!("{err}");
    assert!(msg.contains("initial-state normals per particle"), "got: {msg}");

    // Same run with noise drawn for its own model proceeds.
    assert!(run_cpm_law_with(&law_randoms(11)).is_ok());
}

/// End to end: `run_pmmh` with `rho` set, on a model that draws its initial
/// state. This is the acceptance criterion — it exercises `draw_fresh`'s
/// sizing, `correlate`'s extension over the init block, and the filter's read,
/// through the driver that actually runs a fit.
#[test]
fn pmmh_with_rho_runs_on_a_model_that_draws_its_initial_state() {
    let (compiled, base_params) = three_law_decay_model();
    let init_noise_width = compiled.init_noise_width;
    let n_source_groups = compiled.source_groups.len();
    let compiled = Arc::new(compiled);

    let observations: Vec<sim::inference::if2::Observation> = LAW_OBS.iter()
        .map(|&t| sim::inference::if2::Observation { time: t, value: 55.0 })
        .collect();
    let config = PMMHConfig {
        n_steps: 30,
        n_particles: 32,
        dt: 1.0,
        t_start: 0.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 0,
        adapt_stop: 0,
        thin: 1,
        burn_in: 0,
        rho: Some(0.99),
        n_source_groups,
        init_noise_width,
    };
    let smc_cfg = SMCConfig {
        n_particles: config.n_particles,
        dt: config.dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_model = PoissonTotalObs {
        observations: vec![55.0; LAW_OBS.len()],
        obs_times: LAW_OBS.to_vec(),
    };
    let eval_corr = |theta: &[f64], randoms: &PFRandomState| -> Result<f64, SimError> {
        bootstrap_filter_correlated(&process, &obs_model, theta, &smc_cfg, randoms, 7)
            .map(|r| r.log_likelihood)
    };
    // Vanilla evaluator: unused when `rho` is set and a correlated one is
    // supplied, but `run_pmmh` takes both.
    let eval = |_: &[f64], _: u64| -> Result<f64, SimError> {
        Err(SimError::Validation("the correlated evaluator must be the one used".into()))
    };

    let mu = EstimatedParam {
        name: "mu".into(), index: 0, initial: 0.02, rw_sd: 0.005,
        transform: Transform::Log { lo: 1e-6, hi: 1.0 },
        lower: 1e-6, upper: 1.0, rw_sd_auto: false, perturb_only_at_t0: false,
    };
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let result = run_pmmh(
        &[mu], &priors, &base_params, &[], &config, &observations,
        &eval, Some(&eval_corr), 42, None, None, String::new(),
    ).expect("correlated PMMH must run on a model that draws its initial state");

    assert_eq!(result.n_steps, 30);
    assert!(
        result.steps.iter().all(|s| s.log_likelihood.is_finite()),
        "every recorded correlated-PMMH loglik must be finite"
    );
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
        rho: None, n_source_groups: 0, init_noise_width: 0,
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
        rho: None, n_source_groups: 0, init_noise_width: 0,
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
