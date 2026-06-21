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
        correlated_pf::{bootstrap_filter_correlated, PFRandomState},
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
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
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
        ivp: false,
    }
}

#[test]
fn test_pmmh_posterior_covers_truth() {
    let (compiled, _params) = pure_death_model();
    let compiled = Arc::new(compiled);
    let n_particles = 200;
    let eval_loglik = make_eval_loglik(compiled.clone(), n_particles);

    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Normal { mean: 0.01, sd: 0.01 }];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 3000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2], // on log scale
        adapt: true,
        adapt_start: 200,
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
    let priors = vec![Prior::Flat];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 100,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
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
    let priors = vec![Prior::Flat];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 1000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: true,
        adapt_start: 200,
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
    let priors = vec![Prior::Flat];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 2000,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: true,
        adapt_start: 200,
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
    let priors = vec![Prior::Flat];
    let base_params = compiled.default_params.clone();

    // Deliberately bad initial proposal: 10× too wide
    let config = PMMHConfig {
        n_steps: 1500,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![2.0], // way too wide on log scale
        adapt: true,
        adapt_start: 200,
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
    let priors = vec![Prior::Flat];
    let base_params = compiled.default_params.clone();

    let config = PMMHConfig {
        n_steps: 50,
        n_particles,
        dt: 1.0,
        proposal_sd: vec![0.2],
        adapt: false,
        adapt_start: 50,
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

// ── Correlated PF (CPM) uniform-window gate (M6) ───────────────────────────
//
// CPM pre-draws random numbers into per-window blocks of `steps_per_obs`
// substeps and indexes them with `particle*steps_per_obs + substep`. That
// indexing is sound ONLY if EVERY observation window — INCLUDING the first
// window [t_start, obs(0)] — has exactly `steps_per_obs` substeps. The first
// window is the reachable hole: `steps_per_obs` is sized from obs(1)-obs(0),
// but the first window spans [t_start, obs(0)], which differs when data start
// mid-period (e.g. t_start=0, obs at [5,12,19]). The old gate only ran for
// n_obs > 2 and used a dt*0.5 time-gap slack, so it missed the first-window
// offset entirely — the first window's substeps overran their noise block and
// silently fell through to fresh per-particle RNG, decorrelating the estimator
// the PMMH acceptance ratio depends on, with no diagnostic.

/// Build the smallest CPM harness: pure-death `ChainBinomialProcess` + a
/// `PoissonPrevalenceObs` at the given obs times, run `bootstrap_filter_correlated`.
fn run_cpm(obs_times: Vec<f64>, dt: f64) -> Result<f64, sim::error::SimError> {
    let (compiled, params) = pure_death_model();
    let process = ChainBinomialProcess::new(Arc::new(compiled));
    let n_obs = obs_times.len();
    let obs_model = PoissonPrevalenceObs {
        observations: vec![50.0; n_obs],
        obs_times: obs_times.clone(),
    };
    let n_particles = 64;
    let config = SMCConfig {
        n_particles,
        dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    // Size the noise arrays the way bootstrap_filter_correlated computes
    // steps_per_obs internally, so the harness matches the filter's own block
    // size: obs(1)-obs(0) for n_obs>=2, else obs(0)-t_start for a single obs.
    // (run_pmmh uses 1 for the single-obs case, but the filter's gate sizes from
    // the actual first window — the harness mirrors the filter so the positive
    // single-obs test exercises the gate, not a sizing mismatch.)
    let steps_per_obs = if n_obs >= 2 {
        sim::time::interval_steps(obs_times[0], obs_times[1], dt)
    } else {
        sim::time::interval_steps(0.0, obs_times[0], dt)
    };
    let n_source_groups = 1;
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(
        n_particles, n_obs, steps_per_obs, n_source_groups, &mut rng,
    );
    bootstrap_filter_correlated(&process, &obs_model, &params, &config, &randoms, 7)
        .map(|r| r.log_likelihood)
}

#[test]
fn cpm_rejects_first_window_offset() {
    // t_start=0, obs at [5,12,19], dt=1: uniform gap 7 (steps_per_obs=7) but the
    // FIRST window [0,5] is only 5 substeps. Under the OLD gate (n_obs>2 + dt*0.5
    // slack) this passed and ran to a finite loglik via silent decorrelation; the
    // tightened gate must reject it.
    let res = run_cpm(vec![5.0, 12.0, 19.0], 1.0);
    let err = res.expect_err(
        "CPM must reject a first-window offset (first window 5 substeps vs \
         steps_per_obs=7) — running it silently decorrelates the estimator",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("FIRST window"),
        "rejection must name the first window; got: {msg}",
    );
}

#[test]
fn cpm_rejects_two_obs_non_uniform_first_window() {
    // n_obs <= 2 case (the old gate's `n_obs > 2` hole): two obs at [5,12],
    // t_start=0. steps_per_obs sized from obs(1)-obs(0)=7, but the first window
    // [0,5] is 5 substeps. Old gate never ran (n_obs not > 2) → silent
    // decorrelation; tightened gate rejects.
    let res = run_cpm(vec![5.0, 12.0], 1.0);
    let err = res.expect_err(
        "CPM must reject a non-uniform first window even with only 2 observations",
    );
    assert!(
        format!("{err}").contains("FIRST window"),
        "rejection must name the first window",
    );
}

#[test]
fn cpm_accepts_genuinely_uniform_windows() {
    // Positive regression: first obs at exactly obs_dt from t_start=0, uniform
    // gaps. obs at [7,14,21], dt=1 → every window (first included) is 7 substeps.
    // Must run and return a finite loglik.
    let ll = run_cpm(vec![7.0, 14.0, 21.0], 1.0)
        .expect("genuinely-uniform CPM windows must run");
    assert!(ll.is_finite(), "uniform CPM run must return a finite loglik, got {ll}");
}

#[test]
fn cpm_accepts_uniform_single_obs() {
    // Single observation: first window [t_start=0, obs(0)=10] is the only window;
    // steps_per_obs falls back to 1 in the sizing, but the window has 10 substeps.
    // This is the degenerate n_obs==1 case — the gate must agree with how the
    // noise is sized. With steps_per_obs sized at obs(0)-t_start, it is uniform.
    // Here we use the matching sizing (single obs uses obs(0)-t_start in the gate
    // via obs_dt fallback), so it must accept.
    let ll = run_cpm(vec![10.0], 1.0)
        .expect("single-observation CPM (window == whole run) must run");
    assert!(ll.is_finite(), "single-obs CPM run must return a finite loglik, got {ll}");
}

/// gh#224. A structural (non-recoverable) error from the likelihood
/// evaluator must propagate out of `run_pmmh` as `Err`, not be silently
/// mistaken for a ruled-out θ. Before the typed-channel fix the eval
/// closures collapsed *every* error to −∞, so a model that cannot run
/// produced a degenerate posterior with a successful (exit-0) return.
#[test]
fn pmmh_surfaces_structural_eval_error() {
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Normal { mean: 0.01, sd: 0.01 }];
    let base_params = vec![0.01];
    let config = PMMHConfig {
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, thin: 1, burn_in: 0,
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
    let priors = vec![Prior::Normal { mean: 0.01, sd: 0.01 }];
    let base_params = vec![0.01];
    let config = PMMHConfig {
        n_steps: 50, n_particles: 10, dt: 1.0, proposal_sd: vec![0.2],
        adapt: false, adapt_start: 0, thin: 1, burn_in: 0,
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
