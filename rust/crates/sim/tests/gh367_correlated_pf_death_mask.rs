//! gh#367: the correlated PF must apply the SAME per-particle death mask the
//! bootstrap PF applies — one particle's *recoverable* error kills that
//! particle (−∞ weight), it does not reject the whole θ.
//!
//! Before the fix, `bootstrap_filter_correlated` propagated any particle's
//! error out of the whole evaluation (`step_one(...)?` inside the per-particle
//! closure, then `for r in errors { r?; }`). The PMMH driver maps a
//! non-structural `Err` to −∞ (`fit/pmmh.rs`: `Err(e) if e.is_structural() =>
//! Err(e), Err(_) => Ok(f64::NEG_INFINITY)`), so a single particle hitting
//! `NumericalCollapse` / `NegativeCount{BinomialOvershoot}` rejected the ENTIRE
//! proposal — silently biasing correlated PMMH against boundary regions where
//! occasional particle failure occurs. The bootstrap PF has never behaved that
//! way (`particle_filter.rs`: `Err(e) if e.is_per_particle_recoverable() =>
//! return Ok(true)` → `−∞` weight → resampling discards it).
//!
//! # Harness
//!
//! `bootstrap_filter_correlated` takes a concrete `&ChainBinomialProcess`, so
//! the failing particle cannot be injected through a stub process model. It is
//! instead driven by the pre-drawn correlated noise, which the caller owns
//! completely (`PFRandomState` fields are public):
//!
//! ```text
//!   X --drain--> Y --foi--> Z      drain: mu·X      foi: beta·Y/X
//! ```
//!
//! `X0 = 3`, so a binomial z-value of `+LARGE_Z` for particle 0's `X` source
//! group in substep 0 empties `X` (inverse-CDF branch: `u → 1` ⇒ `k = n`),
//! while `-LARGE_Z` for every other particle draws `k = 0` and leaves `X = 3`.
//! At substep 1 particle 0 therefore evaluates `beta·Y/0` — non-finite ⇒
//! `SimError::NumericalCollapse{DivByZero}`, which
//! `is_per_particle_recoverable()` reports `true`. Both substeps live in ONE
//! observation window, so no resampling intervenes between the drain and the
//! error and exactly one particle dies.
//!
//! The negative control reuses the identical model, noise and drain, changing
//! only `foi`'s rate to `beta·(X − 1)` — negative once `X = 0`, i.e.
//! `SimError::NegativePropensity`, for which `is_per_particle_recoverable()` is
//! `false`. That error must still tear the evaluation down, NOT be masked.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, Expr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    error::SimError,
    inference::{
        correlated_pf::{bootstrap_filter_correlated, PFRandomState},
        obs_loglik::poisson_logpmf,
        traits::{ObservationModel, SMCConfig},
        ChainBinomialProcess, ParticleState,
    },
    rng::StatefulRng,
};

/// Magnitude of the pre-drawn binomial z-value used to force a source group to
/// drain completely (`+`) or not at all (`−`). `Φ(±1e6)` saturates at the
/// inverse-CDF clamp, so the drawn count is deterministically `n` or `0`.
const LARGE_Z: f64 = 1e6;

const N_PARTICLES: usize = 8;
const DT: f64 = 0.5;
/// Two substeps per observation window ⇒ the drain (substep 0) and the error it
/// causes (substep 1) both land inside window 0, before any resampling.
const STEPS_PER_OBS: usize = 2;
const OBS_TIMES: [f64; 2] = [1.0, 2.0];
const X0: f64 = 3.0;
const Y0: f64 = 100.0;

/// Two-transition chain whose second rate is supplied by the caller, so the
/// same drain machinery can produce either a recoverable or a non-recoverable
/// error at `X = 0`.
fn chain_model(foi_rate: Expr) -> (CompiledModel, Vec<f64>) {
    let drain_rate = Expr::bin_op(BinOp::Mul, Expr::param("mu"), Expr::pop("X"));

    let tr = |name: &str, from: &str, to: &str, rate: Expr| Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![
            StoichiometryEntry(from.into(), -1),
            StoichiometryEntry(to.into(), 1),
        ],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    };

    let model = Model {
        ic_grad: Default::default(),
        name: "gh367_chain".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "X".into(), kind: CompartmentKind::Integer },
            Compartment { name: "Y".into(), kind: CompartmentKind::Integer },
            Compartment { name: "Z".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            tr("drain", "X", "Y", drain_rate),
            tr("foi", "Y", "Z", foi_rate),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.5 }, param_kind: None, param_dim: None },
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: 0.3 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::constants({
            let mut m = HashMap::new();
            m.insert("X".into(), X0);
            m.insert("Y".into(), Y0);
            m.insert("Z".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(OBS_TIMES.to_vec()),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: *OBS_TIMES.last().unwrap(),
            time_semantics: "continuous".into(),
            dt: Some(DT),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// `beta·Y/X` — non-finite once `X = 0` ⇒ `NumericalCollapse{DivByZero}`,
/// per-particle recoverable.
fn recoverable_foi_rate() -> Expr {
    Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("Y")),
        Expr::pop("X"),
    )
}

/// `beta·(X − 1)` — negative once `X = 0` ⇒ `NegativePropensity`, which is NOT
/// per-particle recoverable and must still tear the evaluation down.
fn nonrecoverable_foi_rate() -> Expr {
    Expr::bin_op(
        BinOp::Mul,
        Expr::param("beta"),
        Expr::bin_op(BinOp::Sub, Expr::pop("X"), Expr::const_(1.0)),
    )
}

/// Poisson prevalence on `Y`, scored at each observation time.
struct PoissonYObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
    y_idx: usize,
}

impl ObservationModel<ParticleState> for PoissonYObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let lambda = (state.counts[self.y_idx] as f64).max(1e-9);
        poisson_logpmf(self.observations[obs_idx], lambda)
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _s: &ParticleState, _i: usize, _p: &[f64], _rng: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> Vec<f64> { vec![] }
}

/// Pre-drawn noise that empties particle 0's `X` source group at substep 0 of
/// window 0 and leaves every other particle's groups untouched.
///
/// Layout mirrors `bootstrap_filter_correlated`'s own indexing:
///   `binomial_noise[obs][particle*steps_per_obs*n_groups + substep*n_groups + group]`
///   `gamma_noise[obs][particle*steps_per_obs + substep]`
fn crafted_randoms(n_groups: usize, n_obs: usize) -> PFRandomState {
    let per_obs_binom = N_PARTICLES * STEPS_PER_OBS * n_groups;
    let mut binomial_noise: Vec<Vec<f64>> = (0..n_obs)
        .map(|_| vec![-LARGE_Z; per_obs_binom])
        .collect();
    // Particle 0, window 0, substep 0: drain EVERY source group completely.
    // Whichever group index `X` occupies, its exits are maximal, so `X → 0`.
    for group in 0..n_groups {
        binomial_noise[0][group] = LARGE_Z;
    }
    PFRandomState {
        gamma_noise: (0..n_obs).map(|_| vec![0.0; N_PARTICLES * STEPS_PER_OBS]).collect(),
        resample_noise: vec![0.0; n_obs],
        binomial_noise,
        n_source_groups: n_groups,
        // This fixture's `init { }` is deterministic, so the initial-state
        // block is empty and every particle starts at the same state.
        init_noise: Vec::new(),
        init_width: 0,
    }
}

fn smc_config() -> SMCConfig {
    SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    }
}

fn run(foi_rate: Expr) -> Result<f64, SimError> {
    let (compiled, params) = chain_model(foi_rate);
    let y_idx = compiled.model.compartments.iter().position(|c| c.name == "Y").unwrap();
    let n_groups = compiled.source_groups.len();
    assert_eq!(n_groups, 2, "X and Y each have exactly one exit");

    let process = ChainBinomialProcess::new(Arc::new(compiled));
    let obs_model = PoissonYObs {
        observations: vec![Y0; OBS_TIMES.len()],
        obs_times: OBS_TIMES.to_vec(),
        y_idx,
    };
    let randoms = crafted_randoms(n_groups, OBS_TIMES.len());
    bootstrap_filter_correlated(&process, &obs_model, &params, &smc_config(), &randoms, 7)
        .map(|r| r.log_likelihood)
}

/// Anti-vacuity guard for the two tests below: assert the crafted z-values
/// really do drive a particle into a per-particle-**recoverable** error. Drives
/// `chain_binomial::step_one` directly with exactly the z-values particle 0
/// sees inside the correlated filter. Without this, a "finite loglik" could
/// mean the mask worked OR that nothing ever failed.
#[test]
fn harness_produces_a_recoverable_per_particle_error() {
    let (compiled, params) = chain_model(recoverable_foi_rate());

    let mut counts: Vec<i64> = vec![X0 as i64, Y0 as i64, 0];
    let mut flows: Vec<u64> = vec![0; compiled.model.transitions.len()];
    let mut real = sim::state::RealState::new(compiled.real_local_to_global.len());
    let mut rng = StatefulRng::new(7);
    let mut scratch = sim::chain_binomial::StepScratch::new(&compiled);
    let n_groups = compiled.source_groups.len();

    scratch.binomial_z_values = vec![LARGE_Z; n_groups];
    scratch.binomial_z_idx = 0;
    sim::chain_binomial::step_one(
        &compiled, &mut counts, &mut flows, &mut real, &params, 0.0, DT, None,
        sim::rng::BinomialAlgorithm::default(), &mut rng, &mut scratch,
    ).expect("substep 0 drains X but must not error");
    assert_eq!(counts[0], 0, "substep 0 must empty X (got counts {counts:?})");

    scratch.binomial_z_values = vec![-LARGE_Z; n_groups];
    scratch.binomial_z_idx = 0;
    let err = sim::chain_binomial::step_one(
        &compiled, &mut counts, &mut flows, &mut real, &params, DT, DT, None,
        sim::rng::BinomialAlgorithm::default(), &mut rng, &mut scratch,
    ).expect_err("substep 1 must hit beta*Y/0");
    assert!(
        err.is_per_particle_recoverable(),
        "the harness must produce a RECOVERABLE error (the class gh#367 is \
         about); got {err}"
    );
}

/// gh#367 (the fix): exactly one particle hits a per-particle-recoverable
/// error. The correlated PF must mask that particle (−∞ weight) and return a
/// FINITE log-likelihood for θ — not `Err`, and not −∞ for the whole
/// evaluation.
///
/// The returned value is pinned, not just checked for finiteness, because the
/// whole point is *which* particles contribute. Every live particle draws zero
/// exits, so all 7 survivors sit at `Y = Y0` and score `L =
/// poisson_logpmf(Y0, Y0)`:
///
///   window 0: `logsumexp([L×7, −∞]) − ln 8 = L + ln 7 − ln 8`
///   window 1: the dead slot was refilled from a live ancestor at resampling
///             and the mask cleared, so all 8 score `L` ⇒ increment `L`
///
/// ⇒ `2L + ln(7/8)`. A mask that instead scored the dead particle's (garbage,
/// possibly negative-count) state would miss this by more than the tolerance,
/// and one that failed to clear after resampling would lose the second `L`.
#[test]
fn one_recoverable_particle_error_is_masked_not_whole_theta_rejection() {
    let ll = run(recoverable_foi_rate())
        .expect("one particle's recoverable error must not fail the whole evaluation (gh#367)");
    assert!(
        ll.is_finite(),
        "the surviving particles carry a finite likelihood; masking the dead \
         particle must leave the θ score finite, got {ll}"
    );

    let l_alive = poisson_logpmf(Y0, Y0);
    let expected = 2.0 * l_alive + (N_PARTICLES as f64 - 1.0).ln() - (N_PARTICLES as f64).ln();
    assert!(
        (ll - expected).abs() < 1e-9,
        "expected exactly one masked particle: 2L + ln(7/8) = {expected}, got {ll} \
         (L = {l_alive})"
    );
    // Sanity: the masked particle really does move the answer — an
    // all-8-alive run would score `2L`, which this must NOT equal.
    assert!(
        (ll - 2.0 * l_alive).abs() > 1e-3,
        "if no particle were masked the score would be 2L = {}; got {ll}",
        2.0 * l_alive
    );
}

/// Negative control. Same model, same noise, same drained particle — only the
/// error CLASS differs: `NegativePropensity` is not per-particle recoverable
/// (`SimError::is_per_particle_recoverable() == false`), so it must still
/// propagate out of the filter rather than be swallowed by the mask.
#[test]
fn nonrecoverable_particle_error_still_tears_the_evaluation_down() {
    let err = run(nonrecoverable_foi_rate())
        .expect_err("a non-recoverable error must NOT be absorbed by the death mask");
    assert!(
        !err.is_per_particle_recoverable(),
        "control must exercise the non-recoverable branch; got {err}"
    );
    assert!(
        matches!(err, SimError::NegativePropensity { .. }),
        "expected the negative-rate error to propagate verbatim, got {err}"
    );
}
