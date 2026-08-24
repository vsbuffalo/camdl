//! `run_if2` must start from each `EstimatedParam::initial`, not from
//! `base_params`.
//!
//! This is the engine-level, threshold-free guard for the 2026-04-18 incident
//! (`docs/dev/incidents/2026-04-18-if2-ignored-per-chain-initial.md`): the
//! runner built per-chain random starts, handed them to `run_if2` as
//! `if2_params[i].initial`, and the engine then seeded every chain from
//! `base_params` — so 64 "random starts" were one start, diverging only
//! through their per-chain RNG streams.
//!
//! `crates/cli/tests/synthetic_fit_grid.rs` guards the runner→engine wiring
//! end to end, but it can only observe the *filter mean after one full
//! pass*, so its discriminating power depends on how far a pass moves the
//! swarm — which is exactly the kind of thing an inference change perturbs
//! (gh#365 moved the perturbation ahead of the process step, which removed a
//! per-observation post-selection noise injection and retuned that distance).
//! The property itself does not depend on any of that, so assert it directly
//! and exactly: identical `base_params`, identical seed, identical config,
//! differing ONLY in `.initial` must produce different output; identical
//! `.initial` must reproduce bit-for-bit.

use std::sync::Mutex;

use sim::{
    error::SimError,
    inference::{
        if2::{run_if2, EstimatedParam, IF2Config, Transform},
        traits::{ObservationModel, ProcessModel},
        ParticleState,
    },
    rng::StatefulRng,
};

const THETA_IDX: usize = 0;
const N_OBS: usize = 4;

/// Records θ into the state so the observation model can score it. Keeps the
/// test independent of any particular epidemic model: the property under test
/// is about parameter plumbing, not dynamics.
struct ThetaProcess;

impl ProcessModel for ThetaProcess {
    type State = ParticleState;
    type Scratch = ();

    fn n_compartments(&self) -> usize { 1 }
    fn n_transitions(&self) -> usize { 1 }
    fn initial_state_draw(
        &self, _params: &[f64], _rng: &mut StatefulRng,
    ) -> Result<ParticleState, SimError> {
        Ok(ParticleState::new(1, 1, 0))
    }
    fn step(
        &self,
        state: &mut ParticleState,
        params: &[f64],
        _t: f64,
        _dt: f64,
        _per_eval: Option<&[f64]>,
        _rng: &mut StatefulRng,
        _scratch: &mut (),
        _due: &[usize],
    ) -> Result<(), SimError> {
        state.counts[0] = (params[THETA_IDX] * 1e6).round() as i64;
        Ok(())
    }
    fn new_scratch(&self) {}
}

/// Gaussian-ish score pulling θ toward `target`, so the swarm is under real
/// selection rather than drifting freely.
struct PullToward {
    target: f64,
    seen: Mutex<Vec<f64>>,
}

impl ObservationModel<ParticleState> for PullToward {
    fn log_likelihood(&self, state: &ParticleState, _obs_idx: usize, params: &[f64]) -> f64 {
        self.seen.lock().unwrap().push(params[THETA_IDX]);
        let d = state.counts[0] as f64 / 1e6 - self.target;
        -0.5 * d * d / (0.25 * 0.25)
    }
    fn n_observations(&self) -> usize { N_OBS }
    fn obs_time(&self, obs_idx: usize) -> f64 { (obs_idx + 1) as f64 }
}

fn config() -> IF2Config {
    IF2Config {
        n_particles: 32,
        n_iterations: 2,
        cooling_fraction: 0.9,
        cooling_target_iters: 50,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    }
}

fn spec(initial: f64) -> Vec<EstimatedParam> {
    vec![EstimatedParam {
        name: "theta".into(),
        index: THETA_IDX,
        initial,
        rw_sd: 0.05,
        transform: Transform::Log { lo: 0.01, hi: 10.0 },
        lower: 0.01,
        upper: 10.0,
        perturb_only_at_t0: false,
        rw_sd_auto: false,
    }]
}

/// `base_params` deliberately disagrees with every `.initial` below, so a run
/// that seeded from `base_params` would be identical across all of them.
const BASE: f64 = 1.0;

fn run(initial: f64, seed: u64) -> (f64, f64) {
    let process = ThetaProcess;
    let obs = PullToward { target: 2.0, seen: Mutex::new(Vec::new()) };
    let r = run_if2(&process, &obs, &[BASE], &spec(initial), &config(), seed)
        .expect("IF2 run");
    let first_seen = obs.seen.into_inner().unwrap()[0];
    (r.mle[THETA_IDX], first_seen)
}

#[test]
fn if2_seeds_each_run_from_its_own_initial_not_base_params() {
    // Two "chains": same base_params, same seed, same config — only `.initial`
    // differs. If the engine seeded from `base_params` these would be
    // bit-identical.
    let (mle_lo, first_lo) = run(0.2, 4242);
    let (mle_hi, first_hi) = run(5.0, 4242);

    assert_ne!(
        first_lo.to_bits(), first_hi.to_bits(),
        "the very first θ the observation model was handed is identical \
         ({:.12}) for .initial = 0.2 and .initial = 5.0 — run_if2 seeded from \
         base_params ({}) and ignored EstimatedParam::initial \
         (docs/dev/incidents/2026-04-18-if2-ignored-per-chain-initial.md)",
        first_lo, BASE,
    );
    assert!(
        (mle_lo - mle_hi).abs() > 1e-9,
        "two runs differing only in .initial (0.2 vs 5.0) produced the same \
         MLE ({:.12} vs {:.12}); .initial is not authoritative",
        mle_lo, mle_hi,
    );

    // Each run's first handed-out θ must be its OWN initial perturbed, i.e.
    // nearer its own start than the other's. rw_sd is 0.05 on the log scale,
    // so one perturbation cannot cross the 0.2 ↔ 5.0 gap.
    assert!(
        (first_lo.ln() - 0.2_f64.ln()).abs() < (first_lo.ln() - 5.0_f64.ln()).abs(),
        "the .initial = 0.2 run started at {:.6}, closer to 5.0 than to 0.2",
        first_lo,
    );
    assert!(
        (first_hi.ln() - 5.0_f64.ln()).abs() < (first_hi.ln() - 0.2_f64.ln()).abs(),
        "the .initial = 5.0 run started at {:.6}, closer to 0.2 than to 5.0",
        first_hi,
    );
}

#[test]
fn if2_is_reproducible_at_a_fixed_initial() {
    // Negative control for the test above: the difference it detects must come
    // from `.initial`, not from run-to-run nondeterminism.
    let (a_mle, a_first) = run(0.2, 4242);
    let (b_mle, b_first) = run(0.2, 4242);
    assert_eq!(a_mle.to_bits(), b_mle.to_bits(), "IF2 is not reproducible at a fixed seed");
    assert_eq!(a_first.to_bits(), b_first.to_bits(), "first handed-out θ is not reproducible");
}
