//! gh#646: IF2 must carry each particle's own θ from iteration m−1 into
//! iteration m, not reset every particle to the previous swarm mean.
//!
//! Ionides et al. (2015) Algorithm 1 perturbs Θ^F_{0,j} ~ h_0(θ | Θ^{m−1}_j; σ_m):
//! the j-th particle starts from its OWN θ. pomp does the same
//! (pomp 6.4 `R/mif2.R`: the full `paramMatrix` is carried across iterations;
//! the swarm mean goes only into `traces`).
//!
//! The swarm's spread is IF2's only memory of which way a likelihood ridge
//! runs. The observable here is a pure ridge: the likelihood depends on θ₁·θ₂
//! only, so along the ridge (log θ₁ − log θ₂) there is no selection at all and
//! the swarm random-walks. Carrying the swarm, the spread along the ridge grows
//! like √(total perturbation steps) across iterations; collapsing it to the mean
//! restarts the spread from zero every iteration, so it stays at one
//! iteration's worth no matter how many iterations run.

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

const N_OBS: usize = 4;
const N_PARTICLES: usize = 400;
const RW_SD: f64 = 0.05;

struct NoDynamics;

impl ProcessModel for NoDynamics {
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
        &self, _state: &mut ParticleState, _params: &[f64], _t: f64, _dt: f64,
        _per_eval: Option<&[f64]>, _rng: &mut StatefulRng, _scratch: &mut (), _due: &[usize],
    ) -> Result<(), SimError> {
        Ok(())
    }
    fn new_scratch(&self) {}
}

/// log L = −½ (log θ₁ + log θ₂)² / 0.1²: a ridge along θ₁·θ₂ = 1. Records every
/// (θ₁, θ₂) it is handed, in call order.
struct Ridge {
    seen: Mutex<Vec<(f64, f64)>>,
}

impl ObservationModel<ParticleState> for Ridge {
    fn log_likelihood(&self, _state: &ParticleState, _obs_idx: usize, params: &[f64]) -> f64 {
        self.seen.lock().unwrap().push((params[0], params[1]));
        let s = params[0].ln() + params[1].ln();
        -0.5 * s * s / (0.1 * 0.1)
    }
    fn n_observations(&self) -> usize { N_OBS }
    fn obs_time(&self, obs_idx: usize) -> f64 { (obs_idx + 1) as f64 }
}

fn spec(name: &str, index: usize) -> EstimatedParam {
    EstimatedParam {
        name: name.into(),
        index,
        initial: 1.0,
        rw_sd: RW_SD,
        transform: Transform::Log { lo: 1e-4, hi: 1e4 },
        lower: 1e-4,
        upper: 1e4,
        perturb_only_at_t0: false,
        rw_sd_auto: false,
    }
}

/// SD of (log θ₁ − log θ₂) over the θ's handed to the observation model at the
/// final observation of the final iteration.
fn ridge_spread(n_iterations: usize, seed: u64) -> f64 {
    let config = IF2Config {
        n_particles: N_PARTICLES,
        n_iterations,
        // No cooling to speak of: the property is about carrying, not cooling.
        cooling_fraction: 0.999,
        cooling_target_iters: 50,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let obs = Ridge { seen: Mutex::new(Vec::new()) };
    run_if2(&NoDynamics, &obs, &[1.0, 1.0], &[spec("a", 0), spec("b", 1)], &config, seed)
        .expect("IF2 run");
    let seen = obs.seen.into_inner().unwrap();
    assert_eq!(seen.len(), n_iterations * N_OBS * N_PARTICLES, "one call per particle per obs");
    let last = &seen[seen.len() - N_PARTICLES..];
    let d: Vec<f64> = last.iter().map(|(a, b)| a.ln() - b.ln()).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d.len() as f64).sqrt()
}

#[test]
fn if2_swarm_spread_along_a_ridge_accumulates_across_iterations() {
    // Along the ridge each step adds variance 2·RW_SD² (two independent
    // log-scale perturbations), with no selection. One iteration is 1 + N_OBS
    // perturbation steps, so after M iterations a carried swarm has
    // SD ≈ RW_SD·√(2·M·(1+N_OBS)) at its last observation; a swarm reset to the
    // mean every iteration has SD ≈ RW_SD·√(2·(1+N_OBS)) regardless of M.
    // Resampling drift shrinks both somewhat; the √16 = 4× gap leaves a wide
    // margin for the factor-2 assertion.
    let one = ridge_spread(1, 7);
    let many = ridge_spread(16, 7);
    let one_iter_expected = RW_SD * (2.0 * (1 + N_OBS) as f64).sqrt();
    assert!(
        one > 0.5 * one_iter_expected && one < 1.5 * one_iter_expected,
        "sanity: one-iteration ridge spread {one:.4} is far from the random-walk \
         expectation {one_iter_expected:.4}; the test's premise does not hold",
    );
    assert!(
        many > 2.0 * one,
        "after 16 IF2 iterations the swarm's spread along a flat likelihood ridge is \
         {many:.4}, versus {one:.4} after one — it did not accumulate, so each \
         iteration restarted from a single point (the previous swarm mean) instead \
         of carrying each particle's θ forward (gh#646)",
    );
}
