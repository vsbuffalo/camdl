//! IF2 perturbs each estimated parameter at the times its step names
//! (`IF2Config::steps`): a `PerturbAt::TimeZeroOnly` parameter at t = 0 and
//! never again in the iteration, a `PerturbAt::EveryObservation` one at t = 0
//! and before every observation. And `run_if2` refuses steps that do not
//! match the estimated parameters and the simplex groups.
//!
//! The observation model scores every particle 0, so every weight is equal,
//! IF2 never resamples (D77), and particle j's θ at one observation is its θ
//! at the previous one plus whatever perturbation came between.

use std::sync::Mutex;

use sim::{
    error::SimError,
    inference::{
        if2::{run_if2, EstimatedParam, IF2Config, If2Step, SimplexGroup, Transform, ZSd},
        traits::{ObservationModel, ProcessModel},
        ParticleState,
    },
    rng::StatefulRng,
};

const N_OBS: usize = 4;
const N_PARTICLES: usize = 50;
const START: f64 = 1.0;

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

/// Scores every particle 0 and records the (θ₀, θ₁) it is handed, in call
/// order: observation by observation, particle by particle.
struct Flat {
    seen: Mutex<Vec<(f64, f64)>>,
}

impl ObservationModel<ParticleState> for Flat {
    fn log_likelihood(&self, _state: &ParticleState, _obs_idx: usize, params: &[f64]) -> f64 {
        self.seen.lock().unwrap().push((params[0], params[1]));
        0.0
    }
    fn n_observations(&self) -> usize { N_OBS }
    fn obs_time(&self, obs_idx: usize) -> f64 { (obs_idx + 1) as f64 }
}

fn spec(name: &str, index: usize) -> EstimatedParam {
    EstimatedParam {
        name: name.into(), index, initial: START,
        transform: Transform::for_domain(1e-3, 1e3, true).expect("domain"),
    }
}

fn config(steps: Vec<If2Step>, simplex_groups: Vec<SimplexGroup>) -> IF2Config {
    IF2Config {
        n_particles: N_PARTICLES,
        n_iterations: 1,
        cooling: sim::inference::if2::CoolingSchedule { fraction: 1.0, target_iters: 50 },
        dt: 1.0,
        t_start: 0.0,
        simplex_groups,
        steps,
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    }
}

fn sd(x: f64) -> ZSd { ZSd::new(x).unwrap() }

#[test]
fn a_time_zero_only_step_perturbs_at_t0_and_at_no_observation() {
    let steps = vec![If2Step::time_zero_only(sd(0.1)), If2Step::every_observation(sd(0.1))];
    let obs = Flat { seen: Mutex::new(Vec::new()) };
    run_if2(&NoDynamics, &obs, &[START, START], &[spec("a", 0), spec("b", 1)],
        &config(steps, vec![]), 3).expect("IF2 run");
    let seen = obs.seen.into_inner().unwrap();
    assert_eq!(seen.len(), N_OBS * N_PARTICLES, "one call per particle per observation");
    let at = |k: usize, j: usize| seen[k * N_PARTICLES + j];
    for j in 0..N_PARTICLES {
        assert_ne!(at(0, j).0, START, "particle {j}: the t = 0 perturbation did not move `a`");
        for k in 1..N_OBS {
            assert_eq!(at(k, j).0.to_bits(), at(0, j).0.to_bits(),
                "particle {j}: `a` is perturbed at t = 0 only, and moved before observation {k}");
            // The control: the same run perturbs `b` before every observation.
            assert_ne!(at(k, j).1, at(k - 1, j).1,
                "particle {j}: `b` is perturbed before every observation, and did not move \
                 before observation {k}");
        }
    }
}

fn refusal(steps: Vec<If2Step>, groups: Vec<SimplexGroup>) -> String {
    let obs = Flat { seen: Mutex::new(Vec::new()) };
    match run_if2(&NoDynamics, &obs, &[START, START], &[spec("a", 0), spec("b", 1)],
        &config(steps, groups), 3)
    {
        Err(SimError::Validation(msg)) => msg,
        Err(other) => panic!("expected a validation refusal, got {other}"),
        Ok(_) => panic!("expected a validation refusal, the run succeeded"),
    }
}

#[test]
fn run_if2_refuses_steps_that_do_not_match_the_parameters() {
    let single = If2Step::every_observation(sd(0.1));
    let msg = refusal(vec![single], vec![]);
    assert!(msg.contains("1 steps for 2 estimated parameters"), "{msg}");
    let msg = refusal(vec![single, If2Step::SimplexMember], vec![]);
    assert!(msg.contains("`b`") && msg.contains("no simplex group contains it"), "{msg}");
    let group = SimplexGroup { indices: vec![0, 1], sds: vec![sd(0.1), sd(0.1)] };
    let msg = refusal(vec![If2Step::SimplexMember, single], vec![group]);
    assert!(msg.contains("`b`") && msg.contains("also has a step of its own"), "{msg}");
}
