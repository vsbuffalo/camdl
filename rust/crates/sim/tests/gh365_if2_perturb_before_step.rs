//! gh#365 — IF2 must perturb θ BEFORE the process step, so the same
//! perturbed θ drives both the simulation of X_n and the measurement
//! density g(y_n | X_n; θ).
//!
//! Normative source, Ionides, Nguyen, Atchadé, Stoev & King (2015),
//! "Inference for dynamic and latent variable models via iterated,
//! perturbed Bayes maps", PNAS 112(3):719–724, doi:10.1073/pnas.1410597112,
//! Algorithm 1 (IF2), inner loop:
//!
//! ```text
//!   Θ^{P,m}_{n,j} ~ h_n(θ | Θ^{F,m}_{n-1,j}, σ_m)                 [perturb]
//!   X^{P,m}_{n,j} ~ f_{X_n|X_{n-1}}(x_n | X^{F,m}_{n-1,j}; Θ^{P,m}_{n,j})  [propagate]
//!   w^m_{n,j}     = f_{Y_n|X_n}(y*_n | X^{P,m}_{n,j}; Θ^{P,m}_{n,j})       [weight]
//! ```
//!
//! The SAME superscript-P θ appears in the process draw and in the weight.
//! pomp implements exactly this ordering — `pomp:::mif2_pfilter`
//! (`R/mif2.R`, pomp 6.4.0.2, kingaa/pomp@0eaf3c01):
//!
//! ```r
//!   for (nt in seq_len(ntimes)) {
//!     pmag <- cooling.fn(nt,mifiter)$alpha*rw.sd[,nt]
//!     params <- .Call(P_randwalk_perturbation,params,pmag)   # perturb
//!     tparams <- partrans(object,params,dir="fromEst",.gnsi=gnsi)
//!     if (nt == 1L) x <- rinit(object,params=tparams)
//!     X <- rprocess(object,x0=x,...,params=tparams,...)      # propagate
//!     weights <- dmeasure(object,...,x=X,params=tparams,...) # weight
//! ```
//!
//! Before the fix camdl ran propagate → perturb → weight, so the state was
//! generated at Θ^F_{n-1} while the weight was scored at Θ^P_n. For a
//! parameter living in BOTH the process and the observation model that is a
//! genuine coupling error, not a phase offset.
//!
//! The test is structural and exact — no statistics. A mock `ProcessModel`
//! stamps the θ it was stepped with into the particle state (as raw f64
//! bits, so the round-trip is lossless); the observation model reads that
//! stamp back and compares it against the θ it is handed for weighting.
//! Equality is the Algorithm-1 coupling. The likelihood is written so the
//! mismatch is also visible as a log-likelihood penalty, which is the harm
//! a shared process/measurement parameter actually suffers.

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
const N_OBS: usize = 5;

/// Lossless f64 ↔ i64 round-trip through `ParticleState::counts`.
fn stamp(x: f64) -> i64 {
    x.to_bits() as i64
}
fn unstamp(v: i64) -> f64 {
    f64::from_bits(v as u64)
}

/// Mock process whose only action is to record the θ it was stepped with.
/// `try_compiled_model()` stays `None`, which is the supported no-model
/// timeline path (`ExactInferenceTimeline::build(None, ..)`).
struct ThetaStampProcess;

impl ProcessModel for ThetaStampProcess {
    type State = ParticleState;
    type Scratch = ();

    fn n_compartments(&self) -> usize {
        1
    }
    fn n_transitions(&self) -> usize {
        1
    }
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
        _due_effects: &[usize],
    ) -> Result<(), SimError> {
        state.counts[0] = stamp(params[THETA_IDX]);
        Ok(())
    }
    fn new_scratch(&self) {}
}

/// Records (θ that drove the step, θ handed to the weight) for every
/// particle at every observation, and scores agreement: 0 when the two
/// coincide (Algorithm 1), a fixed penalty when they do not.
struct CouplingWitnessObs {
    pairs: Mutex<Vec<(usize, f64, f64)>>,
}

const MISMATCH_PENALTY: f64 = -1000.0;

impl ObservationModel<ParticleState> for CouplingWitnessObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, params: &[f64]) -> f64 {
        let stepped = unstamp(state.counts[0]);
        let weighted = params[THETA_IDX];
        self.pairs.lock().unwrap().push((obs_idx, stepped, weighted));
        if stepped == weighted {
            0.0
        } else {
            MISMATCH_PENALTY
        }
    }
    fn n_observations(&self) -> usize {
        N_OBS
    }
    fn obs_time(&self, obs_idx: usize) -> f64 {
        (obs_idx + 1) as f64
    }
}

#[test]
fn if2_weights_the_same_theta_that_drove_the_step() {
    let process = ThetaStampProcess;
    let obs = CouplingWitnessObs {
        pairs: Mutex::new(Vec::new()),
    };

    let base_params = vec![0.5_f64];
    let if2_params = vec![EstimatedParam {
        name: "theta".into(),
        index: THETA_IDX,
        initial: 0.5,
        // Big enough that a stale θ is unmistakable, small enough that the
        // Log transform never reaches its clamp (bounds are 3 orders wide).
        rw_sd: 0.2,
        transform: Transform::Log { lo: 0.01, hi: 10.0 },
        lower: 0.01,
        upper: 10.0,
        ivp: false,
        rw_sd_auto: false,
    }];

    let config = IF2Config {
        n_particles: 16,
        n_iterations: 3,
        cooling_fraction: 0.9,
        cooling_target_iters: 50,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let result = run_if2(&process, &obs, &base_params, &if2_params, &config, 42)
        .expect("mock IF2 run must succeed");

    let pairs = obs.pairs.into_inner().unwrap();
    assert_eq!(
        pairs.len(),
        config.n_particles * N_OBS * config.n_iterations,
        "one witness record per particle per observation per iteration"
    );

    // Negative control: the test is only meaningful if θ actually moves.
    // With rw_sd = 0 every pair would trivially agree.
    let distinct = pairs
        .iter()
        .map(|&(_, stepped, _)| stepped.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert!(
        distinct.len() > 1,
        "vacuous test: θ never varied across particles/steps ({} distinct values)",
        distinct.len()
    );

    let mismatches: Vec<_> = pairs
        .iter()
        .filter(|&&(_, stepped, weighted)| stepped != weighted)
        .collect();
    assert!(
        mismatches.is_empty(),
        "gh#365: {}/{} particle-observations were weighted at a θ different from the θ \
         that generated their state — IF2 must perturb BEFORE the process step \
         (Ionides et al. 2015 Algorithm 1). First offender: obs {} stepped θ={:.12} \
         but weighted θ={:.12}",
        mismatches.len(),
        pairs.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2,
    );

    // The same fact, expressed as the harm: a parameter shared between the
    // process and the measurement is scored against a state generated at a
    // different value, and the log-likelihood pays for it.
    assert_eq!(
        result.last_loglik, 0.0,
        "coupled θ scores exactly 0 per observation; got {} (≈ {} mismatched observations)",
        result.last_loglik,
        result.last_loglik / MISMATCH_PENALTY
    );
}
