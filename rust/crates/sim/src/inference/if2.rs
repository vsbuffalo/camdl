//! Iterated Filtering (IF2) — MLE via sequential Monte Carlo.
//!
//! Ionides, Bretó & King (2006), Ionides et al. (2015).
//!
//! IF2 runs a particle filter with perturbed parameters: at each
//! observation time, each particle's parameter vector is jittered by
//! a random walk with shrinking variance. Over M iterations, the
//! perturbation scale σ → 0 and the particle swarm concentrates
//! around the MLE.
//!
//! Ionides et al. (2015) Algorithm 1 is the normative reference for what
//! one iteration does, in what order:
//!
//! ```text
//!   Θ^F_{0,j} ~ h_0(θ | Θ^{m-1}_j; σ_m)                      [t=0 perturbation]
//!   X^F_{0,j} ~ f_{X_0}(x_0; Θ^F_{0,j})
//!   for n in 1..N:
//!     Θ^P_{n,j} ~ h_n(θ | Θ^F_{n-1,j}; σ_m)                  [perturb]
//!     X^P_{n,j} ~ f_{X_n|X_{n-1}}(· | X^F_{n-1,j}; Θ^P_{n,j})[propagate]
//!     w_{n,j}   = f_{Y_n|X_n}(y*_n | X^P_{n,j}; Θ^P_{n,j})   [weight]
//!     resample (Θ, X) jointly
//! ```
//!
//! Two properties of that order are load-bearing and were each once
//! wrong. The perturbation precedes the process step, so one Θ^P_n drives
//! both the simulation and the measurement density (gh#365). And x₀ is
//! drawn per particle from *that particle's* t=0-perturbed θ, not once
//! from the swarm mean — without it a parameter reaching the model only
//! through `initial_conditions` gets no initial-state spread, so the
//! weights never select on it (gh#364).
//!
//! Key property: IF2 finds the MLE without computing the transition
//! density — it only needs the simulator (process model) and the
//! observation log-likelihood (dmeasure). This makes it compatible
//! with any simulation backend.

use std::time::Instant;

use rayon::prelude::*;

use crate::rng::StatefulRng;
use crate::error::SimError;
use crate::schedule::Cursor;
use super::degeneracy::{check_pf_degeneracy, check_iteration_budget, window_substep_cost, pf_bail_error};
use super::traits::{ProcessModel, ObservationModel};
use super::types::{ParticleState, log_sum_exp, normalize_log_weights, ess_from_log_weights, LOG_PROB_FLOOR, init_particle_rngs};
use super::resampling::systematic_resample;

// `Transform` and `EstimatedParam` are defined in `types.rs` (shared by all
// inference algorithms). Re-exported here so existing import paths via
// `inference::if2::EstimatedParam` continue to work.
pub use super::types::{Transform, EstimatedParam};

/// A group of parameters with a joint simplex constraint (sum to 1).
/// Uses barycentric (log-ratio + softmax) transform, matching pomp's
/// `parameter_trans(barycentric = ...)`. All members are perturbed
/// jointly in log-ratio space; softmax inverse guarantees sum = 1.
#[derive(Clone, Debug)]
pub struct SimplexGroup {
    /// Indices into the params array for each member.
    pub indices: Vec<usize>,
    /// Per-member rw_sd on the log-ratio scale.
    pub rw_sds: Vec<f64>,
}

impl SimplexGroup {
    /// Forward transform: fractions → log-ratios.
    /// z_i = log(x_i / sum(x)), matching pomp's to_log_barycentric.
    pub fn to_log_barycentric(&self, params: &[f64]) -> Vec<f64> {
        let fracs: Vec<f64> = self.indices.iter()
            .map(|&i| params[i].max(LOG_PROB_FLOOR))
            .collect();
        let sum: f64 = fracs.iter().sum();
        fracs.iter().map(|&f| (f / sum).max(LOG_PROB_FLOOR).ln()).collect()
    }

    /// Inverse transform: log-ratios → fractions via softmax.
    /// Numerically stable (max-subtraction trick). Guarantees sum = 1.
    pub fn from_log_barycentric(z: &[f64]) -> Vec<f64> {
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_z: Vec<f64> = z.iter().map(|&zi| (zi - max_z).exp()).collect();
        let sum: f64 = exp_z.iter().sum();
        exp_z.iter().map(|&e| e / sum).collect()
    }

    /// Perturb in log-ratio space and apply softmax inverse.
    /// Writes the new fractions directly into particle_params.
    pub fn perturb(
        &self,
        particle_params: &mut [f64],
        rng: &mut crate::rng::StatefulRng,
        cooling_now: f64,
    ) {
        let log_ratios = self.to_log_barycentric(particle_params);
        let perturbed: Vec<f64> = log_ratios.iter()
            .zip(&self.rw_sds)
            .map(|(&z, &sd)| z + rng.normal() * sd * cooling_now)
            .collect();
        let fracs = Self::from_log_barycentric(&perturbed);
        for (j, &idx) in self.indices.iter().enumerate() {
            particle_params[idx] = fracs[j];
        }
    }
}

/// IF2 configuration.
pub struct IF2Config {
    pub n_particles: usize,
    pub n_iterations: usize,
    /// Cooling schedule: after `cooling_target_iters` iterations the perturbation
    /// SD is `cooling_fraction` of its initial value, then continues to cool —
    /// pomp's geometric `cooling.fraction.50` (with `cooling_target_iters` playing
    /// pomp's fixed 50). Each iteration consumes `(1 + n_obs)` filtering steps (a
    /// t=0 perturbation plus one per observation); the per-step factor is
    /// `per_step_cooling_factor` (exponent 1, verified against pomp 6.4 source).
    pub cooling_fraction: f64,
    /// Number of iterations over which the cooling fraction applies.
    /// pomp default: 50 (cooling.fraction.50).
    pub cooling_target_iters: usize,
    pub dt: f64,
    /// Simulation start time (before first observation).
    pub t_start: f64,
    /// Simplex parameter groups (barycentric transform). Members are
    /// perturbed jointly in log-ratio space with softmax inverse.
    pub simplex_groups: Vec<SimplexGroup>,
    /// IC-free inference: still weight and resample at the first
    /// observation (pinning x₀ given y₁) but don't accumulate that
    /// step's log-sum-exp into the returned log-likelihood. Requires
    /// per-particle spread at t=0, typically from a `perturb_only_at_t0`
    /// estimated parameter: the t=0 perturbation moves each particle's θ and each
    /// particle then draws its own x₀ from that θ (gh#364), so the
    /// first reweight has something to discriminate between. Without
    /// such a parameter the first reweight is a no-op and ic-free
    /// degenerates to silently dropping y₁ — the fit-config layer
    /// rejects that case.
    /// See docs/dev/proposals/2026-04-18-ic-free-inference.md.
    pub skip_first_obs_from_loglik: bool,

    /// gh#241. Deterministic per-call compute budget (max cumulative
    /// particle-substeps before bailing with `PFIterationBudget`). Default
    /// `degeneracy::ITER_BUDGET`. Reproducible across machines — it replaced
    /// a machine-speed-dependent wall-clock timeout. Not part of run identity.
    pub max_substeps: u64,
}

impl super::traits::InferenceConfig for IF2Config {
    fn n_particles(&self) -> usize { self.n_particles }
    fn dt(&self) -> f64 { self.dt }
}

/// Per-filtering-step cooling factor `c`: the perturbation SD at global step `s`
/// is `initial · c^s`. Chosen so that after `cooling_target_iters` complete
/// iterations — each consuming `(1 + n_obs)` steps (one t=0 perturbation plus one
/// per observation; gh#audit-M2) — the SD reaches `cooling_fraction` of its
/// initial value.
///
/// This is pomp's geometric `cooling.fraction.50` schedule, with
/// `cooling_target_iters` playing pomp's fixed "50". Verified against pomp 6.4
/// source: `pomp:::mif2_cooling` returns `alpha = cooling.fraction.50^(m/50)` at
/// the end of iteration `m`, and `pomp:::mif2_pfilter` perturbs with
/// `pmag = alpha · rw.sd` — so the exponent is **1**. pomp also returns
/// `gamma = alpha²` but does NOT use it for the perturbation; taking the squared
/// value here was the origin of the earlier `2.0` (gh#363), which cooled twice as
/// fast as pomp (fraction reached at the midpoint, `fraction²` at the endpoint).
pub fn per_step_cooling_factor(
    cooling_fraction: f64,
    cooling_target_iters: usize,
    n_obs: usize,
) -> f64 {
    let total_target_steps = cooling_target_iters as f64 * (1 + n_obs) as f64;
    cooling_fraction.powf(1.0 / total_target_steps)
}

/// The cooling multiplier on the perturbation SD at the END of IF2 iteration
/// `iter` (1-based): `cooling_fraction^(iter / cooling_target_iters)`. It reaches
/// `cooling_fraction` exactly at `iter = cooling_target_iters` and continues
/// cooling past it — matching pomp's `alpha`. The `(1 + n_obs)` step granularity
/// cancels, so the per-iteration multiplier is independent of `n_obs`.
pub fn cooling_multiplier_at_iter(
    cooling_fraction: f64,
    cooling_target_iters: usize,
    n_obs: usize,
    iter: usize,
) -> f64 {
    let steps_per_iter = (1 + n_obs) as f64;
    per_step_cooling_factor(cooling_fraction, cooling_target_iters, n_obs)
        .powf(iter as f64 * steps_per_iter)
}

/// Result of one IF2 iteration.
#[derive(Clone, Debug)]
pub struct IF2IterResult {
    pub iteration: usize,
    /// True model log-likelihood P(data | θ̂) at the filter mean params,
    /// evaluated by a clean PF with no perturbation.
    /// Populated post-hoc by the caller (e.g., every N iterations). NaN when not evaluated.
    pub loglik: f64,
    /// IF2 perturbed-model log-likelihood (internal diagnostic only).
    /// Computed during IF2 with heterogeneous particle params. Peaks early
    /// due to perturbation smoothing, then declines as cooling progresses.
    /// NOT useful for model assessment or convergence — use `loglik` instead.
    pub if2_perturbed_loglik: f64,
    /// Parameter means across particles at end of this iteration.
    pub param_means: Vec<f64>,
    /// Per-parameter diagnostics, indexed by position in if2_params.
    pub param_diag: Vec<ParamIterDiag>,
}

/// Per-parameter diagnostics for one IF2 iteration.
#[derive(Clone, Debug)]
pub struct ParamIterDiag {
    pub param_index: usize,
    /// Weighted-variance selection ratio, averaged across observations.
    pub weighted_var_ratio: f64,
    /// Perturbation-to-cloud ratio: rw_sd_effective / sd(θ_k before perturbation).
    /// q ≪ 1 = perturbation too timid (late cooling).
    /// q ≫ 1 = perturbation dominates cloud (reinitializing).
    pub q_ratio: f64,
    /// Effective rw_sd at this iteration (after cooling).
    pub effective_rw_sd: f64,
    /// Fraction of particle-steps that hit the bounds clamp this iteration.
    /// >0.1 means rw_sd is too large — particles are being pushed out of bounds.
    pub clamp_fraction: f64,
}

/// Result of the full IF2 run.
pub struct IF2Result {
    pub iterations: Vec<IF2IterResult>,
    /// MLE estimate: param means from the best-loglik iteration.
    pub mle: Vec<f64>,
    /// Best log-likelihood across all iterations.
    /// Initially set to the perturbed loglik by the IF2 engine.
    /// The caller should overwrite this with the true (PF-evaluated) loglik
    /// after populating `IF2IterResult::loglik` on the iterations.
    pub final_loglik: f64,
    /// Last iteration's perturbed log-likelihood (IF2 engine diagnostic).
    pub last_loglik: f64,
}

/// Observation for IF2 (same as particle_filter::Observation).
/// Kept for backward compatibility with CLI code that constructs observations.
#[derive(Clone)]
pub struct Observation {
    pub time: f64,
    pub value: f64,
}

/// Run IF2.
///
/// # Arguments
/// * `model` — compiled model
/// * `base_params` — starting parameter values (full vector)
/// * `if2_params` — parameters to estimate (subset of base_params)
/// * `observations` — data sorted by time
/// * `config` — IF2 settings
/// * `step_fn` — chain-binomial step function
/// * `project_fn` — extract projected quantity from particle state
/// * `obs_loglik_fn` — observation log-likelihood (takes projected, observed, params)
/// * `seed` — base RNG seed
/// Optional callback invoked after each IF2 iteration.
/// Arguments: `(iteration_index, log_likelihood, param_means)`.
///
/// The `log_likelihood` value is the IF2 in-run perturbed loglik
/// (matches `IF2IterResult.if2_perturbed_loglik`); the post-hoc
/// clean-PF re-evaluation that populates `IF2IterResult.loglik`
/// runs after `run_if2_with_progress` returns and is not visible
/// here. `param_means` is the iteration's filter-mean estimate of
/// every estimated parameter, in the same order as `if2_params`.
/// The runner uses this to stream a per-iteration trace row to
/// `chain_N/parameter_traces.tsv` so users can `tail -f` long
/// scout runs and watch parameters move in real time.
pub type ProgressCallback<'a> = Option<&'a dyn Fn(usize, f64, &[f64])>;

pub fn run_if2<P: ProcessModel<State = ParticleState>>(
    process: &P,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    base_params: &[f64],
    if2_params: &[EstimatedParam],
    config: &IF2Config,
    seed: u64,
) -> Result<IF2Result, SimError> {
    run_if2_with_progress(process, obs_model, base_params, if2_params, config,
        seed, None)
}

pub fn run_if2_with_progress<P: ProcessModel<State = ParticleState>>(
    process: &P,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    base_params: &[f64],
    if2_params: &[EstimatedParam],
    config: &IF2Config,
    seed: u64,
    on_iteration: ProgressCallback,
) -> Result<IF2Result, SimError> {
    let n = config.n_particles;
    let n_int = process.n_compartments();
    let n_tr = process.n_transitions();
    let n_obs = obs_model.n_observations();
    // Per-Interval-stream `acc` bins (multi-cadence Phase 2a), sized from the
    // obs model (the process does not know `n_interval_streams`).
    let n_acc = obs_model.n_interval_streams();

    // Merged timeline spine: the EXACT policy clips each substep to the next
    // observation boundary (same idiom as the bootstrap PF). Constant across IF2
    // iterations, so built once; reproduces dt.min(obs_time - t) exactly. Substep
    // TIME stays accumulated (s*dt deferred, task #14).
    let obs_times: Vec<f64> = (0..n_obs).map(|i| obs_model.obs_time(i)).collect();

    // gh#216: scheduled interventions fire CURSOR-keyed off the timeline's effect
    // boundaries, so an off-grid observation re-tiling the Exact substep grid no
    // longer moves the firing instant. `ExactInferenceTimeline::build` runs the
    // two exact guards FIRST (no inference path can skip a guard — the gh#187
    // class), then gathers the cursor-keyed effect batches. Built ONCE here and
    // constant across IF2 iterations: a parametric `at [<param>]` schedule — whose
    // fire times would be per-particle and per-iteration — is refused by
    // `guard_attimesexpr_exact`, so every scheduled fire time is
    // `base_params`-independent. An off-grid scheduled fire time and always-active
    // events are the other unsupported Exact cases (refused / out of scope). See
    // particle_filter.rs for the same pattern.
    let timeline = crate::intervention::ExactInferenceTimeline::build(
        process.try_compiled_model(),
        base_params,
        config.t_start,
        config.dt,
        crate::boundary_times::ObsTimes::new(obs_times)?,
    )?;
    let schedule = timeline.schedule;
    let scheduled = timeline.effects;

    // Mutable copy of params — updated each iteration with the filter mean.
    // Start from `base_params` for non-estimated slots, then overwrite each
    // estimated slot with that `EstimatedParam`'s `.initial`. For
    // single-start fits `.initial == base_params[idx]` so this is a no-op;
    // for scout with per-chain random starts (or any caller that supplies
    // divergent `.initial` values per chain) this is what actually makes
    // IF2 start from the declared point. Before 2026-04-18 this was a bug
    // — chains supposedly starting from 64 random points all started from
    // the same `base_params` and only diverged via their per-chain RNG on
    // the first perturbation. See docs/dev/incidents/2026-04-18-if2-ignored-per-chain-initial.md.
    let mut current_params = base_params.to_vec();
    for spec in if2_params {
        if spec.index < current_params.len() {
            current_params[spec.index] = spec.initial;
        }
    }

    // Per-filtering-step cooling factor — pomp's geometric cooling.fraction.50
    // (exponent 1: the fraction is reached at `cooling_target_iters`, then cooling
    // continues; gh#363). Each iteration consumes (1 + n_obs) steps — one t=0
    // perturbation plus one per observation (gh#audit-M2). Formula lives in
    // `per_step_cooling_factor`, shared with the preflight preview so they can't
    // drift.
    let per_step_cooling =
        per_step_cooling_factor(config.cooling_fraction, config.cooling_target_iters, n_obs);

    let mut iterations = Vec::with_capacity(config.n_iterations);
    let mut global_step: u64 = 0; // total filtering steps across all iterations

    // gh#110. ESS history for the (deterministic) degeneracy watchdog. The
    // watchdog spans the entire IF2 run, not a single iteration: a
    // pathological init that walks the cooling trajectory through a
    // degeneracy region produces an ESS history that crosses iteration
    // boundaries. We push every obs window's ESS into one trace (across all
    // iterations) so the K-window detector can fire as soon as the cumulative
    // pattern is bad. `t0_if2` is a display-only diagnostic (how long a doomed
    // run took); it never gates the bail — gh#241 removed the machine-dependent
    // wall-clock watchdog in favor of the deterministic substep budget.
    let t0_if2 = Instant::now();
    let mut ess_history: Vec<f64> = Vec::with_capacity(config.n_iterations * n_obs);

    // Pre-allocate particle state, params, RNGs, and scratch buffers once.
    // Re-initialized from current_params at the start of each iteration.
    let mut states: Vec<ParticleState> = (0..n)
        .map(|_| ParticleState::new(n_int, n_tr, n_acc))
        .collect();
    let mut particle_params: Vec<Vec<f64>> = vec![vec![0.0; base_params.len()]; n];
    let mut scratches: Vec<P::Scratch> = (0..n)
        .map(|_| process.new_scratch())
        .collect();
    // Double-buffers for resampling (avoids clone allocation)
    let mut states_buf: Vec<ParticleState> = (0..n)
        .map(|_| ParticleState::new(n_int, n_tr, n_acc))
        .collect();
    let mut params_buf: Vec<Vec<f64>> = vec![vec![0.0; base_params.len()]; n];

    for iter in 0..config.n_iterations {

        // Re-initialize per-particle parameter vectors from current estimate
        for pp in &mut particle_params {
            pp.copy_from_slice(&current_params);
        }

        // IM1 fix (2026-04-19 inference review): per-particle RNG
        // streams via ChaCha8's stream counter. iter in the top
        // 32 bits, particle i in the bottom 32 — fits 2^32
        // iterations × 2^32 particles with room to spare.
        let stream_base = (iter as u64) << 32;
        let mut rngs = init_particle_rngs(seed, n, stream_base);
        let mut resample_rng = StatefulRng::new_stream(
            seed,
            stream_base | super::types::RESAMPLE_RNG_STREAM,
        );

        // Diagnostic accumulators (averaged across observation times)
        let n_if2_params = if2_params.len();
        let mut wvr_accum = vec![0.0_f64; n_if2_params];
        let mut q_k_accum = vec![0.0_f64; n_if2_params];
        let mut clamp_counts = vec![0_usize; n_if2_params];
        let mut diag_count = vec![0_usize; n_if2_params];

        // Build set of simplex member indices (perturbed jointly, skip in per-param loop)
        let simplex_member_indices: std::collections::HashSet<usize> = config.simplex_groups.iter()
            .flat_map(|g| g.indices.iter().copied())
            .collect();

        // Initial parameter perturbation (at t=0)
        {
            let cooling_now = per_step_cooling.powf(global_step as f64);

            // Simplex groups: perturb jointly in log-ratio space
            for group in &config.simplex_groups {
                for i in 0..n {
                    group.perturb(&mut particle_params[i], &mut rngs[i], cooling_now);
                }
            }

            for i in 0..n {
                for (pi, spec) in if2_params.iter().enumerate() {
                    if simplex_member_indices.contains(&spec.index) { continue; }
                    let current = particle_params[i][spec.index];
                    let sd = spec.transformed_sd(spec.rw_sd, current) * cooling_now;
                    let z = spec.to_transformed(current);
                    let new_val = spec.from_transformed(z + rngs[i].normal() * sd);
                    particle_params[i][spec.index] = new_val;
                    // Detect clamp activation (Log transform)
                    if let Transform::Log { lo, hi } = &spec.transform {
                        if (new_val - lo).abs() < 1e-10 || (new_val - hi).abs() < 1e-10 {
                            clamp_counts[pi] += 1;
                        }
                    }
                }
            }
            global_step += 1;
        }

        // X^F_{0,j} ~ f_{X_0}(·; Θ^F_{0,j}) — Ionides et al. (2015) Algorithm 1.
        // Runs AFTER the t=0 perturbation, once per particle, from THAT
        // particle's own θ. A parameter reaching the model only through
        // `initial_conditions` (a pure-IC `perturb_only_at_t0` param — `S0`,
        // `E0`, `I0`, or a simplex composition) has no other channel: it is
        // absent from every
        // rate and from the observation model, so unless it moves x₀ the
        // weights are independent of it, the resampling is a blind subsample,
        // and its filter mean drifts without ever being selected. Evaluating
        // the initial state once from the swarm mean and copying it to every
        // particle did exactly that (gh#364) — and silently, since
        // `ic_free = true` validates that a `perturb_only_at_t0` param exists
        // precisely to guarantee the t=0 spread this now delivers.
        //
        // Seam: `ProcessModel::initial_state_draw` is the existing producer of
        // x₀ and was already the one being called; the fix is to route each
        // particle through it rather than the swarm mean. PGAS's per-particle
        // `Binomial(N₀, θ)` draw (`pgas.rs::csmc_as`) is deliberately NOT
        // reused: it exists because PGAS needs a tractable initial-state
        // *density* p(x₀|θ) for the complete-data likelihood, it is built from
        // `IVPMapping`s that finite-difference a `&CompiledModel` (which
        // `ProcessModel` only optionally exposes), and it would add Monte-Carlo
        // variance that Algorithm 1 does not ask for. IF2 needs a draw, not a
        // density. pomp agrees: `mif2_pfilter` calls
        // `rinit(object, params=tparams)` on the per-particle perturbed
        // parameter matrix.
        //
        // Each particle draws from its OWN stream `rngs[j]`, which is the
        // spread `ic_free` requires. `initial_state_draw` consumes nothing
        // today (no `init {}` entry can declare a law), so the streams the
        // propagation loop reads are unchanged; when a law lands, IF2 gets
        // per-particle initial-state spread here with no further change.
        for ((s, pp), rng) in states.iter_mut()
            .zip(particle_params.iter())
            .zip(rngs.iter_mut())
        {
            let init_state = process.initial_state_draw(pp, rng)?;
            s.counts.copy_from_slice(&init_state.counts);
            s.reset_flows();
            // Per-ITERATION re-seed (NOT a per-observation reset): zero BOTH the
            // per-transition tally and the per-stream `acc` bins blanket, so no
            // stale incidence carries across IF2 iterations (Phase 2a).
            for a in &mut s.acc { *a = 0; }
        }

        let mut log_weights = vec![0.0_f64; n];
        let mut total_loglik = 0.0;
        // IM4 in 2026-04-19 inference review: count observations whose
        // ll_inc came back non-finite (all N particles hit an
        // impossible-state likelihood — NegBin on μ=0 with y>0, a
        // dimension failure, a zero-weight swarm during early
        // perturbation). Skip those from the total instead of
        // poisoning the rest of the iteration with -inf.
        let mut n_skipped_obs: usize = 0;
        let mut t = config.t_start;

        // gh#147 (M3.1). Cumulative particle-substep count for the
        // deterministic compute-budget guard, scoped per IF2 iteration
        // (each iteration is one PF evaluation; `ITER_BUDGET` bounds a
        // single eval, independent of how many iterations the fit runs).
        let mut iters: u64 = 0;

        for obs_idx in 0..n_obs {
            // PERTURB — Ionides et al. (2015) Algorithm 1, first line of the
            // inner loop. This runs BEFORE the process step so the SAME
            // Θ^P_n drives the simulation of X_n AND the measurement density
            // g(y_n | X_n; Θ^P_n) below (gh#365). Propagating at Θ^F_{n-1}
            // and scoring at Θ^P_n decouples the two: for a parameter living
            // in both the process and the observation model that is a genuine
            // error, not a phase offset that vanishes as σ → 0.
            //
            // pomp does the same — `pomp:::mif2_pfilter` (R/mif2.R, pomp
            // 6.4.0.2): `randwalk_perturbation` → `rprocess` → `dmeasure`,
            // all three on one `tparams`.
            //
            // `perturb_only_at_t0` params and simplex members are skipped —
            // the former are perturbed at t=0 only (pomp's `ivp()` in
            // `rw.sd`), simplex members perturbed jointly at t=0 only (they
            // are always initial-state parameters).
            //
            // `cooling_now` is also read by the per-parameter diagnostics
            // after the weighting; the `global_step` accounting is unchanged
            // (one t=0 step plus one per observation), so the SD applied at
            // observation k is still `per_step_cooling^(k+1)`.
            let cooling_now = per_step_cooling.powf(global_step as f64);
            for i in 0..n {
                for (pi, spec) in if2_params.iter().enumerate() {
                    if spec.perturb_only_at_t0 || simplex_member_indices.contains(&spec.index) { continue; }
                    let current = particle_params[i][spec.index];
                    let sd = spec.transformed_sd(spec.rw_sd, current) * cooling_now;
                    let z = spec.to_transformed(current);
                    let new_val = spec.from_transformed(z + rngs[i].normal() * sd);
                    particle_params[i][spec.index] = new_val;
                    if let Transform::Log { lo, hi } = &spec.transform {
                        if (new_val - lo).abs() < 1e-10 || (new_val - hi).abs() < 1e-10 {
                            clamp_counts[pi] += 1;
                        }
                    }
                }
            }
            global_step += 1;

            // Propagate — batched parallel dispatch per observation interval.
            let obs_time = obs_model.obs_time(obs_idx);
            let t_start = t;
            let dt = config.dt;
            let cur = Cursor {
                obs_idx,
                effect_idx: schedule.effect_idx_at(t_start),
                ..Default::default()
            };

            // gh#147 (M3.1). Deterministic compute-budget guard, PRE-window
            // (same closed-form scalar cost + placement as bootstrap_filter):
            // a pathological dt aborts before the substep loop runs.
            let cost = window_substep_cost(n, t, obs_time, dt);
            if let Some(kind) = check_iteration_budget(iters, cost, config.max_substeps) {
                return Err(pf_bail_error(kind, obs_idx, t0_if2.elapsed().as_secs_f64()));
            }
            iters = iters.saturating_add(cost);

            let errors: Vec<Result<(), SimError>> = states.par_iter_mut()
                .zip(particle_params.par_iter())
                .zip(rngs.par_iter_mut())
                .zip(scratches.par_iter_mut())
                .map(|(((state, pp), rng), scratch)| {
                    // gh#272 LICM: stage the per-eval prologue from THIS particle's
                    // θ (`pp`), once before its substep walk — `pp` is fixed across
                    // the window (IF2 perturbs θ at observation boundaries, not
                    // within a window). Per-particle because each carries a distinct
                    // perturbed θ; the scratch is structurally bound to the `pp` it
                    // was computed from, so no cross-particle aliasing is possible.
                    let pe_scratch = process.try_compiled_model()
                        .and_then(|m| crate::resolved_expr::stage_per_eval(m, pp, t_start, config.dt));
                    let per_eval = pe_scratch.as_deref();
                    // Shared inner-substep walk (Schedule::substeps); IF2's body is
                    // just the kernel step with the per-particle perturbed params.
                    // `fired` lands the cursor-keyed scheduled-intervention batch.
                    for (t_local, step_dt, fired) in schedule.substeps(cur, t_start) {
                        let due_iv: &[usize] = match fired {
                            Some(idx) => &scheduled.batches[idx],
                            None => &[],
                        };
                        process.step(state, pp, t_local, step_dt, per_eval, rng, scratch, due_iv)?;
                    }
                    Ok(())
                })
                .collect();
            for r in errors { r?; }
            t = schedule.window_end(cur, t);

            // FOLD (multi-cadence Phase 2a): close this interval's flow into
            // each Interval stream's persistent `acc` bin, once per observation,
            // serial, BEFORE scoring. `flow_accumulators` left untouched
            // (blanket-zeroed only at the per-obs reset below).
            for s in &mut states {
                obs_model.fold_into_acc(&s.flow_accumulators, &mut s.acc);
            }

            // Weight by observation likelihood — at the SAME θ that drove the
            // step above (gh#365).
            for i in 0..n {
                log_weights[i] = obs_model.log_likelihood(&states[i], obs_idx, &particle_params[i]);
            }

            // Per-parameter diagnostics (before resampling, using continuous weights):
            //
            // weighted_var_ratio: Var_w(θ_k) / Var(θ_k post-perturbation)
            //   where Var_w uses normalized importance weights.
            //   Measures selection pressure without resampling noise.
            //
            // q_k: rw_sd_effective / sd(θ_k before perturbation)
            //   Perturbation-to-cloud width ratio.
            {
                let weights = normalize_log_weights(&log_weights);

                for (pi, spec) in if2_params.iter().enumerate() {
                    let nf = n as f64;

                    // Unweighted variance (post-perturbation cloud)
                    let mean_u = particle_params.iter().map(|pp| pp[spec.index]).sum::<f64>() / nf;
                    let var_u = particle_params.iter()
                        .map(|pp| (pp[spec.index] - mean_u).powi(2)).sum::<f64>() / nf;

                    // Weighted variance (what the weights "want" the cloud to look like)
                    let mean_w = particle_params.iter().zip(&weights)
                        .map(|(pp, &w)| pp[spec.index] * w).sum::<f64>();
                    let var_w = particle_params.iter().zip(&weights)
                        .map(|(pp, &w)| w * (pp[spec.index] - mean_w).powi(2)).sum::<f64>();

                    let wvr = if var_u > 1e-30 { var_w / var_u } else { 1.0 };
                    wvr_accum[pi] += wvr;

                    // q_k: effective perturbation / cloud width
                    let sd_u = var_u.sqrt();
                    let eff_sd = spec.transformed_sd(spec.rw_sd, mean_u) * cooling_now;
                    let q = if sd_u > 1e-30 { eff_sd / sd_u } else { 0.0 };
                    q_k_accum[pi] += q;

                    diag_count[pi] += 1;
                }
            }

            // Log-likelihood increment. Under IC-free inference
            // (`config.skip_first_obs_from_loglik`), the first
            // observation still reweights and resamples (that's the
            // pinning of x₀ given y₁) but is dropped from the
            // accumulated log-likelihood. See
            // docs/dev/proposals/2026-04-18-ic-free-inference.md.
            let ll_inc = log_sum_exp(&log_weights) - (n as f64).ln();
            if !(config.skip_first_obs_from_loglik && obs_idx == 0) {
                if ll_inc.is_finite() {
                    total_loglik += ll_inc;
                } else {
                    n_skipped_obs += 1;
                }
            }

            // gh#110 degeneracy watchdog. ESS via the single-sourced
            // `ess_from_log_weights` (IF2's per-iter PF loop holds a
            // `Vec<f64>` of log-weights, not a `ParticleSwarm`). dead_count = 0
            // because IF2 does NOT mark per-particle deaths — `process.step`
            // errors propagate immediately, so AllParticlesDead is unreachable
            // here; ESS collapse or wall-clock fires first.
            let ess_now = ess_from_log_weights(&log_weights);
            ess_history.push(ess_now);
            if let Some(kind) = check_pf_degeneracy(&ess_history, 0, n) {
                // Statistical pathology (ESS collapse / all dead) → PFDegenerate.
                // elapsed is a display-only diagnostic, never the bail trigger.
                return Err(super::degeneracy::pf_bail_error(
                    kind, obs_idx, t0_if2.elapsed().as_secs_f64(),
                ));
            }

            // Resample states AND parameters jointly via double-buffer (no allocation)
            let indices = systematic_resample(&log_weights, &mut resample_rng);
            for (i, &src) in indices.iter().enumerate() {
                states_buf[i].counts.copy_from_slice(&states[src].counts);
                states_buf[i].flow_accumulators.copy_from_slice(&states[src].flow_accumulators);
                // Phase 2a: the per-stream `acc` bins travel with the particle.
                states_buf[i].acc.copy_from_slice(&states[src].acc);
                params_buf[i].copy_from_slice(&particle_params[src]);
            }
            std::mem::swap(&mut states, &mut states_buf);
            std::mem::swap(&mut particle_params, &mut params_buf);

            // Reset. `flow_accumulators` blanket (unchanged); the per-stream
            // `acc` bins per-stream — only Interval streams scheduled at THIS
            // union index zero (Phase 2a).
            for s in &mut states {
                s.reset_flows();
                obs_model.reset_due_acc(obs_idx, &mut s.acc);
            }
            log_weights.fill(0.0);
        }

        // Compute parameter means across particles → next iteration's starting point
        let mut param_means = current_params.clone();
        for spec in if2_params {
            let mean: f64 = particle_params.iter()
                .map(|pp| pp[spec.index])
                .sum::<f64>() / n as f64;
            param_means[spec.index] = mean;
        }

        // Per-parameter diagnostics for this iteration
        // Diagnostic SD at end of `iter`: use (1 + n_obs) steps/iter to match the
        // actual global-step accounting (the perturbation loop), not `n_obs`.
        let cooling_at_iter = per_step_cooling.powf((iter * (1 + n_obs)) as f64);
        // Total perturbation attempts: n particles × (1 t=0 step + n_obs observation steps)
        let total_perturb_steps = n * (1 + n_obs);
        let param_diag: Vec<ParamIterDiag> = if2_params.iter().enumerate().map(|(pi, spec)| {
            let cnt = diag_count[pi].max(1) as f64;
            ParamIterDiag {
                param_index: spec.index,
                weighted_var_ratio: wvr_accum[pi] / cnt,
                q_ratio: q_k_accum[pi] / cnt,
                effective_rw_sd: spec.rw_sd * cooling_at_iter,
                clamp_fraction: clamp_counts[pi] as f64 / total_perturb_steps as f64,
            }
        }).collect();

        iterations.push(IF2IterResult {
            iteration: iter,
            loglik: f64::NAN, // populated post-hoc by CLI via clean PF
            if2_perturbed_loglik: total_loglik,
            param_means: param_means.clone(),
            param_diag,
        });

        // Report progress (passes filter-mean params so the runner can
        // stream a trace row per iteration; see ProgressCallback doc).
        if let Some(cb) = &on_iteration {
            cb(iter, total_loglik, &param_means);
        }
        if n_skipped_obs > 0 {
            log::debug!(
                "if2 iter {}: skipped {}/{} observations with non-finite log-lik \
                 increment (typically early-exploration particles hit \
                 impossible states)",
                iter, n_skipped_obs, n_obs);
        }

        // Feed filter mean back as next iteration's starting params
        current_params = param_means;
    }

    let last_iter = iterations.last().expect("n_iterations ≥ 1 enforced at FitConfigV2::validate");

    // IF2 cooling drives the perturbed-particle swarm to a delta at the MLE
    // asymptotically, so the LAST iteration's filter mean is the MLE by
    // construction. Selecting an earlier iteration whose perturbed-loglik
    // happened to score higher (audit C2) picks a wider, less-converged
    // swarm — the perturbed-loglik field's own docstring (lines 125-129)
    // says "NOT useful for model assessment or convergence."
    //
    // The caller (cli/src/fit/runner.rs ~1183) re-scores per iteration with
    // a clean PF and runs a separate loglik_eval pass; that pass is the
    // authority on cross-chain winner selection (`select_winner_summary`).
    // This struct's `mle` field is now just "the MLE per IF2 theory" rather
    // than "argmax over a deliberately noisy diagnostic field."
    Ok(IF2Result {
        mle: last_iter.param_means.clone(),
        final_loglik: last_iter.if2_perturbed_loglik,
        last_loglik: last_iter.if2_perturbed_loglik,
        iterations,
    })
}

