//! Correlated pseudo-marginal particle filter.
//!
//! Stores all random draws as standard normals and transforms them at
//! evaluation time. The Crank-Nicolson update `u' = ρu + √(1-ρ²)z`
//! correlates successive PF evaluations so the likelihood RATIO has
//! low variance even when individual estimates are noisy.
//!
//! Reference: Deligiannidis, Doucet & Pitt (2018), JRSSB.

use std::time::Instant;

use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use crate::chain_binomial::StepScratch;
use crate::rng::StatefulRng;
use crate::error::{PFDegenerateKind, SimError};
use crate::schedule::{Cursor, Schedule, StepPolicy};
use super::degeneracy::{pf_bail_error, DeathMask};
use super::types::{ParticleState, ParticleSwarm, log_sum_exp, logw_variance, normalize_log_weights, LOG_PROB_FLOOR};
use super::particle_filter::PFilterResult;
use super::chain_binomial_process::ChainBinomialProcess;
use super::traits::{ObservationModel, SMCConfig};

/// Pre-drawn random state for one PF evaluation.
///
/// All values are standard normals. Transformed to the target distribution
/// (Gamma, Uniform) at consumption time.
#[derive(Clone, Serialize, Deserialize)]
pub struct PFRandomState {
    /// Gamma multiplier draws: gamma_noise[obs_idx][particle_idx * steps_per_obs + step]
    /// One normal per overdispersed transition per substep per particle.
    /// Transformed to Gamma(shape, scale) via inverse CDF.
    pub gamma_noise: Vec<Vec<f64>>,

    /// Resampling draws: one normal per observation.
    /// Transformed to Uniform(0,1) via Phi(·) for systematic resampling.
    pub resample_noise: Vec<f64>,

    /// Binomial total-exit draws per source group per substep per particle.
    /// binomial_noise[obs_idx][particle * steps_per_obs * n_groups + step * n_groups + group]
    /// Transformed to binomial counts via normal approximation (large np)
    /// or inverse CDF (small np). This is the dominant variance source
    /// that the broken to_bits() seeding failed to correlate.
    pub binomial_noise: Vec<Vec<f64>>,

    /// Number of source groups (for indexing into binomial_noise).
    pub n_source_groups: usize,
}

impl PFRandomState {
    /// Draw a fresh random state for one PF evaluation.
    pub fn draw_fresh(
        n_particles: usize,
        n_obs: usize,
        steps_per_obs: usize,
        n_source_groups: usize,
        rng: &mut StatefulRng,
    ) -> Self {
        let gamma_noise = (0..n_obs)
            .map(|_| (0..n_particles * steps_per_obs)
                .map(|_| rng.normal())
                .collect())
            .collect();
        let resample_noise = (0..n_obs)
            .map(|_| rng.normal())
            .collect();
        let binomial_noise = (0..n_obs)
            .map(|_| (0..n_particles * steps_per_obs * n_source_groups)
                .map(|_| rng.normal())
                .collect())
            .collect();
        PFRandomState { gamma_noise, resample_noise, binomial_noise, n_source_groups }
    }

    /// Crank-Nicolson update: u' = ρu + √(1-ρ²)z, z ~ N(0,1).
    /// Returns a new PFRandomState correlated with self.
    pub fn correlate(&self, rho: f64, rng: &mut StatefulRng) -> Self {
        let scale = (1.0 - rho * rho).sqrt();
        let gamma_noise = self.gamma_noise.iter()
            .map(|row| row.iter()
                .map(|&x| rho * x + scale * rng.normal())
                .collect())
            .collect();
        let resample_noise = self.resample_noise.iter()
            .map(|&x| rho * x + scale * rng.normal())
            .collect();
        let binomial_noise = self.binomial_noise.iter()
            .map(|row| row.iter()
                .map(|&x| rho * x + scale * rng.normal())
                .collect())
            .collect();
        PFRandomState { gamma_noise, resample_noise, binomial_noise,
                        n_source_groups: self.n_source_groups }
    }
}

/// Inverse binomial CDF: smallest `k` such that `P(X <= k) >= u`, `X ~ Binomial(n, p)`.
///
/// Walks the CDF of the *lighter* tail so the starting probability never
/// underflows. For `p > 0.5` the successes cluster near `n`, so a direct walk
/// from `k=0` seeds `P(X=0) = (1-p)^n`, which underflows to `0` for `p` near 1
/// with large `n` — after which the walk never accumulates and the old code
/// returned `n` (the whole source compartment) for *every* `u`, silently
/// over-draining a fast compartment on the correlated-PF path (gh#362). Instead,
/// for `p > 0.5` walk the failure count `M = n - X ~ Binomial(n, 1-p)` from 0,
/// where `P(M=0) = p^n` is well-scaled, and map back via the inverse-CDF identity
///
/// ```text
///   min{ k : F_X(k) >= u } = n - min{ m : F_M(m) > 1 - u }
/// ```
///
/// (strict `>` because `F_X(k) >= u  <=>  F_M(n-k-1) <= 1-u`).
pub fn binomial_quantile(n: u64, p: f64, u: f64) -> u64 {
    if p > 0.5 {
        // Walk the failure tail M ~ Binomial(n, 1-p); its mass sits near 0.
        let q = p; // = 1 - (failure prob), the "q" of the recurrence for M
        let pf = 1.0 - p; // failure prob
        let thresh = 1.0 - u;
        let mut cdf = 0.0;
        let mut prob = q.powi(n as i32); // P(M=0) = p^n
        for m in 0..=n {
            cdf += prob;
            if cdf > thresh {
                return n - m;
            }
            prob *= (n - m) as f64 / (m + 1) as f64 * pf / q;
            if prob < LOG_PROB_FLOOR {
                break;
            }
        }
        0 // u at the low extreme → k = 0
    } else {
        // Direct walk from k=0 (mass near 0 when p <= 0.5; the caller's np <= 20
        // dispatch keeps `(1-p)^n` well-scaled here).
        let q = 1.0 - p;
        let mut cdf = 0.0;
        let mut binom_prob = q.powi(n as i32); // P(X=0) = (1-p)^n
        for k in 0..=n {
            cdf += binom_prob;
            if cdf >= u {
                return k;
            }
            // P(X=k+1) = P(X=k) * (n-k)/(k+1) * p/(1-p)
            binom_prob *= (n - k) as f64 / (k + 1) as f64 * p / q;
            if binom_prob < LOG_PROB_FLOOR {
                break;
            }
        }
        n // fallback
    }
}

/// Transform a standard normal `z` to a `Gamma(shape, scale)` draw via the exact
/// inverse CDF: `z → u = Φ(z) → scale · GammaQuantile(u; shape)`.
///
/// Deterministic and monotone in `z`, so it preserves the correlated-PF
/// common-random-numbers coupling; and it samples the *exact* Gamma law, so the
/// correlated path now propagates the same distribution as `rng.gamma_multiplier`
/// (`rand_distr::Gamma`) does on every other path. Replaces the Wilson-Hilferty
/// approximation, which was accurate only for `shape ≳ 2` and, below that, biased
/// the multiplier low and clamped a growing fraction of draws to exactly 0 —
/// silently biasing correlated PMMH on overdispersed models (gh#372).
pub fn normal_to_gamma(z: f64, shape: f64, scale: f64) -> f64 {
    if shape < 1e-6 {
        return 1.0; // degenerate: no overdispersion (multiplier ≡ 1)
    }
    scale * numerics::gammp_inv(shape, phi(z))
}

/// Standard normal CDF — delegates to `obs_loglik::normal_cdf`.
pub fn phi(x: f64) -> f64 {
    super::obs_loglik::normal_cdf(x)
}

/// Tolerance for "the first observation coincides with `t_start`". Matches the
/// `Substeps` iterator's own termination slack (`schedule::EFFECT_EPS`, 1e-10):
/// the iterator yields zero substeps for a window `[t_start, obs(0)]` exactly
/// when `obs(0) <= t_start + EFFECT_EPS`, so keying the gate on the same slack
/// keeps the "empty leading window" judgement bit-consistent with the walk.
const LEADING_WINDOW_EPS: f64 = 1e-10;

/// Interior clamp on the correlated base uniform so resampling thresholds stay
/// strictly inside (0, 1). Distinct in concept from `PROB_FRACTION_EPS` (a
/// probability clamp) despite the shared magnitude.
const BASE_UNIFORM_EPS: f64 = 1e-10;

/// Substeps-per-window the CPM pre-drawn-noise arrays are sized for, derived
/// from the first non-trivial spacing `obs(1) - obs(0)` (or `obs(0) - t_start`
/// for a single observation).
pub fn cpm_steps_per_obs(obs_times: &[f64], t_start: f64, dt: f64) -> usize {
    let obs_dt = if obs_times.len() > 1 {
        obs_times[1] - obs_times[0]
    } else if obs_times.len() == 1 {
        obs_times[0] - t_start
    } else {
        1.0
    };
    crate::time::interval_steps(0.0, obs_dt, dt)
}

/// Validate that the CPM pre-drawn-noise indexing is sound for this obs grid.
///
/// The indexing (`noise_idx = particle*steps_per_obs + substep`) is a flat block
/// of `steps_per_obs` substeps per particle per window, so it is sound only if
/// EVERY window has exactly `steps_per_obs` substeps — with ONE exception: a
/// leading window `[t_start, obs(0)]` that coincides with `t_start` (the
/// universal `regular start=0` case). That window is empty — the `Substeps`
/// iterator yields zero substeps — so it consumes no noise block, the indexing
/// for windows `1..` is unaffected, and `obs(0)` is scored at the initial state
/// exactly as the plain bootstrap PF does. Any OTHER non-uniformity (a genuine
/// mid-period start, e.g. obs at `[5,12,19]` with `t_start=0`, or an interior
/// window of the wrong width) is rejected with an actionable message.
///
/// This is the single source of truth for CPM obs-grid validity: the filter
/// calls it defensively, and profile/fit call it once at preflight (gh#193) so
/// a genuine non-uniform grid surfaces this message instead of a swallowed
/// all-(-inf) profile.
pub fn validate_cpm_obs_grid(obs_times: &[f64], t_start: f64, dt: f64) -> Result<(), SimError> {
    let steps_per_obs = cpm_steps_per_obs(obs_times, t_start, dt);
    let mut window_start = t_start;
    for (i, &obs_t) in obs_times.iter().enumerate() {
        // Empty leading window (obs(0) == t_start): allowed, consumes no noise.
        if i == 0 && obs_t - t_start <= LEADING_WINDOW_EPS {
            window_start = obs_t;
            continue;
        }
        let window_steps = crate::time::interval_steps(window_start, obs_t, dt);
        if window_steps != steps_per_obs {
            let which = if i == 0 {
                format!("the FIRST window [t_start={window_start:.4}, obs(0)={obs_t:.4}]")
            } else {
                format!("the window [obs({})={window_start:.4}, obs({i})={obs_t:.4}]", i - 1)
            };
            return Err(SimError::Validation(format!(
                "correlated PF requires every observation window to have the same \
                 number of dt-substeps (the pre-drawn-noise arrays are sized at \
                 {steps_per_obs} substeps/window from obs(1)-obs(0)), but {which} \
                 has {window_steps} substep(s) at dt={dt:.4}. CPM tolerates a \
                 leading window only when the first observation coincides with \
                 t_start (it is then scored at the initial state); a mid-period \
                 start or a mis-sized interior window breaks the indexing. Drop \
                 to vanilla PMMH (rho = None), or align the observation grid so \
                 every window spans the same number of substeps."
            )));
        }
        window_start = obs_t;
    }
    Ok(())
}

/// Run the bootstrap particle filter with pre-drawn correlated randoms.
///
/// The Gamma multiplier for overdispersed transitions is drawn from
/// `randoms.gamma_noise` (transformed from normal to Gamma via inverse CDF).
/// Systematic resampling uses `randoms.resample_noise` (transformed to
/// uniform via Phi). All other draws (binomial in reulermultinom) use
/// per-particle RNGs seeded from the gamma noise for partial correlation.
pub fn bootstrap_filter_correlated(
    process: &ChainBinomialProcess,
    obs_model: &dyn ObservationModel<ParticleState>,
    params: &[f64],
    config: &SMCConfig,
    randoms: &PFRandomState,
    seed: u64,
) -> Result<PFilterResult, SimError> {
    let model = &*process.compiled;
    let n_particles = config.n_particles;
    let dt = config.dt;

    let n_int = model.int_local_to_global.len();
    let n_tr = model.model.transitions.len();
    // Per-Interval-stream `acc` bins (multi-cadence Phase 2a), sized from the
    // obs model (the process does not know `n_interval_streams`).
    let n_acc = obs_model.n_interval_streams();

    // Per-particle RNGs via ChaCha8 stream counter (IM1 fix 2026-04-19).
    // Built before the initial state because drawing x₀ is a draw from a
    // particle's own stream; `new_stream` consumes nothing, so the streams the
    // propagation loop sees below are unchanged.
    let mut rngs: Vec<StatefulRng> = (0..n_particles)
        .map(|i| StatefulRng::new_stream(seed, i as u64))
        .collect();

    // ONE draw, copied to every particle — the pre-split behaviour, kept
    // because no `init {}` entry can declare a law yet, so `initial_state_draw`
    // is deterministic and consumes nothing from `rngs[0]`. Note the pre-drawn
    // correlated randoms in `randoms` cover the transition kernel only; a
    // future initial-state law drawn here would be an *uncorrelated* addition
    // to a CPM proposal, which is a design question step 5 owes an answer.
    // `rngs` is empty when `n_particles == 0`; see the same guard in
    // `particle_filter.rs::bootstrap_filter`.
    let (init_int, _init_real) = match rngs.first_mut() {
        Some(rng0) => model.initial_state_draw(params, rng0)?,
        None => model.initial_state_draw(
            params, &mut StatefulRng::new_stream(seed, 0),
        )?,
    };
    let mut swarm = ParticleSwarm::new(n_particles, n_int, n_tr, n_acc);
    for p in &mut swarm.states {
        p.counts.copy_from_slice(&init_int.counts);
    }

    let mut states_buf: Vec<ParticleState> = (0..n_particles)
        .map(|_| ParticleState::new(n_int, n_tr, n_acc))
        .collect();

    let mut scratches: Vec<StepScratch> = (0..n_particles)
        .map(|_| StepScratch::new(model))
        .collect();

    let n_obs = obs_model.n_observations();
    let mut total_loglik = 0.0;
    let mut ess_trace = Vec::with_capacity(n_obs);
    let mut logw_var_trace = Vec::with_capacity(n_obs);
    let mut ll_increments = Vec::with_capacity(n_obs);
    let mut t = config.t_start;

    // Substeps per window the pre-drawn-noise arrays are sized for. CPM requires
    // (near-)uniform observation spacing because that block size is fixed.
    let obs_times: Vec<f64> = (0..n_obs).map(|i| obs_model.obs_time(i)).collect();

    // gh#216: scheduled interventions fire CURSOR-keyed off the timeline's effect
    // boundaries (registered as `effect_times` below), so an off-grid observation
    // re-tiling the Exact substep grid no longer moves the firing instant. The two
    // unsupported Exact cases are refused loudly (parametric `at [<param>]`; a
    // scheduled fire time off the dt grid — which would also add a substep and
    // break the CPM fixed-`steps_per_obs` noise indexing); events are out of scope.
    crate::intervention::guard_attimesexpr_exact(model, StepPolicy::Exact)?;
    crate::intervention::guard_exact_offgrid_effect_time(
        model, params, config.t_start, dt, StepPolicy::Exact,
    )?;
    let scheduled = crate::intervention::timeline_effects(model, params);

    let steps_per_obs = cpm_steps_per_obs(&obs_times, config.t_start, dt);

    // Validate the obs grid against the pre-drawn-noise indexing. A leading
    // window coinciding with t_start (obs(0) == t_start) is allowed — it is
    // empty (zero substeps), consumes no noise block, and obs(0) is scored at
    // the initial state, exactly as the plain bootstrap PF does. Every other
    // window must be exactly `steps_per_obs` substeps. Single source of truth in
    // `validate_cpm_obs_grid`; profile/fit also preflight it (gh#193) so a
    // genuine non-uniform grid surfaces this message instead of a swallowed
    // all-(-inf) profile.
    validate_cpm_obs_grid(&obs_times, config.t_start, dt)?;

    // Merged timeline spine: the EXACT policy clips each substep to the next
    // observation boundary (same as the bootstrap PF). The Schedule reproduces
    // dt.min(obs_time - t) exactly, so the per-window substep COUNT is preserved
    // and the pre-drawn-noise indexing (noise_idx = i*steps_per_obs + substep)
    // is unaffected. Substep TIME stays accumulated (s*dt deferred, task #14).
    // CPM keeps its guards + cpm_steps_per_obs + validate_cpm_obs_grid above
    // (the noise-block grid check must run before the schedule), so it does NOT
    // route through `ExactInferenceTimeline::build` (which would bundle the guards
    // ahead of validate_cpm_obs_grid). It does adopt the typed constructor: the
    // effect timeline is index-aligned with `scheduled.batches`, so it is
    // validated order-PRESERVING (`from_timeline`), never sorted; obs are
    // validated finite + strictly-increasing through `ObsTimes`.
    let sched_t_end = obs_times.last().copied().unwrap_or(config.t_start);
    let schedule = Schedule::exact_inference(
        dt,
        sched_t_end,
        crate::boundary_times::EffectTimes::from_timeline(&scheduled)?,
        crate::boundary_times::ObsTimes::new(obs_times)?,
    );

    // Gamma shape/scale for the overdispersed transition (precompute).
    //
    // ASSUMPTION: σ² is state-independent (typically a bare parameter like
    // `sigma_se`). We evaluate at a zero state because the expression is
    // precomputed once for all particles and substeps.
    //
    // Check: if σ² depends on compartment counts, CPM can't handle it correctly
    // (would need per-particle per-substep evaluation). Emit an error.
    for re in model.resolved.overdispersion.iter().flatten() {
        if crate::resolved_expr::references_state(re) {
            return Err(SimError::Validation(
                "Correlated pseudo-marginal (CPM) does not support state-dependent \
                 overdispersion (σ² references compartment counts). Use vanilla PMMH \
                 (rho = None) or make σ² a parameter instead.".into()
            ));
        }
    }

    // IM8 in 2026-04-19 inference review: the CPM machinery uses a
    // single `scratch.gamma_override: Option<f64>` that step_one
    // consumes for the FIRST overdispersed transition in a substep
    // and then falls through to fresh rng.gamma_multiplier() draws
    // for any subsequent ones. Plus sigma_sq below is picked from
    // the first overdispersed transition and reused for every
    // gamma draw. Neither issue is recoverable without a larger
    // rewrite (Vec<f64> for gamma_override, per-transition σ²
    // evaluation), so fail fast at preflight rather than silently
    // produce uncorrelated / mis-transformed gamma draws. Users hit
    // by this should drop to vanilla PMMH (rho = None).
    let n_overdispersed = model.resolved.overdispersion.iter()
        .filter(|od| od.is_some())
        .count();
    // Check per-source-group: if any group has >1 overdispersed
    // transition, CPM correlation breaks.
    for (_src, group) in &model.source_groups {
        let n_od_in_group = group.iter()
            .filter(|&&tr_idx| model.resolved.overdispersion[tr_idx].is_some())
            .count();
        if n_od_in_group > 1 {
            return Err(SimError::Validation(format!(
                "Correlated pseudo-marginal (CPM) does not support more than \
                 one overdispersed transition sharing a source compartment \
                 (found {} in this model). The CPM gamma_override machinery \
                 is a single-slot Option<f64> that step_one consumes for the \
                 first overdispersed transition only. Use vanilla PMMH \
                 (rho = None), or collapse the multiple overdispersed \
                 outflows into one.", n_od_in_group
            )));
        }
    }
    // Also reject if different overdispersed transitions evaluate
    // to distinct σ² values — the global sigma_sq picked below
    // would be wrong for all but the first. σ² is state-independent
    // by construction: `CompiledModel::new()` rejects models whose
    // overdispersion σ² references compartment state
    // (docs/dev/incidents/2026-04-22-observation-sampler-scratch-state.md),
    // so evaluation at a zero scratch is sound here.
    if n_overdispersed > 1 {
        let int_s = crate::state::IntState::new(n_int);
        let real_s = crate::state::RealState::new(model.real_local_to_global.len());
        let ctx = crate::propensity::EvalCtx {
            model, int_s: &int_s, real_s: &real_s, params,
            t: 0.0, dt: config.dt, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        let mut first_sq: Option<f64> = None;
        for re in model.resolved.overdispersion.iter().flatten() {
            let sq = crate::resolved_expr::eval_resolved(re, &ctx);
            match first_sq {
                None => first_sq = Some(sq),
                Some(first) if (first - sq).abs() > 1e-12 * first.abs().max(1.0) => {
                    return Err(SimError::Validation(
                        "Correlated pseudo-marginal (CPM) does not support \
                         distinct σ² values across overdispersed \
                         transitions (it uses the first transition's σ² for \
                         every gamma draw). Either share one σ² parameter \
                         across all overdispersed transitions, or drop to \
                         vanilla PMMH (rho = None).".into()
                    ));
                }
                _ => {}
            }
        }
    }

    let sigma_sq = model.resolved.overdispersion.iter()
        .find_map(|od| {
            od.as_ref().map(|re| {
                let int_s = crate::state::IntState::new(n_int);
                let real_s = crate::state::RealState::new(model.real_local_to_global.len());
                let ctx = crate::propensity::EvalCtx {
                    model, int_s: &int_s, real_s: &real_s, params,
                    t: 0.0, dt, projected: None, aux: None, int_float_override: None, per_eval: None,
                };
                crate::resolved_expr::eval_resolved(re, &ctx)
            })
        })
        .unwrap_or(1.0);

    let gamma_shape = dt / sigma_sq;
    let gamma_scale = sigma_sq / dt;

    // gh#272 LICM: stage the per-eval prologue ONCE for this filter (θ = `params`
    // fixed for the whole correlated-PF / PMMH proposal evaluation) and lend it
    // into every particle's every substep. `None` ⇒ on-demand (byte-identical).
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, config.t_start, dt);
    let per_eval = per_eval_scratch.as_deref();

    // gh#367. Per-particle death mask — the SAME policy object the bootstrap PF
    // uses (`degeneracy::DeathMask`): a per-particle-recoverable error kills
    // only that particle (−∞ weight, discarded at resampling); everything else
    // still propagates out of the filter. Before this, one particle's
    // recoverable excursion propagated out of the whole evaluation, which the
    // PMMH driver reads as "θ ruled out" — a silent bias against boundary
    // regions where occasional particle failure is expected.
    let mut deaths = DeathMask::new(n_particles);
    // Display-only diagnostic on the all-dead bail (how long the doomed call
    // ran). Never gates anything — the bail is a pure function of the mask.
    let t0_call = Instant::now();

    for obs_idx in 0..n_obs {
        // The substep walk terminates at this obs via Schedule::substeps (cursor
        // points at obs_idx); no explicit obs_time needed. The effect cursor is
        // positioned at the first scheduled-effect boundary not yet fired by `t`.
        let t_start = t;
        let cur = Cursor {
            obs_idx,
            effect_idx: schedule.effect_idx_at(t_start),
            ..Default::default()
        };

        // Propagate particles with pre-drawn correlated noise (parallel)
        let gamma_row = &randoms.gamma_noise[obs_idx];
        let binom_row = &randoms.binomial_noise[obs_idx];
        let n_groups = randoms.n_source_groups;
        // `Ok(true)` = this particle died recoverably (gh#367 death mask);
        // `Ok(false)` = it propagated cleanly; `Err` = not recoverable, tear the
        // evaluation down.
        let outcomes: Vec<Result<bool, SimError>> = swarm.states.par_iter_mut()
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .enumerate()
            .map(|(i, ((state, rng), scratch))| {
                // Shared inner-substep walk (Schedule::substeps); the CPM body
                // injects the pre-drawn correlated noise keyed on the within-window
                // substep index before each kernel step.
                for (substep, (t_local, step_dt, fired)) in schedule.substeps(cur, t_start).enumerate() {
                    // Inject pre-drawn Gamma multiplier.
                    //
                    // After the uniform-window gate above, every window has exactly
                    // `steps_per_obs` substeps, so noise_idx is always in range. A
                    // miss here would mean the gate regressed and we are about to
                    // silently fall through to fresh per-particle RNG, decorrelating
                    // the estimator with no diagnostic — fail LOUDLY instead.
                    let noise_idx = i * steps_per_obs + substep;
                    debug_assert!(
                        noise_idx < gamma_row.len(),
                        "CPM gamma noise overrun: noise_idx {noise_idx} >= {} \
                         (particle {i}, substep {substep}, steps_per_obs \
                         {steps_per_obs})",
                        gamma_row.len(),
                    );
                    if noise_idx >= gamma_row.len() {
                        return Err(SimError::Validation(format!(
                            "correlated PF gamma-noise overrun: index {noise_idx} \
                             out of {} (particle {i}, substep {substep}, \
                             steps_per_obs {steps_per_obs}, obs window {obs_idx}). \
                             This means a window had more substeps than the noise \
                             array was sized for — the uniform-window gate should \
                             have rejected this run. Report as a bug.",
                            gamma_row.len(),
                        )));
                    }
                    let z = gamma_row[noise_idx];
                    let g = normal_to_gamma(z, gamma_shape, gamma_scale);
                    scratch.gamma_override = Some(g);

                    // Inject pre-drawn binomial z-values per source group.
                    // step_one converts z → count after computing (n, p).
                    scratch.binomial_z_values.clear();
                    scratch.binomial_z_idx = 0;
                    for group in 0..n_groups {
                        let binom_idx = i * steps_per_obs * n_groups + substep * n_groups + group;
                        debug_assert!(
                            binom_idx < binom_row.len(),
                            "CPM binomial noise overrun: binom_idx {binom_idx} >= \
                             {} (particle {i}, substep {substep}, group {group})",
                            binom_row.len(),
                        );
                        if binom_idx >= binom_row.len() {
                            return Err(SimError::Validation(format!(
                                "correlated PF binomial-noise overrun: index \
                                 {binom_idx} out of {} (particle {i}, substep \
                                 {substep}, group {group}, obs window {obs_idx}). \
                                 A window had more substeps than the noise array \
                                 was sized for — the uniform-window gate should \
                                 have rejected this run. Report as a bug.",
                                binom_row.len(),
                            )));
                        }
                        scratch.binomial_z_values.push(binom_row[binom_idx]);
                    }

                    // gh#216: every effect (events + scheduled interventions) is
                    // cursor-keyed from the timeline's effect boundary (`fired`),
                    // registered on `effect_times` via `timeline_effects`. Split
                    // the boundary's batch by kind into the lifecycle halves;
                    // empty off a boundary. step_one applies what we put here.
                    scratch.effect_batch.clear();
                    if let Some(idx) = fired {
                        crate::effects::split_due_batch(
                            model, &scheduled.batches[idx], &mut scratch.effect_batch,
                        );
                    }
                    // KNOWN LIMITATION (docs/dev/incidents/2026-06-07-chain-
                    // binomial-stale-real-state.md, §inference scope): the
                    // correlated PF tracks integer counts only — no real
                    // reservoir is advanced. A zeroed RealState makes a rate
                    // coupling to a real compartment read 0. For real-free
                    // models (n_real == 0) this is empty and byte-identical.
                    let mut real = crate::state::RealState::new(
                        process.compiled.real_local_to_global.len());
                    // `step_dt` is the realized substep (clipped under Exact).
                    // gh#367: route the step's outcome through the shared
                    // death-mask policy — `?` propagates a non-recoverable
                    // error out of the filter, `true` kills only this particle
                    // and stops its walk for this window.
                    if DeathMask::classify(crate::chain_binomial::step_one(
                        model, &mut state.counts, &mut state.flow_accumulators,
                        &mut real,
                        // gh#272 LICM: scratch staged once for this filter, threaded in.
                        params, t_local, step_dt, per_eval, rng, scratch,
                    ))? {
                        return Ok(true);
                    }
                }
                Ok(false)
            })
            .collect();
        deaths.absorb(outcomes)?;

        // gh#367. Every particle died: the limit case the mask cannot absorb —
        // the whole weight vector is −∞, so resampling has nothing to select
        // and `normalize_log_weights` would fall back to uniform over dead
        // states. Bail exactly as the bootstrap PF does. `PFDegenerate` is not
        // structural, so the PMMH driver still reads it as "θ ruled out" (−∞) —
        // the same outcome the pre-gh#367 code produced for this case.
        //
        // Scope note: the bootstrap PF reaches this through
        // `check_pf_degeneracy`, which ALSO carries the ESS-collapse watchdog.
        // The correlated PF has never had that watchdog, and adding it here
        // would change the returned log-likelihood of existing correlated-PMMH
        // fits — a separate policy decision, deliberately not folded into this
        // fix.
        if deaths.all_dead() {
            return Err(pf_bail_error(
                PFDegenerateKind::AllParticlesDead, obs_idx, t0_call.elapsed().as_secs_f64(),
            ));
        }
        t = schedule.window_end(cur, t);

        // FOLD (multi-cadence Phase 2a): close this interval's flow into each
        // Interval stream's persistent `acc` bin, once per observation, serial,
        // BEFORE scoring. `flow_accumulators` left untouched — the resampling
        // sort key below still reads it bit-identically.
        for state in &mut swarm.states {
            obs_model.fold_into_acc(&state.flow_accumulators, &mut state.acc);
        }

        // Compute log-weights. Dead particles (gh#367) get −∞ so the sorted
        // systematic resample below discards them; the obs model is never
        // scored on a particle whose propagation errored out.
        for (i, state) in swarm.states.iter().enumerate() {
            swarm.log_weights[i] = if deaths.is_dead(i) {
                f64::NEG_INFINITY
            } else {
                obs_model.log_likelihood(state, obs_idx, params)
            };
        }

        let ll_increment = log_sum_exp(&swarm.log_weights) - (n_particles as f64).ln();
        total_loglik += ll_increment;
        ll_increments.push(ll_increment);
        ess_trace.push(swarm.ess());
        logw_var_trace.push(logw_variance(&swarm.log_weights));

        // Sorted systematic resampling with correlated uniform
        // Sort particles by projected value for correlation preservation.
        // Use the first flow accumulator sum as a sorting key — this is a
        // heuristic for correlation preservation during resampling.
        let mut sort_order: Vec<usize> = (0..n_particles).collect();
        {
            let projections: Vec<f64> = swarm.states.iter()
                .map(|s| s.flow_accumulators.iter().map(|&v| v as f64).sum())
                .collect();
            sort_order.sort_by(|&a, &b| projections[a].total_cmp(&projections[b]));
        }

        // Resampling using correlated uniform
        let base_uniform = phi(randoms.resample_noise[obs_idx]).clamp(BASE_UNIFORM_EPS, 1.0 - BASE_UNIFORM_EPS);

        // Build sorted weights for resampling
        let sorted_weights: Vec<f64> = sort_order.iter()
            .map(|&i| swarm.log_weights[i])
            .collect();

                // Systematic resample with sorted weights and correlated uniform.
        // Im15 in 2026-04-19 inference review: previously a
        // `_resample_rng` was constructed here and never read —
        // `systematic_resample_fixed_u` takes the correlated
        // `base_uniform` directly and needs no RNG. Deleted.
        let indices = systematic_resample_fixed_u(&sorted_weights, base_uniform);

        // Map sorted indices back to original particle indices
        for (i, &sorted_idx) in indices.iter().enumerate() {
            let orig_idx = sort_order[sorted_idx];
            states_buf[i].counts.copy_from_slice(&swarm.states[orig_idx].counts);
            states_buf[i].flow_accumulators.copy_from_slice(&swarm.states[orig_idx].flow_accumulators);
            // Phase 2a: the per-stream `acc` bins travel with the particle.
            states_buf[i].acc.copy_from_slice(&swarm.states[orig_idx].acc);
        }
        std::mem::swap(&mut swarm.states, &mut states_buf);

        // Reset. `flow_accumulators` blanket (unchanged — the sort key above
        // reads it); the per-stream `acc` bins per-stream — only Interval
        // streams scheduled at THIS union index zero (Phase 2a).
        for state in &mut swarm.states {
            state.reset_flows();
            obs_model.reset_due_acc(obs_idx, &mut state.acc);
        }
        for lw in &mut swarm.log_weights { *lw = 0.0; }

        // gh#367: clear the mask after resampling — every surviving slot was
        // copied from a particle with finite weight, and resampling shuffles by
        // index, so the pre-resample mask no longer refers to anything.
        deaths.clear();
    }

    Ok(PFilterResult {
        log_likelihood: total_loglik,
        ess_trace,
        logw_var_trace,
        ll_increments,
        predictions: None,
        final_states: Some(swarm.states),
        // Correlated PF is used by PMMH; ancestry recording there is
        // a separate feature (smoothing via CSMC with ancestor
        // sampling lives in pgas.rs already). Leaving as None keeps
        // this code path out of scope for the 2026-04-19 PF-traj
        // proposal.
        ancestry: None,
        prequential: None,
    })
}

/// Systematic resampling with a **fixed** base uniform (correlated-PF CRN
/// coupling) rather than a fresh rng draw. Thin wrapper over the shared
/// [`super::resampling::systematic_resample_core`] — the selection loop is
/// identical to [`super::resampling::systematic_resample`]; only the source of
/// the base uniform differs.
fn systematic_resample_fixed_u(log_weights: &[f64], base_uniform: f64) -> Vec<usize> {
    if log_weights.is_empty() { return vec![]; }
    let weights = normalize_log_weights(log_weights);
    super::resampling::systematic_resample_core(&weights, base_uniform)
}
