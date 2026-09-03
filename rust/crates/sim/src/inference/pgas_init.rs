//! How a PGAS chain obtains its initial reference trajectory `X₀` (gh#784).
//!
//! Particle Gibbs needs a starting latent path to condition the first
//! conditional-SMC sweep on. Drawing that path by simulating the model forward
//! at `θ₀` — one unconditioned realization of the process — puts no weight on
//! the observations at all, so on any model where the data are informative the
//! draw routinely lands where the observation model scores `−∞`. The sweep is
//! then asked to repair a reference it has been conditioned on, which gh#780
//! measured failing: the reference can take essentially the whole normalised
//! weight in an early observation window, and the swarm becomes its own
//! descendants.
//!
//! Standard particle-Gibbs practice is to seed the reference from an ORDINARY
//! UNCONDITIONAL particle filter at `θ₀` and hand one particle's ancestral path
//! to the sampler (Andrieu, Doucet & Holenstein 2010, *JRSS-B* 72:269-342, §4.5;
//! Lindsten, Jordan & Schön 2014, *JMLR* 15:2145-2184, §2.3). That filter
//! reweights against the data at every observation, so the path it returns
//! already explains them as well as `N` draws at `θ₀` can. This module is that
//! pass.
//!
//! # Why this is not `bootstrap_filter`
//!
//! [`crate::inference::particle_filter::bootstrap_filter`] cannot produce what
//! PGAS needs. Its `AncestorTrace` records one state vector per particle per
//! OBSERVATION, while a [`PGASTrajectory`] is a per-SUBSTEP record carrying, for
//! each substep, the pre-state, the post-state, the realized per-transition
//! `flows`, and the overdispersion `gammas` that produced them. Those are
//! exactly the quantities `log_transition_density_substep` scores, and they are
//! not recoverable from an observation-resolution state path: reconstructing
//! flows by differencing compartment counts is unsafe under event/balance
//! interactions (gh#48, gh#264), and the gamma multipliers are not a function of
//! the states at all. `bootstrap_filter` is also generic over `ProcessModel`,
//! whose `step` does not surface either quantity.
//!
//! So the pass runs here, in PGAS's own representation, over the same substep
//! grid `run_pgas` built and through the same `step_one` producer seam every
//! other cell uses. What it returns is exact, not a conversion.
//!
//! # Deliberate differences from the production filter
//!
//! **No ESS watchdog.** `bootstrap_filter` bails with `PFDegenerate` when the
//! effective sample size collapses over a window. That check exists to refuse a
//! marginal-likelihood ESTIMATE whose variance is unusable. Here we are not
//! estimating anything — we want one path. A filter whose ESS has collapsed
//! still returns a lineage, and if that lineage has finite complete-data density
//! it is a valid conditioning path for PGAS. Refusing it would refuse chains the
//! sampler can run.
//!
//! **The only failure is an empty support.** The pass fails when no particle in
//! the swarm carries non-zero weight — every particle scored `−∞` at some
//! observation, or every particle died on a recoverable per-particle error. That
//! is a statement about this swarm at this `θ₀`, NOT a claim that
//! `p(y | θ₀) = 0`. gh#784's `STRUCTURALLY_INFEASIBLE` status is reserved for
//! support logic that can prove infeasibility; nothing here ever asserts it.

use rayon::prelude::*;

use crate::chain_binomial::{step_one, StepScratch};
use crate::compiled_model::CompiledModel;
use crate::error::{InitFallback, InitSource, SimError};
use crate::rng::StatefulRng;
use crate::state::RealState;

use super::degeneracy::DeathMask;
use super::multi_stream_obs::MultiStreamObsModel;
use super::particle_filter::Observation;
use super::pgas::{
    complete_data_loglik, draw_free_particle_initial_state, fill_producer_batch,
    sample_categorical_log, simulate_reference_on_grid, EffectFiring, ObsAtSubstep,
    PGASTrajectory, SubstepRecord,
};
use super::resampling::systematic_resample;
use super::types::{init_particle_rngs, RESAMPLE_RNG_STREAM};

/// Stream-separation salt for the initialization pass's RNG, so its draws come
/// from a stream disjoint from the sweep RNG (`StatefulRng::new(seed)`) and from
/// every CSMC sweep seed (`seed ^ (sweep+1)·φ ^ …`, `pgas.rs`).
const INIT_SEED_SALT: u64 = 0x1f83_d9ab_fb41_bd6b;

/// Obtain a PGAS chain's initial reference trajectory `X₀` (gh#784).
///
/// Runs the unconditional SMC pass at `params` and returns its lineage when
/// that lineage has finite complete-data density. Otherwise falls back to the
/// forward `simulate_reference` draw, reporting WHY in the returned
/// [`InitSource`].
///
/// # The correctness bar
///
/// PGAS conditions its first sweep on `X₀`, so `X₀` must be a point the target
/// gives positive density — a path the transition/observation densities score as
/// impossible would reintroduce gh#780's failure in a new place. The finiteness
/// of `log p(y, X₀ | θ₀)` is therefore CHECKED here, on the same
/// `complete_data_loglik` the sampler scores with, not assumed from the fact
/// that a filter produced it.
///
/// # What happens when the pass fails
///
/// The chain falls back to today's behaviour: one forward draw at `θ₀`, taken
/// off the caller's own `forward_rng`, which the pass never touches. A chain
/// that falls back therefore consumes exactly the RNG a pre-gh#784 run consumed
/// and is bit-identical to it. The fallback is RECORDED rather than silent —
/// how often it fires is the measurement that decides whether gh#784's step 3b
/// (the alive extension) is needed at all.
///
/// Nothing here ever asserts `p(y | θ₀) = 0`.
#[allow(clippy::too_many_arguments)]
pub fn initial_reference_trajectory(
    model: &CompiledModel,
    params: &[f64],
    grid: &[(f64, f64)],
    n_particles: usize,
    dt: f64,
    observations: &[Observation],
    obs_model: &MultiStreamObsModel,
    seed: u64,
    obs_at_substep: &ObsAtSubstep,
    firing: EffectFiring<'_>,
    forward_rng: &mut StatefulRng,
) -> Result<(PGASTrajectory, InitSource), SimError> {
    let fallback = match unconditional_smc_pass(
        model, params, grid, n_particles, dt, obs_model, seed, obs_at_substep, firing,
        forward_rng.binomial_algorithm(),
    )? {
        UnconditionalPass::Path(traj) => {
            let ll = complete_data_loglik(
                model, &traj, params, observations, dt, obs_model, obs_at_substep,
            )?;
            if ll.total.is_finite() {
                return Ok((traj, InitSource::UnconditionalFilter));
            }
            // A traced lineage with `-inf` complete-data density. The filter's
            // own weights cannot rule this out: they score the OBSERVATION terms
            // at the observation substeps, while `complete_data_loglik` also
            // charges the initial-state law and every substep's transition
            // density. Report the components — a non-finite TRANSITION term here
            // is a step_one/density disagreement (gh#80), not a bad start.
            InitFallback::NonFiniteDensity {
                transition: ll.transition,
                observation: ll.observation,
                initial_state: ll.initial_state,
            }
        }
        UnconditionalPass::NoSupport(reason) => reason,
    };

    let traj = simulate_reference_on_grid(model, params, dt, grid, firing, forward_rng)?;
    Ok((traj, InitSource::ForwardDraw(fallback)))
}

/// What the unconditional pass produced.
pub enum UnconditionalPass {
    /// One particle's ancestral path, drawn from the final substep's weights.
    Path(PGASTrajectory),
    /// The swarm had no surviving support. Never a claim about `p(y | θ₀)`.
    NoSupport(InitFallback),
}

/// Run an ordinary unconditional bootstrap particle filter at `params` over the
/// PGAS substep grid, and return ONE particle's ancestral path.
///
/// The loop is the free-particle half of [`super::pgas::csmc_as`] with the three
/// things that make that sweep conditional removed: no slot is clamped to a
/// reference's recorded noise, the resample is the ordinary unconditional
/// systematic scheme over the whole swarm rather than
/// `conditional_multinomial_resample`, and there is no ancestor-sampling move.
/// Every particle is a free draw, which is the point — nothing here is
/// conditioned on a path we have diagnosed as the problem.
///
/// `grid[s] = (t0, dt_substep)` is the realized substep grid from
/// `build_substep_grid`, the same one the sweeps will tile against.
///
/// # Errors
///
/// Only structural / non-per-particle-recoverable `SimError`s propagate; a
/// recoverable per-particle failure kills that particle through the shared
/// [`DeathMask`] policy (gh#367), exactly as `bootstrap_filter` does.
#[allow(clippy::too_many_arguments)]
pub fn unconditional_smc_pass(
    model: &CompiledModel,
    params: &[f64],
    grid: &[(f64, f64)],
    n_particles: usize,
    dt: f64,
    obs_model: &MultiStreamObsModel,
    seed: u64,
    obs_at_substep: &ObsAtSubstep,
    firing: EffectFiring<'_>,
    // gh#747: the init pass draws too, so it must use the same sampler the
    // sweeps will -- otherwise a `btrs`-addressed run seeds itself with BTPE.
    binomial: crate::rng::BinomialAlgorithm,
) -> Result<UnconditionalPass, SimError> {
    assert!(n_particles > 0, "unconditional init pass needs at least one particle");
    let t_start = model.model.simulation.t_start;
    let n_substeps = grid.len();
    let n_tr = model.model.transitions.len();

    // gh#272 LICM: stage the per-eval prologue ONCE for this θ, as every other
    // producer/density loop does, and lend it into each substep's rate eval.
    let per_eval_scratch = crate::resolved_expr::stage_per_eval(model, params, t_start, dt);
    let per_eval = per_eval_scratch.as_deref();

    // gh#53: resolve the event step indices once at the runtime dt.
    let fire_steps = model.resolve_fire_steps(dt, params);

    let seed = seed ^ INIT_SEED_SALT;
    let mut rngs = init_particle_rngs(seed, n_particles, 0, binomial);
    let mut resample_rng = StatefulRng::new_stream(seed, RESAMPLE_RNG_STREAM);

    // Each particle draws its own x₀ through the same seam the CSMC free
    // particles use, so the two cannot disagree about what a free draw is.
    let mut counts: Vec<Vec<i64>> = (0..n_particles)
        .map(|j| draw_free_particle_initial_state(model, params, &mut rngs[j], per_eval))
        .collect::<Result<Vec<_>, _>>()?;
    let initial_counts_per_particle: Vec<Vec<i64>> = counts.clone();

    let mut substep_flows: Vec<Vec<u64>> = vec![vec![0u64; n_tr]; n_particles];
    let mut substep_gammas: Vec<Vec<f64>> = vec![Vec::new(); n_particles];

    // KNOWN LIMITATION (docs/dev/incidents/2026-06-07-chain-binomial-stale-
    // real-state.md, §inference scope): like the CSMC free particles, these
    // track integer counts only — no real reservoir is advanced, so rates
    // coupling to a real compartment read 0. Real-coupled models already fail
    // the chain-binomial inference capability check (gh#191) upstream.
    let n_real = model.real_local_to_global.len();
    let mut particle_reals: Vec<RealState> =
        (0..n_particles).map(|_| RealState::new(n_real)).collect();

    let mut cum_flows: Vec<Vec<u64>> = vec![vec![0u64; n_tr]; n_particles];
    let n_acc = obs_model.n_interval_streams();
    let mut acc: Vec<Vec<u64>> = vec![vec![0u64; n_acc]; n_particles];

    let mut scratches: Vec<StepScratch> =
        (0..n_particles).map(|_| StepScratch::new(model)).collect();

    let mut history_counts_before: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_counts_after: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_flows: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_substeps);
    let mut history_gammas: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_substeps);
    let mut ancestors: Vec<Vec<usize>> = Vec::with_capacity(n_substeps);

    let mut log_weights = vec![0.0f64; n_particles];
    // gh#367: the shared per-particle death policy. A particle that dies keeps a
    // `−∞` weight until the next resample removes it, so it can never be an
    // ancestor of a surviving lineage.
    let mut deaths = DeathMask::new(n_particles);

    for (s, &(t, step_dt)) in grid.iter().enumerate() {
        // ── 1. Resample (unconditional, systematic) ──
        // Between observations every live weight is equal, so a resample would
        // only duplicate particles for nothing; the weights go non-uniform
        // exactly at the substep FOLLOWING an observation, and also as soon as
        // any particle is dead, which is when we want the dead purged.
        let weights_are_uniform = log_weights
            .iter()
            .all(|&w| (w - log_weights[0]).abs() < 1e-10);
        let substep_ancestors: Vec<usize> = if weights_are_uniform {
            (0..n_particles).collect()
        } else {
            let indices = systematic_resample(&log_weights, &mut resample_rng);
            let mut new_counts = Vec::with_capacity(n_particles);
            let mut new_cum_flows = Vec::with_capacity(n_particles);
            let mut new_acc = Vec::with_capacity(n_particles);
            for &src in &indices {
                new_counts.push(counts[src].clone());
                new_cum_flows.push(cum_flows[src].clone());
                new_acc.push(acc[src].clone());
            }
            counts = new_counts;
            cum_flows = new_cum_flows;
            acc = new_acc;
            // Any particle that survived a resample had finite weight, so it is
            // alive; the pre-resample dead vector is index-invalid anyway.
            deaths.clear();
            indices
        };

        let prev_counts: Vec<Vec<i64>> = counts.clone();

        // ── 2. Propagate every particle (parallel; gh#209) ──
        // Each particle writes only its own slot and draws from its own stream,
        // so this is byte-identical to the serial loop (the same CRN property
        // `csmc_as` and the bootstrap filter rely on).
        let outcomes: Vec<Result<bool, SimError>> = counts
            .par_iter_mut()
            .zip(substep_flows.par_iter_mut())
            .zip(particle_reals.par_iter_mut())
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .zip(substep_gammas.par_iter_mut())
            .zip(deaths.as_slice().par_iter())
            .map(
                |((((((cnt, flows), real), rng), scratch), gammas), &dead)| {
                    if dead {
                        return Ok(true);
                    }
                    for f in flows.iter_mut() {
                        *f = 0;
                    }
                    scratch.gamma_used.clear();
                    fill_producer_batch(
                        model,
                        &fire_steps,
                        t + step_dt,
                        dt,
                        s,
                        firing,
                        &mut scratch.effect_batch,
                    );
                    // `classify` is the shared policy: `?` tears the pass down on
                    // a non-recoverable error, `true` kills this particle only.
                    let died = DeathMask::classify(step_one(
                        model, cnt, flows, real, params, t, step_dt, per_eval, rng, scratch,
                    ))?;
                    std::mem::swap(gammas, &mut scratch.gamma_used);
                    Ok(died)
                },
            )
            .collect();
        deaths.absorb(outcomes)?;

        if deaths.all_dead() {
            return Ok(UnconditionalPass::NoSupport(
                InitFallback::AllParticlesDied { substep: s },
            ));
        }

        // ── 3. Accumulate this substep's flows into the observation interval ──
        for (cflows, flows) in cum_flows.iter_mut().zip(substep_flows.iter()) {
            for (c, &f) in cflows.iter_mut().zip(flows.iter()) {
                *c += f;
            }
        }

        // ── 4. Weight ──
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            log_weights
                .par_iter_mut()
                .zip(cum_flows.par_iter_mut())
                .zip(acc.par_iter_mut())
                .zip(counts.par_iter())
                .zip(deaths.as_slice().par_iter())
                .for_each(|((((lw, cflows), a), cnt), &dead)| {
                    // FOLD (Phase 2a): close the interval's per-transition flows
                    // into the per-stream bins BEFORE scoring, then reset — the
                    // per-transition tally blanket, the per-stream bins only for
                    // the Interval streams scheduled at this union index.
                    obs_model.fold_into_acc(cflows, a);
                    *lw = if dead {
                        f64::NEG_INFINITY
                    } else {
                        obs_model.log_likelihood_from_flows_and_counts(a, cnt, obs_idx, params)
                    };
                    for f in cflows.iter_mut() {
                        *f = 0;
                    }
                    obs_model.reset_due_acc(obs_idx, a);
                });
            if !log_weights.iter().any(|w| w.is_finite()) {
                return Ok(UnconditionalPass::NoSupport(InitFallback::SwarmCollapsed {
                    obs_index: obs_idx,
                    substep: s,
                }));
            }
        } else {
            for (lw, &dead) in log_weights.iter_mut().zip(deaths.as_slice()) {
                *lw = if dead { f64::NEG_INFINITY } else { 0.0 };
            }
        }

        // ── 5. Store history ──
        history_counts_before.push(prev_counts);
        history_counts_after.push(counts.clone());
        history_flows.push(substep_flows.clone());
        history_gammas.push(substep_gammas.clone());
        ancestors.push(substep_ancestors);
    }

    // ── Draw one lineage and walk it back ──
    // `sample_categorical_log` returns `None` only when every final weight is
    // `−∞`, which the per-observation check above already refuses at a scoring
    // substep; it can still happen when the LAST substep carries no observation
    // and every particle died there.
    let Some(k) = sample_categorical_log(&log_weights, &mut resample_rng) else {
        return Ok(UnconditionalPass::NoSupport(
            InitFallback::AllParticlesDied {
                substep: n_substeps.saturating_sub(1),
            },
        ));
    };

    let mut substeps = Vec::with_capacity(n_substeps);
    let mut particle = k;
    for s in (0..n_substeps).rev() {
        substeps.push(SubstepRecord {
            counts_before: history_counts_before[s][particle].clone(),
            counts_after: history_counts_after[s][particle].clone(),
            flows: history_flows[s][particle].clone(),
            gammas: history_gammas[s][particle].clone(),
            t0: grid[s].0,
            dt_substep: grid[s].1,
        });
        particle = ancestors[s][particle];
    }
    substeps.reverse();

    // The traceback tiles the grid contiguously, each duration in (0, dt] — the
    // same exact-tiling invariant `csmc_as`'s traceback checks.
    if cfg!(debug_assertions) {
        let mut prev_end = t_start;
        for (s, rec) in substeps.iter().enumerate() {
            debug_assert!(
                rec.dt_substep > 0.0 && rec.dt_substep <= dt + 1e-9,
                "init traceback substep {s}: dt_substep {} not in (0, dt={dt}]",
                rec.dt_substep
            );
            debug_assert!(
                (rec.t0 - prev_end).abs() < 1e-9,
                "init traceback substep {s}: t0 {} not contiguous with previous end {prev_end}",
                rec.t0
            );
            prev_end = rec.t0 + rec.dt_substep;
        }
    }

    Ok(UnconditionalPass::Path(PGASTrajectory {
        initial_counts: initial_counts_per_particle[particle].clone(),
        substeps,
    }))
}
