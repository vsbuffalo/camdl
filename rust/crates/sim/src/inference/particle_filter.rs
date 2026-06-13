//! Bootstrap particle filter (Gordon, Salmond & Smith 1993).
//!
//! Estimates log p(y_{1:T} | θ) via sequential importance sampling
//! with systematic resampling. Uses the ProcessModel trait to
//! advance particles — any simulation backend works (chain-binomial,
//! ODE, etc.).

use std::time::Instant;

use rayon::prelude::*;

use crate::rng::StatefulRng;
use crate::error::SimError;
use crate::schedule::{Cursor, Schedule, StepPolicy};
use super::degeneracy::{check_pf_degeneracy, check_iteration_budget, window_substep_cost, pf_bail_error, pf_wallclock_budget, ITER_BUDGET};
use super::traits::{ProcessModel, ObservationModel, SMCConfig};
use super::types::{ParticleState, ParticleSwarm, log_sum_exp, normalize_log_weights, RESAMPLE_RNG_STREAM, init_particle_rngs};
use super::resampling::systematic_resample;
/// Observation: one data point at a specific time.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub time: f64,
    pub value: f64,
}

/// One-step-ahead prediction diagnostics at a single observation time.
#[derive(Clone, Debug)]
pub struct PredictionDiag {
    /// Observation-space: E[y | projected] averaged across particles.
    pub obs_mean: f64,
    /// Observation-space quantiles (process + observation noise).
    pub obs_q05: f64,
    pub obs_q50: f64,
    pub obs_q95: f64,
    /// Latent state quantiles (process uncertainty only).
    pub state_mean: f64,
    pub state_q05: f64,
    pub state_q50: f64,
    pub state_q95: f64,
}

/// Result of a particle filter run.
pub struct PFilterResult {
    /// Estimated log p(y_{1:T} | θ).
    pub log_likelihood: f64,
    /// ESS at each observation time.
    pub ess_trace: Vec<f64>,
    /// Log-likelihood increment at each observation time.
    pub ll_increments: Vec<f64>,
    /// One-step-ahead prediction diagnostics at each observation time.
    /// Only populated when obs model supports sample/mean.
    pub predictions: Option<Vec<PredictionDiag>>,
    /// Final particle states after the last observation (post-resampling).
    /// Only populated when `save_final_state` is true.
    pub final_states: Option<Vec<ParticleState>>,
    /// Per-step pre-resample particle states + ancestry, populated
    /// when `SMCConfig.record_ancestry = true`. Feed this to
    /// `ancestor_trace::sample_paths` for smoothing draws.
    pub ancestry: Option<super::ancestor_trace::AncestorTrace>,
    /// Per-step per-particle predictive samples and log-likelihoods,
    /// populated when `SMCConfig.record_prequential = true`. Feed to
    /// `prequential::build_trace` with the observation series to build
    /// a `PrequentialTrace`.
    pub prequential: Option<PrequentialRecorded>,
}

/// Raw per-step ingredients for prequential trace construction.
///
/// Captured BEFORE obs-reweight and BEFORE resampling, so particles
/// are distributed as the one-step-ahead predictive
/// p(x_t | y_{1:t-1}). In the bootstrap filter the pre-obs weights
/// are uniform (reset to 0 at the end of the previous step), so the
/// caller can compute log-score = logsumexp(log_liks) − log N.
pub struct PrequentialRecorded {
    /// Observation time for each recorded step, length = n_obs.
    pub obs_times: Vec<f64>,
    /// `[obs_idx][particle]` = log p(y_t | x_t^(p), θ).
    pub log_liks: Vec<Vec<f64>>,
    /// `[obs_idx][particle]` = sum across streams of ỹ^(p) ∼ p(y | x_t^(p), θ).
    pub y_pred_samples: Vec<Vec<f64>>,
}

/// Run the bootstrap particle filter.
///
/// # Arguments
/// * `process` — process model (advance state by dt)
/// * `obs_model` — observation model (log-likelihood, sample, mean)
/// * `params` — parameter values
/// * `config` — SMC config (n_particles, dt)
/// * `seed` — RNG seed
pub fn bootstrap_filter<P: ProcessModel<State = ParticleState>>(
    process: &P,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    params: &[f64],
    config: &SMCConfig,
    seed: u64,
) -> Result<PFilterResult, SimError> {
    let n_particles = config.n_particles;
    let dt = config.dt;
    let n_obs = obs_model.n_observations();
    let n_int = process.n_compartments();
    let n_tr = process.n_transitions();
    // Per-Interval-stream `acc` bins (multi-cadence Phase 2a): one `u64` per
    // incidence stream, sized from the OBS model (the process does not know it).
    let n_acc = obs_model.n_interval_streams();

    // Initialize particles from model init. The process sizes `acc` to 0 (it
    // does not know `n_interval_streams`); resize the init state's `acc` to
    // `n_acc` so the `has_predictions` probe (`obs_model.mean(&init, …)`, which
    // projects `acc[k]`) does not index out of bounds. Swarm states are sized
    // by `ParticleSwarm::new`.
    let mut init = process.initial_state(params)?;
    init.acc.resize(n_acc, 0);
    let mut swarm = ParticleSwarm::new(n_particles, n_int, n_tr, n_acc);
    for p in &mut swarm.states {
        p.counts.copy_from_slice(&init.counts);
    }

    // Per-particle RNG streams (deterministic, derived from seed).
    // stream_offset = 0: particles use stream indices [0, n_particles).
    let mut rngs = init_particle_rngs(seed, n_particles, 0);

    // Separate RNG streams for diagnostic draws (rmeasure).
    // Process RNG streams must be identical whether or not predictions are computed.
    // Offset by 2^62 so process-RNG and diag-RNG streams never overlap
    // (u64 stream id is 64 bits; 2^62 is a comfortable gap from low-indexed streams).
    let mut diag_rngs = init_particle_rngs(seed, n_particles, 1u64 << 62);

    // Double-buffer for resampling (avoids clone allocation)
    let mut states_buf: Vec<ParticleState> = (0..n_particles)
        .map(|_| ParticleState::new(n_int, n_tr, n_acc))
        .collect();

    // Per-particle scratch buffers (allocated once, reused across all steps)
    let mut scratches: Vec<P::Scratch> = (0..n_particles)
        .map(|_| process.new_scratch())
        .collect();

    let mut total_loglik = 0.0;
    let mut ess_trace = Vec::with_capacity(n_obs);
    let mut ll_increments = Vec::with_capacity(n_obs);
    let has_predictions = obs_model.n_streams() > 0 && !obs_model.mean(&init, 0, params).is_empty();
    let mut predictions: Vec<PredictionDiag> = if has_predictions {
        Vec::with_capacity(n_obs)
    } else {
        Vec::new()
    };

    let mut t = config.t_start;

    // Merged timeline spine for the inference path: the EXACT policy clips each
    // substep to the next OBSERVATION boundary (the bootstrap PF steps exactly to
    // obs times, where it scores). The Schedule owns the obs times; a per-particle
    // `Cursor` (Copy) walks them, so the immutable Schedule is shared across the
    // parallel swarm without breaking CRN. step_dt = substep(cursor, t) reproduces
    // dt.min(obs_time - t) exactly. (Substep TIME stays accumulated here — the s*dt
    // convention for the EXACT steppers is deferred, task #14.)
    let obs_times: Vec<f64> = (0..n_obs).map(|i| obs_model.obs_time(i)).collect();

    // gh#216: scheduled interventions fire CURSOR-keyed off the timeline's effect
    // boundaries (registered as `effect_times` below), NOT on the `round(t/dt)`
    // key inside step_one — so an off-grid observation re-tiling the Exact substep
    // grid no longer moves the firing instant. Two Exact cases stay unsupported
    // and are refused loudly: a parametric `at [<param>]` schedule (one shared
    // `effect_times` can't hold per-particle times), and a scheduled fire time off
    // the dt grid (the drift-free PGAS walk would need to re-anchor there — a
    // deferred follow-up). Always-active events are out of scope (grid_dt-keyed).
    let scheduled = if let Some(model) = process.try_compiled_model() {
        crate::intervention::guard_attimesexpr_exact(model, StepPolicy::Exact)?;
        crate::intervention::guard_exact_offgrid_effect_time(
            model, params, config.t_start, dt, StepPolicy::Exact,
        )?;
        crate::intervention::scheduled_effects(model, params)
    } else {
        crate::intervention::ScheduledEffects::default()
    };

    let sched_t_end = obs_times.last().copied().unwrap_or(config.t_start);
    let schedule = Schedule::new(
        dt, sched_t_end, dt, StepPolicy::Exact, Vec::new(), scheduled.times.clone(),
    )
    .with_obs(obs_times);

    // gh#147 (M3.1). Cumulative particle-substep count for the
    // deterministic compute-budget guard. Bounds a single PF evaluation;
    // see `ITER_BUDGET`. Checked BEFORE each window's propagation so a
    // pathological dt aborts before the substep loop runs (and hangs).
    let mut iters: u64 = 0;

    // gh#110. Wall-clock timer for the degeneracy watchdog. Started
    // here, checked after each observation window via
    // `check_pf_degeneracy`. The watchdog returns `Err(PFDegenerate)`
    // which propagates through the existing `Err → NEG_INFINITY`
    // collapse in `run_quick_pfilter_with_dt`; PMMH already rejects
    // -∞ proposals, so no caller-side change is needed for the
    // common path. Init-eval callers detect the bail explicitly.
    let t0_call = Instant::now();
    // gh#147 (M3.2). Resolve the wall-clock budget once from config: a CAS
    // fit disables it (`None`) for deterministic loglik; the deterministic
    // substep cap remains the compute-blowup guard.
    let pf_wc = pf_wallclock_budget(config.pf_wallclock_disabled, n_particles);

    // Resampling RNG — reserved stream index, never collides with particle streams.
    let mut resample_rng = StatefulRng::new_stream(seed, RESAMPLE_RNG_STREAM);

    // Ancestry recording (allocated only if requested).
    let mut history_states: Vec<Vec<Vec<f64>>> = if config.record_ancestry {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };
    let mut history_lw: Vec<Vec<f64>> = if config.record_ancestry {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };
    let mut history_ancestors: Vec<Vec<usize>> = if config.record_ancestry {
        Vec::with_capacity(n_obs.saturating_sub(1))
    } else { Vec::new() };
    let mut history_times: Vec<f64> = if config.record_ancestry {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };
    // gh#48: per-step per-particle per-stream projections. Computed via
    // `obs_model.mean(state, obs_idx, params)` at the same point states
    // are recorded — pre-resample, pre-flow-reset, so flow accumulators
    // are still populated for incidence projections. Empty per step
    // when `obs_model.mean()` returns `vec![]` (the trait default for
    // impls that don't override).
    let mut history_projections: Vec<Vec<Vec<f64>>> = if config.record_ancestry {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };

    // Prequential recording (allocated only if requested).
    let mut preq_times: Vec<f64> = if config.record_prequential {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };
    let mut preq_log_liks: Vec<Vec<f64>> = if config.record_prequential {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };
    let mut preq_samples: Vec<Vec<f64>> = if config.record_prequential {
        Vec::with_capacity(n_obs)
    } else { Vec::new() };

    // gh#audit-C5 / C6. Particles that hit a per-particle-recoverable
    // SimError (NumericalCollapse, NegativeCount{BinomialOvershoot})
    // get marked dead; their log-weight is set to −Inf so resampling
    // kills them. Hard errors (UnknownCompartment, config bugs, etc.)
    // still propagate immediately — they are not particle-specific.
    let mut particle_dead: Vec<bool> = vec![false; n_particles];

    for obs_idx in 0..n_obs {
        let obs_time = obs_model.obs_time(obs_idx);

        // gh#147 (M3.1). Deterministic compute-budget guard, PRE-window.
        // The per-window substep cost `n_particles · ceil((obs_time−t)/dt)`
        // is a closed-form scalar (parallel-invariant), so this fires
        // identically regardless of thread count and BEFORE the substep
        // loop below — a sub-nanosecond dt aborts here rather than wedging
        // the propagation. Replaces the machine-speed-dependent wall-clock
        // watchdog's compute-blowup role; the wall-clock check (below,
        // post-window) remains for genuinely-slow-but-bounded filters.
        let cost = window_substep_cost(n_particles, t, obs_time, dt);
        if let Some(kind) = check_iteration_budget(iters, cost, ITER_BUDGET) {
            return Err(pf_bail_error(kind, obs_idx, t0_call.elapsed().as_secs_f64()));
        }
        iters = iters.saturating_add(cost);

        // Propagate all particles from t to obs_time. The schedule clips each
        // substep to obs_time; the cursor points at this observation. The effect
        // cursor is positioned at the first scheduled-effect boundary not yet
        // fired by `t` so the monotone effect walk carries across windows (gh#216).
        let t_start_interval = t;
        let cur = Cursor {
            obs_idx,
            effect_idx: schedule.effect_idx_at(t_start_interval),
            ..Default::default()
        };
        let outcomes: Vec<Result<bool, SimError>> = swarm.states.par_iter_mut()
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .zip(particle_dead.par_iter())
            .map(|(((state, rng), scratch), &dead)| {
                if dead { return Ok(true); }  // already dead; skip
                // Shared inner-substep walk (Schedule::substeps); this body keeps
                // the per-particle death-on-recoverable-error policy. `fired` is
                // Some(effect_idx) when this substep lands on a scheduled-effect
                // boundary — fire that boundary's batch cursor-keyed.
                for (t_local, step_dt, fired) in schedule.substeps(cur, t_start_interval) {
                    let due_iv: &[usize] = match fired {
                        Some(idx) => &scheduled.batches[idx],
                        None => &[],
                    };
                    match process.step(state, params, t_local, step_dt, rng, scratch, due_iv) {
                        Ok(()) => {}
                        Err(e) if e.is_per_particle_recoverable() => {
                            // Mark dead — the caller folds this into the dead vec
                            // and the outer loop sets log_weight = −∞.
                            return Ok(true);
                        }
                        Err(e) => return Err(e),
                    }
                }
                Ok(false)
            })
            .collect();
        for (i, r) in outcomes.into_iter().enumerate() {
            if r? { particle_dead[i] = true; }
        }
        t = schedule.window_end(cur, t);

        // FOLD (multi-cadence Phase 2a, "Option Z"): close this interval's flow
        // into each Interval stream's persistent `acc` bin, ONCE per
        // observation (serial — NOT inside the per-substep par_iter above). The
        // `flow_accumulators` tally is left untouched (blanket-zeroed only at
        // the per-obs reset below, exactly as before). Every reader of `acc`
        // downstream this iteration (predictions via `mean`/`sample`, the gh#48
        // capture, scoring) sees the just-folded bin.
        for state in &mut swarm.states {
            obs_model.fold_into_acc(&state.flow_accumulators, &mut state.acc);
        }

        // Prediction diagnostics
        if has_predictions {
            // Multi-cadence: `mean`/`sample` return `f64::NAN` for a stream NOT
            // scheduled at this union index (proposal 2026-06-10 §3.6). Filter
            // the non-finite entries before summing — a not-scheduled stream
            // contributes nothing to the summed prediction at this union time.
            // Homogeneous (every stream scheduled at every index) never yields a
            // NaN, so this filter is a no-op there.
            let means: Vec<f64> = swarm.states.iter()
                .map(|s| obs_model.mean(s, obs_idx, params)
                    .into_iter().filter(|v| v.is_finite()).sum::<f64>())
                .collect();
            let equal_lw = vec![0.0_f64; n_particles];
            let (state_mean, state_q05, state_q50, state_q95) = weighted_quantiles(&means, &equal_lw);

            let obs_draws: Vec<f64> = swarm.states.iter().enumerate()
                .map(|(i, s)| obs_model.sample(s, obs_idx, params, &mut diag_rngs[i])
                    .into_iter().filter(|v| v.is_finite()).sum())
                .collect();
            let (_, obs_q05, obs_q50, obs_q95) = weighted_quantiles(&obs_draws, &equal_lw);

            let obs_mean = means.iter().sum::<f64>() / n_particles as f64;
            predictions.push(PredictionDiag {
                obs_mean, obs_q05, obs_q50, obs_q95,
                state_mean, state_q05, state_q50, state_q95,
            });
        }

        // Compute log-weights via observation model. Dead particles
        // (gh#audit-C5/C6) get −Inf so resampling discards them.
        for (i, state) in swarm.states.iter().enumerate() {
            swarm.log_weights[i] = if particle_dead[i] {
                f64::NEG_INFINITY
            } else {
                obs_model.log_likelihood(state, obs_idx, params)
            };
        }

        // Record prequential ingredients BEFORE resampling. Particles
        // are currently distributed as the one-step-ahead predictive
        // p(x_t | y_{1:t-1}); pre-obs weights (prior this obs) are
        // uniform, so the caller computes log-score as
        // logsumexp(log_liks) − log N. Samples come from the same
        // particles via obs_model.sample and feed CRPS/PIT.
        if config.record_prequential {
            let log_liks: Vec<f64> = swarm.log_weights.clone();
            // Multi-cadence: drop the not-scheduled streams' NaN before summing
            // (else the prequential sample poisons CRPS/PIT). Homogeneous is a
            // no-op (no NaN). See the prediction block above.
            let y_draws: Vec<f64> = swarm.states.iter().enumerate()
                .map(|(i, s)| obs_model.sample(s, obs_idx, params, &mut diag_rngs[i])
                    .into_iter().filter(|v| v.is_finite()).sum::<f64>())
                .collect();
            preq_times.push(obs_time);
            preq_log_liks.push(log_liks);
            preq_samples.push(y_draws);
        }

        // Record pre-resample filtering state (states + weights) so
        // the caller can reconstruct filtering marginals or, paired
        // with the ancestor indices recorded below, sample smoothing
        // paths. Allocates N×K counts + N weights per obs step; only
        // enabled when the caller opts in.
        if config.record_ancestry {
            // Convert i64 compartment counts → f64 at record time;
            // downstream (path sampling, quantile ribbons) wants
            // real-valued arithmetic. Real-compartment backends would
            // already be f64 conceptually and fall into the same
            // representation here.
            let step_states: Vec<Vec<f64>> = swarm.states.iter()
                .map(|s| s.counts.iter().map(|&c| c as f64).collect())
                .collect();
            // gh#48: capture per-particle per-stream projections via
            // the obs model's `mean()`. This is the model's predicted
            // observation — what `incidence(recovery)` evaluates to,
            // scaled by however the user wrote the likelihood (e.g.
            // `rho * projected`). Recording here (pre-resample,
            // pre-flow-reset) is the only point where flow_accumulators
            // carry the just-completed obs interval's flow integrals,
            // which is what incidence projections need. After
            // resampling + reset two lines below, flow_accumulators
            // start the next interval at zero.
            //
            // Multi-cadence: this stores the PER-STREAM vector unchanged. A
            // stream NOT scheduled at this union index yields `f64::NAN` from
            // `mean()` (proposal 2026-06-10 §3.6). That NaN is NOT summed — it
            // is walked along the ancestor chain into its OWN per-stream column
            // in `--save-paths` (`write_sampled_paths`, which already emits NaN
            // for an absent stream cell). So a not-scheduled stream reads NaN in
            // its column at that union row — the honest "no observation here"
            // marker, NOT a fictitious 0. Homogeneous never produces a NaN.
            let step_projections: Vec<Vec<f64>> = swarm.states.iter()
                .map(|s| obs_model.mean(s, obs_idx, params))
                .collect();
            history_states.push(step_states);
            history_projections.push(step_projections);
            history_lw.push(swarm.log_weights.clone());
            history_times.push(obs_time);
        }

        // Log-marginal increment. Under IC-free inference
        // (`skip_first_obs_from_loglik`), we still compute the
        // reweight-and-resample at the first observation — that's what
        // pins x_0 given y_1 — but we don't accumulate it into the
        // returned log-likelihood. Subsequent observations contribute
        // normally, giving the conditional likelihood
        //   log L_c(θ | y_1) = Σ_{t=2}^{T} log p(y_t | y_{1:t-1}).
        // See docs/dev/proposals/2026-04-18-ic-free-inference.md.
        let ll_increment = log_sum_exp(&swarm.log_weights) - (n_particles as f64).ln();
        if !(config.skip_first_obs_from_loglik && obs_idx == 0) {
            total_loglik += ll_increment;
        }
        ll_increments.push(ll_increment);
        ess_trace.push(swarm.ess());

        // gh#110. Degeneracy watchdog. Check AFTER pushing the current
        // ESS — `check_pf_degeneracy` reads the K-window history off
        // `ess_trace`. Fires on ESS collapse, wall-clock timeout, or
        // every particle dead. The `obs_idx` and elapsed are
        // captured into the error variant so the diagnostic can
        // surface where the bail happened.
        let dead_count = particle_dead.iter().filter(|&&d| d).count();
        let elapsed = t0_call.elapsed();
        if let Some(kind) = check_pf_degeneracy(
            &ess_trace, elapsed, pf_wc, obs_idx, dead_count, n_particles,
        ) {
            // gh#133: WallClockExceeded → PFWallclockTimeout (resource limit),
            // the rest → PFDegenerate (statistical). Single mapping helper.
            return Err(super::degeneracy::pf_bail_error(
                kind, obs_idx, elapsed.as_secs_f64(),
            ));
        }

        // Resample via double-buffer
        let indices = systematic_resample(&swarm.log_weights, &mut resample_rng);
        for (i, &src) in indices.iter().enumerate() {
            states_buf[i].counts.copy_from_slice(&swarm.states[src].counts);
            states_buf[i].flow_accumulators.copy_from_slice(&swarm.states[src].flow_accumulators);
            // Phase 2a: the per-stream `acc` bins travel with the particle, so
            // an ancestor swap carries the right partial bins for streams not
            // observed at this union index.
            states_buf[i].acc.copy_from_slice(&swarm.states[src].acc);
        }
        std::mem::swap(&mut swarm.states, &mut states_buf);

        // gh#audit-C5/C6: clear particle_dead after resampling. Any
        // particle that survived the systematic resample had finite
        // weight, so it can't be dead. Clearing is correct because
        // resampling shuffles particles by index, invalidating the
        // pre-resample dead vector anyway.
        for d in &mut particle_dead { *d = false; }

        // Record the resampling indices as the ancestor map for the
        // NEXT step. Not needed after the last observation (no step
        // t+1 to map into), so we skip recording on the final pass.
        if config.record_ancestry && obs_idx + 1 < n_obs {
            history_ancestors.push(indices);
        }

        // Reset flow accumulators for next observation interval.
        //
        // Im5 in 2026-04-19 inference review: resets ALL flow
        // accumulators indiscriminately, not only those referenced by
        // FlowSum-projected streams. Safe because:
        //   (a) snapshot/prevalence streams don't consume flows;
        //   (b) disjoint FlowSum subsets don't share accumulator
        //       indices;
        //   (c) overlapping subsets both reset to zero anyway.
        // If a future feature ever stores "flow since the most recent
        // per-stream observation" at different cadences per stream,
        // this reset needs to become per-flow and indexed by which
        // stream last observed. Keep this comment as the canary.
        //
        // Phase 2a (the feature the canary predicted): `flow_accumulators` is
        // STILL blanket-reset here (its lifecycle unchanged); the per-stream
        // `acc` bins are reset SEPARATELY and per-stream — only the Interval
        // streams scheduled at THIS union index (`at_union[obs_idx].is_some()`)
        // zero, so a sibling on a different cadence keeps its running bin.
        // Homogeneous (every stream scheduled every interval) ⇒ every `acc`
        // zeroes every interval ⇒ identical to the blanket reset.
        for state in &mut swarm.states {
            state.reset_flows();
            obs_model.reset_due_acc(obs_idx, &mut state.acc);
        }

        // Reset weights
        for lw in &mut swarm.log_weights { *lw = 0.0; }
    }

    let ancestry = if config.record_ancestry {
        Some(super::ancestor_trace::AncestorTrace {
            n_compartments: n_int,
            states: history_states,
            log_weights: history_lw,
            ancestors: history_ancestors,
            obs_times: history_times,
            projections: history_projections,
            stream_names: obs_model.stream_names(),
        })
    } else {
        None
    };

    let prequential = if config.record_prequential {
        Some(PrequentialRecorded {
            obs_times: preq_times,
            log_liks: preq_log_liks,
            y_pred_samples: preq_samples,
        })
    } else {
        None
    };

    Ok(PFilterResult {
        log_likelihood: total_loglik,
        predictions: if has_predictions { Some(predictions) } else { None },
        ess_trace,
        ll_increments,
        final_states: Some(swarm.states),
        ancestry,
        prequential,
    })
}

/// Weighted mean and quantiles from log-weighted samples.
/// Returns (mean, q05, q50, q95).
fn weighted_quantiles(values: &[f64], log_weights: &[f64]) -> (f64, f64, f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let weights = normalize_log_weights(log_weights);

    // Weighted mean
    let mean: f64 = values.iter().zip(&weights).map(|(&v, &w)| v * w).sum();

    // Weighted quantiles: sort by value, walk cumulative weight
    let mut sorted: Vec<(f64, f64)> = values.iter().zip(&weights).map(|(&v, &w)| (v, w)).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let quantile = |p: f64| -> f64 {
        let mut cumw = 0.0;
        for &(val, w) in &sorted {
            cumw += w;
            if cumw >= p { return val; }
        }
        sorted.last().map_or(0.0, |&(v, _)| v)
    };

    (mean, quantile(0.05), quantile(0.50), quantile(0.95))
}
