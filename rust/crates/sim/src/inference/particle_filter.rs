//! Bootstrap particle filter (Gordon, Salmond & Smith 1993).
//!
//! Estimates log p(y_{1:T} | θ) via sequential importance sampling
//! with systematic resampling. Uses the ProcessModel trait to
//! advance particles — any simulation backend works (chain-binomial,
//! tau-leap, etc.).

use rayon::prelude::*;

use crate::rng::StatefulRng;
use crate::error::SimError;
use super::traits::{ProcessModel, ObservationModel, SMCConfig};
use super::types::{ParticleState, ParticleSwarm, log_sum_exp};
use super::resampling::systematic_resample;
/// Observation: one data point at a specific time.
#[derive(Clone)]
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

    // Initialize particles from model init
    let init = process.initial_state(params)?;
    let mut swarm = ParticleSwarm::new(n_particles, n_int, n_tr);
    for p in &mut swarm.states {
        p.counts.copy_from_slice(&init.counts);
    }

    // Per-particle RNG streams (deterministic, derived from seed)
    let mut rngs: Vec<StatefulRng> = (0..n_particles)
        .map(|i| StatefulRng::new_stream(seed, i as u64))
        .collect();

    // Separate RNG streams for diagnostic draws (rmeasure).
    // Process RNG streams must be identical whether or not predictions are computed.
    let mut diag_rngs: Vec<StatefulRng> = (0..n_particles)
        // Diagnostic RNG stream — offset by 2^62 so process-RNG and
        // diag-RNG streams never overlap (u64 stream id is 64 bits;
        // 2^62 is a comfortable gap from the low-indexed process streams).
        .map(|i| StatefulRng::new_stream(seed, (i as u64) | (1u64 << 62)))
        .collect();

    // Double-buffer for resampling (avoids clone allocation)
    let mut states_buf: Vec<ParticleState> = (0..n_particles)
        .map(|_| ParticleState::new(n_int, n_tr))
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

    // Resampling RNG (separate from particle RNGs)
    let mut resample_rng = StatefulRng::new(seed.wrapping_add(0xdeadbeef));

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

    for obs_idx in 0..n_obs {
        let obs_time = obs_model.obs_time(obs_idx);

        // Propagate all particles from t to obs_time.
        let t_start_interval = t;
        let errors: Vec<Result<(), SimError>> = swarm.states.par_iter_mut()
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .map(|((state, rng), scratch)| {
                let mut t_local = t_start_interval;
                while t_local < obs_time - 1e-10 {
                    let step_dt = dt.min(obs_time - t_local);
                    process.step(state, params, t_local, step_dt, rng, scratch)?;
                    t_local += step_dt;
                }
                Ok(())
            })
            .collect();
        for r in errors { r?; }
        while t < obs_time - 1e-10 { t += dt.min(obs_time - t); }

        // Prediction diagnostics
        if has_predictions {
            let means: Vec<f64> = swarm.states.iter()
                .map(|s| obs_model.mean(s, obs_idx, params).into_iter().sum::<f64>())
                .collect();
            let equal_lw = vec![0.0_f64; n_particles];
            let (state_mean, state_q05, state_q50, state_q95) = weighted_quantiles(&means, &equal_lw);

            let obs_draws: Vec<f64> = swarm.states.iter().enumerate()
                .map(|(i, s)| obs_model.sample(s, obs_idx, params, &mut diag_rngs[i]).into_iter().sum())
                .collect();
            let (_, obs_q05, obs_q50, obs_q95) = weighted_quantiles(&obs_draws, &equal_lw);

            let obs_mean = means.iter().sum::<f64>() / n_particles as f64;
            predictions.push(PredictionDiag {
                obs_mean, obs_q05, obs_q50, obs_q95,
                state_mean, state_q05, state_q50, state_q95,
            });
        }

        // Compute log-weights via observation model
        for (i, state) in swarm.states.iter().enumerate() {
            swarm.log_weights[i] = obs_model.log_likelihood(state, obs_idx, params);
        }

        // Record prequential ingredients BEFORE resampling. Particles
        // are currently distributed as the one-step-ahead predictive
        // p(x_t | y_{1:t-1}); pre-obs weights (prior this obs) are
        // uniform, so the caller computes log-score as
        // logsumexp(log_liks) − log N. Samples come from the same
        // particles via obs_model.sample and feed CRPS/PIT.
        if config.record_prequential {
            let log_liks: Vec<f64> = swarm.log_weights.clone();
            let y_draws: Vec<f64> = swarm.states.iter().enumerate()
                .map(|(i, s)| obs_model.sample(s, obs_idx, params, &mut diag_rngs[i])
                    .into_iter().sum::<f64>())
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
            history_states.push(step_states);
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

        // Resample via double-buffer
        let indices = systematic_resample(&swarm.log_weights, &mut resample_rng);
        for (i, &src) in indices.iter().enumerate() {
            states_buf[i].counts.copy_from_slice(&swarm.states[src].counts);
            states_buf[i].flow_accumulators.copy_from_slice(&swarm.states[src].flow_accumulators);
        }
        std::mem::swap(&mut swarm.states, &mut states_buf);

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
        for state in &mut swarm.states {
            state.reset_flows();
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

    // Normalize weights
    let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = if max_lw.is_infinite() {
        vec![1.0 / n as f64; n]
    } else {
        let raw: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let sum: f64 = raw.iter().sum();
        if sum == 0.0 { vec![1.0 / n as f64; n] }
        else { raw.iter().map(|&w| w / sum).collect() }
    };

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
