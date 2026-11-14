//! Deterministic ODE marginal likelihood `p(y | θ, ODE_skeleton)`, and the
//! per-observation record the ODE prequential is built from (gh#312).
//!
//! Phase 1 of the ODE-inference proposal
//! (`docs/dev/proposals/2026-05-04-ode-inference-three-phase.md`).
//!
//! Both consumers walk the trajectory through one function,
//! [`walk_ode_observations`], so the snapshot walk, the per-stream bin fold
//! and the bin reset that define what a deterministic trajectory predicts at
//! each observation are written once: a score the prequential reports is the
//! term the likelihood summed.

use std::ops::ControlFlow;

use crate::compiled_model::CompiledModel;
use crate::inference::MultiStreamObsModel;
use crate::inference::particle_filter::PrequentialRecorded;
use crate::rng::StatefulRng;

/// RNG stream for the observation-model draws of an ODE prequential record
/// ([`ode_prequential_recorded`]). Disjoint from the randomized-PIT stream
/// (`prequential::PIT_RNG_STREAM`, 1<<49), so passing one seed to both cannot
/// correlate the predictive draws with the PIT randomization.
pub const ODE_PREDICTIVE_RNG_STREAM: u64 = 1u64 << 50;

/// Evaluate `p(y | θ, ODE_skeleton)` — the deterministic marginal likelihood.
///
/// Runs `OdeSim` once, then sums `MultiStreamObsModel::log_likelihood_*` over
/// each obs time. Snapshot states are rounded to integer counts at snapshot
/// time (per the ODE backend's standard `to_states` path) — at typhoid-class N
/// the rounding-induced loglik change is sub-nat and well below NLopt's xtol.
///
/// Callers pre-build `obs_model` and pass it in so the per-eval cost is one ODE
/// solve + one obs scoring pass, with no per-call obs-model reconstruction.
///
/// Returns `f64::NEG_INFINITY` if any obs likelihood evaluates to non-finite
/// (NaN, -inf): callers treat this as "model blew up at this θ" without
/// crashing.
pub fn compute_ode_loglik(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    params: &[f64],
    burnin_dt: f64,
) -> Result<f64, crate::error::SimError> {
    let mut total_ll = 0.0;
    let mut blew_up = false;
    walk_ode_observations(
        compiled, obs_model, obs_times, dt, params, burnin_dt,
        |obs_idx, acc, counts| {
            let ll = obs_model.log_likelihood_from_flows_and_counts_real(
                acc, counts, obs_idx, params,
            );
            if !ll.is_finite() {
                blew_up = true;
                return ControlFlow::Break(());
            }
            total_ll += ll;
            ControlFlow::Continue(())
        },
    )?;
    Ok(if blew_up { f64::NEG_INFINITY } else { total_ll })
}

/// The ODE skeleton's predictive at every observation, recorded in the shape
/// the prequential trace builder reads (`prequential::build_trace`) — the
/// deterministic counterpart of a particle filter's
/// [`PrequentialRecorded`] (gh#312).
///
/// Given θ the trajectory `x_t(θ)` is a point, so the one-step-ahead
/// predictive `p(y_t | y_{1:t-1}, θ)` is the observation density at that
/// point, `p(y_t | x_t(θ), θ)`, and conditioning on earlier observations
/// changes nothing. The record therefore carries ONE state support point per
/// step: `log_liks[k] = [log p(y_t | x_t(θ))]`, the exact term
/// [`compute_ode_loglik`] sums (and the per-stream terms it is the sum of),
/// so the builder's `logsumexp − ln 1` is that term itself, not an estimate.
/// The predictive draws are `samples` independent draws per stream from the
/// observation model at `x_t(θ)`, which is all CRPS, PIT and the interval
/// need; their number does not touch the log score.
///
/// A non-finite term is recorded as it is (the score at that step is `-inf`
/// or NaN) rather than abandoning the walk, so a trace shows which
/// observation the trajectory could not produce. An `Err` is the trajectory
/// itself failing (solver failure, a missing snapshot at an observation
/// time), exactly as for [`compute_ode_loglik`].
///
/// Draws come from `rng`; callers seed it on [`ODE_PREDICTIVE_RNG_STREAM`].
#[allow(clippy::too_many_arguments)]
pub fn ode_prequential_recorded(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    params: &[f64],
    burnin_dt: f64,
    samples: usize,
    rng: &mut StatefulRng,
) -> Result<PrequentialRecorded, crate::error::SimError> {
    let n_streams = obs_model.n_streams();
    let mut recorded = PrequentialRecorded {
        obs_times: Vec::with_capacity(obs_times.len()),
        log_liks: Vec::with_capacity(obs_times.len()),
        stream_names: obs_model.stream_names(),
        per_stream_log_liks: Vec::with_capacity(obs_times.len()),
        per_stream_samples: Vec::with_capacity(obs_times.len()),
    };
    walk_ode_observations(
        compiled, obs_model, obs_times, dt, params, burnin_dt,
        |obs_idx, acc, counts| {
            let per_stream = obs_model.log_likelihood_per_stream_from_flows_and_counts_real(
                acc, counts, obs_idx, params,
            );
            // The joint is the sum of the per-stream terms — the same sum
            // `log_likelihood_from_flows_and_counts_real` takes.
            let joint: f64 = per_stream.iter().sum();
            let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); n_streams];
            for _ in 0..samples {
                let y = obs_model.sample_from_flows_and_counts_real(
                    acc, counts, obs_idx, params, rng,
                );
                for (si, v) in y.into_iter().enumerate() {
                    draws[si].push(v);
                }
            }
            recorded.obs_times.push(obs_times[obs_idx]);
            recorded.log_liks.push(vec![joint]);
            recorded.per_stream_log_liks.push(per_stream.into_iter().map(|v| vec![v]).collect());
            recorded.per_stream_samples.push(draws);
            ControlFlow::Continue(())
        },
    )?;
    if recorded.obs_times.len() != obs_times.len() {
        return Err(crate::error::SimError::Validation(format!(
            "internal error: the ODE walk recorded {} of {} observations",
            recorded.obs_times.len(), obs_times.len())));
    }
    Ok(recorded)
}

/// Integrate the ODE skeleton at θ and visit each observation in order:
/// `visit(obs_idx, acc, counts)` sees the per-Interval-stream bins folded
/// over that observation's interval and the snapshot's integer counts, and
/// returns `Break` to stop the walk early (`Ok(())` then). The bins are
/// reset after each visit, per stream, as the likelihood requires.
fn walk_ode_observations(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    params: &[f64],
    burnin_dt: f64,
    mut visit: impl FnMut(usize, &[f64], &[i64]) -> ControlFlow<()>,
) -> Result<(), crate::error::SimError> {
    use crate::config::OdeConfig;

    // `obs_times` must be the model's own union axis: the loop below scores
    // `obs_model` by position along it (gh#833).
    obs_model.check_axis(obs_times).map_err(crate::error::SimError::Validation)?;

    let model_sim = &compiled.model.simulation;
    let ode_cfg = OdeConfig {
        t_start: model_sim.t_start,
        t_end: model_sim.t_end,
        dt,
    };
    // Coarse burn-in (gh#396 follow-on): coarsen the unscored warm-up
    // `[t_start, first_obs)`. `run_ode` treats `burnin_dt <= dt` as inactive
    // (fine path), so passing `Some` unconditionally is safe — the split is the
    // first obs time. Value-path only (deterministic likelihood); the augmented
    // gradient path derives the same split from `obs_times` itself.
    let coarse = obs_times
        .first()
        .map(|&cond_from| crate::ode::CoarseBurnin { burnin_dt, cond_from });
    let traj = crate::ode::run_ode(compiled, params, &ode_cfg, None, coarse)?;

    // Snapshot semantics: each snapshot's `flows` are accumulated since the
    // *previous* snapshot, with reset on every output time in
    // `model.output.times`. For a model with a fine output grid (e.g.
    // typhoid's daily `regular(0, 1, 18250)`) the per-snapshot flow is one
    // step's worth — NOT cumulative-since-last-obs.
    //
    // The chain_binomial PF avoids this by accumulating flows internally
    // between obs times. We do the same here: walk every snapshot, sum
    // flows into a running cumulative vector, and at each obs time hand
    // the running cumulative to the obs likelihood (then reset for the
    // next obs interval). This makes `incidence(infect[s,a])` =
    // "cumulative flow since the last obs" regardless of how fine the
    // model's output schedule is.
    let n_transitions = traj
        .snapshots
        .first()
        .map(|s| s.flows.len())
        .unwrap_or(0);
    // The ODE backend records real-valued flow (continuous `rate·dt`, never
    // rounded), so the running accumulators are `f64`: a sub-unit flow (a slow
    // transition such as TB reactivation) must survive into the likelihood
    // rather than quantize to 0 → `-∞`.
    let mut cum_flows: Vec<f64> = vec![0.0; n_transitions];
    // Phase 2a: per-Interval-stream persistent bin, folded once per obs interval
    // and reset per-stream — ODE-inference is the seventh reset site and scores
    // through the SAME seam as the particle filters (the `_real` siblings).
    let mut acc: Vec<f64> = vec![0.0; obs_model.n_interval_streams()];
    let mut next_obs_idx = 0;
    let n_obs = obs_times.len();

    for (snap_idx, snap) in traj.snapshots.iter().enumerate() {
        // The simulator emits a snapshot at t_start with zero flow; from
        // index 1 onward the flow vector is the per-interval accumulation.
        if snap_idx > 0 {
            for (i, &f) in snap.flows.as_real().iter().enumerate() {
                cum_flows[i] += f;
            }
        }

        // Drain any obs times that match this snapshot. (`while` rather
        // than `if` so a degenerate model with two obs at identical t
        // — they shouldn't, but defensively — doesn't drop one.)
        while next_obs_idx < n_obs
            && (snap.t - obs_times[next_obs_idx]).abs() < 1e-9
        {
            // FOLD (Phase 2a): close this interval's per-transition `cum_flows`
            // into the per-stream `acc` BEFORE scoring; score reads `acc`.
            obs_model.fold_into_acc_real(&cum_flows, &mut acc);
            if visit(next_obs_idx, &acc, &snap.int_state.counts).is_break() {
                return Ok(());
            }
            // Reset for the next obs interval. `cum_flows` blanket-zeroed
            // (unchanged); the per-stream `acc` bins per-stream — only Interval
            // streams scheduled at THIS union index zero.
            cum_flows.fill(0.0);
            obs_model.reset_due_acc_real(next_obs_idx, &mut acc);
            next_obs_idx += 1;
        }

        // If we've already overshot the next obs time without matching,
        // the model's output schedule doesn't include it — bail with a
        // clear diagnostic.
        if next_obs_idx < n_obs && snap.t > obs_times[next_obs_idx] + 1e-9 {
            return Err(crate::error::SimError::Validation(format!(
                "ODE trajectory has no snapshot at obs time {} (snap.t = \
                 {} overshot it). The model's [output] schedule must \
                 include every observation time; declare an explicit \
                 `output {{ at = [...] }}` block aligned to the data, or \
                 ensure the regular schedule's step divides obs intervals.",
                obs_times[next_obs_idx], snap.t,
            )));
        }
    }

    if next_obs_idx < n_obs {
        return Err(crate::error::SimError::Validation(format!(
            "ODE trajectory ended at t = {} before reaching obs time {}; \
             the model's simulate.to must extend at least to the last \
             obs time",
            traj.snapshots.last().map(|s| s.t).unwrap_or(f64::NAN),
            obs_times[next_obs_idx],
        )));
    }
    Ok(())
}
