//! Deterministic ODE marginal likelihood `p(y | θ, ODE_skeleton)`.
//!
//! Phase 1 of the ODE-inference proposal
//! (`docs/dev/proposals/2026-05-04-ode-inference-three-phase.md`).

use crate::compiled_model::CompiledModel;
use crate::inference::MultiStreamObsModel;

/// Evaluate `p(y | θ, ODE_skeleton)` — the deterministic marginal likelihood.
///
/// Runs `OdeSim` once, then sums `MultiStreamObsModel::log_likelihood_*` over
/// each obs time. Snapshot states are rounded to integer counts at snapshot
/// time (per the ODE backend's standard `to_states` path) — at typhoid-class N
/// the rounding-induced loglik change is sub-nat and well below NLopt's xtol;
/// Phase 3 (NUTS) will revisit this with a real-valued obs eval to avoid
/// integer-boundary discontinuities.
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
) -> Result<f64, crate::error::SimError> {
    use crate::config::{OdeConfig, SimConfig};
    use crate::ode::OdeSim;
    use crate::Simulate;

    let model_sim = &compiled.model.simulation;
    let ode_cfg = OdeConfig {
        t_start: model_sim.t_start,
        t_end: model_sim.t_end,
        dt,
    };
    let traj = OdeSim.run(
        compiled,
        params,
        /* seed */ 0,
        &SimConfig::Ode(ode_cfg),
    )?;

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
    let mut total_ll = 0.0;

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
            let ll = obs_model.log_likelihood_from_flows_and_counts_real(
                &acc,
                &snap.int_state.counts,
                next_obs_idx,
                params,
            );
            if !ll.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            total_ll += ll;
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
    Ok(total_ll)
}
