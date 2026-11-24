//! Ancestor tracing: recover coherent per-particle trajectories from a
//! bootstrap particle filter run.
//!
//! After running the filter, the per-step pre-resample particle states
//! give you filtering marginals `p(x_t | y_{1..t})` at each `t` — but
//! joining particles across `t` by index does NOT give sample paths
//! from the smoothing distribution `p(x_{1:T} | y_{1:T})`, because
//! resampling rearranges the swarm at every step.
//!
//! To recover smoothing draws: pick a particle `j` at the final step
//! with probability proportional to its final log-weight, walk its
//! ancestor chain back to the first step, and collect the state at
//! each point. Doing this N times yields N independent samples from
//! the smoothing distribution (all equally weighted).
//!
//! This module is the primitive both `sim::inference::particle_filter`
//! and (future) `sim::inference::pgas` can share. See
//! `docs/dev/proposals/2026-04-19-pf-latent-trajectories.md`.

use crate::rng::StatefulRng;
use super::types::{normalize_log_weights, NoiseAcc};

/// The transitions whose per-interval flows and rate noise an ancestry
/// records beside the states, as indices into the model's transitions in
/// ascending order. The default records neither.
///
/// The selection exists because each recorded transition costs one value per
/// particle per observation stop, the same order of memory as a compartment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathRecording {
    /// Transitions whose count over each interval is recorded.
    pub flows: Vec<usize>,
    /// Overdispersed transitions whose rate multipliers over each interval
    /// are recorded ([`NoiseAcc`]).
    pub noise: Vec<usize>,
}

/// Per-step ancestry + states recorded during a bootstrap-filter run.
///
/// `states[t][i]` is the **pre-resample** state of particle `i` at
/// observation step `t` (the state on which `log_weights[t][i]` is
/// computed).
///
/// `ancestors[t][i]` is the resampling index used for particle `i`
/// at the END of step `t` — i.e. the parent-in-step-`t` of the
/// particle that propagates into position `i` at step `t+1`. The
/// last entry (`ancestors[T-1]`) is unused for path tracing and
/// MAY be omitted by the recorder.
#[derive(Debug, Clone)]
pub struct AncestorTrace {
    /// Compartment names (for schema clarity at write time).
    pub n_compartments: usize,
    /// `states[t][i][k]` = count in compartment `k` of pre-resample
    /// particle `i` at observation step `t`.
    pub states: Vec<Vec<Vec<f64>>>,
    /// `log_weights[t][i]` = filtering log-weight of pre-resample
    /// particle `i` at observation step `t`.
    pub log_weights: Vec<Vec<f64>>,
    /// `ancestors[t][i]` = parent index in step `t`'s pre-resample
    /// swarm of the particle that propagates into position `i` at
    /// step `t+1`. Length `states.len() - 1` (no ancestors needed
    /// after the final step).
    pub ancestors: Vec<Vec<usize>>,
    /// Observation times, one per step (matches `states.len()`).
    pub obs_times: Vec<f64>,
    /// gh#48: per-step per-particle per-stream model-predicted
    /// observation. `projections[t][i][s] = E[y_s | state_{t,i}, params]`,
    /// computed via `obs_model.mean(...)` at recording time. Length
    /// matches `states`. Empty when the obs model doesn't implement
    /// `mean()` (returns `vec![]`) or when `record_ancestry == false`.
    /// Carries the model's declared projection through to
    /// `--save-paths` output so downstream tooling doesn't have to
    /// reconstruct incidence by finite-differencing compartment
    /// counts (which is unsafe under event + balance interactions).
    pub projections: Vec<Vec<Vec<f64>>>,
    /// Stream names for output schema. Length matches `obs_model.
    /// n_streams()`. Read from `obs_model.stream_names()` at record
    /// time; passed through to `SampledPath` for TSV column headers.
    pub stream_names: Vec<String>,
    /// The run's start, where the first step's interval opens. Step `t`'s
    /// interval is `[obs_times[t-1], obs_times[t])`, and step 0's is
    /// `[t_start, obs_times[0])`.
    pub t_start: f64,
    /// `flows[t][i][k]` = the number of firings of transition
    /// `recording.flows[k]` over step `t`'s interval, on the particle that
    /// ends it as pre-resample particle `i`. Empty when nothing is recorded.
    pub flows: Vec<Vec<Vec<u64>>>,
    /// `noise[t][i][k]` = the rate multipliers transition `recording.noise[k]`
    /// drew over step `t`'s interval, on the same particle. Empty when
    /// nothing is recorded.
    pub noise: Vec<Vec<Vec<NoiseAcc>>>,
    /// The transitions `flows` and `noise` hold.
    pub recording: PathRecording,
}

/// One ancestor-traced sample path from the smoothing distribution.
/// `states[t][k]` = count in compartment `k` at observation step
/// `t`. Length `trace.states.len()`.
#[derive(Debug, Clone)]
pub struct SampledPath {
    pub states: Vec<Vec<f64>>,
    pub obs_times: Vec<f64>,
    /// gh#48: per-step per-stream model-predicted observation, walked
    /// along the ancestor chain in lockstep with `states`. Length
    /// matches `states`; inner length matches the trace's
    /// `stream_names`. Empty when the source trace's `projections`
    /// is empty.
    pub projections: Vec<Vec<f64>>,
    /// Stream names corresponding to the inner index of `projections`.
    /// Cloned from the source trace.
    pub stream_names: Vec<String>,
    /// Where the first step's interval opens ([`AncestorTrace::t_start`]).
    pub t_start: f64,
    /// `flows[t][k]`: the trace's flows walked along this path's ancestor
    /// chain, so they are the firings that carried `states[t-1]` to
    /// `states[t]`. Empty when the trace recorded none.
    pub flows: Vec<Vec<u64>>,
    /// `noise[t][k]`: the trace's rate multipliers walked along the same
    /// chain. Empty when the trace recorded none.
    pub noise: Vec<Vec<NoiseAcc>>,
}

/// Sample `n` trajectory paths from the smoothing distribution
/// `p(x_{1:T} | y_{1:T}, θ)` via ancestor tracing on a bootstrap-
/// filter trace.
///
/// Algorithm: for each output path, sample a final-step particle
/// with probability ∝ `exp(log_weights[T-1])`, then walk backwards
/// collecting each step's pre-resample state. Each returned path is
/// an independent, equally-weighted draw from the smoothing
/// distribution.
pub fn sample_paths(
    trace: &AncestorTrace,
    n_paths: usize,
    seed: u64,
) -> Vec<SampledPath> {
    let n_obs = trace.states.len();
    if n_obs == 0 || n_paths == 0 {
        return Vec::new();
    }

    let mut rng = StatefulRng::new(seed.wrapping_add(0xa5ce57ea));
    let final_log_w = &trace.log_weights[n_obs - 1];

    // Normalise final-step weights into a probability vector. The helper
    // returns a uniform fallback when every weight is `-∞` or NaN, so the
    // categorical draw below works in the degenerate case without a
    // separate guard.
    let weights = normalize_log_weights(final_log_w);

    // Projections are recorded only when obs_model implements `mean()`;
    // a degenerate empty trace (n_obs == 0) is already short-circuited
    // above. We walk projections in lockstep with states; if the trace
    // didn't capture any (i.e. `projections.is_empty()`), per-path
    // projections stay empty and the TSV writer omits stream columns.
    let has_projections = trace.projections.len() == n_obs
        && trace.projections.iter().all(|step| step.len() == trace.states[0].len());

    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        // Pick the final particle by inverse-CDF sampling.
        let target = rng.uniform();
        let mut acc = 0.0;
        let mut j_final = weights.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if acc >= target { j_final = i; break; }
        }

        // Walk back. states[t] is pre-resample; ancestors[t] points
        // into states[t]'s indices and IS the parent of position i
        // at step t+1.
        // `lineage[t]` is the step-`t` particle this path reads every
        // per-step record from.
        let mut lineage = vec![j_final; n_obs];
        for t in (0..n_obs - 1).rev() {
            lineage[t] = trace.ancestors[t][lineage[t + 1]];
        }
        let states = (0..n_obs).map(|t| trace.states[t][lineage[t]].clone()).collect();
        let projections = if has_projections {
            (0..n_obs).map(|t| trace.projections[t][lineage[t]].clone()).collect()
        } else {
            Vec::new()
        };
        paths.push(SampledPath {
            states,
            obs_times: trace.obs_times.clone(),
            projections,
            stream_names: trace.stream_names.clone(),
            t_start: trace.t_start,
            flows: walk_lineage(&trace.flows, &lineage),
            noise: walk_lineage(&trace.noise, &lineage),
        });
    }
    paths
}

/// One per-step, per-particle record read along a lineage: `per_step[t]
/// [lineage[t]]` for each step, or empty when the trace recorded none.
fn walk_lineage<T: Clone>(per_step: &[Vec<Vec<T>>], lineage: &[usize]) -> Vec<Vec<T>> {
    if per_step.len() != lineage.len() { return Vec::new(); }
    per_step.iter().zip(lineage).map(|(step, &j)| step[j].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace(
        n_obs: usize,
        n_particles: usize,
        n_comps: usize,
    ) -> AncestorTrace {
        // Synthetic trace: particle i at step t has count[0] = t*100 + i
        // so we can recognise a path by its value sequence.
        let states: Vec<Vec<Vec<f64>>> = (0..n_obs)
            .map(|t| (0..n_particles)
                .map(|i| {
                    let mut s = vec![0.0; n_comps];
                    s[0] = (t * 100 + i) as f64;
                    s
                })
                .collect())
            .collect();
        let log_weights = vec![vec![0.0; n_particles]; n_obs];
        // All particles at step t+1 descend from particle 0 at step t
        // — a "single surviving lineage" ancestry. Every final-step
        // particle walks back to particle 0 at every prior step.
        let ancestors: Vec<Vec<usize>> = (0..n_obs - 1)
            .map(|_| vec![0; n_particles])
            .collect();
        let obs_times: Vec<f64> = (0..n_obs).map(|t| t as f64).collect();
        AncestorTrace {
            n_compartments: n_comps,
            states,
            log_weights,
            ancestors,
            obs_times,
            projections: Vec::new(),
            stream_names: Vec::new(),
            t_start: 0.0,
            flows: Vec::new(),
            noise: Vec::new(),
            recording: PathRecording::default(),
        }
    }

    #[test]
    fn single_lineage_all_paths_go_through_particle_0() {
        // Every ancestor at every step is 0, so every sampled path
        // traces through particle 0 at every t < final. Only the final
        // step is sensitive to the final-step sampling.
        let trace = make_trace(5, 10, 2);
        let paths = sample_paths(&trace, 4, 42);
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert_eq!(p.states.len(), 5, "path spans all obs steps");
            // Steps 0..4 should all have count[0] = t*100 + 0 since
            // every backward walk lands on particle 0.
            for t in 0..4 {
                assert_eq!(p.states[t][0], (t * 100) as f64,
                    "step {} of path walked off the expected lineage", t);
            }
        }
    }

    #[test]
    fn identity_ancestry_preserves_lineages() {
        // ancestors[t][i] = i — no resampling shuffle. Each final
        // particle's backward walk stays on its own lineage; so
        // path for final=j has count[0] = t*100 + j at every t.
        let n_obs = 4;
        let n_particles = 6;
        let mut trace = make_trace(n_obs, n_particles, 1);
        for t in 0..n_obs - 1 {
            trace.ancestors[t] = (0..n_particles).collect();
        }
        // Force final selection to pick a specific particle by making
        // all but one weight -inf.
        let target_final = 3;
        for i in 0..n_particles {
            trace.log_weights[n_obs - 1][i] =
                if i == target_final { 0.0 } else { f64::NEG_INFINITY };
        }
        let paths = sample_paths(&trace, 5, 99);
        for p in &paths {
            for t in 0..n_obs {
                assert_eq!(p.states[t][0], (t * 100 + target_final) as f64,
                    "identity ancestry should preserve lineage {}", target_final);
            }
        }
    }

    #[test]
    fn empty_trace_returns_empty() {
        let trace = AncestorTrace {
            n_compartments: 2,
            states: vec![],
            log_weights: vec![],
            ancestors: vec![],
            obs_times: vec![],
            projections: vec![],
            stream_names: vec![],
            t_start: 0.0,
            flows: vec![],
            noise: vec![],
            recording: PathRecording::default(),
        };
        assert!(sample_paths(&trace, 10, 0).is_empty());
    }

    #[test]
    fn zero_paths_requested_returns_empty() {
        let trace = make_trace(3, 4, 1);
        assert!(sample_paths(&trace, 0, 0).is_empty());
    }

    #[test]
    fn all_neg_inf_weights_falls_back_to_uniform() {
        // Shouldn't panic even in the degenerate case.
        let mut trace = make_trace(3, 5, 1);
        for w in &mut trace.log_weights[2] { *w = f64::NEG_INFINITY; }
        let paths = sample_paths(&trace, 10, 7);
        assert_eq!(paths.len(), 10);
    }

    // ── Projections walked along ancestor chain (gh#48) ──────────────

    #[test]
    fn projections_walk_with_states_along_ancestor_chain() {
        // Build a trace where particle 0 carries projection 100*t at
        // step t, particle 1 carries 200*t, etc. Identity ancestry
        // (every chain stays on its own particle index). After
        // sampling, each path's projections at step t should equal
        // 100*t × <chosen final particle> — i.e. the projection
        // walked the chain coherently with the state.
        let n_obs = 4;
        let n_particles = 5;
        let mut trace = make_trace(n_obs, n_particles, 2);
        // Replace ancestry with identity (i → i).
        for t in 0..n_obs - 1 {
            for i in 0..n_particles {
                trace.ancestors[t][i] = i;
            }
        }
        // Two streams; projections[t][i] = [100*t*(i+1), 200*t*(i+1)].
        trace.stream_names = vec!["a".into(), "b".into()];
        trace.projections = (0..n_obs).map(|t| {
            (0..n_particles).map(|i| {
                let coef = (i + 1) as f64;
                vec![100.0 * (t as f64) * coef, 200.0 * (t as f64) * coef]
            }).collect()
        }).collect();
        // Force final-step weight onto particle 3 so we know which
        // lineage the sampled paths trace.
        for i in 0..n_particles {
            trace.log_weights[n_obs - 1][i] =
                if i == 3 { 0.0 } else { f64::NEG_INFINITY };
        }
        let paths = sample_paths(&trace, 4, 7);
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert_eq!(p.stream_names, vec!["a", "b"]);
            assert_eq!(p.projections.len(), n_obs);
            for t in 0..n_obs {
                let coef = 4.0;  // (3 + 1) for particle 3
                assert_eq!(p.projections[t],
                    vec![100.0 * (t as f64) * coef, 200.0 * (t as f64) * coef],
                    "projection at t={} must walk the same lineage as states", t);
            }
        }
    }

    #[test]
    fn projections_empty_when_obs_model_doesnt_record_them() {
        // Trace with no projections (default-empty) — sample_paths
        // should still work and return paths with empty projection
        // arrays. This is the backward-compat case for obs models
        // that don't override `mean()`.
        let trace = make_trace(3, 4, 1);
        assert!(trace.projections.is_empty());
        let paths = sample_paths(&trace, 5, 0);
        assert_eq!(paths.len(), 5);
        for p in &paths {
            assert!(p.projections.is_empty());
            assert!(p.stream_names.is_empty());
        }
    }
}
