//! Multi-stream observation model implementing ObservationModel<ParticleState>.
//!
//! Constructed once from IR observation blocks + data. Stores resolved
//! likelihood expressions and evaluates them with params at call time —
//! no baked-in params. This means IF2, PGAS, and PMMH all correctly
//! respond to observation-level parameter changes (e.g., sigma_se).
//!
//! A stream's `projection` is one of:
//! - `FlowSum(flow_indices)`    — incidence projections (`Projection::CumulativeFlow`)
//! - `IntCompSum(comp_indices)` — prevalence projections (`Projection::CurrentPop` /
//!                                 `Projection::CurrentPopSum`)
//! - `Expr(resolved)`           — arbitrary state expressions
//!                                 (`Projection::DerivedExpr`)
//!
//! Incidence streams read and reset a per-stream counter; prevalence and
//! expression streams read current compartment counts and do not reset.
//! See docs/dev/proposals/2026-04-17-state-snapshot-projections.md.

use std::cell::RefCell;
use std::sync::Arc;
use crate::compiled_model::CompiledModel;
use crate::rng::StatefulRng;
use crate::propensity::EvalCtx;
use crate::resolved_expr::{ResolvedExpr, ResolvedLikelihood, eval_resolved};
use crate::state::{IntState, RealState};

// IM2 fix (2026-04-19 inference review): per-thread scratch IntState
// to eliminate per-particle, per-stream, per-observation heap
// allocation in the PF/IF2/PGAS hot path. Rayon workers each get
// their own IntState that's grown to the needed size once and reused.
thread_local! {
    static SCRATCH_INT: RefCell<IntState> = RefCell::new(IntState::from_vec(Vec::new()));
}

/// Run `f` with a mutable reference to this thread's scratch IntState,
/// resized (zero-filled) to `n`. Avoids heap allocation in the obs
/// hot path on steady-state calls.
// The zero-scratch helper `with_scratch_int` was deleted as part of
// the GH #6 fix series. It was the footgun at the centre of four
// independent bug sites — each caller had a real `counts` slice in
// hand but was using this helper to get a zero-filled IntState, then
// evaluating likelihood expressions against that empty state. The
// fix in all four sites was to swap to `with_scratch_int_from_counts`,
// which populates the scratch from the caller's real counts. Keeping
// only the populating variant makes "forget to populate" impossible
// because the API doesn't offer it. See incident 2026-04-22.

/// Run `f` with a mutable IntState whose first `n` entries mirror
/// the given `counts` slice. Same reuse pattern as `with_scratch_int`.
fn with_scratch_int_from_counts<R>(counts: &[i64], f: impl FnOnce(&IntState) -> R) -> R {
    SCRATCH_INT.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let n = counts.len();
        if borrow.counts.len() < n {
            borrow.counts.resize(n, 0);
        }
        borrow.counts[..n].copy_from_slice(counts);
        f(&borrow)
    })
}
use super::traits::ObservationModel;
use super::types::ParticleState;
use super::obs_model::{
    resolve_likelihood_from_model, eval_likelihood_resolved,
    eval_likelihood_resolved_grad,
    sample_obs_resolved, eval_obs_mean_resolved,
};

/// How a stream projects simulator state into the scalar `projected` value
/// passed to the likelihood.
#[derive(Clone)]
pub enum StreamProjection {
    /// Sum of per-transition flow counters, reset after each observation.
    /// Used for incidence data (`CumulativeFlow`).
    FlowSum(Vec<usize>),
    /// Sum of integer compartment counts read at the observation instant.
    /// Used for prevalence data (`CurrentPop`, `CurrentPopSum`). No reset.
    IntCompSum(Vec<usize>),
    /// Arbitrary expression over state, evaluated at the observation instant.
    /// Used for `DerivedExpr` (e.g. `B1 + B2`, `I/(S+I+R)`). No reset.
    Expr(ResolvedExpr),
}

impl StreamProjection {
    /// True for projections that accumulate between observations and must be
    /// reset after the likelihood is scored. Only `FlowSum` does.
    pub fn resets_after_observation(&self) -> bool {
        matches!(self, StreamProjection::FlowSum(_))
    }

    /// Build a projection from the IR projection + compiled model. Handles:
    /// `CumulativeFlow` (by flow-name family match), `CurrentPop` /
    /// `CurrentPopSum` (by local int index lookup), and `DerivedExpr` (via
    /// the shared expression resolver).
    ///
    /// Errors carry the observation stream name for a readable message
    /// (caller passes it in `obs_name`).
    pub fn from_ir(
        projection: &ir::observation::Projection,
        compiled: &CompiledModel,
        obs_name: &str,
    ) -> Result<Self, String> {
        use ir::observation::Projection as P;
        match projection {
            P::CumulativeFlow(flow_name) => {
                let idxs: Vec<usize> = compiled.model.transitions.iter().enumerate()
                    .filter(|(_, tr)| tr.name == *flow_name
                        || tr.name.starts_with(&format!("{}_", flow_name)))
                    .map(|(i, _)| i).collect();
                if idxs.is_empty() {
                    return Err(format!(
                        "observation '{}': incidence projection references flow '{}', \
                         but no transition with that name (or family `{}_*`) exists",
                        obs_name, flow_name, flow_name));
                }
                Ok(StreamProjection::FlowSum(idxs))
            }
            P::CurrentPop(comp_name) => {
                let local = resolve_int_comp(compiled, comp_name)
                    .ok_or_else(|| format!(
                        "observation '{}': prevalence projection references \
                         compartment '{}', which is not an integer compartment \
                         in this model",
                        obs_name, comp_name))?;
                Ok(StreamProjection::IntCompSum(vec![local]))
            }
            P::CurrentPopSum(names) => {
                let mut idxs = Vec::with_capacity(names.len());
                for n in names {
                    let local = resolve_int_comp(compiled, n).ok_or_else(|| format!(
                        "observation '{}': prevalence-sum projection references \
                         compartment '{}', which is not an integer compartment",
                        obs_name, n))?;
                    idxs.push(local);
                }
                Ok(StreamProjection::IntCompSum(idxs))
            }
            P::DerivedExpr(expr) => {
                use ir::table::OobPolicy;
                use crate::resolved_expr::{resolve_expr, ResolveCtx};
                let table_meta: Vec<(OobPolicy, usize)> = compiled.model.tables.iter()
                    .zip(&compiled.table_values_cache)
                    .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
                    .collect();
                let ctx = ResolveCtx {
                    comp_index: &compiled.comp_index,
                    param_index: &compiled.param_index,
                    time_func_index: &compiled.time_func_index,
                    table_index: &compiled.table_index,
                    global_to_int: &compiled.global_to_int,
                    global_to_real: &compiled.global_to_real,
                    table_meta: &table_meta,
                };
                let resolved = resolve_expr(expr, &ctx).map_err(|e| format!(
                    "observation '{}': cannot resolve state-snapshot expression: {:?}",
                    obs_name, e))?;
                Ok(StreamProjection::Expr(resolved))
            }
        }
    }
}

/// Evaluate a pre-resolved [`StreamProjection`] at a single snapshot
/// of (flows, counts, params). Shared between the in-sim scoring path
/// (`MultiStreamObsModel::project_stream_with_params`) and the CLI's
/// synthetic-obs emission path (`main.rs::project_all_obs_times`).
///
/// For `FlowSum`, `flows` holds per-transition cumulative counters
/// since the last observation (the caller is responsible for computing
/// the interval delta if the semantics demand it — scoring already
/// does so via per-stream flow accumulators, and the CLI emission path
/// uses the "delta between consecutive obs times" convention).
///
/// For `IntCompSum` and `Expr`, `counts` is the integer-compartment
/// state at the observation instant; `flows` is unread.
///
/// `t` is currently unused by any projection kind but threaded for
/// forward compatibility with time-dependent projections.
pub fn eval_stream_projection(
    projection: &StreamProjection,
    flows: &[u64],
    counts: &[i64],
    params: &[f64],
    compiled: &CompiledModel,
    real_s: &RealState,
    t: f64,
) -> f64 {
    match projection {
        StreamProjection::FlowSum(idxs) => {
            idxs.iter().map(|&i| flows[i] as f64).sum()
        }
        StreamProjection::IntCompSum(idxs) => {
            idxs.iter().map(|&i| counts[i] as f64).sum()
        }
        StreamProjection::Expr(expr) => {
            with_scratch_int_from_counts(counts, |scratch| {
                let ctx = EvalCtx {
                    model: compiled, int_s: scratch, real_s, params,
                    // dt: 0.0 — observation projection runs at obs
                    // boundaries with no integrator step in scope.
                    t, dt: 0.0, projected: None, int_float_override: None,
                };
                eval_resolved(expr, &ctx)
            })
        }
    }
}

fn resolve_int_comp(compiled: &CompiledModel, name: &str) -> Option<usize> {
    let global = *compiled.comp_index.get(name)?;
    compiled.global_to_int[global]
}

/// One observation stream.
struct Stream {
    /// IR-level observation block name. Used by `stream_names()` for
    /// output schemas (`paths.tsv` columns, posterior-predictive
    /// labels). Persisted from `StreamSpec.ir_model.name`.
    name: String,
    projection: StreamProjection,
    /// Resolved likelihood expression tree (pre-resolved at construction,
    /// but evaluates with params at call time — no baked-in values).
    resolved: ResolvedLikelihood,
    /// Observed values indexed by observation time index.
    observations: Vec<f64>,
}

/// Multi-stream observation model.
///
/// Stores resolved likelihoods and evaluates with params at call time.
/// This is the fix for the obs-level parameter bug: PGAS and PMMH now
/// correctly re-evaluate obs likelihood when sigma_se etc. change.
pub struct MultiStreamObsModel {
    streams: Vec<Stream>,
    obs_times: Vec<f64>,
    compiled: Arc<CompiledModel>,
    /// Zero real state; likelihood eval never reads real compartments and
    /// `RealState` has no interior mutability.
    real_s: RealState,
}

/// Specification for building one observation stream.
pub struct StreamSpec {
    pub projection: StreamProjection,
    pub ir_model: ir::observation::ObservationModel,
    pub observations: Vec<f64>,
    pub obs_times: Vec<f64>,
}

impl MultiStreamObsModel {
    /// Create an empty observation model (no streams, no data).
    /// Used when only the transition density is needed (e.g., gradient tests
    /// with no observation data). `log_likelihood_from_flows_and_counts`
    /// returns 0.0. For trait-generic contexts (PF, IF2) that don't need it,
    /// prefer `NullObsModel`.
    pub fn empty(compiled: Arc<CompiledModel>) -> Self {
        let real_s = RealState::new(compiled.real_local_to_global.len());
        MultiStreamObsModel {
            streams: vec![],
            obs_times: vec![],
            compiled,
            real_s,
        }
    }

    /// IM3 in 2026-04-19 inference review: previously this was infallible
    /// and panicked inside `resolve_likelihood_from_model` when any
    /// stream's likelihood expression referenced an unknown parameter /
    /// compartment / table. Now returns `Result<Self, SimError>` so the
    /// CLI can surface a diagnostic. Im3 (multi-stream obs_times
    /// mismatch) is now also checked here: all streams must share the
    /// same obs_times as stream 0; heterogeneous schedules are rejected
    /// at construction rather than silently ignored.
    pub fn new(
        stream_specs: Vec<StreamSpec>,
        compiled: Arc<CompiledModel>,
    ) -> Result<Self, crate::error::SimError> {
        if stream_specs.is_empty() {
            return Err(crate::error::SimError::Validation(
                "at least one observation stream required".to_string()
            ));
        }
        let obs_times = stream_specs[0].obs_times.clone();
        for (si, spec) in stream_specs.iter().enumerate().skip(1) {
            if spec.obs_times != obs_times {
                return Err(crate::error::SimError::Validation(format!(
                    "observation stream {} has obs_times that differ from stream 0; \
                     heterogeneous schedules are not supported yet", si
                )));
            }
        }

        let mut streams = Vec::with_capacity(stream_specs.len());
        for spec in stream_specs {
            let resolved = resolve_likelihood_from_model(
                &spec.ir_model.likelihood, &compiled,
            )?;
            streams.push(Stream {
                name: spec.ir_model.name.clone(),
                projection: spec.projection,
                resolved,
                observations: spec.observations,
            });
        }

        let real_s = RealState::new(compiled.real_local_to_global.len());

        Ok(MultiStreamObsModel {
            streams,
            obs_times,
            compiled,
            real_s,
        })
    }

    /// Evaluate a stream's projection given current particle state and
    /// params. `flows` is the per-stream flow counter slice (ignored for
    /// non-flow projections); `counts` is the integer compartment vector.
    /// `t` is the observation time, threaded so a time-dependent
    /// `StreamProjection::Expr` evaluates at the right instant.
    fn project_stream_with_params(
        &self,
        stream_idx: usize,
        flows: &[u64],
        counts: &[i64],
        params: &[f64],
        t: f64,
    ) -> f64 {
        eval_stream_projection(
            &self.streams[stream_idx].projection,
            flows, counts, params, &self.compiled, &self.real_s, t,
        )
    }

    /// Project + score from raw per-particle arrays. Used by PGAS which
    /// carries `counts` and `cum_flows` as flat Vec<i64>/Vec<u64> and has
    /// no `ParticleState`.
    pub fn log_likelihood_from_flows_and_counts(
        &self,
        cum_flows: &[u64],
        counts: &[i64],
        obs_idx: usize,
        params: &[f64],
    ) -> f64 {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let projected = self.project_stream_with_params(si, cum_flows, counts, params, t);
            let s = &self.streams[si];
            // GitHub #6 fix: the likelihood's p/mean/sd expressions can
            // reference compartment state (e.g. `p = projected / N`
            // with `N = S + I + R`). Evaluate against actual counts,
            // not a zero scratch — the zero scratch silently turned
            // PopSum-valued denominators into 0 → NaN, which the
            // binomial sampler clamped to low values, producing
            // surveys wildly inconsistent with true prevalence.
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved(
                    &s.resolved, t, projected, s.observations[obs_idx],
                    params, &self.compiled, int_s, &self.real_s,
                )
            })
        }).sum()
    }

    /// Deprecated-shape helper kept for tests that exercise the flow-only
    /// branch. Equivalent to passing a zeroed counts slice; snapshot streams
    /// would project 0.
    #[doc(hidden)]
    pub fn log_likelihood_from_flows(
        &self, cum_flows: &[u64], obs_idx: usize, params: &[f64],
    ) -> f64 {
        let zeros = vec![0i64; self.compiled.int_local_to_global.len()];
        self.log_likelihood_from_flows_and_counts(cum_flows, &zeros, obs_idx, params)
    }

    /// Gradient of `log_likelihood_from_flows_and_counts` w.r.t. estimated
    /// parameters. Used by `pgas_grad::complete_data_loglik_grad` to wire
    /// the obs-density gradient term (gh#76). Returns a fresh `Vec<f64>` of
    /// length `estimated_to_model.len()`; sums across all streams.
    ///
    /// Mirrors `log_likelihood_from_flows_and_counts` exactly in stream
    /// iteration order and projection evaluation; only the inner per-stream
    /// step changes from "score" to "score-grad".
    pub fn log_likelihood_grad_from_flows_and_counts(
        &self,
        cum_flows: &[u64],
        counts: &[i64],
        obs_idx: usize,
        params: &[f64],
        estimated_to_model: &[usize],
    ) -> Vec<f64> {
        let d = estimated_to_model.len();
        let mut grad = vec![0.0; d];
        let t = self.obs_times[obs_idx];
        for si in 0..self.streams.len() {
            let projected = self.project_stream_with_params(si, cum_flows, counts, params, t);
            let s = &self.streams[si];
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved_grad(
                    &s.resolved, t, projected, s.observations[obs_idx],
                    params, &self.compiled, int_s, &self.real_s,
                    estimated_to_model, &mut grad,
                );
            });
        }
        grad
    }
}

impl ObservationModel<ParticleState> for MultiStreamObsModel {
    fn log_likelihood(
        &self, state: &ParticleState, obs_idx: usize, params: &[f64],
    ) -> f64 {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let projected = self.project_stream_with_params(
                si, &state.flow_accumulators, &state.counts, params, t,
            );
            let s = &self.streams[si];
            // GH #6 (third-strike fix): evaluate likelihood arg
            // expressions against the particle's actual compartment
            // state, not a zero scratch. This is the path that
            // `bootstrap_filter` calls via the trait. Previous fixes
            // patched adjacent methods (sample, mean, flow-form
            // log_likelihood) but left this one broken; as a result
            // `camdl pfilter` on state-dependent likelihoods produced
            // log-ll off by ~100× (book agent observed -15980 where
            // ~-146 was expected). See incident 2026-04-22.
            with_scratch_int_from_counts(&state.counts, |int_s| {
                eval_likelihood_resolved(
                    &s.resolved, t, projected, s.observations[obs_idx],
                    params, &self.compiled, int_s, &self.real_s,
                )
            })
        }).sum()
    }

    fn n_observations(&self) -> usize { self.obs_times.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { self.streams.len() }

    fn stream_names(&self) -> Vec<String> {
        self.streams.iter().map(|s| s.name.clone()).collect()
    }

    fn sample(
        &self, state: &ParticleState, obs_idx: usize,
        params: &[f64], rng: &mut StatefulRng,
    ) -> Vec<f64> {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let projected = self.project_stream_with_params(
                si, &state.flow_accumulators, &state.counts, params, t,
            );
            let s = &self.streams[si];
            // GitHub #6: evaluate likelihood args against actual state,
            // not zero scratch. Otherwise state-dependent denominators
            // in p/mean/sd expressions blow up.
            with_scratch_int_from_counts(&state.counts, |int_s| {
                sample_obs_resolved(
                    &s.resolved, t, projected, params,
                    &self.compiled, int_s, &self.real_s, rng,
                )
            })
        }).collect()
    }

    fn mean(
        &self, state: &ParticleState, obs_idx: usize, params: &[f64],
    ) -> Vec<f64> {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let projected = self.project_stream_with_params(
                si, &state.flow_accumulators, &state.counts, params, t,
            );
            let s = &self.streams[si];
            // GitHub #6: actual state, not zero scratch.
            with_scratch_int_from_counts(&state.counts, |int_s| {
                eval_obs_mean_resolved(
                    &s.resolved, t, projected, params,
                    &self.compiled, int_s, &self.real_s,
                )
            })
        }).collect()
    }
}

/// No-op observation model for contexts that only need transition density
/// (e.g., gradient tests with no observation data). Returns 0.0 log-likelihood,
/// empty samples/means, zero observations.
pub struct NullObsModel;

impl ObservationModel<ParticleState> for NullObsModel {
    fn log_likelihood(&self, _state: &ParticleState, _obs_idx: usize, _params: &[f64]) -> f64 {
        0.0
    }
    fn n_observations(&self) -> usize { 0 }
    fn obs_time(&self, _obs_idx: usize) -> f64 { 0.0 }
    fn n_streams(&self) -> usize { 0 }
}

