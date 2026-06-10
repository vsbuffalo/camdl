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

/// One observation cell on a stream's grid.
///
/// A stream's per-observation data is `Vec<Option<ObsCell>>`: `None` is a
/// HOLE (a grid time that is present — so incidence accumulators still reset
/// on schedule — but carries no observed value, so it contributes NO term to
/// the joint log-likelihood, i.e. the unobserved value is marginalized,
/// log-contribution 0). `Some(ObsCell::Scalar(v))` is an observed scalar
/// value `v`, scored exactly as a dense observation.
///
/// A hole is NOT an observed zero: `None` ≠ `Some(Scalar(0.0))`. The former
/// omits the likelihood factor; the latter scores the density at `y = 0`.
///
/// Only `Scalar` exists today. A `Counted { value, denom }` variant (for
/// binomial-with-known-denominator survey data) is a later phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObsCell {
    /// A single observed scalar value.
    Scalar(f64),
}

/// Dense-convenience: wrap a dense `Vec<f64>` of observed values into the
/// `Vec<Option<ObsCell>>` cell representation, with every entry observed
/// (`Some(ObsCell::Scalar(_))`) and no holes. This is the no-hole path every
/// existing call site uses; only the CLI data loader (which can read an `NA`
/// token) ever produces `None` entries.
pub fn dense_cells(values: Vec<f64>) -> Vec<Option<ObsCell>> {
    values.into_iter().map(|v| Some(ObsCell::Scalar(v))).collect()
}

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
    /// Classify as incidence ([`TemporalKind::Interval`]) or prevalence
    /// ([`TemporalKind::Instant`]). Agrees by construction with the IR
    /// [`ir::observation::Projection::temporal_kind`] this was built from:
    /// `FlowSum` ⇐ `CumulativeFlow*` (incidence); `IntCompSum`/`Expr` ⇐
    /// `CurrentPop*`/`DerivedExpr` (prevalence).
    pub fn temporal_kind(&self) -> ir::observation::TemporalKind {
        use ir::observation::TemporalKind;
        match self {
            StreamProjection::FlowSum(_) => TemporalKind::Interval,
            StreamProjection::IntCompSum(_) | StreamProjection::Expr(_) => TemporalKind::Instant,
        }
    }

    /// True for projections that accumulate between observations and must be
    /// reset after the likelihood is scored — exactly the `Interval`
    /// (incidence) kind. Only `FlowSum` does.
    pub fn resets_after_observation(&self) -> bool {
        self.temporal_kind() == ir::observation::TemporalKind::Interval
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
                let idx = compiled.model.transitions.iter()
                    .position(|tr| tr.name == *flow_name)
                    .ok_or_else(|| format!(
                        "observation '{}': incidence projection references flow '{}', \
                         but no transition with that name exists",
                        obs_name, flow_name))?;
                Ok(StreamProjection::FlowSum(vec![idx]))
            }
            P::CumulativeFlowSum(flow_names) => {
                // Un-indexed `incidence()` over a stratified transition family:
                // the OCaml compiler resolved the family to explicit per-stratum
                // transition names (§25.4). Sum their cumulative flows.
                let mut idxs = Vec::with_capacity(flow_names.len());
                for fname in flow_names {
                    let idx = compiled.model.transitions.iter()
                        .position(|tr| tr.name == *fname)
                        .ok_or_else(|| format!(
                            "observation '{}': incidence-sum projection references \
                             flow '{}', but no transition with that name exists",
                            obs_name, fname))?;
                    idxs.push(idx);
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
                    binding_index: &compiled.binding_index,
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

/// Severity of a [`Finding`] emitted by [`BoundObs::bind`]. `Error` is fatal
/// (no `BoundObs` escapes); `Warn`/`Info` are advisory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warn,
    Info,
}

/// A single diagnostic produced while validating observation streams. The
/// `message` is the located, actionable text (names the stream, the offending
/// times, and the fix) — same quality bar as the `SimError::Validation`
/// messages these were factored out of.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub message: String,
}

/// Outcome of validating a set of observation streams. Carries the findings;
/// the verdict is *derived* from them, not stored, so it cannot drift from the
/// findings it summarizes.
#[derive(Debug, Clone)]
pub struct BindReport {
    findings: Vec<Finding>,
}

impl BindReport {
    fn new(findings: Vec<Finding>) -> Self {
        BindReport { findings }
    }

    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }

    /// Derived from the maximum finding severity: `Error` if any finding is an
    /// `Error`, else `Warn` if any is a `Warn`, else `Info`. Not a stored
    /// field — recomputed from `findings` on each call.
    pub fn verdict(&self) -> Severity {
        if self.findings.iter().any(|f| f.severity == Severity::Error) {
            Severity::Error
        } else if self.findings.iter().any(|f| f.severity == Severity::Warn) {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    /// True iff any finding is an `Error`. Equivalent to
    /// `self.verdict() == Severity::Error`, but reads at the call site.
    pub fn is_fatal(&self) -> bool {
        self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// Render the findings as a single, newline-joined diagnostic string for
    /// surfacing through a CLI error path. One line per finding, prefixed by
    /// its severity.
    pub fn render(&self) -> String {
        self.findings
            .iter()
            .map(|f| match f.severity {
                Severity::Error => format!("error: {}", f.message),
                Severity::Warn => format!("warning: {}", f.message),
                Severity::Info => format!("info: {}", f.message),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// One validated stream inside a [`BoundObs`]. Private — the only way to
/// obtain a `BoundStream` is through [`BoundObs::bind`], whose constructor
/// guarantees `values.len() == BoundObs.times.len()`.
struct BoundStream {
    ir_model: ir::observation::ObservationModel,
    projection: StreamProjection,
    /// Per-observation cells, indexed by observation-time index. `None` is a
    /// hole (no likelihood term; the grid time still resets incidence).
    /// INVARIANT (enforced by `bind`): `values.len() == BoundObs.times.len()`.
    values: Vec<Option<ObsCell>>,
}

/// A validated, model-shaped observation set. The ONLY way to obtain one is
/// [`BoundObs::bind`], whose constructor enforces every construction-time
/// invariant — so [`MultiStreamObsModel::new`] consumes it without re-checking.
///
/// Today's semantics are dense and homogeneous: all streams share ONE
/// observation axis (`times`), and each stream carries one value per time.
/// The sparse / per-stream-axis generalization is a later phase; this type is
/// deliberately dense.
pub struct BoundObs {
    /// The single shared observation axis (homogeneous across streams today).
    times: Vec<f64>,
    streams: Vec<BoundStream>,
}

impl BoundObs {
    /// Validate + construct from per-stream raw [`StreamSpec`]s. Enforces the
    /// construction-time invariants that previously lived in
    /// `MultiStreamObsModel::new`:
    ///
    /// - at least one stream,
    /// - each stream has at least one observation time,
    /// - stream 0's times are strictly increasing (gh#188),
    /// - every other stream's `obs_times` equals stream 0's (homogeneous
    ///   schedule),
    ///
    /// On any `Error` finding the result is `Err(report)` and NO `BoundObs`
    /// escapes. Otherwise it collapses the (identical) per-stream schedules to
    /// one shared `times` and returns `Ok((bound, report))`; the equal-length
    /// invariant `values.len() == times.len()` then holds by construction.
    ///
    /// Reproduces today's dense/homogeneous semantics exactly.
    pub fn bind(streams: Vec<StreamSpec>) -> Result<(BoundObs, BindReport), BindReport> {
        let mut findings: Vec<Finding> = Vec::new();

        // (1) Empty stream list.
        if streams.is_empty() {
            findings.push(Finding {
                severity: Severity::Error,
                message: "at least one observation stream required".to_string(),
            });
            return Err(BindReport::new(findings));
        }

        // The shared axis is stream 0's schedule; every other stream is pinned
        // to it below.
        let times = streams[0].obs_times.clone();

        // (2) Empty observation series — a header row but no data rows. Left
        // unchecked this panics downstream (`obs_times[0]` on a zero-length
        // vec). Reject with an actionable message.
        if times.is_empty() {
            findings.push(Finding {
                severity: Severity::Error,
                message: format!(
                    "observation stream '{}' has no observations — the data file \
                     has a header row but no data rows. A particle filter needs at \
                     least one observation; check that the --data file is non-empty \
                     and that its time/value columns parse.",
                    streams[0].ir_model.name
                ),
            });
            // No shared axis exists, so further checks are not meaningful.
            return Err(BindReport::new(findings));
        }

        // (3) gh#188: stream 0's observation times must be strictly increasing.
        // The cross-stream pin below carries this to every stream.
        if let Some(w) = times.windows(2).find(|w| w[1] <= w[0]) {
            findings.push(Finding {
                severity: Severity::Error,
                message: format!(
                    "observation stream '{}' has non-increasing observation \
                     times ({} then {}); observation times must be strictly increasing — \
                     remove duplicate rows or sort the --data file by time.",
                    streams[0].ir_model.name, w[0], w[1]
                ),
            });
        }

        // (4) Heterogeneous schedules: every other stream must share stream 0's
        // times. Collapsing to one shared axis is only valid once this holds.
        for (si, spec) in streams.iter().enumerate().skip(1) {
            if spec.obs_times != times {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream {} has obs_times that differ from stream 0; \
                         heterogeneous schedules are not supported yet",
                        si
                    ),
                });
            }
        }

        let report = BindReport::new(findings);
        if report.is_fatal() {
            return Err(report);
        }

        // All schedules validated and identical to `times`; collapse to one
        // shared axis with per-stream values. `values.len() == times.len()`
        // holds because `spec.obs_times == times` for every stream.
        let bound_streams = streams
            .into_iter()
            .map(|spec| BoundStream {
                ir_model: spec.ir_model,
                projection: spec.projection,
                values: spec.observations,
            })
            .collect();

        Ok((BoundObs { times, streams: bound_streams }, report))
    }

    /// The single shared observation axis.
    pub fn times(&self) -> &[f64] {
        &self.times
    }
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
    /// Per-observation cells indexed by observation time index. `None` is a
    /// hole: it contributes no likelihood term (the incidence reset still
    /// fires at its grid index — see `particle_filter.rs`).
    observations: Vec<Option<ObsCell>>,
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
///
/// `observations` is the per-grid-time cell vector: `None` = hole (no term,
/// reset still fires), `Some(ObsCell::Scalar(v))` = observed value `v`. Dense
/// call sites build it with [`dense_cells`] (all observed, no holes); only the
/// CLI loader emits `None` from an `NA` token.
pub struct StreamSpec {
    pub projection: StreamProjection,
    pub ir_model: ir::observation::ObservationModel,
    pub observations: Vec<Option<ObsCell>>,
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

    /// Consume a validated [`BoundObs`] and resolve each stream's likelihood
    /// against `compiled`.
    ///
    /// The construction-time invariants — non-empty stream list, non-empty and
    /// strictly-increasing observation times, homogeneous schedules — have
    /// MOVED to [`BoundObs::bind`], which is the only way to obtain a
    /// `BoundObs`. This constructor therefore re-checks none of them; it only
    /// resolves likelihood expressions (which still needs `compiled` and can
    /// still fail with `SimError` when a stream references an unknown
    /// parameter / compartment / table — IM3 in 2026-04-19 inference review).
    pub fn new(
        bound: BoundObs,
        compiled: Arc<CompiledModel>,
    ) -> Result<Self, crate::error::SimError> {
        let BoundObs { times: obs_times, streams: bound_streams } = bound;

        let mut streams = Vec::with_capacity(bound_streams.len());
        for spec in bound_streams {
            let resolved = resolve_likelihood_from_model(
                &spec.ir_model.likelihood, &compiled,
            )?;
            streams.push(Stream {
                name: spec.ir_model.name.clone(),
                projection: spec.projection,
                resolved,
                observations: spec.values,
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
            let s = &self.streams[si];
            // Hole: this stream has no observed value at `obs_idx`. Omit the
            // likelihood factor entirely (log-contribution 0) — the missing
            // value is marginalized, NOT scored as zero. The projection +
            // accumulator reset are NOT gated on value presence: the filter
            // loop resets per-obs-index regardless, so a hole still closes
            // the fixed incidence bin on schedule (pomp `accumvars`).
            let observed = match s.observations[obs_idx] {
                Some(ObsCell::Scalar(v)) => v,
                None => return 0.0,
            };
            let projected = self.project_stream_with_params(si, cum_flows, counts, params, t);
            // GitHub #6 fix: the likelihood's p/mean/sd expressions can
            // reference compartment state (e.g. `p = projected / N`
            // with `N = S + I + R`). Evaluate against actual counts,
            // not a zero scratch — the zero scratch silently turned
            // PopSum-valued denominators into 0 → NaN, which the
            // binomial sampler clamped to low values, producing
            // surveys wildly inconsistent with true prevalence.
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved(
                    &s.resolved, t, projected, observed,
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
            let s = &self.streams[si];
            // Hole: no term, so no gradient contribution (∂/∂θ of an omitted
            // factor is 0). Mirrors the scoring seam exactly.
            let observed = match s.observations[obs_idx] {
                Some(ObsCell::Scalar(v)) => v,
                None => continue,
            };
            let projected = self.project_stream_with_params(si, cum_flows, counts, params, t);
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved_grad(
                    &s.resolved, t, projected, observed,
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
        // gh#139: this trait path (PF / IF2 / PMMH-via-PF) and the flat
        // path (PGAS, via `log_likelihood_from_flows_and_counts`) were
        // two byte-identical summation loops. A change to one but not
        // the other is the GH#6 / incident-2026-04-22 class of bug
        // (state-dependent likelihoods scored against a zero scratch →
        // log-ll off by ~100×), which has bitten this file twice. Since
        // `ParticleState` is exactly `{ counts, flow_accumulators }`,
        // the trait method is just the flat method with the fields
        // unpacked — so delegate, and keep the per-stream scoring
        // (including the GH#6 actual-state handling) in ONE seam.
        self.log_likelihood_from_flows_and_counts(
            &state.flow_accumulators, &state.counts, obs_idx, params,
        )
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

#[cfg(test)]
mod bind_tests {
    //! Construction-time invariants for observation streams, factored out of
    //! `MultiStreamObsModel::new` into `BoundObs::bind`. Each fatal check
    //! returns `Err(report)` with `report.is_fatal()` and an actionable,
    //! located message; the happy path returns `Ok((bound, report))` with a
    //! non-fatal verdict. gh#188 (strictly increasing) and the empty/
    //! heterogeneous checks all live here now.
    use super::{BoundObs, dense_cells, Severity, StreamProjection, StreamSpec};
    use ir::observation::{
        Likelihood, ObservationModel as IrObservationModel, ObservationSchedule,
        PoissonLikelihood, Projection,
    };
    use ir::expr::{Expr, ProjectedExpr};

    /// Minimal IR observation block. `bind`'s validation runs before any
    /// likelihood resolution, so the likelihood here only has to be
    /// constructible — it is never resolved in these checks.
    fn ir_obs(name: &str) -> IrObservationModel {
        IrObservationModel {
            name: name.into(),
            schedule: ObservationSchedule::AtTimes(vec![]),
            projection: Projection::CumulativeFlow("inc".into()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: Expr::Projected(ProjectedExpr { projected: () }),
            }),
        }
    }

    fn spec(name: &str, obs_times: Vec<f64>, observations: Vec<f64>) -> StreamSpec {
        StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: ir_obs(name),
            observations: dense_cells(observations),
            obs_times,
        }
    }

    /// Extract the fatal `BindReport` from a `bind` call expected to fail —
    /// without requiring `BoundObs: Debug` (the `Ok` half is private-fielded).
    fn expect_fatal(
        r: Result<(super::BoundObs, super::BindReport), super::BindReport>,
        ctx: &str,
    ) -> super::BindReport {
        match r {
            Ok(_) => panic!("{ctx}"),
            Err(report) => report,
        }
    }

    #[test]
    fn empty_stream_list_is_fatal() {
        let report = expect_fatal(
            BoundObs::bind(vec![]),
            "an empty stream list must be rejected",
        );
        assert!(report.is_fatal());
        assert_eq!(report.verdict(), Severity::Error);
        assert!(report.findings().iter().any(|f|
            f.message.contains("at least one observation stream")));
    }

    #[test]
    fn empty_obs_times_is_fatal() {
        // A header row but no data rows. Left unchecked this panics downstream
        // on `obs_times[0]`.
        let report = expect_fatal(
            BoundObs::bind(vec![spec("cases", vec![], vec![])]),
            "a stream with no observations must be rejected",
        );
        assert!(report.is_fatal());
        assert!(report.findings().iter().any(|f|
            f.message.contains("no observations") && f.message.contains("cases")),
            "message must name the stream and the cause: {:?}", report.findings());
    }

    #[test]
    fn non_increasing_obs_times_is_fatal() {
        // gh#188: [3.0, 3.0] previously passed and silently dropped one
        // likelihood (build_obs_at_substep last-wins; exact grid drops the suffix).
        let dup = expect_fatal(
            BoundObs::bind(vec![spec(
                "cases", vec![1.0, 3.0, 3.0, 6.0], vec![0.0, 0.0, 0.0, 0.0],
            )]),
            "duplicate observation times must be rejected",
        );
        assert!(dup.is_fatal());
        assert!(dup.findings().iter().any(|f|
            f.message.contains("strictly increasing") && f.message.contains("cases")),
            "message must name the stream and the rule: {:?}", dup.findings());

        let oo = expect_fatal(
            BoundObs::bind(vec![spec(
                "cases", vec![1.0, 6.0, 3.0], vec![0.0, 0.0, 0.0],
            )]),
            "out-of-order observation times must be rejected",
        );
        assert!(oo.is_fatal());
    }

    #[test]
    fn heterogeneous_schedules_are_fatal() {
        let report = expect_fatal(
            BoundObs::bind(vec![
                spec("a", vec![1.0, 2.0, 3.0], vec![10.0, 11.0, 12.0]),
                spec("b", vec![1.0, 2.0, 4.0], vec![20.0, 21.0, 22.0]),
            ]),
            "a stream whose schedule differs from stream 0 must be rejected",
        );
        assert!(report.is_fatal());
        assert!(report.findings().iter().any(|f|
            f.message.contains("heterogeneous schedules") && f.message.contains("stream 1")),
            "message must name the offending stream index: {:?}", report.findings());
    }

    #[test]
    fn happy_path_multi_stream_binds_and_collapses_axis() {
        // Two streams sharing one strictly-increasing schedule: Ok, non-fatal,
        // one shared axis, per-stream values preserved with the equal-length
        // invariant holding by construction.
        let (bound, report) = BoundObs::bind(vec![
            spec("a", vec![1.0, 2.0, 3.0], vec![10.0, 11.0, 12.0]),
            spec("b", vec![1.0, 2.0, 3.0], vec![20.0, 21.0, 22.0]),
        ]).expect("a homogeneous, strictly-increasing multi-stream input must bind");

        // Negative control: the verdict is NOT fatal on valid input.
        assert!(!report.is_fatal(), "valid input must not produce a fatal report");
        assert_ne!(report.verdict(), Severity::Error);
        assert!(report.findings().is_empty(), "valid dense input has no findings");

        // One shared axis (= stream 0's), collapsed from the identical schedules.
        assert_eq!(bound.times(), &[1.0, 2.0, 3.0]);
        // Equal-length invariant holds by construction for every stream.
        for s in &bound.streams {
            assert_eq!(s.values.len(), bound.times().len());
        }
        assert_eq!(bound.streams[0].values, dense_cells(vec![10.0, 11.0, 12.0]));
        assert_eq!(bound.streams[1].values, dense_cells(vec![20.0, 21.0, 22.0]));
    }
}

#[cfg(test)]
mod temporal_kind_tests {
    //! P1.5: `StreamProjection::temporal_kind()` classifies incidence vs
    //! prevalence, and `resets_after_observation()` is exactly the `Interval`
    //! (incidence) kind — the reset decision has one source of truth.
    use super::StreamProjection;
    use crate::resolved_expr::ResolvedExpr;
    use ir::observation::TemporalKind;

    #[test]
    fn temporal_kind_and_reset_agree_for_every_variant() {
        let flow = StreamProjection::FlowSum(vec![0]); // incidence
        let comp = StreamProjection::IntCompSum(vec![0]); // prevalence
        let expr = StreamProjection::Expr(ResolvedExpr::Const(0.0)); // prevalence-family

        assert_eq!(flow.temporal_kind(), TemporalKind::Interval);
        assert_eq!(comp.temporal_kind(), TemporalKind::Instant);
        assert_eq!(expr.temporal_kind(), TemporalKind::Instant);

        // the reset predicate is exactly "is this Interval" — no second source
        for p in [&flow, &comp, &expr] {
            assert_eq!(
                p.resets_after_observation(),
                p.temporal_kind() == TemporalKind::Interval
            );
        }
        // and concretely: only incidence resets
        assert!(flow.resets_after_observation());
        assert!(!comp.resets_after_observation());
        assert!(!expr.resets_after_observation());
    }
}

#[cfg(test)]
mod hole_scoring_tests {
    //! Sparse/holes correctness at the SCORING seam
    //! (`log_likelihood_from_flows_and_counts`). A hole (`None`) omits the
    //! stream's likelihood factor entirely (log-contribution 0); an observed
    //! zero (`Some(ObsCell::Scalar(0.0))`) scores the density at `y = 0`.
    //! These are different — the core sparse-obs correctness property. The
    //! reset-survives-a-hole property is a filter-loop property and lives in
    //! `tests/sparse_holes_reset.rs`.
    use std::collections::HashMap;
    use std::sync::Arc;
    use super::{dense_cells, BoundObs, MultiStreamObsModel, ObsCell, StreamProjection, StreamSpec};
    use crate::compiled_model::CompiledModel;
    use crate::inference::ParticleState;
    use crate::inference::traits::ObservationModel;
    use ir::{
        expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, ProjectedExpr},
        model::{
            Compartment, CompartmentKind, InitialConditions, OutputConfig,
            OutputSchedule, SimulationConfig,
        },
        observation::{
            Likelihood, ObservationModel as IrObs, ObservationSchedule,
            PoissonLikelihood, Projection,
        },
        parameter::{ParamValue, Parameter},
        transition::{DrawMethod, StoichiometryEntry, Transition},
        Model,
    };

    /// SIR with an incidence stream `cases = incidence(recovery)` and a
    /// Poisson likelihood `rate = rho * projected`. The projected value is
    /// the cumulative `recovery` flow over the interval; the likelihood is a
    /// pure function of (projected, observed), so scoring at a fixed
    /// (flows, counts) lets us isolate the hole-vs-observed-zero behaviour.
    fn model() -> Arc<CompiledModel> {
        let m = Model {
            name: "hole_scoring".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![
                Transition {
                    name: "recovery".into(),
                    stoichiometry: vec![
                        StoichiometryEntry("I".into(), -1),
                        StoichiometryEntry("R".into(), 1),
                    ],
                    rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                        right: Box::new(Expr::Pop(ir::expr::PopExpr { pop: "I".into() })),
                    }}),
                    metadata: None,
                    draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
                },
            ],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![
                IrObs {
                    name: "cases".into(),
                    schedule: ObservationSchedule::AtTimes(vec![]),
                    projection: Projection::CumulativeFlow("recovery".into()),
                    likelihood: Likelihood::Poisson(PoissonLikelihood {
                        // rate = rho * projected
                        rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                            op: BinOp::Mul,
                            left: Box::new(Expr::Param(ParamExpr { param: "rho".into() })),
                            right: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
                        }}),
                    }),
                },
            ],
            bindings: vec![],
            parameters: vec![
                Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None },
                Parameter { name: "rho".into(), value: ParamValue::Fixed { value: 0.5 }, param_kind: None, param_dim: None },
            ],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("S".into(), 950.0); h.insert("I".into(), 40.0); h.insert("R".into(), 10.0); h
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 30.0]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 30.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(42),
            },
            presets: vec![],
            model_structure: None, balance: None, identity_tracked_compartments: vec![],
        };
        Arc::new(CompiledModel::new(m).unwrap())
    }

    /// Build a single-stream incidence obs model from explicit cells.
    fn obs_model(cells: Vec<Option<ObsCell>>, obs_times: Vec<f64>) -> MultiStreamObsModel {
        let compiled = model();
        let rec = compiled.model.transitions.iter()
            .position(|t| t.name == "recovery").unwrap();
        let spec = StreamSpec {
            projection: StreamProjection::FlowSum(vec![rec]),
            ir_model: compiled.model.observations[0].clone(),
            observations: cells,
            obs_times,
        };
        MultiStreamObsModel::new(
            BoundObs::bind(vec![spec]).expect("bind").0, compiled).unwrap()
    }

    /// (a) A `None` cell contributes EXACTLY 0 to the joint log-likelihood —
    /// identical to having no observation at that index — and is DIFFERENT
    /// from `Some(Scalar(0.0))`, which scores the Poisson density at y = 0.
    /// This is the core correctness test: hole ≠ observed-zero.
    #[test]
    fn hole_contributes_zero_and_differs_from_observed_zero() {
        let times = vec![7.0, 14.0, 21.0];
        // A non-trivial flow so `projected = rho * 100 = 50` and the Poisson
        // log-pmf at y = 0 is clearly nonzero (≈ -50). counts unused by this
        // likelihood but must be a valid slice.
        let flows = vec![100u64];
        let counts = vec![900i64, 40, 60];
        let params = model().default_params.clone();

        // Hole at obs_idx 1.
        let holed = obs_model(
            vec![Some(ObsCell::Scalar(30.0)), None, Some(ObsCell::Scalar(20.0))],
            times.clone());
        // Observed zero at obs_idx 1 (same elsewhere).
        let zeroed = obs_model(
            vec![Some(ObsCell::Scalar(30.0)), Some(ObsCell::Scalar(0.0)), Some(ObsCell::Scalar(20.0))],
            times.clone());

        let ll_hole = holed.log_likelihood_from_flows_and_counts(&flows, &counts, 1, &params);
        let ll_zero = zeroed.log_likelihood_from_flows_and_counts(&flows, &counts, 1, &params);

        // The hole contributes EXACTLY 0.0 (omitted factor).
        assert_eq!(ll_hole, 0.0,
            "a hole must contribute exactly 0 to the joint log-likelihood, got {ll_hole}");
        // The observed-zero scores the Poisson density at y=0 with mean 50:
        // log P(0; 50) = -50. Far from zero — proves hole ≠ observed-zero.
        assert!(ll_zero.is_finite() && ll_zero < -10.0,
            "observed-zero must score the density (≈ -50 here), got {ll_zero}");
        assert_ne!(ll_hole, ll_zero,
            "hole (omit term) must DIFFER from observed-zero (score y=0): \
             hole={ll_hole} zero={ll_zero}");
    }

    /// (a, cont.) A hole at one index leaves every OTHER index scored exactly
    /// as in the all-dense series — the hole is local, not a global skip.
    #[test]
    fn non_hole_indices_are_unaffected_by_a_hole_elsewhere() {
        let times = vec![7.0, 14.0, 21.0];
        let flows = vec![100u64];
        let counts = vec![900i64, 40, 60];
        let params = model().default_params.clone();

        let holed = obs_model(
            vec![Some(ObsCell::Scalar(30.0)), None, Some(ObsCell::Scalar(20.0))],
            times.clone());
        let dense = obs_model(
            dense_cells(vec![30.0, 99.0, 20.0]), // index-1 value irrelevant to 0 and 2
            times.clone());

        for idx in [0usize, 2] {
            let h = holed.log_likelihood_from_flows_and_counts(&flows, &counts, idx, &params);
            let d = dense.log_likelihood_from_flows_and_counts(&flows, &counts, idx, &params);
            assert_eq!(h, d,
                "obs_idx {idx} must score identically whether or not index 1 is a hole: \
                 holed={h} dense={d}");
        }
    }

    /// (c) The dense (all-`Some`) path is byte-identical to passing the same
    /// values through `dense_cells` — i.e. wrapping a `Vec<f64>` introduces no
    /// behavioural change for the no-hole case. (`dense_cells` is exactly what
    /// every existing call site now uses; this pins that the wrap is a no-op
    /// for scoring.)
    #[test]
    fn dense_cells_scoring_is_unchanged() {
        let times = vec![7.0, 14.0, 21.0];
        let flows = vec![100u64];
        let counts = vec![900i64, 40, 60];
        let params = model().default_params.clone();
        let values = vec![30.0, 45.0, 20.0];

        let m = obs_model(dense_cells(values.clone()), times.clone());

        // Also build via the trait path (ParticleState) to confirm both seams
        // agree on dense cells.
        let state = ParticleState { counts: counts.clone(), flow_accumulators: flows.clone() };
        for idx in 0..3 {
            let flat = m.log_likelihood_from_flows_and_counts(&flows, &counts, idx, &params);
            let via_state = m.log_likelihood(&state, idx, &params);
            assert!(flat.is_finite(), "dense scoring must be finite at idx {idx}, got {flat}");
            assert_eq!(flat, via_state,
                "flat and trait paths must agree on dense cells at idx {idx}: \
                 flat={flat} state={via_state}");
        }
    }
}

