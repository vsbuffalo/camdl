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
use super::obs_attempt::{NegInfCause, ObsCellState, StreamAttempt};
use super::obs_model::{
    explain_likelihood_neg_inf,
    resolve_likelihood_from_model, eval_likelihood_resolved,
    eval_likelihood_resolved_grad, dlogp_dprojected,
    sample_obs_resolved, eval_obs_mean_resolved,
};

/// Collect every aux column name referenced (by `Expr::ObsColumnRef`) anywhere
/// in a likelihood's argument expressions. These are the per-observation aux
/// columns the stream must bind present-together-or-hole (§3, §6.1).
fn aux_refs_in_likelihood(lik: &ir::observation::Likelihood) -> Vec<String> {
    use ir::observation::Likelihood as L;
    let mut out: Vec<String> = Vec::new();
    // The argument *expressions* (a `Diffable` arg contributes its `.expr`; `n`
    // is a bare `Expr` and can itself be an aux data-column reference, so it is
    // included).
    let args: Vec<&ir::expr::Expr> = match lik {
        L::Poisson(p) => vec![&p.rate.expr],
        L::NegBinomial(nb) => vec![&nb.mean.expr, &nb.dispersion.expr],
        L::Normal(n) => vec![&n.mean.expr, &n.sd.expr],
        L::Binomial(b) => vec![&b.n, &b.p.expr],
        L::BetaBinomial(bb) => vec![&bb.n, &bb.alpha.expr, &bb.beta.expr],
        L::Beta(b) => vec![&b.mean.expr, &b.concentration.expr],
        L::Bernoulli(b) => vec![&b.p.expr],
        L::ZeroInflatedNegBinomial(zi) => vec![&zi.mean.expr, &zi.dispersion.expr, &zi.pi.expr],
    };
    for e in args {
        collect_obs_column_refs(e, &mut out);
    }
    out
}

fn collect_obs_column_refs(e: &ir::expr::Expr, out: &mut Vec<String>) {
    use ir::expr::Expr;
    match e {
        Expr::ObsColumnRef(w) => {
            if !out.iter().any(|n| n == &w.obs_column_ref) {
                out.push(w.obs_column_ref.clone());
            }
        }
        Expr::BinOp(w) => {
            collect_obs_column_refs(&w.bin_op.left, out);
            collect_obs_column_refs(&w.bin_op.right, out);
        }
        Expr::UnOp(w) => collect_obs_column_refs(&w.un_op.arg, out),
        Expr::Cond(w) => {
            collect_obs_column_refs(&w.cond.pred, out);
            collect_obs_column_refs(&w.cond.then, out);
            collect_obs_column_refs(&w.cond.else_, out);
        }
        Expr::TableLookup(w) => {
            for ix in &w.table_lookup.indices {
                collect_obs_column_refs(ix, out);
            }
        }
        Expr::UncheckedDim(w) => collect_obs_column_refs(&w.unchecked_dim.inner, out),
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect_obs_column_refs(t, out);
            }
        }
        Expr::Const(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_)
        | Expr::Time(_) | Expr::Dt(_) | Expr::TimeFunc(_) | Expr::Projected(_)
        // gh#616: a forcing knot, never an aux data column.
        | Expr::ObsAnchor(_)
        | Expr::BindingRef(_) | Expr::PerEvalRef(_) => {}
    }
}

/// If the likelihood's trial denominator `n` is a single `ObsColumnRef`, return
/// that aux column's name (for the `value ≤ n` / `n > 0` per-row data check,
/// §3.2). Only the binomial / beta-binomial families have a capping `n`;
/// everything else returns `None` (no cap to check at bind).
fn denominator_aux_name(lik: &ir::observation::Likelihood) -> Option<&str> {
    use ir::observation::Likelihood as L;
    let n_expr = match lik {
        L::Binomial(b) => &b.n,
        L::BetaBinomial(bb) => &bb.n,
        _ => return None,
    };
    match n_expr {
        ir::expr::Expr::ObsColumnRef(w) => Some(w.obs_column_ref.as_str()),
        _ => None,
    }
}

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
/// Only `Scalar` exists today. The cell-variant axis is the scored OUTCOME
/// shape — a `Vector` variant (multinomial / compositional outcomes) is the
/// planned extension. A per-observation denominator/offset is NOT a cell
/// variant: it is a declared aux column bound from `columns { }` and read by
/// the likelihood by name (observation data-entry proposal, Stage 2).
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
                    per_eval_index: &compiled.per_eval_index,
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
/// (`MultiStreamObsModel::project_stream_from_acc`) and the CLI's
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
                    t, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None,
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

/// Resolve a `DerivedExpr` projection's `∂proj/∂compartment` (IR [`CompGradMap`],
/// compartment-name → derivative) into the int-local-indexed, eval-ready form the
/// ODE observation gradient's factor-2 chain consumes (gh#275 §1h): a list of
/// `(int_local_comp, ∂proj/∂comp)` — the compartment index matches `state_sens`.
/// A nonsmooth (`DerivEntry::Unsupported`) entry is dropped: the gradient gate
/// refuses such a model before scoring, so the dropped term is never reached.
/// Empty in ⇒ empty out (a linear projection carries no state gradient).
fn resolve_projection_state_grad(
    grad: &ir::deriv::CompGradMap,
    compiled: &CompiledModel,
) -> Result<Vec<(usize, ResolvedExpr)>, crate::error::SimError> {
    use crate::error::SimError;
    use crate::resolved_expr::{resolve_expr, ResolveCtx};
    if grad.0.is_empty() {
        return Ok(Vec::new());
    }
    let table_meta: Vec<(ir::table::OobPolicy, usize)> = compiled.model.tables.iter()
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
        per_eval_index: &compiled.per_eval_index,
    };
    let mut out = Vec::with_capacity(grad.0.len());
    for (comp_name, entry) in grad.0.iter() {
        match entry {
            ir::deriv::DerivEntry::Grad(expr) => {
                // A ∂proj/∂(real compartment) entry is dropped, not an error: real
                // compartments carry no int-local forward sensitivity, and the
                // gradient gate refuses any REAL_COMPARTMENTS model for ODE-NUTS,
                // so this entry is never consumed on the gradient path. Erroring
                // here would break the shared obs-model constructor for the
                // gradient-FREE methods (`mh`/`pgas`/…), which never read it.
                let Some(local) = resolve_int_comp(compiled, comp_name) else {
                    continue;
                };
                let resolved = resolve_expr(expr, &ctx).map_err(|e| SimError::Validation(
                    format!("cannot resolve ∂projection/∂{comp_name}: {e:?}")))?;
                out.push((local, resolved));
            }
            // Nonsmooth projection gradient — refused by the capability gate; skip.
            ir::deriv::DerivEntry::Unsupported { .. } => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod projection_state_grad_tests {
    use super::resolve_projection_state_grad;
    use crate::compiled_model::CompiledModel;
    use ir::deriv::{CompGradMap, DerivEntry};
    use ir::expr::{ConstExpr, Expr};
    use ir::model::CompartmentKind;
    use std::collections::HashMap;

    /// A `∂proj/∂(real compartment)` entry is DROPPED, not an error (gh#275 §1h
    /// review): real compartments carry no int-local forward sensitivity, and the
    /// gradient gate refuses any REAL_COMPARTMENTS model for ODE-NUTS, so the entry
    /// is never scored. Erroring here would break the shared obs-model constructor
    /// for the gradient-FREE methods (`mh`/`pgas`/…), which never read it.
    /// `sir_reservoir` has a real reservoir compartment.
    #[test]
    fn resolve_drops_real_compartment_entries_without_error() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/sir_reservoir.ir.json");
        let mut model: ir::Model =
            ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // Fill any unresolved parameter so CompiledModel::new resolves (the value
        // is irrelevant — this test resolves only constant gradient expressions).
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        let compiled = CompiledModel::new(model).unwrap();

        // Find an int compartment and the real (reservoir) compartment by kind, so
        // the test doesn't hard-code names.
        let int_comp = compiled.model.compartments.iter()
            .find(|c| matches!(c.kind, CompartmentKind::Integer)).unwrap().name.clone();
        let real_comp = compiled.model.compartments.iter()
            .find(|c| matches!(c.kind, CompartmentKind::Real)).unwrap().name.clone();

        // A mixed projection gradient touching both.
        let mut inner: HashMap<String, DerivEntry> = HashMap::new();
        inner.insert(int_comp.clone(), DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 })));
        inner.insert(real_comp, DerivEntry::Grad(Expr::Const(ConstExpr { value: 2.0 })));

        let resolved = resolve_projection_state_grad(&CompGradMap(inner), &compiled)
            .expect("a real-compartment gradient entry must be dropped, not an error");
        // Only the int compartment survives; the real one is skipped.
        assert_eq!(resolved.len(), 1, "the real compartment entry must be dropped");
        let expected = compiled.global_to_int[compiled.comp_index[int_comp.as_str()]].unwrap();
        assert_eq!(resolved[0].0, expected, "the surviving entry is the int compartment");
    }
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
/// obtain a `BoundStream` is through [`BoundObs::bind`].
///
/// Multi-cadence (Phase 1): each stream keeps **its own** schedule
/// authoritative (`obs_times`) and derives membership against the shared
/// **union** axis (`at_union`). `cells`/`aux` are per-stream-local, indexed by
/// this stream's own observation index — NOT the union index.
struct BoundStream {
    ir_model: ir::observation::ObservationModel,
    projection: StreamProjection,
    /// Per-observation cells, indexed by THIS stream's own observation index
    /// (`cells.len() == obs_times.len()`). `None` is a **hole** — a scheduled
    /// observation whose value is missing (no likelihood term; the incidence
    /// bin still resets, because the stream IS scheduled here). Distinct from
    /// "not scheduled" (a sibling's union-time), which is `at_union == None`.
    cells: Vec<Option<ObsCell>>,
    /// Membership of this stream in the union axis: for each union-index,
    /// `Some(local)` if this stream is scheduled at that time (`local` indexes
    /// `cells`/`obs_times`/`aux`), `None` if not scheduled (a sibling's time —
    /// no term, and in Phase 2 no reset). INVARIANT (`bind`):
    /// `at_union.len() == BoundObs.times.len()`.
    at_union: Vec<Option<usize>>,
    /// Per-observation auxiliary data (a binomial denominator `n = tested`, a
    /// person-time offset), indexed by THIS stream's own observation index,
    /// each a name→value list the likelihood reads by `Expr::ObsColumnRef`
    /// (§3, §6.1). An entry is empty when the stream references no aux column
    /// OR the cell is a hole. INVARIANT (`bind`): `aux.len() == cells.len()`,
    /// and at a non-hole cell every referenced aux column is present
    /// (present-together-or-hole).
    aux: Vec<Vec<(String, f64)>>,
}

/// A validated, model-shaped observation set. The ONLY way to obtain one is
/// [`BoundObs::bind`], whose constructor enforces every construction-time
/// invariant — so [`MultiStreamObsModel::new`] consumes it without re-checking.
///
/// `times` is the **union** of every stream's schedule (sorted-unique). Each
/// stream carries its own schedule + a `at_union` membership map; a homogeneous
/// model is the special case where every stream is scheduled at every union
/// time. Holes (scheduled-but-missing) live in each stream's `cells`;
/// not-scheduled (a sibling's cadence) is `at_union == None`.
pub struct BoundObs {
    /// The union observation axis: `sorted_unique(⋃ streams.obs_times)`.
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
    pub fn bind(mut streams: Vec<StreamSpec>) -> Result<(BoundObs, BindReport), BindReport> {
        let mut findings: Vec<Finding> = Vec::new();

        // (1) Empty stream list.
        if streams.is_empty() {
            findings.push(Finding {
                severity: Severity::Error,
                message: "at least one observation stream required".to_string(),
            });
            return Err(BindReport::new(findings));
        }

        // (2,3) Per-stream schedule validity. Multi-cadence: each stream keeps
        // its OWN schedule (no longer pinned to stream 0), so the non-empty and
        // strictly-increasing (gh#188) checks run per stream. Findings
        // accumulate; the union is built only from validated schedules below.
        for spec in &streams {
            // (2) Empty observation series — a header row but no data rows.
            // Left unchecked this panics downstream (`obs_times[0]` on a
            // zero-length vec).
            if spec.obs_times.is_empty() {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream '{}' has no observations — the data file \
                         has a header row but no data rows. A particle filter needs at \
                         least one observation; check that the --data file is non-empty \
                         and that its time/value columns parse.",
                        spec.ir_model.name
                    ),
                });
            }
            // Non-finite times (NaN / ±inf) must be rejected here: the union
            // merge sorts the times, and `NaN.partial_cmp` is `None`, which
            // would panic the sort. (The CLI time-column parser also guards
            // this at the source with a located message; this is the
            // defense-in-depth so `bind` never panics on garbage input.)
            if let Some(bad) = spec.obs_times.iter().find(|t| !t.is_finite()) {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream '{}' has a non-finite observation time \
                         ({}); times must be finite — NaN and infinities are not valid \
                         observation times. Fix or remove the offending data row.",
                        spec.ir_model.name, bad
                    ),
                });
            }
            // (3) gh#188: each stream's own observation times must be strictly
            // increasing.
            if let Some(w) = spec.obs_times.windows(2).find(|w| w[1] <= w[0]) {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream '{}' has non-increasing observation \
                         times ({} then {}); observation times must be strictly increasing — \
                         remove duplicate rows or sort the --data file by time.",
                        spec.ir_model.name, w[0], w[1]
                    ),
                });
            }
            // cells are per-stream-local, indexed by this stream's own
            // schedule: `cells.len() == obs_times.len()` (the at_union walk
            // indexes `cells[local]` for each scheduled union time).
            if spec.observations.len() != spec.obs_times.len() {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream '{}': {} cells but {} observation times \
                         (internal binder error — cells must match the stream's own \
                         schedule length)",
                        spec.ir_model.name, spec.observations.len(), spec.obs_times.len()
                    ),
                });
            }
        }
        // (4) Heterogeneous schedules are now SUPPORTED: streams on different
        // cadences merge to the union axis below (multi-cadence Phase 1). The
        // former "must equal stream 0" rejection is gone.

        // An EMPTY aux vector means "this stream declares no aux columns" — the
        // common scalar-outcome case (and every `StreamSpec::dense` site already
        // fills it length-matched). Normalize it to per-observation empties so
        // the downstream `aux.len() == cells.len()` invariant holds. A non-empty
        // but wrong-length aux is still a hard error below (a real binder bug).
        for spec in &mut streams {
            if spec.aux.is_empty() && !spec.observations.is_empty() {
                spec.aux = vec![Vec::new(); spec.observations.len()];
            }
        }

        // (5) Per-observation auxiliary data (§3, §6.1): present-together-or-hole
        // and the binomial-`n` row check (`value ≤ n`, `n > 0`). The aux vector
        // must be length-matched to the cells, every referenced aux present at a
        // non-hole cell (else a hole), and a capping denominator well-formed.
        for spec in &streams {
            let stream_name = &spec.ir_model.name;
            let required = aux_refs_in_likelihood(&spec.ir_model.likelihood);
            let denom = denominator_aux_name(&spec.ir_model.likelihood);
            // gh#812: reported ONCE per stream with the row list, not per row.
            // A file full of zero denominators is still worth shouting about;
            // 5 of 49 is a fact about the testing programme.
            let mut zero_denom_rows: Vec<usize> = Vec::new();

            if spec.aux.len() != spec.observations.len() {
                findings.push(Finding {
                    severity: Severity::Error,
                    message: format!(
                        "observation stream '{}': aux data has {} rows but {} \
                         observation cells (internal binder error)",
                        stream_name, spec.aux.len(), spec.observations.len()
                    ),
                });
                continue;
            }

            for (i, (cell, aux_row)) in
                spec.observations.iter().zip(spec.aux.iter()).enumerate()
            {
                let observed = match cell {
                    // A hole scores no term; the missing-together rule made it a
                    // hole upstream, so no aux check applies here.
                    None => continue,
                    Some(ObsCell::Scalar(v)) => *v,
                };
                // present-together: every referenced aux column must be present
                // at a scored cell. (A missing aux should have been turned into
                // a hole by the loader; if it reaches here, it is a hard error —
                // `binomial(n = NaN)` must be unconstructible.)
                for col in &required {
                    if !aux_row.iter().any(|(k, _)| k == col) {
                        findings.push(Finding {
                            severity: Severity::Error,
                            message: format!(
                                "observation stream '{}' row {}: scored value present \
                                 but referenced aux column '{}' is missing — value and \
                                 every aux column must be present together, or the row \
                                 is a hole",
                                stream_name, i, col
                            ),
                        });
                    }
                }
                // `n >= 0` and `value <= n` for a binomial denominator (§3.2).
                //
                // gh#812: `n == 0` is NOT an error. Zero trials has exactly one
                // possible outcome, so the term is exactly 0 for every parameter
                // value -- non-identifying, not invalid -- and refusing it
                // rejects a likelihood the kernel already computes correctly
                // (`obs_loglik::beta_binomial_logpmf`). Stan admits it too
                // (`check_nonnegative` on the population size). Surveillance data
                // carries a zero denominator on any day nobody was examined, and
                // the alternative -- writing those rows NA -- destroys real
                // information wherever the same column is also bound as a count.
                //
                // The `value <= n` half of this check is the load-bearing one and
                // still applies at n = 0, where it rejects a positive count
                // against zero trials. Only the `n > 0` half is relaxed.
                if let Some(dn) = denom {
                    if let Some((_, n)) = aux_row.iter().find(|(k, _)| k == dn) {
                        if !(*n >= 0.0) {
                            findings.push(Finding {
                                severity: Severity::Error,
                                message: format!(
                                    "observation stream '{}' row {}: denominator '{}' = {} \
                                     must be >= 0",
                                    stream_name, i, dn, n
                                ),
                            });
                        } else if *n == 0.0 {
                            zero_denom_rows.push(i);
                            if observed > 0.0 {
                                findings.push(Finding {
                                    severity: Severity::Error,
                                    message: format!(
                                        "observation stream '{}' row {}: scored value {} \
                                         against denominator '{}' = 0 — no trials cannot \
                                         yield a positive count",
                                        stream_name, i, observed, dn
                                    ),
                                });
                            }
                        } else if observed > *n {
                            findings.push(Finding {
                                severity: Severity::Error,
                                message: format!(
                                    "observation stream '{}' row {}: scored value {} exceeds \
                                     denominator '{}' = {} (value ≤ n required) — check the \
                                     data file (a transposed row?)",
                                    stream_name, i, observed, dn, n
                                ),
                            });
                        }
                    }
                }
            }

            // gh#812: one warning per stream, naming the rows, so a zero-effort
            // row is not silently indistinguishable from a missing one. `NA`
            // means "we do not know what happened"; `n = 0` means "we know
            // nobody was examined". They contribute identically to the
            // likelihood and are different claims about the world.
            if !zero_denom_rows.is_empty() {
                let shown: Vec<String> =
                    zero_denom_rows.iter().take(8).map(|r| r.to_string()).collect();
                let more = if zero_denom_rows.len() > 8 {
                    format!(", +{} more", zero_denom_rows.len() - 8)
                } else {
                    String::new()
                };
                findings.push(Finding {
                    severity: Severity::Warn,
                    message: format!(
                        "observation stream '{}': {} of {} rows have denominator '{}' = 0 \
                         (rows {}{}). These score as zero — with no trials there is one \
                         possible outcome, so the term is non-identifying, not invalid. \
                         If those rows are MISSING rather than zero-effort, write them as NA.",
                        stream_name,
                        zero_denom_rows.len(),
                        spec.observations.len(),
                        denom.unwrap_or("?"),
                        shown.join(", "),
                        more,
                    ),
                });
            }
        }

        let report = BindReport::new(findings);
        if report.is_fatal() {
            return Err(report);
        }

        // Union axis: the sorted-unique merge of every stream's schedule.
        // Identity is EXACT f64 — one deterministic time parser gives one value
        // per calendar instant (§3.2), so no tolerance is needed (and none is
        // used). A homogeneous model yields a union equal to the shared axis.
        let mut times: Vec<f64> =
            streams.iter().flat_map(|s| s.obs_times.iter().copied()).collect();
        // Finiteness is enforced by the per-stream validation above (a
        // non-finite time is a fatal finding → early return before this sort),
        // so `partial_cmp` never returns `None` here.
        times.sort_by(|a, b| a.partial_cmp(b).expect("observation times are finite (checked above)"));
        times.dedup();

        // Each stream keeps its own schedule; `at_union` records, per union
        // index, this stream's own local cell index (or `None` where a sibling
        // observes but this stream does not). Built by a two-pointer walk over
        // the sorted union and the stream's own sorted (subset) schedule.
        let bound_streams = streams
            .into_iter()
            .map(|spec| {
                let mut at_union = Vec::with_capacity(times.len());
                let mut local = 0usize;
                for &ut in &times {
                    if local < spec.obs_times.len() && spec.obs_times[local] == ut {
                        at_union.push(Some(local));
                        local += 1;
                    } else {
                        at_union.push(None);
                    }
                }
                debug_assert_eq!(
                    local, spec.obs_times.len(),
                    "every stream observation time must appear in the union",
                );
                BoundStream {
                    ir_model: spec.ir_model,
                    projection: spec.projection,
                    cells: spec.observations,
                    at_union,
                    aux: spec.aux,
                }
            })
            .collect();

        Ok((BoundObs { times, streams: bound_streams }, report))
    }

    /// The union observation axis (sorted-unique over all streams' schedules).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Test-only: a stream's `at_union` membership over the union axis.
    #[cfg(test)]
    pub(crate) fn at_union_for_test(&self, stream_idx: usize) -> &[Option<usize>] {
        &self.streams[stream_idx].at_union
    }
}

/// One observation stream.
struct Stream {
    /// IR-level observation block name. Used by `stream_names()` for
    /// output schemas (`paths.tsv` columns, posterior-predictive
    /// labels). Persisted from `StreamSpec.ir_model.name`.
    name: String,
    projection: StreamProjection,
    /// ∂projection/∂compartment for a `DerivedExpr` (nonlinear) projection,
    /// resolved and keyed by int-local compartment index (matching `state_sens`).
    /// The ODE observation gradient's factor-2 chain (`∂proj/∂θ = Σ_j ∂proj/∂x_j
    /// · S[j]`, gh#275 §1h) evaluates each at the trajectory state and dots it
    /// with the forward sensitivity. Empty for linear projections (a trivial
    /// selection the factor-2 chain handles directly) and gradient-free builds.
    projection_state_grad: Vec<(usize, ResolvedExpr)>,
    /// Resolved likelihood expression tree (pre-resolved at construction,
    /// but evaluates with params at call time — no baked-in values).
    resolved: ResolvedLikelihood,
    /// Membership over the model's union observation axis: for each union
    /// index, `Some(local)` if this stream is scheduled there (`local` indexes
    /// `observations`/`aux`), `None` if not scheduled (a sibling's cadence —
    /// no term). For a homogeneous model `at_union[i] == Some(i)`.
    at_union: Vec<Option<usize>>,
    /// Per-observation cells indexed by THIS stream's own observation index
    /// (NOT the union index — resolve via `at_union`). `None` is a hole: no
    /// likelihood term (the incidence reset still fires at its grid index —
    /// see `particle_filter.rs`).
    observations: Vec<Option<ObsCell>>,
    /// Per-observation auxiliary data (binomial `n`, Poisson offset), indexed
    /// by this stream's own observation index; read by `Expr::ObsColumnRef`.
    aux: Vec<Vec<(String, f64)>>,
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
    /// One slot per `Interval` (incidence / `FlowSum`) stream, in dense order =
    /// the `ParticleState.acc` layout (multi-cadence Phase 2a). `stream_idx`
    /// points back into `streams`; `flow_indices` are the per-transition indices
    /// this stream sums (the `FlowSum(idxs)` set). Prevalence/`Instant` streams
    /// own no slot.
    interval_slots: Vec<IntervalSlot>,
    /// Per-stream → `acc` slot map (len == `streams.len()`): `Some(k)` if stream
    /// `si` is the k-th Interval slot, `None` for a prevalence/`Instant` stream.
    /// The scoring fork reads `acc[k]` for `Some(k)` and projects from `counts`
    /// for `None`.
    stream_to_slot: Vec<Option<usize>>,
}

/// One `acc` bin for an `Interval` (incidence) stream (multi-cadence Phase 2a).
/// `flow_indices` is exactly the stream's `FlowSum(idxs)` transition set; the
/// fold sums `flow_accumulators` over them once per interval into `acc[k]`.
struct IntervalSlot {
    stream_idx: usize,
    flow_indices: Vec<usize>,
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
    /// Per-observation auxiliary data (a binomial denominator `n = tested`, a
    /// person-time offset), indexed by observation-time index — each a
    /// name→value list the likelihood reads by name (§3, §6.1). The loader
    /// fills this; dense call sites pass empties via [`StreamSpec::dense`].
    /// INVARIANT (checked in `bind`): `aux.len() == observations.len()`.
    pub aux: Vec<Vec<(String, f64)>>,
}

impl StreamSpec {
    /// Build a spec with NO auxiliary data — the common scalar-outcome stream
    /// (no binomial denominator / offset). The aux vector is all-empty,
    /// length-matched to `observations`. Dense and aux-free call sites use this
    /// so they need not spell out the empty `aux`.
    pub fn dense(
        projection: StreamProjection,
        ir_model: ir::observation::ObservationModel,
        observations: Vec<Option<ObsCell>>,
        obs_times: Vec<f64>,
    ) -> Self {
        let aux = vec![Vec::new(); observations.len()];
        StreamSpec { projection, ir_model, observations, obs_times, aux }
    }
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
            interval_slots: vec![],
            stream_to_slot: vec![],
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
            let projection_state_grad =
                resolve_projection_state_grad(&spec.ir_model.projection_state_grad, &compiled)?;
            streams.push(Stream {
                name: spec.ir_model.name.clone(),
                projection: spec.projection,
                projection_state_grad,
                resolved,
                at_union: spec.at_union,
                observations: spec.cells,
                aux: spec.aux,
            });
        }

        // Precompute the per-Interval-stream `acc` layout (Phase 2a): one slot
        // per `FlowSum` (incidence) stream, in stream order = the dense `acc`
        // index. `stream_to_slot[si]` is `Some(k)` iff stream `si` is the k-th
        // Interval slot. Prevalence/`Instant` streams own no slot.
        let mut interval_slots: Vec<IntervalSlot> = Vec::new();
        let mut stream_to_slot: Vec<Option<usize>> = Vec::with_capacity(streams.len());
        for (si, s) in streams.iter().enumerate() {
            match &s.projection {
                StreamProjection::FlowSum(idxs) => {
                    stream_to_slot.push(Some(interval_slots.len()));
                    interval_slots.push(IntervalSlot {
                        stream_idx: si,
                        flow_indices: idxs.clone(),
                    });
                }
                StreamProjection::IntCompSum(_) | StreamProjection::Expr(_) => {
                    stream_to_slot.push(None);
                }
            }
        }

        let real_s = RealState::new(compiled.real_local_to_global.len());

        Ok(MultiStreamObsModel {
            streams,
            obs_times,
            compiled,
            real_s,
            interval_slots,
            stream_to_slot,
        })
    }

    /// Number of `Interval` (incidence) streams — the length of each particle's
    /// `acc` bin vector (Phase 2a).
    pub fn n_interval_streams(&self) -> usize {
        self.interval_slots.len()
    }

    /// Number of declared observation streams — the length of every per-stream
    /// vector this type produces (`log_likelihood_per_stream_from_flows_and_counts`,
    /// `per_stream_observed`, `stream_names`). Counts EVERY stream, incidence and
    /// prevalence alike, unlike [`Self::n_interval_streams`].
    pub fn n_streams(&self) -> usize {
        self.streams.len()
    }

    /// The declared stream names, in the same order as every per-stream vector
    /// (`log_likelihood_per_stream_from_flows_and_counts`, `per_stream_observed`).
    /// The single source of truth for labelling a per-stream column, so a label
    /// and the value under it cannot come from different orderings.
    pub fn stream_names(&self) -> Vec<String> {
        self.streams.iter().map(|s| s.name.clone()).collect()
    }

    /// The `Interval` (incidence / `FlowSum`) streams as `(name, flow_indices)`,
    /// in dense `acc` order. The single source of truth for an `inc_<stream>`
    /// output column: a posterior-trajectory writer sums per-transition flows
    /// over `flow_indices` (the same `FlowSum(idxs)` set scoring uses), so the
    /// emitted incidence is the model's declared projection — never a
    /// finite-difference of compartment counts, which is unsafe under
    /// event/balance interactions (gh#48 / gh#264). Prevalence/`Instant`
    /// streams own no incidence column and are absent here.
    ///
    /// Kept compatible with a future per-stream prequential (gh#269): the same
    /// per-stream `FlowSum` density underlies both an `inc_<stream>` column and
    /// a per-stream score.
    pub fn incidence_streams(&self) -> Vec<(String, Vec<usize>)> {
        self.interval_slots.iter()
            .map(|slot| (
                self.streams[slot.stream_idx].name.clone(),
                slot.flow_indices.clone(),
            ))
            .collect()
    }

    /// Fold this interval's per-transition `flow_accumulators` into the
    /// per-stream `acc`: for each Interval slot `k`,
    /// `acc[k] += Σ_{i ∈ flow_indices(k)} flow_accumulators[i]`. Once per
    /// observation interval, after substeps and BEFORE scoring. Never zeroes
    /// `flow_accumulators`.
    pub fn fold_into_acc(&self, flow_accumulators: &[u64], acc: &mut [u64]) {
        for (k, slot) in self.interval_slots.iter().enumerate() {
            let bin: u64 = slot.flow_indices.iter()
                .map(|&i| flow_accumulators[i])
                .sum();
            acc[k] += bin;
        }
    }

    /// Real-valued fold for the deterministic ODE path: identical to
    /// [`fold_into_acc`] but over `f64` flows (the ODE backend's continuous
    /// `rate·dt` accumulation, never rounded).
    pub fn fold_into_acc_real(&self, flow_accumulators: &[f64], acc: &mut [f64]) {
        self.fold_into_acc_real_blocks(flow_accumulators, acc, 1);
    }

    /// Block-strided sibling of [`fold_into_acc_real`]: `src` carries `width`
    /// consecutive values per transition (`src[i*width + c]`) and `acc` carries
    /// `width` per Interval slot (`acc[k*width + c]`).
    ///
    /// The ODE gradient path folds the forward sensitivity `∂flow/∂θ` this way —
    /// `d` gradient columns per transition, laid out transition-major, so no
    /// contiguous `&[f64]` view of one column exists and the scalar signature
    /// cannot serve. `fold_into_acc_real` IS the `width == 1` case and delegates
    /// here, so a sensitivity bin is folded by exactly the same slot map, in the
    /// same order, as the value bin it differentiates — they cannot drift.
    pub fn fold_into_acc_real_blocks(&self, src: &[f64], acc: &mut [f64], width: usize) {
        for (k, slot) in self.interval_slots.iter().enumerate() {
            for c in 0..width {
                let bin: f64 = slot.flow_indices.iter()
                    .map(|&i| src[i * width + c])
                    .sum();
                acc[k * width + c] += bin;
            }
        }
    }

    /// Zero the `acc` bins for the Interval streams scheduled at union index
    /// `union_idx` (`at_union[union_idx].is_some()`). A stream not scheduled
    /// here keeps its running bin toward its own next observation.
    pub fn reset_due_acc(&self, union_idx: usize, acc: &mut [u64]) {
        for (k, slot) in self.interval_slots.iter().enumerate() {
            let si = slot.stream_idx;
            if self.streams[si].at_union[union_idx].is_some() {
                acc[k] = 0;
            }
        }
    }

    /// Real-valued sibling of [`reset_due_acc`] for the ODE path.
    pub fn reset_due_acc_real(&self, union_idx: usize, acc: &mut [f64]) {
        self.reset_due_acc_real_blocks(union_idx, acc, 1);
    }

    /// Block-strided sibling of [`reset_due_acc_real`], zeroing the whole
    /// `width`-wide block `acc[k*width .. (k+1)*width]` of each due slot. The
    /// gradient path's `∂flow/∂θ` accumulator must close on the same schedule as
    /// the value accumulator; `reset_due_acc_real` is the `width == 1` case and
    /// delegates here so the due-predicate is evaluated in exactly one place.
    pub fn reset_due_acc_real_blocks(&self, union_idx: usize, acc: &mut [f64], width: usize) {
        for (k, slot) in self.interval_slots.iter().enumerate() {
            let si = slot.stream_idx;
            if self.streams[si].at_union[union_idx].is_some() {
                acc[k * width..(k + 1) * width].fill(0.0);
            }
        }
    }

    /// Project stream `si` for SCORING given the per-stream folded `acc` and the
    /// integer `counts` (Phase 2a fork). An Interval stream reads its already-
    /// summed bin `acc[k]` directly (no `idxs` re-sum); a prevalence/`Instant`
    /// stream projects from `counts` via `eval_stream_projection` (flows unused).
    fn project_stream_from_acc(
        &self,
        stream_idx: usize,
        acc: &[u64],
        counts: &[i64],
        params: &[f64],
        t: f64,
    ) -> f64 {
        match self.stream_to_slot[stream_idx] {
            Some(k) => acc[k] as f64,
            None => eval_stream_projection(
                &self.streams[stream_idx].projection,
                // Interval flows are never read for an `Instant` stream; pass
                // an empty slice to make that explicit.
                &[], counts, params, &self.compiled, &self.real_s, t,
            ),
        }
    }

    /// Real-acc sibling of [`project_stream_from_acc`]: the Interval bin
    /// `acc[k]` is already `f64` (the ODE path), so it is read directly with no
    /// `as f64` quantization.
    fn project_stream_from_acc_real(
        &self,
        stream_idx: usize,
        acc: &[f64],
        counts: &[i64],
        params: &[f64],
        t: f64,
    ) -> f64 {
        match self.stream_to_slot[stream_idx] {
            Some(k) => acc[k],
            None => eval_stream_projection(
                &self.streams[stream_idx].projection,
                &[], counts, params, &self.compiled, &self.real_s, t,
            ),
        }
    }

    /// The joint observed value at each union index: the sum, across all bound
    /// streams scheduled-and-observed there, of the cell value. This is the
    /// correct `y_obs` for the prequential trace and the `--trace` `observed`
    /// column — the trace records a single scalar observation per step and
    /// scores it against the joint (cross-stream sum) predictive sample
    /// (`sample().filter(finite).sum()`). A stream not scheduled at this index
    /// (`at_union == None`) or a hole (`observations[local] == None`)
    /// contributes nothing; mirrors [`score_streams`]'s union/hole handling so
    /// the recorded observation and the scored predictive share one convention.
    /// Holes never actually reach here — `--save-prequential` / `--trace` are
    /// hard-rejected when the data has any hole (`check_holes_output_compat`),
    /// so on this path scheduled ⇔ observed and the sum is exact. The returned
    /// vector has length `self.obs_times.len()`.
    pub fn joint_observed(&self) -> Vec<f64> {
        (0..self.obs_times.len())
            .map(|union_idx| {
                self.streams
                    .iter()
                    .map(|s| match s.at_union[union_idx] {
                        Some(local) => match s.observations[local] {
                            Some(ObsCell::Scalar(v)) => v,
                            None => 0.0,
                        },
                        None => 0.0,
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    /// Per-stream sibling of [`joint_observed`]: the observed value of EACH
    /// stream at EACH union index, shape `[obs_idx][stream]` (length
    /// `obs_times.len()` × `streams.len()`). gh#269: feeds the per-district
    /// prequential `y_obs` column.
    ///
    /// A stream not scheduled at this union index (`at_union == None`) or a hole
    /// (`observations[local] == None`) yields `f64::NAN` — the honest "no
    /// observation here" marker, NOT a fictitious 0. `build_trace` skips a
    /// stream whose `y_obs` is non-finite, so an absent stream contributes no
    /// per-stream row at that step. (The JOINT path still uses
    /// [`joint_observed`], whose absent/hole convention is the cross-stream sum
    /// where absent ⇒ 0.)
    pub fn per_stream_observed(&self) -> Vec<Vec<f64>> {
        (0..self.obs_times.len())
            .map(|union_idx| {
                self.streams
                    .iter()
                    .map(|s| match s.at_union[union_idx] {
                        Some(local) => match s.observations[local] {
                            Some(ObsCell::Scalar(v)) => v,
                            None => f64::NAN,
                        },
                        None => f64::NAN,
                    })
                    .collect()
            })
            .collect()
    }

    /// Shared per-stream scoring loop. `project(stream_idx, t)` supplies the
    /// projected value for each stream — the only thing that differs between the
    /// integer (PGAS/PF) and real (ODE) accumulator paths. Hole handling, union
    /// mapping, and likelihood evaluation are identical, so they live here once.
    ///
    /// Returns the PER-STREAM log-likelihood contributions, length
    /// `self.streams.len()`. A stream not scheduled at this union index
    /// (a sibling's cadence) or a hole (value missing) yields `0.0` — the
    /// omitted-factor convention. The JOINT log-likelihood is `.iter().sum()`
    /// of this vector (gh#269: per-stream prequential reads the breakdown, the
    /// joint reads the sum; ONE seam, so `joint == Σ per_stream` structurally).
    fn score_streams(
        &self,
        obs_idx: usize,
        counts: &[i64],
        params: &[f64],
        project: impl Fn(usize, f64) -> f64,
    ) -> Vec<f64> {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let s = &self.streams[si];
            // `obs_idx` indexes the UNION axis; `at_union` maps it to this
            // stream's own local cell. Not scheduled here (a sibling's cadence)
            // ⇒ omit the term entirely.
            let local = match s.at_union[obs_idx] {
                Some(l) => l,
                None => return 0.0,
            };
            // Hole: scheduled but value missing. Omit the likelihood factor
            // (log-contribution 0) — marginalized, NOT scored as zero. The
            // projection + accumulator reset are NOT gated on value presence:
            // the filter loop resets per-obs-index regardless, so a hole still
            // closes the fixed incidence bin on schedule (pomp `accumvars`).
            let observed = match s.observations[local] {
                Some(ObsCell::Scalar(v)) => v,
                None => return 0.0,
            };
            let projected = project(si, t);
            // GitHub #6 fix: the likelihood's p/mean/sd expressions can
            // reference compartment state (e.g. `p = projected / N`
            // with `N = S + I + R`). Evaluate against actual counts,
            // not a zero scratch — the zero scratch silently turned
            // PopSum-valued denominators into 0 → NaN, which the
            // binomial sampler clamped to low values, producing
            // surveys wildly inconsistent with true prevalence.
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved(
                    &s.resolved, t, projected, observed, &s.aux[local],
                    params, &self.compiled, int_s, &self.real_s,
                )
            })
        }).collect()
    }

    /// Project + score from raw per-particle arrays. Used by PGAS which
    /// carries `counts` and the per-stream `acc` as flat Vec<i64>/Vec<u64> and
    /// has no `ParticleState`.
    ///
    /// `acc` is the per-Interval-stream folded bin vector (Phase 2a), length
    /// `n_interval_streams()`, indexed by `stream_to_slot`. An Interval stream
    /// reads `acc[k]` directly; a prevalence stream projects from `counts`.
    pub fn log_likelihood_from_flows_and_counts(
        &self,
        acc: &[u64],
        counts: &[i64],
        obs_idx: usize,
        params: &[f64],
    ) -> f64 {
        self.score_streams(obs_idx, counts, params, |si, t| {
            self.project_stream_from_acc(si, acc, counts, params, t)
        }).iter().sum()
    }

    /// Per-stream sibling of [`log_likelihood_from_flows_and_counts`]: the
    /// vector of per-stream log-likelihood contributions whose sum IS the joint
    /// (gh#269). A stream not scheduled here or a hole contributes `0.0`. Used
    /// by the prequential recorder to expose per-district scores; uses no RNG.
    pub fn log_likelihood_per_stream_from_flows_and_counts(
        &self,
        acc: &[u64],
        counts: &[i64],
        obs_idx: usize,
        params: &[f64],
    ) -> Vec<f64> {
        self.score_streams(obs_idx, counts, params, |si, t| {
            self.project_stream_from_acc(si, acc, counts, params, t)
        })
    }

    /// Real-acc sibling of [`log_likelihood_from_flows_and_counts`] for the
    /// deterministic ODE path: the Interval bins are `f64` (un-rounded
    /// continuous flow), so scoring reads them directly.
    pub fn log_likelihood_from_flows_and_counts_real(
        &self,
        acc: &[f64],
        counts: &[i64],
        obs_idx: usize,
        params: &[f64],
    ) -> f64 {
        self.score_streams(obs_idx, counts, params, |si, t| {
            self.project_stream_from_acc_real(si, acc, counts, params, t)
        }).iter().sum()
    }

    /// Deprecated-shape helper kept for tests that exercise the flow-only
    /// branch. Equivalent to passing a zeroed counts slice; snapshot streams
    /// would project 0. `acc` is the per-Interval-stream bin vector (Phase 2a).
    #[doc(hidden)]
    pub fn log_likelihood_from_flows(
        &self, acc: &[u64], obs_idx: usize, params: &[f64],
    ) -> f64 {
        let zeros = vec![0i64; self.compiled.int_local_to_global.len()];
        self.log_likelihood_from_flows_and_counts(acc, &zeros, obs_idx, params)
    }

    // ── The failure-path diagnostic (what the model managed here) ──────────

    /// Whether stream `stream_idx` had an observation at union index
    /// `obs_idx`, and if so whether it carried a value.
    ///
    /// The three states all contribute `0.0` to the joint log-likelihood, so
    /// only this accessor can tell them apart from outside.
    pub fn cell_state(&self, stream_idx: usize, obs_idx: usize) -> ObsCellState {
        let s = &self.streams[stream_idx];
        match s.at_union[obs_idx] {
            None => ObsCellState::NotScheduled,
            Some(local) => match s.observations[local] {
                Some(ObsCell::Scalar(v)) => ObsCellState::Scored { y_obs: v },
                None => ObsCellState::Hole,
            },
        }
    }

    /// One stream's projected quantity for one particle at union index
    /// `obs_idx` — the value the likelihood family is built on, not the
    /// family's mean.
    ///
    /// The public face of the same fork the scoring path takes: an Interval
    /// (incidence) stream reads its folded bin `acc[k]`; a prevalence /
    /// `Instant` stream projects from `counts`. It can be NaN — an `Expr`
    /// projection such as `I/(S+I+R)` at zero population — and the NaN is
    /// returned rather than laundered.
    pub fn project_stream(
        &self,
        stream_idx: usize,
        acc: &[u64],
        counts: &[i64],
        params: &[f64],
        obs_idx: usize,
    ) -> f64 {
        let t = self.obs_times[obs_idx];
        self.project_stream_from_acc(stream_idx, acc, counts, params, t)
    }

    /// The observation's time on the model axis.
    pub fn obs_time(&self, obs_idx: usize) -> f64 {
        self.obs_times[obs_idx]
    }

    /// The same instant as a calendar label, when the model declares an
    /// `origin`. Uses the one-to-one renderer, so two observations less than a
    /// day apart do not collapse onto the same label — the identifier has to
    /// name ONE measurement to be worth more than the index it replaces.
    pub fn obs_date(&self, obs_idx: usize) -> Option<String> {
        let origin = self.compiled.model.origin.as_deref()?;
        ir::caltime::internal_to_date_hires(
            origin,
            self.obs_times[obs_idx],
            &self.compiled.model.time_unit,
        )
        .ok()
    }

    /// What the ensemble managed at union index `obs_idx`, one record per
    /// declared stream. **Failure path only** — it re-scores and re-projects,
    /// and is called when a chain is being abandoned.
    ///
    /// `live` carries one `(acc, counts)` pair per LIVE particle, in any order.
    /// Dead particles must NOT be passed: a particle killed earlier by the
    /// process model carries `−∞` without the observation model having been
    /// consulted, so folding it into these counts would report a process-model
    /// failure as a unanimous observation refusal. Their number is passed
    /// separately as `n_dead` and reported beside the live reductions.
    ///
    /// A stream not scheduled here yields a `NotScheduled` record with no
    /// projection summary: its accumulator is mid-interval and there is no
    /// observation to read it against.
    pub fn stream_attempts(
        &self,
        obs_idx: usize,
        live: &[(&[u64], &[i64])],
        n_dead: usize,
        params: &[f64],
    ) -> Vec<StreamAttempt> {
        let n_streams = self.streams.len();
        let t = self.obs_time(obs_idx);
        let date = self.obs_date(obs_idx);
        let cells: Vec<ObsCellState> =
            (0..n_streams).map(|si| self.cell_state(si, obs_idx)).collect();

        let mut projections: Vec<Vec<f64>> = vec![Vec::with_capacity(live.len()); n_streams];
        let mut n_nan = vec![0usize; n_streams];
        let mut n_zero = vec![0usize; n_streams];
        let mut causes: Vec<Vec<(NegInfCause, usize)>> = vec![Vec::new(); n_streams];

        for &(acc, counts) in live {
            // ONE call to the scoring seam per particle, so the per-stream
            // terms explained below are the terms the filter actually weighted
            // with — `score_streams` stays the single authority.
            let scores = self.score_streams(obs_idx, counts, params, |si, ti| {
                self.project_stream_from_acc(si, acc, counts, params, ti)
            });
            for si in 0..n_streams {
                if matches!(cells[si], ObsCellState::NotScheduled) {
                    continue;
                }
                let projected = self.project_stream(si, acc, counts, params, obs_idx);
                if projected.is_nan() {
                    n_nan[si] += 1;
                } else {
                    if projected == 0.0 {
                        n_zero[si] += 1;
                    }
                    projections[si].push(projected);
                }
                // A `Hole` has no value to score against, so it can neither
                // refuse nor be explained.
                let Some(observed) = cells[si].y_obs() else { continue };
                if scores[si] != f64::NEG_INFINITY {
                    continue;
                }
                let s = &self.streams[si];
                let local = s.at_union[obs_idx].expect("a scored cell has a local index");
                let cause = with_scratch_int_from_counts(counts, |int_s| {
                    explain_likelihood_neg_inf(
                        &s.resolved, t, projected, observed, &s.aux[local], params,
                        &self.compiled, int_s, &self.real_s,
                    )
                });
                match causes[si].iter_mut().find(|(c, _)| *c == cause) {
                    Some((_, n)) => *n += 1,
                    None => causes[si].push((cause, 1)),
                }
            }
        }

        (0..n_streams)
            .map(|si| {
                // `f64::total_cmp` rather than `partial_cmp().unwrap()`: the
                // NaN projections are already excluded, and a total order can
                // never panic if one ever slips through.
                let mut vals = std::mem::take(&mut projections[si]);
                vals.sort_by(f64::total_cmp);
                let n_neg_inf: usize = causes[si].iter().map(|(_, n)| *n).sum();
                StreamAttempt {
                    stream: self.streams[si].name.clone(),
                    time: t,
                    date: date.clone(),
                    cell: cells[si],
                    projected_max: vals.last().copied(),
                    // Lower median, so the reported value is one a particle
                    // actually held rather than an interpolated fiction.
                    projected_median: vals.get(vals.len().saturating_sub(1) / 2).copied(),
                    n_projected_zero: n_zero[si],
                    n_projected_nan: n_nan[si],
                    n_neg_inf,
                    neg_inf_causes: std::mem::take(&mut causes[si]),
                    n_live: live.len(),
                    n_dead,
                    n_particles: live.len() + n_dead,
                }
            })
            .collect()
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
        acc: &[u64],
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
            // Not scheduled at this union index ⇒ no term, no gradient.
            // Mirrors the scoring seam exactly.
            let local = match s.at_union[obs_idx] {
                Some(l) => l,
                None => continue,
            };
            // Hole: no term, so no gradient contribution (∂/∂θ of an omitted
            // factor is 0).
            let observed = match s.observations[local] {
                Some(ObsCell::Scalar(v)) => v,
                None => continue,
            };
            let projected = self.project_stream_from_acc(si, acc, counts, params, t);
            with_scratch_int_from_counts(counts, |int_s| {
                eval_likelihood_resolved_grad(
                    &s.resolved, t, projected, observed, &s.aux[local],
                    params, &self.compiled, int_s, &self.real_s,
                    estimated_to_model, &mut grad,
                );
            });
        }
        grad
    }

    /// The deterministic ODE likelihood AND its `∇_θ`, scored from the
    /// forward-sensitivity records (gh#275 det_grad, the obs half). For each obs
    /// time and each stream:
    ///
    /// - `value += log p(y | projected)`, with `projected` computed **continuously**
    ///   from the record (§1d — the incidence `inc` or the prevalence `counts`, both
    ///   unrounded), so the value is smooth in θ;
    /// - `grad += factor1 + factor2`, where **factor 1** is the θ-direct term
    ///   ([`eval_likelihood_resolved_grad`], `projected` held fixed — the same seam
    ///   PGAS uses) and **factor 2** is the trajectory chain
    ///   `(∂logp/∂projected)·(∂projected/∂θ)`: the per-stream sensitivity bin
    ///   `acc_sens[slot]` for an `Interval` (incidence / `FlowSum`) stream,
    ///   `Σ_selected state_sens` off the record for an `Instant` (prevalence /
    ///   `IntCompSum`) stream. The `Sensitivity`-kind split is enforced by the
    ///   projection variant.
    ///
    /// An `Interval` stream is scored against its per-stream bin, not the record's
    /// `inc` — the record closes on the UNION observation axis, so under
    /// heterogeneous cadences a stream's own reporting window spans several records
    /// (gh#680). The bins are folded before scoring and reset only for the streams
    /// scheduled at that union index, the same two-level structure
    /// [`compute_ode_loglik`](crate::inference::compute_ode_loglik) uses, and the
    /// value and sensitivity bins go through the same slot map so they cannot drift.
    ///
    /// Refuses (gh#275 §1h) a `DerivedExpr` prevalence projection and a
    /// `projected`-transforming likelihood argument ([`dlogp_dprojected`]) — v1
    /// supports linear selections and `arg = projected`.
    ///
    /// `records[i]` corresponds to `obs_times[i]` (the recorder emits one per obs
    /// time, in order). `estimated_to_model` maps each of the `d` gradient columns
    /// to its model parameter index.
    pub(crate) fn ode_loglik_and_grad(
        &self,
        records: &[crate::ode::ObsSensitivity],
        params: &[f64],
        estimated_to_model: &[usize],
    ) -> Result<(f64, Vec<f64>), crate::error::SimError> {
        use crate::error::SimError;
        let d = estimated_to_model.len();
        if records.len() != self.obs_times.len() {
            return Err(SimError::Validation(format!(
                "ODE gradient: {} sensitivity records but {} observation times",
                records.len(),
                self.obs_times.len()
            )));
        }
        let mut total_ll = 0.0;
        let mut grad = vec![0.0; d];
        // Per-Interval-stream incidence bins, EXACTLY as the value path
        // (`compute_ode_loglik`) keeps them: the recorder hands us `rec.inc` /
        // `rec.inc_sens`, the blanket per-transition tally over the interval since
        // the previous UNION observation time (the recorder's analogue of the value
        // path's `cum_flows`); the per-stream bin below is what a stream is actually
        // scored against, and it is zeroed only for the streams scheduled at that
        // union index. Without it a 14-day stream loses its first week to a sibling's
        // 7-day observation and is scored against half its window (gh#680).
        // `acc_sens` is the same bins, `d` gradient columns wide, folded and reset by
        // the block-strided siblings so the sensitivity cannot bin differently from
        // the value it differentiates.
        let mut acc: Vec<f64> = vec![0.0; self.n_interval_streams()];
        let mut acc_sens: Vec<f64> = vec![0.0; self.n_interval_streams() * d];
        for (obs_idx, rec) in records.iter().enumerate() {
            let t = self.obs_times[obs_idx];
            // FOLD, before scoring — mirrors `compute_ode_loglik`'s fold seam.
            self.fold_into_acc_real(&rec.inc, &mut acc);
            self.fold_into_acc_real_blocks(&rec.inc_sens, &mut acc_sens, d);
            // The recorder emits one record per obs time, in order; a mismatch means
            // the output grid drifted from the obs grid (a caller bug the recorder's
            // own overshoot check should already have caught).
            debug_assert!(
                (rec.t - t).abs() < 1e-9,
                "record {obs_idx} at t={} does not match obs time {t}",
                rec.t
            );
            // Rounded int-state scratch for any likelihood arg that reads a `Pop`
            // directly (rare); the prevalence PROJECTION below uses the continuous
            // `rec.counts`, so §1d smoothness is preserved where it matters. The
            // real-compartment state is the trajectory's value at this obs time.
            let counts_i64: Vec<i64> =
                rec.counts.iter().map(|&c| c.round().max(0.0) as i64).collect();
            let real_s = RealState::from_vec(rec.real.clone());
            for si in 0..self.streams.len() {
                let s = &self.streams[si];
                let local = match s.at_union[obs_idx] {
                    Some(l) => l,
                    None => continue,
                };
                let observed = match s.observations[local] {
                    Some(ObsCell::Scalar(v)) => v,
                    None => continue,
                };
                // Continuous `projected` and `∂projected/∂θ`, read off the record.
                let (projected, dproj): (f64, Vec<f64>) = match &s.projection {
                    StreamProjection::FlowSum(_) => {
                        // Read the per-stream bin, NOT `rec.inc` — the record is the
                        // blanket union-interval tally, and only `acc` carries this
                        // stream's own reporting window (gh#680). Every `FlowSum`
                        // stream owns a slot by construction (`stream_to_slot`).
                        let slot = self.stream_to_slot[si].expect(
                            "a FlowSum stream must own an Interval acc slot",
                        );
                        let p = acc[slot];
                        let dp = acc_sens[slot * d..(slot + 1) * d].to_vec();
                        (p, dp)
                    }
                    StreamProjection::IntCompSum(idxs) => {
                        let p = idxs.iter().map(|&i| rec.counts[i]).sum();
                        let mut dp = vec![0.0; d];
                        for &i in idxs {
                            for k in 0..d {
                                dp[k] += rec.state_sens[i * d + k];
                            }
                        }
                        (p, dp)
                    }
                    StreamProjection::Expr(value_expr) => {
                        // Continuous eval at the trajectory state (`rec.counts`,
                        // unrounded), so the projected value is smooth in θ. Its
                        // ∂/∂compartment chains through the forward sensitivity S:
                        // ∂projected/∂θ_k = Σ_j (∂proj/∂x_j) · state_sens[j,k]
                        // (gh#275 §1h). The int-state is a placeholder — the
                        // continuous state rides `int_float_override`.
                        //
                        // Co-located silent-zero guard: an empty resolved gradient
                        // here would make the entire factor-2 zero. The capability
                        // gate refuses a DerivedExpr with no ∂proj/∂compartment
                        // (and `det_grad` is `ode_loglik_and_grad`'s only caller and
                        // runs the gate first), so this cannot fire — but assert it
                        // next to the risk so a future caller/reorder can't silently
                        // reintroduce a zero gradient.
                        debug_assert!(
                            !s.projection_state_grad.is_empty(),
                            "DerivedExpr stream '{}' reached ODE scoring with an empty projection \
                             gradient — the capability gate should have refused it (silent-zero \
                             factor-2)",
                            s.name
                        );
                        let placeholder = crate::state::IntState::new(rec.counts.len());
                        let ctx = crate::propensity::EvalCtx {
                            model: &self.compiled, int_s: &placeholder, real_s: &real_s,
                            params, t, dt: 0.0, projected: None, aux: None,
                            int_float_override: Some(&rec.counts), per_eval: None,
                        };
                        let p = eval_resolved(value_expr, &ctx);
                        let mut dp = vec![0.0; d];
                        for (j, dproj_dcomp) in &s.projection_state_grad {
                            let g = eval_resolved(dproj_dcomp, &ctx);
                            for k in 0..d {
                                dp[k] += g * rec.state_sens[j * d + k];
                            }
                        }
                        (p, dp)
                    }
                };
                // value + factor 1 inside the scratch; factor 2 (which can refuse)
                // returns ∂logp/∂projected for chaining by ∂projected/∂θ.
                let dl_dproj = with_scratch_int_from_counts(&counts_i64, |int_s| {
                    total_ll += eval_likelihood_resolved(
                        &s.resolved, t, projected, observed, &s.aux[local],
                        params, &self.compiled, int_s, &real_s,
                    );
                    eval_likelihood_resolved_grad(
                        &s.resolved, t, projected, observed, &s.aux[local],
                        params, &self.compiled, int_s, &real_s,
                        estimated_to_model, &mut grad,
                    );
                    dlogp_dprojected(
                        &s.resolved, t, projected, observed, &s.aux[local],
                        params, &self.compiled, int_s, &real_s,
                    )
                });
                for k in 0..d {
                    grad[k] += dl_dproj * dproj[k];
                }
            }
            // RESET, after scoring — only the streams scheduled at THIS union index
            // close their bin; a stream on a longer cadence keeps accumulating
            // toward its own next observation. Mirrors the value path's
            // `reset_due_acc_real`, extended to the sensitivity block.
            self.reset_due_acc_real(obs_idx, &mut acc);
            self.reset_due_acc_real_blocks(obs_idx, &mut acc_sens, d);
        }
        Ok((total_ll, grad))
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
        // `ParticleState` carries the per-stream folded `acc` (Phase 2a); the
        // trait method is just the flat method with the fields unpacked — so
        // delegate, and keep the per-stream scoring (including the GH#6
        // actual-state handling) in ONE seam. `acc` is the per-Interval-stream
        // bin (the filter folds `flow_accumulators` into it before scoring).
        self.log_likelihood_from_flows_and_counts(
            &state.acc, &state.counts, obs_idx, params,
        )
    }

    fn log_likelihood_per_stream(
        &self, state: &ParticleState, obs_idx: usize, params: &[f64],
    ) -> Vec<f64> {
        // gh#269: per-stream breakdown of `log_likelihood`. Routes through the
        // SAME `score_streams` seam, so `.iter().sum()` of this is byte-equal
        // to `log_likelihood` (no parallel summation loop to drift).
        self.log_likelihood_per_stream_from_flows_and_counts(
            &state.acc, &state.counts, obs_idx, params,
        )
    }

    fn n_observations(&self) -> usize { self.obs_times.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize {
        MultiStreamObsModel::n_streams(self)
    }

    fn stream_names(&self) -> Vec<String> {
        MultiStreamObsModel::stream_names(self)
    }

    fn sample(
        &self, state: &ParticleState, obs_idx: usize,
        params: &[f64], rng: &mut StatefulRng,
    ) -> Vec<f64> {
        let t = self.obs_times[obs_idx];
        (0..self.streams.len()).map(|si| {
            let s = &self.streams[si];
            // Not scheduled at this union index (a sibling's cadence): this
            // stream emits no value here. NaN marks the absent cell, keeping
            // the per-stream output-vector shape; no RNG is consumed for it.
            // (Homogeneous: always `Some`, so order and draws are unchanged.)
            let local = match s.at_union[obs_idx] {
                Some(l) => l,
                None => return f64::NAN,
            };
            let projected = self.project_stream_from_acc(
                si, &state.acc, &state.counts, params, t,
            );
            // GitHub #6: evaluate likelihood args against actual state,
            // not zero scratch. Otherwise state-dependent denominators
            // in p/mean/sd expressions blow up.
            with_scratch_int_from_counts(&state.counts, |int_s| {
                sample_obs_resolved(
                    &s.resolved, t, projected, &s.aux[local], params,
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
            let s = &self.streams[si];
            // Not scheduled at this union index: no value here (NaN placeholder,
            // preserving the output-vector shape). Homogeneous: always `Some`.
            let local = match s.at_union[obs_idx] {
                Some(l) => l,
                None => return f64::NAN,
            };
            let projected = self.project_stream_from_acc(
                si, &state.acc, &state.counts, params, t,
            );
            // GitHub #6: actual state, not zero scratch.
            with_scratch_int_from_counts(&state.counts, |int_s| {
                eval_obs_mean_resolved(
                    &s.resolved, t, projected, &s.aux[local], params,
                    &self.compiled, int_s, &self.real_s,
                )
            })
        }).collect()
    }

    // Multi-cadence per-stream incidence bins (Phase 2a). The trait-object
    // filters (PF / IF2 / correlated-PF) reach these through the trait; PGAS
    // calls the inherent methods directly on the concrete type. Both resolve to
    // the same implementations above.
    fn n_interval_streams(&self) -> usize {
        MultiStreamObsModel::n_interval_streams(self)
    }
    fn fold_into_acc(&self, flow_accumulators: &[u64], acc: &mut [u64]) {
        MultiStreamObsModel::fold_into_acc(self, flow_accumulators, acc)
    }
    fn reset_due_acc(&self, union_idx: usize, acc: &mut [u64]) {
        MultiStreamObsModel::reset_due_acc(self, union_idx, acc)
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
        use ir::observation::{ColumnRole, ObsColumn};
        IrObservationModel {
            name: name.into(),
            source: name.into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: name.into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: name.into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: Projection::CumulativeFlow("inc".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
            }),
        }
    }

    fn spec(name: &str, obs_times: Vec<f64>, observations: Vec<f64>) -> StreamSpec {
        StreamSpec::dense(
            StreamProjection::FlowSum(vec![0]),
            ir_obs(name),
            dense_cells(observations),
            obs_times,
        )
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
    fn non_finite_obs_time_is_fatal_not_a_panic() {
        // A NaN/inf time token must produce a located fatal finding, not a
        // panic. The union merge sorts the times; `NaN.partial_cmp` is `None`,
        // which would `.expect`-panic at the sort if bind did not reject
        // non-finite times first.
        let nan = expect_fatal(
            BoundObs::bind(vec![spec(
                "cases", vec![1.0, f64::NAN, 3.0], vec![0.0, 0.0, 0.0],
            )]),
            "a NaN observation time must be rejected, not panic",
        );
        assert!(nan.is_fatal());
        assert!(nan.findings().iter().any(|f|
            f.message.contains("finite") && f.message.contains("cases")),
            "message must name the stream and the rule: {:?}", nan.findings());

        let inf = expect_fatal(
            BoundObs::bind(vec![spec(
                "cases", vec![1.0, f64::INFINITY], vec![0.0, 0.0],
            )]),
            "an infinite observation time must be rejected",
        );
        assert!(inf.is_fatal());
    }

    #[test]
    fn bind_merges_heterogeneous_schedules() {
        // Multi-cadence Phase 1: two streams on DIFFERENT schedules bind to the
        // sorted-unique UNION axis (no longer a hard error). AFP-like {1,3,8} +
        // ES-like {5} → union {1,3,5,8}. Each stream's `at_union` records, per
        // union-index, its own local cell index (or None where it is not
        // scheduled — a sibling's time).
        let (bound, report) = BoundObs::bind(vec![
            spec("afp", vec![1.0, 3.0, 8.0], vec![10.0, 11.0, 12.0]),
            spec("es", vec![5.0], vec![20.0]),
        ]).expect("heterogeneous schedules must bind to the union axis");
        assert!(!report.is_fatal(), "a valid heterogeneous input is not fatal");

        // The union axis is the sorted-unique merge of both schedules.
        assert_eq!(bound.times(), &[1.0, 3.0, 5.0, 8.0]);

        // afp is scheduled at union-indices 0,1,3 (times 1,3,8 → local 0,1,2),
        // NOT at union-index 2 (time 5, an ES-only time).
        assert_eq!(
            bound.at_union_for_test(0),
            &[Some(0), Some(1), None, Some(2)],
            "afp membership over the union",
        );
        // es is scheduled only at union-index 2 (time 5 → local 0).
        assert_eq!(
            bound.at_union_for_test(1),
            &[None, None, Some(0), None],
            "es membership over the union",
        );

        // A third stream on yet another cadence also binds (prevalence-family
        // projection — irrelevant to the merge, which is purely on times).
        let (bound3, _) = BoundObs::bind(vec![
            spec("afp", vec![1.0, 3.0, 8.0], vec![10.0, 11.0, 12.0]),
            spec("es", vec![5.0], vec![20.0]),
            spec("sero", vec![2.0, 6.0], vec![30.0, 31.0]),
        ]).expect("a third cadence must also bind");
        assert_eq!(bound3.times(), &[1.0, 2.0, 3.0, 5.0, 6.0, 8.0]);
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

        // Homogeneous: the union equals the shared axis, and every stream is
        // scheduled at every union index (`at_union[i] == Some(i)`).
        assert_eq!(bound.times(), &[1.0, 2.0, 3.0]);
        for s in &bound.streams {
            assert_eq!(s.cells.len(), bound.times().len());
            assert_eq!(s.at_union, vec![Some(0), Some(1), Some(2)]);
        }
        assert_eq!(bound.streams[0].cells, dense_cells(vec![10.0, 11.0, 12.0]));
        assert_eq!(bound.streams[1].cells, dense_cells(vec![20.0, 21.0, 22.0]));
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
            ic_grad: Default::default(),
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
                    rate_state_grad: Default::default(),
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
                    source: "cases".into(),
                    columns: vec![
                        ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                        ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                    ],
                    scored: "cases".into(),
                    emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
                    stratum: vec![],
                    projection: Projection::CumulativeFlow("recovery".into()),
                    projection_state_grad: Default::default(),
                    likelihood: Likelihood::Poisson(PoissonLikelihood {
                        // rate = rho * projected
                        rate: ir::Diffable::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                            op: BinOp::Mul,
                            left: Box::new(Expr::Param(ParamExpr { param: "rho".into() })),
                            right: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
                        }})),
                    }),
                },
            ],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![
                Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None },
                Parameter { name: "rho".into(), value: ParamValue::Fixed { value: 0.5 }, param_kind: None, param_dim: None },
            ],
            initial_conditions: InitialConditions::constants({
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
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        Arc::new(CompiledModel::new(m).unwrap())
    }

    /// Build a single-stream incidence obs model from explicit cells.
    fn obs_model(cells: Vec<Option<ObsCell>>, obs_times: Vec<f64>) -> MultiStreamObsModel {
        let compiled = model();
        let rec = compiled.model.transitions.iter()
            .position(|t| t.name == "recovery").unwrap();
        let spec = StreamSpec::dense(
            StreamProjection::FlowSum(vec![rec]),
            compiled.model.observations[0].clone(),
            cells,
            obs_times,
        );
        MultiStreamObsModel::new(
            BoundObs::bind(vec![spec]).expect("bind").0, compiled).unwrap()
    }

    /// gh#268: `joint_observed()` is the per-union-index cross-stream sum used as
    /// the prequential `y_obs` and the `--trace` `observed` column. It must
    /// mirror `score_streams`'s union/hole handling: a stream not scheduled at a
    /// union index contributes nothing, and a hole (`None`) contributes nothing.
    /// Two streams on staggered cadences exercise the `at_union == None` gap that
    /// the same-cadence integration tests don't reach.
    #[test]
    fn joint_observed_sums_scheduled_streams_and_omits_holes() {
        let compiled = model();
        let rec = compiled.model.transitions.iter()
            .position(|t| t.name == "recovery").unwrap();
        let mut ir_a = compiled.model.observations[0].clone();
        ir_a.name = "a".into();
        let mut ir_b = compiled.model.observations[0].clone();
        ir_b.name = "b".into();
        // a: t ∈ {1, 3}, value 10 at t=1 and a HOLE at t=3.
        let spec_a = StreamSpec::dense(
            StreamProjection::FlowSum(vec![rec]),
            ir_a,
            vec![Some(ObsCell::Scalar(10.0)), None],
            vec![1.0, 3.0],
        );
        // b: t ∈ {2}, value 20.
        let spec_b = StreamSpec::dense(
            StreamProjection::FlowSum(vec![rec]),
            ir_b,
            dense_cells(vec![20.0]),
            vec![2.0],
        );
        let m = MultiStreamObsModel::new(
            BoundObs::bind(vec![spec_a, spec_b]).expect("bind").0, compiled,
        ).unwrap();
        // union axis = [1, 2, 3]:
        //   t=1: a=10 (scheduled+observed), b not scheduled        -> 10
        //   t=2: a not scheduled,           b=20                    -> 20
        //   t=3: a hole (omitted),          b not scheduled         ->  0
        assert_eq!(m.joint_observed(), vec![10.0, 20.0, 0.0]);
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
        // agree on dense cells. The model has one `FlowSum(vec![recovery])`
        // stream ⇒ one `acc` slot; the seam reads `acc[0]` directly. `flows`
        // here IS that already-summed per-stream bin (single transition), so the
        // flat `acc` arg and `state.acc` are the same vector.
        let state = ParticleState {
            counts: counts.clone(),
            flow_accumulators: flows.clone(),
            acc: flows.clone(),
        };
        for idx in 0..3 {
            let flat = m.log_likelihood_from_flows_and_counts(&flows, &counts, idx, &params);
            let via_state = m.log_likelihood(&state, idx, &params);
            assert!(flat.is_finite(), "dense scoring must be finite at idx {idx}, got {flat}");
            assert_eq!(flat, via_state,
                "flat and trait paths must agree on dense cells at idx {idx}: \
                 flat={flat} state={via_state}");
        }
    }

    /// gh#269 matrix-drift guard: the joint log-likelihood MUST equal the sum
    /// of the per-stream contributions — structurally, since both route through
    /// `score_streams` (`.iter().sum()` is the joint, the Vec is per-stream).
    /// A 2-stream model with both streams SCHEDULED+OBSERVED at the index makes
    /// each term nonzero (so the sum is a real check, not 0 == 0).
    #[test]
    fn sum_of_per_stream_loglik_equals_joint() {
        let compiled = model();
        let rec = compiled.model.transitions.iter()
            .position(|t| t.name == "recovery").unwrap();
        let mut ir_a = compiled.model.observations[0].clone();
        ir_a.name = "a".into();
        let mut ir_b = compiled.model.observations[0].clone();
        ir_b.name = "b".into();
        // Both streams on the SAME cadence so both are scheduled at every union
        // index — distinct observed values so the two terms differ.
        let times = vec![7.0, 14.0];
        let spec_a = StreamSpec::dense(
            StreamProjection::FlowSum(vec![rec]), ir_a,
            dense_cells(vec![30.0, 25.0]), times.clone());
        let spec_b = StreamSpec::dense(
            StreamProjection::FlowSum(vec![rec]), ir_b,
            dense_cells(vec![40.0, 55.0]), times.clone());
        let m = MultiStreamObsModel::new(
            BoundObs::bind(vec![spec_a, spec_b]).expect("bind").0, compiled,
        ).unwrap();

        let counts = vec![900i64, 40, 60];
        // One `acc` slot per Interval stream (two FlowSum streams ⇒ 2 slots).
        let acc = vec![100u64, 100u64];
        let params = model().default_params.clone();
        let state = ParticleState {
            counts: counts.clone(),
            flow_accumulators: vec![100u64],
            acc: acc.clone(),
        };

        for idx in 0..times.len() {
            let joint = m.log_likelihood_from_flows_and_counts(&acc, &counts, idx, &params);
            let per_stream =
                m.log_likelihood_per_stream_from_flows_and_counts(&acc, &counts, idx, &params);
            assert_eq!(per_stream.len(), 2, "two streams ⇒ two contributions");
            // Each stream contributes a real (nonzero, finite) term here.
            for (si, &v) in per_stream.iter().enumerate() {
                assert!(v.is_finite() && v != 0.0,
                    "stream {si} at idx {idx} must contribute a real term, got {v}");
            }
            let sum: f64 = per_stream.iter().sum();
            assert_eq!(sum, joint,
                "Σ per_stream ({sum}) must equal joint ({joint}) at idx {idx}");
            // Trait path agrees too (uses the same seam).
            let via_state = m.log_likelihood_per_stream(&state, idx, &params);
            assert_eq!(via_state, per_stream,
                "trait per-stream must match flat per-stream at idx {idx}");
            assert_eq!(via_state.iter().sum::<f64>(),
                m.log_likelihood(&state, idx, &params),
                "trait: Σ per_stream must equal joint log_likelihood at idx {idx}");
        }
    }
}

