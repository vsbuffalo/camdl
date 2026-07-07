//! Pre-resolved expression trees for hot-path evaluation.
//!
//! `ResolvedExpr` mirrors `ir::expr::Expr` but replaces all string-keyed
//! lookups (param names, compartment names, time function names, table names)
//! with pre-resolved `usize` indices. Constructed once at `CompiledModel::new()`
//! time, evaluated billions of times in the inference inner loop.
//!
//! The resolver (`resolve_expr`) validates all names against the model's index
//! maps, surfacing errors at model construction. The evaluator (`eval_resolved`)
//! is infallible — no `Result`, no HashMap probes, just array indexing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use ir::expr::{BinOp, Expr, UnOp};
use ir::deriv::{CompGradMap, DerivEntry, UnsupportedReason};
use ir::table::OobPolicy;

use crate::error::SimError;
use crate::propensity::{eval_forcing, EvalCtx};

// ── Resolved expression tree ─────────────────────────────────────────────────

/// Pre-resolved expression. All string lookups replaced by `usize` indices.
#[derive(Debug, Clone)]
pub enum ResolvedExpr {
    Const(f64),
    /// Index into `params[]`.
    Param(usize),
    /// Local integer compartment index → `int_s.counts[i] as f64`.
    IntPop(usize),
    /// Local real compartment index → `real_s.values[i]`.
    RealPop(usize),
    /// Sum of integer compartments by local index (common fast path).
    IntPopSum(Vec<usize>),
    /// Sum mixing integer and real compartments (rare — stratified models
    /// that combine integer and real compartments in a single `pop_sum`).
    MixedPopSum {
        int_indices: Vec<usize>,
        real_indices: Vec<usize>,
    },
    Time,
    Dt,  // gh#54: runtime integrator step
    BinOp {
        op: BinOp,
        left: Box<ResolvedExpr>,
        right: Box<ResolvedExpr>,
    },
    UnOp {
        op: UnOp,
        arg: Box<ResolvedExpr>,
    },
    Cond {
        pred: Box<ResolvedExpr>,
        then_: Box<ResolvedExpr>,
        else_: Box<ResolvedExpr>,
    },
    /// Index into `time_func_cache[]`.
    TimeFunc(usize),
    /// Table index + resolved sub-expression for the lookup index.
    TableLookup {
        table_idx: usize,
        /// Cached OOB policy (avoids indirection through model at eval time).
        oob: OobPolicy,
        /// Cached table length.
        table_len: usize,
        index: Box<ResolvedExpr>,
    },
    /// Returns `ctx.projected` (observation likelihood context only).
    Projected,
    /// Per-observation auxiliary data column referenced by name (e.g. a binomial
    /// denominator `n = tested`). Reads `ctx.aux` by name — observation
    /// likelihood context only. 2026-06-10 observation data-entry §3.
    ObsColumnRef(String),
    /// Dimensional escape; transparent at eval time (identity over
    /// `inner`). The asserted dim is a compile-time concern only and
    /// isn't stored here — the dim-checker has already consumed it.
    UncheckedDim { inner: Box<ResolvedExpr> },
    /// n-ary sum over already-resolved terms (Fix D). Evaluated as a left-fold
    /// to match the OCaml Add-chain order bit-for-bit.
    Reduce(Vec<ResolvedExpr>),
    /// Fix B: reference to a model-level binding by slot. Evaluated on-demand
    /// from `ctx.model.resolved.bindings[slot]`.
    BindingRef(usize),
    /// gh#272 LICM: reference to a model-level per-eval binding by slot.
    /// Param/table-only and loop-invariant; evaluated on-demand from
    /// `ctx.model.resolved.per_eval_bindings[slot]` (a later increment adds the
    /// per-eval cache tier). Produced only by the LICM pass.
    PerEvalRef(usize),
}

/// Returns true if the expression references compartment state (Pop, PopSum).
/// Used to check whether an expression can be evaluated at a fixed state
/// or needs per-particle evaluation.
pub fn references_state(expr: &ResolvedExpr) -> bool {
    match expr {
        ResolvedExpr::IntPop(_)
        | ResolvedExpr::RealPop(_)
        | ResolvedExpr::IntPopSum(_)
        | ResolvedExpr::MixedPopSum { .. } => true,
        ResolvedExpr::BinOp { left, right, .. } =>
            references_state(left) || references_state(right),
        ResolvedExpr::UnOp { arg, .. } => references_state(arg),
        ResolvedExpr::Cond { pred, then_, else_ } =>
            references_state(pred) || references_state(then_) || references_state(else_),
        ResolvedExpr::TableLookup { index, .. } => references_state(index),
        ResolvedExpr::UncheckedDim { inner } => references_state(inner),
        ResolvedExpr::Reduce(terms) => terms.iter().any(references_state),
        // Hoisted bindings are state-derived (N/I_agg/F read compartments).
        ResolvedExpr::BindingRef(_) => true,
        // gh#272: per-eval bindings are param/table-only by construction —
        // `CompiledModel::new` rejects any state (or time-varying) reference via
        // `per_eval_staging_violation` (gh#284) — so they never reference state.
        // The contrast with BindingRef above is the keystone invariant boundary.
        ResolvedExpr::PerEvalRef(_) => false,
        _ => false,
    }
}

/// gh#284: the LICM per-eval staging contract, enforced in Rust as well as in
/// the OCaml pass (`licm.ml is_invariant`). The body of per-eval binding `slot`
/// is staged ONCE per θ-stable span (`stage_per_eval`, at `t_start` against a
/// zero scratch) and then read every substep, and `eval_per_eval_scratch` lends
/// body `slot` only the already-filled prefix `&scratch[..slot]`. So the body
/// must be BOTH:
///   - loop-invariant — a function of parameters, tables, and constants only; and
///   - topologically ordered — any `PerEvalRef` it contains must point to a
///     STRICTLY EARLIER slot (`< slot`).
/// Returns `Some(kind)` naming the first node that breaks the contract, or `None`
/// if the body is well-formed.
///
/// The invariance half is strictly stronger than [`references_state`]: it rejects
/// time-varying nodes (`Time` / `Dt` / `TimeFunc`) as well as compartment state.
/// A state-referencing body would PANIC on the zero scratch (`IntState::new(0)`
/// index-OOB); a time-varying one would be staged stale and read wrong every
/// later substep (silent-wrong). The ordering half closes the same panic class
/// from the other direction: a forward/self `PerEvalRef(slot' >= slot)` reads an
/// unfilled scratch slot (staged path) or recurses forever (on-demand fallback).
/// The match is exhaustive on purpose: a new `ResolvedExpr` variant must be
/// classified here, not silently treated as well-formed.
pub fn per_eval_staging_violation(expr: &ResolvedExpr, slot: usize) -> Option<&'static str> {
    match expr {
        // Invariant leaves: constant for the whole span once θ is bound.
        ResolvedExpr::Const(_) | ResolvedExpr::Param(_) => None,
        // An earlier per-eval binding is itself well-formed by this same check;
        // a forward or self reference breaks the topological staging order.
        ResolvedExpr::PerEvalRef(other) if *other >= slot => {
            Some("a forward or cyclic per-eval reference")
        }
        ResolvedExpr::PerEvalRef(_) => None,
        ResolvedExpr::TableLookup { index, .. } => per_eval_staging_violation(index, slot),

        // Compartment state — would panic on the zero stage scratch.
        ResolvedExpr::IntPop(_)
        | ResolvedExpr::RealPop(_)
        | ResolvedExpr::IntPopSum(_)
        | ResolvedExpr::MixedPopSum { .. } => Some("compartment state (Pop / PopSum)"),
        // A hoisted binding is state-derived (reads compartments).
        ResolvedExpr::BindingRef(_) => Some("a state-derived binding"),

        // Time-varying — would be staged stale (silent-wrong every later substep).
        ResolvedExpr::Time => Some("simulation time (t)"),
        ResolvedExpr::Dt => Some("the integrator step (dt)"),
        ResolvedExpr::TimeFunc(_) => Some("a forcing (time-function)"),

        // Observation-context-only — never valid in a dynamics surface anyway.
        ResolvedExpr::Projected => Some("the observation projection"),
        ResolvedExpr::ObsColumnRef(_) => Some("an observation data column"),

        // Compound: well-formed iff every child is.
        ResolvedExpr::BinOp { left, right, .. } => per_eval_staging_violation(left, slot)
            .or_else(|| per_eval_staging_violation(right, slot)),
        ResolvedExpr::UnOp { arg, .. } => per_eval_staging_violation(arg, slot),
        ResolvedExpr::Cond { pred, then_, else_ } => per_eval_staging_violation(pred, slot)
            .or_else(|| per_eval_staging_violation(then_, slot))
            .or_else(|| per_eval_staging_violation(else_, slot)),
        ResolvedExpr::UncheckedDim { inner } => per_eval_staging_violation(inner, slot),
        ResolvedExpr::Reduce(terms) => {
            terms.iter().find_map(|t| per_eval_staging_violation(t, slot))
        }
    }
}

// ── Resolution context ───────────────────────────────────────────────────────

/// Borrows all index maps needed to resolve an `Expr` → `ResolvedExpr`.
/// Constructed once during `CompiledModel::new()`.
pub struct ResolveCtx<'a> {
    pub comp_index: &'a HashMap<String, usize>,
    pub param_index: &'a HashMap<String, usize>,
    pub time_func_index: &'a HashMap<String, usize>,
    pub table_index: &'a HashMap<String, usize>,
    pub global_to_int: &'a [Option<usize>],
    pub global_to_real: &'a [Option<usize>],
    /// Per-table: (oob_policy, cached_values_len).
    pub table_meta: &'a [(OobPolicy, usize)],
    /// Fix B: model-level binding name → slot. `BindingRef(name)` resolves to
    /// `ResolvedExpr::BindingRef(slot)`, like Param/Pop/TableLookup.
    pub binding_index: &'a HashMap<String, usize>,
    /// gh#272 LICM: per-eval binding name → slot. `PerEvalRef(name)` resolves to
    /// `ResolvedExpr::PerEvalRef(slot)`, the sibling of `binding_index`.
    pub per_eval_index: &'a HashMap<String, usize>,
}

/// Resolve an `Expr` tree into a `ResolvedExpr` tree.
///
/// All name-not-found errors surface here at model construction time.
/// The resulting `ResolvedExpr` can be evaluated infallibly.
pub fn resolve_expr(expr: &Expr, ctx: &ResolveCtx<'_>) -> Result<ResolvedExpr, SimError> {
    match expr {
        Expr::Const(c) => Ok(ResolvedExpr::Const(c.value)),

        Expr::Param(p) => {
            let idx = *ctx.param_index.get(p.param.as_str())
                .ok_or_else(|| SimError::UnknownParameter(p.param.clone()))?;
            Ok(ResolvedExpr::Param(idx))
        }

        Expr::Pop(p) => {
            let global = *ctx.comp_index.get(p.pop.as_str())
                .ok_or_else(|| SimError::UnknownCompartment(p.pop.clone()))?;
            if let Some(local) = ctx.global_to_int[global] {
                Ok(ResolvedExpr::IntPop(local))
            } else if let Some(local) = ctx.global_to_real[global] {
                Ok(ResolvedExpr::RealPop(local))
            } else {
                Err(SimError::UnknownCompartment(p.pop.clone()))
            }
        }

        Expr::PopSum(ps) => {
            let mut int_indices = Vec::new();
            let mut real_indices = Vec::new();
            for name in &ps.pop_sum {
                let global = *ctx.comp_index.get(name.as_str())
                    .ok_or_else(|| SimError::UnknownCompartment(name.clone()))?;
                if let Some(local) = ctx.global_to_int[global] {
                    int_indices.push(local);
                } else if let Some(local) = ctx.global_to_real[global] {
                    real_indices.push(local);
                }
            }
            if real_indices.is_empty() {
                Ok(ResolvedExpr::IntPopSum(int_indices))
            } else {
                Ok(ResolvedExpr::MixedPopSum { int_indices, real_indices })
            }
        }

        Expr::Time(_) => Ok(ResolvedExpr::Time),
        Expr::Dt(_)   => Ok(ResolvedExpr::Dt),

        Expr::BinOp(w) => {
            let left = resolve_expr(&w.bin_op.left, ctx)?;
            let right = resolve_expr(&w.bin_op.right, ctx)?;
            Ok(ResolvedExpr::BinOp {
                op: w.bin_op.op.clone(),
                left: Box::new(left),
                right: Box::new(right),
            })
        }

        Expr::UnOp(w) => {
            let arg = resolve_expr(&w.un_op.arg, ctx)?;
            Ok(ResolvedExpr::UnOp {
                op: w.un_op.op.clone(),
                arg: Box::new(arg),
            })
        }

        Expr::Cond(w) => {
            let pred = resolve_expr(&w.cond.pred, ctx)?;
            let then_ = resolve_expr(&w.cond.then, ctx)?;
            let else_ = resolve_expr(&w.cond.else_, ctx)?;
            Ok(ResolvedExpr::Cond {
                pred: Box::new(pred),
                then_: Box::new(then_),
                else_: Box::new(else_),
            })
        }

        Expr::TimeFunc(w) => {
            let idx = *ctx.time_func_index.get(w.time_func.name.as_str())
                .ok_or_else(|| SimError::UnknownTimeFunction(w.time_func.name.clone()))?;
            Ok(ResolvedExpr::TimeFunc(idx))
        }

        Expr::TableLookup(w) => {
            let table_idx = *ctx.table_index.get(w.table_lookup.table.as_str())
                .ok_or_else(|| SimError::UnknownTable(w.table_lookup.table.clone()))?;
            if w.table_lookup.indices.len() != 1 {
                return Err(SimError::TableLookup(format!(
                    "table '{}' requires exactly 1 index, got {}",
                    w.table_lookup.table, w.table_lookup.indices.len()
                )));
            }
            let index = resolve_expr(&w.table_lookup.indices[0], ctx)?;
            let (oob, table_len) = &ctx.table_meta[table_idx];
            Ok(ResolvedExpr::TableLookup {
                table_idx,
                oob: oob.clone(),
                table_len: *table_len,
                index: Box::new(index),
            })
        }

        Expr::Projected(_) => Ok(ResolvedExpr::Projected),

        Expr::ObsColumnRef(w) => {
            // Resolved by name at eval against `ctx.aux` (filled per
            // observation by the scoring loop) — no index map to consult here.
            Ok(ResolvedExpr::ObsColumnRef(w.obs_column_ref.clone()))
        }

        Expr::UncheckedDim(w) => {
            let inner = resolve_expr(&w.unchecked_dim.inner, ctx)?;
            Ok(ResolvedExpr::UncheckedDim { inner: Box::new(inner) })
        }
        Expr::Reduce(w) => {
            let terms = w.reduce.iter()
                .map(|e| resolve_expr(e, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ResolvedExpr::Reduce(terms))
        }
        Expr::BindingRef(w) => {
            let slot = *ctx.binding_index.get(w.binding_ref.as_str())
                .ok_or_else(|| SimError::Validation(
                    format!("reference to unknown binding '{}'", w.binding_ref)))?;
            Ok(ResolvedExpr::BindingRef(slot))
        }
        Expr::PerEvalRef(w) => {
            let slot = *ctx.per_eval_index.get(w.per_eval_ref.as_str())
                .ok_or_else(|| SimError::Validation(
                    format!("reference to unknown per-eval binding '{}'", w.per_eval_ref)))?;
            Ok(ResolvedExpr::PerEvalRef(slot))
        }
    }
}

// ── Binding evaluation cache ─────────────────────────────────────────────────
//
// A Fix-B hoisted binding (`N[l]`, `I_agg[l]`, …) is referenced once per
// destination stratum — ~945× each on a dense P=44 spatial model — and
// `BindingRef` re-evaluates its body on every reference. Within ONE
// propensity-vector evaluation (a single state snapshot) a binding's value is
// constant, so we memoize: compute once per state, reuse across all rate trees.
// Measured: `eval_resolved` is 46–54% of sim-thread compute, dominated by these
// redundant re-evals.
//
// Correctness:
// - thread-local: PF/PGAS parallelise across particles; each worker owns its
//   cache, so there is no cross-particle aliasing.
// - `active` only inside `eval_propensities` (via `CacheScope`): the cache holds
//   one state's values. Observation-likelihood / gradient evals run at other
//   states and outside this scope, so they fall through to on-demand eval —
//   byte-identical to the pre-cache behaviour.
// - O(1) invalidation: a generation counter bumped per `eval_propensities`
//   call; a slot is fresh iff its stamp equals the current generation.
//
// Pinned by the byte-identical A/B gate (`tests/gate_binding_cache_ab.rs`):
// cache on vs off → identical trajectories.

#[derive(Default)]
struct BindingCache {
    val:    Vec<f64>,
    stamp:  Vec<u32>,
    gen:    u32,
    active: bool,
    /// Lifetime hit count on this thread. Read by the A/B gate to prove the
    /// cache actually served hits (else byte-identity proves nothing). Not used
    /// on the hot path beyond a single increment per hit.
    hits:   u64,
}

thread_local! {
    static BINDING_CACHE: RefCell<BindingCache> = RefCell::new(BindingCache::default());
}

thread_local! {
    /// gh#127 (#12): the most recent out-of-range table lookup hit by the
    /// infallible fast evaluator on this thread, as `(table_idx, index, len)`.
    /// `eval_resolved` cannot return `Result` (it is called from ~30 hot,
    /// non-`Result` sites incl. the inference inner loop), so on an OOB it
    /// records the offending lookup here and returns `f64::NAN`. The NaN
    /// propagates to the `Result`-returning boundary (`eval_propensities`),
    /// which clears this at entry and, when a rate evaluates to NaN, consults
    /// it to surface a NAMED `SimError::TableLookup` (table + index + len)
    /// instead of a generic `NumericalCollapse` — and a controlled per-particle
    /// error instead of a process-wide panic. Per-thread, matching the
    /// particle-parallel PF/PGAS model (each worker owns its cell).
    static LAST_TABLE_OOB: std::cell::Cell<Option<(usize, i64, usize)>> =
        const { std::cell::Cell::new(None) };
}

/// gh#127 (#12): clear this thread's pending table-OOB record. Called at the
/// start of `eval_propensities` so a stale OOB from an earlier evaluation can
/// never be attributed to a later, innocent one.
#[inline]
pub fn clear_table_oob() {
    LAST_TABLE_OOB.with(|c| c.set(None));
}

/// gh#127 (#12): take (read and clear) this thread's pending table-OOB record.
/// `eval_propensities` calls this when a rate evaluates to NaN to build a named
/// `SimError::TableLookup`. Returns `(table_idx, floored_index, table_len)`.
#[inline]
pub fn take_table_oob() -> Option<(usize, i64, usize)> {
    LAST_TABLE_OOB.with(|c| c.take())
}

#[inline]
fn record_table_oob(table_idx: usize, index: i64, len: usize) {
    LAST_TABLE_OOB.with(|c| c.set(Some((table_idx, index, len))));
}

thread_local! {
    /// Per-thread test/bench override of the cache state. `None` → fall through
    /// to the `CAMDL_NO_BINDING_CACHE` env default.
    static CACHE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Escape hatch (A/B gate / debugging): `CAMDL_NO_BINDING_CACHE` forces the
/// on-demand path, making a run comparable to the pre-cache evaluator. The
/// per-thread override (set by the A/B gate) wins so cache-on and cache-off can
/// be compared in one process.
fn binding_cache_disabled() -> bool {
    if let Some(off) = CACHE_OVERRIDE.with(|c| c.get()) {
        return off;
    }
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("CAMDL_NO_BINDING_CACHE").is_some())
}

/// Test/bench hook: force the binding cache off (`true`) or on (`false`) for the
/// current thread, overriding `CAMDL_NO_BINDING_CACHE`. The A/B gate uses it to
/// run cache-on and cache-off in one process and assert byte-identity.
pub fn set_binding_cache_disabled(off: bool) {
    CACHE_OVERRIDE.with(|c| c.set(Some(off)));
}

/// Test/bench hook: read and reset this thread's cumulative binding-cache hit
/// count. The A/B gate calls it after a cache-on run to assert hits > 0 (the
/// cache served reuse — so byte-identity is a non-vacuous claim).
pub fn take_binding_cache_hits() -> u64 {
    BINDING_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let n = c.hits;
        c.hits = 0;
        n
    })
}

/// RAII scope that activates the binding cache for one propensity-vector
/// evaluation. `enter` bumps the generation (invalidating the prior state's
/// values) and marks the cache active; `Drop` deactivates it so any eval
/// outside the propensity loop never reads a stale value.
pub struct CacheScope;

impl CacheScope {
    #[inline]
    pub fn enter(n_bindings: usize) -> Self {
        if !binding_cache_disabled() {
            BINDING_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.val.len() != n_bindings {
                    c.val = vec![0.0; n_bindings];
                    c.stamp = vec![0; n_bindings];
                    c.gen = 0;
                }
                c.gen = c.gen.wrapping_add(1);
                if c.gen == 0 {
                    // Generation wrapped: clear stamps so no stale slot aliases gen 0.
                    c.stamp.iter_mut().for_each(|s| *s = 0);
                    c.gen = 1;
                }
                c.active = true;
            });
        }
        CacheScope
    }
}

impl Drop for CacheScope {
    #[inline]
    fn drop(&mut self) {
        if !binding_cache_disabled() {
            BINDING_CACHE.with(|c| c.borrow_mut().active = false);
        }
    }
}

// ── Infallible evaluator ─────────────────────────────────────────────────────

/// Evaluate a pre-resolved expression. **Infallible** — all name validation
/// happened at resolve time. No HashMap lookups, no `Result` propagation.
#[inline]
pub fn eval_resolved(expr: &ResolvedExpr, ctx: &EvalCtx<'_>) -> f64 {
    match expr {
        ResolvedExpr::Const(v) => *v,

        ResolvedExpr::Param(idx) => ctx.params[*idx],

        ResolvedExpr::IntPop(local) => match ctx.int_float_override {
            Some(f) => f[*local],
            None => ctx.int_s.counts[*local] as f64,
        },

        ResolvedExpr::RealPop(local) => ctx.real_s.values[*local],

        ResolvedExpr::IntPopSum(indices) => match ctx.int_float_override {
            Some(f) => indices.iter().map(|&i| f[i]).sum(),
            None => indices.iter().map(|&i| ctx.int_s.counts[i] as f64).sum(),
        },

        ResolvedExpr::MixedPopSum { int_indices, real_indices } => {
            let int_sum: f64 = match ctx.int_float_override {
                Some(f) => int_indices.iter().map(|&i| f[i]).sum(),
                None => int_indices.iter().map(|&i| ctx.int_s.counts[i] as f64).sum(),
            };
            let real_sum: f64 = real_indices.iter().map(|&i| ctx.real_s.values[i]).sum();
            int_sum + real_sum
        }

        ResolvedExpr::Time => ctx.t,
        ResolvedExpr::Dt   => ctx.dt,

        ResolvedExpr::BinOp { op, left, right } => {
            let a = eval_resolved(left, ctx);
            let b = eval_resolved(right, ctx);
            // gh#audit-C6 / S1. eval_resolved is infallible by
            // contract — it returns f64. To mirror eval_expr's typed
            // error behaviour without breaking the signature, the
            // collapse paths return NaN under hard-fail mode (default);
            // eval_propensities detects NaN downstream and converts
            // to SimError::NumericalCollapse. Under
            // --allow-degenerate-rates the legacy 0.0 sentinel is
            // returned so existing models work as before.
            let allow = crate::eval_stats::allow_degenerate_rates();
            match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => {
                    if b == 0.0 {
                        crate::eval_stats::inc_div_by_zero();
                        if allow { 0.0 } else { f64::NAN }
                    } else { a / b }
                }
                BinOp::Pow => {
                    let r = a.powf(b);
                    if r.is_nan() || r.is_infinite() {
                        crate::eval_stats::inc_pow_nan_inf();
                        if allow { 0.0 } else { f64::NAN }
                    } else { r }
                }
                BinOp::Mod => {
                    if b == 0.0 {
                        crate::eval_stats::inc_div_by_zero();
                        if allow { 0.0 } else { f64::NAN }
                    } else { a.rem_euclid(b) }
                }
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                BinOp::Eq  => if a == b { 1.0 } else { 0.0 },
                BinOp::Neq => if a != b { 1.0 } else { 0.0 },
                BinOp::Lt  => if a <  b { 1.0 } else { 0.0 },
                BinOp::Gt  => if a >  b { 1.0 } else { 0.0 },
                BinOp::Le  => if a <= b { 1.0 } else { 0.0 },
                BinOp::Ge  => if a >= b { 1.0 } else { 0.0 },
            }
        }

        ResolvedExpr::UnOp { op, arg } => {
            let a = eval_resolved(arg, ctx);
            let allow = crate::eval_stats::allow_degenerate_rates();
            let result = match op {
                UnOp::Neg   => -a,
                UnOp::Exp   => a.exp(),
                UnOp::Log   => if a > 0.0 { a.ln() } else { f64::NEG_INFINITY },
                UnOp::Sqrt  => if a >= 0.0 { a.sqrt() } else if allow { 0.0 } else { f64::NAN },
                UnOp::Abs   => a.abs(),
                UnOp::Floor => a.floor(),
                UnOp::Ceil  => a.ceil(),
                UnOp::Sin   => a.sin(),
                UnOp::Cos   => a.cos(),
                UnOp::Tanh  => a.tanh(),
            };
            if result.is_nan() {
                crate::eval_stats::inc_unop_nan();
                if allow { 0.0 } else { f64::NAN }
            } else { result }
        }

        ResolvedExpr::Cond { pred, then_, else_ } => {
            if eval_resolved(pred, ctx) > 0.0 {
                eval_resolved(then_, ctx)
            } else {
                eval_resolved(else_, ctx)
            }
        }

        ResolvedExpr::TimeFunc(idx) => {
            // gh#314: apply the optional evaluation-time shift, uniform across
            // every forcing kind. This is the compiled/inference hot path
            // (PGAS/IF2/PF/ODE), so the shift MUST be here too — not only on the
            // AST `eval_expr` path — or a lagged forcing is silently unshifted
            // during inference. `lag` is already in model time units.
            let tf = &ctx.model.time_func_cache[*idx];
            let t_eff = match &tf.lag {
                Some(lag) => ctx.t - eval_resolved(lag, ctx),
                None => ctx.t,
            };
            eval_forcing(&tf.kind, t_eff, ctx)
        }

        ResolvedExpr::TableLookup { table_idx, oob, table_len, index } => {
            let cached = &ctx.model.table_values_cache[*table_idx];
            let raw = eval_resolved(index, ctx);
            let table_idx_val = raw.floor() as i64;
            let n = *table_len as i64;
            match oob {
                // Out-of-range table lookups fail loud: the index is a model
                // assertion ("never out of range"); a hot-path violation is a
                // model bug, not something to silently clamp or wrap (RM3,
                // 2026-04-19 engine review).
                //
                // gh#127 (#12): this evaluator is infallible (`-> f64`) and is
                // called from ~30 hot, non-`Result` sites (incl. the inference
                // inner loop), so it cannot `panic!` — one bad particle must not
                // crash the whole process — and it cannot return `Result`
                // without rippling into those callers. Instead it records the
                // offending lookup on a thread-local and returns NaN. The NaN
                // propagates to the `Result`-returning boundary
                // (`eval_propensities`), which turns it into a named
                // `SimError::TableLookup` (table + index + len) — a controlled
                // per-particle error. For a COMPILE-TIME-CONSTANT index the
                // out-of-range case is already rejected earlier, by `validate`
                // (ir/validate.rs::TableLookupConstantIndexOutOfRange); this arm
                // only fires for a non-constant (state/param-dependent) index.
                // The slow-path evaluator (`eval_expr`, propensity.rs) is already
                // `Result`-returning and surfaces `SimError::TableLookup` for the
                // same out-of-range condition directly (via `table_lookup()`).
                OobPolicy::Error => {
                    if table_idx_val < 0 || table_idx_val >= n {
                        crate::eval_stats::inc_table_oob();
                        record_table_oob(*table_idx, table_idx_val, *table_len);
                        return f64::NAN;
                    }
                    // Table values are live `ResolvedExpr` (const or
                    // param-referencing) — evaluate the selected entry.
                    eval_resolved(&cached[table_idx_val as usize], ctx)
                }
            }
        }

        ResolvedExpr::Projected => {
            // In observation likelihood context, projected is always Some.
            // Outside that context this variant should never appear (resolver
            // only produces it from Expr::Projected which only appears in
            // likelihood fields).
            ctx.projected.unwrap_or(0.0)
        }

        ResolvedExpr::ObsColumnRef(name) => {
            // Per-observation aux value, looked up by name in `ctx.aux` (filled
            // by the scoring loop). Only appears in a likelihood; outside that
            // context (no aux) it floors to 0.0, like `Projected`. A
            // referenced-but-absent aux is a binder error (the cell is a hole),
            // so this miss is not reached on the scored path.
            ctx.aux
                .and_then(|kvs| kvs.iter().find(|(k, _)| k == name).map(|(_, v)| *v))
                .unwrap_or(0.0)
        }

        ResolvedExpr::UncheckedDim { inner } => {
            // Identity at eval time. The dim assertion was consumed at
            // compile time by the dim-checker.
            eval_resolved(inner, ctx)
        }
        // Left-fold (sum() seeds -0.0) → bit-identical to the OCaml
        // `((t0+t1)+…)` Add-chain (-0.0 + t0 == t0). Empty Reduce → -0.0.
        // NaN propagates naturally.
        ResolvedExpr::Reduce(terms) => terms.iter().map(|t| eval_resolved(t, ctx)).sum(),
        // On-demand: evaluate the binding's body. Topologically ordered (a binding
        // only references earlier ones), so this recursion terminates.
        ResolvedExpr::BindingRef(slot) => {
            // Memoized within one propensity-vector evaluation (see BindingCache).
            // The borrow is released before the miss-path recursion below — a
            // binding body may reference earlier bindings, re-entering this arm.
            let hit = BINDING_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.active && *slot < c.stamp.len() && c.stamp[*slot] == c.gen {
                    c.hits = c.hits.wrapping_add(1);
                    Some(c.val[*slot])
                } else {
                    None
                }
            });
            match hit {
                Some(v) => v,
                None => {
                    // Borrow released before recursing — a binding body may
                    // reference earlier bindings, which re-enter this arm.
                    let v = eval_resolved(&ctx.model.resolved.bindings[*slot], ctx);
                    BINDING_CACHE.with(|c| {
                        let mut c = c.borrow_mut();
                        if c.active && *slot < c.val.len() {
                            c.val[*slot] = v;
                            c.stamp[*slot] = c.gen;
                        }
                    });
                    v
                }
            }
        }
        // gh#272 LICM: a per-eval binding body is param/table/const-only and
        // constant within a θ-stable span. When the caller has staged the
        // prologue (`ctx.per_eval == Some(scratch)` — set once per span by
        // `eval_per_eval_scratch`), read the precomputed value by index; that
        // hoist out of the integration loop is the whole optimization. When no
        // scratch is staged (`None` — every non-LICM eval site, and any span the
        // caller chose not to stage), fall through to on-demand eval,
        // byte-identical to the no-LICM path. Bodies are topologically ordered (a
        // body references only earlier slots), so the fallback recursion
        // terminates; correctness does not depend on staging.
        ResolvedExpr::PerEvalRef(slot) => match ctx.per_eval {
            Some(scratch) => scratch[*slot],
            None => eval_resolved(&ctx.model.resolved.per_eval_bindings[*slot], ctx),
        },
    }
}

/// gh#272 LICM: stage the per-eval prologue for one θ-stable span.
///
/// Evaluate every `model.resolved.per_eval_bindings[i]` once, in topological
/// order, into a scratch `Vec`. Each body is param/table/const-only (the
/// keystone invariant, enforced at `CompiledModel::new`), so its value is
/// constant for the whole span the caller is about to loop over (one trajectory
/// / likelihood eval at a fixed θ). The caller lends the returned slice into
/// every loop iteration via `EvalCtx::per_eval`, turning each `PerEvalRef` into a
/// single array index instead of a re-evaluation of the body.
///
/// Body `i` may reference earlier slots (`< i`); we lend the already-filled
/// prefix `&scratch[..i]`, so those `PerEvalRef(j)` reads hit the staged value
/// rather than recursing. The scratch is owned by the caller and passed as data
/// — there is no shared mutable cache, so nothing can alias across particles or
/// serve a value computed at a different θ.
///
/// `t`/`dt` satisfy the one `EvalCtx` type but cannot be read by a per-eval body
/// (no `Time`/`Dt`/state/forcing/`BindingRef`), so their values do not affect the
/// result.
pub fn eval_per_eval_scratch(
    model: &crate::CompiledModel,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Vec<f64> {
    let bindings = &model.resolved.per_eval_bindings;
    // A per-eval body reads no compartment/forcing state (keystone invariant), so
    // these empty states are never indexed; they exist only to satisfy `EvalCtx`.
    let int_s = crate::state::IntState::new(0);
    let real_s = crate::state::RealState::new(0);
    let mut scratch: Vec<f64> = Vec::with_capacity(bindings.len());
    for i in 0..bindings.len() {
        let v = {
            let ctx = EvalCtx {
                model,
                int_s: &int_s,
                real_s: &real_s,
                params,
                t,
                dt,
                projected: None,
                aux: None,
                int_float_override: None,
                // Lend the prefix already filled, so `PerEvalRef(j<i)` reads the
                // staged value instead of recursing.
                per_eval: Some(&scratch[..i]),
            };
            eval_resolved(&bindings[i], &ctx)
        };
        scratch.push(v);
    }
    scratch
}

/// gh#272 LICM: stage the per-eval prologue for one θ-stable span, or `None` when
/// the model has no per-eval bindings (LICM off, or nothing hoistable). The
/// single seam every backend/inference θ-stable boundary routes through: `Some`
/// is computed once and lent into the span's rate evals; `None` falls through to
/// on-demand eval. `t`/`dt` are inert (a per-eval body reads no `Time`/`Dt`).
#[inline]
pub fn stage_per_eval(
    model: &crate::CompiledModel,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Option<Vec<f64>> {
    if model.resolved.per_eval_bindings.is_empty() {
        None
    } else {
        Some(eval_per_eval_scratch(model, params, t, dt))
    }
}

// ── Resolved derivative entries (obs/σ² gradient carriers) ───────────────────

/// Resolved analogue of [`ir::deriv::DerivEntry`] — one differentiable
/// argument's per-parameter gradient entry, carried into the runtime obs/σ²
/// gradient eval.
///
/// `Grad` holds the resolved `∂arg/∂param` expression, evaluated with the value
/// evaluator [`eval_resolved`] — exactly how `pgas_grad` already consumes a
/// transition's `rate_grad`. `Unsupported` is *carried* (so the P5 fit-time
/// preflight can read its `code` and refuse the fit with that reason) but is
/// never evaluated on a gated path: the preflight rejects any estimated
/// parameter it covers before a gradient is taken (proposal
/// `2026-07-03-unified-obs-gradient-autodiff.md` §4.1, §4.3).
#[derive(Debug, Clone)]
pub enum ResolvedDerivEntry {
    /// A real `∂arg/∂param` expression (resolved for hot-path evaluation).
    Grad(ResolvedExpr),
    /// The derivative could not be emitted; the stable `code` is the refusal
    /// reason the P5 gate surfaces. Never evaluated in a gated fit.
    Unsupported { code: UnsupportedReason },
}

/// One differentiable argument's resolved gradient map: `(model_param_idx,
/// entry)` pairs — the obs/σ² analogue of [`CompiledModel::rate_grads_indexed`].
///
/// Keyed by MODEL parameter index (the eval filters to the run's estimated set
/// via `estimated_to_model`). An **absent** model-param key is a genuine zero
/// (mirrors `rate_grad`: the compiler omits a folded `Const 0.0`).
///
/// [`CompiledModel`]: crate::compiled_model::CompiledModel
pub type ResolvedGradMap = Vec<(usize, ResolvedDerivEntry)>;

/// Resolve an IR `HashMap<String, DerivEntry>` gradient map into a
/// [`ResolvedGradMap`]. Mirrors the `rate_grads_indexed` construction
/// (`compiled_model.rs`): each `Grad` expression is resolved for hot-path eval
/// and its parameter NAME is mapped to a MODEL parameter index; an
/// `Unsupported` entry carries its `code` forward for the P5 gate.
///
/// A key that is not a declared model parameter is a malformed IR — reject it
/// loudly (a dropped gradient component reads as zero to NUTS, silently
/// optimizing a different model than the simulator's likelihood), exactly as the
/// rate path does.
pub(crate) fn resolve_grad_map(
    grad: &std::collections::HashMap<String, DerivEntry>,
    ctx: &ResolveCtx<'_>,
) -> Result<ResolvedGradMap, SimError> {
    let mut out = Vec::with_capacity(grad.len());
    for (name, entry) in grad {
        let model_idx = *ctx.param_index.get(name.as_str()).ok_or_else(|| {
            SimError::Validation(format!(
                "a compiler-emitted gradient (rate/observation/σ²) references unknown \
                 parameter '{}' — every grad key must be a declared model parameter. \
                 A dropped gradient component is silently treated as zero by \
                 gradient-based inference (NUTS), which then optimizes a different \
                 model than the simulator. This is a malformed IR (likely a typo'd \
                 or stale autodiff key).",
                name
            ))
        })?;
        let resolved = match entry {
            DerivEntry::Grad(e) => ResolvedDerivEntry::Grad(resolve_expr(e, ctx)?),
            DerivEntry::Unsupported { code, .. } =>
                ResolvedDerivEntry::Unsupported { code: *code },
        };
        out.push((model_idx, resolved));
    }
    Ok(out)
}

/// A resolved compartment-keyed gradient map — `(compartment_idx,
/// ResolvedDerivEntry)` pairs. The resolved form of `rate_state_grad`
/// (∂rate/∂compartment, gh#275). A **newtype**, not an alias for
/// [`ResolvedGradMap`], because its `usize`s are COMPARTMENT indices, not
/// parameter indices — the two are structurally identical, so a swap into the
/// parameter path would silently mis-index the ODE sensitivity assembly. Keeping
/// them distinct types makes that a compile error.
pub struct ResolvedCompGradMap(pub Vec<(usize, ResolvedDerivEntry)>);

/// Resolve a [`CompGradMap`] (compartment → DerivEntry) into a
/// [`ResolvedCompGradMap`]. Mirrors [`resolve_grad_map`], but resolves each key as
/// a COMPARTMENT name via `ctx.comp_index` — the resolver the `CompGradMap`
/// newtype forces (`ir::deriv`). A key that is not a declared compartment is a
/// malformed IR: reject it loudly, exactly as the parameter path does (a dropped
/// ∂rate/∂compartment component reads as zero to the ODE sensitivity, silently
/// integrating a different `J_x` than the model's dynamics).
pub(crate) fn resolve_comp_grad_map(
    grad: &CompGradMap,
    ctx: &ResolveCtx<'_>,
) -> Result<ResolvedCompGradMap, SimError> {
    let mut out = Vec::with_capacity(grad.0.len());
    for (name, entry) in grad.iter() {
        let comp_idx = *ctx.comp_index.get(name.as_str()).ok_or_else(|| {
            SimError::Validation(format!(
                "rate_state_grad references unknown compartment '{}' — every \
                 ∂rate/∂compartment key must be a declared compartment. A dropped \
                 component is silently treated as zero by the ODE forward \
                 sensitivity, integrating a different J_x than the model's \
                 dynamics. Malformed IR (likely a stale WrtPop autodiff key).",
                name
            ))
        })?;
        let resolved = match entry {
            DerivEntry::Grad(e) => ResolvedDerivEntry::Grad(resolve_expr(e, ctx)?),
            DerivEntry::Unsupported { code, .. } =>
                ResolvedDerivEntry::Unsupported { code: *code },
        };
        out.push((comp_idx, resolved));
    }
    Ok(ResolvedCompGradMap(out))
}

/// The single shared seam that turns a compiler-emitted gradient map into a
/// value: evaluate `∂arg/∂θ` for MODEL parameter index `model_idx`.
///
/// - `Grad(e)`  → `eval_resolved(e, ctx)` (the value evaluator — how `rate_grad`
///   is consumed);
/// - absent key → genuine `0.0` (the parameter does not enter this argument);
/// - `Unsupported` → **unreachable on a gated path**: the P5 fit-time preflight
///   refused any estimated parameter it covers before a gradient was taken. Trip
///   a `debug_assert!` (so a regression surfaces in tests) and fall back to
///   `0.0` in release. P5 makes this unreachable-by-construction.
///
/// The observation likelihood arguments, the σ² overdispersion term, and a
/// future ODE-NUTS `det_grad` all route through THIS function, so the emitted
/// gradient is turned into a number in exactly one place — the obs/σ²/ODE cells
/// cannot fork (proposal §4.3, §10).
#[inline]
pub(crate) fn eval_emitted_grad(
    grad: &[(usize, ResolvedDerivEntry)],
    model_idx: usize,
    ctx: &EvalCtx<'_>,
) -> f64 {
    match grad.iter().find(|(mi, _)| *mi == model_idx) {
        Some((_, entry)) => eval_deriv_entry(entry, ctx),
        None => 0.0,
    }
}

/// Evaluate a single resolved gradient entry to a number. A real `Grad` is the
/// value evaluator; an `Unsupported` is **unreachable on a gated path** (the
/// fit-time preflight refused any estimated parameter it covers before a gradient
/// was taken), so it trips a `debug_assert!` in tests and falls back to `0.0` in
/// release. The rate path iterates its (already est-indexed) entries and calls
/// this directly; the obs/σ² path reaches it via [`eval_emitted_grad`]'s
/// find-by-index — so the `Grad`/`Unsupported` policy lives in exactly one place.
#[inline]
pub(crate) fn eval_deriv_entry(entry: &ResolvedDerivEntry, ctx: &EvalCtx<'_>) -> f64 {
    match entry {
        ResolvedDerivEntry::Grad(e) => eval_resolved(e, ctx),
        ResolvedDerivEntry::Unsupported { code } => {
            debug_assert!(
                false,
                "ungated Unsupported gradient ({code:?}) reached eval — the fit-time \
                 preflight invariant (coeff_guard/P4 for rate, P5 for obs/σ²) was violated"
            );
            0.0
        }
    }
}

// ── Resolved observation likelihood ──────────────────────────────────────────

/// Pre-resolved observation likelihood. All `Expr` fields replaced by
/// `ResolvedExpr`, and each differentiable argument carries its resolved
/// gradient map ([`ResolvedGradMap`], the obs analogue of
/// [`CompiledModel::rate_grads_indexed`]). Constructed at closure-build time,
/// captured by obs closures; evaluates with params at call time.
///
/// The `*_grad` carriers are populated by [`resolve_likelihood`] from the IR
/// `*_grad` maps (compiler-emitted, proposal
/// `2026-07-03-unified-obs-gradient-autodiff.md`) and consumed by
/// `eval_likelihood_resolved_grad` via [`eval_emitted_grad`]. `n`
/// (Binomial/BetaBinomial) carries no gradient — it must be θ-independent.
///
/// [`CompiledModel`]: crate::compiled_model::CompiledModel
#[derive(Debug, Clone)]
pub enum ResolvedLikelihood {
    Poisson { rate: ResolvedExpr, rate_grad: ResolvedGradMap },
    NegBinomial {
        mean: ResolvedExpr, mean_grad: ResolvedGradMap,
        dispersion: ResolvedExpr, dispersion_grad: ResolvedGradMap,
    },
    Normal {
        mean: ResolvedExpr, mean_grad: ResolvedGradMap,
        sd: ResolvedExpr, sd_grad: ResolvedGradMap,
    },
    Binomial { n: ResolvedExpr, p: ResolvedExpr, p_grad: ResolvedGradMap },
    BetaBinomial {
        n: ResolvedExpr,
        alpha: ResolvedExpr, alpha_grad: ResolvedGradMap,
        beta: ResolvedExpr, beta_grad: ResolvedGradMap,
    },
    Bernoulli { p: ResolvedExpr, p_grad: ResolvedGradMap },
}

/// Resolve a `Likelihood` into a `ResolvedLikelihood`, resolving both the
/// argument expressions and their compiler-emitted gradient maps.
pub fn resolve_likelihood(
    lik: &ir::observation::Likelihood,
    ctx: &ResolveCtx<'_>,
) -> Result<ResolvedLikelihood, SimError> {
    use ir::observation::Likelihood;
    match lik {
        Likelihood::Poisson(p) => Ok(ResolvedLikelihood::Poisson {
            rate: resolve_expr(&p.rate.expr, ctx)?,
            rate_grad: resolve_grad_map(&p.rate.grad, ctx)?,
        }),
        Likelihood::NegBinomial(nb) => Ok(ResolvedLikelihood::NegBinomial {
            mean: resolve_expr(&nb.mean.expr, ctx)?,
            mean_grad: resolve_grad_map(&nb.mean.grad, ctx)?,
            dispersion: resolve_expr(&nb.dispersion.expr, ctx)?,
            dispersion_grad: resolve_grad_map(&nb.dispersion.grad, ctx)?,
        }),
        Likelihood::Normal(n) => Ok(ResolvedLikelihood::Normal {
            mean: resolve_expr(&n.mean.expr, ctx)?,
            mean_grad: resolve_grad_map(&n.mean.grad, ctx)?,
            sd: resolve_expr(&n.sd.expr, ctx)?,
            sd_grad: resolve_grad_map(&n.sd.grad, ctx)?,
        }),
        Likelihood::Binomial(b) => Ok(ResolvedLikelihood::Binomial {
            n: resolve_expr(&b.n, ctx)?,
            p: resolve_expr(&b.p.expr, ctx)?,
            p_grad: resolve_grad_map(&b.p.grad, ctx)?,
        }),
        Likelihood::BetaBinomial(bb) => Ok(ResolvedLikelihood::BetaBinomial {
            n: resolve_expr(&bb.n, ctx)?,
            alpha: resolve_expr(&bb.alpha.expr, ctx)?,
            alpha_grad: resolve_grad_map(&bb.alpha.grad, ctx)?,
            beta: resolve_expr(&bb.beta.expr, ctx)?,
            beta_grad: resolve_grad_map(&bb.beta.grad, ctx)?,
        }),
        Likelihood::Bernoulli(b) => Ok(ResolvedLikelihood::Bernoulli {
            p: resolve_expr(&b.p.expr, ctx)?,
            p_grad: resolve_grad_map(&b.p.grad, ctx)?,
        }),
    }
}
