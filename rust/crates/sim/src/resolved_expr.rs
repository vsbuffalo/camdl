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
        // gh#272: per-eval bindings are param/table-only by construction (the
        // constructor validation rejects any state reference), so they never
        // reference state. The contrast with BindingRef above is the keystone
        // invariant boundary.
        ResolvedExpr::PerEvalRef(_) => false,
        _ => false,
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

    // gh#272 per-eval tier. Independent of the per-step tier above (own buffers
    // AND own `pe_active`): the per-step `CacheScope` is dropped per RK stage on
    // the ODE path and would otherwise clear a shared `active` mid-trajectory.
    // The generation bumps per `EvalScope` (per theta-stable scope) rather than
    // per step, so a param/table-only `PerEvalRef` is evaluated once per scope.
    pe_val:    Vec<f64>,
    pe_stamp:  Vec<u32>,
    pe_gen:    u32,
    pe_active: bool,
    pe_hits:   u64,
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

// gh#272: per-eval cache tier toggle, the sibling of the per-step machinery
// above. A DISTINCT flag/env/override so the A/B gate can flip the per-eval tier
// independently of the per-step one and prove byte-identity.
thread_local! {
    static PE_CACHE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `CAMDL_NO_PER_EVAL_CACHE` forces the per-eval on-demand path (every
/// `PerEvalRef` re-evaluates its body), making a run comparable to the
/// pre-cache evaluator. The per-thread override (set by the A/B gate) wins.
fn per_eval_cache_disabled() -> bool {
    if let Some(off) = PE_CACHE_OVERRIDE.with(|c| c.get()) {
        return off;
    }
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("CAMDL_NO_PER_EVAL_CACHE").is_some())
}

/// Test/bench hook: force the per-eval cache off/on for the current thread.
pub fn set_per_eval_cache_disabled(off: bool) {
    PE_CACHE_OVERRIDE.with(|c| c.set(Some(off)));
}

/// Test/bench hook: read and reset this thread's cumulative per-eval-cache hit
/// count. A non-vacuity guard distinct from `take_binding_cache_hits` — the
/// per-step counter is non-zero regardless, so the per-eval A/B gate must read
/// THIS one.
pub fn take_per_eval_cache_hits() -> u64 {
    BINDING_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let n = c.pe_hits;
        c.pe_hits = 0;
        n
    })
}

/// gh#272 RAII scope that activates the per-eval cache tier for one theta-stable
/// span (a whole trajectory / likelihood eval, where the parameter vector is
/// fixed). `enter` bumps the per-eval generation (invalidating the prior scope's
/// values) and marks the tier active; `Drop` deactivates it so a `PerEvalRef`
/// evaluated outside any scope falls through to on-demand eval — byte-identical
/// to the no-cache path. Independent of `CacheScope`: nesting a per-step
/// `CacheScope` inside an `EvalScope` (the ODE path does this every RK stage)
/// leaves the per-eval buffers and `pe_active` untouched.
pub struct EvalScope;

impl EvalScope {
    #[inline]
    pub fn enter(n_per_eval: usize) -> Self {
        if !per_eval_cache_disabled() {
            BINDING_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.pe_val.len() != n_per_eval {
                    c.pe_val = vec![0.0; n_per_eval];
                    c.pe_stamp = vec![0; n_per_eval];
                    c.pe_gen = 0;
                }
                c.pe_gen = c.pe_gen.wrapping_add(1);
                if c.pe_gen == 0 {
                    // Generation wrapped: clear stamps so no stale slot aliases gen 0.
                    c.pe_stamp.iter_mut().for_each(|s| *s = 0);
                    c.pe_gen = 1;
                }
                c.pe_active = true;
            });
        }
        EvalScope
    }
}

impl Drop for EvalScope {
    #[inline]
    fn drop(&mut self) {
        if !per_eval_cache_disabled() {
            BINDING_CACHE.with(|c| c.borrow_mut().pe_active = false);
        }
    }
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
            eval_forcing(&ctx.model.time_func_cache[*idx].kind, ctx.t, ctx)
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
        // gh#272 LICM: a per-eval binding body is param/table-only and constant
        // within an `EvalScope` (a theta-stable span). Memoized in the per-eval
        // cache tier, keyed on `pe_gen`; outside a scope (`pe_active == false`)
        // it falls through to on-demand eval — byte-identical to the no-cache
        // path. Bodies are topologically ordered (a body only references earlier
        // per-eval slots), so the miss-path recursion terminates; the borrow is
        // released before recursing (a body may re-enter this arm).
        ResolvedExpr::PerEvalRef(slot) => {
            let hit = BINDING_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.pe_active && *slot < c.pe_stamp.len() && c.pe_stamp[*slot] == c.pe_gen {
                    c.pe_hits = c.pe_hits.wrapping_add(1);
                    Some(c.pe_val[*slot])
                } else {
                    None
                }
            });
            match hit {
                Some(v) => v,
                None => {
                    // Borrow released before recursing — a per-eval body may
                    // reference earlier per-eval slots, re-entering this arm.
                    let v = eval_resolved(&ctx.model.resolved.per_eval_bindings[*slot], ctx);
                    BINDING_CACHE.with(|c| {
                        let mut c = c.borrow_mut();
                        if c.pe_active && *slot < c.pe_val.len() {
                            c.pe_val[*slot] = v;
                            c.pe_stamp[*slot] = c.pe_gen;
                        }
                    });
                    v
                }
            }
        }
    }
}

// ── Resolved observation likelihood ──────────────────────────────────────────

/// Pre-resolved observation likelihood. All `Expr` fields replaced by
/// `ResolvedExpr`. Constructed at closure-build time, captured by obs closures.
#[derive(Debug, Clone)]
pub enum ResolvedLikelihood {
    Poisson { rate: ResolvedExpr },
    NegBinomial { mean: ResolvedExpr, dispersion: ResolvedExpr },
    Normal { mean: ResolvedExpr, sd: ResolvedExpr },
    Binomial { n: ResolvedExpr, p: ResolvedExpr },
    BetaBinomial { n: ResolvedExpr, alpha: ResolvedExpr, beta: ResolvedExpr },
    Bernoulli { p: ResolvedExpr },
}

/// Resolve a `Likelihood` into a `ResolvedLikelihood`.
pub fn resolve_likelihood(
    lik: &ir::observation::Likelihood,
    ctx: &ResolveCtx<'_>,
) -> Result<ResolvedLikelihood, SimError> {
    use ir::observation::Likelihood;
    match lik {
        Likelihood::Poisson(p) => Ok(ResolvedLikelihood::Poisson {
            rate: resolve_expr(&p.rate, ctx)?,
        }),
        Likelihood::NegBinomial(nb) => Ok(ResolvedLikelihood::NegBinomial {
            mean: resolve_expr(&nb.mean, ctx)?,
            dispersion: resolve_expr(&nb.dispersion, ctx)?,
        }),
        Likelihood::Normal(n) => Ok(ResolvedLikelihood::Normal {
            mean: resolve_expr(&n.mean, ctx)?,
            sd: resolve_expr(&n.sd, ctx)?,
        }),
        Likelihood::Binomial(b) => Ok(ResolvedLikelihood::Binomial {
            n: resolve_expr(&b.n, ctx)?,
            p: resolve_expr(&b.p, ctx)?,
        }),
        Likelihood::BetaBinomial(bb) => Ok(ResolvedLikelihood::BetaBinomial {
            n: resolve_expr(&bb.n, ctx)?,
            alpha: resolve_expr(&bb.alpha, ctx)?,
            beta: resolve_expr(&bb.beta, ctx)?,
        }),
        Likelihood::Bernoulli(b) => Ok(ResolvedLikelihood::Bernoulli {
            p: resolve_expr(&b.p, ctx)?,
        }),
    }
}

// ── Forward-mode AD on resolved trees ────────────────────────────────────────

/// Evaluate d(expr)/d(param at index `wrt`) on a pre-resolved tree.
///
/// Mirrors `eval_expr_deriv` but operates on `ResolvedExpr` and is infallible.
/// Pop, PopSum, Time, TimeFunc, TableLookup, Projected have zero derivative
/// (they don't depend on params given fixed state X).
///
/// gh#119: `TimeFunc`/`TableLookup` stay at 0 here on purpose — this is the
/// secondary forward-mode path (obs-likelihood / overdispersion terms). The
/// production dynamics gradient rides the compiler-emitted `rate_grad`, which
/// now carries the analytic ∂forcing/∂coef, so a forcing-coefficient parameter
/// gets its real gradient there, not through this function.
#[inline]
pub fn eval_resolved_deriv(expr: &ResolvedExpr, wrt: usize, ctx: &EvalCtx<'_>) -> f64 {
    match expr {
        ResolvedExpr::Param(idx) => if *idx == wrt { 1.0 } else { 0.0 },

        ResolvedExpr::Const(_)
        | ResolvedExpr::IntPop(_)
        | ResolvedExpr::RealPop(_)
        | ResolvedExpr::IntPopSum(_)
        | ResolvedExpr::MixedPopSum { .. }
        | ResolvedExpr::Time
        | ResolvedExpr::Dt
        | ResolvedExpr::Projected
        | ResolvedExpr::ObsColumnRef(_)
        | ResolvedExpr::TimeFunc(_)
        | ResolvedExpr::TableLookup { .. } => 0.0,

        ResolvedExpr::BinOp { op, left, right } => {
            let a = eval_resolved(left, ctx);
            let b = eval_resolved(right, ctx);
            let da = eval_resolved_deriv(left, wrt, ctx);
            let db = eval_resolved_deriv(right, wrt, ctx);
            match op {
                BinOp::Add => da + db,
                BinOp::Sub => da - db,
                BinOp::Mul => da * b + a * db,
                BinOp::Div => {
                    if b == 0.0 { 0.0 }
                    else { (da * b - a * db) / (b * b) }
                }
                BinOp::Pow => {
                    if a <= 0.0 { 0.0 }
                    else {
                        let val = a.powf(b);
                        val * (b * da / a + a.ln() * db)
                    }
                }
                _ => 0.0, // Mod, comparisons: not differentiable
            }
        }

        ResolvedExpr::UnOp { op, arg } => {
            let a = eval_resolved(arg, ctx);
            let da = eval_resolved_deriv(arg, wrt, ctx);
            match op {
                UnOp::Exp  => a.exp() * da,
                UnOp::Log  => if a > 0.0 { da / a } else { 0.0 },
                UnOp::Neg  => -da,
                UnOp::Sqrt => if a > 0.0 { da / (2.0 * a.sqrt()) } else { 0.0 },
                UnOp::Abs  => da * a.signum(),
                UnOp::Sin  => a.cos() * da,                   // gh#58
                UnOp::Cos  => -a.sin() * da,                  // gh#58
                UnOp::Tanh => (1.0 - a.tanh().powi(2)) * da,  // gh#58
                UnOp::Floor | UnOp::Ceil => 0.0,
            }
        }

        ResolvedExpr::Cond { pred, then_, else_ } => {
            if eval_resolved(pred, ctx) > 0.0 {
                eval_resolved_deriv(then_, wrt, ctx)
            } else {
                eval_resolved_deriv(else_, wrt, ctx)
            }
        }

        ResolvedExpr::UncheckedDim { inner } => {
            eval_resolved_deriv(inner, wrt, ctx)
        }
        ResolvedExpr::Reduce(terms) =>
            terms.iter().map(|t| eval_resolved_deriv(t, wrt, ctx)).sum(),
        // Hoisted bindings are param-free (state-only): d/dp = 0.
        ResolvedExpr::BindingRef(_) => 0.0,
        // gh#272: LICM is scoped to the `eval_resolved` (forward) surfaces, so a
        // PerEvalRef never reaches this secondary forward-mode differentiator (it
        // is param-carrying — a silent 0 would drop a real gradient). The panic
        // enforces the scoping invariant rather than assuming it.
        ResolvedExpr::PerEvalRef(_) =>
            unreachable!("PerEvalRef reached eval_resolved_deriv: LICM scoping invariant violated"),
    }
}
