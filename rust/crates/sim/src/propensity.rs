use std::cell::RefCell;

use crate::{
    compiled_model::{CompiledModel, CompiledTimeFuncKind},
    error::{SimError, CollapseKind},
    eval_stats::{allow_degenerate_rates, eval_unresolved},
    flat_eval::{self, FlatCache},
    resolved_expr::{eval_resolved, ResolvedExpr},
    state::{IntState, RealState},
};
use ir::expr::{BinOp, Expr, UnOp};

/// Per-thread scratch + binding cache for the flat-bytecode propensity path
/// (gh#209, opt-in `CAMDL_EVAL_FLAT`). Thread-local for the same reason the
/// `resolved_expr` `BINDING_CACHE` is: PF/PGAS parallelise across particles, so
/// each worker owns its own scratch buffer and cache and there is no
/// cross-particle aliasing. Starts at `FlatCache::new(0)` + an empty `Vec`; both
/// are sized lazily on first use against the active model's binding count.
struct FlatState {
    cache: FlatCache,
    scratch: Vec<f64>,
}

thread_local! {
    static FLAT_STATE: RefCell<FlatState> =
        RefCell::new(FlatState { cache: FlatCache::new(0), scratch: Vec::new() });
}

/// Evaluation context: bundles all read-only simulation state for a single time step.
/// Passed by reference to `eval_expr` and all callers, eliminating the repeated
/// `(model, int_s, real_s, params, t)` parameter list.
pub struct EvalCtx<'a> {
    pub model:  &'a CompiledModel,
    pub int_s:  &'a IntState,
    pub real_s: &'a RealState,
    pub params: &'a [f64],
    pub t:      f64,
    /// Runtime integrator step (gh#54). Read by `Expr::Dt` to expose
    /// the dt the simulator is advancing at — distinct from any
    /// compile-time time literal in the model. Without this, model
    /// authors had to hardcode a fixed time literal in
    /// discretization-correction expressions like
    /// `(1 - exp(-(γ+μ) * 1 'days))`, valid only at dt=1 day. With
    /// `dt` exposed, models match pomp's `(1-exp(-(γ+μ)*dt))/dt`
    /// formulation and stay dt-invariant in effective R0.
    pub dt:     f64,
    /// Projected observation value — only set when evaluating likelihood Exprs.
    /// `Expr::Projected` returns this value; errors if None.
    pub projected: Option<f64>,
    /// Per-observation auxiliary data, keyed by declared column name (a binomial
    /// denominator `n = tested`, a person-time offset). Only set when scoring an
    /// observation likelihood; `Expr::ObsColumnRef` / `ResolvedExpr::ObsColumnRef`
    /// looks its name up here. `None` (or a missing name) outside the likelihood
    /// scoring path — a referenced-but-absent aux is a binder error, not reached
    /// at eval (the cell is then a hole). 2026-06-10 observation data-entry §3.
    pub aux: Option<&'a [(String, f64)]>,
    /// RM8 in 2026-04-19 engine review: the ODE backend uses f64 for
    /// integer compartment state between snapshots. When Some, `Pop`
    /// and `PopSum` read from this slice (indexed by local-int index)
    /// instead of casting int_s.counts[] to f64. Avoids the
    /// per-substep rounding that quantized RK4 integration.
    pub int_float_override: Option<&'a [f64]>,
    /// gh#272 LICM: the per-eval prologue — values of `model.per_eval_bindings`
    /// computed ONCE for this θ-stable span (a whole trajectory / likelihood
    /// eval). `ResolvedExpr::PerEvalRef(slot)` reads `per_eval[slot]` directly.
    /// Owned by whoever holds θ (a backend run / inference particle) and lent in
    /// as data — so there is no shared mutable cache to alias across particles;
    /// the value is structurally bound to the θ it was computed at. `None` ⇒
    /// on-demand eval (byte-identical, just not amortized), so a path that hasn't
    /// staged the prologue is still correct. Sibling of `int_float_override`.
    pub per_eval: Option<&'a [f64]>,
}

/// Evaluate a single expression. No allocations in steady state.
pub fn eval_expr(expr: &Expr, ctx: &EvalCtx<'_>) -> Result<f64, SimError> {
    match expr {
        Expr::Const(c) => Ok(c.value),

        Expr::Param(p) => {
            let idx = ctx.model.param_index.get(p.param.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownParameter(p.param.clone()))?;
            Ok(ctx.params[idx])
        }

        Expr::Pop(p) => {
            let global = ctx.model.comp_index.get(p.pop.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownCompartment(p.pop.clone()))?;
            if let Some(local) = ctx.model.global_to_int[global] {
                let v = match ctx.int_float_override {
                    Some(f) => f[local],
                    None => ctx.int_s.counts[local] as f64,
                };
                Ok(v)
            } else if let Some(local) = ctx.model.global_to_real[global] {
                Ok(ctx.real_s.values[local])
            } else {
                Err(SimError::UnknownCompartment(p.pop.clone()))
            }
        }

        Expr::PopSum(ps) => {
            let mut sum = 0.0;
            for name in &ps.pop_sum {
                let global = ctx.model.comp_index.get(name.as_str())
                    .copied()
                    .ok_or_else(|| SimError::UnknownCompartment(name.clone()))?;
                if let Some(local) = ctx.model.global_to_int[global] {
                    let v = match ctx.int_float_override {
                        Some(f) => f[local],
                        None => ctx.int_s.counts[local] as f64,
                    };
                    sum += v;
                } else if let Some(local) = ctx.model.global_to_real[global] {
                    sum += ctx.real_s.values[local];
                }
            }
            Ok(sum)
        }

        Expr::Time(_) => Ok(ctx.t),

        Expr::Dt(_) => Ok(ctx.dt),

        Expr::BinOp(w) => {
            let a = eval_expr(&w.bin_op.left, ctx)?;
            let b = eval_expr(&w.bin_op.right, ctx)?;
            // gh#audit-C6 / S1. Numerical-collapse paths used to wrap
            // a sentinel 0.0 in Ok(_) — the signature looked like
            // proper error handling but the bodies silently masked
            // failures. Now: increment the EvalStats counter, then
            // either return a typed error (default) or keep the
            // legacy 0.0 if `--allow-degenerate-rates` was set.
            match w.bin_op.op {
                BinOp::Add => Ok(a + b),
                BinOp::Sub => Ok(a - b),
                BinOp::Mul => Ok(a * b),
                BinOp::Div => {
                    if b == 0.0 {
                        crate::eval_stats::inc_div_by_zero();
                        if allow_degenerate_rates() { Ok(0.0) }
                        else { Err(SimError::NumericalCollapse {
                            kind: CollapseKind::DivByZero, t: ctx.t }) }
                    } else { Ok(a / b) }
                }
                BinOp::Pow => {
                    let r = a.powf(b);
                    if r.is_nan() || r.is_infinite() {
                        crate::eval_stats::inc_pow_nan_inf();
                        if allow_degenerate_rates() { Ok(0.0) }
                        else { Err(SimError::NumericalCollapse {
                            kind: CollapseKind::PowNanInf, t: ctx.t }) }
                    } else { Ok(r) }
                }
                BinOp::Mod => {
                    if b == 0.0 {
                        crate::eval_stats::inc_div_by_zero();
                        if allow_degenerate_rates() { Ok(0.0) }
                        else { Err(SimError::NumericalCollapse {
                            kind: CollapseKind::ModByZero, t: ctx.t }) }
                    } else { Ok(a.rem_euclid(b)) }
                }
                BinOp::Min => Ok(a.min(b)),
                BinOp::Max => Ok(a.max(b)),
                BinOp::Eq  => Ok(if a == b { 1.0 } else { 0.0 }),
                BinOp::Neq => Ok(if a != b { 1.0 } else { 0.0 }),
                BinOp::Lt  => Ok(if a <  b { 1.0 } else { 0.0 }),
                BinOp::Gt  => Ok(if a >  b { 1.0 } else { 0.0 }),
                BinOp::Le  => Ok(if a <= b { 1.0 } else { 0.0 }),
                BinOp::Ge  => Ok(if a >= b { 1.0 } else { 0.0 }),
            }
        }

        Expr::UnOp(w) => {
            let a = eval_expr(&w.un_op.arg, ctx)?;
            // Sqrt of negative and log of non-positive are domain errors (no
            // real result), not IEEE-754 NaN/−inf cascades — flag each
            // specifically so the user sees "Sqrt of negative" / "Log of
            // non-positive" instead of a value that silently poisons the rate
            // (a −inf log used to slip past the is_nan guard below entirely).
            if matches!(w.un_op.op, UnOp::Sqrt) && a < 0.0 {
                crate::eval_stats::inc_unop_nan();
                return if allow_degenerate_rates() { Ok(0.0) }
                else { Err(SimError::NumericalCollapse {
                    kind: CollapseKind::SqrtNegative, t: ctx.t }) };
            }
            if matches!(w.un_op.op, UnOp::Log) && a <= 0.0 {
                crate::eval_stats::inc_unop_nan();
                return if allow_degenerate_rates() { Ok(0.0) }
                else { Err(SimError::NumericalCollapse {
                    kind: CollapseKind::LogNonPositive, t: ctx.t }) };
            }
            let result = match w.un_op.op {
                UnOp::Neg   => -a,
                UnOp::Exp   => a.exp(),
                UnOp::Log   => a.ln(),   // a > 0 guaranteed by check above
                UnOp::Sqrt  => a.sqrt(),  // a ≥ 0 guaranteed by check above
                UnOp::Abs   => a.abs(),
                UnOp::Floor => a.floor(),
                UnOp::Ceil  => a.ceil(),
                UnOp::Sin   => a.sin(),
                UnOp::Cos   => a.cos(),
                UnOp::Tanh  => a.tanh(),
            };
            if result.is_nan() {
                crate::eval_stats::inc_unop_nan();
                if allow_degenerate_rates() { Ok(0.0) }
                else { Err(SimError::NumericalCollapse {
                    kind: CollapseKind::UnOpNan, t: ctx.t }) }
            } else {
                Ok(result)
            }
        }

        Expr::Cond(w) => {
            let pred = eval_expr(&w.cond.pred, ctx)?;
            if pred > 0.0 {
                eval_expr(&w.cond.then, ctx)
            } else {
                eval_expr(&w.cond.else_, ctx)
            }
        }

        Expr::TimeFunc(w) => {
            let idx = ctx.model.time_func_index.get(w.time_func.name.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownTimeFunction(w.time_func.name.clone()))?;
            let tf = &ctx.model.time_func_cache[idx];
            // gh#314: a single evaluation-time shift, uniform across every
            // forcing kind. `lag` is already in model time units, so `t − lag`
            // is a direct subtraction. Absent lag ⇒ `ctx.t` unchanged.
            let t_eff = match &tf.lag {
                Some(lag) => ctx.t - eval_resolved(lag, ctx),
                None => ctx.t,
            };
            Ok(eval_forcing(&tf.kind, t_eff, ctx))
        }

        Expr::TableLookup(w) => {
            let idx = ctx.model.table_index.get(w.table_lookup.table.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownTable(w.table_lookup.table.clone()))?;
            let table = &ctx.model.model.tables[idx];
            let cached = &ctx.model.table_values_cache[idx];
            // Only single-index lookups supported (OCaml compiler pre-flattens multi-dim)
            if w.table_lookup.indices.len() != 1 {
                return Err(SimError::TableLookup(format!(
                    "table '{}' requires exactly 1 index, got {}",
                    w.table_lookup.table, w.table_lookup.indices.len()
                )));
            }
            let raw = eval_expr(&w.table_lookup.indices[0], ctx)?;
            let table_idx = raw.floor() as i64;
            table_lookup(table, cached, table_idx, ctx)
        }

        Expr::Projected(_) => {
            ctx.projected.ok_or_else(|| SimError::Validation(
                "Projected expression used outside observation likelihood context".into()
            ))
        }

        Expr::ObsColumnRef(w) => {
            let name = w.obs_column_ref.as_str();
            ctx.aux
                .and_then(|kvs| kvs.iter().find(|(k, _)| k == name).map(|(_, v)| *v))
                .ok_or_else(|| SimError::Validation(format!(
                    "observation aux column '{name}' referenced outside a scored \
                     observation (no per-observation value bound)"
                )))
        }

        Expr::UncheckedDim(w) => {
            // Dimensional escape is a type-level assertion only; at
            // runtime it's identity semantics — evaluate the inner
            // expression and pass its value through unchanged.
            eval_expr(&w.unchecked_dim.inner, ctx)
        }
        Expr::Reduce(w) => {
            // n-ary sum; left-fold (acc starts at 0.0) to match the OCaml
            // `List.fold_left (+)` Add-chain order bit-for-bit.
            let mut acc = 0.0;
            for t in &w.reduce {
                acc += eval_expr(t, ctx)?;
            }
            Ok(acc)
        }
        Expr::BindingRef(w) => {
            // On-demand: find the binding by name, evaluate its body. Slow path
            // (eval_resolved uses the resolved slot); never hit when no model
            // emits bindings (inc1a).
            let b = ctx.model.model.bindings.iter()
                .find(|b| b.name == w.binding_ref)
                .ok_or_else(|| SimError::Validation(
                    format!("reference to unknown binding '{}'", w.binding_ref)))?;
            eval_expr(&b.expr, ctx)
        }
        Expr::PerEvalRef(w) => {
            // gh#272: on-demand by name (the unresolved differential-validation
            // path, CAMDL_EVAL_UNRESOLVED). Mirrors BindingRef.
            let b = ctx.model.model.per_eval_bindings.iter()
                .find(|b| b.name == w.per_eval_ref)
                .ok_or_else(|| SimError::Validation(
                    format!("reference to unknown per-eval binding '{}'", w.per_eval_ref)))?;
            eval_expr(&b.expr, ctx)
        }
    }
}

/// Perform a table lookup using the table's OobPolicy, evaluating the selected
/// value expression (a live `ResolvedExpr`) against `ctx`.
fn table_lookup(
    table: &ir::table::Table,
    cached: &[ResolvedExpr],
    idx: i64,
    ctx: &EvalCtx<'_>,
) -> Result<f64, SimError> {
    use ir::table::OobPolicy;
    let n = cached.len() as i64;
    let i = match table.out_of_bounds {
        // Out-of-range table lookups fail loud; Clamp/Wrap were removed
        // (silent flat-extrapolation/wrapping masks model bugs).
        OobPolicy::Error => {
            if idx < 0 || idx >= n {
                return Err(SimError::TableLookup(format!(
                    "table '{}': index {} out of bounds [0, {})", table.name, idx, n
                )));
            }
            idx
        }
    };
    Ok(eval_resolved(&cached[i as usize], ctx))
}

// ── Pure per-kind forcing math ──────────────────────────────────────────────
//
// Each function takes already-evaluated scalar coefficients (+ structural
// arrays) and `t`. Keeping the closed-form math pure lets the oracle tests
// (interpolation.rs / periodic_forcing.rs / fourier_oracle.rs) exercise it
// directly, and lets `eval_forcing` (below) be a thin coefficient-resolution
// shim. See proposal `2026-06-09-const-parametric-forcing.md` §3.

/// `baseline + amplitude · sin(2π(t − phase)/period)`.
#[inline]
pub fn sinusoidal_value(amplitude: f64, period: f64, phase: f64, baseline: f64, t: f64) -> f64 {
    baseline + amplitude * (2.0 * std::f64::consts::PI * (t - phase) / period).sin()
}

/// Step function: `values[i]` applies for `t ∈ [breakpoints[i-1], breakpoints[i])`;
/// `values[0]` before the first breakpoint, `values[last]` after the last.
#[inline]
pub fn piecewise_value(breakpoints: &[f64], values: &[f64], t: f64) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut result = values[0];
    for (i, &bp) in breakpoints.iter().enumerate() {
        if t >= bp && i + 1 < values.len() {
            result = values[i + 1];
        }
    }
    result
}

/// Linear interpolation between knots; clamps to the endpoint values outside
/// the knot range.
#[inline]
pub fn interpolated_value(times: &[f64], values: &[f64], t: f64) -> f64 {
    // Compiled forcings are guaranteed aligned and non-empty at construction
    // (`CompiledModel::new`, gh#308); this total-on-empty guard only protects
    // the pub helper called directly with arbitrary arrays.
    if times.is_empty() || values.is_empty() { return 0.0; }
    if t <= times[0] { return values[0]; }
    if t >= *times.last().unwrap() { return *values.last().unwrap(); }
    // Binary search for the bracketing interval `[hi-1, hi]` (knots are strictly
    // increasing), mirroring `constant_value`. Both the exact-knot (`Ok`) and
    // strictly-interior (`Err`) cases resolve to the SAME bracket the former
    // linear scan found first, so the lerp arithmetic is bit-for-bit unchanged —
    // just O(log n) instead of O(n). The endpoint guards above ensure
    // `times[0] < t < times.last()`, so `hi ∈ [1, len-1]` and both indices are
    // in range. This turns the per-step forcing lookup from a scan over every
    // knot into a search, the dominant cost on many-knot interpolated forcings.
    let hi = match times.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
        Ok(i) | Err(i) => i,
    };
    let frac = (t - times[hi - 1]) / (times[hi] - times[hi - 1]);
    values[hi - 1] + frac * (values[hi] - values[hi - 1])
}

/// Piecewise-constant lookup: value at the largest grid point ≤ t. Matches
/// pomp's `covariate_table(order = "constant")`.
#[inline]
pub fn constant_value(times: &[f64], values: &[f64], t: f64) -> f64 {
    // See `interpolated_value`: empty/mismatched knots are rejected at
    // construction; this guard only covers direct callers of the pub helper.
    if times.is_empty() || values.is_empty() { return 0.0; }
    if t <= times[0] { return values[0]; }
    if t >= *times.last().unwrap() { return *values.last().unwrap(); }
    match times.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
        Ok(i) => values[i],
        Err(i) => values[i - 1], // insertion point; i-1 is the last point <= t
    }
}

/// Which periodic bin `t` falls in (equal sub-intervals over `period`), or
/// `None` if degenerate (`n == 0` or `period ≤ 0`).
#[inline]
pub fn periodic_bin(period: f64, n: usize, t: f64) -> Option<usize> {
    if n == 0 || period <= 0.0 { return None; }
    let phase = t.rem_euclid(period);
    let step = period / n as f64;
    let i = (phase / step).floor() as usize;
    Some(i.min(n - 1))
}

/// Step value of a periodic forcing at `t`.
#[inline]
pub fn periodic_value(period: f64, values: &[f64], t: f64) -> f64 {
    match periodic_bin(period, values.len(), t) {
        Some(i) => values[i],
        None => 0.0,
    }
}

/// One Fourier term `a·cos(arg) + b·sin(arg)` with `arg = phase·(k+1)` (`k`
/// 0-based; `phase = 2π·t/period`).
#[inline]
pub fn fourier_term(phase: f64, k: usize, a: f64, b: f64) -> f64 {
    let arg = phase * (k as f64 + 1.0);
    a * arg.cos() + b * arg.sin()
}

/// Finite Fourier series `Σ_k a_k cos(2π k t/period) + b_k sin(…)` (gh#59). No
/// baseline — the model author writes `1 + fourier(t)`, matching the sinusoidal
/// convention of leaving baseline composition to the rate.
#[inline]
pub fn fourier_value(period_inv: f64, harmonics: &[(f64, f64)], t: f64) -> f64 {
    let phase = 2.0 * std::f64::consts::PI * t * period_inv;
    harmonics.iter().enumerate()
        .map(|(k, (a, b))| fourier_term(phase, k, *a, *b))
        .sum()
}

/// Evaluate a compiled time function at time `t`, resolving its live scalar
/// coefficients against `ctx.params`. Structural arrays (interpolation knots,
/// spline bases, periodic-spline coefs, piecewise steps) are already `f64` and
/// pass straight through to the pure math above.
///
/// Coefficient resolution is per-call today; the per-`(forcing, t)` memo
/// (proposal §3) collapses the N-referencing-transitions repeat. `Periodic`
/// evaluates only the selected bin's value and `Fourier` evaluates each
/// harmonic coefficient inline, so neither allocates.
pub fn eval_forcing(kind: &CompiledTimeFuncKind, t: f64, ctx: &EvalCtx<'_>) -> f64 {
    match kind {
        CompiledTimeFuncKind::Sinusoidal { amplitude, period, phase, baseline } =>
            sinusoidal_value(
                eval_resolved(amplitude, ctx),
                eval_resolved(period, ctx),
                eval_resolved(phase, ctx),
                eval_resolved(baseline, ctx),
                t),
        CompiledTimeFuncKind::Piecewise { breakpoints, values } =>
            piecewise_value(breakpoints, values, t),
        CompiledTimeFuncKind::Interpolated { times, values } =>
            interpolated_value(times, values, t),
        CompiledTimeFuncKind::Constant { times, values } =>
            constant_value(times, values, t),
        CompiledTimeFuncKind::CubicSpline(spline) => spline.eval(t),
        CompiledTimeFuncKind::Periodic { period, values } => {
            let p = eval_resolved(period, ctx);
            match periodic_bin(p, values.len(), t) {
                Some(i) => eval_resolved(&values[i], ctx),
                None => 0.0,
            }
        }
        CompiledTimeFuncKind::Fourier { period, harmonics } => {
            // `period` is live; `period_inv = 1/period` per evaluation. A
            // non-positive runtime period yields 0 (the build check rejects a
            // non-positive default).
            let p = eval_resolved(period, ctx);
            if p <= 0.0 { return 0.0; }
            let phase = 2.0 * std::f64::consts::PI * t / p;
            harmonics.iter().enumerate()
                .map(|(k, (a, b))| fourier_term(phase, k, eval_resolved(a, ctx), eval_resolved(b, ctx)))
                .sum()
        }
        CompiledTimeFuncKind::PeriodicSpline { period, n_basis, degree, coefs } =>
            crate::periodic_bspline::eval_periodic_bspline(t, *period, *n_basis, *degree, coefs),
    }
}

/// Evaluate all propensities into `out` (cleared and refilled in-place).
/// No allocation if `out` is already the right size.
///
/// Uses pre-resolved expression trees — no HashMap lookups, no Result
/// propagation from expression evaluation. Only error: negative propensity.
pub fn eval_propensities(
    model: &CompiledModel,
    int_s: &IntState,
    real_s: &RealState,
    params: &[f64],
    t: f64,
    dt: f64,
    // gh#272 LICM: the per-eval prologue for this θ-span, staged once by the
    // caller (`eval_per_eval_scratch`) and lent into every rate eval. `None` ⇒
    // on-demand (byte-identical). Forward backends stage it once before their
    // step loop; inference producer steps pass `None` (Phase 2 wires staging).
    per_eval: Option<&[f64]>,
    out: &mut Vec<f64>,
) -> Result<(), SimError> {
    // gh#81 Phase 2. Detect non-finite parameter values BEFORE rate eval
    // runs — they propagate into every rate expression that touches them
    // and surface downstream as a generic NumericalCollapse{DivByZero}
    // that blames the rate expression. The actual fault is upstream:
    // a NUTS leapfrog step or PMMH proposal produced a NaN/Inf param.
    // Naming the offending parameter here turns a misleading message
    // ("DivByZero in rate expression at t=…") into the precise one
    // ("parameter `beta` is non-finite at t=…"), which surfaces the
    // actual proposal-mechanism failure to the user.
    for (name, &idx) in model.param_index.iter() {
        let v = params[idx];
        if !v.is_finite() {
            return Err(SimError::NonFiniteParameter {
                name: name.clone(),
                value: v,
                t,
            });
        }
    }

    let ctx = EvalCtx { model, int_s, real_s, params, t, dt, projected: None, aux: None, int_float_override: None, per_eval };

    // gh#209: flat-bytecode propensity path (opt-in `CAMDL_EVAL_FLAT`). Built
    // once at construction; `Some` iff the toggle is on. The flat VM uses its
    // own `&mut FlatCache` (NOT the resolved_expr `BINDING_CACHE`), so this path
    // does NOT enter `CacheScope`. Per-rate error handling (NaN → table-OOB →
    // SimError::TableLookup or NumericalCollapse; negative-rate guard) is
    // replicated verbatim from the default path below so the two are
    // byte-for-byte identical in every observable outcome.
    if let Some(vm) = &model.resolved.flat_vm {
        return FLAT_STATE.with(|st| {
            let st = &mut *st.borrow_mut();
            // Size the per-thread cache to this model's binding count (rebuild
            // only if it differs — e.g. first use, or a model swap on this
            // thread). Mirrors `CacheScope::enter`'s `val.len() != n` guard.
            if !st.cache.is_sized(vm.n_bindings) {
                st.cache = FlatCache::new(vm.n_bindings);
            }
            // Bump the generation (invalidate the prior state's cached binding
            // values) and mark active — once per propensity-vector eval, exactly
            // like `CacheScope::enter`.
            st.cache.activate();
            // Reserve scratch so the unchecked executor's raw pointer never sees
            // a realloc mid-eval. `scratch_capacity` is the global ceiling.
            let need = flat_eval::scratch_capacity(vm);
            if st.scratch.capacity() < need {
                st.scratch.reserve(need - st.scratch.len());
            }
            out.clear();
            for (i, tr) in model.model.transitions.iter().enumerate() {
                // gh#127 (#12): clear the table-OOB record before EACH rate (see
                // the default path below for the full rationale).
                crate::resolved_expr::clear_table_oob();
                let mut p = flat_eval::eval_flat(vm, &vm.rates[i], &ctx, &mut st.scratch, &mut st.cache);
                // item 17: a non-finite resolved propensity is never a usable
                // rate. NaN is the strict-mode sentinel a degenerate sub-expr
                // leaves (Div0 / Pow-NaN / Sqrt-neg / Log≤0), possibly a
                // table-OOB (attributed first). ±inf is an overflow (e.g. exp)
                // that escaped the per-op guards — a −inf used to be pushed as
                // a NegativePropensity and a +inf used silently as a rate. Under
                // --allow-degenerate-rates all coerce to a 0 rate (byte-
                // identical to the default path below); by default a hard error.
                if !p.is_finite() {
                    if let Some((table_idx, index, len)) = crate::resolved_expr::take_table_oob() {
                        let table_name = model.model.tables[table_idx].name.clone();
                        return Err(SimError::TableLookup(format!(
                            "table '{table_name}': index {index} out of bounds [0, {len}) \
                             while evaluating rate of transition '{}' at t={t} \
                             (the index is computed from model state/parameters; widen the \
                             table or fix the index expression)",
                            tr.name
                        )));
                    }
                    if allow_degenerate_rates() {
                        p = 0.0;
                    } else {
                        return Err(SimError::NumericalCollapse {
                            kind: crate::error::CollapseKind::DivByZero,
                            t,
                        });
                    }
                }
                if p < 0.0 {
                    return Err(SimError::NegativePropensity {
                        transition: tr.name.clone(),
                        value: p,
                        t,
                    });
                }
                out.push(p);
            }
            Ok(())
        });
    }

    // Activate the per-state binding cache for this propensity vector: each
    // model binding is evaluated at most once instead of on every BindingRef
    // (the on-demand path is restored when `_cache` drops at function exit).
    let _cache = crate::resolved_expr::CacheScope::enter(model.resolved.bindings.len());
    // Bench/validation switch (eval_stats::eval_unresolved): off → the
    // pre-resolved index path (default, hot); on → the string-keyed
    // eval_expr path. Read once here and branched per-transition below,
    // so the off path is identical to not having the switch at all.
    let unresolved = eval_unresolved();
    out.clear();
    for (i, tr) in model.model.transitions.iter().enumerate() {
        // gh#127 (#12): clear the table-OOB record before EACH rate so that, if
        // this rate evaluates to NaN below, any record present is attributable
        // to THIS rate only — an OOB recorded on a Cond branch that was then not
        // selected (so the rate is finite) cannot be mis-attributed to a later
        // rate's NaN from an unrelated cause.
        crate::resolved_expr::clear_table_oob();
        let mut p = if unresolved {
            // String-keyed evaluator. Errors (NumericalCollapse) propagate
            // directly; in the non-degenerate case it returns the same
            // value as eval_resolved, so the is_finite/negative guards below
            // and the resulting trajectory are unchanged.
            eval_expr(&tr.rate, &ctx)?
        } else {
            eval_resolved(&model.resolved.rates[i], &ctx)
        };
        // gh#audit-C6 / S1 + item 17. eval_resolved is infallible and signals a
        // degenerate rate out-of-band by a non-finite return: NaN under hard-
        // fail mode for a domain error (Div-by-zero, Pow → NaN, Sqrt of
        // negative, Log of non-positive), or ±inf for an overflow (e.g. exp)
        // that escaped the per-op guards. Neither is a usable rate — a −inf
        // used to be reported as a NegativePropensity and a +inf used silently
        // as a rate. Convert to a typed error so the inference layer's per-
        // particle recovery (PF: line 188+) can decide whether to kill the
        // particle (recoverable) or propagate (forward-sim CLI).
        if !p.is_finite() {
            // gh#127 (#12): a NaN here may be the sentinel an out-of-range table
            // lookup left behind (the infallible fast evaluator records the
            // offending lookup on a thread-local and returns NaN rather than
            // panicking). If so, surface the NAMED, actionable error (table +
            // index + valid range) instead of the generic NumericalCollapse —
            // a controlled per-particle error in inference, a clear diagnostic
            // in forward sim. Take() clears the record either way.
            if let Some((table_idx, index, len)) = crate::resolved_expr::take_table_oob() {
                let table_name = model.model.tables[table_idx].name.clone();
                return Err(SimError::TableLookup(format!(
                    "table '{table_name}': index {index} out of bounds [0, {len}) \
                     while evaluating rate of transition '{}' at t={t} \
                     (the index is computed from model state/parameters; widen the \
                     table or fix the index expression)",
                    tr.name
                )));
            }
            // --allow-degenerate-rates coerces a degenerate/overflowing rate to
            // 0 (its documented "rate legitimately undefined" escape hatch);
            // otherwise a typed hard error.
            if allow_degenerate_rates() {
                p = 0.0;
            } else {
                return Err(SimError::NumericalCollapse {
                    kind: crate::error::CollapseKind::DivByZero, // generic; eval_stats counter has the specific kind
                    t,
                });
            }
        }
        if p < 0.0 {
            return Err(SimError::NegativePropensity {
                transition: tr.name.clone(),
                value: p,
                t,
            });
        }
        out.push(p);
    }
    Ok(())
}

#[cfg(test)]
mod interp_tests {
    use super::interpolated_value;

    /// The former O(n) linear scan, kept here as the reference oracle: the
    /// binary-search rewrite must reproduce it *bit-for-bit* (same bracketing
    /// interval, same lerp operands and order), not merely approximately.
    fn ref_linear_scan(times: &[f64], values: &[f64], t: f64) -> f64 {
        if times.is_empty() || values.is_empty() { return 0.0; }
        if t <= times[0] { return values[0]; }
        if t >= *times.last().unwrap() { return *values.last().unwrap(); }
        for i in 0..times.len() - 1 {
            if t >= times[i] && t <= times[i + 1] {
                let frac = (t - times[i]) / (times[i + 1] - times[i]);
                return values[i] + frac * (values[i + 1] - values[i]);
            }
        }
        *values.last().unwrap()
    }

    #[test]
    fn interpolated_matches_linear_scan_bit_for_bit() {
        // Irregular, strictly-increasing knots with irrational-ish spacing so
        // exact-knot and strictly-interior branches both fire, and so the lerp
        // has real rounding to expose any operand/order divergence.
        let times: Vec<f64> = (0..64).map(|i| (i as f64) * 1.3 + (i as f64).sqrt()).collect();
        let values: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin() * 3.0 + 1.0).collect();

        let mut ts = vec![f64::MIN, -5.0, times[0], *times.last().unwrap(), 1e9];
        for i in 0..times.len() {
            ts.push(times[i]); // exact knot
            if i + 1 < times.len() {
                let lo = times[i];
                let hi = times[i + 1];
                ts.push((lo + hi) * 0.5); // interior
                ts.push(lo + (hi - lo) * 1e-6); // just above a knot
                ts.push(hi - (hi - lo) * 1e-6); // just below the next knot
            }
        }
        for &t in &ts {
            let got = interpolated_value(&times, &values, t);
            let want = ref_linear_scan(&times, &values, t);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "binary-search interpolation diverged from linear scan at t={t}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn interpolated_known_values_and_clamps() {
        let times = [0.0, 1.0, 2.0];
        let values = [10.0, 20.0, 40.0];
        assert_eq!(interpolated_value(&times, &values, 0.5), 15.0); // lerp 10..20
        assert_eq!(interpolated_value(&times, &values, 1.5), 30.0); // lerp 20..40
        assert_eq!(interpolated_value(&times, &values, 1.0), 20.0); // exact interior knot
        assert_eq!(interpolated_value(&times, &values, 0.0), 10.0); // exact first knot
        assert_eq!(interpolated_value(&times, &values, 2.0), 40.0); // exact last knot
        assert_eq!(interpolated_value(&times, &values, -1.0), 10.0); // clamp low
        assert_eq!(interpolated_value(&times, &values, 5.0), 40.0); // clamp high
    }

    #[test]
    fn interpolated_single_and_empty() {
        assert_eq!(interpolated_value(&[], &[], 3.0), 0.0);
        assert_eq!(interpolated_value(&[5.0], &[7.0], 3.0), 7.0); // t below the lone knot
        assert_eq!(interpolated_value(&[5.0], &[7.0], 9.0), 7.0); // t above the lone knot
        assert_eq!(interpolated_value(&[5.0], &[7.0], 5.0), 7.0); // t on the lone knot
    }
}
