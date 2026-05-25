use crate::{
    compiled_model::{CompiledModel, CompiledTimeFuncKind},
    error::{SimError, CollapseKind},
    eval_stats::allow_degenerate_rates,
    resolved_expr::eval_resolved,
    state::{IntState, RealState},
};
use ir::expr::{BinOp, Expr, UnOp};

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
    /// RM8 in 2026-04-19 engine review: the ODE backend uses f64 for
    /// integer compartment state between snapshots. When Some, `Pop`
    /// and `PopSum` read from this slice (indexed by local-int index)
    /// instead of casting int_s.counts[] to f64. Avoids the
    /// per-substep rounding that quantized RK4 integration.
    pub int_float_override: Option<&'a [f64]>,
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
            // Sqrt of negative is a domain error (no real result), not
            // an IEEE-754 NaN cascade — flag it specifically so the
            // user sees "Sqrt of negative" instead of a generic NaN.
            if matches!(w.un_op.op, UnOp::Sqrt) && a < 0.0 {
                crate::eval_stats::inc_unop_nan();
                return if allow_degenerate_rates() { Ok(0.0) }
                else { Err(SimError::NumericalCollapse {
                    kind: CollapseKind::SqrtNegative, t: ctx.t }) };
            }
            let result = match w.un_op.op {
                UnOp::Neg   => -a,
                UnOp::Exp   => a.exp(),
                UnOp::Log   => if a > 0.0 { a.ln() } else { f64::NEG_INFINITY },
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
            Ok(eval_time_func(&ctx.model.time_func_cache[idx].kind, ctx.t))
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
            table_lookup(table, cached, table_idx)
        }

        Expr::Projected(_) => {
            ctx.projected.ok_or_else(|| SimError::Validation(
                "Projected expression used outside observation likelihood context".into()
            ))
        }

        Expr::UncheckedDim(w) => {
            // Dimensional escape is a type-level assertion only; at
            // runtime it's identity semantics — evaluate the inner
            // expression and pass its value through unchanged.
            eval_expr(&w.unchecked_dim.inner, ctx)
        }
    }
}

/// Forward-mode AD: evaluate d(expr)/d(param at index `wrt`).
///
/// Walks the Expr tree applying standard differentiation rules.
/// Pop, PopSum, Time, TimeFunc, TableLookup, Projected have zero derivative
/// (they don't depend on params given fixed state X).
pub fn eval_expr_deriv(expr: &Expr, wrt: usize, ctx: &EvalCtx<'_>) -> f64 {
    match expr {
        Expr::Param(p) => {
            let idx = ctx.model.param_index
                .get(p.param.as_str()).copied().unwrap_or(usize::MAX);
            if idx == wrt { 1.0 } else { 0.0 }
        }
        Expr::Const(_) | Expr::Pop(_) | Expr::PopSum(_)
        | Expr::Time(_) | Expr::Dt(_) | Expr::Projected(_)
        | Expr::TimeFunc(_) | Expr::TableLookup(_) => 0.0,

        Expr::BinOp(w) => {
            let a = eval_expr(&w.bin_op.left, ctx).unwrap_or(0.0);
            let b = eval_expr(&w.bin_op.right, ctx).unwrap_or(0.0);
            let da = eval_expr_deriv(&w.bin_op.left, wrt, ctx);
            let db = eval_expr_deriv(&w.bin_op.right, wrt, ctx);
            match w.bin_op.op {
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

        Expr::UnOp(w) => {
            let a = eval_expr(&w.un_op.arg, ctx).unwrap_or(0.0);
            let da = eval_expr_deriv(&w.un_op.arg, wrt, ctx);
            match w.un_op.op {
                UnOp::Exp => a.exp() * da,
                UnOp::Log => if a > 0.0 { da / a } else { 0.0 },
                UnOp::Neg => -da,
                UnOp::Sqrt => if a > 0.0 { da / (2.0 * a.sqrt()) } else { 0.0 },
                UnOp::Abs => da * a.signum(),
                UnOp::Sin => a.cos() * da,                   // gh#58
                UnOp::Cos => -a.sin() * da,                  // gh#58
                UnOp::Tanh => (1.0 - a.tanh().powi(2)) * da, // gh#58
                UnOp::Floor | UnOp::Ceil => 0.0,
            }
        }

        Expr::Cond(w) => {
            let pred = eval_expr(&w.cond.pred, ctx).unwrap_or(0.0);
            if pred > 0.0 {
                eval_expr_deriv(&w.cond.then, wrt, ctx)
            } else {
                eval_expr_deriv(&w.cond.else_, wrt, ctx)
            }
        }

        Expr::UncheckedDim(w) => {
            // Derivative propagates through the escape — runtime
            // gradients don't care about dim assertions.
            eval_expr_deriv(&w.unchecked_dim.inner, wrt, ctx)
        }
    }
}

/// Perform a table lookup using the table's OobPolicy and pre-evaluated cached values.
fn table_lookup(table: &ir::table::Table, cached: &[f64], idx: i64) -> Result<f64, SimError> {
    use ir::table::OobPolicy;
    let n = cached.len() as i64;
    let i = match table.out_of_bounds {
        OobPolicy::Clamp => idx.clamp(0, n - 1),
        OobPolicy::Wrap  => {
            if n == 0 { return Err(SimError::TableLookup(format!("table '{}' is empty", table.name))); }
            idx.rem_euclid(n)
        }
        OobPolicy::Error => {
            if idx < 0 || idx >= n {
                return Err(SimError::TableLookup(format!(
                    "table '{}': index {} out of bounds [0, {})", table.name, idx, n
                )));
            }
            idx
        }
    };
    Ok(cached[i as usize])
}

/// Evaluate a compiled time function kind at time `t`.
pub fn eval_time_func(kind: &CompiledTimeFuncKind, t: f64) -> f64 {
    match kind {
        CompiledTimeFuncKind::Sinusoidal { amplitude, period, phase, baseline } => {
            baseline + amplitude * (2.0 * std::f64::consts::PI * (t - phase) / period).sin()
        }
        CompiledTimeFuncKind::Piecewise { breakpoints, values } => {
            // Constant on each interval: values[i] applies for t in [breakpoints[i-1], breakpoints[i])
            // values[0] applies before breakpoints[0]; values[last] applies after breakpoints[last-1]
            if values.is_empty() { return 0.0; }
            let mut result = values[0];
            for (i, &bp) in breakpoints.iter().enumerate() {
                if t >= bp && i + 1 < values.len() {
                    result = values[i + 1];
                }
            }
            result
        }
        CompiledTimeFuncKind::Interpolated { times, values } => {
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
        CompiledTimeFuncKind::Constant { times, values } => {
            // Piecewise constant: return value at the largest grid point <= t.
            // Matches pomp's covariate_table(order = "constant").
            if times.is_empty() || values.is_empty() { return 0.0; }
            if t <= times[0] { return values[0]; }
            if t >= *times.last().unwrap() { return *values.last().unwrap(); }
            // Binary search for the last grid point <= t
            match times.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
                Ok(i) => values[i],
                Err(i) => values[i - 1], // i is insertion point; i-1 is last point <= t
            }
        }
        CompiledTimeFuncKind::CubicSpline(spline) => spline.eval(t),
        CompiledTimeFuncKind::Periodic { period, values } => {
            if values.is_empty() || *period <= 0.0 { return 0.0; }
            let phase = t.rem_euclid(*period);
            let n = values.len();
            let step = period / n as f64;
            let i = (phase / step).floor() as usize;
            values[i.min(n - 1)]
        }
        CompiledTimeFuncKind::Fourier { period_inv, harmonics } => {
            // gh#59: sum_k (a_k cos(2π k t/period) + b_k sin(2π k t/period)).
            // No baseline added here — caller is expected to write
            // `1 + fourier(t)` in the rate expression, matching the
            // sinusoidal kind's convention of leaving baseline composition
            // to the model author.
            let phase = 2.0 * std::f64::consts::PI * t * period_inv;
            let mut sum = 0.0;
            for (k, (a, b)) in harmonics.iter().enumerate() {
                let arg = phase * (k as f64 + 1.0);
                sum += a * arg.cos() + b * arg.sin();
            }
            sum
        }
        CompiledTimeFuncKind::PeriodicSpline { period, n_basis, degree, coefs } => {
            // gh#59 v2: de Boor recurrence + periodic wrap-fold + centering
            // shift. See `crates/sim/src/periodic_bspline.rs` for the
            // algorithm and its primary-source citations; cross-validated
            // against scipy and pomp via the oracle fixtures in
            // `tests/fixtures/periodic_bspline_*.tsv`.
            crate::periodic_bspline::eval_periodic_bspline(
                t, *period, *n_basis, *degree, coefs)
        }
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

    let ctx = EvalCtx { model, int_s, real_s, params, t, dt, projected: None, int_float_override: None };
    out.clear();
    for (i, tr) in model.model.transitions.iter().enumerate() {
        let p = eval_resolved(&model.resolved.rates[i], &ctx);
        // gh#audit-C6 / S1. eval_resolved returns NaN under hard-fail
        // mode when a degenerate path was hit (Div-by-zero, Pow → NaN,
        // Sqrt of negative, etc.); convert to typed error so the
        // inference layer's per-particle recovery (PF: line 188+) can
        // decide whether to kill the particle (recoverable) or
        // propagate (forward-sim CLI).
        if p.is_nan() {
            return Err(SimError::NumericalCollapse {
                kind: crate::error::CollapseKind::DivByZero, // generic; eval_stats counter has the specific kind
                t,
            });
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
