use std::collections::HashMap;
use std::sync::Arc;
use ir::{Model, model::CompartmentKind};
use ir::expr::{BinOp, Expr, UnOp};
use crate::error::SimError;
use crate::resolved_expr::{ResolvedExpr, ResolveCtx, resolve_expr};
use crate::state::{IntState, RealState};

/// A compiled time function.
///
/// **Scalar coefficients are `ResolvedExpr`, evaluated live against the params
/// slice** — exactly like rates and observation likelihoods. A coefficient that
/// references a parameter (`amplitude = alpha`) is therefore not frozen at
/// construction; there is no `f64` slot for it to freeze into. (Incident
/// `2026-06-09-forcing-coefficient-param-frozen-at-construction.md`; proposal
/// `2026-06-09-const-parametric-forcing.md` §3.) Only `Sinusoidal`, `Periodic`,
/// and `Fourier` carry scalar coefficients.
///
/// **Structural data stays precomputed `f64`.** Interpolation knot arrays, the
/// cubic-spline basis (a construction-time Thomas solve), the periodic-spline
/// coefficients (consumed by the de Boor evaluator), and piecewise step
/// arrays/breakpoints are *data*, not coefficients — a param-referencing entry
/// in any of these is rejected at construction (see the build loop) rather than
/// silently frozen.
#[derive(Debug, Clone)]
pub enum CompiledTimeFuncKind {
    Sinusoidal {
        amplitude: ResolvedExpr,
        period: ResolvedExpr,
        phase: ResolvedExpr,
        baseline: ResolvedExpr,
    },
    Piecewise   { breakpoints: Vec<f64>, values: Vec<f64> },
    Interpolated { times: Vec<f64>, values: Vec<f64> },
    /// Piecewise constant: value holds until the next grid point.
    /// Matches pomp's `covariate_table(order = "constant")`.
    Constant { times: Vec<f64>, values: Vec<f64> },
    CubicSpline(CubicSpline),
    Periodic    { period: ResolvedExpr, values: Vec<ResolvedExpr> },
    /// gh#59: finite Fourier series. `period` is a live coefficient
    /// (`period_inv = 1/period` is computed per evaluation); harmonics is a flat
    /// `[(a_1, b_1), (a_2, b_2), …]` of live coefficient pairs.
    Fourier { period: ResolvedExpr, harmonics: Vec<(ResolvedExpr, ResolvedExpr)> },
    /// gh#59 v2 (2026-05-12): periodic B-spline forcing with uniform
    /// knots and standard de Boor recurrence. Evaluated by
    /// `periodic_bspline::eval_periodic_bspline` — see proposal at
    /// `docs/dev/proposals/2026-05-12-periodic-bspline-algorithm.md`
    /// for algorithm provenance (de Boor 1978 §X, Eilers & Marx
    /// 1996, Wand & Ormerod 2008).
    PeriodicSpline { period: f64, n_basis: u32, degree: u32, coefs: Vec<f64> },
}

/// Natural cubic spline with precomputed coefficients.
/// S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)² + d_i(x - x_i)³
#[derive(Debug, Clone)]
pub struct CubicSpline {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl CubicSpline {
    /// Build a natural cubic spline (second derivative = 0 at endpoints).
    /// Thomas algorithm on the tridiagonal system, O(n).
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 2 && n == ys.len());
        // Validate strictly increasing x-values
        for i in 0..n - 1 {
            assert!(xs[i] < xs[i + 1],
                "CubicSpline: x-values must be strictly increasing, but xs[{}]={} >= xs[{}]={}",
                i, xs[i], i + 1, xs[i + 1]);
        }
        if n == 2 {
            let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
            return CubicSpline {
                xs: xs.to_vec(), ys: ys.to_vec(),
                b: vec![slope, slope], c: vec![0.0, 0.0], d: vec![0.0, 0.0],
            };
        }
        let nm1 = n - 1;
        let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();

        // Build tridiagonal system for c coefficients
        // Equations: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
        //            = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        let mut alpha = vec![0.0; n];
        for i in 1..nm1 {
            alpha[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        // Thomas algorithm: forward sweep
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        for i in 1..nm1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        // Back substitution
        let mut c = vec![0.0; n]; // natural: c[0] = c[n-1] = 0
        for j in (0..nm1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }

        // Compute b, d from c
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        for i in 0..nm1 {
            b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        CubicSpline { xs: xs.to_vec(), ys: ys.to_vec(), b, c, d }
    }

    /// Evaluate the spline at time t. Clamps to boundary values.
    pub fn eval(&self, t: f64) -> f64 {
        let n = self.xs.len();
        if t <= self.xs[0] { return self.ys[0]; }
        if t >= self.xs[n - 1] { return self.ys[n - 1]; }
        // Binary search for segment
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] > t { hi = mid; } else { lo = mid; }
        }
        let dx = t - self.xs[lo];
        self.ys[lo] + self.b[lo] * dx + self.c[lo] * dx * dx + self.d[lo] * dx * dx * dx
    }
}

#[derive(Debug, Clone)]
pub struct CompiledTimeFunc {
    pub kind: CompiledTimeFuncKind,
    /// gh#314: optional evaluation-time shift. When `Some(lag)`, the forcing is
    /// evaluated at `t − eval_resolved(lag)` instead of `t` — a single shift
    /// applied uniformly across every `CompiledTimeFuncKind`. `lag` is a live
    /// coefficient (resolved like a forcing coefficient: a constant, a `Param`,
    /// or arithmetic over them), already in model time units, so the subtraction
    /// is direct. `None` ⇒ no shift.
    pub lag: Option<ResolvedExpr>,
}

/// Recursively collect integer compartment local indices referenced in an expression.
fn collect_int_comp_deps(
    expr: &Expr,
    comp_index: &HashMap<String, usize>,
    global_to_int: &[Option<usize>],
    bindings: &HashMap<&str, &Expr>,
    deps: &mut std::collections::HashSet<usize>,
) {
    match expr {
        Expr::Pop(p) => {
            if let Some(&global) = comp_index.get(p.pop.as_str()) {
                if let Some(local) = global_to_int[global] {
                    deps.insert(local);
                }
            }
        }
        Expr::PopSum(ps) => {
            for name in &ps.pop_sum {
                if let Some(&global) = comp_index.get(name.as_str()) {
                    if let Some(local) = global_to_int[global] {
                        deps.insert(local);
                    }
                }
            }
        }
        Expr::BinOp(w) => {
            collect_int_comp_deps(&w.bin_op.left, comp_index, global_to_int, bindings, deps);
            collect_int_comp_deps(&w.bin_op.right, comp_index, global_to_int, bindings, deps);
        }
        Expr::UnOp(w) => {
            collect_int_comp_deps(&w.un_op.arg, comp_index, global_to_int, bindings, deps);
        }
        Expr::Cond(w) => {
            collect_int_comp_deps(&w.cond.pred, comp_index, global_to_int, bindings, deps);
            collect_int_comp_deps(&w.cond.then, comp_index, global_to_int, bindings, deps);
            collect_int_comp_deps(&w.cond.else_, comp_index, global_to_int, bindings, deps);
        }
        Expr::TableLookup(w) => {
            for idx_expr in &w.table_lookup.indices {
                collect_int_comp_deps(idx_expr, comp_index, global_to_int, bindings, deps);
            }
        }
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect_int_comp_deps(t, comp_index, global_to_int, bindings, deps);
            }
        }
        // Fix B: a BindingRef's compartment dependencies are exactly those of
        // the binding body. Recurse into it so Gillespie's sparse propensity
        // updates recompute this transition whenever any compartment the
        // binding reads changes (else stale propensities → silent wrong
        // dynamics). Bindings are topologically ordered (acyclic), so this
        // terminates.
        Expr::BindingRef(w) => {
            if let Some(body) = bindings.get(w.binding_ref.as_str()) {
                collect_int_comp_deps(body, comp_index, global_to_int, bindings, deps);
            }
        }
        // gh#272: a per-eval body is param/table-only (no compartments), so it
        // contributes no integer-compartment dependencies — no descent needed.
        Expr::PerEvalRef(_) => {}
        // gh#336: `unchecked_dim(inner)` is a transparent dimensional-escape
        // wrapper — its compartment dependencies are exactly those of `inner`.
        // Without this arm a compartment referenced only through
        // `unchecked_dim(...)` fell to `_ => {}` and was omitted from the sparse
        // dependency set, so Gillespie would not recompute the propensity when
        // that compartment changed (stale propensities → silent wrong dynamics).
        Expr::UncheckedDim(w) => {
            collect_int_comp_deps(&w.unchecked_dim.inner, comp_index, global_to_int, bindings, deps);
        }
        // Const, Param, Time, TimeFunc: no compartment dependencies
        _ => {}
    }
}

/// Returns true if the expression's value depends on simulation time `t` —
/// either via a named time function (`TimeFunc`, e.g. a seasonal forcing) or a
/// bare time reference (`Time`, e.g. `t` in `lambda / (1 + exp(-(t-tau)/w))`).
///
/// This drives `time_dep_transitions`, which Gillespie re-evaluates at every
/// output/intervention boundary as time advances (the SSA otherwise freezes a
/// transition's propensity between events). Missing `Expr::Time` here means a
/// rate that depends on bare `t` is frozen at its `t=0` value under Gillespie,
/// silently producing wrong dynamics (the chain-binomial backend
/// re-evaluate every substep regardless, so they are unaffected).
fn expr_is_time_dependent(expr: &Expr, bindings: &HashMap<&str, &Expr>) -> bool {
    match expr {
        Expr::Time(_) | Expr::TimeFunc(_) => true,
        Expr::BinOp(w) => {
            expr_is_time_dependent(&w.bin_op.left, bindings)
                || expr_is_time_dependent(&w.bin_op.right, bindings)
        }
        Expr::UnOp(w) => expr_is_time_dependent(&w.un_op.arg, bindings),
        Expr::Cond(w) => {
            expr_is_time_dependent(&w.cond.pred, bindings)
                || expr_is_time_dependent(&w.cond.then, bindings)
                || expr_is_time_dependent(&w.cond.else_, bindings)
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter()
            .any(|e| expr_is_time_dependent(e, bindings)),
        // A binding/sum that transitively reads a time function must count as
        // time-dependent, or Gillespie freezes it at t=0 (silent wrong dynamics).
        Expr::Reduce(w) => w.reduce.iter().any(|e| expr_is_time_dependent(e, bindings)),
        // Fix B: a BindingRef is time-dependent iff its body is. State-only FOI
        // aggregates (N, I_lga) are NOT time-dependent, so a rate that uses them
        // keeps the exact pre-extraction classification — Gillespie stays
        // byte-identical. A binding that transitively reads a forcing correctly
        // propagates time-dependence. Bindings are acyclic → this terminates.
        Expr::BindingRef(w) => match bindings.get(w.binding_ref.as_str()) {
            Some(body) => expr_is_time_dependent(body, bindings),
            None => false,
        },
        // gh#272: a per-eval body is param/table-only (no Time/Dt/forcing) by the
        // keystone invariant, so it is never time-dependent.
        Expr::PerEvalRef(_) => false,
        // gh#336: `unchecked_dim(inner)` is a transparent dimensional-escape
        // wrapper (identity at runtime) — time-dependent iff `inner` is. Without
        // this arm a forcing wrapped to satisfy the dim-checker (e.g.
        // `unchecked_dim(seasonal(t), …)`) fell to `_ => false` and Gillespie
        // froze the propensity at `t=0` (silent wrong dynamics).
        Expr::UncheckedDim(w) => expr_is_time_dependent(&w.unchecked_dim.inner, bindings),
        _ => false,
    }
}

/// Fix B safety net: true if `expr` directly references an estimated `Param`.
///
/// Shared bindings must be **state-only** (`d(binding)/dp ≡ 0`): `autodiff`
/// maps `BindingRef → 0` and `pgas::collect_param_refs` returns `{}` for a
/// `BindingRef`, both correct *only* under that invariant. A `Param` inside a
/// binding body would silently zero a real gradient. Exhaustively matched (no
/// `_` arm) so a future `Expr` variant forces a decision here rather than
/// defaulting to "no param". Does **not** recurse through `BindingRef`: every
/// binding is checked independently by the caller, so a `Param` reachable only
/// via another binding is caught when that binding is itself checked.
fn expr_refs_param(expr: &Expr) -> bool {
    match expr {
        Expr::Param(_) => true,
        Expr::BinOp(w) => expr_refs_param(&w.bin_op.left) || expr_refs_param(&w.bin_op.right),
        Expr::UnOp(w) => expr_refs_param(&w.un_op.arg),
        Expr::Cond(w) => {
            expr_refs_param(&w.cond.pred)
                || expr_refs_param(&w.cond.then)
                || expr_refs_param(&w.cond.else_)
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter().any(expr_refs_param),
        Expr::Reduce(w) => w.reduce.iter().any(expr_refs_param),
        Expr::UncheckedDim(w) => expr_refs_param(&w.unchecked_dim.inner),
        // Leaves / non-param nodes. BindingRef: not traversed (see doc above).
        // PerEvalRef: likewise not traversed — this net only runs on `model.bindings`
        // (which never contain a PerEvalRef), so the value is moot; grouped with
        // BindingRef for consistency.
        Expr::Const(_)
        | Expr::Pop(_)
        | Expr::PopSum(_)
        | Expr::Time(_)
        | Expr::Dt(_)
        | Expr::TimeFunc(_)
        | Expr::Projected(_)
        | Expr::ObsColumnRef(_)
        | Expr::BindingRef(_)
        | Expr::PerEvalRef(_) => false,
    }
}

/// Evaluate a table value expression using only params (no compartment state).
///
/// This is a construction-time evaluator used before `CompiledModel` is fully
/// built — `eval_expr` cannot be used here because it requires an `EvalCtx`
/// with a completed model. Table value expressions are guaranteed to contain
/// only `Const`, `Param`, `BinOp`, and `UnOp` nodes (no `Pop`, `PopSum`,
/// `Time`, `TimeFunc`, or `TableLookup`). The `BinOp`/`UnOp` arms MUST match
/// the semantics in `eval_expr` — if a new operator is added there, it must
/// be added here too.
fn eval_table_expr(
    expr: &Expr,
    param_index: &HashMap<String, usize>,
    params: &[f64],
) -> Result<f64, SimError> {
    match expr {
        Expr::Const(c) => Ok(c.value),
        Expr::Param(p) => {
            let idx = param_index.get(p.param.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownParameter(p.param.clone()))?;
            Ok(params[idx])
        }
        Expr::BinOp(w) => {
            let a = eval_table_expr(&w.bin_op.left, param_index, params)?;
            let b = eval_table_expr(&w.bin_op.right, param_index, params)?;
            Ok(match w.bin_op.op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => if b == 0.0 { 0.0 } else { a / b },
                BinOp::Pow => {
                    // RM6 in 2026-04-19 engine review: align with
                    // eval_expr / eval_resolved, which both guard
                    // NaN/Inf Pow results. Inline tables with Pow
                    // expressions previously could cache NaN values
                    // that fed the hot path as silent wrong answers.
                    let r = a.powf(b);
                    if r.is_nan() || r.is_infinite() { 0.0 } else { r }
                }
                BinOp::Mod => if b == 0.0 { 0.0 } else { a.rem_euclid(b) },
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                BinOp::Eq  => if a == b { 1.0 } else { 0.0 },
                BinOp::Neq => if a != b { 1.0 } else { 0.0 },
                BinOp::Lt  => if a <  b { 1.0 } else { 0.0 },
                BinOp::Gt  => if a >  b { 1.0 } else { 0.0 },
                BinOp::Le  => if a <= b { 1.0 } else { 0.0 },
                BinOp::Ge  => if a >= b { 1.0 } else { 0.0 },
            })
        }
        Expr::UnOp(w) => {
            let a = eval_table_expr(&w.un_op.arg, param_index, params)?;
            let r = match w.un_op.op {
                UnOp::Neg   => -a,
                UnOp::Exp   => a.exp(),
                UnOp::Log   => if a > 0.0 { a.ln() } else { f64::NEG_INFINITY },
                UnOp::Sqrt  => if a >= 0.0 { a.sqrt() } else { 0.0 },
                UnOp::Abs   => a.abs(),
                UnOp::Floor => a.floor(),
                UnOp::Ceil  => a.ceil(),
                UnOp::Sin   => a.sin(),
                UnOp::Cos   => a.cos(),
                UnOp::Tanh  => a.tanh(),
            };
            // Coerce any non-finite table value to 0: a NaN (sqrt of neg — the
            // arm above), a −inf (log of non-positive), or a ±inf (overflow),
            // matching the Pow arm's `is_infinite` guard. A non-finite constant
            // table cell is never a valid coefficient.
            Ok(if !r.is_finite() { 0.0 } else { r })
        }
        Expr::UncheckedDim(w) => eval_table_expr(&w.unchecked_dim.inner, param_index, params),
        _ => Err(SimError::Validation(
            "unsupported expression type in table values (only Const and Param are valid)".to_string()
        )),
    }
}

/// Resolve a forcing scalar coefficient `Expr` into a live `ResolvedExpr`,
/// preserving the historical coefficient grammar whitelist that
/// `eval_table_expr` enforces: only `Const`, `Param`, `BinOp`, `UnOp`, and
/// `UncheckedDim` are valid. A coefficient that references compartment state,
/// time, another forcing, a table, or a binding is rejected here with a clear
/// error — `resolve_expr` would silently admit those, which the spec never
/// defined for a coefficient.
///
/// Needs only `param_index`, so it runs before the full `ResolveCtx` is built
/// (which depends on `table_meta`, derived after the table loop) — no
/// constructor reorder. This is the "coefficient-only resolve context" of
/// proposal §3, realized as a function.
fn resolve_coeff(
    expr: &Expr,
    param_index: &HashMap<String, usize>,
) -> Result<ResolvedExpr, SimError> {
    match expr {
        Expr::Const(c) => Ok(ResolvedExpr::Const(c.value)),
        Expr::Param(p) => {
            let idx = *param_index.get(p.param.as_str())
                .ok_or_else(|| SimError::UnknownParameter(p.param.clone()))?;
            Ok(ResolvedExpr::Param(idx))
        }
        Expr::BinOp(w) => Ok(ResolvedExpr::BinOp {
            op: w.bin_op.op.clone(),
            left: Box::new(resolve_coeff(&w.bin_op.left, param_index)?),
            right: Box::new(resolve_coeff(&w.bin_op.right, param_index)?),
        }),
        Expr::UnOp(w) => Ok(ResolvedExpr::UnOp {
            op: w.un_op.op.clone(),
            arg: Box::new(resolve_coeff(&w.un_op.arg, param_index)?),
        }),
        Expr::UncheckedDim(w) => Ok(ResolvedExpr::UncheckedDim {
            inner: Box::new(resolve_coeff(&w.unchecked_dim.inner, param_index)?),
        }),
        _ => Err(SimError::Validation(
            "unsupported expression in forcing coefficient (only constants, \
             parameters, and arithmetic over them are allowed — a coefficient \
             cannot reference compartment state, time, another forcing, a table, \
             or a binding)".to_string()
        )),
    }
}

/// Collect parameter names referenced in `expr` (in encounter order, deduped) —
/// used to name the offender in a structural-rejection diagnostic.
fn collect_param_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Param(p) => if !out.iter().any(|n| n == &p.param) { out.push(p.param.clone()); },
        Expr::BinOp(w) => {
            collect_param_names(&w.bin_op.left, out);
            collect_param_names(&w.bin_op.right, out);
        }
        Expr::UnOp(w) => collect_param_names(&w.un_op.arg, out),
        Expr::Cond(w) => {
            collect_param_names(&w.cond.pred, out);
            collect_param_names(&w.cond.then, out);
            collect_param_names(&w.cond.else_, out);
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter().for_each(|e| collect_param_names(e, out)),
        Expr::Reduce(w) => w.reduce.iter().for_each(|e| collect_param_names(e, out)),
        Expr::UncheckedDim(w) => collect_param_names(&w.unchecked_dim.inner, out),
        Expr::Const(_) | Expr::Pop(_) | Expr::PopSum(_) | Expr::Time(_) | Expr::Dt(_)
        | Expr::TimeFunc(_) | Expr::Projected(_) | Expr::ObsColumnRef(_)
        | Expr::BindingRef(_) | Expr::PerEvalRef(_) => {}
    }
}

/// Evaluate a *structural* forcing array element (interpolation knot, spline
/// coefficient, periodic-spline coef, piecewise breakpoint/value) to `f64`.
///
/// Unlike scalar coefficients, these feed a structure derived at construction
/// (a sorted knot table, a Thomas-solved spline basis, the de Boor evaluator),
/// so they cannot vary live. A param-referencing entry is rejected with a clear
/// error rather than silently baked to its default-param value (the freeze).
fn eval_structural(
    forcing: &str,
    what: &str,
    expr: &Expr,
    param_index: &HashMap<String, usize>,
    params: &[f64],
) -> Result<f64, SimError> {
    let mut names = Vec::new();
    collect_param_names(expr, &mut names);
    if !names.is_empty() {
        let plist = names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
        return Err(SimError::Validation(format!(
            "forcing '{forcing}': {what} references parameter {plist}, but it is \
             structural data — {what}s are fixed at construction and cannot be \
             estimated. Use a scalar-coefficient forcing (sinusoidal, periodic, \
             fourier) for an estimated parameter, or make this value constant."
        )));
    }
    eval_table_expr(expr, param_index, params)
}

pub struct CompiledModel {
    pub model: Arc<Model>,

    /// compartment name → index in the *combined* compartment list
    pub comp_index: HashMap<String, usize>,

    /// parameter name → index in the params slice passed to simulate
    pub param_index: HashMap<String, usize>,
    /// Fix B: model-level binding name → slot (index into `resolved.bindings`).
    pub binding_index: HashMap<String, usize>,
    /// gh#272 LICM: per-eval binding name → slot (index into
    /// `resolved.per_eval_bindings`). Empty by default.
    pub per_eval_index: HashMap<String, usize>,

    /// time_function name → index in model.time_functions
    pub time_func_index: HashMap<String, usize>,

    /// table name → index in model.tables
    pub table_index: HashMap<String, usize>,

    /// Indices (in the combined compartment list) of integer compartments,
    /// in model order.
    pub int_comp_indices: Vec<usize>,

    /// Indices (in the combined compartment list) of real compartments,
    /// in model order.
    pub real_comp_indices: Vec<usize>,

    /// For each integer compartment (by its local int-index), its global comp index.
    pub int_local_to_global: Vec<usize>,

    /// For each real compartment (by its local real-index), its global comp index.
    pub real_local_to_global: Vec<usize>,

    /// For a global compartment index: Some(local_int_idx) or None.
    pub global_to_int: Vec<Option<usize>>,

    /// For a global compartment index: Some(local_real_idx) or None.
    pub global_to_real: Vec<Option<usize>>,

    /// Default parameter values extracted from model.parameters, in param_index order.
    pub default_params: Vec<f64>,

    /// For each transition, pre-computed stoichiometry as (int_local_idx, delta).
    /// Real compartments cannot appear in stoichiometry (validator enforces this).
    pub transition_stoich: Vec<Vec<(usize, i64)>>,

    /// For each ODE equation, the local real-compartment index.
    pub ode_real_indices: Vec<usize>,

    /// Per-table value expressions, resolved to live `ResolvedExpr` (not baked
    /// to `f64`) so a param-referencing inline-table value tracks the params
    /// slice — the sibling of the forcing-coefficient fix. Indexed in the same
    /// order as model.tables / table_index; each inner vec is the table's
    /// values, looked up by integer index and evaluated at lookup time.
    pub table_values_cache: Vec<Vec<ResolvedExpr>>,

    /// Per-time-function resolved values (Expr fields evaluated at load time).
    /// Indexed in the same order as model.time_functions / time_func_index.
    pub time_func_cache: Vec<CompiledTimeFunc>,

    /// For each integer compartment (local index), the list of transition indices
    /// whose rate expression references that compartment.
    /// Used for sparse incremental propensity updates after stoichiometry changes.
    pub comp_to_transitions: Vec<Vec<usize>>,

    /// Indices of transitions whose rate expression contains a time function.
    /// These must be re-evaluated whenever simulation time advances.
    pub time_dep_transitions: Vec<usize>,

    /// For chain-binomial multinomial draws: transitions grouped by source
    /// compartment. Key = local int index of source compartment, value = list
    /// of transition indices that draw from it. Transitions with no source
    /// (inflows) are not included — they use Poisson draws directly.
    pub source_groups: Vec<(usize, Vec<usize>)>,

    /// Balance constraint: one compartment is overwritten at each substep
    /// to satisfy a population conservation expression.
    pub balance: Option<ResolvedBalance>,

    /// Continuous fire **times** for each intervention/event, in the
    /// model's `time_unit`. Indexed by intervention position; per-
    /// intervention vector lists every wall time at which that
    /// intervention fires (sorted, dt-invariant).
    ///
    /// Backends derive the runtime view (step indices) by calling
    /// [`CompiledModel::resolve_fire_steps`] with their integrator's
    /// `dt`. This split is load-bearing: prior to gh#53 the
    /// CompiledModel stored fire **steps** baked at compile time
    /// using `model.simulation.dt`, which silently went wrong any
    /// time the runtime integrator's dt differed (every
    /// `camdl pfilter --dt 0.5` against a model declared at
    /// `dt = 1.0`). Storing the dt-invariant times here and
    /// resolving on the simulator side keeps the compile/runtime
    /// seam honest.
    pub fire_times: Vec<Vec<f64>>,

    /// Pre-resolved expression trees for all hot-path evaluations.
    pub resolved: ResolvedModel,
}

/// Pre-resolved balance constraint.
#[derive(Debug, Clone)]
pub struct ResolvedBalance {
    /// Local integer compartment index of the target (e.g., R).
    pub local_int_idx: usize,
    /// Pre-resolved expression (e.g., pop(t) - S - E - I).
    pub expr: ResolvedExpr,
}

/// All pre-resolved expression trees for hot-path evaluation.
/// Populated once during `CompiledModel::new()`, used by all simulation
/// backends and inference algorithms.
pub struct ResolvedModel {
    /// Per-transition resolved rate expression.
    pub rates: Vec<ResolvedExpr>,
    /// Fix B: resolved shared-binding bodies, indexed by slot (matches
    /// `CompiledModel.binding_index`). Evaluated on-demand by `BindingRef`.
    pub bindings: Vec<ResolvedExpr>,
    /// gh#272 LICM: resolved per-eval binding bodies, indexed by slot (matches
    /// `CompiledModel.per_eval_index`). Evaluated on-demand by `PerEvalRef` (with
    /// a per-eval cache tier added in a later increment). Empty by default.
    pub per_eval_bindings: Vec<ResolvedExpr>,
    /// Per-transition resolved overdispersion σ² (None for Poisson/Deterministic).
    pub overdispersion: Vec<Option<ResolvedExpr>>,
    /// Per-transition resolved σ² gradient map (`Some` iff `Overdispersed`, in
    /// lockstep with `overdispersion`). The obs/σ² analogue of
    /// `rate_grads_indexed`: model-param-indexed `∂σ²/∂θ` entries consumed by the
    /// PGAS gamma-density gradient via `eval_emitted_grad` (gh#180). An empty map
    /// means every σ² derivative is a genuine zero.
    pub overdispersion_grad: Vec<Option<crate::resolved_expr::ResolvedGradMap>>,
    /// Per-transition resolved rate gradient map, model-param-indexed
    /// (`ResolvedGradMap`, the obs/σ² analogue). Each entry is a real `Grad`
    /// (resolved for hot-path eval) or a carried `Unsupported` refusal; consumed
    /// by the PGAS transition-density gradient via `eval_deriv_entry`. Built via
    /// the shared `resolve_grad_map`, which rejects an unknown-parameter key
    /// loudly (a dropped gradient reads as zero to NUTS — gh#128).
    pub rate_grads_indexed: Vec<crate::resolved_expr::ResolvedGradMap>,
    /// Per-transition resolved ∂rate/∂compartment map, COMPARTMENT-indexed
    /// (`ResolvedCompGradMap`, the `J_x` ingredient for the ODE forward
    /// sensitivities, gh#275). Empty for every transition until the WrtPop
    /// emission is wired into the compiler; then consumed by `det_grad`'s
    /// augmented-sensitivity integration via `eval_deriv_entry`. Built via the
    /// shared `resolve_comp_grad_map`, which rejects an unknown-compartment key
    /// loudly (a dropped component reads as zero to the sensitivity).
    pub rate_state_grads_indexed: Vec<crate::resolved_expr::ResolvedCompGradMap>,
    /// Per-ODE-equation resolved derivative expression.
    pub ode_derivatives: Vec<ResolvedExpr>,
    /// Per-intervention, per-action resolved expression (count/fraction/value).
    pub intervention_exprs: Vec<Vec<ResolvedExpr>>,
    /// gh#69: per-intervention pre-resolved `at [...]` time expressions.
    /// `Some(exprs)` iff the schedule is `InterventionSchedule::AtTimesExpr`,
    /// `None` otherwise. Evaluated once per simulation start in
    /// `CompiledModel::resolve_fire_steps` against the current `params`.
    pub intervention_at_time_exprs: Vec<Option<Vec<ResolvedExpr>>>,
    /// gh#209: flat-bytecode VM for the propensity hot path, built once at
    /// construction. `Some` iff `CAMDL_EVAL_FLAT` is set (see
    /// `flat_eval::eval_flat_enabled`); `None` by default so default models pay
    /// nothing (neither build cost nor storage) and `eval_propensities` takes
    /// the unchanged `eval_resolved` path.
    pub flat_vm: Option<crate::flat_eval::FlatVm>,
}

/// True if the expression tree references the runtime substep `dt`
/// (`Expr::Dt`) anywhere, INCLUDING transitively through a model-level
/// binding. Used by `required_capabilities` to derive the `RUNTIME_DT`
/// requirement from the rate ASTs (gh#54). `bindings` maps a binding name to
/// its body so a `BindingRef` is followed into `model.bindings` — a param-free
/// `let dtf = dt` is hoisted there by the compiler (Fix-B), so treating
/// `BindingRef` as a leaf would let a `dt`-scaled rate slip past the gate and
/// run silently on Gillespie with a frozen nominal `dt`. This mirrors the
/// sibling Gillespie-classification walkers `collect_int_comp_deps` and
/// `expr_is_time_dependent`, which both recurse through `BindingRef`. Bindings
/// are acyclic (topologically ordered), so the recursion terminates.
fn expr_contains_dt(e: &Expr, bindings: &HashMap<&str, &Expr>) -> bool {
    match e {
        Expr::Dt(_) => true,
        Expr::Const(_)
        | Expr::Param(_)
        | Expr::Pop(_)
        | Expr::PopSum(_)
        | Expr::Time(_)
        | Expr::Projected(_)
        | Expr::ObsColumnRef(_)
        // gh#272: a per-eval body is param/table-only — `per_eval_staging_violation`
        // (enforced at CompiledModel::new) rejects `Dt` in a per-eval body — so
        // this leaf is provably false without descending into the body.
        | Expr::PerEvalRef(_) => false,
        // Fix B: a BindingRef references `dt` iff its body does. Follow it.
        Expr::BindingRef(w) => bindings
            .get(w.binding_ref.as_str())
            .is_some_and(|body| expr_contains_dt(body, bindings)),
        Expr::BinOp(w) => {
            expr_contains_dt(&w.bin_op.left, bindings) || expr_contains_dt(&w.bin_op.right, bindings)
        }
        Expr::UnOp(w) => expr_contains_dt(&w.un_op.arg, bindings),
        Expr::Cond(w) => {
            expr_contains_dt(&w.cond.pred, bindings)
                || expr_contains_dt(&w.cond.then, bindings)
                || expr_contains_dt(&w.cond.else_, bindings)
        }
        // A time_func is a named reference to a model-level forcing; its body
        // is not inline here and cannot itself read the substep `dt`.
        Expr::TimeFunc(_) => false,
        Expr::TableLookup(w) => w.table_lookup.indices.iter().any(|e| expr_contains_dt(e, bindings)),
        Expr::UncheckedDim(w) => expr_contains_dt(&w.unchecked_dim.inner, bindings),
        Expr::Reduce(w) => w.reduce.iter().any(|e| expr_contains_dt(e, bindings)),
    }
}

impl CompiledModel {
    /// Compartment name for a local **integer** state index, for
    /// diagnostics. O(n) reverse walk of `comp_index` → `global_to_int`;
    /// only used on error paths (negative-count detection), never in the
    /// hot loop. Falls back to a synthetic label if the slot has no name
    /// (cannot happen for a well-formed model).
    pub fn int_compartment_name(&self, local: usize) -> String {
        self.comp_index
            .iter()
            .find(|(_, &g)| self.global_to_int.get(g).copied().flatten() == Some(local))
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| format!("(local-int-{local})"))
    }

    /// gh#122: reject a source compartment that mixes a `deterministic(...)`
    /// exit with any other outgoing transition, for the stochastic backends.
    ///
    /// A `deterministic(rate)` transition that is the SOLE exit from its source
    /// is supported (`chain_binomial::step_one` fires `round(rate*dt)` capped by
    /// the source count). But when a source has a deterministic exit AND ≥1
    /// other exit, the chain-binomial competing-risk draw would compute the
    /// deterministic flow and the stochastic flow(s) INDEPENDENTLY — each capped
    /// only by the source count — so together they can exceed the source
    /// population and drive the compartment negative. The correct treatment (a
    /// reserve-off-the-top law: draw the deterministic reserve first, then split
    /// the stochastic remainder over what is left) is not implemented; rather
    /// than silently over-draw, reject the model here with a located error.
    ///
    /// The ODE backend runs every transition as a deterministic flow regardless
    /// of `draw_method`, so the mix is well-defined there — ODE paths do NOT
    /// call this. Gillespie ignores `draw_method` entirely (every transition is
    /// an ordinary CTMC event), so it has neither the freeze bug nor the
    /// over-draw hazard and also does not call this.
    ///
    /// Called at the stochastic dispatch chokepoints: the forward chain-binomial
    /// entry (`run_chain_binomial_with_observer`) and the inference dispatch gate
    /// (`fit::methods::check_model_capabilities` for the chain-binomial producer,
    /// plus the standalone `pfilter` command).
    pub fn validate_deterministic_source_exits(&self) -> Result<(), SimError> {
        use ir::transition::DrawMethod;
        for &(src_local, ref group) in &self.source_groups {
            // Sole-exit (group of one) or no deterministic member ⇒ supported.
            if group.len() < 2 {
                continue;
            }
            let determ: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&tr| matches!(self.model.transitions[tr].draw_method, DrawMethod::Deterministic))
                .collect();
            if determ.is_empty() {
                continue;
            }
            let src = self.int_compartment_name(src_local);
            let determ_names: Vec<&str> =
                determ.iter().map(|&tr| self.model.transitions[tr].name.as_str()).collect();
            let other_names: Vec<&str> = group
                .iter()
                .copied()
                .filter(|tr| !determ.contains(tr))
                .map(|tr| self.model.transitions[tr].name.as_str())
                .collect();
            let other_desc = if other_names.is_empty() {
                "(none — every exit is deterministic; two deterministic exits from \
                 one source still compete for the same pool)"
                    .to_string()
            } else {
                other_names.join(", ")
            };
            return Err(SimError::Validation(format!(
                "compartment '{src}' has {n} competing exit transitions, {k} of them \
                 deterministic ({determ_list}); a `deterministic(...)` exit is only \
                 supported when it is the SOLE exit from its source (gh#122). Mixed \
                 with another exit, the deterministic and stochastic flows are drawn \
                 independently and can together exceed the '{src}' population (driving \
                 it negative). Other exit(s) from '{src}': {other_desc}. Fix: use \
                 `@ rate` (a Poisson competing-risk draw) for every exit from '{src}', \
                 or run this model on the `ode` backend (which treats all transitions \
                 as deterministic flows), or restructure so '{src}' has a single \
                 deterministic exit.",
                n = group.len(),
                k = determ.len(),
                determ_list = determ_names.join(", "),
            )));
        }
        Ok(())
    }

    /// gh#121: reject a transition that draws from **two or more** source
    /// compartments (≥2 negative-stoichiometry entries, e.g. `A + B --> C`) on
    /// the stochastic chain-binomial paths.
    ///
    /// Chain-binomial groups a transition under a SINGLE source compartment
    /// (`source_groups` keys on the FIRST negative-stoich entry) and bounds the
    /// drawn flow by that one source's count. A multi-source transition's flow
    /// is therefore capped by only the first source, then applied to every
    /// source — so a *secondary* source with fewer members than the drawn flow
    /// is driven negative: silently in a mild regime (the secondary source stays
    /// abundant) and, in a harsher regime, as a cryptic runtime
    /// `NegativeCount{cause: BinomialOvershoot}` that never names the real cause.
    /// The correct treatment (a joint draw bounded by `min` over all sources)
    /// is not implemented; rather than over-draw, reject the model up front with
    /// a located error.
    ///
    /// Gillespie applies each firing as one atomic CTMC event that decrements
    /// every source together (bounded by the event's own occurrence), and the
    /// ODE backend runs every transition as a continuous flow — neither has the
    /// single-source-bound hazard, so those paths do NOT call this.
    ///
    /// Scans `self.transition_stoich` DIRECTLY (not `self.source_groups`, which
    /// has already collapsed each transition onto its first source and would
    /// hide the secondary sources this check exists to find). Called at the same
    /// stochastic dispatch chokepoints as
    /// [`Self::validate_deterministic_source_exits`]: the forward chain-binomial
    /// entry, the inference dispatch gate (`check_model_capabilities` for the
    /// chain-binomial producer), and the standalone `pfilter` command.
    pub fn validate_single_source_transitions(&self) -> Result<(), SimError> {
        for (tr_idx, stoich) in self.transition_stoich.iter().enumerate() {
            // gh#121 review: count DISTINCT source compartments, not stoich
            // entries. camdlc always collapses stoichiometry per compartment, but
            // the IR is a public contract (`camdl simulate model.ir.json`) and a
            // hand-authored IR may carry one source as several un-collapsed
            // negative entries (`[["S",-1],["S",-1],…]`); dedup so the same
            // reaction can't get opposite verdicts by representation.
            let mut sources: Vec<usize> = stoich
                .iter()
                .filter(|&&(_, d)| d < 0)
                .map(|&(local, _)| local)
                .collect();
            sources.sort_unstable();
            sources.dedup();
            if sources.len() < 2 {
                continue;
            }
            let name = &self.model.transitions[tr_idx].name;
            let src_names: Vec<String> =
                sources.iter().map(|&local| self.int_compartment_name(local)).collect();
            return Err(SimError::Validation(format!(
                "transition '{name}' draws from {n} source compartments ({src_list}); \
                 multi-source stochastic transitions are not supported on \
                 chain_binomial — the drawn flow is bounded by only the first \
                 source ('{first}'), so the secondary source(s) can be driven \
                 negative (gh#121). Use the `gillespie` or `ode` backend (both \
                 apply the multi-source firing correctly), or restructure into \
                 single-source transitions.",
                n = src_names.len(),
                src_list = src_names.join(" + "),
                first = src_names[0],
            )));
        }
        Ok(())
    }

    /// Resolve per-intervention fire **times** for the current
    /// parameter vector. For constant schedules (`AtTimes`,
    /// `Recurring`) returns the baked `self.fire_times`; for
    /// parametric `AtTimesExpr` schedules (gh#69) evaluates the
    /// resolved expressions against `params`.
    pub fn resolve_fire_times(&self, params: &[f64]) -> Vec<Vec<f64>> {
        use crate::intervention::intervention_fire_times;
        self.model.interventions.iter()
            .enumerate()
            .map(|(iv_idx, iv)| {
                match iv.fire.schedule() {
                    Some(sched @ ir::intervention::InterventionSchedule::AtTimesExpr(_)) => {
                        let resolved = self.resolved.intervention_at_time_exprs[iv_idx]
                            .as_deref();
                        intervention_fire_times(sched, resolved, self, params)
                    }
                    // Constant/recurring schedules use the baked times; reactive
                    // fire sources have no static schedule (empty baked slot).
                    _ => self.fire_times[iv_idx].clone(),
                }
            })
            .collect()
    }

    /// Validate the runtime time-axis at simulation start (gh#126): the
    /// integrator step `dt` must be finite and positive, and every
    /// resolved fire time must be finite. Backends call this ONCE at
    /// their entry point, before `resolve_fire_steps` and the substep
    /// loop.
    ///
    /// This is the RELEASE-build guard: the per-conversion checks in
    /// `crate::time` are `debug_assert!`-only and compiled out of
    /// `--release`, so a bad (or parameter-proposed) `dt`/schedule would
    /// otherwise hang the substep loop (`dt <= 0` never advances time) or
    /// silently fire an intervention at a garbage step (`NaN as i64`).
    /// Returns a named [`SimError::Validation`] instead. See
    /// `docs/dev/notes/2026-06-08-static-typing-as-bug-prevention.md` §6.
    pub fn validate_schedule(&self, dt: f64, params: &[f64]) -> Result<(), SimError> {
        crate::time::validate_dt(dt)?;

        // gh#257: the output-step and recurrence-period positivity guards live
        // in `CompiledModel::new` (the construction boundary), not here — the
        // recurring fire-time loop enumerates at construction, so a non-positive
        // period must be rejected before `new` returns, or it OOMs before any
        // backend calls this. A constructed `CompiledModel` therefore already
        // has a positive output step and positive recurrence periods. What
        // remains runtime-dependent is checked here: the integrator `dt` (a
        // config value) and the finiteness of resolved fire times (parametric
        // `AtTimesExpr` schedules resolve against `params`).
        for (iv_idx, times) in self.resolve_fire_times(params).iter().enumerate() {
            crate::time::validate_fire_times(times)?;
            // item 23: a `Recurring` schedule promises "exactly one fire per
            // period regardless of `dt`" (§13.7). That promise breaks when `dt`
            // is coarser than the period: consecutive targets (one period apart)
            // round to the same integrator step via `round(t/dt)`, and the dedup
            // `BTreeSet` in `resolve_fire_steps` silently drops a fire. `dt` is
            // known here (unlike at construction), so detect the collision and
            // hard-error instead of merging.
            //
            // Scoped to `Recurring` deliberately: an explicit `at [...]` list
            // (`AtTimes`/`AtTimesExpr`) carries NO one-per-period promise, and a
            // within-`dt` coincidence of two listed fires MERGES to one fire on
            // purpose, for cross-backend agreement (gh#198). Only the recurring
            // guarantee is at stake here.
            if !matches!(
                self.model.interventions[iv_idx].fire.schedule(),
                Some(ir::intervention::InterventionSchedule::Recurring(_))
            ) {
                continue;
            }
            let mut step_of: std::collections::BTreeMap<i64, f64> =
                std::collections::BTreeMap::new();
            for &t in times {
                let step = crate::time::time_to_step(t, dt);
                match step_of.get(&step) {
                    // Same step already claimed by an EARLIER, DISTINCT fire
                    // time — the two would merge to one fire. (An exact
                    // duplicate `t` is the same fire declared twice, not a
                    // dropped fire, so it is allowed.)
                    Some(&prev) if prev != t => {
                        return Err(SimError::Validation(format!(
                            "intervention '{}': fire times {prev} and {t} both round to \
                             integrator step {step} at dt={dt}, so one fire per period would be \
                             silently dropped (§13.7 guarantees exactly one fire per period). \
                             Use a dt no coarser than the period, or widen the period.",
                            self.model.interventions[iv_idx].name,
                        )));
                    }
                    _ => {
                        step_of.insert(step, t);
                    }
                }
            }
        }
        Ok(())
    }

    /// Derive per-intervention sets of step indices for a given
    /// integrator step `dt` and parameter vector. The returned view is
    /// a runtime projection of `self.resolve_fire_times(params)`;
    /// backends call this once at sim start with `cfg.dt` and `params`
    /// and use the local result for the duration of the run. See
    /// `crate::time::time_to_step` for the rounding semantics, and
    /// gh#53 for the architectural motivation (don't bake dt-dependent
    /// indices on the dt-invariant CompiledModel). gh#69 added the
    /// `params` arg so parametric `at [...]` schedules can be resolved
    /// against the run's parameter vector instead of silently firing
    /// at t=0.
    pub fn resolve_fire_steps(
        &self,
        dt: f64,
        params: &[f64],
    ) -> Vec<std::collections::BTreeSet<i64>> {
        self.resolve_fire_times(params).iter()
            .map(|times| crate::time::fire_times_to_steps(times, dt))
            .collect()
    }

    pub fn new(model: Model) -> Result<Self, SimError> {
        // gh#257: the output schedule's `t += step` and each recurring
        // intervention's `t += period` are infinite-loop hazards with the same
        // shape as a non-positive integrator `dt` — a non-positive step/period
        // never advances the loop cursor. But unlike `dt` (a runtime config
        // value checked at each backend's entry), the recurring fire-time loop
        // below (`Recurring` arm of the `fire_times` enumeration) runs HERE, at
        // construction — so a non-positive period does not merely hang, it
        // `push`es to a `Vec` unbounded and exhausts memory before any backend
        // guard could run. Validate at the construction boundary so a bad model
        // is a controlled setup error, not an OOM: once a `CompiledModel`
        // exists, its schedule is provably safe to enumerate. `dt` and resolved
        // fire-time finiteness stay in `validate_schedule` (they depend on the
        // runtime `dt` / parameter vector, unknown here).
        if let ir::model::OutputSchedule::Regular(reg) = &model.output.times {
            crate::time::validate_output_step(reg.step)?;
        }
        for iv in &model.interventions {
            if let Some(ir::intervention::InterventionSchedule::Recurring(rs)) = iv.fire.schedule() {
                crate::time::validate_recurrence_period(rs.period)?;
            }
        }

        let n_comps = model.compartments.len();

        let mut comp_index = HashMap::with_capacity(n_comps);
        let mut int_local_to_global = Vec::new();
        let mut real_local_to_global = Vec::new();
        let mut global_to_int = vec![None; n_comps];
        let mut global_to_real = vec![None; n_comps];
        let mut int_comp_indices = Vec::new();
        let mut real_comp_indices = Vec::new();

        for (global, comp) in model.compartments.iter().enumerate() {
            comp_index.insert(comp.name.clone(), global);
            match comp.kind {
                CompartmentKind::Integer => {
                    let local = int_local_to_global.len();
                    int_local_to_global.push(global);
                    global_to_int[global] = Some(local);
                    int_comp_indices.push(global);
                }
                CompartmentKind::Real => {
                    let local = real_local_to_global.len();
                    real_local_to_global.push(global);
                    global_to_real[global] = Some(local);
                    real_comp_indices.push(global);
                }
            }
        }

        let mut param_index = HashMap::with_capacity(model.parameters.len());
        let mut default_params = Vec::with_capacity(model.parameters.len());
        for (i, p) in model.parameters.iter().enumerate() {
            param_index.insert(p.name.clone(), i);
            // Only `Fixed` carries a concrete value; `Estimated`/`Required`
            // resolve at runtime (override / inference start). "Has no value"
            // is reachable here exactly when a parameter is still unresolved —
            // the gh#191 conflation is gone, but the demand is unchanged for a
            // concrete forward run.
            let v = p.value.resolved_value().ok_or_else(|| SimError::Validation(
                format!("parameter '{}' has no value; supply it via --params or --param", p.name)
            ))?;
            default_params.push(v);
        }

        let mut time_func_index = HashMap::with_capacity(model.time_functions.len());
        for (i, tf) in model.time_functions.iter().enumerate() {
            time_func_index.insert(tf.name.clone(), i);
        }

        let mut table_index = HashMap::with_capacity(model.tables.len());
        for (i, t) in model.tables.iter().enumerate() {
            table_index.insert(t.name.clone(), i);
        }

        // Pre-compute stoichiometry for integer compartments only.
        // Real compartments cannot appear in stoichiometry (IR validator enforces this).
        let mut transition_stoich = Vec::with_capacity(model.transitions.len());
        for t in &model.transitions {
            let mut stoich = Vec::new();
            for entry in &t.stoichiometry {
                let comp_name = &entry.0;
                let delta = entry.1;
                let global = comp_index.get(comp_name.as_str())
                    .copied()
                    .ok_or_else(|| SimError::UnknownCompartment(comp_name.clone()))?;
                if let Some(local) = global_to_int[global] {
                    stoich.push((local, delta));
                } else if global_to_real[global].is_some() {
                    // Real compartments cannot appear in stoichiometry
                    return Err(SimError::Validation(format!(
                        "real compartment '{}' cannot appear in stoichiometry", comp_name
                    )));
                }
            }
            transition_stoich.push(stoich);
        }

        // Build dependency graph for sparse propensity updates.
        // comp_to_transitions[local_int_idx] = [transition indices that reference it]
        // time_dep_transitions = [transition indices whose rate depends on time
        // `t` — via a TimeFunc forcing or a bare `Time` reference]
        let n_int_comps = int_local_to_global.len();
        let mut comp_to_transitions: Vec<Vec<usize>> = vec![vec![]; n_int_comps];
        let mut time_dep_transitions: Vec<usize> = Vec::new();
        // Fix B: binding name → body, so collect_int_comp_deps can see through
        // a BindingRef to the compartments the binding reads.
        let binding_bodies: HashMap<&str, &Expr> = model.bindings.iter()
            .map(|b| (b.name.as_str(), &b.expr)).collect();
        // Fix B safety net: bindings must be state-only (no estimated Param) —
        // the invariant autodiff (`BindingRef → 0`) and `collect_param_refs`
        // (`{}`) silently rely on. The OCaml extraction guard enforces it for
        // compiler output; reject a hand-written/future IR that smuggles a
        // Param into a binding body rather than zeroing its gradient in silence.
        for b in &model.bindings {
            if expr_refs_param(&b.expr) {
                return Err(SimError::Validation(format!(
                    "binding '{}' references a parameter: bindings must be state-only \
                     (d(binding)/dp must be 0, else the gradient is silently wrong). \
                     Inline this expression at its use sites instead of hoisting it.",
                    b.name
                )));
            }
        }
        for (tr_idx, tr) in model.transitions.iter().enumerate() {
            let mut deps = std::collections::HashSet::new();
            collect_int_comp_deps(&tr.rate, &comp_index, &global_to_int, &binding_bodies, &mut deps);
            for local_idx in deps {
                comp_to_transitions[local_idx].push(tr_idx);
            }
            if expr_is_time_dependent(&tr.rate, &binding_bodies) {
                time_dep_transitions.push(tr_idx);
            }
        }

        // Group transitions by source compartment for multinomial draws.
        //
        // Iteration order of `source_groups` drives RNG consumption in the
        // chain-binomial/PGAS/PMMH paths. HashMap::into_iter() is
        // nondeterministic, so we sort by src_local after collecting — same
        // seed + same model must always produce the same trajectory.
        let source_groups: Vec<(usize, Vec<usize>)> = {
            let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (tr_idx, stoich) in transition_stoich.iter().enumerate() {
                if let Some(&(src_local, _)) = stoich.iter().find(|&&(_, d)| d < 0) {
                    groups.entry(src_local).or_default().push(tr_idx);
                }
            }
            let mut out: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
            out.sort_by_key(|(src, _)| *src);
            out
        };

        // Pre-compute ODE equation → real local index
        let mut ode_real_indices = Vec::with_capacity(model.ode_equations.len());
        for eq in &model.ode_equations {
            let global = comp_index.get(eq.compartment.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownCompartment(eq.compartment.clone()))?;
            let local = global_to_real[global]
                .ok_or_else(|| SimError::Validation(
                    format!("ODE equation references non-real compartment '{}'", eq.compartment)
                ))?;
            ode_real_indices.push(local);
        }

        // Resolve table value expressions to live `ResolvedExpr` (evaluated at
        // lookup time against the params slice, not baked at load).
        // External tables (TableSource::External) are left empty here; the CLI
        // fills them in before calling CompiledModel::new() via --table flags.
        let mut table_values_cache: Vec<Vec<ResolvedExpr>> = Vec::with_capacity(model.tables.len());
        for table in &model.tables {
            match &table.source {
                ir::table::TableSource::Inline { values } => {
                    let vals: Result<Vec<ResolvedExpr>, SimError> = values.iter()
                        .map(|expr| resolve_coeff(expr, &param_index))
                        .collect();
                    table_values_cache.push(vals?);
                }
                ir::table::TableSource::External { external } => {
                    // Rm5 in 2026-04-19 engine review: the CLI is
                    // responsible for replacing External with Inline
                    // before calling CompiledModel::new — unreplaced
                    // externals caused a panic when the empty cached
                    // vec was indexed during propensity eval. Fail
                    // loud at construction instead.
                    return Err(SimError::Validation(format!(
                        "table '{}' is declared external() but was not replaced \
                         before CompiledModel::new; populate TableSource::Inline \
                         from the runtime input first",
                        external
                    )));
                }
            }
        }

        // Build each time function. Scalar coefficients (Sinusoidal / Periodic /
        // Fourier) resolve to live `ResolvedExpr`, evaluated per-step against the
        // params slice; structural arrays (interpolation knots, spline bases,
        // periodic-spline coefs, piecewise steps) stay precomputed `f64` and
        // reject param references. Build-time numeric checks that depend on a
        // now-live coefficient evaluate it at `default_params` for the early
        // diagnostic (the live path guards the runtime value).
        let mut time_func_cache: Vec<CompiledTimeFunc> = Vec::with_capacity(model.time_functions.len());
        for tf in &model.time_functions {
            use ir::time_func::TimeFuncKind;
            let kind = match &tf.kind {
                TimeFuncKind::Sinusoidal(s) => CompiledTimeFuncKind::Sinusoidal {
                    amplitude: resolve_coeff(&s.amplitude, &param_index)?,
                    period:    resolve_coeff(&s.period,    &param_index)?,
                    phase:     resolve_coeff(&s.phase,     &param_index)?,
                    baseline:  resolve_coeff(&s.baseline,  &param_index)?,
                },
                TimeFuncKind::Piecewise(p) => {
                    let bps: Result<Vec<f64>, SimError> = p.breakpoints.iter()
                        .map(|e| eval_structural(&tf.name, "piecewise breakpoint", e, &param_index, &default_params))
                        .collect();
                    let vals: Result<Vec<f64>, SimError> = p.values.iter()
                        .map(|e| eval_structural(&tf.name, "piecewise value", e, &param_index, &default_params))
                        .collect();
                    CompiledTimeFuncKind::Piecewise { breakpoints: bps?, values: vals? }
                }
                TimeFuncKind::Interpolated(i) => {
                    let times: Result<Vec<f64>, SimError> = i.times.iter()
                        .map(|e| eval_structural(&tf.name, "interpolation knot time", e, &param_index, &default_params))
                        .collect();
                    let vals: Result<Vec<f64>, SimError> = i.values.iter()
                        .map(|e| eval_structural(&tf.name, "interpolation knot value", e, &param_index, &default_params))
                        .collect();
                    let ts = times?;
                    let vs = vals?;
                    // The knot arrays must be aligned and non-empty: a length
                    // mismatch mis-pairs times with values, and zero knots makes
                    // every lookup return 0 — silently zeroing any rate that
                    // multiplies through the forcing (gh#308). Every IR producer
                    // funnels through here, so this is the one place the
                    // invariant is enforced for all of them.
                    if ts.len() != vs.len() {
                        return Err(SimError::Validation(format!(
                            "interpolated forcing '{}': {} knot times but {} values \
                             (the time and value columns must have equal length)",
                            tf.name, ts.len(), vs.len())));
                    }
                    if ts.is_empty() {
                        return Err(SimError::Validation(format!(
                            "interpolated forcing '{}': no knots (need at least one \
                             time/value pair to interpolate)",
                            tf.name)));
                    }
                    // Interpolation requires strictly-increasing knot times:
                    // `interpolated_value` indexes by position and `constant_value`
                    // binary-searches, so an out-of-order time axis silently returns
                    // wrong values. The OCaml producers emit knots in file/dimension
                    // order without sorting, so reject a non-monotone axis here — the
                    // one place every IR producer funnels through (gh#345).
                    for w in ts.windows(2) {
                        if !(w[0] < w[1]) {
                            return Err(SimError::Validation(format!(
                                "interpolated forcing '{}': knot times must be strictly \
                                 increasing, but {} is followed by {} — sort the \
                                 forcing's time axis (its data rows or its time dimension's \
                                 levels) into increasing time order",
                                tf.name, w[0], w[1])));
                        }
                    }
                    match i.method {
                        ir::time_func::InterpMethod::Spline =>
                            CompiledTimeFuncKind::CubicSpline(CubicSpline::new(&ts, &vs)),
                        ir::time_func::InterpMethod::Linear =>
                            CompiledTimeFuncKind::Interpolated { times: ts, values: vs },
                        ir::time_func::InterpMethod::Constant =>
                            CompiledTimeFuncKind::Constant { times: ts, values: vs },
                    }
                }
                TimeFuncKind::Periodic(p) => {
                    let period = resolve_coeff(&p.period, &param_index)?;
                    let vals: Result<Vec<ResolvedExpr>, SimError> = p.values.iter()
                        .map(|e| resolve_coeff(e, &param_index))
                        .collect();
                    CompiledTimeFuncKind::Periodic { period, values: vals? }
                }
                TimeFuncKind::Fourier(f) => {
                    // `period <= 0` early check at default params (the live path
                    // guards a non-positive runtime value by returning 0).
                    let period_at_default = eval_table_expr(&f.period, &param_index, &default_params)?;
                    if period_at_default <= 0.0 {
                        return Err(SimError::Validation(format!(
                            "fourier forcing period must be positive, got {}", period_at_default)));
                    }
                    let harmonics: Result<Vec<(ResolvedExpr, ResolvedExpr)>, SimError> = f.harmonics.iter()
                        .map(|(a, b)| Ok((
                            resolve_coeff(a, &param_index)?,
                            resolve_coeff(b, &param_index)?,
                        )))
                        .collect();
                    CompiledTimeFuncKind::Fourier {
                        period: resolve_coeff(&f.period, &param_index)?,
                        harmonics: harmonics?,
                    }
                }
                TimeFuncKind::PeriodicSpline(ps) => {
                    let period = eval_structural(&tf.name, "periodic_spline period", &ps.period, &param_index, &default_params)?;
                    if period <= 0.0 {
                        return Err(SimError::Validation(format!(
                            "periodic_spline period must be positive, got {}", period)));
                    }
                    if ps.n_basis <= ps.degree {
                        return Err(SimError::Validation(format!(
                            "periodic_spline: n_basis ({}) must exceed degree ({})",
                            ps.n_basis, ps.degree)));
                    }
                    let coefs: Result<Vec<f64>, SimError> = ps.coefs.iter()
                        .map(|e| eval_structural(&tf.name, "periodic_spline coefficient", e, &param_index, &default_params))
                        .collect();
                    let cs = coefs?;
                    if cs.len() != ps.n_basis as usize {
                        return Err(SimError::Validation(format!(
                            "periodic_spline: coefs length ({}) must equal n_basis ({})",
                            cs.len(), ps.n_basis)));
                    }
                    CompiledTimeFuncKind::PeriodicSpline {
                        period,
                        n_basis: ps.n_basis,
                        degree: ps.degree,
                        coefs: cs,
                    }
                }
            };
            // gh#314: resolve the optional lag into a live coefficient. It uses
            // the same `resolve_coeff` path as Sinusoidal/Periodic/Fourier
            // coefficients — a constant, a `Param`, or arithmetic over them —
            // so a lag-as-parameter is resolved to a `ResolvedExpr::Param` and
            // evaluated per-call. Structural data (interpolation knots, piecewise
            // grids) is never a valid lag, so `resolve_coeff` is the right (and
            // only) seam.
            let lag = match &tf.lag {
                None => None,
                Some(e) => Some(resolve_coeff(e, &param_index)?),
            };
            time_func_cache.push(CompiledTimeFunc { kind, lag });
        }

        // Precompute fire **times** (continuous, dt-invariant) for all
        // interventions/events with constant schedules. The runtime view
        // (step indices) is derived per-simulation via
        // `CompiledModel::resolve_fire_steps` — see the field docstring
        // and gh#53 for why this split is load-bearing.
        //
        // gh#69: parametric `AtTimesExpr` schedules cannot be evaluated
        // here (we don't have `params` yet). Their slot is left empty
        // and the per-run resolver fills it from
        // `resolved.intervention_at_time_exprs`.
        let fire_times: Vec<Vec<f64>> = model.interventions.iter()
            .map(|iv| match iv.fire.schedule() {
                // Reactive fire sources have no static schedule — their fire
                // times are discovered at runtime (and the capability gate
                // rejects reactive models before any backend runs).
                None => Vec::new(),
                Some(ir::intervention::InterventionSchedule::AtTimes(ts)) => ts.clone(),
                Some(ir::intervention::InterventionSchedule::AtTimesExpr(_)) => Vec::new(),
                Some(ir::intervention::InterventionSchedule::Recurring(rs)) => {
                    let mut times = Vec::new();
                    if let Some(at_day) = rs.at_day {
                        let k0 = ((rs.start - at_day) / rs.period).ceil().max(0.0) as u64;
                        let mut t = at_day + k0 as f64 * rs.period;
                        while t <= rs.end + rs.period * 1e-9 {
                            times.push(t);
                            t += rs.period;
                        }
                    } else {
                        let mut t = rs.start;
                        while t <= rs.end + rs.period * 1e-9 {
                            times.push(t);
                            t += rs.period;
                        }
                    }
                    times
                }
            })
            .collect();

        // ── Pre-resolve all expression trees ─────────────────────────────
        // Build ResolveCtx from the index maps we just constructed.
        let table_meta: Vec<(ir::table::OobPolicy, usize)> = model.tables.iter()
            .zip(&table_values_cache)
            .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
            .collect();

        // Fix B: binding name -> slot (index into resolved.bindings).
        let binding_index: HashMap<String, usize> = model.bindings.iter()
            .enumerate().map(|(i, b)| (b.name.clone(), i)).collect();

        // gh#272 LICM: per-eval binding name -> slot. Assert-unique on insert (a
        // duplicate name would mis-resolve a self-reference); LICM mints
        // collision-proof names, so a duplicate here is a compiler bug.
        let mut per_eval_index: HashMap<String, usize> =
            HashMap::with_capacity(model.per_eval_bindings.len());
        for (i, b) in model.per_eval_bindings.iter().enumerate() {
            if per_eval_index.insert(b.name.clone(), i).is_some() {
                return Err(SimError::Validation(format!(
                    "duplicate per-eval binding name '{}' (gh#272 LICM invariant)", b.name)));
            }
        }

        let resolve_ctx = ResolveCtx {
            comp_index: &comp_index,
            param_index: &param_index,
            time_func_index: &time_func_index,
            table_index: &table_index,
            global_to_int: &global_to_int,
            global_to_real: &global_to_real,
            table_meta: &table_meta,
            binding_index: &binding_index,
            per_eval_index: &per_eval_index,
        };

        // Resolve balance constraint
        let balance = if let Some(ref bs) = model.balance {
            let global = *comp_index.get(bs.target.as_str())
                .ok_or_else(|| SimError::UnknownCompartment(bs.target.clone()))?;
            let local = global_to_int[global]
                .ok_or_else(|| SimError::Validation(
                    format!("balance target '{}' must be an integer compartment", bs.target)
                ))?;
            let resolved_expr = resolve_expr(&bs.expr, &resolve_ctx)?;
            Some(ResolvedBalance { local_int_idx: local, expr: resolved_expr })
        } else {
            None
        };

        // Resolve transition rates + overdispersion + rate_grad
        let rates: Vec<ResolvedExpr> = model.transitions.iter()
            .map(|tr| resolve_expr(&tr.rate, &resolve_ctx))
            .collect::<Result<_, _>>()?;

        // Fix B: resolve shared-binding bodies (slot order matches binding_index).
        let resolved_bindings: Vec<ResolvedExpr> = model.bindings.iter()
            .map(|b| resolve_expr(&b.expr, &resolve_ctx))
            .collect::<Result<_, _>>()?;

        // gh#272 LICM: resolve per-eval binding bodies (slot order matches
        // per_eval_index). Param/table-only and topologically ordered.
        let resolved_per_eval_bindings: Vec<ResolvedExpr> = model.per_eval_bindings.iter()
            .map(|b| resolve_expr(&b.expr, &resolve_ctx))
            .collect::<Result<_, _>>()?;

        // gh#284: enforce the LICM per-eval staging contract here, not just in
        // the OCaml pass. `stage_per_eval` evaluates each body ONCE at `t_start`
        // against a zero scratch, lends body `i` only the prefix `&scratch[..i]`,
        // and reads the result every substep. So each body must be (a)
        // loop-invariant — a body referencing compartment state would panic on
        // the zero scratch (`IntState::new(0)` index-OOB) and a time-varying one
        // would be staged stale (silent-wrong) — AND (b) topologically ordered:
        // a forward/self `PerEvalRef(j >= i)` reads an unfilled scratch slot. The
        // OCaml LICM pass never emits such a body (`licm.ml is_invariant`, and it
        // produces no inter-binding references), but a hand-edited or
        // future-emitted IR could; reject it with a located error. Mirrors the
        // overdispersion σ² and intervention-schedule `references_state` guards
        // below, with the stronger `per_eval_staging_violation`.
        for (i, (b, rb)) in model.per_eval_bindings.iter()
            .zip(&resolved_per_eval_bindings).enumerate()
        {
            if let Some(kind) = crate::resolved_expr::per_eval_staging_violation(rb, i) {
                return Err(SimError::Validation(format!(
                    "per-eval binding '{}': body references {}, which breaks the \
                     loop-invariance the LICM staging relies on. A per-eval \
                     binding must be a function of parameters, tables, constants, \
                     and earlier per-eval bindings only. (These bindings are \
                     compiler-generated; a hand-edited IR is the usual cause of \
                     this error.)",
                    b.name, kind
                )));
            }
        }

        let overdispersion: Vec<Option<ResolvedExpr>> = model.transitions.iter()
            .map(|tr| match &tr.draw_method {
                ir::transition::DrawMethod::Overdispersed { sigma_sq, .. } =>
                    resolve_expr(sigma_sq, &resolve_ctx).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<_, _>>()?;

        // gh#180: resolve each transition's compiler-emitted `∂σ²/∂θ` map into the
        // model-param-indexed carrier the PGAS gamma-density gradient consumes.
        // In lockstep with `overdispersion`: `Some` iff `Overdispersed`.
        let overdispersion_grad: Vec<Option<crate::resolved_expr::ResolvedGradMap>> =
            model.transitions.iter()
                .map(|tr| match &tr.draw_method {
                    ir::transition::DrawMethod::Overdispersed { sigma_sq_grad, .. } =>
                        crate::resolved_expr::resolve_grad_map(sigma_sq_grad, &resolve_ctx)
                            .map(Some),
                    _ => Ok(None),
                })
                .collect::<Result<_, _>>()?;

        // Enforce the "overdispersion σ² is state-independent" invariant
        // that CPM (`correlated_pf.rs`) and PGAS gamma-density eval
        // (`pgas.rs:528`) assume. If σ² references compartment state,
        // those sites would silently evaluate against a zero scratch.
        // Reject at compile time rather than produce a wrong likelihood.
        // Incident: 2026-04-22-observation-sampler-scratch-state.md.
        for (tr, od) in model.transitions.iter().zip(&overdispersion) {
            if let Some(od_expr) = od {
                if crate::resolved_expr::references_state(od_expr) {
                    return Err(SimError::Validation(format!(
                        "transition '{}': overdispersion σ² expression \
                         references compartment state (Pop / PopSum), which \
                         is not supported. σ² must be a function of \
                         parameters, time, and constants only. If you need \
                         state-dependent overdispersion, open an issue — it \
                         requires reworking three independent eval sites \
                         that currently assume σ² is state-independent.",
                        tr.name
                    )));
                }
            }
        }

        // Resolve each transition's rate gradient map to its model-param-indexed
        // form via the shared `resolve_grad_map` — the same seam the obs/σ² grads
        // use, so rate now resolves (and, below, evaluates) through one path, not
        // a fork. It resolves each `Grad` expression, maps the parameter NAME to a
        // model index, carries an `Unsupported` refusal forward for the fit-time
        // gate, and rejects an unknown-parameter key loudly (gh#128: a dropped
        // gradient reads as zero to NUTS, silently optimizing a different model).
        let rate_grads_indexed: Vec<crate::resolved_expr::ResolvedGradMap> =
            model.transitions.iter()
                .map(|tr| crate::resolved_expr::resolve_grad_map(&tr.rate_grad, &resolve_ctx)
                    // Re-add the transition-name context the shared (obs-neutral)
                    // resolver cannot know, so an unknown rate_grad key names its
                    // transition (error-quality; unknown_rate_grad_key_is_rejected).
                    .map_err(|e| SimError::Validation(
                        format!("transition '{}' rate_grad: {}", tr.name, e))))
                .collect::<Result<_, _>>()?;

        // Resolve each transition's ∂rate/∂compartment map to its
        // compartment-indexed form via `resolve_comp_grad_map` (gh#275). Empty for
        // every transition until the WrtPop emission is wired; a state-dependent
        // rate then carries `J_x`'s ingredient for the ODE forward sensitivities.
        // The `CompGradMap` newtype forces the compartment resolver here (not the
        // parameter one); an unknown-compartment key is rejected loudly.
        let rate_state_grads_indexed: Vec<crate::resolved_expr::ResolvedCompGradMap> =
            model.transitions.iter()
                .map(|tr| crate::resolved_expr::resolve_comp_grad_map(&tr.rate_state_grad, &resolve_ctx)
                    .map_err(|e| SimError::Validation(
                        format!("transition '{}' rate_state_grad: {}", tr.name, e))))
                .collect::<Result<_, _>>()?;

        // Resolve ODE derivatives
        let ode_derivatives: Vec<ResolvedExpr> = model.ode_equations.iter()
            .map(|eq| resolve_expr(&eq.derivative, &resolve_ctx))
            .collect::<Result<_, _>>()?;

        // Resolve intervention action expressions
        let intervention_exprs: Vec<Vec<ResolvedExpr>> = model.interventions.iter()
            .map(|iv| {
                iv.actions.iter().map(|action| {
                    let expr = match action {
                        ir::intervention::Action::Add(a) => &a.count,
                        ir::intervention::Action::Set(s) => &s.value,
                        ir::intervention::Action::FractionTransfer(ft) => &ft.fraction,
                        ir::intervention::Action::AbsoluteTransfer(at) => &at.count,
                    };
                    resolve_expr(expr, &resolve_ctx)
                }).collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        // gh#69: resolve parametric `at [...]` time expressions for each
        // intervention/event with an `AtTimesExpr` schedule. We also
        // reject any expression that references compartment state
        // (`Pop`/`PopSum`) or projected output — these would silently
        // evaluate against a zero scratch state in
        // `intervention_fire_times`, which is wrong. Time and dt are
        // similarly meaningless at schedule-resolution time, but the IR
        // expander never emits them in this position; we leave them
        // allowed so future use-cases (e.g. `t_seed = t_start + 5`) can
        // be supported without revisiting the validation.
        let intervention_at_time_exprs: Vec<Option<Vec<ResolvedExpr>>> = model.interventions.iter()
            .enumerate()
            .map(|(idx, iv)| match iv.fire.schedule() {
                Some(ir::intervention::InterventionSchedule::AtTimesExpr(exprs)) => {
                    let resolved: Vec<ResolvedExpr> = exprs.iter()
                        .map(|e| resolve_expr(e, &resolve_ctx))
                        .collect::<Result<_, _>>()?;
                    for r in &resolved {
                        if crate::resolved_expr::references_state(r) {
                            return Err(SimError::Validation(format!(
                                "intervention '{}': `at [...]` schedule \
                                 expression references compartment state \
                                 (Pop / PopSum), which is not supported. \
                                 Schedule times must be a function of \
                                 parameters and constants only.",
                                model.interventions[idx].name
                            )));
                        }
                    }
                    Ok(Some(resolved))
                }
                _ => Ok(None),
            })
            .collect::<Result<_, SimError>>()?;

        // gh#209: build the flat-bytecode VM once, only when the toggle is on,
        // so default models pay nothing. Mirrors `cm.resolved.{rates,bindings}`.
        // gh#272: the flat VM's per-eval tape is deferred (step 1.4), so skip the
        // flat path for per-eval models — they fall back to `eval_resolved`, which
        // handles `PerEvalRef`. (Default-off LICM ⇒ this is never hit today.)
        let flat_vm = if crate::flat_eval::eval_flat_enabled()
            && model.per_eval_bindings.is_empty()
        {
            Some(crate::flat_eval::build(&rates, &resolved_bindings))
        } else {
            None
        };

        let resolved = ResolvedModel {
            rates,
            bindings: resolved_bindings,
            per_eval_bindings: resolved_per_eval_bindings,
            overdispersion,
            overdispersion_grad,
            rate_grads_indexed,
            rate_state_grads_indexed,
            ode_derivatives,
            intervention_exprs,
            intervention_at_time_exprs,
            flat_vm,
        };

        Ok(CompiledModel {
            model: Arc::new(model),
            comp_index,
            param_index,
            binding_index,
            per_eval_index,
            time_func_index,
            table_index,
            int_comp_indices,
            real_comp_indices,
            int_local_to_global,
            real_local_to_global,
            global_to_int,
            global_to_real,
            default_params,
            transition_stoich,
            ode_real_indices,
            table_values_cache,
            time_func_cache,
            source_groups,
            comp_to_transitions,
            time_dep_transitions,
            balance,
            fire_times,
            resolved,
        })
    }

    /// Features this model requires from a backend.
    pub fn required_capabilities(&self) -> crate::Capabilities {
        let mut caps = crate::Capabilities::empty();
        if self.model.transitions.iter().any(|t| matches!(t.draw_method, ir::transition::DrawMethod::Overdispersed { .. })) {
            caps |= crate::Capabilities::OVERDISPERSION;
        }
        if !self.real_comp_indices.is_empty() {
            caps |= crate::Capabilities::REAL_COMPARTMENTS;
        }
        if self.balance.is_some() {
            // gh#audit-C3. balance{} is chain-binomial-only; declaring
            // the requirement makes other backends fail dispatch
            // rather than silently drop it.
            caps |= crate::Capabilities::BALANCE;
        }
        // gh#54. A rate (or its gradient, evaluated in PGAS) that references
        // the runtime substep `dt` (`Expr::Dt`) is only meaningful on a
        // backend that realizes a substep length. Gillespie freezes it to
        // the nominal `simulation.dt`-or-`1.0`, so it would silently produce
        // a different trajectory. Walk the rate ASTs — following `BindingRef`
        // into `model.bindings`, since a param-free `let dtf = dt` is hoisted
        // there — and if any contains `Expr::Dt`, require RUNTIME_DT so
        // gillespie fails dispatch.
        let binding_bodies: HashMap<&str, &Expr> = self.model.bindings.iter()
            .map(|b| (b.name.as_str(), &b.expr)).collect();
        let uses_dt = self.model.transitions.iter().any(|t| {
            expr_contains_dt(&t.rate, &binding_bodies)
                || t.rate_grad.values().any(|de| matches!(de,
                    ir::deriv::DerivEntry::Grad(e) if expr_contains_dt(e, &binding_bodies)))
        });
        if uses_dt {
            caps |= crate::Capabilities::RUNTIME_DT;
        }
        // gh#204. A reactive fire source is parsed and represented in the IR
        // but executed by no backend yet — raise the requirement so dispatch
        // rejects it (no backend grants it) rather than silently dropping the
        // policy.
        if self.model.interventions.iter().any(|iv| iv.fire.is_reactive()) {
            caps |= crate::Capabilities::REACTIVE_INTERVENTIONS;
        }
        caps
    }

    /// Build the initial state from model.initial_conditions + params.
    pub fn initial_state(
        &self,
        params: &[f64],
    ) -> Result<(IntState, RealState), SimError> {
        use ir::model::InitialConditions;
        use crate::propensity::{eval_expr, EvalCtx};

        let n_int = self.int_local_to_global.len();
        let n_real = self.real_local_to_global.len();
        let mut int_counts = vec![0i64; n_int];
        let mut real_values = vec![0.0f64; n_real];

        // Temporary zero state for evaluating parameterized ICs
        let zero_int = IntState::new(n_int);
        let zero_real = RealState::new(n_real);

        match &self.model.initial_conditions {
            InitialConditions::Explicit(map) => {
                for (name, val) in map {
                    let global = self.comp_index.get(name.as_str())
                        .copied()
                        .ok_or_else(|| SimError::UnknownCompartment(name.clone()))?;
                    if let Some(local) = self.global_to_int[global] {
                        int_counts[local] = *val as i64;
                    } else if let Some(local) = self.global_to_real[global] {
                        real_values[local] = *val;
                    }
                }
            }
            InitialConditions::Parameterized(map) => {
                // dt: 0.0 — initial-condition expressions don't have
                // access to a meaningful integrator step (init runs once,
                // before stepping). Users referencing `dt` here get 0.0.
                let ctx = EvalCtx { model: self, int_s: &zero_int, real_s: &zero_real, params, t: 0.0, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None };
                for (name, expr) in map {
                    let global = self.comp_index.get(name.as_str())
                        .copied()
                        .ok_or_else(|| SimError::UnknownCompartment(name.clone()))?;
                    let v = eval_expr(expr, &ctx)?;
                    if let Some(local) = self.global_to_int[global] {
                        int_counts[local] = v.round() as i64;
                    } else if let Some(local) = self.global_to_real[global] {
                        real_values[local] = v;
                    }
                }
            }
            InitialConditions::FromDistribution(_) => {
                // RC3 in 2026-04-19 engine review: this was a silent
                // fall-through to "all zeros," which would start every
                // compartment at 0 and not tell anyone. Hard-fail until
                // the inference-side prior sampling path is wired in.
                return Err(SimError::Validation(
                    "initial_conditions::from_distribution is not yet \
                     supported at the sim layer; draw initial values \
                     via the inference pipeline and pass them in as \
                     explicit initial_conditions instead".to_string()
                ));
            }
        }

        Ok((IntState::from_vec(int_counts), RealState::from_vec(real_values)))
    }

    /// Continuous initial compartment values for the ODE **gradient** path
    /// (gh#275 §1c): the un-rounded initial state, returned as
    /// `(int_as_f64, real)`.
    ///
    /// [`Self::initial_state`] rounds/truncates integer compartments to `i64`,
    /// correct for the discrete backends. The deterministic ODE forward
    /// sensitivity must instead start from the *continuous* initial value:
    /// rounding a `Parameterized` initial condition to an integer makes the
    /// likelihood piecewise-constant in the IC parameter, which contradicts the
    /// `∂init/∂θ` seed (`ic_grad`) and would make the reported gradient
    /// inconsistent with the value it differentiates (an FD of the rounded value
    /// is ~0 or a boundary spike, never the analytic seed). For `Explicit`
    /// (constant) ICs this returns exactly `initial_state`'s values; the two
    /// paths diverge only for a `Parameterized` IC that evaluates to a
    /// non-integer, where the continuous value is the correct one for the ODE
    /// skeleton. The eval context mirrors `initial_state`'s parameterized arm
    /// (zero state, `t = 0`, `dt = 0`), so the two stay in lockstep.
    pub fn initial_state_continuous(
        &self,
        params: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), SimError> {
        use ir::model::InitialConditions;
        use crate::propensity::{eval_expr, EvalCtx};

        let n_int = self.int_local_to_global.len();
        let n_real = self.real_local_to_global.len();
        let mut int_values = vec![0.0f64; n_int];
        let mut real_values = vec![0.0f64; n_real];
        let zero_int = IntState::new(n_int);
        let zero_real = RealState::new(n_real);

        // Place a continuous initial value into the int- or real-compartment slot
        // (unrounded, unlike `initial_state`'s `as i64` / `.round()`).
        macro_rules! place {
            ($name:expr, $v:expr) => {{
                let global = self.comp_index.get($name.as_str())
                    .copied()
                    .ok_or_else(|| SimError::UnknownCompartment($name.clone()))?;
                if let Some(local) = self.global_to_int[global] {
                    int_values[local] = $v;
                } else if let Some(local) = self.global_to_real[global] {
                    real_values[local] = $v;
                }
            }};
        }

        match &self.model.initial_conditions {
            InitialConditions::Explicit(map) => {
                for (name, val) in map {
                    place!(name, *val);
                }
            }
            InitialConditions::Parameterized(map) => {
                let ctx = EvalCtx { model: self, int_s: &zero_int, real_s: &zero_real, params, t: 0.0, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None };
                for (name, expr) in map {
                    let v = eval_expr(expr, &ctx)?;
                    place!(name, v);
                }
            }
            InitialConditions::FromDistribution(_) => {
                return Err(SimError::Validation(
                    "initial_conditions::from_distribution is not yet supported at the \
                     sim layer; draw initial values via the inference pipeline and pass \
                     them in as explicit initial_conditions instead".to_string()
                ));
            }
        }
        Ok((int_values, real_values))
    }

    /// The forward-sensitivity seed `S(t_start) = ∂(initial_state)/∂θ` (`ic_grad`,
    /// gh#275 §1c C-seed), laid out as the `n_int × d` row-major block the ODE
    /// gradient path expects (`state_sens_0`), where `d = estimated.len()` and the
    /// column order matches `estimated_to_model`.
    ///
    /// Reads the compiler-emitted [`ir::Model::ic_grad`] (`compartment → param →
    /// ∂init/∂param`). Zero everywhere for an `Explicit` (constant) IC or a
    /// gradient-free build (empty `ic_grad`). A `Parameterized` IC contributes
    /// `seed[comp_local·d + pos] = ∂(initial comp)/∂param` for each estimated
    /// `param`; a **fixed** parameter in an IC expression contributes no column
    /// (it has no sensitivity to seed). The compartment key is validated against
    /// the int-compartment index — a real compartment (whose ODE-equation
    /// sensitivity is a separate follow-up) or an unknown name is a hard error,
    /// not a silently-dropped seed (schema-review flag B2).
    pub fn ic_grad_seed(
        &self,
        params: &[f64],
        estimated_to_model: &[usize],
    ) -> Result<Vec<f64>, SimError> {
        use crate::propensity::{eval_expr, EvalCtx};
        use ir::deriv::DerivEntry;

        let d = estimated_to_model.len();
        let n_int = self.int_local_to_global.len();
        let mut seed = vec![0.0f64; n_int * d];
        if self.model.ic_grad.is_empty() {
            return Ok(seed); // Explicit IC or no emission → ∂init/∂θ ≡ 0.
        }

        // Estimated-parameter name → seed column (position in `estimated_to_model`).
        let est_pos: std::collections::HashMap<&str, usize> = estimated_to_model
            .iter()
            .enumerate()
            .map(|(pos, &midx)| (self.model.parameters[midx].name.as_str(), pos))
            .collect();

        let zero_int = IntState::new(n_int);
        let zero_real = RealState::new(self.real_local_to_global.len());
        let ctx = EvalCtx { model: self, int_s: &zero_int, real_s: &zero_real, params, t: 0.0, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None };

        for (comp_name, param_map) in &self.model.ic_grad {
            let global = self.comp_index.get(comp_name.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownCompartment(comp_name.clone()))?;
            let comp_local = self.global_to_int[global].ok_or_else(|| {
                SimError::Validation(format!(
                    "ODE gradient (nuts): parameterized initial condition on real \
                     compartment `{comp_name}` — real-compartment forward sensitivity is \
                     not yet supported (gh#275 §1c). Use gradient-free `mh` on `ode`."
                ))
            })?;
            for (param_name, entry) in param_map {
                // A fixed parameter in an IC expression has no seed column.
                let Some(&pos) = est_pos.get(param_name.as_str()) else { continue };
                let v = match entry {
                    DerivEntry::Grad(expr) => eval_expr(expr, &ctx)?,
                    DerivEntry::Unsupported { code, .. } => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nuts): ∂(initial {comp_name})/∂{param_name} is \
                             not differentiable — it {}. Use gradient-free `mh` on `ode`.",
                            code.reason_message()
                        )));
                    }
                };
                seed[comp_local * d + pos] = v;
            }
        }
        Ok(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::expr_is_time_dependent;
    use super::CompiledModel;
    use ir::expr::{BinOp, Expr, UnOp, UncheckedDimExpr, UncheckedDimWrap};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Wrap `inner` in the dimensional-escape node (`unchecked_dim`). The node is
    /// a transparent identity at runtime; every AST analysis must see through it.
    fn escape(inner: Expr) -> Expr {
        Expr::UncheckedDim(UncheckedDimWrap {
            unchecked_dim: UncheckedDimExpr {
                inner: Box::new(inner),
                dim: (0, 0),
                reason: "test".into(),
            },
        })
    }

    /// Load a golden IR fixture and resolve every parameter to a concrete value
    /// (preset first, then a `1.0` placeholder) so `CompiledModel::new` reaches
    /// the rate_grad resolution under test rather than bailing on a missing
    /// parameter value. Mirrors the loader in
    /// `tests/binding_param_free_guard.rs`.
    fn load_with_params(name: &str) -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = PathBuf::from(&manifest)
            .join("../../../ocaml/golden")
            .join(format!("{name}.ir.json"));
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {name}: {e}"));
        let mut model: ir::Model =
            ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {name}: {e}"));
        let preset = model.presets.first().cloned();
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                let v = preset
                    .as_ref()
                    .and_then(|pr| pr.params.get(&p.name).copied())
                    .unwrap_or(1.0);
                p.value = p.value.with_value(v);
            }
        }
        model
    }

    /// Negative control: the unmodified fixture (whose `infection` transition
    /// carries a real, well-keyed `rate_grad`) compiles. Proves the fixture
    /// actually exercises the rate_grad path and that the checked resolution
    /// does not false-positive on a valid key.
    #[test]
    fn well_keyed_rate_grad_compiles() {
        let model = load_with_params("sir_basic");
        let tr = model
            .transitions
            .iter()
            .find(|t| t.name == "infection")
            .expect("sir_basic has an `infection` transition");
        assert!(
            tr.rate_grad.contains_key("beta"),
            "fixture must carry a `beta` rate_grad key, else this test is vacuous"
        );
        assert!(
            CompiledModel::new(model).is_ok(),
            "a model with only well-keyed rate_grad entries must compile"
        );
    }

    /// gh#308: a malformed `Interpolated` forcing — `times` and `values` of
    /// unequal length — is rejected at the IR→compiled boundary, not stored and
    /// silently mis-evaluated. The OCaml loader now keeps the arrays aligned,
    /// but `CompiledModel::new` is the single chokepoint every IR producer
    /// (file-backed forcing, inline forcing, hand-written IR) funnels through,
    /// so the length invariant is enforced here for all of them.
    #[test]
    fn interpolated_mismatched_knot_lengths_rejected() {
        use ir::expr::ConstExpr;
        use ir::time_func::{InterpMethod, Interpolated, TimeFuncKind, TimeFunction};
        let konst = |v: f64| Expr::Const(ConstExpr { value: v });
        let mut model = load_with_params("sir_basic");
        model.time_functions.push(TimeFunction {
            name: "bad_forcing".to_string(),
            kind: TimeFuncKind::Interpolated(Interpolated {
                times: vec![konst(0.0), konst(1.0)],
                values: vec![konst(10.0)], // one short
                method: InterpMethod::Linear,
            }),
            dim: (1, 0),
            lag: None,
        });
        let msg = match CompiledModel::new(model) {
            Ok(_) => panic!("mismatched interpolation knots must be rejected"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("bad_forcing") && msg.contains("knot"),
            "error must name the forcing and the knot mismatch, got: {msg}"
        );
    }

    /// gh#345: a non-monotone time axis silently mis-interpolates
    /// (`interpolated_value` indexes by position, `constant_value`
    /// binary-searches). The OCaml producers emit knots in file/dimension order
    /// without sorting, so reject a non-strictly-increasing axis at construction.
    #[test]
    fn interpolated_nonmonotone_times_rejected() {
        use ir::expr::ConstExpr;
        use ir::time_func::{InterpMethod, Interpolated, TimeFuncKind, TimeFunction};
        let konst = |v: f64| Expr::Const(ConstExpr { value: v });
        let mut model = load_with_params("sir_basic");
        model.time_functions.push(TimeFunction {
            name: "unsorted_forcing".to_string(),
            kind: TimeFuncKind::Interpolated(Interpolated {
                times: vec![konst(20.0), konst(0.0), konst(10.0)], // out of order
                values: vec![konst(1.0), konst(2.0), konst(3.0)],
                method: InterpMethod::Linear,
            }),
            dim: (1, 0),
            lag: None,
        });
        let msg = match CompiledModel::new(model) {
            Ok(_) => panic!("non-monotone interpolation knots must be rejected"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("unsorted_forcing") && msg.contains("increasing"),
            "error must name the forcing and the ordering requirement, got: {msg}"
        );
    }

    /// gh#308: an `Interpolated` forcing with zero knots interpolates to 0 at
    /// every time (`interpolated_value` returns 0 on an empty array), silently
    /// zeroing any rate that multiplies through it — the exact silent-wrong this
    /// PR removes. Reject it at construction instead.
    #[test]
    fn interpolated_empty_knots_rejected() {
        use ir::time_func::{InterpMethod, Interpolated, TimeFuncKind, TimeFunction};
        let mut model = load_with_params("sir_basic");
        model.time_functions.push(TimeFunction {
            name: "empty_forcing".to_string(),
            kind: TimeFuncKind::Interpolated(Interpolated {
                times: vec![],
                values: vec![],
                method: InterpMethod::Linear,
            }),
            dim: (1, 0),
            lag: None,
        });
        let msg = match CompiledModel::new(model) {
            Ok(_) => panic!("empty interpolation knots must be rejected"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("empty_forcing"),
            "error must name the forcing with no knots, got: {msg}"
        );
    }

    /// gh#204: a model carrying a reactive (state/observation-triggered) fire
    /// source is parsed and compiles, and `required_capabilities()` raises
    /// `REACTIVE_INTERVENTIONS` — so the
    /// dispatch gate (`!caps.contains(required)`) routes by backend: forward
    /// chain-binomial grants the capability and runs the agenda (PR2), while
    /// Gillespie and ODE still lack it, so the gate rejects them rather than
    /// silently dropping the policy.
    #[test]
    fn reactive_fire_source_capability_is_granted_only_by_chain_binomial() {
        use crate::Capabilities;
        use crate::{ChainBinomialSim, GillespieSim, OdeSim, Simulate};
        use ir::intervention::{
            Action, CmpOp, FireSource, FractionTransfer, Intervention,
            InterventionKind, ObsReducer, ReactiveTrigger, TriggerExpr, TriggerQuantity,
            TriggerThreshold,
        };

        // Negative control: the stock fixture (no reactive policy) does not
        // raise the flag — proving the assertion below isn't vacuous.
        let baseline = CompiledModel::new(load_with_params("sir_basic")).unwrap();
        assert!(
            !baseline
                .required_capabilities()
                .contains(Capabilities::REACTIVE_INTERVENTIONS),
            "a model with no reactive policy must not require REACTIVE_INTERVENTIONS"
        );

        // Add a reactive policy: `kind = Scenario, fire = Reactive(..)` — the
        // two-axis shape (reactive campaigns are policy interventions, not
        // events; they differ only in how fire times are produced).
        let mut model = load_with_params("sir_basic");
        model.interventions.push(Intervention {
            name: "sia_after_detection".into(),
            base_name: None,
            fire: FireSource::Reactive(ReactiveTrigger {
                when_: TriggerExpr::Cmp {
                    lhs: TriggerQuantity::Observed {
                        stream: "reported_cases".into(),
                        window: None,
                        reducer: ObsReducer::Latest,
                    },
                    op: CmpOp::Ge,
                    rhs: TriggerThreshold::Const(2.0),
                },
                after: 21.0,
                once: true,
                cooldown: None,
            }),
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(),
                dst: "R".into(),
                fraction: Expr::const_(0.7),
            })],
            kind: InterventionKind::Scenario,
        });

        let compiled = CompiledModel::new(model)
            .expect("a reactive model still compiles — it is rejected at dispatch, not compile");
        let required = compiled.required_capabilities();
        assert!(
            required.contains(Capabilities::REACTIVE_INTERVENTIONS),
            "a reactive fire source must raise REACTIVE_INTERVENTIONS"
        );

        // Forward chain-binomial grants it (PR2) ⇒ the gate passes there.
        assert!(
            ChainBinomialSim
                .capabilities()
                .contains(Capabilities::REACTIVE_INTERVENTIONS),
            "forward chain-binomial runs the reactive agenda (PR2)"
        );
        assert!(
            ChainBinomialSim.capabilities().contains(required),
            "the dispatch gate (!caps.contains(required)) must accept a reactive model on chain-binomial"
        );

        // Gillespie and ODE do NOT declare it ⇒ the gate rejects them (PR3).
        for caps in [GillespieSim.capabilities(), OdeSim.capabilities()] {
            assert!(
                !caps.contains(Capabilities::REACTIVE_INTERVENTIONS),
                "gillespie/ode do not run reactive policies yet"
            );
            assert!(
                !caps.contains(required),
                "the dispatch gate (!caps.contains(required)) must reject a reactive model on gillespie/ode"
            );
        }
    }

    /// gh#128: a `rate_grad` entry keyed on a parameter that does not exist in
    /// the model must be rejected at compile time, naming the bad key and the
    /// transition — NOT silently dropped via `filter_map`. A dropped key means
    /// gradient-based inference (NUTS) optimizes a different model than the
    /// simulator's likelihood, with no error surfaced.
    #[test]
    fn unknown_rate_grad_key_is_rejected() {
        let mut model = load_with_params("sir_basic");
        let tr = model
            .transitions
            .iter_mut()
            .find(|t| t.name == "infection")
            .expect("sir_basic has an `infection` transition");
        // Take the real (resolvable) derivative expression for `beta` and
        // re-key it under a parameter name that is NOT declared. The expression
        // itself resolves fine, so a failure can only come from the key check —
        // this keeps the test non-vacuous (it isn't catching a resolve error).
        let grad_expr = tr
            .rate_grad
            .get("beta")
            .expect("infection has a beta gradient")
            .clone();
        tr.rate_grad
            .insert("not_a_real_param".to_string(), grad_expr);
        assert!(
            !model.parameters.iter().any(|p| p.name == "not_a_real_param"),
            "guard: the bogus key must not accidentally be a real parameter"
        );

        let err = match CompiledModel::new(model) {
            Ok(_) => panic!(
                "a rate_grad keyed on an unknown parameter must be rejected, not silently dropped"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("not_a_real_param"),
            "error must name the unknown rate_grad key; got: {msg}"
        );
        assert!(
            msg.contains("infection"),
            "error must name the offending transition; got: {msg}"
        );
    }

    /// gh#275: a `rate_state_grad` keyed by a valid compartment resolves to that
    /// compartment's index — via the compartment resolver the `CompGradMap`
    /// newtype forces (not the parameter resolver) — and is stored in
    /// `rate_state_grads_indexed`, the `J_x` ingredient the sensitivity assembly
    /// consumes. Reuses `beta`'s (resolvable) grad expression re-keyed under a
    /// compartment, so the test exercises the KEY resolution, not the expr.
    #[test]
    fn rate_state_grad_resolves_to_compartment_index() {
        let mut model = load_with_params("sir_basic");
        let comp_idx: std::collections::HashMap<String, usize> = model
            .compartments
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), i))
            .collect();
        let s_idx = comp_idx["S"];
        let inf_pos = model
            .transitions
            .iter()
            .position(|t| t.name == "infection")
            .expect("sir_basic has an `infection` transition");
        let grad_expr = model.transitions[inf_pos]
            .rate_grad
            .get("beta")
            .expect("infection has a beta gradient")
            .clone();
        // sir_basic now carries compiler-EMITTED rate_state_grad ({S,I,R} over
        // N=PopSum); clear it so this test isolates the KEY resolution mechanic
        // (a single compartment key → its compartment index) rather than the
        // emitted map's size.
        model.transitions[inf_pos].rate_state_grad.0.clear();
        model.transitions[inf_pos]
            .rate_state_grad
            .0
            .insert("S".to_string(), grad_expr);

        let cm = CompiledModel::new(model)
            .expect("a rate_state_grad keyed by a valid compartment must compile");
        let rsg = &cm.resolved.rate_state_grads_indexed[inf_pos].0;
        assert_eq!(rsg.len(), 1, "one ∂rate/∂S entry expected");
        assert_eq!(
            rsg[0].0, s_idx,
            "rate_state_grad key 'S' must resolve to S's COMPARTMENT index"
        );
    }

    /// gh#275: a `rate_state_grad` keyed on an unknown compartment must be
    /// rejected at compile time (naming the compartment and the transition), not
    /// silently dropped — a dropped ∂rate/∂compartment reads as zero to the ODE
    /// forward sensitivity, integrating a different `J_x` than the dynamics.
    #[test]
    fn unknown_rate_state_grad_compartment_is_rejected() {
        let mut model = load_with_params("sir_basic");
        let inf = model
            .transitions
            .iter_mut()
            .find(|t| t.name == "infection")
            .expect("sir_basic has an `infection` transition");
        let grad_expr = inf
            .rate_grad
            .get("beta")
            .expect("infection has a beta gradient")
            .clone();
        inf.rate_state_grad
            .0
            .insert("not_a_compartment".to_string(), grad_expr);

        let err = match CompiledModel::new(model) {
            Ok(_) => panic!(
                "a rate_state_grad keyed on an unknown compartment must be rejected, not dropped"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("not_a_compartment"),
            "error must name the unknown compartment; got: {msg}"
        );
        assert!(
            msg.contains("infection"),
            "error must name the offending transition; got: {msg}"
        );
    }

    /// Regression: a bare `Time` reference must classify as time-dependent.
    /// Before the fix, only `TimeFunc` (named forcings) matched, so a rate like
    /// `lambda / (1 + exp(-(t - tau)/w))` was excluded from
    /// `time_dep_transitions` and Gillespie froze its propensity at `t=0`.
    #[test]
    fn bare_time_is_time_dependent() {
        let nb: HashMap<&str, &Expr> = HashMap::new();
        // t
        assert!(expr_is_time_dependent(&Expr::time(), &nb));
        // -(t - tau) / w  — the logistic-pulse exponent: Time nested under
        // BinOp/UnOp must still be detected.
        let exponent = Expr::bin_op(
            BinOp::Div,
            Expr::un_op(UnOp::Neg, Expr::bin_op(BinOp::Sub, Expr::time(), Expr::param("tau"))),
            Expr::param("w"),
        );
        assert!(expr_is_time_dependent(&exponent, &nb));
        // lambda / (1 + exp(exponent))
        let pulse = Expr::bin_op(
            BinOp::Div,
            Expr::param("lambda"),
            Expr::bin_op(BinOp::Add, Expr::const_(1.0), Expr::un_op(UnOp::Exp, exponent)),
        );
        assert!(expr_is_time_dependent(&pulse, &nb));
    }

    /// gh#336: `unchecked_dim(...)` is a transparent dimensional-escape wrapper —
    /// it is time-dependent iff its inner expression is. Before the fix,
    /// `expr_is_time_dependent` had no `UncheckedDim` arm, so it fell to the
    /// `_ => false` catch-all: a forcing wrapped in `unchecked_dim` (a realistic
    /// pattern used to satisfy the dim-checker) was misclassified as
    /// time-INdependent, and Gillespie froze the propensity at `t=0` — silent
    /// wrong dynamics.
    #[test]
    fn unchecked_dim_is_transparent_for_time_dependence() {
        let nb: HashMap<&str, &Expr> = HashMap::new();
        // unchecked_dim(t) — bare Time wrapped in the escape node.
        assert!(expr_is_time_dependent(&escape(Expr::time()), &nb));
        // unchecked_dim(seasonal(t)) modeled as unchecked_dim(exp(t)): a
        // time-varying inner must propagate through the wrapper.
        assert!(expr_is_time_dependent(
            &escape(Expr::un_op(UnOp::Exp, Expr::time())),
            &nb
        ));
        // beta * S * unchecked_dim(exp(t)) — the wrapper nested inside a rate.
        let rate = Expr::bin_op(
            BinOp::Mul,
            Expr::param("beta"),
            Expr::bin_op(
                BinOp::Mul,
                Expr::pop("S"),
                escape(Expr::un_op(UnOp::Exp, Expr::time())),
            ),
        );
        assert!(expr_is_time_dependent(&rate, &nb));
        // Negative: unchecked_dim over a time-free inner stays time-independent.
        assert!(!expr_is_time_dependent(
            &escape(Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S"))),
            &nb
        ));
    }

    /// A rate with no time reference (only params/consts) is not time-dependent.
    #[test]
    fn time_free_rate_is_not_time_dependent() {
        let nb: HashMap<&str, &Expr> = HashMap::new();
        // beta * S * I / N  has no time term (Pop/Param/Const only)
        let rate = Expr::bin_op(
            BinOp::Mul,
            Expr::param("beta"),
            Expr::bin_op(BinOp::Mul, Expr::pop("S"), Expr::pop("I")),
        );
        assert!(!expr_is_time_dependent(&rate, &nb));
        // `dt` is the step size, a constant — not time-varying.
        assert!(!expr_is_time_dependent(&Expr::dt(), &nb));
    }
}
