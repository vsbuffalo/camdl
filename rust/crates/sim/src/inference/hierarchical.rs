//! Hierarchical-prior support: the parameter environment and the expression
//! evaluator used to resolve a hierarchical prior's argument expressions.
//!
//! A "leaf" parameter in a hierarchical / partially-pooled group carries an
//! `ir::parameter::HierarchicalPrior` whose distribution arguments are
//! *expressions over other parameters* (hyperparents). At each log-posterior
//! evaluation those expressions are resolved against the current parameter
//! values via [`eval_prior_arg`] and a [`ParamEnv`].
//!
//! The density math itself lives in `crate::inference::prior` — a hierarchical
//! prior is a `Prior::Hierarchical(Density<ParamArg>)` and shares the single
//! `Density::log_density_env` formula with fixed priors (a `ParamArg::Expr` is
//! resolved here; a fixed prior's `f64` parameter ignores the env). This module
//! owns only the resolution machinery.
//!
//! Gate 2 constraints (from
//! `docs/dev/proposals/notes/hierarchical-priors-gate2-plan.md`):
//! - **B**: `Expr::Param(name)` resolves against an env passed in by the
//!   caller; no rebuild-per-step.
//! - **D3**: `--set` overrides on hyperparents flow through naturally because
//!   the env carries current values.
//! - **E**: NaN / out-of-support → `f64::NEG_INFINITY`, never panic (the
//!   collapse-to-`-∞` happens in `Density::log_density_env`).

use ir::expr::{BinOp, Expr, UnOp};

/// Parameter-value environment used to resolve `Expr::Param(name)` references
/// in hierarchical prior arguments. Indexed by name → current value (the
/// sampler's current state on whichever scale the caller is evaluating
/// against; the density formula consumes the resolved value consistently).
pub trait ParamEnv {
    fn get(&self, name: &str) -> Option<f64>;
}

impl ParamEnv for std::collections::HashMap<String, f64> {
    fn get(&self, name: &str) -> Option<f64> {
        std::collections::HashMap::get(self, name).copied()
    }
}

impl ParamEnv for &[(String, f64)] {
    fn get(&self, name: &str) -> Option<f64> {
        self.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }
}

/// Zero-allocation env backed by parallel name/value slices.
/// Used by the MCMC inner loop where the name slice is constant across
/// proposals and only the value slice moves. Wave 2 / #3 Gate 3.
pub struct NamedParams<'a> {
    pub names:  &'a [String],
    pub values: &'a [f64],
}

impl<'a> ParamEnv for NamedParams<'a> {
    fn get(&self, name: &str) -> Option<f64> {
        // Linear scan is fine — param vectors are small (<100) and this
        // is called O(n_leaves) times per MCMC step, not per-substep.
        self.names.iter().position(|n| n == name).map(|i| self.values[i])
    }
}

/// The "empty env" — used when calling `log_density` (env-free) on a prior.
/// Returns None for every lookup, so any hyperparent reference resolves to
/// `NaN` and the density collapses to `-∞`.
impl ParamEnv for () {
    fn get(&self, _name: &str) -> Option<f64> { None }
}

/// Lightweight expression evaluator for hierarchical-prior arguments.
/// Only the subset of `Expr` that can appear in prior args is handled:
/// constants, parameter references, and arithmetic / math on them.
/// Compartment state (`Pop`, `PopSum`, `Time`, `TimeFunc`, `TableLookup`,
/// `Projected`) is a compile error in prior args and produces
/// `f64::NAN` here as a defence in depth.
pub fn eval_prior_arg<E: ParamEnv>(expr: &Expr, env: &E) -> f64 {
    match expr {
        Expr::Const(c) => c.value,
        Expr::Param(p) => env.get(&p.param).unwrap_or(f64::NAN),
        Expr::BinOp(b) => {
            let l = eval_prior_arg(&b.bin_op.left, env);
            let r = eval_prior_arg(&b.bin_op.right, env);
            match b.bin_op.op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => l / r,
                BinOp::Pow => l.powf(r),
                BinOp::Mod => l % r,
                BinOp::Min => l.min(r),
                BinOp::Max => l.max(r),
                // Comparisons produce 0/1 — useful for conditional hyperparents.
                BinOp::Eq  => if l == r { 1.0 } else { 0.0 },
                BinOp::Neq => if l != r { 1.0 } else { 0.0 },
                BinOp::Lt  => if l <  r { 1.0 } else { 0.0 },
                BinOp::Gt  => if l >  r { 1.0 } else { 0.0 },
                BinOp::Le  => if l <= r { 1.0 } else { 0.0 },
                BinOp::Ge  => if l >= r { 1.0 } else { 0.0 },
            }
        }
        Expr::UnOp(u) => {
            let a = eval_prior_arg(&u.un_op.arg, env);
            match u.un_op.op {
                UnOp::Neg   => -a,
                UnOp::Exp   => a.exp(),
                UnOp::Log   => a.ln(),
                UnOp::Sqrt  => a.sqrt(),
                UnOp::Abs   => a.abs(),
                UnOp::Floor => a.floor(),
                UnOp::Ceil  => a.ceil(),
                UnOp::Sin   => a.sin(),
                UnOp::Cos   => a.cos(),
                UnOp::Tanh  => a.tanh(),
            }
        }
        Expr::Cond(c) => {
            if eval_prior_arg(&c.cond.pred, env) != 0.0 {
                eval_prior_arg(&c.cond.then, env)
            } else {
                eval_prior_arg(&c.cond.else_, env)
            }
        }
        // Classes of expressions that are semantically invalid in prior
        // args. The compiler is supposed to reject these, but returning
        // NaN ensures a bogus prior arg propagates to `-∞` log-density
        // rather than undefined behaviour.
        Expr::Pop(_) | Expr::PopSum(_) | Expr::Time(_) | Expr::Dt(_) | Expr::TimeFunc(_)
        | Expr::TableLookup(_) | Expr::Projected(_) | Expr::ObsColumnRef(_) => f64::NAN,
        // Dimensional escape is transparent — evaluate the inner.
        Expr::UncheckedDim(w) => eval_prior_arg(&w.unchecked_dim.inner, env),
        Expr::Reduce(w) => w.reduce.iter().map(|t| eval_prior_arg(t, env)).sum(),
        // Bindings are state-derived → invalid in prior args (like Pop): NaN.
        Expr::BindingRef(_) => f64::NAN,
        // gh#272: a per-eval ref is equally invalid in a prior argument: NaN.
        Expr::PerEvalRef(_) => f64::NAN,
    }
}
