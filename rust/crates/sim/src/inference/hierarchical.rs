//! Hierarchical-prior log-density evaluation (wave 2 / malaria #3, Gate 2).
//!
//! A "leaf" parameter in a hierarchical / partially-pooled group carries
//! an `ir::parameter::HierarchicalPrior` rather than the plain
//! `ir::parameter::PriorDist`. At each log-posterior evaluation we must:
//!
//! 1. Resolve each prior-argument expression against the current values
//!    of the hyperparameters in the parameter vector.
//! 2. Compute the log-density of the leaf value under the resolved
//!    parameterisation.
//!
//! Gate 2 constraints (from
//! `docs/dev/proposals/notes/hierarchical-priors-gate2-plan.md`):
//! - **A**: formula matches scipy to 1e-10 relative error.
//! - **A3**: same natural-/z-scale contract as `Prior::TransformedNormal`
//!   (natural-scale density here; caller adds `log|dθ/dz|` jacobian).
//! - **B**: `Expr::Param(name)` resolves against an env passed in by
//!   the caller; no rebuild-per-step.
//! - **D3**: `--set` overrides on hyperparents flow through naturally
//!   because the env carries current values.
//! - **E**: NaN / out-of-support → `f64::NEG_INFINITY`, never panic.

use crate::inference::obs_loglik::lgamma;
use crate::inference::prior::Scale;
use ir::expr::{BinOp, Expr, UnOp};
use ir::parameter::{HierarchicalKind, HierarchicalPrior};

/// 0.5 · ln(2π).
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// Parameter-value environment used to resolve `Expr::Param(name)`
/// references in hierarchical prior arguments. Indexed by name → current
/// value (the sampler's current state in the transformed or natural
/// scale, whichever the caller is evaluating against — both are
/// supported as long as the env is consistent with the `scale`
/// parameter passed alongside).
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

/// The "empty env" — used when calling `log_density_env` with a
/// non-hierarchical prior. Returns None for every lookup.
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
        // NaN ensures a bogus prior args propagates to `-∞` log-density
        // rather than undefined behaviour.
        Expr::Pop(_) | Expr::PopSum(_) | Expr::Time(_) | Expr::Dt(_) | Expr::TimeFunc(_)
        | Expr::TableLookup(_) | Expr::Projected(_) | Expr::ObsColumnRef(_) => f64::NAN,
        // Dimensional escape is transparent — evaluate the inner.
        Expr::UncheckedDim(w) => eval_prior_arg(&w.unchecked_dim.inner, env),
        Expr::Reduce(w) => w.reduce.iter().map(|t| eval_prior_arg(t, env)).sum(),
        // Bindings are state-derived → invalid in prior args (like Pop): NaN.
        Expr::BindingRef(_) => f64::NAN,
    }
}

/// Log-density of a hierarchical prior, evaluated at the leaf value
/// `natural` (and for log-transformed leaves also `transformed` = z =
/// log θ). Contract: returns the **natural-scale** density. Callers
/// using the log-transformed space add `log|dθ/dz|` on top — matching
/// `Prior::log_density` exactly (see IC3 fix in prior.rs). The `Scale`
/// phantom argument makes the contract explicit at the type level.
pub fn hierarchical_log_density<E: ParamEnv>(
    hp: &HierarchicalPrior,
    natural: f64,
    transformed: f64,
    env: &E,
    _scale_marker: Scale,
) -> f64 {
    // Helper: fetch kwarg by name; returns NaN if missing (propagates
    // to -∞ via standard NaN rules in the formulas below).
    let arg = |k: &str| -> f64 {
        hp.args.get(k)
            .map(|e| eval_prior_arg(e, env))
            .unwrap_or(f64::NAN)
    };

    // If any required arg is NaN (missing hyperparent value, out-of-
    // bounds evaluation, etc.), return -∞. Defence-in-depth per
    // plan E4 (NaN propagation isolation).
    let finite_or_neg_inf = |v: f64| if v.is_finite() { v } else { f64::NEG_INFINITY };

    match hp.kind {
        HierarchicalKind::Uniform => {
            let lower = arg("lower");
            let upper = arg("upper");
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                return f64::NEG_INFINITY;
            }
            if natural < lower || natural > upper {
                f64::NEG_INFINITY
            } else {
                -((upper - lower).ln())
            }
        }
        HierarchicalKind::Normal => {
            let mu    = arg("mu");
            let sigma = arg("sigma");
            if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let z = (natural - mu) / sigma;
            finite_or_neg_inf(-HALF_LN_2PI - sigma.ln() - 0.5 * z * z)
        }
        HierarchicalKind::LogNormal => {
            // Natural-scale density: log p(θ) = log N(log θ; μ, σ) − log θ.
            // For log-transformed leaves evaluated on z-scale, caller
            // adds log|dθ/dz| = z back (Log transform), recovering
            // log N(z; μ, σ).
            let mu    = arg("mu");
            let sigma = arg("sigma");
            if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if natural <= 0.0 { return f64::NEG_INFINITY; }
            let z_score = (transformed - mu) / sigma;
            finite_or_neg_inf(-transformed - HALF_LN_2PI - sigma.ln() - 0.5 * z_score * z_score)
        }
        HierarchicalKind::HalfNormal => {
            let sigma = arg("sigma");
            if !sigma.is_finite() || sigma <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if natural < 0.0 { return f64::NEG_INFINITY; }
            let z = natural / sigma;
            finite_or_neg_inf(std::f64::consts::LN_2 - sigma.ln() - HALF_LN_2PI - 0.5 * z * z)
        }
        HierarchicalKind::Beta => {
            let alpha = arg("alpha");
            let beta  = arg("beta");
            if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if natural <= 0.0 || natural >= 1.0 { return f64::NEG_INFINITY; }
            finite_or_neg_inf(
                (alpha - 1.0) * natural.ln()
                + (beta - 1.0) * (1.0 - natural).ln()
                - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
            )
        }
        HierarchicalKind::Gamma => {
            let shape = arg("shape");
            let rate  = arg("rate");
            if !shape.is_finite() || !rate.is_finite() || shape <= 0.0 || rate <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if natural <= 0.0 { return f64::NEG_INFINITY; }
            finite_or_neg_inf(
                shape * rate.ln()
                + (shape - 1.0) * natural.ln()
                - rate * natural
                - lgamma(shape)
            )
        }
        HierarchicalKind::Exponential => {
            let rate = arg("rate");
            if !rate.is_finite() || rate <= 0.0 { return f64::NEG_INFINITY; }
            if natural < 0.0 { return f64::NEG_INFINITY; }
            finite_or_neg_inf(rate.ln() - rate * natural)
        }
    }
}
