//! The gradient-method capability gate (gh#275 §1h).
//!
//! "Can this model be fit by a gradient method?" is answered in ONE place, so the
//! refusals do not scatter across the integration, scoring, and preflight sites
//! (the silent-gap class the dense-matrix rule warns about). Two layers:
//!
//! - [`scan_unsupported_gradients`] — the COMMON coverage scan every gradient
//!   method shares: an estimated parameter whose gradient the compiler could not
//!   emit for a rate, observation argument, or σ² term (a serialized
//!   [`DerivEntry::Unsupported`]), or that reaches a Binomial/BetaBinomial `n`.
//!   The PGAS-NUTS preflight (`pgas.rs`) and the ODE-NUTS gate both call it, so
//!   the two cannot drift.
//! - [`preflight_gradient_ode`] — the ODE-NUTS gate: the common scan PLUS the
//!   checks specific to the deterministic forward-sensitivity path (a nonsmooth
//!   `rate_state_grad`, an adaptive integrator, a `dt`-in-rate model, a scheduled
//!   effect, a parameterized initial condition, a `DerivedExpr` prevalence
//!   projection, a `projected`-transforming likelihood argument). One function,
//!   one list, run once before any gradient is taken.

use std::collections::{BTreeMap, HashSet};

use ir::deriv::{DerivEntry, UnsupportedReason};
use ir::expr::Expr;
use ir::observation::{Likelihood, Projection};
use ir::transition::DrawMethod;
use ir::Differentiable;

use crate::boundary_times::EffectTimes;
use crate::compiled_model::CompiledModel;
use crate::error::SimError;

/// Collect names of every `Param` referenced by an expression tree.
pub(crate) fn collect_param_refs(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Param(p) => {
            out.insert(p.param.clone());
        }
        Expr::BinOp(w) => {
            collect_param_refs(&w.bin_op.left, out);
            collect_param_refs(&w.bin_op.right, out);
        }
        Expr::UnOp(w) => collect_param_refs(&w.un_op.arg, out),
        Expr::Cond(w) => {
            collect_param_refs(&w.cond.pred, out);
            collect_param_refs(&w.cond.then, out);
            collect_param_refs(&w.cond.else_, out);
        }
        Expr::PopSum(_) | Expr::Pop(_) | Expr::Const(_) | Expr::Time(_) | Expr::Dt(_)
        | Expr::TimeFunc(_) | Expr::Projected(_) | Expr::ObsColumnRef(_) => {}
        Expr::TableLookup(w) => {
            for ix in &w.table_lookup.indices {
                collect_param_refs(ix, out);
            }
        }
        Expr::UncheckedDim(w) => collect_param_refs(&w.unchecked_dim.inner, out),
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect_param_refs(t, out);
            }
        }
        Expr::BindingRef(_) => {}
        Expr::PerEvalRef(_) => {
            unreachable!("PerEvalRef reached collect_param_refs: LICM scoping invariant violated")
        }
    }
}

/// Collect every `Param` an estimated coordinate could reach through a
/// Binomial/BetaBinomial `n` expression, **with the projection inlined** exactly
/// as the OCaml autodiff inlines it. At a `Projected` node a `DerivedExpr`
/// projection contributes its own param refs; every other projection kind leaves
/// `Projected` θ-independent given the fixed trajectory. `n` carries no gradient
/// field (it is rounded to an integer, so it must be θ-independent), so any
/// estimated param found here is refused as [`UnsupportedReason::ParametricN`].
pub(crate) fn collect_n_param_refs(
    n: &Expr,
    projection: &Projection,
    out: &mut HashSet<String>,
) {
    match n {
        Expr::Projected(_) => {
            if let Projection::DerivedExpr(e) = projection {
                collect_param_refs(e, out);
            }
        }
        Expr::Param(p) => {
            out.insert(p.param.clone());
        }
        Expr::BinOp(w) => {
            collect_n_param_refs(&w.bin_op.left, projection, out);
            collect_n_param_refs(&w.bin_op.right, projection, out);
        }
        Expr::UnOp(w) => collect_n_param_refs(&w.un_op.arg, projection, out),
        Expr::Cond(w) => {
            collect_n_param_refs(&w.cond.pred, projection, out);
            collect_n_param_refs(&w.cond.then, projection, out);
            collect_n_param_refs(&w.cond.else_, projection, out);
        }
        Expr::TableLookup(w) => {
            for ix in &w.table_lookup.indices {
                collect_n_param_refs(ix, projection, out);
            }
        }
        Expr::UncheckedDim(w) => collect_n_param_refs(&w.unchecked_dim.inner, projection, out),
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect_n_param_refs(t, projection, out);
            }
        }
        Expr::PopSum(_) | Expr::Pop(_) | Expr::Const(_) | Expr::Time(_) | Expr::Dt(_)
        | Expr::TimeFunc(_) | Expr::ObsColumnRef(_) | Expr::BindingRef(_) => {}
        Expr::PerEvalRef(_) => {
            unreachable!("PerEvalRef reached collect_n_param_refs: LICM scoping invariant violated")
        }
    }
}

/// True if `e` references the observation projection anywhere. IR-level twin of
/// `resolved_expr::references_projected` (used at preflight, before resolution).
fn expr_references_projected(e: &Expr) -> bool {
    match e {
        Expr::Projected(_) => true,
        Expr::BinOp(w) => {
            expr_references_projected(&w.bin_op.left) || expr_references_projected(&w.bin_op.right)
        }
        Expr::UnOp(w) => expr_references_projected(&w.un_op.arg),
        Expr::Cond(w) => {
            expr_references_projected(&w.cond.pred)
                || expr_references_projected(&w.cond.then)
                || expr_references_projected(&w.cond.else_)
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter().any(expr_references_projected),
        Expr::UncheckedDim(w) => expr_references_projected(&w.unchecked_dim.inner),
        Expr::Reduce(w) => w.reduce.iter().any(expr_references_projected),
        _ => false,
    }
}

/// True if `e` references a shared model-level binding. `collect_param_refs` does
/// not descend into a `BindingRef`, so an initial-condition expression that hides
/// a parameter behind a binding could slip an estimated param past the ic_grad
/// completeness check — the gradient gate refuses such an IC rather than risk a
/// silently-zero seed (gh#275 §1c).
fn expr_has_binding_ref(e: &Expr) -> bool {
    match e {
        Expr::BindingRef(_) => true,
        Expr::BinOp(w) => {
            expr_has_binding_ref(&w.bin_op.left) || expr_has_binding_ref(&w.bin_op.right)
        }
        Expr::UnOp(w) => expr_has_binding_ref(&w.un_op.arg),
        Expr::Cond(w) => {
            expr_has_binding_ref(&w.cond.pred)
                || expr_has_binding_ref(&w.cond.then)
                || expr_has_binding_ref(&w.cond.else_)
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter().any(expr_has_binding_ref),
        Expr::UncheckedDim(w) => expr_has_binding_ref(&w.unchecked_dim.inner),
        Expr::Reduce(w) => w.reduce.iter().any(expr_has_binding_ref),
        _ => false,
    }
}

/// The COMMON gradient-coverage scan (§1h): map each estimated parameter whose
/// gradient the compiler could not emit — for a rate, an observation argument, or
/// a σ² term — to the stable [`UnsupportedReason`] the fit-time message derives
/// from. Deterministic order (a `BTreeMap`); the first reason wins for a parameter
/// uncovered on more than one surface. Shared by PGAS-NUTS and ODE-NUTS.
pub(crate) fn scan_unsupported_gradients(
    model: &ir::Model,
    estimated: &HashSet<&str>,
) -> BTreeMap<String, UnsupportedReason> {
    let mut refused: BTreeMap<String, UnsupportedReason> = BTreeMap::new();
    let note = |grad: &std::collections::HashMap<String, DerivEntry>,
                refused: &mut BTreeMap<String, UnsupportedReason>| {
        for (pname, entry) in grad {
            if let DerivEntry::Unsupported { code, .. } = entry {
                if estimated.contains(pname.as_str()) {
                    refused.entry(pname.clone()).or_insert(*code);
                }
            }
        }
    };

    // (a) Observation likelihood argument gradients (projection already inlined by
    //     the compiler). Via the derived `diffables()` traversal so a new argument
    //     is scanned automatically.
    for om in &model.observations {
        for (_, d) in om.likelihood.diffables() {
            note(&d.grad, &mut refused);
        }
    }
    // (b) Transition rate gradients + σ² overdispersion gradients.
    for t in &model.transitions {
        note(&t.rate_grad, &mut refused);
        if let DrawMethod::Overdispersed { sigma_sq_grad, .. } = &t.draw_method {
            note(sigma_sq_grad, &mut refused);
        }
    }
    // (c) D-n — an estimated param reaching a Binomial/BetaBinomial `n` (after
    //     projection inlining). `n` carries no grad field, so scan it directly.
    for om in &model.observations {
        let n_expr = match &om.likelihood {
            Likelihood::Binomial(l) => Some(&l.n),
            Likelihood::BetaBinomial(l) => Some(&l.n),
            _ => None,
        };
        if let Some(n) = n_expr {
            let mut refs: HashSet<String> = HashSet::new();
            collect_n_param_refs(n, &om.projection, &mut refs);
            for pname in refs {
                if estimated.contains(pname.as_str()) {
                    refused.entry(pname).or_insert(UnsupportedReason::ParametricN);
                }
            }
        }
    }
    // (d) Zero-inflated NegBinomial is scoring-only: the ENTIRE family carries no
    //     emitted gradient (all bare exprs, no `Diffable`, so (a)'s `diffables()`
    //     scan sees nothing). Unlike Binomial's `n` — where only `n` is
    //     θ-independent and the rest of the likelihood IS differentiable — a ZI
    //     stream contributes NO gradient at all: not for a param textually in its
    //     args, and not for a rate param that reaches it only through the shared
    //     trajectory (the factor-2 chain). A per-param scan of the arg exprs is
    //     therefore the wrong shape here — it misses the trajectory-coupled case,
    //     which then hits the `unreachable!` backstop in obs_model.rs
    //     mid-integration (a reachable panic, not a clean refusal). So whenever
    //     the model contains a ZI stream, refuse EVERY estimated param: the honest
    //     model-level refusal that routes the user to a gradient-free method.
    let has_zero_inflated = model
        .observations
        .iter()
        .any(|om| matches!(om.likelihood, Likelihood::ZeroInflatedNegBinomial(_)));
    if has_zero_inflated {
        for pname in estimated {
            refused
                .entry(pname.to_string())
                .or_insert(UnsupportedReason::ZeroInflated);
        }
    }
    refused
}

/// The ODE-NUTS gradient gate (§1h): the common coverage scan PLUS the checks
/// specific to the deterministic ODE forward-sensitivity path. Run once by
/// `det_grad` (and any future ODE gradient-MLE) before a gradient is taken;
/// refuses with the compiler's own reason string where one exists, so the user
/// gets a single actionable message instead of a mid-integration failure.
///
/// `estimated` is the set of estimated parameter NAMES. The nonnegativity-clamp
/// refusal (§1c) stays a runtime check — an *active* clamp is only observable
/// during integration — and lives in `rk4_step`.
pub fn preflight_gradient_ode(
    model: &CompiledModel,
    params: &[f64],
    estimated: &HashSet<&str>,
) -> Result<(), SimError> {
    let m = &model.model;

    // ── Common coverage: obs / rate / σ² Unsupported, ParametricN ───────────────
    let refused = scan_unsupported_gradients(m, estimated);
    if !refused.is_empty() {
        let details: Vec<String> = refused
            .iter()
            .map(|(p, code)| format!("`{}` {}", p, code.reason_message()))
            .collect();
        return Err(SimError::Validation(format!(
            "ODE gradient (nuts) cannot estimate parameter(s) whose gradient the compiler \
             could not emit for a rate, observation, or overdispersion term — NUTS would \
             sample against an incomplete (silently biased) gradient. Refused: {}. Use a \
             gradient-free method (IF2 or PMMH), or fix these parameters.",
            details.join("; ")
        )));
    }

    // ── State-Jacobian present at all (gh#439) ──────────────────────────────────
    // The forward sensitivity chains ∂rate/∂state (`rate_state_grad`, J_x) into
    // dS/dt. A model compiled with `camdlc --no-state-grad` carries it EMPTY on
    // every transition, which would make the sensitivity silently drop the J_x·S
    // coupling term and sample against a biased gradient. A genuine ODE fit has
    // state-dependent dynamics, so at least one transition's rate_state_grad is
    // non-empty; if every one is empty, refuse loudly rather than integrate a
    // wrong sensitivity.
    if !m.transitions.is_empty()
        && m.transitions.iter().all(|t| t.rate_state_grad.is_empty())
    {
        return Err(SimError::Validation(
            "ODE gradient (nuts) requires the state-Jacobian `rate_state_grad`, but this \
             model carries none on any transition — it was compiled with `camdlc \
             --no-state-grad` (or has no state-dependent rates). Recompile WITHOUT \
             --no-state-grad to fit with `nuts` on the ODE backend, or use a gradient-free \
             method (IF2, PMMH, or `mh` on ode)."
                .to_string(),
        ));
    }

    // ── ODE-sensitivity coverage: a nonsmooth ∂rate/∂state (§1a) ────────────────
    // A `floor`/`ceil`/`abs`/`min`/`max`/`mod` of a compartment (or a state-indexed
    // table) has no smooth state derivative, so the WrtPop pass serializes a
    // `DEUnsupported` in `rate_state_grad`. The forward sensitivity `J_x` is then
    // undefined for EVERY parameter, not just one — a model-level refusal.
    for t in &m.transitions {
        for (_comp, entry) in t.rate_state_grad.iter() {
            if let DerivEntry::Unsupported { code, .. } = entry {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nuts): transition `{}` has a rate whose derivative with \
                     respect to a compartment is not smooth — it {}. The ODE forward \
                     sensitivity is undefined for such a rate. Reformulate the rate with a \
                     smooth expression, or use gradient-free `mh` on `ode`.",
                    t.name,
                    code.reason_message()
                )));
            }
        }
    }

    // ── Integrator: fixed-step RK4 only (§1c) ───────────────────────────────────
    if !matches!(m.simulation.integrator, ir::model::Integrator::Rk4) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) requires integrator = rk4 (fixed step); an adaptive \
             integrator (rk45) is refused — its step sequence is discontinuous in θ, \
             breaking the forward-sensitivity/loglik consistency NUTS needs (gh#275 §1c)."
                .to_string(),
        ));
    }
    // ── `dt`-in-rate (RUNTIME_DT): no augmented flow sensitivity (§1c) ──────────
    if model.required_capabilities().contains(crate::Capabilities::RUNTIME_DT) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) cannot differentiate a model that references `dt` in a \
             rate (Expr::Dt): its flow uses the first-order Euler scheme, which carries no \
             augmented sensitivity (gh#275 §1c)."
                .to_string(),
        ));
    }
    // ── Real-valued compartments: the forward sensitivity is integer-compartment
    //    only (`sensitivity_derivs`). A real ODE-equation compartment's ∂x/∂θ is
    //    not tracked, so any rate or projection that reads it would get a
    //    silently-incomplete gradient; worse, for a mixed int/real model the
    //    `rate_state_grad` keys (global compartment index) would mis-index the
    //    int-local sensitivity buffer. Refuse until real-compartment forward
    //    sensitivity is built (gh#275 §1c follow-up) — a hard error, never a
    //    silent-wrong gradient.
    if model.required_capabilities().contains(crate::Capabilities::REAL_COMPARTMENTS) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) does not support real-valued compartments (ODE-equation \
             compartments): the forward sensitivity is integer-compartment only, so a real \
             compartment's ∂x/∂θ is not tracked (gh#275 §1c). Use gradient-free `mh` on \
             `ode` for models with real compartments."
                .to_string(),
        ));
    }
    // ── Scheduled effects: event-jump sensitivity is a follow-up (§1g) ──────────
    if !EffectTimes::from_model(model, params)?.into_vec().is_empty() {
        return Err(SimError::Validation(
            "ODE gradient (nuts) does not yet support scheduled interventions or events: \
             their event-jump sensitivity is a follow-up (gh#275 §1g). Use gradient-free \
             `mh` on `ode` for models with effects."
                .to_string(),
        ));
    }
    // ── Initial condition: the ∂init/∂θ seed (`ic_grad`, §1c C-seed) ────────────
    // A parameterized IC whose expression involves an estimated parameter needs the
    // forward sensitivity seeded at S(t_start) = ∂(initial_state)/∂θ. Admit it only
    // when the compiler emitted a smooth, int-compartment `ic_grad` entry for every
    // (estimated param → IC compartment) pair — otherwise the seed would silently
    // drop that parameter's contribution and NUTS would sample against a gradient
    // that is zero exactly where the initial condition matters (Risk #1).
    match &m.initial_conditions {
        ir::model::InitialConditions::Explicit(_) => {} // ∂init/∂θ ≡ 0
        ir::model::InitialConditions::FromDistribution(_) => {
            return Err(SimError::Validation(
                "ODE gradient (nuts) does not support from_distribution (ivp) initial \
                 conditions: on the deterministic ODE there is no latent initial state to \
                 draw, and the mean-treatment ∂(N·p)/∂p seed is a separate decision (gh#275 \
                 §1c). Write the initial condition as a parameterized expression, or use \
                 gradient-free `mh` on `ode`."
                    .to_string(),
            ));
        }
        ir::model::InitialConditions::Parameterized(map) => {
            for (comp, expr) in map {
                // A binding could hide a param reference the scan below does not
                // descend into — refuse rather than risk a silent-zero seed.
                if expr_has_binding_ref(expr) {
                    return Err(SimError::Validation(format!(
                        "ODE gradient (nuts): the parameterized initial condition of `{comp}` \
                         references a shared binding, which the gradient path does not yet \
                         trace for the ∂init/∂θ seed (gh#275 §1c). Inline the expression, or \
                         use gradient-free `mh` on `ode`."
                    )));
                }
                let mut refs = HashSet::new();
                collect_param_refs(expr, &mut refs);
                for pname in refs.iter().filter(|p| estimated.contains(p.as_str())) {
                    // Real-compartment forward sensitivity is a separate follow-up.
                    let global = model.comp_index.get(comp.as_str()).copied().ok_or_else(|| {
                        SimError::Validation(format!(
                            "ODE gradient (nuts): initial condition names unknown compartment \
                             `{comp}`."
                        ))
                    })?;
                    if model.global_to_int[global].is_none() {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nuts): parameterized initial condition on real \
                             compartment `{comp}` — real-compartment forward sensitivity is not \
                             yet supported (gh#275 §1c). Use gradient-free `mh` on `ode`."
                        )));
                    }
                    match m.ic_grad.get(comp).and_then(|pm| pm.get(pname.as_str())) {
                        Some(ir::deriv::DerivEntry::Grad(_)) => {}
                        Some(ir::deriv::DerivEntry::Unsupported { code, .. }) => {
                            return Err(SimError::Validation(format!(
                                "ODE gradient (nuts): ∂(initial {comp})/∂{pname} is not \
                                 differentiable — it {}. Reformulate the initial condition \
                                 with a smooth expression, or use gradient-free `mh` on `ode`.",
                                code.reason_message()
                            )));
                        }
                        None => {
                            return Err(SimError::Validation(format!(
                                "ODE gradient (nuts): estimated parameter `{pname}` enters the \
                                 initial condition of `{comp}`, but there is no usable \
                                 ∂(initial {comp})/∂{pname} — either the initial condition is not \
                                 differentiable in `{pname}` (e.g. a floor/round of it), or this \
                                 model was compiled before ic_grad emission (gh#275 §1c). Its \
                                 gradient would be silently zero. Reformulate the initial \
                                 condition, recompile, or use gradient-free `mh` on `ode`."
                            )));
                        }
                    }
                }
            }
        }
    }
    // ── Observation projections + likelihood arguments ──────────────────────────
    for om in &m.observations {
        // A DerivedExpr (nonlinear) prevalence projection IS supported — the
        // compiler emits `∂projection/∂compartment` (`projection_state_grad`, the
        // factor-2 ingredient, §1h). What is refused is a NONSMOOTH function of
        // state in the projection (`floor(I/N)`, a `Cond` on `Pop`, a state-indexed
        // table), whose WrtPop gradient the pass serializes as `Unsupported` — the
        // forward-sensitivity chain would otherwise silently drop a real term.
        if matches!(om.projection, Projection::DerivedExpr(_)) {
            // A DerivedExpr is a nonlinear function of state, so its
            // ∂projection/∂compartment must be emitted — an empty map means the
            // factor-2 chain would silently drop the whole ∂projected/∂θ term.
            if om.projection_state_grad.is_empty() {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nuts): the DerivedExpr projection of observation stream `{}` \
                     carries no ∂projection/∂compartment — its factor-2 gradient would be \
                     silently zero. Recompile with a camdlc that emits projection_state_grad \
                     (gh#275 §1h). (If the projection genuinely does not depend on compartment \
                     state, it is not an identifiable observation of the trajectory — it cannot \
                     inform a gradient fit; use gradient-free `mh` on `ode`.)",
                    om.name
                )));
            }
            for (comp, entry) in om.projection_state_grad.iter() {
                if let DerivEntry::Unsupported { code, .. } = entry {
                    return Err(SimError::Validation(format!(
                        "ODE gradient (nuts): the projection of observation stream `{}` is a \
                         nonsmooth function of compartment `{}` — it {}. The ∂projection/∂state \
                         chain is undefined. Reformulate the projection with a smooth \
                         expression, or use gradient-free `mh` on `ode`.",
                        om.name, comp, code.reason_message()
                    )));
                }
            }
        }
        // A likelihood argument that transforms `projected` (a reporting rate
        // `rho·projected`, the He mean-linked variance) IS supported — the compiler
        // emits `∂arg/∂projected` (`proj_grad`). What is refused is a NONSMOOTH
        // function of the projection output (`floor`/`min`/… of `projected`), whose
        // `proj_grad` the WrtProjected pass classifies `Unsupported`.
        for (arg, d) in om.likelihood.diffables() {
            if let Some(DerivEntry::Unsupported { code, .. }) = &d.proj_grad {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nuts): observation stream `{}` argument `{}` is a nonsmooth \
                     function of the projection output — it {}. Reformulate it with a smooth \
                     expression, or use gradient-free `mh` on `ode`.",
                    om.name,
                    arg,
                    code.reason_message()
                )));
            }
        }
        // A Binomial/BetaBinomial `n` that reads the projection output (§1d): `n`
        // carries no `proj_grad` (it is not a differentiable position — it is
        // rounded to an integer), so its projection dependence would be silently
        // dropped by the factor-2 chain. Refuse.
        let n_expr = match &om.likelihood {
            Likelihood::Binomial(l) => Some(&l.n),
            Likelihood::BetaBinomial(l) => Some(&l.n),
            _ => None,
        };
        if let Some(n) = n_expr {
            if expr_references_projected(n) {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nuts): observation stream `{}` has a Binomial/BetaBinomial \
                     denominator `n` that depends on the projection output — `n` is rounded to \
                     an integer and carries no gradient, so its projection dependence cannot be \
                     differentiated (gh#275 §1d). Make `n` a constant or an observed data column.",
                    om.name
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! §1f expected-refusal tests: one per gate bullet. Each starts from a model
    //! that PASSES the gate (`base_model`) and mutates exactly one thing so the
    //! failure is attributable to that mutation.
    use super::*;
    use ir::deriv::DerivEntry;
    use ir::expr::{ConstExpr, Expr, ParamExpr, ProjectedExpr, UnOp};
    use ir::observation::{Likelihood, NegBinomialLikelihood, PoissonLikelihood};
    use ir::Diffable;
    use std::collections::HashMap;

    fn projected() -> Expr {
        Expr::Projected(ProjectedExpr { projected: () })
    }

    /// `seir_observations`, shaped to PASS the ODE gradient gate: explicit init,
    /// `weekly_cases: negbin(mean=projected, dispersion=k)`,
    /// `detection: poisson(rate=projected)`. The compiler-emitted rate_state_grad
    /// is smooth (linear/bilinear SEIR), so the gate admits it.
    fn base_model() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model =
            ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        model.initial_conditions = ir::model::InitialConditions::Explicit(HashMap::from([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]));
        // The golden now carries emitted `ic_grad` for its own parameterized IC;
        // this base forces an Explicit IC, so its ∂init/∂θ must be empty. Each IC
        // test sets `ic_grad` explicitly to model the scenario it exercises.
        model.ic_grad = std::collections::HashMap::new();
        for om in &mut model.observations {
            match om.name.as_str() {
                "weekly_cases" => {
                    om.likelihood = Likelihood::NegBinomial(NegBinomialLikelihood {
                        mean: Diffable { expr: projected(), grad: HashMap::new(), proj_grad: None },
                        dispersion: Diffable {
                            expr: Expr::Param(ParamExpr { param: "k".to_string() }),
                            grad: HashMap::from([(
                                "k".to_string(),
                                DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 })),
                            )]),
                            proj_grad: None,
                        },
                    });
                }
                "detection" => {
                    om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                        rate: Diffable { expr: projected(), grad: HashMap::new(), proj_grad: None },
                    });
                }
                _ => {}
            }
        }
        model
    }

    fn compile(model: ir::Model) -> CompiledModel {
        // Fill any unset params so CompiledModel::new resolves.
        let mut model = model;
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(match p.name.as_str() {
                    "beta" => 0.6, "sigma" => 0.2, "gamma" => 0.1, "k" => 8.0,
                    "N0" => 10000.0, "I0" => 10.0, _ => 0.5,
                });
            }
        }
        CompiledModel::new(model).unwrap()
    }

    fn est<'a>(names: &'a [&'a str]) -> HashSet<&'a str> {
        names.iter().copied().collect()
    }

    fn gate_err(model: ir::Model, estimated: &[&str]) -> String {
        let cm = compile(model);
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        match preflight_gradient_ode(&cm, &params, &est(estimated)) {
            Ok(()) => panic!("gate should have refused"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn base_model_passes_the_gate() {
        let cm = compile(base_model());
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        preflight_gradient_ode(&cm, &params, &est(&["beta", "gamma", "k"]))
            .expect("the base model must pass the ODE gradient gate");
    }

    #[test]
    fn refuses_absent_state_jacobian_no_state_grad() {
        // A model compiled with `camdlc --no-state-grad` carries an EMPTY
        // rate_state_grad on every transition (gh#439). The ODE-NUTS gate must
        // refuse loudly rather than integrate a forward sensitivity that silently
        // drops the J_x·S state-coupling term.
        let mut m = base_model();
        for t in &mut m.transitions {
            t.rate_state_grad = Default::default();
        }
        let err = gate_err(m, &["beta", "gamma", "k"]);
        assert!(
            err.contains("--no-state-grad") && err.contains("rate_state_grad"),
            "expected a --no-state-grad refusal naming rate_state_grad, got: {err}"
        );
    }

    #[test]
    fn refuses_unsupported_rate_grad() {
        // An estimated param with a serialized DerivEntry::Unsupported in rate_grad.
        let mut m = base_model();
        m.transitions[0].rate_grad.insert(
            "beta".to_string(),
            DerivEntry::Unsupported {
                node: "time_func:seasonal".to_string(),
                code: UnsupportedReason::Lag,
            },
        );
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("beta") && msg.contains("could not emit"), "{msg}");
    }

    #[test]
    fn refuses_every_param_when_a_zero_inflated_stream_is_present() {
        // Regression: a zero-inflated stream is scoring-only, so an estimated
        // RATE param (`beta`) that reaches it only through the trajectory —
        // never textually in the ZI args (mean = projected here) — must still be
        // refused. A per-param scan of the arg exprs misses this and lets it
        // reach the `unreachable!` grad backstop in obs_model.rs (a panic). The
        // gate is model-level: any ZI stream → refuse every estimated param.
        let mut m = base_model();
        for om in &mut m.observations {
            if om.name == "weekly_cases" {
                om.likelihood = Likelihood::ZeroInflatedNegBinomial(
                    ir::observation::ZeroInflatedNegBinomialLikelihood {
                        mean: projected(),
                        dispersion: Expr::Param(ParamExpr { param: "k".to_string() }),
                        pi: Expr::Const(ConstExpr { value: 0.3 }),
                    },
                );
            }
        }
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("beta"), "must name the refused param: {msg}");
        assert!(
            msg.contains("zero_inflated") || msg.contains("scoring-only"),
            "must explain the ZI scoring-only refusal: {msg}"
        );
    }

    #[test]
    fn refuses_nonsmooth_rate_state_grad() {
        // A nonsmooth ∂rate/∂compartment (URNonsmoothState) → J_x undefined.
        let mut m = base_model();
        m.transitions[0].rate_state_grad.0.insert(
            "S".to_string(),
            DerivEntry::Unsupported {
                node: "floor(S)".to_string(),
                code: UnsupportedReason::NonsmoothState,
            },
        );
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("not smooth") && msg.contains("infection"), "{msg}");
    }

    #[test]
    fn refuses_real_compartments() {
        // sir_reservoir has a real-valued reservoir compartment (ODE-equation
        // driven). The forward sensitivity is integer-compartment only, so its
        // ∂x/∂θ is untracked (silent-wrong / mis-indexed). The gate must refuse —
        // a hard error, not a silent gradient.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/sir_reservoir.ir.json");
        let m: ir::Model =
            ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("real-valued compartments"), "{msg}");
    }

    #[test]
    fn refuses_adaptive_integrator() {
        let mut m = base_model();
        m.simulation.integrator = ir::model::Integrator::Rk45 { atol: None, rtol: None };
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("rk4") && msg.contains("rk45"), "{msg}");
    }

    /// A parameterized initial condition `I = I0`, with `I0` NOT estimated. The IC
    /// contributes `∂init/∂θ = 0` for every estimated parameter, so the gate must
    /// ADMIT it (no seed needed) — guards against the old blanket refusal.
    fn parameterized_ic_i_from_i0(m: &mut ir::Model) {
        m.initial_conditions = ir::model::InitialConditions::Parameterized(HashMap::from([
            ("S".to_string(), Expr::Const(ConstExpr { value: 9990.0 })),
            ("E".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
            ("I".to_string(), Expr::Param(ParamExpr { param: "I0".to_string() })),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ]));
    }

    #[test]
    fn admits_parameterized_ic_with_only_fixed_params() {
        let mut m = base_model();
        parameterized_ic_i_from_i0(&mut m);
        let cm = compile(m);
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        // I0 is NOT in the estimated set → the IC is θ-independent → admitted.
        preflight_gradient_ode(&cm, &params, &est(&["beta", "gamma"]))
            .expect("a parameterized IC referencing only fixed params must be admitted");
    }

    #[test]
    fn refuses_parameterized_ic_estimated_param_without_ic_grad() {
        // I = I0 with I0 ESTIMATED but no emitted ic_grad → the seed would silently
        // drop I0's IC contribution (Risk #1). The gate must refuse, naming the
        // parameter and the missing ic_grad.
        let mut m = base_model();
        parameterized_ic_i_from_i0(&mut m); // ic_grad stays empty
        let msg = gate_err(m, &["I0"]);
        assert!(
            msg.contains("I0") && msg.contains("silently zero") && msg.contains("ic_grad"),
            "{msg}"
        );
    }

    #[test]
    fn admits_parameterized_ic_with_emitted_ic_grad() {
        // I = I0, I0 estimated, and the compiler emitted ∂(initial I)/∂I0 = 1 → the
        // seed is well-defined → admitted (guards against over-refusal).
        let mut m = base_model();
        parameterized_ic_i_from_i0(&mut m);
        m.ic_grad = HashMap::from([(
            "I".to_string(),
            HashMap::from([(
                "I0".to_string(),
                DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 })),
            )]),
        )]);
        let cm = compile(m);
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        preflight_gradient_ode(&cm, &params, &est(&["I0"]))
            .expect("a parameterized IC with an emitted smooth ic_grad must be admitted");
    }

    #[test]
    fn refuses_nonsmooth_parameterized_ic() {
        // ∂(initial I)/∂I0 emitted as Unsupported → refuse (nonsmooth IC).
        let mut m = base_model();
        parameterized_ic_i_from_i0(&mut m);
        m.ic_grad = HashMap::from([(
            "I".to_string(),
            HashMap::from([(
                "I0".to_string(),
                DerivEntry::Unsupported {
                    node: "floor(I0)".to_string(),
                    code: UnsupportedReason::NonsmoothState,
                },
            )]),
        )]);
        let msg = gate_err(m, &["I0"]);
        assert!(msg.contains("I0") && msg.contains("differentiable"), "{msg}");
    }

    #[test]
    fn refuses_from_distribution_initial_condition() {
        let mut m = base_model();
        m.initial_conditions = ir::model::InitialConditions::FromDistribution(HashMap::new());
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("from_distribution") && msg.contains("ivp"), "{msg}");
    }

    /// Set the `detection` stream to a nonlinear prevalence projection `I / S`.
    fn detection_derivedexpr(m: &mut ir::Model) {
        for om in &mut m.observations {
            if om.name == "detection" {
                om.projection = ir::observation::Projection::DerivedExpr(Expr::bin_op(
                    ir::expr::BinOp::Div, Expr::pop("I"), Expr::pop("S"),
                ));
            }
        }
    }

    #[test]
    fn refuses_derivedexpr_projection_without_emitted_gradient() {
        // A DerivedExpr projection with no ∂proj/∂compartment emitted → the
        // factor-2 chain would be silently zero. Refuse (Risk #3 silent-zero).
        let mut m = base_model();
        detection_derivedexpr(&mut m); // projection_state_grad left empty
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("silently zero") && msg.contains("detection"), "{msg}");
    }

    #[test]
    fn admits_derivedexpr_projection_with_emitted_gradient() {
        // A DerivedExpr with a smooth emitted ∂proj/∂compartment is SUPPORTED now
        // (guards against over-refusal — the whole point of this change).
        let mut m = base_model();
        detection_derivedexpr(&mut m);
        for om in &mut m.observations {
            if om.name == "detection" {
                om.projection_state_grad.0.insert(
                    "I".to_string(), DerivEntry::Grad(Expr::const_(0.5)));
                om.projection_state_grad.0.insert(
                    "S".to_string(), DerivEntry::Grad(Expr::const_(-0.5)));
            }
        }
        let cm = compile(m);
        let params: Vec<f64> = cm.model.parameters.iter()
            .map(|p| p.value.resolved_value().unwrap()).collect();
        preflight_gradient_ode(&cm, &params, &est(&["beta"]))
            .expect("a DerivedExpr with an emitted smooth projection gradient must be admitted");
    }

    #[test]
    fn refuses_nonsmooth_derivedexpr_projection() {
        // A nonsmooth-of-state projection (e.g. floor(I/N)) → the WrtPop pass emits
        // an Unsupported ∂proj/∂compartment. Refuse (the chain is undefined).
        let mut m = base_model();
        detection_derivedexpr(&mut m);
        for om in &mut m.observations {
            if om.name == "detection" {
                om.projection_state_grad.0.insert(
                    "I".to_string(),
                    DerivEntry::Unsupported {
                        node: "floor(I/S)".to_string(),
                        code: UnsupportedReason::NonsmoothState,
                    });
            }
        }
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("nonsmooth") && msg.contains("detection"), "{msg}");
    }

    #[test]
    fn admits_transformed_projected_argument() {
        // mean = -projected TRANSFORMS the projection output — now SUPPORTED (the
        // compiler emits proj_grad = -1). The gate must ADMIT it (guards against
        // the old over-refusal).
        let mut m = base_model();
        for om in &mut m.observations {
            if om.name == "weekly_cases" {
                if let Likelihood::NegBinomial(nb) = &mut om.likelihood {
                    nb.mean = Diffable {
                        expr: Expr::un_op(UnOp::Neg, projected()),
                        grad: HashMap::new(),
                        proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: -1.0 }))),
                    };
                }
            }
        }
        let cm = compile(m);
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        preflight_gradient_ode(&cm, &params, &est(&["beta"]))
            .expect("a transformed `projected` argument (with emitted proj_grad) must be admitted");
    }

    #[test]
    fn refuses_nonsmooth_projection_argument() {
        // mean = floor(projected): a nonsmooth function of the projection output.
        // The WrtProjected pass emits proj_grad = Unsupported{NonsmoothState}, and
        // the gate refuses it (a genuine capability limit, unlike a smooth transform).
        let mut m = base_model();
        for om in &mut m.observations {
            if om.name == "weekly_cases" {
                if let Likelihood::NegBinomial(nb) = &mut om.likelihood {
                    nb.mean = Diffable {
                        expr: Expr::un_op(UnOp::Floor, projected()),
                        grad: HashMap::new(),
                        proj_grad: Some(DerivEntry::Unsupported {
                            node: "floor/ceil expression".to_string(),
                            code: UnsupportedReason::NonsmoothState,
                        }),
                    };
                }
            }
        }
        let msg = gate_err(m, &["beta"]);
        assert!(
            msg.contains("nonsmooth function of the projection output") && msg.contains("weekly_cases"),
            "{msg}"
        );
    }

    #[test]
    fn admits_fixed_param_with_unsupported_grad() {
        // The Unsupported grad is on a param that is NOT estimated → admitted
        // (its gradient is never taken). Guards against over-refusal.
        let mut m = base_model();
        m.transitions[0].rate_grad.insert(
            "beta".to_string(),
            DerivEntry::Unsupported {
                node: "time_func:seasonal".to_string(),
                code: UnsupportedReason::Lag,
            },
        );
        let cm = compile(m);
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        // Estimate gamma/k only; beta (the unsupported one) is fixed.
        preflight_gradient_ode(&cm, &params, &est(&["gamma", "k"]))
            .expect("a fixed param's unsupported gradient must not refuse the fit");
    }
}
