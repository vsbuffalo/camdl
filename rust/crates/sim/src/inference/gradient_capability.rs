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
//!   The PGAS-NUTS preflight (`pgas.rs`) and the ODE gradient gate both call it, so
//!   the two cannot drift.
//! - [`preflight_gradient_ode`] — the ODE gradient gate (`nl-lbfgs`): the common scan PLUS the
//!   checks specific to the deterministic forward-sensitivity path (a nonsmooth
//!   `rate_state_grad`, an adaptive integrator, a `dt`-in-rate model, a scheduled
//!   effect the sensitivity cannot cross, a parameterized initial condition, a
//!   `DerivedExpr` prevalence projection, a `projected`-transforming likelihood
//!   argument). One function, one list, run once before any gradient is taken.

use std::collections::{BTreeMap, HashSet};

use ir::deriv::{DerivEntry, UnsupportedReason};
use ir::expr::Expr;
use ir::observation::{Likelihood, Projection};
use ir::transition::DrawMethod;
use ir::Differentiable;

use crate::compiled_model::CompiledModel;
use crate::error::SimError;

/// Collect names of every `Param` referenced by an expression tree.
pub fn collect_param_refs(e: &Expr, out: &mut HashSet<String>) {
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
        Expr::PopSum(_) | Expr::Pop(_) | Expr::Const(_) | Expr::UnitLiteral(_) | Expr::ConstRef(_) | Expr::Time(_)
        | Expr::Dt(_)
        // gh#616: an anchor carries no parameter reference (its value comes from
        // the data), so it contributes nothing to a gradient's param set.
        | Expr::ObsAnchor(_)
        | Expr::TimeFunc(_) | Expr::ForcingBasis(_) | Expr::Projected(_) | Expr::ObsColumnRef(_) => {}
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
        Expr::PopSum(_) | Expr::Pop(_) | Expr::Const(_) | Expr::UnitLiteral(_) | Expr::ConstRef(_) | Expr::Time(_)
        | Expr::Dt(_) | Expr::ObsAnchor(_)
        | Expr::TimeFunc(_) | Expr::ForcingBasis(_) | Expr::ObsColumnRef(_) | Expr::BindingRef(_) => {}
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
/// uncovered on more than one surface. Shared by PGAS-NUTS and the ODE gradient gate.
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
    refused
}

/// The ODE gradient gate (§1h, `nl-lbfgs`): the common coverage scan PLUS the checks
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
            "ODE gradient (nl-lbfgs) cannot estimate parameter(s) whose gradient the compiler \
             could not emit for a rate, observation, or overdispersion term — the optimizer \
             would search along an incomplete (silently biased) gradient. Refused: {}. Use \
             `nl-sbplx` on `ode`, which needs no gradient, or fix these parameters.",
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
            "ODE gradient (nl-lbfgs) requires the state-Jacobian `rate_state_grad`, but this \
             model carries none on any transition — it was compiled with `camdlc \
             --no-state-grad` (or has no state-dependent rates). Recompile without \
             `--no-state-grad` to fit with `nl-lbfgs`, or use `nl-sbplx` on `ode`, which \
             needs no gradient."
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
                    "ODE gradient (nl-lbfgs): transition `{}` has a rate whose derivative with \
                     respect to a compartment is not smooth — it {}. The ODE forward \
                     sensitivity is undefined for such a rate. Reformulate the rate with a \
                     smooth expression, or use `nl-sbplx` on `ode`, which needs no gradient.",
                    t.name,
                    code.reason_message()
                )));
            }
        }
    }

    // ── Integrator: fixed-step RK4 only (§1c) ───────────────────────────────────
    if !matches!(m.simulation.integrator, ir::model::Integrator::Rk4) {
        return Err(SimError::Validation(
            "ODE gradient (nl-lbfgs) requires integrator = rk4 (fixed step); an adaptive \
             integrator (rk45) is refused — its step sequence is discontinuous in θ, \
             breaking the forward-sensitivity/loglik consistency the gradient method needs \
             (gh#275 §1c)."
                .to_string(),
        ));
    }
    // ── `dt`-in-rate (RUNTIME_DT): no augmented flow sensitivity (§1c) ──────────
    if model.required_capabilities().contains(crate::Capabilities::RUNTIME_DT) {
        return Err(SimError::Validation(
            "ODE gradient (nl-lbfgs) cannot differentiate a model that references `dt` in a \
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
            "ODE gradient (nl-lbfgs) does not support real-valued compartments (ODE-equation \
             compartments): the forward sensitivity is integer-compartment only, so a real \
             compartment's ∂x/∂θ is not tracked (gh#275 §1c). Use `nl-sbplx` on `ode`, \
             which needs no gradient, for models with real compartments."
                .to_string(),
        ));
    }
    // ── Scheduled effects: the event-jump sensitivity (§1g) ─────────────────────
    preflight_scheduled_effects(model, params, estimated)?;
    // ── Initial condition: the ∂init/∂θ seed (`ic_grad`, §1c C-seed) ────────────
    // A parameterized IC whose expression involves an estimated parameter needs the
    // forward sensitivity seeded at S(t_start) = ∂(initial_state)/∂θ. Admit it only
    // when the compiler emitted a smooth, int-compartment `ic_grad` entry for every
    // (estimated param → IC compartment) pair — otherwise the seed would silently
    // drop that parameter's contribution and the optimizer would search along a gradient
    // that is zero exactly where the initial condition matters (Risk #1).
    //
    // The parameter set scanned per compartment is TRANSITIVE along the init
    // dependency graph (gh#733): `init { I = I0   S = N0 - I }` makes `I0` reach
    // `S`'s initial value even though `S`'s own expression never names it, and
    // the compiler's `ic_grad` — which differentiates each entry with the
    // referenced entries substituted in — emits ∂S/∂I0 accordingly. Scanning
    // only the syntactic parameters of `S` would leave that pair unchecked, so a
    // nondifferentiable ∂S/∂I0 could reach the seed as `Unsupported`.
    {
        // Direct + inherited parameter references, filled in dependency order so
        // each entry's dependencies are already resolved when it is reached.
        let mut reaching: std::collections::HashMap<&str, HashSet<String>> = std::collections::HashMap::new();
        let order = ir::init_order::topo(&m.initial_conditions, &m.bindings)
            .map_err(|cycle| SimError::Validation(format!(
                "initial conditions reference each other in a cycle: {}",
                cycle.join(" -> ")
            )))?;
        for &i in &order {
            let (comp, spec) = m.initial_conditions.0
                .get_index(i)
                .expect("init_order indexes initial_conditions");
            // The MEAN expression: the ODE path starts every compartment at its
            // law's mean, and `ic_grad` differentiates that same expression, so
            // it is the one whose parameter reach must be covered.
            let expr = &spec.mean_expr();
            // A binding could hide a param reference the scan below does not
            // descend into — refuse rather than risk a silent-zero seed.
            if expr_has_binding_ref(expr) {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nl-lbfgs): the initial condition of `{comp}` references a \
                     shared binding, which the gradient path does not yet trace for the \
                     ∂init/∂θ seed (gh#275 §1c). Inline the expression, or use \
                     `nl-sbplx` on `ode`, which needs no gradient."
                )));
            }
            let mut refs = HashSet::new();
            collect_param_refs(expr, &mut refs);
            for dep in spec.exprs().into_iter()
                .flat_map(|e| ir::init_order::deps(e, &m.bindings))
            {
                if let Some(inherited) = reaching.get(dep.as_str()) {
                    refs.extend(inherited.iter().cloned());
                }
            }
            for pname in refs.iter().filter(|p| estimated.contains(p.as_str())) {
                // Real-compartment forward sensitivity is a separate follow-up.
                let global = model.comp_index.get(comp.as_str()).copied().ok_or_else(|| {
                    SimError::Validation(format!(
                        "ODE gradient (nl-lbfgs): initial condition names unknown compartment \
                         `{comp}`."
                    ))
                })?;
                if model.global_to_int[global].is_none() {
                    return Err(SimError::Validation(format!(
                        "ODE gradient (nl-lbfgs): parameter-dependent initial condition on real \
                         compartment `{comp}` — real-compartment forward sensitivity is not \
                         yet supported (gh#275 §1c). Use `nl-sbplx` on `ode`, which needs no gradient."
                    )));
                }
                match m.ic_grad.get(comp).and_then(|pm| pm.get(pname.as_str())) {
                    Some(ir::deriv::DerivEntry::Grad(_)) => {}
                    Some(ir::deriv::DerivEntry::Unsupported { code, .. }) => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nl-lbfgs): ∂(initial {comp})/∂{pname} is not \
                             differentiable — it {}. Reformulate the initial condition \
                             with a smooth expression, or use `nl-sbplx` on `ode`, which needs no gradient.",
                            code.reason_message()
                        )));
                    }
                    None => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nl-lbfgs): estimated parameter `{pname}` enters the \
                             initial condition of `{comp}`, but there is no usable \
                             ∂(initial {comp})/∂{pname} — either the initial condition is not \
                             differentiable in `{pname}` (e.g. a floor/round of it), the two \
                             occurrences cancel, or this model was compiled before ic_grad \
                             emission (gh#275 §1c). Its gradient would be silently zero. \
                             Reformulate the initial condition, recompile, or use \
                             `nl-sbplx` on `ode`, which needs no gradient."
                        )));
                    }
                }
            }
            reaching.insert(comp.as_str(), refs);
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
                    "ODE gradient (nl-lbfgs): the DerivedExpr projection of observation stream `{}` \
                     carries no ∂projection/∂compartment — its factor-2 gradient would be \
                     silently zero. Recompile with a camdlc that emits projection_state_grad \
                     (gh#275 §1h). (If the projection genuinely does not depend on compartment \
                     state, it is not an identifiable observation of the trajectory — it cannot \
                     inform a gradient fit; use `nl-sbplx` on `ode`, which needs no gradient.)",
                    om.name
                )));
            }
            for (comp, entry) in om.projection_state_grad.iter() {
                if let DerivEntry::Unsupported { code, .. } = entry {
                    return Err(SimError::Validation(format!(
                        "ODE gradient (nl-lbfgs): the projection of observation stream `{}` is a \
                         nonsmooth function of compartment `{}` — it {}. The ∂projection/∂state \
                         chain is undefined. Reformulate the projection with a smooth \
                         expression, or use `nl-sbplx` on `ode`, which needs no gradient.",
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
                    "ODE gradient (nl-lbfgs): observation stream `{}` argument `{}` is a nonsmooth \
                     function of the projection output — it {}. Reformulate it with a smooth \
                     expression, or use `nl-sbplx` on `ode`, which needs no gradient.",
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
                    "ODE gradient (nl-lbfgs): observation stream `{}` has a Binomial/BetaBinomial \
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

/// The scheduled-effect half of the gate (§1g). The ODE gradient path applies
/// every effect the run fires together with its sensitivity jump
/// `S⁺ = ∂Φ/∂x·S⁻ + ∂Φ/∂θ` (`effects::SensBlock`), which is well defined for a
/// fixed-time effect whose actions are smooth in state and parameters. This
/// refuses, by name, each effect it is not defined for:
///
/// - a **reactive** intervention — its fire time depends on the observed data
///   and the trajectory, an implicit `t_e(x)` the fixed-grid sensitivity does
///   not carry (gh#204);
/// - a fire time that reads an **estimated parameter** (`at [t_seed]`) — a
///   θ-dependent event time carries a boundary term `−(ẋ⁺ − ẋ⁻)·∂t_e/∂θ` the
///   fixed-grid RK4 does not compute; emitting the jump alone would be a
///   silently wrong gradient;
/// - a `transfer(count = …)` — its jump `min(count, source)` is not
///   differentiable in the source compartment;
/// - an amount that **reads compartment state** — the compiler emits
///   `∂amount/∂θ` but not `∂amount/∂state`, so the jump's `∂Φ/∂x` would be
///   silently incomplete (`set I = I + 10`; write `add(I, 10)` instead);
/// - an amount whose `∂amount/∂θ` the compiler could not emit for an estimated
///   parameter (a serialized `Unsupported`), or has no entry for one it reads —
///   a nonsmooth function of it, a cancelled dependence, or an IR compiled
///   before amount gradients existed (ir/VERSION 0.45); either way its
///   contribution would be silently zero.
///
/// Only effects the run will fire are checked — the plan over
/// `[t_start, ∞)`, both its origin batch and its later instants, so an effect
/// at `t_start` counts and one after the run is checked too (the fit's last
/// observation is not known here; the conservative side). A toggleable
/// `interventions {}` entry the fit leaves disabled is not in `model` at all
/// (the scenario filter removes it first).
fn preflight_scheduled_effects(
    model: &CompiledModel,
    params: &[f64],
    estimated: &HashSet<&str>,
) -> Result<(), SimError> {
    use ir::intervention::Action;
    let m = &model.model;
    let alt = "Use `nl-sbplx` on `ode`, which needs no gradient.";
    let noun = |iv: &ir::intervention::Intervention| {
        if iv.kind.is_event() { "event" } else { "intervention" }
    };

    for iv in &m.interventions {
        if iv.fire.is_reactive() {
            return Err(SimError::Validation(format!(
                "ODE gradient (nl-lbfgs): `{}` is a reactive intervention (it fires when \
                 an `observed(...)` quantity crosses a threshold), so its fire time depends \
                 on the data and the trajectory — a dependence the fixed-grid forward \
                 sensitivity does not carry (gh#275 §1g). {alt}",
                iv.name
            )));
        }
    }

    let plan = crate::effect_plan::EffectPlan::build(
        model,
        params,
        m.simulation.t_start,
        f64::INFINITY,
        crate::effect_plan::FireClock::Exact,
        crate::effect_plan::Origin::Init,
    )?;
    let mut firing: std::collections::BTreeSet<usize> = plan.at_origin().iter().copied().collect();
    for i in 0..plan.times().len() {
        firing.extend(plan.batch(i).iter().copied());
    }

    for iv_idx in firing {
        let iv = &m.interventions[iv_idx];
        // A fire time that reads an estimated parameter.
        if let Some(exprs) = model.resolved.intervention_at_time_exprs[iv_idx].as_deref() {
            let wanted = |i: usize| estimated.contains(m.parameters[i].name.as_str());
            if let Some(p) = exprs.iter().find_map(|e| model.first_param_read(e, &wanted)) {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nl-lbfgs): estimated parameter `{}` sets the fire time of \
                     {} `{}`. A θ-dependent event time adds a boundary term \
                     −(ẋ⁺ − ẋ⁻)·∂t_e/∂θ to the sensitivity that the fixed-grid integration \
                     does not compute, so the gradient would be silently wrong (gh#275 §1g). \
                     Fix `{}`, or {}",
                    m.parameters[p].name, noun(iv), iv.name, m.parameters[p].name,
                    alt.to_string().replacen("Use", "use", 1)
                )));
            }
        }
        for (action_idx, action) in iv.actions.iter().enumerate() {
            let (verb, amount, grad) = match action {
                Action::Add(a) => ("add", &a.count, &a.count_grad),
                Action::Set(s) => ("set", &s.value, &s.value_grad),
                Action::FractionTransfer(ft) => ("transfer(fraction = …)", &ft.fraction, &ft.fraction_grad),
                Action::AbsoluteTransfer(at) => {
                    return Err(SimError::Validation(format!(
                        "ODE gradient (nl-lbfgs): {} `{}` moves a fixed count from `{}` to `{}` \
                         (`transfer(count = …)`). Its jump moves min(count, source), which is \
                         not differentiable in the source compartment, so the forward \
                         sensitivity cannot cross it (gh#275 §1g). Write the transfer as a \
                         fraction of the source, or {}",
                        noun(iv), iv.name, at.src, at.dst,
                        alt.to_string().replacen("Use", "use", 1)
                    )));
                }
                Action::GroupedTransfer(gt) => match &gt.amount {
                    ir::intervention::GroupedAmount::Fraction { fraction, fraction_grad } => {
                        ("transfer(fraction = …)", fraction, fraction_grad)
                    }
                    ir::intervention::GroupedAmount::Count { .. } => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nl-lbfgs): {} `{}` moves a fixed count out of the                              stages of a staged compartment ({}; `transfer(count = …)`). Its                              jump moves min(count, source), which is not differentiable in                              the source compartment, so the forward sensitivity cannot cross                              it (gh#275 §1g). Write the transfer as a fraction of the source,                              or {}",
                            noun(iv), iv.name, crate::intervention::grouped_pairs_label(gt),
                            alt.to_string().replacen("Use", "use", 1)
                        )));
                    }
                },
            };
            // An amount that reads compartment state (directly or through a
            // hoisted binding): ∂amount/∂state is not emitted.
            if crate::resolved_expr::references_state(
                &model.resolved.intervention_exprs[iv_idx][action_idx],
            ) {
                return Err(SimError::Validation(format!(
                    "ODE gradient (nl-lbfgs): the amount of {} `{}`'s `{}` reads compartment \
                     state. The compiler emits ∂amount/∂θ but not ∂amount/∂state, so the \
                     jump's ∂Φ/∂x would be silently incomplete (gh#275 §1g). Write the \
                     amount in parameters and constants — `add(I, 10)` rather than \
                     `I = I + 10` — or {}",
                    noun(iv), iv.name, verb,
                    alt.to_string().replacen("Use", "use", 1)
                )));
            }
            // Every estimated parameter the amount reads has a usable ∂amount/∂θ.
            let mut refs = HashSet::new();
            collect_param_refs(amount, &mut refs);
            let mut refs: Vec<String> = refs.into_iter().filter(|p| estimated.contains(p.as_str())).collect();
            refs.sort();
            for pname in refs {
                match grad.get(pname.as_str()) {
                    Some(DerivEntry::Grad(_)) => {}
                    Some(DerivEntry::Unsupported { code, .. }) => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nl-lbfgs): ∂(amount of {} `{}`)/∂{pname} is not \
                             differentiable — it {}. Reformulate the amount with a smooth \
                             expression, or {}",
                            noun(iv), iv.name, code.reason_message(),
                            alt.to_string().replacen("Use", "use", 1)
                        )));
                    }
                    None => {
                        return Err(SimError::Validation(format!(
                            "ODE gradient (nl-lbfgs): estimated parameter `{pname}` enters the \
                             amount of {} `{}` (`{}`), but there is no usable \
                             ∂amount/∂{pname} — either the amount is not differentiable in \
                             `{pname}` (e.g. a floor/round of it), the dependence cancels, or \
                             this IR was compiled before amount gradients were emitted \
                             (ir/VERSION 0.45, gh#275 §1g). Its contribution would be \
                             silently zero. Reformulate the amount, recompile, or {}",
                            noun(iv), iv.name, verb,
                            alt.to_string().replacen("Use", "use", 1)
                        )));
                    }
                }
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
        model.initial_conditions = ir::model::InitialConditions::constants([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]);
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
        // rate_state_grad on every transition (gh#439). The ODE gradient gate must
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
    fn accepts_a_zero_inflated_stream_whose_args_are_differentiated() {
        // The zero-inflated family used to be refused wholesale, because it
        // carried no emitted gradient at all and a rate param reaching it only
        // through the trajectory would otherwise hit the `unreachable!` grad
        // backstop mid-integration. Its three arguments are now `Diffable`, so
        // the ordinary coverage scan in (a) applies and the family needs no
        // special case. `beta` reaches this stream only through the trajectory
        // (`mean = projected`), which is exactly the case the old model-level
        // refusal existed to catch — so its acceptance here is the assertion
        // that the gradient really is carried, not merely that a check was
        // deleted.
        let mut m = base_model();
        for om in &mut m.observations {
            if om.name == "weekly_cases" {
                om.likelihood = Likelihood::ZeroInflatedNegBinomial(
                    ir::observation::ZeroInflatedNegBinomialLikelihood {
                        mean: Diffable::new(projected()),
                        dispersion: Diffable::new(Expr::Param(ParamExpr {
                            param: "k".to_string(),
                        })),
                        pi: Diffable::new(Expr::Const(ConstExpr { value: 0.3 })),
                    },
                );
            }
        }
        let refused = scan_unsupported_gradients(&m, &est(&["beta"]));
        assert!(
            refused.is_empty(),
            "a differentiated zero-inflated stream must not be refused: {refused:?}"
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

    // ── §1g: scheduled effects ──────────────────────────────────────────────────

    use ir::intervention::{Action, AddAction, FireSource, FractionTransfer, Intervention,
        InterventionKind, InterventionSchedule, SetAction};

    /// One scheduled event named `name` with the given actions, firing at `times`.
    fn event(name: &str, times: Vec<f64>, actions: Vec<Action>) -> Intervention {
        Intervention {
            name: name.into(),
            base_name: None,
            fire: FireSource::Scheduled(InterventionSchedule::AtTimes(times)),
            actions,
            kind: InterventionKind::Event,
        }
    }

    fn add_param(m: &mut ir::Model, name: &str, value: f64) {
        m.parameters.push(ir::parameter::Parameter {
            name: name.into(),
            value: ir::parameter::ParamValue::Fixed { value },
            param_kind: None,
            param_dim: None,
        });
    }

    fn gate_ok(model: ir::Model, estimated: &[&str]) {
        let cm = compile(model);
        let params = cm.default_params.clone();
        preflight_gradient_ode(&cm, &params, &est(estimated))
            .unwrap_or_else(|e| panic!("the gate must admit this model, but refused: {e}"));
    }

    /// A fixed-fraction transfer at fixed times — the yearly school-entry /
    /// ageing transfer — is the case the sensitivity jump exists for. Admitted,
    /// including an instance at t_start (the origin batch, which the gradient
    /// path applies before the first row, as `run_ode` does).
    #[test]
    fn admits_fixed_time_fixed_fraction_transfers_including_at_t_start() {
        let mut m = base_model();
        let t0 = m.simulation.t_start;
        m.interventions = vec![event("ageing", vec![t0, 10.5, 20.5], vec![Action::FractionTransfer(
            FractionTransfer {
                src: "S".into(), dst: "R".into(), fraction: Expr::const_(0.1),
                fraction_grad: Default::default(),
            },
        )])];
        gate_ok(m, &["beta", "gamma", "k"]);
    }

    /// A parameterised amount is admitted when the compiler emitted its
    /// ∂amount/∂θ for every estimated parameter it reads.
    #[test]
    fn admits_an_estimated_amount_with_an_emitted_gradient() {
        let mut m = base_model();
        add_param(&mut m, "seed", 5.0);
        let one = DerivEntry::Grad(Expr::const_(1.0));
        m.interventions = vec![event("import", vec![14.0], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::param("seed"),
            count_grad: HashMap::from([("seed".to_string(), one)]),
        })])];
        gate_ok(m, &["beta", "seed"]);
    }

    /// `transfer(count = …)` moves min(count, source): nonsmooth, refused by name.
    #[test]
    fn refuses_a_fixed_count_transfer_by_name() {
        let mut m = base_model();
        m.interventions = vec![event("evacuate", vec![14.0], vec![Action::AbsoluteTransfer(
            ir::intervention::AbsoluteTransfer {
                src: "I".into(), dst: "R".into(), count: Expr::const_(3.0),
            },
        )])];
        let msg = gate_err(m, &["beta"]);
        assert!(
            msg.contains("evacuate") && msg.contains("transfer(count") && msg.contains("min(count, source)"),
            "{msg}"
        );
    }

    fn grouped(amount: ir::intervention::GroupedAmount) -> Action {
        Action::GroupedTransfer(ir::intervention::GroupedTransfer {
            pairs: vec![("E".into(), "R".into()), ("I".into(), "R".into())],
            amount,
        })
    }

    /// A grouped transfer of a fraction is treated as a fraction transfer is:
    /// admitted with a fixed fraction, and with an estimated one only when its
    /// ∂fraction/∂θ was emitted.
    #[test]
    fn treats_a_grouped_fraction_transfer_as_a_fraction_transfer() {
        use ir::intervention::GroupedAmount;
        let mut m = base_model();
        m.interventions = vec![event("clear", vec![14.0], vec![grouped(GroupedAmount::Fraction {
            fraction: Expr::const_(0.1),
            fraction_grad: Default::default(),
        })])];
        gate_ok(m, &["beta", "gamma", "k"]);

        let mut m = base_model();
        add_param(&mut m, "cov", 0.3);
        m.interventions = vec![event("clear", vec![14.0], vec![grouped(GroupedAmount::Fraction {
            fraction: Expr::param("cov"),
            fraction_grad: HashMap::from([("cov".to_string(), DerivEntry::Grad(Expr::const_(1.0)))]),
        })])];
        gate_ok(m, &["beta", "cov"]);

        let mut m = base_model();
        add_param(&mut m, "cov", 0.3);
        m.interventions = vec![event("clear", vec![14.0], vec![grouped(GroupedAmount::Fraction {
            fraction: Expr::param("cov"),
            fraction_grad: Default::default(),
        })])];
        let msg = gate_err(m, &["beta", "cov"]);
        assert!(msg.contains("clear") && msg.contains("no usable") && msg.contains("cov"), "{msg}");
    }

    /// A grouped transfer of a count is refused by name, as a plain
    /// `transfer(count = …)` is.
    #[test]
    fn refuses_a_grouped_count_transfer_by_name() {
        let mut m = base_model();
        m.interventions = vec![event("evacuate", vec![14.0], vec![grouped(
            ir::intervention::GroupedAmount::Count { count: Expr::const_(3.0) },
        )])];
        let msg = gate_err(m, &["beta"]);
        assert!(
            msg.contains("evacuate") && msg.contains("transfer(count") && msg.contains("min(count, source)")
                && msg.contains("E -> R .. I -> R"),
            "{msg}"
        );
    }

    /// An amount that reads state (`set I = I + 10`) has no emitted
    /// ∂amount/∂state: refused, naming the event and the rewrite.
    #[test]
    fn refuses_a_state_dependent_amount() {
        let mut m = base_model();
        m.interventions = vec![event("pulse", vec![14.0], vec![Action::Set(SetAction {
            compartment: "I".into(),
            value: Expr::bin_op(ir::expr::BinOp::Add, Expr::pop("I"), Expr::const_(10.0)),
            value_grad: Default::default(),
        })])];
        let msg = gate_err(m, &["beta"]);
        assert!(
            msg.contains("pulse") && msg.contains("reads compartment state") && msg.contains("add(I, 10)"),
            "{msg}"
        );
    }

    /// An estimated parameter in an amount with no emitted ∂amount/∂θ (an IR
    /// compiled before 0.45, or a nonsmooth dependence) would contribute a
    /// silently-zero jump term: refused.
    #[test]
    fn refuses_an_estimated_amount_without_a_gradient_entry() {
        let mut m = base_model();
        add_param(&mut m, "seed", 5.0);
        m.interventions = vec![event("import", vec![14.0], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::param("seed"),
            count_grad: Default::default(),
        })])];
        let msg = gate_err(m, &["beta", "seed"]);
        assert!(msg.contains("seed") && msg.contains("import") && msg.contains("silently zero"), "{msg}");
    }

    /// The same amount with a serialized `Unsupported` entry: refused with the
    /// compiler's reason.
    #[test]
    fn refuses_an_estimated_amount_with_an_unsupported_gradient() {
        let mut m = base_model();
        add_param(&mut m, "seed", 5.0);
        m.interventions = vec![event("import", vec![14.0], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::param("seed"),
            count_grad: HashMap::from([(
                "seed".to_string(),
                DerivEntry::Unsupported { node: "floor(seed)".into(), code: UnsupportedReason::NonsmoothState },
            )]),
        })])];
        let msg = gate_err(m, &["seed"]);
        assert!(msg.contains("seed") && msg.contains("not differentiable"), "{msg}");
    }

    /// A fixed parameter in an amount needs no gradient (its column is never
    /// taken): admitted even without an entry. Guards against over-refusal.
    #[test]
    fn admits_a_fixed_amount_parameter_without_a_gradient_entry() {
        let mut m = base_model();
        add_param(&mut m, "seed", 5.0);
        m.interventions = vec![event("import", vec![14.0], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::param("seed"),
            count_grad: Default::default(),
        })])];
        gate_ok(m, &["beta"]);
    }

    /// A parametric fire time `at [t_seed]`: admitted while `t_seed` is fixed,
    /// refused by name once it is estimated (the boundary term the fixed grid
    /// does not carry).
    #[test]
    fn a_parametric_fire_time_is_refused_only_when_estimated() {
        let mut m = base_model();
        add_param(&mut m, "t_seed", 14.0);
        let mut iv = event("import", vec![], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::const_(5.0),
            count_grad: Default::default(),
        })]);
        iv.fire = FireSource::Scheduled(InterventionSchedule::AtTimesExpr(vec![Expr::param("t_seed")]));
        m.interventions = vec![iv];
        gate_ok(m.clone(), &["beta"]);
        let msg = gate_err(m, &["beta", "t_seed"]);
        assert!(msg.contains("t_seed") && msg.contains("fire time") && msg.contains("import"), "{msg}");
    }

    /// A reactive intervention's fire time depends on the data: refused by name.
    #[test]
    fn refuses_a_reactive_intervention() {
        use ir::intervention::{CmpOp, ObsReducer, ReactiveTrigger, TriggerExpr, TriggerQuantity,
            TriggerThreshold};
        let mut m = base_model();
        m.interventions = vec![Intervention {
            name: "sia".into(),
            base_name: None,
            fire: FireSource::Reactive(ReactiveTrigger {
                when_: TriggerExpr::Cmp {
                    lhs: TriggerQuantity::Observed {
                        stream: "weekly_cases".into(), window: None, reducer: ObsReducer::Latest,
                    },
                    op: CmpOp::Ge,
                    rhs: TriggerThreshold::Const(50.0),
                },
                after: 0.0,
                once: true,
                cooldown: None,
            }),
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(), dst: "R".into(), fraction: Expr::const_(0.5),
                fraction_grad: Default::default(),
            })],
            kind: InterventionKind::Scenario,
        }];
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("sia") && msg.contains("reactive"), "{msg}");
    }

    /// An initial condition drawn from a law (`R ~ binomial(n = N0, p = p_imm)`)
    /// enters the ODE path through its MEAN `N0·p_imm`, which the compiler
    /// differentiates into `ic_grad`; the gate admits it on that seed. (The
    /// value path rounds the mean to an integer; the gradient path uses it
    /// unrounded — see `CompiledModel::initial_state_continuous`.)
    #[test]
    fn admits_a_drawn_initial_condition_through_its_mean_seed() {
        use ir::model::{InitCountLaw, InitSpec};
        let mut m = base_model();
        add_param(&mut m, "p_imm", 0.2);
        let mut ic = m.initial_conditions.0.clone();
        ic.insert(
            "R".to_string(),
            InitSpec::Count(InitCountLaw::Binomial(ir::observation::BinomialLikelihood {
                n: Expr::param("N0"),
                p: Diffable::new(Expr::param("p_imm")),
            })),
        );
        m.initial_conditions = ir::model::InitialConditions(ic);
        // ∂(N0·p_imm)/∂p_imm = N0, as `differentiate_ic` emits over the mean.
        m.ic_grad = HashMap::from([(
            "R".to_string(),
            HashMap::from([("p_imm".to_string(), DerivEntry::Grad(Expr::param("N0")))]),
        )]);
        gate_ok(m, &["beta", "p_imm"]);
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
        m.initial_conditions = ir::model::InitialConditions::exprs([
            ("S".to_string(), Expr::Const(ConstExpr { value: 9990.0 })),
            ("E".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
            ("I".to_string(), Expr::Param(ParamExpr { param: "I0".to_string() })),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ]);
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

    /// gh#733: `S`'s own expression never names `I0`, but `I0` reaches `S`'s
    /// initial value through `I` — so the ic_grad completeness check has to see
    /// the pair. Without the transitive scan the missing ∂S/∂I0 would go
    /// unchecked and the seed would be silently short a term.
    #[test]
    fn refuses_when_an_inherited_ic_parameter_has_no_seed() {
        let mut m = base_model();
        m.initial_conditions = ir::model::InitialConditions::exprs([
            ("I".to_string(), Expr::param("I0")),
            (
                "S".to_string(),
                Expr::bin_op(ir::expr::BinOp::Sub, Expr::const_(1000.0), Expr::pop("I")),
            ),
        ]);
        // Seed emitted for I but NOT for S — exactly the gap the scan must catch.
        m.ic_grad = HashMap::from([(
            "I".to_string(),
            HashMap::from([(
                "I0".to_string(),
                DerivEntry::Grad(Expr::const_(1.0)),
            )]),
        )]);
        let msg = gate_err(m, &["I0"]);
        assert!(msg.contains("I0") && msg.contains("`S`"), "{msg}");
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
