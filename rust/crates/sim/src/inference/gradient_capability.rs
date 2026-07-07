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
    // ── Scheduled effects: event-jump sensitivity is a follow-up (§1g) ──────────
    if !EffectTimes::from_model(model, params)?.into_vec().is_empty() {
        return Err(SimError::Validation(
            "ODE gradient (nuts) does not yet support scheduled interventions or events: \
             their event-jump sensitivity is a follow-up (gh#275 §1g). Use gradient-free \
             `mh` on `ode` for models with effects."
                .to_string(),
        ));
    }
    // ── Initial condition: the ic_grad seed is a follow-up (§1c) ────────────────
    if !matches!(m.initial_conditions, ir::model::InitialConditions::Explicit(_)) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) v1 supports only explicit (constant) initial conditions. A \
             parameterized initial condition may depend on an estimated parameter, whose \
             forward sensitivity must be seeded at S(t_start) = ∂(initial_state)/∂θ (the \
             `ic_grad` seed, gh#275 §1c) — a follow-up. Use gradient-free `mh` on `ode`."
                .to_string(),
        ));
    }
    // ── Observation projections + likelihood arguments ──────────────────────────
    for om in &m.observations {
        // A DerivedExpr (nonlinear) prevalence projection needs ∂projection/∂state,
        // a compiler-emitted object not yet built (§1h).
        if let Projection::DerivedExpr(_) = om.projection {
            return Err(SimError::Validation(format!(
                "ODE gradient (nuts) does not support the DerivedExpr (nonlinear) projection \
                 of observation stream `{}`: its ∂projection/∂state is a compiler-emitted \
                 object not yet built (gh#275 §1h). Use gradient-free `mh` on `ode`.",
                om.name
            )));
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
    fn refuses_adaptive_integrator() {
        let mut m = base_model();
        m.simulation.integrator = ir::model::Integrator::Rk45 { atol: None, rtol: None };
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("rk4") && msg.contains("rk45"), "{msg}");
    }

    #[test]
    fn refuses_parameterized_initial_condition() {
        let mut m = base_model();
        m.initial_conditions = ir::model::InitialConditions::Parameterized(HashMap::from([(
            "I".to_string(),
            Expr::Param(ParamExpr { param: "I0".to_string() }),
        )]));
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("initial condition") && msg.contains("ic_grad"), "{msg}");
    }

    #[test]
    fn refuses_derivedexpr_projection() {
        let mut m = base_model();
        // A nonlinear prevalence projection on the detection stream.
        for om in &mut m.observations {
            if om.name == "detection" {
                om.projection = ir::observation::Projection::DerivedExpr(Expr::bin_op(
                    ir::expr::BinOp::Div,
                    Expr::pop("I"),
                    Expr::pop("S"),
                ));
            }
        }
        let msg = gate_err(m, &["beta"]);
        assert!(msg.contains("DerivedExpr") && msg.contains("detection"), "{msg}");
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
