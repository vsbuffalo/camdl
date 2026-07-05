//! NUTS guard: refuse to estimate a parameter that drives only a forcing or
//! inline-table coefficient (proposal `2026-06-09-const-parametric-forcing.md`
//! §4/§6).
//!
//! The value half made forcing/table coefficients live, so IF2 and the
//! bootstrap particle filter (gradient-free) now estimate such a parameter
//! correctly. NUTS still cannot: the compiler's autodiff (`autodiff.ml`)
//! differentiates `TimeFunc` and `TableLookup` to `Const 0.0`, so a parameter
//! referenced ONLY inside a coefficient has an identically-zero dynamics
//! gradient — NUTS would propose against a flat surface and silently
//! mis-sample. Until the gradient half emits those derivatives, reject such a
//! fit loudly rather than return a garbage posterior.
//!
//! Scope (matching §6): `body_refs` are the rate / observation / initial-value
//! expressions; `coeff_refs` are forcing-coefficient and inline-table-value
//! expressions. A parameter in both is not flagged — the body gradient carries
//! it. The coefficient sub-expressions live in `model.time_functions[*].kind`
//! and `model.tables[*]`, not in the rate AST, so this needs its own traversal.

use std::collections::HashSet;

use ir::deriv::DerivEntry;
use ir::expr::Expr;
use ir::model::InitialConditions;
use ir::observation::{Likelihood, Projection};
use ir::time_func::TimeFuncKind;
use ir::transition::DrawMethod;

/// Collect parameter names referenced anywhere in `e`.
fn collect(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Param(p) => {
            out.insert(p.param.clone());
        }
        Expr::BinOp(w) => {
            collect(&w.bin_op.left, out);
            collect(&w.bin_op.right, out);
        }
        Expr::UnOp(w) => collect(&w.un_op.arg, out),
        Expr::Cond(w) => {
            collect(&w.cond.pred, out);
            collect(&w.cond.then, out);
            collect(&w.cond.else_, out);
        }
        Expr::TableLookup(w) => {
            // The lookup INDEX is a body sub-expression; the table's VALUES are
            // coefficients and are collected separately from `model.tables`.
            for i in &w.table_lookup.indices {
                collect(i, out);
            }
        }
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect(t, out);
            }
        }
        Expr::UncheckedDim(w) => collect(&w.unchecked_dim.inner, out),
        // Leaves / non-param nodes. A `BindingRef` body is state-only
        // (param-free, enforced at `CompiledModel::new`), so it adds no refs.
        Expr::Const(_)
        | Expr::Pop(_)
        | Expr::PopSum(_)
        | Expr::Time(_)
        | Expr::Dt(_)
        | Expr::TimeFunc(_)
        | Expr::Projected(_)
        | Expr::ObsColumnRef(_) => {}
        // A `BindingRef` body is state-only (param-free, enforced at
        // `CompiledModel::new`), so it adds no refs.
        Expr::BindingRef(_) => {}
        // gh#272 LICM: a per-eval body IS param-carrying. This guard runs on the
        // pre-LICM IR (no PerEvalRef present), so the leaf is correct today. NOTE
        // for the stochastic phase: if this guard is ever run on post-LICM IR, it
        // must traverse the per-eval body (via `per_eval_bindings`) to see the
        // params it carries, or a hoisted coefficient param would be missed.
        Expr::PerEvalRef(_) => {}
    }
}

/// Collect parameters in a likelihood's coefficient expressions.
fn collect_likelihood(lik: &Likelihood, out: &mut HashSet<String>) {
    match lik {
        Likelihood::Poisson(p) => collect(&p.rate, out),
        Likelihood::NegBinomial(nb) => {
            collect(&nb.mean, out);
            collect(&nb.dispersion, out);
        }
        Likelihood::Normal(n) => {
            collect(&n.mean, out);
            collect(&n.sd, out);
        }
        Likelihood::Binomial(b) => {
            collect(&b.n, out);
            collect(&b.p, out);
        }
        Likelihood::BetaBinomial(bb) => {
            collect(&bb.n, out);
            collect(&bb.alpha, out);
            collect(&bb.beta, out);
        }
        Likelihood::Bernoulli(b) => collect(&b.p, out),
    }
}

/// Collect parameters in a forcing's scalar coefficient expressions.
fn collect_forcing(kind: &TimeFuncKind, out: &mut HashSet<String>) {
    match kind {
        TimeFuncKind::Sinusoidal(s) => {
            collect(&s.amplitude, out);
            collect(&s.period, out);
            collect(&s.phase, out);
            collect(&s.baseline, out);
        }
        TimeFuncKind::Piecewise(p) => {
            p.breakpoints.iter().for_each(|e| collect(e, out));
            p.values.iter().for_each(|e| collect(e, out));
        }
        TimeFuncKind::Interpolated(i) => {
            i.times.iter().for_each(|e| collect(e, out));
            i.values.iter().for_each(|e| collect(e, out));
        }
        TimeFuncKind::Periodic(p) => {
            collect(&p.period, out);
            p.values.iter().for_each(|e| collect(e, out));
        }
        TimeFuncKind::Fourier(f) => {
            collect(&f.period, out);
            f.harmonics.iter().for_each(|(a, b)| {
                collect(a, out);
                collect(b, out);
            });
        }
        TimeFuncKind::PeriodicSpline(ps) => {
            collect(&ps.period, out);
            ps.coefs.iter().for_each(|e| collect(e, out));
        }
    }
}

/// Collect the names of every forcing (`TimeFunc`) and inline table
/// (`TableLookup`) referenced anywhere in `e`. Used to partition forcings/tables
/// into coeff_guard's domain (referenced by a rate/IC) versus the obs-gradient
/// preflight's domain (referenced only by an observation) — proposal §4.4.
fn collect_forcing_table_refs(e: &Expr, forcings: &mut HashSet<String>, tables: &mut HashSet<String>) {
    match e {
        Expr::TimeFunc(w) => {
            forcings.insert(w.time_func.name.clone());
        }
        Expr::TableLookup(w) => {
            tables.insert(w.table_lookup.table.clone());
            for i in &w.table_lookup.indices {
                collect_forcing_table_refs(i, forcings, tables);
            }
        }
        Expr::BinOp(w) => {
            collect_forcing_table_refs(&w.bin_op.left, forcings, tables);
            collect_forcing_table_refs(&w.bin_op.right, forcings, tables);
        }
        Expr::UnOp(w) => collect_forcing_table_refs(&w.un_op.arg, forcings, tables),
        Expr::Cond(w) => {
            collect_forcing_table_refs(&w.cond.pred, forcings, tables);
            collect_forcing_table_refs(&w.cond.then, forcings, tables);
            collect_forcing_table_refs(&w.cond.else_, forcings, tables);
        }
        Expr::Reduce(w) => {
            for t in &w.reduce {
                collect_forcing_table_refs(t, forcings, tables);
            }
        }
        Expr::UncheckedDim(w) => collect_forcing_table_refs(&w.unchecked_dim.inner, forcings, tables),
        Expr::Const(_)
        | Expr::Param(_)
        | Expr::Pop(_)
        | Expr::PopSum(_)
        | Expr::Time(_)
        | Expr::Dt(_)
        | Expr::Projected(_)
        | Expr::ObsColumnRef(_)
        | Expr::BindingRef(_) => {}
        // Pre-LICM IR (this guard runs at the CLI fit layer, before hoisting).
        Expr::PerEvalRef(_) => {}
    }
}

/// Forcing/table names referenced by a likelihood's argument expressions.
fn collect_likelihood_forcing_table_refs(
    lik: &Likelihood,
    forcings: &mut HashSet<String>,
    tables: &mut HashSet<String>,
) {
    let mut go = |e: &Expr| collect_forcing_table_refs(e, forcings, tables);
    match lik {
        Likelihood::Poisson(p) => go(&p.rate),
        Likelihood::NegBinomial(nb) => {
            go(&nb.mean);
            go(&nb.dispersion);
        }
        Likelihood::Normal(n) => {
            go(&n.mean);
            go(&n.sd);
        }
        Likelihood::Binomial(b) => {
            go(&b.n);
            go(&b.p);
        }
        Likelihood::BetaBinomial(bb) => {
            go(&bb.n);
            go(&bb.alpha);
            go(&bb.beta);
        }
        Likelihood::Bernoulli(b) => go(&b.p),
    }
}

/// The `DerivEntry::Grad` keys of every observation likelihood argument — the
/// obs analogue of `rate_grad.keys()`. A parameter here has a real emitted
/// observation gradient, so (like a `rate_grad` key) it must not be flagged.
fn obs_grad_keys(lik: &Likelihood) -> Vec<&str> {
    fn grads<'a>(m: &'a std::collections::HashMap<String, DerivEntry>, out: &mut Vec<&'a str>) {
        for (name, entry) in m {
            if matches!(entry, DerivEntry::Grad(_)) {
                out.push(name.as_str());
            }
        }
    }
    let mut out = Vec::new();
    match lik {
        Likelihood::Poisson(p) => grads(&p.rate_grad, &mut out),
        Likelihood::NegBinomial(nb) => {
            grads(&nb.mean_grad, &mut out);
            grads(&nb.dispersion_grad, &mut out);
        }
        Likelihood::Normal(n) => {
            grads(&n.mean_grad, &mut out);
            grads(&n.sd_grad, &mut out);
        }
        Likelihood::Binomial(b) => grads(&b.p_grad, &mut out),
        Likelihood::BetaBinomial(bb) => {
            grads(&bb.alpha_grad, &mut out);
            grads(&bb.beta_grad, &mut out);
        }
        Likelihood::Bernoulli(b) => grads(&b.p_grad, &mut out),
    }
    out
}

/// Estimated parameters that NUTS cannot estimate because their forcing/table
/// gradient is a silent zero. This is coeff_guard's half of a two-gate
/// partition (proposal §4.4): every forcing/table coefficient parameter is
/// classified **exactly once** —
///
/// - here, if the coefficient's forcing/table is referenced by a **rate or
///   initial-condition** body (coeff_guard's domain); or
/// - by the observation-gradient preflight at the `run_pgas` boundary, if the
///   coefficient's forcing/table is referenced **only** through an observation
///   (the obs preflight's domain, refused there with the compiler's own
///   `DerivEntry::Unsupported` reason).
///
/// A forcing/table referenced by both a rate/IC and an observation is
/// rate-referenced — coeff_guard owns it (the tiebreak: rate wins). So the scan
/// **skips** a forcing/table that is obs-referenced and not rate/IC-referenced.
///
/// Within coeff_guard's domain, a coefficient parameter is flagged when it has
/// no usable gradient:
/// - **`has_grad` escape** — a parameter for which the compiler emitted a real
///   derivative (a transition `rate_grad`, a σ² `sigma_sq_grad`, or an
///   observation `*_grad` — all `DerivEntry::Grad`) has a usable NUTS gradient
///   and is never flagged (Sinusoidal/Fourier coefficients, constant-indexed
///   parameter tables). The obs/σ² union is what keeps this honest once the obs
///   path emits gradients: a coefficient with a real emitted obs gradient must
///   not be spuriously refused.
/// - **`body` escape** — a parameter also appearing in a rate/observation/IC
///   body carries its gradient there (the non-periodic `coeff` clause).
/// - **Periodic / `lag`** step values have a live-but-omitted gradient
///   (gh#215/gh#314); they are flagged unconditionally within the domain (no
///   `body`/`has_grad` escape), because the emitted body gradient never includes
///   the forcing contribution.
///
/// Sorted, for a deterministic diagnostic.
pub fn coefficient_only_estimated(model: &ir::Model, estimated: &HashSet<String>) -> Vec<String> {
    // ── Partition the forcings/tables into coeff_guard's domain (rate/IC) vs the
    //    obs preflight's domain (observation-only). ──
    let mut rate_ic_forcings = HashSet::new();
    let mut rate_ic_tables = HashSet::new();
    for t in &model.transitions {
        collect_forcing_table_refs(&t.rate, &mut rate_ic_forcings, &mut rate_ic_tables);
    }
    if let InitialConditions::Parameterized(map) = &model.initial_conditions {
        for e in map.values() {
            collect_forcing_table_refs(e, &mut rate_ic_forcings, &mut rate_ic_tables);
        }
    }
    let mut obs_forcings = HashSet::new();
    let mut obs_tables = HashSet::new();
    for o in &model.observations {
        collect_likelihood_forcing_table_refs(&o.likelihood, &mut obs_forcings, &mut obs_tables);
        if let Projection::DerivedExpr(e) = &o.projection {
            collect_forcing_table_refs(e, &mut obs_forcings, &mut obs_tables);
        }
    }
    // A forcing/table is the obs preflight's domain (skipped here) iff it is
    // observation-referenced and NOT rate/IC-referenced.
    let forcing_is_obs_only =
        |name: &str| obs_forcings.contains(name) && !rate_ic_forcings.contains(name);
    let table_is_obs_only =
        |name: &str| obs_tables.contains(name) && !rate_ic_tables.contains(name);

    let mut body = HashSet::new();
    for t in &model.transitions {
        collect(&t.rate, &mut body);
    }
    for o in &model.observations {
        collect_likelihood(&o.likelihood, &mut body);
    }
    if let InitialConditions::Parameterized(map) = &model.initial_conditions {
        for e in map.values() {
            collect(e, &mut body);
        }
    }

    let mut coeff = HashSet::new();
    for tf in &model.time_functions {
        if forcing_is_obs_only(&tf.name) {
            continue;
        }
        collect_forcing(&tf.kind, &mut coeff);
    }
    for tbl in &model.tables {
        if table_is_obs_only(&tbl.name) {
            continue;
        }
        if let Some(values) = tbl.source.values() {
            for e in values {
                collect(e, &mut coeff);
            }
        }
    }

    // Periodic step values are LIVE coefficients but the compiler never emits
    // their gradient (gh#215, `autodiff.ml`: Periodic → omit). So a Periodic
    // coefficient param has NO usable forcing gradient — and unlike the cases
    // below, that holds even when the same param also drives a rate body: the
    // emitted body gradient is real but does not include the forcing
    // contribution, so NUTS would sample against an incomplete gradient. Flag
    // such a param unconditionally (no body / no `has_grad` escape). No false
    // positive: a Periodic coefficient gradient is never emitted, so a param
    // here never legitimately has its forcing contribution covered. (When
    // gh#215 emits Periodic derivatives, drop this set.) Obs-only forcings are
    // the preflight's domain — skipped here just like the `coeff` scan.
    let mut periodic_coeff = HashSet::new();
    for tf in &model.time_functions {
        if forcing_is_obs_only(&tf.name) {
            continue;
        }
        if let TimeFuncKind::Periodic(p) = &tf.kind {
            collect(&p.period, &mut periodic_coeff);
            p.values.iter().for_each(|e| collect(e, &mut periodic_coeff));
        }
        // gh#314: a `lag` parameter (`lag = tau`) is a forcing-internal
        // coefficient with NO emitted gradient. The compiler's autodiff
        // differentiates a `TimeFunc` only w.r.t. its kind's coefficients
        // (`forcing_coeff_exprs` in `autodiff.ml`), never w.r.t. the
        // evaluation-time shift, so ∂forcing/∂lag is an identically-zero
        // dynamics gradient. Like a Periodic step value, this holds even when
        // the same param also drives a rate body: the emitted body gradient is
        // real but omits the forcing's lag contribution, so NUTS would sample
        // against an incomplete gradient. Flag it unconditionally (no body / no
        // `has_grad` escape) so a lag-as-NUTS-target fit is rejected loudly
        // rather than silently mis-estimated.
        if let Some(lag) = &tf.lag {
            collect(lag, &mut periodic_coeff);
        }
    }

    // Parameters the compiler emitted a real derivative for — these have a
    // usable NUTS gradient and are never blocked. Unioned across all three
    // emitted-gradient surfaces (D-accept, §4.4): transition `rate_grad`, σ²
    // `sigma_sq_grad`, and observation likelihood `*_grad` (only `Grad` entries;
    // an `Unsupported` entry is the preflight's refusal, not a usable gradient).
    let mut has_grad: HashSet<&str> = HashSet::new();
    for t in &model.transitions {
        for name in t.rate_grad.keys() {
            has_grad.insert(name.as_str());
        }
        if let DrawMethod::Overdispersed { sigma_sq_grad, .. } = &t.draw_method {
            for (name, entry) in sigma_sq_grad {
                if matches!(entry, DerivEntry::Grad(_)) {
                    has_grad.insert(name.as_str());
                }
            }
        }
    }
    for o in &model.observations {
        for name in obs_grad_keys(&o.likelihood) {
            has_grad.insert(name);
        }
    }

    let mut offenders: Vec<String> = estimated
        .iter()
        .filter(|p| {
            periodic_coeff.contains(*p)
                || (coeff.contains(*p) && !body.contains(*p) && !has_grad.contains(p.as_str()))
        })
        .cloned()
        .collect();
    offenders.sort();
    offenders
}

/// Error message for a NUTS fit blocked by [`coefficient_only_estimated`].
pub fn nuts_guard_error(offenders: &[String]) -> String {
    format!(
        "NUTS cannot estimate parameter(s) [{}]: each drives a forcing or \
         inline-table coefficient whose gradient is not yet emitted (a periodic \
         step value, an inline-table value via a non-constant index — gh#215 — \
         or a forcing `lag`, whose ∂forcing/∂lag is not emitted — gh#314), so \
         NUTS would sample against an incomplete gradient and silently \
         mis-estimate them. These parameters still evaluate live, so estimate \
         them with IF2 or the bootstrap particle filter (gradient-free), or run \
         PGAS with --no-nuts. To estimate seasonality under NUTS, express it as a \
         sinusoidal or fourier forcing (whose coefficients have analytic \
         gradients); a fitted `lag` has no gradient-based path today.",
        offenders.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ir::expr::Expr;
    use ir::time_func::{Sinusoidal, TimeFunction};

    fn base_model() -> ir::Model {
        ir::Model {
            name: "t".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![ir::model::Compartment {
                name: "S".into(),
                kind: ir::model::CompartmentKind::Integer,
            }],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![],
            initial_conditions: InitialConditions::Parameterized(HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![0.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0,
                t_end: 1.0,
                time_semantics: "continuous".into(),
                dt: None,
                rng_seed: Some(1),
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }

    fn sinusoidal_forcing(amplitude: Expr) -> TimeFunction {
        TimeFunction {
            name: "seasonal".into(),
            kind: TimeFuncKind::Sinusoidal(Sinusoidal {
                amplitude,
                period: Expr::const_(365.0),
                phase: Expr::const_(0.0),
                baseline: Expr::const_(1.0),
            }),
            dim: (0, 0),
            lag: None,
        }
    }

    /// `alpha` drives only the forcing amplitude and no transition emits a
    /// derivative for it (no `rate_grad`) → no usable gradient → flagged. This
    /// is the residual case the guard exists for after the gradient half (e.g.
    /// a forcing referenced only through an observation).
    #[test]
    fn flags_param_only_in_forcing_coefficient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("alpha"))];
        let estimated: HashSet<String> = ["alpha".to_string()].into_iter().collect();
        assert_eq!(coefficient_only_estimated(&m, &estimated), vec!["alpha".to_string()]);
    }

    /// `alpha` also appears in a rate → the rate gradient carries it → not flagged.
    #[test]
    fn does_not_flag_param_also_in_a_rate() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("alpha"))];
        m.transitions = vec![ir::transition::Transition {
            name: "decay".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: Expr::bin_op(ir::expr::BinOp::Mul, Expr::param("alpha"), Expr::pop("S")),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad: HashMap::new(),
            lineage: None,
        }];
        let estimated: HashSet<String> = ["alpha".to_string()].into_iter().collect();
        assert!(coefficient_only_estimated(&m, &estimated).is_empty());
    }

    /// gh#119 gradient half: `alpha` enters only the forcing coefficient, but
    /// the compiler now emits ∂rate/∂alpha (the Sinusoidal derivative), so it
    /// has a usable NUTS gradient and must NOT be flagged — otherwise the guard
    /// would refuse the very fits the gradient half enables (the headline
    /// regression this fixes).
    #[test]
    fn does_not_flag_forcing_param_with_emitted_gradient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("alpha"))];
        // `alpha` is not in the rate body (the forcing is opaque there), but the
        // compiler emitted a derivative entry for it.
        let mut rate_grad = HashMap::new();
        rate_grad.insert("alpha".to_string(), Expr::pop("S"));
        m.transitions = vec![ir::transition::Transition {
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: Expr::pop("S"),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad,
            lineage: None,
        }];
        let estimated: HashSet<String> = ["alpha".to_string()].into_iter().collect();
        assert!(coefficient_only_estimated(&m, &estimated).is_empty(),
            "a forcing param with an emitted rate_grad has a usable gradient and \
             must not be blocked");
    }

    /// A non-estimated coefficient parameter is not flagged.
    #[test]
    fn does_not_flag_unestimated_param() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("alpha"))];
        let estimated: HashSet<String> = HashSet::new();
        assert!(coefficient_only_estimated(&m, &estimated).is_empty());
    }

    /// gh#119/gh#215: a `Periodic` step value has NO emitted gradient (the
    /// compiler omits it; only Sinusoidal/Fourier/const-table are differentiated),
    /// so even when the same param ALSO drives a rate body — where the emitted
    /// body gradient is real but does NOT include the forcing contribution — NUTS
    /// would sample against an incomplete gradient. It must be flagged regardless
    /// of body presence or a (partial) `rate_grad` entry. Contrast
    /// `does_not_flag_forcing_param_with_emitted_gradient`: a Sinusoidal coef's
    /// gradient IS complete, so it stays unflagged.
    #[test]
    fn flags_periodic_coeff_param_even_when_in_a_rate_body() {
        use ir::time_func::Periodic;
        let mut m = base_model();
        m.time_functions = vec![TimeFunction {
            name: "weekly".into(),
            kind: TimeFuncKind::Periodic(Periodic {
                period: Expr::const_(7.0),
                values: vec![Expr::param("wpeak"), Expr::const_(1.0)],
            }),
            dim: (0, 0),
            lag: None,
        }];
        // `wpeak` also appears directly in a rate, with a (partial) rate_grad
        // entry for that body appearance — but it misses the forcing part.
        let mut rate_grad = HashMap::new();
        rate_grad.insert("wpeak".to_string(), Expr::pop("S"));
        m.transitions = vec![ir::transition::Transition {
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: Expr::bin_op(ir::expr::BinOp::Mul, Expr::param("wpeak"), Expr::pop("S")),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad,
            lineage: None,
        }];
        let estimated: HashSet<String> = ["wpeak".to_string()].into_iter().collect();
        assert_eq!(coefficient_only_estimated(&m, &estimated), vec!["wpeak".to_string()],
            "a periodic-coeff param has no emitted forcing gradient; NUTS must be \
             blocked even though it also appears in a rate body");
    }

    /// gh#314: a `lag` parameter has NO emitted gradient — the compiler's
    /// autodiff differentiates a `TimeFunc` only w.r.t. its kind's coefficients,
    /// never the evaluation-time shift, so ∂forcing/∂lag is an identically-zero
    /// dynamics gradient. NUTS must be blocked. Like a Periodic step value, this
    /// holds even when the lag param ALSO drives a rate body (the emitted body
    /// gradient omits the forcing's lag contribution).
    #[test]
    fn flags_lag_param_even_when_in_a_rate_body() {
        let mut m = base_model();
        let mut tf = sinusoidal_forcing(Expr::const_(0.3));
        tf.lag = Some(Expr::param("tau"));
        m.time_functions = vec![tf];
        // `tau` also appears directly in a rate, with a (partial) rate_grad
        // entry for that body appearance — but it misses the forcing's lag part.
        let mut rate_grad = HashMap::new();
        rate_grad.insert("tau".to_string(), Expr::pop("S"));
        m.transitions = vec![ir::transition::Transition {
            name: "decay".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: Expr::bin_op(ir::expr::BinOp::Mul, Expr::param("tau"), Expr::pop("S")),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad,
            lineage: None,
        }];
        let estimated: HashSet<String> = ["tau".to_string()].into_iter().collect();
        assert_eq!(coefficient_only_estimated(&m, &estimated), vec!["tau".to_string()],
            "a lag param has no emitted forcing gradient; NUTS must be blocked \
             even though it also appears in a rate body");
    }

    /// gh#314: a non-estimated lag parameter (e.g. a fixed delay) is not flagged.
    #[test]
    fn does_not_flag_unestimated_lag_param() {
        let mut m = base_model();
        let mut tf = sinusoidal_forcing(Expr::const_(0.3));
        tf.lag = Some(Expr::param("tau"));
        m.time_functions = vec![tf];
        let estimated: HashSet<String> = HashSet::new();
        assert!(coefficient_only_estimated(&m, &estimated).is_empty());
    }

    /// An inline-table value parameter, used nowhere else, is flagged.
    #[test]
    fn flags_param_only_in_inline_table_value() {
        let mut m = base_model();
        m.tables = vec![ir::table::Table {
            name: "k_tbl".into(),
            source: ir::table::TableSource::Inline { values: vec![Expr::param("k")] },
            out_of_bounds: ir::table::OobPolicy::Error,
            cell_kind: None,
        }];
        let estimated: HashSet<String> = ["k".to_string()].into_iter().collect();
        assert_eq!(coefficient_only_estimated(&m, &estimated), vec!["k".to_string()]);
    }

    /// A `time_func:<name>` reference expression.
    fn time_func_ref(name: &str) -> Expr {
        use ir::expr::{TimeFuncRef, TimeFuncWrap};
        Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: name.into() } })
    }

    /// A Poisson observation whose `rate` is `rate` and whose `rate_grad` carries
    /// the given entries — the obs analogue of a transition's `rate_grad`.
    fn poisson_obs(rate: Expr, rate_grad: &[(&str, DerivEntry)]) -> ir::observation::ObservationModel {
        use ir::observation::*;
        ir::observation::ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "cases".into(),
            emit_schedule: None,
            stratum: vec![],
            projection: Projection::CumulativeFlow("infection".into()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate,
                rate_grad: rate_grad.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
            }),
        }
    }

    /// D-scan (§4.4): a forcing referenced ONLY through an observation is the obs
    /// preflight's domain — coeff_guard must NOT flag its coefficient. The
    /// amplitude carries a real emitted obs `DerivEntry::Grad`, so it is admitted
    /// (before P5 this was the spurious refusal: `has_grad` was `rate_grad`-only
    /// and the global forcing scan flagged the amplitude).
    #[test]
    fn does_not_flag_forcing_used_only_in_observation() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        // No transition references `seasonal`; the observation does (via its
        // rate), and the compiler emitted `∂rate/∂amp` as a `Grad`.
        m.observations = vec![poisson_obs(
            time_func_ref("seasonal"),
            &[("amp", DerivEntry::Grad(Expr::const_(1.0)))],
        )];
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(
            coefficient_only_estimated(&m, &estimated).is_empty(),
            "an obs-only forcing coefficient with an emitted obs gradient must not \
             be refused by coeff_guard (it is the preflight's domain)"
        );
    }

    /// Tiebreak (§4.4): a forcing referenced by BOTH a rate and an observation is
    /// rate-referenced — coeff_guard owns it. With no emitted gradient at all the
    /// amplitude is (correctly) flagged, exactly as a rate-only forcing would be.
    #[test]
    fn flags_forcing_referenced_by_rate_and_observation_when_gradientless() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        // A transition references `seasonal` (rate-referenced → coeff_guard's
        // domain), with no rate_grad; the observation references it too.
        m.transitions = vec![ir::transition::Transition {
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: time_func_ref("seasonal"),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad: HashMap::new(),
            lineage: None,
        }];
        m.observations = vec![poisson_obs(time_func_ref("seasonal"), &[])];
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert_eq!(
            coefficient_only_estimated(&m, &estimated), vec!["amp".to_string()],
            "a rate-referenced forcing coefficient with no emitted gradient must be \
             flagged (rate-referenced wins the tiebreak — coeff_guard owns it)"
        );
    }

    /// D-accept (§4.4): the `has_grad` union admits a coeff_guard-domain parameter
    /// whose emitted gradient rides the observation `*_grad` rather than a
    /// `rate_grad`. Isolates the union: the forcing is rate-referenced (so it is
    /// coeff_guard's domain and its `rate_grad` is empty), yet the observation
    /// emitted a real `Grad` for the amplitude — without the union it would be
    /// spuriously flagged.
    #[test]
    fn obs_grad_union_rescues_a_coeff_domain_param() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        m.transitions = vec![ir::transition::Transition {
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: time_func_ref("seasonal"),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad: HashMap::new(),
            lineage: None,
        }];
        m.observations = vec![poisson_obs(
            Expr::Projected(ir::expr::ProjectedExpr { projected: () }),
            &[("amp", DerivEntry::Grad(Expr::const_(1.0)))],
        )];
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(
            coefficient_only_estimated(&m, &estimated).is_empty(),
            "the obs `Grad` union must rescue a coeff_guard-domain param whose \
             gradient was emitted only on the observation surface"
        );
    }

    /// An `Unsupported` obs entry is NOT a usable gradient — it must not enter
    /// `has_grad`. Here `amp` drives an obs-referenced-AND-rate-referenced forcing
    /// (coeff_guard's domain) whose obs gradient is `Unsupported`; with no
    /// `rate_grad` it stays flagged.
    #[test]
    fn obs_unsupported_entry_does_not_rescue() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        m.transitions = vec![ir::transition::Transition {
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate: time_func_ref("seasonal"),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad: HashMap::new(),
            lineage: None,
        }];
        m.observations = vec![poisson_obs(
            time_func_ref("seasonal"),
            &[("amp", DerivEntry::Unsupported {
                node: "time_func:seasonal".into(),
                code: ir::deriv::UnsupportedReason::PeriodicCoeff,
            })],
        )];
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert_eq!(
            coefficient_only_estimated(&m, &estimated), vec!["amp".to_string()],
            "an Unsupported obs entry is a refusal, not a usable gradient — it must \
             not rescue via has_grad"
        );
    }
}
