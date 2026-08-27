//! NUTS guard for the **initial-condition** domain (gh#342 P4): refuse to
//! estimate a parameter that reaches a forcing or inline-table coefficient ONLY
//! through an `init` expression.
//!
//! camdl emits no gradient for initial-condition expressions at all — there is
//! no `ic_grad`; IC / state sensitivity is the separate gh#275 surface. So a
//! parameter whose only path into the likelihood is a forcing/table coefficient
//! inside an initial condition has an identically-zero emitted gradient: NUTS
//! would propose against a flat surface and silently mis-sample. The value half
//! still evaluates it live, so IF2 and the bootstrap particle filter estimate
//! it correctly; only a gradient-based (NUTS) fit is refused, loudly.
//!
//! The **rate** and **observation** domains are NOT handled here. A
//! live-but-omitted rate/obs coefficient (a Periodic step value, a `lag`, a
//! non-constant table index) now serialises a `DerivEntry::Unsupported` (gh#342
//! P1–P3) that the `run_pgas` preflight refuses at the boundary — for every
//! caller, not just the CLI. This guard is the residual the preflight cannot
//! see: no grad map carries an IC-exclusive coefficient's `Unsupported`, so the
//! IC scan must read the source model directly.
//!
//! Scope: `body` are the rate / observation / initial-condition expressions;
//! `coeff` are the forcing-coefficient and inline-table-value expressions of a
//! forcing/table referenced ONLY by an initial condition. A parameter in both a
//! coefficient and a body, or one with a real emitted gradient, is not flagged.
//! The coefficient sub-expressions live in `model.time_functions[*].kind` and
//! `model.tables[*]`, not in the rate AST, so this needs its own traversal.

use std::collections::HashSet;

use ir::deriv::DerivEntry;
use ir::expr::Expr;
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
        // gh#616: no parameter reference — the value comes from the run's data.
        | Expr::ObsAnchor(_)
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
        Likelihood::Poisson(p) => collect(&p.rate.expr, out),
        Likelihood::NegBinomial(nb) => {
            collect(&nb.mean.expr, out);
            collect(&nb.dispersion.expr, out);
        }
        Likelihood::Normal(n) => {
            collect(&n.mean.expr, out);
            collect(&n.sd.expr, out);
        }
        Likelihood::Binomial(b) => {
            collect(&b.n, out);
            collect(&b.p.expr, out);
        }
        Likelihood::BetaBinomial(bb) => {
            collect(&bb.n, out);
            collect(&bb.alpha.expr, out);
            collect(&bb.beta.expr, out);
        }
        Likelihood::Beta(b) => {
            collect(&b.mean.expr, out);
            collect(&b.concentration.expr, out);
        }
        Likelihood::Bernoulli(b) => collect(&b.p.expr, out),
        Likelihood::ZeroInflatedNegBinomial(zi) => {
            collect(&zi.mean.expr, out);
            collect(&zi.dispersion.expr, out);
            collect(&zi.pi.expr, out);
        }
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
/// into this guard's domain (referenced ONLY by an initial condition) versus the
/// `run_pgas` preflight's domain (referenced by a rate or observation, where the
/// compiler emits a `DerivEntry`) — proposal §4.4.
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
        | Expr::ObsAnchor(_)
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
        Likelihood::Poisson(p) => go(&p.rate.expr),
        Likelihood::NegBinomial(nb) => {
            go(&nb.mean.expr);
            go(&nb.dispersion.expr);
        }
        Likelihood::Normal(n) => {
            go(&n.mean.expr);
            go(&n.sd.expr);
        }
        Likelihood::Binomial(b) => {
            go(&b.n);
            go(&b.p.expr);
        }
        Likelihood::BetaBinomial(bb) => {
            go(&bb.n);
            go(&bb.alpha.expr);
            go(&bb.beta.expr);
        }
        Likelihood::Beta(b) => {
            go(&b.mean.expr);
            go(&b.concentration.expr);
        }
        Likelihood::Bernoulli(b) => go(&b.p.expr),
        Likelihood::ZeroInflatedNegBinomial(zi) => {
            go(&zi.mean.expr);
            go(&zi.dispersion.expr);
            go(&zi.pi.expr);
        }
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
        Likelihood::Poisson(p) => grads(&p.rate.grad, &mut out),
        Likelihood::NegBinomial(nb) => {
            grads(&nb.mean.grad, &mut out);
            grads(&nb.dispersion.grad, &mut out);
        }
        Likelihood::Normal(n) => {
            grads(&n.mean.grad, &mut out);
            grads(&n.sd.grad, &mut out);
        }
        Likelihood::Binomial(b) => grads(&b.p.grad, &mut out),
        Likelihood::BetaBinomial(bb) => {
            grads(&bb.alpha.grad, &mut out);
            grads(&bb.beta.grad, &mut out);
        }
        Likelihood::Beta(b) => {
            grads(&b.mean.grad, &mut out);
            grads(&b.concentration.grad, &mut out);
        }
        Likelihood::Bernoulli(b) => grads(&b.p.grad, &mut out),
        Likelihood::ZeroInflatedNegBinomial(zi) => {
            grads(&zi.mean.grad, &mut out);
            grads(&zi.dispersion.grad, &mut out);
            grads(&zi.pi.grad, &mut out);
        }
    }
    out
}

/// Estimated parameters NUTS cannot estimate because their forcing/table
/// gradient is a silent zero **and no gradient surface classifies it** — the
/// initial-condition residual of the gh#342 P4 partition. Every forcing/table
/// coefficient parameter is classified **exactly once**:
///
/// - by the `run_pgas` preflight, if the coefficient's forcing/table is
///   referenced by a **rate or observation** — the compiler emits a
///   `DerivEntry` (Grad, or a live-but-omitted `Unsupported`) for it, and the
///   preflight refuses on the `Unsupported` at the boundary, for every caller;
/// - here, if the coefficient's forcing/table is referenced **only** through an
///   initial condition. camdl emits no gradient for IC expressions at all (no
///   `ic_grad`; IC/state sensitivity is the separate gh#275 surface), so an
///   IC-exclusive coefficient carries no `DerivEntry` anywhere — the preflight
///   cannot see it, and this source-level scan is the only place that can.
///
/// So the scan **skips** any forcing/table that is rate- or obs-referenced
/// (those are the preflight's); it owns only the IC-exclusive set. A forcing
/// referenced by both an IC and a rate/obs is the preflight's — the emitted
/// gradient there covers its rate/obs contribution, and the residual IC
/// incompleteness is the same escape coeff_guard already tolerated (deferred to
/// gh#275).
///
/// Within this guard's IC-exclusive domain, a coefficient parameter is flagged
/// when it has no usable gradient:
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
pub fn ic_coefficient_only_estimated(model: &ir::Model, estimated: &HashSet<String>) -> Vec<String> {
    // ── Partition the forcings/tables by which surface references them. This
    //    guard owns ONLY the initial-condition-exclusive set; the rate and
    //    observation surfaces now refuse a live-but-omitted coefficient at the
    //    `run_pgas` preflight (via a serialized `DerivEntry::Unsupported`). ──
    let mut rate_forcings = HashSet::new();
    let mut rate_tables = HashSet::new();
    for t in &model.transitions {
        collect_forcing_table_refs(&t.rate, &mut rate_forcings, &mut rate_tables);
    }
    let mut ic_forcings = HashSet::new();
    let mut ic_tables = HashSet::new();
    for (_, spec) in &model.initial_conditions {
        // EVERY expression the spec evaluates, not just the mean: a law's `n`
        // or `sd` reaching a forcing is the same silent-zero coefficient.
        for e in spec.exprs() {
            collect_forcing_table_refs(e, &mut ic_forcings, &mut ic_tables);
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
    // A forcing/table is THIS guard's domain iff it is reached ONLY through an
    // initial condition — not by a rate, not by an observation. A rate/obs
    // reference means the compiler emitted a `DerivEntry` (Grad or Unsupported)
    // for it, which the preflight already classifies; an IC reference emits no
    // gradient at all (no `ic_grad`), so an IC-exclusive coefficient is the one
    // silent-zero the preflight cannot see. The scans below skip everything else.
    let forcing_is_ic_only = |name: &str| {
        ic_forcings.contains(name) && !rate_forcings.contains(name) && !obs_forcings.contains(name)
    };
    let table_is_ic_only = |name: &str| {
        ic_tables.contains(name) && !rate_tables.contains(name) && !obs_tables.contains(name)
    };

    let mut body = HashSet::new();
    for t in &model.transitions {
        collect(&t.rate, &mut body);
    }
    for o in &model.observations {
        collect_likelihood(&o.likelihood, &mut body);
    }
    for (_, spec) in &model.initial_conditions {
        for e in spec.exprs() {
            collect(e, &mut body);
        }
    }

    let mut coeff = HashSet::new();
    for tf in &model.time_functions {
        if !forcing_is_ic_only(&tf.name) {
            continue;
        }
        collect_forcing(&tf.kind, &mut coeff);
    }
    for tbl in &model.tables {
        if !table_is_ic_only(&tbl.name) {
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
    // gh#215 emits Periodic derivatives, drop this set.) Only IC-exclusive
    // forcings are scanned; a rate- or obs-referenced Periodic forcing is the
    // preflight's domain (its `Unsupported` rides the emitted grad map), skipped
    // here just like the `coeff` scan.
    let mut periodic_coeff = HashSet::new();
    for tf in &model.time_functions {
        if !forcing_is_ic_only(&tf.name) {
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
        for (name, entry) in &t.rate_grad {
            // Only a real `Grad` is a usable gradient. Post-gh#342 a rate
            // coefficient the compiler could not differentiate (tier-2b Periodic/
            // `lag`/non-const table index) serialises an `Unsupported` here; it
            // must NOT enter `has_grad` (that would let coeff_guard admit a fit
            // whose forcing gradient is missing — the exact silent-mis-estimate
            // this guard exists to prevent), mirroring the σ²/obs branches.
            if matches!(entry, DerivEntry::Grad(_)) {
                has_grad.insert(name.as_str());
            }
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

/// Error message for a NUTS fit blocked by [`ic_coefficient_only_estimated`].
pub fn nuts_guard_error(offenders: &[String]) -> String {
    format!(
        "NUTS cannot estimate parameter(s) [{}]: each reaches a forcing or \
         inline-table coefficient ONLY through an initial condition (`init`), \
         and camdl emits no gradient for initial-condition expressions (IC/state \
         sensitivity is gh#275), so NUTS would sample against an incomplete \
         gradient and silently mis-estimate them. These parameters still evaluate \
         live, so estimate them with IF2 or the bootstrap particle filter \
         (gradient-free), or run PGAS with --no-nuts. Alternatively, drive the \
         parameter through a rate or observation (whose coefficients do carry \
         analytic gradients) rather than only the initial state.",
        offenders.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ir::expr::Expr;
    use ir::model::InitialConditions;
    use ir::time_func::{Sinusoidal, TimeFunction};

    fn base_model() -> ir::Model {
        ir::Model {
            ic_grad: Default::default(),
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
            initial_conditions: InitialConditions::default(),
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
                t_end_anchor: None,
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
            data_source: None,
        }
    }

    /// A `time_func:<name>` reference expression — the shape an `init` RHS (or a
    /// rate/obs) uses to name a forcing.
    fn time_func_ref(name: &str) -> Expr {
        use ir::expr::{TimeFuncRef, TimeFuncWrap};
        Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: name.into() } })
    }

    /// A `table_lookup` expression selecting cell `idx` of `table`.
    fn table_lookup(table: &str, idx: f64) -> Expr {
        use ir::expr::{TableLookupExpr, TableLookupWrap};
        Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr { table: table.into(), indices: vec![Expr::const_(idx)] },
        })
    }

    /// Set `expr` as the initial value of compartment `S`, so any forcing/table
    /// it names becomes initial-condition-referenced — this guard's domain.
    fn ic_referencing(m: &mut ir::Model, expr: Expr) {
        let mut ic = HashMap::new();
        ic.insert("S".to_string(), expr);
        m.initial_conditions = InitialConditions::exprs(ic);
    }

    /// A transition with rate `rate` carrying `rate_grad` — used to make a
    /// forcing rate-referenced (moving it into the preflight's domain).
    fn rate_transition(rate: Expr, rate_grad: &[(&str, DerivEntry)]) -> ir::transition::Transition {
        ir::transition::Transition {
            rate_state_grad: Default::default(),
            name: "infection".into(),
            stoichiometry: vec![ir::transition::StoichiometryEntry("S".into(), -1)],
            rate,
            metadata: None,
            draw_method: ir::transition::DrawMethod::Poisson,
            rate_grad: rate_grad.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
            lineage: None,
        }
    }

    /// A minimal Poisson observation naming `rate` — used to make a forcing
    /// observation-referenced (the preflight's domain).
    fn poisson_obs(rate: Expr) -> ir::observation::ObservationModel {
        use ir::observation::*;
        ObservationModel {
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
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable { expr: rate, grad: HashMap::new(), proj_grad: None },
            }),
        }
    }

    // ── This guard's domain: a coefficient reached ONLY through an `init` ─────
    // camdl emits no gradient for initial-condition expressions (gh#275), so a
    // coefficient reached only there has no usable gradient regardless of the
    // forcing kind — even a Sinusoidal amplitude, whose derivative IS analytic
    // when it drives a rate/obs.

    /// A Sinusoidal amplitude reached only via an initial condition → flagged
    /// (no IC gradient path, so the tier-1 analytic derivative is never emitted).
    #[test]
    fn flags_sinusoidal_amp_reached_only_via_ic() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        ic_referencing(&mut m, time_func_ref("seasonal"));
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert_eq!(ic_coefficient_only_estimated(&m, &estimated), vec!["amp".to_string()]);
    }

    /// A Periodic step value reached only via an initial condition → flagged.
    #[test]
    fn flags_periodic_coeff_reached_only_via_ic() {
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
            data_source: None,
        }];
        ic_referencing(&mut m, time_func_ref("weekly"));
        let estimated: HashSet<String> = ["wpeak".to_string()].into_iter().collect();
        assert_eq!(ic_coefficient_only_estimated(&m, &estimated), vec!["wpeak".to_string()]);
    }

    /// A forcing `lag` reached only via an initial condition → flagged.
    #[test]
    fn flags_lag_reached_only_via_ic() {
        let mut m = base_model();
        let mut tf = sinusoidal_forcing(Expr::const_(0.3));
        tf.lag = Some(Expr::param("tau"));
        m.time_functions = vec![tf];
        ic_referencing(&mut m, time_func_ref("seasonal"));
        let estimated: HashSet<String> = ["tau".to_string()].into_iter().collect();
        assert_eq!(ic_coefficient_only_estimated(&m, &estimated), vec!["tau".to_string()]);
    }

    /// An inline-table value reached only via an initial condition → flagged.
    #[test]
    fn flags_inline_table_value_reached_only_via_ic() {
        let mut m = base_model();
        m.tables = vec![ir::table::Table {
            name: "k_tbl".into(),
            source: ir::table::TableSource::Inline { values: vec![Expr::param("k")] },
            out_of_bounds: ir::table::OobPolicy::Error,
            cell_kind: None,
        }];
        ic_referencing(&mut m, table_lookup("k_tbl", 0.0));
        let estimated: HashSet<String> = ["k".to_string()].into_iter().collect();
        assert_eq!(ic_coefficient_only_estimated(&m, &estimated), vec!["k".to_string()]);
    }

    // ── Boundary: rate/obs surfaces are the preflight's domain, not this one ──

    /// A coefficient reached through a RATE is the preflight's domain (its
    /// `DerivEntry` rides the emitted `rate_grad`); this guard must NOT flag it,
    /// even with a gradientless `rate_grad`. Refusing there is the preflight's
    /// job — double-flagging is the fork gh#342 removes.
    #[test]
    fn does_not_flag_rate_referenced_coefficient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        // `seasonal` is referenced by both a rate and the IC — rate-referenced
        // wins, so this guard skips it (the preflight owns the rate contribution).
        m.transitions = vec![rate_transition(time_func_ref("seasonal"), &[])];
        ic_referencing(&mut m, time_func_ref("seasonal"));
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(ic_coefficient_only_estimated(&m, &estimated).is_empty(),
            "a rate-referenced coefficient is the preflight's domain, not this guard's");
    }

    /// A coefficient reached only through an OBSERVATION is likewise the
    /// preflight's domain — not flagged here.
    #[test]
    fn does_not_flag_obs_only_coefficient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        m.observations = vec![poisson_obs(time_func_ref("seasonal"))];
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(ic_coefficient_only_estimated(&m, &estimated).is_empty(),
            "an obs-referenced coefficient is the preflight's domain, not this guard's");
    }

    // ── Retained escape (verbatim; the IC-incompleteness residual is gh#275) ──

    /// A param reaching an IC-only forcing coefficient that ALSO appears directly
    /// in a rate body is not flagged: the rate body carries a real gradient, and
    /// the residual IC-forcing incompleteness is the escape this guard tolerates
    /// until IC/state gradients exist (gh#275). Pins current behavior.
    #[test]
    fn does_not_flag_ic_coeff_param_also_in_a_rate_body() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        // `seasonal` is IC-only (the rate references `amp` directly, not the
        // forcing), and `amp` carries a real rate_grad from that body appearance.
        m.transitions = vec![rate_transition(
            Expr::bin_op(ir::expr::BinOp::Mul, Expr::param("amp"), Expr::pop("S")),
            &[("amp", DerivEntry::Grad(Expr::pop("S")))],
        )];
        ic_referencing(&mut m, time_func_ref("seasonal"));
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(ic_coefficient_only_estimated(&m, &estimated).is_empty(),
            "the body/has_grad escape is retained verbatim (IC incompleteness → gh#275)");
    }

    // ── Non-triggers ──────────────────────────────────────────────────────────

    /// A non-estimated IC coefficient parameter is not flagged.
    #[test]
    fn does_not_flag_unestimated_ic_coefficient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        ic_referencing(&mut m, time_func_ref("seasonal"));
        let estimated: HashSet<String> = HashSet::new();
        assert!(ic_coefficient_only_estimated(&m, &estimated).is_empty());
    }

    /// A forcing referenced by NOTHING (not rate/obs/IC) drives no dynamics, so
    /// its coefficient's likelihood is flat — the posterior equals the prior and
    /// admitting it is benign, NOT a silent bias. This is the behavior change
    /// from the pre-P4 global scan, which flagged any forcing coefficient.
    #[test]
    fn does_not_flag_unreferenced_forcing_coefficient() {
        let mut m = base_model();
        m.time_functions = vec![sinusoidal_forcing(Expr::param("amp"))];
        // No rate, observation, or initial condition references `seasonal`.
        let estimated: HashSet<String> = ["amp".to_string()].into_iter().collect();
        assert!(ic_coefficient_only_estimated(&m, &estimated).is_empty(),
            "a dead forcing coefficient is benign (posterior = prior), not a silent bias");
    }
}
