use std::collections::HashSet;
use thiserror::Error;
use crate::{
    expr::Expr,
    model::{CompartmentKind, Model},
    quantity::{QuantityBody, QuantitySource, TemporalReduce, TimeReduce, ValueReduce},
};

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("duplicate compartment name: {0}")]
    DuplicateCompartment(String),

    #[error("duplicate transition name: {0}")]
    DuplicateTransition(String),

    #[error("duplicate parameter name: {0}")]
    DuplicateParameter(String),

    #[error("transition '{transition}' stoichiometry references unknown compartment '{compartment}'")]
    UnknownCompartmentInStoichiometry { transition: String, compartment: String },

    #[error("transition '{transition}' stoichiometry entry has zero delta for '{compartment}'")]
    ZeroDeltaInStoichiometry { transition: String, compartment: String },

    #[error("transition '{transition}' stoichiometry references real compartment '{compartment}'; real compartments cannot appear in stoichiometry")]
    RealCompartmentInStoichiometry { transition: String, compartment: String },

    #[error("real compartment '{0}' has no ODE equation")]
    MissingOdeEquation(String),

    #[error("ODE equation targets '{0}' which is not a real compartment")]
    OdeForNonRealCompartment(String),

    #[error("expression references unknown parameter '{0}'")]
    UnknownParameter(String),

    #[error("expression references unknown compartment '{0}'")]
    UnknownCompartment(String),

    #[error("expression references unknown table '{0}'")]
    UnknownTable(String),

    #[error("expression references unknown time function '{0}'")]
    UnknownTimeFunction(String),

    #[error("observation '{obs}' cumulative_flow references unknown transition '{transition}'")]
    UnknownTransitionInObservation { obs: String, transition: String },

    #[error("intervention '{intervention}' action references unknown compartment '{compartment}'")]
    UnknownCompartmentInIntervention { intervention: String, compartment: String },

    #[error("balance constraint targets unknown compartment '{0}'")]
    UnknownCompartmentInBalance(String),

    #[error("initial condition references unknown compartment '{0}'")]
    UnknownCompartmentInInitialConditions(String),

    #[error("table lookup of '{table}' has wrong arity: {got} indices but the IR table \
             is rank-1 (multi-dimensional tables are pre-flattened by the compiler to a \
             single linear index, so a lookup must carry exactly 1 index)")]
    TableLookupArity { table: String, got: usize },

    #[error("table lookup of '{table}' uses a constant index {index} that is out of range: \
             the table has {len} cell(s), so the valid index range is [0, {len}) \
             (the runtime floors a fractional index before this check). Fix the index \
             expression or widen the table to {at_least} cell(s)", at_least = *index + 1)]
    TableLookupConstantIndexOutOfRange { table: String, index: i64, len: usize },

    #[error("initial value for compartment '{compartment}' is not finite (got {value}); \
             initial conditions must be finite numbers")]
    InitialValueNotFinite { compartment: String, value: f64 },

    #[error("initial value for compartment '{compartment}' must be nonnegative (got {value}); \
             a compartment cannot start with a negative population")]
    InitialValueNegative { compartment: String, value: f64 },

    #[error("initial value for integer compartment '{compartment}' must be a whole number \
             (got {value}); a fractional value would be silently truncated. Round it to a \
             whole count, or declare the compartment real if fractional state is intended")]
    InitialValueNotInteger { compartment: String, value: f64 },

    #[error("quantity '{quantity}' uses '{leaf}', which is only meaningful inside a \
             transition rate or a likelihood, not in a quantity (a quantity is read at \
             output cadence over a finished trajectory). Reachable directly or via a \
             referenced binding; remove it from the quantity (and any binding it reaches)")]
    QuantityForbiddenLeaf { quantity: String, leaf: String },
}

pub fn validate(model: &Model) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // ── Build name sets ───────────────────────────────────────────────────────

    let mut comp_names:  HashSet<&str> = HashSet::new();
    let mut real_comps:  HashSet<&str> = HashSet::new();
    let mut int_comps:   HashSet<&str> = HashSet::new();
    let mut param_names: HashSet<&str> = HashSet::new();
    let mut table_names: HashSet<&str> = HashSet::new();
    // gh#127 (#12): per-table linear length, known statically only for Inline
    // tables (External tables get their values at runtime). Used to range-check
    // a compile-time-constant lookup index. External tables are absent from the
    // map → no static range check (the runtime handles them).
    let mut table_lens:  std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut tf_names:    HashSet<&str> = HashSet::new();
    let mut tr_names:    HashSet<&str> = HashSet::new();

    for c in &model.compartments {
        if !comp_names.insert(c.name.as_str()) {
            errors.push(ValidationError::DuplicateCompartment(c.name.clone()));
        }
        match c.kind {
            CompartmentKind::Real    => { real_comps.insert(c.name.as_str()); }
            CompartmentKind::Integer => { int_comps.insert(c.name.as_str()); }
        }
    }

    for p in &model.parameters {
        if !param_names.insert(p.name.as_str()) {
            errors.push(ValidationError::DuplicateParameter(p.name.clone()));
        }
        // (Prior-and-hierarchical-both-set is now unrepresentable: PriorSpec
        // is a single slot. The former runtime check + error variant were
        // deleted with the gh#191 ParamValue ADT.)
    }
    for t in &model.tables {
        table_names.insert(t.name.as_str());
        // Inline tables carry their cell exprs; the linear length is known now.
        // External tables are filled at runtime, so their length is not static.
        if let Some(values) = t.source.values() {
            table_lens.insert(t.name.as_str(), values.len());
        }
    }
    for tf in &model.time_functions {
        tf_names.insert(tf.name.as_str());
    }
    for tr in &model.transitions {
        if !tr_names.insert(tr.name.as_str()) {
            errors.push(ValidationError::DuplicateTransition(tr.name.clone()));
        }
    }

    // ── Stoichiometry checks ──────────────────────────────────────────────────

    for tr in &model.transitions {
        for entry in &tr.stoichiometry {
            let comp = &entry.0;
            let delta = entry.1;
            if !comp_names.contains(comp.as_str()) {
                errors.push(ValidationError::UnknownCompartmentInStoichiometry {
                    transition: tr.name.clone(),
                    compartment: comp.clone(),
                });
            } else if real_comps.contains(comp.as_str()) {
                errors.push(ValidationError::RealCompartmentInStoichiometry {
                    transition: tr.name.clone(),
                    compartment: comp.clone(),
                });
            }
            if delta == 0 {
                errors.push(ValidationError::ZeroDeltaInStoichiometry {
                    transition: tr.name.clone(),
                    compartment: comp.clone(),
                });
            }
        }
    }

    // ── ODE equation checks ───────────────────────────────────────────────────

    let ode_comps: HashSet<&str> = model.ode_equations.iter().map(|e| e.compartment.as_str()).collect();
    for rc in &real_comps {
        if !ode_comps.contains(*rc) {
            errors.push(ValidationError::MissingOdeEquation(rc.to_string()));
        }
    }
    for eq in &model.ode_equations {
        if !real_comps.contains(eq.compartment.as_str()) {
            errors.push(ValidationError::OdeForNonRealCompartment(eq.compartment.clone()));
        }
    }

    // ── Expression reference checks ───────────────────────────────────────────

    let ctx = RefCtx { comp_names: &comp_names, param_names: &param_names, table_names: &table_names, table_lens: &table_lens, tf_names: &tf_names };

    for tr in &model.transitions {
        check_expr(&tr.rate, &ctx, false, &mut errors);
    }
    for eq in &model.ode_equations {
        check_expr(&eq.derivative, &ctx, false, &mut errors);
    }
    for obs in &model.observations {
        // projection
        match &obs.projection {
            crate::observation::Projection::CumulativeFlow(tn) => {
                if !tr_names.contains(tn.as_str()) {
                    errors.push(ValidationError::UnknownTransitionInObservation {
                        obs: obs.name.clone(),
                        transition: tn.clone(),
                    });
                }
            }
            crate::observation::Projection::CumulativeFlowSum(tns) => {
                for tn in tns {
                    if !tr_names.contains(tn.as_str()) {
                        errors.push(ValidationError::UnknownTransitionInObservation {
                            obs: obs.name.clone(),
                            transition: tn.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
        // likelihood exprs (projected is allowed)
        check_likelihood_exprs(&obs.likelihood, &ctx, &mut errors);
    }

    // ── Intervention & event action target checks (gh#123) ────────────────────
    //
    // Interventions (`interventions {}`) and events (`events {}`, marked
    // `always_active`) both lower to `Intervention` in the IR. Every action
    // names compartment(s) it modifies; a dangling target reaches the runtime
    // as a silent no-op or an out-of-range panic. Validate the names — and
    // recurse into the action value/count/fraction expressions, which may
    // reference params/compartments/tables.
    for iv in &model.interventions {
        let check_target = |comp: &str, errors: &mut Vec<ValidationError>| {
            if !comp_names.contains(comp) {
                errors.push(ValidationError::UnknownCompartmentInIntervention {
                    intervention: iv.name.clone(),
                    compartment: comp.to_string(),
                });
            }
        };
        for action in &iv.actions {
            use crate::intervention::Action;
            match action {
                Action::FractionTransfer(ft) => {
                    check_target(&ft.src, &mut errors);
                    check_target(&ft.dst, &mut errors);
                    check_expr(&ft.fraction, &ctx, false, &mut errors);
                }
                Action::AbsoluteTransfer(at) => {
                    check_target(&at.src, &mut errors);
                    check_target(&at.dst, &mut errors);
                    check_expr(&at.count, &ctx, false, &mut errors);
                }
                Action::Set(s) => {
                    check_target(&s.compartment, &mut errors);
                    check_expr(&s.value, &ctx, false, &mut errors);
                }
                Action::Add(a) => {
                    check_target(&a.compartment, &mut errors);
                    check_expr(&a.count, &ctx, false, &mut errors);
                }
            }
        }
    }

    // ── Balance constraint target check (gh#123) ──────────────────────────────
    //
    // The balance constraint overwrites its target compartment with `expr`
    // every substep. A dangling target silently does nothing.
    if let Some(b) = &model.balance {
        if !comp_names.contains(b.target.as_str()) {
            errors.push(ValidationError::UnknownCompartmentInBalance(b.target.clone()));
        }
        check_expr(&b.expr, &ctx, false, &mut errors);
    }

    // ── Initial-condition key checks (gh#114 Rust-side) ────────────────────────
    //
    // Every init key must resolve to a declared (expanded) compartment — a
    // stratified model can otherwise carry an init value for nonexistent `S`
    // while the real cells (e.g. `S_child_kano`) default to zero, silently
    // starting the epidemic in an empty population. The Parameterized variant
    // also carries an expression per key; recurse into it.
    //
    // ── Initial-condition VALUE domain checks (gh#124) ─────────────────────────
    //
    // For the Explicit variant the IR carries a concrete f64 per compartment.
    // The runtime converts an integer init via `*val as i64`
    // (compiled_model.rs), which truncates and saturates: 0.6 → 0, -3 → a
    // negative compartment from t=0, NaN/inf → 0 / i64::MAX. Each is a "model
    // runs but starts in the wrong population" failure, so we reject them here
    // at the contract boundary:
    //   - non-finite (NaN / ±inf) for any compartment,
    //   - negative for any compartment (a count is nonnegative, int or real),
    //   - non-integer for INTEGER compartments (a near-integer tolerance allows
    //     for float round-trip noise; a clearly-fractional value errors).
    // Real compartments may hold fractional (but finite, nonnegative) values.
    // Parameterized / FromDistribution inits carry expressions / priors rather
    // than literals, so there is nothing to range-check statically here; their
    // values are produced (and bounds-enforced) at sim/inference time.
    {
        use crate::model::InitialConditions;
        // Tolerance for the integer check: a value within this of its nearest
        // integer is treated as that integer (absorbs float round-trip noise
        // like 3.0000000001). Mirrors the `1e-9` tolerance the issue specifies
        // for `checked_int_initial_value`.
        const INT_TOL: f64 = 1e-9;
        let check_init_key = |comp: &str, errors: &mut Vec<ValidationError>| {
            if !comp_names.contains(comp) {
                errors.push(ValidationError::UnknownCompartmentInInitialConditions(
                    comp.to_string(),
                ));
            }
        };
        let check_init_value = |comp: &str, v: f64, errors: &mut Vec<ValidationError>| {
            // Only meaningful for declared compartments; the key check above
            // already reports an unknown name, so skip the value check for it
            // (we can't classify int-vs-real for a name we don't know).
            if !comp_names.contains(comp) {
                return;
            }
            if !v.is_finite() {
                errors.push(ValidationError::InitialValueNotFinite {
                    compartment: comp.to_string(),
                    value: v,
                });
                return;
            }
            if v < 0.0 {
                errors.push(ValidationError::InitialValueNegative {
                    compartment: comp.to_string(),
                    value: v,
                });
                return;
            }
            if int_comps.contains(comp) && (v - v.round()).abs() > INT_TOL {
                errors.push(ValidationError::InitialValueNotInteger {
                    compartment: comp.to_string(),
                    value: v,
                });
            }
        };
        match &model.initial_conditions {
            InitialConditions::Explicit(map) => {
                for (k, v) in map {
                    check_init_key(k, &mut errors);
                    check_init_value(k, *v, &mut errors);
                }
            }
            InitialConditions::Parameterized(map) => {
                for (k, e) in map {
                    check_init_key(k, &mut errors);
                    check_expr(e, &ctx, false, &mut errors);
                }
            }
            InitialConditions::FromDistribution(map) => {
                for k in map.keys() {
                    check_init_key(k, &mut errors);
                }
            }
        }
    }

    // ── Generated quantities: state-expression legality (proposal 2026-06-25) ──
    // A quantity's state expression is the shared `Expr` restricted to a
    // validated subset. Name-resolution reuses `check_expr` (above); the
    // forbidden-leaf + transitive-`BindingRef` legality is enforced HERE, at the
    // load boundary, because a constructor-only check over the quantity `Expr`
    // alone cannot see a forbidden leaf reached through a binding body. (LICM does
    // not process quantity bodies, so a `PerEvalRef` never appears legitimately.)
    if !model.quantities.is_empty() {
        let bindings_map: std::collections::HashMap<&str, &Expr> =
            model.bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect();
        for q in &model.quantities {
            for e in quantity_state_exprs(&q.body) {
                check_expr(e, &ctx, false, &mut errors);
                let mut seen: HashSet<&str> = HashSet::new();
                check_quantity_legal(e, &bindings_map, &q.name, &mut errors, &mut seen);
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(())
}

/// Collect every state `Expr` a quantity body evaluates against latent state: the
/// `State` source expr plus any reduction-threshold expr. `Derived` reduction
/// arithmetic has no state `Expr` (its leaves are `QRef`/param/const), so it
/// contributes none.
fn quantity_state_exprs(body: &QuantityBody) -> Vec<&Expr> {
    let mut out = Vec::new();
    match body {
        QuantityBody::Reduced { source, reduce } => {
            // A `State` source is a state expr (name-checked + legality-checked);
            // an `Observation` source reduces `y_sim`, not a state expr — nothing
            // to check here (its stream ref is validated separately). Reduction
            // thresholds below are state exprs in either case.
            if let QuantitySource::State(e) = source {
                out.push(e);
            }
            if let Some(r) = reduce {
                match r {
                    TemporalReduce::Value(ValueReduce::CountAbove(t))
                    | TemporalReduce::Value(ValueReduce::CountBelow(t)) => out.push(t),
                    TemporalReduce::Time(TimeReduce::FirstAbove(t))
                    | TemporalReduce::Time(TimeReduce::FirstBelow(t))
                    | TemporalReduce::Time(TimeReduce::LastAbove(t))
                    | TemporalReduce::Time(TimeReduce::LastBelow(t)) => out.push(t),
                    _ => {}
                }
            }
        }
        QuantityBody::Derived(_) => {}
    }
    out
}

/// Reject the four leaves meaningless in a quantity (read at output cadence over a
/// finished trajectory) — `Dt`/`Projected`/`ObsColumnRef`/`PerEvalRef` — anywhere
/// in the tree, recursing transitively through `BindingRef` over `model.bindings`
/// so a forbidden leaf cannot be smuggled in via a binding body. `seen` guards a
/// (malformed) binding cycle. Name-resolution is `check_expr`'s job, not this.
fn check_quantity_legal<'a>(
    expr: &'a Expr,
    bindings: &std::collections::HashMap<&'a str, &'a Expr>,
    quantity: &str,
    errors: &mut Vec<ValidationError>,
    seen: &mut HashSet<&'a str>,
) {
    fn forbid(errors: &mut Vec<ValidationError>, quantity: &str, leaf: &str) {
        errors.push(ValidationError::QuantityForbiddenLeaf {
            quantity: quantity.to_string(),
            leaf: leaf.to_string(),
        });
    }
    match expr {
        Expr::Dt(_) => forbid(errors, quantity, "dt"),
        Expr::Projected(_) => forbid(errors, quantity, "projected"),
        Expr::ObsColumnRef(_) => forbid(errors, quantity, "obs_column_ref"),
        Expr::PerEvalRef(_) => forbid(errors, quantity, "per_eval_ref"),
        Expr::Const(_) | Expr::Time(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_)
        | Expr::TimeFunc(_) => {}
        Expr::BinOp(w) => {
            check_quantity_legal(&w.bin_op.left, bindings, quantity, errors, seen);
            check_quantity_legal(&w.bin_op.right, bindings, quantity, errors, seen);
        }
        Expr::UnOp(w) => check_quantity_legal(&w.un_op.arg, bindings, quantity, errors, seen),
        Expr::Cond(w) => {
            check_quantity_legal(&w.cond.pred, bindings, quantity, errors, seen);
            check_quantity_legal(&w.cond.then, bindings, quantity, errors, seen);
            check_quantity_legal(&w.cond.else_, bindings, quantity, errors, seen);
        }
        Expr::TableLookup(w) => {
            for idx in &w.table_lookup.indices {
                check_quantity_legal(idx, bindings, quantity, errors, seen);
            }
        }
        Expr::UncheckedDim(w) => {
            check_quantity_legal(&w.unchecked_dim.inner, bindings, quantity, errors, seen);
        }
        Expr::Reduce(w) => {
            for t in &w.reduce {
                check_quantity_legal(t, bindings, quantity, errors, seen);
            }
        }
        Expr::BindingRef(w) => {
            let name = w.binding_ref.as_str();
            if let Some(&body) = bindings.get(name) {
                // Insert→recurse→remove: bounds a malformed cycle without barring
                // a binding legitimately reached via two paths in the DAG.
                if seen.insert(name) {
                    check_quantity_legal(body, bindings, quantity, errors, seen);
                    seen.remove(name);
                }
            }
            // Unknown binding name → resolved/errored at CompiledModel::new.
        }
    }
}

struct RefCtx<'a> {
    comp_names:  &'a HashSet<&'a str>,
    param_names: &'a HashSet<&'a str>,
    table_names: &'a HashSet<&'a str>,
    /// gh#127 (#12): Inline-table name → linear length, for constant-index
    /// range checks. External tables are absent (length not static).
    table_lens:  &'a std::collections::HashMap<&'a str, usize>,
    tf_names:    &'a HashSet<&'a str>,
}

fn check_expr(expr: &Expr, ctx: &RefCtx<'_>, allow_projected: bool, errors: &mut Vec<ValidationError>) {
    match expr {
        Expr::Const(_) | Expr::Time(_) | Expr::Dt(_) => {}
        Expr::Projected(_) => {
            // Allow in likelihood context; validate at call-site via allow_projected
            // (we pass allow_projected=true from check_likelihood_exprs)
            if !allow_projected {
                // We don't emit an error here currently; the schema validator handles it.
            }
        }
        Expr::Param(p) => {
            if !ctx.param_names.contains(p.param.as_str()) {
                errors.push(ValidationError::UnknownParameter(p.param.clone()));
            }
        }
        Expr::Pop(p) => {
            if !ctx.comp_names.contains(p.pop.as_str()) {
                errors.push(ValidationError::UnknownCompartment(p.pop.clone()));
            }
        }
        Expr::PopSum(ps) => {
            for name in &ps.pop_sum {
                if !ctx.comp_names.contains(name.as_str()) {
                    errors.push(ValidationError::UnknownCompartment(name.clone()));
                }
            }
        }
        Expr::BinOp(w) => {
            check_expr(&w.bin_op.left,  ctx, allow_projected, errors);
            check_expr(&w.bin_op.right, ctx, allow_projected, errors);
        }
        Expr::UnOp(w) => {
            check_expr(&w.un_op.arg, ctx, allow_projected, errors);
        }
        Expr::Cond(w) => {
            check_expr(&w.cond.pred,  ctx, allow_projected, errors);
            check_expr(&w.cond.then,  ctx, allow_projected, errors);
            check_expr(&w.cond.else_, ctx, allow_projected, errors);
        }
        Expr::TimeFunc(w) => {
            if !ctx.tf_names.contains(w.time_func.name.as_str()) {
                errors.push(ValidationError::UnknownTimeFunction(w.time_func.name.clone()));
            }
        }
        Expr::TableLookup(w) => {
            if !ctx.table_names.contains(w.table_lookup.table.as_str()) {
                errors.push(ValidationError::UnknownTable(w.table_lookup.table.clone()));
            }
            // Arity check (gh#123, reviewer feedback on the prior #123 attempt):
            // the IR table is rank-1 — the OCaml compiler pre-flattens any
            // multi-dimensional table to a single linear index, and the
            // runtime evaluator rejects any other count (propensity.rs /
            // resolved_expr.rs). A lookup carrying ≠1 index is malformed IR; we
            // reject it here at the contract boundary rather than deferring to a
            // runtime eval error. This is an item-count (arity) check, NOT an
            // out-of-range linear-index check — the runtime already rejects a
            // fully out-of-range index via OobPolicy::Error (gh#112 is the
            // OCaml-side under-index-selects-wrong-cell fix, not this).
            if w.table_lookup.indices.len() != 1 {
                errors.push(ValidationError::TableLookupArity {
                    table: w.table_lookup.table.clone(),
                    got: w.table_lookup.indices.len(),
                });
            }
            // gh#127 (#12): for a COMPILE-TIME-CONSTANT index against an Inline
            // table, the out-of-range condition is knowable now — reject it
            // here (a named diagnostic) instead of deferring to a runtime
            // panic/SimError. The runtime floors a fractional index before
            // bounds-checking (`raw.floor() as i64`, resolved_expr.rs /
            // propensity.rs), so mirror that flooring exactly to report the
            // same index the runtime would. A non-constant (state/param-
            // dependent) index is not statically range-checkable, so it is left
            // to the runtime (which now returns a SimError, never panics).
            if let (Some(&len), [Expr::Const(c)]) =
                (ctx.table_lens.get(w.table_lookup.table.as_str()), w.table_lookup.indices.as_slice())
            {
                // `floor() as i64` matches the runtime; finite by construction
                // here (a literal Const). Skip non-finite literals — they are a
                // separate domain concern, not a linear-index range error.
                if c.value.is_finite() {
                    let idx = c.value.floor() as i64;
                    if idx < 0 || idx >= len as i64 {
                        errors.push(ValidationError::TableLookupConstantIndexOutOfRange {
                            table: w.table_lookup.table.clone(),
                            index: idx,
                            len,
                        });
                    }
                }
            }
            for idx in &w.table_lookup.indices {
                check_expr(idx, ctx, allow_projected, errors);
            }
        }
        Expr::UncheckedDim(w) => {
            // Recurse into the inner expression for name-resolution
            // checks — the escape only affects dim-check, not name
            // resolution.
            check_expr(&w.unchecked_dim.inner, ctx, allow_projected, errors);
        }
        Expr::Reduce(w) => {
            for t in &w.reduce {
                check_expr(t, ctx, allow_projected, errors);
            }
        }
        // Leaf: binding-name resolution happens at CompiledModel::new (binding_index).
        Expr::BindingRef(_) => {}
        // gh#272 LICM: leaf; per-eval-name resolution happens at CompiledModel::new
        // (per_eval_index), which errors on an unknown name.
        Expr::PerEvalRef(_) => {}
        // Leaf: a per-observation aux data column reference. Like `Projected`,
        // it is only meaningful inside a likelihood; the binder resolves the
        // name against the stream's declared aux columns at load. No
        // model-name resolution applies here.
        Expr::ObsColumnRef(_) => {}
    }
}

fn check_likelihood_exprs(
    likelihood: &crate::observation::Likelihood,
    ctx: &RefCtx<'_>,
    errors: &mut Vec<ValidationError>,
) {
    use crate::observation::Likelihood;
    match likelihood {
        Likelihood::Poisson(l)      => check_expr(&l.rate, ctx, true, errors),
        Likelihood::NegBinomial(l)  => {
            check_expr(&l.mean, ctx, true, errors);
            check_expr(&l.dispersion, ctx, true, errors);
        }
        Likelihood::Normal(l) => {
            check_expr(&l.mean, ctx, true, errors);
            check_expr(&l.sd,   ctx, true, errors);
        }
        Likelihood::Binomial(l) => {
            check_expr(&l.n, ctx, true, errors);
            check_expr(&l.p, ctx, true, errors);
        }
        Likelihood::BetaBinomial(l) => {
            check_expr(&l.n,     ctx, true, errors);
            check_expr(&l.alpha, ctx, true, errors);
            check_expr(&l.beta,  ctx, true, errors);
        }
        Likelihood::Bernoulli(l) => {
            check_expr(&l.p, ctx, true, errors);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::{
        ParamValue, Parameter, PriorDist, NormalPrior, PriorSpec, Transform,
        HierarchicalKind, HierarchicalPrior,
    };

    /// An estimated parameter with the given prior spec. (Prior-and-
    /// hierarchical-both-set is unrepresentable now — `PriorSpec` is one slot
    /// — so the former `param_both_set` helper and its rejection test are gone.)
    fn param_estimated(name: &str, prior: PriorSpec) -> Parameter {
        Parameter {
            name: name.into(),
            value: ParamValue::Estimated {
                init: None, bounds: None, prior, transform: Transform::Identity,
            },
            param_kind: None,
            param_dim:  None,
        }
    }

    fn load_sir() -> Model {
        let s = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"), "/../../../ir/golden/sir_basic.ir.json"))
            .expect("read sir_basic.ir.json");
        // gh#audit-C8. Use envelope-aware deserializer.
        crate::from_str(&s).expect("parse sir_basic")
    }

    #[test]
    fn only_prior_is_accepted() {
        let mut m = load_sir();
        m.parameters.push(param_estimated("beta_extra",
            PriorSpec::Dist(PriorDist::Normal(NormalPrior { mean: 0.0, sd: 1.0 }))));
        validate(&m).expect("a single-level prior must validate");
    }

    #[test]
    fn only_hierarchical_is_accepted() {
        let mut m = load_sir();
        m.parameters.push(param_estimated("beta_extra",
            PriorSpec::Hierarchical(HierarchicalPrior {
                kind: HierarchicalKind::Normal,
                args: Default::default(),
                pool_over: "".into(),
            })));
        validate(&m).expect("a hierarchical prior must validate");
    }

    // ── gh#123: reference checks for intervention/event targets, balance,
    //    init keys, and table-lookup arity ──────────────────────────────────

    use crate::intervention::{
        Action, FractionTransfer, Intervention, InterventionSchedule, SetAction,
    };
    use crate::model::{BalanceSpec, InitialConditions};
    use crate::expr::{Expr, TableLookupExpr, TableLookupWrap};
    use crate::table::{OobPolicy, Table, TableSource};

    /// (1a) An intervention `set`/`add` action whose target compartment does
    /// not exist must be rejected. The runtime would otherwise silently no-op
    /// or panic on an out-of-range index (gh#123).
    #[test]
    fn intervention_set_target_unknown_compartment_is_rejected() {
        let mut m = load_sir();
        m.interventions.push(Intervention {
            name: "shock".into(),
            base_name: None,
            fire: crate::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![10.0])),
            actions: vec![Action::Set(SetAction {
                compartment: "Q".into(), // not declared (model has S, I, R)
                value: Expr::const_(0.0),
            })],
            kind: crate::intervention::InterventionKind::Scenario,
        });
        let errs = validate(&m).expect_err("must reject intervention targeting unknown 'Q'");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::UnknownCompartmentInIntervention { intervention, compartment }
                    if intervention == "shock" && compartment == "Q")),
            "expected UnknownCompartmentInIntervention for 'shock'/'Q', got: {:?}", errs);
    }

    /// (1b) An event (always_active intervention) `transfer` action whose
    /// `dst` does not exist must be rejected — events fire every substep, so a
    /// dangling target is a hard model bug.
    #[test]
    fn event_transfer_dst_unknown_compartment_is_rejected() {
        let mut m = load_sir();
        m.interventions.push(Intervention {
            name: "import".into(),
            base_name: None,
            fire: crate::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
            actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(),         // declared
                dst: "Nowhere".into(),   // not declared
                fraction: Expr::const_(0.1),
            })],
            kind: crate::intervention::InterventionKind::Event,
        });
        let errs = validate(&m).expect_err("must reject transfer to unknown 'Nowhere'");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::UnknownCompartmentInIntervention { intervention, compartment }
                    if intervention == "import" && compartment == "Nowhere")),
            "expected UnknownCompartmentInIntervention for 'import'/'Nowhere', got: {:?}", errs);
    }

    /// (2) A balance constraint whose target compartment does not exist must be
    /// rejected. The runtime overwrites the target each substep; a dangling
    /// target silently does nothing.
    #[test]
    fn balance_target_unknown_compartment_is_rejected() {
        let mut m = load_sir();
        m.balance = Some(BalanceSpec {
            target: "Residual".into(), // not declared
            expr: Expr::const_(0.0),
        });
        let errs = validate(&m).expect_err("must reject balance targeting unknown 'Residual'");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::UnknownCompartmentInBalance(c) if c == "Residual")),
            "expected UnknownCompartmentInBalance for 'Residual', got: {:?}", errs);
    }

    /// (3) gh#114 Rust-side: an initial-condition key that does not resolve to
    /// a declared (expanded) compartment must be rejected. A stratified model
    /// can otherwise carry an init value for nonexistent `S` while the real
    /// cells default to zero — a plausible-but-wrong epidemic.
    #[test]
    fn init_key_unknown_compartment_is_rejected() {
        let mut m = load_sir();
        // sir_basic uses Parameterized init keyed on S/I; add a dangling key.
        match &mut m.initial_conditions {
            InitialConditions::Parameterized(map) => {
                map.insert("S_ghost".into(), Expr::const_(0.0));
            }
            other => panic!("expected Parameterized init in sir_basic, got {:?}", other),
        }
        let errs = validate(&m).expect_err("must reject init key for unknown 'S_ghost'");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::UnknownCompartmentInInitialConditions(c) if c == "S_ghost")),
            "expected UnknownCompartmentInInitialConditions for 'S_ghost', got: {:?}", errs);
    }

    /// (4) A table-lookup whose index ARITY differs from the IR table's rank
    /// (1, since the compiler pre-flattens multi-dim tables to a single linear
    /// index) must be rejected by validation, not deferred to a runtime eval
    /// error. This is the arity check, NOT an out-of-range linear index (the
    /// runtime already rejects out-of-range via OobPolicy::Error).
    #[test]
    fn table_lookup_wrong_arity_is_rejected() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        // A two-index lookup against the rank-1 IR table: wrong arity.
        let two_index_lookup = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(0.0), Expr::const_(1.0)],
            },
        });
        // Plant the lookup in a transition rate (a checked Expr location).
        m.transitions[0].rate = two_index_lookup;
        let errs = validate(&m).expect_err("must reject 2-index lookup against rank-1 table");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::TableLookupArity { table, got } if table == "kernel" && *got == 2)),
            "expected TableLookupArity for 'kernel' got=2, got: {:?}", errs);
    }

    /// Negative control for arity: a correct single-index lookup must validate.
    #[test]
    fn table_lookup_single_index_is_accepted() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(0.0)],
            },
        });
        validate(&m).expect("single-index lookup against rank-1 table must validate");
    }

    // ── gh#127 (#12): constant out-of-range table index rejected at validate ──
    //
    // The runtime fast evaluator (resolved_expr.rs) panicked on an out-of-range
    // table index under OobPolicy::Error. For a COMPILE-TIME-CONSTANT index the
    // out-of-range condition is knowable at validate time, so reject it here —
    // a named diagnostic (table + bad index + size) rather than a deferred
    // runtime crash. The non-constant (state/param-dependent) case is handled
    // at eval time (returns a SimError, never a panic).

    /// A constant index past the end of an Inline table must be rejected by
    /// validate(), naming the table, the index, and the table size.
    #[test]
    fn table_lookup_constant_index_above_range_is_rejected() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                // length 2 → valid indices are {0, 1}.
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        // index 5 is out of range for a 2-cell table.
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(5.0)],
            },
        });
        let errs = validate(&m).expect_err("must reject constant index 5 against 2-cell table");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::TableLookupConstantIndexOutOfRange { table, index, len }
                    if table == "kernel" && *index == 5 && *len == 2)),
            "expected TableLookupConstantIndexOutOfRange for 'kernel' index=5 len=2, got: {:?}", errs);
    }

    /// A negative constant index must be rejected — the runtime floors to a
    /// negative i64, which is out of range for any table.
    #[test]
    fn table_lookup_constant_index_negative_is_rejected() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(-1.0)],
            },
        });
        let errs = validate(&m).expect_err("must reject constant index -1");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::TableLookupConstantIndexOutOfRange { table, index, len }
                    if table == "kernel" && *index == -1 && *len == 2)),
            "expected TableLookupConstantIndexOutOfRange for 'kernel' index=-1 len=2, got: {:?}", errs);
    }

    /// The runtime floors a fractional index before bounds-checking (`raw.floor()
    /// as i64`). A constant `2.9` against a 2-cell table floors to 2 → out of
    /// range, and must be rejected with the FLOORED index (matching runtime).
    #[test]
    fn table_lookup_constant_fractional_index_uses_floor() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(2.9)],
            },
        });
        let errs = validate(&m).expect_err("must reject constant index 2.9 (floors to 2)");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::TableLookupConstantIndexOutOfRange { table, index, len }
                    if table == "kernel" && *index == 2 && *len == 2)),
            "expected TableLookupConstantIndexOutOfRange for 'kernel' index=2 (floor of 2.9) len=2, got: {:?}", errs);
    }

    /// Negative control: a constant index that IS in range must validate (the
    /// last valid index, len-1, is accepted). Guards against an off-by-one that
    /// would reject the boundary.
    #[test]
    fn table_lookup_constant_index_in_range_is_accepted() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::const_(1.0)], // last valid index for a 2-cell table
            },
        });
        validate(&m).expect("constant index 1 (in range for a 2-cell table) must validate");
    }

    /// Negative control: a NON-constant index (references compartment state)
    /// must NOT be rejected at validate time even if it could be out of range
    /// at runtime — the in-range property is not statically knowable. The
    /// runtime handles it (SimError, not panic).
    #[test]
    fn table_lookup_nonconstant_index_is_not_range_checked() {
        let mut m = load_sir();
        m.tables.push(Table {
            name: "kernel".into(),
            source: TableSource::Inline {
                values: vec![Expr::const_(1.0), Expr::const_(2.0)],
            },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        // pop("I") is state-dependent — its value is unknown at validate time.
        m.transitions[0].rate = Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::pop("I")],
            },
        });
        validate(&m).expect("a state-dependent table index must not be statically range-checked");
    }

    /// Negative control: the unmodified sir_basic model (with valid init keys,
    /// no interventions, no balance) must validate.
    #[test]
    fn sir_basic_validates() {
        let m = load_sir();
        validate(&m).expect("sir_basic.ir.json must validate");
    }

    // ── gh#124: explicit initial-condition VALUE domain checks ────────────────
    //
    // The runtime converts an explicit integer init via `*val as i64`
    // (compiled_model.rs), which truncates and saturates: I0=0.6 → 0 silently,
    // I0=-3 → a negative compartment from t=0, I0=NaN → 0, I0=1e20 → i64::MAX.
    // Each is a "model runs but starts in the wrong population" failure. Reject
    // them at the contract boundary instead.

    /// sir_basic is Parameterized; swap in an Explicit init map keyed on the
    /// model's (integer) compartments so the VALUE-domain checks have something
    /// to inspect.
    fn sir_with_explicit_init(s: f64, i: f64, r: f64) -> Model {
        let mut m = load_sir();
        let mut map = std::collections::HashMap::new();
        map.insert("S".to_string(), s);
        map.insert("I".to_string(), i);
        map.insert("R".to_string(), r);
        m.initial_conditions = InitialConditions::Explicit(map);
        m
    }

    /// (124a) A negative explicit init value must be rejected — a negative
    /// compartment from t=0 is never physical (population counts are
    /// nonnegative). Reproduces the `I0 = -3` row of gh#124.
    #[test]
    fn init_value_negative_is_rejected() {
        let m = sir_with_explicit_init(99.0, -3.0, 0.0);
        let errs = validate(&m).expect_err("must reject I0 = -3");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::InitialValueNegative { compartment, value }
                    if compartment == "I" && *value == -3.0)),
            "expected InitialValueNegative for 'I' = -3, got: {:?}", errs);
    }

    /// (124b) A non-finite explicit init value (NaN) must be rejected — it
    /// converts to 0 under `as i64` with no warning. Reproduces the
    /// `I0 = NaN` row of gh#124.
    #[test]
    fn init_value_nan_is_rejected() {
        let m = sir_with_explicit_init(99.0, f64::NAN, 0.0);
        let errs = validate(&m).expect_err("must reject I0 = NaN");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::InitialValueNotFinite { compartment, .. }
                    if compartment == "I")),
            "expected InitialValueNotFinite for 'I' = NaN, got: {:?}", errs);
    }

    /// (124b') A positive-infinity explicit init value must be rejected — it
    /// saturates to i64::MAX under `as i64`. Same NaN/inf class as above.
    #[test]
    fn init_value_inf_is_rejected() {
        let m = sir_with_explicit_init(99.0, f64::INFINITY, 0.0);
        let errs = validate(&m).expect_err("must reject I0 = inf");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::InitialValueNotFinite { compartment, .. }
                    if compartment == "I")),
            "expected InitialValueNotFinite for 'I' = inf, got: {:?}", errs);
    }

    /// (124c) A clearly-fractional explicit init value on an INTEGER
    /// compartment must be rejected, not silently truncated. Reproduces the
    /// `I0 = 0.6` row of gh#124 (which `as i64` truncates to 0).
    #[test]
    fn init_value_fractional_on_integer_compartment_is_rejected() {
        let m = sir_with_explicit_init(99.0, 0.6, 0.0);
        let errs = validate(&m).expect_err("must reject I0 = 0.6 on integer compartment");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::InitialValueNotInteger { compartment, value }
                    if compartment == "I" && *value == 0.6)),
            "expected InitialValueNotInteger for 'I' = 0.6, got: {:?}", errs);
    }

    /// Negative control: integer-valued explicit inits (including a within-
    /// tolerance near-integer like 3.0 + 1e-12) must validate.
    #[test]
    fn init_value_integer_on_integer_compartment_is_accepted() {
        let m = sir_with_explicit_init(99.0, 1.0 + 1e-12, 0.0);
        validate(&m).expect("near-integer init within tolerance must validate");
    }

    /// (124d) A fractional value on a REAL compartment must be accepted — real
    /// compartments may hold fractional (but nonnegative, finite) values.
    #[test]
    fn init_value_fractional_on_real_compartment_is_accepted() {
        let mut m = load_sir();
        // Make R a real compartment with an ODE so the model still validates
        // structurally, then give it a fractional init.
        for c in &mut m.compartments {
            if c.name == "R" {
                c.kind = CompartmentKind::Real;
            }
        }
        m.ode_equations.push(crate::ode_equation::OdeEquation {
            compartment: "R".into(),
            derivative: Expr::const_(0.0),
        });
        // R no longer participates in integer stoichiometry in sir_basic's
        // recovery transition; drop any stoichiometry entry naming R so the
        // RealCompartmentInStoichiometry check doesn't fire (we're isolating
        // the init-VALUE domain behaviour, not stoichiometry).
        for tr in &mut m.transitions {
            tr.stoichiometry.retain(|e| e.0 != "R");
        }
        let mut map = std::collections::HashMap::new();
        map.insert("S".to_string(), 99.0);
        map.insert("I".to_string(), 1.0);
        map.insert("R".to_string(), 0.6); // fractional on a real compartment: OK
        m.initial_conditions = InitialConditions::Explicit(map);
        validate(&m).expect("fractional init on a real compartment must validate");
    }

    /// (124e) A negative value on a REAL compartment must still be rejected —
    /// population values are nonnegative regardless of int/real.
    #[test]
    fn init_value_negative_on_real_compartment_is_rejected() {
        let mut m = load_sir();
        for c in &mut m.compartments {
            if c.name == "R" {
                c.kind = CompartmentKind::Real;
            }
        }
        m.ode_equations.push(crate::ode_equation::OdeEquation {
            compartment: "R".into(),
            derivative: Expr::const_(0.0),
        });
        for tr in &mut m.transitions {
            tr.stoichiometry.retain(|e| e.0 != "R");
        }
        let mut map = std::collections::HashMap::new();
        map.insert("S".to_string(), 99.0);
        map.insert("I".to_string(), 1.0);
        map.insert("R".to_string(), -0.5);
        m.initial_conditions = InitialConditions::Explicit(map);
        let errs = validate(&m).expect_err("must reject negative init on real compartment");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::InitialValueNegative { compartment, value }
                    if compartment == "R" && *value == -0.5)),
            "expected InitialValueNegative for 'R' = -0.5, got: {:?}", errs);
    }

    /// Regression guard for the gh#123/gh#114 reference checks: every committed
    /// golden IR (which exercises real interventions, balance, stratified init,
    /// and table lookups) must still validate. A false positive in the new
    /// checks — rejecting legitimate compiler-emitted IR — would surface here.
    #[test]
    fn all_golden_ir_validates() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ir/golden");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(dir).expect("read ir/golden dir") {
            let path = entry.expect("dir entry").path();
            let is_ir = path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".ir.json"))
                .unwrap_or(false);
            if !is_ir {
                continue;
            }
            let s = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let m = crate::from_str(&s)
                .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
            validate(&m)
                .unwrap_or_else(|errs| panic!("{} must validate, got: {:?}", path.display(), errs));
            checked += 1;
        }
        assert!(checked > 0, "no golden .ir.json files found under {dir}");
    }

    // ── Generated quantities: state-expression legality (proposal 2026-06-25) ──
    use crate::quantity::{
        Quantity, QuantityBody, QuantitySource, TemporalReduce, ValueReduce, TimeReduce,
    };
    use crate::model::Binding;

    /// A quantity whose State expr DIRECTLY contains a forbidden leaf (`dt` — the
    /// integrator step is meaningless in a quantity read at output cadence) must
    /// be rejected at the load boundary.
    #[test]
    fn quantity_state_expr_with_dt_is_rejected() {
        let mut m = load_sir();
        m.quantities.push(Quantity {
            name: "bad".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::bin_op(
                    crate::expr::BinOp::Mul, Expr::pop("I"), Expr::dt())),
                reduce: None,
            },
        });
        let errs = validate(&m).expect_err("a quantity with `dt` must be rejected");
        assert!(errs.iter().any(|e| matches!(e,
            ValidationError::QuantityForbiddenLeaf { quantity, leaf }
                if quantity == "bad" && leaf == "dt")),
            "expected QuantityForbiddenLeaf(bad, dt), got: {:?}", errs);
    }

    /// The round-2 smuggle: a quantity's State expr is a `BindingRef` to a model
    /// binding whose BODY contains a forbidden leaf. A constructor-only check over
    /// the quantity Expr alone cannot see this; the validate context recurses
    /// transitively over `model.bindings`. (`dt` is legal in a binding used by a
    /// rate, so the binding itself validates — only the quantity reaching it is
    /// illegal.)
    #[test]
    fn quantity_binding_ref_to_forbidden_leaf_is_rejected_transitively() {
        let mut m = load_sir();
        m.bindings.push(Binding {
            name: "poison".into(),
            expr: Expr::bin_op(crate::expr::BinOp::Mul, Expr::pop("I"), Expr::dt()),
        });
        m.quantities.push(Quantity {
            name: "smuggle".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::binding_ref("poison")),
                reduce: None,
            },
        });
        let errs = validate(&m)
            .expect_err("a quantity reaching `dt` via a binding must be rejected");
        assert!(errs.iter().any(|e| matches!(e,
            ValidationError::QuantityForbiddenLeaf { quantity, leaf }
                if quantity == "smuggle" && leaf == "dt")),
            "expected transitive QuantityForbiddenLeaf(smuggle, dt), got: {:?}", errs);
    }

    /// A forbidden leaf hidden inside a reduction THRESHOLD must also be caught
    /// (the deep walk covers thresholds, not just the source expr).
    #[test]
    fn quantity_reduction_threshold_forbidden_leaf_rejected() {
        let mut m = load_sir();
        m.quantities.push(Quantity {
            name: "thr".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I")),
                reduce: Some(TemporalReduce::Time(TimeReduce::FirstAbove(
                    Expr::Projected(crate::expr::ProjectedExpr { projected: () })))),
            },
        });
        let errs = validate(&m).expect_err("a threshold with `projected` must be rejected");
        assert!(errs.iter().any(|e| matches!(e,
            ValidationError::QuantityForbiddenLeaf { quantity, leaf }
                if quantity == "thr" && leaf == "projected")),
            "expected QuantityForbiddenLeaf(thr, projected), got: {:?}", errs);
    }

    /// A clean quantity (state arithmetic + a `BindingRef` to a clean binding +
    /// a param threshold) must VALIDATE — the context must not over-reject.
    #[test]
    fn clean_quantity_validates() {
        let mut m = load_sir();
        m.bindings.push(Binding {
            name: "Ntot".into(),
            expr: Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]),
        });
        m.quantities.push(Quantity {
            name: "prev".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::bin_op(
                    crate::expr::BinOp::Div, Expr::pop("I"), Expr::binding_ref("Ntot"))),
                reduce: Some(TemporalReduce::Value(ValueReduce::Max)),
            },
        });
        m.quantities.push(Quantity {
            name: "onset".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I")),
                reduce: Some(TemporalReduce::Time(TimeReduce::FirstAbove(Expr::param("beta")))),
            },
        });
        validate(&m).expect("a clean quantity must validate");
    }

    /// A quantity referencing an unknown compartment must still be name-checked
    /// (the existing `check_expr` runs on quantity exprs too).
    #[test]
    fn quantity_unknown_compartment_is_rejected() {
        let mut m = load_sir();
        m.quantities.push(Quantity {
            name: "q".into(),
            stratum: vec![],
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("Q")), // not declared
                reduce: None,
            },
        });
        let errs = validate(&m).expect_err("unknown compartment in a quantity must be rejected");
        assert!(errs.iter().any(|e| matches!(e,
            ValidationError::UnknownCompartment(c) if c == "Q")),
            "expected UnknownCompartment(Q), got: {:?}", errs);
    }
}
