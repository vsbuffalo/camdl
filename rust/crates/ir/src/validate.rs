use std::collections::HashSet;
use thiserror::Error;
use crate::{
    expr::Expr,
    model::{CompartmentKind, Model},
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

    #[error("parameter '{0}': prior and hierarchical are mutually exclusive — \
             a parameter is either fitted under a single-level prior or pooled \
             under a hierarchical prior, not both")]
    PriorAndHierarchicalBothSet(String),
}

pub fn validate(model: &Model) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // ── Build name sets ───────────────────────────────────────────────────────

    let mut comp_names:  HashSet<&str> = HashSet::new();
    let mut real_comps:  HashSet<&str> = HashSet::new();
    let mut int_comps:   HashSet<&str> = HashSet::new();
    let mut param_names: HashSet<&str> = HashSet::new();
    let mut table_names: HashSet<&str> = HashSet::new();
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
        if p.prior.is_some() && p.hierarchical.is_some() {
            errors.push(ValidationError::PriorAndHierarchicalBothSet(p.name.clone()));
        }
    }
    for t in &model.tables {
        table_names.insert(t.name.as_str());
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

    let ctx = RefCtx { comp_names: &comp_names, param_names: &param_names, table_names: &table_names, tf_names: &tf_names };

    for tr in &model.transitions {
        check_expr(&tr.rate, &ctx, false, &mut errors);
    }
    for eq in &model.ode_equations {
        check_expr(&eq.derivative, &ctx, false, &mut errors);
    }
    for obs in &model.observations {
        // projection
        if let crate::observation::Projection::CumulativeFlow(ref tn) = obs.projection {
            // A bare transition name over a stratified family (e.g. `infection`
            // when only `infection_child`, `infection_adult` exist) is the
            // documented "sum over all strata" form (language spec §25.4). The
            // runtime resolves it by matching `tn` exactly OR any `tn_*` family
            // member (see multi_stream_obs.rs::from_ir and
            // main.rs::project_all_obs_times), so validation must accept the
            // same set or it diverges from simulation.
            let prefix = format!("{}_", tn);
            let ok = tr_names.contains(tn.as_str())
                || tr_names.iter().any(|n| n.starts_with(&prefix));
            if !ok {
                errors.push(ValidationError::UnknownTransitionInObservation {
                    obs: obs.name.clone(),
                    transition: tn.clone(),
                });
            }
        }
        // likelihood exprs (projected is allowed)
        check_likelihood_exprs(&obs.likelihood, &ctx, &mut errors);
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(())
}

struct RefCtx<'a> {
    comp_names:  &'a HashSet<&'a str>,
    param_names: &'a HashSet<&'a str>,
    table_names: &'a HashSet<&'a str>,
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
    use crate::parameter::{Parameter, PriorDist, NormalPrior, HierarchicalKind, HierarchicalPrior};

    fn param_both_set() -> Parameter {
        Parameter {
            name:          "beta".into(),
            value:         Some(1.0),
            bounds:        None,
            prior:         Some(PriorDist::Normal(NormalPrior { mean: 0.0, sd: 1.0 })),
            hierarchical:  Some(HierarchicalPrior {
                kind: HierarchicalKind::Normal,
                args: Default::default(),
                pool_over: "".into(),
            }),
            transform:     None,
            initial_value: None,
            param_kind:    None,
            param_dim:     None,
        }
    }

    fn param_only_prior() -> Parameter {
        let mut p = param_both_set();
        p.hierarchical = None;
        p
    }

    fn param_only_hierarchical() -> Parameter {
        let mut p = param_both_set();
        p.prior = None;
        p
    }

    fn load_sir() -> Model {
        let s = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"), "/../../../ir/golden/sir_basic.ir.json"))
            .expect("read sir_basic.ir.json");
        // gh#audit-C8. Use envelope-aware deserializer.
        crate::from_str(&s).expect("parse sir_basic")
    }

    #[test]
    fn prior_and_hierarchical_both_set_is_rejected() {
        let mut m = load_sir();
        m.parameters.push(param_both_set());
        let errs = validate(&m).expect_err("must reject parameter with both fields set");
        assert!(errs.iter().any(|e| matches!(e,
            ValidationError::PriorAndHierarchicalBothSet(name) if name == "beta")),
            "expected PriorAndHierarchicalBothSet for 'beta', got: {:?}", errs);
    }

    #[test]
    fn only_prior_is_accepted() {
        let mut m = load_sir();
        // Use a fresh name to avoid the duplicate-parameter check tripping.
        let mut p = param_only_prior();
        p.name = "beta_extra".into();
        m.parameters.push(p);
        validate(&m).expect("only prior set must validate");
    }

    #[test]
    fn only_hierarchical_is_accepted() {
        let mut m = load_sir();
        let mut p = param_only_hierarchical();
        p.name = "beta_extra".into();
        m.parameters.push(p);
        validate(&m).expect("only hierarchical set must validate");
    }

    // ── Regression: bare CumulativeFlow stem over a stratified family ────────
    // See models/camdl_issues/ISSUE_1_incidence_stratified.md. A bare
    // `incidence(infection)` over a stratified transition expands to a bare
    // `CumulativeFlow("infection")` whose name does not exist post-expansion
    // (only `infection_child`, `infection_adult`, … do). The runtime resolves
    // it as the family sum, so validation must accept the stem too — otherwise
    // `check`/`compile` diverge from `simulate` (the original E507).

    use crate::observation::{
        Likelihood, ObservationModel, ObservationSchedule, PoissonLikelihood, Projection,
    };

    /// A trivial Poisson obs over `CumulativeFlow(stem)`, for validate tests.
    fn obs_cumflow(name: &str, stem: &str) -> ObservationModel {
        ObservationModel {
            name:       name.into(),
            schedule:   ObservationSchedule::AtTimes(vec![1.0]),
            projection: Projection::CumulativeFlow(stem.into()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: Expr::const_(1.0),
            }),
        }
    }

    /// Rename sir_basic's `infection` transition to a stratified family so the
    /// post-expansion transition set is `{infection_child, infection_adult,
    /// recovery}` and there is no bare `infection`.
    fn stratify_infection(m: &mut Model) {
        let proto = m
            .transitions
            .iter()
            .find(|t| t.name == "infection")
            .expect("sir_basic has an `infection` transition")
            .clone();
        m.transitions.retain(|t| t.name != "infection");
        for stratum in ["child", "adult"] {
            let mut t = proto.clone();
            t.name = format!("infection_{stratum}");
            m.transitions.push(t);
        }
    }

    #[test]
    fn bare_stratified_incidence_stem_is_accepted() {
        let mut m = load_sir();
        stratify_infection(&mut m);
        m.observations.push(obs_cumflow("weekly_cases", "infection"));
        validate(&m).expect(
            "bare CumulativeFlow stem over a stratified family must validate \
             (matches runtime family-sum semantics)",
        );
    }

    #[test]
    fn unknown_transition_in_observation_is_rejected() {
        let mut m = load_sir();
        stratify_infection(&mut m);
        m.observations.push(obs_cumflow("bad", "nonexistent"));
        let errs = validate(&m).expect_err("a truly-unknown transition must be rejected");
        assert!(
            errs.iter().any(|e| matches!(e,
                ValidationError::UnknownTransitionInObservation { transition, .. }
                    if transition == "nonexistent")),
            "expected UnknownTransitionInObservation for 'nonexistent', got: {errs:?}"
        );
    }

    #[test]
    fn exact_transition_match_still_accepted() {
        // The exact-name path (non-stratified) must keep working.
        let mut m = load_sir();
        m.observations.push(obs_cumflow("rec", "recovery"));
        validate(&m).expect("exact transition-name match must validate");
    }
}
