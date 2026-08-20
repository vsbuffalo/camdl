//! Runtime resolution of a model's observation anchors (gh#616).
//!
//! The compiler emits three anchored constructs — `simulate { to }`, a preset's
//! `simulate { to }`, and a `piecewise` forcing's `breakpoints` — and leaves
//! each SYMBOLIC, because their value is a property of the run, not of the
//! model. This module is where they become numbers.
//!
//! # Why substitution happens on the model, and where
//!
//! [`substitute`] rewrites the model in place: it writes the resolved time into
//! `simulation.t_end` / a preset's `t_end` and CLEARS the anchor field, and
//! replaces each `Expr::ObsAnchor` knot with a `Const`. That is what makes the
//! resolution visible to run identity — `Model::hash_into` walks `simulation`,
//! `presets` and `time_functions`, so two data vintages that move `last_obs`
//! produce different model digests and cannot share a `run_id`.
//!
//! It follows that the substitution must happen on the model that is **hashed**,
//! not only on the one that is run. On the simulate path those are two separate
//! loads of the same IR file (`build_simulate_cas_sink`'s `base_model` and
//! `resolve_run_model`'s), so both call this function with the same
//! [`ObsAnchorTimes`] — same input, same output, agreement by construction
//! rather than by convention. `profile` and `survey` key their runs by the IR
//! **text** instead (`resolve::model_identity_from_ir`), so there the caller
//! re-emits that text after substituting — the `bool` [`resolve_from_bindings`]
//! returns is exactly that cue.
//!
//! The same rule applies within a single command: EVERY consumer that reads a
//! horizon must read the substituted model. A second fresh load of the compiled
//! IR re-introduces the unresolved marker and every horizon read off it is NaN,
//! which compares equal to nothing — `simulate --obs` shipped exactly that bug
//! and refused with `baseline -> t = NaN` (fixed in 7af5c9fa).
//!
//! # The landmine for `value_at` (F23)
//!
//! `Model::hash_into` deliberately EXCLUDES `quantities` (and `contrasts`) —
//! they are reporting-only and must never re-key a run. So the argument above
//! does **not** extend to `value_at(…, last_obs)`: resolving a quantity anchor
//! changes no hashed field, and two data vintages would share a `run_id` while
//! reporting different numbers. Quantity anchors therefore still refuse under
//! `simulate` (F23), and whoever lifts that restriction MUST key the resolved
//! anchor explicitly — there is no model field that will do it for them.

use ir::anchor::{AnchoredTime, ObsAnchorTimes};
use ir::expr::Expr;

/// One resolved anchor, for the stderr line and the run record.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnchor {
    /// What was anchored, in the model author's vocabulary:
    /// `"simulate { to }"`, `"scenario 'forecast' { to }"`, `"forcing 'ramp'"`.
    pub site: String,
    /// The symbolic form, as written (`last_obs + 28`).
    pub anchor: AnchoredTime,
    /// The model time it resolved to.
    pub resolved: f64,
}

/// Does this model carry any unresolved observation anchor? The cue for a
/// caller to bind observation data before compiling — and, when it cannot, to
/// refuse by name rather than let `CompiledModel::new` do it with less context.
pub fn model_is_anchored(model: &ir::Model) -> bool {
    model.simulation.t_end_anchor.is_some()
        || model.presets.iter().any(|p| p.t_end_anchor.is_some())
        || model.time_functions.iter().any(|tf| {
            forcing_exprs(&tf.kind).into_iter().any(expr_has_anchor)
        })
}

/// Resolve every observation anchor in `model` against `bound` — the
/// observation streams the calling command has ALREADY resolved from its
/// `--data` flags or its `--fit` toml — substituting in place and announcing
/// each resolved time on stderr.
///
/// This is the seam for the three fixed-θ commands (`pfilter`, `profile`,
/// `survey`). Each of them binds observation data before it scores anything, so
/// the window it anchors to is by construction the window it scores; folding
/// from a second, separately-resolved source is how the two would drift apart.
///
/// A model with no anchor is left byte-identical, nothing is printed, and
/// `false` comes back — so a caller may run this unconditionally. `true` means
/// the model MOVED, which is the caller's cue to re-emit any serialized copy it
/// keys the run by. When the model IS anchored and the bindings yield no
/// observation times, the error names each anchored construct — "resolve it" is
/// not actionable without knowing which construct needs the data.
pub fn resolve_from_bindings(
    model: &mut ir::Model,
    bound: &[(String, std::path::PathBuf)],
    dt: f64,
) -> Result<bool, String> {
    resolve_with(model, |m| crate::obs_anchors_from_bindings(m, bound, dt))
}

/// As [`resolve_from_bindings`], for a command holding a fit config rather than
/// a list of bindings (`survey --fit`). The two differ ONLY in where the
/// observed window is read from; the substitution, the stderr report and the
/// refusal wording are shared, so the two commands cannot drift on what a
/// resolved anchor means or on what an unresolvable one says.
pub fn resolve_from_config(
    model: &mut ir::Model,
    config: &crate::fit::config_v2::FitConfigV2,
    dt: f64,
) -> Result<bool, String> {
    resolve_with(model, |m| crate::obs_anchors_from_config(m, config, dt))
}

fn resolve_with(
    model: &mut ir::Model,
    read_window: impl FnOnce(&ir::Model) -> Result<(f64, f64), String>,
) -> Result<bool, String> {
    if !model_is_anchored(model) {
        return Ok(false);
    }
    let (first, last) = read_window(model).map_err(|e| {
        format!(
            "this model is anchored to observed data ({}), and the bound \
             observation data cannot resolve it: {e}",
            anchored_sites(model).join(", ")
        )
    })?;
    let moved = substitute(model, ObsAnchorTimes { first, last })?;
    report(&moved, model);
    Ok(true)
}

/// The sentence a data-binding command appends to its own "no data bound"
/// refusal when the model is anchored.
///
/// Without it the user is told to pass `--data` but not that the model *cannot
/// run at all* without it — an anchored construct is unresolvable, not merely
/// unscored. Empty for an unanchored model, so the existing message is
/// byte-identical there.
pub fn unbound_anchor_clause(model: &ir::Model) -> String {
    if !model_is_anchored(model) {
        return String::new();
    }
    format!(
        "\n  This model is also anchored to observed data ({}); an observation \
         anchor resolves only from a bound observation stream, so there is no \
         horizon to run without one.",
        anchored_sites(model).join(", ")
    )
}

/// Human-readable list of the anchored constructs, for a refusal message.
pub fn anchored_sites(model: &ir::Model) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(a) = &model.simulation.t_end_anchor {
        out.push(format!("`simulate {{ to = {a} }}`"));
    }
    for p in &model.presets {
        if let Some(a) = &p.t_end_anchor {
            out.push(format!("scenario '{}' `simulate {{ to = {a} }}`", p.name));
        }
    }
    for tf in &model.time_functions {
        if forcing_exprs(&tf.kind).into_iter().any(expr_has_anchor) {
            out.push(format!("forcing '{}' breakpoints", tf.name));
        }
    }
    out
}

/// Substitute every observation anchor in `model` for its resolved model time,
/// then validate what the substitution could have broken.
///
/// Idempotent: a model with no anchor is left byte-identical and returns an
/// empty list, so a caller may run this unconditionally.
pub fn substitute(
    model: &mut ir::Model,
    anchors: ObsAnchorTimes,
) -> Result<Vec<ResolvedAnchor>, String> {
    let mut resolved = Vec::new();

    if let Some(a) = model.simulation.t_end_anchor.take() {
        let t = anchors.at(a);
        model.simulation.t_end = t;
        resolved.push(ResolvedAnchor { site: "simulate { to }".into(), anchor: a, resolved: t });
    }
    for p in &mut model.presets {
        if let Some(a) = p.t_end_anchor.take() {
            let t = anchors.at(a);
            p.t_end = Some(t);
            resolved.push(ResolvedAnchor {
                site: format!("scenario '{}' {{ to }}", p.name),
                anchor: a,
                resolved: t,
            });
        }
    }
    for tf in &mut model.time_functions {
        for e in forcing_exprs_mut(&mut tf.kind) {
            substitute_expr(e, anchors, &tf_name(&tf.name), &mut resolved);
        }
    }

    // The horizon is also BAKED into the schedules the compiler derived from it,
    // and those bakes carry the same NaN. Rewrite them to the resolved horizon
    // so a resolved model is indistinguishable from one written with that
    // horizon as a literal — otherwise every guard that compares a baked end to
    // the model horizon (`check_baked_recurring_ends`, the reactive monitoring
    // window) would face a NaN that compares equal to nothing, and would answer
    // "this end did not come from the horizon" when it did.
    //
    // The recurring case is unreachable from a compiling model — E336/E337
    // refuse an anchored horizon that a schedule would bake — but it is
    // rewritten anyway so the invariant does not depend on those two guards
    // staying exhaustive.
    let t_end = model.simulation.t_end;
    for o in &mut model.observations {
        if let Some(ir::observation::ObservationSchedule::Regular(r)) = &mut o.emit_schedule {
            if r.end.is_nan() {
                r.end = t_end;
            }
        }
    }
    for iv in &mut model.interventions {
        if let ir::intervention::FireSource::Scheduled(
            ir::intervention::InterventionSchedule::Recurring(r),
        ) = &mut iv.fire
        {
            if r.end.is_nan() {
                r.end = t_end;
            }
        }
    }

    validate_after_substitution(model)?;
    Ok(resolved)
}

fn tf_name(name: &str) -> String {
    format!("forcing '{name}'")
}

fn substitute_expr(
    e: &mut Expr,
    anchors: ObsAnchorTimes,
    site: &str,
    out: &mut Vec<ResolvedAnchor>,
) {
    match e {
        Expr::ObsAnchor(w) => {
            let a = w.obs_anchor;
            let t = anchors.at(a);
            out.push(ResolvedAnchor { site: site.to_string(), anchor: a, resolved: t });
            *e = Expr::const_(t);
        }
        Expr::BinOp(w) => {
            substitute_expr(&mut w.bin_op.left, anchors, site, out);
            substitute_expr(&mut w.bin_op.right, anchors, site, out);
        }
        Expr::UnOp(w) => substitute_expr(&mut w.un_op.arg, anchors, site, out),
        Expr::Cond(w) => {
            substitute_expr(&mut w.cond.pred, anchors, site, out);
            substitute_expr(&mut w.cond.then, anchors, site, out);
            substitute_expr(&mut w.cond.else_, anchors, site, out);
        }
        Expr::TableLookup(w) => {
            for ix in &mut w.table_lookup.indices {
                substitute_expr(ix, anchors, site, out);
            }
        }
        Expr::Reduce(w) => {
            for t in &mut w.reduce {
                substitute_expr(t, anchors, site, out);
            }
        }
        Expr::UncheckedDim(w) => substitute_expr(&mut w.unchecked_dim.inner, anchors, site, out),
        Expr::Const(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_) | Expr::Time(_)
        | Expr::Dt(_) | Expr::TimeFunc(_) | Expr::Projected(_) | Expr::ObsColumnRef(_)
        | Expr::BindingRef(_) | Expr::PerEvalRef(_) => {}
    }
}

/// The checks that only become possible — and only become NECESSARY — once the
/// anchors carry values.
///
/// A piecewise forcing's knots are read by `piecewise_value`, an order-dependent
/// scan: it walks the knots and returns the value for the first one the query
/// time is below. Nothing validated that the knots were sorted, because with
/// literal knots a model author could see the order. An anchored fork can
/// INVERT under one data vintage and not another (`[last_obs, 60]` is ordered
/// while `last_obs < 60` and inverted after), so the check has to run here, per
/// run, against the resolved values — and the failure it prevents is silent: an
/// unsorted knot list returns a wrong step, not an error.
fn validate_after_substitution(model: &ir::Model) -> Result<(), String> {
    let t_start = model.simulation.t_start;
    let t_end = model.simulation.t_end;
    if !(t_start.is_finite() && t_end.is_finite() && t_end > t_start) {
        return Err(format!(
            "the resolved simulation horizon is not a forward interval: \
             t_start = {t_start}, t_end = {t_end}. An anchored `simulate {{ to }}` \
             resolved to a time at or before the simulation start — check the \
             offset's sign, or whether the bound data ends before `from`."
        ));
    }
    for p in &model.presets {
        if let Some(te) = p.t_end {
            if !(te.is_finite() && te > t_start) {
                return Err(format!(
                    "scenario '{}' resolved to a horizon of {te}, at or before \
                     t_start = {t_start}.",
                    p.name
                ));
            }
        }
    }

    for tf in &model.time_functions {
        let ir::time_func::TimeFuncKind::Piecewise(p) = &tf.kind else { continue };
        // Only constant knots can be checked; a param-valued knot is not known
        // here and is the pre-existing (unanchored) behaviour.
        let knots: Vec<f64> = p.breakpoints.iter().filter_map(const_value).collect();
        if knots.len() != p.breakpoints.len() {
            continue;
        }
        if p.values.len() != p.breakpoints.len() + 1 {
            return Err(format!(
                "forcing '{}' has {} breakpoint(s) but {} value(s); a piecewise \
                 forcing needs exactly one more value than knots (the value \
                 before the first knot, then one per interval).",
                tf.name,
                p.breakpoints.len(),
                p.values.len()
            ));
        }
        for w in knots.windows(2) {
            if !(w[1] >= w[0]) {
                return Err(format!(
                    "forcing '{}' has non-monotone breakpoints after resolving its \
                     observation anchors: {:?}. The runtime reads a piecewise \
                     forcing by scanning the knots in order, so an out-of-order \
                     knot silently selects the wrong step rather than erroring. \
                     Check the anchor offsets against the literal knots.",
                    tf.name, knots
                ));
            }
        }
        if let Some(&k) = knots.first() {
            if k <= t_start {
                return Err(format!(
                    "forcing '{}' has a resolved breakpoint at t = {k}, at or \
                     before t_start = {t_start}. A knot outside the simulation \
                     window selects nothing and the interval before it is never \
                     used.",
                    tf.name
                ));
            }
        }
    }
    Ok(())
}

fn const_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Const(c) => Some(c.value),
        // Unit literals (`60 'days`) lower to a dimensional escape around a
        // constant; see through it, as every other analysis does.
        Expr::UncheckedDim(w) => const_value(&w.unchecked_dim.inner),
        _ => None,
    }
}

fn expr_has_anchor(e: &Expr) -> bool {
    match e {
        Expr::ObsAnchor(_) => true,
        Expr::BinOp(w) => expr_has_anchor(&w.bin_op.left) || expr_has_anchor(&w.bin_op.right),
        Expr::UnOp(w) => expr_has_anchor(&w.un_op.arg),
        Expr::Cond(w) => {
            expr_has_anchor(&w.cond.pred)
                || expr_has_anchor(&w.cond.then)
                || expr_has_anchor(&w.cond.else_)
        }
        Expr::TableLookup(w) => w.table_lookup.indices.iter().any(expr_has_anchor),
        Expr::Reduce(w) => w.reduce.iter().any(expr_has_anchor),
        Expr::UncheckedDim(w) => expr_has_anchor(&w.unchecked_dim.inner),
        Expr::Const(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_) | Expr::Time(_)
        | Expr::Dt(_) | Expr::TimeFunc(_) | Expr::Projected(_) | Expr::ObsColumnRef(_)
        | Expr::BindingRef(_) | Expr::PerEvalRef(_) => false,
    }
}

/// Every expression a forcing kind carries. Exhaustive (no `_` arm) so a new
/// forcing kind must declare whether its expressions are walked.
fn forcing_exprs(k: &ir::time_func::TimeFuncKind) -> Vec<&Expr> {
    use ir::time_func::TimeFuncKind as K;
    match k {
        K::Sinusoidal(s) => vec![&s.amplitude, &s.period, &s.phase, &s.baseline],
        K::Piecewise(p) => p.breakpoints.iter().chain(&p.values).collect(),
        K::Interpolated(i) => i.times.iter().chain(&i.values).collect(),
        K::Periodic(p) => std::iter::once(&p.period).chain(&p.values).collect(),
        K::Fourier(f) => std::iter::once(&f.period)
            .chain(f.harmonics.iter().flat_map(|(a, b)| [a, b]))
            .collect(),
        K::PeriodicSpline(s) => std::iter::once(&s.period).chain(&s.coefs).collect(),
    }
}

fn forcing_exprs_mut(k: &mut ir::time_func::TimeFuncKind) -> Vec<&mut Expr> {
    use ir::time_func::TimeFuncKind as K;
    match k {
        K::Sinusoidal(s) => vec![&mut s.amplitude, &mut s.period, &mut s.phase, &mut s.baseline],
        K::Piecewise(p) => p.breakpoints.iter_mut().chain(&mut p.values).collect(),
        K::Interpolated(i) => i.times.iter_mut().chain(&mut i.values).collect(),
        K::Periodic(p) => std::iter::once(&mut p.period).chain(&mut p.values).collect(),
        K::Fourier(f) => std::iter::once(&mut f.period)
            .chain(f.harmonics.iter_mut().flat_map(|(a, b)| [a, b]))
            .collect(),
        K::PeriodicSpline(s) => std::iter::once(&mut s.period).chain(&mut s.coefs).collect(),
    }
}

/// The stderr line every resolving command prints, so a reader of the log can
/// see which number a run actually used — the same posture `--to` already has.
/// The calendar date is appended when the model is anchored to an origin,
/// because "t = 91" is not something a modeller can check against a data file.
pub fn report(resolved: &[ResolvedAnchor], model: &ir::Model) {
    for r in resolved {
        let cal = model
            .origin
            .as_deref()
            .and_then(|o| ir::caltime::internal_to_date(o, r.resolved, &model.time_unit).ok())
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        eprintln!("{}: {} → t = {}{}", r.site, r.anchor, r.resolved, cal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::anchor::ObsAnchor;
    use ir::time_func::{Piecewise, TimeFuncKind, TimeFunction};

    fn model() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/sir_basic.ir.json");
        let contents = std::fs::read_to_string(&path).expect("read sir_basic");
        ir::from_str(&contents).expect("parse sir_basic")
    }

    fn ramp(breakpoints: Vec<Expr>, n_values: usize) -> TimeFunction {
        TimeFunction {
            name: "ramp".into(),
            kind: TimeFuncKind::Piecewise(Piecewise {
                breakpoints,
                values: (0..n_values).map(|i| Expr::const_(i as f64)).collect(),
            }),
            dim: (0, 0),
            lag: None,
        }
    }

    const W: ObsAnchorTimes = ObsAnchorTimes { first: 7.0, last: 28.0 };

    /// An unanchored model comes out BYTE-IDENTICAL, so a caller may run the
    /// resolver unconditionally without re-keying anything.
    #[test]
    fn an_unanchored_model_is_untouched() {
        let before = model();
        let mut after = model();
        let moved = substitute(&mut after, W).expect("clean model resolves");
        assert!(moved.is_empty(), "nothing to report");
        assert_eq!(before, after, "an unanchored model must not change");
        assert!(!model_is_anchored(&before));
    }

    /// The horizon lands in `simulation.t_end` and the anchor field is CLEARED —
    /// the field is the unresolved marker, so leaving it set would keep
    /// `CompiledModel::new` refusing a model that is in fact resolved.
    #[test]
    fn the_horizon_resolves_and_the_marker_clears() {
        let mut m = model();
        m.simulation.t_end = f64::NAN;
        m.simulation.t_end_anchor =
            Some(AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 });
        let moved = substitute(&mut m, W).expect("resolves");
        assert_eq!(m.simulation.t_end, 56.0, "last_obs(28) + 28");
        assert_eq!(m.simulation.t_end_anchor, None, "the marker must clear");
        assert_eq!(moved.len(), 1);
        assert_eq!(moved[0].resolved, 56.0);
        assert!(!model_is_anchored(&m));
    }

    #[test]
    fn a_preset_horizon_resolves() {
        let mut m = model();
        m.presets[0].t_end = Some(f64::NAN);
        m.presets[0].t_end_anchor =
            Some(AnchoredTime { anchor: ObsAnchor::First, offset: 14.0 });
        substitute(&mut m, W).expect("resolves");
        assert_eq!(m.presets[0].t_end, Some(21.0), "first_obs(7) + 14");
        assert_eq!(m.presets[0].t_end_anchor, None);
    }

    /// Each knot resolves independently, so a mixed literal/anchored fork keeps
    /// its literal knots exactly and moves only the anchored ones.
    #[test]
    fn forcing_knots_resolve_independently() {
        let mut m = model();
        m.time_functions.push(ramp(
            vec![
                Expr::const_(10.0),
                Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last)),
                Expr::obs_anchor(AnchoredTime { anchor: ObsAnchor::Last, offset: 14.0 }),
            ],
            4,
        ));
        let moved = substitute(&mut m, W).expect("resolves");
        let TimeFuncKind::Piecewise(p) = &m.time_functions.last().unwrap().kind else {
            panic!("piecewise")
        };
        let knots: Vec<f64> = p.breakpoints.iter().filter_map(const_value).collect();
        assert_eq!(knots, vec![10.0, 28.0, 42.0]);
        assert_eq!(moved.len(), 2, "only the anchored knots are reported");
    }

    /// The ordering check exists because an anchored fork can invert under one
    /// data vintage and not another — and `piecewise_value` scans in order, so
    /// an inverted knot list silently returns the wrong step.
    #[test]
    fn non_monotone_resolved_knots_are_refused() {
        let mut m = model();
        // Literal 60 sits AFTER last_obs(28) + 0 in the list but before it in
        // time — ordered for a later data vintage, inverted for this one.
        m.time_functions.push(ramp(
            vec![
                Expr::const_(60.0),
                Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last)),
            ],
            3,
        ));
        let err = substitute(&mut m, W).expect_err("inverted knots must be refused");
        assert!(err.contains("non-monotone") && err.contains("ramp"), "{err}");

        // Negative control: the SAME model under a later data vintage, where the
        // knots come out ordered, must resolve cleanly. Without this the test
        // would pass for a resolver that refused every anchored forcing.
        let mut m2 = model();
        m2.time_functions.push(ramp(
            vec![
                Expr::const_(60.0),
                Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last)),
            ],
            3,
        ));
        substitute(&mut m2, ObsAnchorTimes { first: 7.0, last: 90.0 })
            .expect("ordered knots resolve");
    }

    #[test]
    fn wrong_value_count_is_refused() {
        let mut m = model();
        m.time_functions.push(ramp(
            vec![Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last))],
            3, // 1 knot needs 2 values
        ));
        let err = substitute(&mut m, W).expect_err("arity must be refused");
        assert!(err.contains("one more value than knots"), "{err}");
    }

    #[test]
    fn a_knot_at_or_before_t_start_is_refused() {
        let mut m = model();
        m.simulation.t_start = 30.0;
        m.time_functions
            .push(ramp(vec![Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last))], 2));
        let err = substitute(&mut m, W).expect_err("a knot before t_start must be refused");
        assert!(err.contains("t_start"), "{err}");
    }

    /// A horizon that resolves backwards is refused HERE, with a message about
    /// the anchor, rather than falling through to `ir::validate`'s generic one.
    #[test]
    fn a_backwards_resolved_horizon_is_refused() {
        let mut m = model();
        m.simulation.t_start = 0.0;
        m.simulation.t_end = f64::NAN;
        m.simulation.t_end_anchor =
            Some(AnchoredTime { anchor: ObsAnchor::First, offset: -14.0 });
        let err = substitute(&mut m, W).expect_err("a backwards horizon must be refused");
        assert!(err.contains("forward interval") && err.contains("offset's sign"), "{err}");
    }

    #[test]
    fn anchored_sites_names_every_construct() {
        let mut m = model();
        m.simulation.t_end_anchor = Some(AnchoredTime::bare(ObsAnchor::Last));
        m.presets[0].t_end_anchor = Some(AnchoredTime::bare(ObsAnchor::First));
        m.time_functions
            .push(ramp(vec![Expr::obs_anchor(AnchoredTime::bare(ObsAnchor::Last))], 2));
        assert!(model_is_anchored(&m));
        let sites = anchored_sites(&m);
        assert_eq!(sites.len(), 3, "{sites:?}");
        assert!(sites.iter().any(|s| s.contains("simulate")), "{sites:?}");
        assert!(sites.iter().any(|s| s.contains("scenario")), "{sites:?}");
        assert!(sites.iter().any(|s| s.contains("ramp")), "{sites:?}");
    }
}
