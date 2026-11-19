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

use ir::anchor::ObsAnchorTimes;
use ir::expr::Expr;

/// One resolved anchor, for the stderr line and the run record.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAnchor {
    /// What was anchored, in the model author's vocabulary:
    /// `"simulate { to }"`, `"scenario 'forecast' { to }"`, `"forcing 'ramp'"`.
    pub site: String,
    /// The symbolic form, as written (`last_obs + 28`, or a `--from` spec).
    pub anchor: String,
    /// The model time it resolved to.
    pub resolved: f64,
}

/// Does this model carry any unresolved observation anchor? The cue for a
/// caller to bind observation data before compiling — and, when it cannot, to
/// refuse by name rather than let `CompiledModel::new` do it with less context.
pub fn model_is_anchored(model: &ir::Model) -> bool {
    model.simulation.t_start_anchor.is_some()
        || model.simulation.t_end_anchor.is_some()
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
///
/// `holdout_after` is the fit config's cutoff when the command fits that
/// config's training rows (`profile --fit`): the anchors are then the
/// observed window of the training rows (D40). `None` reads every row.
///
/// `start` is the run's start override (a fit config's `[simulate] from`),
/// applied before the anchors through [`apply_start`].
pub fn resolve_from_bindings(
    out: &crate::out::Out, model: &mut ir::Model,
    bound: &[(String, std::path::PathBuf)],
    dt: f64,
    holdout_after: Option<&crate::fit::config_v2::TimeSpecToml>,
    start: Option<&StartOverride>,
) -> Result<bool, String> {
    resolve_with(out, model, start,
        |m| crate::obs_anchors_from_bindings(m, bound, dt, holdout_after))
}

/// As [`resolve_from_bindings`], for a command holding a fit config rather than
/// a list of bindings (`survey --fit`). The two differ ONLY in where the
/// observed window is read from; the substitution, the stderr report and the
/// refusal wording are shared, so the two commands cannot drift on what a
/// resolved anchor means or on what an unresolvable one says.
pub fn resolve_from_config(
    out: &crate::out::Out, model: &mut ir::Model,
    config: &crate::fit::config_v2::Problem,
    dt: f64,
) -> Result<bool, String> {
    let start = config_start(config, model)?;
    resolve_with(out, model, start.as_ref(), |m| crate::obs_anchors_from_config(m, config, dt))
}

/// The observed window a FIT anchors `model` to: the fold over the fit
/// config's `[data.observations]`, loaded at the step the fit integrates at
/// (the anchor reader refuses times that are not multiples of it), by the one
/// dt rule.
///
/// `fit run` resolves the model it fits — and archives — through this, and
/// every later command that has to reproduce that resolution for a stored fit
/// (`fit predict`, the replay model check) calls it with the fit's config, so
/// the window is computed one way rather than re-derived at each site. It reads
/// the data as it is NOW: a stored fit records the substituted model, not the
/// window, so data changed since the fit yields a different window.
pub fn fit_window(
    model: &ir::Model,
    config: &crate::fit::config_v2::Problem,
) -> Result<ObsAnchorTimes, String> {
    let dt0 = crate::dt_rule::resolved_dt(model.simulation.dt, config.config.dt);
    let (first, last) = crate::obs_anchors_from_config(model, config, dt0)?;
    Ok(ObsAnchorTimes { first, last })
}

fn resolve_with(
    out: &crate::out::Out, model: &mut ir::Model,
    start: Option<&StartOverride>,
    read_window: impl FnOnce(&ir::Model) -> Result<(f64, f64), String>,
) -> Result<bool, String> {
    if !needs_window(model, start) {
        let Some(start) = start else { return Ok(false) };
        let r = apply_start(model, start, None)?;
        report(out, &[r], model);
        return Ok(true);
    }
    let what = match start {
        Some(s) if s.is_anchored() && !model_is_anchored(model) =>
            format!("{} \"{}\" is anchored to observed data", s.source, s.raw),
        _ => format!("this model is anchored to observed data ({})", anchored_sites(model).join(", ")),
    };
    let (first, last) = read_window(&window_model(model, start)).map_err(|e| {
        format!("{what}, and the bound observation data cannot resolve it: {e}")
    })?;
    let window = ObsAnchorTimes { first, last };
    let mut moved = Vec::new();
    if let Some(start) = start {
        moved.push(apply_start(model, start, Some(window))?);
    }
    moved.extend(substitute(model, window)?);
    report(out, &moved, model);
    Ok(true)
}

/// The refusal an ANCHORED model earns from a data-binding command that was
/// given nothing to bind. `None` for an unanchored model, whose command is then
/// free to report its own missing-data error wherever it always did.
///
/// The distinction is not cosmetic. A command that binds no data usually still
/// has something better to say first — `pfilter`'s capability gates report
/// "this model has real compartments the filter cannot advance" (gh#191) rather
/// than "--data is required", and that ordering is pinned by test. But an
/// anchored model has no horizon until a stream resolves one, so it cannot
/// reach those gates at all: `CompiledModel::new` refuses it. Hence a refusal
/// here, naming the constructs that need the data and what to supply.
pub fn refuse_without_data(model: &ir::Model, supply: &str) -> Option<String> {
    if !model_is_anchored(model) {
        return None;
    }
    Some(format!(
        "this model is anchored to observed data ({}), but no observation \
         stream is bound. An observation anchor resolves only from bound \
         data, so the model has no horizon to run.\n  Fix: {supply}",
        anchored_sites(model).join(", ")
    ))
}

/// Human-readable list of the anchored constructs, for a refusal message.
pub fn anchored_sites(model: &ir::Model) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(a) = &model.simulation.t_start_anchor {
        out.push(format!("`simulate {{ from = {a} }}`"));
    }
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

    // The start first: `set_start` writes it into every start the compiler
    // derived from it (each baked NaN while the anchor was unresolved).
    if let Some(a) = model.simulation.t_start_anchor {
        let t = anchors.at(a);
        set_start(model, t, "simulate { from }")?;
        resolved.push(ResolvedAnchor { site: "simulate { from }".into(), anchor: a.to_string(), resolved: t });
    }
    if let Some(a) = model.simulation.t_end_anchor.take() {
        let t = anchors.at(a);
        model.simulation.t_end = t;
        resolved.push(ResolvedAnchor { site: "simulate { to }".into(), anchor: a.to_string(), resolved: t });
    }
    for p in &mut model.presets {
        if let Some(a) = p.t_end_anchor.take() {
            let t = anchors.at(a);
            p.t_end = Some(t);
            resolved.push(ResolvedAnchor {
                site: format!("scenario '{}' {{ to }}", p.name),
                anchor: a.to_string(),
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
            out.push(ResolvedAnchor { site: site.to_string(), anchor: a.to_string(), resolved: t });
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
        Expr::Const(_) | Expr::UnitLiteral(_) | Expr::ConstRef(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_)
        | Expr::Time(_) | Expr::Dt(_) | Expr::TimeFunc(_) | Expr::ForcingBasis(_) | Expr::Projected(_)
        | Expr::ObsColumnRef(_) | Expr::BindingRef(_) | Expr::PerEvalRef(_) => {}
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

    // A `bspline` window or breakpoint list read from anchors can invert under
    // one data vintage and not another, exactly as a piecewise fork can. The
    // runtime refuses it too (`bspline::Knots`), but here the message can say
    // that the anchors are the cause.
    for tf in &model.time_functions {
        let ir::time_func::TimeFuncKind::BSpline(b) = &tf.kind else { continue };
        let exprs = bspline_knot_exprs(&b.knots);
        let knots: Vec<f64> = exprs.iter().filter_map(|e| const_value(e)).collect();
        if knots.len() != exprs.len() {
            continue;
        }
        if knots.windows(2).any(|w| w[0].is_nan() || w[1].is_nan() || w[1] <= w[0]) {
            return Err(format!(
                "forcing '{}' has its window or breakpoints out of order after \
                 resolving its observation anchors: {:?}. A bspline needs strictly \
                 increasing knots (window start before window end). Check the \
                 anchor offsets against the literal knots.",
                tf.name, knots
            ));
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
            // NaN arm explicit: `w[1] < w[0]` alone is false for NaN.
            if w[0].is_nan() || w[1].is_nan() || w[1] < w[0] {
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

/// The value of a knot that is constant once its anchors are substituted:
/// a literal, a unit literal, or `+ − × ÷` and negation over those
/// (`60 'days + 2 'weeks`). Anything else — a parameter — is `None`.
fn const_value(e: &Expr) -> Option<f64> {
    use ir::expr::{BinOp, UnOp};
    match e {
        Expr::Const(c) => Some(c.value),
        Expr::UnitLiteral(w) => Some(w.unit_literal.value),
        // A user's dimensional escape around a constant is that constant.
        Expr::UncheckedDim(w) => const_value(&w.unchecked_dim.inner),
        Expr::UnOp(w) if w.un_op.op == UnOp::Neg => const_value(&w.un_op.arg).map(|v| -v),
        Expr::BinOp(w) => {
            let (l, r) = (const_value(&w.bin_op.left)?, const_value(&w.bin_op.right)?);
            match w.bin_op.op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div => Some(l / r),
                _ => None,
            }
        }
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
        Expr::Const(_) | Expr::UnitLiteral(_) | Expr::ConstRef(_) | Expr::Param(_) | Expr::Pop(_) | Expr::PopSum(_)
        | Expr::Time(_) | Expr::Dt(_) | Expr::TimeFunc(_) | Expr::ForcingBasis(_) | Expr::Projected(_)
        | Expr::ObsColumnRef(_) | Expr::BindingRef(_) | Expr::PerEvalRef(_) => false,
    }
}

/// Every expression a forcing kind carries. Exhaustive (no `_` arm) so a new
/// forcing kind must declare whether its expressions are walked.
fn forcing_exprs(k: &ir::time_func::TimeFuncKind) -> Vec<&Expr> {
    use ir::time_func::TimeFuncKind as K;
    match k {
        K::Sinusoidal(s) => vec![&s.amplitude, &s.period, &s.phase, &s.baseline],
        K::Piecewise(p) => p.breakpoints.iter().chain(&p.values).collect(),
        K::Interpolated(i) => match &i.source {
            ir::time_func::InterpSource::Knots { times, values } => times.iter().chain(values).collect(),
            ir::time_func::InterpSource::External(_) => Vec::new(),
        },
        K::Periodic(p) => std::iter::once(&p.period).chain(&p.values).collect(),
        K::Fourier(f) => std::iter::once(&f.period)
            .chain(f.harmonics.iter().flat_map(|(a, b)| [a, b]))
            .collect(),
        K::PeriodicSpline(s) => std::iter::once(&s.period).chain(&s.coefs).collect(),
        K::BSpline(b) => bspline_knot_exprs(&b.knots).into_iter().chain(b.coefs.exprs()).collect(),
    }
}

/// A `bspline` forcing's knot expressions: the window ends, or the breakpoints.
fn bspline_knot_exprs(k: &ir::time_func::BSplineKnots) -> Vec<&Expr> {
    match k {
        ir::time_func::BSplineKnots::Uniform { lower, upper, .. } => vec![lower, upper],
        ir::time_func::BSplineKnots::Explicit { breakpoints } => breakpoints.iter().collect(),
    }
}

fn forcing_exprs_mut(k: &mut ir::time_func::TimeFuncKind) -> Vec<&mut Expr> {
    use ir::time_func::TimeFuncKind as K;
    match k {
        K::Sinusoidal(s) => vec![&mut s.amplitude, &mut s.period, &mut s.phase, &mut s.baseline],
        K::Piecewise(p) => p.breakpoints.iter_mut().chain(&mut p.values).collect(),
        K::Interpolated(i) => match &mut i.source {
            ir::time_func::InterpSource::Knots { times, values } =>
                times.iter_mut().chain(values.iter_mut()).collect(),
            ir::time_func::InterpSource::External(_) => Vec::new(),
        },
        K::Periodic(p) => std::iter::once(&mut p.period).chain(&mut p.values).collect(),
        K::Fourier(f) => std::iter::once(&mut f.period)
            .chain(f.harmonics.iter_mut().flat_map(|(a, b)| [a, b]))
            .collect(),
        K::PeriodicSpline(s) => std::iter::once(&mut s.period).chain(&mut s.coefs).collect(),
        K::BSpline(b) => {
            let knots: Vec<&mut Expr> = match &mut b.knots {
                ir::time_func::BSplineKnots::Uniform { lower, upper, .. } => vec![lower, upper],
                ir::time_func::BSplineKnots::Explicit { breakpoints } => breakpoints.iter_mut().collect(),
            };
            knots.into_iter().chain(b.coefs.exprs_mut()).collect()
        }
    }
}

// ── The run's start: `simulate { from }`, chosen per run ──────────────────────

/// A start given to a run from outside the model file. Precedence, highest
/// first: `camdl simulate --from`, then a fit config's `[simulate] from`, then
/// the model's own `simulate { from }` (which is not an override and needs no
/// value here).
///
/// Applying one turns the model into the model that would have compiled with
/// that start written as a literal ([`set_start`]), so the resolved start
/// enters the run's identity through the model, as an anchored `to` does.
#[derive(Debug, Clone, PartialEq)]
pub struct StartOverride {
    /// The surface that set it, for messages: `"--from"`, `"[simulate] from"`.
    pub source: &'static str,
    /// The spec as written.
    pub raw: String,
    /// Parsed by the shared time grammar of `--to`.
    pub spec: crate::fit::runner::TimeSpec,
}

impl StartOverride {
    pub fn parse(source: &'static str, raw: &str, model: &ir::Model) -> Result<Self, String> {
        let spec = crate::fit::runner::parse_time_spec(
            source, raw, model.origin.as_deref(), &model.time_unit)?;
        Ok(StartOverride { source, raw: raw.trim().to_string(), spec })
    }

    pub fn is_anchored(&self) -> bool {
        matches!(self.spec, crate::fit::runner::TimeSpec::Anchored(_))
    }
}

/// The start a fit config asks for: its `[simulate] from`, if it has one.
pub fn config_start(
    problem: &crate::fit::config_v2::Problem,
    model: &ir::Model,
) -> Result<Option<StartOverride>, String> {
    problem.simulate.as_ref().and_then(|s| s.from.as_ref())
        .map(|f| StartOverride::parse("[simulate] from", &f.raw(), model))
        .transpose()
}

/// Does resolving this run read the observed window: a model anchor, or an
/// anchored start override?
///
/// A start override replaces the model's start, so a model whose only anchor
/// is its `from` needs no data under a literal override.
pub fn needs_window(model: &ir::Model, start: Option<&StartOverride>) -> bool {
    match start {
        None => model_is_anchored(model),
        Some(s) => s.is_anchored()
            || model.simulation.t_end_anchor.is_some()
            || model_is_anchored_beyond_simulation(model),
    }
}

/// An anchor outside `simulate {}`: a scenario's horizon or a forcing knot.
fn model_is_anchored_beyond_simulation(model: &ir::Model) -> bool {
    model.presets.iter().any(|p| p.t_end_anchor.is_some())
        || model.time_functions.iter().any(|tf| {
            forcing_exprs(&tf.kind).into_iter().any(expr_has_anchor)
        })
}

/// The model the observed window is folded from. The observation reader
/// checks rows against `t_start`; under an anchored start override the model's
/// own start is about to be replaced by a value read off those very rows, so
/// the fold reads them as it does for a model whose `from` is anchored, with
/// no start yet (NaN disables the start-relative checks; the run's own load
/// applies them against the resolved start).
pub(crate) fn window_model<'a>(model: &'a ir::Model, start: Option<&StartOverride>) -> std::borrow::Cow<'a, ir::Model> {
    match start {
        Some(s) if s.is_anchored() => {
            let mut m = model.clone();
            m.simulation.t_start = f64::NAN;
            std::borrow::Cow::Owned(m)
        }
        _ => std::borrow::Cow::Borrowed(model),
    }
}

/// Apply `start` to `model`: resolve it (an anchored spec against `window`,
/// which the caller must then supply) and move the model's start there,
/// clearing any start anchor the model carried. Refuses a start at or after a
/// resolved horizon.
pub fn apply_start(
    model: &mut ir::Model,
    start: &StartOverride,
    window: Option<ObsAnchorTimes>,
) -> Result<ResolvedAnchor, String> {
    let t = match start.spec {
        crate::fit::runner::TimeSpec::Absolute(v) => v,
        crate::fit::runner::TimeSpec::Anchored(a) => match window {
            Some(w) => w.at(a),
            None => return Err(format!(
                "{} \"{}\" is anchored to observed data, but no observation \
                 stream is bound to resolve it against.\n  Fix: bind data (`--fit \
                 <fit.toml | fit run dir>` on `simulate`, `--data`/`--fit` \
                 elsewhere), or give the start as a model time or a date.",
                start.source, start.raw
            )),
        },
    };
    if !t.is_finite() {
        return Err(format!("{} \"{}\" resolves to t = {t}, which is not a time.",
                           start.source, start.raw));
    }
    let site = format!("simulate {{ from }} ({})", start.source);
    set_start(model, t, &site)?;
    let t_end = model.simulation.t_end;
    if t_end.is_finite() && t >= t_end {
        return Err(format!(
            "{} \"{}\" resolves to t = {t}, at or after the run's horizon \
             t_end = {t_end}. The start must lie before the end.",
            start.source, start.raw
        ));
    }
    Ok(ResolvedAnchor { site, anchor: start.raw.clone(), resolved: t })
}

/// Move `model`'s start to `s`, together with every start the compiler
/// derived from it, so the result is the model that would have compiled with
/// `from = s` written as a literal.
///
/// The derived starts are the regular output schedule's, each regular
/// `emit_schedule`'s, and a recurring intervention's or event's when it
/// declares no `from` (the `every … at_day …` form never declares one). How a
/// derived start is recognised depends on what the model carries:
///
/// - **An unresolved anchored start** (`t_start_anchor`): every derived start
///   is NaN and every declared one a number, so the NaN ones move. No
///   ambiguity.
/// - **A literal start**: a derived start equals the model's start. A
///   recurring schedule *without* `at_day` whose start equals it may instead
///   have declared that same `from`; the two cannot be told apart, so moving
///   the start is refused, naming the schedules, rather than guessed. The
///   output and emit schedules have no `from` of their own and always move.
///
/// A literal start already equal to `s` is left byte-identical.
pub fn set_start(model: &mut ir::Model, s: f64, site: &str) -> Result<(), String> {
    use ir::intervention::{FireSource, InterventionSchedule};
    let unresolved = model.simulation.t_start_anchor.is_some();
    let old = model.simulation.t_start;
    if !unresolved && old.to_bits() == s.to_bits() {
        return Ok(());
    }
    if !unresolved {
        let ambiguous: Vec<&str> = model.interventions.iter()
            .filter(|iv| matches!(&iv.fire,
                FireSource::Scheduled(InterventionSchedule::Recurring(r))
                    if r.at_day.is_none() && r.start.to_bits() == old.to_bits()))
            .map(|iv| iv.name.as_str())
            .collect();
        if !ambiguous.is_empty() {
            return Err(format!(
                "{site} moves the start to t = {s}, but recurring schedule(s) [{}] start \
                 at the model's own start, t = {old}. camdl cannot tell whether they \
                 declare no `from` (and should move to t = {s}) or declare `from = {old}` \
                 (and should stay), so it will not guess.\n  Fix: write the model's start \
                 as an observation anchor (`simulate {{ from = first_obs … }}`), under which \
                 a schedule with no `from` carries no number until the run resolves it; or \
                 run at the model's own start.",
                ambiguous.join(", ")
            ));
        }
    }
    let derived = |start: f64| if unresolved { start.is_nan() } else { start.to_bits() == old.to_bits() };
    model.simulation.t_start_anchor = None;
    model.simulation.t_start = s;
    if let ir::model::OutputSchedule::Regular(r) = &mut model.output.times {
        if derived(r.start) {
            r.start = s;
        }
    }
    for o in &mut model.observations {
        if let Some(ir::observation::ObservationSchedule::Regular(r)) = &mut o.emit_schedule {
            if derived(r.start) {
                r.start = s;
            }
        }
    }
    for iv in &mut model.interventions {
        if let FireSource::Scheduled(InterventionSchedule::Recurring(r)) = &mut iv.fire {
            if derived(r.start) {
                r.start = s;
            }
        }
    }
    Ok(())
}

/// Resolve `model` as a fit of `problem` runs it: the config's
/// `[simulate] from`, then every observation anchor, against the fit's
/// observed window ([`fit_window`]). The one resolution every fit-config path
/// takes (`fit run`, its per-cell loads, `fit predict`, the replay check), so
/// they cannot disagree on where a fit starts. A model with neither returns
/// unchanged with an empty list.
pub fn resolve_for_fit(
    model: &mut ir::Model,
    problem: &crate::fit::config_v2::Problem,
) -> Result<Vec<ResolvedAnchor>, String> {
    let start = config_start(problem, model)?;
    let mut moved = Vec::new();
    let window = if needs_window(model, start.as_ref()) {
        Some(fit_window(&window_model(model, start.as_ref()), problem)?)
    } else {
        None
    };
    if let Some(start) = &start {
        moved.push(apply_start(model, start, window)?);
    }
    if let Some(w) = window {
        moved.extend(substitute(model, w)?);
    }
    Ok(moved)
}

/// Does `problem` resolve anything in `model` — an anchor, or a start?
pub fn fit_resolves(model: &ir::Model, problem: &crate::fit::config_v2::Problem) -> bool {
    model_is_anchored(model)
        || problem.simulate.as_ref().is_some_and(|s| s.from.is_some())
}

/// The stderr line every resolving command prints, so a reader of the log can
/// see which number a run actually used — the same posture `--to` already has.
/// The calendar date is appended when the model is anchored to an origin,
/// because "t = 91" is not something a modeller can check against a data file.
pub fn report(out: &crate::out::Out, resolved: &[ResolvedAnchor], model: &ir::Model) {
    for r in resolved {
        let cal = model
            .origin
            .as_deref()
            .and_then(|o| ir::caltime::internal_to_date(o, r.resolved, &model.time_unit).ok())
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        out.say(format_args!("{}: {} → t = {}{}", r.site, r.anchor, r.resolved, cal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::anchor::{AnchoredTime, ObsAnchor};
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
            data_source: None,
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

    // ── The run's start ──────────────────────────────────────────────────────

    fn recurring(name: &str, start: f64, at_day: Option<f64>) -> ir::intervention::Intervention {
        ir::intervention::Intervention {
            name: name.into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(
                ir::intervention::InterventionSchedule::Recurring(
                    ir::intervention::RecurringSchedule { start, period: 30.0, end: 70.0, at_day },
                ),
            ),
            actions: vec![],
            kind: ir::intervention::InterventionKind::Scenario,
        }
    }

    fn recurring_start(m: &ir::Model, name: &str) -> f64 {
        m.interventions.iter().find(|iv| iv.name == name).and_then(|iv| match &iv.fire {
            ir::intervention::FireSource::Scheduled(
                ir::intervention::InterventionSchedule::Recurring(r)) => Some(r.start),
            _ => None,
        }).expect("recurring intervention")
    }

    fn output_start(m: &ir::Model) -> f64 {
        match &m.output.times {
            ir::model::OutputSchedule::Regular(r) => r.start,
            _ => panic!("fixture has a regular output schedule"),
        }
    }

    /// A model compiled with an anchored start (every derived start NaN), once
    /// resolved, equals the same model compiled with that start as a literal:
    /// the anchored form's NaN starts move, its declared ones stay.
    #[test]
    fn a_resolved_start_anchor_equals_the_literal_start() {
        let mut literal = model();
        literal.simulation.t_start = 7.0;
        if let ir::model::OutputSchedule::Regular(r) = &mut literal.output.times { r.start = 7.0; }
        literal.interventions.push(recurring("defaulted", 7.0, None));
        literal.interventions.push(recurring("declared", 3.0, None));

        let mut anchored = model();
        anchored.simulation.t_start = f64::NAN;
        anchored.simulation.t_start_anchor = Some(AnchoredTime::bare(ObsAnchor::First));
        if let ir::model::OutputSchedule::Regular(r) = &mut anchored.output.times { r.start = f64::NAN; }
        anchored.interventions.push(recurring("defaulted", f64::NAN, None));
        anchored.interventions.push(recurring("declared", 3.0, None));
        let moved = substitute(&mut anchored, W).expect("resolves");
        assert!(moved.iter().any(|r| r.site == "simulate { from }" && r.resolved == 7.0), "{moved:?}");
        assert_eq!(anchored, literal);
        assert_eq!(runid::inputs::model_ir_hash(&anchored), runid::inputs::model_ir_hash(&literal));
    }

    /// A literal start moved by an override moves the output schedule and the
    /// `at_day` form (neither can declare a `from`) and refuses a recurring
    /// schedule whose start equals the model's, which might be either.
    #[test]
    fn moving_a_literal_start_moves_derived_starts_and_refuses_the_ambiguous() {
        let mut m = model();
        m.simulation.t_start = 0.0;
        m.interventions.push(recurring("yearly", 0.0, Some(5.0)));
        m.interventions.push(recurring("later", 10.0, None));
        set_start(&mut m, 7.0, "--from").expect("no ambiguous schedule");
        assert_eq!(m.simulation.t_start, 7.0);
        assert_eq!(output_start(&m), 7.0);
        assert_eq!(recurring_start(&m, "yearly"), 7.0);
        assert_eq!(recurring_start(&m, "later"), 10.0, "a declared `from` stays");

        let mut m = model();
        m.simulation.t_start = 0.0;
        m.interventions.push(recurring("campaign", 0.0, None));
        let before = m.clone();
        let err = set_start(&mut m, 7.0, "--from").expect_err("ambiguous");
        assert!(err.contains("campaign") && err.contains("first_obs"), "{err}");
        assert_eq!(m, before, "a refused move leaves the model untouched");
        // The same start is not a move, so nothing is ambiguous.
        set_start(&mut m, 0.0, "--from").expect("no-op");
        assert_eq!(m, before);
    }

    #[test]
    fn a_start_override_resolves_against_the_window_and_respects_the_horizon() {
        let mut m = model();
        let ov = StartOverride::parse("--from", "first_obs - 1 week", &m).unwrap();
        assert!(ov.is_anchored());
        let err = apply_start(&mut m.clone(), &ov, None).expect_err("anchored needs a window");
        assert!(err.contains("--from") && err.contains("first_obs - 1 week"), "{err}");
        let r = apply_start(&mut m, &ov, Some(W)).unwrap();
        assert_eq!(r.resolved, 0.0);
        assert_eq!(m.simulation.t_start, 0.0);

        let mut m = model();
        let t_end = m.simulation.t_end;
        let ov = StartOverride::parse("[simulate] from", &format!("{t_end}"), &m).unwrap();
        let err = apply_start(&mut m, &ov, None).expect_err("start at the horizon");
        assert!(err.contains("horizon"), "{err}");
    }
}
