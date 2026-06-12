use crate::{
    compiled_model::CompiledModel,
    error::SimError,
    propensity::EvalCtx,
    resolved_expr::{eval_resolved, ResolvedExpr},
    state::{IntState, RealState},
};
use crate::schedule::StepPolicy;
use ir::intervention::{Action, InterventionSchedule};

/// Short human label for an action, for diagnostics (`"set V"`,
/// `"transfer S -> I (fraction)"`). Error-path only.
fn action_label(action: &Action) -> String {
    match action {
        Action::Add(a) => format!("add {}", a.compartment),
        Action::Set(a) => format!("set {}", a.compartment),
        Action::FractionTransfer(t) => format!("transfer {} -> {} (fraction)", t.src, t.dst),
        Action::AbsoluteTransfer(t) => format!("transfer {} -> {} (absolute)", t.src, t.dst),
    }
}

/// Validate that an intervention/event action's resolved value is finite
/// before it is cast to a count. A non-finite value otherwise casts
/// silently and wrongly — `NaN as i64 == 0`, `inf as i64 == i64::MAX`,
/// `-inf as i64 == i64::MIN` — corrupting the trajectory with no error.
/// The finite guard on the intervention *time* has no analogue for the
/// resolved *value*; this is it. Hard error (not per-particle-recoverable):
/// a non-finite effect amount is a structural/config defect in the action
/// expression, not a stochastic exploration artifact, so it surfaces
/// regardless of caller (forward-sim or inference).
pub(crate) fn finite_action_value(
    value: f64,
    iv_name: &str,
    action: &Action,
    t: f64,
) -> Result<f64, SimError> {
    if !value.is_finite() {
        return Err(SimError::Validation(format!(
            "intervention '{iv_name}' action ({}) resolved to a non-finite \
             value ({value}) at t={t:.3}; a non-finite effect amount would \
             cast silently to a wrong count (NaN→0, +inf→i64::MAX, \
             -inf→i64::MIN) — check the action expression",
            action_label(action)
        )));
    }
    Ok(value)
}

/// Convert an `InterventionSchedule` to a sorted list of fire times.
///
/// For parametric `at [...]` lists (gh#69, `AtTimesExpr`) the caller
/// supplies pre-resolved `ResolvedExpr`s for the entries — evaluated
/// here against the current `params` vector with the rest of `EvalCtx`
/// (state, time, dt) filled by scratch. Schedule-time expressions are
/// constrained at compile time to reference only parameters and
/// constants (see `CompiledModel::new` validation, gh#69), so the
/// scratch values are never consulted.
pub fn intervention_fire_times(
    sched: &InterventionSchedule,
    resolved_at_times: Option<&[ResolvedExpr]>,
    model: &CompiledModel,
    params: &[f64],
) -> Vec<f64> {
    match sched {
        InterventionSchedule::AtTimes(times) => times.clone(),
        InterventionSchedule::AtTimesExpr(_) => {
            let resolved = resolved_at_times
                .expect("AtTimesExpr schedule must be accompanied by resolved exprs");
            let n_int = model.int_local_to_global.len();
            let n_real = model.real_local_to_global.len();
            let scratch_int = IntState::new(n_int);
            let scratch_real = RealState::new(n_real);
            let ctx = EvalCtx {
                model,
                int_s: &scratch_int,
                real_s: &scratch_real,
                params,
                t: 0.0,
                dt: 0.0,
                projected: None,
                int_float_override: None,
            };
            resolved.iter().map(|e| eval_resolved(e, &ctx)).collect()
        }
        InterventionSchedule::Recurring(rs) => {
            let mut times = Vec::new();
            if let Some(at_day) = rs.at_day {
                // Fire at at_day + k*period, for smallest k where target >= start
                let k0 = ((rs.start - at_day) / rs.period).ceil().max(0.0) as u64;
                let mut t = at_day + k0 as f64 * rs.period;
                while t <= rs.end + rs.period * 1e-9 {
                    times.push(t);
                    t += rs.period;
                }
            } else {
                let mut t = rs.start;
                while t <= rs.end + rs.period * 1e-9 {
                    times.push(t);
                    t += rs.period;
                }
            }
            times
        }
    }
}

/// Apply a known due batch of scheduled interventions at time `t_end` (in
/// declaration order). The INTERVENE-stage apply half: it consumes the
/// `intervention_idx` list [`due_effects`](crate::effects::due_effects) already
/// derived, and does ONE job — apply those interventions — instead of
/// re-deriving due-ness via `time_to_step + fire_steps.contains`.
///
/// `intervention_idx` lists exactly the firing scheduled (`!always_active`)
/// interventions in declaration order; this function does not re-check
/// `always_active` or `fire_steps`.
///
/// `dt` is `dt_actual` — the realized integrator substep (not
/// `model.simulation.dt`, which the compiled model carries only as a default —
/// the runtime can override it via `SimConfig.dt`); it drives the effect-amount
/// evaluation. See docs/dev/proposals/2026-06-07-scheduling-spine-v2.md §A/§B
/// for the two step lengths and the due-batch seam.
pub fn apply_effect_batch(
    t_end: f64,
    model: &CompiledModel,
    intervention_idx: &[usize],
    dt: f64,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
) -> Result<bool, SimError> {
    // Rm4 in 2026-04-19 engine review: guard against NaN t silently
    // rounding to step 0. The due batch is derived from `time_to_step`, which
    // debug-asserts finiteness; this keeps the runtime guard on the apply path.
    if !t_end.is_finite() {
        return Err(SimError::Validation(format!(
            "apply_effect_batch: non-finite t = {}", t_end
        )));
    }
    for &iv_idx in intervention_idx {
        let iv = &model.model.interventions[iv_idx];
        crate::effects::apply_intervention_effects(
            model, iv_idx, iv, int_s, real_s, params, t_end, dt,
        )?;
    }
    Ok(!intervention_idx.is_empty())
}

/// Apply always_active event actions directly to `int_s` / `real_s`.
///
/// gh#67: ode/gillespie do not have a `pending_deltas` pipeline
/// (only chain_binomial does, for atomic interleaving with multinomial
/// draws). They call this helper at each intervention boundary instead of
/// `inject_event_deltas`. `t_event` is the time the boundary was scheduled
/// for; `dt` is the same dt used to build `fire_steps` so the step lookup
/// matches.
pub fn apply_events_at(
    t_event: f64,
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    dt: f64,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
) -> Result<bool, SimError> {
    if !t_event.is_finite() {
        return Err(SimError::Validation(format!(
            "apply_events_at: non-finite t = {}", t_event
        )));
    }
    // Events resolve at the boundary `t_event`. `dt` here is the nominal grid the
    // `fire_steps` were built on (gillespie's `iv_resolution_dt`); the realized
    // substep coincides with it on this at-boundary event path, so it is both
    // `dt_actual` and `grid_dt`. `due_effects` keys the firing on `dt` (grid),
    // then `resolve_event_batch` resolves the events against the current state.
    let mut batch = crate::schedule::EffectBatch::default();
    crate::effects::due_effects(model, fire_steps, t_event, dt, &mut batch);
    let mut ev = crate::effects::EffectDeltas::default();
    crate::effects::resolve_event_batch(
        model, &batch.event_idx, int_s, real_s, params, t_event, dt, &mut ev,
    )?;
    let fired = !ev.is_empty();
    for d in &ev.int {
        int_s.counts[d.idx] += d.delta;
    }
    for d in &ev.real {
        real_s.values[d.idx] += d.delta;
    }
    Ok(fired)
}

/// Collect sorted, deduplicated intervention times.
///
/// gh#69: takes `params` so any `AtTimesExpr` schedules can be resolved
/// against the current parameter vector. Parametric schedules' resolved
/// expressions live on `CompiledModel.resolved.intervention_at_time_exprs`.
pub fn all_intervention_times(model: &CompiledModel, params: &[f64]) -> Vec<f64> {
    let mut times: Vec<f64> = model.model.interventions.iter()
        .enumerate()
        .flat_map(|(iv_idx, iv)| {
            let resolved = model.resolved.intervention_at_time_exprs[iv_idx].as_deref();
            intervention_fire_times(&iv.schedule, resolved, model, params)
        })
        .collect();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    times
}

/// Relative tolerance for the on-grid test: a time `t` is "on the dt grid
/// anchored at `t_start`" when `(t - t_start)/dt` is within `GRID_TOL` of an
/// integer. `1e-9` matches the schedule-arithmetic epsilon used elsewhere (e.g.
/// the `Recurring` loop bound in `intervention_fire_times` uses `period * 1e-9`,
/// and `correlated_pf::cpm_steps_per_obs` works in `interval_steps` at the same
/// scale); it is tight enough to catch a genuine off-grid observation (a
/// biweekly ES cadence at `t_start + k·dt + dt/2`) yet loose enough not to flag
/// a float-rounded on-grid time.
const GRID_TOL: f64 = 1e-9;

/// gh#216 STOPGAP GUARD (no-silent-gap interim; the real fix is cursor-keyed
/// effect firing — see
/// `docs/dev/proposals/2026-06-11-spine-effect-firing-consolidation.md`).
///
/// Under `StepPolicy::Exact` the inference filters re-anchor the substep grid at
/// **every observation time** (`build_substep_grid` sets `window_start = obs_t`
/// at each obs it lands on, `pgas.rs:405`), but the firing DECISION for a
/// SCHEDULED (`!is_event`) intervention is still made on a SECOND clock —
/// `round(t/dt)` against a precomputed `fire_steps` table (`effects.rs`). When
/// an OBSERVATION time is off the integrator grid, it re-tiles the Exact substep
/// grid so that an otherwise ON-grid intervention's substep lands at a time
/// `round(t/dt)` rounds to the WRONG step — a silent likelihood error whose
/// firing instant moves when the observation streams move. The gh#216
/// reproduction is exactly this: an intervention at day 35 (= 5·7, ON the dt=7
/// grid) fires at day 35 with AFP-only obs, but at day 37 once a biweekly ES
/// stream contributes OFF-grid observation times that re-anchor the grid. So the
/// trigger is OFF-GRID OBSERVATIONS, not off-grid effect times.
///
/// Until the firing decision moves onto the timeline (the real fix), refuse the
/// combination loudly.
///
/// Scope (exactly):
/// - `Exact` ONLY (`Snap` and forward simulation are never rejected — they stay
///   on / clip to the grid, so `round(t/dt)` is correct);
/// - models with at least one SCHEDULED (`!is_event`) intervention ONLY.
///   Always-active EVENTS are OUT of scope: PF/IF2/correlated-PF key event
///   firing on the NOMINAL `grid_dt` (the StepClock fix, spine-v2 §A), so an
///   off-grid observation does NOT misfire an event — pinned by
///   `inference_event_misfire_guard::pf_runs_off_grid_obs_with_always_active_event`.
///   (PGAS rejects event-bearing models under Exact via its own events-only
///   guard at `pgas.rs:1518`, left untouched.) A model with NO scheduled
///   intervention — no interventions at all (proposal B's plain multi-cadence
///   fit), or events-only — has no scheduled effect to misfire, so it returns
///   `Ok`;
/// - at least one OBSERVATION time OFF the integrator grid, measured RELATIVE TO
///   `t_start` (`r = (obs - t_start)/dt; (r - r.round()).abs() > GRID_TOL`). The
///   t_start-relative key mirrors `correlated_pf::cpm_steps_per_obs` /
///   `validate_cpm_obs_grid`, which already anchor the obs grid at `t_start`.
///
/// On-grid observations (every `obs` an integer multiple of `dt` past `t_start`)
/// keep the substep grid coincident with the global grid, so `round(t/dt)`
/// resolves correctly — that case (the He-2010 weekly-obs and the gh#187 on-grid
/// intervention fits) passes.
pub fn guard_exact_offgrid_obs(
    obs_times: &[f64],
    t_start: f64,
    dt: f64,
    model: &CompiledModel,
    policy: StepPolicy,
) -> Result<(), SimError> {
    if policy != StepPolicy::Exact {
        return Ok(());
    }
    // Only SCHEDULED (non-event) interventions misfire under off-grid obs.
    // Always-active events are handled by the StepClock grid_dt key (PF/IF2/
    // correlated-PF) or PGAS's events-only guard — out of scope here. A model
    // with no scheduled intervention (proposal B's plain multi-cadence fit, or an
    // events-only model) has nothing to misfire — accept it.
    let has_scheduled = model.model.interventions.iter().any(|iv| !iv.kind.is_event());
    if !has_scheduled {
        return Ok(());
    }
    let off_grid: Vec<f64> = obs_times
        .iter()
        .copied()
        .filter(|&obs| {
            let r = (obs - t_start) / dt;
            (r - r.round()).abs() > GRID_TOL
        })
        .collect();
    if off_grid.is_empty() {
        return Ok(());
    }
    let shown: Vec<String> = off_grid.iter().take(4).map(|t| format!("{t:.4}")).collect();
    let more = if off_grid.len() > 4 {
        format!(", … ({} total)", off_grid.len())
    } else {
        String::new()
    };
    Err(SimError::Validation(format!(
        "exact obs-alignment is not yet supported for a model with a scheduled \
         intervention when an OBSERVATION time is off the dt grid (dt={dt}, t_start={t_start}): \
         under Exact the substep grid re-anchors at every observation, so an off-grid \
         observation re-tiles the grid and a scheduled intervention that keys on round(t/dt) \
         fires at the wrong substep — even when the intervention time itself is on-grid (gh#216). \
         Off-grid observation time(s): [{}{more}]. Use `obs_alignment = \"snap\"`, or \
         place the observation time(s) on the dt grid (t_start + an integer multiple of \
         dt).",
        shown.join(", "),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::intervention::SetAction;
    use ir::expr::Expr;

    fn set_v() -> Action {
        Action::Set(SetAction { compartment: "V".into(), value: Expr::const_(0.0) })
    }

    /// Every non-finite kind must error before the cast. `NaN as i64 == 0`,
    /// `+inf as i64 == i64::MAX`, `-inf as i64 == i64::MIN` — all silent
    /// corruption if they reach the cast. The guard rejects all three.
    #[test]
    fn finite_action_value_rejects_non_finite() {
        let action = set_v();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = finite_action_value(bad, "campaign", &action, 5.0)
                .expect_err(&format!("{bad} must be rejected"));
            let msg = err.to_string();
            assert!(msg.contains("non-finite"), "message should name the cause: {msg}");
            assert!(msg.contains("campaign"), "message should name the intervention: {msg}");
            assert!(msg.contains("set V"), "message should label the action: {msg}");
        }
    }

    /// A finite value passes through unchanged (negative is finite — the
    /// negative-count check is a *separate* guard, applied post-state, not
    /// here).
    #[test]
    fn finite_action_value_passes_finite() {
        let action = set_v();
        for ok in [0.0, 1.0, -3.0, 1e9, f64::MIN_POSITIVE] {
            assert_eq!(
                finite_action_value(ok, "campaign", &action, 0.0).unwrap(),
                ok
            );
        }
    }
}
