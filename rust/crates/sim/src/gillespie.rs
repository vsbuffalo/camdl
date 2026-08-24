use crate::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    rng::StatefulRng,
    error::SimError,
    boundary_times::{EffectTimes, OutputTimes},
    intervention::apply_events_at,
    lineage::{DemeId, TransitionId, TransitionObserver},
    ode_integrator::rk4_step,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    schedule::{Cursor, Schedule, StopReason, MIN_STEP_EPS},
    simulate::Simulate,
    state::{Flows, FlowVec, IntState, RealState, Snapshot, Trajectory},
    transition_diagnostics::TransitionDiagnostics,
};

/// The mutable boundary state gillespie threads through the shared
/// [`Schedule::arrive`] seam (gh#233): `apply_effects` mutates the integer/real
/// compartments, `record` reads them and resets the flow tally. A borrow-struct
/// over the loop's loose locals so a single `&mut` can pass to both closures.
struct GillespieBoundary<'a> {
    int_s: &'a mut IntState,
    real_s: &'a mut RealState,
    flows: &'a mut FlowVec,
}

/// Full recompute every N events to prevent floating-point drift in lambda_total.
const FULL_RECOMPUTE_INTERVAL: usize = 10_000;

pub struct GillespieSim;

impl Simulate for GillespieSim {
    fn run(
        &self,
        model: &CompiledModel,
        params: &[f64],
        seed: u64,
        config: &SimConfig,
    ) -> Result<Trajectory, SimError> {
        let cfg = match config {
            SimConfig::Gillespie(c) => c,
            _ => return Err(SimError::ConfigMismatch {
                expected: "Gillespie",
                got: config.variant_name(),
            }),
        };
        // gh#272 LICM: `run_gillespie_with_observer` stages the per-eval prologue
        // once for this θ-stable run and lends it into every rate eval.
        run_gillespie(model, params, seed, cfg)
    }

    // capabilities() / name() below.

    fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities::REAL_COMPARTMENTS | crate::Capabilities::LINEAGES
    }

    fn name(&self) -> &'static str { "gillespie" }
}

/// Evaluate a single transition's propensity for an incremental (sparse) update.
///
/// Rejects negative / NaN propensities with the SAME typed errors the full
/// `eval_propensities` path raises, so the sparse and full paths cannot disagree
/// (gh#208). A negative rate is a model bug, not drift: silently clamping it to 0
/// turns the transition off and produces a wrong trajectory with no error. (FP
/// drift in the running `lambda_total` SUM is a separate concern, still handled
/// by the `lambda_total.max(0.0)` guards at the call sites and the periodic full
/// recompute.)
#[inline]
fn eval_one(tr_idx: usize, ctx: &EvalCtx<'_>) -> Result<f64, SimError> {
    let mut p = eval_resolved(&ctx.model.resolved.rates[tr_idx], ctx);
    // item 17: kept in lockstep with `eval_propensities` — a non-finite
    // propensity is never a usable rate, and the sparse and full paths must not
    // disagree (gh#208). NaN is the strict-mode sentinel of a degenerate op
    // (Div0 / Pow / Sqrt-neg / Log≤0), possibly a table-OOB (attributed first);
    // ±inf is an overflow (e.g. `exp` of a state-dependent argument that grows
    // as the epidemic progresses) that escaped the per-op guards — left on the
    // old `is_nan()` guard, a +inf here would set `lambda_total = +inf` and
    // force a burst of spurious zero-time firings until the periodic full
    // recompute finally errored. Under --allow-degenerate-rates all coerce to a
    // 0 rate; by default a typed hard error.
    if !p.is_finite() {
        // A NaN may be the sentinel an out-of-range table lookup left on the
        // thread-local — surface the named, actionable error if so. take()
        // clears it; ±inf leaves no record, so this is None and falls through.
        if let Some((table_idx, index, len)) = crate::resolved_expr::take_table_oob() {
            let table_name = ctx.model.model.tables[table_idx].name.clone();
            return Err(SimError::TableLookup(format!(
                "table '{table_name}': index {index} out of bounds [0, {len}) \
                 while evaluating rate of transition '{}' at t={} \
                 (the index is computed from model state/parameters; widen the \
                 table or fix the index expression)",
                ctx.model.model.transitions[tr_idx].name, ctx.t
            )));
        }
        if crate::eval_stats::allow_degenerate_rates() {
            p = 0.0;
        } else {
            return Err(SimError::NumericalCollapse {
                kind: crate::error::CollapseKind::DivByZero,
                t: ctx.t,
            });
        }
    }
    if p < 0.0 {
        return Err(SimError::NegativePropensity {
            transition: ctx.model.model.transitions[tr_idx].name.clone(),
            value: p,
            t: ctx.t,
        });
    }
    Ok(p)
}

pub fn run_gillespie(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &GillespieConfig,
) -> Result<Trajectory, SimError> {
    run_gillespie_with_observer(model, params, seed, cfg, None, None)
}

/// Gillespie run with an optional [`TransitionObserver`] attached to the event
/// loop (individual-sampling layer, 2026-05-19 proposal).
///
/// `observer = None` reproduces [`run_gillespie`] byte-for-byte: the observer
/// is only consulted *after* the simulation RNG has selected the firing
/// transition and is passed the pre-stoichiometry state, so it cannot reorder
/// or add draws to the simulation's `StatefulRng`. This is the load-bearing
/// trajectory-invariance invariant (validation Tier 2a).
pub fn run_gillespie_with_observer(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &GillespieConfig,
    mut observer: Option<&mut dyn TransitionObserver>,
    // Per-event progress tick (RNG-free; `None` == byte-identical). Gillespie
    // is event-driven, so this fires once per event with the current time `t`.
    // See chain_binomial.rs and tests/progress_tick_invariance.rs.
    mut tick: Option<&mut dyn FnMut(f64)>,
) -> Result<Trajectory, SimError> {
    // Paired-seed coupling: running baseline and intervention with the same
    // seed produces identical trajectories up to the first state divergence,
    // because the stateful PRNG's output only depends on its prior consumption
    // sequence. Any change that reorders or adds draws before that point also
    // breaks the coupling — this is NOT event-keyed RNG.
    //
    // Constructed before the initial state because building the initial state
    // is itself a draw from this stream (`initial_state_draw`); construction
    // consumes no randomness, so the event loop's stream is unchanged.
    let mut stateful_rng = StatefulRng::new(seed);

    let (mut int_s, mut real_s) = model.initial_state_draw(params, &mut stateful_rng)?;

    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // Per-transition firing diagnostics
    let mut diag_vec: Vec<TransitionDiagnostics> = model.model.transitions.iter()
        .map(|t| TransitionDiagnostics::new(t.name.clone()))
        .collect();

    // Propensity buffer — allocated once, reused
    let mut propensities: Vec<f64> = Vec::with_capacity(n_transitions);

    // Sorted output times
    // gh#53: resolve fire_steps using the model's compile-time dt.
    // Gillespie has no runtime dt of its own (continuous-time SSA); the
    // fire_steps lookup uses model.simulation.dt as a step-rounding
    // resolution. Pre-gh#53 this was implicit inside
    // apply_interventions_at; making it explicit means gillespie shares
    // the same call signature as the dt-parameterised backends without
    // changing observed semantics.
    let iv_resolution_dt = model.model.simulation.dt.unwrap_or(1.0);
    // gh#126: reject a non-finite/non-positive intervention-resolution dt
    // or a non-finite fire time at entry — a RELEASE-build check (the
    // per-conversion guards in `time.rs` are debug_assert only).
    model.validate_schedule(iv_resolution_dt, params)?;
    let fire_steps = model.resolve_fire_steps(iv_resolution_dt, params);

    // gh#272 LICM: stage the per-eval prologue ONCE for this forward run. `params`
    // is fixed for the whole run, so the param/table-only `per_eval_bindings` are
    // evaluated here and lent into every rate eval of every event — not recomputed
    // per event. The scratch is owned here and passed as data (no shared cache to
    // alias). `None` for models without per-eval bindings, where `PerEvalRef` would
    // fall through to on-demand eval anyway. `t`/`dt` are inert (a per-eval body
    // reads no `Time`/`Dt`).
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, cfg.t_start, iv_resolution_dt);
    let per_eval = per_eval_scratch.as_deref();

    // Merged timeline spine. Gillespie is event-driven: it PROPOSES an
    // exponential time and the schedule CLIPS it to the next boundary
    // (Schedule::clip). The grid is iv_resolution_dt (no integrator dt of its
    // own); StepPolicy is irrelevant to clip. The schedule owns the sorted
    // output/effect times; `cursor` walks them. Firing stays inline.
    let schedule = Schedule::ssa_forward(
        iv_resolution_dt,
        cfg.t_end,
        OutputTimes::from_model(model)?,
        EffectTimes::from_model(model, params)?,
    );
    let mut cursor = Cursor::default();

    let mut t = cfg.t_start;
    let mut traj = Trajectory::new();
    let mut current_flows = FlowVec::new(n_transitions);

    // Record initial state. Initial-row convention (see `Trajectory` docs):
    // the t_start snapshot carries zeroed flows so `Σ flow == −Δstate`
    // reconciles over the whole path (gh#270).
    if schedule.output_due_at(&cursor, t) {
        traj.push(Snapshot {
            t,
            int_state: int_s.clone(),
            real_state: real_s.clone(),
            flows: Flows::Int(current_flows.counts.clone()),
        });
        current_flows.reset();
        cursor.pass_output();
    }

    // Initial full propensity evaluation — maintained incrementally from here on.
    eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), per_eval, &mut propensities)?;
    let mut lambda_total: f64 = propensities.iter().sum();
    let mut event_count: usize = 0;

    loop {
        if t >= cfg.t_end { break; }

        // Progress tick: report current time before drawing this event. RNG-free.
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // If lambda_total looks zero (from incremental drift or genuine absorbing state),
        // do a full recompute to verify before treating as absorbing.
        if lambda_total <= 0.0 {
            eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), per_eval, &mut propensities)?;
            lambda_total = propensities.iter().sum();
        }

        if lambda_total <= 0.0 {
            // Absorbing state: no integer reaction will fire. Advance to the next
            // timeline boundary via the single authority (`next_stop`) and dispatch
            // through the shared `arrive` seam (gh#233) — the same seam every exact
            // backend uses, so the effect→output order, the coincident batch, and
            // the terminal output cannot diverge per-backend (gh#70).
            //
            // RAW `next_stop` is safe here precisely because `arrive` couples
            // apply-and-pass: a scheduled effect at `t` is applied AND its cursor
            // advanced in one place, so the next iteration's `next_stop` sees the
            // following boundary and the clock advances — no non-terminating loop
            // (the reason the old hand-rolled flush needed `clip`'s `> t` filter).
            // One boundary at a time keeps the output cursor in lockstep with `t`.
            let Some(stop) = schedule.next_stop(&cursor, t) else { break };
            t = stop.t;
            {
                // Canonical lifecycle: always-active events fire FIRST (reading the
                // start-of-step snapshot — gillespie has no transition step at a
                // boundary), then interventions on the post-event state.
                let mut bs = GillespieBoundary {
                    int_s: &mut int_s,
                    real_s: &mut real_s,
                    flows: &mut current_flows,
                };
                schedule.arrive(
                    &mut cursor,
                    &stop,
                    t,
                    &mut bs,
                    |bs, bt| {
                        apply_events_at(bt, model, &fire_steps, iv_resolution_dt, bs.int_s, bs.real_s, params)?;
                        let mut batch = crate::schedule::EffectBatch::default();
                        crate::effects::due_effects(model, &fire_steps, bt, iv_resolution_dt, &mut batch);
                        crate::lifecycle::apply_post_advance(
                            model, &batch.intervention_idx, bs.int_s, bs.real_s, params,
                            bt - iv_resolution_dt, iv_resolution_dt, None,
                        )
                    },
                    |bs, ot| {
                        traj.push(Snapshot {
                            t: ot,
                            int_state: bs.int_s.clone(),
                            real_state: bs.real_s.clone(),
                            flows: Flows::Int(bs.flows.counts.clone()),
                        });
                        bs.flows.reset();
                    },
                )?;
            }
            // If a scheduled effect fired the state changed — recompute; the model
            // may leave the absorbing state.
            if stop.has(StopReason::ScheduledEffect) {
                eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), per_eval, &mut propensities)?;
                lambda_total = propensities.iter().sum();
            }

            if stop.is_end() { break; }
            continue;
        }

        // Draw time to next event (stateful for global clock, but keyed per transition)
        // For Gillespie: draw total waiting time, then select transition
        let u1: f64 = stateful_rng.uniform();
        let dt = -(1.0 / lambda_total) * u1.ln();
        let t_next = t + dt;

        // `clip` is the SSA reaction-vs-boundary predicate: does the proposed
        // reaction at `t_next` fire before the next boundary? On a boundary win we
        // share the SAME `arrive` dispatch as the absorbing branch and ODE (gh#233).
        let clipped = schedule.clip(&cursor, t, t_next);

        if clipped.hit_boundary {
            let boundary = clipped.t;
            // Advance real state to the boundary without firing an event.
            // TODO(v0.2): replace with PDMP thinning for real compartments.
            if n_real > 0 && (boundary - t) > MIN_STEP_EPS {
                rk4_step(model, &int_s, &mut real_s, params, t, boundary - t)?;
                real_s.clamp_nonneg();
            }
            t = boundary;

            // `clip` and `next_stop` share the boundary min, and the cursor still
            // points at this boundary's unconsumed reasons, so `next_stop` reports
            // them. Dispatch via the shared seam: events FIRST (start-of-step
            // snapshot — gillespie has no transition step here), then interventions
            // on the post-event state, then output.
            let stop = schedule.next_stop(&cursor, t).expect("clip hit a boundary <= t_end");
            {
                let mut bs = GillespieBoundary {
                    int_s: &mut int_s,
                    real_s: &mut real_s,
                    flows: &mut current_flows,
                };
                schedule.arrive(
                    &mut cursor,
                    &stop,
                    t,
                    &mut bs,
                    |bs, bt| {
                        apply_events_at(bt, model, &fire_steps, iv_resolution_dt, bs.int_s, bs.real_s, params)?;
                        let mut batch = crate::schedule::EffectBatch::default();
                        crate::effects::due_effects(model, &fire_steps, bt, iv_resolution_dt, &mut batch);
                        crate::lifecycle::apply_post_advance(
                            model, &batch.intervention_idx, bs.int_s, bs.real_s, params,
                            bt - iv_resolution_dt, iv_resolution_dt, None,
                        )
                    },
                    |bs, ot| {
                        traj.push(Snapshot {
                            t: ot,
                            int_state: bs.int_s.clone(),
                            real_state: bs.real_s.clone(),
                            flows: Flows::Int(bs.flows.counts.clone()),
                        });
                        bs.flows.reset();
                    },
                )?;
            }

            if stop.has(StopReason::ScheduledEffect) {
                // Full recompute after intervention (integer state changed).
                eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), per_eval, &mut propensities)?;
                lambda_total = propensities.iter().sum();
            } else {
                // Time advanced but no state change: re-evaluate time-dependent transitions.
                let ctx = EvalCtx { model, int_s: &int_s, real_s: &real_s, params, t, dt: model.model.simulation.dt.unwrap_or(1.0), projected: None, aux: None, int_float_override: None, per_eval };
                for &tr_idx in &model.time_dep_transitions {
                    let old = propensities[tr_idx];
                    let new_p = eval_one(tr_idx, &ctx)?;
                    propensities[tr_idx] = new_p;
                    lambda_total += new_p - old;
                }
                lambda_total = lambda_total.max(0.0);
            }

            if stop.is_end() { break; }
            continue;
        }

        // Fire an event: select transition proportional to propensity
        let u2: f64 = stateful_rng.uniform();
        let threshold = u2 * lambda_total;
        let mut cumulative = 0.0;
        let mut fired_idx = n_transitions - 1;
        for (i, &p) in propensities.iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                fired_idx = i;
                break;
            }
        }

        // Advance real state to event time
        // TODO(v0.2): replace with PDMP thinning
        if n_real > 0 && dt > MIN_STEP_EPS {
            rk4_step(model, &int_s, &mut real_s, params, t, dt)?;
            real_s.clamp_nonneg();
        }
        t = t_next;

        // Record firing diagnostics
        diag_vec[fired_idx].record_firing(t, propensities[fired_idx]);

        // Lineage observer: called AFTER the simulation RNG picked the firing
        // transition, with the pre-stoichiometry (event-instant) state. The
        // observer owns its own RNG stream, so it cannot perturb the count
        // trajectory. Single-population slice → deme 0, multiplicity 1.
        if let Some(obs) = observer.as_deref_mut() {
            obs.on_fired(TransitionId(fired_idx), DemeId(0), 1, t, &int_s, &real_s, params)?;
        }

        // Apply stoichiometry
        for &(local, delta) in &model.transition_stoich[fired_idx] {
            int_s.counts[local] += delta;
        }

        // gh#audit-C5 / S2. Gillespie's source compartment has at least
        // one individual when a transition fires — by construction. So
        // a negative count post-stoichiometry is a real model bug
        // worth surfacing. Returns SimError::NegativeCount; inference
        // layers catch and recover per-particle (though for Gillespie
        // this should be a structural error, not particle-recoverable).
        if let Some((local, val)) = int_s.first_negative() {
            return Err(SimError::NegativeCount {
                compartment: model.comp_index.iter()
                    .find(|(_, &g)| model.global_to_int.get(g).copied().flatten() == Some(local))
                    .map(|(n, _)| n.clone())
                    .unwrap_or_else(|| format!("(local-int-{local})")),
                attempted_value: val,
                t,
                cause: crate::error::NegativeCountCause::BinomialOvershoot,
            });
        }

        // Track flow
        current_flows.add(fired_idx, 1);

        // --- Sparse propensity update ---
        event_count += 1;
        if event_count.is_multiple_of(FULL_RECOMPUTE_INTERVAL) {
            // Periodic full recompute prevents floating-point drift in lambda_total
            eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), per_eval, &mut propensities)?;
            lambda_total = propensities.iter().sum();
        } else {
            // Incremental update: only recompute transitions whose dependencies changed.
            // `updated` tracks which transitions we've already recomputed this step to
            // avoid evaluating the same transition twice when multiple stoich entries
            // share a dependent transition (e.g., N[p] = S[p] + E[p] + ...).
            let mut updated: Vec<usize> = Vec::with_capacity(16);
            let ctx = EvalCtx { model, int_s: &int_s, real_s: &real_s, params, t, dt: model.model.simulation.dt.unwrap_or(1.0), projected: None, aux: None, int_float_override: None, per_eval };

            // Compartment-dependent transitions
            for &(local, _) in &model.transition_stoich[fired_idx] {
                for &tr_idx in &model.comp_to_transitions()[local] {
                    if !updated.contains(&tr_idx) {
                        let old = propensities[tr_idx];
                        let new_p = eval_one(tr_idx, &ctx)?;
                        propensities[tr_idx] = new_p;
                        lambda_total += new_p - old;
                        updated.push(tr_idx);
                    }
                }
            }

            // Time-dependent transitions at new t (skip if already updated above)
            for &tr_idx in &model.time_dep_transitions {
                if !updated.contains(&tr_idx) {
                    let old = propensities[tr_idx];
                    let new_p = eval_one(tr_idx, &ctx)?;
                    propensities[tr_idx] = new_p;
                    lambda_total += new_p - old;
                }
            }

            // Prevent negative drift accumulation
            lambda_total = lambda_total.max(0.0);
        }

        // Record output at any output times we've passed
        schedule.drain_outputs(&mut cursor, t, |ot| {
            traj.push(Snapshot {
                t: ot,
                int_state: int_s.clone(),
                real_state: real_s.clone(),
                flows: Flows::Int(current_flows.counts.clone()),
            });
            current_flows.reset();
        });

    }

    // Ensure final output time is recorded
    schedule.drain_outputs(&mut cursor, f64::INFINITY, |ot| {
        traj.push(Snapshot {
            t: ot,
            int_state: int_s.clone(),
            real_state: real_s.clone(),
            flows: Flows::Int(current_flows.counts.clone()),
        });
        current_flows.reset();
    });

    traj.transition_diagnostics = diag_vec;
    Ok(traj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use ir::{
        expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, PopExpr, UnOp, UnOpExpr, UnOpWrap},
        model::{
            Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
            SimulationConfig,
        },
        transition::{StoichiometryEntry, Transition},
        Model,
    };

    // A one-transition model whose rate is `exp(1000 * S)`; at S = 1 that is
    // `exp(1000)` = +inf. A +inf rate is a state-dependent overflow that is
    // finite at a small S but blows up as the epidemic grows — the case the
    // sparse update path (`eval_one`) hits mid-run.
    fn overflow_model() -> CompiledModel {
        let m = Model {
            ic_grad: Default::default(),
            name: "overflow".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![Transition {
                rate_state_grad: Default::default(),
                name: "blowup".into(),
                stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                rate: Expr::UnOp(UnOpWrap { un_op: UnOpExpr {
                    op: UnOp::Exp,
                    arg: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Const(ConstExpr { value: 1000.0 })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                    }})),
                }}),
                metadata: None,
                draw_method: Default::default(),
                rate_grad: HashMap::new(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![],
            initial_conditions: InitialConditions::constants([
                ("S".into(), 1.0),
                ("I".into(), 0.0)
            ]),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(1), integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
            quantities: vec![],
            contrasts: vec![],
        };
        CompiledModel::new(m).expect("overflow model compiles")
    }

    /// item 17 / gh#208: the sparse-update evaluator `eval_one` must reject a
    /// +inf propensity with the SAME typed error the full `eval_propensities`
    /// path raises — NOT return `Ok(+inf)`, which would set `lambda_total = +inf`
    /// and force a burst of spurious zero-time firings. Asserts strict-mode
    /// (the default) only: the allow-coerce arm is structurally identical to
    /// `eval_propensities` (whose coercion is covered by the expr_eval
    /// integration tests) and is not re-toggled here because
    /// `ALLOW_DEGENERATE_RATES` is a process-global shared with concurrent tests.
    #[test]
    fn eval_one_rejects_infinite_rate_like_eval_propensities() {
        let model = overflow_model();
        let int_s = IntState::from_vec(vec![1, 0]); // S = 1 → exp(1000) = +inf
        let real_s = RealState::new(0);
        let ctx = EvalCtx {
            model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0,
            projected: None, aux: None, int_float_override: None, per_eval: None,
        };

        // Sparse path: must error, not return Ok(+inf).
        let sparse = eval_one(0, &ctx);
        assert!(
            matches!(sparse, Err(SimError::NumericalCollapse { .. })),
            "eval_one must reject a +inf propensity with NumericalCollapse (strict), got {sparse:?}"
        );
        // Full path agrees — the gh#208 "sparse and full cannot disagree" invariant.
        let mut out = Vec::new();
        let full = eval_propensities(&model, &int_s, &real_s, &[], 0.0, 1.0, None, &mut out);
        assert!(
            matches!(full, Err(SimError::NumericalCollapse { .. })),
            "eval_propensities must reject the same +inf, got {full:?}"
        );
    }
}
