use crate::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    rng::StatefulRng,
    error::SimError,
    intervention::{all_intervention_times, apply_events_at},
    lineage::TransitionObserver,
    ode_integrator::rk4_step,
    output::output_times as get_output_times,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    schedule::{Cursor, Schedule, StepPolicy, StopReason, MIN_STEP_EPS},
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
    let p = eval_resolved(&ctx.model.resolved.rates[tr_idx], ctx);
    if p.is_nan() {
        // Mirror eval_propensities: a NaN may be the sentinel an out-of-range
        // table lookup left on the thread-local — surface the named, actionable
        // error if so, else the generic numerical collapse. take() clears it.
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
        return Err(SimError::NumericalCollapse {
            kind: crate::error::CollapseKind::DivByZero,
            t: ctx.t,
        });
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
    let (mut int_s, mut real_s) = model.initial_state(params)?;

    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // Per-transition firing diagnostics
    let mut diag_vec: Vec<TransitionDiagnostics> = model.model.transitions.iter()
        .map(|t| TransitionDiagnostics::new(t.name.clone()))
        .collect();

    // Propensity buffer — allocated once, reused
    let mut propensities: Vec<f64> = Vec::with_capacity(n_transitions);

    // Paired-seed coupling: running baseline and intervention with the same
    // seed produces identical trajectories up to the first state divergence,
    // because the stateful PRNG's output only depends on its prior consumption
    // sequence. Any change that reorders or adds draws before that point also
    // breaks the coupling — this is NOT event-keyed RNG.
    let mut stateful_rng = StatefulRng::new(seed);

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

    // Merged timeline spine. Gillespie is event-driven: it PROPOSES an
    // exponential time and the schedule CLIPS it to the next boundary
    // (Schedule::clip). The grid is iv_resolution_dt (no integrator dt of its
    // own); StepPolicy is irrelevant to clip. The schedule owns the sorted
    // output/effect times; `cursor` walks them. Firing stays inline.
    let schedule = Schedule::new(
        iv_resolution_dt,
        cfg.t_end,
        iv_resolution_dt,
        StepPolicy::Exact,
        get_output_times(&model.model.output.times),
        all_intervention_times(model, params),
    );
    let mut cursor = Cursor::default();

    let mut t = cfg.t_start;
    let mut traj = Trajectory::new();
    let mut current_flows = FlowVec::new(n_transitions);

    // Record initial state
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
    eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), &mut propensities)?;
    let mut lambda_total: f64 = propensities.iter().sum();
    let mut event_count: usize = 0;

    loop {
        if t >= cfg.t_end { break; }

        // Progress tick: report current time before drawing this event. RNG-free.
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // If lambda_total looks zero (from incremental drift or genuine absorbing state),
        // do a full recompute to verify before treating as absorbing.
        if lambda_total <= 0.0 {
            eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), &mut propensities)?;
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
                eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), &mut propensities)?;
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
                eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), &mut propensities)?;
                lambda_total = propensities.iter().sum();
            } else {
                // Time advanced but no state change: re-evaluate time-dependent transitions.
                let ctx = EvalCtx { model, int_s: &int_s, real_s: &real_s, params, t, dt: model.model.simulation.dt.unwrap_or(1.0), projected: None, aux: None, int_float_override: None };
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
            obs.on_fired(fired_idx, 0, 1, t, &int_s, &real_s, params)?;
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
            eval_propensities(model, &int_s, &real_s, params, t, model.model.simulation.dt.unwrap_or(1.0), &mut propensities)?;
            lambda_total = propensities.iter().sum();
        } else {
            // Incremental update: only recompute transitions whose dependencies changed.
            // `updated` tracks which transitions we've already recomputed this step to
            // avoid evaluating the same transition twice when multiple stoich entries
            // share a dependent transition (e.g., N[p] = S[p] + E[p] + ...).
            let mut updated: Vec<usize> = Vec::with_capacity(16);
            let ctx = EvalCtx { model, int_s: &int_s, real_s: &real_s, params, t, dt: model.model.simulation.dt.unwrap_or(1.0), projected: None, aux: None, int_float_override: None };

            // Compartment-dependent transitions
            for &(local, _) in &model.transition_stoich[fired_idx] {
                for &tr_idx in &model.comp_to_transitions[local] {
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
