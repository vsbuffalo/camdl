//! Pure effect resolution + trivial application.
//!
//! The within-substep effect system has two orthogonal axes: **representation**
//! (integer `i64` vs continuous `f64` compartments) and **purity** (resolving an
//! effect — reading a snapshot, computing a delta, no side effect — vs applying
//! it). This module separates both:
//!
//!   - [`resolve_intervention`] is PURE: it reads an immutable [`StateRef`] and
//!     emits typed [`IntDelta`]/[`RealDelta`] entries into an [`EffectDeltas`].
//!     All the bug-prone arithmetic — rounding mode, clamps, snapshot
//!     subtraction, arena dispatch — lives here, once, testable as plain data.
//!   - [`apply_effects`] is TRIVIAL: it writes the deltas into a [`StateMut`].
//!     No arithmetic, no branch on representation, so it cannot carry a bug.
//!
//! Representation collapses into the delta *type* (no runtime `match` at apply
//! time); purity collapses into the `Ref`/`Mut` *types* (a resolver cannot
//! mutate). The per-action rules reproduce the historical behaviour exactly:
//! `round` for add/set/absolute-transfer, `floor` for fraction-transfer, the
//! `frac ∈ [0,1]` and `.min(src)` clamps; the real arena applies exact `f64`.
//!
//! Two historical asymmetries are unified here (both byte-identical on every
//! current model, since no fixture exercises either): events targeting a real
//! compartment now apply instead of being dropped, and a negative `add`
//! resolves to a hard error on every path, not just the intervention path.

use crate::{
    compiled_model::CompiledModel,
    error::{NegativeCountCause, SimError},
    propensity::EvalCtx,
    resolved_expr::eval_resolved,
    state::{IntState, RealState},
};
use ir::intervention::{Action, Intervention};

/// Immutable read view over the two compartment arenas. A resolver takes this
/// and *cannot* mutate state — the purity half of the seam, type-enforced.
#[derive(Clone, Copy)]
pub struct StateRef<'a> {
    pub int: &'a IntState,
    pub real: &'a RealState,
}

/// Mutable apply target — the other half of the purity seam.
pub struct StateMut<'a> {
    pub int: &'a mut IntState,
    pub real: &'a mut RealState,
}

/// A change to one integer compartment (local int index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntDelta {
    pub idx: usize,
    pub delta: i64,
}

/// A change to one real compartment (local real index).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealDelta {
    pub idx: usize,
    pub delta: f64,
}

/// The typed output of resolution: deltas for each arena. Representation is
/// carried by the entry type, so application needs no runtime branch.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct EffectDeltas {
    pub int: Vec<IntDelta>,
    pub real: Vec<RealDelta>,
}

impl EffectDeltas {
    pub fn is_empty(&self) -> bool {
        self.int.is_empty() && self.real.is_empty()
    }
    pub fn clear(&mut self) {
        self.int.clear();
        self.real.clear();
    }
}

/// Apply resolved deltas in order. Trivial — no arithmetic, no representation
/// branch; a delta either lands in `int` or `real` by its own type.
pub fn apply_effects(d: &EffectDeltas, s: StateMut<'_>) {
    for IntDelta { idx, delta } in &d.int {
        s.int.counts[*idx] += *delta;
    }
    for RealDelta { idx, delta } in &d.real {
        s.real.values[*idx] += *delta;
    }
}

/// Whether the actions came from a scheduled intervention (post-advance,
/// applied in place) or an always-active event (pre-advance snapshot, fused
/// with the draw). Used only for the `CAMDL_TRACE_STEPS` label today.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    Intervention,
    Event,
}

/// gh#217: which read-state an always-active EVENT action resolves against
/// within a chain_binomial substep. The two phases let `step_one` apply inflow
/// and draining event actions against different states without re-resolving the
/// whole batch twice:
///
///   - [`EventPhase::Snapshot`] selects INFLOW-only actions (`Add`): resolved
///     against the start-of-step snapshot and fused into the atomic transition
///     apply. Byte-identical to the pre-gh#217 behaviour.
///   - [`EventPhase::Residual`] selects DRAINING / assignment actions
///     (`FractionTransfer`, `AbsoluteTransfer`, `Set`): resolved against the
///     POST-TRANSITION residual state, so a draining transfer moves a fraction
///     of what survived the interval — matching ODE / Gillespie.
///
/// Action variant → phase is a pure classification; see [`action_phase`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EventPhase {
    /// Inflow `Add` only, resolved against the start-of-step snapshot.
    Snapshot,
    /// `FractionTransfer` / `AbsoluteTransfer` / `Set`, resolved against the
    /// post-transition residual.
    Residual,
}

/// Classify one event action by the read-state it must resolve against (gh#217).
///
/// `Add` is an INFLOW construct (cannot over-draw) → [`EventPhase::Snapshot`].
/// `FractionTransfer` / `AbsoluteTransfer` drain their `from` side and `Set`
/// overwrites the post-dynamics value → [`EventPhase::Residual`].
///
/// NOTE (follow-up, out of scope for gh#217): a NEGATIVE-amount `Add` used as a
/// drain is classified `Snapshot` here like any other `Add`. Such an `Add` is
/// rejected today as a config bug (`InterventionAddNegative`), so it cannot
/// silently over-draw; if negative `Add`-as-drain is ever supported it must move
/// to `Residual`.
fn action_phase(action: &Action) -> EventPhase {
    match action {
        Action::Add(_) => EventPhase::Snapshot,
        Action::FractionTransfer(_) | Action::AbsoluteTransfer(_) | Action::Set(_) => {
            EventPhase::Residual
        }
    }
}

impl EffectKind {
    fn label(self) -> &'static str {
        match self {
            EffectKind::Intervention => "INTERVENTION",
            EffectKind::Event => "EVENT",
        }
    }
}

/// `CAMDL_TRACE_STEPS` observability for one action. Stderr-only, env-gated, no
/// effect on results — kept out of the pure resolver and emitted at the wiring.
fn trace_action(kind: EffectKind, iv_name: &str, action: &Action, v: f64, t: f64) {
    if !crate::chain_binomial::trace_enabled() {
        return;
    }
    let k = kind.label();
    match action {
        Action::Add(a) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: add {} += {} (raw={v:.2})",
            a.compartment, v.round() as i64
        ),
        Action::Set(a) => {
            eprintln!("{k} '{iv_name}' at t={t:.1}: set {} = {v:.2}", a.compartment)
        }
        Action::FractionTransfer(ft) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: transfer {} -> {} (frac={:.2})",
            ft.src, ft.dst, v.clamp(0.0, 1.0)
        ),
        Action::AbsoluteTransfer(at) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: transfer {} -> {} (raw={v:.2})",
            at.src, at.dst
        ),
    }
}

/// The scheduled-intervention path: resolve + apply each action **sequentially**
/// against the live post-advance state, so action `i+1` sees action `i`'s effect
/// (the historical `apply_intervention` semantics — distinct from the event
/// path, which resolves every action against one frozen pre-advance snapshot).
/// Byte-identical to the prior in-place apply.
#[allow(clippy::too_many_arguments)]
pub fn apply_intervention_effects(
    model: &CompiledModel,
    iv_idx: usize,
    iv: &Intervention,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Result<(), SimError> {
    let mut out = EffectDeltas::default();
    for (action_idx, action) in iv.actions.iter().enumerate() {
        out.clear();
        resolve_one(
            model, iv_idx, action_idx, &iv.name, action,
            StateRef { int: int_s, real: real_s }, params, t, dt,
            EffectKind::Intervention, &mut out,
        )?;
        apply_effects(&out, StateMut { int: int_s, real: real_s });
    }
    Ok(())
}

/// Which arena a compartment lives in, plus its local index.
enum Arena {
    Int(usize),
    Real(usize),
}

/// Resolve a compartment name to its arena + local index (the same dispatch the
/// rate evaluator uses: `comp_index → global → global_to_int else global_to_real`).
fn resolve_target(model: &CompiledModel, name: &str) -> Result<Arena, SimError> {
    let g = *model
        .comp_index
        .get(name)
        .ok_or_else(|| SimError::UnknownCompartment(name.to_string()))?;
    if let Some(i) = model.global_to_int[g] {
        Ok(Arena::Int(i))
    } else if let Some(i) = model.global_to_real[g] {
        Ok(Arena::Real(i))
    } else {
        Err(SimError::UnknownCompartment(name.to_string()))
    }
}

/// Resolve one action against `snap`: evaluate its amount expression, finite-
/// check it, trace it, and append the typed delta(s) to `out`. The single
/// per-action path shared by the intervention (sequential) and event (parallel)
/// resolvers. PURE w.r.t. state — no mutation, no RNG.
#[allow(clippy::too_many_arguments)]
fn resolve_one(
    model: &CompiledModel,
    iv_idx: usize,
    action_idx: usize,
    iv_name: &str,
    action: &Action,
    snap: StateRef<'_>,
    params: &[f64],
    t: f64,
    dt: f64,
    kind: EffectKind,
    out: &mut EffectDeltas,
) -> Result<(), SimError> {
    let ctx = EvalCtx {
        model, int_s: snap.int, real_s: snap.real, params, t, dt,
        projected: None, aux: None, int_float_override: None,
    };
    let v = eval_resolved(&model.resolved.intervention_exprs[iv_idx][action_idx], &ctx);
    let v = crate::intervention::finite_action_value(v, iv_name, action, t)?;
    trace_action(kind, iv_name, action, v, t);
    resolve_action(model, action, v, snap, t, out)
}

/// Resolve the actions of one intervention/event matching `phase` against the
/// SAME `snap`, appending the typed deltas to `out` (the parallel idiom — every
/// resolved action sees the same `snap`). PURE.
///
/// gh#217: `phase` filters by action read-state. The EVENT path resolves
/// `Snapshot` (inflow `Add`) against the start-of-step snapshot and `Residual`
/// (draining transfer / `Set`) against the post-transition residual in two
/// passes with different `snap`. `Some(p)` resolves only actions whose
/// [`action_phase`] is `p`; `None` resolves every action (the intervention path,
/// which has no two-phase split — interventions apply post-advance regardless).
#[allow(clippy::too_many_arguments)]
pub fn resolve_intervention(
    model: &CompiledModel,
    iv_idx: usize,
    iv: &Intervention,
    snap: StateRef<'_>,
    params: &[f64],
    t: f64,
    dt: f64,
    kind: EffectKind,
    phase: Option<EventPhase>,
    out: &mut EffectDeltas,
) -> Result<(), SimError> {
    for (action_idx, action) in iv.actions.iter().enumerate() {
        if let Some(p) = phase {
            if action_phase(action) != p {
                continue;
            }
        }
        resolve_one(model, iv_idx, action_idx, &iv.name, action, snap, params, t, dt, kind, out)?;
    }
    Ok(())
}

/// The single due-ness check for a substep: which interventions fire at the
/// boundary `t_end`, pre-split by lifecycle stage into an [`EffectBatch`]. This
/// is the ONE place `time_to_step(t_end, grid_dt) + fire_steps.contains` lives —
/// the events path (PROPOSE) and the interventions path (INTERVENE) each consume
/// the batch instead of re-deriving due-ness, so the two stages can never
/// disagree about what is due (the duplication this seam removes).
///
/// `current_step = time_to_step(t_end, grid_dt)` is computed ONCE. Interventions
/// are scanned in DECLARATION ORDER (`model.model.interventions`); a firing one
/// lands in `event_idx` if `always_active`, else `intervention_idx` — preserving
/// the historical per-stage firing order exactly.
///
/// `grid_dt` is the nominal model dt the `fire_steps` step-index table was built
/// on (`resolve_fire_steps(grid_dt, …)`), so the FIRING KEY is on `grid_dt`, not
/// the realized `dt_actual`. They are equal under Snap and for on-grid Exact
/// substeps; they diverge only when an inference filter clips a substep to land
/// on an off-grid observation — the `StepClock` discipline of
/// docs/dev/proposals/2026-06-07-scheduling-spine-v2.md §A.
///
/// `out` is caller-provided and reused: [`EffectBatch::clear`] resets it without
/// freeing, so the hot inference path (one batch per particle per substep)
/// allocates nothing per call.
pub fn due_effects(
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    t_end: f64,
    grid_dt: f64,
    out: &mut crate::schedule::EffectBatch,
) {
    out.clear();
    let current_step = crate::time::time_to_step(t_end, grid_dt);
    for (iv_idx, iv) in model.model.interventions.iter().enumerate() {
        if !fire_steps[iv_idx].contains(&current_step) {
            continue;
        }
        if iv.kind.is_event() {
            out.event_idx.push(iv_idx);
        } else {
            out.intervention_idx.push(iv_idx);
        }
    }
}

/// Split a flat list of due effect indices — the batch the timeline cursor
/// reports at one effect boundary (`timeline_effects().batches[effect_idx]`) —
/// into the lifecycle-stage halves of an [`EffectBatch`]: always-active EVENTS
/// fire at PROPOSE (`event_idx`, fused with the kernel draw against the
/// start-of-step snapshot); scheduled interventions fire at INTERVENE
/// (`intervention_idx`, applied on the post-advance state). Indices are into
/// `model.model.interventions`, declaration order preserved.
///
/// This is the ONE place the kind→stage routing lives for every Exact-INFERENCE
/// caller (bootstrap PF / IF2 / correlated PF / PGAS producer). It replaces the
/// former `due_events` `round(t_end/grid_dt)` event path: once events are
/// registered on the timeline (`timeline_effects` no longer excludes them), the
/// integrator LANDS on each event time and the cursor reports it in the batch, so
/// firing is cursor-keyed for every kind — no `round()` on an off-grid,
/// obs-anchored substep end (gh#216, the events arm). The Snap-forward backends
/// keep their on-grid `round(t/dt)` key via [`due_effects`].
pub fn split_due_batch(
    model: &CompiledModel,
    due: &[usize],
    out: &mut crate::schedule::EffectBatch,
) {
    out.clear();
    for &iv_idx in due {
        if model.model.interventions[iv_idx].kind.is_event() {
            out.event_idx.push(iv_idx);
        } else {
            out.intervention_idx.push(iv_idx);
        }
    }
}

/// Resolve a known batch of always-active events into typed deltas, for the
/// actions matching `phase`. The EVENT path's apply half: each event in
/// `event_idx` resolves its `phase` actions against `read` at the boundary
/// `t_end`. PURE — the caller fuses `out.int` into the draw and applies
/// `out.real` to the real reservoir. Replaces the historical int-only
/// `inject_event_deltas` (which silently dropped real-targeted events).
///
/// gh#217: events are resolved in two phases by [`EventPhase`]. `step_one`
/// passes `read = start-of-step snapshot, phase = Snapshot` for inflow `Add`
/// (fused with the kernel draw) and `read = post-transition residual, phase =
/// Residual` for draining transfers / `Set` (applied after the draw). Passing
/// the read-state and the matching phase together is the caller's contract.
///
/// `event_idx` (from [`due_effects`]) lists exactly the firing always-active
/// interventions in declaration order — this function does NOT re-check
/// `always_active` or `fire_steps`; it applies the list it was handed.
///
/// `dt` is `dt_actual` — the realized substep length driving the rate / amount
/// evaluation (the `EvalCtx.dt` the resolved intervention exprs see).
#[allow(clippy::too_many_arguments)]
pub fn resolve_event_batch(
    model: &CompiledModel,
    event_idx: &[usize],
    read_int: &IntState,
    read_real: &RealState,
    params: &[f64],
    t_end: f64,
    dt: f64,
    phase: EventPhase,
    out: &mut EffectDeltas,
) -> Result<(), SimError> {
    let snap = StateRef { int: read_int, real: read_real };
    for &iv_idx in event_idx {
        let iv = &model.model.interventions[iv_idx];
        resolve_intervention(model, iv_idx, iv, snap, params, t_end, dt, EffectKind::Event, Some(phase), out)?;
    }
    Ok(())
}

// ── Continuous (ODE) effect application ─────────────────────────────────────
//
// ODE holds its INTEGER compartments as f64 (`int_vals`) and integrates them
// with RK4. The discrete resolver above rounds the integer arena to i64; routing
// ODE through it would discard the fractional state the integrator accumulated
// at every effect boundary (the historical `to_states` round-trip). The
// continuous path below applies effects to the f64 vectors EXACTLY — same action
// structure, but no rounding and no `.floor()` — reading integer compartments as
// f64 via `int_float_override` (the same mechanism `eval_propensities` uses for
// ODE rates). The discrete backends keep the i64 resolver untouched.

/// Evaluate one action's amount expression against f64 compartment values (the
/// ODE read path: integer compartments via `int_float_override`).
fn eval_amount_f64(
    model: &CompiledModel,
    iv_idx: usize,
    action_idx: usize,
    int_f64: &[f64],
    real_f64: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
) -> f64 {
    let placeholder = IntState::new(model.int_local_to_global.len());
    let rs = RealState::from_vec(real_f64.to_vec());
    let ctx = EvalCtx {
        model, int_s: &placeholder, real_s: &rs, params, t, dt,
        projected: None, aux: None, int_float_override: Some(int_f64),
    };
    eval_resolved(&model.resolved.intervention_exprs[iv_idx][action_idx], &ctx)
}

/// Apply one action exactly in f64. `read_int`/`read_real` supply the values the
/// snapshot-relative arithmetic reads (the frozen snapshot for events; the live
/// state for sequential interventions); the result is written to
/// `int_vals`/`real_vals`.
#[allow(clippy::too_many_arguments)]
fn apply_action_f64(
    model: &CompiledModel,
    action: &Action,
    v: f64,
    read_int: &[f64],
    read_real: &[f64],
    int_vals: &mut [f64],
    real_vals: &mut [f64],
) -> Result<(), SimError> {
    match action {
        Action::Add(aa) => {
            // Guard the RAW value, not the rounded one (gh#199): `(-0.3).round()`
            // is `-0.0`, which is not `< 0.0`, so the rounded guard let a negative
            // add in (-0.5, 0) reach `*= v` — a silent subtraction from the
            // continuous reservoir each firing.
            if v < 0.0 {
                return Err(SimError::NegativeCount {
                    compartment: aa.compartment.clone(),
                    attempted_value: v.floor() as i64,
                    t: 0.0,
                    cause: NegativeCountCause::InterventionAddNegative,
                });
            }
            match resolve_target(model, &aa.compartment)? {
                Arena::Int(i) => int_vals[i] += v,
                Arena::Real(i) => real_vals[i] += v,
            }
        }
        Action::Set(sa) => match resolve_target(model, &sa.compartment)? {
            Arena::Int(i) => int_vals[i] = v,
            Arena::Real(i) => {
                // gh#196: reject a negative real `set` on the ODE path too,
                // matching the discrete resolver. Check the explicit value,
                // not the integrated state (RK4 undershoot is cleaned to 0).
                if v < 0.0 {
                    return Err(set_real_negative_err(&sa.compartment, v, 0.0));
                }
                real_vals[i] = v;
            }
        },
        Action::FractionTransfer(ft) => {
            let frac = v.clamp(0.0, 1.0);
            match (resolve_target(model, &ft.src)?, resolve_target(model, &ft.dst)?) {
                (Arena::Int(s), Arena::Int(d)) => {
                    let x = read_int[s] * frac;
                    int_vals[s] -= x;
                    int_vals[d] += x;
                }
                (Arena::Real(s), Arena::Real(d)) => {
                    let x = read_real[s] * frac;
                    real_vals[s] -= x;
                    real_vals[d] += x;
                }
                _ => return Err(mixed_arena_err(&ft.src, &ft.dst)),
            }
        }
        Action::AbsoluteTransfer(at) => {
            match (resolve_target(model, &at.src)?, resolve_target(model, &at.dst)?) {
                (Arena::Int(s), Arena::Int(d)) => {
                    let x = v.min(read_int[s]);
                    int_vals[s] -= x;
                    int_vals[d] += x;
                }
                (Arena::Real(s), Arena::Real(d)) => {
                    let x = v.min(read_real[s]);
                    real_vals[s] -= x;
                    real_vals[d] += x;
                }
                _ => return Err(mixed_arena_err(&at.src, &at.dst)),
            }
        }
    }
    Ok(())
}

/// Apply a known due batch to ODE's continuous compartment vectors, exactly (no
/// rounding). Order matches the discrete lifecycle: always-active EVENTS fire
/// first (`batch.event_idx`), every action reading the frozen pre-intervention
/// snapshot; then scheduled INTERVENTIONS (`batch.intervention_idx`) apply
/// sequentially on the post-event state. ODE carries no balance. `t_boundary` is
/// the effect time; `dt` drives the amount evaluation.
///
/// `batch` (from [`due_effects`]) carries the firing interventions in
/// declaration order per stage — this function does NOT re-derive due-ness; it
/// applies the lists it was handed (the duplication the scheduling-spine §B seam
/// removes from `effects.rs:382`).
pub fn apply_boundary_batch_continuous(
    model: &CompiledModel,
    batch: &crate::schedule::EffectBatch,
    int_vals: &mut [f64],
    real_vals: &mut [f64],
    params: &[f64],
    t_boundary: f64,
    dt: f64,
) -> Result<(), SimError> {
    let t = t_boundary;

    // EVENTS — frozen snapshot, every firing event's actions resolve against it.
    if !batch.event_idx.is_empty() {
        let snap_int = int_vals.to_vec();
        let snap_real = real_vals.to_vec();
        for &iv_idx in &batch.event_idx {
            let iv = &model.model.interventions[iv_idx];
            for (action_idx, action) in iv.actions.iter().enumerate() {
                let v = eval_amount_f64(model, iv_idx, action_idx, &snap_int, &snap_real, params, t, dt);
                let v = crate::intervention::finite_action_value(v, &iv.name, action, t)?;
                trace_action(EffectKind::Event, &iv.name, action, v, t);
                apply_action_f64(model, action, v, &snap_int, &snap_real, int_vals, real_vals)?;
            }
        }
    }

    // INTERVENTIONS — sequential, each action reads the live state.
    for &iv_idx in &batch.intervention_idx {
        let iv = &model.model.interventions[iv_idx];
        for (action_idx, action) in iv.actions.iter().enumerate() {
            let live_int = int_vals.to_vec();
            let live_real = real_vals.to_vec();
            let v = eval_amount_f64(model, iv_idx, action_idx, &live_int, &live_real, params, t, dt);
            let v = crate::intervention::finite_action_value(v, &iv.name, action, t)?;
            trace_action(EffectKind::Intervention, &iv.name, action, v, t);
            apply_action_f64(model, action, v, &live_int, &live_real, int_vals, real_vals)?;
        }
    }
    Ok(())
}

/// The pure arithmetic core: one action + its resolved `f64` value + the
/// snapshot → typed deltas. No model state is read except the snapshot and the
/// arena map. Mirrors the historical `apply_intervention` / `inject_event_deltas`
/// rounding exactly for the integer arena.
fn resolve_action(
    model: &CompiledModel,
    action: &Action,
    v: f64,
    snap: StateRef<'_>,
    t: f64,
    out: &mut EffectDeltas,
) -> Result<(), SimError> {
    match action {
        Action::Add(aa) => {
            // A negative add is always a config bug (you cannot add a negative
            // number of individuals) — hard error on every path. Guard the RAW
            // resolved value, not the rounded one: a raw amount in (-0.5, 0)
            // rounds to -0.0 (=> 0), so a rounded guard let it slip through as a
            // silent no-op on an int target or a silent subtraction on a real
            // one (gh#199).
            if v < 0.0 {
                return Err(SimError::NegativeCount {
                    compartment: aa.compartment.clone(),
                    attempted_value: v.floor() as i64,
                    t,
                    cause: NegativeCountCause::InterventionAddNegative,
                });
            }
            let count = v.round() as i64;
            match resolve_target(model, &aa.compartment)? {
                Arena::Int(i) => out.int.push(IntDelta { idx: i, delta: count }),
                Arena::Real(i) => out.real.push(RealDelta { idx: i, delta: v }),
            }
        }
        Action::Set(sa) => match resolve_target(model, &sa.compartment)? {
            Arena::Int(i) => out.int.push(IntDelta {
                idx: i,
                delta: (v.round() as i64) - snap.int.counts[i],
            }),
            Arena::Real(i) => {
                // gh#196: a `set` to a negative value is a config bug on the
                // real arena too. The integer arena catches this via the
                // post-INTERVENE scan; the real arena is not scanned (ODE's
                // tiny RK4 undershoot, cleaned by `.max(0.0)`, would
                // false-positive). Check the explicit set VALUE here instead.
                if v < 0.0 {
                    return Err(set_real_negative_err(&sa.compartment, v, t));
                }
                out.real.push(RealDelta {
                    idx: i,
                    delta: v - snap.real.values[i],
                })
            }
        },
        Action::FractionTransfer(ft) => {
            let frac = v.clamp(0.0, 1.0);
            match (resolve_target(model, &ft.src)?, resolve_target(model, &ft.dst)?) {
                (Arena::Int(s), Arena::Int(d)) => {
                    let x = ((snap.int.counts[s] as f64) * frac).floor() as i64;
                    out.int.push(IntDelta { idx: s, delta: -x });
                    out.int.push(IntDelta { idx: d, delta: x });
                }
                (Arena::Real(s), Arena::Real(d)) => {
                    let x = snap.real.values[s] * frac;
                    out.real.push(RealDelta { idx: s, delta: -x });
                    out.real.push(RealDelta { idx: d, delta: x });
                }
                _ => return Err(mixed_arena_err(&ft.src, &ft.dst)),
            }
        }
        Action::AbsoluteTransfer(at) => {
            match (resolve_target(model, &at.src)?, resolve_target(model, &at.dst)?) {
                (Arena::Int(s), Arena::Int(d)) => {
                    let x = (v.round() as i64).min(snap.int.counts[s]);
                    out.int.push(IntDelta { idx: s, delta: -x });
                    out.int.push(IntDelta { idx: d, delta: x });
                }
                (Arena::Real(s), Arena::Real(d)) => {
                    let x = v.min(snap.real.values[s]);
                    out.real.push(RealDelta { idx: s, delta: -x });
                    out.real.push(RealDelta { idx: d, delta: x });
                }
                _ => return Err(mixed_arena_err(&at.src, &at.dst)),
            }
        }
    }
    Ok(())
}

/// gh#196: a `set` driving a REAL compartment below zero. The integer arena
/// catches its negatives in the post-INTERVENE scan, but the real arena is not
/// scanned (ODE's `.max(0.0)` RK4-undershoot cleanup would false-positive a
/// state scan), so the check lives at the action site on the explicit value —
/// symmetric with the `add(<0)` guard. `attempted_value` is rounded for the
/// shared i64 diagnostic field; the message names the compartment and time.
fn set_real_negative_err(compartment: &str, v: f64, t: f64) -> SimError {
    SimError::NegativeCount {
        compartment: compartment.to_string(),
        attempted_value: v.round() as i64,
        t,
        cause: NegativeCountCause::InterventionNegative,
    }
}

/// A transfer whose endpoints land in different arenas (one integer, one real)
/// is not representable — error instead of the historical silent no-op.
fn mixed_arena_err(src: &str, dst: &str) -> SimError {
    SimError::Validation(format!(
        "transfer '{src}' -> '{dst}': source and destination must be the same \
         representation (both integer or both real compartments)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_model::CompiledModel;
    use ir::{
        expr::Expr,
        intervention::{
            AbsoluteTransfer, AddAction, FractionTransfer, Intervention, InterventionSchedule,
            SetAction,
        },
        model::{
            Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
            SimulationConfig,
        },
        parameter::Parameter,
        transition::{DrawMethod, StoichiometryEntry, Transition},
        Model,
    };
    use std::collections::HashMap;

    // S, I integer; W real. One trivial transition so the model compiles.
    fn model_with(actions: Vec<Action>) -> CompiledModel {
        let m = Model {
            name: "effects_test".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "W".into(), kind: CompartmentKind::Real },
            ],
            transitions: vec![Transition {
                name: "decay".into(),
                stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                rate: Expr::const_(0.0),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![Intervention {
                name: "iv".into(),
                base_name: None,
                fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
                actions,
                kind: ir::intervention::InterventionKind::Scenario,
            }],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: "p".into(), value: ir::parameter::ParamValue::Fixed { value: 1.0 }, param_kind: None, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("S".into(), 100.0);
                h.insert("I".into(), 0.0);
                h.insert("W".into(), 50.0);
                h
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    // S=100 (local int 0), I=0 (local int 1), W=50.0 (local real 0).
    fn snap<'a>(int_s: &'a IntState, real_s: &'a RealState) -> StateRef<'a> {
        StateRef { int: int_s, real: real_s }
    }

    fn states() -> (IntState, RealState) {
        let int_s = IntState::from_vec(vec![100, 0]);
        let real_s = RealState::from_vec(vec![50.0]);
        (int_s, real_s)
    }

    fn resolve(model: &CompiledModel) -> EffectDeltas {
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        resolve_intervention(model, 0, &model.model.interventions[0], snap(&int_s, &real_s),
                             &model.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap();
        out
    }

    #[test]
    fn add_int_rounds_and_emits_positive_delta() {
        let m = model_with(vec![Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(3.6) })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 1, delta: 4 }]); // round(3.6)=4 to I(local 1)
        assert!(d.real.is_empty());
    }

    #[test]
    fn add_real_is_exact_f64() {
        let m = model_with(vec![Action::Add(AddAction { compartment: "W".into(), count: Expr::const_(2.5) })]);
        let d = resolve(&m);
        assert_eq!(d.real, vec![RealDelta { idx: 0, delta: 2.5 }]); // exact, no round
        assert!(d.int.is_empty());
    }

    #[test]
    fn add_negative_is_hard_error_on_any_path() {
        let m = model_with(vec![Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(-1.0) })]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        assert!(matches!(err, SimError::NegativeCount { cause: NegativeCountCause::InterventionAddNegative, .. }));
    }

    /// gh#199: a negative add amount in the open interval (-0.5, 0) must hard-
    /// error, not slip past the guard. The historical guard tested the ROUNDED
    /// value (`v.round() as i64`), and `(-0.3).round()` is `-0.0` → `0`, so
    /// `count < 0` was false and the negative add was silently accepted (a
    /// no-op on an int target, a silent subtraction on a real target). The fix
    /// guards the RAW resolved value. `-0.3` is the canonical hole representative
    /// (the existing `add_negative_is_hard_error_on_any_path` uses `-1.0`, which
    /// rounds to `-1` and so never exercises this interval).
    #[test]
    fn add_negative_subhalf_int_is_hard_error() {
        let m = model_with(vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::const_(-0.3),
        })]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        assert!(
            matches!(err, SimError::NegativeCount { cause: NegativeCountCause::InterventionAddNegative, .. }),
            "add(int, -0.3) must hard-error like add(int, -1.0); got: {err}"
        );
        assert!(out.is_empty(), "no delta may be emitted when the add is rejected");
    }

    /// gh#199, real target: `add(W, -0.3)` on the discrete resolver. The real
    /// arm pushed `RealDelta { delta: v }` = -0.3 (a silent subtraction), since
    /// the rounded guard did not fire. The raw-value guard must reject it.
    #[test]
    fn add_negative_subhalf_real_is_hard_error() {
        let m = model_with(vec![Action::Add(AddAction {
            compartment: "W".into(),
            count: Expr::const_(-0.3),
        })]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        assert!(
            matches!(err, SimError::NegativeCount { cause: NegativeCountCause::InterventionAddNegative, .. }),
            "add(real, -0.3) must hard-error, not silently subtract; got: {err}"
        );
        assert!(out.is_empty(), "no delta may be emitted when the add is rejected");
    }

    /// gh#199, ODE path: `apply_action_f64`'s add arm tested `v.round() < 0.0`,
    /// and `(-0.3).round()` is `-0.0` (not `< 0.0`), so the negative add reached
    /// `real_vals[i] += v` = a silent subtraction of 0.3 from the reservoir each
    /// firing. Drive it through the continuous boundary batch on a real target
    /// and assert the hard error, with the reservoir left untouched.
    #[test]
    fn continuous_add_negative_subhalf_real_is_hard_error() {
        let m = model_with(vec![Action::Add(AddAction {
            compartment: "W".into(),
            count: Expr::const_(-0.3),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        let err = apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0)
            .unwrap_err();
        assert!(
            matches!(err, SimError::NegativeCount { cause: NegativeCountCause::InterventionAddNegative, .. }),
            "ODE add(real, -0.3) must hard-error, not silently subtract; got: {err}"
        );
        assert_eq!(real_vals[0], 50.0, "the reservoir must be untouched when the add is rejected");
    }

    /// gh#199 boundary: a positive add that ROUNDS to zero (`0.4` → `round` = 0)
    /// is a legitimate quantize-to-zero no-op, NOT a config bug — the raw-value
    /// guard must reject only `v < 0`, leaving `[0, 0.5)` accepted. Pins that the
    /// fix does not over-reject.
    #[test]
    fn add_positive_subhalf_int_is_zero_delta_not_error() {
        let m = model_with(vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::const_(0.4),
        })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 1, delta: 0 }], "round(0.4)=0 is a no-op, not an error");
    }

    #[test]
    fn set_int_emits_snapshot_relative_delta() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(70.4) })]);
        let d = resolve(&m);
        // round(70.4)=70, snapshot S=100 → delta -30 → S ends at 70.
        assert_eq!(d.int, vec![IntDelta { idx: 0, delta: 70 - 100 }]);
    }

    #[test]
    fn set_real_is_exact() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(12.5) })]);
        let d = resolve(&m);
        assert_eq!(d.real, vec![RealDelta { idx: 0, delta: 12.5 - 50.0 }]);
    }

    /// gh#196: a `set` driving a REAL compartment below zero is a config bug,
    /// symmetric with `set(int, <0)` (caught by the post-advance scan) and
    /// `add(<0)` (caught at the action site). The real arena's `set` had no
    /// negativity check, so `set(W, -5)` was silently accepted. The discrete
    /// resolver must now reject it at the action site.
    #[test]
    fn set_real_negative_is_hard_error() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(-5.0) })]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        match err {
            SimError::NegativeCount { compartment, attempted_value, cause, .. } => {
                assert_eq!(compartment, "W");
                assert_eq!(attempted_value, -5);
                assert_eq!(cause, NegativeCountCause::InterventionNegative);
            }
            other => panic!("expected NegativeCount{{InterventionNegative}}, got: {other}"),
        }
    }

    /// gh#196 (ODE path): the continuous `apply_action_f64` real-`Set` arm must
    /// reject a negative value the same way. `set(W, -5)` through the boundary
    /// batch errors instead of writing -5 into the reservoir.
    #[test]
    fn continuous_set_real_negative_is_hard_error() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(-5.0) })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        let err = apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0)
            .unwrap_err();
        assert!(
            matches!(err, SimError::NegativeCount { cause: NegativeCountCause::InterventionNegative, .. }),
            "ODE real `set` to a negative value must error; got: {err}"
        );
        assert_eq!(real_vals[0], 50.0, "the reservoir must be untouched when the set is rejected");
    }

    #[test]
    fn fraction_transfer_int_floors() {
        // 0.337 * 100 = 33.7 → floor 33.
        let m = model_with(vec![Action::FractionTransfer(FractionTransfer {
            src: "S".into(), dst: "I".into(), fraction: Expr::const_(0.337),
        })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 0, delta: -33 }, IntDelta { idx: 1, delta: 33 }]);
    }

    #[test]
    fn absolute_transfer_int_rounds_then_clamps_to_src() {
        // round(250.6)=251, clamped to src S=100.
        let m = model_with(vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "S".into(), dst: "I".into(), count: Expr::const_(250.6),
        })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 0, delta: -100 }, IntDelta { idx: 1, delta: 100 }]);
    }

    #[test]
    fn mixed_arena_transfer_errors() {
        let m = model_with(vec![Action::FractionTransfer(FractionTransfer {
            src: "S".into(), dst: "W".into(), fraction: Expr::const_(0.5),
        })]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        assert!(matches!(err, SimError::Validation(_)));
    }

    #[test]
    fn apply_effects_sums_in_order() {
        let mut int_s = IntState::from_vec(vec![100, 0]);
        let mut real_s = RealState::from_vec(vec![50.0]);
        let d = EffectDeltas {
            int: vec![IntDelta { idx: 0, delta: -30 }, IntDelta { idx: 1, delta: 30 }],
            real: vec![RealDelta { idx: 0, delta: 2.5 }],
        };
        apply_effects(&d, StateMut { int: &mut int_s, real: &mut real_s });
        assert_eq!(int_s.counts, vec![70, 30]);
        assert_eq!(real_s.values, vec![52.5]);
    }

    /// Test shim: compute the due batch at `t_boundary` (the seam `due_effects`
    /// now owns) and route it through `apply_boundary_batch_continuous` — the
    /// byte-identical replacement for the old `apply_boundary_effects_continuous`
    /// that re-derived due-ness internally.
    fn apply_boundary_effects_continuous(
        model: &CompiledModel,
        fire_steps: &[std::collections::BTreeSet<i64>],
        int_vals: &mut [f64],
        real_vals: &mut [f64],
        params: &[f64],
        t_boundary: f64,
        dt: f64,
    ) -> Result<(), SimError> {
        let mut batch = crate::schedule::EffectBatch::default();
        super::due_effects(model, fire_steps, t_boundary, dt, &mut batch);
        apply_boundary_batch_continuous(model, &batch, int_vals, real_vals, params, t_boundary, dt)
    }

    // ── Continuous (ODE) path — exact f64, no rounding ──────────────────────
    // These are the de-quantization oracle at the f64 level (before the output
    // contract rounds integer compartments for display): the continuous apply
    // must carry the fractional integrator state exactly.

    /// A fraction-transfer on a FRACTIONAL integer compartment moves the exact
    /// fraction — no `.floor()`, unlike the discrete path. S = 704.69, transfer
    /// 0.5 → both ends = 352.345.
    #[test]
    fn continuous_fraction_transfer_is_exact() {
        let m = model_with(vec![Action::FractionTransfer(FractionTransfer {
            src: "S".into(), dst: "I".into(), fraction: Expr::const_(0.5),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![704.69_f64, 0.0]; // S (local int 0), I (local int 1)
        let mut real_vals = vec![50.0_f64];        // W
        apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(int_vals[0], 704.69 - 352.345);
        assert_eq!(int_vals[1], 352.345);
        // Discrete would floor(704*0.5)=352 after rounding S to 704 — different.
    }

    /// A `set` on an integer compartment lands the exact f64 (no round).
    #[test]
    fn continuous_set_int_is_exact() {
        let m = model_with(vec![Action::Set(SetAction {
            compartment: "S".into(), value: Expr::const_(70.4),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(int_vals[0], 70.4); // exact, not round(70.4)=70
    }

    // ── Real-source transfers (the two L1 arithmetic holes) ─────────────────
    // The `model_with` helper has a single real compartment (W). A transfer
    // needs two same-arena endpoints, so these use a dedicated model with TWO
    // real reservoirs W1, W2 (no integer transfer endpoints) to pin the real
    // arms of `apply_action_f64`: AbsoluteTransfer = `v.min(src)` (no round),
    // FractionTransfer = `src*frac` (no floor). Both exercised through the same
    // scheduled-intervention path as the existing continuous tests.

    /// A small model with two REAL reservoirs W1, W2 (local real 0, 1) plus the
    /// integer S/I scaffold so it compiles. The single scheduled intervention
    /// carries `actions`, fired at t=1 by `apply_boundary_effects_continuous`.
    fn model_two_real(actions: Vec<Action>) -> CompiledModel {
        let m = Model {
            name: "effects_two_real".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "W1".into(), kind: CompartmentKind::Real },
                Compartment { name: "W2".into(), kind: CompartmentKind::Real },
            ],
            transitions: vec![Transition {
                name: "decay".into(),
                stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                rate: Expr::const_(0.0),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![Intervention {
                name: "iv".into(),
                base_name: None,
                fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
                actions,
                kind: ir::intervention::InterventionKind::Scenario,
            }],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: "p".into(), value: ir::parameter::ParamValue::Fixed { value: 1.0 }, param_kind: None, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("S".into(), 100.0);
                h.insert("I".into(), 0.0);
                h.insert("W1".into(), 10.0);
                h.insert("W2".into(), 0.0);
                h
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    /// AbsoluteTransfer on a REAL source: exact `v.min(src)`, NO round.
    /// W1=10.0, W2=0.0. Two interventions can't share a model here (the helper
    /// has one), so we run two separate models:
    ///   - transfer 3.5: x = min(3.5, 10.0) = 3.5 → W1=6.5, W2=3.5 (exact, no round)
    ///   - transfer 20.0: x = min(20.0, 10.0) = 10.0 (clamp) → W1=0.0, W2=10.0
    #[test]
    fn continuous_absolute_transfer_real_is_exact_and_clamps() {
        // Sub-source amount: moves exactly 3.5, no rounding of the .5.
        let m = model_two_real(vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "W1".into(), dst: "W2".into(), count: Expr::const_(3.5),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0]; // W1=10, W2=0
        apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(real_vals[0], 6.5, "W1 = 10.0 - 3.5"); // NOT round(3.5)
        assert_eq!(real_vals[1], 3.5, "W2 = 0.0 + 3.5");

        // Over-source amount: clamps to src = 10.0, drains W1 to exactly 0.
        let m = model_two_real(vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "W1".into(), dst: "W2".into(), count: Expr::const_(20.0),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0];
        apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(real_vals[0], 0.0, "W1 drained, clamped to src");
        assert_eq!(real_vals[1], 10.0, "W2 received the clamped 10.0, not 20.0");
    }

    /// FractionTransfer whose SOURCE is a real compartment: exact `src*frac`,
    /// NO floor. W1=10.0, frac=0.337 → x = 10.0*0.337 (the exact f64 product,
    /// ≈3.3699999999999997, NOT the discrete floor(3.37)=3). Assert against the
    /// same f64 product, not the decimal literal, so the test is bit-exact.
    #[test]
    fn continuous_fraction_transfer_real_source_is_exact() {
        let m = model_two_real(vec![Action::FractionTransfer(FractionTransfer {
            src: "W1".into(), dst: "W2".into(), fraction: Expr::const_(0.337),
        })]);
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0]; // W1=10, W2=0
        apply_boundary_effects_continuous(&m, &fire, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        let moved = 10.0_f64 * 0.337; // exact f64 product, no floor
        assert_eq!(real_vals[0], 10.0 - moved, "W1 = 10.0 - 10.0*0.337 (exact, no floor)");
        assert_eq!(real_vals[1], moved, "W2 = 10.0*0.337 (exact, no floor)");
        // Discriminating control: the discrete int path would floor(10*0.337)=3;
        // the real path moves ≈3.37, strictly more than 3.
        assert!(moved > 3.0 && moved < 3.4, "moved ≈3.37, not the floored 3: {moved}");
    }

    // ── due_effects: the centralized due-check ──────────────────────────────
    // One model with an always-active EVENT and a scheduled INTERVENTION, both
    // firing at t=1 (step 1 at dt=1). due_effects must route the event into
    // event_idx and the intervention into intervention_idx, in declaration
    // order, and produce an empty batch off-step.

    /// S/I/W scaffold with TWO interventions, declared in this order:
    ///   [0] `evt`  — always_active EVENT, fires at t=1
    ///   [1] `camp` — scheduled INTERVENTION, fires at t=1
    /// Both have a single trivial `set` action so the model compiles.
    fn model_event_and_intervention() -> CompiledModel {
        let mk = |name: &str, always: bool| Intervention {
            name: name.into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
            actions: vec![Action::Set(SetAction {
                compartment: "I".into(),
                value: Expr::const_(1.0),
            })],
            kind: if always { ir::intervention::InterventionKind::Event } else { ir::intervention::InterventionKind::Scenario },
        };
        let m = Model {
            name: "due_effects_test".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "W".into(), kind: CompartmentKind::Real },
            ],
            transitions: vec![Transition {
                name: "decay".into(),
                stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                rate: Expr::const_(0.0),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            // Declaration order: event first, then intervention.
            interventions: vec![mk("evt", true), mk("camp", false)],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: "p".into(), value: ir::parameter::ParamValue::Fixed { value: 1.0 }, param_kind: None, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("S".into(), 100.0);
                h.insert("I".into(), 0.0);
                h.insert("W".into(), 50.0);
                h
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 2.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    #[test]
    fn due_effects_splits_event_and_intervention_at_same_step() {
        let m = model_event_and_intervention();
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut batch = crate::schedule::EffectBatch::default();
        // t_end = 1.0, grid_dt = 1.0 → step 1: both the event (iv 0) and the
        // intervention (iv 1) fire.
        due_effects(&m, &fire, 1.0, 1.0, &mut batch);
        assert_eq!(batch.event_idx.as_slice(), &[0], "always_active event → event_idx");
        assert_eq!(batch.intervention_idx.as_slice(), &[1], "scheduled → intervention_idx");
    }

    #[test]
    fn split_due_batch_routes_by_kind_and_clears_on_reuse() {
        // The shared seam every Exact-inference caller routes through (gh#216
        // events arm): a flat due list — the timeline cursor's per-boundary batch
        // — splits into the lifecycle halves by KIND, events at PROPOSE
        // (event_idx) / scheduled interventions at INTERVENE (intervention_idx).
        // iv 0 is the always-active event, iv 1 the scheduled intervention.
        let m = model_event_and_intervention();
        let mut batch = crate::schedule::EffectBatch::default();

        split_due_batch(&m, &[0, 1], &mut batch);
        assert_eq!(batch.event_idx.as_slice(), &[0], "always-active event → event_idx");
        assert_eq!(batch.intervention_idx.as_slice(), &[1], "scheduled → intervention_idx");

        // Reused per substep on the hot path: a second call CLEARS first, so a
        // stale half can't leak into the next boundary.
        split_due_batch(&m, &[1], &mut batch);
        assert!(batch.event_idx.is_empty(), "clear() reset the event half");
        assert_eq!(batch.intervention_idx.as_slice(), &[1]);

        split_due_batch(&m, &[], &mut batch);
        assert!(batch.is_empty(), "empty due list → empty batch");
    }

    #[test]
    fn due_effects_empty_off_step() {
        let m = model_event_and_intervention();
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut batch = crate::schedule::EffectBatch::default();
        // t_end = 2.0 → step 2: nothing scheduled there.
        due_effects(&m, &fire, 2.0, 1.0, &mut batch);
        assert!(batch.is_empty(), "no effect due at step 2");
        assert!(batch.event_idx.is_empty() && batch.intervention_idx.is_empty());
    }

    #[test]
    fn due_effects_preserves_declaration_order() {
        // Two always-active events and two scheduled interventions, interleaved
        // in declaration order, all firing at the same step. Each stage's list
        // must come out in declaration order (the byte-identical firing order).
        let mk = |name: &str, always: bool| Intervention {
            name: name.into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
            actions: vec![Action::Set(SetAction {
                compartment: "I".into(),
                value: Expr::const_(1.0),
            })],
            kind: if always { ir::intervention::InterventionKind::Event } else { ir::intervention::InterventionKind::Scenario },
        };
        // Declaration order: evt_a(ev), camp_a(iv), evt_b(ev), camp_b(iv).
        let interventions = vec![
            mk("evt_a", true),
            mk("camp_a", false),
            mk("evt_b", true),
            mk("camp_b", false),
        ];
        let base0 = model_event_and_intervention();
        // Rebuild from a model carrying the 4-intervention list.
        let mut model = (*base0.model).clone();
        model.interventions = interventions;
        let base = CompiledModel::new(model).unwrap();
        let fire = base.resolve_fire_steps(1.0, &base.default_params);
        let mut batch = crate::schedule::EffectBatch::default();
        due_effects(&base, &fire, 1.0, 1.0, &mut batch);
        assert_eq!(batch.event_idx.as_slice(), &[0, 2], "events in declaration order");
        assert_eq!(batch.intervention_idx.as_slice(), &[1, 3], "interventions in declaration order");
    }

    #[test]
    fn due_effects_clears_the_reused_batch() {
        // A batch reused across substeps must not accumulate stale indices.
        let m = model_event_and_intervention();
        let fire = m.resolve_fire_steps(1.0, &m.default_params);
        let mut batch = crate::schedule::EffectBatch::default();
        due_effects(&m, &fire, 1.0, 1.0, &mut batch); // fills it
        assert!(!batch.is_empty());
        due_effects(&m, &fire, 2.0, 1.0, &mut batch); // off-step → must clear
        assert!(batch.is_empty(), "reused batch must be cleared on the off-step call");
    }
}
