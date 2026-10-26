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
///     against the start-of-step snapshot, like the transition draw, and summed
///     into the state with the transition deltas.
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
#[derive(Clone, Copy, PartialEq, Eq)]
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
        projected: None, aux: None, int_float_override: None, per_eval: None,
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

/// Split a flat list of due effect indices — one batch of the run's
/// [`EffectPlan`](crate::effect_plan::EffectPlan), declaration order — into the
/// lifecycle-stage halves of an [`EffectBatch`]: always-active EVENTS fire at
/// PROPOSE (`event_idx`, fused with the kernel draw against the start-of-step
/// snapshot); scheduled interventions fire at INTERVENE (`intervention_idx`,
/// applied on the post-advance state). Indices are into
/// `model.model.interventions`, declaration order preserved.
///
/// The ONE place the kind→stage routing lives, for every path that applies a
/// plan batch: the forward backends, the origin batch, the inference filters
/// and the PGAS producers.
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
/// `t_end`. PURE — [`DiscreteArena`] applies `out` with [`apply_effects`].
///
/// gh#217: events are resolved in two phases by [`EventPhase`], in the order
/// [`apply_event_phases`] fixes: `Snapshot` (inflow `Add`) against the
/// start-of-step snapshot, then `Residual` (draining transfers / `Set`) against
/// the post-inflow, post-transition state.
///
/// `event_idx` (the event half of a plan batch, via [`split_due_batch`]) lists
/// exactly the firing always-active interventions in declaration order — this
/// function does not re-check due-ness; it applies the list it was handed.
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

// ── The two event phases of language spec §13.10 ────────────────────────────
//
// At one effect instant the due events apply in two phases: the inflow phase
// (`add`) resolves against the start-of-step snapshot; the drain-and-`set`
// phase (`transfer`, `set`) resolves against the state after the inflows and,
// where the backend has one, after the transition draw between them. Within a
// phase every action reads the same state. [`apply_event_phases`] holds that
// order once; [`EventArena`] is the only thing that differs per representation
// (integer `i64` counts vs the ODE's `f64` vectors).

/// A compartment state the event phases of §13.10 read and write. Implemented
/// by [`DiscreteArena`] (the integer `IntState`/`RealState` pair, via
/// [`resolve_event_batch`] and [`apply_effects`]) and [`ContinuousArena`] (the
/// ODE's `f64` vectors, via `apply_action_f64`). The phase order is not the
/// arena's business: it lives in [`apply_event_phases`].
pub trait EventArena {
    /// Resolve the `phase` actions of every event in `event_idx` (declaration
    /// order) against this arena's current state, every action reading that
    /// same state, then apply them to it.
    #[allow(clippy::too_many_arguments)]
    fn apply_phase(
        &mut self,
        model: &CompiledModel,
        event_idx: &[usize],
        params: &[f64],
        t: f64,
        dt: f64,
        phase: EventPhase,
    ) -> Result<(), SimError>;
}

/// Refuse two or more `set` actions on one compartment among the events due at
/// one instant. In the drain-and-`set` phase every action contributes its
/// change against the same state, so two sets would give `v1 + v2 − x`, a
/// value neither of them names. A config error, raised on every backend.
fn refuse_coincident_sets(model: &CompiledModel, event_idx: &[usize], t: f64) -> Result<(), SimError> {
    let sets = || {
        event_idx.iter().flat_map(move |&iv_idx| {
            let iv = &model.model.interventions[iv_idx];
            iv.actions.iter().filter_map(move |a| match a {
                Action::Set(sa) => Some((sa.compartment.as_str(), iv.name.as_str())),
                _ => None,
            })
        })
    };
    // Fast path: the common batch carries at most one `set`. Finding that walks
    // the due events' actions once, without allocating; it is linear in the
    // batch, not constant.
    if sets().nth(1).is_none() {
        return Ok(());
    }
    let all: Vec<(&str, &str)> = sets().collect();
    for (i, &(comp, _)) in all.iter().enumerate() {
        if all[..i].iter().any(|&(c, _)| c == comp) {
            continue; // reported with its first occurrence
        }
        let mut names: Vec<&str> = all.iter().filter(|&&(c, _)| c == comp).map(|&(_, n)| n).collect();
        if names.len() < 2 {
            continue;
        }
        names.dedup();
        let names = names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
        return Err(SimError::Validation(format!(
            "more than one `set` on compartment '{comp}' at t={t}: events {names} each set it \
             at the same instant, and their combined effect is not defined — give the \
             sets different times or merge them into one"
        )));
    }
    Ok(())
}

/// The integer arena: chain-binomial's and Gillespie's state. Each phase
/// resolves into `deltas` (caller-owned scratch, cleared per phase, so the hot
/// path allocates nothing) and applies them with [`apply_effects`].
pub struct DiscreteArena<'a> {
    pub int: &'a mut IntState,
    pub real: &'a mut RealState,
    pub deltas: &'a mut EffectDeltas,
}

impl EventArena for DiscreteArena<'_> {
    fn apply_phase(
        &mut self,
        model: &CompiledModel,
        event_idx: &[usize],
        params: &[f64],
        t: f64,
        dt: f64,
        phase: EventPhase,
    ) -> Result<(), SimError> {
        self.deltas.clear();
        resolve_event_batch(model, event_idx, self.int, self.real, params, t, dt, phase, self.deltas)?;
        apply_effects(self.deltas, StateMut { int: self.int, real: self.real });
        Ok(())
    }
}

/// Apply the due events `event_idx` at instant `t` in the §13.10 order, on
/// every backend:
///
///   1. the inflow phase (`add`), resolved against the state on entry — the
///      start-of-step snapshot;
///   2. `advance`, when given — chain-binomial's transition deltas, drawn from
///      that same snapshot before this call; Gillespie and the ODE pass `None`,
///      having no transition stage at a boundary;
///   3. the drain-and-`set` phase (`transfer`, `set`), resolved against the
///      state the first two stages leave.
///
/// Scheduled interventions are not applied here; they follow in declaration
/// order on the result ([`crate::lifecycle::apply_post_advance`] for the
/// integer backends, [`apply_interventions_continuous`] for the ODE). With
/// `add(X, 10)` and `set X = 5` at one instant this gives X = 5 on every
/// backend whatever the declaration order, and a drain declared beside an
/// inflow moves a share of the post-inflow count (gh#985, gh#217).
///
/// `dt` is the backend's step for amount evaluation (`EvalCtx.dt`). The drain
/// phase is skipped, along with its resolve, when no event is due.
#[allow(clippy::too_many_arguments)]
pub fn apply_event_phases<A: EventArena>(
    model: &CompiledModel,
    event_idx: &[usize],
    arena: &mut A,
    params: &[f64],
    t: f64,
    dt: f64,
    advance: Option<&mut dyn FnMut(&mut A) -> Result<(), SimError>>,
) -> Result<(), SimError> {
    if !t.is_finite() {
        return Err(SimError::Validation(format!("apply_event_phases: non-finite t = {t}")));
    }
    refuse_coincident_sets(model, event_idx, t)?;
    arena.apply_phase(model, event_idx, params, t, dt, EventPhase::Snapshot)?;
    if let Some(advance) = advance {
        advance(arena)?;
    }
    if !event_idx.is_empty() {
        arena.apply_phase(model, event_idx, params, t, dt, EventPhase::Residual)?;
    }
    Ok(())
}

/// Apply the plan's origin batch to the starting state (D1): the effects at
/// `t_start`, before the first step and before the `t_start` row is recorded or
/// scored. It is §13.10 without the transition stage — [`apply_event_phases`]
/// with no advance hook (inflow events from the starting state, then drains and
/// `set` on the post-inflow state), then `interventions` with the scheduled
/// interventions of the batch, the backend's stage-4 tail
/// ([`apply_origin_batch_discrete`] for the integer backends,
/// [`apply_interventions_continuous`] for the ODE).
///
/// A no-op when the batch is empty: the tail is never reached, so a
/// `balance {}` model's `t_start` row is not rewritten by a balance pass it
/// never had (`lifecycle::apply_post_advance` applies balance whether or not
/// its batch is empty). `dt` is the backend's step, for amount evaluation;
/// amounts see `t = t_start`.
pub fn apply_origin_batch<A: EventArena>(
    model: &CompiledModel,
    plan: &crate::effect_plan::EffectPlan,
    arena: &mut A,
    params: &[f64],
    t_start: f64,
    dt: f64,
    interventions: impl FnOnce(&mut A, &[usize]) -> Result<(), SimError>,
) -> Result<(), SimError> {
    if plan.at_origin().is_empty() {
        return Ok(());
    }
    let mut batch = crate::schedule::EffectBatch::default();
    split_due_batch(model, plan.at_origin(), &mut batch);
    apply_event_phases(model, &batch.event_idx, arena, params, t_start, dt, None)?;
    interventions(arena, &batch.intervention_idx)
}

/// [`apply_origin_batch`] on the integer state of chain-binomial, Gillespie and
/// every inference particle: the stage-4 tail is
/// [`crate::lifecycle::apply_post_advance`] (interventions, `balance`, the
/// non-negativity check). `balance` is the model's on the chain-binomial
/// paths, `None` on Gillespie (which refuses `balance {}` models).
#[allow(clippy::too_many_arguments)]
pub fn apply_origin_batch_discrete(
    model: &CompiledModel,
    plan: &crate::effect_plan::EffectPlan,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
    t_start: f64,
    dt: f64,
    balance: Option<&crate::compiled_model::ResolvedBalance>,
) -> Result<(), SimError> {
    let mut deltas = EffectDeltas::default();
    let mut arena = DiscreteArena { int: int_s, real: real_s, deltas: &mut deltas };
    apply_origin_batch(model, plan, &mut arena, params, t_start, dt, |a, ivs| {
        crate::lifecycle::apply_post_advance(model, ivs, a.int, a.real, params, t_start, dt, balance)
    })
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
        // gh#272: intervention action exprs are never rewritten by LICM, so no
        // PerEvalRef appears here — the prologue is irrelevant.
        per_eval: None,
    };
    eval_resolved(&model.resolved.intervention_exprs[iv_idx][action_idx], &ctx)
}

/// One change an action makes to one compartment of the ODE's `f64` state:
/// add a delta, or write a value (a `set`).
#[derive(Clone, Copy)]
enum ChangeF64 {
    Delta(f64),
    Write(f64),
}

/// The changes one action makes, exactly in f64, given its resolved value `v`.
/// `read_int`/`read_real` supply the values the snapshot-relative arithmetic
/// reads (the phase snapshot for events; the live state for sequential
/// interventions). Appends `(compartment, change)` pairs to `out`.
fn action_changes_f64(
    model: &CompiledModel,
    action: &Action,
    v: f64,
    read_int: &[f64],
    read_real: &[f64],
    out: &mut Vec<(Arena, ChangeF64)>,
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
            out.push((resolve_target(model, &aa.compartment)?, ChangeF64::Delta(v)));
        }
        Action::Set(sa) => {
            let target = resolve_target(model, &sa.compartment)?;
            // gh#196: reject a negative real `set` on the ODE path too,
            // matching the discrete resolver. Check the explicit value,
            // not the integrated state (RK4 undershoot is cleaned to 0).
            if matches!(target, Arena::Real(_)) && v < 0.0 {
                return Err(set_real_negative_err(&sa.compartment, v, 0.0));
            }
            out.push((target, ChangeF64::Write(v)));
        }
        Action::FractionTransfer(ft) => {
            let frac = v.clamp(0.0, 1.0);
            let (s, d) = (resolve_target(model, &ft.src)?, resolve_target(model, &ft.dst)?);
            let x = match s {
                Arena::Int(i) if matches!(d, Arena::Int(_)) => read_int[i] * frac,
                Arena::Real(i) if matches!(d, Arena::Real(_)) => read_real[i] * frac,
                _ => return Err(mixed_arena_err(&ft.src, &ft.dst)),
            };
            out.push((s, ChangeF64::Delta(-x)));
            out.push((d, ChangeF64::Delta(x)));
        }
        Action::AbsoluteTransfer(at) => {
            let (s, d) = (resolve_target(model, &at.src)?, resolve_target(model, &at.dst)?);
            let x = match s {
                Arena::Int(i) if matches!(d, Arena::Int(_)) => v.min(read_int[i]),
                Arena::Real(i) if matches!(d, Arena::Real(_)) => v.min(read_real[i]),
                _ => return Err(mixed_arena_err(&at.src, &at.dst)),
            };
            out.push((s, ChangeF64::Delta(-x)));
            out.push((d, ChangeF64::Delta(x)));
        }
    }
    Ok(())
}

/// The `f64` slot a compartment occupies.
fn slot<'s>(int_vals: &'s mut [f64], real_vals: &'s mut [f64], a: Arena) -> &'s mut f64 {
    match a {
        Arena::Int(i) => &mut int_vals[i],
        Arena::Real(i) => &mut real_vals[i],
    }
}

/// Apply one action exactly in f64, its changes in the order the action
/// lists them (the sequential intervention pass).
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
    let mut changes = Vec::with_capacity(2);
    action_changes_f64(model, action, v, read_int, read_real, &mut changes)?;
    for (a, c) in changes {
        let x = slot(int_vals, real_vals, a);
        match c {
            ChangeF64::Delta(d) => *x += d,
            ChangeF64::Write(w) => *x = w,
        }
    }
    Ok(())
}

/// Rounding slack for the ODE's negative-count refusal, as a multiple of the
/// first-order rounding bound of the sum it guards.
///
/// A compartment's value after an event phase is the f64 sum of its snapshot
/// `base` and its `n` changes. Recursive summation of those `n + 1` terms is
/// within `n·u·S` of the exact sum, where `u = f64::EPSILON / 2` is the unit
/// roundoff and `S = |base| + Σ|change|` (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, 2nd ed., 2002, ch. 4, the recursive-summation
/// bound). Each change also carries one rounding of its own (`x·frac`,
/// `v − base`), at most `u·|change|`, which adds at most `u·S`. So a sum whose
/// exact value is ≥ 0 cannot fall below `−(n + 1)·u·S`; drains that remove
/// exactly the snapshot (fractions summing to one) land inside that band.
/// `K = 2` doubles the first-order bound to cover its second-order terms, and
/// keeps the band far below one person at any realistic scale: two changes
/// at X = 1e9 give a slack of about 1.3e-6, so an overdraw by 0.5 is refused.
const ROUNDING_SLACK_K: f64 = 2.0;

/// The most negative value the ODE's event phase lets through as rounding
/// for a compartment with snapshot `base`, `n` changes, and
/// `sum_abs = |base| + Σ|change|`.
fn rounding_floor(n: usize, sum_abs: f64) -> f64 {
    -ROUNDING_SLACK_K * (n as f64 + 1.0) * (f64::EPSILON / 2.0) * sum_abs
}

/// The ODE's arena: integer compartments held as `f64` (`int_vals`) and the
/// real compartments, both updated exactly, with no rounding. Each phase
/// copies the state once and reads every amount and every transfer source from
/// that copy, so the actions of one phase see the same state.
///
/// Within a phase the actions combine as on the integer arena: each one
/// contributes its change against the phase's snapshot and the changes add, a
/// `set` contributing `v − snapshot`. So `set X = 5` with
/// `transfer(fraction = 0.5, from = X, to = Y)` from X = 10 gives
/// X = 10 + (5 − 10) − 5 = 0 whatever the declaration order. Two rules keep
/// that exact in `f64`:
///
///   - a compartment touched by one action only takes that action's change
///     directly, and a `set` alone on its compartment writes `v` exactly,
///     because `x + (v − x)` need not round to `v`;
///   - a compartment touched by several actions gets the snapshot plus their
///     changes added in ascending order of value, so the result is the same
///     to the last bit in any declaration order.
///
/// Two `set`s on one compartment never reach here ([`apply_event_phases`]
/// refuses them). An integer-kind compartment the phase leaves below zero,
/// beyond the rounding of its own sum ([`ROUNDING_SLACK_K`]), is refused with
/// `NegativeCountCause::InterventionNegative`, as Gillespie's post-effect scan
/// refuses it, instead of being integrated from a negative count.
pub struct ContinuousArena<'a> {
    pub int: &'a mut [f64],
    pub real: &'a mut [f64],
}

impl EventArena for ContinuousArena<'_> {
    fn apply_phase(
        &mut self,
        model: &CompiledModel,
        event_idx: &[usize],
        params: &[f64],
        t: f64,
        dt: f64,
        phase: EventPhase,
    ) -> Result<(), SimError> {
        if event_idx.is_empty() {
            return Ok(());
        }
        let snap_int = self.int.to_vec();
        let snap_real = self.real.to_vec();
        let mut changes: Vec<(Arena, ChangeF64)> = Vec::new();
        for &iv_idx in event_idx {
            let iv = &model.model.interventions[iv_idx];
            for (action_idx, action) in iv.actions.iter().enumerate() {
                if action_phase(action) != phase {
                    continue;
                }
                let v = eval_amount_f64(model, iv_idx, action_idx, &snap_int, &snap_real, params, t, dt);
                let v = crate::intervention::finite_action_value(v, &iv.name, action, t)?;
                trace_action(EffectKind::Event, &iv.name, action, v, t);
                action_changes_f64(model, action, v, &snap_int, &snap_real, &mut changes)?;
            }
        }
        // Group the changes by compartment, in first-appearance order, with a
        // per-arena index map (linear in the number of changes).
        let mut group_int = vec![usize::MAX; snap_int.len()];
        let mut group_real = vec![usize::MAX; snap_real.len()];
        let mut groups: Vec<(Arena, Vec<ChangeF64>)> = Vec::new();
        for &(target, change) in &changes {
            let g = match target {
                Arena::Int(i) => &mut group_int[i],
                Arena::Real(i) => &mut group_real[i],
            };
            if *g == usize::MAX {
                *g = groups.len();
                groups.push((target, Vec::with_capacity(1)));
            }
            groups[*g].1.push(change);
        }
        let mut sorted: Vec<f64> = Vec::new();
        for (target, cs) in &groups {
            let base = match *target {
                Arena::Int(i) => snap_int[i],
                Arena::Real(i) => snap_real[i],
            };
            let as_delta = |c: &ChangeF64| match *c {
                ChangeF64::Delta(d) => d,
                ChangeF64::Write(w) => w - base,
            };
            let x = slot(self.int, self.real, *target);
            if let [change] = cs.as_slice() {
                match *change {
                    ChangeF64::Delta(d) => *x += d,
                    ChangeF64::Write(w) => *x = w,
                }
            } else {
                sorted.clear();
                sorted.extend(cs.iter().map(as_delta));
                sorted.sort_by(|a, b| a.total_cmp(b));
                *x = sorted.iter().fold(base, |acc, &d| acc + d);
            }
            if let Arena::Int(i) = *target {
                let v = self.int[i];
                let sum_abs = base.abs() + cs.iter().map(|c| as_delta(c).abs()).sum::<f64>();
                if v < rounding_floor(cs.len(), sum_abs) {
                    return Err(SimError::NegativeCount {
                        compartment: model.int_compartment_name(i),
                        attempted_value: v.floor() as i64,
                        t,
                        cause: NegativeCountCause::InterventionNegative,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Apply the due scheduled interventions `intervention_idx` to the ODE's
/// continuous compartment vectors, exactly (no rounding), in declaration order
/// and sequentially: each action reads the live state the previous one left.
/// This is the stage after [`apply_event_phases`] at an ODE boundary; ODE
/// carries no balance. `t_boundary` is the effect time; `dt` drives the amount
/// evaluation.
pub fn apply_interventions_continuous(
    model: &CompiledModel,
    intervention_idx: &[usize],
    int_vals: &mut [f64],
    real_vals: &mut [f64],
    params: &[f64],
    t_boundary: f64,
    dt: f64,
) -> Result<(), SimError> {
    let t = t_boundary;
    for &iv_idx in intervention_idx {
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

    // S, I integer; W real. One trivial transition so the model compiles.
    fn model_with(actions: Vec<Action>) -> CompiledModel {
        let m = Model {
            ic_grad: Default::default(),
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
                rate_state_grad: Default::default(),
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
            initial_conditions: InitialConditions::constants([
                ("S".into(), 100.0),
                ("I".into(), 0.0),
                ("W".into(), 50.0)
            ]),
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
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
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

    // ── apply_event_phases: the §13.10 order over both arenas (gh#985) ───────
    // One effect whose actions are declared drain/`set` first and inflow last,
    // so a pass in declaration order would get each case wrong. S=100, I=0.

    fn set_then_add() -> CompiledModel {
        model_with(vec![
            Action::Set(SetAction { compartment: "I".into(), value: Expr::const_(5.0) }),
            Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0) }),
        ])
    }

    fn drain_then_add() -> CompiledModel {
        model_with(vec![
            Action::FractionTransfer(FractionTransfer {
                src: "I".into(),
                dst: "S".into(),
                fraction: Expr::const_(0.5),
            }),
            Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0) }),
        ])
    }

    fn discrete_phases(m: &CompiledModel) -> Vec<i64> {
        let (mut int_s, mut real_s) = states();
        let mut deltas = EffectDeltas::default();
        {
            let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
            apply_event_phases(m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap();
        }
        int_s.counts
    }

    fn continuous_phases(m: &CompiledModel) -> Vec<f64> {
        let mut int_vals = vec![100.0, 0.0];
        let mut real_vals = vec![50.0];
        {
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals };
            apply_event_phases(m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap();
        }
        int_vals
    }

    #[test]
    fn event_phases_set_reads_post_inflow_state_on_both_arenas() {
        let m = set_then_add();
        // Inflow first (I = 10), then `set` overwrites: I = 5. Summing both
        // deltas against I = 0 would give 15.
        assert_eq!(discrete_phases(&m), vec![100, 5]);
        assert_eq!(continuous_phases(&m), vec![100.0, 5.0]);
    }

    #[test]
    fn event_phases_drain_reads_post_inflow_state_on_both_arenas() {
        let m = drain_then_add();
        // Inflow first (I = 10), then the drain moves floor(10 · 0.5) = 5 to S.
        assert_eq!(discrete_phases(&m), vec![105, 5]);
        assert_eq!(continuous_phases(&m), vec![105.0, 5.0]);
    }

    fn continuous_phases_from(m: &CompiledModel, mut int_vals: Vec<f64>) -> Vec<f64> {
        let mut real_vals = vec![50.0];
        {
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals };
            apply_event_phases(m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap();
        }
        int_vals
    }

    #[test]
    fn continuous_set_and_drain_on_one_compartment_sum_their_deltas() {
        // From S = 10: `set S = 5` contributes 5 − 10, the drain −10 · 0.5, so
        // S = 0 and I = 5 in both declaration orders — the integer arena's rule.
        let set = Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(5.0) });
        let drain = Action::FractionTransfer(FractionTransfer {
            src: "S".into(),
            dst: "I".into(),
            fraction: Expr::const_(0.5),
        });
        for actions in [vec![set.clone(), drain.clone()], vec![drain, set]] {
            let m = model_with(actions);
            assert_eq!(continuous_phases_from(&m, vec![10.0, 0.0]), vec![0.0, 5.0]);
        }
    }

    #[test]
    fn continuous_lone_negative_int_set_is_refused() {
        // A lone `set` to −1 on an integer compartment used to be integrated
        // from −1 and printed as 0; it is now refused as Gillespie refuses it.
        let m = model_with(vec![Action::Set(SetAction {
            compartment: "S".into(),
            value: Expr::const_(-1.0),
        })]);
        let mut int_vals = vec![100.0, 0.0];
        let mut real_vals = vec![50.0];
        let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals };
        let res = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None);
        assert!(
            matches!(
                res,
                Err(SimError::NegativeCount { cause: NegativeCountCause::InterventionNegative, .. })
            ),
            "got {res:?}"
        );
    }

    #[test]
    fn continuous_lone_set_writes_its_value_exactly() {
        // 1e16 + (1 − 1e16) rounds to 0 in f64, so the summed-delta form would
        // give S = 0 here; a `set` alone on its compartment must give exactly 1,
        // even with another action on a different compartment in the phase.
        let m = model_with(vec![
            Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(1.0) }),
            Action::AbsoluteTransfer(AbsoluteTransfer {
                src: "I".into(),
                dst: "I".into(),
                count: Expr::const_(0.0),
            }),
        ]);
        let out = continuous_phases_from(&m, vec![1e16, 4.0]);
        assert_eq!(out[0].to_bits(), 1.0f64.to_bits(), "got S = {}", out[0]);
    }

    #[test]
    fn event_phases_advance_runs_between_inflow_and_drain() {
        let m = drain_then_add();
        // Snapshot I = 0 → inflow I = 10 → advance adds 6 (a transition's
        // deltas) → I = 16 → the drain moves floor(16 · 0.5) = 8 to S.
        let (mut int_s, mut real_s) = states();
        let mut deltas = EffectDeltas::default();
        let mut seen_i = None;
        {
            let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
            let mut advance = |a: &mut DiscreteArena<'_>| {
                seen_i = Some(a.int.counts[1]);
                a.int.counts[1] += 6;
                Ok(())
            };
            apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, Some(&mut advance))
                .unwrap();
        }
        assert_eq!(int_s.counts, vec![108, 8]);
        assert_eq!(seen_i, Some(10), "the advance must see the post-inflow state");
    }

    #[test]
    fn event_phases_with_no_due_event_still_advance() {
        let m = drain_then_add();
        let (mut int_s, mut real_s) = states();
        let mut deltas = EffectDeltas::default();
        {
            let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
            let mut advance = |a: &mut DiscreteArena<'_>| {
                a.int.counts[0] -= 1;
                Ok(())
            };
            apply_event_phases(&m, &[], &mut arena, &m.default_params, 1.0, 1.0, Some(&mut advance))
                .unwrap();
        }
        assert_eq!(int_s.counts, vec![99, 0]);
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
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        let err = apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0)
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
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        let err = apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0)
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
    fn multi_action_intervention_applies_every_action() {
        // gh#460: a bare stratified `transfer` lowers to ONE intervention carrying
        // one FractionTransfer per cell. Every committed golden and every other
        // test in this module has a single-action intervention, so nothing
        // exercised the loop that applies the rest — a regression that dropped
        // all but the first action would produce a plausible, silently wrong
        // trajectory. Two disjoint transfers out of one intervention:
        //   S(100) --0.25--> I   and   W(50.0) --0.40--> (real arena)
        // are independent, so the deltas must both appear.
        let m = model_with(vec![
            Action::FractionTransfer(FractionTransfer {
                src: "S".into(), dst: "I".into(), fraction: Expr::const_(0.25),
            }),
            Action::Add(AddAction {
                compartment: "I".into(), count: Expr::const_(7.0),
            }),
        ]);
        let d = resolve(&m);
        // 0.25 * 100 = 25 out of S into I, then +7 more into I.
        assert_eq!(
            d.int,
            vec![
                IntDelta { idx: 0, delta: -25 },
                IntDelta { idx: 1, delta: 25 },
                IntDelta { idx: 1, delta: 7 },
            ],
            "every action in a multi-action intervention must contribute a delta"
        );
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

    /// Test shim: the plan batch due at `t_boundary` (an Exact plan, as on the
    /// ODE), applied as the ODE boundary does — [`apply_event_phases`] on a
    /// [`ContinuousArena`], then [`apply_interventions_continuous`].
    fn apply_boundary_effects_continuous(
        model: &CompiledModel,
        int_vals: &mut [f64],
        real_vals: &mut [f64],
        params: &[f64],
        t_boundary: f64,
        dt: f64,
    ) -> Result<(), SimError> {
        let plan = crate::effect_plan::EffectPlan::build(
            model, params, 0.0, t_boundary,
            crate::effect_plan::FireClock::Exact, crate::effect_plan::Origin::Init,
        )?;
        let idx = plan.times().iter()
            .position(|&t| (t - t_boundary).abs() <= crate::schedule::EFFECT_EPS)
            .expect("an effect is due at t_boundary");
        let mut batch = crate::schedule::EffectBatch::default();
        split_due_batch(model, plan.batch(idx), &mut batch);
        apply_event_phases(
            model, &batch.event_idx,
            &mut ContinuousArena { int: int_vals, real: real_vals },
            params, t_boundary, dt, None,
        )?;
        apply_interventions_continuous(
            model, &batch.intervention_idx, int_vals, real_vals, params, t_boundary, dt,
        )
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
        let mut int_vals = vec![704.69_f64, 0.0]; // S (local int 0), I (local int 1)
        let mut real_vals = vec![50.0_f64];        // W
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
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
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
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
            ic_grad: Default::default(),
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
                rate_state_grad: Default::default(),
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
            initial_conditions: InitialConditions::constants([
                ("S".into(), 100.0),
                ("I".into(), 0.0),
                ("W1".into(), 10.0),
                ("W2".into(), 0.0)
            ]),
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
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    /// AbsoluteTransfer on a REAL source: exact `v.min(src)`, NO round.
    /// W1=10.0, W2=0.0. Two interventions can't share a model here (the helper
    /// has one), so we run two separate models:
    ///   - transfer 3.5: x = min(3.5, 10.0) = 3.5 → W1=6.5, W2=3.5 (exact, no round)
    ///   - transfer 20.0: x = min(20.0, 10.0) = 10.0 (clamp) → W1=0.0, W2=10.0
    fn real_phase(actions: Vec<Action>) -> Result<Vec<f64>, SimError> {
        let m = model_two_real(actions);
        let mut int_vals = vec![100.0, 0.0];
        let mut real_vals = vec![10.0 / 3.0, 0.0];
        {
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals };
            apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None)?;
        }
        Ok(real_vals)
    }

    /// The real-compartment delta path: a `set` and two drains on W1 at one
    /// instant. With W1 = 10/3 the changes (0.7 − 10/3, −0.11·10/3, −0.13)
    /// added in declaration order and in reverse order differ in the last bits
    /// (0.20333333333333303 vs 0.20333333333333314), so the result is only
    /// order-independent because the changes are added in ascending order.
    #[test]
    fn continuous_real_set_and_drains_are_bit_identical_in_either_order() {
        let set = Action::Set(SetAction { compartment: "W1".into(), value: Expr::const_(0.7) });
        let frac = Action::FractionTransfer(FractionTransfer {
            src: "W1".into(),
            dst: "W2".into(),
            fraction: Expr::const_(0.11),
        });
        let count = Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "W1".into(),
            dst: "W2".into(),
            count: Expr::const_(0.13),
        });
        let fwd = real_phase(vec![set.clone(), frac.clone(), count.clone()]).unwrap();
        let rev = real_phase(vec![count, frac, set]).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&fwd), bits(&rev), "forward {fwd:?} vs reversed {rev:?}");
        let base = 10.0_f64 / 3.0;
        let mut d = [0.7 - base, -(base * 0.11), -0.13];
        d.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(fwd[0].to_bits(), d.iter().fold(base, |acc, &x| acc + x).to_bits());
    }

    /// The negative check on a coincident real `set` still fires.
    #[test]
    fn continuous_coincident_real_set_negative_is_hard_error() {
        let res = real_phase(vec![
            Action::Set(SetAction { compartment: "W1".into(), value: Expr::const_(-1.0) }),
            Action::FractionTransfer(FractionTransfer {
                src: "W1".into(),
                dst: "W2".into(),
                fraction: Expr::const_(0.5),
            }),
        ]);
        assert!(
            matches!(
                res,
                Err(SimError::NegativeCount { cause: NegativeCountCause::InterventionNegative, .. })
            ),
            "got {res:?}"
        );
    }

    #[test]
    fn continuous_absolute_transfer_real_is_exact_and_clamps() {
        // Sub-source amount: moves exactly 3.5, no rounding of the .5.
        let m = model_two_real(vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "W1".into(), dst: "W2".into(), count: Expr::const_(3.5),
        })]);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0]; // W1=10, W2=0
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(real_vals[0], 6.5, "W1 = 10.0 - 3.5"); // NOT round(3.5)
        assert_eq!(real_vals[1], 3.5, "W2 = 0.0 + 3.5");

        // Over-source amount: clamps to src = 10.0, drains W1 to exactly 0.
        let m = model_two_real(vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "W1".into(), dst: "W2".into(), count: Expr::const_(20.0),
        })]);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0];
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
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
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![10.0_f64, 0.0]; // W1=10, W2=0
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        let moved = 10.0_f64 * 0.337; // exact f64 product, no floor
        assert_eq!(real_vals[0], 10.0 - moved, "W1 = 10.0 - 10.0*0.337 (exact, no floor)");
        assert_eq!(real_vals[1], moved, "W2 = 10.0*0.337 (exact, no floor)");
        // Discriminating control: the discrete int path would floor(10*0.337)=3;
        // the real path moves ≈3.37, strictly more than 3.
        assert!(moved > 3.0 && moved < 3.4, "moved ≈3.37, not the floored 3: {moved}");
    }

    // ── split_due_batch: the kind→stage routing ─────────────────────────────
    // One model with an always-active EVENT and a scheduled INTERVENTION, both
    // firing at t=1.

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
            ic_grad: Default::default(),
            name: "split_due_batch_test".into(),
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
                rate_state_grad: Default::default(),
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
            initial_conditions: InitialConditions::constants([
                ("S".into(), 100.0),
                ("I".into(), 0.0),
                ("W".into(), 50.0)
            ]),
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
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        CompiledModel::new(m).unwrap()
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
    fn a_plan_batch_splits_in_declaration_order() {
        // Two always-active events and two scheduled interventions, interleaved
        // in declaration order, all firing at the same instant. The plan's batch
        // and each stage's list must come out in declaration order.
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
        let plan = crate::effect_plan::EffectPlan::build(
            &base, &base.default_params, 0.0, 2.0,
            crate::effect_plan::FireClock::Grid { dt: 1.0 }, crate::effect_plan::Origin::Init,
        ).unwrap();
        assert_eq!(plan.batch_ending_substep(0), &[0, 1, 2, 3]);
        let mut batch = crate::schedule::EffectBatch::default();
        split_due_batch(&base, plan.batch_ending_substep(0), &mut batch);
        assert_eq!(batch.event_idx.as_slice(), &[0, 2], "events in declaration order");
        assert_eq!(batch.intervention_idx.as_slice(), &[1, 3], "interventions in declaration order");
    }

}
