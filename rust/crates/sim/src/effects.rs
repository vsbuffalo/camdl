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
use ir::intervention::{Action, GroupedAmount, GroupedTransfer, Intervention};

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
///     (`FractionTransfer`, `AbsoluteTransfer`, `GroupedTransfer`, `Set`):
///     resolved against the
///     POST-TRANSITION residual state, so a draining transfer moves a fraction
///     of what survived the interval — matching ODE / Gillespie.
///
/// Action variant → phase is a pure classification; see [`action_phase`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EventPhase {
    /// Inflow `Add` only, resolved against the start-of-step snapshot.
    Snapshot,
    /// `FractionTransfer` / `AbsoluteTransfer` / `GroupedTransfer` / `Set`,
    /// resolved against the post-transition residual.
    Residual,
}

/// Classify one event action by the read-state it must resolve against (gh#217).
///
/// `Add` is an INFLOW construct (cannot over-draw) → [`EventPhase::Snapshot`].
/// `FractionTransfer` / `AbsoluteTransfer` / `GroupedTransfer` drain their
/// `from` side and `Set` overwrites the post-dynamics value →
/// [`EventPhase::Residual`].
///
/// NOTE (follow-up, out of scope for gh#217): a NEGATIVE-amount `Add` used as a
/// drain is classified `Snapshot` here like any other `Add`. Such an `Add` is
/// rejected today as a config bug (`InterventionAddNegative`), so it cannot
/// silently over-draw; if negative `Add`-as-drain is ever supported it must move
/// to `Residual`.
fn action_phase(action: &Action) -> EventPhase {
    match action {
        Action::Add(_) => EventPhase::Snapshot,
        Action::FractionTransfer(_)
        | Action::AbsoluteTransfer(_)
        | Action::GroupedTransfer(_)
        | Action::Set(_) => EventPhase::Residual,
    }
}

impl EffectKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            EffectKind::Intervention => "INTERVENTION",
            EffectKind::Event => "EVENT",
        }
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

// ── Grouped transfers ───────────────────────────────────────────────────────
//
// A grouped transfer moves individuals out of the stages of one cell of a
// staged compartment (proposal 2026-11-17-unified-staged-residence, "The
// grouped transfer action"). The stages are one population, so its amount
// applies to their total `N = Σ n_i`:
//
//   integer cells, fraction f (clamped to [0, 1])   M = floor(N · f)
//   integer cells, count c                          M = min(round(c), N)
//   then stage i gives up its share of M ([`allocate_by_occupancy`])
//
//   real cells (and the ODE's f64 state), fraction  x_i = n_i · f
//   real cells (and the ODE's f64 state), count     x_i = min(c, N) · n_i / N,
//                                                   0 when N = 0
//
// The integer total is rounded once, for the compartment, and not once per
// stage: eight stages of 10 at a fraction of 0.2/3 move floor(80 · 0.2/3) = 5,
// where eight separate fraction transfers move floor(10 · 0.2/3) = 0 each.

/// Divide `m` individuals among stages holding `counts`, in proportion to
/// occupancy, by the largest-remainder method in integer arithmetic.
///
/// With `N = Σ counts`, stage `i` gets `floor(m · n_i / N)`, and the
/// `m − Σ floor(…)` individuals left over go one each to the stages with the
/// largest remainders `m · n_i mod N`, a tie going to the earlier stage. The
/// result sums to `m` and never exceeds `counts[i]` at stage `i`: a stage
/// that gets an extra individual has a non-zero remainder, so its exact share
/// `m · n_i / N` is not a whole number and lies strictly below `n_i`.
///
/// `None` unless every count is non-negative and `0 ≤ m ≤ N`. With `N = 0`
/// (so `m = 0`) nothing moves.
fn allocate_by_occupancy(counts: &[i64], m: i64) -> Option<Vec<i64>> {
    if counts.iter().any(|&n| n < 0) {
        return None;
    }
    let total: i128 = counts.iter().map(|&n| n as i128).sum();
    if m < 0 || (m as i128) > total {
        return None;
    }
    if total == 0 {
        return Some(vec![0; counts.len()]);
    }
    let m = m as i128;
    let mut shares: Vec<i64> = Vec::with_capacity(counts.len());
    let mut remainders: Vec<i128> = Vec::with_capacity(counts.len());
    for &n in counts {
        let exact = m * n as i128;
        // The quotient is at most `n`, so it fits an i64.
        shares.push((exact / total) as i64);
        remainders.push(exact % total);
    }
    // Σ remainders = left · N and each is below N, so at least `left` stages
    // have a non-zero remainder: the extra individuals all land on one.
    let left = (m - shares.iter().map(|&s| s as i128).sum::<i128>()) as usize;
    let mut order: Vec<usize> = (0..counts.len()).collect();
    // Stable, so equal remainders keep stage order: the earlier stage first.
    order.sort_by(|&a, &b| remainders[b].cmp(&remainders[a]));
    for &i in order.iter().take(left) {
        shares[i] += 1;
    }
    Some(shares)
}

/// The cells of a grouped transfer as `(source, destination)` local indices,
/// all in the integer arena or all in the real one.
enum GroupedCells {
    Int(Vec<(usize, usize)>),
    Real(Vec<(usize, usize)>),
}

fn grouped_cells(model: &CompiledModel, gt: &GroupedTransfer) -> Result<GroupedCells, SimError> {
    let mut int = Vec::new();
    let mut real = Vec::new();
    // The first source of each representation, to name both in a refusal.
    let (mut first_int, mut first_real) = (None, None);
    for (src, dst) in &gt.pairs {
        match (resolve_target(model, src)?, resolve_target(model, dst)?) {
            (Arena::Int(s), Arena::Int(d)) => {
                first_int.get_or_insert(src);
                int.push((s, d));
            }
            (Arena::Real(s), Arena::Real(d)) => {
                first_real.get_or_insert(src);
                real.push((s, d));
            }
            _ => return Err(mixed_arena_err(src, dst)),
        }
    }
    match (first_int, first_real) {
        (Some(_), None) => Ok(GroupedCells::Int(int)),
        (None, Some(_)) => Ok(GroupedCells::Real(real)),
        (Some(a), Some(b)) => Err(mixed_arena_err(a, b)),
        (None, None) => Err(SimError::Validation(
            "a grouped transfer lists no (source, destination) pairs".into(),
        )),
    }
}

/// A plain `transfer(count = …)` that resolved to a negative count, named by
/// its source compartment. `attempted_value` is the floor of the raw count,
/// as for a negative `add`.
fn transfer_negative_count_err(src: &str, v: f64, t: f64) -> SimError {
    SimError::NegativeCount {
        compartment: src.to_string(),
        attempted_value: v.floor() as i64,
        t,
        cause: NegativeCountCause::InterventionTransferNegative,
    }
}

/// A grouped transfer of a negative count. A count is a number of
/// individuals to move out of the source; there is no stage to return a
/// negative one to.
fn grouped_negative_count_err(gt: &GroupedTransfer, v: f64) -> SimError {
    SimError::Validation(format!(
        "transfer ({}) has a negative count {v}: a `transfer(count = …)` out of a staged \
         compartment moves a number of individuals, which cannot be negative — write the \
         transfer in the other direction",
        crate::intervention::grouped_pairs_label(gt)
    ))
}

/// The amount each source cell of a grouped transfer gives up in `f64`
/// arithmetic, given the resolved amount `v` and the cells' values `n`: the
/// real-valued rules of the table above, shared by the real arena of the
/// discrete resolver and the ODE's continuous state.
fn grouped_amounts_f64(gt: &GroupedTransfer, v: f64, n: &[f64]) -> Result<Vec<f64>, SimError> {
    match gt.amount {
        GroupedAmount::Fraction { .. } => {
            let frac = v.clamp(0.0, 1.0);
            Ok(n.iter().map(|&n_i| n_i * frac).collect())
        }
        GroupedAmount::Count { .. } => {
            if v < 0.0 {
                return Err(grouped_negative_count_err(gt, v));
            }
            let total: f64 = n.iter().sum();
            if total > 0.0 {
                let moved = v.min(total);
                Ok(n.iter().map(|&n_i| moved * n_i / total).collect())
            } else {
                Ok(vec![0.0; n.len()])
            }
        }
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
    crate::trace::effect_action(kind, iv_name, action, v, t);
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

    /// Evaluate the amount expression of action `action_idx` of event `iv_idx`
    /// against this arena's current state, exactly as [`Self::apply_phase`]
    /// evaluates it. Read-only. Used by [`refuse_overdrawing_drains`] to read
    /// the fractions of coincident drains before they are applied.
    #[allow(clippy::too_many_arguments)]
    fn eval_amount(
        &self,
        model: &CompiledModel,
        iv_idx: usize,
        action_idx: usize,
        params: &[f64],
        t: f64,
        dt: f64,
    ) -> f64;
}

/// Refuse a `set` beside any other drain-and-`set` action on the same
/// compartment among the events due at one instant: a second `set`, or a
/// `transfer` from or into it. In that phase every action contributes its change
/// against the same state, so the `set` and the other change add — from
/// `X = 10`, `{ X = 5 }` and a half drain of `X` gave `X = 0`, a value neither
/// event names. A config error, raised on every backend. An inflow `add` beside
/// a `set` is not refused: it acts in the earlier inflow phase, and the `set`
/// then overwrites the post-inflow value (language spec §13.10).
fn refuse_coincident_sets(model: &CompiledModel, event_idx: &[usize], t: f64) -> Result<(), SimError> {
    // Most event batches contain no sets. Avoid building an index for them.
    if !event_idx.iter().any(|&i| model.model.interventions[i].actions.iter()
        .any(|a| matches!(a, Action::Set(_)))) {
        return Ok(());
    }

    #[derive(Default)]
    struct Touches<'a> {
        sets: Vec<&'a str>,
        first_transfer: Option<&'a str>,
    }
    let mut targets: std::collections::HashMap<&str, Touches<'_>> =
        std::collections::HashMap::new();
    let mut first_set_order = Vec::new();
    for &i in event_idx {
        let iv = &model.model.interventions[i];
        let name = iv.name.as_str();
        for action in &iv.actions {
            match action {
                Action::Set(sa) => {
                    let comp = sa.compartment.as_str();
                    let entry = targets.entry(comp).or_default();
                    if entry.sets.is_empty() {
                        first_set_order.push(comp);
                    }
                    // Match Vec::dedup on the old ordered, per-target set list:
                    // only consecutive equal names collapse.
                    if entry.sets.last().copied() != Some(name) {
                        entry.sets.push(name);
                    }
                }
                Action::FractionTransfer(ft) => {
                    for comp in [ft.src.as_str(), ft.dst.as_str()] {
                        targets.entry(comp).or_default().first_transfer.get_or_insert(name);
                    }
                }
                Action::AbsoluteTransfer(at) => {
                    for comp in [at.src.as_str(), at.dst.as_str()] {
                        targets.entry(comp).or_default().first_transfer.get_or_insert(name);
                    }
                }
                Action::GroupedTransfer(gt) => {
                    for (src, dst) in &gt.pairs {
                        for comp in [src.as_str(), dst.as_str()] {
                            targets.entry(comp).or_default().first_transfer.get_or_insert(name);
                        }
                    }
                }
                Action::Add(_) => {}
            }
        }
    }
    // Hash iteration must never choose which conflict is reported. Preserve
    // first-set order and the multiple-sets-before-transfer diagnostic priority.
    for comp in first_set_order {
        let entry = &targets[comp];
        let sets = &entry.sets;
        let set_name = sets[0];
        if sets.len() >= 2 {
            let names = sets.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
            return Err(SimError::Validation(format!(
                "more than one `set` on compartment '{comp}' at t={t}: events {names} each set it \
                 at the same instant, and their combined effect is not defined — give the \
                 sets different times or merge them into one"
            )));
        }
        if let Some(other) = entry.first_transfer {
            return Err(SimError::Validation(format!(
                "a `set` and a `transfer` on compartment '{comp}' at t={t}: event '{set_name}' \
                 sets it and event '{other}' moves people from or into it at the same instant, \
                 and their combined value is one neither event names — give them different \
                 times or merge them into one"
            )));
        }
    }
    Ok(())
}

/// One compartment a `transfer` drains among the events due at an instant:
/// the source compartment, its event, where the transfer's amount expression
/// lives, and whether it moves a count (`AbsoluteTransfer`) or a share
/// (`FractionTransfer`). A `GroupedTransfer` is one of these per source cell,
/// by its amount form, marked `grouped`.
struct Drain<'m> {
    src: &'m str,
    event: &'m str,
    iv_idx: usize,
    action_idx: usize,
    is_count: bool,
    grouped: bool,
}

/// How one action drains one compartment.
struct DrainSource<'m> {
    src: &'m str,
    is_count: bool,
    grouped: bool,
}

/// The `k`-th compartment `action` drains: one for a plain transfer, every
/// source cell for a grouped transfer, none for `add` and `set`. `None` past
/// the last.
fn drain_source(action: &Action, k: usize) -> Option<DrainSource<'_>> {
    match action {
        Action::FractionTransfer(ft) => (k == 0)
            .then_some(DrainSource { src: ft.src.as_str(), is_count: false, grouped: false }),
        Action::AbsoluteTransfer(at) => (k == 0)
            .then_some(DrainSource { src: at.src.as_str(), is_count: true, grouped: false }),
        Action::GroupedTransfer(gt) => {
            let is_count = matches!(gt.amount, GroupedAmount::Count { .. });
            gt.pairs.get(k).map(|(src, _)| DrainSource { src: src.as_str(), is_count, grouped: true })
        }
        Action::Add(_) | Action::Set(_) => None,
    }
}

/// `'a', 'b'` — the distinct event names of `drains`, in declaration order.
fn event_list(drains: &[&Drain<'_>]) -> String {
    let mut names: Vec<&str> = Vec::new();
    for d in drains {
        if !names.contains(&d.event) {
            names.push(d.event);
        }
    }
    names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ")
}

/// Refuse coincident drains that can take more than their source holds.
///
/// Within the drain-and-`set` phase every `transfer` reads the same state and
/// the changes add, and each is capped only by that state on its own. Two
/// transfers of 6 from X = 10 therefore leave X = −2: the model does not say
/// which of them gets the people when there are not enough for both. Among the
/// events due at one instant this refuses, on every backend:
///
///   - two or more drains from one compartment when at least one moves a count
///     (`transfer(count = …)`) — refused whatever the compartment holds, so the
///     refusal depends only on which events are due and fires identically on
///     every run and every particle, the first time the pair is due;
///   - fraction-only drains from one compartment whose fractions (each clamped
///     to [0, 1], as the transfer clamps it) sum to more than 1. Fractions
///     summing to at most 1 can never overdraw (each moves the floor of its
///     share) and are not refused. A fraction is evaluated at fire time against
///     the state the drain phase reads, exactly as the transfer evaluates it, so
///     a sum above 1 that comes from the parameters alone is refused identically
///     for every particle; a fraction that reads the state is checked against
///     the value it takes on that particle.
///
/// The fraction sum is formed in ascending order (so the verdict does not
/// depend on declaration order) and compared with 1 plus the rounding of that
/// sum (`n · ε` for `n` fractions), so ten drains of 0.1 are not refused.
///
/// A grouped transfer drains every one of its source cells, with its one
/// fraction (or count) on each, and two more rules follow from how it divides
/// its total ([`allocate_by_occupancy`]):
///
///   - two grouped transfers that share a source cell are refused, whatever
///     their fractions and whatever the cell holds. Each gives the individuals
///     left over by its own division to the stages with the largest
///     remainders, so both can take the same individual: from stages (1, 1),
///     two transfers of one half each take stage 1;
///   - a grouped fraction transfer beside plain fraction transfers from one
///     of its cells cannot overdraw when the fractions sum to at most 1, and
///     is checked by that sum alone. From a cell holding `n`, the grouped one
///     takes at most `ceil(n·f)` and the plain ones `floor(n·f')` each; if
///     `n·f` is whole these sum to at most `n·(f + Σf')`, and if it is not,
///     the floors sum to a whole number below `n − n·f`, so to at most
///     `n − ceil(n·f)`.
///
/// Drains into a compartment, and a single drain capped by its source, are not
/// this function's concern; a `set` beside a `transfer` is refused by
/// [`refuse_coincident_sets`].
fn refuse_overdrawing_drains<A: EventArena>(
    model: &CompiledModel,
    event_idx: &[usize],
    arena: &A,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Result<(), SimError> {
    let drain_actions = || {
        event_idx.iter().flat_map(move |&iv_idx| {
            let iv = &model.model.interventions[iv_idx];
            iv.actions.iter().enumerate().flat_map(move |(action_idx, a)| {
                (0..).map_while(move |k| drain_source(a, k)).map(move |d| Drain {
                    src: d.src,
                    event: iv.name.as_str(),
                    iv_idx,
                    action_idx,
                    is_count: d.is_count,
                    grouped: d.grouped,
                })
            })
        })
    };
    // Fast path: fewer than two drains in the batch cannot collide, and finding
    // that allocates nothing.
    if drain_actions().nth(1).is_none() {
        return Ok(());
    }
    let drains: Vec<Drain<'_>> = drain_actions().collect();
    for (i, d) in drains.iter().enumerate() {
        if drains[..i].iter().any(|e| e.src == d.src) {
            continue; // reported with the compartment's first drain
        }
        let same: Vec<&Drain<'_>> = drains.iter().filter(|e| e.src == d.src).collect();
        if same.len() < 2 {
            continue;
        }
        let comp = d.src;
        let both = if same.len() == 2 { "both" } else { "all of them" };
        if same.iter().any(|e| e.is_count) {
            return Err(SimError::Validation(format!(
                "more than one `transfer` from compartment '{comp}' at t={t}: events {names} \
                 each move people out of it at the same instant, and at least one moves a \
                 fixed count, so the model does not say what happens when there are not \
                 enough people for {both} — give them different times or merge them into \
                 one transfer",
                names = event_list(&same),
            )));
        }
        if same.iter().filter(|e| e.grouped).count() >= 2 {
            return Err(SimError::Validation(format!(
                "more than one `transfer` out of a staged compartment draws from '{comp}' at \
                 t={t}: events {names} each divide their own total among its stages at the \
                 same instant, and can each take the same individual from one stage, so the \
                 model does not say who goes — give them different times or merge them into \
                 one transfer",
                names = event_list(&same),
            )));
        }
        let mut fracs: Vec<f64> = Vec::with_capacity(same.len());
        for e in &same {
            let iv = &model.model.interventions[e.iv_idx];
            let v = arena.eval_amount(model, e.iv_idx, e.action_idx, params, t, dt);
            let v = crate::intervention::finite_action_value(v, &iv.name, &iv.actions[e.action_idx], t)?;
            fracs.push(v.clamp(0.0, 1.0));
        }
        let shown = fracs.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" + ");
        let mut sorted = fracs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sum: f64 = sorted.iter().sum();
        if sum > 1.0 + (sorted.len() as f64) * f64::EPSILON {
            return Err(SimError::Validation(format!(
                "`transfer` fractions from compartment '{comp}' sum to {sum} ({shown}) at \
                 t={t}: events {names} each move a share of it at the same instant, and \
                 together they ask for more than it holds, so the model does not say what \
                 happens when there are not enough people for {both} — give them different \
                 times or merge them into one transfer",
                names = event_list(&same),
            )));
        }
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

    fn eval_amount(
        &self,
        model: &CompiledModel,
        iv_idx: usize,
        action_idx: usize,
        params: &[f64],
        t: f64,
        dt: f64,
    ) -> f64 {
        // The context `resolve_one` builds for the same action and state.
        let ctx = EvalCtx {
            model, int_s: self.int, real_s: self.real, params, t, dt,
            projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        eval_resolved(&model.resolved.intervention_exprs[iv_idx][action_idx], &ctx)
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
        refuse_overdrawing_drains(model, event_idx, arena, params, t, dt)?;
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
            if v < 0.0 {
                return Err(transfer_negative_count_err(&at.src, v, 0.0));
            }
            let (s, d) = (resolve_target(model, &at.src)?, resolve_target(model, &at.dst)?);
            let x = match s {
                Arena::Int(i) if matches!(d, Arena::Int(_)) => v.min(read_int[i]),
                Arena::Real(i) if matches!(d, Arena::Real(_)) => v.min(read_real[i]),
                _ => return Err(mixed_arena_err(&at.src, &at.dst)),
            };
            out.push((s, ChangeF64::Delta(-x)));
            out.push((d, ChangeF64::Delta(x)));
        }
        Action::GroupedTransfer(gt) => {
            // The ODE holds integer compartments as f64, so both arenas take
            // the real-valued rules. Two changes per pair, in pair order.
            let (cells, values): (Vec<(Arena, Arena)>, Vec<f64>) = match grouped_cells(model, gt)? {
                GroupedCells::Int(cells) => (
                    cells.iter().map(|&(s, d)| (Arena::Int(s), Arena::Int(d))).collect(),
                    cells.iter().map(|&(s, _)| read_int[s]).collect(),
                ),
                GroupedCells::Real(cells) => (
                    cells.iter().map(|&(s, d)| (Arena::Real(s), Arena::Real(d))).collect(),
                    cells.iter().map(|&(s, _)| read_real[s]).collect(),
                ),
            };
            let amounts = grouped_amounts_f64(gt, v, &values)?;
            for (&(s, d), &x) in cells.iter().zip(&amounts) {
                out.push((s, ChangeF64::Delta(-x)));
                out.push((d, ChangeF64::Delta(x)));
            }
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
    /// The forward sensitivities riding along on the ODE gradient path
    /// ([`SensBlock`]); `None` on the value path (`run_ode`), whose arithmetic
    /// is then exactly the code above and below with every `sens` branch
    /// skipped.
    pub sens: Option<SensBlock<'a>>,
}

/// The forward-sensitivity block that rides with the ODE's continuous arena
/// through an effect instant (gh#275 §1g). `state` is `S = ∂x_int/∂θ`, an
/// `n_int × d` row-major block over the integer compartments
/// (`d = param_model_idx.len()`); `param_model_idx` maps each sensitivity
/// column to its model parameter index, the key the compiler-emitted
/// `∂amount/∂θ` maps (`*_grad`, `CompiledModel::intervention_amount_grads`) are
/// read through.
///
/// When an arena carries one, every value change an action makes is applied
/// together with the change it makes to `S` — the jump `x⁺ = Φ(x⁻, θ)` and its
/// derivative `S⁺ = (∂Φ/∂x)·S⁻ + ∂Φ/∂θ`, per action:
///
/// ```text
/// add(c, v)                     x_c += v              S[c,:] += ∂v/∂θ
/// set c = v                     x_c  = v              S[c,:]  = ∂v/∂θ
/// transfer(fraction = f, s → d) x = x_s · clamp(f)    dx = clamp(f)·S[s,:] + x_s·∂f/∂θ·[0 ≤ f ≤ 1]
///                               x_s -= x,  x_d += x   S[s,:] -= dx,  S[d,:] += dx
/// ```
///
/// A grouped transfer of a fraction is the last rule on each of its
/// (source, destination) pairs, with the one `f` and `∂f/∂θ`.
///
/// Every read a rule makes — the amount `v`, the source `x_s`, the row `S[s,:]`
/// — is taken from the same snapshot the value change reads (the phase snapshot
/// for events, the live state for sequential interventions), and where several
/// changes land on one compartment their sensitivity rows are summed in the
/// same ascending-delta order the values are, so `S` is the exact derivative of
/// the value computation beside it. The clamp's derivative is taken on its
/// closed active piece `[0, 1]` and is zero outside it, where the value is
/// locally constant in θ. An amount that reads compartment state, a
/// `transfer(count = …)`, grouped or not (a nonsmooth `min`), and a
/// real-compartment target have
/// no rule here: the gradient capability gate refuses them before this path
/// runs, and reaching one is an internal error, never a silently incomplete `S`.
pub struct SensBlock<'a> {
    pub state: &'a mut [f64],
    pub param_model_idx: &'a [usize],
}

/// Evaluate `∂amount/∂θ` of one action for each sensitivity column, against the
/// same f64 state the amount itself is evaluated at ([`eval_amount_f64`]).
#[allow(clippy::too_many_arguments)]
fn eval_amount_grad_f64(
    model: &CompiledModel,
    iv_idx: usize,
    action_idx: usize,
    int_f64: &[f64],
    real_f64: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    param_model_idx: &[usize],
) -> Result<Vec<f64>, SimError> {
    let Some(grad) = model.resolved.intervention_amount_grads[iv_idx][action_idx].as_ref() else {
        return Err(SimError::Validation(format!(
            "internal: the ODE gradient path reached action {} of `{}`, a \
             `transfer(count = …)`, which carries no amount gradient; the gradient \
             capability gate should have refused this model",
            action_idx + 1,
            model.model.interventions[iv_idx].name
        )));
    };
    let placeholder = IntState::new(model.int_local_to_global.len());
    let rs = RealState::from_vec(real_f64.to_vec());
    let ctx = EvalCtx {
        model, int_s: &placeholder, real_s: &rs, params, t, dt,
        projected: None, aux: None, int_float_override: Some(int_f64),
        per_eval: None,
    };
    Ok(param_model_idx
        .iter()
        .map(|&m| crate::resolved_expr::eval_emitted_grad(grad, m, &ctx))
        .collect())
}

/// The sensitivity rows of one action — `d` entries per change, parallel (in
/// order) to the changes [`action_changes_f64`] appended for it in `changes` —
/// given its value `v`, its `∂v/∂θ` (`dv`), and the snapshot values and
/// sensitivities (`read_int`, `snap_sens`) the action reads. The rules are the
/// table on [`SensBlock`].
#[allow(clippy::too_many_arguments)]
fn action_changes_sens(
    model: &CompiledModel,
    action: &Action,
    v: f64,
    dv: &[f64],
    read_int: &[f64],
    snap_sens: &[f64],
    changes: &[(Arena, ChangeF64)],
    out: &mut Vec<Vec<f64>>,
) -> Result<(), SimError> {
    let d = dv.len();
    let int_target = |a: Arena| -> Result<usize, SimError> {
        match a {
            Arena::Int(i) => Ok(i),
            Arena::Real(i) => Err(SimError::Validation(format!(
                "internal: the ODE gradient path applied an effect to real compartment \
                 `{}`, whose forward sensitivity is not tracked; the gradient capability \
                 gate should have refused this model",
                model.model.compartments[model.real_local_to_global[i]].name
            ))),
        }
    };
    match action {
        Action::Add(_) | Action::Set(_) => {
            // One change: `x_c += v` or `x_c = v`, so the row is ∂v/∂θ either way
            // (a `Write` row is the written value's derivative).
            let [(target, _)] = changes else {
                return Err(SimError::Validation("internal: add/set made more than one change".into()));
            };
            int_target(*target)?;
            out.push(dv.to_vec());
        }
        Action::FractionTransfer(_) => {
            // Two changes, `x_s -= x` then `x_d += x`, with `x = x_s · clamp(v)`.
            let [(src, _), (dst, _)] = changes else {
                return Err(SimError::Validation("internal: a fraction transfer did not make two changes".into()));
            };
            let s = int_target(*src)?;
            int_target(*dst)?;
            let frac = v.clamp(0.0, 1.0);
            let active = (0.0..=1.0).contains(&v);
            let dx: Vec<f64> = (0..d)
                .map(|p| {
                    let through_state = frac * snap_sens[s * d + p];
                    if active { through_state + read_int[s] * dv[p] } else { through_state }
                })
                .collect();
            out.push(dx.iter().map(|x| -x).collect());
            out.push(dx);
        }
        Action::GroupedTransfer(gt) => match gt.amount {
            GroupedAmount::Fraction { .. } => {
                // The fraction transfer's rule on each pair: two changes,
                // `x_s -= x` then `x_d += x`, with `x = x_s · clamp(v)`.
                if changes.len() != 2 * gt.pairs.len() {
                    return Err(SimError::Validation(
                        "internal: a grouped transfer did not make two changes per pair".into(),
                    ));
                }
                let frac = v.clamp(0.0, 1.0);
                let active = (0.0..=1.0).contains(&v);
                for pair in changes.chunks_exact(2) {
                    let s = int_target(pair[0].0)?;
                    int_target(pair[1].0)?;
                    let dx: Vec<f64> = (0..d)
                        .map(|p| {
                            let through_state = frac * snap_sens[s * d + p];
                            if active { through_state + read_int[s] * dv[p] } else { through_state }
                        })
                        .collect();
                    out.push(dx.iter().map(|x| -x).collect());
                    out.push(dx);
                }
            }
            GroupedAmount::Count { .. } => {
                return Err(SimError::Validation(
                    "internal: the ODE gradient path reached a grouped `transfer(count = …)`, \
                     whose jump min(count, source) has no sensitivity rule; the gradient \
                     capability gate should have refused this model"
                        .into(),
                ));
            }
        },
        Action::AbsoluteTransfer(_) => {
            return Err(SimError::Validation(
                "internal: the ODE gradient path reached a `transfer(count = …)`, whose \
                 jump min(count, source) has no sensitivity rule; the gradient capability \
                 gate should have refused this model"
                    .into(),
            ));
        }
    }
    Ok(())
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
        // The phase-frozen sensitivities S⁻, which every transfer of this phase
        // reads exactly as its value reads `snap_int`.
        let snap_sens: Option<Vec<f64>> = self.sens.as_ref().map(|s| s.state.to_vec());
        let mut changes: Vec<(Arena, ChangeF64)> = Vec::new();
        // `sens_rows[k]` is the sensitivity row of `changes[k]` (gradient path only).
        let mut sens_rows: Vec<Vec<f64>> = Vec::new();
        for &iv_idx in event_idx {
            let iv = &model.model.interventions[iv_idx];
            for (action_idx, action) in iv.actions.iter().enumerate() {
                if action_phase(action) != phase {
                    continue;
                }
                let v = eval_amount_f64(model, iv_idx, action_idx, &snap_int, &snap_real, params, t, dt);
                let v = crate::intervention::finite_action_value(v, &iv.name, action, t)?;
                crate::trace::effect_action(EffectKind::Event, &iv.name, action, v, t);
                let first = changes.len();
                action_changes_f64(model, action, v, &snap_int, &snap_real, &mut changes)?;
                if let (Some(sb), Some(ss)) = (self.sens.as_ref(), snap_sens.as_ref()) {
                    let dv = eval_amount_grad_f64(
                        model, iv_idx, action_idx, &snap_int, &snap_real, params, t, dt,
                        sb.param_model_idx,
                    )?;
                    action_changes_sens(
                        model, action, v, &dv, &snap_int, ss, &changes[first..], &mut sens_rows,
                    )?;
                    debug_assert_eq!(sens_rows.len(), changes.len());
                }
            }
        }
        // Group the changes by compartment, in first-appearance order, with a
        // per-arena index map (linear in the number of changes).
        let mut group_int = vec![usize::MAX; snap_int.len()];
        let mut group_real = vec![usize::MAX; snap_real.len()];
        let mut groups: Vec<(Arena, Vec<usize>)> = Vec::new();
        for (k, &(target, _)) in changes.iter().enumerate() {
            let g = match target {
                Arena::Int(i) => &mut group_int[i],
                Arena::Real(i) => &mut group_real[i],
            };
            if *g == usize::MAX {
                *g = groups.len();
                groups.push((target, Vec::with_capacity(1)));
            }
            groups[*g].1.push(k);
        }
        let mut sorted: Vec<f64> = Vec::new();
        for (target, ks) in &groups {
            let cs: Vec<ChangeF64> = ks.iter().map(|&k| changes[k].1).collect();
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
            // The same combination on `S[c,:]`: one change applies directly (a
            // `Write` row replaces the row), several add their rows to the
            // snapshot row in the value's ascending-delta order, a `Write`
            // contributing `∂v/∂θ − S⁻[c,:]` as its value contributes `v − x⁻`.
            if let (Some(sb), Some(ss)) = (self.sens.as_mut(), snap_sens.as_ref()) {
                let Arena::Int(c) = *target else {
                    return Err(SimError::Validation(format!(
                        "internal: the ODE gradient path applied an event to real compartment \
                         `{}`, whose forward sensitivity is not tracked",
                        model.model.compartments[model.real_local_to_global[match *target {
                            Arena::Real(i) => i,
                            Arena::Int(_) => unreachable!(),
                        }]].name
                    )));
                };
                let d = sb.param_model_idx.len();
                let row = &mut sb.state[c * d..(c + 1) * d];
                if let [k] = ks.as_slice() {
                    match changes[*k].1 {
                        ChangeF64::Delta(_) => {
                            for (p, r) in row.iter_mut().enumerate() { *r += sens_rows[*k][p]; }
                        }
                        ChangeF64::Write(_) => row.copy_from_slice(&sens_rows[*k]),
                    }
                } else {
                    let mut order: Vec<usize> = ks.clone();
                    order.sort_by(|&a, &b| as_delta(&changes[a].1).total_cmp(&as_delta(&changes[b].1)));
                    for (p, r) in row.iter_mut().enumerate() {
                        let s_minus = ss[c * d + p];
                        *r = order.iter().fold(s_minus, |acc, &k| {
                            acc + match changes[k].1 {
                                ChangeF64::Delta(_) => sens_rows[k][p],
                                ChangeF64::Write(_) => sens_rows[k][p] - s_minus,
                            }
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn eval_amount(
        &self,
        model: &CompiledModel,
        iv_idx: usize,
        action_idx: usize,
        params: &[f64],
        t: f64,
        dt: f64,
    ) -> f64 {
        eval_amount_f64(model, iv_idx, action_idx, self.int, self.real, params, t, dt)
    }
}

/// Apply the due scheduled interventions `intervention_idx` to the ODE's
/// continuous compartment vectors, exactly (no rounding), in declaration order
/// and sequentially: each action reads the live state the previous one left.
/// This is the stage after [`apply_event_phases`] at an ODE boundary; ODE
/// carries no balance. `t_boundary` is the effect time; `dt` drives the amount
/// evaluation. On the gradient path (`arena.sens` present) each action's
/// sensitivity change is applied with its value change, read against the same
/// live state and live `S` ([`SensBlock`]).
pub fn apply_interventions_continuous(
    model: &CompiledModel,
    intervention_idx: &[usize],
    arena: &mut ContinuousArena<'_>,
    params: &[f64],
    t_boundary: f64,
    dt: f64,
) -> Result<(), SimError> {
    let t = t_boundary;
    for &iv_idx in intervention_idx {
        let iv = &model.model.interventions[iv_idx];
        for (action_idx, action) in iv.actions.iter().enumerate() {
            let live_int = arena.int.to_vec();
            let live_real = arena.real.to_vec();
            let v = eval_amount_f64(model, iv_idx, action_idx, &live_int, &live_real, params, t, dt);
            let v = crate::intervention::finite_action_value(v, &iv.name, action, t)?;
            crate::trace::effect_action(EffectKind::Intervention, &iv.name, action, v, t);
            // The sensitivity rows, read against the live `S` before the value
            // apply below moves the state.
            let sens_update: Option<(Vec<(Arena, ChangeF64)>, Vec<Vec<f64>>)> = match arena.sens.as_ref() {
                None => None,
                Some(sb) => {
                    let dv = eval_amount_grad_f64(
                        model, iv_idx, action_idx, &live_int, &live_real, params, t, dt,
                        sb.param_model_idx,
                    )?;
                    let mut changes = Vec::with_capacity(2);
                    action_changes_f64(model, action, v, &live_int, &live_real, &mut changes)?;
                    let mut rows = Vec::with_capacity(2);
                    action_changes_sens(model, action, v, &dv, &live_int, sb.state, &changes, &mut rows)?;
                    Some((changes, rows))
                }
            };
            apply_action_f64(model, action, v, &live_int, &live_real, arena.int, arena.real)?;
            if let (Some(sb), Some((changes, rows))) = (arena.sens.as_mut(), sens_update) {
                let d = sb.param_model_idx.len();
                for ((target, change), row) in changes.iter().zip(rows) {
                    let Arena::Int(c) = *target else {
                        return Err(SimError::Validation(
                            "internal: the ODE gradient path applied an intervention to a real \
                             compartment, whose forward sensitivity is not tracked".into(),
                        ));
                    };
                    let slot = &mut sb.state[c * d..(c + 1) * d];
                    match change {
                        ChangeF64::Delta(_) => {
                            for (r, v) in slot.iter_mut().zip(&row) { *r += v; }
                        }
                        ChangeF64::Write(_) => slot.copy_from_slice(&row),
                    }
                }
            }
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
            // Guard the raw value, as `add` does: a count in (-0.5, 0) rounds
            // to zero and would otherwise pass as a silent no-op.
            if v < 0.0 {
                return Err(transfer_negative_count_err(&at.src, v, t));
            }
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
        Action::GroupedTransfer(gt) => match grouped_cells(model, gt)? {
            GroupedCells::Int(cells) => {
                let counts: Vec<i64> = cells.iter().map(|&(s, _)| snap.int.counts[s]).collect();
                let total: i64 = counts.iter().sum();
                let moved = match gt.amount {
                    GroupedAmount::Fraction { .. } => {
                        // `.min(total)`: above 2^53 the product can round up.
                        (((total as f64) * v.clamp(0.0, 1.0)).floor() as i64).min(total)
                    }
                    GroupedAmount::Count { .. } => {
                        if v < 0.0 {
                            return Err(grouped_negative_count_err(gt, v));
                        }
                        (v.round() as i64).min(total)
                    }
                };
                let shares = allocate_by_occupancy(&counts, moved).ok_or_else(|| {
                    SimError::Validation(format!(
                        "internal: grouped transfer ({}) at t={t} cannot divide {moved} among \
                         stages holding {counts:?}; a stage count is negative",
                        crate::intervention::grouped_pairs_label(gt)
                    ))
                })?;
                for (&(s, d), &x) in cells.iter().zip(&shares) {
                    out.int.push(IntDelta { idx: s, delta: -x });
                    out.int.push(IntDelta { idx: d, delta: x });
                }
            }
            GroupedCells::Real(cells) => {
                let values: Vec<f64> = cells.iter().map(|&(s, _)| snap.real.values[s]).collect();
                let amounts = grouped_amounts_f64(gt, v, &values)?;
                for (&(s, d), &x) in cells.iter().zip(&amounts) {
                    out.real.push(RealDelta { idx: s, delta: -x });
                    out.real.push(RealDelta { idx: d, delta: x });
                }
            }
        },
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
mod grouped_tests;

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
            constants: vec![],
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
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0,
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None, t_start_anchor: None,
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

    // Frozen pre-optimization scan: compare decisions AND diagnostic order.
    fn scan_coincident_sets(model: &CompiledModel, event_idx: &[usize], t: f64) -> Result<(), SimError> {
        // (compartment, event name, is_set) for every drain-and-`set` action.
        let touches = || {
            event_idx.iter().flat_map(move |&iv_idx| {
                let iv = &model.model.interventions[iv_idx];
                iv.actions.iter().flat_map(move |a| {
                    let name = iv.name.as_str();
                    let v: Vec<(&str, &str, bool)> = match a {
                        Action::Set(sa) => vec![(sa.compartment.as_str(), name, true)],
                        Action::FractionTransfer(ft) => {
                            vec![(ft.src.as_str(), name, false), (ft.dst.as_str(), name, false)]
                        }
                        Action::AbsoluteTransfer(at) => {
                            vec![(at.src.as_str(), name, false), (at.dst.as_str(), name, false)]
                        }
                        Action::GroupedTransfer(gt) => gt
                            .pairs
                            .iter()
                            .flat_map(|(s, d)| [(s.as_str(), name, false), (d.as_str(), name, false)])
                            .collect(),
                        Action::Add(_) => vec![],
                    };
                    v.into_iter()
                })
            })
        };
        // Fast path: the common batch carries no `set`. Finding that walks the due
        // events' actions once, without allocating beyond the per-action list.
        if !touches().any(|(_, _, is_set)| is_set) {
            return Ok(());
        }
        let all: Vec<(&str, &str, bool)> = touches().collect();
        for (i, &(comp, set_name, is_set)) in all.iter().enumerate() {
            if !is_set || all[..i].iter().any(|&(c, _, s)| c == comp && s) {
                continue; // reported with the compartment's first `set`
            }
            let mut sets: Vec<&str> = all.iter().filter(|&&(c, _, s)| c == comp && s).map(|&(_, n, _)| n).collect();
            sets.dedup();
            if sets.len() >= 2 {
                let names = sets.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
                return Err(SimError::Validation(format!(
                    "more than one `set` on compartment '{comp}' at t={t}: events {names} each set it \
                     at the same instant, and their combined effect is not defined — give the \
                     sets different times or merge them into one"
                )));
            }
            if let Some(&(_, other, _)) = all.iter().find(|&&(c, _, s)| c == comp && !s) {
                return Err(SimError::Validation(format!(
                    "a `set` and a `transfer` on compartment '{comp}' at t={t}: event '{set_name}' \
                     sets it and event '{other}' moves people from or into it at the same instant, \
                     and their combined value is one neither event names — give them different \
                     times or merge them into one"
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn coincident_set_index_matches_scan_for_ordered_batches() {
        let actions = [
            Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(1.0), value_grad: Default::default() }),
            Action::Set(SetAction { compartment: "I".into(), value: Expr::const_(1.0), value_grad: Default::default() }),
            Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(1.0), value_grad: Default::default() }),
            Action::Add(AddAction { compartment: "S".into(), count: Expr::const_(1.0), count_grad: Default::default() }),
            Action::FractionTransfer(FractionTransfer { src: "S".into(), dst: "I".into(), fraction: Expr::const_(0.5), fraction_grad: Default::default() }),
            Action::AbsoluteTransfer(AbsoluteTransfer { src: "I".into(), dst: "S".into(), count: Expr::const_(1.0) }),
        ];
        let mut m = model_with(vec![]);
        let template = m.model.interventions[0].clone();
        // All four-action batches, with repeated and distinct event names.
        for pattern in 0..3 {
            for code in 0..actions.len().pow(4) {
                let mut n = code;
                std::sync::Arc::make_mut(&mut m.model).interventions.clear();
                for i in 0..4 {
                    let mut iv = template.clone();
                    iv.name = match pattern {
                        0 => format!("event_{i}"),
                        1 => "same".into(),
                        _ => format!("event_{}", i % 2),
                    };
                    iv.actions = vec![actions[n % actions.len()].clone()];
                    n /= actions.len();
                    std::sync::Arc::make_mut(&mut m.model).interventions.push(iv);
                }
                for due in [vec![], vec![0], vec![0, 1, 2, 3], vec![3, 1, 0, 2], vec![0, 1, 0, 3]] {
                    let actual = refuse_coincident_sets(&m, &due, 7.0).map_err(|e| e.to_string());
                    let expected = scan_coincident_sets(&m, &due, 7.0).map_err(|e| e.to_string());
                    assert_eq!(actual, expected, "pattern={pattern} code={code} due={due:?}");
                }
            }
        }
    }

    fn set_then_add() -> CompiledModel {
        model_with(vec![
            Action::Set(SetAction { compartment: "I".into(), value: Expr::const_(5.0), value_grad: Default::default() }),
            Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0), count_grad: Default::default() }),
        ])
    }

    fn drain_then_add() -> CompiledModel {
        model_with(vec![
            Action::FractionTransfer(FractionTransfer {
                src: "I".into(),
                dst: "S".into(),
                fraction: Expr::const_(0.5),
                fraction_grad: Default::default(),
            }),
            Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0), count_grad: Default::default() }),
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
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals, sens: None };
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
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals, sens: None };
            apply_event_phases(m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap();
        }
        int_vals
    }

    #[test]
    fn a_set_beside_a_drain_on_one_compartment_is_refused_on_both_arenas() {
        // From S = 10, `set S = 5` and a half drain of S used to add to S = 0, a
        // value neither event names; it is refused, in either order, on the
        // integer and the continuous arena.
        let set = Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(5.0), value_grad: Default::default() });
        let drain = Action::FractionTransfer(FractionTransfer {
            src: "S".into(),
            dst: "I".into(),
            fraction: Expr::const_(0.5),
            fraction_grad: Default::default(),
        });
        for actions in [vec![set.clone(), drain.clone()], vec![drain, set]] {
            let m = model_with(actions);
            let (mut int_s, mut real_s) = states();
            let mut deltas = EffectDeltas::default();
            let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
            let e = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap_err();
            assert!(e.to_string().contains("a `set` and a `transfer` on compartment 'S'"), "{e}");
            let mut iv = vec![10.0, 0.0];
            let mut rv = vec![50.0];
            let mut arena = ContinuousArena { int: &mut iv, real: &mut rv, sens: None };
            let e = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap_err();
            assert!(e.to_string().contains("a `set` and a `transfer` on compartment 'S'"), "{e}");
        }
    }

    #[test]
    fn a_parameter_fraction_summing_above_one_is_refused_on_both_arenas() {
        // `fraction = p` with p = 1 beside a 0.5 drain of S: the sum (1.5) is
        // known only once p is evaluated at fire time, and is refused on the
        // integer and the continuous arena alike.
        let frac = |f: Expr| Action::FractionTransfer(FractionTransfer { src: "S".into(), dst: "I".into(), fraction: f, fraction_grad: Default::default() });
        let m = model_with(vec![frac(Expr::param("p")), frac(Expr::const_(0.5))]);
        let want = "`transfer` fractions from compartment 'S' sum to 1.5 (1 + 0.5) at t=1";
        let (mut int_s, mut real_s) = states();
        let mut deltas = EffectDeltas::default();
        let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
        let e = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap_err();
        assert!(e.to_string().contains(want), "{e}");
        let mut iv = vec![100.0, 0.0];
        let mut rv = vec![50.0];
        let mut arena = ContinuousArena { int: &mut iv, real: &mut rv, sens: None };
        let e = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap_err();
        assert!(e.to_string().contains(want), "{e}");
    }

    #[test]
    fn continuous_lone_negative_int_set_is_refused() {
        // A lone `set` to −1 on an integer compartment used to be integrated
        // from −1 and printed as 0; it is now refused as Gillespie refuses it.
        let m = model_with(vec![Action::Set(SetAction {
            compartment: "S".into(),
            value: Expr::const_(-1.0),
            value_grad: Default::default(),
        })]);
        let mut int_vals = vec![100.0, 0.0];
        let mut real_vals = vec![50.0];
        let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals, sens: None };
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
            Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(1.0), value_grad: Default::default() }),
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
        let m = model_with(vec![Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(3.6), count_grad: Default::default() })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 1, delta: 4 }]); // round(3.6)=4 to I(local 1)
        assert!(d.real.is_empty());
    }

    #[test]
    fn add_real_is_exact_f64() {
        let m = model_with(vec![Action::Add(AddAction { compartment: "W".into(), count: Expr::const_(2.5), count_grad: Default::default() })]);
        let d = resolve(&m);
        assert_eq!(d.real, vec![RealDelta { idx: 0, delta: 2.5 }]); // exact, no round
        assert!(d.int.is_empty());
    }

    #[test]
    fn add_negative_is_hard_error_on_any_path() {
        let m = model_with(vec![Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(-1.0), count_grad: Default::default() })]);
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
            count_grad: Default::default(),
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
            count_grad: Default::default(),
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
            count_grad: Default::default(),
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
            count_grad: Default::default(),
        })]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 1, delta: 0 }], "round(0.4)=0 is a no-op, not an error");
    }

    fn absolute_transfer(src: &str, dst: &str, count: f64) -> Action {
        Action::AbsoluteTransfer(AbsoluteTransfer {
            src: src.into(),
            dst: dst.into(),
            count: Expr::const_(count),
        })
    }

    fn assert_transfer_negative(err: SimError, src: &str) {
        match err {
            SimError::NegativeCount { compartment, cause, .. } => {
                assert_eq!(compartment, src);
                assert_eq!(cause, NegativeCountCause::InterventionTransferNegative);
            }
            other => panic!("expected NegativeCount{{InterventionTransferNegative}}, got: {other}"),
        }
    }

    /// gh#1278: a plain `transfer(count = <0)` on the integer arena used to
    /// move people from the destination to the source (`min(-5, S) = -5`). It
    /// is refused at the action site, for a count below -0.5 and for one in
    /// (-0.5, 0) that rounds to zero.
    #[test]
    fn transfer_count_negative_is_hard_error_on_integer_arena() {
        for count in [-5.0, -0.3] {
            let m = model_with(vec![absolute_transfer("S", "I", count)]);
            let (int_s, real_s) = states();
            let mut out = EffectDeltas::default();
            let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                           &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
            assert_transfer_negative(err, "S");
            assert!(out.is_empty(), "no delta may be emitted when the transfer is rejected (count {count})");
        }
    }

    /// gh#1278, real source: the guard sits before the arena dispatch, so a
    /// negative count is reported as such even where the endpoints would be
    /// refused for mixing arenas.
    #[test]
    fn transfer_count_negative_is_hard_error_on_real_source() {
        let m = model_with(vec![absolute_transfer("W", "I", -5.0)]);
        let (int_s, real_s) = states();
        let mut out = EffectDeltas::default();
        let err = resolve_intervention(&m, 0, &m.model.interventions[0], snap(&int_s, &real_s),
                                       &m.default_params, 1.0, 1.0, EffectKind::Intervention, None, &mut out).unwrap_err();
        assert_transfer_negative(err, "W");
    }

    /// gh#1278, ODE: the `f64` arena's `v.min(source)` was equally negative.
    #[test]
    fn continuous_transfer_count_negative_is_hard_error() {
        for (src, dst) in [("S", "I"), ("W", "I")] {
            let m = model_with(vec![absolute_transfer(src, dst, -5.0)]);
            let mut int_vals = vec![100.0_f64, 0.0];
            let mut real_vals = vec![50.0_f64];
            let err = apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0)
                .unwrap_err();
            assert_transfer_negative(err, src);
            assert_eq!(int_vals, vec![100.0, 0.0], "state untouched when the transfer is rejected");
            assert_eq!(real_vals, vec![50.0]);
        }
    }

    /// gh#1278 boundary: a count of zero moves nobody and is not an error, so
    /// the guard rejects only `v < 0`.
    #[test]
    fn transfer_count_zero_moves_nobody() {
        let m = model_with(vec![absolute_transfer("S", "I", 0.0)]);
        let d = resolve(&m);
        assert_eq!(d.int, vec![IntDelta { idx: 0, delta: 0 }, IntDelta { idx: 1, delta: 0 }]);
        let mut int_vals = vec![100.0_f64, 0.0];
        let mut real_vals = vec![50.0_f64];
        apply_boundary_effects_continuous(&m, &mut int_vals, &mut real_vals, &m.default_params, 1.0, 1.0).unwrap();
        assert_eq!(int_vals, vec![100.0, 0.0]);
    }

    #[test]
    fn set_int_emits_snapshot_relative_delta() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(70.4), value_grad: Default::default() })]);
        let d = resolve(&m);
        // round(70.4)=70, snapshot S=100 → delta -30 → S ends at 70.
        assert_eq!(d.int, vec![IntDelta { idx: 0, delta: 70 - 100 }]);
    }

    #[test]
    fn set_real_is_exact() {
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(12.5), value_grad: Default::default() })]);
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
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(-5.0), value_grad: Default::default() })]);
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
        let m = model_with(vec![Action::Set(SetAction { compartment: "W".into(), value: Expr::const_(-5.0), value_grad: Default::default() })]);
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
            fraction_grad: Default::default(),
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
                fraction_grad: Default::default(),
            }),
            Action::Add(AddAction {
                compartment: "I".into(), count: Expr::const_(7.0),
                count_grad: Default::default(),
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
            fraction_grad: Default::default(),
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
        let mut arena = ContinuousArena { int: int_vals, real: real_vals, sens: None };
        apply_event_phases(model, &batch.event_idx, &mut arena, params, t_boundary, dt, None)?;
        apply_interventions_continuous(
            model, &batch.intervention_idx, &mut arena, params, t_boundary, dt,
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
            fraction_grad: Default::default(),
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
            value_grad: Default::default(),
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
            constants: vec![],
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
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0,
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None, t_start_anchor: None,
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
            let mut arena = ContinuousArena { int: &mut int_vals, real: &mut real_vals, sens: None };
            apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None)?;
        }
        Ok(real_vals)
    }

    /// The real-compartment delta path: three drains on W1 at one instant.
    /// The result is order-independent because the changes are added in
    /// ascending order.
    #[test]
    fn continuous_real_drains_are_bit_identical_in_either_order() {
        let frac2 = Action::FractionTransfer(FractionTransfer {
            src: "W1".into(),
            dst: "W2".into(),
            fraction: Expr::const_(0.21),
            fraction_grad: Default::default(),
        });
        let frac = Action::FractionTransfer(FractionTransfer {
            src: "W1".into(),
            dst: "W2".into(),
            fraction: Expr::const_(0.11),
            fraction_grad: Default::default(),
        });
        // A third fraction, not a count: a count beside another drain of W1 is
        // refused (`refuse_overdrawing_drains`).
        let frac3 = Action::FractionTransfer(FractionTransfer {
            src: "W1".into(),
            dst: "W2".into(),
            fraction: Expr::const_(0.13),
            fraction_grad: Default::default(),
        });
        let fwd = real_phase(vec![frac2.clone(), frac.clone(), frac3.clone()]).unwrap();
        let rev = real_phase(vec![frac3, frac, frac2]).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&fwd), bits(&rev), "forward {fwd:?} vs reversed {rev:?}");
        let base = 10.0_f64 / 3.0;
        let mut d = [-(base * 0.21), -(base * 0.11), -(base * 0.13)];
        d.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(fwd[0].to_bits(), d.iter().fold(base, |acc, &x| acc + x).to_bits());
    }

    /// A `set` beside a `transfer` into the same real compartment is refused.
    #[test]
    fn continuous_real_set_beside_a_transfer_into_it_is_refused() {
        let res = real_phase(vec![
            Action::Set(SetAction { compartment: "W2".into(), value: Expr::const_(1.0), value_grad: Default::default() }),
            Action::FractionTransfer(FractionTransfer {
                src: "W1".into(),
                dst: "W2".into(),
                fraction: Expr::const_(0.5),
                fraction_grad: Default::default(),
            }),
        ]);
        assert!(
            matches!(&res, Err(SimError::Validation(m)) if m.contains("compartment 'W2'")),
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
            fraction_grad: Default::default(),
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
                value_grad: Default::default(),
            })],
            kind: if always { ir::intervention::InterventionKind::Event } else { ir::intervention::InterventionKind::Scenario },
        };
        let m = Model {
            constants: vec![],
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
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 2.0,
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None, t_start_anchor: None,
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
                value_grad: Default::default(),
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
