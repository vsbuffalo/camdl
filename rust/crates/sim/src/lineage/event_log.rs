//! Layer 1 of the three-layer lineage architecture (2026-05-20 proposal):
//! the **event log** and the **event recorder**.
//!
//! The simulation fixes the *count* trajectory and the ordered event sequence;
//! it draws **no** identities. The [`EventRecorder`] is a [`TransitionObserver`]
//! that, at every firing, records an [`EventRecord`] — and at `#[lineage]`
//! events additionally records the *evaluated* per-pool FOI masses
//! `{(b, w_b·X_b)}`. It mints no individual IDs and consumes no randomness, so
//! a `--event-log` run's count trajectory is byte-identical to a plain run at
//! the same seed (Tier 2a, now trivially true: the simulation is literally
//! unchanged).
//!
//! ## Self-containment
//!
//! The event log is the canonical Layer-1 artifact and is **self-contained**:
//! Layer-2 replay ([`super::realize`]) needs only the event log, not the model
//! or the rate AST. To honor that, the log records — alongside the proposal's
//! §5.1 `EventRecord` sketch — a per-transition [`RouteInfo`] table (source /
//! destination / child deme / candidate parent pools) and the t=0 tracked-pool
//! seeding. These are the structural facts replay would otherwise have to read
//! off the model; recording them once (the route table is one row per
//! transition; the weights are sparse, only at lineage events) keeps the log
//! cheap while making it model-free. This refines the §5.1 sketch
//! (`lineage_weights: Option<Vec<(CompartmentId, f64)>>`) by pairing each
//! candidate pool with its deme in the route table, because in the
//! fully-expanded IR a compartment's deme is otherwise derivable only from the
//! model.
//!
//! ## Pool count == compartment count
//!
//! The recorder reads the per-pool count `X_b` directly from the integer state
//! `pre_int` passed to [`TransitionObserver::on_fired`]. For Gillespie that is
//! the event-instant state; for the batched backends it is the frozen
//! start-of-step state. In both cases it equals the identity-pool size the
//! shipped observer read from its live pool / snapshot — because the pools
//! always mirror the counts exactly. So `w_b·X_b` is computed without any
//! identity bookkeeping.

use serde::{Deserialize, Serialize};

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::propensity::{eval_expr, EvalCtx};
use crate::state::{IntState, RealState};

use super::deme::DemeMap;
use super::{CompartmentId, DemeId, TransitionId, TransitionObserver};

/// Per-transition structural routing, recorded once in the event log so replay
/// is model-free. Indexed by [`TransitionId`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteInfo {
    /// Global compartment id of the source (the `-1` stoichiometry), if any.
    pub source: Option<CompartmentId>,
    /// Deme (stratum) of the source compartment (0 if `source` is `None`).
    pub source_deme: DemeId,
    /// Global compartment id of the destination (the `+1` stoichiometry), if any.
    pub destination: Option<CompartmentId>,
    /// Deme of the destination compartment (0 if `destination` is `None`).
    pub destination_deme: DemeId,
    /// The focal (child) individual's deme: destination stratum for an
    /// inflow/move, else the source stratum for an outflow.
    pub child_deme: DemeId,
    /// `true` if this transition's source/destination/parent pools touch a
    /// tracked compartment (an event for it is recorded).
    pub touches_tracked: bool,
    /// Candidate parent pools `(b, deme_of(b))` for a `#[lineage]` event, in the
    /// order the rate's parent-pool weights were emitted. Empty for a
    /// non-lineage transition. The recorded [`EventRecord::lineage_weights`]
    /// masses are aligned to this order.
    pub parent_pools: Vec<(CompartmentId, DemeId)>,
}

/// One recorded simulation event. Identity-free.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub time: f64,
    pub transition: TransitionId,
    /// Number of identical firings: always 1 for Gillespie, ≥ 1 for the
    /// batched backends (tau-leap / chain-binomial).
    pub multiplicity: u64,
    /// `true` if this event fired inside a batched (frozen-pool) step. Replay
    /// uses this to reproduce the shipped sub-`dt` bias accounting and to
    /// sample all `k` attributions against the start-of-step pools.
    pub batched: bool,
    /// Monotone batched-step index. All events sharing a `step` value fired in
    /// the same tau-leap / chain-binomial step and must, in replay, sample
    /// their identity attributions against the **frozen start-of-step** identity
    /// pools (mirroring the shipped observer's per-step pool snapshot — see
    /// proposal §11 open-question 3: "all attributions in a batched step use
    /// start-of-step pool membership"). Gillespie events are each their own
    /// step (one event per step), so this never groups them. `0` for the first
    /// step / for un-batched events that precede any batched step.
    pub step: u64,
    /// `Some` only at `#[lineage]` events: the evaluated per-pool FOI masses
    /// `w_b·X_b`, aligned to the transition's [`RouteInfo::parent_pools`]. The
    /// realized total `Λ = Σ_b' w_b'·X_b'` is their sum. `None` otherwise.
    pub lineage_weights: Option<Vec<f64>>,
}

/// The canonical Layer-1 artifact: the initial tracked-pool seeding plus the
/// ordered event sequence and the per-transition route table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLog {
    /// Initial tracked-compartment pools to seed at t=0, as
    /// `(deme, compartment, count)`. Mirrors the shipped observer's t=0 seeding
    /// (tracked compartments, in declaration order, each in its own stratum).
    pub initial_pools: Vec<(DemeId, CompartmentId, i64)>,
    /// Per-transition routing, indexed by [`TransitionId`].
    pub transitions: Vec<RouteInfo>,
    /// The ordered event sequence.
    pub events: Vec<EventRecord>,
}

/// The Layer-1 observer: turns the simulation's firings into an [`EventLog`].
/// Draws no randomness and mints no IDs.
pub struct EventRecorder<'m> {
    model: &'m CompiledModel,
    routes: Vec<RouteInfo>,
    /// Resolved parent-pool weight expressions, parallel to `routes` (the
    /// expression for each `parent_pools` entry of a lineage transition).
    parent_weight_exprs: Vec<Option<Vec<ir::expr::Expr>>>,
    initial_pools: Vec<(DemeId, CompartmentId, i64)>,
    events: Vec<EventRecord>,
    /// Active during a batched step (set by `begin_batch_step`); gates the
    /// `batched` flag on recorded events.
    in_batch: bool,
    /// Monotone step counter. Incremented at every `begin_batch_step` (one per
    /// batched step) and, for Gillespie, once per recorded event (so each
    /// Gillespie event is its own step — they never share frozen pools).
    step: u64,
}

impl<'m> EventRecorder<'m> {
    /// Build the recorder and capture the t=0 tracked-pool seeding.
    ///
    /// Mirrors [`super::LineageObserver::new`]: resolves tracked compartments,
    /// precomputes per-transition routing, and records the initial pools — but
    /// owns no identity state and no RNG.
    pub fn new(model: &'m CompiledModel, initial_int: &IntState) -> Result<Self, SimError> {
        let deme_map = DemeMap::build(&model.model, &model.comp_index);

        // Resolve tracked-compartment names → global indices.
        let mut tracked: Vec<CompartmentId> = Vec::new();
        for name in &model.model.identity_tracked_compartments {
            let g = model
                .comp_index
                .get(name.as_str())
                .copied()
                .ok_or_else(|| SimError::UnknownCompartment(name.clone()))?;
            tracked.push(g);
        }
        let is_tracked = |g: CompartmentId| tracked.contains(&g);

        let mut routes = Vec::with_capacity(model.model.transitions.len());
        let mut parent_weight_exprs = Vec::with_capacity(model.model.transitions.len());
        for (tr_idx, tr) in model.model.transitions.iter().enumerate() {
            let mut source: Option<CompartmentId> = None;
            let mut destination: Option<CompartmentId> = None;
            for &(local, delta) in &model.transition_stoich[tr_idx] {
                let g = model.int_local_to_global[local];
                if delta < 0 && source.is_none() {
                    source = Some(g);
                } else if delta > 0 && destination.is_none() {
                    destination = Some(g);
                }
            }

            // Lineage parent-pool decomposition: `(b, deme_of(b))` candidates +
            // their weight expressions (parallel arrays).
            let (parent_pools, weight_exprs) = match &tr.lineage {
                Some(l) if l.is_lineage_event => {
                    let mut pools = Vec::with_capacity(l.parent_pool_weights.len());
                    let mut exprs = Vec::with_capacity(l.parent_pool_weights.len());
                    for (comp_name, weight) in &l.parent_pool_weights {
                        let g = model
                            .comp_index
                            .get(comp_name.as_str())
                            .copied()
                            .ok_or_else(|| SimError::UnknownCompartment(comp_name.clone()))?;
                        pools.push((g, deme_map.deme_of(g)));
                        exprs.push(weight.clone());
                    }
                    (pools, Some(exprs))
                }
                _ => (Vec::new(), None),
            };

            let source_deme = source.map_or(0, |g| deme_map.deme_of(g));
            let destination_deme = destination.map_or(0, |g| deme_map.deme_of(g));
            let child_deme = match (source, destination) {
                (_, Some(dst)) => deme_map.deme_of(dst),
                (Some(src), None) => deme_map.deme_of(src),
                (None, None) => 0,
            };

            let touches_tracked = source.is_some_and(is_tracked)
                || destination.is_some_and(is_tracked)
                || parent_pools.iter().any(|(g, _)| is_tracked(*g));

            routes.push(RouteInfo {
                source,
                source_deme,
                destination,
                destination_deme,
                child_deme,
                touches_tracked,
                parent_pools,
            });
            parent_weight_exprs.push(weight_exprs);
        }

        // t=0 seeding: tracked compartments, in declaration order, with their
        // initial counts, each in its own stratum's deme.
        let mut initial_pools = Vec::new();
        for &g in &tracked {
            if let Some(local) = model.global_to_int[g] {
                let count = initial_int.counts[local];
                if count > 0 {
                    initial_pools.push((deme_map.deme_of(g), g, count));
                }
            }
        }

        Ok(EventRecorder {
            model,
            routes,
            parent_weight_exprs,
            initial_pools,
            events: Vec::new(),
            in_batch: false,
            step: 0,
        })
    }

    /// Consume the recorder and return the completed event log.
    pub fn into_event_log(self) -> EventLog {
        EventLog {
            initial_pools: self.initial_pools,
            transitions: self.routes,
            events: self.events,
        }
    }
}

impl TransitionObserver for EventRecorder<'_> {
    fn begin_batch_step(&mut self) {
        // A new batched step: all events recorded until `end_batch_step` share
        // this step index and (in replay) the same frozen start-of-step pools.
        self.in_batch = true;
        self.step = self.step.wrapping_add(1);
    }

    fn end_batch_step(&mut self) {
        self.in_batch = false;
    }

    #[allow(clippy::too_many_arguments)]
    fn on_fired(
        &mut self,
        transition: TransitionId,
        _deme: DemeId,
        multiplicity: u64,
        time: f64,
        pre_int: &IntState,
        pre_real: &RealState,
        params: &[f64],
    ) -> Result<(), SimError> {
        let route = &self.routes[transition];
        if !route.touches_tracked {
            // Untracked transition — no event recorded, no overhead beyond the
            // flag check (matches the shipped observer's early return).
            return Ok(());
        }

        // At a lineage event, evaluate and record the per-pool FOI masses
        // `w_b·X_b` against the event-instant (Gillespie) / start-of-step
        // (batched) state. `X_b` is read from `pre_int`, which equals the
        // identity-pool count the shipped observer sampled against.
        let lineage_weights = if let Some(exprs) = &self.parent_weight_exprs[transition] {
            let ctx = EvalCtx {
                model: self.model,
                int_s: pre_int,
                real_s: pre_real,
                params,
                t: time,
                dt: self.model.model.simulation.dt.unwrap_or(1.0),
                projected: None,
                int_float_override: None,
            };
            let mut masses = Vec::with_capacity(exprs.len());
            for ((g, _deme), weight_expr) in route.parent_pools.iter().zip(exprs.iter()) {
                let w = eval_expr(weight_expr, &ctx)?.max(0.0);
                let count = match self.model.global_to_int[*g] {
                    Some(local) => pre_int.counts[local].max(0) as f64,
                    None => 0.0,
                };
                masses.push(w * count);
            }
            Some(masses)
        } else {
            None
        };

        // Step index: a batched step's events all share `self.step` (set at
        // `begin_batch_step`). A Gillespie event is its own step, so bump the
        // counter per recorded event — they never share frozen pools.
        if !self.in_batch {
            self.step = self.step.wrapping_add(1);
        }
        self.events.push(EventRecord {
            time,
            transition,
            multiplicity,
            batched: self.in_batch,
            step: self.step,
            lineage_weights,
        });
        Ok(())
    }
}

impl EventLog {
    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` if no events were recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
