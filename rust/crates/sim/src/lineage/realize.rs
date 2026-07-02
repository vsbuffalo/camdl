//! Layer 2 of the three-layer lineage architecture (2026-05-20 proposal):
//! **realize an event log into a line list**.
//!
//! Given an [`EventLog`] (Layer 1) and an `identity_seed`, [`realize`] replays
//! the recorded event sequence, maintains the per-pool identity state, and at
//! each event draws *which specific individuals* were involved — pool
//! `b ∝ w_b·X_b` then uniform within pool for transmissions; uniform within the
//! source pool for recoveries/progressions. It mints IDs and writes one
//! [`LineListEntry`] per event, carrying that event's attribution
//! log-probability (§4a). Different `identity_seed`s give i.i.d. draws from
//! `P(identities | event log)`.
//!
//! The sampling logic is **relocated verbatim** from the shipped inline
//! observer (`LineageObserver::sample_parent` and the per-firing routing): the
//! pool-then-individual draw order, the `swap_remove` within-pool removal, and
//! the mint/move/remove routing are identical, only their location moved from
//! the simulation hot loop to this offline replay. So a realize at identity
//! seed `s` reproduces the shipped observer's line list for the same seed.
//!
//! ## The attribution log-probability (§4a)
//!
//! Per event, with `Λ = Σ_b' w_b'·X_b'` the realized total FOI mass:
//! - **transmission**, parent in pool `b`:
//!   `log P = log(w_b·X_b / Λ) + log(1/X_b) = log(w_b·X_b) − log(X_b) − log Λ`,
//!   which equals `log(w_b/Λ)` — the recorded mass `w_b·X_b` divided by the
//!   live pool size `X_b` (= `|pool_b|`) and by `Λ`.
//! - **recovery / progression**: uniform within the source pool of size
//!   `|I_b|`, so `log P = −log|I_b|`.
//! - **import / seed / non-routable**: no attribution choice, `log P = 0`.
//!
//! Summed over the line list this is `log P(line list | event log)`, the only
//! clean exact likelihood (§4a). [`realize`] returns the running total
//! alongside the per-entry column.

use std::collections::{HashMap, HashSet};

use crate::error::SimError;

use super::event_log::{EventLog, EventRecord, RouteInfo};
use super::writer::{LineListEntry, LineListWriter};
use super::{CompartmentId, DemeId, IdentityState, IndividualId, LineageRng, ParentRef};

/// A frozen clone of the parent pools at the start of a batched step.
///
/// chain-binomial fires `k` events against rates and pools frozen at
/// step start. The shipped observer sampled all `k` *parents* from a snapshot
/// taken at step start, so a child minted earlier in the step is invisible as a
/// same-step parent (proposal §11 open-question 3). Realize reproduces this:
/// during a batched step it reads parent-pool membership (and the within-pool
/// size used in the §4a `1/X_b`) from this snapshot; removals and mints still
/// apply to the live [`IdentityState`]. Gillespie events are each their own
/// step (one event), so they sample the live pool with no snapshot.
struct StepSnapshot {
    pools: HashMap<(DemeId, CompartmentId), Vec<IndividualId>>,
}

impl StepSnapshot {
    fn of(identity: &IdentityState) -> Self {
        StepSnapshot { pools: identity.pools_clone() }
    }
    fn pool_len(&self, deme: DemeId, comp: CompartmentId) -> usize {
        self.pools.get(&(deme, comp)).map_or(0, |v| v.len())
    }
    fn member(&self, deme: DemeId, comp: CompartmentId, idx: usize) -> Option<IndividualId> {
        self.pools.get(&(deme, comp)).and_then(|v| v.get(idx).copied())
    }
}

/// Outcome of a realize pass over an event log.
#[derive(Debug, Clone)]
pub struct RealizeSummary {
    /// `log P(line list | event log)` (§4a): the sum of every entry's
    /// `attribution_logprob`.
    pub total_logprob: f64,
    /// Number of transmission (lineage) edges realized.
    pub edges: u64,
    /// Edge-weighted fraction of transmission edges whose sub-`dt` ordering the
    /// frozen-pool (batched-backend) approximation could not resolve. Exactly
    /// 0.0 for an event log recorded under Gillespie (no batched events).
    pub sub_dt_fraction: f64,
    /// `true` if the event log was recorded by an exact backend (no batched
    /// events) — the realized line list has no sub-`dt` bias.
    pub exact: bool,
}

/// Compartments whose identity pool is ever *read* during replay, so their IDs
/// must be retained. Reads happen at exactly three sites:
///   - source removals (`remove_uniform` / `pool_len` on `route.source`),
///   - parent sampling (`route.parent_pools`),
///   - the sub-`dt` diagnostic, which reads a *lineage* transition's destination
///     pool size at step start.
///
/// A compartment that is none of these (e.g. an absorbing `R` reached only by a
/// non-lineage recovery) is **write-only**: pushing IDs into it grows the pool
/// to O(cumulative arrivals) for no benefit. `realize_event` skips those pushes.
/// The individual is still minted and recorded in the line list — only the
/// never-read pool membership is dropped, so the realized output is unchanged.
fn readable_compartments(routes: &[RouteInfo]) -> HashSet<CompartmentId> {
    let mut readable = HashSet::new();
    for route in routes {
        if let Some(src) = route.source {
            readable.insert(src);
        }
        for &(comp, _deme) in &route.parent_pools {
            readable.insert(comp);
        }
        // A lineage transition (non-empty parent pools) has its destination pool
        // read by the sub-`dt` diagnostic in `realize`.
        if !route.parent_pools.is_empty() {
            if let Some(dst) = route.destination {
                readable.insert(dst);
            }
        }
    }
    readable
}

/// Mutable replay state, threaded through every recorded event. Shared by the
/// in-memory [`realize`] and the streaming [`realize_from_path`] entry points so
/// the per-event attribution logic has a single source of truth regardless of
/// whether the event log is materialised in RAM or streamed from disk.
struct RealizeState {
    rng: LineageRng,
    identity: IdentityState,
    readable: HashSet<CompartmentId>,
    total_logprob: f64,
    edges: u64,
    sub_dt_edges: f64,
    any_batched: bool,
    /// Frozen start-of-step parent pools for the batched step currently being
    /// replayed (`None` outside a batched step / for Gillespie). Refreshed when
    /// the `step` index changes on a batched event.
    snapshot: Option<StepSnapshot>,
    snapshot_step: Option<u64>,
    /// The `(time, step)` of the most-recently processed event. Every recorded
    /// event MUST be non-decreasing in this pair (the "recorded time order"
    /// precondition the snapshot logic and RNG draws depend on); [`process`]
    /// hard-errors on a regression rather than silently miscomputing. Seeded at
    /// `(-inf, 0)` so the first event always passes.
    last_time: f64,
    last_step: u64,
}

impl RealizeState {
    fn new(
        identity_seed: u64,
        initial_pools: &[(DemeId, CompartmentId, i64)],
        transitions: &[RouteInfo],
    ) -> Self {
        let mut identity = IdentityState::new();
        // Seed the t=0 tracked pools exactly as the shipped observer did
        // (declaration order → IDs 0.. minted in the same order).
        for &(deme, comp, count) in initial_pools {
            identity.seed_pool(deme, comp, count);
        }
        RealizeState {
            rng: LineageRng::from_sim_seed(identity_seed),
            identity,
            readable: readable_compartments(transitions),
            total_logprob: 0.0,
            edges: 0,
            sub_dt_edges: 0.0,
            any_batched: false,
            snapshot: None,
            snapshot_step: None,
            last_time: f64::NEG_INFINITY,
            last_step: 0,
        }
    }

    /// Process one recorded event in order: refresh the batched-step snapshot,
    /// accumulate the sub-`dt` diagnostic, and realize the event's identities.
    /// Events MUST be fed in recorded (time) order — the same order the
    /// simulator produced them — for the snapshot logic and RNG draws to match
    /// the shipped observer.
    fn process(
        &mut self,
        rec: &EventRecord,
        transitions: &[RouteInfo],
        writer: &mut dyn LineListWriter,
    ) -> Result<(), SimError> {
        // Guard the "recorded time order" precondition (H12): the snapshot
        // refresh, the sub-`dt` accounting, and the RNG draw order all assume
        // events arrive in non-decreasing `(time, step)` order. A user-edited
        // TSV, shuffled Parquet row groups, or a writer regression would
        // otherwise silently miscompute — so reject a regression outright rather
        // than trust the file order.
        if rec.time < self.last_time
            || (rec.time == self.last_time && rec.step < self.last_step)
        {
            return Err(SimError::Validation(format!(
                "realize: event log is out of recorded order — event at (time {}, \
                 step {}) regresses below the previous event at (time {}, step {}). \
                 Events must be replayed in non-decreasing (time, step) order (the \
                 order the simulator recorded them); a reordered or hand-edited log \
                 would miscompute at-step snapshots and parent draws.",
                rec.time, rec.step, self.last_time, self.last_step
            )));
        }
        self.last_time = rec.time;
        self.last_step = rec.step;

        let route = &transitions[rec.transition.0];
        self.any_batched |= rec.batched;

        if rec.batched {
            // Take the snapshot once per batched step (at its first event).
            if self.snapshot_step != Some(rec.step) {
                self.snapshot = Some(StepSnapshot::of(&self.identity));
                self.snapshot_step = Some(rec.step);
            }
        } else {
            self.snapshot = None;
            self.snapshot_step = None;
        }

        // Sub-`dt` bias accounting, mirroring the shipped observer: only the
        // batched path accumulates mass. `p` is the destination pool size at
        // *step start* — read from the frozen snapshot so later events in a
        // multi-event step still see the start-of-step count (children minted
        // earlier in the step are excluded, exactly as the shipped observer's
        // snapshot did).
        if rec.lineage_weights.is_some() {
            self.edges += rec.multiplicity;
            if let (true, Some(dst), Some(snap)) =
                (rec.batched, route.destination, self.snapshot.as_ref())
            {
                let m = rec.multiplicity as f64;
                let p = snap.pool_len(route.destination_deme, dst) as f64;
                if p + m > 0.0 {
                    self.sub_dt_edges += m * (m / (p + m));
                }
            }
        }

        realize_event(
            rec,
            route,
            &mut self.identity,
            &self.readable,
            self.snapshot.as_ref(),
            &mut self.rng,
            writer,
            &mut self.total_logprob,
        )
    }

    fn finish(self) -> RealizeSummary {
        let exact = !self.any_batched;
        let sub_dt_fraction = if self.edges == 0 {
            0.0
        } else {
            self.sub_dt_edges / self.edges as f64
        };
        RealizeSummary {
            total_logprob: self.total_logprob,
            edges: self.edges,
            sub_dt_fraction,
            exact,
        }
    }
}

/// Replay an in-memory `log` at `identity_seed`, writing the realized line list
/// to `writer`.
///
/// Returns the line-list log-probability, edge count, and sub-`dt` diagnostic.
/// The writer is initialized, written, and finalized within this call.
pub fn realize(
    log: &EventLog,
    identity_seed: u64,
    writer: &mut dyn LineListWriter,
) -> Result<RealizeSummary, SimError> {
    let mut state = RealizeState::new(identity_seed, &log.initial_pools, &log.transitions);
    writer.init()?;
    for rec in &log.events {
        state.process(rec, &log.transitions, writer)?;
    }
    writer.finish()?;
    Ok(state.finish())
}

/// Replay an event log read **incrementally from disk** at `identity_seed`,
/// writing the realized line list to `writer`.
///
/// The event log is never fully materialised in RAM: the metadata (route table,
/// t=0 pools) is read from the file's footer/header, then events are streamed in
/// recorded order (Parquet row group by row group, TSV line by line) and fed to
/// the same per-event logic as [`realize`]. Resident memory is bounded by the
/// identity pools, not the log length. Format is auto-detected by extension.
pub fn realize_from_path(
    path: &std::path::Path,
    identity_seed: u64,
    writer: &mut dyn LineListWriter,
) -> Result<RealizeSummary, SimError> {
    let (initial_pools, transitions) = super::event_log_io::read_metadata(path)?;
    let mut state = RealizeState::new(identity_seed, &initial_pools, &transitions);
    writer.init()?;
    super::event_log_io::for_each_event(path, |rec| {
        state.process(&rec, &transitions, writer)
    })?;
    writer.finish()?;
    Ok(state.finish())
}

/// Realize one event's `multiplicity` firings: sample identities, mint/move IDs,
/// write entries, and accumulate the attribution log-probability.
fn realize_event(
    rec: &EventRecord,
    route: &RouteInfo,
    identity: &mut IdentityState,
    readable: &HashSet<CompartmentId>,
    snapshot: Option<&StepSnapshot>,
    rng: &mut LineageRng,
    writer: &mut dyn LineListWriter,
    total_logprob: &mut f64,
) -> Result<(), SimError> {
    let source = route.source;
    let destination = route.destination;
    let child_deme = route.child_deme;
    // Only retain IDs in pools that are ever read (see `readable_compartments`).
    // Pushing into a write-only pool (e.g. absorbing R) is a pure memory leak.
    let push_if_readable = |identity: &mut IdentityState, deme: DemeId, comp: CompartmentId, id: IndividualId| {
        if readable.contains(&comp) {
            identity.push(deme, comp, id);
        }
    };

    for _ in 0..rec.multiplicity {
        let (individual, parent, parent_deme, src_for_record, dst_for_record, logp) =
            if let Some(masses) = &rec.lineage_weights {
                // Transmission: sample the parent pool `b ∝ w_b·X_b`, then
                // uniform within pool. Accumulate `log(w_b/Λ)`. During a batched
                // step the within-pool member + size come from the frozen
                // snapshot (same-step children invisible as parents).
                let (parent_id, parent_deme, logp) =
                    sample_parent(masses, route, identity, snapshot, rng)?;

                // The source individual (e.g. an S) is consumed if tracked; the
                // child is a fresh ID in the destination. A non-empty source
                // pool means the source compartment is tracked and populated —
                // matches the shipped observer's `remove_uniform` on a tracked
                // source at a lineage event (rare — S is usually untracked). The
                // removal draws from the lineage RNG, preserving draw order.
                if let Some(src) = source {
                    if identity.pool_len(route.source_deme, src) > 0 {
                        let _ = identity.remove_uniform(route.source_deme, src, rng);
                    }
                }
                let child = identity.mint();
                if let Some(dst) = destination {
                    push_if_readable(identity, route.destination_deme, dst, child);
                }
                (
                    child,
                    ParentRef::Individual(parent_id),
                    Some(parent_deme),
                    source,
                    destination,
                    logp,
                )
            } else {
                match (source, destination) {
                    (Some(src), Some(dst)) => {
                        // Progression: move one ID from source → destination.
                        // Uniform within the source pool → `log(1/|src|)`.
                        let n = identity.pool_len(route.source_deme, src);
                        let (id, logp) = match identity.remove_uniform(route.source_deme, src, rng) {
                            Some(id) => (id, -(n as f64).ln()),
                            None => (identity.mint(), 0.0),
                        };
                        push_if_readable(identity, route.destination_deme, dst, id);
                        (id, ParentRef::None, None, Some(src), Some(dst), logp)
                    }
                    (Some(src), None) => {
                        // Outflow (death / removal): uniform within the source
                        // pool → `log(1/|src|)`.
                        let n = identity.pool_len(route.source_deme, src);
                        let (id, logp) = match identity.remove_uniform(route.source_deme, src, rng) {
                            Some(id) => (id, -(n as f64).ln()),
                            None => (identity.mint(), 0.0),
                        };
                        (id, ParentRef::None, None, Some(src), None, logp)
                    }
                    (None, Some(dst)) => {
                        // Inflow (import): mint a new ID, no parent, no choice.
                        let id = identity.mint();
                        push_if_readable(identity, route.destination_deme, dst, id);
                        (id, ParentRef::Import, None, None, Some(dst), 0.0)
                    }
                    (None, None) => {
                        // Nothing routable — should not happen for a tracked
                        // transition; emit no record rather than guess.
                        continue;
                    }
                }
            };

        *total_logprob += logp;
        writer.write(&LineListEntry {
            time: rec.time,
            transition: rec.transition,
            individual,
            source: src_for_record,
            destination: dst_for_record,
            deme: child_deme,
            parent,
            parent_deme,
            attribution_logprob: logp,
        })?;
    }

    Ok(())
}

/// Sample a parent: pool `b ∝ mass_b` (the recorded `w_b·X_b`), then uniform
/// within the live pool `b`. Returns `(parent_id, parent_deme, log P)` with
/// `log P = log(mass_b) − log(|pool_b|) − log Λ = log(w_b / Λ)` (§4a).
///
/// Draw order — `rng.uniform()` for the pool, then `rng.below(len)` for the
/// individual — is identical to the shipped `LineageObserver::sample_parent`,
/// so the realized line list matches the shipped observer at the same seed.
fn sample_parent(
    masses: &[f64],
    route: &RouteInfo,
    identity: &IdentityState,
    snapshot: Option<&StepSnapshot>,
    rng: &mut LineageRng,
) -> Result<(IndividualId, DemeId, f64), SimError> {
    debug_assert_eq!(
        masses.len(),
        route.parent_pools.len(),
        "recorded lineage_weights must align with the route's parent_pools"
    );

    let total: f64 = masses.iter().sum();
    if total <= 0.0 {
        return Err(SimError::Validation(format!(
            "lineage event for transition {} has zero total parent mass Λ; the \
             event log's recorded weights are degenerate (every w_b·X_b = 0)",
            // transition id is not on the route; report the candidate pools.
            route
                .parent_pools
                .first()
                .map_or(0, |(g, _)| g.0)
        )));
    }

    // Pool selection by cumulative mass against a uniform draw — identical to
    // the shipped observer (default to the last pool, matching its fallback).
    let u = rng.uniform() * total;
    let mut cumulative = 0.0;
    let last = route.parent_pools[route.parent_pools.len() - 1];
    let mut chosen = last.0;
    let mut chosen_deme = last.1;
    let mut chosen_mass = masses[masses.len() - 1];
    for ((g, deme), &mass) in route.parent_pools.iter().zip(masses.iter()) {
        cumulative += mass;
        if cumulative >= u {
            chosen = *g;
            chosen_deme = *deme;
            chosen_mass = mass;
            break;
        }
    }

    // Within-pool size + member: the frozen snapshot during a batched step
    // (start-of-step membership; same-step children excluded), the live pool
    // for Gillespie. The size used here is the same `X_b` baked into the
    // recorded `mass_b = w_b·X_b`, so the `1/X_b` cancels to give `w_b/Λ` (§4a).
    let pool_len = match snapshot {
        Some(snap) => snap.pool_len(chosen_deme, chosen),
        None => identity.pool_len(chosen_deme, chosen),
    };
    if pool_len == 0 {
        return Err(SimError::Validation(format!(
            "realize: chosen parent pool (comp {}, deme {}) is empty — the event \
             log's recorded weights have diverged from the replayed pool state",
            chosen.0, chosen_deme.0
        )));
    }
    let idx = rng.below(pool_len);
    let parent = match snapshot {
        Some(snap) => snap.member(chosen_deme, chosen, idx).ok_or_else(|| {
            SimError::Validation(format!(
                "realize: snapshot parent pool (comp {}, deme {}) member {} out of range",
                chosen.0, chosen_deme.0, idx
            ))
        })?,
        None => identity.pool_member(chosen_deme, chosen, idx),
    };

    // §4a: P = (mass_b/Λ) · (1/|pool_b|) = w_b/Λ.
    let logp = chosen_mass.ln() - (pool_len as f64).ln() - total.ln();
    Ok((parent, chosen_deme, logp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::TransitionId;

    fn route(source: Option<CompartmentId>, dest: Option<CompartmentId>, parents: Vec<(CompartmentId, DemeId)>) -> RouteInfo {
        RouteInfo {
            source,
            source_deme: DemeId(0),
            destination: dest,
            destination_deme: DemeId(0),
            child_deme: DemeId(0),
            touches_tracked: true,
            parent_pools: parents,
        }
    }

    /// A [`LineListWriter`] that drops everything — the realize tests exercise
    /// the replay state machine, not the output format.
    struct NullWriter;
    impl LineListWriter for NullWriter {
        fn init(&mut self) -> Result<(), SimError> {
            Ok(())
        }
        fn write(&mut self, _entry: &LineListEntry) -> Result<(), SimError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SimError> {
            Ok(())
        }
    }

    /// H12: replay must reject an event log whose events regress in `(time,
    /// step)` order. A hand-edited TSV or shuffled Parquet row groups would
    /// otherwise silently miscompute snapshots and parent draws.
    #[test]
    fn realize_rejects_out_of_time_order_events() {
        use super::super::event_log::{EventLog, EventRecord};
        // One non-lineage progression route; two events recorded out of order
        // (t=2.0 then t=1.0). The first processes; the second must hard-error.
        let log = EventLog {
            initial_pools: vec![],
            transitions: vec![route(Some(CompartmentId(0)), Some(CompartmentId(1)), vec![])],
            events: vec![
                EventRecord {
                    time: 2.0,
                    transition: TransitionId(0),
                    multiplicity: 1,
                    batched: false,
                    step: 2,
                    lineage_weights: None,
                },
                EventRecord {
                    time: 1.0,
                    transition: TransitionId(0),
                    multiplicity: 1,
                    batched: false,
                    step: 1,
                    lineage_weights: None,
                },
            ],
        };
        let mut writer = NullWriter;
        let err = realize(&log, 0, &mut writer).expect_err("out-of-order log must be rejected");
        assert!(
            matches!(err, SimError::Validation(_)),
            "expected SimError::Validation, got {:?}",
            err
        );
    }

    /// SIR: infection (lineage, S→I, parent pool I) + recovery (I→R, non-lineage).
    /// Readable = sources ∪ parent pools ∪ lineage destinations = {S, I}; the
    /// absorbing R (a write-only recovery destination) is excluded.
    #[test]
    fn readable_excludes_absorbing_recovery_compartment() {
        let s = CompartmentId(0);
        let i = CompartmentId(1);
        let r = CompartmentId(2);
        let routes = vec![
            route(Some(s), Some(i), vec![(i, DemeId(0))]), // #[lineage] infection
            route(Some(i), Some(r), vec![]),               // recovery
        ];
        let readable = readable_compartments(&routes);
        assert!(readable.contains(&s), "infection source S must be readable");
        assert!(readable.contains(&i), "I is a parent pool / lineage destination");
        assert!(!readable.contains(&r), "absorbing R is write-only and must be excluded");
    }

    /// Waning immunity (R→S) makes R a source, so R becomes readable and its
    /// pool must be retained.
    #[test]
    fn readable_includes_recovery_compartment_when_it_is_a_source() {
        let s = CompartmentId(0);
        let i = CompartmentId(1);
        let r = CompartmentId(2);
        let routes = vec![
            route(Some(s), Some(i), vec![(i, DemeId(0))]),
            route(Some(i), Some(r), vec![]),
            route(Some(r), Some(s), vec![]), // waning: R→S
        ];
        let readable = readable_compartments(&routes);
        assert!(readable.contains(&r), "R is a source under waning immunity → readable");
    }
}
