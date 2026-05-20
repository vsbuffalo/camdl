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

use std::collections::HashMap;

use crate::error::SimError;

use super::event_log::{EventLog, EventRecord, RouteInfo};
use super::writer::{LineListEntry, LineListWriter};
use super::{CompartmentId, DemeId, IdentityState, IndividualId, LineageRng, ParentRef};

/// A frozen clone of the parent pools at the start of a batched step.
///
/// tau-leap / chain-binomial fire `k` events against rates and pools frozen at
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

/// Replay `log` at `identity_seed`, writing the realized line list to `writer`.
///
/// Returns the line-list log-probability, edge count, and sub-`dt` diagnostic.
/// The writer is initialized, written, and finalized within this call.
pub fn realize(
    log: &EventLog,
    identity_seed: u64,
    writer: &mut dyn LineListWriter,
) -> Result<RealizeSummary, SimError> {
    let mut rng = LineageRng::from_sim_seed(identity_seed);
    let mut identity = IdentityState::new();

    // Seed the t=0 tracked pools exactly as the shipped observer did
    // (declaration order → IDs 0.. minted in the same order).
    for &(deme, comp, count) in &log.initial_pools {
        identity.seed_pool(deme, comp, count);
    }

    writer.init()?;

    let mut total_logprob = 0.0;
    let mut edges: u64 = 0;
    let mut sub_dt_edges = 0.0;
    let mut any_batched = false;

    // Frozen start-of-step parent pools for the batched step currently being
    // replayed (`None` outside a batched step / for Gillespie). Refreshed when
    // the `step` index changes on a batched event.
    let mut snapshot: Option<StepSnapshot> = None;
    let mut snapshot_step: Option<u64> = None;

    for rec in &log.events {
        let route = &log.transitions[rec.transition];
        any_batched |= rec.batched;

        if rec.batched {
            // Take the snapshot once per batched step (at its first event).
            if snapshot_step != Some(rec.step) {
                snapshot = Some(StepSnapshot::of(&identity));
                snapshot_step = Some(rec.step);
            }
        } else {
            snapshot = None;
            snapshot_step = None;
        }

        // Sub-`dt` bias accounting, mirroring the shipped observer: only the
        // batched path accumulates mass. `p` is the destination pool size at
        // *step start* — read from the frozen snapshot so later events in a
        // multi-event step still see the start-of-step count (children minted
        // earlier in the step are excluded, exactly as the shipped observer's
        // snapshot did).
        if rec.lineage_weights.is_some() {
            edges += rec.multiplicity;
            if let (true, Some(dst), Some(snap)) = (rec.batched, route.destination, snapshot.as_ref())
            {
                let m = rec.multiplicity as f64;
                let p = snap.pool_len(route.destination_deme, dst) as f64;
                if p + m > 0.0 {
                    sub_dt_edges += m * (m / (p + m));
                }
            }
        }

        realize_event(
            rec,
            route,
            &mut identity,
            snapshot.as_ref(),
            &mut rng,
            writer,
            &mut total_logprob,
        )?;
    }

    writer.finish()?;

    let exact = !any_batched;
    let sub_dt_fraction = if edges == 0 {
        0.0
    } else {
        sub_dt_edges / edges as f64
    };

    Ok(RealizeSummary { total_logprob, edges, sub_dt_fraction, exact })
}

/// Realize one event's `multiplicity` firings: sample identities, mint/move IDs,
/// write entries, and accumulate the attribution log-probability.
fn realize_event(
    rec: &EventRecord,
    route: &RouteInfo,
    identity: &mut IdentityState,
    snapshot: Option<&StepSnapshot>,
    rng: &mut LineageRng,
    writer: &mut dyn LineListWriter,
    total_logprob: &mut f64,
) -> Result<(), SimError> {
    let source = route.source;
    let destination = route.destination;
    let child_deme = route.child_deme;

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
                    identity.push(route.destination_deme, dst, child);
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
                        identity.push(route.destination_deme, dst, id);
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
                        identity.push(route.destination_deme, dst, id);
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
                .map_or(0, |(g, _)| *g)
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
            chosen, chosen_deme
        )));
    }
    let idx = rng.below(pool_len);
    let parent = match snapshot {
        Some(snap) => snap.member(chosen_deme, chosen, idx).ok_or_else(|| {
            SimError::Validation(format!(
                "realize: snapshot parent pool (comp {}, deme {}) member {} out of range",
                chosen, chosen_deme, idx
            ))
        })?,
        None => identity.pool_member(chosen_deme, chosen, idx),
    };

    // §4a: P = (mass_b/Λ) · (1/|pool_b|) = w_b/Λ.
    let logp = chosen_mass.ln() - (pool_len as f64).ln() - total.ln();
    Ok((parent, chosen_deme, logp))
}
