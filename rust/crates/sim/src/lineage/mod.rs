//! Individual-sampling (lineage) layer — runtime + offline projection.
//!
//! Implements the Phase-1 slice of the 2026-05-19 individual-sampling-layer
//! proposal (`docs/dev/proposals/2026-05-19-individual-sampling-layer.md`):
//! Gillespie + single-population linear-rate lineage tracking, a streamed
//! append-only line-list writer (TSV + Parquet), and a pure offline pruner
//! that projects a line list to a transmission tree in Newick form.
//!
//! ## The load-bearing invariant: a separate RNG stream
//!
//! Identity-attribution draws (which parent pool, which individual within the
//! pool) come from [`LineageRng`], an *independent* ChaCha8 stream seeded
//! `main_seed ⊕ LINEAGE_RNG_OFFSET`. The simulation's own `StatefulRng` is
//! never touched by the observer. Consequence: a run with `--lineages`
//! produces a count trajectory byte-identical to the same run without it
//! (validation Tier 2a). The observer's [`TransitionObserver::on_fired`] is
//! called by the core loop *after* it has drawn its own RNG and decided what
//! fired, so it cannot reorder the simulation RNG.
//!
//! ## Scope of this slice
//!
//! Gillespie + single population only. tau-leap / chain-binomial declare the
//! [`crate::Capabilities::LINEAGES`] flag (so the capability check passes) but
//! tracking is not yet wired into their loops — requesting `--lineages` on
//! those backends returns a clear "not yet implemented" error at the CLI.
//! Stratified / multi-deme attribution is Phase 2.

pub mod writer;
pub mod tree;

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::propensity::{eval_expr, EvalCtx};
use crate::state::{IntState, RealState};

pub use writer::{LineListEntry, LineListFormat, LineListWriter, TsvLineListWriter};
#[cfg(feature = "lineage-parquet")]
pub use writer::ParquetLineListWriter;

/// Fixed XOR offset that derives the lineage RNG seed from the simulation
/// seed. Chosen as an arbitrary fixed 64-bit constant; the only requirement
/// is that it be a constant so the lineage stream is reproducible, and
/// non-zero so it visibly differs from the main seed.
pub const LINEAGE_RNG_OFFSET: u64 = 0x5ca1ab1e_d15ea5e5;

/// Monotone per-run individual identifier. Minted from a counter starting at 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndividualId(pub u64);

/// Deme (spatial/stratum) identifier. Always 0 in the single-population slice;
/// carried through the line list so Phase 2 can populate it without a schema
/// change.
pub type DemeId = u32;

/// Compartment identifier — the *global* compartment index in the model's
/// combined compartment list (matches `CompiledModel::comp_index` values).
pub type CompartmentId = usize;

/// Transition identifier — the index into `model.transitions`.
pub type TransitionId = usize;

/// Where a tracked individual came from. Recorded per line-list entry; the
/// `Individual` variant is what makes a transmission tree reconstructable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentRef {
    /// A specific tracked individual (the infector at a lineage event).
    Individual(IndividualId),
    /// Exogenous import — minted with no parent (inflow with no source).
    Import,
    /// Initial population seeded at t=0.
    Seed,
    /// Non-lineage event (progression, recovery, death): the focal individual
    /// already exists; no parent is attributed.
    None,
}

/// Independent RNG stream for identity attribution. Wraps ChaCha8 exactly like
/// [`crate::rng::StatefulRng`] but is seeded from `main_seed ⊕
/// LINEAGE_RNG_OFFSET` and is owned solely by the lineage subsystem, so it
/// never perturbs the simulation's draw order.
pub struct LineageRng(ChaCha8Rng);

impl LineageRng {
    /// Derive the lineage stream from the simulation seed.
    pub fn from_sim_seed(sim_seed: u64) -> Self {
        // Reuse the same seed-expansion as StatefulRng so the byte layout is
        // consistent, but XOR in the fixed offset first to make the stream
        // disjoint from the simulation stream.
        let mixed = sim_seed ^ LINEAGE_RNG_OFFSET;
        let seed_bytes = expand_u64_to_seed(mixed.wrapping_add(0xdeadbeef_cafebabe));
        LineageRng(ChaCha8Rng::from_seed(seed_bytes))
    }

    /// Uniform [0, 1).
    pub fn uniform(&mut self) -> f64 {
        use rand::Rng;
        self.0.gen()
    }

    /// Uniform integer in `[0, n)`. `n` must be > 0.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        // Multiply-and-floor on a [0,1) draw. With n bounded by population
        // size this has negligible modulo bias for the count ranges we run.
        let u = self.uniform();
        ((u * n as f64) as usize).min(n - 1)
    }
}

/// Same 32-byte seed expansion as `crate::rng::StatefulRng`. Duplicated rather
/// than exported to keep the lineage stream's derivation self-contained and
/// auditable next to its XOR offset.
fn expand_u64_to_seed(v: u64) -> [u8; 32] {
    let b = v.to_le_bytes();
    let b2 = v.wrapping_mul(0x9e3779b97f4a7c15).to_le_bytes();
    let b3 = v.wrapping_mul(0x6c62272e07bb0142).to_le_bytes();
    let b4 = v.wrapping_mul(0xd800000000000000).to_le_bytes();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&b);
    seed[8..16].copy_from_slice(&b2);
    seed[16..24].copy_from_slice(&b3);
    seed[24..32].copy_from_slice(&b4);
    seed
}

/// Live per-(deme, compartment) identity pools.
///
/// Each pool is a `Vec<IndividualId>` — a multiset of the IDs currently in
/// that compartment. Uniform within-pool sampling picks an index uniformly;
/// removal is an unordered `swap_remove`, which keeps removal O(1) and (because
/// the lineage RNG is separate) does not affect the count trajectory.
pub struct IdentityState {
    pools: HashMap<(DemeId, CompartmentId), Vec<IndividualId>>,
    next: u64,
}

impl IdentityState {
    pub fn new() -> Self {
        IdentityState { pools: HashMap::new(), next: 0 }
    }

    fn mint(&mut self) -> IndividualId {
        let id = IndividualId(self.next);
        self.next = self.next.checked_add(1).expect("IndividualId counter overflow");
        id
    }

    fn pool_mut(&mut self, deme: DemeId, comp: CompartmentId) -> &mut Vec<IndividualId> {
        self.pools.entry((deme, comp)).or_default()
    }

    /// Number of live IDs in a pool (0 if the pool was never created).
    pub fn pool_len(&self, deme: DemeId, comp: CompartmentId) -> usize {
        self.pools.get(&(deme, comp)).map_or(0, |v| v.len())
    }

    /// Total minted IDs (== next counter).
    pub fn total_minted(&self) -> u64 {
        self.next
    }

    /// Mint `count` fresh IDs into a pool (used for t=0 seeding). Returns
    /// nothing; the IDs are appended to the pool.
    fn seed_pool(&mut self, deme: DemeId, comp: CompartmentId, count: i64) {
        if count <= 0 {
            return;
        }
        let mut ids = Vec::with_capacity(count as usize);
        for _ in 0..count {
            ids.push(self.mint());
        }
        self.pools.entry((deme, comp)).or_default().extend(ids);
    }

    /// Remove a uniformly-chosen individual from a pool and return it.
    /// Returns `None` if the pool is empty (a structural error the caller
    /// surfaces).
    fn remove_uniform(&mut self, deme: DemeId, comp: CompartmentId, rng: &mut LineageRng) -> Option<IndividualId> {
        let pool = self.pools.get_mut(&(deme, comp))?;
        if pool.is_empty() {
            return None;
        }
        let idx = rng.below(pool.len());
        Some(pool.swap_remove(idx))
    }

    /// Push an existing individual into a pool.
    fn push(&mut self, deme: DemeId, comp: CompartmentId, id: IndividualId) {
        self.pool_mut(deme, comp).push(id);
    }

    /// Is `id` currently live in `(deme, comp)`? Used by structural tests.
    pub fn contains(&self, deme: DemeId, comp: CompartmentId, id: IndividualId) -> bool {
        self.pools.get(&(deme, comp)).is_some_and(|v| v.contains(&id))
    }
}

impl Default for IdentityState {
    fn default() -> Self {
        Self::new()
    }
}

/// The seam the core simulation loop calls into. The default `None` path is
/// today's behaviour, byte-for-byte. `on_fired` is invoked after the loop has
/// drawn its own RNG and applied stoichiometry-determining decisions.
pub trait TransitionObserver {
    /// One transition fired. `multiplicity` is the number of identical firings
    /// (always 1 for Gillespie; > 1 reserved for tau-leap). `pre_state` is the
    /// integer state *before* the stoichiometry of these firings was applied —
    /// so weight expressions and pool sampling see the event-instant state.
    //
    // The argument list mirrors the proposal's `TransitionObserver` seam
    // (transition / deme / multiplicity / time / pre-state / params); bundling
    // them into a struct would only obscure the call site in the hot loop.
    #[allow(clippy::too_many_arguments)]
    fn on_fired(
        &mut self,
        transition: TransitionId,
        deme: DemeId,
        multiplicity: u64,
        time: f64,
        pre_int: &IntState,
        pre_real: &RealState,
        params: &[f64],
    ) -> Result<(), SimError>;
}

/// Per-transition precomputed routing: which integer-local compartment is the
/// source (delta < 0) and which is the destination (delta > 0), if any, plus
/// the global indices and the lineage decomposition.
struct TransitionRoute {
    /// Global compartment id of the source (the `-1` stoichiometry), if any.
    source: Option<CompartmentId>,
    /// Global compartment id of the destination (the `+1` stoichiometry), if any.
    destination: Option<CompartmentId>,
    /// `true` if this transition's source/destination touches a tracked comp.
    touches_tracked: bool,
    /// `Some(weights)` for a `#[lineage]` event: `(global_comp_id, weight_expr)`
    /// pairs, the linear decomposition of the rate over parent pools. Sampling
    /// picks pool `b ∝ weight_b · count_b`, then uniform within pool.
    parent_weights: Option<Vec<(CompartmentId, ir::expr::Expr)>>,
}

/// The concrete observer: owns the identity pools, the separate RNG stream, and
/// the line-list writer. Built once per run from a [`CompiledModel`].
pub struct LineageObserver<'m, W: LineListWriter> {
    model: &'m CompiledModel,
    identity: IdentityState,
    rng: LineageRng,
    writer: W,
    routes: Vec<TransitionRoute>,
    /// Global compartment ids that carry tracked IDs.
    tracked: Vec<CompartmentId>,
    deme: DemeId,
}

impl<'m, W: LineListWriter> LineageObserver<'m, W> {
    /// Build the observer and seed the initial identity pools at t=0.
    ///
    /// Only `model.identity_tracked_compartments` are seeded; their initial
    /// counts come from `initial_int`. IDs minted at t=0 carry parent
    /// [`ParentRef::Seed`] in any subsequent event.
    pub fn new(
        model: &'m CompiledModel,
        sim_seed: u64,
        initial_int: &IntState,
        mut writer: W,
    ) -> Result<Self, SimError> {
        let deme: DemeId = 0;

        // Resolve the tracked-compartment names to global indices.
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

        // Precompute per-transition routing.
        let mut routes = Vec::with_capacity(model.model.transitions.len());
        for (tr_idx, tr) in model.model.transitions.iter().enumerate() {
            // Find source (delta < 0) and destination (delta > 0) among integer
            // compartments. Single-source/single-destination is the common case;
            // for multi-entry stoichiometry we take the first of each sign, which
            // matches the simple-transition attribution rule (one ID moves).
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

            let parent_weights = match &tr.lineage {
                Some(l) if l.is_lineage_event => {
                    let mut pairs = Vec::with_capacity(l.parent_pool_weights.len());
                    for (comp_name, weight) in &l.parent_pool_weights {
                        let g = model
                            .comp_index
                            .get(comp_name.as_str())
                            .copied()
                            .ok_or_else(|| SimError::UnknownCompartment(comp_name.clone()))?;
                        pairs.push((g, weight.clone()));
                    }
                    Some(pairs)
                }
                _ => None,
            };

            let touches_tracked = source.is_some_and(is_tracked)
                || destination.is_some_and(is_tracked)
                || parent_weights
                    .as_ref()
                    .is_some_and(|w| w.iter().any(|(g, _)| is_tracked(*g)));

            routes.push(TransitionRoute {
                source,
                destination,
                touches_tracked,
                parent_weights,
            });
        }

        let mut identity = IdentityState::new();
        // Seed initial pools for tracked compartments from their t=0 counts.
        for &g in &tracked {
            if let Some(local) = model.global_to_int[g] {
                identity.seed_pool(deme, g, initial_int.counts[local]);
            }
        }

        // Initialise the writer (writes header / schema).
        writer.init()?;

        Ok(LineageObserver {
            model,
            identity,
            rng: LineageRng::from_sim_seed(sim_seed),
            writer,
            routes,
            tracked,
            deme,
        })
    }

    /// Finish writing and return the underlying writer (flushed). Call after
    /// the simulation loop completes.
    pub fn finish(mut self) -> Result<W, SimError> {
        self.writer.finish()?;
        Ok(self.writer)
    }

    /// Borrow the identity state (structural tests).
    pub fn identity(&self) -> &IdentityState {
        &self.identity
    }

    /// Sample a parent pool `b ∝ weight_b · count_b` (uniform within pool).
    /// Returns the chosen parent individual. Errors if every pool is empty or
    /// all weights are zero — a structural inconsistency, since a `#[lineage]`
    /// transition only fires when its rate (and thus some `weight·count`) is
    /// positive.
    fn sample_parent(
        &mut self,
        weights: &[(CompartmentId, ir::expr::Expr)],
        pre_int: &IntState,
        pre_real: &RealState,
        params: &[f64],
        time: f64,
        tr_idx: TransitionId,
    ) -> Result<IndividualId, SimError> {
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

        // Per-pool unnormalised mass = weight_b · count_b.
        let mut masses: Vec<(CompartmentId, f64)> = Vec::with_capacity(weights.len());
        let mut total = 0.0;
        for (g, weight_expr) in weights {
            let w = eval_expr(weight_expr, &ctx)?.max(0.0);
            let count = self.identity.pool_len(self.deme, *g) as f64;
            let mass = w * count;
            total += mass;
            masses.push((*g, mass));
        }

        if total <= 0.0 {
            return Err(SimError::Validation(format!(
                "lineage transition '{}' fired at t={} but every parent pool has \
                 zero weight·count mass; the identity-pool bookkeeping has \
                 diverged from the count state",
                self.model.model.transitions[tr_idx].name, time
            )));
        }

        // Select pool by cumulative mass against a uniform draw.
        let u = self.rng.uniform() * total;
        let mut cumulative = 0.0;
        let mut chosen = masses[masses.len() - 1].0;
        for (g, mass) in &masses {
            cumulative += *mass;
            if cumulative >= u {
                chosen = *g;
                break;
            }
        }

        // Uniform within the chosen pool — but do NOT remove (the parent is
        // not consumed by a lineage event; only the source individual moves).
        let pool = self
            .identity
            .pools
            .get(&(self.deme, chosen))
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                SimError::Validation(format!(
                    "lineage transition '{}': chosen parent pool (comp {}) is \
                     empty at t={}",
                    self.model.model.transitions[tr_idx].name, chosen, time
                ))
            })?;
        let idx = self.rng.below(pool.len());
        Ok(pool[idx])
    }
}

impl<'m, W: LineListWriter> TransitionObserver for LineageObserver<'m, W> {
    #[allow(clippy::too_many_arguments)]
    fn on_fired(
        &mut self,
        transition: TransitionId,
        deme: DemeId,
        multiplicity: u64,
        time: f64,
        pre_int: &IntState,
        pre_real: &RealState,
        params: &[f64],
    ) -> Result<(), SimError> {
        debug_assert_eq!(deme, self.deme, "single-population slice tracks deme 0 only");

        let route = &self.routes[transition];
        if !route.touches_tracked {
            return Ok(()); // Untracked transition — no overhead beyond the flag check.
        }

        // The pre_state passed in already reflects state before *this* firing's
        // stoichiometry. Within a single firing the pools and counts agree.
        // Borrow-checker note: clone the small route fields we need so we can
        // mutate identity/writer afterward.
        let source = route.source;
        let destination = route.destination;
        let parent_weights = route.parent_weights.clone();

        for _ in 0..multiplicity {
            let (individual, parent, src_for_record, dst_for_record) =
                if let Some(weights) = &parent_weights {
                    // Lineage event: sample a parent from the weighted pools,
                    // mint a fresh child in the destination, move/remove the
                    // source individual.
                    let parent_id =
                        self.sample_parent(weights, pre_int, pre_real, params, time, transition)?;

                    // The focal (child) individual: a new ID minted in the
                    // destination. The source individual (e.g. an S) is consumed
                    // — it leaves the source pool but is not itself tracked into
                    // the destination (the destination gets a *new* infectee ID).
                    if let Some(src) = source {
                        if self.tracked.contains(&src) {
                            // S is rarely tracked, but if a cycle pulled it in,
                            // remove the source individual.
                            let _ = self.identity.remove_uniform(deme, src, &mut self.rng);
                        }
                    }
                    let child = self.identity.mint();
                    if let Some(dst) = destination {
                        self.identity.push(deme, dst, child);
                    }
                    (child, ParentRef::Individual(parent_id), source, destination)
                } else {
                    match (source, destination) {
                        (Some(src), Some(dst)) => {
                            // Progression: move one ID from source to destination.
                            // If the source pool is empty (e.g. an untracked
                            // source feeding a tracked destination), mint instead.
                            let id = match self.identity.remove_uniform(deme, src, &mut self.rng) {
                                Some(id) => id,
                                None => self.identity.mint(),
                            };
                            self.identity.push(deme, dst, id);
                            (id, ParentRef::None, Some(src), Some(dst))
                        }
                        (Some(src), None) => {
                            // Outflow (death): remove one ID from the source.
                            let id = self
                                .identity
                                .remove_uniform(deme, src, &mut self.rng)
                                .unwrap_or_else(|| self.identity.mint());
                            (id, ParentRef::None, Some(src), None)
                        }
                        (None, Some(dst)) => {
                            // Inflow (import): mint a new ID with no parent.
                            let id = self.identity.mint();
                            self.identity.push(deme, dst, id);
                            (id, ParentRef::Import, None, Some(dst))
                        }
                        (None, None) => {
                            // Nothing routable — shouldn't happen for a tracked
                            // transition, but emit no record rather than guess.
                            continue;
                        }
                    }
                };

            self.writer.write(&LineListEntry {
                time,
                transition,
                individual,
                source: src_for_record,
                destination: dst_for_record,
                deme,
                parent,
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_rng_is_disjoint_from_main_seed() {
        // The lineage stream must differ from the main stream for the same
        // seed, else identity draws could (in a buggy refactor) be confused
        // with simulation draws.
        let mut main = crate::rng::StatefulRng::new(42);
        let mut lin = LineageRng::from_sim_seed(42);
        let a: Vec<f64> = (0..8).map(|_| main.uniform()).collect();
        let b: Vec<f64> = (0..8).map(|_| lin.uniform()).collect();
        assert_ne!(a, b, "lineage and main streams must not coincide");
    }

    #[test]
    fn lineage_rng_reproducible() {
        let mut a = LineageRng::from_sim_seed(7);
        let mut b = LineageRng::from_sim_seed(7);
        for _ in 0..16 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    #[test]
    fn identity_state_mint_and_pools() {
        let mut s = IdentityState::new();
        s.seed_pool(0, 3, 5);
        assert_eq!(s.pool_len(0, 3), 5);
        assert_eq!(s.total_minted(), 5);
        let mut rng = LineageRng::from_sim_seed(1);
        let id = s.remove_uniform(0, 3, &mut rng).unwrap();
        assert_eq!(s.pool_len(0, 3), 4);
        assert!(!s.contains(0, 3, id));
        s.push(0, 7, id);
        assert!(s.contains(0, 7, id));
    }
}
