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
//! Gillespie only. tau-leap / chain-binomial declare the
//! [`crate::Capabilities::LINEAGES`] flag (so the capability check passes) but
//! tracking is not yet wired into their loops — requesting `--lineages` on
//! those backends returns a clear "not yet implemented" error at the CLI
//! (Phase 3).
//!
//! ## Phase 2: stratified / spatial attribution
//!
//! Demes are real (see [`deme::DemeMap`]). A `#[lineage]` event in stratum `a`
//! samples its parent stratum `b` with probability `∝ weight_b · count_b`,
//! where `weight_b` is the per-class weight the OCaml compiler emitted for the
//! parent compartment `I[b]` (for stratified frequency-dependent transmission,
//! `weight_b = β·C[a,b]·S[a]/N[b]`). The parent is then sampled uniformly
//! within the `(b, I[b])` pool and the line list records `parent_deme = b`,
//! child `deme = a`. Because the expanded IR gives each stratum its own
//! compartment (`I_a`, `I_b`), the per-stratum pools are distinguished by
//! compartment id; the deme is the compartment's stratum index.

pub mod deme;
pub mod writer;
pub mod tree;

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::propensity::{eval_expr, EvalCtx};
use crate::state::{IntState, RealState};

pub use deme::DemeMap;
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

/// Deme (spatial/stratum) identifier — the stratum index of a compartment in
/// the cartesian product of the model's dimensions (see [`deme::DemeMap`]).
/// 0 for unstratified compartments and every compartment in a single-
/// population model, so the single-population slice is the `DemeId = 0`
/// special case.
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

    /// A frozen snapshot of the current pools, for batch (tau-leap /
    /// chain-binomial) parent sampling. See [`PoolSnapshot`].
    fn snapshot(&self) -> PoolSnapshot {
        PoolSnapshot { pools: self.pools.clone() }
    }
}

/// A frozen clone of the identity pools, taken at the start of a batch step.
///
/// tau-leap and chain-binomial fire `k` events against rates and pools frozen
/// at step start. Parent sampling for those `k` events must read parent pools
/// from this snapshot, NOT from the live pools that the step's own child
/// minting mutates — otherwise a child minted earlier this step could be
/// recorded as a same-step parent, an edge the frozen approximation cannot
/// resolve. Gillespie samples from live pools (exact, sequential) and uses no
/// snapshot.
pub struct PoolSnapshot {
    pools: HashMap<(DemeId, CompartmentId), Vec<IndividualId>>,
}

impl PoolSnapshot {
    /// Number of IDs that were in `(deme, comp)` at snapshot time.
    fn pool_len(&self, deme: DemeId, comp: CompartmentId) -> usize {
        self.pools.get(&(deme, comp)).map_or(0, |v| v.len())
    }

    /// The frozen member list for `(deme, comp)`, if any.
    fn pool(&self, deme: DemeId, comp: CompartmentId) -> Option<&Vec<IndividualId>> {
        self.pools.get(&(deme, comp)).filter(|p| !p.is_empty())
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

    /// Begin a batch step (tau-leap / chain-binomial): the backend has frozen
    /// rates and pools at step start and will feed `on_fired` with `multiplicity
    /// > 1`. Default no-op — only the lineage observer freezes a pool snapshot.
    fn begin_batch_step(&mut self) {}

    /// End a batch step. Default no-op.
    fn end_batch_step(&mut self) {}
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
    /// Per-compartment deme (stratum) assignment. The pool key for a
    /// compartment `c` is `(deme_map.deme_of(c), c)`.
    deme_map: DemeMap,
    /// `Some` only during a batch (tau-leap / chain-binomial) step: a frozen
    /// clone of the pools at step start, used for parent sampling so within-step
    /// minting cannot create a same-step parent edge. `None` for Gillespie
    /// (exact, sequential live-pool sampling).
    snapshot: Option<PoolSnapshot>,
    /// Sub-`dt` bias estimator (batch backends only). `edges` counts every
    /// transmission edge (child born at a lineage event); `sub_dt_edges`
    /// accumulates the edge-weighted same-step-parent share. The reported
    /// fraction `sub_dt_edges / edges` is the fraction of edges the frozen
    /// approximation cannot temporally resolve. Exactly 0 for Gillespie.
    edges: u64,
    sub_dt_edges: f64,
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
        // Per-compartment deme (stratum) assignment. 0 everywhere for an
        // unstratified / single-population model — the Phase-1 special case.
        let deme_map = DemeMap::build(&model.model, &model.comp_index);

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
        // Seed initial pools for tracked compartments from their t=0 counts,
        // each in its own stratum's deme.
        for &g in &tracked {
            if let Some(local) = model.global_to_int[g] {
                identity.seed_pool(deme_map.deme_of(g), g, initial_int.counts[local]);
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
            deme_map,
            snapshot: None,
            edges: 0,
            sub_dt_edges: 0.0,
        })
    }

    /// The sub-`dt` bias fraction accumulated over the run: the edge-weighted
    /// fraction of transmission edges the frozen-pool approximation cannot
    /// temporally resolve (a same-step parent that, under an exact sequential
    /// process, might have been a child born earlier in the same step). Exactly
    /// 0.0 for Gillespie (no batch steps) and for a run with no lineage edges.
    pub fn sub_dt_fraction(&self) -> f64 {
        if self.edges == 0 {
            0.0
        } else {
            self.sub_dt_edges / self.edges as f64
        }
    }

    /// Total transmission edges (children born at lineage events) recorded.
    pub fn lineage_edge_count(&self) -> u64 {
        self.edges
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
    /// Returns the chosen parent individual *and its deme* (its stratum). Each
    /// candidate compartment `I[b]` lives in its own stratum's pool
    /// `(deme_of(I[b]), I[b])`, so the per-stratum weight `weight_b` is paired
    /// with the per-stratum count `count_b` — this is the contact-structured
    /// attribution: a stratum with higher `C[a,b]·I[b]/N[b]` wins
    /// proportionally more parents, NOT uniform over all infectious.
    ///
    /// Errors if every pool is empty or all weights are zero — a structural
    /// inconsistency, since a `#[lineage]` transition only fires when its rate
    /// (and thus some `weight·count`) is positive.
    fn sample_parent(
        &mut self,
        weights: &[(CompartmentId, ir::expr::Expr)],
        pre_int: &IntState,
        pre_real: &RealState,
        params: &[f64],
        time: f64,
        tr_idx: TransitionId,
    ) -> Result<(IndividualId, DemeId), SimError> {
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

        // Per-pool unnormalised mass = weight_b · count_b, with count_b taken
        // from the parent compartment's own stratum pool. In a batch step the
        // pool is read from the frozen snapshot (start-of-step state), so all
        // firings this step see the same parent pools — a child minted earlier
        // this step is invisible as a parent. Gillespie (no snapshot) reads the
        // live pool, which is the exact sequential behaviour.
        let mut masses: Vec<(CompartmentId, DemeId, f64)> = Vec::with_capacity(weights.len());
        let mut total = 0.0;
        for (g, weight_expr) in weights {
            let parent_deme = self.deme_map.deme_of(*g);
            let w = eval_expr(weight_expr, &ctx)?.max(0.0);
            let count = match &self.snapshot {
                Some(snap) => snap.pool_len(parent_deme, *g),
                None => self.identity.pool_len(parent_deme, *g),
            } as f64;
            let mass = w * count;
            total += mass;
            masses.push((*g, parent_deme, mass));
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
        let (mut chosen, mut chosen_deme) = {
            let last = &masses[masses.len() - 1];
            (last.0, last.1)
        };
        for (g, d, mass) in &masses {
            cumulative += *mass;
            if cumulative >= u {
                chosen = *g;
                chosen_deme = *d;
                break;
            }
        }

        // Uniform within the chosen pool — but do NOT remove (the parent is
        // not consumed by a lineage event; only the source individual moves).
        // Read members from the snapshot in a batch step, the live pool
        // otherwise (mirrors the count source above).
        let pool = match &self.snapshot {
            Some(snap) => snap.pool(chosen_deme, chosen),
            None => self
                .identity
                .pools
                .get(&(chosen_deme, chosen))
                .filter(|p| !p.is_empty()),
        }
        .ok_or_else(|| {
            SimError::Validation(format!(
                "lineage transition '{}': chosen parent pool (comp {}, deme {}) \
                 is empty at t={}",
                self.model.model.transitions[tr_idx].name, chosen, chosen_deme, time
            ))
        })?;
        let idx = self.rng.below(pool.len());
        Ok((pool[idx], chosen_deme))
    }
}

impl<'m, W: LineListWriter> TransitionObserver for LineageObserver<'m, W> {
    /// Freeze the current pools so all firings this batch step sample parents
    /// from start-of-step state. Gillespie never calls this — it samples live.
    fn begin_batch_step(&mut self) {
        self.snapshot = Some(self.identity.snapshot());
    }

    /// Drop the frozen snapshot; subsequent `on_fired` calls sample live pools.
    fn end_batch_step(&mut self) {
        self.snapshot = None;
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
        // The caller's `deme` hint is not used for pool keying: in the fully-
        // expanded IR every stratum is its own compartment, so a compartment's
        // deme is read from `deme_map`, not from the firing. The backend has
        // no separate notion of which deme fired.

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

        // The focal individual's deme is its compartment's stratum: the
        // destination for an inflow/move, else the source for an outflow.
        let child_deme = match (source, destination) {
            (_, Some(dst)) => self.deme_map.deme_of(dst),
            (Some(src), None) => self.deme_map.deme_of(src),
            (None, None) => 0,
        };

        // Sub-`dt` bias accounting (batch backends only — a snapshot is active).
        // This lineage step mints `m = multiplicity` children into the
        // destination pool, which held `p` IDs at step start (snapshot count).
        // Under the frozen approximation none of those `m` same-step children
        // can be sampled as a parent this step; under exact sequential dynamics
        // a fraction `m/(p+m)` of would-be-live parents are precisely those
        // unresolvable same-step children. Each of the `m` edges therefore
        // carries that share. Accumulating edge-weighted gives
        // `Σ m·(m/(p+m))`; the reported `sub_dt_fraction` is this over total
        // edges. Exactly 0 for Gillespie (no snapshot, m=1 per call, but the
        // branch is gated on `snapshot`).
        if parent_weights.is_some() {
            if let (Some(snap), Some(dst)) = (self.snapshot.as_ref(), destination) {
                let m = multiplicity as f64;
                let dst_deme = self.deme_map.deme_of(dst);
                let p = snap.pool_len(dst_deme, dst) as f64;
                self.edges += multiplicity;
                if p + m > 0.0 {
                    self.sub_dt_edges += m * (m / (p + m));
                }
            }
        }

        for _ in 0..multiplicity {
            let (individual, parent, parent_deme, src_for_record, dst_for_record) =
                if let Some(weights) = &parent_weights {
                    // Lineage event: sample a parent (and its stratum) from the
                    // per-stratum weighted pools, mint a fresh child in the
                    // destination's stratum, move/remove the source individual.
                    let (parent_id, parent_deme) =
                        self.sample_parent(weights, pre_int, pre_real, params, time, transition)?;

                    // The focal (child) individual: a new ID minted in the
                    // destination. The source individual (e.g. an S) is consumed
                    // — it leaves the source pool but is not itself tracked into
                    // the destination (the destination gets a *new* infectee ID).
                    if let Some(src) = source {
                        if self.tracked.contains(&src) {
                            // S is rarely tracked, but if a cycle pulled it in,
                            // remove the source individual from its stratum pool.
                            let src_deme = self.deme_map.deme_of(src);
                            let _ = self.identity.remove_uniform(src_deme, src, &mut self.rng);
                        }
                    }
                    let child = self.identity.mint();
                    if let Some(dst) = destination {
                        self.identity.push(self.deme_map.deme_of(dst), dst, child);
                    }
                    (
                        child,
                        ParentRef::Individual(parent_id),
                        Some(parent_deme),
                        source,
                        destination,
                    )
                } else {
                    match (source, destination) {
                        (Some(src), Some(dst)) => {
                            // Progression: move one ID from the source stratum to
                            // the destination stratum. If the source pool is empty
                            // (untracked source feeding a tracked destination),
                            // mint instead.
                            let src_deme = self.deme_map.deme_of(src);
                            let dst_deme = self.deme_map.deme_of(dst);
                            let id =
                                match self.identity.remove_uniform(src_deme, src, &mut self.rng) {
                                    Some(id) => id,
                                    None => self.identity.mint(),
                                };
                            self.identity.push(dst_deme, dst, id);
                            (id, ParentRef::None, None, Some(src), Some(dst))
                        }
                        (Some(src), None) => {
                            // Outflow (death): remove one ID from the source stratum.
                            let src_deme = self.deme_map.deme_of(src);
                            let id = self
                                .identity
                                .remove_uniform(src_deme, src, &mut self.rng)
                                .unwrap_or_else(|| self.identity.mint());
                            (id, ParentRef::None, None, Some(src), None)
                        }
                        (None, Some(dst)) => {
                            // Inflow (import): mint a new ID with no parent.
                            let id = self.identity.mint();
                            self.identity.push(self.deme_map.deme_of(dst), dst, id);
                            (id, ParentRef::Import, None, None, Some(dst))
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
                deme: child_deme,
                parent,
                parent_deme,
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
