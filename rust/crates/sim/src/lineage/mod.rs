//! Three-layer lineage architecture — event log → line list → tree.
//!
//! Implements Layers 1–2 of the 2026-05-20 proposal
//! (`docs/dev/proposals/2026-05-20-lineage-resampling-and-likelihood.md`),
//! refactoring the shipped two-layer (inline-attribution) design:
//!
//! - **Layer 1 — [`event_log`]:** the simulation records an [`EventLog`] (the
//!   ordered event sequence + evaluated per-pool FOI masses at `#[lineage]`
//!   events). It draws *no* identities. `simulate --event-log` writes it.
//! - **Layer 2 — [`realize`]:** an offline replay (`camdl lineage realize`)
//!   reads the event log, maintains the per-pool [`IdentityState`], samples
//!   *which individuals* at each event from the recorded masses, mints IDs, and
//!   writes a [`LineListEntry`] line list — accumulating the §4a attribution
//!   log-probability per event.
//! - **Layer 3 (downstream, unchanged):** [`tree`] / [`project`] consume the
//!   realized line list (transmission tree, sojourn, cohort).
//!
//! ## The factorization (why the split)
//!
//! `P(augmented) = P(counts) × P(identities | counts)`. The simulation draws
//! the first factor; identity attribution is a separate stochastic layer drawn
//! during replay. One expensive epidemic (event log) → many cheap identity
//! realizations (line lists), each an i.i.d. draw from `P(identities | events)`
//! seeded by `--identity-seed`.
//!
//! ## The separate RNG stream
//!
//! Identity draws (which parent pool, which individual) come from
//! [`LineageRng`], an *independent* ChaCha8 stream seeded `seed ⊕
//! LINEAGE_RNG_OFFSET`. In the refactored design the *count* simulation draws
//! no identities at all (the recorder is identity-free), so a `--event-log`
//! run's count trajectory is byte-identical to a plain run at the same seed —
//! trivially, because the simulation is literally unchanged (Tier 2a).
//!
//! ## Stratified / spatial attribution
//!
//! Demes are real (see [`deme::DemeMap`]). A `#[lineage]` event in stratum `a`
//! samples its parent stratum `b` with probability `∝ w_b·X_b`, where `w_b` is
//! the per-class weight the OCaml compiler emitted for the parent compartment
//! `I[b]` (for stratified frequency-dependent transmission,
//! `w_b = β·C[a,b]·S[a]/N[b]`). The parent is then sampled uniformly within the
//! `(b, I[b])` pool. The event recorder evaluates and stores `w_b·X_b`; realize
//! resamples pool-then-individual from those recorded masses.

pub mod deme;
pub mod event_log;
pub mod event_log_io;
pub mod realize;
pub mod writer;
pub mod tree;
pub mod project;

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::SimError;
use crate::state::{IntState, RealState};

pub use deme::DemeMap;
pub use event_log::{EventLog, EventRecord, EventRecorder, RouteInfo};
pub use realize::{realize, RealizeSummary};
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

    pub(crate) fn mint(&mut self) -> IndividualId {
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

    /// The `idx`-th live member of a pool. Caller guarantees `idx < pool_len`.
    /// Used by [`realize`] for the uniform-within-pool parent draw (the parent
    /// is *not* consumed — only the source individual moves).
    pub(crate) fn pool_member(&self, deme: DemeId, comp: CompartmentId, idx: usize) -> IndividualId {
        self.pools[&(deme, comp)][idx]
    }

    /// Total minted IDs (== next counter).
    pub fn total_minted(&self) -> u64 {
        self.next
    }

    /// Mint `count` fresh IDs into a pool (used for t=0 seeding). Returns
    /// nothing; the IDs are appended to the pool.
    pub(crate) fn seed_pool(&mut self, deme: DemeId, comp: CompartmentId, count: i64) {
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
    pub(crate) fn remove_uniform(&mut self, deme: DemeId, comp: CompartmentId, rng: &mut LineageRng) -> Option<IndividualId> {
        let pool = self.pools.get_mut(&(deme, comp))?;
        if pool.is_empty() {
            return None;
        }
        let idx = rng.below(pool.len());
        Some(pool.swap_remove(idx))
    }

    /// Push an existing individual into a pool.
    pub(crate) fn push(&mut self, deme: DemeId, comp: CompartmentId, id: IndividualId) {
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
/// drawn its own RNG and applied stoichiometry-determining decisions. The
/// concrete implementor is [`EventRecorder`] (Layer 1): it records each firing
/// (and the evaluated per-pool weights at lineage events) but draws no
/// identities, so it cannot perturb the count trajectory.
pub trait TransitionObserver {
    /// One transition fired. `multiplicity` is the number of identical firings
    /// (always 1 for Gillespie; > 1 for the batched backends). `pre_int` is the
    /// integer state *before* the stoichiometry of these firings was applied —
    /// so weight expressions see the event-instant (Gillespie) /
    /// start-of-step (batched) state, and `X_b == pre_int.counts[b]`.
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
    /// >= 1` and `batched = true`. Default no-op — the [`EventRecorder`] flips
    /// its `in_batch` flag so replay can reproduce the frozen-pool semantics.
    fn begin_batch_step(&mut self) {}

    /// End a batch step. Default no-op.
    fn end_batch_step(&mut self) {}
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
