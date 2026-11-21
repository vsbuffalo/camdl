//! `ArtifactKind` — the top "type" partition of the store — and [`run_id`],
//! the leaf's address.

use serde::{Deserialize, Serialize};

use crate::hash::{CanonicalHasher, ContentAddressed, ContentHash, HASH_VERSION};

/// The kind of artifact a leaf produces. This is the top level of the store
/// (`results/sims/`, `results/fits/`, …) and the `kind` discriminator in
/// `run.json`. Each variant has a fixed index, which is folded into
/// [`run_id`] as a fixed-width tag — so two kinds with a coincidentally-equal
/// level-hash sequence cannot alias to the same id.
///
/// A variant's index is its [`tag_index`](ArtifactKind::tag_index), which is
/// written out per variant and never reused: a new kind takes the next unused
/// number, and index 3 stays unassigned, so no kind's `run_id` depends on
/// which other kinds exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A forward-simulation trajectory (`simulate`).
    Sim,
    /// One `fit run` of a fit.toml's `[method]` (`fits/`).
    FitMethod,
    /// A particle-filter loglik evaluation at fixed params (`pfilter`).
    Pfilter,
    /// One grid-point × start of a profile scan (`profile`).
    ProfilePoint,
    /// Reserved for a synthetic-observation sub-artifact under a trajectory.
    /// Not yet constructed: `simulate --obs*` draws currently live in an `obs/`
    /// child dir (carrying `obs.json`, not a `RunRecord`); this variant is the
    /// store kind for promoting them to first-class leaves.
    Obs,
    /// Reserved for a lineage projection (`realize`/`tree`/`cohort`/`sojourn`).
    /// Not yet constructed; defined so its store partition + run-id tag are
    /// pinned ahead of wiring the `lineage` subcommands into the CAS store.
    Projection,
    /// The combined-across-cells wide-format trajectory TSV of a multi-cell
    /// `simulate` (`--replicates`/`--seeds`/multi-scenario/`--draws`). Derived
    /// from — and `deps` on — the N per-cell [`Sim`](ArtifactKind::Sim) leaves;
    /// its identity is the model/config/params plus the full sorted cell set
    /// (the cell COUNT is in the key).
    SimEnsemble,
}

impl ArtifactKind {
    /// The fixed-width tag folded into [`run_id`]. A number is never
    /// reassigned: 3 is unassigned, and a new kind takes the next unused one.
    pub fn tag_index(self) -> u32 {
        match self {
            ArtifactKind::Sim => 0,
            ArtifactKind::FitMethod => 1,
            ArtifactKind::Pfilter => 2,
            ArtifactKind::ProfilePoint => 4,
            ArtifactKind::Obs => 5,
            ArtifactKind::Projection => 6,
            ArtifactKind::SimEnsemble => 7,
        }
    }
}

impl ContentAddressed for ArtifactKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        // Enum rule: the variant's tag as a fixed-width `u32`.
        h.write_u32(self.tag_index());
    }
}

/// The leaf's address: `hash(HASH_VERSION ++ kind_tag ++ count ++ [level
/// hashes in path order])`.
///
/// The root derivation obeys the same framing rules as everything else:
/// `kind_tag` is a fixed-width enum index (not a bare string) and the
/// level-hash list is count-prefixed (`u64` LE), so two kinds with
/// coincidentally-equal level sequences cannot alias, and `([h1,h2], [h1])`
/// cannot collide with a sequence that concatenates to the same bytes. One
/// 32-byte id per leaf, recorded in `run.json`.
pub fn run_id(kind: ArtifactKind, levels: &[ContentHash]) -> ContentHash {
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_u32(kind.tag_index());
    h.write_len(levels.len() as u64);
    for level in levels {
        level.hash_into(&mut h);
    }
    h.finalize()
}
