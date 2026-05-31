//! `ArtifactKind` — the top "type" partition of the store — and [`run_id`],
//! the leaf's address.

use serde::{Deserialize, Serialize};

use crate::hash::{CanonicalHasher, ContentAddressed, ContentHash, HASH_VERSION};

/// The kind of artifact a leaf produces. This is the top level of the store
/// (`results/sims/`, `results/fits/`, …) and the `kind` discriminator in
/// `run.json`. The declaration order fixes each variant's index, which is
/// folded into [`run_id`] as a fixed-width tag — so two kinds with a
/// coincidentally-equal level-hash sequence cannot alias to the same id.
///
/// Adding a variant is append-only: insert new kinds at the end so existing
/// indices (and therefore existing `run_id`s) are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A forward-simulation trajectory (`simulate`/`batch`).
    Sim,
    /// One stage of a fit pipeline (`fit`/`if2`-desugared).
    FitStage,
    /// A particle-filter loglik evaluation at fixed params (`pfilter`).
    Pfilter,
    /// A likelihood-landscape diagnostic (`survey`).
    Survey,
    /// One grid-point × start of a profile scan (`profile`).
    ProfilePoint,
    /// A synthetic-observation sub-artifact under a trajectory.
    Obs,
    /// A lineage projection (`realize`/`tree`/`cohort`/`sojourn`).
    Projection,
}

impl ArtifactKind {
    /// The fixed-width tag folded into [`run_id`]: the variant's declaration
    /// index. Append-only — never renumber.
    pub fn tag_index(self) -> u32 {
        match self {
            ArtifactKind::Sim => 0,
            ArtifactKind::FitStage => 1,
            ArtifactKind::Pfilter => 2,
            ArtifactKind::Survey => 3,
            ArtifactKind::ProfilePoint => 4,
            ArtifactKind::Obs => 5,
            ArtifactKind::Projection => 6,
        }
    }
}

impl ContentAddressed for ArtifactKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        // Enum rule: the variant index as a fixed-width `u32`.
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
