//! `Layout` — the factored, readable store path for a leaf.
//!
//! The leaf's identity is the *ordered tuple of per-level hashes along its
//! path*; the store path is a readable nested **factoring** of that identity,
//! not a flat blob dir, so `list`/`show`/`cat` keep working. Each path
//! segment is `{label}-{hash8}`: the label is **provenance** (a rename → a
//! new dir → a harmless cache miss, never a wrong answer) and the `hash8` is
//! **identity** (the level's `ContentHash`, first 4 bytes as 8 hex). Eight
//! hex chars per segment suffices — a collision needs *every* level on the
//! path to collide simultaneously, and `run.json` records the full 64-char
//! hashes for verification.
//!
//! Navigation and display read `run.json`, never these segments. The store
//! ([`crate::store`]) appends a `~{disambiguator}` to the final segment on a
//! `PathPrefixCollision`; `Layout` always produces the base form, and the
//! reader enumerates sibling dirs rather than reconstructing names — so a
//! `~`-suffixed sibling is found like any other leaf.

use std::path::{Path, PathBuf};

use crate::kind::ArtifactKind;
use crate::record::LevelId;

impl ArtifactKind {
    /// The top-level store partition directory for this kind — the "type"
    /// level of `results/` (`sims/`, `fits/`, …).
    pub fn store_dir(self) -> &'static str {
        match self {
            ArtifactKind::Sim => "sims",
            ArtifactKind::FitStage => "fits",
            ArtifactKind::Pfilter => "pfilters",
            ArtifactKind::Survey => "surveys",
            ArtifactKind::ProfilePoint => "profiles",
            ArtifactKind::Obs => "obs",
            ArtifactKind::Projection => "projections",
        }
    }
}

/// Render a level label into a filesystem-safe path segment component.
///
/// Lowercases and maps any character outside `[a-z0-9._-]` to `_`. Hyphens
/// and dots are preserved so readable compound labels survive intact
/// (`chain_binomial-dt1`, `01-scout`, `seed_42`). The label is provenance,
/// so this is purely cosmetic — identity rides in the `hash8` suffix.
pub fn path_label(label: &str) -> String {
    label
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// One path segment for a level: `{path_label(label)}-{hash8}`.
pub fn segment(level: &LevelId) -> String {
    format!("{}-{}", path_label(&level.label), level.hash.short8())
}

/// The factored store path for a leaf:
/// `{root}/{kind_dir}/{seg_0}/…/{seg_n}`, one segment per level in path
/// order. Each segment is `{label}-{hash8}`.
///
/// This is the base (un-disambiguated) path; `CasStore::commit` resolves a
/// `PathPrefixCollision` by escalating the final segment
/// (`{seg}` → `{seg}~{hash16}` → `{seg}~{full64}`), so two leaves whose
/// short hashes collide on every level still get distinct directories.
pub fn store_path(root: &Path, kind: ArtifactKind, levels: &[LevelId]) -> PathBuf {
    let mut p = root.join(kind.store_dir());
    for level in levels {
        p = p.join(segment(level));
    }
    p
}

#[cfg(test)]
mod tests;
