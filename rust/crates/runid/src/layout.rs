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

use crate::hash::ContentHash;
use crate::kind::ArtifactKind;
use crate::record::LevelId;

/// The maximum bytes a single path component may occupy. POSIX `NAME_MAX` is
/// 255 on every filesystem camdl targets (ext4, APFS, XFS); a segment longer
/// than this fails `mkdir` with `ENAMETOOLONG`.
pub const NAME_MAX: usize = 255;

/// Conservative cap on the *label* portion of a segment, in bytes. A segment
/// is `{label}-{hash8}` (label + `-` + 8 hex). Capping the label at 200 leaves
/// ample headroom under `NAME_MAX` (255) for the `-{hash8}` suffix, the
/// truncation marker, and the store's `~{disambiguator}` it may later append.
const LABEL_CAP: usize = 200;

/// Marker inserted between a truncated label prefix and its full-label hash,
/// so a truncated label reads as deliberately-shortened, not merely cut off.
const TRUNC_MARKER: &str = "..";

/// Hex chars of the full-label digest appended after truncation. 16 hex = 8
/// bytes of SHA-256 — collision-resistant enough to disambiguate two labels
/// that share the same 200-byte prefix.
const TRUNC_HASH_HEX: usize = 16;

impl ArtifactKind {
    /// The top-level store partition directory for this kind — the "type"
    /// level of `results/` (`sims/`, `fits/`, …).
    pub fn store_dir(self) -> &'static str {
        match self {
            ArtifactKind::Sim => "sims",
            ArtifactKind::FitMethod => "fits",
            ArtifactKind::Pfilter => "pfilters",
            ArtifactKind::Survey => "surveys",
            ArtifactKind::ProfilePoint => "profiles",
            ArtifactKind::Obs => "obs",
            ArtifactKind::Projection => "projections",
            ArtifactKind::SimEnsemble => "ensembles",
        }
    }
}

/// Render a level label into a filesystem-safe path segment component.
///
/// Lowercases and maps any character outside `[a-z0-9._-]` to `_`. Hyphens
/// and dots are preserved so readable compound labels survive intact
/// (`chain_binomial-dt1`, `01-scout`, `seed_42`). The label is provenance,
/// so this is purely cosmetic — identity rides in the `hash8` suffix.
///
/// An over-long label (a `--draws` row on a many-parameter model joins every
/// `name=value` pair into one string — easily hundreds of bytes) would
/// overflow `NAME_MAX` when rendered as a single directory component, so the
/// result is capped at [`LABEL_CAP`] bytes: a long sanitized label is
/// truncated to a prefix and suffixed with `..{hash16}`, the first 16 hex of
/// the full sanitized label's SHA-256. Two distinct long labels that share a
/// 200-byte prefix still render distinctly via that hash. Identity is
/// unaffected — the label is provenance, never part of the level hash.
pub fn path_label(label: &str) -> String {
    let sanitized: String = label
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.len() <= LABEL_CAP {
        return sanitized;
    }

    // Truncate the sanitized prefix and disambiguate with a hash of the FULL
    // sanitized label, so distinct long labels stay distinct on disk.
    let hash = ContentHash::digest_bytes(sanitized.as_bytes());
    let tag = &hash.to_hex()[..TRUNC_HASH_HEX];
    let prefix_len = LABEL_CAP - TRUNC_MARKER.len() - TRUNC_HASH_HEX;
    let mut prefix_end = prefix_len;
    // `sanitized` is ASCII (lowercase alnum + `_-.`), so byte == char index
    // and any cut lands on a boundary — but clamp defensively.
    while prefix_end > 0 && !sanitized.is_char_boundary(prefix_end) {
        prefix_end -= 1;
    }
    format!("{}{}{}", &sanitized[..prefix_end], TRUNC_MARKER, tag)
}

/// One path segment for a level: `{path_label(label)}-{hash8}`. The label is
/// capped (see [`path_label`]) so the whole segment fits in `NAME_MAX`.
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
