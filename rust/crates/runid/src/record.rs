//! `RunRecord` — the leaf's `run.json`: its metadata and the cache-validity
//! gate.
//!
//! Hashes *address and verify*; `provenance` and `inputs` are
//! recorded-not-hashed (the readable mirror `show` renders). **`RunRecord`
//! is never hashed** — identity comes only from the `RunInput`; `children`,
//! `artifacts`, and `provenance` are recorded-only, which is *why* adding an
//! `obs/` child cannot change the trajectory's `run_id`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::hash::ContentHash;
use crate::inputs::ArtifactRef;
use crate::kind::ArtifactKind;

/// The current `run.json` schema version. A clean break bumps this; at alpha
/// there is no backward-compatible deserialization path.
pub const FORMAT_VERSION: u16 = 1;

/// Run lifecycle state. Wire form is the snake-case string
/// (`"running"`/`"completed"`/`"failed"`), matching the `run.json` example.
/// A cache hit requires `Completed`; `Running` is an in-flight (or crashed)
/// streamed run; `Failed` records a run that ran to a definite failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
}

/// One factored identity level, in path order: a readable label
/// (provenance) plus the level's `ContentHash` (identity) and its schema
/// version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelId {
    pub name: String,
    pub label: String,
    pub hash: ContentHash,
    pub schema_version: u16,
}

/// Checksum of one of the leaf's OWN files. `bytes` + `mtime` are the cheap
/// gate (a *performance* optimization, coarse — `cp -p`/`rsync` preserve
/// mtime); `digest` is the "never serve wrong bytes" guarantee, verified at
/// consume time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChecksum {
    pub bytes: u64,
    /// `"{secs}.{nanos:09}"` since the Unix epoch — a deterministic,
    /// exactly-comparable encoding of the file's mtime.
    pub mtime: String,
    pub digest: ContentHash,
}

/// Recorded-not-hashed provenance: the readable mirror `show` renders.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub argv: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camdl_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_paths: Vec<String>,
}

/// The leaf's `run.json`. Read by prefix resolution and `show`/`cat`, never
/// the path. Identity (`run_id`, `levels`) addresses + verifies; everything
/// else is recorded-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub format_version: u16,
    pub kind: ArtifactKind,
    pub run_id: ContentHash,
    pub hash_version: u16,
    pub ir_version: u32,
    pub engine_version: String,
    /// The factored identity, in path order.
    pub levels: Vec<LevelId>,
    /// Lineage — the consumed upstream artifacts (`{run_id, kind, artifact,
    /// digest}`). Empty for a leaf with no upstreams.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deps: Vec<ArtifactRef>,
    pub status: RunStatus,
    /// EXACT-SET over the leaf's OWN files only (not the declared children).
    pub artifacts: BTreeMap<String, FileChecksum>,
    /// Declared child sub-artifacts: namespace (`obs`, `paths`, …) → child
    /// `run_id`s. Recorded, NOT hashed; recognized as children (not orphans)
    /// by the exact-set check; validated recursively on their *own* lookup.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub children: BTreeMap<String, Vec<ContentHash>>,
    /// Resolved-input summary for display/audit — provenance, not hashed.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub inputs: serde_json::Value,
    #[serde(default)]
    pub provenance: Provenance,
}

impl RunRecord {
    /// `true` iff `self`'s identity (run_id) matches `other`'s — the identity
    /// gate. A path may hold a record whose full hashes differ from the
    /// expected identity (a short-hash path collision); that is a
    /// `PathPrefixCollision`, not a hit.
    pub fn identity_matches(&self, run_id: &ContentHash) -> bool {
        &self.run_id == run_id
    }
}
