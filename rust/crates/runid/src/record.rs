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
/// mtime). `digest` is the file's SHA-256, recorded in `run.json` for
/// integrity tooling (`camdl verify`); it is NOT checked on read today — no
/// read path (`camdl cat`, the readers) recomputes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChecksum {
    pub bytes: u64,
    /// `"{secs}.{nanos:09}"` since the Unix epoch — a deterministic,
    /// exactly-comparable encoding of the file's mtime.
    pub mtime: String,
    pub digest: ContentHash,
}

/// The semantic role a column plays in a camdl output table — a small closed
/// vocabulary so a consumer can render any tabular output without
/// reverse-engineering its header. `Time` (a physical/calendar axis) and
/// `Iteration` (a sampler/optimizer axis) are deliberately distinct: a
/// trajectory's x-axis is physical time, a trace's is a sampler index, and
/// conflating them is a real rendering bug. Wire form is the snake-case tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    /// Physical/calendar axis: `t`, `time`, `date`.
    Time,
    /// Sampler/optimizer axis: `sweep`, `step`, `draw`, `iteration`, `point_id`.
    Iteration,
    /// MCMC chain key.
    Chain,
    /// Ensemble/batch replicate key.
    Replicate,
    /// Scenario key.
    Scenario,
    /// A stratification key: `patch`, `age`.
    Dimension,
    /// A compartment count/value: `S`, `I`, `R`.
    State,
    /// A transition flow: `flow_infection`.
    Flow,
    /// A per-stream incidence: `inc_<stream>`.
    Incidence,
    /// A sampled (estimated) model parameter.
    ParamEstimated,
    /// A held-constant (fixed) model parameter.
    ParamFixed,
    /// An observation stream's value.
    Observable,
    /// A predictive quantile band: `q05` … `q95`.
    Quantile,
    /// A sampler/fit diagnostic: `loglik`, `log_posterior`, `ESS`, `rhat`,
    /// `accepted`, `n_divergent`, ….
    Diagnostic,
}

/// What kind of table an output file is — the consumer's default view. Wire
/// form is the snake-case tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableRole {
    /// Time × state (compartments, flows): `traj.tsv`, `trajectories.tsv`.
    Trajectory,
    /// Time × observable: `obs.tsv`.
    Observation,
    /// The thinned posterior-draws cloud: `draws.tsv`.
    PosteriorCloud,
    /// The full per-chain trace of a sampler or optimizer: `chain_N/trace.tsv`
    /// (MCMC) or `chain_N/parameter_traces.tsv` (if2 / nlopt).
    Trace,
    /// Predicted-vs-observed bands: `predictive/<stream>.tsv`.
    Predictive,
    /// A parameter-grid evaluation: `landscape.tsv`, `profile.tsv`.
    Landscape,
}

/// One column of a camdl output table: its on-disk name and its semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    pub role: ColumnRole,
}

/// The column schema of one tabular output file: what kind of table it is and,
/// in file order, what each column means. Recorded in `run.json.output_schema`
/// keyed by the file's leaf-relative path (`{n}` = per-chain wildcard).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableSchema {
    pub role: TableRole,
    pub columns: Vec<ColumnSpec>,
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
    /// The IR schema version (e.g. `"0.7"`) — a string, matching `ir/VERSION`.
    pub ir_version: String,
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
    /// Column schema for the leaf's tabular outputs, keyed by leaf-relative
    /// path (`{n}` = per-chain wildcard). Recorded, NOT hashed — lets a
    /// consumer render any output (find the x-axis, group by chain, facet by
    /// dimension) without reverse-engineering a TSV header. Empty when the
    /// command declares no schema.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub output_schema: BTreeMap<String, TableSchema>,
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

#[cfg(test)]
mod output_schema_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn table_schema_roundtrips_with_snake_case_wire_tags() {
        let schema = TableSchema {
            role: TableRole::Trace,
            columns: vec![
                ColumnSpec { name: "sweep".to_string(), role: ColumnRole::Iteration },
                ColumnSpec { name: "log_posterior".to_string(), role: ColumnRole::Diagnostic },
                ColumnSpec { name: "beta".to_string(), role: ColumnRole::ParamEstimated },
            ],
        };
        let json = serde_json::to_string(&schema).unwrap();
        // Wire tags are snake_case — the contract a consumer reads.
        for tag in ["\"trace\"", "\"iteration\"", "\"diagnostic\"", "\"param_estimated\""] {
            assert!(json.contains(tag), "missing {tag} in {json}");
        }
        let back: TableSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, back);
    }

    #[test]
    fn empty_output_schema_is_omitted() {
        // The additive field skips serialization when empty, so existing
        // manifests and readers are unaffected.
        let record = RunRecord {
            format_version: FORMAT_VERSION,
            kind: ArtifactKind::Sim,
            run_id: ContentHash::digest_bytes(b"x"),
            hash_version: 1,
            ir_version: "0.0".to_string(),
            engine_version: "test".to_string(),
            levels: vec![],
            deps: vec![],
            status: RunStatus::Completed,
            artifacts: BTreeMap::new(),
            output_schema: BTreeMap::new(),
            children: BTreeMap::new(),
            inputs: serde_json::Value::Null,
            provenance: Provenance::default(),
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(!json.contains("output_schema"), "empty schema must be omitted: {json}");
    }
}
