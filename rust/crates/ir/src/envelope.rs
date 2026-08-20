//! gh#audit-C8. IR envelope wrapper that enforces a version handshake
//! at the OCaml↔Rust boundary.
//!
//! Before this commit, `ir/schema.json` and `ir/VERSION` were declared
//! as "the contract" in CLAUDE.md but referenced nowhere in source.
//! Both sides hand-mirrored the IR shape; drift manifested as
//! `serde::Error` at golden-test time (best case) or
//! wrong-but-parseable simulation (worst case).
//!
//! The envelope makes the handshake real:
//!
//! - `ir_version` — must match `IR_VERSION` (loaded from `ir/VERSION`
//!   at compile time via `include_str!`). Mismatch → `IrError::
//!   VersionMismatch`.
//! - `validated_by` — optional marker emitted by the OCaml compiler
//!   describing which validator it ran (e.g. "ocaml-compiler-v0.4").
//!   Rust's `validate.rs` checks the marker; if present, can skip
//!   OCaml-mirrored structural checks (audit H14). For now, opaque
//!   string passed through.
//! - `model` — the existing `Model` shape, unchanged.
//!
//! Long-term the goal is to generate `schema.json` from one
//! authoritative side (Option B in the proposal). This commit
//! establishes the version envelope so that subsequent IR changes
//! must bump VERSION and CI catches drift.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use crate::Model;
use crate::parameter::DocBlock;

/// The model's `#'` documentation dictionary, carried at the envelope level —
/// presentation metadata that sits **outside** `Model`, and therefore outside
/// the content-addressed `run_id`. Maps a base declaration name → its doc, by
/// category. Empty for an undocumented model. A downstream consumer (the fit
/// sidecar, a plot, a report) labels any output column by joining the column
/// name against this index.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelDocs {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, DocBlock>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub compartments: BTreeMap<String, DocBlock>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub transitions: BTreeMap<String, DocBlock>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observations: BTreeMap<String, DocBlock>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dimensions: BTreeMap<String, DocBlock>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub quantities: BTreeMap<String, DocBlock>,
}

impl ModelDocs {
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
            && self.compartments.is_empty()
            && self.transitions.is_empty()
            && self.observations.is_empty()
            && self.dimensions.is_empty()
            && self.quantities.is_empty()
    }
}

/// IR schema version, baked at compile time from `ir/VERSION`.
/// `trim()`-ed at use sites because the file ends with a trailing newline.
pub const IR_VERSION: &str = include_str!("../../../../ir/VERSION");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrEnvelope {
    /// IR schema version. Must match `IR_VERSION` (after trim).
    pub ir_version: String,
    /// Optional: validator that produced this IR. None when emitted
    /// from a hand-edited JSON or from Rust's `to_string_pretty`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validated_by: Option<String>,
    /// The actual model.
    pub model: Model,
    /// `#'` documentation dictionary (presentation metadata). Absent for an
    /// undocumented model; never part of `run_id` (it sits outside `Model`).
    #[serde(default, skip_serializing_if = "ModelDocs::is_empty")]
    pub docs: ModelDocs,
}

#[derive(Debug, thiserror::Error)]
pub enum IrError {
    #[error("IR version mismatch: this build expects {expected}, JSON declared {found}. \
             The IR is incompatible — rebuild OCaml side (`make build-ocaml`) and \
             re-emit any persisted IR JSON (`make update-golden`).")]
    VersionMismatch { expected: String, found: String },
    #[error("IR JSON parse error: {0}")]
    Parse(String),
}

impl IrEnvelope {
    /// Wrap a `Model` in the envelope with the current `IR_VERSION`.
    /// `validated_by` is set by the producer (OCaml compiler) — Rust
    /// passes None for hand-emitted IR (e.g. tests).
    pub fn wrap(model: Model, validated_by: Option<String>) -> Self {
        Self {
            ir_version: IR_VERSION.trim().to_string(),
            validated_by,
            model,
            docs: ModelDocs::default(),
        }
    }

    /// Unwrap to a `Model`, asserting version matches. Discards `docs` (use
    /// [`crate::envelope_from_str`] when you need the documentation dictionary).
    pub fn into_model_checked(self) -> Result<Model, IrError> {
        let expected = IR_VERSION.trim();
        if self.ir_version != expected {
            return Err(IrError::VersionMismatch {
                expected: expected.to_string(),
                found:    self.ir_version,
            });
        }
        let mut model = self.model;
        // gh#616: restore the NaN horizon an unresolved anchor implies (JSON has
        // no NaN literal, so it travelled as `null`).
        model.restore_unresolved_horizons();
        Ok(model)
    }
}

#[cfg(test)]
mod doc_dict_tests {
    use super::*;
    use crate::parameter::DocBlock;

    #[test]
    fn model_docs_round_trips_and_omits_empty_categories() {
        let mut docs = ModelDocs::default();
        assert!(docs.is_empty());
        docs.parameters.insert(
            "beta".into(),
            DocBlock { text: Some("rate".into()), symbol: Some("β".into()), reference: None },
        );
        assert!(!docs.is_empty());
        // A generated quantity carries docs too (gh#318) — same DocBlock shape.
        docs.quantities.insert(
            "peak_prev".into(),
            DocBlock { text: Some("peak prevalence".into()), symbol: Some("π_max".into()), reference: None },
        );
        let json = serde_json::to_string(&docs).unwrap();
        // Only the populated categories appear (skip_serializing_if on each map).
        assert!(json.contains("parameters"), "{json}");
        assert!(json.contains("quantities"), "{json}");
        assert!(!json.contains("compartments"), "{json}");
        let back: ModelDocs = serde_json::from_str(&json).unwrap();
        assert_eq!(docs, back);
    }
}
