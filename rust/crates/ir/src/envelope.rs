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
    /// The model's own doc: the file-header `#'` block (gh#750) — what the
    /// model is, what it is fitted to, what it branches from. Not a map, since
    /// there is one model. `None` when the file opens with no `#'` block.
    ///
    /// This is the answer to "what is this model?" that does not require
    /// opening the `.camdl`. It is deliberately envelope metadata rather than a
    /// second `Model::description`: a description inside `Model` re-keys every
    /// fit when corrected, and a description nobody dares correct is worse than
    /// none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<DocBlock>,
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
        self.model.is_none()
            && self.parameters.is_empty()
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
        // The model's own slot is absent, not an empty block, when undocumented.
        assert!(!json.contains("\"model\""), "{json}");
        let back: ModelDocs = serde_json::from_str(&json).unwrap();
        assert_eq!(docs, back);
    }

    /// gh#750: the model's own `#'` block. A multi-line header — prose plus the
    /// `@base`/`@adds`/`@changes` lineage lines the compiler keeps as free text
    /// — must survive the round trip with its line structure intact, because
    /// that text IS the answer to "what is this model" for every consumer that
    /// reads the sidecar instead of the `.camdl`.
    #[test]
    fn model_docs_carry_the_models_own_doc_block() {
        let mut docs = ModelDocs::default();
        assert!(docs.is_empty(), "a default dictionary documents nothing");
        docs.model = Some(DocBlock {
            text: Some(
                "National SEIR with a facility-death delay.\n\
                 @base bvd_national_twocfr.camdl\n\
                 @adds nothing"
                    .into(),
            ),
            symbol:    None,
            reference: Some("Camacho et al. 2015".into()),
        });
        // A model doc alone makes the dictionary non-empty, so the envelope
        // serializes `docs` for a model that documents only itself.
        assert!(!docs.is_empty());
        let json = serde_json::to_string(&docs).unwrap();
        assert!(json.contains("\"model\""), "{json}");
        assert!(!json.contains("parameters"), "{json}");
        let back: ModelDocs = serde_json::from_str(&json).unwrap();
        assert_eq!(docs, back);
        let text = back.model.unwrap().text.unwrap();
        assert_eq!(
            text.lines().count(),
            3,
            "the lineage lines stay on their own lines: {text:?}"
        );
    }

    /// The cross-language contract, on a committed golden the OCaml compiler
    /// actually emitted: `ocaml/golden/sir_basic.camdl` opens with a `#'` header
    /// block, so its golden IR must carry `docs.model` and Rust must read it
    /// under that key. A key-name disagreement between the two sides is
    /// otherwise silent — Rust would simply see `None` and every consumer would
    /// show an undocumented model.
    #[test]
    fn the_ocaml_compiler_emits_the_model_doc_under_the_key_rust_reads() {
        let json = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../ocaml/golden/sir_basic.ir.json"
        ))
        .expect("read the sir_basic golden");
        let env: IrEnvelope = serde_json::from_str(&json).expect("parse the golden envelope");
        let doc = env.docs.model.expect("sir_basic's header block reached docs.model");
        let text = doc.text.expect("the header block carries prose");
        assert!(
            text.contains("susceptible individuals become infectious"),
            "prose survived the compile: {text:?}"
        );
        // The lineage lines are free text kept on their own lines (gh#750).
        assert!(text.lines().any(|l| l.starts_with("@base ")), "{text:?}");
        assert_eq!(doc.reference.as_deref(), Some("Kermack and McKendrick 1927"));
        // The parameter docs of the same model are untouched by the new slot.
        assert!(env.docs.parameters.contains_key("beta"));
    }

    /// A model-level doc is read off the envelope and leaves the model alone.
    /// Run on a real committed model (`ir/golden/sir_basic.ir.json`) rather
    /// than a stub, so it covers every field a model actually carries. The
    /// run-identity half of the claim — that the hash does not move — is
    /// asserted in `runid` (`ir_hash::tests`), which owns `model_ir_hash`.
    #[test]
    fn a_model_doc_parses_off_a_real_envelope_without_touching_the_model() {
        let plain = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../ir/golden/sir_basic.ir.json"
        ))
        .expect("read sir_basic.ir.json");
        // Splice a model-level doc into the envelope, leaving `model` untouched.
        let documented = plain.replacen(
            "\"model\":",
            "\"docs\":{\"model\":{\"text\":\"what this model is\"}},\"model\":",
            1,
        );
        assert_ne!(plain, documented, "the splice must actually change the JSON");
        let a: IrEnvelope = serde_json::from_str(&plain).expect("parse plain");
        let b: IrEnvelope = serde_json::from_str(&documented).expect("parse documented");
        assert!(a.docs.model.is_none());
        assert_eq!(
            b.docs.model.and_then(|d| d.text).as_deref(),
            Some("what this model is"),
            "the doc parsed off the envelope"
        );
        assert_eq!(a.model, b.model, "the model itself is untouched");
    }
}
