//! `runid` — content-addressed run identity for camdl.
//!
//! Every artifact camdl produces is the output of a pure function of a
//! complete, typed input set; its identity is the structural hash of that
//! set. This crate owns the *identity* half of that contract:
//!
//! - [`CanonicalHasher`] + [`ContentHash`] + [`ContentAddressed`] — the one
//!   pinned, structural hash function and the trait that drives it
//!   ([`hash`] module). [`HASH_VERSION`] migrates the whole encoding.
//! - [`FiniteF64`] — the resolved-input float policy (reject non-finite,
//!   normalize `-0.0`); structural IR floats use the raw-bits policy
//!   instead ([`float`] module).
//! - [`ArtifactKind`] + [`run_id`] — the store's top partition and the leaf
//!   address ([`kind`] module).
//!
//! `runid` depends only on `ir`; the CLI depends on `runid`. The hand-written
//! `ContentAddressed` impls for the `ir` type tree (`ir_hash` module) make
//! the foreign IR hashable under the structural-float policy; the
//! `#[derive(RunInput)]` macro (crate `runid-derive`) generates the same
//! encoding for run-input types and is validated against the hand impls.
//!
//! ## Two hashing paths (read before adding an identity input)
//!
//! A leaf's identity is computed one of two ways, and they behave differently
//! — know which one your change touches:
//!
//! 1. **Structural `#[derive(RunInput)]`** (this crate). [`inputs::SimConfig`],
//!    [`inputs::TrajectoryInput`], [`inputs::FitDigest`], … hash field by field
//!    in declaration order. The derive folds **every** non-provenance field,
//!    *include by default* — there is no skip-if-default. Mark a field
//!    `#[run_input(provenance)]` to exclude it entirely (display only). The
//!    `simulate` / `batch` `config` level uses this path.
//! 2. **Hash of canonical JSON** (CLI `fit/cas.rs`). `FitConfigV2` is serde
//!    only; it enters identity as the digest of its key-sorted JSON (the
//!    `FitDigest::fit_toml` field), so `#[serde(skip_serializing_if = …)]`
//!    controls hash *membership* — a default-skipped field stays out of the
//!    hash and does not re-key. The `fit` level uses this path.
//!
//! ## Adding a field to identity
//!
//! - **Classify it.** A change to *which values are computed or stored* (a new
//!   semantic input, or an output subset that changes the leaf's bytes) belongs
//!   in identity: a content-addressed leaf cannot share a `run_id` with one
//!   whose bytes differ. A pure *re-encoding* of the same values (format, time
//!   rendering) is presentation — strip it in `inputs::normalize_for_hash`,
//!   which [`inputs::model_ir_hash`] (and therefore
//!   [`inputs::ModelDigest::from_model`]) applies on **every** identity path,
//!   so a new artifact kind cannot silently opt out (gh#442; see
//!   `output.format`). Never re-implement the strip at a call site.
//! - **Expect turnover.** Adding a field to a `RunInput` struct re-keys *every*
//!   existing leaf of that kind, even at its default value (the field always
//!   contributes bytes). That is intentional, versioned turnover — not a bug to
//!   engineer around. To scope it, bump the version that matches: a single
//!   struct's `#[run_input(schema_version = N)]`, the crate-wide
//!   [`HASH_VERSION`] (re-keys *everything*), or `ir/VERSION` (re-keys all
//!   model-bearing leaves).
//! - **Gate it.** Any re-key is a reviewed change pinned by a `run_id`-stability
//!   test, never collateral.

// The `#[derive(RunInput)]` macro emits `runid::ContentAddressed` /
// `runid::CanonicalHasher` paths so the same expansion compiles in consumer
// crates *and* here, where the digest types are derived. This alias makes
// `runid::…` resolve to the current crate.
extern crate self as runid;

pub mod error;
pub mod float;
pub mod hash;
pub mod inputs;
pub mod ir_hash;
pub mod kind;
pub mod layout;
pub mod record;
pub mod store;

pub use error::ResolveError;
pub use float::{FiniteF64, NonFiniteFloat};
pub use hash::{CanonicalHasher, ContentAddressed, ContentHash, HexError, HASH_VERSION};
pub use kind::{run_id, ArtifactKind};
pub use layout::{path_label, segment, store_path};
pub use record::{FileChecksum, LevelId, Provenance, RunRecord, RunStatus, FORMAT_VERSION};
pub use runid_derive::RunInput;
pub use store::{Artifacts, CasError, CasStore, FsCasStore, LeafIdentity, Lookup, StaleReason};

#[cfg(test)]
mod macro_eq;
#[cfg(test)]
mod tests;
