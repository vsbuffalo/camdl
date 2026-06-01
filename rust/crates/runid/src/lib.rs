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
