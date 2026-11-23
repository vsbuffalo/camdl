//! The structural hash of a compiled model: what makes two models the same
//! model, independent of their name, quantities, contrasts and the
//! compiler-derived gradient maps.
//!
//! [`model_ir_hash`] is the model identity `camdl` prints and compares: the
//! `model_hash` of a PGAS trajectory header, the check that a model given to
//! `predict` or `simulate --fit` is the one a fit ran on, and `camdl diff`'s
//! "same model" line. No stored result's identity holds it: a result keys on
//! the digest of its stored compiled model, the gradient maps included.
//!
//! The encoding is [`hash`]'s, and the fields each IR type contributes are
//! listed in its hand-written [`ContentAddressed`] impl (`impls`). A change to
//! either changes every model's hash.

mod hash;
mod impls;

pub use hash::{CanonicalHasher, ContentAddressed, ContentHash, HexError, HoistedBodies, HASH_VERSION};

/// The model's structural hash: SHA-256 of [`HASH_VERSION`] and the model's
/// canonical bytes, without the gradient maps.
pub fn model_ir_hash(model: &crate::Model) -> ContentHash {
    model.content_hash()
}
