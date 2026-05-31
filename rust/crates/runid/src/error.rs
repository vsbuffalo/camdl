//! Error taxonomy for the resolution → run → commit pipeline.
//!
//! M1 defines only the variants the hasher needs (a non-finite resolved
//! float must be a value, not a panic). The remaining `ResolveError`
//! variants from the proposal (`CompileFailed`, `ParamParse`,
//! `UnknownScenario`, `FileNotFound`, `BadDesignSpec`) and the `RunError`
//! wrapper land with the CLI wiring in M2/M3, when there is a resolver to
//! produce them.

use crate::float::NonFiniteFloat;

/// A failure turning raw CLI/TOML into a fully-resolved leaf input. Failures
/// are values surfaced before any hashing, preserving the totality of the
/// hash pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    /// A resolved param/dt/bound was `NaN` or `±Inf`. Surfaced before the
    /// value could reach a hash.
    #[error("non-finite resolved parameter: {0}")]
    NonFiniteParam(#[from] NonFiniteFloat),
}
