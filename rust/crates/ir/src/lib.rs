pub mod anchor;
pub mod caltime;
pub mod contrast;
pub mod deriv;
pub use deriv::{Diffable, Differentiable, GradMap};
pub use differentiate_derive::Differentiate;
pub mod envelope;
pub mod expr;
pub mod init_order;
pub mod intervention;
pub mod model;
pub mod observation;
pub mod ode_equation;
pub mod parameter;
pub mod quantity;
pub mod table;
pub mod time_func;
pub mod transition;
pub mod validate;

pub use envelope::{IrEnvelope, IrError, ModelDocs, IR_VERSION};
pub use model::Model;

/// Deserialise a `Model` from a JSON string. gh#audit-C8: enforces
/// the `IrEnvelope` wrapper with version handshake. Returns
/// `IrError::VersionMismatch` if the envelope's `ir_version` doesn't
/// match the `IR_VERSION` baked from `ir/VERSION` at compile time.
pub fn from_str(s: &str) -> Result<Model, IrError> {
    let env: IrEnvelope = serde_json::from_str(s)
        .map_err(|e| IrError::Parse(e.to_string()))?;
    env.into_model_checked()
}

/// Deserialise the full `IrEnvelope` (model **plus** the envelope-level `docs`
/// dictionary), enforcing the same version handshake as [`from_str`]. Use this
/// when you need `#'` documentation (e.g. building the fit sidecar or labelling
/// plot axes); most callers want [`from_str`], which discards `docs`.
pub fn envelope_from_str(s: &str) -> Result<IrEnvelope, IrError> {
    let mut env: IrEnvelope = serde_json::from_str(s)
        .map_err(|e| IrError::Parse(e.to_string()))?;
    let expected = IR_VERSION.trim();
    if env.ir_version != expected {
        return Err(IrError::VersionMismatch {
            expected: expected.to_string(),
            found:    env.ir_version,
        });
    }
    // gh#616: the same post-decode normalisation `into_model_checked` applies.
    env.model.restore_unresolved_horizons();
    Ok(env)
}

/// Deserialise a `Model` from a JSON reader. gh#audit-C8: same
/// envelope check as `from_str`.
pub fn from_reader<R: std::io::Read>(r: R) -> Result<Model, IrError> {
    let env: IrEnvelope = serde_json::from_reader(r)
        .map_err(|e| IrError::Parse(e.to_string()))?;
    env.into_model_checked()
}

/// Re-emit a compiled model after changing it: the envelope read from
/// `original` with its model replaced by `model`, pretty-printed. The
/// envelope's `docs` and `declared` sections are carried through unchanged
/// ([`IrEnvelope::with_model`]). This is the one way a command writes back a
/// compiled model it read; there is deliberately no function that serialises a
/// bare `Model` into an envelope, because that would drop both sections
/// without the caller deciding to.
///
/// `original` must be an envelope of the current IR version.
pub fn reemit_with_model(original: &str, model: &Model) -> Result<String, IrError> {
    let env = envelope_from_str(original)?.with_model(model.clone());
    serde_json::to_string_pretty(&env).map_err(|e| IrError::Parse(e.to_string()))
}
