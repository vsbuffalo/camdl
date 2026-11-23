//! The model identity string a fit's provenance records
//! ([`model_identity_from_ir`]).

/// The model's *structural* content identity, hex-encoded: the `runid` model
/// content hash ([`ModelDigest::ir`] = `Model::content_hash`), rendered as
/// hex. Every recorded "model identity" string is computed here, so two
/// records of one model carry the same string.
///
/// Takes the **raw compiled IR JSON**, not an in-memory `ir::Model`: a command
/// that reads a fit config seeds `[estimate].start` into its working model, so
/// hashing the parsed model would make the identity depend on a start value.
/// Hashing the raw IR, the .camdl's compiled output, keeps the identity
/// tracking the model.
///
/// It folds the model IR only, not the engine version or the IR schema
/// string. This is `model_digest`'s `ir` component, not the full per-level
/// [`ModelDigest`], which also folds `engine` (the `ENGINE_ID` build
/// fingerprint). Two builds of camdl therefore record the same identity for
/// the same model.
///
/// Returns an empty string when the IR can't be parsed (a partially-written or
/// absent IR, as the fit sidecar tolerates).
pub fn model_identity_from_ir(model_ir_json: &str) -> String {
    match ir::from_str(model_ir_json) {
        // `model_ir_hash` owns the presentation strip — the same function
        // `ModelDigest::from_model` uses, so this string and the `model` level
        // can never disagree about what "the same model" means (gh#442).
        Ok(model) => runid::inputs::model_ir_hash(&model).to_hex(),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::tiny_model;
