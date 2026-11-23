//! The model identity string a fit's provenance records
//! ([`model_identity_from_ir`]).

/// The model's *structural* content identity, hex-encoded: the model hash
/// ([`ir::model_hash::model_ir_hash`]), rendered as hex. Every recorded
/// "model identity" string is computed here, so two records of one model carry
/// the same string.
///
/// Takes the **raw compiled IR JSON**, not an in-memory `ir::Model`: a command
/// that reads a fit config seeds `[estimate].start` into its working model, so
/// hashing the parsed model would make the identity depend on a start value.
/// Hashing the raw IR, the .camdl's compiled output, keeps the identity
/// tracking the model.
///
/// It folds the model IR only, not the build or the IR schema string, so two
/// builds of camdl record the same identity for the same model. It is not the
/// identity a stored result keys on, which is the digest of its stored compiled
/// model, derivative tables included (`camdl_store::model_part_digest`): two
/// compiles that differ only in their gradient maps have one string here and
/// two keys there.
///
/// Returns an empty string when the IR can't be parsed (a partially-written or
/// absent IR, as the fit sidecar tolerates).
pub fn model_identity_from_ir(model_ir_json: &str) -> String {
    match ir::from_str(model_ir_json) {
        Ok(model) => ir::model_hash::model_ir_hash(&model).to_hex(),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::tiny_model;
