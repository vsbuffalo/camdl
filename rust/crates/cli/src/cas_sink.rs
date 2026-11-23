//! The digest that names a simulation cell's observation blocks
//! ([`obs_subtree_hash`]), which `simulate` folds into each `Obs`
//! by-product's identity.

/// The obs subtree's identity component.
///
/// A hash of the resolved observation blocks (run-spec: changing a reporting
/// parameter re-samples obs without invalidating the cached trajectory), plus
/// any `--emit-every` override (gh#656) — which changes WHICH times the subtree
/// carries, so two cadences must not address the same directory (the
/// count-in-the-key rule).
///
/// One function, because the hash is computed twice for every cell — once to
/// declare the `obs` child in `run.json`, once to name the subtree directory —
/// and a disagreement between the two would declare one child while writing
/// another. Byte-identical to the historical `sha256(json(observations))`
/// whenever no override is in play, so an unflagged run addresses exactly the
/// subtree it always did.
pub(crate) fn obs_subtree_hash(
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    let mut buf = serde_json::to_string(observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    if let Some(e) = emit {
        buf.push_str(&e.identity_repr());
    }
    // The file FORMAT is part of what the subtree holds (gh#833): the header
    // names the declared columns, a windowed stream carries two boundaries,
    // and a row outside the run is not written. Bytes that change under an
    // unchanged key would be a stored artifact the key no longer describes,
    // so the format version enters the key.
    buf.push_str(OBS_FILE_FORMAT);
    Ok(crate::hashing::sha256_hex(buf.as_bytes()))
}

/// The obs-file format the CAS subtree is written in — bumped whenever the
/// bytes written for an unchanged model would change (the count-in-the-key
/// rule).
///
/// 3 (D59): a row's value is drawn from the state, and with the time, where
/// the row is scored (its instant, or its period's stop), not at its label.
/// For a stream whose label is not its scoring time (`day`, `starting_on`,
/// `ending_on`) whose likelihood reads the state or `t`, the stored draws
/// change, so a subtree written before must not answer for one written now.
const OBS_FILE_FORMAT: &str = "\nobs-file-format:3";
