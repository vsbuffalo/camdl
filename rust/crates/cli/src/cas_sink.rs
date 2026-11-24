//! The digest that names a simulation cell's observation blocks
//! ([`obs_subtree_hash`]), which `simulate` folds into each `Obs`
//! by-product's identity.

/// The digest of a cell's observation blocks, which an `Obs` by-product's
/// identity holds (`sim_store::ObsIdentity::observation`).
///
/// It hashes the resolved observation blocks, which are not part of the
/// cell's own identity: a change to a reporting parameter draws new
/// observations from the stored trajectory without recomputing it. It also
/// hashes any `--emit-every` override, which changes the times observations
/// are drawn at, so two cadences are two by-products, and [`OBS_FILE_FORMAT`],
/// so bytes that change for an unchanged model are stored under a new key.
pub(crate) fn obs_subtree_hash(
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    let mut buf = serde_json::to_string(observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    if let Some(e) = emit {
        buf.push_str(&e.identity_repr());
    }
    // The file format is part of what the by-product holds: the header
    // names the declared columns, a windowed stream carries two boundaries,
    // and a row outside the run is not written. Bytes that change under an
    // unchanged key would be a stored artifact the key no longer describes,
    // so the format version enters the key.
    buf.push_str(OBS_FILE_FORMAT);
    Ok(crate::hashing::sha256_hex(buf.as_bytes()))
}

/// The format the observation files are written in — bumped whenever the
/// bytes written for an unchanged model would change (the count-in-the-key
/// rule).
///
/// 3 (D59): a row's value is drawn from the state, and with the time, where
/// the row is scored (its instant, or its period's stop), not at its label.
/// For a stream whose label is not its scoring time (`day`, `starting_on`,
/// `ending_on`) whose likelihood reads the state or `t`, the stored draws
/// change, so files written before must not answer for files written now.
const OBS_FILE_FORMAT: &str = "\nobs-file-format:3";
