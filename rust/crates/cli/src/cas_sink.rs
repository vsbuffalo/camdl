//! Label helpers of the fit store (`fit run`, `camdl label`), and the
//! digest that names a simulation cell's observation blocks
//! ([`obs_subtree_hash`]), which `simulate` folds into each `Obs`
//! by-product's identity.

/// Rewrite a leaf's `run.json` through `edit`, which returns whether it changed
/// the record: the store's one record-update path ([`runid::update_record`]),
/// under the leaf's `.lock` and through the reserved `run.json.tmp`, so no two
/// updates — and no update and an `augment` — interleave, and a crash leaves
/// only files the store's exact-set check ignores (gh#896).
fn rewrite_record(
    leaf_dir: &std::path::Path,
    edit: impl FnOnce(&mut runid::RunRecord) -> Result<bool, String>,
) -> Result<(), String> {
    runid::update_record(leaf_dir, edit).map_err(|e| match e {
        runid::CasError::EditRefused { reason, .. } => reason,
        other => format!("update {}: {other}", leaf_dir.join("run.json").display()),
    })
}

/// Ensure a leaf's `run.json` carries `provenance.label == Some(label)`,
/// rewriting it only when it differs. `provenance` is metadata, not identity —
/// the run_id and exact-set are untouched — applied at write time so an
/// explicit `simulate --label` survives a CAS cache hit, which relabels a sim.
/// Idempotent: a no-op when the leaf already has the requested label. Fit
/// runs use [`set_fit_label`], which also stamps the time.
pub(crate) fn ensure_provenance_label(leaf_dir: &std::path::Path, label: &str) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        if rec.provenance.label.as_deref() == Some(label) {
            return Ok(false);
        }
        rec.provenance.label = Some(label.to_string());
        Ok(true)
    })
}



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
