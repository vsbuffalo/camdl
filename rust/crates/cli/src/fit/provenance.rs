//! Provenance hashing: input_hash (computation identity) and content_hash (tamper detection).

use sha2::{Sha256, Digest};

use crate::version;

// `compute_input_hash` moved to `crate::hashing::fit_input_hash`
// as part of the 2026-04-19 output-tree unification. Call sites
// updated in lockstep.

/// MLE-params tamper-detection hash. Hashes the canonicalised (name,
/// value) pairs from an `mle_params.toml` file so we can detect post-
/// write edits to the parameter values. Naming note: this is **not**
/// a general content-hash helper (unlike `crate::hashing::file_hash` or
/// `sha256_hex`); it's mle-specific because it canonicalises numeric
/// formatting with `{:.12}` precision before hashing. Lives in
/// `fit::provenance` (not `crate::hashing`) for that reason.
pub fn mle_params_tamper_hash(params: &std::collections::BTreeMap<String, f64>) -> String {
    let mut pairs: Vec<(&String, &f64)> = params.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    let mut h = Sha256::new();
    for (name, value) in pairs {
        h.update(format!("{}={:.12}\x00", name, value).as_bytes());
    }
    hex::encode(&h.finalize()[..4]) // 8 hex chars
}

/// Structured provenance for an `mle_params.toml` file, serialized
/// as a `[provenance]` TOML block. Downstream tools (particularly
/// `camdl simulate --params`) read these fields to close the
/// backend-provenance loop — see
/// `docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`.
///
/// Only fields that are `Some` get serialized; legacy fit-produced
/// files may omit any of the optional fields.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MleProvenance {
    pub camdl_version: String,
    pub timestamp: String,
    /// Tamper-detection hash over the numeric parameter values in the
    /// file below (NOT over this `[provenance]` block). Editing a
    /// parameter invalidates this; editing a provenance field does
    /// not.
    pub content_hash: String,
    /// Full fit-level hash of the originating fit, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_hash: Option<String>,
    /// Dynamics fields — load-bearing for the backend-guardrail.
    pub backend: crate::args::types::ForwardBackend,
    pub dt: f64,
    pub model: String,
    pub model_identity: String,
    /// Per-stream (path, hash). Serialized as a table under
    /// `[provenance.data]`.
    #[serde(default)]
    pub data: std::collections::BTreeMap<String, DataEntry>,
    pub seed: u64,
    /// The method that produced this estimate, by the label of its `method`
    /// store level (`if2`, `nl-sbplx`, …) — the same word the config, the
    /// store path and the CLI use (gh#890).
    pub method: String,
    pub chain: usize,
    pub log_likelihood: f64,
    pub loglik_sd: f64,
    pub n_particles: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ess_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ess_min: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataEntry {
    pub path: String,
    pub hash: String,
}

/// Wrapper type for the full mle_params.toml file: a `[provenance]`
/// block plus a flat params table (rendered as top-level `name = value`
/// lines in the output, not nested under any key).
///
/// We serialize the provenance block via toml::to_string and the
/// params section manually, rather than nesting params under a
/// `[params]` table, to keep the file's "numeric params at the top
/// level" invariant that downstream scripts (and the unified
/// `params_resolver`'s `--fixed-file` loader, which uses
/// `util::load_params_toml`) already rely on. The structured
/// provenance is an additive change.
///
/// Write mle_params.toml with a structured `[provenance]` block.
pub fn write_mle_params(
    path: &str,
    all_params: &std::collections::BTreeMap<String, f64>,
    metadata: &MleMetadata,
) -> Result<(), String> {
    use std::io::Write;
    // The hash covers the values as the file holds them. `format_param_value`
    // writes ten decimals, so a hash over the unrounded `f64` would declare a
    // value no reader of the file can recompute, and a check against it
    // would call every real fit's file edited.
    let as_written: std::collections::BTreeMap<String, f64> = all_params
        .iter()
        .map(|(k, v)| {
            let written = crate::fit::runner::format_param_value(*v);
            written.parse::<f64>().map(|read_back| (k.clone(), read_back)).map_err(|_| {
                format!("cannot write {}: `{}` formats to {:?}, which is not a number", path, k, written)
            })
        })
        .collect::<Result<_, String>>()?;
    let content_hash = mle_params_tamper_hash(&as_written);
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;

    let data_map: std::collections::BTreeMap<String, DataEntry> = metadata
        .data_hashes.iter()
        .map(|(name, hash)| {
            // The stored name is either bare ("cases") or
            // "cases (path/to/cases.tsv)" in the legacy comment form.
            // Parse out the path if it's in the parenthesized form; else
            // leave the path field empty. New MleMetadata callers should
            // pass bare names and separately carry paths.
            let (stream_name, path) = if let Some(open) = name.find(" (") {
                let close = name.rfind(')').unwrap_or(name.len());
                (name[..open].to_string(), name[open+2..close].to_string())
            } else {
                (name.clone(), String::new())
            };
            (stream_name, DataEntry { path, hash: hash.clone() })
        })
        .collect();

    let prov = MleProvenance {
        camdl_version: version::VERSION_SHORT.to_string(),
        timestamp: metadata.timestamp.clone(),
        content_hash: content_hash.clone(),
        fit_hash: Some(metadata.input_hash.clone()),
        backend: metadata.backend,
        dt: metadata.dt,
        model: metadata.model_path.clone(),
        model_identity: metadata.model_identity.clone(),
        data: data_map,
        seed: metadata.seed,
        method: metadata.method.clone(),
        // The runner's chain index is 0-based; surface as 1-indexed in the
        // provenance block, matching the `chain_N/` directories.
        chain: metadata.best_chain_index + 1,
        log_likelihood: metadata.loglik,
        loglik_sd: metadata.loglik_sd,
        n_particles: metadata.n_particles,
        ess_mean: metadata.ess_at_mle.map(|(m, _)| m),
        ess_min: metadata.ess_at_mle.map(|(_, mn)| mn),
    };

    // TOML file order: top-level scalars FIRST, then [provenance]
    // section. A bare `key = value` line that follows a `[table]`
    // header parses as a field of that table, not as a top-level
    // scalar — so putting params after [provenance] would silently
    // re-scope them and break every downstream reader. Writing
    // params first keeps `util::load_params_toml` (used by the
    // `params_resolver`'s `--fixed-file` loader) and any third-party
    // toml reader picking them up at the top level, unchanged.
    writeln!(f, "# camdl fit MLE — parameter values first, followed by a \
                 [provenance] block.").unwrap();
    writeln!(f, "# Editing any value in this params section invalidates \
                 provenance.content_hash.").unwrap();
    let mut pairs: Vec<(&String, &f64)> = all_params.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    for (name, value) in pairs {
        writeln!(f, "{} = {}", name, crate::fit::runner::format_param_value(*value)).unwrap();
    }
    writeln!(f).unwrap();

    // Now the [provenance] block.
    #[derive(serde::Serialize)]
    struct ProvWrapper<'a> { provenance: &'a MleProvenance }
    let prov_toml = toml::to_string(&ProvWrapper { provenance: &prov })
        .map_err(|e| format!("cannot serialize provenance: {}", e))?;
    f.write_all(prov_toml.as_bytes()).unwrap();
    Ok(())
}

/// Read the `[provenance]` block from an mle_params.toml file. Returns
/// None if the file has no such block (legacy format or a non-fit
/// params file); Err on malformed TOML.
pub fn read_mle_provenance(path: &str) -> Result<Option<MleProvenance>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        #[serde(default)]
        provenance: Option<MleProvenance>,
    }
    let w: Wrapper = toml::from_str(&contents)
        .map_err(|e| format!("parse error in {}: {}", path, e))?;
    Ok(w.provenance)
}

/// Whether the parameter values in a fit-written params file still hash to
/// the `content_hash` its `[provenance]` block declares.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentVerification {
    Valid,
    /// The values were edited after the block was written: `declared` is the
    /// block's hash, `computed` the hash of the values now in the file.
    Modified { declared: String, computed: String },
}

/// Recompute the tamper hash over the file's top-level numeric parameters —
/// the values `write_mle_params` hashed into `content_hash` — and compare it
/// with `declared`, the hash of the `[provenance]` block the caller has read.
/// The scope is the parameter values only: editing a provenance field does
/// not invalidate the hash, editing a value does. A `[provenance]` table is
/// skipped exactly as `util::load_params_toml` skips it.
pub fn verify_content_hash(path: &str, declared: &str) -> Result<ContentVerification, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let table: toml::Table = contents.parse()
        .map_err(|e| format!("parse error in {}: {}", path, e))?;
    let params: std::collections::BTreeMap<String, f64> = table
        .into_iter()
        .filter(|(k, _)| k != "provenance")
        .filter_map(|(k, v)| match v {
            toml::Value::Float(f) => Some((k, f)),
            toml::Value::Integer(i) => Some((k, i as f64)),
            _ => None,
        })
        .collect();
    let computed = mle_params_tamper_hash(&params);
    Ok(if computed == declared {
        ContentVerification::Valid
    } else {
        ContentVerification::Modified { declared: declared.to_string(), computed }
    })
}

// `file_content_hash` moved to `crate::hashing::file_hash`; callers
// now use the canonical name directly.

pub struct MleMetadata {
    pub input_hash: String,
    pub model_path: String,
    pub model_identity: String,
    pub data_hashes: Vec<(String, String)>,
    pub seed: u64,
    /// The method label, written to `[provenance] method`.
    pub method: String,
    /// The winning chain as the runner's own **0-based** index, which is what
    /// every caller has in hand. `write_mle_params` adds one when it writes
    /// `[provenance] chain`, so the file is 1-based like the `chain_N/`
    /// directories. Named `_index` so it cannot be confused with
    /// [`crate::fit::state::FitState::best_chain`], which is the 1-based
    /// chain *number* as stored (gh#912).
    pub best_chain_index: usize,
    /// Simulation backend the fit used. Load-bearing for the
    /// backend-provenance guardrail in `camdl simulate --params`
    /// — downstream can only auto-match if we record this.
    pub backend: crate::args::types::ForwardBackend,
    /// Timestep used by the fit. Paired with `backend`.
    pub dt: f64,
    pub loglik: f64,
    pub loglik_sd: f64,
    pub n_particles: usize,
    pub ess_at_mle: Option<(f64, f64)>,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#890: the `[provenance]` block names the method by the word every
    /// other surface uses. The file is machine-read — `camdl simulate
    /// --params` matches the backend off this block — so the key is a
    /// contract, and after the `[stages]` → `[method]` split `stage` named a
    /// concept the config, the store path and the CLI no longer have.
    #[test]
    fn the_provenance_block_names_the_method_not_a_stage() {
        let dir = crate::test_support::unique_temp_dir("mle_params_provenance");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.3)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_path: "sir.camdl".into(),
            model_identity: "f00d".into(),
            data_hashes: vec![("cases".into(), "abc".into())],
            seed: 1,
            method: "if2".into(),
            best_chain_index: 0,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
            timestamp: "2026-09-10T00:00:00Z".into(),
        };
        write_mle_params(&path.to_string_lossy(), &params, &meta).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("method = \"if2\""),
            "the block must name the method:\n{text}");
        assert!(!text.contains("stage = "),
            "and must not spell it `stage`, which no other surface does:\n{text}");
        // Read back through the reader that consumers use, so the rename is
        // pinned on both sides of the round trip.
        let prov = read_mle_provenance(&path.to_string_lossy()).unwrap().unwrap();
        assert_eq!(prov.method, "if2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The file a fit writes verifies; a value edited afterwards is reported
    /// against the declared hash; an edited provenance field is not a value
    /// edit.
    #[test]
    fn verify_content_hash_tells_an_edited_value_from_an_unedited_file() {
        let dir = crate::test_support::unique_temp_dir("mle_params_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let path_s = path.to_string_lossy().into_owned();
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_path: "sir.camdl".into(),
            model_identity: "f00d".into(),
            data_hashes: vec![],
            seed: 1,
            method: "if2".into(),
            best_chain_index: 0,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
            timestamp: "2026-10-24T00:00:00Z".into(),
        };
        write_mle_params(&path_s, &params, &meta).unwrap();
        let prov = read_mle_provenance(&path_s).unwrap().unwrap();
        assert_eq!(
            verify_content_hash(&path_s, &prov.content_hash).unwrap(),
            ContentVerification::Valid,
            "the file a fit writes verifies against its own block"
        );

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("beta = 0.3\n"), "{text}");
        std::fs::write(&path, text.replace("beta = 0.3\n", "beta = 0.35\n")).unwrap();
        match verify_content_hash(&path_s, &prov.content_hash).unwrap() {
            ContentVerification::Modified { declared, computed } => {
                assert_eq!(declared, prov.content_hash);
                assert_eq!(computed, mle_params_tamper_hash(
                    &[("beta".to_string(), 0.35), ("gamma".to_string(), 0.1)].into()));
                assert_ne!(declared, computed);
            }
            other => panic!("an edited value must be reported, got {other:?}"),
        }

        // Only the values are hashed: a provenance field may be corrected.
        std::fs::write(&path, text.replace("model = \"sir.camdl\"", "model = \"sir_v2.camdl\"")).unwrap();
        assert_eq!(verify_content_hash(&path_s, &prov.content_hash).unwrap(), ContentVerification::Valid);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn content_hash_stable() {
        let params: std::collections::BTreeMap<String, f64> = [
            ("beta".to_string(), 0.3),
            ("gamma".to_string(), 0.1),
        ].into();
        let h1 = mle_params_tamper_hash(&params);
        let h2 = mle_params_tamper_hash(&params);
        assert_eq!(h1, h2, "same params must produce same hash");
        assert_eq!(h1.len(), 8, "content hash is 8 hex chars");
    }

    #[test]
    fn content_hash_changes_on_value_change() {
        let params1: std::collections::BTreeMap<String, f64> = [("beta".to_string(), 0.3)].into();
        let params2: std::collections::BTreeMap<String, f64> = [("beta".to_string(), 0.31)].into();
        assert_ne!(
            mle_params_tamper_hash(&params1),
            mle_params_tamper_hash(&params2),
            "different values must produce different hashes"
        );
    }

    /// The declared `content_hash` is recomputable from the file: the file
    /// holds `format_param_value`'s ten decimals, so the hash must cover the
    /// values as written, not the unrounded estimate.
    #[test]
    fn content_hash_is_over_the_values_as_written() {
        let dir = crate::test_support::unique_temp_dir("mle_params_as_written");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let path_s = path.to_string_lossy().into_owned();
        // Thirteen decimals: more than the file carries.
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.4321098765432_f64), ("gamma".to_string(), 0.2)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_path: "sir.camdl".into(),
            model_identity: "f00d".into(),
            data_hashes: vec![],
            seed: 1,
            method: "if2".into(),
            best_chain_index: 0,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
            timestamp: "2026-10-24T00:00:00Z".into(),
        };
        write_mle_params(&path_s, &params, &meta).unwrap();
        let declared = read_mle_provenance(&path_s).unwrap().unwrap().content_hash;
        let read_back: std::collections::BTreeMap<String, f64> =
            crate::util::load_params_toml(&path_s).unwrap().into_iter().collect();
        assert_eq!(read_back["beta"], 0.4321098765, "the file holds ten decimals");
        assert_eq!(
            mle_params_tamper_hash(&read_back),
            declared,
            "the declared hash must be the hash of the values the file holds"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
