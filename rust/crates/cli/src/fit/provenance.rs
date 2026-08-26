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
    pub stage: String,
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
    let content_hash = mle_params_tamper_hash(all_params);
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
        stage: metadata.stage.clone(),
        // best_chain is zero-indexed internally; surface as 1-indexed
        // in the provenance block to match human-facing convention.
        chain: metadata.best_chain + 1,
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

// `file_content_hash` moved to `crate::hashing::file_hash`; callers
// now use the canonical name directly.

pub struct MleMetadata {
    pub input_hash: String,
    pub model_path: String,
    pub model_identity: String,
    pub data_hashes: Vec<(String, String)>,
    pub seed: u64,
    pub stage: String,
    pub best_chain: usize,
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

/// Verify content hash of an `mle_params.toml` file. Post-provenance
/// migration the declared hash lives in `provenance.content_hash`
/// (structured TOML field); for legacy files written with the old
/// comment header we still accept `# Content hash: X`.
///
/// Either way, the hash scope is the top-level numeric parameters
/// only — the `[provenance]` block itself is NOT hashed, so editing
/// a provenance field (fixing a typo in `model`) does not invalidate
/// the hash. Editing a parameter value does.
pub fn verify_content_hash(path: &str) -> Result<ContentVerification, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;

    // Try the modern path first: parse as TOML and look for
    // [provenance].content_hash. Provenance block absent or malformed
    // falls through to the legacy comment-header form.
    let declared: Option<String> = match toml::from_str::<toml::Value>(&contents) {
        Ok(v) => v.get("provenance")
            .and_then(|p| p.get("content_hash"))
            .and_then(|h| h.as_str())
            .map(str::to_string),
        Err(_) => None,
    }
    .or_else(|| {
        contents.lines()
            .find(|l| l.starts_with("# Content hash:"))
            .and_then(|l| l.split_whitespace().nth(3))
            .map(str::to_string)
    });

    // Rebuild the params-to-hash map from the top-level numeric TOML
    // keys. Parse via `toml::Value` so we can explicitly skip the
    // `[provenance]` table rather than relying on "not a comment" as
    // a filter — that old filter would have tried to parse every
    // line under `[provenance]` as a params line too.
    let params: std::collections::BTreeMap<String, f64> = match toml::from_str::<toml::Value>(&contents) {
        Ok(toml::Value::Table(map)) => map.into_iter()
            .filter_map(|(k, v)| {
                if k == "provenance" { return None; }
                match v {
                    toml::Value::Float(f) => Some((k, f)),
                    toml::Value::Integer(i) => Some((k, i as f64)),
                    _ => None,
                }
            })
            .collect(),
        _ => {
            // Legacy line-parse fallback.
            contents.lines()
                .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
                .filter_map(|l| {
                    let mut parts = l.splitn(2, '=');
                    let k = parts.next()?.trim().to_string();
                    let v: f64 = parts.next()?.trim().parse().ok()?;
                    Some((k, v))
                })
                .collect()
        }
    };

    let computed = mle_params_tamper_hash(&params);

    match declared {
        Some(ref d) if *d == computed => Ok(ContentVerification::Valid),
        Some(d) => Ok(ContentVerification::Modified { declared: d, computed }),
        None => Ok(ContentVerification::NoHash),
    }
}

pub enum ContentVerification {
    Valid,
    Modified {
        declared: String,
        computed: String,
    },
    NoHash,
}

// ─── Fit-stage hash ──────────────────────────────────────────────────────────

/// Compute the content hash for a single fit stage. Full 64-char hex
/// (256 bits) covering every input that affects the stage's output:
/// model IR, data file bytes, estimate specs, fixed values, stage
/// algorithm settings, seed, and camdl version. Returns `Err` if a
/// data file is unreadable (data must hash to something — we don't
/// silently substitute a placeholder the way display code does).
///
/// **Why this lives in `fit::provenance` rather than `crate::hashing`.**
/// The 2026-04-19 unification proposal's commit-1 goal was to
/// consolidate all hash helpers into `crate::hashing`. This one
/// resisted because it consumes `fit::config_v2::{EstimateSpecV2,
/// Stage}` via `serde_json::to_vec` — moving it to `hashing` would
/// invert the dep graph (`hashing → fit::config_v2`). Keeping it
/// here keeps `hashing` fit-agnostic. The cost is a one-directory
/// split on the hash vocabulary; the benefit is a cleaner boundary.
#[allow(clippy::too_many_arguments)]
pub fn fit_stage_hash(
    model_ir_json: &str,
    observations: &indexmap::IndexMap<String, String>,
    estimate: &indexmap::IndexMap<String, super::config_v2::EstimateSpecV2>,
    fixed_resolved: &indexmap::IndexMap<String, f64>,
    simplex_groups: &[super::config_v2::SimplexGroup],
    stage_name: &str,
    stage: &super::config_v2::Stage,
    seed: u64,
) -> Result<String, String> {
    let mut h = Sha256::new();

    // Model
    h.update(b"model\x00");
    h.update(model_ir_json.as_bytes());

    // Data files (sorted by stream name for stability)
    h.update(b"\x00data\x00");
    let mut data_entries: Vec<_> = observations.iter().collect();
    data_entries.sort_by_key(|(k, _)| k.as_str());
    for (name, path) in &data_entries {
        h.update(name.as_bytes());
        h.update(b"\x00");
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read data file '{}' ({}): {}", name, path, e))?;
        h.update(&bytes);
        h.update(b"\x00");
    }

    // Estimate specs (sorted by name)
    h.update(b"\x00estimate\x00");
    let mut est_entries: Vec<_> = estimate.iter().collect();
    est_entries.sort_by_key(|(k, _)| k.as_str());
    for (name, spec) in &est_entries {
        h.update(name.as_bytes());
        h.update(b"\x00");
        h.update(serde_json::to_vec(spec).unwrap_or_default());
        h.update(b"\x00");
    }

    // Fixed values (sorted by name)
    h.update(b"\x00fixed\x00");
    let mut fix_entries: Vec<_> = fixed_resolved.iter().collect();
    fix_entries.sort_by_key(|(k, _)| k.as_str());
    for (name, val) in &fix_entries {
        h.update(name.as_bytes());
        h.update(b"=");
        h.update(val.to_le_bytes());
        h.update(b"\x00");
    }

    // Simplex groups (preserve order — group identity depends on member
    // order via the log-ratio reference index, so a reorder produces
    // different IF2 dynamics).
    h.update(b"\x00simplex\x00");
    for grp in simplex_groups {
        h.update(serde_json::to_vec(grp).unwrap_or_default());
        h.update(b"\x00");
    }

    // Stage config — uses Stage::identity_payload(), which omits the
    // extension dimension (PGAS sweeps, PMMH iterations) so resume
    // can extend a chain without invalidating its stored state.
    h.update(b"\x00stage\x00");
    h.update(stage_name.as_bytes());
    h.update(b"\x00");
    h.update(serde_json::to_vec(&stage.identity_payload()).unwrap_or_default());

    // Seed
    h.update(b"\x00seed\x00");
    h.update(seed.to_le_bytes());

    // Version
    h.update(b"\x00version\x00");
    h.update(version::VERSION_SHORT.as_bytes());

    Ok(hex::encode(h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn config_hash_v2_stable() {
        use indexmap::IndexMap;
        let obs: IndexMap<String, String> = IndexMap::new();
        let est: IndexMap<String, super::super::config_v2::EstimateSpecV2> = IndexMap::new();
        let fixed: IndexMap<String, f64> = IndexMap::new();
        let stage = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let h1 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage, 1).unwrap();
        let h2 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage, 1).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64, "config hash is 64 hex chars");
    }

    #[test]
    fn config_hash_v2_changes_on_model() {
        use indexmap::IndexMap;
        let obs: IndexMap<String, String> = IndexMap::new();
        let est: IndexMap<String, super::super::config_v2::EstimateSpecV2> = IndexMap::new();
        let fixed: IndexMap<String, f64> = IndexMap::new();
        let stage = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let h1 = fit_stage_hash("model_a", &obs, &est, &fixed, &[], "mle", &stage, 1).unwrap();
        let h2 = fit_stage_hash("model_b", &obs, &est, &fixed, &[], "mle", &stage, 1).unwrap();
        assert_ne!(h1, h2, "different model must produce different hash");
    }

    #[test]
    fn config_hash_v2_changes_on_seed() {
        use indexmap::IndexMap;
        let obs: IndexMap<String, String> = IndexMap::new();
        let est: IndexMap<String, super::super::config_v2::EstimateSpecV2> = IndexMap::new();
        let fixed: IndexMap<String, f64> = IndexMap::new();
        let stage = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let h1 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage, 1).unwrap();
        let h2 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage, 2).unwrap();
        assert_ne!(h1, h2, "different seed must produce different hash");
    }

    #[test]
    fn config_hash_v2_changes_on_stage_settings() {
        use indexmap::IndexMap;
        let obs: IndexMap<String, String> = IndexMap::new();
        let est: IndexMap<String, super::super::config_v2::EstimateSpecV2> = IndexMap::new();
        let fixed: IndexMap<String, f64> = IndexMap::new();
        let stage1 = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let stage2 = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 8, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let h1 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage1, 1).unwrap();
        let h2 = fit_stage_hash("model", &obs, &est, &fixed, &[], "mle", &stage2, 1).unwrap();
        assert_ne!(h1, h2, "different stage settings must produce different hash");
    }

    #[test]
    fn config_hash_v2_changes_on_simplex_groups() {
        // Adding a simplex constraint changes IF2 dynamics (joint
        // barycentric perturbation vs independent), so it MUST
        // invalidate the stage hash. Otherwise a fit run with
        // simplex_groups=[] and one with simplex_groups=[A,B,C] would
        // collide on cache key.
        use indexmap::IndexMap;
        let obs: IndexMap<String, String> = IndexMap::new();
        let est: IndexMap<String, super::super::config_v2::EstimateSpecV2> = IndexMap::new();
        let fixed: IndexMap<String, f64> = IndexMap::new();
        let stage = super::super::config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 1000, iterations: 50,
            cooling: 0.7,
            cooling_target_iters: 50,
            starts_from: super::super::config_v2::StartsFrom::Random,
            loglik_eval: Default::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: Default::default(),
            dt_check: Default::default(),
        };
        let no_groups: Vec<super::super::config_v2::SimplexGroup> = vec![];
        let with_group = vec![super::super::config_v2::SimplexGroup {
            params: vec!["S0_y".into(), "S0_a".into(), "S0_e".into()],
        }];
        let h1 = fit_stage_hash("model", &obs, &est, &fixed,
            &no_groups, "mle", &stage, 1).unwrap();
        let h2 = fit_stage_hash("model", &obs, &est, &fixed,
            &with_group, "mle", &stage, 1).unwrap();
        assert_ne!(h1, h2,
            "adding a simplex_group must invalidate the stage hash");

        // Reordering members within a group must also invalidate —
        // log-ratio encoding depends on member order via the reference
        // index.
        let with_group_reordered = vec![super::super::config_v2::SimplexGroup {
            params: vec!["S0_e".into(), "S0_a".into(), "S0_y".into()],
        }];
        let h3 = fit_stage_hash("model", &obs, &est, &fixed,
            &with_group_reordered, "mle", &stage, 1).unwrap();
        assert_ne!(h2, h3,
            "reordering simplex members must invalidate the stage hash");
    }
}
