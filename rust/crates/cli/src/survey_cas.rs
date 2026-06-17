//! gh#147 (M3.3). The survey CAS identity: map a resolved likelihood-landscape
//! survey into the `runid` factored levels (`model` / `config` / `box` /
//! `seed`) and its leaf `run_id`.
//!
//! A `camdl survey` is a likelihood-landscape diagnostic: N Latin-hypercube
//! points across the declared parameter bounds, each scored. It is a single
//! leaf (one `landscape.tsv`), no grid — the N points are *within* the leaf,
//! not an axis.
//!
//! Factoring (mirrors [`crate::pfilter_cas`] / [`crate::resolve`]):
//!   - **model** — the pure model IR digest.
//!   - **config** — the eval setup + problem context: `eval_method`,
//!     `eval_particles`, `eval_replicates`, the observed data digests, the
//!     fixed params, and the scenario. The eval count knobs are
//!     identity-bearing — they change the stored landscape loglik values, so
//!     they live in the key (the n_trajectories / n_replicates collision
//!     class; the old `SurveyInputs::content_hash` folded them in too).
//!   - **box** — the LHS sampling box: the estimated-param bounds + n_points
//!     (what region, how densely).
//!   - **seed** — the resolved LHS / PF base seed.

use runid::inputs::{DataDigest, EngineVersion, ModelDigest, Seed};
use runid::{run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId};

use crate::fit::cas::{digest_value, ensure_finite};

/// A fully-resolved survey leaf: the four factored levels (in path order) and
/// the leaf `run_id` composed from their hashes.
pub struct ResolvedSurvey {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_survey`], all resolved by the caller.
pub struct SurveyCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// Provenance label for the `model` path segment (the model stem).
    pub stem: &'a str,
    /// `(stream name, sha256-hex)` — already-computed data digests.
    pub data: &'a [(String, String)],
    /// Resolved eval method name (`"pfilter"` / `"simulate"`).
    pub eval_method: &'a str,
    pub eval_particles: u32,
    pub eval_replicates: u32,
    /// The LHS box: `(param, lo, hi)`, the estimated-param bounds.
    pub bounds: &'a [(String, f64, f64)],
    /// Fixed `(param, value)` (excluded from the LHS box).
    pub fixed: &'a [(String, f64)],
    /// Scenario name, if any.
    pub scenario: Option<&'a str>,
    pub n_points: u32,
    /// The resolved LHS / PF base seed (the `seed` level hashes this).
    pub seed: u64,
}

fn level(name: &str, label: &str, hash: ContentHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash, schema_version: 1 }
}

/// Per-stream data digests from the SHA-256 hex hashes, sorted by name.
fn data_digests(data: &[(String, String)]) -> Result<Vec<DataDigest>, String> {
    let mut entries: Vec<&(String, String)> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
        .iter()
        .map(|(name, sha)| {
            ContentHash::from_hex(sha)
                .map(DataDigest)
                .map_err(|e| format!("data hash for '{}' is not a 64-hex SHA-256: {:?}", name, e))
        })
        .collect()
}

/// Resolve a survey leaf's identity: the four factored levels and the `run_id`.
pub fn resolve_survey(ctx: &SurveyCtx) -> Result<ResolvedSurvey, String> {
    // config level — eval setup + problem context. The eval count knobs are
    // identity-bearing (they change the stored landscape).
    let _ = data_digests(ctx.data)?;
    let mut data_sorted: Vec<(&str, &str)> =
        ctx.data.iter().map(|(n, h)| (n.as_str(), h.as_str())).collect();
    data_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut fixed_sorted: Vec<(&str, f64)> =
        ctx.fixed.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    fixed_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let config_blob = serde_json::json!({
        "eval_method": ctx.eval_method,
        "eval_particles": ctx.eval_particles,
        "eval_replicates": ctx.eval_replicates,
        "data": data_sorted,
        "fixed": fixed_sorted,
        "scenario": ctx.scenario,
    });
    ensure_finite(&config_blob)?;

    // box level — the LHS sampling spec.
    let mut bounds_sorted: Vec<(&str, f64, f64)> =
        ctx.bounds.iter().map(|(n, lo, hi)| (n.as_str(), *lo, *hi)).collect();
    bounds_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let box_blob = serde_json::json!({
        "bounds": bounds_sorted,
        "n_points": ctx.n_points,
    });
    ensure_finite(&box_blob)?;

    let model_digest = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        EngineVersion(ctx.engine_version.to_string()),
    );
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };

    let config_label = format!(
        "{}-P{}-r{}", ctx.eval_method, ctx.eval_particles, ctx.eval_replicates);
    let box_label = format!("box-n{}", ctx.n_points);

    let levels = vec![
        level("model", ctx.stem, model_digest.content_hash()),
        level("config", &config_label, digest_value(&config_blob)),
        level("box", &box_label, digest_value(&box_blob)),
        level("seed", &format!("seed_{}", ctx.seed), seed.content_hash()),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::Survey, &level_hashes);
    Ok(ResolvedSurvey { levels, run_id: rid })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> String {
        ContentHash::digest_bytes(&[byte]).to_hex()
    }

    // The `config` and `box` levels are `digest_value` over the same JSON
    // blobs `resolve_survey` builds; unit-testing those digests pins the
    // collision-freeness of the identity components without an `ir::Model`
    // fixture (mirrors `pfilter_cas::tests`).

    fn config_level(method: &str, particles: u32, replicates: u32,
                    data: &[(&str, &str)], scenario: Option<&str>) -> ContentHash {
        let mut d: Vec<(&str, &str)> = data.to_vec();
        d.sort_by(|a, b| a.0.cmp(b.0));
        digest_value(&serde_json::json!({
            "eval_method": method, "eval_particles": particles,
            "eval_replicates": replicates, "data": d,
            "fixed": Vec::<(&str, f64)>::new(), "scenario": scenario,
        }))
    }

    fn box_level(bounds: &[(&str, f64, f64)], n_points: u32) -> ContentHash {
        let mut b: Vec<(&str, f64, f64)> = bounds.to_vec();
        b.sort_by(|a, x| a.0.cmp(x.0));
        digest_value(&serde_json::json!({ "bounds": b, "n_points": n_points }))
    }

    /// The eval count knobs are identity-bearing — `eval_method`,
    /// `eval_particles`, `eval_replicates` each change the stored landscape,
    /// so each must change the `config` hash (the count-in-the-key class).
    #[test]
    fn eval_config_participates_via_config() {
        let h1 = h(1);
        let base = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        assert_ne!(base, config_level("simulate", 200, 1, &[("cases", &h1)], None),
            "eval_method must change the config hash");
        assert_ne!(base, config_level("pfilter", 5000, 1, &[("cases", &h1)], None),
            "eval_particles must change the config hash");
        assert_ne!(base, config_level("pfilter", 200, 3, &[("cases", &h1)], None),
            "eval_replicates must change the config hash");
        assert_ne!(base, config_level("pfilter", 200, 1, &[("cases", &h(2))], None),
            "data must change the config hash");
        assert_eq!(base, config_level("pfilter", 200, 1, &[("cases", &h1)], None), "stable");
    }

    /// The LHS box distinguishes leaves: different bounds or n_points → a
    /// distinct `box` hash; identical is stable + order-independent.
    #[test]
    fn box_distinguishes_bounds_and_n_points() {
        let a = box_level(&[("beta", 0.1, 0.9)], 100);
        assert_ne!(a, box_level(&[("beta", 0.1, 0.5)], 100), "different bounds → distinct box");
        assert_ne!(a, box_level(&[("beta", 0.1, 0.9)], 200), "different n_points → distinct box");
        assert_eq!(a, box_level(&[("beta", 0.1, 0.9)], 100), "stable");
        assert_eq!(
            box_level(&[("beta", 0.1, 0.9), ("gamma", 0.0, 1.0)], 100),
            box_level(&[("gamma", 0.0, 1.0), ("beta", 0.1, 0.9)], 100),
            "bound order must not change the box hash"
        );
    }

    /// The `seed` level hashes the resolved seed.
    #[test]
    fn seed_level_distinguishes() {
        let s7 = Seed { process_seed: 7, base_seed: 7 }.content_hash();
        let s8 = Seed { process_seed: 8, base_seed: 8 }.content_hash();
        assert_ne!(s7, s8, "distinct resolved seeds must hash distinctly");
    }
}
