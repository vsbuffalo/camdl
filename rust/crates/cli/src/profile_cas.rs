//! gh#147 (M3.3). The profile-point CAS identity: map a resolved profile grid
//! point into the `runid` factored levels (`profile` / `point` / `stage` /
//! `seed` / `start`) and its leaf `run_id`. Mirrors
//! [`crate::fit::cas::resolve_fit_stage`].
//!
//! Factoring (design locked in `7493cc0` — not re-litigated here):
//!   - **profile (base)** — the inference *problem*: model + data + the
//!     canonical base config (base params + fixed + obs + priors + fit.toml),
//!     with the focal GRID and the method config EXCLUDED (guardrail 1), plus
//!     the base fit's `starts_from` as a dep (guardrail 3-base). A path segment
//!     with no base-level record (guardrail 2 — enforced by the writer).
//!   - **point** — the single pinned focal value(s) for this grid point.
//!   - **stage** — the sub-fit method + hyperparams (shared across the grid).
//!   - **seed** — the resolved profile seed (hashed; guardrail 3); **start** —
//!     the multi-start index. The `(seed, point, start)` triple pins each job's
//!     RNG deterministically (`job_seed = seed ^ (grid_idx*1000 + start_idx)`).

use runid::float::FiniteF64;
use runid::inputs::{
    ArtifactRef, Deps, EngineVersion, ModelDigest, ParamId, ProfileBase,
    ProfilePointConfig, ProfileStage, Seed, StartLevel,
};
use runid::{run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId};

use crate::fit::cas::{digest_value, ensure_finite};

/// A fully-resolved profile-point leaf: the five factored levels (in path
/// order) and the leaf `run_id` composed from their hashes.
pub struct ResolvedProfilePoint {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_profile_point`], all resolved by the caller. The caller
/// is responsible for building `base_config` with the focal grid AND the method
/// config excluded (guardrail 1 — invisible in the opaque `ContentHash`).
pub struct ProfilePointCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// Provenance label for the `profile` path segment (the model stem).
    pub stem: &'a str,
    /// Display label for the `stage` segment (the method name, e.g. `if2`).
    pub method_name: &'a str,
    /// `(stream name, sha256-hex)` — profile's already-computed data hashes.
    /// SHA-256, the same function as [`ContentHash::digest_bytes`], so
    /// `from_hex` reproduces the file-byte digest without a re-read.
    pub data: &'a [(String, String)],
    /// Canonical base config blob: base params + fixed + obs + priors +
    /// fit.toml. The focal GRID and method config are EXCLUDED here.
    pub base_config: &'a serde_json::Value,
    /// The sub-fit method + hyperparams (algorithm + if2/pmmh).
    pub method_config: &'a serde_json::Value,
    /// The pinned focal `(param, value)` for this grid point.
    pub focal: &'a [(String, f64)],
    /// The resolved profile seed (the `seed` level hashes this).
    pub seed: u64,
    pub start_index: u32,
    /// The base fit's `starts_from` lineage, folded into the base as deps.
    pub deps: Vec<ArtifactRef>,
}

use crate::fit::cas::{data_digests, level};

/// Resolve a profile-point leaf's identity: the five factored levels and the
/// `run_id` derived from their hashes.
pub fn resolve_profile_point(ctx: &ProfilePointCtx) -> Result<ResolvedProfilePoint, String> {
    // Reject non-finite floats before hashing — the JSON serializer would
    // silently null them (a collision), same gate as the fit-stage path.
    ensure_finite(ctx.base_config)?;
    ensure_finite(ctx.method_config)?;

    let base = ProfileBase {
        model: ModelDigest::from_model(
            ctx.model,
            ctx.ir_version.to_string(),
            EngineVersion(ctx.engine_version.to_string()),
        ),
        data: data_digests(ctx.data)?,
        base_config: digest_value(ctx.base_config),
        engine: EngineVersion(ctx.engine_version.to_string()),
        deps: Deps(ctx.deps.clone()),
    };

    let mut focal = Vec::with_capacity(ctx.focal.len());
    for (name, v) in ctx.focal {
        let fv = FiniteF64::new(*v).map_err(|_| format!("non-finite focal value for '{}'", name))?;
        focal.push((ParamId(name.clone()), fv));
    }
    let point = ProfilePointConfig { focal };
    let stage = ProfileStage { config: digest_value(ctx.method_config) };
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };
    let start = StartLevel { index: ctx.start_index };

    let point_label = ctx
        .focal
        .iter()
        .map(|(n, v)| format!("{}={:.4}", n, v))
        .collect::<Vec<_>>()
        .join("__");

    let levels = vec![
        level("profile", ctx.stem, base.content_hash()),
        level("point", &point_label, point.content_hash()),
        level("stage", ctx.method_name, stage.content_hash()),
        level("seed", &format!("seed_{}", ctx.seed), seed.content_hash()),
        level("start", &format!("start_{}", ctx.start_index), start.content_hash()),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::ProfilePoint, &level_hashes);
    Ok(ResolvedProfilePoint { levels, run_id: rid })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runid::ContentAddressed;

    fn cfg(point: &str, val: f64) -> ProfilePointConfig {
        ProfilePointConfig { focal: vec![(ParamId(point.into()), FiniteF64::new(val).unwrap())] }
    }

    /// The grid value distinguishes points: two pinned values produce distinct
    /// `point`-level hashes (so distinct leaves), and an identical value is
    /// stable. This is the collision-freeness the factoring depends on.
    #[test]
    fn pinned_value_distinguishes_points_and_is_stable() {
        let a = cfg("beta", 0.30).content_hash();
        let b = cfg("beta", 0.50).content_hash();
        assert_ne!(a, b, "two pinned focal values must produce distinct point hashes");
        assert_eq!(a, cfg("beta", 0.30).content_hash(), "same pinned value must be stable");
    }

    /// The seed level hashes the resolved seed, and start is separate — so the
    /// (seed, start) pair pins distinct cells with no byte-identical collision.
    #[test]
    fn seed_and_start_are_distinct_levels() {
        let s1 = Seed { process_seed: 7, base_seed: 7 }.content_hash();
        let s2 = Seed { process_seed: 8, base_seed: 8 }.content_hash();
        assert_ne!(s1, s2, "distinct resolved seeds must hash distinctly");
        let t0 = StartLevel { index: 0 }.content_hash();
        let t1 = StartLevel { index: 1 }.content_hash();
        assert_ne!(t0, t1, "distinct start indices must hash distinctly");
    }
}
