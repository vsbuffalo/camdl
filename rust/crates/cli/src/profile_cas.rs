//! gh#147 (M3.3). The profile-point CAS identity: map a resolved profile grid
//! point into the `runid` factored levels (`profile` / `point` / `method` /
//! `seed` / `start`) and its leaf `run_id`.
//!
//! Factoring (design locked in `7493cc0` — not re-litigated here):
//!   - **profile (base)** — the inference *problem*: model + data + the
//!     canonical base config (base params + fixed + obs + priors + fit.toml),
//!     with the focal GRID and the method config EXCLUDED (guardrail 1), plus
//!     the base fit's `starts_from` as a dep (guardrail 3-base). A path segment
//!     with no base-level record (guardrail 2 — enforced by the writer).
//!   - **point** — the single pinned focal value(s) for this grid point.
//!   - **method** — the sub-fit method + hyperparams (shared across the grid).
//!   - **seed** — the job's process seed (hashed; guardrail 3); **start** —
//!     the multi-start index. The process seed is derived here from the profile
//!     seed, the point's position in the canonical grid (axes sorted by name,
//!     values ascending) and the start index, by
//!     [`crate::util::mix_cell_seed`]. Every input to it is part of the
//!     identity and none depends on the order of the `--sweep` flags or of a
//!     listed grid's values, so one leaf always holds one computation.

use runid::float::FiniteF64;
use runid::inputs::{
    ArtifactRef, Deps, EngineIdentity, ModelDigest, ParamId, ProfileBase,
    ProfilePointConfig, ProfileMethod, Seed, StartLevel,
};
use runid::{run_id, ArtifactKind, ContentHash, LevelId};


/// A fully-resolved profile-point leaf: the five factored levels (in path
/// order) and the leaf `run_id` composed from their hashes.
pub struct ResolvedProfilePoint {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
    /// The seed the job's sub-fit runs with, the one the `seed` level hashes.
    pub process_seed: u64,
}

/// Inputs to [`resolve_profile_point`], all resolved by the caller. The caller
/// is responsible for building `base_config` with the focal grid AND the method
/// config excluded (guardrail 1 — invisible in the opaque `ContentHash`).
pub struct ProfilePointCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    /// The engine the base folds: the build and the engine switches set
    /// (`env::Environment::engine_identity`). A point computed with a switch
    /// never shares a leaf with a default point, whose key is the build's alone.
    pub engine: &'a EngineIdentity,
    /// Provenance label for the `profile` path segment (the model stem).
    pub stem: &'a str,
    /// Display label for the `method` segment (the method name, e.g. `if2`).
    pub method_name: &'a str,
    /// `(stream name, sha256-hex)` — profile's already-computed data hashes.
    /// SHA-256, the same function as [`ContentHash::digest_bytes`], so
    /// `from_hex` reproduces the file-byte digest without a re-read.
    pub data: &'a [(String, String)],
    /// Canonical base config blob: base params + fixed + obs + priors +
    /// fit.toml. The focal GRID and method config are EXCLUDED here.
    pub base_config: LevelHash,
    /// The sub-fit method + hyperparams (algorithm + if2/pmmh).
    pub method_config: LevelHash,
    /// The pinned focal `(param, value)` for this grid point.
    pub focal: &'a [(String, f64)],
    /// The full resolved sweep grid — each focal axis name and its values,
    /// shared across every point of this run. Folded into the base identity so
    /// a different grid (wider range, more points, shifted bounds) is a
    /// distinct run rather than a merge onto the previous grid's cells.
    pub grid: &'a [(String, Vec<f64>)],
    /// The profile seed (`--seed`, or one of `--seeds`), from which the job's
    /// process seed is derived.
    pub seed: u64,
    pub start_index: u32,
    /// The base fit's `starts_from` lineage, folded into the base as deps.
    pub deps: Vec<ArtifactRef>,
}

/// Schema version stamped into every [`LevelId`] a profile point's identity
/// holds. Identity-bearing: it folds into every profile `run_id`.
pub(crate) const LEVEL_SCHEMA_VERSION: u16 = 1;

/// A [`LevelId`] with the canonical schema version.
pub(crate) fn level(name: &str, label: &str, hash: LevelHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash: hash.0, schema_version: LEVEL_SCHEMA_VERSION }
}

/// A hash that may become an identity level. Only two functions make one:
/// [`canonical_config_hash`] (a `Serialize` value, gated for finiteness and
/// serialized whole) and [`structural_level_hash`] (a `ContentAddressed`
/// value). A digest of a hand-built JSON literal cannot reach [`level`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LevelHash(ContentHash);

impl LevelHash {
    /// The hash, for embedding inside another structurally hashed input.
    pub(crate) fn into_inner(self) -> ContentHash {
        self.0
    }
}

/// A [`LevelHash`] of a structurally hashed value.
pub(crate) fn structural_level_hash<T: runid::ContentAddressed>(v: &T) -> LevelHash {
    LevelHash(v.content_hash())
}

/// A [`LevelHash`] of a `Serialize` value: finiteness gated on the value,
/// serialized whole, the named top-level keys removed, canonicalized and
/// digested.
pub(crate) fn canonical_config_hash<T: serde::Serialize>(value: &T, exclude: &[&str]) -> Result<LevelHash, String> {
    crate::canonical_json::ensure_finite(value)?;
    let v = crate::canonical_json::serialize_minus(value, exclude)?;
    let bytes = serde_json::to_vec(&crate::canonical_json::canonical(&v))
        .expect("canonical JSON of a finiteness-gated value serializes");
    Ok(LevelHash(ContentHash::digest_bytes(&bytes)))
}

/// Per-stream data digests from SHA-256 hex hashes the caller holds, sorted
/// by name. A hash that is not 64 hex digits is refused.
pub(crate) fn data_digests(data: &[(String, String)]) -> Result<Vec<runid::inputs::DataDigest>, String> {
    let mut entries: Vec<&(String, String)> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
        .iter()
        .map(|(name, sha)| {
            ContentHash::from_hex(sha)
                .map(runid::inputs::DataDigest)
                .map_err(|e| format!("data hash for '{name}' is not a 64-hex SHA-256: {e:?}"))
        })
        .collect()
}

/// The dependency a profile's base holds on the source of its chain starts:
/// the content the starts read, keyed by that content alone. A `from_mle`
/// source is keyed by its `start_values`, any other by the file's bytes.
pub(crate) fn start_source_dep(
    rule: &crate::fit::starts::ChainStarts,
    source: &crate::fit::chain_starts::ResolvedSource,
) -> Result<ArtifactRef, String> {
    use crate::fit::starts::{ChainStarts, Point};
    let bytes = std::fs::read(&source.file).map_err(|e| format!("cannot read {}: {e}", source.file.display()))?;
    let (artifact, digest) = match rule {
        ChainStarts::Point(Point::FromMle { .. }) => (
            "fit_state.toml:start_values",
            ContentHash::digest_bytes(&crate::fit::chain_starts::start_values_part(&bytes)?),
        ),
        _ => ("starts source", ContentHash::digest_bytes(&bytes)),
    };
    Ok(ArtifactRef { run_id: digest, kind: ArtifactKind::FitMethod, artifact: artifact.to_string(), digest })
}

/// Resolve a profile-point leaf's identity: the five factored levels and the
/// `run_id` derived from their hashes.
pub fn resolve_profile_point(ctx: &ProfilePointCtx) -> Result<ResolvedProfilePoint, String> {
    // `base_config` / `method_config` arrive already hashed, by
    // `canonical_config_hash` in the caller — which gates finiteness on the
    // RAW struct before serializing. That ordering is the point: `json!` /
    // `to_value` collapse NaN and Inf to `Null`, so a gate applied to a built
    // `Value` can never fire, and a float field added to either level would be
    // hashed but ungated (NaN colliding with Inf). Taking a `ContentHash`
    // here makes that impossible to get wrong from this side.

    // Canonicalize the sweep grid: axes sorted by name, values ascending, so
    // the identity depends only on the *set* of cells, not `--sweep` order.
    let mut axes: Vec<(String, Vec<f64>)> = ctx.grid.to_vec();
    axes.sort_by(|a, b| a.0.cmp(&b.0));
    let mut grid: Vec<(ParamId, Vec<FiniteF64>)> = Vec::with_capacity(axes.len());
    for (name, mut values) in axes {
        values.sort_by(|a, b| a.partial_cmp(b)
            .expect("grid values are finite (checked at parse)"));
        let mut fvs = Vec::with_capacity(values.len());
        for v in values {
            fvs.push(FiniteF64::new(v)
                .map_err(|_| format!("non-finite grid value for axis '{}'", name))?);
        }
        grid.push((ParamId(name), fvs));
    }

    let base = ProfileBase {
        model: ModelDigest::from_model(
            ctx.model,
            ctx.ir_version.to_string(),
            ctx.engine.clone(),
        ),
        data: data_digests(ctx.data)?,
        base_config: ctx.base_config.into_inner(),
        grid,
        engine: ctx.engine.clone(),
        deps: Deps(ctx.deps.clone()),
    };

    let mut sorted_focal: Vec<&(String, f64)> = ctx.focal.iter().collect();
    sorted_focal.sort_by(|a, b| a.0.cmp(&b.0));
    let mut focal = Vec::with_capacity(sorted_focal.len());
    for (name, v) in &sorted_focal {
        let fv = FiniteF64::new(*v).map_err(|_| format!("non-finite focal value for '{}'", name))?;
        focal.push((ParamId(name.clone()), fv));
    }

    // The point's row-major position in the canonical grid, the first axis
    // varying slowest.
    let mut grid_position: u64 = 0;
    for (axis, values) in &base.grid {
        let v = focal.iter().find(|(n, _)| n == axis).map(|(_, v)| v.get())
            .ok_or_else(|| format!("profile point has no value for grid axis '{}'", axis.0))?;
        let pos = values.iter().position(|g| g.get() == v)
            .ok_or_else(|| format!("profile point value {} is not on the grid of '{}'", v, axis.0))?;
        grid_position = grid_position * values.len() as u64 + pos as u64;
    }
    let process_seed = crate::util::mix_cell_seed(ctx.seed, grid_position, ctx.start_index as u64);

    let point = ProfilePointConfig { focal };
    let method = ProfileMethod { config: ctx.method_config.into_inner() };
    let seed = Seed { process_seed, base_seed: ctx.seed };
    let start = StartLevel { index: ctx.start_index };

    let point_label = sorted_focal
        .iter()
        .map(|(n, v)| format!("{}={:.4}", n, v))
        .collect::<Vec<_>>()
        .join("__");

    let levels = vec![
        level("profile", ctx.stem, structural_level_hash(&base)),
        level("point", &point_label, structural_level_hash(&point)),
        // gh#901: the level is named `method`, matching the fit store's middle
        // level and the word the rest of the tool uses. `run_id` folds level
        // HASHES only and the path segment is built from the label, so the
        // rename re-keys nothing and moves no directory.
        level("method", ctx.method_name, structural_level_hash(&method)),
        level("seed", &format!("seed_{}", ctx.seed), structural_level_hash(&seed)),
        level("start", &format!("start_{}", ctx.start_index), structural_level_hash(&start)),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::ProfilePoint, &level_hashes);
    Ok(ResolvedProfilePoint { levels, run_id: rid, process_seed })
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
