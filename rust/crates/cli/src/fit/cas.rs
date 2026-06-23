//! The fit-stage CAS identity: map a resolved fit stage into the `runid`
//! factored levels (`fit` / `NN-stage` / `seed`) and its leaf `run_id`.
//!
//! Factoring (`FitDigest` excludes `[stages.*]` so editing the posterior's
//! block doesn't re-key the scout; cross-stage invalidation rides the
//! deps-DAG):
//!
//! - **fit level** = [`FitDigest`] = whole-IR model digest + per-stream
//!   training data digests + per-stream `[data.holdout]` content digests
//!   (gh#190) + the canonicalized fit-wide config (with `stages` / `fit_seeds`
//!   / `output_dir` normalized out) + engine version.
//! - **stage level** = [`StageLevel`] = [`StageConfig`] (the stage's
//!   `identity_payload` + `n_trajectories` + obs/flow + `target_length` + the
//!   resolved `obs_alignment`, gh#189) folded with its `deps` (so
//!   `02-posterior`'s hash folds in `01-scout`'s identity).
//! - **seed level** = [`Seed`] = the resolved fit RNG seed.
//!
//! `n_trajectories` (the count of posterior trajectories PGAS writes to
//! `chain_N/trajectories.tsv` under the leaf) is folded into the stage identity
//! ([`Stage::cas_n_trajectories`]). It is an output-shaping knob that
//! `identity_payload` otherwise omits, but it must be in the key: a count that
//! changes stored output has to change the `run_id`, else a stage `run_id`
//! could serve a different trajectory count's output. So each value yields a
//! distinct leaf (count-in-the-key), at the cost of re-fitting when it changes.

use indexmap::IndexMap;

use std::path::Path;

use runid::inputs::{
    ArtifactRef, DataDigest, Deps, EngineVersion, FitDigest, ResolvedObsAlignment, Seed,
    StageConfig, StageLevel,
};
use runid::{run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId, RunRecord};

use super::config_v2::{FitConfigV2, Stage};

/// A fully-resolved fit-stage leaf: the factored identity levels (in path
/// order) and the leaf `run_id` composed from their hashes.
pub struct ResolvedFitStage {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_fit_stage`], all already resolved by the caller
/// (sweep overrides applied to `config`, data paths resolved). Labels are
/// provenance; identity rides in the hashes.
pub struct FitStageCtx<'a> {
    pub model: &'a ir::Model,
    pub fit_stem: &'a str,
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// The resolved (sweep-applied) fit config. Hashed whole, less the
    /// lower-level/provenance slices (see [`fit_config_blob_hash`]).
    pub config: &'a FitConfigV2,
    /// Resolved training observation streams (name → path); their *content*
    /// is digested into `FitDigest.data`.
    pub data_paths: &'a IndexMap<String, String>,
    pub stage_name: &'a str,
    pub stage: &'a Stage,
    /// Zero-padded topological position → the provenance label `NN-stage`.
    pub ordinal: usize,
    /// The resolved fit RNG seed.
    pub seed: u64,
    /// Upstream artifacts consumed (`StartsFrom` → `fit_state.toml`), folded
    /// into the stage level so a regenerated upstream re-keys this stage.
    pub deps: Vec<ArtifactRef>,
}

/// Schema version stamped into every [`LevelId`]. Identity-bearing — it folds
/// into every `run_id`, so a bump re-keys all CAS artifacts and must live in
/// exactly one place.
pub(crate) const LEVEL_SCHEMA_VERSION: u16 = 1;

/// Build a CAS [`LevelId`] with the canonical schema version. The single
/// constructor every artifact-kind resolver (`fit`/`pfilter`/`survey`/
/// `profile`/`sim_ensemble`/`resolve`) routes through.
pub(crate) fn level(name: &str, label: &str, hash: ContentHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash, schema_version: LEVEL_SCHEMA_VERSION }
}

/// Per-stream data digests from **pre-computed** SHA-256 hex hashes, sorted by
/// name. Validates each is a well-formed 64-hex digest (a loud guard against a
/// malformed hand-supplied hash). Distinct from [`build_data_digests`], which
/// reads file *bytes*; this takes hashes the caller already holds.
pub(crate) fn data_digests(data: &[(String, String)]) -> Result<Vec<DataDigest>, String> {
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

/// Recursively sort object keys → canonical JSON, so the hash is stable
/// regardless of map iteration order (and robust to a future
/// `serde_json/preserve_order` feature unification — a silent-collision
/// hazard otherwise).
fn canonical(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for k in keys {
                sorted.insert(k.clone(), canonical(&m[k]));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

pub(crate) fn digest_value(v: &serde_json::Value) -> ContentHash {
    // The Value is built from a finiteness-gated source (see `ensure_finite`),
    // so it contains no non-finite floats and `to_vec` of a valid Value is
    // infallible. `.expect` (not `unwrap_or_default`) so a future caller that
    // feeds a Value from some other source can never silently hash an empty
    // byte string (a collision) instead of failing loudly.
    let bytes = serde_json::to_vec(&canonical(v))
        .expect("canonical JSON of a finiteness-gated Value serializes");
    ContentHash::digest_bytes(&bytes)
}

// ── Non-finite finiteness gate ───────────────────────────────────────────────
//
// serde_json 1.0.150 serializes a non-finite f64/f32 (NaN / ±Inf) as `null`
// in BOTH `to_value` (→ `Value::Null`) and the text serializer (→ writes
// `null`) — neither errors. `inf`/`nan` are valid TOML floats, so two configs
// or stages differing only in a non-finite value would both collapse to the
// same `null` and hash equal: a silent collision. This generic gate rejects
// any non-finite float anywhere in a `Serialize` value *before* it reaches the
// nulling serializer, field-agnostically (a future float field is covered
// automatically) — the include-by-default posture applied to finiteness.

#[derive(Debug)]
struct NonFinite(String);

impl std::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for NonFinite {}
impl serde::ser::Error for NonFinite {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        NonFinite(msg.to_string())
    }
}

/// A `Serializer` that produces nothing and only fails on a non-finite float.
#[derive(Clone, Copy)]
struct FiniteCheck;

fn nonfinite(kind: &str) -> NonFinite {
    NonFinite(format!("non-finite {kind} (NaN/Inf) in a hashed fit input"))
}

impl serde::Serializer for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_f32(self, v: f32) -> Result<(), NonFinite> {
        if v.is_finite() { Ok(()) } else { Err(nonfinite("f32")) }
    }
    fn serialize_f64(self, v: f64) -> Result<(), NonFinite> {
        if v.is_finite() { Ok(()) } else { Err(nonfinite("f64")) }
    }

    fn serialize_bool(self, _: bool) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i8(self, _: i8) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i16(self, _: i16) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i32(self, _: i32) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i64(self, _: i64) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u8(self, _: u8) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u16(self, _: u16) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u32(self, _: u32) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u64(self, _: u64) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_char(self, _: char) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_str(self, _: &str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_none(self) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_some<T: serde::Serialize + ?Sized>(self, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_newtype_struct<T: serde::Serialize + ?Sized>(self, _: &'static str, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: serde::Serialize + ?Sized>(self, _: &'static str, _: u32, _: &'static str, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple(self, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_struct_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
}

impl serde::ser::SerializeSeq for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTuple for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTupleStruct for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTupleVariant for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeMap for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_key<T: serde::Serialize + ?Sized>(&mut self, k: &T) -> Result<(), NonFinite> { k.serialize(FiniteCheck) }
    fn serialize_value<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeStruct for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, _: &'static str, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeStructVariant for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, _: &'static str, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}

/// Reject any non-finite float anywhere in `v` — call before hashing, since
/// the JSON serializer would otherwise silently collapse it to `null`.
pub(crate) fn ensure_finite<T: serde::Serialize>(v: &T) -> Result<(), String> {
    v.serialize(FiniteCheck)
        .map_err(|e| format!("cannot hash fit input: {e}"))
}

/// Per-stream data digests, sorted by stream name for a stable order.
pub(crate) fn build_data_digests(paths: &IndexMap<String, String>) -> Result<Vec<DataDigest>, String> {
    let mut entries: Vec<(&String, &String)> = paths.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = Vec::with_capacity(entries.len());
    for (name, path) in entries {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read data file '{}' ({}): {}", name, path, e))?;
        out.push(DataDigest(ContentHash::digest_bytes(&bytes)));
    }
    Ok(out)
}

/// Content digests of the explicit `[data.holdout]` streams (gh#190), sorted
/// by stream name. The fit.toml blob carries only the holdout *paths*, so
/// without this an edit to a holdout file's bytes (same path) leaves the fit
/// `run_id` unchanged and silently reuses a stale held-out score. Empty when
/// no explicit holdout is configured (temporal `holdout_after` is a numeric
/// threshold already in the blob, not a file). Reuses [`build_data_digests`],
/// the same content-addressing the training streams use.
fn build_holdout_digests(config: &FitConfigV2) -> Result<Vec<DataDigest>, String> {
    match config.data.as_ref().and_then(|d| d.holdout.as_ref()) {
        Some(holdout) => build_data_digests(holdout),
        None => Ok(Vec::new()),
    }
}

/// The fit-wide config blob hash: the whole resolved config (include-by-
/// default — so `ic_free`/`holdout`/`[config]`/priors can't be silently
/// dropped) with the slices owned by a lower level or by provenance
/// normalized out:
///   - `stages` — each stage owns its block at the stage level (excluding it
///     is what lets editing the posterior leave the scout leaf untouched);
///   - `fit_seeds` — the seed level owns the seed;
///   - `output_dir` — pure write-location provenance.
/// model/data *paths* stay (a rename is a harmless over-invalidate; their
/// *content* rides in `FitDigest.model`/`.data`).
fn fit_config_blob_hash(config: &FitConfigV2) -> Result<ContentHash, String> {
    // Reject non-finite floats first — `to_value` would null them (collision).
    ensure_finite(config)?;
    let mut v = serde_json::to_value(config)
        .map_err(|e| format!("cannot serialize fit config for hashing: {}", e))?;
    if let serde_json::Value::Object(ref mut m) = v {
        m.remove("stages");
        m.remove("fit_seeds");
        m.remove("output_dir");
    }
    Ok(digest_value(&v))
}

/// The stage-level config hash: the stage's `identity_payload` (which omits
/// the extension dim + `n_trajectories`) re-augmented with `n_trajectories`,
/// which is count-in-the-key (see the module note and
/// [`Stage::cas_n_trajectories`]).
fn stage_config_hash(stage: &Stage) -> Result<ContentHash, String> {
    // Gate the Stage struct itself: `identity_payload()` already built its
    // Value via `json!`, which would have nulled any non-finite cooling /
    // rho / tempering — so check the source struct before trusting it.
    ensure_finite(stage)?;
    let v = serde_json::json!({
        "identity": stage.identity_payload(),
        "n_trajectories": stage.cas_n_trajectories(),
    });
    Ok(digest_value(&v))
}

/// Build the `fit`-level [`FitDigest`] — the seed-independent, stage-
/// independent identity shared by every stage leaf of a fit. This is the
/// single source of truth for the `fit` level: [`resolve_fit_stage`] folds it
/// into the leaf, and [`fit_segment_dir`] hashes it into the `fits/{stem}-{h8}/`
/// directory name, so the announced fit directory and the directory the leaves
/// actually land in are guaranteed identical (they share this function).
pub fn fit_level_digest(
    model: &ir::Model,
    ir_version: &str,
    engine_version: &str,
    config: &FitConfigV2,
    data_paths: &IndexMap<String, String>,
) -> Result<FitDigest, String> {
    Ok(FitDigest {
        model: crate::resolve::model_digest(model, ir_version, engine_version),
        data: build_data_digests(data_paths)?,
        holdout_data: build_holdout_digests(config)?,
        fit_toml: fit_config_blob_hash(config)?,
        engine: EngineVersion(engine_version.to_string()),
    })
}

/// The fit-level content hash (the `fit` level's `ContentHash`). Identical to
/// the hash `resolve_fit_stage` puts on the `fit` level, so a fit's directory
/// name (`fits/{stem}-{short8}/`) and its leaves' `fit` level always agree.
pub fn fit_level_hash(
    model: &ir::Model,
    ir_version: &str,
    engine_version: &str,
    config: &FitConfigV2,
    data_paths: &IndexMap<String, String>,
) -> Result<ContentHash, String> {
    Ok(fit_level_digest(model, ir_version, engine_version, config, data_paths)?.content_hash())
}

/// The fit segment directory for a known fit-level hash:
/// `<root>/fits/{path_label(stem)}-{short8}/`. This is the grandparent of
/// every stage leaf (`store_path` factors the leaf as
/// `fits/{fit}/{NN-stage}/{seed}/`), so the directory this returns is exactly
/// where `resolve_fit_stage`'s leaves land — `fit run` can announce it and be
/// right. Pass the hash from [`fit_level_hash`] (the `fit`-level
/// `ContentHash`).
pub fn fit_segment_dir(root: &Path, stem: &str, fit_hash: &ContentHash) -> std::path::PathBuf {
    root.join(ArtifactKind::FitStage.store_dir())
        .join(format!("{}-{}", runid::path_label(stem), fit_hash.short8()))
}

/// The resolved observation-time alignment a stage will actually run under,
/// for the stage CAS identity (gh#189). Resolution is the single
/// `crate::fit::methods::resolve_obs_alignment` gate, fed the stage algorithm,
/// whether it is correlated PMMH, and the fit-wide requested
/// `[config] obs_alignment`. The resolved `Ok` value is independent of whether
/// observations sit on the `dt` grid (that flag only governs whether a combo
/// is rejected — never which alignment is returned), so a fixed `true` yields
/// the alignment without needing the observed-data times here; the runner
/// still validates on-grid-ness and errors loudly for an unsupported combo.
/// A non-inference stage (or an unsupported combo, which aborts the run before
/// any output is stored) is keyed as `Snap` — the historical uniform-grid
/// default — so it never silently aliases an `Exact` run.
fn resolved_obs_alignment(stage: &Stage, config: &FitConfigV2) -> ResolvedObsAlignment {
    use crate::run_meta::FitAlgorithm;
    if !matches!(
        stage.method_kind(),
        FitAlgorithm::If2 | FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Pfilter
    ) {
        return ResolvedObsAlignment::Snap;
    }
    let correlated = matches!(stage, Stage::PMMH { rho: Some(_), .. });
    match crate::fit::methods::resolve_obs_alignment(
        stage.method_kind(),
        correlated,
        config.config.obs_alignment,
        /* obs_on_grid = */ true,
    ) {
        Ok(crate::fit::methods::ObsAlignment::Exact) => ResolvedObsAlignment::Exact,
        Ok(crate::fit::methods::ObsAlignment::Snap) | Err(_) => ResolvedObsAlignment::Snap,
    }
}

/// Resolve a fit-stage leaf's identity: the three factored levels (fit /
/// `NN-stage` / seed) and the `run_id` derived from their hashes.
pub fn resolve_fit_stage(ctx: &FitStageCtx) -> Result<ResolvedFitStage, String> {
    let fit = fit_level_digest(
        ctx.model,
        ctx.ir_version,
        ctx.engine_version,
        ctx.config,
        ctx.data_paths,
    )?;

    let stage_config = StageConfig {
        config: stage_config_hash(ctx.stage)?,
        // Fits select observation streams via `[data]` (captured in
        // `FitDigest.data`); there is no fit-level `--obs`/`--flow`.
        obs_block: String::new(),
        flow_indices: Vec::new(),
        target_length: ctx.stage.cas_target_length(),
        // gh#189: the resolved (not requested) obs alignment, keyed per stage.
        obs_alignment: resolved_obs_alignment(ctx.stage, ctx.config),
    };
    let stage_level = StageLevel { config: stage_config, deps: Deps(ctx.deps.clone()) };

    // Fits have no grid-cell seed mixing (each sweep point is a distinct fit
    // leaf via FitDigest); the seed level hashes the resolved fit RNG seed.
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };

    let levels = vec![
        level("fit", ctx.fit_stem, fit.content_hash()),
        level(
            "stage",
            &format!("{:02}-{}", ctx.ordinal, ctx.stage_name),
            stage_level.content_hash(),
        ),
        level("seed", &format!("seed_{}", ctx.seed), seed.content_hash()),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::FitStage, &level_hashes);

    Ok(ResolvedFitStage { levels, run_id: rid })
}

/// The lineage dep for consuming an upstream stage's `fit_state.toml`: the
/// upstream's `run_id` (its identity) + the consumed file's content digest,
/// so a regenerated upstream (different θ̂) re-keys this stage. `None` if the
/// upstream has no `fit_state.toml` (e.g. a PFilter-only upstream).
pub fn cas_dep_ref(run_id: ContentHash, dir: &Path) -> Option<ArtifactRef> {
    let bytes = std::fs::read(dir.join("fit_state.toml")).ok()?;
    Some(ArtifactRef {
        run_id,
        kind: ArtifactKind::FitStage,
        artifact: "fit_state.toml".to_string(),
        digest: ContentHash::digest_bytes(&bytes),
    })
}

/// The lineage dep for an external `StartsFrom::Directory`: prefer the
/// upstream's CAS `run_id` (from its `run.json`); if it isn't a CAS leaf, use
/// the consumed file's digest as a stand-in identity (content still re-keys).
pub fn cas_dep_from_dir(dir: &Path) -> Option<ArtifactRef> {
    let bytes = std::fs::read(dir.join("fit_state.toml")).ok()?;
    let digest = ContentHash::digest_bytes(&bytes);
    let run_id = std::fs::read(dir.join("run.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<RunRecord>(&b).ok())
        .map(|r| r.run_id)
        .unwrap_or(digest);
    Some(ArtifactRef {
        run_id,
        kind: ArtifactKind::FitStage,
        artifact: "fit_state.toml".to_string(),
        digest,
    })
}

/// Build the `deps` entry for a survey consumed by `init = "survey_top_k"`.
/// The survey is a content-addressed `Survey` leaf; folding its `run_id`
/// (content identity) + `landscape.tsv` digest into the fit stage's deps means
/// a regenerated survey — even one written back to the same path — re-keys the
/// fit. Returns `None` if the survey dir is unreadable (the fit will fail in
/// init anyway, producing no stored output to mis-key). The `landscape.tsv`
/// digest is the bytes the top-K rows are actually read from; `run_id` falls
/// back to it if `run.json` is missing.
pub fn cas_survey_dep(dir: &Path) -> Option<ArtifactRef> {
    let landscape = std::fs::read(dir.join("landscape.tsv")).ok()?;
    let digest = ContentHash::digest_bytes(&landscape);
    let run_id = std::fs::read(dir.join("run.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<RunRecord>(&b).ok())
        .map(|r| r.run_id)
        .unwrap_or(digest);
    Some(ArtifactRef {
        run_id,
        kind: ArtifactKind::Survey,
        artifact: "landscape.tsv".to_string(),
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pgas_stage(n_trajectories: usize) -> Stage {
        toml::from_str(&format!(
            "algorithm = \"pgas\"\n\
             backend = \"chain_binomial\"\n\
             chains = 2\n\
             particles = 100\n\
             sweeps = 10\n\
             n_trajectories = {n_trajectories}"
        ))
        .expect("minimal PGAS stage toml must parse")
    }

    /// Guardrail-1 (gh#147 M3.2, deviation-A). `n_trajectories` is folded into
    /// the stage identity, so two stages differing ONLY in it produce DISTINCT
    /// stage hashes — each value gets its own leaf and there is no silent
    /// reuse of the wrong trajectory count. (The collision this guards is the
    /// silent-wrong-answer class: without the fold, both map to one `run_id`
    /// and the first run's trajectories are served for the second.)
    #[test]
    fn n_trajectories_changes_the_stage_identity() {
        let h200 = stage_config_hash(&pgas_stage(200)).unwrap();
        let h10 = stage_config_hash(&pgas_stage(10)).unwrap();
        assert_ne!(
            h200, h10,
            "two stages differing only in n_trajectories must have distinct \
             stage hashes (guardrail-A) — else a changed n_trajectories \
             silently reuses the wrong trajectory count"
        );
        // No spurious sensitivity: identical n_trajectories → identical hash.
        assert_eq!(h200, stage_config_hash(&pgas_stage(200)).unwrap());
    }

    /// The deterministic compute identity is stable: re-resolving the same
    /// stage config yields the same hash (canonical-JSON key sort).
    #[test]
    fn stage_config_hash_is_stable() {
        let a = stage_config_hash(&pgas_stage(50)).unwrap();
        let b = stage_config_hash(&pgas_stage(50)).unwrap();
        assert_eq!(a, b);
    }

    fn minimal_config(extra: &str) -> FitConfigV2 {
        toml::from_str(&format!(
            "[model]\ncamdl = \"models/sir.camdl\"\n\
             [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\
             [fixed]\nN0 = 1000000\n\
             {extra}\
             [stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n"
        ))
        .expect("minimal fit config must parse")
    }

    fn if2_stage(loglik_particles: usize) -> Stage {
        toml::from_str(&format!(
            "algorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\niterations = 10\ncooling = 0.7\n\
             loglik_eval = {{ n_particles = {loglik_particles}, n_replicates = 8 }}"
        ))
        .expect("minimal IF2 stage toml must parse")
    }

    /// gh#189: `loglik_eval` determines the reported θ̂/loglik, so it's part of
    /// the IF2 stage identity (folded via IF2's whole-serialize identity_payload).
    /// Two stages differing only in it must get distinct hashes — there is no CLI
    /// override that could silently re-score under the same key (the flag was
    /// removed; loglik_eval is set in the stage TOML only).
    #[test]
    fn loglik_eval_changes_the_if2_stage_identity() {
        let a = stage_config_hash(&if2_stage(4000)).unwrap();
        let b = stage_config_hash(&if2_stage(8000)).unwrap();
        assert_ne!(a, b, "loglik_eval must fold into the IF2 stage identity (gh#189)");
        assert_eq!(a, stage_config_hash(&if2_stage(4000)).unwrap());
    }

    /// gh#189: `allow_degenerate_rates` is a keyed `[config]` field (was a CLI
    /// flag that bypassed the fit-identity hash). It changes collapse handling
    /// (hard-error → silent-zero), which changes trajectory values, so two fits
    /// differing only in it must get distinct fit-level hashes.
    #[test]
    fn allow_degenerate_rates_changes_the_fit_identity() {
        let off = fit_config_blob_hash(&minimal_config("")).unwrap();
        let on = fit_config_blob_hash(
            &minimal_config("[config]\nallow_degenerate_rates = true\n")).unwrap();
        assert_ne!(off, on,
            "allow_degenerate_rates=true must fold into the fit identity (gh#189)");
        // skip_serializing_if: explicit `false` hashes identically to absent, so
        // existing fits (which omit it) don't spuriously re-key.
        let explicit_false = fit_config_blob_hash(
            &minimal_config("[config]\nallow_degenerate_rates = false\n")).unwrap();
        assert_eq!(off, explicit_false,
            "explicit allow_degenerate_rates=false must match the default (no re-key)");
    }

    /// gh#241 PR F: input-surface differential for the fit-stage identity path.
    /// The identity guarantee made executable, complementing the per-field
    /// sensitivity tests above with the PRESENTATION-inert half: a semantic
    /// fit-config input re-keys the fit blob; a provenance field does not.
    #[test]
    fn differential_fit_config_semantic_vs_presentation() {
        // SEMANTIC: a `[config].dt` change re-keys the fit-wide blob.
        let base = fit_config_blob_hash(&minimal_config("")).unwrap();
        let dt = fit_config_blob_hash(&minimal_config("[config]\ndt = 0.5\n")).unwrap();
        assert_ne!(base, dt, "[config].dt is semantic — must re-key the fit blob");

        // PRESENTATION: `output_dir` is normalized OUT of the fit identity (pure
        // write-location provenance), so two configs differing only in it hash equal.
        let cfg = |out: &str| {
            toml::from_str::<FitConfigV2>(&format!(
                "output_dir = \"{out}\"\n[model]\ncamdl = \"models/sir.camdl\"\n\
                 [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n[fixed]\nN0 = 1000000\n\
                 [stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                 chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n"
            ))
            .expect("config must parse")
        };
        assert_eq!(
            fit_config_blob_hash(&cfg("results/run_a")).unwrap(),
            fit_config_blob_hash(&cfg("results/run_b")).unwrap(),
            "output_dir is provenance — it is normalized out and must NOT affect the fit blob"
        );
    }

    /// gh#241: synthetic fits key their content-addressed directory off the
    /// SAME `runid` fit-level digest as real fits — `fit_segment_dir` over
    /// `fit_level_hash` computed with an EMPTY data map (a synthetic fit has no
    /// input data; the data is generated, and each generated cell keys its own
    /// leaf). With the model held fixed and data empty, the synthetic dir
    /// changes iff the fit-wide blob changes, so this pins the blob's behavior:
    ///   - a semantic `[synthetic]` change (different ground truth / sim seeds →
    ///     different generated data) re-keys the dir;
    ///   - a seed-only `fit_seeds` change does NOT (the fit RNG seed is a lower
    ///     CAS level under the segment, not part of the segment name).
    /// The legacy `fit_content_hash` hashed the whole fit.toml bytes, so it
    /// over-keyed on `fit_seeds`/`output_dir`/stage edits; routing synthetic
    /// through the runid blob fixes that, matching real fits.
    #[test]
    fn synthetic_fit_dir_is_seed_stable_and_semantic_sensitive() {
        // A complete synthetic fit config; `top` is spliced at the very top so a
        // top-level key (`fit_seeds`) isn't absorbed into a `[table]` above it.
        let syn = |top: &str, sim_seeds: &str| -> FitConfigV2 {
            toml::from_str(&format!(
                "{top}\
                 [model]\ncamdl = \"models/sir.camdl\"\n\
                 [synthetic]\ntrue_params = \"truth.toml\"\nsim_seeds = \"{sim_seeds}\"\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n[fixed]\nN0 = 1000000\n\
                 [stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                 chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n"
            ))
            .expect("synthetic fit config must parse")
        };

        let base = fit_config_blob_hash(&syn("", "1:5")).unwrap();

        // Seed-only: `fit_seeds` is normalized out of the blob (the seed CAS
        // level owns it), so the synthetic dir is stable across seed changes.
        let seed_only = fit_config_blob_hash(&syn("fit_seeds = [1, 2, 3]\n", "1:5")).unwrap();
        assert_eq!(
            base, seed_only,
            "a seed-only fit_seeds change must NOT re-key the synthetic fit dir (gh#241)"
        );

        // Semantic: different `[synthetic].sim_seeds` → different generated data
        // → must re-key the synthetic fit dir.
        let resim = fit_config_blob_hash(&syn("", "1:9")).unwrap();
        assert_ne!(
            base, resim,
            "a [synthetic].sim_seeds change generates different data and must \
             re-key the synthetic fit dir (gh#241)"
        );
    }

    /// gh#134: `condition_from` is part of the fit IDENTITY — a different
    /// conditioning window is a different fit / estimand. So a SET value must
    /// fold into the fit-level blob hash and re-key the fit; an UNSET value
    /// (the common case) must leave the hash bit-identical so existing fits'
    /// `run_id`s are unchanged. `skip_serializing_if = Option::is_none` is what
    /// makes the unset case inert. Both surface forms (`All` string, `PerStream`
    /// table) re-key, and distinct values produce distinct hashes.
    /// A `minimal_config` variant with a TOP-LEVEL `condition_from` line. The
    /// `minimal_config(extra)` helper splices `extra` inside `[fixed]`, which is
    /// wrong for a top-level key (it would be absorbed by `FixedParams`'
    /// flattened param map), so we build the doc directly here.
    fn config_with_top_level(cond_line: &str) -> FitConfigV2 {
        toml::from_str(&format!(
            "{cond_line}\
             [model]\ncamdl = \"models/sir.camdl\"\n\
             [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\
             [fixed]\nN0 = 1000000\n\
             [stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n"
        ))
        .expect("fit config must parse")
    }

    #[test]
    fn condition_from_changes_the_fit_identity_when_set() {
        let unset = fit_config_blob_hash(&config_with_top_level("")).unwrap();

        // A SET value (`All` string form) re-keys the fit.
        let set_str = fit_config_blob_hash(
            &config_with_top_level("condition_from = \"first_obs - 1 week\"\n")).unwrap();
        assert_ne!(unset, set_str,
            "a SET condition_from must fold into the fit identity (gh#134)");

        // The `PerStream` table form also re-keys.
        let set_table = fit_config_blob_hash(&config_with_top_level(
            "[condition_from]\ndefault = \"first_obs - 1 week\"\n")).unwrap();
        assert_ne!(unset, set_table,
            "a [condition_from] table must fold into the fit identity");

        // Two DIFFERENT conditioning windows are two different fits.
        let set_str2 = fit_config_blob_hash(
            &config_with_top_level("condition_from = \"first_obs - 2 weeks\"\n")).unwrap();
        assert_ne!(set_str, set_str2,
            "distinct condition_from values must produce distinct fit hashes");

        // No spurious sensitivity: identical value → identical hash.
        assert_eq!(set_str, fit_config_blob_hash(
            &config_with_top_level("condition_from = \"first_obs - 1 week\"\n")).unwrap());
    }

    /// gh#134 (the bit-identical guarantee): an UNSET `condition_from` must NOT
    /// re-key. Because the field carries `skip_serializing_if = Option::is_none`,
    /// the absent key serializes to nothing, so the blob is byte-identical to a
    /// config that never had the field — existing fits keep their `run_id`.
    #[test]
    fn unset_condition_from_does_not_change_the_fit_identity() {
        // The pre-condition_from `minimal_config` (which never mentions the key)
        // must hash identically to a config that explicitly omits it.
        let legacy = fit_config_blob_hash(&minimal_config("")).unwrap();
        let explicit_absent = fit_config_blob_hash(&config_with_top_level("")).unwrap();
        assert_eq!(legacy, explicit_absent,
            "an unset condition_from must be bit-identical — existing fits must \
             keep their run_id (gh#134)");
        // And the parsed config genuinely has it as None.
        assert!(config_with_top_level("").condition_from.is_none());
        assert!(minimal_config("").condition_from.is_none());
    }

    /// gh#189: the integrator `[config] dt` changes the substep grid (and, via
    /// `Expr::Dt` and the obs-alignment substep window, the computed output), so
    /// two fits differing only in `dt` must get distinct fit-level hashes — else
    /// a changed `dt` silently reuses a stale fit / stage leaf (the
    /// count-in-the-key, silent-wrong-answer class). `dt` is keyed at the fit
    /// level (`[config]` is part of the serialized fit blob); the stage leaf
    /// nests under it, so the whole stage artifact (incl. `resume_state.bin`)
    /// relocates when `dt` changes — there is no stale resume.
    #[test]
    fn dt_changes_the_fit_identity() {
        let a = fit_config_blob_hash(&minimal_config("[config]\ndt = 1.0\n")).unwrap();
        let b = fit_config_blob_hash(&minimal_config("[config]\ndt = 3.0\n")).unwrap();
        assert_ne!(
            a, b,
            "two fits differing only in [config] dt must have distinct fit-level \
             hashes (gh#189) — a changed dt changes the substep grid and the \
             output, so it must re-key the fit"
        );
        // No spurious sensitivity: identical dt → identical hash.
        assert_eq!(a, fit_config_blob_hash(&minimal_config("[config]\ndt = 1.0\n")).unwrap());
    }

    /// gh#190: holdout files are digested by CONTENT, not path. Two configs
    /// with the SAME holdout path but DIFFERENT bytes on disk must produce
    /// different holdout digests — so editing a holdout file (same path)
    /// re-keys the fit and a stale held-out score cannot be silently reused.
    #[test]
    fn holdout_content_changes_the_fit_digest() {
        let dir = tempfile::tempdir().unwrap();
        let holdout_path = dir.path().join("holdout.tsv");
        let holdout_str = holdout_path.to_string_lossy().to_string();
        // `minimal_config`'s base already declares `[data.observations]`;
        // `[data.holdout]` is a new sub-table of the same `[data]`.
        let cfg = minimal_config(&format!(
            "[data.holdout]\nweekly_cases = \"{}\"\n",
            holdout_str
        ));
        // Sanity: the holdout block parsed and points at our temp path.
        assert_eq!(
            cfg.data.as_ref().unwrap().holdout.as_ref().unwrap()["weekly_cases"],
            holdout_str
        );

        std::fs::write(&holdout_path, b"time\tweekly_cases\n50\t10\n").unwrap();
        let d1 = build_holdout_digests(&cfg).unwrap();
        std::fs::write(&holdout_path, b"time\tweekly_cases\n50\t999\n").unwrap();
        let d2 = build_holdout_digests(&cfg).unwrap();

        assert_eq!(d1.len(), 1, "one holdout stream digested");
        assert_ne!(
            d1, d2,
            "editing a holdout file's content (same path) must change the holdout \
             digest (gh#190) — content-addressed, not path-addressed"
        );

        // No explicit holdout → empty (temporal holdout_after is a numeric
        // threshold already in the fit.toml blob, not a file).
        let no_holdout = minimal_config("");
        assert!(build_holdout_digests(&no_holdout).unwrap().is_empty());
    }

    /// gh#189: the *resolved* obs alignment is folded into the stage identity,
    /// and resolution is per-algorithm. A PGAS stage resolves to `Snap`; an
    /// IF2 stage resolves to `Exact`. The two must therefore produce distinct
    /// `StageConfig`s (so a snap and an exact fit never collide in the store).
    #[test]
    fn resolved_obs_alignment_is_keyed_per_stage() {
        let if2 = if2_stage(4000);
        let pgas = pgas_stage(200);
        let cfg = minimal_config(""); // obs_alignment unset → per-algorithm default
        assert_eq!(
            resolved_obs_alignment(&if2, &cfg),
            ResolvedObsAlignment::Exact,
            "if2 resolves to exact (steps exactly to obs times)"
        );
        assert_eq!(
            resolved_obs_alignment(&pgas, &cfg),
            ResolvedObsAlignment::Snap,
            "pgas resolves to snap (uniform grid)"
        );

        // The resolved value is folded into the stage-level identity. Holding
        // every other StageConfig field fixed, flipping ONLY the resolved
        // alignment must change the stage hash (snap and exact can't collide).
        let mk = |align: ResolvedObsAlignment| -> ContentHash {
            StageConfig {
                config: ContentHash::from_bytes([0; 32]),
                obs_block: String::new(),
                flow_indices: Vec::new(),
                target_length: 0,
                obs_alignment: align,
            }
            .content_hash()
        };
        assert_ne!(
            mk(ResolvedObsAlignment::Snap),
            mk(ResolvedObsAlignment::Exact),
            "the resolved obs_alignment must distinguish the stage identities (gh#189)"
        );
    }

    /// A survey consumed by `init = "survey_top_k"` is folded into the fit
    /// stage's `deps` by its CONTENT, so two surveys with different landscapes
    /// (even written to the same path, one after the other) produce different
    /// deps → the fit re-keys. Missing landscape → `None` (the fit fails in
    /// init, producing no stored output to mis-key).
    #[test]
    fn cas_survey_dep_is_content_sensitive() {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        std::fs::write(d1.path().join("landscape.tsv"), b"theta\tll\n0.10\t-5.0\n").unwrap();
        std::fs::write(d2.path().join("landscape.tsv"), b"theta\tll\n0.10\t-9.9\n").unwrap();
        let r1 = cas_survey_dep(d1.path()).expect("dep from a readable survey dir");
        let r2 = cas_survey_dep(d2.path()).expect("dep from a readable survey dir");
        assert_ne!(r1.digest, r2.digest,
            "different survey landscape content must yield a different dep digest \
             (regenerating a survey at the same path re-keys the fit)");
        assert_eq!(r1.kind, runid::ArtifactKind::Survey);
        assert_eq!(r1.artifact, "landscape.tsv");

        // No landscape.tsv → no dep (unreadable survey; fit will fail in init).
        let d3 = tempfile::tempdir().unwrap();
        assert!(cas_survey_dep(d3.path()).is_none());
    }
}
