//! gh#147 (M3.2). The fit-stage CAS identity: map a resolved fit stage into
//! the `runid` factored levels (`fit` / `NN-stage` / `seed`) and its leaf
//! `run_id`.
//!
//! Factoring (per the proposal — `FitDigest` excludes `[stages.*]` so editing
//! the posterior's block doesn't re-key the scout; cross-stage invalidation
//! rides the deps-DAG):
//!
//! - **fit level** = [`FitDigest`] = whole-IR model digest + per-stream data
//!   digests + the canonicalized fit-wide config (with `stages` / `fit_seeds`
//!   / `output_dir` normalized out) + engine version.
//! - **stage level** = [`StageLevel`] = [`StageConfig`] (the stage's
//!   `identity_payload` + `n_trajectories` + obs/flow + `target_length`)
//!   folded with its `deps` (so `02-posterior`'s hash folds in `01-scout`'s
//!   identity).
//! - **seed level** = [`Seed`] = the resolved fit RNG seed.
//!
//! **Commit-1 deviation (tracked to M3.3):** PGAS `n_trajectories` is an
//! output-shaping knob that `identity_payload` omits, and PGAS writes
//! `chain_N/trajectories/` nested under the leaf. To avoid a silent reuse of
//! the wrong trajectory count (the same stage `run_id` serving a different
//! `n_trajectories`'s output), Commit 1 folds `n_trajectories` into the stage
//! identity ([`Stage::cas_n_trajectories`]) — so each value yields a distinct
//! leaf (correct), at the cost of re-fitting when it changes. M3.3 relocates
//! `trajectories/` to a root-level child keyed on `n_trajectories` and removes
//! it from the stage identity, at which point changing it is a cheap re-save;
//! existing Commit-1 fit leaves re-key once when that lands (fine pre-beta).

use indexmap::IndexMap;

use std::path::Path;

use runid::inputs::{
    ArtifactRef, DataDigest, Deps, EngineVersion, FitDigest, Seed, StageConfig, StageLevel,
};
use runid::{
    run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId, Provenance, RunRecord, RunStatus,
    FORMAT_VERSION, HASH_VERSION,
};

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

fn level(name: &str, label: &str, hash: ContentHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash, schema_version: 1 }
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

fn digest_value(v: &serde_json::Value) -> ContentHash {
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
fn ensure_finite<T: serde::Serialize>(v: &T) -> Result<(), String> {
    v.serialize(FiniteCheck)
        .map_err(|e| format!("cannot hash fit input: {e}"))
}

/// Per-stream data digests, sorted by stream name for a stable order.
fn build_data_digests(paths: &IndexMap<String, String>) -> Result<Vec<DataDigest>, String> {
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
/// the extension dim + `n_trajectories`) re-augmented with `n_trajectories`
/// (the Commit-1 deviation — see the module note and
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

/// Resolve a fit-stage leaf's identity: the three factored levels (fit /
/// `NN-stage` / seed) and the `run_id` derived from their hashes.
pub fn resolve_fit_stage(ctx: &FitStageCtx) -> Result<ResolvedFitStage, String> {
    let model = crate::resolve::model_digest(ctx.model, ctx.ir_version, ctx.engine_version);
    let data = build_data_digests(ctx.data_paths)?;
    let fit = FitDigest {
        model,
        data,
        fit_toml: fit_config_blob_hash(ctx.config)?,
        engine: EngineVersion(ctx.engine_version.to_string()),
    };

    let stage_config = StageConfig {
        config: stage_config_hash(ctx.stage)?,
        // Fits select observation streams via `[data]` (captured in
        // `FitDigest.data`); there is no fit-level `--obs`/`--flow`.
        obs_block: String::new(),
        flow_indices: Vec::new(),
        target_length: ctx.stage.cas_target_length(),
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

/// Build the `RunRecord` for a fit-stage leaf. `inputs` carries the
/// (recorded-not-hashed) FitStageMeta-equivalent for `show`/`status` display;
/// identity is the `levels` + `deps`. `artifacts` is filled by the store at
/// `finalize` (the recursive exact-set of everything the runners streamed).
pub fn build_fit_stage_record(
    resolved: &ResolvedFitStage,
    deps: &[ArtifactRef],
    ir_version: &str,
    status: RunStatus,
    inputs: serde_json::Value,
    model_path: &str,
) -> RunRecord {
    RunRecord {
        format_version: FORMAT_VERSION,
        kind: ArtifactKind::FitStage,
        run_id: resolved.run_id,
        hash_version: HASH_VERSION,
        ir_version: ir_version.to_string(),
        engine_version: crate::version::VERSION_SHORT.to_string(),
        levels: resolved.levels.clone(),
        deps: deps.to_vec(),
        status,
        artifacts: Default::default(),
        children: Default::default(),
        inputs,
        provenance: Provenance {
            argv: std::env::args().collect(),
            created_at: Some(crate::cas::iso8601_utc(std::time::SystemTime::now())),
            camdl_version: Some(crate::version::VERSION_SHORT.to_string()),
            // The model source path (provenance) so `fit summary`/`table` can
            // recover the model for calendar/instant rendering without a
            // fit-wide record (the fit level is a path segment now).
            source_paths: vec![model_path.to_string()],
            ..Default::default()
        },
    }
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
}
