//! The `Resolve` bridge for `simulate`/`batch`: map already-resolved CLI
//! inputs into the `runid` identity types — `TrajectoryInput`, the factored
//! `LevelId`s, and the leaf `run_id`.
//!
//! Every field that enters a hash is a *resolved value*: param/scenario maps
//! are concrete name→value, the seed is the resolved `process_seed` (never
//! the base `--seed`), and a non-finite resolved float is a `ResolveError`
//! surfaced before any hashing. Presentation-only IR fields
//! (`output.format`, `simulation.time_semantics`) are normalized out of the
//! hashed model so `--format`/time rendering stay inert (provenance) — that
//! strip lives inside `runid::inputs::ModelDigest::from_model`, NOT here, so
//! every artifact kind inherits it (gh#442). This module must never
//! re-implement it.
//!
//! This is the identity half of the wiring; the caller supplies resolved
//! config/params/scenario/seed (from `params_resolver`, the model, and the
//! seed mixer) and feeds the resulting path + record to `runid`'s `Layout`
//! and `CasStore`. The model digest is the whole-IR digest (so an obs-only
//! edit over-invalidates the trajectory, but never under-invalidates).

use std::collections::{BTreeMap, HashMap};

use runid::inputs::{
    Backend, CalendarMode, DataDigest, EngineVersion, InterventionId, ModelDigest, ParamId,
    ResolvedOutputSchedule, ResolvedParams, ResolvedScenario, Seed, SimConfig,
};
use runid::{run_id, ArtifactKind, ContentAddressed, ContentHash, FiniteF64, LevelId, ResolveError};

/// A fully-resolved trajectory leaf: the factored identity levels (in path
/// order) and the leaf `run_id` (composed from the level hashes).
///
/// In the factored scheme the identity is the *tuple of per-level digests*,
/// not a single flat hash, so the leaf is described by its levels — each a
/// disjoint slice of the input set (model / config / params / scenario /
/// seed) — and the `run_id` derived from them.
pub struct ResolvedTrajectory {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_trajectory`], all already resolved by the caller. The
/// labels are provenance (readable path segments); identity rides in the
/// hashes.
pub struct TrajectoryCtx<'a> {
    pub model: &'a ir::Model,
    pub model_stem: &'a str,
    /// IR schema version string (e.g. `"0.7"`), matching `ir/VERSION`.
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// Resolved config.
    pub backend: crate::args::types::ForwardBackend,
    pub dt: f64,
    pub t_start: f64,
    pub t_end: f64,
    pub output: &'a ir::model::OutputSchedule,
    pub allow_degenerate_rates: bool,
    /// Output view (gh#156): the trajectory column filter, folded into the
    /// `config` level so a column-subset leaf re-keys. `--output-every` is NOT
    /// here — it is lowered into `output` (the schedule) upstream.
    pub no_flows: bool,
    pub columns: &'a std::collections::BTreeSet<String>,
    /// Resolved base parameter values (name → value) + table-file digests.
    pub base_params: &'a HashMap<String, f64>,
    pub table_digests: Vec<DataDigest>,
    /// Resolved scenario delta.
    pub enable: &'a [String],
    pub disable: &'a [String],
    pub scen_params: &'a HashMap<String, f64>,
    /// Readable labels (provenance).
    pub param_label: &'a str,
    pub scenario_label: &'a str,
    /// The base `--seed` (readable label) and the resolved `process_seed`
    /// (the actual trajectory driver, and the only seed value hashed).
    pub base_seed: u64,
    pub process_seed: u64,
    /// gh#641 / gh#697: the inferred state this cell was seeded from (`simulate
    /// --init-state`), or `None` when it built its initial state from the
    /// model's `init {}`. The origin ensemble's content digest plus the restored
    /// row — see [`runid::inputs::InitStateDigest`] for why both are identity.
    pub init_state: Option<runid::inputs::InitStateDigest>,
}

/// Map a CLI backend to the resolved `runid` backend.
fn backend(b: crate::args::types::ForwardBackend) -> Backend {
    use crate::args::types::ForwardBackend as B;
    match b {
        B::Gillespie => Backend::Gillespie,
        B::ChainBinomial => Backend::ChainBinomial,
        B::Ode => Backend::Ode,
    }
}

/// The model's *structural* content identity, hex-encoded: the `runid` model
/// content hash ([`ModelDigest::ir`] = presentation-normalized
/// `Model::content_hash`), rendered as hex. The single helper every recorded
/// "model identity" string and the survey↔fit warm-start cross-check goes
/// through, so the survey writer and the fit's recompute can never disagree.
///
/// Takes the **raw compiled IR JSON** (the `model_ir_json` both survey and fit
/// already carry), NOT an in-memory `ir::Model`: survey and fit each seed
/// `[estimate].start` into their working model (gh#92 / gh#34), so hashing the
/// parsed model would make the cross-check spuriously sensitive to a start-value
/// edit. Hashing the raw IR — the .camdl's compiled output — keeps the identity
/// tracking the *model*, exactly as the retired `model_hash` did.
///
/// Deliberately **structural only** — it folds the model IR, not the engine
/// version or IR schema string. This is `model_digest`'s `ir` component, *not*
/// the full per-level [`ModelDigest`] (which also folds `engine` = the volatile
/// `VERSION_SHORT` git hash). The cross-check asks "is this the same *model*?"
/// (a model edit between survey and fit invalidates it), not "the same engine
/// build?": folding the per-commit engine hash would refuse every warm-start
/// from a survey built at a different commit, and engine/schema skew is already
/// the camdlc↔camdl version guard's job. So this stays usable across rebuilds.
///
/// Returns an empty string when the IR can't be parsed (a partially-written or
/// absent IR, as the fit sidecar tolerates) — at the cross-check sites the IR is
/// always a freshly-compiled, valid envelope, so the value is never empty there.
pub fn model_identity_from_ir(model_ir_json: &str) -> String {
    match ir::from_str(model_ir_json) {
        // `model_ir_hash` owns the presentation strip — the same function
        // `ModelDigest::from_model` uses, so this string and the `model` level
        // can never disagree about what "the same model" means (gh#442).
        Ok(model) => runid::inputs::model_ir_hash(&model).to_hex(),
        Err(_) => String::new(),
    }
}

fn finite(x: f64) -> Result<FiniteF64, ResolveError> {
    FiniteF64::new(x).map_err(ResolveError::from)
}

fn resolve_output(sched: &ir::model::OutputSchedule) -> Result<ResolvedOutputSchedule, ResolveError> {
    use ir::model::OutputSchedule as O;
    Ok(match sched {
        O::Regular(r) => ResolvedOutputSchedule::Regular {
            start: finite(r.start)?,
            step: finite(r.step)?,
        },
        O::AtTimes(ts) => {
            let mut v = Vec::with_capacity(ts.len());
            for t in ts {
                v.push(finite(*t)?);
            }
            ResolvedOutputSchedule::AtTimes(v)
        }
    })
}

fn resolve_params(
    values: &HashMap<String, f64>,
    tables: Vec<DataDigest>,
) -> Result<ResolvedParams, ResolveError> {
    let mut m: BTreeMap<ParamId, FiniteF64> = BTreeMap::new();
    for (k, v) in values {
        m.insert(ParamId(k.clone()), finite(*v)?);
    }
    Ok(ResolvedParams { values: m, tables })
}

fn resolve_scenario(
    enable: &[String],
    disable: &[String],
    patch: &HashMap<String, f64>,
) -> Result<ResolvedScenario, ResolveError> {
    let enabled = enable.iter().cloned().map(InterventionId).collect();
    let disabled = disable.iter().cloned().map(InterventionId).collect();
    let mut p: BTreeMap<ParamId, FiniteF64> = BTreeMap::new();
    for (k, v) in patch {
        p.insert(ParamId(k.clone()), finite(*v)?);
    }
    Ok(ResolvedScenario { enabled, disabled, patch: p })
}

/// The readable config label: `{backend}-dt{dt}` (e.g. `chain_binomial-dt1`).
fn config_label(b: crate::args::types::ForwardBackend, dt: f64) -> String {
    format!("{}-dt{}", b.as_str(), dt)
}

use crate::fit::cas::level;

/// Resolve a trajectory leaf's identity: its `TrajectoryInput`, the five
/// factored levels (model/config/params/scenario/seed, in path order), and
/// the `run_id` derived from them.
pub fn resolve_trajectory(ctx: &TrajectoryCtx) -> Result<ResolvedTrajectory, ResolveError> {
    let model = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        EngineVersion(ctx.engine_version.to_string()),
    );
    let config = SimConfig {
        backend: backend(ctx.backend),
        dt: finite(ctx.dt)?,
        t_start: finite(ctx.t_start)?,
        t_end: finite(ctx.t_end)?,
        output: resolve_output(ctx.output)?,
        // Placeholder until the calendar-time work lands and the resolver
        // produces a concrete mode (M2 decision: minimal, provisional).
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: ctx.allow_degenerate_rates,
        no_flows: ctx.no_flows,
        columns: ctx.columns.clone(),
        init_state: ctx.init_state.clone(),
    };
    let params = resolve_params(ctx.base_params, ctx.table_digests.clone())?;
    let scenario = resolve_scenario(ctx.enable, ctx.disable, ctx.scen_params)?;
    let seed = Seed { process_seed: ctx.process_seed, base_seed: ctx.base_seed };

    // Each level hashes a disjoint slice of the input; the union is the whole
    // resolved input set. The label is provenance, the hash is identity.
    let levels = vec![
        level("model", ctx.model_stem, model.content_hash()),
        level("config", &config_label(ctx.backend, ctx.dt), config.content_hash()),
        level("params", ctx.param_label, params.content_hash()),
        level("scenario", ctx.scenario_label, scenario.content_hash()),
        level("seed", &format!("seed_{}", ctx.base_seed), seed.content_hash()),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::Sim, &level_hashes);

    Ok(ResolvedTrajectory { levels, run_id: rid })
}

// ─── The resolved-writer seam (gh#241 PR D) ──────────────────────────────────
//
// One choke point for every CAS write. A command resolves its identity into a
// `ResolvedArtifact`, supplies write-time provenance via `RecordMeta`, and calls
// `begin_resolved_write`. The `RunRecord` is assembled in exactly ONE place
// (`build_record`), so `run.json.inputs` always comes from
// `ResolvedArtifact::display_inputs` and can never drift from identity, and a
// new artifact kind cannot reach the store without a resolved shape first.
//
// Zero-re-key (gh#241 PR D): the path is `store_path(root, kind, levels)` and
// the record is byte-for-byte what the per-kind builders produced — this seam
// is a structural unification, not an identity change.

use std::path::{Path, PathBuf};

use runid::store::StreamClaim;
use runid::{
    store_path, Artifacts, CasError, FsCasStore, Provenance, RunRecord, RunStatus, FORMAT_VERSION,
    HASH_VERSION,
};

/// The identity + display contract for one CAS leaf. Identity rides in
/// `kind`/`levels`/`run_id`; `display_inputs` is the provenance summary written
/// to `run.json.inputs` (NEVER hashed). For a streaming artifact whose summary
/// is a post-run result (e.g. a pfilter's loglik), `display_inputs` is the
/// at-claim value (often `Null`); the final value is supplied to
/// [`ResolvedClaim::finalize`].
pub struct ResolvedArtifact {
    pub kind: ArtifactKind,
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
    pub display_inputs: serde_json::Value,
}

/// Write-time, non-identity record context: lineage + provenance the record
/// schema needs but that never enters the `run_id`.
pub struct RecordMeta {
    pub ir_version: String,
    pub engine_version: String,
    pub deps: Vec<runid::inputs::ArtifactRef>,
    pub children: BTreeMap<String, Vec<ContentHash>>,
    pub source_paths: Vec<String>,
    pub label: Option<String>,
    /// Column schema for the leaf's tabular outputs, attached to the record at
    /// build time — used by atomic-commit paths (e.g. `sim`, whose files are
    /// known in memory before commit); the streaming fit path sets it at
    /// finalize instead.
    pub output_schema: BTreeMap<String, runid::record::TableSchema>,
}

impl RecordMeta {
    /// The common case: engine = this binary, no deps/children, one source path.
    pub fn new(
        ir_version: impl Into<String>,
        model_path: impl Into<String>,
        label: Option<String>,
    ) -> Self {
        RecordMeta {
            ir_version: ir_version.into(),
            engine_version: crate::version::VERSION_SHORT.to_string(),
            deps: Vec::new(),
            children: BTreeMap::new(),
            source_paths: vec![model_path.into()],
            label,
            output_schema: BTreeMap::new(),
        }
    }

    pub fn with_deps(mut self, deps: Vec<runid::inputs::ArtifactRef>) -> Self {
        self.deps = deps;
        self
    }

    pub fn with_children(mut self, children: BTreeMap<String, Vec<ContentHash>>) -> Self {
        self.children = children;
        self
    }
}

/// Which store door to use — mirrors the store's two write modes.
pub enum WriteMode {
    /// Hand over a finished artifact set; committed atomically in one call.
    Atomic(Artifacts),
    /// Claim the leaf, stream output into it, then finalize.
    Streaming,
}

/// What the caller wants done about an artifact that already exists.
///
/// Every command has a `--force`-shaped flag, and before this type each one
/// decided for itself what that meant: batch recomputed and had its bytes
/// discarded at commit, fit and survey hard-errored with `AlreadyCompleted`,
/// pfilter and profile had no force at all, and simulate threaded a flag
/// nothing ever read. One meaning now, in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePolicy {
    /// A valid stored artifact is reused (the default).
    Reuse,
    /// Recompute even if a valid artifact exists; the incumbent is
    /// quarantined, never deleted.
    Force,
}

/// The answer to "does this artifact need computing?", asked BEFORE the work.
///
/// **Deviation from the proposal** (§S2 sketched one `WriteVerdict` returned by
/// `begin_resolved_write`): reuse and write happen at two different moments —
/// the reuse question must be answered before a cell is computed, while
/// `begin_resolved_write` is called after, with results in hand. Folding both
/// into one call would force every command to compute first and ask later,
/// which is precisely the wasted work the cache exists to avoid. So the policy
/// seam is two functions sharing one `WritePolicy`: [`check_reuse`] before,
/// `begin_resolved_write` after.
pub enum ReuseVerdict {
    /// A valid stored artifact exists; skip the work. Carries the leaf dir —
    /// a caller that also needs the stored record reads it from there (the
    /// verdict deliberately does not carry one: the only caller that wants it
    /// asks at a different point in the run than this decision is made).
    CacheHit { dir: PathBuf },
    /// Nothing usable is stored (or `Force` displaced it): do the work, then
    /// write through `begin_resolved_write`, which derives the same leaf path
    /// from the same identity — so the verdict carries none.
    MustRun,
}

/// The one cache-hit decision. Ask this instead of hand-rolling a `lookup`,
/// an `exists()` check, or a bespoke cache scan — each of those grew its own
/// bugs (survey's `landscape.tsv.exists()` had no identity gate at all).
///
/// Under [`WritePolicy::Force`] a matching `Completed` leaf is quarantined
/// first, so the caller always gets `MustRun` and the recompute lands on a
/// clean leaf rather than colliding with the incumbent at commit time.
pub fn check_reuse(
    store: &FsCasStore,
    root: &Path,
    resolved: &ResolvedArtifact,
    policy: WritePolicy,
) -> Result<ReuseVerdict, CasError> {
    let dir = store_path(root, resolved.kind, &resolved.levels);
    let id = runid::LeafIdentity::new(resolved.run_id);
    if policy == WritePolicy::Force {
        store.displace_completed(&dir, &id)?;
        return Ok(ReuseVerdict::MustRun);
    }
    match store.lookup(&dir, &id) {
        runid::Lookup::Hit(_) => Ok(ReuseVerdict::CacheHit { dir }),
        _ => Ok(ReuseVerdict::MustRun),
    }
}

/// The outcome of [`begin_resolved_write`].
pub enum ResolvedWrite {
    /// Atomic: already committed; the destination path.
    Committed(PathBuf),
    /// Streaming: an open claim the caller writes into, then finalizes.
    Streaming(ResolvedClaim),
}

/// An open streaming claim plus the record to finalize with. The caller streams
/// output files (via [`dir`](Self::dir) / [`write`](Self::write)) then calls
/// [`finalize`](Self::finalize) with the post-run display inputs.
pub struct ResolvedClaim {
    claim: StreamClaim,
    record: RunRecord,
}

impl ResolvedClaim {
    pub fn dir(&self) -> &Path {
        self.claim.dir()
    }

    /// Stream one output file into the claimed leaf, fsync'd. Prefer this to
    /// writing under [`dir`](Self::dir) with `std::fs::write`: that path skips
    /// the fsync and (at the sites that discarded its error) could finalize a
    /// `Completed` leaf whose payload file was never written.
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<(), CasError> {
        self.claim.write(name, bytes)
    }

    /// Attach the tabular outputs' column schema, keyed by leaf-relative path,
    /// before `finalize`. Recorded in `run.json`, never hashed — identity was
    /// fixed at claim time, so this cannot re-key the run.
    pub fn set_output_schema(
        &mut self,
        schema: std::collections::BTreeMap<String, runid::record::TableSchema>,
    ) {
        self.record.output_schema = schema;
    }

    /// Commit Running → Completed, writing the post-run `display_inputs` into
    /// `run.json.inputs`. (`StreamClaim::finalize` flips status and builds the
    /// exact-set manifest from the streamed files.)
    pub fn finalize(mut self, display_inputs: serde_json::Value) -> Result<PathBuf, CasError> {
        self.record.inputs = display_inputs;
        self.claim.finalize(self.record)
    }
}

/// Assemble the `RunRecord` for `resolved` + `meta` at `status`. The single
/// record-construction site — every CAS write routes through here.
fn build_record(resolved: &ResolvedArtifact, meta: &RecordMeta, status: RunStatus) -> RunRecord {
    RunRecord {
        format_version: FORMAT_VERSION,
        kind: resolved.kind,
        run_id: resolved.run_id,
        hash_version: HASH_VERSION,
        ir_version: meta.ir_version.clone(),
        engine_version: meta.engine_version.clone(),
        levels: resolved.levels.clone(),
        deps: meta.deps.clone(),
        status,
        artifacts: Default::default(),
        output_schema: meta.output_schema.clone(),
        children: meta.children.clone(),
        inputs: resolved.display_inputs.clone(),
        provenance: Provenance {
            argv: std::env::args().collect(),
            label: meta.label.clone(),
            created_at: Some(crate::cas::iso8601_utc(std::time::SystemTime::now())),
            camdl_version: Some(meta.engine_version.clone()),
            source_paths: meta.source_paths.clone(),
            ..Default::default()
        },
    }
}

/// The one legal CAS writer. Derives the leaf path from `resolved`'s identity,
/// builds the record in one place, and dispatches to the chosen store mode.
pub fn begin_resolved_write(
    store: &FsCasStore,
    root: &Path,
    resolved: &ResolvedArtifact,
    meta: &RecordMeta,
    mode: WriteMode,
) -> Result<ResolvedWrite, CasError> {
    let dir = store_path(root, resolved.kind, &resolved.levels);
    match mode {
        WriteMode::Atomic(artifacts) => {
            let record = build_record(resolved, meta, RunStatus::Completed);
            let dest = store.commit_atomic(&dir, record, artifacts)?;
            Ok(ResolvedWrite::Committed(dest))
        }
        WriteMode::Streaming => {
            let record = build_record(resolved, meta, RunStatus::Running);
            let claim = store.claim_streaming(&dir, record.clone())?;
            Ok(ResolvedWrite::Streaming(ResolvedClaim { claim, record }))
        }
    }
}

#[cfg(test)]
mod tests;
