//! The `Resolve` bridge for `simulate`: map already-resolved CLI
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
    DataDigest, InterventionId, ModelDigest, ParamId, ResolvedParams, ResolvedScenario, Seed,
    SimConfig,
};
use runid::{run_id, ArtifactKind, ContentHash, FiniteF64, LevelId, ResolveError};

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
    /// The readable summary written to the leaf's `run.json.inputs` (never
    /// hashed): the parameter values, scenario and seed the levels were
    /// hashed from, built from the same [`TrajectoryCtx`] in the same call, so
    /// what `camdl show` prints is what the `run_id` names (gh#583 item I).
    pub display_inputs: serde_json::Value,
}

/// Inputs to [`resolve_trajectory`], all already resolved by the caller. The
/// labels are provenance (readable path segments); identity rides in the
/// hashes.
///
/// The model and config levels arrive whole: `model` is the invocation's
/// [`ModelDigest`] (`sim_config::ResolvedModel::digest`) and `config` the cell's
/// [`SimConfig`] (`sim_config::RunConfig::cell_config`), so this bridge copies no
/// configuration field by hand.
pub struct TrajectoryCtx<'a> {
    pub model: ModelDigest,
    pub model_stem: &'a str,
    pub config: SimConfig,
    /// The cell's resolved parameter values (name → value) — every tier
    /// applied, as `util::resolve_run_parameters` hands them to the engine —
    /// plus the table-file digests.
    pub params: &'a HashMap<String, f64>,
    pub table_digests: Vec<DataDigest>,
    /// Resolved scenario delta — the *effective* one, compose-walked
    /// (`params_resolver::resolve_preset_delta` for a preset).
    pub enable: &'a [String],
    pub disable: &'a [String],
    pub scen_params: &'a HashMap<String, f64>,
    /// `scale` factors in application order (gh#573).
    pub scen_scale: &'a [(String, f64)],
    /// Readable labels (provenance).
    pub param_label: &'a str,
    pub scenario_label: &'a str,
    /// The base `--seed` (readable label) and the resolved `process_seed`
    /// (the actual trajectory driver, and the only seed value hashed).
    pub base_seed: u64,
    pub process_seed: u64,
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
/// `ENGINE_ID` build fingerprint). The cross-check asks "is this the same *model*?"
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

use crate::sim_config::finite;

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
    scale: &[(String, f64)],
) -> Result<ResolvedScenario, ResolveError> {
    let enabled = enable.iter().cloned().map(InterventionId).collect();
    let disabled = disable.iter().cloned().map(InterventionId).collect();
    let mut p: BTreeMap<ParamId, FiniteF64> = BTreeMap::new();
    for (k, v) in patch {
        p.insert(ParamId(k.clone()), finite(*v)?);
    }
    // Group by parameter, keeping each parameter's factors in application
    // order: factors on different parameters commute, factors on the same one
    // are applied in sequence (see `ResolvedScenario::scale`).
    let mut sc: BTreeMap<ParamId, Vec<FiniteF64>> = BTreeMap::new();
    for (k, f) in scale {
        sc.entry(ParamId(k.clone())).or_default().push(finite(*f)?);
    }
    Ok(ResolvedScenario { enabled, disabled, patch: p, scale: sc })
}

use crate::fit::cas::{level, structural_level_hash};

/// Resolve a trajectory leaf's identity: its `TrajectoryInput`, the five
/// factored levels (model/config/params/scenario/seed, in path order), and
/// the `run_id` derived from them.
pub fn resolve_trajectory(ctx: &TrajectoryCtx) -> Result<ResolvedTrajectory, ResolveError> {
    let config = &ctx.config;
    let config_label = format!(
        "{}-dt{}",
        crate::sim_config::forward_backend(config.backend).as_str(),
        config.dt.get()
    );
    let params = resolve_params(ctx.params, ctx.table_digests.clone())?;
    let scenario = resolve_scenario(ctx.enable, ctx.disable, ctx.scen_params, ctx.scen_scale)?;
    let seed = Seed { process_seed: ctx.process_seed, base_seed: ctx.base_seed };

    // Each level hashes a disjoint slice of the input; the union is the whole
    // resolved input set. The label is provenance, the hash is identity.
    let levels = vec![
        level("model", ctx.model_stem, structural_level_hash(&ctx.model)),
        level("config", &config_label, structural_level_hash(config)),
        level("params", ctx.param_label, structural_level_hash(&params)),
        level("scenario", ctx.scenario_label, structural_level_hash(&scenario)),
        level("seed", &format!("seed_{}", ctx.base_seed), structural_level_hash(&seed)),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::Sim, &level_hashes);

    let display_inputs = serde_json::json!({
        "params": ctx.params.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect::<BTreeMap<String, f64>>(),
        "scenario": ctx.scenario_label,
        "process_seed": ctx.process_seed,
    });

    Ok(ResolvedTrajectory { levels, run_id: rid, display_inputs })
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
    /// The engine the run's identity folds: `run.json`'s `engine_version` is
    /// its build, `engine_switches` its switches.
    pub engine: runid::inputs::EngineIdentity,
    pub deps: Vec<runid::inputs::ArtifactRef>,
    pub children: BTreeMap<String, Vec<ContentHash>>,
    pub source_paths: Vec<String>,
    pub label: Option<String>,
    /// Column schema for the leaf's tabular outputs, attached to the record at
    /// build time — used by atomic-commit paths (e.g. `sim`, whose files are
    /// known in memory before commit); the streaming fit path sets it at
    /// finalize instead.
    pub output_schema: BTreeMap<String, runid::record::TableSchema>,
    /// The model's calendar axis, recorded as `run.json`'s `calendar` block.
    /// Every record carries one, so it is a constructor argument (D59).
    pub calendar: ir::caltime::CalendarAxis,
    /// The command's forcing-extrapolation policy: under the override, the
    /// record says so and names the forcings the command held. Every command
    /// that writes has one, so it is a constructor argument.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
}

impl RecordMeta {
    /// The common case: engine = this binary with no engine switch, no
    /// deps/children, one source path. A command whose identity keys the
    /// engine switches sets [`Self::engine`] to the identity it keyed.
    pub fn new(
        ir_version: impl Into<String>,
        model_path: impl Into<String>,
        label: Option<String>,
        calendar: ir::caltime::CalendarAxis,
        forcing: &crate::forcing_extrapolation::ForcingPolicy,
    ) -> Self {
        RecordMeta {
            ir_version: ir_version.into(),
            engine: runid::inputs::EngineIdentity::of_build(crate::version::ENGINE_ID),
            deps: Vec::new(),
            children: BTreeMap::new(),
            source_paths: vec![model_path.into()],
            label,
            output_schema: BTreeMap::new(),
            calendar,
            forcing: forcing.clone(),
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
/// decided for itself what that meant: the grid writer recomputed and had its
/// bytes discarded at commit, fit and survey hard-errored with `AlreadyCompleted`,
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
/// The leaf is found by its hashes, not its labels ([`runid::find_leaf`]):
/// a leaf's `run_id` is its level hashes alone, while its directory is
/// `{label}-{hash8}` per level, so a lookup built from the current labels
/// missed every leaf whose model file, params file or scenario preset had
/// since been renamed, and the derived arm label would have orphaned every
/// stored ad-hoc leaf the same way. A store that renamed nothing answers
/// exactly as before: the exact-label path is tried first.
///
/// Under [`WritePolicy::Force`] a matching `Completed` leaf, or one held by a
/// process on another host, is quarantined first, so the caller always gets `MustRun` and the recompute lands on a
/// clean leaf rather than colliding with the incumbent at commit time.
pub fn check_reuse(
    store: &FsCasStore,
    root: &Path,
    resolved: &ResolvedArtifact,
    policy: WritePolicy,
) -> Result<ReuseVerdict, CasError> {
    check_reuse_with(store, root, resolved, policy, crate::version::ENGINE_REUSE)
}

/// The policy a reuse decision runs under for a build whose identity is
/// `engine`: the one asked for, or [`WritePolicy::Force`] when the build's
/// fingerprint is incomplete ([`crate::version::EngineReuse::Unreusable`]).
/// Such a build's `ENGINE_ID` may stand for another build's code, so it never
/// serves a stored leaf: it computes and records every run, and the stored
/// leaf it would have read is quarantined, never deleted. The one
/// fail-closed rule; every reuse decision ([`check_reuse`],
/// [`crate::cas_sink::CasSink`]'s cell check, `profile`'s point scan) asks it.
pub(crate) fn engine_write_policy(policy: WritePolicy, engine: crate::version::EngineReuse) -> WritePolicy {
    match engine {
        crate::version::EngineReuse::Reusable => policy,
        crate::version::EngineReuse::Unreusable { missing } => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                crate::out::detached().say(format_args!(
                    "warning: this camdl build could not determine its engine identity ({missing}); \
                     it computes every run and never reuses a stored one"
                ));
            });
            WritePolicy::Force
        }
    }
}

/// [`check_reuse`] for a build whose identity is `engine`.
pub(crate) fn check_reuse_with(
    store: &FsCasStore,
    root: &Path,
    resolved: &ResolvedArtifact,
    policy: WritePolicy,
    engine: crate::version::EngineReuse,
) -> Result<ReuseVerdict, CasError> {
    let policy = engine_write_policy(policy, engine);
    let found = runid::find_leaf(root, resolved.kind, &resolved.levels);
    if policy == WritePolicy::Force {
        // A completed leaf is found by its hashes; a leaf held from another
        // host is not complete, so it is looked for where the claim will
        // write (`begin_resolved_write`).
        let dir = found.clone()
            .unwrap_or_else(|| store_path(root, resolved.kind, &resolved.levels));
        store.displace(&dir, &runid::LeafIdentity::new(resolved.run_id))?;
        return Ok(ReuseVerdict::MustRun);
    }
    Ok(match found {
        Some(dir) => ReuseVerdict::CacheHit { dir },
        None => ReuseVerdict::MustRun,
    })
}

/// Warn when a leaf found by its hashes ([`runid::find_leaf`]) was computed from
/// a model file of another name than `model_path`.
///
/// A model's file name is a label, not identity, so a byte-identical model
/// under another name has the same run identity and is served the leaf the
/// other file stored — under that file's directory label, with that file in its
/// `provenance.source_paths`. The answer is right; the warning says whose file
/// it was computed from, so a renamed or copied model is not mistaken for a
/// fresh run of the new name. Printed once per (stored, current) file-name pair
/// per process, however many leaves the run reuses. Silent when the leaf
/// records no source path or the same file name (a model moved between
/// directories keeps its label).
pub(crate) fn note_leaf_from_another_model_file(leaf: &Path, model_path: &str) {
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<std::collections::BTreeSet<(String, String)>>> = OnceLock::new();
    let file_name = |p: &str| {
        Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned())
    };
    // A record that cannot be read says nothing about its model file; the
    // reuse itself is decided by the store's lookup, not here.
    let stored = runid::read_leaf(leaf)
        .ok()
        .flatten()
        .and_then(|l| l.record.provenance.source_paths.into_iter().next());
    let (Some(stored), Some(current)) = (stored.as_deref().and_then(file_name), file_name(model_path))
    else {
        return;
    };
    if stored == current {
        return;
    }
    let mut warned = WARNED.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if warned.insert((stored.clone(), current.clone())) {
        crate::out::detached().say(format_args!(
            "warning: reusing stored runs computed from `{stored}`: `{current}` is the same \
             model under another file name (the file name is a label, not part of the run \
             identity). The stored runs keep `{stored}`'s directory names; first: {}",
            leaf.display()
        ));
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
    /// The writing command's policy ([`RecordMeta::forcing`]): the finalized
    /// record names what it held by then.
    forcing: crate::forcing_extrapolation::ForcingPolicy,
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

    /// The writing command's forcing-extrapolation policy.
    pub fn forcing(&self) -> &crate::forcing_extrapolation::ForcingPolicy {
        &self.forcing
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

    /// Stamp the run's label as given now (`provenance.label_set_at`), before
    /// `finalize`: a fit run that completes under `--label L` becomes `@L`.
    /// Recorded, not hashed.
    pub fn set_label_set_at(&mut self, stamp: String) {
        self.record.provenance.label_set_at = Some(stamp);
    }

    /// Commit Running → Completed, writing the post-run `display_inputs` into
    /// `run.json.inputs`. (`StreamClaim::finalize` flips status and builds the
    /// exact-set manifest from the streamed files.)
    pub fn finalize(mut self, mut display_inputs: serde_json::Value) -> Result<PathBuf, CasError> {
        self.forcing.record_in(&mut display_inputs);
        self.record.inputs = display_inputs;
        self.claim.finalize(self.record)
    }

    /// Commit Running → Failed with `failure`, the error that stopped the
    /// run, recorded in `run.json`, and release the lock
    /// (`StreamClaim::fail`). For a run about to exit on an error.
    pub fn fail(self, failure: runid::Failure) -> Result<(), CasError> {
        self.claim.fail(failure)
    }
}

/// Assemble the `RunRecord` for `resolved` + `meta` at `status`. The single
/// record-construction site — every CAS write routes through here.
fn build_record(resolved: &ResolvedArtifact, meta: &RecordMeta, status: RunStatus) -> RunRecord {
    // Display-only, never hashed: a run made under
    // `--allow-forcing-extrapolation` says so, and names what it held.
    let mut inputs = resolved.display_inputs.clone();
    meta.forcing.record_in(&mut inputs);
    RunRecord {
        format_version: FORMAT_VERSION,
        kind: resolved.kind,
        run_id: resolved.run_id,
        hash_version: HASH_VERSION,
        store_layout: runid::STORE_LAYOUT,
        ir_version: meta.ir_version.clone(),
        engine_version: meta.engine.build.clone(),
        engine_switches: meta.engine.switches.clone(),
        levels: resolved.levels.clone(),
        deps: meta.deps.clone(),
        status,
        failure: None,
        artifacts: Default::default(),
        output_schema: meta.output_schema.clone(),
        children: meta.children.clone(),
        calendar: meta.calendar,
        inputs,
        provenance: Provenance {
            argv: std::env::args().collect(),
            label: meta.label.clone(),
            created_at: Some(crate::cas::iso8601_utc(std::time::SystemTime::now())),
            camdl_version: Some(crate::version::VERSION_SHORT.to_string()),
            source_paths: meta.source_paths.clone(),
            // An experimental input is listed on every record it reaches
            // (proposal 2026-11-17, part E); empty, and absent, otherwise.
            experimental: crate::env::get().experimental_inputs(),
            ..Default::default()
        },
    }
}

/// The one legal CAS writer. Derives the leaf path from `resolved`'s identity,
/// builds the record in one place, and dispatches to the chosen store mode.
///
/// A new leaf lands at `store_path` under the current labels. A leaf the store
/// already holds for this identity is written *to*, wherever its labels put
/// it ([`runid::find_leaf`]), so the store's own same-identity rules apply —
/// an atomic commit dedups against it (and refuses divergent bytes), a
/// streaming claim reports it `AlreadyCompleted` — instead of a second copy of
/// one identity appearing under a renamed label.
pub fn begin_resolved_write(
    store: &FsCasStore,
    root: &Path,
    resolved: &ResolvedArtifact,
    meta: &RecordMeta,
    mode: WriteMode,
) -> Result<ResolvedWrite, CasError> {
    let dir = runid::find_leaf(root, resolved.kind, &resolved.levels)
        .unwrap_or_else(|| store_path(root, resolved.kind, &resolved.levels));
    match mode {
        WriteMode::Atomic(artifacts) => {
            let record = build_record(resolved, meta, RunStatus::Completed);
            let dest = store.commit_atomic(&dir, record, artifacts)?;
            Ok(ResolvedWrite::Committed(dest))
        }
        WriteMode::Streaming => {
            let record = build_record(resolved, meta, RunStatus::Running);
            let claim = store.claim_streaming(&dir, record.clone())?;
            Ok(ResolvedWrite::Streaming(ResolvedClaim { claim, record, forcing: meta.forcing.clone() }))
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::tiny_model;
