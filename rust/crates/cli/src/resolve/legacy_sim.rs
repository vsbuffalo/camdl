//! The trajectory identity of the old store, kept only so that the run-id
//! pins of `resolve::tests` and the golden run-id table hold until their sim
//! rows are deleted in a commit of their own. No command writes it.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use runid::inputs::{DataDigest, InterventionId, ParamId, ResolvedParams, ResolvedScenario, Seed};
use runid::{run_id, ArtifactKind, ContentHash, LevelId};

pub(crate) use runid::inputs::{
    Backend, CalendarMode, EngineIdentity, InitStateDigest, ModelDigest, ResolvedOutputSchedule,
    SimConfig,
};
pub(crate) use runid::{FiniteF64, ResolveError};

use crate::args::types::ForwardBackend;
use crate::util::OutputColumns;

/// The model the engine starts every cell from, and its identity level.
pub(crate) struct ResolvedModel {
    /// The base model as `util::load_run_model` loads it for the engine:
    /// observation anchors substituted, `--integrator` applied, no parameter
    /// values or scenario applied (those are per cell).
    pub ir: ir::Model,
    /// The `model` level, computed once from `ir`. The compile settings
    /// (`util::CompileSettings`) change the IR camdlc emits, so they reach
    /// the identity here rather than as a configuration field.
    pub digest: ModelDigest,
    /// The engine identity folded into `digest` (the build and the engine
    /// switches simulate keys), also used by the ensemble's model level and
    /// recorded on every leaf.
    pub engine: EngineIdentity,
    /// The readable path label of the `model` level (provenance).
    pub stem: String,
    /// The model file as named on the command line (provenance: the leaf's
    /// `source_paths`, and the reuse warning for a renamed file).
    pub path: String,
}

impl ResolvedModel {
    pub fn new(ir: ir::Model, ir_version: &str, engine: EngineIdentity, path: &str) -> Self {
        let digest = ModelDigest::from_model(&ir, ir_version.to_string(), engine.clone());
        ResolvedModel {
            ir,
            digest,
            engine,
            stem: crate::hashing::path_stem_slug(path).unwrap_or_else(|| "model".to_string()),
            path: path.to_string(),
        }
    }
}

/// The run-wide part of the `config` level: every resolved setting that is the
/// same for every cell of the invocation.
#[derive(Debug, Clone)]
pub(crate) struct RunConfig {
    pub backend: Backend,
    pub dt: FiniteF64,
    /// The output schedule of the base model (`--output-every` is lowered into
    /// the model before it is loaded, so it is here and in the model level).
    pub output: ResolvedOutputSchedule,
    pub calendar: CalendarMode,
    /// `--allow-degenerate-rates`: a collapsed rate is zero instead of an error.
    pub allow_degenerate_rates: bool,
    /// `--allow-forcing-extrapolation`: hold a forcing's end value outside its
    /// range instead of refusing the run (D44).
    pub allow_forcing_extrapolation: bool,
    /// `--no-flows` / `--columns`: which trajectory columns the leaf stores.
    pub view: OutputColumns,
}

/// The per-cell part of the `config` level.
pub(crate) struct CellWindow {
    pub t_start: FiniteF64,
    pub t_end: FiniteF64,
    /// The forecast-origin state the cell restored (`--init-state`).
    pub init_state: Option<InitStateDigest>,
}

impl CellWindow {
    pub fn new(t_start: f64, t_end: f64, init_state: Option<InitStateDigest>) -> Result<Self, ResolveError> {
        Ok(CellWindow { t_start: finite(t_start)?, t_end: finite(t_end)?, init_state })
    }
}

impl RunConfig {
    pub fn new(
        backend: ForwardBackend,
        dt: f64,
        output: &ir::model::OutputSchedule,
        allow_degenerate_rates: bool,
        allow_forcing_extrapolation: bool,
        view: OutputColumns,
    ) -> Result<Self, ResolveError> {
        Ok(RunConfig {
            backend: backend_level(backend),
            dt: finite(dt)?,
            output: resolve_output(output)?,
            // Placeholder until the calendar-time work gives the resolver a
            // concrete mode (M2 decision: minimal, provisional).
            calendar: CalendarMode::Numeric,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            view,
        })
    }

    /// The `config` level of one cell: this run-wide configuration and the
    /// cell's window. Exhaustive on both sides (see the module doc).
    pub fn cell_config(&self, window: CellWindow) -> SimConfig {
        let RunConfig {
            backend,
            dt,
            output,
            calendar,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            view: OutputColumns { no_flows, allow },
        } = self.clone();
        let CellWindow { t_start, t_end, init_state } = window;
        SimConfig {
            backend,
            dt,
            t_start,
            t_end,
            output,
            calendar,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            no_flows,
            columns: allow,
            init_state,
        }
    }
}

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

/// Resolve a trajectory leaf's identity: the five factored levels
/// (model/config/params/scenario/seed, in path order), and the `run_id` derived
/// from them.
pub fn resolve_trajectory(ctx: &TrajectoryCtx) -> Result<ResolvedTrajectory, ResolveError> {
    let config = &ctx.config;
    let config_label = format!(
        "{}-dt{}",
        forward_backend(config.backend).as_str(),
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

/// The resolved `runid` backend of a `--backend` value.
pub(crate) fn backend_level(b: ForwardBackend) -> Backend {
    match b {
        ForwardBackend::Gillespie => Backend::Gillespie,
        ForwardBackend::ChainBinomial => Backend::ChainBinomial,
        ForwardBackend::Ode => Backend::Ode,
    }
}

/// The engine backend of a resolved `runid` backend; the inverse of [`backend_level`].
pub(crate) fn forward_backend(b: Backend) -> ForwardBackend {
    match b {
        Backend::Gillespie => ForwardBackend::Gillespie,
        Backend::ChainBinomial => ForwardBackend::ChainBinomial,
        Backend::Ode => ForwardBackend::Ode,
    }
}

pub(crate) fn finite(x: f64) -> Result<FiniteF64, ResolveError> {
    FiniteF64::new(x).map_err(ResolveError::from)
}

fn resolve_output(sched: &ir::model::OutputSchedule) -> Result<ResolvedOutputSchedule, ResolveError> {
    use ir::model::OutputSchedule as O;
    Ok(match sched {
        O::Regular(r) => ResolvedOutputSchedule::Regular { start: finite(r.start)?, step: finite(r.step)? },
        O::AtTimes(ts) => {
            ResolvedOutputSchedule::AtTimes(ts.iter().map(|t| finite(*t)).collect::<Result<_, _>>()?)
        }
    })
}

pub(crate) fn view(no_flows: bool, columns: &[&str]) -> OutputColumns {
    OutputColumns { no_flows, allow: columns.iter().map(|c| c.to_string()).collect() }
}


/// The atomic old-store write the fail-closed test commits a leaf with.
pub(crate) fn begin_resolved_write_atomic(
    store: &runid::FsCasStore,
    root: &std::path::Path,
    resolved: &super::ResolvedArtifact,
    meta: &super::RecordMeta,
    artifacts: runid::Artifacts,
) -> Result<std::path::PathBuf, runid::CasError> {
    let dir = runid::find_leaf(root, resolved.kind, &resolved.levels)
        .unwrap_or_else(|| runid::store_path(root, resolved.kind, &resolved.levels));
    let record = super::build_record(resolved, meta, runid::RunStatus::Completed);
    store.commit_atomic(&dir, record, artifacts)
}
