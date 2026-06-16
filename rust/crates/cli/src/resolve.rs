//! The `Resolve` bridge for `simulate`/`batch`: map already-resolved CLI
//! inputs into the `runid` identity types — `TrajectoryInput`, the factored
//! `LevelId`s, and the leaf `run_id`.
//!
//! Every field that enters a hash is a *resolved value*: param/scenario maps
//! are concrete name→value, the seed is the resolved `process_seed` (never
//! the base `--seed`), and a non-finite resolved float is a `ResolveError`
//! surfaced before any hashing. Presentation-only IR fields
//! (`output.format`, `simulation.time_semantics`) are normalized out of the
//! hashed model so `--format`/time rendering stay inert (provenance).
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
    pub backend: crate::args::types::Backend,
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
}

/// Map a CLI backend to the resolved `runid` backend.
fn backend(b: crate::args::types::Backend) -> Backend {
    use crate::args::types::Backend as B;
    match b {
        B::Gillespie => Backend::Gillespie,
        B::ChainBinomial => Backend::ChainBinomial,
        B::Ode => Backend::Ode,
    }
}

/// Strip pure-presentation fields from a model before hashing, so they stay
/// inert. `output.format` (parquet/tsv) and `simulation.time_semantics` never
/// affect computed values — they render *views* of the canonical artifact at
/// `cat` time.
fn normalize_for_hash(model: &ir::Model) -> ir::Model {
    let mut m = model.clone();
    m.output.format = String::new();
    m.simulation.time_semantics = String::new();
    m
}

/// The M2-interim whole-IR model digest (presentation-normalized) + versions.
pub fn model_digest(model: &ir::Model, ir_version: &str, engine_version: &str) -> ModelDigest {
    let normalized = normalize_for_hash(model);
    ModelDigest::from_model(
        &normalized,
        ir_version.to_string(),
        EngineVersion(engine_version.to_string()),
    )
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
            end: finite(r.end)?,
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
fn config_label(b: crate::args::types::Backend, dt: f64) -> String {
    format!("{}-dt{}", b.as_str(), dt)
}

/// Build a `LevelId` from a name, label, and the level's content hash.
fn level(name: &str, label: &str, hash: ContentHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash, schema_version: 1 }
}

/// Resolve a trajectory leaf's identity: its `TrajectoryInput`, the five
/// factored levels (model/config/params/scenario/seed, in path order), and
/// the `run_id` derived from them.
pub fn resolve_trajectory(ctx: &TrajectoryCtx) -> Result<ResolvedTrajectory, ResolveError> {
    let model = model_digest(ctx.model, ctx.ir_version, ctx.engine_version);
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

#[cfg(test)]
mod tests;
