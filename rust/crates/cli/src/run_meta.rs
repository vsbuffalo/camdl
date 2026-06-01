//! Unified run-metadata ADT for the `output/` tree.
//!
//! One `Run` type with a `kind: RunKind` discriminator covers every
//! result camdl produces — simulate runs, top-level fits, and
//! per-stage fits — under one schema. Replaced the parallel
//! `cas::RunMeta` and `fit::provenance::StageProvenance` structs that
//! had ~80 % field overlap with drifting names (version vs
//! camdl_version, created_at vs timestamp, etc.); both are now gone.
//!
//! See `docs/dev/proposals/2026-04-19-unified-output-tree.md` for the
//! original design.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata written to `run.json` at the top of every content-hashed
/// run directory. Shared fields live at the top level; kind-specific
/// fields are inside `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Content hash for this run, full 64-char hex. Scope depends on
    /// `kind`:
    ///   - `Simulate`: hash of (sim_hash, scen_hash, seed).
    ///   - `Fit`: seed-independent content hash of
    ///     (fit.toml, model IR, data files).
    ///   - `FitStage`: stage-scope config hash from `fit_stage_hash`
    ///     (includes stage algorithm + seed).
    /// The 8-char prefix appears in the filesystem path.
    pub hash: String,
    /// camdl version at write time (e.g. "0.1.0+abc1234").
    pub version: String,
    /// ISO 8601 UTC timestamp at completion.
    pub created_at: String,
    /// Original argv that produced this run — `camdl show <hash>`
    /// prints it back for reproducibility.
    pub argv: Vec<String>,
    /// Run lifecycle state. `Running` is set at the first write
    /// (run.json appears as soon as we know the directory layout, so
    /// cancellations / crashes leave a discoverable trace);
    /// `Completed { wall_time_seconds }` replaces it at end-of-run.
    /// Replaces the prior sentinel `wall_time_seconds == 0.0`-means-
    /// running pattern.
    pub status: RunStatus,
    /// User-supplied display label. Optional — validated against
    /// `^[a-zA-Z0-9 ,._-]{1,64}$` after trim. Set at run-time via
    /// `--label` or post-hoc via `camdl label <hash> "<text>"`.
    /// Surfaced in `camdl list`, `camdl fit table`, `camdl show` to
    /// help disambiguate iterations of the same operation that share
    /// the same stem (e.g. multiple `fit_he2010-XXXXXXXX` directories
    /// with different bounds / priors). Applies uniformly to all
    /// `RunKind` variants — sims, fits, profiles, replicate-sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Kind-specific payload.
    pub kind: RunKind,
}

/// Run lifecycle state. Two states:
///
/// - `Running` — the run.json was written but the run is still
///   executing (or crashed before patching). Wall time isn't known.
/// - `Completed { wall_time_seconds }` — the runner finished and
///   patched run.json with the elapsed time.
///
/// Replaces a prior sentinel pattern (`wall_time_seconds == 0.0` ⇒
/// running). Wire format is the snake-case enum:
///
///     "status": "running"
///     "status": { "completed": { "wall_time_seconds": 42.5 } }
///
/// Test fixtures and renderers should not assume `Completed` —
/// `cmd_label`, `camdl show`, etc. handle both states.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed { wall_time_seconds: f64 },
}

impl RunStatus {
    /// `true` iff the run is still executing (or crashed before
    /// finalizing).
    pub fn is_running(&self) -> bool {
        matches!(self, RunStatus::Running)
    }

    /// Wall time in seconds when complete; `None` while running.
    pub fn wall_time_seconds(&self) -> Option<f64> {
        match self {
            RunStatus::Completed { wall_time_seconds } => Some(*wall_time_seconds),
            RunStatus::Running => None,
        }
    }
}

/// Tagged union over the three result shapes. `serde(tag = "kind")`
/// emits a `"kind": "simulate"` (etc.) field in the JSON, so
/// `camdl list` can discriminate without needing to know the directory
/// layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RunKind {
    /// One simulate invocation. The directory contains `traj.tsv` and
    /// optional `obs/<obs_hash>-<obs_seed>/` subdirectories.
    Simulate(SimulateMeta),
    /// A complete fit (potentially multi-stage). The directory
    /// contains per-stage subdirectories, each with its own
    /// stage-level `Run` whose kind is `FitStage`.
    Fit(FitMeta),
    /// One stage of a fit. The directory is a child of a `Fit` run,
    /// at `<fit_dir>/real/fit_<seed>/<stage>/` or
    /// `<fit_dir>/synthetic/ds_NN/fit_<seed>/<stage>/`.
    FitStage(FitStageMeta),
    /// A profile-likelihood scan: Cartesian product over N focal
    /// parameter axes × `n_starts` independent IF2 mini-fits per
    /// grid point. The directory contains `profile.tsv` (derived
    /// rollup) and `points/{idx:05d}/start_{k}/` subtrees, where
    /// each `start_{k}/` is itself a `FitStage` run. See
    /// docs/dev/proposals/2026-04-24-profile-cas-integration.md.
    Profile(ProfileMeta),
    /// A *group* of single-realization runs that share an inner
    /// content (everything except a varying replicate dimension)
    /// and differ only on one input — typically `seed` for
    /// stochastic-method sensitivity, `dataset_idx` for
    /// synthetic-data fits. The directory contains an aggregate
    /// `summary.tsv` and a `replicates/<key>/` subdir per child;
    /// each child has its own `run.json` of the underlying kind.
    /// See docs/dev/proposals/2026-04-28-cas-typed-runs-and-profile-stages.md.
    ReplicateSet(crate::cas::typed::ReplicateSetMeta),
    /// A likelihood-landscape diagnostic: N Latin-hypercube points
    /// across declared parameter bounds, evaluated via PF or single
    /// simulation, written to `landscape.tsv`. NOT a fitting routine
    /// — produces no MLE artifact. See
    /// docs/dev/proposals/2026-05-03-survey-subcommand.md.
    Survey(SurveyMeta),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateMeta {
    /// Model file path or name (display only — not a hash input).
    pub model: String,
    /// Full model IR hash (64 hex chars).
    pub model_hash: String,
    /// Named scenario or "baseline".
    pub scenario: String,
    /// Simulation config hash: model + base params + backend + dt + version.
    pub sim_hash: String,
    /// Scenario delta hash: enable/disable/overrides.
    pub scen_hash: String,
    pub seed: u64,
    pub backend: crate::args::types::Backend,
    pub dt: f64,
    /// Sweep-point param values (empty for single-run `--cas`).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sweep_point: HashMap<String, f64>,
    /// Full Run.hash of the fit whose `mle_params.toml` was passed to
    /// `camdl simulate --params`, when applicable. Populates a
    /// sim → fit provenance link for `camdl list` / `camdl show` to
    /// surface. See `docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_fit_hash: Option<String>,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance. Records
    /// where every parameter's value came from (model default,
    /// scenario, --fixed-file, --fixed-cli), plus the kick-from-
    /// estimate audit field and scenario-override record. Map key is
    /// the parameter name; value is the full [`ParameterProvenance`].
    /// Empty / absent on older run.json files that predate step 9.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitMeta {
    /// Model file path (display only — not a hash input).
    pub model: String,
    /// Structural model IR hash.
    pub model_hash: String,
    /// Path to the fit.toml that produced this fit.
    pub fit_toml_path: String,
    /// Hash of the fit.toml bytes — canonical-form hash from
    /// `FitConfigV2::fit_content_hash`.
    pub fit_toml_hash: String,
    /// Per-stream data file hashes.
    pub data_hashes: HashMap<String, String>,
    /// Names of parameters declared in `[estimate]`.
    pub estimated: Vec<String>,
    /// Resolved fixed params (name → numeric value).
    pub fixed: HashMap<String, f64>,
    /// Stage names declared in fit.toml, in execution order.
    pub stages_declared: Vec<String>,
    /// IC-free inference flag (see 2026-04-18 proposal).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ic_free: bool,
    /// gh#75: per-parameter prior-resolution audit. For every
    /// estimated parameter, names where the prior came from in the
    /// three-tier precedence chain — `fit_toml`, `model_ir`, or
    /// `flat_explicit`. `flat_fallback` is NOT a valid value here:
    /// `validate_priors_present` rejects implicit fallback to flat
    /// before the fit starts, so a fit dir that exists on disk
    /// always has a resolved (and explicitly accountable) prior
    /// for each parameter. Empty when the fit has no Bayesian
    /// stage (IF2-only fits don't consume priors); the field is
    /// `default = []` so older run.json files round-trip unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolved_priors: Vec<ResolvedPriorEntry>,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance — see
    /// [`SimulateMeta::parameters_provenance`] for the shape.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
}

/// Inference algorithm tag — discriminator enum naming the algorithm
/// independent of the simulation backend. Stored on `FitStageMeta`
/// alongside `Backend` to record the (algorithm, backend) pair the
/// stage ran. Wire format matches the lowercased / kebab-cased name
/// the user writes in fit.toml (`algorithm = "if2"`, `algorithm =
/// "nl-sbplx"`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MethodKind {
    #[serde(rename = "if2")]      If2,
    #[serde(rename = "pgas")]     Pgas,
    #[serde(rename = "pmmh")]     Pmmh,
    #[serde(rename = "pfilter")]  Pfilter,
    #[serde(rename = "nl-sbplx")] NlSbplx,
    #[serde(rename = "nl-bobyqa")] NlBobyqa,
}

impl MethodKind {
    /// Wire-format string. Matches the `algorithm = "..."` value in fit.toml
    /// and the `FitStageMeta.algorithm` serialized form.
    pub fn as_str(self) -> &'static str {
        match self {
            MethodKind::If2      => "if2",
            MethodKind::Pgas     => "pgas",
            MethodKind::Pmmh     => "pmmh",
            MethodKind::Pfilter  => "pfilter",
            MethodKind::NlSbplx  => "nl-sbplx",
            MethodKind::NlBobyqa => "nl-bobyqa",
        }
    }
}

impl std::fmt::Display for MethodKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Simulation backend the stage ran on. The (algorithm, backend) pair
/// is constrained by `methods::METHODS`; PF-based algorithms require
/// `chain_binomial`, deterministic-likelihood algorithms require `ode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    ChainBinomial,
    Ode,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::ChainBinomial => "chain_binomial",
            Backend::Ode           => "ode",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitStageMeta {
    /// Hash of the parent fit (matches the `hash` on the enclosing
    /// `Run` of kind `Fit`). Enables walking from a stage back to its
    /// parent without relying on directory-layout inference.
    pub fit_hash: String,
    /// Stage name within the fit (e.g. "scout", "refine").
    pub stage: String,
    /// Stage algorithm discriminator. Wire format: `"if2" | "pgas" |
    /// "pmmh" | "pfilter" | "nl-sbplx" | "nl-bobyqa"`.
    pub method: MethodKind,
    /// Simulation backend this stage ran on. The (algorithm, backend)
    /// pair determines what statistical object the loglik computes —
    /// `chain_binomial` for stochastic process kernel, `ode` for the
    /// deterministic skeleton. Validated against the `methods::METHODS`
    /// registry at config-load time.
    pub backend: Backend,
    // NB: the stage's own content hash lives in the enclosing
    // `Run.hash` field (a FitStage run hashes exactly its stage-scope
    // inputs). Previously FitStageMeta carried a duplicate
    // `stage_hash: String` — removed to collapse the two-source-of-
    // truth smell.
    pub seed: u64,
    pub n_chains: usize,
    /// Stage-specific algorithm settings (chains, particles, cooling,
    /// etc.). A `serde_json::Value` keeps the shape open — each method
    /// (if2/pgas/pmmh/pfilter) has a different parameter set, and the
    /// human-readable record doesn't need a typed schema.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub algorithm: serde_json::Value,
    /// Best loglik across chains; `None` if the stage didn't compute
    /// one (e.g. a pure-diagnostic pass).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_loglik: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_chain: Option<usize>,
    /// Reference to a parent stage this one started from, if any
    /// (e.g. refine → scout). Absent when the stage has no predecessor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starts_from: Option<StartsFromRef>,
    /// Path to the upstream fit dir this fit was derived from
    /// (`camdl fit derive` workflows). Free-form string — the consumer
    /// treats this as a display hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
    /// Parent profile hash, when this FitStage is a grid-point × start
    /// child of a `RunKind::Profile`. Absent for standalone fit stages.
    /// Optional to preserve round-trip compatibility with existing
    /// fit-stage run.json files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_profile_hash: Option<String>,
    /// Grid-point index within the parent profile (flat index over
    /// the Cartesian product of focal axes). Set iff
    /// `parent_profile_hash` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_point_idx: Option<usize>,
    /// Start index within this grid point (0..n_starts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_start_idx: Option<usize>,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance — see
    /// [`SimulateMeta::parameters_provenance`] for the shape.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
    /// gh#83/gh#85 step 9: per-chain init provenance. Records the
    /// init method and the per-chain starting value + source for
    /// every estimated parameter. Absent on stages that don't draw
    /// per-chain starts (single-chain runs, refine-after-scout, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_provenance: Option<InitProvenance>,
}

/// Metadata for a `RunKind::Profile` run. The shape mirrors pomp's
/// and pfilter's convention of fanning out mini-fits over a grid;
/// every child start is a `FitStage` carrying its own seed and MLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMeta {
    /// Model file path (display only).
    pub model: String,
    /// Full model IR hash.
    pub model_hash: String,
    /// Ordered focal params. Order determines column order in the
    /// rollup TSV and is part of the profile-level hash.
    pub focal_params: Vec<String>,
    /// One axis per focal param, each with an explicit value list
    /// (mirroring the `--sweep NAME=V1,V2,...` CLI surface).
    pub grid: Vec<GridAxis>,
    /// Independent IF2 starts per grid point.
    pub n_starts: usize,
    /// Hash of the IF2 stage config (iterations, particles, cooling, dt).
    pub if2_config_hash: String,
    /// Hash of the base parameter vector (before focal-param pinning).
    pub base_params_hash: String,
    /// Seed base. Per-start seeds derive as a function of this +
    /// point_idx + start_idx.
    pub seed_base: u64,
    /// Total (grid_size × n_starts). Display only.
    pub total_jobs: usize,
    /// SHA-256 of the `--fit <toml>` file's bytes (gh#73). `None` when
    /// `--fit` was not supplied. Recorded for provenance: re-running
    /// against the same model with a different fit toml produces a
    /// different CAS hash (different priors / bounds / fixed list →
    /// different inference run), so reviewers can tell at a glance
    /// what configuration produced the artifacts on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_toml_hash: Option<String>,
    /// Per-parameter prior-resolution audit (gh#73). For every
    /// estimated parameter, names where the prior came from in the
    /// `--fit toml > model IR > flat` precedence chain. Empty for
    /// non-PMMH algorithms today (IF2 / NLopt ignore priors by
    /// design); the field is `default = []` for round-trip
    /// compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolved_priors: Vec<ResolvedPriorEntry>,
    /// Diagnostic warnings the user explicitly suppressed via
    /// `--suppress-warnings` (or fit toml's `[diagnostics] suppress`).
    /// Empty when nothing was waived. Recorded loudly so that
    /// reviewers reading `run.json` can see exactly which checks
    /// were waived for a given artifact (gh#73).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suppressed_warnings: Vec<String>,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance — see
    /// [`SimulateMeta::parameters_provenance`] for the shape.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
    /// gh#83/gh#85 step 9: per-cell init provenance for the
    /// non-focal-parameter starts. The init method's `--starts > 1`
    /// path produces one InitProvenance for the profile-level runs;
    /// per-grid-cell `FitStage` children carry their own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_provenance: Option<InitProvenance>,
}

/// One row of the per-parameter prior-resolution audit (gh#73). The
/// CLI's `profile_priors::ResolvedPrior` does not implement
/// `Serialize` directly (it carries a `Prior` enum from the `sim`
/// crate); this lightweight mirror carries the audit-relevant
/// fields — name and source tag — into `run.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedPriorEntry {
    pub param:  String,
    /// `"fit_toml" | "model_ir" | "flat_fallback"` — see
    /// `profile_priors::PriorSource`.
    pub source: String,
}

// ── Parameter-value provenance into `run.json` (gh#83/gh#85 step 9) ─────────
//
// Mirrors `params_resolver::ResolvedParameter` plus the per-chain
// `chain_starts::ChainStart` into a JSON-serializable shape. See
// `docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md`
// §"Provenance into run.json" for the design.
//
// Every subcommand that writes a `run.json` populates
// `parameters_provenance`; inference subcommands that initialize
// chains also populate `init_provenance`. Each entry's `source` field
// matches a [`ValueSource`] or [`InitSource`] variant tag, so a
// downstream reader can route on the tag without parsing the rest of
// the record.

/// One parameter's full provenance: where the resolved value came
/// from, whether the parameter is fixed or estimated, plus optional
/// audit fields for the kick-from-estimate and scenario-override
/// cases. The exact field shape the proposal specifies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterProvenance {
    /// Resolved value as written into `model.parameters[i].value`.
    pub value:  f64,
    /// [`crate::params_resolver::ValueSource::tag`] string
    /// (`"model_default" | "scenario" | "fit_toml_fixed" |
    /// "fixed_file" | "fixed_cli"`).
    pub source: String,
    /// `"fixed" | "estimated"` — matches
    /// [`crate::params_resolver::ParameterRole`].
    pub role:   String,
    /// Present iff the parameter was kicked from `[estimate]` by a
    /// user-explicit `--fixed{,-file}` assertion. The `by` field
    /// records the value source that triggered the kick (e.g.
    /// `"fixed_cli"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kicked_from_estimate: Option<KickedFromEstimate>,
    /// Present iff the active scenario set this parameter to a
    /// different value than the final winner. The proposal calls
    /// this `overrode_scenario`; the renamed `ScenarioOverrideRecord`
    /// struct lives here in `run_meta` so it can be `Serialize`d
    /// without colliding with the unsealed resolver-side
    /// `ScenarioOverride` (which would force re-exporting all the
    /// resolver-side `serde` derives).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrode_scenario: Option<ScenarioOverrideRecord>,
}

/// Audit record for [`crate::params_resolver::FixReason::KickedFromEstimate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KickedFromEstimate {
    /// The ValueSource tag of the source that kicked the parameter
    /// out (`"fixed_cli" | "fixed_file"`).
    pub by: String,
}

/// Audit record for a silent scenario override. Pairs with
/// [`crate::params_resolver::ResolverWarning::ScenarioOverridden`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioOverrideRecord {
    pub scenario:       String,
    pub scenario_value: f64,
}

/// Per-chain init provenance. The `method` field echoes the
/// [`crate::fit::init::InitMethod`] tag; each entry of `chains` is a
/// map from estimated-parameter name to its per-chain start value +
/// source. Restricted to the estimate set by construction (see
/// `ChainStart.values`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitProvenance {
    /// [`crate::fit::init::InitMethod`] `Display` tag — matches the
    /// `Display` impl so a `match` over the impl's possible outputs
    /// is exhaustive.
    pub method: String,
    /// One map per chain; key = estimated-parameter name; value =
    /// the value + per-chain source tag.
    pub chains: Vec<HashMap<String, ChainStartProvenance>>,
}

/// Per-chain per-parameter start value + provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStartProvenance {
    pub value:  f64,
    /// [`crate::fit::chain_starts::InitSource`] tag (e.g.
    /// `"prior_draw" | "posterior_row" | "mle_point" | "params_point"`).
    pub source: String,
}

impl ParameterProvenance {
    /// Build a `ParameterProvenance` entry from a
    /// [`crate::params_resolver::ResolvedParameter`].
    pub fn from_resolved(rp: &crate::params_resolver::ResolvedParameter) -> Self {
        let (role, kicked_from_estimate) = match &rp.role {
            crate::params_resolver::ParameterRole::Estimated =>
                ("estimated".to_string(), None),
            crate::params_resolver::ParameterRole::Fixed { reason } => {
                let kicked = match reason {
                    crate::params_resolver::FixReason::KickedFromEstimate { by } =>
                        Some(KickedFromEstimate { by: by.tag().to_string() }),
                    crate::params_resolver::FixReason::NotInEstimate => None,
                };
                ("fixed".to_string(), kicked)
            }
        };
        let overrode_scenario = rp.overrode_scenario.as_ref().map(|s| {
            ScenarioOverrideRecord {
                scenario:       s.scenario.clone(),
                scenario_value: s.scenario_value,
            }
        });
        ParameterProvenance {
            value:  rp.value,
            source: rp.source.tag().to_string(),
            role,
            kicked_from_estimate,
            overrode_scenario,
        }
    }
}

impl InitProvenance {
    /// Build an `InitProvenance` from a
    /// [`crate::fit::chain_starts::ChainStarts`]. Each chain's
    /// `values` HashMap maps directly to `chains[chain_id]`; the
    /// per-chain source is recorded once (the InitSource tag).
    ///
    /// Output is indexed by `ChainStart.chain_id` so the JSON's
    /// `chains[i]` corresponds to chain i regardless of storage
    /// order — important for downstream consumers that index by
    /// chain id rather than draw order.
    pub fn from_chain_starts(cs: &crate::fit::chain_starts::ChainStarts) -> Self {
        // Allocate `chains` sized to (max chain_id + 1) so an
        // out-of-order Vec<ChainStart> still produces a well-formed
        // index-by-chain_id output. Empty starts yield an empty Vec.
        let n_chains = cs.starts.iter()
            .map(|c| c.chain_id + 1).max().unwrap_or(0);
        let mut chains: Vec<HashMap<String, ChainStartProvenance>> =
            vec![HashMap::new(); n_chains];
        for chain in &cs.starts {
            let source_tag = chain.source.tag().to_string();
            let entry: HashMap<String, ChainStartProvenance> =
                chain.values.iter().map(|(name, &value)| {
                    (name.clone(), ChainStartProvenance {
                        value,
                        source: source_tag.clone(),
                    })
                }).collect();
            chains[chain.chain_id] = entry;
        }
        InitProvenance {
            method: cs.method.to_string(),
            chains,
        }
    }
}

/// One axis of a profile grid. `values` is the explicit list the
/// user supplied via `--sweep NAME=V1,V2,...`; the CLI parser
/// already splits on commas and converts to f64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridAxis {
    pub param: String,
    pub values: Vec<f64>,
}

/// How `camdl survey` evaluates the marginal log-likelihood at each
/// LHS point. The default is `Pfilter` (handles process noise via a
/// PMMH-style MC estimator); `Simulate` is an opt-in fast path for
/// known-deterministic models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lowercase")]
pub enum SurveyEvalMethod {
    /// Bootstrap particle filter, K replicates → logmeanexp combiner.
    /// Estimates p(y|θ) under the chain-binomial process; the safe
    /// default for inference-grade likelihood evaluation. Doucet et
    /// al. 2015 (Biometrika) gives the rule for trustworthy ranks:
    /// per-point loglik SE ≤ ~1.7 nats.
    Pfilter,
    /// Single deterministic simulation per point. 1-sample MC estimator
    /// of the same quantity; cheap (~10× faster than Pfilter at
    /// modest particles/replicates) but biased toward "lucky outliers"
    /// when process noise is non-trivial. Andrieu & Roberts 2009 frame
    /// the failure mode.
    Simulate,
    /// Auto-detect from the compiled model: `Pfilter` when the model
    /// requires `Capabilities::OVERDISPERSION` (i.e. it has stochastic
    /// process noise via `overdispersed()` or similar), `Simulate`
    /// otherwise. Resolved before any persistent state is written —
    /// `SurveyMeta` stores the resolved method, never `Auto`.
    Auto,
}

impl SurveyEvalMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            SurveyEvalMethod::Pfilter  => "pfilter",
            SurveyEvalMethod::Simulate => "simulate",
            SurveyEvalMethod::Auto     => "auto",
        }
    }
}

impl std::fmt::Display for SurveyEvalMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata for a `RunKind::Survey` run. Mirrors `ProfileMeta`'s
/// shape: model + content hashes + the canonical-hashed inputs that
/// determine the LHS box (`bounds`), the evaluator config
/// (`eval_method`/`eval_particles`/`eval_replicates`), the seed, and
/// any fixed / scenario context used to populate the rest of the
/// parameter vector at each LHS point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurveyMeta {
    /// Model file path (display only — not a hash input).
    pub model: String,
    /// Full SHA-256 of the IR JSON.
    pub model_hash: String,
    /// Per-stream data file content hashes. Same shape as `FitMeta` —
    /// content-only, so editing the TSV invalidates the cache (gh#39).
    pub data_hashes: HashMap<String, String>,
    /// LHS box: parameter name → (lo, hi). Resolved per the proposal's
    /// "fit.toml > model" priority. Canonical-hashed by sorting names.
    pub bounds: HashMap<String, (f64, f64)>,
    /// Number of LHS points evaluated.
    pub n_points: usize,
    /// PF or single-simulate.
    pub eval_method: SurveyEvalMethod,
    /// PF particle count (per replicate). Diagnostic when
    /// `eval_method = Simulate`.
    pub eval_particles: usize,
    /// PF replicates per LHS point (logmeanexp combiner). Diagnostic
    /// when `eval_method = Simulate` (always 1 in that case).
    pub eval_replicates: usize,
    /// LHS / PF base seed.
    pub seed: u64,
    /// Resolved fixed parameters (name → value). Canonical-hashed.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub fixed: HashMap<String, f64>,
    /// Named scenario applied before the survey. `None` for the
    /// baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    /// Names of estimated parameters in declaration order — drives
    /// the column order of `landscape.tsv`. Bounds key + this slice
    /// together fix the LHS layout deterministically.
    pub estimated: Vec<String>,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance — see
    /// [`SimulateMeta::parameters_provenance`] for the shape. Survey
    /// is a forward-sim subcommand (no `[estimate]` kick-out), so
    /// most entries will be `role = "fixed"` (or "estimated" for the
    /// LHS-swept names).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
}

/// Stable reference to a parent stage. Uses the stage *name* plus its
/// content hash, not a filesystem path — so the reference survives any
/// tree reorganisation. The path is a cache-lookup concern the caller
/// reconstructs via `run_paths::fit_stage_dir`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartsFromRef {
    pub stage: String,
    /// Content hash of the upstream stage, if its `run.json` could be
    /// read at write time. `None` when the upstream directory had no
    /// readable run.json — absent rather than empty so the provenance
    /// chain doesn't silently corrupt into "has a parent, but we don't
    /// know its hash."
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_hash: Option<String>,
}

/// Unified cache status: result of comparing an expected content hash
/// against the `run.json` in a directory. Applies to both simulate
/// and fit-stage runs.
#[derive(Debug, Clone)]
pub enum CacheStatus {
    /// Run directory exists and its stored hash matches the expected
    /// hash; caller can read results from `run_dir`.
    Hit,
    /// Directory exists but the stored hash differs from the expected
    /// one. Typically triggers a re-run with a warning. (The stored/current
    /// hash detail was dropped in M3.2: the only production reader — the
    /// legacy fit cache check — moved to `runid` lookup; `survey` treats
    /// Stale as a plain re-run signal.)
    Stale,
    /// No `run.json` at the expected location; cache miss.
    Miss,
}

impl Run {
    /// Write `run.json` inside `dir`. Creates parent directories.
    /// Write `run.json` atomically: write to `run.json.tmp`, then
    /// rename. POSIX rename within the same filesystem is atomic —
    /// readers either see the complete new file or nothing at all,
    /// never a half-written / truncated JSON. The invariant this
    /// preserves: if `run.json` exists, every sibling artifact was
    /// written before this rename succeeded, so `run.json`'s mere
    /// presence is an authoritative "stage completed" marker.
    ///
    /// Hardening proposal ship-now #3 — replaces a previous plain
    /// `fs::write` that left a crash window in which sibling
    /// artifacts (mle_params.toml, fit_state.toml) were already on
    /// disk but run.json hadn't been written yet, making partial
    /// stages look complete to any reader of the sibling files.
    pub fn write(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(std::io::Error::other)?;
        let tmp = dir.join("run.json.tmp");
        let final_path = dir.join("run.json");
        // Write tmp + rename. If anything fails mid-write, tmp may be
        // left behind — harmless because readers don't look at it,
        // and the next successful write overwrites it.
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &final_path)
    }

    /// Read `run.json` from `dir`. Returns a serde error kinds in
    /// `ErrorKind::InvalidData` if the file exists but doesn't match
    /// the schema — a sign the directory was written by an older
    /// camdl version or a different tool.
    pub fn read(dir: &std::path::Path) -> std::io::Result<Run> {
        let contents = match std::fs::read_to_string(dir.join("run.json")) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // gh#147 (M3.2): no `run.json` here. A CAS fit *segment*
                // (`fits/{stem}-{h8}/`) has no fit-wide record — derive a
                // fit-level `RunKind::Fit` Run from its stage leaves + the
                // fit-level sidecar so `walk_fits_root` / `table_row` /
                // `fit summary` see one fit entry. Falls back to the original
                // NotFound when `dir` is not a fit segment.
                return read_fit_segment(dir).ok_or(e);
            }
            Err(e) => return Err(e),
        };
        match serde_json::from_str::<Run>(&contents) {
            Ok(run) => Ok(run),
            Err(legacy_err) => {
                // gh#147 (M3.2): a content-addressed fit-stage leaf writes a
                // `runid::RunRecord`, not a legacy `Run`. Synthesize a legacy
                // Run so the transitional fit readers (fit_tree, summary,
                // table, MethodResult) keep working; otherwise surface the
                // original legacy parse error.
                if let Ok(rec) = serde_json::from_str::<runid::RunRecord>(&contents) {
                    if let Some(run) = Run::from_fit_record(&rec) {
                        return Ok(run);
                    }
                }
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, legacy_err))
            }
        }
    }

    /// gh#147 (M3.2). Synthesize a legacy `Run` (kind `FitStage`) from a CAS
    /// fit-stage `RunRecord`. The recorded `inputs` (the FitStageMeta-
    /// equivalent written at `finalize`) + `levels` map back to the legacy
    /// shape the transitional fit readers consume; the typed θ̂ (params,
    /// diagnostics) is still loaded per-stage from `fit_state.toml`. Returns
    /// `None` for a non-FitStage record or malformed inputs.
    pub fn from_fit_record(rec: &runid::RunRecord) -> Option<Run> {
        if rec.kind != runid::ArtifactKind::FitStage {
            return None;
        }
        let inputs = rec.inputs.as_object()?;
        let method: MethodKind = inputs
            .get("method")
            .and_then(|v| serde_json::from_value(v.clone()).ok())?;
        let backend: Backend = inputs
            .get("backend")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(Backend::ChainBinomial);
        let status = match rec.status {
            runid::RunStatus::Completed => RunStatus::Completed {
                wall_time_seconds: inputs
                    .get("wall_time_seconds")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            },
            _ => RunStatus::Running,
        };
        Some(Run {
            hash: rec.run_id.to_hex(),
            version: rec.engine_version.clone(),
            created_at: rec.provenance.created_at.clone().unwrap_or_default(),
            argv: rec.provenance.argv.clone(),
            status,
            label: rec.provenance.label.clone(),
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: rec.levels.first().map(|l| l.hash.to_hex()).unwrap_or_default(),
                stage: inputs.get("stage").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                method,
                backend,
                seed: inputs.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
                n_chains: inputs.get("n_chains").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                algorithm: inputs.get("algorithm").cloned().unwrap_or(serde_json::Value::Null),
                best_loglik: inputs.get("best_loglik").and_then(|v| v.as_f64()),
                best_chain: inputs.get("best_chain").and_then(|v| v.as_u64()).map(|x| x as usize),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
            }),
        })
    }

    /// Check whether `dir` has a `run.json` whose `hash` matches
    /// `expected_hash`. Replaces the sim-side `has_cached_traj` +
    /// `RunMeta.sim_hash` pair and the fit-side provenance.json check
    /// with one uniform code path.
    pub fn check_cache(dir: &std::path::Path, expected_hash: &str) -> CacheStatus {
        match Self::read(dir) {
            Ok(run) if run.hash == expected_hash =>
                CacheStatus::Hit,
            Ok(_) => CacheStatus::Stale,
            Err(_) => CacheStatus::Miss,
        }
    }
}

/// The bare stage name from a `NN-stage` provenance label (`"01-scout"` →
/// `"scout"`); a label without an ordinal prefix is returned unchanged.
/// Splits on the first `-` only, so stage names containing `-` survive.
fn bare_stage_name(stage_label: &str) -> String {
    stage_label
        .split_once('-')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| stage_label.to_string())
}

/// The fit-level provenance sidecar (`fits/{stem}-{h8}/fit.meta.json`). A CAS
/// fit's fit level is a path segment with no `RunRecord`; this sidecar is the
/// single authoritative home for the fit-wide attributes that are NOT carried
/// on the stage leaves — the user `--label` and the fit-wide provenance the
/// legacy fit-wide record used to hold (`resolved_priors` = gh#75 per-parameter
/// prior source, `estimated`/`fixed`/`data_hashes`, `model_hash`, paths).
///
/// It is **derived provenance, not a source of truth**: a faithful readable
/// projection of inputs already hashed into the leaf identity (the `FitDigest`
/// — different priors already produce a different fit identity). It is written
/// post-identity and is never fed back into any hash. The producing `fit.toml`
/// is archived beside it as `fit.toml.original` (the config-diff source for
/// `fit table`). Interim home that M4's derived index subsumes.
///
/// Every field except `resolved_priors`-class provenance defaults, so partial
/// sidecars (test fixtures) round-trip; `read_fit_segment` enforces that a
/// Bayesian fit's `resolved_priors` is present (no silent default).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FitSidecar {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub model_hash: String,
    #[serde(default)]
    pub fit_toml_path: String,
    #[serde(default)]
    pub fit_toml_hash: String,
    #[serde(default)]
    pub data_hashes: HashMap<String, String>,
    #[serde(default)]
    pub estimated: Vec<String>,
    #[serde(default)]
    pub fixed: HashMap<String, f64>,
    #[serde(default)]
    pub resolved_priors: Vec<ResolvedPriorEntry>,
    #[serde(default)]
    pub parameters_provenance: HashMap<String, ParameterProvenance>,
}

/// Write the fit-level sidecar and archive the producing `fit.toml`
/// (`fit.toml.original`, the config-diff source `fit table` loads). The archive
/// is best-effort: a CLI-only fit (no `.toml`) has none and config-diff degrades
/// to identity. Idempotent; the caller writes it once per fit segment.
pub fn write_fit_sidecar(
    fit_segment: &std::path::Path,
    fit_toml_path: &std::path::Path,
    sidecar: &FitSidecar,
) -> std::io::Result<()> {
    std::fs::create_dir_all(fit_segment)?;
    if fit_toml_path.is_file() {
        std::fs::copy(fit_toml_path, fit_segment.join("fit.toml.original"))?;
    }
    let bytes = serde_json::to_vec_pretty(sidecar)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(fit_segment.join("fit.meta.json"), bytes)
}

/// Read the fit-level sidecar; `None` when absent (an incomplete/legacy
/// segment — `read_fit_segment` treats that as a malformed fit and skips it
/// loudly rather than fabricating empty provenance).
pub fn read_fit_sidecar(fit_segment: &std::path::Path) -> Option<FitSidecar> {
    let bytes = std::fs::read(fit_segment.join("fit.meta.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Derive a fit-level `RunKind::Fit` `Run` from a CAS fit segment
/// (`fits/{stem}-{h8}/`): one entry computed from its `FitStage` leaves plus the
/// fit-level [`FitSidecar`]. `run.hash` is the `fit`-level (FitDigest) hash
/// shared by every leaf (the sidecar is never identity-bearing); `created_at`
/// is the latest leaf; `stages_declared` is the bare stage names in execution
/// order (the `NN-` ordinal prefix sorts topologically); `label` and the
/// fit-wide provenance (`resolved_priors`, `estimated`, `fixed`, `data_hashes`,
/// `model_hash`) come from the sidecar — NOT defaulted.
///
/// Provenance integrity (gh#147): a segment with stage leaves but no sidecar is
/// malformed — skipped with a loud error rather than surfaced with empty
/// provenance. For a Bayesian fit (a `pgas`/`pmmh` leaf), an empty
/// `resolved_priors` is itself a bug signal (gh#75) and is flagged.
///
/// Transitional, like [`Run::from_fit_record`]: it keeps the legacy `Run` /
/// `RunKind::Fit` readers (`walk_fits_root`, `table_row::build_row`,
/// `fit summary`) working during M2→M3. M3.3 deletes the legacy fit readers and
/// this adapter with them; a one-boundary translation, not a permanent home.
pub fn read_fit_segment(seg: &std::path::Path) -> Option<Run> {
    let mut leaves: Vec<runid::RunRecord> = crate::cas_read::walk_records(seg)
        .into_iter()
        .map(|(_, r)| r)
        .filter(|r| r.kind == runid::ArtifactKind::FitStage)
        .collect();
    if leaves.is_empty() {
        return None;
    }
    let stage_label = |r: &runid::RunRecord| -> String {
        r.levels
            .iter()
            .find(|l| l.name == "stage")
            .map(|l| l.label.clone())
            .unwrap_or_default()
    };
    // Execution order: the `NN-stage` ordinal prefix sorts topologically.
    leaves.sort_by(|a, b| stage_label(a).cmp(&stage_label(b)));

    let fit_hash = leaves
        .iter()
        .find_map(|r| {
            r.levels
                .iter()
                .find(|l| l.name == "fit")
                .map(|l| l.hash.to_hex())
        })
        .unwrap_or_default();
    let created_at = leaves
        .iter()
        .filter_map(|r| r.provenance.created_at.clone())
        .max()
        .unwrap_or_default();
    let version = leaves
        .first()
        .map(|r| r.engine_version.clone())
        .unwrap_or_default();
    let argv = leaves
        .first()
        .map(|r| r.provenance.argv.clone())
        .unwrap_or_default();
    // Bare stage names in execution order, dedup preserving order.
    let mut stages_declared: Vec<String> = Vec::new();
    for r in &leaves {
        let bare = bare_stage_name(&stage_label(r));
        if !stages_declared.contains(&bare) {
            stages_declared.push(bare);
        }
    }

    // Provenance: from the sidecar, never defaulted. A fit with leaves but no
    // sidecar is malformed — skip it loudly (the writer always writes one).
    let side = match read_fit_sidecar(seg) {
        Some(s) => s,
        None => {
            eprintln!(
                "warning: fit {} has stage leaves but no fit.meta.json provenance \
                 sidecar — skipping (provenance missing)",
                seg.display()
            );
            return None;
        }
    };
    // A Bayesian fit (`pgas`/`pmmh` leaf consumes priors) with empty
    // resolved_priors is a dropped-provenance bug (gh#75), not a valid state.
    let is_bayesian = leaves.iter().any(|r| {
        matches!(
            r.inputs.get("method").and_then(|m| m.as_str()),
            Some("pgas") | Some("pmmh")
        )
    });
    if is_bayesian && side.resolved_priors.is_empty() {
        eprintln!(
            "error: Bayesian fit {} has empty resolved_priors in its sidecar — \
             prior-source provenance was dropped (gh#75)",
            seg.display()
        );
    }
    Some(Run {
        hash: fit_hash,
        version,
        created_at,
        argv,
        status: RunStatus::Completed { wall_time_seconds: 0.0 },
        label: side.label,
        kind: RunKind::Fit(FitMeta {
            model: side.model_path,
            model_hash: side.model_hash,
            fit_toml_path: side.fit_toml_path,
            fit_toml_hash: side.fit_toml_hash,
            data_hashes: side.data_hashes,
            estimated: side.estimated,
            fixed: side.fixed,
            stages_declared,
            ic_free: false,
            resolved_priors: side.resolved_priors,
            parameters_provenance: side.parameters_provenance,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_simulate_run() -> Run {
        Run {
            hash: "abc12345def6789000000000000000000000000000000000000000000000000".into(),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "simulate".into(), "sir.camdl".into()],
            status: RunStatus::Completed { wall_time_seconds: 1.23 },
            label: None,
            kind: RunKind::Simulate(SimulateMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                scenario: "baseline".into(),
                sim_hash: "abc12345".into(),
                scen_hash: "def67890".into(),
                seed: 42,
                backend: crate::args::types::Backend::Gillespie,
                dt: 1.0,
                sweep_point: HashMap::new(),
                from_fit_hash: None,
                parameters_provenance: Default::default(),
                        }),
        }
    }

    fn sample_fit_run() -> Run {
        Run {
            hash: "deadbeef".repeat(8),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into(), "fit.toml".into()],
            status: RunStatus::Completed { wall_time_seconds: 42.0 },
            label: None,
            kind: RunKind::Fit(FitMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                fit_toml_path: "fit.toml".into(),
                fit_toml_hash: "cafebabe".into(),
                data_hashes: {
                    let mut m = HashMap::new();
                    m.insert("cases".into(), "d4ta".repeat(2));
                    m
                },
                estimated: vec!["beta".into(), "gamma".into()],
                fixed: {
                    let mut m = HashMap::new();
                    m.insert("N0".into(), 1000.0);
                    m
                },
                stages_declared: vec!["scout".into(), "refine".into()],
                ic_free: false,
                resolved_priors: vec![],
                parameters_provenance: Default::default(),
                        }),
        }
    }

    fn sample_fit_stage_run() -> Run {
        Run {
            hash: "ae123456".repeat(8),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into(),
                       "fit.toml".into(), "--stage".into(), "refine".into()],
            status: RunStatus::Completed { wall_time_seconds: 10.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "deadbeef".repeat(8),
                stage: "refine".into(),
                method: MethodKind::If2,
                backend: Backend::ChainBinomial,
                seed: 42,
                n_chains: 4,
                algorithm: serde_json::Value::Null,
                best_loglik: Some(-56.7),
                best_chain: Some(1),
                starts_from: Some(StartsFromRef {
                    stage: "scout".into(),
                    stage_hash: Some("beef1234".repeat(8)),
                }),
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        }
    }

    #[test]
    fn simulate_run_roundtrip() {
        let r = sample_simulate_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"simulate""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hash, r.hash);
        match parsed.kind {
            RunKind::Simulate(m) => assert_eq!(m.seed, 42),
            _ => panic!("expected Simulate"),
        }
    }

    #[test]
    fn fit_run_roundtrip() {
        let r = sample_fit_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"fit""#));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Fit(m) => {
                assert_eq!(m.estimated.len(), 2);
                assert_eq!(m.stages_declared, vec!["scout", "refine"]);
            }
            _ => panic!("expected Fit"),
        }
    }

    #[test]
    fn fit_stage_run_roundtrip() {
        let r = sample_fit_stage_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"fit-stage""#));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => {
                assert_eq!(m.stage, "refine");
                assert_eq!(m.best_loglik, Some(-56.7));
                assert!(m.starts_from.is_some());
            }
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn write_read_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_run_meta_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_fit_run();
        r.write(&tmp).unwrap();
        let read = Run::read(&tmp).unwrap();
        assert_eq!(read.hash, r.hash);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn check_cache_hit_stale_miss() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_cache_status_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_simulate_run();
        let stored_hash = r.hash.clone();
        r.write(&tmp).unwrap();

        // Miss before write.
        let empty = tmp.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(matches!(Run::check_cache(&empty, &stored_hash), CacheStatus::Miss));

        // Hit with matching hash.
        assert!(matches!(
            Run::check_cache(&tmp, &stored_hash),
            CacheStatus::Hit
        ));

        // Stale when hash differs.
        assert!(
            matches!(Run::check_cache(&tmp, "different_hash"), CacheStatus::Stale),
            "expected Stale on a hash mismatch",
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn fit_stage_back_pointer_matches_parent_fit() {
        // The FitStageMeta.fit_hash field is how a stage references
        // its parent fit. If the two ever drift, stages become un-
        // attributable — guard against that by constructing both with
        // the same hash string and round-tripping the stage through
        // JSON, asserting the field survives.
        let parent = sample_fit_run();
        let parent_hash = parent.hash.clone();
        let stage = Run {
            hash: "stage0000".repeat(8),
            version: "v".into(),
            created_at: "t".into(),
            argv: vec![],
            status: RunStatus::Completed { wall_time_seconds: 1.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: parent_hash.clone(),
                stage: "scout".into(),
                method: MethodKind::If2,
                backend: Backend::ChainBinomial,
                seed: 1,
                n_chains: 4,
                algorithm: serde_json::Value::Null,
                best_loglik: None,
                best_chain: None,
                starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        };
        let json = serde_json::to_string(&stage).unwrap();
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => assert_eq!(m.fit_hash, parent_hash),
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn atomic_write_leaves_no_tmp_after_success() {
        // On a clean run, run.json.tmp should not remain. Regression
        // guard: if we ever forget the rename, tmp would be left behind
        // and this test catches it.
        let tmp = std::env::temp_dir().join(format!(
            "camdl_atomic_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_simulate_run();
        r.write(&tmp).unwrap();
        assert!(tmp.join("run.json").exists(), "final run.json must exist");
        assert!(!tmp.join("run.json.tmp").exists(),
            "run.json.tmp must not be left behind after successful write");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn atomic_write_mid_crash_leaves_no_visible_run_json() {
        // Simulate the crash window: write the .tmp but never rename.
        // A reader should NOT see run.json — the invariant is "if
        // run.json exists, the write completed."
        let tmp = std::env::temp_dir().join(format!(
            "camdl_crash_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        // Manually create a run.json.tmp to simulate a crashed write.
        std::fs::write(tmp.join("run.json.tmp"),
            r#"{"partial": "data", "oops": "crash"}"#).unwrap();
        // No run.json should exist.
        assert!(!tmp.join("run.json").exists(),
            "crashed write: run.json must not be visible");
        // A reader's check_cache should report Miss, not a malformed
        // run.
        match Run::check_cache(&tmp, "any-hash") {
            CacheStatus::Miss => {}
            other => panic!("expected Miss (no run.json), got {:?}", other),
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn modern_stage_hash_omits_the_field_when_none() {
        // New writes omit the field entirely (skip_serializing_if).
        // Regression guard against someone re-introducing empty-string
        // writes.
        let r = Run {
            hash: "x".repeat(64), version: "v".into(), created_at: "t".into(),
            argv: vec![], status: RunStatus::Running, label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "f".repeat(64),
                stage: "refine".into(), method: MethodKind::If2,
                backend: Backend::ChainBinomial,
                seed: 1, n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: None, best_chain: None,
                starts_from: Some(StartsFromRef {
                    stage: "scout".into(), stage_hash: None,
                }),
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        };
        let json = serde_json::to_string(&r).unwrap();
        // The `stage_hash` key should NOT appear when it's None —
        // otherwise the schema is silently round-tripping an empty
        // value.
        assert!(!json.contains("\"stage_hash\""),
            "None stage_hash should be omitted entirely, got {}", json);
    }

    #[test]
    fn optional_fields_skip_when_empty() {
        // Regression guard: serde(skip_serializing_if) on best_loglik,
        // best_chain, starts_from, sweep_point, ic_free. An empty
        // field shouldn't bloat the JSON.
        let r = Run {
            hash: "x".repeat(64), version: "v".into(),
            created_at: "t".into(), argv: vec![], status: RunStatus::Running,
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "f".repeat(64),
                stage: "mle".into(), method: MethodKind::If2,
                backend: Backend::ChainBinomial,
                seed: 1, n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: None, best_chain: None, starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("best_loglik"));
        assert!(!json.contains("best_chain"));
        assert!(!json.contains("starts_from"));
        assert!(!json.contains("parent_profile_hash"));
    }

    #[test]
    fn profile_run_roundtrip() {
        let r = Run {
            hash: "a".repeat(64),
            version: "v".into(),
            created_at: "2026-04-24T00:00:00Z".into(),
            argv: vec!["camdl".into(), "profile".into()],
            status: RunStatus::Completed { wall_time_seconds: 3600.0 },
            label: None,
            kind: RunKind::Profile(ProfileMeta {
                model: "he2010_london.camdl".into(),
                model_hash: "b".repeat(64),
                focal_params: vec!["R0".into(), "gamma".into()],
                grid: vec![
                    GridAxis { param: "R0".into(), values: vec![40.0, 50.0, 60.0] },
                    GridAxis { param: "gamma".into(), values: vec![0.1, 0.2] },
                ],
                n_starts: 3,
                if2_config_hash: "c".repeat(64),
                base_params_hash: "d".repeat(64),
                seed_base: 42,
                total_jobs: 18,
                fit_toml_hash: None,
                resolved_priors: vec![],
                suppressed_warnings: vec![],
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"profile""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Profile(m) => {
                assert_eq!(m.focal_params, vec!["R0", "gamma"]);
                assert_eq!(m.grid.len(), 2);
                assert_eq!(m.grid[0].values, vec![40.0, 50.0, 60.0]);
                assert_eq!(m.total_jobs, 18);
            }
            _ => panic!("expected Profile kind"),
        }
    }

    #[test]
    fn survey_run_roundtrip() {
        let mut bounds = HashMap::new();
        bounds.insert("beta".into(), (0.001_f64, 1.0_f64));
        bounds.insert("gamma".into(), (0.01_f64, 0.5_f64));
        let mut data_hashes = HashMap::new();
        data_hashes.insert("cases".into(), "0123abcd".repeat(8));
        let r = Run {
            hash: "fa".repeat(32),
            version: "v".into(),
            created_at: "2026-05-03T00:00:00Z".into(),
            argv: vec!["camdl".into(), "survey".into(), "sir.camdl".into()],
            status: RunStatus::Completed { wall_time_seconds: 12.5 },
            label: None,
            kind: RunKind::Survey(SurveyMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                data_hashes,
                bounds,
                n_points: 1000,
                eval_method: SurveyEvalMethod::Pfilter,
                eval_particles: 200,
                eval_replicates: 3,
                seed: 42,
                fixed: HashMap::new(),
                scenario: None,
                estimated: vec!["beta".into(), "gamma".into()],
                parameters_provenance: Default::default(),
                        }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"survey""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Survey(m) => {
                assert_eq!(m.n_points, 1000);
                assert_eq!(m.eval_method, SurveyEvalMethod::Pfilter);
                assert_eq!(m.eval_particles, 200);
                assert_eq!(m.eval_replicates, 3);
                assert_eq!(m.estimated, vec!["beta", "gamma"]);
            }
            _ => panic!("expected Survey"),
        }
    }

    #[test]
    fn survey_eval_method_serializes_lowercase() {
        let p = SurveyEvalMethod::Pfilter;
        assert_eq!(serde_json::to_string(&p).unwrap(), r#""pfilter""#);
        let s = SurveyEvalMethod::Simulate;
        assert_eq!(serde_json::to_string(&s).unwrap(), r#""simulate""#);
    }

    #[test]
    fn fit_stage_with_profile_backref_roundtrips() {
        // A grid-point × start child under a profile: FitStageMeta with
        // parent_profile_hash + profile_point_idx + profile_start_idx
        // populated. Verifies the optional fields round-trip correctly.
        let r = Run {
            hash: "e".repeat(64),
            version: "v".into(),
            created_at: "2026-04-24T00:00:00Z".into(),
            argv: vec!["camdl".into(), "profile".into()],
            status: RunStatus::Completed { wall_time_seconds: 120.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "".into(),    // no parent Fit; parent is a Profile
                stage: "if2".into(),
                method: MethodKind::If2,
                backend: Backend::ChainBinomial,
                seed: 142,
                n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: Some(-5827.35),
                best_chain: Some(0),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: Some("f".repeat(64)),
                profile_point_idx: Some(7),
                profile_start_idx: Some(2),
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("parent_profile_hash"));
        assert!(json.contains("profile_point_idx"));
        assert!(json.contains("profile_start_idx"));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => {
                assert_eq!(m.parent_profile_hash, Some("f".repeat(64)));
                assert_eq!(m.profile_point_idx, Some(7));
                assert_eq!(m.profile_start_idx, Some(2));
            }
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn run_status_running_serializes_as_lowercase_string() {
        let s = RunStatus::Running;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#""running""#);
        let parsed: RunStatus = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_running());
        assert_eq!(parsed.wall_time_seconds(), None);
    }

    #[test]
    fn run_status_completed_carries_wall_time() {
        let s = RunStatus::Completed { wall_time_seconds: 42.5 };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"{"completed":{"wall_time_seconds":42.5}}"#);
        let parsed: RunStatus = serde_json::from_str(&json).unwrap();
        assert!(!parsed.is_running());
        assert_eq!(parsed.wall_time_seconds(), Some(42.5));
    }

    #[test]
    fn run_status_round_trips_inside_run() {
        // The status field round-trips inside a full Run wrapper, so
        // run.json writes can read back into the typed RunStatus.
        let r = sample_simulate_run();
        let json = serde_json::to_string(&r).unwrap();
        let parsed: Run = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed.status,
            RunStatus::Completed { wall_time_seconds }
            if (wall_time_seconds - 1.23).abs() < 1e-9));
    }

    // ─── gh#83/gh#85 step 9: parameter / init provenance round-trip ──

    /// Round-trips `ParameterProvenance` from a resolved parameter
    /// through `run.json` serialization. Covers audit checklist
    /// item 4: every entry's `source` matches a `ValueSource`
    /// variant tag.
    #[test]
    fn parameter_provenance_round_trips_via_simulate_meta() {
        use crate::params_resolver::{
            FixReason, ParameterRole, ResolvedParameter,
            ScenarioOverride, ValueSource,
        };
        // Build one entry per `ValueSource` variant tag — exercises
        // every branch of `ValueSource::tag()` through the round-trip.
        let resolved_entries = vec![
            ResolvedParameter {
                name:  "beta".into(),
                value: 0.42,
                source: ValueSource::ModelDefault,
                role: ParameterRole::Estimated,
                overrode_scenario: None,
            },
            ResolvedParameter {
                name:  "gamma".into(),
                value: 0.10,
                source: ValueSource::FitTomlFixed,
                role: ParameterRole::Fixed {
                    reason: FixReason::NotInEstimate,
                },
                overrode_scenario: None,
            },
            ResolvedParameter {
                name:  "rho".into(),
                value: 0.50,
                source: ValueSource::FixedCli,
                role: ParameterRole::Fixed {
                    reason: FixReason::KickedFromEstimate {
                        by: ValueSource::FixedCli,
                    },
                },
                overrode_scenario: Some(ScenarioOverride {
                    scenario:       "worst_case".into(),
                    scenario_value: 0.30,
                }),
            },
            ResolvedParameter {
                name:  "mu".into(),
                value: 0.05,
                source: ValueSource::Scenario("worst_case".into()),
                role: ParameterRole::Fixed {
                    reason: FixReason::NotInEstimate,
                },
                overrode_scenario: None,
            },
            ResolvedParameter {
                name:  "iota".into(),
                value: 0.01,
                source: ValueSource::FixedFile {
                    path: std::path::PathBuf::from("/tmp/fix.toml"),
                },
                role: ParameterRole::Fixed {
                    reason: FixReason::KickedFromEstimate {
                        by: ValueSource::FixedFile {
                            path: std::path::PathBuf::from("/tmp/fix.toml"),
                        },
                    },
                },
                overrode_scenario: None,
            },
        ];
        let parameters_provenance: HashMap<String, ParameterProvenance> =
            resolved_entries.iter().map(|rp| {
                (rp.name.clone(), ParameterProvenance::from_resolved(rp))
            }).collect();
        // Wrap into a SimulateMeta + Run + serialize + deserialize.
        let run = Run {
            hash:    "deadbeef".repeat(8),
            version: "test".into(),
            created_at: "2026-05-25T00:00:00Z".into(),
            argv:    vec![],
            status:  RunStatus::Completed { wall_time_seconds: 0.0 },
            label:   None,
            kind: RunKind::Simulate(SimulateMeta {
                model: "m.camdl".into(),
                model_hash: "h".repeat(64),
                scenario: "worst_case".into(),
                sim_hash: "s".repeat(64),
                scen_hash: "c".repeat(64),
                seed: 0,
                backend: crate::args::types::Backend::Gillespie,
                dt: 1.0,
                sweep_point: HashMap::new(),
                from_fit_hash: None,
                parameters_provenance,
            }),
        };
        let json = serde_json::to_string(&run).unwrap();
        let parsed: Run = serde_json::from_str(&json).unwrap();
        let RunKind::Simulate(meta) = parsed.kind else {
            panic!("expected Simulate");
        };
        // Non-empty, per audit item 4.
        assert!(!meta.parameters_provenance.is_empty(),
            "parameters_provenance must be populated");
        assert_eq!(meta.parameters_provenance.len(), 5);
        // Every entry's `source` matches a `ValueSource` variant tag.
        let allowed_source_tags: std::collections::HashSet<&str> = [
            "model_default", "scenario", "fit_toml_fixed",
            "fixed_file", "fixed_cli",
        ].iter().copied().collect();
        for (name, prov) in &meta.parameters_provenance {
            assert!(allowed_source_tags.contains(prov.source.as_str()),
                "{}: source tag {} not in ValueSource variants",
                name, prov.source);
            assert!(prov.role == "fixed" || prov.role == "estimated",
                "role must be fixed|estimated, got {}", prov.role);
        }
        // Specific assertions: kick_from_estimate present on rho/iota;
        // overrode_scenario present on rho only.
        let rho = &meta.parameters_provenance["rho"];
        assert!(rho.kicked_from_estimate.is_some());
        assert_eq!(rho.kicked_from_estimate.as_ref().unwrap().by, "fixed_cli");
        assert!(rho.overrode_scenario.is_some());
        assert_eq!(rho.overrode_scenario.as_ref().unwrap().scenario, "worst_case");
        assert!((rho.overrode_scenario.as_ref().unwrap().scenario_value - 0.30).abs() < 1e-12);
        let beta = &meta.parameters_provenance["beta"];
        assert_eq!(beta.role, "estimated");
        assert!(beta.kicked_from_estimate.is_none());
    }

    /// Audit checklist item 5: every `InitMethod` variant has at
    /// least one round-trip producing a `run.json` whose
    /// `init_provenance.method` equals that variant's tag.
    #[test]
    fn init_provenance_method_tag_matches_for_every_variant() {
        use crate::fit::chain_starts::{
            ChainStart, ChainStarts, InitSource,
        };
        use crate::fit::init::{
            InitMethod, MleSource, PosteriorSource,
        };
        // One ChainStarts per (variant, expected tag) pair.
        let cases: Vec<(InitMethod, &str)> = vec![
            (InitMethod::Single,        "single"),
            (InitMethod::Uniform,       "uniform"),
            (InitMethod::Lhs,           "lhs"),
            (InitMethod::SurveyTopK,    "survey_top_k"),
            (InitMethod::FromPrior,     "from_prior"),
            (InitMethod::FromPosterior {
                source: PosteriorSource::DrawsTsv("/tmp/draws.tsv".into()),
            }, "from_posterior"),
            (InitMethod::FromMle {
                source: MleSource::File("/tmp/mle.toml".into()),
            }, "from_mle"),
            (InitMethod::FromParams {
                path: "/tmp/params.toml".into(),
            }, "from_params"),
        ];
        for (method, expected_tag) in &cases {
            // Single-chain ChainStarts → InitProvenance → JSON.
            let cs = ChainStarts {
                starts: vec![ChainStart {
                    chain_id: 0,
                    values: HashMap::from([("beta".into(), 0.5_f64)]),
                    source: InitSource::SeededBase,
                }],
                method: method.clone(),
            };
            let prov = InitProvenance::from_chain_starts(&cs);
            assert_eq!(prov.method, *expected_tag,
                "InitProvenance.method tag mismatch for variant {:?}", method);
            // JSON round-trip preserves the tag.
            let json = serde_json::to_string(&prov).unwrap();
            let parsed: InitProvenance = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.method, *expected_tag);
            assert_eq!(parsed.chains.len(), 1);
            assert!(parsed.chains[0].contains_key("beta"));
        }
    }

    /// Init-source per-chain provenance round-trips through JSON
    /// preserving the InitSource tag. Pairs with the above method-tag
    /// test to cover audit item 5 at the per-chain level.
    #[test]
    fn init_source_per_chain_tags_round_trip() {
        use crate::fit::chain_starts::{
            ChainStart, ChainStarts, InitSource,
        };
        use crate::fit::init::InitMethod;
        let starts = vec![
            ChainStart {
                chain_id: 0,
                values: HashMap::from([("beta".into(), 0.1_f64)]),
                source: InitSource::PriorDraw { seed: 42 },
            },
            ChainStart {
                chain_id: 1,
                values: HashMap::from([("beta".into(), 0.2_f64)]),
                source: InitSource::PosteriorRow {
                    row: 7, path: "/tmp/draws.tsv".into(),
                },
            },
            ChainStart {
                chain_id: 2,
                values: HashMap::from([("beta".into(), 0.3_f64)]),
                source: InitSource::MlePoint { path: "/tmp/mle.toml".into() },
            },
            ChainStart {
                chain_id: 3,
                values: HashMap::from([("beta".into(), 0.4_f64)]),
                source: InitSource::ParamsPoint {
                    path: "/tmp/params.toml".into(),
                },
            },
        ];
        let cs = ChainStarts { starts, method: InitMethod::FromPrior };
        let prov = InitProvenance::from_chain_starts(&cs);
        // Each chain's per-parameter source matches the InitSource tag.
        assert_eq!(prov.chains[0]["beta"].source, "prior_draw");
        assert_eq!(prov.chains[1]["beta"].source, "posterior_row");
        assert_eq!(prov.chains[2]["beta"].source, "mle_point");
        assert_eq!(prov.chains[3]["beta"].source, "params_point");
        // JSON round-trip preserves the tags.
        let json = serde_json::to_string(&prov).unwrap();
        let parsed: InitProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.chains[0]["beta"].source, "prior_draw");
        assert_eq!(parsed.chains[3]["beta"].source, "params_point");
    }

    /// gh#147 (M3.2) regression guard for the dropped-provenance bug: the gh#75
    /// per-parameter prior sources must survive the fit-level sidecar
    /// write → `read_fit_segment` read round trip. Before the sidecar carried
    /// `resolved_priors`, the reader defaulted it empty and this class of bug
    /// shipped silently. Write a Bayesian (`pgas`) stage leaf + a sidecar with
    /// mixed sources, read the fit back, and assert each `.source` matches.
    #[test]
    fn fit_sidecar_resolved_priors_survive_read_fit_segment_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_sidecar_priors_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let seg = tmp.join("fits").join("demo-abc12345");
        let leaf = seg.join("01-posterior-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        // A Bayesian (pgas) stage leaf — `read_fit_segment` requires its
        // sidecar to carry resolved_priors.
        std::fs::write(
            leaf.join("run.json"),
            r#"{"format_version":1,"kind":"fit_stage","run_id":"abc1234500000000000000000000000000000000000000000000000000000000","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{"name":"fit","label":"demo","hash":"abc123450000000000000000000000000000000000000000000000000000000a","schema_version":1},{"name":"stage","label":"01-posterior","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1},{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}],"status":"completed","artifacts":{},"inputs":{"stage":"posterior","method":"pgas","backend":"chain_binomial","seed":1,"n_chains":2},"provenance":{"created_at":"2026-04-19T12:00:00Z","argv":["camdl","fit","run"]}}"#,
        )
        .unwrap();

        let sidecar = FitSidecar {
            estimated: vec!["beta".into(), "gamma".into()],
            resolved_priors: vec![
                ResolvedPriorEntry { param: "beta".into(), source: "model_ir".into() },
                ResolvedPriorEntry { param: "gamma".into(), source: "fit_toml".into() },
            ],
            ..Default::default()
        };
        // No fit.toml on disk → archive step is skipped; the sidecar still writes.
        write_fit_sidecar(&seg, std::path::Path::new("nonexistent.toml"), &sidecar).unwrap();

        let run = read_fit_segment(&seg).expect("read_fit_segment must derive a fit entry");
        let meta = match run.kind {
            RunKind::Fit(m) => m,
            other => panic!("expected RunKind::Fit, got {:?}", other),
        };
        let source = |p: &str| -> Option<&str> {
            meta.resolved_priors
                .iter()
                .find(|e| e.param == p)
                .map(|e| e.source.as_str())
        };
        assert_eq!(source("beta"), Some("model_ir"),
            "beta prior source must survive the sidecar round trip");
        assert_eq!(source("gamma"), Some("fit_toml"),
            "gamma prior source must survive the sidecar round trip");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
