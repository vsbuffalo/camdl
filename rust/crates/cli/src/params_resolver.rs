//! Unified parameter-value resolver — single source of truth for the
//! precedence chain documented in `docs/camdl-run-spec.md §1.3`.
//!
//! Background
//! ----------
//!
//! Before this module, three half-resolvers + several inline blocks
//! implemented the same precedence rules independently:
//!
//!   - `util::resolve_run_model` for `simulate` / `lineage` (no
//!     `[estimate]` semantics)
//!   - `fit::config_v2::FixedParams::resolve_with_model` for `survey`
//!     / `profile` `[fixed]` resolution (no CLI `--fixed`)
//!   - inline blocks in `profile.rs:437-453`, `if2.rs:109-168`,
//!     `pfilter.rs:47-55` (each subcommand-specific)
//!
//! Each was correct on its own slice; together they let small details
//! drift silently. See
//! `docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md` for the
//! full audit and design rationale.
//!
//! Design
//! ------
//!
//! Two verbs, one resolver:
//!
//!   - `--fixed` carries explicit `NAME=VALUE` pairs (CLI side) and
//!     bulk files (`--fixed-file`).
//!   - On inference subcommands, names that appear in `--fixed` are
//!     also kicked out of the `[estimate]` set — `gamma=0.1` on a
//!     profile means "slice through gamma=0.1", which requires gamma
//!     to be fixed at 0.1 *and* not estimated.
//!
//! The resolver owns precedence (`resolve_parameters`), records
//! provenance (`ResolvedParameter.source`), and is the sole writer of
//! `model.parameters[i].value` outside the IR layer.
//!
//! Precedence (last wins)
//! ----------------------
//!
//! This implements `docs/camdl-run-spec.md §1.3` exactly:
//!
//!   1. Model parameter default (`p.value` from DSL)
//!   2. `fit.toml [fixed]` block (when present)
//!   3. `--fixed-file <toml>` (each file layered in order; later
//!      overrides earlier)
//!   3.5. Draw row / sweep point (`point_overrides`) — automated
//!      M-layer variation (a posterior/prior/uniform draw, an
//!      explicit draws file, or a sweep grid point)
//!   4. Scenario (`preset.params` + multiplicative `preset.scale`, or
//!      an inline ad-hoc scenario's `set`/`scale`)
//!   5. `--fixed NAME=VALUE` (highest)
//!
//! The structural distinction between tiers 3.5 and 5: a draw/sweep
//! value is *automated M-layer variation* and is counterfactual-
//! modifiable, so a scenario `set`/`scale` (a counterfactual on M)
//! overrides it. A `--fixed` value is the user's *explicit assertion*
//! about a specific run and overrides everything, scenario included.
//! Inline and named scenarios resolve at the SAME tier 4 — an inline
//! scenario is a preset with no model lookup — so the two are
//! indistinguishable to a parameter's final value (only the
//! provenance label differs).
//!
//! `[estimate]` membership rule:
//!   - Start: `estimate_set = inputs.fit_toml_estimate`
//!   - Remove every name that appears in (3) or (5) — i.e. user-
//!     explicit `--fixed{,-file}` assertions. Neither the scenario
//!     tier (4) nor the draw/sweep tier (3.5) kicks from
//!     `[estimate]`: scenarios are σ-layer constructs (counterfactual
//!     modifications) and draws/sweeps are automated M-layer
//!     variation — neither is a user assertion that a parameter is
//!     fixed at a value.
//!   - Emit a warning (not an error) for each such removal
//!
//! On non-inference subcommands, `inputs.fit_toml_estimate` is empty;
//! the kick-out logic is a no-op.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use indexmap::{IndexMap, IndexSet};
use ir::table::TableSource;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Inputs gathered from the CLI + model. Every subcommand assembles
/// one of these before dispatch; `resolve_parameters` returns the
/// per-parameter outcome plus provenance.
pub struct ParameterInputs<'a> {
    pub model:              &'a ir::Model,
    /// A NAMED scenario preset (looked up in `model.presets`). Mutually
    /// exclusive with the inline-scenario fields below — a scenario
    /// reference is either a preset OR an ad-hoc inline patch, never both.
    pub scenario:           Option<&'a str>,
    pub adhoc_enable:       &'a [String],
    pub adhoc_disable:      &'a [String],
    /// An INLINE ad-hoc scenario's `set`/`scale` and display name. An
    /// inline scenario resolves at the SAME tier as a named preset
    /// (tier 4) — `set` overrides the draw/sweep tier, `scale` multiplies
    /// the current value, both tagged `ValueSource::Scenario(name)`. This
    /// is what makes an inline scenario resolve IDENTICALLY to the
    /// equivalent named preset (spec §1.3). Set only when `scenario` is
    /// `None` (the ad-hoc path); the named-preset path reads `set`/`scale`
    /// from the preset instead. `scenario_inline_name` carries the
    /// display label for provenance; empty `set`/`scale` with no name is
    /// the no-op baseline.
    pub scenario_inline_name:  Option<&'a str>,
    pub scenario_inline_set:   &'a [(String, f64)],
    pub scenario_inline_scale: &'a [(String, f64)],
    /// A draw row / sweep point's per-parameter overrides — automated
    /// M-layer variation (spec §1.3, between `--fixed-file` and scenario).
    /// Distinct from `fixed_cli`: a draw/sweep value is overridden by a
    /// scenario `set`/`scale`, whereas `--fixed` is not. Empty for a plain
    /// single-point run.
    pub point_overrides:    &'a [(String, f64)],
    pub fixed_cli:          &'a [(String, f64)],
    pub fixed_files:        &'a [PathBuf],
    pub fit_toml_fixed:     &'a IndexMap<String, f64>,
    pub fit_toml_estimate:  &'a IndexSet<String>,
    pub table_files:        &'a HashMap<String, PathBuf>,
}

/// Where a parameter's final value came from. Serialised verbatim
/// into `run.json`'s `parameters_provenance` block via
/// [`ValueSource::tag`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueSource {
    /// `p.value` from the DSL — the model's authored default.
    ModelDefault,
    /// A named scenario's `preset.params` entry (or composed entries).
    Scenario(String),
    /// The `[fixed]` block of a `--fit` toml.
    FitTomlFixed,
    /// A `--fixed-file <toml>` invocation; carries the path so
    /// provenance distinguishes which file won under layering.
    FixedFile { path: PathBuf },
    /// A draw row / sweep point override (automated M-layer variation).
    /// Resolves below scenario (spec §1.3): a scenario `set`/`scale` wins
    /// over a draw/sweep value.
    SweepPoint,
    /// A `--fixed NAME=VALUE` CLI flag.
    FixedCli,
}

impl ValueSource {
    /// Stable string tag for `run.json` serialisation.
    pub fn tag(&self) -> &'static str {
        match self {
            ValueSource::ModelDefault    => "model_default",
            ValueSource::Scenario(_)     => "scenario",
            ValueSource::FitTomlFixed    => "fit_toml_fixed",
            ValueSource::FixedFile { .. } => "fixed_file",
            ValueSource::SweepPoint      => "sweep_point",
            ValueSource::FixedCli        => "fixed_cli",
        }
    }
}

/// Resolver-decided role for a parameter. ADT-shaped rather than
/// `bool fixed` so the *reason* a parameter ended up fixed is
/// first-class — the `run.json` provenance distinguishes "never in
/// [estimate]" from "was in [estimate], --fixed kicked it out",
/// which matters for auditing whether a profile slice did what the
/// user intended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterRole {
    Fixed { reason: FixReason },
    Estimated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixReason {
    /// The parameter was never in `[estimate]` to begin with — either
    /// no `--fit` toml was passed, or the toml did not list it. On
    /// non-inference subcommands, every parameter falls here.
    NotInEstimate,
    /// The parameter was listed in `[estimate]`, but `--fixed` /
    /// `--fixed-file` pinned it to an explicit value, kicking it out.
    KickedFromEstimate { by: ValueSource },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioOverride {
    /// The active scenario's name (preset name).
    pub scenario:       String,
    /// The value the scenario tried to set, before a higher-precedence
    /// source overrode it.
    pub scenario_value: f64,
}

#[derive(Debug, Clone)]
pub struct ResolvedParameter {
    pub name:   String,
    pub value:  f64,
    pub source: ValueSource,
    pub role:   ParameterRole,
    /// Present iff the active scenario set this parameter to a value
    /// different from the final winner — i.e. a higher-precedence
    /// source (currently only `--fixed-cli` given the spec §1.3
    /// ordering) silently displaced the scenario value. Pairs with
    /// the [`ResolverWarning::ScenarioOverridden`] warning so the
    /// override is auditable from `run.json` even after stderr is
    /// gone.
    pub overrode_scenario: Option<ScenarioOverride>,
}

#[derive(Debug, Clone)]
pub struct ResolvedParameters {
    pub params:       Vec<ResolvedParameter>,
    pub estimate_set: IndexSet<String>,
    pub model:        ir::Model,
    pub warnings:     Vec<ResolverWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolverWarning {
    KickedFromEstimate { name: String, by: ValueSource },
    /// The active scenario set `name = scenario_value`, but a
    /// higher-precedence source (`by` — `--fixed-cli` or `--fixed-file`,
    /// per spec §1.3 ordering only `--fixed-cli` can actually beat
    /// scenario today) overrode it to `new_value`. Surfaced on stderr
    /// at resolve time and threaded into run.json's
    /// `parameters_provenance.overrode_scenario` field. Not an error
    /// — CLI override of a scenario value is a legitimate quick-test
    /// workflow; the warning ensures the override is never silent.
    ScenarioOverridden {
        name: String,
        scenario: String,
        scenario_value: f64,
        by: ValueSource,
        new_value: f64,
    },
    /// `fit.toml` lists `name` in both `[fixed]` and `[estimate]`.
    /// The resolver treats `[fixed]` as winning (a parameter that is
    /// both fixed and estimated is a config bug; the conservative
    /// interpretation is "the user meant fixed"). Surfaced so the
    /// user fixes their toml.
    FixedEstimateOverlap { name: String },
}

impl ResolverWarning {
    /// Human-readable rendering for stderr.
    pub fn format(&self) -> String {
        match self {
            ResolverWarning::KickedFromEstimate { name, by } => {
                let source_clause = match by {
                    ValueSource::FixedCli => format!("--fixed {}", name),
                    ValueSource::FixedFile { path } => {
                        format!("--fixed-file {}", path.display())
                    }
                    other => format!("source {:?}", other),
                };
                format!(
                    "warning: {} removes `{}` from [estimate]; it will be \
                     pinned to its resolved value rather than inferred.",
                    source_clause, name)
            }
            ResolverWarning::ScenarioOverridden {
                name, scenario, scenario_value, by, new_value,
            } => {
                let source_clause = match by {
                    ValueSource::FixedCli => format!("--fixed {}={}", name, new_value),
                    ValueSource::FixedFile { path } => {
                        format!("--fixed-file {} ({}={})",
                            path.display(), name, new_value)
                    }
                    other => format!("source {:?} ({}={})",
                        other, name, new_value),
                };
                format!(
                    "warning: {} overrides scenario `{}` which would have \
                     set `{}` = {}.",
                    source_clause, scenario, name, scenario_value)
            }
            ResolverWarning::FixedEstimateOverlap { name } => {
                format!(
                    "warning: `{}` appears in both [fixed] and [estimate] in \
                     fit.toml; [fixed] wins. Remove `{}` from one of the \
                     blocks to silence this warning.",
                    name, name)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResolveError {
    UnknownParameter     { name: String, source: ValueSource, candidates: Vec<String> },
    NonFiniteValue       { name: String, value: f64, source: ValueSource },
    UnsetRequired        { name: String },
    SchemaMismatch       { path: PathBuf, msg: String },
    ScenarioNotFound     { name: String, available: Vec<String> },
    ExternalTableMissing { table: String },
    BoundsViolation      { name: String, value: f64, lo: f64, hi: f64 },
    NestedCompose        { name: String },
    /// One or more finite/bounds violations collected together. The
    /// post-resolution validation pass surfaces *all* violations at
    /// once rather than failing on the first, so a user fixing
    /// `--param beta=5 --param gamma=10` against bounded parameters
    /// sees both names in a single error and can fix both before
    /// re-running. Mirrors the legacy `validate_parameter_values`
    /// behaviour, which already collected violations into one
    /// message (pinned by
    /// `parameter_bounds_validation::multiple_oob_params_reported_in_one_error`).
    MultipleViolations(Vec<ResolveError>),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Single-quoted parameter / scenario / table names are the
            // established convention across the codebase (matches
            // `unknown parameter '{0}'` in sim error variants, the
            // legacy `resolve_run_model` messages, and the
            // integration-test assertions in `util.rs`'s test module).
            // Resolver diagnostics follow the same convention so a
            // user can grep stderr without worrying about quoting
            // style varying between code paths.
            ResolveError::UnknownParameter { name, source, candidates } => {
                let src_label = match source {
                    ValueSource::FixedCli => "--fixed".to_string(),
                    ValueSource::FixedFile { path } =>
                        format!("--fixed-file {}", path.display()),
                    ValueSource::FitTomlFixed => "fit.toml [fixed]".to_string(),
                    other => format!("{:?}", other),
                };
                write!(f,
                    "unknown parameter '{name}' from {src_label}.\n  \
                     Available parameters: {}",
                    if candidates.is_empty() { "(none)".to_string() }
                    else { candidates.join(", ") })
            }
            ResolveError::NonFiniteValue { name, value, source } => {
                // "not finite (NaN or ±∞)" mirrors the legacy
                // `validate_parameter_values` wording so existing
                // integration tests
                // (`parameter_bounds_validation::param_positive_infinity_errors`)
                // and the user's mental model both stay green.
                write!(f,
                    "parameter '{name}' = {value} is not finite (NaN or ±∞), \
                     resolved from {}.\n  \
                     Fix: supply a finite numeric value via --fixed, \
                     --fixed-file, or the scenario block.",
                    source.tag())
            }
            ResolveError::UnsetRequired { name } => {
                write!(f,
                    "parameter '{name}' has no value: no model default, no \
                     scenario, no --fit toml entry, no --fixed-file, no \
                     --fixed.\n  \
                     Fix: declare a default in the .camdl model, or pin via \
                     `--fixed {name}=<value>`.")
            }
            ResolveError::SchemaMismatch { path, msg } =>
                write!(f, "schema mismatch in {}: {}", path.display(), msg),
            ResolveError::ScenarioNotFound { name, available } => {
                write!(f, "scenario '{name}' not found in model.\n  Available: {}",
                    if available.is_empty() { "(none)".to_string() }
                    else { available.join(", ") })
            }
            ResolveError::ExternalTableMissing { table } => {
                write!(f,
                    "table '{table}' is declared as external() but --table \
                     {table}=<file> was not provided")
            }
            ResolveError::BoundsViolation { name, value, lo, hi } => {
                write!(f,
                    "parameter '{name}' = {value} is outside declared bounds \
                     [{lo}, {hi}].\n  \
                     Fix: either widen the bounds in the model, or supply a \
                     value within the declared range.")
            }
            ResolveError::NestedCompose { name } => {
                write!(f,
                    "nested compose is not supported. Scenario '{name}' \
                     referenced in compose = [...] itself uses compose.")
            }
            ResolveError::MultipleViolations(errs) => {
                let parts: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
                write!(f, "{}", parts.join("\n"))
            }
        }
    }
}

impl std::error::Error for ResolveError {}

// ─── Shared preset compose-walk ────────────────────────────────────────────────

/// Resolve a named scenario preset to its effective `params` map,
/// walking `compose = [...]` left-to-right and then applying the
/// parent's own `params` on top.
///
/// Single source of truth for "what params does scenario X set",
/// shared by [`resolve_parameters`] (the simulate / forward path) and
/// `fit::config_v2::FixedParams::{expand_from_scenario, resolve_with_model}`
/// (the fit `[fixed] from_scenario` path). Before gh#36 the fit path
/// re-implemented this as a bare copy of `preset.params`, silently
/// dropping every param inherited via `compose` — a fit against a
/// compose-based scenario then failed with "parameters neither
/// estimated nor fixed".
///
/// Semantics (matches the spec's compose ordering):
///   - composed sub-scenarios apply first, in list order;
///   - the parent's own `params` apply last and win on key collision;
///   - nested compose is rejected (a sub-scenario referenced in
///     `compose = [...]` may not itself use `compose`).
///
/// Returns an `IndexMap` so iteration order is deterministic
/// (compose-order, then parent keys) for downstream provenance / diff.
pub fn resolve_preset_params(
    model: &ir::Model,
    preset_name: &str,
) -> Result<IndexMap<String, f64>, ResolveError> {
    let available = || -> Vec<String> {
        model.presets.iter().map(|p| p.name.clone()).collect()
    };
    let preset = model.presets.iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| ResolveError::ScenarioNotFound {
            name: preset_name.to_string(),
            available: available(),
        })?;

    let mut params: IndexMap<String, f64> = IndexMap::new();
    for sc_name in &preset.compose {
        let sub = model.presets.iter().find(|p| p.name == *sc_name)
            .ok_or_else(|| ResolveError::ScenarioNotFound {
                name: sc_name.clone(),
                available: available(),
            })?;
        if !sub.compose.is_empty() {
            return Err(ResolveError::NestedCompose { name: sc_name.clone() });
        }
        for (k, &v) in &sub.params {
            params.insert(k.clone(), v);
        }
    }
    // Parent's own params apply last → win on key collision.
    for (k, &v) in &preset.params {
        params.insert(k.clone(), v);
    }
    Ok(params)
}

/// The composed `scale` overlay a named preset applies, walking its `compose`
/// chain: each composed sub-preset's `scale`, then the parent's own `scale`, in
/// resolver-application order (parent last). Sub-presets may not themselves
/// compose (`NestedCompose`).
///
/// The single authority for "which parameters does this preset scale," shared by
/// the value resolver's tier-4 application (so the scaling is applied) and the
/// scenario×{draws,sweep} collision guards' footprint (so a collision is
/// caught). Mirrors `resolve_preset_params` for `set`. Before this was shared,
/// the footprint saw only the parent's `scale.keys()` while the resolver also
/// applied composed sub-preset scale — so a swept parameter scaled by a COMPOSED
/// sub-preset slipped past the guard and was then silently overwritten (the grid
/// collapsed to one value, mislabeled across distinct `sweep:` columns).
pub fn composed_preset_scale(
    model: &ir::Model,
    preset_name: &str,
) -> Result<Vec<(String, f64)>, ResolveError> {
    let available = || -> Vec<String> {
        model.presets.iter().map(|p| p.name.clone()).collect()
    };
    let preset = model.presets.iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| ResolveError::ScenarioNotFound {
            name: preset_name.to_string(),
            available: available(),
        })?;
    let mut out: Vec<(String, f64)> = Vec::new();
    for sc_name in &preset.compose {
        let sub = model.presets.iter().find(|p| p.name == *sc_name)
            .ok_or_else(|| ResolveError::ScenarioNotFound {
                name: sc_name.clone(),
                available: available(),
            })?;
        if !sub.compose.is_empty() {
            return Err(ResolveError::NestedCompose { name: sc_name.clone() });
        }
        out.extend(sub.scale.iter().map(|(k, &v)| (k.clone(), v)));
    }
    out.extend(preset.scale.iter().map(|(k, &v)| (k.clone(), v)));
    Ok(out)
}

/// The simulation horizon a named scenario declares (`scenarios { x { simulate
/// { to = … } } }`), walking its `compose` chain — or `None` when neither the
/// preset nor anything it composes declares one, in which case the cell runs to
/// `model.simulation.t_end` (gh#561).
///
/// **The single authority for "what horizon does this scenario run to."** Every
/// consumer goes through here — the window (`util::apply_scenario_horizon`), the
/// run identity (`batch::ResolvedEntry::t_end`), and the two guards that refuse a
/// horizon where reductions are differenced (`fit::contrasts`, `fit::predict`).
/// Before this existed, four sites each did their own `model.presets` lookup and
/// agreed only by coincidence; a window/identity divergence is a silent-wrong
/// (the store would serve one trajectory under another's key), so it is worth
/// making unrepresentable rather than merely absent.
///
/// Composition mirrors [`resolve_preset_params`] and [`composed_preset_scale`]
/// exactly — composed sub-scenarios first in list order, the parent's own value
/// last and winning — because a horizon is a preset field like any other and
/// making it the one field that does NOT compose would be its own surprise.
/// Nested compose is rejected, as it is for `set` and `scale`.
pub fn composed_preset_t_end(
    model: &ir::Model,
    preset_name: &str,
) -> Result<Option<f64>, ResolveError> {
    let available = || -> Vec<String> {
        model.presets.iter().map(|p| p.name.clone()).collect()
    };
    let preset = model.presets.iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| ResolveError::ScenarioNotFound {
            name: preset_name.to_string(),
            available: available(),
        })?;
    let mut out: Option<f64> = None;
    for sc_name in &preset.compose {
        let sub = model.presets.iter().find(|p| p.name == *sc_name)
            .ok_or_else(|| ResolveError::ScenarioNotFound {
                name: sc_name.clone(),
                available: available(),
            })?;
        if !sub.compose.is_empty() {
            return Err(ResolveError::NestedCompose { name: sc_name.clone() });
        }
        if let Some(t) = sub.t_end {
            out = Some(t);
        }
    }
    // The parent's own `to` applies last → wins on collision.
    if let Some(t) = preset.t_end {
        out = Some(t);
    }
    Ok(out)
}

/// The horizon a cell actually runs to under `scenario`: the scenario's declared
/// horizon if it has one (composed, per [`composed_preset_t_end`]), else the
/// model's. `None` scenario, an inline ad-hoc patch, or a name that is not a
/// model preset (the implicit `baseline` / `fitted`) all resolve to the model's.
///
/// The resolved-value form every consumer wants; see [`composed_preset_t_end`]
/// for why this is one function and not four.
/// Errors rather than falling back: a `NestedCompose` (or a compose list naming
/// a preset that does not exist) must NOT resolve to "the model horizon". Both
/// guards that refuse an unhonourable horizon read this, so a swallowed error
/// would make them silently compare against the wrong number and pass — a guard
/// that does not guard, which is the class of defect this whole change is about.
pub fn effective_horizon(
    model: &ir::Model,
    scenario: Option<&str>,
) -> Result<f64, ResolveError> {
    let model_end = model.simulation.t_end;
    let Some(name) = scenario else { return Ok(model_end) };
    // A name that is not a model preset is the implicit `baseline` / `fitted`
    // sentinel or an inline ad-hoc patch: no declared horizon, no error.
    if !model.presets.iter().any(|p| p.name == name) {
        return Ok(model_end);
    }
    Ok(composed_preset_t_end(model, name)?.unwrap_or(model_end))
}

/// The set of parameter names a scenario reference touches — its `set` ∪
/// `scale` ∪ composed-preset keys for a named preset, or its inline `params`
/// keys for an ad-hoc patch.
///
/// Single source of truth for "which parameters does scenario X pin/scale",
/// shared by the engine's explicit-`--draws` collision guard
/// ([`crate::engine`]) and `fit predict`'s scenario×sweep collision guard, so
/// the two guards can never disagree about a scenario's footprint. A `Named`
/// reference that is not a model preset (e.g. the implicit `baseline`) touches
/// nothing → an empty set (no collision possible).
pub fn scenario_param_footprint(
    model: &ir::Model,
    scenario: &crate::sim_job::ScenarioRef,
) -> Result<std::collections::BTreeSet<String>, String> {
    use crate::sim_job::ScenarioRef;
    let mut keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    match scenario {
        // Inline ad-hoc patch: only its `set` params (inline scenarios carry no
        // `scale`).
        ScenarioRef::Inline { params, .. } => {
            keys.extend(params.keys().cloned());
        }
        ScenarioRef::Named(name) => {
            // A name that is not a model preset sets nothing — empty footprint.
            if model.presets.iter().all(|p| p.name != *name) {
                return Ok(keys);
            }
            // `set` keys ∪ `scale` keys, BOTH walking the `compose` chain via the
            // SAME authorities the resolver applies (`resolve_preset_params` for
            // set, `composed_preset_scale` for scale) — so a swept parameter
            // scaled by a COMPOSED sub-preset is caught, not just one the parent
            // scales directly.
            keys.extend(
                resolve_preset_params(model, name)
                    .map_err(|e| e.to_string())?
                    .into_keys(),
            );
            keys.extend(
                composed_preset_scale(model, name)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|(k, _)| k),
            );
        }
    }
    Ok(keys)
}

// ─── Entry point ──────────────────────────────────────────────────────────────

/// Resolve a `ParameterInputs` to a `ResolvedParameters`, walking the
/// 5-tier precedence chain and recording provenance.
///
/// Side effect: the returned `ResolvedParameters.model` carries the
/// mutated `parameters[*].value` fields (and any scenario-applied
/// `interventions` filter + filled-in external tables). This is the
/// shape downstream `CompiledModel::new(model)` expects.
pub fn resolve_parameters<'a>(
    inputs: ParameterInputs<'a>,
) -> Result<ResolvedParameters, ResolveError> {
    let mut model = inputs.model.clone();
    let mut warnings: Vec<ResolverWarning> = Vec::new();

    // ── Tier 1: model defaults (already in model.parameters) ────────────
    //
    // No mutation needed — the IR carries `p.value` straight from the
    // DSL. The resolver layers tiers 2..5 on top.
    //
    // Track each parameter's *current* source as we walk tiers. The
    // map starts with whatever the IR supplied (ModelDefault for
    // params with a value, sentinel "unset" for those without).
    let mut current_source: HashMap<String, Option<ValueSource>> =
        model.parameters.iter()
            .map(|p| (p.name.clone(),
                if p.value.resolved_value().is_some() { Some(ValueSource::ModelDefault) } else { None }))
            .collect();

    let model_param_set: HashSet<String> = model.parameters.iter()
        .map(|p| p.name.clone()).collect();
    let model_param_names: Vec<String> = model.parameters.iter()
        .map(|p| p.name.clone()).collect();

    // Pre-resolve the scenario preset (and recursively-composed
    // sub-scenarios) so we know which intervention enable/disable
    // names to apply *and* which params/scales to layer at tier 4.
    // The intervention filter applies regardless of tier ordering
    // because it modifies `model.interventions`, not parameter
    // values.
    let scenario_name = inputs.scenario.map(|s| s.to_string());
    // The label used for scenario-tier provenance + warnings: the named
    // preset's name, or (when an inline ad-hoc scenario supplies
    // `set`/`scale`) the inline scenario's display name. `None` means no
    // scenario contributes parameter values at tier 4 (a bare baseline /
    // adhoc enable-disable-only patch).
    let (scenario_enable, scenario_disable, scenario_params, scenario_scale,
         scenario_label):
        (Vec<String>, Vec<String>, Vec<(String, f64)>, Vec<(String, f64)>,
         Option<String>) =
        if let Some(name) = scenario_name.as_deref() {
            let preset = model.presets.iter().find(|p| p.name == name)
                .ok_or_else(|| ResolveError::ScenarioNotFound {
                    name: name.to_string(),
                    available: model.presets.iter().map(|p| p.name.clone()).collect(),
                })?
                .clone();
            let mut composed_enable: Vec<String> = Vec::new();
            let mut composed_disable: Vec<String> = Vec::new();
            for sc_name in &preset.compose {
                let sub = model.presets.iter().find(|p| p.name == *sc_name)
                    .ok_or_else(|| ResolveError::ScenarioNotFound {
                        name: sc_name.clone(),
                        available: model.presets.iter().map(|p| p.name.clone()).collect(),
                    })?;
                if !sub.compose.is_empty() {
                    return Err(ResolveError::NestedCompose { name: sc_name.clone() });
                }
                composed_enable.extend(sub.enable.clone());
                composed_disable.extend(sub.disable.clone());
            }
            composed_enable.extend(preset.enable.clone());
            composed_disable.extend(preset.disable.clone());
            // Scale walks the same `compose` chain via the shared authority, so
            // the value applied here and the collision guards' footprint
            // (`composed_preset_scale`) can never disagree about which params a
            // composed scenario scales.
            let composed_scale: Vec<(String, f64)> = composed_preset_scale(&model, name)?;
            // gh#36: the params compose-walk is shared with the fit
            // `[fixed] from_scenario` path via `resolve_preset_params`.
            // Returns a deduped (parent-wins) map; applying it below is
            // equivalent to the old last-write-wins Vec since there are no
            // duplicate keys to re-trigger.
            let composed_params: Vec<(String, f64)> =
                resolve_preset_params(&model, name)?.into_iter().collect();
            (composed_enable, composed_disable, composed_params, composed_scale,
             Some(name.to_string()))
        } else {
            // Ad-hoc path: enable/disable always drive the filter; an
            // INLINE scenario's `set`/`scale` resolve at the SAME tier 4 as
            // a named preset, so an inline scenario is identical to the
            // equivalent preset (spec §1.3). The inline scenario
            // contributes a provenance label only when it actually carries
            // params/scale (an enable/disable-only or empty baseline patch
            // sets no parameter values, so it does not "win" any tier-4
            // slot and needs no Scenario provenance).
            let inline_set: Vec<(String, f64)> =
                inputs.scenario_inline_set.to_vec();
            let inline_scale: Vec<(String, f64)> =
                inputs.scenario_inline_scale.to_vec();
            let label = if !inline_set.is_empty() || !inline_scale.is_empty() {
                inputs.scenario_inline_name.map(|s| s.to_string())
            } else {
                None
            };
            (inputs.adhoc_enable.to_vec(), inputs.adhoc_disable.to_vec(),
             inline_set, inline_scale, label)
        };

    // ── Intervention filter (independent of value precedence) ───────────
    //
    // Must be called **unconditionally** — even when no scenario and no
    // adhoc enable/disable lists are present — because the filter is
    // what enforces "toggleable interventions default OFF" (only
    // `always_active` events survive an empty filter). Skipping the
    // call leaves toggleable interventions live when the user didn't
    // ask for them, which is a silent-wrong-answer bug
    // (`intervention_event_defaults::simulate_default_event_fires_intervention_does_not`
    // pins this contract).
    //
    // When `scenario_name.is_some()`, the composed scenario
    // enable/disable lists drive the filter; otherwise the
    // user-supplied `adhoc_enable`/`adhoc_disable` do (empty when the
    // user passed neither — which is precisely the case where
    // toggleables must drop).
    let (filter_enable, filter_disable): (Vec<String>, Vec<String>) =
        if scenario_name.is_some() {
            (scenario_enable.clone(), scenario_disable.clone())
        } else {
            (inputs.adhoc_enable.to_vec(), inputs.adhoc_disable.to_vec())
        };
    crate::util::apply_scenario_filter(
        &mut model, &filter_enable, &filter_disable)
        .map_err(|msg| ResolveError::SchemaMismatch {
            path: PathBuf::from("(scenario filter)"),
            msg,
        })?;

    // ── Tier 2: fit.toml [fixed] block ──────────────────────────────────
    for (name, &v) in inputs.fit_toml_fixed {
        if !model_param_set.contains(name) {
            return Err(ResolveError::UnknownParameter {
                name: name.clone(),
                source: ValueSource::FitTomlFixed,
                candidates: model_param_names.clone(),
            });
        }
        for p in &mut model.parameters {
            if p.name == *name {
                p.value = p.value.with_value(v);
                current_source.insert(name.clone(), Some(ValueSource::FitTomlFixed));
            }
        }
    }

    // ── Tier 3: --fixed-file <toml> (layered, last wins) ────────────────
    for path in inputs.fixed_files {
        let path_str = path.to_string_lossy().into_owned();
        let overrides = crate::util::load_params_toml(&path_str)
            .map_err(|msg| ResolveError::SchemaMismatch {
                path: path.clone(),
                msg,
            })?;
        for name in overrides.keys() {
            if !model_param_set.contains(name) {
                return Err(ResolveError::UnknownParameter {
                    name: name.clone(),
                    source: ValueSource::FixedFile { path: path.clone() },
                    candidates: model_param_names.clone(),
                });
            }
        }
        for p in &mut model.parameters {
            if let Some(&v) = overrides.get(&p.name) {
                p.value = p.value.with_value(v);
                current_source.insert(p.name.clone(),
                    Some(ValueSource::FixedFile { path: path.clone() }));
            }
        }
    }

    // ── Tier 3.5: draw row / sweep point overrides ──────────────────────
    //
    // Automated M-layer variation (spec §1.3): a sweep point or a draw
    // row sits between `--fixed-file` (tier 3) and scenario (tier 4). It
    // overrides the model default / fit-toml / file values, but a
    // scenario `set`/`scale` and `--fixed` (tiers 4/5) override it. This
    // is the structural difference from `--fixed`: a draw/sweep value is
    // counterfactual-modifiable; a `--fixed` value is the user's
    // assertion and is not.
    for (name, v) in inputs.point_overrides {
        if !model_param_set.contains(name) {
            return Err(ResolveError::UnknownParameter {
                name: name.clone(),
                source: ValueSource::SweepPoint,
                candidates: model_param_names.clone(),
            });
        }
        for p in &mut model.parameters {
            if p.name == *name {
                p.value = p.value.with_value(*v);
                current_source.insert(name.clone(), Some(ValueSource::SweepPoint));
            }
        }
    }

    // ── Tier 4: scenario params + scale ─────────────────────────────────
    //
    // Order is spec-§1.3-compliant: scenarios override `--fixed-file`
    // (the legacy `--params FILE`) and the draw/sweep tier. The
    // intervention filter for the scenario was applied earlier; only
    // `params` / `scale` happen here, layered on top of the file +
    // draw/sweep overrides.
    //
    // Tracking for [`ResolverWarning::ScenarioOverridden`]: record
    // what value the scenario *would have* set each named param to,
    // so the post-tier-5 sweep can detect silent overrides by
    // comparing final winner vs scenario intent. Stored as
    // (scenario_name, value) so we know which preset's intent was
    // overridden.
    let mut scenario_assigned: HashMap<String, (String, f64)> = HashMap::new();
    if let Some(name) = scenario_label.as_deref() {
        for (k, v) in &scenario_params {
            for p in &mut model.parameters {
                if p.name == *k {
                    p.value = p.value.with_value(*v);
                    current_source.insert(k.clone(),
                        Some(ValueSource::Scenario(name.to_string())));
                    scenario_assigned.insert(k.clone(),
                        (name.to_string(), *v));
                }
            }
        }
        for (k, factor) in &scenario_scale {
            for p in &mut model.parameters {
                if p.name == *k {
                    if let Some(v) = p.value.resolved_value() {
                        let scaled = v * factor;
                        p.value = p.value.with_value(scaled);
                        current_source.insert(k.clone(),
                            Some(ValueSource::Scenario(name.to_string())));
                        scenario_assigned.insert(k.clone(),
                            (name.to_string(), scaled));
                    }
                }
            }
        }
    }

    // ── Tier 5: --fixed NAME=VALUE (highest) ────────────────────────────
    for (name, v) in inputs.fixed_cli {
        if !model_param_set.contains(name) {
            return Err(ResolveError::UnknownParameter {
                name: name.clone(),
                source: ValueSource::FixedCli,
                candidates: model_param_names.clone(),
            });
        }
        for p in &mut model.parameters {
            if p.name == *name {
                p.value = p.value.with_value(*v);
                current_source.insert(name.clone(), Some(ValueSource::FixedCli));
            }
        }
    }

    // ── fit.toml [fixed] ∩ [estimate] overlap (config-file bug) ─────────
    //
    // Per resolved decision B in the 2026-05-25 CLI UX proposal: a name
    // appearing in both blocks of the same fit.toml is treated as
    // `[fixed]` wins, with a warning emitted so the user fixes their
    // config. Caller-side mutual exclusion is *not* enforced because
    // some toml loaders accept the pathological case silently; this
    // surface emits a clear diagnostic instead of letting the bug
    // pass.
    for name in inputs.fit_toml_fixed.keys() {
        if inputs.fit_toml_estimate.contains(name) {
            warnings.push(ResolverWarning::FixedEstimateOverlap {
                name: name.clone(),
            });
        }
    }

    // ── Estimate-set kick-out + provenance assembly ─────────────────────
    //
    // Build the estimate set from the fit-toml input, then drop names
    // that fit-toml's own [fixed] block claimed — this preserves the
    // [fixed]-wins-over-[estimate] resolution from the overlap check
    // above, so the run.json provenance shows `role = "fixed"` for
    // overlapping names rather than `"estimated"`.
    let mut estimate_set: IndexSet<String> = inputs.fit_toml_estimate.clone();
    estimate_set.retain(|n| !inputs.fit_toml_fixed.contains_key(n));

    // A name is "kicked from [estimate]" if it appears in tier 4 or
    // tier 5 (CLI / file `--fixed*`) — those are user-explicit "pin
    // this" assertions. Tier 3 (fit.toml [fixed]) is a no-op here
    // because the toml's `[fixed]` block already excludes those
    // names from `[estimate]` at config-load time.
    let kicker_names: HashMap<String, ValueSource> = {
        let mut m: HashMap<String, ValueSource> = HashMap::new();
        for path in inputs.fixed_files {
            let path_str = path.to_string_lossy().into_owned();
            // We already validated the file; load it again for the
            // name list. The cost is negligible vs simulation, and
            // it keeps tier 4 / tier 5 source attribution explicit.
            if let Ok(overrides) = crate::util::load_params_toml(&path_str) {
                for name in overrides.keys() {
                    m.insert(name.clone(),
                        ValueSource::FixedFile { path: path.clone() });
                }
            }
        }
        for (name, _) in inputs.fixed_cli {
            m.insert(name.clone(), ValueSource::FixedCli);
        }
        m
    };

    let mut kicked: HashMap<String, ValueSource> = HashMap::new();
    estimate_set.retain(|name| {
        if let Some(by) = kicker_names.get(name) {
            warnings.push(ResolverWarning::KickedFromEstimate {
                name: name.clone(),
                by: by.clone(),
            });
            kicked.insert(name.clone(), by.clone());
            false
        } else {
            true
        }
    });

    // Assemble ResolvedParameter entries in declaration order.
    //
    // Two-phase validation:
    //   (1) `UnsetRequired` is fatal-on-first-hit — a param with no
    //       value at all is a structural error, not a "you typed the
    //       wrong number" error, and reporting later params alongside
    //       it would be noise.
    //   (2) `NonFiniteValue` and `BoundsViolation` are collected
    //       across all parameters and reported together via
    //       `MultipleViolations`, so a user fixing
    //       `--param a=NaN --param b=999` against bounded `a, b`
    //       sees both problems in one stderr block.
    //
    // This mirrors the legacy `validate_parameter_values` behaviour
    // (pinned by `parameter_bounds_validation::multiple_oob_params_*`).
    let mut params: Vec<ResolvedParameter> = Vec::with_capacity(model.parameters.len());
    let mut violations: Vec<ResolveError> = Vec::new();
    for p in &model.parameters {
        let Some(value) = p.value.resolved_value() else {
            return Err(ResolveError::UnsetRequired { name: p.name.clone() });
        };
        if !value.is_finite() {
            let source = current_source.get(&p.name)
                .and_then(|s| s.clone())
                .unwrap_or(ValueSource::ModelDefault);
            violations.push(ResolveError::NonFiniteValue {
                name: p.name.clone(),
                value,
                source,
            });
            continue;
        }
        if let Some((lo, hi)) = p.bounds() {
            if value < lo || value > hi {
                violations.push(ResolveError::BoundsViolation {
                    name: p.name.clone(),
                    value, lo, hi,
                });
                continue;
            }
        }
        let source = current_source.get(&p.name)
            .and_then(|s| s.clone())
            .unwrap_or(ValueSource::ModelDefault);
        let role = if estimate_set.contains(&p.name) {
            ParameterRole::Estimated
        } else if let Some(by) = kicked.get(&p.name) {
            ParameterRole::Fixed {
                reason: FixReason::KickedFromEstimate { by: by.clone() },
            }
        } else {
            ParameterRole::Fixed { reason: FixReason::NotInEstimate }
        };

        // Scenario-override visibility: if the scenario intended to
        // set this parameter to value S but a higher-precedence
        // source (currently only `--fixed-cli` given spec §1.3) won
        // with a different value V, record the override on the
        // provenance and emit a warning. Float-equality is the right
        // comparator here because the scenario writes an explicit
        // value (or scales the pre-tier-4 value by an exact factor)
        // — any deviation means a higher tier overwrote it.
        let overrode_scenario = scenario_assigned.get(&p.name)
            .and_then(|(scen_name, scen_value)| {
                let final_is_scenario = matches!(
                    &source, ValueSource::Scenario(_));
                if !final_is_scenario && (*scen_value != value) {
                    warnings.push(ResolverWarning::ScenarioOverridden {
                        name: p.name.clone(),
                        scenario: scen_name.clone(),
                        scenario_value: *scen_value,
                        by: source.clone(),
                        new_value: value,
                    });
                    Some(ScenarioOverride {
                        scenario: scen_name.clone(),
                        scenario_value: *scen_value,
                    })
                } else {
                    None
                }
            });

        params.push(ResolvedParameter {
            name: p.name.clone(),
            value,
            source,
            role,
            overrode_scenario,
        });
    }

    // Surface collected violations before doing anything downstream
    // (external tables / Ok return). One violation degrades to a
    // single-variant error so callers that pattern-match on a
    // specific variant (e.g. test asserting `BoundsViolation`) still
    // work; two or more roll up into `MultipleViolations`.
    match violations.len() {
        0 => {}
        1 => return Err(violations.into_iter().next().unwrap()),
        _ => return Err(ResolveError::MultipleViolations(violations)),
    }

    // ── External tables ─────────────────────────────────────────────────
    for table in &mut model.tables {
        if let TableSource::External { external: ref name } = table.source {
            let logical_name = name.clone();
            match inputs.table_files.get(&logical_name) {
                None => return Err(ResolveError::ExternalTableMissing {
                    table: logical_name,
                }),
                Some(path) => {
                    let path_str = path.to_string_lossy().into_owned();
                    let values = crate::util::load_table_file(&path_str)
                        .map_err(|msg| ResolveError::SchemaMismatch {
                            path: path.clone(),
                            msg,
                        })?;
                    table.source = TableSource::Inline { values };
                }
            }
        }
    }

    Ok(ResolvedParameters {
        params,
        estimate_set,
        model,
        warnings,
    })
}

/// Render and print every warning in `resolved.warnings` to stderr.
/// Subcommand wrappers call this once after `resolve_parameters`.
///
/// Also runs a structural cross-check between the
/// `ResolverWarning::ScenarioOverridden` warnings and the
/// `ResolvedParameter.overrode_scenario` field: every parameter
/// carrying `overrode_scenario = Some(_)` must have a matching
/// warning, and vice versa. A mismatch is a resolver-internal bug
/// (the warning was emitted without populating provenance, or
/// provenance was written without a stderr warning) — surfaced as
/// `debug_assert!` so it trips in tests but degrades gracefully in
/// release.
pub fn print_warnings(resolved: &ResolvedParameters) {
    for w in &resolved.warnings {
        eprintln!("{}", w.format());
    }

    // Structural agreement check (debug-only, zero cost in release):
    // warnings-of-override and `overrode_scenario` provenance must
    // name the same parameters. This catches a class of resolver
    // bugs where one side was added without the other.
    let warning_names: std::collections::HashSet<&str> =
        resolved.warnings.iter().filter_map(|w| match w {
            ResolverWarning::ScenarioOverridden { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect();
    let provenance_names: std::collections::HashSet<&str> =
        resolved.params.iter()
            .filter(|p| p.overrode_scenario.is_some())
            .map(|p| p.name.as_str())
            .collect();
    debug_assert_eq!(warning_names, provenance_names,
        "ScenarioOverridden warnings and overrode_scenario provenance \
         must name the same parameters");
}


// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ir::model::{InitialConditions, OutputConfig, OutputSchedule, Preset, SimulationConfig};
    use ir::parameter::Parameter;

    /// Minimal `ir::Model` for resolver tests. Parameters supplied via
    /// the argument; everything else is empty-but-valid.
    fn mk_model(parameters: Vec<Parameter>) -> ir::Model {
        ir::Model {
            ic_grad: Default::default(),
            name: "test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters,
            initial_conditions: InitialConditions::Explicit(HashMap::new()),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 1.0,
                time_semantics: "continuous".into(),
                dt: None,
                rng_seed: None,
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }

    fn mk_param(name: &str, value: Option<f64>) -> Parameter {
        Parameter {
            name: name.into(),
            value: match value {
                Some(v) => ir::parameter::ParamValue::Fixed { value: v },
                None => ir::parameter::ParamValue::Required,
            },
            param_kind: None,
            param_dim: None,
        }
    }

    fn mk_param_bounded(name: &str, value: Option<f64>, bounds: (f64, f64)) -> Parameter {
        Parameter {
            name: name.into(),
            value: ir::parameter::ParamValue::Estimated {
                init: value,
                bounds: Some(bounds),
                prior: ir::parameter::PriorSpec::Flat,
                transform: ir::parameter::Transform::Identity,
            },
            param_kind: None,
            param_dim: None,
        }
    }

    fn empty_inputs<'a>(model: &'a ir::Model,
                        fixed_cli: &'a [(String, f64)],
                        fixed_files: &'a [PathBuf],
                        fit_toml_fixed: &'a IndexMap<String, f64>,
                        fit_toml_estimate: &'a IndexSet<String>) -> ParameterInputs<'a> {
        // Static empties (lifetimes work out because we pass refs to
        // owned containers held by the caller).
        ParameterInputs {
            model,
            scenario: None,
            adhoc_enable: &[],
            adhoc_disable: &[],
            scenario_inline_name: None,
            scenario_inline_set: &[],
            scenario_inline_scale: &[],
            point_overrides: &[],
            fixed_cli,
            fixed_files,
            fit_toml_fixed,
            fit_toml_estimate,
            table_files: &EMPTY_TABLES,
        }
    }

    use std::sync::LazyLock;
    static EMPTY_TABLES: LazyLock<HashMap<String, PathBuf>> =
        LazyLock::new(HashMap::new);

    // ── Tier 1: model defaults ──────────────────────────────────────────

    #[test]
    fn tier1_model_default_flows_through() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("resolution should succeed");
        assert_eq!(resolved.params.len(), 1);
        assert_eq!(resolved.params[0].name, "beta");
        assert_eq!(resolved.params[0].value, 0.5);
        assert_eq!(resolved.params[0].source, ValueSource::ModelDefault);
        assert!(matches!(resolved.params[0].role,
            ParameterRole::Fixed { reason: FixReason::NotInEstimate }));
    }

    #[test]
    fn tier1_unset_required_errors() {
        // No model default, no override → UnsetRequired.
        let model = mk_model(vec![mk_param("beta", None)]);
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .unwrap_err();
        assert!(matches!(err, ResolveError::UnsetRequired { ref name } if name == "beta"));
    }

    // ── Tier 2: scenario ────────────────────────────────────────────────

    #[test]
    fn tier2_scenario_overrides_model_default() {
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.9);
        model.presets.push(Preset {
            name: "baseline".into(),
            label: "baseline".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("baseline");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 0.9);
        assert!(matches!(&resolved.params[0].source,
            ValueSource::Scenario(name) if name == "baseline"));
    }

    #[test]
    fn tier2_scenario_not_found_errors() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("nonesuch");
        let err = resolve_parameters(inputs).unwrap_err();
        assert!(matches!(err, ResolveError::ScenarioNotFound { ref name, .. } if name == "nonesuch"));
    }

    // ── resolve_preset_params: shared compose-walk (gh#36) ──────────────

    fn mk_preset(name: &str, params: &[(&str, f64)], compose: &[&str]) -> Preset {
        let mut p = HashMap::new();
        for (k, v) in params { p.insert((*k).to_string(), *v); }
        Preset {
            name: name.into(),
            label: name.into(),
            params: p,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: compose.iter().map(|s| s.to_string()).collect(),
            t_end: None,
        }
    }

    fn mk_preset_t_end(name: &str, t_end: Option<f64>, compose: &[&str]) -> Preset {
        let mut p = mk_preset(name, &[], compose);
        p.t_end = t_end;
        p
    }

    // ── The horizon authority (gh#561) ──────────────────────────────────────
    //
    // These mirror `resolve_preset_params`'s tests one-for-one, because the
    // horizon composes by the SAME rule and the whole argument for composing it
    // (rather than refusing) is that it behaves like its siblings. A divergence
    // here is a silent-wrong: the window and the run identity both read these,
    // so a wrong answer runs one trajectory and files it under another's key.

    #[test]
    fn composed_t_end_walks_compose() {
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset_t_end("child", Some(200.0), &[]));
        model.presets.push(mk_preset_t_end("parent", None, &["child"]));
        assert_eq!(composed_preset_t_end(&model, "parent").expect("ok"), Some(200.0));
    }

    #[test]
    fn composed_t_end_parent_wins_over_composed_member() {
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset_t_end("child", Some(200.0), &[]));
        model.presets.push(mk_preset_t_end("parent", Some(75.0), &["child"]));
        assert_eq!(
            composed_preset_t_end(&model, "parent").expect("ok"),
            Some(75.0),
            "the parent's own `to` applies last and wins, exactly as its own \
             `set` beats a composed member's"
        );
    }

    #[test]
    fn composed_t_end_last_member_in_list_wins() {
        // The multi-member case: two composed members each declaring a horizon.
        // Last in list order wins, matching `resolve_preset_params`'s
        // overwrite-on-duplicate. Order is the part a regression would silently
        // change, so both directions are pinned.
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset_t_end("short", Some(50.0), &[]));
        model.presets.push(mk_preset_t_end("long", Some(200.0), &[]));
        model.presets.push(mk_preset_t_end("a", None, &["short", "long"]));
        model.presets.push(mk_preset_t_end("b", None, &["long", "short"]));
        assert_eq!(composed_preset_t_end(&model, "a").expect("ok"), Some(200.0));
        assert_eq!(composed_preset_t_end(&model, "b").expect("ok"), Some(50.0));
    }

    #[test]
    fn composed_t_end_rejects_nested_compose() {
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset_t_end("leaf", Some(200.0), &[]));
        model.presets.push(mk_preset_t_end("mid", None, &["leaf"]));
        model.presets.push(mk_preset_t_end("top", None, &["mid"]));
        assert!(
            matches!(
                composed_preset_t_end(&model, "top"),
                Err(ResolveError::NestedCompose { .. })
            ),
            "nested compose must error here exactly as it does for set/scale — \
             NOT resolve to the model horizon, which would make the guards that \
             read this compare against the wrong number and pass"
        );
    }

    #[test]
    fn effective_horizon_falls_back_and_propagates() {
        let mut model = mk_model(vec![]);
        model.simulation.t_end = 100.0;
        model.presets.push(mk_preset_t_end("plain", None, &[]));
        model.presets.push(mk_preset_t_end("declared", Some(160.0), &[]));
        model.presets.push(mk_preset_t_end("leaf", Some(200.0), &[]));
        model.presets.push(mk_preset_t_end("mid", None, &["leaf"]));
        model.presets.push(mk_preset_t_end("top", None, &["mid"]));

        // No scenario, and the implicit sentinels that name no preset, take the
        // model horizon.
        assert_eq!(effective_horizon(&model, None).expect("ok"), 100.0);
        assert_eq!(effective_horizon(&model, Some("baseline")).expect("ok"), 100.0);
        // A preset declaring nothing also takes it — this is what makes the
        // change re-key nothing for models that don't use the feature.
        assert_eq!(effective_horizon(&model, Some("plain")).expect("ok"), 100.0);
        assert_eq!(effective_horizon(&model, Some("declared")).expect("ok"), 160.0);
        assert_eq!(effective_horizon(&model, Some("mid")).expect("ok"), 200.0);
        // And an error propagates rather than degrading to the model horizon.
        assert!(effective_horizon(&model, Some("top")).is_err());
    }

    #[test]
    fn resolve_preset_params_walks_compose() {
        // gh#36: a compose-based parent inherits the composed child's params
        // plus its own. This is the shared helper the fit `[fixed]
        // from_scenario` path and the simulate path both call.
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset("child", &[("gamma", 0.1), ("N0", 1000.0)], &[]));
        model.presets.push(mk_preset("parent", &[("beta", 0.3)], &["child"]));
        let resolved = resolve_preset_params(&model, "parent").expect("ok");
        assert_eq!(resolved.get("gamma"), Some(&0.1));
        assert_eq!(resolved.get("N0"), Some(&1000.0));
        assert_eq!(resolved.get("beta"), Some(&0.3));
        assert_eq!(resolved.len(), 3);
    }

    #[test]
    fn footprint_includes_composed_sub_preset_scale() {
        // gh#322 review (silent-wrong): a scenario that COMPOSES a sub-preset
        // which SCALES `k` must report `k` in its footprint — else `--sweep
        // k=…` passes the collision guard, then the resolver applies the
        // composed scale and silently overwrites the swept value, collapsing
        // the grid to one value mislabeled across distinct `sweep:` columns.
        // The footprint must walk the same `compose` chain the resolver does
        // (`composed_preset_scale`), not just the parent's own `scale`.
        use crate::sim_job::ScenarioRef;
        let mut model = mk_model(vec![]);
        // A sub-preset that scales `k`.
        let mut sub_scale = HashMap::new();
        sub_scale.insert("k".to_string(), 2.0);
        model.presets.push(Preset {
            name: "scale_k".into(),
            label: "scale_k".into(),
            params: HashMap::new(),
            scale: sub_scale,
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        // A parent that COMPOSES `scale_k` and sets `beta` (its own scale empty).
        let mut parent_params = HashMap::new();
        parent_params.insert("beta".to_string(), 0.3);
        model.presets.push(Preset {
            name: "combo".into(),
            label: "combo".into(),
            params: parent_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec!["scale_k".into()],
            t_end: None,
        });
        let fp =
            scenario_param_footprint(&model, &ScenarioRef::Named("combo".into())).unwrap();
        assert!(
            fp.contains("k"),
            "footprint must include `k` scaled by the COMPOSED sub-preset; got {fp:?}"
        );
        assert!(fp.contains("beta"), "footprint includes the parent's set param; got {fp:?}");
    }

    #[test]
    fn resolve_preset_params_parent_wins_on_collision() {
        // Parent's own param overrides the composed child's on key collision.
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset("child", &[("gamma", 0.1)], &[]));
        model.presets.push(mk_preset("parent", &[("gamma", 0.2)], &["child"]));
        let resolved = resolve_preset_params(&model, "parent").expect("ok");
        assert_eq!(resolved.get("gamma"), Some(&0.2));
    }

    #[test]
    fn resolve_preset_params_rejects_nested_compose() {
        let mut model = mk_model(vec![]);
        model.presets.push(mk_preset("leaf", &[], &[]));
        model.presets.push(mk_preset("mid", &[("gamma", 0.1)], &["leaf"]));
        model.presets.push(mk_preset("parent", &[("beta", 0.3)], &["mid"]));
        let err = resolve_preset_params(&model, "parent").unwrap_err();
        assert!(matches!(err, ResolveError::NestedCompose { ref name } if name == "mid"));
    }

    #[test]
    fn resolve_preset_params_not_found_errors() {
        let model = mk_model(vec![]);
        let err = resolve_preset_params(&model, "nope").unwrap_err();
        assert!(matches!(err, ResolveError::ScenarioNotFound { ref name, .. } if name == "nope"));
    }

    // ── Tier 3: fit.toml [fixed] ────────────────────────────────────────

    #[test]
    fn tier3_fit_toml_fixed_overrides_model_default() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let fte = IndexSet::new();
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        assert_eq!(resolved.params[0].value, 0.7);
        assert_eq!(resolved.params[0].source, ValueSource::FitTomlFixed);
    }

    #[test]
    fn tier3_unknown_param_in_fit_toml_errors() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("typo".into(), 0.7);
        let fte = IndexSet::new();
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .unwrap_err();
        assert!(matches!(err, ResolveError::UnknownParameter { ref name, .. } if name == "typo"));
    }

    // ── Tier 5: --fixed CLI (highest) ───────────────────────────────────

    #[test]
    fn tier5_fixed_cli_overrides_everything() {
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.9);
        model.presets.push(Preset {
            name: "baseline".into(),
            label: "baseline".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![("beta".to_string(), 1.1)];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("baseline");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 1.1);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
    }

    #[test]
    fn tier5_unknown_cli_param_errors() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![("typo".to_string(), 0.7)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .unwrap_err();
        assert!(matches!(err, ResolveError::UnknownParameter { ref name, ref source, .. }
            if name == "typo" && *source == ValueSource::FixedCli));
    }

    // ── [estimate] kick-out ─────────────────────────────────────────────

    #[test]
    fn cli_fixed_kicks_out_of_estimate_with_warning() {
        let model = mk_model(vec![
            mk_param("beta", Some(0.5)),
            mk_param("gamma", Some(0.1)),
        ]);
        let fcli = vec![("gamma".to_string(), 0.2)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("beta".into());
        fte.insert("gamma".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");

        // beta stayed estimated; gamma got kicked.
        assert!(resolved.estimate_set.contains("beta"));
        assert!(!resolved.estimate_set.contains("gamma"));
        let gamma = resolved.params.iter().find(|p| p.name == "gamma").unwrap();
        assert!(matches!(&gamma.role,
            ParameterRole::Fixed { reason: FixReason::KickedFromEstimate { by } }
            if *by == ValueSource::FixedCli));
        assert_eq!(resolved.warnings.len(), 1);
        match &resolved.warnings[0] {
            ResolverWarning::KickedFromEstimate { name, by } => {
                assert_eq!(name, "gamma");
                assert_eq!(*by, ValueSource::FixedCli);
            }
            other => panic!("expected KickedFromEstimate, got {:?}", other),
        }
    }

    #[test]
    fn fit_toml_fixed_does_not_emit_kickedfromestimate_warning() {
        // Tier-3 (fit-toml `[fixed]`) does NOT emit a
        // `KickedFromEstimate` warning even when it overlaps
        // `[estimate]`. The dedicated `FixedEstimateOverlap`
        // warning handles that case (see
        // `fit_toml_fixed_estimate_overlap_warns_and_fixed_wins`);
        // the kick-out warning is reserved for tier-4 / tier-5
        // CLI-side overrides.
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("beta".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        // No `KickedFromEstimate` warning in the resolver output.
        let has_kicked = resolved.warnings.iter().any(|w|
            matches!(w, ResolverWarning::KickedFromEstimate { .. }));
        assert!(!has_kicked,
            "tier-3 must not emit KickedFromEstimate; saw {:?}",
            resolved.warnings);
    }

    // ── Bounds + finite checks ──────────────────────────────────────────

    #[test]
    fn bounds_violation_errors() {
        let model = mk_model(vec![mk_param_bounded("beta", Some(0.5), (0.0, 1.0))]);
        let fcli = vec![("beta".to_string(), 2.0)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .unwrap_err();
        assert!(matches!(err, ResolveError::BoundsViolation {
            ref name, value, ..
        } if name == "beta" && (value - 2.0).abs() < 1e-12));
    }

    #[test]
    fn non_finite_value_errors() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![("beta".to_string(), f64::NAN)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .unwrap_err();
        assert!(matches!(err, ResolveError::NonFiniteValue { ref name, .. } if name == "beta"));
    }

    // ── Provenance round-trip ───────────────────────────────────────────

    // ── Spec-§1.3 precedence: scenario > --fixed-file > --fixed CLI ─────

    #[test]
    fn scenario_beats_fit_toml_fixed_per_spec_section_1_3() {
        // Spec §1.3 says: params.toml < scenario. The resolver
        // implements this — fit-toml [fixed] (tier 2) is overwritten
        // by scenario params (tier 4). Locked in by the integration
        // test `scenario_runtime_application::scenario_set_replaces_mu_value`.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.9);
        model.presets.push(Preset {
            name: "preset".into(),
            label: "preset".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("preset");
        let resolved = resolve_parameters(inputs).expect("ok");
        // Scenario value wins, not the fit-toml fixed value.
        assert_eq!(resolved.params[0].value, 0.9);
        assert!(matches!(&resolved.params[0].source,
            ValueSource::Scenario(name) if name == "preset"));
    }

    #[test]
    fn fixed_cli_beats_scenario_per_spec_section_1_3() {
        // Spec §1.3 says: scenario < --param CLI. `--fixed CLI`
        // (tier 5) must override scenario (tier 4).
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.9);
        model.presets.push(Preset {
            name: "preset".into(),
            label: "preset".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![("beta".to_string(), 1.5)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("preset");
        let resolved = resolve_parameters(inputs).expect("ok");
        // --fixed CLI wins over scenario.
        assert_eq!(resolved.params[0].value, 1.5);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
    }

    #[test]
    fn scenario_scale_multiplies_resolved_value_not_just_model_default() {
        // Scenario `scale` applies multiplicatively to whatever
        // value is currently in the slot. The order ensures that
        // tier 2 + tier 3 layered values feed into the multiplication.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_scale = HashMap::new();
        scen_scale.insert("beta".to_string(), 2.0);
        model.presets.push(Preset {
            name: "doubled".into(),
            label: "doubled".into(),
            params: HashMap::new(),
            scale: scen_scale,
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);  // tier 2 sets beta=0.7
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("doubled");
        let resolved = resolve_parameters(inputs).expect("ok");
        // 0.7 (fit_toml_fixed) × 2.0 (scale) = 1.4
        assert!((resolved.params[0].value - 1.4).abs() < 1e-12,
            "scenario scale must multiply tier-2/3 value; got {}",
            resolved.params[0].value);
    }

    // ── Draw/sweep tier: BELOW scenario, ABOVE fixed-file (spec §1.3) ───

    #[test]
    fn scenario_set_beats_draw_or_sweep_point() {
        // Spec §1.3: sweep point overrides < scenario params. A scenario
        // `set` must override a draw/sweep value on the same parameter.
        // RED before the draw/sweep tier existed: the draw rode in
        // `fixed_cli` (tier 5) and WON over scenario.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.9);
        model.presets.push(Preset {
            name: "preset".into(),
            label: "preset".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let point = vec![("beta".to_string(), 0.2)];
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("preset");
        inputs.point_overrides = &point;
        let resolved = resolve_parameters(inputs).expect("ok");
        // Scenario value wins over the draw/sweep point.
        assert_eq!(resolved.params[0].value, 0.9,
            "scenario `set` must beat a draw/sweep point (spec §1.3); got {}",
            resolved.params[0].value);
        assert!(matches!(&resolved.params[0].source,
            ValueSource::Scenario(name) if name == "preset"));
    }

    #[test]
    fn scenario_scale_beats_draw_or_sweep_point() {
        // A scenario `scale` multiplies the draw/sweep value and the
        // RESULT is the winner (source = Scenario): scale applies on top
        // of the draw tier, then nothing higher overrides it.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_scale = HashMap::new();
        scen_scale.insert("beta".to_string(), 2.0);
        model.presets.push(Preset {
            name: "doubled".into(),
            label: "doubled".into(),
            params: HashMap::new(),
            scale: scen_scale,
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let point = vec![("beta".to_string(), 0.2)]; // draw sets beta = 0.2
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("doubled");
        inputs.point_overrides = &point;
        let resolved = resolve_parameters(inputs).expect("ok");
        // 0.2 (draw) × 2.0 (scenario scale) = 0.4; scenario is the winner.
        assert!((resolved.params[0].value - 0.4).abs() < 1e-12,
            "scenario `scale` must multiply the draw/sweep value (0.2 × 2.0 = 0.4); got {}",
            resolved.params[0].value);
        assert!(matches!(&resolved.params[0].source,
            ValueSource::Scenario(name) if name == "doubled"));
    }

    #[test]
    fn draw_or_sweep_point_beats_fixed_file_and_model_default() {
        // A draw/sweep point overrides `--fixed-file` (tier 3) and the
        // model default, but not scenario/`--fixed` (those are absent
        // here, so the draw is the winner with source = SweepPoint).
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7); // tier 2 sets beta = 0.7
        let fcli = vec![];
        let ffiles = vec![];
        let fte = IndexSet::new();
        let point = vec![("beta".to_string(), 0.33)];
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.point_overrides = &point;
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 0.33);
        assert_eq!(resolved.params[0].source, ValueSource::SweepPoint);
    }

    #[test]
    fn fixed_cli_beats_draw_or_sweep_point() {
        // `--fixed NAME=V` (tier 5) must override a draw/sweep point
        // (tier ~3.5). Guards against demoting `--param` below draws.
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![("beta".to_string(), 1.5)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let point = vec![("beta".to_string(), 0.2)];
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.point_overrides = &point;
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 1.5);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
    }

    #[test]
    fn draw_or_sweep_unknown_param_errors() {
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let point = vec![("typo".to_string(), 0.2)];
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.point_overrides = &point;
        let err = resolve_parameters(inputs).unwrap_err();
        assert!(matches!(err, ResolveError::UnknownParameter { ref name, ref source, .. }
            if name == "typo" && *source == ValueSource::SweepPoint));
    }

    #[test]
    fn draw_or_sweep_does_not_kick_from_estimate() {
        // A draw/sweep point on an [estimate] parameter does NOT kick it
        // out of [estimate] — only user-explicit `--fixed{,-file}` do
        // (mirrors the scenario rule).
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("beta".into());
        let point = vec![("beta".to_string(), 0.2)];
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.point_overrides = &point;
        let resolved = resolve_parameters(inputs).expect("ok");
        assert!(resolved.estimate_set.contains("beta"),
            "draw/sweep tier must not kick a parameter from [estimate]");
        let has_kicked = resolved.warnings.iter().any(|w|
            matches!(w, ResolverWarning::KickedFromEstimate { .. }));
        assert!(!has_kicked);
    }

    #[test]
    fn resolved_model_carries_mutated_values() {
        // The `model` field in `ResolvedParameters` must carry the
        // post-resolution `parameters[i].value`. Downstream
        // `CompiledModel::new(model)` reads these.
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![("beta".to_string(), 0.9)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        let beta_in_model = resolved.model.parameters.iter()
            .find(|p| p.name == "beta").unwrap();
        assert_eq!(beta_in_model.value.resolved_value(), Some(0.9));
    }

    #[test]
    fn warning_format_is_actionable() {
        // The warning format must name the parameter and the source so
        // a user re-reading stderr can localise the kick-out.
        let model = mk_model(vec![mk_param("gamma", Some(0.1))]);
        let fcli = vec![("gamma".to_string(), 0.2)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("gamma".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        assert_eq!(resolved.warnings.len(), 1);
        let msg = resolved.warnings[0].format();
        assert!(msg.contains("gamma"), "warning must name `gamma`: {}", msg);
        assert!(msg.contains("--fixed"), "warning must mention --fixed: {}", msg);
        assert!(msg.contains("[estimate]"), "warning must mention [estimate]: {}", msg);

        // `print_warnings` is a thin wrapper; smoke-call it to confirm
        // no panic and to keep the symbol live.
        print_warnings(&resolved);
    }

    #[test]
    fn value_source_tag_is_stable() {
        // Tags are serialised verbatim into run.json; pin them.
        assert_eq!(ValueSource::ModelDefault.tag(), "model_default");
        assert_eq!(ValueSource::Scenario("x".into()).tag(), "scenario");
        assert_eq!(ValueSource::FitTomlFixed.tag(), "fit_toml_fixed");
        assert_eq!(ValueSource::FixedFile { path: PathBuf::from("p") }.tag(), "fixed_file");
        assert_eq!(ValueSource::SweepPoint.tag(), "sweep_point");
        assert_eq!(ValueSource::FixedCli.tag(), "fixed_cli");
    }

    #[test]
    fn provenance_distinguishes_sources() {
        let mut model = mk_model(vec![
            mk_param("a", Some(1.0)),  // ModelDefault
            mk_param("b", Some(1.0)),  // Scenario
            mk_param("c", Some(1.0)),  // FitTomlFixed
            mk_param("d", Some(1.0)),  // FixedCli
        ]);
        let mut scen_params = HashMap::new();
        scen_params.insert("b".to_string(), 2.0);
        model.presets.push(Preset {
            name: "preset".into(),
            label: "preset".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![("d".to_string(), 4.0)];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("c".into(), 3.0);
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("preset");
        let resolved = resolve_parameters(inputs).expect("ok");
        let by_name: HashMap<&str, &ResolvedParameter> =
            resolved.params.iter().map(|p| (p.name.as_str(), p)).collect();
        assert_eq!(by_name["a"].source, ValueSource::ModelDefault);
        assert!(matches!(&by_name["b"].source, ValueSource::Scenario(_)));
        assert_eq!(by_name["c"].source, ValueSource::FitTomlFixed);
        assert_eq!(by_name["d"].source, ValueSource::FixedCli);
    }

    // ── Scenario-override visibility ────────────────────────────────────

    #[test]
    fn fixed_cli_override_of_scenario_emits_warning_and_provenance() {
        // Scenario sets beta=0.3; --fixed beta=0.5 wins. The resolver
        // must emit a ScenarioOverridden warning AND record the
        // scenario's intended value on the parameter's
        // `overrode_scenario` field so a future reader sees what the
        // scenario *would* have set, even though the CLI overrode it.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.3);
        model.presets.push(Preset {
            name: "worst_case".into(),
            label: "worst_case".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![("beta".to_string(), 0.5)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");

        // CLI value wins.
        assert_eq!(resolved.params[0].value, 0.5);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);

        // Provenance carries the overridden scenario value.
        let beta = &resolved.params[0];
        assert_eq!(beta.overrode_scenario,
            Some(ScenarioOverride {
                scenario: "worst_case".into(),
                scenario_value: 0.3,
            }),
            "overrode_scenario must record scenario name + intended value");

        // Warning emitted.
        let scen_warns: Vec<_> = resolved.warnings.iter().filter(|w|
            matches!(w, ResolverWarning::ScenarioOverridden { .. })).collect();
        assert_eq!(scen_warns.len(), 1);
        match scen_warns[0] {
            ResolverWarning::ScenarioOverridden {
                name, scenario, scenario_value, by, new_value,
            } => {
                assert_eq!(name, "beta");
                assert_eq!(scenario, "worst_case");
                assert_eq!(*scenario_value, 0.3);
                assert_eq!(*by, ValueSource::FixedCli);
                assert_eq!(*new_value, 0.5);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn scenario_applied_cleanly_does_not_emit_override_warning() {
        // If --fixed-cli doesn't conflict with scenario, the warning
        // must NOT fire. Regression guard against a spurious warning
        // on the legitimate single-source case.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.3);
        model.presets.push(Preset {
            name: "worst_case".into(),
            label: "worst_case".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");

        assert_eq!(resolved.params[0].value, 0.3);
        assert!(matches!(&resolved.params[0].source,
            ValueSource::Scenario(name) if name == "worst_case"));
        assert_eq!(resolved.params[0].overrode_scenario, None);
        let has_scen_warn = resolved.warnings.iter().any(|w|
            matches!(w, ResolverWarning::ScenarioOverridden { .. }));
        assert!(!has_scen_warn,
            "no override warning when scenario wins cleanly; saw {:?}",
            resolved.warnings);
    }

    #[test]
    fn fixed_cli_matching_scenario_value_does_not_warn() {
        // Edge case: scenario sets beta=0.3 and --fixed beta=0.3 also.
        // Final value is 0.3 either way; the resolver records source
        // = FixedCli (last wins) but the warning should not fire
        // because the user got the value they asked for.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let mut scen_params = HashMap::new();
        scen_params.insert("beta".to_string(), 0.3);
        model.presets.push(Preset {
            name: "worst_case".into(),
            label: "worst_case".into(),
            params: scen_params,
            scale: HashMap::new(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
        });
        let fcli = vec![("beta".to_string(), 0.3)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");

        assert_eq!(resolved.params[0].value, 0.3);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
        assert_eq!(resolved.params[0].overrode_scenario, None,
            "no override visibility when values agree");
        let has_scen_warn = resolved.warnings.iter().any(|w|
            matches!(w, ResolverWarning::ScenarioOverridden { .. }));
        assert!(!has_scen_warn);
    }

    #[test]
    fn scenario_override_warning_formats_actionably() {
        // The stderr-printable form must name the parameter, the
        // scenario, the would-have value, and the actual value so a
        // user re-reading stderr can localise the override.
        let w = ResolverWarning::ScenarioOverridden {
            name: "beta".into(),
            scenario: "worst_case".into(),
            scenario_value: 0.3,
            by: ValueSource::FixedCli,
            new_value: 0.5,
        };
        let msg = w.format();
        assert!(msg.contains("--fixed beta=0.5"),
            "must show CLI flag and value: {}", msg);
        assert!(msg.contains("worst_case"),
            "must name scenario: {}", msg);
        assert!(msg.contains("0.3"),
            "must show scenario's intended value: {}", msg);
    }

    // ── fit.toml [fixed] ∩ [estimate] overlap (resolved decision B) ─────

    #[test]
    fn fit_toml_fixed_estimate_overlap_warns_and_fixed_wins() {
        // Pathological config: same name in both `[fixed]` and
        // `[estimate]`. Resolver must:
        //   1. Treat `[fixed]` as winning (parameter is Fixed, not
        //      Estimated).
        //   2. Emit a FixedEstimateOverlap warning so the user sees
        //      the config bug.
        let model = mk_model(vec![mk_param("beta", Some(0.5))]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("beta".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");

        // [fixed] won: not in estimate_set, role is Fixed.
        assert!(!resolved.estimate_set.contains("beta"),
            "beta must be removed from estimate_set when [fixed] wins");
        let beta = &resolved.params[0];
        assert_eq!(beta.value, 0.7);
        assert_eq!(beta.source, ValueSource::FitTomlFixed);
        assert!(matches!(&beta.role,
            ParameterRole::Fixed { reason: FixReason::NotInEstimate }),
            "[fixed]-wins-over-[estimate] yields NotInEstimate (overlap \
             is reported via the dedicated warning, not via FixReason)");

        // Warning emitted.
        let overlap_warns: Vec<_> = resolved.warnings.iter().filter(|w|
            matches!(w, ResolverWarning::FixedEstimateOverlap { .. })).collect();
        assert_eq!(overlap_warns.len(), 1);
        match overlap_warns[0] {
            ResolverWarning::FixedEstimateOverlap { name } => {
                assert_eq!(name, "beta");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn fixed_estimate_overlap_warning_formats_actionably() {
        let w = ResolverWarning::FixedEstimateOverlap {
            name: "gamma".into(),
        };
        let msg = w.format();
        assert!(msg.contains("gamma"), "must name the param: {}", msg);
        assert!(msg.contains("[fixed]"),
            "must mention [fixed] block: {}", msg);
        assert!(msg.contains("[estimate]"),
            "must mention [estimate] block: {}", msg);
    }

    #[test]
    fn no_overlap_means_no_overlap_warning() {
        // Disjoint [fixed] and [estimate] → no overlap warning.
        let model = mk_model(vec![
            mk_param("beta", Some(0.5)),
            mk_param("gamma", Some(0.1)),
        ]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("gamma".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        let has_overlap = resolved.warnings.iter().any(|w|
            matches!(w, ResolverWarning::FixedEstimateOverlap { .. }));
        assert!(!has_overlap);
        assert!(resolved.estimate_set.contains("gamma"));
    }

    // ── Intervention filter must run unconditionally ────────────────────

    fn mk_intervention(name: &str, always_active: bool) -> ir::intervention::Intervention {
        use ir::intervention::{Action, AddAction, InterventionSchedule};
        ir::intervention::Intervention {
            name: name.into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![10.0])),
            actions: vec![Action::Add(AddAction {
                compartment: "S".into(),
                count: ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.0 }),
            })],
            kind: if always_active { ir::intervention::InterventionKind::Event } else { ir::intervention::InterventionKind::Scenario },
        }
    }

    #[test]
    fn no_scenario_no_adhoc_still_drops_toggleable_interventions() {
        // Regression guard for the bug where the resolver skipped
        // `apply_scenario_filter` when scenario/adhoc were both
        // empty, leaving toggleable interventions live by default.
        // Contract: a `simulate` invocation with no `--scenario` and
        // no `--enable` must drop toggleable interventions; only
        // `always_active = true` (events) survive. This is the
        // resolver-level mirror of the
        // `intervention_event_defaults::simulate_default_event_fires_intervention_does_not`
        // integration test.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        model.interventions = vec![
            mk_intervention("event_a", true),   // always_active → must survive
            mk_intervention("interv_b", false), // toggleable → must drop
        ];

        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");

        let names: Vec<&str> = resolved.model.interventions.iter()
            .map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["event_a"],
            "toggleable interventions must drop without --enable / --scenario; \
             survived = {:?}", names);
    }

    #[test]
    fn adhoc_enable_keeps_named_toggleable_intervention() {
        // Counter-test: with adhoc_enable supplying the name,
        // the toggleable intervention is kept.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        model.interventions = vec![
            mk_intervention("event_a", true),
            mk_intervention("interv_b", false),
        ];

        let fcli = vec![];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let fte = IndexSet::new();
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        let adhoc_enable = vec!["interv_b".to_string()];
        inputs.adhoc_enable = &adhoc_enable;
        let resolved = resolve_parameters(inputs).expect("ok");

        let names: Vec<&str> = resolved.model.interventions.iter()
            .map(|iv| iv.name.as_str()).collect();
        // Order in `interventions` is preserved by `retain`, so
        // event_a still comes first; interv_b survives only because
        // it was explicitly enabled.
        assert!(names.contains(&"event_a"),
            "always_active event survives unconditionally");
        assert!(names.contains(&"interv_b"),
            "adhoc-enabled toggleable survives; got {:?}", names);
    }
}
