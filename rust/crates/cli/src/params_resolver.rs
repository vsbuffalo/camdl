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
//! ```text
//!   1. Model parameter default (`p.value` from DSL)
//!   1.5. A single-point `--starts` rule's values (`starts`): the file
//!        `from_params=FILE` names, or the estimate `from_mle=FIT` names
//!        (`profile` only)
//!   2. `fit.toml [fixed]` block (when present)
//!   3. `--fixed-file <toml>` (each file layered in order; later
//!      overrides earlier)
//!   3.5. Draw row / sweep point (`point_overrides`) — automated
//!        M-layer variation (a posterior/prior/uniform draw, an
//!        explicit draws file, or a sweep grid point)
//!   4. Scenario (`preset.params` + multiplicative `preset.scale`, or
//!      an inline ad-hoc scenario's `set`/`scale`)
//!   5. `--fixed NAME=VALUE` (highest)
//! ```
//!
//! The structural distinction between tiers 3.5 and 5: a draw/sweep
//! value is *automated M-layer variation* and is counterfactual-
//! modifiable, so a scenario `set`/`scale` (a counterfactual on M)
//! overrides it. A `--fixed` value is the user's *explicit assertion*
//! about a specific run and overrides everything, scenario included.
//!
//! Precedence orders values; it does not grant permission. A scenario
//! that touches a parameter a *user-authored* coordinate varies — a
//! sweep point set or a `--draws <file>` — is refused before the
//! resolver runs ([`scenario_coordinate_collision`], gh#572): tier 4
//! would discard every value the user listed. So tier 4 over tier 3.5
//! is reached only for distinct parameters and for generated draws
//! (`--draws posterior|prior|uniform`), where a scenario over the
//! draws is the intended counterfactual.
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

use crate::notice::{Notice, Notices, Severity};

// ─── The range a parameter is valid on ───────────────────────────────────────

/// The interval a parameter *kind* carries on its own — the range the declared
/// type promises when the model wrote no `in [lo, hi]`.
///
/// `probability` is the only kind whose type pins a finite interval
/// (`docs/camdl-language-spec.md` §4.1: `probability : ∈ [0, 1]`), and it is
/// also the only kind that selects `Transform::Logit`, the one transform that
/// needs a finite *upper* bound to have a scale at all: with `hi = ∞`,
/// `(x − lo)/(hi − lo)` is 0 for every finite `x` and `(hi − lo)/6` — the auto
/// `rw_sd` — is `∞` (gh#763).
///
/// Every other kind's range is either half-open (`rate`, `positive`, `count`,
/// `duration` → `[0, ∞)`) or unbounded (`real`, `instant`), which is exactly
/// what each consumer's existing `[0, ∞)`-or-unconstrained fallback already
/// expresses. They return `None` and keep their behaviour unchanged —
/// `Transform::Log` documents `hi = +∞` as well-formed, and `instant`/`real`
/// fall to `Transform::None` when a bound is infinite.
pub fn kind_support(kind: ir::parameter::ParamKind) -> Option<(f64, f64)> {
    use ir::parameter::ParamKind::*;
    // Exhaustive over ParamKind (no `_`): a new kind is a compile error here,
    // forcing an explicit range decision — the same gh#191 payoff
    // `fit::runner::derive_transform_with_bounds` takes.
    match kind {
        Probability => Some((0.0, 1.0)),
        Rate | Positive | Count | Duration | Real | Instant => None,
    }
}

/// The range a parameter is valid on: the model's declared `in [lo, hi]` when
/// present, else the range its declared kind carries ([`kind_support`]).
///
/// This is the interval the value check below and the fit's search box and
/// transform clamp must all agree on — a `probability` whose start is checked
/// against `[0, ∞)` but searched over `[0, 1]` gets silently clamped rather
/// than rejected.
///
/// The kind fallback does not change precedence: an explicit `in [...]` in the
/// model still wins over the type's range, and a fit.toml `[estimate].bounds`
/// (which `fit::runner::build_if2_params_from_specs` consults first) still
/// wins over both.
pub fn resolved_bounds(p: &ir::parameter::Parameter) -> Option<(f64, f64)> {
    p.bounds().or_else(|| p.param_kind.and_then(kind_support))
}

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// A single-point `--starts` rule's values (tier 1.5), naming where they
    /// came from.
    Starts(StartsOrigin),
    /// The start fall-back for an `[estimate]` parameter the model gives no
    /// value, seeded before resolution so the parameter has one: its
    /// `[estimate].start` (`drawn: false`), or a draw within its bounds when
    /// the entry gives no start (`drawn: true`). A reference point, not any
    /// chain's start: the starts rule draws those (`init_provenance`).
    /// Attributed after resolution by
    /// [`ResolvedParameters::attribute_seeded_starts`].
    EstimateStart { drawn: bool },
}

/// Where a single-point `--starts` rule's values came from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StartsOrigin {
    /// `--starts from_params=FILE`: a params file the user wrote.
    Params { path: PathBuf },
    /// `--starts from_mle=FIT`: a stored fit's point estimate; `fit` is the
    /// handle as given. Replacing one of these is a warning ([`severity`]).
    Mle { fit: String },
}

/// The values of a single-point `--starts` rule (tier 1.5, just above the
/// model default) and where they came from: `profile --starts
/// from_params=FILE` or `from_mle=FIT` ([`resolve_parameters_with_starts`]).
/// They are the base value of every parameter they name, so every tier above
/// replaces them, and each replacement is reported. A name the model does not
/// declare is ignored, not refused: the file is a starting point, and a fit's
/// estimate may carry names a later model dropped.
#[derive(Debug, Clone, Copy)]
pub struct StartsValues<'a> {
    pub origin: &'a StartsOrigin,
    pub values: &'a [(String, f64)],
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
            ValueSource::Starts(StartsOrigin::Params { .. }) => "starts_from_params",
            ValueSource::Starts(StartsOrigin::Mle { .. })    => "starts_from_mle",
            ValueSource::EstimateStart { drawn: false } => "estimate_start",
            ValueSource::EstimateStart { drawn: true }  => "estimate_start_drawn",
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

/// One replacement in the tier walk: a later tier set `name` to a different
/// value than the one a lower tier had supplied.
///
/// Recorded for every such replacement, in tier order, so no tier can
/// displace a value the user supplied without the user being told. Two
/// replacements are deliberately not overrides:
///
/// - **an equal value** — nothing changed, so there is nothing to report;
/// - **replacing the model default** — the default is what every other tier
///   exists to replace, so reporting it would print a line for every value a
///   params file or `--param` sets.
///
/// Each tier contributes one net value per parameter: a scenario's `set` and
/// its `scale` on the same parameter are one tier-4 value, and a name given
/// twice to `--param`/`--param-vec` is one tier-5 value.
#[derive(Debug, Clone, PartialEq)]
pub struct Override {
    pub name:     String,
    /// The lower tier's value and where it came from.
    pub replaced: (ValueSource, f64),
    /// The value that replaced it, and where that came from.
    pub by:       (ValueSource, f64),
}

/// What the resolver learned about its inputs that bears on how a
/// replacement is reported, but not on any value ([`severity`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolveContext {
    /// The params files (`--params` / `--fixed-file`) that carry a fit's
    /// `[provenance]` block, i.e. files whose values are a fit's estimate.
    pub fit_estimate_files: HashSet<PathBuf>,
}

/// How loudly to report `o`.
///
/// A warning when the replaced value is a fit's estimate (it came from a
/// params file carrying a fit `[provenance]` block, or from
/// `--starts from_mle=FIT`): the run then simulates
/// something other than the fit's estimate for that parameter, which is the
/// replacement most likely to be unintended. Every other replacement is one
/// the user asked for with a flag in this command (`--param`, `--scenario`, a
/// later params file, a sweep or draws), so it is info.
pub fn severity(o: &Override, ctx: &ResolveContext) -> Severity {
    match &o.replaced.0 {
        ValueSource::FixedFile { path } if ctx.fit_estimate_files.contains(path) =>
            Severity::Warning,
        ValueSource::Starts(StartsOrigin::Mle { .. }) => Severity::Warning,
        _ => Severity::Info,
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedParameter {
    pub name:   String,
    pub value:  f64,
    pub source: ValueSource,
    pub role:   ParameterRole,
    /// Every replacement of this parameter's value across the tiers, in tier
    /// order ([`Override`]). Empty when no tier displaced another's value.
    /// Recorded in `run.json`'s `parameters_provenance` and reported on
    /// stderr by [`notices`], so an override is auditable after stderr is
    /// gone.
    pub overrides: Vec<Override>,
}

#[derive(Debug, Clone)]
pub struct ResolvedParameters {
    pub params:       Vec<ResolvedParameter>,
    pub estimate_set: IndexSet<String>,
    pub model:        ir::Model,
    /// Names `[estimate]` lists that `--fixed`/`--fixed-file` pinned.
    pub pinned_over_estimate: Vec<PinnedOverEstimate>,
    pub context:      ResolveContext,
}

impl ResolvedParameters {
    /// Attribute the values a caller seeded into the model before resolution
    /// (`seeded`: name → [`ValueSource::EstimateStart`]) to that source. The
    /// resolver reads a seeded value as the model's own and records it as
    /// `model_default`; only a parameter still sourced there is relabelled,
    /// so a tier that replaced the seed keeps its own attribution.
    pub fn attribute_seeded_starts(&mut self, seeded: &HashMap<String, ValueSource>) {
        for rp in &mut self.params {
            if rp.source == ValueSource::ModelDefault {
                if let Some(src) = seeded.get(&rp.name) {
                    rp.source = src.clone();
                }
            }
        }
    }
}

/// A parameter the fit config lists in `[estimate]` that `--fixed` /
/// `--fixed-file` pins, so it is held at `value` and not estimated.
///
/// A layered override, not a contradiction: the command line is the user's
/// later, more specific say over the fit config, the same way `--fixed` over a
/// scenario value is. Reported at info ([`PinnedOverEstimate::notice`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PinnedOverEstimate {
    pub name:  String,
    /// `FixedCli` or `FixedFile` — the only sources that pin a name out of
    /// `[estimate]`.
    pub by:    ValueSource,
    /// The value that source gives the parameter.
    pub value: f64,
}

impl PinnedOverEstimate {
    /// The notice as a [`Notice`] — its stderr line and its record.
    pub fn notice(&self) -> Notice {
        let PinnedOverEstimate { name, by, value } = self;
        let flag = match by {
            ValueSource::FixedCli => format!("`--fixed {name}={value}` pins it"),
            ValueSource::FixedFile { path } => {
                format!("`--fixed-file {}` pins it at {value}", path.display())
            }
            other => format!("{} pins it at {value}", other.tag()),
        };
        Notice::info("fixed-over-estimate", format!(
            "{name}: {flag}, so it is not estimated (the fit config lists it in [estimate])"))
    }
}

#[derive(Debug, Clone)]
pub enum ResolveError {
    /// `constant_hint`: set when `name` is a model constant
    /// ([`ir::Model::constant_hint`]) — a typed `let`, which is not a
    /// parameter; the message then says how to make it configurable.
    UnknownParameter     { name: String, source: ValueSource, candidates: Vec<String>,
                           constant_hint: Option<String> },
    NonFiniteValue       { name: String, value: f64, source: ValueSource },
    UnsetRequired        { name: String },
    SchemaMismatch       { path: PathBuf, msg: String },
    ScenarioNotFound     { name: String, available: Vec<String> },
    ExternalTableMissing { table: String },
    BoundsViolation      { name: String, value: f64, lo: f64, hi: f64 },
    NestedCompose        { name: String },
    /// A fit config lists these names in both `[fixed]` and `[estimate]`
    /// ([`fixed_estimate_overlap_message`]).
    FixedEstimateOverlap { names: Vec<String> },
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
            ResolveError::UnknownParameter { name, source, candidates, constant_hint } => {
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
                    else { candidates.join(", ") })?;
                if let Some(hint) = constant_hint {
                    write!(f, "\n  hint: {hint}")?;
                }
                Ok(())
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
            ResolveError::FixedEstimateOverlap { names } => {
                let names: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
                write!(f, "{}", fixed_estimate_overlap_message(&names))
            }
            ResolveError::MultipleViolations(errs) => {
                let parts: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
                write!(f, "{}", parts.join("\n"))
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// The refusal for a fit config that lists `names` in both `[fixed]` and
/// `[estimate]`. A parameter is either estimated or given a value, never both,
/// and the file does not say which it means. One wording for every command:
/// `fit run`'s config validation and the resolver behind `profile` / `survey`.
pub fn fixed_estimate_overlap_message(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    let quoted = quoted.join(", ");
    let (noun, is, them) =
        if names.len() == 1 { ("parameter", "is", "it") } else { ("parameters", "are", "them") };
    format!(
        "{noun} {quoted} {is} in both [estimate] and [fixed]; a parameter is either estimated \
         or given a value, not both.\n  \
         Fix: remove {quoted} from [fixed] to estimate {them}, or from [estimate] to hold \
         {them} at {} [fixed] value.",
        if names.len() == 1 { "its" } else { "their" }
    )
}

// ─── Shared preset compose-walk ────────────────────────────────────────────────

/// Resolve a named scenario preset to its effective `params` map,
/// walking `compose = [...]` left-to-right and then applying the
/// parent's own `params` on top.
///
/// Single source of truth for "what params does scenario X set", read by
/// [`resolve_preset_delta`]. The values a scenario *gives* (its `set` with
/// its `scale` applied) are [`scenario_values`].
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

/// Everything a named preset changes about a run's inputs, after walking its
/// `compose` chain: the intervention filter lists, the `set` values, and the
/// `scale` factors. The horizon is resolved separately
/// ([`composed_preset_t_end`]) because it keys the `config` level, not the
/// `scenario` level.
#[derive(Debug, Clone, PartialEq)]
pub struct PresetDelta {
    /// Composed `enable` list: each composed sub-preset's, then the parent's.
    pub enable: Vec<String>,
    /// Composed `disable` list, same order as `enable`.
    pub disable: Vec<String>,
    /// Composed `set` values ([`resolve_preset_params`]; parent wins).
    pub params: IndexMap<String, f64>,
    /// Composed `scale` factors in application order
    /// ([`composed_preset_scale`]). Not deduplicated: a key scaled by a
    /// composed sub-preset *and* the parent is multiplied twice.
    pub scale: Vec<(String, f64)>,
}

/// Resolve a named preset to its effective [`PresetDelta`].
///
/// The single authority for "what does scenario X change," read by both the
/// value resolver ([`resolve_parameters`], which applies the delta) and the
/// run identity (`cas_sink::ResolvedEntry::from_preset`, which hashes it). The
/// two reading one function is what makes it impossible for a preset to
/// change the trajectory without changing the `run_id` (gh#573: the identity
/// used to read the preset's own literal fields, missing `scale` and every
/// field inherited through `compose`).
pub fn resolve_preset_delta(
    model: &ir::Model,
    preset_name: &str,
) -> Result<PresetDelta, ResolveError> {
    let available = || -> Vec<String> {
        model.presets.iter().map(|p| p.name.clone()).collect()
    };
    let preset = model.presets.iter()
        .find(|p| p.name == preset_name)
        .ok_or_else(|| ResolveError::ScenarioNotFound {
            name: preset_name.to_string(),
            available: available(),
        })?;
    let mut enable: Vec<String> = Vec::new();
    let mut disable: Vec<String> = Vec::new();
    for sc_name in &preset.compose {
        let sub = model.presets.iter().find(|p| p.name == *sc_name)
            .ok_or_else(|| ResolveError::ScenarioNotFound {
                name: sc_name.clone(),
                available: available(),
            })?;
        if !sub.compose.is_empty() {
            return Err(ResolveError::NestedCompose { name: sc_name.clone() });
        }
        enable.extend(sub.enable.iter().cloned());
        disable.extend(sub.disable.iter().cloned());
    }
    enable.extend(preset.enable.iter().cloned());
    disable.extend(preset.disable.iter().cloned());
    Ok(PresetDelta {
        enable,
        disable,
        params: resolve_preset_params(model, preset_name)?,
        scale: composed_preset_scale(model, preset_name)?,
    })
}

/// The simulation horizon a named scenario declares (`scenarios { x { simulate
/// { to = … } } }`), walking its `compose` chain — or `None` when neither the
/// preset nor anything it composes declares one, in which case the cell runs to
/// `model.simulation.t_end` (gh#561).
///
/// **The single authority for "what horizon does this scenario run to."** Every
/// consumer goes through here — the window (`util::apply_scenario_horizon`), the
/// run identity (`cas_sink::ResolvedEntry::t_end`), and the two guards that refuse a
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
/// read by [`scenario_coordinate_collision`] — behind both the engine's
/// sweep / draws-file guard and `fit predict`'s `--sweep` guard — so no two
/// commands can disagree about a scenario's footprint. A `Named`
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

// ─── Scenario × user-authored coordinate collisions ───────────────────────────

/// One thing a scenario does to one parameter, and the preset that does it
/// (the scenario itself, or the composed sub-preset the value came from).
#[derive(Debug, Clone, PartialEq)]
pub enum ScenarioParamAction {
    Set { value: f64, preset: String },
    Scale { factor: f64, preset: String },
}

/// How `scenario` touches `param`, for a diagnostic. Which parameters collide
/// is decided by [`scenario_param_footprint`]; this only labels them. A `set`
/// reports the composed effective value ([`resolve_preset_params`]) attributed
/// to the last preset in the compose chain that sets it; a `scale` reports
/// every preset in the chain that scales it (a factor applied twice is listed
/// twice, as [`composed_preset_scale`] applies it).
fn scenario_param_actions(
    model: &ir::Model,
    scenario: &crate::sim_job::ScenarioRef,
    param: &str,
) -> Result<Vec<ScenarioParamAction>, String> {
    use crate::sim_job::ScenarioRef;
    match scenario {
        ScenarioRef::Inline { name, params, .. } => Ok(params
            .get(param)
            .map(|&value| ScenarioParamAction::Set { value, preset: name.clone() })
            .into_iter()
            .collect()),
        ScenarioRef::Named(name) => {
            let Some(parent) = model.presets.iter().find(|p| p.name == *name) else {
                return Ok(Vec::new());
            };
            // Application order: composed sub-presets, then the parent. The
            // lookups cannot fail here — the footprint that flagged `param`
            // already resolved this chain — so a missing sub-preset is skipped.
            let chain: Vec<&ir::model::Preset> = parent
                .compose
                .iter()
                .filter_map(|c| model.presets.iter().find(|p| p.name == *c))
                .chain(std::iter::once(parent))
                .collect();
            let mut out = Vec::new();
            if let Some(&value) =
                resolve_preset_params(model, name).map_err(|e| e.to_string())?.get(param)
            {
                let preset = chain
                    .iter()
                    .rev()
                    .find(|p| p.params.contains_key(param))
                    .map_or_else(|| name.clone(), |p| p.name.clone());
                out.push(ScenarioParamAction::Set { value, preset });
            }
            for p in &chain {
                if let Some(&factor) = p.scale.get(param) {
                    out.push(ScenarioParamAction::Scale { factor, preset: p.name.clone() });
                }
            }
            Ok(out)
        }
    }
}

/// The user-authored coordinate a scenario is checked against: the values a
/// user wrote down, per parameter, which a scenario touching the same
/// parameter would override or rescale in every cell.
///
/// Generated draws (`--draws prior|uniform|posterior`) are deliberately not a
/// variant: a scenario over generated draws is the counterfactual those
/// commands exist to run.
#[derive(Debug, Clone, Copy)]
pub enum UserCoordinate<'a> {
    /// A `--sweep` / `--sweep-file` grid.
    Sweep,
    /// A `--draws <file.tsv>` the user authored.
    DrawsFile(&'a std::path::Path),
}

/// The distinct values each parameter takes across a coordinate's points, in
/// first-seen order — the `varied` input of [`scenario_coordinate_collision`].
pub fn coordinate_values<'a>(
    points: impl IntoIterator<Item = (&'a str, f64)>,
) -> IndexMap<String, Vec<f64>> {
    let mut out: IndexMap<String, Vec<f64>> = IndexMap::new();
    for (k, v) in points {
        let vals = out.entry(k.to_string()).or_default();
        if !vals.iter().any(|x| x.to_bits() == v.to_bits()) {
            vals.push(v);
        }
    }
    out
}

/// Refuse a scenario that touches a parameter a user-authored coordinate
/// varies. `Ok(None)` when the two are disjoint; `Ok(Some(message))` names
/// every colliding parameter, both sources, why the combination is refused,
/// and how to fix it.
///
/// This is collision policy, not precedence. The resolver's tiers still order
/// a scenario above a draw/sweep value, and that order is what applies to
/// *distinct* parameters and to generated draws. On the same parameter, a
/// `set` would override every value the user listed (the coordinate would not
/// vary it), and a `scale` would rescale every one; both are refused
/// (proposal 2026-06-27 §4, gh#572). The one formatter behind the engine's
/// guard ([`crate::engine::check_scenario_coordinate_collision`]) and
/// `fit predict`'s `--sweep` guard, so every command words it the same way.
pub fn scenario_coordinate_collision(
    model: &ir::Model,
    scenario: &crate::sim_job::ScenarioRef,
    coordinate: UserCoordinate<'_>,
    varied: &IndexMap<String, Vec<f64>>,
) -> Result<Option<String>, String> {
    let footprint = scenario_param_footprint(model, scenario)?;
    let clash: Vec<&str> = varied
        .keys()
        .map(|k| k.as_str())
        .filter(|k| footprint.contains(*k))
        .collect();
    if clash.is_empty() {
        return Ok(None);
    }

    let quoted = |ps: &[&str]| ps.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(", ");
    let (source, label) = match coordinate {
        UserCoordinate::Sweep => ("the sweep".to_string(), "sweep:"),
        UserCoordinate::DrawsFile(path) => {
            (format!("the draws file `{}`", path.display()), "draws file:")
        }
    };
    let scen = scenario.name();
    let mut msg = if clash.len() == 1 {
        format!("parameter {} is controlled by both {source} and scenario `{scen}`", quoted(&clash))
    } else {
        format!("parameters {} are controlled by both {source} and scenario `{scen}`", quoted(&clash))
    };

    const SHOWN: usize = 6;
    let width = label.len().max("scenario:".len()) + 2;
    let mut why: Vec<String> = Vec::new();
    for &param in &clash {
        let vals = &varied[param];
        let mut shown: Vec<String> = vals.iter().take(SHOWN).map(|v| v.to_string()).collect();
        if vals.len() > SHOWN {
            shown.push(format!("… ({} distinct values)", vals.len()));
        }
        msg.push_str(&format!("\n  {label:<width$}{param} = [{}]", shown.join(", ")));

        let actions = scenario_param_actions(model, scenario, param)?;
        let mut sets = false;
        for a in &actions {
            let (text, preset) = match a {
                ScenarioParamAction::Set { value, preset } => {
                    sets = true;
                    (format!("set {param} = {value}"), preset)
                }
                ScenarioParamAction::Scale { factor, preset } => {
                    (format!("scale {param} \u{d7} {factor}"), preset)
                }
            };
            let from = if preset.as_str() != scen {
                format!("    (from composed preset `{preset}`)")
            } else {
                String::new()
            };
            msg.push_str(&format!("\n  {:<width$}{text}{from}", "scenario:"));
        }
        why.push(match (coordinate, sets) {
            (UserCoordinate::Sweep, true) => format!(
                "The scenario would override every sweep value, so this sweep would \
                 not vary `{param}`."
            ),
            (UserCoordinate::Sweep, false) => format!(
                "The scenario would rescale every sweep point of `{param}`; composing a \
                 sweep with a scenario that scales the same parameter is not supported."
            ),
            (UserCoordinate::DrawsFile(_), true) => format!(
                "The scenario would override the file's `{param}` in every draw, so that \
                 column would have no effect."
            ),
            (UserCoordinate::DrawsFile(_), false) => format!(
                "The scenario would rescale the file's `{param}` in every draw; composing \
                 a draws file with a scenario that scales the same parameter is not \
                 supported."
            ),
        });
    }
    for w in why {
        msg.push('\n');
        msg.push_str(&w);
    }
    let them = if clash.len() == 1 { "it" } else { "them" };
    msg.push_str(&match coordinate {
        UserCoordinate::Sweep => format!(
            "\nFix: remove {} from the sweep, or use a scenario that does not touch {them}.",
            quoted(&clash)
        ),
        UserCoordinate::DrawsFile(_) => format!(
            "\nFix: drop the {} column{} from the draws file, or use a scenario that does \
             not touch {them}.",
            quoted(&clash),
            if clash.len() == 1 { "" } else { "s" },
        ),
    });
    Ok(Some(msg))
}

/// A scenario's net value for each parameter it touches: its `set` replaces
/// the value beneath it, then each `scale` factor multiplies, in application
/// order. `below(name)` is the value the scenario lands on — the lower tiers'
/// value — and a `scale` of a parameter with no value beneath it and no `set`
/// moves nothing.
///
/// The one fold behind tier 4 of [`resolve_parameters`] and behind
/// [`scenario_fixed_collision`], so the collision check sees exactly the value
/// the resolver would write.
fn fold_scenario_tier<'k>(
    set: impl IntoIterator<Item = &'k (String, f64)>,
    scale: &[(String, f64)],
    below: impl Fn(&str) -> Option<f64>,
) -> IndexMap<String, f64> {
    let mut tier4: IndexMap<String, f64> = IndexMap::new();
    for (k, v) in set {
        tier4.insert(k.clone(), *v);
    }
    for (k, factor) in scale {
        if let Some(v) = tier4.get(k).copied().or_else(|| below(k)) {
            tier4.insert(k.clone(), v * factor);
        }
    }
    tier4
}

// ─── Scenario × fit.toml [fixed] collisions ───────────────────────────────────

/// Refuse a fit whose `scenario` moves a parameter off the value its `[fixed]`
/// block gives it. `Ok(None)` when every `[fixed]` value survives the scenario
/// unchanged (the scenario does not touch it, or lands on the same value);
/// `Ok(Some(message))` names every colliding parameter, both values, both
/// sources and the fix.
///
/// One fit.toml naming two different values for one parameter does not say
/// which it means. Precedence would pick the scenario (tier 4 over tier 2,
/// spec §1.3), but the `[fixed]` value is written in the same file as the
/// scenario's name, so taking either silently is a guess. The same rule
/// `simulate` applies to a scenario over a sweep
/// ([`scenario_coordinate_collision`], gh#572). An equal value is not a
/// collision: both sources agree.
///
/// `fixed` is the resolved `[fixed]` map (inline values, `from_file`, or an
/// expanded `from_scenario`); `fit_toml` is the file it came from, for the
/// message.
pub fn scenario_fixed_collision(
    model: &ir::Model,
    scenario: &str,
    fixed: &IndexMap<String, f64>,
    fit_toml: &str,
) -> Result<Option<String>, String> {
    let delta = resolve_preset_delta(model, scenario).map_err(|e| e.to_string())?;
    let set: Vec<(String, f64)> = delta.params.into_iter().collect();
    let net = fold_scenario_tier(&set, &delta.scale, |k| fixed.get(k).copied());
    let clash: Vec<(&str, f64, f64)> = fixed
        .iter()
        .filter_map(|(k, &f)| match net.get(k) {
            Some(&s) if s.to_bits() != f.to_bits() => Some((k.as_str(), f, s)),
            _ => None,
        })
        .collect();
    if clash.is_empty() {
        return Ok(None);
    }

    let names: Vec<String> = clash.iter().map(|(p, _, _)| format!("`{p}`")).collect();
    let names = names.join(", ");
    let mut msg = if clash.len() == 1 {
        format!("parameter {names} is given two values: by scenario `{scenario}` and by \
                 [fixed] in `{fit_toml}`")
    } else {
        format!("parameters {names} are given two values: by scenario `{scenario}` and by \
                 [fixed] in `{fit_toml}`")
    };
    let scen = crate::sim_job::ScenarioRef::Named(scenario.to_string());
    for &(param, fixed_v, scen_v) in &clash {
        msg.push_str(&format!("\n  [fixed]:   {param} = {fixed_v}"));
        let how =
            describe_scenario_actions(param, scenario, &scenario_param_actions(model, &scen, param)?);
        msg.push_str(&format!("\n  scenario:  {param} = {scen_v}    ({how})"));
    }
    let them = if clash.len() == 1 { "it" } else { "them" };
    let agree: Vec<String> =
        clash.iter().map(|(p, _, s)| format!("`{p} = {s}`")).collect();
    msg.push_str(&format!(
        "\nA fit runs at one value of each parameter, and this file names two, so which \
         one is meant cannot be read from it.\n\
         Fix: remove {names} from scenario `{scenario}` (or name a scenario that does not \
         set {them}), or, to fit under the scenario, give [fixed] in `{fit_toml}` the \
         scenario's value: {}. [fixed] cannot drop {them}: every parameter a fit does not \
         estimate must be in [fixed].",
        agree.join(", ")
    ));
    Ok(Some(msg))
}

// ─── Scenario × fit.toml [estimate] collisions ────────────────────────────────

/// How a scenario's actions on `param` read in a diagnostic:
/// `set beta = 0.8`, `scale beta × 0.5 from composed preset `half``.
fn describe_scenario_actions(
    param: &str,
    scenario: &str,
    actions: &[ScenarioParamAction],
) -> String {
    let how: Vec<String> = actions
        .iter()
        .map(|a| {
            let (text, preset) = match a {
                ScenarioParamAction::Set { value, preset } => {
                    (format!("set {param} = {value}"), preset)
                }
                ScenarioParamAction::Scale { factor, preset } => {
                    (format!("scale {param} \u{d7} {factor}"), preset)
                }
            };
            if preset.as_str() == scenario {
                text
            } else {
                format!("{text} from composed preset `{preset}`")
            }
        })
        .collect();
    how.join(", ")
}

/// One estimated parameter a scenario touches: what the scenario does to it,
/// and its net value where one exists (a `set`, or a `scale` of the model's
/// default).
struct EstimateClash<'a> {
    param:   &'a str,
    actions: Vec<ScenarioParamAction>,
    net:     Option<f64>,
}

impl EstimateClash<'_> {
    fn sets(&self) -> bool {
        self.actions.iter().any(|a| matches!(a, ScenarioParamAction::Set { .. }))
    }

    /// `scenario:    beta = 0.8    (set beta = 0.8)`, or the actions alone
    /// when there is no net value.
    fn scenario_line(&self, scenario: &str) -> String {
        let how = describe_scenario_actions(self.param, scenario, &self.actions);
        match self.net {
            Some(v) => format!("\n  scenario:    {} = {v}    ({how})", self.param),
            None => format!("\n  scenario:    {how}"),
        }
    }
}

/// The parameters of `estimated` that `scenario` sets or scales, through any
/// preset it composes ([`scenario_param_footprint`], the same footprint the
/// sweep and draws-file guards read), in `estimated` order.
fn estimate_clashes<'a>(
    model: &ir::Model,
    scenario: &str,
    estimated: &[&'a str],
) -> Result<Vec<EstimateClash<'a>>, String> {
    let scen = crate::sim_job::ScenarioRef::Named(scenario.to_string());
    let footprint = scenario_param_footprint(model, &scen)?;
    let mut rows = Vec::new();
    for &param in estimated.iter().filter(|p| footprint.contains(**p)) {
        let actions = scenario_param_actions(model, &scen, param)?;
        let mut set: Vec<(String, f64)> = Vec::new();
        let mut scale: Vec<(String, f64)> = Vec::new();
        for a in &actions {
            match a {
                ScenarioParamAction::Set { value, .. } => set.push((param.to_string(), *value)),
                ScenarioParamAction::Scale { factor, .. } => {
                    scale.push((param.to_string(), *factor))
                }
            }
        }
        let net = fold_scenario_tier(&set, &scale, |k| {
            model.parameters.iter().find(|p| p.name == k).and_then(|p| p.value.resolved_value())
        })
        .get(param)
        .copied();
        rows.push(EstimateClash { param, actions, net });
    }
    Ok(rows)
}

/// The header pieces both estimate refusals share: the backticked names,
/// "set" / "scaled" / "set or scaled", and the singular or plural words.
struct ClashWords {
    names: String,
    verb:  &'static str,
    noun:  &'static str,
    is:    &'static str,
    them:  &'static str,
}

fn clash_words(rows: &[EstimateClash<'_>]) -> ClashWords {
    let n_set = rows.iter().filter(|r| r.sets()).count();
    let verb = if n_set == rows.len() {
        "set"
    } else if n_set == 0 {
        "scaled"
    } else {
        "set or scaled"
    };
    let names: Vec<String> = rows.iter().map(|r| format!("`{}`", r.param)).collect();
    let (noun, is, them) =
        if rows.len() == 1 { ("parameter", "is", "it") } else { ("parameters", "are", "them") };
    ClashWords { names: names.join(", "), verb, noun, is, them }
}

/// Refuse a fit whose scenario sets or scales a parameter its `[estimate]`
/// block estimates. `Ok(None)` when the scenario touches none of `estimated`;
/// `Ok(Some(message))` names every colliding parameter, what the scenario does
/// to it, the fit.toml and the fix.
///
/// A parameter is either estimated or given a value, never both. Left to
/// precedence, the estimate would overwrite whatever the scenario set, which
/// would survive at most as a starting value, and nothing would tell the user
/// that the scenario's value was not used. A composed preset's `set` or
/// `scale` is caught as surely as the parent's own ([`estimate_clashes`]).
///
/// This is the scenario a fit *runs under*. The scenario `[fixed]
/// from_scenario` imports is checked by [`from_scenario_estimate_collision`].
pub fn scenario_estimate_collision(
    model: &ir::Model,
    scenario: &str,
    estimated: &[&str],
    fit_toml: &str,
) -> Result<Option<String>, String> {
    let rows = estimate_clashes(model, scenario, estimated)?;
    if rows.is_empty() {
        return Ok(None);
    }
    let ClashWords { names, verb, noun, is, them } = clash_words(&rows);
    let mut msg = format!("{noun} {names} {is} both estimated and {verb} by scenario `{scenario}`");
    for r in &rows {
        msg.push_str(&format!("\n  [estimate]:  {}    (in `{fit_toml}`)", r.param));
        msg.push_str(&r.scenario_line(scenario));
    }
    msg.push_str(
        "\nAn estimated parameter takes its values from the estimation, which would discard \
         the scenario's value, so which one is meant cannot be read from the fit config.",
    );
    // The values a scenario `set` lands on, for the "fit under it" fix. A
    // parameter the scenario only scales has no such value: held in [fixed],
    // the scenario would rescale it off that value, which the [fixed]
    // collision refuses.
    let pinned: Vec<String> = rows
        .iter()
        .filter(|r| r.sets())
        .filter_map(|r| r.net.map(|v| format!("`{} = {v}`", r.param)))
        .collect();
    let other = format!("name a scenario that does not touch {them}");
    msg.push_str(&if pinned.is_empty() {
        format!(
            "\nFix: {other}. A scaled parameter cannot be held at a value of its own: the \
             scenario would rescale that value too."
        )
    } else {
        format!(
            "\nFix: drop {names} from [estimate] in `{fit_toml}` to fit under the scenario's \
             value (give [fixed] the scenario's value: {}), or {other}.",
            pinned.join(", ")
        )
    });
    Ok(Some(msg))
}

/// Refuse `[fixed] from_scenario = "<scenario>"` when that scenario sets or
/// scales a parameter `[estimate]` lists. `Ok(None)` when it touches none of
/// `estimated`.
///
/// `from_scenario` makes every parameter the scenario sets a `[fixed]` value,
/// so a parameter it sets and `[estimate]` lists would be both fixed and
/// estimated — the same contradiction as a name in both blocks, and refused
/// the same way rather than resolved by dropping one side. A parameter the
/// scenario scales is refused too: the scale would apply to a value the
/// estimate replaces.
pub fn from_scenario_estimate_collision(
    model: &ir::Model,
    scenario: &str,
    estimated: &[&str],
) -> Result<Option<String>, String> {
    let rows = estimate_clashes(model, scenario, estimated)?;
    if rows.is_empty() {
        return Ok(None);
    }
    let ClashWords { names, verb, noun, is, them } = clash_words(&rows);
    let mut msg = format!(
        "{noun} {names} {is} both estimated and {verb} by scenario `{scenario}`, which \
         [fixed] from_scenario imports"
    );
    for r in &rows {
        msg.push_str(&format!("\n  [estimate]:  {}", r.param));
        msg.push_str(&r.scenario_line(scenario));
    }
    msg.push_str(
        "\nA parameter cannot be both fixed (imported from the scenario) and estimated.",
    );
    // Dropping a parameter from [estimate] holds it at the scenario's value
    // only if the scenario `set`s it: a scale-only parameter is refused by
    // `scenario_values`, which has no value for the factor to multiply.
    let infinitive = if verb == "scaled" { "scale" } else if verb == "set" { "set" } else { "set or scale" };
    let define = format!(
        "define a scenario that does not {infinitive} {names} and name it in from_scenario"
    );
    msg.push_str(&if rows.iter().all(|r| r.sets()) {
        format!("\nFix: {define}, or drop {names} from [estimate] to hold {them} at the \
                 scenario's value.")
    } else {
        format!("\nFix: {define}.")
    });
    Ok(Some(msg))
}

/// The parameter values scenario `scenario` gives on its own: its composed
/// `set` values with its composed `scale` factors applied — the values
/// `[fixed] from_scenario` imports, and the values `simulate --scenario`
/// lands on for every parameter the scenario sets.
///
/// Reads [`resolve_preset_delta`] and folds it with [`fold_scenario_tier`],
/// the same fold tier 4 of [`resolve_parameters`] applies, so the import and
/// the forward run cannot disagree on what a scenario's values are.
///
/// `Ok(Err(message))` when the scenario scales a parameter it does not set.
/// Tier 4 multiplies the value beneath the scenario; an import has no value
/// beneath it (a camdl model declares no parameter defaults), so the factor
/// would have nothing to multiply and is refused rather than dropped.
pub fn scenario_values(
    model: &ir::Model,
    scenario: &str,
) -> Result<Result<IndexMap<String, f64>, String>, ResolveError> {
    let delta = resolve_preset_delta(model, scenario)?;
    let unset: Vec<&(String, f64)> = delta.scale.iter()
        .filter(|(k, _)| !delta.params.contains_key(k))
        .collect();
    if !unset.is_empty() {
        let mut names: Vec<&str> = Vec::new();
        for (k, _) in &unset {
            if !names.contains(&k.as_str()) {
                names.push(k.as_str());
            }
        }
        let quoted_names =
            |ps: &[&str]| ps.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(", ");
        let (noun, them) = if names.len() == 1 { ("parameter", "it") } else { ("parameters", "them") };
        let mut msg = format!(
            "scenario `{scenario}` scales {noun} {} but does not set {them}, and \
             [fixed] from_scenario imports only the values a scenario sets: the scale \
             has no value to multiply",
            quoted_names(&names),
        );
        for (k, factor) in &unset {
            msg.push_str(&format!("\n  scenario `{scenario}`:  scale `{k}` by {factor}"));
        }
        msg.push_str(&format!(
            "\nFix: set {} in scenario `{scenario}` as well, so the scale applies to \
             a value the scenario gives.",
            quoted_names(&names),
        ));
        return Ok(Err(msg));
    }
    let set: Vec<(String, f64)> = delta.params.into_iter().collect();
    Ok(Ok(fold_scenario_tier(&set, &delta.scale, |_| None)))
}

/// One info notice per focal parameter `scenario` sets or scales: `camdl
/// profile` writes each grid value straight over the focal parameter, so the
/// scenario's value for it is replaced at every grid point.
///
/// A layered override — stepping the parameter across a grid is what the user
/// asked for — so it is reported, not refused. Called once per profile run,
/// so the line prints once, not once per grid point. Scenario-set nuisance
/// (estimated) parameters are not covered here.
pub fn scenario_focal_notices(
    model: &ir::Model,
    scenario: &str,
    focal: &[&str],
) -> Result<Vec<Notice>, String> {
    let scen = crate::sim_job::ScenarioRef::Named(scenario.to_string());
    let footprint = scenario_param_footprint(model, &scen)?;
    let mut out = Vec::new();
    for &param in focal.iter().filter(|p| footprint.contains(**p)) {
        let does: Vec<String> = scenario_param_actions(model, &scen, param)?
            .iter()
            .map(|a| {
                let (text, preset) = match a {
                    ScenarioParamAction::Set { value, preset } => {
                        (format!("sets {param} = {value}"), preset)
                    }
                    ScenarioParamAction::Scale { factor, preset } => {
                        (format!("scales {param} \u{d7} {factor}"), preset)
                    }
                };
                if preset.as_str() == scenario {
                    text
                } else {
                    format!("{text} (from composed preset `{preset}`)")
                }
            })
            .collect();
        out.push(Notice::info("scenario-sets-focal", format!(
            "scenario `{scenario}` {}; {param} is profiled, so the grid values replace it",
            does.join(" and "))));
    }
    Ok(out)
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
    resolve_parameters_with_starts(inputs, None)
}

/// [`resolve_parameters`], with a single-point `--starts` rule's values at
/// tier 1.5 ([`StartsValues`]). Only `profile` has such a rule; every other
/// verb resolves through [`resolve_parameters`].
pub fn resolve_parameters_with_starts<'a>(
    inputs: ParameterInputs<'a>,
    starts: Option<StartsValues<'a>>,
) -> Result<ResolvedParameters, ResolveError> {
    let mut model = inputs.model.clone();
    let mut pinned_over_estimate: Vec<PinnedOverEstimate> = Vec::new();

    // ── Tier 1: model defaults (already in model.parameters) ────────────
    //
    // No mutation needed — the IR carries `p.value` straight from the
    // DSL. The resolver layers tiers 2..5 on top.
    //
    // Track each parameter's *current* source as we walk tiers, and every
    // replacement of one tier's value by another's ([`Layers::set`]). The
    // source map starts with whatever the IR supplied (ModelDefault for
    // params with a value, sentinel "unset" for those without).
    let mut layers = Layers {
        source: model.parameters.iter()
            .map(|p| (p.name.clone(),
                if p.value.resolved_value().is_some() { Some(ValueSource::ModelDefault) } else { None }))
            .collect(),
        overrides: HashMap::new(),
    };
    let mut context = ResolveContext::default();

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
            // The whole composed delta comes from the one authority the run
            // identity also reads (gh#573), so what is applied here and what
            // is hashed cannot disagree. `params` is the deduped (parent-wins)
            // map shared with the fit `[fixed] from_scenario` path (gh#36);
            // `scale` is the same list the collision guards' footprint reads.
            let delta = resolve_preset_delta(&model, name)?;
            (delta.enable, delta.disable, delta.params.into_iter().collect(),
             delta.scale, Some(name.to_string()))
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

    // ── Tier 1.5: a single-point `--starts` rule's values ───────────────
    //
    // Just above the model default, so every other tier replaces them and
    // each replacement is recorded. Names the model does not declare are
    // skipped ([`StartsValues`]).
    if let Some(starts) = starts {
        for (name, v) in starts.values {
            for p in &mut model.parameters {
                if p.name == *name {
                    layers.set(p, ValueSource::Starts(starts.origin.clone()), *v);
                }
            }
        }
    }

    // ── Tier 2: fit.toml [fixed] block ──────────────────────────────────
    for (name, &v) in inputs.fit_toml_fixed {
        if !model_param_set.contains(name) {
            return Err(ResolveError::UnknownParameter {
                name: name.clone(),
                source: ValueSource::FitTomlFixed,
                candidates: model_param_names.clone(),
                constant_hint: model.constant_hint(name),
            });
        }
        for p in &mut model.parameters {
            if p.name == *name {
                layers.set(p, ValueSource::FitTomlFixed, v);
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
                    constant_hint: model.constant_hint(name),
                });
            }
        }
        if carries_fit_provenance(path) {
            context.fit_estimate_files.insert(path.clone());
        }
        for p in &mut model.parameters {
            if let Some(&v) = overrides.get(&p.name) {
                layers.set(p, ValueSource::FixedFile { path: path.clone() }, v);
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
                constant_hint: model.constant_hint(name),
            });
        }
        for p in &mut model.parameters {
            if p.name == *name {
                layers.set(p, ValueSource::SweepPoint, *v);
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
    // The scenario is one tier: its `set` and its `scale` on a parameter are
    // folded into one net value first (`set` replaces the current value,
    // each `scale` factor then multiplies), and that value is layered once —
    // so a replacement is reported against the lower tier's value, never
    // against the scenario's own intermediate `set`.
    if let Some(name) = scenario_label.as_deref() {
        let tier4 = fold_scenario_tier(
            scenario_params.iter().filter(|(k, _)| model_param_set.contains(k)),
            &scenario_scale,
            |k| model.parameters.iter()
                .find(|p| p.name == k)
                .and_then(|p| p.value.resolved_value()),
        );
        for (k, v) in &tier4 {
            for p in &mut model.parameters {
                if p.name == *k {
                    layers.set(p, ValueSource::Scenario(name.to_string()), *v);
                }
            }
        }
    }

    // ── Tier 5: --fixed NAME=VALUE (highest) ────────────────────────────
    //
    // One net value per name (a name given twice — `--param-vec` then
    // `--param` — is one tier-5 value, the last).
    let mut tier5: IndexMap<String, f64> = IndexMap::new();
    for (name, v) in inputs.fixed_cli {
        if !model_param_set.contains(name) {
            return Err(ResolveError::UnknownParameter {
                name: name.clone(),
                source: ValueSource::FixedCli,
                candidates: model_param_names.clone(),
                constant_hint: model.constant_hint(name),
            });
        }
        tier5.insert(name.clone(), *v);
    }
    for (name, v) in &tier5 {
        for p in &mut model.parameters {
            if p.name == *name {
                layers.set(p, ValueSource::FixedCli, *v);
            }
        }
    }

    // ── fit.toml [fixed] ∩ [estimate] overlap: refused ──────────────────
    //
    // A name in both blocks of one fit config is both estimated and given a
    // value, and the file does not say which it means; neither block wins.
    let overlap: Vec<String> = inputs.fit_toml_fixed.keys()
        .filter(|n| inputs.fit_toml_estimate.contains(*n))
        .cloned()
        .collect();
    if !overlap.is_empty() {
        return Err(ResolveError::FixedEstimateOverlap { names: overlap });
    }

    // ── Estimate-set kick-out + provenance assembly ─────────────────────
    let mut estimate_set: IndexSet<String> = inputs.fit_toml_estimate.clone();

    // A name is "kicked from [estimate]" if it appears in tier 4 or
    // tier 5 (CLI / file `--fixed*`) — those are user-explicit "pin
    // this" assertions. Tier 3 (fit.toml [fixed]) is a no-op here
    // because the toml's `[fixed]` block already excludes those
    // names from `[estimate]` at config-load time.
    let kicker_names: HashMap<String, (ValueSource, f64)> = {
        let mut m: HashMap<String, (ValueSource, f64)> = HashMap::new();
        for path in inputs.fixed_files {
            let path_str = path.to_string_lossy().into_owned();
            // We already validated the file; load it again for the
            // name list. The cost is negligible vs simulation, and
            // it keeps tier 4 / tier 5 source attribution explicit.
            if let Ok(overrides) = crate::util::load_params_toml(&path_str) {
                for (name, &value) in &overrides {
                    m.insert(name.clone(),
                        (ValueSource::FixedFile { path: path.clone() }, value));
                }
            }
        }
        for (name, value) in inputs.fixed_cli {
            m.insert(name.clone(), (ValueSource::FixedCli, *value));
        }
        m
    };

    let mut kicked: HashMap<String, ValueSource> = HashMap::new();
    estimate_set.retain(|name| {
        if let Some((by, value)) = kicker_names.get(name) {
            pinned_over_estimate.push(PinnedOverEstimate {
                name: name.clone(),
                by: by.clone(),
                value: *value,
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
            let source = layers.source.get(&p.name)
                .and_then(|s| s.clone())
                .unwrap_or(ValueSource::ModelDefault);
            violations.push(ResolveError::NonFiniteValue {
                name: p.name.clone(),
                value,
                source,
            });
            continue;
        }
        // The declared `in [lo, hi]`, else the support the declared kind
        // carries (gh#763: `probability` is `[0, 1]` whether or not the model
        // wrote it). This must be the SAME interval the fit's transform
        // clamps to — with the two apart, a `probability` start of 1.5 passed
        // validation and was then silently clamped to 1.0 by the logit.
        if let Some((lo, hi)) = resolved_bounds(p) {
            if value < lo || value > hi {
                violations.push(ResolveError::BoundsViolation {
                    name: p.name.clone(),
                    value, lo, hi,
                });
                continue;
            }
        }
        let source = layers.source.get(&p.name)
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

        params.push(ResolvedParameter {
            name: p.name.clone(),
            value,
            source,
            role,
            overrides: layers.overrides.remove(&p.name).unwrap_or_default(),
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
        pinned_over_estimate,
        context,
    })
}

/// The tier walk's running state: each parameter's current source, and every
/// [`Override`] recorded so far.
struct Layers {
    source:    HashMap<String, Option<ValueSource>>,
    overrides: HashMap<String, Vec<Override>>,
}

impl Layers {
    /// Set `p` to `v` from `source`, recording an [`Override`] when this
    /// replaces a different value that a tier other than the model default
    /// supplied. The single writer of a tier's value, so no tier can replace
    /// another's without the replacement being recorded.
    fn set(&mut self, p: &mut ir::parameter::Parameter, source: ValueSource, v: f64) {
        let prev = self.source.get(&p.name).cloned().flatten();
        if let (Some(prev), Some(prev_v)) = (prev, p.value.resolved_value()) {
            if prev != ValueSource::ModelDefault && prev_v != v {
                self.overrides.entry(p.name.clone()).or_default().push(Override {
                    name:     p.name.clone(),
                    replaced: (prev, prev_v),
                    by:       (source.clone(), v),
                });
            }
        }
        p.value = p.value.with_value(v);
        self.source.insert(p.name.clone(), Some(source));
    }
}

/// Whether a params file carries a fit's `[provenance]` block — its values
/// are then a fit's estimate ([`severity`]). A `.tsv` params file has no
/// block, and an unreadable block counts as none: this only chooses how a
/// replacement is reported, and `simulate` refuses a malformed block before
/// it resolves anything.
pub(crate) fn carries_fit_provenance(path: &std::path::Path) -> bool {
    let is_tsv = path.extension()
        .map(|e| e.eq_ignore_ascii_case("tsv"))
        .unwrap_or(false);
    !is_tsv && matches!(
        crate::fit::provenance::read_mle_provenance(&path.to_string_lossy()),
        Ok(Some(_)))
}

// ─── Notices ──────────────────────────────────────────────────────────────────

/// How a notice names `src` as the origin of a replaced value.
fn origin_phrase(src: &ValueSource, ctx: &ResolveContext) -> String {
    match src {
        ValueSource::ModelDefault => "the model default".to_string(),
        ValueSource::Scenario(n) => format!("scenario `{n}`"),
        ValueSource::FitTomlFixed => "fit.toml [fixed]".to_string(),
        ValueSource::FixedFile { path } if ctx.fit_estimate_files.contains(path) =>
            format!("{}, a fit's estimate", path.display()),
        ValueSource::FixedFile { path } => path.display().to_string(),
        ValueSource::SweepPoint => "the draw or sweep point".to_string(),
        ValueSource::FixedCli => "--param/--fixed".to_string(),
        ValueSource::Starts(StartsOrigin::Params { path }) =>
            format!("--starts from_params={}", path.display()),
        ValueSource::Starts(StartsOrigin::Mle { fit }) =>
            format!("--starts from_mle={fit}, a fit's estimate"),
        ValueSource::EstimateStart { drawn: false } => "[estimate].start".to_string(),
        ValueSource::EstimateStart { drawn: true } =>
            "a start drawn within the [estimate] bounds".to_string(),
    }
}

/// How a notice names `src` as the subject that replaced a value.
fn subject_phrase(src: &ValueSource) -> String {
    match src {
        ValueSource::FixedFile { path } => format!("params file {}", path.display()),
        other => origin_phrase(other, &ResolveContext::default()),
    }
}

/// `v`, or `lo to hi` when the values differ (a multi-cell run).
fn value_span(values: &[f64]) -> String {
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo == hi { format!("{lo}") } else { format!("{lo} to {hi}") }
}

/// The parameter, the replaced value's source and the winner's source: one
/// override line per distinct key, however many cells it occurs in.
type OverrideKey = (String, ValueSource, ValueSource);

/// What [`notices`] reads from one resolution: its parameters (with their
/// recorded overrides), its `[estimate]` pins and its context — everything but the
/// valued model. A caller that resolves many times and reports once
/// (contrast arms, one resolution per posterior draw) keeps these
/// ([`ResolvedParameters::into_notice_parts`]) instead of every model copy.
#[derive(Debug, Clone)]
pub struct NoticeParts {
    pub params:   Vec<ResolvedParameter>,
    pub pinned_over_estimate: Vec<PinnedOverEstimate>,
    pub context:  ResolveContext,
}

impl ResolvedParameters {
    /// The parts of this resolution [`notices`] reads, without the model.
    pub fn into_notice_parts(self) -> NoticeParts {
        NoticeParts {
            params: self.params,
            pinned_over_estimate: self.pinned_over_estimate,
            context: self.context,
        }
    }
}

/// A borrowed [`NoticeParts`]: the one shape the notice functions read, so a
/// full resolution and a kept set of parts report identically.
#[derive(Clone, Copy)]
struct PartsRef<'a> {
    params:   &'a [ResolvedParameter],
    pinned_over_estimate: &'a [PinnedOverEstimate],
    context:  &'a ResolveContext,
}

impl<'a> From<&'a ResolvedParameters> for PartsRef<'a> {
    fn from(r: &'a ResolvedParameters) -> Self {
        PartsRef {
            params: &r.params,
            pinned_over_estimate: &r.pinned_over_estimate,
            context: &r.context,
        }
    }
}

impl<'a> From<&'a NoticeParts> for PartsRef<'a> {
    fn from(r: &'a NoticeParts) -> Self {
        PartsRef {
            params: &r.params,
            pinned_over_estimate: &r.pinned_over_estimate,
            context: &r.context,
        }
    }
}

/// One `param-override` notice per [`OverrideKey`] across `resolved`, in
/// first-seen order (cell order, then parameter declaration order, then tier
/// order). When the values differ by cell, each side reports its range and
/// the line says how many cells it covers. The severity is the loudest any
/// cell assigns ([`severity`]).
fn override_notices(resolved: &[PartsRef<'_>]) -> Vec<(OverrideKey, Notice)> {
    struct Agg {
        replaced: Vec<f64>,
        by:       Vec<f64>,
        cells:    usize,
        severity: Severity,
        origin:   String,
    }
    let mut agg: IndexMap<OverrideKey, Agg> = IndexMap::new();
    for r in resolved {
        // A key counts once per cell even if (it cannot today) it recurred.
        let mut seen_in_cell: HashSet<OverrideKey> = HashSet::new();
        for p in r.params {
            for o in &p.overrides {
                let key = (o.name.clone(), o.replaced.0.clone(), o.by.0.clone());
                let sev = severity(o, r.context);
                let e = agg.entry(key.clone()).or_insert_with(|| Agg {
                    replaced: Vec::new(),
                    by:       Vec::new(),
                    cells:    0,
                    severity: sev,
                    origin:   origin_phrase(&o.replaced.0, r.context),
                });
                e.replaced.push(o.replaced.1);
                e.by.push(o.by.1);
                e.severity = e.severity.max(sev);
                if seen_in_cell.insert(key) {
                    e.cells += 1;
                }
            }
        }
    }
    agg.into_iter().map(|(key, a)| {
        let (name, _, by_src) = &key;
        let replaced = value_span(&a.replaced);
        let by = value_span(&a.by);
        let across = if replaced.contains(" to ") || by.contains(" to ") {
            format!(", across {} cells", a.cells)
        } else {
            String::new()
        };
        let message = format!(
            "{} replaced {name} = {replaced} (from {}) with {by}{across}",
            subject_phrase(by_src), a.origin);
        let notice = Notice { severity: a.severity, code: "param-override", message };
        (key, notice)
    }).collect()
}

fn notices_of(resolved: &[PartsRef<'_>]) -> Notices {
    let mut pins: IndexSet<Notice> = IndexSet::new();
    for r in resolved {
        pins.extend(r.pinned_over_estimate.iter().map(PinnedOverEstimate::notice));
    }
    pins.into_iter()
        .chain(override_notices(resolved).into_iter().map(|(_, n)| n))
        .collect()
}

/// The notices a run's resolutions report: the `[estimate]` pins (each
/// distinct line once), then one line per distinct override
/// ([`override_notices`]). One resolution for a single run; one per cell for
/// a multi-cell run, so a condition every cell shares is reported once.
pub fn notices(resolved: &[&ResolvedParameters]) -> Notices {
    let parts: Vec<PartsRef<'_>> = resolved.iter().map(|r| PartsRef::from(*r)).collect();
    notices_of(&parts)
}

/// Emit [`notices`] for `resolved` to stderr. Subcommand wrappers call this
/// once after resolving — over every cell's resolution for a multi-cell run.
///
/// Also runs a structural cross-check, debug-only so it trips in tests but
/// costs nothing in release, between each parameter's recorded
/// [`Override`]s (what `run.json` records) and the lines reported:
///
/// - the overrides chain — each replacement replaces the value the previous
///   one set, and the last one's winner is the parameter's final source and
///   value — so a tier that wrote a value without going through
///   [`Layers::set`] is caught;
/// - the parameters the override lines name are exactly the parameters with
///   recorded overrides, so a line without provenance, or provenance without
///   a line, is caught.
pub fn emit_notices(resolved: &[&ResolvedParameters]) {
    let parts: Vec<PartsRef<'_>> = resolved.iter().map(|r| PartsRef::from(*r)).collect();
    emit_notices_of(&parts);
}

/// [`emit_notices`] over kept [`NoticeParts`].
pub fn emit_notices_from_parts(resolved: &[NoticeParts]) {
    let parts: Vec<PartsRef<'_>> = resolved.iter().map(PartsRef::from).collect();
    emit_notices_of(&parts);
}

fn emit_notices_of(resolved: &[PartsRef<'_>]) {
    notices_of(resolved).emit_stderr();

    if cfg!(debug_assertions) {
        for r in resolved {
            for p in r.params {
                for w in p.overrides.windows(2) {
                    debug_assert_eq!(w[0].by.1.to_bits(), w[1].replaced.1.to_bits(),
                        "override chain broken for `{}`: {:?}", p.name, p.overrides);
                }
                if let Some(last) = p.overrides.last() {
                    debug_assert!(last.by.0 == p.source
                        && last.by.1.to_bits() == p.value.to_bits(),
                        "last override of `{}` is not its final value: {:?} vs ({:?}, {})",
                        p.name, last.by, p.source, p.value);
                }
            }
        }
        let reported: HashSet<String> = override_notices(resolved).into_iter()
            .map(|((name, _, _), _)| name).collect();
        let recorded: HashSet<String> = resolved.iter()
            .flat_map(|r| r.params.iter())
            .filter(|p| !p.overrides.is_empty())
            .map(|p| p.name.clone())
            .collect();
        debug_assert_eq!(reported, recorded,
            "override notices and recorded overrides must name the same parameters");
    }
}

/// The `param-override` notice for `fit predict --sweep`: a swept value
/// replaces each posterior draw's value for that parameter, inside one draw
/// row, so the resolver sees one draw/sweep-tier value and records no
/// override. One info line per swept parameter, naming the replaced draw
/// values as a range and the swept values as theirs.
///
/// `draws` are the posterior draw rows the sweep is folded into;
/// `sweep_points` the expanded grid (one `(param, value)` list per point). A
/// parameter whose draws all already equal every swept value replaced
/// nothing and gets no line, as for any equal-value replacement. A side that
/// spans several values is a range, and the line then says how many draws it
/// covers.
pub fn sweep_over_draws_notices<'a>(
    draws: impl IntoIterator<Item = &'a IndexMap<String, f64>>,
    sweep_points: &[Vec<(String, f64)>],
) -> Notices {
    let mut swept: IndexMap<&str, Vec<f64>> = IndexMap::new();
    for point in sweep_points {
        for (name, v) in point {
            swept.entry(name.as_str()).or_default().push(*v);
        }
    }
    let mut replaced: IndexMap<&str, Vec<f64>> = IndexMap::new();
    for d in draws {
        for name in swept.keys() {
            if let Some(&v) = d.get(*name) {
                replaced.entry(*name).or_default().push(v);
            }
        }
    }
    swept.iter().filter_map(|(name, by)| {
        let was = replaced.get(name).filter(|w| !w.is_empty())?;
        if was.iter().all(|w| by.iter().all(|b| b == w)) {
            return None;
        }
        let (was_span, by_span) = (value_span(was), value_span(by));
        let across = if was_span.contains(" to ") || by_span.contains(" to ") {
            format!(", across {} draws", was.len())
        } else {
            String::new()
        };
        Some(Notice {
            severity: Severity::Info,
            code: "param-override",
            message: format!(
                "--sweep replaced {name} = {was_span} (from the posterior draws) \
                 with {by_span}{across}"),
        })
    }).collect()
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
            constants: vec![],
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
            initial_conditions: InitialConditions::default(),
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
                t_end_anchor: None,
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

    // ── The range a parameter is valid on (gh#763) ────────────────────

    fn mk_param_kinded(
        name: &str, value: f64,
        bounds: Option<(f64, f64)>, kind: ir::parameter::ParamKind,
    ) -> Parameter {
        Parameter {
            name: name.into(),
            value: ir::parameter::ParamValue::Estimated {
                init: Some(value),
                bounds,
                prior: ir::parameter::PriorSpec::Flat,
                transform: ir::parameter::Transform::Identity,
            },
            param_kind: Some(kind),
            param_dim: None,
        }
    }

    /// `probability` is the only kind whose type pins a finite interval. Every
    /// other kind returns `None` and keeps whatever fallback its consumer
    /// already had — `Transform::Log` is well-formed with `hi = ∞`, and
    /// `instant`/`real` fall to `Transform::None` when a bound is infinite, so
    /// none of them has the gh#763 hole.
    #[test]
    fn only_probability_carries_a_range_of_its_own() {
        use ir::parameter::ParamKind::*;
        assert_eq!(kind_support(Probability), Some((0.0, 1.0)),
            "`probability` must carry the [0, 1] the language spec §4.1 promises");
        for kind in [Rate, Positive, Count, Duration, Real, Instant] {
            assert_eq!(kind_support(kind), None,
                "{kind} must keep its consumer's existing fallback, not gain one here");
        }
    }

    /// Precedence: an explicit `in [...]` still wins over the type's range,
    /// and a kind-less parameter still has no range at all.
    #[test]
    fn declared_bounds_win_over_the_kind_range() {
        use ir::parameter::ParamKind::Probability;
        let declared = mk_param_kinded("rho", 0.5, Some((0.1, 0.9)), Probability);
        assert_eq!(resolved_bounds(&declared), Some((0.1, 0.9)),
            "`in [0.1, 0.9]` must not be widened to the type's [0, 1]");

        let undeclared = mk_param_kinded("rho", 0.5, None, Probability);
        assert_eq!(resolved_bounds(&undeclared), Some((0.0, 1.0)),
            "an undeclared `probability` must resolve to the type's [0, 1]");

        let untyped = mk_param_bounded("x", Some(0.5), (0.0, 2.0));
        assert_eq!(resolved_bounds(&untyped), Some((0.0, 2.0)),
            "a kind-less parameter must still report its declared bounds");
        assert_eq!(resolved_bounds(&mk_param("y", Some(0.5))), None,
            "a kind-less parameter with no bounds must still have no range");
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
        assert_eq!(resolved.pinned_over_estimate.len(), 1);
        let PinnedOverEstimate { name, by, .. } = &resolved.pinned_over_estimate[0];
        assert_eq!(name, "gamma");
        assert_eq!(*by, ValueSource::FixedCli);
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
        assert!(resolved.pinned_over_estimate.is_empty());
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
    fn fixed_over_estimate_is_an_info_notice_naming_the_flag_and_value() {
        // `--fixed` over a name `[estimate]` lists is a layered override the
        // user asked for: info, naming the parameter, the flag and its value.
        let model = mk_model(vec![mk_param("gamma", Some(0.1))]);
        let fcli = vec![("gamma".to_string(), 0.2)];
        let ffiles = vec![];
        let ftf = IndexMap::new();
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("gamma".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        let ns: Vec<Notice> = notices(&[&resolved]).iter()
            .filter(|n| n.code == "fixed-over-estimate").cloned().collect();
        assert_eq!(ns.len(), 1, "{ns:?}");
        assert_eq!(ns[0].severity, Severity::Info);
        assert_eq!(ns[0].render(),
            "info: gamma: `--fixed gamma=0.2` pins it, so it is not estimated \
             (the fit config lists it in [estimate])");

        // `emit_notices` is a thin wrapper; smoke-call it to confirm
        // no panic and to keep the symbol live.
        emit_notices(&[&resolved]);
    }

    #[test]
    fn fixed_file_over_estimate_is_an_info_notice_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("pins.toml");
        std::fs::write(&file, "gamma = 0.25\n").unwrap();
        let model = mk_model(vec![mk_param("gamma", Some(0.1))]);
        let fcli = vec![];
        let ffiles = vec![file.clone()];
        let ftf = IndexMap::new();
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("gamma".into());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        let n = PinnedOverEstimate::notice(&resolved.pinned_over_estimate[0]);
        assert_eq!(n.severity, Severity::Info);
        assert_eq!(n.message, format!(
            "gamma: `--fixed-file {}` pins it at 0.25, so it is not estimated \
             (the fit config lists it in [estimate])", file.display()));
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
        assert_eq!(ValueSource::Starts(StartsOrigin::Params { path: PathBuf::from("p") }).tag(),
            "starts_from_params");
        assert_eq!(ValueSource::Starts(StartsOrigin::Mle { fit: "f".into() }).tag(),
            "starts_from_mle");
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
            t_end_anchor: None,
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

    // ── Override reporting ──────────────────────────────────────────────

    /// A preset that `set`s `params` and `scale`s `scale`.
    fn preset(name: &str, params: &[(&str, f64)], scale: &[(&str, f64)]) -> Preset {
        Preset {
            name: name.into(),
            label: name.into(),
            params: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            scale: scale.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            enable: vec![],
            disable: vec![],
            compose: vec![],
            t_end: None,
            t_end_anchor: None,
        }
    }

    /// A params file in a fresh temp dir.
    fn params_file(name: &str, contents: &str) -> PathBuf {
        let dir = crate::test_support::unique_temp_dir("override-notices");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    /// A params file carrying a fit's `[provenance]` block.
    fn fit_params_file(name: &str, params: &str) -> PathBuf {
        params_file(name, &format!("{params}
[provenance]
camdl_version = \"test\"
timestamp = \"2026-10-25T00:00:00Z\"
content_hash = \"00000000\"
backend = \"chain_binomial\"
dt = 1.0
model = \"m.camdl\"
model_identity = \"f00d\"
seed = 1
method = \"if2\"
chain = 1
log_likelihood = -1.0
loglik_sd = 0.0
n_particles = 10
"))
    }

    fn rendered(resolved: &[&ResolvedParameters]) -> Vec<String> {
        notices(resolved).iter().map(Notice::render).collect()
    }

    #[test]
    fn scenario_over_params_file_is_reported_with_both_values_and_the_file() {
        // The maintainer's case: a params file gives I0 = 5, the scenario
        // sets I0 = 3. The run uses 3, and says so.
        let mut model = mk_model(vec![mk_param("I0", Some(1.0))]);
        model.presets.push(preset("lockdown", &[("I0", 3.0)], &[]));
        let file = params_file("p.toml", "I0 = 5\n");
        let ffiles = vec![file.clone()];
        let (fcli, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("lockdown");
        let resolved = resolve_parameters(inputs).expect("ok");

        assert_eq!(resolved.params[0].value, 3.0);
        assert_eq!(resolved.params[0].overrides, vec![Override {
            name: "I0".into(),
            replaced: (ValueSource::FixedFile { path: file.clone() }, 5.0),
            by: (ValueSource::Scenario("lockdown".into()), 3.0),
        }]);
        assert_eq!(rendered(&[&resolved]), vec![format!(
            "info: scenario `lockdown` replaced I0 = 5 (from {}) with 3",
            file.display())]);
    }

    #[test]
    fn later_params_file_over_earlier_is_reported() {
        let model = mk_model(vec![mk_param("beta", Some(0.1))]);
        let a = params_file("a.toml", "beta = 0.2\n");
        let b = params_file("b.toml", "beta = 0.4\n");
        let ffiles = vec![a.clone(), b.clone()];
        let (fcli, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let resolved = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect("ok");
        assert_eq!(resolved.params[0].value, 0.4);
        assert_eq!(rendered(&[&resolved]), vec![format!(
            "info: params file {} replaced beta = 0.2 (from {}) with 0.4",
            b.display(), a.display())]);
    }

    #[test]
    fn fixed_cli_over_scenario_is_reported() {
        // Scenario sets beta=0.3; --fixed beta=0.5 wins, and the scenario's
        // value is recorded as the one replaced.
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        model.presets.push(preset("worst_case", &[("beta", 0.3)], &[]));
        let fcli = vec![("beta".to_string(), 0.5)];
        let (ffiles, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");

        assert_eq!(resolved.params[0].value, 0.5);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
        assert_eq!(resolved.params[0].overrides, vec![Override {
            name: "beta".into(),
            replaced: (ValueSource::Scenario("worst_case".into()), 0.3),
            by: (ValueSource::FixedCli, 0.5),
        }]);
        assert_eq!(rendered(&[&resolved]), vec![
            "info: --param/--fixed replaced beta = 0.3 (from scenario `worst_case`) with 0.5"
                .to_string()]);
        emit_notices(&[&resolved]);
    }

    #[test]
    fn a_fits_estimate_replaced_by_a_scenario_is_a_warning() {
        let mut model = mk_model(vec![mk_param("I0", Some(1.0))]);
        model.presets.push(preset("baseline", &[("I0", 3.0)], &[]));
        let file = fit_params_file("mle_params.toml", "I0 = 5");
        let ffiles = vec![file.clone()];
        let (fcli, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("baseline");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert!(resolved.context.fit_estimate_files.contains(&file));
        assert_eq!(rendered(&[&resolved]), vec![format!(
            "warning: scenario `baseline` replaced I0 = 5 (from {}, a fit's estimate) with 3",
            file.display())]);
    }

    #[test]
    fn an_equal_value_is_not_an_override() {
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        model.presets.push(preset("worst_case", &[("beta", 0.3)], &[]));
        let file = params_file("p.toml", "beta = 0.3\n");
        let ffiles = vec![file];
        let fcli = vec![("beta".to_string(), 0.3)];
        let (ftf, fte) = (IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 0.3);
        assert_eq!(resolved.params[0].source, ValueSource::FixedCli);
        assert!(resolved.params[0].overrides.is_empty());
        assert!(rendered(&[&resolved]).is_empty());
    }

    #[test]
    fn replacing_the_model_default_is_not_an_override() {
        let mut model = mk_model(vec![mk_param("beta", Some(0.5))]);
        model.presets.push(preset("worst_case", &[("beta", 0.3)], &[]));
        let (fcli, ffiles, ftf, fte) = (vec![], vec![], IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("worst_case");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 0.3);
        assert!(resolved.params[0].overrides.is_empty());
        assert!(rendered(&[&resolved]).is_empty());
    }

    #[test]
    fn a_scenarios_set_and_scale_are_one_replacement() {
        // set 0.4 then scale ×0.5 is one tier-4 value, 0.2, reported against
        // the file's value — not as the file → 0.4 → 0.2 chain.
        let mut model = mk_model(vec![mk_param("beta", Some(0.1))]);
        model.presets.push(preset("half", &[("beta", 0.4)], &[("beta", 0.5)]));
        let file = params_file("p.toml", "beta = 0.8\n");
        let ffiles = vec![file.clone()];
        let (fcli, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("half");
        let resolved = resolve_parameters(inputs).expect("ok");
        assert_eq!(resolved.params[0].value, 0.2);
        assert_eq!(resolved.params[0].overrides, vec![Override {
            name: "beta".into(),
            replaced: (ValueSource::FixedFile { path: file }, 0.8),
            by: (ValueSource::Scenario("half".into()), 0.2),
        }]);
    }

    #[test]
    fn every_replacement_is_recorded_in_tier_order() {
        // file a → file b → scenario → --param: three replacements.
        let mut model = mk_model(vec![mk_param("I0", Some(1.0))]);
        model.presets.push(preset("s", &[("I0", 3.0)], &[]));
        let a = params_file("a.toml", "I0 = 5\n");
        let b = params_file("b.toml", "I0 = 4\n");
        let ffiles = vec![a.clone(), b.clone()];
        let fcli = vec![("I0".to_string(), 7.0)];
        let (ftf, fte) = (IndexMap::new(), IndexSet::new());
        let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        inputs.scenario = Some("s");
        let resolved = resolve_parameters(inputs).expect("ok");
        let chain: Vec<(f64, f64)> = resolved.params[0].overrides.iter()
            .map(|o| (o.replaced.1, o.by.1)).collect();
        assert_eq!(chain, vec![(5.0, 4.0), (4.0, 3.0), (3.0, 7.0)]);
        assert_eq!(rendered(&[&resolved]).len(), 3);
        emit_notices(&[&resolved]);
    }

    #[test]
    fn a_multi_cell_run_reports_each_override_once_with_its_range() {
        // A draw per cell over a params file: one line, the draws' range.
        let model = mk_model(vec![mk_param("beta", Some(0.1)), mk_param("gamma", Some(0.2))]);
        let file = params_file("p.toml", "beta = 0.5\ngamma = 0.3\n");
        let ffiles = vec![file.clone()];
        let (fcli, ftf, fte) = (vec![], IndexMap::new(), IndexSet::new());
        let cells: Vec<ResolvedParameters> = [0.2, 0.6, 0.4].iter().map(|&b| {
            let point = vec![("beta".to_string(), b)];
            let mut inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
            inputs.point_overrides = &point;
            resolve_parameters(inputs).expect("ok")
        }).collect();
        let refs: Vec<&ResolvedParameters> = cells.iter().collect();
        assert_eq!(rendered(&refs), vec![format!(
            "info: the draw or sweep point replaced beta = 0.5 (from {}) with 0.2 to 0.6, \
             across 3 cells", file.display())]);
        emit_notices(&refs);
    }

    // ── fit.toml [fixed] ∩ [estimate] overlap: refused ──────────────────

    #[test]
    fn fit_toml_fixed_estimate_overlap_is_refused() {
        // A name in both `[fixed]` and `[estimate]` is both estimated and
        // given a value; neither block wins, the resolver refuses.
        let model = mk_model(vec![
            mk_param("beta", Some(0.5)),
            mk_param("gamma", Some(0.1)),
        ]);
        let fcli = vec![];
        let ffiles = vec![];
        let mut ftf = IndexMap::new();
        ftf.insert("beta".into(), 0.7);
        let mut fte: IndexSet<String> = IndexSet::new();
        fte.insert("beta".into());
        fte.insert("gamma".into());
        let err = resolve_parameters(empty_inputs(&model, &fcli, &ffiles, &ftf, &fte))
            .expect_err("an overlap must be refused");
        match &err {
            ResolveError::FixedEstimateOverlap { names } => assert_eq!(names, &["beta"]),
            other => panic!("expected FixedEstimateOverlap, got {other:?}"),
        }
        assert_eq!(err.to_string(),
            "parameter `beta` is in both [estimate] and [fixed]; a parameter is either \
             estimated or given a value, not both.\n  \
             Fix: remove `beta` from [fixed] to estimate it, or from [estimate] to hold it \
             at its [fixed] value.");
    }

    #[test]
    fn disjoint_fixed_and_estimate_resolve() {
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
        assert!(resolved.estimate_set.contains("gamma"));
        assert!(!resolved.estimate_set.contains("beta"));
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

    // ── The precedence matrix (spec §1.3) ───────────────────────────────
    //
    // Every pair of tiers that can both set one parameter, each checked for
    // the four things the rule promises: the higher tier's value is the one
    // the run uses, one notice names both values and both sources, the
    // notice's severity is a warning exactly when the replaced value is a
    // fit's estimate, and `run.json`'s `parameters_provenance` records the
    // replacement. The expected notice text and record are written out here
    // rather than built with the resolver's own phrase functions, so a change
    // to either side shows up as a disagreement.

    /// A tier that can set `beta`. `StartsParams` and `StartsMle` are a
    /// single-point `--starts` rule's values (`from_params=FILE`,
    /// `from_mle=FIT`, tier 1.5); `FitEstimate` is a params file carrying a
    /// fit's `[provenance]` block (a fit's `mle_params.toml` passed back as
    /// `--params`); `ScenarioSet` and `ScenarioScale` are the two halves of
    /// tier 4.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Tier {
        StartsParams, StartsMle, FitToml, File, FitEstimate, Point, ScenarioSet,
        ScenarioScale, Cli,
    }

    impl Tier {
        const ALL: [Tier; 9] = [
            Tier::StartsParams, Tier::StartsMle,
            Tier::FitToml, Tier::File, Tier::FitEstimate, Tier::Point,
            Tier::ScenarioSet, Tier::ScenarioScale, Tier::Cli,
        ];

        /// Twice the spec §1.3 tier number (3.5 → 7), so the order is integral.
        fn rank(self) -> u8 {
            match self {
                Tier::StartsParams | Tier::StartsMle => 3,
                Tier::FitToml => 4,
                Tier::File | Tier::FitEstimate => 6,
                Tier::Point => 7,
                Tier::ScenarioSet | Tier::ScenarioScale => 8,
                Tier::Cli => 10,
            }
        }

        fn is_file(self) -> bool { matches!(self, Tier::File | Tier::FitEstimate) }
    }

    /// Whether `later` replaces `earlier` when both set one parameter:
    /// a higher tier, or a params file listed after another. A scenario's
    /// `set` and `scale` are one tier-4 value, not a pair, and a run has at
    /// most one starts rule.
    fn replaces(later: Tier, earlier: Tier) -> bool {
        later.rank() > earlier.rank() || (later.is_file() && earlier.is_file())
    }

    /// The tiers stacked onto one parameter, `beta`, in the order pushed.
    struct Stack {
        default: f64,
        starts: Option<(StartsOrigin, Vec<(String, f64)>)>,
        ftf: IndexMap<String, f64>,
        files: Vec<PathBuf>,
        point: Vec<(String, f64)>,
        set: Vec<(String, f64)>,
        scale: Vec<(String, f64)>,
        cli: Vec<(String, f64)>,
    }

    impl Stack {
        fn new(default: f64) -> Self {
            Stack {
                default,
                starts: None,
                ftf: IndexMap::new(),
                files: vec![],
                point: vec![],
                set: vec![],
                scale: vec![],
                cli: vec![],
            }
        }

        /// Add `tier` setting `beta` to `v` (for `ScenarioScale`, `v` is the
        /// factor). Returns the source the resolver must attribute it to.
        fn push(&mut self, tier: Tier, v: f64) -> ValueSource {
            let beta = || "beta".to_string();
            match tier {
                Tier::StartsParams | Tier::StartsMle => {
                    let origin = if tier == Tier::StartsParams {
                        StartsOrigin::Params { path: PathBuf::from("start.toml") }
                    } else {
                        StartsOrigin::Mle { fit: "results/fits/m".into() }
                    };
                    self.starts = Some((origin.clone(), vec![(beta(), v)]));
                    ValueSource::Starts(origin)
                }
                Tier::FitToml => { self.ftf.insert(beta(), v); ValueSource::FitTomlFixed }
                Tier::File => {
                    let path = params_file(&format!("f{}.toml", self.files.len()),
                        &format!("beta = {v}\n"));
                    self.files.push(path.clone());
                    ValueSource::FixedFile { path }
                }
                Tier::FitEstimate => {
                    let path = fit_params_file(&format!("mle{}.toml", self.files.len()),
                        &format!("beta = {v}"));
                    self.files.push(path.clone());
                    ValueSource::FixedFile { path }
                }
                Tier::Point => { self.point.push((beta(), v)); ValueSource::SweepPoint }
                Tier::ScenarioSet => { self.set.push((beta(), v)); ValueSource::Scenario("s".into()) }
                Tier::ScenarioScale => { self.scale.push((beta(), v)); ValueSource::Scenario("s".into()) }
                Tier::Cli => { self.cli.push((beta(), v)); ValueSource::FixedCli }
            }
        }

        fn resolve(&self) -> ResolvedParameters {
            let mut model = mk_model(vec![mk_param("beta", Some(self.default))]);
            let set: Vec<(&str, f64)> = self.set.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let scale: Vec<(&str, f64)> = self.scale.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let has_scenario = !set.is_empty() || !scale.is_empty();
            if has_scenario {
                model.presets.push(preset("s", &set, &scale));
            }
            let fte = IndexSet::new();
            let mut inputs = empty_inputs(&model, &self.cli, &self.files, &self.ftf, &fte);
            inputs.point_overrides = &self.point;
            if has_scenario {
                inputs.scenario = Some("s");
            }
            let starts = self.starts.as_ref()
                .map(|(origin, values)| StartsValues { origin, values });
            resolve_parameters_with_starts(inputs, starts).expect("resolves")
        }
    }

    /// How a notice names `src` as the value's origin — written out by hand.
    fn expected_origin(tier: Tier, src: &ValueSource) -> String {
        let path = match src { ValueSource::FixedFile { path } => path.display().to_string(), _ => String::new() };
        match tier {
            Tier::StartsParams => "--starts from_params=start.toml".into(),
            Tier::StartsMle => "--starts from_mle=results/fits/m, a fit's estimate".into(),
            Tier::FitToml => "fit.toml [fixed]".into(),
            Tier::File => path,
            Tier::FitEstimate => format!("{path}, a fit's estimate"),
            Tier::Point => "the draw or sweep point".into(),
            Tier::ScenarioSet | Tier::ScenarioScale => "scenario `s`".into(),
            Tier::Cli => "--param/--fixed".into(),
        }
    }

    /// How a notice names `src` as the replacing tier — written out by hand.
    fn expected_subject(tier: Tier, src: &ValueSource) -> String {
        match (tier, src) {
            (Tier::File | Tier::FitEstimate, ValueSource::FixedFile { path }) =>
                format!("params file {}", path.display()),
            _ => expected_origin(tier, src),
        }
    }

    /// The `run.json` record of one side of a replacement — written out by hand.
    fn expected_record(tier: Tier, src: &ValueSource, value: f64) -> serde_json::Value {
        let tag = match tier {
            Tier::StartsParams => "starts_from_params",
            Tier::StartsMle => "starts_from_mle",
            Tier::FitToml => "fit_toml_fixed",
            Tier::File | Tier::FitEstimate => "fixed_file",
            Tier::Point => "sweep_point",
            Tier::ScenarioSet | Tier::ScenarioScale => "scenario",
            Tier::Cli => "fixed_cli",
        };
        let mut rec = serde_json::json!({ "source": tag, "value": value });
        match src {
            ValueSource::FixedFile { path }
            | ValueSource::Starts(StartsOrigin::Params { path }) => {
                rec["path"] = path.display().to_string().into();
            }
            ValueSource::Starts(StartsOrigin::Mle { fit }) => { rec["fit"] = fit.clone().into(); }
            ValueSource::Scenario(n) => { rec["scenario"] = n.clone().into(); }
            _ => {}
        }
        rec
    }

    /// The value `tier` contributes in a matrix cell, and what `beta` is
    /// after it: a set tier sets `v`; a scale tier multiplies the current
    /// value by `factor`.
    fn contribute(tier: Tier, v: f64, factor: f64, current: f64) -> (f64, f64) {
        if tier == Tier::ScenarioScale { (factor, current * factor) } else { (v, v) }
    }

    #[test]
    fn every_tier_pair_higher_wins_is_reported_and_is_recorded() {
        // Model default 0.1; the lower tier sets 0.2 (a scale doubles 0.1),
        // the higher sets 0.3 (a scale doubles the lower tier's 0.2 to 0.4).
        // Doubling is exact in binary floating point, so every value compares
        // exactly.
        let mut pairs = 0;
        for &lo in &Tier::ALL {
            for &hi in &Tier::ALL {
                if !replaces(hi, lo) {
                    continue;
                }
                pairs += 1;
                let cell = format!("{lo:?} then {hi:?}");
                let mut st = Stack::new(0.1);
                let (lo_in, lo_val) = contribute(lo, 0.2, 2.0, 0.1);
                let lo_src = st.push(lo, lo_in);
                let (hi_in, hi_val) = contribute(hi, 0.3, 2.0, lo_val);
                let hi_src = st.push(hi, hi_in);
                let r = st.resolve();
                let beta = &r.params[0];

                // The higher tier wins.
                assert_eq!(beta.value, hi_val, "{cell}: value");
                assert_eq!(beta.source, hi_src, "{cell}: source");

                // One replacement, naming both values and both sources.
                assert_eq!(beta.overrides, vec![Override {
                    name: "beta".into(),
                    replaced: (lo_src.clone(), lo_val),
                    by: (hi_src.clone(), hi_val),
                }], "{cell}: overrides");

                // One notice, warning exactly when the replaced value is a
                // fit's estimate.
                let severity = if matches!(lo, Tier::FitEstimate | Tier::StartsMle) {
                    Severity::Warning
                } else {
                    Severity::Info
                };
                let ns = notices(&[&r]);
                let ns: Vec<&Notice> = ns.iter().collect();
                assert_eq!(ns.len(), 1, "{cell}: notices {ns:?}");
                assert_eq!(ns[0].code, "param-override", "{cell}");
                assert_eq!(ns[0].severity, severity, "{cell}: severity");
                let prefix = if severity == Severity::Warning { "warning" } else { "info" };
                assert_eq!(ns[0].render(), format!(
                    "{prefix}: {} replaced beta = {lo_val} (from {}) with {hi_val}",
                    expected_subject(hi, &hi_src), expected_origin(lo, &lo_src)),
                    "{cell}: notice text");

                // `run.json` records it.
                let rec = serde_json::to_value(
                    crate::run_meta::ParameterProvenance::from_resolved(beta)).unwrap();
                assert_eq!(rec["value"], serde_json::json!(hi_val), "{cell}");
                assert_eq!(rec["overrides"], serde_json::json!([{
                    "replaced": expected_record(lo, &lo_src, lo_val),
                    "by": expected_record(hi, &hi_src, hi_val),
                }]), "{cell}: provenance");

                // The emit path's own cross-check of notices against records.
                emit_notices(&[&r]);
            }
        }
        // 6 tier groups of sizes 2, 1, 2, 1, 2, 1 → 33 cross-tier pairs, plus
        // the four orders of two params files (plain or a fit's estimate).
        assert_eq!(pairs, 37);
    }

    #[test]
    fn an_equal_value_from_any_higher_tier_is_not_an_override() {
        // Default 0.15; every tier lands on 0.3 (a lower scale doubles 0.15, a
        // higher scale multiplies by 1).
        for &lo in &Tier::ALL {
            for &hi in &Tier::ALL {
                if !replaces(hi, lo) {
                    continue;
                }
                let cell = format!("{lo:?} then {hi:?}");
                let mut st = Stack::new(0.15);
                let (lo_in, lo_val) = contribute(lo, 0.3, 2.0, 0.15);
                st.push(lo, lo_in);
                let (hi_in, hi_val) = contribute(hi, 0.3, 1.0, lo_val);
                let hi_src = st.push(hi, hi_in);
                assert_eq!((lo_val, hi_val), (0.3, 0.3), "{cell}: fixture");
                let r = st.resolve();
                assert_eq!(r.params[0].value, 0.3, "{cell}");
                assert_eq!(r.params[0].source, hi_src, "{cell}: the later tier is still the source");
                assert!(r.params[0].overrides.is_empty(), "{cell}: {:?}", r.params[0].overrides);
                assert!(rendered(&[&r]).is_empty(), "{cell}: {:?}", rendered(&[&r]));
                let rec = serde_json::to_value(
                    crate::run_meta::ParameterProvenance::from_resolved(&r.params[0])).unwrap();
                assert!(rec.get("overrides").is_none(), "{cell}: {rec}");
            }
        }
    }

    #[test]
    fn replacing_the_model_default_from_any_tier_is_not_an_override() {
        for &tier in &Tier::ALL {
            let mut st = Stack::new(0.1);
            let (v, want) = contribute(tier, 0.3, 2.0, 0.1);
            let src = st.push(tier, v);
            let r = st.resolve();
            assert_eq!(r.params[0].value, want, "{tier:?}");
            assert_eq!(r.params[0].source, src, "{tier:?}");
            assert!(r.params[0].overrides.is_empty(), "{tier:?}: {:?}", r.params[0].overrides);
            assert!(rendered(&[&r]).is_empty(), "{tier:?}");
        }
    }

    #[test]
    fn all_six_tiers_on_one_parameter_replace_in_spec_order() {
        // fit.toml [fixed] → params file → a fit's estimate file → draw/sweep
        // point → scenario (set, then scale: one value) → --param.
        let mut st = Stack::new(0.1);
        let ftf = st.push(Tier::FitToml, 0.2);
        let file = st.push(Tier::File, 0.25);
        let mle = st.push(Tier::FitEstimate, 0.3);
        let point = st.push(Tier::Point, 0.35);
        let scen = st.push(Tier::ScenarioSet, 0.4);
        st.push(Tier::ScenarioScale, 2.0);
        let cli = st.push(Tier::Cli, 0.9);
        let r = st.resolve();
        let beta = &r.params[0];
        assert_eq!((beta.value, &beta.source), (0.9, &cli));
        let chain: Vec<(ValueSource, f64, ValueSource, f64)> = beta.overrides.iter()
            .map(|o| (o.replaced.0.clone(), o.replaced.1, o.by.0.clone(), o.by.1))
            .collect();
        assert_eq!(chain, vec![
            (ftf, 0.2, file.clone(), 0.25),
            (file, 0.25, mle.clone(), 0.3),
            (mle, 0.3, point.clone(), 0.35),
            (point, 0.35, scen.clone(), 0.8),
            (scen, 0.8, cli, 0.9),
        ]);
        let severities: Vec<Severity> = notices(&[&r]).iter().map(|n| n.severity).collect();
        assert_eq!(severities, vec![
            Severity::Info, Severity::Info, Severity::Warning, Severity::Info, Severity::Info,
        ], "only the replacement of the fit's estimate is a warning");
        let rec = serde_json::to_value(
            crate::run_meta::ParameterProvenance::from_resolved(beta)).unwrap();
        assert_eq!(rec["overrides"].as_array().map(Vec::len), Some(5));
        emit_notices(&[&r]);
    }

    // ── A single-point `--starts` rule's values (tier 1.5) ──────────────

    #[test]
    fn starts_values_supply_an_unset_parameter_and_ignore_undeclared_names() {
        // `beta` has no model default; the starts file supplies it, so the
        // resolver does not fire `UnsetRequired`. `delta` is not a model
        // parameter: a starting point may name one, and it is skipped.
        let model = mk_model(vec![mk_param("beta", None), mk_param("gamma", Some(0.1))]);
        let origin = StartsOrigin::Params { path: PathBuf::from("start.toml") };
        let values = vec![("beta".to_string(), 0.5), ("delta".to_string(), 9.0),
                          ("gamma".to_string(), 0.2)];
        let (fcli, ffiles, ftf, fte) = (vec![], vec![], IndexMap::new(), IndexSet::new());
        let inputs = empty_inputs(&model, &fcli, &ffiles, &ftf, &fte);
        let r = resolve_parameters_with_starts(
            inputs, Some(StartsValues { origin: &origin, values: &values })).expect("resolves");
        let by: HashMap<&str, &ResolvedParameter> =
            r.params.iter().map(|p| (p.name.as_str(), p)).collect();
        assert_eq!((by["beta"].value, &by["beta"].source),
            (0.5, &ValueSource::Starts(origin.clone())));
        // The file's value beats the model default, and replacing the default
        // is not an override.
        assert_eq!((by["gamma"].value, &by["gamma"].source),
            (0.2, &ValueSource::Starts(origin)));
        assert!(by["gamma"].overrides.is_empty());
        assert!(rendered(&[&r]).is_empty());
    }

    #[test]
    fn sweep_over_draws_reports_each_swept_parameter_once_as_a_range() {
        let draw = |k: f64, beta: f64| -> IndexMap<String, f64> {
            [("k".to_string(), k), ("beta".to_string(), beta)].into_iter().collect()
        };
        let draws = [draw(9.5, 0.3), draw(10.5, 0.31), draw(10.0, 0.29)];
        let points = vec![
            vec![("k".to_string(), 12.0)],
            vec![("k".to_string(), 14.0)],
        ];
        let ns = sweep_over_draws_notices(draws.iter(), &points);
        let lines: Vec<String> = ns.iter().map(Notice::render).collect();
        assert_eq!(lines, vec![
            "info: --sweep replaced k = 9.5 to 10.5 (from the posterior draws) with 12 to 14, \
             across 3 draws".to_string(),
        ]);
        assert!(ns.iter().all(|n| n.code == "param-override"));

        // One value on each side: no range, no count.
        let fixed = [draw(10.0, 0.3), draw(10.0, 0.31)];
        let one = vec![vec![("k".to_string(), 12.0)]];
        let lines: Vec<String> = sweep_over_draws_notices(fixed.iter(), &one)
            .iter().map(Notice::render).collect();
        assert_eq!(lines, vec![
            "info: --sweep replaced k = 10 (from the posterior draws) with 12".to_string(),
        ]);

        // A sweep onto the value every draw already has replaces nothing.
        let same = vec![vec![("k".to_string(), 10.0)]];
        assert!(sweep_over_draws_notices(fixed.iter(), &same).iter().next().is_none());
        // No sweep: the one empty point.
        assert!(sweep_over_draws_notices(fixed.iter(), &[vec![]]).iter().next().is_none());
    }
}
