//! Fit.toml schema types (run-spec v0.4).
//!
//! The single fit-config schema. The legacy v1 `FitToml` and the
//! `to_legacy_toml()` bridge were deleted in the v1-cleanup pass —
//! `camdl fit run` (the only remaining entry point) consumes
//! `FitConfigV2` directly.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

// ─── Top-level ──────────────────────────────────────────────────────────────

/// A fit.toml v2 — single inference task with named stages.
///
/// `deny_unknown_fields` (gh#173): a misplaced or typo'd top-level key is a
/// hard error, not a silent drop. The honored `dt` lives under `[config]`; a
/// top-level `dt` used to be silently ignored (dt=1/2/5 gave byte-identical
/// fits — a wasted timing experiment). The same strictness is applied to the
/// nested config structs below, except `FixedParams`, whose `#[serde(flatten)]`
/// for arbitrary `param = value` entries is incompatible with — and the very
/// opposite of — `deny_unknown_fields`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfigV2 {
    pub model: ModelRef,

    /// Real-data source. Mutually exclusive with `[synthetic]`: exactly
    /// one of the two must be present. `validate()` enforces this.
    #[serde(default)]
    pub data: Option<DataSpec>,

    /// Synthetic-data source — generates N datasets from known truth and
    /// fits each one (simulation-based calibration). See proposal
    /// docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md §"Config
    /// shape".
    #[serde(default)]
    pub synthetic: Option<SyntheticSpec>,

    /// IF2/PGAS seeds. A list (`[42]` for a single fit, `[101, 102, 103]`
    /// for start-sensitivity sweeps). When absent, the top-level
    /// `--seed` CLI flag (or its default) is used as the single seed.
    /// Duplicates are rejected at validation time — each seed must be
    /// unique to avoid provenance-hash collisions.
    #[serde(default)]
    pub fit_seeds: Option<Vec<u64>>,

    /// Simplex constraints between estimated parameters. Each group's
    /// members must appear in `[estimate]`, be non-negative, and form
    /// a probability simplex (sum = 1). This is a *parameter-space
    /// property*, not an algorithm knob — algorithms read it.
    ///
    /// IF2 perturbs members jointly via barycentric (log-ratio + softmax)
    /// transform; a member's `rw_sd` is interpreted on the log-ratio
    /// scale. PGAS / PMMH / PFilter currently treat members as
    /// independent and rely on the model to enforce sum = 1 indirectly
    /// — `validate()` warns when a non-IF2 stage runs against a fit
    /// that declares simplex groups.
    ///
    /// Forward-compat note: the natural prior on a simplex is Dirichlet,
    /// which lives at the *group* level (one prior over k correlated
    /// quantities). The schema accommodates a future `prior` field on
    /// `SimplexGroup` without breaking changes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub simplex_groups: Vec<SimplexGroup>,

    /// How the initial parameter point is chosen for each fit. Default
    /// matches today's behaviour (`model_default` — start from the
    /// model's declared values). `"prior"` draws from declared priors.
    #[serde(default)]
    pub fit_starts: Option<FitStarts>,

    #[serde(default)]
    pub output_dir: Option<String>,

    /// The free parameters: what the inference algorithm estimates.
    pub estimate: IndexMap<String, EstimateSpecV2>,

    /// The fixed parameters: held constant during inference.
    /// estimate ∪ fixed must cover all model parameters.
    pub fixed: FixedParams,

    /// Inference pipeline stages, executed in declaration order.
    pub stages: IndexMap<String, Stage>,

    /// Backend and time step. Defaults: chain_binomial, dt=1.0.
    #[serde(default)]
    pub config: FitBackendConfig,

    /// Named scenario from the model. Applies scenario's enable/disable lists
    /// and param overrides before inference. Mutually exclusive with
    /// `enable`/`disable`. Per spec §14.4, toggleable interventions default
    /// OFF; events always fire unless explicitly disabled.
    #[serde(default)]
    pub scenario: Option<String>,
    /// Ad-hoc enable list (intervention names or family base_names).
    /// Wildcard `"*"` enables every toggleable intervention.
    #[serde(default)]
    pub enable: Vec<String>,
    /// Ad-hoc disable list. Explicit disable wins over always_active —
    /// the only way to silence an event during inference.
    #[serde(default)]
    pub disable: Vec<String>,

    /// IC-free inference: condition the likelihood on the first
    /// observation rather than an initial-state commitment. Absent or
    /// false means standard inference over `y_{1:T}` with a committed
    /// initial state. True means the PF / IF2 / PGAS weight-and-resample
    /// at y₁ (pinning the initial state) but accumulate log-likelihood
    /// only from y₂ onward. Requires at least one `[estimate.*]` entry
    /// with `ivp = true` to give particles spread at t=0.
    ///
    /// See docs/dev/proposals/2026-04-18-ic-free-inference.md.
    #[serde(default)]
    pub ic_free: Option<bool>,

    /// Burn-in / conditioning window (gh#134). The model is simulated
    /// faithfully over the leading span `[t_start, cond_from)` — full process
    /// noise, interventions, forcings — but **nothing there is scored**, and
    /// the incidence accumulator is reset at `cond_from`, so the first scored
    /// incidence bin is `(cond_from, first_obs]` rather than the whole
    /// `[t_start, first_obs]` gap. Mechanically this inserts `cond_from` as a
    /// leading reset-only HOLE on the observation grid (reset, no likelihood
    /// term — the same machinery sparse-obs `NA` cells use), so PF / IF2 /
    /// PGAS / PMMH all get it through the shared `BoundObs`/obs grid.
    ///
    /// Per-stream and explicit (multi-cadence Phase 3). The conditioning window
    /// is resolved **per incidence stream** keyed on its observation-block label
    /// (the `[data.observations]` key / IR `source`). Two surface forms (see
    /// [`ConditionFrom`]):
    ///
    /// - `condition_from = "first_obs - 1 week"` — a single spec applied as the
    ///   default for **every** stream;
    /// - `[condition_from]` — a table whose optional `default` key is the
    ///   all-streams default and whose other keys *shadow* individual streams by
    ///   label (`es = "first_obs - 2 weeks"`).
    ///
    /// A stream with no shadow and no `default` resolves to NO conditioning. The
    /// wide-first-window detector (`W329`,
    /// `crate::util::check_first_interval_window`) is the enforcer: a
    /// late-starting incidence stream that resolves to no conditioning
    /// HARD-ERRORS, naming `condition_from.<label>` as the fix.
    /// There is no automatic / inferred boundary — the boundary comes only from
    /// an explicit spec.
    ///
    /// Each per-stream value accepts:
    /// - absolute model-time number string (`"14"`),
    /// - absolute `date("…")` / bare ISO date (`"date(\"2020-02-01\")"`),
    ///   resolved via the model origin + `time_unit`,
    /// - relative `"first_obs - <N> <unit>"` (`"first_obs - 1 week"`),
    ///   resolved as `first_obs_s − N·unit` against THAT stream's first obs.
    ///
    /// Validation (in `FitRunConfig::build`): each resolved `cond_from_s ∈
    /// [t_start, first_obs_s)`. `cond_from_s == t_start` (or no spec) is a no-op
    /// (bit-identical — no hole inserted). Orthogonal to `ic_free`; setting BOTH
    /// errors loudly (the inserted leading hole trips the existing "nothing to
    /// condition on" guard).
    ///
    /// `skip_serializing_if None` keeps it OUT of the fit identity hash when
    /// unset, so existing fits' `run_id`s are unchanged. A *set* value re-keys
    /// the fit (a different conditioning window is a different fit / estimand).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_from: Option<ConditionFrom>,

    /// Optional lineage metadata (not used by the runner).
    #[serde(default)]
    pub provenance: Option<FitProvenance>,

    /// Runtime-only: the path to the model **already compiled to IR**
    /// (`.ir.json`). `cmd_fit_run_v2` compiles `model.camdl` → IR exactly
    /// once up front and records the temp path here; every per-stage
    /// `FitRunConfig::build` then loads this pre-compiled IR instead of
    /// re-invoking camdlc per (cell × sweep point × stage). `None` means
    /// "compile from `model.camdl`" (the fallback for unit tests that build a
    /// config directly). Never serialized — `model.camdl` remains the sole
    /// identity-bearing source path (the fit content hash hashes its bytes).
    #[serde(skip)]
    pub compiled_ir: Option<String>,
}

/// The user-facing `condition_from` value before resolution to model time.
///
/// Per-stream and explicit (multi-cadence Phase 3). Two surface forms,
/// dispatched on the TOML value type (a string vs a table):
///
/// - [`ConditionFrom::All`] — `condition_from = "first_obs - 1 week"`. One spec
///   used as the default for **every** stream; no per-stream shadows.
/// - [`ConditionFrom::PerStream`] — `[condition_from]`. A table mapping a
///   reserved `default` key (the all-streams default, optional) and/or
///   observation-block labels (per-stream shadows) to spec strings. A stream
///   resolves to its shadow if present, else `default`, else NO conditioning.
///
/// Each spec string accepts the same three forms (resolved per stream by
/// [`crate::fit::runner::resolve_condition_from`]): a bare model-time number
/// (`"14"`), an absolute calendar date (`date("YYYY-MM-DD")` or a bare
/// `"YYYY-MM-DD"`, resolved via the model origin + `time_unit`), or a relative
/// offset off that stream's first observation (`"first_obs - <N> <unit>"`).
///
/// `#[serde(untagged)]`: a TOML string deserializes to `All`, a TOML table to
/// `PerStream`. The `BTreeMap` keeps the table key order stable, so the
/// round-trip through the fit-identity hash is deterministic.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ConditionFrom {
    /// `condition_from = "<spec>"` — one spec, the default for every stream.
    All(String),
    /// `[condition_from]` — `default = "<spec>"` (all-streams default, optional)
    /// plus zero or more `<label> = "<spec>"` per-stream shadows. The `default`
    /// key is reserved; a stream literally named `default` collides (a hard
    /// error in [`ConditionFrom::resolve_for`]'s caller).
    PerStream(std::collections::BTreeMap<String, String>),
}

/// The reserved key in a `[condition_from]` table that names the all-streams
/// default (vs a per-stream shadow). A stream whose observation-block label is
/// literally this string collides with the reserved key.
pub const CONDITION_FROM_DEFAULT_KEY: &str = "default";

impl ConditionFrom {
    /// Resolve the conditioning spec string for the stream labelled `label`
    /// (its observation-block label / IR `source`). Returns the per-stream
    /// shadow if present, else the all-streams default, else `None` (no
    /// conditioning for this stream). Resolution to a concrete model-time
    /// boundary (and `[t_start, first_obs_s)` validation) is the caller's job
    /// via [`crate::fit::runner::resolve_condition_from`].
    pub fn resolve_for(&self, label: &str) -> Option<&str> {
        match self {
            ConditionFrom::All(spec) => Some(spec.as_str()),
            ConditionFrom::PerStream(map) => map
                .get(label)
                .or_else(|| map.get(CONDITION_FROM_DEFAULT_KEY))
                .map(String::as_str),
        }
    }

    /// Validate the `[condition_from]` shadow labels against the set of real
    /// observation-stream labels (`valid_labels`, the distinct IR `source`s of
    /// the bound streams). Two hard errors (located, naming the valid labels):
    ///
    /// 1. an unknown shadow label (a typo'd stream name) — listing the valid
    ///    labels so the user can correct it;
    /// 2. a stream literally named `default`, which collides with the reserved
    ///    all-streams-default key.
    ///
    /// `All(_)` has no labels to validate, so it always passes. Returns `Ok(())`
    /// for the no-op cases.
    pub fn validate_labels(&self, valid_labels: &[String]) -> Result<(), String> {
        let map = match self {
            ConditionFrom::All(_) => return Ok(()),
            ConditionFrom::PerStream(map) => map,
        };
        // (2) A stream named `default` is indistinguishable from the reserved
        //     all-streams-default key — refuse rather than silently shadow.
        if valid_labels.iter().any(|l| l == CONDITION_FROM_DEFAULT_KEY) {
            return Err(format!(
                "[condition_from]: an observation stream is labelled \
                 '{CONDITION_FROM_DEFAULT_KEY}', which collides with the \
                 reserved all-streams-default key. Rename the stream's \
                 observation-block label (its `[data.observations]` key) so it \
                 is not '{CONDITION_FROM_DEFAULT_KEY}'."
            ));
        }
        // (1) Every shadow key must name a real stream (or be `default`).
        for key in map.keys() {
            if key == CONDITION_FROM_DEFAULT_KEY {
                continue;
            }
            if !valid_labels.iter().any(|l| l == key) {
                let mut labels: Vec<&str> = valid_labels.iter().map(String::as_str).collect();
                labels.sort_unstable();
                return Err(format!(
                    "[condition_from]: '{key}' is not an observation stream. \
                     `condition_from.<label>` shadows a stream by its \
                     observation-block label; valid labels are: {} (or \
                     '{CONDITION_FROM_DEFAULT_KEY}' for the all-streams \
                     default).",
                    labels.join(", ")
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub camdl: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitBackendConfig {
    #[serde(default = "default_dt")]
    pub dt: f64,
    /// How observation times relate to the `dt` grid. `None` = "exact where the
    /// algorithm supports it" (today's behaviour). Gated per algorithm by
    /// `crate::fit::methods::resolve_obs_alignment`. See the unified-timeline
    /// proposal (Stage 2). `skip_serializing_if None` keeps it OUT of the fit
    /// identity hash when unset, so existing fits' `run_id`s are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obs_alignment: Option<crate::fit::methods::ObsAlignment>,
    /// gh#audit-C6 / S1. Treat a numerical collapse in a rate expression
    /// (div-by-zero, `Pow`→NaN/Inf, `Sqrt` of a negative, any unary→NaN) as
    /// `0.0` instead of a hard error. Genuinely semantic — it changes the
    /// trajectory — so it lives in the keyed `[config]` rather than as an
    /// ephemeral CLI flag (which would bypass the fit-identity hash). Rarely
    /// needed for fits (the particle filter kills NaN-rate particles via
    /// per-particle recovery); kept for forward-sim parity and testing.
    /// `skip_serializing_if` keeps the common `false` out of the identity hash,
    /// so existing fits don't re-key.
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_degenerate_rates: bool,
}
fn default_backend() -> crate::args::types::ForwardBackend {
    crate::args::types::ForwardBackend::ChainBinomial
}
fn default_dt() -> f64 { 1.0 }
fn is_false(b: &bool) -> bool { !*b }
impl Default for FitBackendConfig {
    fn default() -> Self {
        FitBackendConfig {
            dt: default_dt(),
            obs_alignment: None,
            allow_degenerate_rates: false,
        }
    }
}

// ─── Data ───────────────────────────────────────────────────────────────────

/// Data file mapping. Keys in `observations` match observation stream names
/// declared in the .camdl file's `observations { }` block. The observation
/// model (likelihood family) and projection (which flow/compartment to
/// accumulate) are defined in the .camdl file — fit.toml only provides the
/// data file paths.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataSpec {
    /// Single-file shorthand: every observation stream declared in the
    /// model expects a column with the same name in this TSV.
    ///
    /// Mutually exclusive with `observations`. Use this form for
    /// stratified models where one wide TSV holds all the columns
    /// (e.g. an indexed `cases[a in age]` block expanding to 5 stream
    /// names → 5 columns in one file). Avoids the per-stream
    /// `cases_a02 = "x.tsv"` / `cases_a25 = "x.tsv"` repetition that
    /// would otherwise be N copies of the same path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,

    /// Per-stream form: explicit map from observation stream name →
    /// data file path. Mutually exclusive with `file`. Use this form
    /// when streams genuinely come from different files (e.g.
    /// observation streams from different surveillance systems).
    #[serde(default)]
    pub observations: IndexMap<String, String>,

    /// Time threshold for temporal holdout: observations at t > this value
    /// are withheld from training. In model time units.
    /// Mutually exclusive with `holdout`.
    #[serde(default)]
    pub holdout_after: Option<f64>,

    /// Explicit holdout data files. Keys match observation stream names.
    /// Mutually exclusive with `holdout_after`.
    #[serde(default)]
    pub holdout: Option<IndexMap<String, String>>,
}

impl DataSpec {
    /// Exactly one of `file` / `observations` must be set.
    pub fn validate(&self) -> Result<(), String> {
        match (self.file.is_some(), !self.observations.is_empty()) {
            (true, true) => Err(
                "[data]: `file = \"...\"` and `[data.observations]` are mutually \
                 exclusive — choose one. Use `file` when one wide TSV holds all \
                 streams; use `[data.observations]` when streams come from \
                 different files.".to_string()),
            (false, false) => Err(
                "[data]: must specify either `file = \"<path>\"` (one wide TSV \
                 with columns matching the model's declared observation streams) \
                 or `[data.observations]` (per-stream file paths).".to_string()),
            _ => Ok(()),
        }
    }

    /// Resolve this spec into the canonical per-stream map, given the
    /// names of the model's declared observation streams. The single-
    /// file shorthand expands by mapping every model-declared stream
    /// to the same file.
    ///
    /// Errors if the resolved map is empty (no streams declared in the
    /// model).
    pub fn effective_observations(
        &self,
        model_obs_names: &[String],
    ) -> Result<IndexMap<String, String>, String> {
        let map = if let Some(file) = &self.file {
            if model_obs_names.is_empty() {
                return Err(format!(
                    "[data] file = \"{}\" but the model declares no observation \
                     streams. Either add an `observations {{ }}` block to the \
                     .camdl file, or remove [data] from fit.toml.",
                    file));
            }
            let mut out = IndexMap::new();
            for name in model_obs_names {
                out.insert(name.clone(), file.clone());
            }
            out
        } else {
            self.observations.clone()
        };
        Ok(map)
    }
}

// ─── Synthetic data ──────────────────────────────────────────────────────────

// ─── Simplex groups ─────────────────────────────────────────────────────────

/// A group of estimated parameters that must form a probability simplex
/// (non-negative, summing to 1). See `FitConfigV2.simplex_groups` for
/// the full design.
///
/// CLI-side type: members are listed by name. At fit-config build time
/// names are resolved to model param indices, and `rw_sd` is read from
/// each member's `EstimateSpecV2.rw_sd` (or auto-derived) — the runtime
/// `sim::inference::if2::SimplexGroup` carries indices + rw_sds on the
/// log-ratio scale.
///
/// Schema is forward-compatible with a future `prior:
/// MultivariatePriorSpec` field for Dirichlet support.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimplexGroup {
    /// Parameter names that form a probability simplex (sum = 1).
    /// Each must appear in `[estimate]`. Order is preserved for
    /// reproducible barycentric encoding (the perturbation result
    /// depends on member order via the log-ratio's reference index).
    pub params: Vec<String>,
}

/// Synthetic-data generation spec. Mutually exclusive with `[data]`:
/// when present, the runner generates `len(sim_seeds)` datasets from
/// `true_params` using the model's observation block, then fits each
/// one. Output directory structure places these under `synthetic/ds_NN/`
/// — see docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticSpec {
    /// Path to a TOML file of `name = value` lines supplying the ground
    /// truth used to generate data and to compute coverage / bias.
    pub true_params: String,

    /// Simulation seeds. Either a range string (`"1:20"`) or an explicit
    /// list (`[7, 42, 101, ...]`). Duplicates are rejected.
    pub sim_seeds: SeedsSpec,

    /// Number of datasets. When omitted, inferred from `len(sim_seeds)`.
    /// When supplied, must equal that length.
    #[serde(default)]
    pub datasets: Option<usize>,

    /// Scenario for data generation (not for fitting). Applies the
    /// named scenario's enable/disable lists and param overrides when
    /// generating synthetic datasets. Fits themselves run against the
    /// scenario-free baseline (unless the top-level `scenario =` is also
    /// set, in which case that applies at fit time).
    #[serde(default)]
    pub scenario: Option<String>,

    /// Forward-simulation backend used to GENERATE the synthetic datasets
    /// (`chain_binomial` | `gillespie` | `ode`). This is a property of data
    /// generation, not of fitting — fit stages declare their own backends.
    /// Relocated from `[config].backend` (gh#241): the backend only ever fed
    /// synthetic generation, so it belongs in the block that owns generation.
    #[serde(default = "default_backend")]
    pub backend: crate::args::types::ForwardBackend,
}

impl SyntheticSpec {
    pub fn validate(&self) -> Result<(), String> {
        // Ensure sim_seeds is non-empty and has no duplicates.
        let seeds = self.sim_seeds.to_vec()
            .map_err(|e| format!("[synthetic] sim_seeds: {}", e))?;
        if seeds.is_empty() {
            return Err("[synthetic] sim_seeds is empty — at least one seed required".into());
        }
        self.sim_seeds.validate_no_duplicates().map_err(|e| format!("[synthetic] {}", e))?;

        if let Some(n) = self.datasets {
            if n != seeds.len() {
                return Err(format!(
                    "[synthetic] datasets = {} but sim_seeds has length {}. \
                     These must match, or omit `datasets` to infer from sim_seeds.",
                    n, seeds.len()));
            }
            if n == 0 {
                return Err("[synthetic] datasets must be ≥ 1".into());
            }
        }
        Ok(())
    }

}

/// Simulation-seeds spec: an explicit list or a range string (`"1:20"`).
/// Custom Deserialize dispatches on the TOML value type directly.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SeedsSpec {
    /// Explicit list of seeds.
    List(Vec<u64>),
    /// Range string, e.g. `"1:20"` meaning `[1, 2, ..., 20]` inclusive.
    Range(String),
}

impl<'de> Deserialize<'de> for SeedsSpec {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let v = toml::Value::deserialize(de)?;
        match v {
            toml::Value::String(s) => Ok(SeedsSpec::Range(s)),
            toml::Value::Array(xs) => {
                if xs.is_empty() {
                    return Err(D::Error::custom("seeds list must be non-empty"));
                }
                let mut out = Vec::with_capacity(xs.len());
                for item in xs {
                    match item {
                        toml::Value::Integer(n) if n >= 0 => out.push(n as u64),
                        toml::Value::Integer(n)           => return Err(D::Error::custom(
                            format!("seed must be non-negative, got {}", n))),
                        other => return Err(D::Error::custom(
                            format!("seeds list must contain integers, got {:?}", other))),
                    }
                }
                Ok(SeedsSpec::List(out))
            }
            other => Err(D::Error::custom(format!(
                "expected a range string like \"1:20\" or a list of integers; got {:?}",
                other))),
        }
    }
}

impl SeedsSpec {
    /// Expand to a concrete list. Parses the range form on demand.
    /// Returns `Err` on malformed range strings (typo `"1-20"` instead
    /// of `"1:20"`, inverted bounds `"20:1"`, non-integer tokens) so a
    /// silently-empty fit replicate set is impossible.
    pub fn to_vec(&self) -> Result<Vec<u64>, String> {
        match self {
            SeedsSpec::List(xs) => Ok(xs.clone()),
            SeedsSpec::Range(s) => parse_seed_range(s).ok_or_else(|| format!(
                "malformed seed range '{}' — use 'start:end' with \
                 start ≤ end, e.g. '1:20'", s)),
        }
    }

    pub fn validate_no_duplicates(&self) -> Result<(), String> {
        let v = self.to_vec()?;
        let mut seen = BTreeSet::new();
        for s in &v {
            if !seen.insert(*s) {
                return Err(format!(
                    "duplicate seed {} — each seed must be unique to avoid \
                     provenance-hash collisions between fits", s));
            }
        }
        Ok(())
    }
}

/// Parse `"N:M"` into `[N, N+1, ..., M]` inclusive.
/// Errors (returning None) when the form is malformed or inverted.
fn parse_seed_range(s: &str) -> Option<Vec<u64>> {
    let (lo, hi) = s.split_once(':')?;
    let lo: u64 = lo.trim().parse().ok()?;
    let hi: u64 = hi.trim().parse().ok()?;
    if lo > hi { return None; }
    Some((lo..=hi).collect())
}

/// How initial parameter points are chosen for each fit-seed replicate.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FitStarts {
    /// Start from the model's declared parameter values (default).
    #[default]
    ModelDefault,
    /// Draw starts from declared priors. Errors if any estimated
    /// parameter lacks a prior.
    Prior,
    // LatinHypercube is reserved; not implemented in the initial landing.
}


// ─── Estimate ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateSpecV2 {
    /// Search bounds. When `None`, the model file's
    /// `parameters { foo : rate in [lo, hi] }` declaration is the source
    /// of truth — `build_if2_params_from_specs` already handles the
    /// fit-toml-bounds-tighten-but-not-loosen rule and falls back to
    /// model bounds when the toml side is absent. Set explicitly in
    /// fit.toml only when you want to *narrow* the search relative
    /// to the model's declared range.
    #[serde(default)]
    pub bounds: Option<(f64, f64)>,

    /// Transform for inference. If omitted, inferred from the parameter's
    /// declared type in the .camdl file.
    #[serde(default)]
    pub transform: Option<Transform>,

    /// Prior distribution. Required for Bayesian methods (PGAS, PMMH).
    /// Optional for MLE (IF2 ignores priors).
    ///
    /// Wire format is externally-tagged (matches the OCaml IR emission):
    ///   `prior = { log_normal = { mu = 0.0, sigma = 1.0 } }`
    ///
    /// gh#75: an explicit flat-prior opt-in is also recognized:
    ///   `prior = { flat = {} }`
    /// The flat variant is *only* meaningful in fit tomls — there is
    /// no DSL `~ flat(...)` syntax. Honored by Bayesian stages as
    /// the "I want flat priors here, on purpose" declaration. The
    /// runner emits provenance `flat_explicit` for each such param.
    ///
    /// See [`EstimatePriorSpec`] for the typed wrapper.
    #[serde(default)]
    pub prior: Option<EstimatePriorSpec>,

    /// Initial value parameter: perturbed only at t=0 in IF2.
    #[serde(default)]
    pub ivp: bool,

    /// Per-parameter random walk SD for IF2. If omitted, auto-scaled from bounds.
    #[serde(default)]
    pub rw_sd: Option<f64>,

    /// Starting value override. If omitted, random from bounds (scout) or
    /// from starts_from (downstream stages).
    #[serde(default)]
    pub start: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Log,
    Logit,
    Identity,
}

impl Transform {
    /// String form expected by `runner::derive_transform`'s
    /// `transform_override` argument. The runner still threads
    /// transforms as `Option<&str>` internally; this is the
    /// thin v2-typed → str adapter so callers don't allocate.
    pub fn as_str(&self) -> &'static str {
        match self {
            Transform::Log => "log",
            Transform::Logit => "logit",
            Transform::Identity => "identity",
        }
    }
}

// Prior specification for `[estimate.<name>.prior]` is `ir::parameter::PriorDist`.
//
// One serialization form across the workspace: the externally-tagged
// enum the OCaml compiler already emits for in-model `~`-syntax priors
// (`{ log_normal = { mu = 0, sigma = 1 } }`). Re-exported here so
// downstream `use config_v2::PriorDist` imports keep working without
// touching the `ir` crate dependency directly.
pub use ir::parameter::PriorDist;

/// gh#75: Marker for explicit flat-prior opt-in in fit tomls.
///
/// Deserializes from `{}` (an empty TOML inline table). Used inside
/// [`EstimatePriorSpec::Flat`] to carry the "I have written this"
/// opt-in marker while preserving room for future flat-prior fields
/// (e.g. an explicit `support = [lo, hi]` improper-uniform window)
/// without breaking the wire format.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct FlatMarker {}

/// gh#75: Fit-toml-side prior specification. Either a regular
/// distribution from the IR's prior catalogue, or an explicit opt-in
/// to a flat (improper uniform) prior via `prior = { flat = {} }`.
///
/// Flat is a fit-toml-only concept: there is no DSL `~ flat(...)`
/// syntax in `.camdl` model files (flat means "no prior", so
/// declaring it inside a parameter block would be a contradiction).
/// `EstimatePriorSpec::Flat` thus lives only at the fit-toml layer
/// — the IR `PriorDist` is unchanged.
///
/// Why a flat opt-in exists at all: `camdl fit run`'s validator
/// rejects estimated parameters that lack a resolved prior (per
/// gh#75's "Flat-fallback fires as an Error" rule). Users who
/// genuinely want flat priors (because the chain is meant to target
/// the unconditioned likelihood / scaled-likelihood posterior)
/// declare the choice accountably — the TOML records the
/// intent, `run_meta.json` records `flat_explicit` as the resolved
/// source, no warning fires.
///
/// Wire format
/// -----------
///
///   prior = { log_normal = { mu = -0.3, sigma = 0.5 } }   # → Dist(...)
///   prior = { uniform = {} }                              # → UniformOverBounds
///   prior = { flat = {} }                                 # → Flat
///
/// Deserialization is `untagged`: serde tries `Dist(PriorDist)` first
/// (matches every `{ uniform / normal / ... = { ... } }` shape with all
/// fields present), then `UniformOverBounds` (the empty `{ uniform = {} }`,
/// which `Dist` rejects for missing lower/upper), then the explicit-flat
/// struct variant. The distinct keys (`uniform` vs `flat`) and the
/// all-fields-present rule keep the shapes from colliding.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum EstimatePriorSpec {
    /// A standard distribution declared via the IR's PriorDist wire
    /// format. Matches everything `~ <dist>(...)` syntax in `.camdl`
    /// can emit.
    Dist(PriorDist),
    /// `prior = { uniform = {} }` — uniform over the parameter's bounds
    /// (the fit's `bounds`, falling back to the model's `in [lo, hi]`).
    /// Resolved to a concrete `Uniform { lower, upper }` against those
    /// bounds; errors at validation if neither source supplies them. The
    /// empty table is what distinguishes it from the explicit
    /// `{ uniform = { lower, upper } }` form (which matches `Dist` first).
    UniformOverBounds {
        uniform: UniformOverBoundsMarker,
    },
    /// Explicit flat-prior opt-in. Matches the wire form
    /// `prior = { flat = {} }`. The struct variant's `flat` field
    /// is the empty marker; the field name itself is the tag.
    Flat {
        flat: FlatMarker,
    },
}

/// Empty marker for `prior = { uniform = {} }`. `deny_unknown_fields` so a
/// half-specified `{ uniform = { lower = .. } }` does NOT silently match the
/// bounds-derived form — it falls through (and `Dist` rejects it for the
/// missing field, surfacing the mistake).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniformOverBoundsMarker {}

// (Previously: `impl EstimatePriorSpec { pub fn is_flat(&self) }` —
// removed in gh#86 when `--draws prior` switched to the unified
// precedence resolver, which handles the explicit-flat case via
// `PriorSource::FlatExplicit`. Callers that need the bool predicate
// inline can use `matches!(spec, EstimatePriorSpec::Flat { .. })`.)

// ─── Fixed ──────────────────────────────────────────────────────────────────

/// Fixed parameters. Supports bulk loading from a file or a .camdl
/// scenario block + inline overrides.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixedParams {
    /// Bulk load from a TOML file (all key=value pairs become fixed).
    /// Inline `values` override file entries on key collision.
    #[serde(default)]
    pub from_file: Option<String>,

    /// Bulk load from a named scenario block declared in the .camdl
    /// model (gh#33). Reads the scenario's `set = { ... }` map and
    /// uses every entry as a fixed value. Mutually exclusive with
    /// `from_file` and inline `values` — by design (see comment on
    /// `expand_scenario` below).
    ///
    /// Resolution requires the loaded model and so happens after
    /// load via `expand_scenario` (called from each fit-pipeline
    /// entry point that has the model in hand). After `expand_scenario`
    /// runs, this field is cleared and the scenario's params land in
    /// `values` for the rest of the pipeline.
    #[serde(default)]
    pub from_scenario: Option<String>,

    /// Inline fixed values. Override `from_file` entries on key
    /// collision. **Mutually exclusive with `from_scenario`** — see
    /// `expand_scenario` for the design rationale.
    #[serde(flatten)]
    pub values: IndexMap<String, f64>,
}

impl FixedParams {
    /// Resolve to a concrete map, with scenario lookup if needed.
    ///
    /// **Design choice — no inline overrides on top of `from_scenario`.**
    /// If both `from_scenario` and inline `values` (or `from_file`) are
    /// set in fit.toml, this errors loudly. Reasoning:
    ///
    /// 1. **Scenario semantics get muddy.** If `[fixed]` can override
    ///    a scenario's params, the fit no longer faithfully represents
    ///    that scenario — it's a hybrid that doesn't correspond to
    ///    anything in the .camdl. Reading the fit.toml in isolation no
    ///    longer tells you what parameters the model actually uses.
    /// 2. **Pressure on naming.** Users who want "baseline with
    ///    low_kappa" are best served by defining a `low_kappa` scenario
    ///    in the .camdl, where the deviation lives next to the canonical
    ///    values. Forcing this surfaces scenario sprawl as a data-
    ///    modeling concern rather than letting it accumulate as
    ///    fit-config drift.
    /// 3. **Cheap to add later** if user demand surfaces, behind a
    ///    loud warning. Until then, the simpler form is one less
    ///    footgun.
    ///
    /// Asymmetry vs `from_file` (which DOES allow inline overrides)
    /// is intentional: `from_file` is just bulk-load convenience for
    /// numbers the user authored; `from_scenario` references a named
    /// abstraction in the .camdl that has its own meaning. Overriding
    /// the named abstraction silently is the whole problem.
    /// Expand `from_scenario` (gh#33) in-place, copying the named
    /// scenario's params into the inline `values` map and clearing
    /// `from_scenario`. Idempotent. After this runs, `resolve()`
    /// returns the right map without needing the model.
    ///
    /// Call this once per fit-pipeline entry point AFTER the model is
    /// loaded but BEFORE `FitConfigV2::validate(&model_params)` (the
    /// every-param-resolved check needs to see the scenario-expanded
    /// values).
    ///
    /// See `resolve_with_model` for the design rationale on
    /// mutual-exclusion of `from_scenario` with `from_file` and
    /// inline `values`.
    /// `estimate` is the fit's `[estimate]` map. gh#37: parameters that
    /// appear in `[estimate]` are carved OUT of the scenario import — a
    /// single `baseline` scenario can serve both forward-sim and the
    /// fit's `[fixed]` source, importing everything EXCEPT the estimated
    /// params. The carve-out is structural (a param can't be both
    /// estimated and fixed; `validate` enforces `estimate ∩ fixed = ∅`),
    /// so removing same-named keys does NOT muddy scenario semantics the
    /// way an inline override with a *different number* would. Inline
    /// overrides still error (preserved).
    pub fn expand_from_scenario(
        &mut self,
        model: &ir::Model,
        estimate: &IndexMap<String, EstimateSpecV2>,
    ) -> Result<(), String> {
        let Some(scen_name) = self.from_scenario.clone() else { return Ok(()); };

        if self.from_file.is_some() {
            return Err(format!(
                "[fixed] from_scenario = \"{}\" and from_file are mutually exclusive. \
                 If you need to override scenario values, define a new scenario in \
                 the .camdl model rather than splitting [fixed] across two sources.",
                scen_name));
        }
        if !self.values.is_empty() {
            let names: Vec<&str> = self.values.keys().map(|s| s.as_str()).collect();
            return Err(format!(
                "[fixed] from_scenario = \"{}\" does not allow inline overrides \
                 (got: {}). Define a new scenario in the .camdl model instead — \
                 fit.toml shouldn't silently mutate scenario semantics.",
                scen_name, names.join(", ")));
        }

        if !model.presets.iter().any(|p| p.name == scen_name) {
            let available: Vec<&str> = model.presets.iter()
                .map(|p| p.name.as_str()).collect();
            return Err(format!(
                "[fixed] from_scenario = \"{}\" not found in model. Available scenarios: {}",
                scen_name,
                if available.is_empty() { "(none declared)".into() }
                else { available.join(", ") }));
        }

        // gh#36: walk `compose = [...]` so the import inherits params from
        // composed sub-scenarios, not just the parent's own `set`. Shared
        // with the simulate path via `resolve_preset_params` — before this
        // the fit path copied only `preset.params`, silently dropping every
        // inherited param and failing with "parameters neither estimated
        // nor fixed".
        let preset_params = crate::params_resolver::resolve_preset_params(model, &scen_name)
            .map_err(|e| e.to_string())?;

        // gh#37 carve-out: import every scenario param EXCEPT the ones
        // being estimated. An estimated param is, by definition, not a
        // fixed param — `validate`'s `estimate ∩ fixed = ∅` check would
        // otherwise reject the import. Applied AFTER the compose-walk so it
        // carves out inherited params too.
        for (k, v) in &preset_params {
            if estimate.contains_key(k) {
                continue;
            }
            self.values.insert(k.clone(), *v);
        }
        self.from_scenario = None;
        Ok(())
    }

    pub fn resolve_with_model(&self, model: &ir::Model) -> Result<IndexMap<String, f64>, String> {
        if let Some(scen_name) = &self.from_scenario {
            if self.from_file.is_some() {
                return Err(format!(
                    "[fixed] from_scenario = \"{}\" and from_file are mutually exclusive. \
                     If you need to override scenario values, define a new scenario in \
                     the .camdl model rather than splitting [fixed] across two sources.",
                    scen_name));
            }
            if !self.values.is_empty() {
                let names: Vec<&str> = self.values.keys().map(|s| s.as_str()).collect();
                return Err(format!(
                    "[fixed] from_scenario = \"{}\" does not allow inline overrides \
                     (got: {}). Define a new scenario in the .camdl model instead — \
                     fit.toml shouldn't silently mutate scenario semantics.",
                    scen_name, names.join(", ")));
            }

            if !model.presets.iter().any(|p| p.name == *scen_name) {
                let available: Vec<&str> = model.presets.iter()
                    .map(|p| p.name.as_str()).collect();
                return Err(format!(
                    "[fixed] from_scenario = \"{}\" not found in model. Available scenarios: {}",
                    scen_name,
                    if available.is_empty() { "(none declared)".into() }
                    else { available.join(", ") }));
            }

            // gh#36: walk compose so the resolved map inherits composed
            // params. Shared with the simulate path via
            // `resolve_preset_params`.
            return crate::params_resolver::resolve_preset_params(model, scen_name)
                .map_err(|e| e.to_string());
        }
        self.resolve()
    }

    /// Resolve to a concrete map: load from_file, then overlay inline values.
    /// Does NOT handle `from_scenario` — call `resolve_with_model` for that.
    /// This method is kept for callers (config_diff, etc.) that don't have
    /// the model loaded; if a fit.toml uses `from_scenario`, those callers
    /// will see an empty map and may produce slightly less informative
    /// output — that's fine for diff/inspection paths, not OK for the
    /// fit pipeline (which uses `resolve_with_model`).
    pub fn resolve(&self) -> Result<IndexMap<String, f64>, String> {
        let mut merged = match &self.from_file {
            Some(path) => {
                let contents = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read fixed params file '{}': {}", path, e))?;
                let table: HashMap<String, toml::Value> = toml::from_str(&contents)
                    .map_err(|e| format!("parse error in '{}': {}", path, e))?;
                let mut map = IndexMap::new();
                for (k, v) in table {
                    match v {
                        toml::Value::Float(f) => { map.insert(k, f); }
                        toml::Value::Integer(i) => { map.insert(k, i as f64); }
                        _ => return Err(format!(
                            "fixed param '{}' in '{}' must be a number, got {:?}",
                            k, path, v
                        )),
                    }
                }
                map
            }
            None => IndexMap::new(),
        };
        // Inline values override file values
        for (k, v) in &self.values {
            merged.insert(k.clone(), *v);
        }
        Ok(merged)
    }
}

// ─── Stages ─────────────────────────────────────────────────────────────────

/// A named inference stage. Tagged by `algorithm`. Each variant carries
/// an explicit `backend` field; the (algorithm, backend) pair is validated
/// against `methods::METHODS` at config-load time. See proposal
/// 2026-05-04-ode-inference-three-phase.md §"Tuple schema" for the
/// rationale (algorithm and backend used to be smuggled together as
/// `method = "if2"` implying chain_binomial).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "algorithm")]
pub enum Stage {
    #[serde(rename = "if2")]
    IF2 {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        particles: usize,
        iterations: usize,
        /// Fraction of initial perturbation magnitude remaining at
        /// `cooling_target_iters` iterations.
        /// Matches pomp's `cooling.fraction.50` semantics:
        /// `cooling = 0.7` and `cooling_target_iters = 50` means
        /// perturbation SD reaches 70% of initial after 50 iterations,
        /// continuing to cool past that.
        cooling: f64,
        /// Iterations over which `cooling` is reached. Default 50 (pomp's
        /// default; not `iterations`). Decoupling target from total length
        /// lets you cool fast then continue at the noise floor.
        #[serde(default = "default_cooling_target_iters")]
        cooling_target_iters: usize,
        /// Toml-side spelling of the "where does this stage's base point
        /// come from?" key. Renamed from the legacy `starts_from` per
        /// proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
        /// The Rust field name stays `starts_from` for now (Step 7 will
        /// rename across the codebase); only the wire key moves.
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        /// How per-chain starting points are drawn. Default `"lhs"`
        /// (Latin-hypercube stratified, scale-aware via Transform).
        /// Other modes: `"single"` (every chain at seeded start),
        /// `"uniform"` (legacy uniform random), `"survey_top_k"` (pull
        /// from top-K rows of a `camdl survey` landscape — requires
        /// `survey_path` + optional `survey_top_k_n` siblings; see
        /// gh#51).
        ///
        /// Toml-side spelling renamed from `init_method` to `init` per
        /// proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema"
        /// (matches the CLI `--init` flag). Rust field name unchanged.
        #[serde(default, rename = "init")]
        init_method: super::init::InitMethod,
        /// Survey CAS directory consumed when `init = "survey_top_k"`.
        /// Must contain `run.json` (kind = survey) and `landscape.tsv`.
        /// Ignored otherwise.
        #[serde(default)]
        survey_path: Option<std::path::PathBuf>,
        /// Number of top-K rows to pull from the survey landscape.
        /// Defaults to `chains` when omitted; in v1 must equal
        /// `chains` (strict K=chains). Ignored when `init_method !=
        /// "survey_top_k"`.
        #[serde(default)]
        survey_top_k_n: Option<usize>,
        /// Clean-evaluation re-scoring of candidate parameter points after
        /// IF2 finishes. See proposal §Proposal 1. Defaults give 4000
        /// particles × 8 replicates combined via logmeanexp.
        #[serde(default)]
        loglik_eval: LoglikEvalConfig,
        /// Compound gate thresholds for chain agreement (Â) and
        /// inter-chain log-likelihood spread (decibans). See proposal
        /// §Proposal 3.
        #[serde(default)]
        gate: GateConfig,
        /// Post-fit Richardson dt-convergence check at θ̂ (gh#52).
        /// Auto-runs at end-of-final-stage; halving-ladder pfilter
        /// eval that warns when the MLE is discretization-dependent.
        /// Defaults to `enabled = true`. Set `enabled = false` on
        /// scout / smoke fits where the check is unnecessary.
        #[serde(default)]
        dt_check: DtCheckConfig,
    },

    #[serde(rename = "pgas")]
    PGAS {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        particles: usize,
        sweeps: usize,
        /// Toml-side spelling renamed from `starts_from` to `init_mle`
        /// per proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        /// Per-chain init draws. See `Stage::IF2` for the full enum
        /// description. Default `lhs`. PGAS also supports `survey_top_k`
        /// (sibling fields `survey_path` / `survey_top_k_n`); survey rows
        /// are usable as MCMC chain seeds because the seed sets only the
        /// chain's starting state, not its stationary distribution
        /// (which the prior governs).
        ///
        /// Toml-side spelling renamed from `init_method` to `init` per
        /// proposal 2026-05-25-cli-init-and-params-ux.
        #[serde(default, rename = "init")]
        init_method: super::init::InitMethod,
        /// Survey CAS directory for `init = "survey_top_k"`.
        /// See `Stage::IF2::survey_path`.
        #[serde(default)]
        survey_path: Option<std::path::PathBuf>,
        /// Top-K count for `init = "survey_top_k"`. See
        /// `Stage::IF2::survey_top_k_n`.
        #[serde(default)]
        survey_top_k_n: Option<usize>,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,

        /// Temperature ladder for parallel tempering (replica
        /// exchange; Earl & Deem 2005, Geyer 1991). Each entry is
        /// β ∈ (0, 1]. The first entry MUST be 1.0 (cold chain).
        /// Only the cold rung contributes posterior samples; heated
        /// rungs explore a flatter likelihood surface (ll × β) and
        /// exchange with adjacent rungs via Metropolis swaps.
        /// Default: `[1.0]` (no tempering, single rung).
        /// Example: `[1.0, 0.7, 0.4, 0.15]`.
        #[serde(default = "default_pgas_tempering")]
        tempering: Vec<f64>,
        /// Maximum NUTS tree depth (Hoffman & Gelman 2014). Stiff
        /// posteriors hit this and need a higher value. Default: 10.
        #[serde(default = "default_max_tree_depth")]
        max_tree_depth: usize,
        /// CSMC-only sweeps before parameter updates begin. The
        /// trajectory is refreshed via CSMC-AS but parameters stay
        /// fixed. Default: 0 (no warm-up).
        #[serde(default)]
        trajectory_warmup: usize,
        /// CSMC trajectory updates per parameter update. Higher
        /// values (3–5) help on long time series where ancestor
        /// sampling is the bottleneck. Default: 1.
        #[serde(default = "default_csmc_sweeps_per_nuts")]
        csmc_sweeps_per_nuts: usize,
        /// Posterior trajectory samples saved to disk (evenly spaced
        /// post-burn-in). Output-side knob, not algorithmic — does
        /// NOT affect the chain hash. Default: 200.
        #[serde(default = "default_n_trajectories")]
        n_trajectories: usize,
        /// NUTS mass matrix shape. `true` = full covariance (handles
        /// parameter correlations like the R0/amplitude ridge),
        /// `false` = diagonal-only (faster but ignores correlations).
        /// Default: true.
        #[serde(default = "default_dense_mass")]
        dense_mass: bool,
        /// Use NUTS (gradient-based) for the θ|X update. `false`
        /// falls back to MH-within-Gibbs. Requires `rate_grad`
        /// expressions in the IR (compiled with autodiff). Default: true.
        #[serde(default = "default_use_nuts")]
        use_nuts: bool,
    },

    #[serde(rename = "pmmh")]
    PMMH {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        particles: usize,
        iterations: usize,
        /// Toml-side spelling renamed from `starts_from` to `init_mle`
        /// per proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        /// Per-chain init draws. See `Stage::IF2` for the full enum
        /// description. Default `lhs`. PMMH also supports `survey_top_k`
        /// (sibling fields `survey_path` / `survey_top_k_n`).
        ///
        /// Toml-side spelling renamed from `init_method` to `init` per
        /// proposal 2026-05-25-cli-init-and-params-ux.
        #[serde(default, rename = "init")]
        init_method: super::init::InitMethod,
        /// Survey CAS directory for `init = "survey_top_k"`.
        /// See `Stage::IF2::survey_path`.
        #[serde(default)]
        survey_path: Option<std::path::PathBuf>,
        /// Top-K count for `init = "survey_top_k"`. See
        /// `Stage::IF2::survey_top_k_n`.
        #[serde(default)]
        survey_top_k_n: Option<usize>,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,

        /// Enable adaptive Metropolis (Haario et al. 2001) — proposal
        /// SDs adapt to past acceptance. Set false to lock the
        /// proposal during a refine run. Default: true.
        #[serde(default = "default_pmmh_adapt")]
        adapt: bool,
        /// MCMC step at which adaptation begins. Earlier values risk
        /// adapting on burn-in noise; later values delay convergence.
        /// Default: 300.
        #[serde(default = "default_pmmh_adapt_start")]
        adapt_start: usize,
        /// Crank-Nicolson correlation for correlated pseudo-marginal
        /// MCMC (Deligiannidis et al. 2018). `None` = vanilla PMMH
        /// with independent PF evaluations. `Some(0.99)` = CPM with
        /// ρ=0.99 (recommended when CPM is enabled). Default: None.
        #[serde(default)]
        rho: Option<f64>,
    },

    /// Metropolis-Hastings on the deterministic ODE marginal likelihood
    /// (`p(y|θ, ODE_skeleton)` via `compute_ode_loglik`). Reuses the PMMH
    /// chain/adaptive-proposal/diagnostics machinery, swapping the
    /// particle-filter likelihood for the deterministic ODE evaluation —
    /// so it carries neither `particles` (no PF) nor `rho` (no correlated
    /// pseudo-marginal noise to re-use). Bayesian posteriors on ODE /
    /// equilibrium models without gradients.
    #[serde(rename = "mh")]
    Mh {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        iterations: usize,
        /// Toml-side spelling renamed from `starts_from` to `init_mle`
        /// per proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        /// Per-chain init draws. See `Stage::IF2` for the full enum
        /// description. Default `lhs`. Mh also supports `survey_top_k`
        /// (sibling fields `survey_path` / `survey_top_k_n`).
        ///
        /// Toml-side spelling renamed from `init_method` to `init` per
        /// proposal 2026-05-25-cli-init-and-params-ux.
        #[serde(default, rename = "init")]
        init_method: super::init::InitMethod,
        /// Survey CAS directory for `init = "survey_top_k"`.
        /// See `Stage::IF2::survey_path`.
        #[serde(default)]
        survey_path: Option<std::path::PathBuf>,
        /// Top-K count for `init = "survey_top_k"`. See
        /// `Stage::IF2::survey_top_k_n`.
        #[serde(default)]
        survey_top_k_n: Option<usize>,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,

        /// Enable adaptive Metropolis (Haario et al. 2001) — proposal
        /// SDs adapt to past acceptance. Set false to lock the
        /// proposal during a refine run. Default: true.
        #[serde(default = "default_pmmh_adapt")]
        adapt: bool,
        /// MCMC step at which adaptation begins. Earlier values risk
        /// adapting on burn-in noise; later values delay convergence.
        /// Default: 300.
        #[serde(default = "default_pmmh_adapt_start")]
        adapt_start: usize,
    },

    #[serde(rename = "pfilter")]
    PFilter {
        backend: crate::run_meta::InferenceBackend,
        particles: usize,
        #[serde(default)]
        replicates: Option<usize>,
        /// Toml-side spelling renamed from `starts_from` to `init_mle`
        /// per proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,

        /// Record per-step ancestor indices for smoothing-path
        /// reconstruction. Off by default (extra memory + copy cost).
        /// See `sim::inference::ancestor_trace`.
        #[serde(default)]
        record_ancestry: bool,
        /// Record per-step predictive samples + log-likelihoods for
        /// `camdl compare`'s prequential scoring (log score, CRPS, PIT).
        /// Roughly N × T f64 per step; cheap relative to the filter
        /// itself. **On by default** — the post-fit PFilter stage is
        /// where prequential is needed and the proposal calls for
        /// it as a first-class output. Set `false` to skip the trace
        /// write (e.g. when running PFilter purely for a loglik SD).
        #[serde(default = "default_record_prequential")]
        record_prequential: bool,
    },

    /// NUTS on the deterministic ODE marginal likelihood (gh#275 Phase 2) — a
    /// gradient-based Bayesian sampler using forward sensitivities (`det_grad`).
    /// Deterministic-likelihood, `ode`-only; on a stochastic backend gradient-NUTS
    /// lives inside `pgas`. Leaner than `PGAS` — no particles, no CSMC, no
    /// tempering; just NUTS warm-up (dual-averaging step size) and sampling.
    #[serde(rename = "nuts")]
    Nuts {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        /// NUTS warm-up (adaptation) iterations — the step size adapts via dual
        /// averaging; these draws are discarded. Default 500.
        #[serde(default = "default_nuts_warmup")]
        warmup: usize,
        /// Posterior draws KEPT per chain (post-warm-up). Default 500.
        #[serde(default = "default_nuts_samples")]
        samples: usize,
        /// Toml-side spelling `init_mle` (see `Stage::PGAS`).
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        /// Per-chain init draws (see `Stage::IF2`). Toml key `init`.
        #[serde(default, rename = "init")]
        init_method: super::init::InitMethod,
        #[serde(default)]
        survey_path: Option<std::path::PathBuf>,
        #[serde(default)]
        survey_top_k_n: Option<usize>,
        /// Maximum NUTS tree depth (Hoffman & Gelman 2014). Default 10.
        #[serde(default = "default_max_tree_depth")]
        max_tree_depth: usize,
        /// Target mean acceptance for dual averaging (Stan default 0.8).
        #[serde(default = "default_target_accept")]
        target_accept: f64,
        /// NUTS mass matrix: `false` = diagonal (Stan's default `diag_e`;
        /// rescales each parameter by its warm-up posterior variance), `true` =
        /// dense (`dense_e`; full covariance, also absorbs parameter
        /// correlations at O(d²) cost and needs more warm-up to estimate). The
        /// warm-up adapts the metric from the sample moments — on an anisotropic
        /// posterior this takes far larger steps at the same acceptance and
        /// frees wide-posterior parameters that identity mass leaves stuck.
        #[serde(default = "default_nuts_dense_mass")]
        dense_mass: bool,
        /// Coarse RK4 step for the *unscored* warm-up `[t_start, first_obs)` on
        /// the ODE gradient path (gh#396 follow-on). `None` (default) or a value
        /// `<= dt` disables it — the whole trajectory integrates at `dt`. A larger
        /// value takes big steps on the transient (state + sensitivity together,
        /// so the NUTS gradient stays consistent with the coarsely-computed value),
        /// cutting the per-gradient cost of a model whose origin is long before the
        /// data. Prevalence (state-scored) streams only in this release; an
        /// incidence stream is refused (its first bin would be coarsened). Identity-
        /// defining: it changes the scored trajectory, so it re-keys the run.
        #[serde(default)]
        burnin_dt: Option<f64>,
    },

    /// NLopt Sbplx (subspace-searching simplex) — deterministic MLE on the
    /// ODE-skeleton likelihood. Default for ODE-backend MLE; robust to
    /// boundary non-smoothness. Phase 1 of the ODE-inference proposal.
    #[serde(rename = "nl-sbplx")]
    NlSbplx(NloptStageConfig),

    /// NLopt BOBYQA — quadratic-trust-region MLE on the ODE-skeleton
    /// likelihood. Faster than Sbplx on smooth interior objectives but
    /// fails at parameter-bound boundaries.
    #[serde(rename = "nl-bobyqa")]
    NlBobyqa(NloptStageConfig),
}

/// Shared config for the two NLopt deterministic MLE stages
/// (`nl-sbplx`, `nl-bobyqa`). Both algorithms read identical knobs;
/// the variant tag picks which NLopt algorithm runs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NloptStageConfig {
    pub backend: crate::run_meta::InferenceBackend,
    /// Number of LHS-spread starting points. Each runs an independent
    /// NLopt optimization to convergence; best-loglik chain wins.
    pub chains: usize,
    /// `xtol_rel` passed to NLopt. Optimizer stops when relative
    /// parameter change between iterations falls below this.
    #[serde(default = "default_nlopt_tolerance")]
    pub tolerance: f64,
    /// Per-chain budget on objective evaluations. Hitting this is a
    /// soft failure (`MaxEvalReached`); successful convergence is
    /// `Success | XtolReached | FtolReached`.
    #[serde(default = "default_nlopt_max_evals")]
    pub max_evals: usize,
    /// Toml-side spelling renamed from `starts_from` to `init_mle`
    /// per proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema".
    #[serde(default, rename = "init_mle")]
    pub starts_from: StartsFrom,
    /// Per-chain init draws. Default `lhs` — Latin-hypercube
    /// stratified sampling, scale-aware via the parameter's
    /// `Transform`. Sbplx/BOBYQA are deterministic optimisers, so
    /// `chains > 1` is only meaningful when chains start from
    /// different points; LHS gives the right coverage for that.
    ///
    /// `init = "single"` defeats multi-start (every chain at
    /// the seeded values converges to the same MLE); use it only for
    /// `chains = 1` runs or when you want pure reproducibility from a
    /// known starting point.
    ///
    /// Caveat for very wide bounds: if `[estimate]` bounds span
    /// regions where transmission collapses (e.g. R0 < 1 in any
    /// setting), some LHS draws may evaluate to
    /// `Poisson(rate=0) | obs > 0 = -inf`. NLopt's xtol-reached
    /// signal can lie there (every neighbouring point also -inf).
    /// For such models, narrow the bounds or pre-validate the LHS
    /// points with a quick `camdl pfilter` loglik check.
    ///
    /// Toml-side spelling renamed from `init_method` to `init` per
    /// proposal 2026-05-25-cli-init-and-params-ux.
    #[serde(default, rename = "init")]
    pub init_method: super::init::InitMethod,
    /// Survey CAS directory for `init = "survey_top_k"` (gh#51).
    /// See `Stage::IF2::survey_path` for the cross-check rules.
    #[serde(default)]
    pub survey_path: Option<std::path::PathBuf>,
    /// Top-K count for `init = "survey_top_k"`. Defaults to
    /// `chains` when omitted.
    #[serde(default)]
    pub survey_top_k_n: Option<usize>,
    /// Convergence-gate thresholds. Two-leg version of IF2's gate
    /// (chain-agreement + decibans-spread); see proposal §"Convergence
    /// diagnostics for NLopt chains".
    #[serde(default)]
    pub gate: GateConfig,
}

fn default_nlopt_tolerance() -> f64 { 1e-6 }
fn default_nlopt_max_evals() -> usize { 5000 }

impl Stage {
    pub fn starts_from(&self) -> &StartsFrom {
        match self {
            Stage::IF2 { starts_from, .. }
            | Stage::PGAS { starts_from, .. }
            | Stage::PMMH { starts_from, .. }
            | Stage::Mh { starts_from, .. }
            | Stage::Nuts { starts_from, .. }
            | Stage::PFilter { starts_from, .. } => starts_from,
            Stage::NlSbplx(c) | Stage::NlBobyqa(c) => &c.starts_from,
        }
    }

    pub fn method_name(&self) -> &'static str {
        self.method_kind().as_str()
    }

    pub fn method_kind(&self) -> crate::run_meta::FitAlgorithm {
        use crate::run_meta::FitAlgorithm;
        match self {
            Stage::IF2      { .. } => FitAlgorithm::If2,
            Stage::PGAS     { .. } => FitAlgorithm::Pgas,
            Stage::PMMH     { .. } => FitAlgorithm::Pmmh,
            Stage::Mh       { .. } => FitAlgorithm::Mh,
            Stage::Nuts     { .. } => FitAlgorithm::Nuts,
            Stage::PFilter  { .. } => FitAlgorithm::Pfilter,
            Stage::NlSbplx  { .. } => FitAlgorithm::NlSbplx,
            Stage::NlBobyqa { .. } => FitAlgorithm::NlBobyqa,
        }
    }

    /// Simulation backend the stage runs on. The (algorithm, backend)
    /// pair is set by the user in fit.toml and validated against
    /// `methods::METHODS`; this accessor returns whichever backend was
    /// declared so dispatch and provenance can branch on it.
    pub fn backend(&self) -> crate::run_meta::InferenceBackend {
        match self {
            Stage::IF2      { backend, .. }
            | Stage::PGAS    { backend, .. }
            | Stage::PMMH    { backend, .. }
            | Stage::Mh      { backend, .. }
            | Stage::Nuts    { backend, .. }
            | Stage::PFilter { backend, .. } => *backend,
            Stage::NlSbplx(c) | Stage::NlBobyqa(c) => c.backend,
        }
    }

    pub fn requires_priors(&self) -> bool {
        matches!(self, Stage::PGAS { .. } | Stage::PMMH { .. } | Stage::Mh { .. } | Stage::Nuts { .. })
    }

    pub fn chains(&self) -> usize {
        match self {
            Stage::IF2 { chains, .. } => *chains,
            Stage::PGAS { chains, .. } => *chains,
            Stage::PMMH { chains, .. } => *chains,
            Stage::Mh { chains, .. } => *chains,
            Stage::Nuts { chains, .. } => *chains,
            Stage::PFilter { .. } => 1,
            Stage::NlSbplx(c) | Stage::NlBobyqa(c) => c.chains,
        }
    }

    /// The per-chain initialisation method (`single` / `lhs` /
    /// `survey_top_k`). NLopt-family and PFilter stages do not carry an
    /// `init_method` field; they report the default (`Lhs`).
    pub fn init_method(&self) -> super::init::InitMethod {
        match self {
            Stage::IF2 { init_method, .. }
            | Stage::PGAS { init_method, .. }
            | Stage::PMMH { init_method, .. }
            | Stage::Mh { init_method, .. }
            | Stage::Nuts { init_method, .. } => init_method.clone(),
            Stage::PFilter { .. } | Stage::NlSbplx(_) | Stage::NlBobyqa(_) => {
                super::init::InitMethod::default()
            }
        }
    }

    /// gh#147 (M3.2). The stage's *extension dimension* (the field
    /// `identity_payload` omits so `--resume` can extend a chain): PGAS
    /// `sweeps`, IF2/PMMH `iterations`. A resumed run is a distinct artifact
    /// keyed on this value, so the CAS stage level folds it in. Single-pass
    /// stages (PFilter) and the NLopt MLE stages (whose `max_evals` is a
    /// budget already in `identity_payload`, not an extension) report 0.
    pub fn cas_target_length(&self) -> u64 {
        match self {
            Stage::IF2 { iterations, .. } => *iterations as u64,
            Stage::PGAS { sweeps, .. } => *sweeps as u64,
            Stage::PMMH { iterations, .. } => *iterations as u64,
            Stage::Mh { iterations, .. } => *iterations as u64,
            Stage::Nuts { samples, .. } => *samples as u64,
            Stage::PFilter { .. } | Stage::NlSbplx(_) | Stage::NlBobyqa(_) => 0,
        }
    }

    /// The number of posterior trajectory samples saved to disk (PGAS only;
    /// default 200). An output-shaping knob that `identity_payload` otherwise
    /// omits, but it is folded into the stage identity (count-in-the-key):
    /// because it changes stored output, changing it yields a distinct leaf
    /// rather than silently reusing the wrong trajectory count, at the cost of
    /// re-fitting when it changes.
    pub fn cas_n_trajectories(&self) -> u64 {
        match self {
            Stage::PGAS { n_trajectories, .. } => *n_trajectories as u64,
            _ => 0,
        }
    }

    /// Hashable subset of the stage that defines its statistical
    /// identity. For PGAS / PMMH this *omits* the extension dimension
    /// (`sweeps` / `iterations` respectively), so `--resume` can extend
    /// a chain by changing only that field without invalidating the
    /// stored `resume_state.bin`. Every other field is identity-
    /// defining: changing chains, particles, burn_in, thin, or
    /// starts_from requires a fresh run.
    ///
    /// IF2 has no extension dimension — its cooling schedule is
    /// determined by the total iteration count, so resuming from the
    /// middle of a different schedule is statistically incoherent.
    /// PFilter is single-pass; nothing to extend.
    ///
    /// Returned as `serde_json::Value` so `provenance::fit_stage_hash`
    /// can hash it via `serde_json::to_vec` (the same canonical form
    /// it used pre-split for the whole stage). Stable across
    /// recompiles because `serde_json` sorts object keys lexically
    /// when serializing maps.
    pub fn identity_payload(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            // PGAS: omit ONLY `sweeps` (extension dimension) and
            // `n_trajectories` (output-only knob; saving more or fewer
            // posterior trajectories doesn't change chain dynamics). Every
            // other field is identity-defining and MUST be hashed —
            // crucially `init_method` / `survey_path` / `survey_top_k_n`,
            // which choose the per-chain starting points and therefore the
            // stored chains/posterior. Dropping them silently served the
            // first run's posterior as a differently-initialised second
            // run's (gh#147 count-in-the-key; survey CONTENT, not just the
            // path, is additionally folded via `ctx.deps` in
            // `resolve_fit_stage`).
            Stage::PGAS {
                backend, chains, particles, starts_from, burn_in, thin,
                tempering, max_tree_depth, trajectory_warmup,
                csmc_sweeps_per_nuts, dense_mass, use_nuts,
                init_method, survey_path, survey_top_k_n,
                ..
            } => json!({
                "algorithm": "pgas",
                "backend": backend,
                "chains": chains,
                "particles": particles,
                "starts_from": starts_from,
                "init_method": init_method,
                "survey_path": survey_path,
                "survey_top_k_n": survey_top_k_n,
                "burn_in": burn_in,
                "thin": thin,
                "tempering": tempering,
                "max_tree_depth": max_tree_depth,
                "trajectory_warmup": trajectory_warmup,
                "csmc_sweeps_per_nuts": csmc_sweeps_per_nuts,
                "dense_mass": dense_mass,
                "use_nuts": use_nuts,
            }),
            // PMMH: omit ONLY `iterations` (extension dimension). All other
            // fields — adapt / adapt_start / rho AND the init selectors
            // (init_method / survey_path / survey_top_k_n) — are
            // identity-defining, for the same reason as PGAS.
            Stage::PMMH {
                backend, chains, particles, starts_from, burn_in, thin,
                adapt, adapt_start, rho,
                init_method, survey_path, survey_top_k_n,
                ..
            } => json!({
                "algorithm": "pmmh",
                "backend": backend,
                "chains": chains,
                "particles": particles,
                "starts_from": starts_from,
                "init_method": init_method,
                "survey_path": survey_path,
                "survey_top_k_n": survey_top_k_n,
                "burn_in": burn_in,
                "thin": thin,
                "adapt": adapt,
                "adapt_start": adapt_start,
                "rho": rho,
            }),
            // Mh (deterministic ODE marginal-likelihood MH): omit ONLY
            // `iterations` (extension dimension). No `particles` / `rho`
            // (deterministic path has neither). All other fields — adapt /
            // adapt_start AND the init selectors — are identity-defining,
            // for the same reason as PMMH.
            Stage::Mh {
                backend, chains, starts_from, burn_in, thin,
                adapt, adapt_start,
                init_method, survey_path, survey_top_k_n,
                ..
            } => json!({
                "algorithm": "mh",
                "backend": backend,
                "chains": chains,
                "starts_from": starts_from,
                "init_method": init_method,
                "survey_path": survey_path,
                "survey_top_k_n": survey_top_k_n,
                "burn_in": burn_in,
                "thin": thin,
                "adapt": adapt,
                "adapt_start": adapt_start,
            }),
            // Nuts (deterministic ODE gradient sampler): omit ONLY `samples`
            // (extension dimension, folded via `cas_target_length` like Mh's
            // `iterations`). Every other field — warmup / max_tree_depth /
            // target_accept / dense_mass / burnin_dt AND the init selectors — is
            // identity-defining, for the same reason as Mh. `burnin_dt` changes the
            // coarsely-integrated warm-up → the scored trajectory → the draws, so it
            // MUST be listed here (leaving it swept into `..` would silently collide
            // two fits that differ only in burnin_dt — the count-in-the-key rule).
            Stage::Nuts {
                backend, chains, warmup, starts_from,
                init_method, survey_path, survey_top_k_n,
                max_tree_depth, target_accept, dense_mass, burnin_dt,
                ..
            } => json!({
                "algorithm": "nuts",
                "backend": backend,
                "chains": chains,
                "warmup": warmup,
                "starts_from": starts_from,
                "init_method": init_method,
                "survey_path": survey_path,
                "survey_top_k_n": survey_top_k_n,
                "max_tree_depth": max_tree_depth,
                "target_accept": target_accept,
                "dense_mass": dense_mass,
                "burnin_dt": burnin_dt,
            }),
            // No extension dimension: hash the full stage. NLopt stages
            // also have no extension dimension — every knob (chains,
            // tolerance, max_evals, init_method, gate) is identity-
            // defining, so hash the full struct.
            Stage::IF2 { .. }
            | Stage::PFilter { .. }
            | Stage::NlSbplx(_)
            | Stage::NlBobyqa(_) =>
                serde_json::to_value(self).unwrap_or(json!({})),
        }
    }

    /// The survey directory feeding `init = "survey_top_k"`, if this stage
    /// uses it. `None` for any other init method — the survey only seeds
    /// chains (and so affects the stored output + identity) under
    /// survey_top_k. The caller folds the survey's CONTENT (its run_id +
    /// landscape digest) into the stage's `deps` so regenerating the survey
    /// re-keys the fit, even at the same path (the path string in
    /// `identity_payload` only catches a *different* directory).
    pub fn survey_init_path(&self) -> Option<&std::path::Path> {
        let (init, path) = match self {
            Stage::IF2 { init_method, survey_path, .. }
            | Stage::PGAS { init_method, survey_path, .. }
            | Stage::PMMH { init_method, survey_path, .. }
            | Stage::Mh { init_method, survey_path, .. }
            | Stage::Nuts { init_method, survey_path, .. } => (init_method, survey_path),
            _ => return None,
        };
        match init {
            super::init::InitMethod::SurveyTopK => path.as_deref(),
            _ => None,
        }
    }
}

// ─── Clean-evaluation + gate (IF2 scout/refine) ─────────────────────────────

/// How to combine M independent particle-filter replicate log-likelihoods
/// into a single score for ranking candidate parameter points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CombineMode {
    /// log( (1/M) Σ exp(ll_k) ) — unbiased on the likelihood scale.
    LogMeanExp,
    /// (1/M) Σ ll_k — biased low, but lower variance.
    Mean,
}

impl Default for CombineMode {
    fn default() -> Self { CombineMode::LogMeanExp }
}

/// Re-evaluate IF2 candidate points (final iter, tail mean, best-in-run)
/// with a high-particle, multi-replicate clean PF before declaring a
/// winner. Closes the ~40-nat extraction bias from argmax over noisy
/// 500-particle in-run evaluations. See proposal §Proposal 1.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoglikEvalConfig {
    /// Particle count per clean PF replicate. Must be ≫ in-run scout
    /// particle count to bring SE under control.
    #[serde(default = "default_loglik_eval_particles")]
    pub n_particles: usize,
    /// Independent PF replicates per candidate. Combined via `combine`.
    #[serde(default = "default_loglik_eval_replicates")]
    pub n_replicates: usize,
    #[serde(default)]
    pub combine: CombineMode,
}

fn default_loglik_eval_particles() -> usize { 4000 }
fn default_loglik_eval_replicates() -> usize { 8 }
/// Pomp's `cooling.fraction.50` default: cooling fraction is reached
/// at iteration 50, then continues at the noise floor.
fn default_cooling_target_iters() -> usize { 50 }

// PGAS defaults
fn default_pgas_tempering() -> Vec<f64> { vec![1.0] }
fn default_max_tree_depth() -> usize { 10 }
fn default_csmc_sweeps_per_nuts() -> usize { 1 }
fn default_n_trajectories() -> usize { 200 }
fn default_dense_mass() -> bool { true }
/// NUTS-on-ODE defaults to a *diagonal* metric (Stan's `diag_e`): cheaper and
/// needs less warm-up than dense, and sufficient for scale-spread anisotropy.
/// Opt into dense (`dense_mass = true`) for a correlated posterior.
fn default_nuts_dense_mass() -> bool { false }
fn default_use_nuts() -> bool { true }
fn default_nuts_warmup() -> usize { 500 }
fn default_nuts_samples() -> usize { 500 }
fn default_target_accept() -> f64 { 0.8 }

// PMMH defaults
fn default_pmmh_adapt() -> bool { true }
fn default_pmmh_adapt_start() -> usize { 300 }

// PFilter defaults
/// Default to recording the prequential trace at the post-fit PFilter
/// stage. Per the 2026-04-20 prequential proposal, every fit pipeline
/// should produce a `PrequentialTrace` as a first-class output —
/// downstream `camdl compare` consumes the per-step log-score / CRPS
/// / PIT samples that this flag toggles. Cost is one extra
/// per-particle obs draw per observation, on the first replicate
/// only; the trace is auto-written to `prequential.{tsv,json}` in
/// the stage dir. Set `record_prequential = false` in `[stages.X]`
/// to opt out (e.g. running PFilter purely for loglik SD without
/// the diagnostic write).
fn default_record_prequential() -> bool { true }

impl Default for LoglikEvalConfig {
    fn default() -> Self {
        Self {
            n_particles: default_loglik_eval_particles(),
            n_replicates: default_loglik_eval_replicates(),
            combine: CombineMode::default(),
        }
    }
}

/// Compound scout-convergence gate: chain agreement (Â) AND inter-chain
/// log-likelihood spread (decibans, with an SE-aware floor). See
/// proposal §Proposal 3.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GateConfig {
    /// Maximum tolerated chain-agreement statistic Â (Gelman–Rubin–style
    /// applied to IF2 chain tails). Pass requires `max(Â) < a_thresh`.
    #[serde(default = "default_a_thresh")]
    pub a_thresh: f64,
    /// Floor on the decibans-spread threshold. The effective threshold
    /// is `max(decibans_thresh, 8 × max(SE) × NATS_TO_DB)` so noisy
    /// chains aren't penalised by Monte-Carlo variance.
    #[serde(default = "default_decibans_thresh")]
    pub decibans_thresh: f64,
}

fn default_a_thresh() -> f64 { 1.01 }
fn default_decibans_thresh() -> f64 { 30.0 }

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            a_thresh: default_a_thresh(),
            decibans_thresh: default_decibans_thresh(),
        }
    }
}

// ─── Richardson dt-convergence check (gh#52) ─────────────────────────

/// Configuration for the post-fit Richardson dt-convergence check.
/// Auto-runs at the end of `camdl fit run`'s final stage (after the
/// compound gate); evaluates `loglik(θ̂; dt)` on a halving ladder
/// `{dt_fit, dt_fit/2, ..., dt_fit/2^n}` and warns when the loglik
/// is still drifting. See `docs/dev/proposals/2026-05-07-richardson-dt-check.md`.
///
/// Defaults are backend-dependent at the *threshold* level (see
/// `effective_threshold_for_backend` in `dt_check.rs`); the struct
/// fields here are pre-resolution.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DtCheckConfig {
    /// Master toggle. Default `true` — every fit gets the audit.
    /// Set `false` to opt out (CI smoke fits, known-converged-dt
    /// rerenders).
    #[serde(default = "default_dt_check_enabled")]
    pub enabled: bool,
    /// Number of halvings beyond `dt_fit`. `n_halvings = 2` evaluates
    /// at `{dt_fit, dt_fit/2, dt_fit/4}` (3 ladder rungs total).
    /// Cost grows like `Σ 2^k = 2^(n+1) − 1` because finer dt has
    /// more sub-steps; default 2 keeps cost ≤ 7× the loglik_eval at
    /// θ̂. `--extended` (n_halvings=3) adds dt_fit/8 for ambiguous
    /// cases.
    #[serde(default = "default_dt_check_halvings")]
    pub n_halvings: usize,
    /// Particle count per ladder-rung evaluation. Default `None` →
    /// inherit from the stage's `loglik_eval.n_particles` (so the
    /// dt-check matches the gate's clean-eval budget).
    #[serde(default)]
    pub n_particles: Option<usize>,
    /// Replicate count per ladder-rung evaluation, combined via
    /// `combine`. Default `None` → inherit from the stage's
    /// `loglik_eval.n_replicates`.
    #[serde(default)]
    pub n_replicates: Option<usize>,
    /// User-set warning threshold floor in nats. The effective
    /// threshold is `max(threshold_nats, 4·σ_max)` so noisy
    /// evaluations don't trip spuriously (mirrors the compound
    /// gate's `8·σ_max·NATS_TO_DB` shape, halved because this is a
    /// per-evaluation comparison rather than a chain-level spread).
    /// Default `None` → backend-specific (2.0 for chain_binomial /
    /// euler_*, 0.5 for ode_rk4).
    #[serde(default)]
    pub threshold_nats: Option<f64>,
    /// Combiner for replicate logliks. Default `None` → inherit
    /// from the stage's `loglik_eval.combine`. Almost always
    /// `LogMeanExp` (unbiased on the likelihood scale).
    #[serde(default)]
    pub combine: Option<CombineMode>,
}

fn default_dt_check_enabled() -> bool { true }
fn default_dt_check_halvings() -> usize { 2 }

impl Default for DtCheckConfig {
    fn default() -> Self {
        Self {
            enabled: default_dt_check_enabled(),
            n_halvings: default_dt_check_halvings(),
            n_particles: None,
            n_replicates: None,
            threshold_nats: None,
            combine: None,
        }
    }
}

/// Where a stage gets its initial parameter values.
/// Deserialized from a string. If the string contains `/` or `\`, it's a
/// directory path; if it equals "random", it's random starts; otherwise
/// it's a stage name reference.
#[derive(Debug, Clone, Default)]
pub enum StartsFrom {
    /// Name of a previous stage in this fit.toml (e.g., "mle").
    Stage(String),
    /// Path to an external results directory.
    Directory(PathBuf),
    /// Random starts from parameter bounds.
    #[default]
    Random,
}

impl serde::Serialize for StartsFrom {
    /// Serializes as a bare string, mirroring the deserializer's
    /// expectations:
    /// - `Stage(name)` → `"name"`
    /// - `Directory(path)` → `"path"` (display form)
    /// - `Random` → `"random"`
    ///
    /// This is the same string form a user would write in fit.toml,
    /// so identity_payload bytes match a hand-written equivalent.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        match self {
            StartsFrom::Stage(s)     => serializer.serialize_str(s),
            StartsFrom::Directory(p) => serializer.serialize_str(&p.to_string_lossy()),
            StartsFrom::Random       => serializer.serialize_str("random"),
        }
    }
}

impl<'de> serde::Deserialize<'de> for StartsFrom {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        // Contains path separator → directory path
        if s.contains('/') || s.contains('\\') {
            Ok(StartsFrom::Directory(PathBuf::from(s)))
        } else if s == "random" {
            Ok(StartsFrom::Random)
        } else {
            // Bare name → stage reference
            Ok(StartsFrom::Stage(s))
        }
    }
}

// ─── Provenance ─────────────────────────────────────────────────────────────

/// Optional metadata linking this fit to a parent.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitProvenance {
    pub derived_from: Option<String>,
    pub reason: Option<String>,
}

// ─── Loading + Validation ───────────────────────────────────────────────────

/// Pre-parse scan for the two legacy fit.toml keys that were renamed in
/// proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema":
///
/// - `[stages.<n>] init_method = "..."` → `[stages.<n>] init = "..."`
/// - `[stages.<n>] starts_from = "..."` → `[stages.<n>] init_mle = "..."`
///
/// After the rename, the old keys would otherwise be silently ignored
/// (serde tolerates unknown fields by default for these structs), which
/// is the silent-wrong-answer failure mode the proposal Step 12 calls
/// out as the migration's primary risk. This detector turns each old
/// key into an actionable load-time error naming the stage, the old
/// key, the replacement, and the proposal that authorises the rename.
///
/// Returns `Ok(())` when no legacy keys are present; `Err(msg)` with
/// the actionable diagnostic when at least one stage carries one.
/// Multiple offending stages are bundled into a single error message
/// rather than failing on the first hit, so the user can fix all of
/// them in one pass.
/// gh#241: `[config].backend` was relocated to `[synthetic].backend` (it only
/// ever fed synthetic-data generation). Catch the old key BEFORE the strict
/// `deny_unknown_fields` parse, so the user gets a migration message naming the
/// replacement instead of a bare "unknown field `backend`" serde error.
fn detect_relocated_config_backend(contents: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // Malformed TOML surfaces from the strict parse with its own message.
        Err(_) => return Ok(()),
    };
    if value.get("config").and_then(|c| c.get("backend")).is_some() {
        return Err(
            "`[config].backend` has moved to `[synthetic].backend` (gh#241).\n  \
             The backend is a forward-simulation setting for synthetic-data \
             generation, not a fit-wide config — fit stages declare their own \
             `backend`. Move it under your `[synthetic]` block, or remove it for \
             a real-data fit.\n  See `camdl docs fit-toml`."
                .into(),
        );
    }
    Ok(())
}

fn detect_legacy_init_keys(contents: &str) -> Result<(), String> {
    // Parse as a generic toml::Value so the walker doesn't depend on the
    // `FitConfigV2` schema (which has already renamed the fields).
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // Don't pre-empt the strongly-typed parser's error reporting on
        // generally-malformed toml; let the typed parse fail downstream
        // with its own message. Pre-scan only catches the rename case.
        Err(_) => return Ok(()),
    };

    let stages = match value.get("stages").and_then(|v| v.as_table()) {
        Some(s) => s,
        None => return Ok(()),
    };

    let mut hits_init_method: Vec<&str> = Vec::new();
    let mut hits_starts_from: Vec<&str> = Vec::new();

    for (stage_name, stage_val) in stages {
        let table = match stage_val.as_table() {
            Some(t) => t,
            None => continue,
        };
        if table.contains_key("init_method") {
            hits_init_method.push(stage_name.as_str());
        }
        if table.contains_key("starts_from") {
            hits_starts_from.push(stage_name.as_str());
        }
    }

    if hits_init_method.is_empty() && hits_starts_from.is_empty() {
        return Ok(());
    }

    let mut msg = String::from(
        "fit.toml uses legacy stage keys removed in CLI UX rev 2 \
         (proposal 2026-05-25-cli-init-and-params-ux §\"fit.toml schema\").\n"
    );
    if !hits_init_method.is_empty() {
        msg.push_str(&format!(
            "\n  error: legacy key `init_method` on stage(s): {}\n  \
             replacement: rename to `init` (matches CLI `--init`).\n  \
             example: `[stages.{}]\\n  init = \"lhs\"` (was: `init_method = \"lhs\"`).\n",
            hits_init_method.iter()
                .map(|s| format!("`{}`", s))
                .collect::<Vec<_>>()
                .join(", "),
            hits_init_method[0],
        ));
    }
    if !hits_starts_from.is_empty() {
        msg.push_str(&format!(
            "\n  error: legacy key `starts_from` on stage(s): {}\n  \
             replacement: rename to `init_mle` (one toml key per concept).\n  \
             example: `[stages.{}]\\n  init_mle = \"<prior-stage>\"` \
             (was: `starts_from = \"<prior-stage>\"`).\n",
            hits_starts_from.iter()
                .map(|s| format!("`{}`", s))
                .collect::<Vec<_>>()
                .join(", "),
            hits_starts_from[0],
        ));
    }
    msg.push_str(
        "\n  See docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md \
         §\"fit.toml schema\" for the full rename table.\n"
    );
    Err(msg)
}

/// gh#241 (C3): reject unknown keys inside a `[stages.*]` block.
///
/// `Stage` is internally tagged (`#[serde(tag = "algorithm")]`), and serde
/// cannot apply `deny_unknown_fields` to such an enum — so a typo on an
/// *optional* stage key (a required-field typo is already caught as "missing
/// field") is silently dropped: neither applied nor reaching the stage identity
/// hash. This post-parse pass compares each stage block's raw keys against the
/// set serde actually recognized — which, because `Stage` carries no
/// `skip_serializing_if`, is exactly the key set the parsed `Stage` serializes
/// back to (renames `init_mle`/`init` and the `algorithm` tag included). Nested
/// sub-tables (`loglik_eval`/`gate`/`dt_check`) are ordinary structs that carry
/// their own `deny_unknown_fields`, so only the top-level stage keys need this.
fn validate_stage_keys(contents: &str, config: &FitConfigV2) -> Result<(), String> {
    let raw: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // A genuine parse error already surfaced from the typed parse upstream.
        Err(_) => return Ok(()),
    };
    let Some(stages) = raw.get("stages").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    for (name, stage) in &config.stages {
        let Some(raw_stage) = stages.get(name).and_then(toml::Value::as_table) else {
            continue;
        };
        let known: BTreeSet<String> = serde_json::to_value(stage)
            .ok()
            .and_then(|v| v.as_object().map(|o| o.keys().cloned().collect()))
            .unwrap_or_default();
        for key in raw_stage.keys() {
            if !known.contains(key) {
                let mut allowed: Vec<&String> = known.iter().collect();
                allowed.sort();
                return Err(format!(
                    "unknown key `{key}` in [stages.{name}] (algorithm = \"{}\").\n  \
                     allowed keys: {}\n  \
                     A typo on an optional stage key is otherwise silently ignored \
                     (serde cannot deny unknown fields on the tagged `Stage` enum) — gh#241.",
                    stage.method_name(),
                    allowed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                ));
            }
        }
    }
    Ok(())
}

impl FitConfigV2 {
    /// Parse a fit.toml string. Performs the Step-12 legacy-key
    /// detection pass (turning the old `init_method` / `starts_from`
    /// keys into actionable errors) before handing the string to the
    /// strongly-typed deserializer.
    pub fn from_toml_str(contents: &str) -> Result<Self, String> {
        detect_legacy_init_keys(contents)?;
        detect_relocated_config_backend(contents)?;
        let config: Self = toml::from_str(contents)
            .map_err(|e| format!("parse error: {}", e))?;
        // gh#241 (C3): catch typo'd stage keys serde silently drops.
        validate_stage_keys(contents, &config)?;
        Ok(config)
    }

    /// Portability lint (gh#307): one warning line per file reference in the
    /// fit config that is written as an ABSOLUTE path. Absolute paths bake one
    /// machine's filesystem layout into the config, breaking sharing and
    /// reproducibility (the content-addressable design) — the fit-config
    /// counterpart of the compiler's W104 on model-file paths. Covered
    /// surfaces: `[model] camdl`, `output_dir`, the wide-TSV `[data] file`, and
    /// every `[data.observations]` stream source.
    ///
    /// Checked on the AS-WRITTEN strings, so it must run BEFORE [`load`]
    /// resolves relative paths against the fit.toml directory (which rewrites
    /// every relative path to an absolute one, erasing the distinction). Pure
    /// and side-effect-free so it is unit-testable; [`load`] prints the returned
    /// lines to stderr.
    pub fn absolute_path_warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut check = |what: &str, path: &str| {
            if std::path::Path::new(path).is_absolute() {
                out.push(format!(
                    "warning: {what} is an absolute path ({path}) — non-portable; \
                     use a path relative to the fit.toml so the fit runs on any machine"
                ));
            }
        };
        check("[model] camdl", &self.model.camdl);
        if let Some(dir) = &self.output_dir {
            check("output_dir", dir);
        }
        if let Some(data) = &self.data {
            if let Some(file) = &data.file {
                check("[data] file", file);
            }
            for (stream, src) in &data.observations {
                check(&format!("[data.observations] {stream}"), src);
            }
        }
        out
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        let mut config = FitConfigV2::from_toml_str(&contents)
            .map_err(|e| format!("error in {}:\n{}", path, e))?;

        // gh#307: warn (do not error) on absolute file references — checked on
        // the as-written paths, before the relative-path resolution below turns
        // every relative path absolute.
        for w in config.absolute_path_warnings() {
            eprintln!("{w}");
        }

        // Resolve toml-relative paths against the toml's directory
        // (Cargo / pyproject convention). Closes GH #22: pre-fix, paths
        // inside the toml were resolved against the user's CWD, which
        // broke any invocation pattern other than "always cd into the
        // toml's directory before camdl fit run". Post-fix, every
        // downstream consumer (the fit-level digest, to_legacy_toml, the
        // runner's data loaders) sees absolute paths regardless of
        // where the binary was invoked from. Absolute paths in the
        // toml pass through unchanged.
        let toml_path = std::path::Path::new(path);
        config.model.camdl = crate::util::resolve_relative_to_toml(
            toml_path, &config.model.camdl);
        if let Some(data) = &mut config.data {
            if let Some(file) = &mut data.file {
                *file = crate::util::resolve_relative_to_toml(toml_path, file);
            }
            for v in data.observations.values_mut() {
                *v = crate::util::resolve_relative_to_toml(toml_path, v);
            }
            if let Some(holdout) = &mut data.holdout {
                for v in holdout.values_mut() {
                    *v = crate::util::resolve_relative_to_toml(toml_path, v);
                }
            }
        }

        Ok(config)
    }

    /// gh#37: expand `[fixed] from_scenario = "name"` in-place, carving
    /// out the parameters that appear in `[estimate]`. Thin wrapper over
    /// `FixedParams::expand_from_scenario` that forwards `&self.estimate`
    /// — the carve-out needs to see which params are being estimated so
    /// a single scenario can serve both forward-sim and the fit's
    /// `[fixed]` source ("import everything EXCEPT the estimated params").
    ///
    /// Call this once per fit-pipeline entry point AFTER the model is
    /// loaded but BEFORE `validate(&model_params)` (the every-param-
    /// resolved check needs to see the scenario-expanded values).
    pub fn expand_fixed_from_scenario(&mut self, model: &ir::Model) -> Result<(), String> {
        self.fixed.expand_from_scenario(model, &self.estimate)
    }

    /// The per-fit subdirectory under the fit segment — always
    /// `real/fit_<seed>/` for real-data fits, and
    /// `synthetic/ds_NN/fit_<seed>/` for synthetic-data fits. The
    /// resulting directory wraps all stage outputs for that fit.
    ///
    /// `dataset_idx` is `None` for real-data fits and `Some(n)` for
    /// synthetic-data fits (1-based dataset index).
    pub fn per_fit_prefix(&self, seed: u64, dataset_idx: Option<usize>) -> PathBuf {
        let source = if self.synthetic.is_some() { "synthetic" } else { "real" };
        let mut p = PathBuf::from(source);
        if let Some(idx) = dataset_idx {
            p = p.join(format_dataset_dir(idx));
        }
        p.join(format!("fit_{}", seed))
    }

    /// Warn on dangling priors: priors declared on estimated parameters
    /// but consumed by no active path in this fit. Returns a
    /// human-readable message, or `None` when every declared prior is
    /// used somewhere (a Bayesian stage, or `fit_starts = "prior"`
    /// initialization).
    ///
    /// IF2 (scout / refine / validate) maximises the likelihood and
    /// ignores priors. A user who declares priors and then runs an
    /// IF2-only pipeline almost certainly didn't mean to: either they
    /// copied a Bayesian `.camdl` example, or they thought IF2 was
    /// Bayesian. Silent-but-wrong is worse than a one-line warning, so
    /// this returns `Some(msg)` that the caller prints to stderr.
    ///
    /// Does NOT error — the staged Bayesian workflow (scout → pgas)
    /// legitimately declares priors in one file and has the IF2 stage
    /// ignore them while the pgas stage consumes them. That case
    /// returns `None` here because the pgas stage *is* a prior
    /// consumer.
    pub fn dangling_priors_warning(&self) -> Option<String> {
        let params_with_priors: Vec<&str> = self.estimate.iter()
            .filter_map(|(name, spec)| spec.prior.as_ref().map(|_| name.as_str()))
            .collect();
        if params_with_priors.is_empty() { return None; }

        let any_bayesian_stage = self.stages.values().any(Stage::requires_priors);
        let starts_from_prior = matches!(self.fit_starts, Some(FitStarts::Prior));
        if any_bayesian_stage || starts_from_prior { return None; }

        Some(format!(
            "priors declared on [{}] but no stage in this fit uses them.\n  \
             IF2 (scout / refine / validate) maximises the likelihood and \
             ignores prior terms.\n  \
             To silence this warning, do one of:\n    \
             - add a Bayesian stage:   [stages.pgas] algorithm = \"pgas\"\n      \
                                        backend = \"chain_binomial\"\n    \
             - use priors for starts:  fit_starts = \"prior\"\n    \
             - remove the priors:      drop `prior = {{...}}` from [estimate.*] entries",
            params_with_priors.join(", ")))
    }

    /// gh#439 A2: does any fit stage read the WrtPop state-Jacobian
    /// (`rate_state_grad` / `projection_state_grad`)? Only `nuts` on the `ode`
    /// backend does — it drives the ODE forward-sensitivity gradient
    /// (`ode_grad::det_grad`). Every other (algorithm, backend) cell — IF2, PGAS,
    /// PMMH, `mh`, the particle filter — is gradient-free with respect to the
    /// state, so the model can compile lean (`camdlc --no-state-grad`), dropping
    /// the dense ~O(G^3) Jacobian that dominates coupled-model IR. Consumed by
    /// `cmd_fit_run_v2` to pick the compile mode; the resulting bit is folded into
    /// the IR-cache key, so a lean entry is never reused for a nuts+ode fit (and
    /// run identity is gradient-independent, so lean vs full hash the same model).
    pub fn needs_state_grad(&self) -> bool {
        use crate::run_meta::{FitAlgorithm, InferenceBackend};
        self.stages.values().any(|s| {
            s.method_kind() == FitAlgorithm::Nuts
                && s.backend() == InferenceBackend::Ode
        })
    }

    /// gh#71: warn when a posterior-sampling stage (PGAS / PMMH) runs
    /// multiple chains from a single shared initialisation.
    ///
    /// `init = "single"` starts every chain at the same point
    /// (`config.estimate[*].initial`). For an MLE method that is merely
    /// wasteful, but for a posterior sampler with `chains > 1` it makes
    /// the between-chain R̂ (Gelman–Rubin) diagnostic uninformative:
    /// chains that begin co-located can't reveal failure to mix from
    /// distinct starting points. A multi-start init (`lhs` /
    /// `survey_top_k`) is needed for R̂ to mean anything.
    ///
    /// Does NOT error — a single-init multi-chain posterior run is
    /// still a valid sample; only the convergence diagnostic is
    /// weakened. Returns `Some(msg)` for the caller to print to stderr.
    pub fn single_init_multichain_warning(&self) -> Option<String> {
        use super::init::InitMethod;
        let offenders: Vec<String> = self.stages.iter()
            .filter(|(_, s)| matches!(s, Stage::PGAS { .. } | Stage::PMMH { .. } | Stage::Mh { .. } | Stage::Nuts { .. }))
            .filter(|(_, s)| s.chains() > 1
                && matches!(s.init_method(), InitMethod::Single))
            .map(|(name, s)| format!("'{}' ({}, chains = {})",
                name, s.method_name(), s.chains()))
            .collect();
        if offenders.is_empty() { return None; }
        Some(format!(
            "posterior-sampling stage(s) {} use init = \"single\" with \
             chains > 1.\n  \
             Every chain then starts at the same point, so the \
             between-chain R̂ (Gelman–Rubin) convergence diagnostic is \
             uninformative — co-located chains cannot reveal a failure \
             to mix.\n  \
             Use a multi-start init for meaningful R̂:\n    \
             - init = \"lhs\"           (Latin-hypercube over bounds)\n    \
             - init = \"survey_top_k\"  (top-K rows of a `camdl survey`)",
            offenders.join(", ")))
    }

    /// Real-data observation paths. Returns an error with a helpful
    /// message when the config is synthetic-only or when neither
    /// source is present (should be caught by `validate()`, but
    /// callers downstream of validation still need a concrete
    /// `DataSpec`).
    pub fn data_spec(&self) -> Result<&DataSpec, String> {
        match (&self.data, &self.synthetic) {
            (Some(d), _)    => Ok(d),
            (None, Some(_)) => Err(
                "this code path requires [data] but the fit config uses [synthetic]. \
                 Synthetic-data fits must be routed through the replicate runner, \
                 which materialises generated datasets before calling the per-fit \
                 path.".to_string()),
            (None, None)    => Err(
                "fit config has neither [data] nor [synthetic] — one must be supplied."
                    .to_string()),
        }
    }

    /// Exhaustive partition check + stage DAG validation + data consistency.
    pub fn validate(&self, model_params: &[String]) -> Result<(), String> {
        // Data source must be exactly one of [data] or [synthetic].
        match (&self.data, &self.synthetic) {
            (Some(_), Some(_)) => return Err(
                "[data] and [synthetic] are mutually exclusive — choose one.\n  \
                 [data] fits against observed data files; [synthetic] generates \
                 datasets from known truth for simulation-based calibration.".to_string()),
            (None, None) => return Err(
                "fit config has neither [data] nor [synthetic] — one must be supplied.".to_string()),
            _ => {}
        }

        // Validate synthetic spec if present.
        if let Some(syn) = &self.synthetic {
            syn.validate()?;
        }

        // Validate [data] block: exactly one of `file` / `observations`.
        if let Some(data) = &self.data {
            data.validate()?;
        }

        // Validate fit_seeds if present (reject duplicates — they would
        // collide on per-cell provenance hashes).
        if let Some(seeds) = &self.fit_seeds {
            if seeds.is_empty() {
                return Err("fit_seeds list is empty — at least one seed required, \
                            or omit the field for single-fit behaviour".to_string());
            }
            let mut seen = BTreeSet::new();
            for &s in seeds {
                if !seen.insert(s) {
                    return Err(format!(
                        "duplicate fit_seed {} — each seed must be unique to avoid \
                         provenance-hash collisions between fits", s));
                }
            }
        }

        // scenario and enable/disable are mutually exclusive (matches simulate).
        if self.scenario.is_some() && (!self.enable.is_empty() || !self.disable.is_empty()) {
            return Err("`scenario` is mutually exclusive with `enable`/`disable`. \
                        Use one approach.".to_string());
        }

        // holdout_after and holdout are mutually exclusive (real-data only;
        // synthetic datasets have no holdout).
        if let Some(data) = &self.data {
            if data.holdout_after.is_some() && data.holdout.is_some() {
                return Err("data.holdout_after and data.holdout are mutually exclusive.\n  \
                            Use holdout_after for temporal splits, holdout for explicit files."
                    .to_string());
            }
        }

        let model_set: BTreeSet<&str> = model_params.iter()
            .map(|s| s.as_str()).collect();
        let estimated: BTreeSet<&str> = self.estimate.keys()
            .map(|s| s.as_str()).collect();

        let fixed_resolved = self.fixed.resolve()?;
        let fixed: BTreeSet<&str> = fixed_resolved.keys()
            .map(|s| s.as_str()).collect();

        // estimate ∩ fixed = ∅
        let overlap: Vec<&&str> = estimated.intersection(&fixed).collect();
        if !overlap.is_empty() {
            return Err(format!(
                "parameters in both [estimate] and [fixed]: {}\n  \
                 Each parameter must be in exactly one section.",
                overlap.iter().map(|s| **s).collect::<Vec<_>>().join(", ")
            ));
        }

        // estimate ∪ fixed = model_params
        let covered: BTreeSet<&str> = estimated.union(&fixed).cloned().collect();
        let missing: Vec<&&str> = model_set.difference(&covered).collect();
        if !missing.is_empty() {
            return Err(format!(
                "parameters neither estimated nor fixed: {}\n  \
                 Every model parameter must appear in [estimate] or [fixed].",
                missing.iter().map(|s| **s).collect::<Vec<_>>().join(", ")
            ));
        }

        let extra: Vec<&&str> = covered.difference(&model_set).collect();
        if !extra.is_empty() {
            return Err(format!(
                "parameters not in model: {}",
                extra.iter().map(|s| **s).collect::<Vec<_>>().join(", ")
            ));
        }

        // (algorithm, backend) must be a supported pair. Method registry
        // is the single source of truth (see fit/methods.rs); errors name
        // the right alternative when the user picked an incoherent combo.
        for (stage_name, stage) in &self.stages {
            // Stage is already typed, so pass the domain values directly — no
            // string round-trip. `validate_combo` is the typed registry gate.
            if let Err(msg) =
                super::methods::validate_combo(stage.method_kind(), stage.backend())
            {
                return Err(format!("stage '{}': {}", stage_name, msg));
            }
        }

        // ic_free / conditioning support gate (F1). `ic_free = true` is
        // honored only by the cells that actually drop y₁ from the
        // accumulated loglik (if2, pfilter, plain pmmh). PGAS, the ODE-MLE
        // optimizers, and correlated PMMH score every obs unconditionally —
        // running ic_free on them would silently compute the UNCONDITIONAL
        // likelihood while the banner claims conditioning. Reject loudly.
        if self.ic_free.unwrap_or(false) {
            for (stage_name, stage) in &self.stages {
                let correlated =
                    matches!(stage, Stage::PMMH { rho: Some(_), .. });
                if let Err(msg) =
                    super::methods::validate_ic_free(stage.method_kind(), correlated)
                {
                    return Err(format!("stage '{}': {}", stage_name, msg));
                }
            }
            // condition_from + ic_free are mutually exclusive. The conditioning
            // warm-up REPLACES the first observation with a reset-only leading
            // hole, leaving ic_free nothing real to condition the initial state
            // on. This must be rejected EXPLICITLY here: the runtime "nothing to
            // condition on" guard fires only when EVERY stream's first cell is a
            // hole (`.all()`), which a PER-STREAM `condition_from` (holing one
            // stream of several) does not satisfy — so relying on that guard lets
            // a multi-stream config slip through and silently condition on the
            // warm-up boundary instead of a real y₁.
            if self.condition_from.is_some() {
                return Err(
                    "condition_from and ic_free cannot be combined: the \
                     conditioning warm-up replaces the first observation with a \
                     reset-only boundary (a leading hole), leaving ic_free nothing \
                     real to condition the initial state on. Use one or the other."
                        .into());
            }
        }

        // IF2 stages require at least one iteration — zero iterations would
        // leave `iterations` empty and cause `last().unwrap()` to panic in
        // `run_if2`. Catch it here so the user gets a config error, not a crash.
        for (stage_name, stage) in &self.stages {
            if let Stage::IF2 { iterations, .. } = stage {
                if *iterations == 0 {
                    return Err(format!(
                        "stage '{}': iterations must be ≥ 1 (got 0). \
                         IF2 needs at least one filtering pass to produce \
                         a parameter estimate.", stage_name));
                }
            }
        }

        // gh#347: a sampler stage retains only its post-burn-in draws. A
        // burn_in ≥ the run length discards EVERY sample, so the fit produces
        // no posterior no matter how well the chain mixes — and the reported
        // post-burn acceptance rate degenerates to 0/0, which reads as a
        // misleading "0% acceptance". Reject at config validation rather than
        // burn compute for an empty result. (The `profile` path already
        // enforces the same steps-vs-burn_in invariant.)
        for (stage_name, stage) in &self.stages {
            let (n_steps, burn_in, len_field, default_burn) = match stage {
                Stage::Mh { iterations, burn_in, .. }
                | Stage::PMMH { iterations, burn_in, .. } => (
                    *iterations,
                    burn_in.unwrap_or(super::pmmh::DEFAULT_BURN_IN),
                    "iterations",
                    super::pmmh::DEFAULT_BURN_IN,
                ),
                Stage::PGAS { sweeps, burn_in, .. } => (
                    *sweeps,
                    burn_in.unwrap_or(super::pgas::DEFAULT_BURN_IN),
                    "sweeps",
                    super::pgas::DEFAULT_BURN_IN,
                ),
                _ => continue,
            };
            if burn_in >= n_steps {
                return Err(format!(
                    "stage '{stage_name}': burn_in ({burn_in}) ≥ {len_field} ({n_steps}) — \
                     every sample is discarded as burn-in, so the fit retains no \
                     posterior draws (and the post-burn acceptance rate degenerates \
                     to 0%). Reduce burn_in or raise {len_field}. \
                     (burn_in defaults to {default_burn} when unset.)"));
            }
        }

        // Bayesian-stage prior presence is checked separately by
        // `validate_priors_present(&ir_priors)`, which needs the model IR
        // in scope to honor the gh#73 precedence fallback. validate()
        // itself only needs parameter names, so the prior check is
        // factored out — production callers do both.

        // Backend validation is now handled at TOML parse time via the
        // typed `Backend` enum (serde rejects unknown strings).

        // Validate stage DAG: starts_from references must be valid
        self.validate_stage_dag()?;

        // Validate bounds. Only check entries that supply explicit
        // fit.toml bounds — entries that omit `bounds = [...]` will
        // resolve to the model's parameters block bounds at
        // build_if2_params_from_specs time, and those have already
        // been validated by the dim-check phase.
        for (name, spec) in &self.estimate {
            if let Some((lo, hi)) = spec.bounds {
                if lo >= hi {
                    return Err(format!(
                        "estimate.{}: bounds [{}, {}] are empty (lo must be < hi)",
                        name, lo, hi
                    ));
                }
            }
        }

        // Validate simplex groups
        self.validate_simplex_groups()?;

        Ok(())
    }

    /// gh#75: Validate that every estimated parameter has a prior available
    /// from at least one source — either this fit toml's
    /// `[estimate.<name>.prior]` block, or the model IR's `~` syntax —
    /// when any stage is Bayesian (PMMH / PGAS).
    ///
    /// This mirrors the gh#73 precedence chain used in `camdl profile`,
    /// extending it to `camdl fit run`. Without the IR fallback, every
    /// fit toml has to reproduce the model's priors verbatim, defeating
    /// the model file as the source of truth.
    ///
    /// Factored out of `validate()` because it needs the model IR in
    /// scope (validate() only needs parameter names). Production callers
    /// invoke both.
    ///
    /// `ir_prior_params` is the set of parameter names that have a
    /// `~` prior declared in the model IR — production callers build it
    /// from `model.parameters.iter().filter_map(|p| p.prior.as_ref().map(|_| p.name.as_str())).collect()`.
    ///
    /// gh#75 — three-tier resolution rule:
    ///
    ///   A parameter's prior is "available" when ANY of:
    ///     (i)   fit toml declares `[estimate.<param>.prior] = { <dist> = ... }`
    ///     (ii)  fit toml declares `[estimate.<param>.prior] = { flat = {} }`
    ///           (explicit opt-in to flat — gh#75)
    ///     (iii) model IR declares a `~ <dist>(...)` prior for the param
    ///           (populated into `ir_prior_params`)
    ///
    /// If none of (i)/(ii)/(iii) holds, the parameter is "missing". The
    /// returned error names every missing parameter and lists all three
    /// remedies so the user can pick whichever fits their workflow.
    ///
    /// The error refuses to start the fit, so downstream consumers of
    /// `fit_summary.json` (which treat the chain as the canonical
    /// posterior) never see a chain that silently targeted the
    /// unconditioned likelihood. Profile's per-cell PMMH still warns
    /// rather than errors on flat fallback because per-cell MLE-as-MAP
    /// is a recoverable case; `fit run` is the authoritative-posterior
    /// surface and the bar is higher.
    pub fn validate_priors_present(
        &self,
        ir_prior_params: &BTreeSet<&str>,
    ) -> Result<(), String> {
        for (stage_name, stage) in &self.stages {
            if stage.requires_priors() {
                // "Missing" = no fit-toml prior of any kind (regular dist
                // *or* explicit flat) AND no IR `~` prior.
                let missing_priors: Vec<&str> = self.estimate.iter()
                    .filter(|(name, spec)| {
                        spec.prior.is_none() && !ir_prior_params.contains(name.as_str())
                    })
                    .map(|(name, _)| name.as_str())
                    .collect();
                if !missing_priors.is_empty() {
                    // Two-column reason table: parameter | why it's missing.
                    // Width derived from the affected set so the output
                    // stays compact when 1–3 params are missing.
                    let name_width = missing_priors.iter()
                        .map(|n| n.len()).max().unwrap_or(0)
                        .max("parameter".len());
                    let mut msg = String::new();
                    msg.push_str(&format!(
                        "stage '{}' (method={}) has parameters with no resolved prior:\n\n",
                        stage_name, stage.method_name(),
                    ));
                    for name in &missing_priors {
                        msg.push_str(&format!(
                            "  {:<width$}   no prior in fit toml, no `~` in model file\n",
                            name, width = name_width,
                        ));
                    }
                    msg.push_str("\nTo proceed, do one of:\n\n");
                    msg.push_str(
                        "  (i)   Declare `prior = { <dist> = { ... } }` in the fit toml's\n        \
                         [estimate.<param>] for each listed parameter.\n");
                    msg.push_str(
                        "  (ii)  Declare a `~ <dist>(...)` prior in the model file for\n        \
                         each listed parameter.\n");
                    msg.push_str(
                        "  (iii) Opt into flat priors explicitly via\n        \
                         `prior = { flat = {} }` in the fit toml — only do this if you\n        \
                         intentionally want the chain to target the unconditioned\n        \
                         likelihood (scaled-likelihood posterior).\n");
                    return Err(msg);
                }
            }
        }
        Ok(())
    }

    /// Validate `[[simplex_groups]]` entries against `[estimate]`.
    /// Rules:
    ///  - `params.len() >= 2` (single-member simplex is degenerate)
    ///  - Every member appears in `[estimate]`
    ///  - No member appears in more than one simplex group
    ///  - No member is `ivp = true` (the simplex transform owns the
    ///    initial perturbation; ivp would conflict)
    ///  - Each member's bounds lower must be ≥ 0 (members are non-negative)
    ///  - (Algorithm-aware) If any non-IF2 stage exists alongside
    ///    simplex groups, emit a warning to stderr — non-IF2 methods
    ///    don't currently honour the constraint.
    fn validate_simplex_groups(&self) -> Result<(), String> {
        if self.simplex_groups.is_empty() {
            return Ok(());
        }

        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (gi, group) in self.simplex_groups.iter().enumerate() {
            if group.params.len() < 2 {
                return Err(format!(
                    "simplex_groups[{}]: must have at least 2 members \
                     (got {}). A 1-member simplex is degenerate (the \
                     constraint forces value = 1).",
                    gi, group.params.len()));
            }
            for name in &group.params {
                let spec = self.estimate.get(name).ok_or_else(|| format!(
                    "simplex_groups[{}]: member '{}' not in [estimate]. \
                     Simplex members must be free parameters.", gi, name))?;
                if !seen.insert(name.as_str()) {
                    return Err(format!(
                        "simplex_groups[{}]: parameter '{}' already \
                         appears in another simplex group. Each parameter \
                         can belong to at most one simplex.", gi, name));
                }
                if spec.ivp {
                    return Err(format!(
                        "simplex_groups[{}]: member '{}' has ivp = true. \
                         The simplex transform owns the initial \
                         perturbation; ivp would conflict. Drop ivp on \
                         simplex members and rely on the simplex's \
                         barycentric perturbation for spread.",
                        gi, name));
                }
                // Skip when fit.toml omits bounds — model bounds get
                // resolved later, and the simplex non-negativity is
                // also enforced at the model level by the dim-check
                // phase. Validating here would force users to mirror
                // bounds in fit.toml just to silence this check.
                let lo = spec.bounds.map(|(lo, _)| lo).unwrap_or(0.0);
                if lo < 0.0 {
                    return Err(format!(
                        "simplex_groups[{}]: member '{}' has bounds \
                         lower {} < 0. Simplex members must be \
                         non-negative.", gi, name, lo));
                }
            }
        }

        // Algorithm-aware warning: non-IF2 stages don't honour simplex.
        let non_if2_stages: Vec<(&str, &str)> = self.stages.iter()
            .filter(|(_, s)| !matches!(s, Stage::IF2 { .. }))
            .map(|(name, s)| (name.as_str(), s.method_name()))
            .collect();
        if !non_if2_stages.is_empty() {
            let names = non_if2_stages.iter()
                .map(|(n, m)| format!("'{}' ({})", n, m))
                .collect::<Vec<_>>().join(", ");
            let use_color = std::io::IsTerminal::is_terminal(&std::io::stderr())
                && std::env::var("NO_COLOR").is_err();
            let tag = if use_color { "\x1b[33mwarning:\x1b[0m" } else { "warning:" };
            eprintln!("{} fit declares simplex_groups, \
                but non-IF2 stage(s) {} do not currently honour the \
                simplex constraint — members will be perturbed \
                independently and rely on the model to enforce sum = 1 \
                indirectly.", tag, names);
        }

        Ok(())
    }

    /// Check that starts_from references point to valid stages or "random".
    fn validate_stage_dag(&self) -> Result<(), String> {
        let stage_names: BTreeSet<&str> = self.stages.keys()
            .map(|s| s.as_str()).collect();

        // Build execution order (declaration order) and check dependencies
        let stage_order: Vec<&str> = self.stages.keys()
            .map(|s| s.as_str()).collect();

        for (i, (name, stage)) in self.stages.iter().enumerate() {
            match stage.starts_from() {
                StartsFrom::Random => continue,
                StartsFrom::Stage(ref dep) => {
                    if !stage_names.contains(dep.as_str()) {
                        return Err(format!(
                            "stage '{}': starts_from = \"{}\" does not match any stage.\n  \
                             Available stages: {}",
                            name, dep, stage_order.join(", ")
                        ));
                    }
                    // Check ordering: dependency must come before this stage
                    let dep_idx = stage_order.iter().position(|s| *s == dep.as_str());
                    if let Some(di) = dep_idx {
                        if di >= i {
                            return Err(format!(
                                "stage '{}': starts_from = \"{}\" but '{}' is declared after '{}'.\n  \
                                 Stages execute in declaration order; dependencies must come first.",
                                name, dep, dep, name
                            ));
                        }
                    }
                }
                StartsFrom::Directory(_) => {
                    // External directory — no DAG check needed
                }
            }
        }
        Ok(())
    }
}

/// Format a dataset index as `ds_01`, `ds_02`, … zero-padded to the
/// minimum width for a 2-digit grid. Grids beyond 99 datasets just
/// stop padding and render as `ds_100`, `ds_101`, etc.
pub(crate) fn format_dataset_dir(idx: usize) -> String {
    format!("ds_{:02}", idx)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn parse(toml_str: &str) -> Result<FitConfigV2, String> {
        // Route every test fixture through the same legacy-key detector
        // the production `load` path uses — so an in-source fixture that
        // accidentally still uses the legacy `init_method` / `starts_from`
        // keys fails loudly here rather than silently parsing as a
        // default-stages config under the new schema. This is the
        // protection that keeps the Step-12 rename from regressing
        // through a stale inline fixture.
        FitConfigV2::from_toml_str(toml_str)
    }

    #[test]
    fn parse_simple_mle() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
rho   = { bounds = [0.001, 1.0] }
k     = { bounds = [0.1, 100.0] }

[fixed]
N0 = 1000000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.estimate.len(), 4);
        assert_eq!(config.fixed.values.len(), 2);
        assert_eq!(config.stages.len(), 1);
        assert!(config.stages.contains_key("mle"));

        match &config.stages["mle"] {
            Stage::IF2 { chains, particles, iterations, cooling, .. } => {
                assert_eq!(*chains, 8);
                assert_eq!(*particles, 1000);
                assert_eq!(*iterations, 80);
                assert!((cooling - 0.70).abs() < 1e-10);
            }
            _ => panic!("expected IF2 stage"),
        }
    }

    // gh#307: absolute-path portability lint over fit-config file references.

    /// A minimal but valid fit config parametrized by the four file-reference
    /// surfaces the lint covers, so a test can flip any of them absolute/relative
    /// without repeating the boilerplate.
    fn cfg_with_paths(camdl: &str, obs: &str, output_dir: &str) -> FitConfigV2 {
        parse(&format!(
            r#"
output_dir = "{output_dir}"

[model]
camdl = "{camdl}"

[data.observations]
weekly_cases = "{obs}"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 10
cooling = 0.7
"#
        ))
        .unwrap()
    }

    #[test]
    fn absolute_path_warnings_flags_all_surfaces() {
        let cfg = cfg_with_paths(
            "/abs/models/sir.camdl",
            "/abs/data/cases.tsv",
            "/abs/out",
        );
        let warnings = cfg.absolute_path_warnings();
        assert_eq!(
            warnings.len(),
            3,
            "one warning each for [model] camdl, [data.observations] weekly_cases, output_dir; got: {warnings:?}"
        );
        // Each warning names its surface and the offending path, and flags it as
        // non-portable (not just "absolute").
        assert!(warnings.iter().any(|w|
            w.contains("[model] camdl") && w.contains("/abs/models/sir.camdl")));
        assert!(warnings.iter().any(|w|
            w.contains("[data.observations] weekly_cases") && w.contains("/abs/data/cases.tsv")));
        assert!(warnings.iter().any(|w|
            w.contains("output_dir") && w.contains("/abs/out")));
        assert!(warnings.iter().all(|w| w.contains("non-portable")));
    }

    #[test]
    fn absolute_path_warnings_silent_on_relative() {
        let cfg = cfg_with_paths(
            "models/sir.camdl",
            "data/cases.tsv",
            "out",
        );
        assert!(
            cfg.absolute_path_warnings().is_empty(),
            "relative paths are portable and must not warn: {:?}",
            cfg.absolute_path_warnings()
        );
    }

    #[test]
    fn absolute_path_warnings_flags_wide_data_file() {
        // The `[data] file = "..."` wide-TSV form is a data source path too.
        let cfg = parse(
            r#"
[model]
camdl = "models/sir.camdl"

[data]
file = "/abs/data/wide.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 10
cooling = 0.7
"#,
        )
        .unwrap();
        let warnings = cfg.absolute_path_warnings();
        assert_eq!(warnings.len(), 1, "got: {warnings:?}");
        assert!(warnings[0].contains("[data] file") && warnings[0].contains("/abs/data/wide.tsv"));
    }

    /// gh#241 C3: serde cannot apply `deny_unknown_fields` to the
    /// internally-tagged `Stage` enum, so a typo'd stage key was silently
    /// dropped (neither applied nor reaching the stage identity hash). A
    /// post-parse pass must reject it with a located error naming the stage
    /// and key; a valid stage (including optional keys) still parses.
    #[test]
    fn stage_rejects_unknown_keys() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\
                    [fixed]\nN0 = 1000000\n";

        let ok = format!(
            "{base}[stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\n"
        );
        assert!(parse(&ok).is_ok(), "a valid IF2 stage must parse");

        // A PGAS optional key (tempering) must be accepted, not falsely flagged.
        let ok_pgas = format!(
            "{base}[stages.post]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\nsweeps = 10\ntempering = [1.0, 0.5]\n"
        );
        assert!(parse(&ok_pgas).is_ok(), "a valid PGAS stage with optional keys must parse");

        // A typo on an OPTIONAL key is the real footgun: every required field
        // is present, so serde parses fine and *silently drops* the typo
        // (using the default), unlike a required-field typo which serde already
        // catches as "missing field". `cooling_target_iters` has a default.
        let bad = format!(
            "{base}[stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\n\
             cooling_target_iterss = 40\n" // typo: cooling_target_iters
        );
        let err = parse(&bad).expect_err("a typo'd optional stage key must be rejected");
        assert!(
            err.contains("cooling_target_iterss") && err.contains("mle"),
            "error must name the unknown key and the stage; got: {err}"
        );
    }

    #[test]
    fn parse_mle_plus_posterior() {
        let config = parse(r#"
[provenance]
derived_from = "fits/01_all_free.toml"
reason = "beta mixing poor in PGAS"

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
gamma = { bounds = [0.05, 1.0], prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
rho   = { bounds = [0.001, 1.0], prior = { beta = { alpha = 2.0, beta = 5.0 } } }
k     = { bounds = [0.1, 100.0], prior = { half_normal = { sigma = 10.0 } } }

[fixed]
beta = 0.34
N0 = 1000000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 2000
iterations = 60
cooling = 0.95
init_mle = "output/fits/01_all_free/mle"

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
init_mle = "mle"

[stages.evaluate]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 10000
replicates = 100
init_mle = "mle"
        "#).unwrap();

        assert_eq!(config.stages.len(), 3);
        let stage_names: Vec<&str> = config.stages.keys().map(|s| s.as_str()).collect();
        assert_eq!(stage_names, vec!["mle", "posterior", "evaluate"]);

        // mle starts from external directory
        match config.stages["mle"].starts_from() {
            StartsFrom::Directory(p) => assert_eq!(p, Path::new("output/fits/01_all_free/mle")),
            other => panic!("expected Directory, got {:?}", other),
        }

        // posterior starts from mle (stage reference)
        match config.stages["posterior"].starts_from() {
            StartsFrom::Stage(s) => assert_eq!(s, "mle"),
            other => panic!("expected Stage, got {:?}", other),
        }

        // All estimated params have priors (needed for PGAS)
        for (_, spec) in &config.estimate {
            assert!(spec.prior.is_some());
        }

        assert!(config.provenance.is_some());
        assert_eq!(config.provenance.as_ref().unwrap().derived_from.as_deref(),
                   Some("fits/01_all_free.toml"));
    }

    #[test]
    fn parse_with_from_file() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 5.0] }

[fixed]
from_file = "params/fixed.toml"
vacc_frac = 0.80

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 100
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.fixed.from_file.as_deref(), Some("params/fixed.toml"));
        assert_eq!(config.fixed.values["vacc_frac"], 0.80);
    }

    #[test]
    fn parse_holdout_after() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data]
holdout_after = 5474.0

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let data = config.data.as_ref().expect("[data] section required in test fixture");
        assert_eq!(data.holdout_after, Some(5474.0));
        assert!(data.holdout.is_none());
    }

    #[test]
    fn validate_complete_partition() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }

[fixed]
N0 = 1000000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        // All params present → OK
        let model_params = vec![
            "beta".to_string(), "gamma".to_string(),
            "N0".to_string(), "I0".to_string(),
        ];
        assert!(config.validate(&model_params).is_ok());
    }

    #[test]
    fn validate_missing_param() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec![
            "beta".to_string(), "gamma".to_string(),
            "N0".to_string(), "I0".to_string(),
        ];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("neither estimated nor fixed"));
        assert!(err.contains("gamma"));
        assert!(err.contains("I0"));
    }

    #[test]
    fn validate_overlap() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
beta = 0.5
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("both [estimate] and [fixed]"));
        assert!(err.contains("beta"));
    }

    #[test]
    fn validate_extra_param() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }
typo_param = { bounds = [0.0, 1.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("not in model"));
        assert!(err.contains("typo_param"));
    }

    #[test]
    fn validate_pgas_requires_priors() {
        // gh#75: prior-presence check is now a separate method
        // `validate_priors_present(&ir_priors)`. validate() no longer
        // looks at priors at all. When called with an empty
        // ir_prior_params set (no IR `~` priors), missing toml priors
        // on Bayesian-stage params still produce the same error.
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        // The partition/dag/etc. check passes — priors are not its concern.
        config.validate(&model_params).expect("validate() should not check priors");
        // The new prior-presence check, with empty IR-priors, still fails.
        // gh#75 reworded the error text to enumerate three remedies
        // (model `~`, fit-toml `prior`, explicit `prior = { flat = {} }`);
        // assert on the stable, structural anchors of the new wording.
        let err = config.validate_priors_present(&BTreeSet::new()).unwrap_err();
        assert!(err.contains("no resolved prior"),
            "error must explain the resolution failure; got:\n{}", err);
        assert!(err.contains("beta"),
            "error must name the offending parameter; got:\n{}", err);
        assert!(err.contains("(i)") && err.contains("(ii)") && err.contains("(iii)"),
            "error must enumerate three remedies (i/ii/iii); got:\n{}", err);
        assert!(err.contains("flat = {}") || err.contains("flat = { }"),
            "error must mention the explicit flat opt-in syntax; got:\n{}", err);
    }

    #[test]
    fn validate_priors_present_passes_when_ir_supplies_prior() {
        // gh#75: the fix. When the toml doesn't declare a prior but the
        // model IR does (via `~` syntax), validate_priors_present must
        // accept it — resolve_prior in fit/runner.rs falls through to
        // the IR prior at fit time.
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let mut ir_priors = BTreeSet::new();
        ir_priors.insert("beta");
        config.validate_priors_present(&ir_priors)
            .expect("IR-declared prior on beta should satisfy the check");
    }

    /// gh#75: an explicit `prior = { flat = {} }` in the fit toml
    /// satisfies the validator without an IR fallback. This is the
    /// "I really do want flat priors" path; silent fallback to flat
    /// is still rejected.
    #[test]
    fn validate_priors_present_passes_with_explicit_flat() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0], prior = { flat = {} } }

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        // No IR priors at all — the only source of beta's prior is the
        // explicit-flat opt-in in the fit toml.
        let ir_priors: BTreeSet<&str> = BTreeSet::new();
        config.validate_priors_present(&ir_priors)
            .expect("explicit prior = { flat = {} } should satisfy validation");

        // And the typed spec correctly identifies the variant.
        let beta_spec = config.estimate.get("beta").expect("beta in estimate");
        let prior = beta_spec.prior.as_ref().expect("prior is set");
        assert!(matches!(prior, EstimatePriorSpec::Flat { .. }),
            "beta's prior should be the explicit-flat variant, got {:?}", prior);
    }

    /// gh#75: parse round-trip — `prior = { flat = {} }` deserializes
    /// to `EstimatePriorSpec::Flat`, and `prior = { log_normal = {...} }`
    /// deserializes to `EstimatePriorSpec::Dist(LogNormal(...))`. The
    /// untagged enum must disambiguate the two wire shapes without a
    /// type hint.
    #[test]
    fn estimate_prior_spec_disambiguates_flat_from_dist() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta  = { bounds = [0.01, 2.0], prior = { flat = {} } }
gamma = { bounds = [0.01, 1.0], prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let beta_prior = config.estimate.get("beta").unwrap().prior.as_ref().unwrap();
        assert!(matches!(beta_prior, EstimatePriorSpec::Flat { .. }),
            "beta with `prior = {{ flat = {{}} }}` should deserialize to Flat, \
             got {:?}", beta_prior);
        let gamma_prior = config.estimate.get("gamma").unwrap().prior.as_ref().unwrap();
        assert!(!matches!(gamma_prior, EstimatePriorSpec::Flat { .. }),
            "gamma with `prior = {{ log_normal = ... }}` should NOT be Flat, \
             got {:?}", gamma_prior);
        // gamma's inner PriorDist is LogNormal.
        match gamma_prior {
            EstimatePriorSpec::Dist(PriorDist::LogNormal(p)) => {
                assert!((p.mu - (-1.2)).abs() < 1e-9);
                assert!((p.sigma - 0.5).abs() < 1e-9);
            }
            other => panic!("expected Dist(LogNormal), got {:?}", other),
        }
    }

    /// gh#155: `prior = { uniform = {} }` (empty) deserializes to
    /// `UniformOverBounds` (uniform over the param's bounds), while
    /// `prior = { uniform = { lower, upper } }` (all fields) stays
    /// `Dist(Uniform)`. The untagged enum disambiguates by field-presence.
    #[test]
    fn estimate_prior_spec_disambiguates_uniform_over_bounds_from_explicit() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
a = { bounds = [0.01, 2.0], prior = { uniform = {} } }
b = { bounds = [0.01, 2.0], prior = { uniform = { lower = 0.1, upper = 0.9 } } }

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 100
        "#).unwrap();

        let a = config.estimate.get("a").unwrap().prior.as_ref().unwrap();
        assert!(matches!(a, EstimatePriorSpec::UniformOverBounds { .. }),
            "`uniform = {{}}` should be UniformOverBounds, got {:?}", a);
        let b = config.estimate.get("b").unwrap().prior.as_ref().unwrap();
        match b {
            EstimatePriorSpec::Dist(PriorDist::Uniform(p)) => {
                assert!((p.lower - 0.1).abs() < 1e-9);
                assert!((p.upper - 0.9).abs() < 1e-9);
            }
            other => panic!("`uniform = {{ lower, upper }}` should be Dist(Uniform), got {:?}", other),
        }
    }

    #[test]
    fn validate_bad_stage_dag() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 2000
iterations = 50
cooling = 0.95
init_mle = "mle"

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("declared after"));
    }

    #[test]
    fn validate_bad_stage_ref() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
init_mle = "nonexistent"
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("does not match any stage"));
    }

    #[test]
    fn validate_empty_bounds() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [2.0, 0.01] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("bounds"));
        assert!(err.contains("empty"));
    }

    #[test]
    fn validate_bad_backend() {
        // A typo'd `[synthetic].backend` (the relocated forward backend,
        // gh#241) is a typed `ForwardBackend`, so an unknown string is
        // rejected at TOML parse time (not at config.validate) — surfacing
        // the error sooner, with a toml/serde location.
        let err = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[synthetic]
true_params = "truth.toml"
sim_seeds = "1:3"
backend = "gilelspie"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).expect_err("typo in backend must reject at parse");
        // Serde reports this as an unknown variant.
        assert!(err.contains("gilelspie") || err.contains("unknown variant"),
            "expected parse error mentioning backend: got {}", err);
    }

    #[test]
    fn validate_simplex_group_rejects_singleton() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y"]
        "#).unwrap();
        let model_params = vec!["S0_y".into(), "beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("at least 2"), "expected size error: {}", err);
    }

    #[test]
    fn validate_simplex_member_must_be_in_estimate() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [0, 1] }
S0_a = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
S0_e = 0.2
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y", "S0_a", "S0_e"]
        "#).unwrap();
        // S0_e is in [fixed], not [estimate] — must reject
        let model_params = vec!["S0_y".into(), "S0_a".into(), "S0_e".into(),
                                "beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("not in [estimate]"), "got: {}", err);
        assert!(err.contains("S0_e"), "got: {}", err);
    }

    #[test]
    fn validate_simplex_member_in_two_groups_rejects() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [0, 1] }
S0_a = { bounds = [0, 1] }
S0_e = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y", "S0_a"]
[[simplex_groups]]
params = ["S0_a", "S0_e"]
        "#).unwrap();
        let model_params = vec!["S0_y".into(), "S0_a".into(), "S0_e".into(),
                                "beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("already appears in another simplex group"),
            "got: {}", err);
    }

    #[test]
    fn validate_simplex_member_with_ivp_rejects() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [0, 1], ivp = true }
S0_a = { bounds = [0, 1] }
S0_e = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y", "S0_a", "S0_e"]
        "#).unwrap();
        let model_params = vec!["S0_y".into(), "S0_a".into(), "S0_e".into(),
                                "beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("ivp = true"), "got: {}", err);
        assert!(err.contains("S0_y"), "got: {}", err);
    }

    #[test]
    fn validate_simplex_member_with_negative_bounds_rejects() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [-0.5, 1] }
S0_a = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
S0_e = 0.2
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y", "S0_a"]
        "#).unwrap();
        let model_params = vec!["S0_y".into(), "S0_a".into(), "S0_e".into(),
                                "beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("non-negative"), "got: {}", err);
    }

    #[test]
    fn validate_simplex_groups_well_formed_passes() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"
[data.observations]
weekly_cases = "data/cases.tsv"
[config]
dt = 1.0
[estimate]
S0_y = { bounds = [0, 1] }
S0_a = { bounds = [0, 1] }
S0_e = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
[[simplex_groups]]
params = ["S0_y", "S0_a", "S0_e"]
        "#).unwrap();
        let model_params = vec!["S0_y".into(), "S0_a".into(), "S0_e".into(),
                                "beta".into(), "N0".into()];
        config.validate(&model_params).expect("well-formed simplex must validate");
    }

    #[test]
    fn validate_data_synthetic_mutex() {
        // Both [data] and [synthetic] supplied — must reject.
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[synthetic]
true_params = "true.toml"
sim_seeds = [1, 2, 3]

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("mutually exclusive"),
            "expected mutex error: got {}", err);
        assert!(err.contains("[data]") && err.contains("[synthetic]"),
            "expected both section names: got {}", err);
    }

    #[test]
    fn data_file_shorthand_parses() {
        // `[data] file = "..."` is the single-file shorthand for stratified
        // models where one wide TSV holds all the columns.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data]
file = "data/typhoid_all.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let data = cfg.data.as_ref().expect("[data] missing");
        assert_eq!(data.file.as_deref(), Some("data/typhoid_all.tsv"));
        assert!(data.observations.is_empty());
    }

    #[test]
    fn data_file_and_observations_are_mutually_exclusive() {
        // Both forms set → DataSpec::validate() rejects.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data]
file = "data/typhoid_all.tsv"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let err = cfg.validate(&["beta".into(), "N0".into()]).unwrap_err();
        assert!(err.contains("mutually exclusive"),
            "error should call out mutual exclusion: {}", err);
        assert!(err.contains("file") && err.contains("observations"),
            "error should name both forms: {}", err);
    }

    #[test]
    fn condition_from_with_ic_free_is_rejected() {
        // condition_from inserts a leading reset-only hole that REPLACES y₁;
        // ic_free needs a real y₁ to condition the initial state on. Setting
        // both must be rejected EXPLICITLY at config-load — not left to the
        // runtime "nothing to condition on" guard, which fires only when EVERY
        // stream's first cell is a hole (`.all()`) and so misses a PER-STREAM
        // `condition_from` that holes only one stream.
        let cfg = parse(r#"
ic_free = true
condition_from = "first_obs - 1 week"

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0], ivp = true }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let err = cfg.validate(&["beta".into(), "N0".into()]).unwrap_err();
        assert!(err.contains("condition_from") && err.contains("ic_free"),
            "error should name both condition_from and ic_free: {}", err);
    }

    #[test]
    fn data_with_neither_file_nor_observations_rejected() {
        // Empty [data] block (no file, no observations) → DataSpec::validate fails.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data]

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let err = cfg.validate(&["beta".into(), "N0".into()]).unwrap_err();
        assert!(err.contains("must specify either"),
            "error should suggest both forms: {}", err);
    }

    #[test]
    fn effective_observations_expands_shorthand() {
        // The shorthand expands to one entry per declared stream in the model,
        // all pointing at the same file.
        let data = DataSpec {
            file: Some("data/x.tsv".into()),
            observations: IndexMap::new(),
            holdout_after: None,
            holdout: None,
        };
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let resolved = data.effective_observations(&names).unwrap();
        assert_eq!(resolved.len(), 3);
        for n in &names {
            assert_eq!(resolved.get(n).map(String::as_str), Some("data/x.tsv"));
        }
    }

    #[test]
    fn effective_observations_passes_through_per_stream_form() {
        let mut obs = IndexMap::new();
        obs.insert("a".to_string(), "data/a.tsv".to_string());
        obs.insert("b".to_string(), "data/b.tsv".to_string());
        let data = DataSpec {
            file: None,
            observations: obs.clone(),
            holdout_after: None,
            holdout: None,
        };
        let resolved = data.effective_observations(&[]).unwrap();
        assert_eq!(resolved, obs);
    }

    // ── gh#33: [fixed] from_scenario shorthand ─────────────────────────

    /// Build a minimal in-memory ir::Model with one scenario for tests.
    fn model_with_scenario(scen: &str, params: &[(&str, f64)]) -> ir::Model {
        use std::collections::HashMap;
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let golden = format!("{}/../../../ir/golden/sir_basic.ir.json", manifest);
        let s = std::fs::read_to_string(&golden).unwrap();
        let mut model: ir::Model = ir::from_str(&s).unwrap();  // gh#audit-C8
        let mut p = HashMap::new();
        for (k, v) in params { p.insert((*k).to_string(), *v); }
        model.presets.push(ir::model::Preset {
            name: scen.to_string(),
            label: format!("test scenario {}", scen),
            params: p,
            enable: vec![],
            disable: vec![],
            scale: HashMap::new(),
            compose: vec![],
            t_end: None,
        });
        model
    }

    #[test]
    fn from_scenario_expands_to_inline_values() {
        // gh#33: `[fixed] from_scenario = "name"` copies the named
        // scenario's `set = { ... }` map into the inline values, so the
        // rest of the pipeline (resolve, validate) sees the same shape
        // it would see for a verbose hand-written [fixed] block.
        let model = model_with_scenario("gh33_only", &[
            ("beta", 0.3), ("gamma", 0.1), ("N0", 1000.0), ("I0", 10.0),
        ]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("gh33_only".into()),
            values: IndexMap::new(),
        };
        fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap();
        assert!(fixed.from_scenario.is_none(), "expansion clears from_scenario");
        let resolved = fixed.resolve().unwrap();
        assert_eq!(resolved.len(), 4);
        assert_eq!(resolved.get("beta"), Some(&0.3));
        assert_eq!(resolved.get("gamma"), Some(&0.1));
    }

    /// Build a model with a compose-based parent scenario `parent` that
    /// inherits from `child` (which carries `child_params`) and layers its
    /// own `parent_params` on top. Mirrors the gh#36 reproducer:
    ///
    /// ```text
    /// scenarios {
    ///   child  { set = { child_params... } }
    ///   parent { compose = [child], set = { parent_params... } }
    /// }
    /// ```
    fn model_with_compose_scenario(
        parent: &str,
        parent_params: &[(&str, f64)],
        child: &str,
        child_params: &[(&str, f64)],
    ) -> ir::Model {
        use std::collections::HashMap;
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let golden = format!("{}/../../../ir/golden/sir_basic.ir.json", manifest);
        let s = std::fs::read_to_string(&golden).unwrap();
        let mut model: ir::Model = ir::from_str(&s).unwrap();
        let mut cp = HashMap::new();
        for (k, v) in child_params { cp.insert((*k).to_string(), *v); }
        model.presets.push(ir::model::Preset {
            name: child.to_string(),
            label: format!("test scenario {}", child),
            params: cp,
            enable: vec![],
            disable: vec![],
            scale: HashMap::new(),
            compose: vec![],
            t_end: None,
        });
        let mut pp = HashMap::new();
        for (k, v) in parent_params { pp.insert((*k).to_string(), *v); }
        model.presets.push(ir::model::Preset {
            name: parent.to_string(),
            label: format!("test scenario {}", parent),
            params: pp,
            enable: vec![],
            disable: vec![],
            scale: HashMap::new(),
            compose: vec![child.to_string()],
            t_end: None,
        });
        model
    }

    #[test]
    fn from_scenario_walks_compose_inherits_params() {
        // gh#36: `[fixed] from_scenario = "parent"` where `parent` is a
        // compose-based scenario must import the params it inherits via
        // `compose = [child]`, not just `parent.set`. Pre-fix the inline
        // walk copied only `parent.params`, so `gamma`/`N0` (which live in
        // `child`) were silently dropped — the fit then errored with
        // "parameters neither estimated nor fixed: N0, gamma".
        let model = model_with_compose_scenario(
            "baseline_compose", &[("beta", 0.3)],
            "fit_pinned", &[("gamma", 0.1), ("N0", 1000.0)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline_compose".into()),
            values: IndexMap::new(),
        };
        fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap();
        assert!(fixed.from_scenario.is_none(), "expansion clears from_scenario");
        let resolved = fixed.resolve().unwrap();
        // Inherited (composed) params must be present, not just parent.set.
        assert_eq!(resolved.get("gamma"), Some(&0.1),
            "composed child param `gamma` must be inherited: {:?}", resolved);
        assert_eq!(resolved.get("N0"), Some(&1000.0),
            "composed child param `N0` must be inherited: {:?}", resolved);
        assert_eq!(resolved.get("beta"), Some(&0.3),
            "parent's own `beta` must be present: {:?}", resolved);
        assert_eq!(resolved.len(), 3,
            "exactly the composed + parent params: {:?}", resolved);
    }

    #[test]
    fn from_scenario_compose_parent_overrides_child_on_collision() {
        // Left-to-right semantics: compose entries apply first, then the
        // parent's own params override on key collision. `child.gamma = 0.1`
        // but `parent.gamma = 0.2` → resolved gamma must be 0.2 (parent wins).
        let model = model_with_compose_scenario(
            "parent", &[("gamma", 0.2)],
            "child", &[("gamma", 0.1), ("N0", 1000.0)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("parent".into()),
            values: IndexMap::new(),
        };
        fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap();
        let resolved = fixed.resolve().unwrap();
        assert_eq!(resolved.get("gamma"), Some(&0.2),
            "parent's own param wins over the composed child's: {:?}", resolved);
        assert_eq!(resolved.get("N0"), Some(&1000.0));
    }

    #[test]
    fn from_scenario_compose_carves_out_estimated_params() {
        // gh#37 carve-out must apply to inherited (composed) params too:
        // an estimated param living in the composed child is carved out of
        // the fixed import, just as it would be for a parent-level param.
        let model = model_with_compose_scenario(
            "baseline_compose", &[("beta", 0.3)],
            "fit_pinned", &[("gamma", 0.1), ("N0", 1000.0)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline_compose".into()),
            values: IndexMap::new(),
        };
        // `gamma` is inherited from the child AND estimated → must be carved.
        fixed.expand_from_scenario(&model, &estimate_set(&["gamma"])).unwrap();
        let resolved = fixed.resolve().unwrap();
        assert!(!resolved.contains_key("gamma"),
            "estimated composed param `gamma` must be carved out: {:?}", resolved);
        assert_eq!(resolved.get("N0"), Some(&1000.0));
        assert_eq!(resolved.get("beta"), Some(&0.3));
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn from_scenario_rejects_nested_compose() {
        // Nested compose is rejected (same as the simulate path). A scenario
        // referenced inside compose = [...] may not itself use compose.
        use std::collections::HashMap;
        let mut model = model_with_compose_scenario(
            "parent", &[("beta", 0.3)],
            "mid", &[("gamma", 0.1)]);
        // Make `mid` itself compose `leaf` → nested.
        model.presets.push(ir::model::Preset {
            name: "leaf".to_string(),
            label: "leaf".to_string(),
            params: HashMap::new(),
            enable: vec![],
            disable: vec![],
            scale: HashMap::new(),
            compose: vec![],
            t_end: None,
        });
        for p in &mut model.presets {
            if p.name == "mid" { p.compose = vec!["leaf".to_string()]; }
        }
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("parent".into()),
            values: IndexMap::new(),
        };
        let err = fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap_err();
        assert!(err.contains("nested compose"),
            "error names the nested-compose rejection: {}", err);
        assert!(err.contains("mid"),
            "error names the offending sub-scenario: {}", err);
    }

    #[test]
    fn from_scenario_idempotent_after_first_call() {
        let model = model_with_scenario("gh33_idem", &[("beta", 0.3)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("gh33_idem".into()),
            values: IndexMap::new(),
        };
        fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap();
        // Second call must be a no-op (from_scenario is already None).
        fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap();
        assert_eq!(fixed.values.len(), 1);
    }

    #[test]
    fn from_scenario_unknown_scenario_errors_with_available_list() {
        let model = model_with_scenario("gh33_present", &[("beta", 0.3)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("gh33_typo".into()),
            values: IndexMap::new(),
        };
        let err = fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap_err();
        assert!(err.contains("gh33_typo"), "error names the bad scenario: {}", err);
        assert!(err.contains("gh33_present"), "error lists what is available: {}", err);
    }

    #[test]
    fn from_scenario_rejects_inline_overrides() {
        // Design choice: no inline overrides on top of from_scenario.
        // Document via test so a future "let's allow it" PR notices the
        // intentional asymmetry vs from_file.
        let model = model_with_scenario("gh33_inline", &[("beta", 0.3)]);
        let mut values = IndexMap::new();
        values.insert("beta".to_string(), 0.5);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("gh33_inline".into()),
            values,
        };
        let err = fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap_err();
        assert!(err.contains("does not allow inline overrides"),
            "error explains the design choice: {}", err);
        assert!(err.contains("beta"),
            "error names the offending key: {}", err);
    }

    #[test]
    fn from_scenario_rejects_alongside_from_file() {
        let model = model_with_scenario("gh33_file", &[("beta", 0.3)]);
        let mut fixed = FixedParams {
            from_file: Some("/some/file.toml".into()),
            from_scenario: Some("gh33_file".into()),
            values: IndexMap::new(),
        };
        let err = fixed.expand_from_scenario(&model, &IndexMap::new()).unwrap_err();
        assert!(err.contains("mutually exclusive") && err.contains("from_file"),
            "error names the conflict: {}", err);
    }

    /// Build a minimal `EstimateSpecV2` (all fields at their declared
    /// defaults) for tests that only care about the *set* of estimated
    /// names, not the search knobs.
    fn estimate_set(names: &[&str]) -> IndexMap<String, EstimateSpecV2> {
        let mut m = IndexMap::new();
        for name in names {
            m.insert((*name).to_string(), EstimateSpecV2 {
                bounds: None,
                transform: None,
                prior: None,
                ivp: false,
                rw_sd: None,
                start: None,
            });
        }
        m
    }

    #[test]
    fn from_scenario_carves_out_estimated_params() {
        // gh#37: a single `baseline` scenario serves both forward-sim and
        // the fit's [fixed] source. `from_scenario = "baseline"` imports
        // everything from baseline EXCEPT the parameters being estimated.
        // Here `beta` is estimated, so the resolved fixed map is the
        // scenario's set MINUS {beta} — no "in both [estimate] and [fixed]"
        // error.
        let model = model_with_scenario("baseline", &[
            ("beta", 0.3), ("gamma", 0.1), ("N0", 1000.0), ("I0", 10.0),
        ]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline".into()),
            values: IndexMap::new(),
        };
        fixed.expand_from_scenario(&model, &estimate_set(&["beta"])).unwrap();
        assert!(fixed.from_scenario.is_none(), "expansion clears from_scenario");
        let resolved = fixed.resolve().unwrap();
        assert!(!resolved.contains_key("beta"),
            "estimated param `beta` must be carved out of the fixed import: {:?}",
            resolved);
        assert_eq!(resolved.get("gamma"), Some(&0.1));
        assert_eq!(resolved.get("N0"), Some(&1000.0));
        assert_eq!(resolved.get("I0"), Some(&10.0));
        assert_eq!(resolved.len(), 3, "exactly the non-estimated scenario params");
    }

    #[test]
    fn config_expand_fixed_from_scenario_carves_out_estimated_params() {
        // gh#37: the FitConfigV2-level wrapper forwards `&self.estimate`
        // so the carve-out can see which params are estimated. End-to-end
        // at the config level: from_scenario="baseline" + [estimate] beta
        // resolves with no estimate∩fixed overlap.
        let model = model_with_scenario("baseline", &[
            ("beta", 0.3), ("gamma", 0.1), ("N0", 1000.0), ("I0", 10.0),
        ]);
        let mut config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
from_scenario = "baseline"

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
        "#).unwrap();
        config.expand_fixed_from_scenario(&model).unwrap();
        assert!(config.fixed.from_scenario.is_none());
        assert!(!config.fixed.values.contains_key("beta"),
            "estimated param carved out: {:?}", config.fixed.values);
        assert_eq!(config.fixed.values.get("gamma"), Some(&0.1));
        assert_eq!(config.fixed.values.len(), 3);
    }

    #[test]
    fn from_scenario_still_rejects_inline_override_with_different_value() {
        // gh#37: the carve-out enables "import minus estimated" but the
        // override case — inline value with a DIFFERENT number for a
        // scenario key — is still a hard error (silent semantic mutation
        // of a named scenario). Here baseline.set.gamma = 0.1 but the
        // fit.toml inlines gamma = 0.5.
        let model = model_with_scenario("baseline", &[
            ("beta", 0.3), ("gamma", 0.1),
        ]);
        let mut values = IndexMap::new();
        values.insert("gamma".to_string(), 0.5);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline".into()),
            values,
        };
        // `gamma` is NOT estimated — it is an inline override that
        // disagrees with the scenario. Must still error.
        let err = fixed.expand_from_scenario(&model, &estimate_set(&["beta"]))
            .unwrap_err();
        assert!(err.contains("does not allow inline overrides"),
            "override-with-different-value still errors: {}", err);
        assert!(err.contains("gamma"),
            "error names the offending key: {}", err);
    }

    #[test]
    fn validate_neither_data_nor_synthetic_rejects() {
        // Both omitted — must reject.
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("neither"),
            "expected 'neither data nor synthetic' error: got {}", err);
    }

    #[test]
    fn validate_scenario_enable_disable_mutex() {
        // scenario + enable list — must reject.
        let config = parse(r#"
scenario = "winter"
enable = ["intervention_a"]

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("mutually exclusive"),
            "expected mutex error: got {}", err);
        assert!(err.contains("scenario"),
            "expected scenario name: got {}", err);
    }

    #[test]
    fn validate_empty_fit_seeds_rejects() {
        let config = parse(r#"
fit_seeds = []

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("empty"),
            "expected empty-list error: got {}", err);
    }

    #[test]
    fn validate_duplicate_fit_seeds_rejects() {
        let config = parse(r#"
fit_seeds = [1, 2, 3, 2]

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("duplicate"),
            "expected duplicate-seed error: got {}", err);
        assert!(err.contains("2"),
            "expected duplicate value in error: got {}", err);
    }

    #[test]
    fn validate_if2_zero_iterations_rejects() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 0
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("iterations must be"),
            "expected iterations error: got {}", err);
        assert!(err.contains("mle"),
            "expected stage name in error: got {}", err);
    }

    #[test]
    fn validate_holdout_mutual_exclusivity() {
        let err = parse(r#"
[model]
camdl = "models/sir.camdl"

[data]
holdout_after = 100.0

[data.observations]
weekly_cases = "data/cases.tsv"

[data.holdout]
weekly_cases = "data/holdout.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err_msg = err.validate(&model_params).unwrap_err();
        assert!(err_msg.contains("mutually exclusive"));
    }

    #[test]
    fn config_optional_defaults() {
        // [config] section omitted entirely — should use defaults
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.config.dt, 1.0);
    }

    #[test]
    fn legacy_config_backend_is_rejected_with_migration_message() {
        // gh#241: `[config].backend` relocated to `[synthetic].backend`. The old
        // key must fail with a migration message naming the replacement, not a
        // bare serde "unknown field" error.
        let err = parse(r#"
[model]
camdl = "m.camdl"
[data.observations]
cases = "d.tsv"
[estimate]
beta = { bounds = [0.01, 2.0] }
[config]
backend = "ode"
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
        "#).unwrap_err();
        assert!(err.contains("[synthetic].backend"), "names the new location: {err}");
        assert!(err.contains("gh#241"), "cites the change: {err}");
        assert!(!err.contains("unknown field"), "not a bare serde error: {err}");
    }

    #[test]
    fn synthetic_backend_parses_and_defaults() {
        use crate::args::types::ForwardBackend;
        // `[synthetic].backend` parses to the typed forward backend (gillespie is
        // valid here — synthetic generation is forward simulation).
        let cfg = parse(r#"
[model]
camdl = "m.camdl"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
gamma = 0.2
[synthetic]
true_params = "truth.toml"
sim_seeds = "1:3"
backend = "gillespie"
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
        "#).unwrap();
        assert_eq!(cfg.synthetic.as_ref().unwrap().backend, ForwardBackend::Gillespie);

        // Omitted → default chain_binomial (matching the old `[config].backend` default).
        let cfg2 = parse(r#"
[model]
camdl = "m.camdl"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
gamma = 0.2
[synthetic]
true_params = "truth.toml"
sim_seeds = "1:3"
[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
        "#).unwrap();
        assert_eq!(cfg2.synthetic.as_ref().unwrap().backend, ForwardBackend::ChainBinomial);
    }

    #[test]
    fn starts_from_directory_detection() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
init_mle = "output/fits/01/mle"
        "#).unwrap();

        match config.stages["mle"].starts_from() {
            StartsFrom::Directory(p) => assert_eq!(p, Path::new("output/fits/01/mle")),
            other => panic!("expected Directory, got {:?}", other),
        }
    }

    #[test]
    fn starts_from_stage_ref() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 2000
iterations = 30
cooling = 0.95
init_mle = "mle"
        "#).unwrap();

        match config.stages["refine"].starts_from() {
            StartsFrom::Stage(s) => assert_eq!(s, "mle"),
            other => panic!("expected Stage, got {:?}", other),
        }
    }

    #[test]
    fn starts_from_default_is_random() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        assert!(matches!(config.stages["mle"].starts_from(), StartsFrom::Random));
    }

    #[test]
    fn fixed_from_file_resolves() {
        // Write a temp params file
        let dir = tempfile::tempdir().unwrap();
        let params_path = dir.path().join("fixed.toml");
        std::fs::write(&params_path, "N0 = 1000000\nI0 = 10\n").unwrap();

        let toml_str = format!(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
from_file = "{}"

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#, params_path.display());

        let config: FitConfigV2 = toml::from_str(&toml_str).unwrap();
        let resolved = config.fixed.resolve().unwrap();
        assert_eq!(resolved["N0"], 1000000.0);
        assert_eq!(resolved["I0"], 10.0);

        // Validate with correct model params
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        assert!(config.validate(&model_params).is_ok());
    }

    #[test]
    fn fixed_from_file_with_inline_override() {
        let dir = tempfile::tempdir().unwrap();
        let params_path = dir.path().join("fixed.toml");
        std::fs::write(&params_path, "N0 = 1000000\nI0 = 10\n").unwrap();

        let toml_str = format!(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
from_file = "{}"
I0 = 50

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#, params_path.display());

        let config: FitConfigV2 = toml::from_str(&toml_str).unwrap();
        let resolved = config.fixed.resolve().unwrap();
        assert_eq!(resolved["N0"], 1000000.0);
        assert_eq!(resolved["I0"], 50.0); // inline overrides from_file
    }

    /// gh#439 A2: `needs_state_grad` is true for exactly the nuts+ode cell — the
    /// only consumer of the WrtPop state-Jacobian — and false for every other
    /// (algorithm, backend) combination, including the near-miss `mh` on `ode`
    /// (Bayesian on the ODE backend, but gradient-free → must compile lean).
    #[test]
    fn needs_state_grad_only_for_nuts_ode() {
        let cfg = |stage: &str| -> FitConfigV2 {
            let toml_str = format!(
                "[model]\ncamdl = \"models/sir.camdl\"\n\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\n\
                 [fixed]\nN0 = 1000\n\n{stage}"
            );
            toml::from_str(&toml_str).unwrap_or_else(|e| panic!("parse {stage:?}: {e}"))
        };

        // nuts + ode → the sole state-Jacobian consumer (true).
        assert!(
            cfg("[stages.post]\nalgorithm = \"nuts\"\nbackend = \"ode\"\nchains = 2")
                .needs_state_grad(),
            "nuts+ode drives the ODE forward-sensitivity gradient — needs the Jacobian"
        );

        // mh + ode → gradient-free Bayesian on the ODE backend → lean (false).
        assert!(
            !cfg("[stages.post]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 100")
                .needs_state_grad(),
            "mh on ode is gradient-free — must compile lean"
        );

        // if2 + chain_binomial → gradient-free MLE → lean (false).
        assert!(
            !cfg("[stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                  chains = 4\nparticles = 100\niterations = 10\ncooling = 0.7")
                .needs_state_grad(),
            "if2+chain_binomial never reads the state-Jacobian"
        );

        // Multi-stage: any nuts+ode stage flips the whole compile to full, even
        // when an earlier gradient-free stage would compile lean on its own.
        let multi: FitConfigV2 = toml::from_str(
            "[model]\ncamdl = \"models/sir.camdl\"\n\n\
             [data.observations]\ncases = \"data/cases.tsv\"\n\n\
             [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\n\
             [fixed]\nN0 = 1000\n\n\
             [stages.scout]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 100\n\n\
             [stages.post]\nalgorithm = \"nuts\"\nbackend = \"ode\"\nchains = 2\n",
        )
        .unwrap();
        assert!(
            multi.needs_state_grad(),
            "a nuts+ode stage anywhere in the arc requires the Jacobian (any-semantics)"
        );
    }

    // ── Synthetic / fit_seeds schema extension ─────────────────────────────

    fn minimal_fit_stages() -> &'static str {
        r#"
[model]
camdl = "models/sir.camdl"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000
I0 = 5
gamma = 0.1

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
"#
    }

    #[test]
    fn synthetic_block_parses() {
        let src = format!(r#"{}
[synthetic]
true_params = "truth.toml"
sim_seeds   = "1:20"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let syn = config.synthetic.as_ref().expect("[synthetic] missing");
        assert_eq!(syn.true_params, "truth.toml");
        assert_eq!(syn.datasets.unwrap_or_else(|| syn.sim_seeds.to_vec().unwrap().len()), 20);
        assert!(syn.scenario.is_none());
    }

    #[test]
    fn synthetic_datasets_inferred_from_sim_seeds() {
        let src = format!(r#"{}
[synthetic]
true_params = "truth.toml"
sim_seeds   = [7, 42, 101]
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let syn = config.synthetic.unwrap();
        assert!(syn.datasets.is_none(), "datasets should be inferred, not set");
        assert_eq!(syn.sim_seeds.to_vec().unwrap().len(), 3);
        syn.validate().expect("inferred count must validate");
    }

    #[test]
    fn synthetic_datasets_explicit_must_match() {
        let src = format!(r#"{}
[synthetic]
true_params = "truth.toml"
datasets    = 20
sim_seeds   = "1:5"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let err = config.synthetic.unwrap().validate().unwrap_err();
        assert!(err.contains("20") && err.contains("5"),
            "error must name both counts: {}", err);
    }

    #[test]
    fn data_and_synthetic_mutually_exclusive() {
        let src = format!(r#"{}
[data.observations]
cases = "data/cases.tsv"

[synthetic]
true_params = "truth.toml"
sim_seeds   = "1:5"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()])
            .unwrap_err();
        assert!(err.contains("[data]") && err.contains("[synthetic]"),
            "error must name both blocks: {}", err);
    }

    #[test]
    fn neither_data_nor_synthetic_errors() {
        let src = minimal_fit_stages().to_string();
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()])
            .unwrap_err();
        assert!(err.contains("[data]") && err.contains("[synthetic]"),
            "error must mention both options: {}", err);
    }

    #[test]
    fn seeds_range_parses() {
        let s = SeedsSpec::Range("1:5".into());
        assert_eq!(s.to_vec().unwrap(), vec![1u64, 2, 3, 4, 5]);
        s.validate_no_duplicates().unwrap();
    }

    #[test]
    fn seeds_inverted_range_errors() {
        let s = SeedsSpec::Range("10:5".into());
        let err = s.to_vec().unwrap_err();
        assert!(err.contains("malformed") || err.contains("start ≤ end"),
            "inverted range must surface a clear error: {}", err);
        let err = s.validate_no_duplicates().unwrap_err();
        assert!(err.contains("malformed") || err.contains("start ≤ end"),
            "validate_no_duplicates must propagate parse error: {}", err);
    }

    #[test]
    fn seeds_malformed_range_errors() {
        let s = SeedsSpec::Range("not-a-range".into());
        let err = s.to_vec().unwrap_err();
        assert!(err.contains("malformed"),
            "malformed range must surface a clear error: {}", err);
    }

    #[test]
    fn seeds_list_duplicates_rejected() {
        let s = SeedsSpec::List(vec![1, 2, 2, 3]);
        let err = s.validate_no_duplicates().unwrap_err();
        assert!(err.contains("duplicate"), "must name duplicate: {}", err);
    }

    #[test]
    fn fit_seeds_list_parses() {
        // Top-level keys like `fit_seeds` must precede any [table] header
        // in TOML, otherwise the key is consumed by the previous table.
        let single_src = format!(r#"fit_seeds = [42]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_stages());
        let config = parse(&single_src).unwrap();
        assert_eq!(config.fit_seeds.unwrap(), vec![42u64]);

        let list_src = format!(r#"fit_seeds = [101, 102, 103]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_stages());
        let config = parse(&list_src).unwrap();
        assert_eq!(config.fit_seeds.unwrap(), vec![101u64, 102, 103]);
    }

    // ── Dangling-priors warning ────────────────────────────────────────────

    fn fit_with_priors_if2_only() -> &'static str {
        r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta  = { bounds = [0.01, 2.0], prior = { log_normal = { mu = -0.3, sigma = 0.5 } } }
gamma = { bounds = [0.05, 1.0], prior = { half_normal = { sigma = 1.0 } } }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 50
cooling = 0.7

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.9
init_mle = "scout"
"#
    }

    #[test]
    fn dangling_priors_warns_on_if2_only() {
        let config = parse(fit_with_priors_if2_only()).unwrap();
        let msg = config.dangling_priors_warning()
            .expect("IF2-only config with priors must warn");
        assert!(msg.contains("beta") && msg.contains("gamma"),
            "warning must name every param whose prior is dangling: {}", msg);
        assert!(msg.contains("IF2") && msg.contains("maximises the likelihood"),
            "warning must explain why priors are unused: {}", msg);
        // Actionable suggestions present.
        assert!(msg.contains("pgas") && msg.contains("fit_starts"),
            "warning must list the fixes: {}", msg);
    }

    #[test]
    fn dangling_priors_silent_when_pgas_stage_present() {
        // Add a PGAS stage to the same config — now priors are live.
        let mut src = fit_with_priors_if2_only().to_string();
        src.push_str(r#"
[stages.pgas]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 1000
sweeps = 1000
init_mle = "refine"
"#);
        let config = parse(&src).unwrap();
        assert!(config.dangling_priors_warning().is_none(),
            "pgas consumes the declared priors — no warning expected");
    }

    #[test]
    fn dangling_priors_silent_when_fit_starts_is_prior() {
        let mut src = fit_with_priors_if2_only().to_string();
        // Prepend fit_starts at the top (TOML: top-level keys must
        // precede the first [table]).
        src = format!("fit_starts = \"prior\"\n{}", src);
        let config = parse(&src).unwrap();
        assert!(config.dangling_priors_warning().is_none(),
            "fit_starts = \"prior\" uses priors for init — no warning expected");
    }

    #[test]
    fn dangling_priors_silent_when_no_priors_declared() {
        // No [estimate.*].prior at all — nothing to be dangling.
        let src = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
gamma = 0.3
N0 = 1000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 500
iterations = 50
cooling = 0.7
"#;
        let config = parse(src).unwrap();
        assert!(config.dangling_priors_warning().is_none(),
            "no priors declared — nothing to warn about");
    }

    // ── gh#71: single-init multi-chain posterior R̂ warning ─────────────────

    /// A PGAS (posterior-sampling) fit with a tunable `init` / `chains`
    /// on the single Bayesian stage. `{init}` / `{chains}` are filled in
    /// per-test so the trigger and its controls share one fixture.
    fn fit_pgas_with_init(init: &str, chains: usize) -> String {
        format!(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta  = {{ bounds = [0.01, 2.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }} }}
gamma = {{ bounds = [0.05, 1.0], prior = {{ half_normal = {{ sigma = 1.0 }} }} }}

[fixed]
N0 = 1000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
init = "{init}"
chains = {chains}
particles = 500
sweeps = 1000
"#, init = init, chains = chains)
    }

    #[test]
    fn single_init_multichain_warns_for_pgas() {
        let config = parse(&fit_pgas_with_init("single", 4)).unwrap();
        let msg = config.single_init_multichain_warning()
            .expect("PGAS with init=single and chains>1 must warn");
        assert!(msg.contains("posterior") && msg.contains("'posterior'"),
            "warning must name the offending stage: {}", msg);
        assert!(msg.contains("pgas") && msg.contains("chains = 4"),
            "warning must report the method and chain count: {}", msg);
        assert!(msg.contains("R\u{0302}") || msg.contains("Gelman"),
            "warning must explain the R-hat consequence: {}", msg);
        assert!(msg.contains("lhs") && msg.contains("survey_top_k"),
            "warning must suggest a multi-start init as the fix: {}", msg);
    }

    #[test]
    fn single_init_multichain_silent_for_single_chain() {
        // Control: init=single but chains=1 — R̂ is not even defined for
        // one chain, so a shared init is harmless. No warning.
        let config = parse(&fit_pgas_with_init("single", 1)).unwrap();
        assert!(config.single_init_multichain_warning().is_none(),
            "chains=1 has no between-chain R̂ to weaken — no warning expected");
    }

    #[test]
    fn single_init_multichain_silent_for_multistart_init() {
        // Control: chains>1 but a multi-start init (lhs) — chains begin
        // dispersed, so R̂ is informative. No warning.
        let config = parse(&fit_pgas_with_init("lhs", 4)).unwrap();
        assert!(config.single_init_multichain_warning().is_none(),
            "lhs disperses chain starts — no R̂ warning expected");
    }

    #[test]
    fn single_init_multichain_silent_for_if2() {
        // Control: IF2 (an MLE method, not a posterior sampler) with
        // init=single and chains>1 — IF2 reports no R̂, so the warning
        // must NOT fire for it.
        let src = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
gamma = 0.3
N0 = 1000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
init = "single"
chains = 4
particles = 500
iterations = 50
cooling = 0.7
"#;
        let config = parse(src).unwrap();
        assert!(config.single_init_multichain_warning().is_none(),
            "IF2 is not a posterior sampler — single-init R̂ warning must not fire");
    }

    #[test]
    fn fit_seeds_duplicates_rejected_during_validate() {
        let src = format!(r#"fit_seeds = [1, 2, 1]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()])
            .unwrap_err();
        assert!(err.contains("duplicate"), "must reject duplicate fit seeds: {}", err);
    }

    // ── ic_free / conditioning support gate (F1) ───────────────────────────
    //
    // `ic_free = true` is honored only by IF2, the bootstrap PF, and plain
    // (uncorrelated) PMMH. PGAS, the ODE-MLE optimizers, and correlated PMMH
    // score every obs unconditionally — running ic_free on them silently
    // computes the unconditional likelihood. validate() must hard-error those
    // cells; the honoring cells must still pass.

    /// Model params for the ic_free fixtures (sir with beta/gamma/N0/I0).
    fn ic_free_model_params() -> Vec<String> {
        vec!["beta".into(), "gamma".into(), "N0".into(), "I0".into()]
    }

    #[test]
    fn ic_free_with_if2_stage_still_validates() {
        // Regression: IF2 honors conditioning — ic_free=true must NOT be
        // rejected by the gate (it would break ic_free_true_with_ivp_succeeds).
        let src = format!(
            "ic_free = true\n{}\n[data.observations]\ncases = \"data/cases.tsv\"\n",
            minimal_fit_stages()
        );
        let config = parse(&src).unwrap();
        config
            .validate(&ic_free_model_params())
            .expect("ic_free=true on an IF2 stage must validate (IF2 honors conditioning)");
    }

    #[test]
    fn ic_free_with_pgas_stage_is_rejected() {
        let src = r#"ic_free = true
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000
I0 = 5
gamma = 0.1

[stages.bayes]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 500
sweeps = 100
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params())
            .expect_err("ic_free=true on a PGAS stage must be rejected (PGAS ignores conditioning)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("pgas"), "error must name the offending stage's algorithm: {err}");
    }

    // gh#347: a sampler stage whose burn_in ≥ the run length retains ZERO
    // posterior draws (every sample discarded), and the post-burn acceptance
    // rate degenerates to 0/0 = a misleading "0%". Reject at config validation.
    #[test]
    fn burn_in_exceeding_iterations_is_rejected() {
        let src = r#"[model]
camdl = "models/sir.camdl"
[data.observations]
cases = "data/cases.tsv"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000
I0 = 5
gamma = 0.1
[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
burn_in = 3000
"#;
        let config = parse(src).unwrap();
        let err = config.validate(&ic_free_model_params())
            .expect_err("burn_in ≥ iterations must be rejected: no retained samples");
        assert!(err.contains("burn_in"), "error must name burn_in: {err}");
        assert!(err.contains("iterations"), "error must name iterations: {err}");
    }

    /// The exact gh#347 repro config: `iterations` below the *default* burn_in
    /// (5000) must be rejected too — the default must not silently discard all.
    #[test]
    fn default_burn_in_exceeding_iterations_is_rejected() {
        let src = r#"[model]
camdl = "models/sir.camdl"
[data.observations]
cases = "data/cases.tsv"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000
I0 = 5
gamma = 0.1
[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
"#;
        let config = parse(src).unwrap();
        let err = config.validate(&ic_free_model_params())
            .expect_err("default burn_in (5000) ≥ iterations (2000) must be rejected");
        assert!(err.contains("5000"), "error should surface the default burn_in: {err}");
    }

    #[test]
    fn burn_in_below_iterations_is_accepted() {
        let src = r#"[model]
camdl = "models/sir.camdl"
[data.observations]
cases = "data/cases.tsv"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000
I0 = 5
gamma = 0.1
[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
burn_in = 500
"#;
        let config = parse(src).unwrap();
        config.validate(&ic_free_model_params())
            .expect("burn_in < iterations must validate");
    }

    #[test]
    fn ic_free_with_ode_mle_stage_is_rejected() {
        let src = r#"ic_free = true
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000
I0 = 5
gamma = 0.1

[stages.mle]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params())
            .expect_err("ic_free=true on an ODE-MLE stage must be rejected (compute_ode_loglik ignores conditioning)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("nl-sbplx"), "error must name the offending algorithm: {err}");
    }

    #[test]
    fn ic_free_with_correlated_pmmh_stage_is_rejected() {
        let src = r#"ic_free = true
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000
I0 = 5
gamma = 0.1

[stages.bayes]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 500
iterations = 100
rho = 0.99
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params())
            .expect_err("ic_free=true on a correlated PMMH stage must be rejected");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
    }

    // ── per_fit_prefix layout ──────────────────────────────────────────────

    fn mini_real() -> FitConfigV2 {
        toml::from_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.7
"#).unwrap()
    }

    #[test]
    fn real_fit_prefix_is_real_fit_seed() {
        let cfg = mini_real();
        assert_eq!(cfg.per_fit_prefix(42, None),
                   std::path::PathBuf::from("real").join("fit_42"));
    }

    #[test]
    fn synthetic_fit_prefix_is_synthetic_ds_fit_seed() {
        let mut cfg = mini_real();
        cfg.data = None;
        cfg.synthetic = Some(SyntheticSpec {
            true_params: "truth.toml".into(),
            sim_seeds: SeedsSpec::Range("1:3".into()),
            datasets: None,
            scenario: None,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
        });
        assert_eq!(cfg.per_fit_prefix(101, Some(2)),
                   std::path::PathBuf::from("synthetic").join("ds_02").join("fit_101"));
    }

    #[test]
    fn pfilter_record_prequential_defaults_to_true() {
        // Per the 2026-04-20 prequential proposal, the post-fit
        // PFilter stage should record a PrequentialTrace by default —
        // omitting the field in TOML must produce `true`, not `false`.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.evaluate]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 1000
        "#).unwrap();

        match &cfg.stages["evaluate"] {
            Stage::PFilter { record_prequential, record_ancestry, .. } => {
                assert!(*record_prequential,
                    "record_prequential must default to true");
                assert!(!*record_ancestry,
                    "record_ancestry stays opt-in (false default)");
            }
            _ => panic!("expected PFilter stage"),
        }
    }

    #[test]
    fn pfilter_record_prequential_can_be_disabled() {
        // Explicit `record_prequential = false` opts out — used when
        // running PFilter purely for a loglik SD without the trace
        // write.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.evaluate]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 1000
record_prequential = false
        "#).unwrap();

        match &cfg.stages["evaluate"] {
            Stage::PFilter { record_prequential, .. } =>
                assert!(!*record_prequential,
                    "explicit record_prequential = false must override the default"),
            _ => panic!("expected PFilter stage"),
        }
    }

    #[test]
    fn if2_stage_loglik_eval_and_gate_default_when_omitted() {
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        match &cfg.stages["scout"] {
            Stage::IF2 { loglik_eval, gate, .. } => {
                assert_eq!(loglik_eval.n_particles, 4000);
                assert_eq!(loglik_eval.n_replicates, 8);
                assert_eq!(loglik_eval.combine, CombineMode::LogMeanExp);
                assert!((gate.a_thresh - 1.01).abs() < 1e-12);
                assert!((gate.decibans_thresh - 30.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }
    }

    #[test]
    fn if2_stage_loglik_eval_and_gate_parse_overrides() {
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
loglik_eval = { n_particles = 8000, n_replicates = 16, combine = "mean" }
gate = { a_thresh = 1.05, decibans_thresh = 60.0 }

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 60
cooling = 0.95

[stages.refine.loglik_eval]
n_particles = 12000

[stages.refine.gate]
decibans_thresh = 100.0
        "#).unwrap();

        match &cfg.stages["scout"] {
            Stage::IF2 { loglik_eval, gate, .. } => {
                assert_eq!(loglik_eval.n_particles, 8000);
                assert_eq!(loglik_eval.n_replicates, 16);
                assert_eq!(loglik_eval.combine, CombineMode::Mean);
                assert!((gate.a_thresh - 1.05).abs() < 1e-12);
                assert!((gate.decibans_thresh - 60.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }

        // refine: partial overrides — unset fields take defaults
        match &cfg.stages["refine"] {
            Stage::IF2 { loglik_eval, gate, .. } => {
                assert_eq!(loglik_eval.n_particles, 12000);
                assert_eq!(loglik_eval.n_replicates, 8);            // default
                assert_eq!(loglik_eval.combine, CombineMode::LogMeanExp); // default
                assert!((gate.a_thresh - 1.01).abs() < 1e-12);     // default
                assert!((gate.decibans_thresh - 100.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }
    }

    #[test]
    fn dataset_dir_is_zero_padded() {
        assert_eq!(format_dataset_dir(1),   "ds_01");
        assert_eq!(format_dataset_dir(9),   "ds_09");
        assert_eq!(format_dataset_dir(10),  "ds_10");
        assert_eq!(format_dataset_dir(100), "ds_100");
    }

    /// Default-equipped PGAS stage for identity tests. Builder pattern
    /// keeps the test fixtures terse as Stage::PGAS grows fields.
    fn make_pgas_stage(sweeps: usize) -> Stage {
        Stage::PGAS {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, sweeps,
            starts_from: StartsFrom::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            burn_in: Some(200), thin: Some(2),
            tempering: vec![1.0],
            max_tree_depth: 10,
            trajectory_warmup: 0,
            csmc_sweeps_per_nuts: 1,
            n_trajectories: 200,
            dense_mass: true,
            use_nuts: true,
        }
    }

    /// Default-equipped PMMH stage for identity tests.
    fn make_pmmh_stage(iterations: usize) -> Stage {
        Stage::PMMH {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations,
            starts_from: StartsFrom::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            burn_in: Some(200), thin: Some(2),
            adapt: true, adapt_start: 300, rho: None,
        }
    }

    #[test]
    fn pgas_identity_payload_omits_sweeps() {
        // Two PGAS stages identical except for `sweeps` must produce
        // the same identity_payload — that's the contract that lets
        // --resume extend a chain by changing the iteration count.
        let s_short = make_pgas_stage(1000);
        let s_long = make_pgas_stage(5000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());

        // Changing any *other* PGAS field must change the payload.
        let s_more_chains = match make_pgas_stage(1000) {
            Stage::PGAS { backend, particles, sweeps, starts_from, init_method,
                survey_path, survey_top_k_n,
                burn_in, thin,
                tempering, max_tree_depth, trajectory_warmup, csmc_sweeps_per_nuts,
                n_trajectories, dense_mass, use_nuts, .. } =>
                Stage::PGAS { backend, chains: 8, particles, sweeps, starts_from, init_method,
                    survey_path, survey_top_k_n,
                    burn_in, thin,
                    tempering, max_tree_depth, trajectory_warmup, csmc_sweeps_per_nuts,
                    n_trajectories, dense_mass, use_nuts },
            _ => unreachable!(),
        };
        assert_ne!(s_short.identity_payload(), s_more_chains.identity_payload());
    }

    #[test]
    fn pgas_identity_payload_omits_n_trajectories() {
        // n_trajectories is an output-side knob (how many posterior
        // samples to save). It MUST NOT be in identity — saving more
        // or fewer samples doesn't change chain dynamics, so resume
        // should accept a different n_trajectories without
        // re-running.
        let mut s_few = make_pgas_stage(1000);
        let mut s_many = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut n_trajectories, .. } = s_few { *n_trajectories = 100; }
        if let Stage::PGAS { ref mut n_trajectories, .. } = s_many { *n_trajectories = 1000; }
        assert_eq!(s_few.identity_payload(), s_many.identity_payload(),
            "n_trajectories is output-only and must not affect identity");
    }

    #[test]
    fn pgas_identity_payload_includes_new_algorithmic_knobs() {
        // tempering, max_tree_depth, trajectory_warmup,
        // csmc_sweeps_per_nuts, dense_mass, use_nuts ALL change chain
        // dynamics and MUST invalidate identity.
        let base = make_pgas_stage(1000);

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut tempering, .. } = s {
            *tempering = vec![1.0, 0.5];
        }
        assert_ne!(base.identity_payload(), s.identity_payload(), "tempering");

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut max_tree_depth, .. } = s { *max_tree_depth = 14; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "max_tree_depth");

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut trajectory_warmup, .. } = s {
            *trajectory_warmup = 100;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(), "trajectory_warmup");

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut csmc_sweeps_per_nuts, .. } = s {
            *csmc_sweeps_per_nuts = 3;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(),
            "csmc_sweeps_per_nuts");

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut dense_mass, .. } = s { *dense_mass = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "dense_mass");

        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut use_nuts, .. } = s { *use_nuts = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "use_nuts");
    }

    #[test]
    fn pgas_identity_payload_includes_init_and_survey() {
        // `init_method` chooses the per-chain starting points (lhs / single /
        // survey_top_k / from_*), which determine the stored chains/posterior.
        // Two PGAS fits differing ONLY in init must NOT collide — otherwise the
        // first run's posterior is silently served as the second's (a wrong
        // scientific result on a multimodal problem). gh#147 count-in-the-key.
        use crate::fit::init::InitMethod;
        let base = make_pgas_stage(1000); // init_method = lhs (default)

        let mut s_single = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, .. } = s_single {
            *init_method = InitMethod::Single;
        }
        assert_ne!(base.identity_payload(), s_single.identity_payload(),
            "init_method lhs vs single must change the identity");

        let mut s_survey = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, .. } = s_survey {
            *init_method = InitMethod::SurveyTopK;
        }
        assert_ne!(base.identity_payload(), s_survey.identity_payload(),
            "init_method lhs vs survey_top_k must change the identity");

        // survey_top_k_n: how many top-K rows seed the chains → distinct starts.
        let mut s_k = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, ref mut survey_top_k_n, .. } = s_k {
            *init_method = InitMethod::SurveyTopK;
            *survey_top_k_n = Some(8);
        }
        assert_ne!(s_survey.identity_payload(), s_k.identity_payload(),
            "survey_top_k_n must change the identity");

        // A different --survey directory feeds different starting points.
        let mut s_a = make_pgas_stage(1000);
        let mut s_b = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, ref mut survey_path, .. } = s_a {
            *init_method = InitMethod::SurveyTopK;
            *survey_path = Some("/tmp/survey_a".into());
        }
        if let Stage::PGAS { ref mut init_method, ref mut survey_path, .. } = s_b {
            *init_method = InitMethod::SurveyTopK;
            *survey_path = Some("/tmp/survey_b".into());
        }
        assert_ne!(s_a.identity_payload(), s_b.identity_payload(),
            "different --survey dir must change the identity");
    }

    #[test]
    fn survey_init_path_only_under_survey_top_k() {
        use crate::fit::init::InitMethod;
        // Default init (lhs) → no survey dep, even if a stray survey_path is set.
        let s = make_pgas_stage(1000);
        assert!(s.survey_init_path().is_none(), "lhs init → no survey dep");

        // survey_top_k + survey_path → that path is surfaced for the dep fold.
        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, ref mut survey_path, .. } = s {
            *init_method = InitMethod::SurveyTopK;
            *survey_path = Some("/tmp/survey_x".into());
        }
        assert_eq!(s.survey_init_path(), Some(std::path::Path::new("/tmp/survey_x")));

        // survey_top_k but no survey_path → None (init will error; nothing to fold).
        let mut s = make_pgas_stage(1000);
        if let Stage::PGAS { ref mut init_method, .. } = s {
            *init_method = InitMethod::SurveyTopK;
        }
        assert!(s.survey_init_path().is_none(), "survey_top_k w/o path → None");
    }

    #[test]
    fn pmmh_identity_payload_includes_init_and_survey() {
        use crate::fit::init::InitMethod;
        let base = make_pmmh_stage(1000);

        let mut s_single = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut init_method, .. } = s_single {
            *init_method = InitMethod::Single;
        }
        assert_ne!(base.identity_payload(), s_single.identity_payload(),
            "PMMH init_method lhs vs single must change the identity");

        let mut s_survey = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut init_method, .. } = s_survey {
            *init_method = InitMethod::SurveyTopK;
        }
        let mut s_k = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut init_method, ref mut survey_top_k_n, .. } = s_k {
            *init_method = InitMethod::SurveyTopK;
            *survey_top_k_n = Some(8);
        }
        assert_ne!(s_survey.identity_payload(), s_k.identity_payload(),
            "PMMH survey_top_k_n must change the identity");
    }

    #[test]
    fn pmmh_identity_payload_omits_iterations() {
        let s_short = make_pmmh_stage(1000);
        let s_long = make_pmmh_stage(8000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());
    }

    #[test]
    fn identity_payload_is_byte_stable_against_recompiles() {
        // Golden bytes for a fixed PGAS stage. Locks the
        // serialization order so a recompile that silently changes
        // serde_json's key ordering would invalidate every
        // resume_state.bin in the wild — we'd rather fail this test
        // than have users discover the breakage later.
        //
        // serde_json::to_vec on serde_json::json!{} preserves the
        // declaration order of keys in the Value tree (BTreeMap-
        // like behavior is opt-in via `preserve_order` feature, off
        // by default; default Map sorts lexically). Either way the
        // result is deterministic, so a golden constant catches drift.
        let stage = make_pgas_stage(1000);
        let payload_bytes = serde_json::to_vec(&stage.identity_payload()).unwrap();
        let payload_str = String::from_utf8(payload_bytes).unwrap();
        let expected = r#"{"algorithm":"pgas","backend":"chain_binomial","burn_in":200,"chains":4,"csmc_sweeps_per_nuts":1,"dense_mass":true,"init_method":"uniform_unconstrained","max_tree_depth":10,"particles":100,"starts_from":"random","survey_path":null,"survey_top_k_n":null,"tempering":[1.0],"thin":2,"trajectory_warmup":0,"use_nuts":true}"#;
        assert_eq!(payload_str, expected,
            "identity_payload byte format drifted — every existing \
             resume_state.bin would be invalidated. If this change is \
             intentional, update the golden constant AND ship a note \
             to users that --resume against pre-change chains will \
             reject.");
    }

    #[test]
    fn pmmh_identity_payload_byte_stable() {
        let stage = make_pmmh_stage(1000);
        let payload_str = serde_json::to_string(&stage.identity_payload()).unwrap();
        let expected = r#"{"adapt":true,"adapt_start":300,"algorithm":"pmmh","backend":"chain_binomial","burn_in":200,"chains":4,"init_method":"uniform_unconstrained","particles":100,"rho":null,"starts_from":"random","survey_path":null,"survey_top_k_n":null,"thin":2}"#;
        assert_eq!(payload_str, expected,
            "PMMH identity_payload byte format drifted — see \
             pgas_identity_payload_byte_stable for context.");
    }

    #[test]
    fn pmmh_identity_payload_includes_new_algorithmic_knobs() {
        let base = make_pmmh_stage(1000);

        let mut s = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut adapt, .. } = s { *adapt = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt");

        let mut s = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut adapt_start, .. } = s { *adapt_start = 1000; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt_start");

        let mut s = make_pmmh_stage(1000);
        if let Stage::PMMH { ref mut rho, .. } = s { *rho = Some(0.99); }
        assert_ne!(base.identity_payload(), s.identity_payload(), "rho");
    }

    #[test]
    fn if2_identity_payload_includes_iterations_and_cooling() {
        // IF2 has no extension dimension — its cooling schedule is
        // determined by the total iteration count, so changing
        // iterations *must* invalidate identity (and thus reject
        // resume). This guards against a future refactor accidentally
        // moving `iterations` out of identity.
        let s50 = Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.95,
            cooling_target_iters: 50,
            starts_from: StartsFrom::default(),
            loglik_eval: LoglikEvalConfig::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        };
        let s100 = Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 100, cooling: 0.95,
            cooling_target_iters: 50,
            starts_from: StartsFrom::default(),
            loglik_eval: LoglikEvalConfig::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        };
        assert_ne!(s50.identity_payload(), s100.identity_payload());

        let s_diff_cooling = Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.70,
            cooling_target_iters: 50,
            starts_from: StartsFrom::default(),
            loglik_eval: LoglikEvalConfig::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        };
        assert_ne!(s50.identity_payload(), s_diff_cooling.identity_payload());

        // cooling_target_iters is identity-defining (different schedule
        // → different chain dynamics).
        let s_diff_target = Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.95,
            cooling_target_iters: 100,
            starts_from: StartsFrom::default(),
            loglik_eval: LoglikEvalConfig::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        };
        assert_ne!(s50.identity_payload(), s_diff_target.identity_payload());
    }

    // ── bounds-optional in [estimate.X] ──────────────────────────────

    #[test]
    fn estimate_bounds_optional_serde_default() {
        // gh#NN-followup: bounds was previously a required (f64, f64);
        // omitting it produced a parse error. Now it's Option<(f64, f64)>
        // with #[serde(default)], so an [estimate.X] block with no
        // explicit bounds should deserialize cleanly to None and let
        // build_if2_params_from_specs fall back to the model file.
        let toml_str = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate.beta]
# bounds intentionally omitted — should resolve from model file

[fixed]
N0 = 1000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config: FitConfigV2 = toml::from_str(toml_str)
            .expect("bounds must be optional");
        assert!(config.estimate.contains_key("beta"));
        assert_eq!(config.estimate["beta"].bounds, None,
            "omitted bounds must deserialize to None, not a default tuple");
    }

    #[test]
    fn estimate_bounds_explicit_still_parses() {
        // Backwards compat: existing fit.toml files that DO supply
        // bounds = [lo, hi] continue to parse and the value lands as
        // Some((lo, hi)) — the gh#42-followup `tighten-but-not-loosen`
        // logic in build_if2_params_from_specs reads this Some.
        let toml_str = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate.beta]
bounds = [0.01, 2.0]

[fixed]
N0 = 1000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config: FitConfigV2 = toml::from_str(toml_str).unwrap();
        assert_eq!(config.estimate["beta"].bounds, Some((0.01, 2.0)));
    }

    #[test]
    fn validate_bounds_skips_none_entries() {
        // bounds = None must not trigger the lo < hi validation;
        // model-file bounds are validated upstream (dim check).
        let toml_str = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate.beta]
# bounds omitted

[fixed]
N0 = 1000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config: FitConfigV2 = toml::from_str(toml_str).unwrap();
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        config.validate(&model_params).expect("validation must pass with omitted bounds");
    }

    #[test]
    fn validate_bounds_still_rejects_inverted_explicit_bounds() {
        // Even with bounds optional, when the user DOES supply bounds
        // and they're inverted (lo >= hi), the validator must still
        // refuse. Regression guard for the Option-aware check.
        let toml_str = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate.beta]
bounds = [2.0, 0.01]

[fixed]
N0 = 1000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config: FitConfigV2 = toml::from_str(toml_str).unwrap();
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        let err = config.validate(&model_params)
            .expect_err("inverted explicit bounds must error");
        assert!(err.contains("are empty") || err.contains("lo must be < hi"),
            "error must name the lo/hi violation; got: {err}");
    }

    // ─── Step 12: legacy-key rejection (CLI UX rev 2) ───────────────────────
    //
    // The TOML keys `init_method` and `starts_from` were renamed to `init`
    // and `init_mle` respectively (proposal 2026-05-25-cli-init-and-params-ux,
    // §"fit.toml schema"). The new spelling matches the CLI flag names
    // (`--init`); the old spelling now produces an actionable load-time
    // error pointing at the rename.

    #[test]
    fn parse_with_renamed_init_key() {
        // `init = "lhs"` (the new toml spelling) must deserialize identically
        // to the legacy `init_method = "lhs"`.
        let config = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7
init = "lhs"
        "#).expect("init = \"lhs\" (renamed key) must parse cleanly");

        match &config.stages["mle"] {
            Stage::IF2 { init_method, .. } => {
                assert_eq!(init_method.clone(), crate::fit::init::InitMethod::Lhs);
            }
            _ => panic!("expected IF2 stage"),
        }
    }

    #[test]
    fn parse_with_renamed_init_mle_key_stage_ref() {
        // `init_mle = "<stage>"` (the new toml spelling) must deserialize
        // identically to the legacy `starts_from = "<stage>"` and produce
        // a StartsFrom::Stage reference.
        let config = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 1000
iterations = 20
cooling = 0.95
init_mle = "mle"
        "#).expect("init_mle = \"mle\" (renamed key) must parse cleanly");

        match config.stages["refine"].starts_from() {
            StartsFrom::Stage(s) => assert_eq!(s, "mle"),
            other => panic!("expected Stage(\"mle\"), got {:?}", other),
        }
    }

    #[test]
    fn parse_with_renamed_init_mle_key_directory() {
        // `init_mle = "<dir/path>"` (containing path separators) must
        // deserialize to StartsFrom::Directory, matching the legacy
        // `starts_from`'s dispatch on path separators.
        let config = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7
init_mle = "output/fits/01/mle"
        "#).expect("init_mle = \"<dir>\" must parse cleanly");

        match config.stages["mle"].starts_from() {
            StartsFrom::Directory(p) => assert_eq!(p, Path::new("output/fits/01/mle")),
            other => panic!("expected Directory, got {:?}", other),
        }
    }

    #[test]
    fn legacy_init_method_key_rejected_with_actionable_error() {
        // The legacy spelling `init_method = "lhs"` must fail loading with
        // an error that names the rename, gives the replacement spelling,
        // and cites the proposal so the user can self-serve.
        let err = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7
init_method = "lhs"
        "#).expect_err("legacy init_method key must produce a load error");

        assert!(err.contains("init_method"),
            "error must name the legacy key; got: {err}");
        assert!(err.contains("init"),
            "error must point at the replacement key `init`; got: {err}");
        assert!(err.contains("stages.mle") || err.contains("stage `mle`")
                || err.contains("stage 'mle'"),
            "error must locate the offending stage by name; got: {err}");
        assert!(err.contains("2026-05-25-cli-init-and-params-ux"),
            "error must cite the proposal for context; got: {err}");
    }

    #[test]
    fn legacy_starts_from_key_rejected_with_actionable_error() {
        // The legacy spelling `starts_from = "<stage>"` must fail loading
        // with an error that names the rename and gives the replacement.
        let err = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7

[stages.refine]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 1000
iterations = 20
cooling = 0.95
starts_from = "mle"
        "#).expect_err("legacy starts_from key must produce a load error");

        assert!(err.contains("starts_from"),
            "error must name the legacy key; got: {err}");
        assert!(err.contains("init_mle"),
            "error must point at the replacement key `init_mle`; got: {err}");
        assert!(err.contains("stages.refine") || err.contains("stage `refine`")
                || err.contains("stage 'refine'"),
            "error must locate the offending stage by name; got: {err}");
        assert!(err.contains("2026-05-25-cli-init-and-params-ux"),
            "error must cite the proposal for context; got: {err}");
    }

    #[test]
    fn legacy_init_method_in_nlopt_stage_also_rejected() {
        // Legacy-key detection must cover the NLopt stage variants too
        // (they share the same toml-key shape via NloptStageConfig).
        let err = FitConfigV2::from_toml_str(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "nl-sbplx"
backend = "ode"
chains = 4
init_method = "lhs"
        "#).expect_err("legacy init_method on an nl-sbplx stage must error");

        assert!(err.contains("init_method") && err.contains("init"),
            "nlopt-stage legacy-key error must mirror the IF2/PGAS shape; got: {err}");
    }

    // ── gh#173: strict fit.toml — unknown keys must hard-error ───────────────

    /// A minimal, valid fit.toml. Tests below inject one bad key into it.
    const STRICT_BASE: &str = r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
"#;

    #[test]
    fn strict_base_still_parses() {
        // Guard: the base used by the rejection tests must itself be valid,
        // so a rejection below is attributable to the injected key, not a
        // broken fixture.
        parse(STRICT_BASE).expect("STRICT_BASE must parse cleanly");
    }

    #[test]
    fn top_level_dt_is_rejected() {
        // The honored dt lives under [config]; a top-level `dt` was silently
        // dropped pre-fix (dt=1/2/5 gave byte-identical fits — a wasted
        // timing experiment, gh#173). It must now hard-error and name `dt`.
        let bad = format!("dt = 5.0\n{STRICT_BASE}");
        let err = parse(&bad).expect_err(
            "a top-level `dt` (belongs under [config]) must be rejected");
        assert!(err.contains("dt"),
            "error must name the offending key `dt`; got: {err}");
    }

    #[test]
    fn typoed_top_level_key_is_rejected() {
        // A typo'd key (here `iteration`, a near-miss for the stage's
        // `iterations`) must not be silently ignored — strict config.
        let bad = format!("iteration = 80\n{STRICT_BASE}");
        let err = parse(&bad).expect_err(
            "a typo'd top-level key must be rejected, not silently dropped");
        assert!(err.contains("iteration"),
            "error must name the offending key `iteration`; got: {err}");
    }

    #[test]
    fn fixed_params_still_accept_arbitrary_param_keys() {
        // Guard the deny_unknown_fields CAVEAT: [fixed] uses serde(flatten)
        // for arbitrary `param = value` entries, so it must NOT gain
        // deny_unknown_fields — a model parameter name is a legitimate
        // "unknown" key there. STRICT_BASE's [fixed] already carries N0; this
        // confirms an additional arbitrary param key flattens in.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000
some_param = 0.5

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
"#).expect("arbitrary [fixed] param keys must still be accepted");
        assert!(cfg.fixed.values.contains_key("some_param"),
            "[fixed] must keep flattening arbitrary param keys");
    }
}
