//! Fit.toml schema types.
//!
//! A `fit.toml` is one (problem, method) pair. The *problem* — model, data,
//! the estimate/fixed partition, scenario, simulator settings — is the half
//! every command that takes `--fit` reads. The *method* — one algorithm with
//! its knobs and a chain-starts rule under `[method]` — is the half only
//! `fit run` reads. The file is parsed flat (serde's `flatten` is incompatible
//! with `deny_unknown_fields`), then split into [`Problem`] and [`Inference`];
//! [`Problem::load`] is the entry point for every non-fit reader and discards
//! the inference half, so a file with no `[method]` is a complete problem for
//! them and `fit run` refuses it by name.
//!
//! The store factors a fit the same way — the fit level hashes the problem
//! alone, the method level hashes `[method]` — so a second way of fitting one
//! problem is a second file, and the two land under one fit-level digest.
//! Proposal: `docs/dev/proposals/2026-09-08-workflow-first-fit-config.md`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

pub use super::starts::{ChainStarts, Point, Spread};

// ─── Top-level ──────────────────────────────────────────────────────────────

/// The file as written: every top-level key, parsed flat.
///
/// `deny_unknown_fields` (gh#173): a misplaced or typo'd top-level key is a
/// hard error, not a silent drop. The honored `dt` lives under `[config]`; a
/// top-level `dt` used to be silently ignored (dt=1/2/5 gave byte-identical
/// fits — a wasted timing experiment). The same strictness is applied to the
/// nested config structs below, except `FixedParams`, whose `#[serde(flatten)]`
/// for arbitrary `param = value` entries is incompatible with — and the very
/// opposite of — `deny_unknown_fields`. This is also why the split into
/// [`Problem`] / [`Inference`] happens after the parse rather than through
/// `flatten` on two nested structs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FitConfigWire {
    model: ModelRef,
    #[serde(default)]
    data: Option<DataSpec>,
    #[serde(default)]
    synthetic: Option<SyntheticSpec>,
    #[serde(default)]
    fit_seeds: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    simplex_groups: Vec<SimplexGroup>,
    #[serde(default)]
    output_dir: Option<String>,
    estimate: IndexMap<String, EstimateSpecV2>,
    fixed: FixedParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    method: Option<Method>,
    #[serde(default)]
    config: FitBackendConfig,
    #[serde(default)]
    scenario: Option<String>,
    #[serde(default)]
    enable: Vec<String>,
    #[serde(default)]
    disable: Vec<String>,
    #[serde(default)]
    ic_free: Option<bool>,
    #[serde(default)]
    provenance: Option<FitProvenance>,
}

/// The inference problem: what is estimated, from what, on which model.
///
/// Every command that accepts `--fit` takes `&Problem` and nothing more. It
/// is what the fit level of the store hashes (`fit::cas::fit_level_hash`), so
/// two files that differ only in `[method]` share a fit-level digest.
#[derive(Debug, Clone, Serialize)]
pub struct Problem {
    pub model: ModelRef,

    /// Real-data source. At least one of `[data]` / `[synthetic]` must be
    /// present. Both may be: `fit run` then fits the real data, and
    /// `fit recovery` reads the observation design from `[data]` and the truth
    /// from `[synthetic]`.
    pub data: Option<DataSpec>,

    /// Synthetic-data source — generates N datasets from known truth and
    /// fits each one (simulation-based calibration). See proposal
    /// docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md §"Config
    /// shape".
    pub synthetic: Option<SyntheticSpec>,

    /// Simplex constraints between estimated parameters. Each group's
    /// members must appear in `[estimate]`, be non-negative, and form
    /// a probability simplex (sum = 1). This is a *parameter-space
    /// property*, not an algorithm knob — algorithms read it.
    ///
    /// IF2 perturbs members jointly via barycentric (log-ratio + softmax)
    /// transform; a member's `rw_sd` is interpreted on the log-ratio
    /// scale. PGAS / PMMH / PFilter currently treat members as
    /// independent and rely on the model to enforce sum = 1 indirectly
    /// — `validate()` warns when a non-IF2 method runs against a fit
    /// that declares simplex groups.
    ///
    /// Forward-compat note: the natural prior on a simplex is Dirichlet,
    /// which lives at the *group* level (one prior over k correlated
    /// quantities). The schema accommodates a future `prior` field on
    /// `SimplexGroup` without breaking changes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub simplex_groups: Vec<SimplexGroup>,

    /// Where the run tree is written. Provenance, never identity
    /// (`fit_level_hash` strips it).
    pub output_dir: Option<String>,

    /// The free parameters: what the inference algorithm estimates.
    pub estimate: IndexMap<String, EstimateSpecV2>,

    /// The fixed parameters: held constant during inference.
    /// estimate ∪ fixed must cover all model parameters.
    pub fixed: FixedParams,

    /// Time step and observation alignment. Default dt=1.0.
    pub config: FitBackendConfig,

    /// Named scenario from the model. Applies scenario's enable/disable lists
    /// and param overrides before inference. Mutually exclusive with
    /// `enable`/`disable`. Per spec §14.4, toggleable interventions default
    /// OFF; events always fire unless explicitly disabled.
    pub scenario: Option<String>,
    /// Ad-hoc enable list (intervention names or family base_names).
    /// Wildcard `"*"` enables every toggleable intervention.
    pub enable: Vec<String>,
    /// Ad-hoc disable list. Explicit disable wins over always_active —
    /// the only way to silence an event during inference.
    pub disable: Vec<String>,

    /// IC-free inference: condition the likelihood on the first
    /// observation rather than an initial-state commitment. Absent or
    /// false means standard inference over `y_{1:T}` with a committed
    /// initial state. True means the PF / IF2 / PGAS weight-and-resample
    /// at y₁ (pinning the initial state) but accumulate log-likelihood
    /// only from y₂ onward. Conditions the estimand; validated per method.
    ///
    /// See docs/dev/proposals/archive/pre-alpha/2026-04-18-ic-free-inference.md.
    pub ic_free: Option<bool>,

    /// Optional lineage metadata (not used by the runner).
    pub provenance: Option<FitProvenance>,

    /// Runtime-only: the path to the model **already compiled to IR**
    /// (`.ir.json`). `cmd_fit_run_v2` compiles `model.camdl` → IR exactly
    /// once up front and records the temp path here; every per-cell
    /// `FitRunConfig::build` then loads this pre-compiled IR instead of
    /// re-invoking camdlc per (cell × sweep point). `None` means "compile from
    /// `model.camdl`" (the fallback for unit tests that build a config
    /// directly). Never serialized — `model.camdl` remains the sole
    /// identity-bearing source path (the fit content hash hashes its bytes).
    #[serde(skip)]
    pub compiled_ir: Option<String>,
}

/// The inference half: what only `fit run` reads.
#[derive(Debug, Clone, Serialize)]
pub struct Inference {
    /// `[method]`. `None` is a complete problem with no method, which every
    /// non-fit reader accepts and `fit run` refuses by name. No map, no order,
    /// no chaining: a second way of fitting the problem is a second file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<Method>,

    /// Fit RNG seeds. A list (`[42]` for a single fit, `[101, 102, 103]`
    /// for start-sensitivity sweeps). When absent, the top-level
    /// `--seed` CLI flag (or its default) is used as the single seed.
    /// Duplicates are rejected at validation time — each seed must be
    /// unique to avoid provenance-hash collisions.
    pub fit_seeds: Option<Vec<u64>>,
}

/// The file: a problem and the inference half. `fit run` reads both;
/// everything else takes [`Problem`].
#[derive(Debug, Clone)]
pub struct FitConfig {
    pub problem: Problem,
    pub inference: Inference,
}

impl FitConfigWire {
    fn split(self) -> FitConfig {
        let FitConfigWire {
            model, data, synthetic, fit_seeds, simplex_groups, output_dir, estimate, fixed,
            method, config, scenario, enable, disable, ic_free, provenance,
        } = self;
        FitConfig {
            problem: Problem {
                model, data, synthetic, simplex_groups, output_dir, estimate, fixed, config,
                scenario, enable, disable, ic_free, provenance, compiled_ir: None,
            },
            inference: Inference { method, fit_seeds },
        }
    }
}

impl FitConfig {
    /// The file's flat shape, for the whole-document serializations
    /// (`config_identity_hash`).
    fn to_wire(&self) -> FitConfigWire {
        let Problem {
            model, data, synthetic, simplex_groups, output_dir, estimate, fixed, config,
            scenario, enable, disable, ic_free, provenance, compiled_ir: _,
        } = self.problem.clone();
        let Inference { method, fit_seeds } = self.inference.clone();
        FitConfigWire {
            model, data, synthetic, fit_seeds, simplex_groups, output_dir, estimate, fixed,
            method, config, scenario, enable, disable, ic_free, provenance,
        }
    }

    /// The one `[method]`, or the error `fit run` prints for a file that has
    /// none.
    pub fn method(&self) -> Result<&Method, String> {
        self.inference.method.as_ref().ok_or_else(|| {
            "this fit.toml declares no `[method]` table, so there is nothing to \
             run. It is a complete problem for `simulate --fit`, `pfilter --fit`, \
             `survey --fit` and `profile --fit`; to fit it, add\n  \
             [method]\n  \
             algorithm = \"pgas\"          # if2 | pgas | pmmh | mh | nuts | pfilter | nl-sbplx | nl-bobyqa\n  \
             backend   = \"chain_binomial\"\n  \
             ...\n  \
             See `camdl docs fit-toml`."
                .to_string()
        })
    }
}

impl Serialize for FitConfig {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.to_wire().serialize(ser)
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
    /// proposal (Algorithm 2). `skip_serializing_if None` keeps it OUT of the fit
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
    /// are withheld from training; `camdl compare` scores them out-of-sample
    /// (gh#585, Algorithm 3 of the 2026-08-29 honest-predictive-evaluation
    /// proposal). Accepts a bare model-time number or the shared time-spec
    /// grammar (`parse_time_spec`: a date under a calendar-anchored model,
    /// or `last_obs - N weeks`), resolved at fit load.
    /// Mutually exclusive with `holdout`.
    #[serde(default)]
    pub holdout_after: Option<TimeSpecToml>,

    /// Explicit holdout data files. Keys match observation stream names.
    /// Mutually exclusive with `holdout_after`.
    #[serde(default)]
    pub holdout: Option<IndexMap<String, String>>,
}

/// A time value in `fit.toml` that feeds the shared time-spec grammar
/// (`parse_time_spec`, gh#626): TOML lets the user write either a bare
/// number (`holdout_after = 120.0`) or a string spec
/// (`holdout_after = "2020-03-01"`, `"last_obs - 6 weeks"`). Both arms
/// canonicalize to the raw string the parser consumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TimeSpecToml {
    Num(f64),
    Spec(String),
}

impl TimeSpecToml {
    /// The raw text handed to `parse_time_spec`.
    pub fn raw(&self) -> String {
        match self {
            TimeSpecToml::Num(v) => format!("{v}"),
            TimeSpecToml::Spec(s) => s.clone(),
        }
    }
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

    /// Perturb this parameter only at t=0, never at an observation — the
    /// IF2 perturbation schedule for an initial-state parameter (`S0`, `I0`,
    /// …), whose effect on the trajectory is spent before the first step.
    /// IF2 is the only algorithm with a perturbation schedule, so declaring
    /// this under any other algorithm is a config-load error.
    #[serde(default)]
    pub perturb_only_at_t0: bool,

    /// Per-parameter random walk SD for IF2. If omitted, auto-scaled from bounds.
    #[serde(default)]
    pub rw_sd: Option<f64>,

    /// Starting value: the base point a `starts = "single"` rule puts every
    /// chain at, and the point `"uniform"` keeps for chain 1. Omitted: the
    /// model's declared value, else a transform-aware draw within the bounds.
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

// ─── Method ─────────────────────────────────────────────────────────────────

/// Serialisation predicate for `Algorithm::PGAS.binomial`.
///
/// Deliberately named for the VALUE, not for the default: absence in a stored
/// payload means BTPE, permanently, because that is what every run predating
/// the field used. If the default flips (gh#761) this must NOT follow it.
fn is_btpe(a: &sim::rng::BinomialAlgorithm) -> bool {
    matches!(a, sim::rng::BinomialAlgorithm::Btpe)
}

/// One inference algorithm with its knobs — the `[method]` table minus
/// `starts`. Tagged by `algorithm`. Each variant carries an explicit `backend`
/// field; the (algorithm, backend) pair is validated against
/// `methods::METHODS` at config-load time. See proposal
/// 2026-05-04-ode-inference-three-phase.md §"Tuple schema" for the rationale
/// (algorithm and backend used to be smuggled together as `method = "if2"`
/// implying chain_binomial).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "algorithm")]
pub enum Algorithm {
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
        /// gh#747: which binomial sampler the chain-binomial draws use.
        /// `btpe` (default) or `btrs`.
        ///
        /// Selecting `btrs` CHANGES DRAWS — a different rejection scheme
        /// accepts different values from the same stream — so this field is
        /// deliberately part of the stage's identity. `identity_payload` is
        /// subtractive, so it is hashed by being here; two runs differing only
        /// in this field get different addresses and cannot be served from one
        /// another's leaf. That is why it is a typed field and not an
        /// environment variable (gh#241 removed the last of those rather than
        /// hash it).
        ///
        /// `serde(default)` is load-bearing: without it every existing
        /// `fit.toml` that predates this field fails to deserialize.
        ///
        /// `skip_serializing_if` keeps a `btpe` stage's payload BYTE-IDENTICAL
        /// to what it was before this field existed, so adding the field
        /// orphans no stored leaf and breaks no in-flight `--resume`. Only a
        /// `btrs` stage serialises it, and so only a `btrs` stage gets a new
        /// address — which is the only place the guarantee is needed.
        ///
        /// The predicate tests `== Btpe`, NOT `== default()`, and the
        /// difference bites exactly once: if the default ever becomes `Btrs`
        /// (gh#761), a default-tracking predicate would silently make ABSENCE
        /// mean `btrs`, retroactively changing what every already-stored
        /// address asserts. Absence must mean BTPE forever, because that is
        /// what every run predating this field actually used.
        #[serde(default, skip_serializing_if = "is_btpe")]
        binomial: sim::rng::BinomialAlgorithm,
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
        /// Run the ancestor-sampling move in the CSMC sweep. `false` is plain
        /// particle Gibbs without AS (Andrieu, Doucet & Holenstein 2010) — a
        /// valid kernel used as a diagnostic control: it measures what AS
        /// contributes to trajectory renewal, and what its density pass costs.
        /// Spelled `ancestor_sampling = false` in the stage TOML or
        /// `--no-ancestor-sampling` on the CLI. Default: true.
        ///
        /// Identity: disabling AS changes the sampled draws, so `false` must
        /// re-key — and it does, by serializing into the payload.
        /// `skip_serializing_if` keeps the default's payload byte-identical to
        /// the pre-field format, so adding the field orphans no stored leaf
        /// and breaks no in-flight `--resume`. The predicate tests `== true`
        /// literally, not `== default()`: absence must mean AS-on permanently,
        /// because that is what every run predating the field did (the same
        /// reasoning as `binomial`'s absence-means-btpe).
        #[serde(default = "default_ancestor_sampling",
                skip_serializing_if = "ancestor_sampling_is_on")]
        ancestor_sampling: bool,
    },

    #[serde(rename = "pmmh")]
    PMMH {
        backend: crate::run_meta::InferenceBackend,
        chains: usize,
        particles: usize,
        iterations: usize,
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
        /// Coarse RK4 step for the *unscored* warm-up `[t_start, first_obs)` on the
        /// deterministic ODE likelihood (gh#396 follow-on). `None` (default) or a
        /// value `<= dt` disables it — the whole trajectory integrates at `dt`. A
        /// larger value takes big steps on the transient, cutting the per-eval cost
        /// of a model whose origin is long before the data (no monodromy, so it is
        /// dimension-free — unlike the equilibrium warm-start). Forcing-only warm-up
        /// only: a model with `events {}` / `balance {}` in the warm-up is refused
        /// (per-substep constructs cannot be coarsened). Identity-defining: it
        /// changes the scored trajectory, so it re-keys the run.
        #[serde(default)]
        burnin_dt: Option<f64>,
        /// Post-fit deterministic-ODE dt-check at the MAP (gh#52, gh#227).
        /// Same schema as `Algorithm::IF2::dt_check`. Identity-defining: the
        /// result is stored in `fit_state.toml.dt_check` (gh#726 — this
        /// field's addition is what made the CLI dt-check flags keyable
        /// on mh stages).
        #[serde(default)]
        dt_check: DtCheckConfig,
    },

    #[serde(rename = "pfilter")]
    PFilter {
        backend: crate::run_meta::InferenceBackend,
        particles: usize,
        #[serde(default)]
        replicates: Option<usize>,

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
    /// Number of starting points, each run as an independent NLopt
    /// optimization to convergence; best-loglik chain wins. Sbplx/BOBYQA are
    /// deterministic, so `chains > 1` is only meaningful under a spread
    /// `starts` rule; `starts = "single"` collapses to one chain.
    ///
    /// Caveat for very wide bounds: if `[estimate]` bounds span regions where
    /// transmission collapses (e.g. R0 < 1 in any setting), some spread draws
    /// may evaluate to `Poisson(rate=0) | obs > 0 = -inf`. NLopt's
    /// xtol-reached signal can lie there (every neighbouring point also
    /// -inf). For such models, narrow the bounds or pre-validate the starts
    /// with a quick `camdl pfilter` loglik check.
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
    /// Convergence-gate thresholds. Two-leg version of IF2's gate
    /// (chain-agreement + decibans-spread); see proposal §"Convergence
    /// diagnostics for NLopt chains".
    #[serde(default)]
    pub gate: GateConfig,
    /// Post-fit deterministic-ODE dt-check at θ̂ (gh#52, gh#227). Same
    /// schema as `Algorithm::IF2::dt_check`. Identity-defining: the result is
    /// stored in `fit_state.toml.dt_check` (gh#726).
    #[serde(default)]
    pub dt_check: DtCheckConfig,
}

fn default_nlopt_tolerance() -> f64 { 1e-6 }
fn default_nlopt_max_evals() -> usize { 5000 }

/// One way of fitting the problem: the `[method]` table. A file carries at
/// most one.
///
/// `starts` is `None` while unresolved — the file omitted the key, and the
/// default is a function of the problem's priors (`from_prior` when every
/// estimated parameter declares one, `uniform_unconstrained` otherwise), which
/// the loader cannot see without the model. `fit run` resolves it through
/// [`Method::resolve_starts`] before the identity is taken, so the hashed
/// payload always carries a concrete rule and a file that spells the default
/// keys the same as one that omits it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Method {
    #[serde(flatten)]
    pub algorithm: Algorithm,
    /// Where the chains begin. See [`ChainStarts`] for the wire form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starts: Option<ChainStarts>,
}

impl Method {
    /// The chain-starts rule this method runs under, or the error a caller
    /// that needs a concrete rule prints when [`Method::resolve_starts`] was
    /// never run — a wiring bug, not a user error.
    pub fn starts(&self) -> Result<&ChainStarts, String> {
        self.starts.as_ref().ok_or_else(|| {
            "internal: `starts` is unresolved; `Method::resolve_starts` must run \
             before the method is used or hashed"
                .to_string()
        })
    }

    /// Resolve an absent `starts` to the default rule (§3.4): `from_prior`
    /// when every estimated parameter's resolved prior is a distribution the
    /// chains can be drawn from, `uniform_unconstrained` otherwise. Returns
    /// what was decided and why, for the run's startup block; a rule the file
    /// spelled out is left alone and reported as declared.
    ///
    /// The reason is measured (gh#876): a bounds-uniform draw at province
    /// scale is routinely a start the bootstrap filter cannot score, because a
    /// fixed relative error in a rate is a standardised residual that grows
    /// with the square root of the population, and wide bounds carry no
    /// scale. A prior does.
    pub fn resolve_starts(&mut self, problem: &Problem, model: &ir::Model) -> StartsResolution {
        if let Some(declared) = &self.starts {
            return StartsResolution::Declared(declared.clone());
        }
        use super::priors_precedence::{resolve_priors_with_precedence, PriorSource};
        let names: Vec<String> = problem.estimate.keys().cloned().collect();
        let without_prior: Vec<String> =
            resolve_priors_with_precedence(&names, &problem.estimate, model)
                .into_iter()
                .filter(|r| {
                    // A flat prior (explicit or fallen back to) has no
                    // distribution to draw from; a hierarchical prior cannot
                    // be sampled without its hyperparameters' values.
                    matches!(r.source, PriorSource::FlatFallback | PriorSource::FlatExplicit)
                        || r.prior.is_hierarchical()
                })
                .map(|r| r.param)
                .collect();
        let resolved = if without_prior.is_empty() {
            StartsResolution::DefaultFromPrior
        } else {
            StartsResolution::DefaultUniformUnconstrained { without_prior }
        };
        self.starts = Some(resolved.rule());
        resolved
    }

    /// Hashable subset of the method that defines its statistical
    /// identity: the algorithm's fields plus `starts`. For PGAS / PMMH this
    /// *omits* the extension dimension (`sweeps` / `iterations`
    /// respectively), so `--resume` can extend a chain by changing only that
    /// field without invalidating the stored `resume_state.bin`. Every other
    /// field is identity-defining: changing chains, particles, burn_in, thin,
    /// or starts requires a fresh run.
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
        // SUBTRACTIVE, not enumerated: serialize the whole method and remove
        // only the keys that must not be hashed. The four sampler arms used
        // to LIST their included fields and destructure the rest with `..`,
        // which made stage identity exclude-by-default — a field added to a
        // variant but forgotten here was silently absent from the key, and
        // two fits differing only in it collided. That is the shape behind
        // gh#514, gh#540 and the 2026-08-23 batch; the invariant even lived
        // in a comment ("`burnin_dt` … MUST be listed here"), because nothing
        // enforced it. Now a new field is hashed unless it is deliberately
        // named below.
        //
        // What is subtracted, and why:
        //  - the EXTENSION DIMENSION (PGAS `sweeps`, PMMH/Mh `iterations`,
        //    Nuts `samples`): a resumed run extends a base run, so the two
        //    must share a prefix identity; the length is folded separately
        //    by `cas_target_length`.
        //  - PGAS `n_trajectories`: an output-shaping count, folded by
        //    `cas_n_trajectories` (count-in-the-key) rather than here, so
        //    hashing it in both places would double-fold it.
        //
        // IF2/PFilter/NLopt have no extension dimension and subtract nothing.
        //
        // Keys are the TOML-side spellings because they come from the
        // method's own serialization; the enumerated arms used the Rust field
        // names, so those two spellings disagreed across variants. `starts`
        // is always concrete here: `resolve_starts` runs before any identity
        // is taken, and `stage_config_hash` refuses an unresolved method.
        // Returned as `serde_json::Value` so
        // `provenance::fit_stage_hash` can hash it via `serde_json::to_vec`,
        // stable across recompiles (serde_json sorts object keys).
        match self.algorithm {
            Algorithm::PGAS { .. } => self.payload_minus(&["sweeps", "n_trajectories"]),
            Algorithm::PMMH { .. } | Algorithm::Mh { .. } => self.payload_minus(&["iterations"]),
            Algorithm::Nuts { .. } => self.payload_minus(&["samples"]),
            Algorithm::IF2 { .. }
            | Algorithm::PFilter { .. }
            | Algorithm::NlSbplx(_)
            | Algorithm::NlBobyqa(_) => self.payload_minus(&[]),
        }
    }

    /// The method serialized in full, minus `exclude`d top-level keys.
    ///
    /// The subtractive primitive behind [`Self::identity_payload`]: include
    /// by default, and name every omission. A missing key here is a
    /// deliberate, greppable decision; a missing field in an enumerated
    /// `json!` was invisible.
    fn payload_minus(&self, exclude: &[&str]) -> serde_json::Value {
        // The shared subtraction (`fit::cas::serialize_minus`), not a third
        // copy of it. Infallible here by the same fallback this always had:
        // `identity_payload` returns a `Value`, and a method that cannot
        // serialize is caught by `stage_config_hash`'s gate, which runs on the
        // method itself.
        super::cas::serialize_minus(self, exclude).unwrap_or_else(|_| serde_json::json!({}))
    }

}

/// What [`Method::resolve_starts`] decided, and why, so the startup block can
/// say it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartsResolution {
    /// The file (or `--starts`) spelled the rule.
    Declared(ChainStarts),
    /// Absent, and every estimated parameter has a sampleable prior.
    DefaultFromPrior,
    /// Absent, and these parameters have no sampleable prior.
    DefaultUniformUnconstrained { without_prior: Vec<String> },
}

impl StartsResolution {
    pub fn rule(&self) -> ChainStarts {
        match self {
            StartsResolution::Declared(rule) => rule.clone(),
            StartsResolution::DefaultFromPrior => ChainStarts::from_prior(),
            StartsResolution::DefaultUniformUnconstrained { .. } => {
                ChainStarts::uniform_unconstrained()
            }
        }
    }

    /// The startup line: the rule, and whether it was declared or defaulted.
    pub fn describe(&self) -> String {
        match self {
            StartsResolution::Declared(rule) => rule.describe(),
            StartsResolution::DefaultFromPrior => format!(
                "{} — default: every estimated parameter declares a prior",
                ChainStarts::from_prior().describe()
            ),
            StartsResolution::DefaultUniformUnconstrained { without_prior } => format!(
                "{} — default: no sampleable prior on {}",
                ChainStarts::uniform_unconstrained().describe(),
                without_prior.join(", ")
            ),
        }
    }
}

impl Algorithm {
    pub fn method_name(&self) -> &'static str {
        self.method_kind().as_str()
    }

    pub fn method_kind(&self) -> crate::run_meta::FitAlgorithm {
        use crate::run_meta::FitAlgorithm;
        match self {
            Algorithm::IF2      { .. } => FitAlgorithm::If2,
            Algorithm::PGAS     { .. } => FitAlgorithm::Pgas,
            Algorithm::PMMH     { .. } => FitAlgorithm::Pmmh,
            Algorithm::Mh       { .. } => FitAlgorithm::Mh,
            Algorithm::Nuts     { .. } => FitAlgorithm::Nuts,
            Algorithm::PFilter  { .. } => FitAlgorithm::Pfilter,
            Algorithm::NlSbplx  { .. } => FitAlgorithm::NlSbplx,
            Algorithm::NlBobyqa { .. } => FitAlgorithm::NlBobyqa,
        }
    }

    /// Simulation backend the stage runs on. The (algorithm, backend)
    /// pair is set by the user in fit.toml and validated against
    /// `methods::METHODS`; this accessor returns whichever backend was
    /// declared so dispatch and provenance can branch on it.
    pub fn backend(&self) -> crate::run_meta::InferenceBackend {
        match self {
            Algorithm::IF2      { backend, .. }
            | Algorithm::PGAS    { backend, .. }
            | Algorithm::PMMH    { backend, .. }
            | Algorithm::Mh      { backend, .. }
            | Algorithm::Nuts    { backend, .. }
            | Algorithm::PFilter { backend, .. } => *backend,
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) => c.backend,
        }
    }

    pub fn requires_priors(&self) -> bool {
        matches!(self, Algorithm::PGAS { .. } | Algorithm::PMMH { .. } | Algorithm::Mh { .. } | Algorithm::Nuts { .. })
    }

    pub fn chains(&self) -> usize {
        match self {
            Algorithm::IF2 { chains, .. } => *chains,
            Algorithm::PGAS { chains, .. } => *chains,
            Algorithm::PMMH { chains, .. } => *chains,
            Algorithm::Mh { chains, .. } => *chains,
            Algorithm::Nuts { chains, .. } => *chains,
            Algorithm::PFilter { .. } => 1,
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) => c.chains,
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
            Algorithm::IF2 { iterations, .. } => *iterations as u64,
            Algorithm::PGAS { sweeps, .. } => *sweeps as u64,
            Algorithm::PMMH { iterations, .. } => *iterations as u64,
            Algorithm::Mh { iterations, .. } => *iterations as u64,
            Algorithm::Nuts { samples, .. } => *samples as u64,
            Algorithm::PFilter { .. } | Algorithm::NlSbplx(_) | Algorithm::NlBobyqa(_) => 0,
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
            Algorithm::PGAS { n_trajectories, .. } => *n_trajectories as u64,
            _ => 0,
        }
    }

    /// Overwrite this method's sampler and output knobs from the CLI.
    ///
    /// gh#514 / gh#540: these are keyed fields — `Method::identity_payload`
    /// folds every one of them into the method hash — but the CLI overrides
    /// used to be applied at the *dispatch* site, well after the CAS claim, so
    /// two runs differing only in a flag shared a `run_id` and the second was
    /// served the first's result. Writing them into the in-memory method
    /// BEFORE the claim makes a different flag a different artifact, exactly
    /// as a different toml value already is. Any CLI override of a keyed field
    /// must write into the config before `cas::fit_level_hash` for the same
    /// reason. `--starts` takes the same route through `Method::starts`.
    ///
    /// A `None` argument leaves the field as the toml declared it, so a run
    /// with no CLI overrides keys identically to a bare `fit run` of the same
    /// file.
    pub fn apply_cli_overrides(&mut self, cli: &CliStageOverrides) {
        // ── Sampler and output overrides (gh#540) ──
        // Each of these used to be written into the `*StageOpts` struct at the
        // dispatch site, AFTER the identity was taken from this one — so a run
        // differing only in `--n-trajectories` or `--no-nuts` shared a `run_id`
        // with the previous run and was served its result. Writing them here
        // means the identity sees them, and the dispatch site has nothing left
        // to override: `*StageOpts::from_stage` reads what is written below.
        match self {
            Algorithm::PGAS {
                tempering, max_tree_depth, trajectory_warmup,
                csmc_sweeps_per_nuts, n_trajectories, dense_mass, use_nuts, binomial,
                ancestor_sampling, ..
            } => {
                if let Some(t) = &cli.tempering { *tempering = t.clone(); }
                if let Some(d) = cli.max_tree_depth { *max_tree_depth = d; }
                if let Some(w) = cli.trajectory_warmup { *trajectory_warmup = w; }
                if let Some(s) = cli.csmc_sweeps_per_nuts { *csmc_sweeps_per_nuts = s; }
                if let Some(n) = cli.n_trajectories { *n_trajectories = n; }
                if cli.diagonal_mass { *dense_mass = false; }
                if cli.no_nuts { *use_nuts = false; }
                // gh#747: resolved INTO the stage, so the flag is keyed into
                // the stage's identity rather than acting as an untracked
                // side channel. Two runs differing only here get different
                // addresses.
                if let Some(b) = cli.binomial { *binomial = b; }
                if cli.no_ancestor_sampling { *ancestor_sampling = false; }
            }
            Algorithm::Nuts { max_tree_depth, dense_mass, .. } => {
                if let Some(d) = cli.max_tree_depth { *max_tree_depth = d; }
                if cli.diagonal_mass { *dense_mass = false; }
            }
            Algorithm::PMMH { adapt, adapt_start, rho, .. } => {
                if cli.no_adapt { *adapt = false; }
                if let Some(s) = cli.adapt_start { *adapt_start = s; }
                if let Some(r) = cli.rho { *rho = Some(r); }
            }
            Algorithm::Mh { adapt, adapt_start, dt_check, backend, .. } => {
                if cli.no_adapt { *adapt = false; }
                if let Some(s) = cli.adapt_start { *adapt_start = s; }
                if cli.no_dt_check { dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { dt_check.n_halvings = n; }
                resolve_dt_check_strict(dt_check, *backend, cli.dt_check_strict);
            }
            Algorithm::IF2 { cooling_target_iters, gate, dt_check, backend, .. } => {
                if let Some(n) = cli.cooling_target_iters { *cooling_target_iters = n; }
                if let Some(db) = cli.decibans_thresh { gate.decibans_thresh = db; }
                if cli.no_dt_check { dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { dt_check.n_halvings = n; }
                resolve_dt_check_strict(dt_check, *backend, cli.dt_check_strict);
            }
            Algorithm::PFilter { record_ancestry, record_prequential, .. } => {
                // One-way overrides to true: the toml can opt out
                // (`record_prequential = false`), the flag opts back in.
                if cli.record_ancestry { *record_ancestry = true; }
                if cli.record_prequential { *record_prequential = true; }
            }
            Algorithm::NlSbplx(cfg) | Algorithm::NlBobyqa(cfg) => {
                if let Some(db) = cli.decibans_thresh { cfg.gate.decibans_thresh = db; }
                if cli.no_dt_check { cfg.dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { cfg.dt_check.n_halvings = n; }
                let backend = cfg.backend;
                resolve_dt_check_strict(&mut cfg.dt_check, backend, cli.dt_check_strict);
            }
        }
    }
}

/// Every CLI flag that changes what a method COMPUTES or STORES, collected in
/// one place so it can be written into the method before its content address
/// is taken. `--starts` is the one exception by design: it writes
/// `Method::starts` directly, which the identity payload also carries.
///
/// The point of the struct is that it is the only route. gh#514 fixed five
/// chain-start flags by folding them into the identity; gh#540 found thirteen
/// more with the same defect, because each was applied at its own dispatch
/// site and nothing forced a new flag to go through the identity. A flag added
/// to `FitRunArgs` and forgotten here still bypasses — but it now has to bypass
/// ONE place with a name, rather than blend into a list of `if let Some(..)`
/// lines two hundred lines from the claim.
///
/// Presentation-only flags (progress bars, `--no-run`, output verbosity) are
/// deliberately absent: they do not change the artifact, so keying on them
/// would invalidate cached fits for nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CliStageOverrides {
    pub tempering: Option<Vec<f64>>,
    pub max_tree_depth: Option<usize>,
    pub trajectory_warmup: Option<usize>,
    pub csmc_sweeps_per_nuts: Option<usize>,
    pub n_trajectories: Option<usize>,
    pub diagonal_mass: bool,
    pub no_nuts: bool,
    /// `--no-ancestor-sampling` (PGAS): one-way override to plain particle
    /// Gibbs without the AS move. Identity-bearing like `no_nuts`.
    pub no_ancestor_sampling: bool,
    pub no_adapt: bool,
    pub adapt_start: Option<usize>,
    pub rho: Option<f64>,
    /// `--cooling-target-iters` (IF2): iterations over which the cooling
    /// fraction is reached. Changes the perturbation schedule and therefore
    /// the MLE.
    pub cooling_target_iters: Option<usize>,
    /// `--decibans-thresh`: overrides `gate.decibans_thresh` on stages that
    /// carry a `GateConfig` (IF2, nl-sbplx, nl-bobyqa). The applied gate is
    /// persisted in the leaf (`resolved_gate`), so it is stored output.
    pub decibans_thresh: Option<f64>,
    /// `--no-dt-check`: disables the post-fit Richardson dt-check, whose
    /// result is stored in `fit_state.toml.dt_check`. Applies to the
    /// stages that run one: if2, mh, nl-* (gh#726).
    pub no_dt_check: bool,
    /// `--dt-check-halvings`: dt-check ladder depth. Same stages as
    /// `no_dt_check`.
    pub dt_check_halvings: Option<usize>,
    /// `--dt-check-strict`: use the strict warning threshold. Resolved to a
    /// concrete `dt_check.threshold_nats` when applied (gh#730), because the
    /// threshold is STORED in `fit_state.toml.dt_check` — it was treated as
    /// leaf-byte-neutral abort policy, but it selects the stored verdict and
    /// threshold whenever the TOML leaves `threshold_nats` unset.
    pub dt_check_strict: bool,
    /// gh#747: `--binomial`. `None` leaves the stage's own value (or its
    /// `serde(default)`) untouched.
    pub binomial: Option<sim::rng::BinomialAlgorithm>,
    /// `--record-ancestry` (PFilter): one-way override to true; adds the
    /// ancestor trace to the stored leaf.
    pub record_ancestry: bool,
    /// `--record-prequential` (PFilter): one-way override to true; adds the
    /// prequential trace to the stored leaf.
    pub record_prequential: bool,
}

/// Resolve `--dt-check-strict` into a concrete stored threshold (gh#730).
///
/// The flag was treated as un-keyed abort policy, but it is not
/// leaf-byte-neutral: `threshold_nats` (and the verdict and notes derived from
/// it) are serialized into `fit_state.toml.dt_check`, so two runs differing
/// only in the flag stored different bytes under one `run_id`. Rather than add
/// a `strict` field to the schema, resolve the flag HERE — before the identity
/// is taken — into the `threshold_nats` the config already carries and already
/// hashes.
///
/// A TOML-set `threshold_nats` wins, preserving today's semantics exactly: the
/// flag was already inert whenever the stage declared its own threshold.
fn resolve_dt_check_strict(
    dt_check: &mut DtCheckConfig,
    backend: crate::run_meta::InferenceBackend,
    strict: bool,
) {
    if strict && dt_check.threshold_nats.is_none() {
        dt_check.threshold_nats =
            Some(super::dt_check::default_threshold_for_backend(backend, true));
    }
}

impl CliStageOverrides {
    /// True when nothing was passed — the caller skips the whole override step,
    /// so a run with no CLI overrides keys byte-identically to before and no
    /// stored fit is invalidated.
    ///
    /// Compared structurally against `Self::default()` so a field added to the
    /// struct can never be forgotten here — the previous hand-enumerated
    /// conjunction silently ignored any field it didn't list.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

// ─── Clean-evaluation + gate (IF2 scout/refine) ─────────────────────────────

/// How to combine M independent particle-filter replicate log-likelihoods
/// into a single score for ranking candidate parameter points.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CombineMode {
    /// log( (1/M) Σ exp(ll_k) ) — unbiased on the likelihood scale.
    #[default]
    LogMeanExp,
    /// (1/M) Σ ll_k — biased low, but lower variance.
    Mean,
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
fn default_ancestor_sampling() -> bool { true }
/// The `skip_serializing_if` predicate for `Algorithm::PGAS::ancestor_sampling`.
/// Deliberately `*b` (i.e. `== true`), NOT `== default_ancestor_sampling()`:
/// absence in a stored payload must mean AS-on permanently, whatever the
/// default may later become — see the field's doc comment.
fn ancestor_sampling_is_on(b: &bool) -> bool { *b }
fn default_nuts_warmup() -> usize { 500 }
fn default_nuts_samples() -> usize { 500 }
fn default_target_accept() -> f64 { 0.8 }

/// Validate a `burnin_dt` coarse warm-up step (gh#396 follow-on) against the
/// fit-wide `dt` and the observation streams. Shared by the `nuts` (gradient) and
/// `mh` (deterministic) ODE paths — the soundness rules are identical: coarsening
/// is valid only for a prevalence (state-scored) fit with a genuine unscored
/// warm-up window and a step LARGER than `dt`. Returns the effective step: `dt`
/// (off) for `None` / `== dt`; the validated `b` otherwise. The `events`/`balance`
/// per-substep refusal is a model-structure gate enforced downstream in `run_ode`
/// / the gradient integrator, not here.
pub fn validate_burnin_dt(
    burnin_dt: Option<f64>,
    dt: f64,
    n_interval_streams: usize,
    first_obs: Option<f64>,
    t_start: f64,
) -> Result<f64, String> {
    match burnin_dt {
        Some(b) if b > dt => {
            // Incidence (interval) streams: the first scored bin accumulates flow
            // from `t_start`, so coarsening the warm-up would bias a scored datum.
            // Prevalence (state-scored) is safe (only the state at each obs matters).
            if n_interval_streams > 0 {
                return Err(format!(
                    "burnin_dt = {b} is only supported for prevalence (state-scored) \
                     streams in this release, but this fit has an incidence (interval) \
                     stream whose first bin accumulates flow from t_start — coarsening \
                     the warm-up would bias it. Remove burnin_dt (incidence \
                     coarse-transient support is a follow-up)."
                ));
            }
            // There must be an unscored warm-up window to coarsen.
            match first_obs {
                Some(fo) if fo > t_start => Ok(b),
                first => Err(format!(
                    "burnin_dt = {b} was set, but the first observation (t = {}) is \
                     at or before the model start (t_start = {t_start}) — there is no \
                     unscored warm-up window to coarsen. Remove burnin_dt, or start \
                     the model earlier so there is a transient to integrate coarsely.",
                    first.map(|f| f.to_string()).unwrap_or_else(|| "none".to_string())
                )),
            }
        }
        Some(b) if b < dt => Err(format!(
            "burnin_dt = {b} is smaller than the integrator step dt = {dt}. The \
             coarse burn-in step must be LARGER than dt — it takes bigger steps on \
             the unscored warm-up; a smaller value would refine, not coarsen. Set \
             burnin_dt >= {dt} (e.g. 7.0), or remove it to integrate the whole \
             trajectory at dt."
        )),
        // Some(b) with b == dt, or None: off (fine step throughout).
        _ => Ok(dt),
    }
}

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

// ─── Provenance ─────────────────────────────────────────────────────────────

/// Optional metadata linking this fit to a parent.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitProvenance {
    pub derived_from: Option<String>,
    pub reason: Option<String>,
}

// ─── Loading + Validation ───────────────────────────────────────────────────

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

/// The error a `condition_from` key in a fit.toml raises. The key opened an
/// incidence stream's first bin somewhere other than `t_start` when nothing
/// else could; every incidence stream now declares what its rows cover, and a
/// first period opens where it says, so the key has no job left. Kept as a
/// hard error rather than an unknown-field rejection so the user is told what
/// replaced it (proposal 2026-09-05-observation-time-as-a-sum-type, ruling 4).
pub const CONDITION_FROM_REMOVED_MSG: &str =
    "condition_from = ... is no longer a fit.toml key: an incidence stream now \
     states what each row covers (`covers = ...` or window columns in the \
     model), and a declared first period opens where it says, so the warm-up \
     before the first row is discarded without a separate setting. Delete the \
     key.";

/// Refuse a removed `condition_from` key before the strict
/// `deny_unknown_fields` parse, so the user gets
/// [`CONDITION_FROM_REMOVED_MSG`] rather than a bare serde "unknown field"
/// error. Fires on the key at the top level (both the `condition_from = "…"`
/// scalar and the `[condition_from]` table spell a top-level key) and on the
/// key anywhere under `[data]`, where a user may have guessed it belongs.
fn detect_removed_condition_from(contents: &str) -> Result<(), String> {
    fn table_has_key_anywhere(value: &toml::Value, key: &str) -> bool {
        match value.as_table() {
            Some(t) => t.contains_key(key)
                || t.values().any(|v| table_has_key_anywhere(v, key)),
            None => false,
        }
    }
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // Malformed TOML surfaces from the strict parse with its own message.
        Err(_) => return Ok(()),
    };
    let at_top = value.get("condition_from").is_some();
    let under_data = value
        .get("data")
        .is_some_and(|d| table_has_key_anywhere(d, "condition_from"));
    if at_top || under_data {
        return Err(CONDITION_FROM_REMOVED_MSG.into());
    }
    Ok(())
}

/// The `[stages]` rejection (proposal 2026-09-08-workflow-first-fit-config,
/// §5). A file that still carries the stage map is refused at load with the
/// rewrite spelled out: the last-declared stage becomes the file's `[method]`,
/// every other stage goes in its own file, `init` is `starts`, and a chained
/// `init_mle` — which started every chain at the upstream point estimate and
/// made R̂ uninformative — is handed back to the author as a decision, because
/// the file cannot name a handle that does not exist until the upstream has
/// run. No silent conversion and no compatibility path.
///
/// Declaration order is read off the raw text (`[stages.<name>]` headers in
/// order), because `toml::Table` sorts its keys and the stage that should
/// survive as `[method]` is the one declared last.
pub fn detect_legacy_stages(contents: &str, file_name: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // Malformed TOML surfaces from the strict parse with its own message.
        Err(_) => return Ok(()),
    };
    let Some(stages) = value.get("stages").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    // Declaration order: `[stages.<name>]` headers as written.
    let mut declared: Vec<String> = Vec::new();
    for line in contents.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("[stages.") {
            if let Some(name) = rest.split(']').next() {
                let name = name.trim().trim_matches('"').to_string();
                if stages.contains_key(&name) && !declared.contains(&name) {
                    declared.push(name);
                }
            }
        }
    }
    // Any stage the header scan missed (a dotted-key form) goes last, sorted.
    for name in stages.keys() {
        if !declared.contains(name) {
            declared.push(name.clone());
        }
    }
    let Some(primary) = declared.last().cloned() else {
        return Err(
            "legacy table `[stages]` (empty)\n  \
             replacement: delete it. A file with no `[method]` is a complete problem for \
             `simulate --fit`, `pfilter --fit`, `survey --fit` and `profile --fit`; \
             `fit run` needs a `[method]`.\n  \
             See `camdl docs fit-toml`."
                .into(),
        );
    };
    let others: Vec<&String> = declared.iter().filter(|n| **n != primary).collect();

    let mut msg = format!(
        "legacy table `[stages.{primary}]`\n  \
         replacement: rename to `[method]` and run it with\n    \
         camdl fit run {file_name}\n"
    );
    if !others.is_empty() {
        let listed: Vec<String> = others.iter().map(|n| format!("`[stages.{n}]`")).collect();
        msg.push_str(&format!(
            "  a file carries one `[method]`; put {} in its own file\n",
            listed.join(" and ")
        ));
    }
    for name in &declared {
        let Some(table) = stages.get(name).and_then(toml::Value::as_table) else { continue };
        if let Some(toml::Value::String(upstream)) = table.get("init_mle") {
            // A stage name becomes a label handle; a directory stays a path.
            let handle = if stages.contains_key(upstream) {
                format!("@{upstream}")
            } else {
                upstream.clone()
            };
            msg.push_str(&format!(
                "  `init_mle = \"{upstream}\"` has no replacement in the file: it started every \
                 chain at\n  \
                 {upstream}'s point estimate, which makes R̂ uninformative. Run {upstream} first \
                 and, if a\n  \
                 warm start is wanted, write one of\n    \
                 starts = {{ from_posterior = \"{handle}\" }}   # one draw per chain (keeps R̂ \
                 meaningful)\n    \
                 starts = {{ from_mle = \"{handle}\" }}         # every chain at one point (R̂ not \
                 assessed)\n"
            ));
        }
    }
    let mut saw_init: Option<String> = None;
    let mut saw_survey = false;
    for name in &declared {
        let Some(table) = stages.get(name).and_then(toml::Value::as_table) else { continue };
        if let Some(toml::Value::String(init)) = table.get("init") {
            if init == "survey_top_k" {
                saw_survey = true;
            } else if saw_init.is_none() {
                saw_init = Some(init.clone());
            }
        }
        if table.contains_key("survey_path") || table.contains_key("survey_top_k_n") {
            saw_survey = true;
        }
    }
    if let Some(init) = saw_init {
        msg.push_str(&format!("  `init = \"{init}\"` becomes `starts = \"{init}\"`\n"));
    }
    if saw_survey {
        msg.push_str(
            "  `init = \"survey_top_k\"` (with `survey_path` / `survey_top_k_n`) was removed: a \
             survey\n  \
             landscape is not a posterior. Use `starts = \"from_prior\"`, or run a short fit \
             and\n  \
             write `starts = { from_posterior = \"@handle\" }`.\n",
        );
    }
    msg.push_str("  See `camdl docs fit-toml`.");
    Err(msg)
}

/// `fit_starts` was a top-level key that was parsed, serialized into every
/// fit-level identity as `null`, consulted once to silence a warning, and read
/// by nothing. Its one meaningful value, `"prior"`, is now the default
/// `starts` rule whenever every estimated parameter declares a prior.
fn detect_removed_fit_starts(contents: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    if value.get("fit_starts").is_some() {
        return Err(
            "`fit_starts` is no longer a fit.toml key. Chain starts are the `starts` key of \
             `[method]`: `from_prior` is the default whenever every estimated parameter \
             declares a prior, so `fit_starts = \"prior\"` is simply deleted; \
             `fit_starts = \"model_default\"` is `starts = \"single\"`. See `camdl docs \
             fit-toml`."
                .into(),
        );
    }
    Ok(())
}

/// gh#241 (C3): reject unknown keys inside `[method]`.
///
/// `Algorithm` is internally tagged (`#[serde(tag = "algorithm")]`), and serde
/// cannot apply `deny_unknown_fields` to such an enum — so a typo on an
/// *optional* method key (a required-field typo is already caught as "missing
/// field") is silently dropped: neither applied nor reaching the method
/// identity hash. This post-parse pass compares the raw `[method]` keys
/// against the set serde actually recognized: the key set the parsed
/// `Method` serializes back to, plus the keys a `skip_serializing_if` keeps
/// out of that serialization at their default (`binomial`,
/// `ancestor_sampling`, and `starts` when unset). Nested sub-tables
/// (`loglik_eval`/`gate`/`dt_check`) are ordinary structs that carry their own
/// `deny_unknown_fields`, so only the top-level keys need this.
fn validate_method_keys(contents: &str, method: &Method) -> Result<(), String> {
    let raw: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // A genuine parse error already surfaced from the typed parse upstream.
        Err(_) => return Ok(()),
    };
    let Some(raw_method) = raw.get("method").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    let mut known: BTreeSet<String> = serde_json::to_value(method)
        .ok()
        .and_then(|v| v.as_object().map(|o| o.keys().cloned().collect()))
        .unwrap_or_default();
    known.insert("starts".into());
    if matches!(method.algorithm, Algorithm::PGAS { .. }) {
        // Serialized only off their default; still legitimate keys.
        known.insert("binomial".into());
        known.insert("ancestor_sampling".into());
    }
    for (key, value) in raw_method {
        if known.contains(key) {
            continue;
        }
        // The keys the `[stages]` → `[method]` split retired name their
        // replacement rather than a bare "unknown key".
        let retired = match key.as_str() {
            "init" => Some(match value.as_str() {
                Some("survey_top_k") => "`init = \"survey_top_k\"` was removed: a survey \
                     landscape is not a posterior. Use `starts = \"from_prior\"`, or run a \
                     short fit and write `starts = { from_posterior = \"@handle\" }`."
                    .to_string(),
                Some(rule) => format!("`init` is now `starts`: write `starts = \"{rule}\"`."),
                None => "`init` is now `starts`.".to_string(),
            }),
            "init_mle" => Some(
                "`init_mle` has no replacement key: a warm start from a stored fit is \
                 `starts = { from_posterior = \"@handle\" }` (one draw per chain, keeps R̂ \
                 meaningful) or `starts = { from_mle = \"@handle\" }` (every chain at one \
                 point, R̂ not assessed)."
                    .to_string(),
            ),
            "survey_path" | "survey_top_k_n" => Some(
                "`survey_top_k` starts were removed with their `survey_path` / \
                 `survey_top_k_n` companions: a survey landscape is not a posterior. Use \
                 `starts = \"from_prior\"`, or a `from_posterior` from a short run."
                    .to_string(),
            ),
            _ => None,
        };
        if let Some(why) = retired {
            return Err(format!("{why} See `camdl docs fit-toml`."));
        }
        let mut allowed: Vec<&String> = known.iter().collect();
        allowed.sort();
        return Err(format!(
            "unknown key `{key}` in [method] (algorithm = \"{}\").\n  \
             allowed keys: {}\n  \
             A typo on an optional method key is otherwise silently ignored \
             (serde cannot deny unknown fields on the tagged `Algorithm` enum) — gh#241.",
            method.algorithm.method_name(),
            allowed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
        ));
    }
    Ok(())
}

impl FitConfig {
    /// Parse a fit.toml string. Runs the migration detectors — the
    /// `[stages]` rejection, the removed `fit_starts` / `condition_from` /
    /// `[config].backend` keys — before handing the string to the
    /// strongly-typed deserializer, so each names its replacement instead of
    /// surfacing as a bare "unknown field".
    pub fn from_toml_str(contents: &str) -> Result<Self, String> {
        Self::from_toml_str_named(contents, "fit.toml")
    }

    /// [`from_toml_str`](Self::from_toml_str) with the file's display name,
    /// which the `[stages]` rewrite quotes in its `camdl fit run …` line.
    pub fn from_toml_str_named(contents: &str, file_name: &str) -> Result<Self, String> {
        detect_legacy_stages(contents, file_name)?;
        detect_removed_fit_starts(contents)?;
        detect_relocated_config_backend(contents)?;
        detect_removed_condition_from(contents)?;
        let wire: FitConfigWire = toml::from_str(contents)
            .map_err(|e| format!("parse error: {}", e))?;
        // gh#241 (C3): catch typo'd method keys serde silently drops.
        if let Some(method) = &wire.method {
            validate_method_keys(contents, method)?;
        }
        Ok(wire.split())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        Self::load_anchored_at(std::path::Path::new(path), std::path::Path::new(path))
    }

    /// Load the config stored at `read_path`, but resolve its relative paths as
    /// if the config file lived at `anchor_toml`.
    ///
    /// The two coincide for a config loaded from where the user wrote it — that
    /// is [`load`](Self::load). They diverge for the copy archived inside a fit
    /// segment (`fit.toml.original`): its relative `[model] camdl` /
    /// `[data.observations]` paths were written against the *original* fit.toml
    /// directory, so anchoring them at the archive's own location resolves
    /// `../data/x.tsv` to `results/fits/data/x.tsv` — a path that never existed
    /// (gh#652). The segment's sidecar records `fit_toml_path`, so the original
    /// directory is known even when that file has moved or changed; pass it here
    /// and every path resolves exactly as it did at fit time.
    pub fn load_anchored_at(
        read_path: &std::path::Path,
        anchor_toml: &std::path::Path,
    ) -> Result<Self, String> {
        let path = read_path.to_string_lossy();
        let contents = std::fs::read_to_string(read_path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        let mut config = FitConfig::from_toml_str_named(&contents, &path)
            .map_err(|e| format!("error in {}:\n{}", path, e))?;

        // gh#307: warn (do not error) on absolute file references — checked on
        // the as-written paths, before the relative-path resolution below turns
        // every relative path absolute.
        for w in config.problem.absolute_path_warnings() {
            eprintln!("{w}");
        }

        config.problem.anchor_paths_at(anchor_toml);
        // A `from_params` path, or a file-shaped `from_mle` / `from_posterior`
        // source, is a path written in the file and anchors there too; a
        // `@label` or hash handle is not a path and is left alone.
        if let Some(method) = &mut config.inference.method {
            if let Some(starts) = &mut method.starts {
                starts.anchor_paths_at(anchor_toml);
            }
        }
        Ok(config)
    }

    /// gh#439 A2: does the method read the WrtPop state-Jacobian
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
        self.inference.method.as_ref().is_some_and(|m| {
            m.algorithm.method_kind() == FitAlgorithm::Nuts
                && m.algorithm.backend() == InferenceBackend::Ode
        })
    }

    /// A note for a point-started multi-chain sampler (`starts = "single"`,
    /// `from_mle`, `from_params` with `chains > 1`): every chain then starts at
    /// the same point, so the between-chain R̂ cannot tell whether the
    /// posterior was explored, and `fit summary` will report it as not
    /// assessed. Not an error — writing the point rule is the acknowledgement
    /// (proposal §3.4) — but said once at startup so the choice is deliberate.
    /// Optimizer-only methods report no R̂ and get no note.
    pub fn point_start_multichain_note(&self) -> Option<String> {
        let method = self.inference.method.as_ref()?;
        if !method.algorithm.requires_priors() || method.algorithm.chains() < 2 {
            return None;
        }
        let starts = method.starts.as_ref()?;
        if !starts.is_point() {
            return None;
        }
        Some(format!(
            "`starts = {}` puts all {} chains at one point, so the between-chain R̂ \
             cannot say whether the posterior was explored; `fit summary` will report \
             R̂ as not assessed. Use a spread rule (`from_prior`, \
             `{{ from_posterior = \"@handle\" }}`, `uniform_unconstrained`, `lhs`) for \
             an informative R̂.",
            starts.spelled(),
            method.algorithm.chains()
        ))
    }

    /// The whole-file check: the problem's own rules, the seed list, and the
    /// method-dependent cells (`ic_free`, burn-in, the algorithm/backend pair).
    ///
    /// `init_law` is the one MODEL fact this otherwise config-only check needs:
    /// whether `init { }` DRAWS a compartment from a law. It decides the
    /// `ic_free` × `pfilter`/`pmmh` cells, because under the bootstrap particle
    /// filter a declared law is the whole source of the swarm's spread at t=0
    /// (gh#732). Derive it at the call site with
    /// `model.initial_conditions.iter().any(|(_, s)| s.is_law())`.
    pub fn validate(
        &self,
        model_params: &[String],
        init_law: super::methods::InitLaw,
    ) -> Result<(), String> {
        self.problem.validate(model_params)?;
        self.inference.validate()?;

        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        let algorithm = &method.algorithm;

        // (algorithm, backend) must be a supported pair. Method registry
        // is the single source of truth (see fit/methods.rs); errors name
        // the right alternative when the user picked an incoherent combo.
        if let Err(msg) = super::methods::validate_combo(algorithm.method_kind(), algorithm.backend()) {
            return Err(format!("[method]: {}", msg));
        }

        // ic_free / conditioning support check (F1). `ic_free = true` is
        // honored only by the cells that BOTH drop y₁ from the accumulated
        // loglik AND give the swarm spread at t=0. PGAS, the ODE-MLE
        // optimizers, and correlated PMMH score every obs unconditionally;
        // `pfilter` / plain `pmmh` do condition, but only have the spread when
        // the MODEL declares an `init { }` law — hence `init_law` here.
        // Running ic_free without either property would silently compute the
        // UNCONDITIONAL likelihood while the banner claims conditioning.
        if self.problem.ic_free.unwrap_or(false) {
            let correlated = matches!(algorithm, Algorithm::PMMH { rho: Some(_), .. });
            if let Err(msg) =
                super::methods::validate_ic_free(algorithm.method_kind(), correlated, init_law)
            {
                return Err(format!("[method]: {}", msg));
            }
        }

        // `perturb_only_at_t0` and `rw_sd` are IF2 schedule knobs living in
        // the problem half. Under one method per file they are inert for a
        // non-IF2 method and the loader says nothing, so that a problem's IF2
        // comparator file and its PGAS file differ only in `[method]` and share
        // a fit-level hash (proposal §5; relocating them is a follow-up).

        // IF2 requires at least one iteration — zero iterations would leave
        // `iterations` empty and cause `last().unwrap()` to panic in
        // `run_if2`. Catch it here so the user gets a config error, not a crash.
        if let Algorithm::IF2 { iterations, .. } = algorithm {
            if *iterations == 0 {
                return Err(
                    "[method]: iterations must be ≥ 1 (got 0). IF2 needs at least one \
                     filtering pass to produce a parameter estimate."
                        .to_string(),
                );
            }
        }

        // gh#347: a sampler retains only its post-burn-in draws. A
        // burn_in ≥ the run length discards EVERY sample, so the fit produces
        // no posterior no matter how well the chain mixes — and the reported
        // post-burn acceptance rate degenerates to 0/0, which reads as a
        // misleading "0% acceptance". Reject at config validation rather than
        // burn compute for an empty result. (The `profile` path already
        // enforces the same steps-vs-burn_in invariant.)
        let burn = match algorithm {
            Algorithm::Mh { iterations, burn_in, .. }
            | Algorithm::PMMH { iterations, burn_in, .. } => Some((
                *iterations,
                burn_in.unwrap_or(super::pmmh::DEFAULT_BURN_IN),
                "iterations",
                super::pmmh::DEFAULT_BURN_IN,
            )),
            Algorithm::PGAS { sweeps, burn_in, .. } => Some((
                *sweeps,
                burn_in.unwrap_or(super::pgas::DEFAULT_BURN_IN),
                "sweeps",
                super::pgas::DEFAULT_BURN_IN,
            )),
            _ => None,
        };
        if let Some((n_steps, burn_in, len_field, default_burn)) = burn {
            if burn_in >= n_steps {
                return Err(format!(
                    "[method]: burn_in ({burn_in}) ≥ {len_field} ({n_steps}) — \
                     every sample is discarded as burn-in, so the fit retains no \
                     posterior draws (and the post-burn acceptance rate degenerates \
                     to 0%). Reduce burn_in or raise {len_field}. \
                     (burn_in defaults to {default_burn} when unset.)"));
            }
        }

        // Algorithm-aware warning: a non-IF2 method does not honour simplex
        // groups.
        if !self.problem.simplex_groups.is_empty() && !matches!(algorithm, Algorithm::IF2 { .. }) {
            let use_color = std::io::IsTerminal::is_terminal(&std::io::stderr())
                && std::env::var("NO_COLOR").is_err();
            let tag = if use_color { "\x1b[33mwarning:\x1b[0m" } else { "warning:" };
            eprintln!("{} fit declares simplex_groups, \
                but the `{}` method does not currently honour the \
                simplex constraint — members will be perturbed \
                independently and rely on the model to enforce sum = 1 \
                indirectly.", tag, algorithm.method_name());
        }

        // Bayesian prior presence is checked separately by
        // `validate_priors_present(&ir_priors)`, which needs the model IR
        // in scope to honor the gh#73 precedence fallback. validate()
        // itself only needs parameter names, so the prior check is
        // factored out — production callers do both.
        Ok(())
    }

    /// gh#75: Validate that every estimated parameter has a prior available
    /// from at least one source — either this fit toml's
    /// `[estimate.<name>.prior]` block, or the model IR's `~` syntax —
    /// when the method is Bayesian (PGAS / PMMH / MH / NUTS).
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
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        if !method.algorithm.requires_priors() {
            return Ok(());
        }
        // "Missing" = no fit-toml prior of any kind (regular dist
        // *or* explicit flat) AND no IR `~` prior.
        let missing_priors: Vec<&str> = self.problem.estimate.iter()
            .filter(|(name, spec)| {
                spec.prior.is_none() && !ir_prior_params.contains(name.as_str())
            })
            .map(|(name, _)| name.as_str())
            .collect();
        if missing_priors.is_empty() {
            return Ok(());
        }
        // Two-column reason table: parameter | why it's missing.
        // Width derived from the affected set so the output
        // stays compact when 1–3 params are missing.
        let name_width = missing_priors.iter()
            .map(|n| n.len()).max().unwrap_or(0)
            .max("parameter".len());
        let mut msg = String::new();
        msg.push_str(&format!(
            "[method] (algorithm = \"{}\") has parameters with no resolved prior:\n\n",
            method.algorithm.method_name(),
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
        Err(msg)
    }
}

impl Inference {
    /// The seed list: non-empty, no duplicates (they would collide on
    /// per-cell provenance hashes).
    pub fn validate(&self) -> Result<(), String> {
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
        Ok(())
    }
}

impl ChainStarts {
    /// Anchor a path-shaped source at the fit.toml that wrote it. A `@label`
    /// or a hash prefix is not a path and is left alone; a `.toml` / `.tsv`
    /// file or anything with a directory separator is.
    fn anchor_paths_at(&mut self, toml_path: &std::path::Path) {
        let anchor = |s: &mut String| {
            let looks_like_path = !s.starts_with('@')
                && (s.contains('/')
                    || s.contains('\\')
                    || s.ends_with(".toml")
                    || s.ends_with(".tsv"));
            if looks_like_path {
                *s = crate::util::resolve_relative_to_toml(toml_path, s);
            }
        };
        match self {
            ChainStarts::Spread(Spread::FromPosterior { source })
            | ChainStarts::Point(Point::FromMle { source }) => anchor(&mut source.0),
            ChainStarts::Point(Point::FromParams { path }) => {
                let mut s = path.to_string_lossy().into_owned();
                *path = PathBuf::from(crate::util::resolve_relative_to_toml(toml_path, &s));
                let _ = &mut s;
            }
            _ => {}
        }
    }
}

impl Problem {
    /// The entry point for every non-fit reader (`simulate --draws prior
    /// --fit`, `pfilter --fit`, `survey --fit`, `profile --fit`): the file's
    /// problem half, with the inference half discarded. A file with no
    /// `[method]` at all loads here.
    pub fn load(path: &str) -> Result<Self, String> {
        FitConfig::load(path).map(|c| c.problem)
    }

    /// Portability lint (gh#307): one warning line per file reference in the
    /// fit config that is written as an ABSOLUTE path. Absolute paths bake one
    /// machine's filesystem layout into the config, breaking sharing and
    /// reproducibility (the content-addressable design) — the fit-config
    /// counterpart of the compiler's W104 on model-file paths. Covered
    /// surfaces: `[model] camdl`, `output_dir`, the wide-TSV `[data] file`, and
    /// every `[data.observations]` stream source.
    ///
    /// Checked on the AS-WRITTEN strings, so it must run BEFORE
    /// [`FitConfig::load`] resolves relative paths against the fit.toml
    /// directory (which rewrites every relative path to an absolute one,
    /// erasing the distinction). Pure and side-effect-free so it is
    /// unit-testable; the loader prints the returned lines to stderr.
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

    /// Resolve toml-relative paths against the toml's directory
    /// (Cargo / pyproject convention). Closes GH #22: pre-fix, paths
    /// inside the toml were resolved against the user's CWD, which
    /// broke any invocation pattern other than "always cd into the
    /// toml's directory before camdl fit run". Post-fix, every
    /// downstream consumer (the fit-level digest, the runner's data loaders)
    /// sees absolute paths regardless of where the binary was invoked from.
    /// Absolute paths in the toml pass through unchanged.
    fn anchor_paths_at(&mut self, toml_path: &std::path::Path) {
        self.model.camdl = crate::util::resolve_relative_to_toml(toml_path, &self.model.camdl);
        // gh#507: `output_dir` anchors here too. It is the one path written in
        // the fit.toml that used to resolve against the process CWD instead,
        // so a single `../` could not be correct for both the inputs and the
        // output — `output_dir = "../data/runs"` in a fit.toml one level down
        // wrote the whole run tree to a SIBLING OF THE REPOSITORY, silently,
        // with the fit reporting success and the announced path (echoed
        // as-written) consistent with either reading.
        //
        // The rule this restores: a path written IN a file anchors at that
        // file; what does not come from the file anchors at the CWD, which is
        // the frame it was typed in — the `results/` default when no
        // `output_dir` is declared, and `CAMDL_OUTPUT_DIR`.
        //
        // gh#531: this used to cite `--output-dir` as the command-line case.
        // `fit run` has no such flag — `FitRunArgs` carries no `output_dir`,
        // and both call sites pass `output_root(None, config.output_dir…)`.
        // The flag exists on `simulate` and `batch run`, neither of which
        // loads a fit.toml, so it documented a precedence layer unreachable
        // from here.
        if let Some(dir) = &mut self.output_dir {
            *dir = crate::util::resolve_relative_to_toml(toml_path, dir);
        }
        if let Some(data) = &mut self.data {
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

    /// Does `fit run` generate its data? Only when `[synthetic]` is present
    /// and `[data]` is not: when both are, the real data are fitted and
    /// `[synthetic]` is `fit recovery`'s truth (proposal §8, item 19).
    pub fn is_synthetic_fit(&self) -> bool {
        self.data.is_none() && self.synthetic.is_some()
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

    /// The problem's own rules: a data source, the estimate/fixed partition
    /// against the model's parameters, bounds, simplex groups, and the
    /// scenario / holdout exclusions. Needs only the model's parameter names.
    pub fn validate(&self, model_params: &[String]) -> Result<(), String> {
        // Data source: at least one of [data] / [synthetic]. Both may be
        // present — `fit run` fits the real data and `fit recovery` reads the
        // design from [data] and the truth from [synthetic] (proposal §8, 19).
        if self.data.is_none() && self.synthetic.is_none() {
            return Err(
                "fit config has neither [data] nor [synthetic] — one must be supplied.".to_string());
        }

        // Validate synthetic spec if present.
        if let Some(syn) = &self.synthetic {
            syn.validate()?;
        }

        // Validate [data] block: exactly one of `file` / `observations`.
        if let Some(data) = &self.data {
            data.validate()?;
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

    /// Validate `[[simplex_groups]]` entries against `[estimate]`.
    /// Rules:
    ///  - `params.len() >= 2` (single-member simplex is degenerate)
    ///  - Every member appears in `[estimate]`
    ///  - No member appears in more than one simplex group
    ///  - No member is `perturb_only_at_t0 = true` (the simplex transform
    ///    owns the initial perturbation; the two would conflict)
    ///  - Each member's bounds lower must be ≥ 0 (members are non-negative)
    ///
    /// The algorithm-aware warning (a non-IF2 method does not honour the
    /// constraint) lives in [`FitConfig::validate`], which sees the method.
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
                if spec.perturb_only_at_t0 {
                    return Err(format!(
                        "simplex_groups[{}]: member '{}' has \
                         perturb_only_at_t0 = true. The simplex transform \
                         owns the initial perturbation; the two would \
                         conflict. Drop perturb_only_at_t0 on simplex \
                         members and rely on the simplex's barycentric \
                         perturbation for spread.",
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
    use crate::fit::methods::InitLaw;
    use crate::fit::starts::Handle;

    fn parse(toml_str: &str) -> Result<FitConfig, String> {
        // Route every test fixture through the same migration detectors the
        // production `load` path uses — so an in-source fixture that still
        // carries a `[stages]` table or a removed key fails loudly here
        // rather than parsing under the new schema by accident.
        FitConfig::from_toml_str(toml_str)
    }

    /// The one `[method]` a fixture declares.
    fn method(cfg: &FitConfig) -> Method {
        cfg.inference.method.clone().expect("fixture declares [method]")
    }

    fn algo(cfg: &FitConfig) -> Algorithm {
        method(cfg).algorithm
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.problem.estimate.len(), 4);
        assert_eq!(config.problem.fixed.values.len(), 2);
        assert!(config.inference.method.is_some());
        assert!(method(&config).starts.is_none(), "an omitted `starts` is unresolved until fit run");

        match &algo(&config) {
            Algorithm::IF2 { chains, particles, iterations, cooling, .. } => {
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
    fn cfg_with_paths(camdl: &str, obs: &str, output_dir: &str) -> FitConfig {
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

[method]
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

    // ── gh#514: a CLI chain-start override must re-key the method ───────

    /// A `pgas` method, for the sampler/output flags that only exist there.
    fn pgas_method() -> Method {
        let cfg = parse(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }

[fixed]
N0 = 1000000

[method]
algorithm  = "pgas"
backend    = "chain_binomial"
chains     = 2
particles  = 100
sweeps     = 10
"#).unwrap();
        method(&cfg)
    }

    /// A `pmmh` method, for `--no-adapt` / `--adapt-start` / `--rho`.
    fn pmmh_method() -> Method {
        let cfg = parse(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }

[fixed]
N0 = 1000000

[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 2
particles  = 100
iterations = 10
"#).unwrap();
        method(&cfg)
    }

    /// A `pfilter` method, for the record-flag overrides. Both record fields
    /// are declared false so the one-way CLI override to true is observable.
    fn pfilter_method() -> Method {
        let cfg = parse(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[method]
algorithm = "pfilter"
backend   = "chain_binomial"
particles = 100
record_ancestry     = false
record_prequential  = false
"#).unwrap();
        method(&cfg)
    }

    /// An `mh` method, for the gh#726 dt-check refusal: Mh stores the
    /// dt-check result in the leaf but has no dt_check TOML field, so the
    /// CLI flags cannot reach its identity and must be refused.
    fn mh_method() -> Method {
        let cfg = parse(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }

[fixed]
N0 = 1000000

[method]
algorithm  = "mh"
backend    = "ode"
chains     = 2
iterations = 10
"#).unwrap();
        method(&cfg)
    }

    /// An `nl-sbplx` method, for the gate and dt-check overrides on the
    /// NloptStageConfig payload (gh#726).
    fn nl_sbplx_method() -> Method {
        let cfg = parse(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[method]
algorithm = "nl-sbplx"
backend   = "ode"
chains    = 2
"#).unwrap();
        method(&cfg)
    }

    /// An `if2` method with `starts` spelled, so its identity is concrete.
    fn scout_method(starts: &str) -> Method {
        let cfg = parse(&format!(r#"
[model]
camdl = "m.camdl"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
N0 = 1000000

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 4
particles  = 100
iterations = 10
cooling    = 0.7
starts     = "{starts}"
"#)).unwrap();
        method(&cfg)
    }

    /// The count-in-the-key rule: anything that changes where the chains
    /// start changes the stored output, so it must change the method's
    /// identity. Before gh#514 the CLI overrides were applied after the CAS
    /// claim, so two runs differing only in the start rule collided and the
    /// second was served the first's result. `fit run --starts` writes the
    /// rule into the in-memory method before the identity is taken — this
    /// pins that the write reaches the payload.
    #[test]
    fn cli_starts_override_changes_the_method_identity() {
        let declared = scout_method("single");
        let mut overridden = scout_method("single");
        overridden.starts = Some(ChainStarts::Spread(Spread::Lhs));

        assert_ne!(declared.identity_payload(), overridden.identity_payload(),
            "a method run under `--starts lhs` must not share an identity with \
             the same method run under its declared `starts = single` — that \
             collision is gh#514, and it silently returns the other run's \
             result");

        // And the override must land on the value the run actually uses, not
        // merely perturb the hash.
        assert_eq!(overridden.identity_payload(), scout_method("lhs").identity_payload(),
            "`--starts lhs` must key identically to `starts = \"lhs\"` written in \
             the toml — they are the same fit");
    }

    /// gh#540: EVERY sampler/output flag must move the stage identity, not
    /// just the five chain-start ones gh#514 covered. Each of these was
    /// applied to the `*StageOpts` struct at the dispatch site, AFTER the
    /// identity was taken from the stage — so `--n-trajectories 500` was
    /// served the 200-trajectory leaf, and `--no-nuts` was served a posterior
    /// drawn by NUTS.
    ///
    /// Table-driven so a flag added to `CliStageOverrides` without a case here
    /// is conspicuous. A per-flag test would let the fourteenth flag slip in
    /// silently, which is exactly how the thirteen got here.
    ///
    /// `--n-trajectories` is deliberately NOT in this table:
    /// `identity_payload` omits it on purpose (see
    /// `pgas_identity_payload_omits_n_trajectories`) because it shapes output
    /// rather than the statistical fit, and it reaches the key by the separate
    /// `cas_n_trajectories` route that `StageConfig` folds in. It gets its own
    /// test below rather than a row here that would assert the wrong thing.
    /// `--starts` is not a `CliStageOverrides` field at all — it writes
    /// `Method::starts` directly; `cli_starts_override_changes_the_method_identity`
    /// pins it.
    #[test]
    fn every_cli_sampler_flag_changes_the_stage_identity() {
        let cases: Vec<(&str, Method, CliStageOverrides)> = vec![
            ("--tempering", pgas_method(),
             CliStageOverrides { tempering: Some(vec![1.0, 0.5]), ..Default::default() }),
            ("--max-tree-depth", pgas_method(),
             CliStageOverrides { max_tree_depth: Some(7), ..Default::default() }),
            ("--trajectory-warmup", pgas_method(),
             CliStageOverrides { trajectory_warmup: Some(3), ..Default::default() }),
            ("--csmc-sweeps-per-nuts", pgas_method(),
             CliStageOverrides { csmc_sweeps_per_nuts: Some(4), ..Default::default() }),
            ("--diagonal-mass", pgas_method(),
             CliStageOverrides { diagonal_mass: true, ..Default::default() }),
            ("--no-nuts", pgas_method(),
             CliStageOverrides { no_nuts: true, ..Default::default() }),
            ("--no-adapt", pmmh_method(),
             CliStageOverrides { no_adapt: true, ..Default::default() }),
            ("--adapt-start", pmmh_method(),
             CliStageOverrides { adapt_start: Some(42), ..Default::default() }),
            ("--rho", pmmh_method(),
             CliStageOverrides { rho: Some(0.9), ..Default::default() }),
            // The 2026-08-23 batch: found applied at the dispatch site,
            // after the CAS claim, exactly like the thirteen above.
            ("--cooling-target-iters", scout_method("single"),
             CliStageOverrides { cooling_target_iters: Some(20), ..Default::default() }),
            ("--decibans-thresh", scout_method("single"),
             CliStageOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
            ("--no-dt-check", scout_method("single"),
             CliStageOverrides { no_dt_check: true, ..Default::default() }),
            ("--dt-check-halvings", scout_method("single"),
             CliStageOverrides { dt_check_halvings: Some(3), ..Default::default() }),
            ("--record-ancestry", pfilter_method(),
             CliStageOverrides { record_ancestry: true, ..Default::default() }),
            ("--record-prequential", pfilter_method(),
             CliStageOverrides { record_prequential: true, ..Default::default() }),
            // gh#726: mh and nl-* store the dt-check result in the leaf;
            // their dt_check TOML field exists precisely so these flags
            // can reach the identity.
            ("--no-dt-check (mh)", mh_method(),
             CliStageOverrides { no_dt_check: true, ..Default::default() }),
            ("--dt-check-halvings (mh)", mh_method(),
             CliStageOverrides { dt_check_halvings: Some(3), ..Default::default() }),
            ("--no-dt-check (nl-sbplx)", nl_sbplx_method(),
             CliStageOverrides { no_dt_check: true, ..Default::default() }),
            // gh#730: --dt-check-strict selects the threshold that is STORED
            // in fit_state.toml.dt_check (verdict + threshold_nats +
            // threshold_se_aware_nats + notes all derive from it), so it was
            // never the leaf-byte-neutral abort policy it was treated as.
            ("--dt-check-strict (if2)", scout_method("single"),
             CliStageOverrides { dt_check_strict: true, ..Default::default() }),
            ("--dt-check-strict (mh)", mh_method(),
             CliStageOverrides { dt_check_strict: true, ..Default::default() }),
            ("--decibans-thresh (nl-sbplx)", nl_sbplx_method(),
             CliStageOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
        ];
        for (flag, base, cli) in cases {
            let mut overridden = base.clone();
            overridden.algorithm.apply_cli_overrides(&cli);
            assert_ne!(
                base.identity_payload(), overridden.identity_payload(),
                "{flag} changes what the stage computes or stores, so it must \
                 change the stage identity. Sharing an identity means the run \
                 is served the OTHER setting's stored result (gh#540)."
            );
        }
    }

    /// gh#730: `--dt-check-strict` resolves to a CONCRETE threshold at
    /// override time, so it is keyed like any other stored setting — and a
    /// stage that declares its own `threshold_nats` is unaffected, exactly as
    /// before (the flag was already inert there).
    #[test]
    fn dt_check_strict_resolves_to_the_stored_threshold() {
        use crate::fit::dt_check::default_threshold_for_backend;
        use crate::run_meta::InferenceBackend;

        let mut if2 = scout_method("single");
        if2.algorithm.apply_cli_overrides(&CliStageOverrides {
            dt_check_strict: true, ..Default::default() });
        match &if2.algorithm {
            Algorithm::IF2 { dt_check, backend, .. } => assert_eq!(
                dt_check.threshold_nats,
                Some(default_threshold_for_backend(*backend, true)),
                "strict must be resolved into the stored threshold"),
            other => panic!("expected IF2, got {}", other.method_name()),
        }
        // chain_binomial strict is 0.5 nats (vs 2.0 routine) — pin the value
        // so a change to the constant surfaces here rather than in a fit.
        assert_eq!(default_threshold_for_backend(InferenceBackend::ChainBinomial, true), 0.5);

        // A TOML-declared threshold wins: the flag stays inert there.
        let mut declared = scout_method("single");
        if let Algorithm::IF2 { dt_check, .. } = &mut declared.algorithm {
            dt_check.threshold_nats = Some(1.25);
        }
        let before = declared.clone();
        declared.algorithm.apply_cli_overrides(&CliStageOverrides {
            dt_check_strict: true, ..Default::default() });
        assert_eq!(before.identity_payload(), declared.identity_payload(),
            "a stage that declares threshold_nats must be unmoved by the flag");
    }

    /// The subtractive payload is include-by-default: every stage field is in
    /// the identity unless deliberately subtracted. The enumerated arms it
    /// replaced swallowed anything they forgot to list, which is the shape
    /// behind gh#514, gh#540 and the 2026-08-23 batch.
    #[test]
    fn identity_payload_includes_every_field_but_the_named_exclusions() {
        let mut with_starts = pgas_method();
        with_starts.starts = Some(ChainStarts::from_prior());
        let payload = with_starts.identity_payload();
        let obj = payload.as_object().expect("payload is an object");
        // The extension dimension and the separately-folded output count are
        // the ONLY omissions.
        assert!(!obj.contains_key("sweeps"),
            "the extension dimension stays out (folded by cas_target_length)");
        assert!(!obj.contains_key("n_trajectories"),
            "folded by cas_n_trajectories instead — hashing it here double-folds");
        // Everything else the stage carries is present, under its TOML spelling.
        for key in ["algorithm", "backend", "chains", "particles", "burn_in", "thin",
                    "tempering", "use_nuts", "dense_mass", "max_tree_depth",
                    "starts", "trajectory_warmup", "csmc_sweeps_per_nuts"] {
            assert!(obj.contains_key(key),
                "'{key}' must be in the stage identity; present keys: {:?}",
                obj.keys().collect::<Vec<_>>());
        }
        // And the point of the change: a knob the enumerated arm never listed
        // now re-keys. `loglik_eval` decides how the stage's stored MLE is
        // re-scored, and IF2 hashed it only because that arm full-serialized.
        let mut hot = scout_method("single");
        if let Algorithm::IF2 { loglik_eval, .. } = &mut hot.algorithm { loglik_eval.n_particles += 1; }
        assert_ne!(scout_method("single").identity_payload(), hot.identity_payload(),
            "a clean-eval knob that changes the stored loglik must re-key");
    }

    /// The extension dimension must stay OUT: a resumed run extends a base
    /// run, so the two share a prefix identity by design — the length reaches
    /// identity through `cas_target_length` instead.
    #[test]
    fn extension_dimension_stays_out_of_the_payload() {
        let base = pmmh_method();
        let mut longer = pmmh_method();
        if let Algorithm::PMMH { iterations, .. } = &mut longer.algorithm { *iterations *= 4; }
        assert_eq!(base.identity_payload(), longer.identity_payload(),
            "iterations is PMMH's extension dimension — it must not re-key the \
             payload, or --resume could never share a prefix identity");
        assert_ne!(base.algorithm.cas_target_length(), longer.algorithm.cas_target_length(),
            "…but it MUST reach identity through cas_target_length");
    }

    /// `--n-trajectories` is the sharpest case, because `cas.rs` documents it
    /// as folded count-in-the-key and it was not. Pinned separately from the
    /// table so the claim in that doc comment has a test under it.
    #[test]
    fn n_trajectories_is_count_in_the_key() {
        let base = pgas_method();
        let mut more = pgas_method();
        more.algorithm.apply_cli_overrides(&CliStageOverrides {
            n_trajectories: Some(500), ..Default::default() });
        assert_ne!(base.algorithm.cas_n_trajectories(), more.algorithm.cas_n_trajectories(),
            "`--n-trajectories` must reach `cas_n_trajectories` — it reads the \
             stage, and the flag used to be applied to the opts struct instead, \
             so a 500-trajectory request was served the 200-trajectory leaf");
        assert_eq!(more.algorithm.cas_n_trajectories(), 500);
    }

    /// AS-off changes the draws, so it is count-in-the-key; AS-on (the
    /// default) keeps the pre-field payload bytes, so nothing re-keys.
    #[test]
    fn ancestor_sampling_off_is_in_the_stage_identity() {
        // Disabling AS changes the sampled draws, so it must re-key
        // (count-in-the-key discipline)…
        let on = pgas_method();
        let mut off = pgas_method();
        if let Algorithm::PGAS { ref mut ancestor_sampling, .. } = off.algorithm {
            *ancestor_sampling = false;
        }
        assert_ne!(on.identity_payload(), off.identity_payload(),
            "an AS-off fit must not be served from (or stored over) an AS-on leaf");
        // …while the DEFAULT serializes to the pre-field bytes, so no stored
        // leaf is orphaned and no in-flight --resume breaks. The byte golden
        // (`identity_payload_is_byte_stable_against_recompiles`) is the other
        // half of this assertion.
        assert!(!on.identity_payload().as_object().unwrap()
                    .contains_key("ancestor_sampling"),
            "the default must keep the payload byte-identical to the pre-field format");
        assert!(off.identity_payload().as_object().unwrap()
                    .contains_key("ancestor_sampling"),
            "the non-default must serialize, or it could not re-key");
    }

    /// `ancestor_sampling = false` is the TOML spelling; absent means on.
    #[test]
    fn ancestor_sampling_parses_from_stage_toml() {
        let toml_src = r#"
            algorithm = "pgas"
            backend = "chain_binomial"
            chains = 2
            particles = 10
            sweeps = 5
            ancestor_sampling = false
        "#;
        let stage: Algorithm = toml::from_str(toml_src).expect("stage parses");
        match stage {
            Algorithm::PGAS { ancestor_sampling, .. } => assert!(!ancestor_sampling),
            other => panic!("expected PGAS, got {}", other.method_name()),
        }
        let default_src = toml_src.replace("ancestor_sampling = false", "");
        match toml::from_str::<Algorithm>(&default_src).expect("stage parses") {
            Algorithm::PGAS { ancestor_sampling, .. } => assert!(ancestor_sampling,
                "an absent field must mean ancestor sampling ON"),
            other => panic!("expected PGAS, got {}", other.method_name()),
        }
    }

    /// `--no-ancestor-sampling` rides the same seam as `--no-nuts` and is
    /// equally keyed.
    #[test]
    fn cli_no_ancestor_sampling_overrides_and_rekeys() {
        let base = pgas_method();
        let mut overridden = pgas_method();
        overridden.algorithm.apply_cli_overrides(&CliStageOverrides {
            no_ancestor_sampling: true, ..Default::default() });
        match &overridden.algorithm {
            Algorithm::PGAS { ancestor_sampling, .. } => assert!(!ancestor_sampling),
            other => panic!("expected PGAS, got {}", other.method_name()),
        }
        assert_ne!(base.identity_payload(), overridden.identity_payload());
    }

    /// The other half, and the one that protects existing users: a run with
    /// no CLI overrides must key exactly as it did before.
    #[test]
    fn no_cli_override_leaves_the_stage_identity_untouched() {
        let declared = scout_method("uniform");
        let mut untouched = scout_method("uniform");
        untouched.algorithm.apply_cli_overrides(&CliStageOverrides::default());
        assert_eq!(declared.identity_payload(), untouched.identity_payload());
    }

    #[test]
    fn absolute_path_warnings_flags_all_surfaces() {
        let cfg = cfg_with_paths(
            "/abs/models/sir.camdl",
            "/abs/data/cases.tsv",
            "/abs/out",
        );
        let warnings = cfg.problem.absolute_path_warnings();
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

    /// gh#507: every path WRITTEN IN the fit.toml anchors at the fit.toml,
    /// `output_dir` included. Before this, `[model] camdl` and the data
    /// streams anchored at the toml while `output_dir` anchored at the
    /// process CWD, so one `../` could not be correct for both — a config
    /// with `camdl = "m.camdl"` and `output_dir = "../data/runs"` wrote its
    /// entire run tree to a sibling of the repository, silently, with the
    /// fit reporting success.
    #[test]
    fn load_anchors_output_dir_at_the_toml_like_every_other_path() {
        let dir = std::env::temp_dir().join("camdl_gh507_anchor");
        std::fs::create_dir_all(&dir).unwrap();
        let toml_path = dir.join("fit.toml");
        std::fs::write(&toml_path, r#"
output_dir = "../runs"

[model]
camdl = "sir.camdl"

[data.observations]
weekly_cases = "../data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 10
cooling = 0.7
"#).unwrap();

        let cfg = FitConfig::load(toml_path.to_str().unwrap()).unwrap();
        let anchor = dir.to_string_lossy();

        // The two that already anchored correctly — the control, so this
        // test cannot pass by everything being left alone.
        assert!(cfg.problem.model.camdl.starts_with(&*anchor),
            "[model] camdl must anchor at the toml: {}", cfg.problem.model.camdl);
        let obs = &cfg.problem.data.as_ref().unwrap().observations["weekly_cases"];
        assert!(obs.starts_with(&*anchor),
            "[data.observations] must anchor at the toml: {obs}");

        // The one that did not.
        let out = cfg.problem.output_dir.as_deref().unwrap();
        assert!(out.starts_with(&*anchor),
            "output_dir must anchor at the toml, not the CWD: {out}");
        assert!(std::path::Path::new(out).is_absolute(),
            "a resolved output_dir is absolute, so nothing downstream can \
             re-anchor it at the CWD: {out}");
    }

    /// An absolute `output_dir` is the user saying exactly where they want
    /// it; anchoring must not touch it (it already warns, via gh#307).
    #[test]
    fn load_passes_an_absolute_output_dir_through_unchanged() {
        let dir = std::env::temp_dir().join("camdl_gh507_abs");
        std::fs::create_dir_all(&dir).unwrap();
        let toml_path = dir.join("fit.toml");
        std::fs::write(&toml_path, r#"
output_dir = "/tmp/camdl_gh507_explicit"

[model]
camdl = "sir.camdl"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 10
cooling = 0.7
"#).unwrap();

        let cfg = FitConfig::load(toml_path.to_str().unwrap()).unwrap();
        assert_eq!(cfg.problem.output_dir.as_deref(), Some("/tmp/camdl_gh507_explicit"));
    }

    #[test]
    fn absolute_path_warnings_silent_on_relative() {
        let cfg = cfg_with_paths(
            "models/sir.camdl",
            "data/cases.tsv",
            "out",
        );
        assert!(
            cfg.problem.absolute_path_warnings().is_empty(),
            "relative paths are portable and must not warn: {:?}",
            cfg.problem.absolute_path_warnings()
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 100
iterations = 10
cooling = 0.7
"#,
        )
        .unwrap();
        let warnings = cfg.problem.absolute_path_warnings();
        assert_eq!(warnings.len(), 1, "got: {warnings:?}");
        assert!(warnings[0].contains("[data] file") && warnings[0].contains("/abs/data/wide.tsv"));
    }

    /// gh#241 C3: serde cannot apply `deny_unknown_fields` to the
    /// internally-tagged `Algorithm` enum, so a typo'd method key was silently
    /// dropped (neither applied nor reaching the method identity hash). A
    /// post-parse pass must reject it with an error naming the table and
    /// key; a valid method (including optional keys) still parses.
    #[test]
    fn method_rejects_unknown_keys() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\
                    [fixed]\nN0 = 1000000\n";

        let ok = format!(
            "{base}[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\n"
        );
        assert!(parse(&ok).is_ok(), "a valid IF2 method must parse");

        // A PGAS optional key (tempering) must be accepted, not falsely flagged.
        let ok_pgas = format!(
            "{base}[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\nsweeps = 10\ntempering = [1.0, 0.5]\n"
        );
        assert!(parse(&ok_pgas).is_ok(), "a valid PGAS method with optional keys must parse");

        // A typo on an OPTIONAL key is the real footgun: every required field
        // is present, so serde parses fine and *silently drops* the typo
        // (using the default), unlike a required-field typo which serde already
        // catches as "missing field". `cooling_target_iters` has a default.
        let bad = format!(
            "{base}[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\n\
             cooling_target_iterss = 40\n" // typo: cooling_target_iters
        );
        let err = parse(&bad).expect_err("a typo'd optional method key must be rejected");
        assert!(
            err.contains("cooling_target_iterss") && err.contains("[method]"),
            "error must name the unknown key and the table; got: {err}"
        );
        // `starts` is a method key on every algorithm, spelled or not.
        assert!(err.contains("starts"), "allowed keys must list `starts`; got: {err}");

        // The keys the split retired get their replacement, not a bare
        // "unknown key": `init` is `starts`, and a chained `init_mle` is one
        // of the two sourced rules — the author's decision, not a rename.
        let stale_init = format!(
            "{base}[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\ninit = \"lhs\"\n"
        );
        let err = parse(&stale_init).expect_err("`init` under [method] is not a key");
        assert!(err.contains("`init`") && err.contains("starts = \"lhs\""),
            "must hand back the `starts` spelling; got: {err}");
        let stale_chain = format!(
            "{base}[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\nsweeps = 10\ninit_mle = \"scout\"\n"
        );
        let err = parse(&stale_chain).expect_err("`init_mle` under [method] is not a key");
        assert!(err.contains("from_posterior") && err.contains("from_mle"),
            "must name both sourced rules; got: {err}");
    }

    /// A three-stage pipeline file — the shape the split retires — is refused
    /// at load with the §5 rewrite: the last stage becomes `[method]`, the
    /// others go in their own files, and each chained `init_mle` is handed
    /// back as the choice between the two sourced `starts` rules.
    #[test]
    fn legacy_pipeline_is_rejected_with_the_rewrite() {
        let err = parse(r#"
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
        "#).expect_err("a [stages] map is refused at load");

        // The last-declared stage is the one that becomes the file's method.
        assert!(err.starts_with("legacy table `[stages.evaluate]`"), "{err}");
        assert!(err.contains("rename to `[method]`") && err.contains("camdl fit run fit.toml"),
            "{err}");
        assert!(err.contains("put `[stages.mle]` and `[stages.posterior]` in its own file"),
            "{err}");
        // A stage-name `init_mle` becomes a `@label` handle; a directory stays
        // a path. Both are the author's decision between the two sourced rules.
        assert!(err.contains("starts = { from_posterior = \"@mle\" }"), "{err}");
        assert!(err.contains("starts = { from_mle = \"@mle\" }"), "{err}");
        assert!(err.contains("starts = { from_mle = \"output/fits/01_all_free/mle\" }"), "{err}");
        assert!(err.contains("R̂ uninformative"), "must say why init_mle has no rewrite: {err}");
        assert!(err.ends_with("See `camdl docs fit-toml`."), "{err}");
    }

    /// The same problem as one `[method]` with a sourced `starts`: the
    /// provenance, priors and rule all parse, and the handle is kept as
    /// written for resolution at fit time.
    #[test]
    fn parse_method_with_sourced_starts() {
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
starts = { from_posterior = "@mle" }
        "#).unwrap();

        let m = method(&config);
        assert!(matches!(m.algorithm, Algorithm::PGAS { .. }));
        assert_eq!(
            m.starts,
            Some(ChainStarts::Spread(Spread::FromPosterior { source: Handle("@mle".into()) }))
        );
        assert_eq!(m.starts.as_ref().unwrap().source().as_deref(), Some("@mle"));

        // All estimated params have priors (needed for PGAS)
        for (_, spec) in &config.problem.estimate {
            assert!(spec.prior.is_some());
        }

        assert!(config.problem.provenance.is_some());
        assert_eq!(config.problem.provenance.as_ref().unwrap().derived_from.as_deref(),
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 100
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.problem.fixed.from_file.as_deref(), Some("params/fixed.toml"));
        assert_eq!(config.problem.fixed.values["vacc_frac"], 0.80);
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let data = config.problem.data.as_ref().expect("[data] section required in test fixture");
        assert_eq!(data.holdout_after, Some(TimeSpecToml::Num(5474.0)));
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

[method]
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
        assert!(config.validate(&model_params, InitLaw::Absent).is_ok());
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

[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        // The partition/dag/etc. check passes — priors are not its concern.
        config.validate(&model_params, InitLaw::Absent).expect("validate() should not check priors");
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

[method]
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

[method]
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
        let beta_spec = config.problem.estimate.get("beta").expect("beta in estimate");
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let beta_prior = config.problem.estimate.get("beta").unwrap().prior.as_ref().unwrap();
        assert!(matches!(beta_prior, EstimatePriorSpec::Flat { .. }),
            "beta with `prior = {{ flat = {{}} }}` should deserialize to Flat, \
             got {:?}", beta_prior);
        let gamma_prior = config.problem.estimate.get("gamma").unwrap().prior.as_ref().unwrap();
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 100
        "#).unwrap();

        let a = config.problem.estimate.get("a").unwrap().prior.as_ref().unwrap();
        assert!(matches!(a, EstimatePriorSpec::UniformOverBounds { .. }),
            "`uniform = {{}}` should be UniformOverBounds, got {:?}", a);
        let b = config.problem.estimate.get("b").unwrap().prior.as_ref().unwrap();
        match b {
            EstimatePriorSpec::Dist(PriorDist::Uniform(p)) => {
                assert!((p.lower - 0.1).abs() < 1e-9);
                assert!((p.upper - 0.9).abs() < 1e-9);
            }
            other => panic!("`uniform = {{ lower, upper }}` should be Dist(Uniform), got {:?}", other),
        }
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
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
[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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
[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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
[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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
S0_y = { bounds = [0, 1], perturb_only_at_t0 = true }
S0_a = { bounds = [0, 1] }
S0_e = { bounds = [0, 1] }
beta = { bounds = [0.01, 2.0] }
[fixed]
N0 = 1000000
[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
        assert!(err.contains("perturb_only_at_t0 = true"), "got: {}", err);
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
[method]
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
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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
[method]
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
        config.validate(&model_params, InitLaw::Absent).expect("well-formed simplex must validate");
    }

    #[test]
    fn validate_data_and_synthetic_coexist() {
        // Both [data] and [synthetic] supplied: `fit run` fits the real data
        // and `fit recovery` reads the truth from [synthetic] (proposal §8,
        // item 19). Not a synthetic fit.
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        config.validate(&model_params, InitLaw::Absent)
            .expect("[data] beside [synthetic] is a real-data fit with a recorded truth");
        assert!(!config.problem.is_synthetic_fit(), "the real data are fitted");
        assert_eq!(config.problem.data_spec().unwrap().observations["weekly_cases"], "data/cases.tsv");
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let data = cfg.problem.data.as_ref().expect("[data] missing");
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let err = cfg.validate(&["beta".into(), "N0".into()], InitLaw::Absent).unwrap_err();
        assert!(err.contains("mutually exclusive"),
            "error should call out mutual exclusion: {}", err);
        assert!(err.contains("file") && err.contains("observations"),
            "error should name both forms: {}", err);
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        let err = cfg.validate(&["beta".into(), "N0".into()], InitLaw::Absent).unwrap_err();
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
                perturb_only_at_t0: false,
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
        // gh#37: the FitConfig-level wrapper forwards `&self.estimate`
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
        "#).unwrap();
        config.problem.expand_fixed_from_scenario(&model).unwrap();
        assert!(config.problem.fixed.from_scenario.is_none());
        assert!(!config.problem.fixed.values.contains_key("beta"),
            "estimated param carved out: {:?}", config.problem.fixed.values);
        assert_eq!(config.problem.fixed.values.get("gamma"), Some(&0.1));
        assert_eq!(config.problem.fixed.values.len(), 3);
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 0
cooling = 0.7
        "#).unwrap();
        let model_params = vec!["beta".into(), "N0".into()];
        let err = config.validate(&model_params, InitLaw::Absent).unwrap_err();
        assert!(err.contains("iterations must be"),
            "expected iterations error: got {}", err);
        assert!(err.contains("[method]"),
            "expected the table named in the error: got {}", err);
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err_msg = err.validate(&model_params, InitLaw::Absent).unwrap_err();
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.problem.config.dt, 1.0);
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
[method]
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
    fn removed_condition_from_key_is_rejected_with_covers_hint() {
        // Ruling 4 of proposal 2026-09-05-observation-time-as-a-sum-type: the
        // key is gone, and every spelling of it must fail with the hint that
        // names `covers`, not a bare serde "unknown field" error. Three
        // spellings: the top-level scalar, the `[condition_from]` table, and
        // the key misplaced under `[data]`.
        let body = r#"
[model]
camdl = "m.camdl"
[estimate]
beta = { bounds = [0.01, 2.0] }
[fixed]
gamma = 0.2
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
"#;
        let spellings = [
            format!("condition_from = \"first_obs - 1 week\"\n[data.observations]\ncases = \"d.tsv\"\n{body}"),
            format!("[condition_from]\ncases = \"14\"\n[data.observations]\ncases = \"d.tsv\"\n{body}"),
            format!("[data]\ncondition_from = \"14\"\n[data.observations]\ncases = \"d.tsv\"\n{body}"),
        ];
        for toml_str in &spellings {
            let err = parse(toml_str).unwrap_err();
            assert_eq!(err, CONDITION_FROM_REMOVED_MSG, "exact hint text for:\n{toml_str}");
            assert!(err.contains("covers"), "points at the replacement: {err}");
            assert!(!err.contains("unknown field"), "not a bare serde error: {err}");
        }
        // The same document without the key parses — the detector is keyed
        // on the name, not on anything else in the fixture.
        parse(&format!("[data.observations]\ncases = \"d.tsv\"\n{body}"))
            .expect("fixture without condition_from must parse");
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
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
        "#).unwrap();
        assert_eq!(cfg.problem.synthetic.as_ref().unwrap().backend, ForwardBackend::Gillespie);

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
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
        "#).unwrap();
        assert_eq!(cfg2.problem.synthetic.as_ref().unwrap().backend, ForwardBackend::ChainBinomial);
    }

    // ── `starts` (proposal §3.1) ───────────────────────────────────────────

    fn method_with_starts(starts_line: &str) -> Result<Method, String> {
        parse(&format!(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

[fixed]
N0 = 1000000

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
{starts_line}
        "#)).map(|c| method(&c))
    }

    #[test]
    fn starts_bare_rules_parse() {
        let cases = [
            ("uniform_unconstrained", ChainStarts::Spread(Spread::UniformUnconstrained)),
            ("lhs", ChainStarts::Spread(Spread::Lhs)),
            ("uniform", ChainStarts::Spread(Spread::Uniform)),
            ("from_prior", ChainStarts::Spread(Spread::FromPrior)),
            ("single", ChainStarts::Point(Point::Declared)),
        ];
        for (name, want) in cases {
            let m = method_with_starts(&format!("starts = \"{name}\"")).unwrap();
            assert_eq!(m.starts, Some(want.clone()), "starts = \"{name}\"");
            assert_eq!(m.starts().unwrap().spelled(), name);
        }
    }

    #[test]
    fn starts_sourced_rules_parse_as_one_key_tables() {
        let m = method_with_starts("starts = { from_posterior = \"@base\" }").unwrap();
        assert_eq!(m.starts, Some(ChainStarts::Spread(Spread::FromPosterior {
            source: Handle("@base".into()) })));
        let m = method_with_starts("starts = { from_mle = \"@mle\" }").unwrap();
        assert_eq!(m.starts, Some(ChainStarts::Point(Point::FromMle {
            source: Handle("@mle".into()) })));
        assert_eq!(m.starts.as_ref().unwrap().spelled(), "from_mle @mle");
        let m = method_with_starts("starts = { from_params = \"theta.toml\" }").unwrap();
        assert_eq!(m.starts, Some(ChainStarts::Point(Point::FromParams {
            path: "theta.toml".into() })));
    }

    #[test]
    fn starts_rejects_the_removed_and_the_unknown() {
        let err = method_with_starts("starts = \"survey_top_k\"").unwrap_err();
        assert!(err.contains("survey_top_k") && err.contains("removed"), "{err}");
        let err = method_with_starts("starts = \"lhss\"").unwrap_err();
        assert!(err.contains("lhss") && err.contains("uniform_unconstrained"),
            "must name the rule and list the spellings: {err}");
        // A sourced rule is a table, not a bare name.
        let err = method_with_starts("starts = \"from_mle\"").unwrap_err();
        assert!(err.contains("from_mle") && err.contains("{ from_mle = "), "{err}");
        // …with exactly one key.
        let err = method_with_starts(
            "starts = { from_mle = \"@a\", from_posterior = \"@b\" }").unwrap_err();
        assert!(err.contains("one sourced rule") && err.contains("2 keys"), "{err}");
    }

    /// An omitted `starts` is left unresolved by the parser: the default is a
    /// function of the priors, which need the model. `Method::starts()` says
    /// so rather than guessing.
    #[test]
    fn starts_absent_is_unresolved_until_fit_run() {
        let m = method_with_starts("").unwrap();
        assert!(m.starts.is_none());
        let err = m.starts().unwrap_err();
        assert!(err.contains("resolve_starts"), "{err}");
    }

    /// The default (§8, item 20): `from_prior` when every estimated
    /// parameter has a prior the chains can be drawn from, otherwise
    /// `uniform_unconstrained`, naming the parameters without one. A spelled
    /// rule is left alone.
    #[test]
    fn starts_default_is_from_prior_only_when_every_parameter_has_one() {
        let model = model_with_scenario("baseline", &[]);
        let problem_with = |estimate: &str| -> FitConfig {
            parse(&format!(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
{estimate}

[fixed]
N0 = 1000000

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 100
            "#)).unwrap()
        };

        // Every estimated parameter declares a prior → from_prior.
        let cfg = problem_with(
            "beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }\n\
             gamma = { bounds = [0.01, 1.0], prior = { half_normal = { sigma = 1.0 } } }");
        let mut m = method(&cfg);
        let decided = m.resolve_starts(&cfg.problem, &model);
        assert_eq!(decided, StartsResolution::DefaultFromPrior);
        assert_eq!(m.starts, Some(ChainStarts::from_prior()));
        assert!(decided.describe().contains("from_prior"), "{}", decided.describe());

        // One parameter without a prior → uniform_unconstrained, and the
        // reason names it.
        let cfg = problem_with(
            "beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }\n\
             gamma = { bounds = [0.01, 1.0] }");
        let mut m = method(&cfg);
        let decided = m.resolve_starts(&cfg.problem, &model);
        assert_eq!(decided, StartsResolution::DefaultUniformUnconstrained {
            without_prior: vec!["gamma".into()] });
        assert_eq!(m.starts, Some(ChainStarts::uniform_unconstrained()));
        let said = decided.describe();
        assert!(said.contains("uniform_unconstrained") && said.contains("gamma"), "{said}");

        // An explicit flat prior is not a distribution to draw from either.
        let cfg = problem_with(
            "beta = { bounds = [0.01, 2.0], prior = { flat = {} } }");
        let mut m = method(&cfg);
        assert!(matches!(m.resolve_starts(&cfg.problem, &model),
            StartsResolution::DefaultUniformUnconstrained { .. }));

        // A spelled rule is left alone whatever the priors say.
        let cfg = problem_with("beta = { bounds = [0.01, 2.0] }");
        let mut m = method(&cfg);
        m.starts = Some(ChainStarts::Spread(Spread::Lhs));
        let decided = m.resolve_starts(&cfg.problem, &model);
        assert_eq!(decided, StartsResolution::Declared(ChainStarts::Spread(Spread::Lhs)));
        assert_eq!(m.starts, Some(ChainStarts::Spread(Spread::Lhs)));
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#, params_path.display());

        let config = FitConfig::from_toml_str(&toml_str).unwrap();
        let resolved = config.problem.fixed.resolve().unwrap();
        assert_eq!(resolved["N0"], 1000000.0);
        assert_eq!(resolved["I0"], 10.0);

        // Validate with correct model params
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        assert!(config.validate(&model_params, InitLaw::Absent).is_ok());
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#, params_path.display());

        let config = FitConfig::from_toml_str(&toml_str).unwrap();
        let resolved = config.problem.fixed.resolve().unwrap();
        assert_eq!(resolved["N0"], 1000000.0);
        assert_eq!(resolved["I0"], 50.0); // inline overrides from_file
    }

    /// gh#439 A2: `needs_state_grad` is true for exactly the nuts+ode cell — the
    /// only consumer of the WrtPop state-Jacobian — and false for every other
    /// (algorithm, backend) combination, including the near-miss `mh` on `ode`
    /// (Bayesian on the ODE backend, but gradient-free → must compile lean).
    #[test]
    fn needs_state_grad_only_for_nuts_ode() {
        let cfg = |stage: &str| -> FitConfig {
            let toml_str = format!(
                "[model]\ncamdl = \"models/sir.camdl\"\n\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\n\
                 [fixed]\nN0 = 1000\n\n{stage}"
            );
            FitConfig::from_toml_str(&toml_str).unwrap_or_else(|e| panic!("parse {stage:?}: {e}"))
        };

        // nuts + ode → the sole state-Jacobian consumer (true).
        assert!(
            cfg("[method]\nalgorithm = \"nuts\"\nbackend = \"ode\"\nchains = 2")
                .needs_state_grad(),
            "nuts+ode drives the ODE forward-sensitivity gradient — needs the Jacobian"
        );

        // mh + ode → gradient-free Bayesian on the ODE backend → lean (false).
        assert!(
            !cfg("[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 100")
                .needs_state_grad(),
            "mh on ode is gradient-free — must compile lean"
        );

        // if2 + chain_binomial → gradient-free MLE → lean (false).
        assert!(
            !cfg("[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                  chains = 4\nparticles = 100\niterations = 10\ncooling = 0.7")
                .needs_state_grad(),
            "if2+chain_binomial never reads the state-Jacobian"
        );

        // A problem with no [method] (a file for the non-fit readers) has
        // nothing that reads the Jacobian.
        assert!(!cfg("").needs_state_grad(), "no method, no gradient consumer");
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

[method]
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
        let syn = config.problem.synthetic.as_ref().expect("[synthetic] missing");
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
        let syn = config.problem.synthetic.unwrap();
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
        let err = config.problem.synthetic.unwrap().validate().unwrap_err();
        assert!(err.contains("20") && err.contains("5"),
            "error must name both counts: {}", err);
    }

    #[test]
    fn data_and_synthetic_may_coexist() {
        let src = format!(r#"{}
[data.observations]
cases = "data/cases.tsv"

[synthetic]
true_params = "truth.toml"
sim_seeds   = "1:5"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()], InitLaw::Absent)
            .expect("[data] and [synthetic] together is a real-data fit with a recorded truth");
        assert!(!config.problem.is_synthetic_fit());
    }

    #[test]
    fn neither_data_nor_synthetic_errors() {
        let src = minimal_fit_stages().to_string();
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()], InitLaw::Absent)
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
        assert_eq!(config.inference.fit_seeds.unwrap(), vec![42u64]);

        let list_src = format!(r#"fit_seeds = [101, 102, 103]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_stages());
        let config = parse(&list_src).unwrap();
        assert_eq!(config.inference.fit_seeds.unwrap(), vec![101u64, 102, 103]);
    }

    // ── point-start note (proposal §3.4) ──────────────────────────────────

    /// A posterior-sampling method with a tunable `starts` / `chains`, filled
    /// in per test so the trigger and its controls share one fixture.
    fn fit_pgas_with_starts(starts: &str, chains: usize) -> String {
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
starts = {starts}
chains = {chains}
particles = 500
sweeps = 1000
"#, starts = starts, chains = chains)
    }

    #[test]
    fn point_start_multichain_note_names_the_rule_and_the_consequence() {
        for starts in ["\"single\"", "{ from_mle = \"@scout\" }", "{ from_params = \"theta.toml\" }"] {
            let config = parse(&fit_pgas_with_starts(starts, 4)).unwrap();
            let msg = config.point_start_multichain_note()
                .unwrap_or_else(|| panic!("a point rule with chains > 1 must be noted: {starts}"));
            assert!(msg.contains("4 chains") && msg.contains("one point"),
                "note must say every chain starts at one point: {msg}");
            assert!(msg.contains("not assessed"),
                "note must say what fit summary will report: {msg}");
            assert!(msg.contains("from_prior") && msg.contains("from_posterior"),
                "note must name the spread rules: {msg}");
        }
    }

    #[test]
    fn point_start_note_silent_for_one_chain_a_spread_rule_or_an_optimizer() {
        // One chain: there is no between-chain R̂ to weaken.
        let config = parse(&fit_pgas_with_starts("\"single\"", 1)).unwrap();
        assert!(config.point_start_multichain_note().is_none());
        // A spread rule: the chains begin apart, R̂ is informative.
        for starts in ["\"lhs\"", "\"from_prior\"", "{ from_posterior = \"@base\" }"] {
            let config = parse(&fit_pgas_with_starts(starts, 4)).unwrap();
            assert!(config.point_start_multichain_note().is_none(), "{starts}");
        }
        // An unresolved `starts` is not yet a point rule.
        let src = fit_pgas_with_starts("\"single\"", 4).replace("starts = \"single\"\n", "");
        assert!(parse(&src).unwrap().point_start_multichain_note().is_none());
        // An optimizer reports no R̂.
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

[method]
algorithm = "if2"
backend = "chain_binomial"
starts = "single"
chains = 4
particles = 500
iterations = 50
cooling = 0.7
"#;
        assert!(parse(src).unwrap().point_start_multichain_note().is_none(),
            "IF2 is not a posterior sampler — no R̂ to protect");
    }

    #[test]
    fn fit_seeds_duplicates_rejected_during_validate() {
        let src = format!(r#"fit_seeds = [1, 2, 1]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_stages());
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()], InitLaw::Absent)
            .unwrap_err();
        assert!(err.contains("duplicate"), "must reject duplicate fit seeds: {}", err);
    }

    // ── ic_free / conditioning support check (F1) ──────────────────────────
    //
    // `ic_free = true` needs an algorithm that drops y₁ from the accumulated
    // loglik AND a swarm whose particles differ in x₀. PGAS, the ODE-MLE
    // optimizers and correlated PMMH score every obs unconditionally, for every
    // model. `pfilter` and plain `pmmh` DO drop y₁, and their bootstrap PF
    // draws x₀ per particle — so they have the spread exactly when the MODEL's
    // `init { }` declares a law, and not when it computes every compartment
    // from an expression (gh#732). `validate()` therefore takes that model fact
    // as `InitLaw`.

    /// Model params for the ic_free fixtures (sir with beta/gamma/N0/I0).
    fn ic_free_model_params() -> Vec<String> {
        vec!["beta".into(), "gamma".into(), "N0".into(), "I0".into()]
    }

    #[test]
    fn ic_free_with_if2_stage_still_validates() {
        // IF2 both drops y₁ from the accumulated loglik AND gives each
        // particle its own x₀ drawn from its own perturbed θ (gh#364), so
        // ic_free = true must NOT be rejected. Negative control for the
        // gh#732 refusals below: without this, "refuse everything" would
        // pass them all.
        let src = format!(
            "ic_free = true\n{}\n[data.observations]\ncases = \"data/cases.tsv\"\n",
            minimal_fit_stages()
        );
        let config = parse(&src).unwrap();
        config
            .validate(&ic_free_model_params(), InitLaw::Absent)
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 500
sweeps = 100
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params(), InitLaw::Absent)
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
[method]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
burn_in = 3000
"#;
        let config = parse(src).unwrap();
        let err = config.validate(&ic_free_model_params(), InitLaw::Absent)
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
[method]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
"#;
        let config = parse(src).unwrap();
        let err = config.validate(&ic_free_model_params(), InitLaw::Absent)
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
[method]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 2000
burn_in = 500
"#;
        let config = parse(src).unwrap();
        config.validate(&ic_free_model_params(), InitLaw::Absent)
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

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params(), InitLaw::Absent)
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

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 500
iterations = 100
rho = 0.99
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect_err("ic_free=true on a correlated PMMH stage must be rejected");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
    }

    /// gh#732. Plain PMMH (no `rho`) wraps the bootstrap particle filter,
    /// which draws x₀ per particle — but on a model whose `init { }` computes
    /// every compartment from an expression each of those draws returns the
    /// same state, so `ic_free`'s first reweight scores every particle
    /// identically and the run drops y₁ instead of conditioning on it. Refuse
    /// at config load, naming the reason.
    #[test]
    fn ic_free_with_plain_pmmh_stage_is_rejected() {
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

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 500
iterations = 100
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect_err("ic_free=true on a plain PMMH stage must be rejected (gh#732)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("pmmh"), "error must name the offending algorithm: {err}");
        assert!(err.contains("bootstrap particle filter"),
            "error must name the mechanism that fails, not just refuse: {err}");
        assert!(err.contains("gh#732"), "error must cite the issue: {err}");
        assert!(err.contains("if2"), "error must name the algorithm that works: {err}");
    }

    /// gh#732, same cell, other algorithm: `pfilter` runs through the same arm
    /// and the same bootstrap PF.
    #[test]
    fn ic_free_with_pfilter_stage_is_rejected() {
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

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 500
"#;
        let config = parse(src).unwrap();
        let err = config
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect_err("ic_free=true on a pfilter stage must be rejected (gh#732)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("pfilter"), "error must name the offending algorithm: {err}");
        assert!(err.contains("no spread at t=0"),
            "error must say why the spread is absent: {err}");
    }

    /// gh#732, the cell this change OPENS, at the config seam: the SAME two
    /// configs that are refused above validate against a model whose `init { }`
    /// declares a law. Only the model fact differs — which is the point: the
    /// refusal is a property of (algorithm × model), not of the algorithm.
    #[test]
    fn ic_free_with_bootstrap_pf_stages_validates_when_the_model_draws_x0() {
        for (who, stage) in [
            ("pmmh", "[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\n\
                      chains = 1\nparticles = 500\niterations = 100\nburn_in = 10\n"),
            ("pfilter", "[method]\nalgorithm = \"pfilter\"\n\
                         backend = \"chain_binomial\"\nparticles = 500\n"),
        ] {
            let src = format!(
                "ic_free = true\n\
                 [model]\ncamdl = \"models/sir.camdl\"\n\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\n\
                 [fixed]\nN0 = 1000\nI0 = 5\ngamma = 0.1\n\n{stage}"
            );
            let config = parse(&src).unwrap();
            config
                .validate(&ic_free_model_params(), InitLaw::Declared)
                .unwrap_or_else(|e| panic!(
                    "{who} + a declared init law must validate — the bootstrap PF \
                     draws x₀ per particle, so the swarm has spread at t=0: {e}"));
            // Non-vacuity: the same config with the same algorithm IS refused
            // when the model's init is deterministic, so the accept above is
            // the model fact doing the work and not a config that would pass
            // either way.
            config
                .validate(&ic_free_model_params(), InitLaw::Absent)
                .expect_err(&format!("{who} + a deterministic init must still be refused"));
        }
    }

    // ── perturb_only_at_t0 (proposal §8, item 12) ──────────────────────────
    //
    // The flag is an IF2 perturbation schedule declared in `[estimate]`, which
    // is the problem half a family of method files shares. It stays there and
    // is inert for every non-IF2 method: refusing a `pgas` file for a flag its
    // sibling `if2` file reads would force an edit to the shared problem.

    fn t0_fit(algorithm: &str, stage_body: &str, declare_flag: bool) -> String {
        let flag = if declare_flag { ", perturb_only_at_t0 = true" } else { "" };
        format!(r#"[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
I0 = {{ bounds = [1, 500]{flag} }}

[fixed]
beta = 0.3
gamma = 0.1
N0 = 1000

[method]
algorithm = "{algorithm}"
{stage_body}
"#)
    }

    const IF2_BODY: &str = "backend = \"chain_binomial\"\nchains = 2\nparticles = 100\n\
                            iterations = 10\ncooling = 0.9\n";
    const PGAS_BODY: &str = "backend = \"chain_binomial\"\nchains = 2\nparticles = 500\n\
                             sweeps = 100\nburn_in = 10\n";

    #[test]
    fn perturb_only_at_t0_with_if2_stage_validates() {
        parse(&t0_fit("if2", IF2_BODY, true)).unwrap()
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect("IF2 honours perturb_only_at_t0 — it is IF2's own schedule");
    }

    /// Every other method accepts the flag and ignores it.
    #[test]
    fn perturb_only_at_t0_is_inert_for_non_if2_methods() {
        for (algo, body) in [
            ("pgas", PGAS_BODY),
            ("mh", "backend = \"ode\"\nchains = 2\niterations = 2000\nburn_in = 500\n"),
            ("nl-sbplx", "backend = \"ode\"\nchains = 1\n"),
        ] {
            parse(&t0_fit(algo, body, true)).unwrap()
                .validate(&ic_free_model_params(), InitLaw::Absent)
                .unwrap_or_else(|e| panic!(
                    "{algo} must accept a perturb_only_at_t0 declaration it does not read: {e}"));
        }
    }

    /// Control: the same methods validate when no `[estimate]` entry declares
    /// the flag, so the accept above is not a fixture that would pass either
    /// way.
    #[test]
    fn a_config_without_the_flag_is_unaffected() {
        for (algo, body) in [
            ("pgas", PGAS_BODY),
            ("mh", "backend = \"ode\"\nchains = 2\niterations = 2000\nburn_in = 500\n"),
            ("nl-sbplx", "backend = \"ode\"\nchains = 1\n"),
        ] {
            parse(&t0_fit(algo, body, false)).unwrap()
                .validate(&ic_free_model_params(), InitLaw::Absent)
                .unwrap_or_else(|e| panic!("{algo} without the flag must validate: {e}"));
        }
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

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 1000
        "#).unwrap();

        match &algo(&cfg) {
            Algorithm::PFilter { record_prequential, record_ancestry, .. } => {
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

[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 1000
record_prequential = false
        "#).unwrap();

        match &algo(&cfg) {
            Algorithm::PFilter { record_prequential, .. } =>
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        match &algo(&cfg) {
            Algorithm::IF2 { loglik_eval, gate, .. } => {
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
loglik_eval = { n_particles = 8000, n_replicates = 16, combine = "mean" }
gate = { a_thresh = 1.05, decibans_thresh = 60.0 }
        "#).unwrap();

        match &algo(&cfg) {
            Algorithm::IF2 { loglik_eval, gate, .. } => {
                assert_eq!(loglik_eval.n_particles, 8000);
                assert_eq!(loglik_eval.n_replicates, 16);
                assert_eq!(loglik_eval.combine, CombineMode::Mean);
                assert!((gate.a_thresh - 1.05).abs() < 1e-12);
                assert!((gate.decibans_thresh - 60.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }

        // The sub-table form, with partial overrides — unset fields take
        // defaults.
        let cfg = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 60
cooling = 0.95

[method.loglik_eval]
n_particles = 12000

[method.gate]
decibans_thresh = 100.0
        "#).unwrap();
        match &algo(&cfg) {
            Algorithm::IF2 { loglik_eval, gate, .. } => {
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

    /// Default-equipped PGAS method for identity tests, with `starts`
    /// resolved as `fit run` resolves it before any identity is taken.
    fn make_pgas_stage(sweeps: usize) -> Method {
        Method {
            algorithm: Algorithm::PGAS {
                backend: crate::run_meta::InferenceBackend::ChainBinomial,
                chains: 4, particles: 100, sweeps,
                burn_in: Some(200), thin: Some(2),
                tempering: vec![1.0],
                max_tree_depth: 10,
                trajectory_warmup: 0,
                csmc_sweeps_per_nuts: 1,
                n_trajectories: 200,
                dense_mass: true,
                use_nuts: true,
                binomial: sim::rng::BinomialAlgorithm::Btpe,
                ancestor_sampling: true,
            },
            starts: Some(ChainStarts::uniform_unconstrained()),
        }
    }

    /// Default-equipped PMMH method for identity tests.
    fn make_pmmh_stage(iterations: usize) -> Method {
        Method {
            algorithm: Algorithm::PMMH {
                backend: crate::run_meta::InferenceBackend::ChainBinomial,
                chains: 4, particles: 100, iterations,
                burn_in: Some(200), thin: Some(2),
                adapt: true, adapt_start: 300, rho: None,
            },
            starts: Some(ChainStarts::uniform_unconstrained()),
        }
    }

    #[test]
    fn pgas_identity_payload_omits_sweeps() {
        // Two PGAS methods identical except for `sweeps` must produce
        // the same identity_payload — that's the contract that lets
        // --resume extend a chain by changing the iteration count.
        let s_short = make_pgas_stage(1000);
        let s_long = make_pgas_stage(5000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());

        // Changing any *other* PGAS field must change the payload.
        let mut s_more_chains = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut chains, .. } = s_more_chains.algorithm { *chains = 8; }
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
        if let Algorithm::PGAS { ref mut n_trajectories, .. } = s_few.algorithm { *n_trajectories = 100; }
        if let Algorithm::PGAS { ref mut n_trajectories, .. } = s_many.algorithm { *n_trajectories = 1000; }
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
        if let Algorithm::PGAS { ref mut tempering, .. } = s.algorithm {
            *tempering = vec![1.0, 0.5];
        }
        assert_ne!(base.identity_payload(), s.identity_payload(), "tempering");

        let mut s = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut max_tree_depth, .. } = s.algorithm { *max_tree_depth = 14; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "max_tree_depth");

        let mut s = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut trajectory_warmup, .. } = s.algorithm {
            *trajectory_warmup = 100;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(), "trajectory_warmup");

        let mut s = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut csmc_sweeps_per_nuts, .. } = s.algorithm {
            *csmc_sweeps_per_nuts = 3;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(),
            "csmc_sweeps_per_nuts");

        let mut s = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut dense_mass, .. } = s.algorithm { *dense_mass = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "dense_mass");

        let mut s = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut use_nuts, .. } = s.algorithm { *use_nuts = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "use_nuts");
    }

    /// `starts` chooses the per-chain starting points, which determine the
    /// stored chains/posterior. Two fits differing only in it must not
    /// collide — otherwise the first run's posterior is silently served as the
    /// second's (a wrong scientific result on a multimodal problem). gh#147
    /// count-in-the-key; the handle as written is part of the key, and a
    /// regenerated source re-keys through the method level's deps.
    #[test]
    fn starts_is_in_the_method_identity() {
        for (who, base) in [("pgas", make_pgas_stage(1000)), ("pmmh", make_pmmh_stage(1000))] {
            let with = |starts: ChainStarts| -> Method {
                let mut m = base.clone();
                m.starts = Some(starts);
                m
            };
            let lhs = with(ChainStarts::Spread(Spread::Lhs));
            let single = with(ChainStarts::Point(Point::Declared));
            let prior = with(ChainStarts::from_prior());
            let mle_a = with(ChainStarts::Point(Point::FromMle { source: Handle("@a".into()) }));
            let mle_b = with(ChainStarts::Point(Point::FromMle { source: Handle("@b".into()) }));
            let post_a = with(ChainStarts::Spread(Spread::FromPosterior { source: Handle("@a".into()) }));
            assert_ne!(base.identity_payload(), lhs.identity_payload(), "{who}: uniform_unconstrained vs lhs");
            assert_ne!(lhs.identity_payload(), single.identity_payload(), "{who}: lhs vs single");
            assert_ne!(lhs.identity_payload(), prior.identity_payload(), "{who}: lhs vs from_prior");
            assert_ne!(prior.identity_payload(), mle_a.identity_payload(), "{who}: from_prior vs from_mle");
            assert_ne!(mle_a.identity_payload(), mle_b.identity_payload(), "{who}: the handle is in the key");
            assert_ne!(mle_a.identity_payload(), post_a.identity_payload(), "{who}: from_mle vs from_posterior");
            // The payload carries the wire form, under its TOML spelling.
            let v = mle_a.identity_payload();
            assert_eq!(v["starts"], serde_json::json!({ "from_mle": "@a" }), "{who}");
            assert_eq!(lhs.identity_payload()["starts"], serde_json::json!("lhs"), "{who}");
        }
    }

    #[test]
    fn pmmh_identity_payload_omits_iterations() {
        let s_short = make_pmmh_stage(1000);
        let s_long = make_pmmh_stage(8000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());
    }

    /// gh#747: the two samplers must not collide in the store.
    ///
    /// This is the assertion the whole design turns on, and it is deliberately
    /// written against the PAYLOAD rather than the attribute: it survives any
    /// future serde mistake in either direction — a `skip_serializing_if` that
    /// starts skipping `btrs`, or one that stops skipping `btpe`.
    #[test]
    fn binomial_sampler_is_in_the_stage_identity() {
        let mut btpe = make_pgas_stage(1000);
        let mut btrs = make_pgas_stage(1000);
        if let Algorithm::PGAS { ref mut binomial, .. } = btrs.algorithm {
            *binomial = sim::rng::BinomialAlgorithm::Btrs;
        }
        if let Algorithm::PGAS { ref mut binomial, .. } = btpe.algorithm {
            *binomial = sim::rng::BinomialAlgorithm::Btpe;
        }
        let a = serde_json::to_string(&btpe.identity_payload()).unwrap();
        let b = serde_json::to_string(&btrs.identity_payload()).unwrap();
        assert_ne!(a, b,
            "btpe and btrs stages produced the SAME identity payload — the two \
             samplers would share one address, and one sampler's posterior \
             would be served from the other's leaf. That is the exact failure \
             an environment variable would have caused, and the reason this is \
             a typed field.");
        assert!(b.contains(r#""binomial":"btrs""#),
            "a btrs stage's payload must NAME the sampler; got {b}");
        // NOTE the key form, not the bare substring: `"backend":"chain_binomial"`
        // contains "binomial" and made the naive check fail.
        assert!(!a.contains(r#""binomial":"#),
            "a btpe stage's payload must stay byte-identical to the pre-field \
             format, so adding this field orphans no stored leaf. Absence means \
             BTPE — permanently, and NOT 'whatever the default is'. If the \
             default ever flips (gh#761), `is_btpe` must not follow it. Got {a}");
    }

    /// gh#747: the TOML spelling round-trips, and an absent field is BTPE.
    #[test]
    fn binomial_sampler_parses_from_stage_toml() {
        let with_btrs: Algorithm = toml::from_str(
            "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 4\n\
             particles = 100\nsweeps = 1000\nbinomial = \"btrs\"\n").unwrap();
        match with_btrs {
            Algorithm::PGAS { binomial, .. } =>
                assert_eq!(binomial, sim::rng::BinomialAlgorithm::Btrs),
            _ => panic!("expected a PGAS stage"),
        }
        // Absent -> BTPE. Every fit.toml written before this field relies on it.
        let absent: Algorithm = toml::from_str(
            "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 4\n\
             particles = 100\nsweeps = 1000\n").unwrap();
        match absent {
            Algorithm::PGAS { binomial, .. } =>
                assert_eq!(binomial, sim::rng::BinomialAlgorithm::Btpe,
                    "an absent `binomial` must mean BTPE"),
            _ => panic!("expected a PGAS stage"),
        }
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
        // Updated with the `[stages]` → `[method]` split (proposal
        // 2026-09-08): the payload is the method's own serialization, so the
        // one chain-start key is `starts` under its TOML spelling, and the
        // retired `init` / `init_mle` / `survey_*` keys are gone. A deliberate
        // re-key of every method leaf, ruled in §8 item 22; see the commit for
        // the --resume consequence.
        let expected = r#"{"algorithm":"pgas","backend":"chain_binomial","burn_in":200,"chains":4,"csmc_sweeps_per_nuts":1,"dense_mass":true,"max_tree_depth":10,"particles":100,"starts":"uniform_unconstrained","tempering":[1.0],"thin":2,"trajectory_warmup":0,"use_nuts":true}"#;
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
        // Updated with the `[stages]` → `[method]` split — see the PGAS golden
        // above for the reason.
        let expected = r#"{"adapt":true,"adapt_start":300,"algorithm":"pmmh","backend":"chain_binomial","burn_in":200,"chains":4,"particles":100,"rho":null,"starts":"uniform_unconstrained","thin":2}"#;
        assert_eq!(payload_str, expected,
            "PMMH identity_payload byte format drifted — see \
             pgas_identity_payload_byte_stable for context.");
    }

    #[test]
    fn pmmh_identity_payload_includes_new_algorithmic_knobs() {
        let base = make_pmmh_stage(1000);

        let mut s = make_pmmh_stage(1000);
        if let Algorithm::PMMH { ref mut adapt, .. } = s.algorithm { *adapt = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt");

        let mut s = make_pmmh_stage(1000);
        if let Algorithm::PMMH { ref mut adapt_start, .. } = s.algorithm { *adapt_start = 1000; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt_start");

        let mut s = make_pmmh_stage(1000);
        if let Algorithm::PMMH { ref mut rho, .. } = s.algorithm { *rho = Some(0.99); }
        assert_ne!(base.identity_payload(), s.identity_payload(), "rho");
    }

    fn if2_method(algorithm: Algorithm) -> Method {
        Method { algorithm, starts: Some(ChainStarts::uniform_unconstrained()) }
    }

    #[test]
    fn if2_identity_payload_includes_iterations_and_cooling() {
        // IF2 has no extension dimension — its cooling schedule is
        // determined by the total iteration count, so changing
        // iterations *must* invalidate identity (and thus reject
        // resume). This guards against a future refactor accidentally
        // moving `iterations` out of identity.
        let s50 = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.95,
            cooling_target_iters: 50,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        });
        let s100 = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 100, cooling: 0.95,
            cooling_target_iters: 50,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        });
        assert_ne!(s50.identity_payload(), s100.identity_payload());

        let s_diff_cooling = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.70,
            cooling_target_iters: 50,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        });
        assert_ne!(s50.identity_payload(), s_diff_cooling.identity_payload());

        // cooling_target_iters is identity-defining (different schedule
        // → different chain dynamics).
        let s_diff_target = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.95,
            cooling_target_iters: 100,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
        });
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config = FitConfig::from_toml_str(toml_str)
            .expect("bounds must be optional");
        assert!(config.problem.estimate.contains_key("beta"));
        assert_eq!(config.problem.estimate["beta"].bounds, None,
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config = FitConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.problem.estimate["beta"].bounds, Some((0.01, 2.0)));
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config = FitConfig::from_toml_str(toml_str).unwrap();
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        config.validate(&model_params, InitLaw::Absent).expect("validation must pass with omitted bounds");
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 100
iterations = 50
cooling = 0.7
"#;
        let config = FitConfig::from_toml_str(toml_str).unwrap();
        let model_params = vec!["beta".to_string(), "N0".to_string(), "I0".to_string()];
        let err = config.validate(&model_params, InitLaw::Absent)
            .expect_err("inverted explicit bounds must error");
        assert!(err.contains("are empty") || err.contains("lo must be < hi"),
            "error must name the lo/hi violation; got: {err}");
    }

    // ─── Migration: `[stages]` and the removed keys (proposal §5) ───────────

    /// `starts` is a key of `[method]` whatever the algorithm — the NLopt
    /// methods included, which used to carry their own `init` inside
    /// `NloptStageConfig` and could answer differently (gh#881).
    #[test]
    fn starts_parses_on_every_algorithm() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }\n\
                    [fixed]\nN0 = 1000000\n";
        for body in [
            "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 4\nparticles = 500\n\
             iterations = 30\ncooling = 0.7\n",
            "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 100\nsweeps = 10\n",
            "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 100\niterations = 10\n",
            "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 10\n",
            "algorithm = \"nuts\"\nbackend = \"ode\"\nchains = 2\n",
            "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 4\n",
            "algorithm = \"nl-bobyqa\"\nbackend = \"ode\"\nchains = 4\n",
            "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 100\n",
        ] {
            let cfg = parse(&format!("{base}[method]\n{body}starts = \"single\"\n"))
                .unwrap_or_else(|e| panic!("{body}: {e}"));
            assert_eq!(method(&cfg).starts, Some(ChainStarts::Point(Point::Declared)), "{body}");
        }
    }

    /// One `[stages.X]` with an `init`: the rewrite is a rename and a
    /// `starts` line.
    #[test]
    fn legacy_single_stage_is_rejected_with_the_rename() {
        let err = parse(r#"
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
        "#).expect_err("[stages.mle] is refused");
        assert_eq!(err,
            "legacy table `[stages.mle]`\n  \
             replacement: rename to `[method]` and run it with\n    \
             camdl fit run fit.toml\n  \
             `init = \"lhs\"` becomes `starts = \"lhs\"`\n  \
             See `camdl docs fit-toml`.");
    }

    /// The chained shape — a scout whose point estimate seeded the posterior —
    /// names both sourced rules, because the file cannot decide between them.
    #[test]
    fn legacy_chained_stages_are_rejected_naming_both_starts_forms() {
        let err = FitConfig::from_toml_str_named(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }

[fixed]
N0 = 1000000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 100
sweeps = 100
init_mle = "scout"
        "#, "fits/ebola.toml").expect_err("a chained pipeline is refused");
        assert_eq!(err,
            "legacy table `[stages.posterior]`\n  \
             replacement: rename to `[method]` and run it with\n    \
             camdl fit run fits/ebola.toml\n  \
             a file carries one `[method]`; put `[stages.scout]` in its own file\n  \
             `init_mle = \"scout\"` has no replacement in the file: it started every chain at\n  \
             scout's point estimate, which makes R̂ uninformative. Run scout first and, if a\n  \
             warm start is wanted, write one of\n    \
             starts = { from_posterior = \"@scout\" }   # one draw per chain (keeps R̂ meaningful)\n    \
             starts = { from_mle = \"@scout\" }         # every chain at one point (R̂ not assessed)\n  \
             See `camdl docs fit-toml`.");
    }

    /// `survey_top_k` and its companions are gone, not renamed.
    #[test]
    fn legacy_survey_init_is_rejected_as_removed() {
        let err = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7
init = "survey_top_k"
survey_top_k_n = 4
        "#).expect_err("[stages.scout] is refused");
        assert!(err.starts_with("legacy table `[stages.scout]`"), "{err}");
        assert!(err.contains("`init = \"survey_top_k\"` (with `survey_path` / `survey_top_k_n`) was removed"),
            "{err}");
        assert!(err.contains("starts = \"from_prior\"") && err.contains("from_posterior"), "{err}");
        assert!(!err.contains("becomes `starts = \"survey_top_k\"`"),
            "a removed rule must not be offered as a rename: {err}");
    }

    /// `fit_starts` was read by nothing; its one meaningful value is now the
    /// default.
    #[test]
    fn removed_fit_starts_key_is_rejected_with_the_default_explained() {
        let err = parse(&format!("fit_starts = \"prior\"\n{STRICT_BASE}"))
            .expect_err("fit_starts is not a key");
        assert!(err.contains("`fit_starts` is no longer a fit.toml key"), "{err}");
        assert!(err.contains("`from_prior` is the default"), "{err}");
        assert!(err.contains("`fit_starts = \"model_default\"` is `starts = \"single\"`"), "{err}");
    }

    /// A file with no `[method]` at all is a complete problem: the non-fit
    /// readers load it, and `fit run` refuses it by name.
    #[test]
    fn a_problem_only_file_loads_and_fit_run_refuses_it() {
        let (problem_half, _) = STRICT_BASE.split_once("[method]").unwrap();
        let cfg = parse(problem_half).expect("a [method]-less file is a valid problem");
        assert!(cfg.inference.method.is_none());
        let err = cfg.method().unwrap_err();
        assert!(err.contains("declares no `[method]` table"), "{err}");
        assert!(err.contains("simulate --fit") && err.contains("[method]\n  algorithm"), "{err}");
        // And the problem half validates on its own.
        cfg.validate(&["beta".into(), "N0".into()], InitLaw::Absent)
            .expect("a problem with no method has nothing method-dependent to check");
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

[method]
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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
"#).expect("arbitrary [fixed] param keys must still be accepted");
        assert!(cfg.problem.fixed.values.contains_key("some_param"),
            "[fixed] must keep flattening arbitrary param keys");
    }
}
