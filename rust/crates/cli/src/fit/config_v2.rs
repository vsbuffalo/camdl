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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    store: Option<String>,
    estimate: IndexMap<String, EstimateSpecV2>,
    fixed: FixedParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    method: Option<Method>,
    #[serde(default)]
    config: FitBackendConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    simulate: Option<SimulateSpec>,
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
    /// transform; a member's `[method.perturbation]` SD is on the log-ratio
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

    /// The run store's root (`camdl fit run`, and the commands that read a
    /// stored fit). A request to the store, never identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,

    /// The free parameters: what the inference algorithm estimates.
    pub estimate: IndexMap<String, EstimateSpecV2>,

    /// The fixed parameters: held constant during inference.
    /// estimate ∪ fixed must cover all model parameters.
    pub fixed: FixedParams,

    /// Time step and observation alignment. Default dt=1.0.
    pub config: FitBackendConfig,

    /// `[simulate]`: run-time choices about the simulated window that the
    /// config makes for the model. `from` overrides the model's
    /// `simulate { from }` (and is itself overridden by `simulate --from`).
    /// Not in the fit's hashed config blob: the resolved start enters the fit
    /// through the model digest, so two spellings of one start are one fit
    /// (`fit::cas::FIT_PROVENANCE_KEYS`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulate: Option<SimulateSpec>,

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

    /// Optional lineage metadata for the reader (not used by the runner).
    /// Not identity: normalized out of the fit level and the config lookup
    /// (`fit::cas::FIT_PROVENANCE_KEYS`, gh#583 item F).
    pub provenance: Option<FitProvenance>,

    /// Runtime-only: the path to the model **already compiled to IR**
    /// (`.ir.json`). `cmd_fit_run_v2` compiles `model.camdl` → IR exactly
    /// once up front and records the temp path here; every per-cell
    /// `FitRunConfig::build` then loads this pre-compiled IR instead of
    /// re-invoking camdlc per (cell × sweep point). Read through
    /// [`Problem::compiled_model`]. Never serialized — `model.camdl` remains
    /// the sole identity-bearing source path (the fit content hash hashes its
    /// bytes).
    #[serde(skip)]
    pub compiled_ir: Option<String>,

    /// Runtime-only: the digest of the stored compiled model `compiled_ir`
    /// must load to, set by `fit run` for each problem it runs. The model a
    /// method loads ([`super::runner::resolve_fit_parameters`]) is refused
    /// unless it has this digest, so a run never computes from a model other
    /// than the compile its identity names. Never serialized.
    #[serde(skip)]
    pub compiled_digest: Option<store::Key>,
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
            model, data, synthetic, fit_seeds, simplex_groups, store, estimate, fixed,
            method, config, simulate, scenario, enable, disable, ic_free, provenance,
        } = self;
        FitConfig {
            problem: Problem {
                model, data, synthetic, simplex_groups, store, estimate, fixed, config,
                simulate, scenario, enable, disable, ic_free, provenance, compiled_ir: None,
                compiled_digest: None,
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
            model, data, synthetic, simplex_groups, store, estimate, fixed, config,
            simulate, scenario, enable, disable, ic_free, provenance, compiled_ir: _,
            compiled_digest: _,
        } = self.problem.clone();
        let Inference { method, fit_seeds } = self.inference.clone();
        FitConfigWire {
            model, data, synthetic, fit_seeds, simplex_groups, store, estimate, fixed,
            method, config, simulate, scenario, enable, disable, ic_free, provenance,
        }
    }

    /// The one `[method]`, or the error `fit run` prints for a file that has
    /// none.
    pub fn method(&self) -> Result<&Method, String> {
        self.inference.method.as_ref().ok_or_else(|| {
            "this fit.toml declares no `[method]` table, so there is nothing to \
             run. It is a complete problem for `simulate --fit`, `pfilter --fit` \
             and `profile --fit`; to fit it, add\n  \
             [method]\n  \
             algorithm = \"pgas\"          # if2 | pgas | pmmh | mh | pfilter | nl-sbplx | nl-bobyqa | nl-lbfgs\n  \
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

/// `[simulate]`: the simulated window, as the config sets it for the model.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimulateSpec {
    /// Where the run starts: a model-time number, a date under a model
    /// `origin`, or an observation anchor (`"first_obs - 1 week"`), in the
    /// grammar of `simulate --to`. Overrides the model's `simulate { from }`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<TimeSpecToml>,
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
    /// Let the fit evaluate a forcing outside the range it is defined on (a
    /// data table past its data, a `bspline` past its window), holding its
    /// end value, with one warning per forcing; without it such a fit is
    /// refused (spec `docs/dev/temporal-architecture.md` §4.6). Keyed: a fit
    /// made under it and one refused without it are different computations,
    /// and a stored fit must not answer for a run that would be refused.
    /// `skip_serializing_if` keeps the common `false` out of the identity
    /// hash, so existing fits do not re-key. `fit run
    /// --allow-forcing-extrapolation` sets it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_forcing_extrapolation: bool,
}
fn default_backend() -> crate::args::types::ForwardBackend {
    crate::args::types::ForwardBackend::ChainBinomial
}
fn default_dt() -> f64 { 1.0 }
fn is_false(b: &bool) -> bool { !*b }
impl FitBackendConfig {
    /// The evaluator settings every model compiled for this fit carries
    /// (`CompiledModel::with_eval_settings`).
    pub fn eval_settings(&self) -> sim::EvalSettings {
        crate::env::eval_settings(self.allow_degenerate_rates)
    }
}

impl Default for FitBackendConfig {
    fn default() -> Self {
        FitBackendConfig {
            dt: default_dt(),
            allow_degenerate_rates: false,
            allow_forcing_extrapolation: false,
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

    /// The files bound to the model's external inputs: `NAME = "path"`
    /// supplies the model's `external("NAME")` table or `data =
    /// external("NAME")` forcing, as `--table NAME=path` does on the command
    /// line (which wins over this where both bind a name). Paths are relative
    /// to this config, as for `[data.observations]`. Only the files' bytes are
    /// identity (`FitProblemIdentity::tables`, by content, never by path); the
    /// skip keeps a config without the section serializing without it.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub tables: IndexMap<String, String>,
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
}

// ─── Synthetic data ──────────────────────────────────────────────────────────

// ─── Simplex groups ─────────────────────────────────────────────────────────

/// A group of estimated parameters that must form a probability simplex
/// (non-negative, summing to 1). See `FitConfigV2.simplex_groups` for
/// the full design.
///
/// CLI-side type: members are listed by name. At fit-config build time
/// names are resolved to model param indices, and each member's SD is
/// its `[method.perturbation]` entry or the default (`fit::steps`) — the runtime
/// `sim::inference::if2::SimplexGroup` carries indices + SDs on the
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
    /// generation, not of fitting — fit methods declare their own backends.
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

    /// Prior distribution. Required for Bayesian methods (PGAS, PMMH).
    /// Optional for MLE (IF2 ignores priors).
    ///
    /// Wire format is externally-tagged (matches the OCaml IR emission):
    ///   `prior = { log_normal = { mu = 0.0, sigma = 1.0 } }`
    ///
    /// gh#75: an explicit flat-prior opt-in is also recognized:
    ///   `prior = { flat = {} }`
    /// The flat variant is *only* meaningful in fit tomls — there is
    /// no DSL `~ flat(...)` syntax. Honored by Bayesian methods as
    /// the "I want flat priors here, on purpose" declaration. The
    /// runner emits provenance `flat_explicit` for each such param.
    ///
    /// See [`EstimatePriorSpec`] for the typed wrapper.
    #[serde(default)]
    pub prior: Option<EstimatePriorSpec>,

    /// Starting value: the base point a `starts = "single"` rule puts every
    /// chain at, and the point `"uniform"` keeps for chain 1. Omitted: the
    /// model's declared value, else a transform-aware draw within the bounds.
    /// It belongs to the run, not the problem: `fit_store::StartSource`
    /// keys it.
    #[serde(default)]
    pub start: Option<f64>,
}

/// One `[method.perturbation]` entry: IF2's random-walk SD for one estimated
/// parameter, in z, and whether the parameter sets only the initial state.
///
/// A bare number is the SD (`beta = 0.05`); a table takes `sd` and
/// `initial_state` and nothing else (`I0 = { sd = 0.174, initial_state =
/// true }`, `I0 = { initial_state = true }`). An entry that states neither is
/// refused when the steps are resolved ([`crate::fit::steps::resolve_steps`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerturbationSpec {
    /// The SD in z; `None` takes the map's default IF2 step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sd: Option<sim::inference::types::ZSd>,
    /// The parameter sets only the initial state, so IF2 perturbs it at
    /// t = 0 only ([`sim::inference::types::PerturbAt::TimeZeroOnly`]).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub initial_state: bool,
}

impl<'de> Deserialize<'de> for PerturbationSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use sim::inference::types::ZSd;
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = PerturbationSpec;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an SD in z (a number), or a table with `sd` and `initial_state`")
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(PerturbationSpec { sd: Some(ZSd::new(v).map_err(E::custom)?), initial_state: false })
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                self.visit_f64(v as f64)
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                self.visit_f64(v as f64)
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = PerturbationSpec { sd: None, initial_state: false };
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "sd" => out.sd = Some(map.next_value::<ZSd>()?),
                        "initial_state" => out.initial_state = map.next_value::<bool>()?,
                        other => return Err(serde::de::Error::unknown_field(other, &["sd", "initial_state"])),
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_any(V)
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

/// The values `[fixed] from_scenario = "name"` imports: the scenario's
/// composed `set` values with its composed `scale` factors applied, i.e. the
/// values `simulate --scenario name` runs at
/// ([`crate::params_resolver::scenario_values`], the one implementation of
/// "what values does a scenario give"). A scenario that scales a parameter it
/// does not set is refused: the import has no base value for the factor.
fn scenario_import(model: &ir::Model, scen_name: &str) -> Result<IndexMap<String, f64>, String> {
    crate::params_resolver::scenario_values(model, scen_name).map_err(|e| e.to_string())?
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
    /// `estimate` is the fit's `[estimate]` map. A scenario that sets or
    /// scales a parameter `[estimate]` lists is refused
    /// ([`crate::params_resolver::from_scenario_estimate_collision`]): the
    /// import would make that parameter both fixed and estimated.
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

        let estimated: Vec<&str> = estimate.keys().map(|k| k.as_str()).collect();
        if let Some(msg) = crate::params_resolver::from_scenario_estimate_collision(
            model, &scen_name, &estimated,
        )? {
            return Err(msg);
        }

        self.values.extend(scenario_import(model, &scen_name)?);
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

            return scenario_import(model, scen_name);
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
    #[expect(clippy::disallowed_methods, reason = "a library parser of `[fixed] from_file`, the params file a fit config names; `fit run` holds the values read in its problem identity (`FitProblemIdentity::fixed`), `--fit` of `simulate`, `pfilter` and `profile` is declared `#[input(identity)]`, and `fit diff`, `fit validate` and `design lhs` store nothing")]
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

    /// Each `[fixed]` name and how it reached the config — inline, from
    /// `from_file`, or from `from_scenario` — for the run's partition
    /// ([`crate::roles::FitFixedVia`]). Call before
    /// [`FixedParams::expand_from_scenario`], which folds a scenario's names
    /// into the inline values. An inline value over a file entry is inline.
    pub fn declarations(
        &self,
        model: &ir::Model,
    ) -> Result<Vec<(String, crate::roles::FitFixedVia)>, String> {
        use crate::roles::FitFixedVia;
        if let Some(scenario) = &self.from_scenario {
            return Ok(self.resolve_with_model(model)?
                .into_keys()
                .map(|n| (n, FitFixedVia::FromScenario(scenario.clone())))
                .collect());
        }
        let mut out: Vec<(String, FitFixedVia)> = match &self.from_file {
            Some(path) => FixedParams {
                from_file: Some(path.clone()),
                from_scenario: None,
                values: IndexMap::new(),
            }
            .resolve()?
            .into_keys()
            .filter(|n| !self.values.contains_key(n))
            .map(|n| (n, FitFixedVia::FromFile(path.into())))
            .collect(),
            None => Vec::new(),
        };
        out.extend(self.values.keys().map(|n| (n.clone(), FitFixedVia::Inline)));
        Ok(out)
    }
}

// ─── Method ─────────────────────────────────────────────────────────────────

/// Serialisation predicate for every method's `binomial` field.
///
/// Deliberately named for the VALUE, not for the default: absence in a stored
/// payload means BTPE, permanently, because that is what every run predating
/// the field used. If the default flips (gh#761) this must NOT follow it.
pub(crate) fn is_btpe(a: &sim::rng::BinomialAlgorithm) -> bool {
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
        /// Auto-runs at end-of-final-method; halving-ladder pfilter
        /// eval that warns when the MLE is discretization-dependent.
        /// Defaults to `enabled = true`. Set `enabled = false` on
        /// scout / smoke fits where the check is unnecessary.
        #[serde(default)]
        dt_check: DtCheckConfig,
        /// gh#747: which binomial sampler the chain-binomial draws use,
        /// `btpe` (default) or `btrs`. Identity-bearing and serialised only
        /// when `btrs`, for the reasons on `Algorithm::PGAS.binomial`.
        #[serde(default, skip_serializing_if = "is_btpe")]
        binomial: sim::rng::BinomialAlgorithm,
        /// `[method.perturbation]`: per estimated parameter, its random-walk
        /// SD in z and whether it sets only the initial state. A name left
        /// out takes the map's default IF2 step and is perturbed before
        /// every observation (`fit::steps::resolve_steps`).
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        perturbation: IndexMap<String, PerturbationSpec>,
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
        /// Selecting `btrs` changes the draws (a different rejection scheme
        /// accepts different values from the same stream), so this field is
        /// part of the method's identity (`fit_store::MethodIdentity`): two
        /// runs differing only in it are two runs. That is why it is a typed
        /// field and not an environment variable.
        ///
        /// `serde(default)` lets a `fit.toml` that does not write the field
        /// parse, as `btpe`. `skip_serializing_if` writes it only for `btrs`,
        /// so a written method (`method.toml`) names it only then. The
        /// predicate tests `== Btpe`, not `== default()`: if the default
        /// became `btrs`, a default-tracking predicate would make an absent
        /// field in an already-written method mean `btrs`.
        #[serde(default, skip_serializing_if = "is_btpe")]
        binomial: sim::rng::BinomialAlgorithm,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,

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
        /// post-burn-in). It changes the stored trajectories, not the
        /// chain's draws, and like every `[method]` key but the length and
        /// `starts` it is part of the run's method identity
        /// (`fit_store::MethodIdentity`), so a different count is a
        /// different run. Default: 200.
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
        /// Spelled `ancestor_sampling = false` in the `[method]` table or
        /// `--no-ancestor-sampling` on the CLI. Default: true.
        ///
        /// Disabling AS changes the sampled draws, so the field is part of the
        /// method's identity. `skip_serializing_if` writes it only when
        /// `false`; the predicate tests `== true` literally, not
        /// `== default()`, so an absent field in a written method means AS on
        /// whatever the default becomes (as for `binomial`).
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
        /// gh#747: which binomial sampler the chain-binomial draws use,
        /// `btpe` (default) or `btrs`. Identity-bearing and serialised only
        /// when `btrs`, for the reasons on `Algorithm::PGAS.binomial`.
        #[serde(default, skip_serializing_if = "is_btpe")]
        binomial: sim::rng::BinomialAlgorithm,
        /// `[method.proposal]`: per estimated parameter, the initial
        /// proposal SD in z, used as written. A name left out takes the
        /// map's default MH step (`fit::steps::resolve_steps`).
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        proposal: IndexMap<String, sim::inference::types::ZSd>,
    },

    /// Metropolis-Hastings on the deterministic ODE marginal likelihood
    /// (`p(y|θ, ODE_skeleton)` via `compute_ode_loglik`). Reuses the PMMH
    /// chain/adaptive-proposal/diagnostics machinery, swapping the
    /// particle-filter likelihood for the deterministic ODE evaluation —
    /// so it carries no `particles` (no PF). Bayesian posteriors on ODE /
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
        /// on mh methods).
        #[serde(default)]
        dt_check: DtCheckConfig,
        /// `[method.proposal]`, as under `pmmh`.
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        proposal: IndexMap<String, sim::inference::types::ZSd>,
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
        /// itself. **On by default** — the post-fit PFilter method is
        /// where prequential is needed and the proposal calls for
        /// it as a first-class output. Set `false` to skip the trace
        /// write (e.g. when running PFilter purely for a loglik SD).
        #[serde(default = "default_record_prequential")]
        record_prequential: bool,
        /// gh#747: which binomial sampler the chain-binomial draws use,
        /// `btpe` (default) or `btrs`. Identity-bearing and serialised only
        /// when `btrs`, for the reasons on `Algorithm::PGAS.binomial`.
        #[serde(default, skip_serializing_if = "is_btpe")]
        binomial: sim::rng::BinomialAlgorithm,
    },

    /// NLopt Sbplx (subspace-searching simplex) — deterministic MLE on the
    /// ODE-skeleton likelihood. Default for ODE-backend MLE; robust to
    /// boundary non-smoothness. Phase 1 of the ODE-inference proposal.
    #[serde(rename = "nl-sbplx")]
    NlSbplx(NloptConfig),

    /// NLopt BOBYQA — quadratic-trust-region MLE on the ODE-skeleton
    /// likelihood. Faster than Sbplx on smooth interior objectives but
    /// fails at parameter-bound boundaries.
    #[serde(rename = "nl-bobyqa")]
    NlBobyqa(NloptConfig),

    /// NLopt L-BFGS — the gradient deterministic MLE on the ODE-skeleton
    /// likelihood. Its objective supplies `∇_θ log p(y | θ, ODE)` from the
    /// forward sensitivities (`sim::inference::ode_grad::det_grad`), so the
    /// model has to be one that gradient can be taken of; the gradient
    /// capability gate refuses the rest at config-validation time, naming
    /// `nl-sbplx` as the derivative-free alternative.
    #[serde(rename = "nl-lbfgs")]
    NlLbfgs(NloptConfig),
}

/// Shared config for the three NLopt deterministic MLE methods
/// (`nl-sbplx`, `nl-bobyqa`, `nl-lbfgs`). All three read identical knobs;
/// the variant tag picks which NLopt algorithm runs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NloptConfig {
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
    /// -inf); such a chain reports a `-inf` log-likelihood and loses the
    /// winner comparison to any finite chain, and when every chain lands
    /// there the method refuses (gh#916). For such models, narrow the bounds.
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
    ///
    /// A `pfilter` method is the exception: it scores one θ, so it takes
    /// [`Method::resolve_pfilter_point`]'s rule instead and never defaults to
    /// a draw.
    pub fn resolve_starts(
        &mut self,
        problem: &Problem,
        model: &ir::Model,
    ) -> Result<StartsResolution, String> {
        if matches!(self.algorithm, Algorithm::PFilter { .. }) {
            return self.resolve_pfilter_point(problem);
        }
        if let Some(declared) = &self.starts {
            return Ok(StartsResolution::Declared(declared.clone()));
        }
        let names: Vec<String> = problem.estimate.keys().cloned().collect();
        let resolved = default_starts(&names, &problem.estimate, model);
        self.starts = Some(resolved.rule());
        Ok(resolved)
    }

    /// The θ a `pfilter` method scores must be determined by the file, not
    /// drawn behind the user's back: a `start` for every estimated parameter
    /// (the `single` rule, which is the default here when they are all
    /// given), a sourced rule (`from_mle`, `from_params`, `from_posterior`),
    /// or `from_prior` asked for by name. The bounds-based spread rules
    /// (`uniform_unconstrained`, `lhs`, `uniform`) draw a spread of chain
    /// starts a one-θ filter has no use for, and are refused.
    fn resolve_pfilter_point(&mut self, problem: &Problem) -> Result<StartsResolution, String> {
        let without_start: Vec<&str> = problem.estimate.iter()
            .filter(|(_, spec)| spec.start.is_none())
            .map(|(name, _)| name.as_str())
            .collect();
        let remedy = "Give every estimated parameter a `start` in [estimate], name a point \
                      with `starts = { from_mle = \"@fit\" }` or `{ from_params = \
                      \"theta.toml\" }`, or ask for one prior draw with `starts = \"from_prior\"`.";
        let missing = || format!(
            "a `pfilter` method scores the log-likelihood at one θ, and this file does not \
             determine it: {} {} no `start` in [estimate]. {remedy}",
            without_start.join(", "),
            if without_start.len() == 1 { "has" } else { "have" });
        match &self.starts {
            Some(rule @ (ChainStarts::Point(Point::FromMle { .. } | Point::FromParams { .. })
                | ChainStarts::Spread(Spread::FromPosterior { .. } | Spread::FromPrior))) => {
                Ok(StartsResolution::Declared(rule.clone()))
            }
            Some(rule @ ChainStarts::Point(Point::Declared)) if without_start.is_empty() => {
                Ok(StartsResolution::Declared(rule.clone()))
            }
            Some(ChainStarts::Point(Point::Declared)) => Err(missing()),
            Some(rule @ ChainStarts::Spread(
                Spread::UniformUnconstrained | Spread::Lhs | Spread::Uniform,
            )) => Err(format!(
                "`starts = \"{}\"` draws a spread of chain starts within the bounds, and a \
                 `pfilter` method scores the log-likelihood at one θ. {remedy}",
                rule.tag())),
            None if without_start.is_empty() => {
                self.starts = Some(ChainStarts::Point(Point::Declared));
                Ok(StartsResolution::DefaultDeclaredPoint)
            }
            None => Err(missing()),
        }
    }

}

/// The chain-starts rule a run that names none draws under (§3.4), for the
/// parameters `names` it estimates: `from_prior` when each one's resolved
/// prior (`estimate` over the model's `~`) is a distribution the chains can
/// be drawn from, `uniform_unconstrained` otherwise. `fit run`
/// ([`Method::resolve_starts`]) and `profile` (over a cell's estimated set)
/// both default through here.
pub fn default_starts(
    names: &[String],
    estimate: &IndexMap<String, EstimateSpecV2>,
    model: &ir::Model,
) -> StartsResolution {
    use super::priors_precedence::{resolve_priors_with_precedence, PriorSource};
    let without_prior: Vec<String> =
        resolve_priors_with_precedence(names, estimate, model)
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
    if without_prior.is_empty() {
        StartsResolution::DefaultFromPrior
    } else {
        StartsResolution::DefaultUniformUnconstrained { without_prior }
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
    /// Absent on a `pfilter` method, and every estimated parameter declares
    /// a `start`: the filter scores that point.
    DefaultDeclaredPoint,
}

impl StartsResolution {
    pub fn rule(&self) -> ChainStarts {
        match self {
            StartsResolution::Declared(rule) => rule.clone(),
            StartsResolution::DefaultFromPrior => ChainStarts::from_prior(),
            StartsResolution::DefaultUniformUnconstrained { .. } => {
                ChainStarts::uniform_unconstrained()
            }
            StartsResolution::DefaultDeclaredPoint => ChainStarts::Point(Point::Declared),
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
            StartsResolution::DefaultDeclaredPoint => format!(
                "{} — default for pfilter: every estimated parameter declares a start",
                ChainStarts::Point(Point::Declared).describe()
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
    /// The binomial sampler this method's particle filter draws with (gh#747).
    /// The methods that run no chain-binomial particle filter (`mh`, nl-*)
    /// have no such field and draw nothing through it; they report BTPE, the
    /// sampler any incidental filter they build would have used before the
    /// field existed.
    pub fn binomial(&self) -> sim::rng::BinomialAlgorithm {
        match self {
            Algorithm::IF2 { binomial, .. }
            | Algorithm::PGAS { binomial, .. }
            | Algorithm::PMMH { binomial, .. }
            | Algorithm::PFilter { binomial, .. } => *binomial,
            Algorithm::Mh { .. }
            | Algorithm::NlSbplx(_) | Algorithm::NlBobyqa(_) | Algorithm::NlLbfgs(_) => {
                sim::rng::BinomialAlgorithm::Btpe
            }
        }
    }

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
            Algorithm::PFilter  { .. } => FitAlgorithm::Pfilter,
            Algorithm::NlSbplx  { .. } => FitAlgorithm::NlSbplx,
            Algorithm::NlBobyqa { .. } => FitAlgorithm::NlBobyqa,
            Algorithm::NlLbfgs  { .. } => FitAlgorithm::NlLbfgs,
        }
    }

    /// Simulation backend the method runs on. The (algorithm, backend)
    /// pair is set by the user in fit.toml and validated against
    /// `methods::METHODS`; this accessor returns whichever backend was
    /// declared so dispatch and provenance can branch on it.
    pub fn backend(&self) -> crate::run_meta::InferenceBackend {
        match self {
            Algorithm::IF2      { backend, .. }
            | Algorithm::PGAS    { backend, .. }
            | Algorithm::PMMH    { backend, .. }
            | Algorithm::Mh      { backend, .. }
            | Algorithm::PFilter { backend, .. } => *backend,
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) => c.backend,
        }
    }

    pub fn requires_priors(&self) -> bool {
        matches!(self, Algorithm::PGAS { .. } | Algorithm::PMMH { .. } | Algorithm::Mh { .. })
    }

    pub fn chains(&self) -> usize {
        match self {
            Algorithm::IF2 { chains, .. } => *chains,
            Algorithm::PGAS { chains, .. } => *chains,
            Algorithm::PMMH { chains, .. } => *chains,
            Algorithm::Mh { chains, .. } => *chains,
            Algorithm::PFilter { .. } => 1,
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) => c.chains,
        }
    }

    /// Overwrite this method's sampler and output knobs from the CLI.
    ///
    /// Every field these flags set is part of the run's method identity
    /// (`fit_store::MethodIdentity`), so they are written into the in-memory
    /// method before that identity is taken: a different flag is then a
    /// different run, as a different toml value is. A flag applied after the
    /// identity was taken would let two runs that differ in it share a run id,
    /// and the second be served the first's result. `--starts` takes the same
    /// route through `Method::starts`.
    ///
    /// A `None` argument leaves the field as the toml declared it, so a run
    /// with no CLI overrides keys identically to a bare `fit run` of the same
    /// file.
    ///
    /// A flag this method does not read is refused, never dropped: which
    /// methods read which flag is the one table [`OverrideFlag::methods`], and
    /// a flag that only means something beside another setting (`--adapt-start`
    /// needs adaptation on; `--max-tree-depth` needs NUTS) is refused when the
    /// overrides leave that setting off.
    pub fn apply_cli_overrides(&mut self, cli: &CliMethodOverrides) -> Result<(), String> {
        let method = self.method_kind();
        for flag in cli.passed() {
            if !flag.methods().contains(&method) {
                return Err(flag.refusal(method));
            }
        }
        self.write_cli_overrides(cli)?;
        self.refuse_inert_cli_overrides(cli)
    }

    /// Write each override into the field it names, on the arms that have
    /// that field. The table in [`OverrideFlag::methods`] must agree with
    /// these arms exactly — a test runs every flag on every method through
    /// this function and checks that it moves the method iff the table lists
    /// it.
    fn write_cli_overrides(&mut self, cli: &CliMethodOverrides) -> Result<(), String> {
        // ── Sampler and output overrides (gh#540) ──
        // Each of these used to be written into the `*MethodOpts` struct at the
        // dispatch site, AFTER the identity was taken from this one — so a run
        // differing only in `--n-trajectories` or `--no-nuts` shared a `run_id`
        // with the previous run and was served its result. Writing them here
        // means the identity sees them, and the dispatch site has nothing left
        // to override: `*MethodOpts::from_algorithm` reads what is written below.
        match self {
            Algorithm::PGAS {
                max_tree_depth, trajectory_warmup,
                csmc_sweeps_per_nuts, n_trajectories, dense_mass, use_nuts, binomial,
                ancestor_sampling, ..
            } => {
                if let Some(d) = cli.max_tree_depth { *max_tree_depth = d; }
                if let Some(w) = cli.trajectory_warmup { *trajectory_warmup = w; }
                if let Some(s) = cli.csmc_sweeps_per_nuts { *csmc_sweeps_per_nuts = s; }
                if let Some(n) = cli.n_trajectories { *n_trajectories = n; }
                if cli.diagonal_mass { *dense_mass = false; }
                if cli.no_nuts { *use_nuts = false; }
                // gh#747: resolved INTO the method, so the flag is keyed into
                // the method's identity rather than acting as an untracked
                // side channel. Two runs differing only here get different
                // addresses.
                if let Some(b) = cli.binomial { *binomial = b; }
                if cli.no_ancestor_sampling { *ancestor_sampling = false; }
            }
            Algorithm::PMMH { adapt, adapt_start, binomial, .. } => {
                if cli.no_adapt { *adapt = false; }
                if let Some(s) = cli.adapt_start { *adapt_start = s; }
                if let Some(b) = cli.binomial { *binomial = b; }
            }
            Algorithm::Mh { adapt, adapt_start, dt_check, backend, .. } => {
                if cli.no_adapt { *adapt = false; }
                if let Some(s) = cli.adapt_start { *adapt_start = s; }
                if cli.no_dt_check { dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { dt_check.n_halvings = n; }
                resolve_dt_check_strict(dt_check, *backend, cli.dt_check_strict)?;
            }
            Algorithm::IF2 { cooling_target_iters, gate, dt_check, backend, binomial, .. } => {
                if let Some(n) = cli.cooling_target_iters { *cooling_target_iters = n; }
                if let Some(b) = cli.binomial { *binomial = b; }
                if let Some(db) = cli.decibans_thresh { gate.decibans_thresh = db; }
                if cli.no_dt_check { dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { dt_check.n_halvings = n; }
                resolve_dt_check_strict(dt_check, *backend, cli.dt_check_strict)?;
            }
            Algorithm::PFilter { record_ancestry, record_prequential, binomial, .. } => {
                if let Some(b) = cli.binomial { *binomial = b; }
                // One-way overrides to true: the toml can opt out
                // (`record_prequential = false`), the flag opts back in.
                if cli.record_ancestry { *record_ancestry = true; }
                if cli.record_prequential { *record_prequential = true; }
            }
            Algorithm::NlSbplx(cfg) | Algorithm::NlBobyqa(cfg) | Algorithm::NlLbfgs(cfg) => {
                if let Some(db) = cli.decibans_thresh { cfg.gate.decibans_thresh = db; }
                if cli.no_dt_check { cfg.dt_check.enabled = false; }
                if let Some(n) = cli.dt_check_halvings { cfg.dt_check.n_halvings = n; }
                let backend = cfg.backend;
                resolve_dt_check_strict(&mut cfg.dt_check, backend, cli.dt_check_strict)?;
            }
        }
        Ok(())
    }

    /// An IF2 or nl-* method running one chain: its convergence gate (chain
    /// agreement and the decibans spread) compares chains, so it judges
    /// nothing, and its settings are inert.
    fn gates_on_one_chain(&self) -> bool {
        match self {
            Algorithm::IF2 { chains, .. } => *chains < 2,
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) => c.chains < 2,
            _ => false,
        }
    }

    /// The second half of the refusal: a flag the method reads, but which the
    /// method's settings — after the overrides above — leave with nothing to
    /// do. Each would otherwise run, re-key and change nothing.
    fn refuse_inert_cli_overrides(&self, cli: &CliMethodOverrides) -> Result<(), String> {
        let method = self.method_name();
        let inert = |flag: &str, why: &str| -> Result<(), String> {
            Err(format!("{flag} has no effect on this `{method}` method: {why}. Drop the flag."))
        };
        if let Algorithm::PGAS { use_nuts: false, .. } = self {
            for (given, flag) in [(cli.max_tree_depth.is_some(), "--max-tree-depth"),
                                  (cli.diagonal_mass, "--diagonal-mass")] {
                if given {
                    return inert(flag, "it tunes the NUTS θ-move, and `use_nuts` is off \
                        (`--no-nuts`, or `use_nuts = false` in [method]), so the θ-move is \
                        Metropolis-within-Gibbs");
                }
            }
        }
        if let Algorithm::PMMH { adapt: false, .. } | Algorithm::Mh { adapt: false, .. } = self {
            if cli.adapt_start.is_some() {
                return inert("--adapt-start", "it sets when proposal adaptation begins, and \
                    adaptation is off (`--no-adapt`, or `adapt = false` in [method])");
            }
        }
        if cli.decibans_thresh.is_some() && self.gates_on_one_chain() {
            return inert("--decibans-thresh", "the gate judges the spread between chains, \
                and this method runs one chain");
        }
        let dt_check = match self {
            Algorithm::IF2 { dt_check, .. } | Algorithm::Mh { dt_check, .. } => Some(dt_check),
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) => Some(&c.dt_check),
            _ => None,
        };
        if let Some(dt_check) = dt_check {
            if !dt_check.enabled {
                for (given, flag) in [(cli.dt_check_halvings.is_some(), "--dt-check-halvings"),
                                      (cli.dt_check_strict, "--dt-check-strict")] {
                    if given {
                        return inert(flag, "the dt-check does not run (`--no-dt-check`, or \
                            `[method.dt_check] enabled = false`)");
                    }
                }
            }
        }
        Ok(())
    }
}

/// One `fit run` flag that writes into the method, i.e. one field of
/// [`CliMethodOverrides`].
///
/// The point is the table in [`OverrideFlag::methods`]: which methods read the
/// flag. A flag passed to any other method is refused by
/// [`Algorithm::apply_cli_overrides`], so no explicit option is ever silently
/// dropped. [`CliMethodOverrides::passed`] destructures the struct without `..`,
/// so a field added there does not compile until it names its flag here, and
/// the match in `methods` does not compile until the flag has a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideFlag {
    MaxTreeDepth,
    TrajectoryWarmup,
    CsmcSweepsPerNuts,
    NTrajectories,
    DiagonalMass,
    NoNuts,
    NoAncestorSampling,
    NoAdapt,
    AdaptStart,
    CoolingTargetIters,
    DecibansThresh,
    NoDtCheck,
    DtCheckHalvings,
    DtCheckStrict,
    Binomial,
    RecordAncestry,
    RecordPrequential,
}

impl OverrideFlag {
    /// Every flag; the table test walks it.
    #[cfg(test)]
    pub const ALL: [OverrideFlag; 17] = [
        OverrideFlag::MaxTreeDepth, OverrideFlag::TrajectoryWarmup,
        OverrideFlag::CsmcSweepsPerNuts, OverrideFlag::NTrajectories,
        OverrideFlag::DiagonalMass, OverrideFlag::NoNuts,
        OverrideFlag::NoAncestorSampling, OverrideFlag::NoAdapt,
        OverrideFlag::AdaptStart, OverrideFlag::CoolingTargetIters,
        OverrideFlag::DecibansThresh, OverrideFlag::NoDtCheck,
        OverrideFlag::DtCheckHalvings, OverrideFlag::DtCheckStrict,
        OverrideFlag::Binomial, OverrideFlag::RecordAncestry,
        OverrideFlag::RecordPrequential,
    ];

    /// The flag as typed on the command line.
    pub fn spelling(self) -> &'static str {
        match self {
            OverrideFlag::MaxTreeDepth => "--max-tree-depth",
            OverrideFlag::TrajectoryWarmup => "--trajectory-warmup",
            OverrideFlag::CsmcSweepsPerNuts => "--csmc-sweeps-per-nuts",
            OverrideFlag::NTrajectories => "--n-trajectories",
            OverrideFlag::DiagonalMass => "--diagonal-mass",
            OverrideFlag::NoNuts => "--no-nuts",
            OverrideFlag::NoAncestorSampling => "--no-ancestor-sampling",
            OverrideFlag::NoAdapt => "--no-adapt",
            OverrideFlag::AdaptStart => "--adapt-start",
            OverrideFlag::CoolingTargetIters => "--cooling-target-iters",
            OverrideFlag::DecibansThresh => "--decibans-thresh",
            OverrideFlag::NoDtCheck => "--no-dt-check",
            OverrideFlag::DtCheckHalvings => "--dt-check-halvings",
            OverrideFlag::DtCheckStrict => "--dt-check-strict",
            OverrideFlag::Binomial => "--binomial",
            OverrideFlag::RecordAncestry => "--record-ancestry",
            OverrideFlag::RecordPrequential => "--record-prequential",
        }
    }

    /// The methods that read this flag. Every other method refuses it.
    pub fn methods(self) -> &'static [crate::run_meta::FitAlgorithm] {
        use crate::run_meta::FitAlgorithm::*;
        match self {
            OverrideFlag::MaxTreeDepth
            | OverrideFlag::TrajectoryWarmup
            | OverrideFlag::CsmcSweepsPerNuts
            | OverrideFlag::NTrajectories
            | OverrideFlag::DiagonalMass
            | OverrideFlag::NoNuts
            | OverrideFlag::NoAncestorSampling => &[Pgas],
            OverrideFlag::Binomial => &[Pgas, Pmmh, If2, Pfilter],
            OverrideFlag::NoAdapt | OverrideFlag::AdaptStart => &[Pmmh, Mh],
            OverrideFlag::CoolingTargetIters => &[If2],
            OverrideFlag::DecibansThresh => &[If2, NlSbplx, NlBobyqa, NlLbfgs],
            OverrideFlag::NoDtCheck
            | OverrideFlag::DtCheckHalvings
            | OverrideFlag::DtCheckStrict => &[If2, Mh, NlSbplx, NlBobyqa, NlLbfgs],
            OverrideFlag::RecordAncestry | OverrideFlag::RecordPrequential => &[Pfilter],
        }
    }

    /// The error for this flag on a method that does not read it.
    fn refusal(self, method: crate::run_meta::FitAlgorithm) -> String {
        let accepted: Vec<&str> = self.methods().iter().map(|m| m.as_str()).collect();
        format!(
            "{} does not apply to this fit's method, `{}`: only {} read{} it, so it would \
             change nothing. Drop the flag, or run a method that reads it.",
            self.spelling(),
            method.as_str(),
            accepted.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", "),
            if accepted.len() == 1 { "s" } else { "" },
        )
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
pub struct CliMethodOverrides {
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
    /// `--cooling-target-iters` (IF2): iterations over which the cooling
    /// fraction is reached. Changes the perturbation schedule and therefore
    /// the MLE.
    pub cooling_target_iters: Option<usize>,
    /// `--decibans-thresh`: overrides `gate.decibans_thresh` on methods that
    /// carry a `GateConfig` (IF2, nl-sbplx, nl-bobyqa, nl-lbfgs). The applied gate is
    /// persisted in the leaf (`resolved_gate`), so it is stored output.
    pub decibans_thresh: Option<f64>,
    /// `--no-dt-check`: disables the post-fit Richardson dt-check, whose
    /// result is stored in `fit_state.toml.dt_check`. Applies to the
    /// methods that run one: if2, mh, nl-* (gh#726).
    pub no_dt_check: bool,
    /// `--dt-check-halvings`: dt-check ladder depth. Same methods as
    /// `no_dt_check`.
    pub dt_check_halvings: Option<usize>,
    /// `--dt-check-strict`: use the strict warning threshold. Resolved to a
    /// concrete `dt_check.threshold_nats` when applied (gh#730), because the
    /// threshold is STORED in `fit_state.toml.dt_check` — it was treated as
    /// leaf-byte-neutral abort policy, but it selects the stored verdict and
    /// threshold whenever the TOML leaves `threshold_nats` unset.
    pub dt_check_strict: bool,
    /// gh#747: `--binomial`. `None` leaves the method's own value (or its
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
/// A TOML-set `threshold_nats` leaves the flag nothing to select, so the flag
/// is refused there rather than silently ignored.
fn resolve_dt_check_strict(
    dt_check: &mut DtCheckConfig,
    backend: crate::run_meta::InferenceBackend,
    strict: bool,
) -> Result<(), String> {
    if !strict {
        return Ok(());
    }
    if let Some(declared) = dt_check.threshold_nats {
        return Err(format!(
            "--dt-check-strict selects the dt-check's default strict threshold, and this \
             method declares its own (`[method.dt_check] threshold_nats = {declared}`), \
             which the flag would not change. Drop the flag, or edit `threshold_nats`."
        ));
    }
    dt_check.threshold_nats = Some(super::dt_check::default_threshold_for_backend(backend, true));
    Ok(())
}

impl CliMethodOverrides {
    /// The flags this set carries, in [`OverrideFlag::ALL`] order.
    ///
    /// Destructured without `..`: a field added to the struct is a compile
    /// error here until it names its [`OverrideFlag`], which in turn cannot
    /// compile without a row in [`OverrideFlag::methods`].
    pub fn passed(&self) -> Vec<OverrideFlag> {
        let CliMethodOverrides {
            max_tree_depth, trajectory_warmup, csmc_sweeps_per_nuts, n_trajectories,
            diagonal_mass, no_nuts, no_ancestor_sampling, no_adapt, adapt_start,
            cooling_target_iters, decibans_thresh, no_dt_check, dt_check_halvings,
            dt_check_strict, binomial, record_ancestry, record_prequential,
        } = self;
        [
            (max_tree_depth.is_some(), OverrideFlag::MaxTreeDepth),
            (trajectory_warmup.is_some(), OverrideFlag::TrajectoryWarmup),
            (csmc_sweeps_per_nuts.is_some(), OverrideFlag::CsmcSweepsPerNuts),
            (n_trajectories.is_some(), OverrideFlag::NTrajectories),
            (*diagonal_mass, OverrideFlag::DiagonalMass),
            (*no_nuts, OverrideFlag::NoNuts),
            (*no_ancestor_sampling, OverrideFlag::NoAncestorSampling),
            (*no_adapt, OverrideFlag::NoAdapt),
            (adapt_start.is_some(), OverrideFlag::AdaptStart),
            (cooling_target_iters.is_some(), OverrideFlag::CoolingTargetIters),
            (decibans_thresh.is_some(), OverrideFlag::DecibansThresh),
            (*no_dt_check, OverrideFlag::NoDtCheck),
            (dt_check_halvings.is_some(), OverrideFlag::DtCheckHalvings),
            (*dt_check_strict, OverrideFlag::DtCheckStrict),
            (binomial.is_some(), OverrideFlag::Binomial),
            (*record_ancestry, OverrideFlag::RecordAncestry),
            (*record_prequential, OverrideFlag::RecordPrequential),
        ]
        .into_iter()
        .filter_map(|(given, flag)| given.then_some(flag))
        .collect()
    }

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
fn default_cooling_target_iters() -> usize { sim::inference::if2::CoolingSchedule::POMP_TARGET_ITERS }

// PGAS defaults
fn default_max_tree_depth() -> usize { 10 }
fn default_csmc_sweeps_per_nuts() -> usize { 1 }
fn default_n_trajectories() -> usize { 200 }
fn default_dense_mass() -> bool { true }
fn default_use_nuts() -> bool { true }
fn default_ancestor_sampling() -> bool { true }
/// The `skip_serializing_if` predicate for `Algorithm::PGAS::ancestor_sampling`.
/// Deliberately `*b` (i.e. `== true`), NOT `== default_ancestor_sampling()`:
/// absence in a stored payload must mean AS-on permanently, whatever the
/// default may later become — see the field's doc comment.
fn ancestor_sampling_is_on(b: &bool) -> bool { *b }

/// Validate a `burnin_dt` coarse warm-up step (gh#396 follow-on) against the
/// fit-wide `dt` and the observation streams, for the `mh` ODE path: coarsening
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
/// Default to recording the prequential trace under a `pfilter` method.
/// Per the 2026-04-20 prequential proposal, every fit
/// should produce a `PrequentialTrace` as a first-class output —
/// downstream `camdl compare` consumes the per-step log-score / CRPS
/// / PIT samples that this flag toggles. Cost is one extra
/// per-particle obs draw per observation, on the first replicate
/// only; the trace is auto-written to `prequential.{tsv,json}` in
/// the method dir. Set `record_prequential = false` in `[method]`
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

/// Compound convergence gate: chain agreement (Â) AND inter-chain
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
/// Auto-runs at the end of `camdl fit run`'s final method (after the
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
    /// inherit from the method's `loglik_eval.n_particles` (so the
    /// dt-check matches the gate's clean-eval budget).
    #[serde(default)]
    pub n_particles: Option<usize>,
    /// Replicate count per ladder-rung evaluation, combined via
    /// `combine`. Default `None` → inherit from the method's
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
    /// from the method's `loglik_eval.combine`. Almost always
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
             generation, not a fit-wide config — fit methods declare their own \
             `backend`. Move it under your `[synthetic]` block, or remove it for \
             a real-data fit.\n  See `camdl docs fit-toml`."
                .into(),
        );
    }
    Ok(())
}

/// Refuse `algorithm = "nuts"` under `[method]` before the strict parse, so the
/// user gets the same steer as the `--algorithm` boundary
/// ([`super::methods::NUTS_STEER`]) instead of serde's bare "unknown variant".
fn detect_nuts_algorithm(contents: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        // Malformed TOML surfaces from the strict parse with its own message.
        Err(_) => return Ok(()),
    };
    let algorithm = value
        .get("method")
        .and_then(|m| m.get("algorithm"))
        .and_then(toml::Value::as_str);
    if algorithm == Some("nuts") {
        return Err(format!(
            "[method]: unknown algorithm \"nuts\".\n  {}\n  See `camdl docs fit-toml`.",
            super::methods::NUTS_STEER.replace('\n', "\n  ")
        ));
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

/// Refuse the removed `[estimate.<name>].transform` key (support-keyed
/// transforms, D2) before the strict parse, so the user is told the key is
/// gone and what replaces it rather than a bare serde "unknown field".
fn detect_removed_estimate_transform(contents: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let Some(estimate) = value.get("estimate").and_then(|e| e.as_table()) else {
        return Ok(());
    };
    for (name, entry) in estimate {
        if entry.get("transform").is_some() {
            return Err(format!(
                "estimate.{name}: `transform` was removed: the transform follows from the \
                 parameter's kind and bounds. Delete the line. See `camdl docs fit-toml`."));
        }
    }
    Ok(())
}

/// One `[estimate]` entry that writes a step key: the step is a setting of
/// the method, in `[method.perturbation]` or `[method.proposal]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyStepKey {
    pub param: String,
    /// `rw_sd` as written, a natural-scale SD.
    pub rw_sd: Option<f64>,
    pub perturb_only_at_t0: bool,
    /// Whether the parameter is a member of a `[[simplex_groups]]` entry,
    /// whose IF2 `rw_sd` was read on its log-ratio coordinate as written.
    pub simplex_member: bool,
}

/// Every `[estimate]` entry of a fit config that writes `rw_sd` or
/// `perturb_only_at_t0`, and the file's `[method] algorithm`.
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyStepKeys {
    pub algorithm: Option<String>,
    pub entries: Vec<LegacyStepKey>,
}

/// The `[estimate]` entries of `raw` that write a step key, or `None` when
/// none does.
pub fn detect_legacy_step_keys(raw: &toml::Value) -> Option<LegacyStepKeys> {
    let estimate = raw.get("estimate")?.as_table()?;
    let members: BTreeSet<&str> = raw
        .get("simplex_groups")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|g| g.get("params").and_then(toml::Value::as_array))
        .flatten()
        .filter_map(toml::Value::as_str)
        .collect();
    let entries: Vec<LegacyStepKey> = estimate
        .iter()
        .filter_map(|(name, entry)| {
            let rw_sd = entry.get("rw_sd");
            let flag = entry.get("perturb_only_at_t0");
            if rw_sd.is_none() && flag.is_none() {
                return None;
            }
            Some(LegacyStepKey {
                param: name.clone(),
                rw_sd: rw_sd.and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64))),
                perturb_only_at_t0: flag.and_then(toml::Value::as_bool).unwrap_or(false),
                simplex_member: members.contains(name.as_str()),
            })
        })
        .collect();
    if entries.is_empty() {
        return None;
    }
    let algorithm = raw.get("method").and_then(|m| m.get("algorithm")).and_then(toml::Value::as_str).map(str::to_string);
    Some(LegacyStepKeys { algorithm, entries })
}

impl LegacyStepKeys {
    /// The refusal: where each key moved for this file's method, and the
    /// formula that converts a natural-scale `rw_sd` to the SD in z each
    /// method takes.
    pub fn message(&self) -> String {
        let fmt = |x: f64| x.to_string();
        let which = match &self.algorithm {
            Some(a) => format!("This file's method is `{a}`."),
            None => "This file names no method.".to_string(),
        };
        let mut m = String::from(
            "[estimate] sets step sizes, which are settings of the method:\n  \
             `rw_sd` (a natural-scale step) and `perturb_only_at_t0` moved to [method],\n  \
             in z, the coordinate each method moves in. ",
        );
        m.push_str(&which);
        m.push_str("\n  Delete both keys from [estimate]");
        let width = self.entries.iter().map(|e| e.param.len()).max().unwrap_or(0);
        let if2_block = || -> String {
            let mut b = String::from("    [method.perturbation]\n");
            for e in &self.entries {
                let value = match (e.rw_sd, e.simplex_member) {
                    (Some(rw), true) => fmt(rw),
                    (Some(_), false) => "<sd_z>".to_string(),
                    (None, _) => String::new(),
                };
                let entry = match (e.perturb_only_at_t0, value.is_empty()) {
                    (true, false) => format!("{{ sd = {value}, initial_state = true }}"),
                    (true, true) => "{ initial_state = true }".to_string(),
                    (false, false) => value,
                    (false, true) => continue,
                };
                let note = match (e.rw_sd, e.simplex_member) {
                    (Some(rw), false) => format!("# rw_sd {}", fmt(rw)),
                    (Some(_), true) => "# a simplex member: its rw_sd was already on its log-ratio scale".to_string(),
                    (None, _) => String::new(),
                };
                let line = format!("    {:width$} = {entry:<40}{note}", e.param);
                b.push_str(line.trim_end());
                b.push('\n');
            }
            b
        };
        let proposal_block = || -> String {
            let mut b = String::from("    [method.proposal]\n");
            for e in self.entries.iter().filter(|e| e.rw_sd.is_some()) {
                b.push_str(&format!("    {:width$} = <5 × sd_z>  # rw_sd {}\n", e.param, fmt(e.rw_sd.unwrap_or_default())));
            }
            b
        };
        let any_rw = self.entries.iter().any(|e| e.rw_sd.is_some());
        let any_flag = self.entries.iter().any(|e| e.perturb_only_at_t0);
        let formula = "\n  where sd_z = rw_sd / |dθ/dz|, the natural-scale step divided by the slope\n  \
            of the parameter's map at its start θ₀ (its `start`, else the model's value,\n  \
            else the middle of its box): θ₀ − lo on [lo, ∞); (θ₀ − lo)(hi − θ₀)/(hi − lo)\n  \
            on a probability or a box that starts at 0; θ₀·ln(θ₀/lo)·ln(hi/θ₀)/ln(hi/lo)\n  \
            on a positive kind's box [lo, hi] with lo > 0; 1 on an unbounded real.\n  \
            This gives the old step at the start only, since the new step is fixed in z.";
        match self.algorithm.as_deref() {
            Some("if2") => {
                m.push_str(" and add:\n\n");
                m.push_str(&if2_block());
                if any_rw {
                    m.push_str(formula);
                }
            }
            Some("pmmh") | Some("mh") => {
                if any_rw {
                    m.push_str(" and add:\n\n");
                    m.push_str(&proposal_block());
                    m.push_str(formula);
                    m.push_str("\n  The old initial proposal was five times that z step; the new table is\n  \
                        the proposal SD itself.");
                } else {
                    m.push('.');
                }
                if any_flag {
                    m.push_str("\n  `perturb_only_at_t0` is IF2's timing, which this method does not read: delete it.");
                }
            }
            Some(other) => {
                m.push_str(&format!(
                    ".\n  `{other}` reads neither key: it takes no per-parameter step, so the lines \
                     had no effect."));
            }
            None => {
                m.push_str(", and add the table of the method that runs the file:\n\n  for `if2`:\n");
                m.push_str(&if2_block());
                if any_rw {
                    m.push_str("\n  for `pmmh` or `mh`:\n");
                    m.push_str(&proposal_block());
                    m.push_str(formula);
                }
            }
        }
        m.push_str(
            "\n\n  Leaving a parameter out uses the method's default step instead, which is a\n  \
             different step. See `camdl docs fit-toml`, \"Step sizes\".",
        );
        m
    }
}

/// Refuse a live fit config whose `[estimate]` writes `rw_sd` or
/// `perturb_only_at_t0`, naming where each moved.
fn detect_legacy_step_keys_in(contents: &str) -> Result<(), String> {
    let Ok(raw) = toml::from_str::<toml::Value>(contents) else { return Ok(()) };
    match detect_legacy_step_keys(&raw) {
        Some(keys) => Err(keys.message()),
        None => Ok(()),
    }
}

/// Remove `rw_sd` and `perturb_only_at_t0` from every `[estimate]` entry of
/// `raw`, a stored copy of a config written before they moved to `[method]`.
/// Returns the notice naming what was removed, or `None` when there was
/// nothing to remove. For readers of a stored run only: a live config with
/// the keys is refused ([`FitConfig::from_toml_str_named`]).
pub fn set_aside_legacy_step_keys(raw: &mut toml::Value, what: &str) -> Option<String> {
    let keys = detect_legacy_step_keys(raw)?;
    let estimate = raw.get_mut("estimate")?.as_table_mut()?;
    for (_, entry) in estimate.iter_mut() {
        if let Some(t) = entry.as_table_mut() {
            t.remove("rw_sd");
            t.remove("perturb_only_at_t0");
        }
    }
    let removed: Vec<String> = keys
        .entries
        .iter()
        .map(|e| {
            let k: Vec<&str> = [e.rw_sd.map(|_| "rw_sd"), e.perturb_only_at_t0.then_some("perturb_only_at_t0")]
                .into_iter()
                .flatten()
                .collect();
            format!("{} ({})", e.param, k.join(", "))
        })
        .collect();
    Some(format!(
        "note: {what} was written before step sizes moved to [method]; its [estimate] keys \
         {} are set aside, and the run is read without them.",
        removed.join(", ")
    ))
}

/// Refuse a step table under a method that does not read it, naming the
/// table the method has instead. Run before the typed parse, since the
/// `nl-*` methods' settings deny unknown fields and would otherwise refuse
/// it without saying where it belongs.
fn detect_misplaced_step_table(contents: &str) -> Result<(), String> {
    let Ok(raw) = toml::from_str::<toml::Value>(contents) else { return Ok(()) };
    let Some(method) = raw.get("method") else { return Ok(()) };
    let Some(algorithm) = method.get("algorithm").and_then(toml::Value::as_str) else { return Ok(()) };
    let has = |key: &str| method.get(key).is_some();
    if has("proposal") && !matches!(algorithm, "pmmh" | "mh") {
        return Err(format!(
            "`[method.proposal]` is the initial proposal SD of `pmmh` and `mh`, and this \
             file's method is `{algorithm}`. IF2's per-parameter setting is \
             `[method.perturbation]`; `pgas`, `pfilter` and the `nl-*` optimizers have none. \
             See `camdl docs fit-toml`."));
    }
    if has("perturbation") && algorithm != "if2" {
        return Err(format!(
            "`[method.perturbation]` is IF2's random-walk perturbation, and this file's \
             method is `{algorithm}`. The initial proposal SD of `pmmh` and `mh` is \
             `[method.proposal]`; `pgas`, `pfilter` and the `nl-*` optimizers have none. \
             See `camdl docs fit-toml`."));
    }
    Ok(())
}

/// Refuse the removed `[config] obs_alignment` key (D43,
/// `docs/dev/temporal-architecture.md` §4.5) before the strict parse, so the
/// user is told the setting is gone and why, rather than a bare serde
/// "unknown field". Where a period's boundaries may lie relative to the step
/// grid is a capability of the method, not a choice.
fn detect_removed_obs_alignment(contents: &str) -> Result<(), String> {
    let value: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    if value.get("config").and_then(|c| c.get("obs_alignment")).is_some() {
        return Err(
            "[config] `obs_alignment` was removed: how an observed period relates to the \
             step grid is not a setting. if2, pfilter and pmmh step exactly to every \
             period's start and end; pgas steps only on its grid t_start + k·dt and \
             refuses a period whose start or end lies off it. Delete the line. See \
             `camdl docs fit-toml`."
                .into(),
        );
    }
    Ok(())
}

/// The `[stages]` rejection (proposal 2026-09-08-workflow-first-fit-config,
/// §5). A file that still carries the method map is refused at load with the
/// rewrite spelled out: the last-declared method becomes the file's `[method]`,
/// every other method goes in its own file, `init` is `starts`, and a chained
/// `init_mle` — which started every chain at the upstream point estimate and
/// made R̂ uninformative — is handed back to the author as a decision, because
/// the file cannot name a handle that does not exist until the upstream has
/// run. No silent conversion and no compatibility path.
///
/// Declaration order is read off the raw text (`[stages.<name>]` headers in
/// order), because `toml::Table` sorts its keys and the method that should
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
             `simulate --fit`, `pfilter --fit` and `profile --fit`; \
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
            "  `init = \"survey_top_k\"` (with `survey_path` / `survey_top_k_n`) was removed \
             with\n  \
             the command that wrote its input. Use `starts = \"from_prior\"`, or run a short \
             fit and\n  \
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
    // Each step table is serialized only when non-empty.
    match method.algorithm {
        Algorithm::IF2 { .. } => { known.insert("perturbation".into()); }
        Algorithm::PMMH { .. } | Algorithm::Mh { .. } => { known.insert("proposal".into()); }
        _ => {}
    }
    for (key, value) in raw_method {
        if known.contains(key) {
            continue;
        }
        // The keys the `[stages]` → `[method]` split retired name their
        // replacement rather than a bare "unknown key".
        let retired = match key.as_str() {
            "init" => Some(match value.as_str() {
                Some("survey_top_k") => "`init = \"survey_top_k\"` was removed with the \
                     command that wrote its input. Use `starts = \"from_prior\"`, or run a \
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
                 `survey_top_k_n` companions and the command that wrote their input. Use \
                 `starts = \"from_prior\"`, or a `from_posterior` from a short run."
                    .to_string(),
            ),
            "tempering" => Some(
                "`tempering` was removed with PGAS parallel tempering (gh#789): every \
                 PGAS chain samples the posterior directly, as the ladder's cold rung \
                 did. Delete the line. Fits stored before the removal are not reused \
                 and cannot be resumed; they run again."
                    .to_string(),
            ),
            "rho" => Some(
                "`rho` was removed with correlated pseudo-marginal PMMH: `pmmh` runs \
                 the bootstrap particle filter with independent likelihood \
                 evaluations, as it did with `rho` unset. Delete the line. PMMH fits \
                 stored before the removal are not reused and cannot be resumed; they \
                 run again."
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

/// Refuse the `[method]` keys this file's method leaves inert. Run by
/// `fit run` only, on the file it is about to run — never by the shared
/// parser, which also reads stored fits' `fit.resolved.toml` for
/// `fit table` and handle resolution: a fit stored before a key became
/// refused must stay listable.
pub fn refuse_inert_method_keys_in(contents: &str, method: &Method) -> Result<(), String> {
    let raw: toml::Value = toml::from_str(contents).map_err(|e| format!("parse error: {e}"))?;
    match raw.get("method").and_then(toml::Value::as_table) {
        Some(raw_method) => refuse_inert_method_keys(raw_method, &method.algorithm),
        None => Ok(()),
    }
}

/// Whether `[method]` in this fit.toml source spells `key` — as opposed to
/// the method carrying it at its serde default. Used where a default and an
/// explicit request must be told apart (an explicit `use_nuts = true` the
/// model cannot honour is refused; the default falls back with a notice).
pub fn method_declares_key(contents: &str, key: &str) -> bool {
    table_declares_key(contents, "method", key)
}

/// The `[config] dt` this fit.toml source writes, or `None` when it leaves
/// the step to its default. For the one dt rule (`crate::dt_rule`), which
/// must tell a written step from a defaulted one.
pub fn written_config_dt(contents: &str) -> Option<f64> {
    let raw: toml::Value = toml::from_str(contents).ok()?;
    let dt = raw.get("config")?.get("dt")?;
    dt.as_float().or_else(|| dt.as_integer().map(|i| i as f64))
}

/// Whether table `[table]` of this fit.toml source spells `key`, as opposed
/// to the parsed config carrying it at its serde default.
pub fn table_declares_key(contents: &str, table: &str, key: &str) -> bool {
    toml::from_str::<toml::Value>(contents)
        .ok()
        .and_then(|v| v.get(table).and_then(|m| m.as_table()).map(|t| t.contains_key(key)))
        .unwrap_or(false)
}


/// The NUTS θ-move was asked for and the model has no rate gradients to run
/// it on (no estimated parameter reaches a transition rate), so PGAS would run
/// Metropolis-within-Gibbs instead. Explicitly requested: refused. The
/// default: a notice, then the fallback.
pub fn nuts_unavailable(use_nuts_declared: bool, nuts_tuning: &[&str]) -> Result<String, String> {
    let why = "the model has no rate gradients to run it on: no parameter reaches a \
               transition rate, so there is nothing for NUTS to differentiate";
    if !use_nuts_declared && !nuts_tuning.is_empty() {
        return Err(format!(
            "{} tune{} the NUTS θ-move, and PGAS will update θ by Metropolis-within-Gibbs: \
             {why}. Drop {}.",
            nuts_tuning.join(", "),
            if nuts_tuning.len() == 1 { "s" } else { "" },
            if nuts_tuning.len() == 1 { "it" } else { "them" }));
    }
    if use_nuts_declared {
        Err(format!(
            "`use_nuts = true` asks PGAS for the NUTS θ-move, and {why}. Set \
             `use_nuts = false` (Metropolis-within-Gibbs), or delete the line."
        ))
    } else {
        Ok(format!(
            "PGAS will update θ by Metropolis-within-Gibbs, not NUTS (the default): {why}. \
             Set `use_nuts = false` to say so and silence this."
        ))
    }
}

/// Refuse a `[method]` key the method knows but the method's own settings
/// leave with nothing to do, so the file never states a setting the run
/// does not use. The CLI flags that write the same fields get the same
/// refusal (`Algorithm::apply_cli_overrides`).
fn refuse_inert_method_keys(
    raw_method: &toml::map::Map<String, toml::Value>,
    algorithm: &Algorithm,
) -> Result<(), String> {
    let inert = |key: &str, why: &str| -> Result<(), String> {
        Err(format!(
            "`{key}` in [method] has no effect here: {why}. Delete the line, or turn the \
             setting it tunes back on. See `camdl docs fit-toml`."
        ))
    };
    match algorithm {
        Algorithm::PGAS { use_nuts: false, .. } => {
            for key in ["max_tree_depth", "dense_mass"] {
                if raw_method.contains_key(key) {
                    return inert(key, "it tunes the NUTS θ-move, and `use_nuts = false` makes \
                        the θ-move Metropolis-within-Gibbs");
                }
            }
        }
        Algorithm::PMMH { adapt: false, .. } | Algorithm::Mh { adapt: false, .. } => {
            if raw_method.contains_key("adapt_start") {
                return inert("adapt_start", "it sets when proposal adaptation begins, and \
                    `adapt = false` turns adaptation off");
            }
        }
        _ => {}
    }
    if raw_method.contains_key("gate") && algorithm.gates_on_one_chain() {
        return inert("[method.gate]", "the gate compares chains, and this method runs \
            one chain");
    }
    // `[method.dt_check] enabled = false`: the check does not run, so every
    // other key in the table tunes nothing.
    if let Some(dt) = raw_method.get("dt_check").and_then(toml::Value::as_table) {
        if dt.get("enabled").and_then(toml::Value::as_bool) == Some(false) {
            if let Some(key) = dt.keys().find(|k| k.as_str() != "enabled") {
                return inert(&format!("dt_check.{key}"),
                    "`[method.dt_check] enabled = false` turns the dt-check off");
            }
        }
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
        detect_removed_estimate_transform(contents)?;
        detect_legacy_step_keys_in(contents)?;
        detect_misplaced_step_table(contents)?;
        detect_removed_obs_alignment(contents)?;
        detect_nuts_algorithm(contents)?;
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
    /// segment (`fit.resolved.toml`): its relative `[model] camdl` /
    /// `[data.observations]` paths were written against the *original* fit.toml
    /// directory, so anchoring them at the archive's own location resolves
    /// `../data/x.tsv` to `results/fits/data/x.tsv` — a path that never existed
    /// (gh#652). The segment's sidecar records `source_config`, the original
    /// file's path relative to the segment, so its directory is known even when
    /// that file has moved or changed; pass `<segment>/<source_config>` here and
    /// every path resolves as it did at fit time.
    #[expect(clippy::disallowed_methods, reason = "a library parser of the fit config at the path its caller passes, through `FitConfig::load`: a `--fit` argument, declared `#[input(identity)]` on `SimulateArgs`, `PfilterArgs` and `ProfileArgs`, or `fit diff` and `simulate --design-from`, which store nothing")]
    pub fn load_anchored_at(
        read_path: &std::path::Path,
        anchor_toml: &std::path::Path,
    ) -> Result<Self, String> {
        let path = read_path.to_string_lossy();
        let contents = std::fs::read_to_string(read_path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        Self::parse_anchored_at(&contents, read_path, anchor_toml)
    }

    /// [`load_anchored_at`](Self::load_anchored_at) over `contents`, the text
    /// already read from `read_path`: for a reader that must parse exactly the
    /// text it read once (`fit run`, which stores that text too).
    pub fn parse_anchored_at(
        contents: &str,
        read_path: &std::path::Path,
        anchor_toml: &std::path::Path,
    ) -> Result<Self, String> {
        let path = read_path.to_string_lossy();
        let mut config = FitConfig::from_toml_str_named(contents, &path)
            .map_err(|e| format!("error in {}:\n{}", path, e))?;

        // gh#307: warn (do not error) on absolute file references — checked on
        // the as-written paths, before the relative-path resolution below turns
        // every relative path absolute.
        for w in config.problem.absolute_path_warnings() {
            crate::out::detached().say(format_args!("{w}"));
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
    /// (`rate_state_grad` / `projection_state_grad`)? The one `ode`-backend
    /// gradient cell does — `nl-lbfgs` (MLE), driven by the ODE
    /// forward-sensitivity gradient (`ode_grad::det_grad`). Every other (algorithm, backend) cell — IF2, PGAS, PMMH, `mh`, `nl-sbplx`,
    /// `nl-bobyqa`, the particle filter — is gradient-free with respect to the
    /// state, so the model can compile lean (`camdlc --no-state-grad`), dropping
    /// the dense ~O(G^3) Jacobian that dominates coupled-model IR. Consumed by
    /// `cmd_fit_run_v2` to pick the compile mode; the resulting bit is folded into
    /// the IR-cache key, so a lean entry is never reused for a gradient fit (and
    /// run identity is gradient-independent, so lean vs full hash the same model).
    pub fn needs_state_grad(&self) -> bool {
        use crate::run_meta::{FitAlgorithm, InferenceBackend};
        self.inference.method.as_ref().is_some_and(|m| {
            m.algorithm.method_kind() == FitAlgorithm::NlLbfgs
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
        // loglik AND give the swarm spread at t=0. PGAS and the ODE-MLE
        // optimizers score every obs unconditionally; `pfilter` / `pmmh` do
        // condition, but only have the spread when the MODEL declares an
        // `init { }` law — hence `init_law` here.
        // Running ic_free without either property would silently compute the
        // UNCONDITIONAL likelihood while the banner claims conditioning.
        if self.problem.likelihood_objective().skips_first_observation() {
            if let Err(msg) =
                super::methods::validate_ic_free(algorithm.method_kind(), init_law)
            {
                return Err(format!("[method]: {}", msg));
            }
        }

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

        // A sampler keeps every `thin`-th post-burn-in draw. `thin = 0` has no
        // such reading: PGAS divides by it in its startup banner, after the
        // leaf is claimed, and PMMH/MH keep a single draw per chain.
        if let Algorithm::PGAS { thin: Some(0), .. }
            | Algorithm::PMMH { thin: Some(0), .. }
            | Algorithm::Mh { thin: Some(0), .. } = algorithm
        {
            return Err(
                "[method]: thin must be ≥ 1 (got 0). `thin = k` keeps every k-th \
                 post-burn-in draw; `thin = 1` keeps them all."
                    .to_string(),
            );
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
            let tag = crate::out::paint::warning_tag();
            crate::out::detached().say(format_args!("{} fit declares simplex_groups, \
                but the `{}` method does not currently honour the \
                simplex constraint — members will be perturbed \
                independently and rely on the model to enforce sum = 1 \
                indirectly.", tag, algorithm.method_name()));
        }

        // Bayesian prior presence is checked separately by
        // `validate_priors_present(&ir_priors)`, which needs the model IR
        // in scope to honor the gh#73 precedence fallback. validate()
        // itself only needs parameter names, so the prior check is
        // factored out — production callers do both.
        Ok(())
    }

    /// Refuse `nl-lbfgs` on a model whose ODE gradient cannot be taken —
    /// here, at config-validation time, rather than partway through a search.
    ///
    /// `nl-lbfgs` is the only method whose objective calls `det_grad`
    /// (`fit::nlopt_method::score_with_gradient`), and `det_grad` runs
    /// `preflight_gradient_ode` — the ONE place that answers "can this model be
    /// fit by the ODE gradient?" — before any gradient is taken. Running that
    /// same gate here, against the same model, converts a mid-search failure
    /// into a refusal before a leaf is claimed: the user is told what is wrong
    /// and which method to reach for instead, and no half-written method is left
    /// behind. The rules are not restated; the gate's own sentence is quoted.
    ///
    /// Like [`Self::validate_priors_present`], this is factored out of
    /// [`Self::validate`] because it needs the model IR in scope; production
    /// callers run both.
    ///
    /// **What the parameter values are for.** `preflight_gradient_ode` reads
    /// `params` in exactly one place: to place the model's scheduled effects
    /// (`EffectPlan::build`), i.e. to resolve a parametric `at [t_seed]` fire
    /// time. That is a question about the model's structure, not about where
    /// the optimizer will end up — and a fire time that reads an *estimated*
    /// parameter is refused by name whatever its value — so any finite value
    /// serves an estimated parameter here: its `start`, the midpoint of its fit
    /// or model box, else a placeholder. The gate must not be skipped for want
    /// of a value: a fit that restates no `bounds` or `start` for a parameter
    /// the model declares with `in [lo, hi]` is the common case, and skipping
    /// it here pushed the refusal into every chain at run time.
    ///
    /// **What it does NOT do** is re-report parameter-resolution problems. If a
    /// value is still missing, or the model does not compile, this returns `Ok`
    /// and leaves the report to `FitRunConfig::build`, whose resolver says
    /// which parameter and which tier — a better message than anything phrased
    /// in terms of gradients.
    /// Refuse an estimated hyperparameter whose hierarchical leaves are held,
    /// under the samplers that read a prior (PMMH, MH, PGAS).
    ///
    /// Their log prior sums over the estimated parameters only, so a held
    /// leaf's density p(leaf | hyper) never enters it: the hyperparameter is
    /// informed by its own prior alone and its posterior is that prior. Until
    /// the held leaves' terms are included, such a fit is refused rather than
    /// run. IF2 and the optimizers read no prior and are unaffected.
    pub fn validate_held_hierarchical_leaves(&self, model: &ir::Model) -> Result<(), String> {
        use crate::run_meta::FitAlgorithm;
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        let kind = method.algorithm.method_kind();
        if !matches!(kind, FitAlgorithm::Pmmh | FitAlgorithm::Mh | FitAlgorithm::Pgas) {
            return Ok(());
        }
        let estimate = &self.problem.estimate;
        let mut held: Vec<(String, Vec<String>)> = Vec::new();
        for p in &model.parameters {
            if estimate.contains_key(&p.name) {
                continue;
            }
            let Some(h) = p.hierarchical() else { continue };
            let mut refs = std::collections::HashSet::new();
            for arg in h.args.values() {
                sim::inference::gradient_capability::collect_param_refs(arg, &mut refs);
            }
            let mut hypers: Vec<String> =
                refs.into_iter().filter(|r| estimate.contains_key(r)).collect();
            hypers.sort();
            if !hypers.is_empty() {
                held.push((p.name.clone(), hypers));
            }
        }
        if held.is_empty() {
            return Ok(());
        }
        let lines: Vec<String> = held.iter()
            .map(|(leaf, hypers)| format!("  {leaf} (held), prior on {}", hypers.join(", ")))
            .collect();
        Err(format!(
            "[estimate]: an estimated hyperparameter has hierarchical leaves that are \
             held, not estimated:\n{}\n\
             Under `{}` the log prior sums over the estimated parameters only, so the held \
             leaves' prior terms p(leaf | hyperparameter) are not yet included and the \
             hyperparameter would not be informed by them: its posterior would be its own \
             prior. Estimate the leaves too, or hold the hyperparameter.",
            lines.join("\n"), kind.as_str()))
    }

    /// Proposal 2026-10-27-estimable-bsplines §6.4: IF2 and the NLopt
    /// optimizers read no prior, and the likelihood depends on a `smooth`
    /// forcing's scale τ and modes z only through w = … + (τ/κ)Mz, so an
    /// estimated τ beside them is a ridge with no maximum. Refused, naming the
    /// scale, its forcing and the remedy.
    pub fn validate_smooth_scale(&self, model: &ir::Model) -> Result<(), String> {
        use crate::run_meta::FitAlgorithm;
        use ir::time_func::{BSpline, SplineCoefs, TimeFuncKind};
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        let kind = method.algorithm.method_kind();
        if !matches!(kind, FitAlgorithm::If2 | FitAlgorithm::NlSbplx | FitAlgorithm::NlBobyqa | FitAlgorithm::NlLbfgs) {
            return Ok(());
        }
        for tf in &model.time_functions {
            let TimeFuncKind::BSpline(BSpline { coefs: SplineCoefs::Smooth { scale, .. }, .. }) = &tf.kind else {
                continue;
            };
            let mut refs = std::collections::HashSet::new();
            sim::inference::gradient_capability::collect_param_refs(scale, &mut refs);
            let mut estimated: Vec<&String> = refs.iter().filter(|p| self.problem.estimate.contains_key(*p)).collect();
            estimated.sort();
            if let Some(p) = estimated.first() {
                return Err(format!(
                    "[estimate]: `{p}` is the scale of the smooth forcing `{}`, and `{}` reads no \
                     prior: the likelihood depends on the scale and the modes only through the \
                     coefficients they make, so an estimated scale beside them has no maximum. \
                     Fix `{p}` (the fit is then unpenalized maximum likelihood over the modes), \
                     or estimate explicit coefficients, or fit with pgas or pmmh.",
                    tf.name,
                    kind.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Proposal 2026-10-27-estimable-bsplines §6.3-§6.4: the level, slope and
    /// modes of a `smooth` forcing are coordinates the prior is built in — the
    /// modes standard normal by construction, w linear in all three — so the
    /// samplers must move them on the identity scale. A finite search box (the
    /// model's `in [lo, hi]` or `[estimate]` bounds) gives such a parameter the
    /// scaled logit transform, which bends that geometry, and for a mode also
    /// truncates the law the whitening assumes. The scale is sampled on the
    /// log scale, and a box there is a hard wall: `log_jacobian` is −∞ outside
    /// it, so a leapfrog step that crosses it counts as a divergence (the
    /// prior-only runs: 10-31 divergences per chain boxed, 0 unboxed). Refused under the
    /// methods that sample a posterior, naming the parameters and the remedy:
    /// declare them without a range so they are searched over (−∞, ∞).
    /// Forcing the identity on a boxed parameter instead would leave the
    /// declared box unenforced, the silent escape of gh#66.
    pub fn validate_smooth_coordinates(&self, model: &ir::Model) -> Result<(), String> {
        use crate::run_meta::FitAlgorithm;
        use ir::time_func::{BSpline, SplineCoefs, TimeFuncKind};
        use sim::inference::types::Transform;
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        let kind = method.algorithm.method_kind();
        if !matches!(kind, FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Mh) {
            return Ok(());
        }
        for tf in &model.time_functions {
            let TimeFuncKind::BSpline(BSpline { coefs: SplineCoefs::Smooth { level, slope, scale, modes, .. }, .. }) = &tf.kind else {
                continue;
            };
            let mut refs = std::collections::HashSet::new();
            for e in [level, slope, scale].into_iter().chain(modes) {
                sim::inference::gradient_capability::collect_param_refs(e, &mut refs);
            }
            let mut bent: Vec<String> = Vec::new();
            for name in refs {
                let Some(spec) = self.problem.estimate.get(&name) else { continue };
                let Some(p) = model.parameters.iter().find(|p| p.name == name) else { continue };
                let Ok(bx) = super::runner::resolve_search_box(&name, spec.bounds, p) else { continue };
                let Ok(pkind) = super::runner::estimated_kind(p) else { continue };
                let Ok(transform) = super::runner::derive_transform_with_bounds(pkind, bx) else { continue };
                // The identity on (−∞, ∞), and θ = eᶻ on [0, ∞), keep the
                // coordinates the smooth prior is built in; every map onto a
                // box with another finite end bends them.
                match transform {
                    Transform::Identity => {}
                    Transform::LowerBounded { lo: 0.0 } => {}
                    Transform::LowerBounded { .. }
                    | Transform::UpperBounded { .. }
                    | Transform::Interval { .. }
                    | Transform::LogInterval { .. } => {
                        let (lo, hi) = transform.bounds();
                        bent.push(format!("  {name}: search box [{lo}, {hi}]"));
                    }
                }
            }
            if !bent.is_empty() {
                bent.sort();
                return Err(format!(
                    "[estimate]: these parameters feed the level, slope, scale or modes of the \
                     smooth forcing `{}` and have a finite search box:\n{}\n\
                     `{}` would sample each through the transform that keeps it in its box: the \
                     scaled logit bends the coordinates the smooth prior is built in (the modes \
                     are standard normal by construction), and a boxed log scale is a wall the \
                     sampler's steps cross as divergences. Declare them without a range, e.g. \
                     `z[mode] : real dim 1 ~ normal(mu = 0.0, sigma = 1.0)` and \
                     `tau : positive dim 1 ~ exponential(rate = 5.99)`, and give no `bounds` \
                     in [estimate], so the level, slope and modes are sampled on the identity \
                     scale and the scale on the log scale over (0, ∞).",
                    tf.name, bent.join("\n"), kind.as_str()));
            }
        }
        Ok(())
    }

    /// Refuse, under PMMH and MH, an estimated hierarchical leaf whose search
    /// box cuts into the support of its prior p(leaf | hyperparameter).
    ///
    /// The samplers score the leaf against the prior density restricted to
    /// the box but not renormalised by P(box | hyperparameter). For a fixed
    /// prior that constant does not move and the omission is harmless; for a
    /// hierarchical leaf it depends on the hyperparameter, so leaving it out
    /// biases the hyperparameter (`mu ~ normal(0, 1)` with four leaves
    /// `a_k ~ normal(mu, 1)` boxed to `[1, 2]` gives a posterior mean of 1.18
    /// for `mu` under a flat likelihood, where the truth is 0). PGAS already
    /// refuses hierarchical priors, and IF2 and the optimizers read no prior.
    ///
    /// The box compared is the effective domain the transform maps onto
    /// ([`sim::inference::types::Transform::bounds`]); an unbounded `real` or
    /// `instant` leaf has the whole line and cannot cut into a support.
    pub fn validate_hierarchical_leaf_boxes(&self, model: &ir::Model) -> Result<(), String> {
        use crate::run_meta::FitAlgorithm;
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        let kind = method.algorithm.method_kind();
        if !matches!(kind, FitAlgorithm::Pmmh | FitAlgorithm::Mh) {
            return Ok(());
        }
        let mut cut: Vec<String> = Vec::new();
        for (name, spec) in &self.problem.estimate {
            // A fit.toml prior displaces the model's hierarchical one.
            if spec.prior.is_some() {
                continue;
            }
            let Some(p) = model.parameters.iter().find(|p| p.name == *name) else { continue };
            let Some(h) = p.hierarchical() else { continue };
            let Ok(bx) = super::runner::resolve_search_box(name, spec.bounds, p) else { continue };
            let Ok(kind) = super::runner::estimated_kind(p) else { continue };
            let Ok(transform) = super::runner::derive_transform_with_bounds(kind, bx) else { continue };
            let kept = transform.bounds();
            if box_cuts_into_support(kept, hierarchical_support(h)) {
                cut.push(format!(
                    "  {name}: bounds [{}, {}], prior {}",
                    kept.0, kept.1, h.kind));
            }
        }
        if cut.is_empty() {
            return Ok(());
        }
        Err(format!(
            "[estimate]: a hierarchical parameter's search bounds cut into the support \
             of its prior:\n{}\n\
             Under `{}` the prior is not renormalised to the bounds. For a hierarchical \
             parameter the missing constant, P(bounds | hyperparameter), depends on the \
             hyperparameter, so leaving it out would bias the hyperparameter. Widen the \
             bounds to the prior's support, or remove the hierarchical prior.",
            cut.join("\n"), kind.as_str()))
    }

    pub fn validate_gradient_capability(&self, model: &ir::Model) -> Result<(), String> {
        let Some(method) = &self.inference.method else {
            return Ok(());
        };
        if method.algorithm.method_kind() != crate::run_meta::FitAlgorithm::NlLbfgs {
            return Ok(());
        }

        // A named scenario composes its own enable/disable lists, and composing
        // them lives inside `params_resolver::resolve_parameters` rather than in
        // a function this could call. Rather than fork that walk, defer: a
        // scenario fit is checked by `det_grad` itself on the first evaluation,
        // which the objective's latch turns into the same sentence one method
        // later. Lateness, not a wrong answer.
        if self.problem.scenario.is_some() {
            return Ok(());
        }

        let mut seeded = model.clone();
        // Toggleable interventions default OFF (spec §14.4), so the model this
        // gate sees must be the one the run will integrate — otherwise a fit
        // that never enables its `interventions { }` block would be refused for
        // a scheduled effect that is not in its trajectory. Same seam the
        // resolver uses.
        if let Err(e) = crate::util::apply_scenario_filter(
            &mut seeded, &self.problem.enable, &self.problem.disable,
        ) {
            // An unresolvable enable/disable name is the resolver's error to
            // report, in its own words.
            let _ = e;
            return Ok(());
        }
        let fixed = self.problem.fixed.resolve().unwrap_or_default();
        for p in &mut seeded.parameters {
            if p.value.resolved_value().is_some() {
                continue;
            }
            let spec = self.problem.estimate.get(&p.name);
            let v = fixed
                .get(&p.name)
                .copied()
                .or_else(|| spec.and_then(|s| s.start))
                .or_else(|| spec.and_then(|s| s.bounds).map(|(lo, hi)| 0.5 * (lo + hi)))
                // The model's own `in [lo, hi]`, which a fit.toml need not restate.
                .or_else(|| p.value.bounds().map(|(lo, hi)| 0.5 * (lo + hi)))
                // An estimated parameter with no box anywhere: the value only
                // places fire times, and a fire time that reads an estimated
                // parameter is refused by name whatever the value is.
                .or_else(|| spec.map(|_| 0.5));
            match v {
                Some(v) => p.value = p.value.with_value(v),
                // A fixed parameter the resolver has not placed: its error to
                // report, in its own words.
                None => return Ok(()),
            }
        }
        let Ok(compiled) = sim::CompiledModel::new(seeded) else {
            return Ok(());
        };
        let compiled = compiled.with_eval_settings(self.problem.config.eval_settings());
        let estimated: std::collections::HashSet<&str> =
            self.problem.estimate.keys().map(|s| s.as_str()).collect();
        let default_params = compiled.default_params.clone();
        match sim::inference::gradient_capability::preflight_gradient_ode(
            &compiled,
            &default_params,
            &estimated,
        ) {
            Ok(()) => Ok(()),
            Err(e) => Err(format!(
                "[method]: `algorithm = \"nl-lbfgs\"` optimizes the ODE marginal \
                 likelihood using its gradient, and the gradient capability check \
                 refuses this model:\n\n  \
                 {e}\n\n  \
                 `algorithm = \"nl-sbplx\"` optimizes the same ODE marginal \
                 likelihood without a gradient, and has no such requirement."
            )),
        }
    }

    /// gh#75: Validate that every estimated parameter has a prior available
    /// from at least one source — either this fit toml's
    /// `[estimate.<name>.prior]` block, or the model IR's `~` syntax —
    /// when the method is Bayesian (PGAS / PMMH / MH).
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
    /// unconditioned likelihood.
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

/// One end of a prior family's support: a known value, or an end set by an
/// expression of other parameters (a hierarchical `uniform`'s `lower` or
/// `upper`), which no fixed box can be shown to contain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SupportEnd {
    At(f64),
    Varies,
}

/// The support of a hierarchical prior family, `(lower end, upper end)`.
pub(crate) fn hierarchical_support(
    h: &ir::parameter::HierarchicalPrior,
) -> (SupportEnd, SupportEnd) {
    use ir::parameter::HierarchicalKind as K;
    use SupportEnd::*;
    let end = |arg: &str| match h.args.get(arg) {
        Some(ir::expr::Expr::Const(c)) => At(c.value),
        _ => Varies,
    };
    // Exhaustive over the families (no `_`): a new family must state its
    // support here.
    match h.kind {
        K::Normal => (At(f64::NEG_INFINITY), At(f64::INFINITY)),
        K::LogNormal | K::HalfNormal | K::Gamma | K::Exponential => (At(0.0), At(f64::INFINITY)),
        K::Beta => (At(0.0), At(1.0)),
        K::Uniform => (end("lower"), end("upper")),
    }
}

/// Does the box `[lo, hi]` leave out part of the support? An end set by an
/// expression counts as cut into unless the box is unbounded on that side.
pub(crate) fn box_cuts_into_support(
    (lo, hi): (f64, f64),
    (s_lo, s_hi): (SupportEnd, SupportEnd),
) -> bool {
    let lo_cuts = match s_lo {
        SupportEnd::At(v) => lo > v,
        SupportEnd::Varies => lo.is_finite(),
    };
    let hi_cuts = match s_hi {
        SupportEnd::At(v) => hi < v,
        SupportEnd::Varies => hi.is_finite(),
    };
    lo_cuts || hi_cuts
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
            ChainStarts::Spread(Spread::FromPosterior { source, .. })
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

/// What a likelihood evaluation of this problem scores, as set by the
/// top-level `ic_free` key.
///
/// Every command that scores a [`Problem`] — `fit run`, `profile --fit`,
/// `pfilter --fit` — reads it from [`Problem::likelihood_objective`], so a
/// profile curve is on the same scale as the fit it belongs to. The two objectives differ by `log p(y₁ | θ)`, a term that
/// varies with θ, so neither can be compared with, or ranked against, the
/// other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LikelihoodObjective {
    /// `log p(y_{1:T} | θ)`: every observation is scored (`ic_free` absent or
    /// false).
    #[default]
    Full,
    /// `log p(y_{2:T} | y₁, θ)` (`ic_free = true`): the filter weights and
    /// resamples at y₁, which conditions the initial state on it, but y₁'s
    /// own term is not added to the log-likelihood.
    ConditionalOnFirstObservation,
}

impl LikelihoodObjective {
    /// Whether the filter leaves y₁'s term out of the accumulated
    /// log-likelihood — the value of `SMCConfig::skip_first_obs_from_loglik`
    /// and `IF2Config::skip_first_obs_from_loglik`.
    pub fn skips_first_observation(self) -> bool {
        matches!(self, LikelihoodObjective::ConditionalOnFirstObservation)
    }
}

impl Problem {
    /// The compiled model this problem's runs read: `compiled_ir`, or a
    /// `model.camdl` that is itself compiled IR (a config that names an
    /// `.ir.json`, as unit tests do). A model source with nothing compiled is
    /// an error, never compiled here: the command compiles it, under its
    /// compile settings, before anything reads it.
    pub fn compiled_model(&self) -> Result<&str, String> {
        match &self.compiled_ir {
            Some(ir) => Ok(ir),
            None if !self.model.camdl.ends_with(".camdl") => Ok(&self.model.camdl),
            None => Err(format!(
                "internal: the model {} was not compiled before it was read",
                self.model.camdl
            )),
        }
    }

    /// The likelihood this problem defines: [`LikelihoodObjective`] from
    /// `ic_free`. The one place the key is read.
    pub fn likelihood_objective(&self) -> LikelihoodObjective {
        if self.ic_free.unwrap_or(false) {
            LikelihoodObjective::ConditionalOnFirstObservation
        } else {
            LikelihoodObjective::Full
        }
    }

    /// The entry point for every non-fit reader (`simulate --draws prior
    /// --fit`, `pfilter --fit`, `profile --fit`): the file's
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
    /// surfaces: `[model] camdl`, `store`, the wide-TSV `[data] file`, and
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
        if let Some(dir) = &self.store {
            check("store", dir);
        }
        if let Some(data) = &self.data {
            if let Some(file) = &data.file {
                check("[data] file", file);
            }
            for (stream, src) in &data.observations {
                check(&format!("[data.observations] {stream}"), src);
            }
            for (name, src) in &data.tables {
                check(&format!("[data.tables] {name}"), src);
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
        // The store root is written in the file, so it anchors at the file
        // like every other path in it.
        if let Some(dir) = &mut self.store {
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
            for v in data.tables.values_mut() {
                *v = crate::util::resolve_relative_to_toml(toml_path, v);
            }
        }
    }

    /// Expand `[fixed] from_scenario = "name"` in-place
    /// (`FixedParams::expand_from_scenario`), refusing a scenario that sets
    /// or scales a parameter in `[estimate]`.
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
    /// The files `[data.tables]` binds to the model's external inputs; none
    /// when the config has no `[data]` (a `[synthetic]` fit).
    pub fn external_bindings(&self) -> crate::external_inputs::ExternalBindings {
        crate::external_inputs::ExternalBindings::from_fit_config(
            self.data.iter().flat_map(|d| d.tables.iter().map(|(n, p)| (n.clone(), p.clone()))))
    }

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
            let names: Vec<&str> = overlap.iter().map(|s| **s).collect();
            return Err(crate::params_resolver::fixed_estimate_overlap_message(&names));
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


/// What a run's identity keys of a method, for the tests below: the record of
/// its `MethodIdentity` under `method`, and the starts rule `StartSource::of`
/// keys under `starts`. Two methods key the same exactly when their payloads
/// are equal, since a record hashes as its identity.
#[cfg(test)]
impl Method {
    fn identity_payload(&self) -> serde_json::Value {
        use store::Identity;
        let (method, _) = crate::fit_store::MethodIdentity::of(self, &no_steps(&self.algorithm)).expect("a finite method keys");
        let mut v = serde_json::json!({ "method": method.record() });
        if let Ok(rule) = self.starts() {
            v["starts"] = rule.identity_value();
        }
        v
    }
}

/// The steps of a method on no estimated parameter, for the identity tests
/// below, which key the method alone.
#[cfg(test)]
fn no_steps(a: &Algorithm) -> crate::fit::steps::ResolvedSteps {
    use crate::fit::steps::ResolvedSteps as S;
    match a {
        Algorithm::IF2 { .. } => S::If2 { params: IndexMap::new(), simplex: Vec::new() },
        Algorithm::PMMH { .. } | Algorithm::Mh { .. } => S::Proposal(IndexMap::new()),
        _ => S::None,
    }
}

/// The length a run's identity holds apart from its method, and the
/// trajectory count its method holds, for the tests below.
#[cfg(test)]
impl Algorithm {
    fn run_length(&self) -> u64 {
        let m = Method { algorithm: self.clone(), starts: Some(crate::fit::starts::ChainStarts::from_prior()) };
        crate::fit_store::MethodIdentity::of(&m, &no_steps(self)).expect("a finite method keys").1
    }

    fn cas_n_trajectories(&self) -> usize {
        match self {
            Algorithm::PGAS { n_trajectories, .. } => *n_trajectories,
            _ => 0,
        }
    }
}

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

    /// A problem's compiled model: `compiled_ir` when set, a `model.camdl`
    /// that is already compiled IR, and an error for model source nothing
    /// compiled, never a compile.
    #[test]
    fn the_compiled_model_is_never_uncompiled_source() {
        let problem = |camdl: &str| {
            parse(&format!("[model]\ncamdl = \"{camdl}\"\n[estimate]\nbeta = {{ bounds = [0.1, 1.0] }}\n[fixed]\n"))
                .unwrap()
                .problem
        };
        let mut source = problem("m.camdl");
        let err = source.compiled_model().unwrap_err();
        assert!(err.contains("m.camdl") && err.contains("was not compiled"), "{err}");
        source.compiled_ir = Some("compiled.ir.json".into());
        assert_eq!(source.compiled_model().unwrap(), "compiled.ir.json");
        assert_eq!(problem("m.ir.json").compiled_model().unwrap(), "m.ir.json");
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
            _ => panic!("expected IF2 method"),
        }
    }

    // gh#307: absolute-path portability lint over fit-config file references.

    /// A minimal but valid fit config parametrized by the four file-reference
    /// surfaces the lint covers, so a test can flip any of them absolute/relative
    /// without repeating the boilerplate.
    fn cfg_with_paths(camdl: &str, obs: &str, store: &str) -> FitConfig {
        parse(&format!(
            r#"
store = "{store}"

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

    /// A `pmmh` method, for `--no-adapt` / `--adapt-start`.
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
    /// NloptConfig payload (gh#726).
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

    /// gh#540: EVERY sampler/output flag must move the method identity, not
    /// just the five chain-start ones gh#514 covered. Each of these was
    /// applied to the `*MethodOpts` struct at the dispatch site, AFTER the
    /// identity was taken from the method — so `--n-trajectories 500` was
    /// served the 200-trajectory leaf, and `--no-nuts` was served a posterior
    /// drawn by NUTS.
    ///
    /// Table-driven so a flag added to `CliMethodOverrides` without a case here
    /// is conspicuous. A per-flag test would let the fourteenth flag slip in
    /// silently, which is exactly how the thirteen got here.
    ///
    /// `--n-trajectories` is not in this table: it is in the method's
    /// settings like every other key (`pgas_identity_payload_omits_n_trajectories`
    /// asserts that two counts key differently), and its own test,
    /// `n_trajectories_is_count_in_the_key`, checks that the flag reaches the
    /// method.
    /// `--starts` is not a `CliMethodOverrides` field at all — it writes
    /// `Method::starts` directly; `cli_starts_override_changes_the_method_identity`
    /// pins it.
    #[test]
    fn every_cli_sampler_flag_changes_the_method_identity() {
        let cases: Vec<(&str, Method, CliMethodOverrides)> = vec![
            ("--max-tree-depth", pgas_method(),
             CliMethodOverrides { max_tree_depth: Some(7), ..Default::default() }),
            ("--trajectory-warmup", pgas_method(),
             CliMethodOverrides { trajectory_warmup: Some(3), ..Default::default() }),
            ("--csmc-sweeps-per-nuts", pgas_method(),
             CliMethodOverrides { csmc_sweeps_per_nuts: Some(4), ..Default::default() }),
            ("--diagonal-mass", pgas_method(),
             CliMethodOverrides { diagonal_mass: true, ..Default::default() }),
            ("--no-nuts", pgas_method(),
             CliMethodOverrides { no_nuts: true, ..Default::default() }),
            ("--no-adapt", pmmh_method(),
             CliMethodOverrides { no_adapt: true, ..Default::default() }),
            ("--adapt-start", pmmh_method(),
             CliMethodOverrides { adapt_start: Some(42), ..Default::default() }),
            // The 2026-08-23 batch: found applied at the dispatch site,
            // after the CAS claim, exactly like the thirteen above.
            ("--cooling-target-iters", scout_method("single"),
             CliMethodOverrides { cooling_target_iters: Some(20), ..Default::default() }),
            ("--decibans-thresh", scout_method("single"),
             CliMethodOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
            ("--no-dt-check", scout_method("single"),
             CliMethodOverrides { no_dt_check: true, ..Default::default() }),
            ("--dt-check-halvings", scout_method("single"),
             CliMethodOverrides { dt_check_halvings: Some(3), ..Default::default() }),
            ("--record-ancestry", pfilter_method(),
             CliMethodOverrides { record_ancestry: true, ..Default::default() }),
            ("--record-prequential", pfilter_method(),
             CliMethodOverrides { record_prequential: true, ..Default::default() }),
            // gh#726: mh and nl-* store the dt-check result in the leaf;
            // their dt_check TOML field exists precisely so these flags
            // can reach the identity.
            ("--no-dt-check (mh)", mh_method(),
             CliMethodOverrides { no_dt_check: true, ..Default::default() }),
            ("--dt-check-halvings (mh)", mh_method(),
             CliMethodOverrides { dt_check_halvings: Some(3), ..Default::default() }),
            ("--no-dt-check (nl-sbplx)", nl_sbplx_method(),
             CliMethodOverrides { no_dt_check: true, ..Default::default() }),
            // gh#730: --dt-check-strict selects the threshold that is STORED
            // in fit_state.toml.dt_check (verdict + threshold_nats +
            // threshold_se_aware_nats + notes all derive from it), so it was
            // never the leaf-byte-neutral abort policy it was treated as.
            ("--dt-check-strict (if2)", scout_method("single"),
             CliMethodOverrides { dt_check_strict: true, ..Default::default() }),
            ("--dt-check-strict (mh)", mh_method(),
             CliMethodOverrides { dt_check_strict: true, ..Default::default() }),
            ("--decibans-thresh (nl-sbplx)", nl_sbplx_method(),
             CliMethodOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
        ];
        for (flag, base, cli) in cases {
            let mut overridden = base.clone();
            overridden.algorithm.apply_cli_overrides(&cli)
                .unwrap_or_else(|e| panic!("{flag}: {e}"));
            assert_ne!(
                base.identity_payload(), overridden.identity_payload(),
                "{flag} changes what the method computes or stores, so it must \
                 change the method identity. Sharing an identity means the run \
                 is served the OTHER setting's stored result (gh#540)."
            );
        }
    }

    /// gh#730: `--dt-check-strict` resolves to a CONCRETE threshold at
    /// override time, so it is keyed like any other stored setting — and a
    /// method that declares its own `threshold_nats` is unaffected, exactly as
    /// before (the flag was already inert there).
    #[test]
    fn dt_check_strict_resolves_to_the_stored_threshold() {
        use crate::fit::dt_check::default_threshold_for_backend;
        use crate::run_meta::InferenceBackend;

        let mut if2 = scout_method("single");
        if2.algorithm.apply_cli_overrides(&CliMethodOverrides {
            dt_check_strict: true, ..Default::default() }).unwrap();
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

        // A TOML-declared threshold leaves the flag nothing to select: it is
        // refused rather than silently ignored.
        let mut declared = scout_method("single");
        if let Algorithm::IF2 { dt_check, .. } = &mut declared.algorithm {
            dt_check.threshold_nats = Some(1.25);
        }
        let err = declared.algorithm.apply_cli_overrides(&CliMethodOverrides {
            dt_check_strict: true, ..Default::default() }).unwrap_err();
        assert!(err.contains("--dt-check-strict") && err.contains("threshold_nats = 1.25"),
            "{err}");
    }

    /// Every field of every `MethodIdentity` variant is reached from its
    /// `[method]` key: changing the key, and only it, changes the method's key.
    /// The cases are checked against the record's own fields, so a field added
    /// to an identity struct fails here until it has a case. The method's
    /// length and `starts` are the `[method]` keys outside the method's key.
    #[test]
    fn each_method_identity_field_changes_the_method_key() {
        use crate::fit_store::MethodIdentity;
        use crate::run_meta::FitAlgorithm as F;
        use store::Identity;
        type Case = (&'static str, Box<dyn Fn(&mut Algorithm)>);

        fn bump(x: &mut usize) { *x += 1 }
        fn bump_opt(x: &mut Option<usize>) { *x = Some(x.unwrap_or(0) + 1) }
        fn nudge(x: &mut f64) { *x = *x * 0.5 + 0.01 }
        fn nudge_opt(x: &mut Option<f64>) { *x = Some(x.unwrap_or(1.0) + 1.0) }
        fn not(x: &mut bool) { *x = !*x }
        fn other_backend(b: &mut crate::run_meta::InferenceBackend) {
            use crate::run_meta::InferenceBackend as B;
            *b = match b { B::ChainBinomial => B::Ode, B::Ode => B::ChainBinomial };
        }
        fn other_binomial(b: &mut sim::rng::BinomialAlgorithm) {
            use sim::rng::BinomialAlgorithm as B;
            *b = match b { B::Btpe => B::Btrs, B::Btrs => B::Btpe };
        }
        fn other_combine(c: &mut CombineMode) {
            *c = match c { CombineMode::LogMeanExp => CombineMode::Mean, CombineMode::Mean => CombineMode::LogMeanExp };
        }
        fn other_combine_opt(c: &mut Option<CombineMode>) {
            *c = Some(match c { Some(CombineMode::LogMeanExp) => CombineMode::Mean, _ => CombineMode::LogMeanExp });
        }
        // One case per field of a variant, reached through `$get`, a function
        // from the method to the variant's field.
        macro_rules! case {
            ($name:literal, $get:expr, $change:expr) => {
                ($name, Box::new(move |a: &mut Algorithm| $change($get(a))) as Box<dyn Fn(&mut Algorithm)>)
            };
        }
        fn gate_cases(get: fn(&mut Algorithm) -> &mut GateConfig) -> Vec<Case> {
            vec![
                case!("gate.a_thresh", get, |g: &mut GateConfig| nudge(&mut g.a_thresh)),
                case!("gate.decibans_thresh", get, |g: &mut GateConfig| nudge(&mut g.decibans_thresh)),
            ]
        }
        fn dt_check_cases(get: fn(&mut Algorithm) -> &mut DtCheckConfig) -> Vec<Case> {
            vec![
                case!("dt_check.enabled", get, |d: &mut DtCheckConfig| not(&mut d.enabled)),
                case!("dt_check.n_halvings", get, |d: &mut DtCheckConfig| bump(&mut d.n_halvings)),
                case!("dt_check.n_particles", get, |d: &mut DtCheckConfig| bump_opt(&mut d.n_particles)),
                case!("dt_check.n_replicates", get, |d: &mut DtCheckConfig| bump_opt(&mut d.n_replicates)),
                case!("dt_check.threshold_nats", get, |d: &mut DtCheckConfig| nudge_opt(&mut d.threshold_nats)),
                case!("dt_check.combine", get, |d: &mut DtCheckConfig| other_combine_opt(&mut d.combine)),
            ]
        }
        // Gives a closure that returns a borrow of its argument the signature
        // `for<'a> Fn(&'a mut Algorithm) -> &'a mut T`.
        fn reach<T, G: Fn(&mut Algorithm) -> &mut T>(g: G) -> G { g }
        // `$v` is the variant pattern, `$f` the field it binds.
        macro_rules! field {
            ($v:path, $f:ident) => {
                reach(|a: &mut Algorithm| match a { $v { $f, .. } => $f, _ => unreachable!() })
            };
        }
        fn nlopt(a: &mut Algorithm) -> &mut NloptConfig {
            match a {
                Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) => c,
                _ => unreachable!(),
            }
        }
        let cases = |algo: F| -> Vec<Case> {
            match algo {
                F::If2 => {
                    let mut v = vec![
                        case!("backend", field!(Algorithm::IF2, backend), other_backend),
                        case!("chains", field!(Algorithm::IF2, chains), bump),
                        case!("particles", field!(Algorithm::IF2, particles), bump),
                        case!("iterations", field!(Algorithm::IF2, iterations), bump),
                        case!("cooling", field!(Algorithm::IF2, cooling), nudge),
                        case!("cooling_target_iters", field!(Algorithm::IF2, cooling_target_iters), bump),
                        case!("loglik_eval.n_particles", field!(Algorithm::IF2, loglik_eval),
                              |l: &mut LoglikEvalConfig| bump(&mut l.n_particles)),
                        case!("loglik_eval.n_replicates", field!(Algorithm::IF2, loglik_eval),
                              |l: &mut LoglikEvalConfig| bump(&mut l.n_replicates)),
                        case!("loglik_eval.combine", field!(Algorithm::IF2, loglik_eval),
                              |l: &mut LoglikEvalConfig| other_combine(&mut l.combine)),
                        case!("binomial", field!(Algorithm::IF2, binomial), other_binomial),
                    ];
                    v.extend(gate_cases(field!(Algorithm::IF2, gate)));
                    v.extend(dt_check_cases(field!(Algorithm::IF2, dt_check)));
                    v
                }
                F::Pgas => vec![
                    case!("backend", field!(Algorithm::PGAS, backend), other_backend),
                    case!("chains", field!(Algorithm::PGAS, chains), bump),
                    case!("particles", field!(Algorithm::PGAS, particles), bump),
                    case!("binomial", field!(Algorithm::PGAS, binomial), other_binomial),
                    case!("burn_in", field!(Algorithm::PGAS, burn_in), bump_opt),
                    case!("thin", field!(Algorithm::PGAS, thin), bump_opt),
                    case!("max_tree_depth", field!(Algorithm::PGAS, max_tree_depth), bump),
                    case!("trajectory_warmup", field!(Algorithm::PGAS, trajectory_warmup), bump),
                    case!("csmc_sweeps_per_nuts", field!(Algorithm::PGAS, csmc_sweeps_per_nuts), bump),
                    case!("n_trajectories", field!(Algorithm::PGAS, n_trajectories), bump),
                    case!("dense_mass", field!(Algorithm::PGAS, dense_mass), not),
                    case!("use_nuts", field!(Algorithm::PGAS, use_nuts), not),
                    case!("ancestor_sampling", field!(Algorithm::PGAS, ancestor_sampling), not),
                ],
                F::Pmmh => vec![
                    case!("backend", field!(Algorithm::PMMH, backend), other_backend),
                    case!("chains", field!(Algorithm::PMMH, chains), bump),
                    case!("particles", field!(Algorithm::PMMH, particles), bump),
                    case!("burn_in", field!(Algorithm::PMMH, burn_in), bump_opt),
                    case!("thin", field!(Algorithm::PMMH, thin), bump_opt),
                    case!("adapt", field!(Algorithm::PMMH, adapt), not),
                    case!("adapt_start", field!(Algorithm::PMMH, adapt_start), bump),
                    case!("binomial", field!(Algorithm::PMMH, binomial), other_binomial),
                ],
                F::Mh => {
                    let mut v = vec![
                        case!("backend", field!(Algorithm::Mh, backend), other_backend),
                        case!("chains", field!(Algorithm::Mh, chains), bump),
                        case!("burn_in", field!(Algorithm::Mh, burn_in), bump_opt),
                        case!("thin", field!(Algorithm::Mh, thin), bump_opt),
                        case!("adapt", field!(Algorithm::Mh, adapt), not),
                        case!("adapt_start", field!(Algorithm::Mh, adapt_start), bump),
                        case!("burnin_dt", field!(Algorithm::Mh, burnin_dt), nudge_opt),
                    ];
                    v.extend(dt_check_cases(field!(Algorithm::Mh, dt_check)));
                    v
                }
                F::Pfilter => vec![
                    case!("backend", field!(Algorithm::PFilter, backend), other_backend),
                    case!("particles", field!(Algorithm::PFilter, particles), bump),
                    case!("replicates", field!(Algorithm::PFilter, replicates), bump_opt),
                    case!("record_ancestry", field!(Algorithm::PFilter, record_ancestry), not),
                    case!("record_prequential", field!(Algorithm::PFilter, record_prequential), not),
                    case!("binomial", field!(Algorithm::PFilter, binomial), other_binomial),
                ],
                F::NlSbplx | F::NlBobyqa | F::NlLbfgs => {
                    let mut v = vec![
                        case!("backend", nlopt, |c: &mut NloptConfig| other_backend(&mut c.backend)),
                        case!("chains", nlopt, |c: &mut NloptConfig| bump(&mut c.chains)),
                        case!("tolerance", nlopt, |c: &mut NloptConfig| nudge(&mut c.tolerance)),
                        case!("max_evals", nlopt, |c: &mut NloptConfig| bump(&mut c.max_evals)),
                    ];
                    v.extend(gate_cases(reach(|a: &mut Algorithm| &mut nlopt(a).gate)));
                    v.extend(dt_check_cases(reach(|a: &mut Algorithm| &mut nlopt(a).dt_check)));
                    v
                }
            }
        };
        // The record's fields, nested structs as dotted paths.
        fn leaves(prefix: &str, v: &serde_json::Value, out: &mut Vec<String>) {
            match v.as_object() {
                Some(o) => for (k, x) in o {
                    let name = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    leaves(&name, x, out);
                },
                None => out.push(prefix.to_string()),
            }
        }
        let key = |m: &Method| MethodIdentity::of(m, &no_steps(&m.algorithm)).expect("a finite method keys").0.key();
        for algo in [F::If2, F::Pgas, F::Pmmh, F::Mh, F::Pfilter, F::NlSbplx, F::NlBobyqa, F::NlLbfgs] {
            let base = base_method_for(algo);
            let (identity, _) = MethodIdentity::of(&base, &no_steps(&base.algorithm)).unwrap();
            let record = serde_json::to_value(identity.record()).unwrap();
            let (variant, fields) = record.as_object().unwrap().iter().next().unwrap();
            assert_eq!(identity.record().algorithm(), algo.as_str(), "{variant}");
            let mut expected = Vec::new();
            leaves("", fields, &mut expected);
            // The resolved steps are keyed from `fit::steps::ResolvedSteps`,
            // not from a field of the method, and are tested with it
            // (`fit_store/tests.rs`).
            expected.retain(|f| !["perturbation", "simplex", "proposal"].iter().any(|s| f.starts_with(s)));
            expected.sort();
            let cases = cases(algo);
            let mut named: Vec<String> = cases.iter().map(|(n, _)| n.to_string()).collect();
            named.sort();
            assert_eq!(named, expected, "{}: one case per field of the identity", algo.as_str());
            for (name, change) in &cases {
                let mut changed = base.clone();
                change(&mut changed.algorithm);
                assert_ne!(key(&base), key(&changed), "{}: `{name}` must change the method's key", algo.as_str());
            }
            // The other two `[method]` keys: the length and the start rule.
            let mut outside = base.clone();
            outside.starts = Some(ChainStarts::Spread(Spread::Lhs));
            match &mut outside.algorithm {
                Algorithm::PGAS { sweeps: n, .. }
                | Algorithm::PMMH { iterations: n, .. }
                | Algorithm::Mh { iterations: n, .. } => *n += 1,
                _ => {}
            }
            assert_eq!(key(&base), key(&outside), "{}: the length and `starts` are outside the method's key", algo.as_str());
        }
    }

    /// The extension dimension must stay OUT: a resumed run extends a base
    /// run, so the two share a prefix identity by design — the length reaches
    /// identity through `run_length` instead.
    #[test]
    fn extension_dimension_stays_out_of_the_payload() {
        let base = pmmh_method();
        let mut longer = pmmh_method();
        if let Algorithm::PMMH { iterations, .. } = &mut longer.algorithm { *iterations *= 4; }
        assert_eq!(base.identity_payload(), longer.identity_payload(),
            "iterations is PMMH's extension dimension — it must not re-key the \
             payload, or --resume could never share a prefix identity");
        assert_ne!(base.algorithm.run_length(), longer.algorithm.run_length(),
            "…but it MUST reach identity through run_length");
    }

    /// `--n-trajectories` is the sharpest case, because `cas.rs` documents it
    /// as folded count-in-the-key and it was not. Pinned separately from the
    /// table so the claim in that doc comment has a test under it.
    #[test]
    fn n_trajectories_is_count_in_the_key() {
        let base = pgas_method();
        let mut more = pgas_method();
        more.algorithm.apply_cli_overrides(&CliMethodOverrides {
            n_trajectories: Some(500), ..Default::default() }).unwrap();
        assert_ne!(base.algorithm.cas_n_trajectories(), more.algorithm.cas_n_trajectories(),
            "`--n-trajectories` must reach `cas_n_trajectories` — it reads the \
             method, and the flag used to be applied to the opts struct instead, \
             so a 500-trajectory request was served the 200-trajectory leaf");
        assert_eq!(more.algorithm.cas_n_trajectories(), 500);
    }

    /// AS-off changes the draws, so it is in the method's identity.
    #[test]
    fn ancestor_sampling_off_is_in_the_method_identity() {
        let on = pgas_method();
        let mut off = pgas_method();
        if let Algorithm::PGAS { ref mut ancestor_sampling, .. } = off.algorithm {
            *ancestor_sampling = false;
        }
        assert_ne!(on.identity_payload(), off.identity_payload(),
            "an AS-off fit must not be served from (or stored over) an AS-on leaf");
    }

    /// `ancestor_sampling = false` is the TOML spelling; absent means on.
    #[test]
    fn ancestor_sampling_parses_from_method_toml() {
        let toml_src = r#"
            algorithm = "pgas"
            backend = "chain_binomial"
            chains = 2
            particles = 10
            sweeps = 5
            ancestor_sampling = false
        "#;
        let method: Algorithm = toml::from_str(toml_src).expect("method parses");
        match method {
            Algorithm::PGAS { ancestor_sampling, .. } => assert!(!ancestor_sampling),
            other => panic!("expected PGAS, got {}", other.method_name()),
        }
        let default_src = toml_src.replace("ancestor_sampling = false", "");
        match toml::from_str::<Algorithm>(&default_src).expect("method parses") {
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
        overridden.algorithm.apply_cli_overrides(&CliMethodOverrides {
            no_ancestor_sampling: true, ..Default::default() }).unwrap();
        match &overridden.algorithm {
            Algorithm::PGAS { ancestor_sampling, .. } => assert!(!ancestor_sampling),
            other => panic!("expected PGAS, got {}", other.method_name()),
        }
        assert_ne!(base.identity_payload(), overridden.identity_payload());
    }

    /// The other half, and the one that protects existing users: a run with
    /// no CLI overrides must key exactly as it did before.
    #[test]
    fn no_cli_override_leaves_the_method_identity_untouched() {
        let declared = scout_method("uniform");
        let mut untouched = scout_method("uniform");
        untouched.algorithm.apply_cli_overrides(&CliMethodOverrides::default()).unwrap();
        assert_eq!(declared.identity_payload(), untouched.identity_payload());
    }

    /// One base method per algorithm, each set so that every flag the
    /// algorithm reads moves something (the pfilter base declares both record
    /// fields false, so the one-way override to true is visible).
    fn base_method_for(algo: crate::run_meta::FitAlgorithm) -> Method {
        use crate::run_meta::FitAlgorithm as A;
        let nl = |wrap: fn(NloptConfig) -> Algorithm| {
            let mut m = nl_sbplx_method();
            if let Algorithm::NlSbplx(cfg) = m.algorithm.clone() { m.algorithm = wrap(cfg); }
            m
        };
        match algo {
            A::If2 => scout_method("single"),
            A::Pgas => pgas_method(),
            A::Pmmh => pmmh_method(),
            A::Mh => mh_method(),
            A::Pfilter => pfilter_method(),
            A::NlSbplx => nl_sbplx_method(),
            A::NlBobyqa => nl(Algorithm::NlBobyqa),
            A::NlLbfgs => nl(Algorithm::NlLbfgs),
        }
    }

    /// The overrides with only `flag` passed, at a value off every default.
    fn only(flag: OverrideFlag) -> CliMethodOverrides {
        let d = CliMethodOverrides::default();
        match flag {
            OverrideFlag::MaxTreeDepth => CliMethodOverrides { max_tree_depth: Some(7), ..d },
            OverrideFlag::TrajectoryWarmup => CliMethodOverrides { trajectory_warmup: Some(3), ..d },
            OverrideFlag::CsmcSweepsPerNuts => CliMethodOverrides { csmc_sweeps_per_nuts: Some(4), ..d },
            OverrideFlag::NTrajectories => CliMethodOverrides { n_trajectories: Some(500), ..d },
            OverrideFlag::DiagonalMass => CliMethodOverrides { diagonal_mass: true, ..d },
            OverrideFlag::NoNuts => CliMethodOverrides { no_nuts: true, ..d },
            OverrideFlag::NoAncestorSampling => CliMethodOverrides { no_ancestor_sampling: true, ..d },
            OverrideFlag::NoAdapt => CliMethodOverrides { no_adapt: true, ..d },
            OverrideFlag::AdaptStart => CliMethodOverrides { adapt_start: Some(42), ..d },
            OverrideFlag::CoolingTargetIters => CliMethodOverrides { cooling_target_iters: Some(20), ..d },
            OverrideFlag::DecibansThresh => CliMethodOverrides { decibans_thresh: Some(5.0), ..d },
            OverrideFlag::NoDtCheck => CliMethodOverrides { no_dt_check: true, ..d },
            OverrideFlag::DtCheckHalvings => CliMethodOverrides { dt_check_halvings: Some(3), ..d },
            OverrideFlag::DtCheckStrict => CliMethodOverrides { dt_check_strict: true, ..d },
            OverrideFlag::Binomial => CliMethodOverrides {
                binomial: Some(sim::rng::BinomialAlgorithm::Btrs), ..d },
            OverrideFlag::RecordAncestry => CliMethodOverrides { record_ancestry: true, ..d },
            OverrideFlag::RecordPrequential => CliMethodOverrides { record_prequential: true, ..d },
        }
    }

    /// Every override flag is in the table, and the table is the truth: on
    /// every method, a flag the table lists is accepted AND changes the
    /// method, and a flag it does not list is refused with a message naming
    /// the flag, the method, and the methods that read it. A flag the table
    /// claims but `apply_cli_overrides` ignores fails the first half; a flag
    /// applied but missing from the table fails the second.
    #[test]
    fn every_override_flag_is_refused_on_a_method_that_does_not_read_it() {
        use crate::run_meta::FitAlgorithm as A;
        // `OverrideFlag::ALL` covers every field: setting every flag in it
        // leaves no field of the struct at its default. The destructure has
        // no `..`, so a new field is a compile error here too.
        let mut all = CliMethodOverrides::default();
        for flag in OverrideFlag::ALL {
            let one = only(flag);
            assert_eq!(one.passed(), vec![flag], "{} round-trips through passed()", flag.spelling());
            let CliMethodOverrides {
                max_tree_depth, trajectory_warmup, csmc_sweeps_per_nuts, n_trajectories,
                diagonal_mass, no_nuts, no_ancestor_sampling, no_adapt, adapt_start,
                cooling_target_iters, decibans_thresh, no_dt_check, dt_check_halvings,
                dt_check_strict, binomial, record_ancestry, record_prequential,
            } = one;
            all.max_tree_depth = all.max_tree_depth.or(max_tree_depth);
            all.trajectory_warmup = all.trajectory_warmup.or(trajectory_warmup);
            all.csmc_sweeps_per_nuts = all.csmc_sweeps_per_nuts.or(csmc_sweeps_per_nuts);
            all.n_trajectories = all.n_trajectories.or(n_trajectories);
            all.diagonal_mass |= diagonal_mass;
            all.no_nuts |= no_nuts;
            all.no_ancestor_sampling |= no_ancestor_sampling;
            all.no_adapt |= no_adapt;
            all.adapt_start = all.adapt_start.or(adapt_start);
            all.cooling_target_iters = all.cooling_target_iters.or(cooling_target_iters);
            all.decibans_thresh = all.decibans_thresh.or(decibans_thresh);
            all.no_dt_check |= no_dt_check;
            all.dt_check_halvings = all.dt_check_halvings.or(dt_check_halvings);
            all.dt_check_strict |= dt_check_strict;
            all.binomial = all.binomial.or(binomial);
            all.record_ancestry |= record_ancestry;
            all.record_prequential |= record_prequential;
        }
        assert_eq!(all.passed(), OverrideFlag::ALL.to_vec(),
            "OverrideFlag::ALL must name every CliMethodOverrides field");

        let methods = [A::If2, A::Pgas, A::Pmmh, A::Mh, A::Pfilter,
                       A::NlSbplx, A::NlBobyqa, A::NlLbfgs];
        for flag in OverrideFlag::ALL {
            for algo in methods {
                let base = base_method_for(algo);
                assert_eq!(base.algorithm.method_kind(), algo);
                // The table agrees with the writer: the flag moves this
                // method exactly when the table lists it. Without this a row
                // dropped from the table would turn a working flag into a
                // refused one, and nothing would notice.
                let mut written = base.clone();
                let wrote = written.algorithm.write_cli_overrides(&only(flag)).is_ok()
                    && format!("{:?}", base.algorithm) != format!("{:?}", written.algorithm);
                assert_eq!(wrote, flag.methods().contains(&algo),
                    "{} on `{}`: the writer {} the method, but the table {} it",
                    flag.spelling(), algo.as_str(),
                    if wrote { "moves" } else { "does not move" },
                    if flag.methods().contains(&algo) { "lists" } else { "does not list" });
                let mut overridden = base.clone();
                let got = overridden.algorithm.apply_cli_overrides(&only(flag));
                if flag.methods().contains(&algo) {
                    got.unwrap_or_else(|e| panic!("{} on {}: {e}", flag.spelling(), algo.as_str()));
                    assert_ne!(format!("{:?}", base.algorithm), format!("{:?}", overridden.algorithm),
                        "{} is listed for `{}` but changed nothing there",
                        flag.spelling(), algo.as_str());
                } else {
                    let err = got.expect_err(&format!(
                        "{} must be refused on `{}`", flag.spelling(), algo.as_str()));
                    assert!(err.contains(flag.spelling()), "{err}");
                    assert!(err.contains(&format!("`{}`", algo.as_str())), "{err}");
                    for m in flag.methods() {
                        assert!(err.contains(&format!("`{}`", m.as_str())), "names `{}`: {err}", m.as_str());
                    }
                }
            }
        }
    }

    /// A flag the method reads, beside a setting that leaves it nothing to do,
    /// is refused rather than run, re-keyed and ignored.
    #[test]
    fn a_flag_left_inert_by_another_setting_is_refused() {
        let cases: Vec<(&str, Method, CliMethodOverrides)> = vec![
            ("--max-tree-depth", pgas_method(),
             CliMethodOverrides { no_nuts: true, max_tree_depth: Some(7), ..Default::default() }),
            ("--diagonal-mass", pgas_method(),
             CliMethodOverrides { no_nuts: true, diagonal_mass: true, ..Default::default() }),
            ("--adapt-start", pmmh_method(),
             CliMethodOverrides { no_adapt: true, adapt_start: Some(42), ..Default::default() }),
            ("--adapt-start", mh_method(),
             CliMethodOverrides { no_adapt: true, adapt_start: Some(42), ..Default::default() }),
            ("--dt-check-halvings", scout_method("single"),
             CliMethodOverrides { no_dt_check: true, dt_check_halvings: Some(3), ..Default::default() }),
            ("--dt-check-strict", nl_sbplx_method(),
             CliMethodOverrides { no_dt_check: true, dt_check_strict: true, ..Default::default() }),
            // One chain: there is no between-chain spread for the gate to judge.
            ("--decibans-thresh", {
                let mut m = scout_method("single");
                if let Algorithm::IF2 { chains, .. } = &mut m.algorithm { *chains = 1; }
                m
            }, CliMethodOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
            ("--decibans-thresh", {
                let mut m = nl_sbplx_method();
                if let Algorithm::NlSbplx(c) = &mut m.algorithm { c.chains = 1; }
                m
            }, CliMethodOverrides { decibans_thresh: Some(5.0), ..Default::default() }),
        ];
        for (flag, base, cli) in cases {
            let mut m = base.clone();
            let err = m.algorithm.apply_cli_overrides(&cli)
                .expect_err(&format!("{flag} is inert here and must be refused"));
            assert!(err.contains(flag), "{err}");
        }
        // The same flag beside the setting it needs is fine.
        let mut m = pgas_method();
        m.algorithm.apply_cli_overrides(&CliMethodOverrides {
            max_tree_depth: Some(7), ..Default::default() }).unwrap();
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
            "one warning each for [model] camdl, [data.observations] weekly_cases, store; got: {warnings:?}"
        );
        // Each warning names its surface and the offending path, and flags it as
        // non-portable (not just "absolute").
        assert!(warnings.iter().any(|w|
            w.contains("[model] camdl") && w.contains("/abs/models/sir.camdl")));
        assert!(warnings.iter().any(|w|
            w.contains("[data.observations] weekly_cases") && w.contains("/abs/data/cases.tsv")));
        assert!(warnings.iter().any(|w|
            w.contains("store") && w.contains("/abs/out")));
        assert!(warnings.iter().all(|w| w.contains("non-portable")));
    }

    /// Every path written in the fit.toml anchors at the fit.toml, `store`
    /// included: one `../` must mean the same for the inputs and the store.
    #[test]
    fn load_anchors_the_store_at_the_toml_like_every_other_path() {
        let dir = std::env::temp_dir().join("camdl_gh507_anchor");
        std::fs::create_dir_all(&dir).unwrap();
        let toml_path = dir.join("fit.toml");
        std::fs::write(&toml_path, r#"
store = "../runs"

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

        let out = cfg.problem.store.as_deref().unwrap();
        assert!(out.starts_with(&*anchor),
            "store must anchor at the toml, not the CWD: {out}");
        assert!(std::path::Path::new(out).is_absolute(),
            "a resolved store is absolute, so nothing downstream can \
             re-anchor it at the CWD: {out}");
    }

    /// An absolute `store` is the user saying exactly where they want it;
    /// anchoring must not touch it (it warns as non-portable).
    #[test]
    fn load_passes_an_absolute_store_through_unchanged() {
        let dir = std::env::temp_dir().join("camdl_gh507_abs");
        std::fs::create_dir_all(&dir).unwrap();
        let toml_path = dir.join("fit.toml");
        std::fs::write(&toml_path, r#"
store = "/tmp/camdl_gh507_explicit"

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
        assert_eq!(cfg.problem.store.as_deref(), Some("/tmp/camdl_gh507_explicit"));
    }

    /// `output_dir` is not a key of the fit file: it is refused as an
    /// unknown field, so a fit is never stored somewhere other than the file
    /// asks.
    #[test]
    fn output_dir_is_an_unknown_key() {
        let err = FitConfig::from_toml_str_named(
            "output_dir = \"runs\"\n[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.01, 2.0] }\n[fixed]\n",
            "fit.toml",
        )
        .unwrap_err();
        assert!(err.contains("unknown field `output_dir`"), "{err}");
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
    /// `algorithm = "nuts"` is refused as an unknown algorithm with the steer
    /// to PGAS's NUTS move and the ODE sampler, on either backend; the same
    /// file with `mh` validates.
    #[test]
    fn nuts_algorithm_is_refused_with_the_steer() {
        for backend in ["ode", "chain_binomial"] {
            let src = format!(
                "[model]\ncamdl = \"models/sir.camdl\"\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\
                 [fixed]\nN0 = 1000\n\
                 [method]\nalgorithm = \"nuts\"\nbackend = \"{backend}\"\nchains = 2\n"
            );
            let err = parse(&src).expect_err("`nuts` is not an algorithm");
            assert!(err.contains("unknown algorithm \"nuts\""), "{backend}: {err}");
            assert!(err.contains("use_nuts") && err.contains("algorithm = \"pgas\""),
                "{backend}: names PGAS's NUTS move: {err}");
            assert!(err.contains("algorithm = \"mh\""), "{backend}: names the ODE sampler: {err}");
            if backend == "ode" {
                let mh = src.replace("algorithm = \"nuts\"", "algorithm = \"mh\"\niterations = 10");
                parse(&mh).unwrap_or_else(|e| panic!("the same file with mh validates: {e}"));
            }
        }
    }

    /// `use_nuts` is defaulted, not declared, unless `[method]` spells it; the
    /// two get different answers when the model has no gradients.
    #[test]
    fn nuts_without_gradients_is_refused_when_asked_for_and_noted_when_defaulted() {
        let explicit = "[method]\nalgorithm = \"pgas\"\nuse_nuts = true\n";
        let defaulted = "[method]\nalgorithm = \"pgas\"\n";
        assert!(method_declares_key(explicit, "use_nuts"));
        assert!(!method_declares_key(defaulted, "use_nuts"));
        assert!(!method_declares_key("use_nuts = true\n[method]\n", "use_nuts"),
            "only a key under [method] counts");
        let err = nuts_unavailable(true, &[]).unwrap_err();
        assert!(err.contains("use_nuts = true") && err.contains("rate gradients"), "{err}");
        let note = nuts_unavailable(false, &[]).unwrap();
        assert!(note.contains("Metropolis-within-Gibbs"), "{note}");
        // On the fallback the NUTS tuning settings are inert, as under
        // `use_nuts = false`.
        for given in ["max_tree_depth", "dense_mass", "--max-tree-depth", "--diagonal-mass"] {
            let err = nuts_unavailable(false, &[given]).unwrap_err();
            assert!(err.contains(given) && err.contains("Metropolis-within-Gibbs"), "{err}");
        }
    }

    /// A known `[method]` key the method's settings leave inert is refused at
    /// load, and the same key beside the setting it needs loads.
    #[test]
    fn method_refuses_a_key_its_own_settings_leave_inert() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\ncases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0], prior = { half_normal = { sigma = 1.0 } } }\n\
                    [fixed]\nN0 = 1000000\n";
        let pgas = "[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
                    chains = 2\nparticles = 100\nsweeps = 10\n";
        let pmmh = "[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\n\
                    chains = 2\nparticles = 100\niterations = 10\n";
        let mh = "[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 10\n";
        for (method, extra, key) in [
            (pgas, "use_nuts = false\nmax_tree_depth = 8\n", "max_tree_depth"),
            (pgas, "use_nuts = false\ndense_mass = false\n", "dense_mass"),
            (pmmh, "adapt = false\nadapt_start = 50\n", "adapt_start"),
            (mh, "adapt = false\nadapt_start = 50\n", "adapt_start"),
        ] {
            let src = format!("{base}{method}{extra}");
            let cfg = parse(&src).expect("the shared parser reads it (stored fits)");
            let err = refuse_inert_method_keys_in(&src, &cfg.inference.method.unwrap())
                .expect_err(&format!("`{key}` is inert beside `{extra}`"));
            assert!(err.contains(&format!("`{key}`")) && err.contains("no effect"), "{err}");
        }
        // [method.dt_check] with the check off: every other key is inert.
        let if2 = "[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                   chains = 2\nparticles = 100\niterations = 10\ncooling = 0.7\n";
        for (method, extra, key) in [
            (if2, "[method.dt_check]\nenabled = false\nn_halvings = 3\n", "n_halvings"),
            (mh, "[method.dt_check]\nenabled = false\nthreshold_nats = 0.5\n", "threshold_nats"),
        ] {
            let src = format!("{base}{method}{extra}");
            let cfg = parse(&src).unwrap();
            let err = refuse_inert_method_keys_in(&src, &cfg.inference.method.unwrap())
                .expect_err(&format!("`{key}` is inert with the dt-check off"));
            assert!(err.contains(key) && err.contains("enabled = false"), "{err}");
        }
        // [method.gate] on a one-chain run: no between-chain spread to judge.
        let if2_one = if2.replace("chains = 2", "chains = 1");
        let nl_one = "[method]\nalgorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\n";
        for method in [if2_one.as_str(), nl_one] {
            let src = format!("{base}{method}[method.gate]\ndecibans_thresh = 10.0\n");
            let cfg = parse(&src).unwrap();
            let err = refuse_inert_method_keys_in(&src, &cfg.inference.method.unwrap())
                .expect_err("[method.gate] is inert on one chain");
            assert!(err.contains("gate") && err.contains("one chain"), "{err}");
        }
        for (method, extra) in [
            (if2, "[method.gate]\ndecibans_thresh = 10.0\n"),
            (if2, "[method.dt_check]\nenabled = false\n"),
            (if2, "[method.dt_check]\nn_halvings = 3\n"),
            (pgas, "use_nuts = true\nmax_tree_depth = 8\ndense_mass = false\n"),
            (pgas, "use_nuts = false\n"),
            (pmmh, "adapt = true\nadapt_start = 50\n"),
        ] {
            let src = format!("{base}{method}{extra}");
            let cfg = parse(&src).unwrap();
            refuse_inert_method_keys_in(&src, &cfg.inference.method.unwrap())
                .unwrap_or_else(|e| panic!("`{extra}` is a live setting: {e}"));
        }
    }

    #[test]
    fn an_obs_alignment_key_in_config_says_it_was_removed() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0] }\n[fixed]\nN0 = 1000000\n\
                    [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
                    chains = 8\nparticles = 1000\niterations = 80\ncooling = 0.70\n";
        for value in ["exact", "snap"] {
            let stale = format!("{base}[config]\ndt = 1.0\nobs_alignment = \"{value}\"\n");
            let err = FitConfig::from_toml_str(&stale).expect_err("`obs_alignment` is not a key");
            assert!(err.contains("`obs_alignment` was removed") && err.contains("Delete the line"),
                "must say the key was removed and to delete the line; got: {err}");
        }
        let without = format!("{base}[config]\ndt = 1.0\n");
        assert!(FitConfig::from_toml_str(&without).is_ok(),
            "the same file without the key must parse: {:?}", FitConfig::from_toml_str(&without).err());
    }

    #[test]
    fn a_transform_key_in_estimate_says_it_was_removed() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [fixed]\nN0 = 1000000\n[method]\nalgorithm = \"if2\"\n\
                    backend = \"chain_binomial\"\nchains = 8\nparticles = 1000\n\
                    iterations = 80\ncooling = 0.70\n";
        let stale = format!("{base}[estimate]\nbeta = {{ bounds = [0.01, 2.0], transform = \"log\" }}\n");
        let err = FitConfig::from_toml_str(&stale).expect_err("`transform` is not a key");
        assert!(err.contains("estimate.beta") && err.contains("`transform` was removed")
            && err.contains("kind and bounds") && err.contains("Delete the line"),
            "must say the key was removed and to delete the line; got: {err}");
        let table = format!("{base}[estimate.beta]\nbounds = [0.01, 2.0]\ntransform = \"logit\"\n");
        assert!(FitConfig::from_toml_str(&table).unwrap_err().contains("`transform` was removed"));
        let without = stale.replace(", transform = \"log\"", "");
        assert!(FitConfig::from_toml_str(&without).is_ok(),
            "the same file without the key must parse: {:?}", FitConfig::from_toml_str(&without).err());
    }

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

        // A PGAS optional key (max_tree_depth) must be accepted, not falsely flagged.
        let ok_pgas = format!(
            "{base}[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\nsweeps = 10\nmax_tree_depth = 8\n"
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

        // `tempering` left with PGAS parallel tempering (gh#789). The message
        // says the key was removed and that the fix is to delete the line, not
        // a bare "unknown key" an author would read as a typo; the same file
        // without the line parses.
        let stale_tempering = format!(
            "{base}[method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\n\
             chains = 2\nparticles = 100\nsweeps = 10\ntempering = [1.0]\n"
        );
        let err = parse(&stale_tempering).expect_err("`tempering` under [method] is not a key");
        assert!(err.contains("`tempering`") && err.contains("removed") && err.contains("elete"),
            "must say the key was removed and to delete the line; got: {err}");
        let without = stale_tempering.replace("tempering = [1.0]\n", "");
        assert!(parse(&without).is_ok(),
            "the same file without the line must parse: {:?}", parse(&without).err());

        // `rho` left with correlated pseudo-marginal PMMH (proposal
        // 2026-10-24 §5.6). Same contract: the message says the key was
        // removed and that the fix is to delete the line; the same file
        // without the line parses.
        let stale_rho = format!(
            "{base}[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\n\
             chains = 1\nparticles = 100\niterations = 50\nburn_in = 10\nrho = 0.99\n"
        );
        let err = parse(&stale_rho).expect_err("`rho` under [method] is not a key");
        assert!(err.contains("`rho`") && err.contains("removed") && err.contains("elete"),
            "must say the key was removed and to delete the line; got: {err}");
        let without = stale_rho.replace("rho = 0.99\n", "");
        assert!(parse(&without).is_ok(),
            "the same file without the line must parse: {:?}", parse(&without).err());
    }

    /// The NLopt methods carry the same five keys whether or not the algorithm
    /// takes a gradient, so a key invented for the gradient one — `gradient =
    /// true`, the shape a user reaches for after reading that `nl-lbfgs` is
    /// "the gradient method" — is not a knob on either. serde cannot deny it
    /// (the `Algorithm` enum is internally tagged, gh#241 C3), so the
    /// post-parse `validate_method_keys` pass is what refuses it; pin that for
    /// `nl-lbfgs` and for a gradient-free sibling so the two cannot drift.
    #[test]
    fn nlopt_methods_reject_a_key_neither_of_them_has() {
        let base = "[model]\ncamdl = \"models/sir.camdl\"\n\
                    [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\
                    [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\
                    [fixed]\nN0 = 1000000\n";
        for algorithm in ["nl-lbfgs", "nl-sbplx"] {
            let ok = format!(
                "{base}[method]\nalgorithm = \"{algorithm}\"\nbackend = \"ode\"\n\
                 chains = 2\ntolerance = 1e-8\nmax_evals = 200\n"
            );
            let cfg = parse(&ok)
                .unwrap_or_else(|e| panic!("a valid {algorithm} method must parse: {e}"));
            assert_eq!(
                cfg.inference.method.as_ref().unwrap().algorithm.method_name(),
                algorithm,
                "the parsed variant must be the one the tag named"
            );

            let bad = format!(
                "{base}[method]\nalgorithm = \"{algorithm}\"\nbackend = \"ode\"\n\
                 chains = 2\ngradient = true\n"
            );
            let err = parse(&bad).err().unwrap_or_else(|| {
                panic!(
                    "`gradient = true` under a {algorithm} [method] must be rejected, \
                     not silently dropped — a dropped key is neither applied nor \
                     hashed into the method identity"
                )
            });
            assert!(
                err.contains("gradient") && err.contains("[method]"),
                "the {algorithm} error must name the unknown key and the table; got: {err}"
            );
        }
    }

    /// `nl-lbfgs` on a model the ODE gradient cannot be taken of is refused at
    /// validation, quoting the capability gate's own sentence and naming the
    /// derivative-free alternative — so the user learns it before a search
    /// starts, not from a chain that failed partway through.
    ///
    /// The refusal here is the adaptive integrator (`rk45`), whose step
    /// sequence is discontinuous in θ. Two controls make the assertion
    /// specific: the same model under `nl-sbplx` validates (the refusal is
    /// about the gradient, not the model), and `nl-lbfgs` on the same model
    /// with `rk4` validates (the refusal is about the integrator, not the
    /// method).
    #[test]
    fn nl_lbfgs_is_refused_at_validation_on_a_model_the_gradient_refuses() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let base_model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read golden: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse golden: {e}"));

        let cfg = |algorithm: &str| -> FitConfig {
            let toml_str = format!(
                "[model]\ncamdl = \"models/seir.camdl\"\n\n\
                 [data.observations]\nweekly_cases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n\n\
                 [fixed]\nsigma = 0.2\ngamma = 0.1\nrho = 0.5\nk = 5.0\n\
                 p_detect = 0.8\nN0 = 100000\nI0 = 10\n\n\
                 [method]\nalgorithm = \"{algorithm}\"\nbackend = \"ode\"\nchains = 2\n"
            );
            FitConfig::from_toml_str(&toml_str)
                .unwrap_or_else(|e| panic!("parse {algorithm}: {e}"))
        };

        let mut adaptive = base_model.clone();
        adaptive.simulation.integrator =
            ir::model::Integrator::Rk45 { atol: None, rtol: None };

        let err = cfg("nl-lbfgs")
            .validate_gradient_capability(&adaptive)
            .expect_err("an rk45 model has no fixed-step forward sensitivity");
        assert!(
            err.contains("rk4"),
            "the gate's own reason must be quoted; got: {err}"
        );
        assert!(
            err.contains("nl-sbplx"),
            "the refusal must name the derivative-free alternative; got: {err}"
        );
        assert!(
            err.starts_with("[method]:"),
            "the refusal must say which table it is about; got: {err}"
        );

        // Control 1: the derivative-free sibling has no gradient requirement.
        cfg("nl-sbplx")
            .validate_gradient_capability(&adaptive)
            .expect("nl-sbplx never asks for a gradient");
        // Control 2: the same method on the fixed-step model is fine.
        cfg("nl-lbfgs")
            .validate_gradient_capability(&base_model)
            .expect("rk4 is exactly what the forward sensitivity needs");
    }

    /// The gate sees the model the run will integrate, not the model as
    /// written. Toggleable interventions default OFF (spec §14.4), so a fit
    /// that never enables a declared `interventions { }` block has no scheduled
    /// effect in its trajectory; one that enables it has — and the gate then
    /// judges that effect on its own terms (gh#275 §1g): the fixed-fraction
    /// SIA the fixture declares is one the sensitivity crosses, so the enabled
    /// fit is admitted; the same intervention rewritten as a fixed-count
    /// transfer is refused, by name.
    ///
    /// Without the filter the first case would be judged on an effect the run
    /// would not have used, with no way for the user to tell the difference
    /// from a real limitation.
    #[test]
    fn the_gradient_gate_sees_interventions_as_the_run_will_resolve_them() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_vaccine.ir.json");
        let model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read golden: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse golden: {e}"));
        assert!(
            model.interventions.iter().any(|iv| iv.name == "sia_round_1"),
            "the fixture must declare a toggleable intervention"
        );

        let cfg = |toggle: &str| -> FitConfig {
            let toml_str = format!(
                "{toggle}\
                 [model]\ncamdl = \"models/seir_vaccine.camdl\"\n\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n\n\
                 [fixed]\nsigma = 0.2\ngamma = 0.1\nomega = 0.003\n\
                 vacc_frac = 0.8\nN0 = 100000\nI0 = 10\n\n\
                 [method]\nalgorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 1\n"
            );
            FitConfig::from_toml_str(&toml_str).unwrap_or_else(|e| panic!("parse: {e}"))
        };

        cfg("").validate_gradient_capability(&model).expect(
            "a declared-but-not-enabled intervention is not in the fitted \
             trajectory, so it cannot be a reason to refuse the gradient",
        );
        cfg("enable = [\"sia_round_1\"]\n\n").validate_gradient_capability(&model).expect(
            "an enabled fixed-fraction transfer at fixed times is a jump the \
             sensitivity crosses (gh#275 §1g); it must be admitted",
        );

        // The same intervention as a fixed-count transfer: its jump is
        // min(count, source), which is not differentiable — refused by name,
        // and only when enabled.
        let mut counted = model.clone();
        for iv in &mut counted.interventions {
            if iv.name == "sia_round_1" {
                iv.actions = vec![ir::intervention::Action::AbsoluteTransfer(
                    ir::intervention::AbsoluteTransfer {
                        src: "S".into(), dst: "V".into(), count: ir::expr::Expr::const_(500.0),
                    },
                )];
            }
        }
        cfg("").validate_gradient_capability(&counted)
            .expect("a disabled fixed-count transfer is not in the trajectory");
        let err = cfg("enable = [\"sia_round_1\"]\n\n")
            .validate_gradient_capability(&counted)
            .expect_err("an enabled fixed-count transfer cannot be crossed by the sensitivity");
        assert!(
            err.contains("sia_round_1") && err.contains("transfer(count"),
            "the refusal must name the intervention and the construct; got: {err}"
        );
        assert!(err.contains("nl-sbplx"), "the refusal must name the alternative; got: {err}");
    }

    /// The gate must run when the estimated parameter has a box only in the
    /// model (`beta : rate in [0.001, 0.5]`) and the fit.toml restates neither
    /// `bounds` nor `start` — the common shape. The gate needs a value for
    /// every parameter only to place fire times, and it used to return `Ok`
    /// when it could not find one, which let an unsupported effect through to
    /// fail every chain at run time instead of being refused here.
    #[test]
    fn the_gradient_gate_runs_when_an_estimated_parameter_has_only_model_bounds() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_vaccine.ir.json");
        let mut model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read golden: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse golden: {e}"));
        let beta = model.parameters.iter().find(|p| p.name == "beta").unwrap();
        assert!(beta.value.bounds().is_some(), "the fixture declares beta's box in the model");
        for iv in &mut model.interventions {
            iv.actions = vec![ir::intervention::Action::AbsoluteTransfer(
                ir::intervention::AbsoluteTransfer {
                    src: "S".into(), dst: "V".into(), count: ir::expr::Expr::const_(500.0),
                },
            )];
        }
        let toml_str = "enable = [\"sia_round_1\"]\n\n\
             [model]\ncamdl = \"models/seir_vaccine.camdl\"\n\n\
             [data.observations]\ncases = \"data/cases.tsv\"\n\n\
             [estimate]\nbeta = {}\n\n\
             [fixed]\nsigma = 0.2\ngamma = 0.1\nomega = 0.003\n\
             vacc_frac = 0.8\nN0 = 100000\nI0 = 10\n\n\
             [method]\nalgorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 1\n";
        let cfg = FitConfig::from_toml_str(toml_str).unwrap_or_else(|e| panic!("parse: {e}"));
        let err = cfg
            .validate_gradient_capability(&model)
            .expect_err("the gate must run, and refuse, without a fit.toml start or bounds");
        assert!(err.contains("transfer(count"), "got: {err}");
    }

    /// A three-stage `[stages]` file — the shape the split retires — is refused
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

        // The last-declared method is the one that becomes the file's method.
        assert!(err.starts_with("legacy table `[stages.evaluate]`"), "{err}");
        assert!(err.contains("rename to `[method]`") && err.contains("camdl fit run fit.toml"),
            "{err}");
        assert!(err.contains("put `[stages.mle]` and `[stages.posterior]` in its own file"),
            "{err}");
        // A method-name `init_mle` becomes a `@label` handle; a directory stays
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
            Some(ChainStarts::Spread(Spread::FromPosterior { source: Handle("@mle".into()), chains: None }))
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
        // on Bayesian-method params still produce the same error.
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
    fn from_scenario_refuses_an_estimated_param_set_by_a_composed_preset() {
        // An estimated param living in the composed child is refused, just as
        // a parent-level one is: the import would make it both fixed and
        // estimated.
        let model = model_with_compose_scenario(
            "baseline_compose", &[("beta", 0.3)],
            "fit_pinned", &[("gamma", 0.1), ("N0", 1000.0)]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline_compose".into()),
            values: IndexMap::new(),
        };
        let err = fixed.expand_from_scenario(&model, &estimate_set(&["gamma"])).unwrap_err();
        assert!(err.starts_with(
            "parameter `gamma` is both estimated and set by scenario `baseline_compose`, \
             which [fixed] from_scenario imports"), "{err}");
        assert!(err.contains("(set gamma = 0.1 from composed preset `fit_pinned`)"), "{err}");
        assert!(fixed.values.is_empty(), "a refused import imports nothing: {:?}", fixed.values);
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
                prior: None,
                start: None,
            });
        }
        m
    }

    #[test]
    fn from_scenario_refuses_an_estimated_param_the_scenario_sets() {
        // `from_scenario = "baseline"` imports every value baseline sets; with
        // `beta` and `gamma` in [estimate] both would be fixed and estimated.
        // One refusal names both, with the scenario's values.
        let model = model_with_scenario("baseline", &[
            ("beta", 0.3), ("gamma", 0.1), ("N0", 1000.0), ("I0", 10.0),
        ]);
        let mut fixed = FixedParams {
            from_file: None,
            from_scenario: Some("baseline".into()),
            values: IndexMap::new(),
        };
        let err = fixed.expand_from_scenario(&model, &estimate_set(&["beta", "gamma"]))
            .unwrap_err();
        assert_eq!(err,
            "parameters `beta`, `gamma` are both estimated and set by scenario `baseline`, \
             which [fixed] from_scenario imports\n  \
             [estimate]:  beta\n  \
             scenario:    beta = 0.3    (set beta = 0.3)\n  \
             [estimate]:  gamma\n  \
             scenario:    gamma = 0.1    (set gamma = 0.1)\n\
             A parameter cannot be both fixed (imported from the scenario) and estimated.\n\
             Fix: define a scenario that does not set `beta`, `gamma` and name it in \
             from_scenario, or drop `beta`, `gamma` from [estimate] to hold them at the \
             scenario's value.");
    }

    #[test]
    fn config_expand_fixed_from_scenario_refuses_an_estimated_param() {
        // The FitConfig-level wrapper forwards `&self.estimate`, so the
        // refusal sees which params are estimated.
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
        let err = config.problem.expand_fixed_from_scenario(&model).unwrap_err();
        assert!(err.starts_with("parameter `beta` is both estimated and set by scenario \
            `baseline`"), "{err}");
    }

    #[test]
    fn from_scenario_still_rejects_inline_override_with_different_value() {
        // An inline value with a DIFFERENT number for a scenario key is a
        // hard error (silent semantic mutation of a named scenario). Here
        // baseline.set.gamma = 0.1 but the fit.toml inlines gamma = 0.5.
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
            source: Handle("@base".into()), chains: None })));
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

    /// A `pfilter` method scores the log-likelihood at one θ, so the file must
    /// determine it: a `start` for every estimated parameter, a sourced point
    /// or cloud, or an explicitly requested `from_prior` draw. With none of
    /// those it is refused — it used to score one random prior draw (the
    /// default rule when every parameter has a prior) and report that as the
    /// filter's log-likelihood.
    #[test]
    fn pfilter_needs_a_determined_theta() {
        let model = model_with_scenario("baseline", &[]);
        let cfg_with = |estimate: &str, starts: &str| -> FitConfig {
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
algorithm = "pfilter"
backend = "chain_binomial"
particles = 100
{starts}
"#)).unwrap()
        };
        let with_prior = "beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }";
        let with_start = "beta = { bounds = [0.01, 2.0], start = 0.4, prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }";

        // Priors and nothing else: refused, not a random prior draw.
        let cfg = cfg_with(with_prior, "");
        let mut m = method(&cfg);
        let err = m.resolve_starts(&cfg.problem, &model)
            .expect_err("pfilter with no determined θ is refused");
        assert!(err.contains("beta") && err.contains("from_prior") && err.contains("start"), "{err}");
        // A spread rule does not determine θ either.
        for rule in ["lhs", "uniform", "uniform_unconstrained", "single"] {
            let cfg = cfg_with(with_prior, &format!("starts = \"{rule}\""));
            let mut m = method(&cfg);
            assert!(m.resolve_starts(&cfg.problem, &model).is_err(), "{rule} is refused");
        }
        // Every parameter has a start: that point, under `single`.
        let cfg = cfg_with(with_start, "");
        let mut m = method(&cfg);
        m.resolve_starts(&cfg.problem, &model).unwrap();
        assert_eq!(m.starts, Some(ChainStarts::Point(Point::Declared)));
        // `single` with every start declared, an explicit from_prior, and a
        // sourced point are accepted as written.
        for (estimate, starts) in [
            (with_start, "starts = \"single\""),
            (with_prior, "starts = \"from_prior\""),
            (with_prior, "starts = { from_params = \"theta.toml\" }"),
            (with_prior, "starts = { from_mle = \"@fit\" }"),
            (with_prior, "starts = { from_posterior = \"@fit\" }"),
        ] {
            let cfg = cfg_with(estimate, starts);
            let mut m = method(&cfg);
            m.resolve_starts(&cfg.problem, &model)
                .unwrap_or_else(|e| panic!("`{starts}` determines θ: {e}"));
        }
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
        let decided = m.resolve_starts(&cfg.problem, &model).unwrap();
        assert_eq!(decided, StartsResolution::DefaultFromPrior);
        assert_eq!(m.starts, Some(ChainStarts::from_prior()));
        assert!(decided.describe().contains("from_prior"), "{}", decided.describe());

        // One parameter without a prior → uniform_unconstrained, and the
        // reason names it.
        let cfg = problem_with(
            "beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }\n\
             gamma = { bounds = [0.01, 1.0] }");
        let mut m = method(&cfg);
        let decided = m.resolve_starts(&cfg.problem, &model).unwrap();
        assert_eq!(decided, StartsResolution::DefaultUniformUnconstrained {
            without_prior: vec!["gamma".into()] });
        assert_eq!(m.starts, Some(ChainStarts::uniform_unconstrained()));
        let said = decided.describe();
        assert!(said.contains("uniform_unconstrained") && said.contains("gamma"), "{said}");

        // An explicit flat prior is not a distribution to draw from either.
        let cfg = problem_with(
            "beta = { bounds = [0.01, 2.0], prior = { flat = {} } }");
        let mut m = method(&cfg);
        assert!(matches!(m.resolve_starts(&cfg.problem, &model).unwrap(),
            StartsResolution::DefaultUniformUnconstrained { .. }));

        // A spelled rule is left alone whatever the priors say.
        let cfg = problem_with("beta = { bounds = [0.01, 2.0] }");
        let mut m = method(&cfg);
        m.starts = Some(ChainStarts::Spread(Spread::Lhs));
        let decided = m.resolve_starts(&cfg.problem, &model).unwrap();
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

    /// gh#439 A2: `needs_state_grad` is true for exactly the one ODE gradient
    /// cell — `nl-lbfgs`, the consumer of the WrtPop state-Jacobian — and false for every other (algorithm, backend)
    /// combination, including the near-misses `mh` on `ode` (Bayesian on the
    /// ODE backend, but gradient-free) and `nl-sbplx` on `ode` (deterministic
    /// MLE on the ODE backend, but derivative-free); both must compile lean.
    #[test]
    fn needs_state_grad_only_for_the_ode_gradient_cells() {
        let cfg = |method: &str| -> FitConfig {
            let toml_str = format!(
                "[model]\ncamdl = \"models/sir.camdl\"\n\n\
                 [data.observations]\ncases = \"data/cases.tsv\"\n\n\
                 [estimate]\nbeta = {{ bounds = [0.01, 2.0] }}\n\n\
                 [fixed]\nN0 = 1000\n\n{method}"
            );
            FitConfig::from_toml_str(&toml_str).unwrap_or_else(|e| panic!("parse {method:?}: {e}"))
        };

        // nl-lbfgs + ode → the forward-sensitivity gradient, used for MLE
        // (true). Without this the model compiles lean and
        // the gradient preflight refuses every nl-lbfgs fit as
        // "--no-state-grad".
        assert!(
            cfg("[method]\nalgorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 2")
                .needs_state_grad(),
            "nl-lbfgs+ode drives the ODE forward-sensitivity gradient — needs the Jacobian"
        );

        // mh + ode → gradient-free Bayesian on the ODE backend → lean (false).
        assert!(
            !cfg("[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 100")
                .needs_state_grad(),
            "mh on ode is gradient-free — must compile lean"
        );

        // nl-sbplx + ode → derivative-free MLE on the ODE backend → lean (false).
        assert!(
            !cfg("[method]\nalgorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 2")
                .needs_state_grad(),
            "nl-sbplx is derivative-free — must compile lean"
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

    fn minimal_fit_methods() -> &'static str {
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
"#, minimal_fit_methods());
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
"#, minimal_fit_methods());
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
"#, minimal_fit_methods());
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
"#, minimal_fit_methods());
        let config = parse(&src).unwrap();
        config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()], InitLaw::Absent)
            .expect("[data] and [synthetic] together is a real-data fit with a recorded truth");
        assert!(!config.problem.is_synthetic_fit());
    }

    #[test]
    fn neither_data_nor_synthetic_errors() {
        let src = minimal_fit_methods().to_string();
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
"#, minimal_fit_methods());
        let config = parse(&single_src).unwrap();
        assert_eq!(config.inference.fit_seeds.unwrap(), vec![42u64]);

        let list_src = format!(r#"fit_seeds = [101, 102, 103]
{}
[data.observations]
cases = "data/cases.tsv"
"#, minimal_fit_methods());
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
"#, minimal_fit_methods());
        let config = parse(&src).unwrap();
        let err = config.validate(&["beta".into(), "gamma".into(), "N0".into(), "I0".into()], InitLaw::Absent)
            .unwrap_err();
        assert!(err.contains("duplicate"), "must reject duplicate fit seeds: {}", err);
    }

    // ── ic_free / conditioning support check (F1) ──────────────────────────
    //
    // `ic_free = true` needs an algorithm that drops y₁ from the accumulated
    // loglik AND a swarm whose particles differ in x₀. PGAS and the ODE-MLE
    // optimizers score every obs unconditionally, for every model. `pfilter`
    // and `pmmh` DO drop y₁, and their bootstrap PF
    // draws x₀ per particle — so they have the spread exactly when the MODEL's
    // `init { }` declares a law, and not when it computes every compartment
    // from an expression (gh#732). `validate()` therefore takes that model fact
    // as `InitLaw`.

    /// Model params for the ic_free fixtures (sir with beta/gamma/N0/I0).
    fn ic_free_model_params() -> Vec<String> {
        vec!["beta".into(), "gamma".into(), "N0".into(), "I0".into()]
    }

    #[test]
    fn ic_free_with_if2_method_still_validates() {
        // IF2 both drops y₁ from the accumulated loglik AND gives each
        // particle its own x₀ drawn from its own perturbed θ (gh#364), so
        // ic_free = true must NOT be rejected. Negative control for the
        // gh#732 refusals below: without this, "refuse everything" would
        // pass them all.
        let src = format!(
            "ic_free = true\n{}\n[data.observations]\ncases = \"data/cases.tsv\"\n",
            minimal_fit_methods()
        );
        let config = parse(&src).unwrap();
        config
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect("ic_free=true on an IF2 method must validate (IF2 honors conditioning)");
    }

    #[test]
    fn ic_free_with_pgas_method_is_rejected() {
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
            .expect_err("ic_free=true on a PGAS method must be rejected (PGAS ignores conditioning)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("pgas"), "error must name the offending method's algorithm: {err}");
    }

    // gh#347: a sampler method whose burn_in ≥ the run length retains ZERO
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

    // `thin = 0` keeps no well-defined set of draws: PGAS divided by it in its
    // startup banner (a panic after the leaf was claimed) and PMMH/MH kept one
    // draw per chain. It is refused at validation for all three.
    #[test]
    fn thin_zero_is_rejected() {
        for (algorithm, len_field) in [("mh", "iterations"), ("pmmh", "iterations"), ("pgas", "sweeps")] {
            let backend = if algorithm == "mh" {
                "backend = \"ode\"\n"
            } else {
                "backend = \"chain_binomial\"\nparticles = 100\n"
            };
            let src = format!(r#"[model]
camdl = "models/sir.camdl"
[data.observations]
cases = "data/cases.tsv"
[estimate]
beta = {{ bounds = [0.01, 2.0] }}
[fixed]
N0 = 1000
I0 = 5
gamma = 0.1
[method]
algorithm = "{algorithm}"
{backend}chains = 2
{len_field} = 8000
burn_in = 1000
thin = 0
"#);
            let config = parse(&src).unwrap_or_else(|e| panic!("{algorithm}: {e}"));
            let err = config.validate(&ic_free_model_params(), InitLaw::Absent)
                .expect_err("thin = 0 must be rejected");
            assert!(err.contains("thin must be ≥ 1"), "{algorithm}: error must name thin: {err}");
        }
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
    fn ic_free_with_ode_mle_method_is_rejected() {
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
            .expect_err("ic_free=true on an ODE-MLE method must be rejected (compute_ode_loglik ignores conditioning)");
        assert!(err.contains("ic_free"), "error must name ic_free: {err}");
        assert!(err.contains("nl-sbplx"), "error must name the offending algorithm: {err}");
    }

    /// gh#732. Plain PMMH (no `rho`) wraps the bootstrap particle filter,
    /// which draws x₀ per particle — but on a model whose `init { }` computes
    /// every compartment from an expression each of those draws returns the
    /// same state, so `ic_free`'s first reweight scores every particle
    /// identically and the run drops y₁ instead of conditioning on it. Refuse
    /// at config load, naming the reason.
    #[test]
    fn ic_free_with_plain_pmmh_method_is_rejected() {
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
            .expect_err("ic_free=true on a plain PMMH method must be rejected (gh#732)");
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
    fn ic_free_with_pfilter_method_is_rejected() {
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
            .expect_err("ic_free=true on a pfilter method must be rejected (gh#732)");
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
    fn ic_free_with_bootstrap_pf_methods_validates_when_the_model_draws_x0() {
        for (who, method) in [
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
                 [fixed]\nN0 = 1000\nI0 = 5\ngamma = 0.1\n\n{method}"
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

    // ── initial_state ──────────────────────────────────────────────────────
    //
    // The flag is IF2's perturbation timing, in IF2's own table. Every other
    // method perturbs nothing and has no `[method.perturbation]`: such a
    // config is refused at load rather than run without it.

    fn t0_fit(algorithm: &str, method_body: &str, declare_flag: bool) -> String {
        let flag = if declare_flag { "[method.perturbation]\nI0 = { initial_state = true }\n" } else { "" };
        format!(r#"[model]
camdl = "models/sir.camdl"

[data.observations]
cases = "data/cases.tsv"

[estimate]
I0 = {{ bounds = [1, 500] }}

[fixed]
beta = 0.3
gamma = 0.1
N0 = 1000

[method]
algorithm = "{algorithm}"
{method_body}
{flag}"#)
    }

    const IF2_BODY: &str = "backend = \"chain_binomial\"\nchains = 2\nparticles = 100\n\
                            iterations = 10\ncooling = 0.9\n";
    const PGAS_BODY: &str = "backend = \"chain_binomial\"\nchains = 2\nparticles = 500\n\
                             sweeps = 100\nburn_in = 10\n";

    #[test]
    fn initial_state_with_if2_method_validates() {
        parse(&t0_fit("if2", IF2_BODY, true)).unwrap()
            .validate(&ic_free_model_params(), InitLaw::Absent)
            .expect("IF2 honours initial_state — it is IF2's own timing");
    }

    /// Every other method refuses IF2's table, naming it, the method and the
    /// table it has instead, rather than running without it.
    #[test]
    fn a_perturbation_table_is_refused_for_non_if2_methods() {
        for (algo, body) in [
            ("pgas", PGAS_BODY),
            ("pmmh", "backend = \"chain_binomial\"\nchains = 2\nparticles = 100\n\
                      iterations = 2000\nburn_in = 500\n"),
            ("mh", "backend = \"ode\"\nchains = 2\niterations = 2000\nburn_in = 500\n"),
            ("nl-sbplx", "backend = \"ode\"\nchains = 1\n"),
        ] {
            let err = parse(&t0_fit(algo, body, true))
                .expect_err(&format!("{algo} must refuse a [method.perturbation] it does not read"));
            assert!(err.contains("[method.perturbation]") && err.contains(&format!("`{algo}`"))
                && err.contains("[method.proposal]"), "{algo}: {err}");
        }
    }

    /// `[method.proposal]` under IF2 is refused with the table IF2 reads.
    #[test]
    fn a_proposal_table_is_refused_under_if2() {
        let text = t0_fit("if2", IF2_BODY, false) + "[method.proposal]\nI0 = 0.3\n";
        let err = parse(&text).expect_err("IF2 has no proposal");
        assert!(err.contains("[method.proposal]") && err.contains("[method.perturbation]") && err.contains("`if2`"), "{err}");
    }

    /// A live config with the old `[estimate]` step keys is refused, naming
    /// the table each moved to for its method, and the formula that converts
    /// a natural-scale `rw_sd` to z.
    #[test]
    fn the_old_step_keys_are_refused_with_their_new_spelling() {
        let old = |algorithm: &str, body: &str| t0_fit(algorithm, body, false)
            .replace("I0 = { bounds = [1, 500] }", "I0 = { bounds = [1, 500], rw_sd = 2, perturb_only_at_t0 = true }");
        let err = parse(&old("if2", IF2_BODY)).expect_err("rw_sd moved");
        assert!(err.contains("[method.perturbation]") && err.contains("I0 = { sd = <sd_z>, initial_state = true }")
            && err.contains("rw_sd / |dθ/dz|"), "{err}");
        let pmmh = "backend = \"chain_binomial\"\nchains = 2\nparticles = 100\niterations = 2000\n";
        let err = parse(&old("pmmh", pmmh)).expect_err("rw_sd moved");
        assert!(err.contains("[method.proposal]") && err.contains("<5 × sd_z>")
            && err.contains("`perturb_only_at_t0` is IF2's timing"), "{err}");
        let err = parse(&old("pgas", PGAS_BODY)).expect_err("rw_sd moved");
        assert!(err.contains("`pgas` reads neither key"), "{err}");
    }

    /// Control: the same methods validate when no `[estimate]` entry declares
    /// the flag, so the accept above is not a fixture that would pass either
    /// way.
    #[test]
    fn a_config_without_the_flag_is_unaffected() {
        for (algo, body) in [
            ("pgas", PGAS_BODY),
            ("pmmh", "backend = \"chain_binomial\"\nchains = 2\nparticles = 100\n\
                      iterations = 2000\nburn_in = 500\n"),
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
        // PFilter method should record a PrequentialTrace by default —
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
            _ => panic!("expected PFilter method"),
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
            _ => panic!("expected PFilter method"),
        }
    }

    #[test]
    fn if2_method_loglik_eval_and_gate_default_when_omitted() {
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
            _ => panic!("expected IF2 method"),
        }
    }

    #[test]
    fn if2_method_loglik_eval_and_gate_parse_overrides() {
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
            _ => panic!("expected IF2 method"),
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
            _ => panic!("expected IF2 method"),
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
    fn make_pgas_method(sweeps: usize) -> Method {
        Method {
            algorithm: Algorithm::PGAS {
                backend: crate::run_meta::InferenceBackend::ChainBinomial,
                chains: 4, particles: 100, sweeps,
                burn_in: Some(200), thin: Some(2),
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
    fn make_pmmh_method(iterations: usize) -> Method {
        Method {
            algorithm: Algorithm::PMMH {
                backend: crate::run_meta::InferenceBackend::ChainBinomial,
                chains: 4, particles: 100, iterations,
                burn_in: Some(200), thin: Some(2),
                adapt: true, adapt_start: 300,
                binomial: sim::rng::BinomialAlgorithm::Btpe,
                proposal: IndexMap::new(),
            },
            starts: Some(ChainStarts::uniform_unconstrained()),
        }
    }

    #[test]
    fn pgas_identity_payload_omits_sweeps() {
        // Two PGAS methods identical except for `sweeps` must produce
        // the same identity_payload — that's the contract that lets
        // --resume extend a chain by changing the iteration count.
        let s_short = make_pgas_method(1000);
        let s_long = make_pgas_method(5000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());

        // Changing any *other* PGAS field must change the payload.
        let mut s_more_chains = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut chains, .. } = s_more_chains.algorithm { *chains = 8; }
        assert_ne!(s_short.identity_payload(), s_more_chains.identity_payload());
    }

    #[test]
    fn pgas_identity_payload_omits_n_trajectories() {
        // n_trajectories does not change the chain's draws, but it changes
        // the trajectories the run stores, so it is in the method's settings.
        let mut s_few = make_pgas_method(1000);
        let mut s_many = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut n_trajectories, .. } = s_few.algorithm { *n_trajectories = 100; }
        if let Algorithm::PGAS { ref mut n_trajectories, .. } = s_many.algorithm { *n_trajectories = 1000; }
        assert_ne!(s_few.identity_payload(), s_many.identity_payload(),
            "n_trajectories changes the stored trajectories, so it is in the method's settings");
    }

    #[test]
    fn pgas_identity_payload_includes_new_algorithmic_knobs() {
        // max_tree_depth, trajectory_warmup, csmc_sweeps_per_nuts,
        // dense_mass, use_nuts ALL change chain dynamics and MUST
        // invalidate identity.
        let base = make_pgas_method(1000);

        let mut s = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut max_tree_depth, .. } = s.algorithm { *max_tree_depth = 14; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "max_tree_depth");

        let mut s = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut trajectory_warmup, .. } = s.algorithm {
            *trajectory_warmup = 100;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(), "trajectory_warmup");

        let mut s = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut csmc_sweeps_per_nuts, .. } = s.algorithm {
            *csmc_sweeps_per_nuts = 3;
        }
        assert_ne!(base.identity_payload(), s.identity_payload(),
            "csmc_sweeps_per_nuts");

        let mut s = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut dense_mass, .. } = s.algorithm { *dense_mass = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "dense_mass");

        let mut s = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut use_nuts, .. } = s.algorithm { *use_nuts = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "use_nuts");
    }

    /// `starts` chooses the per-chain starting points, which determine the
    /// stored chains/posterior. Two fits differing only in it must not
    /// collide — otherwise the first run's posterior is silently served as the
    /// second's (a wrong scientific result on a multimodal problem). gh#147
    /// count-in-the-key. A sourced rule is keyed by its kind; which source it
    /// read keys through the method level's deps, so two spellings of one
    /// source are one run (gh#896).
    #[test]
    fn starts_is_in_the_method_identity() {
        for (who, base) in [("pgas", make_pgas_method(1000)), ("pmmh", make_pmmh_method(1000))] {
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
            let post_a = with(ChainStarts::Spread(Spread::FromPosterior { source: Handle("@a".into()), chains: None }));
            assert_ne!(base.identity_payload(), lhs.identity_payload(), "{who}: uniform_unconstrained vs lhs");
            assert_ne!(lhs.identity_payload(), single.identity_payload(), "{who}: lhs vs single");
            assert_ne!(lhs.identity_payload(), prior.identity_payload(), "{who}: lhs vs from_prior");
            assert_ne!(prior.identity_payload(), mle_a.identity_payload(), "{who}: from_prior vs from_mle");
            assert_eq!(mle_a.identity_payload(), mle_b.identity_payload(),
                "{who}: the handle as typed is not in the key; the source is, through deps");
            assert_ne!(mle_a.identity_payload(), post_a.identity_payload(), "{who}: from_mle vs from_posterior");
            // gh#1056: a chain subset keys its own run; the same subset in any
            // order keys one run.
            let sub = |ids: Vec<usize>| with(ChainStarts::Spread(Spread::FromPosterior {
                source: Handle("@a".into()),
                chains: Some(crate::chain_selection::ChainSelection::from_ids(
                    ids, crate::chain_selection::Polarity::Keep, "`chains`").unwrap()),
            }));
            assert_ne!(post_a.identity_payload(), sub(vec![0, 1]).identity_payload(), "{who}: whole cloud vs subset");
            assert_ne!(sub(vec![0, 1]).identity_payload(), sub(vec![0, 2]).identity_payload(), "{who}: two subsets");
            assert_eq!(sub(vec![1, 0]).identity_payload(), sub(vec![0, 1]).identity_payload(), "{who}: subset order");
            assert_eq!(post_a.identity_payload()["starts"], serde_json::json!("from_posterior"), "{who}");
            // A sourced rule hashes as its kind; a bare rule as its TOML spelling.
            let v = mle_a.identity_payload();
            assert_eq!(v["starts"], serde_json::json!("from_mle"), "{who}");
            assert_eq!(lhs.identity_payload()["starts"], serde_json::json!("lhs"), "{who}");
        }
    }

    #[test]
    fn pmmh_identity_payload_omits_iterations() {
        let s_short = make_pmmh_method(1000);
        let s_long = make_pmmh_method(8000);
        assert_eq!(s_short.identity_payload(), s_long.identity_payload());
    }

    /// gh#747: the two samplers must not collide in the store.
    ///
    /// Written against the payload, the record of the method's identity, so it
    /// checks the value the run is keyed by.
    #[test]
    fn binomial_sampler_is_in_the_method_identity() {
        let mut btpe = make_pgas_method(1000);
        let mut btrs = make_pgas_method(1000);
        if let Algorithm::PGAS { ref mut binomial, .. } = btrs.algorithm {
            *binomial = sim::rng::BinomialAlgorithm::Btrs;
        }
        if let Algorithm::PGAS { ref mut binomial, .. } = btpe.algorithm {
            *binomial = sim::rng::BinomialAlgorithm::Btpe;
        }
        let a = serde_json::to_string(&btpe.identity_payload()).unwrap();
        let b = serde_json::to_string(&btrs.identity_payload()).unwrap();
        assert_ne!(a, b,
            "btpe and btrs methods produced the SAME identity payload — the two \
             samplers would share one address, and one sampler's posterior \
             would be served from the other's leaf. That is the exact failure \
             an environment variable would have caused, and the reason this is \
             a typed field.");
        assert!(b.contains(r#""binomial":"Btrs""#),
            "a btrs method's payload must name the sampler; got {b}");
        assert!(a.contains(r#""binomial":"Btpe""#),
            "a btpe method's payload must name the sampler; got {a}");
    }

    /// gh#747: the TOML spelling round-trips, and an absent field is BTPE.
    #[test]
    fn binomial_sampler_parses_from_method_toml() {
        let with_btrs: Algorithm = toml::from_str(
            "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 4\n\
             particles = 100\nsweeps = 1000\nbinomial = \"btrs\"\n").unwrap();
        match with_btrs {
            Algorithm::PGAS { binomial, .. } =>
                assert_eq!(binomial, sim::rng::BinomialAlgorithm::Btrs),
            _ => panic!("expected a PGAS method"),
        }
        // Absent -> BTPE. Every fit.toml written before this field relies on it.
        let absent: Algorithm = toml::from_str(
            "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 4\n\
             particles = 100\nsweeps = 1000\n").unwrap();
        match absent {
            Algorithm::PGAS { binomial, .. } =>
                assert_eq!(binomial, sim::rng::BinomialAlgorithm::Btpe,
                    "an absent `binomial` must mean BTPE"),
            _ => panic!("expected a PGAS method"),
        }
    }

    /// gh#747, extended past PGAS: every particle-filter method parses
    /// `binomial = "btrs"`, an absent field is BTPE, a BTRS method gets a
    /// distinct identity that names the sampler, and a BTPE one names its own.
    /// One body for the three methods,
    /// so a method added to the field later cannot skip half the checks.
    #[test]
    fn binomial_sampler_parses_and_keys_on_every_particle_method() {
        use sim::rng::BinomialAlgorithm::{Btpe, Btrs};
        let heads = [
            "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\n\
             particles = 100\niterations = 10\n",
            "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
             particles = 100\niterations = 10\ncooling = 0.7\n",
            "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 100\n",
        ];
        for head in heads {
            let parse = |extra: &str| -> Method {
                toml::from_str(&format!("{head}{extra}")).unwrap_or_else(|e| panic!("{head}: {e}"))
            };
            let absent = parse("");
            let btrs = parse("binomial = \"btrs\"\n");
            assert_eq!(absent.algorithm.binomial(), Btpe, "{head}: absent must mean BTPE");
            assert_eq!(btrs.algorithm.binomial(), Btrs, "{head}: the TOML spelling must parse");
            let a = serde_json::to_string(&absent.identity_payload()).unwrap();
            let b = serde_json::to_string(&btrs.identity_payload()).unwrap();
            assert_ne!(a, b, "{head}: btpe and btrs share one identity payload");
            assert!(b.contains(r#""binomial":"Btrs""#), "{head}: btrs payload must name it; got {b}");
            assert!(a.contains(r#""binomial":"Btpe""#), "{head}: btpe payload must name it; got {a}");
        }
    }

    /// The record of a fixed PGAS method's identity, as `identity.json` holds
    /// it, with the starts rule: what a run's chain identity holds of its
    /// method, and what a resumed run's base must match. A change here re-keys
    /// every PGAS run.
    #[test]
    fn identity_payload_is_byte_stable_against_recompiles() {
        let method = make_pgas_method(1000);
        let payload_str = serde_json::to_string(&method.identity_payload()).unwrap();
        let expected = r#"{"method":{"Pgas":{"ancestor_sampling":true,"backend":"ChainBinomial","binomial":"Btpe","burn_in":200,"chains":4,"csmc_sweeps_per_nuts":1,"dense_mass":true,"max_tree_depth":10,"n_trajectories":200,"particles":100,"thin":2,"trajectory_warmup":0,"use_nuts":true}},"starts":"uniform_unconstrained"}"#;
        assert_eq!(payload_str, expected,
            "the PGAS method identity's record changed, which re-keys every \
             PGAS run; if that is intended, update this constant and say so \
             in the commit");
    }

    /// As [`identity_payload_is_byte_stable_against_recompiles`], for PMMH.
    #[test]
    fn pmmh_identity_payload_byte_stable() {
        let method = make_pmmh_method(1000);
        let payload_str = serde_json::to_string(&method.identity_payload()).unwrap();
        let expected = r#"{"method":{"Pmmh":{"adapt":true,"adapt_start":300,"backend":"ChainBinomial","binomial":"Btpe","burn_in":200,"chains":4,"particles":100,"proposal":{},"thin":2}},"starts":"uniform_unconstrained"}"#;
        assert_eq!(payload_str, expected,
            "the PMMH method identity's record changed, which re-keys every \
             PMMH run; if that is intended, update this constant and say so \
             in the commit");
    }

    #[test]
    fn pmmh_identity_payload_includes_new_algorithmic_knobs() {
        let base = make_pmmh_method(1000);

        let mut s = make_pmmh_method(1000);
        if let Algorithm::PMMH { ref mut adapt, .. } = s.algorithm { *adapt = false; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt");

        let mut s = make_pmmh_method(1000);
        if let Algorithm::PMMH { ref mut adapt_start, .. } = s.algorithm { *adapt_start = 1000; }
        assert_ne!(base.identity_payload(), s.identity_payload(), "adapt_start");
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
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            perturbation: IndexMap::new(),
        });
        let s100 = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 100, cooling: 0.95,
            cooling_target_iters: 50,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            perturbation: IndexMap::new(),
        });
        assert_ne!(s50.identity_payload(), s100.identity_payload());

        let s_diff_cooling = if2_method(Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 4, particles: 100, iterations: 50, cooling: 0.70,
            cooling_target_iters: 50,
            loglik_eval: LoglikEvalConfig::default(),
            gate: GateConfig::default(),
            dt_check: DtCheckConfig::default(),
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            perturbation: IndexMap::new(),
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
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            perturbation: IndexMap::new(),
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
    /// `NloptConfig` and could answer differently (gh#881).
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
            "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 4\n",
            "algorithm = \"nl-bobyqa\"\nbackend = \"ode\"\nchains = 4\n",
            "algorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 4\n",
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
        // A typo'd key (here `iteration`, a near-miss for the method's
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

    /// The support each hierarchical family is compared against, and the
    /// rule that a box cuts in when it leaves out any of it — including an
    /// end set by an expression, which no finite box can be shown to contain.
    #[test]
    fn hierarchical_box_cuts_into_support_per_family() {
        use ir::parameter::{HierarchicalKind as K, HierarchicalPrior};
        let konst = |v: f64| ir::expr::Expr::Const(ir::expr::ConstExpr { value: v });
        let param = |n: &str| ir::expr::Expr::Param(ir::expr::ParamExpr { param: n.into() });
        let h = |kind: K, args: Vec<(&str, ir::expr::Expr)>| HierarchicalPrior {
            kind, args: args.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            pool_over: String::new(),
        };
        let inf = f64::INFINITY;
        let normal = hierarchical_support(&h(K::Normal, vec![("mu", param("m")), ("sigma", konst(1.0))]));
        assert!(box_cuts_into_support((1.0, 2.0), normal));
        assert!(box_cuts_into_support((0.0, inf), normal));
        assert!(!box_cuts_into_support((-inf, inf), normal));
        let lognormal = hierarchical_support(&h(K::LogNormal, vec![("mu", param("m")), ("sigma", konst(1.0))]));
        assert!(!box_cuts_into_support((0.0, inf), lognormal));
        assert!(box_cuts_into_support((0.0, 5.0), lognormal));
        let beta = hierarchical_support(&h(K::Beta, vec![("alpha", param("a")), ("beta", konst(2.0))]));
        assert!(!box_cuts_into_support((0.0, 1.0), beta));
        assert!(box_cuts_into_support((0.1, 1.0), beta));
        let uni_const = hierarchical_support(&h(K::Uniform, vec![("lower", konst(0.0)), ("upper", konst(3.0))]));
        assert!(!box_cuts_into_support((0.0, 3.0), uni_const));
        let uni_expr = hierarchical_support(&h(K::Uniform, vec![("lower", konst(0.0)), ("upper", param("u"))]));
        assert!(box_cuts_into_support((0.0, 10.0), uni_expr));
        assert!(!box_cuts_into_support((0.0, inf), uni_expr));
    }
}
