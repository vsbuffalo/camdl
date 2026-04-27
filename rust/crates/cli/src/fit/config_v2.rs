//! New fit.toml types (run-spec v0.4).
//!
//! Coexists with config.rs (FitToml) during migration. The old `camdl fit scout`
//! etc. commands use FitToml; the new `camdl fit run` uses FitConfigV2.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

// ─── Top-level ──────────────────────────────────────────────────────────────

/// A fit.toml v2 — single inference task with named stages.
#[derive(Debug, Clone, Deserialize, Serialize)]
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

    /// Optional lineage metadata (not used by the runner).
    #[serde(default)]
    pub provenance: Option<FitProvenance>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelRef {
    pub camdl: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FitBackendConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_dt")]
    pub dt: f64,
}
fn default_backend() -> String { "chain_binomial".to_string() }
fn default_dt() -> f64 { 1.0 }
impl Default for FitBackendConfig {
    fn default() -> Self { FitBackendConfig { backend: default_backend(), dt: default_dt() } }
}

// ─── Data ───────────────────────────────────────────────────────────────────

/// Data file mapping. Keys in `observations` match observation stream names
/// declared in the .camdl file's `observations { }` block. The observation
/// model (likelihood family) and projection (which flow/compartment to
/// accumulate) are defined in the .camdl file — fit.toml only provides the
/// data file paths.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataSpec {
    /// Map from observation stream name → data file path.
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

// ─── Synthetic data ──────────────────────────────────────────────────────────

/// Synthetic-data generation spec. Mutually exclusive with `[data]`:
/// when present, the runner generates `len(sim_seeds)` datasets from
/// `true_params` using the model's observation block, then fits each
/// one. Output directory structure places these under `synthetic/ds_NN/`
/// — see docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.
#[derive(Debug, Clone, Deserialize, Serialize)]
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
}

impl SyntheticSpec {
    pub fn validate(&self) -> Result<(), String> {
        // Ensure sim_seeds is non-empty and has no duplicates.
        let seeds = self.sim_seeds.to_vec();
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
    pub fn to_vec(&self) -> Vec<u64> {
        match self {
            SeedsSpec::List(xs) => xs.clone(),
            SeedsSpec::Range(s) => parse_seed_range(s).unwrap_or_default(),
        }
    }

    pub fn validate_no_duplicates(&self) -> Result<(), String> {
        let v = self.to_vec();
        let mut seen = BTreeSet::new();
        for s in &v {
            if !seen.insert(*s) {
                return Err(format!(
                    "duplicate seed {} — each seed must be unique to avoid \
                     provenance-hash collisions between fits", s));
            }
        }
        if let SeedsSpec::Range(s) = self {
            if v.is_empty() {
                return Err(format!(
                    "malformed seed range '{}' — use 'start:end' with start ≤ end, \
                     e.g. '1:20'", s));
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
pub struct EstimateSpecV2 {
    /// Search bounds. Required.
    pub bounds: (f64, f64),

    /// Transform for inference. If omitted, inferred from the parameter's
    /// declared type in the .camdl file.
    #[serde(default)]
    pub transform: Option<Transform>,

    /// Prior distribution. Required for Bayesian methods (PGAS, PMMH).
    /// Optional for MLE (IF2 ignores priors).
    #[serde(default)]
    pub prior: Option<PriorSpec>,

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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "dist")]
pub enum PriorSpec {
    #[serde(rename = "log_normal")]
    LogNormal { mu: f64, sigma: f64 },
    #[serde(rename = "normal")]
    Normal { mu: f64, sigma: f64 },
    #[serde(rename = "beta")]
    Beta { alpha: f64, beta: f64 },
    #[serde(rename = "uniform")]
    Uniform,
    #[serde(rename = "half_normal")]
    HalfNormal { sigma: f64 },
}

// ─── Fixed ──────────────────────────────────────────────────────────────────

/// Fixed parameters. Supports bulk loading from a file + inline overrides.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixedParams {
    /// Bulk load from a TOML file (all key=value pairs become fixed).
    #[serde(default)]
    pub from_file: Option<String>,

    /// Inline fixed values. Override from_file if both specify a key.
    #[serde(flatten)]
    pub values: IndexMap<String, f64>,
}

impl FixedParams {
    /// Resolve to a concrete map: load from_file, then overlay inline values.
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

/// A named inference stage. Tagged by method.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "method")]
pub enum Stage {
    #[serde(rename = "if2")]
    IF2 {
        chains: usize,
        particles: usize,
        iterations: usize,
        /// Fraction of initial perturbation magnitude remaining at the final
        /// iteration. 0.70 = perturbations shrink to 70% over the full run.
        cooling: f64,
        #[serde(default)]
        starts_from: StartsFrom,
        /// Clean-evaluation re-scoring of candidate parameter points after
        /// IF2 finishes. See proposal §Proposal 1. Defaults give 4000
        /// particles × 8 replicates combined via logmeanexp.
        #[serde(default)]
        clean_eval: CleanEvalConfig,
        /// Compound gate thresholds for chain agreement (Â) and
        /// inter-chain log-likelihood spread (decibans). See proposal
        /// §Proposal 3.
        #[serde(default)]
        gate: GateConfig,
    },

    #[serde(rename = "pgas")]
    PGAS {
        chains: usize,
        particles: usize,
        sweeps: usize,
        #[serde(default)]
        starts_from: StartsFrom,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,
    },

    #[serde(rename = "pmmh")]
    PMMH {
        chains: usize,
        particles: usize,
        iterations: usize,
        #[serde(default)]
        starts_from: StartsFrom,
        #[serde(default)]
        burn_in: Option<usize>,
        #[serde(default)]
        thin: Option<usize>,
    },

    #[serde(rename = "pfilter")]
    PFilter {
        particles: usize,
        #[serde(default)]
        replicates: Option<usize>,
        #[serde(default)]
        starts_from: StartsFrom,
    },
}

impl Stage {
    pub fn starts_from(&self) -> &StartsFrom {
        match self {
            Stage::IF2 { starts_from, .. }
            | Stage::PGAS { starts_from, .. }
            | Stage::PMMH { starts_from, .. }
            | Stage::PFilter { starts_from, .. } => starts_from,
        }
    }

    pub fn method_name(&self) -> &str {
        match self {
            Stage::IF2 { .. } => "if2",
            Stage::PGAS { .. } => "pgas",
            Stage::PMMH { .. } => "pmmh",
            Stage::PFilter { .. } => "pfilter",
        }
    }

    pub fn requires_priors(&self) -> bool {
        matches!(self, Stage::PGAS { .. } | Stage::PMMH { .. })
    }

    /// Return the gate config for IF2 stages; `None` for Bayesian/PFilter.
    pub fn gate_config(&self) -> Option<&GateConfig> {
        match self {
            Stage::IF2 { gate, .. } => Some(gate),
            _ => None,
        }
    }

    pub fn chains(&self) -> usize {
        match self {
            Stage::IF2 { chains, .. } => *chains,
            Stage::PGAS { chains, .. } => *chains,
            Stage::PMMH { chains, .. } => *chains,
            Stage::PFilter { .. } => 1,
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct CleanEvalConfig {
    /// Particle count per clean PF replicate. Must be ≫ in-run scout
    /// particle count to bring SE under control.
    #[serde(default = "default_clean_eval_particles")]
    pub n_particles: usize,
    /// Independent PF replicates per candidate. Combined via `combine`.
    #[serde(default = "default_clean_eval_replicates")]
    pub n_replicates: usize,
    #[serde(default)]
    pub combine: CombineMode,
}

fn default_clean_eval_particles() -> usize { 4000 }
fn default_clean_eval_replicates() -> usize { 8 }

impl Default for CleanEvalConfig {
    fn default() -> Self {
        Self {
            n_particles: default_clean_eval_particles(),
            n_replicates: default_clean_eval_replicates(),
            combine: CombineMode::default(),
        }
    }
}

/// Compound scout-convergence gate: chain agreement (Â) AND inter-chain
/// log-likelihood spread (decibans, with an SE-aware floor). See
/// proposal §Proposal 3.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

/// Where a stage gets its initial parameter values.
/// Deserialized from a string. If the string contains `/` or `\`, it's a
/// directory path; if it equals "random", it's random starts; otherwise
/// it's a stage name reference.
#[derive(Debug, Clone, Serialize)]
#[derive(Default)]
pub enum StartsFrom {
    /// Name of a previous stage in this fit.toml (e.g., "mle").
    Stage(String),
    /// Path to an external results directory.
    Directory(PathBuf),
    /// Random starts from parameter bounds.
    #[default]
    Random,
}


impl<'de> serde::Deserialize<'de> for StartsFrom {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        // Contains path separator → directory path
        if s.contains('/') || s.contains('\\') {
            Ok(StartsFrom::Directory(PathBuf::from(s)))
        } else {
            // Bare name → stage reference (never interpreted as Random;
            // Random is only the Default when starts_from is omitted)
            Ok(StartsFrom::Stage(s))
        }
    }
}

// ─── Provenance ─────────────────────────────────────────────────────────────

/// Optional metadata linking this fit to a parent.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FitProvenance {
    pub derived_from: Option<String>,
    pub reason: Option<String>,
}

// ─── Legacy bridge ──────────────────────────────────────────────────────────

impl FitConfigV2 {
    /// Convert to a legacy FitToml for runner compatibility.
    /// This is a bridge during migration — the runners will eventually
    /// accept FitConfigV2 directly.
    pub fn to_legacy_toml(&self) -> Result<super::config::FitToml, String> {
        use super::config::*;

        let fixed_resolved = self.fixed.resolve()?;
        let fixed_legacy: HashMap<String, toml::Value> = fixed_resolved.iter()
            .map(|(k, v)| {
                let tv = if *v == (*v as i64) as f64 && v.abs() < 1e15 {
                    toml::Value::Integer(*v as i64)
                } else {
                    toml::Value::Float(*v)
                };
                (k.clone(), tv)
            })
            .collect();

        let estimate_legacy: HashMap<String, EstimateSpec> = self.estimate.iter()
            .map(|(name, spec)| {
                let prior_str = spec.prior.as_ref().map(|p| match p {
                    PriorSpec::LogNormal { mu, sigma } => format!("lognormal({}, {})", mu, sigma),
                    PriorSpec::Normal { mu, sigma } => format!("normal({}, {})", mu, sigma),
                    PriorSpec::Beta { alpha, beta } => format!("beta({}, {})", alpha, beta),
                    PriorSpec::Uniform => "uniform".to_string(),
                    PriorSpec::HalfNormal { sigma } => format!("halfnormal({})", sigma),
                });
                (name.clone(), EstimateSpec {
                    rw_sd: spec.rw_sd,
                    ivp: spec.ivp,
                    start: spec.start,
                    transform: spec.transform.as_ref().map(|t| match t {
                        Transform::Log => "log".to_string(),
                        Transform::Logit => "logit".to_string(),
                        Transform::Identity => "identity".to_string(),
                    }),
                    bounds: Some(spec.bounds),
                    prior: prior_str,
                })
            })
            .collect();

        // Extract IF2-specific stage configs for legacy sections
        let mut scout_config = None;
        let mut refine_config = None;
        let mut validate_config = None;
        let mut pgas_config = None;
        let mut pmmh_config = None;

        for (name, stage) in &self.stages {
            match stage {
                Stage::IF2 { chains, particles, iterations, cooling, clean_eval, gate, .. } => {
                    if *clean_eval != CleanEvalConfig::default() || *gate != GateConfig::default() {
                        eprintln!(
                            "[warning] to_legacy_toml: stage '{}' has custom clean_eval/gate config \
                             that the legacy bridge cannot represent — defaults will be used. \
                             Use 'camdl fit run' to preserve these settings.",
                            name
                        );
                    }
                    let sc = StageConfig {
                        chains: Some(*chains),
                        particles: Some(*particles),
                        iterations: Some(*iterations),
                        cooling: Some(*cooling),
                        rw_sd_scale: None,
                        start_chains: None,
                    };
                    // Map to the closest legacy stage name
                    match name.as_str() {
                        "scout" | "mle" => scout_config = Some(sc),
                        "refine" => refine_config = Some(sc),
                        "validate" | "evaluate" => {
                            validate_config = Some(ValidateConfig {
                                chains: Some(*chains),
                                particles: Some(*particles),
                                iterations: Some(*iterations),
                                cooling: Some(*cooling),
                                rw_sd_scale: None,
                                pfilter_particles: None,
                            });
                        }
                        _ => {
                            // Custom IF2 stage name — use as scout config
                            if scout_config.is_none() { scout_config = Some(sc); }
                        }
                    }
                }
                Stage::PGAS { chains, sweeps, particles, burn_in, thin, .. } => {
                    pgas_config = Some(PGASSampleConfig {
                        chains: Some(*chains),
                        sweeps: Some(*sweeps),
                        particles: Some(*particles),
                        burn_in: *burn_in,
                        thin: *thin,
                        starts_from: None,
                        n_trajectories: None,
                        tempering: None,
                        max_treedepth: None,
                        trajectory_warmup: None,
                        csmc_sweeps_per_nuts: None,
                    });
                }
                Stage::PMMH { chains, particles, iterations, burn_in, thin, .. } => {
                    pmmh_config = Some(PMMHSampleConfig {
                        chains: Some(*chains),
                        steps: Some(*iterations),
                        particles: Some(*particles),
                        burn_in: *burn_in,
                        thin: *thin,
                        adapt: None,
                        adapt_start: None,
                        proposal_from: None,
                        rho: None,
                    });
                }
                Stage::PFilter { .. } => {}
            }
        }

        let data = self.data_spec()?;
        let data_legacy: HashMap<String, String> = data.observations.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let holdout_legacy = data.holdout.as_ref().map(|h| {
            h.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        });

        Ok(FitToml {
            fit: FitSection {
                model: self.model.camdl.clone(),
                output_dir: self.output_dir.clone().unwrap_or_else(
                    || crate::run_paths::DEFAULT_OUTPUT_ROOT.to_string()),
                seed: None,
                scenario: self.scenario.clone(),
                enable: self.enable.clone(),
                disable: self.disable.clone(),
                ic_free: self.ic_free.unwrap_or(false),
            },
            data: data_legacy,
            holdout: holdout_legacy,
            config: FitConfigSection {
                backend: self.config.backend.clone(),
                dt: self.config.dt,
            },
            estimate: estimate_legacy,
            fixed: fixed_legacy,
            scout: scout_config,
            refine: refine_config,
            validate: validate_config,
            pmmh: pmmh_config,
            pgas: pgas_config,
        })
    }
}

// ─── Loading + Validation ───────────────────────────────────────────────────

impl FitConfigV2 {
    pub fn load(path: &str) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        let config: FitConfigV2 = toml::from_str(&contents)
            .map_err(|e| format!("parse error in {}: {}", path, e))?;

        Ok(config)
    }

    /// Seed-independent content hash for the fit directory. Keyed on
    /// (model IR, data files, fit.toml bytes, version). Any edit to
    /// these produces a new hash → new directory, so previous fit
    /// results are never silently overwritten when the config changes.
    pub fn fit_content_hash(&self, config_path: &str) -> Result<String, String> {
        let fit_bytes = std::fs::read(config_path)
            .map_err(|e| format!("cannot read fit.toml at '{}': {}", config_path, e))?;
        let model_ir_bytes = std::fs::read(&self.model.camdl)
            .map_err(|e| format!("cannot read model at '{}': {}", self.model.camdl, e))?;
        let mut data_files: Vec<(String, Vec<u8>)> = Vec::new();
        if let Some(data) = &self.data {
            for (name, path) in &data.observations {
                let bytes = std::fs::read(path)
                    .map_err(|e| format!("cannot read data file '{}' ({}): {}", name, path, e))?;
                data_files.push((name.clone(), bytes));
            }
        }
        // Synthetic fits (no [data]) derive data deterministically
        // from `true_params` + sim_seeds inside the fit.toml, so the
        // fit hash captures them via the fit.toml bytes.
        Ok(crate::hashing::fit_content_hash(&model_ir_bytes, &mut data_files, &fit_bytes))
    }

    /// Output directory for this fit: `<root>/fits/<stem>-<hash[:8]>/`.
    /// Stem from the fit.toml basename; hash from
    /// [`fit_content_hash`]. Recognisable directory names, content-
    /// addressable cache keys, no silent overwrites.
    pub fn fit_dir(&self, config_path: &str) -> Result<PathBuf, String> {
        let stem = crate::hashing::path_stem_slug(config_path);
        let hash = self.fit_content_hash(config_path)?;
        let output_root = crate::run_paths::output_root(
            None, self.output_dir.as_deref());
        Ok(crate::run_paths::fit_run_dir(&output_root, stem.as_deref(), &hash))
    }

    /// The per-fit subdirectory under `fit_dir()` — always
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
             - add a Bayesian stage:   [stages.pgas] method = \"pgas\"\n    \
             - use priors for starts:  fit_starts = \"prior\"\n    \
             - remove the priors:      drop `prior = {{...}}` from [estimate.*] entries",
            params_with_priors.join(", ")))
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

        // Bayesian stages require priors on all estimated params
        for (stage_name, stage) in &self.stages {
            if stage.requires_priors() {
                let missing_priors: Vec<&str> = self.estimate.iter()
                    .filter(|(_, spec)| spec.prior.is_none())
                    .map(|(name, _)| name.as_str())
                    .collect();
                if !missing_priors.is_empty() {
                    return Err(format!(
                        "stage '{}' (method={}) requires priors, but missing for: {}\n  \
                         Add prior = {{ dist = \"...\", ... }} to each [estimate.*] entry.",
                        stage_name, stage.method_name(), missing_priors.join(", ")
                    ));
                }
            }
        }

        // Validate backend
        let valid_backends = ["gillespie", "tau_leap", "chain_binomial", "ode"];
        if !valid_backends.contains(&self.config.backend.as_str()) {
            return Err(format!(
                "unknown backend '{}'. Valid backends: {}",
                self.config.backend, valid_backends.join(", ")
            ));
        }

        // Validate stage DAG: starts_from references must be valid
        self.validate_stage_dag()?;

        // Validate bounds
        for (name, spec) in &self.estimate {
            let (lo, hi) = spec.bounds;
            if lo >= hi {
                return Err(format!(
                    "estimate.{}: bounds [{}, {}] are empty (lo must be < hi)",
                    name, lo, hi
                ));
            }
        }

        // N2: validate gate configs — a_thresh must be > 1.0 because
        // Â ≥ 1.0 by construction (it's a convergence ratio), so any
        // threshold ≤ 1.0 would make Gate 1 always fail.
        for (stage_name, stage) in &self.stages {
            if let Some(gate) = stage.gate_config() {
                if gate.a_thresh <= 1.0 {
                    return Err(format!(
                        "stages.{}: gate.a_thresh = {:.4} must be > 1.0 — \
                         Â ≥ 1.0 by construction so a threshold ≤ 1.0 \
                         makes Gate 1 permanently fail. \
                         Try a_thresh = 1.01 (the default).",
                        stage_name, gate.a_thresh
                    ));
                }
            }
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
        toml::from_str(toml_str)
            .map_err(|e| format!("parse error: {}", e))
    }

    #[test]
    fn parse_simple_mle() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
backend = "chain_binomial"
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
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
gamma = { bounds = [0.05, 1.0], prior = { dist = "log_normal", mu = -2.0, sigma = 1.0 } }
rho   = { bounds = [0.001, 1.0], prior = { dist = "beta", alpha = 2.0, beta = 5.0 } }
k     = { bounds = [0.1, 100.0], prior = { dist = "half_normal", sigma = 10.0 } }

[fixed]
beta = 0.34
N0 = 1000000
I0 = 10

[stages.mle]
method = "if2"
chains = 4
particles = 2000
iterations = 60
cooling = 0.95
starts_from = "output/fits/01_all_free/mle"

[stages.posterior]
method = "pgas"
chains = 4
particles = 50
sweeps = 5000
starts_from = "mle"

[stages.evaluate]
method = "pfilter"
particles = 10000
replicates = 100
starts_from = "mle"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 5.0] }

[fixed]
from_file = "params/fixed.toml"
vacc_frac = 0.80

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }

[fixed]
N0 = 1000000
I0 = 10

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
beta = 0.5
N0 = 1000000

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }
typo_param = { bounds = [0.0, 1.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
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
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.posterior]
method = "pgas"
chains = 4
particles = 50
sweeps = 5000
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("requires priors"));
        assert!(err.contains("beta"));
    }

    #[test]
    fn validate_bad_stage_dag() {
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.refine]
method = "if2"
chains = 4
particles = 2000
iterations = 50
cooling = 0.95
starts_from = "mle"

[stages.mle]
method = "if2"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
starts_from = "nonexistent"
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [2.0, 0.01] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
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
        let config = parse(r#"
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[config]
backend = "gilelspie"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        let model_params = vec!["beta".to_string(), "N0".to_string()];
        let err = config.validate(&model_params).unwrap_err();
        assert!(err.contains("unknown backend"));
        assert!(err.contains("gilelspie"));
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
backend = "chain_binomial"
dt = 1.0

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
N0 = 1000000

[stages.mle]
method = "if2"
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
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
        "#).unwrap();

        assert_eq!(config.config.backend, "chain_binomial");
        assert_eq!(config.config.dt, 1.0);
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
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70
starts_from = "output/fits/01/mle"
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
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.70

[stages.refine]
method = "if2"
chains = 2
particles = 2000
iterations = 30
cooling = 0.95
starts_from = "mle"
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
method = "if2"
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
method = "if2"
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
method = "if2"
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
method = "if2"
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
        assert_eq!(syn.datasets.unwrap_or_else(|| syn.sim_seeds.to_vec().len()), 20);
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
        assert_eq!(syn.sim_seeds.to_vec().len(), 3);
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
        assert_eq!(s.to_vec(), vec![1u64, 2, 3, 4, 5]);
        s.validate_no_duplicates().unwrap();
    }

    #[test]
    fn seeds_inverted_range_errors() {
        let s = SeedsSpec::Range("10:5".into());
        assert_eq!(s.to_vec(), Vec::<u64>::new());
        let err = s.validate_no_duplicates().unwrap_err();
        assert!(err.contains("malformed") || err.contains("start ≤ end"),
            "inverted range must surface a clear error: {}", err);
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
beta  = { bounds = [0.01, 2.0], prior = { dist = "log_normal", mu = -0.3, sigma = 0.5 } }
gamma = { bounds = [0.05, 1.0], prior = { dist = "half_normal", sigma = 1.0 } }

[fixed]
N0 = 1000

[stages.scout]
method = "if2"
chains = 4
particles = 500
iterations = 50
cooling = 0.7

[stages.refine]
method = "if2"
chains = 4
particles = 1000
iterations = 50
cooling = 0.9
starts_from = "scout"
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
method = "pgas"
chains = 4
particles = 1000
sweeps = 1000
starts_from = "refine"
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
method = "if2"
chains = 2
particles = 500
iterations = 50
cooling = 0.7
"#;
        let config = parse(src).unwrap();
        assert!(config.dangling_priors_warning().is_none(),
            "no priors declared — nothing to warn about");
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
method = "if2"
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
        });
        assert_eq!(cfg.per_fit_prefix(101, Some(2)),
                   std::path::PathBuf::from("synthetic").join("ds_02").join("fit_101"));
    }

    #[test]
    fn if2_stage_clean_eval_and_gate_default_when_omitted() {
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
method = "if2"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
        "#).unwrap();

        match &cfg.stages["scout"] {
            Stage::IF2 { clean_eval, gate, .. } => {
                assert_eq!(clean_eval.n_particles, 4000);
                assert_eq!(clean_eval.n_replicates, 8);
                assert_eq!(clean_eval.combine, CombineMode::LogMeanExp);
                assert!((gate.a_thresh - 1.01).abs() < 1e-12);
                assert!((gate.decibans_thresh - 30.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }
    }

    #[test]
    fn if2_stage_clean_eval_and_gate_parse_overrides() {
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
method = "if2"
chains = 4
particles = 500
iterations = 30
cooling = 0.9
clean_eval = { n_particles = 8000, n_replicates = 16, combine = "mean" }
gate = { a_thresh = 1.05, decibans_thresh = 60.0 }

[stages.refine]
method = "if2"
chains = 4
particles = 1000
iterations = 60
cooling = 0.95

[stages.refine.clean_eval]
n_particles = 12000

[stages.refine.gate]
decibans_thresh = 100.0
        "#).unwrap();

        match &cfg.stages["scout"] {
            Stage::IF2 { clean_eval, gate, .. } => {
                assert_eq!(clean_eval.n_particles, 8000);
                assert_eq!(clean_eval.n_replicates, 16);
                assert_eq!(clean_eval.combine, CombineMode::Mean);
                assert!((gate.a_thresh - 1.05).abs() < 1e-12);
                assert!((gate.decibans_thresh - 60.0).abs() < 1e-12);
            }
            _ => panic!("expected IF2 stage"),
        }

        // refine: partial overrides — unset fields take defaults
        match &cfg.stages["refine"] {
            Stage::IF2 { clean_eval, gate, .. } => {
                assert_eq!(clean_eval.n_particles, 12000);
                assert_eq!(clean_eval.n_replicates, 8);            // default
                assert_eq!(clean_eval.combine, CombineMode::LogMeanExp); // default
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
}
