//! Shared run-metadata types for the content-addressed `output/` tree.
//!
//! Every run is a `runid::RunRecord` leaf on disk; the per-kind readers live in
//! `cas_read`, `browse`, and `fit::fit_view`. This module keeps the small,
//! cross-cutting value types those readers and writers share:
//!
//! - [`FitAlgorithm`] / [`InferenceBackend`] — the (algorithm, backend) tags a
//!   fit stage records and a reader projects.
//! - [`SurveyEvalMethod`] — how `camdl survey` evaluates the marginal
//!   log-likelihood at each point.
//! - [`ResolvedPriorEntry`] / [`ParameterProvenance`] / [`InitProvenance`] —
//!   the gh#75 / gh#83/gh#85 provenance records embedded in a leaf's `inputs`
//!   and in the fit-level [`FitSidecar`].
//! - [`FitSidecar`] + [`write_fit_sidecar`] / [`read_fit_sidecar`] — the
//!   fit-level provenance sidecar (`fit.meta.json`) a CAS fit segment carries
//!   alongside its stage leaves.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Inference algorithm tag — discriminator enum naming the algorithm
/// independent of the simulation backend. Recorded in a fit-stage leaf's
/// `inputs` alongside `Backend` to capture the (algorithm, backend) pair the
/// stage ran. Wire format matches the lowercased / kebab-cased name
/// the user writes in fit.toml (`algorithm = "if2"`, `algorithm =
/// "nl-sbplx"`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FitAlgorithm {
    #[serde(rename = "if2")]      If2,
    #[serde(rename = "pgas")]     Pgas,
    #[serde(rename = "pmmh")]     Pmmh,
    #[serde(rename = "mh")]       Mh,
    #[serde(rename = "nuts")]     Nuts,
    #[serde(rename = "pfilter")]  Pfilter,
    #[serde(rename = "nl-sbplx")] NlSbplx,
    #[serde(rename = "nl-bobyqa")] NlBobyqa,
}

impl FitAlgorithm {
    /// Wire-format string. Matches the `algorithm = "..."` value in fit.toml
    /// and the `inputs.method` serialized form on a fit-stage leaf.
    pub fn as_str(self) -> &'static str {
        match self {
            FitAlgorithm::If2      => "if2",
            FitAlgorithm::Pgas     => "pgas",
            FitAlgorithm::Pmmh     => "pmmh",
            FitAlgorithm::Mh       => "mh",
            FitAlgorithm::Nuts     => "nuts",
            FitAlgorithm::Pfilter  => "pfilter",
            FitAlgorithm::NlSbplx  => "nl-sbplx",
            FitAlgorithm::NlBobyqa => "nl-bobyqa",
        }
    }

    /// The per-stage convergence-summary filename a Bayesian sampler writes into
    /// its stage dir: `<algorithm>_summary.json`. The single source of this
    /// naming convention — written by the pgas / pmmh / mh / nuts runners, read
    /// back by `read_convergence` and the `MethodResult` loaders. Deterministic
    /// mh-ODE shares the PMMH runner but gets its OWN file (`mh_summary.json`),
    /// not the misleading `pmmh_summary.json`.
    pub fn summary_filename(self) -> String {
        format!("{}_summary.json", self.as_str())
    }

    /// A Bayesian posterior sampler: it explores the full posterior and writes a
    /// `draws.tsv` cloud, so a completed run has a posterior band to draw. PGAS
    /// (the default Bayesian path), PMMH, and MH are samplers. This is the
    /// authoritative sampler/optimizer partition used to frame `fit predict`'s
    /// refusals — a sampler with no resolvable draws is *incomplete*, never an
    /// "optimizer fit" (gh#343).
    pub fn is_posterior_sampler(self) -> bool {
        matches!(
            self,
            FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Mh | FitAlgorithm::Nuts
        )
    }

    /// An optimizer: it returns a single best-fit point (no posterior cloud), so
    /// a predictive can only ever plug those parameters in — there is no band.
    /// IF2 and the NLopt family (sbplx / bobyqa) are optimizers. `Pfilter` is a
    /// likelihood evaluation, not a fit, so it is *neither* a sampler nor an
    /// optimizer.
    pub fn is_optimizer(self) -> bool {
        matches!(
            self,
            FitAlgorithm::If2 | FitAlgorithm::NlSbplx | FitAlgorithm::NlBobyqa
        )
    }
}

impl std::fmt::Display for FitAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Simulation backend the stage ran on. The (algorithm, backend) pair
/// is constrained by `methods::METHODS`; PF-based algorithms require
/// `chain_binomial`, deterministic-likelihood algorithms require `ode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    ChainBinomial,
    Ode,
}

impl InferenceBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::ChainBinomial => "chain_binomial",
            InferenceBackend::Ode           => "ode",
        }
    }
}

impl std::fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Conversion failure from a [`ForwardBackend`](crate::args::types::ForwardBackend)
/// into an [`InferenceBackend`]: `gillespie` is a valid forward-simulation
/// backend but has no fit/inference interface, so it cannot back a fit stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDomainError {
    NotInferenceBackend,
}

impl std::fmt::Display for BackendDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendDomainError::NotInferenceBackend => f.write_str(
                "gillespie is a forward-simulation backend, not an inference backend; \
                 fit stages support only chain_binomial or ode",
            ),
        }
    }
}

impl std::error::Error for BackendDomainError {}

impl TryFrom<crate::args::types::ForwardBackend> for InferenceBackend {
    type Error = BackendDomainError;

    fn try_from(b: crate::args::types::ForwardBackend) -> Result<Self, Self::Error> {
        use crate::args::types::ForwardBackend as F;
        match b {
            F::ChainBinomial => Ok(Self::ChainBinomial),
            F::Ode => Ok(Self::Ode),
            F::Gillespie => Err(BackendDomainError::NotInferenceBackend),
        }
    }
}

/// Every inference backend is also a valid forward-simulation backend (the fit
/// dynamics are a forward model). Total — used to record the stage's actual
/// backend into forward-facing provenance (`MleMetadata.backend`).
impl From<InferenceBackend> for crate::args::types::ForwardBackend {
    fn from(b: InferenceBackend) -> Self {
        match b {
            InferenceBackend::ChainBinomial => Self::ChainBinomial,
            InferenceBackend::Ode => Self::Ode,
        }
    }
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
    /// otherwise. Resolved before any persistent state is written — the
    /// survey `run.json` records the resolved method, never `Auto`.
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

/// The run's machine-readable observation/dimension schema — a faithful
/// projection of the model's *expanded* observation structure, emitted into
/// `fit.meta.json` so a consumer can facet any stream by its index dimensions
/// and label panels by level name with no DSL parsing.
///
/// Derived as a pure fold over the model's observation leaves
/// ([`ObsSchema::from_model`]). It reads the **same** IR the particle filter
/// binds, so the schema cannot disagree with what was fit — it is derived
/// provenance, never a second source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsSchema {
    /// Each indexing dimension → its ordered levels (union over all streams).
    /// `BTreeMap` so the serialized order is deterministic.
    pub dimensions: std::collections::BTreeMap<String, DimensionLevels>,
    /// One descriptor per **logical** stream — grouped by the data-source key,
    /// so a stratified stream `cases[p in patch]` is a single entry carrying
    /// `index_dims = ["patch"]`, never one entry per expanded leaf.
    pub streams: Vec<StreamDescriptor>,
}

/// A dimension's ordered levels (e.g. `patch → [Bo, Bombali, …]`). A struct
/// rather than a bare `Vec` so the JSON shape (`{"levels": [...]}`) matches the
/// proposal and leaves room for future per-dimension metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionLevels {
    pub levels: Vec<String>,
}

/// One logical observation stream's structure, read straight off the IR record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamDescriptor {
    /// Logical stream name — the `from <label>` data-source key, stable across
    /// the expanded leaves of a stratified stream (the expander suffixes the
    /// leaf `name`, e.g. `cases_Bo`, but every leaf shares this `source`).
    pub name: String,
    /// The dimensions this stream is stratified over, in first-appearance
    /// order. `[]` for a single national series. A consumer facets by these.
    pub index_dims: Vec<String>,
    /// The scored value column — the `~` LHS the likelihood scores.
    pub value_column: String,
    /// DSL kind of the scored value (`count`/`real`/`probability`/…). Absent
    /// only when the model predates the explicit `columns {}` block (no
    /// declared role to read), so a consumer treats `None` as "unspecified".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_kind: Option<String>,
    /// Likelihood family (`poisson`/`neg_binomial`/…), matching the IR tag.
    pub likelihood: String,
}

impl ObsSchema {
    /// Fold a fully-expanded model's observation leaves into the descriptor.
    ///
    /// Groups leaves by their logical `source` key (stable across a stratified
    /// stream's leaves); collects index dims, value column/kind, and likelihood
    /// off each IR record. Dimension levels are the union of stratum levels
    /// seen for that dimension, in first-appearance order. Pure and total — no
    /// I/O, no failure path.
    pub fn from_model(model: &ir::Model) -> ObsSchema {
        Self::from_observations(&model.observations)
    }

    /// The fold proper — over just the observation leaves the filter binds.
    /// `from_model` is a thin wrapper; the schema reads nothing else off the
    /// model, which is exactly why it cannot disagree with what was fit.
    pub fn from_observations(
        observations: &[ir::observation::ObservationModel],
    ) -> ObsSchema {
        use std::collections::BTreeMap;

        struct Acc {
            index_dims:   Vec<String>,
            value_column: String,
            value_kind:   Option<String>,
            likelihood:   String,
        }
        // First-appearance order of logical streams (declaration order), kept
        // separately because `acc`'s BTreeMap iteration is name-sorted.
        let mut order: Vec<String> = Vec::new();
        let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
        let mut dims: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for obs in observations {
            let value_kind = obs.columns.iter()
                .find(|c| c.name == obs.scored)
                .and_then(|c| match &c.role {
                    ir::observation::ColumnRole::Value(k) => Some(k.as_str().to_string()),
                    _ => None,
                });
            let entry = acc.entry(obs.source.clone()).or_insert_with(|| {
                order.push(obs.source.clone());
                Acc {
                    index_dims:   Vec::new(),
                    value_column: obs.scored.clone(),
                    value_kind,
                    likelihood:   obs.likelihood.name().to_string(),
                }
            });
            for sk in &obs.stratum {
                if !entry.index_dims.contains(&sk.dim) {
                    entry.index_dims.push(sk.dim.clone());
                }
                let levels = dims.entry(sk.dim.clone()).or_default();
                if !levels.contains(&sk.level) {
                    levels.push(sk.level.clone());
                }
            }
        }

        let streams = order.into_iter().map(|src| {
            let a = acc.remove(&src).expect("source recorded in `order` is in `acc`");
            StreamDescriptor {
                name:         src,
                index_dims:   a.index_dims,
                value_column: a.value_column,
                value_kind:   a.value_kind,
                likelihood:   a.likelihood,
            }
        }).collect();

        let dimensions = dims.into_iter()
            .map(|(d, levels)| (d, DimensionLevels { levels }))
            .collect();

        ObsSchema { dimensions, streams }
    }
}

/// The fit-level provenance sidecar (`fits/{stem}-{h8}/fit.meta.json`). A CAS
/// fit's fit level is a path segment with no `RunRecord`; this sidecar is the
/// single authoritative home for the fit-wide attributes that are NOT carried
/// on the stage leaves — the user `--label` and the fit-wide provenance
/// (`resolved_priors` = gh#75 per-parameter prior source,
/// `estimated`/`fixed`/`data_hashes`, `model_identity`, paths).
///
/// It is **derived provenance, not a source of truth**: a faithful readable
/// projection of inputs already hashed into the leaf identity (the `FitDigest`
/// — different priors already produce a different fit identity). It is written
/// post-identity and is never fed back into any hash. The producing `fit.toml`
/// is archived beside it as `fit.toml.original`: the config-diff source for
/// `fit table`, the config a run handle recovers, and — canonicalised at lookup
/// time — what a `fit.toml` handle is matched against (gh#653).
///
/// Every field except `resolved_priors`-class provenance defaults, so partial
/// sidecars (test fixtures) round-trip; [`crate::fit::fit_view::FitView`]
/// enforces that a Bayesian fit's `resolved_priors` is present (no silent
/// default).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FitSidecar {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub model_identity: String,
    /// Where the producing `fit.toml` lived when the fit ran. Load-bearing, not
    /// decorative: the archived `fit.toml.original` beside this sidecar carries
    /// the config's text but not its location, and its relative `[model]` /
    /// `[data]` paths were written against THIS directory — so recovering the
    /// config from the segment resolves them here (gh#652,
    /// [`crate::fit::config_v2::FitConfigV2::load_anchored_at`]). Empty for a
    /// CLI-only fit, which has no config to archive and no directory to anchor.
    #[serde(default)]
    pub fit_toml_path: String,
    /// SHA-256 of the producing config's raw bytes — **provenance**: which exact
    /// bytes produced this fit, down to the whitespace. Nothing looks a fit up
    /// by it.
    ///
    /// The neighbouring question — does a config in hand MEAN what this fit was
    /// run from? — is a different fact and has a different answer: the canonical
    /// hash of the parsed value tree ([`crate::fit::cas::config_identity_hash`]),
    /// computed from `fit.toml.original` when a `fit.toml` handle is resolved.
    /// Reflowing a comment changes this field and not that one, which is the
    /// point: it is why the raw hash cannot serve as identity (gh#653) and why
    /// the canonical hash cannot serve as provenance.
    #[serde(default)]
    pub fit_toml_hash: String,
    // gh#542: the three maps below are `BTreeMap`, not `HashMap`, for the same
    // reason gh#519 changed `FitState` — serde emits a map in ITERATION order,
    // and only `serde_json::Value` normalises (its `Map` is a `BTreeMap` with
    // `preserve_order` off). A `HashMap` here made `fit.meta.json` differ
    // between two identical runs by key order alone, which defeats diffing two
    // runs' metadata. The ordering requirement belongs to the artifact, so the
    // type says so.
    #[serde(default)]
    pub data_hashes: BTreeMap<String, String>,
    #[serde(default)]
    pub estimated: Vec<String>,
    #[serde(default)]
    pub fixed: BTreeMap<String, f64>,
    #[serde(default)]
    pub resolved_priors: Vec<ResolvedPriorEntry>,
    #[serde(default)]
    pub parameters_provenance: BTreeMap<String, ParameterProvenance>,
    /// The run's observation/dimension schema ([`ObsSchema`]) — `streams` ×
    /// `dimensions` derived from the model's observation leaves. `None` for a
    /// sidecar written without a model in hand (CLI-only profile fits, test
    /// fixtures). The parameter *roles* are not re-nested here: `estimated` /
    /// `fixed` above are the single source of truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<ObsSchema>,
    /// The model's `#'` documentation dictionary ([`ir::ModelDocs`]): the
    /// model's own header block (`docs.model`, gh#750) plus params /
    /// compartments / transitions / observations / dimensions / quantities →
    /// `{symbol, text, ref}`, keyed by base declaration name. A downstream
    /// consumer labels any output column (posterior-draw parameter names,
    /// trajectory compartments, predict streams, generated quantities) by joining
    /// its name against this, and answers "a fit of WHAT model?" from
    /// `docs.model` without re-reading the `.camdl`. Empty (and omitted) when
    /// the model documents nothing.
    #[serde(default, skip_serializing_if = "ir::ModelDocs::is_empty")]
    pub docs: ir::ModelDocs,
}

/// Write the fit-level sidecar and archive the producing `fit.toml`
/// (`fit.toml.original`, the config-diff source `fit table` loads). The archive
/// is best-effort: a CLI-only fit (no `.toml`) has none and config-diff degrades
/// to identity. Idempotent; the caller writes it once per fit segment.
///
/// The user `label` is **sticky** (gh#29). A multi-stage pipeline
/// (`scout → refine → validate`) rewrites this sidecar on every `fit run`
/// invocation, but the sidecar records the *experiment*, not a single call, so a
/// `--label` set on an earlier invocation must survive a later stage-only re-run
/// that passes no `--label`. When `sidecar.label` is `None`, a non-`None` label
/// already on disk at `fit_segment` is preserved; an explicit `Some(label)`
/// always overrides; a fresh segment with no prior label stays `None`. (The
/// same one home is what `fit label` relabels post-hoc.) Every other field is a
/// derived projection of the current invocation's inputs and is overwritten as
/// before.
pub fn write_fit_sidecar(
    fit_segment: &std::path::Path,
    fit_toml_path: &std::path::Path,
    sidecar: &FitSidecar,
) -> std::io::Result<()> {
    std::fs::create_dir_all(fit_segment)?;
    if fit_toml_path.is_file() {
        std::fs::copy(fit_toml_path, fit_segment.join("fit.toml.original"))?;
        // gh#353: archive the model `.camdl` source too, symmetric with
        // fit.toml.original, so the leaf is self-contained (source + config +
        // IR) and a viewer/reproduction doesn't depend on the original checkout
        // layout. Best-effort and .camdl-only: a model supplied directly as
        // `.ir.json` is already captured by the leaf's `model.ir.json`, and a
        // missing/unreadable source must not fail the fit. The recorded
        // `model_path` is `config.model.camdl`, resolved exactly as the fit
        // loaded it — via `resolve_ir_path` / `std::fs::read`, i.e. against the
        // process CWD (a relative path resolves against CWD; an absolute path
        // stays absolute). Resolving it against the fit.toml's directory would
        // disagree with the loader and silently skip the archive whenever the
        // fit is run from a directory other than the fit.toml's parent.
        if sidecar.model_path.ends_with(".camdl") {
            let src = std::path::Path::new(&sidecar.model_path);
            if src.is_file() {
                let _ = std::fs::copy(src, fit_segment.join("model.camdl.original"));
            }
        }
    }
    // gh#29: keep the label sticky. The `.or_else` reads the on-disk sidecar
    // only when this write carries no label, and we clone only when a prior
    // label actually differs — the common override/fresh paths pay nothing.
    let effective_label = sidecar.label.clone()
        .or_else(|| read_fit_sidecar(fit_segment).and_then(|s| s.label));
    let bytes = if effective_label == sidecar.label {
        serde_json::to_vec_pretty(sidecar)
    } else {
        let mut merged = sidecar.clone();
        merged.label = effective_label;
        serde_json::to_vec_pretty(&merged)
    }
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(fit_segment.join("fit.meta.json"), bytes)
}

/// Read the fit-level sidecar; `None` when absent (an incomplete segment —
/// [`crate::fit::fit_view::FitView`] treats that as a malformed fit and skips
/// it loudly rather than fabricating empty provenance).
pub fn read_fit_sidecar(fit_segment: &std::path::Path) -> Option<FitSidecar> {
    let bytes = std::fs::read(fit_segment.join("fit.meta.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

    /// gh#542: `fit.meta.json` must be byte-reproducible. serde emits a map in
    /// ITERATION order and only `serde_json::Value` normalises, so a `HashMap`
    /// here made two identical runs differ by key order alone — which defeats
    /// diffing two runs' metadata, the thing the sidecar exists for.
    ///
    /// Asserts the emitted key ORDER, not just that two serialisations of the
    /// same instance agree: within one process a `HashMap` iterates the same
    /// way twice, so a round-trip check would pass with the bug present.
    #[test]
    fn fit_meta_json_emits_sorted_keys() {
        let mut side = FitSidecar::default();
        for k in ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"] {
            side.data_hashes.insert(k.to_string(), "h".to_string());
            side.fixed.insert(k.to_string(), 1.0);
        }
        let json = serde_json::to_string(&side).unwrap();

        for field in ["data_hashes", "fixed"] {
            let start = json.find(&format!("\"{field}\"")).expect("field present");
            let seen: Vec<&str> = ["alpha", "bravo", "charlie", "mike", "yankee", "zulu"]
                .iter()
                .map(|k| (json[start..].find(&format!("\"{k}\"")).unwrap(), *k))
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_values()
                .collect();
            assert_eq!(seen, ["alpha", "bravo", "charlie", "mike", "yankee", "zulu"],
                "`{field}` must serialise in sorted key order, got {seen:?} — \
                 insertion order was deliberately unsorted, so this fails if the \
                 field goes back to a HashMap");
        }
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_filename_matches_algorithm_name() {
        // The naming seam: each sampler's summary is `<algorithm>_summary.json`.
        // In particular mh gets its OWN file, never the pmmh one.
        assert_eq!(FitAlgorithm::Pgas.summary_filename(), "pgas_summary.json");
        assert_eq!(FitAlgorithm::Pmmh.summary_filename(), "pmmh_summary.json");
        assert_eq!(FitAlgorithm::Mh.summary_filename(), "mh_summary.json");
        assert_eq!(FitAlgorithm::Nuts.summary_filename(), "nuts_summary.json");
        // mh and pmmh must not collide on disk.
        assert_ne!(
            FitAlgorithm::Mh.summary_filename(),
            FitAlgorithm::Pmmh.summary_filename()
        );
    }

    #[test]
    fn survey_eval_method_serializes_lowercase() {
        let p = SurveyEvalMethod::Pfilter;
        assert_eq!(serde_json::to_string(&p).unwrap(), r#""pfilter""#);
        let s = SurveyEvalMethod::Simulate;
        assert_eq!(serde_json::to_string(&s).unwrap(), r#""simulate""#);
    }

    // ─── gh#241: backend domain types (ForwardBackend / InferenceBackend) ──

    /// Zero-re-key guarantee: `InferenceBackend` (renamed from
    /// `run_meta::Backend`) serializes to the same snake_case string it always
    /// has, so the fit blob is byte-identical and no `run_id` moves.
    #[test]
    fn inference_backend_serde_spelling_is_snake_case() {
        assert_eq!(serde_json::to_string(&InferenceBackend::ChainBinomial).unwrap(),
                   r#""chain_binomial""#);
        assert_eq!(serde_json::to_string(&InferenceBackend::Ode).unwrap(),
                   r#""ode""#);
    }

    /// Zero-re-key guarantee for gh#241 PR C: renaming `MethodKind` ->
    /// `FitAlgorithm` must not change the serialized wire spelling — it is
    /// stored in a fit-stage leaf's `inputs.method`, part of the factored
    /// fit-stage identity. Pins serialize, deserialize, and `as_str` together.
    #[test]
    fn fit_algorithm_serde_spelling_is_pinned() {
        let cases = [
            (FitAlgorithm::If2, "if2"),
            (FitAlgorithm::Pgas, "pgas"),
            (FitAlgorithm::Pmmh, "pmmh"),
            (FitAlgorithm::Mh, "mh"),
            (FitAlgorithm::Pfilter, "pfilter"),
            (FitAlgorithm::NlSbplx, "nl-sbplx"),
            (FitAlgorithm::NlBobyqa, "nl-bobyqa"),
        ];
        for (algo, wire) in cases {
            assert_eq!(serde_json::to_string(&algo).unwrap(), format!("\"{wire}\""));
            assert_eq!(algo.as_str(), wire);
            assert_eq!(
                serde_json::from_str::<FitAlgorithm>(&format!("\"{wire}\"")).unwrap(),
                algo
            );
        }
    }

    /// The sampler/optimizer partition is the authoritative classifier `fit
    /// predict` uses to frame its refusals (gh#343). PGAS / PMMH / MH are
    /// Bayesian samplers (a completed run has a posterior band); IF2 and the
    /// NLopt family are optimizers (a single point, refused); a particle filter
    /// is neither (a likelihood eval, not a fit). Every variant is covered so a
    /// new algorithm cannot silently fall into "neither" unnoticed.
    #[test]
    fn fit_algorithm_sampler_optimizer_partition() {
        for m in [FitAlgorithm::Pgas, FitAlgorithm::Pmmh, FitAlgorithm::Mh] {
            assert!(m.is_posterior_sampler(), "{m} is a posterior sampler");
            assert!(!m.is_optimizer(), "{m} is not an optimizer");
        }
        for m in [FitAlgorithm::If2, FitAlgorithm::NlSbplx, FitAlgorithm::NlBobyqa] {
            assert!(m.is_optimizer(), "{m} is an optimizer");
            assert!(!m.is_posterior_sampler(), "{m} is not a posterior sampler");
        }
        // A particle filter is a likelihood evaluation, not a fit — neither.
        assert!(!FitAlgorithm::Pfilter.is_posterior_sampler());
        assert!(!FitAlgorithm::Pfilter.is_optimizer());
    }

    /// `ForwardBackend` (renamed from `args::types::Backend`) keeps its wire
    /// spelling too — the sim/config identity surfaces are unchanged.
    #[test]
    fn forward_backend_serde_spelling_unchanged() {
        use crate::args::types::ForwardBackend;
        assert_eq!(serde_json::to_string(&ForwardBackend::Gillespie).unwrap(),
                   r#""gillespie""#);
        assert_eq!(serde_json::to_string(&ForwardBackend::ChainBinomial).unwrap(),
                   r#""chain_binomial""#);
        assert_eq!(serde_json::to_string(&ForwardBackend::Ode).unwrap(),
                   r#""ode""#);
    }

    /// The load-bearing type-boundary property: a forward `Gillespie` cannot
    /// become a fit/inference backend.
    #[test]
    fn gillespie_is_not_an_inference_backend() {
        use crate::args::types::ForwardBackend;
        assert_eq!(InferenceBackend::try_from(ForwardBackend::ChainBinomial),
                   Ok(InferenceBackend::ChainBinomial));
        assert_eq!(InferenceBackend::try_from(ForwardBackend::Ode),
                   Ok(InferenceBackend::Ode));
        assert_eq!(InferenceBackend::try_from(ForwardBackend::Gillespie),
                   Err(BackendDomainError::NotInferenceBackend));
    }

    /// The reverse is total: every inference backend is a valid forward
    /// backend (used to record stage provenance into `MleMetadata.backend`).
    #[test]
    fn inference_backend_is_always_a_forward_backend() {
        use crate::args::types::ForwardBackend;
        assert_eq!(ForwardBackend::from(InferenceBackend::ChainBinomial),
                   ForwardBackend::ChainBinomial);
        assert_eq!(ForwardBackend::from(InferenceBackend::Ode),
                   ForwardBackend::Ode);
    }

    // ─── gh#83/gh#85 step 9: parameter / init provenance round-trip ──

    /// Round-trips `ParameterProvenance` from a resolved parameter
    /// through JSON serialization (the shape a leaf's
    /// `inputs.parameters_provenance` carries). Covers audit checklist
    /// item 4: every entry's `source` matches a `ValueSource`
    /// variant tag.
    #[test]
    fn parameter_provenance_round_trips() {
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
        // Round-trip the provenance map directly through JSON (the shape it
        // takes in a leaf's `inputs.parameters_provenance`).
        let json = serde_json::to_string(&parameters_provenance).unwrap();
        let meta: HashMap<String, ParameterProvenance> =
            serde_json::from_str(&json).unwrap();
        // Non-empty, per audit item 4.
        assert!(!meta.is_empty(), "parameters_provenance must be populated");
        assert_eq!(meta.len(), 5);
        // Every entry's `source` matches a `ValueSource` variant tag.
        let allowed_source_tags: std::collections::HashSet<&str> = [
            "model_default", "scenario", "fit_toml_fixed",
            "fixed_file", "fixed_cli",
        ].iter().copied().collect();
        for (name, prov) in &meta {
            assert!(allowed_source_tags.contains(prov.source.as_str()),
                "{}: source tag {} not in ValueSource variants",
                name, prov.source);
            assert!(prov.role == "fixed" || prov.role == "estimated",
                "role must be fixed|estimated, got {}", prov.role);
        }
        // Specific assertions: kick_from_estimate present on rho/iota;
        // overrode_scenario present on rho only.
        let rho = &meta["rho"];
        assert!(rho.kicked_from_estimate.is_some());
        assert_eq!(rho.kicked_from_estimate.as_ref().unwrap().by, "fixed_cli");
        assert!(rho.overrode_scenario.is_some());
        assert_eq!(rho.overrode_scenario.as_ref().unwrap().scenario, "worst_case");
        assert!((rho.overrode_scenario.as_ref().unwrap().scenario_value - 0.30).abs() < 1e-12);
        let beta = &meta["beta"];
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
            (InitMethod::UniformUnconstrained, "uniform_unconstrained"),
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
    /// write → `FitView::read` read round trip. Before the sidecar carried
    /// `resolved_priors`, the reader defaulted it empty and this class of bug
    /// shipped silently. Write a Bayesian (`pgas`) stage leaf + a sidecar with
    /// mixed sources, read the fit back, and assert each `.source` matches.
    #[test]
    fn fit_sidecar_resolved_priors_survive_fit_view_round_trip() {
        use crate::fit::fit_view::FitView;
        let tmp = crate::test_support::unique_temp_dir("sidecar_priors");
        let seg = tmp.join("fits").join("demo-abc12345");
        let leaf = seg.join("01-posterior-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        // A Bayesian (pgas) stage leaf — `FitView::read` requires its
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

        let view = FitView::read(&seg).expect("FitView::read must derive a fit entry");
        let source = |p: &str| -> Option<&str> {
            view.resolved_priors
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

    /// gh#29: the fit-level sidecar `label` is sticky across repeated
    /// `fit run` invocations on the same fit segment. A multi-stage pipeline
    /// (`scout → refine → validate`) rewrites the umbrella sidecar on every
    /// call; a `--label` set on an earlier invocation must survive a later
    /// stage-only re-run that passes no `--label`. So a write whose sidecar
    /// carries `label: None` preserves a non-`None` label already on disk; an
    /// explicit label always overrides; a fresh segment with no prior label
    /// stays `None`.
    #[test]
    fn fit_sidecar_label_is_sticky_across_writes() {
        let tmp = crate::test_support::unique_temp_dir("sidecar_sticky_label");
        let seg = tmp.join("fits").join("demo-5091d4a8");
        let toml = std::path::Path::new("nonexistent.toml");
        let label = |seg: &std::path::Path| read_fit_sidecar(seg).unwrap().label;

        // Fresh segment, no prior label, no incoming label → stays `None`.
        write_fit_sidecar(&seg, toml,
            &FitSidecar { label: None, ..Default::default() }).unwrap();
        assert_eq!(label(&seg), None,
            "a fresh segment with no --label must stay unlabeled");

        // `fit run … --stage scout --label "smoke test scout"`.
        write_fit_sidecar(&seg, toml,
            &FitSidecar { label: Some("smoke test scout".into()), ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("smoke test scout".into()));

        // `fit run … --stage refine` (no --label): the earlier label must stick.
        write_fit_sidecar(&seg, toml,
            &FitSidecar { label: None, ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("smoke test scout".into()),
            "gh#29: a stage-only re-run without --label must not clobber the label");

        // A later invocation with an explicit --label overrides the sticky one.
        write_fit_sidecar(&seg, toml,
            &FitSidecar { label: Some("validate run".into()), ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("validate run".into()),
            "an explicit --label must override the sticky label");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// gh#353: the fit run leaf archives the model `.camdl` source as
    /// `model.camdl.original`, symmetric with `fit.toml.original`, so a consumer
    /// (e.g. camdl-watch's Source tab) can read it from the self-contained leaf
    /// instead of a checkout-relative path that doesn't resolve elsewhere.
    ///
    /// The recorded `model_path` is `config.model.camdl` — the *same* string the
    /// fit loads through `resolve_ir_path` (`std::fs::read(path)`), i.e. resolved
    /// against the **process CWD**, never the fit.toml's directory. A fit is
    /// normally launched from a directory that is not the fit.toml's parent (the
    /// config and model live in subdirs, run from the repo root), so this test
    /// puts the two bases in DIFFERENT places: the model lives in a subdir of
    /// CWD, while the fit.toml lives in a separate temp dir. Resolving the
    /// recorded path against the fit.toml's directory would look for a file that
    /// does not exist and silently skip the archive — so this fails red on the
    /// old (fit.toml-dir-relative) resolution and passes green once the archiver
    /// resolves the recorded path exactly as `resolve_ir_path` does.
    #[test]
    fn fit_sidecar_archives_model_camdl_source() {
        // A model source in a CWD-relative subdir — this mirrors how
        // `resolve_ir_path` found it from `config.model.camdl` (the recorded
        // `model_path`), which is resolved against the process CWD. Unique
        // subdir name so parallel test threads don't collide; removed before any
        // assert can panic so the checkout is never left dirty.
        let unique = crate::test_support::unique_temp_dir("sidecar_cwd_model");
        let sub = unique.file_name().unwrap().to_str().unwrap().to_string();
        let model_dir = std::env::current_dir().unwrap().join(&sub);
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_src = "compartments { S, I }\n"; // content is opaque to the archive
        std::fs::write(model_dir.join("m.camdl"), model_src).unwrap();
        let rel_model_path = format!("{sub}/m.camdl");

        // The fit.toml lives in a SEPARATE temp dir — its parent is neither CWD
        // nor the model's directory. This is the case the bug missed.
        let tmp = crate::test_support::unique_temp_dir("sidecar_fit_dir");
        std::fs::create_dir_all(&tmp).unwrap();
        let toml = tmp.join("fit.toml");
        std::fs::write(&toml, format!("[model]\ncamdl = \"{rel_model_path}\"\n")).unwrap();
        let seg = tmp.join("fits").join("demo-a1b2c3d4");

        let write_result = write_fit_sidecar(
            &seg,
            &toml,
            &FitSidecar { model_path: rel_model_path.clone(), ..Default::default() },
        );

        // A non-.camdl model has no source to archive — the leaf's model.ir.json
        // already captures it, so no stray file. Base-independent.
        let seg_ir = tmp.join("fits").join("demo-irjson0");
        let ir_write_result = write_fit_sidecar(
            &seg_ir,
            &toml,
            &FitSidecar { model_path: "m.ir.json".into(), ..Default::default() },
        );

        // Gather every fact, then clean up BOTH the CWD subdir and the temp dir
        // before asserting — a failed assert must never leave state behind.
        let archived = seg.join("model.camdl.original");
        let archived_is_file = archived.is_file();
        let archived_bytes = std::fs::read_to_string(&archived).ok();
        let ir_archive_absent = !seg_ir.join("model.camdl.original").exists();
        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&tmp).ok();

        write_result.unwrap();
        ir_write_result.unwrap();
        assert!(
            archived_is_file,
            "gh#353: the model source must be archived beside fit.toml.original \
             even when the fit runs from a directory that is not the fit.toml's \
             parent (model_path is CWD-relative, matching resolve_ir_path)"
        );
        assert_eq!(
            archived_bytes.as_deref(),
            Some(model_src),
            "the archived source must be a verbatim copy of the CWD-resolved model file"
        );
        assert!(
            ir_archive_absent,
            "a non-.camdl model_path must not produce a model.camdl.original"
        );
    }

    // ─── ObsSchema: the observation/dimension descriptor fold ──────────────

    use ir::observation::{
        ColumnRole, Likelihood, NegBinomialLikelihood, ObsColumn, ObservationModel,
        PoissonLikelihood, Projection, StratumKey,
    };
    use ir::parameter::ParamKind;

    fn const_expr() -> ir::expr::Expr {
        ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.0 })
    }

    /// Build one expanded observation leaf. `stratum` is `(dim, level)` pairs;
    /// `value_kind` is the role of the scored column (None ⇒ no `columns` block).
    fn leaf(
        name: &str,
        source: &str,
        scored: &str,
        stratum: &[(&str, &str)],
        value_kind: Option<ParamKind>,
        likelihood: Likelihood,
    ) -> ObservationModel {
        let columns = match value_kind {
            Some(k) => vec![ObsColumn { name: scored.to_string(), role: ColumnRole::Value(k) }],
            None => vec![],
        };
        ObservationModel {
            name: name.to_string(),
            source: source.to_string(),
            columns,
            scored: scored.to_string(),
            emit_schedule: None,
            stratum: stratum.iter()
                .map(|(d, l)| StratumKey { dim: d.to_string(), level: l.to_string() })
                .collect(),
            projection: Projection::CumulativeFlow("inc".into()),
            projection_state_grad: Default::default(),
            likelihood,
        }
    }

    fn poisson() -> Likelihood {
        Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(const_expr()) })
    }
    fn neg_binomial() -> Likelihood {
        Likelihood::NegBinomial(NegBinomialLikelihood {
            mean: ir::Diffable::new(const_expr()),
            dispersion: ir::Diffable::new(const_expr()),
        })
    }

    #[test]
    fn schema_unstratified_single_stream() {
        let obs = vec![leaf("cases", "cases", "cases", &[], Some(ParamKind::Count), poisson())];
        let s = ObsSchema::from_observations(&obs);
        assert!(s.dimensions.is_empty(), "no strata ⇒ no dimensions");
        assert_eq!(s.streams.len(), 1);
        let st = &s.streams[0];
        assert_eq!(st.name, "cases");
        assert!(st.index_dims.is_empty(), "a national series has no index dims");
        assert_eq!(st.value_column, "cases");
        assert_eq!(st.value_kind.as_deref(), Some("count"));
        assert_eq!(st.likelihood, "poisson");
    }

    #[test]
    fn schema_groups_stratified_leaves_into_one_logical_stream() {
        // The expander emits one leaf per stratum (`cases_Bo`, `cases_Bombali`)
        // but every leaf shares `source = "cases"`. The descriptor must fold
        // them back into ONE stream with `index_dims = [patch]`, never one
        // descriptor per leaf.
        let obs = vec![
            leaf("cases_Bo",      "cases", "cases", &[("patch", "Bo")],      Some(ParamKind::Count), neg_binomial()),
            leaf("cases_Bombali", "cases", "cases", &[("patch", "Bombali")], Some(ParamKind::Count), neg_binomial()),
        ];
        let s = ObsSchema::from_observations(&obs);
        assert_eq!(s.streams.len(), 1, "stratified leaves collapse to one logical stream");
        let st = &s.streams[0];
        assert_eq!(st.name, "cases");
        assert_eq!(st.index_dims, vec!["patch".to_string()]);
        assert_eq!(st.likelihood, "neg_binomial");
        // levels are the union over leaves, in first-appearance order.
        let patch = s.dimensions.get("patch").expect("patch dimension present");
        assert_eq!(patch.levels, vec!["Bo".to_string(), "Bombali".to_string()]);
    }

    #[test]
    fn schema_two_dim_stream_and_multiple_streams() {
        // `deaths[patch, age]` plus a separate `onset[patch]` stream. Two
        // logical streams; `deaths` carries both index dims; dimensions union
        // levels across BOTH streams (patch appears in each).
        let obs = vec![
            leaf("onset_Bo",        "onset",  "onset",  &[("patch", "Bo")],                Some(ParamKind::Count), poisson()),
            leaf("onset_Kailahun",  "onset",  "onset",  &[("patch", "Kailahun")],          Some(ParamKind::Count), poisson()),
            leaf("deaths_Bo_old",   "deaths", "deaths", &[("patch", "Bo"), ("age", "old")], Some(ParamKind::Count), neg_binomial()),
            leaf("deaths_Bo_young", "deaths", "deaths", &[("patch", "Bo"), ("age", "young")], Some(ParamKind::Count), neg_binomial()),
        ];
        let s = ObsSchema::from_observations(&obs);
        // Declaration order preserved: onset before deaths.
        let names: Vec<&str> = s.streams.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["onset", "deaths"]);
        let deaths = s.streams.iter().find(|d| d.name == "deaths").unwrap();
        assert_eq!(deaths.index_dims, vec!["patch".to_string(), "age".to_string()]);
        let onset = s.streams.iter().find(|d| d.name == "onset").unwrap();
        assert_eq!(onset.index_dims, vec!["patch".to_string()]);
        // patch levels union Bo (onset+deaths) and Kailahun (onset only).
        assert_eq!(
            s.dimensions.get("patch").unwrap().levels,
            vec!["Bo".to_string(), "Kailahun".to_string()]
        );
        assert_eq!(s.dimensions.get("age").unwrap().levels, vec!["old".to_string(), "young".to_string()]);
    }

    #[test]
    fn schema_stream_set_equals_distinct_sources() {
        // Agreement pin: the descriptor's logical streams are EXACTLY the
        // distinct `source` keys the filter binds — not the expanded leaf
        // names. A refactor that grouped by `name` would split `cases` into
        // `cases_Bo`/`cases_Bombali` and fail here.
        let obs = vec![
            leaf("cases_Bo",      "cases",  "cases",  &[("patch", "Bo")],      Some(ParamKind::Count), poisson()),
            leaf("cases_Bombali", "cases",  "cases",  &[("patch", "Bombali")], Some(ParamKind::Count), poisson()),
            leaf("hosp",          "hosp",   "hosp",   &[],                     Some(ParamKind::Count), poisson()),
        ];
        let s = ObsSchema::from_observations(&obs);
        let mut got: Vec<String> = s.streams.iter().map(|d| d.name.clone()).collect();
        got.sort();
        let want: Vec<String> = obs.iter()
            .map(|o| o.source.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn schema_value_kind_absent_without_columns_block() {
        // A model that predates the explicit `columns {}` block has no declared
        // role for the scored column ⇒ value_kind is None (omitted in JSON),
        // not a fabricated default.
        let obs = vec![leaf("cases", "cases", "cases", &[], None, poisson())];
        let s = ObsSchema::from_observations(&obs);
        assert_eq!(s.streams[0].value_kind, None);
        // and it must round-trip with the field omitted entirely.
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("value_kind"), "absent value_kind is omitted, got: {json}");
    }

    #[test]
    fn schema_roundtrips_through_serde() {
        let obs = vec![
            leaf("cases_Bo",      "cases", "cases", &[("patch", "Bo")],      Some(ParamKind::Count), neg_binomial()),
            leaf("cases_Bombali", "cases", "cases", &[("patch", "Bombali")], Some(ParamKind::Count), neg_binomial()),
        ];
        let s = ObsSchema::from_observations(&obs);
        let json = serde_json::to_string_pretty(&s).unwrap();
        let back: ObsSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back, "ObsSchema must survive a JSON round trip unchanged");
    }
}
