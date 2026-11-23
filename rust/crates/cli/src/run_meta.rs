//! Shared run-metadata types of a fit run's files, read by `browse` and
//! `fit::fit_view`. This module keeps the small, cross-cutting value types
//! those readers and writers share:
//!
//! - [`FitAlgorithm`] / [`InferenceBackend`] — the (algorithm, backend) tags a
//!   fit method records and a reader projects.
//! - [`ResolvedPriorEntry`] / [`ParameterProvenance`] / [`InitProvenance`] —
//!   the gh#75 / gh#83/gh#85 provenance records embedded in a leaf's `inputs`
//!   and in the fit-level [`FitSidecar`].
//! - [`FitSidecar`] + [`write_fit_sidecar`] / [`read_fit_sidecar`] — the
//!   fit-level provenance sidecar (`fit.meta.json`) a CAS fit segment carries
//!   alongside its method leaves.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// The `schema` tag every `<algorithm>_summary.json` carries as its first key,
/// in the form the predictive manifests use (`camdl.predictive/v4`,
/// `camdl.predict-report/v2`).
///
/// It exists because two keys of this file changed *meaning* under unchanged
/// names (gh#728): `rhat` was the classic Gelman–Rubin statistic and is now
/// `max(rank-normalized split-R̂, folded split-R̂)`; `ess` was a sum of
/// per-chain Geyer estimates, suppressed to NaN above R̂ 1.1, and is now the
/// cross-chain bulk-ESS of Vehtari et al. (2021), never suppressed. Both
/// vintages sit in one store for as long as a project keeps its fits, and
/// nothing in an untagged file distinguishes them — a summary with no nulls is
/// either a new fit or an old fit that converged. A consumer that labels a
/// column "R̂" is therefore labelling two different estimators.
///
/// **Absence means the pre-tag vintage** — the classic statistics — and this
/// tag means the rank-normalized ones. A later change of meaning bumps the
/// version rather than reusing it.
///
/// v2 (gh#901): the key naming the algorithm is `method`, not `stage`. The
/// `[stages]` → `[method]` split left this file spelling a word the store
/// levels, the fit config and the CLI no longer use, and the file is
/// machine-read. A key that changes name under an unchanged tag is exactly
/// the vintage problem the tag exists to prevent, so the rename is a bump.
pub const FIT_SUMMARY_SCHEMA: &str = "camdl.fit-summary/v2";

/// Check a parsed `<algorithm>_summary.json` against [`FIT_SUMMARY_SCHEMA`].
///
/// The one place the check lives, so a reader cannot half-do it. `filename` is
/// only for the message.
pub fn check_fit_summary_schema(v: &serde_json::Value, filename: &str) -> Result<(), String> {
    match v.get("schema").and_then(|s| s.as_str()) {
        Some(FIT_SUMMARY_SCHEMA) => Ok(()),
        Some(other) => Err(format!(
            "{filename} declares `schema = \"{other}\"`; this camdl reads \
             `{FIT_SUMMARY_SCHEMA}`. The file was written by a different \
             version of camdl — re-run the fit, or read it with the version \
             that wrote it."
        )),
        None => Err(format!(
            "{filename} carries no `schema` tag, so it predates the tag \
             altogether (gh#728) and this camdl reads \
             `{FIT_SUMMARY_SCHEMA}`. Its `rhat` is the classic \
             Gelman–Rubin statistic and its `ess` a suppressed per-chain sum, \
             not the rank-normalized and bulk statistics this camdl reports \
             under those names — the numbers would be read as something they \
             are not. Re-run the fit to refresh it."
        )),
    }
}

/// Inference algorithm tag — discriminator enum naming the algorithm
/// independent of the simulation backend. Recorded in a fit-method leaf's
/// `inputs` alongside `Backend` to capture the (algorithm, backend) pair the
/// method ran. Wire format matches the lowercased / kebab-cased name
/// the user writes in fit.toml (`algorithm = "if2"`, `algorithm =
/// "nl-sbplx"`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FitAlgorithm {
    #[serde(rename = "if2")]      If2,
    #[serde(rename = "pgas")]     Pgas,
    #[serde(rename = "pmmh")]     Pmmh,
    #[serde(rename = "mh")]       Mh,
    #[serde(rename = "pfilter")]  Pfilter,
    #[serde(rename = "nl-sbplx")] NlSbplx,
    #[serde(rename = "nl-bobyqa")] NlBobyqa,
    #[serde(rename = "nl-lbfgs")] NlLbfgs,
}

impl FitAlgorithm {
    /// Wire-format string. Matches the `algorithm = "..."` value in fit.toml
    /// and the `inputs.method` serialized form on a fit-method leaf.
    pub fn as_str(self) -> &'static str {
        match self {
            FitAlgorithm::If2      => "if2",
            FitAlgorithm::Pgas     => "pgas",
            FitAlgorithm::Pmmh     => "pmmh",
            FitAlgorithm::Mh       => "mh",
            FitAlgorithm::Pfilter  => "pfilter",
            FitAlgorithm::NlSbplx  => "nl-sbplx",
            FitAlgorithm::NlBobyqa => "nl-bobyqa",
            FitAlgorithm::NlLbfgs  => "nl-lbfgs",
        }
    }

    /// The per-method convergence-summary filename a Bayesian sampler writes into
    /// its method dir: `<algorithm>_summary.json`. The single source of this
    /// naming convention — written by the pgas / pmmh / mh runners, read
    /// back by `read_convergence` and the `MethodResult` loaders. Deterministic
    /// mh-ODE shares the PMMH runner but gets its OWN file (`mh_summary.json`),
    /// not the misleading `pmmh_summary.json`.
    pub fn summary_filename(self) -> String {
        format!("{}_summary.json", self.as_str())
    }

    /// A Bayesian posterior sampler: it explores the full posterior and writes a
    /// `draws.tsv` cloud, so a completed run has a posterior band to draw. PGAS
    /// (the default Bayesian path), PMMH, and MH are samplers. This is the
    /// authoritative sampler/optimizer partition used to frame `camdl predict`'s
    /// refusals — a sampler with no resolvable draws is *incomplete*, never an
    /// "optimizer fit" (gh#343).
    pub fn is_posterior_sampler(self) -> bool {
        matches!(
            self,
            FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Mh
        )
    }

    /// An optimizer: it returns a single best-fit point (no posterior cloud), so
    /// a predictive can only ever plug those parameters in — there is no band.
    /// IF2 and the NLopt family (sbplx / bobyqa / lbfgs) are optimizers.
    /// `Pfilter` is a
    /// likelihood evaluation, not a fit, so it is *neither* a sampler nor an
    /// optimizer.
    pub fn is_optimizer(self) -> bool {
        matches!(
            self,
            FitAlgorithm::If2
                | FitAlgorithm::NlSbplx
                | FitAlgorithm::NlBobyqa
                | FitAlgorithm::NlLbfgs
        )
    }
}

impl std::fmt::Display for FitAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Simulation backend the method ran on. The (algorithm, backend) pair
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
/// backend but has no fit/inference interface, so it cannot back a fit method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDomainError {
    NotInferenceBackend,
}

impl std::fmt::Display for BackendDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendDomainError::NotInferenceBackend => f.write_str(
                "gillespie is a forward-simulation backend, not an inference backend; \
                 fit methods support only chain_binomial or ode",
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
/// dynamics are a forward model). Total — used to record the method's actual
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
    /// The model's `~` prior (canonical `family(arg=value, ...)` form) when
    /// the fit toml's prior displaced it (gh#369), so an override is visible
    /// after the fact and not only in the run's stderr. Absent when the model
    /// declares no prior or the fit toml supplies none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overridden_model_prior: Option<String>,
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
/// audit fields for the kick-from-estimate case and for every
/// replacement of one tier's value by another's.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterProvenance {
    /// Resolved value as written into `model.parameters[i].value`.
    pub value:  f64,
    /// [`crate::params_resolver::ValueSource::tag`] string
    /// (`"model_default" | "starts_from_params" | "starts_from_mle" |
    /// "fit_toml_fixed" | "fixed_file" | "sweep_point" | "scenario" |
    /// "fixed_cli" | "estimate_start" | "estimate_start_drawn"`).
    pub source: String,
    /// `"fixed" | "estimated"` — matches
    /// [`crate::params_resolver::ParameterRole`] — or, on a profile, `"focal"`
    /// for a swept parameter (pinned per cell at its grid value).
    pub role:   String,
    /// Present iff the parameter was kicked from `[estimate]` by a
    /// user-explicit `--fixed{,-file}` assertion. The `by` field
    /// records the value source that triggered the kick (e.g.
    /// `"fixed_cli"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kicked_from_estimate: Option<KickedFromEstimate>,
    /// Every replacement of this parameter's value by a later tier, in
    /// tier order ([`crate::params_resolver::Override`]). Absent when no
    /// tier displaced another's value.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<OverrideRecord>,
}

/// Audit record for [`crate::params_resolver::FixReason::KickedFromEstimate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KickedFromEstimate {
    /// The ValueSource tag of the source that kicked the parameter
    /// out (`"fixed_cli" | "fixed_file"`).
    pub by: String,
}

/// Audit record for one [`crate::params_resolver::Override`]: the value a
/// lower tier supplied and the value that replaced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverrideRecord {
    pub replaced: SourcedValue,
    pub by:       SourcedValue,
}

/// A value and the tier that supplied it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedValue {
    /// [`crate::params_resolver::ValueSource::tag`].
    pub source: String,
    pub value:  f64,
    /// The scenario's name, for a `scenario` source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    /// The file, for a `fixed_file` source (`--params` / `--fixed-file`) or a
    /// `starts_from_params` source (`--starts from_params=FILE`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The fit handle, for a `starts_from_mle` source
    /// (`--starts from_mle=FIT`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<String>,
}

impl SourcedValue {
    fn from_resolved(src: &crate::params_resolver::ValueSource, value: f64) -> Self {
        use crate::params_resolver::{StartsOrigin, ValueSource};
        SourcedValue {
            source: src.tag().to_string(),
            value,
            scenario: match src {
                ValueSource::Scenario(n) => Some(n.clone()),
                _ => None,
            },
            path: match src {
                ValueSource::FixedFile { path }
                | ValueSource::Starts(StartsOrigin::Params { path }) =>
                    Some(path.display().to_string()),
                _ => None,
            },
            fit: match src {
                ValueSource::Starts(StartsOrigin::Mle { fit }) => Some(fit.clone()),
                _ => None,
            },
        }
    }
}

/// Per-chain init provenance. The `method` field echoes the
/// [`crate::fit::starts::ChainStarts`] tag; each entry of `chains` is a
/// map from estimated-parameter name to its per-chain start value +
/// source. Restricted to the estimate set by construction (see
/// `ChainStart.values`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitProvenance {
    /// The starts rule's tag ([`crate::fit::starts::ChainStarts::tag`]) — one
    /// spelling per rule across every artifact.
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
    /// Every resolved parameter's provenance, by name: the
    /// `parameters_provenance` block each verb records beside its run
    /// (`run.json`'s inputs, or `fit.meta.json`). Display, never hashed.
    pub fn of_all(
        params: &[crate::params_resolver::ResolvedParameter],
    ) -> BTreeMap<String, ParameterProvenance> {
        params.iter().map(|rp| (rp.name.clone(), ParameterProvenance::from_resolved(rp))).collect()
    }

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
        let overrides = rp.overrides.iter().map(|o| OverrideRecord {
            replaced: SourcedValue::from_resolved(&o.replaced.0, o.replaced.1),
            by:       SourcedValue::from_resolved(&o.by.0, o.by.1),
        }).collect();
        ParameterProvenance {
            value:  rp.value,
            source: rp.source.tag().to_string(),
            role,
            kicked_from_estimate,
            overrides,
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
    pub fn from_chain_starts(cs: &crate::fit::chain_starts::DrawnStarts) -> Self {
        // Allocate `chains` sized to (max chain_id + 1) so an
        // out-of-order Vec<ChainStart> still produces a well-formed
        // index-by-chain_id output. Empty starts yield an empty Vec.
        let n_chains = cs.starts.iter()
            .map(|c| c.chain_id + 1).max().unwrap_or(0); // a count, not a label
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
            method: cs.rule.tag().to_string(),
            chains,
        }
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
/// on the method leaves — the user `--label` and the fit-wide provenance
/// (`resolved_priors` = gh#75 per-parameter prior source,
/// `estimated`/`fixed`/`data_hashes`, `model_identity`, paths).
///
/// Which rows the fit trained on (`docs/dev/temporal-architecture.md` §3.2,
/// D38): per stream the number of leading rows that trained and their row
/// digest, the training extent, and the rule that decided it. The record of
/// a `fit::training_partition::TrainingPartition`, value type for
/// [`FitSidecar::training_partition`]; produced by
/// `fit::runner::apply_holdout_declaration` for every fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingPartitionRecord {
    /// `fit::training_partition::PARTITION_RULE_VERSION` when recorded.
    pub rule_version: u16,
    pub rule: PartitionRuleRecord,
    /// The latest end among the training rows of every stream.
    pub extent: TimePoint,
    /// Sorted by stream name.
    pub streams: Vec<RowPrefixRecord>,
}

/// How a partition was decided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PartitionRuleRecord {
    /// No holdout declared: every row trained.
    All,
    /// `holdout_after`: a row trained if its end is at or before `at`.
    Cutoff { at: TimePoint },
    /// `[data.holdout]`: every row of the training files trained.
    SeparateFiles,
}

/// A model time, and the date it falls on when the model has an `origin`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimePoint {
    pub t: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
}

/// One stream's training rows: rows `[0, rows)`, and their row digest (hex).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowPrefixRecord {
    pub stream: String,
    pub rows: usize,
    pub digest: String,
}

impl TrainingPartitionRecord {
    /// The recorded prefix of `stream`, if the fit bound it.
    pub fn prefix(&self, stream: &str) -> Option<&RowPrefixRecord> {
        self.streams.iter().find(|p| p.stream == stream)
    }
}

/// The `schema_version` [`write_fit_sidecar`] stamps on `fit.meta.json`, and
/// the only one [`read_fit_sidecar`] accepts. A record without the field was
/// written before it existed and is version 1.
///
/// v2: the producing config's location is `source_config`, relative to the fit
/// folder that holds the sidecar, replacing v1's `fit_toml_path`, which was
/// relative to whatever directory `fit run` was started from and so resolved
/// wrongly from anywhere else (gh#1009).
///
/// v3: `calendar` is required (D59); a v2 record could omit it.
pub const FIT_META_SCHEMA_VERSION: u32 = 3;

/// The fit-level provenance sidecar (`fits/{stem}-{h8}/fit.meta.json`). A CAS
/// fit's fit level is a path segment with no `RunRecord`; this sidecar is the
/// single authoritative home for the fit-wide attributes that are NOT carried
/// on the method leaves — the user `--label` and the fit-wide provenance
/// (`resolved_priors` = gh#75 per-parameter prior source,
/// `estimated`/`fixed`/`data_hashes`, `model_identity`, paths).
///
/// It is **derived provenance, not a source of truth**: a faithful readable
/// projection of inputs the run's identity already holds (different priors
/// already pose a different problem). It is written
/// post-identity and is never fed back into any hash. The producing `fit.toml`
/// is archived beside it byte for byte as `fit.toml.original`, and the
/// configuration the folder's fits ran under as `fit.resolved.toml`: the
/// config-diff source for `fit table` and the config a run handle recovers.
///
/// Every field except `calendar` and `resolved_priors`-class provenance
/// defaults, so partial sidecars round-trip; [`crate::fit::fit_view::FitView`]
/// enforces that a Bayesian fit's `resolved_priors` is present (no silent
/// default). `calendar` is required: no sidecar is written without the
/// model's axis, so there is no production `Default`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitSidecar {
    /// [`FIT_META_SCHEMA_VERSION`], stamped by [`write_fit_sidecar`] whatever
    /// the caller set; [`read_fit_sidecar`] refuses any other.
    #[serde(default)]
    pub schema_version: u32,
    /// The run store's format ([`store::STORE_FORMAT`]) the run was written
    /// under. Absent, and so 0, in a record written before formats were
    /// numbered. Nothing reads it.
    #[serde(default)]
    pub store_layout: u32,
    /// A profile's label. A fit segment's sidecar carries none: a fit's label
    /// lives on its runs' `run.json` (`provenance.label`, gh#896).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The model file the fit ran on (`[model] camdl`, a `.camdl` or
    /// `.ir.json`), as a path **relative to the fit folder holding this
    /// sidecar**, by the same rule and computation as [`Self::source_config`];
    /// set by [`write_fit_sidecar`]. The archived copies are this folder's
    /// `model.camdl.original` (a `.camdl` source) and `model.ir.json`. `None`
    /// when the model was not a file at write time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_model: Option<String>,
    #[serde(default)]
    pub model_identity: String,
    /// The producing config file (the user's `fit.toml`, which may since have
    /// moved or changed), as a path **relative to the fit folder holding this
    /// sidecar** (`fits/{stem}-{h8}/`), e.g. `../../../fits/sir.toml`; set by
    /// [`write_fit_sidecar`]. Not the archived copy: that is
    /// `fit.toml.original` in the same folder, which carries the config's text
    /// but not its location, as does the resolved config, `fit.resolved.toml`.
    ///
    /// Load-bearing: the stored configs' relative `[model]` / `[data]` paths were
    /// written against the source config's directory, so recovering the config
    /// from the segment anchors them at `<fit folder>/<source_config>` (gh#652,
    /// [`crate::fit::config_v2::FitConfigV2::load_anchored_at`]) — from any
    /// working directory, and after the project directory moves as a whole
    /// (gh#1009). `None` for a CLI-only fit, which has no config to archive and
    /// no directory to anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_config: Option<String>,
    /// SHA-256 of the producing config's raw bytes, as read — the bytes of
    /// `fit.toml.original`, not of `fit.resolved.toml`. **Provenance**: which
    /// exact bytes produced this fit, down to the whitespace. Nothing looks a
    /// fit up by it.
    ///
    /// The neighbouring question — does a config in hand MEAN what this fit was
    /// run from? — is a different fact and has a different answer: the canonical
    /// hash of the parsed value tree ([`crate::fit::cas::config_identity_hash`]),
    /// which each run records in `inputs.config_identities` and a `fit.toml`
    /// handle is matched against.
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
    /// Observation source → full SHA-256 (64 hex characters) of the training
    /// file bound to it, in every `[data]` form: the digests the fit's run
    /// identity folds (`StreamBindings::digests`). A sidecar written before
    /// these were full digests holds the first 8 characters, and one written
    /// before they were keyed by source keys a one-file-shorthand fit by leaf.
    #[serde(default)]
    pub data_hashes: BTreeMap<String, String>,
    /// Observation source → full SHA-256 of the explicit `[data.holdout]` file
    /// bound to it. Empty, and omitted, without one; also absent from sidecars
    /// written before holdout files were recorded.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub holdout_hashes: BTreeMap<String, String>,
    /// External input name → full SHA-256 of the file `[data.tables]` bound
    /// to it; the problem's identity holds the bytes
    /// (`FitProblemIdentity::tables`). The
    /// bytes themselves are stored beside this record, in
    /// [`STORED_INPUTS_DIR`]. Empty, and omitted, for a model with no external
    /// input.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub table_hashes: BTreeMap<String, String>,
    /// Generated from `partition` by
    /// [`crate::roles::PartitionRecord::legacy_fields`] and written no other
    /// way.
    #[serde(default)]
    pub estimated: Vec<String>,
    #[serde(default)]
    pub fixed: BTreeMap<String, f64>,
    /// The run's roles: which parameters were fixed, estimated or swept, and
    /// what declared each. The authoritative record; absent from sidecars
    /// written before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<crate::roles::PartitionField>,
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
    /// Which rows the fit trained on ([`TrainingPartitionRecord`]), recorded
    /// for every fit by the engine that applied it at load. Under a holdout
    /// its *presence* is the positive proof `camdl compare`'s non-leakage
    /// check reads (§3.7.3(b) of the 2026-08-29 honest-predictive-evaluation
    /// proposal): a fit recorded without it is not scored held out. Sticky
    /// like `label`: recorded after the data load, and a rerun that skips the
    /// load (all-cache-hit) must not erase it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_partition: Option<TrainingPartitionRecord>,
    /// The integration step every run in this segment ran at: the one dt
    /// rule's answer (`crate::dt_rule`) over the model's declared
    /// `simulate { dt }` and the config's `[config] dt`. A replay of the fit's
    /// parameters runs at this step by default
    /// ([`crate::fit::fitted_model::stored_fit_simulator`]). `None` in a
    /// sidecar written without a model in hand or before the step was
    /// recorded; the reader then recovers it from the archived config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt: Option<f64>,
    /// The ODE integrator the fit's model declares (`simulate { integrator }`,
    /// rk4 when it declares none): what an ODE-backend run in this segment
    /// integrated with. A fit has no integrator override, so this is the
    /// archived model's own; recorded so the fit states it without its IR
    /// being read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrator: Option<ir::model::Integrator>,
    /// The model's calendar axis (`docs/dev/temporal-architecture.md` §5.3):
    /// its time unit and the date of `t = 0`, so the fit's model times — the
    /// estimates' instants, the training window — can be dated without the
    /// model source. `origin` is null for a model that has none. Required
    /// (D59).
    pub calendar: ir::caltime::CalendarAxis,
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

/// Test fixtures build partial sidecars; the axis they carry is a numeric
/// day axis, which no production path can reach through this impl.
#[cfg(test)]
impl Default for FitSidecar {
    fn default() -> Self {
        FitSidecar {
            schema_version: 0,
            store_layout: 0,
            table_hashes: BTreeMap::new(),
            label: None,
            source_model: None,
            model_identity: String::new(),
            source_config: None,
            fit_toml_hash: String::new(),
            data_hashes: BTreeMap::new(),
            holdout_hashes: BTreeMap::new(),
            estimated: Vec::new(),
            fixed: BTreeMap::new(),
            partition: None,
            resolved_priors: Vec::new(),
            parameters_provenance: BTreeMap::new(),
            schema: None,
            training_partition: None,
            dt: None,
            integrator: None,
            calendar: ir::caltime::CalendarAxis::parse("days", None).expect("a known unit"),
            docs: ir::ModelDocs::default(),
        }
    }
}

/// Write `sidecar` as the folder's `fit.meta.json`, stamped with
/// [`FIT_META_SCHEMA_VERSION`], keeping the label and the training partition
/// an on-disk record of the current version carries when `sidecar` has none.
/// Archives nothing and records `source_config` as given: the call for
/// rewriting a sidecar that was read back.
#[cfg(test)]
pub fn store_fit_sidecar(fit_segment: &std::path::Path, sidecar: &FitSidecar) -> std::io::Result<()> {
    // gh#29: keep a profile's label sticky. gh#585: the training partition is
    // sticky the same way — it is recorded after the data load, and a rerun
    // that never loads (all-cache-hit short-circuit) writes the sidecar with
    // `None`, which must not erase the recorded proof. A record of an older
    // schema carries nothing forward: it is replaced whole.
    let on_disk = if sidecar.label.is_none() || sidecar.training_partition.is_none() {
        read_fit_sidecar(fit_segment).ok().flatten()
    } else {
        None
    };
    let mut merged = sidecar.clone();
    merged.schema_version = FIT_META_SCHEMA_VERSION;
    merged.label = sidecar.label.clone()
        .or_else(|| on_disk.as_ref().and_then(|s| s.label.clone()));
    merged.training_partition = sidecar.training_partition.clone()
        .or_else(|| on_disk.and_then(|s| s.training_partition));
    let bytes = serde_json::to_vec_pretty(&merged).map_err(std::io::Error::other)?;
    // Whole or not at all, through a temporary of this writer's own: a
    // reader never sees a torn record.
    store::write_file_atomic(&fit_segment.join("fit.meta.json"), &bytes).map_err(std::io::Error::other)
}

/// Read the fit-level sidecar; `Ok(None)` when absent (an incomplete segment —
/// [`crate::fit::fit_view::FitView`] treats that as a malformed fit and skips
/// it loudly rather than fabricating empty provenance). `Err` names the file
/// when it cannot be parsed, or when its `schema_version` is not
/// [`FIT_META_SCHEMA_VERSION`]: an older record is refused, never read with
/// its fields reinterpreted.
#[expect(clippy::disallowed_methods, reason = "reads `fit.meta.json` from the fit folder its caller passes: a scratch copy of a stored run read out through the store (`compare`, `fitted_model`, `predict`, `fit_view`, `training_partition`), or the `--fit` directory of `posterior_draws`, whose `[fixed]` values are hashed as parameter values")]
pub fn read_fit_sidecar(fit_segment: &std::path::Path) -> Result<Option<FitSidecar>, String> {
    let path = fit_segment.join("fit.meta.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let version = value.get("schema_version").and_then(|v| v.as_u64()).unwrap_or(1);
    if version != u64::from(FIT_META_SCHEMA_VERSION) {
        return Err(format!(
            "{} is fit.meta.json schema version {version}; this camdl reads version \
             {FIT_META_SCHEMA_VERSION} only (version 3 always records the model's calendar \
             axis; version 2 records the source config relative to the fit folder, where \
             version 1 recorded it relative to the directory the fit was run from). Re-run \
             the fit with this camdl.",
            path.display()));
    }
    serde_json::from_value(value).map(Some).map_err(|e| format!("{}: {e}", path.display()))
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

    /// gh#728: an untagged summary is refused by name, and the refusal says
    /// what the untagged file's numbers actually are.
    ///
    /// The keys `rhat` and `ess` changed estimator without changing name, so
    /// the absence of a tag is the only thing that distinguishes the two
    /// vintages — and it is one-directional in the wrong direction: a file
    /// with no nulls is either a new fit or an old fit that converged.
    #[test]
    fn a_summary_without_a_schema_tag_is_refused_by_name() {
        let tagged = serde_json::json!({
            "schema": FIT_SUMMARY_SCHEMA, "rhat": {"beta": 1.01},
        });
        assert!(check_fit_summary_schema(&tagged, "pgas_summary.json").is_ok());

        let untagged = serde_json::json!({ "rhat": {"beta": 1.01} });
        let why = check_fit_summary_schema(&untagged, "pgas_summary.json")
            .expect_err("an untagged summary is of unknown vintage");
        assert!(why.contains("pgas_summary.json"), "{why}");
        assert!(why.contains("no `schema` tag"), "{why}");
        assert!(why.contains("Gelman"), "the refusal says what the old numbers were: {why}");

        let other = serde_json::json!({ "schema": "camdl.fit-summary/v99" });
        let why = check_fit_summary_schema(&other, "mh_summary.json")
            .expect_err("a tag this camdl does not read is refused too");
        assert!(why.contains("v99") && why.contains(FIT_SUMMARY_SCHEMA), "{why}");
    }

    /// A sidecar write stages its bytes through a temporary file of its own
    /// and renames it into place, so a successful write, first or repeated,
    /// leaves the sidecar and no temporary beside it. A temporary a writer
    /// that died mid-write left behind is not this test's subject.
    #[test]
    fn a_sidecar_write_leaves_no_temporary() {
        let dir = crate::test_support::unique_temp_dir("sidecar_tmp");
        std::fs::create_dir_all(&dir).unwrap();
        store_fit_sidecar(&dir, &FitSidecar::default()).unwrap();
        store_fit_sidecar(&dir, &FitSidecar::default()).unwrap();
        assert!(read_fit_sidecar(&dir).unwrap().is_some());
        let names: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["fit.meta.json".to_string()], "the sidecar and nothing else");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn summary_filename_matches_algorithm_name() {
        // The naming seam: each sampler's summary is `<algorithm>_summary.json`.
        // In particular mh gets its OWN file, never the pmmh one.
        assert_eq!(FitAlgorithm::Pgas.summary_filename(), "pgas_summary.json");
        assert_eq!(FitAlgorithm::Pmmh.summary_filename(), "pmmh_summary.json");
        assert_eq!(FitAlgorithm::Mh.summary_filename(), "mh_summary.json");
        // mh and pmmh must not collide on disk.
        assert_ne!(
            FitAlgorithm::Mh.summary_filename(),
            FitAlgorithm::Pmmh.summary_filename()
        );
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
    /// stored in a fit-method leaf's `inputs.method`, part of the factored
    /// fit-method identity. Pins serialize, deserialize, and `as_str` together.
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
    /// backend (used to record method provenance into `MleMetadata.backend`).
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
            FixReason, Override, ParameterRole, ResolvedParameter,
            ValueSource,
        };
        // Build one entry per `ValueSource` variant tag — exercises
        // every branch of `ValueSource::tag()` through the round-trip.
        let resolved_entries = [
            ResolvedParameter {
                name:  "beta".into(),
                value: 0.42,
                source: ValueSource::ModelDefault,
                role: ParameterRole::Estimated,
                overrides: Vec::new(),
            },
            ResolvedParameter {
                name:  "gamma".into(),
                value: 0.10,
                source: ValueSource::FitTomlFixed,
                role: ParameterRole::Fixed {
                    reason: FixReason::NotInEstimate,
                },
                overrides: Vec::new(),
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
                overrides: vec![Override {
                    name:     "rho".into(),
                    replaced: (ValueSource::Scenario("worst_case".into()), 0.30),
                    by:       (ValueSource::FixedCli, 0.50),
                }],
            },
            ResolvedParameter {
                name:  "mu".into(),
                value: 0.05,
                source: ValueSource::Scenario("worst_case".into()),
                role: ParameterRole::Fixed {
                    reason: FixReason::NotInEstimate,
                },
                overrides: Vec::new(),
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
                overrides: Vec::new(),
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
        // overrides present on rho only.
        let rho = &meta["rho"];
        assert!(rho.kicked_from_estimate.is_some());
        assert_eq!(rho.kicked_from_estimate.as_ref().unwrap().by, "fixed_cli");
        assert_eq!(rho.overrides, vec![OverrideRecord {
            replaced: SourcedValue {
                source: "scenario".into(), value: 0.30,
                scenario: Some("worst_case".into()), path: None, fit: None,
            },
            by: SourcedValue {
                source: "fixed_cli".into(), value: 0.50, scenario: None, path: None, fit: None,
            },
        }]);
        assert!(meta["iota"].overrides.is_empty());
        let beta = &meta["beta"];
        assert_eq!(beta.role, "estimated");
        assert!(beta.kicked_from_estimate.is_none());
    }

    /// Audit checklist item 5: every starts rule has at least one
    /// round-trip producing a `run.json` whose `init_provenance.method`
    /// equals that rule's tag.
    #[test]
    fn init_provenance_method_tag_matches_for_every_variant() {
        use crate::fit::chain_starts::{ChainStart, DrawnStarts, InitSource};
        use crate::fit::starts::{ChainStarts, Handle, Point, Spread};
        // One DrawnStarts per (rule, expected tag) pair.
        let cases: Vec<(ChainStarts, &str)> = vec![
            (ChainStarts::Point(Point::Declared), "single"),
            (ChainStarts::Spread(Spread::Uniform), "uniform"),
            (ChainStarts::Spread(Spread::Lhs), "lhs"),
            (ChainStarts::Spread(Spread::UniformUnconstrained), "uniform_unconstrained"),
            (ChainStarts::Spread(Spread::FromPrior), "from_prior"),
            (ChainStarts::Spread(Spread::FromPosterior {
                source: Handle("draws.tsv".into()), chains: None
            }), "from_posterior"),
            (ChainStarts::Point(Point::FromMle {
                source: Handle("@mle".into()),
            }), "from_mle"),
            (ChainStarts::Point(Point::FromParams {
                path: "params.toml".into(),
            }), "from_params"),
        ];
        for (method, expected_tag) in &cases {
            // Single-chain DrawnStarts → InitProvenance → JSON.
            let cs = DrawnStarts::fresh(
                vec![ChainStart {
                    chain_id: 0,
                    values: HashMap::from([("beta".into(), 0.5_f64)]),
                    source: InitSource::SeededBase,
                }],
                method.clone(),
            );
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
        use crate::fit::chain_starts::{ChainStart, DrawnStarts, InitSource};
        use crate::fit::starts::ChainStarts;
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
        let cs = DrawnStarts::fresh(starts, ChainStarts::from_prior());
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

    /// gh#29: the fit-level sidecar `label` is sticky across repeated
    /// `fit run` invocations on the same fit segment. Every `fit run` into the
    /// segment (one per method file) rewrites the umbrella sidecar; a
    /// `--label` set on an earlier invocation must survive a later re-run
    /// that passes no `--label`. So a write whose sidecar
    /// carries `label: None` preserves a non-`None` label already on disk; an
    /// explicit label always overrides; a fresh segment with no prior label
    /// stays `None`.
    #[test]
    fn fit_sidecar_label_is_sticky_across_writes() {
        let tmp = crate::test_support::unique_temp_dir("sidecar_sticky_label");
        let seg = tmp.join("fits").join("demo-5091d4a8");
        std::fs::create_dir_all(&seg).unwrap();
        let label = |seg: &std::path::Path| read_fit_sidecar(seg).unwrap().unwrap().label;

        // Fresh segment, no prior label, no incoming label → stays `None`.
        store_fit_sidecar(&seg,
            &FitSidecar { label: None, ..Default::default() }).unwrap();
        assert_eq!(label(&seg), None,
            "a fresh segment with no --label must stay unlabeled");

        // `fit run mle.toml --label "smoke test scout"`.
        store_fit_sidecar(&seg,
            &FitSidecar { label: Some("smoke test scout".into()), ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("smoke test scout".into()));

        // `fit run posterior.toml` (no --label): the earlier label must stick.
        store_fit_sidecar(&seg,
            &FitSidecar { label: None, ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("smoke test scout".into()),
            "gh#29: a method-only re-run without --label must not clobber the label");

        // A later invocation with an explicit --label overrides the sticky one.
        store_fit_sidecar(&seg,
            &FitSidecar { label: Some("validate run".into()), ..Default::default() }).unwrap();
        assert_eq!(label(&seg), Some("validate run".into()),
            "an explicit --label must override the sticky label");

        std::fs::remove_dir_all(&tmp).ok();
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
            covers: None,
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
