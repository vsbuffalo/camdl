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
use std::collections::HashMap;

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
            FitAlgorithm::Pfilter  => "pfilter",
            FitAlgorithm::NlSbplx  => "nl-sbplx",
            FitAlgorithm::NlBobyqa => "nl-bobyqa",
        }
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

/// The fit-level provenance sidecar (`fits/{stem}-{h8}/fit.meta.json`). A CAS
/// fit's fit level is a path segment with no `RunRecord`; this sidecar is the
/// single authoritative home for the fit-wide attributes that are NOT carried
/// on the stage leaves — the user `--label` and the fit-wide provenance
/// (`resolved_priors` = gh#75 per-parameter prior source,
/// `estimated`/`fixed`/`data_hashes`, `model_hash`, paths).
///
/// It is **derived provenance, not a source of truth**: a faithful readable
/// projection of inputs already hashed into the leaf identity (the `FitDigest`
/// — different priors already produce a different fit identity). It is written
/// post-identity and is never fed back into any hash. The producing `fit.toml`
/// is archived beside it as `fit.toml.original` (the config-diff source for
/// `fit table`).
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

/// Read the fit-level sidecar; `None` when absent (an incomplete segment —
/// [`crate::fit::fit_view::FitView`] treats that as a malformed fit and skips
/// it loudly rather than fabricating empty provenance).
pub fn read_fit_sidecar(fit_segment: &std::path::Path) -> Option<FitSidecar> {
    let bytes = std::fs::read(fit_segment.join("fit.meta.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
