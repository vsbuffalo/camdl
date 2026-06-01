//! Resolved per-level digest types and leaf input shapes — the identity
//! contract, expressed as types.
//!
//! Every field here is a *resolved value* or *content digest*, never a raw
//! path, unresolved preset, or generating recipe. Provenance fields carry
//! `#[run_input(provenance)]` and are recorded in `run.json` but never
//! hashed. The leaf inputs compose the shared per-level digests
//! ([`ModelDigest`], [`SimConfig`], [`ResolvedParams`], [`ResolvedScenario`])
//! so grouping (seed ensembles, profile grids) falls out of the path tree.
//!
//! **M1 status.** These types *specify the shapes*; the resolver that
//! produces their values (compile-or-cache IR, param/scenario/config
//! resolution, sweep/draw/seed expansion) is M2/M3 work. Two consequences:
//!
//! - [`ModelDigest`] is the **M2 interim** whole-IR digest (over-invalidates
//!   an obs-only edit, never under-invalidates). M2.5 splits it into
//!   dynamics/observation/output to recover latent-trajectory reuse; until
//!   then the obs sub-artifact's model identity rides in the trajectory's
//!   whole-IR digest.
//! - [`CalendarMode`] is a minimal placeholder: the calendar-time work is
//!   in-flight (per CLAUDE.md), so its variant set is provisional and M2
//!   finalizes it against the resolver. It is kept here only so the
//!   [`SimConfig`] shape is concrete.

use std::collections::{BTreeMap, BTreeSet};

use runid_derive::RunInput;
use serde::{Deserialize, Serialize};

use crate::float::FiniteF64;
use crate::hash::{CanonicalHasher, ContentAddressed, ContentHash};
use crate::kind::ArtifactKind;

// ── Resolved leaf-level value types ──────────────────────────────────────────

/// Resolved simulation backend — the value form of `--backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, RunInput)]
pub enum Backend {
    Gillespie,
    TauLeap,
    ChainBinomial,
    Ode,
}

/// Resolved calendar mode. **Placeholder** (see the module note): the
/// calendar-time work is in-flight, so the variant set is provisional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, RunInput)]
pub enum CalendarMode {
    Numeric,
    Calendar,
}

/// A resolved output schedule: concrete cadence/times over [`FiniteF64`].
#[derive(Debug, Clone, PartialEq, RunInput)]
pub enum ResolvedOutputSchedule {
    Regular { start: FiniteF64, step: FiniteF64, end: FiniteF64 },
    AtTimes(Vec<FiniteF64>),
    MatchObservations,
}

/// A resolved parameter name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, RunInput)]
pub struct ParamId(pub String);

/// A resolved intervention name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, RunInput)]
pub struct InterventionId(pub String);

/// Content digest of an external file's *bytes* (a `--table`/`--param-vec`
/// CSV, observed data). Hashed by content, never by path — a regenerated or
/// edited file invalidates correctly.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct DataDigest(pub ContentHash);

/// The runtime engine version string (e.g. `"0.3.0+abc1234"`), folded into
/// the model level. Distinct from the *compiler* version that keys the
/// compile cache — a runtime-only engine change re-keys run identity without
/// recompiling.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct EngineVersion(pub String);

/// The resolved per-process seed driving a trajectory. The seed *level*
/// hashes `process_seed`, NOT the user `--seed`: the same base seed maps to
/// a different `process_seed` depending on grid shape and cell position, so
/// hashing the base seed would alias a lone run and a sweep-point (a silent
/// wrong answer the `run.json` gate cannot catch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, RunInput)]
pub struct Seed {
    /// `process_seed = mix_cell_seed(base, point_idx, rep)` — the actual
    /// trajectory driver, and the only hashed field.
    pub process_seed: u64,
    /// The user-facing base seed. Provenance: the readable `seed_{base}`
    /// path label, never folded into identity.
    #[run_input(provenance)]
    pub base_seed: u64,
}

// ── Per-level digests ────────────────────────────────────────────────────────

/// Resolved base parameters (the `parameters` level): canonical name→value
/// plus the content digests of any `--table`/`--param-vec` files merged in.
/// A draw row hashes its own resolved values here, never the design recipe.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct ResolvedParams {
    pub values: BTreeMap<ParamId, FiniteF64>,
    pub tables: Vec<DataDigest>,
}

/// Resolved scenario delta (the `scenario` level): sorted id-sets + a
/// canonical patch. The empty delta hashes to its **real** `scen_h8`;
/// `baseline` is the display label only, never a literal zero hash.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct ResolvedScenario {
    pub enabled: BTreeSet<InterventionId>,
    pub disabled: BTreeSet<InterventionId>,
    pub patch: BTreeMap<ParamId, FiniteF64>,
}

/// Resolved simulation config (the `config` level).
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct SimConfig {
    pub backend: Backend,
    pub dt: FiniteF64,
    pub t_start: FiniteF64,
    pub t_end: FiniteF64,
    pub output: ResolvedOutputSchedule,
    pub calendar: CalendarMode,
    /// Semantic: `--allow-degenerate-rates` changes collapse handling
    /// (hard-error → silent-zero), which changes trajectory values. A
    /// control-looking flag that is genuinely semantic.
    pub allow_degenerate_rates: bool,
}

/// The model-level digest. **M2 interim:** the whole canonical IR. M2.5
/// splits this into `ModelDynamicsDigest` / `ObservationDigest` /
/// `OutputDigest` to recover latent-trajectory reuse; either way the model
/// level also folds in `ir_version` + `engine`.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct ModelDigest {
    /// `content_hash` of the whole canonical IR.
    pub ir: ContentHash,
    /// The IR schema version (e.g. `"0.7"`), matching `ir/VERSION`.
    pub ir_version: String,
    pub engine: EngineVersion,
}

impl ModelDigest {
    /// Build the M2-interim whole-IR digest from a compiled model.
    ///
    /// (M2 will first apply a normalization pass to the `Model` that strips
    /// pure-presentation fields — `output.format`, `simulation.time_semantics`
    /// — so `--format`/`--dates` stay inert. M1 hashes the model faithfully;
    /// the caller normalizes first.)
    pub fn from_model(model: &ir::Model, ir_version: String, engine: EngineVersion) -> Self {
        Self { ir: model.content_hash(), ir_version, engine }
    }
}

// ── Lineage ──────────────────────────────────────────────────────────────────

/// A reference to a *consumed* upstream artifact — the lineage edge. Hashed
/// as `run_id ++ artifact ++ digest` (the producing leaf's identity, *which*
/// file within it, and that file's content digest); `kind` is display-only.
/// Folding the digest pins which upstream file was consumed, so a change to
/// a sibling artifact under the same leaf does not invalidate the consumer,
/// and a regenerated/reselected upstream invalidates correctly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, RunInput)]
pub struct ArtifactRef {
    pub run_id: ContentHash,
    /// Display-only (the producing kind); not folded into identity.
    #[run_input(provenance)]
    pub kind: ArtifactKind,
    pub artifact: String,
    pub digest: ContentHash,
}

/// `deps` — the set of consumed upstreams, hashed as a set **sorted by
/// `run_id`** (not collection order), so reordering independent upstreams
/// changes no hash. The general `Vec` rule is order-sensitive; `deps` is the
/// documented exception, so it is hand-written rather than derived.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Deps(pub Vec<ArtifactRef>);

impl ContentAddressed for Deps {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_type_tag("runid::inputs::Deps");
        h.write_schema_version(1);
        let mut refs: Vec<&ArtifactRef> = self.0.iter().collect();
        refs.sort_by(|a, b| a.run_id.as_bytes().cmp(b.run_id.as_bytes()));
        h.write_len(refs.len() as u64);
        for r in refs {
            r.hash_into(h);
        }
    }
}

/// Input-side provenance carried on a leaf input and **always skipped** from
/// the hash (`#[run_input(provenance)]` at every embed site). The recorded
/// counterpart that lands in `run.json` is `record::Provenance`. Minimal in
/// M1; the resolver fills it in M2. Deliberately not `ContentAddressed` — it
/// must never be hashed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunProvenance {
    pub argv: Vec<String>,
    pub label: Option<String>,
}

// ── Leaf input shapes ────────────────────────────────────────────────────────
//
// Each leaf composes the shared per-level digests plus its command-specific
// fields; the provenance field is excluded from the hash. The forward-sim
// leaves (Trajectory, SyntheticObs) are wired in M2; the inference leaves
// (FitStage, PfilterEval, Survey, Projection) in M3.

/// `simulate`/`batch` leaf: the latent dynamics — NOT the obs model. (M2
/// interim: `model` is the whole-IR [`ModelDigest`]; M2.5 swaps in the
/// dynamics/output split so an obs-only edit does not re-key the trajectory.)
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct TrajectoryInput {
    pub model: ModelDigest,
    pub config: SimConfig,
    pub params: ResolvedParams,
    pub scenario: ResolvedScenario,
    pub seed: Seed,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

/// Synthetic-observation sub-artifact under a trajectory. M2 interim: the obs
/// *model* identity rides in the trajectory's whole-IR [`ModelDigest`]
/// (referenced via `trajectory`), so this leaf adds the requested streams +
/// the resolved obs seed. M2.5 adds the full `ObservationDigest` here.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct SyntheticObsInput {
    pub trajectory: ArtifactRef,
    pub streams: Vec<String>,
    pub obs_seed: Seed,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

/// Fit-wide digest (the `fit` level): model + data + the whole canonicalized
/// fit.toml + engine. Hashing the *whole* canonicalized document (not an
/// enumerated subset) is the include-by-default posture — dropping a field
/// like `ic_free`/`holdout`/fit-level `dt` would under-invalidate θ̂.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct FitDigest {
    pub model: ModelDigest,
    /// Content digest of each resolved observation stream.
    pub data: Vec<DataDigest>,
    /// Content digest of the whole canonicalized fit.toml document.
    pub fit_toml: ContentHash,
    pub engine: EngineVersion,
}

/// One stage's config (the `stage` level): the whole canonicalized `Stage`
/// config + the resolved obs-block + flow indices + `target_length` + the
/// upstream `deps`. Hashing the whole struct (not a subset) mirrors fit-wide:
/// enumerating a subset is the hash-a-recipe antipattern. M1 sketches the
/// load-bearing fields; M3 binds it to the real `Stage`.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct StageConfig {
    /// Content digest of the whole canonicalized stage config block.
    pub config: ContentHash,
    /// Resolved obs-block name (the `--obs` selection): selects which series
    /// drives the likelihood; not in the toml.
    pub obs_block: String,
    /// Resolved flow-index selection (the `--flow` selection).
    pub flow_indices: Vec<u32>,
    /// Resume target length — a resumed run is a distinct artifact.
    pub target_length: u64,
}

/// `fit` stage leaf (M3). The stage DAG recursion lives in `deps`:
/// `02-posterior` folds in `01-scout`'s *identity*, so the posterior
/// invalidates whenever the scout's inputs change. Sound only once the
/// engine is pure (the M3 fixes).
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct FitStageInput {
    pub fit: FitDigest,
    pub stage: StageConfig,
    pub deps: Deps,
    pub seed: Seed,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

/// The `stage` *level* hash input: a stage's own config **plus its `deps`**.
/// This is what the factored path's stage segment (`{NN}-{name}-{stage_h8}`)
/// and the leaf `run_id` hash for the stage level — so `02-posterior`'s
/// stage hash folds in `01-scout`'s identity (via `deps`), making the
/// posterior re-key whenever the scout it consumes changes, while editing
/// the posterior's own block leaves the scout leaf untouched (scout reuse).
/// Kept distinct from [`StageConfig`] so the bare config (without lineage)
/// stays addressable, and from [`FitStageInput`] which composes all levels.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct StageLevel {
    pub config: StageConfig,
    pub deps: Deps,
}

/// `pfilter` leaf (M3): scores a model at fixed params (does not estimate) —
/// its own kind, keeping "estimate params" and "score at fixed params"
/// unconflated.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct PfilterEvalInput {
    pub model: ModelDigest,
    /// Resolved observation streams consumed.
    pub data: Vec<DataDigest>,
    pub params: ResolvedParams,
    pub particles: u32,
    pub config: SimConfig,
    /// Resolved obs-block + flow-index selection (selects which series drives
    /// the likelihood).
    pub obs_block: String,
    pub flow_indices: Vec<u32>,
    pub seed: Seed,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

/// `survey` leaf (M3): a likelihood-landscape diagnostic over an LHS box.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct SurveyInput {
    pub model: ModelDigest,
    pub data: Vec<DataDigest>,
    /// LHS box: estimated param → (lo, hi), canonical.
    pub bounds: BTreeMap<ParamId, (FiniteF64, FiniteF64)>,
    pub fixed: ResolvedParams,
    pub scenario: ResolvedScenario,
    pub n_points: u32,
    pub seed: Seed,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

/// `lineage realize/tree/cohort/sojourn` projection leaf (M3): a 1–2-hop
/// chain off an upstream artifact. Each hop's upstream is one `ArtifactRef`;
/// every output-determining flag is folded into `spec`.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct ProjectionInput {
    /// The consumed upstream (event-log → line-list → projection).
    pub upstream: ArtifactRef,
    /// Content digest of the resolved, output-determining flag set
    /// (`--event`/`--align-zero`/window/scheme/seed, per subcommand).
    pub spec: ContentHash,
    #[run_input(provenance)]
    pub display: RunProvenance,
}

#[cfg(test)]
mod tests;
