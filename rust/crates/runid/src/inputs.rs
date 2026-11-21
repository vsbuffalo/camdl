//! Resolved per-level digest types — the identity contract, expressed as
//! types.
//!
//! Every field here is a *resolved value* or *content digest*, never a raw
//! path, unresolved preset, or generating recipe. Provenance fields carry
//! `#[run_input(provenance)]` and are recorded in `run.json` but never
//! hashed. The CLI's resolvers hash each level separately ([`ModelDigest`],
//! [`SimConfig`], [`ResolvedParams`], [`ResolvedScenario`], [`Seed`], …) and
//! compose the level hashes into a leaf's `run_id`, so grouping (seed
//! ensembles, profile grids) falls out of the path tree.
//!
//! - [`ModelDigest`] is a whole-IR digest: an observation-only edit re-keys
//!   the trajectory too (it over-invalidates, never under-invalidates). Its
//!   `ir` component is [`model_ir_hash`], the model hash every identity path
//!   shares.
//! - [`CalendarMode`]: every run resolves to `Numeric`; the variant set is
//!   provisional.

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
    ChainBinomial,
    Ode,
}

/// Resolved calendar mode. Every run resolves to `Numeric`; the variant set is
/// provisional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, RunInput)]
pub enum CalendarMode {
    Numeric,
    Calendar,
}

/// A resolved output schedule: concrete cadence/times over [`FiniteF64`].
#[derive(Debug, Clone, PartialEq, RunInput)]
pub enum ResolvedOutputSchedule {
    Regular { start: FiniteF64, step: FiniteF64 },
    AtTimes(Vec<FiniteF64>),
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

/// The forecast origin state a run was seeded from (`simulate --init-state`,
/// gh#641 / gh#697), when it did not build its initial state from the model's
/// `init {}` block.
///
/// Both fields are identity, for different reasons:
///
/// - `ensemble` is the content digest of the **ensemble of origin states** this
///   cell drew its row from — content, never a path or a run reference. A
///   re-filtered / re-fitted ensemble under an unchanged model must re-key, or
///   the store serves yesterday's forecast for today's data. There are two
///   producers and both hash content:
///   - `--init-state FILE` (one θ's particle swarm): SHA-256 of the state
///     file's bytes.
///   - `--init-state fit` (the paired `(θ_i, X_i(T))` posterior): the
///     [`InitStateEnsemble`] digest over the *resolved* ensemble actually used.
///     Deliberately **not** the fit's `run_id` + method + selection rule —
///     keying on provenance means enumerating every knob that changes which
///     draws are selected, and a missed knob is two different clouds colliding
///     on one cache entry. Keying on the ensemble's own content cannot miss
///     one, and two fits that genuinely produce the same states at the same
///     origin *should* share the entry.
/// - `row` is which row of that ensemble this cell restored. It is needed *in
///   addition* to the seed level because two cells can share a `process_seed`
///   (`--seeds 7,7`) while starting from different states; without it those two
///   cells would collide on one `run_id` and the store would serve one
///   trajectory for both (the count-in-the-key rule).
///
/// The origin *time* is not here: it rides in [`SimConfig::t_start`], which the
/// run already keys on (and, for the fit source, in the ensemble digest too).
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct InitStateDigest {
    pub ensemble: DataDigest,
    pub row: u64,
}

/// One restored origin state in a fit-sourced `--init-state` ensemble: the
/// posterior draw it belongs to, and the compartment values at the origin.
///
/// The `(chain, draw)` key rides in the hash alongside the values because it is
/// what makes the pairing checkable: a cloud that restored the same *states*
/// against a different assignment of draws is a different run, and must not
/// share a leaf with the correct one.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct InitStateRow {
    pub chain: u64,
    pub draw: u64,
    /// Integer compartment counts, in the model's integer-compartment order.
    pub counts: Vec<i64>,
    /// Real compartment values, in the model's real-compartment order.
    pub reals: Vec<FiniteF64>,
}

/// The resolved paired `(θ_i, X_i(T))` origin ensemble a `--init-state fit` run
/// forecasts from: the origin time, and every selected draw's restored state in
/// selection order.
///
/// Its [`ContentAddressed::content_hash`] is what
/// [`InitStateDigest::ensemble`] carries for the fit source. Whole-ensemble +
/// row (rather than the row alone) mirrors the file source exactly: a change
/// anywhere in the ensemble — a different fit, a different method, a different
/// forkable subset, a different draw cap, a different order — re-keys every
/// cell. Conservative in the safe direction, and the same rule on both sources.
///
/// θ is **not** here: each draw's resolved parameter values are hashed at the
/// `params` level ([`ResolvedParams`]), so a cell's `run_id` already carries
/// both halves of its pair.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct InitStateEnsemble {
    pub origin_t: FiniteF64,
    pub rows: Vec<InitStateRow>,
}

/// One engine switch that was on for a run: an environment variable that
/// selects an alternative evaluation path (`CAMDL_EVAL_UNRESOLVED`) or turns
/// on an experimental feature (`CAMDL_EXPERIMENTAL_<NAME>`, which carries its
/// value). Which variables are switches is the CLI's environment table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSwitch {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// The engine that computes a run, folded into the model level (and the fit
/// level): the build that ran it and the engine switches that were on.
///
/// `build` is the engine's build fingerprint (the CLI's `ENGINE_ID`, 32 hex
/// characters), distinct from the *compiler* version that keys the compile
/// cache — a runtime-only engine change re-keys run identity without
/// recompiling. `switches` are in the order the environment table lists them,
/// empty for a run made with none.
///
/// **Hashed bytes.** The identity hashes one string, [`Self::hashed_spelling`]:
/// `build`, then for each switch `+<len>:<NAME>` and, for a switch with a
/// value, `=<len>:<VALUE>` (lengths in bytes), under the type tag
/// `runid::inputs::EngineVersion` at schema version 1. That is exactly the
/// newtype string the engine identity was before the switches had a field of
/// their own, so no stored run re-keys. The lengths make the spelling
/// injective in the switches as long as `build` holds no `+`, which the build
/// fingerprint, hex digits, cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineIdentity {
    pub build: String,
    pub switches: Vec<EngineSwitch>,
}

impl EngineIdentity {
    /// A build with no engine switch on.
    pub fn of_build(build: impl Into<String>) -> Self {
        EngineIdentity { build: build.into(), switches: Vec::new() }
    }

    /// The one string the identity hashes (see the type doc).
    pub fn hashed_spelling(&self) -> String {
        let mut s = self.build.clone();
        for EngineSwitch { name, value } in &self.switches {
            s.push_str(&format!("+{}:{name}", name.len()));
            if let Some(value) = value {
                s.push_str(&format!("={}:{value}", value.len()));
            }
        }
        s
    }
}

impl ContentAddressed for EngineIdentity {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_type_tag("runid::inputs::EngineVersion");
        h.write_schema_version(1);
        self.hashed_spelling().hash_into(h);
    }
}

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

/// Resolved scenario delta (the `scenario` level): sorted id-sets, a
/// canonical `set` patch, and the `scale` factors. The empty delta hashes to
/// its **real** `scen_h8`; `baseline` is the display label only, never a
/// literal zero hash.
///
/// The delta is the scenario's *effective* one — every field inherited
/// through `compose` included — as resolved by the same authority the value
/// resolver applies (`params_resolver::resolve_preset_delta` in the CLI).
///
/// `schema_version = 2` (gh#573): added `scale`. Before it, two presets
/// differing only in `scale` (or only in what they `compose`) hashed to one
/// `scen_h8` while producing different trajectories. The field contributes
/// bytes when empty too, so every sim leaf re-keys — deliberate, versioned
/// turnover.
#[derive(Debug, Clone, PartialEq, RunInput)]
#[run_input(schema_version = 2)]
pub struct ResolvedScenario {
    pub enabled: BTreeSet<InterventionId>,
    pub disabled: BTreeSet<InterventionId>,
    pub patch: BTreeMap<ParamId, FiniteF64>,
    /// Multiplicative factors per parameter, in application order. A list,
    /// not a product: the resolver multiplies the value by each factor in
    /// turn, and `(v·a)·b` need not equal `v·(a·b)` bit-for-bit, so the
    /// factor sequence — not its product — is what determines the result.
    pub scale: BTreeMap<ParamId, Vec<FiniteF64>>,
}

/// Resolved simulation config (the `config` level).
///
/// `schema_version = 2` (gh#156): adding the output-view fields (`no_flows` /
/// `columns`) changes the hashed bytes of every `SimConfig`, re-keying all sim
/// leaves — a deliberate, versioned turnover (per the re-key policy in the
/// runid doc), not a collateral churn. Existing cached sims re-run on next use.
///
/// `schema_version = 3` (gh#143): `ResolvedOutputSchedule::Regular` dropped its
/// `end` field — the output horizon collapsed onto `simulation.t_end` (still
/// hashed here as `t_end`), so no identity is lost, only re-keyed. Another
/// deliberate, versioned turnover.
///
/// `schema_version = 4` (gh#641): added `init_state`. A run seeded from a
/// filtered-state file computes a different trajectory from the same model, so
/// the file's bytes and the restored row must re-key; the field contributes
/// bytes at its `None` default too, which re-keys every existing sim leaf.
/// Deliberate, versioned turnover, in the gh#156 / gh#143 line.
///
/// `schema_version = 5` (gh#833): no field changed — the leaf's `traj.tsv`
/// did. Every trajectory row now carries `t_start`/`t_stop` beside `t`, the
/// period its flow columns were accumulated over, so the bytes a leaf holds
/// for an unchanged model, config and seed are different bytes. A stored leaf
/// is the artifact its key names; a key that stayed put would name an
/// artifact a recompute no longer reproduces. The format enters the key by
/// this bump, exactly as the obs-file format enters the `obs/` subtree key.
///
/// `schema_version = 6` (temporal-architecture migration, Stage 4): no field
/// changed — the leaf's `run.json` did. It now carries the model's
/// `calendar` block, so the bytes a leaf holds for an unchanged model, config
/// and seed differ from those an earlier camdl stored; by the gh#833 rule the
/// format enters the key. Every stored sim leaf re-keys once and is
/// recomputed on next use.
///
/// `allow_forcing_extrapolation` (temporal-architecture migration, Stage 2,
/// D44) was added under the store-wide `HASH_VERSION` 5 bump, which re-keys
/// every sim leaf on its own, so the schema version did not move with it.
#[derive(Debug, Clone, PartialEq, RunInput)]
#[run_input(schema_version = 6)]
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
    /// Semantic: `--allow-forcing-extrapolation` lets the run hold a
    /// forcing's end value outside the range it is defined on, where without
    /// it the run is refused (D44). A run under it and the same run without
    /// it are two runs.
    pub allow_forcing_extrapolation: bool,
    /// Output view (gh#156): which trajectory columns are written to the leaf.
    /// `--no-flows` / `--columns` change the leaf's *bytes* (a column subset),
    /// so a content-addressed leaf cannot share a `run_id` with the full one —
    /// these are identity, not presentation. `--output-every` is NOT here: it
    /// lowers into the model's output schedule (`output` above + the model
    /// digest). Default (full output) keeps the historical bytes; a non-default
    /// view is a distinct, reproducible artifact.
    pub no_flows: bool,
    /// `--columns` allow-list of output column names, normalized to a set
    /// (order-invariant — emitted order follows the model). Empty = all.
    pub columns: BTreeSet<String>,
    /// gh#641 / gh#697: the inferred state this run started from, when it did
    /// not start from the model's `init {}`. `None` = the model's own initial
    /// conditions (every run before `--init-state` existed). See
    /// [`InitStateDigest`] for why both the ensemble digest and the row index
    /// are identity, and why the fit source keys on ensemble *content* rather
    /// than on the fit it came from.
    pub init_state: Option<InitStateDigest>,
}

/// The model-level digest: the whole canonical IR, the IR schema version and
/// the engine.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct ModelDigest {
    /// `content_hash` of the whole canonical IR.
    pub ir: ContentHash,
    /// The IR schema version (e.g. `"0.7"`), matching `ir/VERSION`.
    pub ir_version: String,
    pub engine: EngineIdentity,
}

impl ModelDigest {
    /// Build the whole-IR digest from a compiled model. Every identity path
    /// (`sim`, `fit`, `pfilter`, `survey`, `sim_ensemble`, `profile`) builds
    /// its model level through this constructor, so all of them hash the
    /// model by [`model_ir_hash`].
    pub fn from_model(model: &ir::Model, ir_version: String, engine: EngineIdentity) -> Self {
        Self { ir: model_ir_hash(model), ir_version, engine }
    }
}

/// The model's content hash: the [`ModelDigest::ir`] component.
///
/// Every CAS path that needs a structural model identity goes through here:
/// the per-level [`ModelDigest`] via [`ModelDigest::from_model`], and the bare
/// structural identity string (the survey↔fit warm-start cross-check, which
/// leaves out the engine and IR-version fold).
pub fn model_ir_hash(model: &ir::Model) -> ContentHash {
    model.content_hash()
}

/// [`model_ir_hash`] with the compiler-derived gradient maps folded in
/// ([`CanonicalHasher::with_gradients`]): `rate_grad`, `rate_state_grad`,
/// `sigma_sq_grad`, `projection_state_grad`, `ic_grad` and every `Diffable`'s
/// `grad` / `proj_grad`, a reference to a hoisted subtree hashed as its body.
///
/// Model identity leaves those maps out, so that compiling a model with or
/// without them does not re-key the runs that never read them (proposal
/// 2026-07-16). A fit whose method reads them puts this digest in its method
/// level instead (proposal 2026-11-17 §1b): its result depends on the
/// gradients, and the gradients on the compiler's autodiff, which no other
/// input of its identity sees. Two models with one forward model and
/// different gradient maps share a [`model_ir_hash`] and differ here.
pub fn model_ir_hash_with_gradients(model: &ir::Model) -> ContentHash {
    let mut h = CanonicalHasher::new();
    h.write_u16(crate::hash::HASH_VERSION);
    h.with_gradients(|h| model.hash_into(h));
    h.finalize()
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
        // Exhaustive destructure: a field added to `Deps` must not silently
        // escape the hash.
        let Deps(deps) = self;
        let mut refs: Vec<&ArtifactRef> = deps.iter().collect();
        refs.sort_by(|a, b| a.run_id.as_bytes().cmp(b.run_id.as_bytes()));
        h.write_len(refs.len() as u64);
        for r in refs {
            r.hash_into(h);
        }
    }
}

/// Fit-wide digest (the `fit` level): model + data + the whole canonicalized
/// fit.toml + engine. Hashing the *whole* canonicalized document (not an
/// enumerated subset) is the include-by-default posture — dropping a field
/// like `ic_free`/`holdout`/fit-level `dt` would under-invalidate θ̂.
///
/// `schema_version = 2`: the encoding that carries `tables`.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
#[run_input(schema_version = 2)]
pub struct FitDigest {
    pub model: ModelDigest,
    /// Content digest of each resolved *training* observation stream.
    pub data: Vec<DataDigest>,
    /// Content digest of each `[data.holdout]` stream's *bytes* (gh#190). The
    /// fit.toml blob only carries the holdout file *paths*; editing a holdout
    /// file's content while keeping its path would otherwise reuse a stale fit
    /// (and so a stale held-out predictive score) under an unchanged `run_id`.
    /// Empty when no explicit holdout is configured. Sorted by stream name in
    /// the builder for a stable order.
    pub holdout_data: Vec<DataDigest>,
    /// Content digest of each file bound to the model's external inputs
    /// (`[data.tables]`), sorted by external name. By content, never by path:
    /// the paths are left out of `fit_toml`, so the same bytes under another
    /// path are the same fit, and one changed byte is another. Empty for a
    /// model with no external input.
    pub tables: Vec<DataDigest>,
    /// Content digest of the whole canonicalized fit.toml document.
    pub fit_toml: ContentHash,
    pub engine: EngineIdentity,
    /// The version of the rule that decides which rows train
    /// (`cli` `fit::training_partition::PARTITION_RULE_VERSION`). The fields
    /// above key the training files and the config, not the rows the rule
    /// picks from them, so a change to the rule re-keys every fit here rather
    /// than through a `HASH_VERSION` bump someone has to remember.
    pub partition_rule_version: u16,
}

/// One method's config (the `method` level): the whole canonicalized `Method`
/// config + the resolved obs-block + flow indices + `target_length`; its
/// `deps` ride beside it in the method level ([`MethodLevel`]). The whole
/// config is hashed, not a chosen subset, as at the fit level: an enumerated
/// subset misses the field nobody listed.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct MethodConfig {
    /// Content digest of the whole canonicalized method config block.
    pub config: ContentHash,
    /// Resolved obs-block name (the `--obs` selection): selects which series
    /// drives the likelihood; not in the toml.
    pub obs_block: String,
    /// Resolved flow-index selection (the `--flow` selection).
    pub flow_indices: Vec<u32>,
    /// Resume target length — a resumed run is a distinct artifact.
    pub target_length: u64,
}

/// The `method` *level* hash input: a method's own config **plus its `deps`**.
/// This is what the factored path's method segment (`{algorithm}-{method_h8}`)
/// and the leaf `run_id` hash for the method level — so a run started
/// `from_mle = @mle` folds in that source's identity (via `deps`) and re-keys
/// whenever the source changes, while editing its own `[method]` leaves the
/// source leaf untouched.
/// Kept distinct from [`MethodConfig`] so the bare config (without lineage)
/// stays addressable.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct MethodLevel {
    pub config: MethodConfig,
    pub deps: Deps,
}

// ── profile levels ─────────────────────────────────────────────────────────
//
// A profile pins a focal parameter at each value of a grid and sub-fits the
// rest at that pin. Its levels are `[base, point, method, seed, start]`:
//
//   base          — the inference problem (model, data, the canonical base
//                   config), the sweep grid, the engine, and the chain-start
//                   source as a dep. A path segment only, with no record of
//                   its own (as the fit level).
//   point         — the pinned focal value(s) of one grid point.
//   method        — the sub-fit method and its settings, shared across the grid.
//   seed / start  — the resolved RNG seed and the multi-start index; the pair
//                   pins the sub-fit's start deterministically.

/// The shared `profile`-level digest: the inference problem being profiled,
/// including the sweep grid so a distinct grid is a distinct run. `deps`
/// carries the base fit's `starts_from` lineage (same deps-DAG mechanism as a
/// fit method).
///
/// The grid is part of the identity (not `Eq` because it carries floats — same
/// as [`ProfilePointConfig`]). Two runs with the same problem but a *different*
/// grid — a wider range, more points, shifted bounds — get different base dirs
/// rather than silently merging their cells onto one another (which produced a
/// jagged union when the cell coordinates didn't line up). Re-running the
/// *same* grid is stable (a cache hit); the method config still lives in the
/// `method` level, not here.
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct ProfileBase {
    pub model: ModelDigest,
    pub data: Vec<DataDigest>,
    /// Canonical digest of the base config (base params + fixed + obs + priors
    /// + fit.toml) — method config excluded (it lives in `method`).
    pub base_config: ContentHash,
    /// The resolved sweep grid: each focal axis and its values, canonicalized
    /// (axes sorted by name, values ascending) so the identity is independent
    /// of `--sweep` argument order.
    pub grid: Vec<(ParamId, Vec<FiniteF64>)>,
    pub engine: EngineIdentity,
    pub deps: Deps,
}

/// The `point` level: the focal parameter(s) pinned at this grid value. Only
/// the single value lives here; the whole grid is in [`ProfileBase`].
#[derive(Debug, Clone, PartialEq, RunInput)]
pub struct ProfilePointConfig {
    pub focal: Vec<(ParamId, FiniteF64)>,
}

/// The `method` level: the sub-fit method + hyperparams applied at every point.
/// Distinct from [`ProfileBase`] so a method change re-keys the points while
/// the base stays shared.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct ProfileMethod {
    pub config: ContentHash,
}

/// The `start` level: a multi-start index. Kept separate from `seed` so each
/// start's deterministically-derived init is its own addressable leaf.
#[derive(Debug, Clone, PartialEq, Eq, RunInput)]
pub struct StartLevel {
    pub index: u32,
}

#[cfg(test)]
mod tests;
