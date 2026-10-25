//! The resolved engine input behind `camdl simulate` — run-spec §3
//! (`SimulateJob`, `ParamSource`, `Seeds`, `ObsOutput`, `ScenarioRef`) — and
//! scenario-reference resolution.
//!
//! `simulate` lowers its flags into one `SimulateJob`, and `fit predict`
//! builds one for its free-forward replay; a single `crate::engine::run_job`
//! drives the `param-point × scenario × seed-slot` cell loop for both. The
//! per-cell seed arithmetic, iteration order and `SimRun` construction are the
//! engine's (the determinism PIN in `tests/determinism_pin.rs` is the
//! tripwire), and each entry point supplies a `crate::engine::RunSink` for its
//! output shape.

use std::path::PathBuf;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::args::types::ForwardBackend;

// ─── SimulateJob (run-spec §3.1) ──────────────────────────────────────────────

/// Everything needed to run one or more simulations — run-spec §3.1.
/// `camdl simulate` builds one from CLI args (`main.rs`) and `fit predict`
/// builds one for its free-forward replay; both hand it to
/// [`crate::engine::run_job`].
///
/// This is the *resolved* engine input: the front end does its own arg
/// parsing and populates the concrete fields, and the struct carries no
/// serde derives — there is no wire schema for it to drift from.
#[derive(Debug, Clone)]
pub struct SimulateJob {
    /// Resolved IR/`.camdl` path, verbatim from the CLI.
    pub model: String,
    /// `--params` files, applied in order.
    pub params_files: Vec<String>,
    pub backend: ForwardBackend,
    pub dt: f64,
    /// gh#166: optional CLI `--integrator` override (method only); `None` → the
    /// model's declared integrator.
    pub integrator: Option<crate::args::types::IntegratorArg>,
    /// Where parameter vectors come from (the central dispatch).
    pub source: ParamSource,
    /// σ layer — which scenarios to run. Empty ⇒ a single implicit
    /// baseline (run-spec §3.6).
    pub scenarios: Vec<ScenarioRef>,
    /// gh#626: the resolved `--to` horizon override (model time), applied to
    /// every cell after the scenario horizon in `resolve_run_model`, and keyed
    /// into run identity via `ResolvedEntry.t_end`. `None` = no override.
    pub t_end_override: Option<f64>,
    /// Times every cell must record a trajectory snapshot at, beyond the
    /// model's own `output { trajectories { … } }` schedule — copied verbatim
    /// onto each cell's [`crate::util::SimRun`], where
    /// `apply_required_output_times` folds them into the schedule and raises
    /// the integration end to reach them. `fit predict` supplies its forecast
    /// rows' closing boundaries here; empty everywhere else.
    pub required_output_times: Vec<f64>,
    /// gh#641: the loaded `--init-state` file. Shared by every cell (it is one
    /// ensemble of states); the replicate index selects the row. `None` = the
    /// model's own `init {}`.
    pub init_state: Option<std::sync::Arc<InitStateSource>>,
    /// gh#616: the run's resolved observation window, folded ONCE for the
    /// whole job and copied to every cell, so a sweep cannot resolve a
    /// different `last_obs` per cell.
    pub obs_anchors: Option<ir::anchor::ObsAnchorTimes>,
    /// S layer.
    pub seeds: Seeds,
    /// `--param NAME=VALUE` CLI overrides merged on top of every cell
    /// (M layer, highest precedence).
    pub cli_overrides: Vec<(String, f64)>,
    /// A `--draws prior --fit FIT.toml` config's `[fixed]` block, copied to
    /// every cell's `SimRun::fit_fixed` and resolved at the `fit.toml [fixed]`
    /// tier (gh#949). Empty for every other source.
    pub fit_fixed: IndexMap<String, f64>,
    /// `--param-vec PREFIX=FILE` entries (CLI only).
    pub set_vec_entries: Vec<(String, String)>,
    /// `--table NAME=FILE` entries (CLI only).
    pub table_files: Vec<(String, String)>,
    /// Synthetic-observation output mode.
    pub obs: ObsOutput,
    /// Rayon thread count for the simulation phase (1 = sequential).
    pub parallel: usize,
}

/// One forecast-origin state: the compartment values a cell restores instead of
/// building them from the model's `init {}` block.
#[derive(Debug, Clone, PartialEq)]
pub struct OriginState {
    /// Integer compartment counts, in the model's integer-compartment order
    /// (i.e. indexed like `sim::IntState::counts`).
    pub counts: Vec<i64>,
    /// Real compartment values, in the model's real-compartment order. Always
    /// empty for a `--save-final-state` file — the particle filter's
    /// `ParticleState` carries counts only, and the reader refuses a model with
    /// a real compartment rather than defaulting the reservoir to zero.
    pub reals: Vec<f64>,
}

/// Which grid axis picks a cell's row out of an [`InitStateSource`].
///
/// The two sources index different things, and getting this wrong is a silent
/// mis-pairing rather than a crash — so it is a type the row lookup matches on,
/// not a convention a comment asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitStateRowAxis {
    /// `--init-state FILE` (gh#641): one θ's particle swarm, so **replicate**
    /// `i` restores row `i`. Deliberately not "the first N rows of a larger
    /// file" — a post-resampling swarm is ancestor-ordered, so a prefix is not
    /// an exchangeable subsample of the filtering distribution.
    Replicate,
    /// `--init-state fit` (gh#697): the paired `(θ_i, X_i(T))` posterior, so
    /// **draw** `i` restores row `i` — its own inferred state, under its own θ.
    Draw,
}

/// A loaded `--init-state` ensemble: every origin state, the time they all sit
/// at, and the ensemble's content digest.
///
/// One of these is shared by every cell of a job; [`InitStateRowAxis`] says
/// which index selects a cell's row.
#[derive(Debug, Clone)]
pub struct InitStateSource {
    /// The model time the states sit at (the filter's last observation time for
    /// a state file; the terminal snapshot of the saved latent paths for a
    /// fit). Becomes each cell's `simulation.t_start`.
    pub origin_t: f64,
    /// The ensemble, in the order the row axis indexes.
    pub states: Vec<OriginState>,
    pub axis: InitStateRowAxis,
    /// Content digest of the ensemble — the identity input (see
    /// [`runid::inputs::InitStateDigest`]).
    pub ensemble_digest: runid::ContentHash,
}

/// Where parameter vectors come from — run-spec §3.2. Exactly one variant
/// is active per job. The engine expands this into an ordered list of
/// per-cell parameter-override maps (draws, or design points).
#[derive(Debug, Clone)]
pub enum ParamSource {
    /// Single point: base params + CLI overrides. One param-point, run
    /// `replicates` times (each replicate a distinct XOR-mixed seed when
    /// `Seeds::Single`). `simulate --replicates N` (no `--draws`) sets this.
    Point { replicates: usize },
    /// Deterministic grid: Cartesian product of swept values. Each point
    /// overrides the corresponding key in the base params, run `replicates`
    /// times. `simulate --sweep` / `--sweep-file` set this; with `--seeds`
    /// the explicit list drives replication and `replicates` tracks its length.
    Sweep { points: Vec<IndexMap<String, f64>>, replicates: usize },
    /// Pre-resolved parameter draws (posterior file / prior / uniform).
    /// Each row is a complete (or override) parameter vector.
    Draws {
        rows: Vec<IndexMap<String, f64>>,
        /// Stochastic replicates per draw (different seeds, same params).
        replicates: usize,
        /// `Some(path)` iff the draws came from a USER-AUTHORED FILE
        /// (`--draws <file.tsv>`); `None` for generated draws
        /// (`--draws posterior/prior/uniform`). A scenario that sets or
        /// scales a parameter the file's columns also provide is refused
        /// (`engine::check_scenario_coordinate_collision`, gh#572): it would
        /// override that column in every draw. Generated draws are not
        /// checked — a scenario over them is the intended counterfactual, and
        /// the scenario tier applies (spec §1.3). The path is carried so the
        /// diagnostic can name the file.
        explicit_file: Option<PathBuf>,
    },
}

impl ParamSource {
    /// The ordered list of per-cell parameter-override maps. For `Point`
    /// this is a single empty map (base params unchanged).
    pub fn param_points(&self) -> Vec<IndexMap<String, f64>> {
        match self {
            ParamSource::Point { .. } => vec![IndexMap::new()],
            ParamSource::Sweep { points, .. } => {
                if points.is_empty() { vec![IndexMap::new()] } else { points.clone() }
            }
            ParamSource::Draws { rows, .. } => rows.clone(),
        }
    }

    /// Replicates per param-point. Honored for every variant so
    /// `--replicates N` produces N stochastic replicates regardless of
    /// whether params come from a point, a sweep, or a draw set.
    pub fn replicates(&self) -> usize {
        match self {
            ParamSource::Point { replicates }
            | ParamSource::Sweep { replicates, .. }
            | ParamSource::Draws { replicates, .. } => *replicates,
        }
    }
}

/// S layer — run-spec §3.5. The seed values, plus whether they were given
/// *explicitly* (the load-bearing distinction for the determinism
/// contract: explicit seeds index directly, derived replicates XOR-mix).
#[derive(Debug, Clone)]
pub enum Seeds {
    /// A single base seed; replicates derive seeds via XOR-mixing
    /// (`main.rs` historical `seed ^ draw*MIX ^ rep*MIX`).
    Single(u64),
    /// An explicit list (`--seeds`). Each seed is a
    /// seed-slot used verbatim — "seed N means the same trajectory."
    Explicit(Vec<u64>),
}

impl Seeds {
    /// `Some(&[..])` when the seeds were given explicitly, `None` for a
    /// single base seed with derived replicates. Mirrors `main.rs`'s
    /// `seeds_spec_given` flag.
    pub fn explicit(&self) -> Option<&[u64]> {
        match self {
            Seeds::Single(_) => None,
            Seeds::Explicit(v) => Some(v),
        }
    }

    /// The base seed (first explicit seed, or the single seed).
    pub fn base(&self) -> u64 {
        match self {
            Seeds::Single(s) => *s,
            Seeds::Explicit(v) => *v.first().unwrap_or(&1),
        }
    }
}

/// Synthetic-observation output mode — run-spec §3.1.1. `OnlyFile`/
/// `OnlyDir` suppress the trajectory.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ObsOutput {
    #[default]
    None,
    /// Single wide-format TSV (errors if streams have different schedules).
    File(PathBuf),
    /// One TSV per stream in a directory.
    Dir(PathBuf),
    /// Like `File`, trajectory suppressed.
    OnlyFile(PathBuf),
    /// Like `Dir`, trajectory suppressed.
    OnlyDir(PathBuf),
}

impl ObsOutput {
    /// True when the trajectory output is suppressed.
    pub fn suppresses_trajectory(&self) -> bool {
        matches!(self, ObsOutput::OnlyFile(_) | ObsOutput::OnlyDir(_))
    }

    /// `Some(path)` for the single-file modes (`File`/`OnlyFile`).
    pub fn file_path(&self) -> Option<&PathBuf> {
        match self {
            ObsOutput::File(p) | ObsOutput::OnlyFile(p) => Some(p),
            _ => None,
        }
    }

    /// `Some(path)` for the dir modes (`Dir`/`OnlyDir`).
    pub fn dir_path(&self) -> Option<&PathBuf> {
        match self {
            ObsOutput::Dir(p) | ObsOutput::OnlyDir(p) => Some(p),
            _ => None,
        }
    }
}

// ─── ScenarioRef (run-spec §3.6) ──────────────────────────────────────────────

/// A scenario reference in a simulation job. Either a name pointing at a
/// model `scenarios{}` preset, or an inline ad-hoc patch — never both
/// (run-spec §3.6; the exclusivity mirrors `simulate`'s `--scenario` vs
/// `--enable`/`--disable` rule). Untagged so TOML `name = "x"` and a
/// `[[scenario]]` table with `enable`/`params` both deserialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioRef {
    /// Reference a scenario defined in the `.camdl` file.
    Named(String),

    /// Inline definition (a `[[scenario]]` entry carrying patches).
    Inline {
        name: String,
        #[serde(default)]
        enable: Vec<String>,
        #[serde(default)]
        disable: Vec<String>,
        #[serde(default)]
        params: IndexMap<String, f64>,
    },
}

/// The derived label of `simulate`'s ad-hoc arm — the one arm a run without
/// `--scenario` has (proposal 2026-10-24-batch-to-simulate §3 "Arm labels").
///
/// `baseline` when nothing is overridden; otherwise `baseline` followed by
/// `+name` for each enabled intervention, `-name` for each disabled one, and
/// `;param=value` for each `--param`, each group sorted by name and each value
/// in Rust's shortest round-trip form, so the label is a pure function of the
/// resolved arm and `beta=.3` and `beta=0.3` give one label:
/// `baseline+vaccination`, `baseline-sia`, `baseline;beta=0.3`. A derived
/// label cannot collide with a declared scenario, whose name is an identifier.
/// The label is provenance — it names the leaf's path segment, its `run.json`
/// level and the listings — never identity, which the resolved delta hashes.
pub fn adhoc_arm_label(enable: &[String], disable: &[String], params: &[(String, f64)]) -> String {
    let mut label = String::from("baseline");
    let mut enable: Vec<&String> = enable.iter().collect();
    enable.sort();
    for e in enable {
        label.push('+');
        label.push_str(e);
    }
    let mut disable: Vec<&String> = disable.iter().collect();
    disable.sort();
    for d in disable {
        label.push('-');
        label.push_str(d);
    }
    let mut params: Vec<&(String, f64)> = params.iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, value) in params {
        label.push(';');
        label.push_str(name);
        label.push('=');
        label.push_str(&value.to_string());
    }
    label
}

impl ScenarioRef {
    /// The display/slug name of this reference.
    pub fn name(&self) -> &str {
        match self {
            ScenarioRef::Named(n) => n,
            ScenarioRef::Inline { name, .. } => name,
        }
    }

    /// True when this ref carries inline patch fields.
    fn has_inline_fields(&self) -> bool {
        match self {
            ScenarioRef::Named(_) => false,
            ScenarioRef::Inline { enable, disable, params, .. } => {
                !enable.is_empty() || !disable.is_empty() || !params.is_empty()
            }
        }
    }
}

/// The result of resolving a [`ScenarioRef`] against a model.
///
/// Exactly one of the two routes is taken (the locked design decision in
/// the 2026-05-28 proposal):
///   - `Preset` — the name matched a model `scenarios{}` preset; the
///     `params_resolver` preset path supplies the enable/disable/set/scale.
///   - `Adhoc` — the name matched nothing but inline patches were given.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedScenario {
    /// Route through the named-preset branch of `params_resolver`
    /// (`scenario_name = Some(name)`). The model preset is the source of
    /// truth for enable/disable/params/scale/compose.
    Preset { name: String },
    /// Route through the ad-hoc branch (`scenario_name = None`) with the
    /// inline enable/disable/params applied.
    Adhoc {
        name: String,
        enable: Vec<String>,
        disable: Vec<String>,
        params: Vec<(String, f64)>,
    },
}

/// Resolve a `ScenarioRef` against the model's presets, per the locked
/// design decision (proposal §"ScenarioRef semantics"):
///
/// 1. name matches a model preset → resolve via the preset path.
/// 2. name matches nothing but inline enable/disable/params present →
///    ad-hoc patch.
/// 3. name matches nothing and no inline fields → hard error listing
///    available presets.
/// 4. name matches a preset AND has inline fields → hard error (the
///    model scenario is the source of truth).
pub fn resolve_scenario_ref(
    scenario: &ScenarioRef,
    model_preset_names: &[String],
) -> Result<ResolvedScenario, String> {
    let name = scenario.name();
    let is_preset = model_preset_names.iter().any(|p| p == name);
    let has_inline = scenario.has_inline_fields();

    match (is_preset, has_inline) {
        // Case 4: collides with a preset *and* carries inline patches.
        (true, true) => Err(format!(
            "scenario '{name}' names a model preset but also carries inline \
             enable/disable/params. A scenario reference is either a model \
             preset OR an ad-hoc patch, never both — the model scenario is \
             the source of truth.\n  \
             Fix: drop the inline fields to use the model preset, or rename \
             the inline scenario so it does not shadow a preset."
        )),
        // Case 1: pure preset reference.
        (true, false) => Ok(ResolvedScenario::Preset { name: name.to_string() }),
        // Case 2: ad-hoc patch (name matched nothing, inline fields present).
        (false, true) => {
            let (enable, disable, params) = match scenario {
                ScenarioRef::Inline { enable, disable, params, .. } => (
                    enable.clone(),
                    disable.clone(),
                    params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                ),
                // Unreachable: has_inline is false for Named.
                ScenarioRef::Named(_) => (Vec::new(), Vec::new(), Vec::new()),
            };
            Ok(ResolvedScenario::Adhoc {
                name: name.to_string(),
                enable,
                disable,
                params,
            })
        }
        // Case 3a: an implicit-identity sentinel name — `baseline` (run-spec
        // §3.6: "a single implicit baseline — the absence of any scenario
        // patch") for `simulate`, or `fitted` for `camdl fit predict` (the
        // no-overlay row: the fitted model, no scenario applied). Both are
        // always valid even when the model declares no preset by that name:
        // they mean "the model as written, no modifications." Resolves to an
        // empty ad-hoc patch; the empty scenario delta hashes to its real
        // scenario-level digest (the name is the display label only).
        (false, false) if name == "baseline" || name == "fitted" => {
            Ok(ResolvedScenario::Adhoc {
                name: name.to_string(),
                enable: Vec::new(),
                disable: Vec::new(),
                params: Vec::new(),
            })
        }
        // Case 3b: typo / unknown — neither a preset, an ad-hoc patch, nor
        // the implicit baseline.
        (false, false) => {
            let available = if model_preset_names.is_empty() {
                "(none)".to_string()
            } else {
                model_preset_names
                    .iter()
                    .map(|p| format!("'{p}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            Err(format!(
                "scenario '{name}' is not a model preset and defines no inline \
                 enable/disable/params.\n  \
                 Available model presets: {available}.\n  \
                 Fix: use one of the listed presets, add enable/disable/params \
                 to define an ad-hoc scenario, or use 'baseline' for the \
                 unmodified model."
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn adhoc_arm_label_is_a_pure_function_of_the_resolved_arm() {
        assert_eq!(adhoc_arm_label(&[], &[], &[]), "baseline");
        assert_eq!(adhoc_arm_label(&names(&["vaccination"]), &[], &[]), "baseline+vaccination");
        assert_eq!(adhoc_arm_label(&[], &names(&["sia"]), &[]), "baseline-sia");
        let p = |v: f64| vec![("beta".to_string(), v)];
        assert_eq!(adhoc_arm_label(&[], &[], &p(0.3)), "baseline;beta=0.3");
        // `beta=.3` parses to the same f64 as `beta=0.3`: one label.
        assert_eq!(adhoc_arm_label(&[], &[], &p(".3".parse().unwrap())), "baseline;beta=0.3");
        // Groups sorted by name, flag order irrelevant.
        assert_eq!(
            adhoc_arm_label(
                &names(&["sia", "vaccination"]),
                &names(&["closure"]),
                &[("gamma".to_string(), 0.1), ("beta".to_string(), 2.0)],
            ),
            adhoc_arm_label(
                &names(&["vaccination", "sia"]),
                &names(&["closure"]),
                &[("beta".to_string(), 2.0), ("gamma".to_string(), 0.1)],
            )
        );
        assert_eq!(
            adhoc_arm_label(&names(&["vaccination", "sia"]), &names(&["closure"]), &[("beta".to_string(), 2.0)]),
            "baseline+sia+vaccination-closure;beta=2"
        );
    }

    #[test]
    fn resolve_named_preset() {
        let r = resolve_scenario_ref(
            &ScenarioRef::Named("baseline".into()),
            &names(&["baseline", "fast"]),
        )
        .unwrap();
        assert_eq!(r, ResolvedScenario::Preset { name: "baseline".into() });
    }

    #[test]
    fn resolve_inline_matching_preset_name_resolves_as_preset() {
        // An Inline entry with NO patch fields whose name matches a preset
        // is treated as a pure preset reference (case 1) — the shape of a
        // bare arm name that matches a preset.
        let r = resolve_scenario_ref(
            &ScenarioRef::Inline {
                name: "baseline".into(),
                enable: vec![],
                disable: vec![],
                params: IndexMap::new(),
            },
            &names(&["baseline"]),
        )
        .unwrap();
        assert_eq!(r, ResolvedScenario::Preset { name: "baseline".into() });
    }

    #[test]
    fn resolve_adhoc_patch() {
        let mut params = IndexMap::new();
        params.insert("beta".to_string(), 0.5);
        let r = resolve_scenario_ref(
            &ScenarioRef::Inline {
                name: "high".into(),
                enable: vec!["sia".into()],
                disable: vec![],
                params,
            },
            &names(&["baseline"]),
        )
        .unwrap();
        match r {
            ResolvedScenario::Adhoc { name, enable, params, .. } => {
                assert_eq!(name, "high");
                assert_eq!(enable, vec!["sia".to_string()]);
                assert_eq!(params, vec![("beta".to_string(), 0.5)]);
            }
            other => panic!("expected adhoc, got {other:?}"),
        }
    }

    #[test]
    fn resolve_unknown_name_no_inline_errors_lists_presets() {
        let err = resolve_scenario_ref(
            &ScenarioRef::Named("typo".into()),
            &names(&["baseline", "fast"]),
        )
        .unwrap_err();
        assert!(err.contains("'typo'"), "{err}");
        assert!(err.contains("'baseline'"), "{err}");
        assert!(err.contains("'fast'"), "{err}");
    }

    #[test]
    fn resolve_baseline_with_no_presets_is_implicit_identity() {
        // `baseline` on a model with no scenarios{} block is the implicit
        // identity patch, not a typo (run-spec §3.6).
        let r = resolve_scenario_ref(&ScenarioRef::Named("baseline".into()), &[]).unwrap();
        assert_eq!(
            r,
            ResolvedScenario::Adhoc {
                name: "baseline".into(),
                enable: vec![],
                disable: vec![],
                params: vec![],
            }
        );
    }

    #[test]
    fn resolve_fitted_with_no_presets_is_implicit_identity() {
        // `fitted` is `camdl fit predict`'s no-overlay sentinel: valid on a
        // model with no scenarios{} block, resolving to the identity patch (the
        // sibling of `baseline`).
        let r = resolve_scenario_ref(&ScenarioRef::Named("fitted".into()), &[]).unwrap();
        assert_eq!(
            r,
            ResolvedScenario::Adhoc {
                name: "fitted".into(),
                enable: vec![],
                disable: vec![],
                params: vec![],
            }
        );
        // Also via the inline-no-patch form (what `fit predict` builds for the
        // no-`--scenario` case).
        let r2 = resolve_scenario_ref(
            &ScenarioRef::Inline {
                name: "fitted".into(),
                enable: vec![],
                disable: vec![],
                params: IndexMap::new(),
            },
            &[],
        )
        .unwrap();
        assert!(matches!(r2, ResolvedScenario::Adhoc { .. }));
    }

    #[test]
    fn resolve_baseline_inline_no_patch_with_no_presets_is_identity() {
        let r = resolve_scenario_ref(
            &ScenarioRef::Inline {
                name: "baseline".into(),
                enable: vec![],
                disable: vec![],
                params: IndexMap::new(),
            },
            &[],
        )
        .unwrap();
        assert!(matches!(r, ResolvedScenario::Adhoc { .. }));
    }

    #[test]
    fn resolve_preset_plus_inline_errors() {
        let mut params = IndexMap::new();
        params.insert("beta".to_string(), 0.5);
        let err = resolve_scenario_ref(
            &ScenarioRef::Inline {
                name: "baseline".into(),
                enable: vec![],
                disable: vec![],
                params,
            },
            &names(&["baseline"]),
        )
        .unwrap_err();
        assert!(err.contains("source of truth"), "{err}");
    }
}
