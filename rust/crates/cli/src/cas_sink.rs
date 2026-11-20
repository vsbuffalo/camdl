//! `CasSink` — the content-addressed store sink behind `camdl simulate`.
//!
//! Each engine cell (scenario × parameter point × replicate) is written as one
//! `Sim` leaf under `<output>/sims/…` in the factored `runid` layout (model /
//! config / params / scenario / seed levels). A cell's identity is resolved
//! through [`crate::resolve::resolve_trajectory`] from the cell's own `SimRun`,
//! so the leaf is keyed on exactly what ran. Synthetic observations, when
//! requested, are written as an `obs/` child of the leaf
//! ([`write_obs_into_cas`]) addressed by [`obs_subtree_hash`]. Both entry
//! points build the same [`CasSink`], so the leaf each writes for one logical
//! cell is byte-identical.

use std::collections::HashMap;

use crate::version;

/// A `[[scenario]]` entry after resolution against the model's
/// `scenarios{}` presets (CLI review finding #3).
///
/// `route` decides how the per-run `SimRun` is built:
///   - `Some(preset_name)` → route through the named-preset branch of
///     `params_resolver` (`scenario_name = Some(...)`), exactly as
///     `simulate --scenario` does. The model preset is the source of
///     truth for params/enable/disable/scale/compose.
///   - `None` → ad-hoc patch (`scenario_name = None`) using the inline
///     enable/disable/params.
///
/// `enable`/`disable`/`params`/`scale` are the *hash-relevant* delta. For a
/// preset they are its **effective** delta — every field inherited through
/// `compose` included — read from [`crate::params_resolver::resolve_preset_delta`],
/// the same authority the value resolver applies on the preset route, so the
/// run identity cannot disagree with what runs (gh#573). For an ad-hoc patch
/// they are the patch itself, which is also what runs.
///
/// The arm's horizon is not here: each cell's window is read off the model it
/// runs ([`crate::sim_config::ResolvedCell`]), where the scenario's own
/// `simulate { to }` and `--to` are applied.
#[derive(Debug, Clone)]
pub struct ResolvedEntry {
    pub name: String,
    pub route: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub params: HashMap<String, f64>,
    /// `scale` factors in application order. Always empty for an ad-hoc
    /// patch: an inline `[[scenario]]` has no `scale` spelling.
    pub scale: Vec<(String, f64)>,
}

impl ResolvedEntry {
    /// The entry for the model preset `name`, routed through the preset
    /// branch of the value resolver. The hashed delta comes from
    /// [`crate::params_resolver::resolve_preset_delta`] — the authority that
    /// branch applies.
    pub fn from_preset(model: &ir::Model, name: &str) -> Result<Self, String> {
        let delta = crate::params_resolver::resolve_preset_delta(model, name)
            .map_err(|e| e.to_string())?;
        Ok(ResolvedEntry {
            name: name.to_string(),
            route: Some(name.to_string()),
            enable: delta.enable,
            disable: delta.disable,
            params: delta.params.into_iter().collect(),
            scale: delta.scale,
        })
    }
}

// ─── Run metadata ─────────────────────────────────────────────────────────────

/// Descriptor for one completed cell, accumulated on `CasSink.completed_runs`
/// and used by the post-run leaf summary (`report_cas_leaves`). Each cell is
/// independently a `runid::RunRecord` leaf on disk; this is the in-memory
/// tally, not a file. Only the store-relative path is reported.
#[derive(Debug, Clone)]
pub(crate) struct RunEntry {
    pub(crate) run_path: String,
    /// The leaf's `run_id` — the ensemble (simulate, multi-cell) records these
    /// as its `deps` and folds them into its `grid` identity. `None` only if
    /// identity resolution failed (the cell is then also on `errors`).
    pub(crate) run_id: Option<runid::ContentHash>,
    /// SHA-256 of the cell's `traj.tsv` (the artifact the ensemble's combined
    /// TSV is derived from). Pins which upstream file the dep consumed.
    pub(crate) traj_digest: Option<runid::ContentHash>,
    /// The scenario label (display: the ensemble's `run.json` summary and the
    /// combined TSV's `scenario` column — never the `grid` digest), the
    /// resolved process seed, and the id the cell is written under on the
    /// parameter-point axis: a sweep cell's `point_id` (a sweep file's own
    /// column, else the 0-based design index), the draw index of a `--draws`
    /// cell. `(point_id, process_seed)` plus the cell `run_id` are what the
    /// ensemble's `grid` digest sorts over.
    pub(crate) scenario: String,
    pub(crate) process_seed: u64,
    pub(crate) point_id: usize,
    /// The cell's `params` level hash — the identity of the parameter values
    /// it ran. The ensemble's `params` level is the set of these (see
    /// `sim_ensemble_cas`). `None` only if identity resolution failed.
    pub(crate) params_level: Option<runid::ContentHash>,
}

// ─── CasSink: the content-addressed output strategy ───────────────────────────

/// `RunSink` over the content-addressed store: writes each cell into `sims/`
/// (the `runid` factored sim layout), resolving each cell's identity via
/// [`crate::resolve::resolve_trajectory`] so the path, `run.json` `run_id`
/// and recorded sweep point are keyed on exactly what ran. Cells the store
/// already holds are skipped via `should_run` (the engine never simulates
/// them); `simulate`'s own sink (`main::SimSink`) delegates to it and reads a
/// skipped cell back for the combined outputs.
pub(crate) struct CasSink {
    /// The invocation's resolved configuration: the base model and its
    /// `model` level, the run-wide `config` level (backend, `dt`, the column
    /// view, the evaluator flags) and the scenario arms. Every cell's identity
    /// is built from it ([`CasSink::cell_resolve`]), and the engine's settings
    /// come from the same value (`ResolvedSimConfig::engine_settings`).
    ///
    /// The base model (`sim.model.ir`) is never `cell.model()` (which has the
    /// scenario and the sweep point applied), and it is the model each cell's
    /// parameter values are resolved over. There is deliberately no base
    /// parameter map: a cell's `params` level is read off the model the cell
    /// runs, resolved from its own `SimRun` by `sim_config::ResolvedCell::resolve`,
    /// the resolution the engine makes (gh#583).
    pub(crate) sim: std::sync::Arc<crate::sim_config::ResolvedSimConfig>,
    /// The engine's settings, derived once from `sim`
    /// (`ResolvedSimConfig::engine_settings`): what each cell is checked
    /// against, and the forcing policy each leaf's record names.
    pub(crate) engine: crate::sim_config::EngineSettings,
    /// Absolute `<output>/sims` subtree.
    pub(crate) runs_dir: String,
    pub(crate) obs_enabled: bool,
    /// gh#656: `simulate --emit-every`, the per-stream emission-cadence
    /// override. It sets the emitted times AND (via [`obs_subtree_hash`]) the
    /// obs subtree's address, so two cadences are two artifacts under one
    /// trajectory leaf. `None` for every `simulate` without the flag — those
    /// key exactly as they always did.
    pub(crate) emit_every: Option<crate::emit_every::EmitEvery>,
    pub(crate) force: bool,
    pub(crate) total: usize,
    pub(crate) counter: usize,
    pub(crate) completed_runs: Vec<RunEntry>,
    pub(crate) errors: Vec<String>,
    /// User-supplied display label, recorded on each leaf's
    /// `RunRecord.provenance.label`. `simulate --label` sets this.
    pub(crate) label: Option<String>,
    /// Upstream-fit lineage recorded on each leaf's `RunRecord.deps`. Set
    /// when `simulate --params` consumes a fit-MLE params file carrying a
    /// `[provenance]` block (so the sim records which fit it came from);
    /// empty for standalone (non-fit) params. Provenance only —
    /// a sim's identity is its factored levels (resolve_trajectory), never
    /// `deps`, so this does not change the sim's run_id or store path.
    pub(crate) fit_dep: Vec<runid::inputs::ArtifactRef>,
    /// The overall cells progress bar (advanced once per committed or
    /// cache-hit cell, finished on the last). `Some` only for a multi-cell run
    /// (`total > 1`) — a LONE `simulate` gets the engine's per-timestep bar
    /// instead, so its overall bar would be a redundant `1/1`. `None` is the
    /// inert case (single cell, or `--progress none`/tests). Honours the
    /// `--progress` mode via [`crate::out::progress::Task`].
    pub(crate) progress: Option<crate::out::progress::Task>,
    /// The run id each planned cell resolved to before it ran, by `run_idx`.
    /// The cache-hit check has to ask before anything runs, so it resolves
    /// the cell from its `SimRun` over the base model (`should_run`); the
    /// leaf written in `merge_cell` is keyed from the cell the engine ran
    /// (`CellResult::cell`), and the two must agree — a cell whose pre-run
    /// and post-run identities differ is refused, not written.
    pub(crate) pre_run: HashMap<usize, runid::ContentHash>,
}

/// The overall cells bar for a multi-cell run, or `None` for a single cell
/// (the engine's per-timestep bar covers that) or when no bar applies. The
/// `Task` itself honours `--progress none`/`plain` internally.
pub(crate) fn cells_progress(total: usize, label: impl Into<String>) -> Option<crate::out::progress::Task> {
    (total > 1).then(|| crate::out::progress::Reporter::new().task(total as u64, label, "cells"))
}

impl CasSink {
    /// Advance the overall cells bar by one cell (committed or cache-hit) and
    /// finish it once the last cell lands. The bar replaces the old
    /// `[i/total] scenario=… seed=…` per-cell stderr lines; the per-cell
    /// detail is intentionally dropped in favour of a clean `pos/len` bar
    /// (drill into individuals with `camdl list --kind sim`). No-op when
    /// `progress` is `None` (single cell / `--progress none` / tests).
    fn advance_progress(&mut self) {
        if let Some(t) = self.progress.as_mut() {
            t.inc(1);
        }
        if self.counter >= self.total {
            if let Some(t) = self.progress.take() {
                t.finish();
            }
        }
    }

    /// The store root (`<output>`, the parent of `<output>/sims`).
    ///
    /// `pub(crate)` for `simulate --dry-run`'s store view (`main::print_store_view`).
    pub(crate) fn root(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.runs_dir)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// Resolve a cell's trajectory identity + factored store path before it
    /// runs: the cell resolved from its `SimRun` over the base model the
    /// `model` level is computed from, by the resolution the engine itself
    /// runs ([`crate::sim_config::ResolvedCell`]), and
    /// [`CasSink::cell_identity`] over it.
    ///
    /// This pre-run resolution is what a lookup uses when nothing has run: the
    /// cache-hit check (`should_run`, which records it in `pre_run`), a skipped
    /// cell's read-back and `--dry-run`. A cell that runs is keyed in
    /// `merge_cell` from the cell the engine ran, and refused if that differs
    /// from the recorded pre-run identity.
    ///
    /// `pub(crate)` for `simulate`'s read-back of a stored cell and for the
    /// `--dry-run` store view (`main::SimSink::read_stored`, `main::print_store_view`).
    pub(crate) fn cell_resolve(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<(crate::resolve::ResolvedTrajectory, std::path::PathBuf, String), String> {
        let (_, rt) = self.resolved_cell(spec)?;
        let root = self.root();
        let dir = runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels);
        let rel = dir.strip_prefix(&root).unwrap_or(&dir).to_string_lossy().into_owned();
        Ok((rt, dir, rel))
    }

    /// The cell resolved, and its identity built from it.
    pub(crate) fn resolved_cell(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<(crate::sim_config::ResolvedCell, crate::resolve::ResolvedTrajectory), String> {
        let cell = crate::sim_config::ResolvedCell::resolve(self.sim.model.ir.clone(), &spec.sim_run)?;
        let rt = self.cell_identity(spec, &cell)?;
        Ok((cell, rt))
    }

    /// The trajectory identity of one resolved cell. From `self.sim`: the
    /// invocation's `model` level, run-wide `config` and the cell's scenario
    /// arm (`self.sim.arms`, found by the spec's scenario name). From `cell`,
    /// the value the engine runs: the window, parameter values, table digests,
    /// origin state and seed.
    ///
    /// The destructure of `cell` names every field and binds none to `_`: a
    /// field added to [`crate::sim_config::ResolvedCell`] does not compile here
    /// until it is named, and naming it means using it or refusing on it. What
    /// the identity takes from `self.sim` rather than `cell` is checked against
    /// the cell: a cell whose backend, step, evaluator settings, sampler,
    /// required snapshot times or intervention set differ from what the run
    /// and its arm are keyed on is refused rather than keyed on a computation
    /// it did not make.
    fn cell_identity(
        &self,
        spec: &crate::engine::CellSpec,
        cell: &crate::sim_config::ResolvedCell,
    ) -> Result<crate::resolve::ResolvedTrajectory, String> {
        let crate::sim_config::ResolvedCell {
            run, tables, seed, init_state, backend, dt, binomial, eval, required_output_times,
        } = cell;
        let engine = &self.engine;
        if *backend != engine.backend || dt.to_bits() != engine.dt.to_bits() || *eval != engine.eval {
            return Err(format!(
                "internal error: a cell runs on backend {backend:?} at dt = {dt} with {eval:?}, \
                 but the run is keyed on backend {:?} at dt = {} with {:?}",
                engine.backend, engine.dt, engine.eval));
        }
        // Only the default sampler has an identity; another is refused on the
        // store-writing path before any cell is planned (`resolve_sim_config`).
        if *binomial != sim::rng::BinomialAlgorithm::Btpe {
            return Err(format!(
                "internal error: a cell runs the {binomial:?} binomial sampler, which has no run identity"));
        }
        // Extra snapshot times change the trajectory and have no identity
        // route (`fit predict` sets them, and its replay writes no leaf).
        if !required_output_times.is_empty() {
            return Err("internal error: a cell records snapshots beyond the model's schedule \
                        (`required_output_times`), which have no run identity".to_string());
        }

        let name = spec.scenario.name();
        let arm = self.sim.arms.iter()
            .find(|s| s.name == name || s.route.as_deref() == Some(name))
            .ok_or_else(|| format!("cell scenario '{}' not among the resolved scenarios", name))?;
        // The `scenario` level keys the arm's enable / disable lists; the
        // engine filtered the cell's interventions from its own scenario,
        // inline lists and replayed fit (`SimRun::scenario_name`,
        // `adhoc_enable`, `adhoc_disable`, `fit_base`). A cell running
        // another set of interventions than the arm describes is refused.
        let mut keyed = self.sim.model.ir.clone();
        crate::util::apply_scenario_filter(&mut keyed, &arm.enable, &arm.disable)?;
        let names = |m: &ir::Model| m.interventions.iter().map(|iv| iv.name.clone()).collect::<Vec<_>>();
        if names(&keyed) != names(&run.model) {
            return Err(format!(
                "internal error: the cell runs interventions {:?}, but its scenario '{}' is keyed \
                 as {:?}", names(&run.model), name, names(&keyed)));
        }

        // The `params` level is the cell's RESOLVED parameter values — every
        // tier (model default, `--params`, draw/sweep point, scenario,
        // `--param`, `--param-vec`) applied in the resolver's order, read off
        // the model the cell runs.
        let params: HashMap<String, f64> =
            crate::util::resolved_parameter_values(&run.model).into_iter().collect();

        let scenario_label = if arm.name.is_empty() {
            crate::sim_job::IMPLICIT_ARM_LABEL
        } else {
            arm.name.as_str()
        };
        // The label names the point's parameters at the values the cell RAN
        // (resolved), not the values the point asked for: a `--param` or a
        // scenario that shadows a draw column must not leave the path
        // advertising the draw's value.
        let ran_point: indexmap::IndexMap<String, f64> = spec.point_overrides.keys()
            .filter_map(|k| params.get(k).map(|v| (k.clone(), *v)))
            .collect();
        let param_label = params_path_label(&ran_point);

        let window = crate::sim_config::CellWindow::new(
            // The cell's EFFECTIVE start and horizon, read off the model it
            // runs: a run seeded from a filtered state begins at that state's
            // origin (gh#641), and a scenario's own `simulate { to }` or `--to`
            // moves the end (gh#561, gh#626). A different window is a different
            // trajectory, so it re-keys; a cell on the model's own window
            // hashes the base model's values.
            run.model.simulation.t_start,
            run.model.simulation.t_end,
            // gh#641 / gh#697: the state this cell restored. The origin
            // ENSEMBLE's content digest and the restored ROW both key the run —
            // a re-filtered / re-fitted ensemble under an unchanged model must
            // not serve the cached forecast, and two cells sharing a
            // `process_seed` (`--seeds 7,7`) must not collide while starting
            // from different rows. `None` for every cell without `--init-state`.
            init_state.as_ref().map(|i| runid::inputs::InitStateDigest {
                ensemble: runid::inputs::DataDigest(i.ensemble_digest),
                row: i.row,
            }),
        )
        .map_err(|e| format!("resolve error: {e}"))?;
        let ctx = crate::resolve::TrajectoryCtx {
            model: self.sim.model.digest.clone(),
            model_stem: &self.sim.model.stem,
            config: self.sim.run.cell_config(window),
            params: &params,
            table_digests: tables.clone(),
            enable: &arm.enable,
            disable: &arm.disable,
            scen_params: &arm.params,
            scen_scale: &arm.scale,
            param_label: &param_label,
            scenario_label,
            base_seed: *seed,
            process_seed: *seed,
        };
        let mut rt = crate::resolve::resolve_trajectory(&ctx).map_err(|e| format!("resolve error: {e}"))?;
        // Display-only, never hashed (the run_id is fixed above, from the
        // levels): each parameter's provenance, with every override of one
        // tier's value by another's, and the notices this cell's resolution
        // reported.
        if let Some(obj) = rt.display_inputs.as_object_mut() {
            let provenance = crate::run_meta::ParameterProvenance::of_all(&run.params);
            obj.insert("parameters_provenance".into(),
                serde_json::to_value(&provenance).map_err(|e| e.to_string())?);
            let notices = crate::params_resolver::notices(&[run]);
            obj.insert("notices".into(),
                serde_json::to_value(&notices).map_err(|e| e.to_string())?);
        }
        Ok(rt)
    }

    /// Where the store holds a committed leaf of this identity — found by its
    /// hashes, whatever labels its path carries ([`runid::find_leaf`]) — or,
    /// when it holds none, where a new one would land (`store_path` under the
    /// current labels). Returns the directory and its store-relative path.
    pub(crate) fn stored_leaf(
        &self,
        rt: &crate::resolve::ResolvedTrajectory,
    ) -> (std::path::PathBuf, String) {
        let root = self.root();
        let dir = match runid::find_leaf(&root, runid::ArtifactKind::Sim, &rt.levels) {
            Some(dir) => {
                crate::resolve::note_leaf_from_another_model_file(&dir, &self.sim.model.path);
                dir
            }
            None => runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels),
        };
        let rel = dir.strip_prefix(&root).unwrap_or(&dir).to_string_lossy().into_owned();
        (dir, rel)
    }
}

/// Human-readable provenance label for the `params` path level.
///
/// Sparse overrides — a scenario sweep point like `beta=0.3_gamma=0.1` — render
/// as a readable, key-sorted `k=v` join so `ls results/sims/` stays skimmable. A
/// *full* parameter vector (a `--draws` row over a stratified model) would blow a
/// single path component past NAME_MAX (gh#169), so it collapses to a short
/// `draws` tag. Identity is the level's content hash, never this label (see
/// `resolve.rs`: label and hash are separate inputs to `level()`), so the
/// collapse is lossless — the full resolved values live in the leaf's
/// `run.json.inputs.params` (`ResolvedTrajectory::display_inputs`).
fn params_path_label(overrides: &indexmap::IndexMap<String, f64>) -> String {
    if overrides.is_empty() {
        return "base".to_string();
    }
    let mut sorted: Vec<(&String, &f64)> = overrides.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let joined = sorted
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("_");
    // Readable when sparse; a full draw vector is too long for a path
    // component, so tag it `draws` (the level hash disambiguates the values).
    const LABEL_CAP: usize = 96;
    if joined.len() <= LABEL_CAP {
        joined
    } else {
        "draws".to_string()
    }
}

#[cfg(test)]
mod gh169_params_label {
    use super::params_path_label;
    use indexmap::IndexMap;

    fn im(pairs: &[(&str, f64)]) -> IndexMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn empty_overrides_is_base() {
        assert_eq!(params_path_label(&IndexMap::new()), "base");
    }

    #[test]
    fn sparse_scenario_overrides_stay_readable() {
        // a scenario sweep point: a couple of params -> readable, key-sorted join
        assert_eq!(
            params_path_label(&im(&[("gamma", 0.1), ("beta", 0.3)])),
            "beta=0.3_gamma=0.1"
        );
    }

    #[test]
    fn full_draw_vector_collapses_to_short_tag() {
        // a --draws row over a stratified model: the whole vector is overridden,
        // which previously rendered as one 600+ byte path component -> ENAMETOOLONG (gh#169).
        let overrides: IndexMap<String, f64> = (0..40)
            .map(|i| (format!("beta_age{}_patch{}", i % 2, i), 0.3125))
            .collect();
        let label = params_path_label(&overrides);
        assert_eq!(label, "draws", "a full draw vector must collapse to a short tag");
        // the rendered path component `{label}-{hash8}` must fit in NAME_MAX (255)
        assert!(label.len() + 1 + 8 <= 255);
    }
}

impl CasSink {
    /// Where the store already holds this cell: a `Completed`, integrity-checked
    /// leaf of the cell's identity — found by its hashes, whatever its labels
    /// — that holds every artifact this run asks for. `None` means the cell
    /// must run. Pure: nothing is displaced or written, so a dry run can ask
    /// the same question `should_run` asks.
    ///
    /// The trajectory alone is not a hit when observations were asked for:
    /// the obs subtree is a child written beside the trajectory, drawn with
    /// its own seed and never reconstructed from `traj.tsv`, so a leaf
    /// without it would be reported "cached" for observations it never wrote
    /// (the gh#738 / gh#599 class). The check is the EXACT subtree
    /// `obs/<hash8>-<obs_seed>/` — addressed from the scenario-applied model
    /// the cell runs (the resolved cell's model, the one `merge_cell` writes
    /// from), with every requested stream file and its
    /// `obs.json` present. A leaf holding another cadence's or seed's subtree
    /// is a miss, and directory presence alone is not a hit.
    pub(crate) fn stored_hit(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<Option<std::path::PathBuf>, String> {
        let (cell, rt) = self.resolved_cell(spec)?;
        self.stored_hit_of(spec, &cell, &rt)
    }

    /// [`CasSink::stored_hit`] for a cell already resolved.
    fn stored_hit_of(
        &self,
        spec: &crate::engine::CellSpec,
        cell: &crate::sim_config::ResolvedCell,
        rt: &crate::resolve::ResolvedTrajectory,
    ) -> Result<Option<std::path::PathBuf>, String> {
        // A build whose identity is incomplete serves no stored cell.
        if crate::resolve::engine_write_policy(crate::resolve::WritePolicy::Reuse, version::ENGINE_REUSE)
            == crate::resolve::WritePolicy::Force
        {
            return Ok(None);
        }
        let Some(dir) = runid::find_leaf(&self.root(), runid::ArtifactKind::Sim, &rt.levels) else {
            return Ok(None);
        };
        if self.obs_enabled && !self.sim.model.ir.observations.is_empty() {
            let model = &cell.run.model;
            let obs_dir = obs_subtree_dir(
                &dir, &model.observations, self.emit_every.as_ref(), spec.obs_seed,
            )?;
            if !obs_dir.join("obs.json").is_file() {
                return Ok(None);
            }
            for o in &model.observations {
                if !obs_dir.join(format!("{}.tsv", o.name)).is_file() {
                    return Ok(None);
                }
            }
        }
        crate::resolve::note_leaf_from_another_model_file(&dir, &self.sim.model.path);
        Ok(Some(dir))
    }
}

impl CasSink {
    /// Whether a stored cell is displaced (quarantined) and recomputed rather
    /// than reused: under `--force`, and always for a build whose identity is
    /// incomplete ([`crate::resolve::engine_write_policy`]).
    fn displaces(&self) -> bool {
        self.force
            || crate::resolve::engine_write_policy(crate::resolve::WritePolicy::Reuse, version::ENGINE_REUSE)
                == crate::resolve::WritePolicy::Force
    }
}

impl crate::engine::RunSink for CasSink {
    fn should_run(&mut self, spec: &crate::engine::CellSpec) -> bool {
        // The pre-run resolution (see `pre_run`): the only one a cache hit
        // ever has, and the one a cell that runs is checked against.
        let resolved = self.resolved_cell(spec);
        if let Ok((_, rt)) = &resolved {
            self.pre_run.insert(spec.run_idx, rt.run_id);
        }
        if self.displaces() {
            // Through the one policy seam: under Force the incumbent is
            // displaced (quarantined) HERE, so the recompute lands on a clean
            // leaf — previously force only skipped this check and the
            // recomputed bytes were discarded at commit as an
            // already-completed no-op, which is why `--force` could not change
            // a stored artifact. A failure surfaces in `merge_cell`.
            if let Ok((_, rt)) = &resolved {
                let root = self.root();
                let store = runid::FsCasStore::new(&root);
                let artifact = crate::resolve::ResolvedArtifact {
                    kind: runid::ArtifactKind::Sim,
                    levels: rt.levels.clone(),
                    run_id: rt.run_id,
                    display_inputs: rt.display_inputs.clone(),
                };
                let _ = crate::resolve::check_reuse(
                    &store, &root, &artifact, crate::resolve::WritePolicy::Force,
                );
            }
            return true;
        }
        // `true` means RUN: run unless the store holds everything this run
        // asks for. A failed resolve runs too; its error surfaces in
        // `merge_cell`.
        match &resolved {
            Ok((cell, rt)) => !matches!(self.stored_hit_of(spec, cell, rt), Ok(Some(_))),
            Err(_) => true,
        }
    }

    fn on_skip(&mut self, spec: &crate::engine::CellSpec) {
        // A cache hit still contributes to a multi-cell ensemble's deps, so
        // recover the leaf's run_id (from the resolve) and its traj.tsv digest
        // (from the existing leaf's run.json, recorded at first commit). The
        // leaf is read where the store holds it — found by hash, so a leaf
        // written under an earlier label is the one reported and read.
        let (rel, run_id, traj_digest, params_level) = match self.cell_resolve(spec) {
            Ok((rt, _, _)) => {
                let (dir, rel) = self.stored_leaf(&rt);
                let digest = read_traj_digest(&dir);
                // Keep an explicit `--label` current even on a pure cache-hit
                // skip (same reasoning as the commit path; no-op when None).
                if let Some(ref label) = self.label {
                    let _ = ensure_provenance_label(&dir, label);
                }
                (rel, Some(rt.run_id), digest, params_level_hash(&rt))
            }
            Err(_) => (String::new(), None, None, None),
        };
        let name = spec.scenario.name().to_string();
        self.counter += 1;
        self.advance_progress();
        self.completed_runs.push(RunEntry {
            run_path: rel,
            run_id,
            traj_digest,
            scenario: name,
            process_seed: spec.process_seed,
            point_id: spec.point_id,
            params_level,
        });
    }

    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let spec = &cell.spec;
        let name = spec.scenario.name().to_string();

        // The written leaf is keyed from the cell the engine ran, and must
        // agree with the identity the cache-hit check resolved before it ran.
        let identity = self.cell_identity(spec, &cell.cell).and_then(|rt| {
            match self.pre_run.get(&spec.run_idx) {
                Some(pre) if *pre != rt.run_id => Err(format!(
                    "internal error: the cell was looked up in the store as {} before it ran \
                     but resolves to {} from what ran; nothing was written",
                    pre.to_hex(), rt.run_id.to_hex())),
                _ => Ok(rt),
            }
        });
        let rt = match identity {
            Ok(rt) => rt,
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: {}", name, spec.process_seed, e));
                return Ok(());
            }
        };

        let cols = self.sim.run.view.cols(cell.model());
        let traj_bytes = crate::util::traj_tsv_bytes(&cell.traj, &cols);
        let traj_digest = runid::ContentHash::digest_bytes(&traj_bytes);

        // Declare the obs ensemble as a child sub-artifact so the leaf's
        // exact-set doesn't orphan the `obs/` subdir (written after the
        // trajectory commits). M2-interim: the obs child is recorded + its
        // streams written via the existing ensemble writer; a full
        // obs-child RunRecord identity is a follow-up.
        let mut children: std::collections::BTreeMap<String, Vec<runid::ContentHash>> =
            std::collections::BTreeMap::new();
        let has_obs = self.obs_enabled && !cell.model().observations.is_empty();
        if has_obs {
            let obs_seed = spec.process_seed ^ crate::util::SEED_MIX_OBS;
            // The SAME hash `write_obs_into_cas` names the subtree with — a
            // declared child that did not match the directory written beside it
            // would be provenance pointing at nothing.
            let obs_hash = match obs_subtree_hash(
                &cell.model().observations, self.emit_every.as_ref(),
            ) {
                Ok(h) => h,
                Err(e) => {
                    self.counter += 1;
                    self.errors.push(format!("scenario={} seed={}: {}",
                        name, spec.process_seed, e));
                    return Ok(());
                }
            };
            let obs_id = runid::ContentHash::digest_bytes(
                format!("{}:{}:{}", rt.run_id.to_hex(), obs_seed, obs_hash).as_bytes(),
            );
            children.insert("obs".to_string(), vec![obs_id]);
        }
        // `simulate --event-log`: the recorded event log is a child of the
        // leaf at its own address, as the obs subtree is, never one of the
        // leaf's own files — so the leaf's manifest is the same whether or not
        // any invocation asked for the log.
        let event_log_bytes = match cell.event_log.as_ref().map(sim::lineage::event_log_io::to_tsv_bytes) {
            None => None,
            Some(Ok(bytes)) => {
                children.insert(EVENT_LOG_CHILD.to_string(), vec![event_log_child_id(&rt.run_id)]);
                Some(bytes)
            }
            Some(Err(e)) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: event log serialize: {:?}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        };

        // Atomic write through the one resolved-writer seam (gh#241 PR D).
        // Identity (`Sim` / `levels` / `run_id`) is copied verbatim from the
        // resolved trajectory; the obs child + the fit dep ride in the record.
        let resolved_artifact = crate::resolve::ResolvedArtifact {
            kind: runid::ArtifactKind::Sim,
            levels: rt.levels.clone(),
            run_id: rt.run_id,
            // The resolved parameter values the params level hashed (and the
            // scenario and seed), so `camdl show` can say what this leaf ran.
            display_inputs: rt.display_inputs.clone(),
        };
        // The base model's axis: a scenario or sweep changes parameters, never
        // the time unit or the origin.
        let calendar = ir::caltime::CalendarAxis::from_model(&self.sim.model.ir)
            .map_err(|e| e.to_string())?;
        let mut meta = crate::resolve::RecordMeta::new(
            ir::IR_VERSION.trim(), self.sim.model.path.clone(), self.label.clone(), calendar,
            &self.engine.forcing)
            .with_deps(self.fit_dep.clone())
            .with_children(children.clone());
        // The engine identity the model level folded, so a leaf keyed by an
        // engine switch (`env::Environment::engine_identity`) says which.
        meta.engine = self.sim.model.engine.clone();

        let mut artifacts = runid::Artifacts::new();
        artifacts.insert("traj.tsv", traj_bytes);
        // Kept alongside the staged set so they can be re-added to an
        // ALREADY-COMPLETED leaf below: a cache hit discards the staged
        // artifacts wholesale.
        let mut optional_artifacts: Vec<(&'static str, Vec<u8>)> = Vec::new();
        // gh#204: a run with an active reactive policy declares its firing log
        // as a first-class artifact in THIS leaf alongside `traj.tsv` — present
        // even with zero firings (`Some` carries that), absent when the model
        // has no active reactive policy. The agenda's realized-obs draws run on
        // a dedicated RNG salt off the run seed, so the trajectory bytes and the
        // run_id are unchanged: a leaf a plain `simulate` would write simply
        // gains one more declared artifact, never an optional-on-cache-hit one.
        if let Some(ref firings) = cell.traj.reactive_log {
            let bytes = sim::reactive::format_reactive_log(firings).into_bytes();
            optional_artifacts.push(("reactive_log.tsv", bytes.clone()));
            artifacts.insert("reactive_log.tsv", bytes);
        }
        // Declare the tabular outputs' column schema in run.json — classify the
        // in-memory artifact headers (they're committed atomically, not yet on
        // disk). Recorded, not hashed.
        meta.output_schema = crate::output_schema::sim_output_schema(&artifacts.files);
        let root = self.root();
        let store = runid::FsCasStore::new(&root);
        // The commit half of the force policy. `should_run` covers callers
        // that skip cached cells, but `simulate` deliberately runs every cell
        // (its `-o` mirror needs them all) and reaches commit with a cache hit
        // in place — where the staged bytes were silently discarded, so
        // `--force` could never change a stored artifact. Displacing here
        // (quarantine, not delete) gives every commit path the same one
        // meaning of force, and is a no-op when nothing is stored.
        if self.displaces() {
            if let Err(e) = crate::resolve::check_reuse(
                &store, &root, &resolved_artifact, crate::resolve::WritePolicy::Force,
            ) {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: force displace failed: {}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        }
        let dest = match crate::resolve::begin_resolved_write(
            &store, &root, &resolved_artifact, &meta,
            crate::resolve::WriteMode::Atomic(artifacts),
        ) {
            Ok(crate::resolve::ResolvedWrite::Committed(d)) => d,
            Ok(crate::resolve::ResolvedWrite::Streaming(_)) => {
                unreachable!("Atomic write mode never returns a streaming claim")
            }
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: commit failed: {}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        };

        // A cache hit at commit returns the INCUMBENT leaf and discards
        // everything we staged, so an optional artifact this run recorded but
        // the stored leaf lacks would simply vanish. `augment` adds it under
        // the leaf's lock; it is a no-op when the bytes already match, which is
        // the fresh-commit case, and it refuses divergent bytes for the same
        // reason the commit path does.
        for (artifact, bytes) in &optional_artifacts {
            if let Err(e) = store.augment(
                &dest, &runid::LeafIdentity::new(rt.run_id), artifact, bytes)
            {
                self.errors.push(format!(
                    "scenario={} seed={}: recording {} into {}: {}",
                    name, spec.process_seed, artifact, dest.display(), e));
            }
        }

        // The event-log child: declared in the leaf's record first (a fresh
        // commit already declares it; a leaf the store already held gains the
        // declaration, which is recorded and never hashed), then written. In
        // that order a crash leaves a declared child with no record, which
        // the next `--event-log` run writes, never an undeclared directory,
        // which the exact-set check would read as an orphan and stale the
        // leaf. A failure is reported and fails the command, since the log
        // is what `--event-log` was asked for.
        if let Some(bytes) = &event_log_bytes {
            let written = EVENT_LOG.declare_and_write(&dest, &rt.run_id, bytes);
            if let Err(e) = written {
                self.errors.push(format!("scenario={} seed={}: event log: {}",
                    name, spec.process_seed, e));
            }
        }

        // Obs ensemble: written into the committed leaf's `obs/` child. A
        // failure here is non-fatal — children are independent, so a missing
        // obs child never staleens the parent trajectory.
        if has_obs {
            // The commit above declared this subtree as a child only on a
            // FRESH leaf; a leaf the store already held (a dedup at commit, or
            // one first written without observations) keeps its stored record.
            // An `obs/` directory a record does not declare is an orphan to the
            // exact-set check, which made the leaf `Stale`, and the next
            // recompute reclaimed the directory — taking every earlier
            // cadence's subtree with it. `children` is recorded, never hashed,
            // so declaring the subtree in place changes no identity.
            if let Some(ids) = children.get("obs") {
                if let Err(e) = ensure_children_declared(&dest, "obs", ids) {
                    self.errors.push(format!("scenario={} seed={}: declaring obs child: {}",
                        name, spec.process_seed, e));
                }
            }
            // gh#641: a restarted cell's obs axis begins at the forecast
            // origin, exactly as the `-o`/`--obs` mirror's does. `None` for
            // every ordinary cell.
            let restart_origin = spec.sim_run.init_state.as_ref().map(|i| i.origin_t);
            if let Err(e) = write_obs_into_cas(
                &dest, cell.model(), &cell.traj, spec.process_seed, restart_origin,
                self.emit_every.as_ref(), spec.sim_run.eval,
            ) {
                self.errors.push(format!("scenario={} seed={}: obs ensemble: {}",
                    name, spec.process_seed, e));
            }
        }

        // A `--label` must land on the leaf even when `commit_atomic` resolved
        // to an existing (cache-hit) leaf whose `run.json` carried a different
        // or absent label — the content-addressed dedup must not silently drop
        // the user's explicit label. `provenance` is metadata, not identity, so
        // this never changes the run_id (same mechanism as `camdl label`). A
        // no-op on a fresh commit, whose record already carries `self.label`.
        if let Some(ref label) = self.label {
            if let Err(e) = ensure_provenance_label(&dest, label) {
                self.errors.push(format!("scenario={} seed={}: label update: {}",
                    name, spec.process_seed, e));
            }
        }

        // The path the leaf actually landed at: `store_path` for a new leaf,
        // or the leaf the store already held for this identity under an
        // earlier label (`begin_resolved_write` finds it by hash).
        let rel = dest.strip_prefix(&root).unwrap_or(&dest).to_string_lossy().into_owned();
        self.counter += 1;
        self.advance_progress();
        self.completed_runs.push(RunEntry {
            run_path: rel,
            run_id: Some(rt.run_id),
            traj_digest: Some(traj_digest),
            scenario: name,
            process_seed: spec.process_seed,
            point_id: spec.point_id,
            params_level: params_level_hash(&rt),
        });
        Ok(())
    }
}

/// The hash of a resolved cell's `params` level.
fn params_level_hash(rt: &crate::resolve::ResolvedTrajectory) -> Option<runid::ContentHash> {
    rt.levels.iter().find(|l| l.name == "params").map(|l| l.hash)
}

/// Read a committed leaf's `traj.tsv` SHA-256 from its `run.json` artifact
/// manifest (recorded at commit). Used by the cache-hit path (`on_skip`) to
/// recover the dep digest without re-reading the trajectory bytes.
/// `None` when the record cannot be read or records no `traj.tsv`.
fn read_traj_digest(dir: &std::path::Path) -> Option<runid::ContentHash> {
    let leaf = runid::read_leaf(dir).ok()??;
    leaf.record.artifacts.get("traj.tsv").map(|c| c.digest)
}

/// Rewrite a leaf's `run.json` through `edit`, which returns whether it changed
/// the record: the store's one record-update path ([`runid::update_record`]),
/// under the leaf's `.lock` and through the reserved `run.json.tmp`, so no two
/// updates — and no update and an `augment` — interleave, and a crash leaves
/// only files the store's exact-set check ignores (gh#896).
fn rewrite_record(
    leaf_dir: &std::path::Path,
    edit: impl FnOnce(&mut runid::RunRecord) -> Result<bool, String>,
) -> Result<(), String> {
    runid::update_record(leaf_dir, edit).map_err(|e| match e {
        runid::CasError::EditRefused { reason, .. } => reason,
        other => format!("update {}: {other}", leaf_dir.join("run.json").display()),
    })
}

/// Ensure a leaf's `run.json` carries `provenance.label == Some(label)`,
/// rewriting it only when it differs. `provenance` is metadata, not identity —
/// the run_id and exact-set are untouched — applied at write time so an
/// explicit `simulate --label` survives a CAS cache hit, which relabels a sim.
/// Idempotent: a no-op when the leaf already has the requested label. Fit
/// runs use [`set_fit_label`], which also stamps the time.
pub(crate) fn ensure_provenance_label(leaf_dir: &std::path::Path, label: &str) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        if rec.provenance.label.as_deref() == Some(label) {
            return Ok(false);
        }
        rec.provenance.label = Some(label.to_string());
        Ok(true)
    })
}

/// Give a fit run `label`, stamped `label_set_at`: the run becomes the
/// label's holder, since a fit label names the completed run carrying it with
/// the latest stamp. A label the run already carries is re-stamped; another
/// is replaced, and the previous one is returned.
///
/// The one way a fit run's label is written. It changes only the two
/// provenance fields, never anything identity-bearing, and goes through the
/// store's record-update path (the leaf lock, `run.json.tmp`, fsync, rename).
/// The caller holds [`fit_label_lock`] and has checked that no live run
/// carries the label.
pub(crate) fn set_fit_label(
    leaf_dir: &std::path::Path,
    label: &str,
    label_set_at: &str,
) -> Result<Option<String>, String> {
    let mut previous = None;
    rewrite_record(leaf_dir, |rec| {
        previous = rec.provenance.label.clone().filter(|l| l != label);
        rec.provenance.label = Some(label.to_string());
        rec.provenance.label_set_at = Some(label_set_at.to_string());
        Ok(true)
    })?;
    Ok(previous)
}

/// Now, as a `label_set_at` stamp: RFC 3339 UTC with millisecond resolution.
pub(crate) fn label_stamp() -> String {
    crate::cas::iso8601_utc_millis(std::time::SystemTime::now())
}

/// The store-wide lock under which a fit label is checked for uniqueness and
/// written (gh#896): an `flock(2)` exclusive lock on `<root>/.fit-labels.lock`
/// (beside the store's kind directories, not inside `fits/`, which holds only
/// fit folders),
/// so two writers cannot both find a label free and both take it. The kernel
/// releases it when the holder drops it, exits or crashes, so there is no stale
/// lock to reclaim and no pid to trust; the file stays in the store and holds
/// `pid <n> on <host>` of the latest holder, for the refusal message only.
// Off Unix the lock is refused, so nothing constructs one.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) struct LabelLock {
    _file: std::fs::File,
}

impl std::fmt::Debug for LabelLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LabelLock")
    }
}

/// How long [`fit_label_lock`] waits for another holder before refusing.
const LABEL_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

pub(crate) fn fit_label_lock(root: &std::path::Path) -> Result<LabelLock, String> {
    fit_label_lock_within(root, LABEL_LOCK_WAIT)
}

#[cfg(unix)]
fn fit_label_lock_within(
    root: &std::path::Path,
    wait: std::time::Duration,
) -> Result<LabelLock, String> {
    use std::io::{Seek, Write};
    use std::os::unix::io::AsRawFd;
    std::fs::create_dir_all(root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
    let path = root.join(".fit-labels.lock");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let deadline = std::time::Instant::now() + wait;
    loop {
        // SAFETY: flock on an fd we own; LOCK_NB makes it non-blocking.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            let _ = file.set_len(0);
            let _ = file.rewind();
            let _ = write!(file, "pid {} on {}", std::process::id(), hostname());
            return Ok(LabelLock { _file: file });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(format!("cannot lock {}: {err}", path.display()));
        }
        if std::time::Instant::now() >= deadline {
            let recorded = std::fs::read_to_string(&path).unwrap_or_default();
            let recorded = recorded.trim();
            return Err(format!(
                "the store's label lock {} is held by another process (the latest holder \
                 recorded {}); a fit label is being written. Retry when it has finished.",
                path.display(),
                if recorded.is_empty() { "nothing" } else { recorded }
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Off Unix there is no `flock(2)`, and writing a label without the lock would
/// let two writers both find it free and both take it: refused, not skipped.
#[cfg(not(unix))]
fn fit_label_lock_within(
    root: &std::path::Path,
    _wait: std::time::Duration,
) -> Result<LabelLock, String> {
    Err(format!(
        "cannot take the store's label lock {}: fit label locking is unsupported on \
         this platform (it needs flock(2), available only on Unix), and a label is \
         not written without it",
        root.join(".fit-labels.lock").display()
    ))
}

/// This machine's hostname, for the label lock's holder note.
#[cfg(unix)]
fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes into buf.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return "unknown host".to_string();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Ensure a leaf's `run.json` lists `value` in its `inputs[key]` array,
/// rewriting it only when it is missing. `inputs` is the recorded-not-hashed
/// display projection, so the run_id and the exact-set are untouched. Used for
/// the config identities a fit run was reached from (gh#896): a cache hit from
/// a second config with the same run identity adds that config's.
pub(crate) fn ensure_input_listed(
    leaf_dir: &std::path::Path,
    key: &str,
    value: &str,
) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        let obj = rec
            .inputs
            .as_object_mut()
            .ok_or_else(|| "`inputs` is not an object".to_string())?;
        let list = obj.entry(key.to_string()).or_insert_with(|| serde_json::json!([]));
        let arr = list
            .as_array_mut()
            .ok_or_else(|| format!("`inputs.{key}` is not a list"))?;
        if arr.iter().any(|v| v.as_str() == Some(value)) {
            return Ok(false);
        }
        arr.push(serde_json::Value::String(value.to_string()));
        Ok(true)
    })
}

/// Ensure a leaf's `run.json` declares every id in `ids` under its `child`
/// (`obs`, [`EVENT_LOG_CHILD`]), rewriting it atomically (tmp + rename) only
/// when one is missing. `children` is metadata, not identity — the run_id and
/// exact-set of the leaf's own files are untouched — so this is the same
/// in-place record update `ensure_provenance_label` performs. Idempotent: a
/// no-op when the record already declares them (the fresh-commit case).
fn ensure_children_declared(
    leaf_dir: &std::path::Path,
    child: &str,
    ids: &[runid::ContentHash],
) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        let declared = rec.children.entry(child.to_string()).or_default();
        let missing: Vec<runid::ContentHash> =
            ids.iter().filter(|id| !declared.contains(id)).copied().collect();
        if missing.is_empty() {
            return Ok(false);
        }
        declared.extend(missing);
        Ok(true)
    })
}

// ─── A child holding one file and its record ───────────────────────────────

/// A child of a leaf that holds one file and a record of it:
/// `<leaf>/<name>/<file>` and `<leaf>/<name>/<record>` (the parent run id and
/// the file's size and SHA-256, written after the file, so its presence says
/// the child is complete), declared in the leaf's `run.json` `children` under
/// `name` with the id SHA256("{run_id}:{name}"). The address is fixed: such a
/// child is derived from the leaf's result alone, so one run id has one of
/// it, and what varies by invocation is only whether it was asked for. That
/// is why it is a child and never one of the leaf's own files. The event log
/// ([`EVENT_LOG`]) and a survey's rendered landscape ([`LANDSCAPE_HTML`]) are
/// two.
pub(crate) struct RecordedChild {
    /// The name under `children`, and the child's directory.
    pub name: &'static str,
    /// The file the child holds.
    pub file: &'static str,
    /// The child's record.
    pub record: &'static str,
    /// What the file is, for messages ("event log").
    pub what: &'static str,
}

/// `simulate --event-log`'s child: `<leaf>/event_log/event_log.tsv`.
pub(crate) const EVENT_LOG: RecordedChild = RecordedChild {
    name: "event_log",
    file: "event_log.tsv",
    record: "event_log.json",
    what: "event log",
};

/// `survey --render`'s child: `<leaf>/landscape_html/landscape.html`, the
/// interactive pair-plot rendered from the leaf's `landscape.tsv` and its
/// identity (the HTML names the run's short hash), and from nothing else.
pub(crate) const LANDSCAPE_HTML: RecordedChild = RecordedChild {
    name: "landscape_html",
    file: "landscape.html",
    record: "landscape_html.json",
    what: "rendered landscape",
};

impl RecordedChild {
    /// Where the child's file is in `leaf`.
    pub(crate) fn file_in(&self, leaf: &std::path::Path) -> std::path::PathBuf {
        leaf.join(self.name).join(self.file)
    }

    /// The id a leaf's record declares for this child.
    fn id(&self, run_id: &runid::ContentHash) -> runid::ContentHash {
        runid::ContentHash::digest_bytes(format!("{}:{}", run_id.to_hex(), self.name).as_bytes())
    }

    /// Declare the child in `leaf`'s record, then write `bytes` into it. In
    /// that order a crash leaves a declared child with no record, which the
    /// next request writes, never an undeclared directory, which the exact-set
    /// check would read as an orphan and stale the leaf. `Ok(false)` when the
    /// child already held these bytes and nothing was written.
    pub(crate) fn declare_and_write(
        &self,
        leaf: &std::path::Path,
        run_id: &runid::ContentHash,
        bytes: &[u8],
    ) -> Result<bool, String> {
        ensure_children_declared(leaf, self.name, &[self.id(run_id)])?;
        self.write(leaf, run_id, bytes)
    }

    /// Write `bytes` into `leaf`'s child, unless the child already holds them.
    ///
    /// A complete child (its record present, naming these bytes, and the file
    /// holding them) is left as it is. A record naming other bytes is refused
    /// and nothing is overwritten: one run id has one such file, so different
    /// bytes under it are an identity defect, the same rule the store's commit
    /// path applies. Anything else — no child, or a crash between the file and
    /// its record — is (re)written, each file atomically
    /// ([`runid::write_file_atomic`]).
    fn write(&self, leaf: &std::path::Path, run_id: &runid::ContentHash, bytes: &[u8]) -> Result<bool, String> {
        let RecordedChild { name, file, record, what } = *self;
        let dir = leaf.join(name);
        let file_path = dir.join(file);
        let record_path = dir.join(record);
        let digest = runid::ContentHash::digest_bytes(bytes).to_hex().to_string();
        match std::fs::read(&record_path) {
            Ok(text) => {
                let rec: serde_json::Value = serde_json::from_slice(&text).map_err(|e| {
                    format!("the {what} record {} does not parse ({e}); `--force` recomputes the leaf",
                        record_path.display())
                })?;
                let stored = rec["digest"].as_str().unwrap_or_default();
                if stored != digest {
                    return Err(format!(
                        "the stored {what} {} has SHA-256 {stored}, and this run made {digest} \
                         under the same run id {}; nothing was overwritten. One run id has one \
                         {what}, so this is an identity defect: please report it. `--force` \
                         recomputes the leaf.",
                        file_path.display(), run_id.to_hex()));
                }
                if std::fs::read(&file_path).is_ok_and(|on_disk| on_disk == bytes) {
                    return Ok(false);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot read {}: {e}", record_path.display())),
        }
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let rec = serde_json::json!({
            "parent_run_id": run_id.to_hex().to_string(),
            "file": file,
            "bytes": bytes.len(),
            "digest": digest,
        });
        let record_bytes = serde_json::to_vec_pretty(&rec)
            .map_err(|e| format!("cannot serialize the {what} record: {e}"))?;
        for (path, bytes) in [(&file_path, bytes), (&record_path, record_bytes.as_slice())] {
            runid::write_file_atomic(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        Ok(true)
    }
}

// ─── Event-log child (`simulate --event-log`) ──────────────────────────────

/// The name under which a leaf declares its event-log child in `run.json`'s
/// `children`, and the directory that holds it: `<leaf>/event_log/`, with
/// [`EVENT_LOG_FILE`] and its record [`EVENT_LOG_RECORD`].
///
/// A fixed name, because nothing the log depends on lies outside the leaf's
/// run id. The recorder draws no randomness; it reads the compiled model (its
/// `#[lineage]` annotations included), the parameters, the seed, the backend,
/// its step and its sampler, all of which the leaf's levels key, and the
/// recorder and its file format are the engine's, whose version the `model`
/// level folds. One run id has one event log; what varies by invocation is
/// only whether it was asked for, which is why it is a child rather than one
/// of the leaf's own files.
pub(crate) const EVENT_LOG_CHILD: &str = EVENT_LOG.name;

/// The event log's file in the child: the TSV `lineage realize` reads.
pub(crate) const EVENT_LOG_FILE: &str = EVENT_LOG.file;

/// The child's record: the parent run id and the log's size and SHA-256. It
/// is written after the log, so its presence says the child is complete.
const EVENT_LOG_RECORD: &str = EVENT_LOG.record;

/// Where a leaf's event log is: `<leaf>/event_log/event_log.tsv`. One function
/// for the writer and every reader (`simulate`'s report, `camdl cat`).
pub(crate) fn event_log_child_file(leaf: &std::path::Path) -> std::path::PathBuf {
    EVENT_LOG.file_in(leaf)
}

/// The leaf's event log, when its child is complete: the record present and
/// the file the size and SHA-256 the record names. The one check every reader
/// makes (`camdl cat --stream event_log.tsv`, `simulate`'s `lineage realize`
/// hint), so no reader hands on a log the writer did not finish. The error
/// says what is wrong and how to write the log.
pub(crate) fn complete_event_log(leaf: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let file = event_log_child_file(leaf);
    let record_path = leaf.join(EVENT_LOG_CHILD).join(EVENT_LOG_RECORD);
    let remedy = "re-run the same `camdl simulate` with `--event-log` to write it, or with \
                  `--force` to recompute the leaf and its children";
    let text = match std::fs::read(&record_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(if file.exists() {
                format!("the event log {} has no record ({EVENT_LOG_RECORD}), so its write did not \
                         finish; {remedy}", file.display())
            } else {
                format!("the run {} has no event log; {remedy}", leaf.display())
            });
        }
        Err(e) => return Err(format!("cannot read {}: {e}", record_path.display())),
    };
    let rec: serde_json::Value = serde_json::from_slice(&text)
        .map_err(|e| format!("the event-log record {} does not parse ({e}); {remedy}", record_path.display()))?;
    let bytes = std::fs::read(&file)
        .map_err(|e| format!("cannot read the event log {}: {e}; {remedy}", file.display()))?;
    let digest = runid::ContentHash::digest_bytes(&bytes).to_hex().to_string();
    if rec["bytes"].as_u64() != Some(bytes.len() as u64) || rec["digest"].as_str() != Some(digest.as_str()) {
        return Err(format!(
            "the event log {} is {} bytes with SHA-256 {digest}, and its record names {} bytes \
             with {}; {remedy}",
            file.display(), bytes.len(), rec["bytes"], rec["digest"]));
    }
    Ok(file)
}

/// The id a leaf's record declares for its event-log child.
fn event_log_child_id(run_id: &runid::ContentHash) -> runid::ContentHash {
    EVENT_LOG.id(run_id)
}

/// Write the event log into `leaf`'s child, unless the child already holds it
/// ([`RecordedChild::write`]).
#[cfg(test)]
fn write_event_log_child(
    leaf: &std::path::Path,
    run_id: &runid::ContentHash,
    bytes: &[u8],
) -> Result<(), String> {
    EVENT_LOG.write(leaf, run_id, bytes).map(|_| ())
}

// ─── Observation ensemble writer (CLI review #4) ───────────────────────────

/// Sample synthetic observations for one completed run and write them into
/// the CAS obs subtree: `run_dir/obs/{obs_hash[:8]}-{obs_seed}/<stream>.tsv`
/// plus an `obs.json` provenance file (the layout designed in
/// `cas/mod.rs`).
///
/// Reuses the *same* sampling primitives as `simulate --obs`
/// (`compile_obs_sample_pf`, `project_all_obs_times`, `obs_schedule_times`,
/// `snap_at`) and the canonical obs RNG derivation
/// (`process_seed ^ SEED_MIX_OBS`), so a given seed produces the same
/// observation bytes regardless of which entry point generated them. One
/// file per stream means multi-cadence streams are handled correctly with
/// no single-file kludge.
///
/// `process_seed` is the run's simulation seed; in this single-realization
/// path it *is* the `obs_seed` recorded in the directory name (run-spec
/// §"Gating risk": a future `[obs] replicates = K` fans the obs layer by
/// mixing K distinct obs seeds, leaving the trajectory RNG untouched).
/// The obs subtree's identity component.
///
/// A hash of the resolved observation blocks (run-spec: changing a reporting
/// parameter re-samples obs without invalidating the cached trajectory), plus
/// any `--emit-every` override (gh#656) — which changes WHICH times the subtree
/// carries, so two cadences must not address the same directory (the
/// count-in-the-key rule).
///
/// One function, because the hash is computed twice for every cell — once to
/// declare the `obs` child in `run.json`, once to name the subtree directory —
/// and a disagreement between the two would declare one child while writing
/// another. Byte-identical to the historical `sha256(json(observations))`
/// whenever no override is in play, so an unflagged run addresses exactly the
/// subtree it always did.
pub(crate) fn obs_subtree_hash(
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    let mut buf = serde_json::to_string(observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    if let Some(e) = emit {
        buf.push_str(&e.identity_repr());
    }
    // The file FORMAT is part of what the subtree holds (gh#833): the header
    // names the declared columns, a windowed stream carries two boundaries,
    // and a row outside the run is not written. Bytes that change under an
    // unchanged key would be a stored artifact the key no longer describes,
    // so the format version enters the key.
    buf.push_str(OBS_FILE_FORMAT);
    Ok(crate::hashing::sha256_hex(buf.as_bytes()))
}

/// The obs-file format the CAS subtree is written in — bumped whenever the
/// bytes written for an unchanged model would change (the count-in-the-key
/// rule).
///
/// 3 (D59): a row's value is drawn from the state, and with the time, where
/// the row is scored (its instant, or its period's stop), not at its label.
/// For a stream whose label is not its scoring time (`day`, `starting_on`,
/// `ending_on`) whose likelihood reads the state or `t`, the stored draws
/// change, so a subtree written before must not answer for one written now.
const OBS_FILE_FORMAT: &str = "\nobs-file-format:3";

/// The directory a leaf's observation subtree lives in:
/// `<leaf>/obs/<hash8>-<obs_seed>/`, the `hash8` being the first eight hex
/// of [`obs_subtree_hash`] over the scenario-applied model's observation
/// blocks. One function for the writer and for the store-hit check
/// ([`CasSink::stored_hit`]), so the address asked about is the address
/// written.
pub(crate) fn obs_subtree_dir(
    leaf: &std::path::Path,
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
    obs_seed: u64,
) -> Result<std::path::PathBuf, String> {
    let obs_hash = obs_subtree_hash(observations, emit)?;
    Ok(leaf.join("obs").join(format!("{}-{}", &obs_hash[..8.min(obs_hash.len())], obs_seed)))
}

fn write_obs_into_cas(
    run_dir: &std::path::Path,
    model: &ir::Model,
    traj: &sim::Trajectory,
    process_seed: u64,
    // gh#641: `Some(T)` when this cell was restarted from a filtered state at
    // `T`; the obs axis then begins there, because the trajectory does. `None`
    // for every ordinary cell.
    restart_origin: Option<f64>,
    // gh#656: the `--emit-every` override — it sets the emitted times AND names
    // the subtree (via `obs_subtree_hash`), so the two cadences coexist under one
    // trajectory leaf instead of one overwriting the other.
    emit: Option<&crate::emit_every::EmitEvery>,
    // The evaluator setting the cell ran under (`SimRun::eval`): the
    // observation model is compiled under it, as the cell's rates were.
    eval: sim::EvalSettings,
) -> Result<(), String> {
    if model.observations.is_empty() {
        return Ok(());
    }

    let obs_hash = obs_subtree_hash(&model.observations, emit)?;
    let obs_seed = process_seed ^ crate::util::SEED_MIX_OBS;
    let obs_dir = obs_subtree_dir(run_dir, &model.observations, emit, obs_seed)?;

    // Preflight EVERY stream before creating any file (gh#589 review).
    //
    // The per-stream loop below validates and writes in the same iteration, so
    // an aligned stream followed by a misaligned one would leave the first
    // stream's `.tsv` on disk while the error aborts before `obs.json` is
    // written — a child directory that exists, is incomplete, and carries no
    // provenance to say so. Validation is cheap and pure, so hoisting it turns
    // a partial write into no write.
    // The rows each stream's declaration assigns to its emit times, within the
    // run (gh#833) — planned once, so the preflight and the write loop cannot
    // disagree about which rows exist.
    let run_start = crate::run_start_of(traj, model);
    let mut plans = Vec::with_capacity(model.observations.len());
    for obs_ir in &model.observations {
        // gh#829: the draws below pass no data columns. Both callers refuse
        // such a stream before any cell runs; this keeps the writer honest
        // for any caller that does not.
        crate::obs_emit::check_data_columns_supplied(
            obs_ir, crate::obs_emit::ColumnSupply::Nothing,
        )?;
        // Must use the SAME horizon the write loop below uses — a preflight that
        // validates a different set of times than gets written is worse than no
        // preflight (gh#561 + gh#589).
        let times =
            crate::obs_emit_schedule_times(obs_ir, restart_origin, model.simulation.t_end, emit)?;
        let plan = crate::obs_emit::plan_emission(
            obs_ir, &times, run_start, model.simulation.t_end,
            emit.and_then(|e| e.resolve_for(&obs_ir.source)),
        )?;
        crate::project_coverages(traj, obs_ir, model, &plan.scored(), eval)?;
        plans.push(plan);
    }

    std::fs::create_dir_all(&obs_dir)
        .map_err(|e| format!("cannot create {}: {}", obs_dir.display(), e))?;

    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model.clone())
            .map_err(|e| format!("model compile error for obs: {:?}", e))?
            .with_eval_settings(eval),
    );
    let params = compiled.default_params.clone();
    // One obs RNG, consumed in declaration order across streams — the same
    // order simulate's --obs loop uses (main.rs).
    let mut obs_rng = sim::rng::StatefulRng::new(obs_seed);

    let mut stream_names: Vec<String> = Vec::new();
    for (obs_ir, plan) in model.observations.iter().zip(&plans) {
        let sampler = sim::inference::obs_model::compile_obs_sample_pf(
            obs_ir, compiled.clone(), &params,
        ).map_err(|e| e.to_string())?;
        // The cell's own horizon (a per-scenario `simulate { to }` has already
        // moved `model.simulation.t_end`) bounded the plan, so the CAS `obs/`
        // subtree never carries rows past the end of the trajectory beside it
        // (gh#561).
        let rows = plan.scored();
        let projected = crate::project_coverages(traj, obs_ir, model, &rows, eval)?;

        // Each row's state and time are read where it is scored.
        let draws: Vec<f64> = rows.iter().enumerate()
            .map(|(ti, &(at, _))| {
                let snap = crate::snap_at(traj, at);
                sampler(projected[ti], at, &snap.int_state.counts, &[], &mut obs_rng)
            })
            .collect();
        // The declared columns, so the file re-loads under its own model
        // (gh#833, gh#830) — the same writer every simulated dataset uses.
        let path = obs_dir.join(format!("{}.tsv", obs_ir.name));
        // No extra columns: this writer draws every stream with an empty aux
        // slice, so a ratio stream's data-column `n` is not written here (the
        // design-preserving emitter is where it is, proposal 2026-09-09).
        crate::obs_emit::write_stream_file(&path, plan, &draws, &[])?;
        stream_names.push(obs_ir.name.clone());
    }

    // obs.json provenance: the inputs that produced this obs draw.
    let obs_meta = serde_json::json!({
        "obs_hash": obs_hash,
        "obs_seed": obs_seed,
        "process_seed": process_seed,
        "streams": stream_names,
        "version": version::VERSION_SHORT,
    });
    std::fs::write(
        obs_dir.join("obs.json"),
        serde_json::to_string_pretty(&obs_meta).unwrap_or_default(),
    )
    .map_err(|e| format!("cannot write obs.json: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// The event-log child is written once and then left alone: the same
    /// bytes again are a no-op, other bytes under the same run id are refused
    /// with nothing overwritten, and a log left without its record (a crash
    /// between the two writes) is rewritten.
    #[test]
    fn the_event_log_child_is_written_once_and_never_overwritten() {
        let leaf = crate::test_support::unique_temp_dir("event_log_child");
        let run_id = runid::ContentHash::digest_bytes(b"run");
        let file = event_log_child_file(&leaf);
        let record = leaf.join(EVENT_LOG_CHILD).join(EVENT_LOG_RECORD);

        write_event_log_child(&leaf, &run_id, b"first\n").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"first\n");
        let rec: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
        assert_eq!(rec["digest"].as_str().unwrap(),
            runid::ContentHash::digest_bytes(b"first\n").to_hex().to_string());
        assert_eq!(rec["parent_run_id"].as_str().unwrap(), run_id.to_hex().to_string());

        write_event_log_child(&leaf, &run_id, b"first\n").unwrap();
        let err = write_event_log_child(&leaf, &run_id, b"second\n").unwrap_err();
        assert!(err.contains("identity defect"), "{err}");
        assert_eq!(std::fs::read(&file).unwrap(), b"first\n", "nothing was overwritten");

        std::fs::remove_file(&record).unwrap();
        std::fs::write(&file, b"torn").unwrap();
        write_event_log_child(&leaf, &run_id, b"first\n").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"first\n", "an incomplete child is rewritten");
        assert!(record.is_file());
    }

    /// A leaf directory holding a minimal fit-method `run.json` with `label`.
    fn record_leaf(tag: &str, label: Option<&str>) -> std::path::PathBuf {
        let leaf = crate::test_support::unique_temp_dir(tag).join("if2-00000000").join("seed_1-00000000");
        std::fs::create_dir_all(&leaf).unwrap();
        let mut rec = serde_json::json!({
            "format_version": 1, "kind": "fit_method", "run_id": "ab".repeat(32),
            "hash_version": 1, "calendar": {"days_per_unit": 1.0, "origin": null, "time_unit": "days"}, "ir_version": "0.7", "engine_version": "0.1.0+test",
            "levels": [], "status": "completed", "artifacts": {}, "inputs": {"method": "if2"},
            "provenance": {},
        });
        if let Some(l) = label {
            rec["provenance"]["label"] = serde_json::json!(l);
        }
        std::fs::write(leaf.join("run.json"), rec.to_string()).unwrap();
        leaf
    }

    fn stored_label(leaf: &std::path::Path) -> Option<String> {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
        v["provenance"]["label"].as_str().map(String::from)
    }

    /// gh#896: a fit run's label is compare-and-set — another label on it is
    /// refused and left in place, the same label is a no-op, none is written.
    /// A fit label is written with its stamp: a fresh run takes it, a run
    /// carrying it is re-stamped, and a run carrying another has it replaced,
    /// the old one returned. Nothing but the two provenance fields changes.
    #[test]
    fn a_fit_label_is_written_with_its_stamp() {
        let stamp = |leaf: &std::path::Path| -> Option<String> {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
            v["provenance"]["label_set_at"].as_str().map(String::from)
        };
        let fresh = record_leaf("label_set_fresh", None);
        assert_eq!(set_fit_label(&fresh, "wide", "2026-11-14T00:00:00.000Z").unwrap(), None);
        assert_eq!(stored_label(&fresh).as_deref(), Some("wide"));
        assert_eq!(stamp(&fresh).as_deref(), Some("2026-11-14T00:00:00.000Z"));

        let held = record_leaf("label_set_held", Some("scout"));
        let before: serde_json::Value =
            serde_json::from_slice(&std::fs::read(held.join("run.json")).unwrap()).unwrap();
        assert_eq!(set_fit_label(&held, "scout", "2026-11-14T00:00:01.000Z").unwrap(), None,
            "re-stamping the same label replaces nothing");
        assert_eq!(set_fit_label(&held, "renamed", "2026-11-14T00:00:02.000Z").unwrap().as_deref(),
            Some("scout"), "the replaced label is returned");
        assert_eq!(stored_label(&held).as_deref(), Some("renamed"));
        assert_eq!(stamp(&held).as_deref(), Some("2026-11-14T00:00:02.000Z"));
        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(held.join("run.json")).unwrap()).unwrap();
        for key in ["run_id", "levels", "deps", "inputs", "artifacts", "status"] {
            assert_eq!(before[key], after[key], "{key} is untouched");
        }
    }

    #[test]
    fn a_label_stamp_is_rfc3339_with_milliseconds() {
        let s = label_stamp();
        assert_eq!(s.len(), "2026-11-14T00:00:00.000Z".len(), "{s}");
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T' && s.as_bytes()[19] == b'.', "{s}");
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_millis(1_794_614_400_042);
        assert_eq!(crate::cas::iso8601_utc_millis(t), "2026-11-14T00:00:00.042Z");
    }

    /// gh#896: concurrent read-modify-writes of one record lose nothing — each
    /// is one step under the record lock, through its own temporary file.
    #[test]
    fn concurrent_record_edits_lose_no_update() {
        let leaf = record_leaf("record_race", None);
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let leaf = leaf.clone();
                std::thread::spawn(move || {
                    ensure_input_listed(&leaf, "config_identities", &format!("id{i}")).unwrap()
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
        let ids = v["inputs"]["config_identities"].as_array().unwrap();
        assert_eq!(ids.len(), 16, "every writer's entry survives: {ids:?}");
    }

    /// gh#896: the store's label lock excludes a second holder while held, is
    /// free again once dropped, and is never blocked by a lock file whose
    /// recorded holder is not holding it — a crashed process, or a live
    /// process that reused its pid.
    #[cfg(unix)]
    #[test]
    fn the_label_lock_is_held_only_while_held() {
        let root = crate::test_support::unique_temp_dir("label_lock");
        let short = std::time::Duration::from_millis(100);
        let held = fit_label_lock(&root).unwrap();
        let err = fit_label_lock_within(&root, short).expect_err("a held lock excludes");
        assert!(err.contains("held by another process"), "{err}");
        assert!(err.contains(&format!("pid {}", std::process::id())), "names the recorded holder: {err}");
        drop(held);
        let again = fit_label_lock_within(&root, short).expect("free once dropped");
        drop(again);
        // A file naming a live pid (ours) that does not hold the lock.
        std::fs::write(root.join(".fit-labels.lock"), format!("{}", std::process::id())).unwrap();
        fit_label_lock_within(&root, short).expect("a recorded pid alone does not hold the lock");
        std::fs::remove_dir_all(&root).ok();
    }

    use super::*;

    // ── --table content folds into the run identity (Q1) ─────────────────────

    use ir::model::{
        InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule,
        SimulationConfig,
    };

    fn tiny_model() -> Model {
        Model {
            constants: vec![],
            ic_grad: Default::default(),
            name: "sir".into(),
            version: "1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            // One parameter with a default, so a draw/sweep point on `mu`
            // resolves (the resolver refuses a point on an unknown name).
            parameters: vec![ir::parameter::Parameter {
                name: "mu".into(),
                value: ir::parameter::ParamValue::Fixed { value: 0.5 },
                param_kind: None,
                param_dim: None,
            }],
            bindings: vec![],
            per_eval_bindings: vec![],
            initial_conditions: InitialConditions::default(),
            output: OutputConfig {
                times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 100.0,
                time_semantics: "continuous".into(),
                dt: Some(1.0),
                rng_seed: None,
                integrator: Default::default(),
                t_end_anchor: None, t_start_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }

    /// A `CasSink` over `tiny_model` with default identity inputs. What varies
    /// between cells is the cell's own `SimRun` (its `--table` map, its point),
    /// which is where `cell_resolve` reads them from.
    fn sink(runs_dir: &str) -> CasSink {
        sink_over(tiny_model(), runs_dir)
    }

    /// [`sink`] over `model`.
    fn sink_over(model: Model, runs_dir: &str) -> CasSink {
        let run = crate::sim_config::RunConfig::new(
            crate::args::types::ForwardBackend::ChainBinomial,
            1.0,
            &model.output.times,
            false,
            false,
            crate::util::OutputColumns::default(),
        )
        .unwrap();
        let mut resolved_model = crate::sim_config::ResolvedModel::new(
            model, ir::IR_VERSION.trim(), runid::inputs::EngineIdentity::of_build(version::ENGINE_ID), "model.ir.json",
        );
        resolved_model.stem = "sir".to_string();
        let sim = std::sync::Arc::new(crate::sim_config::ResolvedSimConfig {
            model: resolved_model,
            run,
            arms: vec![ResolvedEntry {
                name: "baseline".to_string(),
                route: None,
                enable: vec![],
                disable: vec![],
                params: HashMap::new(),
                scale: Vec::new(),
            }],
        });
        CasSink {
            engine: sim.engine_settings(),
            sim,
            runs_dir: runs_dir.to_string(),
            obs_enabled: false,
            emit_every: None,
            force: false,
            total: 1,
            counter: 0,
            completed_runs: Vec::new(),
            errors: Vec::new(),
            label: None,
            fit_dep: Vec::new(),
            progress: None,
            pre_run: HashMap::new(),
        }
    }

    fn baseline_spec(table_files: HashMap<String, String>) -> crate::engine::CellSpec {
        crate::engine::CellSpec {
            run_idx: 0,
            point_idx: 0,
            point_id: 0,
            scenario: crate::sim_job::ScenarioRef::Inline {
                name: "baseline".to_string(),
                enable: vec![],
                disable: vec![],
                params: indexmap::IndexMap::new(),
            },
            point_overrides: indexmap::IndexMap::new(),
            process_seed: 1,
            obs_seed: 1 ^ crate::util::SEED_MIX_OBS,
            sim_run: crate::util::SimRun { table_files, ..crate::util::SimRun::default() },
        }
    }

    /// `--table NAME=PATH` is a runtime override: the IR carries only the
    /// table reference, so the file's *content* must enter the run_id. Two
    /// tables with the same NAME but different CONTENT must resolve to
    /// DIFFERENT sim run_ids (else a changed `matrix.tsv` silently serves the
    /// stale cached trajectory); identical content → the same run_id
    /// (count-in-the-key for tables). Exercises the shared `CasSink`, so the
    /// fix covers the lone-run and multi-cell paths alike.
    #[test]
    fn table_content_folds_into_sim_run_id() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();
        // A model that reads `contact`: a 2 × 2 table declared
        // `C : g × g = external("contact")`.
        let mut model = tiny_model();
        model.tables.push(ir::table::Table {
            name: "C".into(),
            source: ir::table::TableSource::External {
                external: "contact".into(),
                shape: vec![
                    ir::table::TableAxis { dim: "g".into(), levels: 2 },
                    ir::table::TableAxis { dim: "g".into(), levels: 2 },
                ],
            },
            out_of_bounds: ir::table::OobPolicy::Error,
            cell_kind: None,
        });
        let sink = |runs_dir: &str| sink_over(model.clone(), runs_dir);

        let path_a = dir.path().join("matrix.tsv");
        let path_b = dir.path().join("matrix2.tsv");
        let path_c = dir.path().join("matrix3.tsv");
        std::fs::write(&path_a, b"1.0\t2.0\n3.0\t4.0\n").unwrap();
        std::fs::write(&path_b, b"9.9\t8.8\n7.7\t6.6\n").unwrap(); // different content
        std::fs::write(&path_c, b"1.0\t2.0\n3.0\t4.0\n").unwrap(); // same content as A

        let mk = |p: &std::path::Path| {
            let mut tf = HashMap::new();
            tf.insert("contact".to_string(), p.to_str().unwrap().to_string());
            baseline_spec(tf)
        };

        let (rt_a, _, _) = sink(runs_dir).cell_resolve(&mk(&path_a)).unwrap();
        let (rt_b, _, _) = sink(runs_dir).cell_resolve(&mk(&path_b)).unwrap();
        let (rt_c, _, _) = sink(runs_dir).cell_resolve(&mk(&path_c)).unwrap();

        // Negative control: with NO --table there is no run at all — the
        // model reads `contact`, so leaving it unbound is refused by name
        // rather than run, or keyed, without the table.
        let unbound = sink(runs_dir).cell_resolve(&baseline_spec(HashMap::new()));
        let err = match unbound {
            Ok(_) => panic!("a model reading `contact` must not resolve without it"),
            Err(e) => e,
        };
        assert!(err.contains("external input 'contact' has no file"), "{err}");

        assert_ne!(
            rt_a.run_id, rt_b.run_id,
            "different --table content (same name) MUST produce different sim run_ids"
        );
        assert_eq!(
            rt_a.run_id, rt_c.run_id,
            "identical --table content MUST produce the same sim run_id"
        );

        // The difference is isolated to the params level (tables live there),
        // never the model/config/scenario/seed levels.
        assert_eq!(rt_a.levels[0].hash, rt_b.levels[0].hash, "model level unchanged");
        assert_eq!(rt_a.levels[1].hash, rt_b.levels[1].hash, "config level unchanged");
        assert_ne!(rt_a.levels[2].hash, rt_b.levels[2].hash, "params level (tables) must differ");
        assert_eq!(rt_a.levels[3].hash, rt_b.levels[3].hash, "scenario level unchanged");
        assert_eq!(rt_a.levels[4].hash, rt_b.levels[4].hash, "seed level unchanged");
    }

    /// A `CellSpec` carrying a single param-point override (a sweep point /
    /// design point), at process seed 1, baseline scenario.
    fn spec_with_override(k: &str, v: f64) -> crate::engine::CellSpec {
        let mut overrides = indexmap::IndexMap::new();
        overrides.insert(k.to_string(), v);
        crate::engine::CellSpec {
            run_idx: 0,
            point_idx: 0,
            point_id: 0,
            scenario: crate::sim_job::ScenarioRef::Inline {
                name: "baseline".to_string(),
                enable: vec![],
                disable: vec![],
                params: indexmap::IndexMap::new(),
            },
            point_overrides: overrides.clone(),
            process_seed: 1,
            obs_seed: 1 ^ crate::util::SEED_MIX_OBS,
            sim_run: crate::util::SimRun {
                point_overrides: overrides.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                ..crate::util::SimRun::default()
            },
        }
    }

    /// The identity reads a cell's window from the model the cell runs, so a
    /// per-cell input that moves the window reaches the identity because it
    /// reaches the engine: a cell whose `SimRun` carries a `--to` horizon keys
    /// on that horizon though the scenario arm records none. (The identity
    /// used to take the horizon from the arm, a separate copy that had to be
    /// kept equal to what the engine applied.)
    #[test]
    fn a_cell_is_keyed_on_the_window_it_runs() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();
        let base = baseline_spec(HashMap::new());
        let mut shorter = baseline_spec(HashMap::new());
        shorter.sim_run.t_end_override = Some(50.0);

        let (rt_base, _, _) = sink(runs_dir).cell_resolve(&base).unwrap();
        let (rt_shorter, _, _) = sink(runs_dir).cell_resolve(&shorter).unwrap();
        assert_ne!(rt_base.run_id, rt_shorter.run_id,
            "a cell run to t = 50 must not share the run id of the cell run to the model's horizon");
        let config = |rt: &crate::resolve::ResolvedTrajectory| {
            rt.levels.iter().find(|l| l.name == "config").map(|l| l.hash)
        };
        assert_ne!(config(&rt_base), config(&rt_shorter), "the window is in the config level");
    }

    /// The backend, the step and the evaluator settings are keyed once, by the
    /// run-wide `config` level. A cell that would run other values is refused,
    /// not keyed on a computation it did not make.
    #[test]
    fn a_cell_running_other_engine_settings_than_the_run_is_keyed_on_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();
        let refused = |edit: &dyn Fn(&mut crate::util::SimRun)| {
            let mut spec = baseline_spec(HashMap::new());
            edit(&mut spec.sim_run);
            match sink(runs_dir).cell_resolve(&spec) {
                Ok(_) => panic!("a cell whose engine settings differ from the keyed run must be refused"),
                Err(e) => e,
            }
        };
        assert!(refused(&|r| r.backend = crate::args::types::ForwardBackend::Ode).contains("backend Ode"));
        assert!(refused(&|r| r.dt = 0.5).contains("dt = 0.5"));
        assert!(refused(&|r| r.eval = sim::EvalSettings { allow_degenerate_rates: true })
            .contains("allow_degenerate_rates: true"));
        assert!(refused(&|r| r.binomial = sim::rng::BinomialAlgorithm::Btrs)
            .contains("binomial sampler"));
        sink(runs_dir).cell_resolve(&baseline_spec(HashMap::new()))
            .expect("the cell the run is keyed on resolves");
    }

    /// The leaf a cell writes is keyed from the cell the engine ran, and that
    /// identity is checked against the one the cache-hit check resolved before
    /// the cell ran: a cell whose run drifted from its lookup is refused, and
    /// nothing is written.
    #[test]
    fn a_cell_whose_run_differs_from_its_lookup_is_not_written() {
        use crate::engine::RunSink;
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();
        let ran = |spec: &crate::engine::CellSpec| {
            let cell = crate::sim_config::ResolvedCell::resolve(tiny_model(), &spec.sim_run).unwrap();
            let traj = crate::util::simulate_compiled(&cell.compile().unwrap(), &cell, None).unwrap();
            crate::engine::CellResult { spec: spec.clone(), traj, cell, event_log: None }
        };

        // The control: what ran is what was looked up.
        let spec = baseline_spec(HashMap::new());
        let mut s = sink(runs_dir);
        assert!(s.should_run(&spec), "an empty store holds no hit");
        s.merge_cell(&ran(&spec)).unwrap();
        assert!(s.errors.is_empty(), "{:?}", s.errors);
        assert_eq!(s.completed_runs.len(), 1);

        // What ran drifted from the lookup (another seed): refused.
        let mut s = sink(runs_dir);
        let mut other = baseline_spec(HashMap::new());
        other.run_idx = 1;
        let _ = s.should_run(&other); // records the pre-run identity
        let mut drifted = other.clone();
        drifted.sim_run.seed = 99;
        s.merge_cell(&crate::engine::CellResult { spec: other.clone(), ..ran(&drifted) }).unwrap();
        assert!(s.errors.iter().any(|e| e.contains("looked up in the store as")), "{:?}", s.errors);
        assert!(s.completed_runs.is_empty(), "a refused cell writes nothing");
    }

    /// The `scenario` level keys the arm's enable / disable lists. A cell
    /// running another set of interventions than its arm describes is
    /// refused rather than keyed as the arm.
    #[test]
    fn a_cell_running_other_interventions_than_its_arm_is_refused() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ocaml/golden/seir_vaccine.ir.json");
        let (mut model, _) = crate::util::load_ir(path.to_str().unwrap()).unwrap();
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                let v = p.value.bounds().map(|(lo, hi)| 0.5 * (lo + hi)).unwrap_or(0.1);
                p.value = p.value.with_value(v);
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();
        let s = sink_over(model, runs_dir);

        s.cell_resolve(&baseline_spec(HashMap::new())).expect("the arm's own interventions resolve");
        let mut enabled = baseline_spec(HashMap::new());
        enabled.sim_run.adhoc_enable = vec!["sia_round_1".to_string()];
        match s.cell_resolve(&enabled) {
            Ok(_) => panic!("a cell enabling `sia_round_1` under an arm that keys none must be refused"),
            Err(e) => assert!(e.contains("sia_round_1"), "{e}"),
        }
    }

    /// Extra snapshot times change the trajectory and have no identity route.
    #[test]
    fn a_cell_with_required_output_times_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let mut spec = baseline_spec(HashMap::new());
        spec.sim_run.required_output_times = vec![2.5];
        match sink(runs_dir.to_str().unwrap()).cell_resolve(&spec) {
            Ok(_) => panic!("a cell with required snapshot times must be refused"),
            Err(e) => assert!(e.contains("required_output_times"), "{e}"),
        }
    }

    /// gh#241 PR E2 — THE dedupe proof. A `[design.*]` cell and a normal
    /// `[sweep]` / `simulate` sim cell with the SAME params/scenario/seed must
    /// resolve to the SAME `Sim` `run_id` and the SAME store path. The design
    /// path routes its generated points through the same `CasSink`/
    /// `cell_resolve` as the normal sweep (the point value enters the `params`
    /// identity level), so an identical cell collapses to one canonical leaf
    /// rather than two parallel copies under `designs/` vs `sims/`.
    ///
    /// The design method/name/point_idx are NOT in the identity: this test
    /// resolves a bare point override with no design label anywhere in the
    /// resolved inputs, which is exactly what `cell_resolve` sees for both
    /// entry points.
    #[test]
    fn design_cell_dedupes_with_normal_sim_cell() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();

        // "Design" cell: point value 0.3 for `mu`, supplied as a design point.
        let design_spec = spec_with_override("mu", 0.3);
        // "Normal sweep" cell: the SAME point value 0.3 for `mu`.
        let sweep_spec = spec_with_override("mu", 0.3);
        // A DIFFERENT point value — must NOT dedupe.
        let other_spec = spec_with_override("mu", 0.7);

        // Both paths build the same `CasSink` shape (same base model, root,
        // backend, dt, no tables) — only the spec's point override differs.
        let (rt_design, dir_design, rel_design) =
            sink(runs_dir).cell_resolve(&design_spec).unwrap();
        let (rt_sweep, dir_sweep, rel_sweep) =
            sink(runs_dir).cell_resolve(&sweep_spec).unwrap();
        let (rt_other, _, _) =
            sink(runs_dir).cell_resolve(&other_spec).unwrap();

        assert_eq!(
            rt_design.run_id, rt_sweep.run_id,
            "an identical design cell and normal sim cell MUST share one run_id (dedupe)"
        );
        assert_eq!(
            dir_design, dir_sweep,
            "the identical cell must resolve to the SAME store path (one leaf on disk)"
        );
        assert_eq!(rel_design, rel_sweep, "and the same store-relative path");

        assert_ne!(
            rt_design.run_id, rt_other.run_id,
            "a different design-point value must resolve to a different run_id"
        );
    }
}
