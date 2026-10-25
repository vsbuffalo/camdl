//! `CasSink` — the content-addressed store sink shared by `camdl simulate`
//! and `camdl batch run`.
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
    /// The scenario's own simulation horizon (`scenarios { x { simulate { to
    /// = … } } }`), when it declares one — otherwise the cell inherits
    /// `model.simulation.t_end` (gh#561).
    ///
    /// Identity-relevant: it changes the stored trajectory, so it is folded
    /// into the cell's `config` level by [`CasSink::cell_resolve`]. A model
    /// preset carries whatever it declared; an inline ad-hoc scenario is
    /// always `None`, because a horizon is a model-file declaration and there
    /// is no CLI spelling for one.
    pub t_end: Option<f64>,
}

impl ResolvedEntry {
    /// The entry for the model preset `name`, routed through the preset
    /// branch of the value resolver. The hashed delta comes from
    /// [`crate::params_resolver::resolve_preset_delta`] — the authority that
    /// branch applies — and `t_end` is supplied by the caller, because
    /// `simulate --to` may override the preset's composed horizon.
    pub fn from_preset(model: &ir::Model, name: &str, t_end: Option<f64>) -> Result<Self, String> {
        let delta = crate::params_resolver::resolve_preset_delta(model, name)
            .map_err(|e| e.to_string())?;
        Ok(ResolvedEntry {
            name: name.to_string(),
            route: Some(name.to_string()),
            enable: delta.enable,
            disable: delta.disable,
            params: delta.params.into_iter().collect(),
            scale: delta.scale,
            t_end,
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
    /// resolved process seed, and the cell's 0-based position on the
    /// parameter-point axis: the design index of a sweep cell, the draw index
    /// of a `--draws` cell. `(point_id, process_seed)` plus the cell `run_id`
    /// are what the ensemble's `grid` digest sorts over.
    pub(crate) scenario: String,
    pub(crate) process_seed: u64,
    pub(crate) point_id: usize,
    /// The cell's `params` level hash — the identity of the parameter values
    /// it ran. The ensemble's `params` level is the set of these (see
    /// `sim_ensemble_cas`). `None` only if identity resolution failed.
    pub(crate) params_level: Option<runid::ContentHash>,
}

// ─── CasSink: batch's content-addressed output strategy ───────────────────────

/// `RunSink` for `camdl batch run`: writes each cell into the content-addressed
/// store under `sims/` (the `runid` factored sim layout), resolving each cell's
/// identity via [`crate::resolve::resolve_trajectory`] so the path, `run.json`
/// `run_id`, and recorded sweep point match the `simulate` path exactly. Cache
/// hits are skipped via `should_run` (the engine never simulates them).
pub(crate) struct CasSink {
    /// Resolved `[[scenario]]` entries, looked up by name for the
    /// hash-relevant enable/disable/params delta.
    pub(crate) resolved_scenarios: Vec<ResolvedEntry>,
    pub(crate) model_path: String,
    pub(crate) model_stem: Option<String>,
    /// The **base** model — its whole-IR digest is the (constant-across-cells)
    /// model level. Never `cell.model` (which has scenario + sweep applied).
    /// Loaded through the same seam the engine loads from
    /// (`util::load_run_model` for `simulate`; `load_manifest_model` for
    /// `batch`, whose cells carry no anchors and no `--integrator`), and it is
    /// also the model each cell's parameter values are resolved over.
    ///
    /// There is deliberately no base parameter map here. A cell's `params`
    /// level is resolved from the cell's own `SimRun` — the object the engine
    /// runs — through `util::resolve_run_parameters`, the resolver call the
    /// engine makes (gh#583). A parallel map rebuilt beside it put a draw/sweep
    /// value above `--param` where the resolver puts it below.
    pub(crate) base_model: ir::Model,
    pub(crate) backend: crate::args::types::ForwardBackend,
    pub(crate) dt: f64,
    pub(crate) allow_degenerate_rates: bool,
    /// Output view (gh#156): the resolved `--no-flows` / `--columns` filter,
    /// applied to both this cell's leaf `traj.tsv` (writer) and its `config`
    /// identity (`cell_resolve`). Default = full output. `--output-every` is
    /// not here — it is lowered into the model schedule upstream.
    pub(crate) output_cols: crate::util::OutputColumns,
    /// Absolute `<output>/sims` subtree.
    pub(crate) runs_dir: String,
    pub(crate) obs_enabled: bool,
    /// gh#656: `simulate --emit-every`, the per-stream emission-cadence
    /// override. It sets the emitted times AND (via [`obs_subtree_hash`]) the
    /// obs subtree's address, so two cadences are two artifacts under one
    /// trajectory leaf. `None` for `batch run`, which has no such key, and for
    /// every `simulate` without the flag — those key exactly as they always did.
    pub(crate) emit_every: Option<crate::emit_every::EmitEvery>,
    pub(crate) force: bool,
    pub(crate) total: usize,
    pub(crate) counter: usize,
    pub(crate) completed_runs: Vec<RunEntry>,
    pub(crate) errors: Vec<String>,
    /// User-supplied display label, recorded on each leaf's
    /// `RunRecord.provenance.label`. `simulate --label` sets this; batch
    /// leaves it `None`.
    pub(crate) label: Option<String>,
    /// Upstream-fit lineage recorded on each leaf's `RunRecord.deps`. Set
    /// when `simulate --params` consumes a fit-MLE params file carrying a
    /// `[provenance]` block (so the sim records which fit it came from);
    /// empty for batch and for standalone (non-fit) params. Provenance only —
    /// a sim's identity is its factored levels (resolve_trajectory), never
    /// `deps`, so this does not change the sim's run_id or store path.
    pub(crate) fit_dep: Vec<runid::inputs::ArtifactRef>,
    /// The overall cells progress bar (advanced once per committed or
    /// cache-hit cell, finished on the last). `Some` only for a multi-cell run
    /// (`total > 1`) — a LONE `simulate` gets the engine's per-timestep bar
    /// instead, so its overall bar would be a redundant `1/1`. `None` is the
    /// inert case (single cell, or `--progress none`/tests). Honours the
    /// `--progress` mode via [`crate::progress::Task`].
    pub(crate) progress: Option<crate::progress::Task>,
}

/// The overall cells bar for a multi-cell run, or `None` for a single cell
/// (the engine's per-timestep bar covers that) or when no bar applies. The
/// `Task` itself honours `--progress none`/`plain` internally.
pub(crate) fn cells_progress(total: usize, label: impl Into<String>) -> Option<crate::progress::Task> {
    (total > 1).then(|| crate::progress::Reporter::new().task(total as u64, label, "cells"))
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
    /// `pub(crate)` for `batch`'s dry-run / status prediction (`CasSink::predict_cells`).
    pub(crate) fn root(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.runs_dir)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// Content digests of the external `--table` files, sorted by table name,
    /// for folding into the `params` level (`ResolvedParams.tables`). Each
    /// file's *bytes* are hashed (never its path), so a regenerated or edited
    /// table re-keys the run and a stale cached trajectory is never served.
    ///
    /// Mirrors `profile_cas::data_digests`: sort by name, `digest_bytes` per
    /// file (= `ContentHash::from_hex(sha256_hex(bytes))`). Empty `table_files`
    /// → empty vec, identical to the no-`--table` path (so a plain `simulate`
    /// is unaffected). Read from the cell's `SimRun::table_files` — the map the
    /// resolver loads the tables from when the cell runs.
    fn table_digests(
        table_files: &HashMap<String, String>,
    ) -> Result<Vec<runid::inputs::DataDigest>, String> {
        let mut names: Vec<&String> = table_files.keys().collect();
        names.sort();
        names
            .iter()
            .map(|name| {
                let path = &table_files[*name];
                let bytes = std::fs::read(path)
                    .map_err(|e| format!("cannot read --table file '{}' ({}): {}", name, path, e))?;
                Ok(runid::inputs::DataDigest(runid::ContentHash::digest_bytes(&bytes)))
            })
            .collect()
    }

    /// Resolve a cell's trajectory identity + factored store path.
    ///
    /// The model level is the **base** model digest (constant across the
    /// sweep — never `cell.model`, which has scenario + sweep applied). The
    /// `params` level is the cell's RESOLVED parameter values — every tier
    /// (model default, `--params`, draw/sweep point, scenario, `--param`,
    /// `--param-vec`) applied in the resolver's order by the resolver itself,
    /// over the cell's own `SimRun`, exactly as the engine resolves them when
    /// it runs the cell. The named scenario's enable/disable/params is the
    /// `scenario` level. `process_seed` is engine-resolved.
    ///
    /// `pub(crate)` for `batch`'s dry-run / status prediction (`CasSink::predict_cells`).
    pub(crate) fn cell_resolve(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<(crate::resolve::ResolvedTrajectory, std::path::PathBuf, String), String> {
        let name = spec.scenario.name();
        let resolved = self.resolved_scenarios.iter()
            .find(|s| s.name == name || s.route.as_deref() == Some(name))
            .ok_or_else(|| format!("cell scenario '{}' not among resolved batch scenarios", name))?;

        let params: HashMap<String, f64> = crate::util::resolved_parameter_values(
            &crate::util::resolve_run_parameters(&self.base_model, &spec.sim_run)?.model,
        ).into_iter().collect();

        let scenario_label = if resolved.name.is_empty() { "baseline" } else { resolved.name.as_str() };
        // The label names the point's parameters at the values the cell RAN
        // (resolved), not the values the point asked for: a `--param` or a
        // scenario that shadows a draw column must not leave the path
        // advertising the draw's value.
        let ran_point: indexmap::IndexMap<String, f64> = spec.point_overrides.keys()
            .filter_map(|k| params.get(k).map(|v| (k.clone(), *v)))
            .collect();
        let param_label = params_path_label(&ran_point);

        let ctx = crate::resolve::TrajectoryCtx {
            model: &self.base_model,
            model_stem: self.model_stem.as_deref().unwrap_or("model"),
            ir_version: ir::IR_VERSION.trim(),
            engine_version: version::VERSION_SHORT,
            backend: self.backend,
            dt: self.dt,
            // gh#641: the cell's EFFECTIVE start, not the base model's. A run
            // seeded from a filtered state begins at that state's origin and
            // therefore covers a different window — the same argument the
            // horizon override makes below, from the other end.
            t_start: spec
                .sim_run
                .init_state
                .as_ref()
                .map(|i| i.origin_t)
                .unwrap_or(self.base_model.simulation.t_start),
            // gh#561: the cell's EFFECTIVE horizon, not the base model's. A
            // scenario declaring its own `simulate { to }` produces a different
            // trajectory, so it must re-key (`feedback: count-in-the-key`) —
            // otherwise a horizon menu whose scenarios carry no `set`/`enable`
            // delta resolves every cell to one `run_id` and the store serves
            // one trajectory for all of them.
            //
            // This folds the RESOLVED VALUE into the existing `SimConfig::t_end`
            // rather than adding a field to the scenario level, so it costs
            // nothing for models that don't use the feature: a preset with no
            // `to`, or one equal to the model horizon, hashes exactly as before.
            t_end: resolved.t_end.unwrap_or(self.base_model.simulation.t_end),
            output: &self.base_model.output.times,
            allow_degenerate_rates: self.allow_degenerate_rates,
            no_flows: self.output_cols.no_flows,
            columns: &self.output_cols.allow,
            params: &params,
            table_digests: Self::table_digests(&spec.sim_run.table_files)?,
            enable: &resolved.enable,
            disable: &resolved.disable,
            scen_params: &resolved.params,
            scen_scale: &resolved.scale,
            param_label: &param_label,
            scenario_label,
            base_seed: spec.process_seed,
            process_seed: spec.process_seed,
            // gh#641 / gh#697: the state this cell restored. The origin
            // ENSEMBLE's content digest and the restored ROW both key the run —
            // a re-filtered / re-fitted ensemble under an unchanged model must
            // not serve the cached forecast, and two cells sharing a
            // `process_seed` (`--seeds 7,7`) must not collide while starting
            // from different rows. `None` for every `batch run` cell (no
            // `init_state` key — CLI-only, like `to`).
            init_state: spec.sim_run.init_state.as_ref().map(|i| {
                runid::inputs::InitStateDigest {
                    ensemble: runid::inputs::DataDigest(i.ensemble_digest),
                    row: i.row,
                }
            }),
        };
        let rt = crate::resolve::resolve_trajectory(&ctx).map_err(|e| format!("resolve error: {e}"))?;
        let root = self.root();
        let dir = runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels);
        let rel = dir.strip_prefix(&root).unwrap_or(&dir).to_string_lossy().into_owned();
        Ok((rt, dir, rel))
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
        let dir = runid::find_leaf(&root, runid::ArtifactKind::Sim, &rt.levels)
            .unwrap_or_else(|| runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels));
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
    /// the cell runs ([`crate::util::resolve_run_ir`], the same model
    /// `merge_cell` writes from), with every requested stream file and its
    /// `obs.json` present. A leaf holding another cadence's or seed's subtree
    /// is a miss, and directory presence alone is not a hit.
    pub(crate) fn stored_hit(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<Option<std::path::PathBuf>, String> {
        let (rt, _, _) = self.cell_resolve(spec)?;
        let Some(dir) = runid::find_leaf(&self.root(), runid::ArtifactKind::Sim, &rt.levels) else {
            return Ok(None);
        };
        if self.obs_enabled && !self.base_model.observations.is_empty() {
            let model = crate::util::resolve_run_ir(&spec.sim_run)?;
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
        Ok(Some(dir))
    }
}

impl crate::engine::RunSink for CasSink {
    fn should_run(&mut self, spec: &crate::engine::CellSpec) -> bool {
        if self.force {
            // Through the one policy seam: under Force the incumbent is
            // displaced (quarantined) HERE, so the recompute lands on a clean
            // leaf — previously force only skipped this check and the
            // recomputed bytes were discarded at commit as an
            // already-completed no-op, which is why `--force` could not change
            // a stored artifact. A failure surfaces in `merge_cell`.
            if let Ok((rt, _, _)) = self.cell_resolve(spec) {
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
        !matches!(self.stored_hit(spec), Ok(Some(_)))
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
            point_id: spec.point_idx,
            params_level,
        });
    }

    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let spec = &cell.spec;
        let name = spec.scenario.name().to_string();

        let (rt, _dir, _rel) = match self.cell_resolve(spec) {
            Ok(x) => x,
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: {}", name, spec.process_seed, e));
                return Ok(());
            }
        };

        let cols = self.output_cols.cols(&cell.model);
        let traj_bytes = crate::util::traj_tsv_bytes(&cell.traj, &cols);
        let traj_digest = runid::ContentHash::digest_bytes(&traj_bytes);

        // Declare the obs ensemble as a child sub-artifact so the leaf's
        // exact-set doesn't orphan the `obs/` subdir (written after the
        // trajectory commits). M2-interim: the obs child is recorded + its
        // streams written via the existing ensemble writer; a full
        // obs-child RunRecord identity is a follow-up.
        let mut children: std::collections::BTreeMap<String, Vec<runid::ContentHash>> =
            std::collections::BTreeMap::new();
        let has_obs = self.obs_enabled && !cell.model.observations.is_empty();
        if has_obs {
            let obs_seed = spec.process_seed ^ crate::util::SEED_MIX_OBS;
            // The SAME hash `write_obs_into_cas` names the subtree with — a
            // declared child that did not match the directory written beside it
            // would be provenance pointing at nothing.
            let obs_hash = match obs_subtree_hash(
                &cell.model.observations, self.emit_every.as_ref(),
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
        let mut meta = crate::resolve::RecordMeta::new(
            ir::IR_VERSION.trim(), self.model_path.clone(), self.label.clone())
            .with_deps(self.fit_dep.clone())
            .with_children(children.clone());

        let mut artifacts = runid::Artifacts::new();
        artifacts.insert("traj.tsv", traj_bytes);
        // `simulate --event-log`: the recorded event log is a first-class
        // artifact in THIS leaf, alongside `traj.tsv`, declared in the
        // exact-set (so the on-disk `event_log.tsv` path is a valid input to
        // `lineage realize`). The recorder is passive (Tier 2a) so the run_id
        // is unchanged — a freshly-committed leaf a plain `simulate` would
        // write simply gains one more artifact. Recording into a leaf that
        // already exists WITHOUT the log is handled below by `augment`, which
        // adds it in place; the store gained that door precisely because this
        // case used to lose the log. (This comment previously said `--force`
        // or a fresh identity was needed — neither was true: forcing
        // re-commits into the same discard.)
        // Kept alongside the staged set so they can be re-added to an
        // ALREADY-COMPLETED leaf below: a cache hit discards the staged
        // artifacts wholesale, which is how `--event-log` against an existing
        // leaf silently lost its log.
        let mut optional_artifacts: Vec<(&'static str, Vec<u8>)> = Vec::new();
        if let Some(ref el) = cell.event_log {
            match sim::lineage::event_log_io::to_tsv_bytes(el) {
                Ok(bytes) => {
                    optional_artifacts.push(("event_log.tsv", bytes.clone()));
                    artifacts.insert("event_log.tsv", bytes);
                }
                Err(e) => {
                    self.counter += 1;
                    self.errors.push(format!("scenario={} seed={}: event log serialize: {:?}",
                        name, spec.process_seed, e));
                    return Ok(());
                }
            }
        }
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
        if self.force {
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
        // the stored leaf lacks — `--event-log` against a leaf first written
        // without it — would simply vanish. (The old comment here claimed
        // `--force` was the workaround; it was not: that path re-commits and
        // reaches the same discard.) `augment` adds it under the leaf's lock;
        // it is a no-op when the bytes already match, which is the fresh-commit
        // case, and it refuses divergent bytes for the same reason the commit
        // path does.
        for (artifact, bytes) in &optional_artifacts {
            if let Err(e) = store.augment(
                &dest, &runid::LeafIdentity::new(rt.run_id), artifact, bytes)
            {
                self.errors.push(format!(
                    "scenario={} seed={}: recording {} into {}: {}",
                    name, spec.process_seed, artifact, dest.display(), e));
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
                if let Err(e) = ensure_obs_children_declared(&dest, ids) {
                    self.errors.push(format!("scenario={} seed={}: declaring obs child: {}",
                        name, spec.process_seed, e));
                }
            }
            // gh#641: a restarted cell's obs axis begins at the forecast
            // origin, exactly as the `-o`/`--obs` mirror's does. `None` for
            // every ordinary cell — and for every `batch run` cell, which has
            // no `init_state` key.
            let restart_origin = spec.sim_run.init_state.as_ref().map(|i| i.origin_t);
            if let Err(e) = write_obs_into_cas(
                &dest, &cell.model, &cell.traj, spec.process_seed, restart_origin,
                self.emit_every.as_ref(),
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
            point_id: spec.point_idx,
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
fn read_traj_digest(dir: &std::path::Path) -> Option<runid::ContentHash> {
    let bytes = std::fs::read(dir.join("run.json")).ok()?;
    let rec: runid::RunRecord = serde_json::from_slice(&bytes).ok()?;
    rec.artifacts.get("traj.tsv").map(|c| c.digest)
}

/// Ensure a leaf's `run.json` carries `provenance.label == Some(label)`,
/// rewriting it atomically (tmp + rename) only when it differs. `provenance`
/// is metadata, not identity — the run_id and exact-set are untouched — so this
/// is the same in-place relabel `camdl label` performs, applied at write time
/// so an explicit `--label` survives a CAS cache hit. Idempotent: a no-op when
/// the leaf already has the requested label (e.g. a fresh commit).
fn ensure_provenance_label(leaf_dir: &std::path::Path, label: &str) -> Result<(), String> {
    let path = leaf_dir.join("run.json");
    let txt = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut rec: runid::RunRecord = serde_json::from_str(&txt)
        .map_err(|e| format!("parse {}: {}", path.display(), e))?;
    if rec.provenance.label.as_deref() == Some(label) {
        return Ok(()); // already current — no rewrite
    }
    rec.provenance.label = Some(label.to_string());
    let tmp = leaf_dir.join("run.json.tmp");
    let json = serde_json::to_string_pretty(&rec)
        .map_err(|e| format!("serialize run.json: {}", e))?;
    std::fs::write(&tmp, json)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .map_err(|e| format!("write {}: {}", path.display(), e))
}

/// Ensure a leaf's `run.json` declares every id in `ids` under its `obs`
/// child, rewriting it atomically (tmp + rename) only when one is missing.
/// `children` is metadata, not identity — the run_id and exact-set of the
/// leaf's own files are untouched — so this is the same in-place record
/// update `ensure_provenance_label` performs. Idempotent: a no-op when the
/// record already declares them (the fresh-commit case).
fn ensure_obs_children_declared(
    leaf_dir: &std::path::Path,
    ids: &[runid::ContentHash],
) -> Result<(), String> {
    let path = leaf_dir.join("run.json");
    let txt = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut rec: runid::RunRecord = serde_json::from_str(&txt)
        .map_err(|e| format!("parse {}: {}", path.display(), e))?;
    let declared = rec.children.entry("obs".to_string()).or_default();
    let missing: Vec<runid::ContentHash> =
        ids.iter().filter(|id| !declared.contains(id)).copied().collect();
    if missing.is_empty() {
        return Ok(()); // already current — no rewrite
    }
    declared.extend(missing);
    let tmp = leaf_dir.join("run.json.tmp");
    let json = serde_json::to_string_pretty(&rec)
        .map_err(|e| format!("serialize run.json: {}", e))?;
    std::fs::write(&tmp, json)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .map_err(|e| format!("write {}: {}", path.display(), e))
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
const OBS_FILE_FORMAT: &str = "\nobs-file-format:2";

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
    // for every ordinary cell. NOTE this function serves `simulate` cells as
    // well as `batch run` ones — the obs subtree is written for both.
    restart_origin: Option<f64>,
    // gh#656: the `--emit-every` override — it sets the emitted times AND names
    // the subtree (via `obs_subtree_hash`), so the two cadences coexist under one
    // trajectory leaf instead of one overwriting the other.
    emit: Option<&crate::emit_every::EmitEvery>,
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
        crate::project_coverages(traj, obs_ir, model, &plan.coverages())?;
        plans.push(plan);
    }

    std::fs::create_dir_all(&obs_dir)
        .map_err(|e| format!("cannot create {}: {}", obs_dir.display(), e))?;

    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model.clone())
            .map_err(|e| format!("model compile error for obs: {:?}", e))?,
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
        let rows = plan.coverages();
        let projected = crate::project_coverages(traj, obs_ir, model, &rows)?;

        let draws: Vec<f64> = rows.iter().enumerate()
            .map(|(ti, &(obs_t, _))| {
                let snap = crate::snap_at(traj, obs_t);
                sampler(projected[ti], obs_t, &snap.int_state.counts, &[], &mut obs_rng)
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
    use super::*;

    // ── --table content folds into the run identity (Q1) ─────────────────────

    use ir::model::{
        InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule,
        SimulationConfig,
    };

    fn tiny_model() -> Model {
        Model {
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
                t_end_anchor: None,
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
        CasSink {
            resolved_scenarios: vec![ResolvedEntry {
                name: "baseline".to_string(),
                route: None,
                enable: vec![],
                disable: vec![],
                params: HashMap::new(),
                scale: Vec::new(),
                t_end: None,
            }],
            model_path: "model.ir.json".to_string(),
            model_stem: Some("sir".to_string()),
            base_model: tiny_model(),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            allow_degenerate_rates: false,
            output_cols: crate::util::OutputColumns::default(),
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
        }
    }

    fn baseline_spec(table_files: HashMap<String, String>) -> crate::engine::CellSpec {
        crate::engine::CellSpec {
            run_idx: 0,
            point_idx: 0,
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
    /// fix covers both `simulate` and `batch`.
    #[test]
    fn table_content_folds_into_sim_run_id() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();

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

        // Negative control: with NO --table, the run_id is yet another value,
        // and it must differ from the with-table run_ids (so the table digest
        // is genuinely folded in, not silently dropped).
        let (rt_none, _, _) = sink(runs_dir)
            .cell_resolve(&baseline_spec(HashMap::new()))
            .unwrap();

        assert_ne!(
            rt_a.run_id, rt_b.run_id,
            "different --table content (same name) MUST produce different sim run_ids"
        );
        assert_eq!(
            rt_a.run_id, rt_c.run_id,
            "identical --table content MUST produce the same sim run_id"
        );
        assert_ne!(
            rt_a.run_id, rt_none.run_id,
            "a --table override must change the run_id vs no table (digest folded in)"
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

    /// gh#583 item C's guard. `SimRun` is everything the engine runs a cell
    /// from, so every field of it must reach the cell's identity or say why it
    /// does not. The destructure is exhaustive: a new `SimRun` field does not
    /// compile here until someone classifies it — the decision whose absence
    /// let `--integrator` and `--param-vec` run without being hashed.
    #[test]
    fn every_simrun_field_has_an_identity_route() {
        let crate::util::SimRun {
            // model level: `base_model` is `util::load_run_model` of this path.
            ir_path: _,
            // model level: substituted and applied inside `load_run_model`.
            obs_anchors: _,
            integrator: _,
            // params level: resolved by `util::resolve_run_parameters`, the
            // engine's own resolver call (tiers 3, 5, 3.5, 5, and 4).
            params_files: _,
            overrides: _,
            point_overrides: _,
            fit_fixed: _,
            set_vec_entries: _,
            scenario_name: _,
            scenario_inline_set: _,
            scenario_inline_scale: _,
            // params level: each file's content digest (`table_digests`).
            table_files: _,
            // scenario level: the `ResolvedEntry` enable / disable delta.
            adhoc_enable: _,
            adhoc_disable: _,
            // provenance: the scenario label only; its values are above.
            scenario_inline_name: _,
            // config level: `SimConfig::t_end` via `ResolvedEntry::t_end`.
            t_end_override: _,
            // config level: `SimConfig::t_start` and the `init_state` digest.
            init_state: _,
            // config level: the job's backend and dt, which
            // `engine::build_cell_sim_run` copies onto every cell.
            backend: _,
            dt: _,
            // seed level: `CellSpec::process_seed`, the value copied here.
            seed: _,
            // refused on the CAS-writing path (`build_simulate_cas_sink`):
            // it changes the trajectory and has no identity route yet.
            required_output_times: _,
        } = crate::util::SimRun::default();
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
