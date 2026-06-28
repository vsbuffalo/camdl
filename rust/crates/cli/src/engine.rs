//! `run_job` — the single simulation engine over [`SimulateJob`].
//!
//! Both `camdl simulate` (`main::run_simulate`) and `camdl batch run`
//! (`batch::cmd_batch_run`) build a [`SimulateJob`] and hand it here. This
//! is the run-spec §3.1 convergence: the multi-run logic — cell
//! expansion, the determinism-critical seed arithmetic, per-cell
//! [`SimRun`] construction, and simulation — lives in exactly one place.
//! Each entry point supplies a [`RunSink`] for its output shape, so the
//! *orchestration* is shared while the *writer* is pluggable.
//!
//! ## Determinism contract (CLAUDE.md §"RNG and paired-seed coupling")
//!
//! The reroute MUST NOT reorder RNG draws. [`process_seed_for`] reproduces
//! the historical `main.rs` arithmetic verbatim:
//!
//! ```text
//! process_seed = if explicit --seeds { seeds[rep] }
//!                else if single run  { base_seed }
//!                else { base_seed ^ (point_idx * SEED_MIX_DRAW)
//!                                 ^ (rep       * SEED_MIX_REP) }
//! obs_seed     = process_seed ^ SEED_MIX_OBS
//! // scenario is DELIBERATELY ABSENT from the seed mix → CRN coupling.
//! ```
//!
//! Per-cell `process_seed` depends only on `(explicit_seeds, base_seed,
//! point_idx, rep, total_runs)` — never on the scenario or on iteration
//! order — so a parallel sink (batch) and a sequential one (simulate)
//! produce identical per-cell trajectories. The grid is built first; cells
//! run in `scenario → point → rep` index order (sequential) or via Rayon
//! (parallel), and the sink merges results in that canonical order. The
//! PIN suite (`tests/determinism_pin.rs`) locks seed derivation, seed
//! coherence, and determinism.

use std::collections::HashMap;

use indexmap::IndexMap;
use rayon::prelude::*;

use crate::sim_job::{ScenarioRef, SimulateJob};
use crate::util::{self, SimRun, SEED_MIX_OBS};

/// Derive the process seed for one cell, reproducing `main.rs:833-841`.
///
/// * `explicit_seeds` — `Some(&[..])` when `--seeds` / batch `seeds`
///   listed seeds explicitly. Then the `rep` index selects directly.
/// * `total_runs == 1` — a lone run uses `base_seed` untouched.
/// * otherwise — XOR-mix `point_idx` and `rep` into `base_seed`.
///
/// Scenario index is never folded in: that is what makes paired scenarios
/// (e.g. `--enable`) share the baseline RNG byte-for-byte until the
/// intervention fires (CRN coupling).
pub fn process_seed_for(
    explicit_seeds: Option<&[u64]>,
    base_seed: u64,
    point_idx: usize,
    rep: usize,
    total_runs: usize,
) -> u64 {
    if let Some(seeds) = explicit_seeds {
        seeds[rep]
    } else if total_runs == 1 {
        base_seed
    } else {
        util::mix_cell_seed(base_seed, point_idx as u64, rep as u64)
    }
}

/// The fully-resolved spec for one cell, computed before simulation so the
/// grid can be planned (and parallelised) up front. Cheap to clone.
#[derive(Clone)]
pub struct CellSpec {
    /// 0-based global run index in canonical order (`scenario → point →
    /// rep`). Drives the wide-format `replicate` column (`run_idx + 1`).
    pub run_idx: usize,
    /// 0-based param-point index (the "draw" / sweep point). Drives the
    /// wide-format `draw` column (`point_idx + 1`).
    pub point_idx: usize,
    /// The scenario reference for this cell.
    pub scenario: ScenarioRef,
    /// The param-point override map for this cell (sweep point / draw row).
    pub point_overrides: IndexMap<String, f64>,
    /// The process seed (drives the trajectory RNG).
    pub process_seed: u64,
    /// The observation seed (`process_seed ^ SEED_MIX_OBS`).
    pub obs_seed: u64,
    /// The `SimRun` that produces this cell.
    pub sim_run: SimRun,
}

/// A completed cell: its spec plus the simulated trajectory and resolved
/// model. Handed to [`RunSink::merge_cell`] in canonical order.
pub struct CellResult {
    pub spec: CellSpec,
    pub traj: sim::Trajectory,
    pub model: ir::Model,
    /// The recorded identity-free event log, present only for the
    /// `simulate --event-log` path (the recorder is passive — Tier 2a — so a
    /// cell carrying `Some(..)` has the same `traj`/run_id as one carrying
    /// `None`). A `CasSink` writes it into the leaf as `event_log.tsv`.
    pub event_log: Option<sim::lineage::EventLog>,
}

/// The grid shape, computed once up front so a sink can size headers /
/// print a banner before any cell runs.
pub struct Grid {
    pub n_scenarios: usize,
    pub n_points: usize,
    pub total_runs: usize,
    /// Rayon thread count requested by the job (1 = sequential).
    pub parallel: usize,
}

/// Output strategy for [`run_job`].
///
/// The engine plans the grid, runs every cell (sequentially or via Rayon),
/// then calls [`RunSink::merge_cell`] for each in canonical
/// `scenario → point → rep` order. A sink that writes independent per-cell
/// artifacts (batch's CAS tree) does its filesystem work inside
/// `merge_cell`; a sink that builds combined output (simulate's
/// wide-format TSV) accumulates and flushes in `on_finish`.
pub trait RunSink {
    /// Called once before any cell runs.
    fn on_start(&mut self, _grid: &Grid) {}
    /// Whether a planned cell should be simulated. Default `true`. A
    /// content-addressed sink (batch's CAS tree) returns `false` for a
    /// cache hit so the engine skips the (expensive) simulation; `on_skip`
    /// then records the cached cell. Called in canonical order, before the
    /// simulation phase, so per-cell seed indices are unaffected.
    fn should_run(&mut self, _spec: &CellSpec) -> bool {
        true
    }
    /// Called for a cell whose `should_run` returned `false`. Default
    /// no-op. Called in canonical order.
    fn on_skip(&mut self, _spec: &CellSpec) {}
    /// Called for each completed cell, in canonical order. Returns an error
    /// to abort the job.
    fn merge_cell(&mut self, cell: &CellResult) -> Result<(), String>;
    /// Called once after all cells merged (success path).
    fn on_finish(&mut self, _grid: &Grid) -> Result<(), String> {
        Ok(())
    }
}

/// Plan the cell grid for `job`: the full ordered list of [`CellSpec`]s,
/// plus the [`Grid`] shape. Pure — no simulation. Iteration order is
/// `scenario → point → rep`, matching the pre-unification `run_simulate`
/// loop.
pub fn plan_grid(job: &SimulateJob) -> (Vec<CellSpec>, Grid) {
    let param_points = job.source.param_points();
    let replicates = effective_replicates(job);
    let explicit_seeds = job.seeds.explicit();
    let base_seed = job.seeds.base();
    let scenarios = effective_scenarios(job);

    let total_runs = scenarios.len() * param_points.len() * replicates;
    let grid = Grid {
        n_scenarios: scenarios.len(),
        n_points: param_points.len(),
        total_runs,
        parallel: job.parallel.max(1),
    };

    let table_files: HashMap<String, String> =
        job.table_files.iter().cloned().collect();

    let mut specs = Vec::with_capacity(total_runs);
    let mut run_idx = 0usize;
    for scenario in scenarios.iter() {
        for (point_idx, point_overrides) in param_points.iter().enumerate() {
            for rep in 0..replicates {
                let process_seed = process_seed_for(
                    explicit_seeds, base_seed, point_idx, rep, total_runs,
                );
                let obs_seed = process_seed ^ SEED_MIX_OBS;
                let sim_run = build_cell_sim_run(
                    job, scenario, point_overrides, &table_files, process_seed,
                );
                specs.push(CellSpec {
                    run_idx,
                    point_idx,
                    scenario: scenario.clone(),
                    point_overrides: point_overrides.clone(),
                    process_seed,
                    obs_seed,
                    sim_run,
                });
                run_idx += 1;
            }
        }
    }
    (specs, grid)
}

/// Drive the full `scenario × param-point × replicate` grid for `job`,
/// merging each completed cell into `sink` in canonical order. The single
/// engine behind `simulate` and `batch run`.
///
/// When `job.parallel > 1` the simulation phase runs cells via Rayon; the
/// merge phase is always in-order. Per-cell seeds are order-independent
/// (see [`process_seed_for`]), so parallelism never perturbs trajectories.
pub fn run_job(job: &SimulateJob, sink: &mut dyn RunSink) -> Result<(), String> {
    check_explicit_draws_scenario_collision(job)?;
    let (specs, grid) = plan_grid(job);
    sink.on_start(&grid);

    // Cache classification — in canonical order, before any simulation, so
    // a content-addressed sink can skip cache hits deterministically
    // without perturbing per-cell seed indices.
    let to_run: Vec<CellSpec> = specs
        .into_iter()
        .filter(|spec| {
            if sink.should_run(spec) {
                true
            } else {
                sink.on_skip(spec);
                false
            }
        })
        .collect();

    // Simulation phase. Each cell is independent; run sequentially or via
    // Rayon. Results carry their spec so the merge can stay ordered.
    //
    // Single-cell special case: a lone `camdl simulate` run (one scenario ×
    // one param-point × one replicate) shows a per-timestep `t/t_end` + ETA
    // progress bar so the user isn't staring at a silent terminal for ~45s.
    // The bar is driven by an RNG-free tick threaded into the backend loop
    // (see `util::run_simulation_with_progress`); the multi-cell path passes
    // `None` and is byte-identical. Ensembles keep the existing behaviour
    // (no inner bar in this commit — the sink owns any per-cell bar).
    let results: Vec<Result<CellResult, String>> = if grid.total_runs == 1 && to_run.len() == 1 {
        let spec = to_run.into_iter().next().expect("len checked == 1");
        vec![run_one_cell_with_progress(spec)]
    } else if grid.parallel > 1 {
        to_run
            .into_par_iter()
            .map(run_one_cell)
            .collect()
    } else {
        to_run.into_iter().map(run_one_cell).collect()
    };

    // Merge phase — strictly in canonical order.
    for r in results {
        let cell = r?;
        sink.merge_cell(&cell)?;
    }

    sink.on_finish(&grid)
}

/// Run a single planned cell to a [`CellResult`].
fn run_one_cell(spec: CellSpec) -> Result<CellResult, String> {
    let (traj, model) = util::run_simulation(&spec.sim_run)?;
    Ok(CellResult { spec, traj, model, event_log: None })
}

/// Run a lone cell with a per-timestep `t/t_end` + ETA progress bar on stderr.
///
/// Respects the `--progress` mode via `crate::progress`:
///   - Pretty (TTY): a live steady-tick bar
///     `simulate · <backend>  ████░░ ETA 11s`, positioned by an RNG-free tick
///     (so the trajectory is byte-identical to the bar-less path), cleared on
///     completion.
///   - Plain (off-TTY): the bar's draw target is hidden; we emit a single
///     status line instead so logs/CI show motion without carriage returns.
///   - None: nothing — falls straight through to the byte-identical
///     `run_one_cell` (no bar, no line).
fn run_one_cell_with_progress(spec: CellSpec) -> Result<CellResult, String> {
    use crate::args::types::ForwardBackend;

    if crate::progress::is_none() {
        return run_one_cell(spec);
    }

    // Resolve/compile the model BEFORE the simulate bar exists. The compile
    // step (camdlc, via resolve_run_model) shows its own spinner; if the
    // simulate bar were already steady-ticking, the two indicatif draw targets
    // would stomp each other on stderr (garbled bar; an orphaned compile-
    // spinner line left on screen, the reported Ctrl-C residue). Serializing
    // them is the fix: the compile spinner finishes and clears here, then the
    // simulate bar starts. resolve_run_model is the same call run_one_cell
    // makes via util::run_simulation, so this does not double-compile.
    let (compiled, model) = util::resolve_run_model(&spec.sim_run)?;

    let backend = match spec.sim_run.backend {
        ForwardBackend::Gillespie => "gillespie",
        ForwardBackend::ChainBinomial => "chain_binomial",
        ForwardBackend::Ode => "ode",
    };

    // Length 1000 matches the tick's `frac * 1000` scale in
    // `util::simulate_compiled`. Hidden in plain/none modes.
    let pb = indicatif::ProgressBar::with_draw_target(
        Some(1000),
        crate::progress::draw_target(),
    );
    // indicatif 0.17's `{bar}` element already renders a trailing percentage,
    // so the template deliberately omits a separate `{percent}` (a `{percent}`
    // produced a duplicated "61%  61%" in testing). Bar + ETA is the display.
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "simulate \u{b7} {prefix}  {bar:24.cyan/blue} ETA {eta}",
        )
        .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
        .progress_chars("\u{2588}\u{2591} "),
    );
    pb.set_prefix(backend.to_string());
    // Steady-tick redraw (separate render thread; never touches the sim) so
    // the bar appears and the ETA updates smoothly between `set_position`
    // calls. A no-op against a hidden draw target (plain/none).
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    if crate::progress::is_plain() {
        // One up-front line; the live bar is invisible off-TTY (hidden target).
        log::info!("simulate \u{b7} {backend}: running \u{2026}");
    }

    let result = util::simulate_compiled(&compiled, &model, &spec.sim_run, Some(&pb))
        .map(|traj| CellResult { spec, traj, model, event_log: None });

    pb.finish_and_clear();
    result
}
/// Effective replicate count. With explicit seeds, replicate count tracks
/// the seed-list length (each seed is one seed-slot); otherwise it is the
/// `ParamSource`'s replicate count (Draws) or 1.
fn effective_replicates(job: &SimulateJob) -> usize {
    match job.seeds.explicit() {
        Some(seeds) => seeds.len(),
        None => job.source.replicates(),
    }
}

/// The scenario list, with the empty case expanded to a single implicit
/// baseline (`ScenarioRef::Named("baseline")` resolves to the identity
/// patch via `resolve_scenario_ref`'s baseline exemption).
fn effective_scenarios(job: &SimulateJob) -> Vec<ScenarioRef> {
    if job.scenarios.is_empty() {
        vec![ScenarioRef::Named("baseline".to_string())]
    } else {
        job.scenarios.clone()
    }
}

/// Hard-error when a USER-AUTHORED draws file collides with a scenario on a
/// parameter both touch (spec §1.3 / the run-spec's "draws and sweeps are
/// different operations" rule).
///
/// A `--draws <file.tsv>` pins θ per draw from columns the user authored. A
/// scenario that ALSO sets/scales one of those parameters is ambiguous: the
/// user asserted the value two ways. Rather than silently letting the scenario
/// win (correct for *generated* draws, where the file is not authoritative),
/// this names the parameter, the scenario, and the file, and tells the user how
/// to fix it. Generated draws (`posterior`/`prior`/`uniform`) and sweeps do
/// NOT error — `explicit_file` is `false` there, and the scenario just wins.
fn check_explicit_draws_scenario_collision(job: &SimulateJob) -> Result<(), String> {
    use crate::sim_job::ParamSource;
    // Only an explicit user file triggers the collision check.
    let ParamSource::Draws { rows, explicit_file: Some(file_path), .. } = &job.source else {
        return Ok(());
    };
    // The set of parameters the draws FILE provides (union across rows — a
    // ragged file still flags any column that appears).
    let mut file_cols: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for row in rows {
        for k in row.keys() {
            file_cols.insert(k.as_str());
        }
    }
    if file_cols.is_empty() {
        return Ok(());
    }

    // A scenario's parameter footprint = its `set` ∪ `scale` ∪ composed-preset
    // keys (the shared `scenario_param_footprint`, also driving `fit predict`'s
    // scenario×sweep guard — so the two never disagree). The model is loaded
    // once, now that there is a draws file to check against.
    let scenarios = effective_scenarios(job);
    let (model, _) = crate::util::load_model(&job.model)?;
    for scenario in &scenarios {
        let scen_keys = crate::params_resolver::scenario_param_footprint(&model, scenario)?;
        let mut collisions: Vec<&str> = scen_keys.iter()
            .map(|k| k.as_str())
            .filter(|k| file_cols.contains(k))
            .collect();
        if !collisions.is_empty() {
            collisions.sort();
            collisions.dedup();
            let param_list = collisions.join(", ");
            return Err(format!(
                "scenario '{scen_name}' sets parameter(s) [{param_list}] that the \
                 --draws file '{file}' also provides as column(s). A draws file pins \
                 these parameters per draw, so applying a scenario that also sets them \
                 is ambiguous.\n  \
                 Fix: drop the column(s) [{param_list}] from the draws file, or use a \
                 scenario that does not touch them.",
                scen_name = scenario.name(),
                file = file_path.display(),
            ));
        }
    }
    Ok(())
}

/// Build the per-cell [`SimRun`], routing each M-layer input to the tier
/// it belongs to (spec §1.3, last wins): genuine `--param` CLI overrides
/// and an inline scenario's `set` go to `overrides` (the resolver's
/// `fixed_cli` tier 5); a draw row / sweep point goes to `point_overrides`
/// (the draw/sweep tier, below scenario), so a scenario `set`/`scale`
/// wins over a draw/sweep value while `--param` still wins over both.
///
/// The scenario is routed through either the named-preset path
/// (`scenario_name = Some`) — the resolver applies the preset's
/// enable/disable/set/scale/compose at tier 4 — or the ad-hoc inline path
/// (`scenario_name = None`, inline `set` overlaid via `overrides`). An
/// inline scenario's `set` is applied at the same logical scenario tier as
/// a preset by virtue of overriding the draw/sweep tier; `--param` (also
/// in `overrides`) still wins over both.
pub fn build_cell_sim_run(
    job: &SimulateJob,
    scenario: &ScenarioRef,
    point_overrides: &IndexMap<String, f64>,
    table_files: &HashMap<String, String>,
    process_seed: u64,
) -> SimRun {
    // Draw row / sweep point → the draw/sweep tier (below scenario).
    let point_overrides_map: HashMap<String, f64> =
        point_overrides.iter().map(|(k, v)| (k.clone(), *v)).collect();

    // `--param` CLI overrides → the highest M tier (`fixed_cli`).
    let overrides: HashMap<String, f64> =
        job.cli_overrides.iter().cloned().collect();

    let (scenario_name, adhoc_enable, adhoc_disable,
         scenario_inline_name, scenario_inline_set) = match scenario {
        // Named preset → params_resolver applies the preset's
        // enable/disable/set/scale/compose at the scenario tier (tier 4).
        ScenarioRef::Named(name) => {
            (Some(name.clone()), Vec::new(), Vec::new(), None, Vec::new())
        }
        // Inline ad-hoc patch → no preset; the inline enable/disable drive
        // the intervention filter and the inline `set` params resolve at
        // the SAME scenario tier as a preset (above the draw/sweep tier,
        // below `--param`) — NOT folded into `overrides`/`fixed_cli`, so an
        // inline scenario is identical to the equivalent named preset
        // (spec §1.3). (Inline scenarios carry only `set`, never `scale`.)
        ScenarioRef::Inline { name, enable, disable, params } => (
            None,
            enable.clone(),
            disable.clone(),
            Some(name.clone()),
            params.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>(),
        ),
    };

    SimRun {
        ir_path: job.model.clone(),
        params_files: job.params_files.clone(),
        overrides,
        point_overrides: point_overrides_map,
        set_vec_entries: job.set_vec_entries.clone(),
        table_files: table_files.clone(),
        scenario_name,
        adhoc_enable,
        adhoc_disable,
        scenario_inline_name,
        scenario_inline_set,
        scenario_inline_scale: Vec::new(),
        backend: job.backend,
        dt: job.dt,
        seed: process_seed,
        integrator: job.integrator, // gh#166: CLI --integrator override
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PIN the seed arithmetic at the unit level (the CLI-level PIN in
    /// `tests/determinism_pin.rs` is the integration guard; this catches a
    /// regression without spawning the binary).
    #[test]
    fn process_seed_explicit_indexes_directly() {
        let seeds = [7u64, 8, 9];
        assert_eq!(process_seed_for(Some(&seeds), 0, 0, 0, 3), 7);
        assert_eq!(process_seed_for(Some(&seeds), 0, 0, 1, 3), 8);
        assert_eq!(process_seed_for(Some(&seeds), 0, 0, 2, 3), 9);
    }

    #[test]
    fn process_seed_single_run_is_base() {
        assert_eq!(process_seed_for(None, 42, 0, 0, 1), 42);
    }

    #[test]
    fn process_seed_multi_run_xor_mixes() {
        let base = 42u64;
        let s00 = process_seed_for(None, base, 0, 0, 6);
        let s01 = process_seed_for(None, base, 0, 1, 6);
        let s10 = process_seed_for(None, base, 1, 0, 6);
        // The multi-run branch must route through the canonical mix.
        assert_eq!(s00, base); // point 0 rep 0 ⇒ base ^ 0 ^ 0
        assert_eq!(s01, util::mix_cell_seed(base, 0, 1));
        assert_eq!(s10, util::mix_cell_seed(base, 1, 0));
        assert_ne!(s01, s10);
    }
}
