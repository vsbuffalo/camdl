//! `ResolvedSimConfig`: the one resolved configuration of a `camdl simulate`
//! invocation, from which both the cells' identities and the engine's
//! settings are derived.
//!
//! ```text
//! SimulateArgs ──(precedence, in run_simulate)──► ResolvedSimConfig ──engine_settings()──► EngineSettings
//!                                                       │                                      │
//!                                                       │                                      ▼
//!                                                       │                 SimulateJob ──► SimRun (per cell)
//!                                                       │                                      │
//!                                                       ▼                         ResolvedCell::resolve
//!                                   sim_command::plan_cell ◄──────────────────────────── ResolvedCell
//!                                     (SimIdentity, run id)                                    │
//!                                                                                              ▼
//!                                                                          util::simulate_compiled (engine)
//! ```
//!
//! A simulate invocation runs many cells, and identity is per cell, so the
//! configuration is not hashed as one value. The run-wide settings
//! ([`RunConfig`]) and the cell's [`ResolvedCell`], the value the engine runs,
//! together give each cell's identity (`crate::sim_store::SimIdentity`): its
//! window and parameter values are read off the cell's resolved model, so an
//! input that moves the horizon or a parameter reaches the identity because it
//! reaches the engine. What the identity keys from the invocation rather than
//! the cell (the backend, step and evaluator settings) is checked against the
//! cell, and a cell that differs is refused, as is one with required snapshot
//! times or another binomial sampler.
//!
//! The thread count is [`Execution`]: how the cells are computed, never what.
//! `--label`, `--store` and `--force` are requests to the store, and the
//! exports (`-o`, `--obs`, `--dates`, …) are renderings of stored files; none
//! is a field of this configuration, so none can reach the engine through it.

use crate::args::types::ForwardBackend;
use crate::util::OutputColumns;

/// The resolved configuration of one `simulate` invocation.
pub(crate) struct ResolvedSimConfig {
    pub model: ResolvedModel,
    pub run: RunConfig,
}

/// The model the engine starts every cell from.
pub(crate) struct ResolvedModel {
    /// The base model as `util::load_run_model` loads it for the engine:
    /// observation anchors substituted, the start override and `--integrator`
    /// applied, no parameter values or scenario applied (those are per cell).
    pub ir: ir::Model,
    /// The model file's stem: the label of the model's directory.
    pub stem: String,
}

impl ResolvedModel {
    pub fn new(ir: ir::Model, path: &str) -> Self {
        ResolvedModel { ir, stem: crate::hashing::path_stem_slug(path).unwrap_or_else(|| "model".to_string()) }
    }
}

/// Every resolved setting that is the same for every cell of the invocation.
#[derive(Debug, Clone)]
pub(crate) struct RunConfig {
    pub backend: ForwardBackend,
    pub dt: f64,
    /// `--allow-degenerate-rates`: a collapsed rate is zero instead of an error.
    pub allow_degenerate_rates: bool,
    /// `--allow-forcing-extrapolation`: hold a forcing's end value outside its
    /// range instead of refusing the run.
    pub allow_forcing_extrapolation: bool,
    /// `--no-flows` / `--columns`: which trajectory columns a cell stores.
    pub view: OutputColumns,
}

/// What the engine is told, derived from a [`ResolvedSimConfig`] by
/// [`ResolvedSimConfig::engine_settings`] and from nowhere else. Derived once
/// per invocation: the forcing policy it holds is the record of what the run
/// held.
#[derive(Debug, Clone)]
pub(crate) struct EngineSettings {
    pub backend: ForwardBackend,
    pub dt: f64,
    /// The evaluator settings every cell's compiled model carries
    /// (`SimRun::eval`, `CompiledModel::with_eval_settings`).
    pub eval: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, as the policy the run's forcing notes
    /// are routed into.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
}

impl EngineSettings {
    /// Route the run's forcing notes into its policy until the guard drops.
    /// Called once, before any cell runs, and held for the command.
    pub fn install_forcing_policy(&self) -> crate::forcing_extrapolation::Installed {
        self.forcing.install()
    }
}

impl ResolvedSimConfig {
    /// The engine's settings: the one conversion from the resolved
    /// configuration to what runs.
    pub fn engine_settings(&self) -> EngineSettings {
        let RunConfig { backend, dt, allow_degenerate_rates, allow_forcing_extrapolation, view: _ } = &self.run;
        EngineSettings {
            backend: *backend,
            dt: *dt,
            eval: crate::env::eval_settings(*allow_degenerate_rates),
            forcing: crate::forcing_extrapolation::ForcingPolicy::new(*allow_forcing_extrapolation),
        }
    }
}

/// One cell, resolved: the value both the cell's engine run
/// (`util::simulate_compiled`) and its identity (`sim_command::plan_cell`)
/// are built from — for a cell that runs, the same instance, carried back in
/// `engine::CellResult`.
///
/// Built by [`ResolvedCell::resolve`] from the cell's `SimRun`, through the
/// engine's own resolution (`util::resolve_loaded_run`). Both consumers
/// destructure it exhaustively, and the identity's destructure binds no field
/// to `_`, so a field added here does not compile there until it is named and
/// used. A `SimRun` field reaches the engine only through this value: either as
/// one of its fields or through the resolved `run.model`, whose parameter
/// values, window and interventions the identity reads or checks.
pub(crate) struct ResolvedCell {
    /// The parameter resolution of the cell; `run.model` is the model the cell
    /// executes — the base model with its horizon, start, scenario, parameter
    /// values and required snapshots applied — and `run.params` / `run.context`
    /// say where each value came from.
    pub run: crate::params_resolver::ResolvedParameters,
    /// The process seed the trajectory RNG starts from.
    pub seed: u64,
    /// The forecast-origin state the cell restores (`--init-state`).
    pub init_state: Option<crate::util::CellInitState>,
    /// The restart as the observation emitters take it: the origin, with the
    /// model's own start as it stood before the restart moved `run.model`'s
    /// (gh#1081). `None` for a cell that does not restart.
    pub restart: Option<crate::obs_emit::Restart>,
    pub backend: ForwardBackend,
    pub dt: f64,
    pub binomial: sim::rng::BinomialAlgorithm,
    /// The evaluator settings the compiled model carries ([`ResolvedCell::compile`]).
    pub eval: sim::EvalSettings,
    /// Snapshot times the run must record beyond the model's own schedule
    /// (`fit predict`'s forecast rows); applied to `run.model`, and refused by
    /// the store identity, which has no route for them.
    pub required_output_times: Vec<f64>,
}

impl ResolvedCell {
    /// Resolve the cell `run` over `base`, the model `util::load_run_model`
    /// returns for it.
    pub fn resolve(base: ir::Model, run: &crate::util::SimRun) -> Result<Self, String> {
        let crate::util::SimRun {
            // These shape `run.model`, through `util::resolve_loaded_run`
            // below and the `util::load_run_model` that produced `base`. The
            // identity reads the model level from the same load, and the
            // parameter values and window from the resolved model.
            ir_path: _,
            obs_anchors: _,
            integrator: _,
            params_files: _,
            overrides: _,
            point_overrides: _,
            fit_fixed: _,
            set_vec_entries: _,
            scenario_name: _,
            scenario_inline_set: _,
            scenario_inline_scale: _,
            adhoc_enable: _,
            adhoc_disable: _,
            fit_base: _,
            t_end_override: _,
            t_start_override: _,
            // Display only: the scenario label and the source a notice names.
            scenario_inline_name: _,
            overrides_label: _,
            // Fields of the resolved cell.
            table_files: _,
            init_state,
            backend,
            dt,
            seed,
            binomial,
            eval,
            required_output_times,
        } = run;
        // Read before `resolve_loaded_run` moves the start to the origin.
        // `base` is the model `util::load_run_model` returned, so a
        // `simulate { from = first_obs }` anchor is already a number here.
        let restart = init_state.as_ref().map(|init| crate::obs_emit::Restart {
            origin: init.origin_t,
            model_start: base.simulation.t_start,
        });
        Ok(ResolvedCell {
            run: crate::util::resolve_loaded_run(base, run)?,
            seed: *seed,
            init_state: init_state.clone(),
            restart,
            backend: *backend,
            dt: *dt,
            binomial: *binomial,
            eval: *eval,
            required_output_times: required_output_times.clone(),
        })
    }

    /// The model the cell executes, compiled under its evaluator settings.
    pub fn compile(&self) -> Result<sim::CompiledModel, String> {
        sim::CompiledModel::new(self.run.model.clone())
            .map_err(|e| format!("model compile error: {:?}", e))
            .map(|c| c.with_eval_settings(self.eval))
    }
}

/// How the result is computed, never what: not a field of
/// [`ResolvedSimConfig`], and checked by the class checks to leave the run
/// ids and the stored bytes unchanged.
#[derive(Debug, Clone)]
pub(crate) struct Execution {
    /// Rayon threads for the cell grid (`--parallel`; 1 = sequential).
    pub threads: usize,
}

/// The column view as a set, for tests that build a [`RunConfig`] by hand.
#[cfg(test)]
pub(crate) fn view(no_flows: bool, columns: &[&str]) -> OutputColumns {
    OutputColumns { no_flows, allow: columns.iter().map(|c| c.to_string()).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_settings_come_from_the_run_config() {
        let run = RunConfig {
            backend: ForwardBackend::Ode,
            dt: 0.25,
            allow_degenerate_rates: true,
            allow_forcing_extrapolation: true,
            view: view(false, &[]),
        };
        let cfg = ResolvedSimConfig { model: ResolvedModel::new(crate::resolve::tiny_model(), "sir.camdl"), run };
        let EngineSettings { backend, dt, eval, forcing } = cfg.engine_settings();
        assert_eq!(backend, ForwardBackend::Ode);
        assert_eq!(dt, 0.25);
        assert_eq!(eval, crate::env::eval_settings(true));
        assert!(forcing.allowed());
    }
}
