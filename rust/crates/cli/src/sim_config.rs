//! `ResolvedSimConfig`: the one resolved configuration of a `camdl simulate`
//! invocation, from which both the run identity and the engine's settings are
//! derived (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "The shape").
//!
//! ```text
//! SimulateArgs ──(precedence, in run_simulate)──► ResolvedSimConfig
//!                                                   │             │
//!                                                   │             │  engine_settings()
//!                                                   │             ▼
//!                                                   │         EngineSettings ──► SimulateJob ──► SimRun (per cell)
//!                                                   │                                              │
//!                                                   │                              ResolvedCell::resolve
//!                                                   │                                              ▼
//!                         CasSink::cell_identity ◄──┴────────────────────────────────────── ResolvedCell
//!                                   │                                                              │
//!                                   ▼                                                              ▼
//!                       resolve_trajectory levels (run_id)                    util::simulate_compiled (engine)
//! ```
//!
//! **How a field reaches the identity.** A simulate invocation runs many cells,
//! and identity is per cell, so the configuration is not hashed as one value.
//! Each part reaches a level of the cell's identity
//! ([`crate::resolve::resolve_trajectory`]):
//!
//! - [`ResolvedModel::digest`] *is* the `model` level ([`ModelDigest`], derived
//!   `RunInput`), computed once from the model the engine loads.
//! - [`RunConfig`] is the run-wide part of the `config` level. [`RunConfig::cell_config`]
//!   destructures it exhaustively into a [`SimConfig`] (derived `RunInput`): a
//!   field added to `RunConfig` does not compile until it is named there, and a
//!   field added to `SimConfig` does not compile until it is set there. The
//!   three per-cell fields (`t_start`, `t_end`, `init_state`) come from the
//!   cell's [`CellWindow`].
//! - [`ResolvedSimConfig::arms`] are the `scenario` level of each arm.
//! - The per-cell parts — the window, the parameter values, the table digests,
//!   the seed and the restored origin state — come from the cell's
//!   [`ResolvedCell`], the value the engine runs.
//!
//! The proposal sketches `ResolvedSimConfig` deriving `RunInput` with each level
//! nested inside it. It does not: a simulate invocation has many cells, and its
//! run-wide configuration is not any one cell's identity, so what is hashed is
//! each cell's levels.
//!
//! **The per-cell value.** [`ResolvedCell::resolve`] resolves a cell's `SimRun`
//! through the engine's own resolution (`util::resolve_loaded_run`): the model
//! the cell executes, its parameter provenance, its table digests, seed and
//! origin state, and its backend, step, sampler and evaluator settings. The
//! engine (`util::simulate_compiled`, `ResolvedCell::compile`) runs from it
//! and hands it back in `engine::CellResult`; `CasSink::merge_cell` keys the
//! written leaf from that instance (`CasSink::cell_identity`), whose
//! destructure binds no field to `_`, so a field added to the cell does not
//! compile there until it is named and used. The window and parameter values
//! are read off the resolved model, so an input that moves the horizon or a
//! parameter reaches the identity because it reaches the engine. What the
//! identity keys from the invocation rather than the cell — the backend, step
//! and evaluator settings ([`RunConfig`]), the intervention set (the scenario
//! arm) — is checked against the cell, and a cell that differs is refused, as
//! is one with required snapshot times. A lookup before anything runs (the
//! cache-hit check) resolves the cell over the base model; the written leaf's
//! identity is checked against that pre-run one.
//!
//! Two limits remain. The model level is computed once from the sink's base
//! model (`util::load_run_model` of the job's run), not from each cell's own
//! load, so `ir_path`, `integrator` and `obs_anchors` are keyed from the
//! base: every cell copies them from the job, but a cell carrying other
//! values would not be told apart. And a `SimRun` field that changes the
//! resolved model other than through its parameter values, window or
//! interventions reaches the engine without reaching the identity.
//! `ResolvedCell::resolve` names every `SimRun` field, but a field named
//! there as `field: _` compiles, so the name is a review point, not a proof.
//!
//! **What is not here.** The thread count, the store root and `--force` are
//! [`Execution`]: how the result is computed, never what. `--label`, the command
//! line and the mirrors (`-o`, `--obs`, `--dates`, …) are provenance and
//! presentation and are not fields of this configuration, so they cannot reach
//! the engine through it.
//!
//! `sim` does not depend on `runid`, so the engine cannot be made by type to
//! accept only a resolved configuration. [`ResolvedSimConfig::engine_settings`]
//! is the single conversion, and the differential tests in
//! `tests/simulate_input_classes.rs` are the assurance.

use runid::inputs::{
    Backend, CalendarMode, EngineIdentity, InitStateDigest, ModelDigest, ResolvedOutputSchedule,
    SimConfig,
};
use runid::{FiniteF64, ResolveError};

use crate::args::types::ForwardBackend;
use crate::cas_sink::ResolvedEntry;
use crate::util::OutputColumns;

/// The resolved configuration of one `simulate` invocation.
pub(crate) struct ResolvedSimConfig {
    pub model: ResolvedModel,
    pub run: RunConfig,
    /// Each scenario arm's hash-relevant delta and horizon, in the order the
    /// arms were named (one implicit arm when none was).
    pub arms: Vec<ResolvedEntry>,
}

/// The model the engine starts every cell from, and its identity level.
pub(crate) struct ResolvedModel {
    /// The base model as `util::load_run_model` loads it for the engine:
    /// observation anchors substituted, `--integrator` applied, no parameter
    /// values or scenario applied (those are per cell).
    pub ir: ir::Model,
    /// The `model` level, computed once from `ir`. The compile settings
    /// (`util::CompileSettings`) change the IR camdlc emits, so they reach
    /// the identity here rather than as a configuration field.
    pub digest: ModelDigest,
    /// The engine identity folded into `digest` (the build and the engine
    /// switches simulate keys), also used by the ensemble's model level and
    /// recorded on every leaf.
    pub engine: EngineIdentity,
    /// The readable path label of the `model` level (provenance).
    pub stem: String,
    /// The model file as named on the command line (provenance: the leaf's
    /// `source_paths`, and the reuse warning for a renamed file).
    pub path: String,
}

impl ResolvedModel {
    pub fn new(ir: ir::Model, ir_version: &str, engine: EngineIdentity, path: &str) -> Self {
        let digest = ModelDigest::from_model(&ir, ir_version.to_string(), engine.clone());
        ResolvedModel {
            ir,
            digest,
            engine,
            stem: crate::hashing::path_stem_slug(path).unwrap_or_else(|| "model".to_string()),
            path: path.to_string(),
        }
    }
}

/// The run-wide part of the `config` level: every resolved setting that is the
/// same for every cell of the invocation.
#[derive(Debug, Clone)]
pub(crate) struct RunConfig {
    pub backend: Backend,
    pub dt: FiniteF64,
    /// The output schedule of the base model (`--output-every` is lowered into
    /// the model before it is loaded, so it is here and in the model level).
    pub output: ResolvedOutputSchedule,
    pub calendar: CalendarMode,
    /// `--allow-degenerate-rates`: a collapsed rate is zero instead of an error.
    pub allow_degenerate_rates: bool,
    /// `--allow-forcing-extrapolation`: hold a forcing's end value outside its
    /// range instead of refusing the run (D44).
    pub allow_forcing_extrapolation: bool,
    /// `--no-flows` / `--columns`: which trajectory columns the leaf stores.
    pub view: OutputColumns,
}

/// The per-cell part of the `config` level.
pub(crate) struct CellWindow {
    pub t_start: FiniteF64,
    pub t_end: FiniteF64,
    /// The forecast-origin state the cell restored (`--init-state`).
    pub init_state: Option<InitStateDigest>,
}

impl CellWindow {
    pub fn new(t_start: f64, t_end: f64, init_state: Option<InitStateDigest>) -> Result<Self, ResolveError> {
        Ok(CellWindow { t_start: finite(t_start)?, t_end: finite(t_end)?, init_state })
    }
}

impl RunConfig {
    pub fn new(
        backend: ForwardBackend,
        dt: f64,
        output: &ir::model::OutputSchedule,
        allow_degenerate_rates: bool,
        allow_forcing_extrapolation: bool,
        view: OutputColumns,
    ) -> Result<Self, ResolveError> {
        Ok(RunConfig {
            backend: backend_level(backend),
            dt: finite(dt)?,
            output: resolve_output(output)?,
            // Placeholder until the calendar-time work gives the resolver a
            // concrete mode (M2 decision: minimal, provisional).
            calendar: CalendarMode::Numeric,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            view,
        })
    }

    /// The `config` level of one cell: this run-wide configuration and the
    /// cell's window. Exhaustive on both sides (see the module doc).
    pub fn cell_config(&self, window: CellWindow) -> SimConfig {
        let RunConfig {
            backend,
            dt,
            output,
            calendar,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            view: OutputColumns { no_flows, allow },
        } = self.clone();
        let CellWindow { t_start, t_end, init_state } = window;
        SimConfig {
            backend,
            dt,
            t_start,
            t_end,
            output,
            calendar,
            allow_degenerate_rates,
            allow_forcing_extrapolation,
            no_flows,
            columns: allow,
            init_state,
        }
    }
}

/// What the engine is told, derived from a [`ResolvedSimConfig`] by
/// [`ResolvedSimConfig::engine_settings`] and from nowhere else. Derived once
/// per invocation: the forcing policy it holds is the record of what the run
/// held, so the CAS sink writes from this instance.
#[derive(Debug, Clone)]
pub(crate) struct EngineSettings {
    pub backend: ForwardBackend,
    pub dt: f64,
    /// The evaluator settings every cell's compiled model carries
    /// (`SimRun::eval`, `CompiledModel::with_eval_settings`).
    pub eval: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, as the policy the run's forcing notes
    /// are routed into and its leaves' records name.
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
        let RunConfig { backend, dt, allow_degenerate_rates, allow_forcing_extrapolation, .. } = &self.run;
        EngineSettings {
            backend: forward_backend(*backend),
            dt: dt.get(),
            eval: crate::env::eval_settings(*allow_degenerate_rates),
            forcing: crate::forcing_extrapolation::ForcingPolicy::new(*allow_forcing_extrapolation),
        }
    }
}

/// One cell, resolved: the value both the cell's engine run
/// (`util::simulate_compiled`) and its identity (`CasSink::cell_identity`)
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
    /// Content digests of the cell's `--table` files, sorted by table name
    /// (the values themselves are in `run.model`).
    pub tables: Vec<runid::inputs::DataDigest>,
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
            table_files,
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
            tables: table_digests(table_files)?,
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

/// Content digests of the `--table` files, sorted by table name, for the
/// `params` level (`ResolvedParams.tables`). Each file's bytes are hashed, never
/// its path, so a regenerated or edited table re-keys the run. No tables, an
/// empty list: a plain `simulate` is unaffected.
fn table_digests(
    table_files: &std::collections::HashMap<String, String>,
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

/// How the result is computed, never what: not a field of
/// [`ResolvedSimConfig`], and checked by the `execution` differential tests to
/// leave the run id and the stored bytes unchanged.
#[derive(Debug, Clone)]
pub(crate) struct Execution {
    /// Rayon threads for the cell grid (`--parallel`; 1 = sequential).
    pub threads: usize,
    /// The store root (`--output-dir`).
    pub store_root: String,
    /// `--force`: recompute and displace a stored leaf of the same identity.
    pub force: bool,
}

/// The resolved `runid` backend of a `--backend` value.
pub(crate) fn backend_level(b: ForwardBackend) -> Backend {
    match b {
        ForwardBackend::Gillespie => Backend::Gillespie,
        ForwardBackend::ChainBinomial => Backend::ChainBinomial,
        ForwardBackend::Ode => Backend::Ode,
    }
}

/// The engine backend of a resolved `runid` backend; the inverse of [`backend_level`].
pub(crate) fn forward_backend(b: Backend) -> ForwardBackend {
    match b {
        Backend::Gillespie => ForwardBackend::Gillespie,
        Backend::ChainBinomial => ForwardBackend::ChainBinomial,
        Backend::Ode => ForwardBackend::Ode,
    }
}

pub(crate) fn finite(x: f64) -> Result<FiniteF64, ResolveError> {
    FiniteF64::new(x).map_err(ResolveError::from)
}

fn resolve_output(sched: &ir::model::OutputSchedule) -> Result<ResolvedOutputSchedule, ResolveError> {
    use ir::model::OutputSchedule as O;
    Ok(match sched {
        O::Regular(r) => ResolvedOutputSchedule::Regular { start: finite(r.start)?, step: finite(r.step)? },
        O::AtTimes(ts) => {
            ResolvedOutputSchedule::AtTimes(ts.iter().map(|t| finite(*t)).collect::<Result<_, _>>()?)
        }
    })
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
    fn backend_level_and_forward_backend_are_inverse() {
        for b in [ForwardBackend::Gillespie, ForwardBackend::ChainBinomial, ForwardBackend::Ode] {
            assert_eq!(forward_backend(backend_level(b)), b);
        }
    }

    #[test]
    fn engine_settings_come_from_the_run_config() {
        let out = ir::model::OutputSchedule::Regular(ir::model::RegularOutputSchedule { start: 0.0, step: 1.0 });
        let run = RunConfig::new(ForwardBackend::Ode, 0.25, &out, true, true, view(false, &[])).unwrap();
        let cfg = ResolvedSimConfig {
            model: ResolvedModel::new(crate::resolve::tiny_model(), "0.7", EngineIdentity::of_build("0.3.0+test"), "sir.camdl"),
            run,
            arms: Vec::new(),
        };
        let EngineSettings { backend, dt, eval, forcing } = cfg.engine_settings();
        assert_eq!(backend, ForwardBackend::Ode);
        assert_eq!(dt, 0.25);
        assert_eq!(eval, crate::env::eval_settings(true));
        assert!(forcing.allowed());
    }

    /// Where each field of the resolved configuration comes from: the inputs
    /// (command-line fields, and `env:`-prefixed environment fields) it is
    /// resolved from. The exhaustive destructure below does not compile until a
    /// new field is named, so a field cannot join the configuration without a
    /// row here.
    const SOURCES: &[(&str, &[&str])] = &[
        ("model.ir", &["model", "integrator", "every", "no_licm", "to", "fit", "env:no_licm", "env:no_constant_fold"]),
        ("model.digest", &["model", "integrator", "every", "no_licm", "to", "fit", "env:no_licm", "env:no_constant_fold"]),
        ("model.engine", &["env:eval_unresolved", "env:no_binding_cache", "env:experimental"]),
        ("run.backend", &["backend", "params", "fit"]),
        ("run.dt", &["dt", "params", "fit"]),
        ("run.output", &["model", "every"]),
        ("run.calendar", &[]),
        ("run.allow_degenerate_rates", &["allow_degenerate_rates"]),
        ("run.allow_forcing_extrapolation", &["allow_forcing_extrapolation"]),
        ("run.view", &["no_flows", "columns"]),
        ("arms", &["scenarios", "enable", "disable", "to", "fit", "draws"]),
    ];

    /// Fields that are not configuration: readable labels of the model level
    /// and the model file's name, recorded in `run.json`, never hashed.
    const NON_SEMANTIC: &[&str] = &["model.stem", "model.path"];

    /// A provenance, presentation or execution input is not a field of the
    /// resolved configuration, so it cannot reach the engine through it.
    #[test]
    fn presentation_and_provenance_are_not_configuration() {
        let ResolvedSimConfig {
            model: ResolvedModel { ir: _, digest: _, engine: _, stem: _, path: _ },
            run:
                RunConfig {
                    backend: _,
                    dt: _,
                    output: _,
                    calendar: _,
                    allow_degenerate_rates: _,
                    allow_forcing_extrapolation: _,
                    view: _,
                },
            arms: _,
        } = ResolvedSimConfig {
            model: ResolvedModel::new(crate::resolve::tiny_model(), "0.7", EngineIdentity::of_build("e"), "sir.camdl"),
            run: RunConfig::new(
                ForwardBackend::ChainBinomial,
                1.0,
                &crate::resolve::tiny_model().output.times,
                false,
                false,
                view(false, &[]),
            )
            .unwrap(),
            arms: Vec::new(),
        };
        let destructured = [
            "model.ir", "model.digest", "model.engine", "model.stem", "model.path", "run.backend", "run.dt",
            "run.output", "run.calendar", "run.allow_degenerate_rates", "run.allow_forcing_extrapolation",
            "run.view", "arms",
        ];
        let listed: Vec<&str> = SOURCES.iter().map(|(f, _)| *f).chain(NON_SEMANTIC.iter().copied()).collect();
        assert_eq!(
            listed.iter().copied().collect::<std::collections::BTreeSet<_>>(),
            destructured.iter().copied().collect(),
            "every field of the resolved configuration has a SOURCES row or is NON_SEMANTIC"
        );

        use crate::input_class::InputClass;
        for (field, inputs) in SOURCES {
            for input in *inputs {
                let class = match input.strip_prefix("env:") {
                    Some(f) => crate::env::VARS.iter().find(|v| v.field == f).map(|v| v.class),
                    None => crate::args::SimulateArgs::INPUTS.iter().find(|(i, _)| i == input).map(|(_, c)| *c),
                };
                let class = class.unwrap_or_else(|| panic!("{field}: `{input}` is not an input of simulate"));
                assert!(
                    matches!(class, InputClass::Keyed(_) | InputClass::Expands),
                    "{field} is resolved from `{input}`, whose class is {class:?}: only a keyed or \
                     expanding input may reach the resolved configuration"
                );
            }
        }
    }
}
