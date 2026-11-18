//! `ResolvedSimConfig`: the one resolved configuration of a `camdl simulate`
//! invocation, from which both the run identity and the engine's settings are
//! derived (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "The shape").
//!
//! ```text
//! SimulateArgs ──(precedence, in run_simulate)──► ResolvedSimConfig
//!                                                   │             │
//!                         cell_identity (per cell)  │             │  engine_settings()
//!                                                   ▼             ▼
//!                                   TrajectoryInput levels    EngineSettings
//!                                   (run_id)                  (SimulateJob, evaluator settings)
//! ```
//!
//! **How a field reaches the identity.** A simulate invocation runs many cells,
//! and identity is per cell, so the configuration is not hashed as one value.
//! Each part reaches a level of the cell's [`runid::inputs::TrajectoryInput`]:
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
//! - The `params` and `seed` levels are per cell, resolved by the resolver call
//!   and seed arithmetic the engine itself makes (`CasSink::cell_resolve`).
//!
//! The proposal sketches `ResolvedSimConfig` deriving `RunInput` with each level
//! nested inside it. That would hash new bytes (a nested type tag per level)
//! and re-key every stored leaf; this keeps the hashed types and their bytes
//! exactly, and gets include-by-default from the exhaustive destructure instead.
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
    Backend, CalendarMode, EngineVersion, InitStateDigest, ModelDigest, ResolvedOutputSchedule,
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

/// Compile-time settings: the first thing a `simulate` resolves, because the
/// compile that produces [`ResolvedModel::ir`] reads them. They change the IR
/// camdlc emits, so they reach the identity through [`ResolvedModel::digest`]
/// and are not a field of [`ResolvedSimConfig`] of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompileSettings {
    /// `--no-licm`: compile without loop-invariant code motion.
    pub licm_disabled: bool,
}

impl CompileSettings {
    /// Install the process-wide compile setting the IR loader reads. Called once
    /// per invocation, before the first compile.
    pub fn install(self) {
        crate::util::set_licm_disabled(self.licm_disabled);
    }
}

/// The model the engine starts every cell from, and its identity level.
pub(crate) struct ResolvedModel {
    /// The base model as `util::load_run_model` loads it for the engine:
    /// observation anchors substituted, `--integrator` applied, no parameter
    /// values or scenario applied (those are per cell).
    pub ir: ir::Model,
    /// The `model` level, computed once from `ir`.
    pub digest: ModelDigest,
    /// The engine identity string folded into `digest`, also used by the
    /// ensemble's model level.
    pub engine: String,
    /// The readable path label of the `model` level (provenance).
    pub stem: String,
    /// The model file as named on the command line (provenance: the leaf's
    /// `source_paths`, and the reuse warning for a renamed file).
    pub path: String,
}

impl ResolvedModel {
    pub fn new(ir: ir::Model, ir_version: &str, engine: &str, path: &str) -> Self {
        let digest = ModelDigest::from_model(&ir, ir_version.to_string(), EngineVersion(engine.to_string()));
        ResolvedModel {
            ir,
            digest,
            engine: engine.to_string(),
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
/// [`ResolvedSimConfig::engine_settings`] and from nowhere else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EngineSettings {
    pub backend: ForwardBackend,
    pub dt: f64,
    pub allow_degenerate_rates: bool,
    pub allow_forcing_extrapolation: bool,
}

impl EngineSettings {
    /// Install the process-wide evaluator settings the engine and its run-note
    /// reporter read, while those still exist (phase 3 step 3 removes them).
    /// Called once, before any cell runs.
    pub fn install_evaluator_settings(&self) {
        sim::eval_stats::set_allow_degenerate_rates(self.allow_degenerate_rates);
        crate::forcing_extrapolation::set_allowed(self.allow_forcing_extrapolation);
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
            allow_degenerate_rates: *allow_degenerate_rates,
            allow_forcing_extrapolation: *allow_forcing_extrapolation,
        }
    }
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
            model: ResolvedModel::new(crate::resolve::tiny_model(), "0.7", "0.3.0+test", "sir.camdl"),
            run,
            arms: Vec::new(),
        };
        assert_eq!(
            cfg.engine_settings(),
            EngineSettings {
                backend: ForwardBackend::Ode,
                dt: 0.25,
                allow_degenerate_rates: true,
                allow_forcing_extrapolation: true,
            }
        );
    }
}
