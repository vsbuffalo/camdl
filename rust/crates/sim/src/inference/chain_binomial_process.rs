//! Chain-binomial process model implementing ProcessModel + DensityProcess.
//!
//! This is the only process backend that supports PGAS (via DensityProcess).
//! PF, IF2, and PMMH work through ProcessModel alone.

use std::sync::Arc;
use crate::chain_binomial::{step_one, StepScratch};
use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::rng::StatefulRng;
use super::traits::{ProcessModel, DensityProcess};
use super::types::ParticleState;

/// Chain-binomial process model.
///
/// Wraps a `CompiledModel` and delegates to `step_one` for simulation.
/// Implements `ProcessModel` (for PF, IF2, PMMH) and `DensityProcess`
/// (for PGAS). The only process backend that supports PGAS.
pub struct ChainBinomialProcess {
    pub compiled: Arc<CompiledModel>,
    /// Integrator step for this process. gh#53 — the CompiledModel
    /// stores dt-invariant `fire_times`; the per-run `fire_steps`
    /// view depends on the runtime dt and must be resolved with that
    /// value, not the compile-time `model.simulation.dt`. Resolution
    /// now happens per-step inside `step` (was pre-resolved at
    /// construction; broken for parametric event schedules per gh#69).
    pub(crate) dt: f64,
}

impl ChainBinomialProcess {
    /// Construct a process for a model with integrator step `dt`.
    /// `dt` is required because `fire_steps` (the runtime view of
    /// the model's intervention schedule) must be resolved with it
    /// (see gh#53). Reusing the same process across runs at
    /// different dts is unsupported — build a fresh process per
    /// dt; the gh#52 Richardson ladder already does this via
    /// `run_quick_pfilter_with_dt`'s per-rung config rebuild.
    pub fn new(compiled: Arc<CompiledModel>, dt: f64) -> Self {
        // fire_steps used to be pre-resolved here against default
        // params. That was incorrect for models with parametric event
        // schedules (`events { ... at [param] }`, gh#69): different
        // particles / different PMMH proposals carry different values
        // for `param`, so each `step` call needs fire_steps resolved
        // against THIS call's `params`. The pre-resolved value is
        // dropped; `step` re-resolves per call (linear walk over the
        // intervention list — negligible compared to a chain-binomial
        // step's propensity eval + multinomial draws).
        ChainBinomialProcess { compiled, dt }
    }
}

impl ProcessModel for ChainBinomialProcess {
    type State = ParticleState;
    type Scratch = StepScratch;

    fn n_compartments(&self) -> usize {
        self.compiled.int_local_to_global.len()
    }

    fn n_transitions(&self) -> usize {
        self.compiled.model.transitions.len()
    }

    fn initial_state(&self, params: &[f64]) -> Result<ParticleState, SimError> {
        let (init_int, _) = self.compiled.initial_state(params)?;
        let mut state = ParticleState::new(
            self.n_compartments(), self.n_transitions(),
        );
        state.counts.copy_from_slice(&init_int.counts);
        Ok(state)
    }

    fn step(
        &self,
        state: &mut ParticleState,
        params: &[f64],
        t: f64,
        dt: f64,
        rng: &mut StatefulRng,
        scratch: &mut StepScratch,
    ) -> Result<(), SimError> {
        // Re-resolve fire_steps per call from the caller's params.
        // For models without parametric event schedules, this is a
        // pure function of `dt` (and identical across calls); for
        // models WITH parametric schedules (gh#69), each particle /
        // PMMH proposal carries its own value of the schedule
        // parameter and gets its own fire_steps. Cost: linear walk
        // over the intervention list (typically O(few)) — small
        // compared to a chain-binomial step's per-transition
        // propensity eval and multinomial draws.
        let fire_steps = self.compiled.resolve_fire_steps(self.dt, params);
        // KNOWN LIMITATION (docs/dev/incidents/2026-06-07-chain-binomial-
        // stale-real-state.md, §inference scope): inference particles
        // (`ParticleState`) track integer counts only — there is no real
        // compartment state being advanced anywhere in the filter. We pass a
        // freshly-zeroed `RealState` here, so a model whose rate couples to a
        // real compartment is fit with that real value pinned at 0 (== its
        // init for cholera SIWR). For real-free models (`n_real == 0`) this is
        // an empty vector and the step is byte-identical to before. Fitting
        // real-coupled models on chain-binomial is a separate, larger fix
        // (the particle state must carry and RK4-advance the real reservoir).
        let mut real = crate::state::RealState::new(self.compiled.real_local_to_global.len());
        // `dt` is the realized substep the filter handed us (clipped under Exact
        // to land on an off-grid observation); `self.dt` is the nominal model grid
        // the `fire_steps` were built on, so it keys the event/intervention firing.
        step_one(
            &self.compiled,
            &mut state.counts,
            &mut state.flow_accumulators,
            &mut real,
            params, t, dt, self.dt, rng, scratch,
            &fire_steps,
        )
    }

    fn new_scratch(&self) -> StepScratch {
        StepScratch::new(&self.compiled)
    }

    fn try_compiled_model(&self) -> Option<&CompiledModel> {
        Some(&self.compiled)
    }
}

impl DensityProcess for ChainBinomialProcess {
    fn log_transition_density(
        &self,
        counts_before: &[i64],
        flows: &[u64],
        gammas: &[f64],
        params: &[f64],
        t: f64,
        dt: f64,
    ) -> Result<f64, SimError> {
        super::pgas::log_transition_density_substep(
            &self.compiled, counts_before, flows, gammas, params, t, dt,
        )
    }

    fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }
}
