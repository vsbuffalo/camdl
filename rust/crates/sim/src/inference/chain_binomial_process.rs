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
///
/// Holds no `dt`: the integrator step arrives per call (`step`/`density` take
/// `dt`), and effect firing is decided by the driver from the run's
/// [`crate::effect_plan::EffectPlan`] — there is no per-process firing view.
pub struct ChainBinomialProcess {
    pub compiled: Arc<CompiledModel>,
}

impl ChainBinomialProcess {
    /// Construct a process for `compiled`. The integrator step is supplied per
    /// call (`step`/`density` take `dt`); the process stores none.
    pub fn new(compiled: Arc<CompiledModel>) -> Self {
        ChainBinomialProcess { compiled }
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

    fn initial_state_draw(
        &self,
        params: &[f64],
        rng: &mut StatefulRng,
    ) -> Result<ParticleState, SimError> {
        let (init_int, _) = self.compiled.initial_state_draw(params, rng)?;
        // `acc` sized 0 here: the process does not know `n_interval_streams`
        // (the obs model owns it). The filter copies only `init.counts` into the
        // swarm and allocates each swarm state's `acc` sized from
        // `obs_model.n_interval_streams()`, so this init state's `acc` is never
        // read.
        let mut state = ParticleState::new(
            self.n_compartments(), self.n_transitions(), 0,
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
        per_eval: Option<&[f64]>,
        rng: &mut StatefulRng,
        scratch: &mut StepScratch,
        due_effects: &[usize],
    ) -> Result<(), SimError> {
        // The driver decided due-ness from the effect plan: the schedule lands
        // on each of the plan's firing instants and the cursor reports that
        // instant's batch here — empty off a boundary. Split it by kind into the
        // lifecycle halves: events at PROPOSE (fused with the kernel draw),
        // interventions at INTERVENE. step_one applies what we put here.
        crate::effects::split_due_batch(&self.compiled, due_effects, &mut scratch.effect_batch);
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
        // to land on an off-grid observation).
        step_one(
            &self.compiled,
            &mut state.counts,
            &mut state.flow_accumulators,
            &mut real,
            // gh#272 LICM: the scratch staged at the filter's θ-stable boundary
            // (or `None` ⇒ on-demand). Threaded, NOT staged here (per-substep
            // staging would defeat the hoist).
            params, t, dt, per_eval, rng, scratch,
        )
    }

    fn new_scratch(&self) -> StepScratch {
        StepScratch::new(&self.compiled)
    }

    fn try_compiled_model(&self) -> Option<&CompiledModel> {
        Some(&self.compiled)
    }

    fn apply_origin(
        &self,
        state: &mut ParticleState,
        plan: &crate::effect_plan::EffectPlan,
        params: &[f64],
        t_start: f64,
        dt: f64,
    ) -> Result<(), SimError> {
        if plan.at_origin().is_empty() {
            return Ok(());
        }
        // Integer counts only, as in `step`: the particle carries no real
        // reservoir (the known limitation below), so the real state is zeroed.
        let mut int_s = crate::state::IntState::from_vec(state.counts.clone());
        let mut real = crate::state::RealState::new(self.compiled.real_local_to_global.len());
        crate::effects::apply_origin_batch_discrete(
            &self.compiled, plan, &mut int_s, &mut real, params, t_start, dt,
            self.compiled.balance.as_ref(),
        )?;
        state.counts.copy_from_slice(&int_s.counts);
        Ok(())
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
        per_eval: Option<&[f64]>,
    ) -> Result<f64, SimError> {
        super::pgas::log_transition_density_substep(
            &self.compiled, counts_before, flows, gammas, params, t, dt, per_eval,
        )
    }

    fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }
}
