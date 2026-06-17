//! Core inference traits: ProcessModel, DensityProcess, ObservationModel.
//!
//! These traits define the interface between inference algorithms and
//! model implementations. All algorithms (PF, IF2, PMMH, PGAS) program
//! against these traits rather than concrete closure types.
//!
//! Design rationale:
//! - `ProcessModel`: any stochastic process that can simulate forward.
//!   PF, IF2, PMMH all need this.
//! - `DensityProcess`: extends ProcessModel with transition density
//!   evaluation. Only chain-binomial implements this. PGAS requires it
//!   (compile-time enforcement via trait bound).
//! - `ObservationModel`: maps latent state to data likelihood. Always
//!   takes `params` — IF2 needs per-particle params, PGAS/PMMH need
//!   updated params each sweep/step for obs-level parameter estimation.
//! - `Resettable`: state lifecycle — reset flow accumulators between
//!   observation intervals. On the state type, not the obs model.

use crate::error::SimError;
use crate::rng::StatefulRng;

/// Trait bound for particle state types that support observation-interval resets.
///
/// After each observation time, the inference algorithm resets accumulators
/// (e.g., flow counters for incidence projection). This is a STATE lifecycle
/// concern — it belongs on the state type, not on the observation model.
pub trait Resettable {
    /// Reset observation-interval accumulators to zero.
    fn reset_accumulators(&mut self);
}

/// A stochastic process model that can simulate forward in time.
///
/// Owns the model structure (compartments, transitions, stoichiometry)
/// and provides methods to initialize state, advance by one timestep,
/// and allocate reusable scratch buffers.
///
/// Generic over State and Scratch to support different backends
/// (chain-binomial, Gillespie) without boxing.
pub trait ProcessModel: Send + Sync {
    /// Particle state type.
    /// - `Clone`: resampling copies particles.
    /// - `Send`: rayon propagates particles across threads.
    /// - `Resettable`: algorithms reset flow accumulators between observations.
    type State: Clone + Send + Resettable;

    /// Pre-allocated scratch buffers, one per particle.
    /// Avoids heap allocation in the inner loop.
    type Scratch: Send;

    /// Number of integer compartments (for sizing).
    fn n_compartments(&self) -> usize;

    /// Number of transitions (for sizing flow accumulators).
    fn n_transitions(&self) -> usize;

    /// Create the initial state from parameters.
    fn initial_state(&self, params: &[f64]) -> Result<Self::State, SimError>;

    /// Advance state by one timestep.
    ///
    /// This is the hot path — called n_particles × n_substeps × n_obs times.
    /// Must not allocate. Uses scratch buffers for temporaries.
    ///
    /// `due_effects` lists EVERY effect — always-active events AND scheduled
    /// interventions — that fires at the END of this substep (`t + dt`), decided
    /// CURSOR-keyed by the driver from the timeline's effect boundaries
    /// (`timeline_effects().batches[effect_idx]`). It is empty on every
    /// off-boundary substep. The process splits it by kind via
    /// [`crate::effects::split_due_batch`]: events at PROPOSE (fused with the
    /// kernel draw), interventions at INTERVENE (post-advance). Events are no
    /// longer keyed on `round(t/dt)` internally — the integrator lands on each
    /// event time, so off-grid observation anchoring can't shift the firing
    /// instant (gh#216, the events arm).
    fn step(
        &self,
        state: &mut Self::State,
        params: &[f64],
        t: f64,
        dt: f64,
        rng: &mut StatefulRng,
        scratch: &mut Self::Scratch,
        due_effects: &[usize],
    ) -> Result<(), SimError>;

    /// Allocate a fresh scratch buffer sized for this model.
    fn new_scratch(&self) -> Self::Scratch;

    /// The underlying `CompiledModel`, when this process wraps one.
    ///
    /// Default `None` so non-model process mocks (e.g. the watchdog's
    /// `CountingProcess`) need not synthesize one. The production
    /// `ChainBinomialProcess` returns `Some`, which lets the generic PF/IF2
    /// drivers reach the model for setup-time guards (gh#216: refuse Exact +
    /// off-grid observations for a scheduled-intervention-bearing model). Named
    /// distinctly from
    /// `DensityProcess::compiled_model` (PGAS-only, infallible `&CompiledModel`)
    /// to avoid a two-trait method-name collision; this is the optional,
    /// `ProcessModel`-level hook.
    fn try_compiled_model(&self) -> Option<&crate::compiled_model::CompiledModel> {
        None
    }
}

/// Observation model: maps latent state to data likelihood.
///
/// Encapsulates projection (state → observable quantity), likelihood
/// evaluation, sampling, and mean computation. Handles multi-stream
/// observations internally.
///
/// The `obs_idx` parameter indexes into the observation time series.
/// `params` is always passed — IF2 uses per-particle params, PGAS/PMMH
/// pass updated params each sweep/step. This is the correct interface:
/// the observation likelihood IS a function of params in general
/// (e.g., sigma_se for overdispersion).
pub trait ObservationModel<S>: Send + Sync {
    /// Joint log p(y_{obs_idx} | state, params) across all streams.
    ///
    /// This is the ONLY method required for inference. All algorithms
    /// (PF, IF2, PMMH, PGAS) call this for particle weighting.
    fn log_likelihood(&self, state: &S, obs_idx: usize, params: &[f64]) -> f64;

    /// Number of observation times.
    fn n_observations(&self) -> usize;

    /// Observation time at index `obs_idx`.
    fn obs_time(&self, obs_idx: usize) -> f64;

    /// Number of observation streams.
    fn n_streams(&self) -> usize { 1 }

    /// Sample y ~ p(y | state, params) for prediction diagnostics.
    /// Returns one draw per stream.
    fn sample(
        &self, _state: &S, _obs_idx: usize,
        _params: &[f64], _rng: &mut StatefulRng,
    ) -> Vec<f64> {
        vec![]
    }

    /// E[y | state, params] for prediction diagnostics.
    /// Returns one value per stream.
    fn mean(&self, _state: &S, _obs_idx: usize, _params: &[f64]) -> Vec<f64> {
        vec![]
    }

    /// Per-stream names for output schemas (paths.tsv columns,
    /// posterior-predictive labels). Returns one name per stream;
    /// length must equal `n_streams()`. Default is generic
    /// `["stream_0", ...]` for impls that don't carry IR-level
    /// observation names.
    fn stream_names(&self) -> Vec<String> {
        (0..self.n_streams()).map(|i| format!("stream_{}", i)).collect()
    }

    // ── Multi-cadence per-stream incidence bins (Phase 2a, "Option Z") ───────
    //
    // An `Interval` (incidence) stream is scored against the flow accumulated
    // since ITS OWN last observation. With streams on different cadences the
    // single global per-transition accumulator cannot be blanket-reset at every
    // observation (an ES-only union time would wrongly zero AFP's monthly bin).
    // So each Interval stream carries its own persistent bin in
    // `ParticleState.acc`, folded once per interval and reset per-stream.
    //
    // Default impls keep `NullObsModel` and non-incidence mocks no-ops:
    // `n_interval_streams() == 0` ⇒ `acc` is empty ⇒ fold/reset are vacuous.

    /// Number of `Interval` (incidence) streams — the length each particle's
    /// `acc` bin vector must have. `0` for models with no incidence stream.
    fn n_interval_streams(&self) -> usize { 0 }

    /// Fold this interval's per-transition `flow_accumulators` into the
    /// per-stream `acc`: for each Interval stream `k`,
    /// `acc[k] += Σ_{i ∈ flow_indices(k)} flow_accumulators[i]`. Called once per
    /// observation interval, after the substeps and BEFORE scoring. Does NOT
    /// zero `flow_accumulators`.
    fn fold_into_acc(&self, _flow_accumulators: &[u64], _acc: &mut [u64]) {}

    /// Reset (zero) the per-stream `acc` bins for exactly the Interval streams
    /// SCHEDULED at union index `union_idx` (`at_union[union_idx].is_some()`).
    /// A stream not scheduled here (a sibling's cadence) keeps accumulating
    /// toward its own next observation. Called after scoring + resampling, in
    /// lockstep with the blanket `reset_flows()` of `flow_accumulators`.
    fn reset_due_acc(&self, _union_idx: usize, _acc: &mut [u64]) {}
}

/// Extension of ProcessModel for algorithms that need transition density
/// evaluation (PGAS, CSMC-AS).
///
/// Only chain-binomial processes implement this. If you try to use PGAS
/// with a process model that doesn't implement `DensityProcess`, you get
/// a compile-time error — not a runtime panic or silent wrong answer.
///
/// The `compiled_model()` escape hatch exposes the concrete `CompiledModel`.
/// PGAS is coupled to chain-binomial by design (source groups, stoichiometry,
/// balance constraints). Adding 10 accessor methods with one implementor
/// each would be abstraction theater.
pub trait DensityProcess: ProcessModel {
    /// Log transition density for one substep.
    ///
    /// Evaluates log p(flows | state_before, params, gammas, t, dt).
    /// Returns -inf for impossible transitions (flow > source count).
    fn log_transition_density(
        &self,
        counts_before: &[i64],
        flows: &[u64],
        gammas: &[f64],
        params: &[f64],
        t: f64,
        dt: f64,
    ) -> Result<f64, crate::error::SimError>;

    /// Access to the underlying compiled model for PGAS internals.
    fn compiled_model(&self) -> &crate::compiled_model::CompiledModel;
}

/// Configuration shared by all SMC-based algorithms.
///
/// Note: `seed` is NOT here. Seed is per-chain, passed to the algorithm
/// call, not bundled into config. Config describes the statistical problem.
#[derive(Clone, Debug)]
pub struct SMCConfig {
    pub n_particles: usize,
    pub dt: f64,
    /// Simulation start time (before first observation).
    pub t_start: f64,
    /// IC-free inference: weight and resample at the first observation
    /// (so y₁ pins the initial state via Bayesian update on the particle
    /// cloud) but do NOT accumulate that step's log-sum-exp into the
    /// returned log-likelihood. Log-likelihood accumulation starts from
    /// the second observation.
    ///
    /// The caller is responsible for ensuring particle spread at t=0 —
    /// typically via an `ivp = true` estimated parameter. Without
    /// spread, the first reweight is a no-op and ic-free degenerates to
    /// silently dropping the first observation. Validation is at the
    /// fit-config layer. See
    /// docs/dev/proposals/2026-04-18-ic-free-inference.md.
    pub skip_first_obs_from_loglik: bool,

    /// Record per-step pre-resample particle states + log-weights and
    /// per-step ancestor indices so the caller can reconstruct
    /// filtering marginals or sample smoothing paths via ancestor
    /// tracing. Off by default (extra memory + copy cost). See
    /// `sim::inference::ancestor_trace` for the consumer side and
    /// `docs/dev/proposals/2026-04-19-pf-latent-trajectories.md`
    /// for why this is gated.
    #[doc(alias = "save_trajectories")]
    pub record_ancestry: bool,
    /// Record the per-step per-particle predictive samples and
    /// log-likelihoods needed to build a `PrequentialTrace` (log
    /// score, CRPS, PIT). Roughly N × T f64 per step; cheap
    /// relative to the filter itself. See
    /// `docs/dev/proposals/2026-04-20-prequential-evaluation.md`.
    pub record_prequential: bool,

    /// gh#241. The deterministic per-call compute budget: the maximum
    /// cumulative particle-substep count one filter evaluation may execute
    /// before bailing with `PFIterationBudget`. Default
    /// [`degeneracy::ITER_BUDGET`](crate::inference::degeneracy::ITER_BUDGET).
    /// A typed input (`--pf-max-substeps`), reproducible across machines —
    /// it replaced a machine-speed-dependent wall-clock timeout (which made a
    /// fit's log-likelihood depend on hardware and was an un-typed env-var
    /// channel). Bigger ⇒ more headroom before a misconfiguration aborts; it
    /// never changes a *completing* filter's result, so it is not part of run
    /// identity.
    pub max_substeps: u64,
}

/// Shared interface for algorithm config structs that drive a particle filter.
///
/// All three inference algorithms — IF2, PGAS, PMMH — carry `n_particles` and
/// `dt` with identical types and semantics. This trait makes that surface
/// explicit, enabling generic progress logging and sizing code.
pub trait InferenceConfig {
    fn n_particles(&self) -> usize;
    fn dt(&self) -> f64;
}
