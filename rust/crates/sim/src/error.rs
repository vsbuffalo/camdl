#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("config variant does not match simulator: expected {expected}, got {got}")]
    ConfigMismatch { expected: &'static str, got: &'static str },

    #[error("unknown compartment '{0}'")]
    UnknownCompartment(String),

    #[error("unknown parameter '{0}'")]
    UnknownParameter(String),

    #[error("unknown time function '{0}'")]
    UnknownTimeFunction(String),

    #[error("unknown table '{0}'")]
    UnknownTable(String),

    #[error("table lookup error: {0}")]
    TableLookup(String),

    #[error("division by zero in expression at t={0}")]
    DivisionByZero(f64),

    #[error("negative propensity {value} for transition '{transition}' at t={t}")]
    NegativePropensity { transition: String, value: f64, t: f64 },

    #[error("op '{op}' requires {expected} args but got {got}")]
    WrongArgCount { op: String, expected: usize, got: usize },

    #[error("unknown op '{0}'")]
    UnknownOp(String),

    #[error("model validation error: {0}")]
    Validation(String),

    #[error("absorbing state: total propensity is zero at t={0}")]
    AbsorbingState(f64),

    /// gh#audit-C6 / S1. Expression evaluation hit a numerically
    /// degenerate path. Previously `eval_expr` silently returned 0.0
    /// (wrapped in Ok(_)), masking malformed rate expressions: a small
    /// patch that empties at runtime would produce silent zero
    /// force-of-infection rather than an error. The CLI surfaces this
    /// via `--allow-degenerate-rates` if the user has a defensible
    /// reason (e.g. force-of-infection legitimately undefined when
    /// N=0); default is hard error.
    #[error("numerical collapse ({kind:?}) in rate expression at t={t}")]
    NumericalCollapse { kind: CollapseKind, t: f64 },

    /// gh#audit-C5. Compartment count went below zero. Two distinct
    /// causes: BinomialOvershoot (rate·dt → 1 in chain-binomial split,
    /// transient under inference exploration) vs InterventionAddNegative
    /// (config bug: an Action::Add expression resolved to a negative
    /// value). Previously silently clamped to 0, making the population
    /// non-conservative. Inference layers catch BinomialOvershoot and
    /// convert to −Inf log-likelihood for the offending particle;
    /// forward-sim CLI propagates as a user-facing error.
    #[error("compartment '{compartment}' would go to {attempted_value} (cause: {cause:?}) at t={t}")]
    NegativeCount {
        compartment: String,
        attempted_value: i64,
        t: f64,
        cause: NegativeCountCause,
    },

    /// gh#110. Particle filter degeneracy: the filter cannot return a
    /// finite log-likelihood in bounded time at the supplied parameters.
    /// Three triggers, all detected at observation-window boundaries:
    ///
    /// - `EssCollapsed`: effective sample size has been at or below
    ///   `ESS_FLOOR` (= 2.0) for `ESS_COLLAPSE_WINDOWS` (= 3)
    ///   consecutive observation windows. Sustained collapse — not a
    ///   single-window dip during epidemic peaks.
    /// - `WallClockExceeded`: per-call elapsed time exceeded the
    ///   wall-clock budget — `max(WALLCLOCK_FLOOR_S = 120, n_particles ·
    ///   per-particle)`, overridable via `CAMDL_PF_WALLCLOCK_TIMEOUT_S`
    ///   (`0` disables it). gh#133: this is a *resource/timeout* limit, not
    ///   a statistical pathology — a slow-but-healthy big filter trips it,
    ///   so the remedy is fewer particles or a larger/disabled budget, NOT
    ///   more particles. (A future split would surface this as a distinct
    ///   non-degenerate error; for now it rides `PFDegenerate`.)
    /// - `AllParticlesDead`: every particle hit a per-particle-recoverable
    ///   error (NumericalCollapse / NegativeCount{BinomialOvershoot}) —
    ///   the limit case of ESS collapse, but cheap to detect and
    ///   diagnostically distinct.
    ///
    /// Not per-particle-recoverable — this is a whole-call bail.
    /// `Err(PFDegenerate)` collapses to NEG_INFINITY through the
    /// existing `run_quick_pfilter_with_dt → Err(_) → NEG_INFINITY`
    /// path so PMMH iteration steps reject the proposal cleanly.
    /// Init-eval callers detect the bail explicitly to surface a
    /// `BadInit` diagnostic and skip the chain.
    #[error("particle filter degeneracy ({kind:?}) at obs_window {obs_window}, elapsed {elapsed_s:.2}s")]
    PFDegenerate {
        kind: PFDegenerateKind,
        obs_window: usize,
        elapsed_s: f64,
    },

    /// gh#133. The particle filter's per-call wall-clock budget was
    /// exceeded — a *resource/timeout* limit, surfaced distinctly from the
    /// statistical `PFDegenerate` pathologies (EssCollapsed/AllParticlesDead).
    /// A slow-but-healthy big filter trips this; the remedy is fewer
    /// particles or a larger/disabled budget, NOT more particles. Like
    /// `PFDegenerate` it is a whole-call bail (not per-particle-recoverable).
    #[error("particle filter wall-clock budget exceeded at obs_window {obs_window}, elapsed {elapsed_s:.2}s (gh#133: a slow-but-healthy filter — reduce --particles, or raise/disable the budget via --pf-wallclock-timeout / CAMDL_PF_WALLCLOCK_TIMEOUT_S=<secs|0>)")]
    PFWallclockTimeout {
        obs_window: usize,
        elapsed_s: f64,
    },

    /// gh#147 (M3.1). The particle filter's *deterministic* compute budget
    /// was exceeded: propagating an observation window would push the
    /// cumulative particle-substep count past the fixed engine budget
    /// `ITER_BUDGET`. This is the content-addressing-safe replacement for
    /// the wall-clock watchdog's compute-blowup role — the bound is a
    /// closed-form scalar (`n_particles · ceil((obs_time − t)/dt)` summed
    /// over windows), so it fires identically regardless of machine speed
    /// or thread count and never makes a fit's log-likelihood depend on
    /// wall-clock.
    ///
    /// Unlike `PFWallclockTimeout` (per-chain, machine-dependent) this is a
    /// *configuration*-level limit: the per-window cost depends only on
    /// `n_particles`, `dt`, and the observation schedule — none of which
    /// vary across chains or iterations of a fit — so if it trips, it trips
    /// identically for every chain. It therefore propagates as a fatal
    /// error (no skip-and-continue): the remedy is a larger `dt`, a shorter
    /// horizon, or fewer particles, not retrying another chain.
    #[error("particle filter compute budget exceeded at obs_window {obs_window}: \
             propagating this window needs {attempted_substeps} cumulative particle-substeps, \
             over the engine budget of {budget_substeps} (gh#147 M3.1: a deterministic \
             compute-blowup guard — your dt is too small, the horizon too long, or there are \
             too many particles for the budget; reduce --dt resolution, shorten t_end, or lower \
             --particles)")]
    PFIterationBudget {
        obs_window: usize,
        attempted_substeps: u64,
        budget_substeps: u64,
    },

    /// gh#81 Phase 2. A model parameter reached the rate evaluator
    /// already non-finite (NaN / ±Inf). The rate expression itself is
    /// innocent — propagating NaN through `beta * S * I / N` produces
    /// NaN downstream, which the legacy NaN-propagation guard at
    /// `eval_propensities` then surfaced as a generic
    /// `NumericalCollapse { kind: DivByZero }`. That diagnostic blamed
    /// the rate expression and the simulation time, hiding the actual
    /// upstream fault: a NUTS leapfrog step or PMMH random-walk proposal
    /// produced a non-finite parameter (typically from step-size
    /// adaptation going pathological, or a transform-vs-bounds violation
    /// at the edge of f64 range).
    ///
    /// Classified as per-particle recoverable: inference proposal
    /// mechanisms must reject the offending proposal and continue,
    /// not tear down the chain. Forward-sim CLI propagates as a
    /// user-facing error with the parameter name.
    #[error("parameter `{name}` is non-finite (value: {value}) at t = {t}.\n\
             This is upstream of rate evaluation — a NUTS leapfrog step or PMMH proposal\n\
             produced a NaN/Inf parameter, which would then propagate into every rate\n\
             expression that references it. The error is in the proposal mechanism, not\n\
             in the rate expression. The chain rejects this proposal and continues; if\n\
             you see thousands of these warnings, NUTS step-size adaptation is unstable\n\
             on this model. Consider:\n  \
               - init_method = \"survey_top_k\" for better starting points (gh#51)\n  \
               - increasing n_particles if PMMH; pinning some params if PGAS\n  \
               - investigating gradient stability via --check-grads (gh#78, not yet implemented)")]
    NonFiniteParameter {
        name: String,
        value: f64,
        t: f64,
    },
}

/// gh#audit-C6 / S1. Specific numerical-degeneracy mode that
/// produced a NumericalCollapse. Lets the caller distinguish
/// "div by zero" (often a population-zero edge case) from
/// "Pow NaN" (often a domain bug — negative base to fractional
/// power) for actionable error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseKind {
    DivByZero,
    PowNanInf,
    UnOpNan,
    SqrtNegative,
    ModByZero,
}

/// gh#audit-C5. Cause discriminator for NegativeCount.
/// BinomialOvershoot is expected during inference exploration and
/// gets caught by the inference layer; InterventionAddNegative is
/// always a config bug and propagates regardless of caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeCountCause {
    BinomialOvershoot,
    InterventionAddNegative,
}

/// gh#110. Specific degeneracy mode that triggered the bail.
/// Each variant carries the data needed for downstream diagnostics:
/// the K-window ESS history for `EssCollapsed`, nothing extra for
/// the wall-clock and all-dead cases (the elapsed time and obs
/// window live on the outer `SimError::PFDegenerate`).
#[derive(Debug, Clone, PartialEq)]
pub enum PFDegenerateKind {
    /// ESS has been at or below the floor for K consecutive obs
    /// windows. `last_ess` carries the K-window history (most
    /// recent last) so the diagnostic message can show the trend.
    EssCollapsed { last_ess: Vec<f64> },
    /// Per-call wall-clock has exceeded the timeout.
    WallClockExceeded,
    /// gh#147 (M3.1). The deterministic cumulative-substep budget would
    /// be exceeded by propagating the next observation window. Carries
    /// the projected cumulative substep count and the budget so the
    /// diagnostic can report both. Maps to `SimError::PFIterationBudget`
    /// (a resource limit), not a statistical pathology.
    IterationBudgetExceeded {
        attempted_substeps: u64,
        budget_substeps: u64,
    },
    /// Every particle hit a per-particle-recoverable error and is
    /// marked dead. Resampling on the next step would have zero
    /// weight everywhere; bail before the divide-by-zero.
    AllParticlesDead,
}

impl SimError {
    /// gh#audit-C5 / C6. True when the error represents a
    /// per-particle numerical degeneracy that the inference layer
    /// should catch and convert to a −Inf log-likelihood (killing the
    /// offending particle in resampling) rather than tearing down the
    /// whole filter run.
    ///
    /// Recoverable: NumericalCollapse (DivByZero, PowNanInf, UnOpNan,
    /// SqrtNegative, ModByZero), NegativeCount with cause
    /// BinomialOvershoot, and NonFiniteParameter (gh#81 Phase 2 — a
    /// NUTS/PMMH proposal produced a NaN/Inf parameter) — these all
    /// arise from particles or proposals exploring extreme parameter
    /// regions and should reject-and-continue rather than die.
    ///
    /// Not recoverable: structural errors (UnknownCompartment,
    /// UnknownParameter, ConfigMismatch, …), AbsorbingState (model-
    /// level absorbing condition, not particle-specific), and
    /// NegativeCount{InterventionAddNegative} (config bug).
    pub fn is_per_particle_recoverable(&self) -> bool {
        matches!(
            self,
            SimError::NumericalCollapse { .. }
            | SimError::NegativeCount { cause: NegativeCountCause::BinomialOvershoot, .. }
            | SimError::NonFiniteParameter { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#110. PFDegenerate is a whole-call bail; it must NOT be
    /// folded into the per-particle dead-mask path. If it were, the
    /// outer loop would silently mark every particle dead on the
    /// first call and resampling would divide by zero on the next
    /// observation window. The discriminator is `is_per_particle_recoverable`.
    #[test]
    fn pf_degenerate_is_not_per_particle_recoverable() {
        let err = SimError::PFDegenerate {
            kind: PFDegenerateKind::WallClockExceeded,
            obs_window: 42,
            elapsed_s: 121.0,
        };
        assert!(
            !err.is_per_particle_recoverable(),
            "PFDegenerate is a whole-call bail; must not be \
             absorbed into the per-particle dead-mask"
        );
    }

    #[test]
    fn pf_degenerate_ess_collapsed_carries_history() {
        let err = SimError::PFDegenerate {
            kind: PFDegenerateKind::EssCollapsed {
                last_ess: vec![1.5, 1.2, 1.0],
            },
            obs_window: 10,
            elapsed_s: 3.2,
        };
        // Round-trip via the Display impl exercises the format!() string
        let s = format!("{}", err);
        assert!(s.contains("EssCollapsed"), "kind name should be in the message: {}", s);
        assert!(s.contains("obs_window 10"), "obs_window should be in the message: {}", s);
    }

    /// Per-particle-recoverable variants stay recoverable — guards
    /// against the inverse mistake (accidentally widening the
    /// discriminator in a way that would silently kill swarms).
    #[test]
    fn per_particle_recoverable_set_is_minimal() {
        // Recoverable
        assert!(SimError::NumericalCollapse {
            kind: CollapseKind::DivByZero,
            t: 1.0,
        }.is_per_particle_recoverable());
        assert!(SimError::NegativeCount {
            compartment: "S".into(),
            attempted_value: -1,
            t: 1.0,
            cause: NegativeCountCause::BinomialOvershoot,
        }.is_per_particle_recoverable());
        // Not recoverable
        assert!(!SimError::NegativeCount {
            compartment: "S".into(),
            attempted_value: -1,
            t: 1.0,
            cause: NegativeCountCause::InterventionAddNegative,
        }.is_per_particle_recoverable());
        assert!(!SimError::AbsorbingState(0.0).is_per_particle_recoverable());
        assert!(!SimError::PFDegenerate {
            kind: PFDegenerateKind::AllParticlesDead,
            obs_window: 0,
            elapsed_s: 0.0,
        }.is_per_particle_recoverable());
    }
}
