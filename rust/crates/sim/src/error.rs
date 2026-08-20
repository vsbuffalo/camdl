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
    /// - `AllParticlesDead`: every particle hit a per-particle-recoverable
    ///   error (NumericalCollapse / NegativeCount{BinomialOvershoot}) —
    ///   the limit case of ESS collapse, but cheap to detect and
    ///   diagnostically distinct.
    ///
    /// Not per-particle-recoverable — this is a whole-call bail. It is
    /// also not `is_structural` (gh#224): at a *proposed* θ a degenerate
    /// filter means "this θ is uninformative," so the PMMH / PF eval
    /// closures report −∞ and the MH step rejects the proposal cleanly.
    /// Init-eval callers detect the bail explicitly to surface a
    /// `BadInit` diagnostic and skip the chain.
    #[error("particle filter degeneracy ({kind:?}) at obs_window {obs_window}, elapsed {elapsed_s:.2}s")]
    PFDegenerate {
        kind: PFDegenerateKind,
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
    /// This is a *configuration*-level limit: the per-window cost depends only on
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

    /// gh#607. A PGAS chain's START has zero posterior density and did not
    /// recover: the complete-data log-posterior at `(θ₀, X₀)` —
    /// `log p(y, X₀ | θ₀) + log p(θ₀)` — is non-finite, and is STILL non-finite
    /// after the sampler's first complete Gibbs sweep.
    ///
    /// The two-part test is load-bearing. A `−∞` at `(θ₀, X₀)` is usually the
    /// observation term, which is a property of the PAIR: `X₀` is one
    /// stochastic reference draw, and the `X|θ,y` move routinely replaces it
    /// with a trajectory that explains the data at the same `θ₀` (measured on
    /// three of this repository's own PGAS fixtures, all of which start a chain
    /// at `−∞` and are finite by the first recorded sweep). Refusing on the
    /// initial value alone would kill those chains.
    ///
    /// Once a full sweep has failed, though, the chain is frozen: the `θ|X`
    /// step scores each proposal against the current `X`, so with both at `−∞`
    /// the MH ratio is NaN and rejects, and NUTS is worse — `log p = −∞` gives
    /// `h0 = +∞`, so `(h_new − h0).abs() > delta_max` flags every doubling
    /// divergent and the tree stops at depth 0. Every later sweep is an
    /// independent retry of the same failed `X`-move at the same `θ₀`. Measured
    /// on the 40 000-sweep, 8-chain production fit that motivated this: 40 000
    /// consecutive failures — acceptance 0.000 and `n_divergent` 1.000 on every
    /// sweep, ONE distinct parameter vector across 7 600 retained draws, an
    /// eighth of a 2 h 29 m run pooled into the posterior and R̂.
    ///
    /// Not `is_structural`: like `PFDegenerate` this is a property of the
    /// chain's START, not of the model — sibling chains of the same fit run
    /// normally. The driver (`cli/src/fit/pgas.rs`) turns it into a `BadInit`
    /// diagnostic and skips the chain, erroring only when EVERY chain is
    /// refused.
    #[error("chain start has zero posterior density and did not recover on its \
             first trajectory update: the initial complete-data log-posterior is \
             {log_posterior} (log-likelihood terms: transition {transition}, \
             observation {observation}, ivp {ivp}; log prior {log_prior})")]
    NonFiniteChainStart {
        /// `transition + observation + ivp + log_prior` — the number the
        /// chain would have been seeded with.
        log_posterior: f64,
        /// Complete-data transition term. Non-finite here means a
        /// `step_one` / `log_transition_density_substep` disagreement — a
        /// bug, not a bad start (gh#80).
        transition: f64,
        /// Observation term `log p(y | X₀)`. The common cause: the reference
        /// trajectory predicts zero where the data is positive.
        observation: f64,
        /// Initial-state (IVP) Binomial term.
        ivp: f64,
        /// `Σ log p(θ₀)` over the estimated parameters. Non-finite here means
        /// the start is outside its own prior's support.
        log_prior: f64,
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
    /// `log(x)` for `x ≤ 0` — a domain error (no real result), the log
    /// analogue of `SqrtNegative`. Routed through the same typed collapse
    /// so `log` is not silently `−inf` while `sqrt` errors.
    LogNonPositive,
    ModByZero,
}

/// gh#audit-C5. Cause discriminator for NegativeCount.
/// BinomialOvershoot is expected during inference exploration and
/// gets caught by the inference layer; InterventionAddNegative and
/// InterventionNegative are always config bugs and propagate regardless
/// of caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeCountCause {
    /// A chain-binomial split drew more exits than the source held
    /// (rate·dt → 1). Transient during inference parameter exploration;
    /// the inference layer catches it and kills the offending particle.
    BinomialOvershoot,
    /// An `add()` action resolved to a negative amount — caught eagerly
    /// at the action site (you cannot add a negative number of
    /// individuals). Config bug; propagates regardless of caller.
    InterventionAddNegative,
    /// A compartment was left negative *after* the INTERVENE/BALANCE
    /// stage — e.g. a `set()` to a value below zero, or a transfer that
    /// overdrew — caught by the centralized post-INTERVENE scan. The
    /// balance target is exempt (its negativity is a separate, warned
    /// signal). Config bug; propagates regardless of caller.
    InterventionNegative,
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
    /// SqrtNegative, LogNonPositive, ModByZero), NegativeCount with cause
    /// BinomialOvershoot, NonFiniteParameter (gh#81 Phase 2 — a
    /// NUTS/PMMH proposal produced a NaN/Inf parameter), and TableLookup
    /// (gh#127 #12 — a state/parameter-dependent table index that went
    /// out of range at runtime) — these all arise from particles or
    /// proposals exploring extreme parameter regions and should
    /// reject-and-continue rather than die.
    ///
    /// On `TableLookup`: a runtime out-of-range lookup is reached only by a
    /// NON-constant (state/parameter-dependent) index — a constant OOB index is
    /// rejected statically by `validate` (gh#127). An inference proposal can
    /// sweep such an index out of range for one particle; per the issue, that
    /// "one bad particle should not panic the entire process," so it is a
    /// controlled per-particle failure, not a whole-run bail. The structural
    /// `TableLookup` (wrong arity) surfaces at `CompiledModel::new`, before any
    /// particle runs, so it never reaches this discriminator.
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
            | SimError::TableLookup(..)
        )
    }

    /// gh#224. Inference-eval discriminator: does this error mean the
    /// **model or its configuration cannot run** (→ surface as a hard
    /// failure), as opposed to a per-θ excursion or a degenerate /
    /// over-budget particle filter (→ the likelihood evaluator reports
    /// −∞ and the MH step rejects that θ)?
    ///
    /// The PMMH / PF parameter-eval closures use this to separate a
    /// legitimate "θ ruled out" (−∞, seen routinely and rejected) from a
    /// structural failure that must NOT be mistaken for one — otherwise a
    /// model that cannot run returns a degenerate posterior with a
    /// successful (exit-0) status. This is a *different* question from
    /// `is_per_particle_recoverable` (which asks whether the per-particle
    /// dead-mask can absorb the error inside one filter call): a
    /// `PFDegenerate` bail is `false` for both — not dead-mask-absorbable,
    /// but also not structural — because at a *proposed* θ it means "this
    /// filter run is uninformative," which the MH ratio rejects.
    ///
    /// The match is exhaustive on purpose: a new `SimError` variant must
    /// be classified here (it will not compile otherwise), so the
    /// surface-vs-reject decision can never be silently defaulted.
    ///
    /// `true` (surface): model/config errors that fire regardless of θ —
    /// every proposal would hit them, so the fit is meaningless — plus
    /// `PFIterationBudget`, the *deterministic* compute-budget bail that,
    /// per its own contract (gh#147 M3.1), trips identically for every
    /// chain and is meant to be fatal rather than retried.
    ///
    /// `false` (reject as −∞): per-particle excursions, θ-dependent
    /// runtime conditions (`DivisionByZero`, `NegativePropensity`,
    /// `AbsorbingState`), the whole-call PF bail (`PFDegenerate`), and the
    /// PGAS chain-start refusal (`NonFiniteChainStart`). Init-eval callers
    /// treat the last two specially — a `BadInit` skip — via the CLI init
    /// guard and the PGAS driver respectively.
    pub fn is_structural(&self) -> bool {
        use NegativeCountCause::*;
        match self {
            // Model / configuration errors — fire regardless of θ.
            SimError::ConfigMismatch { .. }
            | SimError::UnknownCompartment(_)
            | SimError::UnknownParameter(_)
            | SimError::UnknownTimeFunction(_)
            | SimError::UnknownTable(_)
            | SimError::UnknownOp(_)
            | SimError::WrongArgCount { .. }
            | SimError::Validation(_)
            // Deterministic compute-budget bail: trips identically for
            // every chain/iteration; meant to be fatal, not retried.
            | SimError::PFIterationBudget { .. } => true,

            // A config-bug intervention (adds / leaves a compartment
            // negative) is structural; a binomial overshoot is a transient
            // per-θ excursion the inference layer rejects.
            SimError::NegativeCount { cause, .. } => {
                matches!(cause, InterventionAddNegative | InterventionNegative)
            }

            // Per-θ excursions, θ-dependent runtime conditions, and the
            // whole-call PF degeneracy bail — reject this θ as −∞.
            SimError::TableLookup(_)
            | SimError::DivisionByZero(_)
            | SimError::NegativePropensity { .. }
            | SimError::AbsorbingState(_)
            | SimError::NumericalCollapse { .. }
            | SimError::NonFiniteParameter { .. }
            | SimError::PFDegenerate { .. }
            // A start with zero posterior density is a property of THIS
            // chain's starting point, not of the model — the fit's other
            // chains are unaffected, so the driver skips this one rather
            // than aborting the run.
            | SimError::NonFiniteChainStart { .. } => false,
        }
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
            kind: PFDegenerateKind::AllParticlesDead,
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
        assert!(!SimError::NegativeCount {
            compartment: "S".into(),
            attempted_value: -1,
            t: 1.0,
            cause: NegativeCountCause::InterventionNegative,
        }.is_per_particle_recoverable());
        // gh#127 (#12): a runtime out-of-range table lookup (non-constant
        // index swept OOB by an inference proposal) is recoverable — kill the
        // offending particle, don't tear down the run.
        assert!(SimError::TableLookup(
            "table 'k': index 5 out of bounds [0, 2)".into()
        ).is_per_particle_recoverable());
        assert!(!SimError::AbsorbingState(0.0).is_per_particle_recoverable());
        assert!(!SimError::PFDegenerate {
            kind: PFDegenerateKind::AllParticlesDead,
            obs_window: 0,
            elapsed_s: 0.0,
        }.is_per_particle_recoverable());
    }

    /// gh#82. The two discriminators must not drift apart: everything
    /// `is_per_particle_recoverable()` admits must also be NON-structural.
    ///
    /// PGAS's θ-proposal boundary (`pgas.rs::theta_proposal_score`) rejects on
    /// `!is_structural()`, so if a per-particle-recoverable variant were ever
    /// classified structural it would start tearing chains down again — the
    /// exact regression gh#82 fixed. The implication is one-directional: the
    /// converse does not hold (`NegativePropensity`, `AbsorbingState` and
    /// `PFDegenerate` are neither recoverable nor structural), which is why the
    /// θ-eval boundary uses the wider `is_structural()` test.
    #[test]
    fn recoverable_errors_are_never_structural() {
        let recoverable: Vec<SimError> = vec![
            SimError::NumericalCollapse { kind: CollapseKind::DivByZero, t: 1.0 },
            SimError::NumericalCollapse { kind: CollapseKind::PowNanInf, t: 1.0 },
            SimError::NumericalCollapse { kind: CollapseKind::UnOpNan, t: 1.0 },
            SimError::NumericalCollapse { kind: CollapseKind::SqrtNegative, t: 1.0 },
            SimError::NumericalCollapse { kind: CollapseKind::LogNonPositive, t: 1.0 },
            SimError::NumericalCollapse { kind: CollapseKind::ModByZero, t: 1.0 },
            SimError::NegativeCount {
                compartment: "S".into(), attempted_value: -1, t: 1.0,
                cause: NegativeCountCause::BinomialOvershoot,
            },
            SimError::NonFiniteParameter { name: "tau".into(), value: f64::NEG_INFINITY, t: -101.0 },
            SimError::TableLookup("table 'k': index 5 out of bounds [0, 2)".into()),
        ];
        for err in recoverable {
            assert!(
                err.is_per_particle_recoverable(),
                "list must contain only recoverable variants; {err} is not one",
            );
            assert!(
                !err.is_structural(),
                "a per-particle-recoverable error must never be structural, or the \
                 PGAS θ-proposal boundary would tear the chain down on it (gh#82): {err}",
            );
        }
    }

    /// gh#224. The inference-eval discriminator must surface model/config
    /// errors as fatal while rejecting per-θ excursions and PF bails as
    /// −∞. The load-bearing case is `PFDegenerate`: it is NOT structural,
    /// so adaptive PMMH (whose wide warmup proposals routinely hit
    /// degenerate θ regions) rejects them rather than aborting the fit.
    #[test]
    fn structural_set_surfaces_config_errors_not_per_theta_excursions() {
        // Surface (model/config can't run):
        assert!(SimError::Validation("bad model".into()).is_structural());
        assert!(SimError::UnknownCompartment("ghost".into()).is_structural());
        assert!(SimError::UnknownParameter("ghost".into()).is_structural());
        assert!(SimError::ConfigMismatch { expected: "a", got: "b" }.is_structural());
        assert!(SimError::NegativeCount {
            compartment: "S".into(), attempted_value: -1, t: 1.0,
            cause: NegativeCountCause::InterventionAddNegative,
        }.is_structural());
        // Deterministic compute-budget bail is meant to be fatal (gh#147).
        assert!(SimError::PFIterationBudget {
            obs_window: 3, attempted_substeps: 10, budget_substeps: 5,
        }.is_structural());

        // Reject as −∞ (per-θ / PF-uninformative, NOT structural):
        assert!(!SimError::PFDegenerate {
            kind: PFDegenerateKind::EssCollapsed { last_ess: vec![1.0, 1.0, 1.0] },
            obs_window: 6, elapsed_s: 0.01,
        }.is_structural());
        assert!(!SimError::NumericalCollapse { kind: CollapseKind::DivByZero, t: 1.0 }.is_structural());
        assert!(!SimError::NonFiniteParameter { name: "beta".into(), value: f64::NAN, t: 1.0 }.is_structural());
        assert!(!SimError::NegativeCount {
            compartment: "S".into(), attempted_value: -1, t: 1.0,
            cause: NegativeCountCause::BinomialOvershoot,
        }.is_structural());
        assert!(!SimError::AbsorbingState(0.0).is_structural());
    }

    /// gh#607. A PGAS chain start with zero posterior density must be
    /// classified like `PFDegenerate`: NOT structural (the fit's other chains
    /// are fine, so the driver skips this one and continues) and NOT
    /// per-particle recoverable (there is no particle to kill — the whole
    /// chain is refused, at the end of its first sweep).
    ///
    /// Classifying it structural would abort a multi-chain fit on one bad
    /// start, which is the failure the skip exists to prevent.
    #[test]
    fn non_finite_chain_start_is_a_skip_not_a_fatal() {
        let err = SimError::NonFiniteChainStart {
            log_posterior: f64::NEG_INFINITY,
            transition: -1234.5,
            observation: f64::NEG_INFINITY,
            ivp: 0.0,
            log_prior: -2.5,
        };
        assert!(!err.is_structural(),
            "a bad chain START must not abort the whole fit: {err}");
        assert!(!err.is_per_particle_recoverable(),
            "the chain is refused whole; there is no particle to absorb it: {err}");
        // The components must survive into the message — the driver quotes
        // them in the BadInit reason, and `observation = -inf` vs
        // `transition = -inf` are different findings (bad start vs gh#80 bug).
        let s = format!("{err}");
        assert!(s.contains("observation -inf"), "message must name the components: {s}");
        assert!(s.contains("log prior -2.5"), "message must name the prior term: {s}");
    }
}
