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
