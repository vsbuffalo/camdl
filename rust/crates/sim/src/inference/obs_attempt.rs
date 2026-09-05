//! What the ensemble managed at one observation, named by the measurement
//! rather than by its position in the observation queue.
//!
//! A refusal that reports "observation 16" names nothing a modeller can act on
//! once several streams are bound and interleaved in time, and the index is not
//! a stable identifier either: it is a position on the union axis, whose
//! composition changes whenever the bound stream set changes. Unbinding two
//! streams renumbers everything, so the index cannot be used to compare one
//! ablation against another. A record keyed on `(stream, time)` survives that.
//!
//! [`StreamAttempt`] is that record. It is built on the failure path only — the
//! swarm has already lost support and the chain is being abandoned — so it may
//! re-score and re-project freely; the hot loop's `f64` return is untouched.
//!
//! # Every field is reduced over LIVE particles only
//!
//! A particle killed earlier by a chain-binomial overshoot carries `−∞` without
//! the observation model having been consulted at all (`pgas_init.rs` scores
//! `*lw = if dead { NEG_INFINITY } else { … }`). A mixture — most particles
//! dead from the process model, the rest scoring `−∞` on the observation —
//! reaches the collapse check and would read as a unanimous observation
//! refusal. So the reductions here cover the live particles, [`n_dead`] is
//! reported beside them, and a refusal with `n_dead` near `n_particles` is a
//! process-model finding rather than an observation-model one.
//!
//! [`n_dead`]: StreamAttempt::n_dead

use serde::{Deserialize, Serialize};

/// What the ensemble managed at one stream of one observation.
///
/// Every count is over the LIVE particles unless its name says otherwise, and
/// `n_live + n_dead == n_particles`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamAttempt {
    /// The declared stream name (the observation block / district), not its
    /// queue position.
    pub stream: String,
    /// The observation's time on the model axis. Matches
    /// `ObsFilterEss.time` and `MultiStreamObsModel`'s union axis.
    pub time: f64,
    /// The same instant as a calendar label, when the model declares an
    /// `origin`. `None` for an unanchored model.
    pub date: Option<String>,
    /// Scheduled-and-observed, scheduled-but-missing, or not scheduled here.
    /// The three are distinct in the scoring path and all three contribute
    /// `0.0` to the joint, so collapsing them into one number loses the
    /// distinction that says whether this stream was even asked a question.
    pub cell: ObsCellState,
    /// The stream's projected quantity across the live particles — the value
    /// the likelihood family is built on, NOT the family's mean in general.
    /// `None` when no particle was live, when every live projection was NaN,
    /// or when the stream is not scheduled at this observation.
    pub projected_max: Option<f64>,
    /// Median of the same non-NaN live projections (lower median for an even
    /// count, so the reported value is always one a particle actually held).
    pub projected_median: Option<f64>,
    /// Live particles whose projection was exactly zero.
    pub n_projected_zero: usize,
    /// Live particles whose projection was NaN — an `Expr` projection such as
    /// `I/(S+I+R)` at zero population. Excluded from the two summaries above,
    /// and reported rather than absorbed: `f64::max` silently returns the
    /// non-NaN operand and `partial_cmp().unwrap()` panics, and neither is
    /// acceptable on a path already handling a failure.
    pub n_projected_nan: usize,
    /// Live particles whose per-stream log-density at this observation was
    /// `−∞`. Zero for a `Hole` or a `NotScheduled` cell, which contribute no
    /// likelihood factor at all.
    pub n_neg_inf: usize,
    /// Which guard produced those `−∞`s, and how many live particles each
    /// accounted for. Sums to `n_neg_inf`. The causes have different fixes —
    /// see [`NegInfCause`], whose variants say where each one is fixed.
    pub neg_inf_causes: Vec<(NegInfCause, usize)>,
    /// Particles that reached the observation model.
    pub n_live: usize,
    /// Particles already dead when this observation was scored. They carry
    /// `−∞` without the observation model having been consulted.
    pub n_dead: usize,
    /// `n_live + n_dead`.
    pub n_particles: usize,
}

/// Whether this stream had an observation at this union index, and if so
/// whether it carried a value.
///
/// The scoring path separates all three and returns `0.0` from every one of
/// them, as does a genuine zero-density row (`binom_logpmf` and
/// `beta_binomial_logpmf` return exactly `0.0` for `n == 0, k == 0`, which is
/// routine surveillance data, not an error). Three states behind one number is
/// what this enum exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ObsCellState {
    /// Scheduled here and observed, with the value that was scored.
    Scored { y_obs: f64 },
    /// Scheduled here, value missing (`NA`). No likelihood term — the value is
    /// marginalized, not scored as zero — but the accumulator reset still
    /// fires on schedule.
    Hole,
    /// Not scheduled at this union index: a sibling stream's cadence.
    NotScheduled,
}

impl ObsCellState {
    /// The observed value, when there is one.
    pub fn y_obs(&self) -> Option<f64> {
        match self {
            ObsCellState::Scored { y_obs } => Some(*y_obs),
            ObsCellState::Hole | ObsCellState::NotScheduled => None,
        }
    }

    /// True when a likelihood factor was evaluated here. `Hole` and
    /// `NotScheduled` contribute none.
    pub fn is_scored(&self) -> bool {
        matches!(self, ObsCellState::Scored { .. })
    }
}

/// Which `−∞` guard the observation likelihood took, for one live particle.
///
/// `log_likelihood_per_stream_from_flows_and_counts` says WHICH stream returned
/// `−∞`; it cannot say why, and the causes have different fixes. For
/// `beta_binomial_logpmf` the four guards are `n == 0 && k != 0`,
/// `alpha.is_nan() || beta.is_nan()`, `k > n`, and `alpha <= 0 || beta <= 0`.
/// Resolving them is the difference between "your denominator column is wrong"
/// and "your mean expression goes NaN here".
///
/// # Where a cause is fixed
///
/// Each variant's doc says whether a reader has to change the data or the
/// model. This is deliberately documentation rather than a method: for the two
/// count guards the answer depends on how the likelihood's `n` was written, and
/// by the time a resolved likelihood is being scored that distinction is no
/// longer visible here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cause", rename_all = "snake_case")]
pub enum NegInfCause {
    /// The observed value itself is not a number.
    ObservationNotFinite,
    /// Zero trials (`n == 0`) against a positive observed count.
    ///
    /// When `n` is a bound denominator COLUMN this is data only — no parameter
    /// and no modelled flow can move it — and it is then unreachable here,
    /// because `BoundObs::bind` refuses that data outright ("no trials cannot
    /// yield a positive count"). It reaches the runtime only when `n` is an
    /// EXPRESSION the binder cannot evaluate ahead of a trajectory, in which
    /// case the expression is what to look at.
    ZeroTrialsPositiveCount,
    /// The observed count exceeds the denominator (`k > n`).
    ///
    /// Same split as [`Self::ZeroTrialsPositiveCount`]: with `n` a bound
    /// column the value is constant across particles and across θ — so a
    /// modelled flow exceeding the denominator cannot be its cause — and
    /// `BoundObs::bind` rejects it before a chain ever runs. With `n` an
    /// expression it is reachable, and then it is state- or θ-dependent.
    CountExceedsTrials,
    /// A likelihood argument evaluated to NaN at this particle's state. Model
    /// side: a mean written `k * projected / denom` is NaN exactly where
    /// `denom` is zero.
    ArgumentNaN { arg: String },
    /// A likelihood argument is finite but outside the family's domain (a
    /// non-positive shape, a non-positive dispersion, a non-positive scale).
    /// Model side.
    ArgumentOutOfDomain { arg: String, value: f64 },
    /// The arguments are in domain, but they put the observed value outside
    /// the family's support — a positive count under a zero rate, a proportion
    /// outside `(0, 1)`. Reading this needs both the data and the parameters.
    ObservedOutsideSupport { observed: f64 },
    /// The per-stream density was `−∞` and no guard accounted for it. Never
    /// expected; reported rather than attributed to the wrong guard.
    Unclassified,
}

impl std::fmt::Display for NegInfCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NegInfCause::ObservationNotFinite => write!(f, "the observed value is not finite"),
            NegInfCause::ZeroTrialsPositiveCount => write!(
                f,
                "the denominator is zero trials against a positive observed count"
            ),
            NegInfCause::CountExceedsTrials => {
                write!(f, "the observed count exceeds the denominator")
            }
            NegInfCause::ArgumentNaN { arg } => write!(f, "likelihood argument '{arg}' is NaN"),
            NegInfCause::ArgumentOutOfDomain { arg, value } => {
                write!(f, "likelihood argument '{arg}' = {value} is outside its domain")
            }
            NegInfCause::ObservedOutsideSupport { observed } => write!(
                f,
                "the observed value {observed} is outside the family's support at these parameters"
            ),
            NegInfCause::Unclassified => write!(f, "cause not attributed"),
        }
    }
}

impl std::fmt::Display for StreamAttempt {
    /// One line naming the measurement and what the ensemble managed at it.
    /// This is the prose half of the record, rendered from the same fields the
    /// structured half carries so the two cannot drift.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream '{}' at t={}", self.stream, self.time)?;
        if let Some(d) = &self.date {
            write!(f, " ({d})")?;
        }
        match &self.cell {
            ObsCellState::NotScheduled => return write!(f, ": not scheduled here"),
            ObsCellState::Hole => return write!(f, ": scheduled, value missing (no term)"),
            ObsCellState::Scored { y_obs } => write!(f, ": observed {y_obs}")?,
        }
        match (self.projected_max, self.projected_median) {
            (Some(max), Some(med)) => write!(
                f,
                ", projected max {max} / median {med} over {} live particles",
                self.n_live
            )?,
            _ => write!(
                f,
                ", no finite projection over {} live particles",
                self.n_live
            )?,
        }
        if self.n_projected_nan > 0 {
            write!(f, " ({} NaN)", self.n_projected_nan)?;
        }
        if self.n_dead > 0 {
            write!(f, ", {} of {} particles already dead", self.n_dead, self.n_particles)?;
        }
        if self.n_neg_inf > 0 {
            write!(f, "; {} live particles scored -inf", self.n_neg_inf)?;
            for (cause, n) in &self.neg_inf_causes {
                write!(f, " [{n} × {cause}]")?;
            }
        }
        Ok(())
    }
}
