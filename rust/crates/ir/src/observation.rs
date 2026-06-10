use serde::{Deserialize, Serialize};
use crate::expr::Expr;

// ── Projection ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    CumulativeFlow(String),
    CurrentPop(String),
    CurrentPopSum(Vec<String>),
    DerivedExpr(Expr),
    // New variants append at the END: the run_id hash (runid::ir_hash) tags
    // variants by position, so declaration order == hash index, and that
    // index is permanent. Inserting earlier would churn stored run_ids.
    CumulativeFlowSum(Vec<String>),
}

/// Whether an observation stream measures a quantity accumulated over a
/// reporting *interval* (incidence) or sampled at an *instant* (prevalence).
///
/// This is a **derived classification of [`Projection`], never a stored
/// field** — every projection variant maps to exactly one kind (see
/// [`Projection::temporal_kind`]), so an independently-stored `kind` could
/// only ever *disagree* with the projection and would be an illegal state to
/// validate against. Code that needs the distinction (reset semantics,
/// missing-data handling, cadence) derives it; it is not serialized and does
/// not appear in the IR.
///
/// - [`Interval`](TemporalKind::Interval) — incidence: a flow accumulated
///   between observations. The accumulator resets on the reporting cadence.
/// - [`Instant`](TemporalKind::Instant) — prevalence: a function of state read
///   at the observation instant. No accumulation, no reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalKind {
    Interval,
    Instant,
}

impl Projection {
    /// Classify this projection as incidence ([`Interval`](TemporalKind::Interval))
    /// or prevalence ([`Instant`](TemporalKind::Instant)). Total over all
    /// variants — the single source of truth for the distinction.
    pub fn temporal_kind(&self) -> TemporalKind {
        match self {
            // incidence — cumulative flow over the reporting interval
            Projection::CumulativeFlow(_) | Projection::CumulativeFlowSum(_) => {
                TemporalKind::Interval
            }
            // prevalence — state read at the observation instant
            Projection::CurrentPop(_)
            | Projection::CurrentPopSum(_)
            | Projection::DerivedExpr(_) => TemporalKind::Instant,
        }
    }
}

// ── Likelihood ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoissonLikelihood {
    pub rate: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NegBinomialLikelihood {
    pub mean:       Expr,
    pub dispersion: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalLikelihood {
    pub mean: Expr,
    pub sd:   Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinomialLikelihood {
    pub n: Expr,
    pub p: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BetaBinomialLikelihood {
    pub n:     Expr,
    pub alpha: Expr,
    pub beta:  Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BernoulliLikelihood {
    pub p: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Likelihood {
    Poisson(PoissonLikelihood),
    NegBinomial(NegBinomialLikelihood),
    Normal(NormalLikelihood),
    Binomial(BinomialLikelihood),
    BetaBinomial(BetaBinomialLikelihood),
    Bernoulli(BernoulliLikelihood),
}

// ── Observation schedule ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegularSchedule {
    pub start: f64,
    pub step:  f64,
    pub end:   f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSchedule {
    AtTimes(Vec<f64>),
    Regular(RegularSchedule),
}

// ── Observation model ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationModel {
    pub name:        String,
    pub schedule:    ObservationSchedule,
    pub projection:  Projection,
    pub likelihood:  Likelihood,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ConstExpr, Expr};

    #[test]
    fn temporal_kind_classifies_every_projection_variant() {
        use TemporalKind::*;
        // incidence — accumulated over a reporting interval
        assert_eq!(Projection::CumulativeFlow("inc".into()).temporal_kind(), Interval);
        assert_eq!(
            Projection::CumulativeFlowSum(vec!["a".into(), "b".into()]).temporal_kind(),
            Interval
        );
        // prevalence — read at the observation instant
        assert_eq!(Projection::CurrentPop("I".into()).temporal_kind(), Instant);
        assert_eq!(
            Projection::CurrentPopSum(vec!["B1".into(), "B2".into()]).temporal_kind(),
            Instant
        );
        assert_eq!(
            Projection::DerivedExpr(Expr::Const(ConstExpr { value: 0.0 })).temporal_kind(),
            Instant
        );
    }
}
