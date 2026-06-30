//! Generated quantities (proposal 2026-06-25): named reductions of what a
//! simulation produces — the non-scored twin of an observation. Reporting-only,
//! non-identity (excluded from `Model::hash_into`). v1 reduces latent state; the
//! `observations.<stream>` source is a v1.1 additive variant on `QuantitySource`.
//!
//! These are the IR *definition* types. The runtime evaluator (`sim::quantity`)
//! and its censoring/banding types live in the backend, not here.
//!
//! A quantity's *state expression* is a plain [`Expr`], restricted to a validated
//! subset (no `Dt`/`Projected`/`ObsColumnRef`/`PerEvalRef`, transitive over
//! `BindingRef`). That restriction is enforced by `ir::validate`, not by a
//! constructor — a newtype-over-`Expr` would deserialize transparently and bypass
//! any smart constructor, and the OCaml IR has no private-constructor newtype.

use serde::{Deserialize, Serialize};
use crate::expr::{BinOp, Expr, UnOp};
use crate::observation::StratumKey;

// ── Quantity ───────────────────────────────────────────────────────────────────

/// The non-scored twin of an `ObservationModel`: a named reduction of a
/// simulation output to a reported summary, with no likelihood. Stratified
/// quantities are fully expanded (one leaf per cell, tagged with `stratum`),
/// exactly like `ObservationModel` — there is no index-binding field in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantity {
    pub name: String,
    /// The `(dimension, level)` cell this expanded leaf reports; empty for a
    /// whole-population quantity. Mirrors `ObservationModel::stratum`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stratum: Vec<StratumKey>,
    pub body: QuantityBody,
    /// Resolved dimension of the reduced value as `(P exponent, T exponent)`
    /// (prerequisite #5 of the counterfactual-contrasts proposal). Computed by
    /// the OCaml `dimcheck` pass and stored so the contrast reducer can check
    /// operand-dimension agreement without re-deriving. `None` when the dimension
    /// is undetermined; omitted on the wire when absent, so a model whose
    /// quantities carry no resolved dimension is byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<(i32, i32)>,
}

/// Either a reduction of a source series, or reduction arithmetic over earlier
/// scalar quantities. Externally tagged (`{"reduced": …}` / `{"derived": …}`) so
/// the wire shape is stable as variants are added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityBody {
    /// A series, optionally reduced over time. `reduce = None` ⇒ a series (one
    /// value per output time); `Some` ⇒ a scalar.
    Reduced {
        source: QuantitySource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reduce: Option<TemporalReduce>,
    },
    /// Reduction arithmetic: a scalar built from already-defined scalar
    /// quantities (and consts/params). Always a scalar.
    Derived(ScalarExpr),
}

/// What a `Reduced` quantity folds over. **Externally tagged, never untagged** —
/// `State` serializes `{"state": <expr>}` and stays byte-identical when the v1.1
/// `Observation { stream }` variant is appended (a pure additive change: no golden
/// churn, no run-id move). The single v1 variant fixes the wrapper shape now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantitySource {
    /// Latent truth: a quantity-validated `Expr` evaluated against each
    /// trajectory snapshot. `I`, `I / N`, `if I > thresh then 1 else 0`.
    State(Expr),
    /// The simulated observation series of a declared stream (`observations.afp`):
    /// the measurement draw the run already produced — `y_sim` — reduced like any
    /// other series. v1.1. The reduction folds over the stream's per-obs-time
    /// values; the stream must be declared (`E289` otherwise) and materialized by
    /// the run (a runtime check in each command's materialization path).
    Observation { stream: String },
}

// ── Temporal reductions ─────────────────────────────────────────────────────────

/// A reduction over *time*, with the result *kind* in the type so the output
/// dimension is not a function of the runtime variant. `Value` preserves the
/// series dimension; `Time` yields a time (dimension `T`); `Integral` yields
/// `dim(series)·T`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalReduce {
    Value(ValueReduce),
    Time(TimeReduce),
    /// The trapezoidal area under the series (person-time).
    Integral,
}

/// A reduction whose result has the same dimension as the series. (`Total`/`sum`
/// is deferred to the flow source — summing a *stock* over snapshots is
/// cadence-dependent; it is only meaningful for a per-interval flow.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueReduce {
    /// The last value — the right reducer for an absorbing stock (`final(D)`).
    Final,
    Max,
    Min,
    Mean,
    /// Number of output times at which the series exceeds the threshold.
    CountAbove(Expr),
    /// Number of output times at which the series is below the threshold.
    CountBelow(Expr),
}

/// A reduction whose result is a *time* (dimension `T`). A non-firing crossing is
/// right-censored at runtime — that is a property of the per-draw value, not of
/// this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeReduce {
    /// `time_of_max` — the first time the series attains its maximum.
    TimeOfMax,
    /// `time_of_min` — the first time the series attains its minimum.
    TimeOfMin,
    /// `first_above(series, thresh)` — first time the series exceeds the threshold.
    FirstAbove(Expr),
    /// `first_below(series, thresh)` — first time the series falls below it.
    FirstBelow(Expr),
    /// `last_above(series, thresh)` — last time the series exceeds the threshold.
    LastAbove(Expr),
    /// `last_below(series, thresh)` — last time the series falls below it.
    LastBelow(Expr),
}

// ── Reduction arithmetic ────────────────────────────────────────────────────────

/// A reference to an earlier *scalar* quantity, carrying the stratum it resolves
/// in so a cross-stratum mismatch is unconstructible at resolution rather than
/// runtime-checked. (Populated per-cell by the expander, mirroring stratified
/// observation expansion.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QRef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stratum: Vec<StratumKey>,
}

/// Reduction-arithmetic expression: closed, total, scalar-valued. A dedicated ADT
/// *outside* the shared rate `Expr` (the `TriggerExpr` precedent) so a reduced
/// scalar can never appear in a propensity, and a rate leaf can never appear
/// here. Externally tagged single-key objects — derived serde suffices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarExpr {
    Const(f64),
    Param(String),
    /// A previously-defined scalar quantity (topologically ordered).
    QRef(QRef),
    UnOp {
        op: UnOp,
        arg: Box<ScalarExpr>,
    },
    BinOp {
        op: BinOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    Cond {
        pred: Box<ScalarExpr>,
        then: Box<ScalarExpr>,
        #[serde(rename = "else")]
        else_: Box<ScalarExpr>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    fn rt_quantity(q: &Quantity) {
        let json = serde_json::to_string(q).expect("serialize");
        let back: Quantity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*q, back, "quantity round-trip changed value; json was {json}");
    }

    #[test]
    fn round_trips_state_series_and_scalar() {
        // A series state quantity: `prevalence = I / N`, no reduction.
        let series = Quantity {
            name: "prevalence".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::bin_op(
                    BinOp::Div,
                    Expr::pop("I"),
                    Expr::pop("N"),
                )),
                reduce: None,
            },
        };
        // A value-reduction scalar: `peak = max(I / N)`.
        let peak = Quantity {
            name: "peak_prevalence".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::bin_op(
                    BinOp::Div,
                    Expr::pop("I"),
                    Expr::pop("N"),
                )),
                reduce: Some(TemporalReduce::Value(ValueReduce::Max)),
            },
        };
        // A time reduction with an Expr threshold: `first_above(I, i_thresh)`.
        let onset = Quantity {
            name: "takeoff_time".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I_total")),
                reduce: Some(TemporalReduce::Time(TimeReduce::FirstAbove(Expr::param(
                    "i_thresh",
                )))),
            },
        };
        // An integral: `person_days = integral(I)`.
        let pd = Quantity {
            name: "person_days_inf".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I")),
                reduce: Some(TemporalReduce::Integral),
            },
        };
        // A stratified count reduction.
        let pos = Quantity {
            name: "positive_months".into(),
            stratum: vec![StratumKey { dim: "patch".into(), level: "p1".into() }],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I_p1")),
                reduce: Some(TemporalReduce::Value(ValueReduce::CountAbove(Expr::param(
                    "i_thresh",
                )))),
            },
        };
        for q in [&series, &peak, &onset, &pd, &pos] {
            rt_quantity(q);
        }
    }

    #[test]
    fn round_trips_reduction_arithmetic() {
        // `outbreak_dur = fadeout_time - takeoff_time`, with an abs and a cond to
        // exercise UnOp/BinOp/Cond/QRef.
        let dur = Quantity {
            name: "outbreak_dur".into(),
            stratum: vec![StratumKey { dim: "patch".into(), level: "p1".into() }],
            dimension: None,
            body: QuantityBody::Derived(ScalarExpr::UnOp {
                op: UnOp::Abs,
                arg: Box::new(ScalarExpr::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(ScalarExpr::QRef(QRef {
                        name: "fadeout_time".into(),
                        stratum: vec![StratumKey { dim: "patch".into(), level: "p1".into() }],
                    })),
                    right: Box::new(ScalarExpr::Cond {
                        pred: Box::new(ScalarExpr::Const(1.0)),
                        then: Box::new(ScalarExpr::QRef(QRef {
                            name: "takeoff_time".into(),
                            stratum: vec![],
                        })),
                        else_: Box::new(ScalarExpr::Param("t0".into())),
                    }),
                }),
            }),
        };
        rt_quantity(&dur);
    }

    #[test]
    fn pins_wire_tags() {
        // The exact on-wire shape the OCaml serde must match.
        let q = Quantity {
            name: "p".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::State(Expr::pop("I")),
                reduce: Some(TemporalReduce::Time(TimeReduce::TimeOfMax)),
            },
        };
        assert_eq!(
            serde_json::to_string(&q).unwrap(),
            r#"{"name":"p","body":{"reduced":{"source":{"state":{"pop":"I"}},"reduce":{"time":"time_of_max"}}}}"#
        );
        // Integral unit variant, Derived, and a const scalar.
        let d = Quantity {
            name: "d".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Derived(ScalarExpr::Const(2.5)),
        };
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            r#"{"name":"d","body":{"derived":{"const":2.5}}}"#
        );
    }

    #[test]
    fn round_trips_observation_source() {
        // v1.1: `first_above(observations.afp, 0)` — an Observation source.
        let q = Quantity {
            name: "first_afp".into(),
            stratum: vec![],
            dimension: None,
            body: QuantityBody::Reduced {
                source: QuantitySource::Observation { stream: "afp".into() },
                reduce: Some(TemporalReduce::Time(TimeReduce::FirstAbove(Expr::const_(0.0)))),
            },
        };
        rt_quantity(&q);
        // The externally-tagged wire shape the OCaml serde must match.
        assert_eq!(
            serde_json::to_string(&QuantitySource::Observation { stream: "afp".into() }).unwrap(),
            r#"{"observation":{"stream":"afp"}}"#
        );
    }
}
