use serde::{Deserialize, Serialize};
use crate::expr::Expr;
use crate::deriv::Diffable;
use crate::Differentiate;
use crate::parameter::ParamKind;

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
    /// Σ wᵢ · incidence(flowᵢ) — a per-term-weighted union of flows, accumulated
    /// over the reporting interval (Increment B1 of the 2026-07-31 aggregation
    /// proposal). The case a unit-weighted [`Projection::CumulativeFlowSum`]
    /// cannot express: several strata pooled into ONE observed column with a
    /// DIFFERENT reporting rate each.
    ///
    /// Weights are constant over the observation interval (enforced in the
    /// frontend, B3): the projection is evaluated once, at the observation
    /// instant, so `w(t_obs)·ΣΔN` equals `∫w(s)dN(s)` only when `w` does not
    /// move. That is why a weight may not read `t`, a time function, a `cond`,
    /// or state.
    ///
    /// Deliberately a projection VARIANT rather than a flow-read node in
    /// [`Expr`]: it keeps [`Projection::temporal_kind`] a total function of the
    /// variant, and makes `∂proj/∂flowᵢ = wᵢ` structurally free.
    WeightedFlowSum(Vec<WeightedFlow>),
}

/// One `weight × flow` term of a [`Projection::WeightedFlowSum`].
///
/// A named struct rather than the proposal's `(Expr, String)` tuple — a
/// documented deviation. The IR is read by humans (every golden diff is
/// reviewed), and `{"weight": …, "flow": "infection_child"}` says which side is
/// which where `[…, "infection_child"]` does not. The wire cost is one key per
/// term; the review cost of a positional pair is paid on every golden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedFlow {
    /// Constant over the observation interval — see [`Projection::WeightedFlowSum`].
    pub weight: Expr,
    /// The transition whose interval-accumulated flow this term weights.
    pub flow: String,
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
            Projection::CumulativeFlow(_)
            | Projection::CumulativeFlowSum(_)
            | Projection::WeightedFlowSum(_) => {
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

// Each differentiable likelihood argument is a [`Diffable`] — its expression
// paired with its per-parameter classified gradient (`Known | Unsupported` via
// [`DerivEntry`]), populated by the OCaml obs-gradient autodiff pass. `n`
// (Binomial/BetaBinomial) is **not** a `Diffable` — it must be θ-independent and
// carries no gradient, so it is a bare [`Expr`] carrying `#[differentiate(skip)]`
// (an unskipped non-`Diffable` field would fail to compile — the seal).
//
// `#[derive(Differentiate)]` folds every `Diffable` field into `diffables()`, so
// the fit-time preflight and the run_id hash iterate the positions uniformly and
// a new argument cannot be forgotten. On the wire a `Diffable` field is the
// nested shape `{"mean": {"expr": …, "grad": …}}` (grad omitted when empty).

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct PoissonLikelihood {
    pub rate: Diffable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct NegBinomialLikelihood {
    pub mean:       Diffable,
    pub dispersion: Diffable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct NormalLikelihood {
    pub mean: Diffable,
    pub sd:   Diffable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct BinomialLikelihood {
    /// The number of trials. Must be θ-independent (a constant or an observed
    /// data column) — it is rounded to an integer, so it carries no gradient.
    #[differentiate(skip)]
    pub n: Expr,
    pub p: Diffable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct BetaBinomialLikelihood {
    /// The number of trials. Must be θ-independent (see [`BinomialLikelihood::n`]).
    #[differentiate(skip)]
    pub n:     Expr,
    pub alpha: Diffable,
    pub beta:  Diffable,
}

/// Beta likelihood for a continuous proportion `x ∈ (0, 1)` (an observed rate,
/// coverage, or positivity given directly as a fraction — not a `k`-of-`n` count,
/// which is [`BetaBinomialLikelihood`]). Mean-linked like [`NegBinomialLikelihood`]:
/// the model predicts `mean` and `concentration` (φ) is the dispersion knob, with
/// shape parameters `α = mean·φ`, `β = (1 − mean)·φ`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct BetaLikelihood {
    pub mean:          Diffable,
    pub concentration: Diffable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct BernoulliLikelihood {
    pub p: Diffable,
}

/// Zero-inflated negative binomial: a structural-zero mass `pi` mixed with a
/// `NegBinomial(mean, dispersion)`. `P(Y=0) = pi + (1-pi)·f(0)`,
/// `P(Y=k>0) = (1-pi)·f(k)`. **Scoring-only** — every field is a bare `Expr`
/// (no `Diffable`), so the family carries no gradient at all; the fit-time
/// gradient-capability gate refuses PGAS/NUTS on a model that uses it, while
/// MH/PMMH/PF/IF2 score it. The surface is the `zero_inflated(base =
/// neg_binomial(...), pi = ...)` wrapper, desugared to this flat variant by the
/// OCaml parser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
pub struct ZeroInflatedNegBinomialLikelihood {
    #[differentiate(skip)]
    pub mean: Expr,
    #[differentiate(skip)]
    pub dispersion: Expr,
    #[differentiate(skip)]
    pub pi: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Differentiate)]
#[serde(rename_all = "snake_case")]
pub enum Likelihood {
    Poisson(PoissonLikelihood),
    NegBinomial(NegBinomialLikelihood),
    Normal(NormalLikelihood),
    Binomial(BinomialLikelihood),
    BetaBinomial(BetaBinomialLikelihood),
    Beta(BetaLikelihood),
    Bernoulli(BernoulliLikelihood),
    ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood),
}

impl Likelihood {
    /// The snake_case family name (`"poisson"`, `"neg_binomial"`, …), matching
    /// the `serde` variant tag. Used to label a stream's likelihood in the
    /// machine-readable observation schema without re-deriving the mapping at
    /// each call site.
    pub fn name(&self) -> &'static str {
        match self {
            Likelihood::Poisson(_)      => "poisson",
            Likelihood::NegBinomial(_)  => "neg_binomial",
            Likelihood::Normal(_)       => "normal",
            Likelihood::Binomial(_)     => "binomial",
            Likelihood::BetaBinomial(_) => "beta_binomial",
            Likelihood::Beta(_)         => "beta",
            Likelihood::Bernoulli(_)    => "bernoulli",
            Likelihood::ZeroInflatedNegBinomial(_) => "zero_inflated_neg_binomial",
        }
    }
}

// ── Observation schedule ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegularSchedule {
    pub start: f64,
    pub step:  f64,
    /// The emit window's end, baked from the model horizon — NaN while that
    /// horizon is an unresolved anchor (see [`crate::anchor::null_as_nan`]).
    /// Since gh#143/gh#561 the runtime derives emission from the RUN horizon
    /// and ignores this value, and the gh#616 resolver overwrites it with the
    /// resolved horizon so it never stays NaN on a model that runs.
    #[serde(with = "crate::anchor::null_as_nan")]
    pub end:   f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSchedule {
    AtTimes(Vec<f64>),
    Regular(RegularSchedule),
}

// ── Declared file columns ──────────────────────────────────────────────────────

/// The role of a declared file column (the `columns { name : role }` block;
/// 2026-06-10 observation data-entry §2.2).
///
/// Serialises to match the OCaml IR: `Time` → the bare string `"time"`;
/// `Dim(d)` → `{"dim": d}`; `Value(k)` → `{"value": "<param_kind>"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    /// The time axis (exactly one per stream) — the FIT time source.
    Time,
    /// A model dimension; values bind to that dimension's levels.
    Dim(String),
    /// An observed value of the given DSL type (count/real/probability/…) —
    /// either the `~` LHS (scored) or RHS-referenced auxiliary data.
    Value(ParamKind),
}

/// One declared file column: header name + role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObsColumn {
    pub name: String,
    pub role: ColumnRole,
}

/// One `(dimension, level)` pair identifying a stratum cell. Named fields
/// (not a tuple) so an illegal half-built selector is unrepresentable, and
/// the JSON shape mirrors `ColumnRole::Dim`'s `{"dim": d}` object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StratumKey {
    pub dim:   String,
    pub level: String,
}

// ── Observation model ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationModel {
    pub name:          String,
    /// The `from <label>` data-source key — the thing `--data label=file`
    /// binds a file to (defaults to `name`).
    pub source:        String,
    /// The explicit file schema. The loader binds the data file's columns by
    /// these declared names; the `Time` column is the fit time source.
    pub columns:       Vec<ObsColumn>,
    /// The `~` LHS — the declared value column the likelihood scores.
    pub scored:        String,
    /// SIMULATE-only emission cadence (`emit_schedule`). The fit path reads
    /// the data file's time column and never consults this; `None` for a
    /// fit-only model that omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emit_schedule: Option<ObservationSchedule>,
    /// For a stratified observation stream (`cases[p in patch] ~ ...`), the
    /// (dimension, level) pairs identifying which stratum cell this expanded
    /// leaf observes — `[{dim: "patch", level: "p1"}]`. Empty for an
    /// unstratified stream. Populated by the OCaml expander from the stream's
    /// header indices; the long-form loader routes each file row to the leaf
    /// whose `stratum` matches the row's `: dim` column values BY NAME.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stratum:       Vec<StratumKey>,
    pub projection:    Projection,
    /// ∂projection/∂compartment for a `DerivedExpr` (nonlinear) projection — the
    /// WrtPop differentiation the ODE observation gradient's factor-2 chain
    /// consumes (`∂proj/∂θ = Σ_j ∂proj/∂x_j · S[j]`, gh#275 §1h). Empty (and
    /// omitted) for a linear projection (`CurrentPop*`/`CumulativeFlow*` — a
    /// trivial selection, not a nonlinear function of state) and on gradient-free
    /// backends; populated by the OCaml WrtPop pass (the `rate_state_grad`
    /// analogue for a projection expression).
    #[serde(default, skip_serializing_if = "crate::deriv::CompGradMap::is_empty")]
    pub projection_state_grad: crate::deriv::CompGradMap,
    pub likelihood:    Likelihood,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ConstExpr, Expr};

    /// The OCaml↔Rust contract for `WeightedFlowSum`, pinned to the EXACT bytes
    /// the OCaml emitter produces.
    ///
    /// The two sides derive their serde independently — OCaml hand-writes
    /// `projection_to_json` in `serde.ml`, Rust uses `#[derive(Deserialize)]` —
    /// so nothing structural forces them to agree. The golden corpus contains no
    /// `WeightedFlowSum`, so the round-trip suite that walks the goldens does not
    /// exercise this variant on either side: without this test the first model to
    /// use it would have been the integration test.
    ///
    /// The literal below is copied from the OCaml side's pinned wire assertion
    /// (`ocaml/test/test_ir_roundtrip.ml`, `weighted_flow_sum_serde_test`). If
    /// either emitter changes, one of the two tests fails — which is the point.
    #[test]
    fn weighted_flow_sum_accepts_the_ocaml_wire_bytes() {
        // Exactly what camdlc writes.
        let ocaml_wire = r#"{"weighted_flow_sum":[{"weight":{"param":"rho_child"},"flow":"infection_child"},{"weight":{"bin_op":{"op":"mul","left":{"param":"rho_adult"},"right":{"const":0.5}}},"flow":"infection_adult"}]}"#;

        let p: Projection = serde_json::from_str(ocaml_wire)
            .expect("Rust must accept the bytes camdlc emits for weighted_flow_sum");

        match &p {
            Projection::WeightedFlowSum(terms) => {
                assert_eq!(terms.len(), 2, "both terms survive deserialization");
                assert_eq!(terms[0].flow, "infection_child");
                assert_eq!(terms[1].flow, "infection_adult");
                // Order is identity (see the run_id hash arm) — assert it, so a
                // reordering shows up here rather than as a silent re-key.
                assert!(matches!(terms[0].weight, Expr::Param(_)),
                    "first weight is a bare param");
            }
            other => panic!("expected WeightedFlowSum, got {other:?}"),
        }

        // And Rust must re-emit the same bytes, or a round-trip through the
        // runtime would silently rewrite a model's IR.
        let reemitted = serde_json::to_string(&p).expect("serializes");
        assert_eq!(reemitted, ocaml_wire,
            "Rust re-emission must be byte-identical to the OCaml wire");

        assert_eq!(p.temporal_kind(), TemporalKind::Interval,
            "a weighted flow union accumulates over the interval");
    }

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
