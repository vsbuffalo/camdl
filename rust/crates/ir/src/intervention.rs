use serde::{Deserialize, Serialize};
use crate::expr::Expr;

// ── Schedule ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurringSchedule {
    pub start:  f64,
    pub period: f64,
    /// The window end. Compiler-baked from the model horizon when the schedule
    /// declares no `to`, so it is NaN while that horizon is an unresolved
    /// anchor — see [`crate::anchor::null_as_nan`].
    #[serde(with = "crate::anchor::null_as_nan")]
    pub end:    f64,
    /// Day within each period when the event fires. Fire times are
    /// `at_day + k * period` for the smallest k where target >= start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_day: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionSchedule {
    AtTimes(Vec<f64>),
    /// gh#69: parametric `at [...]` lists. Each `Expr` is evaluated
    /// once per simulation start against the current `params` vector
    /// to yield a concrete fire time. The OCaml expander emits this
    /// variant only when at least one entry references a parameter
    /// (or other non-constant expression); fully-constant lists stay
    /// in `AtTimes` so existing golden IRs remain byte-identical.
    AtTimesExpr(Vec<Expr>),
    Recurring(RecurringSchedule),
}

// ── Actions ───────────────────────────────────────────────────────────────────
//
// The three smooth-jump actions carry `∂amount/∂θ` (`*_grad`, a
// [`ParamGradMap`](crate::deriv::ParamGradMap)): the source term of the
// event-jump sensitivity `S⁺ = ∂Φ/∂x·S⁻ + ∂Φ/∂θ` the ODE gradient path applies
// when the effect fires (gh#275 §1g). The `rate_grad` analogue for an action's
// amount: emitted by the compiler's `differentiate_action`, keyed by parameter
// name, an absent key a genuine zero, a live-but-undifferentiated coefficient a
// serialized `DerivEntry::Unsupported` the gradient gate refuses on. Omitted on
// the wire when empty (a constant amount). `AbsoluteTransfer` carries none: its
// jump `min(count, source)` is nonsmooth, so the gate refuses it rather than
// differentiate it. Not run identity (gradient maps are compiler-derived;
// `runid::ir_hash`, SV = 2).

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FractionTransfer {
    pub src:      String,
    pub dst:      String,
    pub fraction: Expr,
    /// `∂fraction/∂θ` (see the module note above).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub fraction_grad: crate::deriv::ParamGradMap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbsoluteTransfer {
    pub src:   String,
    pub dst:   String,
    pub count: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetAction {
    pub compartment: String,
    pub value:       Expr,
    /// `∂value/∂θ` (see the module note above).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub value_grad:  crate::deriv::ParamGradMap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddAction {
    pub compartment: String,
    pub count:       Expr,
    /// `∂count/∂θ` (see the module note above).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub count_grad:  crate::deriv::ParamGradMap,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    FractionTransfer(FractionTransfer),
    AbsoluteTransfer(AbsoluteTransfer),
    Set(SetAction),
    Add(AddAction),
}

// ── Reactive trigger predicate ───────────────────────────────────────────────

/// A comparison operator in a reactive trigger predicate (gh#204). Deliberately
/// a *separate* type from the rate-expression `BinOp` comparisons: a trigger is
/// a different language with different leaves, so its operators do not belong in
/// the shared `Expr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
}

/// How a windowed observation stream is reduced to a single trigger value
/// (gh#204). `observed(s)` is `Latest`; `sum_observed(s, window=..)` is `Sum`.
/// `Mean`/`Max` are carried by the IR for future DSL surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsReducer {
    /// The most recent observed value of the stream (`observed(s)`).
    Latest,
    /// Sum over the trailing window (`sum_observed(s, window=..)`).
    Sum,
    /// Mean over the trailing window.
    Mean,
    /// Maximum over the trailing window.
    Max,
}

/// The observed quantity a trigger comparison reads — policy-visible
/// surveillance history, NOT latent model state (gh#204). An enum (one variant
/// today) so latent/derived quantities can be added behind their own scope gate
/// later. Crucially this lives outside the shared rate `Expr`, so
/// `observed(...)` cannot appear in a transition rate by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerQuantity {
    /// `observed(stream)` / `sum_observed(stream, window=..)`: the observation
    /// data stream visible to policy, reduced over an optional trailing window.
    Observed {
        /// The observation stream name (expanded, e.g. `weekly_afp_borno`).
        stream: String,
        /// Trailing window length in model time units; `None` ⇒ the current
        /// observation only (point value).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<f64>,
        reducer: ObsReducer,
    },
}

/// The right-hand side a trigger comparison tests against (gh#204). A static
/// scalar: a literal or a parameter (resolved once per likelihood evaluation,
/// shared across particles). (A restricted static `Expr` may be added later;
/// phase 1 keeps it to the two cases the examples use.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerThreshold {
    /// A numeric literal threshold.
    Const(f64),
    /// A model parameter used as the threshold.
    Param(String),
}

/// A reactive trigger predicate (gh#204) — a boolean over observed-quantity
/// comparisons. A dedicated ADT rather than the shared `Expr`: it is
/// boolean-valued *by construction* (so a non-boolean `when` is unrepresentable,
/// not merely rejected), and its `observed(...)` leaves cannot leak into rate
/// expressions. Evaluated by no backend yet — the capability gate rejects
/// reactive models at dispatch (PR1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerExpr {
    /// A single comparison `quantity <op> threshold`.
    Cmp {
        lhs: TriggerQuantity,
        op: CmpOp,
        rhs: TriggerThreshold,
    },
    /// Both sub-predicates must hold.
    And(Box<TriggerExpr>, Box<TriggerExpr>),
    /// Either sub-predicate holds.
    Or(Box<TriggerExpr>, Box<TriggerExpr>),
    /// Negation.
    Not(Box<TriggerExpr>),
}

// ── Fire source ─────────────────────────────────────────────────────────────

/// A reactive (state/observation-triggered) fire source (gh#204): fire when
/// `when_` holds, `after` a non-negative lag, optionally rate-limited by
/// `cooldown`. The action grammar and effect resolution are shared with
/// scheduled interventions — only the fire source differs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactiveTrigger {
    /// The trigger predicate — a [`TriggerExpr`] over `observed(stream)` /
    /// `sum_observed(stream, window=..)` comparisons. `when` is a Rust keyword,
    /// so the field is `when_`; the wire key stays `when`.
    #[serde(rename = "when")]
    pub when_: TriggerExpr,
    /// Non-negative lag (model time units) between the trigger firing and the
    /// effect being applied. Default `0`.
    pub after: f64,
    /// Fire-and-disable: `true` ⇒ the policy fires at most once. Mutually
    /// exclusive with `cooldown` (rejected by the compiler).
    pub once: bool,
    /// Minimum time between firings when `once = false`. Absent ⇒ no rate limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown: Option<f64>,
}

/// How an intervention's fire times are produced (gh#204). Orthogonal to
/// [`InterventionKind`] (the toggling/structural axis): a reactive policy is
/// `kind = Scenario, fire = Reactive(..)`. Splitting the fire source from the
/// kind makes the illegal pairings unrepresentable — a scheduled time list on
/// a reactive policy, or a trigger on an `Event`, cannot be constructed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FireSource {
    /// Fixed schedule: `at [...]`, recurring, or parametric `at`. Used by both
    /// `Scenario` and `Event` interventions.
    Scheduled(InterventionSchedule),
    /// State/observation-triggered policy. Fire times are discovered at runtime
    /// by the reactive agenda; no backend executes them yet — the capability
    /// gate rejects reactive models at dispatch (PR1).
    Reactive(ReactiveTrigger),
}

impl FireSource {
    /// The static schedule, if this is a scheduled fire source. `Reactive`
    /// sources have no static schedule (their fire times are discovered at
    /// runtime), so the fire-time machinery skips them — and the capability
    /// gate rejects reactive models before any backend runs.
    pub fn schedule(&self) -> Option<&InterventionSchedule> {
        match self {
            FireSource::Scheduled(s) => Some(s),
            FireSource::Reactive(_) => None,
        }
    }

    /// True for a `Reactive` fire source. Used by `required_capabilities` to
    /// raise the `REACTIVE_INTERVENTIONS` flag.
    pub fn is_reactive(&self) -> bool {
        matches!(self, FireSource::Reactive(_))
    }
}

// ── Intervention ──────────────────────────────────────────────────────────────

/// Distinguishes the two DSL constructs that both lower to [`Intervention`]
/// (gh#107). Replaces the former `always_active: bool` — a named enum names the
/// distinction without bolting on a second bool. This is the *toggling*
/// (structural) axis; how fire times are *produced* is the orthogonal
/// [`FireSource`] axis (gh#204), so a reactive policy is `kind = Scenario,
/// fire = Reactive(..)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionKind {
    /// `interventions {}` — toggled by enable/disable/set/scale scenarios.
    #[default]
    Scenario,
    /// `events {}` — fires unconditionally every substep.
    Event,
}

impl InterventionKind {
    /// True for `Scenario` — the serialisation default, skipped on the wire
    /// (mirrors the former `always_active` skip-false discipline, so a
    /// scenario intervention carries no `kind` key).
    pub fn is_scenario(&self) -> bool {
        matches!(self, Self::Scenario)
    }
    /// True for `Event` — fires unconditionally, not scenario-toggled.
    /// Reads at call sites exactly where `always_active` did.
    pub fn is_event(&self) -> bool {
        matches!(self, Self::Event)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    pub name:     String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_name: Option<String>,
    /// How fire times are produced — a fixed schedule
    /// ([`FireSource::Scheduled`]) or a reactive trigger
    /// ([`FireSource::Reactive`], gh#204). Replaces the former
    /// `schedule: InterventionSchedule`.
    pub fire: FireSource,
    pub actions:  Vec<Action>,
    /// Which DSL construct declared this — `Event` (fires unconditionally,
    /// from `events {}`) or `Scenario` (scenario-toggled, from
    /// `interventions {}`). Absent on the wire ⇒ `Scenario` (the default).
    #[serde(default, skip_serializing_if = "InterventionKind::is_scenario")]
    pub kind: InterventionKind,
}

#[cfg(test)]
mod reactive_serde_tests {
    use super::*;

    /// Pins the OCaml→Rust wire contract for a reactive intervention (gh#204):
    /// the exact JSON `camdlc` emits must deserialize to the expected structure,
    /// and round-trip. A drift in either serde impl trips this.
    #[test]
    fn reactive_intervention_round_trips_camdlc_wire_shape() {
        let json = r#"{"name":"sia","fire":{"reactive":{"after":21.0,"cooldown":180.0,"once":false,"when":{"cmp":{"lhs":{"observed":{"reducer":"sum","stream":"weekly_afp","window":28.0}},"op":"ge","rhs":{"param":"afp_trigger_threshold"}}}}},"actions":[{"fraction_transfer":{"src":"S","dst":"V","fraction":{"param":"sia_coverage"}}}]}"#;
        let iv: Intervention =
            serde_json::from_str(json).expect("deserialize reactive intervention");
        assert_eq!(iv.name, "sia");
        // No `kind` key on the wire ⇒ Scenario (a reactive policy is toggleable).
        assert_eq!(iv.kind, InterventionKind::Scenario);
        let trig = match &iv.fire {
            FireSource::Reactive(t) => t,
            FireSource::Scheduled(_) => panic!("expected a reactive fire source"),
        };
        assert_eq!(trig.after, 21.0);
        assert!(!trig.once);
        assert_eq!(trig.cooldown, Some(180.0));
        match &trig.when_ {
            TriggerExpr::Cmp {
                lhs: TriggerQuantity::Observed { stream, window, reducer },
                op,
                rhs,
            } => {
                assert_eq!(stream, "weekly_afp");
                assert_eq!(*window, Some(28.0));
                assert_eq!(*reducer, ObsReducer::Sum);
                assert_eq!(*op, CmpOp::Ge);
                assert_eq!(*rhs, TriggerThreshold::Param("afp_trigger_threshold".into()));
            }
            _ => panic!("expected a single observed >= param comparison"),
        }
        let reser = serde_json::to_string(&iv).expect("re-serialize");
        let iv2: Intervention = serde_json::from_str(&reser).expect("re-deserialize");
        assert_eq!(iv, iv2, "reactive intervention must round-trip");
    }

    /// Pins the OCaml→Rust wire contract for an action's `∂amount/∂θ` (gh#275
    /// §1g): the keys `camdlc` emits (`count_grad`, `value_grad`,
    /// `fraction_grad`) must land in the fields the ODE gradient path reads. A
    /// key mismatch between the two serde implementations would not fail
    /// deserialization — `#[serde(default)]` would silently leave the map
    /// empty, and the event jump would carry a zero source term.
    #[test]
    fn action_amount_grad_round_trips_camdlc_wire_shape() {
        use crate::deriv::DerivEntry;
        let json = r#"{"name":"importation","fire":{"scheduled":{"at_times":[200.0]}},"actions":[{"add":{"compartment":"I","count":{"param":"seed"},"count_grad":{"seed":{"grad":{"const":1.0}}}}},{"set":{"compartment":"V","value":{"param":"v0"},"value_grad":{"v0":{"grad":{"const":1.0}}}}},{"fraction_transfer":{"src":"S","dst":"V","fraction":{"param":"cov"},"fraction_grad":{"cov":{"grad":{"const":1.0}}}}},{"absolute_transfer":{"src":"S","dst":"V","count":{"const":5.0}}}],"kind":"event"}"#;
        let iv: Intervention = serde_json::from_str(json).expect("deserialize");
        let one = DerivEntry::Grad(Expr::const_(1.0));
        match &iv.actions[0] {
            Action::Add(a) => assert_eq!(a.count_grad.get("seed"), Some(&one)),
            other => panic!("expected add, got {other:?}"),
        }
        match &iv.actions[1] {
            Action::Set(s) => assert_eq!(s.value_grad.get("v0"), Some(&one)),
            other => panic!("expected set, got {other:?}"),
        }
        match &iv.actions[2] {
            Action::FractionTransfer(ft) => assert_eq!(ft.fraction_grad.get("cov"), Some(&one)),
            other => panic!("expected fraction_transfer, got {other:?}"),
        }
        let reser = serde_json::to_string(&iv).expect("re-serialize");
        assert!(reser.contains("\"count_grad\"") && reser.contains("\"value_grad\"")
            && reser.contains("\"fraction_grad\""), "{reser}");
        let iv2: Intervention = serde_json::from_str(&reser).expect("re-deserialize");
        assert_eq!(iv, iv2, "action gradients must round-trip");

        // An empty map is omitted on the wire (golden-neutral for a constant amount).
        let plain = Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::const_(5.0),
            count_grad: Default::default(),
        });
        let s = serde_json::to_string(&plain).unwrap();
        assert!(!s.contains("count_grad"), "an empty amount gradient must be omitted: {s}");
    }

    /// The compound predicate variants (and/or/not) and the other reducers /
    /// threshold cases round-trip.
    #[test]
    fn compound_trigger_round_trips() {
        let trig = ReactiveTrigger {
            when_: TriggerExpr::And(
                Box::new(TriggerExpr::Cmp {
                    lhs: TriggerQuantity::Observed {
                        stream: "a".into(),
                        window: None,
                        reducer: ObsReducer::Latest,
                    },
                    op: CmpOp::Gt,
                    rhs: TriggerThreshold::Const(1.0),
                }),
                Box::new(TriggerExpr::Not(Box::new(TriggerExpr::Cmp {
                    lhs: TriggerQuantity::Observed {
                        stream: "b".into(),
                        window: Some(7.0),
                        reducer: ObsReducer::Mean,
                    },
                    op: CmpOp::Le,
                    rhs: TriggerThreshold::Param("k".into()),
                }))),
            ),
            after: 0.0,
            once: true,
            cooldown: None,
        };
        let fs = FireSource::Reactive(trig);
        let json = serde_json::to_string(&fs).expect("serialize");
        let fs2: FireSource = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(fs, fs2, "compound reactive fire source must round-trip");
    }
}
