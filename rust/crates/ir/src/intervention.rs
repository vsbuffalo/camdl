use serde::{Deserialize, Serialize};
use crate::expr::Expr;

// ── Schedule ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurringSchedule {
    pub start:  f64,
    pub period: f64,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FractionTransfer {
    pub src:      String,
    pub dst:      String,
    pub fraction: Expr,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddAction {
    pub compartment: String,
    pub count:       Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    FractionTransfer(FractionTransfer),
    AbsoluteTransfer(AbsoluteTransfer),
    Set(SetAction),
    Add(AddAction),
}

// ── Fire source ─────────────────────────────────────────────────────────────

/// The scope a reactive policy's trigger reads from — the inference-safety axis
/// (gh#204). The IR must carry it so inference can reject unsafe combinations;
/// phase 1 supports only [`SharedExogenous`](AgendaScope::SharedExogenous).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgendaScope {
    /// All particles share one agenda. Trigger inputs are external
    /// observations / deterministic data, identical for every particle at a
    /// boundary, so the shared `Schedule` CRN coupling is preserved. DSL:
    /// `scope = exogenous`.
    SharedExogenous,
    /// Each particle has its own agenda because the trigger reads latent state.
    /// DSL: `scope = particle`. Rejected in inference until agenda state is
    /// part of particle state (later phase).
    ParticleLocal,
}

/// A reactive (state/observation-triggered) fire source (gh#204): fire when
/// `when_` holds, `after` a non-negative lag, optionally rate-limited by
/// `cooldown`. The action grammar and effect resolution are shared with
/// scheduled interventions — only the fire source differs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactiveTrigger {
    /// The trigger predicate — a boolean expression over allowed trigger inputs
    /// (`observed(stream)`, `sum_observed(stream, window=...)`, parameters,
    /// constants). `when` is a Rust keyword, so the field is `when_`; the wire
    /// key stays `when`.
    #[serde(rename = "when")]
    pub when_: Expr,
    /// Non-negative lag (model time units) between the trigger firing and the
    /// effect being applied. Default `0`.
    pub after: f64,
    /// Fire-and-disable: `true` ⇒ the policy fires at most once. Mutually
    /// exclusive with `cooldown` (rejected by the compiler).
    pub once: bool,
    /// Minimum time between firings when `once = false`. Absent ⇒ no rate limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown: Option<f64>,
    /// Trigger scope — the inference-safety axis (see [`AgendaScope`]).
    pub scope: AgendaScope,
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
