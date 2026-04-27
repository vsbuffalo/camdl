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
    Recurring(RecurringSchedule),
    External(String),
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

// ── Intervention ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    pub name:     String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_name: Option<String>,
    pub schedule: InterventionSchedule,
    pub actions:  Vec<Action>,
    /// If true, this event fires unconditionally (not toggled by scenarios).
    /// Events declared in `events {}` have this set to true.
    /// Interventions declared in `interventions {}` default to false.
    /// Defaults to `true` so that old IR files without this field
    /// deserialize as unconditional (the original intent).
    #[serde(default = "bool_true")]
    pub always_active: bool,
}

fn bool_true() -> bool { true }
