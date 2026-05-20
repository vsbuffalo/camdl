use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    expr::Expr,
    intervention::Intervention,
    observation::ObservationModel,
    ode_equation::OdeEquation,
    parameter::Parameter,
    table::Table,
    time_func::TimeFunction,
    transition::Transition,
};

fn default_time_unit() -> String { "days".to_string() }

// ── Compartment ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompartmentKind {
    Integer,
    Real,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compartment {
    pub name: String,
    pub kind: CompartmentKind,
}

// ── Initial conditions ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialConditions {
    Explicit(HashMap<String, f64>),
    Parameterized(HashMap<String, Expr>),
    FromDistribution(HashMap<String, crate::parameter::PriorDist>),
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegularOutputSchedule {
    pub start: f64,
    pub step:  f64,
    pub end:   f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputSchedule {
    Regular(RegularOutputSchedule),
    AtTimes(Vec<f64>),
    MatchObservations,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    pub times:        OutputSchedule,
    pub format:       String,
    pub trajectory:   bool,
    pub observations: bool,
}

// ── Simulation config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub t_start:        f64,
    pub t_end:          f64,
    pub time_semantics: String,
    pub dt:             Option<f64>,
    pub rng_seed:       Option<i64>,
}

// ── Presets ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name:   String,
    pub label:  String,
    pub params: HashMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enable:  Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disable: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub scale:   HashMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compose: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_end:  Option<f64>,
}

// ── Model structure ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dimension {
    pub name:   String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStructure {
    pub dimensions:               Vec<Dimension>,
    pub compartment_dims:         HashMap<String, Vec<String>>,
    pub base_compartments:        Vec<String>,
    pub transmission_transitions: Vec<String>,
    pub infectious_compartments:  Vec<String>,
}

// ── Balance constraint ───────────────────────────────────────────────────────

/// A balance constraint forces one compartment to absorb demographic residuals.
/// After all transitions and interventions, the target compartment is overwritten
/// with the value of the expression (typically `pop(t) - S - E - I`).
///
/// This matches pomp's `R = nearbyint(pop) - S - E - I` pattern for models
/// where the population trajectory is externally specified and the demographic
/// rates don't exactly reproduce it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BalanceSpec {
    pub target: String,
    pub expr: Expr,
}

// ── Top-level model ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub name:               String,
    pub version:            String,
    #[serde(default = "default_time_unit")]
    pub time_unit:          String,
    pub description:        Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin:             Option<String>,
    pub compartments:       Vec<Compartment>,
    pub transitions:        Vec<Transition>,
    pub ode_equations:      Vec<OdeEquation>,
    pub time_functions:     Vec<TimeFunction>,
    pub tables:             Vec<Table>,
    pub interventions:      Vec<Intervention>,
    pub observations:       Vec<ObservationModel>,
    pub parameters:         Vec<Parameter>,
    pub initial_conditions: InitialConditions,
    pub output:             OutputConfig,
    pub simulation:         SimulationConfig,
    #[serde(default, rename = "scenarios")]
    pub presets:            Vec<Preset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_structure:    Option<ModelStructure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balance:            Option<BalanceSpec>,
    /// Compartments whose individuals carry tracked IDs (individual-sampling
    /// layer, 2026-05-19 proposal). Forward-reachable closure from
    /// `#[lineage]` event destinations ∪ parent pools. Empty when no
    /// `#[lineage]` annotations exist — the lineage subsystem is then
    /// statically inert. Cached here so the runtime does not recompute it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identity_tracked_compartments: Vec<String>,
}
