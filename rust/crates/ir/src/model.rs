use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{
    contrast::Contrast,
    expr::Expr,
    intervention::Intervention,
    observation::ObservationModel,
    ode_equation::OdeEquation,
    parameter::Parameter,
    quantity::Quantity,
    table::Table,
    time_func::TimeFunction,
    transition::Transition,
};

fn default_time_unit() -> String { "days".to_string() }

/// ODE integrator selection (gh#166). `Rk45` carries its adaptive tolerances, so
/// the orphan state — tolerances without rk45, or rk4 with tolerances — is
/// UNREPRESENTABLE (illegal-states-unrepresentable). Serializes internally-tagged:
/// `{"method":"rk4"}` / `{"method":"rk45","atol":…,"rtol":…}`; omitted entirely
/// from `simulation_config` at the `Rk4` default (no IR-body change for the
/// pre-gh#166 corpus).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Integrator {
    /// Fixed-step classic RK4 (the default).
    #[default]
    Rk4,
    /// Adaptive Dormand–Prince RK4(5). `atol`/`rtol` are dimensionless; `None` →
    /// the runtime's calibrated default.
    Rk45 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        atol: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rtol: Option<f64>,
    },
}

fn is_default_integrator(i: &Integrator) -> bool { matches!(i, Integrator::Rk4) }

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputSchedule {
    Regular(RegularOutputSchedule),
    AtTimes(Vec<f64>),
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
    /// ODE integrator + (for rk45) its adaptive tolerances (gh#166). `default`/
    /// `skip_serializing_if` so pre-gh#166 IR (no field) deserializes to `Rk4`
    /// and a default model adds no JSON noise.
    #[serde(default, skip_serializing_if = "is_default_integrator")]
    pub integrator:     Integrator,
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
    /// Compiler-derived proleptic-Gregorian day number of `origin`. The
    /// runtime reads this so it never re-parses the origin string
    /// (2026-05-22 calendar-time §6.2). Derived by the OCaml compiler via
    /// the same `days_of_date` the `date()` literal path uses, so it cannot
    /// drift from `caltime::rata_die`. `None` when no `origin` is declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_rata_die:    Option<i64>,
    pub compartments:       Vec<Compartment>,
    pub transitions:        Vec<Transition>,
    pub ode_equations:      Vec<OdeEquation>,
    pub time_functions:     Vec<TimeFunction>,
    pub tables:             Vec<Table>,
    pub interventions:      Vec<Intervention>,
    pub observations:       Vec<ObservationModel>,
    pub parameters:         Vec<Parameter>,
    /// Fix B: shared per-coordinate bindings, topologically ordered. `default`
    /// so pre-Fix-B IR (no field) deserializes to empty; `skip_serializing_if`
    /// so an empty list adds no JSON noise (inc1a emits none).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings:           Vec<Binding>,
    /// gh#272 LICM: param/table-only loop-invariant bindings, topologically
    /// ordered. Produced by the LICM pass (post-autodiff), which is on by default
    /// (`CAMDL_NO_LICM` / `--no-licm` disables it); empty only when the pass is
    /// off or the model has no hoistable subexpression.
    /// `default`/`skip_serializing_if` mirror `bindings` so an empty field is
    /// omitted (byte-identical to a model that produced no per-eval bindings).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_eval_bindings:  Vec<Binding>,
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
    /// Generated quantities (proposal 2026-06-25): named reductions of what a
    /// simulation produces — derived reports, NOT scored data. `default`/
    /// `skip_serializing_if` so a model with no `quantities {}` block is
    /// byte-identical (the field is omitted), and **excluded from
    /// `Model::hash_into`** — a quantity is non-identity and must never re-key a
    /// sim/fit (the one Model field deliberately outside the run-id walk).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantities: Vec<Quantity>,
    /// Counterfactual contrasts (proposal 2026-06-25): named differences of two
    /// run-rooted operands (cases averted). Like `quantities`, reporting-only and
    /// **excluded from `Model::hash_into`** — a contrast is non-identity and must
    /// never re-key a sim/fit. `default`/`skip_serializing_if` so a model with no
    /// `contrasts {}` block is byte-identical (the field is omitted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contrasts: Vec<Contrast>,
}

/// A model-level shared binding (Fix B): a named value (e.g. N[l], I_agg[l],
/// spatial force F[l]) referenced by `Expr::BindingRef`, defined once instead of
/// inlined into every (patch,age) rate. Topologically ordered — a binding's body
/// may reference earlier bindings via `BindingRef`. Evaluated on-demand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    pub name: String,
    pub expr: Expr,
}

#[cfg(test)]
mod integrator_serde_tests {
    use super::Integrator;

    // Pins the cross-language JSON contract for the tagged integrator — exactly
    // the forms the OCaml compiler emits (internally-tagged on "method"). gh#166.
    #[test]
    fn integrator_tagged_json_roundtrips() {
        let cases = [
            (Integrator::Rk4, r#"{"method":"rk4"}"#),
            (Integrator::Rk45 { atol: None, rtol: None }, r#"{"method":"rk45"}"#),
            (
                Integrator::Rk45 { atol: Some(1e-8), rtol: Some(1e-6) },
                r#"{"method":"rk45","atol":1e-8,"rtol":1e-6}"#,
            ),
        ];
        for (val, json) in cases {
            assert_eq!(serde_json::to_string(&val).unwrap(), json, "serialize {val:?}");
            assert_eq!(
                serde_json::from_str::<Integrator>(json).unwrap(),
                val,
                "deserialize {json}"
            );
        }
    }

    #[test]
    fn integrator_rk45_partial_tolerances_parse() {
        // OCaml omits a None tolerance; the present one must still parse.
        let i: Integrator = serde_json::from_str(r#"{"method":"rk45","atol":1e-9}"#).unwrap();
        assert_eq!(i, Integrator::Rk45 { atol: Some(1e-9), rtol: None });
    }
}
