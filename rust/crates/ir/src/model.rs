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

/// What one compartment's initial value is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitSpec {
    /// `S = N0 - I`: an expression over constants, parameters and other
    /// compartments' initial values. Whether it happens to be constant is a
    /// runtime build detail, not a distinction the IR draws.
    Deterministic(Expr),
}

/// One spec per compartment, keyed by expanded compartment name.
///
/// **Ordered.** The JSON object's key order is the model's declaration order,
/// preserved on both sides (OCaml association list, Rust `IndexMap`) and folded
/// into the model's content hash. The runtime evaluates the entries in
/// *dependency* order — an entry may read another compartment's initial value —
/// which `CompiledModel::new` derives by topologically sorting this map;
/// declaration order is the tie-break between independent entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InitialConditions(pub indexmap::IndexMap<String, InitSpec>);

/// Order-**sensitive**, unlike the `IndexMap` equality it wraps.
///
/// `IndexMap`'s own `PartialEq` compares entry sets and ignores order. That is
/// the wrong contract here: the order is folded into the model's content hash
/// (see the `ContentAddressed` impl in `runid`), so an order-blind `==` would
/// let two models compare equal and key differently — and a serialization
/// round-trip that silently lost the order would pass an `assert_eq!(model,
/// reparsed)` while every stored run under it moved.
impl PartialEq for InitialConditions {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.0.iter().eq(other.0.iter())
    }
}

impl InitialConditions {
    /// Deterministic literals — `[("S", 999.0), ("I", 1.0)]`.
    ///
    /// **The iteration order of `entries` becomes the declaration order**, which
    /// is folded into the model's content hash and is the runtime's tie-break
    /// between independent entries. Pass an ordered source (an array, a `Vec`);
    /// a `HashMap` compiles but gives a different key on every run for any model
    /// whose identity is compared.
    pub fn constants(entries: impl IntoIterator<Item = (String, f64)>) -> Self {
        InitialConditions(
            entries
                .into_iter()
                .map(|(k, v)| (k, InitSpec::Deterministic(Expr::const_(v))))
                .collect(),
        )
    }

    /// Deterministic expressions — `[("S", N0 - I)]`. Same ordering contract as
    /// [`Self::constants`].
    pub fn exprs(entries: impl IntoIterator<Item = (String, Expr)>) -> Self {
        InitialConditions(
            entries
                .into_iter()
                .map(|(k, e)| (k, InitSpec::Deterministic(e)))
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn len(&self) -> usize { self.0.len() }

    /// Iterate `(compartment, spec)` in declaration order.
    pub fn iter(&self) -> indexmap::map::Iter<'_, String, InitSpec> { self.0.iter() }
}

impl<'a> IntoIterator for &'a InitialConditions {
    type Item = (&'a String, &'a InitSpec);
    type IntoIter = indexmap::map::Iter<'a, String, InitSpec>;
    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
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
    /// The simulation horizon. Encoded through `null_as_nan` because an
    /// unresolved anchored horizon travels as JSON `null` (gh#616) — JSON has
    /// no NaN literal, and a sentinel NUMBER is exactly what this design
    /// refuses (two sentinels compare equal).
    #[serde(with = "crate::anchor::null_as_nan")]
    pub t_end:          f64,
    pub time_semantics: String,
    pub dt:             Option<f64>,
    pub rng_seed:       Option<i64>,
    /// ODE integrator + (for rk45) its adaptive tolerances (gh#166). `default`/
    /// `skip_serializing_if` so pre-gh#166 IR (no field) deserializes to `Rk4`
    /// and a default model adds no JSON noise.
    #[serde(default, skip_serializing_if = "is_default_integrator")]
    pub integrator:     Integrator,
    /// gh#616: `simulate { to = last_obs + 4 'weeks }` — an observation-anchored
    /// horizon whose value is not known until a run binds its data.
    ///
    /// **This field IS the unresolved marker.** While it is `Some`, `t_end`
    /// carries `f64::NAN`, deliberately: every equality-based horizon guard then
    /// fails closed instead of passing on a coincidence (two placeholder
    /// horizons comparing equal is the gh#561 silent-drop class). The runtime
    /// resolver substitutes the resolved time into `t_end` and CLEARS this, and
    /// `CompiledModel::new` refuses any model that still carries it — one guard
    /// at the choke point every path goes through, rather than one per entry
    /// point. `None` for a literal horizon; omitted from the JSON then, so an
    /// unanchored model's IR is byte-identical to its pre-gh#616 form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_end_anchor:   Option<crate::anchor::AnchoredTime>,
}

impl SimulationConfig {
    /// gh#616: restore the "unresolved means not a usable number" invariant
    /// after decoding. JSON has no NaN literal, so an anchored horizon travels
    /// as `null` (see `serde.ml`); this turns it back into NaN, from the
    /// ANCHOR's presence rather than from whatever the file happened to say.
    /// Enforcing it here means a hand-edited or third-party IR cannot present a
    /// usable horizon alongside an unresolved anchor.
    fn restore_unresolved_horizon(&mut self) {
        if self.t_end_anchor.is_some() {
            self.t_end = f64::NAN;
        }
    }
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
    /// gh#616: a preset's own anchored horizon. Same contract as
    /// [`SimulationConfig::t_end_anchor`] — while this is `Some`, `t_end` is
    /// `Some(NAN)`, and the resolver clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_end_anchor: Option<crate::anchor::AnchoredTime>,
}

impl Preset {
    /// See [`SimulationConfig::restore_unresolved_horizon`]. A preset's `t_end`
    /// is optional, so `null` is ambiguous on its own — an anchored horizon and
    /// "declares no horizon" both write it. The anchor field disambiguates.
    fn restore_unresolved_horizon(&mut self) {
        if self.t_end_anchor.is_some() {
            self.t_end = Some(f64::NAN);
        }
    }
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
    /// ∂(initial_state)/∂θ per initial-condition compartment — the
    /// forward-sensitivity seed `S(t_start)` for the ODE gradient (gh#275), keyed
    /// `compartment → (param → DerivEntry)`. A `WrtParam` differentiation of the
    /// [`InitSpec::Deterministic`] expressions, each first closed over the other
    /// entries (gh#733: `S = N0 - I` differentiates `I`'s own initial expression
    /// too, because the runtime evaluates the block in dependency order);
    /// parameter-keyed (hence [`crate::deriv::ParamGradMap`]), the `rate_grad`
    /// analogue for the IC map. Compartments whose initial value does not depend
    /// on any parameter are omitted, so a block of literals emits nothing.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub ic_grad:            std::collections::HashMap<String, crate::deriv::ParamGradMap>,
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

impl Model {
    /// gh#616: the post-decode normalisation every IR load applies (see
    /// [`SimulationConfig::restore_unresolved_horizon`]). Idempotent, and a
    /// no-op for a model that declares no anchor.
    pub fn restore_unresolved_horizons(&mut self) {
        self.simulation.restore_unresolved_horizon();
        for p in &mut self.presets {
            p.restore_unresolved_horizon();
        }
    }
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
