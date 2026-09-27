//! gh#938 — every name an initial-state expression reads is checked when the
//! model is built, not when a gradient path first evaluates it.
//!
//! `CompiledModel::new` resolves every rate expression's names at build. The
//! initial-state seam (`initial_state_mean` / `_draw` / `_logpdf` /
//! `_logpdf_grad`, `initial_state_continuous`, `ic_grad_seed`) evaluates its
//! expressions with the name-keyed `propensity::eval_expr`, so a wrong name in
//! a law's emitted `∂arg/∂θ` map, in `ic_grad`, or in an `init {}` right-hand
//! side used to surface as `UnknownParameter` only on the first gradient call —
//! after the fit had started. A wrong *key* in a law's gradient map was worse:
//! `initial_state_logpdf_grad` skipped it, silently dropping a gradient
//! component (the gh#128 class).
//!
//! Each test builds a one-compartment pure-death model with one defect and
//! asserts `CompiledModel::new` refuses it with a message that names both the
//! offending name and the compartment whose initial condition carries it.

use ir::{
    deriv::DerivEntry,
    expr::{BinOp, Expr},
    model::{
        Compartment, CompartmentKind, InitCountLaw, InitSpec, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::PoissonLikelihood,
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::{compiled_model::CompiledModel, error::SimError};

fn fixed(name: &str, value: f64) -> Parameter {
    Parameter {
        name: name.into(),
        value: ir::parameter::ParamValue::Fixed { value },
        param_kind: None,
        param_dim: None,
    }
}

/// `N → ∅` at rate `mu·N`, with the given `init {}` entry for `N` and the
/// given `ic_grad`.
fn death_model(
    init: InitSpec,
    ic_grad: std::collections::HashMap<String, ir::deriv::ParamGradMap>,
) -> Model {
    Model {
        ic_grad,
        name: "gh938_death".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "N".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "death".into(),
            stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
            rate: Expr::bin_op(BinOp::Mul, Expr::param("mu"), Expr::pop("N")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![fixed("mu", 0.01), fixed("N0", 100.5)],
        initial_conditions: InitialConditions(std::iter::once(("N".to_string(), init)).collect()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![5.0, 10.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 10.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

/// `N ~ poisson(rate = N0)` whose emitted gradient map is `{key: d}`.
fn poisson_law(key: &str, d: Expr) -> InitSpec {
    let mut rate = Diffable::new(Expr::param("N0"));
    rate.grad.insert(key.into(), DerivEntry::Grad(d));
    InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood { rate }))
}

fn refusal(model: Model) -> String {
    match CompiledModel::new(model) {
        Ok(_) => panic!("CompiledModel::new must refuse the model; it built"),
        Err(e) => {
            assert!(e.is_structural(), "a build refusal must be structural, got {e:?}");
            e.to_string()
        }
    }
}

fn assert_names(msg: &str, needles: &[&str]) {
    for n in needles {
        assert!(msg.contains(n), "refusal must name `{n}`; got: {msg}");
    }
}

/// Control: the well-formed law builds, so each refusal below is attributable
/// to its one defect.
#[test]
fn well_formed_law_and_ic_grad_build() {
    let mut ic = std::collections::HashMap::new();
    ic.insert("N".to_string(), {
        let mut m = ir::deriv::ParamGradMap::new();
        m.insert("N0".into(), DerivEntry::Grad(Expr::const_(1.0)));
        m
    });
    CompiledModel::new(death_model(poisson_law("N0", Expr::const_(1.0)), ic))
        .expect("the well-formed fixture must build");
}

/// The issue's case: `∂rate/∂N0 = ghost`.
#[test]
fn unknown_name_in_init_law_derivative_is_refused_at_build() {
    let msg = refusal(death_model(poisson_law("N0", Expr::param("ghost")), Default::default()));
    assert_names(&msg, &["ghost", "'N'"]);
}

/// A gradient-map KEY that is not a parameter. At run time this was skipped,
/// silently dropping the component.
#[test]
fn unknown_key_in_init_law_gradient_map_is_refused_at_build() {
    let msg = refusal(death_model(poisson_law("ghost", Expr::const_(1.0)), Default::default()));
    assert_names(&msg, &["ghost", "'N'"]);
}

/// An `init {}` right-hand side naming an undeclared parameter.
#[test]
fn unknown_name_in_init_value_is_refused_at_build() {
    let init = InitSpec::Deterministic(Expr::param("ghost"));
    let msg = refusal(death_model(init, Default::default()));
    assert_names(&msg, &["ghost", "'N'"]);
}

/// An `ic_grad` entry whose derivative expression names an undeclared parameter.
#[test]
fn unknown_name_in_ic_grad_is_refused_at_build() {
    let mut ic = std::collections::HashMap::new();
    ic.insert("N".to_string(), {
        let mut m = ir::deriv::ParamGradMap::new();
        m.insert("N0".into(), DerivEntry::Grad(Expr::param("ghost")));
        m
    });
    let init = InitSpec::Deterministic(Expr::param("N0"));
    let msg = refusal(death_model(init, ic));
    assert_names(&msg, &["ghost", "'N'"]);
}

/// An `ic_grad` entry keyed by a compartment the model does not declare.
#[test]
fn unknown_compartment_in_ic_grad_is_refused_at_build() {
    let mut ic = std::collections::HashMap::new();
    ic.insert("Ghost".to_string(), ir::deriv::ParamGradMap::new());
    let init = InitSpec::Deterministic(Expr::param("N0"));
    let msg = refusal(death_model(init, ic));
    assert_names(&msg, &["Ghost"]);
}

/// The refusal is the resolver's: the same error class a rate expression's
/// unknown name gets, wrapped with the initial condition's location.
#[test]
fn refusal_is_a_validation_error() {
    let err = CompiledModel::new(death_model(
        poisson_law("N0", Expr::param("ghost")),
        Default::default(),
    ))
    .err()
    .expect("must refuse");
    assert!(matches!(err, SimError::Validation(_)), "got {err:?}");
}
