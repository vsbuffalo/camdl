//! gh#733: an `init {}` entry that names another compartment must read that
//! compartment's initial value, not zero.
//!
//! Before this change, `CompiledModel::initial_state` evaluated every
//! parameterized initial condition against a throwaway all-zero state, so
//!
//! ```camdl
//! init { A = A0          # A0 = 500
//!        B = A0 - A }    # reads A as 0
//! ```
//!
//! seeded `B = 500` instead of `B = 0`. It compiled with no diagnostic, the
//! reference survived into the IR, and the run started from a population that
//! was double what the model file says. There was no ordering to appeal to
//! either: the entries lived in a `HashMap`, which has no iteration order, so
//! even "evaluate in declaration order" was not available.
//!
//! The fix is a topological sort over the entries' compartment references and
//! evaluation against the partially built state. What this file pins:
//!
//! 1. **The value.** `B` is 0, and `A + B` is the declared total — the
//!    population-budget property that makes `S = N0 - I` mean what it says.
//! 2. **The order is derived, not declarative.** The same model written with
//!    the dependant FIRST gives the same answer. Declaration order alone would
//!    read a zero here, so this is the assertion that fails if the sort is
//!    dropped and entries are simply walked in order.
//! 3. **A later entry reads the ROUNDED integer count**, which is the state a
//!    discrete backend actually starts from — and the continuous ODE-gradient
//!    path reads the UNROUNDED value, because that is the value its
//!    forward-sensitivity seed differentiates.
//! 4. **A reference cycle is refused** at `CompiledModel::new`, naming the
//!    cycle, rather than resolved by picking an arbitrary order.

use ir::{
    expr::{BinOp, Expr},
    model::{
        Binding, Compartment, CompartmentKind, InitSpec, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    Model,
};
use sim::compiled_model::CompiledModel;

/// Two integer compartments `A`, `B`, one parameter `A0`, and whatever `init`
/// entries the caller supplies (in the order supplied). No transitions — the
/// initial state is the whole subject.
fn model(a0: f64, init: Vec<(&str, InitSpec)>, bindings: Vec<Binding>) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "gh733".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "A".into(), kind: CompartmentKind::Integer },
            Compartment { name: "B".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings,
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "A0".into(),
            value: ParamValue::Fixed { value: a0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions(
            init.into_iter().map(|(k, s)| (k.to_string(), s)).collect(),
        ),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(1),
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

fn det(e: Expr) -> InitSpec {
    InitSpec::Deterministic(e)
}

/// `A0 - A`.
fn a0_minus_a() -> Expr {
    Expr::bin_op(BinOp::Sub, Expr::param("A0"), Expr::pop("A"))
}

/// The integer initial counts, keyed by compartment name.
fn counts(m: Model) -> Vec<(String, i64)> {
    let compiled = CompiledModel::new(m).expect("model must build");
    let (int_s, _) = compiled
        .initial_state_mean(&compiled.default_params.clone())
        .expect("initial state");
    compiled
        .int_local_to_global
        .iter()
        .enumerate()
        .map(|(local, &global)| {
            (compiled.model.compartments[global].name.clone(), int_s.counts[local])
        })
        .collect()
}

#[test]
fn an_init_entry_reads_the_referenced_compartments_seeded_value() {
    // `A = A0` (500), then `B = A0 - A` — B is identically 0. Pre-fix this
    // evaluated `A` as 0 and seeded B = 500.
    let got = counts(model(500.0, vec![("A", det(Expr::param("A0"))), ("B", det(a0_minus_a()))], vec![]));
    assert_eq!(
        got,
        vec![("A".to_string(), 500), ("B".to_string(), 0)],
        "`B = A0 - A` must read the A this same call seeded, so B is 0"
    );

    // The property that makes the budget hold without a `balance {}` block:
    // whatever A is, A + B is the declared total.
    let total: i64 = got.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 500, "A + B must be the declared total A0");
}

#[test]
fn the_order_is_derived_from_the_references_not_from_the_declaration() {
    // Same model, dependant declared FIRST. Walking the entries in declaration
    // order would evaluate `B = A0 - A` against an unseeded A and give 500;
    // only the topological sort gives 0. This is the assertion that goes red if
    // the sort is removed and the map is simply iterated.
    let got = counts(model(500.0, vec![("B", det(a0_minus_a())), ("A", det(Expr::param("A0")))], vec![]));
    let b = got.iter().find(|(n, _)| n == "B").expect("B seeded").1;
    assert_eq!(b, 0, "declaration order must not decide the value; got B = {b}");
}

#[test]
fn a_binding_in_an_init_expression_is_followed_to_the_compartments_it_reads() {
    // `let total = A` used in an init RHS is a real dependency on A: the
    // binding body is evaluated against whatever state exists at that moment,
    // so treating the `BindingRef` as a leaf would order B first and read zero.
    let bindings = vec![Binding { name: "total".into(), expr: Expr::pop("A") }];
    let got = counts(model(
        500.0,
        vec![
            ("B", det(Expr::bin_op(BinOp::Sub, Expr::param("A0"), Expr::binding_ref("total")))),
            ("A", det(Expr::param("A0"))),
        ],
        bindings,
    ));
    let b = got.iter().find(|(n, _)| n == "B").expect("B seeded").1;
    assert_eq!(b, 0, "a binding that reads A makes B depend on A; got B = {b}");
}

#[test]
fn a_later_entry_reads_the_rounded_count_and_the_continuous_path_does_not() {
    // A0 = 10.6 on an INTEGER compartment: `A` is placed as 11, and `B = A`
    // reads 11 — the state a discrete backend starts from. The ODE gradient
    // path is continuous end to end and reads 10.6, because that is the value
    // its forward-sensitivity seed (`ic_grad`) differentiates.
    let m = model(10.6, vec![("A", det(Expr::param("A0"))), ("B", det(Expr::pop("A")))], vec![]);
    let compiled = CompiledModel::new(m).expect("model must build");
    let params = compiled.default_params.clone();

    let (int_s, _) = compiled.initial_state_mean(&params).expect("mean");
    assert_eq!(int_s.counts, vec![11, 11], "discrete path: B reads the rounded A");

    let (int_c, _) = compiled.initial_state_continuous(&params).expect("continuous");
    assert_eq!(int_c, vec![10.6, 10.6], "continuous path: B reads the unrounded A");
}

#[test]
fn a_reference_cycle_is_refused_and_the_error_names_the_cycle() {
    // `A = B + 1` beside `B = A - 1` has no evaluation order. Refuse rather
    // than pick one and report a number.
    let m = model(
        500.0,
        vec![
            ("A", det(Expr::bin_op(BinOp::Add, Expr::pop("B"), Expr::const_(1.0)))),
            ("B", det(Expr::bin_op(BinOp::Sub, Expr::pop("A"), Expr::const_(1.0)))),
        ],
        vec![],
    );
    let err = CompiledModel::new(m).err().expect("a cyclic init must not build");
    let msg = format!("{err}");
    assert!(
        msg.contains("cycle") && msg.contains("A -> B -> A"),
        "the error must name the whole cycle so the author can see which entry \
         to change; got: {msg}"
    );
}
