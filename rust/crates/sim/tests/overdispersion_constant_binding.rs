//! The overdispersion σ² of `overdispersed(rate, σ²)` must be refused only
//! when it really reads compartment state.
//!
//! `CompiledModel::new` refuses a σ² that references state, because the PGAS
//! gamma-density sites evaluate σ² against a scratch state. The compiler hoists
//! a parameter-free constant `let` (`let s2 = 0.08 'days`) into a model-level
//! binding, so such a σ² arrives as a `BindingRef`; the check treated every
//! `BindingRef` as state-derived and refused a constant with the message that
//! it "references compartment state". The check now follows a binding into its
//! body: a constant binding is accepted, and simulates exactly as the same
//! constant written inline, while a binding that reads state (directly or
//! through another binding) is still refused.

use std::collections::HashMap;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, ParamExpr, PopExpr},
    model::{Binding, Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

/// S --> I at rate `beta * S`, drawn with overdispersion `sigma_expr`, with the
/// given model-level bindings.
fn overdispersed_model(sigma_expr: Expr, bindings: Vec<Binding>) -> Model {
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "od_binding".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                },
            }),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Overdispersed {
                sigma_sq: sigma_expr,
                sigma_sq_grad: Default::default(),
            },
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings,
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter {
                name: "beta".into(),
                value: ir::parameter::ParamValue::Fixed { value: 0.01 },
                param_kind: None,
                param_dim: None,
            },
            // Finite, so the gh#81 non-finite-parameter guard passes it — and
            // zero, so any expression dividing by it is not.
            Parameter {
                name: "phi".into(),
                value: ir::parameter::ParamValue::Fixed { value: 0.0 },
                param_kind: None,
                param_dim: None,
            },
        ],
        initial_conditions: InitialConditions::constants({
            let mut m = HashMap::new();
            m.insert("S".into(), 10000.0);
            m.insert("I".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None, t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

fn constant(v: f64) -> Expr { Expr::Const(ConstExpr { value: v }) }
fn binding_ref(name: &str) -> Expr { Expr::binding_ref(name) }
fn binding(name: &str, expr: Expr) -> Binding { Binding { name: name.into(), expr } }

fn flows(model: Model, seed: u64) -> Vec<u64> {
    let compiled = CompiledModel::new(model).expect("the model must compile");
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: 0.0, t_end: 1.0, dt: 1.0, binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    let traj = ChainBinomialSim.run(&compiled, &params, seed, &config).expect("simulate");
    traj.snapshots.iter().map(|s| s.flows.as_int()[0]).collect()
}

fn refusal(model: Model) -> String {
    match CompiledModel::new(model) {
        Ok(_) => panic!("a σ² that reads compartment state must be refused"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_constant_binding_is_accepted_and_draws_as_the_inline_constant() {
    let via_binding = overdispersed_model(binding_ref("s2"), vec![binding("s2", constant(0.5))]);
    let inline = overdispersed_model(constant(0.5), vec![]);
    for seed in [1, 42, 7] {
        let a = flows(via_binding.clone(), seed);
        let b = flows(inline.clone(), seed);
        assert_eq!(a, b, "seed {seed}: a σ² read through a constant binding must draw exactly as the constant inline");
    }
    assert!(flows(inline, 42).iter().any(|&f| f > 0), "control produced no infections; the fixture is wrong");
}

#[test]
fn a_binding_that_reads_state_is_still_refused() {
    let direct = overdispersed_model(
        binding_ref("n"),
        vec![binding("n", Expr::Pop(PopExpr { pop: "S".into() }))],
    );
    let msg = refusal(direct);
    assert!(msg.contains("references compartment state"), "unexpected refusal: {msg}");

    // Through a chain of bindings: the constant-looking outer one reads an
    // earlier one that reads `I`.
    let chained = overdispersed_model(
        binding_ref("outer"),
        vec![
            binding("inner", Expr::Pop(PopExpr { pop: "I".into() })),
            binding("outer", Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                op: BinOp::Mul,
                left: Box::new(constant(2.0)),
                right: Box::new(binding_ref("inner")),
            }})),
        ],
    );
    let msg = refusal(chained);
    assert!(msg.contains("references compartment state"), "unexpected refusal: {msg}");
}
