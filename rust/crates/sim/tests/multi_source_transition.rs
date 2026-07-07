//! gh#121 — a transition that draws from TWO OR MORE source compartments
//! (`A + B --> C`, two negative-stoichiometry entries) is bounded by only the
//! FIRST source on chain_binomial, so the secondary source can be driven
//! negative — silently in a mild regime, or as a cryptic runtime
//! `NegativeCount{cause: BinomialOvershoot}` otherwise. gillespie applies each
//! firing as one atomic CTMC event decrementing every source together; ODE runs
//! every transition as a continuous flow. So the multi-source model must be
//! REJECTED up front on the stochastic chain-binomial paths (forward
//! chain_binomial + every stochastic inference producer via the shared
//! validation the dispatch gate calls), while still RUNNING on gillespie/ode.
//!
//! These tests build `ir::Model` structs directly (mirroring
//! `sourced_deterministic_transition.rs`) so the model is exact, not desugared.

use std::collections::HashMap;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        RegularOutputSchedule, SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

// ── tiny expression builders (mirror sourced_deterministic_transition.rs) ────
fn param(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn pop(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn mul(l: Expr, r: Expr) -> Expr {
    Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(l), right: Box::new(r) },
    })
}
fn fixed(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}

/// A two-source Poisson transition `src_a + src_b --> dst @ rate`.
fn multi_source(name: &str, src_a: &str, src_b: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![
            StoichiometryEntry(src_a.into(), -1),
            StoichiometryEntry(src_b.into(), -1),
            StoichiometryEntry(dst.into(), 1),
        ],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    }
}
fn poisson(name: &str, src: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![StoichiometryEntry(src.into(), -1), StoichiometryEntry(dst.into(), 1)],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    }
}

fn build(
    name: &str,
    comps: &[&str],
    transitions: Vec<Transition>,
    parameters: Vec<Parameter>,
    init: &[(&str, f64)],
    t_end: f64,
) -> Model {
    Model {
        ic_grad: Default::default(),
        name: name.into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: comps
            .iter()
            .map(|c| Compartment { name: (*c).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions,
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters,
        initial_conditions: InitialConditions::Explicit(
            init.iter().map(|(k, v)| ((*k).into(), *v)).collect::<HashMap<String, f64>>(),
        ),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(7),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

const DT: f64 = 1.0;

/// A vector-borne-style bimolecular infection `Sh + Im --> Ih @ bite*Im` with a
/// large host pool (Sh = 1e6) and a small infectious-vector pool (Im = 50). The
/// drawn flow is `~bite*Im*dt`, which exceeds Im, so on chain_binomial the
/// secondary source Im is driven negative. The model must be REJECTED up front
/// (a located gh#121 error) on the stochastic paths, while gillespie/ode run it.
#[test]
fn multi_source_rejected_on_chain_binomial_but_runs_on_gillespie_and_ode() {
    const T_END: f64 = 5.0;
    let model = build(
        "vector_infect",
        &["Sh", "Im", "Ih"],
        vec![multi_source("infect", "Sh", "Im", "Ih", mul(param("bite"), pop("Im")))],
        vec![fixed("bite", 2.0)],
        &[("Sh", 1e6), ("Im", 50.0), ("Ih", 0.0)],
        T_END,
    );
    let compiled = CompiledModel::new(model).expect("a multi-source model still COMPILES (rejected at dispatch)");

    // The shared structural validation (the inference dispatch gate delegates to
    // this) errors with a located, gh#121-tagged message naming the transition
    // and its two source compartments.
    let err = compiled
        .validate_single_source_transitions()
        .expect_err("multi-source transition must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("gh#121"), "message must cite the issue: {msg}");
    assert!(msg.contains("infect"), "message must name the transition: {msg}");
    assert!(msg.contains("Sh"), "message must name the first source Sh: {msg}");
    assert!(msg.contains("Im"), "message must name the second source Im: {msg}");

    // Forward chain_binomial hard-errors up front (does NOT over-draw the source
    // into a cryptic NegativeCount, and does NOT silently proceed).
    let cb = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: DT });
    let cb_res = ChainBinomialSim.run(&compiled, &compiled.default_params, 1, &cb);
    assert!(cb_res.is_err(), "forward chain_binomial must reject a multi-source transition");
    assert!(
        cb_res.unwrap_err().to_string().contains("gh#121"),
        "chain_binomial rejection must be the located gh#121 error, not a runtime NegativeCount"
    );

    // gillespie runs it: each firing is one atomic CTMC event decrementing both
    // sources together.
    let gil = SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: T_END, output_dt: None });
    let gil_res = GillespieSim.run(&compiled, &compiled.default_params, 1, &gil);
    assert!(gil_res.is_ok(), "gillespie must run a multi-source transition: {:?}", gil_res.err());

    // ODE runs it: every transition is a continuous flow.
    let ode = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: DT });
    let ode_res = OdeSim.run(&compiled, &compiled.default_params, 1, &ode);
    assert!(ode_res.is_ok(), "ode must run a multi-source transition: {:?}", ode_res.err());
}

/// A plain single-source model is NOT rejected (the check is scoped to ≥2
/// negative-stoichiometry sources) and runs unchanged on chain_binomial.
#[test]
fn single_source_model_is_accepted_on_chain_binomial() {
    let model = build(
        "sir_poisson",
        &["S", "I", "R"],
        vec![
            poisson("infect", "S", "I", mul(param("beta"), pop("I"))),
            poisson("recover", "I", "R", mul(param("gamma"), pop("I"))),
        ],
        vec![fixed("beta", 0.0003), fixed("gamma", 0.1)],
        &[("S", 990.0), ("I", 10.0), ("R", 0.0)],
        20.0,
    );
    let compiled = CompiledModel::new(model).unwrap();
    assert!(
        compiled.validate_single_source_transitions().is_ok(),
        "a single-source model must not be rejected"
    );
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 20.0, dt: DT });
    assert!(
        ChainBinomialSim.run(&compiled, &compiled.default_params, 1, &cfg).is_ok(),
        "a single-source model must still run on chain_binomial"
    );
}

/// gh#121 review: a SINGLE source written as several un-collapsed negative
/// entries (`[["S",-1],["S",-1],…]`) must be ACCEPTED. camdlc always collapses
/// stoichiometry per compartment, but the IR is a public contract, so a
/// hand-authored IR can carry this shape. The validator counts DISTINCT source
/// compartments, not stoich entries — otherwise the same reaction gets opposite
/// verdicts by representation (the collapsed `[["S",-2],…]` is accepted).
#[test]
fn single_source_as_duplicate_entries_is_accepted() {
    let dup_entry = Transition {
        rate_state_grad: Default::default(),
        name: "react".into(),
        stoichiometry: vec![
            StoichiometryEntry("S".into(), -1),
            StoichiometryEntry("S".into(), -1),
            StoichiometryEntry("I".into(), 1),
        ],
        rate: mul(param("k"), pop("S")),
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    };
    let model = build(
        "dup_entry_single_source",
        &["S", "I"],
        vec![dup_entry],
        vec![fixed("k", 0.0001)],
        &[("S", 1000.0), ("I", 0.0)],
        20.0,
    );
    let compiled = CompiledModel::new(model).unwrap();
    assert!(
        compiled.validate_single_source_transitions().is_ok(),
        "a single source written as duplicate negative entries must NOT be \
         rejected as multi-source (it dedups to one distinct compartment)"
    );
}
