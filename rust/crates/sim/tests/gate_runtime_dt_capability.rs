//! Capability gate for `Expr::Dt`-in-a-rate (gh#54): a model whose transition
//! rate references the runtime substep `dt` requires a backend that realizes a
//! substep length (`EvalCtx.dt = dt_actual`). ODE (RK4 flow accumulation, see
//! `ode_dt_rate_flow.rs`) and chain_binomial (StepClock, see
//! `gate_dt_rate_exact_clip.rs`) provide it; Gillespie does NOT — its SSA loop
//! has no substep, so it freezes the `Expr::Dt` node to the nominal
//! `simulation.dt`-or-`1.0` (gillespie.rs:269/366). Before this gate that
//! produced a DIFFERENT trajectory on each backend with NO warning — exactly the
//! BALANCE failure mode (`Capabilities::BALANCE`, gh#audit-C3).
//!
//! This mirrors the BALANCE precedent: the requirement is auto-derived by
//! `CompiledModel::required_capabilities()` (walking the rate ASTs for
//! `Expr::Dt`), the backend `capabilities()` sets declare support, and the
//! dispatch gate (`required - backend.capabilities()`) rejects the mismatch.
//!
//! Fixture: `tests/fixtures/corner_cases/ir/dt_rate.ir.json` — its `infection`
//! rate carries an explicit `(dt / tau)` factor.

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    Capabilities, ChainBinomialSim, GillespieSim, OdeSim,
};

fn load_dt_rate() -> CompiledModel {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/fixtures/corner_cases/ir/dt_rate.ir.json"
    );
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let model = ir::from_str(&json).expect("parse dt_rate IR");
    CompiledModel::new(model).expect("compile dt_rate")
}

/// The model's required capabilities must include RUNTIME_DT, because its
/// `infection` rate references `Expr::Dt` via the `(dt / tau)` factor.
#[test]
fn dt_in_rate_requires_runtime_dt_capability() {
    let compiled = load_dt_rate();
    let required = compiled.required_capabilities();
    assert!(
        required.contains(Capabilities::RUNTIME_DT),
        "a model whose rate references `dt` (Expr::Dt) must require RUNTIME_DT; \
         got {required:?}"
    );
}

/// Gillespie does not realize a substep dt, so dispatching the dt_rate model on
/// it must be rejected by the capability gate (`required - capabilities()` is
/// non-empty) rather than silently running with a frozen nominal dt.
#[test]
fn gillespie_rejects_dt_in_rate() {
    let compiled = load_dt_rate();
    let required = compiled.required_capabilities();
    let missing = required - GillespieSim.capabilities();
    assert!(
        missing.contains(Capabilities::RUNTIME_DT),
        "gillespie has no substep — a dt-in-rate model must be a capability \
         mismatch on gillespie; missing = {missing:?}"
    );
}

/// item 19 / gh#54 hole: a rate whose only `dt` reference is transitive through a
/// hoisted model-level binding (`let dtf = dt` — param-free, so the OCaml Fix-B
/// hoister moves it into `model.bindings` and the rate becomes `binding_ref:
/// dtf`) must STILL require RUNTIME_DT. `expr_contains_dt` treated `BindingRef`
/// as a leaf that "cannot contain Dt", so gillespie ran such a model silently
/// with a frozen nominal dt — while `collect_int_comp_deps` /
/// `expr_is_time_dependent` (the sibling Gillespie-classification walkers) both
/// recurse through `BindingRef`. The capability derivation must too.
#[test]
fn dt_hidden_in_binding_requires_runtime_dt() {
    use std::collections::HashMap;
    use ir::{
        expr::{BinOp, BinOpExpr, BinOpWrap, BindingRefWrap, DtExpr, Expr, ParamExpr, PopExpr},
        model::{
            Binding, Compartment, CompartmentKind, InitialConditions, OutputConfig,
            OutputSchedule, SimulationConfig,
        },
        parameter::{ParamValue, Parameter},
        transition::{StoichiometryEntry, Transition},
        Model,
    };

    // A rate `beta * S * dtf` where `dtf` is a model-level binding whose body is
    // `Expr::Dt` — the exact shape the Fix-B hoister produces for `let dtf = dt`.
    let m = Model {
        ic_grad: Default::default(),
        name: "hoisted_dt".into(),
        version: "0.1".into(),
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
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                op: BinOp::Mul,
                left: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                }})),
                // the only `dt` in the whole model lives behind this binding_ref
                right: Box::new(Expr::BindingRef(BindingRefWrap { binding_ref: "dtf".into() })),
            }}),
            metadata: None,
            draw_method: Default::default(),
            rate_grad: HashMap::new(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![Binding { name: "dtf".into(), expr: Expr::Dt(DtExpr { dt: () }) }],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "beta".into(),
            value: ParamValue::Fixed { value: 0.5 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), 990.0);
            h.insert("I".into(), 10.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 5.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 5.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
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
    };

    let compiled = CompiledModel::new(m).expect("hoisted-dt model compiles");
    let required = compiled.required_capabilities();
    assert!(
        required.contains(Capabilities::RUNTIME_DT),
        "a rate whose `dt` is hidden behind a hoisted binding must still require \
         RUNTIME_DT (else gillespie runs it silently with a frozen nominal dt); \
         got {required:?}"
    );
    // And therefore gillespie (no RUNTIME_DT) must be a capability mismatch.
    assert!(
        (required - GillespieSim.capabilities()).contains(Capabilities::RUNTIME_DT),
        "gillespie must reject a dt-behind-binding model at the capability gate"
    );
}

/// ODE and chain_binomial realize a substep dt, so they DO declare RUNTIME_DT
/// and the dt_rate model dispatches and runs on them.
#[test]
fn ode_and_chain_binomial_accept_and_run_dt_in_rate() {
    let compiled = load_dt_rate();
    let required = compiled.required_capabilities();
    let params = compiled.default_params.clone();
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;

    for (name, sim, config) in [
        (
            "ode",
            &OdeSim as &dyn Simulate,
            SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 }),
        ),
        (
            "chain_binomial",
            &ChainBinomialSim as &dyn Simulate,
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 }),
        ),
    ] {
        assert!(
            sim.capabilities().contains(Capabilities::RUNTIME_DT),
            "{name} must declare RUNTIME_DT (it realizes a substep dt)"
        );
        assert!(
            (required - sim.capabilities()).is_empty(),
            "{name} must satisfy the dt_rate model's capabilities"
        );
        sim.run(&compiled, &params, 42, &config)
            .unwrap_or_else(|e| panic!("{name} should run the dt_rate model: {e}"));
    }
}

/// Guard against false positives: a model WITHOUT `Expr::Dt` in any rate must
/// NOT require RUNTIME_DT (so gillespie still runs ordinary SIR). Built inline
/// by dropping the dt factor would require recompiling; instead assert via a
/// gillespie run of a known dt-free corner-case fixture.
#[test]
fn dt_free_model_does_not_require_runtime_dt() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/fixtures/corner_cases/ir/off_grid_obs.ir.json"
    );
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let model = ir::from_str(&json).expect("parse off_grid_obs IR");
    let compiled = CompiledModel::new(model).expect("compile off_grid_obs");
    let required = compiled.required_capabilities();
    assert!(
        !required.contains(Capabilities::RUNTIME_DT),
        "a model with no Expr::Dt in any rate must not require RUNTIME_DT; \
         got {required:?}"
    );
    // And it must still run on gillespie (no spurious gate).
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    assert!(
        (required - GillespieSim.capabilities()).is_empty(),
        "dt-free model must satisfy gillespie capabilities"
    );
    GillespieSim
        .run(
            &compiled,
            &compiled.default_params,
            42,
            &SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        )
        .expect("gillespie should run the dt-free model");
}
