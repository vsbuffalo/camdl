//! Real-valued compartments are integrated by their own equation on every
//! backend, whatever order the `ode {}` block lists the equations in.
//!
//! The IR holds real compartments in declaration order and ODE equations in
//! `ode {}` block order. The RK4 step the chain-binomial and Gillespie backends
//! use for real compartments indexed derivatives by equation position and
//! applied them by compartment position, so with compartments `W, V` and
//! equations `V, W` each compartment was advanced by the other's derivative.
//! The ODE backend maps through `ode_real_indices` and was always right.
//!
//! The model here has dV/dt = 0 and dW/dt = -delta * W, with no dependence of
//! either on the integer state, so the answer is closed-form on all three
//! backends: V stays at 50 and W = 100 * exp(-delta * t).

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const T_END: f64 = 5.0;
const DELTA: f64 = 1.0;

fn two_real_model() -> CompiledModel {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_reservoir.ir.json").unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let m = &mut v["model"];
    // Declaration order of real compartments: W, then V.
    m["compartments"].as_array_mut().unwrap()
        .push(serde_json::json!({ "name": "V", "kind": "real" }));
    m["initial_conditions"]["W"] = serde_json::json!({ "deterministic": { "const": 100.0 } });
    m["initial_conditions"]["V"] = serde_json::json!({ "deterministic": { "const": 50.0 } });
    // Equation order: V first, then W — the reverse of the declaration order.
    m["ode_equations"].as_array_mut().unwrap()
        .insert(0, serde_json::json!({ "compartment": "V", "derivative": { "const": 0.0 } }));
    let mut model: ir::Model = ir::from_str(&v.to_string()).unwrap();
    assert_eq!(
        model.ode_equations.iter().map(|e| e.compartment.as_str()).collect::<Vec<_>>(),
        ["V", "W"],
    );
    // xi = 0 removes shedding, so W's equation is pure decay.
    let values = [("beta", 0.3), ("gamma", 0.1), ("xi", 0.0), ("delta", DELTA),
                  ("N0", 1000.0), ("I0", 10.0)];
    for p in &mut model.parameters {
        let x = values.iter().find(|(n, _)| *n == p.name).unwrap().1;
        p.value = p.value.with_value(x);
    }
    CompiledModel::new(model).unwrap()
}

fn final_real(compiled: &CompiledModel, traj: &sim::state::Trajectory, name: &str) -> f64 {
    let global = compiled.model.compartments.iter().position(|c| c.name == name).unwrap();
    let local = compiled.real_local_to_global.iter().position(|&g| g == global).unwrap();
    traj.snapshots.last().unwrap().real_state.values[local]
}

fn check(backend: &str, compiled: &CompiledModel, traj: &sim::state::Trajectory) {
    let w = final_real(compiled, traj, "W");
    let v = final_real(compiled, traj, "V");
    let w_exact = 100.0 * (-DELTA * T_END).exp();
    assert!((v - 50.0).abs() < 1e-9, "{backend}: V has dV/dt = 0 and must stay 50, got {v}");
    assert!((w - w_exact).abs() < 1e-3 * w_exact,
        "{backend}: W must decay to {w_exact}, got {w}");
}

#[test]
fn each_real_compartment_follows_its_own_equation_on_every_backend() {
    let compiled = two_real_model();
    let p = &compiled.default_params;

    let ode = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: 0.01 });
    check("ode", &compiled, &OdeSim.run(&compiled, p, 1, &ode).unwrap());

    let cb = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 0.01 });
    check("chain_binomial", &compiled, &ChainBinomialSim.run(&compiled, p, 1, &cb).unwrap());

    let gil = SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: T_END, output_dt: Some(1.0) });
    check("gillespie", &compiled, &GillespieSim.run(&compiled, p, 1, &gil).unwrap());
}
