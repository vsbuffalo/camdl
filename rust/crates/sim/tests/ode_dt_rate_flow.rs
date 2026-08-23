//! gh#126 §#11 (upstream-rust-engine-review 2026-05-26): the ODE
//! flow-accumulation eval must evaluate a `dt`-referencing rate at the REALIZED
//! substep length (`dt_actual`), not the nominal grid `cfg.dt` — the same
//! `EvalCtx.dt = dt_actual` rule the StepClock work (scheduling-spine-v2 §A)
//! decided and `gate_dt_rate_exact_clip` enforces for the step_one / PGAS path.
//!
//! `ode.rs` fed `cfg.dt` into `eval_propensities` for `flow_acc` while
//! multiplying the result by the actual substep `dt`, so a transition RATE that
//! references `Expr::Dt` (gh#54) produced wrong REPORTED flows on truncated
//! boundary substeps (outputs / interventions / end) — shifting incidence, and
//! thus the likelihood, at exactly the windows inference reads. The RK4 state
//! update already used the actual `dt`, so this was a flow-vs-state
//! inconsistency, not a trajectory error.
//!
//! This is the ODE-flow sibling of `gate_dt_rate_exact_clip` (which covers only
//! the chain_binomial / PGAS path). Fixture:
//! `tests/fixtures/corner_cases/dt_rate.camdl` — the infection hazard carries an
//! explicit `(dt / tau)` factor, so the realized substep length enters the rate
//! expression linearly.

use sim::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    propensity::eval_propensities,
    simulate::Simulate,
    OdeSim,
};

const SEED: u64 = 11;

fn load_dt_rate() -> (CompiledModel, Vec<f64>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/fixtures/corner_cases/ir/dt_rate.ir.json"
    );
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let model = ir::from_str(&json).expect("parse dt_rate IR");
    let compiled = CompiledModel::new(model).expect("compile dt_rate");
    let params = compiled.default_params.clone();
    (compiled, params)
}

#[test]
fn ode_flow_uses_realized_substep_dt_not_grid_dt() {
    let (compiled, mut params) = load_dt_rate();

    // transitions: [infection, recovery]; infection's rate reads `dt`.
    let infection = 0usize;
    assert_eq!(compiled.model.transitions[infection].name, "infection");

    // Pin beta/tau so the realized-vs-grid flows are clearly distinct after the
    // u64 rounding the snapshot does (avoids a vacuous test if defaults are tiny).
    let pidx = |name: &str| {
        compiled.model.parameters.iter().position(|p| p.name == name)
            .unwrap_or_else(|| panic!("param {name} not found"))
    };
    params[pidx("beta")] = 2.0;
    params[pidx("tau")] = 1.0;

    // dt_rate outputs are regular {start:0, step:1.0}. A grid dt LARGER than the
    // output step truncates the first integrated substep to land on the t=1
    // output boundary: dt_actual = 1.0 while cfg.dt = 3.0. The first window
    // [0,1] is therefore a SINGLE substep whose flow is computed from the
    // exactly-known initial state.
    let cfg_dt = 3.0;
    let dt_actual = 1.0;
    let t_start = compiled.model.simulation.t_start;
    assert_eq!(t_start, 0.0);

    let (int0, real0) = compiled.initial_state_mean(&params).expect("initial state");

    // Oracle built from `eval_propensities` itself (no hardcoded beta/tau).
    // The ODE backend records the continuous flow `rate · dt` WITHOUT rounding,
    // so the oracle is the raw real product.
    let flow = |dt_arg: f64| -> f64 {
        let mut p = Vec::new();
        eval_propensities(&compiled, &int0, &real0, &params, t_start, dt_arg, None, &mut p)
            .expect("eval propensities");
        p[infection] * dt_actual
    };
    let flow_realized = flow(dt_actual); // correct: Expr::Dt sees the realized 1.0
    let flow_grid = flow(cfg_dt); //        bug: Expr::Dt sees the nominal 3.0
    assert!(
        (flow_realized - flow_grid).abs() > 1e-9,
        "vacuous test: realized-dt and grid-dt flows must differ \
         (realized={flow_realized}, grid={flow_grid})"
    );

    // Run the ODE backend; read the first window's reported infection flow.
    let cfg = SimConfig::Ode(OdeConfig { t_start, t_end: 2.0, dt: cfg_dt });
    let traj = OdeSim.run(&compiled, &params, SEED, &cfg).expect("ode run");
    let snap = traj.snapshots.iter()
        .find(|s| (s.t - 1.0).abs() < 1e-9)
        .expect("a snapshot at the t=1 output boundary");
    let got = snap.flows.as_real()[infection];

    assert!(
        (got - flow_realized).abs() < 1e-9,
        "ODE flow accumulation must evaluate the dt-referencing rate at the \
         realized substep dt ({dt_actual}), not the nominal grid dt ({cfg_dt}): \
         got {got}, realized-dt oracle {flow_realized}, grid-dt (buggy) {flow_grid}"
    );
}

/// Phase 0 (proposal 2026-06-15-ode-gradient-inference): the ODE backend records
/// continuous flow `rate · dt` WITHOUT rounding. A sub-unit per-window flow — a
/// slow transition such as TB reactivation — must survive into the trajectory.
/// The previous `x.round() as u64` snapshot quantization zeroed it, which
/// collapsed `compute_ode_loglik` to `-∞` across the entire slow-rate regime
/// (the exact regime of a TB latency fit).
#[test]
fn ode_subunit_flow_is_not_rounded_to_zero() {
    let (compiled, mut params) = load_dt_rate();
    let infection = 0usize;
    assert_eq!(compiled.model.transitions[infection].name, "infection");
    let pidx = |name: &str| {
        compiled.model.parameters.iter().position(|p| p.name == name)
            .unwrap_or_else(|| panic!("param {name} not found"))
    };
    // Tiny hazard → a sub-unit infection flow over one output window.
    params[pidx("beta")] = 1e-3;
    params[pidx("tau")] = 1.0;

    let t_start = compiled.model.simulation.t_start;
    let (int0, real0) = compiled.initial_state_mean(&params).expect("initial state");

    // Oracle: the continuous flow over the first unit window — a single substep
    // from the exactly-known initial state with dt = 1.
    let dt = 1.0;
    let mut p = Vec::new();
    eval_propensities(&compiled, &int0, &real0, &params, t_start, dt, None, &mut p)
        .expect("eval propensities");
    let oracle = p[infection] * dt;
    assert!(
        oracle > 0.0 && oracle < 0.5,
        "test setup: expected a sub-unit infection flow, got {oracle}"
    );
    assert_eq!(
        oracle.round(), 0.0,
        "test setup: the pre-fix `round()` must map this flow to 0 (oracle={oracle})"
    );

    let cfg = SimConfig::Ode(OdeConfig { t_start, t_end: 2.0, dt: 1.0 });
    let traj = OdeSim.run(&compiled, &params, SEED, &cfg).expect("ode run");
    let snap = traj.snapshots.iter()
        .find(|s| (s.t - 1.0).abs() < 1e-9)
        .expect("a snapshot at the t=1 output boundary");
    let got = snap.flows.as_real()[infection];

    assert!(
        (got - oracle).abs() < 1e-12 && got > 0.0,
        "sub-unit ODE flow must be preserved unrounded: got {got}, oracle {oracle} \
         (the pre-fix `round()` would report 0.0 → likelihood -∞)"
    );
}
