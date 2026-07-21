//! Forward-sim progress-tick invariance (commit 2 of the progress-feedback
//! work).
//!
//! The CLI shows a per-timestep progress bar for single `camdl simulate`
//! runs by passing an `Option<&mut dyn FnMut(f64)>` "tick" closure into the
//! backend timestep loop. The closure is called once per step (per event for
//! Gillespie) with the current time `t`; it does **read-only** work (advances
//! an indicatif bar) and consumes NO randomness.
//!
//! This is the load-bearing safety property for that feature: attaching the
//! tick MUST NOT change the trajectory. A run with `tick = Some(..)` must be
//! BYTE-IDENTICAL to the same run (same model / params / seed / config) with
//! `tick = None`. If it isn't, the hook is perturbing the simulation — a
//! silent-wrong-answer class bug (camdl informs public-health decisions), so
//! this test asserts exact `Trajectory` equality (snapshot-by-snapshot,
//! bit-exact on f64) for all four backends, plus a non-vacuity check that the
//! tick actually fired.

use std::path::Path;

use sim::{
    chain_binomial::run_chain_binomial_with_observer,
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig},
    gillespie::run_gillespie_with_observer,
    ode::run_ode,
    state::Trajectory,
};

/// Path to a committed golden IR fixture (same source the invariant tests use,
/// e.g. `chain_binomial_invariants.rs`).
fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

/// Load `sir_basic` and pin its parameters (the golden ships `N0`/`I0`/`beta`/
/// `gamma` without defaults; we set concrete values directly).
fn load_sir() -> (CompiledModel, Vec<f64>) {
    let contents = std::fs::read_to_string(golden_path("sir_basic"))
        .expect("could not read golden sir_basic");
    let mut model: ir::Model =
        ir::from_str(&contents).unwrap_or_else(|e| panic!("parse sir_basic: {e}"));
    for p in &mut model.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.4,
            "gamma" => 0.2,
            "N0" => 10_000.0,
            "I0" => 10.0,
            _ => continue,
        };
        p.value = p.value.with_value(v);
    }
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

const SEED: u64 = 12345;
const T_END: f64 = 30.0;

/// Assert two trajectories are byte-identical snapshot-by-snapshot. `Snapshot`
/// is not `PartialEq`, so compare its observable contents (`t`, `IntState`,
/// `RealState`, `FlowVec`) directly, bit-exact on every f64.
fn assert_traj_eq(a: &Trajectory, b: &Trajectory, ctx: &str) {
    assert_eq!(
        a.snapshots.len(),
        b.snapshots.len(),
        "{ctx}: snapshot count differs ({} vs {}) — the tick changed the run",
        a.snapshots.len(),
        b.snapshots.len()
    );
    for (i, (sa, sb)) in a.snapshots.iter().zip(b.snapshots.iter()).enumerate() {
        assert_eq!(sa.t.to_bits(), sb.t.to_bits(), "{ctx}: t differs at snapshot {i}");
        assert_eq!(
            sa.int_state.counts, sb.int_state.counts,
            "{ctx}: integer compartments differ at snapshot {i} (t={})",
            sa.t
        );
        let ra: Vec<u64> = sa.real_state.values.iter().map(|v| v.to_bits()).collect();
        let rb: Vec<u64> = sb.real_state.values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(ra, rb, "{ctx}: real compartments differ at snapshot {i} (t={})", sa.t);
        // Compare the `Flows` enum directly (it derives PartialEq), so this
        // works for both the stochastic (Int) and ODE (Real) backends —
        // `ode_tick_does_not_change_trajectory` exercises the Real arm.
        assert_eq!(
            sa.flows, sb.flows,
            "{ctx}: flows differ at snapshot {i} (t={})",
            sa.t
        );
    }
}

#[test]
fn chain_binomial_tick_does_not_change_trajectory() {
    let (model, params) = load_sir();
    let cfg = ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 };

    let traj_none =
        run_chain_binomial_with_observer(&model, &params, SEED, &cfg, None, None, Default::default())
            .unwrap();

    let mut ticks: Vec<f64> = Vec::new();
    let mut tick = |t: f64| ticks.push(t);
    let traj_ticked = run_chain_binomial_with_observer(
        &model, &params, SEED, &cfg, None, Some(&mut tick), Default::default(),
    )
    .unwrap();

    assert_traj_eq(&traj_none, &traj_ticked, "chain_binomial");
    assert!(!ticks.is_empty(), "tick never fired — test is vacuous");
}

#[test]
fn gillespie_tick_does_not_change_trajectory() {
    let (model, params) = load_sir();
    let cfg = GillespieConfig { t_start: 0.0, t_end: T_END, output_dt: None };

    let traj_none =
        run_gillespie_with_observer(&model, &params, SEED, &cfg, None, None).unwrap();

    let mut ticks: Vec<f64> = Vec::new();
    let mut tick = |t: f64| ticks.push(t);
    let traj_ticked =
        run_gillespie_with_observer(&model, &params, SEED, &cfg, None, Some(&mut tick)).unwrap();

    assert_traj_eq(&traj_none, &traj_ticked, "gillespie");
    assert!(!ticks.is_empty(), "tick never fired — test is vacuous");
}

#[test]
fn ode_tick_does_not_change_trajectory() {
    // The ODE backend runs on `sir_basic` too: integer compartments are
    // integrated as real-valued via `int_float_override`. No ODE-specific
    // fixture is needed for the tick-invariance property (ODE has no RNG at
    // all, so the tick is doubly safe — but we still assert it changes
    // nothing).
    let (model, params) = load_sir();
    let cfg = OdeConfig { t_start: 0.0, t_end: T_END, dt: 1.0 };

    let traj_none = run_ode(&model, &params, &cfg, None, None).unwrap();

    let mut ticks: Vec<f64> = Vec::new();
    let mut tick = |t: f64| ticks.push(t);
    let traj_ticked = run_ode(&model, &params, &cfg, Some(&mut tick), None).unwrap();

    assert_traj_eq(&traj_none, &traj_ticked, "ode");
    assert!(!ticks.is_empty(), "tick never fired — test is vacuous");
}
