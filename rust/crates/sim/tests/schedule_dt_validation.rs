//! gh#126: schedule / time-step validation must run in RELEASE builds.
//!
//! Before the fix, `time.rs` guarded `dt > 0` and `t.is_finite()` only with
//! `debug_assert!`, which is compiled out of `--release`. A bad (or
//! parameter-proposed) `dt` therefore slipped past validation in production: a
//! non-positive `dt` made the backend emit a trajectory FROZEN at the initial
//! state (the substep loop's `dt <= 1e-15` break fires immediately), and a
//! non-finite `dt` fed `NaN`/`±∞` straight into the kernel — a silent wrong
//! answer either way, not a controlled error. The fix wires
//! `CompiledModel::validate_schedule` (always-on) into every backend entry
//! point, so a bad time-axis is now a named `SimError::Validation` *before* any
//! stepping.
//!
//! These tests run regardless of build profile (`cargo test` and
//! `cargo test --release`), pinning that the rejection is NOT debug-only.

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};
use std::path::Path;

fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

fn load_model(name: &str) -> ir::Model {
    let contents = std::fs::read_to_string(golden_path(name))
        .unwrap_or_else(|_| panic!("could not read golden {name}"));
    let mut model: ir::Model =
        ir::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"));
    // sir_basic carries parameter bounds but no inline `value`; supply
    // in-bounds values so `CompiledModel::new` resolves `default_params`
    // and the backend is runnable. The dynamics are irrelevant here — the
    // tests only assert the dt/schedule gate fires (or, for the positive
    // control, that a valid dt still produces snapshots).
    let defaults: &[(&str, f64)] =
        &[("beta", 0.5), ("gamma", 0.2), ("N0", 1000.0), ("I0", 10.0)];
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    model
}

fn load_compiled(name: &str) -> CompiledModel {
    CompiledModel::new(load_model(name)).unwrap()
}

fn assert_named_dt_error(res: Result<sim::Trajectory, sim::SimError>, ctx: &str) {
    let err = res.expect_err(&format!("{ctx}: a bad dt must be rejected, not silently run"));
    assert!(
        matches!(err, sim::SimError::Validation(_)),
        "{ctx}: expected SimError::Validation, got {err:?}"
    );
    assert!(
        format!("{err}").contains("dt"),
        "{ctx}: error must name dt: {err}"
    );
}

#[test]
fn chain_binomial_rejects_zero_dt() {
    let compiled = load_compiled("sir_basic");
    let params = compiled.default_params.clone();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: 0.0 });
    assert_named_dt_error(ChainBinomialSim.run(&compiled, &params, 1, &cfg), "chain_binomial dt=0");
}

#[test]
fn chain_binomial_rejects_negative_dt() {
    let compiled = load_compiled("sir_basic");
    let params = compiled.default_params.clone();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: -1.0 });
    assert_named_dt_error(ChainBinomialSim.run(&compiled, &params, 1, &cfg), "chain_binomial dt<0");
}

#[test]
fn chain_binomial_rejects_nan_dt() {
    let compiled = load_compiled("sir_basic");
    let params = compiled.default_params.clone();
    let cfg =
        SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: f64::NAN });
    assert_named_dt_error(ChainBinomialSim.run(&compiled, &params, 1, &cfg), "chain_binomial dt=NaN");
}

#[test]
fn ode_rejects_nonpositive_dt() {
    let compiled = load_compiled("sir_basic");
    let params = compiled.default_params.clone();
    let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 30.0, dt: 0.0 });
    assert_named_dt_error(OdeSim.run(&compiled, &params, 1, &cfg), "ode dt=0");
}

#[test]
fn chain_binomial_still_runs_with_valid_dt() {
    // Guard against over-rejection: a valid dt must still produce a trajectory.
    let compiled = load_compiled("sir_basic");
    let params = compiled.default_params.clone();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: 1.0 });
    let traj = ChainBinomialSim
        .run(&compiled, &params, 1, &cfg)
        .expect("valid dt must run");
    assert!(!traj.snapshots.is_empty(), "valid run must emit snapshots");
}

#[test]
fn gillespie_rejects_nonfinite_resolution_dt() {
    // Gillespie keys interventions on `model.simulation.dt` (its
    // iv_resolution_dt). A model whose simulation.dt is non-finite/non-positive
    // must be rejected at entry, not used as a step-rounding resolution.
    let mut model = load_model("sir_basic");
    model.simulation.dt = Some(f64::NAN);
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    let cfg = SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: 30.0, output_dt: Some(1.0) });
    assert_named_dt_error(GillespieSim.run(&compiled, &params, 1, &cfg), "gillespie dt=NaN");
}

// ── gh#257: output-step / recurrence-period positivity ────────────────────────
//
// `output_times` (`t += step`) and each `Recurring` intervention's fire-time
// enumeration (`t += period`) are infinite-loop hazards with the same shape as
// a non-positive `dt`: a `step`/`period` of `0` (or a negative) never advances
// the loop cursor, so the loop `push`es to a `Vec` UNBOUNDED and exhausts
// memory (an OOM, not merely a hang). The recurring fire-time loop enumerates
// inside `CompiledModel::new` — fire times are baked at construction — so a
// non-positive period would OOM before any backend guard could run. The guard
// therefore lives at the construction boundary (`CompiledModel::new`): a bad
// step/period is rejected before `new` returns, so the loop is never entered.
//
// These tests assert `CompiledModel::new` returns a named error. They must NOT
// `.unwrap()` a bad model (constructing it IS the OOM) nor drive a `run` (the
// model never constructs). The builders return the pre-compile `Model`; the
// caller compiles it, which is where the gate fires.

use ir::intervention::{
    Action, AddAction, FireSource, Intervention, InterventionSchedule, RecurringSchedule,
};
use ir::model::{OutputSchedule, RegularOutputSchedule};

fn assert_named_construction_error(
    res: Result<CompiledModel, sim::SimError>,
    needle: &str,
    ctx: &str,
) {
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("{ctx}: bad schedule must be rejected at construction, not constructed"),
    };
    assert!(
        matches!(err, sim::SimError::Validation(_)),
        "{ctx}: expected SimError::Validation, got {err:?}"
    );
    assert!(
        format!("{err}").contains(needle),
        "{ctx}: error must name `{needle}`: {err}"
    );
}

/// Build a `sir_basic` model whose `Regular` output schedule carries the given
/// step. Returns the pre-compile `Model`; the caller runs `CompiledModel::new`,
/// which is where the gh#257 positivity gate fires.
fn model_with_output_step(step: f64) -> ir::Model {
    let mut model = load_model("sir_basic");
    model.output.times = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step });
    model
}

/// Build a `sir_basic` model with one `Recurring` intervention of the given
/// period. A minimal `add(S, 0)` action keeps the model well-formed; only the
/// schedule period is under test. Returns the pre-compile `Model` (see
/// [`model_with_output_step`]).
fn model_with_recurring_period(period: f64) -> ir::Model {
    let mut model = load_model("sir_basic");
    model.interventions.push(Intervention {
        name: "pulse".to_string(),
        base_name: None,
        fire: FireSource::Scheduled(InterventionSchedule::Recurring(RecurringSchedule {
            start: 0.0,
            period,
            end: 30.0,
            at_day: None,
        })),
        actions: vec![Action::Add(AddAction {
            compartment: "S".to_string(),
            count: ir::expr::Expr::const_(0.0),
        })],
        kind: Default::default(),
    });
    model
}

#[test]
fn rejects_zero_output_step() {
    assert_named_construction_error(
        CompiledModel::new(model_with_output_step(0.0)),
        "step",
        "output step = 0",
    );
}

#[test]
fn rejects_negative_output_step() {
    assert_named_construction_error(
        CompiledModel::new(model_with_output_step(-1.0)),
        "step",
        "output step < 0",
    );
}

#[test]
fn rejects_zero_recurrence_period() {
    assert_named_construction_error(
        CompiledModel::new(model_with_recurring_period(0.0)),
        "period",
        "recurrence period = 0",
    );
}

#[test]
fn rejects_negative_recurrence_period() {
    assert_named_construction_error(
        CompiledModel::new(model_with_recurring_period(-7.0)),
        "period",
        "recurrence period < 0",
    );
}

#[test]
fn still_runs_with_valid_output_step_and_recurrence() {
    // Over-rejection guard: a valid output step AND a valid recurring period
    // must construct AND produce a trajectory.
    let compiled = CompiledModel::new(model_with_recurring_period(7.0))
        .expect("valid step + period must construct");
    let params = compiled.default_params.clone();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: 1.0 });
    let traj = ChainBinomialSim
        .run(&compiled, &params, 1, &cfg)
        .expect("valid step + period must run");
    assert!(!traj.snapshots.is_empty(), "valid run must emit snapshots");
}
