//! gh#204 PR2 slice 6: forward chain-binomial reactive *behavior* goldens.
//!
//! Three properties the proposal calls out, tested through the sim API (direct
//! `Trajectory` access — no TSV parsing, fully deterministic given the seed):
//!
//!   - **cooldown** suppresses re-firing within the window, then re-fires;
//!   - the **equivalence oracle** — a reactive policy and a fixed intervention
//!     placed at the resulting fire time produce a *byte-identical* trajectory
//!     (the realized-obs draws run on a separate RNG, so the dynamics RNG is
//!     consumed identically and both apply the same transfer at the same
//!     boundary);
//!   - **reporting scale** — with `rho != 1` the trigger reads the realized
//!     `rho`-scaled report, not the raw incidence.
//!
//! lag / once / default-off are pinned by `reactive_log.rs`, `reactive_obs.rs`,
//! and `reactive_capability.rs`.

use ir::intervention::{FireSource, InterventionSchedule, ReactiveTrigger};
use sim::config::{ChainBinomialConfig, SimConfig};
use sim::simulate::Simulate;
use sim::{ChainBinomialSim, CompiledModel};
use std::path::PathBuf;

fn load_golden() -> ir::Model {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    ir::from_str(&s).unwrap_or_else(|e| panic!("deser fixture: {e:?}"))
}

fn cfg() -> SimConfig {
    SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 365.0, dt: 1.0 })
}

/// Compile the model and return it with a params vector carrying `overrides`
/// (by name). Any parameter the golden leaves estimated is filled to a concrete
/// placeholder first so compilation has defaults.
fn compiled_with(mut model: ir::Model, overrides: &[(&str, f64)]) -> (CompiledModel, Vec<f64>) {
    for prm in &mut model.parameters {
        if prm.value.resolved_value().is_none() {
            prm.value = prm.value.with_value(0.5);
        }
    }
    let c = CompiledModel::new(model).expect("compile reactive golden");
    let mut params = c.default_params.clone();
    for (name, v) in overrides {
        params[c.param_index[*name]] = *v;
    }
    (c, params)
}

/// The golden's single reactive trigger, mutable (for cooldown/once tweaks).
fn reactive_trigger_mut(model: &mut ir::Model) -> &mut ReactiveTrigger {
    for iv in &mut model.interventions {
        if let FireSource::Reactive(t) = &mut iv.fire {
            return t;
        }
    }
    panic!("the golden has a reactive policy");
}

/// A realistic-but-deterministic SIR run (R0 = 3).
const BASE: &[(&str, f64)] = &[
    ("beta", 0.3), ("gamma", 0.1), ("rho", 0.2),
    ("trigger_threshold", 2.0), ("sia_cov", 0.7), ("N0", 1000.0), ("I0", 10.0),
];

/// `cooldown` suppresses re-firing within the window, then re-fires. With
/// `threshold = 0` every weekly emit crosses (`obs >= 0` always holds), so the
/// firing pattern is governed purely by the cooldown — not the dynamics — and
/// `sia_cov = 0` keeps the run unperturbed so the trigger stays crossable.
#[test]
fn cooldown_suppresses_then_refires() {
    let mut model = load_golden();
    {
        let t = reactive_trigger_mut(&mut model);
        t.once = false;
        t.cooldown = Some(30.0);
    }
    let (c, params) = compiled_with(model, &[
        ("beta", 0.3), ("gamma", 0.1), ("rho", 0.2),
        ("trigger_threshold", 0.0), ("sia_cov", 0.0), ("N0", 1000.0), ("I0", 10.0),
    ]);
    let traj = ChainBinomialSim.run(&c, &params, 1, &cfg()).unwrap();
    let log = traj.reactive_log.expect("active agenda records a log");

    assert!(log.len() >= 2, "cooldown=30 over weekly emits fires several times, got {}", log.len());
    assert_eq!(log[0].trigger_time, 7.0, "first crossing is the first weekly emit");
    for w in log.windows(2) {
        assert!(
            w[1].trigger_time - w[0].trigger_time >= 30.0,
            "consecutive firings must be >= cooldown apart: {} -> {}",
            w[0].trigger_time, w[1].trigger_time
        );
    }
}

/// The equivalence oracle: a reactive policy and a fixed intervention placed at
/// the resulting fire time produce the SAME trajectory, byte-for-byte.
#[test]
fn reactive_equals_scheduled_at_the_fire_time() {
    // 1. reactive run (once=true), find the fire time it produces.
    let (c_r, params_r) = compiled_with(load_golden(), BASE);
    let traj_r = ChainBinomialSim.run(&c_r, &params_r, 1, &cfg()).unwrap();
    let log = traj_r.reactive_log.clone().expect("active agenda");
    assert_eq!(log.len(), 1, "once=true fires exactly once");
    let fire_time = log[0].fire_time;

    // 2. scheduled variant: the same model, but the reactive policy becomes a
    //    fixed intervention firing at exactly `fire_time` (same actions/kind).
    let mut model_s = load_golden();
    for iv in &mut model_s.interventions {
        if matches!(iv.fire, FireSource::Reactive(_)) {
            iv.fire = FireSource::Scheduled(InterventionSchedule::AtTimes(vec![fire_time]));
        }
    }
    let (c_s, params_s) = compiled_with(model_s, BASE);
    let traj_s = ChainBinomialSim.run(&c_s, &params_s, 1, &cfg()).unwrap();

    // 3. byte-identical: same compartment counts and flows at every snapshot.
    assert_eq!(traj_r.snapshots.len(), traj_s.snapshots.len(), "same number of snapshots");
    for (a, b) in traj_r.snapshots.iter().zip(&traj_s.snapshots) {
        assert_eq!(a.t, b.t, "snapshot times align");
        assert_eq!(a.int_state.counts, b.int_state.counts, "compartments diverge at t={}", a.t);
        assert_eq!(a.flows.as_int(), b.flows.as_int(), "flows diverge at t={}", a.t);
    }

    // Guard against a vacuous pass: the campaign must actually move mass at the
    // fire time (S drops into V), or "identical" would just mean "no-op".
    let v_idx = c_r.model.compartments.iter().position(|c| c.name == "V").expect("V compartment");
    let before = traj_r.snapshots.iter().rev()
        .find(|s| s.t < fire_time).map(|s| s.int_state.counts[v_idx]).unwrap_or(0);
    let after = traj_r.snapshots.iter()
        .find(|s| s.t >= fire_time).map(|s| s.int_state.counts[v_idx]).expect("a snapshot at/after fire_time");
    assert!(after > before, "the campaign must vaccinate (V {before} -> {after} at fire_time {fire_time})");
}

/// With `rho != 1` the trigger reads the realized `rho`-scaled report, not the
/// raw incidence: the logged `trigger_value` is far below the week's raw
/// infections and consistent with `Poisson(rho * raw)`.
#[test]
fn trigger_reads_the_realized_report_not_raw_incidence() {
    let rho = 0.2;
    let (c, params) = compiled_with(load_golden(), &[
        ("beta", 0.3), ("gamma", 0.1), ("rho", rho),
        ("trigger_threshold", 1.0), ("sia_cov", 0.0), ("N0", 1000.0), ("I0", 10.0),
    ]);
    let traj = ChainBinomialSim.run(&c, &params, 1, &cfg()).unwrap();
    let log = traj.reactive_log.expect("active agenda");
    assert_eq!(log[0].trigger_time, 7.0, "fires at the first weekly emit");
    let reported = log[0].trigger_value; // realized Poisson(rho * raw)

    // raw incidence over the first week = sum of flow_infection (transition 0)
    // in (0, 7] — the same span the weekly_cases interval accumulator covers.
    let inf = 0usize;
    let raw: u64 = traj.snapshots.iter()
        .filter(|s| s.t > 0.0 && s.t <= 7.0 + 1e-9)
        .map(|s| s.flows.as_int()[inf])
        .sum();
    let raw = raw as f64;
    assert!(raw > 10.0, "the first week has real incidence to report (raw={raw})");

    let mean = rho * raw;
    assert!(
        reported < 0.5 * raw,
        "reported {reported} must be far below raw incidence {raw} (rho={rho}) — \
         a raw-incidence trigger would report ~{raw}"
    );
    assert!(
        (reported - mean).abs() < 5.0 * mean.sqrt() + 3.0,
        "reported {reported} must be consistent with Poisson(rho*raw = {mean})"
    );
}
