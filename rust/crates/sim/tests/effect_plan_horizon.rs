//! The particle filter fires effects on the plan of the model's declared
//! window `[t_start, simulation.t_end]` — the plan `simulate` builds — not one
//! whose horizon is the last observation. So the data never move an effect: an
//! observation appended later cannot change where an earlier effect fires, and
//! an effect `simulate` fires at the last observed step is seen by the filter.
//!
//! The model is `sir_basic` (S, I, R) with one `add(I, ·)` event and a Poisson
//! reading of `I`; `beta = gamma = 0`, so the only state change is the event.

use std::sync::Arc;

use serde_json::{json, Value};
use sim::compiled_model::CompiledModel;
use sim::inference::obs_loglik::poisson_logpmf;
use sim::inference::particle_filter::bootstrap_filter;
use sim::inference::traits::{ObservationModel, SMCConfig};
use sim::inference::{ChainBinomialProcess, ParticleState};

/// `sir_basic` with `I = 10`, zero rates, `simulate { to = t_end }`, and
/// `add(I, amount)` events at `at` (none when empty).
fn model(at: &[f64], amount: f64, t_end: f64) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let m = &mut doc["model"];
    if !at.is_empty() {
        m["interventions"] = json!([{
            "name": "surge",
            "fire": {"scheduled": {"at_times": at}},
            "actions": [{"add": {"compartment": "I", "count": {"const": amount}}}],
            "kind": "event",
        }]);
    }
    let mut model = ir::from_str(&serde_json::to_string(&doc).unwrap()).expect("parse");
    model.simulation.t_end = t_end;
    for p in &mut model.parameters {
        let v = match p.name.as_str() { "beta" | "gamma" => 0.0, "N0" => 1000.0, "I0" => 10.0, _ => 1.0 };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(model).expect("compile"))
}

/// Poisson readings of `I` (local int index 1) at `times`.
struct PoissonI {
    y: Vec<f64>,
    times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonI {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        poisson_logpmf(self.y[obs_idx], (state.counts[1] as f64).max(0.1))
    }
    fn n_observations(&self) -> usize { self.y.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.times[obs_idx] }
}

fn increments(m: &Arc<CompiledModel>, times: &[f64]) -> Vec<f64> {
    let obs = PoissonI { y: times.iter().map(|_| 11.0).collect(), times: times.to_vec() };
    let config = SMCConfig {
        n_particles: 8, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false, record_ancestry: false,
        record_prequential: false, record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter(&ChainBinomialProcess::new(m.clone()), &obs, &m.default_params, &config, 3)
        .expect("filter")
        .ll_increments
}

/// An effect in the model's shortened last step (`to = 4.5`, dt = 1; 4.3 is
/// nearer 4.5 than 4) fires at 4.5 however the data end: appending an
/// observation at 4.5 leaves the increments of 3.5, 4 and 4.4 unchanged, and
/// the reading at 4.4 does not see the event. With the last observation as the
/// horizon, the data [3.5, 4, 4.4] made 4.4 the end of a shortened step and
/// fired the event there.
#[test]
fn appending_an_observation_does_not_move_an_earlier_effect() {
    let m = model(&[4.3], 1.0, 4.5);
    let short = increments(&m, &[3.5, 4.0, 4.4]);
    let long = increments(&m, &[3.5, 4.0, 4.4, 4.5]);
    assert_eq!(short, long[..3], "the first three increments must not move");
    let p10 = poisson_logpmf(11.0, 10.0);
    let p11 = poisson_logpmf(11.0, 11.0);
    assert_eq!(short, vec![p10, p10, p10], "the event is not seen at 4.4");
    assert_eq!(long[3], p11, "the event is seen at 4.5, where it fires");
}

/// `add(I, 500)` at 10.3 with dt = 1 and `to = 20` fires at the end of the step
/// ending at 10 in `simulate` (⌊10.3 + ½⌋ = 10), so a reading at 10 — the last
/// observation — sees it, exactly as one at 10.0 would.
#[test]
fn the_filter_sees_an_effect_simulate_fires_at_the_last_observation() {
    let times: Vec<f64> = (1..=10).map(f64::from).collect();
    let late = increments(&model(&[10.3], 500.0, 20.0), &times);
    let on_grid = increments(&model(&[10.0], 500.0, 20.0), &times);
    let none = increments(&model(&[], 0.0, 20.0), &times);
    assert_eq!(late, on_grid, "10.3 fires where 10.0 does");
    assert_ne!(late[9], none[9], "the reading at 10 sees the event");
    assert_eq!(late[..9], none[..9], "no earlier reading sees it");
}
