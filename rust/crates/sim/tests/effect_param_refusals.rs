//! D3 (proposal 2026-10-04 §5): an estimated parameter that the fit's θ-move
//! cannot score is refused, with the parameter and the effect named, instead of
//! being "fitted" to its starting value or its prior.
//!
//! - IF2 (gh#984): the effect plan is built once per run from the base
//!   parameters, so an estimated parameter in any effect's schedule — events
//!   included — would leave the fire time at its starting value for every
//!   particle while the swarm's value for it drifts. Refused by
//!   `ExactInferenceTimeline::build`; an estimated amount is allowed, since
//!   `step` reads it with the particle's own θ.
//!
//! The model is the book's seeding idiom on `sir_basic`:
//! `events { founders_arrive : add(I, n_seed) at [tau] }`.

use std::sync::Arc;

use serde_json::{json, Value};
use sim::boundary_times::ObsTimes;
use sim::compiled_model::CompiledModel;
use sim::effect_plan::FireClock;
use sim::intervention::ExactInferenceTimeline;

/// `sir_basic` plus `tau` and `n_seed` and the seeding event.
fn seeding_model() -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let m = &mut doc["model"];
    for (name, v) in [("tau", 2.0), ("n_seed", 3.0)] {
        m["parameters"].as_array_mut().unwrap().push(json!({
            "name": name, "value": {"mode": "fixed", "value": v},
            "param_kind": null, "param_dim": null,
        }));
    }
    m["interventions"] = json!([{
        "name": "founders_arrive",
        "fire": {"scheduled": {"at_times_expr": [{"param": "tau"}]}},
        "actions": [{"add": {"compartment": "I", "count": {"param": "n_seed"}}}],
        "kind": "event",
    }]);
    let mut model = ir::from_str(&serde_json::to_string(&doc).unwrap()).expect("parse");
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.name.as_str() { "beta" => 0.5, "gamma" => 0.2, "N0" => 100.0, _ => 5.0 };
            p.value = ir::parameter::ParamValue::Fixed { value: v };
        }
    }
    Arc::new(CompiledModel::new(model).expect("compile"))
}

fn build_if2(model: &CompiledModel, estimated: &[&str]) -> Result<ExactInferenceTimeline, String> {
    let idx: Vec<usize> = estimated.iter().map(|n| model.param_index[*n]).collect();
    ExactInferenceTimeline::build(
        Some(model),
        &model.default_params,
        0.0,
        1.0,
        FireClock::Grid { dt: 1.0 },
        ObsTimes::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        &idx,
    )
    .map_err(|e| e.to_string())
}

/// IF2 with `tau` estimated is refused, naming `tau` and `founders_arrive`.
#[test]
fn if2_refuses_an_estimated_parameter_in_an_event_schedule() {
    let model = seeding_model();
    let err = build_if2(&model, &["beta", "tau"]).err().expect("tau sets the fire time: must be refused");
    assert_eq!(
        err,
        "model validation error: if2: `tau` is estimated but sets the fire time of event \
         `founders_arrive`, which IF2 resolves once per run, so the fit cannot move it. Fit with \
         PMMH, or fix `tau`."
    );
}

/// IF2 with only the amount `n_seed` estimated is accepted, and the fire time
/// is the base value of `tau` (2): the plan carries one firing instant at 2.
#[test]
fn if2_accepts_an_estimated_amount() {
    let model = seeding_model();
    let timeline = build_if2(&model, &["beta", "n_seed"]).expect("an estimated amount is allowed under IF2");
    assert_eq!(timeline.plan.times(), &[2.0]);
    // A shared θ (PF, PMMH) is never refused: it rebuilds its plan per evaluation.
    build_if2(&model, &[]).expect("a shared θ is not refused");
}

/// The refusal is wired into `run_if2` itself: its estimated set is what
/// reaches the build.
#[test]
fn run_if2_refuses_an_estimated_event_time() {
    use sim::inference::if2::{run_if2, EstimatedParam, IF2Config, Transform};
    use sim::inference::multi_stream_obs::MultiStreamObsModel;
    use sim::inference::ChainBinomialProcess;
    let model = seeding_model();
    let process = ChainBinomialProcess::new(model.clone());
    let obs = MultiStreamObsModel::empty(model.clone());
    let tau = model.param_index["tau"];
    let if2_params = vec![EstimatedParam {
        name: "tau".into(),
        index: tau,
        initial: 2.0,
        rw_sd: 0.1,
        transform: Transform::None,
        lower: 0.0,
        upper: 10.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }];
    let config = IF2Config {
        n_particles: 4,
        n_iterations: 1,
        cooling_fraction: 0.5,
        cooling_target_iters: 50,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let err = run_if2(&process, &obs, &model.default_params, &if2_params, &config, 1)
        .err()
        .expect("IF2 with tau estimated must be refused");
    assert!(err.to_string().contains("if2: `tau` is estimated but sets the fire time of event `founders_arrive`"), "{err}");
}
