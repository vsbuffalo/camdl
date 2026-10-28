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
//! - PGAS (gh#986): the complete-data density scores the recorded path; the
//!   effect map and a deterministic `init {}` entry appear in no term, so an
//!   estimated parameter there has a posterior equal to its prior. Refused at
//!   the `run_pgas` boundary for an effect amount, an effect schedule and a
//!   plain `init` assignment.
//!
//! The model is the book's seeding idiom on `sir_basic`:
//! `events { founders_arrive : add(I, n_seed) at [tau] }`. A periodic-spline
//! variant reaches the amount through a forcing coefficient instead.

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
    let Err(err) = build_if2(&model, &["beta", "tau"]) else { panic!("tau sets the fire time: must be refused") };
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
    let Err(err) = run_if2(&process, &obs, &model.default_params, &if2_params, &config, 1) else {
        panic!("IF2 with tau estimated must be refused")
    };
    assert!(err.to_string().contains("if2: `tau` is estimated but sets the fire time of event `founders_arrive`"), "{err}");
}

// ─── PGAS ───────────────────────────────────────────────────────────────────

fn run_pgas_estimating(model: &Arc<CompiledModel>, estimated: &[&str]) -> Result<(), String> {
    use sim::inference::if2::{EstimatedParam, Transform};
    use sim::inference::multi_stream_obs::MultiStreamObsModel;
    use sim::inference::pgas::{run_pgas, PGASConfig};
    use sim::inference::pmmh::Prior;
    let if2_params: Vec<EstimatedParam> = estimated
        .iter()
        .map(|n| {
            let index = model.param_index[*n];
            EstimatedParam {
                name: (*n).into(),
                index,
                initial: model.default_params[index],
                rw_sd: 0.1,
                transform: Transform::None,
                lower: 0.0,
                upper: 1e4,
                rw_sd_auto: false,
                perturb_only_at_t0: false,
            }
        })
        .collect();
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat); if2_params.len()];
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 4, n_sweeps: 1, burn_in: 0, thin: 1, dt: 1.0,
        use_nuts: false, dense_mass: false, max_tree_depth: 4,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };
    let obs = MultiStreamObsModel::empty(model.clone());
    run_pgas(model, &if2_params, &priors, &model.default_params, &config, &[], &obs, 1, None, None, "d3".into())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// An estimated effect AMOUNT is refused under PGAS, naming it and the event.
#[test]
fn pgas_refuses_an_estimated_effect_amount() {
    let err = run_pgas_estimating(&seeding_model(), &["beta", "n_seed"]).expect_err("must be refused");
    assert_eq!(
        err,
        "model validation error: pgas: `n_seed` is estimated and sets the amount of event \
         `founders_arrive`, which the PGAS parameter move cannot score, so the sampler cannot \
         move `n_seed` through it. Fit with PMMH, or fix `n_seed`."
    );
}

/// An estimated effect SCHEDULE is refused under PGAS.
#[test]
fn pgas_refuses_an_estimated_effect_schedule() {
    let err = run_pgas_estimating(&seeding_model(), &["tau"]).expect_err("must be refused");
    assert_eq!(
        err,
        "model validation error: pgas: `tau` is estimated and sets the fire time of event \
         `founders_arrive`, which the PGAS parameter move cannot score, so the sampler cannot \
         move `tau` through it. Fit with PMMH, or fix `tau`."
    );
}

/// A parameter in a plain `init {}` assignment is refused under PGAS, pointing
/// at the law spelling. `sir_basic` has `S = N0 - I0` and `I = I0`; the
/// refusal names the entry that is the parameter itself.
#[test]
fn pgas_refuses_an_estimated_plain_init_assignment() {
    let err = run_pgas_estimating(&seeding_model(), &["I0"]).expect_err("must be refused");
    assert_eq!(
        err,
        "model validation error: pgas: `I0` is estimated and enters `init { I = I0 }`, which the \
         PGAS parameter move cannot score, so the sampler cannot move `I0` through it. Declare a \
         law, `I ~ poisson(rate = I0)`, or fit with PMMH."
    );
}

/// A rate parameter alone is not refused: PGAS scores it through the
/// transition densities. (The run then proceeds; one sweep is enough here.)
#[test]
fn pgas_accepts_a_rate_parameter() {
    run_pgas_estimating(&seeding_model(), &["beta"]).expect("a rate parameter is scored by PGAS");
}

/// A spline coefficient is live (proposal 2026-10-27-estimable-bsplines §4.2,
/// §4.3), so a parameter can reach an effect amount through the forcing. PGAS
/// must refuse it as it refuses a direct amount: the event `add(I, season)`
/// reads forcing `season`, whose second coefficient is `c2`. Checked for both
/// spline kinds.
fn refuses_a_spline_coefficient_in_an_effect_amount(kind: ir::time_func::TimeFuncKind) {
    use ir::expr::{Expr, TimeFuncRef, TimeFuncWrap};
    use ir::time_func::TimeFunction;
    let base = seeding_model();
    let mut model = (*base.model).clone();
    model.parameters.push(ir::parameter::Parameter {
        name: "c2".into(),
        value: ir::parameter::ParamValue::Fixed { value: 2.0 },
        param_kind: None,
        param_dim: None,
    });
    model.time_functions.push(TimeFunction {
        name: "season".into(),
        kind,
        dim: (0, 0),
        lag: None,
        data_source: None,
    });
    let mut doc = serde_json::to_value(&model).unwrap();
    doc["interventions"][0]["actions"][0]["add"]["count"] =
        serde_json::to_value(Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: "season".into() } })).unwrap();
    let model: ir::Model = serde_json::from_value(doc).unwrap();
    let model = Arc::new(CompiledModel::new(model).expect("compile"));
    let err = run_pgas_estimating(&model, &["beta", "c2"]).expect_err("must be refused");
    assert_eq!(
        err,
        "model validation error: pgas: `c2` is estimated and sets the amount of event \
         `founders_arrive`, which the PGAS parameter move cannot score, so the sampler cannot \
         move `c2` through it. Fit with PMMH, or fix `c2`."
    );
    // Control: the same model with only `beta` estimated is not refused.
    run_pgas_estimating(&model, &["beta"]).expect("a rate parameter alone is scored by PGAS");
}

fn spline_coefs() -> Vec<ir::expr::Expr> {
    use ir::expr::Expr;
    vec![
        Expr::const_(1.0), Expr::param("c2"), Expr::const_(1.0),
        Expr::const_(1.0), Expr::const_(1.0), Expr::const_(1.0),
    ]
}

#[test]
fn pgas_refuses_an_estimated_spline_coefficient_in_an_effect_amount() {
    use ir::time_func::{PeriodicSpline, SplineLink, TimeFuncKind};
    refuses_a_spline_coefficient_in_an_effect_amount(TimeFuncKind::PeriodicSpline(PeriodicSpline {
        period: ir::expr::Expr::const_(4.0),
        n_basis: 6,
        degree: 3,
        coefs: spline_coefs(),
        link: SplineLink::Identity,
    }));
}

#[test]
fn pgas_refuses_an_estimated_bspline_coefficient_in_an_effect_amount() {
    use ir::time_func::{BSpline, BSplineKnots, SplineCoefs, SplineLink, TimeFuncKind};
    refuses_a_spline_coefficient_in_an_effect_amount(TimeFuncKind::BSpline(BSpline {
        knots: BSplineKnots::Uniform {
            lower: ir::expr::Expr::const_(0.0),
            upper: ir::expr::Expr::const_(10.0),
            n_intervals: 3,
        },
        degree: 3,
        coefs: SplineCoefs::Explicit { coefs: spline_coefs() },
        link: SplineLink::Identity,
    }));
}
