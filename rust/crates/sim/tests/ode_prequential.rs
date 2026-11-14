//! gh#312: the ODE skeleton's per-observation record, which `camdl compare`
//! builds an ODE fit's prequential trace from.
//!
//! Given θ the ODE state `x_t(θ)` is a point, so the one-step-ahead
//! predictive at an observation is the observation density at that point.
//! The record must therefore carry, at each observation, exactly the term the
//! ODE likelihood (`compute_ode_loglik`) sums, per stream and jointly, and the
//! trace built from it must score each step by that term — not by an estimate
//! of it.
//!
//! The oracle is computed by hand from the integrator's own reported flows:
//! the negative-binomial mass is written out as a product of ratios
//! (`Γ(y+r)/(Γ(r)·y!) = Π_{j<y} (r+j)/(j+1)`, exact for integer `y` and any
//! real `r`), so no likelihood code of the engine is used to check the engine.
//!
//! Fixture: the committed `seir_observations` golden — weekly incidence under
//! `neg_binomial(mean = rho * projected, r = k)` and a fortnightly Bernoulli
//! detection stream, so the record also covers two cadences and a hole.

use std::sync::Arc;

use sim::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    inference::{
        compute_ode_loglik, dense_cells, ode_prequential_recorded,
        multi_stream_obs::{StreamProjection, StreamSpec},
        prequential::build_trace,
        BoundObs, MultiStreamObsModel, ODE_PREDICTIVE_RNG_STREAM,
    },
    rng::StatefulRng,
    simulate::Simulate,
    OdeSim,
};

const DT: f64 = 1.0;
const SAMPLES: usize = 400;

/// (name, value) for every parameter: the golden's `baseline` scenario.
const THETA: [(&str, f64); 8] = [
    ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1), ("rho", 0.5),
    ("k", 5.0), ("p_detect", 0.8), ("N0", 100000.0), ("I0", 10.0),
];

fn compiled() -> Arc<CompiledModel> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ocaml/golden/seir_observations.ir.json");
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut model = ir::from_str(&json).expect("parse seir_observations IR");
    for p in &mut model.parameters {
        p.value = ir::parameter::ParamValue::Fixed { value: theta(&p.name) };
    }
    Arc::new(CompiledModel::new(model).expect("compile seir_observations"))
}

fn params(compiled: &CompiledModel) -> Vec<f64> {
    let p = compiled.default_params.clone();
    for (name, v) in THETA {
        assert_eq!(p[compiled.param_index[name]], v, "{name} carries its value");
    }
    p
}

fn theta(name: &str) -> f64 {
    THETA.iter().find(|(n, _)| *n == name).unwrap().1
}

/// The negative-binomial log mass with mean `mu` and size `r`, by hand.
fn nb_logpmf(y: f64, mu: f64, r: f64) -> f64 {
    let yi = y as u64;
    assert_eq!(yi as f64, y, "the hand oracle needs an integer count");
    let log_coef: f64 = (0..yi).map(|j| ((r + j as f64) / (j as f64 + 1.0)).ln()).sum();
    log_coef + r * (r / (r + mu)).ln() + y * (mu / (r + mu)).ln()
}

/// Weekly cases at 7..=70 (one hole, at t = 28) and detection at 14..=70.
const WEEKS: [f64; 10] = [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0];
const CASES: [Option<f64>; 10] = [
    Some(3.0), Some(6.0), Some(14.0), None, Some(50.0),
    Some(90.0), Some(160.0), Some(300.0), Some(480.0), Some(700.0),
];
const FORTNIGHTS: [f64; 5] = [14.0, 28.0, 42.0, 56.0, 70.0];
const DETECTION: [f64; 5] = [1.0, 0.0, 1.0, 1.0, 0.0];

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let infection = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let i_comp = compiled.model.compartments.iter().position(|c| c.name == "I").unwrap();
    let cases_ir = compiled.model.observations.iter().find(|o| o.name == "weekly_cases").unwrap();
    let det_ir = compiled.model.observations.iter().find(|o| o.name == "detection").unwrap();
    let cases = StreamSpec::dense(
        0.0, StreamProjection::FlowSum(vec![infection]), cases_ir.clone(),
        CASES.iter().map(|c| c.map(sim::inference::multi_stream_obs::ObsCell::Scalar)).collect(),
        WEEKS.to_vec());
    let detection = StreamSpec::dense(
        0.0, StreamProjection::IntCompSum(vec![i_comp]), det_ir.clone(),
        dense_cells(DETECTION.to_vec()), FORTNIGHTS.to_vec());
    let (bound, report) = BoundObs::bind(0.0, vec![cases, detection]).expect("binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
}

/// The integrator's own infection flow over `(t − 7, t]`, per week.
fn weekly_incidence(compiled: &CompiledModel, params: &[f64]) -> Vec<f64> {
    let infection = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: compiled.model.simulation.t_end, dt: DT });
    let traj = OdeSim.run(compiled, params, 1, &cfg).expect("ode run");
    WEEKS.iter().map(|&t| traj.snapshots.iter()
        .filter(|s| s.t > t - 7.0 + 1e-9 && s.t <= t + 1e-9)
        .map(|s| s.flows.as_real()[infection])
        .sum()).collect()
}

fn record(compiled: &Arc<CompiledModel>, obs: &MultiStreamObsModel, params: &[f64])
    -> sim::inference::particle_filter::PrequentialRecorded
{
    let mut rng = StatefulRng::new_stream(7, ODE_PREDICTIVE_RNG_STREAM);
    ode_prequential_recorded(compiled, obs, obs.obs_times(), DT, params, DT, SAMPLES, &mut rng)
        .expect("the ODE record")
}

/// Each recorded term is the exact observation density at the ODE state,
/// stream by stream, and the joint is their sum; summed over observations the
/// record reproduces the ODE likelihood.
#[test]
fn the_record_carries_the_exact_density_at_the_ode_state() {
    let compiled = compiled();
    let params = params(&compiled);
    let obs = obs_model(&compiled);
    let rec = record(&compiled, &obs, &params);
    let inc = weekly_incidence(&compiled, &params);

    assert_eq!(rec.obs_times, WEEKS.to_vec(), "one entry per union observation time");
    assert_eq!(rec.stream_names, vec!["weekly_cases".to_string(), "detection".to_string()]);
    let (rho, k, p) = (theta("rho"), theta("k"), theta("p_detect"));
    let mut hand_total = 0.0;
    for (w, &t) in WEEKS.iter().enumerate() {
        let hand_cases = CASES[w].map_or(0.0, |y| nb_logpmf(y, rho * inc[w], k));
        let hand_det = FORTNIGHTS.iter().position(|&f| f == t).map_or(0.0, |d| {
            let y = DETECTION[d];
            y * p.ln() + (1.0 - y) * (1.0 - p).ln()
        });
        assert_eq!(rec.log_liks[w].len(), 1, "one deterministic state support point");
        let got_cases = rec.per_stream_log_liks[w][0][0];
        let got_det = rec.per_stream_log_liks[w][1][0];
        assert!((got_cases - hand_cases).abs() < 1e-9 * hand_cases.abs().max(1.0),
            "t = {t}: weekly_cases term {got_cases}, by hand {hand_cases}");
        assert!((got_det - hand_det).abs() < 1e-12,
            "t = {t}: detection term {got_det}, by hand {hand_det}");
        assert_eq!(rec.log_liks[w][0], got_cases + got_det, "the joint is the per-stream sum");
        hand_total += hand_cases + hand_det;
    }
    let ll = compute_ode_loglik(&compiled, &obs, obs.obs_times(), DT, &params, DT).unwrap();
    let recorded_total: f64 = rec.log_liks.iter().map(|l| l[0]).sum();
    assert!((recorded_total - ll).abs() < 1e-9 * ll.abs(),
        "the record sums to the ODE likelihood: {recorded_total} vs {ll}");
    assert!((ll - hand_total).abs() < 1e-9 * ll.abs(), "and to the hand total {hand_total}");
}

/// The predictive draws are draws from the observation model at the ODE
/// state: `SAMPLES` per scheduled stream, none (NaN) where a stream is not
/// scheduled, Bernoulli draws in {0, 1}, and the negative-binomial draws
/// centred on `rho · incidence`.
#[test]
fn the_record_draws_from_the_observation_model_at_the_ode_state() {
    let compiled = compiled();
    let params = params(&compiled);
    let obs = obs_model(&compiled);
    let rec = record(&compiled, &obs, &params);
    let inc = weekly_incidence(&compiled, &params);
    let (rho, k) = (theta("rho"), theta("k"));
    for (w, &t) in WEEKS.iter().enumerate() {
        let cases = &rec.per_stream_samples[w][0];
        assert_eq!(cases.len(), SAMPLES);
        let mu = rho * inc[w];
        let mean = cases.iter().sum::<f64>() / SAMPLES as f64;
        let se = ((mu + mu * mu / k) / SAMPLES as f64).sqrt();
        assert!((mean - mu).abs() < 5.0 * se,
            "t = {t}: draw mean {mean} vs NB mean {mu} (se {se})");
        let det = &rec.per_stream_samples[w][1];
        assert_eq!(det.len(), SAMPLES);
        if FORTNIGHTS.contains(&t) {
            assert!(det.iter().all(|&v| v == 0.0 || v == 1.0), "t = {t}: Bernoulli draws");
        } else {
            assert!(det.iter().all(|v| v.is_nan()), "t = {t}: detection is not scheduled");
        }
    }
}

/// The trace built from the record scores each step by the recorded term
/// itself: the trace builder's `logsumexp − ln N` over one support point is
/// the term, exact. The hole is not scored, the not-scheduled stream has no
/// entry, and the single support point is reported as an ESS of 1 without an
/// ESS-collapse warning.
#[test]
fn the_trace_scores_each_step_by_the_exact_term() {
    let compiled = compiled();
    let params = params(&compiled);
    let obs = obs_model(&compiled);
    let rec = record(&compiled, &obs, &params);
    let ess = vec![1.0; rec.obs_times.len()];
    let trace = build_trace(&rec, &obs.per_stream_observed(), &obs.per_stream_coverage(),
        &ess, 0, 7, obs.first_observation_opens_after_run_start(0.0), None);
    assert_eq!(trace.steps.len(), WEEKS.len());
    for (w, step) in trace.steps.iter().enumerate() {
        assert_eq!(step.log_score, rec.log_liks[w][0], "t = {}: the step's score is the term", step.t);
        assert_eq!(step.ess, 1.0);
        let names: Vec<&str> = step.per_stream.iter().map(|s| s.stream.as_str()).collect();
        let mut want = Vec::new();
        if CASES[w].is_some() { want.push("weekly_cases"); }
        if FORTNIGHTS.contains(&step.t) { want.push("detection"); }
        assert_eq!(names, want, "t = {}: the streams observed there", step.t);
        for ss in &step.per_stream {
            assert_eq!(ss.y_pred_samples.len(), SAMPLES);
        }
    }
    assert!(!trace.warnings.iter().any(|w| matches!(w,
        score::trace::PrequentialWarning::EssCollapse { .. })),
        "{:?}", trace.warnings);
    let ll = compute_ode_loglik(&compiled, &obs, obs.obs_times(), DT, &params, DT).unwrap();
    assert!((trace.elpd() - ll).abs() < 1e-9 * ll.abs(),
        "an in-sample plug-in ODE trace's elpd is the ODE log-likelihood: {} vs {ll}", trace.elpd());
}
