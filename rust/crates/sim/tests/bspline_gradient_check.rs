//! R2 of proposal 2026-10-27-estimable-bsplines §8.1: the compiler-emitted
//! gradient of a spline coefficient against central finite differences.
//!
//! - Forcing level (`bspline_forcing_grad`): each transition's rate is
//!   `forcing * X`, so its emitted `rate_grad` is the forcing's coefficient
//!   derivative times X. Checked at points inside the window, at every knot, at
//!   both window ends and past the end, for an identity link, a log link with a
//!   constant lag, a user-written `exp` of an identity-link spline, coefficients
//!   that are arithmetic over one parameter, and a log-link periodic spline.
//! - Likelihood level (`sir_bspline_fit`): `complete_data_loglik_grad` for a
//!   log-link spline in the transmission rate and an identity-link spline in a
//!   Poisson observation mean, against the value path's central differences.
//!
//! The criterion is R2's, not the harness default: step ε = 1e-5·max(|θ|, 1);
//! relative error < 1e-4 for every component with |g_fd| ≥ 1e-3·max_k |g_fd,k|,
//! and absolute error ≤ 1e-7·max_k |g_fd,k| for the rest.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{build_obs_at_substep, complete_data_loglik, simulate_reference};
use sim::inference::pgas_grad::complete_data_loglik_grad;
use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};
use sim::resolved_expr::{eval_resolved, ResolvedDerivEntry};
use sim::rng::StatefulRng;

fn load(name: &str, values: &[(&str, f64)]) -> Arc<CompiledModel> {
    let path = format!("{}/../../../tests/fixtures/gradient/ir/{name}.ir.json", env!("CARGO_MANIFEST_DIR"));
    let mut m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for p in &mut m.parameters {
        let v = values.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no value for parameter {}", p.name));
        p.value = p.value.with_value(v);
    }
    Arc::new(CompiledModel::new(m).unwrap())
}

/// R2's step.
fn step(theta: f64) -> f64 {
    1e-5 * theta.abs().max(1.0)
}

/// R2's pass rule over one gradient vector against its finite differences.
/// Returns the worst relative error among the appreciably nonzero components.
fn r2(label: &str, names: &[String], analytic: &[f64], fd: &[f64]) -> f64 {
    let scale = fd.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    assert!(scale > 0.0, "{label}: every finite difference is zero — the check is vacuous");
    let mut worst: f64 = 0.0;
    for ((n, &a), &f) in names.iter().zip(analytic).zip(fd) {
        if f.abs() >= 1e-3 * scale {
            let rel = (a - f).abs() / f.abs();
            assert!(rel < 1e-4, "{label} ∂/∂{n}: analytic {a:.10e} vs fd {f:.10e}, rel err {rel:.2e}");
            worst = worst.max(rel);
        } else {
            assert!((a - f).abs() <= 1e-7 * scale, "{label} ∂/∂{n}: analytic {a:.3e} vs fd {f:.3e} near zero (scale {scale:.3e})");
        }
    }
    worst
}

const FORCING_VALUES: &[(&str, f64)] =
    &[("c1", -0.8), ("c2", 0.4), ("c3", 1.1), ("c4", -0.2), ("c5", 0.7), ("c6", 0.3), ("a", 0.6)];

#[test]
fn spline_coefficient_gradients_match_finite_differences_pointwise() {
    let model = load("bspline_forcing_grad", FORCING_VALUES);
    let params = model.default_params.clone();
    let (int_s, real_s) = model.initial_state_mean(&params).unwrap();
    let names: Vec<String> = {
        let mut v = vec![String::new(); params.len()];
        for (n, &i) in &model.param_index {
            v[i] = n.clone();
        }
        v
    };
    // Inside, at every knot of each layout (0, 10, 20, 25, 30, 40, 60), at both
    // ends, past the end, before the start, and across the lag (t − 3 = 60).
    let times = [-5.0, 0.0, 0.5, 7.3, 10.0, 17.9, 20.0, 25.0, 30.0, 33.3, 40.0, 51.2, 59.9, 60.0, 63.0, 70.0, 80.0];
    let eval = |rate: &sim::resolved_expr::ResolvedExpr, p: &[f64], t: f64| {
        let ctx = sim::propensity::EvalCtx {
            model: &model, int_s: &int_s, real_s: &real_s, params: p,
            t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        eval_resolved(rate, &ctx)
    };
    let mut worst: f64 = 0.0;
    let mut checked = 0;
    for (ti, tr) in model.model.transitions.iter().enumerate() {
        let rate = &model.resolved.rates[ti];
        let grads = &model.resolved.rate_grads_indexed[ti];
        assert!(!grads.is_empty(), "{} carries no rate_grad", tr.name);
        for &t in &times {
            let mut cols = Vec::new();
            let (mut analytic, mut fd) = (Vec::new(), Vec::new());
            for (pi, entry) in grads {
                let ResolvedDerivEntry::Grad(g) = entry else { panic!("{}: an Unsupported entry", tr.name) };
                let h = step(params[*pi]);
                let (mut up, mut dn) = (params.clone(), params.clone());
                up[*pi] += h;
                dn[*pi] -= h;
                cols.push(names[*pi].clone());
                analytic.push(eval(g, &params, t));
                fd.push((eval(rate, &up, t) - eval(rate, &dn, t)) / (2.0 * h));
            }
            if fd.iter().all(|g| *g == 0.0) {
                continue; // a coefficient set with no support at t (cannot happen here; guarded)
            }
            worst = worst.max(r2(&format!("{} t={t}", tr.name), &cols, &analytic, &fd));
            checked += 1;
        }
    }
    assert_eq!(checked, 5 * times.len(), "every transition at every time was checked");
    eprintln!("forcing-level spline gradients: {checked} points, worst rel err {worst:.2e}");
}

/// Weekly Poisson observations of incidence, bound for the likelihood path.
fn observed(model: &Arc<CompiledModel>, times: &[f64], counts: &[f64]) -> MultiStreamObsModel {
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.to_vec()).unwrap(),
            projection,
            ir_model: om.clone(),
            observations: dense_cells(counts.to_vec()),
            aux: vec![],
        }
    }).collect();
    MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap()
}

/// `complete_data_loglik_grad` against the value path, for the rate spline
/// (b1…b5), the observation-mean spline (r1…r4) and `gamma`.
#[test]
fn spline_coefficient_gradients_through_the_complete_data_likelihood() {
    let values: &[(&str, f64)] = &[
        ("gamma", 0.15), ("b1", -1.0), ("b2", -0.9), ("b3", -1.4), ("b4", -1.7), ("b5", -1.2),
        ("r1", 0.6), ("r2", 0.5), ("r3", 0.4), ("r4", 0.45),
    ];
    let model = load("sir_bspline_fit", values);
    let params = model.default_params.clone();
    let dt = 1.0;
    let mut rng = StatefulRng::new(7);
    let traj = simulate_reference(&model, &params, model.model.simulation.t_end, dt, &mut rng).unwrap();
    let times: Vec<f64> = (1..=9).map(|w| 7.0 * w as f64).collect();
    let counts: Vec<f64> = (0..times.len()).map(|i| (20 + 7 * i) as f64).collect();
    let obs_model = observed(&model, &times, &counts);
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let oas = build_obs_at_substep(&observations, model.model.simulation.t_start, dt).unwrap();

    let n = params.len();
    let estimated: Vec<usize> = (0..n).collect();
    let model_to_estimated: Vec<Option<usize>> = (0..n).map(Some).collect();
    let rate_grads = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &model.resolved.rate_grads_indexed, &model_to_estimated);
    let (ll, grad) = complete_data_loglik_grad(
        &model, &traj, &params, &observations, dt, &obs_model, n, &rate_grads, &oas, &estimated,
    ).unwrap();
    assert!(ll.is_finite());
    let total = |p: &[f64]| complete_data_loglik(&model, &traj, p, &observations, dt, &obs_model, &oas).unwrap().total;
    assert!((ll - total(&params)).abs() <= 1e-9 * ll.abs().max(1.0), "gradient fn energy equals the value path");

    let mut names = Vec::new();
    let (mut analytic, mut fd) = (Vec::new(), Vec::new());
    for (name, _) in values {
        let i = model.param_index[*name];
        let h = step(params[i]);
        let (mut up, mut dn) = (params.clone(), params.clone());
        up[i] += h;
        dn[i] -= h;
        names.push(name.to_string());
        analytic.push(grad[i]);
        fd.push((total(&up) - total(&dn)) / (2.0 * h));
        eprintln!("  ∂ll/∂{name:5} = {:14.6e} (analytic) vs {:14.6e} (fd)", grad[i], fd.last().unwrap());
    }
    // Non-vacuity: both splines' coefficients move the likelihood.
    for name in ["b2", "r2"] {
        let k = names.iter().position(|n| n == name).unwrap();
        assert!(fd[k].abs() > 1e-3, "∂ll/∂{name} must be appreciably nonzero, got {:.3e}", fd[k]);
    }
    let worst = r2("complete-data", &names, &analytic, &fd);
    eprintln!("complete-data spline gradients: worst rel err {worst:.2e}");
}

/// §5.4: with the coefficients carrying a `Grad` entry, the PGAS-NUTS preflight
/// admits them, NUTS is the active kernel, and a short run stays finite.
#[test]
fn pgas_nuts_admits_spline_coefficients() {
    use sim::inference::if2::{EstimatedParam, Transform};
    use sim::inference::pgas::{nuts_active, run_pgas, PGASConfig};
    use sim::inference::pmmh::Prior;
    let values: &[(&str, f64)] = &[
        ("gamma", 0.15), ("b1", -1.0), ("b2", -0.9), ("b3", -1.4), ("b4", -1.7), ("b5", -1.2),
        ("r1", 0.6), ("r2", 0.5), ("r3", 0.4), ("r4", 0.45),
    ];
    let model = load("sir_bspline_fit", values);
    assert!(nuts_active(true, &model), "the rate spline's coefficients make NUTS the kernel");
    let times: Vec<f64> = (1..=9).map(|w| 7.0 * w as f64).collect();
    let counts: Vec<f64> = (0..times.len()).map(|i| (20 + 7 * i) as f64).collect();
    let obs_model = observed(&model, &times, &counts);
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let est: Vec<EstimatedParam> = ["b1", "b2", "b3", "b4", "b5", "r2"].iter().map(|n| {
        let index = model.param_index[*n];
        let (lower, upper) = if n.starts_with('r') { (0.0, 1.0) } else { (-5.0, 2.0) };
        EstimatedParam {
            name: (*n).into(), index, initial: model.default_params[index], rw_sd: 0.05,
            transform: Transform::None, lower, upper, rw_sd_auto: false, perturb_only_at_t0: false,
        }
    }).collect();
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat); est.len()];
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 16, n_sweeps: 6, burn_in: 2, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: false, max_tree_depth: 4,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };
    let result = run_pgas(&model, &est, &priors, &model.default_params, &config, &observations,
                          &obs_model, 3, None, None, "bspline-nuts".into())
        .expect("PGAS-NUTS must admit spline coefficients");
    assert!(!result.sweeps.is_empty());
    for s in &result.sweeps {
        assert!(s.params.iter().all(|v| v.is_finite()), "finite draws: {:?}", s.params);
    }
}
