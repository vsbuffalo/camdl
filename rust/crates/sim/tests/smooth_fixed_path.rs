//! Diagnostic, not a check: NUTS on the parameters with the latent epidemic
//! path held fixed, against full PGAS+NUTS (proposal
//! 2026-10-27-estimable-bsplines §8.1, R3). If the parameters mix well given a
//! fixed path but not in PGAS, the slow convergence is state–parameter
//! coupling; if they mix badly even given the path, it is the conditional
//! NUTS problem itself.
//!
//! For each recovery dataset (data seeds 23, 31, 47, 59; `synthesize` as in
//! `bspline_inference.rs`) and each form (explicit coefficients with
//! normal(−1.5, 1) priors; `smooth` with its fixture's priors), the path is
//! held at (i) the true path the data were simulated from and (ii) a
//! posterior path, the final reference path of one PGAS+NUTS chain (1,000
//! adaptation + 1,000 sweeps). On it, 4 chains of NUTS on the complete-data
//! posterior p(θ | x, y): identity scale for coefficients, level, slope and
//! modes, log scale for τ; 1,500 warm-up iterations (dual averaging to a 0.9
//! acceptance target, a dense mass matrix estimated from iterations 500–1,000,
//! then re-tuned step size), then 3,000 kept. Prints, per run, the largest
//! rank-normalized R̂ and smallest bulk-ESS of log β(t) over 41 times in the
//! middle 80% of the window, of the coefficients w, and of z, τ, a, b; ESS per
//! second; divergences; and mean leapfrog steps.
//!
//! cargo test --release -p sim --test smooth_fixed_path -- --ignored --nocapture

use std::sync::Arc;
use std::time::Instant;

use sim::bspline::{span, Knots};
use sim::compiled_model::{CompiledModel, CompiledTimeFuncKind};
use sim::inference::convergence::rank_convergence;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::nuts::{nuts_step, DualAveraging, MassMatrix, NUTSConfig};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{build_obs_at_substep, run_pgas, simulate_reference, PGASConfig, PGASTrajectory};
use sim::inference::pgas_grad::{complete_data_loglik_grad, resolve_rate_grad_for_run};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};
use sim::rng::StatefulRng;

const TRUTH: [f64; 5] = [-0.7, -1.0, -1.6, -1.4, -1.9];
const FIXED: [(&str, f64); 7] = [
    ("sigma", 0.2), ("gamma", 0.2), ("rho", 0.5), ("k", 10.0), ("N0", 1_000_000.0), ("E0", 100.0), ("I0", 100.0),
];

fn load(name: &str, values: &[(&str, f64)]) -> Arc<CompiledModel> {
    let path = format!("{}/../../../tests/fixtures/bspline_inference/ir/{name}.ir.json", env!("CARGO_MANIFEST_DIR"));
    let mut m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for p in &mut m.parameters {
        let v = values.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no value for {}", p.name));
        p.value = p.value.with_value(v);
    }
    Arc::new(CompiledModel::new(m).unwrap())
}

/// The R3 datasets: the true path and the daily onsets (as `synthesize`).
fn synthesize(model: &CompiledModel, seed: u64) -> (PGASTrajectory, Vec<f64>, Vec<f64>) {
    let mut rng = StatefulRng::new(seed);
    let traj = simulate_reference(model, &model.default_params, model.model.simulation.t_end, 1.0, &mut rng).unwrap();
    let prog = model.model.transitions.iter().position(|t| t.name == "progression").unwrap();
    let mut obs_rng = StatefulRng::new(seed ^ 0x5eed);
    let (mut times, mut counts) = (Vec::new(), Vec::new());
    for (s, sub) in traj.substeps.iter().enumerate() {
        times.push((s + 1) as f64);
        counts.push(obs_rng.neg_binomial_dispersion(0.5 * sub.flows[prog] as f64, 10.0) as f64);
    }
    (traj, times, counts)
}

fn observed(model: &Arc<CompiledModel>, times: &[f64], counts: &[f64]) -> MultiStreamObsModel {
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.to_vec()).unwrap(),
            projection, ir_model: om.clone(), observations: dense_cells(counts.to_vec()), aux: vec![],
        }
    }).collect();
    MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap()
}

/// The estimated parameters and priors of each form, on the scales `fit run`
/// samples them.
fn problem(model: &CompiledModel, smooth: bool) -> (Vec<EstimatedParam>, Vec<Prior>) {
    let inf = f64::INFINITY;
    let ep = |name: &str, transform: Transform, lo: f64, hi: f64, initial: f64| EstimatedParam {
        name: name.into(), index: model.param_index[name], initial, rw_sd: 0.05, transform,
        lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
    };
    let normal = |mean: f64, sd: f64| Prior::Fixed(Density::Normal { mean, sd });
    if !smooth {
        let est = (1..=5).map(|i| ep(&format!("b{i}"), Transform::None, -inf, inf, -1.5)).collect();
        return (est, vec![normal(-1.5, 1.0); 5]);
    }
    let mut est = vec![
        ep("a", Transform::None, -inf, inf, -1.5),
        ep("b", Transform::None, -inf, inf, 0.0),
        ep("tau", Transform::Log { lo: 0.0, hi: f64::INFINITY }, 0.0, f64::INFINITY, 0.2),
    ];
    let mut priors = vec![normal(-1.5, 1.0), normal(0.0, 0.02), Prior::Fixed(Density::Exponential { rate: 5.99 })];
    for i in 1..=3 {
        est.push(ep(&format!("z_m{i}"), Transform::None, -inf, inf, 0.0));
        priors.push(normal(0.0, 1.0));
    }
    (est, priors)
}

fn log_prior(p: &Prior, x: f64) -> (f64, f64) {
    match p {
        Prior::Fixed(Density::Normal { mean, sd }) => (-0.5 * ((x - mean) / sd).powi(2), -(x - mean) / (sd * sd)),
        Prior::Fixed(Density::Exponential { rate }) => (-rate * x, -rate),
        _ => unreachable!("only the priors this file builds"),
    }
}

/// The spline coefficients w a full parameter vector implies.
fn coefficients(model: &CompiledModel, params: &[f64]) -> Vec<f64> {
    let CompiledTimeFuncKind::BSpline(spline) = &model.time_func_cache[0].kind else { unreachable!() };
    let v = |n: &str| params[model.param_index[n]];
    let Some(w) = spline.whitening() else {
        return (1..=5).map(|i| v(&format!("b{i}"))).collect();
    };
    let kap = sim::spline_penalty::kappa(w.n_basis(), 7.0);
    (0..w.n_basis()).map(|j| v("a") + v("b") * (w.xi[j] - w.t_mid)
        + v("tau") / kap * (0..w.n_modes()).map(|i| w.m.at(j, i) * v(&format!("z_m{}", i + 1))).sum::<f64>())
        .collect()
}

fn log_beta(c: &[f64], t: f64) -> f64 {
    let knots = Knots::uniform(0.0, 86.0, 2).unwrap();
    let s = span(&knots, 3, t);
    (0..=3).map(|r| s.values[r] * c[s.first + r]).sum()
}

struct ChainOut {
    draws: Vec<Vec<f64>>, // full parameter vectors
    divergent: usize,
    leapfrog: usize,
}

/// NUTS on θ | x with the path fixed: warm-up 1,500 (dual averaging to 0.9;
/// dense mass from iterations 500–1,000, then the step size re-tuned), 3,000 kept.
fn fixed_path_chain(model: &CompiledModel, path: &PGASTrajectory, obs: &MultiStreamObsModel,
                    observations: &[Observation], est: &[EstimatedParam], priors: &[Prior], seed: u64) -> ChainOut {
    let d = est.len();
    let n = model.default_params.len();
    let mut model_to_est: Vec<Option<usize>> = vec![None; n];
    for (i, p) in est.iter().enumerate() {
        model_to_est[p.index] = Some(i);
    }
    let rate_grads = resolve_rate_grad_for_run(&model.resolved.rate_grads_indexed, &model_to_est);
    let oas = build_obs_at_substep(observations, 0.0, 1.0).unwrap();
    let est_to_model: Vec<usize> = est.iter().map(|p| p.index).collect();
    let to_theta = |u: &[f64]| -> Vec<f64> {
        let mut th = model.default_params.clone();
        for (p, &x) in est.iter().zip(u) {
            th[p.index] = match p.transform { Transform::Log { .. } => x.exp(), _ => x };
        }
        th
    };
    let lp = |u: &[f64]| -> (f64, Vec<f64>) {
        let th = to_theta(u);
        let (ll, g) = complete_data_loglik_grad(model, path, &th, observations, 1.0, obs, d, &rate_grads, &oas, &est_to_model)
            .unwrap_or((f64::NEG_INFINITY, vec![0.0; d]));
        let mut total = ll;
        let mut grad = vec![0.0; d];
        for i in 0..d {
            let x = th[est[i].index];
            let (lpr, dpr) = log_prior(&priors[i], x);
            total += lpr;
            match est[i].transform {
                Transform::Log { .. } => {
                    total += u[i];
                    grad[i] = (g[i] + dpr) * x + 1.0;
                }
                _ => grad[i] = g[i] + dpr,
            }
        }
        if !total.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; d]);
        }
        (total, grad)
    };
    let mut rng = StatefulRng::new(seed);
    let mut u: Vec<f64> = est.iter().map(|p| match p.transform {
        Transform::Log { .. } => p.initial.ln(), _ => p.initial,
    } + 0.1 * rng.normal()).collect();
    let (mut l, mut g) = lp(&u);
    let mut cfg = NUTSConfig { max_tree_depth: 10, step_size: 0.05, mass_matrix: MassMatrix::identity(d) };
    let mut da = DualAveraging::new(cfg.step_size, 0.9);
    let mut window: Vec<Vec<f64>> = Vec::new();
    let mut out = ChainOut { draws: Vec::new(), divergent: 0, leapfrog: 0 };
    for it in 0..4500 {
        let r = nuts_step(&u, l, &g, &cfg, &lp, &mut rng);
        u = r.params;
        (l, g) = lp(&u);
        if it < 1500 {
            cfg.step_size = da.update(r.mean_accept_prob);
            if (500..1000).contains(&it) {
                window.push(u.clone());
            }
            if it == 999 {
                let m = window.len() as f64;
                let mean: Vec<f64> = (0..d).map(|i| window.iter().map(|w| w[i]).sum::<f64>() / m).collect();
                let mut cov = vec![0.0; d * d];
                for w in &window {
                    for i in 0..d {
                        for j in 0..d {
                            cov[i * d + j] += (w[i] - mean[i]) * (w[j] - mean[j]) / (m - 1.0);
                        }
                    }
                }
                cfg.mass_matrix = MassMatrix::dense_from_covariance(&cov, d);
                da = DualAveraging::new(cfg.step_size, 0.9);
            }
            if it == 1499 {
                cfg.step_size = da.final_step_size();
            }
        } else {
            out.divergent += r.divergent as usize;
            out.leapfrog += r.n_leapfrog;
            out.draws.push(to_theta(&u));
        }
    }
    out
}

/// Largest R̂ and smallest bulk-ESS over quantities f(params), 4 chains.
fn worst(chains: &[ChainOut], fs: &[Box<dyn Fn(&[f64]) -> f64 + '_>]) -> (f64, f64) {
    fs.iter().map(|f| {
        let per: Vec<Vec<f64>> = chains.iter().map(|c| c.draws.iter().map(|p| f(p)).collect()).collect();
        let r = rank_convergence(&per).unwrap();
        (r.rhat, r.ess_bulk)
    }).fold((0.0_f64, f64::INFINITY), |(r, e), (r2, e2)| (r.max(r2), e.min(e2)))
}

#[test]
#[ignore = "diagnostic; about twenty minutes in release; run by the smooth-diagnosis target"]
fn nuts_with_the_latent_path_fixed() {
    let mut truth: Vec<(&str, f64)> = FIXED.to_vec();
    truth.extend(["b1", "b2", "b3", "b4", "b5"].into_iter().zip(TRUTH));
    let explicit = load("seir_bspline_recovery", &truth);
    let mut sv: Vec<(&str, f64)> = FIXED.to_vec();
    sv.extend([("a", -1.5), ("b", 0.0), ("tau", 0.2), ("z_m1", 0.0), ("z_m2", 0.0), ("z_m3", 0.0)]);
    let smooth_model = load("seir_smooth_recovery", &sv);
    let grid: Vec<f64> = (0..=40).map(|i| 0.1 * 86.0 + 0.8 * 86.0 * i as f64 / 40.0).collect();
    println!("seed\tform\tpath\twall_s\tdivergent\tleapfrog_mean\trhat_beta\tess_beta\tess_per_s_beta\trhat_w\tess_w\trhat_z\tess_z\trhat_tau\tess_tau\trhat_ab\tess_ab");
    for seed in [23u64, 31, 47, 59] {
        let (true_path, times, counts) = synthesize(&explicit, seed);
        let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
        for (form, model) in [("explicit", explicit.clone()), ("smooth", smooth_model.clone())] {
            let obs = observed(&model, &times, &counts);
            let (est, priors) = problem(&model, form == "smooth");
            // A posterior path: the final reference path of one PGAS chain.
            let mut base = model.default_params.clone();
            for p in &est {
                base[p.index] = p.initial;
            }
            let config = PGASConfig {
                binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
                n_particles: 64, n_sweeps: 2000, burn_in: 1000, thin: 1, dt: 1.0,
                use_nuts: true, dense_mass: true, max_tree_depth: 6,
                trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
            };
            let post = run_pgas(&model, &est, &priors, &base, &config, &observations, &obs, 9, None, None,
                "fixed-path".into()).expect("PGAS for a posterior path").final_trajectory;
            for (label, path) in [("true", &true_path), ("posterior", &post)] {
                let start = Instant::now();
                let chains: Vec<ChainOut> = std::thread::scope(|sc| {
                    let hs: Vec<_> = (1..=4u64).map(|c| {
                        let (model, obs, est, priors, observations) = (&model, &obs, &est, &priors, &observations);
                        sc.spawn(move || fixed_path_chain(model, path, obs, observations, est, priors, 1000 * seed + c))
                    }).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let wall = start.elapsed().as_secs_f64();
                let m = &model;
                let beta: Vec<Box<dyn Fn(&[f64]) -> f64>> = grid.iter()
                    .map(|&t| Box::new(move |p: &[f64]| log_beta(&coefficients(m, p), t)) as Box<dyn Fn(&[f64]) -> f64>)
                    .collect();
                let w: Vec<Box<dyn Fn(&[f64]) -> f64>> = (0..5)
                    .map(|j| Box::new(move |p: &[f64]| coefficients(m, p)[j]) as Box<dyn Fn(&[f64]) -> f64>)
                    .collect();
                let named = |names: &[&str]| -> Vec<Box<dyn Fn(&[f64]) -> f64>> {
                    names.iter().map(|n| {
                        let i = m.param_index[*n];
                        Box::new(move |p: &[f64]| p[i]) as Box<dyn Fn(&[f64]) -> f64>
                    }).collect()
                };
                let (rb, eb) = worst(&chains, &beta);
                let (rw, ew) = worst(&chains, &w);
                let fmt = |x: Option<(f64, f64)>| x.map_or("-\t-".to_string(), |(r, e)| format!("{r:.3}\t{e:.0}"));
                let (z, tau, ab) = if form == "smooth" {
                    (Some(worst(&chains, &named(&["z_m1", "z_m2", "z_m3"]))), Some(worst(&chains, &named(&["tau"]))),
                     Some(worst(&chains, &named(&["a", "b"]))))
                } else {
                    (None, None, None)
                };
                let div: usize = chains.iter().map(|c| c.divergent).sum();
                let lf = chains.iter().map(|c| c.leapfrog).sum::<usize>() as f64 / (4.0 * 3000.0);
                println!("{seed}\t{form}\t{label}\t{wall:.1}\t{div}\t{lf:.1}\t{rb:.3}\t{eb:.0}\t{:.1}\t{rw:.3}\t{ew:.0}\t{}\t{}\t{}",
                    eb / wall, fmt(z), fmt(tau), fmt(ab));
            }
        }
    }
}
