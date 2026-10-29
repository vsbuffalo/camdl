//! Diagnostic ladder for the PGAS+NUTS divergences (proposal
//! 2026-10-27-estimable-bsplines §8.1, R3). Not a check: every test is
//! `#[ignore]`d and prints a table row per run.
//!
//! - Rung 1, `rung1_bare_nuts_on_a_gaussian`: the NUTS kernel PGAS uses
//!   (`nuts::nuts_step`), no PGAS and no simulator, on the independent
//!   Gaussian of the smooth prior's a, b and z (and, separately, the
//!   log-exponential target of log τ), diagonal mass equal to the true
//!   variances, at the step size dual averaging adapts to 0.8 and at fixed
//!   smaller and larger steps.
//! - Rung 3, `rung3_target_values_reaching_nuts`: the per-parameter target
//!   value `Prior::log_density + EstimatedParam::log_jacobian` that the
//!   PGAS NUTS closure adds, against the analytic log densities.
//! - Rungs 2 and 4, `rungs_2_and_4_through_pgas`: `run_pgas` on models with
//!   one all-missing observation stream: (2) scalars a, b, τ, z with X0 = 0,
//!   τ boxed and unboxed; (4a) the compiled smooth spline with X0 = 0;
//!   (4b) the smooth spline with X0 = 10⁴, so NUTS targets p(θ | X), not the
//!   prior; (4c) the full SEIR recovery fit to data (seed 31).
//!
//! Per run: divergences per chain after adaptation, the adapted step size,
//! the mean leapfrog steps, the smallest bulk-ESS over the estimated
//! parameters, and the energy error ΔH. For rung 1 ΔH comes from a leapfrog
//! probe at the kept states (fresh momentum, the adapted step, as many steps
//! as the transition took); `run_pgas` does not expose its integrator, so for
//! the PGAS rungs the column is the mean acceptance statistic instead.
//!
//! cargo test --release -p sim --test nuts_rungs -- --ignored --nocapture --test-threads 1

use std::path::PathBuf;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::convergence::rank_convergence;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::nuts::{nuts_step, DualAveraging, MassMatrix, NUTSConfig};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};
use sim::rng::StatefulRng;

const LAMBDA: f64 = 5.99;

fn min_ess(chains: &[Vec<Vec<f64>>]) -> f64 {
    let d = chains[0][0].len();
    (0..d).map(|j| {
        let per: Vec<Vec<f64>> = chains.iter().map(|c| c.iter().map(|v| v[j]).collect()).collect();
        rank_convergence(&per).unwrap().ess_bulk
    }).fold(f64::INFINITY, f64::min)
}

fn q(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// One leapfrog trajectory of `n` steps from `u` with momentum drawn for the
/// diagonal mass `var` (M⁻¹ = var); returns max |H − H₀| along it.
fn probe(lp: &dyn Fn(&[f64]) -> (f64, Vec<f64>), u: &[f64], var: &[f64], eps: f64, n: usize, rng: &mut StatefulRng) -> f64 {
    let d = u.len();
    let mut x = u.to_vec();
    let mut p: Vec<f64> = (0..d).map(|i| rng.normal() / var[i].sqrt()).collect();
    let kin = |p: &[f64]| 0.5 * p.iter().zip(var).map(|(pi, v)| pi * pi * v).sum::<f64>();
    let (l0, mut g) = lp(&x);
    let h0 = -l0 + kin(&p);
    let mut worst: f64 = 0.0;
    for _ in 0..n.max(1) {
        for i in 0..d { p[i] += 0.5 * eps * g[i]; }
        for i in 0..d { x[i] += eps * var[i] * p[i]; }
        let (l, g2) = lp(&x);
        g = g2;
        for i in 0..d { p[i] += 0.5 * eps * g[i]; }
        let dh = (-l + kin(&p)) - h0;
        worst = worst.max(if dh.is_finite() { dh.abs() } else { f64::INFINITY });
    }
    worst
}

/// Rung 1 on a separable target: 4 chains, 1,000 warm-up (dual averaging to
/// 0.8 unless `fixed_eps`), 4,000 kept, diagonal mass `var`.
fn bare(label: &str, lp: &(dyn Fn(&[f64]) -> (f64, Vec<f64>) + Sync), start: &[f64], var: &[f64], fixed_eps: Option<f64>) {
    let runs: Vec<(Vec<Vec<f64>>, usize, f64, f64, Vec<f64>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (1..=4u64).map(|seed| sc.spawn(move || {
            let mut rng = StatefulRng::new(seed);
            let mut cfg = NUTSConfig { max_tree_depth: 10, step_size: fixed_eps.unwrap_or(0.1), mass_matrix: MassMatrix::diagonal(var.to_vec()) };
            let mut da = DualAveraging::new(cfg.step_size, 0.8);
            let mut u = start.to_vec();
            let (mut l, mut g) = lp(&u);
            let (mut kept, mut div, mut lf, mut dh) = (Vec::new(), 0, 0usize, Vec::new());
            for it in 0..5000 {
                let r = nuts_step(&u, l, &g, &cfg, lp, &mut rng);
                if it < 1000 {
                    if fixed_eps.is_none() {
                        cfg.step_size = if it == 999 { da.final_step_size() } else { da.update(r.mean_accept_prob) };
                    }
                } else {
                    div += r.divergent as usize;
                    lf += r.n_leapfrog;
                    if it % 10 == 0 {
                        dh.push(probe(lp, &u, var, cfg.step_size, r.n_leapfrog, &mut rng));
                    }
                }
                u = r.params;
                (l, g) = lp(&u);
                if it >= 1000 { kept.push(u.clone()); }
            }
            (kept, div, cfg.step_size, lf as f64 / 4000.0, dh)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let chains: Vec<Vec<Vec<f64>>> = runs.iter().map(|r| r.0.clone()).collect();
    let dh: Vec<f64> = runs.iter().flat_map(|r| r.4.clone()).collect();
    println!("{label}\tdiv {:?}\tε {}\tleapfrog {:.1}\tminESS {:.0}\tΔH median {:.3} q99 {:.3} max {:.3}",
        runs.iter().map(|r| r.1).collect::<Vec<_>>(),
        runs.iter().map(|r| format!("{:.3}", r.2)).collect::<Vec<_>>().join("/"),
        runs.iter().map(|r| r.3).sum::<f64>() / 4.0, min_ess(&chains), q(dh.clone(), 0.5), q(dh.clone(), 0.99), q(dh, 1.0));
}

#[test]
#[ignore = "diagnostic; seconds in release"]
fn rung1_bare_nuts_on_a_gaussian() {
    // a ~ N(−1, 0.6²), b ~ N(0, 0.01²), z1..z4 ~ N(0, 1).
    let mean = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sd = [0.6, 0.01, 1.0, 1.0, 1.0, 1.0];
    let var: Vec<f64> = sd.iter().map(|s| s * s).collect();
    let gauss = |u: &[f64]| -> (f64, Vec<f64>) {
        let l = u.iter().zip(mean.iter().zip(&sd)).map(|(x, (m, s))| -0.5 * ((x - m) / s).powi(2)).sum();
        (l, u.iter().zip(mean.iter().zip(&sd)).map(|(x, (m, s))| -(x - m) / (s * s)).collect())
    };
    bare("rung1 gaussian adapted", &gauss, &mean, &var, None);
    for eps in [0.5, 1.0, 1.5] {
        bare(&format!("rung1 gaussian ε={eps}"), &gauss, &mean, &var, Some(eps));
    }
    // log τ under exponential(λ), with and without the walls the box [1e-6, 2.5] puts at ln 1e-6 and ln 2.5.
    for (name, walls) in [("logtau unboxed", None), ("logtau box", Some(((1e-6_f64).ln(), (2.5_f64).ln())))] {
        let lp = move |u: &[f64]| -> (f64, Vec<f64>) {
            if let Some((lo, hi)) = walls {
                if u[0] < lo || u[0] > hi { return (f64::NEG_INFINITY, vec![0.0]); }
            }
            (u[0] - LAMBDA * u[0].exp(), vec![1.0 - LAMBDA * u[0].exp()])
        };
        let var = [1.64]; // Var(log τ) = π²/6 for an exponential
        bare(&format!("rung1 {name} adapted"), &lp, &[-2.0], &var, None);
        for eps in [0.5, 1.0] {
            bare(&format!("rung1 {name} ε={eps}"), &lp, &[-2.0], &var, Some(eps));
        }
    }
}

#[test]
#[ignore = "diagnostic; instant"]
fn rung3_target_values_reaching_nuts() {
    // The value the PGAS NUTS closure adds per parameter is
    // prior.log_density(θ(z), z) + param.log_jacobian(z) (pgas.rs, the
    // log_prob_and_grad closure). Up to a constant it must equal the analytic
    // log density of the coordinate z: for an identity normal, log N(z); for
    // τ on the log scale, z − λeᶻ.
    let ep = |transform: Transform, lo: f64, hi: f64| EstimatedParam {
        name: "p".into(), index: 0, initial: 0.0, rw_sd: 0.1, transform, lower: lo, upper: hi,
        rw_sd_auto: false, perturb_only_at_t0: false,
    };
    let inf = f64::INFINITY;
    let cases: Vec<(&str, Prior, EstimatedParam, Box<dyn Fn(f64) -> f64>, Vec<f64>)> = vec![
        ("normal(-1, 0.6), identity", Prior::Fixed(Density::Normal { mean: -1.0, sd: 0.6 }),
         ep(Transform::None, -inf, inf), Box::new(|z| -0.5 * ((z + 1.0) / 0.6_f64).powi(2)), vec![-3.0, -1.0, 0.4]),
        ("normal(0, 0.01), identity", Prior::Fixed(Density::Normal { mean: 0.0, sd: 0.01 }),
         ep(Transform::None, -inf, inf), Box::new(|z| -0.5 * (z / 0.01_f64).powi(2)), vec![-0.03, 0.0, 0.02]),
        ("exponential(5.99), log, unboxed", Prior::Fixed(Density::Exponential { rate: LAMBDA }),
         ep(Transform::Log { lo: 0.0, hi: inf }, 0.0, inf), Box::new(|z| z - LAMBDA * z.exp()), vec![-12.0, -2.0, 0.0, 1.5]),
        ("exponential(5.99), log, box [1e-6, 2.5]", Prior::Fixed(Density::Exponential { rate: LAMBDA }),
         ep(Transform::Log { lo: 1e-6, hi: 2.5 }, 1e-6, 2.5), Box::new(|z| z - LAMBDA * z.exp()), vec![-12.0, -2.0, 0.0, 0.9]),
    ];
    for (label, prior, p, analytic, zs) in &cases {
        let got: Vec<f64> = zs.iter().map(|&z| prior.log_density(p.from_transformed(z), z) + p.log_jacobian(z)).collect();
        let offsets: Vec<f64> = got.iter().zip(zs).map(|(g, &z)| g - analytic(z)).collect();
        let spread = offsets.iter().fold(0.0_f64, |m, o| m.max((o - offsets[0]).abs()));
        println!("rung3 {label}: offset spread {spread:.2e} over z = {zs:?}; transform_deriv {:?}, jacobian_grad {:?}",
            zs.iter().map(|&z| p.transform_deriv(z)).collect::<Vec<_>>(),
            zs.iter().map(|&z| p.jacobian_grad(z)).collect::<Vec<_>>());
        assert!(spread <= 1e-9, "{label}: the target value differs from the analytic density by a non-constant");
    }
    // Outside the box, the boxed log transform is a wall.
    let p = ep(Transform::Log { lo: 1e-6, hi: 2.5 }, 1e-6, 2.5);
    println!("rung3 box walls: log_jacobian(1.0) = {}, log_jacobian(-14.0) = {}", p.log_jacobian(1.0), p.log_jacobian(-14.0));
}

// ── Rungs 2 and 4: through run_pgas ─────────────────────────────────────────

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn compile(fixture: &str) -> ir::Model {
    let out = std::env::temp_dir().join(format!("nuts_rungs_{}_{}.ir.json", std::process::id(), fixture.replace('/', "_")));
    let o = std::process::Command::new(repo().join("ocaml/_build/default/bin/camdlc.exe"))
        .arg(repo().join(fixture)).arg("-o").arg(&out).output().unwrap();
    assert!(o.status.success(), "{fixture}: {}", String::from_utf8_lossy(&o.stderr));
    ir::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap()
}

fn pgas_rung(label: &str, fixture: &str, fixed: &[(&str, f64)], data: Option<(&str, u64)>) {
    let mut m = compile(fixture);
    let mut names = Vec::new();
    let mut priors_ir = Vec::new();
    for p in &mut m.parameters {
        if let Some(&(_, v)) = fixed.iter().find(|(n, _)| *n == p.name) {
            p.value = p.value.with_value(v);
            continue;
        }
        if let Some(ir::parameter::PriorSpec::Dist(d)) = p.value.prior() {
            names.push(p.name.clone());
            priors_ir.push((Prior::from_ir(d), p.value.bounds(), p.param_kind));
        }
        p.value = p.value.with_value(if p.name == "tau" { 0.2 } else { 0.0 });
    }
    let model = Arc::new(CompiledModel::new(m).unwrap());
    let inf = f64::INFINITY;
    let est: Vec<EstimatedParam> = names.iter().zip(&priors_ir).map(|(n, (_, bounds, kind))| {
        let positive = matches!(kind, Some(ir::parameter::ParamKind::Positive));
        let (lo, hi) = bounds.unwrap_or(if positive { (0.0, inf) } else { (-inf, inf) });
        let initial = if n == "tau" { 0.2 } else if n == "a" { -1.5 } else { 0.0 };
        EstimatedParam {
            name: n.clone(), index: model.param_index[n.as_str()], initial, rw_sd: 0.05,
            transform: if positive { Transform::Log { lo, hi } } else { Transform::None },
            lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
        }
    }).collect();
    let priors: Vec<Prior> = priors_ir.into_iter().map(|(p, _, _)| p).collect();
    // Data: every value missing, or the R3 dataset of `seed` simulated from `truth_fixture`.
    let (times, cells): (Vec<f64>, Vec<Option<sim::inference::multi_stream_obs::ObsCell>>) = match data {
        None => {
            let times: Vec<f64> = (1..=60).map(|t| t as f64).collect();
            let n = times.len();
            (times, vec![None; n])
        }
        Some((truth_fixture, seed)) => {
            let mut tm = compile(truth_fixture);
            let truth = [("sigma", 0.2), ("gamma", 0.2), ("rho", 0.5), ("k", 10.0), ("N0", 1e6), ("E0", 100.0), ("I0", 100.0),
                         ("b1", -0.7), ("b2", -1.0), ("b3", -1.6), ("b4", -1.4), ("b5", -1.9)];
            for p in &mut tm.parameters {
                p.value = p.value.with_value(truth.iter().find(|(n, _)| *n == p.name).unwrap().1);
            }
            let tmodel = CompiledModel::new(tm).unwrap();
            let mut rng = StatefulRng::new(seed);
            let traj = simulate_reference(&tmodel, &tmodel.default_params, tmodel.model.simulation.t_end, 1.0, &mut rng).unwrap();
            let prog = tmodel.model.transitions.iter().position(|t| t.name == "progression").unwrap();
            let mut orng = StatefulRng::new(seed ^ 0x5eed);
            let counts: Vec<f64> = traj.substeps.iter().map(|s| orng.neg_binomial_dispersion(0.5 * s.flows[prog] as f64, 10.0) as f64).collect();
            // `$CAMDL_RUNG_DATA_DUMP`: write the counts as the `fit run` diagnosis's
            // onsets TSV, to compare the two datasets byte for byte.
            if let Ok(path) = std::env::var("CAMDL_RUNG_DATA_DUMP") {
                let mut body = String::from("time\tonsets\n");
                for (i, c) in counts.iter().enumerate() {
                    body.push_str(&format!("{}\t{c}\n", i + 1));
                }
                std::fs::write(path, body).unwrap();
            }
            ((1..=counts.len()).map(|t| t as f64).collect(), dense_cells(counts))
        }
    };
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, &model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.clone()).unwrap(),
            projection, ir_model: om.clone(), observations: cells.clone(), aux: vec![],
        }
    }).collect();
    let obs = MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap();
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
        n_particles: if data.is_some() { 64 } else { 16 }, n_sweeps: 5000, burn_in: 1000, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 8,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };
    type Out = (Vec<Vec<f64>>, usize, f64, f64, f64);
    let runs: Vec<Out> = std::thread::scope(|sc| {
        let hs: Vec<_> = (1..=4u64).map(|seed| {
            let (model, est, priors, observations, obs, config) = (&model, &est, &priors, &observations, &obs, &config);
            sc.spawn(move || {
                let mut base = model.default_params.clone();
                for p in est { base[p.index] = p.initial; }
                let r = run_pgas(model, est, priors, &base, config, observations, obs, seed, None, None, "nuts-rungs".into())
                    .expect("PGAS run");
                let n = r.sweeps.len() as f64;
                (r.sweeps.iter().map(|s| est.iter().map(|p| s.params[p.index]).collect()).collect(),
                 r.n_divergent_post_burn,
                 r.sweeps.last().unwrap().nuts.step_size,
                 r.sweeps.iter().map(|s| s.nuts.n_leapfrog as f64).sum::<f64>() / n,
                 r.sweeps.iter().map(|s| s.nuts.accept_stat).sum::<f64>() / n)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let chains: Vec<Vec<Vec<f64>>> = runs.iter().map(|r| r.0.clone()).collect();
    println!("{label}\tdiv {:?}\tε {}\tleapfrog {:.1}\tminESS {:.0}\tmean accept {:.3}",
        runs.iter().map(|r| r.1).collect::<Vec<_>>(),
        runs.iter().map(|r| format!("{:.3}", r.2)).collect::<Vec<_>>().join("/"),
        runs.iter().map(|r| r.3).sum::<f64>() / 4.0, min_ess(&chains),
        runs.iter().map(|r| r.4).sum::<f64>() / 4.0);
}

#[test]
#[ignore = "diagnostic; several minutes in release"]
fn rungs_2_and_4_through_pgas() {
    pgas_rung("rung2 scalars, τ box, X0=0", "tests/fixtures/prior_only/prior_only_all.camdl", &[], None);
    pgas_rung("rung2 scalars, τ unboxed, X0=0", "tests/fixtures/prior_only/prior_only_all_unboxed.camdl", &[], None);
    pgas_rung("rung4a smooth, τ box, X0=0", "tests/fixtures/prior_only/smooth_x0_boxed.camdl", &[], None);
    pgas_rung("rung4a smooth, τ unboxed, X0=0", "tests/fixtures/bspline_inference/prior_only_smooth.camdl", &[], None);
    pgas_rung("rung4b smooth, τ unboxed, X0=1e4", "tests/fixtures/prior_only/smooth_x0_10k_unboxed.camdl", &[], None);
    let fixed = [("sigma", 0.2), ("gamma", 0.2), ("rho", 0.5), ("k", 10.0), ("N0", 1e6), ("E0", 100.0), ("I0", 100.0)];
    pgas_rung("rung4c SEIR data seed 31, τ box", "tests/fixtures/prior_only/seir_smooth_boxed.camdl", &fixed,
        Some(("tests/fixtures/bspline_inference/seir_bspline_recovery.camdl", 31)));
    pgas_rung("rung4c SEIR data seed 31, τ unboxed", "tests/fixtures/bspline_inference/seir_smooth_recovery.camdl", &fixed,
        Some(("tests/fixtures/bspline_inference/seir_bspline_recovery.camdl", 31)));
}

/// Rung 4c's dataset only (seed 31), for `$CAMDL_RUNG_DATA_DUMP`.
#[test]
#[ignore = "diagnostic; about two minutes in release"]
fn rung4c_seed_31_alone() {
    let fixed = [("sigma", 0.2), ("gamma", 0.2), ("rho", 0.5), ("k", 10.0), ("N0", 1e6), ("E0", 100.0), ("I0", 100.0)];
    pgas_rung("rung4c SEIR data seed 31, τ unboxed", "tests/fixtures/bspline_inference/seir_smooth_recovery.camdl", &fixed,
        Some(("tests/fixtures/bspline_inference/seir_bspline_recovery.camdl", 31)));
}
