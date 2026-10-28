//! Experimental: a small simulation-based calibration (SBC; Talts et al.
//! 2018) of PGAS+NUTS on a `smooth` spline (proposal
//! 2026-10-27-estimable-bsplines §8.1, R5). A diagnostic, not a merge gate:
//! it is `#[ignore]`d and run by neither `make test` nor `make
//! validate-bspline`.
//!
//! `sbc_smooth` is X --> D @ f·X with X0 = 10⁵ and daily Poisson removal
//! counts; f is a log-link K = 6 smooth spline. For each of 100 draws θ̃ from
//! the prior it simulates a dataset, fits it (1 chain, 500 adaptation + 990
//! kept sweeps, thinned to 99 draws) and records the rank of θ̃ among the 99
//! draws for a, b, τ, each zᵢ and log f at t = 10, 30, 50. Under a correct
//! sampler each rank is uniform on {0, …, 99}; the test prints, per
//! quantity, a χ² statistic over 10 rank bins (9 degrees of freedom; 21.7 is
//! the 1% point) and the counts.
//!
//! cargo test --release -p sim --test smooth_sbc -- --ignored --nocapture

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};
use sim::resolved_expr::{eval_resolved, ResolvedExpr};
use sim::rng::StatefulRng;

const N_SIMS: usize = 100;
const LAMBDA: f64 = 5.99;

fn load() -> Arc<CompiledModel> {
    let path = format!("{}/../../../tests/fixtures/bspline_inference/ir/sbc_smooth.ir.json", env!("CARGO_MANIFEST_DIR"));
    let mut m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for p in &mut m.parameters {
        p.value = p.value.with_value(if p.name == "tau" { 0.2 } else if p.name == "a" { -3.0 } else { 0.0 });
    }
    Arc::new(CompiledModel::new(m).unwrap())
}

fn log_f(model: &CompiledModel, params: &[f64], t: f64) -> f64 {
    let (int_s, real_s) = model.initial_state_mean(params).unwrap();
    let ctx = sim::propensity::EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt: 1.0,
        projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    eval_resolved(&ResolvedExpr::TimeFunc(0), &ctx).ln()
}

/// The quantities ranked: a, b, τ, z₁…z₄, log f(10), log f(30), log f(50).
fn quantities(model: &CompiledModel, params: &[f64]) -> Vec<f64> {
    let ix = |n: &str| params[model.param_index[n]];
    let mut q = vec![ix("a"), ix("b"), ix("tau")];
    q.extend((1..=4).map(|i| ix(&format!("z_m{i}"))));
    q.extend([10.0, 30.0, 50.0].map(|t| log_f(model, params, t)));
    q
}

fn one(model: &Arc<CompiledModel>, sim_id: usize) -> Vec<usize> {
    let mut rng = StatefulRng::new(7_000 + sim_id as u64);
    let mut truth = model.default_params.clone();
    let set = |truth: &mut Vec<f64>, n: &str, v: f64| truth[model.param_index[n]] = v;
    set(&mut truth, "a", -3.0 + 0.3 * rng.normal());
    set(&mut truth, "b", 0.005 * rng.normal());
    set(&mut truth, "tau", -(1.0 - rng.uniform()).ln() / LAMBDA);
    for i in 1..=4 {
        set(&mut truth, &format!("z_m{i}"), rng.normal());
    }
    let traj = simulate_reference(model, &truth, 60.0, 1.0, &mut rng).unwrap();
    let decay = model.model.transitions.iter().position(|t| t.name == "decay").unwrap();
    let times: Vec<f64> = (1..=traj.substeps.len()).map(|s| s as f64).collect();
    let counts: Vec<f64> = traj.substeps.iter().map(|s| rng.poisson(s.flows[decay] as f64) as f64).collect();

    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.clone()).unwrap(),
            projection, ir_model: om.clone(), observations: dense_cells(counts.clone()), aux: vec![],
        }
    }).collect();
    let obs_model = MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let inf = f64::INFINITY;
    let ep = |name: &str, transform: Transform, lo: f64, hi: f64, initial: f64| EstimatedParam {
        name: name.into(), index: model.param_index[name], initial, rw_sd: 0.05, transform,
        lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
    };
    let mut est = vec![
        ep("a", Transform::None, -inf, inf, -3.0),
        ep("b", Transform::None, -inf, inf, 0.0),
        ep("tau", Transform::Log { lo: 1e-6, hi: 2.5 }, 1e-6, 2.5, 0.2),
    ];
    let mut priors = vec![
        Prior::Fixed(Density::Normal { mean: -3.0, sd: 0.3 }),
        Prior::Fixed(Density::Normal { mean: 0.0, sd: 0.005 }),
        Prior::Fixed(Density::Exponential { rate: LAMBDA }),
    ];
    for i in 1..=4 {
        est.push(ep(&format!("z_m{i}"), Transform::None, -inf, inf, 0.0));
        priors.push(Prior::Fixed(Density::Normal { mean: 0.0, sd: 1.0 }));
    }
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
        n_particles: 32, n_sweeps: 1490, burn_in: 500, thin: 10, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 8,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };
    let mut base = model.default_params.clone();
    for p in &est {
        base[p.index] = p.initial;
    }
    let r = run_pgas(model, &est, &priors, &base, &config, &observations, &obs_model, 100 + sim_id as u64,
        None, None, "smooth-sbc".into()).expect("SBC fit");
    let draws: Vec<Vec<f64>> = r.sweeps.iter().take(99).map(|s| quantities(model, &s.params)).collect();
    assert_eq!(draws.len(), 99, "99 thinned draws");
    let t = quantities(model, &truth);
    (0..t.len()).map(|q| draws.iter().filter(|d| d[q] < t[q]).count()).collect()
}

#[test]
#[ignore = "experimental diagnostic; minutes in release; never run by make"]
fn sbc_ranks_of_the_smooth_prior() {
    let model = load();
    let ranks: Vec<Vec<usize>> = std::thread::scope(|sc| {
        let model = &model;
        let hs: Vec<_> = (0..8).map(|w| sc.spawn(move || {
            (0..N_SIMS).filter(|i| i % 8 == w).map(|i| (i, one(model, i))).collect::<Vec<_>>()
        })).collect();
        let mut all: Vec<(usize, Vec<usize>)> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
        all.sort_by_key(|(i, _)| *i);
        all.into_iter().map(|(_, r)| r).collect()
    });
    let names = ["a", "b", "tau", "z1", "z2", "z3", "z4", "logf(10)", "logf(30)", "logf(50)"];
    eprintln!("quantity\tchi2_9df\tbin_counts");
    for (q, name) in names.iter().enumerate() {
        let mut bins = [0usize; 10];
        for r in &ranks {
            bins[(r[q] / 10).min(9)] += 1;
        }
        let e = N_SIMS as f64 / 10.0;
        let chi2: f64 = bins.iter().map(|&c| (c as f64 - e).powi(2) / e).sum();
        eprintln!("{name}\t{chi2:.1}\t{bins:?}");
    }
}
