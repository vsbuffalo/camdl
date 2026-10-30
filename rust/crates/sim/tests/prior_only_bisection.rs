//! Diagnostic, not a check: where do PGAS+NUTS divergences on a pure prior
//! come from? Every model here has X0 = 0 and one all-missing observation
//! stream, so the complete-data likelihood is constant and the target is the
//! prior alone: independent normals on the identity scale and an exponential
//! on the log scale, which NUTS should sample with essentially no
//! divergences.
//!
//! - (i) `prior_only_all`: a, b, τ, z1..z4 as scalars, no spline;
//! - (ii) `prior_only_tau`: τ ~ exponential(5.99) alone, log transform, on
//!   its box [1e-6, 2.5]; and `prior_only_tau_unboxed`, the same with no box;
//! - (iii) `prior_only_z`: z1..z4 ~ N(0, 1) alone, identity;
//! - (iv) `prior_only_smooth`: the smooth spline fixture (skipped where the
//!   compiler does not know `smooth`).
//!
//! Each is compiled at run time by `$CAMDLC` (default: this checkout's
//! camdlc), so the same file runs on a checkout without the spline work.
//! 4 chains × (1,000 adaptation + 4,000 kept) sweeps, 4 particles, dense
//! mass, max tree depth 8. Prints per chain the post-adaptation divergences,
//! the adapted step size and the mean leapfrog steps; and, for each model,
//! the mean and quantiles of every parameter (log τ for τ, and |z| for the
//! modes) at the start of the divergent transitions against all kept sweeps.
//!
//! CAMDLC=… cargo test --release -p sim --test prior_only_bisection -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::inference::{BoundObs, MultiStreamObsModel};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn compile(fixture: &str) -> Option<ir::Model> {
    let camdlc = std::env::var("CAMDLC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo().join("ocaml/_build/default/bin/camdlc.exe"));
    let dir = tempfile_dir();
    let out = dir.join(format!("{}.ir.json", fixture.replace('/', "_")));
    let o = std::process::Command::new(&camdlc).arg(repo().join(fixture)).arg("-o").arg(&out).output().unwrap();
    if !o.status.success() {
        eprintln!("{fixture}: does not compile here, skipped");
        return None;
    }
    Some(ir::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap())
}

fn tempfile_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("prior_only_bisection_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Chain {
    params: Vec<Vec<f64>>, // kept sweeps, estimated parameters
    divergent: Vec<bool>,
    step_size: f64,
    leapfrog: f64,
}

fn run(fixture: &str) -> Option<(Vec<String>, Vec<Chain>)> {
    let mut m = compile(fixture)?;
    let mut names = Vec::new();
    for p in &mut m.parameters {
        if let Some(ir::parameter::PriorSpec::Dist(_)) = p.value.prior() {
            names.push(p.name.clone());
        }
        let v = if p.name == "tau" { 0.2 } else { 0.0 };
        p.value = p.value.with_value(v);
    }
    let priors_ir: Vec<_> = names.iter().map(|n| {
        let p = m.parameters.iter().find(|p| &p.name == n).unwrap();
        let Some(ir::parameter::PriorSpec::Dist(d)) = p.value.prior() else { unreachable!() };
        (Prior::from_ir(d), p.value.bounds(), p.param_kind)
    }).collect();
    let model = Arc::new(CompiledModel::new(m).unwrap());
    let inf = f64::INFINITY;
    let est: Vec<EstimatedParam> = names.iter().zip(&priors_ir).map(|(n, (_, bounds, kind))| {
        let positive = matches!(kind, Some(ir::parameter::ParamKind::Positive));
        let (lo, hi) = bounds.unwrap_or(if positive { (0.0, inf) } else { (-inf, inf) });
        EstimatedParam {
            name: n.clone(), index: model.param_index[n.as_str()],
            initial: if n == "tau" { 0.2 } else { 0.0 }, rw_sd: 0.05,
            transform: if positive { Transform::Log { lo, hi } } else { Transform::None },
            lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
        }
    }).collect();
    let priors: Vec<Prior> = priors_ir.into_iter().map(|(p, _, _)| p).collect();
    let times: Vec<f64> = (1..=60).map(|t| t as f64).collect();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, &model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.clone()).unwrap(),
            projection, ir_model: om.clone(), observations: vec![None; times.len()], aux: vec![],
        }
    }).collect();
    let obs = MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap();
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
        n_particles: 4, n_sweeps: 5000, burn_in: 1000, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 8,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };
    let chains: Vec<Chain> = std::thread::scope(|sc| {
        let hs: Vec<_> = (1..=4u64).map(|seed| {
            let (model, est, priors, observations, obs, config) = (&model, &est, &priors, &observations, &obs, &config);
            sc.spawn(move || {
                let mut base = model.default_params.clone();
                for p in est {
                    base[p.index] = p.initial;
                }
                let r = run_pgas(model, est, priors, &base, config, observations, obs, seed, None, None,
                    "prior-only-bisection".into()).expect("PGAS run");
                Chain {
                    params: r.sweeps.iter().map(|s| est.iter().map(|p| s.params[p.index]).collect()).collect(),
                    divergent: r.sweeps.iter().map(|s| s.nuts.n_divergent > 0).collect(),
                    step_size: r.sweeps.last().unwrap().nuts.step_size,
                    leapfrog: r.sweeps.iter().map(|s| s.nuts.n_leapfrog as f64).sum::<f64>() / r.sweeps.len() as f64,
                }
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    Some((names, chains))
}

fn quantiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("mean {:.3} [q05 {:.3}, q50 {:.3}, q95 {:.3}]", v.iter().sum::<f64>() / v.len() as f64, q(0.05), q(0.5), q(0.95))
}

#[test]
#[ignore = "diagnostic; about two minutes in release"]
fn prior_only_divergences_by_model() {
    for fixture in [
        "tests/fixtures/prior_only/prior_only_all.camdl",
        "tests/fixtures/prior_only/prior_only_tau.camdl",
        "tests/fixtures/prior_only/prior_only_tau_unboxed.camdl",
        "tests/fixtures/prior_only/prior_only_z.camdl",
        "tests/fixtures/bspline_inference/prior_only_smooth.camdl",
    ] {
        let Some((names, chains)) = run(fixture) else { continue };
        println!("== {fixture}");
        for (i, c) in chains.iter().enumerate() {
            println!("  chain {}: {} divergent of {} kept, step size {:.4}, mean leapfrog {:.1}",
                i + 1, c.divergent.iter().filter(|d| **d).count(), c.divergent.len(), c.step_size, c.leapfrog);
        }
        // A sweep's recorded parameters are the state after its NUTS step,
        // so a divergent transition starts from the previous sweep's state.
        let mut starts: Vec<&Vec<f64>> = Vec::new();
        let mut all: Vec<&Vec<f64>> = Vec::new();
        for c in &chains {
            for s in 1..c.params.len() {
                all.push(&c.params[s - 1]);
                if c.divergent[s] {
                    starts.push(&c.params[s - 1]);
                }
            }
        }
        let zs: Vec<usize> = names.iter().enumerate().filter(|(_, n)| n.starts_with('z')).map(|(i, _)| i).collect();
        let feature = |label: &str, f: &dyn Fn(&[f64]) -> f64| {
            println!("  {label:6} divergent-start {} | all {}",
                quantiles(starts.iter().map(|v| f(v)).collect()), quantiles(all.iter().map(|v| f(v)).collect()));
        };
        for (i, n) in names.iter().enumerate() {
            if n == "tau" {
                feature("log τ", &|v| v[i].ln());
            } else if !n.starts_with('z') {
                feature(n, &|v| v[i]);
            }
        }
        if !zs.is_empty() {
            feature("|z|", &|v| zs.iter().map(|&i| v[i] * v[i]).sum::<f64>().sqrt());
        }
    }
}

/// The NUTS kernel alone (no PGAS) on the target of `prior_only_tau`:
/// u = log τ, log p(u) = u − λeᵘ, optionally with the box's walls (−∞ outside
/// [ln 1e-6, ln 2.5], as the `Log { lo, hi }` transform scores it), at fixed
/// step sizes and unit mass. Prints divergences per 20,000 transitions.
#[test]
#[ignore = "diagnostic; seconds in release"]
fn nuts_kernel_alone_on_the_tau_target() {
    use sim::inference::nuts::{nuts_step, MassMatrix, NUTSConfig};
    let lambda = 5.99;
    for (label, walls) in [("unboxed", None), ("box [1e-6, 2.5]", Some(((1e-6_f64).ln(), (2.5_f64).ln())))] {
        for eps in [0.5, 0.8, 1.0, 1.2] {
            let lp = move |u: &[f64]| -> (f64, Vec<f64>) {
                if let Some((lo, hi)) = walls {
                    if u[0] < lo || u[0] > hi {
                        return (f64::NEG_INFINITY, vec![0.0]);
                    }
                }
                (u[0] - lambda * u[0].exp(), vec![1.0 - lambda * u[0].exp()])
            };
            let cfg = NUTSConfig { max_tree_depth: 8, step_size: eps, mass_matrix: MassMatrix::identity(1) };
            let mut rng = sim::rng::StatefulRng::new(5);
            let mut u = vec![-2.0];
            let (mut l, mut g) = lp(&u);
            let (mut div, mut lf) = (0, 0);
            for _ in 0..20_000 {
                let r = nuts_step(&u, l, &g, &cfg, &lp, &mut rng);
                div += r.divergent as usize;
                lf += r.n_leapfrog;
                u = r.params;
                (l, g) = lp(&u);
            }
            println!("  kernel {label:16} ε = {eps}: {div} divergent / 20000, mean leapfrog {:.1}", lf as f64 / 20_000.0);
        }
    }
}
