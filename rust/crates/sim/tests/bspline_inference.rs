//! R3 of proposal 2026-10-27-estimable-bsplines §8.1: spline coefficients
//! through PGAS with NUTS, on chain-binomial SEIR epidemics whose transmission
//! rate is β(t) = exp(Σ cₖ Bₖ(t)) (a log-link `bspline`).
//!
//! - `small_epidemic_runs_finite_and_reproducibly`: N0 = 10⁴, 60 days, six
//!   coefficients, daily Poisson onsets; 2 chains × 200 sweeps with a dense
//!   mass matrix. Every recorded log-likelihood and the gradient at each
//!   chain's last draw are finite, and a second run at the same seed is
//!   bitwise identical.
//! - `recovers_a_spline_transmission_rate`: N0 = 10⁶, 100 days, five
//!   coefficients, daily negative-binomial onsets (r = 10), truth in the spline
//!   space, E0, I0, rho and k fixed; 4 chains × 3000 sweeps. Every true coefficient lies in
//!   its central 99% posterior interval, and the posterior median of log β(t)
//!   is within 0.2 of the truth at every t in the middle 80% of the window.
//!
//! The coefficients are explicit parameters with normal(−1.5, 1) priors: the
//! `smooth` prior the proposal's version of these tests uses is not built yet,
//! and its prior-only check is a named placeholder in `bspline_harness.rs`.

use std::sync::Arc;

use sim::bspline::{span, Knots};
use sim::compiled_model::CompiledModel;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{build_obs_at_substep, run_pgas, simulate_reference, PGASConfig, PGASResult};
use sim::inference::pgas_grad::complete_data_loglik_grad;
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};
use sim::rng::StatefulRng;

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

/// Daily onsets simulated at the truth, observed as Poisson (`k = None`) or
/// negative-binomial with dispersion `k` around `rho * onsets`.
fn synthesize(model: &CompiledModel, rho: f64, k: Option<f64>, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StatefulRng::new(seed);
    let traj = simulate_reference(model, &model.default_params, model.model.simulation.t_end, 1.0, &mut rng).unwrap();
    let prog = model.model.transitions.iter().position(|t| t.name == "progression").unwrap();
    let mut obs_rng = StatefulRng::new(seed ^ 0x5eed);
    let (mut times, mut counts) = (Vec::new(), Vec::new());
    for (s, sub) in traj.substeps.iter().enumerate() {
        let mean = rho * sub.flows[prog] as f64;
        let y = match k {
            None => obs_rng.poisson(mean),
            Some(k) => obs_rng.neg_binomial_dispersion(mean, k),
        };
        times.push((s + 1) as f64);
        counts.push(y as f64);
    }
    (times, counts)
}

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

struct Fit {
    model: Arc<CompiledModel>,
    obs_model: MultiStreamObsModel,
    observations: Vec<Observation>,
    est: Vec<EstimatedParam>,
    priors: Vec<Prior>,
}

impl Fit {
    fn new(model: Arc<CompiledModel>, times: &[f64], counts: &[f64], coefs: &[&str], start: f64) -> Self {
        let obs_model = observed(&model, times, counts);
        let observations = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
        let est = coefs.iter().map(|n| EstimatedParam {
            name: (*n).into(),
            index: model.param_index[*n],
            initial: start,
            rw_sd: 0.05,
            transform: Transform::None,
            lower: -8.0,
            upper: 2.0,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }).collect::<Vec<_>>();
        let priors = vec![Prior::Fixed(Density::Normal { mean: -1.5, sd: 1.0 }); est.len()];
        Fit { model, obs_model, observations, est, priors }
    }

    fn run(&self, config: &PGASConfig, seed: u64) -> PGASResult {
        let mut base = self.model.default_params.clone();
        for p in &self.est {
            base[p.index] = p.initial;
        }
        run_pgas(&self.model, &self.est, &self.priors, &base, config, &self.observations,
                 &self.obs_model, seed, None, None, "bspline-r3".into())
            .expect("PGAS-NUTS run")
    }
}

fn nuts_config(n_particles: usize, n_sweeps: usize, burn_in: usize) -> PGASConfig {
    PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles, n_sweeps, burn_in, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 6,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    }
}

const SMALL_TRUTH: [f64; 6] = [-0.6, -0.5, -0.9, -1.5, -1.8, -1.8];

#[test]
#[ignore = "multi-minute in a debug build; run in release by `make validate-bspline`"]
fn small_epidemic_runs_finite_and_reproducibly() {
    let coefs = ["b1", "b2", "b3", "b4", "b5", "b6"];
    let mut values: Vec<(&str, f64)> = vec![
        ("sigma", 0.25), ("gamma", 0.2), ("rho", 0.5), ("N0", 10_000.0), ("E0", 5.0), ("I0", 5.0),
    ];
    values.extend(coefs.iter().copied().zip(SMALL_TRUTH));
    let model = load("seir_bspline_small", &values);
    let (times, counts) = synthesize(&model, 0.5, None, 11);
    assert!(counts.iter().sum::<f64>() > 200.0, "the epidemic must take off: {counts:?}");
    let fit = Fit::new(model.clone(), &times, &counts, &coefs, -1.5);
    let config = nuts_config(64, 200, 50);

    for seed in [1, 2] {
        let a = fit.run(&config, seed);
        assert!(a.sweeps.iter().all(|s| s.log_complete_data_ll.is_finite()
            && s.params.iter().all(|v| v.is_finite())), "chain {seed}: a non-finite sweep");
        // The gradient at the chain's last draw, on its last trajectory.
        // A sweep records the full model parameter vector.
        let theta = a.sweeps.last().unwrap().params.clone();
        let n = theta.len();
        let model_to_est: Vec<Option<usize>> = (0..n).map(Some).collect();
        let rate_grads = sim::inference::pgas_grad::resolve_rate_grad_for_run(
            &model.resolved.rate_grads_indexed, &model_to_est);
        let oas = build_obs_at_substep(&fit.observations, 0.0, 1.0).unwrap();
        let (ll, grad) = complete_data_loglik_grad(&model, &a.final_trajectory, &theta, &fit.observations,
            1.0, &fit.obs_model, n, &rate_grads, &oas, &(0..n).collect::<Vec<_>>()).unwrap();
        assert!(ll.is_finite() && grad.iter().all(|g| g.is_finite()), "chain {seed}: gradient {grad:?}");

        let b = fit.run(&config, seed);
        assert_eq!(a.sweeps.len(), b.sweeps.len());
        for (x, y) in a.sweeps.iter().zip(&b.sweeps) {
            let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x.params), bits(&y.params), "chain {seed}, sweep {}: draws differ across runs", x.sweep);
            assert_eq!(x.log_complete_data_ll.to_bits(), y.log_complete_data_ll.to_bits());
        }
        eprintln!("chain {seed}: {} sweeps, {} divergent", a.sweeps.len(), a.n_divergent_total);
    }
}

const RECOVERY_TRUTH: [f64; 5] = [-0.7, -1.0, -1.6, -1.4, -1.9];

/// log β(t) = Σ cₖ Bₖ(t) over the fixture's window [0, 86], five cubic basis
/// functions (2 equal intervals).
fn log_beta(c: &[f64], t: f64) -> f64 {
    let knots = Knots::uniform(0.0, 86.0, 2).unwrap();
    let s = span(&knots, 3, t);
    (0..=3).map(|r| s.values[r] * c[s.first + r]).sum()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Data seed and length are deliberate. PGAS mixes slowly along the strongly
/// correlated early coefficients (c1–c2 posterior correlation ≈ −0.9): at 800
/// sweeps from a common start, two of four data seeds put c2's truth at or
/// outside its 99% interval, and starting the chains at the truth moved the
/// posterior toward it — under-convergence, not a wrong target (R2 pins the
/// gradient to the likelihood). At 3000 sweeps seeds 31 (used here) and 59
/// pass with margin (worst curve error 0.111, 0.075) and seed 23 at the edge
/// (c2's truth at its 0.6% quantile, curve error 0.186) — the converged
/// posterior for that realization; seed 47 passed already at 800 (0.090).
#[test]
#[ignore = "about eight minutes in release; run by `make validate-bspline`"]
fn recovers_a_spline_transmission_rate() {
    let coefs = ["b1", "b2", "b3", "b4", "b5"];
    let mut values: Vec<(&str, f64)> = vec![
        ("sigma", 0.2), ("gamma", 0.2), ("rho", 0.5), ("k", 10.0),
        ("N0", 1_000_000.0), ("E0", 100.0), ("I0", 100.0),
    ];
    values.extend(coefs.iter().copied().zip(RECOVERY_TRUTH));
    let model = load("seir_bspline_recovery", &values);
    let (times, counts) = synthesize(&model, 0.5, Some(10.0), 31);
    eprintln!("onsets: total {}, peak {}", counts.iter().sum::<f64>(), counts.iter().fold(0.0_f64, |m, c| m.max(*c)));
    let fit = Fit::new(model.clone(), &times, &counts, &coefs, -1.5);
    let config = nuts_config(64, 3000, 1500);

    let mut draws: Vec<Vec<f64>> = Vec::new();
    for seed in 1..=4 {
        let r = fit.run(&config, seed);
        // A sweep records the full parameter vector; keep the coefficients.
        draws.extend(r.sweeps.iter().map(|s| fit.est.iter().map(|p| s.params[p.index]).collect::<Vec<_>>()));

        eprintln!("chain {seed}: {} kept, {} divergent", r.sweeps.len(), r.n_divergent_total);
    }

    for (j, &truth) in RECOVERY_TRUTH.iter().enumerate() {
        let mut v: Vec<f64> = draws.iter().map(|d| d[j]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (lo, med, hi) = (quantile(&v, 0.005), quantile(&v, 0.5), quantile(&v, 0.995));
        eprintln!("  c{}: truth {truth:+.3}, median {med:+.3}, 99% [{lo:+.3}, {hi:+.3}]", j + 1);
        assert!(lo <= truth && truth <= hi, "c{}: truth {truth} outside its 99% interval [{lo}, {hi}]", j + 1);
    }
    let mut worst: f64 = 0.0;
    for i in 0..=40 {
        let t = 0.1 * 86.0 + 0.8 * 86.0 * i as f64 / 40.0;
        let mut g: Vec<f64> = draws.iter().map(|d| log_beta(d, t)).collect();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let err = (quantile(&g, 0.5) - log_beta(&RECOVERY_TRUTH, t)).abs();
        worst = worst.max(err);
        assert!(err <= 0.2, "t = {t:.1}: posterior median log β is {err:.3} from the truth");
    }
    eprintln!("recovery: worst |median log β − truth| over the middle 80% = {worst:.3}");
}
