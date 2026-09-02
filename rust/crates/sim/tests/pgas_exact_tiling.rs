//! Exact-PGAS producer tiling — Stage 3 (2c) behavior gates.
//!
//! Commits A–C built the machinery (`build_substep_grid`, the grid-driven
//! `simulate_reference_on_grid`, csmc reading the grid off the reference, the
//! relaxed exact-tiling invariant) and proved every snap path stays
//! byte-identical. These gates exercise the NEW behavior: a genuinely shortened
//! (off-grid) substep, which only exists once the exact producer runs.
//!
//! Three arms:
//!   (a) ON-GRID BYTE-PARITY — on an all-on-grid model the exact producer and the
//!       snap producer emit a bit-identical trajectory. The behavior change must
//!       be invisible where snap and exact coincide.
//!   (c) SHORTENED-SUBSTEP VALUE RECOMPUTE — off-grid obs produce `dt_substep ≠ dt`
//!       (non-vacuity); an INDEPENDENT recompute of the transition density from
//!       the records (`rec.t0`, `rec.dt_substep`) equals what `complete_data_loglik`
//!       computes (consumer reads the records), and DIFFERS from a uniform `s·dt`
//!       reconstruction (the records are load-bearing — catches a producer that
//!       silently wrote uniform values, which `s·dt` reconstruction can't see).
//!   (b) SHORTENED-SUBSTEP GRADIENT — cross-function FD: the value
//!       (`complete_data_loglik`) vs the analytic gradient
//!       (`complete_data_loglik_grad`) agree on a trajectory whose substeps are
//!       genuinely shortened, so value and gradient reconstruct the SAME realized
//!       `(t0, dt_substep)`. This is the seasonal/magnitude oracle arm made
//!       genuine — the prior arms ran on a uniform (non-shortened) grid.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{
    build_substep_grid, complete_data_loglik, log_transition_density_substep,
    simulate_reference, simulate_reference_on_grid, ObsAtSubstep, PGASTrajectory,
};
use sim::inference::pgas_grad::{complete_data_loglik_grad, resolve_rate_grad_for_run};
use sim::inference::particle_filter::Observation;
use sim::inference::MultiStreamObsModel;
use sim::rng::StatefulRng;
use sim::schedule::StepPolicy;

const SEED: u64 = 42;

/// Compile `seir_vaccine_seasonal` with the canonical params. Time-inhomogeneous
/// (β·seasonal(t)) so the realized `t0` is load-bearing; carries `rate_grad` for
/// the gradient arm. Its scheduled SIA intervention is non-`always_active`, so
/// the chain-binomial `step_one` producer path never applies it — the dynamics
/// here are the pure seasonal SEIR, deterministic given the seed.
fn seir_seasonal() -> (Arc<CompiledModel>, Vec<f64>) {
    let json = std::fs::read_to_string("../../../ocaml/golden/seir_vaccine_seasonal.ir.json")
        .expect("read seir_vaccine_seasonal IR");
    let mut model = ir::from_str(&json).expect("parse seir_vaccine_seasonal IR");
    assert!(
        model.transitions.iter().any(|t| !t.rate_grad.is_empty()),
        "seasonal model must carry rate_grad (run make update-golden)"
    );
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.3, "sigma" => 0.2, "gamma" => 0.1,
                "omega" => 0.003, "reversion_rate" => 1e-6,
                "alpha" => 0.15, "phi_season" => 90.0,
                "vacc_frac" => 0.8, "N0" => 1_000_000.0, "I0" => 10.0,
                _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).expect("compile seasonal"));
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
    }
    (compiled, params)
}

fn obs_at(times: &[f64]) -> Vec<Observation> {
    times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect()
}

fn trajectories_bit_identical(a: &PGASTrajectory, b: &PGASTrajectory) -> bool {
    if a.initial_counts != b.initial_counts || a.substeps.len() != b.substeps.len() {
        return false;
    }
    a.substeps.iter().zip(&b.substeps).all(|(x, y)| {
        x.counts_before == y.counts_before
            && x.counts_after == y.counts_after
            && x.flows == y.flows
            && x.gammas.iter().zip(&y.gammas).all(|(&p, &q)| p.to_bits() == q.to_bits())
            && x.gammas.len() == y.gammas.len()
            && x.t0.to_bits() == y.t0.to_bits()
            && x.dt_substep.to_bits() == y.dt_substep.to_bits()
    })
}

/// (a) On an all-on-grid model the exact producer is byte-identical to the snap
/// producer: same grid (proven in the keystone unit tests) → same RNG draw order
/// → same trajectory, bit-for-bit.
#[test]
fn exact_equals_snap_on_grid() {
    let (compiled, params) = seir_seasonal();
    let t_start = compiled.model.simulation.t_start;
    let dt = 1.0;
    // On-grid obs (integer multiples of dt from t_start).
    let observations = obs_at(&[50.0, 100.0, 150.0, 200.0]);
    let last_obs = 200.0;

    let mut rng_snap = StatefulRng::new(SEED);
    let snap = simulate_reference(&compiled, &params, last_obs, dt, sim::rng::BinomialAlgorithm::default(), &mut rng_snap).unwrap();

    let grid = build_substep_grid(t_start, dt, &observations, &[], StepPolicy::Exact).unwrap();
    let mut rng_exact = StatefulRng::new(SEED);
    let exact = simulate_reference_on_grid(&compiled, &params, dt, &grid.steps, None, sim::rng::BinomialAlgorithm::default(), &mut rng_exact).unwrap();

    assert!(
        trajectories_bit_identical(&snap, &exact),
        "exact-on-grid trajectory must be byte-identical to snap"
    );
    // Sanity: no shortened substep on an on-grid model.
    assert!(exact.substeps.iter().all(|r| r.dt_substep.to_bits() == dt.to_bits()),
        "on-grid model must have no shortened substeps");
}

/// (c) Off-grid obs produce genuinely shortened substeps; the recorded
/// (t0, dt_substep) are load-bearing and the consumer reads them.
///
/// The observations exist only to carve the off-grid windows that produce
/// shortened substeps; the transition-density recompute is obs-independent, so
/// `complete_data_loglik` is called with no observations (no obs-scoring). Obs→
/// substep mapping correctness is pinned separately in the keystone unit tests.
#[test]
fn exact_shortened_substep_density_recompute() {
    let (compiled, params) = seir_seasonal();
    let t_start = compiled.model.simulation.t_start;
    let dt = 1.0;
    // Off-grid obs with VARIED fractional gaps, so each window is off-grid
    // relative to the previous re-anchor (equal gaps would re-align after the
    // first remainder and produce only one shortened substep).
    let observations = obs_at(&[40.5, 90.2, 140.8, 190.3]);

    let grid = build_substep_grid(t_start, dt, &observations, &[], StepPolicy::Exact).unwrap();
    let mut rng = StatefulRng::new(SEED);
    let traj = simulate_reference_on_grid(&compiled, &params, dt, &grid.steps, None, sim::rng::BinomialAlgorithm::default(), &mut rng).unwrap();

    // Non-vacuity: genuinely shortened substeps exist, each in (0, dt).
    let n_short = traj.substeps.iter().filter(|r| r.dt_substep != dt).count();
    eprintln!("  shortened substeps: {n_short} / {}", traj.substeps.len());
    assert!(n_short >= 1, "off-grid obs must produce at least one shortened substep");
    for r in traj.substeps.iter().filter(|r| r.dt_substep != dt) {
        assert!(r.dt_substep > 0.0 && r.dt_substep < dt,
            "shortened substep dt_substep {} must be in (0, dt)", r.dt_substep);
    }

    // Independent recompute of the transition density from the records.
    let recompute = |use_uniform: bool| -> f64 {
        let mut total = 0.0;
        for (s, rec) in traj.substeps.iter().enumerate() {
            let (t, dt_s) = if use_uniform {
                (t_start + s as f64 * dt, dt) // the buggy uniform reconstruction
            } else {
                (rec.t0, rec.dt_substep) // the realized grid
            };
            total += log_transition_density_substep(
                &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt_s, None,
            ).expect("finite per-substep density");
        }
        total
    };
    let d_exact = recompute(false);
    let d_uniform = recompute(true);
    assert!(d_exact.is_finite(), "recomputed exact density must be finite");

    // The consumer (complete_data_loglik) reads rec.(t0, dt_substep): its
    // transition component must equal the independent recompute bit-for-bit
    // (this model has no overdispersion → no gamma term). No obs-scoring.
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let no_obs: Vec<Observation> = vec![];
    let no_map = ObsAtSubstep::new();
    let comps = complete_data_loglik(
        &compiled, &traj, &params, &no_obs, dt, &obs_model, &no_map,
    ).unwrap();
    assert_eq!(comps.transition.to_bits(), d_exact.to_bits(),
        "complete_data_loglik must read rec.(t0,dt_substep): {} vs recompute {}",
        comps.transition, d_exact);

    // Load-bearing: the uniform `s·dt` reconstruction is materially different, so
    // a producer that silently wrote uniform records (or a consumer that ignored
    // them) would change the density — this is the regression the gate guards.
    eprintln!("  d_exact={d_exact:.6e}  d_uniform={d_uniform:.6e}  Δ={:.3e}", d_exact - d_uniform);
    assert!((d_exact - d_uniform).abs() > 1e-3,
        "realized grid must differ from the uniform reconstruction (Δ={:.3e})",
        d_exact - d_uniform);
}

/// (b) Cross-function FD on a SHORTENED-substep trajectory: the value and the
/// analytic gradient reconstruct the same realized `(t0, dt_substep)`. Miss a
/// site and the gradient drifts from the FD of the value here. Obs-independent
/// (transition + gamma density only) — called with no observations.
#[test]
fn exact_shortened_substep_gradient_matches_fd() {
    let (compiled, params) = seir_seasonal();
    let t_start = compiled.model.simulation.t_start;
    let dt = 1.0;
    let observations = obs_at(&[40.5, 90.2, 140.8, 190.3]);
    let n_params = compiled.param_index.len();

    let grid = build_substep_grid(t_start, dt, &observations, &[], StepPolicy::Exact).unwrap();
    let mut rng = StatefulRng::new(SEED);
    let traj = simulate_reference_on_grid(&compiled, &params, dt, &grid.steps, None, sim::rng::BinomialAlgorithm::default(), &mut rng).unwrap();
    assert!(traj.substeps.iter().any(|r| r.dt_substep != dt),
        "gate is vacuous without a shortened substep");

    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let no_obs: Vec<Observation> = vec![];
    let no_map = ObsAtSubstep::new();
    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let estimated_to_model: Vec<usize> = (0..n_params).collect();
    let rate_grads_for_run =
        resolve_rate_grad_for_run(&compiled.resolved.rate_grads_indexed, &model_to_estimated);

    // Analytic gradient on the shortened-substep trajectory.
    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &traj, &params, &no_obs, dt, &obs_model, n_params, &rate_grads_for_run, &no_map, &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "LL must be finite");

    // FD the INDEPENDENT value function (complete_data_loglik) — cross-function,
    // so a t/dt drift confined to one function cannot cancel.
    let value = |p: &[f64]| -> f64 {
        complete_data_loglik(&compiled, &traj, p, &no_obs, dt, &obs_model, &no_map)
            .unwrap()
            .total
    };
    // beta threads through seasonal(t) (time path) AND p=1-exp(-rate·dt_substep)
    // (magnitude path); gamma/sigma are time-homogeneous regression checks.
    let eps = 1e-5;
    for name in ["beta", "gamma", "sigma"] {
        let idx = compiled.param_index[name];
        let mut pp = params.clone();
        let mut pm = params.clone();
        pp[idx] += eps;
        pm[idx] -= eps;
        let fd = (value(&pp) - value(&pm)) / (2.0 * eps);
        let analytic = grad[idx];
        let denom = analytic.abs().max(1.0);
        let rel_err = (fd - analytic).abs() / denom;
        eprintln!("  {name}: analytic={analytic:.6e} fd={fd:.6e} rel_err={rel_err:.2e}");
        assert!(rel_err < 1e-4,
            "{name}: analytic gradient {analytic:.6e} disagrees with FD {fd:.6e} \
             on the shortened-substep trajectory (rel_err {rel_err:.2e})");
    }
}
