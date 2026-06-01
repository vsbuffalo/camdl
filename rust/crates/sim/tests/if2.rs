//! IF2 integration tests.
//!
//! Test 1: Convergence from dispersed start on sir_basic.
//! Test 2: Parameters stay within bounds (scaled logit enforcement).
//! Test 3: No-cooling mode explores without contracting.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr, PopSumExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    chain_binomial::{step_one, StepScratch},
    inference::{
        obs_loglik::negbin_logpmf,
        if2::{run_if2, IF2Config, EstimatedParam, Transform},
        ChainBinomialProcess,
        traits::ObservationModel,
        ParticleState,
    },
    rng::StatefulRng,
};

/// Test-only observation model: observes flow_accumulators[1] (recovery flow)
/// with NegBin(obs, projected, k=10) likelihood.
struct NegBinFlowObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
    flow_index: usize,
}

impl ObservationModel<ParticleState> for NegBinFlowObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let projected = state.flow_accumulators[self.flow_index] as f64;
        negbin_logpmf(self.observations[obs_idx], projected.max(0.1), 10.0)
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _state: &ParticleState, _obs_idx: usize, _params: &[f64], _rng: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _state: &ParticleState, _obs_idx: usize, _params: &[f64]) -> Vec<f64> { vec![] }
}

fn sir_model() -> (CompiledModel, Vec<f64>) {
    let model = Model {
        name: "sir_if2_test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Div,
                    left: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                            op: BinOp::Mul,
                            left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                            right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                        }})),
                        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                    }})),
                    right: Box::new(Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] })),
                }}),
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
            },
            Transition {
                name: "recovery".into(),
                stoichiometry: vec![
                    StoichiometryEntry("I".into(), -1),
                    StoichiometryEntry("R".into(), 1),
                ],
                rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                }}),
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: Some(0.3), bounds: Some((0.01, 2.0)), prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None, hierarchical: None },
            Parameter { name: "gamma".into(), value: Some(0.1), bounds: Some((0.01, 1.0)), prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None, hierarchical: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("S".into(), 990.0);
            m.insert("I".into(), 10.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 80.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 80.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Generate synthetic weekly data from a single simulation.
/// Returns (obs_times, obs_values) for the recovery flow.
fn generate_data(compiled: &CompiledModel, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StatefulRng::new(42);
    let n_int = compiled.int_local_to_global.len();
    let n_tr = compiled.model.transitions.len();
    let mut state = ParticleState::new(n_int, n_tr);
    let (init, _) = compiled.initial_state(params).unwrap();
    state.counts.copy_from_slice(&init.counts);

    let mut scratch = StepScratch::new(compiled);
    let mut obs_times = Vec::new();
    let mut obs_values = Vec::new();
    let mut t = 0.0;
    while t < 77.0 {
        for _ in 0..7 {
            step_one(compiled, &mut state.counts, &mut state.flow_accumulators, params, t, 1.0, &mut rng, &mut scratch, &compiled.resolve_fire_steps(1.0, params)).unwrap();
            t += 1.0;
        }
        // Project: recovery flow (index 1)
        let cases = state.flow_accumulators[1] as f64;
        obs_times.push(t);
        obs_values.push(cases);
        state.reset_flows();
    }
    (obs_times, obs_values)
}

#[test]
fn test_if2_converges_from_dispersed_start() {
    let (compiled, true_params) = sir_model();
    let (obs_times, obs_values) = generate_data(&compiled, &true_params);

    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone(), 1.0);
    let obs_model = NegBinFlowObs {
        observations: obs_values,
        obs_times,
        flow_index: 1,
    };

    // Start from dispersed parameters (beta=0.8, gamma=0.3 — wrong by 2-3×)
    let mut start_params = true_params.clone();
    start_params[0] = 0.8;  // beta (true: 0.3)
    start_params[1] = 0.3;  // gamma (true: 0.1)

    let if2_params = vec![
        EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.8,
            rw_sd: 0.05, transform: Transform::Logit { lo: 0.01, hi: 2.0 }, lower: 0.01, upper: 2.0, ivp: false, rw_sd_auto: false,
        },
        EstimatedParam {
            name: "gamma".into(), index: 1, initial: 0.3,
            rw_sd: 0.01, transform: Transform::Logit { lo: 0.01, hi: 1.0 }, lower: 0.01, upper: 1.0, ivp: false, rw_sd_auto: false,
        },
    ];

    let config = IF2Config {
        n_particles: 200,
        n_iterations: 30,
        cooling_fraction: 0.90,
        cooling_target_iters: 50, simplex_groups: vec![],
        dt: 1.0,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        pf_wallclock_disabled: false,
    };

    let result = run_if2(
        &process, &obs_model, &start_params, &if2_params, &config, 42,
    ).unwrap();

    let final_beta = result.mle[0];
    let final_gamma = result.mle[1];

    eprintln!("IF2 converged: beta={:.4}, gamma={:.4}, loglik={:.2}",
        final_beta, final_gamma, result.final_loglik);

    // Should converge toward true values.
    assert!((final_beta - 0.3).abs() < 0.15,
        "IF2 beta={:.3}, expected ~0.3 (started at 0.8)", final_beta);
    assert!((final_gamma - 0.1).abs() < 0.05,
        "IF2 gamma={:.3}, expected ~0.1 (started at 0.3)", final_gamma);

    // Log-likelihood must be finite and better than the first iteration
    let first_ll = result.iterations[0].if2_perturbed_loglik;
    let last_ll = result.final_loglik;
    assert!(last_ll.is_finite(), "final loglik should be finite, got {}", last_ll);
    assert!(last_ll >= first_ll - 5.0,
        "loglik should not degrade: first={:.1}, last={:.1}", first_ll, last_ll);

    // The last few iterations should have stable loglik (not oscillating wildly)
    let tail_lls: Vec<f64> = result.iterations.iter().rev().take(5)
        .map(|it| it.if2_perturbed_loglik).filter(|l| l.is_finite()).collect();
    if tail_lls.len() >= 3 {
        let mean_ll = tail_lls.iter().sum::<f64>() / tail_lls.len() as f64;
        let max_dev = tail_lls.iter().map(|&l| (l - mean_ll).abs()).fold(0.0f64, f64::max);
        assert!(max_dev < 20.0,
            "tail loglik oscillating too much: mean={:.1}, max_dev={:.1}", mean_ll, max_dev);
    }
}

#[test]
fn test_if2_respects_bounds() {
    let (compiled, true_params) = sir_model();
    let (obs_times, obs_values) = generate_data(&compiled, &true_params);

    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone(), 1.0);
    let obs_model = NegBinFlowObs {
        observations: obs_values,
        obs_times,
        flow_index: 1,
    };

    // Use tight bounds to test enforcement
    let if2_params = vec![
        EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.3,
            rw_sd: 0.1, // aggressive — would escape without bounds
            transform: Transform::Logit { lo: 0.1, hi: 0.5 }, lower: 0.1, upper: 0.5, ivp: false, rw_sd_auto: false,
        },
        EstimatedParam {
            name: "gamma".into(), index: 1, initial: 0.1,
            rw_sd: 0.03,
            transform: Transform::Logit { lo: 0.05, hi: 0.2 }, lower: 0.05, upper: 0.2, ivp: false, rw_sd_auto: false,
        },
    ];

    let config = IF2Config {
        n_particles: 100,
        n_iterations: 20,
        cooling_fraction: 0.95,
        cooling_target_iters: 50, simplex_groups: vec![],
        dt: 1.0,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        pf_wallclock_disabled: false,
    };

    let result = run_if2(
        &process, &obs_model, &true_params, &if2_params, &config, 42,
    ).unwrap();

    // All iterations should have beta in [0.1, 0.5] and gamma in [0.05, 0.2]
    for it in &result.iterations {
        let beta = it.param_means[0];
        let gamma = it.param_means[1];
        assert!((0.1..=0.5).contains(&beta),
            "beta={:.4} outside bounds [0.1, 0.5] at iter {}", beta, it.iteration);
        assert!((0.05..=0.2).contains(&gamma),
            "gamma={:.4} outside bounds [0.05, 0.2] at iter {}", gamma, it.iteration);
    }
}

#[test]
fn test_if2_no_cooling_explores() {
    let (compiled, true_params) = sir_model();
    let (obs_times, obs_values) = generate_data(&compiled, &true_params);

    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone(), 1.0);
    let obs_model = NegBinFlowObs {
        observations: obs_values,
        obs_times,
        flow_index: 1,
    };

    let if2_params = vec![
        EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.3,
            rw_sd: 0.02, transform: Transform::Logit { lo: 0.01, hi: 2.0 }, lower: 0.01, upper: 2.0, ivp: false, rw_sd_auto: false,
        },
    ];

    // No cooling (cooling_fraction = 1.0) — perturbations don't shrink
    let config = IF2Config {
        n_particles: 100,
        n_iterations: 15,
        cooling_fraction: 1.0,
        cooling_target_iters: 50, simplex_groups: vec![],
        dt: 1.0,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        pf_wallclock_disabled: false,
    };

    let result = run_if2(
        &process, &obs_model, &true_params, &if2_params, &config, 42,
    ).unwrap();

    // With no cooling, beta should still be wandering (not converged tight)
    let betas: Vec<f64> = result.iterations.iter().map(|it| it.param_means[0]).collect();
    let beta_range = betas.iter().cloned().fold(f64::INFINITY, f64::min)
        ..=betas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = *beta_range.end() - *beta_range.start();

    // Should have noticeable spread (>1% of value) across iterations
    assert!(spread > 0.003,
        "no-cooling beta spread={:.4} — should be wandering, not converging", spread);
}

// ── Transform tests ─────────────────────────────────────────────────────

#[test]
fn log_transform_clamps_to_bounds() {
    let param = EstimatedParam {
        name: "test".into(), index: 0, initial: 1.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 0.01, hi: 100.0 },
        lower: 0.01, upper: 100.0, ivp: false, rw_sd_auto: false,
    };
    // Extreme positive z → exp(z) → clamped to hi
    assert_eq!(param.from_transformed(1000.0), 100.0);
    // Extreme negative z → exp(z) → clamped to lo
    assert_eq!(param.from_transformed(-1000.0), 0.01);
    // Normal z → within bounds
    let v = param.from_transformed(1.0); // e^1 ≈ 2.718
    assert!(v > 0.01 && v < 100.0);
}

#[test]
fn logit_transform_enforces_bounds() {
    let param = EstimatedParam {
        name: "test".into(), index: 0, initial: 0.5, rw_sd: 0.1,
        transform: Transform::Logit { lo: 0.0, hi: 1.0 },
        lower: 0.0, upper: 1.0, ivp: false, rw_sd_auto: false,
    };
    // Any z → result in [lo, hi] (saturates at bounds for extreme z)
    let v1 = param.from_transformed(100.0);
    let v2 = param.from_transformed(-100.0);
    let v3 = param.from_transformed(0.0);
    assert!((0.0..=1.0).contains(&v1), "logit(100) = {} not in [0,1]", v1);
    assert!((0.0..=1.0).contains(&v2), "logit(-100) = {} not in [0,1]", v2);
    assert!((v3 - 0.5).abs() < 1e-10, "logit(0) should be 0.5, got {}", v3);
    // Moderate z → strictly interior
    let v4 = param.from_transformed(2.0);
    assert!(v4 > 0.01 && v4 < 0.99, "logit(2) = {} should be interior", v4);
}

#[test]
fn log_round_trip_within_bounds() {
    let param = EstimatedParam {
        name: "test".into(), index: 0, initial: 5.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 0.001, hi: 1000.0 },
        lower: 0.001, upper: 1000.0, ivp: false, rw_sd_auto: false,
    };
    for &x in &[0.001, 0.01, 1.0, 10.0, 100.0, 1000.0] {
        let z = param.to_transformed(x);
        let back = param.from_transformed(z);
        assert!((back - x).abs() < 1e-10 * x, "round-trip failed: {} → {} → {}", x, z, back);
    }
}

#[test]
fn logit_round_trip() {
    let param = EstimatedParam {
        name: "test".into(), index: 0, initial: 0.5, rw_sd: 0.1,
        transform: Transform::Logit { lo: 0.0, hi: 1.0 },
        lower: 0.0, upper: 1.0, ivp: false, rw_sd_auto: false,
    };
    for &x in &[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let z = param.to_transformed(x);
        let back = param.from_transformed(z);
        assert!((back - x).abs() < 1e-8, "round-trip failed: {} → {} → {}", x, z, back);
    }
}

#[test]
fn scaled_logit_round_trip() {
    // Logit on [0.01, 0.10] — the s0 case
    let param = EstimatedParam {
        name: "s0".into(), index: 0, initial: 0.03, rw_sd: 0.005,
        transform: Transform::Logit { lo: 0.01, hi: 0.10 },
        lower: 0.01, upper: 0.10, ivp: true, rw_sd_auto: false,
    };
    for &x in &[0.015, 0.03, 0.05, 0.08, 0.095] {
        let z = param.to_transformed(x);
        let back = param.from_transformed(z);
        assert!((back - x).abs() < 1e-8, "scaled logit round-trip: {} → {} → {}", x, z, back);
    }
}

// ── Cooling schedule regression tests ──────────────────────────────────

/// Verify cooling schedule matches pomp's cooling.fraction.50 semantics.
#[test]
fn cooling_cf50_matches_pomp_semantics() {
    let cf50 = 0.05_f64;
    let target_iters = 50_usize;
    let n_obs = 100_usize;

    let total_target_steps = target_iters as f64 * n_obs as f64;
    let per_step = cf50.powf(2.0 / total_target_steps);

    let steps_halfway = (target_iters / 2) * (1 + n_obs);
    let cooling_halfway = per_step.powi(steps_halfway as i32);

    let ratio = cooling_halfway / cf50;
    assert!(ratio > 0.8 && ratio < 1.0,
        "halfway cooling should be close to cf50={}: got {:.6} (ratio={:.4})",
        cf50, cooling_halfway, ratio);

    let steps_full = target_iters * (1 + n_obs);
    let cooling_full = per_step.powi(steps_full as i32);
    let expected_full = cf50 * cf50;
    let ratio_full = cooling_full / expected_full;
    assert!(ratio_full > 0.5 && ratio_full < 1.0,
        "full cooling should be close to cf50²={:.6}: got {:.6} (ratio={:.4})",
        expected_full, cooling_full, ratio_full);
}

#[test]
fn cooling_per_step_valid_range() {
    for &cf50 in &[0.01_f64, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99] {
        for &n_obs in &[10, 50, 100, 500] {
            for &target_iters in &[30, 50, 100] {
                let total = target_iters as f64 * n_obs as f64;
                let per_step = cf50.powf(2.0 / total);
                assert!(per_step > 0.0 && per_step < 1.0,
                    "per_step={} for cf50={}, n_obs={}, target_iters={}",
                    per_step, cf50, n_obs, target_iters);
                assert!(per_step > 0.9,
                    "per_step={} suspiciously aggressive for cf50={}, n_obs={}, target_iters={}",
                    per_step, cf50, n_obs, target_iters);
            }
        }
    }
}

#[test]
fn cooling_fraction_1_means_no_cooling() {
    let per_step = 1.0_f64.powf(2.0 / (50.0 * 100.0));
    assert_eq!(per_step, 1.0, "cf50=1.0 should give per_step=1.0");
}

#[test]
fn cooling_decreases_monotonically() {
    let cf50 = 0.05_f64;
    let n_obs = 50_usize;
    let target_iters = 50_usize;
    let per_step = cf50.powf(2.0 / (target_iters as f64 * n_obs as f64));

    let mut prev = 1.0_f64;
    for iter in 0..100 {
        let global_step = iter * (1 + n_obs);
        let cooling_now = per_step.powi(global_step as i32);
        assert!(cooling_now <= prev + 1e-15,
            "cooling not monotonically decreasing at iter {}: {} > {}",
            iter, cooling_now, prev);
        assert!(cooling_now >= 0.0, "cooling went negative at iter {}", iter);
        prev = cooling_now;
    }
}
