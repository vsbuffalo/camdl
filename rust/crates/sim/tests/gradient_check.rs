//! Gradient validation: compare analytical gradients (from compiler-emitted
//! derivative expressions) against finite-difference approximations.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{IVPMapping, simulate_reference, complete_data_loglik, build_obs_at_substep};
use sim::inference::pgas_grad::complete_data_loglik_grad;
use sim::inference::MultiStreamObsModel;
use sim::inference::particle_filter::Observation;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

#[test]
fn test_gradient_vs_finite_differences_sir() {
    // Load a golden SIR model (compiled with autodiff → has rate_grad)
    let model = load_model("../../../ocaml/golden/sir_basic.ir.json");

    // Verify rate_grad is populated
    let has_grads = model.transitions.iter().any(|t| !t.rate_grad.is_empty());
    if !has_grads {
        eprintln!("  skipping: no rate_grad in golden file (run make update-golden)");
        return;
    }

    // Set parameter values (the golden file may not have defaults)
    let mut model = model;
    for p in &mut model.parameters {
        if p.value.is_none() {
            p.value = Some(match p.name.as_str() {
                "beta" => 0.4,
                "gamma" => 0.1,
                "mu" => 0.01,
                _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());

    let n_params = compiled.model.parameters.len();
    let param_names: Vec<String> = compiled.model.parameters.iter()
        .map(|p| p.name.clone()).collect();
    let param_indices: Vec<usize> = param_names.iter()
        .map(|n| *compiled.param_index.get(n.as_str()).unwrap())
        .collect();

    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    // Build rate_grads_for_run: all model params are "estimated" (est_idx == model_idx)
    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );

    // Simulate a trajectory
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];

    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt);

    // Analytical gradient
    let estimated_to_model: Vec<usize> = (0..n_params).collect();
    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings,
        n_params, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();

    eprintln!("  log-likelihood: {:.4}", ll);
    assert!(ll.is_finite(), "LL must be finite");

    // Finite-difference gradient for each parameter
    let eps = 1e-5;
    let mut max_rel_err = 0.0_f64;
    for i in 0..param_names.len() {
        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[param_indices[i]] += eps;
        p_minus[param_indices[i]] -= eps;

        let ll_plus = complete_data_loglik(
            &compiled, &trajectory, &p_plus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            &compiled, &trajectory, &p_minus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;

        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let rel_err = if fd.abs() > 1e-8 {
            (grad[i] - fd).abs() / fd.abs()
        } else {
            (grad[i] - fd).abs()
        };
        max_rel_err = max_rel_err.max(rel_err);

        eprintln!("  d(ll)/d({:12}) = {:12.4} (analytical) vs {:12.4} (fd), rel_err = {:.2e}",
            param_names[i], grad[i], fd, rel_err);

        assert!(rel_err < 0.01,
            "gradient mismatch for {}: analytical={:.6}, fd={:.6}, rel_err={:.2e}",
            param_names[i], grad[i], fd, rel_err);
    }

    eprintln!("  max relative error: {:.2e}", max_rel_err);
}

/// T1: Full NUTS target gradient check (LL + prior + Jacobian on z scale).
/// This tests the gradient composition that NUTS actually uses, including
/// the chain rule through parameter transforms and the Jacobian correction.
/// Bug #1 (double chain rule) lived in exactly this layer.
#[test]
fn test_nuts_target_gradient_on_z_scale() {
    use sim::inference::if2::{EstimatedParam, Transform};
    use sim::inference::pmmh::Prior;

    let model = load_model("../../../ocaml/golden/sir_basic.ir.json");
    let has_grads = model.transitions.iter().any(|t| !t.rate_grad.is_empty());
    if !has_grads {
        eprintln!("  skipping: no rate_grad in golden file");
        return;
    }

    let mut model = model;
    for p in &mut model.parameters {
        if p.value.is_none() {
            p.value = Some(match p.name.as_str() {
                "beta" => 0.4, "gamma" => 0.1, "mu" => 0.01, _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());

    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &[0.4, 0.1, 1000.0, 10.0], t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt);

    // Build EstimatedParams with Log transforms (like real inference)
    let if2_params: Vec<EstimatedParam> = compiled.model.parameters.iter().enumerate()
        .map(|(i, p)| EstimatedParam {
            index: i,
            name: p.name.clone(),
            initial: p.value.unwrap_or(0.5),
            lower: p.bounds.map_or(0.001, |b| b.0),
            upper: p.bounds.map_or(100.0, |b| b.1),
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.001, hi: 100.0 },
            rw_sd_auto: false,
            ivp: false,
        })
        .collect();

    let priors: Vec<Prior> = if2_params.iter().map(|_| Prior::Flat).collect();
    let base_params = vec![0.4, 0.1, 1000.0, 10.0];
    let param_names: Vec<String> = if2_params.iter().map(|p| p.name.clone()).collect();
    let d_nuts = if2_params.len();

    // Build rate_grads_for_run: map model param indices → estimated param indices
    let n_model_params = compiled.model.parameters.len();
    let mut model_to_estimated_nuts: Vec<Option<usize>> = vec![None; n_model_params];
    for (est_idx, spec) in if2_params.iter().enumerate() {
        model_to_estimated_nuts[spec.index] = Some(est_idx);
    }
    let rate_grads_for_run_nuts = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated_nuts,
    );

    // Current z values (transformed scale)
    let z: Vec<f64> = if2_params.iter()
        .map(|p| p.to_transformed(base_params[p.index]))
        .collect();

    let estimated_to_model_nuts: Vec<usize> = if2_params.iter().map(|p| p.index).collect();

    // Build the FULL NUTS target closure (same structure as run_pgas)
    let log_prob_and_grad = |z_val: &[f64]| -> (f64, Vec<f64>) {
        let mut params = base_params.clone();
        for (i, spec) in if2_params.iter().enumerate() {
            params[spec.index] = spec.from_transformed(z_val[i]);
        }

        let (ll, ll_grad_theta) = sim::inference::pgas_grad::complete_data_loglik_grad(
            &compiled, &trajectory, &params, &observations, dt,
            &obs_model, &ivp_mappings,
            d_nuts, &rate_grads_for_run_nuts, &oas,
            &estimated_to_model_nuts,
        ).unwrap_or((f64::NEG_INFINITY, vec![0.0; d_nuts]));

        let mut log_p = ll;
        let mut grad_z = vec![0.0; d_nuts];

        for i in 0..d_nuts {
            let theta = params[if2_params[i].index];
            let dtheta_dz = match &if2_params[i].transform {
                Transform::Log { .. } => z_val[i].exp(),
                Transform::Logit { lo, hi } => {
                    let p = 1.0 / (1.0 + (-z_val[i]).exp());
                    (hi - lo) * p * (1.0 - p)
                }
                Transform::None => 1.0,
            };

            // LL: chain rule
            grad_z[i] += ll_grad_theta[i] * dtheta_dz;

            // Prior (Flat → 0)
            log_p += priors[i].log_density(theta, z_val[i]);

            // Jacobian
            let log_jac = match &if2_params[i].transform {
                Transform::Log { .. } => z_val[i],
                Transform::Logit { lo, hi } => {
                    let p = 1.0 / (1.0 + (-z_val[i]).exp());
                    ((hi - lo) * p * (1.0 - p)).ln()
                }
                Transform::None => 0.0,
            };
            let jac_grad = match &if2_params[i].transform {
                Transform::Log { .. } => 1.0,
                Transform::Logit { .. } => {
                    let p = 1.0 / (1.0 + (-z_val[i]).exp());
                    1.0 - 2.0 * p
                }
                Transform::None => 0.0,
            };
            log_p += log_jac;
            grad_z[i] += jac_grad;
        }

        (log_p, grad_z)
    };

    let (val, grad) = log_prob_and_grad(&z);
    assert!(val.is_finite(), "NUTS target must be finite");

    // Finite-difference check on z scale
    let eps = 1e-5;
    let mut max_rel_err = 0.0_f64;
    for i in 0..z.len() {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[i] += eps;
        z_minus[i] -= eps;

        let fd = (log_prob_and_grad(&z_plus).0 - log_prob_and_grad(&z_minus).0) / (2.0 * eps);

        let rel_err = if fd.abs() > 1e-8 {
            (grad[i] - fd).abs() / fd.abs()
        } else {
            (grad[i] - fd).abs()
        };
        max_rel_err = max_rel_err.max(rel_err);

        eprintln!("  d(target)/dz({:12}) = {:12.4} (analytical) vs {:12.4} (fd), rel_err = {:.2e}",
            param_names[i], grad[i], fd, rel_err);

        assert!(rel_err < 0.01,
            "NUTS target gradient mismatch for {}: analytical={:.6}, fd={:.6}, rel_err={:.2e}",
            param_names[i], grad[i], fd, rel_err);
    }
    eprintln!("  max relative error: {:.2e}", max_rel_err);
}

/// T2: NUTS invariance on a known 2D Gaussian target.
/// Runs NUTS for 5K steps on N([3, -1], [[1, 0.5], [0.5, 2]]).
/// Verifies sample mean within 3σ of true mean.
#[test]
fn test_nuts_invariance_gaussian() {
    use sim::inference::nuts::{NUTSConfig, nuts_step, DualAveraging};

    // Target: 2D Gaussian with mean [3, -1], precision [[2, -0.5], [-0.5, 1]]
    // (inverse of [[1, 0.5], [0.5, 2]] ≈ [[1.143, -0.286], [-0.286, 0.571]])
    let true_mean = [3.0, -1.0];
    let prec = [[2.0_f64, -0.5], [-0.5, 1.0]]; // precision matrix

    let log_prob_and_grad = |z: &[f64]| -> (f64, Vec<f64>) {
        let dz = [z[0] - true_mean[0], z[1] - true_mean[1]];
        let log_p = -0.5 * (prec[0][0] * dz[0] * dz[0] + 2.0 * prec[0][1] * dz[0] * dz[1]
                           + prec[1][1] * dz[1] * dz[1]);
        let grad = vec![
            -(prec[0][0] * dz[0] + prec[0][1] * dz[1]),
            -(prec[0][1] * dz[0] + prec[1][1] * dz[1]),
        ];
        (log_p, grad)
    };

    let mut rng = StatefulRng::new(123);
    let mut z = vec![0.0, 0.0];
    let (mut log_p, mut grad) = log_prob_and_grad(&z);

    // Warmup: adapt step size
    let mut dual_avg = DualAveraging::new(0.5, 0.80);
    let mut step_size = 0.5;
    let warmup = 500;
    for _ in 0..warmup {
        let config = NUTSConfig { max_tree_depth: 10, step_size, mass_matrix: sim::inference::nuts::MassMatrix::identity(2) };
        let result = nuts_step(&z, log_p, &grad, &config, &log_prob_and_grad, &mut rng);
        step_size = dual_avg.update(result.mean_accept_prob);
        if result.accepted {
            z = result.params;
            log_p = result.log_posterior;
            let (_, g) = log_prob_and_grad(&z);
            grad = g;
        }
    }
    step_size = dual_avg.final_step_size();

    // Sampling
    let n_samples = 5000;
    let mut sum = [0.0_f64; 2];
    let config = NUTSConfig { max_tree_depth: 10, step_size, mass_matrix: sim::inference::nuts::MassMatrix::identity(2) };

    for _ in 0..n_samples {
        let result = nuts_step(&z, log_p, &grad, &config, &log_prob_and_grad, &mut rng);
        if result.accepted {
            z = result.params;
            log_p = result.log_posterior;
            let (_, g) = log_prob_and_grad(&z);
            grad = g;
        }
        sum[0] += z[0];
        sum[1] += z[1];
    }

    let mean = [sum[0] / n_samples as f64, sum[1] / n_samples as f64];
    eprintln!("  NUTS Gaussian test: step_size={:.4}", step_size);
    eprintln!("  sample mean: [{:.3}, {:.3}], true: [{:.1}, {:.1}]",
        mean[0], mean[1], true_mean[0], true_mean[1]);

    // With 5K samples from a Gaussian with var ~1-2, SE ≈ sqrt(2/5000) ≈ 0.02.
    // Allow 5σ = 0.1 tolerance.
    assert!((mean[0] - true_mean[0]).abs() < 0.3,
        "NUTS mean[0]={:.3}, expected {:.1}", mean[0], true_mean[0]);
    assert!((mean[1] - true_mean[1]).abs() < 0.3,
        "NUTS mean[1]={:.3}, expected {:.1}", mean[1], true_mean[1]);
}

/// Test dense mass matrix on a highly correlated 2D Gaussian (r=0.95).
/// With identity mass matrix, NUTS zigzags. With the true covariance as
/// mass matrix, NUTS should follow the ridge and give much higher ESS.
#[test]
fn test_nuts_dense_mass_matrix_correlated() {
    use sim::inference::nuts::{NUTSConfig, nuts_step, DualAveraging, MassMatrix};

    let true_mean = [0.0, 0.0];
    // Covariance: [[1.0, 0.95], [0.95, 1.0]] — correlation r=0.95
    let cov = [1.0, 0.95, 0.95, 1.0];
    // Precision = inv(cov) ≈ [[10.256, -9.744], [-9.744, 10.256]]
    let det = 1.0 * 1.0 - 0.95 * 0.95; // 0.0975
    let prec = [[1.0 / det, -0.95 / det], [-0.95 / det, 1.0 / det]];

    let log_prob_and_grad = |z: &[f64]| -> (f64, Vec<f64>) {
        let log_p = -0.5 * (prec[0][0] * z[0] * z[0] + 2.0 * prec[0][1] * z[0] * z[1]
                           + prec[1][1] * z[1] * z[1]);
        let grad = vec![
            -(prec[0][0] * z[0] + prec[0][1] * z[1]),
            -(prec[0][1] * z[0] + prec[1][1] * z[1]),
        ];
        (log_p, grad)
    };

    let mut rng = StatefulRng::new(456);
    let mass = MassMatrix::dense_from_covariance(&cov, 2);

    let mut z = vec![0.0, 0.0];
    let (mut log_p, mut grad) = log_prob_and_grad(&z);

    // Warmup with dense mass matrix
    let mut dual_avg = DualAveraging::new(0.5, 0.80);
    let mut step_size = 0.5;
    for _ in 0..200 {
        let config = NUTSConfig { max_tree_depth: 10, step_size, mass_matrix: mass.clone() };
        let result = nuts_step(&z, log_p, &grad, &config, &log_prob_and_grad, &mut rng);
        step_size = dual_avg.update(result.mean_accept_prob);
        if result.accepted {
            z = result.params; log_p = result.log_posterior;
            let (_, g) = log_prob_and_grad(&z); grad = g;
        }
    }
    step_size = dual_avg.final_step_size();

    // Sampling
    let n_samples = 2000;
    let mut samples_0 = Vec::with_capacity(n_samples);
    let mut samples_1 = Vec::with_capacity(n_samples);
    let config = NUTSConfig { max_tree_depth: 10, step_size, mass_matrix: mass.clone() };

    for _ in 0..n_samples {
        let result = nuts_step(&z, log_p, &grad, &config, &log_prob_and_grad, &mut rng);
        if result.accepted {
            z = result.params; log_p = result.log_posterior;
            let (_, g) = log_prob_and_grad(&z); grad = g;
        }
        samples_0.push(z[0]);
        samples_1.push(z[1]);
    }

    let mean_0 = samples_0.iter().sum::<f64>() / n_samples as f64;
    let mean_1 = samples_1.iter().sum::<f64>() / n_samples as f64;
    let var_0 = samples_0.iter().map(|&x| (x - mean_0).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
    let var_1 = samples_1.iter().map(|&x| (x - mean_1).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
    let cov_01 = samples_0.iter().zip(&samples_1)
        .map(|(&x, &y)| (x - mean_0) * (y - mean_1)).sum::<f64>() / (n_samples - 1) as f64;
    let r = cov_01 / (var_0.sqrt() * var_1.sqrt());

    eprintln!("  dense mass matrix test (r=0.95 target):");
    eprintln!("    step_size={:.4}", step_size);
    eprintln!("    mean=[{:.3}, {:.3}], var=[{:.3}, {:.3}], r={:.3}",
        mean_0, mean_1, var_0, var_1, r);

    assert!((mean_0 - true_mean[0]).abs() < 0.2, "mean[0]={:.3}", mean_0);
    assert!((mean_1 - true_mean[1]).abs() < 0.2, "mean[1]={:.3}", mean_1);
    assert!((var_0 - 1.0).abs() < 0.3, "var[0]={:.3}, expected ~1.0", var_0);
    assert!((var_1 - 1.0).abs() < 0.3, "var[1]={:.3}, expected ~1.0", var_1);
    assert!((r - 0.95).abs() < 0.1, "correlation={:.3}, expected ~0.95", r);
}
