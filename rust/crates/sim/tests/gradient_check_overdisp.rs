//! Finite-difference regression test for the gamma-multiplier-density
//! gradient term in `complete_data_loglik_grad` (gh#20).
//!
//! Simulates an SIR-with-overdispersion trajectory at fixed σ², then
//! compares the analytic gradient (which now includes the
//! d/dθ log Γ(g; dt/σ², σ²/dt) chain rule through `log_gamma_density_grad_substep`)
//! against a central finite difference of the complete-data log-likelihood.
//! The acceptance bar is 1e-4 relative — the same bar the per-distribution
//! gradient helpers in `obs_loglik.rs` are tested at.
//!
//! These tests bypass `pgas::run_pgas`'s preflight gate (which still
//! blocks obs-likelihood params with NUTS until gh#76 lands) by calling
//! `complete_data_loglik_grad` directly — the same function NUTS would
//! invoke. σ² estimation with PGAS+NUTS is unblocked after this commit.

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

fn set_param_defaults(model: &mut ir::Model, defaults: &[(&str, f64)]) {
    for p in &mut model.parameters {
        if p.value.is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = Some(v);
            } else {
                p.value = Some(0.5);
            }
        }
    }
}

fn build_params_and_names(compiled: &CompiledModel) -> (Vec<f64>, Vec<String>) {
    let n = compiled.param_index.len();
    let mut params = vec![0.0; n];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    let mut names = vec![String::new(); n];
    for p in &compiled.model.parameters {
        names[compiled.param_index[p.name.as_str()]] = p.name.clone();
    }
    (params, names)
}

fn fd_check(
    compiled: &Arc<CompiledModel>,
    trajectory: &sim::inference::pgas::PGASTrajectory,
    observations: &[Observation],
    obs_model: &MultiStreamObsModel,
    params: &[f64],
    param_names: &[String],
    estimated_indices: &[usize],
    params_to_check: &[usize],
    dt: f64,
    rel_tol: f64,
    name: &str,
) {
    let d = estimated_indices.len();
    let n_model_params = compiled.model.parameters.len();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; n_model_params];
    for (est_idx, &model_idx) in estimated_indices.iter().enumerate() {
        model_to_estimated[model_idx] = Some(est_idx);
    }
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(observations, compiled.model.simulation.t_start, dt);
    let estimated_to_model: Vec<usize> = estimated_indices.to_vec();

    let (ll, grad) = complete_data_loglik_grad(
        compiled, trajectory, params, observations, dt,
        obs_model, &ivp_mappings,
        d, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "[{}] log-likelihood must be finite, got {}", name, ll);
    eprintln!("[{}] LL = {:.4}", name, ll);

    for &est_idx in params_to_check {
        let model_idx = estimated_indices[est_idx];
        let p_val = params[model_idx];
        let eps = (1e-5 * p_val.abs()).max(1e-8);

        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[model_idx] += eps;
        p_minus[model_idx] -= eps;

        let ll_plus = complete_data_loglik(
            compiled, trajectory, &p_plus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            compiled, trajectory, &p_minus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let analytic = grad[est_idx];
        let rel_err = if fd.abs() > 1e-10 {
            (analytic - fd).abs() / fd.abs()
        } else {
            (analytic - fd).abs()
        };

        eprintln!(
            "[{}] d(ll)/d({:12}) = {:14.6e} (analytic) vs {:14.6e} (fd), rel_err = {:.2e}",
            name, param_names[model_idx], analytic, fd, rel_err
        );

        assert!(
            rel_err < rel_tol,
            "[{}] gradient mismatch for {}: analytic={:.6e}, fd={:.6e}, rel_err={:.2e} (tol={:.0e})",
            name, param_names[model_idx], analytic, fd, rel_err, rel_tol
        );
    }
}

fn run_gh20_check(sigma_se: f64, seed: u64, dt: f64) {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", sigma_se),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(seed);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let total_gammas: usize = trajectory.substeps.iter().map(|s| s.gammas.len()).sum();
    assert!(total_gammas > 0,
        "gh#20 fixture must produce a trajectory with overdispersed gammas; got 0");

    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let sigma_idx = compiled.param_index["sigma_se"];
    let beta_idx = compiled.param_index["beta"];
    let gamma_idx = compiled.param_index["gamma"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // sigma_se is the new term added by gh#20.
        // beta and gamma are regression checks — gh#20 must not break the
        // existing rate-density gradient.
        &[sigma_idx, beta_idx, gamma_idx],
        dt, 1e-4,
        &format!("gh20_sigma_se={}_seed={}", sigma_se, seed),
    );
}

#[test]
fn gh20_gamma_density_grad_matches_fd_small_sigma() {
    // Small σ²: shape = dt/σ² is large, gammas tightly concentrated near 1.
    run_gh20_check(0.01, 42, 1.0);
}

#[test]
fn gh20_gamma_density_grad_matches_fd_medium_sigma() {
    run_gh20_check(0.1, 43, 1.0);
}

#[test]
fn gh20_gamma_density_grad_matches_fd_large_sigma() {
    // Large σ²: shape ≈ 1, broader gamma distribution — exercises the
    // digamma + ln(g) terms in a regime where each contribution is non-trivial.
    run_gh20_check(1.0, 44, 1.0);
}
