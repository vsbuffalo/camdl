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
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
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
        if let Some(v) = p.value.resolved_value() {
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

    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();

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

/// Fix B settling check: the trajectory gate proves forward dynamics are
/// byte-identical after shared-binding extraction, but it does NOT exercise
/// gradients. This does. `seir_defines_patch` hoists `let N[p]` into
/// per-patch bindings, and `d(infection)/dβ = S[p]·I[p]/BindingRef(N_p)` —
/// so a correct analytical gradient must evaluate the BindingRef. If
/// extraction had broken the gradient (e.g. the autodiff `BindingRef→0` arm
/// firing on a param-bearing binding, or eval not resolving the ref), the
/// analytical gradient would diverge from finite differences here.
#[test]
fn test_gradient_vs_finite_differences_spatial_bindings() {
    let model = load_model("../../../ocaml/golden/seir_defines_patch.ir.json");

    // The point of the test: the model must actually carry hoisted bindings,
    // and a rate gradient must reference one. Otherwise it proves nothing.
    assert!(!model.bindings.is_empty(),
        "seir_defines_patch must carry hoisted bindings (regen goldens)");
    let beta_grad_uses_binding = model.transitions.iter()
        .filter(|t| t.name.starts_with("infection"))
        .filter_map(|t| t.rate_grad.get("beta"))
        .any(|g| format!("{:?}", g).contains("BindingRef"));
    assert!(beta_grad_uses_binding,
        "d(infection)/dβ must reference a BindingRef — otherwise this test \
         does not exercise gradient-through-binding");

    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.3,
                "sigma" => 0.2,
                "gamma" => 0.1,
                "I0" => 5.0,
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
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );

    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();
    // gh#76: complete_data_loglik_grad gained estimated_to_model (estimated→model
    // param index). This test estimates all params in model order → identity map.
    let estimated_to_model: Vec<usize> = (0..n_params).collect();

    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings,
        n_params, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    eprintln!("  log-likelihood: {:.4}", ll);
    assert!(ll.is_finite(), "LL must be finite");

    let eps = 1e-5;
    let mut max_rel_err = 0.0_f64;
    let mut checked_beta = false;
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

        if param_names[i] == "beta" {
            checked_beta = true;
            // β flows through the spatial FOI denominator BindingRef(N_p);
            // a nonzero, FD-matching gradient is the whole point.
            assert!(fd.abs() > 1e-6,
                "β gradient should be materially nonzero (got fd={:.3e})", fd);
        }
        assert!(rel_err < 0.01,
            "gradient mismatch for {}: analytical={:.6}, fd={:.6}, rel_err={:.2e}",
            param_names[i], grad[i], fd, rel_err);
    }
    assert!(checked_beta, "β must be among the model parameters");
    eprintln!("  max relative error: {:.2e}", max_rel_err);
}

/// Mean-field coupling: the Σ-fold that drops a `Reduce` of zero derivatives
/// (2026-07-16) must leave the surviving rate gradients numerically CORRECT, not
/// merely smaller. `sir_coupling` is a two-patch model whose infection rate
/// carries the global denominator `.../reduce[N_young, N_old]`. Before the fold,
/// `d(infection)/dp` for a `p` absent from the rate (`gamma`, `N0`, `I0`) was
/// emitted as `beta·S·reduce[0/g², 0/g²]` — structurally present, zero-valued.
/// The fold drops those keys; this test proves the gradient it leaves behind
/// matches finite differences of the forward likelihood (ground truth), on the
/// exact model shape the fold fires on.
#[test]
fn test_gradient_vs_finite_differences_meanfield_coupling() {
    let model = load_model("../../../ocaml/golden/sir_coupling.ir.json");

    // Pin the fold: infection_young's rate mentions only beta, so its rate_grad
    // must carry exactly beta. gamma (recovery), N0/I0 (initial conditions) do
    // not occur in an infection rate — a Reduce of their zero derivatives must
    // have folded away, not survived as a spurious key.
    let inf = model.transitions.iter()
        .find(|t| t.name == "infection_young")
        .expect("sir_coupling must have an infection_young transition");
    let mut inf_grad_params: Vec<&str> = inf.rate_grad.keys().map(|s| s.as_str()).collect();
    inf_grad_params.sort();
    assert_eq!(inf_grad_params, vec!["beta"],
        "infection_young rate_grad must be exactly [beta] after the Σ-fold; a \
         spurious zero-valued Reduce for gamma/N0/I0 was kept: {:?}", inf_grad_params);
    // The surviving gradient still couples through the hoisted denominator.
    assert!(format!("{:?}", inf.rate_grad.get("beta")).contains("BindingRef"),
        "d(infection)/dβ must reference the shared N binding (gradient-through-binding)");

    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.05,
                "gamma" => 0.1,
                "N0" => 1000.0,
                "I0" => 10.0,
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
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );

    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();
    let estimated_to_model: Vec<usize> = (0..n_params).collect();

    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings,
        n_params, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "complete-data LL must be finite");

    // FD ground truth for the rate-flowing parameters. gamma flows through the
    // recovery transition (unaffected by the fold); beta flows through the
    // folded infection gradient. N0/I0 parameterize the initial conditions — an
    // orthogonal IVP-gradient path (ic_grad), not exercised here (empty
    // ivp_mappings), so they are not asserted.
    let rate_params = ["beta", "gamma"];
    let eps = 1e-5;
    let mut checked = 0;
    for i in 0..param_names.len() {
        if !rate_params.contains(&param_names[i].as_str()) { continue; }
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
        eprintln!("  d(ll)/d({:8}) = {:12.4} (analytic) vs {:12.4} (fd), rel_err = {:.2e}",
            param_names[i], grad[i], fd, rel_err);
        assert!(fd.abs() > 1e-6,
            "{} gradient should be materially nonzero (fd={:.3e})", param_names[i], fd);
        assert!(rel_err < 0.01,
            "folded gradient mismatch for {}: analytic={:.6}, fd={:.6}, rel_err={:.2e}",
            param_names[i], grad[i], fd, rel_err);
        checked += 1;
    }
    assert_eq!(checked, 2, "both beta and gamma must be checked against FD");
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
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
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
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();

    // Build EstimatedParams with Log transforms (like real inference)
    let if2_params: Vec<EstimatedParam> = compiled.model.parameters.iter().enumerate()
        .map(|(i, p)| EstimatedParam {
            index: i,
            name: p.name.clone(),
            initial: p.value.resolved_value().unwrap_or(0.5),
            lower: p.bounds().map_or(0.001, |b| b.0),
            upper: p.bounds().map_or(100.0, |b| b.1),
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.001, hi: 100.0 },
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        })
        .collect();

    let priors: Vec<Prior> = if2_params.iter().map(|_| Prior::Fixed(sim::inference::prior::Density::Flat)).collect();
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

/// T1-seasonal: cross-function FD gradient check on a TIME-INHOMOGENEOUS model —
/// the Stage-3 (exact-PGAS) gate's time-reconstruction arm, runnable today
/// against the current (snap) PGAS.
///
/// Every other FD-gradient test runs on a time-HOMOGENEOUS model (sir_basic,
/// seir_observations), where the gradient never consumes the substep time `t`.
/// So none can see a gradient site that reconstructs the WRONG time — exactly
/// the exact-PGAS hazard: the eight `t = t_start + s·dt` / `s·dt` sites
/// (`pgas.rs:268,568,605,704,716,869,1079`, `pgas_grad.rs:397`) convert to the
/// realized `(rec.t0, rec.dt_substep)` together; miss one and value and gradient
/// reconstruct different times, silently shifting the density.
///
/// Shape matters. This checks the gradient (`complete_data_loglik_grad`) against
/// the INDEPENDENT value function (`complete_data_loglik`) — the cross-function
/// pairing of `test_gradient_vs_finite_differences_sir`. A NUTS-target z-scale
/// test instead FD-differences `complete_data_loglik_grad`'s own value against
/// its own gradient, so a `t` that drifts CONSISTENTLY inside that one function
/// cancels and is invisible. The refactor's eight sites span BOTH functions, so
/// cross-function drift is the bug to catch; this shape catches it.
///
/// In `seir_vaccine_seasonal` the infection rate is `beta · seasonal(t) · S·I/N`,
/// so `∂L/∂beta` carries a `TimeFunc(seasonal)` factor (verified structurally
/// below) evaluated at the reconstructed time. Sensitivity is intrinsically
/// modest: `seasonal` has period 365.25 d, so `|d seasonal/dt| ≈ 2.6e-3 /day`
/// and a per-substep time error of ~0.5 d perturbs the gradient only ~0.1%.
/// Hence the 1e-4 tolerance (not the SIR test's 1e-2): correct code matches to
/// the FD floor (~1e-7), so 1e-4 still flags a systematic time drift of
/// O(0.05 day). The DOMINANT exposure — the `dt_substep` MAGNITUDE on a
/// genuinely shortened substep — produces O(10%) density errors and is the
/// strong gate; it needs exact-PGAS to produce a short substep, so it lands with
/// that increment, not here.
///
/// `alpha`/`phi_season` (seasonal amplitude/phase) enter the rate ONLY through
/// the seasonal `TimeFunc`. This was a doubly-silent zero gradient before
/// gh#119: `autodiff.ml` differentiated `TimeFunc` to `Const 0.0` AND the Rust
/// runtime baked the coefficient to a constant. Both are now fixed — the
/// coefficient is a live `ResolvedExpr` (value half) and `autodiff` emits the
/// analytic ∂forcing/∂coef through the sinusoidal closed form (gradient half),
/// so `alpha`/`phi_season` carry real, time-dependent gradients. This test is
/// the end-to-end validation: the analytic rate-density gradient (gradient
/// half) against a finite difference of the loglik (which exercises the live
/// value, value half). `beta` threads through `seasonal(t)` multiplicatively;
/// `gamma`/`sigma` are time-independent regression checks.
#[test]
fn test_gradient_vs_finite_differences_seasonal() {
    let mut model = load_model("../../../ocaml/golden/seir_vaccine_seasonal.ir.json");
    let has_grads = model.transitions.iter().any(|t| !t.rate_grad.is_empty());
    assert!(has_grads, "seasonal model must carry rate_grad (run make update-golden)");

    // Structural non-vacuity guard: ∂(infection)/∂beta MUST reference the
    // seasonal TimeFunc, or this test does not exercise the time path at all.
    let beta_grad_uses_timefunc = model.transitions.iter()
        .filter(|t| t.name == "infection")
        .filter_map(|t| t.rate_grad.get("beta"))
        .any(|g| format!("{:?}", g).contains("TimeFunc"));
    assert!(beta_grad_uses_timefunc,
        "∂(infection)/∂beta must reference the seasonal TimeFunc — otherwise this \
         test does not exercise the s·dt time-reconstruction path");

    // Defaults from seir_vaccine_seasonal.params.toml.
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
    // Focused window: epidemic growth+peak (infection fires → beta gradient
    // well-conditioned) + one SIA round (t=180) + ~half a 365.25-day seasonal
    // period of variation. The natural fractional t_end (1095.7275) is rounded
    // away under snap; its shortened-substep role is the Stage-3 variant.
    model.simulation.t_end = 200.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());

    let dt = 1.0;
    let t_end = compiled.model.simulation.t_end;
    let n_params = compiled.param_index.len();

    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
    }

    let mut rng = StatefulRng::new(42);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();

    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );
    let estimated_to_model: Vec<usize> = (0..n_params).collect();

    // Analytic gradient (the rate-density gradient NUTS consumes).
    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings,
        n_params, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "seasonal complete-data LL must be finite");
    eprintln!("  log-likelihood: {:.4}", ll);

    let beta_idx = compiled.param_index["beta"];
    assert!(grad[beta_idx].abs() > 1e-6,
        "∂L/∂beta must be materially nonzero (got {:.3e})", grad[beta_idx]);

    // beta = time-dependent gate; alpha/phi_season = forcing-coefficient
    // gradients (gh#119, both halves); gamma + sigma = time-independent
    // regression.
    let mut max_rel_err = 0.0_f64;
    for name in ["beta", "gamma", "sigma", "alpha", "phi_season"] {
        let i = compiled.param_index[name];
        let p_val = params[i];
        let eps = (1e-5 * p_val.abs()).max(1e-8);

        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;

        let ll_plus = complete_data_loglik(
            &compiled, &trajectory, &p_plus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            &compiled, &trajectory, &p_minus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let rel_err = if fd.abs() > 1e-10 {
            (grad[i] - fd).abs() / fd.abs()
        } else {
            (grad[i] - fd).abs()
        };
        max_rel_err = max_rel_err.max(rel_err);

        eprintln!("  d(ll)/d({:8}) = {:14.4} (analytical) vs {:14.4} (fd), rel_err = {:.2e}",
            name, grad[i], fd, rel_err);

        assert!(rel_err < 1e-4,
            "seasonal gradient mismatch for {}: analytical={:.6}, fd={:.6}, rel_err={:.2e}",
            name, grad[i], fd, rel_err);
    }
    eprintln!("  max relative error: {:.2e}", max_rel_err);
}

/// Lagged-forcing gradient gate (incident 2026-07-05). Sibling of the seasonal
/// test above, but the sinusoidal forcing carries `lag = 60 days`. The runtime
/// evaluates the forcing at `t − lag`, so the emitted ∂rate/∂{alpha,phi_season}
/// must too. Before the fix, `sinusoidal_closed` built the closed form over bare
/// `Time`, so the analytic gradient (at `t`) diverged from the value (at
/// `t − lag`) — sign-flipping over the period. This FD check catches it; the
/// no-lag seasonal test above stays green, so together they pin the
/// forcing-coefficient × {lag, no-lag} cells.
#[test]
fn test_gradient_vs_finite_differences_lagged_forcing() {
    let mut model = load_model("../../../tests/fixtures/gradient/ir/seir_seasonal_lagged.ir.json");
    let has_grads = model.transitions.iter().any(|t| !t.rate_grad.is_empty());
    assert!(has_grads, "lagged model must carry rate_grad (run make update-gradient-golden)");

    // Non-vacuity: the fixture must actually declare a forcing lag, or this is
    // just the no-lag seasonal case again.
    assert!(model.time_functions.iter().any(|tf| tf.lag.is_some()),
        "seir_seasonal_lagged must declare a forcing `lag` — otherwise this test \
         does not exercise the lagged-gradient path");

    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.3, "sigma" => 0.2, "gamma" => 0.1,
                "alpha" => 0.15, "phi_season" => 90.0,
                "N0" => 1_000_000.0, "I0" => 10.0,
                _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());

    let dt = 1.0;
    let t_end = compiled.model.simulation.t_end;
    let n_params = compiled.param_index.len();

    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
    }

    let mut rng = StatefulRng::new(42);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let observations: Vec<Observation> = vec![];
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();

    let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );
    let estimated_to_model: Vec<usize> = (0..n_params).collect();

    let (ll, grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings,
        n_params, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "lagged complete-data LL must be finite");
    eprintln!("  log-likelihood: {:.4}", ll);

    // alpha/phi_season are the forcing coefficients whose gradient was wrong
    // under lag; beta threads through seasonal(t) multiplicatively (correct even
    // pre-fix — ∂/∂beta keeps the TimeFunc node); gamma/sigma are
    // time-independent regressions.
    let mut max_rel_err = 0.0_f64;
    for name in ["alpha", "phi_season", "beta", "gamma", "sigma"] {
        let i = compiled.param_index[name];
        let p_val = params[i];
        let eps = (1e-5 * p_val.abs()).max(1e-8);

        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;

        let ll_plus = complete_data_loglik(
            &compiled, &trajectory, &p_plus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            &compiled, &trajectory, &p_minus, &observations, dt,
            &obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let rel_err = if fd.abs() > 1e-10 {
            (grad[i] - fd).abs() / fd.abs()
        } else {
            (grad[i] - fd).abs()
        };
        max_rel_err = max_rel_err.max(rel_err);

        eprintln!("  d(ll)/d({:10}) = {:14.4} (analytical) vs {:14.4} (fd), rel_err = {:.2e}",
            name, grad[i], fd, rel_err);

        assert!(rel_err < 1e-4,
            "lagged-forcing gradient mismatch for {}: analytical={:.6}, fd={:.6}, rel_err={:.2e}",
            name, grad[i], fd, rel_err);
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

