//! Integration smoke test for gh#20 + gh#76: PGAS+NUTS on a model where
//! both σ² (process noise) and an obs-model parameter are in the
//! estimated set. The acceptance bar is qualitative — chain must run
//! to completion without diverging, step size must adapt above
//! machine-epsilon, and posterior acceptance > 0.
//!
//! This test is marked `#[ignore]` so it doesn't run on every `cargo
//! test`. To run it:
//!
//!   cargo test --release -p sim --test pgas_obs_overdisp_smoke -- --ignored --nocapture
//!
//! Why "qualitative" rather than a numerical assertion: the brief
//! explicitly says "report pre-fix and post-fix step sizes in the close
//! comment — this is a smoke test, not a precision claim." A
//! load-bearing numerical assertion would also be sensitive to small
//! changes in the chain_binomial RNG path that have nothing to do with
//! the gradient fix.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::BoundObs;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamSpec, StreamProjection};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

fn set_param_defaults(model: &mut ir::Model, defaults: &[(&str, f64)]) {
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            } else {
                p.value = p.value.with_value(0.5);
            }
        }
    }
}

/// gh#20 smoke: estimate σ² with PGAS+NUTS. Pre-fix, the C1 gate
/// prevented this from running at all. Post-fix, the chain runs and
/// NUTS sees a non-zero gradient on the σ² axis.
#[test]
#[ignore]
fn smoke_pgas_nuts_estimates_sigma_se() {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", 0.1),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    // Simulate "truth" trajectory; use it as a fake observation stream
    // (project incidence with rho=1.0 effectively → no obs-density gradient
    // contribution from the smoke; this isolates gh#20).
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Build incidence-style observations (cumulative infection flow,
    // weekly). No obs-likelihood params estimated — just σ_se.
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        // assume transition index 0 is infection (it's the first in the
        // sir_overdispersion fixture).
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum_infection as f64 });
            cum_infection = 0;
        }
    }

    // Build a NegBin obs model with FIXED rho=1 and k=10. Only σ_se is
    // estimated.
    let obs_model_ = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: ir::observation::ObservationModel {
                name: "weekly_cases".into(),
                source: "weekly_cases".into(),
                columns: vec![
                    ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                    ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ],
                scored: "weekly_cases".into(),
                emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: ir::observation::Projection::CumulativeFlow("infection".into()),
                projection_state_grad: Default::default(),
                likelihood: ir::observation::Likelihood::NegBinomial(
                    ir::observation::NegBinomialLikelihood {
                        mean: ir::Diffable::new(ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                            bin_op: ir::expr::BinOpExpr {
                                op: ir::expr::BinOp::Add,
                                left: Box::new(ir::expr::Expr::Projected(
                                    ir::expr::ProjectedExpr { projected: () })),
                                right: Box::new(ir::expr::Expr::Const(
                                    ir::expr::ConstExpr { value: 0.1 })),
                            },
                        })),
                        dispersion: ir::Diffable::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 10.0 })),
                    }),
            },
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![
        EstimatedParam {
            name: "sigma_se".into(),
            index: compiled.param_index["sigma_se"],
            initial: 0.1,
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.001, hi: 2.0 },
            lower: 0.001,
            upper: 2.0,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        },
    ];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
            trajectory_representation: Default::default(),
            trajectory_kernel: Default::default(),
        n_particles: 100,
        n_sweeps: 100,
        burn_in: 30,
        thin: 1,
        dt,
        use_nuts: true,        // NUTS — exercises the gradient
        dense_mass: false,
        max_tree_depth: 8,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model_, 12345, None, None, "smoke_gh20".into(),
    ).unwrap();

    // Acceptance bar: chain ran to completion, posterior acceptance > 0,
    // adapted step size > floor.
    let final_ll = result.sweeps.last().unwrap().log_complete_data_ll;
    assert!(final_ll.is_finite(), "final LL must be finite, got {}", final_ll);

    let post_burn = &result.sweeps[config.burn_in..];
    let accept_count: usize = post_burn.iter()
        .map(|s| s.accepted.iter().filter(|&&x| x).count())
        .sum();
    let total_props: usize = post_burn.iter().map(|s| s.accepted.len()).sum();
    let accept_rate = accept_count as f64 / total_props.max(1) as f64;

    eprintln!("[gh#20 smoke] n_sweeps={}, burn_in={}", config.n_sweeps, config.burn_in);
    eprintln!("[gh#20 smoke] post-burn acceptance rate: {:.3}", accept_rate);
    eprintln!("[gh#20 smoke] adapted NUTS step size: {:.4}", result.resume_state.nuts_step_size);
    eprintln!("[gh#20 smoke] final LL: {:.2}", final_ll);
    eprintln!("[gh#20 smoke] n_divergent_post_burn: {}", result.n_divergent_post_burn);

    assert!(accept_rate > 0.0, "post-burn NUTS acceptance rate must be > 0");
    assert!(result.resume_state.nuts_step_size > 1e-8,
        "adapted step size must be > 1e-8 (got {:.2e})",
        result.resume_state.nuts_step_size);
}

/// gh#76 smoke: estimate ρ (NegBin obs-model param) with PGAS+NUTS on
/// an SEIR model. Pre-fix, the C1 gate prevented this from running at
/// all. Post-fix, the chain runs and NUTS sees a non-zero gradient on
/// the ρ axis.
#[test]
#[ignore]
fn smoke_pgas_nuts_estimates_rho() {
    let mut model = load_model("../../../ocaml/golden/seir_observations.ir.json");
    // Strip the bernoulli stream to avoid coupling the gh#76 smoke to
    // p_detect's gradient.
    model.observations.retain(|o| o.name == "weekly_cases");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1),
        ("rho", 0.5), ("k", 5.0), ("p_detect", 0.8),
        ("N0", 10000.0), ("I0", 10.0),
    ]);
    model.simulation.t_end = 60.0;
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let dt = 1.0;
    let mut rng = StatefulRng::new(43);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Build weekly NegBin obs from incidence(infection).
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            // Observed = rho * cum_infection (no noise — sharpens the
            // posterior on rho).
            let obs_v = (params[compiled.param_index["rho"]] * cum_infection as f64).round();
            obs.push(Observation { time: t, value: obs_v });
            cum_infection = 0;
        }
    }

    // Re-use the original observation block (NegBinomial with rho * incidence).
    let obs_model_ = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(
                compiled.model.transitions.iter().enumerate()
                    .filter(|(_, t)| t.name == "infection")
                    .map(|(i, _)| i)
                    .collect(),
            ),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![
        EstimatedParam {
            name: "rho".into(),
            index: compiled.param_index["rho"],
            initial: 0.5,
            rw_sd: 0.02,
            transform: Transform::Logit { lo: 0.001, hi: 0.999 },
            lower: 0.001,
            upper: 0.999,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        },
    ];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
            trajectory_representation: Default::default(),
            trajectory_kernel: Default::default(),
        n_particles: 100,
        n_sweeps: 100,
        burn_in: 30,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 8,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model_, 67890, None, None, "smoke_gh76".into(),
    ).unwrap();

    let final_ll = result.sweeps.last().unwrap().log_complete_data_ll;
    assert!(final_ll.is_finite(), "final LL must be finite, got {}", final_ll);

    let post_burn = &result.sweeps[config.burn_in..];
    let accept_count: usize = post_burn.iter()
        .map(|s| s.accepted.iter().filter(|&&x| x).count())
        .sum();
    let total_props: usize = post_burn.iter().map(|s| s.accepted.len()).sum();
    let accept_rate = accept_count as f64 / total_props.max(1) as f64;

    eprintln!("[gh#76 smoke] n_sweeps={}, burn_in={}", config.n_sweeps, config.burn_in);
    eprintln!("[gh#76 smoke] post-burn acceptance rate: {:.3}", accept_rate);
    eprintln!("[gh#76 smoke] adapted NUTS step size: {:.4}", result.resume_state.nuts_step_size);
    eprintln!("[gh#76 smoke] final LL: {:.2}", final_ll);
    eprintln!("[gh#76 smoke] n_divergent_post_burn: {}", result.n_divergent_post_burn);

    assert!(accept_rate > 0.0, "post-burn NUTS acceptance rate must be > 0");
    assert!(result.resume_state.nuts_step_size > 1e-8,
        "adapted step size must be > 1e-8 (got {:.2e})",
        result.resume_state.nuts_step_size);
}
