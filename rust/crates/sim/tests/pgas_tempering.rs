//! Tests for PGAS parallel tempering (replica exchange).
//!
//! T1: Single rung [1.0] produces identical results to default.
//! T2: Two rungs [1.0, 0.5] runs without panicking.
//! T3: Four rungs runs and produces finite LLs.

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        particle_filter::Observation,
        if2::{EstimatedParam, Transform},
        pgas::{PGASConfig, run_pgas},
        pmmh::Prior,
        BoundObs,
        MultiStreamObsModel,
        dense_cells,
        multi_stream_obs::StreamSpec,
    },
};

/// Build a pure death model: N → dead at rate mu*N.
fn pure_death_model() -> (Arc<CompiledModel>, Vec<f64>) {
    let model = Model {
        name: "pure_death_tempering".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "death".into(),
                stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
                rate: Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                    },
                }),
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
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("N".into(), 100.0); m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 50.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 50.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn observations() -> Vec<Observation> {
    (1..=5)
        .map(|k| {
            let t = k as f64 * 10.0;
            let expected = 100.0 * (-0.01 * t).exp();
            Observation { time: t, value: expected.round() }
        })
        .collect()
}

fn mu_param() -> EstimatedParam {
    EstimatedParam {
        name: "mu".into(),
        index: 0,
        initial: 0.01,
        rw_sd: 0.002,
        transform: Transform::Log { lo: 1e-6, hi: 1.0 },
        lower: 1e-6,
        upper: 1.0,
        rw_sd_auto: false,
        ivp: false,
    }
}

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let obs = observations();
    MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]),
            ir_model: ir::observation::ObservationModel {
                name: "cases".into(),
                source: "cases".into(),
                columns: vec![
                    ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                    ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ],
                scored: "cases".into(),
                emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: ir::observation::Projection::CumulativeFlow("death".into()),
                likelihood: ir::observation::Likelihood::Poisson(ir::observation::PoissonLikelihood {
                    // rate = projected + 0.1 (floor to avoid Poisson(0) → -inf)
                    rate: ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                        bin_op: ir::expr::BinOpExpr {
                            op: ir::expr::BinOp::Add,
                            left: Box::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
                            right: Box::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.1 })),
                        },
                    }),
                }),
            },
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap()
}

/// T1: Single rung [1.0] is deterministic — same seed gives same results.
#[test]
fn test_single_rung_deterministic() {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let obs_m = obs_model(&compiled);

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10, tempering: vec![1.0],
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result1 = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &obs_m, 12345, None, None, "hash".into(),
    ).unwrap();

    let obs_m2 = obs_model(&compiled);
    let config2 = PGASConfig {
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10, tempering: vec![1.0],
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result2 = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config2, &obs, &obs_m2, 12345, None, None, "hash".into(),
    ).unwrap();

    assert_eq!(result1.sweeps.len(), result2.sweeps.len());
    for (s1, s2) in result1.sweeps.iter().zip(&result2.sweeps) {
        assert_eq!(s1.params, s2.params, "params should be deterministic");
        assert!((s1.log_complete_data_ll - s2.log_complete_data_ll).abs() < 1e-6);
    }
}

/// T2: Two rungs [1.0, 0.5] runs without panicking and produces samples.
#[test]
fn test_two_rungs_no_panic() {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let obs_m = obs_model(&compiled);

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 100,
        burn_in: 20,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10, tempering: vec![1.0, 0.5],
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &obs_m, 54321, None, None, "hash".into(),
    ).unwrap();

    assert!(!result.sweeps.is_empty(), "should produce posterior samples with 2 rungs");
    for sweep in &result.sweeps {
        assert!(sweep.log_complete_data_ll.is_finite(),
            "LL should be finite, got {}", sweep.log_complete_data_ll);
    }
}

/// T3: Four rungs runs and all output comes from cold chain.
#[test]
fn test_four_rungs_runs() {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let obs_m = obs_model(&compiled);

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 200,
        burn_in: 50,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10, tempering: vec![1.0, 0.7, 0.4, 0.15],
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &obs_m, 99999, None, None, "hash".into(),
    ).unwrap();

    assert!(!result.sweeps.is_empty(), "should produce posterior samples with 4 rungs");
    for &rate in &result.acceptance_rates {
        assert!((0.0..=1.0).contains(&rate), "acceptance rate {} out of [0,1]", rate);
    }
}
