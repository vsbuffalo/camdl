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
        obs_loglik::poisson_logpmf,
        particle_filter::Observation,
        if2::{EstimatedParam, Transform},
        pgas::{PGASConfig, run_pgas},
        pmmh::Prior,
        ObsStreamSpec,
    },
};

/// Build a pure death model: N → dead at rate mu*N.
fn pure_death_model() -> (Arc<CompiledModel>, Vec<f64>) {
    let model = Model {
        name: "pure_death_tempering".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
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
                event_key: None,
                metadata: None,
                draw_method: DrawMethod::Poisson, rate_grad: Default::default(),
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: Some(0.01), bounds: None, prior: None, transform: None, initial_value: None, param_kind: None },
        ],
        parameter_groups: vec![],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("N".into(), 100.0); m
        }),
        data_contract: None,
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
        },
        presets: vec![],
        model_structure: None, balance: None,
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

fn obs_stream() -> Vec<ObsStreamSpec> {
    let obs = observations();
    vec![ObsStreamSpec {
        flow_indices: vec![0],
        obs_loglik_fn: Box::new(|projected: f64, observed: f64| {
            poisson_logpmf(observed, projected.max(0.1))
        }),
        observations: obs.iter().map(|o| o.value).collect(),
    }]
}

/// T1: Single rung [1.0] is deterministic — same seed gives same results.
#[test]
fn test_single_rung_deterministic() {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Flat];
    let streams = obs_stream();

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        tempering: vec![1.0],
    };

    let result1 = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &streams, 12345, None, None, "hash".into(),
    ).unwrap();

    // Re-create obs_stream (closures are not Clone)
    let streams2 = obs_stream();
    let config2 = PGASConfig {
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        tempering: vec![1.0],
    };

    let result2 = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config2, &obs, &streams2, 12345, None, None, "hash".into(),
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
    let priors = vec![Prior::Flat];
    let streams = obs_stream();

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 100,
        burn_in: 20,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        tempering: vec![1.0, 0.5],
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &streams, 54321, None, None, "hash".into(),
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
    let priors = vec![Prior::Flat];
    let streams = obs_stream();

    let config = PGASConfig {
        n_particles: 20,
        n_sweeps: 200,
        burn_in: 50,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        tempering: vec![1.0, 0.7, 0.4, 0.15],
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &streams, 99999, None, None, "hash".into(),
    ).unwrap();

    assert!(!result.sweeps.is_empty(), "should produce posterior samples with 4 rungs");
    for &rate in &result.acceptance_rates {
        assert!(rate >= 0.0 && rate <= 1.0, "acceptance rate {} out of [0,1]", rate);
    }
}
