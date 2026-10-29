//! Determinism pin for the PGAS chain: the same seed gives the same draws.
//!
//! Two `run_pgas` calls with one seed must agree on every retained sweep's
//! parameters and complete-data log-likelihood. The draws this test produces
//! are the byte-level reference a change to `pgas.rs` is checked against when
//! it claims not to move a single number.

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
        constants: vec![],
        ic_grad: Default::default(),
        name: "pure_death".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
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
        initial_conditions: InitialConditions::constants({
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
            t_end_anchor: None,
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
        transform: Transform::for_domain(1e-6, 1.0, true).expect("domain"),
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let obs = observations();
    MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![StreamSpec {
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
                covers: None,
                projection: ir::observation::Projection::CumulativeFlow("death".into()),
                projection_state_grad: Default::default(),
                likelihood: ir::observation::Likelihood::Poisson(ir::observation::PoissonLikelihood {
                    // rate = projected + 0.1 (floor to avoid Poisson(0) → -inf)
                    rate: ir::Diffable::new(ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                        bin_op: ir::expr::BinOpExpr {
                            op: ir::expr::BinOp::Add,
                            left: Box::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
                            right: Box::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.1 })),
                        },
                    })),
                }),
            },
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            times: sim::inference::multi_stream_obs::StreamTimes::contiguous_for(
                &sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]), 0.0,
                obs.iter().map(|o| o.time).collect(),
            ).unwrap(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap()
}

/// Same seed, same draws.
#[test]
fn cold_chain_is_deterministic_for_a_fixed_seed() {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let if2_params = vec![mu_param()];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let obs_m = obs_model(&compiled);

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result1 = run_pgas(
        &compiled, &if2_params, &priors, &base_params,
        &config, &obs, &obs_m, 12345, None, None, "hash".into(),
    ).unwrap();

    let obs_m2 = obs_model(&compiled);
    let config2 = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 20,
        n_sweeps: 50,
        burn_in: 10,
        thin: 1,
        dt: 1.0,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10,
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
