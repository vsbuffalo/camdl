//! Guard test (gh#175): PGAS must refuse hierarchical priors with a clear
//! error.
//!
//! Background: the NUTS gradient for a hierarchical leaf is stubbed to -inf
//! (`pgas.rs::prior_log_density_and_grad_z`, "Gate 3b"), and the MH
//! fallback's non-env `log_density` is likewise -inf. So a hierarchical
//! prior makes the log-posterior -inf everywhere and silently *freezes* the
//! chain at its starting point — 100% divergent transitions, 0% acceptance,
//! a posterior warm-started at truth that looks tight and well-mixed. For a
//! public-health tool that silent-wrong mode is the dangerous one.
//!
//! This test:
//!   1. Grafts a hierarchical (`Normal`) prior onto the estimated `beta`.
//!   2. Calls `run_pgas` configured to estimate `beta` via NUTS.
//!   3. Asserts the call returns `Err(SimError::Validation(...))` naming the
//!      parameter and pointing at `algorithm = pmmh`.
//!
//! Pre-guard the call returns `Ok` with a frozen-at-init chain; the
//! assertion that the error fires catches that silent-wrong regime.

use std::collections::BTreeMap;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::error::SimError;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::BoundObs;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json).unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
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

/// A plain Poisson obs over the infection flow — keeps the only "unsupported"
/// feature in this model the hierarchical prior, so the guard is what fires.
fn build_poisson_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
        }),
    }
}

#[test]
fn gh175_pgas_refuses_hierarchical_prior_with_clear_error() {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(
        &mut model,
        &[
            ("beta", 0.3),
            ("gamma", 0.1),
            ("sigma_se", 0.1),
            ("N0", 1000.0),
            ("I0", 10.0),
        ],
    );
    model.observations = vec![build_poisson_obs_block()];

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Synthetic weekly obs by summing the infection flow (transition 0).
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum_infection as f64 });
            cum_infection = 0;
        }
    }

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    )
    .unwrap();

    let if2_params = vec![EstimatedParam {
        name: "beta".into(),
        index: compiled.param_index["beta"],
        initial: 0.3,
        rw_sd: 0.02,
        transform: Transform::Log { lo: 0.001, hi: 10.0 },
        lower: 0.001,
        upper: 10.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }];

    // A hierarchical (Normal) prior on `beta`. The arg expressions are not
    // resolved — the guard fires on the variant, before any density eval —
    // so dummy hyperparameter names are fine.
    let mut args: BTreeMap<String, ir::expr::Expr> = BTreeMap::new();
    args.insert(
        "mu".into(),
        ir::expr::Expr::Param(ir::expr::ParamExpr { param: "beta_mu".into() }),
    );
    args.insert(
        "sigma".into(),
        ir::expr::Expr::Param(ir::expr::ParamExpr { param: "beta_sd".into() }),
    );
    let priors = vec![Prior::from_hierarchical_ir(&ir::parameter::HierarchicalPrior {
        kind: ir::parameter::HierarchicalKind::Normal,
        args,
        pool_over: String::new(),
    })];

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 50,
        n_sweeps: 5,
        burn_in: 2,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 4,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled,
        &if2_params,
        &priors,
        &params,
        &config,
        &obs,
        &obs_model,
        12345,
        None,
        None,
        "gate_hierarchical".into(),
    );

    match result {
        Err(SimError::Validation(msg)) => {
            assert!(
                msg.contains("beta"),
                "error must name the parameter; got: {}",
                msg
            );
            assert!(
                msg.to_lowercase().contains("hierarchical"),
                "error must name hierarchical priors; got: {}",
                msg
            );
            assert!(
                msg.contains("pmmh"),
                "error must point the user at `algorithm = pmmh`; got: {}",
                msg
            );
            eprintln!("[gate test] saw expected validation error:\n  {}", msg);
        }
        Err(e) => panic!("expected SimError::Validation, got: {:?}", e),
        Ok(_) => panic!(
            "run_pgas must refuse a hierarchical prior (the NUTS gradient is \
             stubbed to -inf, so the chain freezes at init); pre-guard this \
             returned Ok with a frozen-at-truth posterior."
        ),
    }
}
