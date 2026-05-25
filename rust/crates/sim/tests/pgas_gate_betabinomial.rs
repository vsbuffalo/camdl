//! Regression test for the audit-C1 preflight gate reintroduction.
//!
//! Background: gh#76 originally removed the C1 gate after wiring obs-density
//! and σ²-density gradients. But the gradient has two documented no-op arms:
//!
//!   1. `BetaBinomial`: `eval_likelihood_resolved_grad` falls through with
//!      `grad` unchanged — no helper exists in `obs_loglik.rs`.
//!   2. Parametric `DerivedExpr` projections: the chain-rule term
//!      ∂L/∂(projected) · ∂(projected)/∂θ is omitted.
//!
//! Either route lands the user in the silent-zero-gradient regime that gh#76
//! was filed against. The narrowed C1 gate refuses estimation of any
//! parameter whose reachability graph traverses one of these arms.
//!
//! This test:
//!   1. Builds an SIR model with a BetaBinomial observation referencing a
//!      parameter (`alpha_obs`) in its `alpha` argument.
//!   2. Calls `run_pgas` configured to estimate `alpha_obs` via NUTS.
//!   3. Asserts the call returns `Err(SimError::Validation(...))` with a
//!      message naming the blocked parameter and the uncovered arm.
//!
//! Pre-fix (b981d60 state) the gate is gone and the call would return Ok with
//! a silent-zero gradient on `alpha_obs`; the assertion that the error fires
//! correctly catches the regression.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::error::SimError;
use sim::inference::if2::{EstimatedParam, Transform};
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
        if p.value.is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = Some(v);
            } else {
                p.value = Some(0.5);
            }
        }
    }
}

/// Build a BetaBinomial obs stream over the infection flow with alpha that
/// references a new parameter `alpha_obs`. The literal `n = total_incidence`
/// is irrelevant for the gate check — what matters is that `alpha` depends
/// on a parameter.
fn build_betabinomial_obs_block(alpha_param: &str) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;

    // n = projected (cum_flow over an interval — passed through unchanged)
    let n = Expr::Projected(ProjectedExpr { projected: () });
    // alpha = alpha_obs (an estimated parameter)
    let alpha = Expr::Param(ParamExpr { param: alpha_param.into() });
    // beta = 1.0 (constant)
    let beta = Expr::Const(ConstExpr { value: 1.0 });

    ObservationModel {
        name: "weekly_cases".into(),
        data_stream: "weekly_cases".into(),
        schedule: ObservationSchedule::FromData,
        projection: Projection::CumulativeFlow("infection".into()),
        likelihood: Likelihood::BetaBinomial(BetaBinomialLikelihood { n, alpha, beta }),
    }
}

#[test]
fn gh76_followup_pgas_refuses_betabinomial_routed_param_with_nuts() {
    // Reuse the SIR-overdispersion fixture as a host model, then graft a
    // BetaBinomial obs onto it (replacing whatever default obs the model
    // might carry) and add an `alpha_obs` parameter that the BetaBinomial
    // alpha expression references.
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", 0.1),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    // Add the BetaBinomial-bound parameter.
    model.parameters.push(ir::parameter::Parameter {
        name: "alpha_obs".into(),
        value: Some(2.0),
        bounds: Some((0.01, 100.0)),
        prior: None,
        hierarchical: None,
        transform: None,
        initial_value: None,
        param_kind: Some("positive".into()),
        param_dim: None,
    });
    // Replace any existing obs block(s) with our BetaBinomial-on-infection.
    model.observations = vec![build_betabinomial_obs_block("alpha_obs")];

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Build synthetic weekly obs by summing the infection flow into windows.
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            // Halve to make it valid BetaBinomial (k <= n)
            obs.push(Observation { time: t, value: (cum_infection as f64 * 0.5).round() });
            cum_infection = 0;
        }
    }

    let obs_model = MultiStreamObsModel::new(
        vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: compiled.model.observations[0].clone(),
            observations: obs.iter().map(|o| o.value).collect(),
            obs_times: obs.iter().map(|o| o.time).collect(),
        }],
        compiled.clone(),
    ).unwrap();

    // Estimate `alpha_obs` — the parameter routed through the
    // BetaBinomial `alpha` arg. Pre-fix: silent zero gradient. Post-fix:
    // the C1 gate fires.
    let if2_params = vec![
        EstimatedParam {
            name: "alpha_obs".into(),
            index: compiled.param_index["alpha_obs"],
            initial: 2.0,
            rw_sd: 0.02,
            transform: Transform::Log { lo: 0.01, hi: 100.0 },
            lower: 0.01,
            upper: 100.0,
            rw_sd_auto: false,
            ivp: false,
        },
    ];
    let priors = vec![Prior::Flat];

    let config = PGASConfig {
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
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model, 12345, None, None, "gate_betabinomial".into(),
    );

    match result {
        Err(SimError::Validation(msg)) => {
            // The gate must (a) name the blocked parameter and (b) point at
            // the uncovered arm so the user knows what to do.
            assert!(msg.contains("alpha_obs"),
                "error must name the blocked parameter; got: {}", msg);
            assert!(msg.contains("BetaBinomial") || msg.contains("beta_binomial")
                    || msg.contains("beta-binomial"),
                "error must name the uncovered BetaBinomial arm; got: {}", msg);
            eprintln!("[gate test] saw expected validation error:\n  {}", msg);
        }
        Err(e) => panic!("expected SimError::Validation, got: {:?}", e),
        Ok(_) => panic!(
            "run_pgas must refuse to estimate a BetaBinomial-routed parameter \
             via NUTS (gradient arm is a documented no-op); pre-fix this returned \
             Ok with a silent-zero gradient on alpha_obs."
        ),
    }
}
