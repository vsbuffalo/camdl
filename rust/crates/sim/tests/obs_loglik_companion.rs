//! `compile_obs_loglik_pf`, the density companion of the observation sampler,
//! is each family's log mass or density at the arguments the likelihood
//! resolves: `projected` passed in, constants, and a data column read from
//! the caller's aux (a binomial's and a beta-binomial's trials).

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{ConstExpr, Expr, ObsColumnRefExpr, ProjectedExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    observation::{
        BernoulliLikelihood, BetaBinomialLikelihood, BetaLikelihood, BinomialLikelihood, ColumnRole, Likelihood,
        NegBinomialLikelihood, NormalLikelihood, ObsColumn, ObservationModel as IrObs, ObservationSchedule,
        PoissonLikelihood, Projection, ZeroInflatedNegBinomialLikelihood,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::obs_loglik as ll;

fn projected() -> Diffable {
    Diffable::new(Expr::Projected(ProjectedExpr { projected: () }))
}

fn constant(value: f64) -> Diffable {
    Diffable::new(Expr::Const(ConstExpr { value }))
}

fn tested() -> Expr {
    Expr::ObsColumnRef(ObsColumnRefExpr { obs_column_ref: "tested".into() })
}

/// A one-compartment model observed by `likelihood`, with a `tested` data
/// column.
fn model(likelihood: Likelihood) -> Arc<CompiledModel> {
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "companion".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "R".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "inflow".into(),
            stoichiometry: vec![StoichiometryEntry("R".into(), 1)],
            rate: Expr::Const(ConstExpr { value: 1.0 }),
            metadata: None,
            draw_method: DrawMethod::Deterministic,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![IrObs {
            name: "y".into(),
            source: "y".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: "y".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ObsColumn { name: "tested".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "y".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            covers: None,
            projection: Projection::CumulativeFlow("inflow".into()),
            projection_state_grad: Default::default(),
            likelihood,
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter { name: "dummy".into(), value: ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig { times: OutputSchedule::AtTimes(vec![0.0, 28.0]), trajectory: true, observations: false },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 28.0,
            dt: Some(1.0),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
            t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

fn density(likelihood: Likelihood, y: f64, projected: f64, aux: &[(String, f64)]) -> f64 {
    let compiled = model(likelihood);
    let obs = &compiled.model.observations[0];
    let f = sim::inference::obs_model::compile_obs_loglik_pf(obs, compiled.clone(), &compiled.default_params)
        .expect("the fixture resolves");
    f(y, projected, 7.0, &[0], aux)
}

#[test]
fn the_density_is_each_familys_at_the_resolved_arguments() {
    let aux = [("tested".to_string(), 40.0)];
    let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() < 1e-12 && a.is_finite(), "{what}: {a} vs {b}");
    close(density(Likelihood::Poisson(PoissonLikelihood { rate: projected() }), 5.0, 7.3, &[]), ll::poisson_logpmf(5.0, 7.3), "poisson");
    close(
        density(Likelihood::NegBinomial(NegBinomialLikelihood { mean: projected(), dispersion: constant(4.0) }), 5.0, 7.3, &[]),
        ll::negbin_logpmf(5.0, 7.3, 4.0),
        "neg_binomial",
    );
    close(
        density(Likelihood::Normal(NormalLikelihood { mean: projected(), sd: constant(2.0) }), 5.0, 7.3, &[]),
        ll::discretized_normal_logpmf(5.0, 7.3, 4.0),
        "normal",
    );
    close(
        density(Likelihood::Binomial(BinomialLikelihood { n: tested(), p: projected() }), 11.0, 0.3, &aux),
        ll::binom_logpmf(11, 40, 0.3),
        "binomial with its trials from the data column",
    );
    close(
        density(Likelihood::BetaBinomial(BetaBinomialLikelihood { n: tested(), alpha: constant(2.0), beta: constant(5.0) }), 11.0, 0.3, &aux),
        ll::beta_binomial_logpmf(11, 40, 2.0, 5.0),
        "beta_binomial with its trials from the data column",
    );
    close(
        density(Likelihood::Beta(BetaLikelihood { mean: projected(), concentration: constant(10.0) }), 0.25, 0.3, &[]),
        ll::beta_logpdf(0.25, 0.3, 10.0),
        "beta",
    );
    close(density(Likelihood::Bernoulli(BernoulliLikelihood { p: projected() }), 1.0, 0.3, &[]), ll::bernoulli_logpmf(1.0, 0.3), "bernoulli");
    close(
        density(
            Likelihood::ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood {
                mean: projected(),
                dispersion: constant(4.0),
                pi: constant(0.2),
            }),
            5.0,
            7.3,
            &[],
        ),
        ll::zi_negbin_logpmf(5.0, 7.3, 4.0, 0.2),
        "zero_inflated_neg_binomial",
    );
    // Without the data column a binomial has no trials: the density of a
    // positive count is zero.
    assert_eq!(density(Likelihood::Binomial(BinomialLikelihood { n: tested(), p: projected() }), 11.0, 0.3, &[]), f64::NEG_INFINITY);
}
