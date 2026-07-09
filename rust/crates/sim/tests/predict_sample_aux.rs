//! Regression: the posterior-predictive observation sampler must carry a
//! DATA-column binomial denominator forward.
//!
//! `fit predict`'s free-forward predictive drew `y_rep` from
//! `binomial(n = n_examined, p = projected)` where `n_examined` is a data
//! column. Before the fix `compile_obs_sample_pf` sampled with NO aux, so `n`
//! resolved to 0 and every draw was `binomial(0, p) = 0` — an all-zero
//! predictive band. The fix threads the CALLER's aux into the draw, so the
//! observed survey denominator is used: `y_rep ~ binomial(n_examined, p̂)`.
//!
//! This pins the sampler contract directly (no fit): given the aux the draw is
//! a real `binomial(n, p)`; given no aux it is 0 (the honest data-free value).

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{ConstExpr, Expr, ObsColumnRefExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::{
        BinomialLikelihood, Likelihood, ObsColumn, ObservationModel as IrObs,
        ObservationSchedule, Projection,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    ColumnRole, Diffable, Model,
};
use sim::{compiled_model::CompiledModel, rng::StatefulRng};

/// A one-compartment inflow model observed by `binomial(n = n_tested,
/// p = projected)`, where `n_tested` is a DATA column (an aux the caller
/// supplies at sample time). The projection is irrelevant here — the sampler
/// takes `projected` as a call argument.
fn binom_survey_model() -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "binom_survey".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment {
            name: "R".into(),
            kind: CompartmentKind::Integer,
        }],
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
            name: "prevalence".into(),
            source: "prevalence".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn {
                    name: "n_positive".into(),
                    role: ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
                ObsColumn {
                    name: "n_tested".into(),
                    role: ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "n_positive".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: Projection::CumulativeFlow("inflow".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Binomial(BinomialLikelihood {
                // n = the data column `n_tested` (an aux ObsColumnRef).
                n: Expr::ObsColumnRef(ObsColumnRefExpr {
                    obs_column_ref: "n_tested".into(),
                }),
                // p = projected (supplied as a call argument to the sampler).
                p: Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "dummy".into(),
            value: ParamValue::Fixed { value: 0.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 28.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 28.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(1),
            integrator: Default::default(),
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

#[test]
fn predictive_sampler_carries_data_column_denominator() {
    let compiled = binom_survey_model();
    let obs = &compiled.model.observations[0];
    let params = compiled.default_params.clone();
    let sampler =
        sim::inference::obs_model::compile_obs_sample_pf(obs, compiled.clone(), &params);

    let counts = vec![0_i64]; // one integer compartment (R)
    let p = 0.5_f64; // projected positivity
    let n_tested = 100.0_f64;

    // WITH the observed denominator: draws are Binomial(100, 0.5) — mean ~50,
    // never all-zero, and bounded by n_tested.
    let mut rng = StatefulRng::new(42);
    let with_aux: Vec<f64> = (0..400)
        .map(|_| sampler(p, 7.0, &counts, &[("n_tested".to_string(), n_tested)], &mut rng))
        .collect();
    let mean_with: f64 = with_aux.iter().sum::<f64>() / with_aux.len() as f64;
    assert!(
        with_aux.iter().all(|&y| (0.0..=n_tested).contains(&y)),
        "every draw must lie in [0, n_tested]"
    );
    assert!(
        with_aux.iter().any(|&y| y > 0.0),
        "the carried denominator must produce non-zero draws (was the all-zero bug)"
    );
    assert!(
        (mean_with - 50.0).abs() < 8.0,
        "Binomial(100, 0.5) mean ≈ 50, got {mean_with}"
    );

    // WITHOUT aux: no denominator available → Binomial(0, p) = 0. This is the
    // honest data-free behaviour (e.g. `simulate --obs` with no data file); it
    // is ALSO exactly the pre-fix predictive bug when the caller failed to
    // forward the observed aux.
    let mut rng0 = StatefulRng::new(42);
    let no_aux: Vec<f64> = (0..50)
        .map(|_| sampler(p, 7.0, &counts, &[], &mut rng0))
        .collect();
    assert!(
        no_aux.iter().all(|&y| y == 0.0),
        "no aux ⇒ denominator resolves to 0 ⇒ all draws 0, got {no_aux:?}"
    );
}
