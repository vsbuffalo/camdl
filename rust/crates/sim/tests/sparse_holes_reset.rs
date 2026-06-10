//! Sparse/holes: a HOLE must NOT suppress the incidence-accumulator reset.
//!
//! The fixed-bin (pomp `accumvars`) semantics require that the running
//! incidence tally resets on the OBSERVATION GRID, once per observation index
//! — regardless of whether that index carries an observed value or is a hole.
//! A missing week therefore still closes its weekly bin; it does NOT merge two
//! weeks of incidence into the next observed bin.
//!
//! This is a filter-loop property (`particle_filter.rs`: the reset at the end
//! of each obs-index iteration is unconditional, not gated on value presence).
//! We probe it behaviourally with a DETERMINISTIC incidence flow, so the
//! predicted incidence at a bin is an exact function of how many substeps'
//! flow accumulated into it. With a constant deterministic inflow of `K` per
//! unit time and a weekly (7-unit) grid:
//!   - reset fires at the hole  → week-(k+1) tally = 7·K (one week)
//!   - reset SKIPPED at the hole → week-(k+1) tally = 14·K (merged k + k+1)
//! Asserting the week-(k+1) prediction equals 7·K (not 14·K) pins the reset.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{ConstExpr, Expr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::{
        Likelihood, ObservationModel as IrObs, ObservationSchedule,
        NormalLikelihood, Projection,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        bootstrap_filter, dense_cells,
        ChainBinomialProcess, BoundObs, MultiStreamObsModel, ObsCell,
        traits::SMCConfig,
        multi_stream_obs::{StreamProjection, StreamSpec},
    },
};

/// Inflow `--> R @ deterministic(K)`, observed as `incidence` on it with a
/// Normal likelihood `mean = projected`. Deterministic so the flow per
/// substep is exactly `nearbyint(K·dt)` — no particle noise in the projection.
fn model(k_per_unit: f64) -> Arc<CompiledModel> {
    let m = Model {
        name: "sparse_holes_reset".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "inflow".into(),
                stoichiometry: vec![StoichiometryEntry("R".into(), 1)],
                rate: Expr::Const(ConstExpr { value: k_per_unit }),
                metadata: None,
                draw_method: DrawMethod::Deterministic,
                rate_grad: Default::default(),
                lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![
            IrObs {
                name: "cases".into(),
                schedule: ObservationSchedule::AtTimes(vec![]),
                projection: Projection::CumulativeFlow("inflow".into()),
                likelihood: Likelihood::Normal(NormalLikelihood {
                    // mean = projected (the weekly incidence tally)
                    mean: Expr::Projected(ProjectedExpr { projected: () }),
                    // a wide, constant sd so the likelihood is finite and
                    // benign; the prediction (`obs_mean`) is what we assert.
                    sd: Expr::Const(ConstExpr { value: 50.0 }),
                }),
            },
        ],
        bindings: vec![],
        parameters: vec![
            Parameter { name: "dummy".into(), value: ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0); h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 28.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 28.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

fn obs_model(compiled: Arc<CompiledModel>, cells: Vec<Option<ObsCell>>, obs_times: Vec<f64>) -> MultiStreamObsModel {
    let inflow = compiled.model.transitions.iter()
        .position(|t| t.name == "inflow").unwrap();
    let spec = StreamSpec {
        projection: StreamProjection::FlowSum(vec![inflow]),
        ir_model: compiled.model.observations[0].clone(),
        observations: cells,
        obs_times,
    };
    MultiStreamObsModel::new(BoundObs::bind(vec![spec]).unwrap().0, compiled).unwrap()
}

#[test]
fn hole_does_not_suppress_incidence_reset() {
    let k = 10.0; // deterministic inflow per unit time → 70/week at dt=1
    let compiled = model(k);
    let params = compiled.default_params.clone();
    let dt = 1.0;
    let weekly = 7.0 * k; // 70 per week

    // Weekly grid: 7, 14, 21, 28. Put a HOLE at week index 1 (t=14).
    let times = vec![7.0, 14.0, 21.0, 28.0];
    let holed_cells = vec![
        Some(ObsCell::Scalar(weekly)),
        None, // hole at t=14
        Some(ObsCell::Scalar(weekly)),
        Some(ObsCell::Scalar(weekly)),
    ];

    let process = ChainBinomialProcess::new(compiled.clone(), dt);
    let cfg = SMCConfig {
        n_particles: 4,
        dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        pf_wallclock_disabled: true,
    };

    let m_holed = obs_model(compiled.clone(), holed_cells, times.clone());
    let res = bootstrap_filter(&process, &m_holed, &params, &cfg, 7)
        .expect("holed pfilter must run");

    let preds = res.predictions.expect("incidence obs → predictions recorded");
    // Prediction at obs index 2 (t=21), the week AFTER the hole. If the reset
    // fired at the hole (t=14), this bin tallies ONLY week-3 flow = 70. If the
    // reset were (wrongly) gated on value-presence, the hole would skip its
    // reset and t=21 would carry weeks 2+3 merged = 140.
    let week_after_hole = preds[2].obs_mean;
    assert!((week_after_hole - weekly).abs() < 1e-6,
        "week after a hole must tally ONE week of incidence ({weekly}) — the hole's \
         reset must still fire. Got {week_after_hole} (merged-bin bug would give {}).",
        2.0 * weekly);

    // Sanity: the bin BEFORE the hole (t=7) is also one week.
    assert!((preds[0].obs_mean - weekly).abs() < 1e-6,
        "first bin must tally one week ({weekly}), got {}", preds[0].obs_mean);

    // And the loglik increment at the hole index (t=14) is finite and equals
    // the no-observation increment (log-sum-exp of all-zero weights − ln N = 0),
    // confirming the hole contributed no term while the filter still advanced.
    assert!((res.ll_increments[1] - 0.0).abs() < 1e-9,
        "hole index must contribute 0 to the loglik, got {}", res.ll_increments[1]);
}

/// Negative control / cross-check: with the SAME deterministic flow but a
/// DENSE series, the week-after prediction is identical (the reset fires every
/// week regardless), so the holed-vs-dense difference is ONLY the omitted term
/// at the hole — not a change in the latent tally.
#[test]
fn dense_baseline_matches_predictions_at_non_hole_indices() {
    let k = 10.0;
    let compiled = model(k);
    let params = compiled.default_params.clone();
    let dt = 1.0;
    let weekly = 7.0 * k;
    let times = vec![7.0, 14.0, 21.0, 28.0];

    let process = ChainBinomialProcess::new(compiled.clone(), dt);
    let cfg = SMCConfig {
        n_particles: 4, dt, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false,
        pf_wallclock_disabled: true,
    };

    let dense = obs_model(
        compiled.clone(),
        dense_cells(vec![weekly, weekly, weekly, weekly]),
        times.clone());
    let res = bootstrap_filter(&process, &dense, &params, &cfg, 7)
        .expect("dense pfilter must run");
    let preds = res.predictions.expect("predictions");

    for (i, p) in preds.iter().enumerate() {
        assert!((p.obs_mean - weekly).abs() < 1e-6,
            "dense bin {i} must tally one week ({weekly}), got {}", p.obs_mean);
    }
    // Every increment is finite (Normal density, deterministic mean = data).
    for (i, &inc) in res.ll_increments.iter().enumerate() {
        assert!(inc.is_finite(), "dense increment {i} must be finite, got {inc}");
    }
}
