//! Multi-cadence per-stream incidence reset (Phase 2a, "Option Z").
//!
//! An incidence (`Interval`) stream is scored against the flow accumulated
//! since ITS OWN last observation. With two streams on DIFFERENT cadences
//! sharing one transition — AFP at every 30 days, ES at every 14 days — the
//! blanket per-observation reset is WRONG: an ES-only union time would zero
//! AFP's monthly bin, so AFP would tally only the flow since the last ES
//! observation instead of its full 30-day window.
//!
//! Phase 2a fixes this with a per-stream persistent bin (`ParticleState.acc` /
//! the PGAS `acc` buffer), folded once per observation interval and reset
//! PER-STREAM (only the streams scheduled at the current union index). This
//! file pins that property on the shared seam every filter routes through —
//! `MultiStreamObsModel::{fold_into_acc, reset_due_acc, n_interval_streams}` —
//! plus an end-to-end filter smoke test.
//!
//! The CLI still loud-rejects heterogeneous cadences end-to-end (Phase 2b
//! opens them); this test bypasses that by building the obs model DIRECTLY via
//! `bind` of two `StreamSpec`s on different cadences (`bind` merges them onto
//! the union axis — the Phase 1 substrate).
//!
//! Determinism: a single deterministic `inflow @ deterministic(K)` transition,
//! so the per-substep flow is exactly `nearbyint(K·dt)` with no particle noise.

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
        ChainBinomialProcess, BoundObs, MultiStreamObsModel,
        traits::{ObservationModel, SMCConfig},
        multi_stream_obs::{StreamProjection, StreamSpec},
    },
};

/// The union observation axis, read off the obs model via the trait.
fn union_axis(obs: &MultiStreamObsModel) -> Vec<f64> {
    (0..ObservationModel::<sim::inference::ParticleState>::n_observations(obs))
        .map(|i| ObservationModel::<sim::inference::ParticleState>::obs_time(obs, i))
        .collect()
}

/// `--> R @ deterministic(K)`, with a single `inflow` transition observed as
/// `incidence` (a `Normal(mean = projected)` likelihood). Deterministic so the
/// flow per substep is exactly `nearbyint(K·dt)`.
fn model(k_per_unit: f64) -> Arc<CompiledModel> {
    let m = Model {
        name: "per_stream_reset".into(),
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
        // Two observation blocks, both `incidence(inflow)` — one per stream.
        observations: vec![
            ir_incidence_obs("afp"),
            ir_incidence_obs("es"),
        ],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "dummy".into(), value: ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0); h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 90.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 90.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

fn ir_incidence_obs(name: &str) -> IrObs {
    IrObs {
        name: name.into(),
        source: name.into(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn { name: name.into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("inflow".into()),
        likelihood: Likelihood::Normal(NormalLikelihood {
            mean: Expr::Projected(ProjectedExpr { projected: () }),
            sd: Expr::Const(ConstExpr { value: 50.0 }),
        }),
    }
}

/// Build a 2-stream multi-cadence obs model DIRECTLY via `bind` (bypassing the
/// CLI homogeneous-schedule guard). Both streams project the same `inflow`
/// transition; `afp_times` and `es_times` are their own (different) grids.
fn multi_cadence_obs(
    compiled: Arc<CompiledModel>,
    afp_times: Vec<f64>,
    es_times: Vec<f64>,
) -> MultiStreamObsModel {
    let inflow = compiled.model.transitions.iter()
        .position(|t| t.name == "inflow").unwrap();
    let afp = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![0.0; afp_times.len()]), // values irrelevant: we read predictions / bins
        afp_times,
    );
    let es = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[1].clone(),
        dense_cells(vec![0.0; es_times.len()]),
        es_times,
    );
    MultiStreamObsModel::new(BoundObs::bind(vec![afp, es]).expect("bind multi-cadence").0, compiled).unwrap()
}

/// (i) + (ii): the per-stream fold/reset seam directly.
///
/// Drive the exact `acc` lifecycle the five filter paths run — fold per
/// observation interval, score, reset_due — over a KNOWN per-transition flow
/// per interval, and assert each stream's bin closes on ITS OWN cadence.
///
/// AFP at {30,60,90}, ES at {14,28,42,56,70,84}; union axis is the sorted
/// merge. Constant flow `K` per unit time ⇒ over a union interval of width `w`
/// the per-transition `cum_flows` increment is `w·K`. AFP's bin at t=30 must be
/// the FULL `(0,30] = 30·K`; ES's bin at t=14 is `(0,14] = 14·K`, at t=28 is
/// `(14,28] = 14·K`, etc.
#[test]
fn per_stream_acc_closes_on_own_cadence_not_union_step() {
    let k = 10.0;
    let compiled = model(k);
    let afp_times = vec![30.0, 60.0, 90.0];
    let es_times = vec![14.0, 28.0, 42.0, 56.0, 70.0, 84.0];
    let obs = multi_cadence_obs(compiled, afp_times.clone(), es_times.clone());

    // Two Interval streams ⇒ two acc slots (AFP=slot 0, ES=slot 1; stream order).
    assert_eq!(obs.n_interval_streams(), 2,
        "two incidence streams ⇒ two per-stream acc bins");

    // The union axis (sorted-unique merge).
    let union: Vec<f64> = union_axis(&obs);
    assert_eq!(union, vec![14.0, 28.0, 30.0, 42.0, 56.0, 60.0, 70.0, 84.0, 90.0]);

    // Walk the union exactly as a filter does. `acc[0]` = AFP, `acc[1]` = ES.
    // `cum_flows` has one transition (`inflow`); its per-interval increment is
    // `(union[i] - prev) * k`.
    let n_tr = 1usize;
    let mut acc = vec![0u64; obs.n_interval_streams()];

    let mut prev = 0.0_f64;
    let mut afp_bins: Vec<u64> = Vec::new();
    let mut es_bins: Vec<u64> = Vec::new();
    for (ui, &ut) in union.iter().enumerate() {
        let width = ut - prev;
        let inc = (width * k).round() as u64;
        // The filter's per-substep accumulation collapses (deterministic) to one
        // per-interval increment into the per-transition tally.
        let cum_flows = vec![inc; n_tr];

        // FOLD: close this interval's flow into every Interval stream's acc.
        obs.fold_into_acc(&cum_flows, &mut acc);

        // SCORE (read the per-stream bin) at any stream scheduled here.
        if afp_times.contains(&ut) {
            afp_bins.push(acc[0]);
        }
        if es_times.contains(&ut) {
            es_bins.push(acc[1]);
        }

        // RESET per-stream: only the streams scheduled at THIS union index.
        obs.reset_due_acc(ui, &mut acc);
        prev = ut;
    }

    // (i) AFP's scored bins are each ONE 30-day window = 300, NOT a 14-day
    // slice (140) nor a merged span. The first bin (0,30] crosses the ES-only
    // union times 14 and 28 — proving the ES resets did NOT touch AFP's acc.
    assert_eq!(afp_bins, vec![300, 300, 300],
        "each AFP bin must tally its full 30-day window (30·K = 300); an ES-only \
         union time must NOT reset AFP — got {afp_bins:?}");

    // ES's bins are each one 14-day window = 140.
    assert_eq!(es_bins, vec![140, 140, 140, 140, 140, 140],
        "each ES bin must tally its 14-day window (14·K = 140) — got {es_bins:?}");
}

/// (ii) MUTATION CHECK, made executable: if the reset were BLANKET (zero all
/// acc at every union index, the pre-Phase-2a behaviour) instead of per-stream,
/// AFP's bin would collapse to only the flow since the last union time — far
/// short of its 30-day window. This re-runs the same walk with a blanket reset
/// and asserts it gives the WRONG (too-small) AFP bin, so the per-stream test
/// above is non-vacuous.
#[test]
fn blanket_reset_would_undercount_afp_mutation_guard() {
    let k = 10.0;
    let compiled = model(k);
    let afp_times = vec![30.0, 60.0, 90.0];
    let es_times = vec![14.0, 28.0, 42.0, 56.0, 70.0, 84.0];
    let obs = multi_cadence_obs(compiled, afp_times.clone(), es_times.clone());
    let union: Vec<f64> = union_axis(&obs);

    let mut acc = vec![0u64; obs.n_interval_streams()];
    let mut prev = 0.0_f64;
    let mut afp_bins: Vec<u64> = Vec::new();
    for (_ui, &ut) in union.iter().enumerate() {
        let inc = ((ut - prev) * k).round() as u64;
        obs.fold_into_acc(&vec![inc; 1], &mut acc);
        if afp_times.contains(&ut) {
            afp_bins.push(acc[0]);
        }
        // BLANKET reset (the bug): zero EVERY acc bin at EVERY union index.
        for a in &mut acc { *a = 0; }
        prev = ut;
    }

    // The first AFP bin at t=30 would tally only (28,30] = 2·K = 20 (the flow
    // since the previous union time t=28), NOT the correct (0,30] = 300. This
    // is exactly the undercount the per-stream reset prevents.
    assert_eq!(afp_bins[0], 20,
        "under a blanket reset the AFP bin collapses to (28,30] = 20 (the \
         undercount Phase 2a fixes); got {afp_bins:?}");
    assert_ne!(afp_bins[0], 300,
        "blanket reset must NOT produce the correct 30-day bin — that would \
         make the per-stream guard vacuous");
}

/// Homogeneous (both streams on the SAME cadence): the per-stream acc reset is
/// indistinguishable from the old blanket reset — both bins close every
/// interval. This is the bit-identity guard at the seam level (the full-filter
/// bit-identity is the existing he2010 / sparse_holes / parity suite).
#[test]
fn homogeneous_acc_equals_blanket() {
    let k = 10.0;
    let compiled = model(k);
    let times = vec![7.0, 14.0, 21.0, 28.0];
    let obs = multi_cadence_obs(compiled, times.clone(), times.clone());
    let union: Vec<f64> = union_axis(&obs);
    // Homogeneous ⇒ the union equals the shared axis (no new union times).
    assert_eq!(union, times);
    assert_eq!(obs.n_interval_streams(), 2);

    // Per-stream walk.
    let mut acc = vec![0u64; 2];
    let mut prev = 0.0_f64;
    let mut bins: Vec<(u64, u64)> = Vec::new();
    for (ui, &ut) in union.iter().enumerate() {
        let inc = ((ut - prev) * k).round() as u64;
        obs.fold_into_acc(&vec![inc; 1], &mut acc);
        bins.push((acc[0], acc[1]));
        obs.reset_due_acc(ui, &mut acc);
        prev = ut;
    }

    // Every interval is one 7-day window for BOTH streams (= 70), since both are
    // scheduled at every union index and reset together — identical to the
    // blanket reset.
    for (i, &(a, e)) in bins.iter().enumerate() {
        assert_eq!((a, e), (70, 70),
            "homogeneous: bin {i} must be one 7-day window for both streams (70,70); got {:?}",
            (a, e));
    }
}

/// End-to-end smoke: the bootstrap filter runs to completion on the
/// multi-cadence obs model (exercising the real fold + reset_due_acc +
/// resample-copy code path in `particle_filter.rs`, not just the seam) and
/// returns a finite log-likelihood. The CLI guard would reject this config
/// upstream; the filter itself accepts a directly-bound union obs model.
#[test]
fn bootstrap_filter_runs_multi_cadence() {
    let k = 10.0;
    let compiled = model(k);
    let afp_times = vec![30.0, 60.0, 90.0];
    let es_times = vec![14.0, 28.0, 42.0, 56.0, 70.0, 84.0];
    // Use the true 30-/14-day bins as the observed values so the Normal
    // likelihood is finite and benign.
    let inflow = compiled.model.transitions.iter()
        .position(|t| t.name == "inflow").unwrap();
    let afp = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![300.0; afp_times.len()]),
        afp_times,
    );
    let es = StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[1].clone(),
        dense_cells(vec![140.0; es_times.len()]),
        es_times,
    );
    let obs = MultiStreamObsModel::new(
        BoundObs::bind(vec![afp, es]).expect("bind").0, compiled.clone()).unwrap();

    let process = ChainBinomialProcess::new(compiled.clone());
    let cfg = SMCConfig {
        n_particles: 8, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let params = compiled.default_params.clone();
    let res = bootstrap_filter(&process, &obs, &params, &cfg, 7)
        .expect("multi-cadence pfilter must run");
    assert!(res.log_likelihood.is_finite(),
        "multi-cadence loglik must be finite, got {}", res.log_likelihood);
    // The deterministic flow makes each scored bin equal its observed value, so
    // every increment is finite (Normal density at the mode).
    for (i, &inc) in res.ll_increments.iter().enumerate() {
        assert!(inc.is_finite(), "increment {i} must be finite, got {inc}");
    }
}
