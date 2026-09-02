//! gh#742: the complete-data observation term, resolved by declared stream.
//!
//! `complete_data_loglik` reports `observation` — the observation log-density
//! summed over every stream — and, alongside it, `observation_per_stream`: the
//! same terms kept separate, one entry per declared observation block. A fit
//! with several streams can then be asked WHICH stream it is straining against
//! without re-running the filter.
//!
//! Two properties are pinned here, at full `f64` precision (the on-disk trace
//! writes 4 decimals, so the CLI test `pgas_trace_obs_ll_per_stream.rs` can only
//! check agreement to that; this file is the precise one):
//!
//! 1. **The decomposition adds up.** `Σ observation_per_stream` equals
//!    `observation` to floating-point reassociation — the two sum the same
//!    terms in different orders (`observation` is time-major, the per-stream
//!    vector is stream-major), so they agree to round-off, not bitwise.
//!
//! 2. **A stream's entry is that stream's own likelihood.** The oracle is an
//!    INDEPENDENT fit of each stream on its own: scoring the same trajectory at
//!    the same parameters against a one-stream observation model built from only
//!    that stream must reproduce, to round-off, the entry the two-stream run
//!    reports for it. That is the check that matters under multi-cadence — the
//!    one-stream model has none of its sibling's observation times in its union
//!    axis, so an implementation that accumulated per union index, or reset a
//!    stream's running total when a sibling was scored, disagrees.
//!
//! The fixture is deliberately multi-cadence: `afp` every 30 days against `es`
//! every 14 days, both projecting the same deterministic `inflow` transition.
//! Their observed values differ (300 vs 100 against a true 140), so the two
//! streams' contributions are far apart and a broadcast bug — writing the joint
//! into every slot, or one stream's value into both — cannot pass by
//! coincidence.

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
        dense_cells, BoundObs, MultiStreamObsModel,
        multi_stream_obs::{StreamProjection, StreamSpec},
        particle_filter::Observation,
        pgas::{build_obs_at_substep, complete_data_loglik, PGASTrajectory},
    },
    rng::StatefulRng,
};

const DT: f64 = 1.0;
const T_END: f64 = 90.0;
/// Deterministic inflow per unit time, so the reference path carries no noise.
const K: f64 = 10.0;

const AFP_TIMES: [f64; 3] = [30.0, 60.0, 90.0];
const ES_TIMES: [f64; 6] = [14.0, 28.0, 42.0, 56.0, 70.0, 84.0];
/// `afp` is observed at its true 30-day bin (30·K) — residual zero.
const AFP_OBSERVED: f64 = 300.0;
/// `es` is observed well below its true 14-day bin (14·K = 140) — a residual of
/// 40, which pushes its per-observation density clearly below `afp`'s.
const ES_OBSERVED: f64 = 100.0;

fn ir_incidence_obs(name: &str) -> IrObs {
    IrObs {
        name: name.into(),
        source: name.into(),
        columns: vec![
            ir::observation::ObsColumn {
                name: "time".into(),
                role: ir::observation::ColumnRole::Time,
            },
            ir::observation::ObsColumn {
                name: name.into(),
                role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("inflow".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Normal(NormalLikelihood {
            mean: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
            sd: ir::Diffable::new(Expr::Const(ConstExpr { value: 50.0 })),
        }),
    }
}

/// `--> R @ deterministic(K)`, observed by two incidence blocks (`afp`, `es`).
fn model() -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "pgas_obs_ll_per_stream".into(),
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
            rate: Expr::Const(ConstExpr { value: K }),
            metadata: None,
            draw_method: DrawMethod::Deterministic,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![ir_incidence_obs("afp"), ir_incidence_obs("es")],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "dummy".into(),
            value: ParamValue::Fixed { value: 0.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, T_END]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            time_semantics: "continuous".into(),
            dt: Some(DT),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
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

/// One `StreamSpec` per observation block, on that block's own cadence.
/// `which` indexes `model.observations` (0 = afp, 1 = es).
fn spec(compiled: &CompiledModel, which: usize, times: &[f64], observed: f64) -> StreamSpec {
    let inflow = compiled
        .model
        .transitions
        .iter()
        .position(|t| t.name == "inflow")
        .unwrap();
    StreamSpec::dense(
        StreamProjection::FlowSum(vec![inflow]),
        compiled.model.observations[which].clone(),
        dense_cells(vec![observed; times.len()]),
        times.to_vec(),
    )
}

fn bind(compiled: Arc<CompiledModel>, specs: Vec<StreamSpec>) -> MultiStreamObsModel {
    MultiStreamObsModel::new(BoundObs::bind(specs).expect("bind streams").0, compiled).unwrap()
}

/// The complete-data components of `trajectory` under `obs` at `params`, with
/// the substep→observation map built from `obs`'s own union axis.
fn score(
    compiled: &Arc<CompiledModel>,
    trajectory: &PGASTrajectory,
    obs: &MultiStreamObsModel,
    times: &[f64],
) -> sim::inference::pgas::LogLikComponents {
    let observations: Vec<Observation> =
        times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, DT)
        .expect("substep map");
    let params = compiled.default_params.clone();
    complete_data_loglik(compiled, trajectory, &params, &observations, DT, obs, &oas)
        .expect("complete_data_loglik")
}

/// The sorted-unique merge of the two cadences — the union axis `bind` produces.
fn union_times() -> Vec<f64> {
    let mut v: Vec<f64> = AFP_TIMES.iter().chain(ES_TIMES.iter()).copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

#[test]
fn per_stream_obs_ll_sums_to_obs_ll_and_matches_each_stream_scored_alone() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(11);
    let trajectory =
        sim::inference::pgas::simulate_reference(&compiled, &params, T_END, DT, sim::rng::BinomialAlgorithm::default(), &mut rng)
            .expect("reference trajectory");

    // The two-stream, two-cadence fit.
    let joint = bind(
        compiled.clone(),
        vec![
            spec(&compiled, 0, &AFP_TIMES, AFP_OBSERVED),
            spec(&compiled, 1, &ES_TIMES, ES_OBSERVED),
        ],
    );
    let union = union_times();
    let both = score(&compiled, &trajectory, &joint, &union);

    assert_eq!(
        both.observation_per_stream.len(),
        2,
        "one entry per DECLARED stream, not per scored observation or per union index",
    );
    for (i, v) in both.observation_per_stream.iter().enumerate() {
        assert!(
            v.is_finite() && *v != 0.0,
            "stream {i} must carry its own non-zero contribution; a zero here is a \
             stream whose terms were never accumulated. Got {:?}",
            both.observation_per_stream,
        );
    }
    assert_ne!(
        both.observation_per_stream[0], both.observation_per_stream[1],
        "the two streams score different data against different cadences, so equal \
         entries mean one value was broadcast into both slots: {:?}",
        both.observation_per_stream,
    );

    // (1) The decomposition adds up. Different summation order ⇒ round-off,
    // not bitwise.
    let summed: f64 = both.observation_per_stream.iter().sum();
    let tol = 1e-9 * both.observation.abs().max(1.0);
    assert!(
        (summed - both.observation).abs() <= tol,
        "Σ observation_per_stream = {summed} must equal observation = {} to \
         floating-point round-off (|Δ| = {:.3e}, tol {:.3e})",
        both.observation,
        (summed - both.observation).abs(),
        tol,
    );

    // (2) Each entry equals that stream scored ON ITS OWN — same trajectory,
    // same parameters, a one-stream observation model whose union axis holds
    // only that stream's times. Under multi-cadence this is the discriminating
    // check: `afp`'s 30-day windows span two `es`-only union times in the joint
    // fit and none at all in the solo fit.
    let afp_alone = bind(
        compiled.clone(),
        vec![spec(&compiled, 0, &AFP_TIMES, AFP_OBSERVED)],
    );
    let solo_afp = score(&compiled, &trajectory, &afp_alone, &AFP_TIMES);
    let es_alone = bind(
        compiled.clone(),
        vec![spec(&compiled, 1, &ES_TIMES, ES_OBSERVED)],
    );
    let solo_es = score(&compiled, &trajectory, &es_alone, &ES_TIMES);

    for (name, joint_v, solo) in [
        ("afp", both.observation_per_stream[0], solo_afp.observation),
        ("es", both.observation_per_stream[1], solo_es.observation),
    ] {
        let tol = 1e-9 * solo.abs().max(1.0);
        assert!(
            (joint_v - solo).abs() <= tol,
            "obs_ll_{name} from the two-stream fit ({joint_v}) must equal `{name}` \
             scored alone ({solo}) — |Δ| = {:.3e}, tol {:.3e}. A difference means the \
             per-stream accumulation is contaminated by the sibling's cadence.",
            (joint_v - solo).abs(),
            tol,
        );
    }

    // Non-vacuity: the two solo scores are far apart, so the assertions above
    // cannot both pass under a swap or a broadcast.
    assert!(
        (solo_afp.observation - solo_es.observation).abs() > 1.0,
        "the fixture must separate the streams by more than round-off: afp = {}, es = {}",
        solo_afp.observation,
        solo_es.observation,
    );
}

/// A model declaring ONE stream reports exactly one entry, equal to
/// `observation` — the degenerate case a decomposition must not get wrong.
#[test]
fn single_stream_reports_one_entry_equal_to_obs_ll() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(11);
    let trajectory =
        sim::inference::pgas::simulate_reference(&compiled, &params, T_END, DT, sim::rng::BinomialAlgorithm::default(), &mut rng)
            .expect("reference trajectory");

    let only = bind(
        compiled.clone(),
        vec![spec(&compiled, 0, &AFP_TIMES, AFP_OBSERVED)],
    );
    let c = score(&compiled, &trajectory, &only, &AFP_TIMES);

    assert_eq!(c.observation_per_stream.len(), 1, "one declared stream ⇒ one entry");
    assert!(
        c.observation.is_finite() && c.observation != 0.0,
        "the fixture must produce a real observation term, got {}",
        c.observation,
    );
    let tol = 1e-9 * c.observation.abs().max(1.0);
    assert!(
        (c.observation_per_stream[0] - c.observation).abs() <= tol,
        "with one stream the single entry IS obs_ll: {} vs {}",
        c.observation_per_stream[0],
        c.observation,
    );
}

/// A model with no observation block at all: an empty per-stream vector, and an
/// `observation` term of zero. Pins that the decomposition's length tracks the
/// DECLARED stream count rather than defaulting to one.
#[test]
fn no_streams_reports_an_empty_decomposition() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(11);
    let trajectory =
        sim::inference::pgas::simulate_reference(&compiled, &params, T_END, DT, sim::rng::BinomialAlgorithm::default(), &mut rng)
            .expect("reference trajectory");

    let empty = MultiStreamObsModel::empty(compiled.clone());
    let c = score(&compiled, &trajectory, &empty, &[]);
    assert!(c.observation_per_stream.is_empty(), "no streams ⇒ no entries");
    assert_eq!(c.observation, 0.0, "no streams ⇒ no observation term");
}
