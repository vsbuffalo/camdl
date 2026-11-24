//! D79: a held-out value is a distinct cell, `ObsCell::HeldOut`, that the
//! bootstrap filter never learns from and the scoring likelihood evaluates.
//!
//! The invariant pinned here: for a fixed seed, changing the held-out values,
//! or replacing them by `NA`, leaves every output of the filter bit-identical
//! — the log-likelihood, its increments, the ESS and log-weight-variance
//! traces, the final particles, the per-particle joint and per-stream
//! log-likelihoods and the per-stream predictive draws (which consume the
//! diagnostic random streams) — and changes only the per-particle held-out
//! log-likelihoods the scorer reads.
//!
//! The held-out log-likelihood is the fitting likelihood (D56): at the first
//! held-out stop, the value a filter records for a held-out cell equals, bit
//! for bit, the per-stream log-likelihood a filter records when the same value
//! is an observed cell there, on the same particles before anything is
//! assimilated at that stop.
//!
//! The fixture is a closed SIR on the chain-binomial backend with two
//! incidence streams on different irregular cadences: `weekly` (negative
//! binomial) and `aux` (Poisson). Held-out cells sit at a stop where the other
//! stream assimilates an observed value in the same joint update (t = 42: aux
//! held out, weekly observed), at stops where the other stream is not
//! scheduled (t = 30 for aux, t = 35 for weekly), and on trailing periods
//! after the last observation (t = 67, 74, 81), the shape a forecast's targets
//! take.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        Likelihood, NegBinomialLikelihood, ObservationModel as IrObs, ObservationSchedule,
        PoissonLikelihood, Projection,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::{
    bootstrap_filter, bootstrap_filter_recording_at,
    multi_stream_obs::{MultiStreamObsModel, Period, StreamProjection, StreamSpec},
    obs_attempt::ObsCellState,
    particle_filter::PFilterResult,
    traits::{ObservationModel, SMCConfig},
    BoundObs, ChainBinomialProcess, ObsCell, PrequentialStops,
};

const DT: f64 = 1.0;
const N_PARTICLES: usize = 200;
const SEEDS: [u64; 3] = [1, 7, 20261117];
const T_END: f64 = 90.0;

fn p(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn c(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(a), right: Box::new(b) } })
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Div, left: Box::new(a), right: Box::new(b) } })
}
fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}
fn transition(name: &str, sto: Vec<StoichiometryEntry>, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(), stoichiometry: sto, rate, metadata: None,
        draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
    }
}
fn incidence_block(name: &str, likelihood: Likelihood) -> IrObs {
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
        covers: None,
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood,
    }
}
fn projected() -> Expr {
    Expr::Projected(ProjectedExpr { projected: () })
}

/// Closed SIR (N = 10 000) with two incidence streams on the infection flow.
fn sir() -> Arc<CompiledModel> {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "d79_held_out_cells".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: ["S", "I", "R"].iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![
            transition("infection",
                vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                div(mul(mul(p("beta"), c("S")), c("I")), n)),
            transition("recovery",
                vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry("R".into(), 1)],
                mul(p("gamma"), c("I"))),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![
            incidence_block("weekly", Likelihood::NegBinomial(NegBinomialLikelihood {
                mean: ir::Diffable::new(mul(p("rho"), projected())),
                dispersion: ir::Diffable::new(p("k")),
            })),
            incidence_block("aux", Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(mul(p("rho"), projected())),
            })),
        ],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![param("beta", 0.4), param("gamma", 0.15), param("rho", 0.6), param("k", 10.0)],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), 9990.0);
            h.insert("I".into(), 10.0);
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, T_END]),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            dt: Some(DT),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None, t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).unwrap())
}

/// What the held-out cells of the fixture hold.
#[derive(Clone, Copy, Debug)]
enum HeldOutValues {
    /// `NA`: the rows the filter is handed carry no value at all.
    Missing,
    /// Held-out cells, each value scaled by this factor.
    Scaled(f64),
}

/// The two streams. `weekly`'s rows at t = 35 and its trailing periods at
/// t = 67, 74, 81 are held out; `aux`'s row at t = 30, where `weekly` is not
/// scheduled, is held out, and so is its row at t = 42, where `weekly`'s
/// observed row is assimilated in the same joint update.
fn bind(compiled: &Arc<CompiledModel>, held: HeldOutValues) -> MultiStreamObsModel {
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let cell = |v: f64, held_out: bool| -> Option<ObsCell> {
        match (held_out, held) {
            (false, _) => Some(ObsCell::Scalar(v)),
            (true, HeldOutValues::Missing) => None,
            (true, HeldOutValues::Scaled(f)) => Some(ObsCell::HeldOut((v * f).round())),
        }
    };
    let w_ends = [7.0, 12.0, 21.0, 23.0, 35.0, 42.0, 50.0, 60.0, 67.0, 74.0, 81.0];
    let w_vals = [20.0, 150.0, 260.0, 300.0, 1500.0, 900.0, 400.0, 150.0, 60.0, 30.0, 12.0];
    let w_held = [false, false, false, false, true, false, false, false, true, true, true];
    let a_ends = [3.0, 9.0, 12.0, 17.0, 30.0, 33.0, 42.0, 55.0, 60.0];
    let a_vals = [2.0, 30.0, 70.0, 110.0, 1100.0, 1200.0, 700.0, 250.0, 60.0];
    let a_held = [false, false, false, false, true, false, true, false, false];
    let specs = vec![
        StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
            compiled.model.observations[0].clone(),
            w_vals.iter().zip(w_held).map(|(&v, h)| cell(v, h)).collect(),
            Period::contiguous(0.0, &w_ends).unwrap()),
        StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
            compiled.model.observations[1].clone(),
            a_vals.iter().zip(a_held).map(|(&v, h)| cell(v, h)).collect(),
            Period::contiguous(0.0, &a_ends).unwrap()),
    ];
    let (bound, report) = BoundObs::bind(0.0, specs).expect("fixture binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
}

fn smc() -> SMCConfig {
    SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, path_recording: Default::default(),
        record_prequential: true,
        record_predictions: false,
        record_stream_means: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn index_of(obs: &MultiStreamObsModel, t: f64) -> usize {
    obs.obs_times().iter().position(|&u| u == t)
        .unwrap_or_else(|| panic!("no stop at t = {t} in {:?}", obs.obs_times()))
}

/// Every output of the filter apart from the held-out log-likelihoods.
fn assert_same_filter(a: &PFilterResult, b: &PFilterResult, what: &str) {
    assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits(),
        "{what}: log-likelihood {} vs {}", a.log_likelihood, b.log_likelihood);
    assert_eq!(bits(&a.ll_increments), bits(&b.ll_increments), "{what}: log-likelihood increments");
    assert_eq!(bits(&a.ess_trace), bits(&b.ess_trace), "{what}: ESS trace");
    assert_eq!(bits(&a.logw_var_trace), bits(&b.logw_var_trace), "{what}: log-weight variance trace");
    let fs = |r: &PFilterResult| r.final_states.as_ref().unwrap().iter()
        .map(|s| (s.counts.clone(), s.flow_accumulators.clone(), s.acc.clone()))
        .collect::<Vec<_>>();
    assert_eq!(fs(a), fs(b), "{what}: final particles (the filtered state)");
    let (pa, pb) = (a.prequential.as_ref().unwrap(), b.prequential.as_ref().unwrap());
    assert_eq!(pa.log_liks.len(), pb.log_liks.len());
    for i in 0..pa.log_liks.len() {
        let t = pa.obs_times[i];
        assert_eq!(bits(&pa.log_liks[i]), bits(&pb.log_liks[i]), "{what}: joint log-liks at t = {t}");
        assert_eq!(pa.per_stream_log_liks[i].len(), pb.per_stream_log_liks[i].len());
        for si in 0..pa.per_stream_log_liks[i].len() {
            assert_eq!(bits(&pa.per_stream_log_liks[i][si]), bits(&pb.per_stream_log_liks[i][si]),
                "{what}: per-stream log-liks, stream {si}, t = {t}");
            assert_eq!(bits(&pa.per_stream_samples[i][si]), bits(&pb.per_stream_samples[i][si]),
                "{what}: predictive draws, stream {si}, t = {t} (a difference means a random \
                 stream was drawn differently)");
        }
    }
}

/// The held-out log-likelihoods at each held-out (stream, stop).
fn held_out_cells(obs: &MultiStreamObsModel) -> Vec<(usize, usize)> {
    vec![
        (1, index_of(obs, 30.0)),
        (1, index_of(obs, 42.0)),
        (0, index_of(obs, 35.0)),
        (0, index_of(obs, 67.0)),
        (0, index_of(obs, 74.0)),
        (0, index_of(obs, 81.0)),
    ]
}

/// The headline: held-out values move nothing but the held-out
/// log-likelihoods, which they do move.
#[test]
fn held_out_values_change_only_the_held_out_log_likelihoods() {
    let compiled = sir();
    let process = ChainBinomialProcess::new(compiled.clone());
    let params = &compiled.default_params;
    let missing = bind(&compiled, HeldOutValues::Missing);
    let a = bind(&compiled, HeldOutValues::Scaled(1.0));
    let b = bind(&compiled, HeldOutValues::Scaled(1.7));
    let cells = held_out_cells(&a);

    // The fixture is not vacuous: each held-out stop is a held-out cell to
    // the model and not an assimilation event unless another stream observes
    // there, and the run assimilates elsewhere.
    for &(si, ui) in &cells {
        assert!(matches!(a.cell_state(si, ui), ObsCellState::HeldOut { .. }), "{:?}", a.cell_state(si, ui));
        assert_eq!(missing.cell_state(si, ui), ObsCellState::Hole);
        assert_eq!(a.assimilates_at(ui), missing.assimilates_at(ui),
            "a held-out cell never makes a stop an assimilation event");
        assert!(a.holds_held_out_at(ui) && !missing.holds_held_out_at(ui));
    }
    assert!(!a.assimilates_at(index_of(&a, 35.0)) && !a.assimilates_at(index_of(&a, 67.0)));
    // t = 42: one stream observed, the other held out, in one joint update.
    let joint = index_of(&a, 42.0);
    assert!(a.assimilates_at(joint) && a.holds_held_out_at(joint));
    assert!(matches!(a.cell_state(0, joint), ObsCellState::Scored { .. }));
    assert!(matches!(a.cell_state(1, joint), ObsCellState::HeldOut { .. }));

    for seed in SEEDS {
        let what = format!("seed {seed}");
        let rm = bootstrap_filter(&process, &missing, params, &smc(), seed).unwrap();
        let ra = bootstrap_filter(&process, &a, params, &smc(), seed).unwrap();
        let rb = bootstrap_filter(&process, &b, params, &smc(), seed).unwrap();
        assert!(ra.log_likelihood.is_finite(), "{what}: the fixture filters");
        assert_same_filter(&ra, &rm, &format!("{what}: held out vs NA"));
        assert_same_filter(&ra, &rb, &format!("{what}: two sets of held-out values"));

        let (pm, pa, pb) = (rm.prequential.unwrap(), ra.prequential.unwrap(), rb.prequential.unwrap());
        assert!(pm.per_stream_held_out_log_liks.iter().all(Vec::is_empty),
            "{what}: with no held-out cell, no scoring likelihood is recorded");
        for &(si, ui) in &cells {
            let t = pa.obs_times[ui];
            let (la, lb) = (&pa.per_stream_held_out_log_liks[ui], &pb.per_stream_held_out_log_liks[ui]);
            assert_eq!((la.len(), la[si].len()), (2, N_PARTICLES), "{what}: held-out record at t = {t}");
            assert!(la[si].iter().all(|x| x.is_finite() && *x < 0.0),
                "{what}: held-out log-likelihoods at t = {t} are finite: {:?}", &la[si][..4]);
            assert_ne!(bits(&la[si]), bits(&lb[si]), "{what}: the held-out value moves its score at t = {t}");
            // The other stream holds no held-out value at this stop: NaN, never 0.
            assert!(la[1 - si].iter().all(|x| x.is_nan()), "{what}: no held-out value, no score, at t = {t}");
        }
    }
}

/// The record a forecast reads is kept at the selected stops, bit-identical
/// to a full recording's there (D84 with held-out cells).
#[test]
fn selected_stops_keep_the_full_recordings_held_out_log_likelihoods() {
    let compiled = sir();
    let process = ChainBinomialProcess::new(compiled.clone());
    let params = &compiled.default_params;
    let a = bind(&compiled, HeldOutValues::Scaled(1.0));
    let cells = held_out_cells(&a);
    let keep: Vec<usize> = cells.iter().map(|&(_, ui)| ui).skip(2).collect();
    for seed in SEEDS {
        let full = bootstrap_filter(&process, &a, params, &smc(), seed).unwrap();
        let sel = bootstrap_filter_recording_at(&process, &a, params, &smc(), seed,
            &PrequentialStops::only(keep.clone())).unwrap();
        let (f, s) = (full.prequential.unwrap(), sel.prequential.unwrap());
        for i in 0..f.obs_times.len() {
            if keep.contains(&i) {
                assert_eq!(bits(&s.per_stream_held_out_log_liks[i][0]), bits(&f.per_stream_held_out_log_liks[i][0]),
                    "seed {seed}: held-out log-liks at t = {}", f.obs_times[i]);
            } else {
                assert!(s.per_stream_held_out_log_liks[i].is_empty(), "an unselected stop keeps nothing");
            }
        }
    }
}

/// D56: the held-out log-likelihood is the fitting likelihood. Over the rows
/// ending at or before t = 30, a filter whose last aux value (t = 30) is held
/// out and one where it is observed are identical up to that stop; there the
/// held-out filter's scoring likelihood equals, per particle, the per-stream
/// log-likelihood the other records before assimilating the same value.
#[test]
fn the_held_out_log_likelihood_is_the_assimilation_likelihood_of_the_same_value() {
    let compiled = sir();
    let process = ChainBinomialProcess::new(compiled.clone());
    let params = &compiled.default_params;
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let model = |held_out: bool| {
        let w_ends = [7.0, 12.0, 21.0, 23.0];
        let w_vals = [20.0, 150.0, 260.0, 300.0];
        let a_ends = [3.0, 9.0, 12.0, 17.0, 30.0];
        let a_vals = [2.0, 30.0, 70.0, 110.0, 1100.0];
        let last = a_vals.len() - 1;
        let specs = vec![
            StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
                compiled.model.observations[0].clone(),
                w_vals.iter().map(|&v| Some(ObsCell::Scalar(v))).collect(),
                Period::contiguous(0.0, &w_ends).unwrap()),
            StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
                compiled.model.observations[1].clone(),
                a_vals.iter().enumerate()
                    .map(|(i, &v)| Some(match i == last && held_out {
                        true => ObsCell::HeldOut(v),
                        false => ObsCell::Scalar(v),
                    }))
                    .collect(),
                Period::contiguous(0.0, &a_ends).unwrap()),
        ];
        let (bound, _) = BoundObs::bind(0.0, specs).unwrap();
        MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
    };
    let (held, observed) = (model(true), model(false));
    let ui = index_of(&held, 30.0);
    assert_eq!(index_of(&observed, 30.0), ui, "one axis");
    assert!(!held.assimilates_at(ui) && observed.assimilates_at(ui));
    for seed in SEEDS {
        let rh = bootstrap_filter(&process, &held, params, &smc(), seed).unwrap();
        let ro = bootstrap_filter(&process, &observed, params, &smc(), seed).unwrap();
        let (ph, po) = (rh.prequential.unwrap(), ro.prequential.unwrap());
        assert!(po.per_stream_log_liks[ui][1].iter().any(|x| *x != po.per_stream_log_liks[ui][1][0]),
            "seed {seed}: the control's log-likelihoods vary over particles");
        assert_eq!(bits(&ph.per_stream_held_out_log_liks[ui][1]), bits(&po.per_stream_log_liks[ui][1]),
            "seed {seed}: the held-out value is scored by the code that would have assimilated it");
        // The held-out filter does not assimilate it: its per-stream term is 0.
        assert!(ph.per_stream_log_liks[ui][1].iter().all(|x| *x == 0.0), "seed {seed}");
    }
}
