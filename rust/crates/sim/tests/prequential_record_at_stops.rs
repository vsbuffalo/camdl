//! D84: the bootstrap filter can keep its prequential record at selected stops
//! only.
//!
//! The invariant pinned here: for a fixed seed, a filter that keeps the record
//! at a list of stops returns, at those stops, the same per-particle
//! log-likelihoods, per-stream log-likelihoods and per-stream predictive draws
//! as a filter that keeps the record everywhere, bit for bit; it keeps nothing
//! at the other stops; and every other output — the log-likelihood, its
//! increments, the ESS and log-weight-variance traces, the final particles —
//! is the same as with a full recording and as with no recording at all.
//!
//! The defect it guards against is an RNG shift. A predictive draw consumes
//! the particle's diagnostic stream, so a filter that skipped the draw at a
//! stop it does not keep would move every later kept draw. The selection
//! therefore always leaves unselected stops before selected ones.
//!
//! The fixture is a closed SIR on the chain-binomial backend with two
//! incidence streams on irregular, different cadences: `weekly` (negative
//! binomial) with `NA` rows, and `aux` (Poisson) with `NA` rows and trailing
//! value-less periods, the shape a forecast's targets take. The selected stops
//! include a stop where a selected stream's cell is `NA`, a stop where one
//! stream is not scheduled, a stop where both are `NA`, and those trailing
//! periods.

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
use sim::error::SimError;
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

const T_END: f64 = 90.0;

/// Closed SIR (N = 10 000) with two incidence streams on the infection flow.
fn sir() -> Arc<CompiledModel> {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "prequential_record_at_stops".into(),
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

/// Contiguous periods from 0 through `ends`, each with its value (`None` is
/// `NA`).
fn rows(ends: &[f64], values: &[Option<f64>]) -> (Vec<Period>, Vec<Option<ObsCell>>) {
    assert_eq!(ends.len(), values.len());
    let periods = Period::contiguous(0.0, ends).unwrap();
    (periods, values.iter().map(|v| v.map(ObsCell::Scalar)).collect())
}

/// `weekly`: irregular periods, two `NA` rows. `aux`: a different irregular
/// cadence, `NA` rows, and three trailing value-less periods after t = 60
/// (the forecast shape: rows the filter does not assimilate, at which a
/// caller reads the predictive).
fn bind(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let (wp, wv) = rows(
        &[7.0, 12.0, 21.0, 23.0, 35.0, 42.0, 50.0, 60.0],
        &[Some(20.0), Some(150.0), None, Some(300.0), Some(1500.0), None, Some(400.0), Some(150.0)],
    );
    let (ap, av) = rows(
        &[3.0, 9.0, 12.0, 17.0, 30.0, 33.0, 42.0, 55.0, 60.0, 67.0, 74.0, 81.0],
        &[Some(2.0), Some(30.0), None, Some(110.0), Some(1100.0), None, None, Some(250.0), Some(60.0),
          None, None, None],
    );
    let specs = vec![
        StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
            compiled.model.observations[0].clone(), wv, wp),
        StreamSpec::dense_covering(StreamProjection::FlowSum(vec![inf]),
            compiled.model.observations[1].clone(), av, ap),
    ];
    let (bound, report) = BoundObs::bind(0.0, specs).expect("fixture binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
}

fn smc(record: bool) -> SMCConfig {
    SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: record,
        record_predictions: false,
        record_stream_means: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Bitwise equality of two per-particle vectors, naming the first particle
/// that differs rather than printing both.
fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
        panic!("{what}: particle {i} is {} here and {} in the full recording", a[i], b[i]);
    }
}

fn index_of(obs: &MultiStreamObsModel, t: f64) -> usize {
    obs.obs_times().iter().position(|&u| u == t)
        .unwrap_or_else(|| panic!("no stop at t = {t} in {:?}", obs.obs_times()))
}

/// The stops the tests keep, chosen to cover each kind of cell a reader of the
/// record meets, each with unselected stops before it.
fn selected(obs: &MultiStreamObsModel) -> Vec<usize> {
    // t = 21: weekly NA, aux not scheduled. t = 30: aux observed, weekly not
    // scheduled. t = 42: both NA. t = 50: weekly observed, aux not scheduled.
    // t = 67, 81: aux value-less after the last observation, weekly not
    // scheduled (the forecast reads).
    [21.0, 30.0, 42.0, 50.0, 67.0, 81.0].iter().map(|&t| index_of(obs, t)).collect()
}

/// The filter's outputs other than the prequential record.
fn assert_same_filter(a: &PFilterResult, b: &PFilterResult, what: &str) {
    assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits(),
        "{what}: log-likelihood {} vs {}", a.log_likelihood, b.log_likelihood);
    assert_eq!(bits(&a.ll_increments), bits(&b.ll_increments), "{what}: log-likelihood increments");
    assert_eq!(bits(&a.ess_trace), bits(&b.ess_trace), "{what}: ESS trace");
    assert_eq!(bits(&a.logw_var_trace), bits(&b.logw_var_trace), "{what}: log-weight variance trace");
    let fs = |r: &PFilterResult| r.final_states.as_ref().unwrap().iter()
        .map(|s| (s.counts.clone(), s.flow_accumulators.clone(), s.acc.clone()))
        .collect::<Vec<_>>();
    assert_eq!(fs(a), fs(b), "{what}: final particles");
}

/// The fixture is not vacuous: the selected stops have the cells the module
/// doc names, unselected stops come before them, the run assimilates, and the
/// draws at the selected stops are finite and spread.
fn assert_fixture_covers(obs: &MultiStreamObsModel, full: &PFilterResult, keep: &[usize]) {
    let n = obs.n_observations();
    let cells: Vec<(ObsCellState, ObsCellState)> =
        keep.iter().map(|&i| (obs.cell_state(0, i), obs.cell_state(1, i))).collect();
    assert!(cells.iter().any(|(w, a)| *w == ObsCellState::Hole && *a == ObsCellState::NotScheduled),
        "a selected stop where weekly is NA and aux not scheduled: {cells:?}");
    assert!(cells.iter().any(|(w, a)| *w == ObsCellState::Hole && *a == ObsCellState::Hole),
        "a selected stop where both streams are NA: {cells:?}");
    assert!(cells.iter().any(|(_, a)| matches!(a, ObsCellState::Scored { .. })),
        "a selected stop where aux is observed: {cells:?}");
    assert!(keep.iter().any(|&i| !obs.assimilates_at(i)) && keep.iter().any(|&i| obs.assimilates_at(i)),
        "selected stops both assimilate and do not");
    assert!(keep.iter().all(|&i| i > 0 && (0..i).any(|j| !keep.contains(&j))),
        "every selected stop has an unselected stop before it");
    assert!((0..n).filter(|i| !keep.contains(i)).count() >= 8, "most stops are unselected");
    // Irregular spacing: the union axis's gaps are not all equal.
    let gaps: Vec<f64> = obs.obs_times().windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.iter().any(|g| *g != gaps[0]), "irregular stops: {gaps:?}");
    assert!(full.log_likelihood.is_finite(), "fixture filter is finite");
    let preq = full.prequential.as_ref().unwrap();
    let last = *keep.last().unwrap();
    let ys = &preq.per_stream_samples[last][1];
    assert!(ys.iter().all(|y| y.is_finite()), "aux draws at the last target are finite");
    assert!(ys.iter().any(|y| *y != ys[0]), "aux draws at the last target are spread: {ys:?}");
}

/// The headline. At every selected stop the record is the full recording's,
/// bit for bit; at every other stop it is empty, with the stop's time kept;
/// the rest of the filter is unchanged against a full recording and against
/// no recording.
#[test]
fn selected_stops_keep_the_full_recordings_records_and_the_filter_is_unchanged() {
    let compiled = sir();
    let obs = bind(&compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let params = &compiled.default_params;
    let keep = selected(&obs);
    let n = obs.n_observations();
    for seed in SEEDS {
        let what = format!("seed {seed}");
        let full = bootstrap_filter(&process, &obs, params, &smc(true), seed).unwrap();
        let sel = bootstrap_filter_recording_at(
            &process, &obs, params, &smc(true), seed, &PrequentialStops::only(keep.clone())).unwrap();
        let none = bootstrap_filter(&process, &obs, params, &smc(false), seed).unwrap();
        assert_fixture_covers(&obs, &full, &keep);

        assert_same_filter(&sel, &full, &format!("{what}, selected vs full recording"));
        assert_same_filter(&sel, &none, &format!("{what}, selected vs no recording"));

        let (f, s) = (full.prequential.as_ref().unwrap(), sel.prequential.as_ref().unwrap());
        assert_eq!(bits(&s.obs_times), bits(&f.obs_times), "{what}: the record keeps every stop's time");
        assert_eq!(s.stream_names, f.stream_names);
        assert_eq!((s.log_liks.len(), s.per_stream_log_liks.len(), s.per_stream_samples.len()), (n, n, n),
            "{what}: the record stays aligned with the observation axis");
        for i in 0..n {
            let t = f.obs_times[i];
            if keep.contains(&i) {
                assert_bits(&s.log_liks[i], &f.log_liks[i], &format!("{what}: log_liks at t = {t}"));
                assert_eq!(s.per_stream_log_liks[i].len(), 2, "{what}: per-stream log-liks at t = {t}");
                for si in 0..2 {
                    assert_bits(&s.per_stream_log_liks[i][si], &f.per_stream_log_liks[i][si],
                        &format!("{what}: per-stream log-liks, stream {si}, t = {t}"));
                    assert_bits(&s.per_stream_samples[i][si], &f.per_stream_samples[i][si],
                        &format!("{what}: per-stream predictive draws, stream {si}, t = {t} (a \
                         difference means the diagnostic stream was drawn differently before \
                         this stop)"));
                }
            } else {
                assert!(s.log_liks[i].is_empty() && s.per_stream_log_liks[i].is_empty()
                        && s.per_stream_samples[i].is_empty(),
                    "{what}: an unselected stop (t = {t}) keeps no record");
            }
        }
    }
}

/// Selecting every stop is a full recording.
#[test]
fn selecting_every_stop_is_the_full_recording() {
    let compiled = sir();
    let obs = bind(&compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let params = &compiled.default_params;
    let n = obs.n_observations();
    let full = bootstrap_filter(&process, &obs, params, &smc(true), SEEDS[0]).unwrap();
    let all = bootstrap_filter_recording_at(
        &process, &obs, params, &smc(true), SEEDS[0], &PrequentialStops::only((0..n).rev())).unwrap();
    assert_same_filter(&all, &full, "every stop listed");
    let (f, a) = (full.prequential.unwrap(), all.prequential.unwrap());
    for i in 0..n {
        assert_eq!(bits(&a.log_liks[i]), bits(&f.log_liks[i]));
        for si in 0..2 {
            assert_eq!(bits(&a.per_stream_samples[i][si]), bits(&f.per_stream_samples[i][si]));
            assert_eq!(bits(&a.per_stream_log_liks[i][si]), bits(&f.per_stream_log_liks[i][si]));
        }
    }
}

/// A stop past the end of the axis is refused, not silently ignored.
#[test]
fn a_stop_past_the_axis_is_refused() {
    let compiled = sir();
    let obs = bind(&compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let n = obs.n_observations();
    let err = bootstrap_filter_recording_at(&process, &obs, &compiled.default_params, &smc(true), 1,
        &PrequentialStops::only([0, n])).err().expect("refused");
    assert!(matches!(&err, SimError::Validation(m) if m.contains(&format!("stop {n}"))), "{err}");
}
