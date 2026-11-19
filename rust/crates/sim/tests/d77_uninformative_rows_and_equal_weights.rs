//! D77: the unconditional filters do not resample on exactly equal weights, and
//! a row whose likelihood is constant by its structure is missing to them.
//!
//! Two rules, pinned for the bootstrap filter and IF2:
//!
//! 1. *Equal normalised log weights do not resample.* An exactly uniform swarm
//!    already represents the filtering distribution, so the resample is
//!    skipped. That leaves the resampling stream undrawn, so a run with an
//!    equal-weight observation computes what the same run computes with that
//!    row `NA`, except for the row's own (constant) likelihood term.
//! 2. *A binomial row with zero trials from the data (`0` of `0`) is a hole.*
//!    Its likelihood is exactly 0 for every particle, and the binder turns it
//!    into a hole, so `0 of 0` and `NA` are the same row to the filters:
//!    alone at a time, and beside an informative stream at the same time.
//!
//! Each comparison is bitwise (`to_bits`): the defect each rule removes is a
//! spent resampling draw (or, in IF2, a spent perturbation), which moves every
//! later draw rather than changing any number by a little.
//!
//! The fixture is a closed SIR (N = 10 000) on the chain-binomial backend, plus
//! a compartment `D` fed by a deterministic inflow of 10 a day. Three streams:
//! weekly incidence of infection under a negative binomial (`weekly`); the
//! inflow's incidence under a Normal (`flat`), which every particle projects
//! identically, so its weights are equal; and a binomial prevalence survey of
//! `I` with a per-row denominator `tested` read from the data (`survey`).

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        BinomialLikelihood, Likelihood, NegBinomialLikelihood, NormalLikelihood,
        ObservationModel as IrObs, ObservationSchedule, Projection,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::{
    bootstrap_filter, dense_cells,
    if2::{run_if2, CoolingSchedule, EstimatedParam, IF2Config, IF2Result},
    multi_stream_obs::{MultiStreamObsModel, Period, StreamProjection, StreamSpec, StreamTimes},
    particle_filter::PFilterResult,
    traits::{ObservationModel, SMCConfig},
    types::Transform,
    BoundObs, ChainBinomialProcess, ObsCell,
};

const DT: f64 = 1.0;
const N_PARTICLES: usize = 300;
const SEEDS: [u64; 3] = [1, 2, 20261118];
const I_IDX: usize = 1;

const WEEKLY_TIMES: [f64; 11] = [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0];
const WEEKLY_CASES: [f64; 11] = [16.0, 166.0, 626.0, 1303.0, 1260.0, 1023.0, 327.0, 91.0, 58.0, 6.0, 2.0];

fn p(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn c(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn konst(value: f64) -> Expr {
    Expr::Const(ConstExpr { value })
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op, left: Box::new(a), right: Box::new(b) } })
}
fn projected() -> Expr {
    Expr::Projected(ProjectedExpr { projected: () })
}
fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}
fn transition(name: &str, sto: Vec<StoichiometryEntry>, rate: Expr, draw: DrawMethod) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(), stoichiometry: sto, rate, metadata: None,
        draw_method: draw, rate_grad: Default::default(), lineage: None,
    }
}
fn obs_block(name: &str, projection: Projection, likelihood: Likelihood) -> IrObs {
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
        projection,
        projection_state_grad: Default::default(),
        likelihood,
    }
}

fn model() -> Arc<CompiledModel> {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "d77_fixture".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: ["S", "I", "R", "D"].iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![
            transition("infection",
                vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                bin(BinOp::Div, bin(BinOp::Mul, bin(BinOp::Mul, p("beta"), c("S")), c("I")), n),
                DrawMethod::Poisson),
            transition("recovery",
                vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry("R".into(), 1)],
                bin(BinOp::Mul, p("gamma"), c("I")), DrawMethod::Poisson),
            transition("inflow", vec![StoichiometryEntry("D".into(), 1)],
                konst(10.0), DrawMethod::Deterministic),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![
            obs_block("weekly", Projection::CumulativeFlow("infection".into()),
                Likelihood::NegBinomial(NegBinomialLikelihood {
                    mean: ir::Diffable::new(bin(BinOp::Mul, p("rho"), projected())),
                    dispersion: ir::Diffable::new(p("k")),
                })),
            obs_block("flat", Projection::CumulativeFlow("inflow".into()),
                Likelihood::Normal(NormalLikelihood {
                    mean: ir::Diffable::new(projected()),
                    sd: ir::Diffable::new(konst(50.0)),
                })),
            obs_block("survey", Projection::CurrentPop("I".into()),
                Likelihood::Binomial(BinomialLikelihood {
                    n: Expr::obs_column_ref("tested"),
                    p: ir::Diffable::new(bin(BinOp::Div, projected(), konst(10_000.0))),
                })),
        ],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![param("beta", 0.4), param("gamma", 0.15), param("rho", 0.6), param("k", 10.0)],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            for (k, v) in [("S", 9990.0), ("I", 10.0), ("R", 0.0), ("D", 0.0)] { h.insert(k.into(), v); }
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 80.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 80.0,
            time_semantics: "continuous".into(),
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

/// One `flat` row: the period it covers and its value (`None` is `NA`).
type FlatRows = Vec<(Period, Option<f64>)>;
/// One `survey` row: its time, the positive count and `tested` (`None` is a
/// row written `NA`, both columns missing together).
type SurveyRows = Vec<(f64, Option<(f64, f64)>)>;

fn bind(compiled: &Arc<CompiledModel>, flat: FlatRows, survey: SurveyRows) -> MultiStreamObsModel {
    let t_idx = |name: &str| compiled.model.transitions.iter().position(|t| t.name == name).unwrap();
    let mut specs = vec![StreamSpec {
        ir_model: compiled.model.observations[0].clone(),
        times: StreamTimes::Intervals(Period::contiguous(0.0, &WEEKLY_TIMES).unwrap()),
        projection: StreamProjection::FlowSum(vec![t_idx("infection")]),
        observations: dense_cells(WEEKLY_CASES.to_vec()),
        aux: vec![vec![]; WEEKLY_CASES.len()],
    }];
    if !flat.is_empty() {
        let (periods, values): (Vec<Period>, Vec<Option<f64>>) = flat.into_iter().unzip();
        specs.push(StreamSpec::dense_covering(
            StreamProjection::FlowSum(vec![t_idx("inflow")]),
            compiled.model.observations[1].clone(),
            values.into_iter().map(|v| v.map(ObsCell::Scalar)).collect(),
            periods,
        ));
    }
    if !survey.is_empty() {
        let times: Vec<f64> = survey.iter().map(|(t, _)| *t).collect();
        // A row written `NA` is a hole with its `tested` missing too; the
        // loader's present-together rule makes it so.
        let aux = survey.iter()
            .map(|(_, r)| match r {
                Some((_, tested)) => vec![("tested".to_string(), *tested)],
                None => vec![],
            })
            .collect();
        specs.push(StreamSpec {
            ir_model: compiled.model.observations[2].clone(),
            times: StreamTimes::Instants(times),
            projection: StreamProjection::IntCompSum(vec![I_IDX]),
            observations: survey.iter().map(|(_, r)| r.map(|(pos, _)| ObsCell::Scalar(pos))).collect(),
            aux,
        });
    }
    let (bound, report) = BoundObs::bind(0.0, specs).expect("fixture binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
}

fn period(a: f64, b: f64) -> Period {
    Period::new(a, b).unwrap()
}

fn smc() -> SMCConfig {
    SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: true,
        record_prequential: true,
        record_predictions: true,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    }
}

fn pf(compiled: &Arc<CompiledModel>, obs: &MultiStreamObsModel, seed: u64) -> PFilterResult {
    let process = ChainBinomialProcess::new(compiled.clone());
    bootstrap_filter(&process, obs, &compiled.default_params, &smc(), seed).unwrap()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn index_of(obs: &MultiStreamObsModel, t: f64) -> usize {
    obs.obs_times().iter().position(|&u| u == t)
        .unwrap_or_else(|| panic!("t = {t} not on the axis {:?}", obs.obs_times()))
}

/// Every output of two bootstrap filters on one union axis, compared bitwise,
/// except at the indices in `skip` (where the two data sets differ by design).
fn assert_pf_identical(a: &PFilterResult, b: &PFilterResult, skip: &[usize], what: &str) {
    let n = a.ll_increments.len();
    assert_eq!(n, b.ll_increments.len(), "{what}: different union axes");
    let keep = |v: &[f64]| (0..n).filter(|i| !skip.contains(i)).map(|i| v[i]).collect::<Vec<_>>();
    assert_eq!(bits(&keep(&a.ll_increments)), bits(&keep(&b.ll_increments)),
        "{what}: log-likelihood increments\n a: {:?}\n b: {:?}", a.ll_increments, b.ll_increments);
    assert_eq!(bits(&keep(&a.ess_trace)), bits(&keep(&b.ess_trace)),
        "{what}: ESS trace\n a: {:?}\n b: {:?}", a.ess_trace, b.ess_trace);
    assert_eq!(bits(&keep(&a.logw_var_trace)), bits(&keep(&b.logw_var_trace)), "{what}: τ² trace");
    let fs = |r: &PFilterResult| r.final_states.as_ref().unwrap().iter()
        .map(|s| s.counts.clone()).collect::<Vec<_>>();
    assert_eq!(fs(a), fs(b), "{what}: final particle states");
    let (ta, tb) = (a.ancestry.as_ref().unwrap(), b.ancestry.as_ref().unwrap());
    assert_eq!(ta.states, tb.states, "{what}: pre-resample particle states");
    assert_eq!(ta.ancestors, tb.ancestors, "{what}: ancestry");
    for i in (0..n).filter(|i| !skip.contains(i)) {
        assert_eq!(bits(&ta.log_weights[i]), bits(&tb.log_weights[i]), "{what}: weights at index {i}");
    }
}

fn if2(compiled: &Arc<CompiledModel>, obs: &MultiStreamObsModel, seed: u64,
       only_at_t0: bool, n_iterations: usize) -> IF2Result {
    let process = ChainBinomialProcess::new(compiled.clone());
    let beta = Transform::Interval { lo: 0.05, hi: 2.0 };
    let gamma = Transform::Interval { lo: 0.01, hi: 1.0 };
    let params = vec![
        EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.5,
            rw_sd: beta.step_from_natural(0.02, 0.5), transform: beta,
            perturb_only_at_t0: only_at_t0, rw_sd_auto: false,
        },
        EstimatedParam {
            name: "gamma".into(), index: 1, initial: 0.2,
            rw_sd: gamma.step_from_natural(0.01, 0.2), transform: gamma,
            perturb_only_at_t0: only_at_t0, rw_sd_auto: false,
        },
    ];
    let config = IF2Config {
        n_particles: 200,
        n_iterations,
        cooling: CoolingSchedule { fraction: 0.5, target_iters: 50 },
        simplex_groups: vec![],
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    };
    run_if2(&process, obs, &compiled.default_params, &params, &config, seed).unwrap()
}

/// The IF2 parameter path: every iteration's parameter means, and the estimate.
fn assert_if2_paths_identical(a: &IF2Result, b: &IF2Result, what: &str) {
    assert_eq!(bits(&a.mle), bits(&b.mle), "{what}: IF2 estimate {:?} vs {:?}", a.mle, b.mle);
    for (ia, ib) in a.iterations.iter().zip(&b.iterations) {
        assert_eq!(bits(&ia.param_means), bits(&ib.param_means),
            "{what}: iteration {} parameter means {:?} vs {:?}", ia.iteration, ia.param_means, ib.param_means);
    }
}

// ─── Rule 1: equal weights do not resample ──────────────────────────────────

/// The `flat` row at t = 10 is observed, and every particle projects the same
/// inflow (100 over (0, 10]), so every weight there is equal. The bootstrap
/// filter assimilates it and does not resample: everything it computes is what
/// it computes with that row `NA`, except the row's own likelihood term (and
/// the weights holding it). The ancestry at t = 10 is the identity, which an
/// equal-weight systematic resample also gives; what tells the two apart is the
/// resampling stream, whose draw at t = 10 would move every later resample.
#[test]
fn equal_weights_at_an_observation_do_not_resample_in_the_bootstrap_filter() {
    let compiled = model();
    let na = bind(&compiled, vec![(period(0.0, 10.0), None)], vec![]);
    let seen = bind(&compiled, vec![(period(0.0, 10.0), Some(103.0))], vec![]);
    let i10 = index_of(&seen, 10.0);
    assert!(seen.assimilates_at(i10) && !na.assimilates_at(i10), "fixture: the row is the only difference");
    for &seed in &SEEDS {
        let a = pf(&compiled, &na, seed);
        let b = pf(&compiled, &seen, seed);
        let lw = &b.ancestry.as_ref().unwrap().log_weights[i10];
        assert!(lw.iter().all(|w| w.to_bits() == lw[0].to_bits()) && lw[0] < 0.0,
            "fixture: the weights at t = 10 are equal and the row is scored ({})", lw[0]);
        assert_ne!(b.ll_increments[i10], 0.0, "the equal-weight row is assimilated");
        assert_pf_identical(&a, &b, &[i10], &format!("equal-weight row, seed {seed}"));
    }
}

/// The same row under IF2. Both parameters are perturbed at t = 0 only and the
/// run is one iteration long, so the row's assimilation draws no perturbation;
/// the only draw it could spend is the resample. It spends none, so the
/// parameter path is the one IF2 computes with the row `NA`.
#[test]
fn equal_weights_at_an_observation_do_not_resample_in_if2() {
    let compiled = model();
    let na = bind(&compiled, vec![(period(0.0, 10.0), None)], vec![]);
    let seen = bind(&compiled, vec![(period(0.0, 10.0), Some(103.0))], vec![]);
    for &seed in &SEEDS {
        let a = if2(&compiled, &na, seed, true, 1);
        let b = if2(&compiled, &seen, seed, true, 1);
        assert_if2_paths_identical(&a, &b, &format!("equal-weight row, seed {seed}"));
    }
}

// ─── Rule 2: `0 of 0` is missing ────────────────────────────────────────────

/// Survey rows with a real denominator at t = 21 and t = 35, and the row at
/// `t` either `0 of 0` or `NA`.
fn survey_with(t: f64, row: Option<(f64, f64)>) -> SurveyRows {
    vec![(t, row), (21.0, Some((60.0, 400.0))), (35.0, Some((90.0, 400.0)))]
}

/// A `0 of 0` row at a time no other stream observes is no assimilation event,
/// exactly like `NA` there: the bootstrap filter and IF2 (perturbing at every
/// event, three iterations) compute the same numbers with either.
#[test]
fn a_zero_of_zero_row_alone_is_missing() {
    let compiled = model();
    let zero = bind(&compiled, vec![], survey_with(10.0, Some((0.0, 0.0))));
    let na = bind(&compiled, vec![], survey_with(10.0, None));
    let i10 = index_of(&zero, 10.0);
    assert!(!zero.assimilates_at(i10), "a `0 of 0` row alone is no assimilation event");
    for &seed in &SEEDS {
        assert_pf_identical(&pf(&compiled, &zero, seed), &pf(&compiled, &na, seed), &[],
            &format!("`0 of 0` alone, seed {seed}"));
        assert_if2_paths_identical(&if2(&compiled, &zero, seed, false, 3), &if2(&compiled, &na, seed, false, 3),
            &format!("`0 of 0` alone, seed {seed}"));
    }
}

/// A `0 of 0` survey row at t = 14, where the weekly stream is observed: the
/// weights there are not uniform (the weekly count selects), so the swarm
/// resamples. The `0 of 0` row must not move that: with it or with `NA` in its
/// place, the likelihood, the ESS trace, the weights, the ancestry, the
/// particles, the per-stream predictive record and the IF2 parameter path are
/// bit-identical.
#[test]
fn a_zero_of_zero_row_beside_an_informative_stream_is_missing() {
    let compiled = model();
    let zero = bind(&compiled, vec![], survey_with(14.0, Some((0.0, 0.0))));
    let na = bind(&compiled, vec![], survey_with(14.0, None));
    let i14 = index_of(&zero, 14.0);
    assert!(zero.assimilates_at(i14), "the weekly value at t = 14 is assimilated");
    for &seed in &SEEDS {
        let a = pf(&compiled, &zero, seed);
        let b = pf(&compiled, &na, seed);
        let lw = &a.ancestry.as_ref().unwrap().log_weights[i14];
        assert!(lw.iter().any(|w| w.to_bits() != lw[0].to_bits()),
            "fixture: the weights at t = 14 are not uniform");
        assert_ne!(a.ancestry.as_ref().unwrap().ancestors[i14], (0..N_PARTICLES).collect::<Vec<_>>(),
            "fixture: the swarm resamples at t = 14");
        assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits(), "seed {seed}: log-likelihood");
        assert_pf_identical(&a, &b, &[], &format!("`0 of 0` beside weekly, seed {seed}"));
        let (pa, pb) = (a.prequential.as_ref().unwrap(), b.prequential.as_ref().unwrap());
        assert_eq!(pa.log_liks, pb.log_liks, "seed {seed}: joint predictive log scores");
        assert_eq!(pa.per_stream_log_liks, pb.per_stream_log_liks, "seed {seed}: per-stream log scores");
        assert_if2_paths_identical(&if2(&compiled, &zero, seed, false, 3), &if2(&compiled, &na, seed, false, 3),
            &format!("`0 of 0` beside weekly, seed {seed}"));
    }
}

/// Zero trials cannot yield a positive count: `3 of 0` is still refused at bind,
/// with the row located. Only `0 of 0` becomes a hole.
#[test]
fn a_positive_count_against_zero_trials_is_still_refused() {
    let compiled = model();
    let spec = StreamSpec {
        ir_model: compiled.model.observations[2].clone(),
        times: StreamTimes::Instants(vec![10.0, 21.0]),
        projection: StreamProjection::IntCompSum(vec![I_IDX]),
        observations: dense_cells(vec![3.0, 60.0]),
        aux: vec![vec![("tested".into(), 0.0)], vec![("tested".into(), 400.0)]],
    };
    let Err(report) = BoundObs::bind(0.0, vec![spec]) else {
        panic!("`3 of 0` must be refused at bind");
    };
    assert!(report.is_fatal());
    let text = format!("{:?}", report.findings());
    assert!(text.contains("row 0") && text.contains("no trials cannot yield a positive count"), "{text}");
}
