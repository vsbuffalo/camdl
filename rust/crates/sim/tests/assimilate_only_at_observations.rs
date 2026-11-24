//! D72: a filter assimilates only where the bound data carry an observed value.
//!
//! The union axis a filter stops at holds every stream's period ends and every
//! declared period's start. Some of those stops carry no observation: a row
//! whose every stream is `NA`, or a declared period's start (where a bin opens
//! and nothing is scored). A stop like that exists so the accumulators can fold
//! and reset there. It is not an assimilation event: it contributes no
//! likelihood term, opens no collapse-check window and does not resample.
//!
//! The invariant pinned here: for a fixed seed, a grid-aligned stop introduced
//! only by a missing observation or a value-less period start leaves the latent
//! process, the particle ancestry, IF2's parameter evolution and the
//! likelihood exactly as they are without it. Each comparison is bitwise
//! (`to_bits`), because the defect it guards against is an RNG shift: a
//! resampling draw spent at the extra stop moves every later draw.
//!
//! The fixture is a closed SIR on the chain-binomial backend with weekly
//! incidence scored under a negative binomial, plus a second incidence stream,
//! `aux`, whose rows are all `NA`. Binding `aux` adds stops and nothing else.

use sim::inference::if2::{If2Step, ZSd};
use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        Likelihood, NegBinomialLikelihood, NormalLikelihood, ObservationModel as IrObs,
        ObservationSchedule, PoissonLikelihood, Projection,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::inference::{
    bootstrap_filter, dense_cells,
    if2::{run_if2, CoolingSchedule, EstimatedParam, IF2Config, IF2Result},
    multi_stream_obs::{MultiStreamObsModel, Period, StreamProjection, StreamSpec, StreamTimes},
    particle_filter::PFilterResult,
    traits::{ObservationModel, SMCConfig},
    types::Transform,
    BoundObs, ChainBinomialProcess, ObsCell,
};
use sim::compiled_model::CompiledModel;
use sim::error::{PFDegenerateKind, SimError};

const DT: f64 = 1.0;
const N_PARTICLES: usize = 400;
const SEEDS: [u64; 4] = [1, 2, 3, 20261116];

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
fn transition(name: &str, sto: Vec<StoichiometryEntry>, rate: Expr, draw: DrawMethod) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(), stoichiometry: sto, rate, metadata: None,
        draw_method: draw, rate_grad: Default::default(), lineage: None,
    }
}
fn incidence_block(name: &str, transition: &str, likelihood: Likelihood) -> IrObs {
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
        projection: Projection::CumulativeFlow(transition.into()),
        projection_state_grad: Default::default(),
        likelihood,
    }
}
fn projected() -> Expr {
    Expr::Projected(ProjectedExpr { projected: () })
}

fn model_shell(name: &str, compartments: &[&str], transitions: Vec<Transition>,
               observations: Vec<IrObs>, parameters: Vec<Parameter>,
               init: &[(&str, f64)], t_end: f64) -> Arc<CompiledModel> {
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: name.into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: compartments.iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions,
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations,
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters,
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            for (k, v) in init { h.insert((*k).into(), *v); }
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, t_end]),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end,
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

/// Closed SIR (N = 10 000), weekly incidence(infection) ~ NegBin(ρ·proj, k),
/// and the all-`NA` `aux` stream on the same flow under a Poisson.
fn sir() -> Arc<CompiledModel> {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    model_shell(
        "assimilate_only_at_observations",
        &["S", "I", "R"],
        vec![
            transition("infection",
                vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                div(mul(mul(p("beta"), c("S")), c("I")), n), DrawMethod::Poisson),
            transition("recovery",
                vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry("R".into(), 1)],
                mul(p("gamma"), c("I")), DrawMethod::Poisson),
        ],
        vec![
            incidence_block("weekly", "infection", Likelihood::NegBinomial(NegBinomialLikelihood {
                mean: ir::Diffable::new(mul(p("rho"), projected())),
                dispersion: ir::Diffable::new(p("k")),
            })),
            incidence_block("aux", "infection", Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(projected()),
            })),
        ],
        vec![param("beta", 0.4), param("gamma", 0.15), param("rho", 0.6), param("k", 10.0)],
        &[("S", 9990.0), ("I", 10.0), ("R", 0.0)],
        80.0,
    )
}

const WEEKLY_TIMES: [f64; 11] = [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0];
const WEEKLY_CASES: [f64; 11] = [16.0, 166.0, 626.0, 1303.0, 1260.0, 1023.0, 327.0, 91.0, 58.0, 6.0, 2.0];

/// The `aux` rows to bind beside the weekly stream: each a declared period
/// with its value (`None` is `NA`).
type AuxRows = Vec<(Period, Option<f64>)>;

fn bind(compiled: &Arc<CompiledModel>, weekly_cases: &[f64], aux: AuxRows) -> MultiStreamObsModel {
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let mut specs = vec![StreamSpec {
        ir_model: compiled.model.observations[0].clone(),
        times: StreamTimes::Intervals(Period::contiguous(0.0, &WEEKLY_TIMES).unwrap()),
        projection: StreamProjection::FlowSum(vec![inf]),
        observations: dense_cells(weekly_cases.to_vec()),
        aux: vec![vec![]; weekly_cases.len()],
    }];
    if !aux.is_empty() {
        let (periods, values): (Vec<Period>, Vec<Option<f64>>) = aux.into_iter().unzip();
        specs.push(StreamSpec::dense_covering(
            StreamProjection::FlowSum(vec![inf]),
            compiled.model.observations[1].clone(),
            values.into_iter().map(|v| v.map(ObsCell::Scalar)).collect(),
            periods,
        ));
    }
    let (bound, report) = BoundObs::bind(0.0, specs).expect("fixture binds");
    assert!(!report.is_fatal(), "{:?}", report.findings());
    MultiStreamObsModel::new(bound, compiled.clone()).unwrap()
}

fn period(a: f64, b: f64) -> Period {
    Period::new(a, b).unwrap()
}

fn smc(record: bool) -> SMCConfig {
    SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: record,
        record_prequential: record,
        record_predictions: record,
        record_stream_means: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    }
}

fn pf(compiled: &Arc<CompiledModel>, obs: &MultiStreamObsModel, cfg: &SMCConfig, seed: u64)
    -> Result<PFilterResult, SimError>
{
    let process = ChainBinomialProcess::new(compiled.clone());
    bootstrap_filter(&process, obs, &compiled.default_params, cfg, seed)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The union indices of `with`'s axis that `base`'s axis also has, in order,
/// and the ones it adds. Every base time must appear in `with`.
fn align(base: &[f64], with: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let shared: Vec<usize> = base.iter()
        .map(|t| with.iter().position(|u| u == t)
            .unwrap_or_else(|| panic!("base time {t} missing from {with:?}")))
        .collect();
    let extra = (0..with.len()).filter(|i| !shared.contains(i)).collect();
    (shared, extra)
}

/// Every output a filter's caller can read that the invariant covers: the
/// log-likelihood, the per-step log-likelihood increments and ESS at the
/// shared times, the final particle states, the per-step pre-resample states
/// (which is where the RNG consumption of the propagation shows), the weights,
/// the ancestry, and the per-particle prequential log scores. At the stops
/// `with` adds: no increment, no resampling (identity ancestry).
fn assert_filters_identical(base: &PFilterResult, base_obs: &MultiStreamObsModel,
                            with: &PFilterResult, with_obs: &MultiStreamObsModel, what: &str) {
    let (shared, extra) = align(base_obs.obs_times(), with_obs.obs_times());
    assert!(!extra.is_empty(), "{what}: vacuous fixture, the variant adds no stop");
    for &e in &extra {
        assert!(!with_obs.assimilates_at(e),
            "{what}: fixture error, added stop {e} carries an observation");
    }

    assert_eq!(with.log_likelihood.to_bits(), base.log_likelihood.to_bits(),
        "{what}: log-likelihood {} with the extra stops, {} without",
        with.log_likelihood, base.log_likelihood);
    let pick = |v: &[f64]| shared.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    assert_eq!(bits(&pick(&with.ll_increments)), bits(&base.ll_increments),
        "{what}: per-step log-likelihood increments differ at the shared times\n with: {:?}\n base: {:?}",
        pick(&with.ll_increments), base.ll_increments);
    assert_eq!(bits(&pick(&with.ess_trace)), bits(&base.ess_trace), "{what}: ESS at the shared times");
    for &e in &extra {
        assert_eq!(with.ll_increments[e].to_bits(), 0.0f64.to_bits(),
            "{what}: a stop with no observation contributes no likelihood term");
    }

    let fs = |r: &PFilterResult| r.final_states.as_ref().unwrap().iter()
        .map(|s| s.counts.clone()).collect::<Vec<_>>();
    assert_eq!(fs(with), fs(base), "{what}: final particle states differ");

    let (Some(ta), Some(tb)) = (with.ancestry.as_ref(), base.ancestry.as_ref()) else {
        return;
    };
    for (k, &i) in shared.iter().enumerate() {
        assert_eq!(ta.states[i], tb.states[k],
            "{what}: pre-resample particle states at t = {} differ: the propagation drew \
             different numbers", tb.obs_times[k]);
        assert_eq!(bits(&ta.log_weights[i]), bits(&tb.log_weights[k]),
            "{what}: weights at t = {}", tb.obs_times[k]);
    }
    let identity: Vec<usize> = (0..N_PARTICLES).collect();
    for &e in &extra {
        if e < ta.ancestors.len() {
            assert_eq!(ta.ancestors[e], identity,
                "{what}: the stop at t = {} resampled", ta.obs_times[e]);
        }
    }
    // Ancestry with the identity maps at the added stops removed is the base's.
    let kept: Vec<&Vec<usize>> = ta.ancestors.iter().enumerate()
        .filter(|(i, _)| !extra.contains(i)).map(|(_, a)| a).collect();
    let base_anc: Vec<&Vec<usize>> = tb.ancestors.iter().collect();
    assert_eq!(kept, base_anc, "{what}: ancestry differs");

    let (Some(pa), Some(pb)) = (with.prequential.as_ref(), base.prequential.as_ref()) else {
        return;
    };
    for (k, &i) in shared.iter().enumerate() {
        assert_eq!(bits(&pa.log_liks[i]), bits(&pb.log_liks[k]),
            "{what}: per-particle predictive log scores at t = {}", pb.obs_times[k]);
    }
}

fn base_and(aux: AuxRows) -> (Arc<CompiledModel>, MultiStreamObsModel, MultiStreamObsModel) {
    let compiled = sir();
    let base = bind(&compiled, &WEEKLY_CASES, vec![]);
    let with = bind(&compiled, &WEEKLY_CASES, aux);
    (compiled, base, with)
}

/// Test 1 and 7. An all-`NA` row at a grid time (t = 10, and t = 30) adds a
/// stop at each. With the same seed the filter's every output is the same with
/// and without those rows, and the particle states the propagation reaches at
/// the next observation are identical — the propagation across the inserted
/// boundary draws the same numbers.
#[test]
fn an_all_na_row_leaves_the_filter_unchanged() {
    // `aux` periods [0,10) and [10,30): no new period start (0 is the run's
    // start, 10 is the first period's end), so the stops are 10 and 30, both NA.
    let (compiled, base_obs, with_obs) =
        base_and(vec![(period(0.0, 10.0), None), (period(10.0, 30.0), None)]);
    assert_eq!(with_obs.obs_times().len(), WEEKLY_TIMES.len() + 2);
    for &seed in &SEEDS {
        let base = pf(&compiled, &base_obs, &smc(true), seed).unwrap();
        let with = pf(&compiled, &with_obs, &smc(true), seed).unwrap();
        assert_filters_identical(&base, &base_obs, &with, &with_obs, &format!("NA rows, seed {seed}"));
    }
}

/// The same invariant with the likelihood-only configuration (no recording),
/// which is what PMMH and the CLI's log-likelihood evaluations run.
#[test]
fn an_all_na_row_leaves_the_likelihood_only_filter_unchanged() {
    let (compiled, base_obs, with_obs) =
        base_and(vec![(period(0.0, 10.0), None), (period(10.0, 30.0), None)]);
    for &seed in &SEEDS {
        let base = pf(&compiled, &base_obs, &smc(false), seed).unwrap();
        let with = pf(&compiled, &with_obs, &smc(false), seed).unwrap();
        assert_filters_identical(&base, &base_obs, &with, &with_obs, &format!("NA rows (no recording), seed {seed}"));
    }
}

/// Test 2. A declared period that opens at t = 17 and closes at t = 21, where
/// the weekly stream already stops: the only stop it adds is its start, which
/// carries no value. A second period [40,45) with NA adds a start and a close.
#[test]
fn a_value_less_period_start_leaves_the_filter_unchanged() {
    let (compiled, base_obs, with_obs) =
        base_and(vec![(period(17.0, 21.0), None), (period(40.0, 45.0), None)]);
    let times = with_obs.obs_times();
    assert!(times.contains(&17.0) && times.contains(&40.0) && times.contains(&45.0), "{times:?}");
    assert_eq!(times.len(), WEEKLY_TIMES.len() + 3);
    for &seed in &SEEDS {
        let base = pf(&compiled, &base_obs, &smc(true), seed).unwrap();
        let with = pf(&compiled, &with_obs, &smc(true), seed).unwrap();
        assert_filters_identical(&base, &base_obs, &with, &with_obs, &format!("period start, seed {seed}"));
    }
}

/// Test 3. A row where one stream is observed and another is `NA` is one
/// assimilation event: the observed stream is scored and the swarm resamples.
/// Here `aux` is `NA` over [7,14), closing where the weekly stream observes.
#[test]
fn a_row_with_one_stream_observed_still_assimilates() {
    let compiled = sir();
    let base_obs = bind(&compiled, &WEEKLY_CASES, vec![]);
    let with_obs = bind(&compiled, &WEEKLY_CASES, vec![(period(7.0, 14.0), None)]);
    let times = with_obs.obs_times().to_vec();
    // 7 is already a stop and 14 is the weekly stream's, so the axis is unchanged.
    assert_eq!(times, base_obs.obs_times());
    let i14 = times.iter().position(|&t| t == 14.0).unwrap();
    assert!(with_obs.assimilates_at(i14), "the weekly value at t = 14 makes it an assimilation event");

    for &seed in &SEEDS {
        let base = pf(&compiled, &base_obs, &smc(true), seed).unwrap();
        let with = pf(&compiled, &with_obs, &smc(true), seed).unwrap();
        assert!(with.ll_increments[i14] < -1.0,
            "the observed stream is scored at t = 14: increment {}", with.ll_increments[i14]);
        assert_eq!(with.ll_increments[i14].to_bits(), base.ll_increments[i14].to_bits());
        let anc = &with.ancestry.as_ref().unwrap().ancestors[i14];
        assert_ne!(anc, &(0..N_PARTICLES).collect::<Vec<_>>(), "the swarm resampled at t = 14");
        assert_eq!(with.log_likelihood.to_bits(), base.log_likelihood.to_bits());
    }
}

/// A deterministic inflow of `K` a day into `R`, observed weekly as incidence
/// under a wide Normal, with the all-`NA` `aux` stream on the same flow.
fn inflow(k: f64) -> Arc<CompiledModel> {
    let normal = Likelihood::Normal(NormalLikelihood {
        mean: ir::Diffable::new(projected()),
        sd: ir::Diffable::new(Expr::Const(ConstExpr { value: 50.0 })),
    });
    model_shell(
        "assimilate_only_inflow",
        &["R"],
        vec![transition("inflow", vec![StoichiometryEntry("R".into(), 1)],
            Expr::Const(ConstExpr { value: k }), DrawMethod::Deterministic)],
        vec![incidence_block("weekly", "inflow", normal.clone()), incidence_block("aux", "inflow", normal)],
        vec![param("dummy", 0.0)],
        &[("R", 0.0)],
        28.0,
    )
}

/// Test 4. The accumulators still fold and reset at a skipped stop, exactly as
/// before: the weekly bin closing at t = 14 holds the whole week (7, 14] of
/// flow even though the filter stopped at t = 10 without assimilating, and
/// the `aux` bin closing at t = 10 holds (0, 10].
#[test]
fn a_skipped_stop_still_folds_and_resets_the_accumulators() {
    const K: f64 = 10.0;
    let compiled = inflow(K);
    let inflow_idx = 0;
    let weekly_times = [7.0, 14.0, 21.0, 28.0];
    let spec_weekly = || StreamSpec::dense_covering(
        StreamProjection::FlowSum(vec![inflow_idx]),
        compiled.model.observations[0].clone(),
        dense_cells(vec![7.0 * K; 4]),
        Period::contiguous(0.0, &weekly_times).unwrap(),
    );
    let spec_aux = StreamSpec::dense_covering(
        StreamProjection::FlowSum(vec![inflow_idx]),
        compiled.model.observations[1].clone(),
        vec![None],
        vec![period(0.0, 10.0)],
    );
    let base_obs = MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![spec_weekly()]).unwrap().0, compiled.clone()).unwrap();
    let with_obs = MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![spec_weekly(), spec_aux]).unwrap().0, compiled.clone()).unwrap();
    assert_eq!(with_obs.obs_times(), &[7.0, 10.0, 14.0, 21.0, 28.0]);

    let cfg = SMCConfig { n_particles: 8, ..smc(true) };
    let base = pf(&compiled, &base_obs, &cfg, 5).unwrap();
    let with = pf(&compiled, &with_obs, &cfg, 5).unwrap();
    // Projections recorded pre-reset, per stream: [weekly, aux].
    let proj = &with.ancestry.as_ref().unwrap().projections;
    assert_eq!(proj[1][0][1], 10.0 * K, "aux bin at t = 10 covers (0, 10]");
    assert_eq!(proj[2][0][0], 7.0 * K, "weekly bin at t = 14 covers (7, 14] across the skipped stop");
    let preds = with.predictions.as_ref().unwrap();
    let base_preds = base.predictions.as_ref().unwrap();
    assert_eq!(preds[2].state_mean, base_preds[1].state_mean);
    assert_eq!(with.log_likelihood.to_bits(), base.log_likelihood.to_bits());
}

/// Test 5. The collapse check reads one ESS per assimilation event. Weekly
/// counts ten times what the model produces collapse the swarm at every
/// observation; the check bails at the third. With an `NA` row between every
/// pair of observations the check still bails there, with the same ESS window:
/// the skipped stops do not enter the window (on main their ESS of N broke
/// every run of collapsed windows, so the check never fired).
#[test]
fn a_skipped_stop_opens_no_collapse_check_window() {
    let compiled = sir();
    let misfit: Vec<f64> = WEEKLY_CASES.iter().map(|&y| 10.0 * y + 500.0).collect();
    let base_obs = bind(&compiled, &misfit, vec![]);
    let aux: AuxRows = std::iter::once(0.0).chain(WEEKLY_TIMES.iter().copied())
        .zip(WEEKLY_TIMES.iter().copied())
        .map(|(a, b)| (period(a, b - 3.0), None))
        .collect();
    let with_obs = bind(&compiled, &misfit, aux);
    assert_eq!(with_obs.obs_times().len(), 2 * WEEKLY_TIMES.len());

    let last_ess = |r: Result<PFilterResult, SimError>| match r {
        Err(SimError::PFDegenerate { kind: PFDegenerateKind::EssCollapsed { last_ess }, .. }) => last_ess,
        Err(e) => panic!("expected an ESS collapse, got {e}"),
        Ok(r) => panic!("expected an ESS collapse, the filter ran to the end (ll {})", r.log_likelihood),
    };
    for &seed in &SEEDS {
        let base = last_ess(pf(&compiled, &base_obs, &smc(false), seed));
        let with = last_ess(pf(&compiled, &with_obs, &smc(false), seed));
        assert_eq!(bits(&with), bits(&base), "seed {seed}: the collapse window differs");
    }
}

fn if2(compiled: &Arc<CompiledModel>, obs: &MultiStreamObsModel, seed: u64) -> IF2Result {
    let process = ChainBinomialProcess::new(compiled.clone());
    let beta = Transform::Interval { lo: 0.05, hi: 2.0 };
    let gamma = Transform::Interval { lo: 0.01, hi: 1.0 };
    let params = vec![
        EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.5,
            transform: beta,
        },
        EstimatedParam {
            name: "gamma".into(), index: 1, initial: 0.2,
            transform: gamma,
        },
    ];
    let config = IF2Config {
        n_particles: 200,
        n_iterations: 4,
        cooling: CoolingSchedule { fraction: 0.5, target_iters: 50 },
        simplex_groups: vec![],
        steps: vec![If2Step::every_observation(ZSd::new(beta.step_from_natural(0.02, 0.5)).unwrap()), If2Step::every_observation(ZSd::new(gamma.step_from_natural(0.01, 0.2)).unwrap())],
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    };
    run_if2(&process, obs, &compiled.default_params, &params, &config, seed).unwrap()
}

/// Test 6. IF2 perturbs the parameters, cools and resamples per assimilation
/// event, so its parameter trajectory, the per-iteration diagnostics and the
/// estimate are the same with and without the `NA` rows and the value-less
/// period start.
#[test]
fn if2_parameter_evolution_ignores_skipped_stops() {
    let compiled = sir();
    let base_obs = bind(&compiled, &WEEKLY_CASES, vec![]);
    let with_obs = bind(&compiled, &WEEKLY_CASES,
        vec![(period(0.0, 10.0), None), (period(17.0, 21.0), None), (period(40.0, 45.0), None),
             (period(77.0, 80.0), None)]);
    assert!(with_obs.obs_times().len() >= WEEKLY_TIMES.len() + 4);
    for &seed in &SEEDS[..2] {
        let a = if2(&compiled, &base_obs, seed);
        let b = if2(&compiled, &with_obs, seed);
        assert_eq!(bits(&b.mle), bits(&a.mle), "seed {seed}: IF2 estimate {:?} vs {:?}", b.mle, a.mle);
        for (ia, ib) in a.iterations.iter().zip(&b.iterations) {
            assert_eq!(bits(&ib.param_means), bits(&ia.param_means),
                "seed {seed} iteration {}: parameter means", ia.iteration);
            assert_eq!(ib.if2_perturbed_loglik.to_bits(), ia.if2_perturbed_loglik.to_bits(),
                "seed {seed} iteration {}: perturbed log-likelihood", ia.iteration);
            for (da, db) in ia.param_diag.iter().zip(&ib.param_diag) {
                assert_eq!(db.weighted_var_ratio.to_bits(), da.weighted_var_ratio.to_bits());
                assert_eq!(db.q_ratio.to_bits(), da.q_ratio.to_bits());
                assert_eq!(db.effective_sd.to_bits(), da.effective_sd.to_bits());
            }
        }
    }
}

/// The obs model's own definition: a union index assimilates when some
/// stream scheduled there holds an observed value.
#[test]
fn assimilation_events_are_the_indices_with_an_observed_value() {
    let compiled = sir();
    let obs = bind(&compiled, &WEEKLY_CASES,
        vec![(period(0.0, 10.0), None), (period(17.0, 21.0), None), (period(21.0, 24.0), Some(5.0))]);
    let flags: Vec<(f64, bool)> = (0..obs.n_observations())
        .map(|i| (obs.obs_time(i), obs.assimilates_at(i))).collect();
    let expect = |t: f64| WEEKLY_TIMES.contains(&t) || t == 24.0;
    for &(t, a) in &flags {
        assert_eq!(a, expect(t), "t = {t}: {flags:?}");
    }
}
