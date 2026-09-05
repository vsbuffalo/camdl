//! What the model managed at an observation: the refusal names the measurement,
//! not its position in the observation queue.
//!
//! A PGAS chain whose initialization swarm loses support used to report
//! `every particle scored -inf at observation 16 (substep 96)`. With several
//! streams bound and interleaved in time that index names nothing a modeller
//! can act on, and it is not a stable identifier either: it is a position on
//! the union axis, so unbinding a stream renumbers it and two ablations cannot
//! be compared. These tests pin the replacement — a per-stream record keyed on
//! `(stream, time)` that says what the ensemble managed there and, when the
//! observation model refused, which guard fired.
//!
//! Two of the tests drive the whole initialization pass (`unconditional_smc_pass`)
//! because the wiring is where the mistake would be — in particular that dead
//! particles are excluded from the reduction. The rest call
//! `MultiStreamObsModel::stream_attempts` directly, because the property under
//! test is a property of that reduction and a simulation would only add noise
//! between the fixture and the assertion.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, Expr, UnOp},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        BetaBinomialLikelihood, BinomialLikelihood, ColumnRole, Likelihood,
        ObsColumn, ObservationModel as IrObs, ObservationSchedule, PoissonLikelihood, Projection,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::error::InitFallback;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{
    BoundObs, MultiStreamObsModel, ObsCell, StreamProjection, StreamSpec,
};
use sim::inference::obs_attempt::{NegInfCause, ObsCellState, StreamAttempt};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{build_obs_at_substep, EffectFiring};
use sim::inference::pgas_init::{unconditional_smc_pass, UnconditionalPass};

const DT: f64 = 1.0;
const N_PARTICLES: usize = 200;
const SEED: u64 = 11;
/// A calendar anchor, so the refusal can name a date and not only a model time.
const ORIGIN: &str = "2019-01-01";

// ── expression helpers ─────────────────────────────────────────────────────

fn konst(v: f64) -> Expr {
    Expr::const_(v)
}
fn projected() -> Expr {
    Expr::Projected(ir::expr::ProjectedExpr { projected: () })
}
fn pop(name: &str) -> Expr {
    Expr::pop(name)
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::bin_op(op, a, b)
}
fn cond(pred: Expr, then: Expr, else_: Expr) -> Expr {
    Expr::Cond(ir::expr::CondWrap {
        cond: ir::expr::CondExpr {
            pred: Box::new(pred),
            then: Box::new(then),
            else_: Box::new(else_),
        },
    })
}

fn value_column(name: &str) -> ObsColumn {
    ObsColumn {
        name: name.into(),
        role: ColumnRole::Value(ir::parameter::ParamKind::Count),
    }
}

/// An observation block carrying `likelihood` over `projection`. `aux_columns`
/// are the declared data columns the likelihood may read by name.
fn obs_block(
    name: &str,
    projection: Projection,
    likelihood: Likelihood,
    aux_columns: &[&str],
) -> IrObs {
    let mut columns = vec![
        ObsColumn { name: "time".into(), role: ColumnRole::Time },
        value_column(name),
    ];
    columns.extend(aux_columns.iter().map(|c| value_column(c)));
    IrObs {
        name: name.into(),
        source: name.into(),
        columns,
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection,
        projection_state_grad: Default::default(),
        likelihood,
    }
}

// ── the models ─────────────────────────────────────────────────────────────

/// `sir_basic` at `I₀ = 0`, anchored to a calendar origin.
///
/// With no infectives the infection rate `β·S·I/N` is identically zero, so
/// EVERY particle carries `I = 0` at every observation. Scored against a
/// positive prevalence under `poisson(rate = projected)` — no floor — that is
/// exactly `-inf` for every live particle, and the swarm loses support at the
/// first observation. Nothing dies: no transition fires, so there is no
/// overshoot and no numerical collapse.
fn silent_sir() -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.origin = Some(ORIGIN.into());
    m.origin_rata_die = Some(ir::caltime::rata_die(2019, 1, 1));
    m.observations = vec![obs_block(
        "prevalence",
        Projection::CurrentPop("I".into()),
        Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(projected()) }),
        &[],
    )];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = 20.0;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.35,
            "gamma" => 0.12,
            "N0" => 400.0,
            "I0" => 0.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

/// A model that kills SOME particles at substep 1 while the survivors score
/// `-inf` on the observation there — the mixture the report must not collapse.
///
/// `A --> B` is an ordinary chain-binomial flow, so `B` differs across
/// particles after substep 0. The second transition's rate is
/// `if t < 0.5 then 0 else if B > 4.5 then 0 else sqrt(B − 5)`: a particle
/// whose `B` came out below 5 evaluates `sqrt` of a negative, which the
/// propensity evaluator turns into a per-particle `NumericalCollapse` and the
/// death mask absorbs. A particle whose `B` is at least 5 gets a rate of
/// exactly zero, so nothing flows into `W`, which therefore stays empty for
/// every survivor — and `poisson(rate = W)` against a positive observed count
/// is `-inf`. The time guard exists because `B` is zero at substep 0, where the
/// unguarded rate would kill the whole swarm before it ever differed.
fn mixed_death_model() -> Arc<CompiledModel> {
    let guarded_rate = cond(
        bin(BinOp::Lt, Expr::time(), konst(0.5)),
        konst(0.0),
        cond(
            bin(BinOp::Gt, pop("B"), konst(4.5)),
            konst(0.0),
            Expr::un_op(UnOp::Sqrt, bin(BinOp::Sub, pop("B"), konst(5.0))),
        ),
    );
    let m = Model {
        ic_grad: Default::default(),
        name: "mixed_death".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: Some(ORIGIN.into()),
        origin_rata_die: Some(ir::caltime::rata_die(2019, 1, 1)),
        compartments: ["A", "B", "W"]
            .iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
                name: "spread".into(),
                stoichiometry: vec![
                    StoichiometryEntry("A".into(), -1),
                    StoichiometryEntry("B".into(), 1),
                ],
                // The chain-binomial backend reads a transition rate as a
                // TOTAL propensity and divides by the source count, so this is
                // a per-capita hazard of 0.0513 out of A = 100: the flow into
                // `B` is Binomial(100, 1 − exp(−0.0513)), mean 5.
                rate: bin(BinOp::Mul, konst(0.0513), pop("A")),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
            Transition {
                rate_state_grad: Default::default(),
                name: "leak".into(),
                stoichiometry: vec![
                    StoichiometryEntry("A".into(), -1),
                    StoichiometryEntry("W".into(), 1),
                ],
                rate: guarded_rate,
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![obs_block(
            "leaked",
            Projection::CurrentPop("W".into()),
            Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(projected()) }),
            &[],
        )],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "dummy".into(),
            value: ParamValue::Fixed { value: 0.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants(HashMap::from([
            ("A".into(), 100.0),
            ("B".into(), 0.0),
            ("W".into(), 0.0),
        ])),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 2.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 2.0,
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
    Arc::new(CompiledModel::new(m).expect("compile mixed_death"))
}

/// A three-compartment carrier with one flow, used by the tests that call the
/// reduction directly. Nothing is simulated on it: it exists so streams have
/// compartments and a transition to project from.
fn carrier(observations: Vec<IrObs>) -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "carrier".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: Some(ORIGIN.into()),
        origin_rata_die: Some(ir::caltime::rata_die(2019, 1, 1)),
        compartments: ["S", "I", "R"]
            .iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: konst(0.1),
            metadata: None,
            draw_method: DrawMethod::Deterministic,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations,
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "dummy".into(),
            value: ParamValue::Fixed { value: 0.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants(HashMap::from([
            ("S".into(), 100.0),
            ("I".into(), 1.0),
            ("R".into(), 0.0),
        ])),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 30.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 30.0,
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
    Arc::new(CompiledModel::new(m).expect("compile carrier"))
}

// ── driving the initialization pass ────────────────────────────────────────

/// Run the unconditional pass and return the reason it lost support.
fn collapse(
    compiled: &Arc<CompiledModel>,
    obs_model: &MultiStreamObsModel,
    obs: &[Observation],
    n_substeps: usize,
) -> InitFallback {
    let grid: Vec<(f64, f64)> = (0..n_substeps).map(|s| (s as f64 * DT, DT)).collect();
    let obs_at_substep = build_obs_at_substep(obs, compiled.model.simulation.t_start, DT)
        .expect("obs_at_substep");
    let pass = unconditional_smc_pass(
        compiled,
        &compiled.default_params.clone(),
        &grid,
        N_PARTICLES,
        DT,
        obs_model,
        SEED,
        &obs_at_substep,
        EffectFiring::default(),
        sim::rng::BinomialAlgorithm::Btpe,
    )
    .expect("unconditional pass");
    match pass {
        UnconditionalPass::NoSupport(r) => r,
        UnconditionalPass::Path(_) => {
            panic!("the fixture must lose support, else the test proves nothing")
        }
    }
}

// ── 1. the refusal names the measurement, not the index ────────────────────

/// A stream that cannot produce a positive observation refuses naming that
/// stream and that date.
///
/// This is the property the index lacks. `observation 0` is a position on the
/// union axis whose composition changes with the bound stream set;
/// `prevalence` at `2019-01-06` is the measurement itself, and it survives
/// unbinding a sibling (see `unbinding_a_stream_changes_nothing_for_the_rest`).
///
/// Deliberately asserted through the refusal's own prose, which is what a user
/// reads first and what the CLI already puts in `BadInit.reason`. The prose is
/// rendered from the structured records rather than written beside them, so
/// this also pins that the two cannot drift.
#[test]
fn a_stream_that_cannot_explain_its_data_is_named_with_its_date() {
    let compiled = silent_sir();
    let times = vec![5.0, 10.0, 15.0, 20.0];
    let values = vec![3.0, 5.0, 13.0, 14.0];
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![1]), // `I` is int-local index 1
            compiled.model.observations[0].clone(),
            dense_cells(values.clone()),
            times.clone(),
        )])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");
    let obs: Vec<Observation> = times
        .iter()
        .zip(&values)
        .map(|(&time, &value)| Observation { time, value })
        .collect();

    let fallback = collapse(&compiled, &obs_model, &obs, 20);
    let msg = format!("{fallback}");

    assert!(
        msg.contains("stream 'prevalence'"),
        "the refusal must name the STREAM that could not explain its data: {msg}"
    );
    assert!(
        msg.contains("2019-01-06"),
        "the refusal must name the observation's calendar date (t = 5 days after \
         {ORIGIN}): {msg}"
    );
    assert!(
        msg.contains("observed 3"),
        "the refusal must name the value that could not be explained: {msg}"
    );
    // And it must still never be phrased as a claim about theta.
    assert!(
        !msg.contains("infeasible"),
        "an initialization failure must not assert p(y | theta) = 0: {msg}"
    );
}

/// The structured half of the same refusal. `projected_max` exactly zero
/// against a positive `y_obs` is the reading the proposal calls structural:
/// the model cannot produce this observation at all, as opposed to producing
/// it rarely.
#[test]
fn the_refusal_carries_the_structured_record_the_prose_is_rendered_from() {
    let compiled = silent_sir();
    let times = vec![5.0, 10.0, 15.0, 20.0];
    let values = vec![3.0, 5.0, 13.0, 14.0];
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![1]),
            compiled.model.observations[0].clone(),
            dense_cells(values.clone()),
            times.clone(),
        )])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");
    let obs: Vec<Observation> = times
        .iter()
        .zip(&values)
        .map(|(&time, &value)| Observation { time, value })
        .collect();

    let fallback = collapse(&compiled, &obs_model, &obs, 20);
    let attempts = fallback.attempts();
    assert_eq!(attempts.len(), 1, "one record per declared stream: {attempts:?}");
    let a = &attempts[0];
    assert_eq!(a.stream, "prevalence");
    assert_eq!(a.time, 5.0);
    assert_eq!(a.date.as_deref(), Some("2019-01-06"));
    assert_eq!(a.cell, ObsCellState::Scored { y_obs: 3.0 });
    assert_eq!(
        a.projected_max,
        Some(0.0),
        "with I0 = 0 no particle can produce a single infective, so the largest \
         projection across the swarm is exactly zero"
    );
    assert_eq!(a.n_projected_zero, a.n_live);
    assert_eq!(a.n_projected_nan, 0);
    assert_eq!(a.n_dead, 0, "no transition fires here, so nothing dies");
    assert_eq!(a.n_live, N_PARTICLES);
    assert_eq!(a.n_particles, N_PARTICLES);
    assert_eq!(a.n_neg_inf, a.n_live, "every live particle refused");
    assert_eq!(
        a.neg_inf_causes,
        vec![(NegInfCause::ObservedOutsideSupport { observed: 3.0 }, N_PARTICLES)],
        "a positive count under a zero Poisson rate is the support guard, not a \
         NaN or a domain violation"
    );
}

// ── 2. a NaN shape parameter, against a finite projection ──────────────────

/// A `beta_binomial` whose shape parameters go NaN reports every live particle
/// as `-inf` with the NaN-shape cause, against a FINITE `projected_max`.
///
/// The shapes are written `projected · denom / denom` with `denom` an aux data
/// column bound to zero — the `k · projected / denom` form the proposal names,
/// which is NaN exactly where the denominator vanishes. That distinction is the
/// point: the projection is finite and positive, so nothing about the modelled
/// flow is wrong; the likelihood's own argument expression is what fails, and a
/// report that only said "this stream returned -inf" would send a modeller to
/// the wrong place.
#[test]
fn nan_shape_parameters_are_reported_as_such_against_a_finite_projection() {
    let denom = || Expr::obs_column_ref("denom");
    let nan_shape = || {
        ir::Diffable::new(bin(
            BinOp::Div,
            bin(BinOp::Mul, projected(), denom()),
            denom(),
        ))
    };
    let block = obs_block(
        "confirmed",
        Projection::CumulativeFlow("infection".into()),
        Likelihood::BetaBinomial(BetaBinomialLikelihood {
            n: Expr::obs_column_ref("tested"),
            alpha: nan_shape(),
            beta: nan_shape(),
        }),
        &["tested", "denom"],
    );
    let compiled = carrier(vec![block.clone()]);
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),
            ir_model: block,
            observations: dense_cells(vec![18.0]),
            obs_times: vec![5.0],
            aux: vec![vec![("tested".into(), 40.0), ("denom".into(), 0.0)]],
        }])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");

    let params = compiled.default_params.clone();
    let counts = vec![90_i64, 11, 0];
    let accs: Vec<Vec<u64>> = vec![vec![7], vec![12], vec![3]];
    let live: Vec<(&[u64], &[i64])> =
        accs.iter().map(|a| (a.as_slice(), counts.as_slice())).collect();

    let attempts = obs_model.stream_attempts(0, &live, 0, &params);
    let a = &attempts[0];
    assert_eq!(a.stream, "confirmed");
    assert_eq!(a.n_live, 3);
    assert_eq!(a.n_neg_inf, a.n_live, "every live particle must refuse: {a}");
    assert_eq!(
        a.projected_max,
        Some(12.0),
        "the projection itself is finite — 12 confirmations is the largest bin \
         in the swarm — so the failure is not in the projection: {a}"
    );
    assert_eq!(a.n_projected_nan, 0, "the PROJECTION is finite; the shapes are not");
    assert_eq!(
        a.neg_inf_causes,
        vec![(NegInfCause::ArgumentNaN { arg: "alpha".into() }, 3)],
        "the report must name the guard that fired — and 'alpha', not 'beta', \
         because that is the argument the value function reads first: {a}"
    );
}

/// The `k > n` guard, reported as itself rather than folded into the NaN case.
///
/// A correction to the proposal, which lists `k > n` as a live beta-binomial
/// failure mode with `n = tests` bound from an aux column. On that exact
/// configuration the guard is UNREACHABLE at runtime: `BoundObs::bind` refuses
/// the data before a chain starts (its "exceeds denominator" finding), and the
/// same holds for `n == 0` against a positive count. The guard is reachable
/// only when `n` is an EXPRESSION the binder cannot evaluate ahead of a
/// trajectory -- here the modelled population `S + I + R` -- and in that form
/// it is state-dependent, so the proposal's reading of it as constant across
/// particles and across theta does not carry over.
///
/// It is still worth resolving: `k > n` and a NaN shape send a modeller to
/// different places, and only the guard identity distinguishes them.
#[test]
fn a_count_exceeding_an_expression_denominator_names_that_guard() {
    let block = obs_block(
        "confirmed",
        Projection::CumulativeFlow("infection".into()),
        Likelihood::BetaBinomial(BetaBinomialLikelihood {
            // A modelled denominator, not a data column: the survey covers the
            // whole modelled population.
            n: Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]),
            alpha: ir::Diffable::new(bin(BinOp::Add, projected(), konst(1.0))),
            beta: ir::Diffable::new(konst(2.0)),
        }),
        &[],
    );
    let compiled = carrier(vec![block.clone()]);
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::FlowSum(vec![0]),
            block,
            dense_cells(vec![50.0]),
            vec![5.0],
        )])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");

    let params = compiled.default_params.clone();
    // 50 confirmations against a modelled population of 15.
    let counts = vec![10_i64, 5, 0];
    let accs: Vec<Vec<u64>> = vec![vec![7], vec![55]];
    let live: Vec<(&[u64], &[i64])> =
        accs.iter().map(|a| (a.as_slice(), counts.as_slice())).collect();

    let a = &obs_model.stream_attempts(0, &live, 0, &params)[0];
    assert_eq!(a.n_neg_inf, 2, "both live particles must refuse: {a}");
    assert_eq!(
        a.neg_inf_causes,
        vec![(NegInfCause::CountExceedsTrials, 2)],
        "the count guard, not the shape guards -- the two particles' \
         projections differ (7 vs 55) and neither shape is NaN: {a}"
    );
}

/// The other half of that correction, as a test so it cannot rot: with the
/// denominator bound from a data column, `k > n` never reaches the runtime,
/// because binding refuses the data.
#[test]
fn a_data_column_denominator_smaller_than_its_count_is_refused_at_bind() {
    let block = obs_block(
        "confirmed",
        Projection::CumulativeFlow("infection".into()),
        Likelihood::BetaBinomial(BetaBinomialLikelihood {
            n: Expr::obs_column_ref("tested"),
            alpha: ir::Diffable::new(bin(BinOp::Add, projected(), konst(1.0))),
            beta: ir::Diffable::new(konst(2.0)),
        }),
        &["tested"],
    );
    let bound = BoundObs::bind(vec![StreamSpec {
        projection: StreamProjection::FlowSum(vec![0]),
        ir_model: block,
        observations: dense_cells(vec![50.0]),
        obs_times: vec![5.0],
        // 50 confirmations out of 40 specimens.
        aux: vec![vec![("tested".into(), 40.0)]],
    }]);
    let err = match bound {
        Ok(_) => panic!("binding must refuse a count above its bound denominator"),
        Err(report) => report,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exceeds denominator"),
        "the refusal must be the denominator check, not something else: {msg}"
    );
}

// ── 3. dead particles are not an observation refusal ───────────────────────

/// A mixture of dead and `-inf`-scoring particles reports the two separately.
///
/// A particle killed earlier by the process model carries `-inf` without the
/// observation model having been consulted at all. Pooling the two would
/// report a process-model failure as a unanimous observation refusal, which is
/// the exact confusion this record exists to remove: with `n_dead` near
/// `n_particles` the finding is about the process model, and the reader has to
/// be able to see that.
#[test]
fn a_dead_particle_is_counted_apart_from_one_the_observation_refused() {
    let compiled = mixed_death_model();
    let obs = vec![Observation { time: 2.0, value: 5.0 }];
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![2]), // `W`
            compiled.model.observations[0].clone(),
            dense_cells(vec![5.0]),
            vec![2.0],
        )])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");

    let fallback = collapse(&compiled, &obs_model, &obs, 2);
    let a = &fallback.attempts()[0];

    // The fixture's premise: there really was a mixture. Without both halves
    // the separation below is vacuous.
    assert!(
        a.n_dead > 0,
        "the fixture must kill SOME particles at the scoring substep: {a}"
    );
    assert!(
        a.n_live > 0,
        "the fixture must leave SOME particles alive to be refused: {a}"
    );
    assert_eq!(a.n_live + a.n_dead, N_PARTICLES, "the denominators must close: {a}");
    assert_eq!(a.n_particles, N_PARTICLES);

    // The separation itself: every count about the observation model covers
    // the live particles only.
    assert_eq!(
        a.n_neg_inf, a.n_live,
        "n_neg_inf counts LIVE refusals; it must not absorb the dead: {a}"
    );
    assert_eq!(
        a.n_projected_zero, a.n_live,
        "the projection summary is over live particles too: {a}"
    );
    let counted: usize = a.neg_inf_causes.iter().map(|(_, n)| n).sum();
    assert_eq!(counted, a.n_neg_inf, "the cause tally must sum to n_neg_inf: {a}");

    let msg = format!("{fallback}");
    assert!(
        msg.contains(&format!("{} of {} particles already dead", a.n_dead, N_PARTICLES)),
        "the prose must say how many particles never reached the observation \
         model, or a process-model failure reads as an observation refusal: {msg}"
    );
}

// ── 4. the three cell states, and the fourth thing that is not an error ────

/// `Scored`, `Hole` and `NotScheduled` are distinguishable — and a genuine
/// zero-density row is a fourth thing that is routine, not an error.
///
/// All four return `0.0` from the scoring path: a stream on another cadence
/// contributes no term, a hole marginalizes its missing value, and
/// `beta_binomial(k = 0 | n = 0)` is exactly `0.0` because zero trials have
/// exactly one outcome — routine surveillance data for a day nobody was
/// examined. One number for four states is what this record refuses to do.
#[test]
fn the_cell_states_are_distinguishable_and_a_zero_effort_row_is_not_a_failure() {
    let scored = obs_block(
        "scored",
        Projection::CurrentPop("I".into()),
        Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(bin(BinOp::Add, projected(), konst(1.0))),
        }),
        &[],
    );
    let hole = obs_block(
        "hole",
        Projection::CurrentPop("I".into()),
        Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(bin(BinOp::Add, projected(), konst(1.0))),
        }),
        &[],
    );
    let elsewhere = obs_block(
        "elsewhere",
        Projection::CurrentPop("I".into()),
        Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(bin(BinOp::Add, projected(), konst(1.0))),
        }),
        &[],
    );
    let zero_effort = obs_block(
        "zero_effort",
        Projection::CurrentPop("I".into()),
        Likelihood::Binomial(BinomialLikelihood {
            n: Expr::obs_column_ref("tested"),
            p: ir::Diffable::new(konst(0.3)),
        }),
        &["tested"],
    );
    let compiled = carrier(vec![
        scored.clone(),
        hole.clone(),
        elsewhere.clone(),
        zero_effort.clone(),
    ]);
    let prevalence = || StreamProjection::IntCompSum(vec![1]); // `I`
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![
            StreamSpec::dense(prevalence(), scored, dense_cells(vec![4.0]), vec![5.0]),
            // A hole: scheduled at the same time, no value.
            StreamSpec::dense(prevalence(), hole, vec![None], vec![5.0]),
            // A sibling on another cadence entirely.
            StreamSpec::dense(prevalence(), elsewhere, dense_cells(vec![9.0]), vec![12.0]),
            // Zero trials, zero positives — a day nobody was examined.
            StreamSpec {
                projection: prevalence(),
                ir_model: zero_effort,
                observations: dense_cells(vec![0.0]),
                obs_times: vec![5.0],
                aux: vec![vec![("tested".into(), 0.0)]],
            },
        ])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");

    let params = compiled.default_params.clone();
    let counts = vec![90_i64, 6, 0];
    let acc: Vec<u64> = vec![];
    let live: Vec<(&[u64], &[i64])> = vec![(acc.as_slice(), counts.as_slice())];

    // Union index 0 is t = 5.0; `elsewhere` lives at t = 12.0 (union index 1).
    let attempts = obs_model.stream_attempts(0, &live, 0, &params);
    let by_name = |n: &str| -> &StreamAttempt {
        attempts.iter().find(|a| a.stream == n).expect("stream present")
    };

    assert_eq!(by_name("scored").cell, ObsCellState::Scored { y_obs: 4.0 });
    assert_eq!(by_name("hole").cell, ObsCellState::Hole);
    assert_eq!(by_name("elsewhere").cell, ObsCellState::NotScheduled);
    assert_eq!(
        by_name("zero_effort").cell,
        ObsCellState::Scored { y_obs: 0.0 },
        "a zero-effort row IS scored — it is an observation with the value 0, \
         not a missing one"
    );

    // What separates the fourth from the other three: it was scored, and it
    // scored finitely.
    assert_eq!(
        by_name("zero_effort").n_neg_inf,
        0,
        "n = 0, k = 0 is exactly probability one, not an impossible observation"
    );
    for name in ["hole", "elsewhere"] {
        assert_eq!(
            by_name(name).n_neg_inf,
            0,
            "{name} contributes no likelihood factor, so it cannot refuse"
        );
    }
    assert_eq!(
        by_name("elsewhere").projected_max,
        None,
        "a stream not scheduled here has no observation to read its accumulator \
         against, so no projection is summarized"
    );
    assert_eq!(
        by_name("hole").projected_max,
        Some(6.0),
        "a hole IS scheduled: the projection it would have been scored against \
         is real and is reported"
    );
}

// ── 5. renumbering: the identifier survives unbinding a sibling ────────────

/// Unbinding a stream changes no field of the records for the streams that
/// remain.
///
/// This is the property the queue index lacks and the reason the identifier
/// changed: dropping `es` shifts every later union index, so a refusal keyed on
/// the index cannot be compared between the two ablations, while one keyed on
/// `(stream, time)` is unchanged.
#[test]
fn unbinding_a_stream_changes_nothing_for_the_rest() {
    let blocks: Vec<IrObs> = ["afp", "es", "sero"]
        .iter()
        .map(|n| {
            obs_block(
                n,
                Projection::CurrentPop("I".into()),
                Likelihood::Poisson(PoissonLikelihood {
                    rate: ir::Diffable::new(projected()),
                }),
                &[],
            )
        })
        .collect();
    let compiled = carrier(blocks.clone());
    let params = compiled.default_params.clone();
    // `es` sits between the other two on the union axis, so dropping it shifts
    // `sero`'s index from 2 to 1 — the renumbering under test.
    let cadence = [("afp", 5.0), ("es", 8.0), ("sero", 12.0)];

    let build = |keep: &[&str]| -> MultiStreamObsModel {
        let specs: Vec<StreamSpec> = blocks
            .iter()
            .zip(&cadence)
            .filter(|(_, (name, _))| keep.contains(name))
            .map(|(b, (_, t))| {
                StreamSpec::dense(
                    StreamProjection::IntCompSum(vec![1]),
                    b.clone(),
                    dense_cells(vec![4.0]),
                    vec![*t],
                )
            })
            .collect();
        MultiStreamObsModel::new(BoundObs::bind(specs).expect("bind").0, compiled.clone())
            .expect("obs model")
    };

    let counts = vec![90_i64, 0, 0];
    let acc: Vec<u64> = vec![];
    let live: Vec<(&[u64], &[i64])> = vec![(acc.as_slice(), counts.as_slice())];

    let all = build(&["afp", "es", "sero"]);
    let dropped = build(&["afp", "sero"]);

    // `sero` is at union index 2 with three streams bound and at index 1 with
    // two — the renumbering the index-keyed report could not survive.
    assert_eq!(all.obs_time(2), 12.0);
    assert_eq!(dropped.obs_time(1), 12.0);

    let find = |m: &MultiStreamObsModel, obs_idx: usize, name: &str| -> StreamAttempt {
        m.stream_attempts(obs_idx, &live, 0, &params)
            .into_iter()
            .find(|a| a.stream == name)
            .expect("stream present")
    };
    assert_eq!(
        find(&all, 2, "sero"),
        find(&dropped, 1, "sero"),
        "every field of the record for a surviving stream must be unchanged by \
         unbinding a sibling"
    );
    assert_eq!(find(&all, 0, "afp"), find(&dropped, 0, "afp"));
}

// ── 6. a NaN projection is reported, never absorbed ────────────────────────

/// An `Expr` projection that goes NaN yields `projected_max: None` with
/// `n_projected_nan` set, and neither panics nor reports a spurious maximum.
///
/// `I/(S+I+R)` is NaN at zero population — the division-by-zero the expression
/// evaluator returns rather than erroring on, because this is a likelihood
/// argument and not a rate. `f64::max` would silently return the non-NaN
/// operand (and `-inf` when every operand is NaN, a maximum no particle held);
/// `partial_cmp().unwrap()` would panic on a path that is already handling a
/// failure. Neither is acceptable, so the NaNs are counted and excluded.
#[test]
fn a_nan_projection_is_counted_and_excluded_rather_than_absorbed() {
    let proportion = bin(
        BinOp::Div,
        pop("I"),
        Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]),
    );
    let block = obs_block(
        "positivity",
        Projection::DerivedExpr(proportion.clone()),
        Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(projected()) }),
        &[],
    );
    let compiled = carrier(vec![block.clone()]);
    let params = compiled.default_params.clone();
    let resolved = StreamProjection::from_ir(
        &Projection::DerivedExpr(proportion),
        &compiled,
        "positivity",
    )
    .expect("resolve projection");
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            resolved,
            block,
            dense_cells(vec![1.0]),
            vec![5.0],
        )])
        .expect("bind")
        .0,
        compiled.clone(),
    )
    .expect("obs model");

    let acc: Vec<u64> = vec![];
    let empty = vec![0_i64, 0, 0]; // zero population: I/(S+I+R) is 0/0
    let peopled = vec![90_i64, 10, 0];

    // (a) Every live projection is NaN: there is no maximum, and reporting one
    // would be reporting a value no particle held.
    let all_nan: Vec<(&[u64], &[i64])> =
        vec![(acc.as_slice(), empty.as_slice()), (acc.as_slice(), empty.as_slice())];
    let a = &obs_model.stream_attempts(0, &all_nan, 0, &params)[0];
    assert_eq!(a.n_projected_nan, 2, "both NaN projections must be counted: {a}");
    assert_eq!(a.projected_max, None, "there is no maximum to report: {a}");
    assert_eq!(a.projected_median, None, "and no median: {a}");
    assert_eq!(a.n_projected_zero, 0, "a NaN is not a zero: {a}");

    // (b) One live particle has a finite projection: it is the maximum, and the
    // NaN is still reported rather than silently dropped.
    let mixed: Vec<(&[u64], &[i64])> =
        vec![(acc.as_slice(), empty.as_slice()), (acc.as_slice(), peopled.as_slice())];
    let b = &obs_model.stream_attempts(0, &mixed, 0, &params)[0];
    assert_eq!(b.n_projected_nan, 1, "the NaN must still be counted: {b}");
    assert_eq!(b.projected_max, Some(0.1), "10 of 100 is the one real projection: {b}");
    assert_eq!(b.n_live, 2, "the NaN particle was live and is in the denominator: {b}");
}

/// A hole carries no observed value, so `ObsCell` must not be confused with an
/// observed zero anywhere in this path.
#[test]
fn a_hole_is_not_an_observed_zero() {
    assert_ne!(Some(ObsCell::Scalar(0.0)), None::<ObsCell>);
    assert_eq!(ObsCellState::Hole.y_obs(), None);
    assert_eq!(ObsCellState::Scored { y_obs: 0.0 }.y_obs(), Some(0.0));
}
