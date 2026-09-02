//! gh#80 — PGAS density evaluator on models with deterministic events.
//!
//! The gh#80 issue claimed `simulate_reference` returns -∞ density on any
//! model with `events { add(...) at [...] }`, with the proposed fix being
//! an event-aware density evaluator. After tracing the data flow in
//! `chain_binomial.rs::step_one` and `pgas.rs::log_transition_density_substep`,
//! the actual story is:
//!
//! 1. `step_one` records `flows` from stochastic transitions ONLY; the event
//!    delta goes through `inject_event_deltas` → `pending_deltas` → direct
//!    write to `counts`. The flows the trajectory carries never include
//!    the event delta.
//! 2. `simulate_reference` captures `counts_before` BEFORE `step_one` runs,
//!    so the trajectory's `counts_before` is pre-event AND pre-stochastic-
//!    transitions.
//! 3. `log_transition_density_substep` recomputes rates from
//!    `counts_before` and scores the recorded flows. Because both
//!    `counts_before` and `flows` came from the same pre-event state,
//!    the math agrees with the simulator — at the event substep all flows
//!    are 0, all rates are 0, density is 0 (finite).
//!
//! So the proposed "apply events to counts_before then evaluate rates"
//! would actually break the density/simulator agreement: it would score
//! the recorded flow=0 against post-event rates that are *non-zero*,
//! producing a *negative* log-density for the outcome the simulator was
//! forced to produce.
//!
//! These tests therefore lock the property the diagnosis identified: the
//! transition density of `simulate_reference`'s own trajectory is finite
//! at its own parameters. They pass on the current code — see
//! `docs/dev/notes/2026-05-25-pgas-event-density-diagnosis.md` for the
//! full trace.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, ConstExpr, Expr, ParamExpr, PopExpr},
    intervention::{Action, AddAction, Intervention, InterventionSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::BoundObs;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{log_transition_density_substep, run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value: value }, param_kind: Some(ir::parameter::ParamKind::Rate), param_dim: None }
}

fn mk_transition(name: &str, src: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![
            StoichiometryEntry(src.into(), -1),
            StoichiometryEntry(dst.into(),  1),
        ],
        rate, metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(), lineage: None,
    }
}

/// Build SIR with `events { boom : add(I, 5) at [5] }`. I is the destination
/// of `infection` and the source of `recovery`, so the seed pulse exercises
/// both density-evaluation paths.
fn sir_with_seed_event() -> Model {
    let n_expr = Expr::bin_op(
        BinOp::Add,
        Expr::Pop(PopExpr { pop: "S".into() }),
        Expr::bin_op(
            BinOp::Add,
            Expr::Pop(PopExpr { pop: "I".into() }),
            Expr::Pop(PopExpr { pop: "R".into() }),
        ),
    );
    let infection_rate = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::Param(ParamExpr { param: "beta".into() }),
            Expr::bin_op(
                BinOp::Mul,
                Expr::Pop(PopExpr { pop: "S".into() }),
                Expr::Pop(PopExpr { pop: "I".into() }),
            ),
        ),
        n_expr,
    );
    let recovery_rate = Expr::bin_op(
        BinOp::Mul,
        Expr::Param(ParamExpr { param: "gamma".into() }),
        Expr::Pop(PopExpr { pop: "I".into() }),
    );

    let seed_event = Intervention {
        name: "boom".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![5.0])),
        actions: vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::Const(ConstExpr { value: 5.0 }),
        })],
        kind: ir::intervention::InterventionKind::Event,
    };

    let mut init = HashMap::new();
    init.insert("S".into(), 999.0);
    init.insert("I".into(),   0.0);
    init.insert("R".into(),   0.0);

    Model {
        ic_grad: Default::default(),
        name: "sir_seed_event".into(),
        version: "0.3".into(), time_unit: "days".into(),
        description: None, origin: None, origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("I"), int_comp("R")],
        transitions: vec![
            mk_transition("infection", "S", "I", infection_rate),
            mk_transition("recovery",  "I", "R", recovery_rate),
        ],
        ode_equations: vec![], time_functions: vec![], tables: vec![],
        observations: vec![],
        parameters: vec![param("beta", 0.4), param("gamma", 0.143)],
        initial_conditions: InitialConditions::constants(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes((0..=20).map(|t| t as f64).collect()),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 20.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        interventions: vec![seed_event],
        presets: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

/// Build SEIR with `events { founders_arrive : add(E, n_seed) at [tau] }`,
/// mirroring the WA seed-timing chapter setup.
fn seir_with_seed_event(n_seed: i64, tau: f64) -> Model {
    let n_expr = Expr::bin_op(
        BinOp::Add, Expr::Pop(PopExpr { pop: "S".into() }),
        Expr::bin_op(
            BinOp::Add, Expr::Pop(PopExpr { pop: "E".into() }),
            Expr::bin_op(
                BinOp::Add, Expr::Pop(PopExpr { pop: "I".into() }),
                Expr::Pop(PopExpr { pop: "R".into() }),
            ),
        ),
    );
    let infection_rate = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::Param(ParamExpr { param: "beta".into() }),
            Expr::bin_op(
                BinOp::Mul,
                Expr::Pop(PopExpr { pop: "S".into() }),
                Expr::Pop(PopExpr { pop: "I".into() }),
            ),
        ),
        n_expr,
    );
    let progression_rate = Expr::bin_op(
        BinOp::Mul,
        Expr::Param(ParamExpr { param: "sigma".into() }),
        Expr::Pop(PopExpr { pop: "E".into() }),
    );
    let recovery_rate = Expr::bin_op(
        BinOp::Mul,
        Expr::Param(ParamExpr { param: "gamma".into() }),
        Expr::Pop(PopExpr { pop: "I".into() }),
    );

    let seed_event = Intervention {
        name: "founders_arrive".into(), base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![tau])),
        actions: vec![Action::Add(AddAction {
            compartment: "E".into(),
            count: Expr::Const(ConstExpr { value: n_seed as f64 }),
        })],
        kind: ir::intervention::InterventionKind::Event,
    };

    let mut init = HashMap::new();
    init.insert("S".into(), 1000.0);
    init.insert("E".into(),    0.0);
    init.insert("I".into(),    0.0);
    init.insert("R".into(),    0.0);

    Model {
        ic_grad: Default::default(),
        name: "seir_seed_event".into(),
        version: "0.3".into(), time_unit: "days".into(),
        description: None, origin: None, origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("E"), int_comp("I"), int_comp("R")],
        transitions: vec![
            mk_transition("infection",   "S", "E", infection_rate),
            mk_transition("progression", "E", "I", progression_rate),
            mk_transition("recovery",    "I", "R", recovery_rate),
        ],
        ode_equations: vec![], time_functions: vec![], tables: vec![],
        observations: vec![],
        parameters: vec![
            param("beta",  0.5),
            param("sigma", 0.33),
            param("gamma", 0.18),
        ],
        initial_conditions: InitialConditions::constants(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes((0..=30).map(|t| t as f64).collect()),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 30.0,
            time_semantics: "continuous".into(),
            dt: Some(0.5), rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        interventions: vec![seed_event],
        presets: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

/// gh#80 acceptance criterion 1: the SIR + seed event trajectory has finite
/// transition density at its own parameters. (Already true on current code;
/// this test pins it as a regression guard.)
#[test]
fn pgas_simulate_reference_finite_density_on_event_model() {
    let model = sir_with_seed_event();
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();
    let dt = 1.0;
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(7);

    let traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let t_start = compiled.model.simulation.t_start;
    let mut total_ll = 0.0;
    for (s, rec) in traj.substeps.iter().enumerate() {
        let t = t_start + s as f64 * dt;
        let td = log_transition_density_substep(
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt, None,
        ).unwrap();
        assert!(
            td.is_finite(),
            "substep {} (t={:.1}) produced non-finite transition density: \
             counts_before={:?}, counts_after={:?}, flows={:?}",
            s, t, rec.counts_before, rec.counts_after, rec.flows,
        );
        total_ll += td;
    }
    assert!(total_ll.is_finite(),
        "total transition log-density must be finite, got {}", total_ll);
}

/// gh#80 acceptance criterion (WA seed-timing variant): SEIR with discrete
/// seeding into E via `events { founders_arrive : add(E, n_seed) at [tau] }`.
#[test]
fn pgas_simulate_reference_finite_density_on_seir_event_model() {
    let model = seir_with_seed_event(5, 5.0);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();
    let dt = 0.5;
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(7);

    let traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let t_start = compiled.model.simulation.t_start;
    let mut total_ll = 0.0;
    let mut event_substep_seen = false;
    for (s, rec) in traj.substeps.iter().enumerate() {
        let t = t_start + s as f64 * dt;
        let td = log_transition_density_substep(
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt, None,
        ).unwrap();
        assert!(
            td.is_finite(),
            "SEIR substep {} (t={:.1}): non-finite td={}, \
             counts_before={:?}, counts_after={:?}, flows={:?}",
            s, t, td, rec.counts_before, rec.counts_after, rec.flows,
        );
        // The substep where the event fires has counts_before E=0 and
        // counts_after E=5: lock the event-into-E identification.
        if rec.counts_before[1] == 0 && rec.counts_after[1] == 5 {
            event_substep_seen = true;
            assert_eq!(td, 0.0,
                "event substep should score 0 transition log-density (all \
                 rates are 0 at pre-event state, all stochastic flows are 0). \
                 Got td = {} — the density evaluator is no longer in sync with \
                 step_one's pre-event rate evaluation.", td);
        }
        total_ll += td;
    }
    assert!(event_substep_seen, "the event-firing substep should appear in the trajectory");
    assert!(total_ll.is_finite());
}

/// gh#80 smoke: PGAS+NUTS on SEIR with a discrete seed event runs cleanly
/// (no -inf density, NUTS adapts to a non-trivial step size, MH acceptance
/// > 0). Marked `#[ignore]` — multi-minute compute, not for every-PR runs.
///
/// To run:
///   cargo test --release -p sim --test pgas_event_density \
///     pgas_nuts_runs_cleanly_on_seir_with_discrete_seed_event \
///     -- --ignored --nocapture
#[test]
#[ignore]
fn pgas_nuts_runs_cleanly_on_seir_with_discrete_seed_event() {
    let model = seir_with_seed_event(5, 10.0);  // event at t=10 (within [0, 30])
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    // Truth trajectory under the seed event.
    let dt = 0.5;
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(101);
    let truth = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Daily-cadence observations of incidence(infection) = transition 0.
    // Sum daily flows on integer days, dropping noise (NegBin obs noise
    // would add variance unrelated to gh#80 — the smoke is about whether
    // the chain *runs cleanly*, not about parameter recovery precision).
    let mut cum_infection: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth.substeps.iter().enumerate() {
        cum_infection += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        // Daily observation
        if (t - t.round()).abs() < 1e-9 && t > 0.0 {
            obs.push(Observation { time: t, value: cum_infection as f64 });
            cum_infection = 0;
        }
    }

    // NegBin obs model.
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),  // infection
            ir_model: ir::observation::ObservationModel {
                name: "cases".into(),
                source: "cases".into(),
                columns: vec![
                    ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                    ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                ],
                scored: "cases".into(),
                emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: ir::observation::Projection::CumulativeFlow("infection".into()),
                projection_state_grad: Default::default(),
                likelihood: ir::observation::Likelihood::NegBinomial(
                    ir::observation::NegBinomialLikelihood {
                        mean: ir::Diffable::new(ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                            bin_op: ir::expr::BinOpExpr {
                                op: ir::expr::BinOp::Add,
                                left: Box::new(ir::expr::Expr::Projected(
                                    ir::expr::ProjectedExpr { projected: () })),
                                right: Box::new(ir::expr::Expr::Const(
                                    ir::expr::ConstExpr { value: 0.1 })),
                            },
                        })),
                        dispersion: ir::Diffable::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 10.0 })),
                    }),
            },
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![
        EstimatedParam {
            name: "beta".into(),
            index: compiled.param_index["beta"],
            initial: 0.5,
            rw_sd: 0.05,
            transform: Transform::Log { lo: 0.1, hi: 2.0 },
            lower: 0.1, upper: 2.0,
            rw_sd_auto: false, perturb_only_at_t0: false,
        },
    ];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        n_particles: 50,
        n_sweeps: 50,
        burn_in: 15,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 8,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let result = run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model, 4242, None, None, "gh80_smoke".into(),
    ).unwrap();

    let final_ll = result.sweeps.last().unwrap().log_complete_data_ll;
    assert!(final_ll.is_finite(), "final LL must be finite, got {}", final_ll);

    let post_burn = &result.sweeps[config.burn_in..];
    let accept_count: usize = post_burn.iter()
        .map(|s| s.accepted.iter().filter(|&&x| x).count()).sum();
    let total_props: usize = post_burn.iter().map(|s| s.accepted.len()).sum();
    let accept_rate = accept_count as f64 / total_props.max(1) as f64;

    eprintln!("[gh#80 smoke] final LL: {:.2}", final_ll);
    eprintln!("[gh#80 smoke] post-burn acceptance: {:.3}", accept_rate);
    eprintln!("[gh#80 smoke] adapted NUTS step: {:.4}",
        result.resume_state.nuts_step_size);

    assert!(accept_rate > 0.0, "NUTS acceptance must be > 0");
    assert!(result.resume_state.nuts_step_size > 1e-8,
        "adapted step size must be > 1e-8 (got {:.2e})",
        result.resume_state.nuts_step_size);
}

/// Stage 3 (2c): exact obs-alignment is REFUSED on models with always-active
/// events. Their firing keys on `round(t/dt)` (intervention.rs), which a
/// shortened exact substep would shift off the intended step — a silent
/// mis-fire. `run_pgas` returns a clean error before doing any work. (The guard
/// fires ahead of the grid build, so empty observations suffice to reach it.)
#[test]
fn exact_alignment_rejected_on_always_active_event_model() {
    let model = seir_with_seed_event(5, 10.0); // founders_arrive: always_active
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    assert!(compiled.model.interventions.iter().any(|iv| iv.kind.is_event()),
        "fixture precondition: model has an always-active event");
    let params = compiled.default_params.clone();

    let if2_params = vec![EstimatedParam {
        name: "beta".into(),
        index: compiled.param_index["beta"],
        initial: 0.5, rw_sd: 0.05,
        transform: Transform::Log { lo: 0.1, hi: 2.0 },
        lower: 0.1, upper: 2.0, rw_sd_auto: false, perturb_only_at_t0: false,
    }];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        n_particles: 10, n_sweeps: 1, burn_in: 0, thin: 1, dt: 0.5,
        use_nuts: false, dense_mass: false, max_tree_depth: 4,
        tempering: vec![1.0], trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Exact,
    };

    match run_pgas(
        &compiled, &if2_params, &priors, &params, &config,
        &[], &obs_model, 1, None, None, "exact_event_guard".into(),
    ) {
        Ok(_) => panic!("exact + always-active event must be refused"),
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(msg.contains("always-active"),
                "guard error should name always-active events, got: {msg}");
        }
    }
}
