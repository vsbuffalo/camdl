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
use sim::inference::pgas::{log_transition_density_substep, simulate_reference};
use sim::rng::StatefulRng;

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

fn param(name: &str, value: f64) -> Parameter {
    Parameter {
        name: name.into(), value: Some(value),
        bounds: None, prior: None, hierarchical: None,
        transform: None, initial_value: None,
        param_kind: Some("rate".into()), param_dim: None,
    }
}

fn mk_transition(name: &str, src: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
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
        schedule: InterventionSchedule::AtTimes(vec![5.0]),
        actions: vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::Const(ConstExpr { value: 5.0 }),
        })],
        always_active: true,
    };

    let mut init = HashMap::new();
    init.insert("S".into(), 999.0);
    init.insert("I".into(),   0.0);
    init.insert("R".into(),   0.0);

    Model {
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
        initial_conditions: InitialConditions::Explicit(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes((0..=20).map(|t| t as f64).collect()),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 20.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
        },
        interventions: vec![seed_event],
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
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
        schedule: InterventionSchedule::AtTimes(vec![tau]),
        actions: vec![Action::Add(AddAction {
            compartment: "E".into(),
            count: Expr::Const(ConstExpr { value: n_seed as f64 }),
        })],
        always_active: true,
    };

    let mut init = HashMap::new();
    init.insert("S".into(), 1000.0);
    init.insert("E".into(),    0.0);
    init.insert("I".into(),    0.0);
    init.insert("R".into(),    0.0);

    Model {
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
        initial_conditions: InitialConditions::Explicit(init),
        output: OutputConfig {
            times: OutputSchedule::AtTimes((0..=30).map(|t| t as f64).collect()),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 30.0,
            time_semantics: "continuous".into(),
            dt: Some(0.5), rng_seed: Some(42),
        },
        interventions: vec![seed_event],
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
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
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt,
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
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt,
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
