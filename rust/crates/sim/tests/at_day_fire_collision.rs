//! item 23 (2026-07-17 spec-review triage): the `at_day` / recurring-schedule
//! guarantee "exactly one fire per period regardless of `dt`" (§13.7) is FALSE
//! on a Grid clock when `dt` is coarse relative to the fire spacing. Fire times
//! map to steps of the run's grid (`effect_plan::fire_step`), and the plan fires
//! one intervention once per step, so when two distinct fire times fall on the
//! same step a fire is dropped.
//!
//! The dispatch pre-flight (`CompiledModel::validate_schedule`, called once at
//! every forward backend's entry) must reject this collision with a clear error
//! rather than silently drop fires — under `FireClock::Grid` only. Under
//! `FireClock::Exact` (ODE, Gillespie) each fire acts at its own instant, so the
//! same schedule drops nothing and is accepted.

use std::collections::HashMap;

use ir::{
    expr::{ConstExpr, Expr},
    intervention::{
        AddAction, Action, FireSource, Intervention, InterventionKind, InterventionSchedule,
        RecurringSchedule,
    },
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::{ParamValue, Parameter},
    transition::{StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::effect_plan::FireClock;

fn grid(dt: f64) -> FireClock {
    FireClock::Grid { dt }
}

/// A model with a single intervention firing on `schedule` over `[0, end]`.
fn model_with_schedule(schedule: InterventionSchedule, end: f64) -> CompiledModel {
    let m = Model {
        ic_grad: Default::default(),
        name: "at_day_collision".into(),
        version: "0.1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "decay".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: Expr::const_(0.0),
            metadata: None,
            draw_method: Default::default(),
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![Intervention {
            name: "cohort_entry".into(),
            base_name: None,
            fire: FireSource::Scheduled(schedule),
            actions: vec![Action::Add(AddAction {
                compartment: "S".into(),
                count: Expr::Const(ConstExpr { value: 1.0 }),
            })],
            kind: InterventionKind::Scenario,
        }],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "p".into(),
            value: ParamValue::Fixed { value: 1.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), 100.0);
            h.insert("I".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, end]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: end,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
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
    CompiledModel::new(m).expect("model compiles")
}

/// A recurring schedule firing every `period` days (phase `at_day = 0`) over
/// `[0, end]`, i.e. at t = 0, period, 2*period, ...
fn recurring(period: f64) -> InterventionSchedule {
    InterventionSchedule::Recurring(RecurringSchedule { start: 0.0, period, end: 10.0, at_day: Some(0.0) })
}

/// dt (4) is coarser than the fire spacing (period 2): fires at t = 2, 4 both
/// fall on step 1, and t = 6, 8 both on step 2 — two fires silently
/// dropped. `validate_schedule` must reject this rather than merge.
#[test]
fn coarse_dt_drops_recurring_fires_and_is_rejected() {
    let model = model_with_schedule(recurring(2.0), 10.0);
    // Sanity: the schedule really does enumerate one fire per period.
    let times = &model.resolve_fire_times(&model.default_params)[0];
    assert_eq!(times, &vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], "one fire per period over [0,10]");

    let err = model
        .validate_schedule(4.0, &model.default_params, 0.0, grid(4.0))
        .expect_err("dt=4 collapses distinct fires onto shared steps — must hard-error");
    let msg = format!("{err}");
    assert!(
        msg.contains("cohort_entry"),
        "the error must name the offending intervention; got: {msg}"
    );
}

/// Negative control: dt == period (2) keeps every fire on its own step, and a
/// finer dt (1) certainly does — neither may be rejected.
#[test]
fn dt_at_or_below_period_keeps_every_fire() {
    let model = model_with_schedule(recurring(2.0), 10.0);
    for dt in [2.0, 1.0, 0.5] {
        model
            .validate_schedule(dt, &model.default_params, 0.0, grid(dt))
            .unwrap_or_else(|e| panic!("dt={dt} <= period=2 must not drop any fire; got {e}"));
    }
}

/// The gh#198 boundary: an explicit `at [...]` list is a DIFFERENT contract —
/// it makes no one-per-period promise, and two listed fires within one dt step
/// are one fire on a Grid clock on purpose. The item-23 check must NOT touch it:
/// `at [2.3, 2.4]` at dt=1 (both on step 2) is allowed.
#[test]
fn explicit_at_times_within_one_dt_is_not_rejected() {
    let model = model_with_schedule(InterventionSchedule::AtTimes(vec![2.3, 2.4]), 5.0);
    model
        .validate_schedule(1.0, &model.default_params, 0.0, grid(1.0))
        .expect("an explicit at[...] list must not be rejected — gh#198 merges it on purpose");
}

/// gh#449: the collision half of `validate_schedule` is factored out as
/// `validate_recurring_dt_collisions(t_start, clock)` so the fit pre-flight — which runs
/// before estimated parameters are resolved — can call it without inventing a
/// parameter vector. Pin that the extraction is behaviour-preserving: for a
/// `Recurring` schedule the two entry points must agree at every dt, since
/// `resolve_fire_times` returns the baked `fire_times` unchanged on that arm.
#[test]
fn param_free_collision_check_agrees_with_validate_schedule() {
    let model = model_with_schedule(recurring(2.0), 10.0);
    for dt in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0] {
        let full = model.validate_schedule(dt, &model.default_params, 0.0, grid(dt));
        let param_free = model.validate_recurring_dt_collisions(0.0, grid(dt));
        assert_eq!(
            full.is_err(),
            param_free.is_err(),
            "dt={dt}: validate_schedule and validate_recurring_dt_collisions must agree; \
             got {full:?} vs {param_free:?}"
        );
    }
    // Negative control: the loop above is only meaningful if it spans BOTH
    // outcomes. dt=4 must reject and dt=1 must accept, or the equality above
    // would hold vacuously (e.g. if both sides always returned Ok).
    assert!(model.validate_recurring_dt_collisions(0.0, grid(4.0)).is_err(),
        "dt=4 > period=2 must be rejected — otherwise the agreement test is vacuous");
    assert!(model.validate_recurring_dt_collisions(0.0, grid(1.0)).is_ok(),
        "dt=1 < period=2 must be accepted — otherwise the agreement test is vacuous");
}

/// The param-free check must not need a parameter vector to be *correct*: an
/// explicit `at [...]` list is still exempt (gh#198), same as through
/// `validate_schedule`.
#[test]
fn param_free_collision_check_leaves_explicit_at_times_alone() {
    let model = model_with_schedule(InterventionSchedule::AtTimes(vec![2.3, 2.4]), 5.0);
    model
        .validate_recurring_dt_collisions(0.0, grid(1.0))
        .expect("an explicit at[...] list must not be rejected — gh#198 merges it on purpose");
}

/// Under `FireClock::Exact` (ODE, Gillespie) every fire acts at its own
/// instant, so a period shorter than `dt` drops nothing: the schedule the Grid
/// clock rejects is accepted, through both entry points.
#[test]
fn exact_clock_accepts_the_schedule_grid_rejects() {
    let model = model_with_schedule(recurring(2.0), 10.0);
    assert!(
        model.validate_schedule(4.0, &model.default_params, 0.0, grid(4.0)).is_err(),
        "precondition: dt=4 collides on the Grid clock"
    );
    model
        .validate_schedule(4.0, &model.default_params, 0.0, FireClock::Exact)
        .expect("under Exact two fires closer than dt are distinct instants");
    model
        .validate_recurring_dt_collisions(0.0, FireClock::Exact)
        .expect("the param-free check does not apply under Exact");
}
