//! gh#53 regression test: integrated intervention fire count over a
//! fixed wall-time interval must be **dt-invariant**.
//!
//! Pre-fix, `CompiledModel.fire_steps` was a baked step-index set
//! computed at compile time using `model.simulation.dt`. At runtime,
//! a different `dt` (e.g. via `camdl pfilter --dt 0.5` against a
//! model declared at `dt = 1.0`) walked the integrator at that
//! finer dt while still checking against the compile-time step
//! indices. The cohort impulse on He et al. 2010 measles fired at
//! the wrong wall times and (because `every 365.25 days` was
//! interpreted as steps not days) at the wrong period — fired
//! roughly twice per year, half a year early. Visible only against
//! pomp on the He2010 reproducer; gh#52 Richardson ladder caught it.
//!
//! These tests pin the structural invariant: for any model with a
//! periodic intervention, the **count of fires over a fixed wall-
//! time interval** must equal the count any other dt would produce.
//! That's the property the bug violated, regardless of the exact
//! mechanism (step-baking vs runtime-rescaling).

use std::collections::HashMap;

use ir::{
    expr::{ConstExpr, Expr},
    intervention::{Action, AbsoluteTransfer, Intervention, InterventionSchedule, RecurringSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    effects::due_effects,
    intervention::apply_effect_batch,
    schedule::EffectBatch,
    state::{IntState, RealState},
};

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

/// Test adapter preserving the old `apply_interventions_at` call shape: derive
/// the due batch at the boundary `t` (the seam `due_effects` now owns) and apply
/// the scheduled interventions. Byte-identical to the retired
/// `apply_interventions_at` — the firing decision still keys on `grid_dt`.
#[allow(clippy::too_many_arguments)]
fn apply_interventions_at(
    t: f64,
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    dt: f64,
    grid_dt: f64,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
    _tolerance: f64,
) -> Result<bool, sim::error::SimError> {
    let mut batch = EffectBatch::default();
    due_effects(model, fire_steps, t, grid_dt, &mut batch);
    apply_effect_batch(t, model, &batch.intervention_idx, dt, int_s, real_s, params)
}

/// Two compartments S, V, and a single periodic intervention
/// `transfer S → V` of 1 unit, every `period` time units, starting
/// at `at_day`. No transitions; the only state change is the
/// intervention firing.
fn periodic_xfer_model(at_day: f64, period: f64, end: f64, model_dt: Option<f64>) -> Model {
    let intervention = Intervention {
        name: "periodic_xfer".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::Recurring(RecurringSchedule {
            at_day: Some(at_day),
            start: 0.0,
            end,
            period,
        })),
        kind: ir::intervention::InterventionKind::Scenario,
        actions: vec![
            Action::AbsoluteTransfer(AbsoluteTransfer {
                src: "S".into(),
                dst: "V".into(),
                count: Expr::Const(ConstExpr { value: 1.0 }),
            }),
        ],
    };

    Model {
        ic_grad: Default::default(),
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("V")],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![intervention],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![],
        initial_conditions: InitialConditions::Parameterized(HashMap::new()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: end,
            time_semantics: "continuous".into(),
            dt: model_dt,
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

/// Walk an integer-step grid at `runtime_dt` from `t_start` to
/// `t_end`, calling `apply_interventions_at` at each step's wall
/// time. Returns the number of times the intervention fired
/// (= total `S → V` transfers, since each fire transfers 1 unit
/// and S starts at `n_fires_max`).
fn simulate_and_count_fires(
    model: &CompiledModel,
    runtime_dt: f64,
    t_start: f64,
    t_end: f64,
    initial_s: i64,
) -> i64 {
    let fire_steps = model.resolve_fire_steps(runtime_dt, &[]);
    let mut int_s = IntState::from_vec(vec![initial_s, 0]);
    let mut real_s = RealState::new(0);

    let mut t = t_start;
    while t <= t_end + 1e-9 {
        apply_interventions_at(
            t, model, &fire_steps, runtime_dt, runtime_dt,
            &mut int_s, &mut real_s, &[], 1e-10,
        ).unwrap();
        t += runtime_dt;
    }

    int_s.counts[1]  // V count = number of fires
}

#[test]
fn periodic_intervention_fire_count_dt_invariant_at_compile_dt_1() {
    // Cohort-style: at_day 258, every 365.25 days, simulate 5 years.
    // Expect exactly 5 fires regardless of runtime dt.
    // Compile-time dt = 1.0 (the He2010 model's declared dt).
    let model = CompiledModel::new(periodic_xfer_model(
        258.0, 365.25, 365.25 * 5.0, Some(1.0),
    )).unwrap();

    for dt in [1.0, 0.5, 0.25, 0.125] {
        let fires = simulate_and_count_fires(&model, dt, 0.0, 365.25 * 5.0, 100);
        assert_eq!(fires, 5,
            "runtime_dt={}: expected exactly 5 cohort fires over 5 \
             years; got {}. This is the gh#53 regression test — \
             pre-fix, runtime_dt < compile_dt fired multiple times \
             per year due to fire_steps being baked at compile dt.",
            dt, fires);
    }
}

#[test]
fn periodic_intervention_fire_count_dt_invariant_with_compile_dt_unset() {
    // Compile-time dt unset (defaults to 1.0). Same expected count
    // at every runtime dt.
    let model = CompiledModel::new(periodic_xfer_model(
        258.0, 365.25, 365.25 * 5.0, None,
    )).unwrap();

    for dt in [1.0, 0.5, 0.25, 0.125] {
        let fires = simulate_and_count_fires(&model, dt, 0.0, 365.25 * 5.0, 100);
        assert_eq!(fires, 5,
            "runtime_dt={}: expected 5 fires; got {}", dt, fires);
    }
}

#[test]
fn periodic_intervention_fire_count_dt_invariant_with_compile_dt_smaller() {
    // The opposite mismatch: model declared at dt = 0.5, runtime
    // walks at dt = 1.0. Pre-fix this would also misfire (the bug
    // is bidirectional: fire_steps baked at compile dt 0.5, runtime
    // walks at dt 1.0 → fires at half the expected wall times).
    let model = CompiledModel::new(periodic_xfer_model(
        258.0, 365.25, 365.25 * 5.0, Some(0.5),
    )).unwrap();

    for dt in [1.0, 0.5, 0.25] {
        let fires = simulate_and_count_fires(&model, dt, 0.0, 365.25 * 5.0, 100);
        assert_eq!(fires, 5,
            "runtime_dt={} (compile dt=0.5): expected 5 fires; got {}",
            dt, fires);
    }
}

#[test]
fn at_times_intervention_fire_count_dt_invariant() {
    // Schedule at exact wall times, not periodic. Same invariant:
    // count of fires = count of times in the schedule, regardless
    // of runtime dt.
    let intervention = Intervention {
        name: "at_times_xfer".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
        kind: ir::intervention::InterventionKind::Scenario,
        actions: vec![
            Action::AbsoluteTransfer(AbsoluteTransfer {
                src: "S".into(),
                dst: "V".into(),
                count: Expr::Const(ConstExpr { value: 1.0 }),
            }),
        ],
    };
    let mut model_def = periodic_xfer_model(0.0, 1.0, 100.0, Some(1.0));
    model_def.interventions = vec![intervention];
    let model = CompiledModel::new(model_def).unwrap();

    for dt in [1.0, 0.5, 0.25, 0.125] {
        let fires = simulate_and_count_fires(&model, dt, 0.0, 100.0, 100);
        assert_eq!(fires, 5,
            "runtime_dt={}: expected 5 at-time fires; got {}", dt, fires);
    }
}

#[test]
fn periodic_intervention_fires_at_absolute_wall_time_independent_of_t_start() {
    // gh#53 follow-up audit: the cohort schedule `at_day 258, every
    // 365.25 days` is in absolute calendar-day terms. Whether the
    // simulator starts at t_start = 0 or t_start = 100, the
    // intervention should fire at the SAME absolute wall times — the
    // schedule is independent of t_start.
    //
    // This catches a cousin class of the gh#53 bug: any place where
    // the runtime confuses absolute time (used for fire_steps lookup)
    // with sim-relative time (advanced by the integrator from
    // t_start). Pre-fix, fire_steps was baked at compile dt and
    // checked against runtime step counter; post-fix, both sides use
    // absolute step indices, so a t_start shift should be a no-op on
    // intervention timing.

    let model_at_zero = CompiledModel::new(periodic_xfer_model(
        258.0, 365.25, 365.25, Some(1.0),
    )).unwrap();

    // Walk from t_start=0 to t=300 at dt=0.5. Cohort fires at t=258
    // → V counter should be 1 by t=300.
    let fire_steps = model_at_zero.resolve_fire_steps(0.5, &[]);
    let mut int_s = IntState::from_vec(vec![100, 0]);
    let mut real_s = RealState::new(0);
    let mut t = 0.0;
    while t <= 300.0 + 1e-9 {
        apply_interventions_at(t, &model_at_zero, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 1,
        "t_start=0 walk to t=300 should fire cohort once at t=258");

    // Now walk from t_start=100 to t=300 at dt=0.5. Cohort still
    // fires at absolute wall time 258 → V should still be 1.
    let mut int_s = IntState::from_vec(vec![100, 0]);
    let mut real_s = RealState::new(0);
    let mut t = 100.0;
    while t <= 300.0 + 1e-9 {
        apply_interventions_at(t, &model_at_zero, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 1,
        "t_start=100 walk to t=300 should still fire cohort once at \
         t=258 (absolute wall time, independent of t_start)");

    // And walk from t_start=200 to t=300 — wall time 258 is in
    // range, cohort should still fire.
    let mut int_s = IntState::from_vec(vec![100, 0]);
    let mut real_s = RealState::new(0);
    let mut t = 200.0;
    while t <= 300.0 + 1e-9 {
        apply_interventions_at(t, &model_at_zero, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 1,
        "t_start=200 walk to t=300 should fire cohort once at \
         t=258 (in-range)");

    // Finally walk from t_start=300 to t=400 — wall time 258 is in
    // the PAST, cohort should NOT fire (no future fires in this
    // single-period model).
    let mut int_s = IntState::from_vec(vec![100, 0]);
    let mut real_s = RealState::new(0);
    let mut t = 300.0;
    while t <= 400.0 + 1e-9 {
        apply_interventions_at(t, &model_at_zero, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 0,
        "t_start=300 walk to t=400 should fire cohort 0 times — its \
         wall time 258 is in the past relative to sim window");
}

#[test]
fn periodic_intervention_period_sweep_dt_invariant() {
    // gh#53 follow-up audit: vary the intervention period across a
    // wide range and confirm fire counts are exact + dt-invariant.
    // The cohort fix made fire-step resolution use the runtime dt;
    // this test sweeps the period to catch any period-dependent
    // rounding or alignment bug at the extremes (very fast / very
    // slow periodic schedules).
    // (period, at_day, sim_end, expected_fires) — sim_end is set
    // ≥ 1 dt past the last expected fire so end-of-sim rounding
    // can't drop the last fire on the away-from-zero half-step.
    // The recurring schedule produces fires at at_day, at_day+period,
    // at_day+2*period, ... while ≤ end. Counting carefully so the
    // expected value is exact:
    for &(period_days, at_day, sim_end, expected_fires) in &[
        (7.0,    1.0,  30.0,            5),  // 1, 8, 15, 22, 29 → 5
        (30.0,   5.0,  100.0,           4),  // 5, 35, 65, 95 → 4
        (365.25, 100.0, 365.25 * 5.0,   5),  // 100, 465.25, 830.5, 1195.75, 1561 → 5
        (1.0,    0.5,  10.0,            10), // 0.5, 1.5, …, 9.5 → 10
    ] {
        let model = CompiledModel::new(periodic_xfer_model(
            at_day, period_days, sim_end, Some(1.0),
        )).unwrap();
        for &dt in &[1.0, 0.5, 0.25] {
            let fires = simulate_and_count_fires(&model, dt, 0.0, sim_end, 1000);
            assert_eq!(fires, expected_fires,
                "period={} days, at_day={}, dt={}: expected {} fires; got {}",
                period_days, at_day, dt, expected_fires, fires);
        }
    }
}

#[test]
fn periodic_intervention_fires_at_correct_wall_time_under_sub_day_dt() {
    // The original gh#53 wall-time signature: at_day 258 must fire
    // when wall time is near 258 days, NOT near 129 days (the
    // pre-fix bug, where step 258 at runtime dt 0.5 = wall 129).
    let model = CompiledModel::new(periodic_xfer_model(
        258.0, 365.25, 365.25, Some(1.0),
    )).unwrap();

    // Drive to t = 130.0 at dt = 0.5 — pre-fix, the intervention
    // would have fired at wall time 129 (step 258 × dt 0.5), so by
    // t = 130 the V count would be 1. Post-fix it must be 0.
    let fire_steps = model.resolve_fire_steps(0.5, &[]);
    let mut int_s = IntState::from_vec(vec![100, 0]);
    let mut real_s = RealState::new(0);
    let mut t = 0.0;
    while t <= 130.0 + 1e-9 {
        apply_interventions_at(t, &model, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 0,
        "intervention should not have fired by t=130 (target wall \
         time 258); pre-fix it fired at wall 129 due to fire_steps \
         baked at compile dt 1.0 being misinterpreted as runtime \
         dt 0.5 step indices.");

    // Continue past wall time 258. After t = 258 it must have fired.
    while t <= 260.0 + 1e-9 {
        apply_interventions_at(t, &model, &fire_steps, 0.5, 0.5,
            &mut int_s, &mut real_s, &[], 1e-10).unwrap();
        t += 0.5;
    }
    assert_eq!(int_s.counts[1], 1,
        "intervention should have fired exactly once by t=260");
}
