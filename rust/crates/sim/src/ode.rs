use crate::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    error::SimError,
    intervention::all_intervention_times,
    output::output_times as get_output_times,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    schedule::{Cursor, Schedule, StepPolicy},
    simulate::Simulate,
    state::{FlowVec, IntState, RealState, Snapshot, Trajectory},
};

pub struct OdeSim;

impl Simulate for OdeSim {
    fn run(
        &self,
        model: &CompiledModel,
        params: &[f64],
        _seed: u64,
        config: &SimConfig,
    ) -> Result<Trajectory, SimError> {
        let cfg = match config {
            SimConfig::Ode(c) => c,
            _ => return Err(SimError::ConfigMismatch {
                expected: "Ode",
                got: config.variant_name(),
            }),
        };
        run_ode(model, params, cfg, None)
    }

    fn capabilities(&self) -> crate::Capabilities {
        // RUNTIME_DT: the RK4 flow accumulation evaluates a `dt`-referencing
        // rate at the REALIZED substep length (`dt_actual`), see
        // `ode_dt_rate_flow.rs`. gh#54.
        crate::Capabilities::REAL_COMPARTMENTS | crate::Capabilities::RUNTIME_DT
    }

    fn name(&self) -> &'static str { "ode" }
}

/// Evaluate ODE derivatives at the current (int_vals, real_vals) state.
///
/// RM8 in 2026-04-19 engine review: integer compartments are read at
/// their full f64 value during substeps via
/// `EvalCtx::int_float_override`. Previously this rounded to i64 at
/// every substep, quantizing state and producing O(1/N) relative error
/// that caused premature extinction at small N. Rounding now happens
/// only when snapshotting to the output trajectory.
fn ode_derivs(
    model: &CompiledModel,
    int_vals: &[f64],
    real_vals: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    d_int: &mut [f64],
    d_real: &mut [f64],
) -> Result<(), SimError> {
    // Placeholder i64 int_s — never read because int_float_override overrides.
    let int_s = IntState::from_vec(vec![0_i64; int_vals.len()]);
    let real_s = RealState::from_vec(real_vals.to_vec());

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt,
        projected: None,
        aux: None,
        int_float_override: Some(int_vals),
    };

    // Integer compartment derivatives from transition stoichiometry × rate.
    let n_tr = model.model.transitions.len();
    let mut propensities = Vec::with_capacity(n_tr);
    for i in 0..n_tr {
        propensities.push(eval_resolved(&model.resolved.rates[i], &ctx));
    }

    for v in d_int.iter_mut() { *v = 0.0; }
    for (tr_idx, stoich) in model.transition_stoich.iter().enumerate() {
        let rate = propensities[tr_idx];
        for &(local, delta) in stoich {
            d_int[local] += delta as f64 * rate;
        }
    }

    // Real compartment derivatives from explicit ODE equations.
    for v in d_real.iter_mut() { *v = 0.0; }
    for (eq_idx, _eq) in model.model.ode_equations.iter().enumerate() {
        let local = model.ode_real_indices[eq_idx];
        d_real[local] = eval_resolved(&model.resolved.ode_derivatives[eq_idx], &ctx);
    }

    Ok(())
}

/// Single RK4 step over the combined (int_vals, real_vals) state.
fn rk4_step(
    model: &CompiledModel,
    int_vals: &mut Vec<f64>,
    real_vals: &mut Vec<f64>,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Result<(), SimError> {
    let ni = int_vals.len();
    let nr = real_vals.len();

    let mut di = vec![0.0f64; ni];
    let mut dr = vec![0.0f64; nr];

    // k1
    ode_derivs(model, int_vals, real_vals, params, t, dt, &mut di, &mut dr)?;
    let k1i: Vec<f64> = di.clone();
    let k1r: Vec<f64> = dr.clone();

    // k2
    let s2i: Vec<f64> = int_vals.iter().zip(&k1i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s2r: Vec<f64> = real_vals.iter().zip(&k1r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s2i, &s2r, params, t + 0.5 * dt, dt, &mut di, &mut dr)?;
    let k2i: Vec<f64> = di.clone();
    let k2r: Vec<f64> = dr.clone();

    // k3
    let s3i: Vec<f64> = int_vals.iter().zip(&k2i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s3r: Vec<f64> = real_vals.iter().zip(&k2r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s3i, &s3r, params, t + 0.5 * dt, dt, &mut di, &mut dr)?;
    let k3i: Vec<f64> = di.clone();
    let k3r: Vec<f64> = dr.clone();

    // k4
    let s4i: Vec<f64> = int_vals.iter().zip(&k3i).map(|(x, k)| x + dt * k).collect();
    let s4r: Vec<f64> = real_vals.iter().zip(&k3r).map(|(x, k)| x + dt * k).collect();
    ode_derivs(model, &s4i, &s4r, params, t + dt, dt, &mut di, &mut dr)?;
    let k4i = &di;
    let k4r = &dr;

    // Combine
    for i in 0..ni {
        int_vals[i] += dt / 6.0 * (k1i[i] + 2.0 * k2i[i] + 2.0 * k3i[i] + k4i[i]);
        int_vals[i] = int_vals[i].max(0.0);
    }
    for i in 0..nr {
        real_vals[i] += dt / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
        real_vals[i] = real_vals[i].max(0.0);
    }

    Ok(())
}

/// Convert (int_vals, real_vals) floats to the (IntState, RealState) used by
/// the intervention machinery and output snapshots.
fn to_states(int_vals: &[f64], real_vals: &[f64]) -> (IntState, RealState) {
    let int_s = IntState::from_vec(int_vals.iter().map(|&x| x.max(0.0).round() as i64).collect());
    let real_s = RealState::from_vec(real_vals.to_vec());
    (int_s, real_s)
}

/// Deterministic ODE integration.
///
/// `tick` is an optional per-timestep progress callback, called once at the
/// top of each step with the current time `t`. ODE has no RNG, so the tick
/// trivially cannot perturb the trajectory; `None` and `Some(..)` produce
/// byte-identical output (asserted in tests/progress_tick_invariance.rs).
pub fn run_ode(
    model: &CompiledModel,
    params: &[f64],
    cfg: &OdeConfig,
    mut tick: Option<&mut dyn FnMut(f64)>,
) -> Result<Trajectory, SimError> {
    // gh#126: reject a non-finite/non-positive dt or a non-finite fire
    // time at the entry point — a RELEASE-build check (the per-conversion
    // guards in `time.rs` are debug_assert only). A non-positive dt would
    // otherwise spin the RK4 substep loop forever (time never advances).
    model.validate_schedule(cfg.dt, params)?;

    let (int_s0, real_s0) = model.initial_state(params)?;
    let mut int_vals: Vec<f64> = int_s0.counts.iter().map(|&c| c as f64).collect();
    let mut real_vals: Vec<f64> = real_s0.values.clone();

    let n_transitions = model.model.transitions.len();
    // Merged timeline spine. ODE is dt-independent, so EXACT and snap coincide;
    // it uses the EXACT policy (land on each output/effect boundary). Firing stays
    // inline; the schedule owns the sorted times and `cursor` walks them.
    let schedule = Schedule::new(
        cfg.dt,
        cfg.t_end,
        cfg.dt,
        StepPolicy::Exact,
        get_output_times(&model.model.output.times),
        all_intervention_times(model, params),
    );
    let mut cursor = Cursor::default();

    // gh#53: fire_steps depend on the runtime cfg.dt, not the
    // compile-time model.simulation.dt. See chain_binomial.rs for the
    // architectural rationale. gh#69: also threads `params` for
    // parametric `at [...]` schedules.
    let fire_steps = model.resolve_fire_steps(cfg.dt, params);

    let mut traj = Trajectory::new();
    // Accumulated continuous flows (rate × dt); rounded to u64 at each snapshot.
    let mut flow_acc: Vec<f64> = vec![0.0; n_transitions];
    let mut t = cfg.t_start;

    // Record initial snapshot
    let snapshot_flows = |flow_acc: &[f64]| {
        FlowVec::from_vec(flow_acc.iter().map(|&x| x.round() as u64).collect())
    };

    if schedule.output_due_at(&cursor, t) {
        let (is, rs) = to_states(&int_vals, &real_vals);
        traj.push(Snapshot {
            t,
            int_state: is,
            real_state: rs,
            flows: snapshot_flows(&flow_acc),
        });
        for v in flow_acc.iter_mut() { *v = 0.0; }
        cursor.pass_output();
    }

    while t < cfg.t_end {
        // Progress tick: report current time before this step. RNG-free (ODE
        // has no RNG at all).
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // The schedule is the single source of truth for the step size,
        // dt.min(next_boundary - t) — the original formula, bit-exact.
        let dt = schedule.substep(&cursor, t).expect("t < t_end inside loop");

        if dt <= 1e-15 {
            // At a boundary — apply intervention or record output
            if schedule.effect_time(&cursor).is_some_and(|iv| (iv - t).abs() < 1e-10) {
                // Continuous lifecycle: events (frozen snapshot) fire before
                // interventions (sequential, post-event). Applied EXACTLY to the
                // f64 vectors — no `to_states` round-trip — so the fractional
                // integrator state survives the boundary (the de-quantization the
                // ODE backend exists to provide). The due batch is derived once
                // at the boundary `t` (grid = cfg.dt).
                let mut batch = crate::schedule::EffectBatch::default();
                crate::effects::due_effects(model, &fire_steps, t, cfg.dt, &mut batch);
                crate::effects::apply_boundary_batch_continuous(
                    model, &batch, &mut int_vals, &mut real_vals, params, t, cfg.dt,
                )?;
                while schedule.effect_due_at(&cursor, t) { cursor.pass_effect(); }
            }
            while schedule.output_due_at(&cursor, t) {
                let ot = schedule.output_time(&cursor).expect("due implies present");
                let (is, rs) = to_states(&int_vals, &real_vals);
                traj.push(Snapshot {
                    t: ot,
                    int_state: is,
                    real_state: rs,
                    flows: snapshot_flows(&flow_acc),
                });
                for v in flow_acc.iter_mut() { *v = 0.0; }
                cursor.pass_output();
            }
            if t >= cfg.t_end { break; }
            continue;
        }

        // Accumulate flows before the step (propensities × dt approximation)
        {
            let (is, rs) = to_states(&int_vals, &real_vals);
            let mut propensities = Vec::with_capacity(n_transitions);
            // gh#126 §#11: evaluate at the REALIZED substep `dt` (dt_actual),
            // NOT the nominal grid `cfg.dt` — a rate referencing `Expr::Dt`
            // (gh#54) must see the clipped length on truncated boundary
            // substeps, matching the RK4 derivs (`:271`) and the StepClock rule
            // (`EvalCtx.dt = dt_actual`, scheduling-spine-v2 §A). Overloading it
            // with cfg.dt mis-scaled reported flows → incidence → likelihood.
            eval_propensities(model, &is, &rs, params, t, dt, &mut propensities)?;
            for (i, &p) in propensities.iter().enumerate() {
                flow_acc[i] += p * dt;
            }
        }

        rk4_step(model, &mut int_vals, &mut real_vals, params, t, dt)?;
        t += dt;

        // Apply intervention if now at that time. Canonical lifecycle: events
        // (reading the start-of-step snapshot `is`/`rs`, pre-intervention) fire
        // BEFORE interventions, which read the post-event state. Matches
        // chain_binomial.
        if schedule.effect_time(&cursor).is_some_and(|iv| (iv - t).abs() < 1e-10) {
            // Continuous lifecycle (events then interventions), applied EXACTLY
            // to the f64 vectors so the fractional integrator state survives. The
            // due batch is derived once at the boundary `t` (grid = cfg.dt).
            let mut batch = crate::schedule::EffectBatch::default();
            crate::effects::due_effects(model, &fire_steps, t, cfg.dt, &mut batch);
            crate::effects::apply_boundary_batch_continuous(
                model, &batch, &mut int_vals, &mut real_vals, params, t, cfg.dt,
            )?;
            while schedule.effect_due_at(&cursor, t) { cursor.pass_effect(); }
        }

        // Record outputs
        schedule.drain_outputs(&mut cursor, t, |ot| {
            let (is, rs) = to_states(&int_vals, &real_vals);
            traj.push(Snapshot {
                t: ot,
                int_state: is,
                real_state: rs,
                flows: snapshot_flows(&flow_acc),
            });
            for v in flow_acc.iter_mut() { *v = 0.0; }
        });
    }

    // Flush any remaining output times
    schedule.drain_outputs(&mut cursor, f64::INFINITY, |ot| {
        let (is, rs) = to_states(&int_vals, &real_vals);
        traj.push(Snapshot {
            t: ot,
            int_state: is,
            real_state: rs,
            flows: snapshot_flows(&flow_acc),
        });
        for v in flow_acc.iter_mut() { *v = 0.0; }
    });

    Ok(traj)
}
