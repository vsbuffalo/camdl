use crate::{
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    error::SimError,
    intervention::all_intervention_times,
    output::output_times as get_output_times,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    schedule::{Cursor, Schedule, StepPolicy, EFFECT_EPS, MIN_STEP_EPS},
    simulate::Simulate,
    state::{Flows, IntState, RealState, Snapshot, Trajectory},
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
/// `d_flow[i]` receives the augmented-flow derivative `dc_i/dt = propensity_i`
/// (one per transition) — the SAME per-transition propensities computed for the
/// compartment derivatives, so carrying flows through the RK4 stages costs no
/// extra rate evaluations (B1: the standalone Euler flow eval is dropped, 5→4
/// evals/step). The propensities are evaluated at the UNROUNDED float state
/// (`int_float_override`), unlike the old Euler flow which read the rounded
/// integer state — so the augmented incidence is both higher-order AND
/// evaluated at the same state the integrator sees.
fn ode_derivs(
    model: &CompiledModel,
    int_vals: &[f64],
    real_vals: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    d_int: &mut [f64],
    d_real: &mut [f64],
    d_flow: &mut [f64],
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

    // Activate the per-state binding cache for this stage: each model binding
    // (N_p, I_agg_p, spatial FOI, …) is evaluated at most once across all rates
    // and ODE equations at THIS (state, t), instead of on every `BindingRef`.
    // Byte-identical to the uncached path (value memoization — gate_binding_cache_ab),
    // and the lever that makes coupled ODE models (cVDPV2) fast. Restored here
    // because B1 dropped the standalone `eval_propensities` call that used to be
    // the only cache-entering ODE eval; the RK4 stages now own it. Dropped at the
    // end of this stage so the next stage (different state) recomputes.
    let _cache = crate::resolved_expr::CacheScope::enter(model.resolved.bindings.len());

    // Integer compartment derivatives from transition stoichiometry × rate.
    let n_tr = model.model.transitions.len();
    let mut propensities = Vec::with_capacity(n_tr);
    for i in 0..n_tr {
        propensities.push(eval_resolved(&model.resolved.rates[i], &ctx));
    }

    // Augmented flow derivative: dc_i/dt = propensity_i (per transition). Reuses
    // the propensities just computed — no additional rate evaluation.
    d_flow.copy_from_slice(&propensities);

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

/// Single RK4 step over the combined (int_vals, real_vals) state, optionally
/// carrying the augmented flow.
///
/// When `flow` is `Some`, the per-transition cumulative flow integrals ride
/// along as augmented state: `dc_i/dt = propensity_i` integrated by the SAME RK4
/// stages as the compartments (Q1B). Because `dc/dt` depends only on (int, real,
/// t) — never on `c` itself — the flow needs no stage-state perturbation; its
/// stage slopes are exactly the `d_flow` each `ode_derivs` already returns.
/// Flow is NOT clamped (a monotone non-decreasing accumulator; propensities ≥ 0).
///
/// When `flow` is `None` the compartment integration is identical and flow is
/// left to the caller (the `Expr::Dt` / RUNTIME_DT Euler path).
fn rk4_step(
    model: &CompiledModel,
    int_vals: &mut Vec<f64>,
    real_vals: &mut Vec<f64>,
    flow: Option<&mut Vec<f64>>,
    params: &[f64],
    t: f64,
    dt: f64,
) -> Result<(), SimError> {
    let ni = int_vals.len();
    let nr = real_vals.len();
    let nf = model.model.transitions.len();

    let mut di = vec![0.0f64; ni];
    let mut dr = vec![0.0f64; nr];
    let mut df = vec![0.0f64; nf];

    // k1
    ode_derivs(model, int_vals, real_vals, params, t, dt, &mut di, &mut dr, &mut df)?;
    let k1i: Vec<f64> = di.clone();
    let k1r: Vec<f64> = dr.clone();
    let k1f: Vec<f64> = df.clone();

    // k2
    let s2i: Vec<f64> = int_vals.iter().zip(&k1i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s2r: Vec<f64> = real_vals.iter().zip(&k1r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s2i, &s2r, params, t + 0.5 * dt, dt, &mut di, &mut dr, &mut df)?;
    let k2i: Vec<f64> = di.clone();
    let k2r: Vec<f64> = dr.clone();
    let k2f: Vec<f64> = df.clone();

    // k3
    let s3i: Vec<f64> = int_vals.iter().zip(&k2i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s3r: Vec<f64> = real_vals.iter().zip(&k2r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s3i, &s3r, params, t + 0.5 * dt, dt, &mut di, &mut dr, &mut df)?;
    let k3i: Vec<f64> = di.clone();
    let k3r: Vec<f64> = dr.clone();
    let k3f: Vec<f64> = df.clone();

    // k4
    let s4i: Vec<f64> = int_vals.iter().zip(&k3i).map(|(x, k)| x + dt * k).collect();
    let s4r: Vec<f64> = real_vals.iter().zip(&k3r).map(|(x, k)| x + dt * k).collect();
    ode_derivs(model, &s4i, &s4r, params, t + dt, dt, &mut di, &mut dr, &mut df)?;
    let k4i = &di;
    let k4r = &dr;
    let k4f = &df;

    // Combine compartments (clamped ≥ 0).
    for i in 0..ni {
        int_vals[i] += dt / 6.0 * (k1i[i] + 2.0 * k2i[i] + 2.0 * k3i[i] + k4i[i]);
        int_vals[i] = int_vals[i].max(0.0);
    }
    for i in 0..nr {
        real_vals[i] += dt / 6.0 * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i]);
        real_vals[i] = real_vals[i].max(0.0);
    }

    // Combine augmented flow (NOT clamped).
    if let Some(flow) = flow {
        for i in 0..nf {
            flow[i] += dt / 6.0 * (k1f[i] + 2.0 * k2f[i] + 2.0 * k3f[i] + k4f[i]);
        }
    }

    Ok(())
}

/// One integrated ODE state: the integer- and real-compartment values plus the
/// cumulative per-transition flow integrals. `flow` is `∫ rate dt` accumulated
/// since the last output reset (reset to 0 at each output boundary so
/// `snapshot.flows` stays per-interval incidence — pomp's accumulator-variable
/// semantics, King et al. 2016 JSS). Compartments are clamped `≥ 0`; `flow` is
/// not (a monotone non-decreasing accumulator within an interval).
struct OdeState {
    int:  Vec<f64>,
    real: Vec<f64>,
    flow: Vec<f64>,
}

/// Advance the integrated state across ONE `[t, t + h_max]` boundary interval.
/// `h_max` is the RAW distance to the next output / intervention / `t_end`
/// boundary (from [`Schedule::next_boundary`]); the stepper MUST NOT cross it.
/// Returns the step actually taken: `Rk4Fixed` takes `min(dt, h_max)` and the
/// driver re-enters until the boundary is reached; an adaptive stepper (Phase C
/// `Dopri5`) takes `≤ h_max` and is likewise re-entered. This seam is the single
/// place "support both integrators" lives — `run_ode` stays integrator-agnostic.
trait OdeStepper {
    fn advance(
        &mut self,
        model: &CompiledModel,
        params: &[f64],
        t: f64,
        h_max: f64,
        state: &mut OdeState,
    ) -> Result<f64, SimError>;
}

/// Fixed-step classic RK4 — today's integrator, the default and the golden
/// reference. Carries `dt` (the nominal step) as the analogue of an adaptive
/// stepper's carried step guess; one `advance` takes `min(dt, h_max)`. (The
/// proposal sketches a unit struct; carrying `dt` makes the re-entry contract
/// self-contained and parallels `Dopri5`'s carried `h`.)
///
/// `euler_flow` selects the flow-accounting scheme (B2): when the model
/// references the step size in a rate (`Expr::Dt` / RUNTIME_DT) the augmented
/// flow has no single `dt` to thread through the stages, so those models keep
/// the first-order Euler flow; every other model gets augmented (RK4) flow.
struct Rk4Fixed {
    dt: f64,
    euler_flow: bool,
}

impl OdeStepper for Rk4Fixed {
    fn advance(
        &mut self,
        model: &CompiledModel,
        params: &[f64],
        t: f64,
        h_max: f64,
        state: &mut OdeState,
    ) -> Result<f64, SimError> {
        // Clip the nominal step to land on the boundary: `dt.min(h_max)`,
        // bit-identical to the old `Schedule::substep` (= `dt.min(boundary - t)`).
        let h = self.dt.min(h_max);

        if self.euler_flow {
            // RUNTIME_DT models (B2): keep the O(h) Euler flow (`c += rate(t)·h`,
            // a left-rectangle rule) at the REALIZED substep `h` — augmented flow
            // is undefined when the rate depends on the step size. gh#126 §#11:
            // the `Expr::Dt` rate (gh#54) sees `h` (dt_actual), matching the RK4
            // derivs and the StepClock rule. Evaluated at the start-of-step
            // ROUNDED integer state (`int_float_override: None`), as it was before
            // this change. The user is warned once at load (see the CLI).
            let (is, rs) = to_states(&state.int, &state.real);
            let mut propensities = Vec::with_capacity(state.flow.len());
            eval_propensities(model, &is, &rs, params, t, h, &mut propensities)?;
            for (i, &p) in propensities.iter().enumerate() {
                state.flow[i] += p * h;
            }
            rk4_step(model, &mut state.int, &mut state.real, None, params, t, h)?;
        } else {
            // Augmented flow (Q1B): `dc_i/dt = propensity_i` carried through the
            // SAME RK4 stages as the compartments — one mechanism, integrator-
            // order incidence, and no standalone 5th propensity eval.
            rk4_step(model, &mut state.int, &mut state.real, Some(&mut state.flow), params, t, h)?;
        }
        Ok(h)
    }
}

// ── Adaptive Dormand–Prince RK4(5) — gh#166 Phase C ────────────────────────────
//
// Canonical DOPRI5: 7-stage explicit RK with an embedded 4th-order solution for
// error estimation, and a PI step-size controller. Tableau transcribed from
// Dormand & Prince (1980), J. Comp. Appl. Math. 6(1):19–26 / Hairer, Nørsett &
// Wanner (1993) "Solving ODEs I", Table 5.2 — consistency-verified (every a-row
// sums to its c; b and b̂ each sum to 1; FSAL a[6]=b[..6]). The error is taken as
// y5−y4 directly (the b−b̂ weights are derived, not hand-transcribed). Flows ride
// as augmented state through the same stages (Phase B), 5th-order, not
// error-controlled (a quadrature whose accuracy follows the state step).

/// Stage abscissae c_i.
const DP_C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
/// Lower-triangular stage coefficients a[i][j] (j < i); unused entries are 0.
const DP_A: [[f64; 6]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
];
/// 5th-order solution weights b_i (the accepted step).
const DP_B: [f64; 7] =
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0];
/// Embedded 4th-order weights b̂_i (for the error estimate y5−y4).
const DP_BH: [f64; 7] = [
    5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
    -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0,
];

// PI step-size controller (Gustafsson) — standard DOPRI5 defaults (Hairer-Nørsett-
// Wanner §II.4; confirmed for this implementation). These affect EFFICIENCY (the
// step sequence), not correctness: the error control bounds accuracy regardless.
const DP_SAFETY: f64 = 0.9;
const DP_ALPHA: f64 = 0.7 / 5.0; // err exponent (k = embedded order + 1 = 5)
const DP_BETA: f64 = 0.4 / 5.0;  // PI memory exponent on the previous error
const DP_FACMIN: f64 = 0.2;      // shrink no more than 5× per step
const DP_FACMAX: f64 = 5.0;      // grow   no more than 5× per step
const DP_MAX_REJECTIONS: u32 = 10;

/// Default adaptive tolerances when the model/CLI specify none. These are the
/// C8-calibrated values: the `rk45_tolerance_calibration` sweep found every
/// candidate matched fine-`dt` RK4 to sub-nat loglik (by 4–9 orders of
/// magnitude) on the SIR/SEIR/TB validation models, so the proposal's example
/// values were kept rather than over-tuned to one model. Rationale + ecosystem
/// comparison: `docs/dev/notes/2026-06-16-deterministic-ode-integration.md`.
pub const DEFAULT_ATOL: f64 = 1e-8;
pub const DEFAULT_RTOL: f64 = 1e-6;

/// Adaptive Dormand–Prince RK4(5) integrator. `h` is the controller's carried
/// next-step guess (clipped to `h_max` each `advance`); the driver re-enters
/// until the boundary is reached. Opt-in (`integrator = "rk45"`); capability-
/// gated out of `Expr::Dt` / RUNTIME_DT models (no single fixed step).
struct Dopri5 {
    atol: f64,
    rtol: f64,
    h: f64,
    err_prev: f64,
    h_min: f64,
}

impl Dopri5 {
    fn new(atol: f64, rtol: f64, cfg: &OdeConfig) -> Self {
        let span = (cfg.t_end - cfg.t_start).abs().max(1.0);
        Dopri5 {
            atol,
            rtol,
            // Initial step guess = the nominal dt (a good seed; the controller
            // adapts immediately, rejecting+shrinking if it overshoots tolerance).
            h: if cfg.dt.is_finite() && cfg.dt > 0.0 { cfg.dt } else { span },
            err_prev: 1.0,
            h_min: 1e-10 * span,
        }
    }
}

/// One trial DOPRI5 step of size `h` from `state` at `t`. Returns the proposed
/// (int, real) at `t+h` (5th-order, clamped ≥0), the per-transition flow
/// increment over `[t, t+h]` (augmented, 5th-order), and the scaled error norm
/// (RMS of (y5−y4)/scale over int+real). Does not mutate `state`.
fn dopri5_try_step(
    model: &CompiledModel,
    params: &[f64],
    t: f64,
    h: f64,
    state: &OdeState,
    atol: f64,
    rtol: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64), SimError> {
    let (ni, nr, nf) = (state.int.len(), state.real.len(), state.flow.len());
    let mut ki: Vec<Vec<f64>> = Vec::with_capacity(7);
    let mut kr: Vec<Vec<f64>> = Vec::with_capacity(7);
    let mut kf: Vec<Vec<f64>> = Vec::with_capacity(7);
    let mut di = vec![0.0f64; ni];
    let mut dr = vec![0.0f64; nr];
    let mut df = vec![0.0f64; nf];

    for i in 0..7 {
        // Stage state: x + h·Σ_{j<i} a[i][j]·k_j. Stages are NOT clamped (RK
        // stages may transiently dip <0); only the accepted result is clamped,
        // matching rk4_step.
        let mut si = state.int.clone();
        let mut sr = state.real.clone();
        for j in 0..i {
            let a = DP_A[i][j];
            if a != 0.0 {
                for m in 0..ni { si[m] += h * a * ki[j][m]; }
                for m in 0..nr { sr[m] += h * a * kr[j][m]; }
            }
        }
        // dt = h passed for signature parity; rk45 rejects RUNTIME_DT models, so
        // no rate reads Expr::Dt here.
        ode_derivs(model, &si, &sr, params, t + DP_C[i] * h, h, &mut di, &mut dr, &mut df)?;
        ki.push(di.clone());
        kr.push(dr.clone());
        kf.push(df.clone());
    }

    let mut y5_int = state.int.clone();
    let mut y5_real = state.real.clone();
    let mut flow_inc = vec![0.0f64; nf];
    let mut err_sq = 0.0f64;
    let n = ni + nr;

    for m in 0..ni {
        let (mut s5, mut s4) = (0.0, 0.0);
        for i in 0..7 { s5 += DP_B[i] * ki[i][m]; s4 += DP_BH[i] * ki[i][m]; }
        let y5 = state.int[m] + h * s5;
        let y4 = state.int[m] + h * s4;
        let sc = atol + rtol * state.int[m].abs().max(y5.abs());
        err_sq += ((y5 - y4) / sc).powi(2);
        y5_int[m] = y5.max(0.0);
    }
    for m in 0..nr {
        let (mut s5, mut s4) = (0.0, 0.0);
        for i in 0..7 { s5 += DP_B[i] * kr[i][m]; s4 += DP_BH[i] * kr[i][m]; }
        let y5 = state.real[m] + h * s5;
        let y4 = state.real[m] + h * s4;
        let sc = atol + rtol * state.real[m].abs().max(y5.abs());
        err_sq += ((y5 - y4) / sc).powi(2);
        y5_real[m] = y5.max(0.0);
    }
    // Flow increment: the 5th-order quadrature of `dc_i/dt = propensity_i ≥ 0`.
    // Unlike RK4's (1,2,2,1)/6, `DP_B` has a NEGATIVE weight (b[4]=-2187/6784),
    // so `flow_inc` is not non-negative by a positive-weights argument — its
    // non-negativity/monotonicity rests on the error controller resolving the
    // (smooth, non-decreasing) integral accurately, not on a sign guarantee.
    // We deliberately do NOT clamp it: clamping would bias the incidence integral
    // (accuracy, not clamping, keeps the accumulator faithful).
    for m in 0..nf {
        let mut s5 = 0.0;
        for i in 0..7 { s5 += DP_B[i] * kf[i][m]; }
        flow_inc[m] = h * s5;
    }

    let err = if n > 0 { (err_sq / n as f64).sqrt() } else { 0.0 };
    Ok((y5_int, y5_real, flow_inc, err))
}

impl OdeStepper for Dopri5 {
    fn advance(
        &mut self,
        model: &CompiledModel,
        params: &[f64],
        t: f64,
        h_max: f64,
        state: &mut OdeState,
    ) -> Result<f64, SimError> {
        let mut h = self.h.min(h_max);
        if !(h > 0.0) { h = h_max; }
        let mut rejections = 0u32;
        loop {
            let (y5_int, y5_real, flow_inc, err) =
                dopri5_try_step(model, params, t, h, state, self.atol, self.rtol)?;
            if err <= 1.0 {
                // Accept. Commit state + augmented flow.
                state.int = y5_int;
                state.real = y5_real;
                for m in 0..state.flow.len() { state.flow[m] += flow_inc[m]; }
                // PI controller for the next step (err floored to avoid div-by-0
                // and an unbounded grow on an exact step).
                let e = err.max(1e-10);
                let fac = DP_SAFETY * e.powf(-DP_ALPHA) * self.err_prev.powf(DP_BETA);
                self.h = h * fac.clamp(DP_FACMIN, DP_FACMAX);
                self.err_prev = err.max(1e-4);
                return Ok(h);
            }
            // Reject: shrink (elementary, no PI memory on rejection).
            rejections += 1;
            if rejections > DP_MAX_REJECTIONS {
                return Err(SimError::Validation(format!(
                    "rk45: step rejected {DP_MAX_REJECTIONS}+ times at t={t} (h={h:.3e}); \
                     the model may be too stiff for the explicit DOPRI5 integrator — \
                     use integrator = \"rk4\" with a fine dt, or loosen atol/rtol"
                )));
            }
            let e = err.max(1e-10);
            h *= (DP_SAFETY * e.powf(-DP_ALPHA)).clamp(DP_FACMIN, 1.0);
            if h < self.h_min {
                return Err(SimError::Validation(format!(
                    "rk45: step-size underflow at t={t} (h={h:.3e} < h_min={:.3e}); cannot \
                     meet (atol={}, rtol={}) — loosen tolerances or use integrator = \"rk4\"",
                    self.h_min, self.atol, self.rtol
                )));
            }
        }
    }
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
    let n_transitions = model.model.transitions.len();
    let mut state = OdeState {
        int:  int_s0.counts.iter().map(|&c| c as f64).collect(),
        real: real_s0.values.clone(),
        flow: vec![0.0; n_transitions],
    };

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

    // gh#166: B2 — models that reference the step size in a rate (`Expr::Dt` /
    // RUNTIME_DT) keep the first-order Euler flow on fixed RK4; all others use
    // augmented (RK4-integrated) flow. Computed once per run.
    let euler_flow = model
        .required_capabilities()
        .contains(crate::Capabilities::RUNTIME_DT);

    // Select the integrator from the model's declared config (CLI `--integrator`
    // override is C4). The driver below is integrator-agnostic: it hands each
    // stepper the raw distance to the next boundary and re-enters until the
    // boundary is reached, so fixed RK4 and adaptive Dopri5 share one loop.
    let mut stepper: Box<dyn OdeStepper> = match &model.model.simulation.integrator {
        ir::model::Integrator::Rk4 => Box::new(Rk4Fixed { dt: cfg.dt, euler_flow }),
        ir::model::Integrator::Rk45 { atol, rtol } => {
            // C3 capability gate: a `dt`-in-rate (RUNTIME_DT) model has no single
            // fixed step — adaptive stepping is undefined. Honest hard error, never
            // a silent rk4 fallback.
            if euler_flow {
                return Err(SimError::Validation(
                    "integrator = rk45 cannot run a model that references `dt` in a \
                     rate (Expr::Dt): adaptive stepping has no fixed step size — use \
                     integrator = rk4.".to_string(),
                ));
            }
            let atol = atol.unwrap_or(DEFAULT_ATOL);
            let rtol = rtol.unwrap_or(DEFAULT_RTOL);
            Box::new(Dopri5::new(atol, rtol, cfg))
        }
    };

    let mut traj = Trajectory::new();
    // The ODE flow is genuinely real-valued; recorded as `Flows::Real` WITHOUT
    // rounding, so a sub-unit flow (a slow transition such as TB reactivation)
    // survives into the likelihood instead of quantizing to 0 → `-∞`.
    let snapshot_flows = |flow: &[f64]| Flows::Real(flow.to_vec());
    let mut t = cfg.t_start;

    // Record initial snapshot
    if schedule.output_due_at(&cursor, t) {
        let (is, rs) = to_states(&state.int, &state.real);
        traj.push(Snapshot {
            t,
            int_state: is,
            real_state: rs,
            flows: snapshot_flows(&state.flow),
        });
        for v in state.flow.iter_mut() { *v = 0.0; }
        cursor.pass_output();
    }

    while t < cfg.t_end {
        // Progress tick: report current time before this step. RNG-free (ODE
        // has no RNG at all).
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // Raw distance to the next boundary (output/effect/t_end), NOT clipped to
        // `cfg.dt`. The stepper chooses its own internal step ≤ this; fixed RK4
        // takes `min(dt, h_max)`, bit-identical to the old `schedule.substep`.
        let boundary = schedule.next_boundary(&cursor, t).expect("t < t_end inside loop");
        let h_max = boundary - t;

        if h_max <= MIN_STEP_EPS {
            // At a boundary — apply effects or record output. Same threshold as
            // the old `substep <= 1e-15`: for `cfg.dt > 1e-15`,
            // `dt.min(h_max) <= 1e-15  ⇔  h_max <= 1e-15`.
            if schedule.effect_time(&cursor).is_some_and(|iv| (iv - t).abs() < EFFECT_EPS) {
                // Continuous lifecycle: events (frozen snapshot) fire before
                // interventions (sequential, post-event). Applied EXACTLY to the
                // f64 vectors — no `to_states` round-trip — so the fractional
                // integrator state survives the boundary (the de-quantization the
                // ODE backend exists to provide). The due batch is derived once
                // at the boundary `t` (grid = cfg.dt).
                let mut batch = crate::schedule::EffectBatch::default();
                crate::effects::due_effects(model, &fire_steps, t, cfg.dt, &mut batch);
                crate::effects::apply_boundary_batch_continuous(
                    model, &batch, &mut state.int, &mut state.real, params, t, cfg.dt,
                )?;
                while schedule.effect_due_at(&cursor, t) { cursor.pass_effect(); }
            }
            while schedule.output_due_at(&cursor, t) {
                let ot = schedule.output_time(&cursor).expect("due implies present");
                let (is, rs) = to_states(&state.int, &state.real);
                traj.push(Snapshot {
                    t: ot,
                    int_state: is,
                    real_state: rs,
                    flows: snapshot_flows(&state.flow),
                });
                for v in state.flow.iter_mut() { *v = 0.0; }
                cursor.pass_output();
            }
            if t >= cfg.t_end { break; }
            continue;
        }

        // Advance one integrator step toward the boundary (fixed RK4: exactly one
        // `min(dt, h_max)` step, accumulating the Euler flow; re-entered by the
        // loop until the boundary). `t` advances by exactly the step taken.
        let h_taken = stepper.advance(model, params, t, h_max, &mut state)?;
        t += h_taken;

        // Apply effects if we just landed on a boundary. Canonical lifecycle:
        // events (reading the start-of-step snapshot, pre-intervention) fire
        // BEFORE interventions, which read the post-event state; applied EXACTLY
        // to the f64 vectors so the fractional integrator state survives. A no-op
        // on intermediate substeps (no effect time within 1e-10 of `t`).
        if schedule.effect_time(&cursor).is_some_and(|iv| (iv - t).abs() < EFFECT_EPS) {
            let mut batch = crate::schedule::EffectBatch::default();
            crate::effects::due_effects(model, &fire_steps, t, cfg.dt, &mut batch);
            crate::effects::apply_boundary_batch_continuous(
                model, &batch, &mut state.int, &mut state.real, params, t, cfg.dt,
            )?;
            while schedule.effect_due_at(&cursor, t) { cursor.pass_effect(); }
        }

        // Record outputs due at or before t (a no-op on intermediate substeps).
        schedule.drain_outputs(&mut cursor, t, |ot| {
            let (is, rs) = to_states(&state.int, &state.real);
            traj.push(Snapshot {
                t: ot,
                int_state: is,
                real_state: rs,
                flows: snapshot_flows(&state.flow),
            });
            for v in state.flow.iter_mut() { *v = 0.0; }
        });
    }

    // Flush any remaining output times
    schedule.drain_outputs(&mut cursor, f64::INFINITY, |ot| {
        let (is, rs) = to_states(&state.int, &state.real);
        traj.push(Snapshot {
            t: ot,
            int_state: is,
            real_state: rs,
            flows: snapshot_flows(&state.flow),
        });
        for v in state.flow.iter_mut() { *v = 0.0; }
    });

    Ok(traj)
}
