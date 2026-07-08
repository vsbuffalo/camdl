use crate::{
    boundary_times::{EffectTimes, OutputTimes},
    compiled_model::CompiledModel,
    config::{OdeConfig, SimConfig},
    error::SimError,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::{eval_deriv_entry, eval_emitted_grad, eval_resolved},
    schedule::{Cursor, Schedule, MIN_STEP_EPS},
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
    // gh#272: the per-eval prologue for this θ-span (computed once in `run_ode`),
    // lent into every RK-stage eval. `None` ⇒ on-demand (byte-identical).
    per_eval: Option<&[f64]>,
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
        per_eval,
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

/// The forward-sensitivity blocks carried alongside `(x, flow)` as augmented
/// state (gh#275 §1c). `d` is the number of estimated parameters.
///
/// - `state[i*d + p] = ∂x_i/∂θ_p` — the compartment sensitivity `S`, chained
///   against `∂g/∂x` for `Prevalence` (`Instant`) observation streams.
/// - `flow[r*d + p]  = ∂(cumulative flow_r)/∂θ_p` — the **raw per-transition
///   incidence-flow derivative** `acc_sens`, carrying NO stoichiometry (each
///   transition owns its own flow slot `r`), which the `Incidence` (`Interval`)
///   chain rule folds per stream. Structurally distinct from `state`'s
///   stoich-weighted `Ṡ`: reusing `Ṡ` for the accumulator computes a
///   net-compartment-change sensitivity, not an incidence-flow one — a
///   silent-wrong gradient the two-`J` split exists to prevent (§1c).
struct Sens {
    state: Vec<f64>, // n_int × d, row-major
    flow: Vec<f64>,  // n_transitions × d, row-major
    /// Whether the `∂rate/∂θ` forcing term is active (gh#396). `true` for the
    /// parameter-sensitivity paths (`S = ∂x/∂θ`, seeded by `ic_grad` or zero);
    /// `false` for the **monodromy** path (`S = ∂x/∂x₀`, identity-seeded), where
    /// the columns are compartments, not parameters, so the param-keyed forcing
    /// summand must be dropped rather than mis-indexed into `param_model_idx`.
    forcing: bool,
}

/// Forward-sensitivity derivatives, assembled **sparse and transitionwise**
/// (gh#275 §1c) rather than as a dense `n×n` matmul — at national scale `n` is in
/// the hundreds–thousands and the dense form is the bottleneck the sparse-coupling
/// work exists to avoid. For each transition `r`, its `total_dr_dθ[p]` chains the
/// param-gradient with the state-gradient through `S`, then feeds **both** blocks:
///
/// ```text
/// total_dr_dθ[p] = ∂rate_r/∂θ_p + Σ_j ∂rate_r/∂x_j · S[j,p]
/// d_state[:, p] += stoich_r · total_dr_dθ[p]     # Ṡ — state Jacobian THROUGH stoich
/// d_flow[r,  p]  = total_dr_dθ[p]                # ∂flow_r/∂θ — raw, NO stoich
/// ```
///
/// `param_model_idx[p]` maps estimated param `p` to its MODEL parameter index (the
/// param-keyed `rate_grad` lookup). `s` is `S` at the current stage state
/// (`n_int × d`); `d_state` (`n_int × d`) and `d_flow` (`n_tr × d`) receive the
/// derivatives. The eval context is byte-identical to [`ode_derivs`] (propensities
/// at the UNROUNDED float state via `int_float_override`), so `J_x`/`J_θ` are
/// evaluated at exactly the state the integrator sees. Integer compartments only
/// for now (transition-driven); real ODE-equation sensitivity is a follow-up.
fn sensitivity_derivs(
    model: &CompiledModel,
    int_vals: &[f64],
    real_vals: &[f64],
    params: &[f64],
    param_model_idx: &[usize],
    // gh#396: when `false`, the param-keyed `∂rate/∂θ` forcing summand is dropped
    // so this integrates the pure state-Jacobian chain `Ṡ = J_x·S` — the monodromy
    // variational equation. `true` restores the full `Ṡ = J_x·S + J_θ`.
    forcing: bool,
    t: f64,
    dt: f64,
    per_eval: Option<&[f64]>,
    s: &[f64],
    d_state: &mut [f64],
    d_flow: &mut [f64],
) {
    let d = param_model_idx.len();
    let int_s = IntState::from_vec(vec![0_i64; int_vals.len()]);
    let real_s = RealState::from_vec(real_vals.to_vec());
    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt,
        projected: None,
        aux: None,
        int_float_override: Some(int_vals),
        per_eval,
    };
    let _cache = crate::resolved_expr::CacheScope::enter(model.resolved.bindings.len());

    for v in d_state.iter_mut() { *v = 0.0; }
    for (tr_idx, stoich) in model.transition_stoich.iter().enumerate() {
        let rate_grad = &model.resolved.rate_grads_indexed[tr_idx];            // param-keyed ∂rate/∂θ
        let rate_state_grad = &model.resolved.rate_state_grads_indexed[tr_idx].0; // comp-keyed ∂rate/∂x
        for p in 0..d {
            // total_dr_dθ[p] = ∂rate/∂θ_p + Σ_j ∂rate/∂x_j · S[j, p]  (chain through state).
            // Monodromy mode (`forcing = false`): drop the ∂rate/∂θ_p term → pure
            // `Ṡ = J_x·S`, so S(T) = ∂x(T)/∂x₀ when seeded S(0) = I (gh#396).
            let mut total = if forcing {
                eval_emitted_grad(rate_grad, param_model_idx[p], &ctx)
            } else {
                0.0
            };
            for (j, entry) in rate_state_grad {
                total += eval_deriv_entry(entry, &ctx) * s[j * d + p];
            }
            // ∂flow_r/∂θ_p = total_dr_dθ[p] — the raw per-transition flow
            // derivative (dc_r/dt = rate_r, so d(∂c_r/∂θ)/dt = ∂rate_r/∂θ), NO
            // stoichiometry: each transition owns flow slot `tr_idx`.
            d_flow[tr_idx * d + p] = total;
            // Ṡ[:, p] += stoich_r · total_dr_dθ[p]  — the state Jacobian THROUGH
            // stoichiometry (distinct from the raw incidence-flow derivative).
            for &(local, delta) in stoich {
                d_state[local * d + p] += delta as f64 * total;
            }
        }
    }
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
    per_eval: Option<&[f64]>,
    // gh#275: optional forward-sensitivity blocks `S = ∂x/∂θ` and `acc_sens =
    // ∂flow/∂θ` ([`Sens`]), carried by the SAME four RK4 stages as the state so
    // `J_x`, `J_θ`, `S`, and `acc_sens` advance in lockstep (the augmented
    // `(x, flow, S, acc_sens)` system). `None` ⇒ the value-only path,
    // byte-identical to before (every x-line below is untouched). `param_model_idx`
    // maps each of the `d` sensitivity columns to its model parameter index.
    sens: Option<&mut Sens>,
    param_model_idx: &[usize],
) -> Result<(), SimError> {
    let ni = int_vals.len();
    let nr = real_vals.len();
    let nf = model.model.transitions.len();
    let d = param_model_idx.len();

    let mut di = vec![0.0f64; ni];
    let mut dr = vec![0.0f64; nr];
    let mut df = vec![0.0f64; nf];

    // k1
    ode_derivs(model, int_vals, real_vals, params, t, dt, per_eval, &mut di, &mut dr, &mut df)?;
    let k1i: Vec<f64> = di.clone();
    let k1r: Vec<f64> = dr.clone();
    let k1f: Vec<f64> = df.clone();

    // k2
    let s2i: Vec<f64> = int_vals.iter().zip(&k1i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s2r: Vec<f64> = real_vals.iter().zip(&k1r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s2i, &s2r, params, t + 0.5 * dt, dt, per_eval, &mut di, &mut dr, &mut df)?;
    let k2i: Vec<f64> = di.clone();
    let k2r: Vec<f64> = dr.clone();
    let k2f: Vec<f64> = df.clone();

    // k3
    let s3i: Vec<f64> = int_vals.iter().zip(&k2i).map(|(x, k)| x + 0.5 * dt * k).collect();
    let s3r: Vec<f64> = real_vals.iter().zip(&k2r).map(|(x, k)| x + 0.5 * dt * k).collect();
    ode_derivs(model, &s3i, &s3r, params, t + 0.5 * dt, dt, per_eval, &mut di, &mut dr, &mut df)?;
    let k3i: Vec<f64> = di.clone();
    let k3r: Vec<f64> = dr.clone();
    let k3f: Vec<f64> = df.clone();

    // k4
    let s4i: Vec<f64> = int_vals.iter().zip(&k3i).map(|(x, k)| x + dt * k).collect();
    let s4r: Vec<f64> = real_vals.iter().zip(&k3r).map(|(x, k)| x + dt * k).collect();
    ode_derivs(model, &s4i, &s4r, params, t + dt, dt, per_eval, &mut di, &mut dr, &mut df)?;
    let k4i = &di;
    let k4r = &dr;
    let k4f = &df;

    // Sensitivity stage slopes (gh#275): the augmented `(x, flow, S, acc_sens)`
    // system evaluated at the SAME four stage states as x — the sensitivity at
    // stage k uses the x-stage state (s{k}i, s{k}r) AND the S-stage state (S
    // advanced by the previous slope). `acc_sens` (flow sensitivity) does not feed
    // back — `d(∂flow/∂θ)/dt` depends on `S` and θ, never on `acc_sens` itself —
    // so it needs no stage perturbation, exactly like the value `flow`; its stage
    // slopes are the `d_flow` each `sensitivity_derivs` returns. Read `int_vals`
    // here, BEFORE the compartment combine mutates it.
    let stage_state = ni * d;
    let stage_flow = nf * d;
    let (ks_state, ks_flow) = if let Some(ref s) = sens {
        let mut d_state = vec![0.0f64; stage_state];
        let mut d_flow = vec![0.0f64; stage_flow];
        let mut ks_state = [const { Vec::new() }; 4];
        let mut ks_flow = [const { Vec::new() }; 4];
        let forcing = s.forcing;

        // k1 at the entry S.
        sensitivity_derivs(model, int_vals, real_vals, params, param_model_idx, forcing, t, dt, per_eval, &s.state, &mut d_state, &mut d_flow);
        ks_state[0] = d_state.clone();
        ks_flow[0] = d_flow.clone();
        // k2, k3 at the half-step x-stages, with S advanced by the previous slope.
        let s2s: Vec<f64> = s.state.iter().zip(&ks_state[0]).map(|(x, k)| x + 0.5 * dt * k).collect();
        sensitivity_derivs(model, &s2i, &s2r, params, param_model_idx, forcing, t + 0.5 * dt, dt, per_eval, &s2s, &mut d_state, &mut d_flow);
        ks_state[1] = d_state.clone();
        ks_flow[1] = d_flow.clone();
        let s3s: Vec<f64> = s.state.iter().zip(&ks_state[1]).map(|(x, k)| x + 0.5 * dt * k).collect();
        sensitivity_derivs(model, &s3i, &s3r, params, param_model_idx, forcing, t + 0.5 * dt, dt, per_eval, &s3s, &mut d_state, &mut d_flow);
        ks_state[2] = d_state.clone();
        ks_flow[2] = d_flow.clone();
        // k4 at the full-step x-stage.
        let s4s: Vec<f64> = s.state.iter().zip(&ks_state[2]).map(|(x, k)| x + dt * k).collect();
        sensitivity_derivs(model, &s4i, &s4r, params, param_model_idx, forcing, t + dt, dt, per_eval, &s4s, &mut d_state, &mut d_flow);
        ks_state[3] = d_state;
        ks_flow[3] = d_flow;
        (ks_state, ks_flow)
    } else {
        (Default::default(), Default::default())
    };

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

    // Combine the sensitivity blocks (NOT clamped — S and acc_sens are signed).
    // The clamp on the state above is a nonsmooth operation S does not model (§1c
    // clamp caveat); under `nuts` an active clamp is refused, so a valid gradient
    // trajectory never clamps. `acc_sens` mirrors the value `flow` combine.
    if let Some(s) = sens {
        let [k1, k2, k3, k4] = &ks_state;
        for k in 0..stage_state {
            s.state[k] += dt / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
        }
        let [k1, k2, k3, k4] = &ks_flow;
        for k in 0..stage_flow {
            s.flow[k] += dt / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
        }
    }

    Ok(())
}

/// gh#396: integrate the augmented `(x, S)` system over one forcing period
/// `[t_start, t_start + period]`, returning `(x(end), S(end))` — the building
/// block of the periodic-equilibrium solver.
///
/// - `x0` seeds `x(t_start)` (integer compartments; real compartments are refused
///   on the gradient path).
/// - `s0` seeds `S(t_start)` (`n_int × d`, row-major). With `forcing = false` and
///   `s0 = Iₙ` (so `d = n_int`), `S(end)` is the **monodromy** `M = ∂x(end)/∂x₀`.
///   With `forcing = true` and `s0 = 0` (`d = n_params`), `S(end)` is the
///   **partial** `∂Φ_P/∂θ` (initial state held fixed).
/// - `param_model_idx` maps the `d` columns to model parameters (ignored when
///   `forcing = false`).
///
/// Fixed-step RK4, last step clipped to land exactly on the period endpoint. No
/// effects/output/obs recording: the validity gate guarantees `[t_start, t_start
/// + period]` is effect-free and `P`-periodic, and the forcing is read at absolute
/// `t` so the stroboscopic phase is correct. `per_eval` is `t`-inert, so staging it
/// at `t_start` is exact.
///
/// Caveat: `rk4_step` clamps the value state at 0 but not `S` (§1c). On a genuinely
/// endemic (interior) cycle the clamp is inactive and `M` is exact; the solver's
/// oracles (equilibrium == burn-in, sensitivity three-way) are the gate that a
/// clamped iterate did not corrupt the result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn period_flow(
    model: &CompiledModel,
    params: &[f64],
    x0: &[f64],
    s0: &[f64],
    param_model_idx: &[usize],
    forcing: bool,
    t_start: f64,
    period: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>), SimError> {
    let ni = x0.len();
    let d = param_model_idx.len();
    debug_assert_eq!(s0.len(), ni * d, "S(t_start) seed must be n_int × d");
    let n_tr = model.model.transitions.len();
    let per_eval = crate::resolved_expr::stage_per_eval(model, params, t_start, dt);
    let mut int_vals = x0.to_vec();
    let mut real_vals: Vec<f64> = Vec::new();
    let mut sens = Sens { state: s0.to_vec(), flow: vec![0.0; n_tr * d], forcing };
    let t_end = t_start + period;
    let mut t = t_start;
    while t < t_end - MIN_STEP_EPS {
        let h = dt.min(t_end - t);
        rk4_step(
            model,
            &mut int_vals,
            &mut real_vals,
            None, // no flow accumulation — the solver needs only x and S
            params,
            t,
            h,
            per_eval.as_deref(),
            Some(&mut sens),
            param_model_idx,
        )?;
        t += h;
    }
    Ok((int_vals, sens.state))
}

/// gh#396: the integer-compartment derivative field `dx/dt` at a reference state
/// and absolute time `t`. Used by the warm-start periodicity gate to test that
/// the vector field (hence the forcing) is `P`-periodic near `T_eq` — a table /
/// `interpolated` forcing carries no declared period, so periodicity is checked
/// numerically rather than read off the forcing kind.
pub(crate) fn derivs_at(
    model: &CompiledModel,
    params: &[f64],
    int_vals: &[f64],
    t: f64,
    dt: f64,
) -> Result<Vec<f64>, SimError> {
    let per_eval = crate::resolved_expr::stage_per_eval(model, params, t, dt);
    let real_vals: Vec<f64> = Vec::new();
    let mut di = vec![0.0; int_vals.len()];
    let mut dr = vec![0.0; real_vals.len()];
    let mut df = vec![0.0; model.model.transitions.len()];
    ode_derivs(
        model,
        int_vals,
        &real_vals,
        params,
        t,
        dt,
        per_eval.as_deref(),
        &mut di,
        &mut dr,
        &mut df,
    )?;
    Ok(di)
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
/// boundary (`stop.t - t`, from [`Schedule::next_stop`]); the stepper MUST NOT
/// cross it.
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
        per_eval: Option<&[f64]>,
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
        per_eval: Option<&[f64]>,
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
            // ROUNDED integer state (`int_float_override: None, per_eval: None`), as it was before
            // this change. The user is warned once at load (see the CLI).
            let (is, rs) = to_states(&state.int, &state.real);
            let mut propensities = Vec::with_capacity(state.flow.len());
            eval_propensities(model, &is, &rs, params, t, h, per_eval, &mut propensities)?;
            for (i, &p) in propensities.iter().enumerate() {
                state.flow[i] += p * h;
            }
            rk4_step(model, &mut state.int, &mut state.real, None, params, t, h, per_eval, None, &[])?;
        } else {
            // Augmented flow (Q1B): `dc_i/dt = propensity_i` carried through the
            // SAME RK4 stages as the compartments — one mechanism, integrator-
            // order incidence, and no standalone 5th propensity eval.
            rk4_step(model, &mut state.int, &mut state.real, Some(&mut state.flow), params, t, h, per_eval, None, &[])?;
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
    per_eval: Option<&[f64]>,
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
        ode_derivs(model, &si, &sr, params, t + DP_C[i] * h, h, per_eval, &mut di, &mut dr, &mut df)?;
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
        per_eval: Option<&[f64]>,
    ) -> Result<f64, SimError> {
        let mut h = self.h.min(h_max);
        if !(h > 0.0) { h = h_max; }
        let mut rejections = 0u32;
        loop {
            let (y5_int, y5_real, flow_inc, err) =
                dopri5_try_step(model, params, t, h, state, self.atol, self.rtol, per_eval)?;
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

    // gh#272: stage the per-eval prologue ONCE for this θ-stable span. `params`
    // is fixed for this whole `run_ode` call, so the param/table-only
    // `per_eval_bindings` are evaluated here and lent (`per_eval.as_deref()`) into
    // every RK stage of every step — NOT recomputed per stage. The scratch is
    // owned by this call and passed as data, so it is structurally bound to this
    // θ (no shared cache to alias). `None` for models without per-eval bindings,
    // in which case `PerEvalRef` would fall through to on-demand eval anyway. This
    // single site covers forward ODE simulate AND `compute_ode_loglik` (both route
    // through `run_ode`).
    let per_eval: Option<Vec<f64>> =
        crate::resolved_expr::stage_per_eval(model, params, cfg.t_start, cfg.dt);

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
    let schedule = Schedule::exact_forward(
        cfg.dt,
        cfg.t_end,
        OutputTimes::from_model(model)?,
        EffectTimes::from_model(model, params)?,
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

    // Record initial snapshot. Initial-row convention (see `Trajectory` docs):
    // the t_start snapshot carries zeroed flows so `Σ flow == −Δstate`
    // reconciles over the whole path (gh#270).
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

    // gh#233 Layer 7: drive the merged timeline through the single boundary
    // authority. `next_stop` reports the next boundary + every reason it matters;
    // the ODE dynamics (stepper.advance + the h_max re-entry) stay HERE, and the
    // boundary dispatch (effects → output, in canonical order, with the
    // coincident batch and terminal output) goes through the shared `arrive` seam
    // — the logic that diverged per-backend in gh#70. Byte-identical to the old
    // next_boundary + two-dispatch-block + final-flush form (gate_corner_case_baseline).
    while let Some(stop) = schedule.next_stop(&cursor, t) {
        // Progress tick: report current time. RNG-free (ODE has no RNG at all).
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // Integrate toward the boundary. The stepper takes ≤ h_max and is
        // re-entered until it arrives (fixed RK4 takes min(dt, h_max); adaptive
        // rk45 takes several internal steps). `h_max <= MIN_STEP_EPS` ⇔ arrived.
        let h_max = stop.t - t;
        if h_max > MIN_STEP_EPS {
            t += stepper.advance(model, params, t, h_max, &mut state, per_eval.as_deref())?;
            continue;
        }

        // Arrived: dispatch effects-then-output via the shared seam. Effects apply
        // EXACTLY to the f64 vectors (no to_states round-trip) so the fractional
        // integrator state survives the boundary (the de-quantization the ODE
        // backend exists to provide); the due batch is derived at the boundary `t`
        // (grid = cfg.dt). Output records the post-effect state.
        schedule.arrive(
            &mut cursor,
            &stop,
            t,
            &mut state,
            |st, bt| {
                let mut batch = crate::schedule::EffectBatch::default();
                crate::effects::due_effects(model, &fire_steps, bt, cfg.dt, &mut batch);
                crate::effects::apply_boundary_batch_continuous(
                    model, &batch, &mut st.int, &mut st.real, params, bt, cfg.dt,
                )
            },
            |st, ot| {
                let (is, rs) = to_states(&st.int, &st.real);
                traj.push(Snapshot {
                    t: ot,
                    int_state: is,
                    real_state: rs,
                    flows: snapshot_flows(&st.flow),
                });
                for v in st.flow.iter_mut() { *v = 0.0; }
            },
        )?;

        if stop.is_end() { break; }
    }

    // Flush any output times beyond t_end (none under the standard schedule;
    // kept for parity with the old final drain — byte-identical).
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

/// One observation time's continuous state and forward sensitivities, recorded
/// by [`integrate_obs_sensitivity`] for the ODE-NUTS gradient scorer (gh#275).
/// This is the sensitivity analogue of a `compute_ode_loglik` snapshot: it
/// carries the UNROUNDED f64 state (§1d continuous prevalence) plus the two
/// sensitivity blocks the chain rule consumes — `state_sens` for `Prevalence`
/// streams, `inc_sens` for `Incidence` streams.
pub(crate) struct ObsSensitivity {
    pub t: f64,
    /// Continuous int-compartment state `x(t)` (unrounded, §1d).
    pub counts: Vec<f64>,
    /// Real-compartment state.
    pub real: Vec<f64>,
    /// `S = ∂counts/∂θ` at this obs time (`n_int × d`, row-major).
    pub state_sens: Vec<f64>,
    /// Per-transition cumulative flow since the previous obs time (`n_tr`) — the
    /// incidence over this observation interval (pomp accumulator semantics).
    pub inc: Vec<f64>,
    /// `∂inc/∂θ` (`n_tr × d`, row-major) — the incidence sensitivity over this
    /// interval, folded per stream by the scorer.
    pub inc_sens: Vec<f64>,
}

/// The augmented state carried through the boundary walk: the value `OdeState`
/// plus its forward sensitivities. Generic parameter `S` of [`Schedule::arrive`]
/// is instantiated at this type, so the shared boundary authority
/// (effects-then-output ordering, coincident batching, terminal drain) is reused
/// verbatim while the closures add sensitivity behavior (gh#275 architecture: own
/// thin driver over the shared `Schedule`/`rk4_step` seams, `run_ode` untouched).
struct AugState {
    ode: OdeState,
    sens: Sens,
}

/// Drive the fixed-step RK4 integration carrying the augmented `(x, flow, S,
/// acc_sens)` system, recording the continuous state and forward sensitivities at
/// each observation time (gh#275 §1c/§1d). The integration half of `det_grad`:
/// the returned records feed the obs-score chain rule (a separate step).
///
/// Routes every boundary decision through the shared `Schedule` authority
/// (`next_stop` / `arrive` / `MIN_STEP_EPS`) exactly as `run_ode` does — only the
/// per-step dynamics (`rk4_step` WITH sensitivity) and the per-boundary recording
/// differ. Scores on the OUTPUT grid (obs times must coincide with output
/// snapshots), matching `compute_ode_loglik`, so the two remain reconcilable.
///
/// `state_sens_0` is the `S(t_start) = ∂(initial_state)/∂θ` seed (`ic_grad`, §1c
/// C-seed): zero for a parameter that does not enter an initial condition, nonzero
/// for an `ivp` parameter. Passed in (not derived here) so the seed computation is
/// a separate, independently-tested concern.
///
/// Refuses (hard error, §1c/§1h) an adaptive integrator, a `dt`-in-rate
/// (RUNTIME_DT) model, and any scheduled effect — none of which have a defined
/// fixed-step forward sensitivity in v1.
pub(crate) fn integrate_obs_sensitivity(
    model: &CompiledModel,
    params: &[f64],
    param_model_idx: &[usize],
    // gh#396 Phase A: injected start state x(t_start). `None` → build it from the
    // model (`initial_state_continuous`, today's behaviour). `Some` → begin from
    // this continuous integer-compartment state (the periodic-equilibrium warm
    // start; real compartments are refused on the gradient path). The `state_sens_0`
    // seed and `cfg.t_start` were already caller-supplied.
    x0: Option<&[f64]>,
    state_sens_0: &[f64],
    cfg: &OdeConfig,
    obs_times: &[f64],
) -> Result<Vec<ObsSensitivity>, SimError> {
    let d = param_model_idx.len();
    let n_tr = model.model.transitions.len();

    model.validate_schedule(cfg.dt, params)?;

    // §1c: fixed-step RK4 only. An adaptive integrator sizes steps from the state
    // error alone (S under-resolved) and its step sequence is discontinuous in θ
    // (loglik only piecewise-smooth) — a silent value/gradient mismatch.
    if !matches!(model.model.simulation.integrator, ir::model::Integrator::Rk4) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) requires integrator = rk4 (fixed step); an adaptive \
             integrator (rk45) is refused — its step sequence is discontinuous in θ, \
             breaking the forward-sensitivity/loglik consistency NUTS needs (gh#275 §1c)"
                .to_string(),
        ));
    }
    // §1c: a `dt`-in-rate model keeps the first-order Euler flow, which has no
    // augmented sensitivity — refuse rather than emit an inconsistent gradient.
    if model.required_capabilities().contains(crate::Capabilities::RUNTIME_DT) {
        return Err(SimError::Validation(
            "ODE gradient (nuts) cannot differentiate a model that references `dt` in a \
             rate (Expr::Dt): its flow uses the first-order Euler scheme, which carries \
             no augmented sensitivity (gh#275 §1c)"
                .to_string(),
        ));
    }

    let per_eval: Option<Vec<f64>> =
        crate::resolved_expr::stage_per_eval(model, params, cfg.t_start, cfg.dt);
    // Continuous (un-rounded) initial state: the ODE gradient path must start
    // from the smooth initial value so value and forward-sensitivity are
    // consistent in θ (§1c; see `CompiledModel::initial_state_continuous`).
    let (int_s0, real_s0) = match x0 {
        // Phase A: warm-start from an injected continuous state. Real compartments
        // are refused on this path (`gradient_capability.rs`), so the injected
        // state is integer-compartments only and `real` starts empty.
        Some(x) => (x.to_vec(), Vec::new()),
        None => model.initial_state_continuous(params)?,
    };
    let ni = int_s0.len();
    debug_assert_eq!(
        state_sens_0.len(),
        ni * d,
        "S(t_start) seed must be n_int × d"
    );

    let mut st = AugState {
        ode: OdeState {
            int: int_s0,
            real: real_s0,
            flow: vec![0.0; n_tr],
        },
        sens: Sens {
            state: state_sens_0.to_vec(),
            flow: vec![0.0; n_tr * d],
            forcing: true,
        },
    };

    let schedule = Schedule::exact_forward(
        cfg.dt,
        cfg.t_end,
        OutputTimes::from_model(model)?,
        EffectTimes::from_model(model, params)?,
    );
    let mut cursor = Cursor::default();
    let fire_steps = model.resolve_fire_steps(cfg.dt, params);

    // Running incidence (and its sensitivity) accumulated across output snapshots
    // since the previous obs time — reset per obs, exactly as `compute_ode_loglik`
    // bridges a fine output grid to a sparse obs grid.
    let mut cum_flow = vec![0.0; n_tr];
    let mut cum_flow_sens = vec![0.0; n_tr * d];
    let mut records = Vec::with_capacity(obs_times.len());
    let mut next_obs = 0usize;
    let n_obs = obs_times.len();

    // Record the current boundary as an obs time if it matches, then reset the
    // running incidence. Returns whether an obs was recorded (for the overshoot
    // check). `t` is the boundary time; `st`/`cum_*` are read at their post-effect,
    // post-output-reset values.
    let try_record = |t: f64,
                          st: &AugState,
                          cum_flow: &mut [f64],
                          cum_flow_sens: &mut [f64],
                          records: &mut Vec<ObsSensitivity>,
                          next_obs: &mut usize|
     -> bool {
        if *next_obs < n_obs && (t - obs_times[*next_obs]).abs() < 1e-9 {
            records.push(ObsSensitivity {
                t,
                counts: st.ode.int.clone(),
                real: st.ode.real.clone(),
                state_sens: st.sens.state.clone(),
                inc: cum_flow.to_vec(),
                inc_sens: cum_flow_sens.to_vec(),
            });
            cum_flow.fill(0.0);
            cum_flow_sens.fill(0.0);
            *next_obs += 1;
            true
        } else {
            false
        }
    };

    let mut t = cfg.t_start;

    // Initial boundary at t_start (zero incidence). Mirror run_ode's cursor
    // management: record-if-obs, then pass the output so the walk is aligned.
    if schedule.output_due_at(&cursor, t) {
        try_record(t, &st, &mut cum_flow, &mut cum_flow_sens, &mut records, &mut next_obs);
        cursor.pass_output();
    }

    while let Some(stop) = schedule.next_stop(&cursor, t) {
        let h_max = stop.t - t;
        if h_max > MIN_STEP_EPS {
            // Fixed RK4 substep, clipped to the boundary; re-entered until arrival.
            let h = cfg.dt.min(h_max);
            rk4_step(
                model,
                &mut st.ode.int,
                &mut st.ode.real,
                Some(&mut st.ode.flow),
                params,
                t,
                h,
                per_eval.as_deref(),
                Some(&mut st.sens),
                param_model_idx,
            )?;
            t += h;
            continue;
        }

        // Arrived: dispatch effects-then-output through the shared `arrive` seam.
        // The output closure accumulates this interval's flow (and its sensitivity)
        // into the running cum, then resets — mirroring run_ode's per-output flow
        // reset, extended to `acc_sens`.
        schedule.arrive(
            &mut cursor,
            &stop,
            t,
            &mut st,
            |_st, bt| {
                let mut batch = crate::schedule::EffectBatch::default();
                crate::effects::due_effects(model, &fire_steps, bt, cfg.dt, &mut batch);
                if !batch.is_empty() {
                    // §1g: fixed-time θ-magnitude event sensitivity is a follow-up;
                    // v1 refuses any scheduled effect under the gradient path rather
                    // than emit a jump-term-only (silent-wrong) gradient.
                    return Err(SimError::Validation(
                        "ODE gradient (nuts) does not yet support scheduled interventions \
                         or events: their event-jump sensitivity is a follow-up (gh#275 \
                         §1g). Use gradient-free `mh` on `ode` for models with effects."
                            .to_string(),
                    ));
                }
                Ok(())
            },
            |st, _ot| {
                for i in 0..n_tr {
                    cum_flow[i] += st.ode.flow[i];
                }
                for k in 0..n_tr * d {
                    cum_flow_sens[k] += st.sens.flow[k];
                }
                st.ode.flow.fill(0.0);
                st.sens.flow.fill(0.0);
            },
        )?;

        let recorded = try_record(
            t,
            &st,
            &mut cum_flow,
            &mut cum_flow_sens,
            &mut records,
            &mut next_obs,
        );
        if !recorded && next_obs < n_obs && t > obs_times[next_obs] + 1e-9 {
            return Err(SimError::Validation(format!(
                "ODE output grid has no snapshot at obs time {} (integration reached \
                 t = {} without landing on it); the [output] schedule must include \
                 every observation time (gh#275 scores on the output grid)",
                obs_times[next_obs], t
            )));
        }

        if stop.is_end() {
            break;
        }
    }

    if next_obs < n_obs {
        return Err(SimError::Validation(format!(
            "ODE trajectory ended at t = {} before reaching obs time {}; simulate.to \
             must extend at least to the last obs time",
            t, obs_times[next_obs]
        )));
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::deriv::DerivEntry;
    use ir::expr::Expr;

    /// Load the `pure_death` golden and hand-set its θ- and state-gradients so the
    /// sensitivity spine has a `rate_grad` (`∂rate/∂mu`) and a `rate_state_grad`
    /// (`∂rate/∂N`) to consume. `pure_death` is the ideal oracle: one compartment
    /// `N`, one parameter `mu`, one transition `N →∅` at rate `mu·N`, so
    /// `dN/dt = -mu·N`, `N(t) = N0·e^{-mu·t}`, and `∂N/∂mu = -t·N(t)` in closed
    /// form — with the two Jacobians `∂rate/∂mu = N` and `∂rate/∂N = mu` equally
    /// trivial. The compiler does not yet EMIT these (WrtPop emission is the last
    /// step of unit 1), so the test hand-installs them — validating the Rust
    /// sensitivity assembly independently of, and ahead of, the OCaml emission.
    fn compiled_pure_death() -> CompiledModel {
        let mut model = load_golden("pure_death");
        let death = &mut model.transitions[0];
        // ∂(mu·N)/∂mu = N  (a Pop leaf — the param-keyed rate_grad, J_θ).
        death
            .rate_grad
            .insert("mu".to_string(), DerivEntry::Grad(Expr::pop("N")));
        // ∂(mu·N)/∂N = mu  (a Param leaf — the compartment-keyed rate_state_grad,
        // J_x — resolved through the CompGradMap compartment resolver).
        death
            .rate_state_grad
            .0
            .insert("N".to_string(), DerivEntry::Grad(Expr::param("mu")));
        CompiledModel::new(model).expect("pure_death with hand-set grads must compile")
    }

    /// `two_state`: A⇌B reversible linear reaction — forward `alpha·A`, backward
    /// `beta_r·B` — hand-set with its (trivial, linear) θ- and state-gradients.
    /// Unlike `pure_death` this has `d=2` parameters over `ni=2` compartments with
    /// genuine off-diagonal coupling (the forward transition moves mass A→B, so
    /// `∂B/∂alpha ≠ 0` even though `alpha` enters only the A-rate), so it exercises
    /// the row-major `s[comp*d + param]` indexing a `d=1` oracle cannot.
    fn compiled_two_state() -> CompiledModel {
        let mut model = load_golden("two_state");
        for tr in &mut model.transitions {
            match tr.name.as_str() {
                "forward" => {
                    // rate = alpha·A
                    tr.rate_grad
                        .insert("alpha".to_string(), DerivEntry::Grad(Expr::pop("A")));
                    tr.rate_state_grad
                        .0
                        .insert("A".to_string(), DerivEntry::Grad(Expr::param("alpha")));
                }
                "backward" => {
                    // rate = beta_r·B
                    tr.rate_grad
                        .insert("beta_r".to_string(), DerivEntry::Grad(Expr::pop("B")));
                    tr.rate_state_grad
                        .0
                        .insert("B".to_string(), DerivEntry::Grad(Expr::param("beta_r")));
                }
                other => panic!("unexpected two_state transition {other}"),
            }
        }
        CompiledModel::new(model).expect("two_state with hand-set grads must compile")
    }

    fn load_golden(name: &str) -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ir/golden")
            .join(format!("{name}.ir.json"));
        let contents =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {name}: {e}"))
    }

    /// Integrate the compartments `x` and cumulative flow (and, when `with_sens`,
    /// the forward sensitivities [`Sens`]: `state[comp*d+param]=∂x/∂θ` and
    /// `flow[tr*d+param]=∂(cumulative flow)/∂θ`) from `t=0` to `t_end` with fixed
    /// step `dt`. `params` is passed by value at eval (read by index, no folding),
    /// so the value path can be re-evaluated at a perturbed θ for the
    /// finite-difference oracle without recompiling. `S(0)=0` and `acc_sens(0)=0`
    /// (initial counts/flows are constants, not functions of θ). Returns
    /// `(x(t_end), cumulative flow(t_end), Sens(t_end)?)`.
    fn integrate(
        cm: &CompiledModel,
        params: &[f64],
        init: &[f64],
        param_model_idx: &[usize],
        t_end: f64,
        dt: f64,
        with_sens: bool,
    ) -> (Vec<f64>, Vec<f64>, Option<Sens>) {
        let ni = init.len();
        let nf = cm.model.transitions.len();
        let d = param_model_idx.len();
        let mut int = init.to_vec();
        let mut real: Vec<f64> = vec![];
        let mut flow = vec![0.0f64; nf];
        let mut sens = Sens { state: vec![0.0f64; ni * d], flow: vec![0.0f64; nf * d], forcing: true };

        let n_steps = (t_end / dt).round() as usize;
        let mut t = 0.0;
        for _ in 0..n_steps {
            if with_sens {
                rk4_step(
                    cm, &mut int, &mut real, Some(&mut flow), params, t, dt, None,
                    Some(&mut sens), param_model_idx,
                )
                .expect("rk4_step");
            } else {
                rk4_step(cm, &mut int, &mut real, Some(&mut flow), params, t, dt, None, None, &[])
                    .expect("rk4_step");
            }
            t += dt;
        }
        (int, flow, if with_sens { Some(sens) } else { None })
    }

    /// The forward-sensitivity spine, validated end-to-end against BOTH a
    /// finite-difference of the integrated trajectory and the closed-form
    /// `∂N/∂mu = -t·N(t)`. This is the gate for the Rust sensitivity assembly
    /// (`sensitivity_derivs` + the augmented `(x, S)` RK4 stages): it integrates
    /// `x(t)` and `S(t) = ∂x/∂mu` together, then checks `S(t_end)` against a
    /// central finite difference of `x(t_end)` under a `mu`-perturbation — which
    /// validates the derivative assembly AND the integration coupling, not just
    /// one rate evaluation.
    #[test]
    fn forward_sensitivity_matches_finite_difference_and_analytic() {
        let cm = compiled_pure_death();
        let mu = 0.1;
        let n0 = 1000.0; // pure_death initial_conditions: {"explicit": {"N": 1000}}
        let t_end = 5.0;
        let dt = 0.01;
        let pmi = [0usize]; // mu is model param index 0 (the only parameter)

        // Integrate x and S together.
        let (int, _flow, sens) = integrate(&cm, &[mu], &[n0], &pmi, t_end, dt, true);
        let n_final = int[0];
        let s_final = sens.unwrap().state[0];

        // Central finite difference of the SAME discrete integrator — S(t_end) is
        // the exact derivative of the discrete N(t_end), so this matches to FD
        // truncation (O(eps²)), far tighter than the analytic comparison.
        let eps = 1e-5;
        let (int_p, _, _) = integrate(&cm, &[mu + eps], &[n0], &pmi, t_end, dt, false);
        let (int_m, _, _) = integrate(&cm, &[mu - eps], &[n0], &pmi, t_end, dt, false);
        let s_fd = (int_p[0] - int_m[0]) / (2.0 * eps);

        // Closed form: N(t) = N0·e^{-mu·t}, ∂N/∂mu = -t·N(t). Compared at a looser
        // tolerance because it carries the RK4 discretization error the FD does not.
        let n_analytic = n0 * (-mu * t_end).exp();
        let s_analytic = -t_end * n_analytic;

        // Sanity: the value path itself tracks the exponential (RK4, dt=0.01).
        assert!(
            (n_final - n_analytic).abs() < 1e-3 * n_analytic,
            "N(t_end) {n_final} vs analytic {n_analytic}"
        );

        // S vs finite difference — the assembly + integration are self-consistent.
        assert!(
            (s_final - s_fd).abs() < 1e-4 * s_fd.abs(),
            "S(t_end) {s_final} vs finite difference {s_fd} (rel err {})",
            ((s_final - s_fd) / s_fd).abs()
        );

        // S vs analytic — the assembly integrates the RIGHT sensitivity, not just
        // a self-consistent wrong one.
        assert!(
            (s_final - s_analytic).abs() < 1e-3 * s_analytic.abs(),
            "S(t_end) {s_final} vs analytic {s_analytic} (rel err {})",
            ((s_final - s_analytic) / s_analytic).abs()
        );
    }

    /// The flow-sensitivity (`acc_sens = ∂(cumulative flow)/∂θ`) validated against
    /// both a finite difference of the integrated cumulative flow and the closed
    /// form. For pure_death the single transition's cumulative flow is
    /// `∫ mu·N dt = N0(1−e^{−mu·t}) = N0 − N(t)`, so `∂flow/∂mu = N0·t·e^{−mu·t} =
    /// −∂N/∂mu` — the flow sensitivity is the negative of the state sensitivity,
    /// an independent check on the raw (no-stoich) `d_flow` accumulator.
    #[test]
    fn flow_sensitivity_matches_finite_difference_and_analytic() {
        let cm = compiled_pure_death();
        let mu = 0.1;
        let n0 = 1000.0;
        let t_end = 5.0;
        let dt = 0.01;
        let pmi = [0usize];

        let (_, flow, sens) = integrate(&cm, &[mu], &[n0], &pmi, t_end, dt, true);
        let flow_final = flow[0];
        let flow_sens = sens.unwrap().flow[0];

        // Central FD of the cumulative flow.
        let eps = 1e-5;
        let (_, fp, _) = integrate(&cm, &[mu + eps], &[n0], &pmi, t_end, dt, false);
        let (_, fm, _) = integrate(&cm, &[mu - eps], &[n0], &pmi, t_end, dt, false);
        let flow_fd = (fp[0] - fm[0]) / (2.0 * eps);

        let flow_analytic = n0 * (1.0 - (-mu * t_end).exp());
        let flow_sens_analytic = n0 * t_end * (-mu * t_end).exp();

        assert!(
            (flow_final - flow_analytic).abs() < 1e-3 * flow_analytic,
            "cumulative flow {flow_final} vs analytic {flow_analytic}"
        );
        assert!(
            (flow_sens - flow_fd).abs() < 1e-4 * flow_fd.abs(),
            "∂flow/∂mu {flow_sens} vs finite difference {flow_fd}"
        );
        assert!(
            (flow_sens - flow_sens_analytic).abs() < 1e-3 * flow_sens_analytic,
            "∂flow/∂mu {flow_sens} vs analytic {flow_sens_analytic}"
        );
    }

    /// The `d>1` / off-diagonal gate: on `two_state` (A⇌B, params alpha & beta_r)
    /// the full `2×2` sensitivity `S[comp, param]` is integrated and checked
    /// column-by-column against a central finite difference in each parameter
    /// independently. This exercises the row-major `s[comp*d + param]` indexing and
    /// the cross-compartment coupling (`∂B/∂alpha ≠ 0` though alpha enters only the
    /// A-rate) — a transposition bug that `pure_death` (`d=ni=1`) cannot see.
    #[test]
    fn forward_sensitivity_two_state_offdiagonal_matches_fd() {
        let cm = compiled_two_state();
        let base = [0.5f64, 0.3]; // alpha, beta_r  (model param indices 0, 1)
        let init = [80.0f64, 20.0]; // A, B          (compartment indices 0, 1)
        let pmi = [0usize, 1];
        let d = 2;
        let t_end = 4.0;
        let dt = 0.01;

        let (_, _, sens) = integrate(&cm, &base, &init, &pmi, t_end, dt, true);
        let sens = sens.unwrap();
        let s = &sens.state; // [∂A/∂α, ∂A/∂β, ∂B/∂α, ∂B/∂β]
        let fs = &sens.flow; // [∂flow_fwd/∂α, ∂flow_fwd/∂β, ∂flow_bwd/∂α, ∂flow_bwd/∂β]

        // Central FD in each parameter independently, for BOTH the state and the
        // cumulative-flow trajectories (nf=2 transitions, d=2 params).
        let eps = 1e-5;
        let mut fd_state = [0.0f64; 4];
        let mut fd_flow = [0.0f64; 4];
        for (pk, _) in pmi.iter().enumerate() {
            let mut pp = base;
            let mut pm = base;
            pp[pk] += eps;
            pm[pk] -= eps;
            let (xp, flp, _) = integrate(&cm, &pp, &init, &pmi, t_end, dt, false);
            let (xm, flm, _) = integrate(&cm, &pm, &init, &pmi, t_end, dt, false);
            for comp in 0..init.len() {
                // FD[comp][pk] lands at the same row-major slot the assembly fills.
                fd_state[comp * d + pk] = (xp[comp] - xm[comp]) / (2.0 * eps);
            }
            for tr in 0..2 {
                fd_flow[tr * d + pk] = (flp[tr] - flm[tr]) / (2.0 * eps);
            }
        }

        for k in 0..4 {
            assert!(
                (s[k] - fd_state[k]).abs() < 1e-4 * fd_state[k].abs().max(1.0),
                "S[{k}] = {} vs finite difference {} (comp {}, param {})",
                s[k],
                fd_state[k],
                k / d,
                k % d
            );
            assert!(
                (fs[k] - fd_flow[k]).abs() < 1e-4 * fd_flow[k].abs().max(1.0),
                "flow_sens[{k}] = {} vs finite difference {} (transition {}, param {})",
                fs[k],
                fd_flow[k],
                k / d,
                k % d
            );
        }
        // Guard that the off-diagonals are genuinely non-trivial (else the test
        // would pass on an all-zero S): ∂B/∂α (slot 2) must be materially nonzero,
        // and likewise the forward flow must depend on beta_r (slot 1) through the
        // backward-transition coupling.
        assert!(
            s[2].abs() > 1.0,
            "off-diagonal ∂B/∂alpha should be materially nonzero, got {}",
            s[2]
        );
        assert!(
            fs[1].abs() > 1.0,
            "off-diagonal ∂flow_fwd/∂beta_r should be materially nonzero, got {}",
            fs[1]
        );
    }

    /// The schedule-driven recorder (`integrate_obs_sensitivity`) — the det_grad
    /// integration half — validated over ≥2 obs reset intervals (the gh#187
    /// silent-gap gate: a prevalence-only or single-interval check passes with the
    /// incidence-reset bugged). On two_state (A⇌B) with obs at t=10,20,30:
    ///
    /// - `state_sens` and per-interval `inc_sens` are checked against central
    ///   finite differences of the recorder's own `counts`/`inc` — validating the
    ///   schedule-driven sensitivity carry + the `arrive` reuse.
    /// - a mass-balance identity `A(30) = A0 − Σinc_fwd + Σinc_bwd` checks the
    ///   incidence RESET independently: a broken reset makes both `inc` and
    ///   `inc_sens` consistently wrong (so FD alone can't see it), but over-counts
    ///   the summed incidence and fails the balance.
    #[test]
    fn obs_sensitivity_recorder_matches_fd_over_reset_intervals() {
        let cm = compiled_two_state();
        let base = [0.5f64, 0.3];
        let pmi = [0usize, 1];
        let d = 2;
        let ni = 2; // A, B
        let n_tr = 2; // forward, backward
        let cfg = OdeConfig { t_start: 0.0, t_end: 30.0, dt: 0.25 };
        let obs = [10.0f64, 20.0, 30.0];
        let seed = vec![0.0f64; ni * d]; // params don't enter two_state's initial counts

        let recs = integrate_obs_sensitivity(&cm, &base, &pmi, None, &seed, &cfg, &obs).unwrap();
        assert_eq!(recs.len(), 3, "one record per obs time");

        // Central FD in each parameter (hoisted: perturbation depends on the param,
        // not the record).
        let eps = 1e-5;
        let perturbed: Vec<(Vec<ObsSensitivity>, Vec<ObsSensitivity>)> = (0..d)
            .map(|pk| {
                let mut pp = base;
                let mut pm = base;
                pp[pk] += eps;
                pm[pk] -= eps;
                (
                    integrate_obs_sensitivity(&cm, &pp, &pmi, None, &seed, &cfg, &obs).unwrap(),
                    integrate_obs_sensitivity(&cm, &pm, &pmi, None, &seed, &cfg, &obs).unwrap(),
                )
            })
            .collect();

        for (ri, rec) in recs.iter().enumerate() {
            // The record is tagged with its obs time, and two_state has no
            // real-valued compartments (the recorder still carries the block).
            assert!((rec.t - obs[ri]).abs() < 1e-9, "record {ri} tagged t={}", rec.t);
            assert!(rec.real.is_empty(), "two_state has no real compartments");
            for pk in 0..d {
                let (rp, rm) = &perturbed[pk];
                for comp in 0..ni {
                    let fd = (rp[ri].counts[comp] - rm[ri].counts[comp]) / (2.0 * eps);
                    let sym = rec.state_sens[comp * d + pk];
                    assert!(
                        (sym - fd).abs() < 1e-4 * fd.abs().max(1.0),
                        "state_sens rec{ri} comp{comp} p{pk}: {sym} vs FD {fd}"
                    );
                }
                for tr in 0..n_tr {
                    let fd = (rp[ri].inc[tr] - rm[ri].inc[tr]) / (2.0 * eps);
                    let sym = rec.inc_sens[tr * d + pk];
                    assert!(
                        (sym - fd).abs() < 1e-4 * fd.abs().max(1.0),
                        "inc_sens rec{ri} tr{tr} p{pk}: {sym} vs FD {fd}"
                    );
                }
            }
        }

        // Independent reset check: mass balance over the whole [0,30] window.
        // forward removes A, backward adds A; the per-interval incidences must sum
        // to the total flow, so A(30) = A0 − Σinc_fwd + Σinc_bwd.
        let sum_fwd: f64 = recs.iter().map(|r| r.inc[0]).sum();
        let sum_bwd: f64 = recs.iter().map(|r| r.inc[1]).sum();
        let a_final = recs.last().unwrap().counts[0];
        let a0 = 80.0;
        assert!(
            (a_final - (a0 - sum_fwd + sum_bwd)).abs() < 1e-6,
            "mass balance A(30)={a_final} vs A0−Σfwd+Σbwd={}",
            a0 - sum_fwd + sum_bwd
        );
        // The reset is genuinely exercised: per-interval forward incidence must
        // DECREASE across intervals (A relaxes toward equilibrium), which a
        // never-reset cumulative accumulator would violate (it would increase).
        assert!(
            recs[2].inc[0] < recs[0].inc[0],
            "per-interval forward incidence should decrease (reset working): \
             {} !< {}",
            recs[2].inc[0],
            recs[0].inc[0]
        );
    }

    /// The value path (`S = None`) must be byte-identical after the augmented-RK4
    /// change: integrating with `with_sens=false` reproduces the same `N(t_end)`
    /// the sensitivity run produces for `x`, so carrying `S` never perturbs the
    /// compartment trajectory (the augmented system is triangular — `dx/dt` does
    /// not depend on `S`).
    #[test]
    fn value_path_unchanged_by_sensitivity_carry() {
        let cm = compiled_pure_death();
        let (with, _, _) = integrate(&cm, &[0.1], &[1000.0], &[0], 5.0, 0.01, true);
        let (without, _, _) = integrate(&cm, &[0.1], &[1000.0], &[0], 5.0, 0.01, false);
        assert_eq!(
            with[0].to_bits(),
            without[0].to_bits(),
            "carrying S must not change the compartment trajectory bit-for-bit"
        );
    }
}
