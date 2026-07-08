//! Periodic-equilibrium warm-start solver (gh#396).
//!
//! Solves the seasonal limit cycle as the fixed point of the one-period
//! stroboscopic (Poincaré) map anchored at the warm-start time `T_eq`:
//!
//! ```text
//! X* = Φ_P(X*; θ),   Φ_P = flow of the ODE over [T_eq, T_eq + P] at θ.
//! ```
//!
//! Newton uses the monodromy `M = ∂Φ_P/∂X₀` (from [`crate::ode::period_flow`] in
//! its `forcing = false`, identity-seeded mode). The equilibrium sensitivity
//! follows from the implicit function theorem, `∂X*/∂θ = (I − M)⁻¹ ∂Φ_P/∂θ`, and
//! reuses the same factorization.
//!
//! ## Conservation
//!
//! A closed model conserves a linear quantity `cᵀx` (e.g. per-village population),
//! making `cᵀM = cᵀ` so `(I − M)` is singular (a unit Floquet multiplier),
//! *endemic or not*. Any Runge–Kutta method preserves linear invariants exactly,
//! so the conserved singular values of `(I − M)` sit at floating-point roundoff
//! (~1e-12) while the slowest genuine mode of a decades-long relaxation is at
//! `1 − |μ| ≈ 1e-2` — a ~10-order-of-magnitude gap, so an SVD threshold cleanly
//! separates them (no fragile "0.99 vs 1.00" call). We solve on the conservation
//! manifold `{ CᵀX = CᵀX_init }` via a bordered/KKT system, which is nonsingular
//! whenever the *transverse* cycle is hyperbolic.

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use nalgebra::{DMatrix, DVector};

/// Singular values of `(I − M)` below `CONS_EPS · σ_max` are conserved/neutral
/// directions. Genuine conservation is exact (RK preserves linear invariants), so
/// these sit at ~1e-12; the slowest genuine mode of a many-decade relaxation is at
/// ~1e-2. Any value in `[1e-10, 1e-4]` separates them.
const CONS_EPS: f64 = 1e-8;

/// The endemic gate. If the smallest **non-conserved** singular value of `(I − M)`
/// falls below `ENDEMIC_MIN · σ_max`, the transverse cycle is (near-)non-hyperbolic
/// — a mode with relaxation > ~1e6 periods, i.e. the model is not settling to a
/// seasonal cycle. Refuse rather than seed a garbage equilibrium. Sits far above
/// `CONS_EPS`, so a genuine slow mode is never mistaken for conservation.
const ENDEMIC_MIN: f64 = 1e-6;

/// Newton converges when `‖Φ_P(X) − X‖∞ < TOL_EQ · (1 + ‖X‖∞)`.
const TOL_EQ: f64 = 1e-8;

/// Newton iteration cap. A hyperbolic cycle converges in a handful of steps;
/// exhausting this signals a marginally-stable (near-non-endemic) cycle.
const MAX_NEWTON: usize = 60;

/// Fit-time configuration for periodic-equilibrium warm-start. Threaded from the
/// fit config (CLI/TOML) through to [`super::inference::ode_grad::det_grad`].
#[derive(Clone, Copy, Debug)]
pub struct WarmStart {
    /// The warm-start anchor time `T_eq` (absolute model time): integration begins
    /// here from the solved equilibrium instead of from `origin`. A single scalar
    /// `≤ min(first_obs)`, decoupled from the per-stream conditioning window.
    pub t_eq: f64,
    /// The forcing fundamental period `P` (model time units) — declared, since an
    /// `interpolated`/table forcing carries no period.
    pub period: f64,
}

/// The solved seasonal equilibrium at `T_eq` and its parameter sensitivity.
pub struct Equilibrium {
    /// `X*` at `T_eq`: `n_int` continuous compartment values.
    pub x_star: Vec<f64>,
    /// `∂X*/∂θ`, `n_int × d` row-major (`d = param_model_idx.len()`) — the
    /// forward-sensitivity seed `S(T_eq)` for the data-window integration.
    pub x_star_sens: Vec<f64>,
    /// Number of conserved directions detected (diagnostic).
    pub n_conserved: usize,
    /// Newton iterations taken (diagnostic).
    pub iters: usize,
}

/// Solve the seasonal equilibrium `X*(θ)` and its sensitivity `∂X*/∂θ` at `T_eq`.
///
/// - `param_model_idx` — the `d` estimated parameters (model indices).
/// - `ic_grad_seed` — `∂X_init/∂θ` (`n_int × d`, the same seed `det_grad` builds):
///   supplies the conserved-direction sensitivity `∂b/∂θ = Cᵀ · ic_grad_seed`.
/// - `period` (`P`) and `dt` — the forcing period and the fixed RK4 step.
///
/// Assumes the pre-`T_eq` dynamics are effect-free and `P`-periodic (the caller's
/// validity gate). Refuses (hard error) a non-endemic parameterization.
pub fn solve_equilibrium(
    model: &CompiledModel,
    params: &[f64],
    param_model_idx: &[usize],
    ic_grad_seed: &[f64],
    t_eq: f64,
    period: f64,
    dt: f64,
) -> Result<Equilibrium, SimError> {
    let (int_s0, _real) = model.initial_state_continuous(params)?;
    let ni = int_s0.len();
    let d = param_model_idx.len();
    if ni == 0 {
        return Err(SimError::Validation(
            "periodic-equilibrium warm-start: model has no integer compartments".into(),
        ));
    }

    // Identity seed for the monodromy (n_int × n_int, row-major) and a dummy
    // parameter map (unused when forcing = false).
    let mut ident = vec![0.0f64; ni * ni];
    for i in 0..ni {
        ident[i * ni + i] = 1.0;
    }
    let mono_idx: Vec<usize> = (0..ni).collect();

    let x_init = DVector::from_column_slice(&int_s0);
    let mut x = x_init.clone();

    // Conserved directions (structural: detected once from the first monodromy and
    // reused — a linear invariant of the vector field, independent of x and θ).
    let (c_mat, b_vec) = {
        let (_phi0, m0_flat) = crate::ode::period_flow(
            model, params, x.as_slice(), &ident, &mono_idx, false, t_eq, period, dt,
        )?;
        let m0 = DMatrix::from_row_slice(ni, ni, &m0_flat);
        let (c, _sigma_min) = detect_conserved(&m0, ni)?;
        let b = c.transpose() * &x_init; // b = Cᵀ X_init — the pinned totals
        (c, b)
    };
    let k = c_mat.ncols();

    // --- Newton on the stroboscopic fixed point, on the conservation manifold ---
    let mut iters = 0usize;
    let mut converged = false;
    for it in 0..MAX_NEWTON {
        iters = it + 1;
        let (phi_vec, m_flat) = crate::ode::period_flow(
            model, params, x.as_slice(), &ident, &mono_idx, false, t_eq, period, dt,
        )?;
        let phi = DVector::from_vec(phi_vec);
        let m = DMatrix::from_row_slice(ni, ni, &m_flat);

        let r = &phi - &x; // residual Φ_P(X) − X
        let rinf = r.amax();
        if rinf < TOL_EQ * (1.0 + x.amax()) {
            converged = true;
            break;
        }

        let bordered = build_bordered(&m, &c_mat, ni, k);
        let mut rhs = DVector::<f64>::zeros(ni + k);
        rhs.rows_mut(0, ni).copy_from(&r);
        if k > 0 {
            let cons_resid = &b_vec - c_mat.transpose() * &x;
            rhs.rows_mut(ni, k).copy_from(&cons_resid);
        }
        let sol = bordered.lu().solve(&rhs).ok_or_else(non_endemic_err)?;
        x += sol.rows(0, ni);
    }
    if !converged {
        return Err(SimError::Validation(format!(
            "periodic-equilibrium Newton did not converge in {MAX_NEWTON} iterations. \
             The seasonal cycle is (near-)marginally stable at these parameters \
             (non-endemic); fit without warm-start or adjust the parameters."
        )));
    }

    // --- Sensitivity at the converged X* (monodromy re-evaluated there) ---
    let (_phi_star, m_star_flat) = crate::ode::period_flow(
        model, params, x.as_slice(), &ident, &mono_idx, false, t_eq, period, dt,
    )?;
    let m_star = DMatrix::from_row_slice(ni, ni, &m_star_flat);
    // Endemic gate on the transverse conditioning at X*.
    endemic_gate(&m_star, ni, k)?;

    // ∂Φ_P/∂θ — the PARTIAL (initial state held fixed): a forcing-ON, zero-seeded
    // one-period solve at X*.
    let zero_seed = vec![0.0f64; ni * d];
    let (_phi2, dphi_flat) = crate::ode::period_flow(
        model, params, x.as_slice(), &zero_seed, param_model_idx, true, t_eq, period, dt,
    )?;
    let dphi = DMatrix::from_row_slice(ni, d, &dphi_flat); // n_int × d

    // ∂b/∂θ = Cᵀ · ∂X_init/∂θ (reuses the ic_grad seed for the conserved part).
    let ic_grad = DMatrix::from_row_slice(ni, d, ic_grad_seed);
    let db_dtheta = if k > 0 {
        c_mat.transpose() * &ic_grad
    } else {
        DMatrix::zeros(0, d)
    };

    // Solve the bordered system column-by-column, reusing one factorization.
    let bordered = build_bordered(&m_star, &c_mat, ni, k);
    let lu = bordered.lu();
    let mut x_star_sens = vec![0.0f64; ni * d];
    for col in 0..d {
        let mut rhs = DVector::<f64>::zeros(ni + k);
        for i in 0..ni {
            rhs[i] = dphi[(i, col)];
        }
        for kk in 0..k {
            rhs[ni + kk] = db_dtheta[(kk, col)];
        }
        let sol = lu.solve(&rhs).ok_or_else(non_endemic_err)?;
        for i in 0..ni {
            x_star_sens[i * d + col] = sol[i];
        }
    }

    Ok(Equilibrium {
        x_star: x.as_slice().to_vec(),
        x_star_sens,
        n_conserved: k,
        iters,
    })
}

/// Detect the conserved directions `C` (`n_int × k`) as the left-null space of
/// `(I − M)` — columns of `U` (from the SVD) whose singular value is below
/// `CONS_EPS · σ_max`. Returns `C` and the smallest non-conserved singular value.
fn detect_conserved(m: &DMatrix<f64>, ni: usize) -> Result<(DMatrix<f64>, f64), SimError> {
    let i_minus_m = DMatrix::<f64>::identity(ni, ni) - m;
    let svd = i_minus_m.svd(true, false);
    let u = svd
        .u
        .as_ref()
        .ok_or_else(|| SimError::Validation("equilibrium: SVD produced no U".into()))?;
    let sv = &svd.singular_values;
    let sigma_max = sv.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    // Singular values are descending; the conserved (near-zero) ones are at the end.
    let mut cols: Vec<usize> = Vec::new();
    let mut smallest_non_conserved = sigma_max;
    for j in 0..ni {
        if sv[j] < CONS_EPS * sigma_max {
            cols.push(j);
        } else {
            smallest_non_conserved = sv[j]; // last (smallest) non-conserved as we descend
        }
    }
    let k = cols.len();
    let mut c = DMatrix::<f64>::zeros(ni, k);
    for (ci, &j) in cols.iter().enumerate() {
        c.set_column(ci, &u.column(j));
    }
    Ok((c, smallest_non_conserved))
}

/// Endemic gate: refuse if the smallest non-conserved mode is (near-)singular.
fn endemic_gate(m: &DMatrix<f64>, ni: usize, _k: usize) -> Result<(), SimError> {
    let i_minus_m = DMatrix::<f64>::identity(ni, ni) - m;
    let svd = i_minus_m.svd(false, false);
    let sv = &svd.singular_values;
    let sigma_max = sv.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    // Smallest singular value that is NOT a conserved direction.
    let mut smallest_non_conserved = sigma_max;
    for j in 0..ni {
        if sv[j] >= CONS_EPS * sigma_max {
            smallest_non_conserved = sv[j];
        }
    }
    if smallest_non_conserved < ENDEMIC_MIN * sigma_max {
        return Err(non_endemic_err());
    }
    Ok(())
}

/// Assemble the bordered/KKT matrix `[[I − M, C], [Cᵀ, 0]]` of size `n_int + k`.
fn build_bordered(m: &DMatrix<f64>, c: &DMatrix<f64>, ni: usize, k: usize) -> DMatrix<f64> {
    let i_minus_m = DMatrix::<f64>::identity(ni, ni) - m;
    let mut bordered = DMatrix::<f64>::zeros(ni + k, ni + k);
    bordered.view_mut((0, 0), (ni, ni)).copy_from(&i_minus_m);
    if k > 0 {
        bordered.view_mut((0, ni), (ni, k)).copy_from(c);
        bordered.view_mut((ni, 0), (k, ni)).copy_from(&c.transpose());
    }
    bordered
}

fn non_endemic_err() -> SimError {
    SimError::Validation(
        "periodic-equilibrium warm-start: no isolated stable seasonal cycle at these \
         parameters (a non-conservation Floquet multiplier is at/above 1 — the model \
         is not settling to a periodic equilibrium). Fit without warm-start, or adjust \
         the parameters that break endemicity (gh#396)."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_model::CompiledModel;

    const P: f64 = 365.25;
    const DT: f64 = 1.0;

    /// A closed seasonal SEIRS(+V≡0): `seir_vaccine_seasonal` with the SIA
    /// intervention removed (the ODE gradient path refuses scheduled effects; with
    /// no SIA, V stays 0). N = S+E+I+R is conserved (k = 1) — exercises the
    /// bordered/KKT solve. `N0` scaled to 1000 for manageable finite-difference.
    fn load_model() -> CompiledModel {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // Freshly-compiled IR (carries `rate_state_grad`; the `ir/golden/` copies
        // predate the gh#275 emission — see testdata/README.md).
        let path = std::path::PathBuf::from(&manifest)
            .join("testdata/seir_seasonal_closed.ir.json");
        let contents = std::fs::read_to_string(&path).unwrap();
        let mut model: ir::Model = ir::from_str(&contents).unwrap();
        model.interventions = Vec::new();
        // The golden's parameters are scenario-valued; apply concrete baseline
        // values (R0 = β/γ = 3, seasonal amplitude 0.15) so `CompiledModel::new`
        // has values. N scaled to 1e3 (frequency-dependent β·S·I/N ⇒ endemic
        // fractions invariant to N) for manageable finite differences.
        let vals: &[(&str, f64)] = &[
            ("beta", 0.3),
            ("sigma", 0.2),
            ("gamma", 0.1),
            ("omega", 0.003),
            ("reversion_rate", 1e-6),
            ("alpha", 0.15),
            ("phi_season", 90.0),
            ("vacc_frac", 0.8),
            ("N0", 1000.0),
            ("I0", 10.0),
        ];
        for p in &mut model.parameters {
            if let Some(&(_, v)) = vals.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            }
        }
        CompiledModel::new(model).expect("compile seasonal closed model")
    }

    fn baseline_params(cm: &CompiledModel) -> Vec<f64> {
        let n = cm.model.parameters.len();
        let mut params = vec![0.0; n];
        for p in &cm.model.parameters {
            params[cm.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        params
    }

    fn identity_seed(ni: usize) -> Vec<f64> {
        let mut m = vec![0.0; ni * ni];
        for i in 0..ni {
            m[i * ni + i] = 1.0;
        }
        m
    }

    fn param_idx(cm: &CompiledModel, name: &str) -> usize {
        cm.param_index[name]
    }

    #[test]
    fn equilibrium_is_a_fixed_point_and_conserves_n() {
        let cm = load_model();
        let params = baseline_params(&cm);
        let est: Vec<usize> = vec![param_idx(&cm, "beta"), param_idx(&cm, "gamma")];
        let ni = cm.initial_state_continuous(&params).unwrap().0.len();
        let ic_grad = vec![0.0; ni * est.len()]; // constant ICs
        let eq = solve_equilibrium(&cm, &params, &est, &ic_grad, 0.0, P, DT).unwrap();

        // Independent recompute: X* is a fixed point of the one-period map.
        let ident = identity_seed(ni);
        let midx: Vec<usize> = (0..ni).collect();
        let (phi, _m) =
            crate::ode::period_flow(&cm, &params, &eq.x_star, &ident, &midx, false, 0.0, P, DT)
                .unwrap();
        let resid: f64 = phi
            .iter()
            .zip(&eq.x_star)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let scale = eq.x_star.iter().cloned().fold(0.0, f64::max);
        assert!(resid < 1e-5 * (1.0 + scale), "X* not a fixed point: resid {resid}");

        // Conservation: N = ΣX detected (k=1) and preserved from the init totals.
        assert!((1..=MAX_NEWTON).contains(&eq.iters), "Newton iters out of range: {}", eq.iters);
        assert_eq!(eq.n_conserved, 1, "closed SEIRS should have k=1 conservation");
        let n_star: f64 = eq.x_star.iter().sum();
        let (x_init, _) = cm.initial_state_continuous(&params).unwrap();
        let n_init: f64 = x_init.iter().sum();
        assert!(
            (n_star - n_init).abs() < 1e-6 * n_init,
            "N not preserved: init {n_init}, equilibrium {n_star}"
        );
    }

    #[test]
    fn equilibrium_matches_long_burn_in() {
        // The solver's X* must be the ATTRACTING cycle the transient reaches — the
        // headline oracle. Iterate the period map (Picard = burn-in) from the model
        // IC and confirm it converges to X*.
        let cm = load_model();
        let params = baseline_params(&cm);
        let est: Vec<usize> = vec![param_idx(&cm, "beta")];
        let (x_init, _) = cm.initial_state_continuous(&params).unwrap();
        let ni = x_init.len();
        let ic_grad = vec![0.0; ni * est.len()];
        let eq = solve_equilibrium(&cm, &params, &est, &ic_grad, 0.0, P, DT).unwrap();

        let ident = identity_seed(ni);
        let midx: Vec<usize> = (0..ni).collect();
        let mut x = x_init.clone();
        for _ in 0..400 {
            x = crate::ode::period_flow(&cm, &params, &x, &ident, &midx, false, 0.0, P, DT)
                .unwrap()
                .0;
        }
        let scale = eq.x_star.iter().cloned().fold(0.0, f64::max);
        for i in 0..ni {
            assert!(
                (x[i] - eq.x_star[i]).abs() < 1e-3 * (1.0 + scale),
                "burn-in[{i}]={} != X*[{i}]={}",
                x[i],
                eq.x_star[i]
            );
        }
    }

    #[test]
    fn equilibrium_sensitivity_matches_finite_difference() {
        // The seed-trap catcher: ∂X*/∂θ from (I−M)⁻¹∂Φ/∂θ must match a central FD of
        // re-solving the equilibrium at θ±ε. A wrong (total-not-partial) ∂Φ/∂θ seed
        // fails here.
        let cm = load_model();
        let params = baseline_params(&cm);
        let est: Vec<usize> = vec![param_idx(&cm, "beta"), param_idx(&cm, "gamma")];
        let d = est.len();
        let ni = cm.initial_state_continuous(&params).unwrap().0.len();
        let ic_grad = vec![0.0; ni * d];
        let eq = solve_equilibrium(&cm, &params, &est, &ic_grad, 0.0, P, DT).unwrap();

        for (col, &pidx) in est.iter().enumerate() {
            let eps = 1e-4 * params[pidx].abs().max(1e-3);
            let mut pp = params.clone();
            pp[pidx] += eps;
            let mut pm = params.clone();
            pm[pidx] -= eps;
            let eqp = solve_equilibrium(&cm, &pp, &est, &ic_grad, 0.0, P, DT).unwrap();
            let eqm = solve_equilibrium(&cm, &pm, &est, &ic_grad, 0.0, P, DT).unwrap();
            let scale = eq.x_star.iter().cloned().fold(0.0, f64::max);
            for i in 0..ni {
                let fd = (eqp.x_star[i] - eqm.x_star[i]) / (2.0 * eps);
                let analytic = eq.x_star_sens[i * d + col];
                assert!(
                    (fd - analytic).abs() < 1e-2 * (1.0 + analytic.abs()) + 1e-3 * scale,
                    "∂X*[{i}]/∂θ[{col}]: analytic {analytic}, FD {fd}"
                );
            }
        }
    }

    #[test]
    fn monodromy_matches_finite_difference() {
        let cm = load_model();
        let params = baseline_params(&cm);
        let (x_init, _) = cm.initial_state_continuous(&params).unwrap();
        let ni = x_init.len();
        let ident = identity_seed(ni);
        let midx: Vec<usize> = (0..ni).collect();
        // A near-cycle interior state.
        let mut x = x_init.clone();
        for _ in 0..20 {
            x = crate::ode::period_flow(&cm, &params, &x, &ident, &midx, false, 0.0, P, DT)
                .unwrap()
                .0;
        }
        let (phi_base, m_flat) =
            crate::ode::period_flow(&cm, &params, &x, &ident, &midx, false, 0.0, P, DT).unwrap();
        for j in 0..ni {
            let eps = 1e-4 * x[j].abs().max(1.0);
            let mut xp = x.clone();
            xp[j] += eps;
            let phip = crate::ode::period_flow(&cm, &params, &xp, &ident, &midx, false, 0.0, P, DT)
                .unwrap()
                .0;
            // Central where the downward perturbation stays ≥ 0; forward otherwise
            // (a compartment at the clamp floor, e.g. V ≡ 0, would clamp the
            // downward step and halve a central difference).
            let (phim, denom) = if x[j] - eps > 0.0 {
                let mut xm = x.clone();
                xm[j] -= eps;
                let pm = crate::ode::period_flow(&cm, &params, &xm, &ident, &midx, false, 0.0, P, DT)
                    .unwrap()
                    .0;
                (pm, 2.0 * eps)
            } else {
                (phi_base.clone(), eps)
            };
            for i in 0..ni {
                let fd = (phip[i] - phim[i]) / denom;
                let analytic = m_flat[i * ni + j];
                assert!(
                    (fd - analytic).abs() < 2e-2 * (1.0 + analytic.abs()),
                    "M[{i},{j}]: analytic {analytic}, FD {fd}"
                );
            }
        }
    }
}
