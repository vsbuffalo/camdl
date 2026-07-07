//! No-U-Turn Sampler (NUTS) — Hoffman & Gelman (2014).
//!
//! Efficient HMC variant that automatically selects the number of leapfrog
//! steps via a tree-doubling procedure with a U-turn criterion. No manual
//! tuning of trajectory length.
//!
//! Used in PGAS as the θ|X update step, replacing one-at-a-time MH.
//! The target density is the complete-data log-posterior:
//!   log π(θ | X, y) = complete_data_loglik(θ, X, y) + log_prior(θ)

use serde::{Serialize, Deserialize};
use crate::rng::StatefulRng;

/// Mass matrix for HMC/NUTS. Controls how momentum translates into movement.
///
/// Diagonal: rescales each parameter independently by its posterior variance.
///   Handles scale differences (R0 ~30 vs sigma ~0.1) but NOT correlations.
///
/// Dense: full covariance matrix via Cholesky decomposition. Handles both
///   scale differences AND correlations (e.g., R0-amplitude ridge with r=0.94).
///   NUTS proposes along the ridge instead of zigzagging across it.
#[derive(Clone, Serialize, Deserialize)]
pub enum MassMatrix {
    /// Diagonal: M_inv[i] = Var(z_i). Identity when all 1.0.
    Diagonal(Vec<f64>),
    /// Dense: stores L_cov = Cholesky(Σ) where Σ = M^{-1} = empirical covariance.
    /// Following Stan's convention:
    ///   M^{-1} p = Σ p = L_cov (L_cov^T p)
    ///   p ~ N(0, M): solve L_cov p = z (forward substitution), z ~ N(0,I)
    ///   kinetic = 0.5 * ||L_cov^T p||^2
    Dense {
        dim: usize,
        /// Lower Cholesky factor of Σ (the covariance = M^{-1}), row-major.
        l_cov: Vec<f64>,
    },
}

impl MassMatrix {
    pub fn identity(d: usize) -> Self {
        MassMatrix::Diagonal(vec![1.0; d])
    }

    pub fn diagonal(variances: Vec<f64>) -> Self {
        MassMatrix::Diagonal(variances)
    }

    /// Build a dense mass matrix from an empirical covariance matrix (row-major).
    /// Stores L_cov = Cholesky(Σ) where Σ = covariance = M^{-1}.
    pub fn dense_from_covariance(cov: &[f64], d: usize) -> Self {
        assert_eq!(cov.len(), d * d);
        let mut reg = cov.to_vec();
        for i in 0..d {
            reg[i * d + i] += 1e-6; // regularize for numerical stability
        }
        let l_cov = cholesky_lower(&reg, d);
        MassMatrix::Dense { dim: d, l_cov }
    }

    /// Draw momentum: p ~ N(0, M) = N(0, Σ^{-1}).
    /// Diagonal: p_i = z_i / sqrt(Σ_ii) where z ~ N(0,I)
    /// Dense: p = L_cov^{-T} z where z ~ N(0,I)
    ///   Cov(p) = L_cov^{-T} (L_cov^{-T})^T = L_cov^{-T} L_cov^{-1} = Σ^{-1} = M. ✓
    pub fn draw_momentum(&self, rng: &mut StatefulRng) -> Vec<f64> {
        match self {
            MassMatrix::Diagonal(m_inv) => {
                m_inv.iter().map(|&mi| rng.normal() / mi.sqrt()).collect()
            }
            MassMatrix::Dense { dim, l_cov } => {
                let d = *dim;
                let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
                // Solve L_cov^T p = z (back substitution) → p = L_cov^{-T} z
                solve_upper_triangular_from_lower(l_cov, &z, d)
            }
        }
    }

    /// Kinetic energy: 0.5 * p^T M^{-1} p = 0.5 * p^T Σ p = 0.5 * ||L_cov^T p||^2
    pub fn kinetic_energy(&self, p: &[f64]) -> f64 {
        match self {
            MassMatrix::Diagonal(m_inv) => {
                p.iter().zip(m_inv).map(|(&pi, &mi)| pi * pi * mi).sum::<f64>() * 0.5
            }
            MassMatrix::Dense { dim, l_cov } => {
                let v = matvec_lower_transpose(l_cov, p, *dim);
                v.iter().map(|&vi| vi * vi).sum::<f64>() * 0.5
            }
        }
    }

    /// M^{-1} * p = Σ * p = L_cov (L_cov^T p)
    pub fn m_inv_times(&self, p: &[f64]) -> Vec<f64> {
        match self {
            MassMatrix::Diagonal(m_inv) => {
                p.iter().zip(m_inv).map(|(&pi, &mi)| pi * mi).collect()
            }
            MassMatrix::Dense { dim, l_cov } => {
                let v = matvec_lower_transpose(l_cov, p, *dim);
                matvec_lower(l_cov, &v, *dim)
            }
        }
    }
}

/// Lower triangular matrix × vector (row-major storage).
fn matvec_lower(l: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0; d];
    for i in 0..d {
        for j in 0..=i {
            y[i] += l[i * d + j] * x[j];
        }
    }
    y
}

/// Lower triangular matrix TRANSPOSE × vector (row-major storage).
fn matvec_lower_transpose(l: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0; d];
    for i in 0..d {
        for j in i..d {
            y[i] += l[j * d + i] * x[j];
        }
    }
    y
}

/// Solve L^T x = b where L is lower triangular (back substitution on the transpose).
fn solve_upper_triangular_from_lower(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut sum = b[i];
        for j in (i + 1)..d {
            sum -= l[j * d + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = sum / l[i * d + i];
    }
    x
}

/// Cholesky decomposition: A = L L^T. Returns L (lower triangular, row-major).
/// Uses shared `linalg::cholesky_lower`; falls back to identity-scaled factor
/// if the matrix is not positive definite.
fn cholesky_lower(a: &[f64], d: usize) -> Vec<f64> {
    match super::linalg::cholesky_lower(a, d) {
        Some(l) => l,
        None => {
            // Fallback: use a small diagonal factor so NUTS doesn't crash.
            // The mass matrix geometry will be wrong, but the sampler will
            // self-correct via step-size adaptation.
            let mut l = vec![0.0; d * d];
            for i in 0..d {
                l[i * d + i] = 1e-5;
            }
            l
        }
    }
}


/// Configuration for the NUTS sampler.
pub struct NUTSConfig {
    /// Maximum tree depth (number of doublings). Default 10 → up to 1024 leapfrog steps.
    pub max_tree_depth: usize,
    /// Step size for leapfrog integration. Adapted during warmup.
    pub step_size: f64,
    /// Mass matrix (diagonal or dense).
    pub mass_matrix: MassMatrix,
}

/// Result of one NUTS step.
pub struct NUTSStepResult {
    /// Proposed parameter values (on transformed scale).
    pub params: Vec<f64>,
    /// Log-posterior at the proposed point.
    pub log_posterior: f64,
    /// Whether the proposal was accepted (MH correction).
    pub accepted: bool,
    /// Number of leapfrog steps taken.
    pub n_leapfrog: usize,
    /// Tree depth reached.
    pub tree_depth: usize,
    /// Whether a divergence was detected (numerical instability).
    pub divergent: bool,
    /// Mean acceptance probability across the tree (for dual averaging).
    pub mean_accept_prob: f64,
    /// Initial Hamiltonian energy `H0 = -log_p + KE(p)` at the freshly drawn
    /// momentum — the per-iteration energy used for E-BFMI diagnostics (matches
    /// Stan's `energy__`, since momentum is resampled each step).
    pub energy: f64,
}

/// One NUTS step: propose all parameters jointly using gradients.
pub fn nuts_step(
    current_z: &[f64],
    current_log_p: f64,
    current_grad: &[f64],
    config: &NUTSConfig,
    log_prob_and_grad: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    rng: &mut StatefulRng,
) -> NUTSStepResult {
    let _d = current_z.len();
    let eps = config.step_size;
    let max_depth = config.max_tree_depth;

    let momentum = config.mass_matrix.draw_momentum(rng);
    let h0 = -current_log_p + config.mass_matrix.kinetic_energy(&momentum);
    let log_slice = -h0 - rng.exp(1.0);

    let mut z_minus = current_z.to_vec();
    let mut z_plus = current_z.to_vec();
    let mut p_minus = momentum.clone();
    let mut p_plus = momentum.clone();
    let mut grad_minus = current_grad.to_vec();
    let mut grad_plus = current_grad.to_vec();

    let mut z_proposal = current_z.to_vec();
    let mut log_p_proposal = current_log_p;
    let mut n_valid = 1usize;
    let mut n_leapfrog = 0usize;
    let mut tree_depth = 0usize;
    let mut divergent = false;
    let mut sum_accept_prob = 0.0;
    let mut n_accept_steps = 0usize;

    let delta_max = 1000.0;

    for depth in 0..max_depth {
        let direction: f64 = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };

        let (z_new, p_new, grad_new, z_prime, log_p_prime,
             n_prime, stop_prime, div_prime, n_lf, sum_ap, n_as) = if direction > 0.0 {
            build_tree(
                &z_plus, &p_plus, &grad_plus, direction, depth, eps,
                &config.mass_matrix, log_slice, h0, delta_max,
                log_prob_and_grad, rng,
            )
        } else {
            build_tree(
                &z_minus, &p_minus, &grad_minus, direction, depth, eps,
                &config.mass_matrix, log_slice, h0, delta_max,
                log_prob_and_grad, rng,
            )
        };

        n_leapfrog += n_lf;
        sum_accept_prob += sum_ap;
        n_accept_steps += n_as;

        if !stop_prime && n_prime > 0 {
            // gh#audit-H1. Hoffman & Gelman (2014) Algorithm 6 line 4
            // (slice-NUTS uniform-acceptance combine):
            //   p_accept = min(n_prime / n_valid, 1)
            // The previous form `n_prime / (n_valid + n_prime)` was
            // close to Algorithm 3's biased-toward-newer-subtree
            // combine and was undocumented — non-standard and a real
            // departure from H&G. The Alg-6 form is the canonical
            // slice-NUTS combine for the doubling-tree termination
            // scheme this implementation uses (cf. nuts.rs:323 slice
            // indicator).
            let accept_prob = if n_valid == 0 {
                1.0
            } else {
                (n_prime as f64 / n_valid as f64).min(1.0)
            };
            if rng.uniform() < accept_prob {
                z_proposal = z_prime;
                log_p_proposal = log_p_prime;
            }
        }

        n_valid += n_prime;
        divergent = divergent || div_prime;

        if direction > 0.0 {
            z_plus = z_new; p_plus = p_new; grad_plus = grad_new;
        } else {
            z_minus = z_new; p_minus = p_new; grad_minus = grad_new;
        }

        let stop = stop_prime || uturn(&z_minus, &z_plus, &p_minus, &p_plus,
                                        &config.mass_matrix);
        tree_depth = depth + 1;
        if stop { break; }
    }

    // Im14 in 2026-04-19 inference review: Vec!= here is bit-pattern
    // equality on f64 elements. A multinomial tree move that happens
    // to land on a numerically-equal proposal reports "rejected."
    // Vanishingly improbable; the correct notion — "some actual move
    // happened" — matches in practice.
    //
    // gh#81 Phase 2. Defense in depth: refuse to "accept" a proposal
    // whose z components or log_p are non-finite. The build_tree
    // slice-indicator already drops NaN-energy leaves from n_valid
    // (so z_proposal usually stays at current_z), but that is a
    // happy side-effect rather than an enforced invariant. The
    // explicit finiteness gate here documents the invariant: a
    // committed NUTS proposal MUST be finite. Without it, a future
    // refactor that changes the slice-indicator branch (or an
    // implementation choice where n_valid defaults to 1) would
    // silently regress this safety property.
    let proposal_finite = log_p_proposal.is_finite()
        && z_proposal.iter().all(|x| x.is_finite());
    let accepted = proposal_finite && z_proposal != current_z;
    let mean_accept_prob = if n_accept_steps > 0 {
        sum_accept_prob / n_accept_steps as f64
    } else { 0.0 };

    // If the proposal is non-finite, return the current state instead
    // of the corrupted one. Callers reading `result.params` blindly
    // (e.g. for adaptation Welford updates) get a usable f64 vector.
    let (out_params, out_log_p) = if proposal_finite {
        (z_proposal, log_p_proposal)
    } else {
        (current_z.to_vec(), current_log_p)
    };

    NUTSStepResult {
        params: out_params, log_posterior: out_log_p, accepted,
        n_leapfrog, tree_depth, divergent, mean_accept_prob, energy: h0,
    }
}

/// Leapfrog integrator: one step of Störmer-Verlet.
fn leapfrog(
    z: &[f64], p: &[f64], grad: &[f64],
    eps: f64, direction: f64, mass: &MassMatrix,
    log_prob_and_grad: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>) {
    let d = z.len();
    let dt = eps * direction;

    // Half-step momentum
    let mut p_half: Vec<f64> = (0..d).map(|i| p[i] + 0.5 * dt * grad[i]).collect();

    // Full-step position: z += dt * M^{-1} * p
    let m_inv_p = mass.m_inv_times(&p_half);
    let z_new: Vec<f64> = (0..d).map(|i| z[i] + dt * m_inv_p[i]).collect();

    let (log_p_new, grad_new) = log_prob_and_grad(&z_new);

    // Half-step momentum
    for i in 0..d {
        p_half[i] += 0.5 * dt * grad_new[i];
    }

    (z_new, p_half, log_p_new, grad_new)
}

/// Recursively build a balanced binary tree of leapfrog states.
#[allow(clippy::too_many_arguments)]
fn build_tree(
    z: &[f64], p: &[f64], grad: &[f64],
    direction: f64, depth: usize, eps: f64,
    mass: &MassMatrix, log_slice: f64, h0: f64, delta_max: f64,
    log_prob_and_grad: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    rng: &mut StatefulRng,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, usize, bool, bool, usize, f64, usize) {
    if depth == 0 {
        let (z_new, p_new, log_p_new, grad_new) =
            leapfrog(z, p, grad, eps, direction, mass, log_prob_and_grad);

        let h_new = -log_p_new + mass.kinetic_energy(&p_new);
        // gh#81 Phase 2. A non-finite proposal energy is ALWAYS
        // divergent. The legacy check `(h_new - h0).abs() > delta_max`
        // evaluates to `false` for NaN h_new (IEEE-754 unordered
        // comparison), so a NaN-energy leaf was reported as
        // non-divergent and the doubling tree happily continued past
        // it. The corollary at the top of `nuts_step` (slice indicator
        // `log_slice <= -h_new` also returns false for NaN h_new) just
        // happened to drop NaN proposals from the multinomial choice,
        // but the divergence flag itself was wrong — under-reporting
        // chain pathology to the adaptation loop and to the user.
        let energy_nonfinite = !h_new.is_finite();
        let n_valid = if !energy_nonfinite && log_slice <= -h_new { 1 } else { 0 };
        let divergent = energy_nonfinite || (h_new - h0).abs() > delta_max;
        let accept_prob = if energy_nonfinite { 0.0 } else { ((-h_new + h0).exp()).min(1.0) };

        return (z_new.clone(), p_new, grad_new, z_new, log_p_new,
                n_valid, divergent, divergent, 1, accept_prob, 1);
    }

    // Left subtree
    let (z_inner, p_inner, grad_inner, z_prime, log_p_prime,
         n_prime, stop_prime, div_prime, n_lf1, sum_ap1, n_as1) =
        build_tree(z, p, grad, direction, depth - 1, eps, mass,
                   log_slice, h0, delta_max, log_prob_and_grad, rng);

    if stop_prime {
        return (z_inner, p_inner, grad_inner, z_prime, log_p_prime,
                n_prime, true, div_prime, n_lf1, sum_ap1, n_as1);
    }

    // Right subtree
    let (z_outer, p_outer, grad_outer, z_dprime, log_p_dprime,
         n_dprime, stop_dprime, div_dprime, n_lf2, sum_ap2, n_as2) =
        build_tree(&z_inner, &p_inner, &grad_inner, direction, depth - 1, eps, mass,
                   log_slice, h0, delta_max, log_prob_and_grad, rng);

    // Random choice (Hoffman & Gelman Algorithm 6)
    let (z_proposal, log_p_proposal) = if n_dprime > 0 && n_prime + n_dprime > 0 {
        if rng.uniform() < n_dprime as f64 / (n_prime + n_dprime) as f64 {
            (z_dprime, log_p_dprime)
        } else {
            (z_prime, log_p_prime)
        }
    } else {
        (z_prime, log_p_prime)
    };

    let n_valid = n_prime + n_dprime;
    let divergent = div_prime || div_dprime;

    let z_minus = if direction > 0.0 { z.to_vec() } else { z_outer.clone() };
    let z_plus = if direction > 0.0 { z_outer.clone() } else { z.to_vec() };
    let p_minus = if direction > 0.0 { p.to_vec() } else { p_outer.clone() };
    let p_plus = if direction > 0.0 { p_outer.clone() } else { p.to_vec() };
    let stop = stop_dprime || uturn(&z_minus, &z_plus, &p_minus, &p_plus, mass);

    (z_outer, p_outer, grad_outer, z_proposal, log_p_proposal,
     n_valid, stop, divergent, n_lf1 + n_lf2, sum_ap1 + sum_ap2, n_as1 + n_as2)
}

/// U-turn criterion: (z+ - z-) · M^{-1} p < 0 for either endpoint.
fn uturn(z_minus: &[f64], z_plus: &[f64], p_minus: &[f64], p_plus: &[f64],
         mass: &MassMatrix) -> bool {
    let d = z_minus.len();
    let dz: Vec<f64> = (0..d).map(|i| z_plus[i] - z_minus[i]).collect();
    let m_inv_p_minus = mass.m_inv_times(p_minus);
    let m_inv_p_plus = mass.m_inv_times(p_plus);
    let dot_minus: f64 = dz.iter().zip(&m_inv_p_minus).map(|(&a, &b)| a * b).sum();
    let dot_plus: f64 = dz.iter().zip(&m_inv_p_plus).map(|(&a, &b)| a * b).sum();
    dot_minus < 0.0 || dot_plus < 0.0
}

/// Dual averaging for step size adaptation (Nesterov 2009).
pub struct DualAveraging {
    target_accept: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    mu: f64,
    log_eps_bar: f64,
    h_bar: f64,
    count: usize,
}

impl DualAveraging {
    pub fn new(initial_eps: f64, target_accept: f64) -> Self {
        DualAveraging {
            target_accept, gamma: 0.05, t0: 10.0, kappa: 0.75,
            mu: (10.0 * initial_eps).ln(),
            log_eps_bar: 0.0, h_bar: 0.0, count: 0,
        }
    }

    pub fn update(&mut self, accept_prob: f64) -> f64 {
        self.count += 1;
        let m = self.count as f64;
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - accept_prob);
        let log_eps = self.mu - self.h_bar * m.sqrt() / self.gamma;
        let eta = m.powf(-self.kappa);
        self.log_eps_bar = (1.0 - eta) * self.log_eps_bar + eta * log_eps;
        log_eps.exp()
    }

    pub fn final_step_size(&self) -> f64 {
        self.log_eps_bar.exp()
    }
}

/// Mass-matrix adaptation strategy — Stan's `metric` (Stan Reference Manual,
/// "HMC algorithm parameters"): whether and how warm-up learns the posterior
/// scale (and correlations) in the transformed `z`-space. The single most
/// important knob for gradient-NUTS efficiency on an ill-conditioned posterior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassMetric {
    /// Identity mass, no adaptation (Stan's `unit_e`). Correct only when the
    /// `z`-posterior is already isotropic; slow otherwise.
    Unit,
    /// Diagonal mass from the warm-up sample variances (Stan's default `diag_e`).
    /// Rescales each parameter to ~unit variance — fixes scale spread, but not
    /// correlations.
    Diagonal,
    /// Dense mass from the warm-up sample covariance (Stan's `dense_e`). Also
    /// absorbs parameter correlations (the identifiability ridge), at O(d²) cost.
    Dense,
}

/// Stan-style windowed warm-up schedule: `(init_buffer, term_buffer,
/// window_ends)`. `init_buffer` sweeps at the front do step-size adaptation only
/// (let the chain reach the typical set); the middle is a series of *expanding*
/// (doubling) windows, each closing at a sweep in `window_ends` where the metric
/// is re-estimated; `term_buffer` sweeps at the end refine the step under the
/// final metric. Returns empty `window_ends` when `n_warmup` is too short to
/// estimate a metric at all (step-size-only warm-up).
fn warmup_schedule(n_warmup: usize) -> (usize, usize, Vec<usize>) {
    if n_warmup < 20 {
        return (n_warmup, 0, Vec::new());
    }
    let (mut init_buffer, mut term_buffer, mut base_window) = (75usize, 50usize, 25usize);
    if init_buffer + base_window + term_buffer > n_warmup {
        // Buffers don't fit: scale to Stan's fallback proportions.
        init_buffer = ((0.15 * n_warmup as f64).ceil() as usize).max(1);
        term_buffer = ((0.10 * n_warmup as f64).ceil() as usize).max(1);
        base_window = n_warmup.saturating_sub(init_buffer + term_buffer).max(1);
    }
    let metric_end = n_warmup - term_buffer;
    let mut ends = Vec::new();
    let mut w_start = init_buffer;
    let mut w = base_window;
    while w_start < metric_end {
        let next = w_start + w;
        // If the *following* (doubled) window would overshoot the metric region,
        // let this window absorb the remainder — matches Stan's last-window rule.
        let this_end = if next + 2 * w > metric_end { metric_end } else { next };
        ends.push(this_end);
        w_start = this_end;
        w *= 2;
    }
    (init_buffer, term_buffer, ends)
}

/// Windowed warm-up adaptation for NUTS (Stan's scheme): interleaves step-size
/// dual averaging with mass-matrix estimation over expanding windows. The metric
/// is re-estimated at each window boundary from *that window's* samples — which
/// are drawn under the previous window's improved metric — so the covariance
/// estimate converges to the true posterior covariance, instead of being frozen
/// once from a poorly-mixed identity-mass phase (the failure mode that leaves a
/// correlated posterior's ridge un-absorbed and caps NUTS's effective sample
/// size). Owns the schedule, the Welford accumulators, and the dual averaging in
/// one place so a forward model (`run_ode_nuts`) and a synthetic test drive the
/// identical adaptation.
pub struct WarmupAdapter {
    d: usize,
    metric: MassMetric,
    init_buffer: usize,
    metric_end: usize,
    window_ends: Vec<usize>,
    target_accept: f64,
    dual_avg: DualAveraging,
    step_size: f64,
    mass: MassMatrix,
    // Per-window Welford moments (reset at each window boundary).
    w_n: f64,
    w_mean: Vec<f64>,
    w_m2: Vec<f64>,
    w_cov: Vec<f64>, // dense only; empty otherwise
}

impl WarmupAdapter {
    pub fn new(
        metric: MassMetric,
        d: usize,
        n_warmup: usize,
        init_step: f64,
        target_accept: f64,
    ) -> Self {
        let (init_buffer, term_buffer, window_ends) = warmup_schedule(n_warmup);
        let dense = matches!(metric, MassMetric::Dense);
        WarmupAdapter {
            d,
            metric,
            init_buffer,
            metric_end: n_warmup.saturating_sub(term_buffer),
            window_ends,
            target_accept,
            dual_avg: DualAveraging::new(init_step, target_accept),
            step_size: init_step,
            mass: MassMatrix::identity(d),
            w_n: 0.0,
            w_mean: vec![0.0; d],
            w_m2: vec![0.0; d],
            w_cov: vec![0.0; if dense { d * d } else { 0 }],
        }
    }

    pub fn step_size(&self) -> f64 {
        self.step_size
    }
    pub fn mass(&self) -> &MassMatrix {
        &self.mass
    }

    /// Diagonal z-standard-deviations of the currently frozen metric (for
    /// logging). Empty if the metric has not been estimated yet.
    pub fn metric_sd(&self) -> Vec<f64> {
        match &self.mass {
            MassMatrix::Diagonal(v) if v.iter().any(|&x| x != 1.0) => {
                v.iter().map(|&x| x.sqrt()).collect()
            }
            MassMatrix::Dense { dim, l_cov } => {
                // diag(Σ) = row-sq-norms of L_cov (Σ = L L^T)
                (0..*dim)
                    .map(|i| (0..=i).map(|j| l_cov[i * dim + j].powi(2)).sum::<f64>().sqrt())
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Call once per warm-up sweep, after the `nuts_step`, with the current
    /// position `z` and the sweep's mean acceptance probability. Advances the step
    /// size every sweep and, at a window boundary, re-estimates the metric.
    /// Returns `true` when the metric was re-frozen this sweep.
    pub fn observe(&mut self, sweep: usize, z: &[f64], accept_prob: f64) -> bool {
        // Step-size dual averaging runs on every warm-up sweep.
        self.step_size = self.dual_avg.update(accept_prob);

        let adapting = !matches!(self.metric, MassMetric::Unit);
        // Accumulate metric statistics only inside the metric windows.
        if adapting && sweep >= self.init_buffer && sweep < self.metric_end {
            self.welford_add(z);
        }
        // At the close of a window, freeze the metric and restart the step search
        // (the geometry just changed, so the previous step is no longer calibrated).
        if adapting && self.window_ends.contains(&(sweep + 1)) {
            let froze = self.freeze_metric();
            self.reset_window();
            let carry = self.dual_avg.final_step_size();
            self.dual_avg = DualAveraging::new(carry, self.target_accept);
            return froze;
        }
        false
    }

    /// Carry the smoothed step size out of warm-up (called once, after the loop).
    pub fn finalize(&mut self) {
        self.step_size = self.dual_avg.final_step_size();
    }

    /// The final adapted metric (consumes the adapter).
    pub fn into_mass(self) -> MassMatrix {
        self.mass
    }

    fn welford_add(&mut self, z: &[f64]) {
        self.w_n += 1.0;
        let old_mean = self.w_mean.clone();
        for i in 0..self.d {
            let delta = z[i] - self.w_mean[i];
            self.w_mean[i] += delta / self.w_n;
            let delta2 = z[i] - self.w_mean[i];
            self.w_m2[i] += delta * delta2;
        }
        if !self.w_cov.is_empty() {
            let d = self.d;
            for i in 0..d {
                for j in 0..d {
                    self.w_cov[i * d + j] += (z[i] - old_mean[i]) * (z[j] - self.w_mean[j]);
                }
            }
        }
    }

    fn freeze_metric(&mut self) -> bool {
        // Need enough samples for a stable estimate; else keep the current metric.
        if self.w_n <= (self.d as f64).max(10.0) {
            return false;
        }
        let d = self.d;
        self.mass = match self.metric {
            MassMetric::Dense => {
                let cov: Vec<f64> = self.w_cov.iter().map(|c| c / (self.w_n - 1.0)).collect();
                MassMatrix::dense_from_covariance(&cov, d)
            }
            MassMetric::Diagonal => {
                let var: Vec<f64> =
                    (0..d).map(|i| (self.w_m2[i] / (self.w_n - 1.0)).max(1e-10)).collect();
                MassMatrix::diagonal(var)
            }
            MassMetric::Unit => return false,
        };
        true
    }

    fn reset_window(&mut self) {
        self.w_n = 0.0;
        self.w_mean.iter_mut().for_each(|x| *x = 0.0);
        self.w_m2.iter_mut().for_each(|x| *x = 0.0);
        self.w_cov.iter_mut().for_each(|x| *x = 0.0);
    }
}

#[cfg(test)]
mod warmup_tests {
    //! The windowed warm-up adapter must, starting from identity mass, converge a
    //! metric good enough for NUTS to draw near-independent samples on a
    //! *correlated* posterior — the case a single-freeze warm-up (frozen once from
    //! a poorly-mixed identity-mass phase) leaves under-conditioned, capping
    //! ESS/iter near ~0.4 (gh#275; garki friction F22/F24). This is the regression
    //! guard for that fix: it fails if the adapter reverts to a bad metric.
    use super::*;
    use crate::rng::StatefulRng;

    fn inv3(m: &[f64]) -> Vec<f64> {
        let (a, b, c, d, e, f, g, h, i) =
            (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        [
            e * i - f * h, c * h - b * i, b * f - c * e,
            f * g - d * i, a * i - c * g, c * d - a * f,
            d * h - e * g, b * g - a * h, a * e - b * d,
        ]
        .iter()
        .map(|x| x / det)
        .collect()
    }
    fn matvec3(m: &[f64], z: &[f64]) -> Vec<f64> {
        (0..3).map(|r| (0..3).map(|c| m[r * 3 + c] * z[c]).sum()).collect()
    }
    /// ESS/iter via Geyer's initial-positive-sequence estimator.
    fn ess_per_iter(x: &[f64]) -> f64 {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / n as f64;
        if var == 0.0 {
            return 0.0;
        }
        let rho = |k: usize| {
            x[..n - k].iter().zip(&x[k..]).map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>()
                / (n as f64 * var)
        };
        let mut s = 1.0;
        let mut k = 1;
        while k < n - 1 {
            let pk = rho(k);
            if pk <= 0.0 {
                break;
            }
            s += 2.0 * pk;
            k += 1;
        }
        (n as f64 / s) / n as f64
    }

    #[test]
    fn windowed_warmup_reaches_high_ess_on_correlated_gaussian() {
        // Target N(0, Σ) with the garki-like correlation ridge
        // (g-r1=+0.62, g-a2=+0.12, r1-a2=-0.48).
        let sigma = [1.0, 0.62, 0.12, 0.62, 1.0, -0.48, 0.12, -0.48, 1.0];
        let sinv = inv3(&sigma);
        let target = |z: &[f64]| -> (f64, Vec<f64>) {
            let sz = matvec3(&sinv, z);
            let lp = -0.5 * z.iter().zip(&sz).map(|(a, b)| a * b).sum::<f64>();
            (lp, sz.iter().map(|v| -v).collect())
        };

        // Windowed warm-up from IDENTITY mass — the adapter must learn Σ itself.
        let mut rng = StatefulRng::new(20260707);
        let mut z = vec![0.0; 3];
        let (mut lp, mut g) = target(&z);
        let mut adapter = WarmupAdapter::new(MassMetric::Dense, 3, 1000, 0.5, 0.8);
        for sweep in 0..1000 {
            let cfg = NUTSConfig {
                max_tree_depth: 10,
                step_size: adapter.step_size(),
                mass_matrix: adapter.mass().clone(),
            };
            let r = nuts_step(&z, lp, &g, &cfg, &target, &mut rng);
            if r.accepted {
                z = r.params;
                lp = r.log_posterior;
                let (_, gg) = target(&z);
                g = gg;
            }
            adapter.observe(sweep, &z, r.mean_accept_prob);
        }
        adapter.finalize();
        let step = adapter.step_size();
        let mass = adapter.into_mass();

        // Sample under the adapted (step, metric).
        let cfg = NUTSConfig { max_tree_depth: 10, step_size: step, mass_matrix: mass };
        let n = 3000usize;
        let mut cols = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        for _ in 0..n {
            let r = nuts_step(&z, lp, &g, &cfg, &target, &mut rng);
            if r.accepted {
                z = r.params;
                lp = r.log_posterior;
                let (_, gg) = target(&z);
                g = gg;
            }
            for j in 0..3 {
                cols[j].push(z[j]);
            }
        }
        let ess: Vec<f64> = (0..3).map(|j| ess_per_iter(&cols[j])).collect();
        eprintln!("windowed-warmup ESS/iter on correlated Gaussian: {ess:?} (step={step:.3})");

        // A single-freeze / mis-estimated metric caps this near ~0.4; a converged
        // metric gives near-independent draws. Assert every dimension clears 0.6 —
        // comfortably above the bad-metric ceiling, comfortably below the ~0.9 a
        // converged dense metric delivers.
        assert!(
            ess.iter().all(|&e| e > 0.6),
            "windowed warm-up should reach ESS/iter > 0.6 on all dims; got {ess:?} \
             — the metric adaptation is not converging to the posterior covariance"
        );
    }
}
