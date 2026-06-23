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
