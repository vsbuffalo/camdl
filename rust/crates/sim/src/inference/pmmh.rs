//! Particle Marginal Metropolis-Hastings (PMMH) — Bayesian posterior
//! sampling via MCMC with particle filter likelihood estimation.
//!
//! Andrieu, Doucet & Holenstein (2010).
//!
//! PMMH wraps a standard Metropolis-Hastings sampler around the bootstrap
//! particle filter: at each MCMC step, propose θ' from a random walk on
//! the transformed scale, run the PF at θ' to get an unbiased estimate
//! of log p(y|θ'), then accept/reject via the MH ratio. The PF estimate
//! is noisy but unbiased, which is sufficient for the MCMC to target the
//! exact posterior (pseudo-marginal property).
//!
//! Key differences from IF2:
//! - No cooling schedule — each PF runs at fixed θ.
//! - No parameter perturbation inside the PF.
//! - Acceptance/rejection step gives valid posterior samples.
//! - Optional adaptive proposal covariance (Haario et al. 2001).

use serde::{Serialize, Deserialize};
use crate::rng::StatefulRng;
use crate::error::SimError;
use super::types::{EstimatedParam, restore_z_values};
use super::if2::Observation;
pub use super::prior::Prior;

// ── Configuration ──────────────────────────────────────────────────

/// PMMH configuration.
pub struct PMMHConfig {
    pub n_steps: usize,
    pub n_particles: usize,
    pub dt: f64,
    /// Initial proposal SD on the transformed scale (diagonal: one per estimated param).
    pub proposal_sd: Vec<f64>,
    /// Enable adaptive Metropolis (Haario et al. 2001).
    pub adapt: bool,
    /// Start adapting after this many steps.
    pub adapt_start: usize,
    /// Record every `thin`-th step.
    pub thin: usize,
    /// Discard first `burn_in` steps from output.
    pub burn_in: usize,
    /// Crank-Nicolson correlation for correlated pseudo-marginal.
    /// None = vanilla PMMH (independent PF evaluations).
    /// Some(0.99) = CPM with ρ=0.99 (recommended).
    pub rho: Option<f64>,
    /// Number of source groups in the model (for sizing binomial noise).
    /// Set by the CLI from model.source_groups.len().
    pub n_source_groups: usize,
}

impl super::traits::InferenceConfig for PMMHConfig {
    fn n_particles(&self) -> usize { self.n_particles }
    fn dt(&self) -> f64 { self.dt }
}

// ── Output types ───────────────────────────────────────────────────

/// One recorded MCMC step.
#[derive(Clone, Debug)]
pub struct PMMHStep {
    pub step: usize,
    /// Parameter values on the natural scale (full param vector).
    pub params: Vec<f64>,
    /// PF log-likelihood estimate.
    pub log_likelihood: f64,
    /// Log prior density.
    pub log_prior: f64,
    pub accepted: bool,
}

/// Result of a PMMH run.
pub struct PMMHResult {
    pub steps: Vec<PMMHStep>,
    pub acceptance_rate: f64,
    pub n_steps: usize,
    /// MAP (highest posterior) parameter set.
    pub map_params: Vec<f64>,
    pub map_loglik: f64,
    pub map_log_posterior: f64,
    /// Resume state for chain continuation. Populated at end of every run.
    pub resume_state: PMMHResumeState,
}

/// Serializable chain state for `--resume`. Saved to `chain_N/resume_state.bin`
/// via bincode at end of every PMMH run, enabling continuation without
/// re-doing burn-in or adaptive proposal warm-up.
#[derive(Clone, Serialize, Deserialize)]
pub struct PMMHResumeState {
    /// Config hash — only resume if the statistical problem matches.
    pub config_hash: String,
    /// Number of MCMC steps completed (resume starts from here).
    pub completed_steps: usize,
    /// Current parameter values (natural scale, full model param vector).
    pub params: Vec<f64>,
    /// Current transformed parameters (unconstrained scale).
    pub transformed: Vec<f64>,
    /// Estimated parameter names (for reordering on resume).
    pub param_names: Vec<String>,
    /// Current PF log-likelihood estimate.
    pub current_ll: f64,
    /// Current log prior density.
    pub current_log_prior: f64,
    /// Number of accepted proposals so far.
    pub n_accepted: usize,
    /// Adaptive proposal state (None if adapt=false).
    pub adaptive: Option<AdaptiveProposal>,
    /// CPM random state (None if rho=None).
    pub current_randoms: Option<super::correlated_pf::PFRandomState>,
    /// MAP parameter values.
    pub map_params: Vec<f64>,
    /// MAP log-likelihood.
    pub map_loglik: f64,
    /// MAP log-posterior.
    pub map_log_posterior: f64,
}

// ── Adaptive proposal: Haario covariance + Robbins–Monro global scaling ──
//
// The proposal is θ' = θ + λ·L·z (z ~ N(0, I)). Two things adapt, and they are
// complementary:
//
//   * L — the covariance *shape* (orientation + relative widths), learned
//     online from the empirical covariance of visited states
//     (Haario, Saksman & Tamminen 2001, "An adaptive Metropolis algorithm",
//     *Bernoulli* 7(2):223–242).
//
//   * λ — a scalar *global scale*, adapted by Robbins–Monro stochastic
//     approximation toward a target acceptance rate (see `adapt_scale`).
//
// Why both: Haario learns the shape from movement, so a chain stuck at 0%
// acceptance (a too-large initial scale — gh#347) has ~zero visited covariance
// and *cannot* recover from the shape term alone. The Robbins–Monro scale
// adapts from the accept/reject signal itself, which is informative even when
// the chain is frozen, so it shrinks λ until proposals land. This is the
// standard robust construction of Andrieu & Thoms (2008, "A tutorial on
// adaptive MCMC", *Stat. Comput.* 18:343–373) and Roberts & Rosenthal (2009,
// "Examples of Adaptive MCMC", *JCGS* 18(2):349–367).

/// Running mean + covariance via Welford's online algorithm, a Cholesky factor
/// for sampling N(0, Σ) (the Haario *shape* term), and a Robbins–Monro global
/// scale (the `log_scale` term — see [`AdaptiveProposal::adapt_scale`]).
#[derive(Clone, Serialize, Deserialize)]
pub struct AdaptiveProposal {
    d: usize,
    n: usize,
    mean: Vec<f64>,
    /// Sum of outer products of deviations: M₂[i*d+j]
    m2: Vec<f64>,
    /// Cholesky factor L such that L Lᵀ = scaled covariance.
    /// Row-major, lower-triangular. Updated every `chol_interval` steps.
    chol: Vec<f64>,
    chol_interval: usize,
    steps_since_chol: usize,
    /// Whether Cholesky has been computed at least once.
    chol_valid: bool,
    /// Log of the Robbins–Monro global scale λ = exp(log_scale). Starts at 0
    /// (λ = 1) and is driven toward the target acceptance by [`adapt_scale`].
    /// `serde(default)` = 0.0 so a checkpoint written before this field
    /// resumes at the neutral scale.
    #[serde(default)]
    log_scale: f64,
}

impl AdaptiveProposal {
    fn new(d: usize) -> Self {
        AdaptiveProposal {
            d,
            n: 0,
            mean: vec![0.0; d],
            m2: vec![0.0; d * d],
            chol: vec![0.0; d * d],
            chol_interval: 100,
            steps_since_chol: 0,
            chol_valid: false,
            log_scale: 0.0,
        }
    }

    /// The Robbins–Monro global scale λ = exp(log_scale) applied to every
    /// perturbation. 1.0 until [`adapt_scale`](Self::adapt_scale) moves it.
    fn scale(&self) -> f64 {
        self.log_scale.exp()
    }

    /// Target acceptance rate for the scale adaptation. The asymptotically
    /// optimal random-walk-Metropolis acceptance is 0.234 as d → ∞ (Roberts,
    /// Gelman & Gilks 1997, "Weak convergence and optimal scaling of random
    /// walk Metropolis algorithms", *Ann. Appl. Probab.* 7(1):110–120), rising
    /// to ≈0.44 for d = 1. Interpolate `0.234 + 0.206/d` so a 1–2 parameter fit
    /// targets the higher low-dimensional optimum.
    fn target_accept(&self) -> f64 {
        0.234 + 0.206 / self.d.max(1) as f64
    }

    /// Robbins–Monro update of the global scale from one accept/reject outcome.
    ///
    /// This is stochastic approximation (Robbins & Monro 1951, "A Stochastic
    /// Approximation Method", *Ann. Math. Statist.* 22(3):400–407) applied to
    /// adaptive MCMC. We want the scale λ whose expected acceptance equals the
    /// target `a*` — i.e. the root of `g(λ) = E[accept | λ] − a* = 0`. `a(λ)` is
    /// not available in closed form, but each step's `accepted` indicator is a
    /// noisy one-sample estimate of it, which is exactly what Robbins–Monro
    /// consumes. On the log scale (to keep λ > 0):
    ///
    /// ```text
    ///   log λ ← log λ + γ_t · (accepted − a*)
    /// ```
    ///
    /// Accept above target → λ grows (bolder steps); below → λ shrinks. The gain
    /// `γ_t = (t+1)^(−κ)`, κ = 0.6 ∈ (½, 1], satisfies the Robbins–Monro
    /// conditions Σγ_t = ∞ (λ can travel from any start to the root) and
    /// Σγ_t² < ∞ (the measurement noise is summable, so it converges). The
    /// vanishing gain is also the *diminishing adaptation* that keeps the
    /// adaptive chain ergodic (Roberts & Rosenthal 2007, "Coupling and
    /// ergodicity of adaptive MCMC", *J. Appl. Probab.* 44:458–475).
    ///
    /// gh#347: a chain stuck at 0% acceptance has `(accepted − a*) = −a* < 0`
    /// every step, so log λ decreases monotonically — the scale shrinks with no
    /// dependence on the chain ever moving, breaking the deadlock the Haario
    /// covariance term cannot (it needs movement to learn a shape).
    fn adapt_scale(&mut self, accepted: bool, step: usize) {
        let gamma = ((step + 1) as f64).powf(-0.6);
        let accept_indicator = if accepted { 1.0 } else { 0.0 };
        self.log_scale += gamma * (accept_indicator - self.target_accept());
    }

    /// Update running statistics with a new sample (on transformed scale).
    fn update(&mut self, x: &[f64]) {
        self.n += 1;
        let n = self.n as f64;
        let d = self.d;

        // Welford: delta = x - old_mean, update mean, delta2 = x - new_mean
        let mut delta = vec![0.0; d];
        for i in 0..d {
            delta[i] = x[i] - self.mean[i];
            self.mean[i] += delta[i] / n;
        }
        for i in 0..d {
            let delta2_i = x[i] - self.mean[i];
            for j in 0..d {
                self.m2[i * d + j] += delta[i] * (x[j] - self.mean[j]);
                // Symmetrize using the Welford identity for cross-terms:
                // We need M₂[i,j] = Σ (x_k - mean)(x_k - mean)ᵀ
                // The update above handles the rank-1 correction correctly
                // for both i,j simultaneously.
                let _ = delta2_i; // used implicitly via the outer product
            }
        }

        self.steps_since_chol += 1;
        if self.steps_since_chol >= self.chol_interval && self.n > self.d {
            self.update_cholesky();
        }
    }

    /// Recompute Cholesky of the scaled proposal covariance:
    /// Σ_prop = (2.38² / d) × Cov + ε × I
    fn update_cholesky(&mut self) {
        let d = self.d;
        let n = self.n as f64;
        let scale = 2.38_f64.powi(2) / d as f64;
        let eps = 1e-6;

        // Build scaled covariance + regularization
        let mut a = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                a[i * d + j] = scale * self.m2[i * d + j] / (n - 1.0);
            }
            a[i * d + i] += eps;
        }

        // Cholesky decomposition
        if let Some(l) = super::linalg::cholesky_lower(&a, d) {
            self.chol.copy_from_slice(&l);
            self.chol_valid = true;
        }
        // If Cholesky fails (shouldn't with eps regularization), keep old factor.

        self.steps_since_chol = 0;
    }

    /// Sample a proposal perturbation: Δθ ~ N(0, Σ_prop).
    /// Returns the perturbation vector (on transformed scale).
    /// Falls back to diagonal if Cholesky isn't ready yet.
    fn sample_perturbation(&self, rng: &mut StatefulRng, fallback_sd: &[f64]) -> Vec<f64> {
        let d = self.d;
        let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();

        if self.chol_valid {
            // Δ = L × z
            let mut delta = vec![0.0; d];
            for i in 0..d {
                for j in 0..=i {
                    delta[i] += self.chol[i * d + j] * z[j];
                }
            }
            delta
        } else {
            // Diagonal fallback
            z.iter().zip(fallback_sd).map(|(&zi, &sd)| zi * sd).collect()
        }
    }
}

#[cfg(test)]
mod adaptive_scale_tests {
    use super::AdaptiveProposal;

    /// gh#347: the Robbins–Monro global scale must SHRINK under a persistent
    /// 0%-acceptance deadlock — this is the property that rescues a chain stuck
    /// at a too-large initial proposal, which the Haario covariance term alone
    /// (learned only from *movement*) cannot. Each all-reject step contributes
    /// `γ_t·(0 − a*) < 0`, so log_scale is monotonically non-increasing and λ
    /// collapses toward 0.
    #[test]
    fn rm_scale_shrinks_under_zero_acceptance_deadlock() {
        let mut ap = AdaptiveProposal::new(2);
        assert_eq!(ap.scale(), 1.0, "λ starts at 1");
        let mut prev = ap.log_scale;
        for step in 0..500 {
            ap.adapt_scale(false, step);
            assert!(ap.log_scale <= prev, "log_scale must be non-increasing under all-reject");
            prev = ap.log_scale;
        }
        assert!(ap.scale() < 0.05,
            "λ must collapse under a 0%-acceptance deadlock, got {}", ap.scale());
    }

    /// The mirror: accepting far above target means the steps are too timid, so
    /// λ must GROW to explore faster.
    #[test]
    fn rm_scale_grows_when_accepting_above_target() {
        let mut ap = AdaptiveProposal::new(2);
        for step in 0..200 {
            ap.adapt_scale(true, step);
        }
        assert!(ap.scale() > 1.0, "λ must grow when accepting above target, got {}", ap.scale());
    }

    /// Target acceptance is dimension-aware: 0.234 as d → ∞ (Roberts, Gelman &
    /// Gilks 1997), ≈0.44 for d = 1.
    #[test]
    fn target_accept_is_dimension_aware() {
        assert!((AdaptiveProposal::new(1).target_accept() - 0.44).abs() < 1e-9);
        assert!(AdaptiveProposal::new(2).target_accept() > 0.30);
        assert!(AdaptiveProposal::new(1000).target_accept() < 0.24);
    }
}

// ── Core PMMH algorithm ────────────────────────────────────────────

/// Run PMMH.
///
/// Unlike PF/IF2/PGAS, PMMH intentionally uses a closure-based API rather
/// than `ProcessModel`/`ObservationModel` traits. This is the right design:
/// PMMH wraps a Metropolis-Hastings loop around a black-box likelihood
/// estimator. It doesn't need to know how the PF is constructed — only that
/// `eval_loglik(params, seed) -> Result<log L̂(θ)>` returns an unbiased estimate.
/// This decoupling means PMMH works with any likelihood estimator (vanilla
/// PF, correlated PF, importance sampling, etc.) without code changes.
///
/// `eval_loglik` runs a particle filter at the given params and returns
/// `Ok(log L̂(θ))` — where `Ok(-∞)` is a legitimate "θ ruled out" result the
/// MH ratio rejects — or `Err(structural)` for a non-recoverable model failure
/// the caller must surface rather than mistake for a ruled-out θ (gh#224). The
/// closure owns the recoverable-vs-structural classification
/// (`SimError::is_per_particle_recoverable`); `run_pmmh` only propagates.
/// Built in the CLI layer from `run_quick_pfilter`. Takes `(full_params, pf_seed)`.
///
/// `on_step` optional progress callback: `(step, loglik, accepted)`.
/// Correlated PF evaluator for CPM-MCMC.
/// Takes (params, randoms) → `Ok(loglik)` / `Err(structural)`, same contract.
pub type CorrelatedEvalFn<'a> = dyn Fn(&[f64], &super::correlated_pf::PFRandomState)
    -> Result<f64, SimError> + 'a;

// `param_names`: full parameter names, parallel to `base_params` (positional,
// not subset to estimated params). Used to build the name → value env for
// hierarchical prior resolution. Wave 2 / #3 Gate 3a. When no hierarchical
// priors are in `priors`, this slice can be empty — no lookups happen.
pub fn run_pmmh(
    if2_params: &[EstimatedParam],
    priors: &[Prior],
    base_params: &[f64],
    param_names: &[String],
    config: &PMMHConfig,
    observations: &[Observation],
    eval_loglik: &dyn Fn(&[f64], u64) -> Result<f64, SimError>,
    eval_loglik_correlated: Option<&CorrelatedEvalFn>,
    seed: u64,
    on_step: Option<&dyn Fn(usize, f64, bool, &[f64])>,
    resume_from: Option<PMMHResumeState>,
    config_hash: String,
) -> Result<PMMHResult, SimError> {
    let d = if2_params.len();
    assert_eq!(d, priors.len(), "priors must match if2_params length");
    assert_eq!(d, config.proposal_sd.len(), "proposal_sd must match if2_params length");
    // Whether any prior is hierarchical — determines if we need to
    // build an env. Zero-cost when the model has no hierarchical
    // priors (common case).
    let has_hierarchical = priors.iter().any(|p| matches!(p, Prior::Hierarchical(_)));
    if has_hierarchical {
        assert_eq!(param_names.len(), base_params.len(),
            "param_names must be aligned with base_params for hierarchical priors");
    }

    use super::correlated_pf::PFRandomState;

    // CPM sizing: derived from observation times and dt rather than config fields.
    let n_obs = observations.len();
    let steps_per_obs = if observations.len() >= 2 {
        crate::time::interval_steps(observations[0].time, observations[1].time, config.dt)
    } else {
        1
    };

    let start_step;
    let mut current_params: Vec<f64>;
    let mut current_transformed: Vec<f64>;
    let mut current_ll: f64;
    let mut current_log_prior: f64;
    let mut current_randoms: Option<PFRandomState>;
    let mut map_log_posterior: f64;
    let mut map_params: Vec<f64>;
    let mut map_loglik: f64;
    let mut adaptive: Option<AdaptiveProposal>;
    let mut n_accepted: usize;
    let mut n_accepted_post_burn: usize = 0;  // gh#audit-M4
    let mut rng: StatefulRng;

    if let Some(state) = resume_from {
        eprintln!("  resuming from step {}...", state.completed_steps);
        start_step = state.completed_steps;
        current_params = state.params;
        n_accepted = state.n_accepted;
        current_ll = state.current_ll;
        current_log_prior = state.current_log_prior;
        current_randoms = state.current_randoms;
        adaptive = state.adaptive;
        map_params = state.map_params;
        map_loglik = state.map_loglik;
        map_log_posterior = state.map_log_posterior;

        current_transformed = restore_z_values(
            &state.param_names, &state.transformed, if2_params, &current_params,
        );

        // Enforce bounds on restored params
        for (i, spec) in if2_params.iter().enumerate() {
            let clamped = spec.from_transformed(current_transformed[i]);
            current_params[spec.index] = clamped;
        }

        // Derive RNG from seed ^ completed_steps for continuation
        rng = StatefulRng::new(seed ^ start_step as u64);
    } else {
        start_step = 0;
        rng = StatefulRng::new(seed);
        current_params = base_params.to_vec();

        // Current state on transformed scale
        current_transformed = if2_params.iter()
            .map(|p| p.to_transformed(current_params[p.index]))
            .collect();

        // CPM random state (if correlated mode)
        current_randoms = config.rho.map(|_| {
            PFRandomState::draw_fresh(
                config.n_particles, n_obs, steps_per_obs,
                config.n_source_groups, &mut rng,
            )
        });

        // Initial PF evaluation
        current_ll = if let (Some(ref randoms), Some(eval_corr)) =
            (&current_randoms, &eval_loglik_correlated)
        {
            eval_corr(&current_params, randoms)?
        } else {
            eval_loglik(&current_params, seed.wrapping_add(0))?
        };
        {
            let env = crate::inference::hierarchical::NamedParams {
                names: param_names,
                values: &current_params,
            };
            current_log_prior = if2_params.iter().zip(priors.iter())
                .zip(current_transformed.iter())
                .map(|((p, prior), &z)| prior.log_density_env(current_params[p.index], z, &env))
                .sum();
        }

        // Track MAP
        map_log_posterior = current_ll + current_log_prior;
        map_params = current_params.clone();
        map_loglik = current_ll;

        // Adaptive proposal
        adaptive = if config.adapt {
            let mut ap = AdaptiveProposal::new(d);
            ap.update(&current_transformed);
            Some(ap)
        } else {
            None
        };

        n_accepted = 0;
    }

    let mut current_log_jacobian: f64 = if2_params.iter()
        .zip(current_transformed.iter())
        .map(|(p, &z)| p.log_jacobian(z))
        .sum();

    let mut steps = Vec::new();

    if start_step >= config.n_steps {
        eprintln!("  warning: chain already completed {} steps (requested {}). \
                   Increase steps in fit.toml to continue.", start_step, config.n_steps);
    }

    for step in start_step..config.n_steps {
        // Propose: θ' = θ + λ·Δ on transformed scale. Δ is the shape term
        // (Haario Cholesky once learned, else the diagonal proposal_sd); λ is
        // the Robbins–Monro global scale. gh#347: λ scales BOTH branches and
        // adapts from step 0 (gated on `adapt`, not `adapt_start` — which gates
        // only the covariance shape), so it can break a 0%-acceptance deadlock
        // during the pre-`adapt_start` phase where the shape term is still the
        // fixed, possibly-too-large diagonal.
        let mut delta = if let Some(ref ap) = adaptive {
            if config.adapt && step >= config.adapt_start {
                ap.sample_perturbation(&mut rng, &config.proposal_sd)
            } else {
                (0..d).map(|i| rng.normal() * config.proposal_sd[i]).collect()
            }
        } else {
            (0..d).map(|i| rng.normal() * config.proposal_sd[i]).collect::<Vec<f64>>()
        };
        if config.adapt {
            if let Some(ref ap) = adaptive {
                let lambda = ap.scale();
                for dz in delta.iter_mut() {
                    *dz *= lambda;
                }
            }
        }

        let proposed_transformed: Vec<f64> = current_transformed.iter()
            .zip(delta.iter())
            .map(|(&z, &dz)| z + dz)
            .collect();

        // Back to natural scale
        let mut proposed_params = current_params.clone();
        for (i, spec) in if2_params.iter().enumerate() {
            proposed_params[spec.index] = spec.from_transformed(proposed_transformed[i]);
        }

        // Log prior at proposed params (env-aware for hierarchical priors;
        // plain variants ignore the env and return the same value as before).
        let proposed_log_prior: f64 = {
            let env = crate::inference::hierarchical::NamedParams {
                names: param_names,
                values: &proposed_params,
            };
            if2_params.iter().zip(priors.iter())
                .zip(proposed_transformed.iter())
                .map(|((p, prior), &z)| prior.log_density_env(proposed_params[p.index], z, &env))
                .sum()
        };

        // Evaluate PF: correlated or independent
        let proposed_randoms: Option<PFRandomState>;
        let proposed_ll;
        if let (Some(rho), Some(ref cur_rand), Some(eval_corr)) =
            (config.rho, &current_randoms, &eval_loglik_correlated)
        {
            let pr = cur_rand.correlate(rho, &mut rng);
            proposed_ll = eval_corr(&proposed_params, &pr)?;
            proposed_randoms = Some(pr);
        } else {
            let pf_seed = seed.wrapping_add(step as u64 + 1);
            proposed_ll = eval_loglik(&proposed_params, pf_seed)?;
            proposed_randoms = None;
        };

        // Jacobian correction
        let proposed_log_jacobian: f64 = if2_params.iter()
            .zip(proposed_transformed.iter())
            .map(|(p, &z)| p.log_jacobian(z))
            .sum();

        // MH acceptance ratio (log scale)
        let log_alpha = (proposed_ll + proposed_log_prior + proposed_log_jacobian)
                      - (current_ll + current_log_prior + current_log_jacobian);

        let accepted = log_alpha.is_finite() && rng.uniform().ln() < log_alpha;

        // gh#347: Robbins–Monro update of the global proposal scale from this
        // accept/reject outcome. Runs from step 0 (independent of `adapt_start`)
        // so it can shrink an over-large initial scale before the covariance
        // shape adaptation begins. Diminishing gain → diminishing adaptation.
        if config.adapt {
            if let Some(ref mut ap) = adaptive {
                ap.adapt_scale(accepted, step);
            }
        }

        if accepted {
            current_params.copy_from_slice(&proposed_params);
            current_transformed.copy_from_slice(&proposed_transformed);
            current_ll = proposed_ll;
            current_log_prior = proposed_log_prior;
            current_log_jacobian = proposed_log_jacobian;
            if proposed_randoms.is_some() {
                current_randoms = proposed_randoms;
            }
            n_accepted += 1;
            // gh#audit-M4. Track post-burn-in acceptances separately
            // so the reported acceptance_rate excludes the warm-up
            // exploration phase. Bare n_accepted / n_steps reports
            // the wrong rate when burn_in is large.
            if step >= config.burn_in {
                n_accepted_post_burn += 1;
            }

            let log_posterior = current_ll + current_log_prior;
            if log_posterior > map_log_posterior {
                map_log_posterior = log_posterior;
                map_params.copy_from_slice(&current_params);
                map_loglik = current_ll;
            }
        }

        // Update adaptive proposal with current position (whether accepted or not
        // is debated — we include all steps, matching the original Haario algorithm)
        if let Some(ref mut ap) = adaptive {
            ap.update(&current_transformed);
        }

        // Record step (respecting burn-in and thinning)
        if step >= config.burn_in && (step - config.burn_in).is_multiple_of(config.thin) {
            steps.push(PMMHStep {
                step,
                params: current_params.clone(),
                log_likelihood: current_ll,
                log_prior: current_log_prior,
                accepted,
            });
        }

        if let Some(cb) = on_step {
            cb(step, current_ll, accepted, &current_params);
        }
    }

    // gh#audit-M4. acceptance_rate now divides by post-burn-in step
    // count so it reports the rate over the *sampling* phase, not
    // the exploration phase. The struct field name is unchanged so
    // downstream readers don't need to migrate, but the semantics
    // are now Stan-canonical (post-warmup acceptance).
    let post_burn_steps = config.n_steps.saturating_sub(config.burn_in);
    let acceptance_rate = if post_burn_steps > 0 {
        n_accepted_post_burn as f64 / post_burn_steps as f64
    } else {
        0.0
    };

    let resume_state = PMMHResumeState {
        config_hash,
        completed_steps: config.n_steps,
        params: current_params.clone(),
        transformed: current_transformed,
        param_names: if2_params.iter().map(|p| p.name.clone()).collect(),
        current_ll,
        current_log_prior,
        n_accepted,
        adaptive,
        current_randoms,
        map_params: map_params.clone(),
        map_loglik,
        map_log_posterior,
    };

    // gh#347: the recorded `map_loglik` is the SINGLE likelihood estimate at the
    // accepted MAP θ. For a particle-filter likelihood that estimate is the
    // MAXIMUM over many noisy per-step estimates, so it is biased UPWARD — and
    // the bias grows with better mixing (more distinct θ visited → a higher max
    // of noisy draws ≈ σ·√(2 ln n) above the truth). Report an honest
    // re-evaluation at the MAP θ instead: the mean of a few independent
    // estimates. A deterministic (ODE) likelihood is exact in one eval, so its
    // mean reproduces the recorded value; only a stochastic (PF) likelihood is
    // corrected. This makes the reported best-loglik a faithful estimate of
    // L(θ̂), independent of how well the chain happened to mix. The RESUME state
    // keeps the recorded value (its MAP tracking must stay on the same scale it
    // was accumulated on).
    const MAP_REEVAL_REPS: usize = 8;
    let honest_map_loglik = {
        let mut samples = Vec::with_capacity(MAP_REEVAL_REPS);
        for r in 0..MAP_REEVAL_REPS {
            if let Ok(ll) = eval_loglik(&map_params, seed.wrapping_add(0x5347_0000 + r as u64)) {
                if ll.is_finite() {
                    samples.push(ll);
                }
            }
        }
        if samples.is_empty() {
            map_loglik // every re-eval failed (ruled-out θ): keep the recorded value
        } else {
            samples.iter().sum::<f64>() / samples.len() as f64
        }
    };
    // Keep posterior = loglik + prior consistent (the prior part is unchanged).
    let honest_map_log_posterior = honest_map_loglik + (map_log_posterior - map_loglik);

    Ok(PMMHResult {
        steps,
        acceptance_rate,
        n_steps: config.n_steps,
        map_params,
        map_loglik: honest_map_loglik,
        map_log_posterior: honest_map_log_posterior,
        resume_state,
    })
}

// ── MCMC diagnostics ───────────────────────────────────────────────

/// Effective sample size via Geyer (1992) initial-positive-sequence
/// estimator with **pair-sum** truncation — the canonical variant
/// used by Stan, PyMC, and BDA3.
///
/// IM11 in 2026-04-19 inference review batch 3: the previous
/// implementation truncated on the first negative single-lag
/// autocorrelation, which overestimates ESS by 2–5× on chains
/// with non-monotonic autocorrelation (NUTS during tuning, PMMH
/// near a mode boundary). Pair-sum is strictly more conservative:
/// it stops when `ρ_{2k} + ρ_{2k+1} < 0`.
///
/// Formula:
///   ESS = n / (1 + 2 · Σ_{k=1}^{K} ρ_k)
/// where K is the largest odd index such that every consecutive
/// pair sum ρ_{2j} + ρ_{2j+1} for j = 1, …, (K-1)/2 is
/// non-negative.
pub fn mcmc_ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 { return n as f64; }

    let mean = chain.iter().sum::<f64>() / n as f64;
    let var = chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-30 { return 1.0; }

    let rho_at = |lag: usize| -> f64 {
        chain.iter().zip(chain.iter().skip(lag))
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>() / (n as f64 * var)
    };

    // Accumulate pair sums ρ_{2k−1} + ρ_{2k} for k = 1, 2, …
    // Stop at the first non-positive pair sum. Include both lags
    // that make up the positive pair in sum_rho.
    let mut sum_rho = 0.0;
    let mut k = 1;
    while 2 * k < n {
        let rho_a = rho_at(2 * k - 1);
        let rho_b = rho_at(2 * k);
        if rho_a + rho_b <= 0.0 {
            break;
        }
        sum_rho += rho_a + rho_b;
        k += 1;
    }
    (n as f64 / (1.0 + 2.0 * sum_rho)).max(1.0)
}
