//! Inference module: particle filter, IF2, PMMH, PGAS.
//!
//! All inference algorithms program against three core traits:
//!   `ProcessModel`      — advance state one dt (any simulation backend)
//!   `DensityProcess`    — extends ProcessModel with transition density (PGAS)
//!   `ObservationModel`  — log p(y | state, θ) for particle weighting
//!
//! Architecture:
//!   ParticleFilter    — bootstrap filter using ProcessModel + ObservationModel
//!   IF2               — iterated filtering (MLE via perturbed PF)
//!   PGAS              — Particle Gibbs with Ancestor Sampling (default Bayesian path)
//!   PMMH              — Particle Marginal Metropolis-Hastings (Bayesian; prefer PGAS for long series)

pub mod numerics;  // gh#audit-H3
pub mod degeneracy;  // gh#110
pub mod traits;
pub mod obs_loglik;
pub mod resampling;
pub mod particle_filter;
pub mod ancestor_trace;
pub mod if2;
pub mod types;
pub mod obs_model;
pub mod multi_stream_obs;
pub mod chain_binomial_process;
pub mod prior;
pub mod hierarchical;
pub mod pmmh;
pub mod correlated_pf;
pub mod pgas;
pub mod pgas_grad;
pub mod state_transition; // state-space kernel spike (2026-09-02 note)
pub mod state_pgbs; // csmc_bs — the experimental backward kernel (same spike)
pub mod pgas_init;  // gh#784 — X₀ from an unconditional SMC pass
pub mod nuts;
pub mod linalg;
pub mod diagnostic;
pub mod convergence;  // gh#84 — Vehtari et al. 2021 rank-normalized R̂ / ESS
pub mod prequential;
pub mod ode_loglik;
pub mod ode_grad;
pub mod gradient_capability;
pub mod ode_nuts;
#[cfg(feature = "ode")]
pub mod deterministic;

// Re-exports
pub use types::{ParticleState, ParticleSwarm};
pub use obs_loglik::{negbin_logpmf, normal_logpdf, discretized_normal_logpmf, normal_cdf, normal_quantile};
pub use particle_filter::bootstrap_filter;
pub use traits::{ProcessModel, DensityProcess, ObservationModel, Resettable, SMCConfig, InferenceConfig};
pub use chain_binomial_process::ChainBinomialProcess;
pub use multi_stream_obs::{
    dense_cells, BindReport, BoundObs, Finding, MultiStreamObsModel, NullObsModel, ObsCell,
    Severity,
};
pub use prior::Prior;
pub use ode_loglik::compute_ode_loglik;

/// gh#226. The absorbing-`-inf` backstop predicate: `true` when a
/// log-likelihood value is not a finite anchor a fit can rest on.
///
/// A fit is *degenerate* when NOT ONE surviving chain reached a finite
/// log-likelihood. Callers pass the best (largest) loglik a chain
/// reached — the MAP loglik for PMMH / MH, the clean-eval winner loglik
/// for IF2, the best complete-data sweep for PGAS. When that best is
/// non-finite the Metropolis ratio is stuck (`-inf - (-inf) = NaN`,
/// which never accepts) or the whole reachable surface is `-inf`; the
/// run would otherwise complete with a degenerate posterior and exit 0
/// (gh#226).
///
/// This is the load-bearing half of the condition
/// `acceptance_rate == 0.0 && !best_loglik.is_finite()`: `acceptance_rate
/// == 0` is implied for MH samplers (a finite `best_loglik` can only be
/// reached by evaluating a finite loglik, which starts / moves the MAP),
/// and IF2 has no MH acceptance at all — so `best_loglik` is the one
/// quantity every path shares. Because a driver passes the GLOBAL best
/// (max across surviving chains), `no_finite_anchor(global_best)` is
/// exactly "every surviving chain has no finite anchor"; a single finite
/// chain (e.g. mixed inits, some ruled out) makes the global best finite
/// and the fit proceeds — the false-positive guard the backstop must not
/// trip on legitimate fits.
#[inline]
pub fn no_finite_anchor(best_loglik: f64) -> bool {
    !best_loglik.is_finite()
}

/// The Metropolis accept decision from the log acceptance ratio `log_alpha` and
/// the log of a Uniform(0,1) draw `u_ln` (both on the log scale).
///
/// gh#334: the raw comparison `u_ln < log_alpha` is the correct MH decision for
/// EVERY case by IEEE-754 semantics — where a `log_alpha.is_finite()` guard was
/// not. A chain at a legitimately −∞ start (a θ predicting extinction) that
/// proposes a finite θ has `log_alpha = finite − (−∞) = +∞`; its true accept
/// probability is min(1, e^{+∞}) = 1, so it must accept (the escape). `u_ln <
/// +∞` is always true → accept. Moving TO −∞ gives `log_alpha = −∞` → `u_ln <
/// −∞` is false → reject. Both −∞ gives `log_alpha = NaN` → any comparison with
/// NaN is false → reject (no information). The old `is_finite()` guard uniquely
/// mis-rejected the `+∞` escape, so a −∞-init chain could never move.
#[inline]
pub fn mh_accept(log_alpha: f64, u_ln: f64) -> bool {
    u_ln < log_alpha
}

// Lives here, beside [`no_finite_anchor`], rather than in `pmmh` — the two are
// siblings: one-line cross-algorithm predicates about how the inference stack
// treats a non-finite log-density, from gh#334 and gh#226 respectively. It was
// briefly private to `pmmh`, which meant `pgas` had to reach through that
// module's namespace for something `pmmh` does not own (gh#471).

#[cfg(test)]
mod mh_accept_tests {
    use super::mh_accept;

    /// gh#334: a chain sitting at a legitimately −∞ start (a θ predicting
    /// epidemic extinction) must be able to ESCAPE to a finite proposal. That
    /// move has `log_alpha = finite − (−∞) = +∞`, whose true Metropolis accept
    /// probability is min(1, e^{+∞}) = 1, so it must ALWAYS accept. The
    /// `log_alpha.is_finite()` guard rejected exactly this move, trapping a
    /// −∞-init chain forever.
    #[test]
    fn escapes_from_neg_inf_start_to_finite() {
        assert!(mh_accept(f64::INFINITY, (0.5f64).ln()), "must accept the +∞ escape");
        assert!(mh_accept(f64::INFINITY, f64::NEG_INFINITY), "must accept the escape even at u→0");
    }

    /// The other non-finite cases stay correct by IEEE comparison semantics:
    /// moving TO −∞ (`log_alpha = −∞`) rejects; both −∞ (`log_alpha = NaN`)
    /// rejects (no information to move on).
    #[test]
    fn rejects_move_to_neg_inf_and_nan() {
        assert!(!mh_accept(f64::NEG_INFINITY, (0.5f64).ln()), "must not move to −∞");
        assert!(!mh_accept(f64::NAN, (0.5f64).ln()), "NaN ratio must reject");
    }

    #[test]
    fn finite_ratio_is_standard_metropolis() {
        assert!(mh_accept(0.5, -1.0), "ln(u)=−1 < 0.5 accepts");
        assert!(!mh_accept(-2.0, -1.0), "ln(u)=−1 is not < −2.0");
    }
}
