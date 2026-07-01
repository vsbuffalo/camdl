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
pub mod nuts;
pub mod linalg;
pub mod diagnostic;
pub mod prequential;
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
