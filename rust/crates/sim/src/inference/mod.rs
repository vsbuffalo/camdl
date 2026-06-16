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
//!   PGAS              — Particle Gibbs with Ancestor Sampling (Bayesian)
//!   PMMH              — Particle Marginal Metropolis-Hastings (experimental)

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
