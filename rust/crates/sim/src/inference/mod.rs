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
pub mod pmmh;
pub mod correlated_pf;
pub mod pgas;
pub mod pgas_grad;
pub mod nuts;
pub mod linalg;
pub mod diagnostic;
pub mod prequential;

// Re-exports
pub use types::{ParticleState, ParticleSwarm};
pub use obs_loglik::{negbin_logpmf, normal_logpdf, discretized_normal_logpmf, normal_cdf};
pub use particle_filter::bootstrap_filter;
pub use traits::{ProcessModel, DensityProcess, ObservationModel, Resettable, SMCConfig};
pub use chain_binomial_process::ChainBinomialProcess;
pub use multi_stream_obs::{MultiStreamObsModel, NullObsModel};
pub use prior::Prior;
