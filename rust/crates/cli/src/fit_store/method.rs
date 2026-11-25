//! A fit run's method identity: the algorithm and every `[method]` setting it
//! reads, typed, one variant per `config_v2::Algorithm`.
//!
//! The method's length (PGAS `sweeps`, PMMH and MH `iterations`) is not here:
//! it is [`super::FitRunIdentity::length`], so a resumed run and its base share
//! a method. Nor is `starts`, which is [`super::StartSource`]. A run's method
//! group folder is hashed from this value, so two runs whose methods differ in
//! any field sit in two groups.

use std::collections::BTreeMap;

use sim::inference::types::{If2Step, PerturbAt, Perturbation, ZSd};
use store::Finite;

use crate::fit::steps::ResolvedSteps;

use crate::fit::config_v2::{self, Method};
use crate::pfilter_store::Binomial;

/// How a run moves through its problem.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method", version = 2)]
pub(crate) enum MethodIdentity {
    If2(If2Identity),
    Pgas(PgasIdentity),
    Pmmh(PmmhIdentity),
    Mh(MhIdentity),
    Pfilter(PfilterMethodIdentity),
    NlSbplx(NloptIdentity),
    NlBobyqa(NloptIdentity),
    NlLbfgs(NloptIdentity),
}

/// The simulation backend a method runs on.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "fit_backend", version = 1)]
pub(crate) enum Backend {
    ChainBinomial,
    Ode,
}

impl From<crate::run_meta::InferenceBackend> for Backend {
    fn from(b: crate::run_meta::InferenceBackend) -> Self {
        match b {
            crate::run_meta::InferenceBackend::ChainBinomial => Backend::ChainBinomial,
            crate::run_meta::InferenceBackend::Ode => Backend::Ode,
        }
    }
}

/// How replicate log-likelihoods are combined into one score.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "fit_combine", version = 1)]
pub(crate) enum Combine {
    LogMeanExp,
    Mean,
}

impl From<config_v2::CombineMode> for Combine {
    fn from(c: config_v2::CombineMode) -> Self {
        match c {
            config_v2::CombineMode::LogMeanExp => Combine::LogMeanExp,
            config_v2::CombineMode::Mean => Combine::Mean,
        }
    }
}

/// IF2's re-scoring of its candidate points with a larger filter, which sets
/// the log-likelihood the run stores.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_loglik_eval", version = 1)]
pub(crate) struct LoglikEvalIdentity {
    pub n_particles: u64,
    pub n_replicates: u64,
    pub combine: Combine,
}

/// The convergence gate's thresholds, which set the verdict a run stores.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_gate", version = 1)]
pub(crate) struct GateIdentity {
    pub a_thresh: Finite,
    pub decibans_thresh: Finite,
}

/// The post-fit dt-convergence check, whose result a run stores. A `None`
/// field takes its value from the method's clean evaluation or the backend
/// when the check runs.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_dt_check", version = 1)]
pub(crate) struct DtCheckIdentity {
    pub enabled: bool,
    pub n_halvings: u64,
    pub n_particles: Option<u64>,
    pub n_replicates: Option<u64>,
    pub threshold_nats: Option<Finite>,
    pub combine: Option<Combine>,
}

/// An IF2 method. IF2 cannot be extended: `iterations` sets its cooling
/// schedule, so it is a setting, not a length.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_if2", version = 1)]
pub(crate) struct If2Identity {
    pub backend: Backend,
    pub chains: u64,
    pub particles: u64,
    pub iterations: u64,
    /// The fraction of the initial perturbation left at `cooling_target_iters`.
    pub cooling: Finite,
    pub cooling_target_iters: u64,
    pub loglik_eval: LoglikEvalIdentity,
    pub gate: GateIdentity,
    pub dt_check: DtCheckIdentity,
    pub binomial: Binomial,
    /// Every estimated parameter's resolved perturbation, by name; a simplex
    /// member is not here but in `simplex`.
    pub perturbation: BTreeMap<String, PerturbationIdentity>,
    /// One map per `[[simplex_groups]]` entry: each member's SD on the
    /// group's log-ratio coordinate.
    pub simplex: Vec<BTreeMap<String, Finite>>,
}

/// IF2's perturbation of one parameter: its SD in z and its timing.
#[derive(store::Identity, Clone, Copy, Debug)]
#[identity(tag = "fit_if2_perturbation", version = 1)]
pub(crate) struct PerturbationIdentity {
    pub sd: Finite,
    pub at: PerturbAtIdentity,
}

/// When IF2 perturbs a parameter (`sim::inference::types::PerturbAt`).
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "fit_perturb_at", version = 1)]
pub(crate) enum PerturbAtIdentity {
    EveryObservation,
    TimeZeroOnly,
}

impl From<PerturbAt> for PerturbAtIdentity {
    fn from(at: PerturbAt) -> Self {
        match at {
            PerturbAt::EveryObservation => PerturbAtIdentity::EveryObservation,
            PerturbAt::TimeZeroOnly => PerturbAtIdentity::TimeZeroOnly,
        }
    }
}

impl PerturbationIdentity {
    /// The identity of `p`.
    pub fn of(p: &Perturbation) -> PerturbationIdentity {
        PerturbationIdentity { sd: z_sd(p.sd), at: p.at.into() }
    }
}

/// A PGAS method, without its length (`sweeps`).
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_pgas", version = 1)]
pub(crate) struct PgasIdentity {
    pub backend: Backend,
    pub chains: u64,
    pub particles: u64,
    pub binomial: Binomial,
    pub burn_in: Option<u64>,
    pub thin: Option<u64>,
    pub max_tree_depth: u64,
    pub trajectory_warmup: u64,
    pub csmc_sweeps_per_nuts: u64,
    /// The number of posterior trajectories the run stores. The chain's draws
    /// do not depend on it; the stored trajectories do.
    pub n_trajectories: u64,
    pub dense_mass: bool,
    pub use_nuts: bool,
    pub ancestor_sampling: bool,
}

/// A PMMH method, without its length (`iterations`).
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_pmmh", version = 1)]
pub(crate) struct PmmhIdentity {
    pub backend: Backend,
    pub chains: u64,
    pub particles: u64,
    pub burn_in: Option<u64>,
    pub thin: Option<u64>,
    pub adapt: bool,
    pub adapt_start: u64,
    pub binomial: Binomial,
    /// Every estimated parameter's resolved initial proposal SD in z, by
    /// name: the SD the sampler starts with.
    pub proposal: BTreeMap<String, Finite>,
}

/// An MH method on the deterministic likelihood, without its length
/// (`iterations`).
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_mh", version = 1)]
pub(crate) struct MhIdentity {
    pub backend: Backend,
    pub chains: u64,
    pub burn_in: Option<u64>,
    pub thin: Option<u64>,
    pub adapt: bool,
    pub adapt_start: u64,
    /// The coarse step over the unscored warm-up, as written.
    pub burnin_dt: Option<Finite>,
    pub dt_check: DtCheckIdentity,
    /// As [`PmmhIdentity::proposal`].
    pub proposal: BTreeMap<String, Finite>,
}

/// A `pfilter` method: the filter run at one θ.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_pfilter", version = 1)]
pub(crate) struct PfilterMethodIdentity {
    pub backend: Backend,
    pub particles: u64,
    pub replicates: Option<u64>,
    /// Whether the run stores the per-step ancestor indices.
    pub record_ancestry: bool,
    /// Whether the run stores the prequential trace.
    pub record_prequential: bool,
    pub binomial: Binomial,
}

/// One of the three NLopt methods; the [`MethodIdentity`] variant says which.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method_nlopt", version = 1)]
pub(crate) struct NloptIdentity {
    pub backend: Backend,
    pub chains: u64,
    /// NLopt's `xtol_rel`.
    pub tolerance: Finite,
    pub max_evals: u64,
    pub gate: GateIdentity,
    pub dt_check: DtCheckIdentity,
}

impl MethodIdentity {
    /// The identity of `method` and its length, 0 for a method that cannot be
    /// extended, with `steps` its resolved per-parameter steps
    /// ([`crate::fit::steps::resolve_steps`] of the same method). A NaN or
    /// infinite setting is refused with a message naming it.
    ///
    /// Each `config_v2::Algorithm` variant is taken apart with no `..`, so a
    /// field added to one does not compile until it is placed in the identity
    /// here or left out by name. The written step tables are left out: the
    /// identity holds `steps`, so a step written at its default and an
    /// omitted one key the same.
    pub fn of(method: &Method, steps: &ResolvedSteps) -> Result<(MethodIdentity, u64), String> {
        use config_v2::Algorithm as A;
        let Method { algorithm, starts: _ } = method;
        Ok(match algorithm {
            A::IF2 {
                backend, chains, particles, iterations, cooling, cooling_target_iters,
                loglik_eval, gate, dt_check, binomial, perturbation: _,
            } => {
                let ResolvedSteps::If2 { params, simplex } = steps else {
                    return Err(steps_mismatch("if2"));
                };
                let config_v2::LoglikEvalConfig { n_particles, n_replicates, combine } = loglik_eval;
                let id = If2Identity {
                    backend: (*backend).into(),
                    chains: count(*chains),
                    particles: count(*particles),
                    iterations: count(*iterations),
                    cooling: finite("cooling", *cooling)?,
                    cooling_target_iters: count(*cooling_target_iters),
                    loglik_eval: LoglikEvalIdentity {
                        n_particles: count(*n_particles),
                        n_replicates: count(*n_replicates),
                        combine: (*combine).into(),
                    },
                    gate: GateIdentity::of(gate)?,
                    dt_check: DtCheckIdentity::of(dt_check)?,
                    binomial: (*binomial).into(),
                    perturbation: params
                        .iter()
                        .filter_map(|(name, r)| match r.step {
                            If2Step::Single(p) => Some((name.clone(), PerturbationIdentity::of(&p))),
                            If2Step::SimplexMember => None,
                        })
                        .collect(),
                    simplex: simplex
                        .iter()
                        .map(|g| g.iter().map(|(name, r)| (name.clone(), z_sd(r.sd))).collect())
                        .collect(),
                };
                (MethodIdentity::If2(id), 0)
            }
            A::PGAS {
                backend, chains, particles, sweeps, binomial, burn_in, thin, max_tree_depth,
                trajectory_warmup, csmc_sweeps_per_nuts, n_trajectories, dense_mass, use_nuts,
                ancestor_sampling,
                // Can only refuse a run; never changes what one computes.
                prior_only: _,
            } => {
                let id = PgasIdentity {
                    backend: (*backend).into(),
                    chains: count(*chains),
                    particles: count(*particles),
                    binomial: (*binomial).into(),
                    burn_in: burn_in.map(count),
                    thin: thin.map(count),
                    max_tree_depth: count(*max_tree_depth),
                    trajectory_warmup: count(*trajectory_warmup),
                    csmc_sweeps_per_nuts: count(*csmc_sweeps_per_nuts),
                    n_trajectories: count(*n_trajectories),
                    dense_mass: *dense_mass,
                    use_nuts: *use_nuts,
                    ancestor_sampling: *ancestor_sampling,
                };
                (MethodIdentity::Pgas(id), count(*sweeps))
            }
            A::PMMH {
                backend, chains, particles, iterations, burn_in, thin, adapt, adapt_start, binomial, proposal: _,
                // Can only refuse a run; never changes what one computes.
                prior_only: _,
            } => {
                let id = PmmhIdentity {
                    backend: (*backend).into(),
                    chains: count(*chains),
                    particles: count(*particles),
                    burn_in: burn_in.map(count),
                    thin: thin.map(count),
                    adapt: *adapt,
                    adapt_start: count(*adapt_start),
                    binomial: (*binomial).into(),
                    proposal: proposal_of(steps, "pmmh")?,
                };
                (MethodIdentity::Pmmh(id), count(*iterations))
            }
            A::Mh {
                backend, chains, iterations, burn_in, thin, adapt, adapt_start, burnin_dt, dt_check, proposal: _,
                // Can only refuse a run; never changes what one computes.
                prior_only: _,
            } => {
                let id = MhIdentity {
                    backend: (*backend).into(),
                    chains: count(*chains),
                    burn_in: burn_in.map(count),
                    thin: thin.map(count),
                    adapt: *adapt,
                    adapt_start: count(*adapt_start),
                    burnin_dt: burnin_dt.map(|b| finite("burnin_dt", b)).transpose()?,
                    dt_check: DtCheckIdentity::of(dt_check)?,
                    proposal: proposal_of(steps, "mh")?,
                };
                (MethodIdentity::Mh(id), count(*iterations))
            }
            A::PFilter { backend, particles, replicates, record_ancestry, record_prequential, binomial } => {
                let id = PfilterMethodIdentity {
                    backend: (*backend).into(),
                    particles: count(*particles),
                    replicates: replicates.map(count),
                    record_ancestry: *record_ancestry,
                    record_prequential: *record_prequential,
                    binomial: (*binomial).into(),
                };
                (MethodIdentity::Pfilter(id), 0)
            }
            A::NlSbplx(c) => (MethodIdentity::NlSbplx(NloptIdentity::of(c)?), 0),
            A::NlBobyqa(c) => (MethodIdentity::NlBobyqa(NloptIdentity::of(c)?), 0),
            A::NlLbfgs(c) => (MethodIdentity::NlLbfgs(NloptIdentity::of(c)?), 0),
        })
    }
}

impl GateIdentity {
    fn of(g: &config_v2::GateConfig) -> Result<GateIdentity, String> {
        let config_v2::GateConfig { a_thresh, decibans_thresh } = g;
        Ok(GateIdentity {
            a_thresh: finite("gate.a_thresh", *a_thresh)?,
            decibans_thresh: finite("gate.decibans_thresh", *decibans_thresh)?,
        })
    }
}

impl DtCheckIdentity {
    fn of(d: &config_v2::DtCheckConfig) -> Result<DtCheckIdentity, String> {
        let config_v2::DtCheckConfig { enabled, n_halvings, n_particles, n_replicates, threshold_nats, combine } = d;
        Ok(DtCheckIdentity {
            enabled: *enabled,
            n_halvings: count(*n_halvings),
            n_particles: n_particles.map(count),
            n_replicates: n_replicates.map(count),
            threshold_nats: threshold_nats.map(|t| finite("dt_check.threshold_nats", t)).transpose()?,
            combine: combine.map(Combine::from),
        })
    }
}

impl NloptIdentity {
    fn of(c: &config_v2::NloptConfig) -> Result<NloptIdentity, String> {
        let config_v2::NloptConfig { backend, chains, tolerance, max_evals, gate, dt_check } = c;
        Ok(NloptIdentity {
            backend: (*backend).into(),
            chains: count(*chains),
            tolerance: finite("tolerance", *tolerance)?,
            max_evals: count(*max_evals),
            gate: GateIdentity::of(gate)?,
            dt_check: DtCheckIdentity::of(dt_check)?,
        })
    }
}

impl MethodIdentityRecord {
    /// The algorithm, as `fit.toml` names it (`if2`, `nl-sbplx`, …).
    pub fn algorithm(&self) -> &'static str {
        use crate::run_meta::FitAlgorithm as F;
        match self {
            MethodIdentityRecord::If2(_) => F::If2,
            MethodIdentityRecord::Pgas(_) => F::Pgas,
            MethodIdentityRecord::Pmmh(_) => F::Pmmh,
            MethodIdentityRecord::Mh(_) => F::Mh,
            MethodIdentityRecord::Pfilter(_) => F::Pfilter,
            MethodIdentityRecord::NlSbplx(_) => F::NlSbplx,
            MethodIdentityRecord::NlBobyqa(_) => F::NlBobyqa,
            MethodIdentityRecord::NlLbfgs(_) => F::NlLbfgs,
        }
        .as_str()
    }
}

/// The resolved proposal SDs of a PMMH or MH method, by name.
fn proposal_of(steps: &ResolvedSteps, method: &str) -> Result<BTreeMap<String, Finite>, String> {
    match steps {
        ResolvedSteps::Proposal(map) => Ok(map.iter().map(|(name, r)| (name.clone(), z_sd(r.sd))).collect()),
        _ => Err(steps_mismatch(method)),
    }
}

fn steps_mismatch(method: &str) -> String {
    format!("internal error: the resolved steps are not those of a `{method}` method")
}

/// A step in z as a key. A [`ZSd`] is finite by construction.
fn z_sd(sd: ZSd) -> Finite {
    Finite::new(sd.get()).expect("a ZSd is finite")
}

fn count(x: usize) -> u64 {
    x as u64
}

fn finite(key: &str, x: f64) -> Result<Finite, String> {
    Finite::new(x).map_err(|_| format!("[method] {key} = {x} is not a finite number"))
}
