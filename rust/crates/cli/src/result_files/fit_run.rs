//! A fit run's files, by method family.
//!
//! Which files a fit run stores depends on its method, so each method family
//! has an enum of its own: [`CommonFile`] for the files every method stores,
//! [`McmcFile`] for MH, PMMH and PGAS, [`PgasFile`] beside it for PGAS,
//! [`If2File`], [`NloptFile`] and [`PfilterMethodFile`]. A file that several
//! families write, such as `diagnostics.json`, is a variant in each.
//!
//! A method's marker type ([`Mh`], [`Pmmh`], [`Pgas`], [`If2`], [`Nlopt`],
//! [`PfilterMethod`]) names its own enum as [`FitMethod::Own`] and lists the
//! files a run of it stores ([`MethodFiles::files`]), computed before the run
//! from its `[method]` table and its number of chains. A method writes its
//! files in a scratch working directory; [`MethodOutput`] then writes exactly
//! the declared files into the run and refuses any other file it finds there,
//! naming it. An MH run's output is a `MethodOutput<Mh>`, whose
//! [`MethodOutput::create`] takes only `CommonFile` and `McmcFile` variants,
//! so a line that writes an IF2 file there does not compile.

use std::collections::BTreeMap;
use std::path::Path;

use super::{Column, Format, ResultFile, Role};
use crate::fit::config_v2::Algorithm;
use crate::fit_store::FitRun;

fn chain_dir(chain: u32) -> String {
    score::convergence::ChainIndex(chain as usize).dir_name()
}

const ESTIMATED: Column = Column::Each("each estimated parameter");
const DRAWS: [Column; 3] = [Column::Named("chain"), Column::Named("draw"), Column::Each("every parameter, estimated and fixed")];
const MH_TRACE: [Column; 5] = [
    Column::Named("step"),
    Column::Named("log_likelihood"),
    Column::Named("log_posterior"),
    Column::Named("accepted"),
    ESTIMATED,
];
const PGAS_TRACE: [Column; 4] = [
    Column::Named("sweep"),
    Column::Named("log_complete_data_ll"),
    Column::Named("log_posterior"),
    Column::Each("the sampler's per-sweep diagnostics, then each estimated parameter"),
];
const CHAIN_STARTS: [Column; 7] = [
    Column::Named("chain_id"),
    Column::Named("attempt"),
    Column::Named("status"),
    Column::Named("source"),
    ESTIMATED,
    Column::Named("ess"),
    Column::Named("reason"),
];
const TRAJECTORIES: [Column; 4] = [
    Column::Named("chain"),
    Column::Named("draw"),
    Column::Named("time"),
    Column::Each("the step's bounds or date, each compartment, each `flow_<name>` and each projection"),
];
const FILTER_ESS: [Column; 6] = [
    Column::Named("chain"),
    Column::Named("obs"),
    Column::Named("time"),
    Column::Named("ess_mean"),
    Column::Named("ess_min"),
    Column::Named("n_sweeps"),
];
const PARAMETER_TRACES: [Column; 4] =
    [Column::Named("iteration"), Column::Named("loglik"), Column::Named("if2_perturbed_loglik"), ESTIMATED];
const CHAIN_EVALUATIONS: [Column; 8] = [
    Column::Named("chain"),
    Column::Named("loglik"),
    Column::Named("se"),
    Column::Named("ess_mean"),
    Column::Named("ess_min"),
    Column::Named("ess_min_step"),
    Column::Named("n_neg_inf_incr"),
    ESTIMATED,
];
const ITERATION_DIAGNOSTICS: [Column; 4] =
    [Column::Named("chain"), Column::Named("iteration"), Column::Named("loglik"), Column::Named("if2_perturbed_loglik")];
const CHAIN_RESULTS: [Column; 5] =
    [Column::Named("chain"), Column::Named("loglik"), Column::Named("status"), Column::Named("n_evals"), ESTIMATED];
const OPT_TRACE: [Column; 5] =
    [Column::Named("chain"), Column::Named("eval"), Column::Named("loglik"), Column::Named("best_loglik"), ESTIMATED];
const LOGLIKS: [Column; 2] = [Column::Named("replicate"), Column::Named("loglik")];
const PREQUENTIAL_JOINT: [Column; 3] = [Column::Named("t"), Column::Named("log_score"), Column::Named("ess")];
const PREQUENTIAL_STREAMS: [Column; 11] = [
    Column::Named("t"),
    Column::Named("stream"),
    Column::Named("y_obs"),
    Column::Named("y_pred_q05"),
    Column::Named("y_pred_q25"),
    Column::Named("y_pred_q50"),
    Column::Named("y_pred_q75"),
    Column::Named("y_pred_q95"),
    Column::Named("log_score"),
    Column::Named("crps"),
    Column::Named("pit"),
];

// ── Every method ─────────────────────────────────────────────────────────────

/// The files every method stores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CommonFile {
    /// `method.toml`: the run's merged `[method]` and its `[estimate]`
    /// perturbation keys.
    Method,
    /// `summary.json`: the run's fit summary.
    Summary,
    /// A starts file named directly, stored by its extension.
    StartsInput(String),
    /// `fit.toml`, the configuration as given (provenance).
    FitConfig,
    /// `run_inputs.json`, what the run was asked to do (provenance).
    RunInputs,
    /// `fit.meta.json`, the run's sidecar (provenance).
    FitMeta,
    /// `execution.json`, the attempt's times and process (provenance).
    Execution,
}

impl ResultFile for CommonFile {
    fn logical(&self) -> String {
        match self {
            CommonFile::Method => "method".into(),
            CommonFile::Summary => "summary".into(),
            CommonFile::StartsInput(_) => "inputs/starts".into(),
            CommonFile::FitConfig => "fit_config".into(),
            CommonFile::RunInputs => "run_inputs".into(),
            CommonFile::FitMeta => "fit_meta".into(),
            CommonFile::Execution => "execution".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            CommonFile::Method => crate::fit_store::METHOD_FILE.into(),
            CommonFile::Summary => crate::fit::run_summary::FILE.into(),
            CommonFile::StartsInput(ext) => format!("inputs/starts.{ext}"),
            CommonFile::FitConfig => crate::fit_store::FIT_TOML_FILE.into(),
            CommonFile::RunInputs => crate::fit_store::RUN_INPUTS_FILE.into(),
            CommonFile::FitMeta => crate::fit_store::FIT_META_FILE.into(),
            CommonFile::Execution => crate::fit::execution::EXECUTION_RECORD.into(),
        }
    }
    fn role(&self) -> Role {
        match self {
            CommonFile::Method | CommonFile::Summary | CommonFile::StartsInput(_) => Role::Result,
            CommonFile::FitConfig | CommonFile::RunInputs | CommonFile::FitMeta | CommonFile::Execution => Role::Provenance,
        }
    }
    fn format(&self) -> Format {
        match self {
            CommonFile::Method | CommonFile::FitConfig => Format::Toml,
            CommonFile::StartsInput(ext) if ext == "toml" => Format::Toml,
            CommonFile::StartsInput(_) => Format::Tsv,
            CommonFile::Summary | CommonFile::RunInputs | CommonFile::FitMeta | CommonFile::Execution => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            CommonFile::Method => "The run's `[method]` table as merged, with its perturbation keys.",
            CommonFile::Summary => "`camdl.fit-run-summary/v1`: the run's fit summary, without times, paths or a baseline.",
            CommonFile::StartsInput(_) => "A starts file named directly, as read.",
            CommonFile::FitConfig => "The fit configuration file as given.",
            CommonFile::RunInputs => "What the run was asked to do and its best log-likelihood and chain.",
            CommonFile::FitMeta => "The run's sidecar: the data digests, the roles and the training partition.",
            CommonFile::Execution => "The attempt's wall times, process and cost.",
        }
    }
}

// ── MH, PMMH and PGAS ────────────────────────────────────────────────────────

/// An MCMC sampler, which names its summary file and decides its trace's
/// columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sampler {
    Mh,
    Pmmh,
    Pgas,
}

impl Sampler {
    fn name(self) -> &'static str {
        match self {
            Sampler::Mh => "mh",
            Sampler::Pmmh => "pmmh",
            Sampler::Pgas => "pgas",
        }
    }
}

/// The files of MH, PMMH and PGAS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum McmcFile {
    Draws,
    ChainTrace(Sampler, u32),
    ResumeState(u32),
    SamplerSummary(Sampler),
    ChainStarts,
    Diagnostics,
    FitState,
}

impl ResultFile for McmcFile {
    fn logical(&self) -> String {
        match self {
            McmcFile::Draws => "draws".into(),
            McmcFile::ChainTrace(_, c) => format!("{}/trace", chain_dir(*c)),
            McmcFile::ResumeState(c) => format!("{}/resume_state", chain_dir(*c)),
            McmcFile::SamplerSummary(_) => "sampler_summary".into(),
            McmcFile::ChainStarts => "chain_starts".into(),
            McmcFile::Diagnostics => "diagnostics".into(),
            McmcFile::FitState => "fit_state".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            McmcFile::Draws => "draws.tsv".into(),
            McmcFile::ChainTrace(_, c) => format!("{}/trace.tsv", chain_dir(*c)),
            McmcFile::ResumeState(c) => format!("{}/resume_state.bin", chain_dir(*c)),
            McmcFile::SamplerSummary(s) => format!("{}_summary.json", s.name()),
            McmcFile::ChainStarts => "chain_starts.tsv".into(),
            McmcFile::Diagnostics => "diagnostics.json".into(),
            McmcFile::FitState => "fit_state.toml".into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            McmcFile::Draws | McmcFile::ChainTrace(..) | McmcFile::ChainStarts => Format::Tsv,
            McmcFile::ResumeState(_) => Format::Binary,
            McmcFile::SamplerSummary(_) | McmcFile::Diagnostics => Format::Json,
            McmcFile::FitState => Format::Toml,
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            McmcFile::Draws => &DRAWS,
            McmcFile::ChainTrace(Sampler::Pgas, _) => &PGAS_TRACE,
            McmcFile::ChainTrace(_, _) => &MH_TRACE,
            McmcFile::ChainStarts => &CHAIN_STARTS,
            _ => &[],
        }
    }
    fn commented(&self) -> bool {
        matches!(self, McmcFile::ChainStarts)
    }
    fn about(&self) -> &'static str {
        match self {
            McmcFile::Draws => "The thinned post-warm-up draws of every chain, with every parameter, estimated and fixed.",
            McmcFile::ChainTrace(Sampler::Pgas, _) => "One chain's every sweep: the log densities, the sampler's diagnostics and the parameters.",
            McmcFile::ChainTrace(..) => "One chain's every step: the log densities, whether it was accepted, and the parameters.",
            McmcFile::ResumeState(_) => "One chain's internal state for `--resume`.",
            McmcFile::SamplerSummary(_) => "The sampler's summary: the convergence and acceptance figures.",
            McmcFile::ChainStarts => "Each chain's starting point, and every start a chain tried and was refused.",
            McmcFile::Diagnostics => "An array of findings, each with `kind`, `severity`, `message` and `method`.",
            McmcFile::FitState => "The run's state: its method, chains and verdict.",
        }
    }
}

/// The files PGAS stores beside its [`McmcFile`] set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PgasFile {
    Trajectories(u32),
    TrajectoriesMeta(u32),
    CrossChainCompat,
    FilterEss,
    LatentConvergence,
}

impl ResultFile for PgasFile {
    fn logical(&self) -> String {
        match self {
            PgasFile::Trajectories(c) => format!("{}/trajectories", chain_dir(*c)),
            PgasFile::TrajectoriesMeta(c) => format!("{}/trajectories_meta", chain_dir(*c)),
            PgasFile::CrossChainCompat => "cross_chain_compat".into(),
            PgasFile::FilterEss => "filter_ess".into(),
            PgasFile::LatentConvergence => "latent_convergence".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            PgasFile::Trajectories(c) => format!("{}/trajectories.tsv", chain_dir(*c)),
            PgasFile::TrajectoriesMeta(c) => format!("{}/trajectories.json", chain_dir(*c)),
            PgasFile::CrossChainCompat => crate::fit::cross_chain_compat::CROSS_CHAIN_COMPAT_JSON.into(),
            PgasFile::FilterEss => crate::fit::filter_ess::FILTER_ESS_TSV.into(),
            PgasFile::LatentConvergence => crate::fit::latent_convergence::LATENT_CONVERGENCE_TSV.into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            PgasFile::Trajectories(_) | PgasFile::FilterEss | PgasFile::LatentConvergence => Format::Tsv,
            PgasFile::TrajectoriesMeta(_) | PgasFile::CrossChainCompat => Format::Json,
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            PgasFile::Trajectories(_) => &TRAJECTORIES,
            PgasFile::FilterEss => &FILTER_ESS,
            _ => &[],
        }
    }
    fn commented(&self) -> bool {
        matches!(self, PgasFile::Trajectories(_))
    }
    fn about(&self) -> &'static str {
        match self {
            PgasFile::Trajectories(_) => "One chain's saved posterior paths, stacked, one row per draw and time.",
            PgasFile::TrajectoriesMeta(_) => "What one chain's `trajectories.tsv` holds: its draws, columns and granularity.",
            PgasFile::CrossChainCompat => "The log density of each chain's final path under every other chain's parameters.",
            PgasFile::FilterEss => "The conditional filter's effective sample size at every observation, pooled over sweeps.",
            PgasFile::LatentConvergence => "Whether the chains agree about the latent states, per cell of the paths.",
        }
    }
}

/// PGAS's own files: its MCMC set and its paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PgasOwn {
    Mcmc(McmcFile),
    Pgas(PgasFile),
}

impl ResultFile for PgasOwn {
    fn logical(&self) -> String {
        match self {
            PgasOwn::Mcmc(f) => f.logical(),
            PgasOwn::Pgas(f) => f.logical(),
        }
    }
    fn stored(&self) -> String {
        match self {
            PgasOwn::Mcmc(f) => f.stored(),
            PgasOwn::Pgas(f) => f.stored(),
        }
    }
    fn role(&self) -> Role {
        match self {
            PgasOwn::Mcmc(f) => f.role(),
            PgasOwn::Pgas(f) => f.role(),
        }
    }
    fn format(&self) -> Format {
        match self {
            PgasOwn::Mcmc(f) => f.format(),
            PgasOwn::Pgas(f) => f.format(),
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            PgasOwn::Mcmc(f) => f.columns(),
            PgasOwn::Pgas(f) => f.columns(),
        }
    }
    fn commented(&self) -> bool {
        match self {
            PgasOwn::Mcmc(f) => f.commented(),
            PgasOwn::Pgas(f) => f.commented(),
        }
    }
    fn about(&self) -> &'static str {
        match self {
            PgasOwn::Mcmc(f) => f.about(),
            PgasOwn::Pgas(f) => f.about(),
        }
    }
}

// ── IF2 ──────────────────────────────────────────────────────────────────────

/// The files of IF2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum If2File {
    ParameterTraces(u32),
    ChainFinalParams(u32),
    FinalParams,
    MleParams,
    ChainEvaluations,
    IterationDiagnostics,
    ChainStarts,
    Diagnostics,
    FitState,
}

impl ResultFile for If2File {
    fn logical(&self) -> String {
        match self {
            If2File::ParameterTraces(c) => format!("{}/parameter_traces", chain_dir(*c)),
            If2File::ChainFinalParams(c) => format!("{}/final_params", chain_dir(*c)),
            If2File::FinalParams => "final_params".into(),
            If2File::MleParams => "mle_params".into(),
            If2File::ChainEvaluations => "chain_evaluations".into(),
            If2File::IterationDiagnostics => "iteration_diagnostics".into(),
            If2File::ChainStarts => "chain_starts".into(),
            If2File::Diagnostics => "diagnostics".into(),
            If2File::FitState => "fit_state".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            If2File::ParameterTraces(c) => format!("{}/parameter_traces.tsv", chain_dir(*c)),
            If2File::ChainFinalParams(c) => format!("{}/final_params.toml", chain_dir(*c)),
            If2File::FinalParams => "final_params.toml".into(),
            If2File::MleParams => "mle_params.toml".into(),
            If2File::ChainEvaluations => "chain_evaluations.tsv".into(),
            If2File::IterationDiagnostics => "diagnostics.tsv".into(),
            If2File::ChainStarts => "chain_starts.tsv".into(),
            If2File::Diagnostics => "diagnostics.json".into(),
            If2File::FitState => "fit_state.toml".into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            If2File::ParameterTraces(_) | If2File::ChainEvaluations | If2File::IterationDiagnostics | If2File::ChainStarts => {
                Format::Tsv
            }
            If2File::ChainFinalParams(_) | If2File::FinalParams | If2File::MleParams | If2File::FitState => Format::Toml,
            If2File::Diagnostics => Format::Json,
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            If2File::ParameterTraces(_) => &PARAMETER_TRACES,
            If2File::ChainEvaluations => &CHAIN_EVALUATIONS,
            If2File::IterationDiagnostics => &ITERATION_DIAGNOSTICS,
            If2File::ChainStarts => &CHAIN_STARTS,
            _ => &[],
        }
    }
    fn commented(&self) -> bool {
        matches!(
            self,
            If2File::ParameterTraces(_) | If2File::ChainEvaluations | If2File::IterationDiagnostics | If2File::ChainStarts
        )
    }
    fn about(&self) -> &'static str {
        match self {
            If2File::ParameterTraces(_) => "One chain's estimate at every iteration, with its log-likelihoods.",
            If2File::ChainFinalParams(_) => "One chain's final estimate.",
            If2File::FinalParams => "The winning chain's estimate: the same values as `mle_params`.",
            If2File::MleParams => "The point estimate, parameters at the top level and a `[provenance]` table; read by `--params`.",
            If2File::ChainEvaluations => "Each chain's final estimate, evaluated by a fresh filter, with its standard error and ESS.",
            If2File::IterationDiagnostics => "Each chain's log-likelihoods at every iteration.",
            If2File::ChainStarts => "Each chain's starting point, and every start a chain tried and was refused.",
            If2File::Diagnostics => "An array of findings, each with `kind`, `severity`, `message` and `method`.",
            If2File::FitState => "The run's state: its method, chains and verdict.",
        }
    }
}

// ── NLopt ────────────────────────────────────────────────────────────────────

/// The files of nl-sbplx, nl-bobyqa and nl-lbfgs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NloptFile {
    ChainResults,
    MleParams,
    Diagnostics,
    FitState,
    /// `opt_trace.tsv`, each search's running best, a provenance file: its
    /// rows depend on wall-clock time.
    OptTrace,
}

impl ResultFile for NloptFile {
    fn logical(&self) -> String {
        match self {
            NloptFile::ChainResults => "chain_results".into(),
            NloptFile::MleParams => "mle_params".into(),
            NloptFile::Diagnostics => "diagnostics".into(),
            NloptFile::FitState => "fit_state".into(),
            NloptFile::OptTrace => "opt_trace".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            NloptFile::ChainResults => "chain_results.tsv".into(),
            NloptFile::MleParams => "mle_params.toml".into(),
            NloptFile::Diagnostics => "diagnostics.json".into(),
            NloptFile::FitState => "fit_state.toml".into(),
            NloptFile::OptTrace => crate::fit::opt_trace::OPT_TRACE.into(),
        }
    }
    fn role(&self) -> Role {
        match self {
            NloptFile::OptTrace => Role::Provenance,
            _ => Role::Result,
        }
    }
    fn format(&self) -> Format {
        match self {
            NloptFile::ChainResults | NloptFile::OptTrace => Format::Tsv,
            NloptFile::MleParams | NloptFile::FitState => Format::Toml,
            NloptFile::Diagnostics => Format::Json,
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            NloptFile::ChainResults => &CHAIN_RESULTS,
            NloptFile::OptTrace => &OPT_TRACE,
            _ => &[],
        }
    }
    fn about(&self) -> &'static str {
        match self {
            NloptFile::ChainResults => "Each search's end: its log-likelihood, status, evaluations and estimate.",
            NloptFile::MleParams => "The point estimate, parameters at the top level and a `[provenance]` table; read by `--params`.",
            NloptFile::Diagnostics => "An array of findings, each with `kind`, `severity`, `message` and `method`.",
            NloptFile::FitState => "The run's state: its method, chains and verdict.",
            NloptFile::OptTrace => "Each search's every evaluation and its running best.",
        }
    }
}

// ── The pfilter method ───────────────────────────────────────────────────────

/// The files of the pfilter method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PfilterMethodFile {
    Logliks,
    PrequentialJoint,
    PrequentialStreams,
    PrequentialRecord,
}

impl ResultFile for PfilterMethodFile {
    fn logical(&self) -> String {
        match self {
            PfilterMethodFile::Logliks => "logliks".into(),
            PfilterMethodFile::PrequentialJoint => "prequential_joint".into(),
            PfilterMethodFile::PrequentialStreams => "prequential_streams".into(),
            PfilterMethodFile::PrequentialRecord => "prequential_record".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            PfilterMethodFile::Logliks => "logliks.tsv".into(),
            PfilterMethodFile::PrequentialJoint => "prequential.tsv".into(),
            PfilterMethodFile::PrequentialStreams => "prequential.streams.tsv".into(),
            PfilterMethodFile::PrequentialRecord => "prequential.json".into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            PfilterMethodFile::PrequentialRecord => Format::Json,
            _ => Format::Tsv,
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            PfilterMethodFile::Logliks => &LOGLIKS,
            PfilterMethodFile::PrequentialJoint => &PREQUENTIAL_JOINT,
            PfilterMethodFile::PrequentialStreams => &PREQUENTIAL_STREAMS,
            PfilterMethodFile::PrequentialRecord => &[],
        }
    }
    fn about(&self) -> &'static str {
        match self {
            PfilterMethodFile::Logliks => "Each replicate filter's log-likelihood.",
            PfilterMethodFile::PrequentialJoint => "At each observation time, the joint one-step-ahead log score and the ESS.",
            PfilterMethodFile::PrequentialStreams => "At each observation time and stream, the predictive quantiles and the scores.",
            PfilterMethodFile::PrequentialRecord => "The prequential record: the scores and each stream's predictive samples.",
        }
    }
}

// ── The methods ──────────────────────────────────────────────────────────────

/// A file of a run of a method whose own files are `O`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MethodFileRef<O> {
    Common(CommonFile),
    Own(O),
}

impl<O: ResultFile> ResultFile for MethodFileRef<O> {
    fn logical(&self) -> String {
        match self {
            MethodFileRef::Common(f) => f.logical(),
            MethodFileRef::Own(f) => f.logical(),
        }
    }
    fn stored(&self) -> String {
        match self {
            MethodFileRef::Common(f) => f.stored(),
            MethodFileRef::Own(f) => f.stored(),
        }
    }
    fn role(&self) -> Role {
        match self {
            MethodFileRef::Common(f) => f.role(),
            MethodFileRef::Own(f) => f.role(),
        }
    }
    fn format(&self) -> Format {
        match self {
            MethodFileRef::Common(f) => f.format(),
            MethodFileRef::Own(f) => f.format(),
        }
    }
    fn columns(&self) -> &'static [Column] {
        match self {
            MethodFileRef::Common(f) => f.columns(),
            MethodFileRef::Own(f) => f.columns(),
        }
    }
    fn commented(&self) -> bool {
        match self {
            MethodFileRef::Common(f) => f.commented(),
            MethodFileRef::Own(f) => f.commented(),
        }
    }
    fn about(&self) -> &'static str {
        match self {
            MethodFileRef::Common(f) => f.about(),
            MethodFileRef::Own(f) => f.about(),
        }
    }
}

/// A file of a fit run of any method: the kind's file enum, one variant per
/// method family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FitRunFile {
    Common(CommonFile),
    Mcmc(McmcFile),
    Pgas(PgasFile),
    If2(If2File),
    Nlopt(NloptFile),
    Pfilter(PfilterMethodFile),
}

impl FitRunFile {
    fn inner(&self) -> &dyn ResultFile {
        match self {
            FitRunFile::Common(f) => f,
            FitRunFile::Mcmc(f) => f,
            FitRunFile::Pgas(f) => f,
            FitRunFile::If2(f) => f,
            FitRunFile::Nlopt(f) => f,
            FitRunFile::Pfilter(f) => f,
        }
    }
}

impl ResultFile for FitRunFile {
    fn logical(&self) -> String {
        self.inner().logical()
    }
    fn stored(&self) -> String {
        self.inner().stored()
    }
    fn role(&self) -> Role {
        self.inner().role()
    }
    fn format(&self) -> Format {
        self.inner().format()
    }
    fn columns(&self) -> &'static [Column] {
        self.inner().columns()
    }
    fn commented(&self) -> bool {
        self.inner().commented()
    }
    fn about(&self) -> &'static str {
        self.inner().about()
    }
}

impl super::Files for FitRun {
    type File = FitRunFile;
}

impl From<McmcFile> for FitRunFile {
    fn from(f: McmcFile) -> Self {
        FitRunFile::Mcmc(f)
    }
}

impl From<PgasOwn> for FitRunFile {
    fn from(f: PgasOwn) -> Self {
        match f {
            PgasOwn::Mcmc(f) => FitRunFile::Mcmc(f),
            PgasOwn::Pgas(f) => FitRunFile::Pgas(f),
        }
    }
}

impl From<If2File> for FitRunFile {
    fn from(f: If2File) -> Self {
        FitRunFile::If2(f)
    }
}

impl From<NloptFile> for FitRunFile {
    fn from(f: NloptFile) -> Self {
        FitRunFile::Nlopt(f)
    }
}

impl From<PfilterMethodFile> for FitRunFile {
    fn from(f: PfilterMethodFile) -> Self {
        FitRunFile::Pfilter(f)
    }
}

impl<O: Into<FitRunFile>> From<MethodFileRef<O>> for FitRunFile {
    fn from(f: MethodFileRef<O>) -> Self {
        match f {
            MethodFileRef::Common(c) => FitRunFile::Common(c),
            MethodFileRef::Own(o) => o.into(),
        }
    }
}

/// A fit method, by the files it stores besides [`CommonFile`].
pub(crate) trait FitMethod {
    /// Its own files: [`McmcFile`] for MH and PMMH, [`PgasOwn`] for PGAS,
    /// [`If2File`], [`NloptFile`], [`PfilterMethodFile`].
    type Own: ResultFile + Clone + PartialEq + Into<FitRunFile>;
    /// The `algorithm` key.
    fn name(&self) -> &'static str;
}

/// How one chain of a run ended, as the final row of its chain in
/// `chain_starts.tsv` records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChainEnd {
    /// `accepted`: the chain ran from its start to the end.
    Accepted,
    /// `skipped`: the chain ran from its start and was dropped before the
    /// end (IF2, a filter that degenerated).
    Skipped,
    /// `refused`: no start could run, so the chain never ran.
    Refused,
}

/// What a completed run records about how it ended: each chain's end and,
/// for PGAS, what its summary says it computed. Read from the run's own
/// records ([`FitOutcome::read`]), and from these alone the run's outcome
/// files follow ([`MethodFiles::outcome_files`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FitOutcome {
    /// Each chain's end, by chain number. A chain with no row in
    /// `chain_starts.tsv`, as in a method that writes none, ran.
    pub chains: BTreeMap<u32, ChainEnd>,
    pub pgas: PgasOutcome,
    /// The chains `chain_evaluations.tsv` (IF2) has a row for: the chains
    /// that produced a result. `None` when there is no such record.
    pub evaluated: Option<std::collections::BTreeSet<u32>>,
}

/// What `pgas_summary.json` records of the tables PGAS computes when the
/// outcome allows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PgasOutcome {
    /// Each chain's number of saved posterior paths (`trajectories.n_saved`).
    pub n_saved: BTreeMap<u32, usize>,
    /// The summary holds the filter-ESS block.
    pub filter_ess: bool,
    /// The summary holds the latent-convergence block.
    pub latent_convergence: bool,
}

impl FitOutcome {
    /// The outcome `dir`'s records state: `chain_starts.tsv` (the final row
    /// of each chain) and `pgas_summary.json`. A record that is absent or
    /// does not read states nothing: every chain ran and no PGAS table was
    /// computed.
    #[expect(clippy::disallowed_methods, reason = "reads the run's records in a method's working directory, or in a stored run's folder in a test, to know which of its files the outcome allows")]
    pub(crate) fn read(dir: &Path) -> FitOutcome {
        let mut chains = BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(dir.join(McmcFile::ChainStarts.stored())) {
            let mut lines = text.lines().filter(|l| !l.starts_with('#'));
            let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
            let col = |name: &str| header.iter().position(|h| *h == name);
            if let (Some(c), Some(s)) = (col("chain_id"), col("status")) {
                for row in lines {
                    let cells: Vec<&str> = row.split('\t').collect();
                    let (Some(chain), Some(status)) = (cells.get(c).and_then(|v| v.parse::<u32>().ok()), cells.get(s)) else {
                        continue;
                    };
                    let end = match *status {
                        "accepted" => ChainEnd::Accepted,
                        "skipped" => ChainEnd::Skipped,
                        "refused" => ChainEnd::Refused,
                        _ => continue,
                    };
                    chains.insert(chain, end);
                }
            }
        }
        let mut pgas = PgasOutcome::default();
        let summary = std::fs::read(dir.join(McmcFile::SamplerSummary(Sampler::Pgas).stored()))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        if let Some(s) = summary {
            pgas.filter_ess = s.get(crate::fit::filter_ess::FILTER_ESS_KEY).is_some_and(|v| !v.is_null());
            pgas.latent_convergence = s.get(crate::fit::latent_convergence::LATENT_CONVERGENCE_KEY).is_some_and(|v| !v.is_null());
            if let Some(t) = s.get("trajectories") {
                let ids = t["chain_id"].as_array().cloned().unwrap_or_default();
                let saved = t["n_saved"].as_array().cloned().unwrap_or_default();
                for (id, n) in ids.iter().zip(&saved) {
                    if let (Some(id), Some(n)) = (id.as_u64(), n.as_u64()) {
                        pgas.n_saved.insert(id as u32, n as usize);
                    }
                }
            }
        }
        let evaluated = std::fs::read_to_string(dir.join(If2File::ChainEvaluations.stored())).ok().map(|text| {
            let mut lines = text.lines().filter(|l| !l.starts_with('#'));
            let col = lines.next().unwrap_or("").split('\t').position(|h| h == "chain");
            lines.filter_map(|l| col.and_then(|c| l.split('\t').nth(c)).and_then(|v| v.parse::<u32>().ok())).collect()
        });
        FitOutcome { chains, pgas, evaluated }
    }

    /// How chain `c` ended.
    pub(crate) fn end(&self, c: u32) -> ChainEnd {
        self.chains.get(&c).copied().unwrap_or(ChainEnd::Accepted)
    }

    /// Whether chain `c` produced a result: it has a row in
    /// `chain_evaluations.tsv`, or there is no such record.
    pub(crate) fn evaluated(&self, c: u32) -> bool {
        self.evaluated.as_ref().is_none_or(|set| set.contains(&c))
    }

    /// The number of chains that ran to the end.
    fn accepted(&self, n_chains: u32) -> usize {
        (0..n_chains).filter(|c| self.end(*c) == ChainEnd::Accepted).count()
    }
}

/// The files a run of a method stores: those that follow from its `[method]`
/// table and its number of chains, known before the run starts, and those
/// that follow from its recorded outcome.
pub(crate) trait MethodFiles: FitMethod {
    /// Every file a run on `n_chains` chains stores whatever its outcome,
    /// besides a starts file named directly.
    fn files(&self, n_chains: u32) -> Vec<MethodFileRef<Self::Own>>;
    /// The files a run with the recorded `outcome` stores besides
    /// [`MethodFiles::files`]: each chain's files, by how the chain ended,
    /// and the tables PGAS computes across chains and paths when its summary
    /// says it did. A completed run holds exactly these and the files of
    /// [`MethodFiles::files`].
    fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<MethodFileRef<Self::Own>> {
        let _ = (n_chains, outcome);
        Vec::new()
    }
}

/// The files every method stores, besides a starts file.
fn common<O>() -> Vec<MethodFileRef<O>> {
    [CommonFile::Method, CommonFile::Summary, CommonFile::FitConfig, CommonFile::RunInputs, CommonFile::FitMeta, CommonFile::Execution]
        .into_iter()
        .map(MethodFileRef::Common)
        .collect()
}

/// The MCMC set of `sampler` that does not depend on the outcome.
fn mcmc(sampler: Sampler) -> Vec<McmcFile> {
    vec![McmcFile::Draws, McmcFile::SamplerSummary(sampler), McmcFile::ChainStarts, McmcFile::Diagnostics, McmcFile::FitState]
}

/// Each chain's MCMC files by how it ended: a chain that ran has its trace
/// and its resume state; a refused chain's trace is opened before its start
/// is scored, so it has its trace, with no row, and no resume state.
fn mcmc_chains(sampler: Sampler, n_chains: u32, outcome: &FitOutcome) -> Vec<McmcFile> {
    let mut v = Vec::new();
    for c in 0..n_chains {
        v.push(McmcFile::ChainTrace(sampler, c));
        if outcome.end(c) != ChainEnd::Refused {
            v.push(McmcFile::ResumeState(c));
        }
    }
    v
}

/// `mh`.
pub(crate) struct Mh;
/// `pmmh`.
pub(crate) struct Pmmh;
/// `pgas`, with the number of posterior paths each chain saves.
pub(crate) struct Pgas {
    pub n_trajectories: usize,
}
/// `if2`.
pub(crate) struct If2;
/// `nl-sbplx`, `nl-bobyqa` or `nl-lbfgs`, by its `algorithm` key.
pub(crate) struct Nlopt {
    pub name: &'static str,
}
/// `pfilter` as a fit method, with whether it records its prequential scores.
pub(crate) struct PfilterMethod {
    pub record_prequential: bool,
}

impl FitMethod for Mh {
    type Own = McmcFile;
    fn name(&self) -> &'static str {
        "mh"
    }
}

impl MethodFiles for Mh {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<McmcFile>> {
        let mut v = common();
        v.extend(mcmc(Sampler::Mh).into_iter().map(MethodFileRef::Own));
        v
    }
    fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<MethodFileRef<McmcFile>> {
        mcmc_chains(Sampler::Mh, n_chains, outcome).into_iter().map(MethodFileRef::Own).collect()
    }
}

impl FitMethod for Pmmh {
    type Own = McmcFile;
    fn name(&self) -> &'static str {
        "pmmh"
    }
}

impl MethodFiles for Pmmh {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<McmcFile>> {
        let mut v = common();
        v.extend(mcmc(Sampler::Pmmh).into_iter().map(MethodFileRef::Own));
        v
    }
    fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<MethodFileRef<McmcFile>> {
        mcmc_chains(Sampler::Pmmh, n_chains, outcome).into_iter().map(MethodFileRef::Own).collect()
    }
}

impl FitMethod for Pgas {
    type Own = PgasOwn;
    fn name(&self) -> &'static str {
        "pgas"
    }
}

impl MethodFiles for Pgas {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<PgasOwn>> {
        let mut v = common();
        v.extend(mcmc(Sampler::Pgas).into_iter().map(|f| MethodFileRef::Own(PgasOwn::Mcmc(f))));
        v
    }
    /// Besides each chain's MCMC files: a chain that ran and saved paths
    /// (`n_trajectories` above zero, and the summary's `n_saved` above
    /// zero) has its paths and their description; the cross-chain table
    /// compares the final paths of two or more chains that ran; the
    /// filter-ESS and latent-convergence tables are stored when the summary
    /// holds their blocks.
    fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<MethodFileRef<PgasOwn>> {
        let mut v: Vec<MethodFileRef<PgasOwn>> =
            mcmc_chains(Sampler::Pgas, n_chains, outcome).into_iter().map(|f| MethodFileRef::Own(PgasOwn::Mcmc(f))).collect();
        let mut own = Vec::new();
        if outcome.pgas.filter_ess {
            own.push(PgasFile::FilterEss);
        }
        if outcome.accepted(n_chains) >= 2 {
            own.push(PgasFile::CrossChainCompat);
        }
        if outcome.pgas.latent_convergence {
            own.push(PgasFile::LatentConvergence);
        }
        for c in 0..n_chains {
            let saved = outcome.pgas.n_saved.get(&c).copied().unwrap_or(0);
            if self.n_trajectories > 0 && outcome.end(c) == ChainEnd::Accepted && saved > 0 {
                own.push(PgasFile::Trajectories(c));
                own.push(PgasFile::TrajectoriesMeta(c));
            }
        }
        v.extend(own.into_iter().map(|f| MethodFileRef::Own(PgasOwn::Pgas(f))));
        v
    }
}

impl FitMethod for If2 {
    type Own = If2File;
    fn name(&self) -> &'static str {
        "if2"
    }
}

impl MethodFiles for If2 {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<If2File>> {
        let mut v = common();
        v.extend(
            [
                If2File::FinalParams,
                If2File::MleParams,
                If2File::ChainEvaluations,
                If2File::IterationDiagnostics,
                If2File::ChainStarts,
                If2File::Diagnostics,
                If2File::FitState,
            ]
            .into_iter()
            .map(MethodFileRef::Own),
        );
        v
    }
    /// IF2 runs every chain, from its last start when every start was
    /// refused, so every chain streams its parameter traces; a chain has its
    /// final estimate when it produced a result, as its row in
    /// `chain_evaluations.tsv` records.
    fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<MethodFileRef<If2File>> {
        let mut v = Vec::new();
        for c in 0..n_chains {
            v.push(If2File::ParameterTraces(c));
            if outcome.evaluated(c) {
                v.push(If2File::ChainFinalParams(c));
            }
        }
        v.into_iter().map(MethodFileRef::Own).collect()
    }
}

impl FitMethod for Nlopt {
    type Own = NloptFile;
    fn name(&self) -> &'static str {
        self.name
    }
}

impl MethodFiles for Nlopt {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<NloptFile>> {
        let mut v = common();
        v.extend(
            [NloptFile::ChainResults, NloptFile::MleParams, NloptFile::Diagnostics, NloptFile::FitState, NloptFile::OptTrace]
                .into_iter()
                .map(MethodFileRef::Own),
        );
        v
    }
}

impl FitMethod for PfilterMethod {
    type Own = PfilterMethodFile;
    fn name(&self) -> &'static str {
        "pfilter"
    }
}

impl MethodFiles for PfilterMethod {
    fn files(&self, _n_chains: u32) -> Vec<MethodFileRef<PfilterMethodFile>> {
        let mut v = common();
        v.push(MethodFileRef::Own(PfilterMethodFile::Logliks));
        if self.record_prequential {
            for f in [PfilterMethodFile::PrequentialJoint, PfilterMethodFile::PrequentialStreams, PfilterMethodFile::PrequentialRecord] {
                v.push(MethodFileRef::Own(f));
            }
        }
        v
    }
}

// ── Writing a run's files ────────────────────────────────────────────────────

/// The run a method's files are written into, with the method and its chain
/// count, from which the files it declares follow.
pub(crate) struct MethodOutput<'r, M: FitMethod> {
    run: &'r mut store::Run<FitRun>,
    method: &'r M,
    n_chains: u32,
    /// The files that do not depend on the outcome, a starts file included.
    required: Vec<MethodFileRef<M::Own>>,
}

impl<'r, M: MethodFiles> MethodOutput<'r, M> {
    /// The output of a run of `method` on `n_chains` chains into `run`, with
    /// `starts`, the extension of a starts file named directly.
    pub(crate) fn new(run: &'r mut store::Run<FitRun>, method: &'r M, n_chains: u32, starts: Option<&str>) -> Self {
        let mut required = method.files(n_chains);
        if let Some(ext) = starts {
            required.push(MethodFileRef::Common(CommonFile::StartsInput(ext.to_string())));
        }
        MethodOutput { run, method, n_chains, required }
    }

    /// Opens `file` in the run, in its role.
    #[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
    pub(crate) fn create(&self, file: MethodFileRef<M::Own>) -> store::Result<store::FileHandle> {
        match file.role() {
            Role::Result => self.run.result_file(&file.stored()),
            Role::Provenance => self.run.provenance_file(&file.stored()),
        }
    }

    /// Writes the files of `work`, the method's working directory, into the
    /// run, and declares that the run holds exactly its required files and
    /// the outcome files its records in `work` state
    /// ([`FitOutcome::read`], [`MethodFiles::outcome_files`]), so a declared
    /// file the method did not write fails the run at `finish`.
    ///
    /// When `whole` (the method completed), a file the run does not declare
    /// is refused, by name and before any file is written. For a run that is
    /// failing, each such file is kept as an attempt file, with no digest, and
    /// its name returned for the failure's reason. The progress snapshot and
    /// the event log are passed to the run's own records while the method
    /// runs, and are not files of the run.
    #[expect(clippy::disallowed_methods, reason = "reads the files a method wrote in this process's temporary working directory, to write them into the run")]
    pub(crate) fn transfer(&mut self, work: &Path, whole: bool) -> Result<Vec<String>, String> {
        let mut found: Vec<String> = Vec::new();
        let mut stack = vec![work.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))? {
                let p = e.map_err(|e| e.to_string())?.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    found.push(p.strip_prefix(work).expect("under the working directory").to_string_lossy().into_owned());
                }
            }
        }
        found.sort();
        found.retain(|name| !is_attempt_channel(name));
        let outcome = FitOutcome::read(work);
        let declared: Vec<MethodFileRef<M::Own>> =
            self.required.iter().cloned().chain(self.method.outcome_files(self.n_chains, &outcome)).collect();
        let lookup = |name: &str| declared.iter().find(|f| f.stored() == name).cloned();
        if whole {
            if let Some(name) = found.iter().find(|n| lookup(n).is_none()) {
                let mut names: Vec<String> = declared.iter().map(ResultFile::logical).collect();
                names.sort();
                return Err(format!(
                    "the {} method wrote `{name}`, which is not among this run's files: {}",
                    self.method.name(),
                    names.join(", ")
                ));
            }
        }
        let mut stray = Vec::new();
        for name in &found {
            let path = work.join(name);
            let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let e = |e: store::Error| format!("{name}: {e}");
            match lookup(name) {
                Some(file) => self.create(file).map_err(e)?.write_all(&bytes).map_err(e)?,
                None => {
                    self.run.attempt_file(name).map_err(e)?.write_all(&bytes).map_err(e)?;
                    stray.push(name.clone());
                }
            }
        }
        self.run.expect(super::expected(declared));
        Ok(stray)
    }
}
/// Whether `name`, a file of a method's working directory, is the progress
/// snapshot or the event log: the channels the run's own records carry
/// while the method runs (`Run::progress`, `Run::event`).
pub(crate) fn is_attempt_channel(name: &str) -> bool {
    name == io::PROGRESS_FILE || name == crate::out::events::EVENT_LOG
}

/// A fit run's method, chosen by its `algorithm`, with the settings that
/// change which files it stores.
pub(crate) enum PlannedMethod {
    Mh(Mh),
    Pmmh(Pmmh),
    Pgas(Pgas),
    If2(If2),
    Nlopt(Nlopt),
    Pfilter(PfilterMethod),
}

impl PlannedMethod {
    pub(crate) fn of(algorithm: &Algorithm) -> PlannedMethod {
        match algorithm {
            Algorithm::Mh { .. } => PlannedMethod::Mh(Mh),
            Algorithm::PMMH { .. } => PlannedMethod::Pmmh(Pmmh),
            Algorithm::PGAS { n_trajectories, .. } => PlannedMethod::Pgas(Pgas { n_trajectories: *n_trajectories }),
            Algorithm::IF2 { .. } => PlannedMethod::If2(If2),
            Algorithm::NlSbplx(_) | Algorithm::NlBobyqa(_) | Algorithm::NlLbfgs(_) => {
                PlannedMethod::Nlopt(Nlopt { name: algorithm.method_name() })
            }
            Algorithm::PFilter { record_prequential, .. } => {
                PlannedMethod::Pfilter(PfilterMethod { record_prequential: *record_prequential })
            }
        }
    }

    /// Every file a run on `n_chains` chains stores whatever its outcome,
    /// with a starts file of extension `starts` when one was named directly:
    /// the set a fit run declares when it begins. [`PlannedMethod::write`]
    /// declares the outcome files it finds besides.
    pub(crate) fn files(&self, n_chains: u32, starts: Option<&str>) -> Vec<FitRunFile> {
        fn of<M: MethodFiles>(m: &M, n_chains: u32) -> Vec<FitRunFile> {
            m.files(n_chains).into_iter().map(FitRunFile::from).collect()
        }
        let mut v = match self {
            PlannedMethod::Mh(m) => of(m, n_chains),
            PlannedMethod::Pmmh(m) => of(m, n_chains),
            PlannedMethod::Pgas(m) => of(m, n_chains),
            PlannedMethod::If2(m) => of(m, n_chains),
            PlannedMethod::Nlopt(m) => of(m, n_chains),
            PlannedMethod::Pfilter(m) => of(m, n_chains),
        };
        if let Some(ext) = starts {
            v.push(FitRunFile::Common(CommonFile::StartsInput(ext.to_string())));
        }
        v
    }

    /// The files a run on `n_chains` chains with the recorded `outcome`
    /// stores besides [`PlannedMethod::files`] ([`MethodFiles::outcome_files`]).
    #[cfg(test)]
    pub(crate) fn outcome_files(&self, n_chains: u32, outcome: &FitOutcome) -> Vec<FitRunFile> {
        fn of<M: MethodFiles>(m: &M, n_chains: u32, outcome: &FitOutcome) -> Vec<FitRunFile> {
            m.outcome_files(n_chains, outcome).into_iter().map(FitRunFile::from).collect()
        }
        match self {
            PlannedMethod::Mh(m) => of(m, n_chains, outcome),
            PlannedMethod::Pmmh(m) => of(m, n_chains, outcome),
            PlannedMethod::Pgas(m) => of(m, n_chains, outcome),
            PlannedMethod::If2(m) => of(m, n_chains, outcome),
            PlannedMethod::Nlopt(m) => of(m, n_chains, outcome),
            PlannedMethod::Pfilter(m) => of(m, n_chains, outcome),
        }
    }

    /// Writes a run's files into `run`: the declared files found in `work`
    /// ([`MethodOutput::transfer`]) and, when a starts file was named
    /// directly, its bytes as `inputs/starts.<ext>`. Returns the names of
    /// the undeclared files a failing run kept as attempt files.
    pub(crate) fn write(
        &self,
        run: &mut store::Run<FitRun>,
        n_chains: u32,
        work: &Path,
        starts: Option<(&str, &[u8])>,
        whole: bool,
    ) -> Result<Vec<String>, String> {
        fn write<M: MethodFiles>(
            m: &M,
            run: &mut store::Run<FitRun>,
            n_chains: u32,
            work: &Path,
            starts: Option<(&str, &[u8])>,
            whole: bool,
        ) -> Result<Vec<String>, String> {
            let mut out = MethodOutput::new(run, m, n_chains, starts.map(|(ext, _)| ext));
            let stray = out.transfer(work, whole)?;
            if let Some((ext, bytes)) = starts {
                let file = MethodFileRef::Common(CommonFile::StartsInput(ext.to_string()));
                let name = file.stored();
                let e = |e: store::Error| format!("{name}: {e}");
                out.create(file).map_err(e)?.write_all(bytes).map_err(e)?;
            }
            Ok(stray)
        }
        match self {
            PlannedMethod::Mh(m) => write(m, run, n_chains, work, starts, whole),
            PlannedMethod::Pmmh(m) => write(m, run, n_chains, work, starts, whole),
            PlannedMethod::Pgas(m) => write(m, run, n_chains, work, starts, whole),
            PlannedMethod::If2(m) => write(m, run, n_chains, work, starts, whole),
            PlannedMethod::Nlopt(m) => write(m, run, n_chains, work, starts, whole),
            PlannedMethod::Pfilter(m) => write(m, run, n_chains, work, starts, whole),
        }
    }
}
