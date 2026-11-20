//! `ResolvedPfilterConfig`: the one resolved configuration of a `camdl
//! pfilter`, from which both its stored leaf's identity and what the particle
//! filter runs are built (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "The shape"; step 4 for `pfilter`).
//!
//! ```text
//! PfilterArgs + --fit's stored problem + defaults ──resolve──► ResolvedPfilterConfig
//!                                                                │          │
//!                                       pfilter_cas::resolve_pfilter      the filter
//! ```
//!
//! [`resolve`] reads every argument once, with the `PfilterArgs` destructured
//! field by field so an argument added to the command does not compile until
//! it is placed: into the resolved configuration (an input that can change
//! the stored log-likelihood), [`PfilterOutputs`] (a file written outside the
//! store, or no leaf at all) or [`PfilterExecution`] (the thread budget). The
//! `--fit` handle is resolved here, once ([`crate::fit::handle::resolve_fit_problem`]:
//! a `fit.toml` names the file's problem, a run handle the stored fit's), and
//! its `[config] allow_degenerate_rates` and `allow_forcing_extrapolation` are
//! folded with the flags into [`ResolvedPfilterConfig::eval`] and
//! [`ResolvedPfilterConfig::forcing`]. Nothing after resolution reads the
//! command line: `cmd_pfilter` hands the three values to the run, where the
//! arguments are not in scope.
//!
//! **What reaches the identity.** The model level hashes the resolved model
//! (compiled under [`ResolvedPfilterConfig::compile`], `--no-licm` passed to the
//! compile) and [`ResolvedPfilterConfig::engine`], this build and the engine
//! switches set; the config level the scoring setup (particles, replicates,
//! the step, the data, the objective, the two policies, the training rows
//! `--save-final-state` keeps under the fit's `holdout_after`, the binomial
//! sampler); the params level the scored point; the seed level the seed.
//! `--pf-max-substeps` and `--time-format` decide whether a run is refused,
//! never what a run that completes stores, so neither is keyed.

use crate::args::types::DataSpec;
use crate::fit::config_v2::LikelihoodObjective;
use crate::fit::fitted_model::FittedSettings;
use crate::fit::handle::FitProblem;

/// The resolved configuration of one `camdl pfilter`.
pub(crate) struct ResolvedPfilterConfig {
    /// The model file as named.
    pub model_path: String,
    /// The settings every compile of the model runs under (`--no-licm`).
    pub compile: crate::util::CompileSettings,
    /// The fit `--fit` names, read once: its problem supplies the objective,
    /// the scenario selection, the data when no `--data` is given, the written
    /// `[config] dt`, `holdout_after` and the two policies below.
    pub fit: Option<FitProblem>,
    /// The likelihood scored: the fit's `ic_free`, else full.
    pub objective: LikelihoodObjective,
    /// `--allow-degenerate-rates`, or the fit's `[config]
    /// allow_degenerate_rates`: carried on the compiled model every particle
    /// runs.
    pub eval: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, or the fit's `[config]
    /// allow_forcing_extrapolation`: the policy the filter runs under and the
    /// leaf's record names.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
    /// `--scenario`, `--enable`, `--disable`, as given.
    pub scenario: ScenarioFlags,
    /// `--param`, `--params`, `--table`, as given.
    pub params: ParamFlags,
    /// `--data`, as given; empty means the fit's `[data]`.
    pub data: Vec<DataSpec>,
    /// `--obs`: the stream a single `--data PATH` binds.
    pub obs: Option<String>,
    /// `--dt`, checked against the model's declared step by the one dt rule.
    pub dt: Option<f64>,
    pub particles: usize,
    pub replicates: usize,
    pub seed: u64,
    /// `--binomial`, else BTPE.
    pub binomial: sim::rng::BinomialAlgorithm,
    /// `--pf-max-substeps`, else the engine's budget: a budget that trips
    /// refuses the run, so it never changes a stored result.
    pub max_substeps: u64,
    /// `--time-format`: how the data's time column is read; a format the
    /// column does not hold is refused.
    pub time_format: crate::caltime_load::TimeFormat,
    /// `--save-final-state`: the file the final particle states are written
    /// to. Under the fit's `holdout_after` it also cuts the bound rows to the
    /// training rows, which the config level keys.
    pub save_final_state: Option<String>,
    /// The engine the identity folds: this build and the engine switches set.
    pub engine: runid::inputs::EngineIdentity,
}

/// `--scenario`, `--enable`, `--disable`, as given on the command line.
pub(crate) struct ScenarioFlags {
    pub scenario: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
}

impl ScenarioFlags {
    pub fn any(&self) -> bool {
        self.scenario.is_some() || !self.enable.is_empty() || !self.disable.is_empty()
    }
}

/// `--param NAME=VALUE`, `--params FILE` and `--table NAME=FILE`, as given.
pub(crate) struct ParamFlags {
    pub param: Vec<(String, f64)>,
    pub params: Vec<std::path::PathBuf>,
    pub table: Vec<(String, std::path::PathBuf)>,
}

/// What `pfilter` writes outside the store, and whether it stores a leaf:
/// none of it changes the stored log-likelihood.
pub(crate) struct PfilterOutputs {
    pub trace: Option<String>,
    pub pf_health: Option<String>,
    pub output: Option<String>,
    /// `--save-paths` with `--n-paths`.
    pub save_paths: Option<(usize, String)>,
    pub save_filtering: Option<String>,
    /// `--save-filtered-streams`: each stream's conditional mean, summarised
    /// before and after each step is scored.
    pub save_filtered_streams: Option<String>,
    pub save_prequential: Option<String>,
    /// `--score-from`: the window the prequential trace scores.
    pub score_from: Option<String>,
    /// `--no-save-samples` negated: the prequential trace keeps its predictive
    /// samples.
    pub save_samples: bool,
    /// `--no-store`: the evaluation, without a stored leaf.
    pub no_store: bool,
}

/// How `pfilter` computes, never what: the thread budget.
pub(crate) struct PfilterExecution {
    /// `--parallel` / CAMDL_PARALLEL: the scoped pool's size (0 = the global
    /// pool, every core).
    pub parallel: usize,
}

/// Resolve `pfilter`'s configuration from its arguments, the `--fit` handle's
/// problem and the defaults, under the compile settings `compile`; refuse
/// (through `out`) a combination that cannot run. Returns the three values and
/// the forcing policy's installation, held for the command.
pub(crate) fn resolve(
    a: &crate::args::PfilterArgs,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
) -> (ResolvedPfilterConfig, PfilterOutputs, PfilterExecution, crate::forcing_extrapolation::Installed) {
    // gh#686: the replicate path reports the spread of the log-likelihood
    // estimate over independent runs and writes nothing per run, so every
    // per-run output would be silently not written — at exit 0, leaving any
    // stale file of that name to be read as this run's. Refuse instead.
    if let Err(e) = super::pfilter::check_per_run_outputs(a) {
        out.fail(format!("error: {e}"));
    }
    let crate::args::PfilterArgs {
        model,
        model_overrides: crate::args::ModelOverrides { params, param, table },
        scenario: crate::args::ScenarioArgs { scenario, enable, disable },
        particles,
        inference:
            crate::args::InferenceCore {
                dt,
                seed,
                parallel,
                allow_degenerate_rates,
                forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
                time_format,
                pf_max_substeps,
                binomial,
            },
        stream: crate::args::StreamSelection { obs },
        data,
        fit,
        replicates,
        trace,
        pf_health,
        output,
        save_final_state,
        save_paths,
        n_paths,
        save_filtering,
        save_filtered_streams,
        save_prequential,
        score_from,
        no_save_samples,
        no_store,
    } = a;
    let path = |p: &std::path::PathBuf| p.to_string_lossy().into_owned();

    // The fit `--fit` names: read once, here.
    let fit: Option<FitProblem> = fit.as_ref().map(|handle| {
        crate::fit::handle::resolve_fit_problem(&handle.to_string_lossy()).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {}: {e}", handle.display()));
        })
    });
    let problem = fit.as_ref().map(|f| &f.config.problem);
    // A rate that collapses (a division by an empty compartment) is 0 under
    // `--allow-degenerate-rates` or the fit's `[config] allow_degenerate_rates`,
    // which `fit run` applied to the fit, and kills the particle otherwise
    // (gh#1235). Carried on the compiled model every particle runs, and keyed
    // in the store leaf from the same value.
    let settings = problem
        .map(|p| FittedSettings::of(&p.config))
        .unwrap_or_default()
        .with_flags(*allow_degenerate_rates, *allow_forcing_extrapolation);
    let eval = crate::env::eval_settings(settings.allow_degenerate_rates);
    // A forcing evaluated outside its range holds its end value under
    // `--allow-forcing-extrapolation` or the fit's `[config]
    // allow_forcing_extrapolation`, and is refused otherwise: the policy the
    // filter runs under and the leaf's record names, installed for the rest of
    // the command and keyed in the leaf from this value.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::new(settings.allow_forcing_extrapolation);
    let installed = forcing.install();
    // The likelihood scored: the fit's own (`ic_free`), read through the
    // function `fit run` uses, so a pfilter at the fit's MLE reports the number
    // the fit maximised. Without `--fit` every observation is scored.
    let objective = problem.map(|p| p.likelihood_objective()).unwrap_or(LikelihoodObjective::Full);

    let resolved = ResolvedPfilterConfig {
        model_path: path(model),
        compile,
        fit,
        objective,
        eval,
        forcing,
        scenario: ScenarioFlags { scenario: scenario.clone(), enable: enable.clone(), disable: disable.clone() },
        params: ParamFlags {
            param: param.iter().map(|p| (p.name.clone(), p.value)).collect(),
            params: params.clone(),
            table: table.iter().map(|t| (t.name.clone(), t.path.clone())).collect(),
        },
        data: data.clone(),
        obs: obs.clone(),
        dt: *dt,
        particles: *particles,
        replicates: *replicates,
        seed: *seed,
        binomial: binomial.unwrap_or_default(),
        max_substeps: pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET),
        time_format: *time_format,
        save_final_state: save_final_state.as_ref().map(path),
        engine: crate::env::get().engine_identity(crate::version::ENGINE_ID),
    };
    let outputs = PfilterOutputs {
        trace: trace.clone(),
        pf_health: pf_health.clone(),
        output: output.as_ref().map(path),
        save_paths: save_paths.as_ref().map(|p| (*n_paths, path(p))),
        save_filtering: save_filtering.as_ref().map(path),
        save_filtered_streams: save_filtered_streams.as_ref().map(path),
        save_prequential: save_prequential.clone(),
        score_from: score_from.clone(),
        save_samples: !*no_save_samples,
        no_store: *no_store,
    };
    (resolved, outputs, PfilterExecution { parallel: *parallel }, installed)
}
