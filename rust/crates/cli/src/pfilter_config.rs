//! `ResolvedPfilterConfig`: the one resolved configuration of a `camdl
//! pfilter`, from which the filter run's identity and what the particle filter
//! runs are both built.
//!
//! ```text
//! PfilterInvocation ──split──► declared │ execution │ exports │ request │ given
//!                                  │
//!                               resolve (here) ──► ResolvedPfilterConfig + ByProductRequests
//! ```
//!
//! [`resolve`] takes the declared inputs of `pfilter` (`crate::args::PfilterArgs`,
//! split by its `#[derive(Inputs)]`) apart in full, so an input added to the
//! command does not compile until it is placed here. The `--fit` argument is
//! read once ([`crate::fit_store::FitArg::read`]: a fit file names the problem
//! it states, read in process by the caller before the run store is chosen,
//! since its `store` key chooses it; a handle names a stored fit's problem,
//! read here through the run store), and its
//! `[config] allow_degenerate_rates` and `allow_forcing_extrapolation` are
//! folded with the flags into [`ResolvedPfilterConfig::eval`] and
//! [`ResolvedPfilterConfig::forcing`]. What reaches the filter run's identity
//! is `crate::pfilter_store::PfilterIdentity`, built from this value once the
//! model is compiled and the data read. A `--sweep-file` design is a list of
//! points, each the filter run of the single call at its values; the file
//! itself reaches no identity.

use std::path::PathBuf;

use crate::args::types::DataSpec;
use crate::fit::config_v2::LikelihoodObjective;
use crate::fit::fitted_model::FittedSettings;
use crate::fit_store::FitArg;

/// The resolved configuration of one `camdl pfilter`.
pub(crate) struct ResolvedPfilterConfig {
    /// The model file as named.
    pub model_path: String,
    /// The settings every compile of the model runs under (`--no-licm`).
    pub compile: crate::util::CompileSettings,
    /// The fit `--fit` names, read once: its problem supplies the objective,
    /// the scenario selection, the data when no `--data` is given, the written
    /// `[config] dt`, `holdout_after` and the two policies below.
    pub fit: Option<FitArg>,
    /// The likelihood scored: the fit's `ic_free`, else full.
    pub objective: LikelihoodObjective,
    /// `--allow-degenerate-rates`, or the fit's `[config]
    /// allow_degenerate_rates`: carried on the compiled model every particle
    /// runs.
    pub eval: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, or the fit's `[config]
    /// allow_forcing_extrapolation`.
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
    /// `--sweep-file`: the design whose points are scored, each as the single
    /// call at the point's values would score it.
    pub sweep_file: Option<PathBuf>,
    pub seed: u64,
    /// `--binomial`; `None` when not given.
    pub binomial: Option<sim::rng::BinomialAlgorithm>,
    /// `--save-final-state` was given: under the fit's `holdout_after` the
    /// bound rows are cut to the training rows, which the identity keys.
    pub save_final_state: bool,
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
    pub params: Vec<PathBuf>,
    pub table: Vec<(String, PathBuf)>,
}

/// The by-products of the run the command asks for, each with the path its
/// stored copy is exported to.
pub(crate) struct ByProductRequests {
    /// `--trace`: a path, or `-` for stdout.
    pub trace: Option<String>,
    /// `--pf-health`: a path, or `-` for stdout.
    pub pf_health: Option<String>,
    pub final_state: Option<PathBuf>,
    /// `--save-paths`, with `--n-paths` and the column selection.
    pub paths: Option<PathsRequest>,
    pub filtering: Option<PathBuf>,
    pub filtered_streams: Option<PathBuf>,
    /// `--save-prequential`: the stem of its three files.
    pub prequential: Option<String>,
    /// `--score-from`, as given.
    pub score_from: Option<String>,
    /// `--no-save-samples` negated.
    pub samples: bool,
}

/// `--save-paths`: how many smoothing paths, the file they are exported to,
/// and which columns they hold.
pub(crate) struct PathsRequest {
    /// `--n-paths`.
    pub n_paths: usize,
    pub path: PathBuf,
    /// `--no-flows` / `--columns`, not yet checked against the model.
    pub columns: crate::args::ColumnSelection,
}

/// How the filter computes, never what.
pub(crate) struct FilterExecution {
    /// `--parallel` / CAMDL_PARALLEL: the scoped pool's size (0 = the global
    /// pool, every core).
    pub parallel: usize,
    /// `--time-format`.
    pub time_format: crate::caltime_load::TimeFormat,
    /// `--pf-max-substeps`, else the engine's budget: a budget that trips
    /// refuses the run, so it never changes a stored result.
    pub max_substeps: u64,
}

/// Resolves `pfilter`'s configuration from its declared inputs, the `--fit`
/// handle's problem and the defaults; refuses (through `out`) a combination
/// that cannot run. `fit_file` is the fit file `--fit` names, already read;
/// a stored fit is read in the run store at `store_root`. Returns the
/// configuration, the by-products asked for and the forcing policy's
/// installation, held for the command.
pub(crate) fn resolve(
    declared: crate::args::PfilterDeclared,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
    fit_file: Option<FitArg>,
    from_manifest: Option<crate::resolve_file::FromManifest>,
    store_root: &std::path::Path,
) -> (ResolvedPfilterConfig, ByProductRequests, crate::forcing_extrapolation::Installed) {
    let (
        model,
        model_overrides,
        scenario,
        particles,
        inference,
        stream,
        data,
        fit,
        _from_manifest,
        replicates,
        sweep_file,
        trace,
        pf_health,
        save_final_state,
        save_paths,
        n_paths,
        path_columns,
        save_filtering,
        save_filtered_streams,
        save_prequential,
        score_from,
        no_save_samples,
    ) = declared.into_parts();
    let (params, param, table) = model_overrides.into_parts();
    let (scenario, enable, disable) = scenario.into_parts();
    let (dt, seed, allow_degenerate_rates, forcing_extrapolation, binomial) = inference.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let (obs,) = stream.into_parts();
    let path = |p: &PathBuf| p.to_string_lossy().into_owned();

    // The fit `--fit` names: a fit file as the caller read it, else a stored
    // fit, read here in the store this filter run is stored in.
    crate::fit_store::set_reader_root(store_root.to_path_buf());
    let fit_store = crate::camdl_store::open_existing(store_root);
    // A fit run named by a manifest is read by its run id, as `--fit <run>`.
    let fit = match &from_manifest {
        Some(m) => Some(PathBuf::from(m.fit_run(store_root).unwrap_or_else(|e| out.fail(format!("error: {e}"))))),
        None => fit,
    };
    let fit: Option<FitArg> = fit_file.or_else(|| fit.as_ref().map(|handle| {
        FitArg::read(&handle.to_string_lossy(), fit_store.as_ref()).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {}: {e}", handle.display()));
        })
    }));
    // A stored problem keeps its training lines only: filtering it would score
    // fewer rows than the same problem named by its fit file. Under
    // `--save-final-state` the filter binds the training rows alone either
    // way, so the two are one computation.
    if let Some(stored) = fit.as_ref().and_then(|f| f.stored.as_ref()).filter(|_| save_final_state.is_none()) {
        let held = stored.fit.held_out_rows();
        if !held.is_empty() {
            let rows: Vec<String> = held.iter().map(|(s, n)| format!("{s}: {n}")).collect();
            out.fail(format!(
                "error: --fit {}: the fit held rows out of training ({}), and a stored fit keeps no \
                 held-out values, so a filter run on it would score its training rows only. \
                 `--fit <the fit config>` scores every row; held-out scoring of stored fits \
                 arrives with the run store's scores (step 7).",
                fit.as_ref().map(|f| f.handle.as_str()).unwrap_or_default(),
                rows.join(", ")
            ));
        }
    }
    // A manifest's fit run gives its point estimate, `mle_params`, as the
    // first parameters file, which `--params` files given after it override.
    let params = match (&from_manifest, &fit) {
        (Some(m), Some(f)) => {
            let mle = point_estimate(f).unwrap_or_else(|e| out.fail(format!("error: --from-manifest {}: {e}", m.path().display())));
            std::iter::once(mle).chain(params).collect()
        }
        _ => params,
    };
    let problem = fit.as_ref().map(|f| &f.config.problem);
    // A rate that collapses (a division by an empty compartment) is 0 under
    // `--allow-degenerate-rates` or the fit's `[config] allow_degenerate_rates`,
    // which `fit run` applied to the fit, and kills the particle otherwise.
    let settings = problem
        .map(|p| FittedSettings::of(&p.config))
        .unwrap_or_default()
        .with_flags(allow_degenerate_rates, allow_forcing_extrapolation);
    let eval = crate::env::eval_settings(settings.allow_degenerate_rates);
    // A forcing evaluated outside its range holds its end value under the
    // override and is refused otherwise: the policy the filter runs under,
    // installed for the rest of the command.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::new(settings.allow_forcing_extrapolation);
    let installed = forcing.install();
    // The likelihood scored: the fit's own (`ic_free`), read through the
    // function `fit run` uses. Without `--fit` every observation is scored.
    let objective = problem.map(|p| p.likelihood_objective()).unwrap_or(LikelihoodObjective::Full);

    let resolved = ResolvedPfilterConfig {
        model_path: path(&model),
        compile,
        fit,
        objective,
        eval,
        forcing,
        scenario: ScenarioFlags { scenario, enable, disable },
        params: ParamFlags {
            param: param.iter().map(|p| (p.name.clone(), p.value)).collect(),
            params,
            table: table.iter().map(|t| (t.name.clone(), t.path.clone())).collect(),
        },
        data,
        obs,
        dt,
        particles,
        replicates,
        sweep_file,
        seed,
        binomial,
        save_final_state: save_final_state.is_some(),
    };
    let requests = ByProductRequests {
        trace,
        pf_health,
        final_state: save_final_state,
        paths: save_paths.map(|path| {
            let (no_flows, columns) = path_columns.into_parts();
            PathsRequest { n_paths, path, columns: crate::args::ColumnSelection { no_flows, columns } }
        }),
        filtering: save_filtering,
        filtered_streams: save_filtered_streams,
        prequential: save_prequential,
        score_from,
        samples: !no_save_samples,
    };
    (resolved, requests, installed)
}

/// The point estimate of the stored fit run `fit` names, `mle_params`, as
/// the path of its copy in the scratch directory `fit` holds open, which a
/// `--params` file is read from. Refused for a fit problem, and for a run
/// whose method stores no point estimate, such as an MCMC run, naming the
/// method.
fn point_estimate(fit: &FitArg) -> Result<PathBuf, String> {
    use crate::result_files::ResultFile;
    let opened = fit.stored.as_deref().ok_or_else(|| format!("{} is not a stored fit run", fit.handle))?;
    let (Some(run), Some(dir)) = (&opened.fit.run, opened.run_dir()) else {
        return Err(format!("{} names a fit problem, not a fit run", fit.handle));
    };
    let name = crate::result_files::fit_run::If2File::MleParams.stored();
    if !run.result_files().contains(&name.as_str()) {
        return Err(format!(
            "the fit run {} is a `{}` run, which stores no `mle_params`, the point estimate pfilter reads as its \
             parameters; name an if2 or nl-* run, or run `pfilter --fit {} --params FILE`",
            run.id().key().short12(),
            run.identity().chain.method.algorithm(),
            fit.handle
        ));
    }
    Ok(dir.join(name))
}

/// The execution inputs of `pfilter`, with the engine's budget filled in.
pub(crate) fn execution(core: &crate::args::PfilterCoreExecution) -> FilterExecution {
    FilterExecution {
        parallel: core.parallel,
        time_format: core.time_format,
        max_substeps: core.pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET),
    }
}
