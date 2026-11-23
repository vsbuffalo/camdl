//! `ResolvedProfileConfig`: the one resolved configuration of a `camdl
//! profile`, from which both every profile point's identity
//! (`crate::profile_store::ProfilePointIdentity`) and what each cell runs are
//! built.
//!
//! ```text
//! ProfileInvocation ──split──► declared │ execution │ exports │ request │ given
//!                                  │
//!                               resolve (here) ──► ResolvedProfileConfig
//! ```
//!
//! [`resolve`] takes the declared inputs of `profile` (`crate::args::ProfileArgs`,
//! split by its `#[derive(Inputs)]`) apart in full, so an input added to the
//! command does not compile until it is placed here. The algorithm and
//! backend are parsed, the IF2 defaults applied (`--iterations`, `--cooling`,
//! `--cooling-target-iters`), the `--fit` argument resolved once
//! ([`crate::fit_store::FitArg::read`]), its `[config] allow_degenerate_rates`
//! and `allow_forcing_extrapolation` folded with the flags, its nl-* method's
//! budget and `[method] starts` read, the starts rule in force decided
//! (`--starts`, else the fit's, qualified by `--starts-chains`) and `--seeds`
//! expanded. The removed spellings are refused before the split
//! ([`refuse_removed`]).

use crate::args::types::{RwSd, SweepSpec};
use crate::fit::fitted_model::FittedSettings;
use crate::fit_store::FitArg as FitProblem;
use crate::pfilter_config::ScenarioFlags;
use crate::profile::ProfileAlgo;

/// The resolved configuration of one `camdl profile`.
pub(crate) struct ResolvedProfileConfig {
    /// The model file as named.
    pub model_path: String,
    /// The settings every compile of the model runs under (`--no-licm`).
    pub compile: crate::util::CompileSettings,
    /// The `--algorithm` / `--backend` combination, from the method registry.
    pub method: &'static crate::fit::methods::InferenceMethod,
    /// What each cell runs: IF2, or an nl-* optimizer.
    pub algo: ProfileAlgo,
    /// The fit `--fit` names, read once: its problem supplies the estimated
    /// set and its bounds and priors, `[fixed]`, the scenario selection, the
    /// data when no `--data` is given, the written `[config] dt`, the holdout,
    /// the objective, `[[simplex_groups]]` and the two policies below.
    pub fit: Option<FitProblem>,
    /// `--allow-degenerate-rates`, or the fit's `[config]
    /// allow_degenerate_rates`: carried on the compiled model every cell runs.
    pub eval: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, or the fit's `[config]
    /// allow_forcing_extrapolation`: the policy every cell runs under and every
    /// record names.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
    /// `--scenario`, `--enable`, `--disable`, as given.
    pub scenario: ScenarioFlags,
    /// `--fixed NAME=VALUE`, as given.
    pub fixed_cli: Vec<(String, f64)>,
    /// `--fixed-file`, as given.
    pub fixed_files: Vec<std::path::PathBuf>,
    /// `--table NAME=FILE`, as given.
    pub tables: Vec<(String, std::path::PathBuf)>,
    /// `--dt`, checked against the model's declared step by the one dt rule.
    pub dt: Option<f64>,
    /// `--sweep`: the focal axes and their values.
    pub sweep: Vec<SweepSpec>,
    /// `--particles` on an IF2 cell (required there); 0 on an nl-* cell,
    /// which runs no particle filter.
    pub particles: usize,
    /// `--iterations`, else the profile default.
    pub iterations: usize,
    /// `--cooling` reached at `--cooling-target-iters`, else the defaults: the
    /// schedule a fit's `cooling` would mean.
    pub cooling: sim::inference::if2::CoolingSchedule,
    /// `--n-starts`: how many starts each cell runs.
    pub n_starts: usize,
    /// The starts rule in force, before its default is decided against the
    /// estimated set.
    pub starts: DeclaredStarts,
    /// An nl-* cell's optimizer budget.
    pub nlopt: NloptBudget,
    /// `--rw-sd`: the estimated set and each IF2 step, or `auto`.
    pub rw_sd: Option<RwSd>,
    /// `--seed`: draws the chain starts and the `[estimate]` start fall-back.
    pub seed: u64,
    /// `--seeds`, else the single `--seed`. One set of leaves each.
    pub seeds: Vec<u64>,
    /// `--pf-max-substeps`, as given (the method level keys it as given).
    pub pf_max_substeps: Option<u64>,
    /// `--binomial`, else BTPE.
    pub binomial: sim::rng::BinomialAlgorithm,
}

/// The starts rule as declared: `--starts`, else the fit's `[method]
/// starts`, qualified by `--starts-chains`; `None` leaves the default to be
/// decided against the estimated set.
pub(crate) struct DeclaredStarts {
    pub rule: Option<crate::fit::starts::ChainStarts>,
    /// Where it came from, for the startup line.
    pub from: &'static str,
    /// `--allow-nonconverged-source`: a source fit whose verdict is not
    /// converged is used rather than refused. It decides whether the profile
    /// starts, never what it computes.
    pub allow_nonconverged_source: bool,
}

/// An nl-* cell's `xtol_rel` and evaluation budget: the fit's nl-* method's,
/// else the profile defaults.
pub(crate) struct NloptBudget {
    pub tolerance: f64,
    pub max_evals: usize,
    /// Whether they are the fit's.
    pub from_fit: bool,
}

/// How `profile` computes, never what: the thread budget and how the data's
/// time column is read.
pub(crate) struct CellExecution {
    /// `--parallel` / CAMDL_PARALLEL: the scoped pool's size (0 = the global
    /// pool, every core).
    pub parallel: usize,
    /// `--time-format`: a format the column does not hold is refused.
    pub time_format: crate::caltime_load::TimeFormat,
}

/// The execution inputs that decide only whether a profile may run.
pub(crate) struct ProfileGates {
    /// `--backend`: each algorithm runs on one backend, so it is checked
    /// against the algorithm and selects nothing.
    pub backend: Option<String>,
    /// `--allow-nonconverged-source`.
    pub allow_nonconverged_source: bool,
}

/// Refuses the spellings `profile` accepts only to refuse them, each naming
/// what to write instead: `--pmmh-*`, the `--init` family, `--param`,
/// `--output-dir`, and `--data` and `--obs`, which the problem `--fit` names
/// sets. They are skip-class inputs, which the split does not carry, so they
/// are read here, before it.
pub(crate) fn refuse_removed(a: &crate::args::ProfileArgs, out: &crate::out::Out) {
    use crate::profile::PROFILE_PMMH_REMOVED;
    // The `--pmmh-*` flags went with per-cell PMMH. They are trapped so a
    // stale script is answered with the reason rather than clap's
    // "unexpected argument", before any file is read.
    for (flag, given) in [
        ("--pmmh-steps", a._removed_pmmh_steps.as_deref()),
        ("--pmmh-particles", a._removed_pmmh_particles.as_deref()),
        ("--pmmh-rho", a._removed_pmmh_rho.as_deref()),
    ] {
        if let Some(raw) = given {
            out.fail(format!("error: {flag} {raw} is no longer accepted: {PROFILE_PMMH_REMOVED}"));
        }
    }
    // gh#1354: data and stream selection beside `--fit`'s problem.
    if !a.data.is_empty() {
        out.fail("error: --data binds data beside --fit, whose problem names its own data; \
            put the data in the fit file's [data] (gh#1354).");
    }
    if a.obs.is_some() {
        out.fail("error: --obs selects a stream beside --fit, whose problem binds its own streams; \
            bind the streams in the fit file's [data.observations] (gh#1354).");
    }
    // M-1 break per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
    // §"Migration": fail loudly on removed flags before any work.
    a.model_overrides.check_removed_flags("profile");
    // gh#889: the `--init` family, answered with the same `--starts`
    // replacements `fit run` gives, from the same builder.
    if let Some(msg) = crate::fit::removed_flags_message(
        "profile",
        crate::fit::starts_family_removed_lines(
            a._removed_init.as_deref(),
            a._removed_posterior.as_deref(),
            a._removed_mle.as_deref(),
            a._removed_params.as_deref(),
        ),
    ) {
        out.error(crate::out::CommandError {
            code: crate::out::ExitCode(2),
            ..crate::out::CommandError::new(format!("error: {msg}"))
        })
        .exit();
    }
}

/// The fit file `--fit` names, read and parsed once; `None` for a stored fit's
/// handle, which is read from the run store once the store is known.
pub(crate) fn read_fit_file(fit: Option<&std::path::Path>) -> Result<Option<FitProblem>, String> {
    let Some(path) = fit else { return Ok(None) };
    let text = path.to_string_lossy();
    if !crate::fit_store::is_config_file(&text) {
        return Ok(None);
    }
    FitProblem::read(&text, None).map(Some)
}

/// Resolve `profile`'s configuration from its declared inputs, the `--fit`
/// handle's problem and the defaults, under the compile settings `compile`;
/// refuse (through `out`) a combination that cannot run. `--backend` and
/// `--allow-nonconverged-source` are execution inputs, given apart: each
/// decides only whether the profile may run. `fit_file` is the fit file
/// `--fit` names, already read ([`read_fit_file`]); a stored fit is read in the
/// run store at `store_root`. Returns the values and the forcing policy's
/// installation, held for the command.
pub(crate) fn resolve(
    declared: crate::args::ProfileDeclared,
    gates: ProfileGates,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
    fit_file: Option<FitProblem>,
    store_root: &std::path::Path,
) -> (ResolvedProfileConfig, crate::forcing_extrapolation::Installed) {
    use crate::profile::PROFILE_PMMH_REMOVED;
    let (
        model,
        model_overrides,
        scenario,
        inference,
        fit,
        sweep,
        particles,
        iterations,
        n_starts,
        starts,
        starts_chains,
        cooling,
        cooling_target_iters,
        rw_sd,
        seeds,
        algorithm,
    ) = declared.into_parts();
    let (fixed_cli, fixed_files, table) = model_overrides.into_parts();
    let (scenario, enable, disable) = scenario.into_parts();
    let (dt, seed, allow_degenerate_rates, forcing_extrapolation, pf_max_substeps, binomial) = inference.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let ProfileGates { backend, allow_nonconverged_source } = gates;

    // Parse the CLI strings into the typed registry entry (the string boundary);
    // all downstream dispatch reads the typed FitAlgorithm / InferenceBackend.
    let algo_name = algorithm.as_deref().unwrap_or("if2");
    let backend_name = backend.as_deref().unwrap_or("chain_binomial");
    // Refused on any backend, before the registry's own (algorithm, backend)
    // check, so `--algorithm pmmh --backend ode` gets the same answer.
    if crate::fit::methods::parse_algorithm(algo_name) == Some(crate::run_meta::FitAlgorithm::Pmmh) {
        out.fail(format!("error: --algorithm pmmh is not accepted: {PROFILE_PMMH_REMOVED}"));
    }
    let method = crate::fit::methods::parse_combo(algo_name, backend_name).unwrap_or_else(|msg| {
        out.fail(format!("error: {}", msg));
    });
    let algo = match method.algorithm {
        crate::run_meta::FitAlgorithm::If2 => ProfileAlgo::If2,
        crate::run_meta::FitAlgorithm::NlSbplx => ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx),
        crate::run_meta::FitAlgorithm::NlBobyqa => ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa),
        // gh#275: the same per-cell `optimize_cell` the derivative-free NLopt
        // algorithms use, taking the gradient branch. `profile` reaches it
        // without further wiring because it loads the model through
        // `util::load_model_with`, which always compiles the full IR — the
        // WrtPop state-Jacobian the forward sensitivity reads is present.
        crate::run_meta::FitAlgorithm::NlLbfgs => ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs),
        other => {
            out.fail(format!(
                "error: --algorithm = \"{}\" is not yet supported for `camdl profile`. \
                 Currently supported: if2 (chain_binomial), \
                 nl-sbplx (ode), nl-bobyqa (ode), nl-lbfgs (ode).",
                other
            ));
        }
    };
    // Registry-driven caveat for methods carrying a status_note (Beta methods,
    // plus Stable methods with usage guidance — e.g. nl-*). After the profile
    // dispatch, so a refused algorithm is not first shown a caveat.
    crate::fit::methods::emit_status_banner(out, method.algorithm, method.backend, None);

    // The IF2-only flags are refused on an nl-* cell, and `--particles` is
    // required on an IF2 one; past this point each cell reads only its own.
    let particles = match algo {
        ProfileAlgo::If2 => particles.unwrap_or_else(|| {
            out.fail("error: --particles is required for an if2 profile");
        }),
        ProfileAlgo::Nlopt(_) => {
            let given = crate::profile::If2Flags {
                particles,
                iterations,
                cooling,
                cooling_target_iters,
                pf_max_substeps,
                binomial,
                rw_sd: rw_sd.as_ref(),
            };
            if let Err(e) = crate::profile::refuse_if2_flags_for_nlopt(&given, algo_name) {
                out.fail(format!("error: {e}"));
            }
            0
        }
    };
    // A profile's problem is a stored fit problem: the one a fit file poses,
    // or a stored fit's. Flags do not pose one.
    // gh#1354: a problem posed by flags, and `--data` or `--obs` beside `--fit`.
    let Some(fit) = fit else {
        out.fail(
            "error: a profile is of a fit problem: pass `--fit FIT.toml` (the problem the file poses, \
             stored as `camdl fit run` stores it) or `--fit @RUN` (a stored fit's). A problem posed \
             by `--data` and the other flags alone is not accepted yet (gh#1354).",
        );
    };
    let fit = Some(fit);

    // The fit `--fit` names: the file already read, or a stored fit, read
    // here from the store this profile is stored in.
    crate::fit_store::set_reader_root(store_root.to_path_buf());
    let fit_store = crate::camdl_store::open_existing(store_root);
    let fit: Option<FitProblem> = fit.as_ref().map(|handle| {
        let f = fit_file.unwrap_or_else(|| FitProblem::read(&handle.to_string_lossy(), fit_store.as_ref()).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {}: {e}", handle.display()));
        }));
        out.say(format_args!("profile: using --fit '{}' for priors / bounds / [fixed]", f.handle));
        f
    });
    let problem = fit.as_ref().map(|f| &f.config.problem);
    // A rate that collapses (a division by an empty compartment, a power to
    // NaN) is 0 under `--allow-degenerate-rates` or the fit's `[config]
    // allow_degenerate_rates`, and an error otherwise. It changes every cell's
    // log-likelihood, so it is applied before any cell runs and keyed in the
    // base identity.
    let settings = problem
        .map(|p| FittedSettings::of(&p.config))
        .unwrap_or_default()
        .with_flags(allow_degenerate_rates, allow_forcing_extrapolation);
    let eval = crate::env::eval_settings(settings.allow_degenerate_rates);
    // A forcing evaluated outside its range holds its end value under
    // `--allow-forcing-extrapolation` or the fit's `[config]
    // allow_forcing_extrapolation` (gh#1235), and is refused otherwise: the
    // policy every cell runs under and every record names, installed for the
    // rest of the command and keyed in the base identity from this value.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::new(settings.allow_forcing_extrapolation);
    let installed = forcing.install();

    // An nl-* cell's optimizer budget: the fit's own, when its [method] is an
    // nl-* optimizer (so a profile of that fit optimizes each cell as the fit
    // did), else the profile's defaults. Recorded in the method identity
    // either way.
    let fit_method = fit.as_ref().and_then(|f| f.config.inference.method.as_ref());
    let (tolerance, max_evals, from_fit) = crate::profile::nlopt_cell_budget(fit_method.map(|m| &m.algorithm));
    // Where each cell's chains begin: `--starts <spec>` (the rule grammar and
    // typed `ChainStarts` `fit run` takes, gh#889), else the fit's `[method]
    // starts` (its path sources anchored at the file, as `fit run` reads
    // them), else `fit run`'s default rule, decided once the estimated set is
    // known. `--starts-chains` restricts the rule in force, the file's
    // included, to chains of its `from_posterior` source, by the function
    // `fit run` applies it with (gh#1056).
    let file_starts = fit_method.and_then(|m| m.starts.clone());
    let (rule_in_force, from): (Option<crate::fit::starts::ChainStarts>, &'static str) = match (starts, &file_starts) {
        (Some(rule), _) => (Some(rule), "from --starts"),
        (None, Some(rule)) => (Some(rule.clone()), "from the --fit config's [method] starts"),
        (None, None) => (None, ""),
    };
    let rule = match starts_chains {
        Some(subset) => Some(
            crate::fit::starts::apply_starts_chains(rule_in_force, &subset, "profile")
                .unwrap_or_else(|e| out.fail(format!("error: {e}"))),
        ),
        None => rule_in_force,
    };
    // `--seeds` wins; the default is the single `--seed`.
    let seeds: Vec<u64> = match seeds {
        Some(spec) => spec.expand(),
        None => vec![seed],
    };
    if seeds.is_empty() {
        out.fail("error: --seeds expanded to empty list");
    }

    let resolved = ResolvedProfileConfig {
        model_path: model.to_string_lossy().into_owned(),
        compile,
        method,
        algo,
        fit,
        eval,
        forcing,
        scenario: ScenarioFlags { scenario, enable, disable },
        fixed_cli: fixed_cli.iter().map(|p| (p.name.clone(), p.value)).collect(),
        fixed_files,
        tables: table.iter().map(|t| (t.name.clone(), t.path.clone())).collect(),
        dt,
        sweep,
        particles,
        iterations: iterations.unwrap_or(crate::profile::PROFILE_DEFAULT_ITERATIONS),
        cooling: crate::profile::profile_cooling_schedule(cooling, cooling_target_iters),
        n_starts,
        starts: DeclaredStarts { rule, from, allow_nonconverged_source },
        nlopt: NloptBudget { tolerance, max_evals, from_fit },
        rw_sd,
        seed,
        seeds,
        pf_max_substeps,
        binomial: binomial.unwrap_or_default(),
    };
    (resolved, installed)
}
