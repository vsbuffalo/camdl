//! `ResolvedProfileConfig`: the one resolved configuration of a `camdl
//! profile`, from which both every profile point's identity and what each
//! cell runs are built (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "The shape"; step 4 for `profile`).
//!
//! ```text
//! ProfileArgs + --fit's stored problem + defaults ──resolve──► ResolvedProfileConfig
//!                                                                │            │
//!                                    profile_cas::resolve_profile_point     the cells
//! ```
//!
//! [`resolve`] reads every argument once, with `ProfileArgs` destructured field
//! by field so an argument added to the command does not compile until it is
//! placed: into the resolved configuration, [`ProfileOutputs`] (the table
//! written outside the store), [`ProfileExecution`] (the thread budget) or the
//! label (provenance). The removed spellings are refused here, before any
//! file is read. The algorithm and backend are parsed, the IF2 defaults
//! applied (`--iterations`, `--cooling`, `--cooling-target-iters`), the
//! `--fit` handle resolved once ([`crate::fit::handle::resolve_fit_problem`]),
//! its `[config] allow_degenerate_rates` and `allow_forcing_extrapolation`
//! folded with the flags, its nl-* method's budget and `[method] starts` read,
//! the starts rule in force decided (`--starts`, else the fit's, qualified by
//! `--starts-chains`) and `--seeds` expanded. Nothing after resolution reads
//! the command line: `cmd_profile` hands the values to the run, where the
//! arguments are not in scope.
//!
//! **What reaches the identity.** A point's `profile` level hashes the
//! resolved model (compiled under [`ResolvedProfileConfig::compile`]), the
//! data, the problem (fixed values, priors, the fit's config text, the
//! objective, the training rows, the two policies), the sweep grid and
//! [`ResolvedProfileConfig::engine`], this build and the engine switches set;
//! its `point` level the cell's focal values; its `method` level the cell's
//! algorithm and settings; its `seed` and `start` levels the seed and start.

use crate::args::types::{DataSpec, RwSd, SweepSpec};
use crate::fit::handle::FitProblem;
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
    /// `--data`, as given; empty means the fit's `[data]`.
    pub data: Vec<DataSpec>,
    /// `--obs`: the stream a single `--data PATH` binds.
    pub obs: Option<String>,
    /// `--dt`, checked against the model's declared step by the one dt rule.
    pub dt: Option<f64>,
    /// `--time-format`: how the data's time column is read.
    pub time_format: crate::caltime_load::TimeFormat,
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
    /// The engine the identity folds: this build and the engine switches set.
    pub engine: runid::inputs::EngineIdentity,
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

/// The table a profile prints, outside the store.
pub(crate) struct ProfileOutputs {
    /// `--output`: the file, else stdout.
    pub output: Option<String>,
    pub format: crate::args::ProfileTableFormat,
}

/// How `profile` computes, never what: the thread budget.
pub(crate) struct ProfileExecution {
    /// `--parallel` / CAMDL_PARALLEL: the scoped pool's size (0 = the global
    /// pool, every core).
    pub parallel: usize,
}

/// Resolve `profile`'s configuration from its arguments, the `--fit` handle's
/// problem and the defaults, under the compile settings `compile`; refuse
/// (through `out`) a combination that cannot run. Returns the values, the
/// label, and the forcing policy's installation, held for the command.
pub(crate) fn resolve(
    a: &crate::args::ProfileArgs,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
) -> (ResolvedProfileConfig, ProfileOutputs, ProfileExecution, Option<String>, crate::forcing_extrapolation::Installed) {
    use crate::profile::{PROFILE_NO_ESTIMATED_SET, PROFILE_PMMH_REMOVED};
    let crate::args::ProfileArgs {
        model,
        model_overrides,
        scenario: crate::args::ScenarioArgs { scenario, enable, disable },
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
        sweep,
        particles: _,
        iterations,
        n_starts,
        starts,
        starts_chains,
        allow_nonconverged_source,
        _removed_init,
        _removed_posterior,
        _removed_mle,
        _removed_params,
        _removed_pmmh_steps,
        _removed_pmmh_particles,
        _removed_pmmh_rho,
        cooling: _,
        cooling_target_iters: _,
        rw_sd,
        output,
        format,
        seeds,
        label,
        algorithm,
        backend,
    } = a;
    let crate::args::InferenceModelOverrides { fixed_cli, fixed_files, table, _removed_param: _ } = model_overrides;

    // The `--pmmh-*` flags went with per-cell PMMH. They are trapped so a
    // stale script is answered with the reason rather than clap's
    // "unexpected argument", before any file is read.
    for (flag, given) in [
        ("--pmmh-steps", _removed_pmmh_steps.as_deref()),
        ("--pmmh-particles", _removed_pmmh_particles.as_deref()),
        ("--pmmh-rho", _removed_pmmh_rho.as_deref()),
    ] {
        if let Some(raw) = given {
            out.fail(format!("error: {flag} {raw} is no longer accepted: {PROFILE_PMMH_REMOVED}"));
        }
    }
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
        ProfileAlgo::If2 => a.particles.unwrap_or_else(|| {
            out.fail("error: --particles is required for an if2 profile");
        }),
        ProfileAlgo::Nlopt(_) => {
            if let Err(e) = crate::profile::refuse_if2_flags_for_nlopt(a, algo_name) {
                out.fail(format!("error: {e}"));
            }
            0
        }
    };
    let label: Option<String> = label.as_deref().map(|raw| {
        crate::fit::validate_label(raw).unwrap_or_else(|e| {
            out.fail(format!("error: invalid --label: {}", e));
        })
    });
    // M-1 break per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
    // §"Migration": fail loudly on removed flags before any work.
    model_overrides.check_removed_flags("profile");
    // gh#889: the `--init` family, answered with the same `--starts`
    // replacements `fit run` gives, from the same builder.
    if let Some(msg) = crate::fit::removed_flags_message(
        "profile",
        crate::fit::starts_family_removed_lines(
            _removed_init.as_deref(),
            _removed_posterior.as_deref(),
            _removed_mle.as_deref(),
            _removed_params.as_deref(),
        ),
    ) {
        out.error(crate::out::CommandError {
            code: crate::out::ExitCode(2),
            ..crate::out::CommandError::new(format!("error: {msg}"))
        })
        .exit();
    }
    // What names the estimated set: `--rw-sd` when given, else the --fit
    // config's [estimate]. Without either there is nothing that says what a
    // cell estimates.
    if rw_sd.is_none() && fit.is_none() {
        out.fail(PROFILE_NO_ESTIMATED_SET);
    }

    // The fit `--fit` names: read once, here.
    let fit: Option<FitProblem> = fit.as_ref().map(|handle| {
        let f = crate::fit::handle::resolve_fit_problem(&handle.to_string_lossy()).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {}: {e}", handle.display()));
        });
        out.say(format_args!("profile: using --fit '{}' for priors / bounds / [fixed]", f.handle));
        f
    });
    let problem = fit.as_ref().map(|f| &f.config.problem);
    // A rate that collapses (a division by an empty compartment, a power to
    // NaN) is 0 under `--allow-degenerate-rates` or the fit's `[config]
    // allow_degenerate_rates`, and an error otherwise. It changes every cell's
    // log-likelihood, so it is applied before any cell runs and keyed in the
    // base identity.
    let eval = sim::EvalSettings {
        allow_degenerate_rates: *allow_degenerate_rates || problem.is_some_and(|p| p.config.allow_degenerate_rates),
    };
    // A forcing evaluated outside its range holds its end value under
    // `--allow-forcing-extrapolation` or the fit's `[config]
    // allow_forcing_extrapolation` (gh#1235), and is refused otherwise: the
    // policy every cell runs under and every record names, installed for the
    // rest of the command and keyed in the base identity from this value.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::new(
        *allow_forcing_extrapolation || problem.is_some_and(|p| p.config.allow_forcing_extrapolation),
    );
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
        (Some(rule), _) => (Some(rule.clone()), "from --starts"),
        (None, Some(rule)) => (Some(rule.clone()), "from the --fit config's [method] starts"),
        (None, None) => (None, ""),
    };
    let rule = match starts_chains {
        Some(subset) => Some(
            crate::fit::starts::apply_starts_chains(rule_in_force, subset, "profile")
                .unwrap_or_else(|e| out.fail(format!("error: {e}"))),
        ),
        None => rule_in_force,
    };
    // `--seeds` wins; the default is the single `--seed`.
    let seeds: Vec<u64> = match seeds {
        Some(spec) => spec.expand(),
        None => vec![*seed],
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
        scenario: ScenarioFlags { scenario: scenario.clone(), enable: enable.clone(), disable: disable.clone() },
        fixed_cli: fixed_cli.iter().map(|p| (p.name.clone(), p.value)).collect(),
        fixed_files: fixed_files.clone(),
        tables: table.iter().map(|t| (t.name.clone(), t.path.clone())).collect(),
        data: data.clone(),
        obs: obs.clone(),
        dt: *dt,
        time_format: *time_format,
        sweep: sweep.clone(),
        particles,
        iterations: iterations.unwrap_or(crate::profile::PROFILE_DEFAULT_ITERATIONS),
        cooling: crate::profile::profile_cooling_schedule(a),
        n_starts: *n_starts,
        starts: DeclaredStarts { rule, from, allow_nonconverged_source: *allow_nonconverged_source },
        nlopt: NloptBudget { tolerance, max_evals, from_fit },
        rw_sd: rw_sd.clone(),
        seed: *seed,
        seeds,
        pf_max_substeps: *pf_max_substeps,
        binomial: binomial.unwrap_or_default(),
        engine: crate::env::get().engine_identity(crate::version::ENGINE_ID),
    };
    let outputs = ProfileOutputs { output: output.as_ref().map(|p| p.to_string_lossy().into_owned()), format: *format };
    (resolved, outputs, ProfileExecution { parallel: *parallel }, label, installed)
}
