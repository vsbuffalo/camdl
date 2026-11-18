//! `ResolvedSurveyConfig`: the one resolved configuration of a `camdl survey`,
//! from which both its stored leaf's identity and what each point evaluates
//! are built (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`,
//! phase 3 "The shape"; step 4 for `survey`).
//!
//! ```text
//! SurveyArgs + --fit's stored problem + defaults ──resolve──► ResolvedSurveyConfig
//!                                                               │           │
//!                                        survey_cas::resolve_survey      the points
//! ```
//!
//! [`resolve`] reads every argument once, with `SurveyArgs` destructured field
//! by field so an argument added to the command does not compile until it is
//! placed: into the resolved configuration, [`SurveyOutputs`] (what is
//! rendered), [`SurveyExecution`] (where the store is, the thread budget,
//! whether a stored leaf is recomputed) or the label (provenance). The input
//! mode is checked (a `--fit`, or `--data` with `--estimate`), the evaluation
//! counts validated, the `--fit` handle resolved once
//! ([`crate::fit::handle::resolve_fit_problem`]), and its `[config]
//! allow_degenerate_rates` and `allow_forcing_extrapolation` folded with the
//! flags. `--eval auto` and `--n-points 0` are resolved against the model by
//! the run, which reads them from here. Nothing after resolution reads the
//! command line.
//!
//! **What reaches the identity.** The model level hashes the resolved model
//! (compiled under [`ResolvedSurveyConfig::compile`]) and
//! [`ResolvedSurveyConfig::engine`], this build and the engine switches set;
//! the config level the evaluation (method, particles, replicates), the data,
//! the fixed values, the scenario, the objective, the training rows and the
//! two policies; the box level the sampled bounds and the number of points;
//! the seed level the seed.

use crate::args::types::EstimateBoundsSpec;
use crate::fit::handle::FitProblem;
use crate::run_meta::SurveyEvalMethod;

/// The resolved configuration of one `camdl survey`.
pub(crate) struct ResolvedSurveyConfig {
    /// The model file as named; the store leaf is named after it.
    pub model_path: String,
    /// The settings every compile of the model runs under (`--no-licm`).
    pub compile: crate::util::CompileSettings,
    /// The fit `--fit` names, read once: its problem supplies the sampled
    /// `[estimate]` bounds, `[fixed]`, the data, the scenario selection, the
    /// holdout, the objective and the two policies below. `None` in inline
    /// mode.
    pub fit: Option<FitProblem>,
    /// `--estimate NAME=LO:HI` (inline mode).
    pub estimate: Vec<EstimateBoundsSpec>,
    /// `--data` (inline mode).
    pub data: Option<String>,
    /// `--fixed NAME=VALUE`.
    pub fixed: Vec<(String, f64)>,
    /// `--scenario`.
    pub scenario: Option<String>,
    /// `--n-points`; 0 is scaled to the dimension by the run.
    pub n_points: usize,
    /// `--eval`; `auto` is resolved against the model by the run.
    pub eval: SurveyEvalMethod,
    pub eval_particles: usize,
    pub eval_replicates: usize,
    pub seed: u64,
    /// `--allow-degenerate-rates`, or the fit's `[config]
    /// allow_degenerate_rates`: carried on the compiled model every point is
    /// scored with.
    pub eval_settings: sim::EvalSettings,
    /// `--allow-forcing-extrapolation`, or the fit's `[config]
    /// allow_forcing_extrapolation`: the policy every point runs under and the
    /// leaf's record names.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
    /// The engine the identity folds: this build and the engine switches set.
    pub engine: runid::inputs::EngineIdentity,
}

/// What a survey renders beside its landscape.
pub(crate) struct SurveyOutputs {
    /// `--render`: the interactive pair-plot `landscape.html`.
    pub render: bool,
}

/// How a survey is computed and where it is stored, never what it holds.
pub(crate) struct SurveyExecution {
    /// `--output`: the store root (default `./results`).
    pub output: Option<String>,
    /// `--force`: recompute a leaf the store holds, quarantining it.
    pub force: bool,
    /// `--parallel` / CAMDL_PARALLEL: the scoped pool's size (0 = the global
    /// pool, every core).
    pub parallel: usize,
}

/// Resolve `survey`'s configuration from its arguments, the `--fit` handle's
/// problem and the defaults, under the compile settings `compile`; refuse
/// (through `out`) a combination that cannot run. Returns the values, the
/// label, and the forcing policy's installation, held for the command.
pub(crate) fn resolve(
    a: &crate::args::SurveyArgs,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
) -> (ResolvedSurveyConfig, SurveyOutputs, SurveyExecution, Option<String>, crate::forcing_extrapolation::Installed) {
    let crate::args::SurveyArgs {
        model,
        forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
        allow_degenerate_rates,
        fit,
        estimate,
        data,
        fixed,
        scenario,
        n_points,
        eval,
        eval_particles,
        eval_replicates,
        seed,
        render,
        output,
        label,
        force,
        parallel,
    } = a;
    // The input mode: a fit config, or `--data` with `--estimate` flags.
    if fit.is_none() {
        let Some(data) = data else {
            out.fail(
                "error: camdl survey requires either --fit FIT.toml \
                 (fit-aware mode) or --data DATA.tsv with --estimate \
                 NAME=LO:HI flags (inline mode).\n\
                 Got neither.",
            );
        };
        if estimate.is_empty() {
            out.fail(format!(
                "error: --data {} given without any --estimate flags. \
                 Pass --estimate NAME=LO:HI for each parameter to vary \
                 across the LHS box (repeat for multiple parameters).",
                data.display()
            ));
        }
    }
    if *eval_replicates == 0 {
        out.fail("error: --eval-replicates must be >= 1 (got 0).");
    }
    if *eval_particles == 0 && matches!(eval, SurveyEvalMethod::Pfilter | SurveyEvalMethod::Auto) {
        out.fail(
            "error: --eval-particles must be >= 1 \
             (in case --eval auto resolves to pfilter).",
        );
    }
    let label: Option<String> = label.as_deref().map(|raw| {
        crate::fit::validate_label(raw).unwrap_or_else(|e| {
            out.fail(format!("error: invalid --label: {}", e));
        })
    });

    // The fit `--fit` names: read once, here.
    let fit: Option<FitProblem> = fit.as_ref().map(|handle| {
        let f = crate::fit::handle::resolve_fit_problem(&handle.to_string_lossy()).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {}: {e}", handle.display()));
        });
        // The survey holds the fit's [fixed]: a sweep point's archive does not
        // hold the point's values, so it is refused rather than surveyed as
        // the un-swept problem.
        if let Err(e) = f.refuse_a_sweep_point() {
            out.fail(format!("error: {e}"));
        }
        f
    });
    let problem = fit.as_ref().map(|f| &f.config.problem);
    // A rate that collapses (a division by an empty compartment) is 0 under
    // `--allow-degenerate-rates` or the fit's `[config] allow_degenerate_rates`,
    // which `fit run` applied to the fit, and fails the trajectory otherwise;
    // under the particle filter a point where only some particles collapse
    // still scores, to a different value (gh#1235).
    let eval_settings = sim::EvalSettings {
        allow_degenerate_rates: *allow_degenerate_rates || problem.is_some_and(|p| p.config.allow_degenerate_rates),
    };
    // The forcing-extrapolation policy every point is evaluated under and the
    // leaf's record names, installed for the rest of the command.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::new(
        *allow_forcing_extrapolation || problem.is_some_and(|p| p.config.allow_forcing_extrapolation),
    );
    let installed = forcing.install();

    let resolved = ResolvedSurveyConfig {
        model_path: model.to_string_lossy().into_owned(),
        compile,
        fit,
        estimate: estimate.clone(),
        data: data.as_ref().map(|p| p.to_string_lossy().into_owned()),
        fixed: fixed.iter().map(|p| (p.name.clone(), p.value)).collect(),
        scenario: scenario.clone(),
        n_points: *n_points,
        eval: *eval,
        eval_particles: *eval_particles,
        eval_replicates: *eval_replicates,
        seed: *seed,
        eval_settings,
        forcing,
        engine: crate::env::get().engine_identity(crate::version::ENGINE_ID),
    };
    let execution = SurveyExecution {
        output: output.as_ref().map(|p| p.to_string_lossy().into_owned()),
        force: *force,
        parallel: *parallel,
    };
    (resolved, SurveyOutputs { render: *render }, execution, label, installed)
}
