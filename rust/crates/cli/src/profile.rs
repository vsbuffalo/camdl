//! `camdl profile` — profile likelihood via parallel per-cell optimization
//! (IF2 or deterministic nl-* MLE, per `--algorithm`).
//!
//! For one or more focal parameters, fix them at a grid of values and
//! run IF2 to maximise over the remaining parameters at each grid
//! point. The profile likelihood shows how the MLE changes as you move
//! the focal parameter(s) — revealing identifiability, confidence
//! intervals, and parameter interactions. 2D profiles (two `--sweep`
//! flags) produce a likelihood surface suitable for contour plotting.
//!
//! Each `(grid point × seed × start)` is one profile point on the run store
//! ([`crate::profile_store::ProfilePoint`]), whose identity holds the problem,
//! the focal values, the method, the start's values and a seed derived from
//! them, and nothing about the grid. The points are claimed together, those
//! already stored are reused, and the rest are computed in parallel, each
//! written in place under its claim. The profile table is rendered from the
//! points the invocation asked for.

use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2, IF2Config},
        particle_filter::Observation,
        BoundObs, ChainBinomialProcess, MultiStreamObsModel,
    },
};
use std::collections::HashMap;
use std::sync::Arc;



/// Why `camdl profile` has no per-cell PMMH, and what to run instead.
///
/// A profile is the maximum over the nuisance parameters λ of the
/// likelihood L(ψ, λ) at each grid value of the focal parameter ψ, so each
/// cell is an optimisation. A per-cell MCMC chain with ψ pinned samples
/// p(λ | ψ, y) correctly, but the number it reported — the log-likelihood
/// at the highest-posterior sample the chain visited — is neither that
/// maximum (under an informative prior the MAP sample sits below it, by an
/// amount that depends on how well a short chain explored) nor the marginal
/// posterior p(ψ | y), which needs each cell's marginal likelihood
/// p(y | ψ). Named once so the `--algorithm pmmh` refusal and the removed
/// `--pmmh-*` flags give the same answer.
pub(crate) const PROFILE_PMMH_REMOVED: &str = "per-cell PMMH was removed from `camdl profile`. \
    A profile likelihood is a maximum over the other parameters at each grid \
    value, so each cell is fitted by maximisation: use `--algorithm if2` \
    (chain_binomial) or an `nl-*` optimizer (`nl-sbplx`, `nl-bobyqa`, \
    `nl-lbfgs`; `--backend ode`). The per-cell chain reported the \
    log-likelihood at its highest-posterior sample, which is neither the \
    profile likelihood nor the marginal posterior of the swept parameter. For \
    a posterior of that parameter, fit once with `camdl fit run` (PMMH or \
    PGAS) and read its marginal from the draws.";

/// Per-cell optimizer choice. `--algorithm if2 --backend chain_binomial`
/// (the default) keeps the historical PF-based per-cell MLE; the new
/// `--algorithm nl-* --backend ode` paths run NLopt deterministic MLE
/// against `compute_ode_loglik`. See gh#47 for the dispatch rationale
/// and the proposal for the (algorithm, backend) matrix.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProfileAlgo {
    If2,
    Nlopt(sim::inference::deterministic::NloptAlgorithm),
}

impl ProfileAlgo {
    fn method_kind(self) -> crate::run_meta::FitAlgorithm {
        match self {
            ProfileAlgo::If2  => crate::run_meta::FitAlgorithm::If2,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx) =>
                crate::run_meta::FitAlgorithm::NlSbplx,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa) =>
                crate::run_meta::FitAlgorithm::NlBobyqa,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs) =>
                crate::run_meta::FitAlgorithm::NlLbfgs,
        }
    }
}

// Observation family resolution lives in `crate::util::resolve_data_specs`
// (gh#90). Profile's previous `resolve_obs_family` (gh#38) is subsumed
// there: `--data PATH --obs <family-root>` matches every IR obs whose
// name starts with `<root>_`, and `--data NAME=PATH` (gh#90 named form)
// also expands NAME as a family root. Same semantics, single dispatch.

/// Normalized coordinates of grid cell `gi` — each focal value mapped to
/// `[0, 1]` over its axis's grid range, so the nearest-to-best ordering is
/// scale-fair across axes with very different magnitudes. Ordering only; it
/// never touches a loglik or a leaf.
fn profile_cell_norm(
    grid_points: &[Vec<(usize, f64)>],
    axis_ranges: &[(f64, f64)],
    gi: usize,
) -> Vec<f64> {
    grid_points[gi].iter().enumerate().map(|(k, &(_, v))| {
        let (mn, mx) = axis_ranges[k];
        if (mx - mn).abs() < 1e-12 { 0.0 } else { (v - mn) / (mx - mn) }
    }).collect()
}

/// Squared Euclidean distance between two normalized cell coordinate vectors
/// (see [`profile_cell_norm`]). Squared — ordering only needs the monotone
/// comparison.
fn profile_cell_dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// What to pass when `profile` has no observation data bound. Named once so the
/// anchored refusal and the ordinary missing-data refusal cannot drift apart.
const PROFILE_SUPPLY: &str = "bind the streams in the fit file's [data.observations].";

/// `--iterations` when not given: IF2 iterations per cell.
pub(crate) const PROFILE_DEFAULT_ITERATIONS: usize = 50;
/// `--cooling` when not given.
const PROFILE_DEFAULT_COOLING: f64 = 0.95;
/// An nl-* cell's `xtol_rel` when `--fit` names no nl-* method: looser than a
/// fit's default, because a profile explores a landscape rather than
/// certifying one optimum.
const PROFILE_NLOPT_TOLERANCE: f64 = 1e-4;
/// An nl-* cell's evaluation budget when `--fit` names no nl-* method.
const PROFILE_NLOPT_MAX_EVALS: usize = 1500;

/// An nl-* cell's `(tolerance, max_evals, from the fit)`: the `--fit`
/// method's own when it is an nl-* optimizer, else the profile defaults.
pub(crate) fn nlopt_cell_budget(
    fit_method: Option<&crate::fit::config_v2::Algorithm>,
) -> (f64, usize, bool) {
    use crate::fit::config_v2::Algorithm;
    match fit_method {
        Some(Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c)) => {
            (c.tolerance, c.max_evals, true)
        }
        _ => (PROFILE_NLOPT_TOLERANCE, PROFILE_NLOPT_MAX_EVALS, false),
    }
}

/// The IF2 settings as given on `profile`'s command line, each `None` when
/// not given, and `--rw-sd`: what [`refuse_if2_flags_for_nlopt`] checks.
#[derive(Clone, Copy)]
pub(crate) struct If2Flags<'a> {
    pub particles: Option<usize>,
    pub iterations: Option<usize>,
    pub cooling: Option<f64>,
    pub cooling_target_iters: Option<usize>,
    pub pf_max_substeps: Option<u64>,
    pub binomial: Option<sim::rng::BinomialAlgorithm>,
    pub rw_sd: Option<&'a crate::args::types::RwSd>,
}

/// Refuse the IF2-only flags on an nl-* profile: its cells run no particle
/// filter and perturb nothing, so each would be hashed into the run and read
/// by nothing. `--rw-sd` still names the estimated set there, so names
/// (`NAME=auto`, or `auto`) are accepted and a step size is refused.
pub(crate) fn refuse_if2_flags_for_nlopt(flags: &If2Flags<'_>, algorithm: &str) -> Result<(), String> {
    let If2Flags { particles, iterations, cooling, cooling_target_iters, pf_max_substeps, binomial, rw_sd } = *flags;
    let given: Vec<&str> = [
        (particles.is_some(), "--particles"),
        (iterations.is_some(), "--iterations"),
        (cooling.is_some(), "--cooling"),
        (cooling_target_iters.is_some(), "--cooling-target-iters"),
        (pf_max_substeps.is_some(), "--pf-max-substeps"),
        (binomial.is_some(), "--binomial"),
    ]
    .into_iter()
    .filter_map(|(on, flag)| on.then_some(flag))
    .collect();
    if !given.is_empty() {
        return Err(format!(
            "{} {} IF2 settings, and an `{algorithm}` cell runs no particle filter and \
             perturbs nothing, so {} would change nothing. Drop {}.",
            given.join(", "),
            if given.len() == 1 { "is an" } else { "are" },
            if given.len() == 1 { "it" } else { "they" },
            if given.len() == 1 { "it" } else { "them" },
        ));
    }
    if let Some(crate::args::types::RwSd::Map(m)) = rw_sd {
        let mut with_sd: Vec<&str> =
            m.iter().filter(|(_, v)| v.is_some()).map(|(k, _)| k.as_str()).collect();
        with_sd.sort();
        if !with_sd.is_empty() {
            return Err(format!(
                "--rw-sd gives a random-walk step for {}, and an `{algorithm}` cell perturbs \
                 nothing. Name the estimated parameters without a step (`NAME=auto`), or \
                 use `--rw-sd auto`.",
                with_sd.join(", ")));
        }
    }
    Ok(())
}

/// With `--fit`, an explicit `--rw-sd NAME=…` list that leaves out an
/// `[estimate]` parameter (not swept, not pinned) would hold that parameter at
/// its start in every cell, unannounced, while the config says it is
/// estimated. The message names the parameters and both ways out.
pub(crate) fn refuse_rw_sd_list_short_of_estimate(
    fit_estimate: &indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    exclude: &std::collections::HashSet<String>,
    listed: &HashMap<String, Option<f64>>,
    fit_path: &str,
) -> Option<String> {
    let left_out: Vec<&str> = fit_estimate.keys()
        .filter(|n| !exclude.contains(*n) && !listed.contains_key(*n))
        .map(String::as_str)
        .collect();
    if left_out.is_empty() {
        return None;
    }
    Some(format!(
        "--rw-sd lists the estimated parameters, and it leaves out {} of the --fit \
         config '{fit_path}' [estimate], which every cell would then hold at its start. \
         Add {} to --rw-sd (`NAME=auto` takes the automatic step), drop --rw-sd to \
         estimate the config's [estimate], or move {} to [fixed] in a copy of the config.",
        left_out.join(", "),
        if left_out.len() == 1 { "it" } else { "them" },
        if left_out.len() == 1 { "it" } else { "them" },
    ))
}

/// A profile cell's IF2 cooling schedule: the one `fit run` uses, from the
/// same two numbers, so `--cooling c` means what `cooling = c` means in a
/// fit — `c` reached at `--cooling-target-iters` (default 50), whatever
/// `--iterations` is.
pub(crate) fn profile_cooling_schedule(
    cooling: Option<f64>,
    cooling_target_iters: Option<usize>,
) -> sim::inference::if2::CoolingSchedule {
    sim::inference::if2::CoolingSchedule {
        fraction: cooling.unwrap_or(PROFILE_DEFAULT_COOLING),
        target_iters: cooling_target_iters.unwrap_or(sim::inference::if2::CoolingSchedule::POMP_TARGET_ITERS),
    }
}

/// How cool an IF2 run's perturbation got: the cooling factor of its last
/// iteration, `last`, the one multiplier that iteration's diagnostics apply
/// to every parameter's step (`ParamIterDiag::effective_sd` is each step
/// times it). It is the same for every parameter, so it is a property of the
/// run and has no unit; a mean of the parameters' SDs would mix their
/// coordinates.
pub(crate) fn cooling_final(
    schedule: &sim::inference::if2::CoolingSchedule,
    n_assimilations: usize,
    last: &sim::inference::if2::IF2IterResult,
) -> f64 {
    // `iteration` counts from 0; `multiplier_at_iter` takes iterations completed.
    schedule.multiplier_at_iter(n_assimilations, last.iteration + 1)
}

/// The IF2 configuration every IF2 profile cell runs with, built in one
/// place from the resolved configuration: the particles, the iterations, the
/// cooling schedule a fit's `cooling` would mean (`profile_cooling_schedule`),
/// the step, the degeneracy budget and the binomial sampler.
#[allow(clippy::too_many_arguments)]
pub(crate) fn profile_if2_config(
    n_particles: usize,
    n_iterations: usize,
    cooling: sim::inference::if2::CoolingSchedule,
    pf_max_substeps: Option<u64>,
    binomial: sim::rng::BinomialAlgorithm,
    dt: f64,
    t_start: f64,
    simplex_groups: Vec<sim::inference::if2::SimplexGroup>,
    steps: Vec<sim::inference::if2::If2Step>,
    skip_first_obs_from_loglik: bool,
) -> IF2Config {
    IF2Config {
        n_particles,
        n_iterations,
        cooling,
        dt,
        t_start,
        simplex_groups,
        steps,
        skip_first_obs_from_loglik,
        max_substeps: pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET), // gh#241
        binomial, // gh#747
    }
}

/// `camdl profile`: its inputs split by class, the configuration resolved
/// from the declared inputs ([`crate::profile_config::resolve`]), and the run
/// built from it, which does not see the arguments.
pub fn cmd_profile(inv: crate::args::ProfileInvocation) {
    // profile has no `--no-color`: its painters follow the environment and
    // whether each stream is a terminal.
    let out = crate::out::Out::new(false);
    crate::profile_config::refuse_removed(&inv.args, &out);
    let link_root = crate::link_tree::link_root(inv.args.link_root.as_deref());
    let bindings = crate::bindings::claim(&out, inv.args.manifest.as_deref(), link_root, inv.args.label.as_deref());
    // A fit file is read once, here, before the store is chosen: its `store`
    // key places the profile, and the fit problem it stores, in that fit's
    // store (run spec §2.1).
    let fit_file = crate::profile_config::read_fit_file(inv.args.fit.as_deref()).unwrap_or_else(|e| {
        out.fail(format!("error: --fit {}: {e}", inv.args.fit.as_deref().unwrap_or(std::path::Path::new("")).display()))
    });
    let file_store = fit_file.as_ref().and_then(|f| f.config.problem.store.as_ref()).map(std::path::PathBuf::from);
    let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
    let (declared, no_licm) = declared.into_parts();
    let crate::args::ProfileInvocationExecution { args: execution, no_ir_cache } = execution;
    let crate::args::ProfileExecution {
        model_overrides: (),
        scenario: (),
        inference: crate::args::ProfileCoreExecution { parallel, forcing_extrapolation: (), time_format },
        allow_nonconverged_source,
        backend,
    } = execution;
    let compile = crate::util::CompileSettings::resolve(no_licm, no_ir_cache);
    let (exports,) = exports.into_parts();
    let ((), (), _inference_exports, output, format, _manifest, _link_root) = exports.into_parts();
    let (request,) = request.into_parts();
    let ((), (), _inference_request, label, force, root) = request.into_parts();
    let request = crate::camdl_store::store_request_with_fit_file(label, force, root, file_store);
    let execution_record = crate::args::ProfileInvocationExecution {
        args: crate::args::ProfileExecution {
            model_overrides: (),
            scenario: (),
            inference: crate::args::ProfileCoreExecution { parallel, forcing_extrapolation: (), time_format },
            allow_nonconverged_source,
            backend: backend.clone(),
        },
        no_ir_cache,
    };
    let gates = crate::profile_config::ProfileGates { backend, allow_nonconverged_source };
    let (config, _forcing_route) = crate::profile_config::resolve(declared, gates, compile, &out, fit_file, &crate::camdl_store::root_of(&request));
    run_profile(
        &out,
        config,
        TableExport { output, format },
        crate::profile_config::CellExecution { parallel, time_format },
        StoreSide { request, given, execution: execution_record, bindings },
    );
}

/// The store-class inputs of one `profile`, and what its records hold.
struct StoreSide {
    request: store::StoreRequest,
    given: store::Given,
    execution: crate::args::ProfileInvocationExecution,
    /// `--manifest` and `--link-root`, claimed before the store was opened.
    bindings: crate::bindings::Bindings,
}

/// The profile table's export: `--output`, else stdout, in `--format`.
struct TableExport {
    output: Option<std::path::PathBuf>,
    format: crate::args::ProfileTableFormat,
}

/// Whether the resolver's estimated set and values are the stored problem's:
/// the same `[estimate]`, in order, and each `[fixed]` value of the problem
/// equal, bit for bit, to the resolved model's. A parameter the problem
/// neither estimates nor fixes takes the compile's value or its scenario's,
/// which only the resolver computes (the problem stores the scenario's
/// selection, not its values), so there is no second value to compare.
fn resolution_matches_problem(
    estimated: &indexmap::IndexSet<String>,
    resolved_value: impl Fn(&str) -> Option<f64>,
    problem_estimate: &indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    problem_fixed: &indexmap::IndexMap<String, f64>,
) -> Result<(), String> {
    if !estimated.iter().eq(problem_estimate.keys()) {
        return Err(format!(
            "the parameters resolved as estimated ({}) are not the stored problem's [estimate] ({})",
            estimated.iter().cloned().collect::<Vec<_>>().join(", "),
            problem_estimate.keys().cloned().collect::<Vec<_>>().join(", ")));
    }
    for (name, &value) in problem_fixed {
        let resolved = resolved_value(name);
        if resolved.map(f64::to_bits) != Some(value.to_bits()) {
            return Err(format!(
                "{name} resolved to {} where the stored problem's [fixed] holds {value}",
                resolved.map_or("no value".to_string(), |v| v.to_string())));
        }
    }
    Ok(())
}

/// A fit problem's `[fixed]` values, its scenario's values expanded into
/// them, and its `[estimate]` names, as the resolver reads a fit config.
fn named_fixed_and_estimate(
    problem: &crate::fit::config_v2::Problem,
    model: &ir::Model,
) -> Result<(indexmap::IndexMap<String, f64>, indexmap::IndexSet<String>), String> {
    let mut fixed = problem.fixed.clone();
    fixed.expand_from_scenario(model, &problem.estimate)?;
    let fixed = fixed.resolve_with_model(model)
        .map_err(|e| format!("--fit toml [fixed].resolve_with_model: {e}"))?;
    Ok((fixed, problem.estimate.keys().cloned().collect()))
}

/// The cells of one resolved `profile`, their stored points and the table.
fn run_profile(
    out: &crate::out::Out,
    config: crate::profile_config::ResolvedProfileConfig,
    table_export: TableExport,
    execution: crate::profile_config::CellExecution,
    store_side: StoreSide,
) {
    let crate::profile_config::ResolvedProfileConfig {
        model_path: ir_path,
        compile,
        method,
        algo: profile_algo,
        fit,
        eval,
        forcing,
        scenario: scenario_flags,
        fixed_cli: fixed_cli_vec,
        fixed_files: fixed_files_vec,
        tables,
        dt: dt_flag,
        sweep,
        particles: n_particles,
        iterations: n_iterations,
        cooling: cooling_schedule,
        n_starts,
        starts: crate::profile_config::DeclaredStarts { rule: declared_starts, from: starts_from, allow_nonconverged_source },
        nlopt: crate::profile_config::NloptBudget { tolerance: nl_tolerance, max_evals: nl_max_evals, from_fit: nl_from_fit },
        rw_sd,
        seed: seed_base,
        seeds,
        pf_max_substeps,
        binomial,
    } = config;
    let crate::profile_config::CellExecution { parallel, time_format } = execution;
    let cooling = cooling_schedule.fraction;
    // The model as named, whose stem labels the points.
    let ir_path_named = ir_path.clone();

    // ── The store, and the problem every point is of ──────────────────────
    let StoreSide { request, given, execution: execution_record, bindings } = store_side;
    let store = crate::camdl_store::open(&request)
        .unwrap_or_else(|e| out.fail(format!("error: the run store at {}: {e}", crate::camdl_store::root_of(&request).display())));
    let info = store::RunInfo::new(&given, &execution_record, &request)
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let named_fit = fit.unwrap_or_else(|| out.fail("error: internal: a profile resolved without --fit"));
    let posed = crate::profile_store::pose_problem(
        out, &store, &info, &named_fit, &ir_path, compile,
        &crate::profile_store::ProblemOverrides {
            fixed: &fixed_cli_vec,
            fixed_files: &fixed_files_vec,
            swept: &sweep.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            tables: &tables,
            scenario: &scenario_flags,
            allow_degenerate_rates: eval.allow_degenerate_rates,
            allow_forcing_extrapolation: forcing.allowed(),
            dt: dt_flag,
        },
    ).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let crate::profile_store::PosedProblem { dep: problem_dep, fit: posed_fit, model_ir, presentation } = posed;
    // `nl-lbfgs` climbs the ODE forward-sensitivity gradient, which reads the
    // compile with the state Jacobian; every other cell runs the problem's.
    let jacobian = matches!(profile_algo, ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs))
        .then(|| crate::profile_store::jacobian_compile(out, &store, &info, &posed_fit, presentation.clone(), &model_ir, compile)
            .unwrap_or_else(|e| out.fail(format!("error: {e}"))));
    let (ir_path, jacobian_model) = match jacobian {
        Some((ir, dep)) => (ir, Some(dep)),
        None => (model_ir, None),
    };
    let fit = Some(posed_fit);
    let focal_names: Vec<String> = sweep.iter().map(|s| s.name.clone()).collect();

    struct FocalGrid { name: String, values: Vec<f64>, param_idx: usize }
    let mut focal_grids: Vec<FocalGrid> = Vec::new();

    let rw_sd_auto = matches!(rw_sd, Some(crate::args::types::RwSd::Auto));
    let rw_sd_map_raw: HashMap<String, Option<f64>> = match &rw_sd {
        Some(crate::args::types::RwSd::Map(m)) => m.clone(),
        Some(crate::args::types::RwSd::Auto) | None => HashMap::new(),
    };

    // Load model (pre-resolution). `mut` because the gh#34
    // `[estimate].start` fall-back below seeds values into
    // `model_pre.parameters[i].value` before the resolver call —
    // same pattern as `fit/runner.rs:144-180`. It is the stored problem's
    // compile, or for `nl-lbfgs` the compile with the state Jacobian.
    let (mut model_pre, _) = crate::util::load_ir(&ir_path)
        .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    // Every point runs this model, so it must be the stored compile the
    // points' identities name, through their problem or the Jacobian compile.
    let stored_model = match &jacobian_model {
        Some(dep) => dep.digest(),
        None => fit.as_ref().and_then(|f| f.stored.as_ref()).map(|o| o.fit.model.digest())
            .unwrap_or_else(|| out.fail("error: internal: the profiled problem is not stored")),
    };
    if let Err(e) = crate::camdl_store::require_stored_model(&model_pre, stored_model, "profile", &ir_path) {
        out.fail(format!("error: {e}"));
    }
    // The model before any value is written into it, for the partition.
    let model_declared = model_pre.clone();

    // ── Optional --fit toml resolution (gh#73) ──────────────────────
    //
    // When `--fit <path.toml>` is supplied, the fit-toml is the source
    // of truth for priors, bounds, and the default fixed list — same
    // shape `camdl fit run` reads. The toml's `[fixed]` block is
    // pre-processed against the model so scenario-driven fixed values
    // come through; the resulting IndexMap is fed to the unified
    // resolver as `fit_toml_fixed` (per 2026-05-25 CLI UX rev 2).
    //
    // The estimate IndexMap returned here is the canonical prior
    // source used downstream by `resolve_priors_with_precedence`. When
    // `--fit` is absent the map is empty and the resolver falls
    // through to model-IR priors (tier 2) for every parameter.
    // The fit config's holdout, applied to the bound data below by the function
    // `fit run` applies it with: `holdout_after` cuts the training rows,
    // `[data.holdout]` files are checked to lie after them.
    let mut fit_holdout_after: Option<crate::fit::config_v2::TimeSpecToml> = None;
    let mut fit_holdout_files: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
    let (fit_estimate, fit_toml_fixed_indexmap, fit_scenario, fit_simplex_groups,
         objective, fit_fixed_via):
        (indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
         indexmap::IndexMap<String, f64>,
         (Option<String>, Vec<String>, Vec<String>),
         Vec<crate::fit::config_v2::SimplexGroup>,
         crate::fit::config_v2::LikelihoodObjective,
         Result<Vec<(String, crate::roles::FitFixedVia)>, String>)
        = if let Some(f) = fit.as_ref() {
        let fit_cfg = f.config.problem.clone();
        // How each [fixed] name reached the config, for the partition.
        let fixed_via = fit_cfg.fixed.declarations(&model_pre);
        // Resolve [fixed] (file load, scenario, inline overlay) the
        // same way `camdl fit run` does.
        let mut fixed_cfg = fit_cfg.fixed.clone();
        fixed_cfg.expand_from_scenario(&model_pre, &fit_cfg.estimate)
            .unwrap_or_else(|e| {
                out.fail(format!("error: {}", e));
            });
        let fixed_resolved = fixed_cfg.resolve_with_model(&model_pre)
            .unwrap_or_else(|e| {
                out.fail(format!("error: --fit toml [fixed].resolve_with_model: {}", e));
            });
        // The likelihood every cell scores is the fit's own (`ic_free`), read
        // through the same function `fit run` uses, so a profile curve is on
        // the scale of the fit it profiles.
        let objective = fit_cfg.likelihood_objective();
        // The fit's holdout, applied once the data are loaded (gh#1059).
        if let Some(data) = fit_cfg.data.as_ref() {
            fit_holdout_after = data.holdout_after.clone();
            fit_holdout_files = data.holdout.clone().unwrap_or_default();
        }
        (fit_cfg.estimate, fixed_resolved,
         (fit_cfg.scenario, fit_cfg.enable, fit_cfg.disable),
         fit_cfg.simplex_groups, objective, fixed_via)
    } else {
        (indexmap::IndexMap::new(), indexmap::IndexMap::new(),
         (None, Vec::new(), Vec::new()), Vec::new(),
         crate::fit::config_v2::LikelihoodObjective::Full, Ok(Vec::new()))
    };
    // A rate that collapses is 0 under `--allow-degenerate-rates` or the
    // fit's `[config] allow_degenerate_rates` (resolved with the forcing
    // policy, `profile_config::resolve`), keyed in the base identity below.
    let allow_degenerate_rates = eval.allow_degenerate_rates;

    // The simulator every cell integrates with, under the fitting rule: the
    // method's backend, the model's declared step (a `--dt` or `--fit` file's
    // written `[config] dt` that differs is refused), and, on chain-binomial,
    // `--binomial`, else the `--fit` method's sampler, else BTPE: a fit's
    // estimate is profiled on the binomial draws it was estimated on. The
    // cells' own configurations take plain numbers.
    let fit_written_dt: Option<f64> =
        fit.as_ref().and_then(|f| crate::fit::config_v2::written_config_dt(f.config_text()));
    let backend = crate::args::types::ForwardBackend::from(method.backend);
    let fit_binomial = match (&fit, backend) {
        (Some(f), crate::args::types::ForwardBackend::ChainBinomial) => {
            f.binomial().unwrap_or_else(|e| { out.fail(format!("error: --fit: {e}")); })
        }
        _ => None,
    };
    let (simulator, _) = crate::simulator::Simulator::resolve(
        crate::simulator::ModelSimulate::of(&model_pre.simulation),
        crate::simulator::WrittenSimulator {
            dt: dt_flag,
            config_dt: fit_written_dt,
            binomial: binomial.or(fit_binomial),
            ..Default::default()
        },
        crate::simulator::Basis::Fitting {
            backend,
            run: "this profile",
            file: crate::dt_rule::FitFile::Flag,
        },
    )
    .unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
    let dt = simulator.dt().get();
    let binomial = simulator.binomial();

    // An nl-* cell's optimizer budget and the starts rule in force are
    // resolved (`profile_config::resolve`): the `--fit` method's own, else the
    // profile defaults; `--starts`, else the fit's `[method] starts`.

    // The scenario this profile runs under: the fit config's top-level
    // `scenario`/`enable`/`disable` combined with the CLI flags by the rule
    // `pfilter --fit` uses.
    let crate::util::ScenarioSelection {
        scenario: scenario_name,
        enable: adhoc_enable,
        disable: adhoc_disable,
    } = crate::util::combine_fit_scenario(
        crate::util::ScenarioChoice {
            scenario: scenario_flags.scenario.as_deref(),
            enable: &scenario_flags.enable,
            disable: &scenario_flags.disable,
        },
        crate::util::ScenarioChoice {
            scenario: fit_scenario.0.as_deref(),
            enable: &fit_scenario.1,
            disable: &fit_scenario.2,
        },
    ).unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
    // The rules `fit run` applies to the same file: a scenario (from the fit
    // config or `--scenario`) that sets or scales an estimated parameter, or
    // moves a [fixed] parameter to another value, is refused, not resolved.
    // The profiled (swept) parameters are left out of the estimated set: the
    // grid replaces a scenario's value for them, which is reported at info
    // further down, not refused.
    if let (Some(scenario), Some(f)) = (scenario_name.as_deref(), fit.as_ref()) {
        let fit_path_str: &str = &f.handle;
        let estimated: Vec<&str> = fit_estimate.keys()
            .map(|k| k.as_str())
            .filter(|k| !sweep.iter().any(|s| s.name == *k))
            .collect();
        let refusals: Vec<String> = [
            crate::params_resolver::scenario_estimate_collision(
                &model_pre, scenario, &estimated,
                crate::params_resolver::EstimatedBy::FitToml(fit_path_str),
            ),
            crate::params_resolver::scenario_fixed_collision(
                &model_pre, scenario, &fit_toml_fixed_indexmap, fit_path_str,
            ),
        ].into_iter()
            .filter_map(|r| r.unwrap_or_else(|e| { out.fail(format!("error: {e}")); }))
            .collect();
        if !refusals.is_empty() {
            out.fail(format!("error: {}", refusals.join("\n\nerror: ")));
        }
    }

    // A `--fixed` the scenario sets or scales would override it in every cell:
    // refused, not resolved (gh#571).
    if let Err(msg) = crate::params_resolver::refuse_fixed_flag_over_scenario(
        &model_pre, scenario_name.as_deref(), &fixed_cli_vec,
    ) {
        out.fail(format!("error: {msg}"));
    }

    // A single-point starts rule (`--starts from_params=<toml>`,
    // `--starts from_mle=<fit>`) loads one parameter point. Its values are
    // the base value of every parameter they name — so a parameter the
    // profile neither fixes nor estimates is HELD at the file's value — and
    // the chain-start draw below reads the same rule again to start the
    // estimated (nuisance) parameters there. They enter the resolver as
    // their own tier, just above the model default
    // (`ValueSource::Starts`), so a later tier replacing one is reported and
    // recorded. Without them the resolver would fire `UnsetRequired` on a
    // parameter with no DSL default even though the user named a file
    // carrying it. See `starts_values_from_init_method`.
    let starts_values: Option<(crate::params_resolver::StartsOrigin, Vec<(String, f64)>)> =
        declared_starts.as_ref().map(starts_values_from_init_method).transpose()
            .map(Option::flatten)
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });

    // The `[estimate]` start fall-back, by the function `fit run` uses
    // (run-spec §1.4): an entry with a `start` carries it, one without
    // (`beta = {}`) gets a draw within its bounds, recorded as
    // `estimate_start_drawn`. Applied BEFORE the resolver, which refuses a
    // parameter with no value. A swept parameter takes its value from the
    // sweep at every point, so it needs no start of its own: it is never
    // drawn from a prior, and a swept parameter still without a value
    // afterwards is held at the first grid value, only so the resolver has
    // one to carry.
    let swept: std::collections::HashSet<String> =
        sweep.iter().map(|s| s.name.clone()).collect();
    let seeded = crate::fit::runner::seed_estimate_starts(
        &mut model_pre, &fit_estimate, seed_base, &swept);
    for sw in &sweep {
        if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == sw.name) {
            if p.value.resolved_value().is_none() {
                if let Some(&first) = sw.grid.expand().first() {
                    p.value = p.value.with_value(first);
                }
            }
        }
    }

    // ── Resolve --data and --obs against the IR's observation list ──
    //
    // gh#90: polymorphic `--data` is the primary multi-stream surface.
    //   --data PATH         → single-stream, optionally narrowed by --obs.
    //   --data NAME=PATH    → multi-stream, joint scoring of every bound
    //                         stream (repeatable; one flag per stream).
    //
    // gh#38 family resolution survives: in the single-PATH form,
    // `--obs <root>` matches every IR obs whose name starts with
    // `<root>_`, so a single wide TSV covers an indexed
    // `cases[s,a]` family with one `--data <file>` flag plus
    // `--obs cases`.
    //
    // fit-toml fallback: when no `--data` flags supplied AND `--fit`
    // is, read `[data]` / `[data.observations]` from the toml. CLI
    // flags always win when both are supplied.
    //
    // This sits BEFORE the resolver and the compile because the observation
    // anchors below fold over exactly these bindings (gh#616), and
    // `CompiledModel::new` refuses a model that still carries one. Binding is a
    // pure read of the args plus `model.observations`, which neither the
    // `[estimate].start` seeding above nor the resolver below touches.
    //
    // `None` means nothing was bound at all (no `--data`, no `--fit`). That is
    // reported LATER, where it always was — after the compile — so the
    // capability gate keeps first say on an invocation that has a more specific
    // problem than a missing flag.
    let model_obs_names: Vec<String> = model_pre.observations.iter()
        .map(|o| o.name.clone()).collect();
    let bound_streams: Option<Vec<(String, std::path::PathBuf)>> = fit.as_ref().map(|f| {
        crate::pfilter::data_observations_of(&f.config.problem, &model_pre, &f.handle)
            .unwrap_or_else(|e| out.fail(format!("error: --fit {}: {e}", f.handle)))
    });

    // gh#616: profile binds observation data, so it RESOLVES a model's
    // observation anchors instead of letting `CompiledModel::new` refuse them.
    // The window is folded from the bindings just resolved, so it is the same
    // window every profile point scores.
    //
    // Everything downstream reads the substituted model: `model_pre` is the
    // resolver's input and `resolved.model` is what compiles, and this path has
    // exactly one `load_model` and one `CompiledModel::new`. A second fresh load
    // of the compiled IR would re-introduce the unresolved marker (gh#616
    // regression, commit 7af5c9fa).
    //
    // With nothing bound, an ANCHORED model is refused here rather than
    // deferred: it has no horizon at all, so nothing later can be computed from
    // it. An unanchored model falls through untouched.
    // A fit config's `[simulate] from` moves the start as `fit run` does.
    let start = fit.as_ref()
        .map(|f| crate::obs_anchor::config_start(&f.config.problem, &model_pre))
        .transpose()
        .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })
        .flatten();
    match &bound_streams {
        Some(bound) => {
            crate::obs_anchor::resolve_from_bindings(out, &mut model_pre, bound, dt,
                fit_holdout_after.as_ref(), start.as_ref())
                .unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
        }
        None => {
            if let Some(start) = start.as_ref() {
                match crate::obs_anchor::apply_start(&mut model_pre, start, None) {
                    Ok(r) => crate::obs_anchor::report(out, &[r], &model_pre),
                    Err(e) => out.fail(format!("error: {e}")),
                }
            }
            if let Some(msg) =
                crate::obs_anchor::refuse_without_data(&model_pre, PROFILE_SUPPLY)
            {
                out.fail(format!("error: {msg}"));
            }
        }
    }
    // gh#561: every profile point runs the filter, whose window is the
    // observation times — the model horizon is never read here. A scenario's
    // own `simulate { to }` therefore moves nothing, so refuse it rather than
    // discard it silently (the same guard as `pfilter` and `camdl predict`).
    //
    // It runs AFTER the substitution above so it compares two RESOLVED numbers:
    // an anchored model and an anchored scenario that resolve to the same time
    // are a no-op and must pass, and two that differ must be named with times a
    // reader can check against the data.
    if let Err(e) = crate::util::refuse_scenario_horizon(
        &model_pre, scenario_name.as_deref(), "profile",
        "each profile point scores through a particle filter at the observation \
         times, so the window comes from the data, not the model horizon",
    ) {
        out.fail(format!("error: {e}"));
    }

    // Run the unified resolver.
    // The resolver reads the fit as named, before the command line's
    // `--fixed` and `--fixed-file` were written into the stored problem, so it
    // reports each value they replace as the other commands do (run spec
    // §1.3). The cells compute from its model, and the points are keyed on
    // the stored problem, so the two must agree: `resolution_matches_problem`
    // refuses any difference in the estimated set or in a `[fixed]` value.
    let (named_fixed, named_estimate) = named_fixed_and_estimate(&named_fit.config.problem, &model_pre)
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    drop(named_fit);
    // The stored compile has the problem's tables bound, `--table` among them.
    let external = crate::external_inputs::ExternalBindings::none();
    let resolved = crate::params_resolver::resolve_parameters_with_starts(
        crate::params_resolver::ParameterInputs {
            model: &model_pre,
            scenario: scenario_name.as_deref(),
            adhoc_enable: &adhoc_enable,
            adhoc_disable: &adhoc_disable,
            scenario_inline_name: None,
            scenario_inline_set: &[],
            scenario_inline_scale: &[],
            point_overrides: &[],
            fixed_cli: &fixed_cli_vec,
            fixed_files: &fixed_files_vec,
            swept: &focal_names,
            fit_toml_fixed: &named_fixed,
            fit_toml_estimate: &named_estimate,
            external: &external,
        },
        starts_values.as_ref().map(|(origin, values)|
            crate::params_resolver::StartsValues { origin, values }),
    ).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    let mut resolved = resolved;
    let resolved_value = |name: &str| resolved.model.parameters.iter().find(|p| p.name == name).and_then(|p| p.value.resolved_value());
    if let Err(e) = resolution_matches_problem(&resolved.estimate_set, resolved_value, &fit_estimate, &fit_toml_fixed_indexmap) {
        out.fail(format!("error: internal: {e}"));
    }
    resolved.attribute_seeded_starts(&seeded);
    crate::params_resolver::emit_notices(out, &[&resolved]);
    // The grid writes each focal value straight over the parameter, so a
    // scenario's value for a focal parameter is replaced at every point: said
    // once here, for the run, not per grid point.
    let focal_notices: crate::out::notice::Notices = match scenario_name.as_deref() {
        Some(scenario) => {
            let focal: Vec<&str> = focal_names.iter().map(|s| s.as_str()).collect();
            crate::params_resolver::scenario_focal_notices(&model_pre, scenario, &focal)
                .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })
                .into_iter()
                .collect()
        }
        None => crate::out::notice::Notices::default(),
    };
    out.notices(&focal_notices);
    let model = resolved.model.clone();

    // Maintain the downstream `fit_toml_fixed: HashMap` view that the
    // rest of profile.rs reads (focal-vs-fixed conflict check, etc.).
    let fit_toml_fixed: HashMap<String, f64> = fit_toml_fixed_indexmap
        .iter().map(|(k, &v)| (k.clone(), v)).collect();

    let compiled = Arc::new(CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { out.fail(format!("{:?}", e)); })
        .with_eval_settings(crate::env::eval_settings(allow_degenerate_rates)));
    let base_params = compiled.default_params.clone();

    // Reject overdispersed models on the ODE backend (and any other
    // backend / model-capability mismatch). The `validate_combo` call
    // above is structural-only; this check sees the actual model.
    if let Err(msg) = crate::fit::methods::check_model_capabilities(
        method.backend, &compiled,
    ) {
        out.fail(format!("error: {}", msg));
    }

    // A conditional objective (`ic_free = true`) is admitted per method by the
    // check `fit run` applies: IF2 drops y₁'s term, the deterministic ODE
    // likelihood cannot, so an `nl-*` profile of such a fit is refused rather
    // than run on the full likelihood under the fit's name.
    if objective.skips_first_observation() {
        let init_law = if compiled.has_init_law {
            crate::fit::methods::InitLaw::Declared
        } else {
            crate::fit::methods::InitLaw::Absent
        };
        if let Err(msg) =
            crate::fit::methods::validate_ic_free(profile_algo.method_kind(), init_law)
        {
            out.fail(format!("error: the --fit config sets ic_free = true. {msg}"));
        }
        out.say("profile: ic_free = true — every cell scores log p(y₂:T | y₁, θ), \
            the fit's conditional likelihood");
    }

    // The missing-data refusal, reported here rather than where the binding was
    // resolved, so every gate above keeps first say (see the binding block).
    let bound_streams = bound_streams.unwrap_or_else(|| {
        out.fail(format!("error: profile needs observation data to score: {PROFILE_SUPPLY}"));
    });
    if bound_streams.is_empty() {
        out.fail("error: zero streams resolved from --data / --fit toml.");
    }

    // Resolve the bound observation streams (BY SOURCE) and load each one's
    // per-observation values + aux via the single shared seam that `fit run`
    // and `pfilter` also route through (`fit::runner::resolve_and_load_obs_streams`).
    // The seam filters `model.observations` by `source`, dispatches long-form vs
    // wide loading (holes + aux), resolves each projection, and runs the per-
    // stream origin / first-window guards — so a stratified family bound by its
    // ROOT (`[data.observations] prevalence = FILE`, source `prevalence`) fans
    // out to every leaf exactly as `fit run` binds it. `bound_streams` (the
    // CLI/toml key-space bindings) is mapped to the by-source `effective` map at
    // the boundary; it is retained below for the data tables the problem
    // holds, by binding name (leaf name / family root). The typo guard (a
    // binding key matching no source or name) lives in the adapter.
    log::debug!("profile: the cells bind the stored problem's data");
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: compiled.model.simulation.t_start,
        format: time_format,
    };
    let effective = crate::fit::runner::data_bindings_to_effective(&model, &bound_streams)
        .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    let mut streams = crate::fit::runner::resolve_and_load_obs_streams(
        &model, &compiled, &effective, &time_opts,
    ).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    // The fit config's holdout, applied to the bound data (`--data` or the
    // config's own) by the function `fit run` applies it with, so every cell
    // fits the rows the fit fits: `holdout_after` keeps the rows ending at or
    // before the cutoff, and `[data.holdout]` files leave the training rows
    // whole and must end after them.
    let (training_partition, partition_lines) = crate::fit::runner::apply_holdout_declaration(
        &mut streams, fit_holdout_after.as_ref(), &fit_holdout_files, &model,
        &time_opts,
    ).unwrap_or_else(|e| { out.fail(format!("error: --fit config: {e}")); });
    partition_lines.say();
    let streams = streams;

    if streams.len() > 1 {
        out.say(format_args!(
            "profile: {} streams bound (joint loglik = sum across all): {}",
            streams.len(),
            streams.iter().map(|s| s.name.as_str())
                .collect::<Vec<_>>().join(", "),
        ));
    } else {
        out.say(format_args!("profile: using observation model '{}' from IR",
            streams[0].name));
    }

    // gh#90: silent-wrong-answer warning. If the model declares N>1 observation
    // blocks but only M<N are bound, the unbound streams contribute zero to the
    // likelihood — the result looks plausible but is methodologically wrong. The
    // bound set is the RESOLVED leaf names (a family root fans out to its
    // leaves), not the raw binding keys — else an indexed family bound by its
    // root would look entirely unbound and false-warn.
    {
        let bound_names: Vec<String> = streams.iter()
            .map(|s| s.name.clone()).collect();
        if let Some(w) = crate::util::format_unbound_streams_warning(
            "profile", &model_obs_names, &bound_names,
        ) {
            out.say_raw(format_args!("{}", w));
        }
    }

    // Canonical schedule: the UNION axis the bound observation model scores
    // along — every stream's scoring boundaries plus every declared period's
    // opening boundary inside the run (multi-cadence, proposal 2026-06-10
    // §3.3; periods, gh#833). Downstream code reads it for `obs_times` (the
    // substep grid + the ODE-MLE / PMMH consumers, which index the obs model
    // by position along it), so it is taken FROM `bind` — the same call that
    // builds the model below — never rebuilt from the rows' labels, which
    // agree with the boundaries only under `closing_at`.
    let observations: Vec<Observation> = {
        let specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        let (bound, _report) = BoundObs::bind(compiled.model.simulation.t_start, specs)
            .unwrap_or_else(|report| {
                out.fail(format!("error: observation data invalid:\n{}", report.render()));
            });
        bound.times().iter().map(|&time| Observation { time, value: 0.0 }).collect()
    };
    let observations = Arc::new(observations);

    for sw in &sweep {
        let idx = compiled.param_index.get(sw.name.as_str()).copied()
            .unwrap_or_else(|| {
                out.fail(format!("focal parameter '{}' not found{}", sw.name,
                    crate::util::constant_hints(&compiled.model, [sw.name.as_str()])));
            });
        let values = sw.grid.expand();
        // A grid value outside the parameter's range would be written straight
        // over the parameter at that cell, where no other check sees it (a
        // `--starts` value goes through the resolver's bounds check; a sweep
        // value does not). The range is the model's (`resolved_bounds`, read
        // off the IR as loaded), narrowed by the fit config's `[estimate]`
        // bounds when it gives them.
        let model_bounds = model_pre.parameters.iter()
            .find(|p| p.name == sw.name)
            .and_then(crate::params_resolver::resolved_bounds);
        let fit_bounds = fit_estimate.get(&sw.name).and_then(|e| e.bounds);
        let bounds = match (model_bounds, fit_bounds) {
            (Some((mlo, mhi)), Some((flo, fhi))) => Some((mlo.max(flo), mhi.min(fhi))),
            (m, f) => f.or(m),
        };
        if let Some((lo, hi)) = bounds {
            let outside: Vec<String> = values.iter()
                .filter(|&&v| !(lo..=hi).contains(&v))
                .map(|v| v.to_string())
                .collect();
            if !outside.is_empty() {
                out.fail(format!(
                    "error: --sweep {}: value(s) {} outside the parameter's bounds [{}, {}]{}. \
                     Sweep only inside the bounds.",
                    sw.name, outside.join(", "), lo, hi,
                    if fit_bounds.is_some() { " (the model's, narrowed by --fit [estimate])" }
                    else { " (the model's)" },
                ));
            }
        }
        focal_grids.push(FocalGrid {
            name: sw.name.clone(),
            values,
            param_idx: idx,
        });
    }
    // Axes sorted by name, the order of a point's focal values. Each point's
    // `mle.toml` lists the focal values in this order, so the bytes stored
    // under a run id do not depend on the order of the `--sweep` flags.
    focal_grids.sort_by(|a, b| a.name.cmp(&b.name));

    // Build the resolved fixed set: CLI `--fixed` wins for collisions
    // with `--fit toml`'s `[fixed]` (per gh#73 §2 precedence). The
    // toml's `[fixed]` is the *artifact's* default; the CLI flag is
    // the per-invocation override.
    //
    // Per 2026-05-25 CLI UX rev 2, `--fixed NAME=VALUE` carries
    // *both* a value (handled by the resolver above) and the kick-out
    // assertion (this set). Names from `--fixed-file` are pulled in
    // via the resolver's provenance — they kick out of [estimate]
    // identically, and `--rw-sd auto` never estimates them. A file name a
    // later tier (a scenario, `--fixed`) re-valued is still pinned: the
    // override record keeps the file as the replaced source. A focal name
    // the file also carries stays focal (the sweep writes over it per cell).
    let cli_fixed_names: std::collections::HashSet<String> =
        fixed_cli_vec.iter().map(|(n, _)| n.clone()).collect();
    let mut fixed_names: std::collections::HashSet<String> = cli_fixed_names.clone();
    for k in fit_toml_fixed.keys() {
        fixed_names.insert(k.clone());
    }
    // Every name a file carries is pinned, read from the files rather than from
    // the resolver's override records: a later tier (a scenario) that re-valued
    // the name to an equal value leaves no record, and the name would be
    // estimated and then refused as a scenario-set estimated parameter.
    for path in &fixed_files_vec {
        let Ok(values) = crate::util::load_params_toml(&path.to_string_lossy()) else { continue };
        for name in values.keys().filter(|n| !focal_names.contains(*n)) {
            fixed_names.insert(name.clone());
        }
    }

    // Focal-vs-fixed conflict check (gh#73 §2): a parameter cannot
    // simultaneously be the sweep axis and a fixed value. Surface
    // *which* source declared the fixed entry so the user knows where
    // to remove it.
    for sw in &focal_names {
        let in_cli = cli_fixed_names.contains(sw);
        let in_toml = fit_toml_fixed.contains_key(sw);
        if in_cli || in_toml {
            let source = match (in_cli, in_toml) {
                (true, true)  => "both `--fixed` and the fit toml's [fixed] block",
                (true, false) => "`--fixed`",
                (false, true) => "the fit toml's [fixed] block",
                (false, false) => unreachable!(),
            };
            out.fail(format!(
                "error: parameter '{}' is in both `--sweep` and {}. \
                 A swept parameter is pinned per cell at the sweep value; \
                 listing it as fixed is contradictory. Drop it from {}.{}",
                sw, source,
                if in_cli && !in_toml { "`--fixed`" }
                else if in_toml && !in_cli { "the fit toml's [fixed]" }
                else { "both" },
                if in_toml {
                    " In a fit config a swept parameter is listed in [estimate], whose \
                     bounds bound the grid; each cell pins it and estimates the rest."
                } else { "" },
            ));
        }
    }
    let exclude: std::collections::HashSet<String> = focal_names.iter()
        .chain(fixed_names.iter()).cloned().collect();

    let param_names_to_estimate: Vec<String> = match &rw_sd {
        Some(crate::args::types::RwSd::Auto) => model.parameters.iter()
            .filter(|p| !exclude.contains(&p.name))
            .filter(|p| compiled.param_index.contains_key(p.name.as_str()))
            .map(|p| p.name.clone())
            .collect(),
        // In the model's order, as `auto` and [estimate] give it: the list is
        // a hash map, whose order changes from one process to the next, and a
        // start rule draws per parameter in this order. Names the model does
        // not declare follow, sorted, for the spec builder to refuse.
        Some(crate::args::types::RwSd::Map(_)) => {
            let mut names: Vec<String> = model.parameters.iter()
                .filter(|p| rw_sd_map_raw.contains_key(&p.name) && !exclude.contains(&p.name))
                .map(|p| p.name.clone())
                .collect();
            let mut unknown: Vec<String> = rw_sd_map_raw.keys()
                .filter(|n| !exclude.contains(*n) && !model.parameters.iter().any(|p| &p.name == *n))
                .cloned()
                .collect();
            unknown.sort();
            names.extend(unknown);
            names
        }
        None => fit_estimate.keys()
            .filter(|name| !exclude.contains(*name))
            .cloned()
            .collect(),
    };
    // An explicit `--rw-sd` list with --fit names the estimated set itself, so
    // an [estimate] parameter it leaves out would be held at its start in
    // every cell while the config, and the record of the run, say it is
    // estimated. Refused, naming the parameters.
    if let (Some(crate::args::types::RwSd::Map(_)), Some(f)) = (&rw_sd, fit.as_ref()) {
        if let Some(msg) = refuse_rw_sd_list_short_of_estimate(
            &fit_estimate, &exclude, &rw_sd_map_raw, &f.handle,
        ) {
            out.fail(format!("error: {msg}"));
        }
    }

    // The scenario × estimate rule, on the set this profile estimates. The
    // check at the top of the function reads the fit config's `[estimate]`;
    // the estimated set is what `--rw-sd` selects (every parameter that is not
    // swept or pinned under `auto`), which a fit config need not list, and
    // which exists without `--fit` (run spec §1.3).
    if let Some(scenario) = scenario_name.as_deref() {
        let fit_path_str: Option<String> = fit.as_ref().map(|f| f.handle.clone());
        let estimated: Vec<&str> = param_names_to_estimate.iter().map(|n| n.as_str()).collect();
        let clash = crate::params_resolver::scenario_estimate_collision(
            &model_pre, scenario, &estimated,
            match (&rw_sd, fit_path_str.as_deref()) {
                (None, Some(fit_path)) =>
                    crate::params_resolver::EstimatedBy::FitToml(fit_path),
                _ => crate::params_resolver::EstimatedBy::RwSd { auto: rw_sd_auto },
            },
        ).unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
        if let Some(msg) = clash {
            out.fail(format!("error: {msg}"));
        }
    }

    // The profile's partition: today's selection, translated into role
    // declarations by the one adapter that reads it. Recorded, not acted on.
    let profile_roles: Result<crate::roles::Partition, String> =
        legacy_role_inputs(LegacySelection {
            estimated: &param_names_to_estimate,
            focal: &focal_names,
            fit_estimate: &fit_estimate,
            fit_fixed: fit_fixed_via,
            cli_fixed: &fixed_cli_vec.iter()
                .map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            fixed_files: &fixed_files_vec,
            model: &model_declared,
        })
        .and_then(|decls| crate::roles::resolve_roles(
            &decls.inputs(&model_declared, &fit_simplex_groups), &crate::roles::PROFILE,
        ).map_err(|e| e.to_string()));

    // Specs honour the fit-toml `[estimate]` bounds when supplied (gh#73),
    // through `build_if2_params_from_specs`'s fit-toml-bounds-within-model
    // resolver.
    let specs: Vec<crate::fit::runner::ParamSpec> = param_names_to_estimate.iter().map(|name| {
        let from_fit = fit_estimate.get(name);
        crate::fit::runner::ParamSpec {
            name: name.clone(),
            bounds: from_fit.and_then(|e| e.bounds),
        }
    }).collect();

    let if2_params: Vec<sim::inference::if2::EstimatedParam> = crate::fit::runner::build_if2_params_from_specs(
        &model, &compiled, &base_params, &specs,
    ).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    // The steps an IF2 cell takes: `--rw-sd NAME=SD` (natural scale) when
    // given, else the `--fit` config's `[method.perturbation]` SD in z when
    // its method is IF2, else the map's default. The timing is the table's
    // `initial_state`. An nl-* cell perturbs nothing and reads no step; a
    // table it would ignore is said once.
    let fit_perturbation: indexmap::IndexMap<String, crate::fit::config_v2::PerturbationSpec> =
        match fit.as_ref().and_then(|f| f.config.inference.method.as_ref()).map(|m| &m.algorithm) {
            Some(crate::fit::config_v2::Algorithm::IF2 { perturbation, .. }) => perturbation.clone(),
            _ => indexmap::IndexMap::new(),
        };
    if !matches!(profile_algo, ProfileAlgo::If2) && !fit_perturbation.is_empty() {
        out.say(format_args!(
            "note: the --fit config's [method.perturbation] is IF2's, and `{}` perturbs nothing: it is not read.",
            format!("{:?}", profile_algo).to_lowercase()));
    }
    let (fit_perturbation, written_steps): (indexmap::IndexMap<_, _>, Vec<crate::fit::runner::WrittenStep>) =
        if matches!(profile_algo, ProfileAlgo::If2) {
            let steps = if2_params.iter().map(|p| {
                use crate::fit::runner::{WrittenSd, WrittenStep};
                let table = fit_perturbation.get(&p.name);
                let sd = match (rw_sd_map_raw.get(&p.name).and_then(|v| *v), table.and_then(|t| t.sd)) {
                    (Some(natural), _) => WrittenSd::Natural(natural),
                    (None, Some(z)) => WrittenSd::Z(z.get()),
                    (None, None) => WrittenSd::Default,
                };
                let at = if table.is_some_and(|t| t.initial_state) {
                    sim::inference::types::PerturbAt::TimeZeroOnly
                } else {
                    sim::inference::types::PerturbAt::EveryObservation
                };
                WrittenStep::of(p, sd, at)
            }).collect();
            (fit_perturbation, steps)
        } else {
            (indexmap::IndexMap::new(), Vec::new())
        };
    // Each IF2 cell perturbs by `rw_sd`, and a derived one must be able to
    // move the parameter.
    if matches!(profile_algo, ProfileAlgo::If2) {
        if let Err(e) = crate::fit::runner::refuse_degenerate_auto_rw_sd(
            &if2_params, &written_steps, "`--rw-sd NAME=SD`, or `NAME = …` in the fit config's [method.perturbation]",
        ) {
            out.fail(format!("error: {e}"));
        }
    }

    // `nl-lbfgs` takes each cell's optimum with the ODE forward-sensitivity
    // gradient, so ask the gradient capability gate once, here, rather than
    // letting every cell in the grid discover the same refusal and report it as
    // a failed cell. The estimated set is the non-focal free parameters, which
    // is exactly what `optimize_cell` will differentiate with respect to —
    // pinned focal parameters never enter it. (`fit run` asks the same question
    // in `FitConfig::validate_gradient_capability`.)
    if matches!(
        profile_algo,
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs)
    ) {
        let estimated: std::collections::HashSet<&str> =
            if2_params.iter().map(|p| p.name.as_str()).collect();
        if let Err(e) = sim::inference::gradient_capability::preflight_gradient_ode(
            &compiled, &base_params, &estimated,
        ) {
            out.fail(format!(
                "error: --algorithm nl-lbfgs takes each cell's optimum with the ODE \
                 forward-sensitivity gradient, and the gradient capability check refuses \
                 this model:\n\n  {e}\n\n  \
                 `--algorithm nl-sbplx` profiles the same ODE marginal likelihood without \
                 a gradient."
            ));
        }
    }

    // The fit config's `[[simplex_groups]]`: members that must sum to one.
    // IF2 keeps the constraint by perturbing them jointly on the log-ratio
    // scale, so every member has to be one of this profile's estimated
    // parameters — a swept or pinned member would leave the others free to
    // drift off the simplex. The nl-* optimizers do not honour the constraint
    // (as under `fit run`), which is said rather than dropped silently.
    let simplex_groups: Vec<sim::inference::if2::SimplexGroup> = if fit_simplex_groups.is_empty() {
        Vec::new()
    } else {
        for (gi, group) in fit_simplex_groups.iter().enumerate() {
            for name in &group.params {
                if !if2_params.iter().any(|p| p.name == *name) {
                    let why = if focal_names.contains(name) { "it is swept (--sweep)" }
                        else if fixed_names.contains(name) { "it is pinned" }
                        else { "it is not in the estimated set (--rw-sd, or the --fit [estimate])" };
                    out.fail(format!(
                        "error: simplex_groups[{gi}] member '{name}' is not estimated by this \
                         profile: {why}. The members of a simplex group sum to one and are \
                         estimated jointly, so every member must be estimated: profile a \
                         parameter outside the group, or drop the group from the fit config."));
                }
            }
        }
        if !matches!(profile_algo, ProfileAlgo::If2) {
            out.say(format_args!("warning: the fit config declares simplex_groups, but `{}` does not \
                honour the simplex constraint: members are optimised independently.",
                format!("{:?}", profile_algo).to_lowercase()));
        }
        // Each member's SD on the log-ratio coordinate: the `--fit` table's,
        // else the member's default.
        let sds: Vec<Vec<sim::inference::types::ZSd>> = fit_simplex_groups.iter().map(|g| {
            g.params.iter().map(|name| {
                match fit_perturbation.get(name).and_then(|t| t.sd) {
                    Some(sd) => Ok(sd),
                    None => Ok(crate::fit::steps::simplex_member_default_step()),
                }
            }).collect::<Result<Vec<_>, String>>()
        }).collect::<Result<Vec<_>, String>>().unwrap_or_else(|e| out.fail(format!("error: {e}")));
        crate::fit::runner::resolve_simplex_groups(&fit_simplex_groups, &compiled.param_index, &sds)
            .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })
    };
    // IF2's step for each estimated parameter, the same in every cell. Only
    // an IF2 cell reads it.
    let if2_steps: Vec<sim::inference::if2::If2Step> = if matches!(profile_algo, ProfileAlgo::If2) {
        crate::fit::runner::if2_steps(&if2_params, &written_steps, &simplex_groups)
            .unwrap_or_else(|e| out.fail(format!("error: {e}")))
    } else {
        Vec::new()
    };
    // Member names per group, in order, for the method identity.
    let simplex_names: Vec<Vec<String>> =
        fit_simplex_groups.iter().map(|g| g.params.clone()).collect();

    // A `--starts from_params` file starts the estimated parameters. A value
    // for a parameter the profile neither estimates, fixes nor sweeps would
    // set it in every cell outside the problem, so it is refused.
    if let Some((crate::params_resolver::StartsOrigin::Params { path }, values)) = &starts_values {
        let stray: Vec<&str> = values.iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| !if2_params.iter().any(|p| p.name == *n)
                && !fixed_names.contains(*n) && !focal_names.iter().any(|f| f == n))
            .collect();
        if !stray.is_empty() {
            out.fail(format!(
                "error: --starts from_params={}: {} {} neither estimated nor fixed by the profile's problem; \
                 a starts file starts the estimated parameters only. Drop {} from the file, or give {} a \
                 value in the fit file's [fixed].",
                path.display(), stray.join(", "), if stray.len() == 1 { "is" } else { "are" },
                if stray.len() == 1 { "it" } else { "them" }, if stray.len() == 1 { "it" } else { "them" }));
        }
    }
    let if2_params = Arc::new(if2_params);

    // ── Prior resolution (gh#73) ─────────────────────────────────────
    //
    // IF2 and the nl-* optimizers maximise the likelihood and read no prior.
    // The per-parameter source (fit_toml / model_ir / flat_fallback) is still
    // resolved once, up front, because it is provenance: it is recorded in the
    // profile-base sidecar and folded into the base identity (it is also what
    // keys the estimated *set*).
    let resolved_priors: Vec<crate::fit::priors_precedence::ResolvedPrior> =
        crate::fit::priors_precedence::resolve_priors_with_precedence(
            &if2_params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &fit_estimate,
            &model,
        );
    // gh#369: a `--fit` prior that displaces a model `~` prior, said once
    // before the grid (as `fit run` does). Over every `[estimate]` entry,
    // not only the non-focal set: the override holds wherever a prior is read.
    let prior_overrides = crate::fit::priors_precedence::prior_overrides(
        &fit_estimate.keys().cloned().collect::<Vec<_>>(), &fit_estimate, &model);
    if let Some(w) =
        crate::fit::priors_precedence::format_prior_override_warning(&prior_overrides)
    {
        out.say_raw(format_args!("{}", w));
    }

    // The starts rule: the one `--starts` or the config names, else the rule
    // `fit run` defaults to, over the set each cell estimates (`from_prior`
    // when every one has a prior to draw from, `uniform_unconstrained`
    // otherwise). Said at startup with where it came from.
    let starts_resolution = match &declared_starts {
        Some(rule) => crate::fit::config_v2::StartsResolution::Declared(rule.clone()),
        None => crate::fit::config_v2::default_starts(
            &if2_params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &fit_estimate, &model),
    };
    let init_method: crate::fit::starts::ChainStarts = starts_resolution.rule();
    match &starts_resolution {
        crate::fit::config_v2::StartsResolution::Declared(_) => out.say(format_args!(
            "profile: starts: {} — {starts_from}", starts_resolution.describe())),
        // The default's description says it is the default, and why.
        _ => out.say(format_args!("profile: starts: {}", starts_resolution.describe())),
    }

    // Per-start init draws across non-focal estimated params (gh#42).
    // Computed once, reused at every grid cell so start_idx=k means the
    // same draw across cells (lets the per-start TSV rows be compared
    // cell-to-cell). `None` → every start uses `if2_params` directly
    // (Single mode; or `--starts 1`).
    // Every rule draws through the one seam the fit runners use
    // (`chain_starts::draw_chain_starts`), with a sourced rule's handle
    // resolved once. A source that did not converge is refused, as on
    // `fit run`, unless `--allow-nonconverged-source`: starting a profile
    // from an unconverged fit launders its multi-modality into every cell.
    if allow_nonconverged_source {
        if let Err(e) = crate::fit::chain_starts::refuse_inert_allow_nonconverged(
            &init_method, "profile",
        ) {
            out.fail(format!("error: {e}"));
        }
    }
    // A stored fit a sourced rule names is read in the run store this
    // profile is stored in.
    let fit_store = crate::camdl_store::open_existing(&crate::fit_store::reader_root());
    let resolved_starts = crate::fit::chain_starts::resolve_starts(
        &init_method, allow_nonconverged_source, fit_store.as_ref())
        .unwrap_or_else(|e| {
            out.fail(format!("error: profile --starts {}: {}", init_method, e));
        });
    let drawn_starts = {
        let priors: Vec<(String, sim::inference::pmmh::Prior)> = if2_params
            .iter()
            .map(|s| (s.name.clone(), crate::fit::runner::resolve_prior(&s.name, &fit_estimate, &resolved.model).0))
            .collect();
        // The rules bind the resolver's `estimate_set`, which here is the fit
        // config's [estimate] keys: it still holds a swept name (pinned per
        // cell, so a source need not carry it) and is empty without --fit
        // (so a `from_posterior` source would start nothing). A cell
        // estimates `if2_params`, so that is the set the starts bind.
        let mut starts_view = resolved.clone();
        starts_view.estimate_set = if2_params.iter().map(|p| p.name.clone()).collect();
        let ctx = crate::fit::chain_starts::StartContext {
            resolved: &starts_view,
            base_specs: &if2_params,
            priors: &priors,
        };
        crate::fit::chain_starts::draw_chain_starts(&ctx, &resolved_starts, n_starts, seed_base)
            .unwrap_or_else(|e| {
                out.fail(format!("error: profile --starts {}: {}", init_method, e));
            })
    };
    let per_start_params: Arc<Vec<Vec<sim::inference::if2::EstimatedParam>>> =
        Arc::new(drawn_starts.to_estimated_params(&if2_params));

    let process = Arc::new(ChainBinomialProcess::new(compiled.clone()));
    // Build the multi-stream observation model from the loaded `ObsStream`s via
    // the single shared builder. Each stream's projection + likelihood come from
    // its `observations { }` block (the modern observation system, exactly as
    // `fit run` / `pfilter` resolve them) — there is no `--flow` override.
    // Concrete `Arc<MultiStreamObsModel>` — the IF2 per-cell call site
    // accepts `&dyn ObservationModel<ParticleState>` (auto-coerced from
    // `&MultiStreamObsModel`); the NLopt path needs the concrete type
    // for `optimize_cell` (`compute_ode_loglik` reads
    // `log_likelihood_from_flows_and_counts` directly).
    let obs_times_vec: Vec<f64> = observations.iter().map(|o| o.time).collect();

    let obs_model_obj: Arc<MultiStreamObsModel> = {
        // The `ObsStream -> StreamSpec` mapping is the single shared builder
        // (`stream_specs_from_obs_streams`): authoritative per-grid-time cells
        // (holes = `None`, skipped in the likelihood) + survey denominators
        // (`aux`), each stream fed its OWN schedule; `bind` re-merges to the
        // union and validates aux — identical to how `fit run` and `pfilter`
        // build their obs model, so an indexed survey family scores identically.
        let stream_specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        let (bound, report) = BoundObs::bind(compiled.model.simulation.t_start, stream_specs)
            .unwrap_or_else(|report| {
            out.fail(format!("error: observation data invalid:\n{}", report.render()));
        });

        // gh#812: a non-fatal bind report was previously discarded, so the
        // Warn channel was write-only. Surface it — a zero denominator scores
        // as zero, and the user must be told which rows and why.
        if !report.findings().is_empty() {
            out.say(format_args!("{}", report.render()));
        }
        Arc::new(MultiStreamObsModel::new(bound, compiled.clone())
            .unwrap_or_else(|e| {
                out.fail(format!("error: observation model construction failed: {:?}", e));
            }))
    };

    // Build Cartesian product of all focal grids.
    let mut grid_points: Vec<Vec<(usize, f64)>> = vec![vec![]];
    for fg in &focal_grids {
        let mut expanded = Vec::new();
        for existing in &grid_points {
            for &val in &fg.values {
                let mut point = existing.clone();
                point.push((fg.param_idx, val));
                expanded.push(point);
            }
        }
        grid_points = expanded;
    }

    // gh#73: the per-parameter source list is stored in the same
    // order the resolver emitted (declaration order of estimated
    // params). The `inner_hash` sort key normalises this to a
    // canonical form for caching; we keep the natural order here so
    // `run.json` is human-friendly.
    let resolved_priors_kv: Vec<(String, String)> = resolved_priors.iter()
        .map(|r| {
            let source_str = match r.source {
                crate::fit::priors_precedence::PriorSource::FitToml      => "fit_toml",
                crate::fit::priors_precedence::PriorSource::ModelIr      => "model_ir",
                crate::fit::priors_precedence::PriorSource::FlatFallback => "flat_fallback",
                // gh#75: explicit flat-prior opt-in via `prior = { flat = {} }`.
                // Not reachable on the profile path today (the resolver
                // gets its fit-toml priors from `--fit`, which doesn't yet
                // surface explicit flat — but the resolver's output is
                // shared, so the match must be exhaustive).
                crate::fit::priors_precedence::PriorSource::FlatExplicit => "flat_explicit",
            };
            (r.param.clone(), source_str.to_string())
        })
        .collect();
    // gh#83/gh#85 step 9: build parameters_provenance from the
    // unified resolver output (populated upstream). Each entry maps a
    // resolved parameter name to the value + source + role + audit
    // fields. Provenance: no identity holds it.
    //
    // The resolver knows only `[estimate]` and the pins, so on its own it
    // records every parameter of a CLI-only profile as `fixed`. The roles
    // this profile actually gives them are set here: a swept parameter is
    // `focal` (pinned per cell at its grid value; the value recorded is the
    // base value the grid writes over), and every parameter IF2 or the
    // optimizer moves is `estimated`.
    let mut parameters_provenance =
        crate::run_meta::ParameterProvenance::of_all(&resolved.params);
    for (name, prov) in parameters_provenance.iter_mut() {
        if focal_names.contains(name) {
            prov.role = "focal".to_string();
        } else if if2_params.iter().any(|p| p.name == *name) {
            prov.role = "estimated".to_string();
        }
    }
    // Per-start init provenance, straight from the drawn starts.
    let init_provenance: Option<crate::run_meta::InitProvenance> =
        Some(crate::run_meta::InitProvenance::from_chain_starts(&drawn_starts));

    // Run-level provenance recorded in every point's `provenance.json`, a
    // provenance file no identity holds: per-parameter resolution provenance,
    // the notices, per-chain init provenance, the prior sources and the
    // training partition.
    let run_provenance_json = serde_json::json!({
        "parameters_provenance":
            serde_json::to_value(&parameters_provenance).unwrap_or(serde_json::Value::Null),
        "notices":
            serde_json::to_value(
                crate::params_resolver::notices(&[&resolved]).iter()
                    .chain(focal_notices.iter())
                    .cloned()
                    .collect::<crate::out::notice::Notices>())
                .unwrap_or(serde_json::Value::Null),
        "init_provenance":
            serde_json::to_value(&init_provenance).unwrap_or(serde_json::Value::Null),
        // Where each estimated parameter's prior came from, and the model
        // prior a `--fit` prior displaced.
        "resolved_priors": resolved_priors_kv.iter()
            .map(|(n, s)| crate::run_meta::ResolvedPriorEntry {
                param: n.clone(), source: s.clone(),
                overridden_model_prior: prior_overrides.iter()
                    .find(|o| &o.param == n).map(|o| o.model.clone()) })
            .collect::<Vec<_>>(),
        // Which rows each stream trained on.
        "training_partition": ir::caltime::CalendarAxis::from_model(&model)
            .map(|calendar| training_partition.record(&calendar))
            .ok(),
    });

    // Each grid point's partition: the focal parameters fixed at the point's
    // values. Recorded in that point's leaves.
    let cell_partitions: Vec<serde_json::Value> = grid_points.iter().map(|gp| {
        let point: Vec<(String, f64)> = focal_grids.iter().zip(gp.iter())
            .map(|(fg, &(_, v))| (fg.name.clone(), v))
            .collect();
        let field = match &profile_roles {
            Ok(p) => match p.cell(&point) {
                Ok(cell) => crate::roles::PartitionField::Resolved(cell.record(&resolved)),
                Err(reason) => crate::roles::PartitionField::Unresolved { reason },
            },
            Err(reason) => crate::roles::PartitionField::Unresolved { reason: reason.clone() },
        };
        serde_json::to_value(&field).unwrap_or(serde_json::Value::Null)
    }).collect();

    // A start drawn from a stored run is keyed by its values; the run is
    // recorded as lineage.
    if let Some(src) = &resolved_starts.source {
        if let crate::fit_store::StartInput::Run { source } = &src.input {
            info.lineage(source);
        }
    }
    let finite = |what: &str, x: f64| store::Finite::new(x).unwrap_or_else(|e| out.fail(format!("error: {what}: {e}")));
    let stem = crate::hashing::path_stem_slug(&ir_path_named).unwrap_or_else(|| "model".to_string());

    // How each cell is fitted.
    let algorithm = profile_algo.method_kind().as_str().to_string();
    let method_identity = crate::profile_store::ProfileMethod {
        algorithm: algorithm.clone(),
        cell: match profile_algo {
            ProfileAlgo::If2 => crate::profile_store::CellSettings::If2 {
                particles: n_particles as u64,
                iterations: n_iterations as u64,
                cooling: finite("--cooling", cooling),
                cooling_target_iters: cooling_schedule.target_iters as u64,
                max_substeps: pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET),
                binomial: binomial.into(),
            },
            ProfileAlgo::Nlopt(_) => crate::profile_store::CellSettings::Nlopt {
                tolerance: finite("the optimizer's tolerance", nl_tolerance),
                max_evals: nl_max_evals as u64,
            },
        },
        dt: finite("dt", dt),
        estimated: if2_params.iter().enumerate()
            .map(|(i, p)| crate::profile_store::EstimatedSetting::of(p, written_steps.get(i)))
            .collect::<Result<Vec<_>, String>>()
            .unwrap_or_else(|e| out.fail(format!("error: {e}"))),
        simplex_groups: simplex_names.clone(),
        jacobian_model,
    };

    let total_jobs = grid_points.len() * n_starts * seeds.len();
    let dim_str = focal_grids.iter()
        .map(|fg| format!("{}={}", fg.name, fg.values.len()))
        .collect::<Vec<_>>().join(" × ");
    // Banner names the actual per-cell optimizer + its real budget — a
    // deterministic nl-* cell has no particles/iterations, so labelling every
    // run "IF2 (N particles × M iter)" was misleading.
    let algo_label = match profile_algo {
        ProfileAlgo::If2 => "IF2",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx) => "nl-sbplx",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa) => "nl-bobyqa",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs) => "nl-lbfgs",
    };
    out.say(format_args!("profile: {} grid ({}) × {} starts × {} seeds = {} {} runs",
        grid_points.len(), dim_str, n_starts, seeds.len(), total_jobs, algo_label));
    match profile_algo {
        ProfileAlgo::If2 => out.say(format_args!(
            "profile: IF2 per cell = {} particles × {} iterations", n_particles, n_iterations)),
        ProfileAlgo::Nlopt(_) => out.say(format_args!(
            "profile: deterministic ODE MLE per cell (no particle filter), tolerance {:e}, \
             at most {} evaluations{}",
            nl_tolerance, nl_max_evals,
            if nl_from_fit { " (the --fit method's)" } else { "" })),
    }
    log::debug!("profile: cells run on backend {}", method.backend.as_str());
    log::debug!("profile: a starts source's verdict is lifted: {allow_nonconverged_source}");

    // ── Each point's identity ─────────────────────────────────────────────
    // One job per (seed, cell, start), in that order: the order the
    // invocation lists its points in, which the table follows.
    let start_values: Vec<std::collections::BTreeMap<String, store::Finite>> = per_start_params.iter()
        .map(|specs| specs.iter().map(|p| (p.name.clone(), finite(&format!("the start of `{}`", p.name), p.initial))).collect())
        .collect();
    let cell_focal: Vec<std::collections::BTreeMap<String, store::Finite>> = grid_points.iter()
        .map(|gp| focal_grids.iter().zip(gp.iter())
            .map(|(fg, &(_, v))| (fg.name.clone(), finite(&format!("the grid value of `{}`", fg.name), v)))
            .collect())
        .collect();
    // A start's replicate index: how many earlier starts have its values.
    let replicate: Vec<u32> = (0..start_values.len())
        .map(|si| start_values[..si].iter().filter(|v| **v == start_values[si]).count() as u32)
        .collect();
    let jobs: Vec<(usize, usize, usize)> = (0..seeds.len())
        .flat_map(|seed_idx| (0..grid_points.len())
            .flat_map(move |gi| (0..n_starts).map(move |si| (seed_idx, gi, si))))
        .collect();
    let members: Vec<(crate::profile_store::ProfilePointIdentity, crate::profile_store::ProfilePointLabels, store::Wanted<crate::profile_store::ProfilePoint>)> =
        jobs.iter().map(|&(seed_idx, gi, si)| {
            let focal = cell_focal[gi].clone();
            let start = start_values[si].clone();
            let seed = crate::profile_store::point_seed(seeds[seed_idx], &focal, &start, replicate[si]);
            let labels = crate::profile_store::ProfilePointLabels {
                model: stem.clone(),
                focal: crate::profile_store::focal_label(&focal),
                method: method_identity.label(),
                seed: seeds[seed_idx],
                start: si as u32,
            };
            let identity = crate::profile_store::ProfilePointIdentity {
                problem: problem_dep.clone(),
                focal,
                method: method_identity.clone(),
                start,
                seed,
            };
            (identity, labels, store::Wanted::none())
        }).collect();
    // `begin_all` begins each distinct identity once, and answers in the
    // order first asked for: a seed listed twice in `--seeds` names its
    // points once.
    let mut distinct: Vec<usize> = Vec::new();
    {
        let mut seen = std::collections::BTreeSet::new();
        for (ji, (identity, _, _)) in members.iter().enumerate() {
            if seen.insert(store::Identity::key(identity)) {
                distinct.push(ji);
            }
        }
    }
    let mut failures: Vec<String> = Vec::new();
    let mut claimed: Vec<(store::Run<crate::profile_store::ProfilePoint>, usize)> = Vec::new();
    let mut n_stored = 0usize;
    let mut batch = store.declare::<crate::profile_store::ProfilePoint>(&members, &info, request.reuse())
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let begun_all = batch.begin_all().unwrap_or_else(|e| out.fail(format!("error: {e}")));
    for (begun, &ji) in begun_all.into_iter().zip(&distinct) {
        match crate::camdl_store::declared(begun, crate::result_files::profile::ProfilePointFile::ALL) {
            store::Begin::Reused(_) => n_stored += 1,
            store::Begin::Busy(owner) => failures.push(format!(
                "point {} {}: {}", members[ji].1.focal, members[ji].1.start, store::Error::InProgress(owner))),
            store::Begin::Claimed(run) => claimed.push((run, ji)),
        }
    }
    if n_stored > 0 {
        out.say(format_args!("profile: {} of {} points already stored", n_stored, distinct.len()));
    }
    // While points are claimed, a forcing refusal is kept rather than ending
    // the process from a worker thread, so each claimed point is failed and
    // none is left `running`.
    let _refusal_deferred = forcing.defer_while_claimed();
    let n_points = distinct.len();
    let batch = std::sync::Mutex::new(batch);
    let finished = std::sync::atomic::AtomicUsize::new(n_stored);
    let progress = || {
        let n = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let snapshot = serde_json::json!({ "phase": "profiling", "step": n, "total": n_points });
        if let Err(e) = batch.lock().unwrap_or_else(|e| e.into_inner()).progress(&snapshot) {
            log::debug!("profile: cannot write the progress snapshot: {e}");
        }
    };

    // ── Progress bar ──────────────────────────────────────────────────────
    // One overall bar over the points this run computes, ticked from the
    // parallel loop (`Task` is `Send + Sync`). The Reporter honors --progress
    // (Pretty=bar, Plain=throttled `profile pos/len` log lines, None=silent).
    let bar = crate::out::progress::Reporter::new().task(claimed.len() as u64, "profile", "points");

    // gh#audit-H13: --parallel / CAMDL_PARALLEL throttles the rayon thread
    // budget. Build a SCOPED local pool and run the parallel job sweep inside
    // `pool.install(...)`: by the time profile reaches here the global pool is
    // already initialised, so `build_global` would be ignored. parallel == 0
    // means "use rayon's default" (all logical cores): leave the pool unset
    // and run on the global pool.
    let prof_pool: Option<rayon::ThreadPool> = if parallel > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .unwrap_or_else(|e| {
                    out.fail(format!("error: failed to build thread pool (--parallel {}): {}", parallel, e));
                }),
        )
    } else {
        None
    };

    // ── Run the claimed points: greedy priority work-queue ────────────────
    // Per focal axis, its grid's value range — for normalizing cell
    // coordinates so the nearest-to-best ordering is scale-fair.
    let axis_ranges: Vec<(f64, f64)> = focal_grids.iter().map(|fg| {
        let mn = fg.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = fg.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mn, mx)
    }).collect();
    let cell_norm: Vec<Vec<f64>> = (0..grid_points.len())
        .map(|gi| profile_cell_norm(&grid_points, &axis_ranges, gi))
        .collect();
    // The best (finite) profile-loglik seen and the grid cell that produced
    // it. Workers drill toward this cell; the bar surfaces the loglik. This is
    // a read-only side-channel — each point is computed from its own identity,
    // so it never changes any output, only the order cells are visited in.
    let best: std::sync::Mutex<(f64, Option<usize>)> =
        std::sync::Mutex::new((f64::NEG_INFINITY, None));
    let point_failures: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    // A compiled forcing records once that it was evaluated outside its
    // range, for every clone of the compiled model (`sim::forcing_range`). So
    // that each point records the forcings its own computation held, a point
    // run under the override, of a model with a forcing that has a range,
    // computes on a model compiled for it alone.
    let compile_per_point = forcing.allowed() && compiled.time_func_cache.iter().any(|f| f.range.is_some());
    let compile_point = || -> Result<(Arc<CompiledModel>, Arc<ChainBinomialProcess>, Arc<MultiStreamObsModel>), String> {
        let compiled = Arc::new(CompiledModel::new(model.clone())
            .map_err(|e| format!("{e:?}"))?
            .with_eval_settings(crate::env::eval_settings(allow_degenerate_rates)));
        let specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        let (bound, _report) = BoundObs::bind(compiled.model.simulation.t_start, specs)
            .map_err(|report| format!("observation data invalid:\n{}", report.render()))?;
        let obs = MultiStreamObsModel::new(bound, compiled.clone())
            .map_err(|e| format!("observation model construction failed: {e:?}"))?;
        Ok((compiled.clone(), Arc::new(ChainBinomialProcess::new(compiled)), Arc::new(obs)))
    };

    // A point claimed or computed after a forcing refusal is failed with it.
    let refused = |run: store::Run<crate::profile_store::ProfilePoint>| -> Option<store::Run<crate::profile_store::ProfilePoint>> {
        match forcing.refused() {
            Some(reason) => {
                let _ = run.fail("forcing outside its range", &reason);
                None
            }
            None => Some(run),
        }
    };
    let eval_job = |run: store::Run<crate::profile_store::ProfilePoint>, ji: usize| {
        let (_, grid_idx, start_idx) = jobs[ji];
        let Some(run) = refused(run) else { return };
        let (compiled, process, obs_model_obj) = if compile_per_point {
            match compile_point() {
                Ok(models) => models,
                Err(e) => {
                    let reason = format!("compiling the model for the point: {e}");
                    let _ = run.fail("compiling", &reason);
                    point_failures.lock().unwrap_or_else(|e| e.into_inner())
                        .push(format!("profile grid point {grid_idx} (start {start_idx}): {reason}"));
                    return;
                }
            }
        } else {
            (Arc::clone(&compiled), Arc::clone(&process), Arc::clone(&obs_model_obj))
        };
        let if2_params = Arc::clone(&if2_params);
        let focal_values: Vec<f64> = grid_points[grid_idx].iter().map(|&(_, v)| v).collect();

        // Pin focal parameters
        let mut params = base_params.clone();
        for &(idx, val) in &grid_points[grid_idx] {
            params[idx] = val;
        }

        let job_seed = run.identity().seed.seed;
        let job_t0 = std::time::Instant::now();

        // gh#42: each start uses its own draw across the non-focal estimated
        // params. The focal params are already pinned in `params` above, so
        // the draw on non-focal indices doesn't perturb the grid point.
        let per_start_specs: &[sim::inference::if2::EstimatedParam] = per_start_params[start_idx].as_slice();

        // Dispatch on algorithm. IF2 keeps the historical PF-driven
        // per-cell MLE; NLopt branches into the
        // deterministic-MLE path (gh#47) — same `compute_ode_loglik`
        // closure that fit.toml methods call, with the focal grid value
        // pinned in `params` and optimization confined to the non-focal
        // estimated indices.
        //
        // gh#74 Option B: each branch also populates a
        // `PerStartDiagnostics` carrying the convergence info this
        // start produced (cooling-end state for IF2, completion flag
        // for everyone). The rollup phase
        // aggregates these across K starts → per-cell columns.
        let mut diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let (final_loglik, mle_params): (f64, Vec<f64>) = match profile_algo {
            ProfileAlgo::If2 => {
                let config = profile_if2_config(
                    n_particles, n_iterations, cooling_schedule, pf_max_substeps, binomial, dt,
                    process.compiled.model.simulation.t_start,
                    simplex_groups.clone(),
                    if2_steps.clone(),
                    objective.skips_first_observation(),
                );
                let result = run_if2(
                    &*process, &*obs_model_obj, &params, per_start_specs, &config, job_seed,
                );
                // gh#74 Option B: per-start IF2 diagnostics. Populate
                // the trace + cooling end-state on success; on Err the
                // engine returned, leave `diag` at defaults so the
                // aggregator records "not completed" for this start.
                diag.algo = Some(crate::profile_diagnostics::DiagAlgo::If2);
                match result {
                    Ok(r) => {
                        // Clean-eval re-pass at the swarm-mean MLE. IF2's
                        // `final_loglik` is the perturbed-params loglik
                        // from the last cooling iteration
                        // (`if2.rs:540: final_loglik:
                        // last_iter.if2_perturbed_loglik`), NOT the
                        // loglik at `r.mle` (the swarm mean). Without
                        // this re-pass, mle.toml reports a loglik that
                        // isn't reproducible by `camdl pfilter` at the
                        // saved MLE — historically a ~40-nat extraction
                        // bias on noisy PF runs, and finite-vs-(-inf)
                        // catastrophic on PF-degenerate models (events
                        // blocks, GH #68). Fit-run already does this via
                        // `run_quick_pfilter` (cf. runner.rs:1208-1224);
                        // this mirrors the same pattern for profile's
                        // per-cell path.
                        let smc_config = sim::inference::traits::SMCConfig {
                            n_particles: n_particles.min(500),
                            dt,
                            t_start: process.compiled.model.simulation.t_start,
                            skip_first_obs_from_loglik: objective.skips_first_observation(),
                            record_ancestry: false, path_recording: Default::default(),
                            record_prequential: false,
                            record_predictions: false, // gh#520: only `.log_likelihood` is read below
                            record_stream_means: false,
                            max_substeps: pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET), // gh#241
                            // gh#747: the cell's own sampler, so the re-scored
                            // loglik is the one its IF2 run would reproduce.
                            binomial,
                        };
                        // Distinct seed from the IF2 inner run so the
                        // clean-eval PF doesn't reuse IF2's last
                        // resample-RNG state. Adding 1 is sufficient —
                        // ChaCha8 (StatefulRng) produces uncorrelated
                        // sequences for seeds 1 apart.
                        let clean_seed = job_seed.wrapping_add(1);
                        let true_ll = sim::inference::bootstrap_filter(
                            &*process, &*obs_model_obj, &r.mle, &smc_config, clean_seed,
                        ).map(|res| res.log_likelihood)
                          .unwrap_or(f64::NEG_INFINITY);
                        // Trace = per-iteration `if2_perturbed_loglik`.
                        // The `IF2IterResult.loglik` field is NaN here
                        // (the post-hoc clean-PF re-eval only runs at
                        // the swarm mean, not per iter — see
                        // if2.rs:121-129), so we surface the engine's
                        // own perturbed-loglik trace. It's documented
                        // as "NOT useful for model assessment" but is
                        // a valid signal for chain-level agreement
                        // across the K starts, which is exactly what
                        // Rhat measures.
                        let trace: Vec<f64> = r.iterations.iter()
                            .map(|it| it.if2_perturbed_loglik)
                            .collect();
                        let last_iter = r.iterations.last();
                        diag.completed = true_ll.is_finite();
                        diag.iterations_used = last_iter.map(|it| it.iteration + 1);
                        let n_assimilations = {
                            use sim::inference::traits::ObservationModel;
                            let obs: &dyn ObservationModel<sim::inference::ParticleState> = &*obs_model_obj;
                            (0..obs.n_observations()).filter(|&i| obs.assimilates_at(i)).count()
                        };
                        diag.cooling_final = last_iter
                            .map(|it| cooling_final(&config.cooling, n_assimilations, it));
                        diag.loglik_trace = trace;
                        (true_ll, r.mle)
                    }
                    Err(_) => {
                        // diag.completed stays false; iterations_used /
                        // cooling_final / loglik_trace stay at defaults
                        // — the aggregator records NaN for this start's
                        // contribution.
                        (f64::NEG_INFINITY, params.clone())
                    }
                }
            }
            ProfileAlgo::Nlopt(nlopt_algo) => {
                // Per-cell starting point: copy `params` (focal pinned)
                // and overwrite the non-focal estimated slots with the
                // per-start LHS draws. `per_start_specs` contains only
                // non-focal estimated specs, so the focal pin survives.
                let mut full_start = params.clone();
                for spec in per_start_specs {
                    full_start[spec.index] = spec.initial;
                }
                let est_indices: Vec<usize> =
                    per_start_specs.iter().map(|p| p.index).collect();
                let bounds: Vec<(f64, f64)> =
                    per_start_specs.iter().map(|p| p.bounds()).collect();
                let ode_dt = dt;
                // Per-cell NLopt budget: the `--fit` file's nl-* method's,
                // else the profile defaults (`PROFILE_NLOPT_*`).
                let tolerance = nl_tolerance;
                let max_evals = nl_max_evals;
                diag.algo = Some(crate::profile_diagnostics::DiagAlgo::Nlopt);
                match crate::fit::nlopt_method::optimize_cell(
                    nlopt_algo,
                    &compiled,
                    &obs_model_obj,
                    &obs_times_vec,
                    ode_dt,
                    &bounds,
                    &est_indices,
                    &full_start,
                    tolerance,
                    max_evals,
                    // A profile's progress unit is a grid cell, not an
                    // evaluation; `profile` reports its own bar (gh#900).
                    None,
                ) {
                    Ok(r) => {
                        // Reproject the optimizer-space result back into
                        // a full param vector (focal pinned, non-focal
                        // optimized, fixed at base).
                        let mut mle = full_start.clone();
                        for (slot, &model_idx) in est_indices.iter().enumerate() {
                            mle[model_idx] = r.params[slot];
                        }
                        diag.completed = r.loglik.is_finite();
                        (r.loglik, mle)
                    }
                    Err(e) => {
                        out.say(format_args!(
                            "warning: nlopt optimize_cell failed at grid {}/start {}: {}",
                            grid_idx, start_idx, e
                        ));
                        (f64::NEG_INFINITY, params.clone())
                    }
                }
            }
        };
        let elapsed = job_t0.elapsed().as_secs_f64();

        // Report this cell's loglik to the shared best: the drill target for
        // the work-queue and the bar's researcher metric. Never gates the leaf
        // write below (order-independent output).
        if final_loglik.is_finite() {
            if let Ok(mut b) = best.lock() {
                if final_loglik > b.0 {
                    *b = (final_loglik, Some(grid_idx));
                    bar.set(crate::out::progress::best_ll(final_loglik));
                }
            }
        }

        // A non-finite cell loglik is a completed point with `final_loglik =
        // -inf`, and every reader of it (the table, the curve) then has
        // nothing to plot for this grid point. Say so on stderr: a dropped
        // grid point is a result the user must see, not a gap to infer. The
        // usual cause is a particle filter too small to score a θ far from
        // the optimum (the swarm dies and the estimate is −inf);
        // `--particles` is the lever.
        if !final_loglik.is_finite() {
            out.say(format_args!(
                "warning: profile grid point {} (start {}) produced a non-finite \
                 log-likelihood ({}); it is stored with no value and the \
                 start counts as not completed in the profile table (a cell with no \
                 completed start is a row of NaN). This is usually a \
                 particle filter too small to score a θ this far from the optimum — \
                 re-run with more `--particles`.",
                grid_idx, start_idx, final_loglik,
            ));
        }
        let algorithm_payload = match profile_algo {
            ProfileAlgo::If2 => serde_json::json!({
                "particles": n_particles, "iterations": n_iterations,
                "cooling": cooling, "cooling_target_iters": cooling_schedule.target_iters,
                "dt": dt }),
            ProfileAlgo::Nlopt(_) => serde_json::json!({
                "tolerance": nl_tolerance, "max_evals": nl_max_evals,
                "dt": dt }),
        };
        // The run's provenance, with this cell's partition.
        let mut provenance = run_provenance_json.clone();
        provenance["partition"] = cell_partitions[grid_idx].clone();
        let provenance_json = serde_json::json!({
            "method": profile_algo.method_kind().as_str(),
            "seed": job_seed,
            "n_chains": 1,
            "algorithm": algorithm_payload,
            "best_loglik": if final_loglik.is_finite() { Some(final_loglik) } else { None },
            // The class of the profiled loglik follows the profile method
            // (gh#280): if2 / marginal / ode_marginal — a marginal in every
            // case, never PGAS's joint.
            "loglik_type": crate::fit::loglik::LoglikType::from(profile_algo.method_kind()).tag(),
            // Which likelihood `best_loglik` is: `full` scores every
            // observation, `conditional_on_first_observation` (`ic_free`)
            // leaves y₁'s term out. Compare a cell only with a fit of the same
            // objective.
            "likelihood_objective": objective,
            "wall_time_seconds": elapsed,
            "provenance": provenance,
        });
        let Some(run) = refused(run) else { return };
        // Under the forcing override, each forcing this point held outside
        // its range.
        let mut provenance_json = provenance_json;
        let held: Vec<sim::forcing_range::OutsideRange> = compiled.time_func_cache.iter()
            .filter_map(|f| f.range.as_ref())
            .flat_map(|r| r.held())
            .collect();
        forcing.record_held_in(&mut provenance_json, &held);
        let mle_toml = render_mle_toml(&if2_params, &focal_values,
            &focal_grids.iter().map(|fg| fg.name.as_str()).collect::<Vec<_>>(),
            &mle_params, final_loglik, &diag);
        let write = || -> Result<(), String> {
            let e = |e: store::Error| e.to_string();
            run.result_file(crate::profile_store::MLE_FILE).map_err(e)?.write_all(mle_toml.as_bytes()).map_err(e)?;
            run.provenance_file(crate::profile_store::PROVENANCE_FILE).map_err(e)?
                .write_all(&serde_json::to_vec_pretty(&provenance_json).unwrap_or_default()).map_err(e)
        };
        let failure = match write() {
            Ok(()) => run.finish().err().map(|e| e.to_string()),
            Err(reason) => {
                let _ = run.fail("storing", &reason);
                Some(reason)
            }
        };
        if let Some(reason) = failure {
            point_failures.lock().unwrap_or_else(|e| e.into_inner())
                .push(format!("storing profile grid point {grid_idx} (start {start_idx}): {reason}"));
        }

        // Passive progress tick. `Task` handles Pretty (redraw) / Plain
        // (throttled `profile pos/len` log line) / None (no-op) internally.
        bar.inc(1);
    };

    // N workers pull the pending point whose cell is nearest the current best
    // cell, keeping every core busy while evaluation drills toward the
    // optimum. Order-independent (each point is computed from its own
    // identity), so no barrier and no determinism requirement — the pick just
    // reorders which cell a freed core visits next.
    let pending: std::sync::Mutex<Vec<(store::Run<crate::profile_store::ProfilePoint>, usize)>> =
        std::sync::Mutex::new(claimed);
    let worker = || {
        loop {
            // Snapshot the drill target (release the lock before touching
            // `pending`, so the two locks never nest).
            let target = best.lock().map(|b| b.1).unwrap_or(None);
            let (run, ji) = {
                let mut p = match pending.lock() {
                    Ok(p) => p,
                    Err(_) => break,
                };
                if p.is_empty() { break; }
                let pick = match target {
                    // Nearest unevaluated cell to the best so far.
                    Some(bc) => {
                        let mut best_i = 0usize;
                        let mut best_d = f64::INFINITY;
                        for (i, (_, cand)) in p.iter().enumerate() {
                            let d = profile_cell_dist2(&cell_norm[jobs[*cand].1], &cell_norm[bc]);
                            if d < best_d { best_d = d; best_i = i; }
                        }
                        best_i
                    }
                    // No finite best yet — take the next pending point.
                    None => 0,
                };
                p.swap_remove(pick)
            };
            eval_job(run, ji);
            progress();
        }
    };
    // One worker loop per pool thread, placed by `broadcast` in that thread's
    // private queue (gh#821, `fit::chain_fanout`). Spawned as ordinary scope
    // jobs, a not-yet-started loop could be stolen by a worker blocked inside
    // an IF2 run's particle `par_iter`, which would then drain the whole queue
    // on top of the run it had not finished.
    let run_sweep = || {
        log::debug!("profile: computing on {} threads", rayon::current_num_threads());
        rayon::broadcast(|_| worker());
    };
    match &prof_pool {
        Some(pool) => pool.install(run_sweep),
        None => run_sweep(),
    }

    bar.finish();
    failures.extend(point_failures.into_inner().unwrap_or_else(|e| e.into_inner()));
    drop(batch);
    let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    if let Some(refusal) = forcing.refused() {
        out.fail(refusal);
    }
    if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
        out.say(format_args!("profile: @{label} now names invocation {}, which held it from invocation {before}", invocation.seq()));
    }
    if !failures.is_empty() {
        for f in &failures {
            out.say(format_args!("error: {f}"));
        }
        out.say(format_args!("profile: {} of {} point(s) were not stored; no table was written", failures.len(), n_points));
        crate::out::ExitCode(1).exit();
    }

    let n_cells = grid_points.len();
    out.say(format_args!(
        "profile: {} cell{} ({} point{}) stored under {}",
        n_cells, if n_cells == 1 { "" } else { "s" },
        n_points, if n_points == 1 { "" } else { "s" },
        crate::run_paths::display_absolute(store.root()).display()));

    // The profile table: one row per grid cell, rendered from the stored
    // points the invocation asked for, so a rerun served from the store
    // prints the same table.
    let options = crate::profile_store::ProfileTableOptions {
        estimated: if2_params.iter().map(|p| p.name.clone()).collect(),
        objective,
        loglik_type: crate::fit::loglik::LoglikType::from(profile_algo.method_kind()).tag(),
        format: table_export.format,
    };
    match table_export.output {
        Some(path) => {
            invocation.render::<crate::profile_store::ProfileTableRendering>(&store, &options, &store::ExportPath(path.clone()))
                .unwrap_or_else(|e| out.fail(format!("error: cannot write --output {}: {e}", path.display())));
            out.say(format_args!("profile: table written to {}", path.display()));
        }
        None => {
            let mut bytes = Vec::new();
            invocation.render_to::<crate::profile_store::ProfileTableRendering>(&store, &options, &mut bytes)
                .unwrap_or_else(|e| out.fail(format!("error: the profile table: {e}")));
            out.data_raw(String::from_utf8_lossy(&bytes).into_owned());
        }
    }
    crate::bindings::write(out, &bindings, &store, &invocation);
}

/// The values a single-point starts rule supplies, and where they came from:
/// `--starts from_params=<toml>` reads the file, `--starts from_mle=<fit>` the
/// stored fit's point estimate (`fit_state.toml` `start_values`). They enter
/// the resolver at tier 1.5 ([`crate::params_resolver::StartsValues`]),
/// above the model default and below every other tier, so the file's value
/// wins over the DSL default and anything the user passes explicitly
/// (`--fit` `[fixed]`, `--fixed-file`, the scenario, `--fixed`) wins over the
/// file, with the replacement reported.
///
/// Per-chain-varying rules (`from_prior`, `from_posterior`) and the spread
/// rules (single, uniform, lhs) have no single point and return `None`; users
/// of those supply a base value another way (`--fit` with `[estimate].start`,
/// or `--fixed-file`).
///
/// Sorted by name, so the resolver applies them in a reproducible order.
fn starts_values_from_init_method(
    init_method: &crate::fit::starts::ChainStarts,
) -> Result<Option<(crate::params_resolver::StartsOrigin, Vec<(String, f64)>)>, String> {
    use crate::fit::starts::{ChainStarts, Point};
    use crate::params_resolver::StartsOrigin;
    let (origin, values): (StartsOrigin, HashMap<String, f64>) = match init_method {
        ChainStarts::Point(Point::FromParams { path }) => {
            let values = crate::util::load_params_toml(&path.to_string_lossy())
                .map_err(|e| format!(
                    "loading the --starts from_params file: {}", e))?;
            (StartsOrigin::Params { path: path.clone() }, values)
        }
        ChainStarts::Point(Point::FromMle { source }) => {
            // The stored fit's point estimate, from the run the handle names.
            let root = crate::fit_store::reader_root();
            let store = crate::camdl_store::open_existing(&root)
                .ok_or_else(|| format!("--starts from_mle={source}: no run store at {}", root.display()))?;
            let run = crate::fit_store::resolve_run(&store, &source.0)
                .map_err(|e| format!("--starts from_mle={source}: {e}"))?;
            let files = crate::fit_store::RunFiles::of(&run)
                .map_err(|e| format!("--starts from_mle={source}: {e}"))?;
            let state = crate::fit::state::FitState::load_input(&files.path().to_string_lossy())
                .map_err(|e| format!("--starts from_mle={}: {}", source, e))?;
            crate::fit::chain_starts::refuse_sampler_as_mle_source(&state.method, &source.0)?;
            (StartsOrigin::Mle { fit: source.0.clone() },
             state.start_values.into_iter().collect())
        }
        _ => return Ok(None),
    };
    let mut values: Vec<(String, f64)> = values.into_iter().collect();
    values.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Some((origin, values)))
}

/// A point's `mle.toml`: its final log-likelihood, focal values and
/// estimates, which the profile table reads back.
///
/// It ends with a `[diagnostics]` block holding the per-start convergence
/// record.
fn render_mle_toml(
    if2_params: &[sim::inference::if2::EstimatedParam],
    focal_values: &[f64],
    focal_names: &[&str],
    mle: &[f64],
    final_loglik: f64,
    diag: &crate::profile_diagnostics::PerStartDiagnostics,
) -> String {
    let mut body = String::new();
    body.push_str("# Per-start MLE for one profile grid point.\n\n");
    body.push_str(&format!("final_loglik = {}\n", final_loglik));
    body.push('\n');
    body.push_str("[focal]\n");
    for (name, v) in focal_names.iter().zip(focal_values.iter()) {
        body.push_str(&format!("{} = {}\n", name, v));
    }
    body.push_str("\n[mle]\n");
    for spec in if2_params.iter() {
        body.push_str(&format!("{} = {}\n", spec.name, mle[spec.index]));
    }
    body.push_str("\n[diagnostics]\n");
    body.push_str(&diag.to_toml_fragment());
    body
}


/// Today's `profile` selection: the estimated set `--rw-sd` chose (its names,
/// or with `auto` every parameter not swept, held or a constant) and the
/// declarations that held the rest.
struct LegacySelection<'a> {
    estimated: &'a [String],
    focal: &'a [String],
    fit_estimate: &'a indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    /// `Err` when the fit config's `[fixed]` cannot be read.
    fit_fixed: Result<Vec<(String, crate::roles::FitFixedVia)>, String>,
    /// `--fixed` names.
    cli_fixed: &'a [String],
    fixed_files: &'a [std::path::PathBuf],
    /// The model as declared, against which a fit artifact's legacy
    /// constant entries are set aside ([`crate::params_resolver::load_fixed_file`]).
    model: &'a ir::Model,
}

/// [`LegacySelection`] as role declarations: the owned half of a
/// [`crate::roles::RoleInputs`].
struct LegacyRoleDecls {
    fit_estimate: indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    fit_fixed: Vec<(String, crate::roles::FitFixedVia)>,
    cli_estimate: Vec<crate::roles::EstimateDecl>,
    cli_fixed: Vec<String>,
    files: Vec<crate::roles::FixedFileDecl>,
    design: Vec<(String, crate::roles::DesignOrigin)>,
    slice: bool,
}

impl LegacyRoleDecls {
    fn inputs<'a>(
        &'a self,
        model: &'a ir::Model,
        simplex_groups: &'a [crate::fit::config_v2::SimplexGroup],
    ) -> crate::roles::RoleInputs<'a> {
        crate::roles::RoleInputs {
            model,
            fit_estimate: &self.fit_estimate,
            fit_fixed: &self.fit_fixed,
            simplex_groups,
            cli_estimate: &self.cli_estimate,
            cli_fixed: &self.cli_fixed,
            cli_fixed_files: &self.files,
            design: &self.design,
            slice: self.slice,
            // A scenario is a value source in today's `profile`, its values
            // the starting points of the estimated parameters (proposal
            // §3.2 item 11); it is checked against the roles from Stage 3.
            scenarios: &[],
        }
    }
}

/// The one adapter between `profile`'s legacy selection and the partition
/// (proposal `2026-10-26-parameter-roles.md` §11, Stage 1): `roles` never
/// sees `--rw-sd`, and no rule in `resolve_roles` depends on it. Stage 3
/// deletes this and builds the declarations from `--estimate` and
/// `[estimate]` directly.
///
/// The translation states today's outcome as declarations:
///
/// - an estimated name is its `[estimate]` entry, else an `--estimate`
///   declaration with no bounds of its own;
/// - a focal name keeps its `[estimate]` entry, which makes it the focal
///   parameter of the profile;
/// - an `[estimate]` name neither estimated nor focal is held: it is left
///   undeclared, fixed by default, or held by the `--fixed` that named it;
/// - a plain `--fixed-file`'s entries for a focal name are dropped (the grid
///   writes over them today); a fit artifact's are kept, and the partition
///   lists them as unused;
/// - nothing estimated is a slice.
fn legacy_role_inputs(sel: LegacySelection<'_>) -> Result<LegacyRoleDecls, String> {
    let kept = |name: &String| sel.estimated.contains(name) || sel.focal.contains(name);
    // In the estimated set's order, which is the order of the method's
    // parameters, then the focal entries.
    let fit_estimate = sel.estimated.iter().chain(sel.focal.iter())
        .filter(|n| kept(n))
        .filter_map(|n| sel.fit_estimate.get(n).map(|s| (n.clone(), s.clone())))
        .collect();
    let cli_estimate = sel.estimated.iter()
        .filter(|n| !sel.fit_estimate.contains_key(*n))
        .map(|n| crate::roles::EstimateDecl { name: n.clone(), bounds: None })
        .collect();
    let fit_fixed = sel.fit_fixed?;
    let cli_fixed = sel.cli_fixed.to_vec();
    let mut files = Vec::new();
    for path in sel.fixed_files {
        let fit_artifact = crate::params_resolver::carries_fit_provenance(path);
        let mut names: Vec<String> = crate::params_resolver::load_fixed_file(sel.model, path)
            .map_err(|e| e.to_string())?.0
            .into_keys()
            .filter(|n| fit_artifact || !sel.focal.contains(n))
            .collect();
        names.sort();
        files.push(crate::roles::FixedFileDecl { path: path.clone(), names, fit_artifact });
    }
    Ok(LegacyRoleDecls {
        fit_estimate,
        fit_fixed,
        cli_estimate,
        cli_fixed,
        files,
        design: sel.focal.iter()
            .map(|n| (n.clone(), crate::roles::DesignOrigin::Sweep))
            .collect(),
        slice: sel.estimated.is_empty(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D21: `cooling_final` is the last iteration's cooling factor, the same
    /// for every parameter, and not a mean of the parameters' SDs. For
    /// cooling 0.9 over 50 iterations, the last of 20 iterations (index 19)
    /// ends at factor 0.9^(20/50) = 0.9587; with steps 0.1 and 0.3 the mean
    /// of their cooled SDs would be 0.1917.
    #[test]
    fn cooling_final_is_the_last_iterations_cooling_factor() {
        use sim::inference::if2::{CoolingSchedule, IF2IterResult, ParamIterDiag};
        let schedule = CoolingSchedule { fraction: 0.9, target_iters: 50 };
        let n_assimilations = 14;
        let factor = schedule.multiplier_at_iter(n_assimilations, 20);
        let diag = |sd: f64| ParamIterDiag { param_index: 0, weighted_var_ratio: 1.0, q_ratio: 1.0, effective_sd: sd * factor };
        let last = IF2IterResult {
            iteration: 19, loglik: f64::NAN, if2_perturbed_loglik: -50.0,
            param_means: vec![], param_diag: vec![diag(0.1), diag(0.3)],
        };
        let got = cooling_final(&schedule, n_assimilations, &last);
        assert_eq!(got, factor);
        assert!((got - 0.9_f64.powf(20.0 / 50.0)).abs() < 1e-12, "{got}");
    }

    /// The resolver's result is checked against the stored problem: an
    /// estimated set or a `[fixed]` value that differs, by one bit, is an
    /// error naming the parameter.
    #[test]
    fn a_resolution_unlike_the_stored_problem_is_refused() {
        let cfg = crate::fit::config_v2::FitConfig::from_toml_str(
            "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.1, 2.0] }\n[fixed]\nN0 = 1000\n").unwrap();
        let estimate = cfg.problem.estimate.clone();
        let fixed: indexmap::IndexMap<String, f64> = [("N0".to_string(), 1000.0)].into_iter().collect();
        let model = |n0: f64| move |name: &str| match name { "beta" => Some(0.5), "N0" => Some(n0), _ => None };
        let estimated: indexmap::IndexSet<String> = ["beta".to_string()].into_iter().collect();
        assert_eq!(resolution_matches_problem(&estimated, model(1000.0), &estimate, &fixed), Ok(()));
        let off_by_a_bit = f64::from_bits(1000.0f64.to_bits() + 1);
        let e = resolution_matches_problem(&estimated, model(off_by_a_bit), &estimate, &fixed).unwrap_err();
        assert!(e.starts_with("N0 resolved to"), "{e}");
        let none: indexmap::IndexSet<String> = indexmap::IndexSet::new();
        let e = resolution_matches_problem(&none, model(1000.0), &estimate, &fixed).unwrap_err();
        assert!(e.contains("not the stored problem's [estimate] (beta)"), "{e}");
    }

    /// An nl-* profile optimizes each cell with the `--fit` file's own
    /// tolerance and budget when its [method] is an nl-* optimizer; it used to
    /// hard-code 1e-4 / 1500 and ignore them.
    #[test]
    fn nlopt_cells_use_the_fit_methods_budget() {
        let cfg = crate::fit::config_v2::FitConfig::from_toml_str(
            "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.1, 2.0] }\n\
             [fixed]\nN0 = 1000\n[method]\nalgorithm = \"nl-sbplx\"\nbackend = \"ode\"\n\
             chains = 2\ntolerance = 1e-7\nmax_evals = 900\n").unwrap();
        let method = cfg.inference.method.unwrap();
        assert_eq!(nlopt_cell_budget(Some(&method.algorithm)), (1e-7, 900, true));
        assert_eq!(nlopt_cell_budget(None),
            (PROFILE_NLOPT_TOLERANCE, PROFILE_NLOPT_MAX_EVALS, false));
    }

    // ── starts_values_from_init_method: the starts rule's point ───────
    //
    // A single-point starts rule (`from_params=FILE`, `from_mle=FIT`) loads
    // one parameter point, which enters the resolver as its own tier. These
    // pin what is loaded and how it is attributed; the tier's precedence is
    // pinned by the resolver's own tests
    // (`params_resolver::tests::starts_values_*`).

    fn write_toml(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn starts_from_params_loads_the_file_named_by_its_path() {
        let tmp = tempfile::tempdir().unwrap();
        let toml_path = write_toml(tmp.path(), "start.toml",
            "gamma = 0.1\nbeta = 0.5\n");
        let init = crate::fit::starts::ChainStarts::Point(
            crate::fit::starts::Point::FromParams { path: toml_path.clone() });
        let (origin, values) = starts_values_from_init_method(&init).unwrap().unwrap();
        assert_eq!(origin, crate::params_resolver::StartsOrigin::Params { path: toml_path });
        assert_eq!(values, vec![("beta".to_string(), 0.5), ("gamma".to_string(), 0.1)],
            "every value in the file, sorted by name");
    }

    #[test]
    fn starts_without_a_single_point_supply_no_values() {
        // from_prior / lhs have per-chain-varying values — there is no
        // single point to supply to the resolver.
        let init_prior = crate::fit::starts::ChainStarts::from_prior();
        assert!(starts_values_from_init_method(&init_prior).unwrap().is_none(),
            "from_prior must supply no base values — they are per-chain");
        let init_lhs = crate::fit::starts::ChainStarts::Spread(crate::fit::starts::Spread::Lhs);
        assert!(starts_values_from_init_method(&init_lhs).unwrap().is_none(),
            "lhs must supply no base values");
    }

    // ── render_mle_toml: the per-start leaf body ──────────────────────

    fn estimated_param(name: &str, index: usize) -> sim::inference::if2::EstimatedParam {
        sim::inference::if2::EstimatedParam {
            name: name.into(), index,
            initial: 0.5,
            transform: sim::inference::types::Transform::Interval { lo: 0.0, hi: 1.0 },
        }
    }

    #[test]
    fn render_mle_toml_writes_loglik_focal_mle_and_diagnostics() {
        // The whole body of one IF2 / nl-* point, pinned byte for byte: the
        // profile table reads it back.
        let if2 = vec![estimated_param("beta", 0)];
        let diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let body = render_mle_toml(&if2, &[25.0], &["R0"], &[0.42], -11.0, &diag);
        assert_eq!(
            body,
            "# Per-start MLE for one profile grid point.\n\n\
             final_loglik = -11\n\n\
             [focal]\nR0 = 25\n\n\
             [mle]\nbeta = 0.42\n\n\
             [diagnostics]\ncompleted = false\n",
        );
    }

}

