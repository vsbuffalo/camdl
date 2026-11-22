//! `camdl fit run` on the run store (`docs/dev/proposals/2026-11-20-run-store-api.md`,
//! "Worked example: `camdl fit run`").
//!
//! ```text
//! FitRunInvocation ──split──► declared │ execution │ request │ given
//!        declared ──run_config::resolve──► ResolvedFitConfig
//!                                            │
//!     per dataset × sweep point: resolve_problem ──► FitProblem (atomic, written once)
//!                                            │
//!                  per seed: FitRunIdentity ──► FitRun (in place, under `runs/`)
//!                                            │
//!                    fit::run_method in a working directory, ingested at the end
//! ```
//!
//! A fit is a problem and the runs on it. The problem holds the training rows
//! of each bound data file, the held-out rows' dates, the tables, the
//! estimated set and the settings; it is stored once and every method run on
//! it shares it. Each run holds the method's settings, `rw_sd`,
//! `perturb_only_at_t0` and `start` of each estimated parameter, its seed,
//! where its chains start, and its length; a resume is a new run with a longer
//! length whose identity holds a dependency on the base's carried chain files.
//!
//! The method writes its files into a working directory outside the store,
//! because every runner writes, and reads back, its files by path. While it
//! runs, its progress snapshot and each line of its event log are passed to
//! the run's own progress and event channels; when it ends, every file of the
//! working directory is written into the run, and the run is finished. A
//! failure the run's identity determines is not distinguished from one a
//! rerun could change: a method that stops with an error leaves the run
//! failed, and the next run of the command computes it again.

// The store's error carries the paths and owners a person acts on, and is
// returned once per result.
#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use store::{Begin, ContentAddressed, ContentArtifact, Identity, Reuse, Stored};

use crate::camdl_store::{CompiledModel, ProcessSeed};
use crate::fit::config_v2::{self, Algorithm};
use crate::fit_store::{
    ChainIdentity, FitProblem, FitRun, FitRunIdentity, MethodSettings, ModelSourceFile, ProblemLabels,
    RunLabels, StartInput, StartSource,
};
use crate::out::Out;

/// The files a resume carries from its base, per chain.
const CARRIED: [&str; 3] = crate::fit::resumed_chain::CARRIED_CHAIN_FILES;

/// `camdl fit run`: its inputs split by class, the configuration resolved from
/// the declared inputs, and the problems and runs it names computed on the
/// run store.
pub(crate) fn cmd_fit_run(inv: crate::args::FitRunInvocation) {
    let out = Out::new(false);
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start(); // gh#audit-H5
    // Removed flags fail before any work, each naming its replacement.
    crate::fit::run_config::refuse_removed_flags(&inv.args, &out);
    let removed = RemovedFlags::of(&inv.args);
    let (declared, execution, _exports, request, given) = store::Inputs::split(inv).into_parts();
    let (declared, no_licm) = declared.into_parts();
    let (config, problem, run, method) = declared.into_parts();
    let (forcing_extrapolation, emit_every, sweep) = problem.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let (seed, starts, starts_chains, resume) = run.into_parts();
    let (
        decibans_thresh,
        no_dt_check,
        dt_check_strict,
        binomial,
        dt_check_halvings,
        cooling_target_iters,
        max_tree_depth,
        trajectory_warmup,
        csmc_sweeps_per_nuts,
        n_trajectories,
        diagonal_mass,
        no_nuts,
        no_ancestor_sampling,
        no_adapt,
        adapt_start,
        record_ancestry,
        record_prequential,
    ) = method.into_parts();
    let crate::args::FitRunInvocationExecution { args: run_execution, no_ir_cache } = &execution;
    let compile = crate::util::CompileSettings::resolve(no_licm, *no_ir_cache);
    let (request,) = request.into_parts();
    let (_problem_request, _run_request, _method_request, force, label) = request.into_parts();

    // The resolution reads the arguments reassembled, every declared input
    // placed back in the struct it reads.
    let a = crate::args::FitRunArgs {
        config: config.clone(),
        problem: crate::args::FitProblemFlags {
            forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
            emit_every,
            sweep,
        },
        run: crate::args::FitRunFlags { seed, starts, starts_chains, resume },
        method: crate::args::FitMethodFlags {
            decibans_thresh,
            no_dt_check,
            dt_check_strict,
            binomial,
            dt_check_halvings,
            cooling_target_iters,
            max_tree_depth,
            trajectory_warmup,
            csmc_sweeps_per_nuts,
            n_trajectories,
            diagonal_mass,
            no_nuts,
            no_ancestor_sampling,
            no_adapt,
            adapt_start,
            record_ancestry,
            record_prequential,
        },
        allow_nonconverged_source: run_execution.allow_nonconverged_source,
        parallel: run_execution.parallel,
        force,
        label: label.clone(),
        ..removed.into_args()
    };

    // gh#162: a fit nests Rayon parallelism (chains × particle filter) on the
    // global pool; `--parallel` / CAMDL_PARALLEL caps it (0 = all cores).
    if a.parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.parallel).build_global();
    }
    log::debug!("fit run: running on {} threads", rayon::current_num_threads());
    if let Some(raw) = label.as_deref() {
        if let Err(e) = crate::fit::validate_label(raw) {
            out.fail(format!("error: invalid --label: {e}"));
        }
    }

    // The fit file, read once: the text every step parses and what each run
    // keeps as its `fit.toml`.
    let source = store::SourceBytes::read(&config)
        .unwrap_or_else(|e| out.fail(format!("error: cannot read {}: {e}", config.display())));
    let text = String::from_utf8(source.bytes().to_vec())
        .unwrap_or_else(|e| out.fail(format!("error: {}: {e}", config.display())));
    let (resolved, _forcing_route) = crate::fit::run_config::resolve(&a, text, compile, &out);
    refuse_before_storing(&out, &resolved, &config.to_string_lossy());

    // The run store: the file's `store` key, then `CAMDL_STORE`, then the
    // default.
    let env = store::StoreRequest::new(None, false, crate::env::get().vars.store.clone());
    let request = store::StoreRequest::new(label, force, resolved.store_root.clone()).merge(env);
    let store = crate::camdl_store::open(&request).unwrap_or_else(|e| {
        out.fail(format!("error: the run store at {}: {e}", crate::camdl_store::root_of(&request).display()))
    });
    let given = given.with_file("config", store::Given::default(), &source);
    let info = store::RunInfo::new(&given, &execution, &request).unwrap_or_else(|e| out.fail(format!("error: {e}")));

    let fit_path = config.to_string_lossy().into_owned();
    let failures = run_fit(out, FitCommand { fit_path, source, resolved, store: &store, info: &info, reuse: request.reuse() });
    let out = Out::new(false);
    match info.finish(&store) {
        Ok(invocation) => {
            if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
                out.say(format_args!("note: @{label} now names run(s) of invocation {}, which took it from invocation {before}",
                    invocation.seq()));
            }
        }
        Err(e) => out.fail(format!("error: recording the invocation: {e}")),
    }
    if !failures.is_empty() {
        out.fail(format!("error: {} of this invocation's problems and runs did not complete:\n  {}",
            failures.len(), failures.join("\n  ")));
    }
}

/// The skipped removed-flag fields, which the split does not carry; the
/// refusal above has already answered every one that was given.
struct RemovedFlags;

impl RemovedFlags {
    fn of(_a: &crate::args::FitRunArgs) -> Self {
        RemovedFlags
    }

    fn into_args(self) -> crate::args::FitRunArgs {
        crate::args::FitRunArgs {
            config: PathBuf::new(),
            problem: Default::default(),
            run: Default::default(),
            method: Default::default(),
            allow_nonconverged_source: false,
            parallel: 0,
            force: false,
            label: None,
            _removed_stage: None,
            _removed_init: None,
            _removed_posterior: None,
            _removed_mle: None,
            _removed_params: None,
            _removed_survey_path: None,
            _removed_survey_top_k: None,
            _removed_allow_nonconverged_scout: false,
            _removed_init_method: None,
            _removed_starts_from: None,
            _removed_tempering: None,
            _removed_rho: None,
            _removed_output_dir: None,
        }
    }
}

/// What one `fit run` computes with, past its resolution.
struct FitCommand<'a> {
    /// The fit file as typed.
    fit_path: String,
    /// Its bytes, as read: what each run keeps as its `fit.toml`.
    source: store::SourceBytes,
    resolved: crate::fit::run_config::ResolvedFitConfig,
    store: &'a store::Store,
    info: &'a store::RunInfo,
    reuse: Reuse,
}

/// One problem of the invocation: a dataset (the file's data, or one
/// generated by `[synthetic]`) at one sweep point.
struct ProblemCell {
    /// The problem the methods run on, the sweep point's values in `[fixed]`.
    problem: config_v2::Problem,
    sweep_point: Vec<(String, f64)>,
    dataset: Option<usize>,
}

/// Run the fit `cx.resolved` describes: each problem stored once, each run on
/// it computed or reused. Returns one line per problem or run that did not
/// complete.
/// Refuses, before the store is opened, what the fit file and the flags ask
/// that cannot run: a sweep over an estimated or undeclared parameter, and a
/// scenario that sets a parameter `[estimate]`, `[fixed]` or the sweep also
/// sets. A refused run leaves nothing behind.
fn refuse_before_storing(out: &Out, resolved: &crate::fit::run_config::ResolvedFitConfig, fit_path: &str) {
    let config = &resolved.config;
    let model = &resolved.model;
    let sweep_specs = &resolved.sweep;
    // ── Validate sweeps ───────────────────────────────────────────────────
    let fixed_resolved = config.problem.fixed.resolve().unwrap_or_default();
    for (name, _) in sweep_specs.iter() {
        if config.problem.estimate.contains_key(name) {
            out.fail(format!("error: cannot sweep '{name}' — it is in [estimate].\n  \
                       Sweeps override [fixed] parameters. Move '{name}' to [fixed] first."));
        }
        if !fixed_resolved.contains_key(name) {
            out.fail(format!("error: sweep parameter '{}' not found in [fixed].\n  \
                       Available fixed params: {}",
                name, fixed_resolved.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
    }

    // ── Refuse a scenario that disagrees with [estimate], [fixed] or the sweep
    // The resolver puts the scenario above both [fixed] and a sweep point
    // (spec §1.3); on the same parameter that would silently discard what
    // the user wrote (gh#572).
    if let Some(scenario) = config.problem.scenario.as_deref() {
        let mut refusals: Vec<String> = Vec::new();
        let estimated: Vec<&str> = config.problem.estimate.keys().map(|k| k.as_str()).collect();
        refusals.extend(crate::params_resolver::scenario_estimate_collision(
            model, scenario, &estimated, crate::params_resolver::EstimatedBy::FitToml(fit_path),
        ).unwrap_or_else(|e| out.fail(format!("error: {e}"))));
        refusals.extend(crate::params_resolver::scenario_fixed_collision(
            model, scenario, &fixed_resolved, fit_path,
        ).unwrap_or_else(|e| out.fail(format!("error: {e}"))));
        if !sweep_specs.is_empty() {
            let varied = crate::params_resolver::coordinate_values(
                sweep_specs.iter().flat_map(|(k, vs)| vs.iter().map(move |v| (k.as_str(), *v))),
            );
            refusals.extend(crate::params_resolver::scenario_coordinate_collision(
                model,
                &crate::sim_job::ScenarioRef::Named(scenario.to_string()),
                crate::params_resolver::UserCoordinate::Sweep,
                &varied,
            ).unwrap_or_else(|e| out.fail(format!("error: {e}"))));
        }
        if !refusals.is_empty() {
            out.fail(format!("error: {}", refusals.join("\n\nerror: ")));
        }
    }

}

fn run_fit(mut out: Out, cx: FitCommand<'_>) -> Vec<String> {
    let FitCommand { fit_path, source, resolved, store, info, reuse } = cx;
    let crate::fit::run_config::ResolvedFitConfig {
        config_text,
        config,
        model,
        tables,
        forcing,
        seeds: fit_seeds,
        sweep: sweep_specs,
        emit_every,
        resume,
        store_root: _,
        problem_ir,
        nuts_tuning_flags,
        fixed_via,
        starts: crate::fit::run_config::StartsPlan { resolution: starts_resolution, allow_nonconverged_source, origin: starts_origin },
    } = resolved;
    let method: config_v2::Method = config.inference.method.clone().expect("resolved with a method");
    let method_name = method.algorithm.method_name();
    let mut failures: Vec<String> = Vec::new();

    let fixed_resolved = config.problem.fixed.resolve().unwrap_or_default();

    // The fit's partition into estimated and fixed, from its role
    // declarations; recorded in each run's `fit.meta.json`.
    let fit_roles = crate::fit::resolve_fit_roles(&config.problem, &model, fixed_via, &sweep_specs);

    // Expand the Cartesian product of the sweep points.
    let sweep_points: Vec<Vec<(String, f64)>> = {
        let mut points: Vec<Vec<(String, f64)>> = vec![vec![]];
        for (name, values) in &sweep_specs {
            points = points
                .iter()
                .flat_map(|pt| values.iter().map(move |&v| {
                    let mut p = pt.clone();
                    p.push((name.clone(), v));
                    p
                }))
                .collect();
        }
        points
    };
    let has_sweep = sweep_points.len() > 1;
    if has_sweep {
        out.say(format_args!("sweep: {} points", sweep_points.len()));
    }

    // gh#191: gate the model's required capabilities against the method's
    // declared backend, before any fitting work, and decide which kernel's
    // band the method's acceptance diagnostics apply.
    let acceptance_kernel = crate::fit::capability_gate(&out, &method, &model, &config, &config_text,
        &nuts_tuning_flags, &fit_path);

    // gh#604: every key under `[data.observations]` / `[data.holdout]` must
    // name a declared observation source, checked before any file is read.
    if let Ok(ds) = config.problem.data_spec() {
        if let Err(e) = crate::fit::stream_bindings::StreamBindings::resolve(&model, ds) {
            out.fail(format!("error: {e}"));
        }
    }

    // ── The compiled model the problem is posed on ───────────────────────
    let model_path = config.problem.model.camdl.clone();
    let model_bytes = std::fs::read(&model_path)
        .unwrap_or_else(|e| out.fail(format!("error: cannot read the model {model_path}: {e}")));
    let model_stem = crate::hashing::path_stem_slug(&model_path).unwrap_or_else(|| "model".to_string());
    let model_source = || ModelSourceFile { bytes: model_bytes.clone(), camdl: crate::util::model_is_camdl_source(&model_path) };
    // The quantities part of the compile the runs run, which each run's
    // labels in this invocation record.
    let run_quantities: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    let put_model = |ir_path: &str| -> store::Dep<CompiledModel> {
        let text = std::fs::read_to_string(ir_path)
            .unwrap_or_else(|e| out.fail(format!("error: cannot read the compiled model {ir_path}: {e}")));
        let envelope = ir::envelope_from_str(&text)
            .unwrap_or_else(|e| out.fail(format!("error: IR load error from {ir_path}: {e}")));
        let parts = crate::camdl_store::CompiledParts::split(
            envelope, model_stem.clone(),
            crate::camdl_store::ModelSource { bytes: &model_bytes, path: Path::new(&model_path) },
        ).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let crate::camdl_store::CompiledParts { model: part, quantities, labels, provenance } = parts;
        // The quantities and contrasts do not change the fit, so the problem
        // names them as lineage: `fit predict` evaluates them.
        let quantities = store.put::<crate::camdl_store::Quantities>(quantities, &labels, &store::Provenance::none(), info)
            .and_then(|s| s.dep(crate::camdl_store::Quantities::FILE))
            .unwrap_or_else(|e| out.fail(format!("error: storing the quantities: {e}")));
        info.lineage(&quantities);
        *run_quantities.borrow_mut() = Some(quantities.id().key().to_hex());
        store.put::<CompiledModel>(part, &labels, &provenance, info)
            .and_then(|s| s.dep(CompiledModel::FILE))
            .unwrap_or_else(|e| out.fail(format!("error: storing the compiled model: {e}")))
    };
    let run_ir = config.problem.compiled_model().unwrap_or_else(|e| out.fail(format!("error: {e}"))).to_string();
    let (problem_model, problem_model_dep, jacobian_model): (ir::Model, store::Dep<CompiledModel>, Option<store::Dep<CompiledModel>>) =
        match &problem_ir {
            None => (model.clone(), put_model(&run_ir), None),
            Some(lean) => {
                let (lean_model, lean_json) = crate::util::load_ir(lean).unwrap_or_else(|e| out.fail(format!("error: {e}")));
                let (_, full_json) = crate::util::load_ir(&run_ir).unwrap_or_else(|e| out.fail(format!("error: {e}")));
                // The two compiles must be one model: the structural hash,
                // which ignores derivative tables and hoisting names, agrees.
                if crate::resolve::model_identity_from_ir(&lean_json) != crate::resolve::model_identity_from_ir(&full_json) {
                    out.fail("error: internal error: the compile with the state Jacobian is not the same model \
                              as the ordinary compile");
                }
                (lean_model, put_model(lean), Some(put_model(&run_ir)))
            }
        };

    // The opening block, as a record first: each run keeps it.
    let run_header = crate::fit::report::RunHeader {
        config: fit_path.to_string(),
        method: method_name.to_string(),
        model: config.problem.model.camdl.clone(),
        estimate: config.problem.estimate.keys().cloned().collect(),
        fixed: fixed_resolved.keys().cloned().collect(),
        starts: starts_resolution.describe(),
        output: crate::fit::report::display_path(store.root()),
    };
    out.say(format_args!("fit: {} (method={})", run_header.config, run_header.method));
    out.say(format_args!("  model:    {}", run_header.model));
    out.say(format_args!("  estimate: {}", run_header.estimate.join(", ")));
    out.say(format_args!("  fixed:    {}", run_header.fixed.join(", ")));
    out.say(format_args!("  starts:   {}", run_header.starts));
    out.say(format_args!("  output:   {}", run_header.output));
    crate::fit::say_ic_free(&out, &config, &model);

    // ── The problems: one per dataset and sweep point ─────────────────────
    let work_root = tempfile::Builder::new().prefix("camdl-fit-").tempdir()
        .unwrap_or_else(|e| out.fail(format!("error: cannot make a working directory: {e}")));
    let datasets: Vec<crate::fit::synthetic::SyntheticDataset> = match &config.problem.synthetic {
        Some(spec) if config.problem.is_synthetic_fit() => {
            let dir = work_root.path().join("synthetic");
            let datasets = crate::fit::synthetic::generate_synthetic_datasets(
                spec,
                config.problem.compiled_model().unwrap_or_else(|e| out.fail(format!("error: {e}"))),
                &dir,
                config.problem.config.dt,
                config.problem.config.eval_settings(),
                emit_every.as_ref(),
            ).unwrap_or_else(|e| out.fail(format!("error: synthetic-data generation failed: {e}")));
            out.say(format_args!("synthetic: generated {} dataset{}; each is a problem's training rows",
                datasets.len(), if datasets.len() == 1 { "" } else { "s" }));
            datasets
        }
        _ => Vec::new(),
    };
    let mut cells: Vec<ProblemCell> = Vec::new();
    let data_problems: Vec<(Option<usize>, config_v2::Problem)> = if datasets.is_empty() {
        vec![(None, config.problem.clone())]
    } else {
        let spec = config.problem.synthetic.clone().expect("datasets are generated from a [synthetic] table");
        let seeds = spec.sim_seeds.to_vec().unwrap_or_else(|e| out.fail(format!("error: {e}")));
        datasets.iter().map(|ds| {
            let mut p = config.problem.clone();
            // The dataset's own generation: the table at its one seed.
            p.synthetic = Some(config_v2::SyntheticSpec {
                sim_seeds: config_v2::SeedsSpec::List(vec![seeds[ds.idx - 1]]),
                datasets: None,
                ..spec.clone()
            });
            p.data = Some(config_v2::DataSpec {
                file: None,
                observations: ds.files.iter().map(|(s, path)| (s.clone(), path.to_string_lossy().into_owned())).collect(),
                holdout_after: None,
                holdout: None,
                // Generated data replaces the observations only: the model's
                // external inputs stay bound to the files the config names.
                tables: config.problem.data.as_ref().map(|d| d.tables.clone()).unwrap_or_default(),
            });
            (Some(ds.idx), p)
        }).collect()
    };
    for (dataset, problem) in data_problems {
        for point in &sweep_points {
            let mut p = problem.clone();
            for (name, val) in point {
                p.fixed.values.insert(name.clone(), *val);
            }
            cells.push(ProblemCell { problem: p, sweep_point: point.clone(), dataset });
        }
    }
    let total_runs = cells.len() * fit_seeds.len();
    if total_runs > 1 {
        out.say(format_args!("grid: {} problem{} × {} seed{}", cells.len(), if cells.len() == 1 { "" } else { "s" },
            fit_seeds.len(), if fit_seeds.len() == 1 { "" } else { "s" }));
    }

    // ── Where the chains start ────────────────────────────────────────────
    let starts_rule = method.starts().cloned().unwrap_or_else(|e| out.fail(format!("error: {e}")));
    log::debug!("fit run: a starts source's verdict is lifted: {allow_nonconverged_source}");
    let resolved_starts = crate::fit::chain_starts::resolve_starts(&starts_rule, allow_nonconverged_source, Some(store))
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let starts_source_record: serde_json::Value = match &resolved_starts.source {
        None => serde_json::Value::Null,
        Some(src) => serde_json::json!({
            "handle": starts_rule.source(),
            "chains": starts_rule.chains().map(crate::fit::starts::kept_ids),
            "run_id": src.run_id(),
        }),
    };
    if let Some(src) = &resolved_starts.source {
        match src.run_id() {
            Some(id) => out.say(format_args!("  starts source: run {} ({})", &id[..12],
                src.file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())),
            None => out.say(format_args!("  starts source: {}", src.file.display())),
        }
    }
    // A pfilter scores one θ, so a point source must name every estimated
    // parameter.
    if matches!(method.algorithm, Algorithm::PFilter { .. }) {
        if let Some(src) = &resolved_starts.source {
            let estimated: Vec<&str> = config.problem.estimate.keys().map(|k| k.as_str()).collect();
            if let Err(e) = crate::fit::refuse_pfilter_point_source_missing(&resolved_starts.rule, &src.file, &estimated) {
                out.fail(format!("error: {starts_origin}: {e}"));
            }
        }
    }
    let start_input = resolved_starts.source.as_ref().map(|s| s.input.clone()).unwrap_or(StartInput::None);
    let starts_identity = StartSource::of(&method, start_input).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let (method_settings, length) = MethodSettings::of(&method).unwrap_or_else(|e| out.fail(format!("error: {e}")));

    // ── Each problem, and the runs on it ──────────────────────────────────
    for (cell_i, cell) in cells.iter().enumerate() {
        if has_sweep || cells.len() > 1 {
            let slug = point_slug(&cell.sweep_point);
            out.say(format_args!("\n═══ problem {}/{}{}{} ═══", cell_i + 1, cells.len(),
                cell.dataset.map(|i| format!(": ds_{i:02}")).unwrap_or_default(),
                if slug.is_empty() { String::new() } else { format!(" {slug}") }));
        }
        // IC4 (2026-04-19 inference review): a prior × transform combination
        // that silently produces a different prior than the user wrote.
        if let Err(e) = crate::fit::runner::validate_prior_transform_compat(&cell.problem.estimate, &model) {
            out.fail(format!("error: {e}"));
        }
        let labels = ProblemLabels {
            model: model_stem.clone(),
            sweep_point: (!cell.sweep_point.is_empty()).then(|| point_slug(&cell.sweep_point)),
        };
        let origin = if cell.dataset.is_some() { crate::fit_store::DataOrigin::Generated } else { crate::fit_store::DataOrigin::Files };
        let posed_problem = crate::fit_store::resolve_problem(&cell.problem, &problem_model, problem_model_dep.clone(),
            model_source(), origin)
            .unwrap_or_else(|e| out.fail(format!("error: {fit_path}: {e}")));
        // The one place `fit run` says which rows train; the build is the one
        // each method's runner makes.
        let window = crate::fit::runner::FitRunConfig::build(
            &cell.problem, Some(&method.algorithm), 1, 1, 1, 1.0, 1, fit_seeds[0], false,
        ).ok();
        if let Some(w) = &window {
            w.partition_lines.say();
        }
        let stored_problem: Stored<FitProblem> = match store.begin::<FitProblem>(&posed_problem.identity, &labels, info, Reuse::IfCompleted) {
            Ok(Begin::Reused(stored)) => stored,
            Ok(Begin::Busy(owner)) => {
                failures.push(format!("problem {}: {}", cell_i + 1, store::Error::InProgress(owner)));
                continue;
            }
            Ok(Begin::Claimed(mut run)) => match crate::fit_store::write_problem(&mut run, &posed_problem) {
                Ok(()) => match run.finish() {
                    Ok(stored) => stored,
                    Err(e) => {
                        failures.push(format!("problem {}: {e}", cell_i + 1));
                        continue;
                    }
                },
                Err(e) => {
                    let _ = run.fail("writing the problem", &e);
                    failures.push(format!("problem {}: {e}", cell_i + 1));
                    continue;
                }
            },
            Err(e) => out.fail(format!("error: storing the problem: {e}")),
        };
        // Which rows trained, as the stored problem records it. The methods
        // run on the problem as resolved in memory, so the rows the runner's
        // build trains on must be the stored problem's.
        let partition = crate::fit_store::read_fit(store, &store::Handle::IdPrefix(stored_problem.id().key().to_hex()))
            .map(|f| f.partition)
            .unwrap_or_else(|e| out.fail(format!("error: reading the stored problem: {e}")));
        if let Some(w) = &window {
            let calendar = ir::caltime::CalendarAxis::from_model(&w.model).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let ran = w.training_partition.record(&calendar);
            if serde_json::to_value(&ran).ok() != serde_json::to_value(&partition).ok() {
                out.fail(format!(
                    "error: internal error: the rows the method trains on are not the stored problem's training \
                     rows ({:?} against {:?})",
                    ran.streams, partition.streams));
            }
        }
        let (_, perturbation) = crate::fit_store::split_estimate(&cell.problem.estimate)
            .unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let data_digests = crate::fit::read_data_digests(cell.problem.data_spec().unwrap_or_else(|e| out.fail(format!("error: {e}"))), &model)
            .unwrap_or_else(|e| out.fail(format!("error: {e}")))
            .with_tables(&tables);

        for &seed in &fit_seeds {
            if fit_seeds.len() > 1 {
                out.say(format_args!("\n━━━ fit_seed={seed} ━━━"));
            }
            out.say(format_args!("\n── method: {method_name} ──"));
            let chain = ChainIdentity {
                problem: stored_problem.dep_all(),
                method: method_settings.clone(),
                perturbation: perturbation.clone(),
                jacobian_model: jacobian_model.clone(),
                seed: ProcessSeed { seed },
                starts: starts_identity.clone(),
            };
            // A resume: the base's chain identity must be this run's, and its
            // length shorter than this run's.
            let base: Option<ResumeBase> = resume.as_deref().map(|handle| {
                ResumeBase::resolve(store, handle, &chain, length, &method.algorithm)
                    .unwrap_or_else(|e| out.fail(format!("error: --resume: {e}")))
            });
            let identity = FitRunIdentity {
                chain,
                length,
                resumed_from: base.as_ref().map(|b| b.dep.clone()),
            };
            let run_labels = RunLabels { method: method_name.to_string(), seed, quantities: run_quantities.borrow().clone() };
            let mut run = match store.begin::<FitRun>(&identity, &run_labels, info, reuse) {
                Ok(Begin::Reused(stored)) => {
                    out.say(format_args!("  {}", crate::out::paint::stderr().paint(
                        &format!("cache hit — reusing {}", relative(stored.folder(), store.root())),
                        crate::out::paint::Style::Warn)));
                    continue;
                }
                Ok(Begin::Busy(owner)) => {
                    failures.push(format!("{method_name} seed {seed}: {}", store::Error::InProgress(owner)));
                    continue;
                }
                Ok(Begin::Claimed(run)) => run,
                Err(e) => out.fail(format!("error: claiming the run: {e}")),
            };
            let ran = (|| -> Result<(), String> {
                let e = |e: store::Error| e.to_string();
                run.provenance_file(crate::fit_store::FIT_TOML_FILE).map_err(e)?.write_all(source.bytes()).map_err(e)?;
                crate::fit_store::write_method(&mut run, &method)
            })();
            if let Err(e) = ran {
                let _ = run.fail("writing the run's settings", &e);
                failures.push(format!("{method_name} seed {seed}: {e}"));
                continue;
            }
            let work = work_root.path().join(format!("run-{}", run.id().key().short12()));
            std::fs::create_dir_all(&work).unwrap_or_else(|e| out.fail(format!("error: cannot create {}: {e}", work.display())));
            if let Some(b) = &base {
                if let Err(e) = b.carry_into(&mut run, &work) {
                    let _ = run.fail("carrying the base's chains", &e);
                    failures.push(format!("{method_name} seed {seed}: {e}"));
                    continue;
                }
            }
            let resume_guard = identity.chain.key().to_hex();
            let finished = compute_run(&mut out, RunCompute {
                run,
                work: &work,
                problem: &cell.problem,
                method: &method,
                model: &model,
                seed,
                starts: &resolved_starts,
                base: base.as_ref(),
                resume_guard: &resume_guard,
                header: &run_header,
                acceptance_kernel,
                data_digests: &data_digests,
                input_hash: &stored_problem.id().key().to_hex(),
                forcing: &forcing,
                record: RunRecord {
                    starts_rule: &starts_rule,
                    starts_source: &starts_source_record,
                    config: &config,
                    config_text: &config_text,
                    model: &model,
                    roles: &fit_roles,
                    sweep_point: &cell.sweep_point,
                    partition: &partition,
                },
                store_root: store.root(),
            });
            if let Err(e) = finished {
                failures.push(format!("{method_name} seed {seed}: {e}"));
            }
        }
    }
    if cells.len() > 1 || config.problem.is_synthetic_fit() {
        out.say("note: grid summary / coverage are derived views — owed by `fit recovery` (gh#150 / gh#154)");
    }
    failures
}

/// `name_value__name_value`, the way a sweep point is printed and labelled.
fn point_slug(point: &[(String, f64)]) -> String {
    point.iter().map(|(k, v)| format!("{k}_{v:.3}")).collect::<Vec<_>>().join("__")
}

/// `p` relative to `root` when it is under it.
fn relative(p: &Path, root: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).display().to_string()
}

/// The base a `--resume` continues: the stored run, the dependency this run's
/// identity holds on its carried chain files, and a scratch copy of its files
/// for the records a resume extends (its warm-up record, its lineage, its
/// segments' wall times).
struct ResumeBase {
    stored: Stored<FitRun>,
    dep: store::Dep<FitRun>,
    files: crate::fit_store::RunFiles,
    carried: Vec<String>,
}

impl ResumeBase {
    fn resolve(
        store: &store::Store,
        handle: &str,
        chain: &ChainIdentity,
        length: u64,
        algorithm: &Algorithm,
    ) -> Result<ResumeBase, String> {
        let stored = crate::fit_store::resolve_run(store, handle)?;
        let base = stored.identity();
        if base.chain.content_hash() != chain.content_hash() {
            let why = if base.chain.problem.id().key() != chain.problem.id().key() {
                "another problem, or the same problem fitted by another build of camdl"
            } else if base.chain.method.content_hash() != chain.method.content_hash() {
                "other method settings"
            } else if base.chain.seed.seed != chain.seed.seed {
                "another seed"
            } else {
                "other starts, perturbation settings or compile"
            };
            return Err(format!(
                "{handle} is not this run continued: it was fitted with {why}. A resume keeps \
                 everything but the length; run without --resume, or resume the matching run."
            ));
        }
        if !matches!(algorithm, Algorithm::PGAS { .. } | Algorithm::PMMH { .. }) {
            return Err(format!("only PGAS and PMMH runs can be resumed; this run's method is '{}'",
                algorithm.method_name()));
        }
        if length <= base.length {
            let unit = if matches!(algorithm, Algorithm::PGAS { .. }) { "sweeps" } else { "steps" };
            return Err(format!(
                "the base {handle} already holds {} {unit}, and this fit's target is {length}. A resume \
                 extends its base to a longer target; the base is itself the completed fit at its own \
                 length, and a shorter one is a fresh run without --resume.",
                base.length
            ));
        }
        let carried: Vec<String> = stored
            .result_files()
            .into_iter()
            .filter(|name| {
                let mut parts = name.splitn(2, '/');
                let (dir, file) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                score::convergence::ChainIndex::parse_dir_name(dir).is_some() && CARRIED.contains(&file)
            })
            .map(str::to_string)
            .collect();
        if carried.is_empty() {
            return Err(format!("{handle} holds no chain state to resume from"));
        }
        let refs: Vec<&str> = carried.iter().map(String::as_str).collect();
        let dep = stored.dep_files(&refs).map_err(|e| e.to_string())?;
        let files = crate::fit_store::RunFiles::of(&stored)?;
        Ok(ResumeBase { stored, dep, files, carried })
    }

    /// Write the base's carried chain files into the working directory,
    /// read through the run's dependency on them.
    fn carry_into(&self, run: &mut store::Run<FitRun>, work: &Path) -> Result<(), String> {
        for name in &self.carried {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut run.open_in(&self.dep, name).map_err(|e| e.to_string())?, &mut bytes)
                .map_err(|e| format!("reading the base's {name}: {e}"))?;
            let path = work.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        Ok(())
    }
}

/// What a run keeps of how it was asked for, besides its identity.
struct RunRecord<'a> {
    starts_rule: &'a crate::fit::starts::ChainStarts,
    starts_source: &'a serde_json::Value,
    config: &'a config_v2::FitConfig,
    config_text: &'a str,
    model: &'a ir::Model,
    roles: &'a Result<crate::roles::Partition, String>,
    sweep_point: &'a [(String, f64)],
    partition: &'a crate::run_meta::TrainingPartitionRecord,
}

/// One claimed run, and what computing it needs.
struct RunCompute<'a> {
    run: store::Run<FitRun>,
    work: &'a Path,
    problem: &'a config_v2::Problem,
    method: &'a config_v2::Method,
    model: &'a ir::Model,
    seed: u64,
    starts: &'a crate::fit::chain_starts::ResolvedStarts,
    base: Option<&'a ResumeBase>,
    resume_guard: &'a str,
    header: &'a crate::fit::report::RunHeader,
    acceptance_kernel: Option<sim::inference::diagnostic::AcceptanceKernel>,
    data_digests: &'a crate::fit::stream_bindings::BoundDataDigests,
    input_hash: &'a str,
    forcing: &'a crate::forcing_extrapolation::ForcingPolicy,
    record: RunRecord<'a>,
    store_root: &'a Path,
}

/// Compute one claimed run: the method in its working directory, its files
/// written into the run, and the run finished. A method that stops with an
/// error ends the command through `out`, which fails the run first.
fn compute_run(out: &mut Out, rc: RunCompute<'_>) -> Result<(), String> {
    let RunCompute {
        run, work, problem, method, model, seed, starts, base, resume_guard, header, acceptance_kernel,
        data_digests, input_hash, forcing, record, store_root,
    } = rc;
    let method_name = method.algorithm.method_name();
    let run_id = run.id().key().to_hex();
    let held: Arc<Mutex<Option<store::Run<FitRun>>>> = Arc::new(Mutex::new(Some(run)));
    // A run that fails keeps what its method wrote: the files, in the run's
    // folder, and the rest of its event log, before the record is marked
    // failed.
    let mirrored: Arc<Mutex<Mirrored>> = Arc::default();
    {
        let held = Arc::clone(&held);
        let mirrored = Arc::clone(&mirrored);
        let work_dir = work.to_path_buf();
        out.hold_failure(
            Box::new(move |operation: &str, reason: &str| {
                match held.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    Some(run) => {
                        mirror_once(&run, &work_dir, &mut mirrored.lock().unwrap_or_else(|e| e.into_inner()));
                        let _ = ingest(&run, &work_dir);
                        let failed = run.fail(operation, reason).map_err(|e| e.to_string());
                        // The process ends after this, so the command's
                        // working directory (this run's parent) is removed
                        // here: the run's files are now in the store.
                        if let Some(root) = work_dir.parent() {
                            let _ = std::fs::remove_dir_all(root);
                        }
                        failed
                    }
                    None => Ok(()),
                }
            }),
            forcing.clone(),
            method_name,
        );
    }
    let _refusal_deferred = forcing.defer_while_claimed();

    // The declared `start` values a spread rule discards (gh#506, gh#881).
    let unused_starts: Vec<String> = if record.starts_rule.ignores_base_point(method.algorithm.chains()) {
        problem.estimate.iter().filter(|(_, s)| s.start.is_some()).map(|(n, _)| n.clone()).collect()
    } else {
        Vec::new()
    };
    let mut attempt = crate::fit::execution::Execution::begin();
    attempt.write(work).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let method_t0 = std::time::Instant::now();
    let heartbeat = Arc::new(crate::fit::method_heartbeat(work, &method.algorithm));
    // The run's header names its own folder within its store, so its report
    // and its record agree, and neither depends on where the store is.
    let header = &{
        let mut h = header.clone();
        if let Some(dir) = held.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|r| r.folder()) {
            h.output = relative(dir, store_root);
        }
        h
    };
    let collector = out.begin_method(header.clone(), method_name, acceptance_kernel, heartbeat.clone(), work)
        .unwrap_or_else(|e| {
            heartbeat.failed(format!("error: {e}").as_str());
            out.fail(format!("error: {e}"))
        });
    let mirror = Mirror::start(Arc::clone(&held), work.to_path_buf(), Arc::clone(&mirrored));

    if !matches!(method.algorithm, Algorithm::PMMH { .. }) {
        crate::fit::methods::emit_status_banner(out, method.algorithm.method_kind(), method.algorithm.backend(), None);
        if let Some(note) = crate::fit::methods::status_note(method.algorithm.method_kind(), method.algorithm.backend()) {
            out.check(crate::fit::report::Check::MethodCaveat {
                method: method_name.to_string(), note: note.to_string(), series_len: None,
            });
        }
    }
    if !unused_starts.is_empty() {
        let note = crate::fit::report::unused_starts_note(record.starts_rule.tag(), &unused_starts,
            crate::fit::still_reads_declared_starts(&method.algorithm));
        out.say(format_args!("  {} {}", crate::out::paint::note_tag(), note));
        out.check(crate::fit::report::Check::Starts {
            mode: record.starts_rule.tag().to_string(), params: unused_starts.clone(), note,
        });
    }

    let ran = crate::fit::run_method(out, crate::fit::MethodContext {
        problem,
        method,
        model,
        method_dir: work,
        seed,
        starts,
        resume_base: base.map(|b| b.files.path()),
        resume_guard,
        heartbeat: &heartbeat,
        collector: &collector,
        data_hashes: crate::fit::provenance_data_hashes(data_digests),
        input_hash,
        forcing,
    });
    if let Err(e) = out.events_result() {
        let _ = out.close_events();
        crate::fit::abandon_method(out, format!("error: {e}"));
    }
    out.method_done();

    let method_elapsed = method_t0.elapsed();
    // What the run was asked to do, and what only the run can tell: its best
    // log-likelihood and chain, and for a resume, its lineage.
    let mut inputs = serde_json::json!({
        "run_id": run_id,
        "problem_id": input_hash,
        "method": method_name,
        "backend": method.algorithm.backend().as_str(),
        "seed": seed,
        "n_chains": method.algorithm.chains(),
        "algorithm": crate::fit::algorithm_record(&method.algorithm),
        "binomial": match &method.algorithm {
            Algorithm::IF2 { .. } | Algorithm::PGAS { .. } | Algorithm::PMMH { .. } | Algorithm::PFilter { .. } =>
                serde_json::to_value(method.algorithm.binomial()).unwrap_or(serde_json::Value::Null),
            _ => serde_json::Value::Null,
        },
        "starts": record.starts_rule.spelled(),
        "chain_starts_kind": record.starts_rule.kind().as_str(),
        "starts_source": record.starts_source,
        "header": header,
        "unused_starts": { "rule": record.starts_rule.tag(), "params": unused_starts },
        "best_loglik": ran.best_loglik,
        "best_chain": ran.best_chain,
    });
    // The forcing override, and the forcings it held, as the record of the
    // command that ran under it.
    forcing.record_in(&mut inputs);
    if let Some(b) = base {
        let index_column = if matches!(method.algorithm, Algorithm::PGAS { .. }) { "sweep" } else { "step" };
        let parent_id = b.stored.id().key().to_hex();
        let parent_inputs = crate::fit_store::read_run_inputs(b.files.path())
            .unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: --resume lineage: {e}")));
        let lineage = crate::fit::resumed_chain::ResumeLineage::extend(
            &parent_id, &parent_inputs, index_column, &run_id, ran.resume_boundary.clone(),
        ).unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: --resume lineage: {e}")));
        attempt.earlier_segments = crate::fit::execution::Execution::segments_resuming(b.files.path(), &parent_id);
        inputs[crate::fit::resumed_chain::RESUME_KEY] = serde_json::to_value(&lineage)
            .unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: --resume lineage: {e}")));
    }
    attempt.finish(method_elapsed.as_secs_f64());
    if let Some(sampling) = out.sampling() {
        attempt.record(sampling);
    }
    attempt.write(work).unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: {e}")));
    let write_json = |name: &str, v: &serde_json::Value| -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?;
        std::fs::write(work.join(name), bytes).map_err(|e| format!("cannot write {name}: {e}"))
    };
    write_json(crate::fit_store::RUN_INPUTS_FILE, &inputs)
        .unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: {e}")));
    let sidecar = crate::fit::run_sidecar(&crate::fit::SidecarInputs {
        config: record.config,
        config_text: record.config_text,
        model: record.model,
        problem,
        data: data_digests,
        roles: record.roles,
        sweep_point: record.sweep_point,
        seed,
        partition: record.partition.clone(),
    }, out)
    .unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: {e}")));
    write_json(crate::fit_store::FIT_META_FILE, &sidecar)
        .unwrap_or_else(|e| crate::fit::abandon_method(out, format!("error: {e}")));
    // The command's own policy: a forcing refusal kept at any point of the
    // run fails it here.
    if let Some(message) = forcing.refused() {
        crate::fit::abandon_method(out, message);
    }

    // The method is done: its report is the event log's last line, so it is
    // written, and passed on, before the run is sealed. Then the run is taken
    // back and every file the method made is written into it.
    let record = out.end_method().expect("the method was begun before its dispatch");
    let folder = held.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
        .and_then(|r| r.folder().map(Path::to_path_buf))
        .expect("a fit run is written in its folder");
    match crate::fit::completed_outcome(method_name, work, &folder, record, ran.warmup,
        method_elapsed.as_secs_f64(), model)
    {
        Ok(outcome) => out.method_report(&outcome),
        Err(e) => out.say(format_args!("warning: the run's report could not be built: {e}")),
    }
    mirror.stop();
    out.release_failure();
    let run = held.lock().unwrap_or_else(|e| e.into_inner()).take()
        .expect("the run is held until the method ends");
    let stored = match ingest(&run, work).and_then(|()| store_start_file(&run, starts)) {
        Ok(()) => run.finish().map_err(|e| e.to_string()),
        Err(e) => {
            let _ = run.fail("storing the run's files", &e);
            Err(e)
        }
    };
    let stored = match stored {
        Ok(s) => s,
        Err(e) => {
            let _ = out.close_events();
            return Err(e);
        }
    };
    out.method_committed();
    out.milestone("stored", format!("{} \u{b7} {} \u{b7} {:.1}s", method_name,
        relative(stored.folder(), store_root), method_elapsed.as_secs_f64()));
    if let Some(hint) = ran.stored_hint {
        out.hint(hint);
    }
    if let Err(e) = out.close_events() {
        out.say(format_args!("warning: {e}"));
    }
    Ok(())
}

/// The files of a method's working directory that are not the run's result:
/// what the store writes itself (the progress snapshot and the event log,
/// passed on while the method ran) and the records of this execution, kept
/// as provenance.
fn role_of(name: &str) -> FileRole {
    match name {
        "progress.json" | "events.jsonl" => FileRole::Skip,
        "execution.json" | "opt_trace.tsv" => FileRole::Provenance,
        n if n == crate::fit_store::RUN_INPUTS_FILE || n == crate::fit_store::FIT_META_FILE => FileRole::Provenance,
        _ => FileRole::Result,
    }
}

enum FileRole {
    Result,
    Provenance,
    Skip,
}

/// A starts file named directly (`from_params`, or a `from_mle` or
/// `from_posterior` file outside the store) is part of the run's identity by
/// its bytes, so the run stores them, as `inputs/starts.<ext>`.
fn store_start_file(run: &store::Run<FitRun>, starts: &crate::fit::chain_starts::ResolvedStarts) -> Result<(), String> {
    let Some(StartInput::File { file }) = starts.source.as_ref().map(|s| &s.input) else { return Ok(()) };
    let name = format!("inputs/starts.{}", file.ext);
    let handle = run.result_file(&name).map_err(|e| format!("{name}: {e}"))?;
    handle.write_all(&file.bytes()).map_err(|e| format!("{name}: {e}"))
}

/// Write every file under `work` into `run`, by its role.
fn ingest(run: &store::Run<FitRun>, work: &Path) -> Result<(), String> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![work.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    for p in files {
        let name = p.strip_prefix(work).expect("under the working directory").to_string_lossy().into_owned();
        let handle = match role_of(&name) {
            FileRole::Skip => continue,
            FileRole::Provenance => run.provenance_file(&name),
            FileRole::Result => run.result_file(&name),
        }
        .map_err(|e| format!("{name}: {e}"))?;
        let bytes = std::fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        handle.write_all(&bytes).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

/// Passes a running method's progress snapshot and each new line of its event
/// log, both written in its working directory, to the run's own channels,
/// every few seconds and once more when stopped.
struct Mirror {
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// How far a run's progress and event log have been passed on, shared by the
/// mirror's thread and the run's failure, which passes on the rest.
#[derive(Default)]
struct Mirrored {
    sent: usize,
    last_progress: Option<Vec<u8>>,
}

/// Passes on to `run` what `work`'s progress snapshot and event log hold
/// beyond `state`.
fn mirror_once(run: &store::Run<FitRun>, work: &Path, state: &mut Mirrored) {
    if let Ok(bytes) = std::fs::read(work.join("progress.json")) {
        if state.last_progress.as_deref() != Some(&bytes[..]) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                let _ = run.progress(&v);
            }
            state.last_progress = Some(bytes);
        }
    }
    if let Ok(text) = std::fs::read_to_string(work.join("events.jsonl")) {
        // Only whole lines: the last may be half written.
        let whole = text.rfind('\n').map_or(0, |i| i + 1);
        if whole > state.sent {
            for line in text[state.sent..whole].lines() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    let _ = run.event(&v);
                }
            }
            state.sent = whole;
        }
    }
}

impl Mirror {
    fn start(run: Arc<Mutex<Option<store::Run<FitRun>>>>, work: PathBuf, state: Arc<Mutex<Mirrored>>) -> Mirror {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::spawn(move || loop {
            let stopping = flag.load(std::sync::atomic::Ordering::Acquire);
            if let Some(run) = run.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                mirror_once(run, &work, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
            }
            if stopping {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        });
        Mirror { stop, thread: Some(thread) }
    }

    fn stop(mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests;
