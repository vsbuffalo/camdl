//! `camdl pfilter` — bootstrap particle filter for log-likelihood estimation.
//!
//! Usage:
//!   camdl pfilter MODEL --params P.toml --data cases.tsv \
//!       --particles 5000 --dt 1.0 --seed 1
//!
//! Output: log-likelihood estimate to stdout.
//! With --trace: per-observation TSV (time, ll_increment, ESS).

use rayon::prelude::*;
use store::{Artifact, ContentArtifact};

use crate::pfilter_store::Loglik;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        bootstrap_filter,
        particle_filter::Observation,
        traits::SMCConfig,
        ChainBinomialProcess,
        BoundObs,
        MultiStreamObsModel,
    },
};

/// What to pass when `pfilter` has no observation data bound. Named once so the
/// anchored refusal and the ordinary missing-data refusal cannot drift apart.
const PFILTER_SUPPLY: &str = "pass `--data PATH` (single-stream), \
    `--data NAME=PATH` (repeatable, multi-stream), or `--fit FOO.toml` with a \
    [data.observations] section.";

/// Refuse `--sweep-file`, or `--replicates N` (N > 1), beside any output that
/// describes one filter run. A design writes one log-likelihood row per point
/// and the replicate path only the per-replicate log-likelihoods; every flag
/// named here is written on the single-run path alone.
pub(crate) fn check_per_run_outputs(a: &crate::args::PfilterArgs) -> Result<(), String> {
    if a.replicates <= 1 && a.sweep_file.is_none() {
        return Ok(());
    }
    let per_run: Vec<&str> = [
        ("--trace", a.trace.is_some()),
        ("--pf-health", a.pf_health.is_some()),
        ("--save-final-state", a.save_final_state.is_some()),
        ("--save-paths", a.save_paths.is_some()),
        ("--save-filtering", a.save_filtering.is_some()),
        ("--save-filtered-streams", a.save_filtered_streams.is_some()),
        ("--save-prequential", a.save_prequential.is_some()),
    ]
    .into_iter()
    .filter_map(|(flag, set)| set.then_some(flag))
    .collect();
    if per_run.is_empty() {
        return Ok(());
    }
    if let Some(file) = &a.sweep_file {
        return Err(format!(
            "--sweep-file {} cannot be combined with {}: {} one filter run, and \
             `--sweep-file` writes one log-likelihood row per design point without \
             writing any single run's outputs. For the per-run file of one point, \
             run `pfilter` without `--sweep-file`, passing the point's values with \
             `--param`.",
            file.display(),
            per_run.join(", "),
            if per_run.len() == 1 { "that output describes" } else { "those outputs describe" },
        ));
    }
    Err(format!(
        "--replicates {} cannot be combined with {}: {} one filter run, and \
         `--replicates` reports the spread of the log-likelihood over {} \
         independent runs without writing any single run's outputs. Run \
         `--replicates 1` (with a separate `--seed` per run if you want \
         several) for the per-run file, and `--replicates` on its own for the \
         spread.",
        a.replicates,
        per_run.join(", "),
        if per_run.len() == 1 { "that output describes" } else { "those outputs describe" },
        a.replicates,
    ))
}

/// The status `pfilter` exits with when the particle filter itself failed at
/// the parameters it was given (its weights collapsed, the state went
/// negative): the inputs were read and the filter ran. Every other error exits
/// with status 1. `camdl compare` reads it to tell a posterior draw the filter
/// rejected from a failure of the fit's inputs (gh#1009).
pub const FILTER_FAILED: crate::out::ExitCode = crate::out::ExitCode(3);

fn filter_failed(message: String) -> crate::out::CommandError {
    crate::out::CommandError { code: FILTER_FAILED, message, hint: None }
}

/// The sentence for a filter whose log-likelihood is `-inf` because every
/// particle that was alive scored `-inf` at some observation, naming the
/// observation, the stream, the observed value and the guard that produced
/// `-inf` (`sim::inference::obs_attempt::NegInfCause`). `None` for a result
/// that is finite, or `-inf` for a reason the filter did not record.
fn collapse_message(result: &sim::inference::particle_filter::PFilterResult) -> Option<String> {
    if result.log_likelihood != f64::NEG_INFINITY {
        return None;
    }
    result.collapse.as_ref().map(|c| format!(
        "error: the log-likelihood is -inf at these parameters: {c}"))
}

/// The rows a `--save-final-state` run under the fit's `holdout_after` binds
/// (gh#1138): the cutoff it resolved to, the training extent (the latest end
/// among the kept rows, where the state is saved), how many leading rows of
/// each stream it kept, by stream name, and whether that dropped any row.
struct TrainingCut {
    at: f64,
    extent: f64,
    rows: Vec<(String, usize)>,
    dropped: bool,
}

impl TrainingCut {
    /// The training rows the stored leaf keys: `Some` only when the cut
    /// dropped a row from some stream, since only then does the stored
    /// log-likelihood differ from the same evaluation without the cut. A run
    /// bound to its training rows alone keeps the run_id it has without
    /// `--save-final-state`.
    fn identity(&self) -> Option<&[(String, usize)]> {
        self.dropped.then_some(self.rows.as_slice())
    }
}

/// Keep only the rows the fit trains on under `holdout_after` (resolved to
/// `at`; `raw` is the value as written): in each stream the prefix
/// [`crate::fit::training_partition::cutoff_prefix`] cuts, the one rule the
/// fit's training partition uses. Every stream must keep a row with a value,
/// as in `fit run`. Unlike the fit's partition, nothing need be held out: a
/// stream bound to its training rows alone is kept whole.
fn cut_to_training_rows(
    at: f64,
    raw: &str,
    streams: &mut [crate::fit::runner::ObsStream],
) -> Result<TrainingCut, String> {
    use crate::fit::training_partition::PartitionError;
    let refuse = |e| crate::fit::runner::partition_error(e, Some((raw, at)));
    let mut rows = Vec::with_capacity(streams.len());
    let mut dropped = false;
    for s in streams.iter_mut() {
        let n = crate::fit::training_partition::cutoff_prefix(&s.rows, at)
            .map_err(|position| refuse(PartitionError::NotAPrefix {
                stream: s.name.clone(), position,
            }))?;
        if !s.rows[..n].iter().any(|r| match r.cell {
            Some(sim::inference::ObsCell::Scalar(_)) => true,
            Some(sim::inference::ObsCell::HeldOut(_)) | None => false,
        }) {
            return Err(refuse(PartitionError::NoTrainingRows { stream: s.name.clone() }));
        }
        dropped |= n < s.rows.len();
        s.truncate_rows(n);
        rows.push((s.name.clone(), n));
    }
    rows.sort();
    let extent = streams.iter()
        .flat_map(|s| s.rows.iter().map(|r| r.measures.end()))
        .reduce(f64::max)
        .ok_or_else(|| refuse(PartitionError::NoRows))?;
    Ok(TrainingCut { at, extent, rows, dropped })
}

/// `camdl pfilter`: its inputs split by class, the configuration resolved
/// from the declared inputs ([`crate::pfilter_config::resolve`]), and the run
/// built from it, which does not see the arguments.
pub fn cmd_pfilter(inv: crate::args::PfilterInvocation) {
    // pfilter has no `--no-color`: its painters follow the environment and
    // whether each stream is a terminal.
    let out = crate::out::Out::new(false);
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();  // gh#audit-H5
    // The replicate path reports the spread of the log-likelihood estimate
    // over independent runs and writes nothing per run, so a per-run output
    // beside it is refused rather than silently not written (gh#686).
    if let Err(e) = check_per_run_outputs(&inv.args) {
        out.fail(format!("error: {e}"));
    }
    // A file an export writes after the run is stored must have a directory
    // to go to; one that has none is refused before the filter runs.
    {
        let a = &inv.args;
        let stem = a.save_prequential.as_deref().map(std::path::PathBuf::from);
        let file_exports: [(&str, Option<&std::path::Path>); 8] = [
            ("--output", a.output.as_deref()),
            ("--trace", a.trace.as_deref().map(std::path::Path::new)),
            ("--pf-health", a.pf_health.as_deref().map(std::path::Path::new)),
            ("--save-final-state", a.save_final_state.as_deref()),
            ("--save-paths", a.save_paths.as_deref()),
            ("--save-filtering", a.save_filtering.as_deref()),
            ("--save-filtered-streams", a.save_filtered_streams.as_deref()),
            ("--save-prequential", stem.as_deref()),
        ];
        for (flag, path) in file_exports {
            if let Some(path) = path {
                if let Err(e) = crate::util::refuse_missing_export_dir(flag, path) {
                    out.fail(format!("error: {e}"));
                }
            }
        }
    }
    // A fit file is read once, here, before the store is chosen: its `store`
    // key places the filter run in that fit's store (run spec §2.1). A stored
    // fit's handle is read in the store, below.
    let fit_file: Option<crate::fit_store::FitArg> = inv.args.fit.as_deref()
        .map(|p| p.to_string_lossy().into_owned())
        .filter(|text| crate::fit_store::is_config_file(text))
        .map(|text| crate::fit_store::FitArg::read(&text, None).unwrap_or_else(|e| {
            out.fail(format!("error: --fit {text}: {e}"));
        }));
    let file_store = fit_file.as_ref().and_then(|f| f.config.problem.store.as_ref()).map(std::path::PathBuf::from);
    let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
    let (declared, no_licm) = declared.into_parts();
    let crate::args::PfilterInvocationExecution { args: execution, no_ir_cache } = execution;
    let crate::args::PfilterExecution { model_overrides: (), scenario: (), inference, stream: () } = execution;
    let compile = crate::util::CompileSettings::resolve(no_licm, no_ir_cache);
    let filter_execution = crate::pfilter_config::execution(&inference);
    let (exports,) = exports.into_parts();
    let ((), (), _inference_exports, (), output) = exports.into_parts();
    let (request,) = request.into_parts();
    let ((), (), _inference_request, (), no_store, force, label, root) = request.into_parts();
    let request = crate::camdl_store::store_request_with_fit_file(label, force, root, file_store);
    let execution_record = crate::args::PfilterInvocationExecution {
        args: crate::args::PfilterExecution { model_overrides: (), scenario: (), inference, stream: () },
        no_ir_cache,
    };
    let (config, requests, _forcing_route) = crate::pfilter_config::resolve(declared, compile, &out, fit_file, &crate::camdl_store::root_of(&request));
    let store = StoreSide { request, no_store, given, execution: execution_record };
    run_pfilter(&out, config, requests, RunExports { output }, filter_execution, store);
}

/// The store-class inputs of one `pfilter`, and what its records hold.
struct StoreSide {
    request: store::StoreRequest,
    /// `--no-store`: compute in a temporary store, deleted at the end.
    no_store: bool,
    given: store::Given,
    execution: crate::args::PfilterInvocationExecution,
}

/// The exports a command asks for besides its by-products.
struct RunExports {
    /// `-o`: the log-likelihood table, rendered from the stored result.
    output: Option<std::path::PathBuf>,
}

/// The particle filter of one resolved `pfilter`, and its stored result.
#[expect(clippy::disallowed_methods, reason = "reads the model file `model` names (declared `#[input(identity)]` on `PfilterArgs`), kept as the stored compiled model's provenance")]
fn run_pfilter(
    out: &crate::out::Out,
    config: crate::pfilter_config::ResolvedPfilterConfig,
    requests: crate::pfilter_config::ByProductRequests,
    exports: RunExports,
    execution: crate::pfilter_config::FilterExecution,
    store_side: StoreSide,
) {
    let crate::pfilter_config::ResolvedPfilterConfig {
        model_path: ir_path,
        compile,
        fit,
        objective,
        eval,
        forcing,
        scenario: scenario_flags,
        params: param_flags,
        data: cli_data_specs,
        obs: obs_name,
        dt: dt_flag,
        particles: n_particles,
        replicates: n_replicates,
        sweep_file,
        seed,
        binomial,
        save_final_state,
    } = config;
    let crate::pfilter_config::ByProductRequests {
        trace: trace_path,
        pf_health,
        final_state: save_final_state_path,
        paths: save_paths,
        filtering: save_filtering,
        filtered_streams: save_filtered_streams,
        prequential: save_prequential,
        score_from,
        samples: save_samples,
    } = requests;
    let RunExports { output: output_path } = exports;
    let crate::pfilter_config::FilterExecution { parallel, time_format, max_substeps } = execution;
    let fit_problem: Option<&crate::fit::config_v2::Problem> = fit.as_ref().map(|f| &f.config.problem);


    // The interventions this run scores under. With `--fit` it is the fit's
    // own problem, so the fit config's `scenario` / `enable` / `disable` is
    // applied, as `fit run` applied it — combined with the flags by the rule
    // `profile --fit` uses: either side alone selects, and
    // two selections that differ are refused rather than one winning.
    //
    // A selection taken from the fit config contributes its intervention
    // lists only ([`crate::util::FitBase`]): its parameter effect is already
    // the fit's `[fixed]` value, and applying a named scenario's `set` at
    // tier 4 here would override `--params` — the silent-wrong-θ hazard
    // gh#194 refuses `--scenario` beside `--params` for. A `--scenario` flag
    // keeps its full meaning, and that gh#194 refusal.
    let flag_given = scenario_flags.any();
    let fit_base: Option<crate::util::FitBase> = fit_problem.map(|p| {
        crate::util::FitBase::of(p).unwrap_or_else(|e| {
            out.fail(format!("error: {e}"));
        })
    });
    if let Some(base) = &fit_base {
        let flag = crate::util::ScenarioChoice {
            scenario: scenario_flags.scenario.as_deref(),
            enable: &scenario_flags.enable,
            disable: &scenario_flags.disable,
        };
        let config = crate::util::ScenarioChoice {
            scenario: base.selection.scenario.as_deref(),
            enable: &base.selection.enable,
            disable: &base.selection.disable,
        };
        if let Err(e) = crate::util::combine_fit_scenario(flag, config) {
            out.fail(format!("error: {e}"));
        }
    }
    // Scored under the fit's selection exactly when the flags select nothing
    // and the fit config does; its lists are resolved once the model is
    // loaded. Otherwise the flags are the selection (a flag naming the fit's
    // own scenario is the one agreement `combine_fit_scenario` admits).
    let scored_fit_base: Option<crate::util::FitBase> =
        fit_base.filter(|b| !flag_given && !b.is_empty());
    let crate::pfilter_config::ScenarioFlags {
        scenario: scenario_name, enable: mut adhoc_enable, disable: mut adhoc_disable,
    } = scenario_flags;

    // A `.camdl` is compiled through the IR cache under the command's compile
    // settings, to the full IR (`--no-ir-cache` compiles every time); an
    // `.ir.json` is read as given.
    let (resolved_ir, ir_tmp) = crate::util::resolve_ir_path(&ir_path, true, None, compile)
        .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    let loaded = crate::util::load_ir(&resolved_ir);
    if let Some(tmp) = &ir_tmp {
        let _ = std::fs::remove_file(tmp);
    }
    let (mut model_in, model_json) = loaded.unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
    if let Some(base) = &scored_fit_base {
        let (enable, disable) = base.lists(&model_in).unwrap_or_else(|e| {
            out.fail(format!("error: --fit: {e}"));
        });
        out.say(format_args!("pfilter: scoring under the fit's own {} (from --fit)", base.describe()));
        adhoc_enable = enable;
        adhoc_disable = disable;
    }
    // The one dt rule (`crate::dt_rule`): the model's declared step; a
    // `--dt` or `--fit` `[config] dt` that differs is refused; 1.0 when
    // nothing declares one.
    let fit_written_dt: Option<f64> =
        fit.as_ref().and_then(|f| crate::fit::config_v2::written_config_dt(f.config_text()));
    let dt = crate::dt_rule::effective_dt_for_command(
        dt_flag, fit_written_dt, model_in.simulation.dt,
    ).unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });

    // ── Resolve --data flags (gh#90) ─────────────────────────────────
    //
    // Polymorphic `--data` mirrors the existing `--table NAME=FILE`
    // pattern. Two forms: `--data PATH` (single-stream) and
    // `--data NAME=PATH` (repeatable, multi-stream). The resolver
    // returns one (stream_name, path) pair per bound stream.
    //
    // fit-toml fallback: when no --data flags supplied AND --fit was,
    // pull the multi-stream binding from the toml's
    // `[data.observations]` block. CLI flags always win; we emit an
    // info line if both forms were supplied.
    //
    // This runs BEFORE parameter resolution and before the compile because the
    // observation anchors below fold over exactly these bindings (gh#616), and
    // `CompiledModel::new` refuses a model that still carries one. Binding is a
    // pure read of the args plus `model.observations`, neither of which
    // parameter resolution touches.
    //
    // `None` means the invocation bound nothing at all (no `--data`, no
    // `--fit`). That is reported LATER, at the point it always was — after the
    // compile — so the capability gates keep first say on a params-only
    // invocation. gh#191's is the one that matters: "this model has real
    // compartments the filter cannot advance" is a far better answer than
    // "--data is required", and `gh191_pfilter_real_compartments` pins it.
    let model_obs_names: Vec<String> = model_in.observations.iter()
        .map(|o| o.name.clone()).collect();
    let bound_streams: Option<Vec<(String, std::path::PathBuf)>> = if cli_data_specs.is_empty() {
        // No CLI --data. Try the fit-toml fallback.
        fit.as_ref().map(|f| {
            data_observations_of(&f.config.problem, &model_in, &f.handle)
                .unwrap_or_else(|e| {
                    out.fail(format!("error: --fit toml fallback for --data: {}", e));
                })
        })
    } else {
        if fit.is_some() {
            out.say("pfilter: --data on CLI overrides --fit toml [data.observations]");
        }
        Some(crate::util::resolve_data_specs(&cli_data_specs, &model_obs_names, obs_name.as_deref())
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); }))
    };

    // gh#616: pfilter binds observation data, so it RESOLVES a model's
    // observation anchors instead of letting `CompiledModel::new` refuse them —
    // `pfilter --fit` is the cheapest fixed-θ probe there is, and an anchored
    // model that cannot be probed cannot be diagnosed. The window is folded from
    // the bindings just resolved above, so it is the same window this run
    // scores.
    //
    // Everything downstream reads the substituted model: there is exactly one
    // `load_model` and one `CompiledModel::new` on this path, and both the
    // scenario-horizon guard below and the resolver read `model_in`. A second
    // fresh load of the compiled IR would re-introduce the unresolved marker
    // (gh#616 regression, commit 7af5c9fa).
    //
    // With nothing bound, an ANCHORED model is refused here rather than
    // deferred: it has no horizon at all, so no later diagnostic can be
    // computed from it. An unanchored model falls through untouched.
    // A fit config's `[simulate] from` moves the run's start the same way
    // `fit run` does, so `pfilter --fit` scores the model the fit fits.
    let start = fit.as_ref()
        .map(|f| crate::obs_anchor::config_start(&f.config.problem, &model_in))
        .transpose()
        .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })
        .flatten();
    match &bound_streams {
        Some(bound) => {
            if let Err(e) = crate::obs_anchor::resolve_from_bindings(
                out, &mut model_in, bound, dt, None, start.as_ref())
            {
                out.fail(format!("error: {e}"));
            }
        }
        None => {
            if let Some(start) = start.as_ref() {
                match crate::obs_anchor::apply_start(&mut model_in, start, None) {
                    Ok(r) => crate::obs_anchor::report(out, &[r], &model_in),
                    Err(e) => out.fail(format!("error: {e}")),
                }
            }
            if let Some(msg) = crate::obs_anchor::refuse_without_data(&model_in, PFILTER_SUPPLY) {
                out.fail(format!("error: {msg}"));
            }
        }
    }

    // gh#561: a filter's window is the OBSERVATION times, not the model horizon
    // — `SMCConfig` carries a `t_start` and no `t_end`, and the schedule is
    // built from `obs_model.obs_time(i)`. So a scenario's own `simulate { to }`
    // cannot move anything here, including the `--save-paths` grid. Refuse
    // rather than ignore it: silently discarding a declared horizon is the bug
    // gh#561 exists to remove, and the same guard sits on `camdl predict`.
    //
    // It runs AFTER the substitution above so it compares two RESOLVED numbers:
    // an anchored model and an anchored scenario that resolve to the same time
    // are a no-op and must pass, and two that differ must be named with the
    // times a reader can check against the data.
    if let Err(e) = crate::util::refuse_scenario_horizon(
        &model_in, scenario_name.as_deref(), "pfilter",
        "a particle filter scores at the observation times, so its window comes \
         from the data, not the model horizon",
    ) {
        out.fail(format!("error: {e}"));
    }

    // The design points of `--sweep-file`, read against the model as the run
    // scores it, and refused before any filter runs when a tier above 3.5
    // would override a design column at every point.
    let design: Option<Vec<(usize, crate::sweep::DesignPoint)>> = sweep_file.as_deref().map(|file| {
        let points = crate::sweep::read_sweep_file(file, &model_in)
            .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })
            .points;
        if let Err(e) = refuse_design_overrides(
            &model_in, file, &points, &param_flags.param, scenario_name.as_deref(),
        ) {
            out.fail(format!("error: {e}"));
        }
        points
    });

    // The scored parameters, resolved by the tiers of `docs/camdl-run-spec.md`
    // §1.3: the fit config's `[fixed]` (tier 2, with `--fit`), `--params`, a
    // design point (tier 3.5), the scenario and `--param`. pfilter scores one
    // parameter set and estimates nothing, so the `[estimate]` set is empty
    // and no `--fixed`-over-[estimate] notice can fire.
    use crate::params_resolver::{ParameterInputs, resolve_parameters, emit_notices};
    use indexmap::{IndexMap, IndexSet};

    let crate::pfilter_config::ParamFlags { param: fixed_cli, params: fixed_files, table } = param_flags;
    let external = fit.as_ref()
        .map(|f| f.config.problem.external_bindings())
        .unwrap_or_default()
        .overlay(crate::external_inputs::ExternalBindings::from_cli(table));
    // The `[fixed]` table as `fit run` reads it: a `from_scenario` import or a
    // `from_file` and the inline values, against the loaded model.
    let ftf: IndexMap<String, f64> = match fit_problem {
        Some(p) => {
            let mut fixed = p.fixed.clone();
            fixed.expand_from_scenario(&model_in, &p.estimate)
                .and_then(|()| fixed.resolve_with_model(&model_in))
                .unwrap_or_else(|e| { out.fail(format!("error: --fit [fixed]: {e}")); })
        }
        None => IndexMap::new(),
    };
    let fte: IndexSet<String> = IndexSet::new();

    // The compiled model the run's identity names: the model as compiled and
    // rewritten from data (its anchored times, its bound tables), before the
    // parameter values and the scenario selection are applied. Those two
    // are identity fields of their own, which `camdl_store::apply` sets, so
    // every point of a design shares this one compiled model.
    let mut keyed_model = model_in.clone();
    crate::external_inputs::bind(&mut keyed_model, &external)
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));

    // gh#audit-H13: --parallel / CAMDL_PARALLEL throttles the rayon thread
    // budget for the particle filter: a scoped pool, on which the
    // bootstrap_filter calls and the points of a design run. The global pool
    // is already initialised by the time cmd_pfilter reaches here, so
    // `build_global` cannot be used. Nested rayon work (particle_filter.rs
    // par_iter_mut) inherits the surrounding pool. A value of 0 means rayon's
    // default (all logical cores): the pool is left unset and the global pool
    // used.
    let pf_pool: Option<rayon::ThreadPool> = if parallel > 0 {
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

    // ── The store ───────────────────────────────────────────────────────────
    //
    // Opened by the first point to pass its gates, so an invocation refused
    // before any filter runs leaves no store; the compiled model and each
    // data table are stored once, for every point.
    let StoreSide { request, no_store, given, execution: execution_record } = store_side;
    let stem = crate::hashing::path_stem_slug(&ir_path).unwrap_or_else(|| "model".to_string());
    let prepared: std::sync::OnceLock<Prepared> = std::sync::OnceLock::new();
    let prepare = |bound_streams: &[(String, std::path::PathBuf)]| -> Prepared {
        let store = if no_store { crate::camdl_store::scratch() } else { crate::camdl_store::open(&request) }
            .unwrap_or_else(|e| out.fail(format!("error: the run store at {}: {e}", crate::camdl_store::root_of(&request).display())));
        if no_store {
            out.remove_on_error(store.root().to_path_buf());
        }
        let info = store::RunInfo::new(&given, &execution_record, &request)
            .unwrap_or_else(|e| out.fail(format!("error: {e}")));
        // Each bound stream's table, read once, by content. The model was
        // rewritten from these rows (its anchored times), so its record names
        // them as lineage.
        let mut tables = Vec::new();
        for (name, path) in bound_streams {
            let table = store::InputFile::<crate::camdl_store::Table>::new(path.clone())
                .content()
                .unwrap_or_else(|e| out.fail(format!("error: --data {name}: {e}")));
            info.lineage(store::Lineage::Input { field: format!("data.{name}"), digest: table.digest() });
            tables.push((name.clone(), path.clone(), table));
        }
        let source = std::fs::read(&ir_path)
            .unwrap_or_else(|e| out.fail(format!("error: cannot read {ir_path}: {e}")));
        let envelope = ir::envelope_from_str(&model_json)
            .unwrap_or_else(|e| out.fail(format!("error: IR load error from {ir_path}: {e}")))
            .with_model(keyed_model.clone());
        let parts = crate::camdl_store::CompiledParts::split(
            envelope, stem.clone(),
            crate::camdl_store::ModelSource { bytes: &source, path: std::path::Path::new(&ir_path) },
        ).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let crate::camdl_store::CompiledParts { model: model_part, quantities, labels: model_labels, provenance } = parts;
        store::release(quantities, "a filter run reports no quantities");
        let model = store
            .put::<crate::camdl_store::CompiledModel>(model_part, &model_labels, &provenance, &info)
            .unwrap_or_else(|e| out.fail(format!("error: storing the compiled model: {e}")))
            .dep(crate::camdl_store::CompiledModel::FILE)
            .unwrap_or_else(|e| out.fail(format!("error: storing the compiled model: {e}")));
        let mut data = std::collections::BTreeMap::new();
        for (name, path, table) in tables {
            let labels = crate::camdl_store::TableLabels { stream: name.clone(), path: path.display().to_string() };
            let dep = store
                .put::<crate::camdl_store::DataTable>(table, &labels, &store::Provenance::none(), &info)
                .unwrap_or_else(|e| out.fail(format!("error: storing the data table of `{name}`: {e}")))
                .dep(crate::camdl_store::DataTable::FILE)
                .unwrap_or_else(|e| out.fail(format!("error: storing the data table of `{name}`: {e}")));
            data.insert(name, dep);
        }
        Prepared { store, info, model, data }
    };

    // ── One parameter point ─────────────────────────────────────────────────
    //
    // Everything from the parameter values on: resolve them with `overrides`
    // at tier 3.5 (a design point; empty for the single call), compile the
    // model at them, bind the data, and run the filter under the point's
    // claim. The single call and every design point are this one closure, so
    // a design row is the single call at its values by construction.
    let score_point = |overrides: &[(String, f64)], role: PointRole| -> PointResult {
        // The lines every point of a call would print alike.
        let announce = role.announces();
        let resolved = resolve_parameters(ParameterInputs {
            model: &model_in,
            scenario: scenario_name.as_deref(),
            adhoc_enable: &adhoc_enable,
            adhoc_disable: &adhoc_disable,
            scenario_inline_name: None,
            scenario_inline_set: &[],
            scenario_inline_scale: &[],
            point_overrides: overrides,
            fixed_cli: &fixed_cli,
            fixed_files: &fixed_files,
            swept: &[],
            fit_toml_fixed: &ftf,
            fit_toml_estimate: &fte,
            external: &external,
        }).unwrap_or_else(|e| {
            out.fail(format!("error: {}", e.with_flags(crate::params_resolver::FlagStyle::Params)));
        });
        emit_notices(out, &[&resolved]);
        let model = resolved.model.clone();

        let compiled = CompiledModel::new(model.clone())
            .unwrap_or_else(|e| { out.fail(format!("compile error: {:?}", e)); })
            .with_eval_settings(eval);
        // gh#122: the particle filter is a stochastic (chain_binomial) producer; a
        // source that mixes a deterministic exit with another exit would over-draw
        // the source. Reject it here (the standalone `pfilter` command bypasses
        // `check_model_capabilities`).
        compiled
            .validate_deterministic_source_exits()
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        // gh#121: a multi-source stochastic transition (`A + B --> C`) is bounded by
        // only its first source on chain_binomial, driving the secondary source
        // negative. The standalone pfilter bypasses check_model_capabilities, so
        // gate it here too.
        compiled
            .validate_single_source_transitions()
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        // A transition consuming more than one unit of its source per firing has no
        // meaning in the chain-binomial kernel the filter steps with; refuse it here
        // for the same reason.
        compiled
            .validate_unit_source_consumption()
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        // gh#191: the particle filter is a chain_binomial INFERENCE producer — it
        // carries no real state and never advances a reservoir, so a real-coupled
        // (REAL_COMPARTMENTS) model would be filtered with its reservoir frozen at
        // its init value: silently mis-fit. `camdl fit`/`profile` route through
        // `check_model_capabilities`, but the standalone `pfilter` command bypasses
        // it — so gate here too, with the same shared inference gate.
        crate::fit::methods::check_model_capabilities(
            crate::run_meta::InferenceBackend::ChainBinomial, &compiled,
        ).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        let params = compiled.default_params.clone();

        // The conditional objective (`ic_free = true`) is admitted by the check
        // `fit run` applies to its `pfilter` method: it needs particles that
        // differ in x₀, which a deterministic `init { }` does not give.
        if let Some(f) = fit.as_ref() {
            let fit_name = f.name();
            if objective.skips_first_observation() {
                let init_law = if compiled.has_init_law {
                    crate::fit::methods::InitLaw::Declared
                } else {
                    crate::fit::methods::InitLaw::Absent
                };
                if let Err(msg) = crate::fit::methods::validate_ic_free(
                    crate::run_meta::FitAlgorithm::Pfilter, init_law,
                ) {
                    out.fail(format!("error: the --fit config sets ic_free = true. {msg}"));
                }
                if announce {
                    out.say(format_args!("pfilter: objective: conditional on the first observation \
                        (ic_free = true in {fit_name}); omit --fit and pass --data for the \
                        full likelihood"));
                }
            } else if announce {
                out.say(format_args!("pfilter: objective: full likelihood ({fit_name} does not set \
                    ic_free = true)"));
            }
        }

        // The missing-data refusal, reported here rather than where the binding was
        // resolved, so every gate above keeps first say (see the binding block).
        let bound_streams: &[(String, std::path::PathBuf)] = bound_streams.as_deref().unwrap_or_else(|| {
            out.fail(format!("error: pfilter needs observation data to score: {PFILTER_SUPPLY}"));
        });
        if bound_streams.is_empty() {
            out.fail("error: zero streams resolved from --data / --fit toml. \
                       pfilter requires at least one observation stream.");
        }

        // Load data — route the time column through the calendar-time boundary
        // translator (numeric or dated, per --time-format + the model origin).
        let time_opts = TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt,
            t_start: compiled.model.simulation.t_start,
            format: time_format,
        };

        // Resolve the bound observation streams (BY SOURCE) and load each one's
        // per-observation values + aux via the single shared seam that `fit run`
        // and `profile` also route through (`fit::runner::resolve_and_load_obs_streams`).
        // The seam filters `model.observations` by `source`, dispatches long-form vs
        // wide loading (holes + aux), resolves each projection, and runs the per-
        // stream origin / first-window guards — so a stratified family bound by its
        // ROOT (`[data.observations] cases = FILE`, source `cases`) fans out to all
        // its leaves exactly as `fit run` binds them. `bound_streams` (the CLI/toml
        // key-space bindings) is mapped to the by-source `effective` map at the
        // boundary; it is retained below only for the CAS data-hash + the unbound-
        // streams warning, whose key-space (leaf name / family root) must not change.
        let effective = crate::fit::runner::data_bindings_to_effective(&model, bound_streams)
            .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        let mut streams = crate::fit::runner::resolve_and_load_obs_streams(
            &model, &compiled, &effective, &time_opts,
        ).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
        let n_streams = streams.len();

        // gh#1138: a saved final state is a forecast origin. Under the fit's
        // `holdout_after`, one filtered through rows past the cutoff would already
        // condition on the data a forecast from it is scored against, so a
        // `--save-final-state` run binds only the training rows, cut by the rule
        // the fit's partition uses, and saves the state at their end. Without
        // `--save-final-state` the filter scores every bound row, held-out ones
        // included. The fit is a fit file here: a stored fit that held rows out
        // keeps no held-out values and is refused before this
        // (`pfilter_config::resolve`).
        //
        // The cutoff resolves as `fit run` resolved it. A bare time or date needs
        // no rows. An anchored one (`holdout_after = "last_obs - 4 weeks"`)
        // resolves over the fit config's own files: resolved over `--data` files
        // holding only the training rows, it would move back and cut the very
        // rows that trained. Those files are read only in that case.
        let mut training_cut: Option<TrainingCut> = None;
        if save_final_state {
            let holdout_after = fit_problem
                .and_then(|p| p.data.as_ref())
                .and_then(|d| d.holdout_after.as_ref());
            if let (Some(spec), Some(f)) = (holdout_after, fit.as_ref()) {
                let anchored = matches!(
                    crate::fit::runner::parse_time_spec(
                        "holdout_after", &spec.raw(), model.origin.as_deref(), &model.time_unit),
                    Ok(crate::fit::runner::TimeSpec::Anchored(_)));
                let cutoff_rows = |ss: &[crate::fit::runner::ObsStream]| -> std::collections::BTreeMap<
                    String, Vec<crate::fit::observed_rows::ObservedRow>,
                > {
                    ss.iter().map(|s| (s.name.clone(), s.rows.clone())).collect()
                };
                let rows = if !anchored || cli_data_specs.is_empty() {
                    cutoff_rows(&streams)
                } else {
                    let fit_streams = data_observations_of(&f.config.problem, &model, &f.handle)
                        .and_then(|b| crate::fit::runner::data_bindings_to_effective(&model, &b))
                        .and_then(|e| crate::fit::runner::resolve_and_load_obs_streams(
                            &model, &compiled, &e, &time_opts))
                        .unwrap_or_else(|e| {
                            out.fail(format!(
                                "error: --fit toml data, read to resolve holdout_after: {e}"));
                        });
                    cutoff_rows(&fit_streams)
                };
                let raw = spec.raw();
                let cut = crate::fit::runner::resolve_holdout_cutoff(spec, &rows, &model)
                    .and_then(|at| cut_to_training_rows(at, &raw, &mut streams))
                    .unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
                // `--score-from` asks for a held-out window, and a cut that dropped
                // rows leaves none: every row it could score trained. Scoring them
                // would present in-sample rows as the held-out score.
                if cut.dropped {
                    if let Some(raw_from) = score_from.as_ref() {
                        out.fail(format!(
                            "error: --score-from {raw_from} scores a held-out window, but under \
                             --save-final-state this run binds only the training rows of \
                             holdout_after = \"{raw}\" (t = {}), so every row it could score \
                             trained.\n  Fix: run --save-prequential --score-from without \
                             --save-final-state (the filter then scores the full series), and \
                             save the state in a separate run.", cut.at));
                    }
                }
                out.say(format_args!(
                    "pfilter: --save-final-state under holdout_after = \"{raw}\" (t = {}): \
                     state saved at t = {}, the end of the training rows (holdout_after); \
                     the log-likelihood shown covers training rows only, as does every \
                     other output of this run (--trace, --save-prequential, --save-paths, \
                     --save-filtering, --save-filtered-streams, --pf-health)",
                    cut.at, cut.extent));
                training_cut = Some(cut);
            }
        }

        // Holes (missing observations via `NA`) are correct for the filter
        // log-likelihood — a hole contributes no term but still resets the bin
        // (the authoritative per-stream cells carry `None`). But the prequential
        // and `--trace` outputs read the dense placeholder view, where a hole shows
        // as 0: prequential would score elpd/CRPS/PIT against a fictitious observed
        // 0, and the trace's `observed` column would report 0 at a missing week.
        // Rather than emit a silently-wrong diagnostic, reject the combination until
        // those paths thread holes through (follow-up). The plain filter loglik is
        // unaffected.
        // gh#636: holes are first-class in the diagnostic outputs — per-stream
        // prequential scores skip hole (stream, time) cells (they carry no term),
        // the joint scores cover the present streams only, and the trace's
        // observed column prints NA where nothing is present. Say so once, so a
        // reader of the outputs knows rows were skipped rather than zero-filled.
        let has_holes = streams.iter().any(|s| s.cells.iter().any(|c| match c {
            Some(sim::inference::ObsCell::Scalar(_)) => false,
            Some(sim::inference::ObsCell::HeldOut(_)) | None => true,
        }));
        if announce && has_holes && (save_prequential.is_some() || trace_path.is_some()) {
            out.say("pfilter: data has missing observations (NA) — prequential/trace \
                 scores skip the missing (stream, time) cells; a step with no \
                 present stream is omitted from the prequential and prints NA in \
                 the trace.");
        }

        if announce && n_streams > 1 {
            out.say(format_args!("  streams: {}", streams.iter()
                .map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")));
        }

        if announce { out.say(format_args!("pfilter: bound streams: {}", streams.iter()
            .map(|s| format!("{}({})", s.name, match &s.obs_model_ir.likelihood {
                ir::observation::Likelihood::NegBinomial(_)  => "neg_binomial",
                ir::observation::Likelihood::Normal(_)       => "normal",
                ir::observation::Likelihood::Poisson(_)      => "poisson",
                ir::observation::Likelihood::Binomial(_)     => "binomial",
                ir::observation::Likelihood::BetaBinomial(_) => "beta_binomial",
                ir::observation::Likelihood::Beta(_)         => "beta",
                ir::observation::Likelihood::Bernoulli(_)    => "bernoulli",
                ir::observation::Likelihood::ZeroInflatedNegBinomial(_) => "zero_inflated_neg_binomial",
            })).collect::<Vec<_>>().join(", "))); }

        // gh#90: emit unbound-streams warning if a multi-block model is only
        // partially covered. The bound set is the RESOLVED leaf names (a family root
        // fans out to its leaves), not the raw binding keys — else an indexed family
        // bound by its root would look entirely unbound and false-warn.
        {
            let all_names: Vec<String> = model.observations.iter()
                .map(|o| o.name.clone()).collect();
            let bound_names: Vec<String> = streams.iter()
                .map(|s| s.name.clone()).collect();
            if let Some(w) = crate::util::format_unbound_streams_warning(
                "pfilter", &all_names, &bound_names,
            ).filter(|_| announce) {
                out.say_raw(format_args!("{}", w));
            }
        }

        // Build process + observation model via traits. The `ObsStream -> StreamSpec`
        // mapping is the single shared builder (`stream_specs_from_obs_streams`):
        // authoritative per-grid-time cells (holes = `None`) + survey denominators
        // (`aux`), each stream fed its OWN schedule; `bind` re-merges to the union
        // and validates aux — identical to how `fit run` builds its obs model.
        let compiled = std::sync::Arc::new(compiled);
        let process = ChainBinomialProcess::new(compiled.clone());
        let stream_specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        // The observed window `--score-from first_obs/last_obs` anchors to (D40):
        // the earliest start and the latest end of a row with a value.
        let observed_window =
            crate::fit::training_partition::observed_window(streams.iter().flat_map(|s| &s.rows));
        let (bound, report) = BoundObs::bind(compiled.model.simulation.t_start, stream_specs)
            .unwrap_or_else(|report| {
            out.fail(format!("error: observation data invalid:\n{}", report.render()));
        });

        // gh#812: a non-fatal bind report was previously discarded, so the
        // Warn channel was write-only. Surface it — a zero denominator scores
        // as zero, and the user must be told which rows and why.
        if announce && !report.findings().is_empty() {
            out.say(format_args!("{}", report.render()));
        }
        let obs_model = MultiStreamObsModel::new(bound, compiled.clone())
            .unwrap_or_else(|e| {
                out.fail(format!("error: observation model construction failed: {:?}", e));
            });

        // Canonical observations: the UNION axis the bound model scores along —
        // every stream's scoring boundaries plus every declared period's opening
        // boundary inside the run (multi-cadence, proposal 2026-06-10 §3.3;
        // periods, gh#833) — taken FROM the model, never rebuilt from the rows'
        // labels, which agree with it only under `closing_at`. Downstream
        // single-stream code paths (trace, prequential, save_filtering, save_paths,
        // save_final_state) consume `obs.time` (the union grid) and `obs.value` (a
        // never-scored placeholder 0.0 — the per-stream scored values live in each
        // stream's cells).
        let observations: Vec<Observation> = obs_model.obs_times().iter()
            .map(|&time| Observation { time, value: 0.0 })
            .collect();

        if announce {
            out.say(format_args!("pfilter: {} observation boundaries × {} streams, {} particles, dt={}, seed={}",
                observations.len(), n_streams, n_particles, dt, seed));
        }

        // Record ancestry when either --save-paths or --save-filtering is
        // active. Both flags consume the same per-step snapshot data; the
        // difference is only in what we write to disk at the end.
        let need_ancestry = save_paths.is_some() || save_filtering.is_some();
        // gh#520: the PredictionDiag block is only read by the single-run
        // `--trace` writer below (`result.predictions`, ~line 660). It used to
        // compute unconditionally whenever the obs model could project a state —
        // which meant `--replicates N` (which never reads `.predictions` at all,
        // only `.log_likelihood`) and a plain single run with no `--trace` both
        // paid for it and threw it away.
        let want_predictions = trace_path.is_some() && n_replicates <= 1;
        let smc_config = SMCConfig {
            n_particles,
            dt,
            t_start: compiled.model.simulation.t_start,
            skip_first_obs_from_loglik: objective.skips_first_observation(),
            record_ancestry: need_ancestry,
            record_prequential: save_prequential.is_some(),
            record_predictions: want_predictions,
            // Single-run only: `--replicates` refuses `--save-filtered-streams`.
            record_stream_means: save_filtered_streams.is_some() && n_replicates <= 1,
            // gh#241: deterministic compute budget (the engine default unless the
            // user passes `--pf-max-substeps`). Reproducible across machines — there
            // is no wall-clock watchdog, so a content-addressed `pfilters/` leaf is
            // produced identically regardless of hardware.
            max_substeps,
            // gh#747: `--binomial`; the run's stored address records it.
            binomial,
        };
        log::debug!("pfilter: a budget of {} particle-substeps", smc_config.max_substeps);

        // --save-filtering caveat log. Fires unconditionally (not quietable)
        // because the failure mode — plotting filtering marginals as if
        // they were smoothing paths — is silent. See
        // docs/dev/proposals/2026-04-19-pf-latent-trajectories.md.
        if save_filtering.is_some() {
            out.say(format_args!("[info] --save-filtering emits filtering marginals \
                       p(x_t | y_{{1..t}}), not smoothing paths. Joining \
                       particles across time by index does NOT yield \
                       trajectory samples from the posterior. For coherent \
                       sample paths use --save-paths N PATH."));
        }

        // ── The prequential window ──────────────────────────────────────────────
        //
        // `--score-from TIME` windows the prequential trace: observations at
        // t <= TIME are assimilated but not scored. It is resolved here, before
        // the run, to the model time its by-product's identity holds.
        let score_from_t: Option<f64> = match (&save_prequential, &score_from) {
            (Some(_), Some(raw)) => {
                let last = observations.last().map(|o| o.time).unwrap_or(0.0);
                let (first_obs, last_obs) = observed_window.unwrap_or((0.0, last));
                let spec = crate::fit::runner::parse_time_spec(
                    "--score-from", raw, model.origin.as_deref(), &model.time_unit,
                ).unwrap_or_else(|e| {
                    out.fail(format!("error: {}", e));
                });
                let t = match spec {
                    crate::fit::runner::TimeSpec::Absolute(v) => v,
                    crate::fit::runner::TimeSpec::Anchored(at) => {
                        let anchor = match at.anchor {
                            crate::fit::runner::ObsAnchor::First => first_obs,
                            crate::fit::runner::ObsAnchor::Last => last_obs,
                        };
                        at.resolve(anchor)
                    }
                };
                if t >= last - 1e-9 {
                    out.fail(format!(
                        "error: --score-from {raw} resolves to t = {t}, at or \
                         after the last observation (t = {last}) — nothing \
                         would be scored."));
                }
                Some(t)
            }
            _ => None,
        };

        // ── The engine ──────────────────────────────────────────────────────────
        //
        // The filter itself, run only under a claim: `--replicates N` runs N
        // independent filters and keeps each log-likelihood; one run keeps the
        // whole result, which the by-products are written from. A failure the
        // run's identity determines (every particle's weight reaching zero, an
        // error the model raises at these parameters) is part of the result.
        // Only the compute budget, an execution input, refuses the run.
        let run_filter = || -> Result<Filtered, crate::out::CommandError> {
            if n_replicates > 1 {
                if announce {
                    out.say(format_args!("pfilter: {} replicates × {} particles{}",
                        n_replicates, n_particles,
                        if parallel > 0 { format!(" (parallel = {})", parallel) } else { String::new() }));
                }
                // The metric is the running-mean loglik: replicates are noisy
                // estimates of one loglik at fixed parameters. A design point has
                // no bar of its own: the design counts points.
                let bar = matches!(role, PointRole::Single).then(|| {
                    crate::out::progress::Reporter::new().task(n_replicates as u64, "pfilter", "reps")
                });
                let acc = std::sync::Mutex::new((0.0_f64, 0usize)); // (Σ loglik, count)
                let each: Vec<Result<(u64, f64, Option<String>), String>> = run_pooled(&pf_pool, || {
                    (0..n_replicates).into_par_iter().map(|rep| {
                        let rep_seed = replicate_seed(seed, rep);
                        let result = match bootstrap_filter(&process, &obs_model, &params, &smc_config, rep_seed) {
                            Ok(result) => result,
                            Err(e) if refuses_the_run(&e) => {
                                return Err(format!("pfilter replicate {} error: {:?}", rep + 1, e));
                            }
                            Err(e) => {
                                if let Some(bar) = bar.as_ref() { bar.inc(1); }
                                return Ok((rep_seed, f64::NEG_INFINITY, Some(format!(
                                    "pfilter replicate {} error: {}", rep + 1, failure_text(&e)))));
                            }
                        };
                        // The replicate is added to the running mean before its
                        // count is ticked, so the line printed at the last count
                        // names the mean of every finite replicate.
                        if let (true, Some(bar)) = (result.log_likelihood.is_finite(), bar.as_ref()) {
                            if let Ok(mut g) = acc.lock() {
                                g.0 += result.log_likelihood;
                                g.1 += 1;
                                bar.set(crate::out::progress::ll(g.0 / g.1 as f64));
                            }
                        }
                        if let Some(bar) = bar.as_ref() { bar.inc(1); }
                        Ok((rep_seed, result.log_likelihood, collapse_message(&result)))
                    }).collect()
                });
                if let Some(bar) = bar { bar.finish(); }
                let mut rows = Vec::with_capacity(n_replicates);
                let mut failures = Vec::with_capacity(n_replicates);
                for r in each {
                    let (s, ll, c) = r.map_err(filter_failed)?;
                    rows.push((s, ll));
                    failures.push(c);
                }
                Ok(Filtered::Replicates { rows, failures })
            } else {
                // The scoped pool throttles the particle-level parallelism inside
                // bootstrap_filter (par_iter_mut over the swarm).
                match run_pooled(&pf_pool, || bootstrap_filter(&process, &obs_model, &params, &smc_config, seed)) {
                    Ok(result) => Ok(Filtered::Single(Box::new(result))),
                    Err(e) if refuses_the_run(&e) => Err(filter_failed(format!(
                        "pfilter error: {}", crate::util::sim_error_text(&e)))),
                    Err(e) => Ok(Filtered::Failed(format!("pfilter error: {}", failure_text(&e)))),
                }
            }
        };

        // What one computed run stores, from the filter's outcome. The writers
        // are those of the files the flags export, so a stored by-product and
        // its export are the same bytes.
        let run_files = |filtered: &Filtered, model: &ir::Model| -> RunFiles {
            // Where each scored value came from, the notices the resolution
            // reported, and under the forcing override each forcing the run held
            // outside its range: what the run records besides its result.
            let mut record = serde_json::json!({
                "parameters_provenance": crate::run_meta::ParameterProvenance::of_all(&resolved.params),
                "notices": crate::params_resolver::notices(&[&resolved]),
            });
            forcing.record_in(&mut record);
            let parameters = serde_json::to_vec_pretty(&record).expect("the run's record encodes");
            let mut f = RunFiles {
                loglik: Loglik {
                    loglik: None,
                    n_replicates: n_replicates as u64,
                    n_particles: n_particles as u64,
                    loglik_sd: None,
                    n_failed: 0,
                    failure: None,
                },
                replicates: None,
                parameters,
                trace: None,
                pf_health: None,
                final_state: None,
                paths: None,
                prequential: None,
                filtering: None,
                filtered_streams: None,
            };
            let result = match filtered {
                Filtered::Failed(message) => {
                    f.loglik.n_failed = 1;
                    f.loglik.failure = Some(message.clone());
                    return f;
                }
                Filtered::Replicates { rows, failures } => {
                    let n = rows.len();
                    f.replicates = Some(replicates_bytes(rows));
                    // A replicate that failed has no log-likelihood, so neither
                    // has the mean: the count and the first failure are stored.
                    let n_failed = failures.iter().flatten().count();
                    if n_failed > 0 {
                        f.loglik.n_failed = n_failed as u64;
                        f.loglik.failure = failures.iter().flatten().next().cloned();
                    } else {
                        let mean = rows.iter().map(|r| r.1).sum::<f64>() / n as f64;
                        let var = rows.iter().map(|r| (r.1 - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                        f.loglik.loglik = Some(mean);
                        f.loglik.loglik_sd = Some(var.sqrt());
                    }
                    return f;
                }
                Filtered::Single(result) => result,
            };
            match collapse_message(result) {
                Some(message) => {
                    f.loglik.n_failed = 1;
                    f.loglik.failure = Some(message);
                }
                None => f.loglik.loglik = Some(result.log_likelihood),
            }
            if trace_path.is_some() {
                f.trace = Some(trace_bytes(result, &observations, &obs_model));
            }
            if pf_health.is_some() {
                f.pf_health = Some(pf_health_bytes(result, &observations, n_particles));
            }
            // An unweighted sample from p(x_T | y_{1:T}) at this θ, at the last
            // observation time on the union grid: the swarm was resampled after
            // scoring it. That time rides in the file's header, because
            // `simulate --init-state` has no other way to learn the forecast
            // origin.
            if save_final_state_path.is_some() {
                if let Some(states) = &result.final_states {
                    let origin_t = observations.last().map(|o| o.time).unwrap_or(model.simulation.t_start);
                    let columns = io::trajectories::TrajColumnSpec::from_model(model, &[]);
                    f.final_state = Some(io::final_states_bytes(states, &columns, origin_t)
                        .unwrap_or_else(|e| out.fail(format!("error writing final states: {}", e))));
                }
            }
            if let Some((n_paths, _)) = &save_paths {
                let trace = result.ancestry.as_ref().expect("record_ancestry is set when save_paths is");
                let paths = sim::inference::ancestor_trace::sample_paths(trace, *n_paths, seed);
                f.paths = Some(paths_bytes(&paths, model));
            }
            if save_prequential.is_some() {
                let recorded = result.prequential.as_ref().expect("record_prequential is set when save_prequential is");
                // Each stream is scored against its own observed values.
                let per_stream_obs = obs_model.per_stream_observed();
                let t0 = score_from_t.map_or(0, |t| crate::fit::runner::first_scored_index(&recorded.obs_times, t));
                // What each scored value was accumulated over, so `compare` can
                // refuse to difference two traces that scored different windows.
                let per_stream_cov = obs_model.per_stream_coverage();
                let mut trace = sim::inference::prequential::build_trace(
                    recorded, &per_stream_obs, &per_stream_cov, &result.ess_trace, t0, seed,
                    obs_model.first_observation_opens_after_run_start(smc_config.t_start),
                    score_from_t);
                // The forcings the filter held outside their ranges, with the
                // scores they affected.
                trace.warnings.extend(forcing.trace_warnings());
                // How this trace was computed: one pass at the parameters this
                // invocation was given.
                trace.derivation = Some(score::trace::Derivation::single_pass(
                    &trace, n_particles, dt, seed,
                    score::trace::ThetaSource::Supplied,
                    adhoc_enable.clone(), adhoc_disable.clone()));
                if !save_samples {
                    for step in &mut trace.steps {
                        for ss in &mut step.per_stream { ss.y_pred_samples.clear(); }
                    }
                    trace.warnings.push(score::trace::PrequentialWarning::SamplesNotSaved);
                }
                let calendar = ir::caltime::CalendarAxis::from_model(model)
                    .unwrap_or_else(|e| out.fail(format!("error: {e}")));
                f.prequential = Some(crate::prequential_out::prequential_files(&trace, &calendar)
                    .unwrap_or_else(|e| out.fail(format!("error writing prequential: {}", e))));
            }
            if save_filtering.is_some() {
                let trace = result.ancestry.as_ref().expect("record_ancestry is set when save_filtering is");
                f.filtering = Some(filtering_bytes(trace, model));
            }
            if save_filtered_streams.is_some() {
                let steps = result.stream_means.as_deref().expect("record_stream_means is set when save_filtered_streams is");
                f.filtered_streams = Some(filtered_streams_bytes(steps, &observations, &obs_model.stream_names()));
            }
            f
        };

        // ── The run ─────────────────────────────────────────────────────────────
        let Prepared { store, info, model: model_dep, data } = prepared.get_or_init(|| prepare(bound_streams));
        use crate::pfilter_store::*;
        let finite = |what: &str, x: f64| store::Finite::new(x).unwrap_or_else(|e| out.fail(format!("error: {what}: {e}")));
        let identity = PfilterIdentity {
            model: model_dep.clone(),
            config: PfilterConfig {
                particles: n_particles as u64,
                replicates: n_replicates as u64,
                dt: finite("dt", dt),
                data: data.clone(),
                objective: objective.into(),
                allow_forcing_extrapolation: forcing.allowed(),
                allow_degenerate_rates: eval.allow_degenerate_rates,
                training_rows: training_cut.as_ref().and_then(TrainingCut::identity)
                    .map(|rows| rows.iter().map(|(n, c)| (n.clone(), *c as u64)).collect()),
                binomial: smc_config.binomial.into(),
            },
            params: crate::camdl_store::ResolvedParams::new(resolved.params.iter().map(|p| (p.name.clone(), p.value)))
                .unwrap_or_else(|e| out.fail(format!("error: the scored point: {e}"))),
            scenario: crate::camdl_store::ActiveInterventions::of(&model),
            seed: crate::camdl_store::ProcessSeed { seed },
            fit: fit.as_ref().and_then(crate::fit_store::FitArg::problem_dep),
        };
        let point = match role {
            PointRole::Single => None,
            PointRole::Design { row, point_id, .. } => Some(DesignPointLabel { row, point_id, values: overrides.to_vec() }),
        };
        let labels = PfilterLabels { model: stem.clone(), seed, point };
        let filter = store::Identity::record(&identity);
        let asked_by_products = trace_path.is_some() || pf_health.is_some() || save_final_state_path.is_some()
            || save_paths.is_some() || save_prequential.is_some() || save_filtering.is_some()
            || save_filtered_streams.is_some();
        let mut wanted = store::Wanted::<Pfilter>::none();
        if trace_path.is_some() {
            wanted = wanted.with::<Trace>(TraceIdentity { filter: filter.clone() }, ());
        }
        if pf_health.is_some() {
            wanted = wanted.with::<PfHealth>(PfHealthIdentity { filter: filter.clone() }, ());
        }
        if save_final_state_path.is_some() {
            wanted = wanted.with::<FinalState>(FinalStateIdentity { filter: filter.clone() }, ());
        }
        if let Some((n_paths, _)) = &save_paths {
            wanted = wanted.with::<SavedPaths>(SavedPathsIdentity { filter: filter.clone(), paths: *n_paths as u64 }, ());
        }
        if save_prequential.is_some() {
            let score_from = score_from_t.map(|t| finite("--score-from", t));
            wanted = wanted.with::<Prequential>(PrequentialIdentity { filter: filter.clone(), score_from, samples: save_samples }, ());
        }
        if save_filtering.is_some() {
            wanted = wanted.with::<Filtering>(FilteringIdentity { filter: filter.clone() }, ());
        }
        if save_filtered_streams.is_some() {
            wanted = wanted.with::<FilteredStreams>(FilteredStreamsIdentity { filter }, ());
        }

        // A failure here is not a property of the point: an I/O or store error,
        // or another process holding the run. The point is not stored, and a
        // rerun computes it.
        let not_completed = |message: String| PointResult::Failed(crate::out::CommandError::new(message));
        let begun = match store.begin_with::<Pfilter>(&identity, &labels, &wanted, info, request.reuse()) {
            Ok(b) => b,
            Err(e) => return not_completed(format!("error: {e}")),
        };
        let result = match begun {
            store::Begin::Reused(stored) => PointResult::Stored { stored, kept: false, by_products_failed: false },
            store::Begin::Busy(owner) => not_completed(busy_message(owner)),
            store::Begin::Claimed(mut run) => 'claimed: {
                // A claim on a stored run exists only for by-products: the run
                // is computed again and compared, and keeps nothing of its own.
                let reproducing = run.reproducing();
                let filtered = match run_filter() {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = run.fail("filtering", &e.message);
                        break 'claimed PointResult::Failed(e);
                    }
                };
                // The by-products are written against the model the run read
                // from the store, the bytes its identity is keyed by.
                let run_model = match run.load(&identity.model) {
                    Ok(m) => m,
                    Err(e) => {
                        let message = format!("error: reading the stored compiled model: {e}");
                        let _ = run.fail("filtering", &message);
                        break 'claimed not_completed(message);
                    }
                };
                // The model the filter ran is the stored one with the identity's
                // parameters and interventions applied, byte for byte.
                let run_model = crate::camdl_store::apply(run_model.model(), &identity.params, &identity.scenario);
                if crate::camdl_store::model_part_digest(&run_model) != crate::camdl_store::model_part_digest(&model) {
                    let message = "internal: the model the filter ran is not the stored model with this run's \
                        parameters and interventions applied";
                    let _ = run.fail("filtering", message);
                    break 'claimed not_completed(format!("error: {message}"));
                }
                let files = run_files(&filtered, &run_model);
                // A filter that stopped with an error left no state to write the
                // by-products from: each one asked for is recorded as failed, and
                // the run itself is stored.
                let by_products_failed = match &filtered {
                    Filtered::Failed(message) if asked_by_products => {
                        run.fail_by_products("filtering", message);
                        true
                    }
                    _ => false,
                };
                if let Err(e) = write_run(&mut run, &files) {
                    let message = format!("error: writing the filter run: {e}");
                    let _ = run.fail("storing", &e.to_string());
                    break 'claimed not_completed(message);
                }
                match run.finish() {
                    // Something new is kept unless the run was reproduced and
                    // every by-product it was reproduced for failed.
                    Ok(stored) => PointResult::Stored { stored, kept: !(reproducing && by_products_failed), by_products_failed },
                    Err(e) => not_completed(format!("error: storing the filter run: {e}")),
                }
            }
        };
        if let (PointRole::Design { point_id, .. }, PointResult::Stored { stored, kept: computed, .. }) = (role, &result) {
            if !no_store {
                let folder = crate::run_paths::display_absolute(stored.folder());
                out.milestone(if *computed { "stored" } else { "cached" }, format_args!("point {point_id} {}", folder.display()));
            }
        }
        result
    };

    match design {
        Some(points) => {
            let results = score_design(out, &points, &pf_pool, &score_point);
            let Some(Prepared { store, info, .. }) = prepared.into_inner() else {
                unreachable!("a point that returns has opened the store")
            };
            report_design(out, &points, results, store, info, output_path.as_deref(), no_store);
        }
        None => {
            let result = score_point(&[], PointRole::Single);
            let Some(Prepared { store, info, .. }) = prepared.into_inner() else {
                unreachable!("a point that returns has opened the store")
            };
            let paths = ExportPaths {
                trace: trace_path.as_deref(),
                pf_health: pf_health.as_deref(),
                final_state: save_final_state_path.as_deref(),
                paths: save_paths.as_ref().map(|(n, p)| (*n, p.as_path())),
                prequential: save_prequential.as_deref(),
                filtering: save_filtering.as_deref(),
                filtered_streams: save_filtered_streams.as_deref(),
            };
            report_single(out, result, store, info, &paths, output_path.as_deref(), no_store, &model_in);
        }
    }
}
/// What every point of one `pfilter` stores through: the store, the
/// invocation's record, and the compiled model and data tables each point's
/// identity names. Made once, by the first point to reach the store.
struct Prepared {
    store: store::Store,
    info: store::RunInfo,
    model: store::Dep<crate::camdl_store::CompiledModel>,
    data: std::collections::BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>,
}

/// Which evaluation a point is.
#[derive(Clone, Copy)]
enum PointRole {
    /// The one evaluation of a call without a design: every line the run
    /// prints, the replicate progress bar, and the exports.
    Single,
    /// One point of a `--sweep-file` design: the `row`-th of the file,
    /// written under `point_id`. The `first` point prints the lines every
    /// point would print alike.
    Design { row: usize, point_id: usize, first: bool },
}

impl PointRole {
    /// Whether this evaluation prints the lines every point prints alike.
    fn announces(self) -> bool {
        match self {
            PointRole::Single => true,
            PointRole::Design { first, .. } => first,
        }
    }
}

/// What became of one point.
enum PointResult {
    /// A completed filter run: computed now, or found stored. `kept` says
    /// whether this call stored anything: the run, or a by-product of a run
    /// already stored. `by_products_failed` says the filter stopped with an
    /// error, so no by-product asked for could be written. Its
    /// `loglik.toml` says whether the filter scored the point or failed at
    /// it.
    Stored { stored: store::Stored<crate::pfilter_store::Pfilter>, kept: bool, by_products_failed: bool },
    /// The point did not complete, for a reason that is not a property of
    /// the point (the compute budget, an I/O or store error, another process
    /// holding the run). Nothing is stored for it, and a rerun computes it.
    Failed(crate::out::CommandError),
}

/// The paths the single call's by-products are exported to.
struct ExportPaths<'a> {
    trace: Option<&'a str>,
    pf_health: Option<&'a str>,
    final_state: Option<&'a std::path::Path>,
    paths: Option<(usize, &'a std::path::Path)>,
    prequential: Option<&'a str>,
    filtering: Option<&'a std::path::Path>,
    filtered_streams: Option<&'a std::path::Path>,
}

/// Seed stride between replicates: replicate `r` of a call with `--seed s`
/// runs on `s + r * SEED_STRIDE`. The golden-ratio multiplier decorrelates
/// the low bits of the ChaCha8 initial states, which a stride of one leaves
/// highly correlated.
const SEED_STRIDE: u64 = 0x9e3779b97f4a7c15;

/// The seed replicate `rep` of a call with `--seed seed` runs on.
fn replicate_seed(seed: u64, rep: usize) -> u64 {
    seed.wrapping_add((rep as u64).wrapping_mul(SEED_STRIDE))
}

/// Whether a filter error refuses the run instead of being its result. The
/// compute budget is the one such error: it depends on `--pf-max-substeps`,
/// an execution input, so a run it stops is not a function of its identity.
/// Every other error the filter returns is determined by the model, the
/// parameters, the data, the particles and the seed.
fn refuses_the_run(e: &sim::SimError) -> bool {
    matches!(e, sim::SimError::PFIterationBudget { .. })
}

/// A filter error as a stored result holds it: the text depends only on the
/// run's identity, so a degeneracy error leaves out the wall-clock time it
/// carries.
fn failure_text(e: &sim::SimError) -> String {
    match e {
        sim::SimError::PFDegenerate { kind, obs_window, .. } =>
            format!("particle filter degeneracy ({kind:?}) at obs_window {obs_window}"),
        other => crate::util::sim_error_text(other),
    }
}

/// The refusal of a run another process holds. An owner on another host may
/// have stopped without this host being able to tell, so the refusal names
/// `--force`, which takes such a claim over.
fn busy_message(owner: store::Owner) -> String {
    let elsewhere = owner.host != store::this_host();
    let host = owner.host.clone();
    let mut message = format!("error: {}", store::Error::InProgress(owner));
    if elsewhere {
        message.push_str(&format!("; it may still be running on {host}. If that process has stopped, \
            run again with --force"));
    }
    message
}

/// Run a closure on the scoped pool if one was built, else on the global pool
/// (`--parallel 0`). Generic over the closure so the borrowed captures
/// (process, observation model, all `Sync`) keep the closure `Send` for
/// `ThreadPool::install`.
fn run_pooled<R: Send>(pool: &Option<rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    let f = || {
        log::debug!("pfilter: filtering on {} threads", rayon::current_num_threads());
        f()
    };
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Refuse a `--sweep-file` design whose column a higher tier would override
/// at every point: a `--param` on a design column (tier 5), or a `--scenario`
/// that sets or scales one (tier 4). Either would score every row at the same
/// value while its `sweep:<param>` column said otherwise.
fn refuse_design_overrides(
    model: &ir::Model,
    file: &std::path::Path,
    points: &[(usize, crate::sweep::DesignPoint)],
    param_flags: &[(String, f64)],
    scenario: Option<&str>,
) -> Result<(), String> {
    let columns: std::collections::BTreeSet<&str> =
        points.iter().flat_map(|(_, pt)| pt.iter().map(|(k, _)| k.as_str())).collect();
    for (name, value) in param_flags {
        if columns.contains(name.as_str()) {
            return Err(format!(
                "parameter `{name}` is both a column of --sweep-file {file} and pinned by \
                 `--param {name}={value}`.\n  \
                 A `--param` outranks a design point, so the design would not vary `{name}`: \
                 every row would be scored at {value} while its `sweep:{name}` column said \
                 otherwise.\n  \
                 Fix: drop `--param {name}=…`, or remove the `{name}` column from the file.",
                file = file.display(),
            ));
        }
    }
    if let Some(scenario) = scenario {
        let varied = crate::params_resolver::coordinate_values(
            points.iter().flat_map(|(_, pt)| pt.iter().map(|(k, v)| (k.as_str(), *v))),
        );
        if let Some(msg) = crate::params_resolver::scenario_coordinate_collision(
            model,
            &crate::sim_job::ScenarioRef::Named(scenario.to_string()),
            crate::params_resolver::UserCoordinate::Sweep,
            &varied,
        )? {
            return Err(format!("--sweep-file {} and --scenario {scenario}: {msg}", file.display()));
        }
    }
    Ok(())
}

/// Score every point of a `--sweep-file` design, in file order. The first
/// point runs alone, prints the lines every point would print alike and opens
/// the store; the rest run concurrently on the call's pool.
fn score_design(
    out: &crate::out::Out,
    points: &[(usize, crate::sweep::DesignPoint)],
    pool: &Option<rayon::ThreadPool>,
    score_point: &(dyn Fn(&[(String, f64)], PointRole) -> PointResult + Sync),
) -> Vec<PointResult> {
    let n_points = points.len();
    out.say(format_args!("pfilter: {n_points} design points from --sweep-file"));
    let bar = crate::out::progress::Reporter::new().task(n_points as u64, "pfilter", "points");
    let score_at = |i: usize| -> PointResult {
        let (point_id, point) = &points[i];
        let result = score_point(point, PointRole::Design { row: i, point_id: *point_id, first: i == 0 });
        bar.inc(1);
        result
    };
    let first = score_at(0);
    let rest: Vec<PointResult> = run_pooled(pool, || (1..n_points).into_par_iter().map(score_at).collect());
    bar.finish();
    std::iter::once(first).chain(rest).collect()
}

/// End a design's invocation and write its table, rendered from the stored
/// points, to `output` or stdout. A point that did not complete is reported
/// and the table is not written, since a rendering of the invocation needs
/// every point; the run exits non-zero and a rerun computes only those
/// points. Otherwise the run exits with `FILTER_FAILED` only when the filter
/// failed at every point.
fn report_design(
    out: &crate::out::Out,
    points: &[(usize, crate::sweep::DesignPoint)],
    results: Vec<PointResult>,
    store: store::Store,
    info: store::RunInfo,
    output: Option<&std::path::Path>,
    no_store: bool,
) {
    use crate::pfilter_store::*;
    let n_points = points.len();
    let options = DesignTableOptions { run_id: !no_store };
    let mut not_completed = Vec::new();
    let mut every_failed: Vec<String> = Vec::new();
    for ((point_id, _), result) in points.iter().zip(results) {
        match result {
            PointResult::Failed(e) => not_completed.push((*point_id, e.message)),
            PointResult::Stored { stored, .. } => {
                let loglik = crate::camdl_store::read_stored(&stored, LOGLIK_FILE)
                    .and_then(|b| Loglik::from_bytes(&b))
                    .unwrap_or_else(|e| out.fail(format!("error: {e}")));
                if loglik.every_replicate_failed() {
                    every_failed.push(loglik.failure.unwrap_or_default());
                }
            }
        }
    }
    let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    if !not_completed.is_empty() {
        for (point_id, message) in &not_completed {
            out.say(format_args!("point {point_id}: {}", message.strip_prefix("error: ").unwrap_or(message)));
        }
        out.fail(format!(
            "error: {} of {n_points} design points did not complete, so the design table is not \
             written; a rerun computes only those points",
            not_completed.len()));
    }
    if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
        out.say(format_args!("pfilter: @{label} now names invocation {}, which held it from invocation {before}",
            invocation.seq()));
    }
    match output {
        Some(path) => {
            invocation
                .render::<DesignTable>(&store, &options, &store::ExportPath(path.to_path_buf()))
                .unwrap_or_else(|e| out.fail(format!("cannot write {}: {e}", path.display())));
            out.say(format_args!("design table ({n_points} points) written to {}", path.display()));
        }
        None => {
            let mut table = Vec::new();
            invocation
                .render_to::<DesignTable>(&store, &options, &mut table)
                .unwrap_or_else(|e| out.fail(format!("error: {e}")));
            out.data_raw(String::from_utf8_lossy(&table));
        }
    }
    if let Some(first) = every_failed.first() {
        let message = crate::pfilter_store::one_line(first);
        if every_failed.len() == n_points {
            out.error(filter_failed(format!(
                "error: every one of the {n_points} design points failed (every replicate \
                 scored -inf or stopped with an error); the first: {message}"))).exit();
        }
        out.say(format_args!(
            "warning: {} of {n_points} design points failed (every replicate scored -inf \
             or stopped with an error), so they have no log-likelihood; the first: {message}",
            every_failed.len()));
    }
}

/// End the single call's invocation, export the by-products asked for from
/// the store, and print the stored log-likelihood. A filter that failed at
/// every replicate exits with `FILTER_FAILED`, naming the stored failure.
#[allow(clippy::too_many_arguments)]
fn report_single(
    out: &crate::out::Out,
    result: PointResult,
    store: store::Store,
    info: store::RunInfo,
    paths: &ExportPaths<'_>,
    output_path: Option<&std::path::Path>,
    no_store: bool,
    model: &ir::Model,
) {
    use crate::pfilter_store::*;
    let (stored, computed, by_products_failed) = match result {
        PointResult::Stored { stored, kept, by_products_failed } => (stored, kept, by_products_failed),
        PointResult::Failed(e) => {
            let _ = info.finish(&store);
            drop(store);
            out.error(e).exit();
        }
    };
    let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
        out.say(format_args!("pfilter: @{label} now names invocation {}, which held it from invocation {before}",
            invocation.seq()));
    }
    let loglik = crate::camdl_store::read_stored(&stored, LOGLIK_FILE)
        .and_then(|b| Loglik::from_bytes(&b))
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let n_particles = loglik.n_particles;
    let failed = loglik.n_failed > 0;

    // ── The exports ─────────────────────────────────────────────────────────
    //
    // A filter that stopped with an error stored no by-product, so a run that
    // failed exports the ones it has.
    let stored_bytes = |kind: &str, file: &str| -> Option<Vec<u8>> {
        match by_product_bytes(&store, &invocation, kind, file) {
            Ok(b) => Some(b),
            Err(_) if failed => None,
            Err(e) => out.fail(format!("error: {e}")),
        }
    };
    let trace_to_stdout = paths.trace == Some("-");
    if let Some(path) = paths.trace {
        if let Some(bytes) = stored_bytes(Trace::KIND, Trace::FILE) {
            export_bytes(out, path, &bytes, "trace");
        }
    }
    if let Some(path) = paths.pf_health {
        if let Some(bytes) = stored_bytes(PfHealth::KIND, PfHealth::FILE) {
            export_bytes(out, path, &bytes, "pf-health");
            say_pf_health_summary(out, &bytes, n_particles as usize);
        }
    }
    if let Some(path) = paths.final_state {
        if let Some(bytes) = stored_bytes(FinalState::KIND, FinalState::FILE) {
            // The file's header names the origin, and each row after the two
            // header lines is one particle.
            let text = String::from_utf8_lossy(&bytes);
            let origin = text.lines().next().and_then(|l| l.split_once("\tt=")).map(|(_, t)| t.to_string()).unwrap_or_default();
            let n = text.lines().count().saturating_sub(2);
            export_bytes(out, &path.to_string_lossy(), &bytes, &format!("final particle states ({n} particles) at t={origin}"));
        }
    }
    if let Some((n_paths, path)) = paths.paths {
        if let Some(bytes) = stored_bytes(SavedPaths::KIND, SavedPaths::FILE) {
            export_bytes(out, &path.to_string_lossy(), &bytes, &format!("{n_paths} sample paths"));
        }
    }
    if let Some(stem) = paths.prequential {
        if let Some(json) = stored_bytes(Prequential::KIND, Prequential::JSON) {
            let tsv = format!("{stem}.tsv");
            let write = |to: &std::path::Path, file: &str| {
                let bytes = stored_bytes(Prequential::KIND, file).unwrap_or_default();
                std::fs::write(to, bytes)
                    .unwrap_or_else(|e| out.fail(format!("error writing prequential: {e}")));
            };
            write(&crate::prequential_out::streams_sibling(std::path::Path::new(&tsv)), Prequential::STREAMS);
            write(std::path::Path::new(&tsv), Prequential::JOINT);
            write(std::path::Path::new(&format!("{stem}.json")), Prequential::JSON);
            let trace: score::trace::PrequentialTrace = serde_json::from_slice(&json)
                .unwrap_or_else(|e| out.fail(format!("error: the stored prequential trace: {e}")));
            let calendar = ir::caltime::CalendarAxis::from_model(model)
                .unwrap_or_else(|e| out.fail(format!("error: {e}")));
            out.say(format_args!("prequential trace written: {}", crate::prequential_out::summary_line(&trace)));
            out.say_raw(format_args!("{}", crate::prequential_out::surprise_table(&trace, &calendar)));
        }
    }
    if let Some(path) = paths.filtering {
        if let Some(bytes) = stored_bytes(Filtering::KIND, Filtering::FILE) {
            export_bytes(out, &path.to_string_lossy(), &bytes, "filtering marginals");
        }
    }
    if let Some(path) = paths.filtered_streams {
        if let Some(bytes) = stored_bytes(FilteredStreams::KIND, FilteredStreams::FILE) {
            export_bytes(out, &path.to_string_lossy(), &bytes, "filtered stream means");
        }
    }
    if by_products_failed {
        out.say("warning: no by-product asked for was stored or exported: the filter stopped with an \
            error at these parameters, so it left no state to write them from");
    }
    let milestone = || {
        if !no_store {
            let folder = crate::run_paths::display_absolute(stored.folder());
            out.milestone(if computed { "stored" } else { "cached" }, folder.display());
            out.hint(format_args!("camdl cat {}", stored.id().key().to_hex()));
        }
    };

    // A filter that failed at every replicate has no likelihood to report:
    // the run is stored with its failure, and the command fails with it.
    if loglik.every_replicate_failed() {
        milestone();
        let first = loglik.failure.clone().unwrap_or_default();
        let message = if loglik.n_replicates > 1 {
            format!("{first} (all {} replicates; replicate 1 shown)", loglik.n_replicates)
        } else {
            first
        };
        drop(store);
        out.error(filter_failed(message)).exit();
    }

    // ── The log-likelihood ──────────────────────────────────────────────────
    if loglik.n_replicates > 1 {
        match (loglik.loglik, loglik.loglik_sd) {
            (Some(ll), sd) => out.say(format_args!("loglik = {:.1} ± {:.1} ({} replicates, N={})",
                ll, sd.unwrap_or(0.0), loglik.n_replicates, loglik.n_particles)),
            (None, _) => out.say(format_args!("loglik = -inf ({} replicates, N={})",
                loglik.n_replicates, loglik.n_particles)),
        }
        if let Some(first) = &loglik.failure {
            out.say(format_args!("warning: {} of {} replicates failed, so the mean is -inf; the first: {}",
                loglik.n_failed, loglik.n_replicates, first.strip_prefix("error: ").unwrap_or(first)));
        }
    }
    match output_path {
        Some(path) => {
            invocation
                .render::<LoglikTable>(&store, &(), &store::ExportPath(path.to_path_buf()))
                .unwrap_or_else(|e| out.fail(format!("cannot write {}: {e}", path.display())));
            if loglik.n_replicates > 1 {
                out.say(format_args!("replicate logliks written to {}", path.display()));
            } else {
                out.say(format_args!("loglik written to {}", path.display()));
            }
        }
        None => {
            let mut table = Vec::new();
            invocation
                .render_to::<LoglikTable>(&store, &(), &mut table)
                .unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let table = String::from_utf8_lossy(&table);
            if trace_to_stdout && loglik.n_replicates <= 1 {
                out.say(table.trim_end());
            } else {
                out.data_raw(table);
            }
        }
    }
    milestone();
}

/// The filter's outcome: one run, an error the filter stopped with, or each
/// replicate's log-likelihood and the cause of each one that failed.
enum Filtered {
    Single(Box<sim::inference::particle_filter::PFilterResult>),
    /// The filter stopped with an error the run's identity determines.
    Failed(String),
    Replicates { rows: Vec<(u64, f64)>, failures: Vec<Option<String>> },
}

/// Everything one computed run stores, in memory: the run's own files and
/// each by-product asked for.
struct RunFiles {
    loglik: Loglik,
    replicates: Option<Vec<u8>>,
    parameters: Vec<u8>,
    trace: Option<Vec<u8>>,
    pf_health: Option<Vec<u8>>,
    final_state: Option<Vec<u8>>,
    paths: Option<Vec<u8>>,
    prequential: Option<crate::prequential_out::PrequentialFiles>,
    filtering: Option<Vec<u8>>,
    filtered_streams: Option<Vec<u8>>,
}

/// Writes a run's own files and each by-product of it that was claimed.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
fn write_run(run: &mut store::Run<crate::pfilter_store::Pfilter>, f: &RunFiles) -> store::Result<()> {
    use crate::pfilter_store::*;
    run.result_file(LOGLIK_FILE)?.write_all(&f.loglik.to_bytes())?;
    if let Some(b) = &f.replicates {
        run.result_file(REPLICATES_FILE)?.write_all(b)?;
    }
    run.provenance_file(PROVENANCE_FILE)?.write_all(&f.parameters)?;
    #[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
    fn one<B>(run: &mut store::Run<Pfilter>, files: &[(&str, &[u8])]) -> store::Result<()>
    where
        B: store::ByProduct<Of = Pfilter>,
        B::Identity: Send + Sync,
    {
        if let Some(bp) = run.also::<B>()? {
            for (name, bytes) in files {
                bp.result_file(name)?.write_all(bytes)?;
            }
        }
        Ok(())
    }
    if let Some(b) = &f.trace {
        one::<Trace>(run, &[(Trace::FILE, b)])?;
    }
    if let Some(b) = &f.pf_health {
        one::<PfHealth>(run, &[(PfHealth::FILE, b)])?;
    }
    if let Some(b) = &f.final_state {
        one::<FinalState>(run, &[(FinalState::FILE, b)])?;
    }
    if let Some(b) = &f.paths {
        one::<SavedPaths>(run, &[(SavedPaths::FILE, b)])?;
    }
    if let Some(p) = &f.prequential {
        one::<Prequential>(run, &[(Prequential::JOINT, &p.joint), (Prequential::STREAMS, &p.streams), (Prequential::JSON, &p.json)])?;
    }
    if let Some(b) = &f.filtering {
        one::<Filtering>(run, &[(Filtering::FILE, b)])?;
    }
    if let Some(b) = &f.filtered_streams {
        one::<FilteredStreams>(run, &[(FilteredStreams::FILE, b)])?;
    }
    Ok(())
}

/// The bytes of one file of the by-product of kind `kind` this invocation
/// asked for.
fn by_product_bytes(store: &store::Store, invocation: &store::Invocation, kind: &str, file: &str) -> Result<Vec<u8>, String> {
    use crate::pfilter_store::*;
    macro_rules! read {
        ($($t:ty),*) => {
            $(if kind == <$t as store::Artifact>::KIND {
                let (id, ()) = invocation.members::<$t>().map_err(|e| e.to_string())?.into_iter().next()
                    .ok_or_else(|| format!("the invocation lists no {kind}"))?;
                let stored = store.get(&id).map_err(|e| e.to_string())?
                    .ok_or_else(|| format!("the {kind} {} is not stored", id.key().short12()))?;
                return crate::camdl_store::read_stored(&stored, file);
            })*
        };
    }
    read!(Trace, PfHealth, FinalState, SavedPaths, Prequential, Filtering, FilteredStreams);
    Err(format!("no by-product of kind {kind}"))
}

/// Writes `bytes` to `path`, or to stdout when `path` is `-`, and says so.
fn export_bytes(out: &crate::out::Out, path: &str, bytes: &[u8], what: &str) {
    if path == "-" {
        let _ = out.data().write_all(bytes);
        return;
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| out.fail(format!("cannot write {path}: {e}")));
    out.say(format_args!("{what} written to {path}"));
}

/// The `--pf-health` summary, read from the stored report: ESS-fraction
/// health and the Snyder implied-particle-count estimate.
fn say_pf_health_summary(out: &crate::out::Out, bytes: &[u8], n_particles: usize) {
    let text = String::from_utf8_lossy(bytes);
    let rows: Vec<(f64, f64)> = text.lines().skip(1).filter_map(|l| {
        let cols: Vec<&str> = l.split('\t').collect();
        Some((cols.get(2)?.parse().ok()?, cols.get(3)?.parse().ok()?))
    }).collect();
    if rows.is_empty() {
        return;
    }
    let median = |v: &[f64]| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[s.len() / 2]
    };
    let ess_frac: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let tau2: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let min_ess_frac = ess_frac.iter().copied().fold(f64::INFINITY, f64::min);
    let med_ess_frac = median(&ess_frac);
    let med_tau2 = median(&tau2);
    let max_tau2 = tau2.iter().copied().fold(0.0_f64, f64::max);
    out.say(format_args!("\npf-health summary ({} obs, N={}):", rows.len(), n_particles));
    out.say(format_args!("  ESS fraction:           median {:.1}%, worst {:.1}%",
        100.0 * med_ess_frac, 100.0 * min_ess_frac));
    out.say(format_args!("  log-weight variance τ²: median {:.2}, max {:.2}", med_tau2, max_tau2));
    out.say(format_args!("  implied N to avoid collapse exp(τ²/2): ~{} (median step), ~{} (worst step)",
        fmt_implied_n((0.5 * med_tau2).exp()),
        fmt_implied_n((0.5 * max_tau2).exp())));
    out.say("  [Snyder et al. 2008 heuristic — an order-of-magnitude floor at THIS θ and N, not a guarantee]");
}

use crate::caltime_load::{check_substeps_and_grid, convert_time_column, TimeOpts};


/// gh#90: fit-toml fallback for `camdl pfilter --fit fit.toml` (no CLI
/// `--data` flags). Reads `[data]` from the toml and returns a list of
/// (key, path) bindings — the shape `resolve_data_specs` produces, so
/// downstream code is uniform.
///
/// The keys are [`StreamBindings::identity_keyed_training`]'s: the run
/// identities of `pfilter` and `profile` fold one digest per binding, so the
/// layout is identity, not presentation.
///
/// [`StreamBindings::identity_keyed_training`]:
///     crate::fit::stream_bindings::StreamBindings::identity_keyed_training
pub fn load_data_observations_from_fit_toml(
    fit_path: &std::path::Path,
    model: &ir::Model,
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let path_str = fit_path.to_string_lossy().into_owned();
    let fit_cfg = crate::fit::config_v2::Problem::load(&path_str)
        .map_err(|e| format!("failed to load --fit toml '{}': {}", path_str, e))?;
    data_observations_of(&fit_cfg, model, &path_str)
}

/// [`load_data_observations_from_fit_toml`] for a problem already read:
/// the `[data]` bindings of `problem`, the fit `handle` names.
pub(crate) fn data_observations_of(
    problem: &crate::fit::config_v2::Problem,
    model: &ir::Model,
    handle: &str,
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let effective = crate::fit::stream_bindings::StreamBindings::resolve(model, problem.data_spec()?)?
        .identity_keyed_training();
    if effective.is_empty() {
        return Err(format!(
            "--fit '{}' [data] resolves to zero observation streams.",
            handle));
    }
    // Sort by name for deterministic ordering (matches fit/runner.rs).
    let mut entries: Vec<(String, std::path::PathBuf)> = effective.iter()
        .map(|(k, v)| (k.clone(), std::path::PathBuf::from(v)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// What the family that reads a column requires of each observed cell in it.
///
/// A count family scores non-negative integers, and the likelihood code takes
/// that as given: it coerces its argument (`y.round().max(0.0)`) so that the
/// filter never meets a value outside the support. Coercion is right inside
/// the filter and wrong at the file, where it turns a `-4` into an observed `0`
/// and a `4.4` into a `4` and scores them with a finite log-likelihood. So the
/// loader refuses such a cell, naming where it is (`camdl-data-spec.md`,
/// "Observed values must be in the family's support"), and nothing downstream
/// sees one. A `NA` hole is not a value and is never checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellDomain {
    /// Any finite number: a normal or beta outcome, or a covariate the
    /// likelihood reads through an expression.
    Real,
    /// A non-negative integer: the outcome of a count family, or a binomial
    /// trial count `n = <column>`. `family` names the likelihood for the
    /// diagnostic.
    Count { family: &'static str },
    /// 0 or 1: a Bernoulli outcome.
    Binary,
}

impl CellDomain {
    /// The domain of `column` in `obs`'s file: the scored column takes its
    /// family's support, a column bound bare to a binomial `n` is a count,
    /// and every other column is real.
    pub fn of(obs: &ir::observation::ObservationModel, column: &str) -> CellDomain {
        use ir::observation::Likelihood as L;
        let family = obs.likelihood.name();
        if column == obs.scored {
            return match &obs.likelihood {
                L::Poisson(_) | L::NegBinomial(_) | L::ZeroInflatedNegBinomial(_)
                | L::Binomial(_) | L::BetaBinomial(_) => CellDomain::Count { family },
                L::Bernoulli(_) => CellDomain::Binary,
                L::Normal(_) | L::Beta(_) => CellDomain::Real,
            };
        }
        let trials = match &obs.likelihood {
            L::Binomial(b) => Some(&b.n),
            L::BetaBinomial(bb) => Some(&bb.n),
            _ => None,
        };
        match trials {
            Some(ir::expr::Expr::ObsColumnRef(w)) if w.obs_column_ref == column =>
                CellDomain::Count { family },
            _ => CellDomain::Real,
        }
    }

    /// `None` when `v` is in the domain; otherwise what the value should have
    /// been, phrased to follow "is not ...".
    fn refuses(self, v: f64) -> Option<String> {
        match self {
            CellDomain::Real => None,
            CellDomain::Count { family } => (v < 0.0 || v.fract() != 0.0).then(|| format!(
                "a count — {family} scores non-negative integers")),
            CellDomain::Binary => (v != 0.0 && v != 1.0).then(|| {
                "0 or 1 — bernoulli scores a binary outcome".to_string()
            }),
        }
    }
}

/// The located refusal for a cell outside its column's [`CellDomain`].
fn check_cell_domain(
    domain: CellDomain,
    v: f64,
    raw: &str,
    path: &str,
    column: &str,
    line: usize,
) -> Result<(), String> {
    match domain.refuses(v) {
        None => Ok(()),
        Some(should_be) => Err(format!(
            "{path}, line {line}: value '{raw}' in column '{column}' is not {should_be}. \
             Fix the row in the file; write `NA` if the value was not observed.")),
    }
}

/// What a column-lookup diagnostic calls the column it could not find.
///
/// The shared raw parser takes two columns — a temporal axis and a value — but
/// not every caller puts the stream's `: time` column in the first slot: a
/// window boundary goes there, and so does each covariate the likelihood reads
/// (a binomial denominator, a person-time offset), because those are read by
/// the same strict by-name path. Reporting a missing covariate as a missing
/// time column, with the `time : time` rule as the fix, sends the modeller to
/// look at a time column that is present and fine and away from the
/// denominator that is not (gh#886) — so the slot says what it holds.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnPurpose {
    /// The stream's `: time` axis column.
    Time,
    /// A per-row window boundary (`window_start` / `window_stop`).
    WindowBoundary,
    /// A column the likelihood reads by name, declared in `columns { }` with
    /// its own type (`tested : count`).
    Covariate,
}

impl ColumnPurpose {
    /// What the diagnostic calls this kind of column.
    fn what(self) -> &'static str {
        match self {
            ColumnPurpose::Time => "time",
            ColumnPurpose::WindowBoundary => "window boundary",
            ColumnPurpose::Covariate => "covariate",
        }
    }

    /// The declaration the header must match, named as the modeller wrote it.
    fn declaration(self) -> &'static str {
        match self {
            ColumnPurpose::Time => "the declared `time : time` column name",
            ColumnPurpose::WindowBoundary =>
                "the declared `window_start` / `window_stop` column name",
            ColumnPurpose::Covariate => "the declared `columns { }` name",
        }
    }
}

/// Read the rows of two named TSV columns — the time column and one other —
/// as their raw cells, one pair per data row, with the file line number of
/// each row. Nothing is converted here: this half of the reader answers only
/// "which columns, and what does each row say", so a caller that wants a time
/// (dates included) and a caller that wants a value share the header lookup
/// and the row iteration without sharing a conversion only one of them
/// wants.
///
/// Strict by-name column binding, no positional fallback (G1).
fn parse_column_cells_raw<'a>(
    content: &'a str,
    path: &str,
    time_column: &str,
    purpose: ColumnPurpose,
    column: &str,
) -> Result<(Vec<&'a str>, Vec<&'a str>, Vec<usize>), String> {
    // gh#344: skip leading `#` comment lines (a provenance/attribution header is
    // a common convention) and blank lines before the column header —
    // consistent with the `interpolated` forcing reader and polars
    // `comment_prefix`. Enumerate the whole file so error line numbers stay
    // accurate past any skipped comments.
    let mut lines = content.lines().enumerate();
    let (_, header) = lines
        .by_ref()
        .find(|(_, l)| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .ok_or("empty data file (or only comment/blank lines)")?;
    let cols: Vec<&str> = header.split('\t').collect();

    // Find the TIME column index BY NAME (the by-name-time flip — no
    // positional "column 0 is time" fallback; 2026-06-10 §6.2). A file whose
    // headers do not match the declared `time` column is a located error.
    let time_idx = cols.iter().position(|&c| c == time_column)
        .ok_or_else(|| format!(
            "{} column '{time_column}' not found in data file '{path}'. \
             Headers present: [{}]. Fix: add the column, or rename the data \
             column to '{time_column}' (it must match {}, case-sensitive).",
            purpose.what(), cols.join(", "), purpose.declaration()))?;

    // Find the VALUE column index for the requested stream. Binding is
    // strict by name — there is NO positional fallback. A typo'd,
    // wrong-cased, or renamed header is a located error, not a silent
    // bind to whatever column happens to be positionally first (G1: a
    // wrong-answer-with-exit-0). The data column header must match the
    // declared `scored` column exactly.
    let col_idx = cols.iter().position(|&c| c == column)
        .ok_or_else(|| {
            let available = cols.iter().filter(|&&c| c != time_column)
                .copied().collect::<Vec<_>>();
            let available = if available.is_empty() {
                "(no value columns — only a time column)".to_string()
            } else {
                available.join(", ")
            };
            format!(
                "observation column '{column}' not found in data file '{path}'. \
                 Value column headers present: [{available}]. \
                 Fix: rename the data column to '{column}' (it must match the \
                 declared `scored` column exactly, case-sensitive), or rename the \
                 model's `columns {{ }}` to match the data header."
            )
        })?;

    let max_idx = time_idx.max(col_idx);

    // Two passes: the raw cells are collected here, and a caller converts
    // the time column whole, in one go (whole-column detection — proposal
    // §6.3).
    let mut time_cells: Vec<&str> = Vec::new();
    let mut value_cells: Vec<&str> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    for (line_idx, line) in lines {
        // `line_idx` is the 0-based file line index (enumerate over the whole
        // file), so the human line number is `line_idx + 1` — accurate past any
        // skipped leading `#` comments (gh#344). Skip blank + comment rows.
        if line.trim().is_empty() || line.trim_start().starts_with('#') { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= max_idx {
            return Err(format!("line {}: expected {}+ columns, got {}",
                line_idx + 1, max_idx + 1, fields.len()));
        }
        time_cells.push(fields[time_idx]);
        value_cells.push(fields[col_idx]);
        rows.push(line_idx + 1);
    }

    Ok((time_cells, value_cells, rows))
}

/// Parse the raw rows of one named TSV column into per-row cells. A cell is
/// `None` for a hole (the missing-value token `NA`) and `Some(v)` for an
/// observed finite value. The time of a hole row is retained (the row is
/// kept) so the observation grid is unchanged — only the value is absent.
///
/// `NaN`/`inf` are rejected as garbage (a hole is `NA`, not a non-finite
/// number), and so is a value outside `domain` — a negative or fractional
/// count. Strict by-name column binding, no positional fallback (G1).
///
/// The core of [`load_data_tsv_column_cells_as_written`] (the value column)
/// and [`load_stream_aux`] (the covariate columns).
fn parse_column_cells<'a>(
    content: &'a str,
    path: &str,
    time_column: &str,
    purpose: ColumnPurpose,
    column: &str,
    domain: CellDomain,
) -> Result<(Vec<&'a str>, Vec<Option<f64>>, Vec<usize>), String> {
    let (time_cells, value_cells, rows) =
        parse_column_cells_raw(content, path, time_column, purpose, column)?;

    let mut cells: Vec<Option<f64>> = Vec::with_capacity(value_cells.len());
    for (i, &cell) in value_cells.iter().enumerate() {
        let raw = cell.trim();
        // TODO: make the missing-value token (`NA`) a user option (CLI flag /
        // config) — hard-coded for now.
        let parsed = if raw == "NA" {
            None // hole: time retained, value absent → no likelihood term
        } else {
            let value: f64 = raw.parse()
                .map_err(|_| format!("line {}: cannot parse value '{}' in column '{}'",
                    rows[i], cell, column))?;
            if !value.is_finite() {
                return Err(format!(
                    "line {} (t='{}'): non-finite observation value '{}' in column '{}' \
                     — NaN and infinities are not valid observations (a missing value \
                     is the token `NA`). Fix or remove the row.",
                    rows[i], time_cells[i].trim(), raw, column));
            }
            check_cell_domain(domain, value, raw, path, column, rows[i])?;
            Some(value)
        };
        cells.push(parsed);
    }

    Ok((time_cells, cells, rows))
}

/// The declared `Time`-role column name for an observation stream — the fit
/// time source (the by-name-time flip; 2026-06-10 §2.5/§6.2). A stream's
/// `columns { }` must declare exactly one `: time` column (the OCaml expander
/// enforces this at compile); this surfaces a clear error if a malformed IR
/// (no time column) somehow reaches the loader.
pub fn obs_time_column(obs: &ir::observation::ObservationModel) -> Result<&str, String> {
    // A windowed stream takes its time from the window's closing boundary —
    // the proposal's "the fit time source for a windowed stream is
    // `window_stop`" — so every downstream "observation time" consumer keeps
    // the position it has for an unwindowed stream. That is a position on the
    // axis only; it says nothing about when the observation was reported
    // (gh#833).
    if let Some(c) = column_with_role(obs, &ir::observation::ColumnRole::WindowStop) {
        return Ok(c);
    }
    obs.columns.iter()
        .find(|c| c.role == ir::observation::ColumnRole::Time)
        .map(|c| c.name.as_str())
        .ok_or_else(|| format!(
            "observation stream '{}' declares no `: time` column in `columns {{ }}` \
             — cannot determine the time axis to bind.",
            obs.name))
}

/// The name of the column carrying `role`, if the stream declares one.
pub(crate) fn column_with_role<'a>(
    obs: &'a ir::observation::ObservationModel,
    role: &ir::observation::ColumnRole,
) -> Option<&'a str> {
    obs.columns.iter().find(|c| &c.role == role).map(|c| c.name.as_str())
}

/// Read a per-row window boundary column (`window_start`) as raw cells, with
/// each row's file line number — the shape [`convert_time_column`] takes.
///
/// A boundary is a time, so it is read through the raw half of the parser and
/// converted by the time path: an ISO date is exactly as legal here as it is
/// in a `: time` column, which is what the data spec's window examples print.
/// Putting it through [`parse_column_cells`] instead would run the value
/// column's `f64` parse over it and reject every dated file before the
/// conversion that handles dates ever saw the cells.
///
/// `NA` is not a boundary. The hole token says a row's observed *value* is
/// missing — the row keeps its period and contributes no likelihood term — so
/// a row whose window has no start states no period at all, and there is
/// nothing to read it as. It is an error naming the row, never a skipped or
/// defaulted window. (A missing `window_stop` is refused too, a few lines on:
/// the stop is the stream's time column, so it reaches
/// [`convert_time_column`], which has no reading for `NA` either.)
fn read_boundary_column<'a>(
    content: &'a str,
    path: &str,
    column: &str,
) -> Result<(Vec<&'a str>, Vec<usize>), String> {
    // The raw reader wants a time column and a value column; here they are the
    // same column, and only one copy of the cells is wanted.
    let (cells, _same, rows) =
        parse_column_cells_raw(content, path, column, ColumnPurpose::WindowBoundary, column)?;
    for (i, &cell) in cells.iter().enumerate() {
        if cell.trim() == "NA" {
            return Err(format!(
                "line {}: window boundary column '{}' in '{}' is `NA` — every row must \
                 say what period it covers. `NA` says a row's observed value is missing \
                 (the row keeps its window and is simply not scored); a window with no \
                 start has no reading at all. Fix: give the row its boundary, or drop \
                 the row — dropping it states a gap, and the flow over it is discarded.",
                rows[i], column, path));
        }
    }
    Ok((cells, rows))
}

/// The error for an interval stream whose IR carries no `covers`: the compiler
/// refuses to produce one (E350), so the IR was hand-edited or predates the
/// rule. There is no reading to fall back on — the same label means different
/// periods in different files — so the loader and the emitters both refuse.
pub(crate) fn missing_covers_error(obs: &ir::observation::ObservationModel) -> String {
    format!(
        "observation stream '{}' accumulates flow but its IR does not say what each \
         row covers: the compiler requires `covers = ...` or `window_start`/`window_stop` \
         columns on an incidence stream (E350), so this IR did not come from the current \
         compiler. Recompile the model.",
        obs.name
    )
}

/// The `window_start` cells of every row of `path`, in model time and as
/// written, for a stream declared with per-row windows; `None` for any other
/// stream, which has no such column to read.
#[expect(clippy::disallowed_methods, reason = "reads a bound observation data file whose path its caller passes (`fit::observed_rows`): a stored fit problem's training tables, a filter run's or prediction's `DataTable` dependency, or a command that stores nothing from it (`simulate --design-from`, anchoring to the data)")]
pub(crate) fn load_window_starts_if_windowed(
    obs: &ir::observation::ObservationModel,
    path: &str,
    opts: &TimeOpts,
) -> Result<Option<(Vec<f64>, Vec<crate::caltime_load::TemporalCell>)>, String> {
    if obs.covers != Some(ir::observation::Covers::WindowColumns) {
        return Ok(None);
    }
    let start_col = column_with_role(obs, &ir::observation::ColumnRole::WindowStart)
        .ok_or_else(|| format!(
            "observation stream '{}' is declared with per-row windows but has no \
             `window_start` column", obs.name))?;
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let (raw, rows) = read_boundary_column(&content, path, start_col)?;
    let row_offset = rows.first().copied().unwrap_or(2);
    let starts = convert_time_column(&raw, opts, row_offset)?;
    let written = crate::caltime_load::cells_as_written(&raw, &starts, opts);
    Ok(Some((starts, written)))
}

/// Build the [`StreamTimes`] for one stream from its labels: what its rows
/// cover, per its `covers` declaration (gh#833).
///
/// `label_times` are the already-converted values of the stream's own label
/// column — the `: time` column, or `window_stop` for a windowed stream. The
/// lowering tests' view of [`stream_times_of_rows`]: every loader reads its
/// rows through [`crate::fit::observed_rows::observed_rows`] and uses that.
/// Both lower labels through the one
/// [`crate::fit::observed_rows::measures_of`].
///
/// An instant stream reads at its labels. An interval stream gets the periods
/// its declaration assigns, and its scoring boundary follows them — for a
/// daily file labelled by the day the count describes, one bucket after the
/// label, which is the correction this exists to make. An interval stream
/// with no declaration is an IR the compiler cannot have produced (E350), and
/// is refused rather than read under a convention the file never stated.
///
/// Labels handed in directly carry no written form, so a refusal quotes them
/// as model time; the tests of dated quoting read through
/// [`crate::fit::observed_rows::observed_rows`].
#[cfg(test)]
pub fn stream_times_for(
    obs: &ir::observation::ObservationModel,
    projection: &sim::inference::multi_stream_obs::StreamProjection,
    label_times: &[f64],
    path: &str,
    opts: &TimeOpts,
) -> Result<sim::inference::StreamTimes, String> {
    let window_starts = load_window_starts_if_windowed(obs, path, opts)?;
    let measures = crate::fit::observed_rows::measures_of(
        obs,
        projection.temporal_kind(),
        label_times,
        window_starts.as_ref().map(|(starts, _)| starts.as_slice()),
        path,
    )?;
    stream_times_of_measures(obs, &measures, || false, path, opts)
}

/// Build the [`StreamTimes`] for one stream from the rows
/// [`crate::fit::observed_rows::observed_rows`] read: the periods (or
/// instants) the rows already carry, checked against the run.
pub(crate) fn stream_times_of_rows(
    obs: &ir::observation::ObservationModel,
    rows: &[crate::fit::observed_rows::ObservedRow],
    path: &str,
    opts: &TimeOpts,
) -> Result<sim::inference::StreamTimes, String> {
    let measures: Vec<_> = rows.iter().map(|r| r.measures).collect();
    stream_times_of_measures(
        obs, &measures, || crate::fit::observed_rows::rows_were_dated(rows), path, opts)
}

/// The [`StreamTimes`] of rows that measure `measures`, refused when a period
/// opens before the run starts or, under a uniform form, leaves a gap. `dated`
/// says whether the file wrote the rows' times as dates, so a refusal quotes
/// them that way; it is asked only on a refusal.
fn stream_times_of_measures(
    obs: &ir::observation::ObservationModel,
    measures: &[crate::fit::observed_rows::Measures],
    dated: impl Fn() -> bool,
    path: &str,
    opts: &TimeOpts,
) -> Result<sim::inference::StreamTimes, String> {
    use crate::fit::observed_rows::Measures;
    use ir::observation::Covers;
    use sim::inference::StreamTimes;

    // `measures_of` gives every row of a stream without `covers` an instant,
    // and every row of a stream with one a period.
    let Some(covers) = &obs.covers else {
        return Ok(StreamTimes::Instants(measures.iter().map(Measures::end).collect()));
    };
    let periods: Vec<_> = measures.iter().map(|m| match m {
        Measures::Period(p) => *p,
        Measures::Instant(_) => unreachable!("a stream that declares `covers` measures periods"),
    }).collect();

    // A period may not open before the run does: time before `t_start` is
    // never simulated, so the period would be scored against flow that does
    // not exist. Opening exactly AT `t_start` is fine — the accumulator is
    // already empty there. Opening after it is the normal warm-up shape and is
    // handled by resetting the bin at the period's start (gh#833). A
    // non-finite `t_start` means no fit window is in force; nothing to check.
    if opts.t_start.is_finite() {
        if let Some(first) = periods.first() {
            if first.start() < opts.t_start - 1e-9 {
                let dated = dated();
                let q = |t: f64| crate::caltime_load::quote_time(t, dated, opts);
                // On a dated file the run's start is quoted as the date it is,
                // and the fix names the anchor that moves it. On a numeric one
                // `t_start` is what the modeller sees and `simulate.from` is
                // what sets it, so the message says so.
                let (start, fix) = match dated {
                    true => (
                        q(opts.t_start),
                        format!(
                            "Fix: make the run start at {} or earlier — move the model's \
                             `origin` back, or lower `simulate.from` — or drop the rows the \
                             model cannot cover.", q(first.start())),
                    ),
                    false => (
                        format!("t_start = {}", opts.t_start),
                        format!(
                            "Fix: move `simulate.from` back to {} or earlier, or drop the \
                             rows the model cannot cover.", first.start()),
                    ),
                };
                return Err(format!(
                    "observation stream '{}' ({}): its first declared period opens at {} but \
                     the run starts at {} — time before the run starts is never simulated, \
                     so the period would be scored against flow that does not exist.\n  {}",
                    obs.name, path, q(first.start()), start, fix));
            }
        }
    }

    // A gap between consecutive periods means different things under
    // different forms, and only the loader can see the form. Under a UNIFORM
    // form (`day` / `starting_on` / `ending_on`) every row's window is
    // implied by its label, so a gap can only mean a row is MISSING — and a
    // missing row is exactly the shape that used to widen the next bin
    // silently. Refuse it, naming both boundaries. A row with no observation
    // belongs in the file as `NA`; a window that genuinely spans several days
    // belongs in the per-row form. Under `window_start`/`window_stop` a gap
    // is a statement and is legal: the binder discards the uncovered flow.
    if matches!(covers, Covers::From { .. } | Covers::Until { .. }) {
        // Within `BOUNDARY_EPS` is contiguous: off a whole-unit axis the next
        // label and the previous stop are the same boundary a few ulps apart.
        let eps = sim::inference::multi_stream_obs::BOUNDARY_EPS;
        if let Some(w) = periods.windows(2).find(|w| w[1].start() > w[0].stop() + eps) {
            let dated = dated();
            let q = |t: f64| crate::caltime_load::quote_time(t, dated, opts);
            return Err(format!(
                "observation stream '{}' ({}): consecutive rows cover [{}, {}) and \
                 [{}, {}), leaving {} to {} covered by neither — under a uniform \
                 `covers` form that means a row is missing, which would silently \
                 change what the next row is scored against.\n  \
                 Fix: keep every scheduled row and write an unobserved one as NA; \
                 or, if one row genuinely covers the whole span, state its window \
                 with `window_start`/`window_stop` columns.",
                obs.name, path, q(w[0].start()), q(w[0].stop()), q(w[1].start()),
                q(w[1].stop()), q(w[0].stop()), q(w[1].start())));
        }
    }
    Ok(StreamTimes::Intervals(periods))
}

/// Load one wide stream's rows from a TSV by NAME: both `time_column` (the
/// time axis) and `column` (the value) must match header fields exactly, with
/// no positional fallback for either. Returns the converted observation
/// `times`, the time cells as written, and the per-row cells, where `None` is
/// a hole (the `NA` token): its time stays in the grid (so the incidence
/// accumulator still resets there) but it carries no value (no likelihood
/// term). Only the value column may contain `NA`. Read through
/// [`crate::fit::observed_rows::observed_rows`].
#[expect(clippy::disallowed_methods, reason = "reads a bound observation data file whose path its caller passes (`fit::observed_rows`): a stored fit problem's training tables, a filter run's or prediction's `DataTable` dependency, or a command that stores nothing from it (`simulate --design-from`, anchoring to the data)")]
pub(crate) fn load_data_tsv_column_cells_as_written(
    path: &str,
    time_column: &str,
    column: &str,
    domain: CellDomain,
    opts: &TimeOpts,
) -> Result<
    (Vec<f64>, Vec<crate::caltime_load::TemporalCell>, Vec<Option<sim::inference::ObsCell>>),
    String,
> {
    use sim::inference::ObsCell;
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let (time_cells, cells, rows) =
        parse_column_cells(&content, path, time_column, ColumnPurpose::Time, column, domain)?;

    // Convert the time column + run the distinct-substep/off-grid/ordering
    // checks via the same back-half used by the dense path — but on the time
    // axis only, since holes have no value. We materialize a value vector
    // where holes use a placeholder (0.0) purely to reuse `finalize_observations`
    // for time conversion + grid checks; the placeholder is then discarded and
    // the authoritative cells are returned.
    let placeholder_values: Vec<f64> = cells.iter().map(|c| c.unwrap_or(0.0)).collect();
    let observations = finalize_observations(&time_cells, placeholder_values, rows, opts)?;

    let times: Vec<f64> = observations.iter().map(|o| o.time).collect();
    let written = crate::caltime_load::cells_as_written(&time_cells, &times, opts);
    let obs_cells: Vec<Option<ObsCell>> = cells.iter()
        .map(|c| c.map(ObsCell::Scalar))
        .collect();
    Ok((times, written, obs_cells))
}

/// Is this stream's data in LONG (stratified) form? True iff its declared
/// `columns { }` contains at least one `: dim` column — the dispatch axis
/// between the wide/by-name loader (one value column per file) and the
/// long-form loader (one row per `(time, level, value)`, routed by name to
/// the matching stratum leaf; 2026-06-10 observation data-entry §4.2).
pub fn is_long_form_stream(obs: &ir::observation::ObservationModel) -> bool {
    obs.columns.iter()
        .any(|c| matches!(c.role, ir::observation::ColumnRole::Dim(_)))
}

/// The declared `: dim` column names of a stream, in declaration order. Empty
/// for an unstratified stream.
fn dim_column_names(obs: &ir::observation::ObservationModel) -> Vec<&str> {
    obs.columns.iter()
        .filter_map(|c| match &c.role {
            ir::observation::ColumnRole::Dim(d) => Some(d.as_str()),
            _ => None,
        })
        .collect()
}

/// Load ONE leaf of a stratified observation family from a LONG-FORM file
/// (§4.2). The file carries a `time` column, one or more `: dim` columns, the
/// scored value column, and any aux columns; each row is `(time, {dim→level},
/// value, aux…)`. This loader:
///
/// - builds the UNION time axis across every row in the file (all strata
///   share it — the normal partial-coverage serosurvey shape);
/// - routes each row to the leaf whose `stratum` matches the row's
///   `{dim→level}` BY NAME (order-independent (dim,level) set match);
/// - for THIS `obs_block`, emits one cell per union time: `Some(Scalar(v))`
///   when a matching row carries a finite value, `None` for a hole — both a
///   `NA` value in a present row AND a union time where the leaf has no row
///   (partial coverage). A hole carries no likelihood term and no false zero.
///
/// `siblings` is every observation block sharing this leaf's `source` (the
/// whole stratified family, including `obs_block`); their strata define the
/// valid level set per dim. A row whose level for some dim is absent from
/// every sibling's stratum is a hard error (E281) — never silently remapped.
///
/// Returns `(times, written, cells, aux)` in the shape the wide loader
/// ([`load_data_tsv_column_cells_as_written`]) produces, with this leaf's time
/// cells as written beside their converted times. Read through
/// [`crate::fit::observed_rows::observed_rows`].
#[expect(clippy::disallowed_methods, reason = "reads a bound observation data file whose path its caller passes (`fit::observed_rows`): a stored fit problem's training tables, a filter run's or prediction's `DataTable` dependency, or a command that stores nothing from it (`simulate --design-from`, anchoring to the data)")]
pub(crate) fn load_long_form_stream_as_written(
    path: &str,
    obs_block: &ir::observation::ObservationModel,
    siblings: &[&ir::observation::ObservationModel],
    opts: &TimeOpts,
) -> Result<
    (
        Vec<f64>,
        Vec<crate::caltime_load::TemporalCell>,
        Vec<Option<sim::inference::ObsCell>>,
        Vec<Vec<(String, f64)>>,
    ),
    String,
> {
    use sim::inference::ObsCell;
    use std::collections::BTreeMap;

    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    // gh#344: skip leading `#` comment lines + blank lines before the header,
    // consistent with the forcing reader / polars `comment_prefix` and with
    // `parse_column_cells`. Enumerate the whole file so error line numbers stay
    // accurate past skipped comments.
    let mut lines = content.lines().enumerate();
    let (_, header) = lines
        .by_ref()
        .find(|(_, l)| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .ok_or("empty data file (or only comment/blank lines)")?;
    let cols: Vec<&str> = header.split('\t').collect();

    let col_idx = |name: &str, what: &str| -> Result<usize, String> {
        cols.iter().position(|&c| c == name).ok_or_else(|| format!(
            "{what} column '{name}' not found in data file '{path}'. \
             Headers present: [{}]. Fix: rename the data column to '{name}' \
             (it must match the declared `columns {{ }}` name, case-sensitive).",
            cols.join(", ")))
    };

    let time_col = obs_time_column(obs_block)?;
    let time_idx = col_idx(time_col, "time")?;
    let value_idx = col_idx(&obs_block.scored, "observation value")?;

    let dim_names = dim_column_names(obs_block);
    let dim_idxs: Vec<(String, usize)> = dim_names.iter()
        .map(|d| Ok::<_, String>(((*d).to_string(), col_idx(d, "dimension")?)))
        .collect::<Result<_, _>>()?;

    let aux_cols = stream_aux_columns(obs_block);
    let aux_idxs: Vec<(String, usize)> = aux_cols.iter()
        .map(|c| Ok::<_, String>((c.clone(), col_idx(c, "covariate")?)))
        .collect::<Result<_, _>>()?;

    // Valid level set per dim = the UNION of all sibling leaves' strata for
    // that dim. The IR is self-contained: a level the model iterates over is
    // exactly a level some sibling leaf observes. (Sorted for a stable error.)
    let mut valid_levels: BTreeMap<&str, std::collections::BTreeSet<String>> =
        BTreeMap::new();
    for sib in siblings {
        for sk in &sib.stratum {
            valid_levels.entry(sk.dim.as_str()).or_default().insert(sk.level.clone());
        }
    }

    // This leaf's stratum as a `{dim→level}` map for the routing match.
    let leaf_key: BTreeMap<&str, &str> = obs_block.stratum.iter()
        .map(|sk| (sk.dim.as_str(), sk.level.as_str()))
        .collect();

    // Pass 1: parse every row of the file. Collect (raw_time, file_row,
    // {dim→level}, scored cell, aux cells). Validate levels against the model
    // (E281) here so an unknown level fails regardless of which leaf owns it.
    struct Row<'a> {
        raw_time: &'a str,
        file_row: usize,
        key:      BTreeMap<&'a str, String>,
        value:    Option<f64>,
        aux:      Vec<(String, Option<f64>)>,
    }
    let max_idx = [time_idx, value_idx].into_iter()
        .chain(dim_idxs.iter().map(|(_, i)| *i))
        .chain(aux_idxs.iter().map(|(_, i)| *i))
        .max().unwrap();

    let parse_cell = |raw: &str, what: &str, column: &str, file_row: usize|
        -> Result<Option<f64>, String> {
        let raw = raw.trim();
        if raw == "NA" { return Ok(None); }
        let v: f64 = raw.parse().map_err(|_| format!(
            "line {file_row}: cannot parse {what} value '{raw}'"))?;
        if !v.is_finite() {
            return Err(format!(
                "line {file_row}: non-finite {what} value '{raw}' — NaN and \
                 infinities are not valid observations (a missing value is the \
                 token `NA`). Fix or remove the row."));
        }
        check_cell_domain(CellDomain::of(obs_block, column), v, raw, path, column, file_row)?;
        Ok(Some(v))
    };

    let mut rows: Vec<Row> = Vec::new();
    for (line_idx, line) in lines {
        if line.trim().is_empty() || line.trim_start().starts_with('#') { continue; }
        let file_row = line_idx + 1;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= max_idx {
            return Err(format!("line {}: expected {}+ columns, got {}",
                file_row, max_idx + 1, fields.len()));
        }
        // Dim levels — validate each against the model's level set (E281).
        let mut key: BTreeMap<&str, String> = BTreeMap::new();
        for (dim, idx) in &dim_idxs {
            let level = fields[*idx].trim().to_string();
            let known = valid_levels.get(dim.as_str())
                .is_some_and(|set| set.contains(&level));
            if !known {
                let levels = valid_levels.get(dim.as_str())
                    .map(|s| s.iter().cloned().collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                return Err(format!(
                    "[E281] line {file_row} in '{path}': unknown level '{level}' \
                     in column '{dim}'; model '{dim}' levels are [{levels}]. \
                     A `: dim` column's values match the model dimension's levels \
                     BY NAME — re-bin the data upstream (or aggregate, §5); never \
                     silently remapped."));
            }
            key.insert(dim.as_str(), level);
        }
        let value = parse_cell(fields[value_idx], "observation", &obs_block.scored, file_row)?;
        let mut aux = Vec::with_capacity(aux_idxs.len());
        for (name, idx) in &aux_idxs {
            aux.push((name.clone(), parse_cell(fields[*idx], "aux", name, file_row)?));
        }
        rows.push(Row {
            raw_time: fields[time_idx],
            file_row,
            key,
            value,
            aux,
        });
    }

    // Per-leaf OWN time axis: only the rows whose `{dim→level}` equals THIS
    // leaf's stratum (set match), in file order. Each leaf reports its OWN
    // schedule to `bind` (exactly like an unstratified separate stream), which
    // merges the family onto the union axis and marks the times where this leaf
    // has NO row as NOT-SCHEDULED (`at_union = None`). That distinction is
    // load-bearing for INCIDENCE: a leaf's accumulator resets only at ITS OWN
    // observations, never at a sibling stratum's time. Routing every leaf onto a
    // shared union with holes at sibling-only times instead would make each leaf
    // "scheduled" everywhere and reset incidence bins on the union cadence —
    // silently truncating a ragged-per-stratum incidence stream (PR#218 #1).
    //
    // An explicit `NA` row (value or aux missing) that BELONGS to the leaf stays
    // a HOLE on the leaf's own axis (scheduled, no term, still resets for
    // incidence) — the "we sampled, got nothing usable" case, distinct from an
    // absent row ("we don't sample on this cadence", not-scheduled).
    let mut own_raw: Vec<&str> = Vec::new();
    let mut own_file_rows: Vec<usize> = Vec::new();
    let mut cells: Vec<Option<ObsCell>> = Vec::new();
    let mut aux: Vec<Vec<(String, f64)>> = Vec::new();
    for r in &rows {
        let belongs = leaf_key.len() == r.key.len()
            && leaf_key.iter().all(|(&d, &lv)| r.key.get(d).map(String::as_str) == Some(lv));
        if !belongs { continue; }
        // Two rows of the SAME leaf at the SAME time is a data error (ambiguous
        // cell). `own_raw` is short (this leaf's rows), so a linear scan is fine.
        if own_raw.contains(&r.raw_time) {
            return Err(format!(
                "line {} in '{}': duplicate row for stratum {:?} at the same time \
                 — each (time, stratum) cell must be unique.",
                r.file_row, path,
                leaf_key.iter().map(|(d, l)| format!("{d}={l}")).collect::<Vec<_>>()));
        }
        own_raw.push(r.raw_time);
        own_file_rows.push(r.file_row);
        // present-together-or-hole: a `NA` scored value OR any `NA` aux ⇒ hole.
        let any_aux_na = r.aux.iter().any(|(_, v)| v.is_none());
        match (r.value, any_aux_na) {
            (Some(v), false) => {
                cells.push(Some(ObsCell::Scalar(v)));
                aux.push(r.aux.iter()
                    .map(|(name, v)| (name.clone(), v.expect("checked no NA")))
                    .collect());
            }
            _ => {
                cells.push(None); // hole: value or aux absent
                aux.push(Vec::new());
            }
        }
    }

    // Convert + validate THIS leaf's own times (per-leaf dated/numeric detection
    // + substep/off-grid/ordering checks, with the leaf's REAL file-row line
    // numbers in any diagnostic — not synthetic union positions).
    let placeholder = vec![0.0_f64; own_raw.len()];
    let own_obs = finalize_observations(&own_raw, placeholder, own_file_rows, opts)?;
    let own_times: Vec<f64> = own_obs.iter().map(|o| o.time).collect();
    let written = crate::caltime_load::cells_as_written(&own_raw, &own_times, opts);

    Ok((own_times, written, cells, aux))
}

/// The aux column names a stream's likelihood references (`Expr::ObsColumnRef`):
/// a binomial denominator `n = tested`, a person-time offset, a reporting
/// fraction. Returns them in declaration-stable order (de-duplicated).
pub fn stream_aux_columns(obs: &ir::observation::ObservationModel) -> Vec<String> {
    fn walk(e: &ir::expr::Expr, out: &mut Vec<String>) {
        use ir::expr::Expr;
        match e {
            Expr::ObsColumnRef(w) => {
                if !out.iter().any(|n| n == &w.obs_column_ref) {
                    out.push(w.obs_column_ref.clone());
                }
            }
            Expr::BinOp(w) => { walk(&w.bin_op.left, out); walk(&w.bin_op.right, out); }
            Expr::UnOp(w) => walk(&w.un_op.arg, out),
            Expr::Cond(w) => { walk(&w.cond.pred, out); walk(&w.cond.then, out); walk(&w.cond.else_, out); }
            Expr::TableLookup(w) => { for ix in &w.table_lookup.indices { walk(ix, out); } }
            Expr::UncheckedDim(w) => walk(&w.unchecked_dim.inner, out),
            Expr::Reduce(w) => { for t in &w.reduce { walk(t, out); } }
            _ => {}
        }
    }
    use ir::observation::Likelihood as L;
    let args: Vec<&ir::expr::Expr> = match &obs.likelihood {
        L::Poisson(p) => vec![&p.rate.expr],
        L::NegBinomial(nb) => vec![&nb.mean.expr, &nb.dispersion.expr],
        L::Normal(n) => vec![&n.mean.expr, &n.sd.expr],
        L::Binomial(b) => vec![&b.n, &b.p.expr],
        L::BetaBinomial(bb) => vec![&bb.n, &bb.alpha.expr, &bb.beta.expr],
        L::Beta(b) => vec![&b.mean.expr, &b.concentration.expr],
        L::Bernoulli(b) => vec![&b.p.expr],
        L::ZeroInflatedNegBinomial(zi) => vec![&zi.mean.expr, &zi.dispersion.expr, &zi.pi.expr],
    };
    let mut out = Vec::new();
    for e in args { walk(e, &mut out); }
    out
}

/// Load the per-observation auxiliary data for one stream (§3, §6.1): one
/// `Option<f64>` per aux column per row (`None` for `NA`). Row count matches
/// the scored column's. Returns, per row, the name→value list of PRESENT aux
/// values, plus a per-row `force_hole` flag set when any referenced aux is
/// missing (`NA`) — the scored cell then becomes a hole (present-together-or-
/// hole; `binomial(n = NA)` is unconstructible). An aux column declared but
/// whose header is absent is a located error (strict by-name, no fallback).
/// Each column is read in its own [`CellDomain`] (a binomial `n` is a count).
#[expect(clippy::disallowed_methods, reason = "reads a bound observation data file whose path its caller passes (`fit::observed_rows`): a stored fit problem's training tables, a filter run's or prediction's `DataTable` dependency, or a command that stores nothing from it (`simulate --design-from`, anchoring to the data)")]
pub fn load_stream_aux(
    path: &str,
    aux_cols: &[(String, CellDomain)],
    n_rows_expected: usize,
) -> Result<(Vec<Vec<(String, f64)>>, Vec<bool>), String> {
    if aux_cols.is_empty() {
        return Ok((vec![Vec::new(); n_rows_expected], vec![false; n_rows_expected]));
    }
    let content = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    // Parse each covariate column independently, reusing the strict by-name +
    // NA-hole parser: the column goes in both of the parser's slots and only
    // the value cells are kept. The slot is tagged `Covariate` so a missing
    // header is reported as the covariate it is rather than as a missing time
    // column (gh#886).
    let mut per_col: Vec<Vec<Option<f64>>> = Vec::with_capacity(aux_cols.len());
    for (col, domain) in aux_cols {
        let (_t, cells, _rows) =
            parse_column_cells(&content, path, col, ColumnPurpose::Covariate, col, *domain)?;
        if cells.len() != n_rows_expected {
            return Err(format!(
                "aux column '{}' in '{}' has {} data rows but the scored column has {} \
                 — every column of a stream's file must have the same rows",
                col, path, cells.len(), n_rows_expected));
        }
        per_col.push(cells);
    }
    let mut aux = vec![Vec::new(); n_rows_expected];
    let mut force_hole = vec![false; n_rows_expected];
    for (ci, (col, _)) in aux_cols.iter().enumerate() {
        for r in 0..n_rows_expected {
            match per_col[ci][r] {
                Some(v) => aux[r].push((col.clone(), v)),
                None => force_hole[r] = true, // present-together-or-hole
            }
        }
    }
    Ok((aux, force_hole))
}

/// Shared back-half: convert the raw time column, run the distinct-substep +
/// off-grid checks, validate chronological order, and zip into `Observation`s.
fn finalize_observations(
    time_cells: &[&str],
    values: Vec<f64>,
    rows: Vec<usize>,
    opts: &TimeOpts,
) -> Result<Vec<Observation>, String> {
    let row_offset = rows.first().copied().unwrap_or(2);
    let times = convert_time_column(time_cells, opts, row_offset)?;
    let was_dated = crate::caltime_load::cells_are_dated(time_cells, opts);
    check_substeps_and_grid(&times, &rows, opts, was_dated)?;

    let observations: Vec<Observation> = times
        .iter()
        .zip(values.iter())
        .map(|(&time, &value)| Observation { time, value })
        .collect();

    // Validate chronological ordering (equal times OK — multi-stream observations)
    for i in 1..observations.len() {
        if observations[i].time < observations[i - 1].time {
            return Err(format!(
                "observations not in chronological order: t={} at row {} follows t={} at row {}",
                observations[i].time, rows[i], observations[i - 1].time, rows[i - 1]
            ));
        }
    }

    Ok(observations)
}

use std::io::Write;

/// Write ancestor-traced smoothing paths as a long-format TSV.
/// Schema matches `camdl simulate --replicates N` for pipeline reuse:
/// columns `path`, `time`, and one column per integer compartment.
/// Each `path ∈ 1..=N` is an equally-weighted sample from the
/// smoothing distribution; no log_weight column needed.
fn paths_bytes(
    paths: &[sim::inference::ancestor_trace::SampledPath],
    model: &ir::Model,
) -> Vec<u8> {
    let mut f: Vec<u8> = Vec::new();

    write!(f, "path\ttime").unwrap();
    let comp_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str())
        .collect();
    for name in &comp_names {
        write!(f, "\t{}", name).unwrap();
    }
    // gh#48: emit one column per observation stream alongside the
    // compartment columns. Carries the model's declared projection
    // (incidence/prevalence/arithmetic) so downstream tooling doesn't
    // have to reconstruct it by finite-differencing compartment
    // counts — which silently breaks under event + balance
    // interactions (the constrained compartment becomes an
    // unfaithful tracker of its own dynamics, e.g. ΔR records fake
    // flows on cohort-event days).
    //
    // Column source: `SampledPath.projections[t][stream]` (= `mean()`
    // from the obs model at record time, walked along the ancestor
    // chain in lockstep with states). Skipped entirely when the obs
    // model's `mean()` returns empty (the trait default).
    let stream_names: &[String] = paths.first()
        .map(|p| p.stream_names.as_slice())
        .unwrap_or(&[]);
    let has_projections = !stream_names.is_empty()
        && paths.iter().all(|p| !p.projections.is_empty());
    if has_projections {
        for name in stream_names {
            write!(f, "\t{}", name).unwrap();
        }
    }
    writeln!(f).unwrap();

    for (i, p) in paths.iter().enumerate() {
        for (t_idx, &obs_t) in p.obs_times.iter().enumerate() {
            write!(f, "{}\t{}", i + 1, obs_t).unwrap();
            // Only the first n_comp_names columns of the state are
            // integer compartments; the PF records all state counts,
            // but we present only the public compartments.
            for k in 0..comp_names.len() {
                write!(f, "\t{}", p.states[t_idx][k]).unwrap();
            }
            if has_projections {
                let row_projs = &p.projections[t_idx];
                for s in 0..stream_names.len() {
                    if let Some(&v) = row_projs.get(s) {
                        write!(f, "\t{}", v).unwrap();
                    } else {
                        write!(f, "\tNaN").unwrap();
                    }
                }
            }
            writeln!(f).unwrap();
        }
    }
    f
}

/// Write filtering marginals as a long-format TSV. Schema:
/// `time`, `particle`, one column per integer compartment, and
/// `log_weight`. `particle` is an in-step index only — it is NOT
/// stable across `time`, and joining particles across `time` by
/// index is NOT a sample path.
fn filtering_bytes(
    trace: &sim::inference::ancestor_trace::AncestorTrace,
    model: &ir::Model,
) -> Vec<u8> {
    let mut f: Vec<u8> = Vec::new();

    write!(f, "time\tparticle").unwrap();
    let comp_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str())
        .collect();
    for name in &comp_names {
        write!(f, "\t{}", name).unwrap();
    }
    writeln!(f, "\tlog_weight").unwrap();

    for (t_idx, &obs_t) in trace.obs_times.iter().enumerate() {
        for (i, state) in trace.states[t_idx].iter().enumerate() {
            write!(f, "{}\t{}", obs_t, i + 1).unwrap();
            for k in 0..comp_names.len() {
                write!(f, "\t{}", state[k]).unwrap();
            }
            writeln!(f, "\t{:.6}", trace.log_weights[t_idx][i]).unwrap();
        }
    }
    f
}

/// `--save-filtered-streams`: one row per stop of the observation axis,
/// stream scheduled there, and phase (`predictive`, then `filtered`), holding
/// the weighted mean and quantiles over the swarm of the stream's conditional
/// mean `E[y_t | x_t, θ]`, which the `quantity` column labels `expectation`.
/// `time` is the stop's model time, the `--trace` column of the same name. A
/// summary with no weight behind it prints `NA`.
fn filtered_streams_bytes(
    steps: &[sim::inference::particle_filter::StreamMeanStep],
    observations: &[Observation],
    stream_names: &[String],
) -> Vec<u8> {
    let cell = |v: f64| if v.is_finite() { format!("{v}") } else { "NA".to_string() };
    let mut w: Vec<u8> = Vec::new();
    let quantity = crate::quantile::Quantity::Expectation.as_str();
    writeln!(w, "time\tstream\tphase\tquantity\tmean\tq05\tq25\tq50\tq75\tq95").unwrap();
    for step in steps {
        for (phase, s) in [("predictive", &step.predictive), ("filtered", &step.filtered)] {
            writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                observations[step.obs_idx].time, stream_names[step.stream], phase, quantity,
                cell(s.mean), cell(s.q05), cell(s.q25), cell(s.q50), cell(s.q75), cell(s.q95)).unwrap();
        }
    }
    w
}

/// `--trace`: per observation, the time, the log-likelihood increment and
/// the ESS, with the one-step predictive quantiles of the observation and the
/// state when the filter recorded them, and the observed value (summed over
/// the streams present at the step; `NA` where none is).
fn trace_bytes(
    result: &sim::inference::particle_filter::PFilterResult,
    observations: &[Observation],
    obs_model: &MultiStreamObsModel,
) -> Vec<u8> {
    let mut w: Vec<u8> = Vec::new();
    let observed = obs_model.joint_observed();
    let per_stream_view = obs_model.per_stream_observed();
    let observed_cell = |i: usize| -> String {
        if per_stream_view[i].iter().any(|v| v.is_finite()) {
            format!("{}", observed[i])
        } else {
            "NA".to_string()
        }
    };
    if let Some(ref preds) = result.predictions {
        writeln!(w, "time\tll_increment\tESS\tobs_mean\tobs_q05\tobs_q50\tobs_q95\tstate_mean\tstate_q05\tstate_q50\tstate_q95\tobserved").unwrap();
        for (i, obs) in observations.iter().enumerate() {
            let p = &preds[i];
            writeln!(w, "{}\t{:.4}\t{:.1}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
                obs.time, result.ll_increments[i], result.ess_trace[i],
                p.obs_mean, p.obs_q05, p.obs_q50, p.obs_q95,
                p.state_mean, p.state_q05, p.state_q50, p.state_q95,
                observed_cell(i)).unwrap();
        }
    } else {
        writeln!(w, "time\tll_increment\tESS\tobserved").unwrap();
        for (i, obs) in observations.iter().enumerate() {
            writeln!(w, "{}\t{:.4}\t{:.1}\t{:.0}",
                obs.time, result.ll_increments[i], result.ess_trace[i],
                observed_cell(i)).unwrap();
        }
    }
    w
}

/// `--pf-health`: per observation, the ESS, its fraction of the swarm and
/// the log-weight variance τ² (Snyder et al. 2008).
fn pf_health_bytes(
    result: &sim::inference::particle_filter::PFilterResult,
    observations: &[Observation],
    n_particles: usize,
) -> Vec<u8> {
    let mut w: Vec<u8> = Vec::new();
    let n = n_particles as f64;
    writeln!(w, "time\tESS\tESS_frac\ttau2").unwrap();
    for i in 0..result.ess_trace.len() {
        let t = observations.get(i).map(|o| o.time).unwrap_or(i as f64);
        writeln!(w, "{}\t{:.2}\t{:.4}\t{:.4}",
            t, result.ess_trace[i], result.ess_trace[i] / n, result.logw_var_trace[i]).unwrap();
    }
    w
}

/// Format a Snyder implied-particle-count estimate `exp(τ²/2)` for reading.
///
/// The quantity spans the whole float range: it is most useful exactly where
/// it is astronomically large, because that is the case where the answer is
/// "no affordable N will do — change the observation model" rather than
/// "buy more particles". A raw integer defeats that reading — gh#509 reported
/// a 48-digit one — so anything at or above a million prints as one
/// significant decimal in scientific notation, where the exponent is the part
/// that carries the decision. Below a million the plain integer stays, since
/// there it is directly comparable to a `--particles` value.
fn fmt_implied_n(x: f64) -> String {
    if !x.is_finite() {
        return format!("{}", x);
    }
    if x < 1e6 {
        format!("{:.0}", x)
    } else {
        format!("{:.1e}", x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caltime_load::TimeFormat;

    // ── pf-health implied-N formatting (gh#509) ─────────────────────────
    #[test]
    fn implied_n_switches_to_scientific_above_a_million() {
        // Below the threshold the plain integer is what you want: it is
        // directly comparable to a `--particles` value you might actually
        // type.
        assert_eq!(fmt_implied_n(1.0), "1");
        assert_eq!(fmt_implied_n(999_999.0), "999999");
        // At and above it, only the exponent is read. gh#509's report had
        // exp(218.90/2) print as a 48-digit integer, where the difference
        // between "expensive" (1e12) and "impossible" (1e47) took a moment
        // of counting digits to see.
        assert_eq!(fmt_implied_n(2_536_463.0), "2.5e6");
        assert_eq!(fmt_implied_n((0.5 * 218.90_f64).exp()), "3.4e47");
        // τ² large enough to overflow the exponential must not print as a
        // bare "inf" with no units of meaning — but it must not panic or
        // print garbage either.
        assert_eq!(fmt_implied_n(f64::INFINITY), "inf");
    }

    // ── test data files ─────────────────────────────────────────────────

    fn write_temp_tsv(name: &str, content: &str) -> String {
        let path = std::env::temp_dir().join(format!("camdl_test_{}.tsv", name));
        std::fs::write(&path, content).unwrap();
        path.to_str().unwrap().to_string()
    }

    // ── gh#886: a column-lookup diagnostic names the role it looked for ──

    /// A column the likelihood reads — a binomial denominator `n = tested`,
    /// declared `tested : count` — is not the time axis. When the file omits
    /// it, the message must say so and point at `columns { }`; reporting it as
    /// a missing `time : time` column sends the modeller to look at a time
    /// column that is fine, and away from the denominator that is not.
    #[test]
    fn a_missing_covariate_column_is_not_reported_as_a_missing_time_column() {
        let path = write_temp_tsv(
            "gh886_missing_covariate", "time\tpositives\n0\t3\n1\t5\n");
        let e = load_stream_aux(&path, &[("tested".to_string(), CellDomain::Real)], 2).unwrap_err();
        assert!(e.contains("covariate column 'tested'"),
            "the message names the column and what kind of column it is: {e}");
        assert!(e.contains("`columns { }`"),
            "and the declaration that governs it: {e}");
        assert!(!e.contains("time column") && !e.contains("`time : time`"),
            "and says nothing about the time column, which is present and fine: {e}");
    }

    /// The same, for a per-row window boundary: `window_start` is a temporal
    /// column but not the `: time` axis, and the fix is to add the boundary
    /// column, not to rename a time column.
    #[test]
    fn a_missing_window_boundary_is_not_reported_as_a_missing_time_column() {
        let path = write_temp_tsv(
            "gh886_missing_boundary", "window_stop\tcases\n1\t3\n2\t5\n");
        let content = std::fs::read_to_string(&path).unwrap();
        let e = read_boundary_column(&content, &path, "window_start").unwrap_err();
        assert!(e.contains("window boundary column 'window_start'"),
            "the message names the boundary column as one: {e}");
        assert!(!e.contains("`time : time`"),
            "and does not prescribe the time-column rule: {e}");
    }

    fn numeric_opts() -> TimeOpts<'static> {
        TimeOpts {
            origin: None,
            time_unit: "days",
            dt: 1.0,
            t_start: 0.0,
            format: TimeFormat::Auto,
        }
    }

    /// A state stream reading a `time` column and a `cases` value column: no
    /// `covers`, so each row measures the instant of its time cell.
    fn state_stream() -> ir::observation::ObservationModel {
        let mut obs = daily_incidence_stream();
        obs.covers = None;
        obs.projection = ir::observation::Projection::CurrentPop("I".into());
        obs
    }

    /// The `(time, value)` of each row [`crate::fit::observed_rows::observed_rows`]
    /// reads from `path` for [`state_stream`]; a hole's value is `None`.
    fn read_state_stream(path: &str, opts: &TimeOpts) -> Result<Vec<(f64, Option<f64>)>, String> {
        let obs = state_stream();
        let rows = crate::fit::observed_rows::observed_rows(path, &obs, &[&obs], opts)?;
        Ok(rows.iter().map(|r| (r.measures.end(), r.cell.map(|c| match c {
            sim::inference::ObsCell::Scalar(v) | sim::inference::ObsCell::HeldOut(v) => v,
        }))).collect())
    }

    // ── gh#833: a gap under a UNIFORM covers form is a missing row ───────

    /// An incidence stream declaring `covers = day(time)` in its lowered form
    /// (open at the label, one day wide).
    fn daily_incidence_stream() -> ir::observation::ObservationModel {
        use ir::observation::*;
        ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "cases".into(),
            emit_schedule: None,
            stratum: vec![],
            covers: Some(Covers::From { offset: 0.0, span: 1.0 }),
            projection: Projection::CumulativeFlow("infection".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    #[test]
    fn a_missing_row_under_a_uniform_covers_form_is_refused_naming_both_boundaries() {
        // Proposal Testing item 1. Labels 3, 4, 6 under `day(time)` imply
        // windows [3,4), [4,5), [6,7): row 5 is missing, so (5,6) is covered by
        // neither. Under a uniform form that can only be a mistake — before
        // this arc it silently widened the next bin to (4,6].
        let obs = daily_incidence_stream();
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let err = stream_times_for(&obs, &proj, &[3.0, 4.0, 6.0], "cases.tsv", &numeric_opts())
            .expect_err("a missing row under a uniform form must be refused");
        assert!(err.contains("[4, 5)") && err.contains("[6, 7)"),
            "must name the two periods either side of the gap: {err}");
        assert!(err.contains("5 to 6"), "must name the uncovered span: {err}");
        assert!(err.contains("NA") && err.contains("window_start"),
            "must point at both remedies — an NA row, or the per-row form: {err}");
    }

    #[test]
    fn contiguous_labels_under_a_uniform_covers_form_build_shifted_periods() {
        let obs = daily_incidence_stream();
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let times = stream_times_for(&obs, &proj, &[3.0, 4.0, 5.0], "cases.tsv", &numeric_opts())
            .expect("contiguous labels are fine");
        let periods = times.periods().expect("a declared stream carries periods");
        assert_eq!(periods.len(), 3);
        assert_eq!((periods[0].start(), periods[0].stop()), (3.0, 4.0));
        assert_eq!((periods[2].start(), periods[2].stop()), (5.0, 6.0),
            "day(D) covers [D, D+1): the last row closes one day AFTER its label");
    }

    #[test]
    fn a_first_period_opening_before_t_start_is_refused() {
        let obs = daily_incidence_stream();
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let mut opts = numeric_opts();
        opts.t_start = 4.0;
        let err = stream_times_for(&obs, &proj, &[3.0, 4.0], "cases.tsv", &opts)
            .expect_err("a period opening before the run starts must be refused");
        assert!(err.contains("opens at 3") && err.contains("t_start = 4"),
            "must name both the period start and t_start: {err}");
        // Opening exactly AT t_start is fine — the bin is already empty there.
        opts.t_start = 3.0;
        stream_times_for(&obs, &proj, &[3.0, 4.0], "cases.tsv", &opts)
            .expect("a period opening at t_start is legal");
    }

    // ── gh#842: an alignment error quotes times the way the file states them ──

    /// A dated daily-incidence file whose labels are `dates`, written to a
    /// temp path. Returns `(path, label_times)`.
    fn dated_daily_file(tag: &str, dates: &[&str], opts: &TimeOpts) -> (String, Vec<f64>) {
        let mut body = String::from("time\tcases\n");
        for (i, d) in dates.iter().enumerate() {
            body.push_str(&format!("{d}\t{}\n", i + 1));
        }
        (write_temp_tsv(tag, &body), as_internal(dates, opts))
    }

    /// [`stream_times_of_rows`] over the rows the row reader reads from `path`.
    fn stream_times_of_file(
        obs: &ir::observation::ObservationModel,
        path: &str,
        opts: &TimeOpts,
    ) -> Result<sim::inference::StreamTimes, String> {
        let rows = crate::fit::observed_rows::observed_rows(path, obs, &[obs], opts)?;
        stream_times_of_rows(obs, &rows, path, opts)
    }

    /// A period that opens before the run does is refused, and on a dated file
    /// the refusal quotes the dates. A modeller whose every other time is a
    /// calendar date should not have to convert `3` and `4` back by hand to
    /// find which row the message means.
    #[test]
    fn a_period_before_t_start_on_a_dated_file_is_reported_in_dates() {
        let obs = daily_incidence_stream();
        let mut opts = dated_opts();
        let (path, labels) =
            dated_daily_file("gh842_before_origin", &["2026-07-04", "2026-07-05"], &opts);
        // The run starts the day after the first row's window opens.
        opts.t_start = labels[1];
        let err = stream_times_of_file(&obs, &path, &opts)
            .expect_err("a period opening before the run starts must be refused");
        assert!(err.contains("opens at 2026-07-04") && err.contains("2026-07-05"),
            "both the offending boundary and the run's start are dates: {err}");
        assert!(!err.contains("opens at 3"),
            "and neither is quoted as bare model time: {err}");
        assert!(err.contains("`origin`"),
            "the fix names the anchor a dated model would move: {err}");
    }

    /// The missing-row refusal under a uniform `covers` form, likewise: the
    /// four boundaries and the uncovered span are the file's own dates.
    #[test]
    fn a_gap_on_a_dated_file_names_the_dates_either_side() {
        let obs = daily_incidence_stream();
        let opts = dated_opts();
        // Labels 4, 5 and 7 July under `day(time)`: the 6th is missing, so
        // [6 Jul, 7 Jul) is covered by neither of the rows either side.
        let (path, _labels) = dated_daily_file(
            "gh842_gap", &["2026-07-04", "2026-07-05", "2026-07-07"], &opts);
        let err = stream_times_of_file(&obs, &path, &opts)
            .expect_err("a missing row under a uniform form must be refused");
        assert!(err.contains("[2026-07-05, 2026-07-06)")
                && err.contains("[2026-07-07, 2026-07-08)"),
            "the two periods either side of the gap are dated: {err}");
        assert!(err.contains("2026-07-06 to 2026-07-07"),
            "and so is the span neither covers: {err}");
        assert!(!err.contains("[4, 5)"), "no bare model time in the message: {err}");
    }

    fn incidence_stream_with(covers: ir::observation::Covers) -> ir::observation::ObservationModel {
        let mut obs = daily_incidence_stream();
        obs.covers = Some(covers);
        obs
    }

    fn period_pairs(times: &sim::inference::StreamTimes) -> Vec<(f64, f64)> {
        times.periods().expect("a declared stream carries periods")
            .iter().map(|p| (p.start(), p.stop())).collect()
    }

    /// The two named forms through the loader: `starting_on(time, 7 'days)`
    /// opens at the label and `ending_on(time, 7 'days)` closes the day after
    /// it (the labelled day included), each on contiguous weekly labels.
    #[test]
    fn starting_on_and_ending_on_build_the_periods_their_names_say() {
        use ir::observation::Covers;
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);

        let starting = incidence_stream_with(Covers::From { offset: 0.0, span: 7.0 });
        let times = stream_times_for(&starting, &proj, &[7.0, 14.0, 21.0], "cases.tsv", &numeric_opts())
            .expect("contiguous weekly labels");
        assert_eq!(period_pairs(&times), vec![(7.0, 14.0), (14.0, 21.0), (21.0, 28.0)]);
        assert_eq!(times.closes(), vec![14.0, 21.0, 28.0], "scored a week after each label");

        // ending_on(D, 7d) = [D − 6, D + 1): the 11th is in a row labelled 11.
        let ending = incidence_stream_with(Covers::Until { offset: 1.0, span: 7.0 });
        let times = stream_times_for(&ending, &proj, &[7.0, 14.0, 21.0], "cases.tsv", &numeric_opts())
            .expect("contiguous weekly labels");
        assert_eq!(period_pairs(&times), vec![(1.0, 8.0), (8.0, 15.0), (15.0, 22.0)]);
        assert_eq!(times.closes(), vec![8.0, 15.0, 22.0], "scored the day after each label");
    }

    /// `day(time)` on a `'weeks` axis: the span is `1/7`, so a row's stop
    /// (`label + 1/7`) and the next row's start (the next label) are the same
    /// boundary computed two ways, a few ulps apart. That is not a gap.
    #[test]
    fn a_daily_form_off_a_whole_unit_axis_is_contiguous_not_gapped() {
        use ir::observation::Covers;
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let one_day = 1.0 / 7.0;
        let daily_on_weeks = incidence_stream_with(Covers::From { offset: 0.0, span: one_day });
        let labels: Vec<f64> = (0..40).map(|k| 1.0 + f64::from(k) * one_day).collect();
        // Non-vacuous: the arithmetic really does disagree somewhere in the list.
        assert!(labels.windows(2).any(|w| w[0] + one_day != w[1]),
            "the fixture must contain at least one pair the exact comparison would split");
        let times = stream_times_for(&daily_on_weeks, &proj, &labels, "cases.tsv", &numeric_opts())
            .expect("daily rows on a weeks axis tile the span");
        let ps = period_pairs(&times);
        assert_eq!(ps.len(), 40);
        for w in ps.windows(2) {
            assert!((w[1].0 - w[0].1).abs() <= sim::inference::multi_stream_obs::BOUNDARY_EPS,
                "consecutive periods share a boundary: {:?} then {:?}", w[0], w[1]);
        }
    }

    // ── gh#833: a per-row window boundary is a time, dates included ─────

    /// The same incidence stream stating each row's window in the file — a
    /// `window_start`/`window_stop` pair in place of the `: time` column,
    /// which is what `covers = window_columns` means.
    fn windowed_incidence_stream() -> ir::observation::ObservationModel {
        use ir::observation::*;
        let mut obs = daily_incidence_stream();
        obs.columns = vec![
            ObsColumn { name: "window_start".into(), role: ColumnRole::WindowStart },
            ObsColumn { name: "window_stop".into(), role: ColumnRole::WindowStop },
            ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ];
        obs.covers = Some(Covers::WindowColumns);
        obs
    }

    /// A model anchored to a calendar, so a dated cell has an origin to
    /// convert against.
    fn dated_opts() -> TimeOpts<'static> {
        TimeOpts { origin: Some("2026-07-01"), ..numeric_opts() }
    }

    fn as_internal(dates: &[&str], opts: &TimeOpts) -> Vec<f64> {
        dates.iter()
            .map(|d| ir::caltime::date_to_internal(opts.origin.unwrap(), d, opts.time_unit).unwrap())
            .collect()
    }

    /// A windowed file writes its boundaries the way the data spec prints them
    /// — ISO dates in both columns, one row wider than a day because
    /// publication slipped. Each bound period must be that row's own two
    /// dates, converted through the model's `origin` and `time_unit`.
    ///
    /// No such file loaded at all before this: the start column went through
    /// the *value* parser on its way to the time conversion, so `2026-07-11`
    /// was refused as an unparseable `f64` before the conversion that handles
    /// dates ever saw it. Every other window test writes numeric boundaries,
    /// which parse as `f64` and so never reach that parse.
    #[test]
    fn a_dated_window_column_binds_the_periods_its_dates_name() {
        let path = write_temp_tsv("dated_windows", concat!(
            "window_start\twindow_stop\tcases\tdays_covered\n",
            "2026-07-11\t2026-07-12\t0\t1\n",
            "2026-07-12\t2026-07-13\t4\t1\n",
            "2026-07-13\t2026-07-18\t61\t5\n"));
        let obs = windowed_incidence_stream();
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let opts = dated_opts();

        // The stops arrive already converted — they are the stream's time
        // column, so the caller has put them through the time path.
        let stops = as_internal(&["2026-07-12", "2026-07-13", "2026-07-18"], &opts);
        let times = stream_times_for(&obs, &proj, &stops, &path, &opts)
            .expect("a windowed file with ISO dates must load");

        let starts = as_internal(&["2026-07-11", "2026-07-12", "2026-07-13"], &opts);
        let expected: Vec<(f64, f64)> =
            starts.iter().zip(&stops).map(|(&a, &b)| (a, b)).collect();
        assert_eq!(period_pairs(&times), expected,
            "each row covers its own two dates, converted through origin + time_unit");
        // Pin the arithmetic as well, so a conversion that moved could not
        // satisfy both sides with the same wrong numbers: origin 2026-07-01 on
        // a days axis puts 11 July at day 10, and the last row spans five days.
        assert_eq!(period_pairs(&times), vec![(10.0, 11.0), (11.0, 12.0), (12.0, 17.0)],
            "11 July is day 10 from a 1 July origin, and 13→18 July is five days wide");
    }

    /// `NA` says a row's observed value is missing — the row keeps its window
    /// and is not scored. A row whose window has no start states no period at
    /// all, so it is refused by row rather than skipped or defaulted.
    #[test]
    fn an_na_window_start_is_refused_naming_the_row() {
        let path = write_temp_tsv("na_window_start", concat!(
            "window_start\twindow_stop\tcases\n",
            "2026-07-11\t2026-07-12\t0\n",
            "NA\t2026-07-13\t4\n"));
        let obs = windowed_incidence_stream();
        let proj = sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]);
        let opts = dated_opts();
        let stops = as_internal(&["2026-07-12", "2026-07-13"], &opts);
        let err = stream_times_for(&obs, &proj, &stops, &path, &opts)
            .expect_err("a window with no start has no reading");
        assert!(err.contains("line 3"), "must name the row: {err}");
        assert!(err.contains("window_start") && err.contains("NA"),
            "must name the column and the token: {err}");
    }

    #[test]
    fn load_data_rejects_out_of_order() {
        let path = write_temp_tsv("out_of_order", "time\tcases\n7\t10\n14\t20\n10\t15\n21\t30\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_err(), "should reject out-of-order times");
        let err = result.err().unwrap();
        assert!(err.contains("not in chronological order"), "error message: {}", err);
        assert!(err.contains("t=10"), "should mention the offending time: {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_accepts_equal_times() {
        // Equal times are valid (multi-stream observations at same time point)
        let path = write_temp_tsv("equal_times", "time\tcases\n7\t10\n7\t5\n14\t20\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_ok(), "equal times should be accepted: {:?}", result.err());
        let obs = result.unwrap();
        assert_eq!(obs.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_accepts_sorted() {
        let path = write_temp_tsv("sorted", "time\tcases\n7\t10\n14\t20\n21\t30\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
        std::fs::remove_file(&path).ok();
    }

    // ── G1: by-name binding is strict — no positional fallback ──────────
    //
    // A stream requested by name against a file whose column does not match
    // (typo / wrong case / renamed header) must ERROR with a located
    // message, never silently bind a value column by position. Pre-fix, the
    // inner 2-column fallback (`if cols.len() == 2 { Some(1) }`) made a
    // mis-cased single-value file load against column 1 — a wrong answer
    // with exit 0.

    #[test]
    fn row_reader_rejects_miscased_name_in_2col_file() {
        // Header column is `Cases` (capital C); model asks for `cases`.
        // Pre-fix: 2-column fallback binds column 1 and loads. Post-fix:
        // located error naming the requested column + available headers.
        let path = write_temp_tsv("miscased_2col", "time\tCases\n7\t10\n14\t20\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_err(),
            "a mis-cased column name must NOT silently bind by position; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"),
            "error must name the requested stream/column 'cases': {}", err);
        assert!(err.contains("Cases"),
            "error must list the headers actually present (incl. 'Cases'): {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_reader_rejects_renamed_name_in_wide_file() {
        // Multi-column file, requested name absent. (This path already
        // errored pre-fix, but pin the located-message quality.)
        let path = write_temp_tsv("renamed_wide",
            "time\tcase_count\tdeaths\n7\t10\t1\n14\t20\t2\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_err(), "absent column name must error");
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name requested column: {}", err);
        assert!(err.contains("case_count") && err.contains("deaths"),
            "error must list available headers: {}", err);
        std::fs::remove_file(&path).ok();
    }

    /// gh#344: a leading `#` provenance comment must be SKIPPED, not treated as
    /// the header — consistent with the `interpolated` forcing reader and polars
    /// `comment_prefix`. Pre-fix the comment line was taken as the header, so the
    /// declared `time` column wasn't found and the fit failed with a confusing
    /// "Headers present: [# ...]" error.
    #[test]
    fn row_reader_skips_leading_comment_header() {
        let path = write_temp_tsv("comment_header",
            "# Crude all-age prevalence, village 408\ntime\tcases\n7\t10\n14\t20\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_ok(),
            "a leading `#` comment must be skipped, not treated as the header; got {:?}",
            result.as_ref().err());
        std::fs::remove_file(&path).ok();
    }

    // ── G1/NaN: non-finite observation values are rejected at load ──────
    //
    // `"NaN".parse::<f64>()` returns `Ok(NaN)` and `"inf"`/`"Infinity"`
    // return `Ok(±inf)`. Pre-fix these flowed straight into the likelihood.
    // Post-fix: located error (file path implied by caller, column, row).

    #[test]
    fn row_reader_rejects_nan_value() {
        let path = write_temp_tsv("nan_value", "time\tcases\n7\t10\n14\tNaN\n21\t30\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_err(),
            "a NaN observation value must be rejected at load; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name the column: {}", err);
        assert!(err.contains('3') || err.contains("14"),
            "error must locate the offending row/time (line 3, t=14): {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_reader_rejects_inf_value() {
        let path = write_temp_tsv("inf_value", "time\tcases\n7\t10\n14\tinf\n21\t30\n");
        let result = read_state_stream(&path, &numeric_opts());
        assert!(result.is_err(),
            "an infinite observation value must be rejected at load; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name the column: {}", err);
        std::fs::remove_file(&path).ok();
    }

    // ── Negative control: well-formed input is unchanged ────────────────
    //
    // Guards against a vacuous test: a matching column name loads the
    // SAME values as the 2-column loader on the equivalent file, and the
    // by-name path binds the requested column (not column 1) in a wide
    // file. This is the happy path that must remain byte-identical.

    #[test]
    fn row_reader_happy_path_loads_named_column() {
        // Wide file: `cases` is column 2 (not column 1). By-name binding
        // must pick column 2's values, not deaths in column 1.
        let path = write_temp_tsv("happy_wide",
            "time\tdeaths\tcases\n7\t1\t10\n14\t2\t20\n21\t3\t30\n");
        let obs = read_state_stream(&path, &numeric_opts())
            .expect("well-formed named column must load");
        assert_eq!(obs.len(), 3);
        assert_eq!(obs, vec![(7.0, Some(10.0)), (14.0, Some(20.0)), (21.0, Some(30.0))]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_reader_happy_path_2col_named_file() {
        // The happy single-stream case: a `time\tcases` file binds the
        // `cases` column *because it is named* `cases` (which happens to
        // be column 1), not by position. Values must round-trip exactly —
        // this is the legacy single-stream schema, unchanged by the
        // fallback deletion.
        let path = write_temp_tsv("happy_2col", "time\tcases\n7\t10\n14\t20\n21\t30\n");
        let obs = read_state_stream(&path, &numeric_opts())
            .expect("named single-stream load");
        assert_eq!(obs.len(), 3);
        assert_eq!(obs, vec![(7.0, Some(10.0)), (14.0, Some(20.0)), (21.0, Some(30.0))]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_dated_matches_numeric() {
        // §9.4 byte-identity: dated cells against an origin produce the same
        // observation vector as the equivalent numeric day-numbers.
        let dated = write_temp_tsv(
            "dated",
            "time\tcases\n2020-03-01\t10\n2020-03-08\t20\n2020-03-15\t30\n",
        );
        let numeric = write_temp_tsv("dated_num", "time\tcases\n0\t10\n7\t20\n14\t30\n");
        let mut o = numeric_opts();
        o.origin = Some("2020-03-01");
        let from_dates = read_state_stream(&dated, &o).unwrap();
        let from_nums = read_state_stream(&numeric, &numeric_opts()).unwrap();
        assert_eq!(from_dates, from_nums);
        std::fs::remove_file(&dated).ok();
        std::fs::remove_file(&numeric).ok();
    }

    // ── Sparse/holes: `NA` loads as a hole on the cells path ────────────
    //
    // The missing-value token `NA` becomes a hole (`None`) whose TIME stays
    // in the grid (the row is kept). NaN/inf are still rejected as garbage.

    #[test]
    fn cells_loader_treats_na_as_a_hole_with_time_retained() {
        use sim::inference::ObsCell;
        // Three weekly rows; the middle one is NA (a hole).
        let path = write_temp_tsv("na_hole", "time\tcases\n7\t10\n14\tNA\n21\t30\n");
        let obs = state_stream();
        let rows = crate::fit::observed_rows::observed_rows(&path, &obs, &[&obs], &numeric_opts())
            .expect("NA must load as a hole, not error");
        let times: Vec<f64> = rows.iter().map(|r| r.measures.end()).collect();

        // All three grid times are retained — the hole's time stays.
        assert_eq!(times, vec![7.0, 14.0, 21.0],
            "the hole row's TIME must stay in the grid; got {:?}", times);
        // The middle cell is a hole; the others are observed scalars.
        assert_eq!(rows[0].cell, Some(ObsCell::Scalar(10.0)));
        assert_eq!(rows[1].cell, None, "the NA cell must be a hole (None)");
        assert_eq!(rows[2].cell, Some(ObsCell::Scalar(30.0)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cells_loader_still_rejects_nan_and_inf() {
        // A hole is `NA`, not a non-finite number — those remain garbage.
        for (name, body) in [
            ("cells_nan", "time\tcases\n7\t10\n14\tNaN\n21\t30\n"),
            ("cells_inf", "time\tcases\n7\t10\n14\tinf\n21\t30\n"),
        ] {
            let path = write_temp_tsv(name, body);
            let result = read_state_stream(&path, &numeric_opts());
            assert!(result.is_err(),
                "non-finite values must still be rejected on the cells path ({name})");
            let err = result.err().unwrap();
            assert!(err.contains("cases"), "error must name the column: {}", err);
            std::fs::remove_file(&path).ok();
        }
    }

    // ── §4.2 long-form by-name level matching ───────────────────────────
    //
    // Build one expanded leaf of a stratified `cases[p in patch]` family.
    // `level` is this leaf's patch level; the sibling list defines the valid
    // level set (the model's `patch` levels).
    fn long_form_leaf(level: &str) -> ir::observation::ObservationModel {
        use ir::observation::{
            ColumnRole, Likelihood, ObsColumn, ObservationModel, PoissonLikelihood,
            Projection, StratumKey,
        };
        use ir::parameter::ParamKind;
        ObservationModel {
            name:   format!("cases_{level}"),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(),  role: ColumnRole::Time },
                ObsColumn { name: "patch".into(), role: ColumnRole::Dim("patch".into()) },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value(ParamKind::Count) },
            ],
            scored: "cases".into(),
            emit_schedule: None,
            stratum: vec![StratumKey { dim: "patch".into(), level: level.into() }],
            covers: None,
            projection: Projection::CurrentPop(format!("I_{level}")),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    /// The times and cells [`crate::fit::observed_rows::observed_rows`] reads
    /// for the leaf `obs` of a long-form file.
    fn read_leaf(
        path: &str,
        obs: &ir::observation::ObservationModel,
        siblings: &[&ir::observation::ObservationModel],
    ) -> Result<(Vec<f64>, Vec<Option<sim::inference::ObsCell>>), String> {
        let rows = crate::fit::observed_rows::observed_rows(path, obs, siblings, &numeric_opts())?;
        Ok(rows.iter().map(|r| (r.measures.end(), r.cell)).unzip())
    }

    #[test]
    fn long_form_routes_rows_to_strata_by_name() {
        use sim::inference::ObsCell;
        // A 2-level patch family; long-form file with interleaved rows. Each
        // leaf must carry only its own stratum's values, by name (NOT by
        // position — the rows alternate p1/p2).
        let p1 = long_form_leaf("p1");
        let p2 = long_form_leaf("p2");
        let siblings: Vec<&ir::observation::ObservationModel> = vec![&p1, &p2];
        let path = write_temp_tsv("lf_route",
            "time\tpatch\tcases\n7\tp1\t10\n7\tp2\t99\n14\tp1\t20\n14\tp2\t88\n");

        let (t1, c1) = read_leaf(&path, &p1, &siblings).expect("p1 leaf loads");
        let (t2, c2) = read_leaf(&path, &p2, &siblings).expect("p2 leaf loads");

        assert_eq!(t1, vec![7.0, 14.0], "union time axis");
        assert_eq!(t2, vec![7.0, 14.0], "union time axis shared across leaves");
        // p1 gets 10, 20 — NOT 99, 88 (the sibling's column).
        assert_eq!(c1, vec![Some(ObsCell::Scalar(10.0)), Some(ObsCell::Scalar(20.0))],
            "p1 leaf must carry p1's rows, routed by name");
        assert_eq!(c2, vec![Some(ObsCell::Scalar(99.0)), Some(ObsCell::Scalar(88.0))],
            "p2 leaf must carry p2's rows, routed by name");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn long_form_unknown_level_is_rejected() {
        // A row carries patch=p3, which is absent from every leaf's stratum
        // (the model only has p1, p2). E281, located, listing the valid levels.
        let p1 = long_form_leaf("p1");
        let p2 = long_form_leaf("p2");
        let siblings: Vec<&ir::observation::ObservationModel> = vec![&p1, &p2];
        let path = write_temp_tsv("lf_unknown",
            "time\tpatch\tcases\n7\tp1\t10\n7\tp3\t5\n");

        let result = read_leaf(&path, &p1, &siblings);
        assert!(result.is_err(), "an unknown level must be a hard error");
        let err = result.err().unwrap();
        assert!(err.contains("E281"), "must be E281: {err}");
        assert!(err.contains("p3"), "must name the offending level p3: {err}");
        assert!(err.contains("patch"), "must name the dim column: {err}");
        assert!(err.contains("p1") && err.contains("p2"),
            "must list the valid level set [p1, p2]: {err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn long_form_absent_row_is_not_scheduled() {
        use sim::inference::ObsCell;
        // p1 has rows at {7,14,21}; p2 only at {7,21} (NO row at 14). p2 is not
        // OBSERVED at 14 — that is a SIBLING's time. Each leaf reports its OWN
        // schedule, so 14 is absent from p2's axis: `bind` marks it not-scheduled
        // for p2 (no score AND no reset). Routing p2 onto the full union with a
        // hole at 14 instead would reset p2's incidence accumulator at a sibling's
        // cadence — the ragged-cadence bin truncation this fixes (PR#218 review #1).
        let p1 = long_form_leaf("p1");
        let p2 = long_form_leaf("p2");
        let siblings: Vec<&ir::observation::ObservationModel> = vec![&p1, &p2];
        let path = write_temp_tsv("lf_absent",
            "time\tpatch\tcases\n\
             7\tp1\t10\n7\tp2\t1\n\
             14\tp1\t20\n\
             21\tp1\t30\n21\tp2\t3\n");

        let (t2, c2) = read_leaf(&path, &p2, &siblings).expect("p2 leaf loads");
        assert_eq!(t2, vec![7.0, 21.0],
            "p2's OWN schedule is {{7,21}} — 14 is a sibling's time, not p2's");
        assert_eq!(c2, vec![Some(ObsCell::Scalar(1.0)), Some(ObsCell::Scalar(3.0))]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn long_form_explicit_na_row_is_a_hole() {
        use sim::inference::ObsCell;
        // p2 HAS a row at 14 with value NA → 14 IS p2's own scheduled time, scored
        // as a hole (None — no term, NOT a false zero) and, for incidence,
        // resetting the bin there. The explicit-NA hole is the bookkeeping
        // distinction from an ABSENT row (not-scheduled, above): "we sampled and
        // got nothing usable" vs "we don't sample on this cadence".
        let p1 = long_form_leaf("p1");
        let p2 = long_form_leaf("p2");
        let siblings: Vec<&ir::observation::ObservationModel> = vec![&p1, &p2];
        let path = write_temp_tsv("lf_na_hole",
            "time\tpatch\tcases\n\
             7\tp1\t10\n7\tp2\t1\n\
             14\tp1\t20\n14\tp2\tNA\n\
             21\tp1\t30\n21\tp2\t3\n");

        let (t2, c2) = read_leaf(&path, &p2, &siblings).expect("p2 leaf loads");
        assert_eq!(t2, vec![7.0, 14.0, 21.0], "explicit NA row keeps 14 in p2's schedule");
        assert_eq!(c2, vec![Some(ObsCell::Scalar(1.0)), None, Some(ObsCell::Scalar(3.0))]);
        assert_ne!(c2[1], Some(ObsCell::Scalar(0.0)),
            "a coverage hole must be None, never an observed 0");
        std::fs::remove_file(&path).ok();
    }
}
