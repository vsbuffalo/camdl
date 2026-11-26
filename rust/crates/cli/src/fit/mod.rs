//! `camdl fit` — structured inference workflow.
//!
//! `camdl fit run FIT.toml [--seed N] [--starts SPEC] [--label "..."]
//! [--force]` runs the file's one `[method]` on its problem and stores the
//! result as a content-addressed leaf; `fit summary`, `camdl predict`, `fit
//! table`, `fit diff` and `fit new` read and derive from those. See
//! `docs/dev/proposals/2026-09-08-workflow-first-fit-config.md`.

/// gh#616: resolve a model's observation anchors against this fit's `[data]`
/// and re-emit the compiled IR with them substituted, returning the path every
/// downstream loader should use.
///
/// Returns `ir_path` UNCHANGED when the model declares no anchor — the common
/// case, and then not a byte is written or copied.
///
/// The envelope is re-emitted whole (not just the model) so the `#'`
/// documentation dictionary survives; a fit sidecar's parameter legend reads it
/// back off this path.
#[expect(clippy::disallowed_methods, reason = "reads the compiled IR `util::resolve_ir_path` produced from the fit's `[model]`; `fit run` and `profile` store the model rewritten from it as the `CompiledModel` the problem's identity names")]
pub(crate) fn resolve_anchors_into_temp_ir(
    out: Option<&crate::out::Out>, ir_path: &str,
    config: &config_v2::Problem,
) -> Result<String, String> {
    let src = std::fs::read_to_string(ir_path)
        .map_err(|e| format!("cannot read compiled IR {ir_path}: {e}"))?;
    let mut env = ir::envelope_from_str(&src)
        .map_err(|e| format!("IR load error from {ir_path}: {e}"))?;
    // The config's `[simulate] from` is resolved here too, with the anchors:
    // the start it names is written into the model every later load reads.
    if !crate::obs_anchor::fit_resolves(&env.model, config) {
        return Ok(ir_path.to_string());
    }
    let moved = crate::obs_anchor::resolve_for_fit(&mut env.model, config)
        .map_err(|e| format!("resolving this model's start and observation anchors from [data]: {e}"))?;
    if let Some(out) = out {
        crate::obs_anchor::report(out, &moved, &env.model);
    }

    // A fresh temp, never the input path: `resolve_ir_path` returns a
    // user-supplied `.ir.json` unchanged, and a command must not rewrite the
    // user's file. Persists for the process, like the compiled-IR temp itself.
    let out = std::env::temp_dir().join(format!(
        "camdl_anchored_{}_{}.ir.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let json = serde_json::to_string(&env)
        .map_err(|e| format!("re-emitting the resolved IR: {e}"))?;
    std::fs::write(&out, json)
        .map_err(|e| format!("writing the resolved IR to {}: {e}", out.display()))?;
    Ok(out.to_string_lossy().into_owned())
}

/// Bind the model at `ir_path` to the files the fit config's `[data.tables]`
/// names, and re-emit it to a fresh temp IR, so every downstream loader (the
/// runner, each sweep point, the synthetic generator, the archived copy) reads
/// one bound model and no later read of the files can disagree with the
/// first. The anchors' counterpart ([`resolve_anchors_into_temp_ir`]). A model
/// with no external input and a config binding none returns `ir_path`
/// unchanged; a binding the model does not declare, or an input left unbound,
/// is refused here, before anything runs.
#[expect(clippy::disallowed_methods, reason = "reads the compiled IR `util::resolve_ir_path` produced from the fit's `[model]`; `fit run` and `profile` store the model rewritten from it as the `CompiledModel` the problem's identity names")]
pub(crate) fn bind_tables_into_temp_ir(
    ir_path: &str,
    config: &config_v2::Problem,
) -> Result<(String, crate::external_inputs::BoundFiles), String> {
    let src = std::fs::read_to_string(ir_path)
        .map_err(|e| format!("cannot read compiled IR {ir_path}: {e}"))?;
    let mut env = ir::envelope_from_str(&src)
        .map_err(|e| format!("IR load error from {ir_path}: {e}"))?;
    let bound = crate::external_inputs::bind(&mut env.model, &config.external_bindings())
        .map_err(|e| e.to_string())?;
    if bound.is_empty() {
        return Ok((ir_path.to_string(), bound));
    }
    let out = std::env::temp_dir().join(format!(
        "camdl_bound_{}_{}.ir.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let json = serde_json::to_string(&env)
        .map_err(|e| format!("re-emitting the bound IR: {e}"))?;
    std::fs::write(&out, json)
        .map_err(|e| format!("writing the bound IR to {}: {e}", out.display()))?;
    Ok((out.to_string_lossy().into_owned(), bound))
}

pub mod coeff_guard;  // gh#342 P4: NUTS guard — param reaching a coefficient only via an init
pub mod config_v2;
pub(crate) mod run_config; // ResolvedFitConfig: fit run's one resolved configuration
pub mod loglik;  // gh#280: LoglikType — the single typed source for loglik class
pub mod state;
pub(crate) mod execution; // execution.json: one attempt's times, process and cost, outside the manifest
pub(crate) mod stream_bindings; // which leaves a fit's [data] binds, by source, and to which files
pub mod provenance;
pub(crate) mod observed_rows; // one leaf's data rows, each with the instant or period it measures
pub(crate) mod training_partition; // which rows trained: per stream a prefix and its row digest
pub(crate) mod forecast_plan; // what a fixed-origin forecast predicts: its origin, reach and targets
pub(crate) mod forecast_covariates; // the table forcings a forecast reads after its origin, and their statements
pub(crate) mod forecast_driver; // the fixed-origin forecast: one bootstrap filter per posterior draw
pub(crate) mod forecast_arms; // the scenario and sweep arms of a forecast, each differing only after the origin
pub(crate) mod forecast_contrast; // the difference of two arms' predictive means and its Monte Carlo error (D83)
pub(crate) mod forecast_output; // `camdl predict --mode forecast`: the plan, the driver, the artifact
pub(crate) mod fixed_point; // `camdl predict MODEL`: the point, its data and its numerics, stored before the prediction
pub(crate) mod forecast_score; // the marginal and joint log scores of a forecast against held-out values (D79)
pub(crate) mod scoring; // `camdl score`'s rows: the cells a recomputed prediction hands back, and the rules
pub mod runner;
pub mod priors_precedence;  // gh#75: shared prior-resolution chain for fit run + profile
pub mod fit_tree;
pub mod fit_view;
pub mod method_result;
pub mod config_diff;
pub mod table_row;
pub mod fit_table;
pub use fit_table::cmd_fit_table;
pub mod fit_summary;
pub(crate) mod run_summary; // summary.json: a fit run's stored summary, without its rendered fields
pub(crate) use fit_summary::cmd_fit_summary;
pub mod pmmh;
// The fit report: what a method run shows (proposal 2026-11-08).
pub mod report;
// Likelihood-noise preflight shared by the pseudo-marginal methods: the
// measured log L-hat spread and the acceptance ceiling it implies (gh#764).
pub mod pf_noise;
pub mod pgas;
pub(crate) mod output; // where a running method writes its files: the run's own, opened by name
pub mod trace_writer;
pub(crate) mod resumed_chain; // a resumed sampler leaf holds the whole chain, and records its segments
pub mod synthetic;
pub mod gating;
pub mod chain_diagnostics;  // gh#406, gh#1008: per-chain loglik comparison, recorded at run end
pub mod cross_chain_compat; // gh#785: log p(x_j | θ_i) across the path-augmented chains
pub mod path_renewal;       // gh#791: trajectory renewal resolved in time, and its two scalars
pub mod latent_convergence; // gh#822: R̂/ESS of every latent state at every substep across chains
pub mod filter_ess;         // gh#685: the conditional filter's ESS at every observation, pooled over sweeps
pub mod row_convergence;    // gh#794: R̂/ESS of the value in one predictive/quantity row
pub mod dt_check;
pub mod init;
pub mod starts;         // the `starts` rule: where a method's chains begin
pub mod steps;          // each estimated parameter's step for the method that runs
pub mod param_surfaces; // where each parameter enters the compiled model, and the checks a method makes of it
pub mod chain_starts;
pub mod chain_fanout;   // gh#821: run chains so each started chain keeps advancing
pub mod loglik_eval;
pub mod methods;
#[cfg(feature = "ode")]
pub mod nlopt_method;
#[cfg(feature = "ode")]
pub(crate) mod opt_trace; // opt_trace.tsv: each NLopt search's running best, written live
pub mod fitted_model; // is the given model the one a fit ran on (the replay check)
pub mod joint;    // gh#322: keyed-joint (θ, X) read — LatentPath classifier + join
pub mod failures; // the deterministic-failure record a predictive run writes as report.json
pub mod predict;  // `camdl predict`: free-forward posterior predictive verb + types
pub(crate) mod predict_streams; // which declared streams a predictive covers, skips or fails (D76)
pub mod contrasts; // gh#322: counterfactual `contrasts {}` two-arm replay reducer (stage C)

/// `camdl fit methods` — print the supported (algorithm, backend) pairs.
/// Reads from `methods::METHODS`, the single source of truth.
pub fn cmd_fit_methods() {
    let out = crate::out::Out::new(false);
    out.report_raw(format_args!("{}", methods::render_matrix()));
}

// ─── New `camdl fit run` entry point (config_v2) ────────────────────────────

/// Where the run's `starts` rule came from, for the prefix of an error about
/// it: `--starts <spec>` when the flag set it, else the fit file.
fn starts_origin(a: &crate::args::FitRunArgs, fit_path: &str) -> String {
    match &a.run.starts {
        Some(rule) => format!("--starts {}", rule.spelled()),
        None => fit_path.to_string(),
    }
}

/// A `from_mle` / `from_params` point that does not name every estimated
/// parameter leaves the missing ones to a bounds-uniform draw. Under a
/// `pfilter`, which scores exactly one θ, that is a random θ: refused, naming
/// the parameters. (`from_posterior` already refuses a missing column.)
pub(crate) fn refuse_pfilter_point_source_missing(
    rule: &starts::ChainStarts,
    file: &std::path::Path,
    estimated: &[&str],
) -> Result<(), String> {
    use starts::{ChainStarts, Point};
    let names: std::collections::HashSet<String> = match rule {
        ChainStarts::Point(Point::FromMle { .. }) => {
            let dir = file.parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
            state::FitState::load_input(&dir)?.start_values.into_keys().collect()
        }
        ChainStarts::Point(Point::FromParams { .. }) => {
            crate::util::load_params_toml(&file.to_string_lossy())?.into_keys().collect()
        }
        _ => return Ok(()),
    };
    let missing: Vec<&str> = estimated.iter().copied().filter(|n| !names.contains(*n)).collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "a `pfilter` method scores the log-likelihood at one θ, and `starts = {}` ({}) \
         gives no value for {}, which would be drawn at random within the bounds. Add {} \
         to the source, or give {} a `start` and use `starts = \"single\"`.",
        rule.spelled(), file.display(), missing.join(", "),
        if missing.len() == 1 { "it" } else { "them" },
        if missing.len() == 1 { "it" } else { "each" },
    ))
}

/// gh#191: the model-capability gate must run on the fit-run path, against
/// the method's declared simulation backend. Returns the offending message
/// prefixed with the method's name so the user knows where to look.
fn gate_run_method_against_model(
    algorithm: &config_v2::Algorithm,
    compiled: &sim::CompiledModel,
    dt: f64,
) -> Result<(), String> {
    let method_name = algorithm.method_name();
    {
        let method = algorithm;
        if let Err(msg) = methods::check_model_capabilities(method.backend(), compiled) {
            return Err(format!("method '{}': {}", method_name, msg));
        }
        // gh#449: the recurring-fire collision guard (gh#447) lived only at the
        // three forward backends' entry points; check it here too, per method,
        // so a coarse-`dt` fit fails loudly instead of quietly losing fires.
        //
        // Only a Grid clock can drop a fire (effect plan, proposal 2026-10-04):
        // the particle-filter methods step chain-binomial, which fires one
        // intervention once per step. An ODE-backed method fires every effect at
        // its own instant — at `dt` and at a coarse warm-up `burnin_dt` alike —
        // so for it the check does not apply.
        let clock = match method.backend() {
            crate::run_meta::InferenceBackend::Ode => sim::effect_plan::FireClock::Exact,
            _ => sim::effect_plan::FireClock::Grid { dt },
        };
        if let Err(e) =
            compiled.validate_recurring_dt_collisions(compiled.model.simulation.t_start, clock)
        {
            return Err(format!("method '{}' (dt = {}): {}", method_name, dt, e));
        }
    }
    // gh#166 B2: warn if an ODE-backed method will fit a `dt`-in-rate model
    // with first-order Euler incidence (the high-order augmented flow is undefined
    // when a rate depends on the step size).
    if algorithm.backend() == crate::run_meta::InferenceBackend::Ode {
        methods::warn_if_ode_euler_flow(compiled);
    }
    Ok(())
}

/// How often the heartbeat thread rewrites a running method's `progress.json`.
/// Wall-clock and fixed, deliberately independent of step cadence: one
/// national-scale PGAS sweep can take minutes, so a step-boundary heartbeat
/// would be as stale as the trace it is meant to substitute for (gh#278).
pub(crate) const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// The liveness/progress heartbeat one method of `algorithm` runs under: what
/// its step counter counts, how many of those a complete run takes, and — for
/// the one method that reports per-chain liveness — the roster it was
/// configured to run.
///
/// Built here, at the one point every method reaches the dispatch through,
/// rather than inside a method's own runner. A heartbeat constructed per runner
/// is the heartbeat some other method silently does not have: before gh#900
/// only PGAS built one, so a five-hour PMMH fit and a dead one looked identical
/// to any reader deciding liveness from the artifact's freshness (the gh#278
/// contract).
///
/// The step unit, one line each:
///
/// - PGAS — one sweep, split into warm-up and sampling at `burn_in`. The chain
///   roster rides along so the artifact can say "1 of 24 sampling" from its
///   first write (gh#751).
/// - PMMH and `mh` — one MCMC iteration, split the same way.
/// - IF2 — one cooling iteration; a search, so a single `optimizing` phase.
/// - `nl-sbplx` / `nl-bobyqa` / `nl-lbfgs` — one objective evaluation against
///   `max_evals`. The gradient method counts the same unit: one `det_grad`
///   call is one evaluation.
/// - `pfilter` — replicates of one point, which is neither an MCMC phase nor a
///   search. It gets an inert guard and writes no `progress.json` at all,
///   rather than a counter labelled with a phase it is not in.
///
/// A method whose options fail to parse also gets an inert guard: the dispatch
/// arm reports that error and exits, and there is no run to report progress
/// for.
pub(crate) fn method_heartbeat(
    sink: io::ProgressSink,
    algorithm: &config_v2::Algorithm,
) -> io::HeartbeatGuard {
    use config_v2::Algorithm as A;
    let dir = sink;
    let held = io::HeartbeatGuard::new;
    match algorithm {
        A::PGAS { .. } => match pgas::PgasMethodOpts::from_algorithm(algorithm) {
            Ok(o) => held(io::Heartbeat::mcmc(
                dir, o.burn_in as u64, o.n_sweeps as u64, HEARTBEAT_INTERVAL, o.n_chains)),
            Err(_) => io::HeartbeatGuard::inert(),
        },
        // The roster is 0, so no `chains` block is written: gh#751 wired
        // per-chain reporting into PGAS only, and a table of rows that never
        // fill would say less than no table at all.
        A::PMMH { .. } | A::Mh { .. } => match pmmh::PmmhMethodOpts::from_algorithm(algorithm) {
            Ok(o) => held(io::Heartbeat::mcmc(
                dir, o.burn_in as u64, o.n_steps as u64, HEARTBEAT_INTERVAL, 0)),
            Err(_) => io::HeartbeatGuard::inert(),
        },
        A::IF2 { iterations, .. } =>
            held(io::Heartbeat::optimizing(dir, *iterations as u64, HEARTBEAT_INTERVAL)),
        A::NlSbplx(c) | A::NlBobyqa(c) | A::NlLbfgs(c) =>
            held(io::Heartbeat::optimizing(dir, c.max_evals as u64, HEARTBEAT_INTERVAL)),
        A::PFilter { .. } => io::HeartbeatGuard::inert(),
    }
}

/// Write a method's findings to the run's `diagnostics.json`, replacing any
/// earlier write.
///
/// A failed write is the run's failure. `diagnostics.json` is the only record
/// of the run's findings, and `fit summary` reads its count and its verdict
/// back from it, so a run that reported success without it would leave a fit
/// whose problems no later command can show.
pub(crate) fn write_diagnostics(
    collector: &sim::inference::diagnostic::DiagnosticCollector,
    files: &dyn output::RunWriter,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&collector.snapshot())
        .map_err(|e| format!("cannot write diagnostics.json: {e}"))?;
    files.append(DIAGNOSTICS_FILE)
        .and_then(|f| f.replace(json.as_bytes()))
        .map_err(|e| format!("cannot write diagnostics.json: {e}"))
}

/// The record of a method's findings.
pub(crate) const DIAGNOSTICS_FILE: &str = "diagnostics.json";

/// [`write_diagnostics`] on a path where the run is already failing with
/// `reason`: the run's own error stays the message, and a failed write is
/// added to it rather than replacing it.
pub(crate) fn write_diagnostics_then_fail(
    collector: &sim::inference::diagnostic::DiagnosticCollector,
    files: &dyn output::RunWriter,
    reason: String,
) -> String {
    match write_diagnostics(collector, files) {
        Ok(()) => reason,
        Err(e) => format!("{reason}\n  and its findings were not recorded: {e}"),
    }
}

/// Abandon the running method, from the dispatch, the runners and the chain
/// workers alike (`Out` is `Sync`): the findings it recorded go to its leaf's
/// `diagnostics.json`, its report as far as it got is drawn and recorded as
/// the event log's `failed` `method_report`, and the error is printed and
/// recorded as the heartbeat's terminal `failed` state in `progress.json`
/// (`Out::error`), then the process exits non-zero. The run's own error stays
/// first; a record that could not be written is appended to it.
///
/// `std::process::exit` runs no destructor, so the heartbeat guard's `Drop`
/// backstop cannot see these paths. Routing every fatal exit of a running
/// method through this one function is what makes "a method always leaves a
/// terminal record" true rather than nearly true.
/// Stop a method whose run was refused for evaluating a forcing outside its
/// range ([`crate::out::Out::forcing_refusal`]): its claim is failed with
/// the refusal. Polled once per iteration by the samplers' progress
/// callbacks, on every chain; the command polls its own policy once more
/// before the finalize, which covers every method.
pub(crate) fn stop_if_refused(out: &crate::out::Out) {
    match out.forcing_refusal() {
        crate::out::ForcingRefusal::Refused(message) => abandon_method(out, message),
        // Outside a held claim nothing defers a refusal: the reporter has
        // already ended the command.
        crate::out::ForcingRefusal::Clear | crate::out::ForcingRefusal::NoClaim => {}
    }
}

pub(crate) fn abandon_method(out: &crate::out::Out, message: String) -> ! {
    let message = match out.method_files() {
        Some(files) => write_diagnostics_then_fail(&out.collector(), files.as_ref(), message),
        None => message,
    };
    if let Some(r) = out.method_record() {
        out.method_report(&report::build::MethodOutcome::Failed {
            header: r.header, findings: r.findings, error: message.clone(),
        });
    }
    let message = match out.events_result() {
        Ok(()) => message,
        Err(e) => format!("{message}\n  and its events were not recorded: {e}"),
    };
    out.error(crate::out::CommandError::new(message)).exit()
}

/// What one method's run needs besides the `Out` it reports through: the
/// problem and method it runs, the run's files it writes, its seed and
/// starts, the base it resumes, and the heartbeat and collector the
/// command began it with.
pub(crate) struct MethodContext<'a> {
    pub problem: &'a config_v2::Problem,
    pub method: &'a config_v2::Method,
    /// The resolved steps of `method`, which the run is keyed by.
    pub steps: &'a steps::ResolvedSteps,
    pub model: &'a ir::Model,
    pub files: &'a dyn output::RunWriter,
    pub seed: u64,
    pub starts: &'a chain_starts::ResolvedStarts,
    /// The base's carried chain files are already open in `files`; this is
    /// the scratch copy of the base's own files, read for its warm-up record.
    pub resume_base: Option<&'a std::path::Path>,
    /// The key a resumed chain's state must carry: the run's chain identity.
    pub resume_guard: &'a str,
    pub heartbeat: &'a std::sync::Arc<io::HeartbeatGuard>,
    pub collector: &'a std::sync::Arc<sim::inference::diagnostic::DiagnosticCollector>,
    /// The problem's run id, which `mle_params.toml` records as the fit it
    /// came from.
    pub input_hash: &'a str,
    pub forcing: &'a crate::forcing_extrapolation::ForcingPolicy,
}

/// What a method's run hands back to the command.
pub(crate) struct MethodRan {
    pub best_loglik: Option<f64>,
    pub best_chain: Option<score::convergence::ChainIndex>,
    pub warmup: Vec<report::ChainWarmup>,
    /// Per chain, the index the resumed segment begins at; empty for a fresh
    /// run.
    pub resume_boundary: Vec<usize>,
    /// A hint said under the `stored` line (IF2's best chain).
    pub stored_hint: Option<String>,
}

/// Run `cx.method` on `cx.problem`, writing every file it makes into
/// `cx.files`. A fatal error abandons the method through
/// [`abandon_method`], which ends the command.
pub(crate) fn run_method(out: &mut crate::out::Out, cx: MethodContext<'_>) -> MethodRan {
    use config_v2::Algorithm;
    let MethodContext { problem: sweep_config, method, steps, model, files, seed, starts: resolved_starts,
        resume_base, resume_guard, heartbeat, collector, .. } = cx;
    let method_name = method.algorithm.method_name();
    let method_best_loglik: Option<f64>;
    let mut method_best_chain: Option<score::convergence::ChainIndex> = None;
    let mut method_warmup: Vec<report::ChainWarmup> = Vec::new();
    let mut resume_boundary: Vec<usize> = Vec::new();
    let mut stored_hint: Option<String> = None;
    match &method.algorithm {
        Algorithm::IF2 { backend, chains, particles, iterations, cooling, cooling_target_iters, loglik_eval, gate, dt_check, .. } => {
            // clean_eval comes straight from the `[method]` table — it is part
            // of the run's identity (`If2Identity::loglik_eval`), so it has
            // no CLI override
            // (gh#189: a CLI override bypassed the run_id and silently
            // re-scored under the same key). The gate's `--decibans-thresh`
            // override is applied through `apply_cli_overrides` BEFORE the
            // identity is taken, so `gate` here already carries it (gh#540
            // seam).
            let effective_loglik_eval = loglik_eval.clone();
            let effective_gate = gate.clone();

            let mut run_config = runner::FitRunConfig::build(
                sweep_config,
                Some(&method.algorithm), steps,
                *chains, *particles, *iterations,
                *cooling, *cooling_target_iters,
                // gh#506: per-chain dispersion is the starts rule's job;
                // the base point carries `[estimate].start`.
                seed, false,
            ).unwrap_or_else(|e| {
                abandon_method(out, format!("error building run config: {e}"))
            });
            run_config.loglik_eval = effective_loglik_eval.clone();
            run_config.gate = effective_gate.clone();

            let t0 = std::time::Instant::now();
            // Per-chain starting points, through the one seam every runner
            // draws from, under the resolved `starts` rule.
            let drawn = runner::draw_chain_starts_for(
                &run_config, &sweep_config.estimate, resolved_starts, *chains, seed,
            ).unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            runner::refuse_starts_on_an_end(&run_config, &drawn)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            // gh#887: a spread start the filter cannot score is redrawn,
            // up to MAX_START_ATTEMPTS, before any chain runs; every
            // attempt is on the record below.
            let drawn = runner::preflight_spread_starts(
                out,
                &run_config, &sweep_config.estimate, resolved_starts, drawn, seed,
            ).unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            let per_chain_params: Vec<Vec<sim::inference::types::EstimatedParam>> =
                drawn.to_estimated_params(&run_config.estimated_params);
            // The audit sidecar, captured before the first filter pass.
            runner::record_chain_starts(out, files, &run_config, &drawn);
            let chain_init_source = drawn.rule.spelled();
            let chain_starts_kind = drawn.rule.kind();
            let chain_results = runner::run_chains_with_per_chain_params(
                out,
                &run_config, Some(&per_chain_params), collector,
                Some(files), heartbeat)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            let elapsed = t0.elapsed();
            // A chain skipped mid-run started from an accepted start, so
            // the file written before the run calls it `accepted`. Write
            // it again marking each such chain `skipped`: the file is the
            // record of what every chain did.
            if !chain_results.skipped.is_empty() {
                let drawn = drawn.clone().with_skipped(chain_results.skipped.clone());
                runner::record_chain_starts(out, files, &run_config, &drawn);
            }

            // Write outputs
            let param_names: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
            runner::write_chain_outputs(
                files, &chain_results.results,
                &run_config.estimated_params, &param_names,
                &run_config.base_params, &run_config.compiled,
                Some(&chain_results.loglik_eval),
            ).unwrap_or_else(|e| out.say(format_args!("warning: {}", e)));
            runner::write_clean_eval_tsv(
                files,
                &chain_results.loglik_eval, &run_config.estimated_params,
            ).unwrap_or_else(|e| out.say(format_args!("warning: {}", e)));
            runner::write_run_root_final_params(
                files,
                &chain_results.loglik_eval, &run_config.estimated_params,
                &param_names, &run_config.base_params, &run_config.compiled,
            ).unwrap_or_else(|e| out.say(format_args!("warning: {}", e)));
            runner::write_iteration_trace(files, &chain_results.results)
                .unwrap_or_else(|e| out.say(format_args!("warning: {}", e)));

            // Write fit_state.toml — the leaf's result record.
            // Source params from the clean-eval winner θ̂ (GH #16) so
            // mle_params.toml and final_params.toml agree, and so a
            // `from_mle` start from this leaf lands in the basin clean-eval
            // actually picked.
            let winner_theta = chain_results.winner_theta();
            let start_values = runner::collect_all_params(
                winner_theta, &run_config.estimated_params, &run_config.model,
                &run_config.base_params, &run_config.compiled,
            );

            // Post-fit Richardson dt-convergence check at θ̂
            // (gh#52). Auto-runs when `dt_check.enabled = true`
            // (default); evaluates loglik(θ̂; dt) on a halving
            // ladder and warns when the MLE is discretization-
            // dependent. Catches the silent-wrong-answer mode
            // where coarse dt creates a fake basin that synth-
            // recovery can't detect (it shares the same dt).
            // See docs/dev/proposals/2026-05-07-richardson-dt-check.md.
            // --no-dt-check / --dt-check-halvings are applied through
            // `apply_cli_overrides` BEFORE the identity is taken, so
            // `dt_check` here already carries them (gh#540 seam — the
            // result is stored in fit_state.toml, so the knobs are
            // identity-defining). --dt-check-strict is resolved into the
            // stored threshold the same way (gh#730).
            let effective_dt_check = dt_check.clone();
            // Grouped to spell its mnemonic, not in equal-width groups.
            #[allow(clippy::unusual_byte_groupings)]
            let dt_check_seed = seed.wrapping_add(0xd7c4ec_5eed); // "dtchec seed"
            let dt_check_result = dt_check::run_richardson_ladder(
                &run_config,
                winner_theta,
                &effective_dt_check,
                *backend,
                &dt_check::DtCheckInherits {
                    n_particles:  effective_loglik_eval.n_particles,
                    n_replicates: effective_loglik_eval.n_replicates,
                    combine:      effective_loglik_eval.combine,
                },
                dt_check_seed,
            )
            .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            dt_check::print_terminal_report(out, &dt_check_result);
            let fit_state = state::FitState {
                method: method_name.to_string(),
                seed,
                input_hash: None,
                camdl_version: Some(crate::version::VERSION_SHORT.into()),
                best_loglik: chain_results.best_loglik,
                initial_loglik: f64::NEG_INFINITY,
                best_chain: chain_results.best_chain,
                n_chains: *chains,
                // The chains that reached the end. Fewer than `n_chains`
                // makes the gate verdict `incomplete` (`gating::assess_if2`).
                n_good_chains: Some(chain_results.results.len()),
                start_values,
                loglik_type: Some(loglik::LoglikType::If2),
                acceptance_rate: None,
                tail_chain_agreement: chain_results.chain_agreement.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                initial_state_params: run_config.steps.initial_state_params(),
                chain_logliks: chain_results.results.iter()
                    .map(|(_, r)| r.final_loglik).collect(),
                chain_eval_logliks: chain_results.chain_eval_logliks(),
                chain_eval_ids: chain_results.chain_eval_ids(),
                chain_eval_ses: chain_results.chain_eval_ses(),
                // Persist the gate / clean-eval config that was
                // *actually in force* — `effective_gate` and
                // `effective_loglik_eval` above already collapsed the
                // priority chain (CLI flag > `[method]` table > defaults).
                // `summary` reads these so its verdict line reports
                // against the threshold the run was judged by, not
                // whatever `fit.toml` says at summary-time.
                // See proposal §Phase 3.
                resolved_gate: Some(effective_gate.clone()),
                resolved_loglik_eval: Some(effective_loglik_eval.clone()),
                chain_init_source: Some(chain_init_source.clone()),
                chain_starts_kind: Some(chain_starts_kind),
                dt_check: if matches!(dt_check_result.verdict,
                    dt_check::DtCheckVerdict::Skipped)
                {
                    None  // skipped → omit the block, mirroring legacy semantics
                } else {
                    Some(dt_check_result.clone())
                },
                // gh#764: IF2 maximises, it does not accept on a likelihood
                // ratio, so the acceptance ceiling does not apply to it.
                pf_noise: None,
                optimizer_convergence: None,
                // The per-chain comparison of the clean evaluations, computed
                // once here and read by the warm-start source check. Not
                // raised as findings: the clean-eval standard error of a
                // replicate mean is small enough that starts 0.07 nats apart
                // differ "significantly" (the reporting fixture's IF2 fit),
                // and whether such a spread matters is the decibans leg's
                // question, which has a floor in log-likelihood units. The
                // comparison names chains inside that refusal.
                chain_loglik_agreement: chain_diagnostics::clean_eval_agreement(
                    &chain_results.chain_eval_ids(),
                    &chain_results.chain_eval_logliks(),
                    &chain_results.chain_eval_ses(),
                ),
            };
            fit_state.save(files).unwrap_or_else(|e| {
                out.say(format_args!("warning: could not save fit_state: {}", e));
            });

            // Write mle_params.toml — clean-eval winner θ̂ (GH #16).
            let all_params = runner::collect_all_params(
                winner_theta, &run_config.estimated_params, &run_config.model,
                &run_config.base_params, &run_config.compiled,
            );
            let model_identity =
                crate::resolve::model_identity_from_ir(&run_config.model_ir_json);
            let rescore = &chain_results.loglik_eval.per_chain
                [chain_results.loglik_eval.overall_winner_idx];
            let metadata = provenance::MleMetadata {
                // Full fit-level hash — lets a reader locate the
                // originating fit dir from just the mle_params.toml.
                // The fit hash (not a model-only digest) so it pins
                // the model+data+config triple, not just the model.
                input_hash: cx.input_hash.to_string(),
                // The leaf is `<fit folder>/<method>-<h8>/seed_<n>-<h8>`.
                model_identity: model_identity.clone(),
                seed,
                method: method_name.to_string(),
                best_chain: chain_results.best_chain,
                // Record the backend the method actually fit on (gh#241):
                // the `simulate --params` guardrail replays θ̂ with this,
                // so it must be the method's backend. `InferenceBackend`
                // is a valid `ForwardBackend` (total `From`).
                backend: method.algorithm.backend().into(),
                dt: sweep_config.config.dt,
                // The recorded log-likelihood is the loglik-eval re-score
                // of the winning chain, so its uncertainty is that
                // re-score's: the Monte Carlo standard error of the
                // combined estimate, the particles of each replicate, and
                // the filter's effective sample size at the estimate. The
                // IF2 swarm's own size (`particles`) is not the precision
                // of this number.
                loglik: chain_results.best_loglik,
                loglik_sd: rescore.se,
                n_particles: effective_loglik_eval.n_particles,
                ess_at_mle: (rescore.filter_stats.ess_mean.is_finite()
                    && rescore.filter_stats.ess_min.is_finite())
                    .then_some((rescore.filter_stats.ess_mean,
                                rescore.filter_stats.ess_min)),
            };
            provenance::mle_params_text(&all_params, &metadata)
                .and_then(|text| files.put("mle_params.toml", text.as_bytes()))
                .unwrap_or_else(|e| out.say(format_args!("warning: {}", e)));

            // Recorded, as every other method's are, so `fit summary`
            // shows the findings this run printed; the end of the run
            // prints them.
            write_diagnostics(collector, files)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));

            method_best_loglik = Some(chain_results.best_loglik);
            method_best_chain = Some(chain_results.best_chain);
            stored_hint = Some(format!("best ll={:.1} (chain {}) in {:.1}s",
                chain_results.best_loglik, chain_results.best_chain, elapsed.as_secs_f64()));
        }
        Algorithm::PGAS { .. } => {
            let pgas_opts = pgas::PgasMethodOpts::from_algorithm(&method.algorithm)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            // gh#540: no CLI overrides here. Every flag that reaches this
            // method was written into it before its content address was
            // taken, so `from_algorithm` above already carries them.
            resume_boundary = pgas::run_method(
                sweep_config,
                method,
                files,
                pgas_opts,
                seed, false,
                resume_base,
                resume_guard,
                resolved_starts,
                heartbeat,
                out,
            ).unwrap_or_else(|e| {
                abandon_method(out, format!("error running pgas: {e}"))
            });
            // Bubble loglik from fit_state.toml written by PGAS runner
            let fs = load_method_result_or_exit(out, method_name, files.folder());
            method_best_loglik = Some(fs.best_loglik);
            method_best_chain = Some(fs.best_chain);
        }
        Algorithm::PMMH { .. } => {
            let pmmh_opts = pmmh::PmmhMethodOpts::from_algorithm(&method.algorithm)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            (method_warmup, resume_boundary) = pmmh::run_method(
                sweep_config,
                method, steps,
                files,
                pmmh_opts,
                seed, false,
                resume_base,
                resume_guard,
                resolved_starts,
                // A resumed chain extends its base's warm-up record.
                &base_warmup(resume_base, &method.algorithm),
                // PMMH's dt-check is the PF-based one wired on the IF2 path.
                /* dt_check_opt */ None,
                heartbeat,
                out,
            ).unwrap_or_else(|e| {
                abandon_method(out, format!("error running pmmh: {e}"))
            });
            let fs = load_method_result_or_exit(out, method_name, files.folder());
            method_best_loglik = Some(fs.best_loglik);
            method_best_chain = Some(fs.best_chain);
        }
        Algorithm::Mh { dt_check, .. } => {
            // Deterministic-ODE Metropolis-Hastings. Routes through the
            // shared PMMH machinery (chains, adaptive proposal, priors,
            // MAP, R̂/ESS, trace output); `pmmh::run_method` swaps the PF
            // likelihood for `compute_ode_loglik` when the algorithm is the
            // Mh variant. `PmmhMethodOpts::from_algorithm` parses the Mh
            // fields with `n_particles = 0` (the deterministic path has
            // no filter).
            let pmmh_opts = pmmh::PmmhMethodOpts::from_algorithm(&method.algorithm)
                .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            // Deterministic ODE dt-check at the MAP (gh#52, gh#227). On by
            // default; honours the same CLI flags as the IF2 path
            // (--no-dt-check / --dt-check-halvings / --dt-check-strict).
            // gh#726: the method's dt_check field — CLI overrides were
            // applied through `apply_cli_overrides` BEFORE the identity
            // was taken, and the field is in `MhIdentity` (the
            // result is stored in fit_state.toml.dt_check).
            let mh_dt_check = dt_check.clone();
            (method_warmup, resume_boundary) = pmmh::run_method(
                sweep_config,
                method, steps,
                files,
                pmmh_opts,
                seed, false,
                resume_base,
                resume_guard,
                resolved_starts,
                // A resumed chain extends its base's warm-up record.
                &base_warmup(resume_base, &method.algorithm),
                Some(mh_dt_check),
                heartbeat,
                out,
            ).unwrap_or_else(|e| {
                abandon_method(out, format!("error running mh: {e}"))
            });
            let fs = load_method_result_or_exit(out, method_name, files.folder());
            method_best_loglik = Some(fs.best_loglik);
            method_best_chain = Some(fs.best_chain);
        }
        Algorithm::NlSbplx(nl_cfg)
        | Algorithm::NlBobyqa(nl_cfg)
        | Algorithm::NlLbfgs(nl_cfg) => {
            #[cfg(feature = "ode")]
            {
                // Model identity + data digests for the mle_params.toml
                // provenance block (same shape as the IF2 path uses).
                // Load the canonical IR JSON from the pre-compiled IR so this
                // provenance read doesn't re-invoke camdlc.
                let model_ir_json = sweep_config.compiled_model()
                    .and_then(crate::util::load_ir)
                    .map(|(_, ir_json)| ir_json)
                    .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
                let model_identity_for_prov =
                    crate::resolve::model_identity_from_ir(&model_ir_json);
                // Deterministic ODE dt-check at θ̂ (gh#52, gh#227). On by
                // default. gh#726: the method's dt_check field — CLI
                // overrides were applied through `apply_cli_overrides`
                // BEFORE the identity was taken, and NloptConfig
                // full-serializes into its identity (the result is
                // stored in fit_state.toml.dt_check).
                let nl_dt_check = nl_cfg.dt_check.clone();

                nlopt_method::run_method(
                    out,
                    sweep_config,
                    method,
                    files,
                    seed,
                    resolved_starts,
                    cx.input_hash,
                    &model_identity_for_prov,
                    &nl_dt_check,
                    heartbeat,
                ).unwrap_or_else(|e| {
                    abandon_method(out, format!("error running {method_name}: {e}"),
                    )
                });
                // Recorded as every other method's findings are. The
                // optimiser raises none of the sampler diagnostics, so the
                // file holds what the run pushed (possibly nothing): an
                // empty `diagnostics.json` says the run raised no
                // findings, where no file says only that nobody asked.
                write_diagnostics(&out.collector(), files)
                    .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
                let fs = load_method_result_or_exit(out, method_name, files.folder());
                method_best_loglik = Some(fs.best_loglik);
                method_best_chain = Some(fs.best_chain);
            }
            #[cfg(not(feature = "ode"))]
            {
                let _ = (method_name, &sweep_config, files, seed, nl_cfg, &resolved_starts);
                abandon_method(&out, format!(
                        "error: this binary was built without --features ode, \
                         which is required for algorithm = \"{}\". Rebuild \
                         with `cargo build --features ode` (default).",
                        method.algorithm.method_name()
                    ),
                );
            }
        }
        Algorithm::PFilter { particles, replicates, record_ancestry, record_prequential, .. } => {
            let n_reps = replicates.unwrap_or(1);
            // record_ancestry: CLI flag is a one-way override to true
            // (TOML default false); no flag means use TOML.
            // record_prequential: TOML default true (per the
            // 2026-04-20 prequential proposal); explicit
            // `record_prequential = false` in [method] opts out,
            // and the CLI flag can re-enable it on a per-invocation
            // basis without editing the TOML.
            let record_ancestry = *record_ancestry;
            let want_prequential = *record_prequential;

            // Build run config (reuse IF2 builder with 1 chain, N particles).
            // cooling_target_iters=1 here is harmless: PFilter doesn't
            // cool, so the IF2-shaped config field is never read.
            let run_config = runner::FitRunConfig::build(
                sweep_config,
                Some(&method.algorithm), steps,
                1, *particles, 1, 1.0, 1, seed, false,
            ).unwrap_or_else(|e| {
                abandon_method(out, format!("error building pfilter config: {e}"))
            });

            // The point the filter scores: the base point, or the point a
            // sourced `starts` rule names (`from_mle = "@fit"` scores at
            // that fit's estimate). A pfilter runs replicates of one point,
            // so one draw of the rule is the point.
            let drawn = runner::draw_chain_starts_for(
                &run_config, &sweep_config.estimate, resolved_starts, 1, seed,
            ).unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            let mle_params: Vec<f64> = drawn
                .to_param_vecs(&run_config.estimated_params, &run_config.base_params)
                .into_iter()
                .next()
                .unwrap_or_else(|| run_config.base_params.clone());
            let t0 = std::time::Instant::now();

            let mut logliks = Vec::new();
            // Prequential: record on the first replicate only; scoring
            // is a property of the point estimate, not a per-rep
            // quantity. Subsequent reps just build the loglik SD.
            let mut preq_trace: Option<score::trace::PrequentialTrace> = None;
            for r in 0..n_reps {
                let pf_seed = seed ^ ((r as u64).wrapping_mul(0x7f4a7c15_u64));
                let process = run_config.build_process();
                let obs_model = run_config.build_obs_model();
                // Prequential / ancestry recording: gated by the
                // user-facing flags from Algorithm::PFilter. Prequential
                // is per-method scoring (point-estimate property),
                // so we only record it on the first replicate;
                // subsequent reps just build the loglik SD.
                let record_preq = want_prequential && r == 0;
                let smc_config = sim::inference::traits::SMCConfig {
                    record_prequential: record_preq,
                    record_ancestry, path_recording: Default::default(),
                    ..run_config.smc_config()
                };
                let result = sim::inference::bootstrap_filter(
                    &process, &obs_model, &mle_params, &smc_config, pf_seed,
                ).unwrap_or_else(|e| {
                    abandon_method(out, format!("pfilter error: {}", crate::util::sim_error_text(&e)))
                });
                if record_preq {
                    if let Some(ref recorded) = result.prequential {
                        // gh#268/gh#648/gh#269: score each stream against
                        // its real observed values, the same seam `camdl
                        // pfilter --save-prequential` uses. NOT
                        // `run_config.observations[i].value` — that is the
                        // canonical union TIME axis, whose `value` is a
                        // never-scored 0.0 placeholder (`runner.rs`), so it
                        // scored every forecast against a vector of zeros.
                        let per_stream_obs = obs_model.per_stream_observed();
                        // gh#833: what each value was accumulated over,
                        // for `compare`'s window gate.
                        let per_stream_cov = obs_model.per_stream_coverage();
                        let mut trace = sim::inference::prequential::build_trace(
                            recorded, &per_stream_obs, &per_stream_cov,
                            &result.ess_trace, 0,
                            pf_seed,
                            obs_model.first_observation_opens_after_run_start(smc_config.t_start),
                            None);
                        // The forcings the fit's policy held outside their
                        // ranges, as `camdl pfilter --save-prequential`
                        // records them, so `compare` reading this trace
                        // reports them on its row. The policy is the
                        // command's: every sweep point and seed of this fit
                        // shares it, as do the steps before its one
                        // `[method]` runs (the capability scan, the
                        // training-partition build, the `[synthetic]`
                        // generator). So the list can name a forcing held
                        // by one of those and not by this trace's pass, an
                        // over-report; it never misses one this pass held.
                        trace.warnings.extend(cx.forcing.trace_warnings());
                        // Schema 5: how this trace was computed, recorded
                        // as `camdl pfilter --save-prequential` records
                        // it — one pass (replicate 0) at the point this
                        // method evaluates, under the fit's intervention
                        // selection.
                        let base = crate::util::FitBase::of(sweep_config)
                            .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
                        let (enabled, disabled) = if base.is_empty() {
                            (Vec::new(), Vec::new())
                        } else {
                            base.lists(&run_config.model_declared).unwrap_or_else(|e| {
                                abandon_method(out, format!("error: {e}"))
                            })
                        };
                        trace.derivation = Some(
                            score::trace::Derivation::single_pass(
                                &trace, *particles, smc_config.dt, pf_seed,
                                score::trace::ThetaSource::Supplied,
                                enabled, disabled));
                        preq_trace = Some(trace);
                    }
                }
                logliks.push(result.log_likelihood);
                if n_reps <= 10 || r % (n_reps / 10) == 0 {
                    out.say(format_args!("  pfilter rep {}/{}: loglik={:.1}", r + 1, n_reps, result.log_likelihood));
                }
            }
            let elapsed = t0.elapsed();

            let result = crate::fit::method_result::PfilterRun::from_run(&logliks, *particles);
            out.say(format_args!("\n  loglik = {:.1} ± {:.1} ({} reps, {} particles, {:.1}s)",
                result.ll_mean, result.ll_sd, n_reps, particles, elapsed.as_secs_f64()));

            // Write logliks.tsv
            {
                use std::io::Write;
                let mut f: Vec<u8> = Vec::new();
                writeln!(f, "replicate\tloglik").unwrap();
                for (i, ll) in logliks.iter().enumerate() {
                    writeln!(f, "{}\t{}", i + 1, crate::fit::method_result::loglik_tsv_cell(*ll)).unwrap();
                }
                files.put("logliks.tsv", &f).unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
            }

            // Write prequential trace (plug-in predictive at MLE).
            // Scoring is a point-estimate property — rep 0 only.
            // Routed through the shared writer so the fit-method TSV
            // carries the same tidy/long schema as `camdl pfilter
            // --save-prequential` (gh#650).
            if let Some(ref trace) = preq_trace {
                let calendar = ir::caltime::CalendarAxis::from_model(&run_config.model)
                    .unwrap_or_else(|e| abandon_method(out, format!("error: {e}")));
                let written = crate::prequential_out::prequential_files(trace, &calendar)
                    .map_err(|e| e.to_string())
                    .and_then(|p| {
                        files.put("prequential.streams.tsv", &p.streams)?;
                        files.put("prequential.tsv", &p.joint)?;
                        files.put("prequential.json", &p.json)
                    });
                written.unwrap_or_else(|e| {
                    abandon_method(out, format!("error writing prequential: {e}"),
                    )
                });
                out.say(format_args!("  prequential: {}",
                    crate::prequential_out::summary_line(trace)));
                out.say_raw(format_args!("{}", crate::prequential_out::surprise_table(trace, &calendar)));
            }
            method_best_loglik = Some(result.ll_mean);
        }
    }
    MethodRan {
        best_loglik: method_best_loglik,
        best_chain: method_best_chain,
        warmup: method_warmup,
        resume_boundary,
        stored_hint,
    }
}

/// gh#191: gate the model's required capabilities against the method's
/// declared backend, before any fitting work, and say when PGAS falls back
/// from NUTS. Returns the kernel whose band the method's acceptance
/// diagnostics apply.
pub(crate) fn capability_gate(
    out: &crate::out::Out,
    method: &config_v2::Method,
    model: &ir::Model,
    config: &config_v2::FitConfig,
    config_text: &str,
    nuts_tuning_flags: &[&'static str],
    fit_path: &str,
) -> Option<sim::inference::diagnostic::AcceptanceKernel> {
    use config_v2::Algorithm;
    // `required_capabilities()` is structural; the parameter values are
    // irrelevant to it, but `CompiledModel::new` requires every parameter to
    // carry one, so a value-less parameter gets a placeholder for the scan.
    let mut cap_model = model.clone();
    for p in &mut cap_model.parameters {
        if p.value.resolved_value().is_none() {
            let placeholder = p.initial_value()
                .or_else(|| crate::params_resolver::resolved_bounds(p).map(|(lo, hi)| 0.5 * (lo + hi)))
                .unwrap_or(1.0);
            p.value = p.value.with_value(placeholder);
        }
    }
    let compiled = sim::CompiledModel::new(cap_model)
        .unwrap_or_else(|e| out.fail(format!("error: {e:?}")))
        .with_eval_settings(config.problem.config.eval_settings());
    if let Err(msg) = gate_run_method_against_model(&method.algorithm, &compiled, config.problem.config.dt) {
        out.fail(format!("error: {msg}"));
    }
    let kernel = methods::acceptance_kernel(&method.algorithm, &compiled);
    // PGAS runs NUTS only where the compiler emitted rate gradients; elsewhere
    // it falls back to Metropolis-within-Gibbs. Say so, and refuse an explicit
    // `use_nuts = true` the model cannot honour.
    if let Algorithm::PGAS { use_nuts: true, .. } = &method.algorithm {
        if !sim::inference::pgas::nuts_active(true, &compiled) {
            let declared = config_v2::method_declares_key(config_text, "use_nuts");
            let tuning: Vec<&str> = [
                (config_v2::method_declares_key(config_text, "max_tree_depth"), "max_tree_depth"),
                (config_v2::method_declares_key(config_text, "dense_mass"), "dense_mass"),
            ].into_iter().filter_map(|(on, name)| on.then_some(name))
                .chain(nuts_tuning_flags.iter().copied())
                .collect();
            match config_v2::nuts_unavailable(declared, &tuning) {
                Ok(note) => out.say(format_args!("{} {note}", crate::out::paint::note_tag())),
                Err(e) => out.fail(format!("error: {fit_path}: {e}")),
            }
        }
    }
    kernel
}

/// IC-free inference: when `ic_free = true`, the startup block says the
/// filter computes log L conditional on y₁, and names every source of the
/// t = 0 spread.
pub(crate) fn say_ic_free(out: &crate::out::Out, config: &config_v2::FitConfig, model: &ir::Model) {
    if !config.problem.likelihood_objective().skips_first_observation() {
        return;
    }
    let perturb_only_at_t0: Vec<&str> = match config.inference.method.as_ref().map(|m| &m.algorithm) {
        Some(config_v2::Algorithm::IF2 { perturbation, .. }) => perturbation.iter()
            .filter(|(_, spec)| spec.initial_state)
            .map(|(n, _)| n.as_str())
            .collect(),
        _ => Vec::new(),
    };
    out.say(format_args!("\n  {} conditioning on y₁",
        crate::out::paint::stderr().paint("ic-free inference:", crate::out::paint::Style::Info)));
    let mut sources: Vec<String> = Vec::new();
    let drawn: Vec<&str> = model.initial_conditions.iter()
        .filter(|(_, s)| s.is_law())
        .map(|(name, _)| name.as_str())
        .collect();
    if !drawn.is_empty() {
        sources.push(format!("`init {{ }}` laws on [{}]", drawn.join(", ")));
    }
    if !perturb_only_at_t0.is_empty() {
        sources.push(format!("initial_state params [{}] (if2 only)", perturb_only_at_t0.join(", ")));
    }
    out.say(format_args!("    - initial state spread from: {}", sources.join("; ")));
    out.say("    - log-likelihood accumulation from t = 2 (y₁ reweights and resamples only)");
}

/// The method's settings a reader shows, as the run's inputs record them.
pub(crate) fn algorithm_record(algorithm: &config_v2::Algorithm) -> serde_json::Value {
    use config_v2::Algorithm;
    let algo_tag = algorithm.method_name();
    let backend_tag = algorithm.backend().as_str();
    match algorithm {
        Algorithm::IF2 { chains, particles, iterations, cooling, .. } =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations, "cooling": cooling }),
        Algorithm::PGAS { chains, particles, sweeps, .. } =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "sweeps": sweeps }),
        Algorithm::PMMH { chains, particles, iterations, .. } =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations }),
        Algorithm::Mh { chains, iterations, .. } =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "iterations": iterations }),
        Algorithm::PFilter { particles, replicates, .. } =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "particles": particles, "replicates": replicates }),
        Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) | Algorithm::NlLbfgs(c) =>
            serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": c.chains, "tolerance": c.tolerance, "max_evals": c.max_evals }),
    }
}

/// What a run's `fit.meta.json` is built from.
pub(crate) struct SidecarInputs<'a> {
    pub config: &'a config_v2::FitConfig,
    pub config_text: &'a str,
    pub model: &'a ir::Model,
    /// The run's problem: a sweep point's values in `[fixed]`.
    pub problem: &'a config_v2::Problem,
    pub data: &'a stream_bindings::BoundDataDigests,
    pub roles: &'a Result<crate::roles::Partition, String>,
    pub sweep_point: &'a [(String, f64)],
    pub seed: u64,
    /// Which rows trained, as the stored problem records it.
    pub partition: crate::run_meta::TrainingPartitionRecord,
    /// The table forcings whose values before the training extent depend on
    /// rows after it.
    pub forcing_interpolation: Vec<crate::run_meta::ForcingInterpolationRecord>,
}

/// The run's `fit.meta.json`: the fit's provenance as the run resolved it —
/// each prior's source, every parameter's value and source, the partition
/// into estimated and fixed. A provenance file of the run; no path in it is
/// read back.
pub(crate) fn run_sidecar(i: &SidecarInputs<'_>, out: &crate::out::Out) -> Result<serde_json::Value, String> {
    let mut side = build_fit_sidecar(i.config, i.config_text.as_bytes(), i.model, i.data)?;
    let fp = runner::resolve_fit_parameters(i.problem, i.seed);
    if let Ok(fp) = &fp {
        crate::params_resolver::emit_notices(out, &[&fp.resolved]);
        side.parameters_provenance = crate::run_meta::ParameterProvenance::of_all(&fp.resolved.params);
    }
    let roles = crate::roles::RoleRecord::new(
        i.roles.as_ref().map_err(String::clone)
            .and_then(|p| p.cell(i.sweep_point))
            .and_then(|cell| fp.as_ref().map(|fp| cell.record(&fp.resolved)).map_err(String::clone)),
        || unpartitioned_role_fields(i.problem),
    );
    side.estimated = roles.estimated;
    side.fixed = roles.fixed;
    side.partition = Some(roles.partition);
    side.training_partition = Some(i.partition.clone());
    side.forcing_interpolation = i.forcing_interpolation.clone();
    side.schema_version = crate::run_meta::FIT_META_SCHEMA_VERSION;
    serde_json::to_value(&side).map_err(|e| format!("cannot encode fit.meta.json: {e}"))
}

/// The report of a method that completed: built by the one builder from what
/// the run kept in memory — its opening block, the checks it showed, its
/// findings, its warm-up and its wall time — and the typed result of the leaf
/// it finalized (`MethodResult::load_from`, which reads `run.json` `inputs`,
/// written by `finalize`). `fit summary` builds the same report from the leaf
/// alone (`fit_summary::load_method_facts`).
pub(crate) fn completed_outcome(
    method_name: &str,
    leaf: &std::path::Path,
    shown: &std::path::Path,
    record: crate::out::MethodRecord,
    warmup: Vec<report::ChainWarmup>,
    total_seconds: f64,
    model: &ir::Model,
) -> Result<report::build::MethodOutcome, String> {
    use method_result::MethodResult;
    let result = MethodResult::load_from(leaf, method_name).map_err(|e| e.to_string())?;
    let chain_detail = match &result {
        MethodResult::Pgas(_) | MethodResult::Pmmh(_) => Some(report::ChainDetail::gather(
            leaf,
            fit_summary::recorded_chain_logliks(&result),
            matches!(result, MethodResult::Pgas(_)),
            None,
            fit_summary::forkability(&leaf.to_string_lossy(), method_name, None),
        )),
        MethodResult::If2(_) => Some(report::ChainDetail::default()),
        MethodResult::Nlopt(_) | MethodResult::Pfilter(_) => None,
    };
    let facts = report::MethodFacts {
        name: method_name.to_string(),
        method: method_name.to_string(),
        method_dir: leaf.to_path_buf(),
        header: Some(record.header),
        result,
        checks: record.checks,
        warmup,
        findings: record.findings,
        findings_unreadable: Vec::new(),
        stored: Some(report::Stored {
            leaf: report::display_path(shown),
            total_seconds: Some(total_seconds),
        }),
        chain_detail,
    };
    // The calendar of the model the fit ran, in hand: reading it back from the
    // leaf would compile the model's source again.
    let cal = report::build::CalendarContext::of_model(model);
    Ok(report::build::MethodOutcome::Completed(report::build::build_method_report(facts, &cal)))
}

/// After a runner reported success, its `fit_state.toml` is the channel
/// carrying the result back to this orchestrator (`best_loglik` /
/// `best_chain` end up in the finalized run.json inputs). A missing or
/// corrupt file at that point is a runner bug or a torn write; the previous
/// `if let Ok` swallow left a silent `null` in run.json where the result
/// belonged. Fail loudly instead.
fn load_method_result_or_exit(
    out: &mut crate::out::Out,
    method_name: &str,
    method_dir: &std::path::Path,
) -> state::FitState {
    state::FitState::load_input(&method_dir.to_string_lossy()).unwrap_or_else(|e| {
        abandon_method(
            out,
            format!(
                "error: method '{}' reported success but its fit_state.toml \
                 cannot be read back from {}: {}",
                method_name, method_dir.display(), e),
        )
    })
}

/// Whether the method's pre-run likelihood check and initial proposal read the
/// declared `[estimate].start` values even when its starts rule does not
/// start chains there: PMMH and `mh`.
pub(crate) fn still_reads_declared_starts(algorithm: &config_v2::Algorithm) -> bool {
    matches!(algorithm, config_v2::Algorithm::PMMH { .. } | config_v2::Algorithm::Mh { .. })
}

/// The base leaf's per-chain warm-up record, for a resumed PMMH or `mh` run
/// (`report::ChainWarmup`). Empty for a fresh run, and for a base written
/// before the record existed.
pub(crate) fn base_warmup(
    base: Option<&std::path::Path>,
    algorithm: &config_v2::Algorithm,
) -> Vec<report::ChainWarmup> {
    let summary = match algorithm {
        config_v2::Algorithm::Mh { .. } => crate::run_meta::FitAlgorithm::Mh,
        _ => crate::run_meta::FitAlgorithm::Pmmh,
    }
    .summary_filename();
    base.map(|b| report::read_warmup(&b.join(summary))).unwrap_or_default()
}

// gh#147 (M3.2): `archive_fit_toml` (the legacy `fit.toml.original` writer
// under `fit_dir`) was removed — fits are content-addressed now and the
// fit-level config archive for `fit table` config_diff is reworked in M3.3
// alongside the fit-level outputs' CAS relocation.

/// The digests of every file a fit's `[data]` binds, read once
/// ([`stream_bindings::StreamBindings::digests`]). Every site that computes a
/// fit-level or method-level identity takes its data digests from here, so
/// they agree.
pub(crate) fn read_data_digests(
    data: &config_v2::DataSpec,
    model: &ir::Model,
) -> Result<stream_bindings::BoundDataDigests, String> {
    stream_bindings::StreamBindings::resolve(model, data)?.digests()
}

/// `camdl fit diff`'s data side: the full SHA-256 of each training file a
/// config binds, keyed by source
/// ([`stream_bindings::StreamBindings::training_by_source`]). With no model in
/// scope the shorthand cannot be expanded and the map is empty;
/// `[data.observations]` is then hashed as written, which is by source too. A
/// file that cannot be read is left out: this compares two configs, it builds
/// no identity.
#[expect(clippy::disallowed_methods, reason = "`fit diff` hashes the data files two fit configs bind, to compare them; it stores nothing")]
fn resolved_data_hashes(
    problem: &config_v2::Problem,
    model: Option<&ir::Model>,
) -> std::collections::HashMap<String, String> {
    let Some(data) = problem.data.as_ref() else { return Default::default() };
    let by_source = match model {
        Some(m) => stream_bindings::StreamBindings::resolve(m, data).ok()
            .map(|b| b.training_by_source().clone()),
        None => data.file.is_none().then(|| data.observations.clone()),
    };
    by_source
        .map(|obs| obs.into_iter()
            .filter_map(|(name, path)| {
                std::fs::read(&path).ok().map(|b| (name, crate::hashing::sha256_hex(&b)))
            })
            .collect())
        .unwrap_or_default()
}

/// The fit's partition (proposal `2026-10-26-parameter-roles.md`, Stage 1),
/// from the fit config's `[estimate]` and `[fixed]`, the sweep's names and
/// the scenario. `fixed_via` is read before the `from_scenario` expansion.
///
/// `Err` says why the fit's roles form no partition, for an `unresolved`
/// record.
pub(crate) fn resolve_fit_roles(
    problem: &config_v2::Problem,
    model: &ir::Model,
    fixed_via: Result<Vec<(String, crate::roles::FitFixedVia)>, String>,
    sweep: &[(String, Vec<f64>)],
) -> Result<crate::roles::Partition, String> {
    use crate::roles::{DesignOrigin, RoleInputs, ScenarioFootprint};
    let fit_fixed: Vec<(String, crate::roles::FitFixedVia)> = fixed_via?;
    let design: Vec<(String, DesignOrigin)> =
        sweep.iter().map(|(n, _)| (n.clone(), DesignOrigin::Sweep)).collect();
    let scenarios: Vec<ScenarioFootprint> = match problem.scenario.as_deref() {
        Some(s) => vec![ScenarioFootprint {
            scenario: s.to_string(),
            params: crate::params_resolver::scenario_param_footprint(
                model, &crate::sim_job::ScenarioRef::Named(s.to_string()),
            )?.into_iter().collect(),
        }],
        None => Vec::new(),
    };
    crate::roles::resolve_roles(
        &RoleInputs {
            model,
            fit_estimate: &problem.estimate,
            fit_fixed: &fit_fixed,
            simplex_groups: &problem.simplex_groups,
            cli_estimate: &[],
            cli_fixed: &[],
            cli_fixed_files: &[],
            design: &design,
            slice: false,
            scenarios: &scenarios,
        },
        &crate::roles::FIT_RUN,
    )
    .map_err(|e| e.to_string())
}

/// The `estimated` and `fixed` fields as `fit run` wrote them before the
/// partition: the `[estimate]` names and the `[fixed]` values. Used only for a
/// fit whose roles form no partition ([`resolve_fit_roles`]), which Stage 2 of
/// the parameter-roles proposal refuses; it goes with that refusal.
fn unpartitioned_role_fields(
    problem: &config_v2::Problem,
) -> (Vec<String>, std::collections::BTreeMap<String, f64>) {
    (
        problem.estimate.keys().cloned().collect(),
        problem.fixed.resolve().unwrap_or_default().into_iter().collect(),
    )
}

/// Build the fit-level provenance sidecar for a fit.toml. Fields that require
/// I/O (model IR, data files, fit.toml bytes) are read here and digested.
/// Silent fallbacks (empty strings / empty maps) cover the read-error case so a
/// partially-written fit still produces a sidecar `camdl list` can display.
/// The readable provenance projection is written once per fit segment.
///
/// `model` is the compiled, expanded IR model. It resolves per-parameter prior
/// provenance (`fit_toml / model_ir / flat_explicit`), the observation schema
/// ([`crate::run_meta::ObsSchema`]) and the calendar axis every sidecar
/// carries.
fn build_fit_sidecar(
    config: &config_v2::FitConfig,
    fit_toml: &[u8],
    model: &ir::Model,
    data: &stream_bindings::BoundDataDigests,
) -> Result<crate::run_meta::FitSidecar, String> {
    let problem = &config.problem;
    // The pre-compiled IR (set by `cmd_fit_run_v2`, under its compile
    // settings).
    let model_src = problem.compiled_model()?;
    let model_identity = crate::resolve::model_identity_from_ir(&crate::util::load_ir(model_src)?.1);
    let fit_toml_hash = crate::hashing::sha256_hex(fit_toml);
    // gh#771: the digests of the files the RESOLVED bindings name, by source
    // (`StreamBindings`), the same ones the run identity folds. The one-file
    // `[data] file = "..."` shorthand leaves `[data.observations]` empty and
    // binds one wide TSV to every source the model declares; hashing
    // `observations` directly wrote an empty `data_hashes` for exactly that
    // config style, and the `camdl compare` data preflight (gh#713) then
    // reported such a fit as unchecked.
    // gh#75: resolve per-parameter prior provenance. This sidecar belongs to
    // the fit segment, which every method run on the problem shares
    // (`[method]` is not in the fit-level hash), so it records the problem's
    // prior resolution whichever method is running. Gating it on the method
    // made an if2 run after a pmmh run empty the record the pmmh leaf needs.
    // A sampler's `validate_priors_present` refuses an implicit flat prior
    // before it starts, so `flat_fallback` appears only on a problem no
    // sampler could run on.
    let names: Vec<String> = problem.estimate.keys().cloned().collect();
    let overrides = crate::fit::priors_precedence::prior_overrides(
        &names, &problem.estimate, model,
    );
    // Each entry's bounds are those of the transform the method's
    // `EstimatedParam` carries (`steps::domains`).
    let domains = crate::fit::steps::domains(&problem.estimate, model)?;
    let resolved_priors: Vec<crate::run_meta::ResolvedPriorEntry> =
        crate::fit::priors_precedence::resolve_priors_with_precedence(
            &names, &problem.estimate, model,
        )
        .iter()
        .map(|r| {
            let bounds = domains.get(&r.param)
                .ok_or_else(|| format!("estimate.{}: no effective domain", r.param))?
                .bounds();
            crate::fit::priors_precedence::entry(r, bounds, model, &overrides)
        })
        .collect::<Result<_, String>>()?;
    let calendar = ir::caltime::CalendarAxis::from_model(model).map_err(|e| e.to_string())?;
    Ok(crate::run_meta::FitSidecar {
        // Stamped by `write_fit_sidecar_with_config`.
        store_layout: u32::from(store::STORE_FORMAT),
        table_hashes: data.tables_hex(),
        // A fit's label is its run's, on the leaf (gh#896); the sidecar
        // describes the problem.
        label: None,
        model_identity,
        // Recorded by `write_fit_sidecar`, relative to the segment it writes.
        source_config: None,
        source_model: None,
        schema_version: crate::run_meta::FIT_META_SCHEMA_VERSION,
        fit_toml_hash,
        // gh#542: the in-memory maps stay `HashMap`; the ARTIFACT is ordered.
        // Same seam gh#519 used for `FitState` — the ordering requirement
        // belongs to `fit.meta.json`, not to the computation that feeds it.
        data_hashes: data.training_hex(),
        holdout_hashes: data.holdout_hex(),
        // Per segment, generated from the segment's partition record.
        estimated: Vec::new(),
        fixed: Default::default(),
        partition: None,
        resolved_priors,
        // Per segment: the sweep point changes `[fixed]`, so the segment's
        // own resolution fills this where the segment's sidecar is written.
        parameters_provenance: Default::default(),
        // The observation/dimension schema — a pure fold over the model's
        // expanded observation leaves; emitted for every fit (not gated on
        // Bayesian-ness — an IF2 fit's streams/dims are just as describable).
        schema: Some(crate::run_meta::ObsSchema::from_model(model)),
        // gh#585: recorded post-load by `record_training_partition` (the proof
        // must come from the code path that applied the declaration); sticky
        // across cache-hit reruns in `write_fit_sidecar`.
        training_partition: None,
        // Recorded by `run_sidecar` from the forcings and the partition.
        forcing_interpolation: Vec::new(),
        // The simulator the segment's runs integrate on. `config.dt` is the
        // step the dt rule resolved before this is built (`cmd_fit_run_v2`);
        // `resolved_dt` gives the same answer for a config read without it.
        dt: Some(crate::dt_rule::resolved_dt(model.simulation.dt, problem.config.dt)),
        integrator: Some(model.simulation.integrator.clone()),
        calendar,
    })
}

/// One side of `camdl fit diff`: what a config resolves to outside its text.
/// `model_identity` is `Err` with the reason when the model does not load
/// (a `.camdl` needs `camdlc`), so the caller can say it did not compare the
/// models rather than report them equal.
struct DiffSide {
    model_identity: Result<String, String>,
    data_hashes: std::collections::HashMap<String, String>,
}

impl DiffSide {
    fn resolve(config: &config_v2::FitConfig, compile: crate::util::CompileSettings) -> Self {
        let problem = &config.problem;
        let model_src = problem.compiled_ir.as_deref().unwrap_or(&problem.model.camdl);
        match crate::util::load_model(model_src, compile) {
            Ok((model, ir_json)) => DiffSide {
                model_identity: Ok(crate::resolve::model_identity_from_ir(&ir_json)),
                data_hashes: resolved_data_hashes(problem, Some(&model)),
            },
            Err(e) => DiffSide {
                model_identity: Err(e),
                data_hashes: resolved_data_hashes(problem, None),
            },
        }
    }
}

/// `camdl fit diff A.toml B.toml`
pub fn cmd_fit_diff(args: &crate::args::FitDiffArgs, compile: crate::util::CompileSettings) {
    use config_v2::FitConfig;
    // fit diff has no `--no-color`: its painters follow the environment and
    // whether each stream is a terminal.
    let out = crate::out::Out::new(false);

    let a_path = args.a.to_string_lossy().into_owned();
    let b_path = args.b.to_string_lossy().into_owned();
    let load = |path: &str| FitConfig::load(path).unwrap_or_else(|e| {
        out.fail(format!("error loading {}: {}", path, e));
    });
    let a_cfg = load(&a_path);
    let b_cfg = load(&b_path);

    // gh#595: the problem half is the typed `ConfigDiff` `fit table` uses, not
    // a comparison of its own. The model and data are resolved here the way a
    // run resolves them (`build_fit_sidecar`): the compiled model's identity
    // and each resolved stream's content hash.
    let a_side = DiffSide::resolve(&a_cfg, compile);
    let b_side = DiffSide::resolve(&b_cfg, compile);
    let unresolved = a_side.model_identity.is_err() || b_side.model_identity.is_err();
    // A side whose model did not load is reported below, not compared: equal
    // placeholders keep `model_changed` from asserting a change it cannot know.
    let identity = |side: &DiffSide| match (&side.model_identity, unresolved) {
        (Ok(id), false) => id.clone(),
        _ => String::new(),
    };
    let (a_id, b_id) = (identity(&a_side), identity(&b_side));
    let diff = config_diff::ConfigDiff::compare_resolved(
        &b_cfg,
        &a_cfg,
        &config_diff::ResolvedInputs { model_identity: &b_id, data_hashes: &b_side.data_hashes },
        &config_diff::ResolvedInputs { model_identity: &a_id, data_hashes: &a_side.data_hashes },
    );

    out.report_line(format_args!("diff: {} → {}", a_path, b_path));
    out.report_line("");

    out.report_line("Model:");
    let (a_model, b_model) = (&a_cfg.problem.model.camdl, &b_cfg.problem.model.camdl);
    if unresolved {
        for (path, side) in [(&a_path, &a_side), (&b_path, &b_side)] {
            if let Err(e) = &side.model_identity {
                out.report_line(format_args!("  not compared — the model of {path} did not load: {e}"));
            }
        }
        if a_model != b_model {
            out.report_line(format_args!("  {a_model} → {b_model}"));
        }
    } else if diff.model_changed {
        if a_model == b_model {
            out.report_line(format_args!("  {a_model}: compiled model differs"));
        } else {
            out.report_line(format_args!("  {a_model} → {b_model} (compiled model differs)"));
        }
    } else if a_model != b_model {
        out.report_line(format_args!("  {a_model} → {b_model} (same compiled model)"));
    } else {
        out.report_line("  (no model changes)");
    }

    out.report_line("Data:");
    let dh = &diff.data_hashes;
    for name in &dh.added { out.report_line(format_args!("  {name}: (added)")); }
    for name in &dh.removed { out.report_line(format_args!("  {name}: (removed)")); }
    for name in &dh.modified { out.report_line(format_args!("  {name}: content differs")); }
    if dh.added.is_empty() && dh.removed.is_empty() && dh.modified.is_empty() {
        out.report_line("  (no data changes)");
    }

    out.report_line("Parameters:");
    let fixed = |cfg: &FitConfig| cfg.problem.fixed.resolve().unwrap_or_default()
        .into_iter().collect::<std::collections::BTreeMap<String, f64>>();
    let lines = diff.problem_lines(&fixed(&a_cfg), &fixed(&b_cfg));
    if lines.is_empty() {
        out.report_line("  (no parameter changes)");
    }
    for line in lines {
        out.report_line(format_args!("  {line}"));
    }

    let (a_method, b_method) = (a_cfg.inference.method, b_cfg.inference.method);
    out.report_line("");
    out.report_line("Method:");
    match (&a_method, &b_method) {
        (None, None) => out.report_line("  (neither file declares a [method])"),
        (None, Some(m)) => out.report_line(format_args!("  [method]: (new) {}", m.algorithm.method_name())),
        (Some(_), None) => out.report_line("  [method]: (removed)"),
        (Some(ma), Some(mb)) => {
            // `b` is the side being described, `a` its baseline: the
            // header reads `a → b`, and so does every line under it.
            let changes = config_diff::method_setting_changes(mb, ma);
            if changes.is_empty() {
                out.report_line("  (no method changes)");
            } else {
                for c in changes {
                    out.report_line(format_args!("  {}: {} → {}", c.key, c.from, c.to));
                }
            }
        }
    }
}

// ─── camdl fit new ──────────────────────────────────────────────────────────

#[expect(clippy::disallowed_methods, reason = "`fit new` reads the fit config it copies and writes the copy outside the store; it stores nothing")]
pub fn cmd_fit_new(a: &crate::args::FitNewArgs, compile: crate::util::CompileSettings) {
    let out = crate::out::Out::new(false);
    let from = a.from.to_string_lossy().into_owned();
    let to   = a.dest.to_string_lossy().into_owned();

    if std::path::Path::new(&to).exists() {
        out.fail(format!("error: {} already exists. Choose a different name.", to));
    }

    // Read source, inject provenance
    let mut content = std::fs::read_to_string(&from).unwrap_or_else(|e| {
        out.fail(format!("error reading {}: {}", from, e));
    });

    // Check if [provenance] already exists
    if !content.contains("[provenance]") {
        // Add provenance block at the top, after the first blank line or at start
        let prov_block = format!(
            "[provenance]\nderived_from = \"{}\"\nreason = \"\"\n\n",
            from
        );
        // Insert after any leading comments
        if let Some(pos) = content.find("\n[") {
            content.insert_str(pos + 1, &prov_block);
        } else {
            content = format!("{}{}", prov_block, content);
        }
    } else {
        // Update existing provenance
        // Simple approach: just warn
        out.say(format_args!("note: {} already has [provenance]. Update derived_from manually.", to));
    }

    // How the derived file can warm-start from a run of the source, by handle.
    out.say(format_args!("  [provenance] derived_from = \"{}\"", from));
    out.say("  hint: to warm-start the derived fit from a run of the source, write under its [method]");
    out.say("        starts = { from_posterior = \"@<label>\" }   # one draw per chain");
    out.say("        starts = { from_mle = \"@<label>\" }         # every chain at its estimate");
    out.say("        (a handle is @label, a run-id prefix or the run's folder; `camdl list` shows them)");
    let _ = compile;

    std::fs::write(&to, &content).unwrap_or_else(|e| {
        out.fail(format!("error writing {}: {}", to, e));
    });

    out.say(format_args!("created {}", to));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A findings file that cannot be written is an error naming the path, not
    /// a discarded result; on a run already failing, the run's own error stays
    /// first and the unwritten record is added to it.
    #[test]
    fn an_unwritable_diagnostics_file_is_an_error() {
        use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind};
        let dir = tempfile::tempdir().unwrap();
        let collector = DiagnosticCollector::new("pmmh");
        collector.push(DiagnosticKind::InitialLoglikInfinite);

        write_diagnostics(&collector, &output::DirWriter::new(dir.path())).expect("a writable directory takes the file");
        let written = std::fs::read_to_string(dir.path().join("diagnostics.json")).unwrap();
        assert!(written.contains("initial_loglik_infinite"), "{written}");

        // A directory where the file should go: the write fails.
        let blocked = dir.path().join("blocked");
        std::fs::create_dir_all(blocked.join("diagnostics.json")).unwrap();
        let err = write_diagnostics(&collector, &output::DirWriter::new(&blocked)).expect_err("the path is a directory");
        assert!(err.starts_with("cannot write ") && err.contains("diagnostics.json"), "{err}");

        let msg = write_diagnostics_then_fail(&collector, &output::DirWriter::new(&blocked), "all chains refused".into());
        assert!(msg.starts_with("all chains refused\n"), "{msg}");
        assert!(msg.contains("its findings were not recorded: cannot write"), "{msg}");
        assert_eq!(write_diagnostics_then_fail(&collector, &output::DirWriter::new(dir.path()), "x".into()), "x");
    }

    /// The `from_mle` branch of the pfilter point refusal: a stored fit (here
    /// an nl-* leaf, whose fit_state.toml holds the estimated values) that
    /// lacks one of this fit's estimated parameters is refused; one that
    /// names them all is accepted.
    #[test]
    fn a_pfilter_from_mle_source_missing_a_parameter_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("fit_state.toml");
        std::fs::write(&state, "schema = \"camdl.fit-state/1\"\nmethod = \"nl-sbplx\"\nseed = 1\ntimestamp = \"t\"\n\
            best_loglik = -1.0\ninitial_loglik = -2.0\nbest_chain = 0\nn_chains = 1\n\
            [start_values]\nbeta = 0.4\n").unwrap();
        let rule = starts::ChainStarts::Point(starts::Point::FromMle {
            source: starts::Handle("@nl".into()),
        });
        let err = refuse_pfilter_point_source_missing(&rule, &state, &["beta", "gamma"])
            .expect_err("gamma is missing from the source");
        assert!(err.contains("gives no value for gamma") && err.contains("from_mle"), "{err}");
        refuse_pfilter_point_source_missing(&rule, &state, &["beta"]).unwrap();
    }

    // ── gh#191: per-method capability gate on `fit run` ─────────────

    /// A chain_binomial IF2 method — the inference path that carries no
    /// real reservoir state (gh#191).
    fn chain_binomial_if2_method() -> config_v2::Algorithm {
        config_v2::Algorithm::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 1,
            particles: 10,
            iterations: 1,
            cooling: 0.7,
            cooling_target_iters: 1,
            loglik_eval: config_v2::LoglikEvalConfig::default(),
            gate: config_v2::GateConfig::default(),
            dt_check: config_v2::DtCheckConfig::default(),
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            perturbation: Default::default(),
        }
    }

    /// An `mh`-on-ODE method carrying `burnin_dt`, for the gh#449 warm-up-step
    /// checks. Only `backend` and `burnin_dt` matter to the gate.
    fn mh_method_with_burnin_dt(burnin_dt: Option<f64>) -> config_v2::Algorithm {
        config_v2::Algorithm::Mh {
            backend: crate::run_meta::InferenceBackend::Ode,
            chains: 1,
            iterations: 1,
            burn_in: None,
            thin: None,
            adapt: true,
            adapt_start: 300,
            burnin_dt,
            dt_check: config_v2::DtCheckConfig::default(),
            proposal: Default::default(),
            prior_only: Default::default(),
        }
    }

    /// Load a golden envelope (`{ model: {...} }`) and fill null param
    /// values so the model compiles — values are irrelevant to the
    /// capability scan.
    fn compiled_golden(rel: &str) -> sim::CompiledModel {
        let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let envv: serde_json::Value =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"));
        let mut model: ir::Model = serde_json::from_value(envv["model"].clone())
            .unwrap_or_else(|e| panic!("deserialize {path}: {e}"));
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        sim::CompiledModel::new(model).unwrap_or_else(|e| panic!("compile {path}: {e:?}"))
    }

    #[test]
    fn fit_run_rejects_real_compartments_on_chain_binomial_method() {
        // gh#191: `fit run` never gated the model against the per-method
        // backend, so a real-compartment (ODE-coupled) model on a
        // chain_binomial inference method was silently mis-fit — the filter
        // loops freeze the real reservoir at its init value. The fit-run
        // path must REJECT it with the REAL_COMPARTMENTS message, naming the
        // offending method.
        let compiled = compiled_golden("ocaml/golden/sir_reservoir_mixed.ir.json");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::REAL_COMPARTMENTS),
            "fixture must actually require REAL_COMPARTMENTS"
        );
        let method = chain_binomial_if2_method();
        let err = gate_run_method_against_model(&method, &compiled, 1.0)
            .expect_err("real-coupled model on a chain_binomial fit method must be rejected");
        assert!(err.contains("gh#191"), "should cite the tracking issue: {err}");
        assert!(
            err.contains("frozen"),
            "should explain the frozen-reservoir reason: {err}"
        );
        assert!(err.contains("if2"), "should name the offending method: {err}");
    }

    #[test]
    fn fit_run_accepts_balance_on_chain_binomial_method() {
        // gh#192: a `balance{}` model is a chain-binomial-only construct the
        // inference loops apply via step_one — `fit run` accepts it, so the
        // gate must too (and not falsely reject it once wired in). Inject a
        // balance{} block into the sir_basic golden (target = integer
        // compartment R) so the only required capability is BALANCE.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../ocaml/golden/sir_basic.ir.json"
        );
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let envv: serde_json::Value =
            serde_json::from_str(&json).expect("parse sir_basic envelope");
        let mut model: ir::Model = serde_json::from_value(envv["model"].clone())
            .expect("deserialize sir_basic model");
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        model.balance = Some(ir::model::BalanceSpec {
            target: "R".to_string(),
            expr: ir::expr::Expr::param("N0"),
        });
        let compiled = sim::CompiledModel::new(model).expect("compile sir_basic + balance");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::BALANCE),
            "fixture must actually require BALANCE"
        );
        let method = chain_binomial_if2_method();
        gate_run_method_against_model(&method, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("fit run must ACCEPT a balance{{}} model on a chain_binomial method: {e}")
        });
    }

    // ── gh#449: recurring-fire collision guard on the FIT path ────

    /// gh#449 (follow-up to gh#447). The recurring-fire collision guard was
    /// added to `CompiledModel::validate_schedule`, which the three forward
    /// backends call at entry — but nothing on the inference/fit path calls it,
    /// and the inference producers fire one intervention once per grid step, so
    /// two recurring fires on one step are one. So a coarse-`dt` `fit` silently
    /// dropped recurring fires,
    /// which is the item-23 silent-wrong condition surviving on the inference
    /// cells — against the "every backend x method cell works or fails loudly"
    /// doctrine.
    ///
    /// `seir_seasonal_importation` has a recurring importation on a 365.25-day
    /// period; at dt = 400 consecutive fires round to the same integrator step.
    #[test]
    fn fit_run_rejects_coarse_dt_that_drops_recurring_fires() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        // The fixture must actually carry a recurring schedule, or this test
        // passes for the wrong reason.
        assert!(
            compiled.model.interventions.iter().any(|iv| matches!(
                iv.fire.schedule(),
                Some(ir::intervention::InterventionSchedule::Recurring(_))
            )),
            "fixture must carry a recurring schedule"
        );
        // Sanity: the forward path already rejects this dt. The bug is that the
        // fit path does not.
        assert!(
            compiled
                .validate_recurring_dt_collisions(
                    compiled.model.simulation.t_start,
                    sim::effect_plan::FireClock::Grid { dt: 400.0 },
                )
                .is_err(),
            "precondition: dt=400 must collide on this model"
        );

        let method = chain_binomial_if2_method();
        let err = gate_run_method_against_model(&method, &compiled, 400.0)
            .expect_err("a fit dt that drops recurring fires must be rejected, not silently merged");
        assert!(err.contains("if2"), "should name the offending method: {err}");
        assert!(
            err.contains("silently dropped"),
            "should carry the collision diagnostic: {err}"
        );
    }

    /// Negative control: a `dt` finer than the period must still be accepted on
    /// the same model, so the guard above is not just rejecting everything.
    #[test]
    fn fit_run_accepts_dt_finer_than_the_recurrence_period() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        let method = chain_binomial_if2_method();
        gate_run_method_against_model(&method, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("dt=1 is far finer than the 365.25-day period and must be accepted: {e}")
        });
    }

    /// An `mh` method (ODE-backed) with a coarse `burnin_dt` is accepted: the
    /// ODE fires every effect at its own instant (proposal 2026-10-04, D2), so
    /// a warm-up step coarser than the recurrence period drops no fire. The
    /// coarse warm-up still clips onto each effect instant (`run_ode`).
    #[test]
    fn fit_run_accepts_coarse_burnin_dt_on_an_ode_method() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        let method = mh_method_with_burnin_dt(Some(400.0));
        gate_run_method_against_model(&method, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("an ODE method fires each effect at its own instant; burnin_dt=400 drops nothing: {e}")
        });
        // And at a coarse dt too: the check is a Grid-clock check.
        gate_run_method_against_model(&method, &compiled, 400.0).unwrap_or_else(|e| {
            panic!("an ODE method fires each effect at its own instant; dt=400 drops nothing: {e}")
        });
    }

    /// The same method with no `burnin_dt` set passes too.
    #[test]
    fn fit_run_accepts_an_mh_method_without_burnin_dt() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        let method = mh_method_with_burnin_dt(None);
        gate_run_method_against_model(&method, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("an mh method with no burnin_dt must be accepted at dt=1: {e}")
        });
    }

    // ── gh#771: the sidecar hashes the RESOLVED stream set ─────────

    /// Deserialize a golden IR envelope into an `ir::Model` (no compile —
    /// only the declared observation names are read here).
    /// A compiled model that cannot be loaded fails the sidecar rather than
    /// recording an empty model identity.
    #[test]
    fn a_sidecar_whose_compiled_model_does_not_load_is_an_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let toml_src = "[model]\ncamdl = \"m.camdl\"\n[data.observations]\n\
                        [estimate]\nbeta = { bounds = [0.01, 0.4] }\n[fixed]\nN0 = 1000\n";
        let fit_path = tmp.path().join("fit.toml");
        std::fs::write(&fit_path, toml_src).unwrap();
        let mut config = config_v2::FitConfig::from_toml_str(toml_src).unwrap();
        config.problem.compiled_ir = Some(tmp.path().join("absent.ir.json").to_string_lossy().into_owned());
        let model = golden_model("ocaml/golden/seir_observations.ir.json");
        let err = build_fit_sidecar(&config, toml_src.as_bytes(), &model, &Default::default()).unwrap_err();
        assert!(err.contains("absent.ir.json"), "{err}");
        config.problem.compiled_ir = None;
        let err = build_fit_sidecar(&config, toml_src.as_bytes(), &model, &Default::default()).unwrap_err();
        assert!(err.contains("was not compiled"), "{err}");
    }

    /// The absolute path of a committed golden IR, as a fit's `compiled_ir`.
    fn golden_path(rel: &str) -> String {
        format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel)
    }

    fn golden_model(rel: &str) -> ir::Model {
        let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let envv: serde_json::Value =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"));
        serde_json::from_value(envv["model"].clone())
            .unwrap_or_else(|e| panic!("deserialize {path}: {e}"))
    }

    /// `fit.meta.json`'s `data_hashes` is keyed by source in every `[data]`
    /// form. Under the one-file shorthand it was keyed by leaf, so a model
    /// whose `cases[p in patch]` block expands to `cases_urban`/`cases_rural`
    /// recorded two keys there and the one key `cases` under
    /// `[data.observations]`, and `compare`, which reads the sidecar by the
    /// keys the config binds, looked for one of the two and missed it.
    #[test]
    fn sidecar_data_hashes_are_keyed_by_source_under_the_shorthand() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_path = tmp.path().join("cases_long.tsv");
        std::fs::write(&data_path, b"time\tpatch\tcases\n7\turban\t12\n7\trural\t3\n").unwrap();
        let fit_path = tmp.path().join("fit.toml");
        let toml_src = format!(
            "[model]\ncamdl = \"no-such-model.ir.json\"\n\
             [data]\nfile = \"{}\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n[fixed]\nrho = 0.5\n\
             [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 1\nparticles = 10\niterations = 1\ncooling = 0.7\n",
            data_path.display());
        std::fs::write(&fit_path, &toml_src).unwrap();
        let mut config = config_v2::FitConfig::from_toml_str(&toml_src).unwrap();
        config.problem.compiled_ir = Some(golden_path("ocaml/golden/sir_two_patch_long_obs.ir.json"));
        let model = golden_model("ocaml/golden/sir_two_patch_long_obs.ir.json");
        let data = read_data_digests(config.problem.data_spec().unwrap(), &model).unwrap();
        let sidecar = build_fit_sidecar(&config, &std::fs::read(&fit_path).unwrap(), &model, &data).unwrap();
        assert_eq!(sidecar.data_hashes.keys().collect::<Vec<_>>(), vec!["cases"],
            "one entry, keyed by the source both leaves read: {:?}", sidecar.data_hashes);
    }

    /// `fit.meta.json` carries the model's calendar axis
    /// (`docs/dev/temporal-architecture.md` §5.3), so the fit's model times can
    /// be dated without the model source.
    #[test]
    fn sidecar_records_the_models_calendar() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_path = tmp.path().join("cases_long.tsv");
        std::fs::write(&data_path, b"time\tpatch\tcases\n7\turban\t12\n7\trural\t3\n").unwrap();
        let fit_path = tmp.path().join("fit.toml");
        let toml_src = format!(
            "[model]\ncamdl = \"no-such-model.ir.json\"\n\
             [data]\nfile = \"{}\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n[fixed]\nrho = 0.5\n\
             [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 1\nparticles = 10\niterations = 1\ncooling = 0.7\n",
            data_path.display());
        std::fs::write(&fit_path, &toml_src).unwrap();
        let mut config = config_v2::FitConfig::from_toml_str(&toml_src).unwrap();
        config.problem.compiled_ir = Some(golden_path("ocaml/golden/sir_two_patch_long_obs.ir.json"));
        let mut model = golden_model("ocaml/golden/sir_two_patch_long_obs.ir.json");
        model.origin = Some("2020-03-01".into());
        let data = read_data_digests(config.problem.data_spec().unwrap(), &model).unwrap();
        let sidecar = build_fit_sidecar(&config, &std::fs::read(&fit_path).unwrap(), &model, &data).unwrap();
        let seg = tmp.path().join("seg");
        std::fs::create_dir_all(&seg).unwrap();
        crate::run_meta::store_fit_sidecar(&seg, &sidecar).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(seg.join("fit.meta.json")).unwrap()).unwrap();
        assert_eq!(
            written["calendar"],
            serde_json::json!({"days_per_unit": 1.0, "origin": "2020-03-01", "time_unit": model.time_unit}),
            "fit.meta.json: {written:#}"
        );
    }

    /// `fit.meta.json` records the full SHA-256 of every bound file, training
    /// and `[data.holdout]`, by source: the digests the run identity folds.
    #[test]
    fn sidecar_records_full_training_and_holdout_digests() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let train = tmp.path().join("train.tsv");
        std::fs::write(&train, b"time\tpatch\tcases\n7\turban\t12\n7\trural\t3\n").unwrap();
        let holdout = tmp.path().join("holdout.tsv");
        std::fs::write(&holdout, b"time\tpatch\tcases\n14\turban\t18\n14\trural\t6\n").unwrap();
        let fit_path = tmp.path().join("fit.toml");
        let toml_src = format!(
            "[model]\ncamdl = \"no-such-model.ir.json\"\n\
             [data.observations]\ncases = \"{}\"\n[data.holdout]\ncases = \"{}\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n[fixed]\nrho = 0.5\n\
             [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 1\nparticles = 10\niterations = 1\ncooling = 0.7\n",
            train.display(), holdout.display());
        std::fs::write(&fit_path, &toml_src).unwrap();
        let mut config = config_v2::FitConfig::from_toml_str(&toml_src).unwrap();
        config.problem.compiled_ir = Some(golden_path("ocaml/golden/sir_two_patch_long_obs.ir.json"));
        let model = golden_model("ocaml/golden/sir_two_patch_long_obs.ir.json");
        let data = read_data_digests(config.problem.data_spec().unwrap(), &model).unwrap();
        let sidecar = build_fit_sidecar(&config, &std::fs::read(&fit_path).unwrap(), &model, &data).unwrap();
        let sha = |p: &std::path::Path| crate::hashing::sha256_hex(&std::fs::read(p).unwrap());
        assert_eq!(sidecar.data_hashes, std::collections::BTreeMap::from([("cases".into(), sha(&train))]));
        assert_eq!(sidecar.holdout_hashes, std::collections::BTreeMap::from([("cases".into(), sha(&holdout))]));
    }

    #[test]
    fn sidecar_hashes_every_stream_under_the_single_file_shorthand() {
        // gh#771: `[data] file = "..."` binds one wide TSV to every
        // observation source the model declares (here each stream is its own
        // source). Fit IDENTITY resolved that shorthand; the sidecar hashed
        // `[data.observations]` — empty under the shorthand — so
        // `fit.meta.json` carried no `data_hashes` at all, and the
        // `camdl compare` data preflight reported the fit as unchecked.
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_path = tmp.path().join("wide.tsv");
        std::fs::write(&data_path, b"time\tweekly_cases\tdetection\n1\t5\t0.4\n")
            .expect("write data");
        let fit_path = tmp.path().join("fit.toml");
        // `.ir.json` (not `.camdl`) and non-existent: `build_fit_sidecar`
        // reads the compiled model (`compiled_ir`, the golden the model under
        // test is loaded from), as `fit run` sets it.
        let toml_src = format!(
            r#"
[model]
camdl = "no-such-model.ir.json"

[data]
file = "{}"

[estimate]
beta = {{ bounds = [0.01, 0.4] }}

[fixed]
N0 = 1000

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 10
iterations = 1
cooling = 0.7
"#,
            data_path.display()
        );
        std::fs::write(&fit_path, &toml_src).expect("write fit.toml");
        let mut config = config_v2::FitConfig::from_toml_str(&toml_src)
            .expect("fixture fit.toml should parse");
        config.problem.compiled_ir = Some(golden_path("ocaml/golden/seir_observations.ir.json"));

        let model = golden_model("ocaml/golden/seir_observations.ir.json");
        let stream_names: Vec<String> =
            model.observations.iter().map(|o| o.name.clone()).collect();
        assert_eq!(stream_names.len(), 2, "fixture must declare two streams");

        let data = read_data_digests(config.problem.data_spec().unwrap(), &model).unwrap();
        let sidecar = build_fit_sidecar(
            &config, &std::fs::read(&fit_path).unwrap(), &model, &data).unwrap();

        let expected = crate::hashing::sha256_hex(&std::fs::read(&data_path).unwrap());
        assert_eq!(
            sidecar.data_hashes.len(), stream_names.len(),
            "the shorthand must produce one digest per declared stream; got {:?}",
            sidecar.data_hashes);
        for name in &stream_names {
            assert_eq!(
                sidecar.data_hashes.get(name).map(String::as_str),
                Some(expected.as_str()),
                "stream `{name}` should carry the wide file's digest; got {:?}",
                sidecar.data_hashes);
        }
    }

    /// `fit.meta.json` belongs to the fit segment, which every method run on
    /// the problem shares (`[method]` is not in the fit-level hash). Its prior
    /// resolution is a fact about the problem — `[estimate]` and the model —
    /// so a pmmh run and an if2 run on the same problem must write the same
    /// `resolved_priors`. When it was written only for a sampler, an if2 run
    /// after a pmmh run emptied it, and every later `fit table` / `fit
    /// summary` reported the pmmh leaf as a dropped-provenance bug (gh#75).
    #[test]
    fn resolved_priors_do_not_depend_on_the_method() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = golden_model("ocaml/golden/seir_observations.ir.json");
        let sidecar_for = |method: &str| {
            let toml_src = format!(r#"
[model]
camdl = "no-such-model.ir.json"

[data.observations]

[estimate]
beta = {{ bounds = [0.01, 0.4], prior = {{ log_normal = {{ mu = -1.0, sigma = 0.5 }} }} }}

[fixed]
N0 = 1000

[method]
{method}
"#);
            let fit_path = tmp.path().join("fit.toml");
            std::fs::write(&fit_path, &toml_src).expect("write fit.toml");
            let mut config = config_v2::FitConfig::from_toml_str(&toml_src)
                .expect("fixture fit.toml should parse");
            config.problem.compiled_ir = Some(golden_path("ocaml/golden/seir_observations.ir.json"));
            // The data digests are not under test here.
            build_fit_sidecar(&config, &std::fs::read(&fit_path).unwrap(), &model,
                &Default::default()).unwrap()
        };
        let pmmh = sidecar_for(
            "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 1\n\
             particles = 10\niterations = 10\nburn_in = 5");
        let if2 = sidecar_for(
            "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
             particles = 10\niterations = 1\ncooling = 0.7");
        let entries = |s: &crate::run_meta::FitSidecar| s.resolved_priors.iter()
            .map(|e| (e.param.clone(), e.source.clone())).collect::<Vec<_>>();
        assert_eq!(entries(&pmmh), vec![("beta".to_string(), "fit_toml".to_string())],
            "a sampler's problem records each estimated parameter's prior source");
        assert_eq!(entries(&if2), entries(&pmmh),
            "the fit-level record is the problem's, whichever method wrote it last");
    }
}

