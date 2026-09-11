//! `camdl fit` — structured inference workflow.
//!
//! `camdl fit run FIT.toml [--seed N] [--starts SPEC] [--label "..."]
//! [--force]` runs the file's one `[method]` on its problem and stores the
//! result as a content-addressed leaf; `fit summary`, `fit predict`, `fit
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
fn resolve_anchors_into_temp_ir(
    ir_path: &str,
    config: &config_v2::Problem,
) -> Result<String, String> {
    let src = std::fs::read_to_string(ir_path)
        .map_err(|e| format!("cannot read compiled IR {ir_path}: {e}"))?;
    let mut env = ir::envelope_from_str(&src)
        .map_err(|e| format!("IR load error from {ir_path}: {e}"))?;
    if !crate::obs_anchor::model_is_anchored(&env.model) {
        return Ok(ir_path.to_string());
    }
    let dt0 = env.model.simulation.dt.unwrap_or(1.0);
    let (first, last) = crate::obs_anchors_from_config(&env.model, config, dt0)
        .map_err(|e| format!("resolving this model's observation anchors from [data]: {e}"))?;
    let moved = crate::obs_anchor::substitute(
        &mut env.model,
        ir::anchor::ObsAnchorTimes { first, last },
    )?;
    crate::obs_anchor::report(&moved, &env.model);

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

pub mod cas;  // gh#147 M3.2: fit-stage CAS identity (resolve_fit_stage)
pub mod coeff_guard;  // gh#342 P4: NUTS guard — param reaching a coefficient only via an init
pub mod config_v2;
pub mod loglik;  // gh#280: LoglikType — the single typed source for loglik class
pub mod state;
pub mod provenance;
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
pub use fit_summary::cmd_fit_summary;
pub mod pmmh;
// Likelihood-noise preflight shared by the pseudo-marginal stages: the
// measured log L-hat spread and the acceptance ceiling it implies (gh#764).
pub mod pf_noise;
pub mod pgas;
pub mod nuts;  // gh#275 Phase 2: nuts on ode
pub mod trace_writer;
pub mod synthetic;
pub mod gating;
pub mod chain_diagnostics;  // gh#406: per-chain loglik outlier z-scores (read-side)
pub mod cross_chain_compat; // gh#785: log p(x_j | θ_i) across the path-augmented chains
pub mod path_renewal;       // gh#791: trajectory renewal resolved in time, and its two scalars
pub mod latent_convergence; // gh#822: R̂/ESS of every latent state at every substep across chains
pub mod filter_ess;         // gh#685: the conditional filter's ESS at every observation, pooled over sweeps
pub mod row_convergence;    // gh#794: R̂/ESS of the value in one predictive/quantity row
pub mod dt_check;
pub mod init;
pub mod starts;         // the `starts` rule: where a method's chains begin
pub mod chain_starts;
pub mod loglik_eval;
pub mod methods;
#[cfg(feature = "ode")]
pub mod nlopt_stage;
pub mod handle;   // gh#322: fit handles (@label / hash / run-dir / fit.toml) → segment
pub mod joint;    // gh#322: keyed-joint (θ, X) read — LatentPath classifier + join
pub mod failures; // the deterministic-failure record a predictive run writes as report.json
pub mod predict;  // `camdl fit predict`: free-forward posterior predictive verb + types
pub mod contrasts; // gh#322: counterfactual `contrasts {}` two-arm replay reducer (stage C)

/// `camdl fit methods` — print the supported (algorithm, backend) pairs.
/// Reads from `methods::METHODS`, the single source of truth.
pub fn cmd_fit_methods() {
    print!("{}", methods::render_matrix());
}

// ─── New `camdl fit run` entry point (config_v2) ────────────────────────────

/// gh#191: the model-capability gate must run on the fit-run path, against
/// the method's declared simulation backend. Returns the offending message
/// prefixed with the method's name so the user knows where to look.
fn gate_run_method_against_model(
    algorithm: &config_v2::Algorithm,
    compiled: &sim::CompiledModel,
    dt: f64,
) -> Result<(), String> {
    let stage_name = algorithm.method_name();
    {
        let stage = algorithm;
        if let Err(msg) = methods::check_model_capabilities(stage.backend(), compiled) {
            return Err(format!("method '{}': {}", stage_name, msg));
        }
        // gh#449: the recurring-fire collision guard (gh#447) lived only at the
        // three forward backends' entry points; the inference path calls
        // `resolve_fire_steps` directly, whose dedup `BTreeSet` is where a
        // colliding fire is silently dropped. Check it here, per stage, so a
        // coarse-`dt` fit fails loudly instead of quietly losing fires.
        //
        // Two step sizes reach the integrator on this path, and BOTH can
        // collide. `burnin_dt` (gh#396) is the more dangerous of the two: it
        // exists precisely to be COARSER than `dt` on the unscored warm-up, so
        // a schedule safe at `dt` can still drop fires during warm-up. It
        // postdates gh#447, which is why the original guard never considered
        // it.
        let burnin_dt = match stage {
            config_v2::Algorithm::Mh { burnin_dt, .. }
            | config_v2::Algorithm::Nuts { burnin_dt, .. } => *burnin_dt,
            _ => None,
        };
        for (label, step) in [("dt", Some(dt)), ("burnin_dt", burnin_dt)] {
            let Some(step) = step else { continue };
            if let Err(e) = compiled.validate_recurring_dt_collisions(step) {
                return Err(format!("method '{}' ({} = {}): {}", stage_name, label, step, e));
            }
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

/// How often the heartbeat thread rewrites a running stage's `progress.json`.
/// Wall-clock and fixed, deliberately independent of step cadence: one
/// national-scale PGAS sweep can take minutes, so a step-boundary heartbeat
/// would be as stale as the trace it is meant to substitute for (gh#278).
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// The liveness/progress heartbeat one stage of `algorithm` runs under: what
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
/// - NUTS — one warm-up or sampling iteration, counted end to end, so `warmup`
///   is the burn-in boundary and `warmup + samples` the total.
/// - IF2 — one cooling iteration; a search, so a single `optimizing` phase.
/// - `nl-sbplx` / `nl-bobyqa` — one objective evaluation against `max_evals`.
/// - `pfilter` — replicates of one point, which is neither an MCMC phase nor a
///   search. It gets an inert guard and writes no `progress.json` at all,
///   rather than a counter labelled with a phase it is not in.
///
/// A method whose options fail to parse also gets an inert guard: the dispatch
/// arm reports that error and exits, and there is no run to report progress
/// for.
fn stage_heartbeat(
    stage_dir: &std::path::Path,
    algorithm: &config_v2::Algorithm,
) -> io::HeartbeatGuard {
    use config_v2::Algorithm as A;
    let dir = stage_dir.to_path_buf();
    let held = io::HeartbeatGuard::new;
    match algorithm {
        A::PGAS { .. } => match pgas::PgasStageOpts::from_algorithm(algorithm) {
            Ok(o) => held(io::Heartbeat::mcmc(
                dir, o.burn_in as u64, o.n_sweeps as u64, HEARTBEAT_INTERVAL, o.n_chains)),
            Err(_) => io::HeartbeatGuard::inert(),
        },
        // The roster is 0, so no `chains` block is written: gh#751 wired
        // per-chain reporting into PGAS only, and a table of rows that never
        // fill would say less than no table at all.
        A::PMMH { .. } | A::Mh { .. } => match pmmh::PmmhStageOpts::from_algorithm(algorithm) {
            Ok(o) => held(io::Heartbeat::mcmc(
                dir, o.burn_in as u64, o.n_steps as u64, HEARTBEAT_INTERVAL, 0)),
            Err(_) => io::HeartbeatGuard::inert(),
        },
        A::Nuts { warmup, samples, .. } => held(io::Heartbeat::mcmc(
            dir, *warmup as u64, (warmup + samples) as u64, HEARTBEAT_INTERVAL, 0)),
        A::IF2 { iterations, .. } =>
            held(io::Heartbeat::optimizing(dir, *iterations as u64, HEARTBEAT_INTERVAL)),
        A::NlSbplx(c) | A::NlBobyqa(c) =>
            held(io::Heartbeat::optimizing(dir, c.max_evals as u64, HEARTBEAT_INTERVAL)),
        A::PFilter { .. } => io::HeartbeatGuard::inert(),
    }
}

/// Abandon the running stage: record the terminal `failed` state in its
/// `progress.json`, carrying the same sentence the user is about to read, then
/// print it and exit non-zero.
///
/// `std::process::exit` runs no destructor, so the guard's `Drop` backstop
/// cannot see these paths. Routing every fatal exit in the dispatch through one
/// function is what makes "a stage always leaves a terminal record" true rather
/// than nearly true — and it is why a reader holding only `progress.json`
/// learns what a reader of stderr learns.
fn abandon_stage(heartbeat: &io::HeartbeatGuard, message: String) -> ! {
    heartbeat.failed(message.as_str());
    eprintln!("{}", message);
    std::process::exit(1);
}

pub fn cmd_fit_run_v2(a: &crate::args::FitRunArgs) {
    use config_v2::{Algorithm, FitConfig};

    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();  // gh#audit-H5
    // allow_degenerate_rates is set from the loaded `[config]` below (it's a
    // keyed config field, not a CLI flag — gh#189: a CLI override would bypass
    // the fit-identity hash).
    // gh#162: a fit nests Rayon parallelism (chains × particle filter) on the
    // global pool, which otherwise defaults to ALL logical cores regardless of
    // `chains`. Cap it from `--parallel` / CAMDL_PARALLEL (0 = all cores) so the
    // thread budget is explicit and matches pfilter/profile/survey/batch.
    // `build_global` is one-shot per process; ignore the already-initialised
    // Err so re-entry (tests) stays safe. Mirrors pfilter.rs.
    if a.parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(a.parallel)
            .build_global();
    }
    // Removed flags fail loudly before any work, each naming its replacement
    // (alpha posture: a redirect, not a shim). Every one present is reported,
    // so `--stage posterior --init from_prior` is answered in one message.
    if let Some(msg) = removed_flag_message(a) {
        eprintln!("error: {}", msg);
        std::process::exit(1);
    }
    let fit_path              = a.config.to_string_lossy().into_owned();
    let base_seed             = a.seed.unwrap_or(1);
    let force                 = a.force;

    // gh#540: every CLI flag that changes what the method computes or stores,
    // collected once so the method identity can see all of them. Applied to
    // the in-memory method below, before the CAS claim — the dispatch sites
    // no longer override anything, so there is no route around the key.
    //
    // `--tempering` is validated HERE rather than at dispatch: it now feeds the
    // identity, and a malformed value should fail before a run_id is computed
    // from it, not after the claim.
    if let Some(t) = &a.tempering {
        if t.is_empty() || (t[0] - 1.0).abs() > 1e-9 {
            eprintln!("error: --tempering must start with β=1.0 (cold chain). \
                       Got: {:?}", t);
            std::process::exit(1);
        }
    }
    if let Some(r) = a.rho {
        if !(0.0..1.0).contains(&r) {
            eprintln!("error: --rho must be in [0, 1). Got: {}", r);
            std::process::exit(1);
        }
    }
    let cli_overrides = crate::fit::config_v2::CliStageOverrides {
        tempering:            a.tempering.clone(),
        max_tree_depth:       a.max_tree_depth,
        trajectory_warmup:    a.trajectory_warmup,
        csmc_sweeps_per_nuts: a.csmc_sweeps_per_nuts,
        n_trajectories:       a.n_trajectories,
        diagonal_mass:        a.diagonal_mass,
        no_nuts:              a.no_nuts,
        no_ancestor_sampling: a.no_ancestor_sampling,
        no_adapt:             a.no_adapt,
        adapt_start:          a.adapt_start,
        rho:                  a.rho,
        cooling_target_iters: a.cooling_target_iters,
        decibans_thresh:      a.decibans_thresh,
        no_dt_check:          a.no_dt_check,
        dt_check_halvings:    a.dt_check_halvings,
        dt_check_strict:      a.dt_check_strict,
        binomial:             a.binomial,
        record_ancestry:      a.record_ancestry,
        record_prequential:   a.record_prequential,
    };
    let sweep_specs: Vec<(String, Vec<f64>)> = a.sweep.iter()
        .map(|s| (s.name.clone(), s.grid.expand()))
        .collect();

    // Load the file: the problem half and the one `[method]`.
    let mut config = FitConfig::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // `fit run` runs the file's one method and refuses a file that has none,
    // naming the table. Every other reader takes the problem alone.
    if let Err(e) = config.method() {
        eprintln!("error: {}: {}", fit_path, e);
        std::process::exit(1);
    }
    // gh#audit-C6 / gh#189: degenerate-rate handling is a keyed `[config]` field
    // (folds into the fit-identity hash via the config blob), set before any rate
    // evaluation. Was a CLI flag, which silently bypassed the run_id.
    sim::eval_stats::set_allow_degenerate_rates(config.problem.config.allow_degenerate_rates);

    // gh#656: `--emit-every` reaches exactly one thing on this command — the
    // `[synthetic]` generator, which is the only fit path where the emission
    // cadence determines data that is then fitted. A fit against REAL data
    // scores at its data files' own times and never consults `emit_schedule`,
    // so the flag would silently do nothing; refuse and say why rather than
    // leaving the user to wonder which cadence they got.
    //
    // Deliberately NOT written into `config`: the fit-identity hash serializes
    // that document, so parking it there would re-key a real-data fit over a
    // knob that fit cannot see. The override travels as an argument to
    // generation, and keys the fit the honest way — through the generated
    // data's bytes, which `FitDigest.data` already hashes.
    let emit_every = crate::emit_every::EmitEvery::from_cli_specs(&a.emit_every)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    if emit_every.is_some() && !config.problem.is_synthetic_fit() {
        eprintln!(
            "error: --emit-every sets the cadence at which SYNTHETIC \
             observations are generated, and {} {}.\n  \
             A fit against real data scores each stream at its own data file's \
             times — `emit_schedule` never enters the likelihood — so this flag \
             would change nothing.\n  \
             Drop the flag, or add a `[synthetic]` block to generate data.",
            fit_path,
            if config.problem.synthetic.is_some() {
                "declares `[data]` beside `[synthetic]`, so it fits the real data"
            } else {
                "declares no `[synthetic]` block"
            }
        );
        std::process::exit(1);
    }

    // gh#514 / gh#540: the CLI overrides — `--starts` and the sampler knobs —
    // all change where the chains begin or what the method computes, and
    // therefore the stored output. They are written INTO the method here,
    // before `cas::fit_level_hash` and before the claim, so a different
    // override is a different artifact. Leaving them unset leaves the toml's
    // values in place.
    {
        let method = config.inference.method.as_mut().expect("checked above");
        if !cli_overrides.is_empty() {
            method.algorithm.apply_cli_overrides(&cli_overrides);
        }
        if let Some(starts) = &a.starts {
            method.starts = Some(starts.clone());
        }
    }

    // A sourced `starts` rule reads a stored fit or a file: resolve the handle
    // now, before the model is compiled, so a bad handle is refused at once
    // and the dep folded into every leaf is one resolution. The source's
    // convergence verdict is checked here (`--allow-nonconverged-source`
    // lifts the refusal) — the one seam an upstream fit is consumed. A bare
    // rule has nothing to resolve until the default is known, below.
    let early_resolved_starts: Option<chain_starts::ResolvedStarts> = {
        let method = config.inference.method.as_ref().expect("checked above");
        match &method.starts {
            Some(rule) if rule.source().is_some() => Some(
                chain_starts::resolve_starts(rule, a.allow_nonconverged_source).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }),
            ),
            _ => None,
        }
    };

    // Compile `model.camdl` → IR EXACTLY ONCE for the whole fit. Every
    // per-(cell × sweep point) `FitRunConfig::build` then loads this
    // pre-compiled IR instead of re-invoking camdlc per unit (a swept fit
    // otherwise recompiled the model dozens of times). The resolved path is
    // recorded on `config.problem.compiled_ir`; `model.camdl` is left
    // untouched so the fit's content hash still hashes the original `.camdl`
    // source bytes (identity is unchanged by the hoist). The temp IR persists
    // for the process — `resolve_ir_path`'s returned path is not a drop guard.
    //
    // gh#439 A2: only `nuts` on the `ode` backend reads the WrtPop state-Jacobian
    // (`rate_state_grad` / `projection_state_grad`, via the ODE forward-sensitivity
    // gradient in `ode_grad::det_grad`). Otherwise compile lean
    // (`--no-state-grad`), which omits the dense ~O(G^3) Jacobian that dominates
    // coupled-model IR. The bit is folded into the IR-cache key, so a lean entry is
    // never reused for a nuts+ode fit (and run identity is gradient-independent, so
    // lean vs full share the same model digest).
    let needs_state_grad = config.needs_state_grad();
    let (compiled_ir, _ir_tmp) = crate::util::resolve_ir_path(&config.problem.model.camdl, needs_state_grad)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // gh#616: if the model declares observation anchors, resolve them ONCE here
    // — from this fit's own `[data]` — and re-emit the compiled IR with the
    // anchors substituted, so every downstream loader (the runner, each sweep
    // point, the archived copy) reads the SAME resolved model.
    //
    // The alternative, substituting in memory at each of those loads, would put
    // the resolution in half a dozen places that must agree; this puts it in
    // one. A FRESH temp is written rather than mutating `compiled_ir` in place,
    // because that path can be a user-supplied `.ir.json` (`resolve_ir_path`
    // returns it unchanged) and a command must never rewrite the user's file.
    //
    // Fit identity is unaffected and already correct: it hashes the original
    // `.camdl` source plus the data digests, so two data vintages against one
    // anchored model key differently through the data, exactly as two vintages
    // of any fit do.
    let compiled_ir = resolve_anchors_into_temp_ir(&compiled_ir, &config.problem)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // Load model and validate completeness (from the pre-compiled IR).
    let (model, _) = crate::util::load_model(&compiled_ir).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#33: expand `[fixed] from_scenario = "name"` into the inline
    // values map by looking up the named scenario in the model. Must
    // happen after model load but before validate, so the every-param-
    // resolved check sees the expanded values.
    config.problem.expand_fixed_from_scenario(&model).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_params: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
    // The one model fact `validate` needs beyond the parameter names: does
    // `init { }` DRAW a compartment from a law? It decides the ic_free ×
    // pfilter/pmmh cells (gh#732) — under the bootstrap particle filter a
    // declared law is the whole source of the swarm's spread at t=0.
    let init_law = if model.initial_conditions.iter().any(|(_, s)| s.is_law()) {
        crate::fit::methods::InitLaw::Declared
    } else {
        crate::fit::methods::InitLaw::Absent
    };
    config.validate(&model_params, init_law).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#75: prior-presence check honoring the IR fallback. Has to run
    // after partition validation but with the model IR in scope.
    let ir_prior_params: std::collections::BTreeSet<&str> = model.parameters.iter()
        .filter(|p| p.prior_dist().is_some() || p.hierarchical().is_some())
        .map(|p| p.name.as_str())
        .collect();
    config.validate_priors_present(&ir_prior_params).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // The chain-starts rule, resolved against the problem before the identity
    // is taken (proposal §3.4): `from_prior` when every estimated parameter
    // declares a prior, `uniform_unconstrained` otherwise, unless the file or
    // `--starts` spelled one. Said at startup either way, so the rule a run
    // used is never a thing to infer.
    let starts_resolution = {
        let method = config.inference.method.as_mut().expect("checked above");
        method.resolve_starts(&config.problem, &model)
    };
    // From here the method is read-only: the identity below sees exactly what
    // the runners consume.
    let method: config_v2::Method = config.inference.method.clone().expect("checked above");
    let allow_nonconverged_source = a.allow_nonconverged_source;
    if let Some(msg) = config.point_start_multichain_note() {
        // gh#71: a point-started multi-chain sampler's R̂ is uninformative;
        // the summary will report it as not assessed. A note, not an error —
        // the sample is still valid.
        eprintln!("\x1b[33mnote:\x1b[0m {}", msg);
    }

    // ── Validate sweeps ───────────────────────────────────────────────────
    // Validate: swept params must be in [fixed], not [estimate]
    let fixed_resolved = config.problem.fixed.resolve().unwrap_or_default();
    for (name, _) in &sweep_specs {
        if config.problem.estimate.contains_key(name) {
            eprintln!("error: cannot sweep '{}' — it is in [estimate].\n  \
                       Sweeps override [fixed] parameters. Move '{}' to [fixed] first.",
                name, name);
            std::process::exit(1);
        }
        if !fixed_resolved.contains_key(name) {
            eprintln!("error: sweep parameter '{}' not found in [fixed].\n  \
                       Available fixed params: {}",
                name, fixed_resolved.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
            std::process::exit(1);
        }
    }

    // Expand Cartesian product of sweep points
    let sweep_points: Vec<Vec<(String, f64)>> = if sweep_specs.is_empty() {
        vec![vec![]]
    } else {
        let mut points: Vec<Vec<(String, f64)>> = vec![vec![]];
        for (name, values) in &sweep_specs {
            let mut next = Vec::new();
            for pt in &points {
                for &v in values {
                    let mut new_pt = pt.clone();
                    new_pt.push((name.clone(), v));
                    next.push(new_pt);
                }
            }
            points = next;
        }
        points
    };
    let has_sweep = sweep_points.len() > 1;
    if has_sweep {
        eprintln!("sweep: {} points", sweep_points.len());
    }

    // Validate --resume requires a PGAS or PMMH method. Other methods
    // have no extension dimension (IF2's cooling depends on total
    // iterations, PFilter is single-pass), so resuming would be
    // statistically incoherent.
    if a.resume.is_some()
        && !matches!(method.algorithm, Algorithm::PGAS { .. } | Algorithm::PMMH { .. })
    {
        eprintln!("error: --resume is only supported for PGAS and PMMH methods; \
                   this fit's method is '{}'.", method.algorithm.method_name());
        std::process::exit(1);
    }

    // gh#191: gate the model's required capabilities against the method's
    // declared backend, before any fitting work. `profile` already runs this
    // check, but `fit run` never did — so a real-compartment (ODE-coupled)
    // model on a chain_binomial method was silently mis-fit (the filter loops
    // freeze the real reservoir at its init value). Fail fast with the
    // actionable message instead.
    {
        // `required_capabilities()` is STRUCTURAL (transitions / compartments /
        // balance) — the parameter VALUES are irrelevant to it. But
        // `CompiledModel::new` requires every parameter to carry a value, and
        // estimated parameters carry `value = None` in the IR (their start is
        // resolved from `[estimate].start` later). So fill any value-less
        // parameter with a harmless placeholder purely for this capability
        // scan — without it the gate errored "parameter '<estimated>' has no
        // value" on every estimate-only fit (gh#191: the gate must not demand
        // resolved params it doesn't use).
        let mut cap_model = model.clone();
        for p in &mut cap_model.parameters {
            if p.value.resolved_value().is_none() {
                let placeholder = p.initial_value()
                    .or_else(|| crate::params_resolver::resolved_bounds(p)
                        .map(|(lo, hi)| 0.5 * (lo + hi)))
                    .unwrap_or(1.0);
                p.value = p.value.with_value(placeholder);
            }
        }
        let compiled = sim::CompiledModel::new(cap_model).unwrap_or_else(|e| {
            eprintln!("error: {:?}", e);
            std::process::exit(1);
        });
        if let Err(msg) = gate_run_method_against_model(&method.algorithm, &compiled, config.problem.config.dt) {
            eprintln!("error: {}", msg);
            std::process::exit(1);
        }
    }

    // gh#audit-H12: --record-prequential and --record-ancestry only have
    // effect on a PFilter method. Refuse them on any other, so they are never
    // silently dropped.
    if (a.record_prequential || a.record_ancestry)
        && !matches!(method.algorithm, Algorithm::PFilter { .. })
    {
        let flag = if a.record_prequential { "--record-prequential" } else { "--record-ancestry" };
        eprintln!("error: {} applies to a `pfilter` method, and this fit's method is \
                   `{}`.", flag, method.algorithm.method_name());
        std::process::exit(1);
    }

    // gh#604: every key the user typed under `[data.observations]` /
    // `[data.holdout]` must name a declared observation source. Checked HERE —
    // before the identity digests below read a single byte — because those
    // digests open each bound path, so an unbound key would otherwise surface
    // as "cannot read data file '<key>'", diagnosing a missing file when the
    // real fault is a binding that names no stream. The motivating case is a
    // top-level key such as `ic_free` written below the `[data.observations]`
    // header, which TOML scopes into the table.
    if let Ok(ds) = config.problem.data_spec() {
        for (origin, table) in [
            ("[data.observations]", Some(&ds.observations)),
            ("[data.holdout]", ds.holdout.as_ref()),
        ] {
            let Some(table) = table else { continue };
            if let Err(e) = runner::check_bound_sources(&model, origin, table) {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }

    // ── Compute the fit-wide identity + sidecar (no fit-root run.json) ──
    //
    // The method leaf's run.json lives inside the leaf; the fit as a whole is
    // the `fits/{stem}-{h8}/` path segment, not a separate record. The
    // seed-independent parent fit hash computed here is reused by every leaf
    // as its `fit`-level hash — computing it once avoids the O(cells × full-I/O
    // rehash) pattern.
    let fit_start = std::time::Instant::now();
    // Validate --label early so we fail before any I/O. The same
    // validator is reused by `cmd_label` (post-hoc relabel).
    let validated_label = match a.label.as_deref() {
        Some(raw) => match validate_label(raw) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("error: invalid --label: {}", e);
                std::process::exit(1);
            }
        },
        None => None,
    };
    // gh#147 (M3.2): the fit identity is a CAS *path segment*
    // (`fits/{fit}-{h8}/`), not a separate fit-wide `run.json` — so there is no
    // fit-wide record; the segment is the fit-level home. Fit-level outputs
    // (synthetic data) live directly under it.
    // The fit-level identity (the `fit` CAS level) and the directory its
    // method leaves actually land in. `parent_fit_hash` is the fit-level
    // `ContentHash` (the same hash `resolve_fit_stage` puts on the `fit`
    // level, and the same one `FitView.fit_hash` reads back from `run.json`),
    // and `announced_fit_dir` is `fits/{stem}-{h8}/` built from it — so the
    // path `fit run` announces is exactly where the `{method}-{h8}` leaves
    // are written.
    //
    // Real and synthetic share this one `runid` fit-level digest. A real fit
    // folds its base `[data]` stream digests; a synthetic fit has no input
    // data (it generates data per-cell from the model + `[synthetic]`, both
    // already in the digest), so it hashes with an EMPTY data map. Either way
    // the container is keyed on the problem + engine, and the per-cell method
    // leaves resolve their own segment (folding the generated data).
    let announce_cas_root = crate::run_paths::output_root(None, config.problem.output_dir.as_deref());
    let announce_stem = crate::hashing::path_stem_slug(&fit_path)
        .unwrap_or_else(|| "fit".to_string());
    let announce_ir_version = ir::IR_VERSION.trim().to_string();
    // Real fits resolve their base `[data]` streams; synthetic fits have none
    // (empty map → no data digests folded).
    let fit_data_paths: indexmap::IndexMap<String, String> = config.problem.data_spec().ok()
        .and_then(|ds| {
            let model_obs_names: Vec<String> =
                model.observations.iter().map(|o| o.name.clone()).collect();
            ds.effective_observations(&model_obs_names).ok()
        })
        .unwrap_or_default();
    let parent_fit_hash_ch = cas::fit_level_hash(
        &model,
        &announce_ir_version,
        crate::version::VERSION_SHORT,
        &config.problem,
        &fit_data_paths,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: fit-level identity: {}", e);
        std::process::exit(1);
    });
    let announced_fit_dir = cas::fit_segment_dir(
        &announce_cas_root, &announce_stem, &parent_fit_hash_ch);
    let parent_fit_hash = parent_fit_hash_ch.to_hex();
    // Synthetic-data generation writes under the same content-addressed
    // segment.
    let fit_dir = announced_fit_dir.clone();

    let fit_sidecar = build_fit_sidecar(&config, &fit_path, validated_label, Some(&model));

    eprintln!("fit: {} (method={})", fit_path, method.algorithm.method_name());
    eprintln!("  model:    {}", config.problem.model.camdl);
    eprintln!("  estimate: {}", config.problem.estimate.keys()
        .map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    eprintln!("  fixed:    {}", {
        let resolved = config.problem.fixed.resolve().unwrap_or_default();
        resolved.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    });
    eprintln!("  starts:   {}", starts_resolution.describe());
    // gh#507: absolute, not as-written. A relative string here is consistent
    // with either base (the CWD or the fit.toml's directory), so it cannot
    // tell you which one you got — which is exactly how a run tree ended up
    // outside its repository unnoticed.
    eprintln!("  output:   {}",
        crate::run_paths::display_absolute(&announced_fit_dir).display());

    // IC-free inference diagnostic: when ic_free = true, make it
    // visible on the startup block so the user can confirm the PF is
    // computing log L_c (conditional on y₁) rather than log L. Silent
    // when ic_free is false or absent. See
    // docs/dev/proposals/archive/pre-alpha/2026-04-18-ic-free-inference.md.
    if config.problem.ic_free.unwrap_or(false) {
        let perturb_only_at_t0_params: Vec<&str> = config.problem.estimate.iter()
            .filter(|(_, spec)| spec.perturb_only_at_t0)
            .map(|(n, _)| n.as_str())
            .collect();
        eprintln!("\n  \x1b[36mic-free inference:\x1b[0m conditioning on y₁");
        // Name every ACTUAL source of the t=0 spread. There are two — a
        // declared `init { }` law, which every filter draws per particle
        // (gh#732), and, under IF2, a `perturb_only_at_t0` parameter (gh#364).
        // Reporting only the second told a user with a law that the list was
        // empty, i.e. that there was no spread — the one thing this line exists
        // to confirm.
        let mut sources: Vec<String> = Vec::new();
        let drawn: Vec<&str> = model.initial_conditions.iter()
            .filter(|(_, s)| s.is_law())
            .map(|(name, _)| name.as_str())
            .collect();
        if !drawn.is_empty() {
            sources.push(format!("`init {{ }}` laws on [{}]", drawn.join(", ")));
        }
        if !perturb_only_at_t0_params.is_empty() {
            sources.push(format!("perturb_only_at_t0 params [{}] (if2 only)",
                perturb_only_at_t0_params.join(", ")));
        }
        eprintln!("    - initial state spread from: {}", sources.join("; "));
        eprintln!("    - log-likelihood accumulation from t = 2 (y₁ reweights and resamples only)");
    }

    // ── Build the replicate grid: (dataset_idx, fit_seed) cells ──────────
    //
    // Four canonical modes, all routed through the same grid. Each cell is a
    // content-addressed fit (its own FitDigest base).
    //   Mode                           synthetic?  fit_seeds     Cells
    //   Single fit                     no          None/scalar   1
    //   Start-sensitivity              no          list of M     M  (seed levels, one base)
    //   Parameter recovery             yes         None/scalar   N  (one base per dataset)
    //   Parameter recovery × starts    yes         list of M     N × M
    //
    // For synthetic modes the datasets are generated once up front and the
    // per-cell DataSpec is materialised from their on-disk paths. See
    // docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md. A file that
    // declares `[data]` beside `[synthetic]` is a real-data fit here: the
    // truth is `fit recovery`'s to read (proposal §8, item 19).
    //
    // Multi-seed fits produce sibling CAS cells under one `fit`-level base
    // (the seed is a factored `runid` level), with no cross-cell aggregator.
    // A cross-seed roll-up view (per-method chain-R̂ across fit_seeds) is a
    // derived index, deferred to M4 (gh#154) — the same home as the profile
    // and grid roll-ups. The same applies to `dataset_idx` for synthetic fits.
    let fit_seeds: Vec<u64> = match &config.inference.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    let synthetic_datasets: Vec<synthetic::SyntheticDataset> = match &config.problem.synthetic {
        Some(spec) if config.problem.is_synthetic_fit() => {
            let datasets = synthetic::generate_synthetic_datasets(
                spec,
                // Pre-compiled IR (compiled once above) so per-dataset generation
                // doesn't re-invoke camdlc; falls back to the source path.
                config.problem.compiled_ir.as_deref().unwrap_or(&config.problem.model.camdl),
                &fit_dir,
                config.problem.config.dt,
                emit_every.as_ref(),
            ).unwrap_or_else(|e| {
                eprintln!("error: synthetic-data generation failed: {}", e);
                std::process::exit(1);
            });
            eprintln!("synthetic: generated {} dataset{} under {}/synthetic/data/",
                datasets.len(),
                if datasets.len() == 1 { "" } else { "s" },
                fit_dir.display());
            datasets
        }
        _ => Vec::new(),
    };

    // A cell is one (data_source, fit_seed) pair. Real-data cells carry
    // `dataset_idx = None` and leave the existing `[data]` in place;
    // synthetic cells carry `Some(idx)` and replace `[data]` with a
    // DataSpec pointing at the generated TSV.
    struct Cell {
        dataset_idx: Option<usize>,
        fit_seed: u64,
        // None → keep the problem's data; Some → overwrite with synthetic path.
        data_override: Option<config_v2::DataSpec>,
    }
    let cells: Vec<Cell> = if synthetic_datasets.is_empty() {
        fit_seeds.iter().map(|&s| Cell {
            dataset_idx: None,
            fit_seed: s,
            data_override: None,
        }).collect()
    } else {
        // Synthetic generation writes one file per observation stream, keyed
        // by the `source` the loader binds it to — the same `[data.observations]`
        // shape a real fit declares, so the generated dataset is read back
        // through the loader the real data uses (gh#831).
        let mut out = Vec::with_capacity(synthetic_datasets.len() * fit_seeds.len());
        for ds in &synthetic_datasets {
            let observations: indexmap::IndexMap<String, String> = ds.files.iter()
                .map(|(source, path)| (source.clone(), path.to_string_lossy().to_string()))
                .collect();
            let data_spec = config_v2::DataSpec {
                file: None,
                observations,
                holdout_after: None,
                holdout: None,
            };
            for &fs in &fit_seeds {
                out.push(Cell {
                    dataset_idx: Some(ds.idx),
                    fit_seed: fs,
                    data_override: Some(data_spec.clone()),
                });
            }
        }
        out
    };

    let total_cells = cells.len();
    if total_cells > 1 {
        eprintln!("grid: {} cell{}", total_cells,
            if total_cells == 1 { "" } else { "s" });
    }

    // ── gh#147 (M3.2): content-addressed store root + fit-level label ──
    // Fits write to `<output_root>/fits/{fit}-{h8}/{method}-{h8}/seed_N-{h8}/`
    // (symmetric to sims under `<output_root>/sims/`). The fit level is a
    // path segment, so there is no separate fit-wide record.
    let cas_root = crate::run_paths::output_root(None, config.problem.output_dir.as_deref());
    let fit_stem = crate::hashing::path_stem_slug(&fit_path)
        .unwrap_or_else(|| "fit".to_string());
    let ir_version_str = ir::IR_VERSION.trim().to_string();
    // gh#147 (M3.2): fit segments whose fit-level sidecar (label + model hash
    // + the `fit.toml.original` config-diff archive) has been written this
    // run. The fit level is a path segment with no CAS record, so this sidecar
    // is the fit-wide home `walk_fits_root` / `table_row` read; write it once
    // per segment (each sweep point keys its own FitDigest → its own segment).
    let mut written_fit_segments: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();

    // The rule every cell draws from and folds into its identity: the sourced
    // resolution taken above, or the bare rule (declared or defaulted) now.
    let starts_rule = method.starts().cloned().unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let resolved_starts = early_resolved_starts.unwrap_or_else(|| {
        chain_starts::resolve_starts(&starts_rule, allow_nonconverged_source).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        })
    });
    if let Some(src) = &resolved_starts.source {
        match &src.leaf_dir {
            Some(leaf) => eprintln!(
                "  starts source: {} ({})",
                leaf.display(),
                src.file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            None => eprintln!("  starts source: {}", src.file.display()),
        }
    }

    // ── Execute grid: cell × sweep_point ──
    for (cell_i, cell) in cells.iter().enumerate() {
        let mut cell_problem = config.problem.clone();
        if let Some(spec) = &cell.data_override {
            // Materialise the synthetic cell's data path. `[synthetic]`
            // stays set — `data_spec()` returns `data` when both are
            // present, which is the per-cell behaviour we want.
            cell_problem.data = Some(spec.clone());
        }
        let seed = cell.fit_seed;
        if total_cells > 1 {
            match cell.dataset_idx {
                Some(idx) => eprintln!("\n━━━ cell {}/{}: ds_{:02} × fit_seed={} ━━━",
                    cell_i + 1, total_cells, idx, seed),
                None      => eprintln!("\n━━━ cell {}/{}: fit_seed={} ━━━",
                    cell_i + 1, total_cells, seed),
            }
        }

    // Execute: sweep_point
    for (pt_idx, sweep_point) in sweep_points.iter().enumerate() {
        // The problem with the swept values applied to [fixed]. The method is
        // unchanged by a sweep.
        let mut sweep_config = cell_problem.clone();
        for (name, val) in sweep_point {
            sweep_config.fixed.values.insert(name.clone(), *val);
        }

        // IC4 in 2026-04-19 inference review batch 3: reject
        // prior × transform combinations that silently produce a
        // different prior than the user wrote (log_normal on
        // Transform::None → Normal; log_normal on Logit → logit-
        // normal; etc.). Runs after sweep-value substitution since
        // sweep can change a param's role, but the prior/transform
        // binding itself is fixed across sweep points — this is
        // equivalent to a one-shot check at config load, but
        // putting it here means every cell sees its own validation.
        if let Err(e) = runner::validate_prior_transform_compat(&sweep_config.estimate, &model) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }

        if has_sweep {
            let slug: String = sweep_point.iter()
                .map(|(k, v)| format!("{}_{:.3}", k, v))
                .collect::<Vec<_>>()
                .join("__");
            if pt_idx == 0 {
                eprintln!();
            }
            eprintln!("═══ sweep point {}/{}: {} ═══", pt_idx + 1, sweep_points.len(), slug);
        }

        let stage_name = method.algorithm.method_name();
        eprintln!("\n── method: {} ──", stage_name);

        // Resolve data the runners load from (also feeds the data digests).
        let data_spec = sweep_config.data_spec().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // Expand the [data] shorthand (`file = "..."`) into the canonical
        // per-stream map before hashing, so the shorthand and the
        // verbose `[data.observations]` form produce identical method
        // hashes when they reference the same data.
        let model_obs_names: Vec<String> =
            model.observations.iter().map(|o| o.name.clone()).collect();
        let effective_obs = data_spec.effective_observations(&model_obs_names)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
        // The method level's `deps`: the content of a sourced `starts` rule's
        // file, so a regenerated source re-keys this run (gh#541), plus the
        // `--resume` base.
        let mut deps: Vec<runid::inputs::ArtifactRef> =
            resolved_starts.source.iter().map(|s| s.dep.clone()).collect();

        // gh#147 (M3.2): --resume <base ref> reads a prior leaf read-only; the
        // resumed run writes a distinct leaf keyed on the new target_length
        // plus a dep on the base. Resolve the ref and fold the dep here; the
        // chain state is copied into the new leaf after the claim below.
        let resume_base: Option<std::path::PathBuf> = a.resume.as_deref().map(|r| {
            let base = resolve_base_ref(r, &cas_root).unwrap_or_else(|| {
                eprintln!("error: --resume base '{}' not found (run_id prefix or leaf path)", r);
                std::process::exit(1);
            });
            if let Some(dep) = cas::cas_dep_from_dir(&base) {
                deps.push(dep);
            }
            base
        });

        // ── CAS identity + claim ──
        // sweep_config is the authoritative problem inside the sweep loop.
        let ctx = cas::FitStageCtx {
            model: &model,
            fit_stem: &fit_stem,
            ir_version: &ir_version_str,
            engine_version: crate::version::VERSION_SHORT,
            problem: &sweep_config,
            data_paths: &effective_obs,
            method: &method,
            seed,
            deps: deps.clone(),
        };
        let resolved = cas::resolve_fit_stage(&ctx).unwrap_or_else(|e| {
            eprintln!("error: fit-method identity: {}", e);
            std::process::exit(1);
        });
        let cas_path = runid::store_path(&cas_root, runid::ArtifactKind::FitStage, &resolved.levels);
        // gh#147 (M3.2): write the fit-level sidecar once per fit segment
        // (`cas_path`'s grandparent: `.../fits/{stem}-{h8}/`). Done before the
        // cache-hit short-circuit below so the label/archive stay current even
        // on an all-cache-hit rerun.
        if let Some(seg) = cas_path.parent().and_then(|p| p.parent()) {
            if written_fit_segments.insert(seg.to_path_buf()) {
                // gh#147 (M3.2): the fit-level provenance sidecar — a faithful
                // readable projection of the fit-wide provenance `build_fit_run`
                // computed (resolved_priors with gh#75 sources,
                // estimated/fixed/data_hashes/model_identity). Derived provenance,
                // never identity-bearing (the priors are already hashed into the
                // FitDigest); written once per segment, even on a cached rerun.
                if let Err(e) = crate::run_meta::write_fit_sidecar(
                    seg,
                    std::path::Path::new(&fit_path),
                    &fit_sidecar,
                ) {
                    eprintln!("warning: cannot write fit-level sidecar {}: {}", seg.display(), e);
                }
                // gh#322: archive the compiled base-model IR in the fit segment so
                // downstream verbs (`fit predict`) are self-contained and the run
                // is portable — they resolve the model from this archive rather
                // than recompiling the loose `.camdl`, which may have moved. An
                // artifact addition, identity-neutral (not a hashed level; mirrors
                // `batch.rs`'s sibling `model.ir.json`). The base model IR is
                // structurally identical across sweep cells (a sweep overrides
                // parameter *values* at resolve time), so one archive per segment.
                if let Some(ir_src) = config.problem.compiled_ir.as_deref() {
                    let dest = seg.join("model.ir.json");
                    match std::fs::read(ir_src) {
                        Ok(bytes) => {
                            if let Err(e) = std::fs::write(&dest, &bytes) {
                                eprintln!("warning: cannot archive model IR {}: {}",
                                    dest.display(), e);
                            }
                        }
                        Err(e) => eprintln!(
                            "warning: cannot read compiled IR {} to archive: {}", ir_src, e),
                    }
                }
                // Archive the model's display render (`model.render.json`) and
                // the flow graph (`model.graph.json`) beside the IR so a viewer
                // (camdl-watch) can show the model's math without recompiling.
                // Best-effort + identity-neutral, like the IR archive above; a
                // render failure never aborts the fit.
                //
                // gh#536: guarded on the model being SOURCE. `camdlc render`
                // does not read IR, and `[model] camdl` accepts a compiled
                // `.ir.json` — several tests fit against one. Unguarded, every
                // such fit printed "parse error in …/sir.ir.json" twice, which
                // reads as "your compiled IR is malformed" when nothing is
                // wrong. gh#496 fixed exactly this for `batch run` and reasoned
                // that the fit path was safe because `config.model.camdl` "is
                // source by construction"; it is not. One guard over both
                // blocks, since they share the precondition.
                if !crate::util::model_is_camdl_source(&config.problem.model.camdl) {
                    eprintln!(
                        "note: model given as compiled IR; skipping model.render.json / \
                         model.graph.json (pass the .camdl source to archive the display render)"
                    );
                } else {
                match crate::util::render_model_json(std::path::Path::new(&config.problem.model.camdl)) {
                    Ok(json) => {
                        let dest = seg.join("model.render.json");
                        if let Err(e) = std::fs::write(&dest, &json) {
                            eprintln!("warning: cannot archive model render {}: {}",
                                dest.display(), e);
                        }
                    }
                    Err(e) => eprintln!("warning: cannot render model for archive: {}", e),
                }
                // Archive the structured flow graph (`model.graph.json`) beside
                // the display render so a viewer can draw the compartmental flow
                // diagram. Same best-effort, identity-neutral treatment.
                match crate::util::render_model_graph_json(std::path::Path::new(&config.problem.model.camdl)) {
                    Ok(json) => {
                        let dest = seg.join("model.graph.json");
                        if let Err(e) = std::fs::write(&dest, &json) {
                            eprintln!("warning: cannot archive model graph {}: {}",
                                dest.display(), e);
                        }
                    }
                    Err(e) => eprintln!("warning: cannot render model graph for archive: {}", e),
                }
                }
            }
        }
        let store = runid::FsCasStore::new(&cas_root);
        // Streaming write through the one resolved-writer seam (gh#241 PR D).
        // The running record carries Null inputs (the method's loglik summary
        // is a post-run result); the final inputs are supplied to `finalize`.
        // The upstream lineage deps ride in `RecordMeta`.
        let resolved_artifact = crate::resolve::ResolvedArtifact {
            kind: runid::ArtifactKind::FitStage,
            levels: resolved.levels.clone(),
            run_id: resolved.run_id,
            display_inputs: serde_json::Value::Null,
        };
        let meta = crate::resolve::RecordMeta::new(
            &ir_version_str, &sweep_config.model.camdl, None)
            .with_deps(deps.clone());
        // Reuse decision through the one policy seam. `--resume` deliberately
        // does NOT reuse: it extends a base run into a NEW leaf keyed on the
        // longer chain, so this method must run even when its own identity is
        // already stored. `--force` displaces the incumbent (quarantined) —
        // previously it skipped the lookup only to die at the claim with
        // "artifact already completed", which is why the flag never worked
        // on a completed leaf.
        let policy = if force {
            crate::resolve::WritePolicy::Force
        } else {
            crate::resolve::WritePolicy::Reuse
        };
        if a.resume.is_none() {
            match crate::resolve::check_reuse(&store, &cas_root, &resolved_artifact, policy) {
                Ok(crate::resolve::ReuseVerdict::CacheHit { dir, .. }) => {
                    eprintln!("  \x1b[33mcache hit — reusing {}\x1b[0m",
                        dir.strip_prefix(&cas_root).unwrap_or(&dir).display());
                    continue;
                }
                Ok(crate::resolve::ReuseVerdict::MustRun) => {}
                Err(e) => {
                    eprintln!("error: fit method cache check {}: {}", cas_path.display(), e);
                    std::process::exit(1);
                }
            }
        }
        let mut write = match crate::resolve::begin_resolved_write(
            &store, &cas_root, &resolved_artifact, &meta,
            crate::resolve::WriteMode::Streaming,
        ) {
            Ok(crate::resolve::ResolvedWrite::Streaming(c)) => c,
            Ok(crate::resolve::ResolvedWrite::Committed(_)) => {
                unreachable!("Streaming write mode never returns a committed path")
            }
            Err(e) => {
                eprintln!("error: claim fit method {}: {}", cas_path.display(), e);
                std::process::exit(1);
            }
        };
        let stage_dir = write.dir().to_path_buf();

        // gh#147 (M3.2): seed the resumed leaf with the base chain's state
        // (resume_state.bin + parameter_traces.tsv per chain), copied from the
        // read-only base. The runner then loads/extends these in the new leaf;
        // the base is never written.
        if let Some(base) = &resume_base {
            copy_resume_carryover(base, &stage_dir).unwrap_or_else(|e| {
                eprintln!("error: staging resume carry-over from {}: {}", base.display(), e);
                std::process::exit(1);
            });
        }

        let stage_t0 = std::time::Instant::now();
        // Uninitialized on purpose: every dispatch arm must assign a
        // loglik (or exit) — the compiler enforces that no arm can fall
        // through to the finalize below with a silent None.
        let stage_best_loglik: Option<f64>;
        // PFilter has replicates, not competing chains, so it legitimately
        // leaves this None.
        let mut stage_best_chain: Option<usize> = None;

        // Surface the registry caveat for Beta/Experimental methods, once per
        // executing method (after the cache-hit skip above, so reused leaves
        // stay silent). Registry-driven so it can't drift from `fit methods`.
        methods::emit_status_banner(method.algorithm.method_kind(), method.algorithm.backend());

        // gh#506 follow-up, gh#881: a declared `start` that the chosen starts
        // rule discards is a silent no-op. Not an error — the spread rules
        // ignore it on purpose — but the user who wrote the value should hear
        // that it had no effect, rather than inferring a start that never
        // happened. Every method kind draws its chain starts the same way, so
        // this runs once here, before the dispatch.
        if starts_rule.ignores_base_point(method.algorithm.chains()) {
            let declared: Vec<&str> = sweep_config.estimate.iter()
                .filter(|(_, spec)| spec.start.is_some())
                .map(|(n, _)| n.as_str())
                .collect();
            if !declared.is_empty() {
                eprintln!(
                    "  \x1b[33mnote:\x1b[0m `starts = \"{}\"` draws every chain's \
                     start, so `[estimate].start` is unused here for: {}. \
                     Use `starts = \"single\"` to start every chain at the \
                     declared values, or drop the `start` entries.",
                    starts_rule.tag(), declared.join(", "));
            }
        }

        // gh#278, gh#900: the stage's liveness/progress heartbeat. A background
        // thread writes `progress.json` into the stage dir on a fixed
        // wall-clock timer, and the guard writes the terminal record however
        // the stage exits — `done()` below when it returns, `abandon_stage` on
        // every fatal path, its `Drop` on a panic. Built here, before the
        // dispatch, so no method can silently be the one without a heartbeat.
        let heartbeat = stage_heartbeat(&stage_dir, &method.algorithm);

        match &method.algorithm {
            Algorithm::IF2 { backend, chains, particles, iterations, cooling, cooling_target_iters, loglik_eval, gate, dt_check, .. } => {
                // clean_eval comes straight from the method TOML — it is part
                // of the fit's identity (folded into the IF2 method's
                // whole-serialize identity_payload), so it has no CLI override
                // (gh#189: a CLI override bypassed the run_id and silently
                // re-scored under the same key). The gate's `--decibans-thresh`
                // override is applied through `apply_cli_overrides` BEFORE the
                // identity is taken, so `gate` here already carries it (gh#540
                // seam).
                let effective_loglik_eval = loglik_eval.clone();
                let effective_gate = gate.clone();

                let mut run_config = runner::FitRunConfig::build(
                    &sweep_config,
                    Some(&method.algorithm),
                    *chains, *particles, *iterations,
                    *cooling, *cooling_target_iters,
                    // gh#506: per-chain dispersion is the starts rule's job;
                    // the base point carries `[estimate].start`.
                    seed, false,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error building run config: {e}"))
                });
                run_config.loglik_eval = effective_loglik_eval.clone();
                run_config.gate = effective_gate.clone();

                // gh#585: the applied training window is the §3.7.3(b)
                // proof — recorded from the build that actually truncated,
                // into the fit-level sidecar written above.
                if let Some(seg) = cas_path.parent().and_then(|p| p.parent()) {
                    crate::run_meta::record_training_window(
                        seg, run_config.applied_training_window);
                }

                std::fs::create_dir_all(&stage_dir).unwrap_or_else(|e| {
                    abandon_stage(
                        &heartbeat,
                        format!("error creating {}: {e}", stage_dir.display()),
                    )
                });

                let collector = sim::inference::diagnostic::DiagnosticCollector::new(stage_name);
                let t0 = std::time::Instant::now();
                // Per-chain starting points, through the one seam every runner
                // draws from, under the resolved `starts` rule.
                let drawn = runner::draw_chain_starts_for(
                    &run_config, &sweep_config.estimate, &resolved_starts, *chains, seed,
                ).unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                // gh#887: a spread start the filter cannot score is redrawn,
                // up to MAX_START_ATTEMPTS, before any chain runs; every
                // attempt is on the record below.
                let drawn = runner::preflight_spread_starts(
                    &run_config, &sweep_config.estimate, &resolved_starts, drawn, seed,
                ).unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                let per_chain_params: Vec<Vec<sim::inference::types::EstimatedParam>> =
                    drawn.to_estimated_params(&run_config.estimated_params);
                // The audit sidecar, captured before the first filter pass.
                runner::record_chain_starts(&stage_dir, &run_config, &drawn);
                let chain_init_source = drawn.rule.spelled();
                let chain_starts_kind = drawn.rule.kind();
                let stage_dir_str = stage_dir.to_string_lossy();
                let chain_results = runner::run_chains_with_per_chain_params(
                    &run_config, Some(&per_chain_params), &collector,
                    Some(stage_dir_str.as_ref()), &heartbeat)
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                let elapsed = t0.elapsed();

                // Write outputs
                let param_names: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
                runner::write_chain_outputs(
                    &stage_dir.to_string_lossy(), &chain_results.results,
                    &run_config.estimated_params, &param_names,
                    &run_config.base_params, &run_config.compiled,
                    Some(&chain_results.loglik_eval),
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_clean_eval_tsv(
                    &stage_dir.to_string_lossy(),
                    &chain_results.loglik_eval, &run_config.estimated_params,
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_run_root_final_params(
                    &stage_dir.to_string_lossy(),
                    &chain_results.loglik_eval, &run_config.estimated_params,
                    &param_names, &run_config.base_params, &run_config.compiled,
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_diagnostics(&stage_dir.to_string_lossy(), &chain_results.results)
                    .unwrap_or_else(|e| eprintln!("warning: {}", e));

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
                let rw_sd = match runner::auto_rw_sd(&chain_results.results, &run_config.estimated_params) {
                    Ok((rw, _)) => rw,
                    Err(_) => run_config.estimated_params.iter()
                        .map(|s| (s.name.clone(), s.rw_sd * 0.5))
                        .collect(),
                };

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
                .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                dt_check::print_terminal_report(&dt_check_result);
                let fit_state = state::FitState {
                    method: stage_name.to_string(),
                    seed,
                    timestamp: crate::cas::iso8601_utc(std::time::SystemTime::now()),
                    input_hash: None,
                    camdl_version: Some(crate::version::VERSION_SHORT.into()),
                    best_loglik: chain_results.best_loglik,
                    initial_loglik: f64::NEG_INFINITY,
                    best_chain: chain_results.best_chain,
                    n_chains: *chains,
                    n_good_chains: None,
                    start_values,
                    rw_sd: rw_sd.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                    loglik_type: Some(loglik::LoglikType::If2),
                    acceptance_rate: None,
                    tail_chain_agreement: chain_results.chain_agreement.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                    perturb_only_at_t0_params: run_config.estimated_params.iter()
                        .filter(|p| p.perturb_only_at_t0)
                        .map(|p| p.name.clone()).collect(),
                    chain_logliks: chain_results.results.iter()
                        .map(|(_, r)| r.final_loglik).collect(),
                    chain_eval_logliks: chain_results.chain_eval_logliks(),
                    chain_eval_ids: chain_results.chain_eval_ids(),
                    chain_eval_ses: chain_results.chain_eval_ses(),
                    // Persist the gate / clean-eval config that was
                    // *actually in force* — `effective_gate` and
                    // `effective_loglik_eval` above already collapsed the
                    // priority chain (CLI flag > method TOML > defaults).
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
                };
                fit_state.save(&stage_dir.to_string_lossy()).unwrap_or_else(|e| {
                    eprintln!("warning: could not save fit_state: {}", e);
                });

                // Write mle_params.toml — clean-eval winner θ̂ (GH #16).
                let all_params = runner::collect_all_params(
                    winner_theta, &run_config.estimated_params, &run_config.model,
                    &run_config.base_params, &run_config.compiled,
                );
                let mle_path = format!("{}/mle_params.toml", stage_dir.display());
                let model_identity =
                    crate::resolve::model_identity_from_ir(&run_config.model_ir_json);
                let data_hashes: Vec<(String, String)> = sweep_config.data_spec()
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")))
                    .observations.iter()
                    .map(|(name, path)| {
                        let bytes = std::fs::read(path).unwrap_or_default();
                        let hash = {
                            use sha2::{Sha256, Digest};
                            let result = Sha256::digest(&bytes);
                            hex::encode(&result[..4])
                        };
                        (format!("{} ({})", name, path), hash)
                    })
                    .collect();
                let metadata = provenance::MleMetadata {
                    // Full fit-level hash — lets a reader locate the
                    // originating fit dir from just the mle_params.toml.
                    // The fit hash (not a model-only digest) so it pins
                    // the model+data+config triple, not just the model.
                    input_hash: parent_fit_hash.clone(),
                    model_path: sweep_config.model.camdl.clone(),
                    model_identity: model_identity.clone(),
                    data_hashes: data_hashes.clone(),
                    seed,
                    method: stage_name.to_string(),
                    best_chain: chain_results.best_chain,
                    // Record the backend the method actually fit on (gh#241):
                    // the `simulate --params` guardrail replays θ̂ with this,
                    // so it must be the method's backend. `InferenceBackend`
                    // is a valid `ForwardBackend` (total `From`).
                    backend: method.algorithm.backend().into(),
                    dt: sweep_config.config.dt,
                    loglik: chain_results.best_loglik,
                    loglik_sd: 0.0,
                    n_particles: *particles,
                    ess_at_mle: None,
                    timestamp: fit_state.timestamp.clone(),
                };
                provenance::write_mle_params(&mle_path, &all_params, &metadata)
                    .unwrap_or_else(|e| eprintln!("warning: {}", e));

                collector.render_to_stderr(sim::inference::diagnostic::HintContext {
                    chains_completed: Some(chain_results.results.len()),
                });

                stage_best_loglik = Some(chain_results.best_loglik);
                stage_best_chain = Some(chain_results.best_chain);

                eprintln!();
                crate::status::done("stored", format!("{} \u{b7} {}/", stage_name, stage_dir.display()));
                crate::status::hint(format!("best ll={:.1} (chain {}) in {:.1}s",
                    chain_results.best_loglik, chain_results.best_chain + 1, elapsed.as_secs_f64()));
            }
            Algorithm::PGAS { .. } => {
                let pgas_opts = pgas::PgasStageOpts::from_algorithm(&method.algorithm)
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                // gh#540: no CLI overrides here. Every flag that reaches this
                // method was written into it before its content address was
                // taken, so `from_algorithm` above already carries them.
                pgas::run_stage(
                    &sweep_config,
                    &method,
                    &stage_dir,
                    pgas_opts,
                    seed, force,
                    a.resume.is_some(),
                    &resolved_starts,
                    &heartbeat,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error running pgas: {e}"))
                });
                // Bubble loglik from fit_state.toml written by PGAS runner
                let fs = load_stage_result_or_exit(&heartbeat, stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Algorithm::PMMH { .. } => {
                let pmmh_opts = pmmh::PmmhStageOpts::from_algorithm(&method.algorithm)
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                pmmh::run_stage(
                    &sweep_config,
                    &method,
                    &stage_dir,
                    pmmh_opts,
                    seed, force,
                    a.resume.is_some(),
                    &resolved_starts,
                    // PMMH's dt-check is the PF-based one wired on the IF2 path.
                    /* dt_check_opt */ None,
                    &heartbeat,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error running pmmh: {e}"))
                });
                let fs = load_stage_result_or_exit(&heartbeat, stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Algorithm::Mh { dt_check, .. } => {
                // Deterministic-ODE Metropolis-Hastings. Routes through the
                // shared PMMH machinery (chains, adaptive proposal, priors,
                // MAP, R̂/ESS, trace output); `pmmh::run_stage` swaps the PF
                // likelihood for `compute_ode_loglik` when the algorithm is the
                // Mh variant. `PmmhStageOpts::from_algorithm` parses the Mh
                // fields with `n_particles = 0` / `rho = None` (the
                // deterministic path uses neither).
                let pmmh_opts = pmmh::PmmhStageOpts::from_algorithm(&method.algorithm)
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                // Deterministic ODE dt-check at the MAP (gh#52, gh#227). On by
                // default; honours the same CLI flags as the IF2 path
                // (--no-dt-check / --dt-check-halvings / --dt-check-strict).
                // gh#726: the method's dt_check field — CLI overrides were
                // applied through `apply_cli_overrides` BEFORE the identity
                // was taken, and the field is in Mh's identity_payload (the
                // result is stored in fit_state.toml.dt_check).
                let mh_dt_check = dt_check.clone();
                pmmh::run_stage(
                    &sweep_config,
                    &method,
                    &stage_dir,
                    pmmh_opts,
                    seed, force,
                    a.resume.is_some(),
                    &resolved_starts,
                    Some(mh_dt_check),
                    &heartbeat,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error running mh: {e}"))
                });
                let fs = load_stage_result_or_exit(&heartbeat, stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Algorithm::Nuts { .. } => {
                // Gradient-based Bayesian sampling of the deterministic ODE
                // marginal likelihood (gh#275 Phase 2) via `det_grad` + NUTS.
                let nuts_opts = nuts::NutsStageOpts::from_algorithm(&method.algorithm)
                    .unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
                nuts::run_stage(
                    &sweep_config,
                    &method,
                    &stage_dir,
                    nuts_opts,
                    seed, force,
                    a.resume.is_some(),
                    &resolved_starts,
                    &heartbeat,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error running nuts: {e}"))
                });
                let fs = load_stage_result_or_exit(&heartbeat, stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Algorithm::NlSbplx(nl_cfg) | Algorithm::NlBobyqa(nl_cfg) => {
                #[cfg(feature = "ode")]
                {
                    // Model identity + data digests for the mle_params.toml
                    // provenance block (same shape as the IF2 path uses).
                    // Load the canonical IR JSON from the pre-compiled IR so this
                    // provenance read doesn't re-invoke camdlc.
                    let model_src = sweep_config.compiled_ir.as_deref()
                        .unwrap_or(&sweep_config.model.camdl);
                    let model_ir_json = crate::util::load_model(model_src)
                        .ok()
                        .map(|(_, ir_json)| ir_json)
                        .unwrap_or_default();
                    let model_identity_for_prov =
                        crate::resolve::model_identity_from_ir(&model_ir_json);
                    let data_hashes_for_prov: Vec<(String, String)> = sweep_config
                        .data_spec()
                        .map(|d| d.observations.iter()
                            .map(|(name, path)| {
                                let bytes = std::fs::read(path).unwrap_or_default();
                                let hash = {
                                    use sha2::{Sha256, Digest};
                                    let result = Sha256::digest(&bytes);
                                    hex::encode(&result[..4])
                                };
                                (format!("{} ({})", name, path), hash)
                            })
                            .collect())
                        .unwrap_or_default();
                    // Deterministic ODE dt-check at θ̂ (gh#52, gh#227). On by
                    // default. gh#726: the method's dt_check field — CLI
                    // overrides were applied through `apply_cli_overrides`
                    // BEFORE the identity was taken, and NloptStageConfig
                    // full-serializes into its identity (the result is
                    // stored in fit_state.toml.dt_check).
                    let nl_dt_check = nl_cfg.dt_check.clone();

                    nlopt_stage::run_stage(
                        &sweep_config,
                        &method,
                        &stage_dir,
                        seed,
                        &resolved_starts,
                        &parent_fit_hash,
                        &model_identity_for_prov,
                        &data_hashes_for_prov,
                        &nl_dt_check,
                        &heartbeat,
                    ).unwrap_or_else(|e| {
                        abandon_stage(
                            &heartbeat,
                            format!("error running {stage_name}: {e}"),
                        )
                    });
                    let fs = load_stage_result_or_exit(&heartbeat, stage_name, &stage_dir);
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
                #[cfg(not(feature = "ode"))]
                {
                    let _ = (stage_name, &sweep_config, &stage_dir, seed, nl_cfg, &resolved_starts);
                    abandon_stage(
                        &heartbeat,
                        format!(
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
                    &sweep_config,
                    Some(&method.algorithm),
                    1, *particles, 1, 1.0, 1, seed, false,
                ).unwrap_or_else(|e| {
                    abandon_stage(&heartbeat, format!("error building pfilter config: {e}"))
                });

                // gh#585: same §3.7.3(b) proof recording as the IF2/Bayesian
                // paths — a fit may run a PFilter method alone.
                if let Some(seg) = cas_path.parent().and_then(|p| p.parent()) {
                    crate::run_meta::record_training_window(
                        seg, run_config.applied_training_window);
                }

                std::fs::create_dir_all(&stage_dir).unwrap_or_else(|e| {
                    abandon_stage(
                        &heartbeat,
                        format!("error creating {}: {e}", stage_dir.display()),
                    )
                });

                // The point the filter scores: the base point, or the point a
                // sourced `starts` rule names (`from_mle = "@fit"` scores at
                // that fit's estimate). A pfilter runs replicates of one point,
                // so one draw of the rule is the point.
                let drawn = runner::draw_chain_starts_for(
                    &run_config, &sweep_config.estimate, &resolved_starts, 1, seed,
                ).unwrap_or_else(|e| abandon_stage(&heartbeat, format!("error: {e}")));
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
                let mut preq_trace: Option<sim::inference::prequential::PrequentialTrace> = None;
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
                        record_ancestry,
                        ..run_config.smc_config()
                    };
                    let result = sim::inference::bootstrap_filter(
                        &process, &obs_model, &mle_params, &smc_config, pf_seed,
                    ).unwrap_or_else(|e| {
                        abandon_stage(&heartbeat, format!("pfilter error: {e:?}"))
                    });
                    if record_preq {
                        if let Some(ref recorded) = result.prequential {
                            // gh#268/gh#648: score against the real observed
                            // values (cross-stream sum on the union axis), the
                            // same seam `camdl pfilter --save-prequential`
                            // uses. NOT `run_config.observations[i].value` —
                            // that is the canonical union TIME axis, whose
                            // `value` is a never-scored 0.0 placeholder
                            // (`runner.rs`), so it scored every forecast
                            // against a vector of zeros.
                            let y_obs: Vec<f64> = obs_model.joint_observed();
                            // gh#269: per-stream observed values for the
                            // per-district score breakdown.
                            let per_stream_obs = obs_model.per_stream_observed();
                            // gh#833: what each value was accumulated over,
                            // for `compare`'s window gate.
                            let per_stream_cov = obs_model.per_stream_coverage();
                            preq_trace = Some(sim::inference::prequential::build_trace(
                                recorded, &y_obs, &per_stream_obs, &per_stream_cov,
                                &result.ess_trace, 0,
                                pf_seed,
                                obs_model.first_observation_opens_after_run_start(smc_config.t_start),
                                None));
                        }
                    }
                    logliks.push(result.log_likelihood);
                    if n_reps <= 10 || r % (n_reps / 10) == 0 {
                        eprintln!("  pfilter rep {}/{}: loglik={:.1}", r + 1, n_reps, result.log_likelihood);
                    }
                }
                let elapsed = t0.elapsed();

                let mean_ll = logliks.iter().sum::<f64>() / logliks.len() as f64;
                let sd_ll = if logliks.len() > 1 {
                    let var = logliks.iter().map(|l| (l - mean_ll).powi(2)).sum::<f64>() / (logliks.len() - 1) as f64;
                    var.sqrt()
                } else { 0.0 };

                eprintln!("\n  loglik = {:.1} ± {:.1} ({} reps, {} particles, {:.1}s)",
                    mean_ll, sd_ll, n_reps, particles, elapsed.as_secs_f64());

                // Write logliks.tsv
                {
                    use std::io::Write;
                    let path = format!("{}/logliks.tsv", stage_dir.display());
                    let mut f = std::fs::File::create(&path).unwrap();
                    writeln!(f, "replicate\tloglik").unwrap();
                    for (i, ll) in logliks.iter().enumerate() {
                        writeln!(f, "{}\t{:.4}", i + 1, ll).unwrap();
                    }
                }

                // Write prequential trace (plug-in predictive at MLE).
                // Scoring is a point-estimate property — rep 0 only.
                // Routed through the shared writer so the fit-method TSV
                // carries the same tidy/long schema as `camdl pfilter
                // --save-prequential` (gh#650).
                if let Some(ref trace) = preq_trace {
                    let stem = format!("{}/prequential", stage_dir.display());
                    crate::prequential_out::write_prequential_outputs(&stem, trace)
                        .unwrap_or_else(|e| {
                            abandon_stage(
                                &heartbeat,
                                format!("error writing prequential: {e}"),
                            )
                        });
                    eprintln!("  prequential: elpd={:.2}, mean_crps={:.3}, PIT 90% cov={:.2}",
                        trace.elpd(), trace.mean_crps(), trace.pit_coverage(0.90));
                    eprint!("{}", crate::prequential_out::surprise_table(trace));
                }
                stage_best_loglik = Some(mean_ll);
            }
        }

        // The stage is done: every arm either reached here or abandoned the run
        // through `abandon_stage`. The terminal write goes down BEFORE
        // `finalize` below, so the timer thread is stopped before the leaf's
        // manifest is taken over its files — a `progress.json` rewritten during
        // the manifest walk would be hashed mid-flight.
        heartbeat.done();

        // ── finalize the CAS fit-method leaf ──
        // The runners streamed every output (chains, fit_state.toml,
        // draws.tsv, trajectories/, …) into `stage_dir = claim.dir()`;
        // `finalize` builds the recursive exact-set manifest over them and
        // commits Running→Completed. The display fields ride in `run.json`
        // `inputs` (recorded, never hashed) for show/status.
        let stage_elapsed = stage_t0.elapsed();
        let algo_tag = method.algorithm.method_name();
        let backend_tag = method.algorithm.backend().as_str();
        let algo_json = match &method.algorithm {
            Algorithm::IF2 { chains, particles, iterations, cooling, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations, "cooling": cooling }),
            Algorithm::PGAS { chains, particles, sweeps, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "sweeps": sweeps }),
            Algorithm::PMMH { chains, particles, iterations, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations }),
            Algorithm::Mh { chains, iterations, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "iterations": iterations }),
            Algorithm::Nuts { chains, warmup, samples, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "warmup": warmup, "samples": samples }),
            Algorithm::PFilter { particles, replicates, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "particles": particles, "replicates": replicates }),
            Algorithm::NlSbplx(c) | Algorithm::NlBobyqa(c) =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": c.chains, "tolerance": c.tolerance, "max_evals": c.max_evals }),
        };
        // gh#901: no `stage` key. It held exactly `algo_tag` — the method's
        // own name — so the leaf carried the same string twice under two
        // vocabularies, and `FitStageView` now reads the label off `method`.
        let inputs_json = serde_json::json!({
            "method": algo_tag,
            "backend": backend_tag,
            "seed": seed,
            "n_chains": method.algorithm.chains(),
            "best_loglik": stage_best_loglik,
            "best_chain": stage_best_chain,
            "algorithm": algo_json,
            "starts": starts_rule.spelled(),
            "chain_starts_kind": starts_rule.kind().as_str(),
            "fit_hash": resolved.levels.first().map(|l| l.hash.to_hex()),
            "wall_time_seconds": stage_elapsed.as_secs_f64(),
        });
        // Declare the tabular outputs' column schema in run.json (proposal
        // 2026-07-15): classify each written file's real header so a consumer
        // reads roles instead of reverse-engineering columns. Recorded, not
        // hashed — the run's identity was fixed at claim time.
        {
            let estimated: std::collections::HashSet<&str> =
                fit_sidecar.estimated.iter().map(String::as_str).collect();
            let all_params: std::collections::HashSet<&str> = fit_sidecar
                .estimated
                .iter()
                .map(String::as_str)
                .chain(fit_sidecar.fixed.keys().map(String::as_str))
                .collect();
            let schema =
                crate::output_schema::fit_output_schema(write.dir(), &all_params, &estimated);
            write.set_output_schema(schema);
        }
        let dest = match write.finalize(inputs_json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not finalize fit method leaf {}: {}", cas_path.display(), e);
                std::process::exit(1);
            }
        };
        crate::status::done("stored",
            format!("{} \u{b7} {} \u{b7} {:.1}s", stage_name, dest.display(), stage_elapsed.as_secs_f64()));

    } // end sweep_points
    } // end cells

    // ── Grid roll-ups (summary.tsv / coverage.tsv): deferred to M4 ──
    //
    // Each grid cell is now a content-addressed fit — its own FitDigest base,
    // keyed by that cell's dataset digest × fit-seed — readable individually
    // via `camdl list`/`show`/`cat`. The cross-cell summary and the synthetic
    // parameter-recovery coverage table are derived views with no home in the
    // per-cell tree; `fit recovery` (proposal increment 3) owes the roll-up
    // (gh#150 / gh#154), where coverage gains a truth-within-interval
    // correctness check.
    if cells.len() > 1 || config.problem.is_synthetic_fit() {
        eprintln!("note: grid summary / coverage are derived views — \
                   owed by `fit recovery` (gh#150 / gh#154)");
    }

    // gh#147 (M3.2): no fit-wide `run.json` rewrite — the fit identity is a
    // CAS path segment, and each method leaf records its own wall time in
    // `run.json` `inputs` at `finalize`. `fit_start` paces the run; per-leaf
    // timing is the honest unit now.
    let _ = fit_start;
}

/// The `--init` / `--posterior` / `--mle` / `--params` family, each answered
/// with the `--starts` rule that replaced it.
///
/// Shared by `fit run` and `camdl profile` (gh#889): the same four flags were
/// removed from both, so one flag has one answer rather than two that drift.
pub(crate) fn starts_family_removed_lines(
    init: Option<&str>,
    posterior: Option<&str>,
    mle: Option<&str>,
    params: Option<&str>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    if let Some(s) = init {
        let replacement = match s {
            "from_posterior" | "from_mle" | "from_params" => format!("--starts {s}=<source>"),
            other => format!("--starts {other}"),
        };
        lines.push(format!("--init {s}: write `{replacement}`."));
    }
    if let Some(s) = posterior {
        lines.push(format!("--posterior {s}: write `--starts from_posterior={s}`."));
    }
    if let Some(s) = mle {
        lines.push(format!("--mle {s}: write `--starts from_mle={s}`."));
    }
    if let Some(s) = params {
        lines.push(format!("--params {s}: write `--starts from_params={s}`."));
    }
    lines
}

/// Wrap the per-flag replacement lines for `command` into the one message a
/// user sees, or `None` when nothing removed was passed.
pub(crate) fn removed_flags_message(command: &str, lines: Vec<String>) -> Option<String> {
    if lines.is_empty() {
        return None;
    }
    let mut msg = format!(
        "these `camdl {command}` flags were removed with the `[stages]` → `[method]` \
         split (proposal 2026-09-08-workflow-first-fit-config):",
    );
    for l in lines {
        msg.push_str("\n  ");
        msg.push_str(&l);
    }
    msg.push_str(&format!(
        "\n  See `camdl {command} --help` and `camdl docs fit-toml`."));
    Some(msg)
}

/// The removed `fit run` flags, each answered with its replacement. Every
/// one the user passed is named in one message, so a habitual
/// `--stage posterior --init from_prior` is corrected in one round.
fn removed_flag_message(a: &crate::args::FitRunArgs) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    if let Some(s) = a._removed_stage.as_deref() {
        lines.push(format!(
            "--stage {s}: a fit.toml carries one `[method]`, so there is nothing to \
             select; drop the flag. A second way of fitting the same problem is a \
             second file (`camdl fit new --from fit.toml fit-{s}.toml`)."
        ));
    }
    lines.extend(starts_family_removed_lines(
        a._removed_init.as_deref(),
        a._removed_posterior.as_deref(),
        a._removed_mle.as_deref(),
        a._removed_params.as_deref(),
    ));
    if a._removed_survey_path.is_some() || a._removed_survey_top_k.is_some() {
        lines.push(
            "--survey-path / --survey-top-k: the `survey_top_k` start rule was removed (a \
             survey landscape is not a posterior). Use `--starts from_prior`, or run a short \
             fit and `--starts from_posterior=@handle`."
                .to_string(),
        );
    }
    if a._removed_allow_nonconverged_scout {
        lines.push(
            "--allow-nonconverged-scout: the in-file scout/refine gate is gone with in-file \
             chaining; a warm start's source is checked where it is consumed, and \
             `--allow-nonconverged-source` lifts that refusal."
                .to_string(),
        );
    }
    if let Some(raw) = a._removed_starts_from.as_deref() {
        lines.push(format!("--starts-from {raw}: write `--starts from_mle={raw}`."));
    }
    if let Some(raw) = a._removed_init_method.as_deref() {
        lines.push(format!("--init-method {raw}: write `--starts {raw}`."));
    }
    removed_flags_message("fit run", lines)
}

/// After a runner reported success, its `fit_state.toml` is the channel
/// carrying the result back to this orchestrator (`best_loglik` /
/// `best_chain` end up in the finalized run.json inputs). A missing or
/// corrupt file at that point is a runner bug or a torn write; the previous
/// `if let Ok` swallow left a silent `null` in run.json where the result
/// belonged. Fail loudly instead.
fn load_stage_result_or_exit(
    heartbeat: &io::HeartbeatGuard,
    stage_name: &str,
    stage_dir: &std::path::Path,
) -> state::FitState {
    state::FitState::load(&stage_dir.to_string_lossy()).unwrap_or_else(|e| {
        abandon_stage(
            heartbeat,
            format!(
                "error: method '{}' reported success but its fit_state.toml \
                 cannot be read back from {}: {}",
                stage_name, stage_dir.display(), e),
        )
    })
}

/// Resolve a `--resume <base ref>` to a base method-leaf dir: an existing
/// path, else a `run_id` hex prefix matched under `<cas_root>/fits/`. `None`
/// when no unique match (caller errors).
fn resolve_base_ref(reference: &str, cas_root: &std::path::Path) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(reference);
    if p.is_dir() {
        return Some(p.to_path_buf());
    }
    let mut matches = crate::cas_read::resolve_fit_prefix(cas_root, reference);
    if matches.len() == 1 {
        Some(matches.remove(0).dir)
    } else {
        None
    }
}

/// Copy the base leaf's per-chain resume state (`chain_*/resume_state.bin` +
/// `parameter_traces.tsv`) into the resumed leaf so the runner extends them
/// there. The base is read-only.
fn copy_resume_carryover(base: &std::path::Path, new_leaf: &std::path::Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(base)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("chain_") || !entry.path().is_dir() {
            continue;
        }
        let dst = new_leaf.join(name.as_ref());
        std::fs::create_dir_all(&dst)?;
        for f in ["resume_state.bin", "parameter_traces.tsv"] {
            let src = entry.path().join(f);
            if src.is_file() {
                std::fs::copy(&src, dst.join(f))?;
            }
        }
    }
    Ok(())
}

// gh#147 (M3.2): `archive_fit_toml` (the legacy `fit.toml.original` writer
// under `fit_dir`) was removed — fits are content-addressed now and the
// fit-level config archive for `fit table` config_diff is reworked in M3.3
// alongside the fit-level outputs' CAS relocation.

/// Build the fit-level provenance sidecar for a fit.toml. Fields that require
/// I/O (model IR, data files, fit.toml bytes) are read here and digested.
/// Silent fallbacks (empty strings / empty maps) cover the read-error case so a
/// partially-written fit still produces a sidecar `camdl list` can display.
/// The readable provenance projection is written once per fit segment.
///
/// `model` is the compiled, expanded IR model. It resolves per-parameter prior
/// provenance (`fit_toml / model_ir / flat_explicit`) and the observation
/// schema ([`crate::run_meta::ObsSchema`]); pass `None` when no model is in
/// scope (`fit where` doesn't load the IR for a path-only resolution), in which
/// case `resolved_priors` is empty and `schema` is `None`.
fn build_fit_sidecar(
    config: &config_v2::FitConfig,
    fit_path: &str,
    label: Option<String>,
    model: Option<&ir::Model>,
) -> crate::run_meta::FitSidecar {
    let problem = &config.problem;
    // Prefer the pre-compiled IR (set by `cmd_fit_run_v2`); `load_model`
    // re-invokes camdlc when handed a raw `.camdl`. The resolved model identity
    // is byte-identical either way; empty string if the IR can't be loaded.
    let model_src = problem.compiled_ir.as_deref().unwrap_or(&problem.model.camdl);
    let model_identity = crate::util::load_model(model_src)
        .ok()
        .map(|(_, ir_json)| crate::resolve::model_identity_from_ir(&ir_json))
        .unwrap_or_default();
    let fit_toml_bytes = std::fs::read(fit_path).unwrap_or_default();
    let fit_toml_hash = crate::hashing::sha256_hex(&fit_toml_bytes);
    // gh#771: hash the RESOLVED stream set — the same seam the fit identity
    // uses (`effective_observations`, see `cmd_fit_run_v2`). The single-file
    // `[data] file = "..."` shorthand leaves `[data.observations]` empty and
    // binds one wide TSV to every stream the model declares; hashing
    // `observations` directly wrote an empty `data_hashes` for exactly that
    // config style, and the `camdl compare` data preflight (gh#713) then
    // reports such a fit as unchecked instead of comparing its digests.
    // With no model in scope (`fit where`) the shorthand cannot be expanded,
    // and `effective_observations` errors — the map stays empty, as before.
    let model_obs_names: Vec<String> = model
        .map(|m| m.observations.iter().map(|o| o.name.clone()).collect())
        .unwrap_or_default();
    let data_hashes: std::collections::HashMap<String, String> = problem
        .data.as_ref()
        .and_then(|d| d.effective_observations(&model_obs_names).ok())
        .map(|obs| obs.into_iter()
            .filter_map(|(name, path)| {
                crate::hashing::file_hash(&path).map(|h| (name, h))
            })
            .collect())
        .unwrap_or_default();
    let estimated: Vec<String> = problem.estimate.keys().cloned().collect();
    let fixed: std::collections::HashMap<String, f64> = problem.fixed
        .resolve().unwrap_or_default().into_iter().collect();
    // gh#75: resolve per-parameter prior provenance. We only emit
    // entries when the method is Bayesian (an IF2 fit doesn't consume
    // priors, and surfacing `model_ir` for a bunch of params the fit will
    // never read from is misleading noise). `validate_priors_present` has
    // already rejected any silent flat-fallback by the time we get here, so
    // any entry we emit is one of {fit_toml, model_ir, flat_explicit}.
    let any_bayesian = config
        .inference
        .method
        .as_ref()
        .is_some_and(|m| m.algorithm.requires_priors());
    let resolved_priors: Vec<crate::run_meta::ResolvedPriorEntry> = match model {
        Some(model) if any_bayesian => {
            let names: Vec<String> = problem.estimate.keys().cloned().collect();
            crate::fit::priors_precedence::resolve_priors_with_precedence(
                &names, &problem.estimate, model,
            )
            .into_iter()
            .map(|r| {
                let source = match r.source {
                    crate::fit::priors_precedence::PriorSource::FitToml      => "fit_toml",
                    crate::fit::priors_precedence::PriorSource::ModelIr      => "model_ir",
                    crate::fit::priors_precedence::PriorSource::FlatExplicit => "flat_explicit",
                    // FlatFallback shouldn't reach here — validate_priors_present
                    // rejects it. If it does, surface it so reviewers can
                    // see the contract was broken.
                    crate::fit::priors_precedence::PriorSource::FlatFallback => "flat_fallback",
                };
                crate::run_meta::ResolvedPriorEntry {
                    param:  r.param,
                    source: source.to_string(),
                }
            })
            .collect()
        }
        _ => Vec::new(),
    };
    crate::run_meta::FitSidecar {
        label,
        model_path: problem.model.camdl.clone(),
        model_identity,
        fit_toml_path: fit_path.to_string(),
        fit_toml_hash,
        // gh#542: the in-memory maps stay `HashMap`; the ARTIFACT is ordered.
        // Same seam gh#519 used for `FitState` — the ordering requirement
        // belongs to `fit.meta.json`, not to the computation that feeds it.
        data_hashes: data_hashes.into_iter().collect(),
        estimated,
        fixed: fixed.into_iter().collect(),
        resolved_priors,
        // gh#83/gh#85 step 9: top-level parameter provenance is populated by
        // the fit-finalization layer that owns the resolved-params view.
        parameters_provenance: Default::default(),
        // The observation/dimension schema — a pure fold over the model's
        // expanded observation leaves; emitted for every fit (not gated on
        // Bayesian-ness — an IF2 fit's streams/dims are just as describable).
        schema: model.map(crate::run_meta::ObsSchema::from_model),
        // gh#585: recorded post-load by `record_training_window` (the proof
        // must come from the code path that applied the declaration); sticky
        // across cache-hit reruns in `write_fit_sidecar`.
        training_window: None,
        // The `#'` doc dictionary (presentation metadata), loaded from the same
        // compiled IR. Empty when the model documents nothing.
        docs: crate::util::load_model_docs(model_src).unwrap_or_default(),
    }
}

fn format_prior(p: &Option<config_v2::EstimatePriorSpec>) -> String {
    match p {
        None => "(none)".to_string(),
        Some(spec) => crate::fit::config_diff::format_prior(spec),
    }
}

/// `camdl fit where FIT.toml [--seed N]`
///
pub fn cmd_fit_diff(args: &crate::args::FitDiffArgs) {
    use config_v2::FitConfig;

    let a_path = args.a.to_string_lossy().into_owned();
    let b_path = args.b.to_string_lossy().into_owned();
    let (a, a_method) = FitConfig::load(&a_path).map(|c| (c.problem, c.inference.method))
        .unwrap_or_else(|e| {
            eprintln!("error loading {}: {}", a_path, e);
            std::process::exit(1);
        });
    let (b, b_method) = FitConfig::load(&b_path).map(|c| (c.problem, c.inference.method))
        .unwrap_or_else(|e| {
            eprintln!("error loading {}: {}", b_path, e);
            std::process::exit(1);
        });

    println!("diff: {} → {}", a_path, b_path);
    println!();

    // Parameter changes
    let a_est: std::collections::BTreeSet<&str> = a.estimate.keys().map(|s| s.as_str()).collect();
    let b_est: std::collections::BTreeSet<&str> = b.estimate.keys().map(|s| s.as_str()).collect();
    let a_fixed = a.fixed.resolve().unwrap_or_default();
    let b_fixed = b.fixed.resolve().unwrap_or_default();
    let a_fix_keys: std::collections::BTreeSet<&str> = a_fixed.keys().map(|s| s.as_str()).collect();
    let b_fix_keys: std::collections::BTreeSet<&str> = b_fixed.keys().map(|s| s.as_str()).collect();

    let mut param_changes = false;
    // Moved from estimate → fixed
    for name in a_est.difference(&b_est) {
        if b_fix_keys.contains(name) {
            println!("  {}: [estimate] → [fixed] = {}", name, b_fixed.get(*name).unwrap());
            param_changes = true;
        }
    }
    // Moved from fixed → estimate
    for name in b_est.difference(&a_est) {
        if a_fix_keys.contains(name) {
            println!("  {}: [fixed] = {} → [estimate]", name, a_fixed.get(*name).unwrap());
            param_changes = true;
        }
    }
    // Fixed value changed
    for name in a_fix_keys.intersection(&b_fix_keys) {
        let va = a_fixed.get(*name).unwrap();
        let vb = b_fixed.get(*name).unwrap();
        if (va - vb).abs() > 1e-15 {
            println!("  {}: [fixed] {} → {}", name, va, vb);
            param_changes = true;
        }
    }
    // Bounds changed (Option-aware after bounds became optional in
    // [estimate.X]: a present↔omit transition is a real change because
    // omit means "fall back to model file's parameters block bounds").
    for name in a_est.intersection(&b_est) {
        let ab = a.estimate[*name].bounds;
        let bb = b.estimate[*name].bounds;
        let render = |o: Option<(f64, f64)>| match o {
            Some((lo, hi)) => format!("[{}, {}]", lo, hi),
            None => "(from model)".to_string(),
        };
        let differ = match (ab, bb) {
            (None, None) => false,
            (Some(a), Some(b)) => (a.0 - b.0).abs() > 1e-15 || (a.1 - b.1).abs() > 1e-15,
            _ => true,
        };
        if differ {
            println!("  {}: bounds {} → {}", name, render(ab), render(bb));
            param_changes = true;
        }
    }
    // Prior changes
    for name in a_est.intersection(&b_est) {
        let ap = &a.estimate[*name].prior;
        let bp = &b.estimate[*name].prior;
        let ap_str = format_prior(ap);
        let bp_str = format_prior(bp);
        if ap_str != bp_str {
            println!("  {}: prior {} → {}", name, ap_str, bp_str);
            param_changes = true;
        }
    }
    if !param_changes {
        println!("  (no parameter changes)");
    }

    // Method changes
    println!();
    println!("Method:");
    match (&a_method, &b_method) {
        (None, None) => println!("  (neither file declares a [method])"),
        (None, Some(m)) => println!("  [method]: (new) {}", m.algorithm.method_name()),
        (Some(_), None) => println!("  [method]: (removed)"),
        (Some(ma), Some(mb)) => {
            let changes = config_diff::method_setting_changes(ma, mb);
            if changes.is_empty() {
                println!("  (no method changes)");
            } else {
                for c in changes {
                    println!("  {}: {} → {}", c.key, c.from, c.to);
                }
            }
        }
    }
}

// ─── camdl fit new ──────────────────────────────────────────────────────────

pub fn cmd_fit_new(a: &crate::args::FitNewArgs) {
    let from = a.from.to_string_lossy().into_owned();
    let to   = a.dest.to_string_lossy().into_owned();

    if std::path::Path::new(&to).exists() {
        eprintln!("error: {} already exists. Choose a different name.", to);
        std::process::exit(1);
    }

    // Read source, inject provenance
    let mut content = std::fs::read_to_string(&from).unwrap_or_else(|e| {
        eprintln!("error reading {}: {}", from, e);
        std::process::exit(1);
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
        eprintln!("note: {} already has [provenance]. Update derived_from manually.", to);
    }

    // Best-effort: name the source fit's content-addressed segment, so the
    // derived file can warm-start from it by handle. The exact leaf path
    // (`{method}-{h8}/seed_N-{h8}`) needs the method + seed hashes, so we name
    // the segment and defer the leaf to `camdl list`.
    if let Ok(cfg) = config_v2::Problem::load(&from) {
        let seg = crate::util::load_model(&cfg.model.camdl).ok().and_then(|(m, _)| {
            let ir_version = ir::IR_VERSION.trim().to_string();
            let data_paths = cfg.data_spec().ok()
                .and_then(|ds| {
                    let names: Vec<String> =
                        m.observations.iter().map(|o| o.name.clone()).collect();
                    ds.effective_observations(&names).ok()
                })
                .unwrap_or_default();
            cas::fit_level_hash(&m, &ir_version, crate::version::VERSION_SHORT, &cfg, &data_paths)
                .ok()
                .map(|h| {
                    let root = crate::run_paths::output_root(None, cfg.output_dir.as_deref());
                    let stem = crate::hashing::path_stem_slug(&from)
                        .unwrap_or_else(|| "fit".to_string());
                    cas::fit_segment_dir(&root, &stem, &h)
                })
        });
        if let Some(seg) = seg {
            eprintln!("  [provenance] derived_from = \"{}\"", from);
            eprintln!("  hint: to warm-start the derived fit from {}, write under its \
                       [method]", seg.display());
            eprintln!("        starts = {{ from_posterior = \"@<label>\" }}   # one draw per chain");
            eprintln!("        starts = {{ from_mle = \"@<label>\" }}         # every chain at its estimate");
            eprintln!("        (a handle is @label, a fit-id prefix, or the leaf directory; \
                       `camdl list` shows them)");
        }
    }

    std::fs::write(&to, &content).unwrap_or_else(|e| {
        eprintln!("error writing {}: {}", to, e);
        std::process::exit(1);
    });

    eprintln!("created {}", to);
}

// ─── Labels (proposal §5) ─────────────────────────────────────────────

/// Validate a user-supplied label string against the proposal's
/// rule: 1–64 characters after trim, restricted to letters, digits,
/// spaces, commas, dot, underscore, hyphen. Returns the trimmed
/// label on success, or a descriptive Err message.
///
/// Why a custom regex check rather than a clap value parser: we
/// want the same validator on every `--label` flag (fit, simulate,
/// profile, …) and on `camdl label` at relabel time, with identical
/// error messages. A function call from each entry point is the
/// simplest way to keep them aligned.
pub fn validate_label(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("label is empty after trim — \
                    pass at least one printable character".into());
    }
    let n = trimmed.chars().count();
    if n > 64 {
        return Err(format!(
            "label is {} characters; max is 64 after trim", n));
    }
    for (i, c) in trimmed.chars().enumerate() {
        let ok = c.is_ascii_alphanumeric()
            || c == ' ' || c == ',' || c == '.' || c == '_' || c == '-';
        if !ok {
            return Err(format!(
                "label contains invalid character `{}` at position {} — \
                 allowed: letters, digits, spaces, commas, dot, underscore, hyphen",
                c, i + 1));
        }
    }
    Ok(trimmed.to_string())
}

/// Set or update the user-display label on any run kind (sim, fit,
/// profile, replicate-set, fit-stage).
///
/// Resolves the hash prefix by walking `<root>/{sims,fits,profiles}/**`
/// for `run.json` files whose `run_id` (or legacy `hash`) starts with the
/// prefix. The label is validated, written to the record's `label`, and
/// the run.json is rewritten atomically. Refuses to relabel a still-running
/// fit (`status == Running`).
///
/// Concurrent invocations are last-write-wins; we don't lock the
/// file. For single-user workflows this is fine; if cross-process
/// label edits ever become a concern, a flock on run.json is the
/// minimal extension.
pub fn cmd_label(args: &crate::args::LabelArgs) {
    let new_label = match validate_label(&args.label) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: invalid label: {}", e);
            std::process::exit(1);
        }
    };

    let root = args.root.clone();
    if !root.exists() {
        eprintln!("error: no results root at {}", root.display());
        std::process::exit(1);
    }

    // Match by hash prefix. Two homes for the label, by kind:
    //   - sims / pfilters / surveys → a per-leaf `run.json` whose
    //     `provenance.label` IS the label home; resolved by leaf `run_id`
    //     prefix via the same `cas_read::resolve_*_prefix` machinery `show`
    //     uses, so anything `show` can address, `label` can too.
    //   - fits (gh#147 M3.2) / profiles (M3.3) have no fit-wide `run.json` —
    //     their fit-/profile-level hash is derived from the leaves and the
    //     label lives in the base sidecar (`fit.meta.json`), so they resolve
    //     by base-hash prefix → segment and relabel that sidecar (NOT
    //     per-leaf: the label is a fit-wide mutable attribute with one home).
    use std::collections::HashSet;
    let mut matches: Vec<std::path::PathBuf> = Vec::new();
    let mut seen_leaves: HashSet<std::path::PathBuf> = HashSet::new();
    for leaf in crate::cas_read::resolve_sim_prefix(&root, &args.hash).into_iter()
        .chain(crate::cas_read::resolve_pfilter_prefix(&root, &args.hash))
        .chain(crate::cas_read::resolve_survey_prefix(&root, &args.hash))
    {
        if seen_leaves.insert(leaf.dir.clone()) {
            matches.push(leaf.dir);
        }
    }
    let mut fit_segments: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root.join("fits")) {
        for e in rd.flatten() {
            let seg = e.path();
            if !seg.is_dir() { continue; }
            if let Some(view) = crate::fit::fit_view::FitView::read(&seg) {
                if view.fit_hash.starts_with(&args.hash) {
                    fit_segments.push(seg);
                }
            }
        }
    }
    // Profiles (gh#147 M3.3): resolve by the profile-base hash (the `profile`
    // level of any leaf) → base segment; relabel via the same sidecar-rewrite
    // path as fits. Dedup so a multi-leaf profile yields one segment.
    {
        let mut seen: HashSet<std::path::PathBuf> = HashSet::new();
        for leaf in crate::cas_read::walk_profile_leaves(&root) {
            let base_hash = leaf.record.levels.first()
                .map(|l| l.hash.to_hex()).unwrap_or_default();
            if !base_hash.starts_with(&args.hash) { continue; }
            if let Some(seg) = leaf.dir.ancestors().nth(4) {
                if seen.insert(seg.to_path_buf()) {
                    fit_segments.push(seg.to_path_buf());
                }
            }
        }
    }

    let total = matches.len() + fit_segments.len();
    if total == 0 {
        eprintln!("error: no run found with hash prefix `{}` under {}",
            args.hash, root.display());
        std::process::exit(1);
    }
    if total > 1 {
        eprintln!("error: hash prefix `{}` matches {} runs — \
                   use a longer prefix", args.hash, total);
        for p in matches.iter().chain(fit_segments.iter()).take(8) {
            eprintln!("  {}", p.display());
        }
        std::process::exit(1);
    }

    // gh#147 (M3.2): the single match is a CAS fit segment — rewrite the
    // fit-level sidecar's label (its authoritative home), leaving the archived
    // `fit.toml.original` untouched.
    if let Some(seg) = fit_segments.into_iter().next() {
        let mut side = crate::run_meta::read_fit_sidecar(&seg).unwrap_or_default();
        let prior = side.label.clone();
        side.label = Some(new_label.clone());
        if let Err(e) = crate::run_meta::write_fit_sidecar(
            &seg, std::path::Path::new(&side.fit_toml_path), &side,
        ) {
            eprintln!("error: cannot write fit-level sidecar {}: {}", seg.display(), e);
            std::process::exit(1);
        }
        match prior {
            Some(p) if p != new_label =>
                eprintln!("ok: label updated from \"{}\" to \"{}\" on {}", p, new_label, seg.display()),
            Some(_) => eprintln!("ok: label unchanged (\"{}\") on {}", new_label, seg.display()),
            None => eprintln!("ok: label set to \"{}\" on {}", new_label, seg.display()),
        }
        return;
    }

    let run_dir = matches.into_iter().next().unwrap();
    let run_json_path = run_dir.join("run.json");

    // Per-leaf kinds (sim / pfilter / survey): the label lives in the leaf's
    // `run.json` `provenance.label`. These are always written `Completed`, so
    // there is no in-progress gate. Rewrite atomically (write tmp + rename).
    if let Ok(txt) = std::fs::read_to_string(&run_json_path) {
        if let Ok(mut rec) = serde_json::from_str::<runid::RunRecord>(&txt) {
            let prior = rec.provenance.label.clone();
            rec.provenance.label = Some(new_label.clone());
            let tmp = run_dir.join("run.json.tmp");
            let json = serde_json::to_string_pretty(&rec).unwrap_or_default();
            if let Err(e) = std::fs::write(&tmp, json).and_then(|_| std::fs::rename(&tmp, &run_json_path)) {
                eprintln!("error: cannot write {}: {}", run_json_path.display(), e);
                std::process::exit(1);
            }
            match prior {
                Some(p) if p != new_label =>
                    eprintln!("ok: label updated from \"{}\" to \"{}\" on {}", p, new_label, run_dir.display()),
                Some(_) => eprintln!("ok: label unchanged (\"{}\") on {}", new_label, run_dir.display()),
                None => eprintln!("ok: label set to \"{}\" on {}", new_label, run_dir.display()),
            }
            return;
        }
    }

    // Leaf matches come from `resolve_*_prefix`, which only surfaces dirs
    // holding a parseable `RunRecord` `run.json` (handled above). A match
    // whose `run.json` failed to re-parse here is malformed — surface it
    // rather than fabricating a label write.
    eprintln!(
        "error: {} is not a recognized (new-format) run.json — cannot relabel",
        run_json_path.display());
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gh#191: per-stage capability gate on `fit run` ─────────────

    /// A chain_binomial IF2 method — the inference path that carries no
    /// real reservoir state (gh#191).
    fn chain_binomial_if2_stage() -> config_v2::Algorithm {
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
        }
    }

    /// An `mh`-on-ODE stage carrying `burnin_dt`, for the gh#449 warm-up-step
    /// checks. Only `backend` and `burnin_dt` matter to the gate.
    fn mh_stage_with_burnin_dt(burnin_dt: Option<f64>) -> config_v2::Algorithm {
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
    fn fit_run_rejects_real_compartments_on_chain_binomial_stage() {
        // gh#191: `fit run` never gated the model against the per-stage
        // backend, so a real-compartment (ODE-coupled) model on a
        // chain_binomial inference stage was silently mis-fit — the filter
        // loops freeze the real reservoir at its init value. The fit-run
        // path must REJECT it with the REAL_COMPARTMENTS message, naming the
        // offending stage.
        let compiled = compiled_golden("ocaml/golden/sir_reservoir_mixed.ir.json");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::REAL_COMPARTMENTS),
            "fixture must actually require REAL_COMPARTMENTS"
        );
        let stage = chain_binomial_if2_stage();
        let err = gate_run_method_against_model(&stage, &compiled, 1.0)
            .expect_err("real-coupled model on a chain_binomial fit stage must be rejected");
        assert!(err.contains("gh#191"), "should cite the tracking issue: {err}");
        assert!(
            err.contains("frozen"),
            "should explain the frozen-reservoir reason: {err}"
        );
        assert!(err.contains("if2"), "should name the offending method: {err}");
    }

    #[test]
    fn fit_run_accepts_balance_on_chain_binomial_stage() {
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
        let stage = chain_binomial_if2_stage();
        gate_run_method_against_model(&stage, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("fit run must ACCEPT a balance{{}} model on a chain_binomial stage: {e}")
        });
    }

    // ── gh#449: recurring-fire collision guard on the FIT path ────

    /// gh#449 (follow-up to gh#447). The recurring-fire collision guard was
    /// added to `CompiledModel::validate_schedule`, which the three forward
    /// backends call at entry — but nothing on the inference/fit path calls it.
    /// The PGAS producer goes straight to `resolve_fire_steps(dt, params)`,
    /// whose dedup `BTreeSet` is exactly where a colliding fire is silently
    /// dropped. So a coarse-`dt` `fit` still silently drops recurring fires,
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
            compiled.validate_recurring_dt_collisions(400.0).is_err(),
            "precondition: dt=400 must collide on this model"
        );

        let stage = chain_binomial_if2_stage();
        let err = gate_run_method_against_model(&stage, &compiled, 400.0)
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
        let stage = chain_binomial_if2_stage();
        gate_run_method_against_model(&stage, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("dt=1 is far finer than the 365.25-day period and must be accepted: {e}")
        });
    }

    /// An `mh` stage with a coarse `burnin_dt` — the case gh#449 does not
    /// mention and gh#447 could not have, since `burnin_dt` (gh#396) postdates
    /// it. `burnin_dt` exists to be COARSER than `dt` on the unscored warm-up,
    /// so a schedule that is perfectly safe at the run's `dt` can still drop
    /// recurring fires during warm-up. The gate must check both step sizes.
    #[test]
    fn fit_run_rejects_coarse_burnin_dt_even_when_dt_is_fine() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        let stage = mh_stage_with_burnin_dt(Some(400.0));
        // dt = 1.0 is fine on its own — proven by the negative control above —
        // so any rejection here can only come from `burnin_dt`.
        let err = gate_run_method_against_model(&stage, &compiled, 1.0).expect_err(
            "a coarse burnin_dt that drops recurring fires must be rejected even when dt is fine",
        );
        assert!(err.contains("mh"), "should name the offending method: {err}");
        assert!(
            err.contains("burnin_dt"),
            "should name burnin_dt as the offending step, not dt: {err}"
        );
        assert!(
            err.contains("silently dropped"),
            "should carry the collision diagnostic: {err}"
        );
    }

    /// Negative control for the burnin_dt arm: the same stage with no
    /// `burnin_dt` set must pass at the same `dt`, so the test above is
    /// attributable to `burnin_dt` and not to the stage kind.
    #[test]
    fn fit_run_accepts_an_mh_stage_without_burnin_dt() {
        let compiled = compiled_golden("ocaml/golden/seir_seasonal_importation.ir.json");
        let stage = mh_stage_with_burnin_dt(None);
        gate_run_method_against_model(&stage, &compiled, 1.0).unwrap_or_else(|e| {
            panic!("an mh stage with no burnin_dt must be accepted at dt=1: {e}")
        });
    }

    // ── validate_label ────────────────────────────────────────────

    #[test]
    fn validate_label_accepts_canonical_examples() {
        // The `--label` documentation lists these as the expected
        // shapes; assert each one is accepted with the trimmed value
        // returned verbatim.
        for ok in [
            "narrow R0, take 1",
            "iota free",
            "log_normal R0 prior",
            "take 1, attempt 2",
            "a",                    // single char (min length)
            "a-b_c.d 0,1",          // every allowed punctuation
        ] {
            let out = validate_label(ok)
                .unwrap_or_else(|e| panic!("`{}` should validate; got error: {}", ok, e));
            assert_eq!(out, ok);
        }
    }

    #[test]
    fn validate_label_trims_surrounding_whitespace() {
        let out = validate_label("   narrow R0   ").unwrap();
        assert_eq!(out, "narrow R0");
    }

    #[test]
    fn validate_label_rejects_empty_after_trim() {
        for empty in ["", "   ", "\t \n"] {
            let err = validate_label(empty).expect_err("empty must reject");
            assert!(err.contains("empty"), "err should mention empty: {}", err);
        }
    }

    #[test]
    fn validate_label_rejects_over_64_chars() {
        let too_long: String = "a".repeat(65);
        let err = validate_label(&too_long).expect_err("65-char label must reject");
        assert!(err.contains("64"), "err should mention max length: {}", err);
    }

    #[test]
    fn validate_label_accepts_64_chars_exactly() {
        let just_right: String = "a".repeat(64);
        validate_label(&just_right).expect("64-char label should validate");
    }

    #[test]
    fn validate_label_rejects_disallowed_characters() {
        // Each of these contains exactly one disallowed char; the
        // error message should call it out by character + position.
        for (raw, bad_char) in [
            ("R0/2",      "/"),
            ("alpha=2",   "="),
            ("name:tag",  ":"),
            ("a;b",       ";"),
            ("a*b",       "*"),
            ("emoji 🎯",   "🎯"),
        ] {
            let err = validate_label(raw)
                .expect_err(&format!("`{}` should reject", raw));
            assert!(err.contains(bad_char),
                "err for `{}` should call out `{}` by character; got: {}",
                raw, bad_char, err);
        }
    }

    // ── gh#771: the sidecar hashes the RESOLVED stream set ─────────

    /// Deserialize a golden IR envelope into an `ir::Model` (no compile —
    /// only the declared observation names are read here).
    fn golden_model(rel: &str) -> ir::Model {
        let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let envv: serde_json::Value =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"));
        serde_json::from_value(envv["model"].clone())
            .unwrap_or_else(|e| panic!("deserialize {path}: {e}"))
    }

    #[test]
    fn sidecar_hashes_every_stream_under_the_single_file_shorthand() {
        // gh#771: `[data] file = "..."` binds one wide TSV to every
        // observation stream the model declares. Fit IDENTITY resolves that
        // shorthand through `effective_observations`; the sidecar hashed
        // `[data.observations]` — empty under the shorthand — so
        // `fit.meta.json` carried no `data_hashes` at all, and the
        // `camdl compare` data preflight reported the fit as unchecked.
        let tmp = tempfile::tempdir().expect("tempdir");
        let data_path = tmp.path().join("wide.tsv");
        std::fs::write(&data_path, b"time\tweekly_cases\tdetection\n1\t5\t0.4\n")
            .expect("write data");
        let fit_path = tmp.path().join("fit.toml");
        // `.ir.json` (not `.camdl`) and non-existent: `build_fit_sidecar`
        // falls back to an empty model identity rather than shelling out to
        // camdlc. The model under test is passed in directly.
        let toml_src = format!(
            r#"
[model]
camdl = "no-such-model.ir.json"

[data]
file = "{}"

[estimate]
beta = {{ bounds = [0.01, 2.0] }}

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
        let config = config_v2::FitConfig::from_toml_str(&toml_src)
            .expect("fixture fit.toml should parse");

        let model = golden_model("ocaml/golden/seir_observations.ir.json");
        let stream_names: Vec<String> =
            model.observations.iter().map(|o| o.name.clone()).collect();
        assert_eq!(stream_names.len(), 2, "fixture must declare two streams");

        let sidecar = build_fit_sidecar(
            &config, fit_path.to_str().unwrap(), None, Some(&model));

        let expected = crate::hashing::file_hash(data_path.to_str().unwrap())
            .expect("the fixture data file must hash");
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
}

