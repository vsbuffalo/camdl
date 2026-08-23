//! `camdl fit` — structured inference workflow.
//!
//! Single entry point: `camdl fit run FIT.toml [--seed N] [--stage NAME]
//! [--label "..."] [--force]`. The fit.toml v2 schema declares stages
//! inline; the runner walks them in order. See
//! `docs/dev/proposals/2026-04-15-fit-run-spec-v0.4.md`.

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
    config: &config_v2::FitConfigV2,
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
pub mod pgas;
pub mod nuts;  // gh#275 Phase 2: nuts on ode
pub mod trace_writer;
pub mod synthetic;
pub mod gating;
pub mod chain_diagnostics;  // gh#406: per-chain loglik outlier z-scores (read-side)
pub mod dt_check;
pub mod init;
pub mod chain_starts;
pub mod loglik_eval;
pub mod methods;
#[cfg(feature = "ode")]
pub mod nlopt_stage;
pub mod handle;   // gh#322: fit handles (@label / hash / run-dir / fit.toml) → segment
pub mod joint;    // gh#322: keyed-joint (θ, X) read — LatentPath classifier + join
pub mod predict;  // `camdl fit predict`: free-forward posterior predictive verb + types
pub mod contrasts; // gh#322: counterfactual `contrasts {}` two-arm replay reducer (stage C)

/// `camdl fit methods` — print the supported (algorithm, backend) pairs.
/// Reads from `methods::METHODS`, the single source of truth.
pub fn cmd_fit_methods() {
    print!("{}", methods::render_matrix());
}

// ─── New `camdl fit run` entry point (config_v2) ────────────────────────────

/// gh#191: the model-capability gate must run on the fit-run path, PER STAGE.
/// Each stage carries its own simulation backend (a config can mix e.g. an
/// `ode` nl-sbplx scout with a `chain_binomial` pgas refine), so gating once
/// against a single backend would be wrong; we check every stage's declared
/// backend against the compiled model. Returns the first offending stage's
/// message (prefixed with the stage name) so the user knows where to look.
fn gate_run_stages_against_model(
    stages: &[(&str, &config_v2::Stage)],
    compiled: &sim::CompiledModel,
    dt: f64,
) -> Result<(), String> {
    for (stage_name, stage) in stages {
        if let Err(msg) = methods::check_model_capabilities(stage.backend(), compiled) {
            return Err(format!("stage '{}': {}", stage_name, msg));
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
            config_v2::Stage::Mh { burnin_dt, .. }
            | config_v2::Stage::Nuts { burnin_dt, .. } => *burnin_dt,
            _ => None,
        };
        for (label, step) in [("dt", Some(dt)), ("burnin_dt", burnin_dt)] {
            let Some(step) = step else { continue };
            if let Err(e) = compiled.validate_recurring_dt_collisions(step) {
                return Err(format!("stage '{}' ({} = {}): {}", stage_name, label, step, e));
            }
        }
    }
    // gh#166 B2: warn (once) if any ODE-backed stage will fit a `dt`-in-rate model
    // with first-order Euler incidence (the high-order augmented flow is undefined
    // when a rate depends on the step size).
    if stages.iter().any(|(_, s)| s.backend() == crate::run_meta::InferenceBackend::Ode) {
        methods::warn_if_ode_euler_flow(compiled);
    }
    Ok(())
}

pub fn cmd_fit_run_v2(a: &crate::args::FitRunArgs) {
    use config_v2::{FitConfigV2, Stage, StartsFrom};

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
    // M-1 break per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
    // §"Migration": fail loudly on removed flags before any work.
    if let Some(raw) = a._removed_starts_from.as_deref() {
        eprintln!(
            "error: --starts-from is no longer accepted on `camdl fit run`. \
             Replacement:\n  \
             --init from_mle --mle <fit-dir>       \
                 (warm-start every chain from a prior fit's MLE)\n  \
             --init from_params --params <toml>    \
                 (warm-start from a hand-written params TOML)\n\
             Saw --starts-from {}.\n\
             See `camdl fit run --help` (INIT MODES section).",
            raw);
        std::process::exit(1);
    }
    if let Some(raw) = a._removed_init_method.as_deref() {
        eprintln!(
            "error: --init-method is no longer accepted on `camdl fit run`. \
             It was renamed to --init for parity with `camdl profile`.\n\
             Saw --init-method {}.\n\
             See `camdl fit run --help` (INIT MODES section).",
            raw);
        std::process::exit(1);
    }
    let fit_path              = a.config.to_string_lossy().into_owned();
    let base_seed             = a.seed.unwrap_or(1);
    let force                 = a.force;
    let stage_filter          = a.stage.clone();
    // `--init from_mle --mle <dir>` is the new spelling for the
    // pre-rev-2 `--starts-from <dir>` flag. The mle-fitdir path
    // becomes the "all chains at prior-fit MLE" warm start that the
    // dispatcher's `starts_from_override` already implements.
    let starts_from_override: Option<String> = a.mle.as_ref()
        .filter(|_| matches!(a.init,
            Some(crate::args::InitModeTag::FromMle)))
        .map(|p| {
            let s = p.to_string_lossy().to_string();
            resolve_starts_from_arg(&s)
        });
    let allow_nonconverged_scout = a.allow_nonconverged_scout;
    // Construct the full InitMethod payload from the CLI tag +
    // companion paths. When `--init from_mle` is used the path-bearing
    // variant is consumed by `starts_from_override` above (the legacy
    // dispatcher's `--starts-from` semantic). For the other warm-start
    // variants (from_prior / from_posterior / from_params) the typed
    // InitMethod flows into `cli_init_method`, which reaches every
    // stage through `CliStageOverrides::init` -> `apply_cli_overrides`
    // (gh#514) — the stage's own `init_method` field carries it from
    // there.
    let cli_init_method: Option<crate::fit::init::InitMethod> = match a.init {
        Some(tag) if !matches!(tag, crate::args::InitModeTag::FromMle) => {
            Some(tag.to_init_method(
                a.posterior.as_ref(),
                a.mle.as_ref(),
                a.init_params.as_ref(),
            ).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }))
        }
        _ => None,
    };

    // gh#540: every CLI flag that changes what a stage computes or stores,
    // collected once so the stage identity can see all of them. Applied to the
    // in-memory stage below, before the CAS claim — the dispatch sites no
    // longer override anything, so there is no route around the key.
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
        init:                 cli_init_method.clone(),
        survey_path:          a.survey_path.clone(),
        survey_top_k:         a.survey_top_k,
        tempering:            a.tempering.clone(),
        max_tree_depth:       a.max_tree_depth,
        trajectory_warmup:    a.trajectory_warmup,
        csmc_sweeps_per_nuts: a.csmc_sweeps_per_nuts,
        n_trajectories:       a.n_trajectories,
        diagonal_mass:        a.diagonal_mass,
        no_nuts:              a.no_nuts,
        no_adapt:             a.no_adapt,
        adapt_start:          a.adapt_start,
        rho:                  a.rho,
        cooling_target_iters: a.cooling_target_iters,
        decibans_thresh:      a.decibans_thresh,
        no_dt_check:          a.no_dt_check,
        dt_check_halvings:    a.dt_check_halvings,
        record_ancestry:      a.record_ancestry,
        record_prequential:   a.record_prequential,
    };
    let sweep_specs: Vec<(String, Vec<f64>)> = a.sweep.iter()
        .map(|s| (s.name.clone(), s.grid.expand()))
        .collect();

    // Load v2 config
    let mut config = FitConfigV2::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#audit-C6 / gh#189: degenerate-rate handling is a keyed `[config]` field
    // (folds into the fit-identity hash via the config blob), set before any rate
    // evaluation. Was a CLI flag, which silently bypassed the run_id.
    sim::eval_stats::set_allow_degenerate_rates(config.config.allow_degenerate_rates);

    // gh#134: `--condition-from` mirrors the top-level fit.toml `condition_from`
    // and OVERRIDES it. Written into the in-memory config BEFORE the
    // fit-identity hash is computed (`cas::fit_level_hash` serializes this
    // config via `fit_config_blob_hash`), so a CLI-set conditioning window
    // re-keys the fit exactly as a toml-set one does — no silent identity
    // bypass. The CLI carries one value, so it always sets the all-streams
    // default (`ConditionFrom::All`); per-stream shadows are toml-only. The
    // spec string (bare number / date / `first_obs - <N> <unit>`) is resolved
    // per stream at build time.
    if let Some(raw) = &a.condition_from {
        config.condition_from = Some(config_v2::ConditionFrom::All(raw.trim().to_string()));
    }

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
    if emit_every.is_some() && config.synthetic.is_none() {
        eprintln!(
            "error: --emit-every sets the cadence at which SYNTHETIC \
             observations are generated, and {} declares no `[synthetic]` \
             block.\n  \
             A fit against real data scores each stream at its own data file's \
             times — `emit_schedule` never enters the likelihood — so this flag \
             would change nothing.\n  \
             Drop the flag, or add a `[synthetic]` block to generate data.",
            fit_path
        );
        std::process::exit(1);
    }

    // gh#514: the same treatment for the chain-start overrides. `--init`,
    // `--posterior`, `--mle`, `--params`, `--survey-path` and `--survey-top-k`
    // all change where the chains begin, and therefore the stored output — but
    // they used to be applied at the DISPATCH site, well after the CAS claim,
    // while the stage's `identity_payload` carried only the toml's values. Two
    // runs differing solely in `--init` shared a run_id, and the second was
    // served the first's result with a "cache hit" line and no warning.
    // `--force` did not rescue it either: that path errors with "artifact
    // already completed".
    //
    // Clap requires `--stage` alongside each of these, so exactly one stage is
    // targeted. Writing them here — before `cas::fit_level_hash` and before the
    // per-stage claim — makes a different override a different artifact.
    // Leaving them unset leaves the toml's values in place, so a run with no
    // overrides keys exactly as it did before and no cached fit is invalidated.
    if let Some(stage_name) = a.stage.as_deref() {
        if !cli_overrides.is_empty() {
            match config.stages.get_mut(stage_name) {
                Some(stage) => stage.apply_cli_overrides(&cli_overrides)
                    .unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    }),
                None => {
                    eprintln!("error: --stage '{}' is not a stage in {} \
                               (stages: {})",
                        stage_name, fit_path,
                        config.stages.keys().cloned().collect::<Vec<_>>().join(", "));
                    std::process::exit(1);
                }
            }
        }
    }

    // Compile `model.camdl` → IR EXACTLY ONCE for the whole fit. Every
    // per-(cell × sweep point × stage) `FitRunConfig::build` then loads this
    // pre-compiled IR instead of re-invoking camdlc per unit (a multi-stage
    // swept fit otherwise recompiled the model dozens of times). The resolved
    // path is recorded on `config.compiled_ir`; `config.model.camdl` is left
    // untouched so the fit's content hash still hashes the original `.camdl`
    // source bytes (identity is unchanged by the hoist). The temp IR persists
    // for the process — `resolve_ir_path`'s returned path is not a drop guard.
    //
    // gh#439 A2: only `nuts` on the `ode` backend reads the WrtPop state-Jacobian
    // (`rate_state_grad` / `projection_state_grad`, via the ODE forward-sensitivity
    // gradient in `ode_grad::det_grad`). If no stage is nuts+ode, compile lean
    // (`--no-state-grad`), which omits the dense ~O(G^3) Jacobian that dominates
    // coupled-model IR. The bit is folded into the IR-cache key, so a lean entry is
    // never reused for a nuts+ode fit (and run identity is gradient-independent, so
    // lean vs full share the same model digest).
    let needs_state_grad = config.needs_state_grad();
    let (compiled_ir, _ir_tmp) = crate::util::resolve_ir_path(&config.model.camdl, needs_state_grad)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    config.compiled_ir = Some(compiled_ir.clone());

    // gh#616: if the model declares observation anchors, resolve them ONCE here
    // — from this fit's own `[data]` — and re-emit the compiled IR with the
    // anchors substituted, so every downstream loader (the runner, each sweep
    // point, each stage, the archived copy) reads the SAME resolved model.
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
    let compiled_ir = resolve_anchors_into_temp_ir(&compiled_ir, &config)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    config.compiled_ir = Some(compiled_ir.clone());

    // Load model and validate completeness (from the pre-compiled IR).
    let (model, _) = crate::util::load_model(&compiled_ir).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#33: expand `[fixed] from_scenario = "name"` into the inline
    // values map by looking up the named scenario in the model. Must
    // happen after model load but before validate, so the every-param-
    // resolved check sees the expanded values.
    config.expand_fixed_from_scenario(&model).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_params: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
    config.validate(&model_params).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#75: prior-presence check honoring the IR fallback. Has to run
    // after partition/dag validation but with the model IR in scope.
    let ir_prior_params: std::collections::BTreeSet<&str> = model.parameters.iter()
        .filter(|p| p.prior_dist().is_some() || p.hierarchical().is_some())
        .map(|p| p.name.as_str())
        .collect();
    config.validate_priors_present(&ir_prior_params).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    if let Some(msg) = config.dangling_priors_warning() {
        // Warning, not error: a staged Bayesian workflow (scout → pgas)
        // legitimately declares priors that the IF2 stage ignores — so
        // we can't refuse here. But silent would hide the copy-paste /
        // mental-model-mismatch class of bug that's the actual risk.
        eprintln!("\x1b[33mwarning:\x1b[0m {}", msg);
    }
    if let Some(msg) = config.single_init_multichain_warning() {
        // gh#71: single-init multi-chain posterior runs produce an
        // uninformative R̂. Warning, not error — the sample is still
        // valid; only the convergence diagnostic is weakened.
        eprintln!("\x1b[33mwarning:\x1b[0m {}", msg);
    }

    // ── Validate sweeps ───────────────────────────────────────────────────
    // Validate: swept params must be in [fixed], not [estimate]
    let fixed_resolved = config.fixed.resolve().unwrap_or_default();
    for (name, _) in &sweep_specs {
        if config.estimate.contains_key(name) {
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

    // Validate --starts-from requires --stage
    if starts_from_override.is_some() && stage_filter.is_none() {
        eprintln!("error: --starts-from requires --stage to disambiguate which stage it applies to.");
        std::process::exit(1);
    }

    // Validate --resume requires a PGAS or PMMH stage. Other methods
    // have no extension dimension (IF2's cooling depends on total
    // iterations, PFilter is single-pass), so resuming would be
    // statistically incoherent.
    if a.resume.is_some() {
        if let Some(ref name) = stage_filter {
            match config.stages.get(name.as_str()) {
                Some(s) if matches!(s, Stage::PGAS { .. } | Stage::PMMH { .. }) => {}
                Some(s) => {
                    eprintln!("error: --resume is only supported for PGAS and PMMH stages; \
                               '{}' is method '{}'.", name, s.method_name());
                    std::process::exit(1);
                }
                None => {} // The stage_filter check below will report this.
            }
        }
    }

    // Determine which stages to run
    let stages_to_run: Vec<(&str, &Stage)> = if let Some(ref name) = stage_filter {
        match config.stages.get(name.as_str()) {
            Some(stage) => vec![(name.as_str(), stage)],
            None => {
                let available: Vec<&str> = config.stages.keys().map(|s| s.as_str()).collect();
                eprintln!("error: stage '{}' not found. Available: {}", name, available.join(", "));
                std::process::exit(1);
            }
        }
    } else {
        config.stages.iter().map(|(k, v)| (k.as_str(), v)).collect()
    };

    // gh#191: gate the model's required capabilities against EACH stage's
    // declared backend, before any fitting work. `profile` already runs this
    // check, but `fit run` never did — so a real-compartment (ODE-coupled)
    // model on a chain_binomial inference stage was silently mis-fit (the
    // filter loops freeze the real reservoir at its init value). Fail fast
    // with the actionable per-stage message instead.
    {
        // `required_capabilities()` is STRUCTURAL (transitions / compartments /
        // balance) — the parameter VALUES are irrelevant to it. But
        // `CompiledModel::new` requires every parameter to carry a value, and
        // estimated parameters carry `value = None` in the IR (their start is
        // resolved per-stage from `[estimate].start` later). So fill any
        // value-less parameter with a harmless placeholder purely for this
        // capability scan — without it the gate errored "parameter '<estimated>'
        // has no value" on every `init = survey_top_k` / estimate-only fit
        // (gh#191: the gate must not demand resolved params it doesn't use).
        let mut cap_model = model.clone();
        for p in &mut cap_model.parameters {
            if p.value.resolved_value().is_none() {
                let placeholder = p.initial_value()
                    .or_else(|| p.bounds().map(|(lo, hi)| 0.5 * (lo + hi)))
                    .unwrap_or(1.0);
                p.value = p.value.with_value(placeholder);
            }
        }
        let compiled = sim::CompiledModel::new(cap_model).unwrap_or_else(|e| {
            eprintln!("error: {:?}", e);
            std::process::exit(1);
        });
        if let Err(msg) = gate_run_stages_against_model(&stages_to_run, &compiled, config.config.dt) {
            eprintln!("error: {}", msg);
            std::process::exit(1);
        }
    }

    // gh#audit-H12: --record-prequential and --record-ancestry only have
    // effect inside the Stage::PFilter arm of the stage match. clap
    // enforces `requires = "stage"`, but doesn't validate that the named
    // stage *resolves* to a PFilter — so before this check, passing the
    // flags with `--stage scout` (or any non-PFilter stage) silently
    // dropped them. List the available PFilter stages on error so the
    // user knows what they should have passed.
    if a.record_prequential || a.record_ancestry {
        let on_pfilter = stages_to_run.iter()
            .all(|(_, s)| matches!(s, Stage::PFilter { .. }));
        if !on_pfilter {
            let pf_stages: Vec<&str> = config.stages.iter()
                .filter(|(_, s)| matches!(s, Stage::PFilter { .. }))
                .map(|(k, _)| k.as_str())
                .collect();
            let flag = if a.record_prequential { "--record-prequential" } else { "--record-ancestry" };
            if pf_stages.is_empty() {
                eprintln!("error: {} requires --stage <pfilter-stage>, \
                    but this fit config has no PFilter stages.", flag);
            } else {
                eprintln!("error: {} requires --stage <pfilter-stage>. \
                    Available PFilter stages in this config: {}",
                    flag, pf_stages.join(", "));
            }
            std::process::exit(1);
        }
    }

    // gh#604: every key the user typed under `[data.observations]` /
    // `[data.holdout]` must name a declared observation source. Checked HERE —
    // before the identity digests below read a single byte — because those
    // digests open each bound path, so an unbound key would otherwise surface
    // as "cannot read data file '<key>'", diagnosing a missing file when the
    // real fault is a binding that names no stream. The motivating case is a
    // top-level key (`condition_from`) written below the `[data.observations]`
    // header, which TOML scopes into the table.
    if let Ok(ds) = config.data_spec() {
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
    // Per-stage run.json records live inside each stage dir; the fit as a
    // whole is the `fits/{stem}-{h8}/` path segment, not a separate record
    // (see the note below where `build_fit_run` is called). The
    // seed-independent parent fit hash computed here is reused by every
    // stage as its `fit`-level hash — computing it once avoids the
    // O(stages × full-I/O rehash) pattern.
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
    // (grid summary, sweep_failures, synthetic data) live directly under it.
    // The fit-level identity (the `fit` CAS level) and the directory its
    // stage leaves actually land in. `parent_fit_hash` is the fit-level
    // `ContentHash` (the same hash `resolve_fit_stage` puts on the `fit`
    // level, and the same one `FitView.fit_hash` reads back from `run.json`),
    // and `announced_fit_dir` is `fits/{stem}-{h8}/` built from it — so the
    // path `fit run` announces is exactly where the `NN-stage-{h8}` leaves
    // are written.
    //
    // Real and synthetic share this one `runid` fit-level digest. A real fit
    // folds its base `[data]` stream digests; a synthetic fit has no input
    // data (it generates data per-cell from the model + `[synthetic]`, both
    // already in the digest), so it hashes with an EMPTY data map. Either way
    // the container is keyed on model + config + engine, and the per-cell
    // stage leaves resolve their own segment (folding the generated data).
    let announce_cas_root = crate::run_paths::output_root(None, config.output_dir.as_deref());
    let announce_stem = crate::hashing::path_stem_slug(&fit_path)
        .unwrap_or_else(|| "fit".to_string());
    let announce_ir_version = ir::IR_VERSION.trim().to_string();
    // Real fits resolve their base `[data]` streams; synthetic fits have none
    // (empty map → no data digests folded).
    let fit_data_paths: indexmap::IndexMap<String, String> = config.data_spec().ok()
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
        &config,
        &fit_data_paths,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: fit-level identity: {}", e);
        std::process::exit(1);
    });
    let announced_fit_dir = cas::fit_segment_dir(
        &announce_cas_root, &announce_stem, &parent_fit_hash_ch);
    let parent_fit_hash = parent_fit_hash_ch.to_hex();
    // Synthetic-data generation + fit-level outputs (grid summary,
    // sweep_failures) write under the same content-addressed segment.
    let fit_dir = announced_fit_dir.clone();

    let fit_sidecar = build_fit_sidecar(&config, &fit_path, validated_label, Some(&model));

    eprintln!("fit: {} ({} stage{})",
        fit_path,
        stages_to_run.len(),
        if stages_to_run.len() == 1 { "" } else { "s" },
    );
    eprintln!("  model:    {}", config.model.camdl);
    eprintln!("  estimate: {}", config.estimate.keys()
        .map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    eprintln!("  fixed:    {}", {
        let resolved = config.fixed.resolve().unwrap_or_default();
        resolved.keys().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    });
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
    // docs/dev/proposals/2026-04-18-ic-free-inference.md.
    if config.ic_free.unwrap_or(false) {
        let ivp_params: Vec<&str> = config.estimate.iter()
            .filter(|(_, spec)| spec.ivp)
            .map(|(n, _)| n.as_str())
            .collect();
        eprintln!("\n  \x1b[36mic-free inference:\x1b[0m conditioning on y₁");
        eprintln!("    - initial state spread from ivp params: [{}]", ivp_params.join(", "));
        eprintln!("    - log-likelihood accumulation from t = 2 (y₁ reweights and resamples only)");
    }

    // ── Build the replicate grid: (dataset_idx, fit_seed) cells ──────────
    //
    // Four canonical modes, all routed through the same grid. Each cell is a
    // content-addressed fit (its own FitDigest base); the legacy literal cell
    // dirs (`real/fit_<seed>/`, `synthetic/ds_NN/fit_<seed>/`) are retired.
    //   Mode                           synthetic?  fit_seeds     Cells
    //   Single fit                     no          None/scalar   1
    //   Start-sensitivity              no          list of M     M  (seed levels, one base)
    //   Parameter recovery             yes         None/scalar   N  (one base per dataset)
    //   Parameter recovery × starts    yes         list of M     N × M
    //
    // For synthetic modes the datasets are generated once up front and the
    // per-cell DataSpec is materialised from their on-disk paths. See
    // docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.
    //
    // Multi-seed fits produce sibling CAS cells under one `fit`-level base
    // (the seed is a factored `runid` level), with no cross-cell aggregator.
    // A cross-seed roll-up view (per-stage chain-R̂ across fit_seeds) is a
    // derived index, deferred to M4 (gh#154) — the same home as the profile
    // and grid roll-ups. The same applies to `dataset_idx` for synthetic fits.
    let fit_seeds: Vec<u64> = match &config.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    let synthetic_datasets: Vec<synthetic::SyntheticDataset> = if let Some(spec) = &config.synthetic {
        let datasets = synthetic::generate_synthetic_datasets(
            spec,
            // Pre-compiled IR (compiled once above) so per-dataset generation
            // doesn't re-invoke camdlc; falls back to the source path.
            config.compiled_ir.as_deref().unwrap_or(&config.model.camdl),
            &fit_dir,
            config.config.dt,
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
    } else {
        Vec::new()
    };

    // A cell is one (data_source, fit_seed) pair. Real-data cells carry
    // `dataset_idx = None` and leave the existing `config.data` in place;
    // synthetic cells carry `Some(idx)` and replace `config.data` with a
    // DataSpec pointing at the generated TSV.
    struct Cell {
        dataset_idx: Option<usize>,
        fit_seed: u64,
        // None → keep config.data; Some → overwrite with synthetic path.
        data_override: Option<config_v2::DataSpec>,
    }
    let cells: Vec<Cell> = if synthetic_datasets.is_empty() {
        fit_seeds.iter().map(|&s| Cell {
            dataset_idx: None,
            fit_seed: s,
            data_override: None,
        }).collect()
    } else {
        // Determine the observation stream name(s) for the generated TSVs
        // from the model itself — synthetic generation writes one column
        // per declared observation block, so the fit data map points each
        // stream name at the same ds_NN.tsv file (the data loader picks
        // its named column).
        // Reuse the already-loaded model (compiled once above); no extra
        // camdlc call. Observation-block names are structural, scenario-
        // independent, so the validation-load model is the right source.
        let obs_names: Vec<String> = model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let mut out = Vec::with_capacity(synthetic_datasets.len() * fit_seeds.len());
        for ds in &synthetic_datasets {
            let mut observations = indexmap::IndexMap::new();
            for n in &obs_names {
                observations.insert(n.clone(), ds.path.to_string_lossy().to_string());
            }
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

    // Fix 2026-04-19 (surfaced when testing camdl-book profiles): collect per-sweep-point
    // gate failures instead of exit(1). A sweep is explicitly a
    // grid of cells where edge values are expected to fail
    // convergence — treating the first failure as fatal destroys
    // the profile-likelihood use case. Collect (cell_i, pt_idx,
    // stage_name, reason) tuples; when all cells finish, print a
    // summary of passed/failed cells.
    let mut sweep_failures: Vec<(usize, usize, String, String)> = Vec::new();

    // ── gh#147 (M3.2): content-addressed store root + fit-level label ──
    // Fits write to `<output_root>/fits/{fit}-{h8}/{NN-stage}-{h8}/seed_N-{h8}/`
    // (symmetric to sims under `<output_root>/sims/`), NOT the legacy
    // per-fit `fit_dir`. The fit level is a path segment, so there is no
    // separate fit-wide record.
    let cas_root = crate::run_paths::output_root(None, config.output_dir.as_deref());
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

    // ── Execute grid: cell × sweep_point × stage ──
    for (cell_i, cell) in cells.iter().enumerate() {
        let mut cell_config = config.clone();
        if let Some(spec) = &cell.data_override {
            // Materialise the synthetic cell's data path. Keep
            // `synthetic` set so `per_fit_prefix` picks the
            // `synthetic/ds_NN/fit_<seed>/` branch; `data_spec()`
            // returns `data` when both are present, which is the
            // per-cell behaviour we want.
            cell_config.data = Some(spec.clone());
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

    // Execute stages: sweep_point × stage
    for (pt_idx, sweep_point) in sweep_points.iter().enumerate() {
        // Build a config with swept values applied to [fixed]
        let mut sweep_config = cell_config.clone();
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

        // Per-cell output directory:
        //   real-data:  <fit_dir>/real/fit_<seed>/<stage>/
        //   synthetic:  <fit_dir>/synthetic/ds_NN/fit_<seed>/<stage>/
        // Sweep slug (when present) is nested under the per-fit prefix.
        let per_fit_prefix = sweep_config.per_fit_prefix(seed, cell.dataset_idx);
        let sweep_fit_dir = if has_sweep {
            let slug: String = sweep_point.iter()
                .map(|(k, v)| format!("{}_{:.3}", k, v))
                .collect::<Vec<_>>()
                .join("__");
            if pt_idx == 0 {
                eprintln!();
            }
            eprintln!("═══ sweep point {}/{}: {} ═══", pt_idx + 1, sweep_points.len(), slug);
            fit_dir.join(&per_fit_prefix).join(slug)
        } else {
            fit_dir.join(&per_fit_prefix)
        };

    // gh#147 (M3.2): memoize each completed stage's CAS identity so a
    // downstream `StartsFrom` resolves to the upstream leaf and folds an
    // `ArtifactRef` dep into its stage hash (the deps-DAG). Scoped per
    // (cell, sweep_point) — the pipeline that ran together.
    let mut stage_identities: std::collections::HashMap<String, (runid::ContentHash, std::path::PathBuf)> =
        std::collections::HashMap::new();
    let _ = &sweep_fit_dir; // legacy layout retired; CAS root is output_root
    for (stage_name, stage) in &stages_to_run {
        eprintln!("\n── stage: {} (method={}) ──", stage_name, stage.method_name());

        // Resolve data the runners load from (also feeds the data digests).
        let fixed_resolved = sweep_config.fixed.resolve().unwrap_or_default();
        let data_spec = sweep_config.data_spec().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // Expand the [data] shorthand (`file = "..."`) into the canonical
        // per-stream map before hashing, so the shorthand and the
        // verbose `[data.observations]` form produce identical stage
        // hashes when they reference the same data.
        // The model's declared observation stream names, used to expand the
        // `[data] file = "..."` single-file shorthand. The prior code navigated
        // `model_json["observations"]` — the IR *envelope*'s top level — but the
        // streams live under `model.observations`, so this always saw zero
        // streams and the single-file form errored with "model declares no
        // observation streams". Use the typed model (already loaded above), as
        // the sibling sites at lines 414 / 547 do.
        let model_obs_names: Vec<String> =
            model.observations.iter().map(|o| o.name.clone()).collect();
        let effective_obs = data_spec.effective_observations(&model_obs_names)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
        // ── gh#147 (M3.2): resolve StartsFrom → upstream CAS leaf + dep ──
        // CLI --starts-from wins (single-stage only); else the stage's
        // init_mle. `effective_starts` is the dir the runner loads the prior
        // θ̂ from (now an upstream CAS leaf); `deps` folds the upstream's
        // identity + consumed `fit_state.toml` digest into this stage's hash.
        let cli_starts = starts_from_override.as_ref()
            .filter(|_| stages_to_run.len() == 1)
            .cloned();
        let (effective_starts, mut deps): (Option<String>, Vec<runid::inputs::ArtifactRef>) =
            if let Some(dir) = cli_starts {
                let dep = cas::cas_dep_from_dir(std::path::Path::new(&dir));
                (Some(dir), dep.into_iter().collect())
            } else {
                match stage.starts_from() {
                    StartsFrom::Random => (None, Vec::new()),
                    StartsFrom::Stage(ref dep_name) => match stage_identities.get(dep_name) {
                        Some((up_run_id, up_dir)) => {
                            let dep = cas::cas_dep_ref(*up_run_id, up_dir);
                            (Some(up_dir.to_string_lossy().to_string()), dep.into_iter().collect())
                        }
                        None => {
                            eprintln!("error: stage '{}' starts_from '{}', which has not run \
                                       in this pipeline", stage_name, dep_name);
                            std::process::exit(1);
                        }
                    },
                    StartsFrom::Directory(ref path) => {
                        let dep = cas::cas_dep_from_dir(path);
                        (Some(path.to_string_lossy().to_string()), dep.into_iter().collect())
                    }
                }
            };

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

        // gh#147: when this stage seeds chains from a survey
        // (`init = "survey_top_k"`), fold the survey's CONTENT (run_id +
        // landscape.tsv digest) into `deps` so a regenerated survey re-keys
        // the fit — the path string in `identity_payload` only distinguishes a
        // *different* directory, not the same path rewritten. Folded for every
        // stage kind via the `survey_init_path` accessor.
        if let Some(survey_dir) = stage.survey_init_path() {
            if let Some(dep) = cas::cas_survey_dep(survey_dir) {
                deps.push(dep);
            }
        }

        // gh#541: same treatment for the other two file-backed chain-start
        // sources. `--posterior`'s draws.tsv and `--params`' TOML were folded
        // by PATH only (gh#514), so rewriting either in place left the run_id
        // unchanged and the fit was served from cache — the previous file's
        // starting values, silently. `--survey-path` above and `--mle` (via
        // `cas_dep_from_dir` on the upstream leaf) already keyed on content.
        if let Some((src, artifact)) = stage.init_source_file() {
            if let Some(dep) = cas::cas_file_dep(&src, artifact) {
                deps.push(dep);
            }
        }

        // ── CAS identity + claim ──
        // sweep_config is the authoritative document inside the sweep loop;
        // reading `config` here was correct only because a sweep never adds
        // or removes stages.
        let ordinal = sweep_config.stages.get_index_of(*stage_name).map(|i| i + 1).unwrap_or(0);
        let ctx = cas::FitStageCtx {
            model: &model,
            fit_stem: &fit_stem,
            ir_version: &ir_version_str,
            engine_version: crate::version::VERSION_SHORT,
            config: &sweep_config,
            data_paths: &effective_obs,
            stage_name: *stage_name,
            stage,
            ordinal,
            seed,
            deps: deps.clone(),
        };
        let resolved = cas::resolve_fit_stage(&ctx).unwrap_or_else(|e| {
            eprintln!("error: fit-stage identity: {}", e);
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
                if let Some(ir_src) = config.compiled_ir.as_deref() {
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
                if !crate::util::model_is_camdl_source(&config.model.camdl) {
                    eprintln!(
                        "note: model given as compiled IR; skipping model.render.json / \
                         model.graph.json (pass the .camdl source to archive the display render)"
                    );
                } else {
                match crate::util::render_model_json(std::path::Path::new(&config.model.camdl)) {
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
                match crate::util::render_model_graph_json(std::path::Path::new(&config.model.camdl)) {
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
        if !force && a.resume.is_none() {
            if let runid::Lookup::Hit(_) =
                store.lookup(&cas_path, &runid::LeafIdentity::new(resolved.run_id))
            {
                eprintln!("  \x1b[33mcache hit — reusing {}\x1b[0m",
                    cas_path.strip_prefix(&cas_root).unwrap_or(&cas_path).display());
                stage_identities.insert(stage_name.to_string(), (resolved.run_id, cas_path));
                continue;
            }
        }
        // Streaming write through the one resolved-writer seam (gh#241 PR D).
        // The running record carries Null inputs (the stage's loglik summary is
        // a post-run result); the final inputs are supplied to `finalize`. The
        // upstream lineage deps ride in `RecordMeta`.
        let resolved_artifact = crate::resolve::ResolvedArtifact {
            kind: runid::ArtifactKind::FitStage,
            levels: resolved.levels.clone(),
            run_id: resolved.run_id,
            display_inputs: serde_json::Value::Null,
        };
        let meta = crate::resolve::RecordMeta::new(
            &ir_version_str, &sweep_config.model.camdl, None)
            .with_deps(deps.clone());
        let mut write = match crate::resolve::begin_resolved_write(
            &store, &cas_root, &resolved_artifact, &meta,
            crate::resolve::WriteMode::Streaming,
        ) {
            Ok(crate::resolve::ResolvedWrite::Streaming(c)) => c,
            Ok(crate::resolve::ResolvedWrite::Committed(_)) => {
                unreachable!("Streaming write mode never returns a committed path")
            }
            Err(e) => {
                eprintln!("error: claim fit stage {}: {}", cas_path.display(), e);
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
        // Uninitialized on purpose: every dispatch arm must assign a stage
        // loglik (or exit) — the compiler enforces that no arm can fall
        // through to the finalize below with a silent None.
        let stage_best_loglik: Option<f64>;
        // PFilter has replicates, not competing chains, so it legitimately
        // leaves this None.
        let mut stage_best_chain: Option<usize> = None;

        // Surface the registry caveat for Beta/Experimental methods, once per
        // executing stage (after the cache-hit skip above, so reused stages
        // stay silent). Registry-driven so it can't drift from `fit methods`.
        methods::emit_status_banner(stage.method_kind(), stage.backend());

        match stage {
            Stage::IF2 { backend, chains, particles, iterations, cooling, cooling_target_iters, init_method, survey_path, survey_top_k_n, loglik_eval, gate, dt_check, .. } => {
                // clean_eval comes straight from the stage TOML — it is part of
                // the fit's identity (folded into the IF2 stage's whole-serialize
                // identity_payload), so it has no CLI override (gh#189: a CLI
                // override bypassed the run_id and silently re-scored under the
                // same key). The gate's `--decibans-thresh` override is applied
                // through `apply_cli_overrides` BEFORE the identity is taken,
                // so `gate` here already carries it (gh#540 seam).
                let effective_loglik_eval = loglik_eval.clone();
                let effective_gate = gate.clone();
                let prior_state = effective_starts.as_ref().and_then(|dir| {
                    state::FitState::load(dir).ok()
                });

                // Gate 1 — pre-stage: if this stage consumes a prior
                // stage (starts_from), refuse to run when the prior
                // stage's tail Â failed convergence on any
                // non-IVP param. Skipped when starts_from is absent
                // (this stage is itself the scout). Overridable via
                // --allow-nonconverged-scout. See proposal
                // docs/dev/proposals/2026-04-19-refine-gates-scout-convergence.md.
                let (scout_best_for_gate2, scout_chain_logliks_for_gate2):
                    (Option<f64>, Vec<f64>) = match prior_state.as_ref() {
                    Some(ps) => {
                        use gating::ScoutGateVerdict;
                        // Compound gate (Â + decibans-spread). Reads
                        // the GateConfig from the *consuming* stage —
                        // i.e. refine's [stages.refine.gate] governs
                        // how strictly we judge the scout it consumes.
                        // CLI overrides already merged into
                        // `effective_gate` above (Step 4).
                        match gating::check_scout_convergence(ps, &effective_gate) {
                            ScoutGateVerdict::Ok => {}
                            ScoutGateVerdict::SoftWarn { param_agreement } => {
                                eprintln!("\x1b[33m  warning:\x1b[0m prior stage tail Â in \
                                           SoftWarn band ([{:.2}, {:.2})) for: {}",
                                    gating::A_SOFT, effective_gate.a_thresh,
                                    param_agreement.iter()
                                        .map(|(n, r)| format!("{} (Â={:.2})", n, r))
                                        .collect::<Vec<_>>().join(", "));
                            }
                            ScoutGateVerdict::Hard { failing, all_structural, ivp, loglik_spread } => {
                                let msg = gating::format_hard_verdict(
                                    &effective_gate,
                                    &failing, &all_structural, &ivp,
                                    loglik_spread, ps.best_loglik, None);
                                if allow_nonconverged_scout {
                                    eprintln!("\x1b[33m  warning:\x1b[0m {}", msg);
                                    eprintln!("\n  --allow-nonconverged-scout: proceeding anyway.");
                                } else if has_sweep {
                                    // Sweep-gate fix 2026-04-19 (testing camdl-book): don't
                                    // kill the whole sweep on one cell's gate
                                    // failure. Record, skip remaining stages for
                                    // this sweep point, continue to next point.
                                    eprintln!("\x1b[33m  sweep-skip:\x1b[0m {}", msg);
                                    sweep_failures.push((
                                        cell_i, pt_idx,
                                        stage_name.to_string(),
                                        "scout_tail_agreement_gate".to_string(),
                                    ));
                                    break; // exit stages loop for this sweep point
                                } else {
                                    eprintln!("error: {}", msg);
                                    std::process::exit(1);
                                }
                            }
                            ScoutGateVerdict::DecibansSpread {
                                delta_db, threshold_db, sigma_max, chain_logliks,
                            } => {
                                let msg = gating::format_decibans_spread_verdict(
                                    delta_db, threshold_db, sigma_max, &chain_logliks);
                                if allow_nonconverged_scout {
                                    eprintln!("\x1b[33m  warning:\x1b[0m {}", msg);
                                    eprintln!("\n  --allow-nonconverged-scout: proceeding anyway.");
                                } else if has_sweep {
                                    eprintln!("\x1b[33m  sweep-skip:\x1b[0m {}", msg);
                                    sweep_failures.push((
                                        cell_i, pt_idx,
                                        stage_name.to_string(),
                                        "scout_decibans_spread_gate".to_string(),
                                    ));
                                    break;
                                } else {
                                    eprintln!("error: {}", msg);
                                    std::process::exit(1);
                                }
                            }
                        }
                        (Some(ps.best_loglik), ps.chain_logliks.clone())
                    }
                    None => (None, Vec::new()),
                };

                let mut run_config = runner::FitRunConfig::build(
                    &sweep_config,
                    prior_state.as_ref(),
                    *chains, *particles, *iterations,
                    *cooling, *cooling_target_iters,
                    // gh#506: NOT `effective_starts.is_none()`. That asked
                    // `build` to overwrite every `EstimatedParam::initial`
                    // with a uniform draw whenever the stage had no upstream
                    // `starts_from` — i.e. on every scout stage — which threw
                    // away `[estimate].start` and made `init = "single"` mean
                    // "single random point". Per-chain dispersion is `init`'s
                    // job (`uniform` / `lhs` / `uniform_unconstrained`), and
                    // this predated that machinery.
                    seed, false,
                ).unwrap_or_else(|e| {
                    eprintln!("error building run config: {}", e);
                    std::process::exit(1);
                });
                run_config.loglik_eval = effective_loglik_eval.clone();
                run_config.gate = effective_gate.clone();

                std::fs::create_dir_all(&stage_dir).unwrap_or_else(|e| {
                    eprintln!("error creating {}: {}", stage_dir.display(), e);
                    std::process::exit(1);
                });

                let collector = sim::inference::diagnostic::DiagnosticCollector::new(stage_name);
                let t0 = std::time::Instant::now();
                // Per-chain starting points. When this stage consumes
                // a prior stage (`starts_from`), every chain starts from
                // that stage's MLE (intent of the handoff) regardless
                // of init_method — that's what makes refine-after-scout
                // meaningful. Otherwise dispatch on `init_method`
                // (gh#42): Single = all chains at the seeded start
                // (legacy refine semantics, useful when bounds are
                // tight); Uniform = per-chain uniform random within
                // bounds (v1 default — keeps existing fit.toml files
                // unchanged); Lhs = Latin-hypercube stratified, scale-
                // aware via Transform.
                //
                // CLI `--init` / `--survey-path` / `--survey-top-k` were
                // already written INTO the stage by `apply_cli_overrides`
                // (gh#514, applied before the identity was taken; the flags
                // require --stage, and --stage filters `stages_to_run` to
                // exactly the overridden stage). Re-merging the CLI values
                // here was a second merge site that could only agree with
                // the first or drift from it — read the stage fields.
                let effective_init: crate::fit::init::InitMethod = init_method.clone();
                let effective_survey_path: Option<std::path::PathBuf> = survey_path.clone();
                let effective_survey_top_k_n: Option<usize> = *survey_top_k_n;
                // gh#506 follow-up: a declared `start` that the chosen init
                // mode discards is a silent no-op. Not an error — the
                // spreading modes ignore it on purpose — but the user who
                // wrote the value should hear that it had no effect, rather
                // than inferring a start that never happened.
                if effective_starts.is_none()
                    && init::ignores_base_point(&effective_init, *chains)
                {
                    let declared: Vec<&str> = sweep_config.estimate.iter()
                        .filter(|(_, spec)| spec.start.is_some())
                        .map(|(n, _)| n.as_str())
                        .collect();
                    if !declared.is_empty() {
                        eprintln!(
                            "  \x1b[33mnote:\x1b[0m `init = \"{}\"` draws every chain's \
                             start, so `[estimate].start` is unused here for: {}. \
                             Use `init = \"single\"` to start every chain at the \
                             declared values, or drop the `start` entries.",
                            effective_init, declared.join(", "));
                    }
                }
                // For survey_top_k we need to keep the SurveyTopKResult
                // around (not just the per-chain `chains`) so the
                // chain_init_source / chain_starts.tsv writers can pull
                // the survey's full hash out of it. Plain Lhs / Uniform
                // / Single produce no such result.
                let (per_chain_params, survey_top_k_result) =
                    if effective_starts.is_some() {
                        (None, None)
                    } else {
                        let model_identity_str =
                            crate::resolve::model_identity_from_ir(&run_config.model_ir_json);
                        let data_hashes = init::compute_data_hashes(&effective_obs)
                            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                        let estimate_names: Vec<String> =
                            sweep_config.estimate.keys().cloned().collect();
                        let fixed_hashmap: std::collections::HashMap<String, f64> =
                            fixed_resolved.iter().map(|(k, v)| (k.clone(), *v)).collect();
                        let ctx = init::SurveyFitContext {
                            model_identity: &model_identity_str,
                            data_hashes: &data_hashes,
                            fixed: &fixed_hashmap,
                            estimate_names: &estimate_names,
                        };
                        // Build a `ResolvedParameters` view so the
                        // step-7 warm-start variants can dispatch
                        // through `chain_starts::draw_chain_starts`.
                        // For legacy modes the view is unused; for
                        // FromPrior/FromPosterior/FromMle/FromParams
                        // it's the seam that lets us read the model
                        // priors / bounds / estimate set.
                        let resolved_view = init::build_resolved_view_for_init(
                            &run_config.model,
                            &run_config.base_params,
                            &run_config.estimated_params,
                        );
                        init::resolve_per_chain_starts_from_method(
                            &effective_init,
                            effective_survey_path.as_deref(),
                            effective_survey_top_k_n,
                            stage_name,
                            &run_config.estimated_params,
                            *chains,
                            seed,
                            &ctx,
                            Some(&resolved_view),
                        ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
                    };

                // Write chain_starts.tsv sidecar for audit (gh#51).
                // Best-effort; failure logs but doesn't abort the fit.
                if let Err(e) = init::write_chain_starts_tsv(
                    &stage_dir,
                    &run_config.estimated_params,
                    per_chain_params.as_deref(),
                    *chains,
                    &effective_init,
                    survey_top_k_result.as_ref(),
                ) {
                    eprintln!("warning: could not write chain_starts.tsv: {}", e);
                }
                let chain_init_source = init::format_chain_init_source(
                    &effective_init, survey_top_k_result.as_ref(),
                );
                let stage_dir_str = stage_dir.to_string_lossy();
                let chain_results = runner::run_chains_with_per_chain_params(
                    &run_config, per_chain_params.as_deref(), &collector,
                    Some(stage_dir_str.as_ref()))
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                let elapsed = t0.elapsed();

                // Gate 2 — post-stage: refine must not regress below
                // scout's best. Not overridable — a regression is a
                // pipeline failure regardless of user preference.
                // Fires only when a prior stage was consumed (scout→
                // refine handoff). Fails before writing any
                // "stage completed" artefacts so the filesystem tells
                // the truth.
                if let Some(scout_best) = scout_best_for_gate2 {
                    if let Err(msg) = gating::check_loglik_regression(
                        scout_best, chain_results.best_loglik,
                        &scout_chain_logliks_for_gate2,
                    ) {
                        if has_sweep {
                            // Sweep-gate fix 2026-04-19 (testing camdl-book): same
                            // non-halting treatment as the scout gate.
                            eprintln!("\x1b[33m  sweep-skip:\x1b[0m {}", msg);
                            sweep_failures.push((
                                cell_i, pt_idx,
                                stage_name.to_string(),
                                "regression_gate".to_string(),
                            ));
                            break;
                        }
                        eprintln!("error: {}", msg);
                        std::process::exit(1);
                    }
                }

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
                // Pre-filter starts — records whatever per-chain
                // initial points IF2 actually received. With the
                // per-chain random-start builder above, this file now
                // shows genuine independence across chains when
                // `starts_from` is None.
                runner::write_chain_starts(
                    &stage_dir.to_string_lossy(),
                    per_chain_params.as_deref(),
                    &run_config.estimated_params, *chains,
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_diagnostics(&stage_dir.to_string_lossy(), &chain_results.results)
                    .unwrap_or_else(|e| eprintln!("warning: {}", e));

                // Write fit_state.toml for downstream stages.
                // Source params from the clean-eval winner θ̂ (GH #16) so
                // mle_params.toml and final_params.toml agree, and so
                // refine starts in the basin clean-eval actually picked.
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
                // identity-defining). --dt-check-strict stays a plain
                // runtime arg: it only escalates the warning to a fatal
                // exit and never changes the stored leaf.
                let effective_dt_check = dt_check.clone();
                let dt_check_seed = seed.wrapping_add(0xd7c4ec_5eed); // "dtchec seed"
                let dt_check_result = dt_check::run_richardson_ladder(
                    &run_config,
                    winner_theta,
                    &effective_dt_check,
                    *backend,
                    a.dt_check_strict,
                    &dt_check::DtCheckInherits {
                        n_particles:  effective_loglik_eval.n_particles,
                        n_replicates: effective_loglik_eval.n_replicates,
                        combine:      effective_loglik_eval.combine,
                    },
                    dt_check_seed,
                )
                .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                dt_check::print_terminal_report(&dt_check_result);
                let fit_state = state::FitState {
                    stage: stage_name.to_string(),
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
                    ivp_params: run_config.estimated_params.iter()
                        .filter(|p| p.ivp).map(|p| p.name.clone()).collect(),
                    chain_logliks: chain_results.results.iter()
                        .map(|(_, r)| r.final_loglik).collect(),
                    chain_eval_logliks: chain_results.chain_eval_logliks(),
                    chain_eval_ses: chain_results.chain_eval_ses(),
                    // Persist the gate / clean-eval config that was
                    // *actually in force* — `effective_gate` and
                    // `effective_loglik_eval` above already collapsed the
                    // priority chain (CLI flag > stage TOML > defaults).
                    // `summary` reads these so its verdict line reports
                    // against the threshold the run was judged by, not
                    // whatever `fit.toml` says at summary-time.
                    // See proposal §Phase 3.
                    resolved_gate: Some(effective_gate.clone()),
                    resolved_loglik_eval: Some(effective_loglik_eval.clone()),
                    chain_init_source: Some(chain_init_source.clone()),
                    dt_check: if matches!(dt_check_result.verdict,
                        dt_check::DtCheckVerdict::Skipped)
                    {
                        None  // skipped → omit the block, mirroring legacy semantics
                    } else {
                        Some(dt_check_result.clone())
                    },
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
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
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
                    stage: stage_name.to_string(),
                    best_chain: chain_results.best_chain,
                    // Record the backend the STAGE actually fit on (gh#241): the
                    // `simulate --params` guardrail replays θ̂ with this, so it
                    // must be the stage's backend. `InferenceBackend` is a valid
                    // `ForwardBackend` (total `From`).
                    backend: stage.backend().into(),
                    dt: sweep_config.config.dt,
                    loglik: chain_results.best_loglik,
                    loglik_sd: 0.0,
                    n_particles: *particles,
                    ess_at_mle: None,
                    timestamp: fit_state.timestamp.clone(),
                };
                provenance::write_mle_params(&mle_path, &all_params, &metadata)
                    .unwrap_or_else(|e| eprintln!("warning: {}", e));

                collector.render_to_stderr();

                stage_best_loglik = Some(chain_results.best_loglik);
                stage_best_chain = Some(chain_results.best_chain);

                eprintln!();
                crate::status::done("stored", format!("{} \u{b7} {}/", stage_name, stage_dir.display()));
                crate::status::hint(format!("best ll={:.1} (chain {}) in {:.1}s",
                    chain_results.best_loglik, chain_results.best_chain + 1, elapsed.as_secs_f64()));
            }
            Stage::PGAS { .. } => {
                let pgas_opts = pgas::PgasStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                // gh#540: no CLI overrides here. Every flag that reaches this
                // stage was written into `stage` before its content address was
                // taken, so `from_stage` above already carries them. Applying
                // one here again would be the bug this fixed: a value the run
                // uses that its `run_id` never saw.

                pgas::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    pgas_opts,
                    seed, force,
                    a.resume.is_some(),
                    effective_starts.as_deref(),
                ).unwrap_or_else(|e| {
                    eprintln!("error running pgas stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                // Bubble loglik from fit_state.toml written by PGAS runner
                let fs = load_stage_result_or_exit(stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Stage::PMMH { .. } => {
                let pmmh_opts = pmmh::PmmhStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                // gh#540: overrides are applied to `stage` before its
                // content address is taken; `from_stage` carries them.

                pmmh::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    pmmh_opts,
                    seed, force,
                    /* check_variance */ false,
                    a.resume.is_some(),
                    effective_starts.as_deref(),
                    // PMMH's dt-check is the PF-based one wired on the IF2 path.
                    /* dt_check_opt */ None,
                ).unwrap_or_else(|e| {
                    eprintln!("error running pmmh stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                let fs = load_stage_result_or_exit(stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Stage::Mh { .. } => {
                // Deterministic-ODE Metropolis-Hastings. Routes through the
                // shared PMMH machinery (chains, adaptive proposal, priors,
                // MAP, R̂/ESS, trace output); `pmmh::run_stage` swaps the PF
                // likelihood for `compute_ode_loglik` when the stage is the Mh
                // variant. `PmmhStageOpts::from_stage` parses the Mh fields
                // with `n_particles = 0` / `rho = None` (the deterministic path
                // uses neither).
                let pmmh_opts = pmmh::PmmhStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                // gh#540: overrides are applied to `stage` before its
                // content address is taken; `from_stage` carries them.

                // Deterministic ODE dt-check at the MAP (gh#52, gh#227). On by
                // default; honours the same CLI flags as the IF2 path
                // (--no-dt-check / --dt-check-halvings / --dt-check-strict).
                // gh#726: Mh has no dt_check TOML field, so the CLI flags
                // cannot be keyed into the identity — `apply_cli_overrides`
                // refuses them on this stage. Defaults only until the field
                // lands. --dt-check-strict remains allowed (abort policy,
                // leaf-byte-neutral).
                let mh_dt_check = config_v2::DtCheckConfig::default();

                pmmh::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    pmmh_opts,
                    seed, force,
                    /* check_variance */ false,
                    a.resume.is_some(),
                    effective_starts.as_deref(),
                    Some((mh_dt_check, a.dt_check_strict)),
                ).unwrap_or_else(|e| {
                    eprintln!("error running mh stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                let fs = load_stage_result_or_exit(stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Stage::Nuts { .. } => {
                // Gradient-based Bayesian sampling of the deterministic ODE
                // marginal likelihood (gh#275 Phase 2) via `det_grad` + NUTS.
                let nuts_opts = nuts::NutsStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                // gh#540: overrides are applied to `stage` before its
                // content address is taken; `from_stage` carries them.

                nuts::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    nuts_opts,
                    seed, force,
                    a.resume.is_some(),
                ).unwrap_or_else(|e| {
                    eprintln!("error running nuts stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                let fs = load_stage_result_or_exit(stage_name, &stage_dir);
                stage_best_loglik = Some(fs.best_loglik);
                stage_best_chain = Some(fs.best_chain);
            }
            Stage::NlSbplx(_) | Stage::NlBobyqa(_) => {
                #[cfg(feature = "ode")]
                {
                    // Model identity + data digests for the mle_params.toml
                    // provenance block (same shape as the IF2 path uses below).
                    // Load the canonical IR JSON from the pre-compiled IR so this
                    // provenance read doesn't re-invoke camdlc per PFilter stage.
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
                    // default. gh#726: nl-* stages have no dt_check TOML
                    // field, so --no-dt-check / --dt-check-halvings cannot be
                    // keyed into the identity — `apply_cli_overrides` refuses
                    // them here. Defaults only until the field lands;
                    // --dt-check-strict remains allowed (abort policy,
                    // leaf-byte-neutral).
                    let nl_dt_check = config_v2::DtCheckConfig::default();

                    nlopt_stage::run_stage(
                        &sweep_config,
                        stage_name,
                        stage,
                        &stage_dir,
                        seed,
                        effective_starts.as_deref(),
                        &parent_fit_hash,
                        &model_identity_for_prov,
                        &data_hashes_for_prov,
                        &nl_dt_check,
                        a.dt_check_strict,
                    ).unwrap_or_else(|e| {
                        eprintln!("error running nlopt stage '{}': {}", stage_name, e);
                        std::process::exit(1);
                    });
                    let fs = load_stage_result_or_exit(stage_name, &stage_dir);
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
                #[cfg(not(feature = "ode"))]
                {
                    let _ = (stage_name, &sweep_config, &stage_dir, seed, effective_starts.as_deref());
                    eprintln!(
                        "error: this binary was built without --features ode, \
                         which is required for algorithm = \"{}\". Rebuild \
                         with `cargo build --features ode` (default).",
                        stage.method_name()
                    );
                    std::process::exit(1);
                }
            }
            Stage::PFilter { particles, replicates, record_ancestry, record_prequential, .. } => {
                let n_reps = replicates.unwrap_or(1);
                // record_ancestry: CLI flag is a one-way override to true
                // (TOML default false); no flag means use TOML.
                // record_prequential: TOML default true (per the
                // 2026-04-20 prequential proposal); explicit
                // `record_prequential = false` in [stages.X] opts out,
                // and the CLI flag can re-enable it on a per-invocation
                // basis without editing the TOML.
                let record_ancestry = *record_ancestry;
                let want_prequential = *record_prequential;
                let prior_state = effective_starts.as_ref().and_then(|dir| {
                    state::FitState::load(dir).ok()
                });
                if prior_state.is_none() && !effective_starts.as_ref().is_none_or(|s| s.is_empty()) {
                    eprintln!("warning: could not load fit_state from starts_from");
                }

                // Build run config (reuse IF2 builder with 1 chain, N particles).
                // cooling_target_iters=1 here is harmless: PFilter doesn't
                // cool, so the IF2-shaped config field is never read.
                let run_config = runner::FitRunConfig::build(
                    &sweep_config,
                    prior_state.as_ref(),
                    1, *particles, 1, 1.0, 1, seed, false,
                ).unwrap_or_else(|e| {
                    eprintln!("error building pfilter config: {}", e);
                    std::process::exit(1);
                });

                std::fs::create_dir_all(&stage_dir).unwrap_or_else(|e| {
                    eprintln!("error creating {}: {}", stage_dir.display(), e);
                    std::process::exit(1);
                });

                // Run PF at MLE params
                let mle_params = run_config.base_params.clone();
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
                    // user-facing flags from Stage::PFilter. Prequential
                    // is per-stage scoring (point-estimate property),
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
                        eprintln!("pfilter error: {:?}", e);
                        std::process::exit(1);
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
                            preq_trace = Some(sim::inference::prequential::build_trace(
                                recorded, &y_obs, &per_stream_obs, &result.ess_trace, 0));
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
                if let Some(ref trace) = preq_trace {
                    use std::io::Write;
                    let tsv_path = format!("{}/prequential.tsv", stage_dir.display());
                    let mut f = std::fs::File::create(&tsv_path).unwrap();
                    writeln!(f, "t\ty_obs\tlog_score\tcrps\tpit\tess").unwrap();
                    for s in &trace.steps {
                        writeln!(f, "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.2}",
                            s.t, s.y_obs, s.log_score, s.crps, s.pit, s.ess).unwrap();
                    }
                    let json_path = format!("{}/prequential.json", stage_dir.display());
                    let json = serde_json::to_string_pretty(trace).unwrap();
                    std::fs::write(&json_path, json).unwrap();
                    eprintln!("  prequential: elpd={:.2}, mean_crps={:.3}, PIT 90% cov={:.2}",
                        trace.elpd(), trace.mean_crps(), trace.pit_coverage(0.90));
                }
                stage_best_loglik = Some(mean_ll);
            }
        }

        // ── finalize the CAS fit-stage leaf ──
        // The runners streamed every output (chains, fit_state.toml,
        // draws.tsv, trajectories/, …) into `stage_dir = claim.dir()`;
        // `finalize` builds the recursive exact-set manifest over them and
        // commits Running→Completed. The display fields ride in `run.json`
        // `inputs` (recorded, never hashed) for show/status.
        let stage_elapsed = stage_t0.elapsed();
        let algo_tag = stage.method_name();
        let backend_tag = stage.backend().as_str();
        let algo_json = match stage {
            Stage::IF2 { chains, particles, iterations, cooling, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations, "cooling": cooling }),
            Stage::PGAS { chains, particles, sweeps, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "sweeps": sweeps }),
            Stage::PMMH { chains, particles, iterations, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations }),
            Stage::Mh { chains, iterations, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "iterations": iterations }),
            Stage::Nuts { chains, warmup, samples, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "warmup": warmup, "samples": samples }),
            Stage::PFilter { particles, replicates, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "particles": particles, "replicates": replicates }),
            Stage::NlSbplx(c) | Stage::NlBobyqa(c) =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": c.chains, "tolerance": c.tolerance, "max_evals": c.max_evals }),
        };
        let inputs_json = serde_json::json!({
            "stage": stage_name,
            "method": algo_tag,
            "backend": backend_tag,
            "seed": seed,
            "n_chains": stage.chains(),
            "best_loglik": stage_best_loglik,
            "best_chain": stage_best_chain,
            "algorithm": algo_json,
            "starts_from": effective_starts,
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
                eprintln!("error: could not finalize fit stage {}: {}", cas_path.display(), e);
                std::process::exit(1);
            }
        };
        crate::status::done("stored",
            format!("{} \u{b7} {} \u{b7} {:.1}s", stage_name, dest.display(), stage_elapsed.as_secs_f64()));
        stage_identities.insert(stage_name.to_string(), (resolved.run_id, dest));

    } // end stages
    } // end sweep_points
    } // end cells

    // Sweep-gate fix 2026-04-19 (testing camdl-book): emit a sweep summary when
    // any cells were skipped due to gate failures. Also write a
    // machine-readable record to <fit_dir>/sweep_failures.tsv so
    // downstream tooling (profile-likelihood plots, etc.) can
    // distinguish "cell didn't converge" from "cell wasn't run."
    if has_sweep && !sweep_failures.is_empty() {
        let total_runs = cells.len() * sweep_points.len();
        let n_failed = sweep_failures.len();
        eprintln!("\n━━━ sweep summary ━━━");
        eprintln!("  {} / {} cells skipped gate", n_failed, total_runs);
        for (cell_i, pt_idx, stage, reason) in &sweep_failures {
            let slug: String = sweep_points[*pt_idx].iter()
                .map(|(k, v)| format!("{}={:.3}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("    cell {:>2} / pt {:>2} ({}) stage={} reason={}",
                cell_i + 1, pt_idx + 1, slug, stage, reason);
        }
        // Land beside the leaves: the fit segment dir `fit run` announced
        // (the FitDigest `fits/{stem}-{h8}/`), not the divergent legacy
        // `fit_dir`. Create it first — a sweep keys each cell's own segment,
        // so the base segment may not yet exist on disk.
        let _ = std::fs::create_dir_all(&announced_fit_dir);
        let path = announced_fit_dir.join("sweep_failures.tsv");
        let mut tsv = String::from("cell\tsweep_point\tsweep_values\tstage\treason\n");
        for (cell_i, pt_idx, stage, reason) in &sweep_failures {
            let slug: String = sweep_points[*pt_idx].iter()
                .map(|(k, v)| format!("{}={:.6}", k, v))
                .collect::<Vec<_>>()
                .join(";");
            tsv.push_str(&format!("{}\t{}\t{}\t{}\t{}\n",
                cell_i, pt_idx, slug, stage, reason));
        }
        if let Err(e) = std::fs::write(&path, tsv) {
            eprintln!("warning: could not write {}: {}", path.display(), e);
        } else {
            eprintln!("  details: {}", path.display());
        }
    }

    // ── Grid roll-ups (summary.tsv / coverage.tsv): deferred to M4 ──
    //
    // Each grid cell is now a content-addressed fit — its own FitDigest base,
    // keyed by that cell's dataset digest × fit-seed — readable individually
    // via `camdl list`/`show`/`cat`. The cross-cell summary and the synthetic
    // parameter-recovery coverage table are derived views with no home in the
    // per-cell tree; the M4 reindex rebuilds them from the CAS leaves (gh#150 /
    // gh#154), where coverage gains a truth-within-interval correctness check.
    if cells.len() > 1 || config.synthetic.is_some() {
        eprintln!("note: grid summary / coverage are derived views — \
                   rebuilt by the reindex in M4 (gh#150 / gh#154)");
    }

    // gh#147 (M3.2): no fit-wide `run.json` rewrite — the fit identity is a
    // CAS path segment, and each stage leaf records its own wall time in
    // `run.json` `inputs` at `finalize`. `fit_start` paces the run; per-stage
    // timing is the honest unit now.
    let _ = fit_start;
}

/// Resolve a `--resume <base ref>` to a base stage-leaf dir: an existing path,
/// else a `run_id` hex prefix matched under `<cas_root>/fits/`. `None` when no
/// unique match (caller errors).
/// After a stage runner reported success, its `fit_state.toml` is the channel
/// carrying the stage result back to this orchestrator (`best_loglik` /
/// `best_chain` end up in the finalized run.json inputs). A missing or
/// corrupt file at that point is a runner bug or a torn write; the previous
/// `if let Ok` swallow left a silent `null` in run.json where the result
/// belonged. Fail loudly instead.
fn load_stage_result_or_exit(stage_name: &str, stage_dir: &std::path::Path) -> state::FitState {
    state::FitState::load(&stage_dir.to_string_lossy()).unwrap_or_else(|e| {
        eprintln!(
            "error: stage '{}' reported success but its fit_state.toml \
             cannot be read back from {}: {}",
            stage_name, stage_dir.display(), e);
        std::process::exit(1);
    })
}

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
    config: &config_v2::FitConfigV2,
    fit_path: &str,
    label: Option<String>,
    model: Option<&ir::Model>,
) -> crate::run_meta::FitSidecar {
    // Prefer the pre-compiled IR (set by `cmd_fit_run_v2`); `load_model`
    // re-invokes camdlc when handed a raw `.camdl`. The resolved model identity
    // is byte-identical either way; empty string if the IR can't be loaded.
    let model_src = config.compiled_ir.as_deref().unwrap_or(&config.model.camdl);
    let model_identity = crate::util::load_model(model_src)
        .ok()
        .map(|(_, ir_json)| crate::resolve::model_identity_from_ir(&ir_json))
        .unwrap_or_default();
    let fit_toml_bytes = std::fs::read(fit_path).unwrap_or_default();
    let fit_toml_hash = crate::hashing::sha256_hex(&fit_toml_bytes);
    let data_hashes: std::collections::HashMap<String, String> = config
        .data.as_ref()
        .map(|d| d.observations.iter()
            .filter_map(|(name, path)| {
                crate::hashing::file_hash(path).map(|h| (name.clone(), h))
            })
            .collect())
        .unwrap_or_default();
    let estimated: Vec<String> = config.estimate.keys().cloned().collect();
    let fixed: std::collections::HashMap<String, f64> = config.fixed
        .resolve().unwrap_or_default().into_iter().collect();
    let stages_declared: Vec<String> = config.stages.keys().cloned().collect();
    // gh#75: resolve per-parameter prior provenance. We only emit
    // entries when the fit has at least one Bayesian stage (IF2-only
    // fits don't consume priors, and surfacing `model_ir` for a
    // bunch of params the fit will never read from is misleading
    // noise — both `camdl fit where` and an IF2-only run can leave
    // this empty). `validate_priors_present` has already rejected
    // any silent flat-fallback by the time we get here, so any
    // entry we emit is one of {fit_toml, model_ir, flat_explicit}.
    let any_bayesian = config.stages.values().any(config_v2::Stage::requires_priors);
    let resolved_priors: Vec<crate::run_meta::ResolvedPriorEntry> = match model {
        Some(model) if any_bayesian => {
            let names: Vec<String> = config.estimate.keys().cloned().collect();
            crate::fit::priors_precedence::resolve_priors_with_precedence(
                &names, &config.estimate, model,
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
    let _ = stages_declared; // stages come from the leaves; sidecar omits them.
    crate::run_meta::FitSidecar {
        label,
        model_path: config.model.camdl.clone(),
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
    use config_v2::FitConfigV2;

    let a_path = args.a.to_string_lossy().into_owned();
    let b_path = args.b.to_string_lossy().into_owned();
    let a = FitConfigV2::load(&a_path).unwrap_or_else(|e| {
        eprintln!("error loading {}: {}", a_path, e);
        std::process::exit(1);
    });
    let b = FitConfigV2::load(&b_path).unwrap_or_else(|e| {
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

    // Stage changes
    println!();
    println!("Stages:");
    let a_stages: std::collections::BTreeSet<&str> = a.stages.keys().map(|s| s.as_str()).collect();
    let b_stages: std::collections::BTreeSet<&str> = b.stages.keys().map(|s| s.as_str()).collect();
    let mut stage_changes = false;
    for name in b_stages.difference(&a_stages) {
        let s = &b.stages[*name];
        println!("  stage '{}': (new) {}", name, s.method_name());
        stage_changes = true;
    }
    for name in a_stages.difference(&b_stages) {
        println!("  stage '{}': (removed)", name);
        stage_changes = true;
    }
    for name in a_stages.intersection(&b_stages) {
        let sa = &a.stages[*name];
        let sb = &b.stages[*name];
        let sa_json = serde_json::to_string(sa).unwrap_or_default();
        let sb_json = serde_json::to_string(sb).unwrap_or_default();
        if sa_json != sb_json {
            // Show detailed changes
            let mut details = Vec::new();
            if sa.method_name() != sb.method_name() {
                details.push(format!("method {}→{}", sa.method_name(), sb.method_name()));
            }
            if sa.chains() != sb.chains() {
                details.push(format!("chains {}→{}", sa.chains(), sb.chains()));
            }
            // Compare serialized for catch-all
            if details.is_empty() {
                details.push("settings changed".to_string());
            }
            println!("  stage '{}': {}", name, details.join(", "));
            stage_changes = true;
        }
    }
    if !stage_changes {
        println!("  (no stage changes)");
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

    // Best-effort: point the user at the source fit's content-addressed
    // segment so they can set `starts_from` on the derived fit's first stage.
    // The exact stage-leaf path (`{NN-stage}-{h8}/seed_N-{h8}`) needs the
    // stage + seed hashes, so we name the segment and defer the leaf to
    // `camdl list`.
    if let Some(cfg) = config_v2::FitConfigV2::load(&from).ok() {
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
            eprintln!("  hint: set starts_from on your first stage to the last stage \
                       leaf under {}", seg.display());
            eprintln!("        (run `camdl list` to find the exact stage-leaf path)");
        }
    }

    std::fs::write(&to, &content).unwrap_or_else(|e| {
        eprintln!("error writing {}: {}", to, e);
        std::process::exit(1);
    });

    eprintln!("created {}", to);
}

/// Accept either a directory path or a git-style short hash for
/// `--starts-from`. The heuristic: contains `/` or `\\` → path;
/// else → resolve as a leaf `run_id` prefix via
/// `browse::resolve_stage_by_hash` against the default output
/// root. Errors on zero or multiple matches.
///
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

/// Resolve a `--init from_mle --mle <hash-or-path>` argument to a
/// concrete stage directory path. If the argument looks like a path,
/// pass it through; otherwise treat as a short content-hash prefix and
/// resolve against the default output root. Hardening proposal
/// ship-now #9 (originally for the removed `--starts-from` flag,
/// now reachable via `--init from_mle --mle <hash>`).
fn resolve_starts_from_arg(raw: &str) -> String {
    if raw.contains('/') || raw.contains('\\') || raw == "." || raw == ".." {
        return raw.to_string();
    }
    // Treat as a short hash prefix. Resolve against the default
    // output root; if the user has a non-default output location
    // they can still pass the full path.
    let root = format!("./{}", crate::run_paths::DEFAULT_OUTPUT_ROOT);
    match crate::browse::resolve_stage_by_hash(&root, raw) {
        Ok(path) => path.to_string_lossy().to_string(),
        Err(e) => {
            eprintln!("error: --init from_mle --mle '{}': {}", raw, e);
            eprintln!("  Tip: pass a full path (e.g. results/fits/FOO/real/fit_1/scout)");
            eprintln!("  or a longer hash prefix.");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gh#191: per-stage capability gate on `fit run` ─────────────

    /// A chain_binomial IF2 stage — the inference path that carries no
    /// real reservoir state (gh#191).
    fn chain_binomial_if2_stage() -> config_v2::Stage {
        config_v2::Stage::IF2 {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 1,
            particles: 10,
            iterations: 1,
            cooling: 0.7,
            cooling_target_iters: 1,
            starts_from: config_v2::StartsFrom::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            loglik_eval: config_v2::LoglikEvalConfig::default(),
            gate: config_v2::GateConfig::default(),
            dt_check: config_v2::DtCheckConfig::default(),
        }
    }

    /// An `mh`-on-ODE stage carrying `burnin_dt`, for the gh#449 warm-up-step
    /// checks. Only `backend` and `burnin_dt` matter to the gate.
    fn mh_stage_with_burnin_dt(burnin_dt: Option<f64>) -> config_v2::Stage {
        config_v2::Stage::Mh {
            backend: crate::run_meta::InferenceBackend::Ode,
            chains: 1,
            iterations: 1,
            starts_from: config_v2::StartsFrom::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            burn_in: None,
            thin: None,
            adapt: true,
            adapt_start: 300,
            burnin_dt,
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
        let stages = vec![("scout", &stage)];
        let err = gate_run_stages_against_model(&stages, &compiled, 1.0)
            .expect_err("real-coupled model on a chain_binomial fit stage must be rejected");
        assert!(err.contains("gh#191"), "should cite the tracking issue: {err}");
        assert!(
            err.contains("frozen"),
            "should explain the frozen-reservoir reason: {err}"
        );
        assert!(err.contains("scout"), "should name the offending stage: {err}");
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
        let stages = vec![("scout", &stage)];
        gate_run_stages_against_model(&stages, &compiled, 1.0).unwrap_or_else(|e| {
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
        let stages = vec![("scout", &stage)];
        let err = gate_run_stages_against_model(&stages, &compiled, 400.0)
            .expect_err("a fit dt that drops recurring fires must be rejected, not silently merged");
        assert!(err.contains("scout"), "should name the offending stage: {err}");
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
        let stages = vec![("scout", &stage)];
        gate_run_stages_against_model(&stages, &compiled, 1.0).unwrap_or_else(|e| {
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
        let stages = vec![("refine", &stage)];
        // dt = 1.0 is fine on its own — proven by the negative control above —
        // so any rejection here can only come from `burnin_dt`.
        let err = gate_run_stages_against_model(&stages, &compiled, 1.0).expect_err(
            "a coarse burnin_dt that drops recurring fires must be rejected even when dt is fine",
        );
        assert!(err.contains("refine"), "should name the offending stage: {err}");
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
        let stages = vec![("refine", &stage)];
        gate_run_stages_against_model(&stages, &compiled, 1.0).unwrap_or_else(|e| {
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
}

