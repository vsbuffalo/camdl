//! `ResolvedFitConfig`: the one resolved configuration of a `camdl fit run`,
//! from which both the fit's identity and what each method runs are built
//! (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`, phase 3
//! "The shape"; step 4 for `fit run`).
//!
//! ```text
//! FitRunArgs + fit.toml + defaults ──resolve──► ResolvedFitConfig
//!                                                 │            │
//!                          FitMethodCtx (per leaf)│            │ FitRunConfig::build, the runners
//!                                                 ▼            ▼
//!                               cas::resolve_fit_method     the engine
//! ```
//!
//! [`resolve`] applies every flag that changes what the fit computes to the
//! configuration — `--allow-forcing-extrapolation` into `[config]`,
//! `--starts`, `--starts-chains` and the sampler flags into `[method]` — then
//! compiles the model under the command's compile settings
//! ([`crate::util::CompileSettings`], `--no-licm` passed to the compile rather
//! than read from a process-wide setting), resolves the observation anchors,
//! the step (the dt rule), `[fixed] from_scenario` and the chain-starts rule,
//! and validates the whole, all before anything runs. `cmd_fit_run_v2`
//! destructures the value with every field named, so a field added here does
//! not compile there until it is used.
//!
//! **What reaches the identity, and how.** The fit's identity is still the
//! canonical-JSON hash of its configuration (`fit::cas`; the typed identity is
//! the follow-up proposal): the problem half of [`ResolvedFitConfig::config`]
//! at the `fit` level, its method at the `method` level, each seed at the
//! `seed` level. [`ResolvedFitConfig::model`] and [`ResolvedFitConfig::engine`]
//! (this build and the engine switches set, `env::Environment::engine_identity`)
//! are the fit level's model and engine. The compile settings reach it through
//! the IR they produce. `--sweep` and `--seed`/`fit_seeds` expand the run into
//! leaves; `--emit-every` keys a synthetic fit through the data it generates;
//! `--resume` through the method level's target length and its dep on the base.
//!
//! **What is not here.** `--parallel` and `--force` are execution: how the
//! leaves are computed, never what. `--label` is provenance, and the progress
//! flags presentation; neither is a field of this configuration, so neither
//! can reach the engine through it.

use super::chain_starts;
use super::config_v2::{self, FitConfig};

/// The resolved configuration of one `camdl fit run`.
pub(crate) struct ResolvedFitConfig {
    /// The fit file's text, read once: what `config` was parsed from, what each
    /// fit folder keeps as `fit.toml.original` and builds its
    /// `fit.resolved.toml` from ([`resolved_config_text`]), and what its config
    /// identity and `fit_toml_hash` are taken from. Nothing after resolution
    /// reads the file again, so an edit to it while the fit runs reaches no
    /// leaf.
    pub config_text: String,
    /// The configuration the fit computes from, keys and stores: the file with
    /// every flag applied, the step resolved, `[fixed] from_scenario`
    /// expanded, the starts rule resolved and `problem.compiled_ir` the
    /// compiled model's file.
    pub config: FitConfig,
    /// The model compiled under the command's compile settings, observation
    /// anchors resolved and external inputs bound: what every method loads
    /// (from `config.problem.compiled_ir`) and what the fit level hashes.
    pub model: ir::Model,
    /// The files `[data.tables]` bound into `model`, read once: their digests
    /// enter the fit level (`FitDigest.tables`) and their bytes are stored in
    /// each fit folder. Empty for a model with no external input.
    pub tables: crate::external_inputs::BoundFiles,
    /// `[config] allow_forcing_extrapolation`, as the policy every model the
    /// fit evaluates runs under and every record it writes names; built once,
    /// from `config`.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
    /// The fit seeds: the file's `fit_seeds`, else `--seed`, else 1. One leaf
    /// each.
    pub seeds: Vec<u64>,
    /// `--sweep`: each point's values replace `[fixed]` ones; one fit folder
    /// per point.
    pub sweep: Vec<(String, Vec<f64>)>,
    /// `--emit-every`: the cadence the `[synthetic]` generator emits at.
    pub emit_every: Option<crate::emit_every::EmitEvery>,
    /// `--resume`: the base a PGAS or PMMH chain continues, as named. Keyed
    /// through the run's length and its dependency on the base's carried
    /// chain files.
    pub resume: Option<String>,
    /// The run store's root the file names (its `store` key); `CAMDL_STORE`
    /// and the default `./camdl-store` come after it.
    pub store_root: Option<std::path::PathBuf>,
    /// The compile without the state Jacobian, rewritten from the data, when
    /// `config.problem.compiled_ir` is the compile with it (`nl-lbfgs` on the
    /// ODE backend): the compile the problem is posed on. `None` when the
    /// two are one compile.
    pub problem_ir: Option<String>,
    /// The NUTS tuning flags given on the command line (`--max-tree-depth`,
    /// `--diagonal-mass`), by spelling, for the note a PGAS fit gives when the
    /// model cannot run NUTS. Their values are in `config`'s method.
    pub nuts_tuning_flags: Vec<&'static str>,
    /// How each `[fixed]` name reached the config, read before the
    /// `from_scenario` expansion, for the fit's partition.
    pub fixed_via: Result<Vec<(String, crate::roles::FitFixedVia)>, String>,
    /// The chain-starts rule as resolved.
    pub starts: StartsPlan,
}

/// The chain-starts rule, resolved. A sourced rule's handle is resolved in
/// the run store, once it is open.
pub(crate) struct StartsPlan {
    /// Where the rule came from and what it is, as the run header says it.
    pub resolution: config_v2::StartsResolution,
    /// `--allow-nonconverged-source`: a source fit whose verdict is not
    /// converged is used rather than refused. It decides whether the run
    /// starts, never what it computes once started.
    pub allow_nonconverged_source: bool,
    /// Where the rule came from, for the prefix of an error about it:
    /// `--starts <spec>` or the fit file.
    pub origin: String,
}

/// Resolve `fit run`'s configuration from its arguments, its file and the
/// defaults, under the compile settings `compile`; refuse (through `out`) a
/// combination that cannot run. Returns the value and the forcing policy's
/// installation, held for the command.
pub(crate) fn resolve(
    a: &crate::args::FitRunArgs,
    config_text: String,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
) -> (ResolvedFitConfig, crate::forcing_extrapolation::Installed) {
    let fit_path              = a.config.to_string_lossy().into_owned();
    let base_seed             = a.run.seed.unwrap_or(1);

    // gh#540: every CLI flag that changes what the method computes or stores,
    // collected once so the method identity can see all of them. Applied to
    // the in-memory method below, before the CAS claim — the dispatch sites
    // no longer override anything, so there is no route around the key.
    let crate::args::FitMethodFlags {
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
    } = a.method.clone();
    let cli_overrides = crate::fit::config_v2::CliMethodOverrides {
        max_tree_depth,
        trajectory_warmup,
        csmc_sweeps_per_nuts,
        n_trajectories,
        diagonal_mass,
        no_nuts,
        no_ancestor_sampling,
        no_adapt,
        adapt_start,
        cooling_target_iters,
        decibans_thresh,
        no_dt_check,
        dt_check_halvings,
        dt_check_strict,
        binomial,
        record_ancestry,
        record_prequential,
    };
    let sweep_specs: Vec<(String, Vec<f64>)> = a.problem.sweep.iter()
        .map(|s| (s.name.clone(), s.grid.expand()))
        .collect();

    // Load the file: the problem half and the one `[method]`.
    let path = std::path::Path::new(&fit_path);
    let mut config = FitConfig::parse_anchored_at(&config_text, path, path).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // `fit run` runs the file's one method and refuses a file that has none,
    // naming the table. Every other reader takes the problem alone.
    if let Err(e) = config.method() {
        out.fail(format!("error: {}: {}", fit_path, e));
    }
    // `[method]` keys the method's own settings leave inert. Here and not in
    // the shared parser, which also reads stored fits' archived configs.
    {
        let method = config.inference.method.as_ref().expect("checked above");
        if let Err(e) = config_v2::refuse_inert_method_keys_in(&config_text, method) {
            out.fail(format!("error: {fit_path}: {e}"));
        }
    }
    // gh#audit-C6 / gh#189: degenerate-rate handling is a keyed `[config]` field
    // (folds into the fit-identity hash via the config blob). Every model this
    // fit compiles carries it (`FitBackendConfig::eval_settings`): the
    // capability scan and gradient preflight below, `FitRunConfig::build` for
    // each method, and the `[synthetic]` generator's runs.
    // A forcing outside its range: the flag sets the keyed `[config]` field,
    // so the override reaches the fit identity either way.
    // What the flag folds into the problem is recorded, so the fit folder
    // stores the config the fit ran under (gh#1268).
    if a.problem.forcing_extrapolation.allow_forcing_extrapolation {
        config.problem.config.allow_forcing_extrapolation = true;
    }
    // The policy every model this fit evaluates runs under, and every record
    // it writes names: installed for the rest of the command.
    let forcing =
        crate::forcing_extrapolation::ForcingPolicy::new(config.problem.config.allow_forcing_extrapolation);
    let _forcing_route = forcing.install();

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
    let emit_every = crate::emit_every::EmitEvery::from_cli_specs(&a.problem.emit_every)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    if emit_every.is_some() && !config.problem.is_synthetic_fit() {
        out.fail(format!(
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
        ));
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
            // A flag the method does not read is refused here, before any
            // work, rather than dropped (`OverrideFlag::methods`).
            if let Err(e) = method.algorithm.apply_cli_overrides(&cli_overrides) {
                out.fail(format!("error: {e}"));
            }
        }
        if let Some(starts) = &a.run.starts {
            method.starts = Some(starts.clone());
        }
        // gh#1056: `--starts-chains` qualifies the rule in force, from the
        // flag or the file; only `from_posterior` takes it.
        if let Some(subset) = &a.run.starts_chains {
            method.starts = Some(
                crate::fit::starts::apply_starts_chains(method.starts.clone(), subset, "fit")
                    .unwrap_or_else(|e| out.fail(format!("error: {e}"))));
        }
    }

    // Compile `model.camdl` → IR EXACTLY ONCE for the whole fit. Every
    // per-(cell × sweep point) `FitRunConfig::build` then loads this
    // pre-compiled IR instead of re-invoking camdlc per unit (a swept fit
    // otherwise recompiled the model dozens of times). The resolved path is
    // recorded on `config.problem.compiled_ir`; `model.camdl` is left
    // untouched so the fit's content hash still hashes the original `.camdl`
    // source bytes (identity is unchanged by the hoist). The temp IR persists
    // for the process — `resolve_ir_path`'s returned path is not a drop guard.
    //
    // gh#439 A2: only `nl-lbfgs` on the `ode` backend reads the WrtPop
    // state-Jacobian (`rate_state_grad` / `projection_state_grad`, via the ODE
    // forward-sensitivity gradient in `ode_grad::det_grad`). Otherwise compile lean
    // (`--no-state-grad`), which omits the dense ~O(G^3) Jacobian that dominates
    // coupled-model IR. The bit is folded into the IR-cache key, so a lean entry is
    // never reused for an nl-lbfgs+ode fit (and run identity is gradient-independent,
    // so lean vs full share the same model digest).
    let needs_state_grad = config.needs_state_grad();
    let (compiled_ir, _ir_tmp) = crate::util::resolve_ir_path(
        &config.problem.model.camdl, needs_state_grad, None, compile)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
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
    // The config's `[simulate] from` is resolved here too. Fit identity keys
    // on the model loaded from the re-emitted IR, so the resolved start and
    // horizon are in it: two windows are two fits, and one start spelled two
    // ways is one.
    let compiled_ir = super::resolve_anchors_into_temp_ir(Some(out), &compiled_ir, &config.problem)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // Bind the model's external tables and forcings to the files
    // `[data.tables]` names, once, the same way: every loader after this reads
    // the bound model, and what was read is kept for the identity and the fit
    // folder.
    let (compiled_ir, tables) = super::bind_tables_into_temp_ir(&compiled_ir, &config.problem)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {fit_path}: {e}"));
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // The problem is posed on the ordinary compile, which every method
    // shares; only `nl-lbfgs` on the ODE backend runs on the compile with the
    // state Jacobian, which is then the run's own content. The ordinary
    // compile is made here too, rewritten from the data the same way.
    let problem_ir: Option<String> = if needs_state_grad {
        let (lean, _lean_tmp) = crate::util::resolve_ir_path(&config.problem.model.camdl, false, None, compile)
            .unwrap_or_else(|e| out.fail(format!("error: {}", e)));
        let lean = super::resolve_anchors_into_temp_ir(None, &lean, &config.problem)
            .unwrap_or_else(|e| out.fail(format!("error: {}", e)));
        let (lean, _) = super::bind_tables_into_temp_ir(&lean, &config.problem)
            .unwrap_or_else(|e| out.fail(format!("error: {fit_path}: {e}")));
        Some(lean)
    } else {
        None
    };

    // Load model and validate completeness (from the pre-compiled IR).
    let (model, _) = crate::util::load_ir(&compiled_ir).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // The one dt rule (`crate::dt_rule`): the model's declared step is the
    // integration step for every method; a `[config] dt` the file writes that
    // differs is refused. The resolved step is written into the config before
    // anything reads it — the observation grid, the synthetic generator, every
    // runner, and the fit identity, which so keys on the step the fit ran at.
    {
        let written = config_v2::table_declares_key(&config_text, "config", "dt")
            .then_some(config.problem.config.dt);
        match crate::dt_rule::effective_dt(model.simulation.dt, written) {
            Ok(dt) => config.problem.config.dt = dt,
            Err(m) => {
                out.fail(format!("error: {fit_path}: {}", m.message("[config] dt")));
            }
        }
    }
    // How each `[fixed]` name reached the config, for the partition — read
    // before the expansion below folds a scenario's names into the inline
    // values. An error is the expansion's to report.
    let fixed_via = config.problem.fixed.declarations(&model);
    // gh#33: expand `[fixed] from_scenario = "name"` into the inline
    // values map by looking up the named scenario in the model. Must
    // happen after model load but before validate, so the every-param-
    // resolved check sees the expanded values.
    config.problem.expand_fixed_from_scenario(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
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
        // A model constant named in [estimate] or [fixed] is an unknown
        // parameter; the hint says how to make it configurable.
        let fixed = config.problem.fixed.resolve().unwrap_or_default();
        let named = config.problem.estimate.keys().chain(fixed.keys()).map(|s| s.as_str());
        out.fail(format!("error: {}{}", e, crate::util::constant_hints(&model, named)));
    });
    // gh#75: prior-presence check honoring the IR fallback. Has to run
    // after partition validation but with the model IR in scope.
    let ir_prior_params: std::collections::BTreeSet<&str> = model.parameters.iter()
        .filter(|p| p.prior_dist().is_some() || p.hierarchical().is_some())
        .map(|p| p.name.as_str())
        .collect();
    config.validate_priors_present(&ir_prior_params).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // gh#369: a fit.toml prior that displaces a model `~` prior is said once,
    // here, before the grid — a property of the config, not of a cell/chain.
    let estimated_names: Vec<String> = config.problem.estimate.keys().cloned().collect();
    if let Some(w) = crate::fit::priors_precedence::format_prior_override_warning(
        &crate::fit::priors_precedence::prior_overrides(
            &estimated_names, &config.problem.estimate, &model),
    ) {
        out.say_raw(format_args!("{}", w));
    }
    // `nl-lbfgs` is the one method whose objective needs a gradient of this
    // model. Ask the gradient capability gate now, before a leaf is claimed, so
    // a model it refuses costs a message rather than a half-run method.
    config.validate_gradient_capability(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_held_hierarchical_leaves(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_hierarchical_leaf_boxes(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_smooth_scale(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_smooth_coordinates(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });

    // The chain-starts rule, resolved against the problem before the identity
    // is taken (proposal §3.4): `from_prior` when every estimated parameter
    // declares a prior, `uniform_unconstrained` otherwise, unless the file or
    // `--starts` spelled one. Said at startup either way, so the rule a run
    // used is never a thing to infer.
    let starts_resolution = {
        let method = config.inference.method.as_mut().expect("checked above");
        method.resolve_starts(&config.problem, &model).unwrap_or_else(|e| {
            out.fail(format!("error: {}: {e}", super::starts_origin(a, &fit_path)));
        })
    };
    let starts_origin = super::starts_origin(a, &fit_path);
    // From here the method is read-only: the identity below sees exactly what
    // the runners consume.
    let method: &config_v2::Method = config.inference.method.as_ref().expect("checked above");
    let allow_nonconverged_source = a.allow_nonconverged_source;
    // Only a fit source carries a convergence verdict to lift.
    if allow_nonconverged_source {
        let rule = method.starts().unwrap_or_else(|e| { out.fail(format!("error: {e}")) });
        if let Err(e) = chain_starts::refuse_inert_allow_nonconverged(rule, "fit") {
            out.fail(format!("error: {e}"));
        }
    }
    if let Some(msg) = config.point_start_multichain_note() {
        // gh#71: a point-started multi-chain sampler's R̂ is uninformative;
        // the summary will report it as not assessed. A note, not an error —
        // the sample is still valid.
        out.say(format_args!("{} {}", crate::out::paint::note_tag(), msg));
    }
    // `--resume` continues a PGAS or PMMH chain. Other methods have no
    // extension dimension (IF2's cooling depends on total iterations, the
    // particle filter is one pass), so resuming would be statistically
    // incoherent.
    if a.run.resume.is_some()
        && !matches!(method.algorithm, config_v2::Algorithm::PGAS { .. } | config_v2::Algorithm::PMMH { .. })
    {
        out.fail(format!("error: --resume is only supported for PGAS and PMMH methods; \
                   this fit's method is '{}'.", method.algorithm.method_name()));
    }
    if a.run.seed.is_some() && config.inference.fit_seeds.is_some() {
        out.fail(format!("error: --seed sets the single seed of a fit, and {fit_path} declares \
                   `fit_seeds`, which replaces it, so the flag would change nothing. Drop \
                   --seed, or delete `fit_seeds` from the file."));
    }
    let fit_seeds: Vec<u64> = match &config.inference.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    // The run store's root the file names, anchored at the file; the
    // environment and the default come after it.
    let store_root = config.problem.store.as_ref().map(std::path::PathBuf::from);
    let nuts_tuning_flags: Vec<&'static str> =
        [(a.method.max_tree_depth.is_some(), "--max-tree-depth"), (a.method.diagonal_mass, "--diagonal-mass")]
            .into_iter()
            .filter_map(|(on, name)| on.then_some(name))
            .collect();
    let resolved = ResolvedFitConfig {
        config_text,
        config,
        model,
        tables,
        forcing,
        seeds: fit_seeds,
        sweep: sweep_specs,
        emit_every,
        resume: a.run.resume.clone(),
        store_root,
        problem_ir,
        nuts_tuning_flags,
        fixed_via,
        starts: StartsPlan {
            resolution: starts_resolution,
            allow_nonconverged_source,
            origin: starts_origin,
        },
    };
    (resolved, _forcing_route)
}

/// Refuses every removed spelling `a` carries, each with its replacement.
/// Run on the arguments as clap parsed them: the split that follows does not
/// carry the removed fields.
pub(crate) fn refuse_removed_flags(a: &crate::args::FitRunArgs, out: &crate::out::Out) {
    if let Some(msg) = super::removed_flag_message(a) {
        out.fail(format!("error: {}", msg));
    }
    // PGAS parallel tempering was removed (gh#789); the flag is trapped so a
    // stale script is answered with the fix rather than clap's "unexpected
    // argument".
    if let Some(raw) = a._removed_tempering.as_deref() {
        out.fail(format!(
            "error: --tempering is no longer accepted on `camdl fit run`: PGAS parallel \
             tempering was removed (gh#789), and every PGAS chain samples the posterior \
             directly, as the ladder's cold rung did. Drop `--tempering {raw}`; a \
             `tempering = [...]` line under [method] must be deleted too. Fits stored \
             before the removal are not reused and cannot be resumed; they run again."
        ));
    }
    // Correlated pseudo-marginal PMMH was removed; `--rho` is trapped the same
    // way.
    if let Some(raw) = a._removed_rho.as_deref() {
        out.fail(format!(
            "error: --rho is no longer accepted on `camdl fit run`: correlated \
             pseudo-marginal PMMH was removed, and `pmmh` runs the bootstrap particle \
             filter with independent likelihood evaluations. Drop `--rho {raw}`; a \
             `rho = ...` line under [method] must be deleted too. PMMH fits stored \
             before the removal are not reused and cannot be resumed; they run again."
        ));
    }
}
