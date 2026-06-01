//! `camdl fit` — structured inference workflow.
//!
//! Single entry point: `camdl fit run FIT.toml [--seed N] [--stage NAME]
//! [--label "..."] [--force]`. The fit.toml v2 schema declares stages
//! inline; the runner walks them in order. See
//! `docs/dev/proposals/2026-04-15-fit-run-spec-v0.4.md`.

pub mod config_v2;
pub mod state;
pub mod provenance;
pub mod runner;
pub mod priors_precedence;  // gh#75: shared prior-resolution chain for fit run + profile
pub mod fit_tree;
pub mod method_result;
pub mod config_diff;
pub mod table_row;
pub mod fit_table;
pub use fit_table::cmd_fit_table;
pub mod grid_summary;
pub mod fit_summary;
pub use fit_summary::cmd_fit_summary;
pub mod pmmh;
pub mod pgas;
pub mod trace_writer;
pub mod synthetic;
pub mod gating;
pub mod dt_check;
pub mod init;
pub mod chain_starts;
pub mod loglik_eval;
pub mod methods;
#[cfg(feature = "ode")]
pub mod nlopt_stage;

/// `camdl fit methods` — print the supported (algorithm, backend) pairs.
/// Reads from `methods::METHODS`, the single source of truth.
pub fn cmd_fit_methods() {
    print!("{}", methods::render_matrix());
}

pub fn cmd_fit_status(a: &crate::args::FitStatusArgs) {
    let path_str = match &a.path {
        Some(p) => p.to_string_lossy().into_owned(),
        None => {
            eprintln!("usage: camdl fit status [FILE_OR_DIR]");
            std::process::exit(1);
        }
    };
    let p = std::path::Path::new(&path_str);
    // Directory → walk it directly
    if p.is_dir() {
        run_status_v2_dir(&path_str);
        return;
    }
    // Treat the path as a v2 fit.toml. v1 schema is gone — any fit
    // config written before the v2-only cleanup landed will fail to
    // parse here; users on stale toml files get a typed parse error
    // pointing at the offending key, which is the right signal.
    let config = config_v2::FitConfigV2::load(&path_str).unwrap_or_else(|e| {
        eprintln!("error parsing fit.toml: {}", e);
        std::process::exit(1);
    });
    match config.fit_dir(&path_str) {
        Ok(fit_dir) if fit_dir.exists() => {
            run_status_v2_dir(&fit_dir.to_string_lossy());
        }
        Ok(fit_dir) => {
            eprintln!("no results found at {}", fit_dir.display());
        }
        Err(e) => {
            eprintln!("error computing fit directory: {}", e);
            std::process::exit(1);
        }
    }
}

/// Walk a results directory and report status of all stages found.
fn run_status_v2_dir(dir: &str) {
    let path = std::path::Path::new(dir);
    if !path.exists() {
        eprintln!("no results at {}", dir);
        return;
    }

    println!("{}/", dir);

    // Check for sweep subdirectories (contain stage dirs)
    // or direct stage dirs
    let mut found_stages = false;
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .unwrap_or_else(|e| { eprintln!("cannot read {}: {}", dir, e); std::process::exit(1); })
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let entry_path = entry.path();

        if entry_path.is_dir() {
            // Check if this is a stage dir (has fit_state.toml or run.json)
            let has_fit_state = entry_path.join("fit_state.toml").exists();
            let has_run_json  = entry_path.join("run.json").exists();

            if has_fit_state || has_run_json {
                // Direct stage
                print_stage_status(&name, &entry_path.to_string_lossy());
                found_stages = true;
            } else {
                // Might be a sweep point dir — check children
                let mut child_entries: Vec<_> = std::fs::read_dir(&entry_path)
                    .into_iter().flatten().flatten().collect();
                child_entries.sort_by_key(|e| e.file_name());
                let has_child_stages = child_entries.iter().any(|c| {
                    c.path().join("fit_state.toml").exists() || c.path().join("run.json").exists()
                });
                if has_child_stages {
                    println!("\n  \x1b[1m{}/\x1b[0m", name);
                    for child in &child_entries {
                        let child_name = child.file_name().to_string_lossy().to_string();
                        if child.path().is_dir() {
                            let child_has = child.path().join("fit_state.toml").exists()
                                || child.path().join("provenance.json").exists();
                            if child_has {
                                print_stage_status(&child_name, &child.path().to_string_lossy());
                            }
                        }
                    }
                    found_stages = true;
                }
            }
        }
    }

    if !found_stages {
        println!("  (no completed stages found)");
    }
}

fn print_stage_status(name: &str, stage_dir: &str) {
    use crate::run_meta::{Run, RunKind};

    // A completed v2 stage always has a FitStage run.json. The
    // fit_state.toml path (checked by the caller's directory walk)
    // is written earlier in the stage, so a dir with fit_state.toml
    // but no run.json is an interrupted run.
    match Run::read(std::path::Path::new(stage_dir)) {
        Ok(run) => {
            if let RunKind::FitStage(m) = &run.kind {
                let ll    = m.best_loglik.map(|l| format!("{:.1}", l)).unwrap_or_else(|| "—".into());
                let chain = m.best_chain.map(|c| format!(" (chain {})", c + 1)).unwrap_or_default();
                let wall = run.status.wall_time_seconds()
                    .map(|t| format!("{:.0}s", t))
                    .unwrap_or_else(|| "running".to_string());
                println!("    {:12} \x1b[32m✓\x1b[0m {} — loglik={}{}, {}",
                    name, m.method, ll, chain, wall);
            }
        }
        Err(_) => {
            // GH #18: a stage that ran IF2 + clean-eval but failed
            // the compound gate exits before writing run.json. The
            // user has nonzero results on disk (fit_state.toml,
            // mle_params.toml, etc.) and very much wants to know
            // *why* refine refused to advance. Detect this case by
            // checking for fit_state.toml; if present, point them at
            // `summary` rather than lying with "(no completed stages
            // found)". Full verdict lives in `camdl fit summary`.
            let stage_path = std::path::Path::new(stage_dir);
            if stage_path.join("fit_state.toml").exists() {
                let parent = stage_path.parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| ".".into());
                println!("    {:12} \x1b[31m✗\x1b[0m gate failed — see `camdl fit summary {}`",
                    name, parent);
            } else {
                println!("    {:12} \x1b[33m⚠\x1b[0m incomplete (no run.json)", name);
            }
        }
    }
}

// ─── New `camdl fit run` entry point (config_v2) ────────────────────────────

pub fn cmd_fit_run_v2(a: &crate::args::FitRunArgs) {
    use config_v2::{FitConfigV2, Stage, StartsFrom};

    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();  // gh#audit-H5
    sim::eval_stats::set_allow_degenerate_rates(a.allow_degenerate_rates);  // gh#audit-C6
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
    // CLI overrides for clean_eval / gate. clap enforces requires=stage so
    // these only fire when a single stage is selected, keeping scout and
    // refine independently overridable.
    let cli_loglik_eval_particles = a.loglik_eval_particles;
    let cli_loglik_eval_reps      = a.loglik_eval_reps;
    let cli_decibans_thresh      = a.decibans_thresh;
    // Construct the full InitMethod payload from the CLI tag +
    // companion paths. When `--init from_mle` is used the path-bearing
    // variant is consumed by `starts_from_override` above (the legacy
    // dispatcher's `--starts-from` semantic). For the other warm-start
    // variants (from_prior / from_posterior / from_params) the typed
    // InitMethod flows into `cli_init_method` and is dispatched
    // through `pgas_opts.init_method` / `pmmh_opts.init_method` /
    // the IF2 stage's `effective_init`.
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
    let cli_survey_path          = a.survey_path.clone();
    let cli_survey_top_k         = a.survey_top_k;
    let sweep_specs: Vec<(String, Vec<f64>)> = a.sweep.iter()
        .map(|s| (s.name.clone(), s.grid.expand()))
        .collect();

    // Load v2 config
    let mut config = FitConfigV2::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // Load model and validate completeness
    let (model, model_json) = crate::util::load_model(&config.model.camdl).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#33: expand `[fixed] from_scenario = "name"` into the inline
    // values map by looking up the named scenario in the model. Must
    // happen after model load but before validate, so the every-param-
    // resolved check sees the expanded values.
    config.fixed.expand_from_scenario(&model).unwrap_or_else(|e| {
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
        .filter(|p| p.prior.is_some() || p.hierarchical.is_some())
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
    if a.resume {
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

    let fit_dir = config.fit_dir(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // ── Build + write top-level run.json at the fit root ──────────────
    //
    // Per-stage run.json records live inside each stage dir; this one
    // describes the fit as a whole so `camdl list` / `camdl show` can
    // surface fits alongside simulate runs. `Run.hash` is the seed-
    // independent content hash (same suffix used in the directory name).
    //
    // We write once here (so the fit is listable even if interrupted)
    // and rewrite once at end-of-fit to capture `wall_time_seconds`.
    // The parent fit hash is also reused by every stage to populate
    // `FitStageMeta.fit_hash` — computing it once here avoids the O(stages
    // × full-I/O rehash) pattern.
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
    let mut run_fit = build_fit_run(&config, &fit_path, validated_label, Some(&model));
    let parent_fit_hash = run_fit.hash.clone();
    if let Err(e) = run_fit.write(&fit_dir) {
        eprintln!("warning: cannot write {}/run.json: {}", fit_dir.display(), e);
    }

    // Archive the fit.toml verbatim under <fit_dir>/fit.toml.original.
    // Step 6 of the experiment-management proposal: `fit table`'s
    // config_diff reader consumes this archive (not FitMeta.fit_toml_path,
    // which can move/change after the run). Write-once-on-first-run;
    // on cached re-entry into the same content-hashed fit_dir, verify
    // the current fit.toml is byte-identical to the archive (it must
    // be, by content-hash construction — a divergence here is a hash
    // collision or a bug, surfaced loudly).
    if let Err(e) = archive_fit_toml(&fit_path, &fit_dir) {
        eprintln!("warning: cannot archive fit.toml.original at {}: {}",
            fit_dir.display(), e);
    }

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
    eprintln!("  output:   {}", fit_dir.display());

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
    // Four canonical modes, all routed through the same grid:
    //   Mode                     synthetic?  fit_seeds     Cells
    //   Single fit               no          None/scalar   1       → real/fit_<base>/
    //   Start-sensitivity        no          list of M     M       → real/fit_<s_i>/
    //   SBC (classical)          yes         None/scalar   N       → synthetic/ds_NN/fit_<base>/
    //   SBC × start-sensitivity  yes         list of M     N × M   → synthetic/ds_NN/fit_<s_i>/
    //
    // For synthetic modes the datasets are generated once up front and the
    // per-cell DataSpec is materialised from their on-disk paths. See
    // docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.
    //
    // TODO(typed-cas): formalize fit_seeds as `cas::typed::ReplicateSet`.
    // Today this loop produces a sibling-cells layout that's *semantically*
    // a replicate set (one fit_content_hash umbrella, N seed-distinct
    // cells) but isn't wrapped in the formal ReplicateSet machinery —
    // no RunKind::ReplicateSet umbrella, no `replicates/seed_<S>/` path
    // convention, no cross-cell aggregator. Pull when (a) someone wants
    // a per-stage chain-Â diagnostic across fit_seeds, or (b) a uniform
    // path layout matters more than the breaking change to existing fit
    // trees. Same applies to dataset_idx as a nested replicate dimension.
    // See docs/dev/proposals/2026-04-28-cas-typed-runs-and-profile-stages.md
    // (Implementation checklist → "Fit run" deferred items).
    let fit_seeds: Vec<u64> = match &config.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    let synthetic_datasets: Vec<synthetic::SyntheticDataset> = if let Some(spec) = &config.synthetic {
        let datasets = synthetic::generate_synthetic_datasets(
            spec,
            &config.model.camdl,
            &fit_dir,
            config.config.backend,
            config.config.dt,
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
        let model_for_obs = {
            let (m, _) = crate::util::load_model(&config.model.camdl).unwrap_or_else(|e| {
                eprintln!("error loading model for obs stream names: {}", e);
                std::process::exit(1);
            });
            m
        };
        let obs_names: Vec<String> = model_for_obs.observations.iter()
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

    for (stage_name, stage) in &stages_to_run {
        let stage_dir = sweep_fit_dir.join(stage_name);
        eprintln!("\n── stage: {} (method={}) ──", stage_name, stage.method_name());

        // Config hash staleness check
        let fixed_resolved = sweep_config.fixed.resolve().unwrap_or_default();
        let data_spec = sweep_config.data_spec().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // Expand the [data] shorthand (`file = "..."`) into the canonical
        // per-stream map before hashing, so the shorthand and the
        // verbose `[data.observations]` form produce identical stage
        // hashes when they reference the same data.
        let model_obs_names: Vec<String> = serde_json::from_str::<serde_json::Value>(&model_json)
            .ok()
            .and_then(|v| v.get("observations").cloned())
            .and_then(|obs| serde_json::from_value::<Vec<serde_json::Value>>(obs).ok())
            .map(|obs| obs.into_iter()
                .filter_map(|o| o.get("name").and_then(|n| n.as_str().map(String::from)))
                .collect())
            .unwrap_or_default();
        let effective_obs = data_spec.effective_observations(&model_obs_names)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
        let config_hash = provenance::fit_stage_hash(
            &model_json, &effective_obs, &sweep_config.estimate,
            &fixed_resolved, &sweep_config.simplex_groups,
            stage_name, stage, seed,
        ).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        if !force && !a.resume {
            match crate::run_meta::Run::check_cache(&stage_dir, &config_hash) {
                crate::run_meta::CacheStatus::Hit => {
                    eprintln!("  \x1b[33mskipped — results already exist for these inputs.\x1b[0m");
                    eprintln!("  config_hash: {}", &config_hash[..16]);
                    eprintln!("  Use --force to re-run, or --resume to continue.");
                    continue;
                }
                crate::run_meta::CacheStatus::Stale { stored, current } => {
                    eprintln!("  \x1b[33mstale results detected — config has changed. Re-running.\x1b[0m");
                    eprintln!("  stored:  {}", &stored[..16.min(stored.len())]);
                    eprintln!("  current: {}", &current[..16.min(current.len())]);
                }
                crate::run_meta::CacheStatus::Miss => {}
            }
        }

        // Resolve starts_from: CLI override > stage config
        let effective_starts = if let Some(ref cli_sf) = starts_from_override {
            // CLI --starts-from applies to the target stage only
            if stages_to_run.len() == 1 {
                Some(cli_sf.clone())
            } else {
                None // only applies when running a single stage
            }
        } else {
            match stage.starts_from() {
                StartsFrom::Random => None,
                StartsFrom::Stage(ref dep_name) => {
                    // Resolve to the directory of a prior stage in this fit
                    Some(sweep_fit_dir.join(dep_name).to_string_lossy().to_string())
                }
                StartsFrom::Directory(ref path) => {
                    Some(path.to_string_lossy().to_string())
                }
            }
        };

        let stage_t0 = std::time::Instant::now();
        let mut stage_best_loglik: Option<f64> = None;
        let mut stage_best_chain: Option<usize> = None;

        match stage {
            Stage::IF2 { backend, chains, particles, iterations, cooling, cooling_target_iters, init_method, survey_path, survey_top_k_n, loglik_eval, gate, dt_check, .. } => {
                // Resolve effective clean_eval / gate: stage TOML, then CLI
                // override (per Step 4 — overrides are stage-scoped because
                // clap requires --stage). CLI flags pass `requires = "stage"`
                // so they cannot be set when running multiple stages, which
                // would otherwise apply the same value to scout and refine
                // and defeat independent tuning.
                let mut effective_loglik_eval = loglik_eval.clone();
                if let Some(n) = cli_loglik_eval_particles { effective_loglik_eval.n_particles = n; }
                if let Some(m) = cli_loglik_eval_reps      { effective_loglik_eval.n_replicates = m; }
                let mut effective_gate = gate.clone();
                if let Some(db) = cli_decibans_thresh     { effective_gate.decibans_thresh = db; }
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

                let effective_cooling_target_iters = a.cooling_target_iters
                    .unwrap_or(*cooling_target_iters);
                let mut run_config = runner::FitRunConfig::build(
                    &sweep_config,
                    prior_state.as_ref(),
                    *chains, *particles, *iterations,
                    *cooling, effective_cooling_target_iters,
                    seed, effective_starts.is_none(),
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
                // CLI `--init` overrides the stage-config init_method
                // when a single stage is selected (clap requires --stage
                // with --init).
                let effective_init: crate::fit::init::InitMethod =
                    cli_init_method.clone().unwrap_or_else(|| init_method.clone());
                // CLI overrides for survey_top_k siblings (require
                // --stage; clap enforces). When the stage TOML sets
                // them too, CLI wins.
                let effective_survey_path: Option<std::path::PathBuf> =
                    cli_survey_path.clone().or_else(|| survey_path.clone());
                let effective_survey_top_k_n: Option<usize> =
                    cli_survey_top_k.or(*survey_top_k_n);
                // For survey_top_k we need to keep the SurveyTopKResult
                // around (not just the per-chain `chains`) so the
                // chain_init_source / chain_starts.tsv writers can pull
                // the survey's full hash out of it. Plain Lhs / Uniform
                // / Single produce no such result.
                let (per_chain_params, survey_top_k_result) =
                    if effective_starts.is_some() {
                        (None, None)
                    } else {
                        let model_hash_str = crate::hashing::model_hash(&model_json);
                        let data_hashes = init::compute_data_hashes(&effective_obs)
                            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                        let estimate_names: Vec<String> =
                            sweep_config.estimate.keys().cloned().collect();
                        let fixed_hashmap: std::collections::HashMap<String, f64> =
                            fixed_resolved.iter().map(|(k, v)| (k.clone(), *v)).collect();
                        let ctx = init::SurveyFitContext {
                            model_hash: &model_hash_str,
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
                    Some(stage_dir_str.as_ref()));
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
                // Apply CLI overrides on top of the stage TOML.
                // --no-dt-check forces enabled=false; --dt-check-halvings
                // overrides n_halvings; --dt-check-strict bumps the
                // backend default down to the strict threshold.
                let mut effective_dt_check = dt_check.clone();
                if a.no_dt_check {
                    effective_dt_check.enabled = false;
                }
                if let Some(n) = a.dt_check_halvings {
                    effective_dt_check.n_halvings = n;
                }
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
                );
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
                    rw_sd,
                    loglik_type: Some("if2".into()),
                    acceptance_rate: None,
                    tail_chain_agreement: chain_results.chain_agreement.clone(),
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
                let model_hash = crate::hashing::model_hash(&run_config.model_ir_json);
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
                    // Full fit content hash — lets a reader locate
                    // the originating fit dir from just the
                    // mle_params.toml. Pre-hardening this was
                    // model_hash[..8], which only collided when data
                    // and params happened to match across fits of the
                    // same model. Hardening ship-now #2.
                    input_hash: parent_fit_hash.clone(),
                    model_path: sweep_config.model.camdl.clone(),
                    model_hash: model_hash.clone(),
                    data_hashes: data_hashes.clone(),
                    seed,
                    stage: stage_name.to_string(),
                    best_chain: chain_results.best_chain,
                    backend: sweep_config.config.backend,
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

                eprintln!("\n{} complete in {:.1}s: {}/", stage_name, elapsed.as_secs_f64(), stage_dir.display());
                eprintln!("  best loglik: {:.1} (chain {})", chain_results.best_loglik, chain_results.best_chain + 1);
            }
            Stage::PGAS { .. } => {
                let mut pgas_opts = pgas::PgasStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                // Apply CLI overrides (each requires --stage; clap enforces).
                if let Some(ref t) = a.tempering {
                    if t.is_empty() || (t[0] - 1.0).abs() > 1e-9 {
                        eprintln!("error: --tempering must start with β=1.0 (cold chain). \
                                   Got: {:?}", t);
                        std::process::exit(1);
                    }
                    pgas_opts.tempering = t.clone();
                }
                if let Some(d) = a.max_tree_depth { pgas_opts.max_tree_depth = d; }
                if let Some(w) = a.trajectory_warmup { pgas_opts.trajectory_warmup = w; }
                if let Some(s) = a.csmc_sweeps_per_nuts { pgas_opts.csmc_sweeps_per_nuts = s; }
                if let Some(n) = a.n_trajectories { pgas_opts.n_trajectories = n; }
                if a.diagonal_mass { pgas_opts.dense_mass = false; }
                if a.no_nuts       { pgas_opts.use_nuts   = false; }
                if let Some(m) = cli_init_method.clone() { pgas_opts.init_method = m; }
                // gh#51 v2: CLI survey-path / survey-top-k overrides.
                // Same priority as IF2 / PMMH: CLI > stage TOML.
                if let Some(ref p) = cli_survey_path {
                    pgas_opts.survey_path = Some(p.clone());
                }
                if let Some(k) = cli_survey_top_k {
                    pgas_opts.survey_top_k_n = Some(k);
                }

                pgas::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    pgas_opts,
                    seed, force,
                    a.resume,
                    effective_starts.as_deref(),
                ).unwrap_or_else(|e| {
                    eprintln!("error running pgas stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                // Bubble loglik from fit_state.toml written by PGAS runner
                if let Ok(fs) = state::FitState::load(&stage_dir.to_string_lossy()) {
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
            }
            Stage::PMMH { .. } => {
                let mut pmmh_opts = pmmh::PmmhStageOpts::from_stage(stage)
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                if a.no_adapt { pmmh_opts.adapt = false; }
                if let Some(s) = a.adapt_start { pmmh_opts.adapt_start = s; }
                if let Some(r) = a.rho {
                    if !(0.0..1.0).contains(&r) {
                        eprintln!("error: --rho must be in [0, 1). Got: {}", r);
                        std::process::exit(1);
                    }
                    pmmh_opts.rho = Some(r);
                }
                if let Some(m) = cli_init_method.clone() { pmmh_opts.init_method = m; }
                // gh#51 v2: CLI survey-path / survey-top-k flags
                // override the stage-config values. Same priority chain
                // as IF2: CLI > stage TOML > default. `pmmh::run_stage`
                // does the cross-check resolution internally once it
                // has built the FitRunConfig.
                if let Some(ref p) = cli_survey_path {
                    pmmh_opts.survey_path = Some(p.clone());
                }
                if let Some(k) = cli_survey_top_k {
                    pmmh_opts.survey_top_k_n = Some(k);
                }

                pmmh::run_stage(
                    &sweep_config,
                    stage_name,
                    stage,
                    &stage_dir,
                    pmmh_opts,
                    seed, force,
                    /* check_variance */ false,
                    a.resume,
                    effective_starts.as_deref(),
                ).unwrap_or_else(|e| {
                    eprintln!("error running pmmh stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });
                if let Ok(fs) = state::FitState::load(&stage_dir.to_string_lossy()) {
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
            }
            Stage::NlSbplx(_) | Stage::NlBobyqa(_) => {
                #[cfg(feature = "ode")]
                {
                    // Hash data + model for the mle_params.toml provenance
                    // block (same shape as the IF2 path uses below).
                    let model_ir_json = std::fs::read_to_string(&sweep_config.model.camdl)
                        .ok()
                        .and_then(|_| {
                            // The fit runner has already loaded + compiled
                            // the model; ask `util::load_model` for the
                            // canonical IR JSON used to compute the hash.
                            crate::util::load_model(&sweep_config.model.camdl).ok()
                        })
                        .map(|(_, ir_json)| ir_json)
                        .unwrap_or_default();
                    let model_hash_for_prov = crate::hashing::model_hash(&model_ir_json);
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
                    nlopt_stage::run_stage(
                        &sweep_config,
                        stage_name,
                        stage,
                        &stage_dir,
                        seed,
                        effective_starts.as_deref(),
                        &parent_fit_hash,
                        &model_hash_for_prov,
                        &data_hashes_for_prov,
                    ).unwrap_or_else(|e| {
                        eprintln!("error running nlopt stage '{}': {}", stage_name, e);
                        std::process::exit(1);
                    });
                    if let Ok(fs) = state::FitState::load(&stage_dir.to_string_lossy()) {
                        stage_best_loglik = Some(fs.best_loglik);
                        stage_best_chain = Some(fs.best_chain);
                    }
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
                let record_ancestry = *record_ancestry || a.record_ancestry;
                let want_prequential = *record_prequential || a.record_prequential;
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
                            let y_obs: Vec<f64> = run_config.observations.iter()
                                .map(|o| o.value).collect();
                            preq_trace = Some(sim::inference::prequential::build_trace(
                                recorded, &y_obs, &result.ess_trace, 0));
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

        // ── Shared run.json write (all stage types) ─────────────────────────
        let stage_elapsed = stage_t0.elapsed();
        // Resolve the upstream stage's name + hash from its run.json
        // (if it exists). `effective_starts` is a directory path — for
        // in-fit references it points to a sibling stage that's already
        // written its run.json by the time we run; for external
        // `--starts-from` it points to an arbitrary directory whose
        // run.json may or may not exist.
        let starts_from_ref = effective_starts.as_ref().map(|dir_path| {
            let p = std::path::Path::new(dir_path);
            let stage_name = p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            // Try to read the upstream run.json. On failure, record
            // `None` + warn — absent is the honest signal for "we
            // can't prove what this refers to", as distinct from empty
            // string which used to masquerade as a real hash.
            let stage_hash = match crate::run_meta::Run::read(p) {
                Ok(r) => Some(r.hash),
                Err(e) => {
                    eprintln!("warning: starts_from = {} has no readable \
                              run.json ({}); provenance chain will record \
                              stage_hash: null", dir_path, e);
                    None
                }
            };
            crate::run_meta::StartsFromRef { stage: stage_name, stage_hash }
        });
        let algo_tag = stage.method_name();
        let backend_tag = stage.backend().as_str();
        let algo_json = match stage {
            Stage::IF2 { chains, particles, iterations, cooling, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations, "cooling": cooling }),
            Stage::PGAS { chains, particles, sweeps, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "sweeps": sweeps }),
            Stage::PMMH { chains, particles, iterations, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": chains, "particles": particles, "iterations": iterations }),
            Stage::PFilter { particles, replicates, .. } =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "particles": particles, "replicates": replicates }),
            Stage::NlSbplx(c) | Stage::NlBobyqa(c) =>
                serde_json::json!({ "algorithm": algo_tag, "backend": backend_tag, "chains": c.chains, "tolerance": c.tolerance, "max_evals": c.max_evals }),
        };
        let n_chains = stage.chains();
        let stage_inputs = crate::cas::fit_inputs::StageInputs {
            fit_stage_hash: config_hash.clone(),
            stage_dir: stage_dir.clone(),
            meta: crate::run_meta::FitStageMeta {
                fit_hash: parent_fit_hash.clone(),
                stage: stage_name.to_string(),
                method: stage.method_kind(),
                backend: stage.backend(),
                seed,
                n_chains,
                algorithm: algo_json,
                best_loglik: stage_best_loglik,
                best_chain: stage_best_chain,
                starts_from: starts_from_ref,
                derived_from: sweep_config.provenance.as_ref()
                    .and_then(|p| p.derived_from.clone()),
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                // gh#83/gh#85 step 9: stage-level provenance is
                // populated by the inference algorithm runner (which
                // has the per-stage `ResolvedParameters` /
                // `ChainStarts` in scope); the CAS-hash skeleton
                // constructed here leaves them empty.
                parameters_provenance: Default::default(),
                init_provenance: None,
            },
        };
        use crate::cas::typed::CasInputs;
        let mut stage_run = stage_inputs.to_run(
            crate::version::VERSION_SHORT.to_string(),
            std::env::args().collect(),
        );
        stage_run.status = crate::run_meta::RunStatus::Completed {
            wall_time_seconds: stage_elapsed.as_secs_f64(),
        };
        if let Err(e) = stage_run.write(&stage_dir) {
            eprintln!("warning: could not write {}/run.json: {}", stage_dir.display(), e);
        }

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
        let path = fit_dir.join("sweep_failures.tsv");
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

    // ── Post-grid aggregation: summary.tsv (+ coverage.tsv for synthetic)
    //
    // Walk each cell's terminal-stage output, parse the `mle_params.toml`
    // back into a row, and write the tables. `summary.tsv` lives under
    // `real/` or `synthetic/` — the visual subdir that groups all of a
    // fit's cells.
    let terminal_stage = stages_to_run.last()
        .map(|(n, _)| n.to_string())
        .unwrap_or_else(|| "mle".to_string());

    let source = if config.synthetic.is_some() { "synthetic" } else { "real" };
    let mut rows: Vec<grid_summary::SummaryRow> = Vec::new();
    for (cell_i, cell) in cells.iter().enumerate() {
        let (dataset, cell_dir) = match cell.dataset_idx {
            Some(idx) => {
                let ds = format!("ds_{:02}", idx);
                let dir = fit_dir.join("synthetic").join(&ds).join(format!("fit_{}", cell.fit_seed));
                (ds, dir)
            }
            None => {
                let dir = fit_dir.join("real").join(format!("fit_{}", cell.fit_seed));
                ("real".to_string(), dir)
            }
        };
        match grid_summary::read_cell_row(&cell_dir, &terminal_stage, &dataset, cell.fit_seed) {
            Some(r) => rows.push(r),
            None    => eprintln!(
                "warning: cell {}/{} ({} × fit_seed={}) produced no mle_params.toml at {}",
                cell_i + 1, cells.len(), dataset, cell.fit_seed,
                cell_dir.join(&terminal_stage).display()),
        }
    }

    if !rows.is_empty() {
        match grid_summary::write_summary(&fit_dir, source, &rows) {
            Ok(p)  => eprintln!("summary: {}", p.display()),
            Err(e) => eprintln!("warning: could not write summary.tsv: {}", e),
        }
        if config.synthetic.is_some() {
            match grid_summary::load_truth(&fit_dir) {
                Ok(truth) => match grid_summary::write_coverage(&fit_dir, &truth, &rows) {
                    Ok(p)  => eprintln!("coverage: {}", p.display()),
                    Err(e) => eprintln!("warning: could not write coverage.tsv: {}", e),
                },
                Err(e) => eprintln!("warning: no truth for coverage: {}", e),
            }
        }
    }

    // ── Final rewrite: top-level run.json with accumulated wall time ──
    //
    // The top-level Run::Fit was written at fit-start with wall_time=0
    // so the fit is listable even if interrupted. Now that every stage
    // has completed (or aggregate post-processing has finished), patch
    // the wall-clock so `camdl list` / `camdl show` report honest
    // totals.
    run_fit.status = crate::run_meta::RunStatus::Completed {
        wall_time_seconds: fit_start.elapsed().as_secs_f64(),
    };
    if let Err(e) = run_fit.write(&fit_dir) {
        eprintln!("warning: cannot rewrite {}/run.json: {}", fit_dir.display(), e);
    }
}

/// Read an IR JSON string from a model path, compiling .camdl → IR on
/// the fly. Returns "" on any read/compile failure (callers then skip
/// hashing, matching the pre-existing `unwrap_or_default` semantics).
/// Fixes the gh #3 panic where a .camdl source was handed straight to
/// model_hash, which parses with serde_json and panicked on the source.
fn read_ir_json_or_empty(model_path: &str) -> String {
    if model_path.ends_with(".camdl") {
        crate::util::run_camdlc(model_path).unwrap_or_default()
    } else {
        std::fs::read_to_string(model_path).unwrap_or_default()
    }
}

/// Archive the fit.toml verbatim under `<fit_dir>/fit.toml.original`.
/// Step 6 of the experiment-management proposal.
///
/// The fit_dir is content-hashed on (model_hash, fit.toml bytes,
/// data hashes), so any two runs landing in the same fit_dir
/// necessarily had byte-identical fit.toml content at hash time. The
/// archive captures that fit.toml so `fit table`'s config_diff
/// reader can compare fits without depending on `FitMeta.fit_toml_path`
/// — the user's original file path, which can move or change after
/// the run.
///
/// Policy:
/// - **Write once.** If `<fit_dir>/fit.toml.original` does not exist,
///   copy the current fit.toml there.
/// - **Verify on cached hit.** If it already exists, read both the
///   archive and the current fit.toml; warn loudly if they differ.
///   (They must not — a divergence indicates a hash collision or a
///   bug in fit_content_hash. The warning is the alarm; the archive
///   is preserved as the canonical version.)
///
/// Returns Err on filesystem errors; the caller's responsibility to
/// surface the failure mode (the runner logs it as a warning rather
/// than aborting, since fit-state writes happen later anyway).
fn archive_fit_toml(fit_path: &str, fit_dir: &std::path::Path)
    -> std::io::Result<()>
{
    let archive_path = fit_dir.join("fit.toml.original");
    let current = std::fs::read(fit_path)?;
    if archive_path.exists() {
        let archived = std::fs::read(&archive_path)?;
        if archived != current {
            eprintln!(
                "warning: fit.toml.original at {} differs from current {}; \
                 this should be impossible (content-hash mismatch). \
                 Archive preserved.",
                archive_path.display(), fit_path);
        }
        return Ok(());
    }
    std::fs::write(&archive_path, &current)
}

/// Build the top-level `Run::Fit` record for a fit.toml. Fields that
/// require I/O (model IR, data files, fit.toml bytes) are read here
/// and hashed; `wall_time_seconds` is initialised to 0 and patched at
/// end-of-fit by the caller. Silent fallbacks (empty strings / empty
/// maps) cover the read-error case so a partially-written fit still
/// produces something `camdl list` can display.
///
/// `model_for_priors` is the compiled IR model used to resolve
/// per-parameter prior provenance (`fit_toml / model_ir /
/// flat_explicit`); pass `None` when no model is in scope (`fit
/// where` doesn't load the IR for a path-only resolution) and the
/// `resolved_priors` audit will be empty in that case.
fn build_fit_run(
    config: &config_v2::FitConfigV2,
    fit_path: &str,
    label: Option<String>,
    model_for_priors: Option<&ir::Model>,
) -> crate::run_meta::Run {
    let fit_hash = config.fit_content_hash(fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_ir_json = read_ir_json_or_empty(&config.model.camdl);
    let model_hash = if model_ir_json.is_empty() {
        String::new()
    } else {
        crate::hashing::model_hash(&model_ir_json)
    };
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
    let resolved_priors: Vec<crate::run_meta::ResolvedPriorEntry> = match model_for_priors {
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
    let inputs = crate::cas::fit_inputs::FitInputs {
        fit_content_hash: fit_hash,
        stem: crate::hashing::path_stem_slug(fit_path),
        meta: crate::run_meta::FitMeta {
            model: config.model.camdl.clone(),
            model_hash,
            fit_toml_path: fit_path.to_string(),
            fit_toml_hash,
            data_hashes,
            estimated,
            fixed,
            stages_declared,
            ic_free: config.ic_free.unwrap_or(false),
            resolved_priors,
            // gh#83/gh#85 step 9: top-level fit-meta provenance is
            // populated by the fit-finalization layer that owns the
            // resolved-params view.
            parameters_provenance: Default::default(),
        },
    };
    use crate::cas::typed::CasInputs;
    let mut run = inputs.to_run(
        crate::version::VERSION_SHORT.to_string(),
        std::env::args().collect(),
    );
    run.label = label;
    run
}

fn format_prior(p: &Option<config_v2::EstimatePriorSpec>) -> String {
    match p {
        None => "(none)".to_string(),
        Some(spec) => crate::fit::config_diff::format_prior(spec),
    }
}

/// `camdl fit where FIT.toml [--seed N]`
///
/// Resolves a fit.toml to its fit directory under the content-
/// addressable output tree and prints the path on stdout. Without
/// `--seed`, prints the top-level fit root
/// (`results/fits/<stem>-<hash[:8]>/`); with `--seed N`, prints the
/// cell dir (`.../real/fit_N/`).
///
/// Doesn't run anything — pure path resolution. Useful for scripts
/// that need to find the fit dir programmatically without globbing
/// on the stem prefix.
///
/// Hardening proposal ship-now #8.
pub fn cmd_fit_where(a: &crate::args::FitWhereArgs) {
    let fit_path = a.config.to_string_lossy().into_owned();
    let seed     = a.seed;

    // v2 fit.toml only — v1 schema deleted in the v1-cleanup pass.
    let mut config = config_v2::FitConfigV2::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("error parsing fit.toml: {}", e);
        std::process::exit(1);
    });

    // Run the same deeper validation `fit run` runs (gh#35). Loading +
    // light parsing isn't enough — `[estimate]` entries with missing
    // `start =`, `[fixed]` blocks that don't cover every model param,
    // and other completeness checks only fire once the model's
    // parameter list is known. Without this, `fit where` silently
    // accepts toml that `fit run` would reject — a misleading
    // affordance for scripts using `where` as a "is this valid?" probe.
    let (model, _) = crate::util::load_model(&config.model.camdl).unwrap_or_else(|e| {
        eprintln!("error loading model '{}': {}", config.model.camdl, e);
        std::process::exit(1);
    });
    // gh#33: expand `[fixed] from_scenario = "name"` (must run before
    // validate so the every-param-resolved check sees the scenario's
    // values).
    config.fixed.expand_from_scenario(&model).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_params: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
    config.validate(&model_params).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#75: mirror `cmd_fit_run`'s prior-presence check so `where`
    // doesn't silently accept a toml that `run` would reject.
    let ir_prior_params: std::collections::BTreeSet<&str> = model.parameters.iter()
        .filter(|p| p.prior.is_some() || p.hierarchical.is_some())
        .map(|p| p.name.as_str())
        .collect();
    config.validate_priors_present(&ir_prior_params).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let root = config.fit_dir(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let dir = match seed {
        None => root,
        Some(s) => root.join("real").join(format!("fit_{}", s)),
    };
    println!("{}", dir.display());
}

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

    // Find the first stage and update starts_from to point to source's results
    let source_config = config_v2::FitConfigV2::load(&from).ok();
    if let Some(ref cfg) = source_config {
        let source_fit_dir = match cfg.fit_dir(&from) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("warning: could not compute source fit dir: {}", e);
                return;
            }
        };
        if let Some(last_stage) = cfg.stages.keys().last() {
            let starts_path = source_fit_dir.join(last_stage);
            if starts_path.exists() {
                eprintln!("  [provenance] derived_from = \"{}\"", from);
                eprintln!("  hint: set starts_from = \"{}\" on your first stage",
                    starts_path.display());
            }
        }
    }

    std::fs::write(&to, &content).unwrap_or_else(|e| {
        eprintln!("error writing {}: {}", to, e);
        std::process::exit(1);
    });

    eprintln!("created {}", to);
}

/// Accept either a directory path or a git-style short hash for
/// `--starts-from`. The heuristic: contains `/` or `\\` → path
/// (today's behavior); else → resolve as Run.hash prefix via
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
/// for `run.json` files whose `Run.hash` starts with the prefix. The
/// label is validated, written to `Run.label`, and the run.json is
/// rewritten atomically. Refuses to relabel a still-running fit
/// (`status == Running`).
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

    let root = args.root.clone().unwrap_or_else(||
        std::path::PathBuf::from(crate::run_paths::DEFAULT_OUTPUT_ROOT));
    if !root.exists() {
        eprintln!("error: no results root at {}", root.display());
        std::process::exit(1);
    }

    // Walk every `run.json` under <root>/{sims,fits,profiles}/**.
    // Match by Run.hash prefix; collect ambiguous results for diagnostics.
    let mut matches: Vec<std::path::PathBuf> = Vec::new();
    for top in ["sims", "fits", "profiles"] {
        let subroot = root.join(top);
        if !subroot.exists() { continue; }
        find_runs_with_prefix(&subroot, &args.hash, &mut matches);
    }

    let run_dir = match matches.len() {
        0 => {
            eprintln!("error: no run found with hash prefix `{}` under {}",
                args.hash, root.display());
            std::process::exit(1);
        }
        1 => matches.into_iter().next().unwrap(),
        n => {
            eprintln!("error: hash prefix `{}` matches {} runs — \
                       use a longer prefix", args.hash, n);
            for p in &matches[..n.min(8)] {
                eprintln!("  {}", p.display());
            }
            std::process::exit(1);
        }
    };

    let run_json_path = run_dir.join("run.json");

    // New-format (`runid::RunRecord`) sims: the label lives in `provenance`.
    // (Sims are always written `Completed`, so there is no in-progress gate.)
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

    // Legacy `run_meta::Run` (fit / profile / survey / replicate-set).
    let mut run = match crate::run_meta::Run::read(&run_dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", run_json_path.display(), e);
            std::process::exit(1);
        }
    };

    // Atomicity: refuse to relabel a still-running fit. The runner
    // sets wall_time_seconds at end-of-fit; mid-run it stays at the
    // RunStatus::Running. A running fit will overwrite our label
    // change when it finishes. Sim/profile/replicate-set writes set
    // status to Completed at write time, so the gate only fires for
    // mid-run fits.
    if run.status.is_running() {
        eprintln!(
            "error: run at {} is still in progress (status = running). \
             Wait for it to finish, or pass --label at run time.",
            run_dir.display());
        std::process::exit(1);
    }

    let prior = run.label.clone();
    run.label = Some(new_label.clone());
    if let Err(e) = run.write(&run_dir) {
        eprintln!("error: cannot write {}: {}", run_json_path.display(), e);
        std::process::exit(1);
    }
    match prior {
        Some(p) if p != new_label =>
            eprintln!("ok: label updated from \"{}\" to \"{}\" on {}",
                p, new_label, run_dir.display()),
        Some(_) =>
            eprintln!("ok: label unchanged (\"{}\") on {}",
                new_label, run_dir.display()),
        None =>
            eprintln!("ok: label set to \"{}\" on {}",
                new_label, run_dir.display()),
    }
}

/// Recursively find directories under `root` whose `run.json` has a
/// `Run.hash` starting with `prefix`. Reads the JSON shallowly via
/// `serde_json::Value` to avoid deserializing every kind variant just
/// to check the hash. Bounded depth comes for free from the on-disk
/// layout (sims: 3 levels, fits: up to 5, profiles: up to 5), so a
/// plain recursive walk is fine without a max-depth guard.
fn find_runs_with_prefix(
    root: &std::path::Path,
    prefix: &str,
    out: &mut Vec<std::path::PathBuf>,
) {
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() { continue; }
        let run_json = p.join("run.json");
        if run_json.is_file() {
            if let Ok(txt) = std::fs::read_to_string(&run_json) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    // Legacy `Run.hash` or new-format `RunRecord.run_id`.
                    let hash = v.get("hash").and_then(|h| h.as_str()).unwrap_or("");
                    let run_id = v.get("run_id").and_then(|h| h.as_str()).unwrap_or("");
                    let matched = (!hash.is_empty() && hash.starts_with(prefix))
                        || (!run_id.is_empty() && run_id.starts_with(prefix));
                    if matched {
                        out.push(p.clone());
                    }
                }
            }
        }
        find_runs_with_prefix(&p, prefix, out);
    }
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

    fn unique_tmp(tag: &str) -> std::path::PathBuf {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(
            format!("camdl_archive_{}_{}_{}", tag, std::process::id(), ns));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// First-run path: archive does not exist, so `archive_fit_toml`
    /// writes a verbatim copy of the source fit.toml.
    #[test]
    fn archive_fit_toml_writes_on_first_run() {
        let tmp = unique_tmp("first_run");
        let fit_path = tmp.join("fit.toml");
        let body = "# fit.toml\n[fit]\nmodel = \"sir.camdl\"\n";
        std::fs::write(&fit_path, body).unwrap();
        let fit_dir = tmp.join("fit_dir");
        std::fs::create_dir_all(&fit_dir).unwrap();

        archive_fit_toml(&fit_path.to_string_lossy(), &fit_dir).unwrap();

        let archived = std::fs::read_to_string(fit_dir.join("fit.toml.original")).unwrap();
        assert_eq!(archived, body, "archive must be byte-identical to source");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Cached-hit path: archive already exists with matching content
    /// (the canonical case under content-hash routing). `archive_fit_toml`
    /// verifies bytes and is a no-op — does not re-write, does not
    /// emit any warning.
    #[test]
    fn archive_fit_toml_no_op_on_matching_archive() {
        let tmp = unique_tmp("matching");
        let body = "# matching\n";
        let fit_path = tmp.join("fit.toml");
        std::fs::write(&fit_path, body).unwrap();
        let fit_dir = tmp.join("fit_dir");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let archive = fit_dir.join("fit.toml.original");
        std::fs::write(&archive, body).unwrap();
        // Capture mtime of existing archive so we can verify no rewrite.
        let mtime_before = std::fs::metadata(&archive).unwrap().modified().unwrap();

        archive_fit_toml(&fit_path.to_string_lossy(), &fit_dir).unwrap();

        // File should not have been rewritten — same content, same mtime.
        let mtime_after = std::fs::metadata(&archive).unwrap().modified().unwrap();
        assert_eq!(mtime_before, mtime_after,
            "archive must not be rewritten when content matches");
        let archived = std::fs::read_to_string(&archive).unwrap();
        assert_eq!(archived, body);

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Cached-hit path with mismatched archive: indicates a hash
    /// collision or a runner bug. `archive_fit_toml` returns Ok (the
    /// runner does not abort), preserves the existing archive (does
    /// not overwrite), and emits a loud stderr warning. The test
    /// verifies the preserve-existing semantic; the stderr warning is
    /// observed in the `cargo test --nocapture` mode and is not
    /// asserted-on here.
    #[test]
    fn archive_fit_toml_preserves_archive_on_mismatch() {
        let tmp = unique_tmp("mismatch");
        let archived_body = "# archived (canonical)\n";
        let current_body  = "# divergent (this should not happen)\n";
        let fit_path = tmp.join("fit.toml");
        std::fs::write(&fit_path, current_body).unwrap();
        let fit_dir = tmp.join("fit_dir");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let archive = fit_dir.join("fit.toml.original");
        std::fs::write(&archive, archived_body).unwrap();

        archive_fit_toml(&fit_path.to_string_lossy(), &fit_dir).unwrap();

        // Existing archive preserved, NOT overwritten with current.
        let archived = std::fs::read_to_string(&archive).unwrap();
        assert_eq!(archived, archived_body,
            "on mismatch, the archive must be preserved (not overwritten \
             with the divergent current fit.toml)");

        std::fs::remove_dir_all(&tmp).ok();
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

