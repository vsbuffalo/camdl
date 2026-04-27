//! `camdl fit` — structured inference workflow.
//!
//! Usage:
//!   camdl fit scout    fit.toml [--seed N] [--force]
//!   camdl fit refine   fit.toml --starts-from scout/ [--seed N] [--force]
//!   camdl fit validate fit.toml --starts-from refine/ [--seed N] [--force]
//!   camdl fit pmmh     fit.toml [--starts-from validate/] [--seed N] [--force] [--resume] [--check-variance]
//!   camdl fit pgas     fit.toml [--starts-from validate/] [--seed N] [--force]
//!   camdl fit status   fit.toml

pub mod config;
pub mod config_v2;
pub mod state;
pub mod provenance;
pub mod runner;
pub mod grid_summary;
pub mod fit_summary;
pub use fit_summary::cmd_fit_summary;
pub mod pmmh;
pub mod pgas;
pub mod trace_writer;
pub mod synthetic;
pub mod gating;
pub mod clean_eval;

use config::FitToml;

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
    // Try v2 config format
    match config_v2::FitConfigV2::load(&path_str) {
        Ok(config) => {
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
            return;
        }
        Err(e) => {
            // Check if it has [stages] (v2 marker) — if so, the error is real
            if let Ok(contents) = std::fs::read_to_string(&path_str) {
                if contents.contains("[stages.") || contents.contains("[stages]") {
                    eprintln!("error parsing v2 fit.toml: {}", e);
                    std::process::exit(1);
                }
            }
            // Otherwise fall through to v1
        }
    }
    // Fall back to v1 (legacy): use fit.toml's own seed field if present.
    let fit = FitToml::load(&path_str).unwrap_or_else(|e| {
        eprintln!("error: {}", e); std::process::exit(1);
    });
    let seed = fit.fit.seed.unwrap_or(1);
    let cell = fit.cell_dir(&path_str, seed).unwrap_or_else(|e| {
        eprintln!("error: {}", e); std::process::exit(1);
    });
    if !cell.exists() {
        eprintln!("no results at {}", cell.display());
        return;
    }
    run_status_v2_dir(&cell.to_string_lossy());
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
                println!("    {:12} \x1b[32m✓\x1b[0m {} — loglik={}{}, {:.0}s",
                    name, m.method, ll, chain, run.wall_time_seconds);
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

    let fit_path              = a.config.to_string_lossy().into_owned();
    let base_seed             = a.seed.unwrap_or(1);
    let force                 = a.force;
    let stage_filter          = a.stage.clone();
    let starts_from_override  = a.starts_from.as_ref().map(|s| resolve_starts_from_arg(s));
    let allow_nonconverged_scout = a.allow_nonconverged_scout;
    // CLI overrides for clean_eval / gate. clap enforces requires=stage so
    // these only fire when a single stage is selected, keeping scout and
    // refine independently overridable.
    let cli_clean_eval_particles = a.clean_eval_particles;
    let cli_clean_eval_reps      = a.clean_eval_reps;
    let cli_decibans_thresh      = a.decibans_thresh;
    let sweep_specs: Vec<(String, Vec<f64>)> = a.sweep.iter()
        .map(|s| (s.name.clone(), s.values.clone()))
        .collect();

    // Load v2 config
    let config = FitConfigV2::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // Load model and validate completeness
    let (model, model_json) = crate::util::load_model(&config.model.camdl).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_params: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
    config.validate(&model_params).unwrap_or_else(|e| {
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
    let mut run_fit = build_fit_run(&config, &fit_path);
    let parent_fit_hash = run_fit.hash.clone();
    if let Err(e) = run_fit.write(&fit_dir) {
        eprintln!("warning: cannot write {}/run.json: {}", fit_dir.display(), e);
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
    let fit_seeds: Vec<u64> = match &config.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    let synthetic_datasets: Vec<synthetic::SyntheticDataset> = if let Some(spec) = &config.synthetic {
        let datasets = synthetic::generate_synthetic_datasets(
            spec,
            &config.model.camdl,
            &fit_dir,
            &config.config.backend,
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

        // Recalculate the legacy bridge with swept values
        let sweep_legacy = sweep_config.to_legacy_toml().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

        // IC4 in 2026-04-19 inference review batch 3: reject
        // prior × transform combinations that silently produce a
        // different prior than the user wrote (log_normal on
        // Transform::None → Normal; log_normal on Logit → logit-
        // normal; etc.). Runs after sweep-value substitution since
        // sweep can change a param's role, but the prior/transform
        // binding itself is fixed across sweep points — this is
        // equivalent to a one-shot check at config load, but
        // putting it here means every cell sees its own validation.
        if let Err(e) = runner::validate_prior_transform_compat(&sweep_legacy, &model) {
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
        let config_hash = provenance::fit_stage_hash(
            &model_json, &data_spec.observations, &sweep_config.estimate,
            &fixed_resolved, stage_name, stage, seed,
        ).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        if !force {
            match crate::run_meta::Run::check_cache(&stage_dir, &config_hash) {
                crate::run_meta::CacheStatus::Hit => {
                    eprintln!("  \x1b[33mskipped — results already exist for these inputs.\x1b[0m");
                    eprintln!("  config_hash: {}", &config_hash[..16]);
                    eprintln!("  Use --force to re-run.");
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
            Stage::IF2 { chains, particles, iterations, cooling, clean_eval, gate, .. } => {
                // Resolve effective clean_eval / gate: stage TOML, then CLI
                // override (per Step 4 — overrides are stage-scoped because
                // clap requires --stage). CLI flags pass `requires = "stage"`
                // so they cannot be set when running multiple stages, which
                // would otherwise apply the same value to scout and refine
                // and defeat independent tuning.
                let mut effective_clean_eval = clean_eval.clone();
                if let Some(n) = cli_clean_eval_particles { effective_clean_eval.n_particles = n; }
                if let Some(m) = cli_clean_eval_reps      { effective_clean_eval.n_replicates = m; }
                let mut effective_gate = gate.clone();
                if let Some(db) = cli_decibans_thresh     { effective_gate.decibans_thresh = db; }
                let prior_state = effective_starts.as_ref().and_then(|dir| {
                    state::FitState::load(dir).ok()
                });

                // G6: announce effective gate thresholds at stage start
                // so users know what they're being judged against before
                // seeing results. Only when a prior stage exists (Gate 1
                // will fire); a scout with no predecessor is ungated.
                if effective_starts.is_some() {
                    eprintln!("  gate thresholds:  Â < {:.3}; decibans spread < {:.1} dB (or SE-adaptive floor if higher)",
                        effective_gate.a_thresh, effective_gate.decibans_thresh);
                }

                // Gate 1 — pre-stage: if this stage consumes a prior
                // stage (starts_from), refuse to run when the prior
                // stage's tail Â failed convergence on any
                // non-IVP param. Skipped when starts_from is absent
                // (this stage is itself the scout). Overridable via
                // --allow-nonconverged-scout. See proposal
                // docs/dev/proposals/2026-04-19-refine-gates-scout-convergence.md.
                let mut soft_warn_params: Vec<String> = vec![];
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
                            // Mi1: legacy state skipped gate — warn explicitly
                            // so the user knows convergence was not verified.
                            ScoutGateVerdict::LegacySkipped => {
                                eprintln!("\x1b[33m  warning:\x1b[0m prior stage \
                                           fit_state.toml predates chain-agreement tracking — \
                                           Gate 1 skipped (convergence unverified). \
                                           Re-run scout to populate Â data.");
                            }
                            ScoutGateVerdict::SoftWarn { param_agreement } => {
                                eprintln!("\x1b[33m  warning:\x1b[0m prior stage tail Â in \
                                           SoftWarn band ([{:.2}, {:.2})) for: {}",
                                    gating::A_SOFT, effective_gate.a_thresh,
                                    param_agreement.iter()
                                        .map(|(n, r)| format!("{} (Â={:.2})", n, r))
                                        .collect::<Vec<_>>().join(", "));
                                // G3: record soft-warn params so fit_state persists them
                                soft_warn_params = param_agreement.iter()
                                    .map(|(n, r)| format!("{} (Â={:.2})", n, r))
                                    .collect();
                            }
                            ScoutGateVerdict::Hard { ref failing, ref all_structural, ref ivp, loglik_spread } => {
                                // M1: build per-param best-chain values from the prior
                                // stage's winning θ̂. These are the "copy into [estimate.*]
                                // bounds" hint — the most operationally useful guidance on
                                // a Hard failure and previously always None.
                                let best_vals: Vec<(String, f64)> = all_structural.iter()
                                    .filter_map(|(name, _)| {
                                        ps.start_values.get(name)
                                            .map(|&v| (name.clone(), v))
                                    })
                                    .collect();
                                let best_vals_ref: Option<&[(String, f64)]> =
                                    if best_vals.is_empty() { None }
                                    else { Some(&best_vals) };
                                let msg = gating::format_hard_verdict(
                                    failing, all_structural, ivp,
                                    loglik_spread, ps.best_loglik, best_vals_ref);
                                // M2: write gate_failure.toml so fit summary can surface
                                // the reason even without the terminal output.
                                let gfr = gating::GateFailureRecord {
                                    kind: gating::GateFailureKind::Gate1Hard,
                                    prior_stage: prior_state.as_ref()
                                        .map(|p| p.stage.clone()),
                                    timestamp: crate::cas::iso8601_utc(
                                        std::time::SystemTime::now()),
                                    a_thresh: Some(effective_gate.a_thresh),
                                    max_a: failing.iter().map(|(_, r)| *r)
                                        .reduce(f64::max),
                                    max_a_param: failing.first().map(|(n, _)| n.clone()),
                                    failing_params: failing.iter()
                                        .map(|(n, r)| format!("{}={:.3}", n, r))
                                        .collect(),
                                    loglik_spread: Some(loglik_spread),
                                    delta_db: None, threshold_db: None,
                                    scout_best_loglik: None, stage_best_loglik: None,
                                    regression_epsilon: None,
                                };
                                gating::write_gate_failure(&stage_dir, &gfr);
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
                                delta_db, threshold_db, sigma_max, se_floor_db, ref chain_logliks,
                            } => {
                                let msg = gating::format_decibans_spread_verdict(
                                    delta_db, threshold_db, sigma_max, se_floor_db, chain_logliks);
                                // M2: write gate_failure.toml for decibans-spread block
                                let gfr = gating::GateFailureRecord {
                                    kind: gating::GateFailureKind::Gate1DecibansSpread,
                                    prior_stage: prior_state.as_ref()
                                        .map(|p| p.stage.clone()),
                                    timestamp: crate::cas::iso8601_utc(
                                        std::time::SystemTime::now()),
                                    a_thresh: None, max_a: None, max_a_param: None,
                                    failing_params: vec![],
                                    loglik_spread: None,
                                    delta_db: Some(delta_db),
                                    threshold_db: Some(threshold_db),
                                    scout_best_loglik: None, stage_best_loglik: None,
                                    regression_epsilon: None,
                                };
                                gating::write_gate_failure(&stage_dir, &gfr);
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
                    &sweep_legacy,
                    prior_state.as_ref(),
                    *chains, *particles, *iterations,
                    *cooling, seed, effective_starts.is_none(),
                ).unwrap_or_else(|e| {
                    eprintln!("error building run config: {}", e);
                    std::process::exit(1);
                });
                run_config.clean_eval = effective_clean_eval.clone();
                run_config.gate = effective_gate.clone();

                std::fs::create_dir_all(&stage_dir).unwrap_or_else(|e| {
                    eprintln!("error creating {}: {}", stage_dir.display(), e);
                    std::process::exit(1);
                });

                let collector = sim::inference::diagnostic::DiagnosticCollector::new(stage_name);
                let t0 = std::time::Instant::now();
                // When a stage has no `starts_from` predecessor and runs
                // > 1 chain, give each chain its own random starting
                // point from bounds. This matches v1 scout's default
                // and is what makes Â across chains meaningful —
                // chains starting from the same point only diverge via
                // per-chain RNG, so their between-chain variance
                // reflects sampling noise rather than
                // independence-of-starts. When `starts_from` resolves
                // to a prior stage, every chain correctly starts from
                // that stage's MLE (the intent of the handoff).
                let per_chain_params = if effective_starts.is_none() && *chains > 1 {
                    Some(runner::build_random_chain_starts(&run_config, seed, *chains))
                } else {
                    None
                };
                let chain_results = runner::run_chains_with_per_chain_params(
                    &run_config, per_chain_params.as_deref(), &collector);
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
                        // M3: persist the regression failure so fit summary
                        // can surface the reason without requiring the user
                        // to have kept the terminal output.
                        let eps = gating::loglik_regression_epsilon(
                            &scout_chain_logliks_for_gate2);
                        let gfr = gating::GateFailureRecord {
                            kind: gating::GateFailureKind::Gate2Regression,
                            prior_stage: prior_state.as_ref()
                                .map(|p| p.stage.clone()),
                            timestamp: crate::cas::iso8601_utc(
                                std::time::SystemTime::now()),
                            a_thresh: None, max_a: None, max_a_param: None,
                            failing_params: vec![],
                            loglik_spread: None,
                            delta_db: None, threshold_db: None,
                            scout_best_loglik: Some(scout_best),
                            stage_best_loglik: Some(chain_results.best_loglik),
                            regression_epsilon: Some(eps),
                        };
                        gating::write_gate_failure(&stage_dir, &gfr);
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
                    Some(&chain_results.clean_eval),
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_clean_eval_tsv(
                    &stage_dir.to_string_lossy(),
                    &chain_results.clean_eval, &run_config.estimated_params,
                ).unwrap_or_else(|e| eprintln!("warning: {}", e));
                runner::write_run_root_final_params(
                    &stage_dir.to_string_lossy(),
                    &chain_results.clean_eval, &run_config.estimated_params,
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
                    chain_clean_logliks: chain_results.chain_clean_logliks(),
                    chain_clean_ses: chain_results.chain_clean_ses(),
                    // G3: record soft-warn params so fit summary can note
                    // them even after a successful run.
                    soft_warn_params,
                    // Persist the gate / clean-eval config that was
                    // *actually in force* — `effective_gate` and
                    // `effective_clean_eval` above already collapsed the
                    // priority chain (CLI flag > stage TOML > defaults).
                    // `summary` reads these so its verdict line reports
                    // against the threshold the run was judged by, not
                    // whatever `fit.toml` says at summary-time.
                    // See proposal §Phase 3.
                    resolved_gate: Some(effective_gate.clone()),
                    resolved_clean_eval: Some(effective_clean_eval.clone()),
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
                    backend: sweep_config.config.backend.clone(),
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
            Stage::PGAS { chains, particles, sweeps, burn_in, thin, .. } => {
                // Override legacy PGAS config with v2 stage values
                let mut legacy_pgas = sweep_legacy.pgas.clone().unwrap_or_default();
                legacy_pgas.chains = Some(*chains);
                legacy_pgas.particles = Some(*particles);
                legacy_pgas.sweeps = Some(*sweeps);
                if let Some(b) = burn_in { legacy_pgas.burn_in = Some(*b); }
                if let Some(t) = thin { legacy_pgas.thin = Some(*t); }
                legacy_pgas.starts_from = effective_starts.clone();
                let mut legacy_with_pgas = sweep_legacy.clone();
                legacy_with_pgas.pgas = Some(legacy_pgas);
                legacy_with_pgas.fit.output_dir = sweep_fit_dir.to_string_lossy().to_string();

                pgas::run_pgas_cli(
                    &legacy_with_pgas,
                    effective_starts.as_deref(),
                    seed, force, true, true, false,
                ).unwrap_or_else(|e| {
                    eprintln!("error running pgas stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });

                // Rename pgas/ → {stage_name}/ if they differ.
                // PGAS runner writes to {output_dir}/pgas/. If our stage has a
                // different name, move the output to the correct stage directory.
                //
                // Collision handling: previously this did a silent
                // `remove_dir_all` on the target. Under concurrent fits
                // against the same fit_dir, or mid-edit iteration, that
                // silently clobbered an existing stage's results. Now
                // we error unless `--force` is explicitly given.
                // cleanup-proposal §ship-now/#4.
                let pgas_dir = sweep_fit_dir.join("pgas");
                if stage_name != &"pgas" && pgas_dir.exists() {
                    if stage_dir.exists() {
                        if !force {
                            eprintln!("error: {} already exists. Refusing to \
                                      overwrite a previous stage's results.",
                                stage_dir.display());
                            eprintln!("  Pass --force to re-run this stage, \
                                      or remove the directory manually.");
                            std::process::exit(1);
                        }
                        eprintln!("warning: --force — removing existing {}", stage_dir.display());
                        if let Err(e) = std::fs::remove_dir_all(&stage_dir) {
                            eprintln!("error: could not remove {}: {}",
                                stage_dir.display(), e);
                            std::process::exit(1);
                        }
                    }
                    std::fs::rename(&pgas_dir, &stage_dir).unwrap_or_else(|e| {
                        eprintln!("error: could not rename pgas/ to {}: {}", stage_name, e);
                        std::process::exit(1);
                    });
                }
                // Bubble loglik from fit_state.toml written by PGAS runner
                if let Ok(fs) = state::FitState::load(&stage_dir.to_string_lossy()) {
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
            }
            Stage::PMMH { chains, particles, iterations, burn_in, thin, .. } => {
                let mut legacy_pmmh = sweep_legacy.pmmh.clone().unwrap_or_default();
                legacy_pmmh.chains = Some(*chains);
                legacy_pmmh.particles = Some(*particles);
                legacy_pmmh.steps = Some(*iterations);
                if let Some(b) = burn_in { legacy_pmmh.burn_in = Some(*b); }
                if let Some(t) = thin { legacy_pmmh.thin = Some(*t); }
                let mut legacy_with_pmmh = sweep_legacy.clone();
                legacy_with_pmmh.pmmh = Some(legacy_pmmh);
                legacy_with_pmmh.fit.output_dir = sweep_fit_dir.to_string_lossy().to_string();

                pmmh::run_pmmh_cli(
                    &legacy_with_pmmh,
                    effective_starts.as_deref(),
                    seed, force, false, false,
                ).unwrap_or_else(|e| {
                    eprintln!("error running pmmh stage '{}': {}", stage_name, e);
                    std::process::exit(1);
                });

                let pmmh_dir = sweep_fit_dir.join("pmmh");
                if stage_name != &"pmmh" && pmmh_dir.exists() {
                    if stage_dir.exists() {
                        let _ = std::fs::remove_dir_all(&stage_dir);
                    }
                    std::fs::rename(&pmmh_dir, &stage_dir).unwrap_or_else(|e| {
                        eprintln!("error: could not rename pmmh/ to {}: {}", stage_name, e);
                        std::process::exit(1);
                    });
                }
                if let Ok(fs) = state::FitState::load(&stage_dir.to_string_lossy()) {
                    stage_best_loglik = Some(fs.best_loglik);
                    stage_best_chain = Some(fs.best_chain);
                }
            }
            Stage::PFilter { particles, replicates, .. } => {
                let n_reps = replicates.unwrap_or(1);
                let prior_state = effective_starts.as_ref().and_then(|dir| {
                    state::FitState::load(dir).ok()
                });
                if prior_state.is_none() && !effective_starts.as_ref().is_none_or(|s| s.is_empty()) {
                    eprintln!("warning: could not load fit_state from starts_from");
                }

                // Build run config (reuse IF2 builder with 1 chain, N particles)
                let run_config = runner::FitRunConfig::build(
                    &sweep_legacy,
                    prior_state.as_ref(),
                    1, *particles, 1, 1.0, seed, false,
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
                    let record_preq = r == 0;
                    let smc_config = sim::inference::traits::SMCConfig {
                        record_prequential: record_preq,
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
        let algo_json = match stage {
            Stage::IF2 { chains, particles, iterations, cooling, .. } =>
                serde_json::json!({ "method": "if2", "chains": chains, "particles": particles, "iterations": iterations, "cooling": cooling }),
            Stage::PGAS { chains, particles, sweeps, .. } =>
                serde_json::json!({ "method": "pgas", "chains": chains, "particles": particles, "sweeps": sweeps }),
            Stage::PMMH { chains, particles, iterations, .. } =>
                serde_json::json!({ "method": "pmmh", "chains": chains, "particles": particles, "iterations": iterations }),
            Stage::PFilter { particles, replicates, .. } =>
                serde_json::json!({ "method": "pfilter", "particles": particles, "replicates": replicates }),
        };
        let n_chains = match stage {
            Stage::IF2  { chains, .. } => *chains,
            Stage::PGAS { chains, .. } => *chains,
            Stage::PMMH { chains, .. } => *chains,
            Stage::PFilter { .. }      => 1,
        };
        let stage_run = crate::run_meta::Run {
            hash: config_hash.clone(),
            version: crate::version::VERSION_SHORT.to_string(),
            created_at: crate::cas::iso8601_utc(std::time::SystemTime::now()),
            argv: std::env::args().collect(),
            wall_time_seconds: stage_elapsed.as_secs_f64(),
            kind: crate::run_meta::RunKind::FitStage(crate::run_meta::FitStageMeta {
                fit_hash: parent_fit_hash.clone(),
                stage: stage_name.to_string(),
                method: stage.method_name().to_string(),
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
            }),
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
    run_fit.wall_time_seconds = fit_start.elapsed().as_secs_f64();
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

/// Build the top-level `Run::Fit` record for a fit.toml. Fields that
/// require I/O (model IR, data files, fit.toml bytes) are read here
/// and hashed; `wall_time_seconds` is initialised to 0 and patched at
/// end-of-fit by the caller. Silent fallbacks (empty strings / empty
/// maps) cover the read-error case so a partially-written fit still
/// produces something `camdl list` can display.
fn build_fit_run(config: &config_v2::FitConfigV2, fit_path: &str) -> crate::run_meta::Run {
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
    crate::run_meta::Run {
        hash: fit_hash,
        version: crate::version::VERSION_SHORT.to_string(),
        created_at: crate::cas::iso8601_utc(std::time::SystemTime::now()),
        argv: std::env::args().collect(),
        wall_time_seconds: 0.0,
        kind: crate::run_meta::RunKind::Fit(crate::run_meta::FitMeta {
            model: config.model.camdl.clone(),
            model_hash,
            fit_toml_path: fit_path.to_string(),
            fit_toml_hash,
            data_hashes,
            estimated,
            fixed,
            stages_declared,
            ic_free: config.ic_free.unwrap_or(false),
        }),
    }
}

fn format_prior(p: &Option<config_v2::PriorSpec>) -> String {
    match p {
        None => "(none)".to_string(),
        Some(config_v2::PriorSpec::LogNormal { mu, sigma }) => format!("log_normal(mu={}, sigma={})", mu, sigma),
        Some(config_v2::PriorSpec::Normal { mu, sigma }) => format!("normal(mu={}, sigma={})", mu, sigma),
        Some(config_v2::PriorSpec::Beta { alpha, beta }) => format!("beta(alpha={}, beta={})", alpha, beta),
        Some(config_v2::PriorSpec::Uniform) => "uniform".to_string(),
        Some(config_v2::PriorSpec::HalfNormal { sigma }) => format!("half_normal(sigma={})", sigma),
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

    // Try v2 first (the common path); fall through to v1 on parse
    // failure. Mirrors the detection logic in cmd_fit_status.
    let v2 = config_v2::FitConfigV2::load(&fit_path);
    let dir = match v2 {
        Ok(config) => {
            let root = config.fit_dir(&fit_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
            match seed {
                None => root,
                Some(s) => root.join("real").join(format!("fit_{}", s)),
            }
        }
        Err(v2_err) => {
            let fit = FitToml::load(&fit_path).unwrap_or_else(|_| {
                eprintln!("error: {} is neither a v1 nor v2 fit.toml", fit_path);
                eprintln!("  v2 parse error: {}", v2_err);
                std::process::exit(1);
            });
            match seed {
                None => fit.fit_root(&fit_path).unwrap_or_else(|e| {
                    eprintln!("error: {}", e); std::process::exit(1);
                }),
                Some(s) => fit.cell_dir(&fit_path, s).unwrap_or_else(|e| {
                    eprintln!("error: {}", e); std::process::exit(1);
                }),
            }
        }
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
    // Bounds changed
    for name in a_est.intersection(&b_est) {
        let ab = a.estimate[*name].bounds;
        let bb = b.estimate[*name].bounds;
        if (ab.0 - bb.0).abs() > 1e-15 || (ab.1 - bb.1).abs() > 1e-15 {
            println!("  {}: bounds [{}, {}] → [{}, {}]", name, ab.0, ab.1, bb.0, bb.1);
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
/// Hardening proposal ship-now #9.
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
            eprintln!("error: --starts-from '{}': {}", raw, e);
            eprintln!("  Tip: pass a full path (e.g. results/fits/FOO/real/fit_1/scout)");
            eprintln!("  or a longer hash prefix.");
            std::process::exit(1);
        }
    }
}

