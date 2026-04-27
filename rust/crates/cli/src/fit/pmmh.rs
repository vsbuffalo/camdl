//! `camdl fit pmmh` — PMMH posterior sampling.
//!
//! Runs multiple MCMC chains in parallel, each using the bootstrap particle
//! filter as an unbiased likelihood estimator. Outputs per-chain trace files,
//! convergence diagnostics (R̂, ESS), and a summary JSON.

use crate::fit::config::FitToml;
use crate::fit::state::FitState;
use crate::fit::runner::{self, FitRunConfig};
use crate::cas::iso8601_utc;
use rayon::prelude::*;
use sim::inference::{
    if2::EstimatedParam,
    pmmh::{run_pmmh, Prior, PMMHConfig, PMMHResult, PMMHResumeState},
    diagnostic::{DiagnosticCollector, DiagnosticKind},
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_CHAINS: usize = 4;
const DEFAULT_STEPS: usize = 50_000;
const DEFAULT_PARTICLES: usize = 2000;
const DEFAULT_BURN_IN: usize = 5000;
const DEFAULT_THIN: usize = 10;
const DEFAULT_ADAPT_START: usize = 500;

pub fn run_pmmh_cli(
    fit: &FitToml,
    starts_from: Option<&str>,
    seed: u64,
    force: bool,
    check_variance: bool,
    resume: bool,
) -> Result<(), String> {
    eprintln!("\x1b[33m⚠ PMMH is experimental. For models with T > 500 observations,\x1b[0m");
    eprintln!("\x1b[33m  acceptance rates may be too low for reliable posterior sampling.\x1b[0m");
    eprintln!("\x1b[33m  Correlated pseudo-marginal (rho config) helps but has limits\x1b[0m");
    eprintln!("\x1b[33m  on discrete-state models. PGAS is planned for production use.\x1b[0m");
    eprintln!();

    let collector = DiagnosticCollector::new("pmmh");

    let stage_dir = format!("{}/pmmh", fit.fit.output_dir);
    let sc = fit.pmmh.as_ref();

    let n_chains = sc.and_then(|s| s.chains).unwrap_or(DEFAULT_CHAINS);
    let n_steps = sc.and_then(|s| s.steps).unwrap_or(DEFAULT_STEPS);
    let n_particles = sc.and_then(|s| s.particles).unwrap_or(DEFAULT_PARTICLES);
    let burn_in = sc.and_then(|s| s.burn_in).unwrap_or(DEFAULT_BURN_IN);
    let thin = sc.and_then(|s| s.thin).unwrap_or(DEFAULT_THIN);
    let adapt = sc.and_then(|s| s.adapt).unwrap_or(true);
    let adapt_start = sc.and_then(|s| s.adapt_start).unwrap_or(DEFAULT_ADAPT_START);

    // Load prior state if --starts-from provided
    let prior_state = starts_from.map(FitState::load).transpose()?;

    // Build FitRunConfig (reuse existing builder)
    let config = FitRunConfig::build(
        fit, prior_state.as_ref(),
        n_chains, n_particles, 1, // iterations=1 unused for PMMH
        1.0, // cooling unused
        seed, false,
    )?;

    let dt = config.if2_config.dt;

    // Build proposal SDs
    let proposal_sd = build_proposal_sd(&config, sc, starts_from)?;

    // Preflight: PF variance check
    eprintln!("\npfilter variance check ({} particles, 20 replicates)...", n_particles);
    let base = prior_state.as_ref().map(|s| {
        let mut p = config.base_params.clone();
        for spec in &config.estimated_params {
            if let Some(&v) = s.start_values.get(&spec.name) {
                p[spec.index] = v;
            }
        }
        p
    }).unwrap_or_else(|| config.base_params.clone());

    let logliks: Vec<f64> = (0..20)
        .map(|i| runner::run_quick_pfilter(&config, &base, n_particles, seed + i))
        .collect();
    let ll_mean = logliks.iter().sum::<f64>() / logliks.len() as f64;
    let ll_var = logliks.iter().map(|&l| (l - ll_mean).powi(2)).sum::<f64>() / (logliks.len() - 1) as f64;
    let ll_sd = ll_var.sqrt();

    eprintln!("  log L̂ mean = {:.1}, sd = {:.2}", ll_mean, ll_sd);
    if ll_sd > 5.0 {
        eprintln!("  \x1b[33m⚠ PF variance high (sd={:.1} > 5). Consider doubling particles to {}.\x1b[0m",
            ll_sd, n_particles * 2);
    } else if ll_sd < 0.5 && n_particles > 200 {
        eprintln!("  \x1b[32m✓ PF variance low (sd={:.2}). Could halve particles to {} for 2× speed.\x1b[0m",
            ll_sd, n_particles / 2);
    } else {
        eprintln!("  \x1b[32m✓ PF variance OK (target: 1-3)\x1b[0m");
    }

    if check_variance {
        // Also run correlation check if rho is set
        if let Some(rho) = sc.and_then(|c| c.rho) {
            eprintln!("\nCPM correlation check (rho={}, 50 correlated pairs)...", rho);
            let n_source_groups = config.compiled.source_groups.len();
            let n_obs = config.observations.len();
            let obs_spacing = if config.observations.len() >= 2 {
                config.observations[1].time - config.observations[0].time
            } else { 7.0 };
            let steps_per_obs = (obs_spacing / dt).round() as usize;
            let mut corr_rng = sim::rng::StatefulRng::new(seed + 999);

            let process = config.build_process();
            let obs_model_trait = config.build_obs_model();
            let smc_config = config.smc_config();

            let eval_corr = |params: &[f64], randoms: &sim::inference::correlated_pf::PFRandomState| -> f64 {
                match sim::inference::correlated_pf::bootstrap_filter_correlated(
                    &process, &obs_model_trait, params, &smc_config, randoms, seed,
                ) {
                    Ok(r) => r.log_likelihood,
                    Err(_) => f64::NEG_INFINITY,
                }
            };

            let mut ll_a = Vec::new();
            let mut ll_b = Vec::new();
            for i in 0..50 {
                let u = sim::inference::correlated_pf::PFRandomState::draw_fresh(
                    n_particles, n_obs, steps_per_obs, n_source_groups, &mut corr_rng,
                );
                let u_prime = u.correlate(rho, &mut corr_rng);
                let la = eval_corr(&base, &u);
                let lb = eval_corr(&base, &u_prime);
                ll_a.push(la);
                ll_b.push(lb);
                eprint!("\r  pair {}/50: Δ={:.2}    ", i + 1, (la - lb).abs());
            }
            eprintln!();

            // Compute correlation
            let n = ll_a.len() as f64;
            let mean_a = ll_a.iter().sum::<f64>() / n;
            let mean_b = ll_b.iter().sum::<f64>() / n;
            let var_a = ll_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n - 1.0);
            let var_b = ll_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n - 1.0);
            let cov = ll_a.iter().zip(&ll_b).map(|(&a, &b)| (a - mean_a) * (b - mean_b)).sum::<f64>() / (n - 1.0);
            let rho_eff = if var_a > 0.0 && var_b > 0.0 { cov / (var_a.sqrt() * var_b.sqrt()) } else { 0.0 };

            let diffs: Vec<f64> = ll_a.iter().zip(&ll_b).map(|(&a, &b)| a - b).collect();
            let diff_mean = diffs.iter().sum::<f64>() / n;
            let diff_sd = (diffs.iter().map(|&d| (d - diff_mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

            eprintln!("  ρ_eff = {:.3}", rho_eff);
            eprintln!("  sd(individual) = {:.2}", var_a.sqrt());
            eprintln!("  sd(difference) = {:.2}", diff_sd);
            if rho_eff > 0.95 {
                eprintln!("  \x1b[32m✓ CPM correlation excellent — mixing issue is proposal tuning\x1b[0m");
            } else if rho_eff > 0.8 {
                eprintln!("  \x1b[33m~ CPM correlation moderate — binomial correlation partial\x1b[0m");
            } else {
                eprintln!("  \x1b[31m✗ CPM correlation low — check binomial z-value injection\x1b[0m");
            }
        }

        eprintln!("\n--check-variance: stopping here (no MCMC run).");
        return Ok(());
    }

    if !force && !resume {
        let state_path = format!("{}/fit_state.toml", stage_dir);
        if std::path::Path::new(&state_path).exists() {
            eprintln!("\x1b[33mpmmh results already exist in {}. Use --force to re-run or --resume to continue.\x1b[0m", stage_dir);
            return Ok(());
        }
    }

    std::fs::create_dir_all(&stage_dir)
        .map_err(|e| format!("cannot create {}: {}", stage_dir, e))?;

    // Resolve priors: fit.toml override → model IR → Flat
    let priors: Vec<Prior> = config.estimated_params.iter()
        .map(|spec| super::runner::resolve_prior(&spec.name, fit, &config.model).0)
        .collect();

    // Active interventions + events — see same block in pgas.rs.
    crate::util::print_scheduled_actions_summary(&config.model_declared, &config.model);
    crate::util::print_observations_summary(&config.model);

    let dt = config.if2_config.dt;

    // Compute config hash — identifies the statistical problem.
    let config_hash = super::runner::compute_config_hash(fit, &config);

    // Load resume states if --resume
    let resume_states: Vec<Option<PMMHResumeState>> = if resume {
        let mut states = Vec::with_capacity(n_chains);
        let mut any_failed = false;
        for chain_id in 0..n_chains {
            let path = format!("{}/chain_{}/resume_state.bin", stage_dir, chain_id + 1);
            match std::fs::read(&path) {
                Ok(data) => match bincode::deserialize::<PMMHResumeState>(&data) {
                    Ok(state) => {
                        if state.config_hash != config_hash {
                            eprintln!("error: config hash mismatch for chain {} — \
                                model/data/priors have changed since the original run. \
                                Cannot resume. Re-run from scratch with --force.",
                                chain_id + 1);
                            std::process::exit(1);
                        }
                        eprintln!("  chain {}: resuming from step {}", chain_id + 1, state.completed_steps);
                        states.push(Some(state));
                    }
                    Err(e) => {
                        eprintln!("error: cannot deserialize resume state for chain {}: {}. \
                            Resume state format may have changed — re-run with --force.", chain_id + 1, e);
                        any_failed = true;
                        states.push(None);
                    }
                }
                Err(_) => {
                    eprintln!("error: no resume state file for chain {} ({})", chain_id + 1, path);
                    any_failed = true;
                    states.push(None);
                }
            }
        }
        if any_failed {
            eprintln!("error: --resume requires resume state files for all chains.");
            eprintln!("  These are written automatically at the end of every PMMH run.");
            eprintln!("  If the original run was interrupted before saving, use --force to start fresh.");
            std::process::exit(1);
        }
        states
    } else {
        vec![None; n_chains]
    };

    eprintln!("\npmmh: {} chains × {} steps × {} particles, burn_in={}, thin={}, adapt={}",
        n_chains, n_steps, n_particles, burn_in, thin, adapt);
    eprintln!("  proposal_sd (transformed): [{}]",
        config.estimated_params.iter().zip(&proposal_sd)
            .map(|(p, &sd)| format!("{}={:.4}", p.name, sd))
            .collect::<Vec<_>>().join(", "));

    let mp = indicatif::MultiProgress::with_draw_target(crate::progress::draw_target());
    let bar_style = indicatif::ProgressStyle::default_bar()
        .template("  chain {prefix} [{bar:25.cyan/dim}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("━╸─");

    let bars: Vec<indicatif::ProgressBar> = (0..n_chains).map(|chain_id| {
        let pb = mp.add(indicatif::ProgressBar::new(n_steps as u64));
        pb.set_style(bar_style.clone());
        pb.set_prefix(format!("{}", chain_id + 1));
        pb
    }).collect();

    // Pre-create chain directories
    for chain_id in 0..n_chains {
        let chain_dir = format!("{}/chain_{}", stage_dir, chain_id + 1);
        std::fs::create_dir_all(&chain_dir)
            .map_err(|e| format!("cannot create {}: {}", chain_dir, e))?;
    }

    let t0 = std::time::Instant::now();

    // Run chains in parallel
    let results: Vec<(usize, PMMHResult)> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let chain_seed = crate::util::derive_chain_seed(seed, chain_id);

            let pmmh_config = PMMHConfig {
                n_steps,
                n_particles,
                dt,
                proposal_sd: proposal_sd.clone(),
                adapt,
                adapt_start,
                thin,
                burn_in,
                rho: sc.and_then(|c| c.rho),
                n_source_groups: config.compiled.source_groups.len(),
            };

            // Build the loglik evaluator closure for this chain
            let eval_loglik = |params: &[f64], pf_seed: u64| -> f64 {
                runner::run_quick_pfilter(&config, params, n_particles, pf_seed)
            };

            // Correlated PF evaluator (when rho is set)
            let process = config.build_process();
            let obs_model_trait = config.build_obs_model();
            let smc_cfg = config.smc_config();
            let eval_correlated: Option<Box<dyn Fn(&[f64], &sim::inference::correlated_pf::PFRandomState) -> f64>> =
                if pmmh_config.rho.is_some() {
                    Some(Box::new(move |params: &[f64], randoms: &sim::inference::correlated_pf::PFRandomState| -> f64 {
                        match sim::inference::correlated_pf::bootstrap_filter_correlated(
                            &process, &obs_model_trait, params, &smc_cfg, randoms, chain_seed,
                        ) {
                            Ok(r) => r.log_likelihood,
                            Err(_) => f64::NEG_INFINITY,
                        }
                    }))
                } else {
                    None
                };

            let eval_corr_ref: Option<&dyn Fn(&[f64], &sim::inference::correlated_pf::PFRandomState) -> f64> =
                eval_correlated.as_deref();

            let bar = &bars[chain_id];
            let accepted_count = AtomicUsize::new(0);
            let chain_start = std::time::Instant::now();
            // Plain-mode progress emission (GH #14). Pretty and plain
            // modes are mutually exclusive at a point in time: under
            // pretty, the bar is live and the log line would be noise;
            // under plain, the bar is hidden and the log line is the
            // only signal the user gets.
            let plain = crate::progress::is_plain();
            let no_progress = crate::progress::is_none();

            // Streaming trace: use TraceWriter with append mode when resuming
            let chain_dir = format!("{}/pmmh/chain_{}", fit.fit.output_dir, chain_id + 1);
            let _ = std::fs::create_dir_all(&chain_dir);
            let trace_path = format!("{}/trace.tsv", chain_dir);
            let is_resuming = resume_states[chain_id].is_some();
            let param_names: Vec<String> = config.estimated_params.iter()
                .map(|s| s.name.clone()).collect();
            let trace_writer = super::trace_writer::TraceWriter::new(
                &trace_path, "step", &["accepted"],
                &param_names, is_resuming,
            );

            let progress_cb = |step: usize, loglik: f64, accepted: bool, params: &[f64]| {
                if accepted { accepted_count.fetch_add(1, Ordering::Relaxed); }
                let acc = accepted_count.load(Ordering::Relaxed) as f64 / (step + 1) as f64;

                // Stream trace row to disk (respecting burn-in/thin)
                if step >= burn_in && (step - burn_in).is_multiple_of(thin) {
                    let env = sim::inference::hierarchical::NamedParams {
                        names: &config.param_names,
                        values: params,
                    };
                    let log_prior: f64 = config.estimated_params.iter().zip(priors.iter())
                        .map(|(spec, prior)| {
                            let theta = params[spec.index];
                            let z = spec.to_transformed(theta);
                            prior.log_density_env(theta, z, &env)
                        })
                        .sum();
                    let log_posterior = loglik + log_prior;
                    let accepted_str = if accepted { "1" } else { "0" };
                    let param_vals: Vec<f64> = config.estimated_params.iter()
                        .map(|s| params[s.index]).collect();
                    trace_writer.write_row(
                        step, loglik, log_posterior,
                        &[accepted_str], &param_vals,
                    );
                }

                // Progress display (always, regardless of burn-in/thin)
                if step.is_multiple_of(100) || step == n_steps - 1 {
                    if no_progress {
                        // --progress none: suppress entirely.
                    } else if plain {
                        let elapsed = chain_start.elapsed().as_secs();
                        if loglik.is_finite() {
                            log::info!(
                                "pmmh chain {}: step {}/{} ({:.0}%) acc={:.0}% ll={:.1} elapsed={}s",
                                chain_id + 1, step, n_steps,
                                step as f64 / n_steps as f64 * 100.0,
                                acc * 100.0, loglik, elapsed);
                        } else {
                            log::info!(
                                "pmmh chain {}: step {}/{} ({:.0}%) acc={:.0}% ll=-inf elapsed={}s",
                                chain_id + 1, step, n_steps,
                                step as f64 / n_steps as f64 * 100.0,
                                acc * 100.0, elapsed);
                        }
                    } else {
                        bar.set_position(step as u64 + 1);
                        if loglik.is_finite() {
                            bar.set_message(format!("ll={:.1} acc={:.0}%", loglik, acc * 100.0));
                        } else {
                            bar.set_message(format!("ll=-inf acc={:.0}%", acc * 100.0));
                        }
                    }
                }
            };

            let result = run_pmmh(
                &config.estimated_params, &priors, &config.base_params,
                &config.param_names,
                &pmmh_config, &config.observations, &eval_loglik, eval_corr_ref, chain_seed,
                Some(&progress_cb), resume_states[chain_id].clone(), config_hash.clone(),
            );

            bar.finish_with_message(format!(
                "ll={:.1} acc={:.0}%", result.map_loglik, result.acceptance_rate * 100.0
            ));
            if plain {
                log::info!("pmmh chain {} done: MAP ll={:.1} acc={:.0}%",
                    chain_id + 1, result.map_loglik, result.acceptance_rate * 100.0);
            }

            // Save resume state for future --resume
            let resume_path = format!("{}/resume_state.bin", chain_dir);
            if let Ok(encoded) = bincode::serialize(&result.resume_state) {
                let _ = std::fs::write(&resume_path, encoded);
            }

            (chain_id, result)
        })
        .collect();

    let elapsed = t0.elapsed();

    // Traces already written by streaming callback — no post-hoc write needed.

    // Compute diagnostics
    let diagnostics = compute_diagnostics(&results, &config.estimated_params);

    // Report
    eprintln!("\nacceptance rates:");
    for (chain_id, result) in &results {
        let status = if result.acceptance_rate < 0.10 {
            "\x1b[31m✗ too low\x1b[0m"
        } else if result.acceptance_rate > 0.50 {
            "\x1b[33m~ high\x1b[0m"
        } else {
            "\x1b[32m✓\x1b[0m"
        };
        eprintln!("  chain {}: {:.1}% {}", chain_id + 1, result.acceptance_rate * 100.0, status);
        if result.acceptance_rate < 0.10 || result.acceptance_rate > 0.50 {
            collector.push(DiagnosticKind::AcceptanceRateUnhealthy {
                rate: result.acceptance_rate, param: None,
            });
        }
    }

    if n_chains > 1 {
        eprintln!("\nRhat:");
        for spec in &config.estimated_params {
            if let Some(&rhat) = diagnostics.rhat.get(&spec.name) {
                let status = if rhat < 1.1 { "\x1b[32m✓\x1b[0m" } else if rhat < 1.5 { "\x1b[33m~\x1b[0m" } else { "\x1b[31m✗\x1b[0m" };
                let ess = diagnostics.ess.get(&spec.name).copied().unwrap_or(0.0);
                eprintln!("  {:12} Rhat={:.3} {} ESS={:.0}", spec.name, rhat, status, ess);
                if rhat > 1.1 {
                    collector.push(DiagnosticKind::RhatHigh {
                        param: spec.name.clone(), rhat, threshold: 1.1,
                    });
                }
            }
        }
    }

    // Find MAP across all chains
    let (map_chain, map_result) = results.iter()
        .max_by(|a, b| a.1.map_log_posterior.total_cmp(&b.1.map_log_posterior))
        .unwrap();

    // Write summary JSON
    write_summary(&stage_dir, &results, &config, &diagnostics)?;

    // Write fit_state.toml
    let mut start_values = HashMap::new();
    for spec in config.estimated_params.iter() {
        start_values.insert(spec.name.clone(), map_result.map_params[spec.index]);
    }
    // Include fixed params too
    for p in &config.model.parameters {
        if !start_values.contains_key(&p.name) {
            if let Some(&idx) = config.compiled.param_index.get(p.name.as_str()) {
                start_values.insert(p.name.clone(), config.base_params[idx]);
            }
        }
    }

    let state = FitState {
        stage: "pmmh".into(),
        seed,
        timestamp: iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik: map_result.map_loglik,
        initial_loglik: ll_mean,
        best_chain: *map_chain,
        n_chains,
        n_good_chains: None,
        start_values,
        rw_sd: HashMap::new(),
        loglik_type: Some("marginal".into()),
        acceptance_rate: Some(map_result.acceptance_rate),
        // Bayesian stages don't produce an IF2-style Â table; the
        // refine-gates proposal only gates on scout→refine handoffs.
        tail_chain_agreement: HashMap::new(),
        ivp_params: Vec::new(),
        chain_logliks: Vec::new(),
        chain_clean_logliks: Vec::new(),
        chain_clean_ses: Vec::new(),
        soft_warn_params: Vec::new(),
        // Bayesian path — compound gate doesn't apply to PMMH.
        resolved_gate: None,
        resolved_clean_eval: None,
    };
    state.save(&stage_dir)?;

    // Write draws.tsv: complete-M posterior draws (all params, estimated + fixed)
    // Reads the per-chain trace.tsv files (already burn-in/thin filtered) and
    // adds fixed parameter columns.
    {
        use std::io::Write;
        let draws_path = format!("{}/draws.tsv", stage_dir);
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&draws_path)
                .map_err(|e| format!("cannot create {}: {}", draws_path, e))?
        );

        // Header: estimated params + fixed params
        let est_names: Vec<String> = config.estimated_params.iter()
            .map(|s| s.name.clone()).collect();
        let fixed_names: Vec<String> = config.model.parameters.iter()
            .filter(|p| !config.estimated_params.iter().any(|e| e.name == p.name))
            .map(|p| p.name.clone())
            .collect();
        let mut all_names = est_names.clone();
        all_names.extend(fixed_names.iter().cloned());
        writeln!(f, "{}", all_names.join("\t")).unwrap();

        let fixed_vals: Vec<f64> = fixed_names.iter().map(|name| {
            config.compiled.param_index.get(name.as_str())
                .map(|&idx| config.base_params[idx])
                .unwrap_or(0.0)
        }).collect();

        // Read each chain's trace.tsv and extract param columns
        let mut n_draws = 0usize;
        for chain_id in 0..n_chains {
            let trace_path = format!("{}/chain_{}/trace.tsv", stage_dir, chain_id + 1);
            if let Ok(content) = std::fs::read_to_string(&trace_path) {
                let mut lines = content.lines();
                let header = lines.next().unwrap_or("");
                let cols: Vec<&str> = header.split('\t').collect();
                // Find column indices for estimated params
                let param_col_indices: Vec<usize> = est_names.iter().map(|name| {
                    cols.iter().position(|c| c == name).unwrap_or(usize::MAX)
                }).collect();

                for line in lines {
                    if line.trim().is_empty() { continue; }
                    let fields: Vec<&str> = line.split('\t').collect();
                    let mut vals: Vec<String> = param_col_indices.iter().map(|&col_idx| {
                        if col_idx < fields.len() {
                            fields[col_idx].to_string()
                        } else {
                            eprintln!("warning: trace.tsv missing column at index {}", col_idx);
                            "NaN".to_string()
                        }
                    }).collect();
                    vals.extend(fixed_vals.iter().map(|v| format!("{:.17e}", v)));
                    writeln!(f, "{}", vals.join("\t")).unwrap();
                    n_draws += 1;
                }
            }
        }
        drop(f);
        eprintln!("  draws.tsv: {} posterior samples (all {} params)", n_draws, all_names.len());
    }

    // Render and persist diagnostics
    collector.render_to_stderr();
    let _ = collector.write_json(&format!("{}/diagnostics.json", stage_dir));

    let wall_secs = elapsed.as_secs_f64();
    let total_pf_calls = n_chains * n_steps;
    eprintln!("\npmmh complete in {:.1}s ({} PF evaluations, {:.1}ms/eval): {}/",
        wall_secs, total_pf_calls,
        wall_secs * 1000.0 / total_pf_calls as f64 * n_chains as f64,
        stage_dir);
    eprintln!("  MAP loglik: {:.1} (chain {})", map_result.map_loglik, map_chain + 1);

    Ok(())
}

/// Build proposal SDs on the transformed scale.
fn build_proposal_sd(
    config: &FitRunConfig,
    sc: Option<&crate::fit::config::PMMHSampleConfig>,
    starts_from: Option<&str>,
) -> Result<Vec<f64>, String> {
    // Try to load scout chain endpoints for empirical covariance
    let proposal_dir = sc.and_then(|s| s.proposal_from.as_deref()).or(starts_from);
    if let Some(dir) = proposal_dir {
        if let Ok(sds) = load_scout_proposal_sd(dir, &config.estimated_params) {
            eprintln!("  proposal_sd seeded from chain spread in {}/", dir);
            return Ok(sds);
        }
    }

    // Fallback: use rw_sd from [estimate], scaled up for MH jumps
    // IF2 rw_sd is per-perturbation-step; PMMH needs per-proposal (larger)
    Ok(config.estimated_params.iter().map(|p| {
        p.transformed_sd(p.rw_sd, p.initial) * 5.0
    }).collect())
}

/// Load chain endpoint parameters from a prior stage and compute
/// empirical SD on the transformed scale. Scale by 2.38/√d (optimal RWM).
fn load_scout_proposal_sd(dir: &str, if2_params: &[EstimatedParam]) -> Result<Vec<f64>, String> {
    // Find chain directories
    let mut chain_dirs: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir, e))? {
        let entry = entry.map_err(|e| format!("{}", e))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("chain_") && entry.path().is_dir() {
            chain_dirs.push(entry.path().to_string_lossy().to_string());
        }
    }
    if chain_dirs.len() < 2 {
        return Err("need at least 2 chains for empirical covariance".into());
    }

    // Read final params from each chain
    let d = if2_params.len();
    let mut transformed_endpoints: Vec<Vec<f64>> = Vec::new();

    for chain_dir in &chain_dirs {
        let toml_path = format!("{}/final_params.toml", chain_dir);
        let contents = std::fs::read_to_string(&toml_path)
            .map_err(|e| format!("{}: {}", toml_path, e))?;
        let parsed: HashMap<String, toml::Value> = toml::from_str(&contents)
            .map_err(|e| format!("{}: {}", toml_path, e))?;

        let mut z = Vec::with_capacity(d);
        for spec in if2_params {
            let v = parsed.get(&spec.name)
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
                .ok_or_else(|| format!("missing {} in {}", spec.name, toml_path))?;
            z.push(spec.to_transformed(v));
        }
        transformed_endpoints.push(z);
    }

    // Compute per-parameter SD on transformed scale
    let n = transformed_endpoints.len() as f64;
    let scale = 2.38 / (d as f64).sqrt();

    let sds: Vec<f64> = (0..d).map(|i| {
        let mean = transformed_endpoints.iter().map(|z| z[i]).sum::<f64>() / n;
        let var = transformed_endpoints.iter().map(|z| (z[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (var.sqrt() * scale).max(0.01) // floor to prevent zero proposal
    }).collect();

    Ok(sds)
}

struct Diagnostics {
    rhat: HashMap<String, f64>,
    ess: HashMap<String, f64>,
}

fn compute_diagnostics(
    results: &[(usize, PMMHResult)],
    estimated_params: &[EstimatedParam],
) -> Diagnostics {
    let mut rhat_map = HashMap::new();
    let mut ess_map = HashMap::new();

    for spec in estimated_params {
        let chains: Vec<Vec<f64>> = results.iter()
            .map(|(_, r)| r.steps.iter().map(|s| s.params[spec.index]).collect())
            .collect();

        let (rhat, ess) = super::runner::compute_rhat_ess(&chains);
        if rhat.is_finite() {
            rhat_map.insert(spec.name.clone(), rhat);
        }
        ess_map.insert(spec.name.clone(), ess);
    }

    Diagnostics { rhat: rhat_map, ess: ess_map }
}

// write_chain_traces removed — streaming callback now handles trace output
// with correct log_posterior and burn-in/thin filtering.

fn write_summary(
    dir: &str,
    results: &[(usize, PMMHResult)],
    config: &FitRunConfig,
    diagnostics: &Diagnostics,
) -> Result<(), String> {
    let acceptance_rates: Vec<f64> = results.iter().map(|(_, r)| r.acceptance_rate).collect();

    let (map_chain, map_result) = results.iter()
        .max_by(|a, b| a.1.map_log_posterior.total_cmp(&b.1.map_log_posterior))
        .unwrap();

    let map_params: HashMap<String, f64> = config.estimated_params.iter()
        .map(|spec| (spec.name.clone(), map_result.map_params[spec.index]))
        .collect();

    let summary = serde_json::json!({
        "stage": "pmmh",
        "n_chains": results.len(),
        "steps_per_chain": results.first().map(|(_, r)| r.n_steps).unwrap_or(0),
        "acceptance_rate": acceptance_rates,
        "rhat": diagnostics.rhat,
        "ess": diagnostics.ess,
        "map_loglik": map_result.map_loglik,
        "map_chain": map_chain + 1,
        "map_params": map_params,
    });

    let path = format!("{}/pmmh_summary.json", dir);
    let contents = serde_json::to_string_pretty(&summary)
        .map_err(|e| format!("json error: {}", e))?;
    std::fs::write(&path, contents)
        .map_err(|e| format!("cannot write {}: {}", path, e))
}
