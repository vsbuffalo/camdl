//! `camdl if2` — iterated filtering for maximum likelihood estimation.
//!
//! Usage:
//!   camdl if2 MODEL --params P.toml --data cases.tsv \
//!       --rw-sd "R0=5,sigma=0.01" --particles 2000 --iterations 100 \
//!       [--chains 4] [--regime scout|refine|validate] \
//!       [--cooling 0.95] [--dt 1.0] [--seed 1] [--parallel 4]
//!
//! Regimes set sensible defaults (authoritative constants live in
//! rust/crates/cli/src/fit/{scout,refine}.rs; camdl if2 --regime uses
//! the same values). Cooling fractions follow pomp's cf50 convention —
//! see docs/methods/cooling.md for the worked formula.
//!   scout:    8 chains, 500 particles,  30 iters, cooling=0.70  (mild — find basins)
//!   refine:   4 chains, 1000 particles, 50 iters, cooling=0.05  (aggressive — converge to MLE)
//!   validate: 4 chains, 5000 particles, 100 iters, cooling=0.05  (aggressive — final polish)
//!
//! Output: parameter traces TSV to stdout, diagnostics + Â (chain agreement) to stderr.
//!   With --output-dir: writes per-chain traces + summary JSON.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2_with_progress, IF2Config, EstimatedParam, IF2Result, Observation, Transform},
        ParticleState,
        ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
        traits::{SMCConfig, ObservationModel},
    },
};
use std::collections::HashMap;
use std::sync::Arc;

fn run_one_chain(
    chain_id: usize,
    process: &ChainBinomialProcess,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    params: &[f64],
    if2_params: &[EstimatedParam],
    config: &IF2Config,
    base_seed: u64,
    pb: Option<&ProgressBar>,
) -> IF2Result {
    let chain_seed = crate::util::derive_chain_seed(base_seed, chain_id);
    let n_iter = config.n_iterations;
    let plain = crate::progress::is_plain();
    // RefCell so the progress closure can be `Fn`: run_if2_with_progress
    // takes `&dyn Fn`. The callback is single-threaded per chain, so
    // interior mutation is safe. Cadence from progress::DEFAULT_THROTTLE.
    let throttle = std::cell::RefCell::new(crate::progress::Throttle::default());

    let progress_cb = |iter: usize, loglik: f64| {
        if let Some(bar) = pb {
            bar.set_position((iter + 1) as u64);
            if loglik.is_finite() {
                bar.set_message(format!("ll={:.1}", loglik));
            } else {
                bar.set_message("ll=-inf".to_string());
            }
        }
        if plain && throttle.borrow_mut().ready() {
            if loglik.is_finite() {
                log::info!("if2 chain {} iter {}/{} ll={:.1}",
                    chain_id + 1, iter + 1, n_iter, loglik);
            } else {
                log::info!("if2 chain {} iter {}/{} ll=-inf",
                    chain_id + 1, iter + 1, n_iter);
            }
        }
    };

    let result = run_if2_with_progress(
        process, obs_model, params, if2_params, config, chain_seed,
        Some(&progress_cb),
    ).unwrap_or_else(|e| {
        eprintln!("chain {} error: {:?}", chain_id + 1, e);
        std::process::exit(1);
    });

    if let Some(bar) = pb {
        bar.finish_with_message(format!("ll={:.1}", result.final_loglik));
    }
    if plain {
        log::info!("if2 chain {} done iter {}/{} final_ll={:.1}",
            chain_id + 1, n_iter, n_iter, result.final_loglik);
    }

    result
}

pub fn cmd_if2(a: &crate::args::If2Args) {
    eprintln!(
        "[note] 'camdl if2' uses the legacy IF2 argmax estimator and does not \
         apply clean-eval de-biasing. For production parameter estimates, \
         use 'camdl fit run' instead."
    );
    let ir_path = a.model.to_string_lossy().into_owned();
    let data_path = a.data.to_string_lossy().into_owned();
    let dt = a.inference.dt;
    let seed = a.inference.seed;
    let parallel = a.inference.parallel;
    let output_path: Option<String> = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
    let output_dir: Option<String> = a.output_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let trace_path: Option<String> = a.trace.as_ref().map(|p| p.to_string_lossy().into_owned());
    let scenario_name = a.scenario.scenario.clone();
    let _adhoc_enable = a.scenario.enable.clone();
    let flow_name = a.flow.flow.clone();
    let overrides: HashMap<String, f64> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value))
        .collect();

    // Apply regime defaults, then fill in from explicit flags
    let mut n_chains = a.chains.unwrap_or(1);
    let mut n_particles = a.inference.particles;
    let mut n_iterations = a.iterations;
    let mut cooling = a.cooling;

    if let Some(ref r) = a.regime {
        match r.as_str() {
            "scout" => {
                if n_chains == 1 { n_chains = 8; }
                n_particles = if a.inference.particles == 0 { 500 } else { n_particles };
                n_iterations = n_iterations.or(Some(30));
                cooling = cooling.or(Some(0.70));
            }
            "refine" => {
                if n_chains == 1 { n_chains = 4; }
                n_particles = if a.inference.particles == 0 { 1000 } else { n_particles };
                n_iterations = n_iterations.or(Some(50));
                cooling = cooling.or(Some(0.05));
            }
            "validate" => {
                if n_chains == 1 { n_chains = 4; }
                n_particles = if a.inference.particles == 0 { 5000 } else { n_particles };
                n_iterations = n_iterations.or(Some(100));
                cooling = cooling.or(Some(0.05));
            }
            other => {
                eprintln!("unknown regime '{}'. Use scout, refine, or validate.", other);
                std::process::exit(1);
            }
        }
    }

    let n_particles = if n_particles == 0 { 2000 } else { n_particles };
    let n_iterations = n_iterations.unwrap_or(100);
    let cooling = cooling.unwrap_or(0.95);

    let rw_sd = a.rw_sd.as_ref().unwrap_or_else(|| {
        eprintln!("error: --rw-sd required (e.g., --rw-sd \"R0=5,sigma=0.01\" or --rw-sd auto)");
        std::process::exit(1);
    });

    let rw_sd_auto = matches!(rw_sd, crate::args::types::RwSd::Auto);
    let rw_sd_map: HashMap<String, Option<f64>> = match rw_sd {
        crate::args::types::RwSd::Auto => HashMap::new(),
        crate::args::types::RwSd::Map(m) => m.clone(),
    };

    let ivp_set: std::collections::HashSet<String> = a.ivp.iter().cloned().collect();
    let fixed_set: std::collections::HashSet<String> = a.fixed.iter().cloned().collect();

    // Load model
    let (mut model, _model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    for pf in &a.model_overrides.params {
        crate::util::apply_params_file(&mut model, &pf.to_string_lossy())
            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    }

    if let Some(ref name) = scenario_name {
        if let Some(preset) = model.presets.iter().find(|p| p.name == *name) {
            for p in &mut model.parameters {
                if let Some(&v) = preset.params.get(&p.name) { p.value = Some(v); }
            }
        }
    }

    for p in &mut model.parameters {
        if let Some(&v) = overrides.get(&p.name) { p.value = Some(v); }
    }

    let compiled = CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("compile error: {:?}", e); std::process::exit(1); });
    let params = compiled.default_params.clone();

    // Load data
    let observations: Vec<Observation> = crate::pfilter::load_data_tsv_pub(&data_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
        .into_iter()
        .map(|o| Observation { time: o.time, value: o.value })
        .collect();

    let flow_indices = crate::util::resolve_flow_indices(&model, flow_name.as_deref())
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // Build EstimatedParam specs from --rw-sd
    //
    // Two modes:
    //   Explicit: --rw-sd "R0=5,sigma=0.01" → only named params estimated.
    //     The rw_sd list IS the partition. --fixed is ignored.
    //   Auto:     --rw-sd auto → all params estimated unless in --fixed.
    //     Uses bounds-based heuristic: (hi-lo)/6 on transformed scale.
    //     --fixed carves out non-estimable params (N0, mu, k, etc.).
    //     Per-param auto: --rw-sd "R0=5,sigma=auto" mixes explicit + auto.
    let param_names_to_estimate: Vec<String> = if rw_sd_auto {
        let names: Vec<String> = model.parameters.iter()
            .filter(|p| !fixed_set.contains(&p.name))
            .map(|p| p.name.clone())
            .collect();

        // Warn if estimating everything (no --fixed with auto)
        if fixed_set.is_empty() && names.len() > 4 {
            eprintln!("\x1b[33mwarning: --rw-sd auto is estimating ALL {} parameters.\x1b[0m", names.len());
            eprintln!("  Use --fixed to exclude parameters that should be held constant.");
            eprintln!("  Estimating: {}", names.join(", "));
            eprintln!("  Example: --fixed \"N0,mu,k\"");
            eprintln!("  For calibrated rw_sd, use 'camdl fit' instead.");
            eprintln!();
        }
        names
    } else {
        // Explicit mode: --fixed is ignored, rw_sd list is the partition
        rw_sd_map.keys().cloned().collect()
    };

    let specs: Vec<crate::fit::runner::ParamSpec> = param_names_to_estimate.iter().map(|name| {
        crate::fit::runner::ParamSpec {
            name: name.clone(),
            rw_sd: rw_sd_map.get(name).and_then(|v| *v),
            transform: None,
            ivp: ivp_set.contains(name),
        }
    }).collect();
    let any_auto = specs.iter().any(|s| s.rw_sd.is_none());

    let if2_params = crate::fit::runner::build_if2_params_from_specs(
        &model, &compiled, &params, &specs,
    ).unwrap_or_else(|e| {
        eprintln!("error: {}", e); std::process::exit(1);
    });

    // Report auto rw_sd values
    if any_auto || rw_sd_auto {
        eprintln!("if2: auto rw_sd (heuristic from bounds — use 'camdl fit' for calibrated values):");
        for spec in &if2_params {
            let explicit = rw_sd_map.get(&spec.name).and_then(|v| *v);
            let source = if explicit.is_some() { "explicit" } else { "auto" };
            let transform_name = match &spec.transform {
                Transform::Log { .. } => "log",
                Transform::Logit { .. } => "logit",
                Transform::None => "none",
            };
            let bounds_str = if spec.lower.is_finite() && spec.upper.is_finite() {
                format!("[{}, {}]", spec.lower, spec.upper)
            } else if spec.lower.is_finite() {
                format!("[{}, ∞)", spec.lower)
            } else {
                "unbounded".to_string()
            };
            eprintln!("  {:12} rw_sd={:<10.4} ({}, {}, {})", spec.name, spec.rw_sd, transform_name, bounds_str, source);
        }
    }

    let n_fixed = model.parameters.len() - if2_params.len();
    let regime_name = a.regime.as_deref().unwrap_or("manual");
    eprintln!("if2: {} observations, {} chains × {} particles × {} iterations, cooling={}, dt={}, seed={}",
        observations.len(), n_chains, n_particles, n_iterations, cooling, dt, seed);
    let effective_threads = if parallel > 0 { parallel } else { rayon::current_num_threads() };
    eprintln!("if2: regime={}, estimating {} parameters, {} fixed, threads={}",
        regime_name, if2_params.len(), n_fixed, effective_threads);

    // Parameter scale diagnostics
    for spec in &if2_params {
        let value = params[spec.index];
        if value.abs() < 1e-10 { continue; }
        let ratio = spec.rw_sd / value.abs();
        if ratio > 0.5 {
            eprintln!("warning: rw_sd for '{}' is {:.0}% of its value ({:.4}) — \
                consider reducing --rw-sd {}={:.4}",
                spec.name, ratio * 100.0, value, spec.name, value.abs() * 0.1);
        }
        if ratio < 0.001 {
            eprintln!("warning: rw_sd for '{}' is {:.2}% of its value ({:.4}) — \
                consider increasing --rw-sd {}={:.4}",
                spec.name, ratio * 100.0, value, spec.name, value.abs() * 0.05);
        }
    }

    let compiled = Arc::new(compiled);

    let config = IF2Config {
        n_particles,
        n_iterations,
        cooling_fraction: cooling,
        cooling_target_iters: n_iterations, simplex_groups: vec![],
        dt,
        t_start: compiled.model.simulation.t_start,
        // Legacy `camdl fit if2` subcommand doesn't surface ic_free.
        skip_first_obs_from_loglik: false,
    };

    // Build process + observation model via traits
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_times: Vec<f64> = observations.iter().map(|o| o.time).collect();
    let obs_values: Vec<f64> = observations.iter().map(|o| o.value).collect();

    let obs_model_obj: Box<dyn ObservationModel<ParticleState> + Sync> = if let Some(obs_block) = model.observations.first() {
        eprintln!("if2: using observation model '{}' from IR", obs_block.name);
        // `--flow` forces incidence over a specific transition; otherwise
        // derive the projection (incidence / prevalence / snapshot) from
        // the observation block.
        let projection = if flow_name.is_some() {
            sim::inference::multi_stream_obs::StreamProjection::FlowSum(flow_indices.clone())
        } else {
            sim::inference::multi_stream_obs::StreamProjection::from_ir(
                &obs_block.projection, &compiled, &obs_block.name,
            ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
        };
        Box::new(MultiStreamObsModel::new(
            vec![StreamSpec {
                projection,
                ir_model: obs_block.clone(),
                observations: obs_values,
                obs_times,
            }],
            compiled.clone(),
        ).unwrap_or_else(|e| {
            eprintln!("error: observation model construction failed: {:?}", e);
            std::process::exit(1);
        }))
    } else {
        eprintln!("error: model has no observations block");
        std::process::exit(1);
    };

    let if2_params = Arc::new(if2_params);

    // ── Multi-chain execution with indicatif progress ──────────────────────
    // GH #14: draw target reflects `--progress` mode. In plain/none modes
    // the bars are hidden; per-chain log::info! lines emitted from the
    // progress callback provide agent-/tee-/ssh-friendly progress.
    let mp = MultiProgress::with_draw_target(crate::progress::draw_target());
    let bar_style = ProgressStyle::default_bar()
        .template("  chain {prefix} [{bar:25.cyan/dim}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("━╸─");

    let bars: Vec<ProgressBar> = (0..n_chains).map(|chain_id| {
        let pb = mp.add(ProgressBar::new(n_iterations as u64));
        pb.set_style(bar_style.clone());
        pb.set_prefix(format!("{}", chain_id + 1));
        pb
    }).collect();

    // Initialize rayon global pool (controls all parallelism: chains + particles).
    // parallel=0 means use rayon default (num_cpus).
    if parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build_global();
    }

    let chain_results: Vec<(usize, IF2Result)> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let result = run_one_chain(chain_id, &process, &*obs_model_obj,
                &params, &if2_params, &config,
                seed, Some(&bars[chain_id]));
            (chain_id, result)
        })
        .collect();

    // ── Evaluate true loglik (clean PF) at selected iterations ─────────
    let mut chain_results = chain_results;
    {
        let eval_interval = 10;
        let n_eval_particles = n_particles.min(500);
        let smc_config = SMCConfig {
            n_particles: n_eval_particles,
            dt,
            t_start: compiled.model.simulation.t_start,
            skip_first_obs_from_loglik: false,
            record_ancestry: false,
            record_prequential: false,
        };

        eprintln!("\nevaluating loglik (every {} iterations, all {} chains)...", eval_interval, n_chains);
        for (chain_id, result) in chain_results.iter_mut() {
            for it in &mut result.iterations {
                if it.iteration % eval_interval == 0 || it.iteration == n_iterations - 1 {
                    let eval_params = &it.param_means;
                    let pf_seed = seed + *chain_id as u64 * 1000 + it.iteration as u64;
                    match sim::inference::bootstrap_filter(
                        &process, &*obs_model_obj, eval_params, &smc_config, pf_seed,
                    ) {
                        Ok(r) => it.loglik = r.log_likelihood,
                        Err(_) => it.loglik = f64::NEG_INFINITY,
                    }
                }
            }
            let true_ll = result.iterations.last()
                .map(|it| it.loglik).unwrap_or(f64::NEG_INFINITY);
            result.final_loglik = true_ll;
            eprint!("\r  chain {}: ll={:.1}    ", *chain_id + 1, true_ll);
        }
        eprintln!();
    }

    // ── Compute Â (chain agreement) across chains ──────────────────────────
    let chain_agreement = crate::fit::runner::compute_chain_agreement(&chain_results, &if2_params, n_iterations);
    if n_chains > 1 {
        let n_tail = (n_iterations / 2).max(1);
        eprintln!("\nÂ (across {} chains, last {} iterations):", n_chains, n_tail);
        for spec in if2_params.iter() {
            if let Some(&r) = chain_agreement.get(&spec.name) {
                let status = if r < 1.1 { "✓" } else if r < 1.5 { "~" } else { "✗" };
                eprintln!("  {:12} Â={:.2} {}", spec.name, r, status);
            }
        }
    }

    // ── Find best chain ──────────────────────────────────────────────────
    let best = chain_results.iter()
        .max_by(|a, b| a.1.final_loglik.total_cmp(&b.1.final_loglik))
        .unwrap();

    eprintln!("\nBest chain: {} (loglik={:.2})", best.0 + 1, best.1.final_loglik);
    if n_chains > 1 {
        let logliks: Vec<f64> = chain_results.iter().map(|(_, r)| r.final_loglik).collect();
        eprintln!("Chain logliks: [{}]",
            logliks.iter().map(|l| format!("{:.1}", l)).collect::<Vec<_>>().join(", "));
        let spread = logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - logliks.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!("Loglik spread: {:.1}", spread);
    }

    // ── Output: combined traces ────────────────────────────────────────
    {
        use std::io::Write as _;
        let mut out: Box<dyn std::io::Write> = match &output_path {
            Some(path) => {
                let f = std::fs::File::create(path)
                    .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
                Box::new(std::io::BufWriter::new(f))
            }
            None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
        };

        write!(out, "chain\titeration\tif2_perturbed_loglik").unwrap();
        for spec in if2_params.iter() {
            write!(out, "\t{}", spec.name).unwrap();
        }
        writeln!(out).unwrap();

        for (chain_id, result) in &chain_results {
            for iter_result in &result.iterations {
                write!(out, "{}\t{}\t{:.2}", chain_id + 1, iter_result.iteration, iter_result.if2_perturbed_loglik).unwrap();
                for spec in if2_params.iter() {
                    write!(out, "\t{:.6}", iter_result.param_means[spec.index]).unwrap();
                }
                writeln!(out).unwrap();
            }
        }

        if let Some(ref path) = output_path {
            eprintln!("traces written to {}", path);
        }
    }

    // ── Write long-format trace with diagnostics ─────────────────────────
    if let Some(ref path) = trace_path {
        use std::io::Write as _;
        let mut f = std::fs::File::create(path)
            .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
        writeln!(f, "chain\titeration\tparam\tvalue\trw_sd_eff\twvr\tq\tclamp\tloglik\tif2_perturbed_loglik").unwrap();
        for (chain_id, result) in &chain_results {
            for it in &result.iterations {
                for (pi, spec) in if2_params.iter().enumerate() {
                    let diag = it.param_diag.get(pi);
                    let wvr = diag.map(|d| d.weighted_var_ratio).unwrap_or(f64::NAN);
                    let q = diag.map(|d| d.q_ratio).unwrap_or(f64::NAN);
                    let eff_rw = diag.map(|d| d.effective_rw_sd).unwrap_or(f64::NAN);
                    let clamp = diag.map(|d| d.clamp_fraction).unwrap_or(0.0);
                    let loglik_str = if it.loglik.is_nan() { "NA".to_string() }
                        else { format!("{:.2}", it.loglik) };
                    writeln!(f, "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.2}",
                        chain_id + 1, it.iteration, spec.name,
                        it.param_means[spec.index], eff_rw, wvr, q, clamp,
                        loglik_str, it.if2_perturbed_loglik).unwrap();
                }
            }
        }
        eprintln!("trace diagnostics written to {}", path);
    }

    // ── Final MLE from best chain ────────────────────────────────────────
    eprintln!("\nMLE estimates (best chain):");
    for spec in if2_params.iter() {
        eprintln!("  {} = {:.6}", spec.name, best.1.mle[spec.index]);
    }
    eprintln!("  loglik = {:.2}", best.1.final_loglik);

    // ── Write output dir if requested ────────────────────────────────────
    if let Some(ref dir) = output_dir {
        std::fs::create_dir_all(dir).ok();
        for (chain_id, result) in &chain_results {
            let chain_dir = format!("{}/chain_{}", dir, chain_id + 1);
            std::fs::create_dir_all(&chain_dir).ok();

            // Parameter traces
            let trace_path = format!("{}/parameter_traces.tsv", chain_dir);
            let mut f = std::fs::File::create(&trace_path).unwrap();
            use std::io::Write;
            write!(f, "iteration\tloglik\tif2_perturbed_loglik").unwrap();
            for spec in if2_params.iter() { write!(f, "\t{}", spec.name).unwrap(); }
            writeln!(f).unwrap();
            for it in &result.iterations {
                let loglik_str = if it.loglik.is_finite() { format!("{:.2}", it.loglik) } else { "NA".into() };
                write!(f, "{}\t{}\t{:.2}", it.iteration, loglik_str, it.if2_perturbed_loglik).unwrap();
                for spec in if2_params.iter() { write!(f, "\t{:.6}", it.param_means[spec.index]).unwrap(); }
                writeln!(f).unwrap();
            }

            // Final params TOML
            let toml_path = format!("{}/final_params.toml", chain_dir);
            let mut f = std::fs::File::create(&toml_path).unwrap();
            for spec in if2_params.iter() {
                writeln!(f, "{} = {}", spec.name, result.mle[spec.index]).unwrap();
            }
        }
        eprintln!("Output written to {}/", dir);
    }
}
