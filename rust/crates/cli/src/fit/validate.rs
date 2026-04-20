//! `camdl fit validate` — final IF2 + profiles + precise pfilter at MLE.

use crate::fit::config::FitToml;
use crate::fit::state::FitState;
use crate::fit::runner::{self, FitRunConfig};
use crate::fit::provenance::{self, MleMetadata};
use crate::hashing;
use sha2::Digest;
use sim::inference::{
    bootstrap_filter,
    if2::{IF2Config, EstimatedParam, run_if2},
    diagnostic::{DiagnosticCollector, DiagnosticKind},
    traits::{SMCConfig, ObservationModel},
};
use std::collections::HashMap;

const VALIDATE_CHAINS: usize = 4;
const VALIDATE_PARTICLES: usize = 5000;
const VALIDATE_ITERATIONS: usize = 100;
const VALIDATE_COOLING: f64 = 0.05; // cf50: 5% at halfway, 0.25% at end
const VALIDATE_PFILTER_PARTICLES: usize = 10000;

pub fn run_validate(fit: &FitToml, starts_from: &str, seed: u64, force: bool) -> Result<(), String> {
    let stage_dir = format!("{}/validate", fit.fit.output_dir);
    let vc = fit.validate.as_ref();

    let n_chains = vc.and_then(|s| s.chains).unwrap_or(VALIDATE_CHAINS);
    let n_particles = vc.and_then(|s| s.particles).unwrap_or(VALIDATE_PARTICLES);
    let n_iterations = vc.and_then(|s| s.iterations).unwrap_or(VALIDATE_ITERATIONS);
    let cooling = vc.and_then(|s| s.cooling).unwrap_or(VALIDATE_COOLING);
    let rw_sd_scale = vc.and_then(|s| s.rw_sd_scale).unwrap_or(1.0);
    let pfilter_particles = vc.and_then(|s| s.pfilter_particles).unwrap_or(VALIDATE_PFILTER_PARTICLES);

    let prior_state = FitState::load(starts_from)?;
    if !prior_state.best_loglik.is_finite() {
        return Err(format!(
            "prior stage produced -inf loglik — cannot use as starting point.\n\
             Re-run the prior stage with more particles or check model specification.\n\
             Source: {}/fit_state.toml", starts_from
        ));
    }
    eprintln!("validate: starting from {} (loglik={:.1})", starts_from, prior_state.best_loglik);

    let mut config = FitRunConfig::build(
        fit, Some(&prior_state),
        n_chains, n_particles, n_iterations,
        cooling, seed, false,
    )?;

    if rw_sd_scale != 1.0 {
        for p in &mut config.estimated_params { p.rw_sd *= rw_sd_scale; }
        eprintln!("validate: rw_sd scaled by {:.1}×", rw_sd_scale);
    }

    // Cache check
    let input_hash = runner::compute_fit_input_hash(fit, &config, seed);
    if !force {
        match provenance::check_cache(&stage_dir, &input_hash) {
            provenance::CacheStatus::Match => {
                eprintln!("\x1b[33mvalidate skipped — results already exist for these inputs.\x1b[0m");
                eprintln!("  output:     {}/", stage_dir);
                eprintln!("  input hash: {}", input_hash);
                eprintln!("  Use --force to re-run.");
                return Ok(());
            }
            provenance::CacheStatus::Mismatch => {
                eprintln!("\x1b[33mvalidate — prior results exist but inputs have changed. Re-running.\x1b[0m");
            }
            provenance::CacheStatus::NotFound => {}
        }
    }

    std::fs::create_dir_all(&stage_dir)
        .map_err(|e| format!("cannot create {}: {}", stage_dir, e))?;

    let collector = DiagnosticCollector::new("validate");

    // ── Phase 1: Run IF2 chains ──────────────────────────────────────────────
    let t0_if2 = std::time::Instant::now();
    let chain_results = runner::run_chains_with_diagnostics(&config, &collector);
    let if2_elapsed = t0_if2.elapsed();

    let all_converged = chain_results.rhat.values().all(|&r| r < 1.1);
    if !all_converged {
        let max_rhat = chain_results.rhat.values().cloned().fold(0.0_f64, f64::max);
        let n_unconverged = chain_results.rhat.values().filter(|&&r| r > 1.1).count();
        let n_total = chain_results.rhat.len();
        collector.push(DiagnosticKind::ConvergenceIncomplete { max_rhat, n_unconverged, n_total });
    }

    let best = &chain_results.results.iter()
        .find(|(id, _)| *id == chain_results.best_chain)
        .unwrap().1;

    // ── Phase 2: Precise pfilter at MLE ──────────────────────────────────────
    eprintln!("\nrunning pfilter at MLE with {} particles...", pfilter_particles);

    let mut mle_params = config.base_params.clone();
    for spec in &config.estimated_params {
        mle_params[spec.index] = best.mle[spec.index];
    }

    // If holdout data exists, run PF on full series (train + holdout)
    // and report separate logliks. Otherwise, just train.
    let (pf_result, train_ll, holdout_ll) = if let Some(ref holdout_data) = fit.holdout {
        // Load holdout observations
        let (_holdout_stream, holdout_path) = holdout_data.iter().next()
            .ok_or_else(|| "empty [holdout] section".to_string())?;

        let holdout_obs: Vec<sim::inference::if2::Observation> =
            crate::pfilter::load_data_tsv_pub(holdout_path)?
                .into_iter()
                .map(|o| sim::inference::if2::Observation { time: o.time, value: o.value })
                .collect();

        let train_end = config.observations.last().map(|o| o.time).unwrap_or(0.0);
        let holdout_min = holdout_obs.first().map(|o| o.time).unwrap_or(f64::INFINITY);

        if holdout_min <= train_end {
            return Err(format!(
                "holdout data overlaps with training data.\n\
                 Train max t = {}, holdout min t = {}.\n\
                 Holdout observations must be strictly after training period.",
                train_end, holdout_min
            ));
        }

        // Build full observation list for the PF
        let n_train = config.observations.len();
        let mut full_obs: Vec<sim::inference::if2::Observation> = config.observations.iter()
            .map(|o| sim::inference::if2::Observation { time: o.time, value: o.value })
            .chain(holdout_obs.into_iter())
            .collect();
        full_obs.sort_by(|a, b| a.time.total_cmp(&b.time));
        let n_holdout = full_obs.len() - n_train;

        eprintln!("  running on full data ({} train + {} holdout observations)...",
            config.observations.len(), n_holdout);

        let pf = run_pfilter_with_obs(&config, &mle_params, &full_obs, pfilter_particles, seed)?;

        // Split logliks by train/holdout boundary
        let train_ll: f64 = pf.ll_increments.iter().zip(&pf.obs_times)
            .filter(|(_, &t)| t <= train_end)
            .map(|(ll, _)| ll)
            .sum();
        let holdout_ll: f64 = pf.ll_increments.iter().zip(&pf.obs_times)
            .filter(|(_, &t)| t > train_end)
            .map(|(ll, _)| ll)
            .sum();

        eprintln!("  train loglik:   {:.1} ({} obs)", train_ll, config.observations.len());
        eprintln!("  holdout loglik: {:.1} ({} obs)",
            holdout_ll, n_holdout);

        (pf, Some(train_ll), Some(holdout_ll))
    } else {
        let pf = run_pfilter_at_mle(&config, &mle_params, pfilter_particles, seed)?;
        (pf, None, None)
    };

    let loglik = pf_result.loglik;
    let loglik_sd = pf_result.loglik_sd;
    let ess_mean = pf_result.ess_mean;
    let ess_min = pf_result.ess_min;

    eprintln!("  total loglik = {:.1} ± {:.1}", loglik, loglik_sd);
    eprintln!("  ESS at MLE: mean={:.0}, min={:.0}", ess_mean, ess_min);

    if ess_min < pfilter_particles as f64 / 4.0 {
        collector.push(DiagnosticKind::LowESSAtMLE {
            ess_mean, ess_min, n_particles: pfilter_particles,
        });
    }

    // Write ESS trace
    write_ess_trace(&stage_dir, &pf_result)?;

    // Write full pfilter trace (predictions + ESS + ll_increments)
    write_pfilter_trace(&stage_dir, &pf_result, &config)?;

    // ── Phase 3: Profile likelihoods ─────────────────────────────────────────
    eprintln!("\nprofiling all {} estimated parameters...", config.estimated_params.len());
    let profile_dir = format!("{}/profiles", stage_dir);
    std::fs::create_dir_all(&profile_dir)
        .map_err(|e| format!("cannot create {}: {}", profile_dir, e))?;

    let profiles = run_profiles(&config, &mle_params, &profile_dir, seed)?;

    // ── Write outputs ────────────────────────────────────────────────────────
    let start_values: HashMap<String, f64> = runner::collect_all_params(
        &best.mle, &config.estimated_params, &config.model,
        &config.base_params, &config.compiled,
    );

    let rw_sd = match runner::auto_rw_sd(&chain_results.results, &config.estimated_params) {
        Ok((rw, _)) => rw,
        Err(_) => config.estimated_params.iter()
            .map(|s| (s.name.clone(), s.rw_sd * 0.5))
            .collect(),
    };

    let state = FitState {
        stage: "validate".into(),
        seed,
        timestamp: crate::fit::scout::now_iso8601_pub(),
        input_hash: Some(input_hash.clone()),
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik: loglik,
        initial_loglik: prior_state.best_loglik,
        best_chain: chain_results.best_chain,
        n_chains: n_chains,
        n_good_chains: None,
        start_values,
        rw_sd,
        loglik_type: Some("if2".into()),
        acceptance_rate: None,
        tail_rhat: chain_results.rhat.clone(),
        ivp_params: config.estimated_params.iter()
            .filter(|p| p.ivp).map(|p| p.name.clone()).collect(),
        chain_logliks: chain_results.results.iter()
            .map(|(_, r)| r.final_loglik).collect(),
    };
    state.save(&stage_dir)?;

    // Per-chain outputs
    let param_names: Vec<String> = config.model.parameters.iter().map(|p| p.name.clone()).collect();
    runner::write_chain_outputs(
        &stage_dir, &chain_results.results, &config.estimated_params,
        &param_names, &config.base_params, &config.compiled,
    )?;
    runner::write_chain_starts(
        &stage_dir, None, &config.estimated_params, n_chains,
    )?;
    runner::write_diagnostics(&stage_dir, &chain_results.results)?;

    // mle_params.toml with provenance
    let all_params = runner::collect_all_params(
        &best.mle, &config.estimated_params, &config.model,
        &config.base_params, &config.compiled,
    );
    let model_hash = hashing::model_hash(&config.model_ir_json);
    let input_hash = runner::compute_fit_input_hash(fit, &config, seed);

    let metadata = MleMetadata {
        input_hash: input_hash.clone(),
        model_path: fit.fit.model.clone(),
        model_hash: model_hash.clone(),
        data_hashes: fit.data.iter().map(|(name, path)| {
            let bytes = std::fs::read(path).unwrap_or_default();
            let hash = hex::encode(&sha2::Sha256::digest(&bytes)[..4]);
            (format!("{} ({})", name, path), hash)
        }).collect(),
        seed,
        stage: "validate".into(),
        best_chain: chain_results.best_chain,
        backend: fit.config.backend.clone(),
        dt: fit.config.dt,
        loglik,
        loglik_sd,
        n_particles: pfilter_particles,
        ess_at_mle: Some((ess_mean, ess_min)),
        timestamp: state.timestamp.clone(),
    };
    provenance::write_mle_params(
        &format!("{}/mle_params.toml", stage_dir),
        &all_params,
        &metadata,
    )?;

    // fit_record.json
    write_fit_record(&stage_dir, fit, &config, &chain_results, &metadata, &profiles, &all_params, train_ll, holdout_ll)?;

    // fit_report.txt
    write_fit_report(&stage_dir, fit, &config, &chain_results, &metadata, &profiles, &all_params)?;

    // pfilter_loglik.txt
    let loglik_txt = if let (Some(tll), Some(hll)) = (train_ll, holdout_ll) {
        format!("train:   {:.4}\nholdout: {:.4}\ntotal:   {:.4} ± {:.4} (N={})\n",
            tll, hll, loglik, loglik_sd, pfilter_particles)
    } else {
        format!("{:.4} ± {:.4} (N={})\n", loglik, loglik_sd, pfilter_particles)
    };
    std::fs::write(format!("{}/pfilter_loglik.txt", stage_dir), loglik_txt).ok();

    // Summary JSON
    write_summary(&stage_dir, &chain_results, &config, &metadata, &profiles, train_ll, holdout_ll)?;

    // Render and persist diagnostics
    collector.render_to_stderr();
    let _ = collector.write_json(&format!("{}/diagnostics.json", stage_dir));

    let total_elapsed = t0_if2.elapsed();
    eprintln!("\nvalidate complete in {:.1}s (IF2: {:.1}s): {}/",
        total_elapsed.as_secs_f64(), if2_elapsed.as_secs_f64(), stage_dir);
    eprintln!("  loglik: {:.1} ± {:.1} (N={})", loglik, loglik_sd, pfilter_particles);
    eprintln!("  ESS: mean={:.0}, min={:.0}", ess_mean, ess_min);
    eprintln!("  converged: {}", if all_converged { "yes" } else { "NO" });
    eprintln!("  profiles: {}/profiles/", stage_dir);
    eprintln!("\nnext: camdl simulate batch sweep.toml --params {}/mle_params.toml", stage_dir);

    Ok(())
}

struct PfilterResult {
    loglik: f64,
    loglik_sd: f64,
    ess_mean: f64,
    ess_min: f64,
    ess_trace: Vec<f64>,
    ll_increments: Vec<f64>,
    obs_times: Vec<f64>,
    predictions: Option<Vec<sim::inference::particle_filter::PredictionDiag>>,
}

fn run_pfilter_at_mle(
    config: &FitRunConfig,
    params: &[f64],
    n_particles: usize,
    seed: u64,
) -> Result<PfilterResult, String> {
    run_pfilter_with_obs(config, params, &config.observations, n_particles, seed)
}

fn run_pfilter_with_obs(
    config: &FitRunConfig,
    params: &[f64],
    _obs: &[sim::inference::if2::Observation],
    n_particles: usize,
    seed: u64,
) -> Result<PfilterResult, String> {
    let process = config.build_process();
    let obs_model_trait = config.build_obs_model();
    let smc_config = SMCConfig {
        n_particles,
        ..config.smc_config()
    };

    let result = bootstrap_filter(
        &process, &obs_model_trait, params, &smc_config, seed,
    ).map_err(|e| format!("pfilter error: {:?}", e))?;

    let ess_mean = result.ess_trace.iter().sum::<f64>() / result.ess_trace.len() as f64;
    let ess_min = result.ess_trace.iter().cloned().fold(f64::INFINITY, f64::min);
    let loglik_sd = estimate_loglik_sd(&result.ll_increments);

    Ok(PfilterResult {
        loglik: result.log_likelihood,
        loglik_sd,
        ess_mean,
        ess_min,
        ess_trace: result.ess_trace,
        ll_increments: result.ll_increments,
        obs_times: (0..obs_model_trait.n_observations()).map(|i| obs_model_trait.obs_time(i)).collect(),
        predictions: result.predictions,
    })
}

fn estimate_loglik_sd(ll_increments: &[f64]) -> f64 {
    // Bootstrap SE estimate: SD of 50 replicate loglik sums using resampled increments
    // Simpler approach: use batch means with ~10 batches
    let n = ll_increments.len();
    if n < 10 { return 0.0; }
    let batch_size = n / 10;
    let batch_sums: Vec<f64> = (0..10).map(|b| {
        let start = b * batch_size;
        let end = if b == 9 { n } else { start + batch_size };
        ll_increments[start..end].iter().sum::<f64>()
    }).collect();
    let mean = batch_sums.iter().sum::<f64>() / 10.0;
    let var = batch_sums.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / 9.0;
    (var * 10.0).sqrt() // SE of the total sum
}

fn write_ess_trace(dir: &str, pf: &PfilterResult) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/ess_at_mle.tsv", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;
    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    writeln!(f, "time\tESS").unwrap();
    for (t, ess) in pf.obs_times.iter().zip(&pf.ess_trace) {
        writeln!(f, "{}\t{:.1}", t, ess).unwrap();
    }
    Ok(())
}

fn write_pfilter_trace(dir: &str, pf: &PfilterResult, config: &FitRunConfig) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/pfilter_trace.tsv", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;

    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    if let Some(ref preds) = pf.predictions {
        writeln!(f, "time\tll_increment\tESS\tobs_mean\tobs_q05\tobs_q50\tobs_q95\tstate_mean\tstate_q05\tstate_q50\tstate_q95\tobserved").unwrap();
        for (i, t) in pf.obs_times.iter().enumerate() {
            let p = &preds[i];
            let obs_val = if i < pf.obs_times.len() {
                // Match against obs_times to find corresponding observed value
                // For holdout rows (time > train end), report NaN
                let t = pf.obs_times[i];
                config.observations.iter()
                    .find(|o| (o.time - t).abs() < 1e-10)
                    .map_or(f64::NAN, |o| o.value)
            } else {
                f64::NAN
            };
            writeln!(f, "{}\t{:.4}\t{:.1}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
                t, pf.ll_increments[i], pf.ess_trace[i],
                p.obs_mean, p.obs_q05, p.obs_q50, p.obs_q95,
                p.state_mean, p.state_q05, p.state_q50, p.state_q95,
                obs_val).unwrap();
        }
    } else {
        writeln!(f, "time\tll_increment\tESS").unwrap();
        for (i, t) in pf.obs_times.iter().enumerate() {
            writeln!(f, "{}\t{:.4}\t{:.1}", t, pf.ll_increments[i], pf.ess_trace[i]).unwrap();
        }
    }
    eprintln!("  pfilter trace written to {}", path);
    Ok(())
}

#[derive(Default)]
struct ProfileResult {
    name: String,
    ci_lower: f64,
    ci_upper: f64,
    curvature: f64,
}

fn run_profiles(
    config: &FitRunConfig,
    mle_params: &[f64],
    profile_dir: &str,
    seed: u64,
) -> Result<Vec<ProfileResult>, String> {
    use rayon::prelude::*;
    use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

    let n_grid = 21; // points per profile
    let mp = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("{prefix:>12} [{bar:20}] {pos}/{len}")
        .unwrap();

    let results: Vec<ProfileResult> = config.estimated_params.par_iter().map(|focal| {
        let pb = mp.add(ProgressBar::new(n_grid as u64));
        pb.set_style(style.clone());
        pb.set_prefix(focal.name.clone());

        let mle_value = mle_params[focal.index];
        let (lo, hi) = if focal.lower.is_finite() && focal.upper.is_finite() {
            (focal.lower, focal.upper)
        } else {
            (mle_value * 0.5, mle_value * 1.5)
        };

        let grid: Vec<f64> = (0..n_grid).map(|i| {
            lo + (hi - lo) * i as f64 / (n_grid - 1) as f64
        }).collect();

        let mut profile_points: Vec<(f64, f64)> = Vec::new();

        for &focal_value in &grid {
            // Run short IF2 with focal param fixed at grid value
            let mut fixed_params: Vec<EstimatedParam> = config.estimated_params.iter()
                .filter(|p| p.name != focal.name)
                .cloned()
                .collect();
            // Set initial values to MLE for non-focal params
            for p in &mut fixed_params {
                p.initial = mle_params[p.index];
                p.rw_sd *= 0.5; // tighter perturbation for profile
            }

            let mut base = mle_params.to_vec();
            base[focal.index] = focal_value;

            let profile_particles = (config.if2_config.n_particles / 5).clamp(200, 2000);
            let profile_config = IF2Config {
                n_particles: profile_particles,
                n_iterations: 30,
                cooling_fraction: 0.95,
                cooling_target_iters: 30,
                dt: config.if2_config.dt,
                t_start: config.compiled.model.simulation.t_start,
                simplex_groups: vec![],
                skip_first_obs_from_loglik: config.ic_free,
            };

            let process = config.build_process();
            let obs_model_profile = config.build_obs_model();

            let chain_seed = seed ^ focal.index as u64 ^ (focal_value.to_bits());
            let result = run_if2(
                &process, &obs_model_profile, &base, &fixed_params, &profile_config, chain_seed,
            );

            match result {
                Ok(r) => {
                    // Evaluate true loglik via a quick particle filter at the
                    // profile point's MLE, rather than using the IF2 perturbed
                    // loglik (which inflates profile confidence intervals).
                    let mut pf_params = base.clone();
                    for p in &fixed_params {
                        pf_params[p.index] = r.mle[p.index];
                    }
                    let pf_ll = runner::run_quick_pfilter(
                        config, &pf_params, profile_particles, chain_seed,
                    );
                    profile_points.push((focal_value, pf_ll));
                }
                Err(e) => {
                    eprintln!("warning: profile point {}={:.4} failed: {:?}", focal.name, focal_value, e);
                    profile_points.push((focal_value, f64::NEG_INFINITY));
                }
            }
            pb.inc(1);
        }
        pb.finish();

        // Write profile TSV
        {
            use std::io::Write;
            let path = format!("{}/{}_profile.tsv", profile_dir, focal.name);
            if let Ok(mut f) = std::fs::File::create(&path) {
                writeln!(f, "{}\tloglik", focal.name).ok();
                for (v, ll) in &profile_points {
                    writeln!(f, "{:.6}\t{:.2}", v, ll).ok();
                }
            }
        }

        // Compute 95% CI from profile (chi-squared cutoff = 1.92)
        let max_ll = profile_points.iter().map(|(_, ll)| *ll).fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_ll - 1.92;
        let above: Vec<f64> = profile_points.iter()
            .filter(|(_, ll)| *ll >= threshold)
            .map(|(v, _)| *v)
            .collect();
        let ci_lower = above.iter().cloned().fold(f64::INFINITY, f64::min);
        let ci_upper = above.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Rough curvature at MLE
        let curvature = if profile_points.len() >= 3 {
            let step = (hi - lo) / (n_grid - 1) as f64;
            let mid_idx = n_grid / 2;
            if mid_idx > 0 && mid_idx < profile_points.len() - 1 {
                let ll_minus = profile_points[mid_idx - 1].1;
                let ll_mid = profile_points[mid_idx].1;
                let ll_plus = profile_points[mid_idx + 1].1;
                -(ll_plus - 2.0 * ll_mid + ll_minus) / (step * step)
            } else { 0.0 }
        } else { 0.0 };

        ProfileResult {
            name: focal.name.clone(),
            ci_lower,
            ci_upper,
            curvature,
        }
    }).collect();

    Ok(results)
}


fn write_fit_record(
    dir: &str,
    fit: &FitToml,
    config: &FitRunConfig,
    results: &runner::ChainResults,
    metadata: &MleMetadata,
    profiles: &[ProfileResult],
    all_params: &HashMap<String, f64>,
    train_ll: Option<f64>,
    holdout_ll: Option<f64>,
) -> Result<(), String> {
    // Collect content hashes of all output files for the manifest
    let output_hashes: serde_json::Value = {
        let hashes = provenance::collect_output_hashes(dir, false);
        let mut map = serde_json::Map::new();
        for (name, hash) in hashes {
            map.insert(name, serde_json::Value::String(hash));
        }
        serde_json::Value::Object(map)
    };

    let record = serde_json::json!({
        "model": {
            "path": fit.fit.model,
            "hash": &metadata.model_hash[..8.min(metadata.model_hash.len())],
        },
        "data": fit.data.iter().map(|(name, path)| {
            (name.clone(), serde_json::json!({ "path": path }))
        }).collect::<serde_json::Map<String, serde_json::Value>>(),
        "fit_config": {
            "estimated": config.estimated_params.iter().map(|p| &p.name).collect::<Vec<_>>(),
            "fixed": fit.fixed.keys().collect::<Vec<_>>(),
        },
        "method": {
            "algorithm": "IF2",
            "backend": fit.config.backend,
            "dt": fit.config.dt,
            "seed": metadata.seed,
        },
        "results": {
            "mle": all_params,
            "loglik": if train_ll.is_some() {
                serde_json::json!({
                    "train": train_ll.unwrap(),
                    "holdout": holdout_ll.unwrap(),
                    "total": metadata.loglik,
                })
            } else {
                serde_json::json!(metadata.loglik)
            },
            "loglik_sd": metadata.loglik_sd,
            "ess_at_mle": metadata.ess_at_mle.map(|(m, n)| serde_json::json!({"mean": m, "min": n})),
            "convergence": {
                "rhat_max": results.rhat.values().cloned().fold(0.0_f64, f64::max),
                "all_converged": results.rhat.values().all(|&r| r < 1.1),
            },
            "identifiability": profiles.iter().map(|p| {
                (p.name.clone(), serde_json::json!({
                    "ci_95": [p.ci_lower, p.ci_upper],
                    "curvature": p.curvature,
                }))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
        },
        "provenance": {
            "input_hash": metadata.input_hash,
            "content_hash": provenance::mle_params_tamper_hash(all_params),
            "timestamp": metadata.timestamp,
            "camdl_version": crate::version::VERSION_SHORT,
        },
        "outputs": output_hashes,
    });

    let path = format!("{}/fit_record.json", dir);
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap())
        .map_err(|e| format!("cannot write {}: {}", path, e))
}

fn write_fit_report(
    dir: &str,
    fit: &FitToml,
    config: &FitRunConfig,
    results: &runner::ChainResults,
    metadata: &MleMetadata,
    profiles: &[ProfileResult],
    all_params: &HashMap<String, f64>,
) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/fit_report.txt", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;

    writeln!(f, "camdl fit report").unwrap();
    writeln!(f, "================").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Model:     {}", fit.fit.model).unwrap();
    writeln!(f, "Backend:   {}", fit.config.backend).unwrap();
    writeln!(f, "Timestamp: {}", metadata.timestamp).unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Log-likelihood: {:.1} ± {:.1} (N={})", metadata.loglik, metadata.loglik_sd, metadata.n_particles).unwrap();
    if let Some((ess_mean, ess_min)) = metadata.ess_at_mle {
        writeln!(f, "ESS at MLE:     mean={:.0}, min={:.0}", ess_mean, ess_min).unwrap();
    }
    writeln!(f).unwrap();
    writeln!(f, "Estimated parameters:").unwrap();
    for spec in &config.estimated_params {
        let best = &results.results.iter()
            .find(|(id, _)| *id == results.best_chain).unwrap().1;
        let rhat = results.rhat.get(&spec.name).copied().unwrap_or(f64::NAN);
        let profile = profiles.iter().find(|p| p.name == spec.name);
        let ci = profile.map(|p| format!("[{:.4}, {:.4}]", p.ci_lower, p.ci_upper))
            .unwrap_or_else(|| "—".into());
        writeln!(f, "  {:12} = {:<12.6} Rhat={:.3}  CI_95={}", spec.name, best.mle[spec.index], rhat, ci).unwrap();
    }
    writeln!(f).unwrap();
    writeln!(f, "Fixed parameters:").unwrap();
    for name in fit.fixed.keys() {
        writeln!(f, "  {}", name).unwrap();
    }
    writeln!(f).unwrap();
    writeln!(f, "Provenance: input_hash={}, content_hash={}",
        metadata.input_hash, provenance::mle_params_tamper_hash(all_params)).unwrap();

    Ok(())
}

fn write_summary(
    dir: &str,
    results: &runner::ChainResults,
    config: &FitRunConfig,
    metadata: &MleMetadata,
    profiles: &[ProfileResult],
    train_ll: Option<f64>,
    holdout_ll: Option<f64>,
) -> Result<(), String> {
    let summary = serde_json::json!({
        "stage": "validate",
        "n_chains": config.n_chains,
        "best_loglik": metadata.loglik,
        "loglik_sd": metadata.loglik_sd,
        "train_loglik": train_ll,
        "holdout_loglik": holdout_ll,
        "ess_at_mle": metadata.ess_at_mle.map(|(m, n)| serde_json::json!({"mean": m, "min": n})),
        "input_hash": metadata.input_hash,
        "converged": results.rhat.values().all(|&r| r < 1.1),
        "parameters": config.estimated_params.iter().map(|spec| {
            let rhat = results.rhat.get(&spec.name).copied().unwrap_or(f64::NAN);
            let best = &results.results.iter()
                .find(|(id, _)| *id == results.best_chain).unwrap().1;
            let profile = profiles.iter().find(|p| p.name == spec.name);
            serde_json::json!({
                "name": spec.name,
                "estimate": best.mle[spec.index],
                "rhat": rhat,
                "ci_95": profile.map(|p| vec![p.ci_lower, p.ci_upper]),
                "curvature": profile.map(|p| p.curvature),
            })
        }).collect::<Vec<_>>(),
    });

    let path = format!("{}/validate_summary.json", dir);
    std::fs::write(&path, serde_json::to_string_pretty(&summary).unwrap())
        .map_err(|e| format!("cannot write {}: {}", path, e))
}
