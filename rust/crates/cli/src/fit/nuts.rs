//! `nuts` on `ode` stage runner (gh#275 Phase 2).
//!
//! Gradient-based Bayesian sampling of the deterministic ODE marginal likelihood
//! via `sim::inference::ode_nuts::run_ode_nuts` (which composes `det_grad` with
//! the prior + Jacobian into a NUTS target). Leaner than the PGAS stage — no
//! particles, no CSMC, no tempering; just per-chain NUTS and the shared trace /
//! fit_state output. On a stochastic backend, gradient-NUTS lives inside `pgas`,
//! so this path is `ode`-only (routed by the method registry).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::ode_nuts::OdeNutsConfig;
use sim::inference::pmmh::Prior;

use crate::cas::iso8601_utc;
use crate::fit::loglik::LoglikType;
use crate::fit::state::FitState;

/// CLI knobs for a `Stage::Nuts`, extracted from the fit.toml stage (and CLI
/// overrides applied at the dispatch site).
pub struct NutsStageOpts {
    pub n_chains: usize,
    pub warmup: usize,
    pub samples: usize,
    pub max_tree_depth: usize,
    pub target_accept: f64,
    pub dense_mass: bool,
    pub init_method: super::init::InitMethod,
    pub survey_path: Option<std::path::PathBuf>,
    pub survey_top_k_n: Option<usize>,
    /// Coarse warm-up step (gh#396 follow-on); `None` = off. Validated against the
    /// fit-wide `dt` and the observation streams in `run_stage`.
    pub burnin_dt: Option<f64>,
}

impl NutsStageOpts {
    pub fn from_stage(stage: &super::config_v2::Stage) -> Result<Self, String> {
        match stage {
            super::config_v2::Stage::Nuts {
                chains, warmup, samples, max_tree_depth, target_accept, dense_mass,
                init_method, survey_path, survey_top_k_n, burnin_dt, ..
            } => Ok(NutsStageOpts {
                n_chains: *chains,
                warmup: *warmup,
                samples: *samples,
                max_tree_depth: *max_tree_depth,
                target_accept: *target_accept,
                dense_mass: *dense_mass,
                init_method: init_method.clone(),
                survey_path: survey_path.clone(),
                survey_top_k_n: *survey_top_k_n,
                burnin_dt: *burnin_dt,
            }),
            other => Err(format!(
                "NutsStageOpts::from_stage: expected Stage::Nuts, got {}",
                other.method_name()
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    fit: &super::config_v2::FitConfigV2,
    stage_name: &str,
    _stage: &super::config_v2::Stage,
    stage_dir: &Path,
    opts: NutsStageOpts,
    seed: u64,
    force: bool,
    resume: bool,
) -> Result<(), String> {
    if !force && !resume && stage_dir.join("fit_state.toml").exists() {
        eprintln!(
            "\x1b[33mnuts results already exist in {}. Use --force to re-run.\x1b[0m",
            stage_dir.display()
        );
        return Ok(());
    }
    std::fs::create_dir_all(stage_dir)
        .map_err(|e| format!("cannot create {}: {}", stage_dir.display(), e))?;

    let config = super::runner::FitRunConfig::build(
        fit, None, opts.n_chains, 1, 1, 1.0, 1, seed, false,
    )?;
    let dt = config.if2_config.dt;

    // Bayesian posteriors require a proper prior — refuse an implicit
    // improper-uniform, exactly as PGAS/PMMH do (gh#audit-C4): a credible
    // interval from a silent flat prior is a rescaled likelihood profile.
    let estimate = &fit.estimate;
    let resolved: Vec<(Prior, &'static str)> = config
        .estimated_params
        .iter()
        .map(|s| super::runner::resolve_prior(&s.name, estimate, &config.model))
        .collect();
    let missing: Vec<&str> = config
        .estimated_params
        .iter()
        .zip(&resolved)
        .filter(|(_, (_, src))| *src == "flat (default)")
        .map(|(s, _)| s.name.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "nuts refuses to run with implicit improper-uniform priors. The following \
             estimated parameters have no prior: {}. Add a `[estimate.<name>.prior]` \
             block in fit.toml or a `~` prior in the model.",
            missing.join(", ")
        ));
    }
    let priors: Vec<Prior> = resolved.into_iter().map(|(p, _)| p).collect();

    crate::util::print_scheduled_actions_summary(&config.model_declared, &config.model);
    crate::util::print_observations_summary(&config.model);

    let compiled = Arc::new(
        CompiledModel::new(config.model.clone()).map_err(|e| format!("compile: {e}"))?,
    );
    let obs_model = config.build_obs_model();
    let obs_times: Vec<f64> = config.observations.iter().map(|o| o.time).collect();
    let param_names: Vec<String> =
        config.estimated_params.iter().map(|s| s.name.clone()).collect();

    // Coarse burn-in step (gh#396 follow-on). Validated against the fit-wide `dt`
    // and the observation streams before any chain runs; `None`, or a value `<= dt`,
    // is off (fine step throughout). The warm-up/scored split is the first
    // observation (derived inside `det_grad`); here we only reject the cases the
    // gradient path cannot coarsen soundly.
    let burnin_dt: f64 = super::config_v2::validate_burnin_dt(
        opts.burnin_dt,
        dt,
        obs_model.n_interval_streams(),
        obs_times.first().copied(),
        config.model.simulation.t_start,
    )?;

    // Per-chain starting points. Honors the stage's `init` method: a dispersed
    // method (`uniform_unconstrained` / `lhs`) gives each chain its own
    // over-dispersed start — Stan's basis for a meaningful between-chain R-hat and
    // the standard defense against all chains sharing one warm-up pathology (a bad
    // early metric that builds runaway trees). `single` (default) keeps the shared
    // start. Reuses the tested init machinery (same as PGAS/PMMH); unsupported
    // methods (survey / warm-start) error actionably here rather than being
    // silently ignored.
    let chain_starts: Vec<Vec<f64>> = super::init::build_chain_param_vecs(
        &opts.init_method,
        &config.estimated_params,
        &config.base_params,
        opts.n_chains,
        seed,
    )
    .map_err(|e| format!("nuts: {}", e))?
    .unwrap_or_else(|| {
        // `Single` (and other non-dispersing methods): every chain starts at the
        // estimated parameters' declared `initial` values — NOT whatever
        // `base_params` holds at those indices (which can be a resolved/data
        // value, not the fit's starting point). `run_ode_nuts` seeds `z` from
        // these, so getting them wrong starts the chain at the wrong parameter.
        let mut single = config.base_params.clone();
        for ep in &config.estimated_params {
            single[ep.index] = ep.initial;
        }
        vec![single; opts.n_chains]
    });

    // Chains are independent (own seed, own RNG, own trace file) and their outputs
    // reduce order-independently (max-loglik chain + summed divergences), so run
    // them in parallel across the rayon pool — the same pattern PGAS/PMMH/IF2 use.
    // Parallelism does not change results: each chain's seed and draws are fixed.
    use rayon::prelude::*;
    struct ChainOut {
        chain_id: usize,
        n_divergent: usize,
        max_depth_hits: usize,
        warmup_max_depth_hits: usize,
        best_loglik: f64,
        best_params: Vec<f64>,
        /// This chain's post-warmup posterior draws, one row per sample, columns
        /// in `config.estimated_params` order. Kept (not discarded) so the stage
        /// can compute shared R̂/ESS diagnostics + write a combined `draws.tsv`,
        /// exactly as PGAS/PMMH do.
        samples: Vec<Vec<f64>>,
        status: String,
    }
    let chain_outs: Vec<ChainOut> = (0..opts.n_chains)
        .into_par_iter()
        .map(|chain_id| -> Result<ChainOut, String> {
            let chain_dir = stage_dir.join(format!("chain_{}", chain_id));
            std::fs::create_dir_all(&chain_dir)
                .map_err(|e| format!("cannot create {}: {}", chain_dir.display(), e))?;

            let cfg = OdeNutsConfig {
                n_warmup: opts.warmup,
                n_samples: opts.samples,
                max_tree_depth: opts.max_tree_depth,
                target_accept: opts.target_accept,
                init_step_size: 0.1,
                // Warm-up adapts a mass matrix (Stan's `metric`): diagonal by
                // default (rescales each parameter to ~unit posterior variance),
                // or dense (`dense_mass = true`) to also absorb parameter
                // correlations — the identifiability ridge. On the anisotropic
                // Garki posterior, diagonal takes ~7× larger steps at the same
                // acceptance and frees the wide-posterior parameter (a2) that
                // identity mass leaves stuck at its bound (gh#275).
                metric: if opts.dense_mass {
                    sim::inference::nuts::MassMetric::Dense
                } else {
                    sim::inference::nuts::MassMetric::Diagonal
                },
                dt,
                burnin_dt,
                // Independent chains: own dispersed start (below) + distinct RNG.
                seed: seed.wrapping_add(chain_id as u64),
            };

            // This chain's start: fixed params at their values, estimated params at
            // this chain's (possibly dispersed) initial. `run_ode_nuts` seeds its
            // starting `z` from `EstimatedParam::initial`, so overwrite those.
            let chain_start = &chain_starts[chain_id];
            let chain_estimated: Vec<sim::inference::types::EstimatedParam> = config
                .estimated_params
                .iter()
                .map(|ep| {
                    let mut e = ep.clone();
                    e.initial = chain_start[ep.index];
                    e
                })
                .collect();

            // Stream each posterior draw to `chain_N/trace.tsv` as it is produced,
            // so a long run fills the file incrementally (a watcher can `tail -f`
            // and plot) instead of the chain dir staying empty until the very end.
            // `TraceWriter` writes `draw`, `log_likelihood`, `log_posterior` as its
            // three fixed leading columns; `divergent` and `tree_depth` are the
            // extras. `tree_depth` lets a watcher tell a slow-but-progressing chain
            // from one pinned at the depth cap (the runaway/hung signature).
            let trace_path = chain_dir.join("trace.tsv");
            let writer = super::trace_writer::TraceWriter::new(
                &trace_path.to_string_lossy(),
                "draw",
                "log_likelihood",
                &["divergent", "tree_depth"],
                &param_names,
                /* append */ false,
            );
            // Throttle the progress log to ~10 lines per phase (per chain).
            let warmup_every = opts.warmup.max(1).div_ceil(10);
            let sample_every = opts.samples.max(1).div_ceil(10);
            let on_iter = {
                use sim::inference::ode_nuts::{NutsIter, NutsPhase};
                // Borrows `writer` (not `move`): the callback streams rows through
                // it, and it must remain live to flush after the run.
                |it: &NutsIter| match it.phase {
                    NutsPhase::Sampling => {
                        let div = if it.divergent { "1" } else { "0" };
                        let depth = it.tree_depth.to_string();
                        writer.write_row(
                            it.iter, it.loglik, it.log_posterior, &[div, &depth],
                            &it.params_natural,
                        );
                        if it.iter % sample_every == 0 || it.iter + 1 == it.total {
                            log::info!(target: "nuts",
                                "chain {} sampling {}/{} · depth={} · logpost={:.1}{}",
                                chain_id + 1, it.iter + 1, it.total, it.tree_depth,
                                it.log_posterior, if it.divergent { " · DIVERGENT" } else { "" });
                        }
                    }
                    NutsPhase::Warmup => {
                        if it.iter % warmup_every == 0 || it.iter + 1 == it.total {
                            log::info!(target: "nuts",
                                "chain {} warmup {}/{} · depth={} · step={:.4}",
                                chain_id + 1, it.iter + 1, it.total, it.tree_depth, it.step_size);
                        }
                    }
                }
            };
            let result = sim::inference::ode_nuts::run_ode_nuts_with_progress(
                &compiled, &obs_model, &obs_times, chain_start,
                &chain_estimated, &priors, &cfg, Some(&on_iter),
            )
            .map_err(|e| format!("nuts chain {} error: {}", chain_id + 1, e))?;
            // `on_iter`'s borrow of `writer` ends at the call above (its last use),
            // so the writer can now be flushed to disk.
            drop(writer); // flush the buffered trace to disk

            // Best draw (point estimate) from the returned samples — the rows were
            // already streamed to disk by the callback above.
            let mut best_loglik = f64::NEG_INFINITY;
            let mut best_params = chain_start.clone();
            for (i, sample) in result.samples.iter().enumerate() {
                if result.sample_loglik[i] > best_loglik {
                    best_loglik = result.sample_loglik[i];
                    let mut p = chain_start.clone();
                    for (j, ep) in config.estimated_params.iter().enumerate() {
                        p[ep.index] = sample[j];
                    }
                    best_params = p;
                }
            }

            // A chain stuck at max depth through warm-up never reaches sampling
            // (the "hung" case), so surface the warm-up hit count too.
            let depth_note = match (result.warmup_max_depth_hits, result.max_depth_hits) {
                (0, 0) => String::new(),
                (w, s) => format!(" · max-depth hits: {w} warmup / {s} sampling"),
            };
            let status = format!(
                "chain {} · {} draws · {} divergent · accept={:.2} · step={:.4} · tree_depth={:.1}{}",
                chain_id + 1,
                result.samples.len(),
                result.n_divergent,
                result.mean_accept,
                result.step_size,
                result.mean_tree_depth,
                depth_note,
            );
            Ok(ChainOut {
                chain_id,
                n_divergent: result.n_divergent,
                max_depth_hits: result.max_depth_hits,
                warmup_max_depth_hits: result.warmup_max_depth_hits,
                best_loglik,
                best_params,
                samples: result.samples,
                status,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Report per-chain status in deterministic chain order (completion order is
    // nondeterministic under parallelism).
    for c in &chain_outs {
        crate::status::step("nuts", c.status.clone());
    }

    // Reduce across chains: total divergences + the single best-loglik draw.
    // Ties keep the EARLIER chain (fold with `>=`), matching the strict-`>`
    // sequential loop this replaced — so the parallel run is byte-identical.
    let total_divergent: usize = chain_outs.iter().map(|c| c.n_divergent).sum();
    let best = chain_outs
        .iter()
        .fold(None::<&ChainOut>, |acc, c| match acc {
            Some(b) if b.best_loglik >= c.best_loglik => Some(b),
            _ => Some(c),
        })
        .expect("at least one chain runs");
    let best_loglik = best.best_loglik;
    let best_chain = best.chain_id;
    let best_params = best.best_params.clone();

    if total_divergent > 0 {
        crate::status::hint(format!(
            "{} divergent transition(s) across chains — the posterior geometry may be \
             difficult; consider a lower dt or reparameterizing.",
            total_divergent
        ));
    }
    let total_max_depth: usize = chain_outs.iter().map(|c| c.max_depth_hits).sum();
    let total_warmup_max_depth: usize =
        chain_outs.iter().map(|c| c.warmup_max_depth_hits).sum();
    if total_max_depth + total_warmup_max_depth > 0 {
        crate::status::hint(format!(
            "max tree depth ({}) hit on {} warm-up + {} sampling step(s) across chains — \
             the sampler is building maximal trees (slow; a chain pinned through warm-up is \
             the \"hung\" case). Set `dense_mass = true` for a correlated posterior, use a \
             dispersed `init` (e.g. `uniform_unconstrained`), raise `max_tree_depth`, or \
             reparameterize.",
            opts.max_tree_depth, total_warmup_max_depth, total_max_depth,
        ));
    }

    // Shared posterior diagnostics + a combined draws.tsv — the same artifacts
    // PGAS/PMMH write, so nuts loads through the same `MethodResult` path and
    // reports ESS/iteration + ESS/second instead of erroring in `fit summary` /
    // `fit table`. Chains are visited in deterministic id order so the diagnostic
    // (and draws.tsv row order) is reproducible regardless of completion order.
    let mut ordered: Vec<&ChainOut> = chain_outs.iter().collect();
    ordered.sort_by_key(|c| c.chain_id);
    let chain_samples: Vec<&Vec<Vec<f64>>> = ordered.iter().map(|c| &c.samples).collect();
    let diag = nuts_diagnostics(&config.estimated_params, &chain_samples);
    write_nuts_summary(stage_dir, opts.n_chains, &diag, total_divergent)?;
    write_nuts_draws(
        stage_dir,
        &config.estimated_params,
        ordered.iter().map(|c| (c.chain_id, &c.samples)),
    )?;

    let start_values: HashMap<String, f64> = config
        .estimated_params
        .iter()
        .map(|s| (s.name.clone(), s.initial))
        .collect();

    let state = FitState {
        stage: stage_name.to_string(),
        seed,
        timestamp: iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik,
        initial_loglik: f64::NEG_INFINITY,
        best_chain,
        n_chains: opts.n_chains,
        n_good_chains: None,
        start_values,
        rw_sd: HashMap::new(),
        // The deterministic ODE marginal likelihood (same kind as MH-on-ode).
        loglik_type: Some(LoglikType::OdeMarginal),
        acceptance_rate: None,
        tail_chain_agreement: HashMap::new(),
        ivp_params: Vec::new(),
        chain_logliks: Vec::new(),
        chain_eval_logliks: Vec::new(),
        chain_eval_ses: Vec::new(),
        resolved_gate: None,
        resolved_loglik_eval: None,
        chain_init_source: None,
        dt_check: None,
    };
    let _ = best_params; // point-estimate is derivable from the trace draws
    state
        .save(&stage_dir.to_string_lossy())
        .map_err(|e| format!("cannot write fit_state.toml: {e}"))?;

    Ok(())
}

/// Per-estimated-param R̂ + Geyer ESS from the chains' posterior draws, computed
/// through the shared [`crate::fit::runner::compute_rhat_ess`] every Bayesian
/// method routes through. `chain_samples[chain][draw][param]`, columns in
/// `estimated_params` order. Mirrors PGAS's `compute_diagnostics`; the only
/// nuts-specific part is the draw layout.
struct NutsDiag {
    rhat: BTreeMap<String, f64>,
    ess: BTreeMap<String, f64>,
    ess_per_chain: BTreeMap<String, Vec<f64>>,
}

fn nuts_diagnostics(
    estimated_params: &[sim::inference::types::EstimatedParam],
    chain_samples: &[&Vec<Vec<f64>>],
) -> NutsDiag {
    let mut rhat = BTreeMap::new();
    let mut ess = BTreeMap::new();
    let mut ess_per_chain = BTreeMap::new();
    for (j, ep) in estimated_params.iter().enumerate() {
        // Column j of each chain's draw matrix = this param's per-chain trace.
        let chains: Vec<Vec<f64>> = chain_samples
            .iter()
            .map(|draws| draws.iter().map(|row| row[j]).collect())
            .collect();
        let d = super::runner::compute_rhat_ess(&chains);
        // R̂ is NaN below the structural minimum (≥2 chains, ≥4 samples); only
        // record it when finite, matching PGAS. ESS is always recorded (the
        // gate on the *joint* sum lives inside `compute_rhat_ess`; a NaN ess
        // serializes to null → the loader reads it as absent).
        if d.rhat.is_finite() {
            rhat.insert(ep.name.clone(), d.rhat);
        }
        ess.insert(ep.name.clone(), d.ess_total);
        if !d.ess_per_chain.is_empty() {
            ess_per_chain.insert(ep.name.clone(), d.ess_per_chain);
        }
    }
    NutsDiag { rhat, ess, ess_per_chain }
}

/// Write `nuts_summary.json` — the R̂/ESS/thin the `MethodResult` loader reads.
/// nuts does not thin, so `thin = 1` (`n_samples × 1` = raw sampling iters).
fn write_nuts_summary(
    dir: &Path,
    n_chains: usize,
    diag: &NutsDiag,
    n_divergent: usize,
) -> Result<(), String> {
    let summary = serde_json::json!({
        "stage": "nuts",
        "n_chains": n_chains,
        "rhat": diag.rhat,
        "ess": diag.ess,
        "ess_per_chain": diag.ess_per_chain,
        "n_divergent": n_divergent,
        // nuts draws are unthinned: n_samples (kept) × thin = raw sampling iters.
        "thin": 1,
    });
    let path = dir.join(crate::run_meta::FitAlgorithm::Nuts.summary_filename());
    let contents =
        serde_json::to_string_pretty(&summary).map_err(|e| format!("json error: {}", e))?;
    std::fs::write(&path, contents)
        .map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

/// Write the combined `draws.tsv` (post-warmup posterior draws, all chains).
/// Leading `chain` `draw` key columns match the PGAS/PMMH layout; the shared
/// draws loader keys on the estimated-param columns by name. nuts emits only
/// the estimated params (it writes no trajectories to join fixed params to).
fn write_nuts_draws<'a>(
    dir: &Path,
    estimated_params: &[sim::inference::types::EstimatedParam],
    chains: impl Iterator<Item = (usize, &'a Vec<Vec<f64>>)>,
) -> Result<(), String> {
    use std::io::Write;
    let path = dir.join("draws.tsv");
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&path).map_err(|e| format!("cannot create {}: {}", path.display(), e))?,
    );
    let names: Vec<&str> = estimated_params.iter().map(|s| s.name.as_str()).collect();
    writeln!(f, "chain\tdraw\t{}", names.join("\t")).unwrap();
    for (chain_id, draws) in chains {
        for (draw_idx, row) in draws.iter().enumerate() {
            let vals: Vec<String> = row.iter().map(|v| format!("{:.17e}", v)).collect();
            writeln!(f, "{}\t{}\t{}", chain_id, draw_idx, vals.join("\t")).unwrap();
        }
    }
    // Explicit flush: BufWriter swallows write errors on drop.
    f.flush()
        .map_err(|e| format!("cannot write {}: {}", path.display(), e))
}
