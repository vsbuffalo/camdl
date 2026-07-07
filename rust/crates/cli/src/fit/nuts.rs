//! `nuts` on `ode` stage runner (gh#275 Phase 2).
//!
//! Gradient-based Bayesian sampling of the deterministic ODE marginal likelihood
//! via `sim::inference::ode_nuts::run_ode_nuts` (which composes `det_grad` with
//! the prior + Jacobian into a NUTS target). Leaner than the PGAS stage — no
//! particles, no CSMC, no tempering; just per-chain NUTS and the shared trace /
//! fit_state output. On a stochastic backend, gradient-NUTS lives inside `pgas`,
//! so this path is `ode`-only (routed by the method registry).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::ode_nuts::{run_ode_nuts, OdeNutsConfig};
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
}

impl NutsStageOpts {
    pub fn from_stage(stage: &super::config_v2::Stage) -> Result<Self, String> {
        match stage {
            super::config_v2::Stage::Nuts {
                chains, warmup, samples, max_tree_depth, target_accept, dense_mass,
                init_method, survey_path, survey_top_k_n, ..
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

    // Chains are independent (own seed, own RNG, own trace file) and their outputs
    // reduce order-independently (max-loglik chain + summed divergences), so run
    // them in parallel across the rayon pool — the same pattern PGAS/PMMH/IF2 use.
    // Parallelism does not change results: each chain's seed and draws are fixed.
    use rayon::prelude::*;
    struct ChainOut {
        chain_id: usize,
        n_divergent: usize,
        best_loglik: f64,
        best_params: Vec<f64>,
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
                dt,
                // Independent chains: same start, distinct RNG stream.
                seed: seed.wrapping_add(chain_id as u64),
            };
            let result = run_ode_nuts(
                &compiled, &obs_model, &obs_times, &config.base_params,
                &config.estimated_params, &priors, &cfg,
            )
            .map_err(|e| format!("nuts chain {} error: {}", chain_id + 1, e))?;

            // Trace: one row per draw — data loglik, log posterior, per-draw
            // divergence flag, then the estimated parameters (natural scale).
            // `TraceWriter` writes `draw`, `log_likelihood`, `log_posterior` as its
            // three fixed leading columns; `divergent` is the only extra.
            let trace_path = chain_dir.join("trace.tsv");
            let writer = super::trace_writer::TraceWriter::new(
                &trace_path.to_string_lossy(),
                "draw",
                "log_likelihood",
                &["divergent"],
                &param_names,
                /* append */ false,
            );
            let mut best_loglik = f64::NEG_INFINITY;
            let mut best_params = config.base_params.clone();
            for (i, sample) in result.samples.iter().enumerate() {
                let div = if result.sample_divergent[i] { "1" } else { "0" };
                writer.write_row(i, result.sample_loglik[i], result.sample_logpost[i], &[div], sample);
                if result.sample_loglik[i] > best_loglik {
                    best_loglik = result.sample_loglik[i];
                    let mut p = config.base_params.clone();
                    for (j, ep) in config.estimated_params.iter().enumerate() {
                        p[ep.index] = sample[j];
                    }
                    best_params = p;
                }
            }
            drop(writer); // flush the buffered trace to disk

            let status = format!(
                "chain {} · {} draws · {} divergent · accept={:.2} · step={:.4}",
                chain_id + 1,
                result.samples.len(),
                result.n_divergent,
                result.mean_accept,
                result.step_size
            );
            Ok(ChainOut {
                chain_id,
                n_divergent: result.n_divergent,
                best_loglik,
                best_params,
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
