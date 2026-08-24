//! `camdl fit pmmh` — PMMH posterior sampling.
//!
//! Runs multiple MCMC chains in parallel, each using the bootstrap particle
//! filter as an unbiased likelihood estimator. Outputs per-chain trace files,
//! convergence diagnostics (R̂, ESS), and a summary JSON.

use crate::fit::state::FitState;
use crate::fit::loglik::LoglikType;
use crate::fit::runner::{self, FitRunConfig, StageConvergence};
use crate::cas::iso8601_utc;
use rayon::prelude::*;
use sim::inference::{
    if2::EstimatedParam,
    pmmh::{run_pmmh, Prior, PMMHConfig, PMMHResult, PMMHResumeState},
    diagnostic::{DiagnosticCollector, DiagnosticKind},
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-stage knobs extracted from a `Stage::PMMH { ... }` variant by
/// the `camdl fit run` dispatcher and passed verbatim into `run_stage`.
/// Mirrors every PMMH field. v1 calls iterations `steps`; v2 calls
/// them `iterations` (we keep the internal `n_steps` name to match
/// `sim::PMMHConfig.n_steps`).
pub struct PmmhStageOpts {
    pub n_chains: usize,
    pub n_particles: usize,
    pub n_steps: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub adapt: bool,
    pub adapt_start: usize,
    pub rho: Option<f64>,
    pub init_method: super::init::InitMethod,
    /// Survey CAS directory consumed when
    /// `init_method = InitMethod::SurveyTopK` (gh#51 v2). `None`
    /// for other init methods. The dispatcher fills this from the
    /// stage TOML (`survey_path = "..."` on `[stages.X]`) or the CLI
    /// override (`--survey-path`).
    pub survey_path: Option<std::path::PathBuf>,
    /// Top-K count for `init_method = SurveyTopK`. `None` → defaults
    /// to `chains`. v2 enforces `top_k == chains` (strict K=chains;
    /// K > chains with stratified sub-sampling is v3).
    pub survey_top_k_n: Option<usize>,
}

pub(crate) const DEFAULT_BURN_IN: usize = 5000;
const DEFAULT_THIN: usize = 10;

impl PmmhStageOpts {
    /// Build from a `Stage::PMMH { ... }` variant. Errors if `stage` is
    /// not the PMMH variant — caller's responsibility to dispatch.
    pub fn from_stage(stage: &super::config_v2::Stage) -> Result<Self, String> {
        match stage {
            super::config_v2::Stage::PMMH {
                chains, particles, iterations, burn_in, thin,
                adapt, adapt_start, rho, init_method,
                survey_path, survey_top_k_n,
                ..
            } => {
                if let Some(r) = rho {
                    if !(0.0..1.0).contains(r) {
                        return Err(format!(
                            "stage rho must be in [0, 1) for correlated \
                             pseudo-marginal MCMC. Got: {}", r));
                    }
                }
                Ok(PmmhStageOpts {
                    n_chains: *chains,
                    n_particles: *particles,
                    n_steps: *iterations,
                    burn_in: burn_in.unwrap_or(DEFAULT_BURN_IN),
                    thin: thin.unwrap_or(DEFAULT_THIN),
                    adapt: *adapt,
                    adapt_start: *adapt_start,
                    rho: *rho,
                    init_method: init_method.clone(),
                    survey_path: survey_path.clone(),
                    survey_top_k_n: *survey_top_k_n,
                })
            }
            // Deterministic-ODE MH reuses the PMMH machinery via `run_stage`'s
            // `is_ode_mh` seam. It carries neither `particles` (no PF) nor
            // `rho` (no correlated pseudo-marginal noise), so `n_particles` is
            // 0 (unused on the deterministic path) and `rho` is None.
            super::config_v2::Stage::Mh {
                chains, iterations, burn_in, thin,
                adapt, adapt_start, init_method,
                survey_path, survey_top_k_n,
                ..
            } => {
                Ok(PmmhStageOpts {
                    n_chains: *chains,
                    n_particles: 0,
                    n_steps: *iterations,
                    burn_in: burn_in.unwrap_or(DEFAULT_BURN_IN),
                    thin: thin.unwrap_or(DEFAULT_THIN),
                    adapt: *adapt,
                    adapt_start: *adapt_start,
                    rho: None,
                    init_method: init_method.clone(),
                    survey_path: survey_path.clone(),
                    survey_top_k_n: *survey_top_k_n,
                })
            }
            other => Err(format!(
                "PmmhStageOpts::from_stage: expected Stage::PMMH or Stage::Mh, got {}",
                other.method_name())),
        }
    }
}

// See pgas::run_stage for the comment on this allow.
#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    fit: &super::config_v2::FitConfigV2,
    stage_name: &str,
    stage: &super::config_v2::Stage,
    stage_dir: &Path,
    pmmh_opts: PmmhStageOpts,
    seed: u64,
    force: bool,
    resume: bool,
    starts_from: Option<&str>,
    // Post-fit deterministic ODE dt-check config (gh#52, gh#227). `Some` only on
    // the `mh` (ODE) dispatch; PMMH passes `None` (its dt-check is PF-based and
    // wired on the IF2 path). The bool is `--dt-check-strict`.
    dt_check_opt: Option<(super::config_v2::DtCheckConfig, bool)>,
) -> Result<(), String> {
    // The PMMH prefer-PGAS-for-long-series caveat banner is emitted by the
    // dispatch chokepoint (`methods::emit_status_banner`), driven by the
    // registry `status_note` so it can't drift from `camdl fit methods`.
    let collector = DiagnosticCollector::new("pmmh");
    let estimate = &fit.estimate;

    let n_chains = pmmh_opts.n_chains;
    let n_steps = pmmh_opts.n_steps;
    let n_particles = pmmh_opts.n_particles;
    let burn_in = pmmh_opts.burn_in;
    let thin = pmmh_opts.thin;
    let adapt = pmmh_opts.adapt;
    let adapt_start = pmmh_opts.adapt_start;

    // Deterministic-ODE Metropolis-Hastings: the stage is `Stage::Mh`. The MH
    // chain/adaptive-proposal/diagnostics machinery below is shared with PMMH;
    // the ONLY difference is the per-step likelihood evaluation — the
    // deterministic `compute_ode_loglik` instead of the bootstrap particle
    // filter. With no PF there is no correlated pseudo-marginal noise, so `rho`
    // is forced to None (skips the CPM obs-grid preflight and the correlated
    // evaluator) and the PF-variance preflight is skipped entirely.
    let is_ode_mh = matches!(stage, super::config_v2::Stage::Mh { .. });
    let rho: Option<f64> = if is_ode_mh { None } else { pmmh_opts.rho };

    // Load prior state if --starts-from provided
    let prior_state = starts_from.map(FitState::load).transpose()?;

    // Build FitRunConfig (reuse existing builder). iterations,
    // cooling, cooling_target_iters are IF2-specific and never read
    // by PMMH — pass harmless values.
    let config = FitRunConfig::build(
        fit, prior_state.as_ref(),
        n_chains, n_particles, 1,
        1.0, 1,
        seed, false,
    )?;

    let dt = config.if2_config.dt;

    // Deterministic-ODE MH eval inputs, built once and shared (read-only)
    // across all chains. `obs_model` goes behind an `Arc` so each chain's
    // per-step closure can borrow it cheaply for the parallel chain loop. Only
    // populated on the `is_ode_mh` path; PMMH leaves these unused.
    let ode_obs_model: Option<std::sync::Arc<sim::inference::MultiStreamObsModel>> =
        if is_ode_mh {
            Some(std::sync::Arc::new(config.build_obs_model()))
        } else {
            None
        };
    let ode_obs_times: Vec<f64> = if is_ode_mh {
        config.observations.iter().map(|o| o.time).collect()
    } else {
        Vec::new()
    };
    let ode_dt: f64 = if is_ode_mh { runner::ode_step_dt(&config) } else { dt };
    // gh#396 follow-on: coarse warm-up step for the deterministic ODE-MH
    // likelihood. Validated up front (same soundness rules as `nuts`: prevalence
    // only, genuine warm-up window, step LARGER than dt) so a misconfigured fit
    // fails before any chain runs; the per-substep `events`/`balance` refusal is
    // enforced downstream in `run_ode`. `None` ⇒ `ode_dt` (off).
    let ode_burnin_dt: f64 = if is_ode_mh {
        let burnin_opt = match stage {
            super::config_v2::Stage::Mh { burnin_dt, .. } => *burnin_dt,
            _ => None,
        };
        let n_interval = ode_obs_model.as_ref().map_or(0, |m| m.n_interval_streams());
        super::config_v2::validate_burnin_dt(
            burnin_opt,
            ode_dt,
            n_interval,
            ode_obs_times.first().copied(),
            config.compiled.model.simulation.t_start,
        )?
    } else {
        ode_dt
    };

    // gh#193 preflight: correlated PMMH (CPM, rho > 0) pre-draws a fixed-size
    // noise block per observation window and so requires a (near-)uniform obs
    // grid. The check is θ-independent (obs grid only) — run it ONCE here and
    // surface the actionable message, instead of letting every per-step PF eval
    // swallow the filter Err into -inf (a silent all-(-inf) chain). A leading
    // window coinciding with t_start is fine; see validate_cpm_obs_grid.
    // Skipped for ODE-MH (rho is forced None: there is no correlated PF).
    if rho.is_some() {
        let obs_times: Vec<f64> = config.observations.iter().map(|o| o.time).collect();
        sim::inference::correlated_pf::validate_cpm_obs_grid(
            &obs_times, config.smc_config().t_start, dt,
        ).map_err(|e| e.to_string())?;
    }

    // Build proposal SDs
    let proposal_sd = build_proposal_sd(&config, starts_from)?;

    // Preflight: PF variance check (skipped for ODE-MH — deterministic, no PF).
    if is_ode_mh {
        eprintln!("\nODE marginal-likelihood check at base θ (deterministic)...");
    } else {
        eprintln!("\npfilter variance check ({} particles, 20 replicates)...", n_particles);
    }
    let base = prior_state.as_ref().map(|s| {
        let mut p = config.base_params.clone();
        for spec in &config.estimated_params {
            if let Some(&v) = s.start_values.get(&spec.name) {
                p[spec.index] = v;
            }
        }
        p
    }).unwrap_or_else(|| config.base_params.clone());

    // Per-chain starting parameters (gh#42, gh#51 v2).
    // Precedence:
    // 1. `--starts-from` — every chain at the prior MLE (`base`).
    //    Mutually exclusive with `init = "survey_top_k"`: `init_mle`
    //    already commits every chain to the same point (the scout MLE),
    //    so any sibling survey_top_k seed would be silently overwritten.
    //    Refuse early instead.
    // 2. `init = "survey_top_k"` (gh#51 v2) — resolved here via the
    //    shared helper. Requires `survey_path = "..."` set on the
    //    stage or via CLI override.
    // 3. `init` dispatch on Lhs / Uniform / Single. Default `lhs` gives
    //    stratified posterior coverage at low chain counts. `Single`
    //    and `Uniform`-with-n_chains=1 return None; we then materialise
    //    N copies of `base`.
    let mut survey_top_k_result: Option<super::init::SurveyTopKResult> = None;
    let chain_starts: Vec<Vec<f64>> = if prior_state.is_some() {
        if pmmh_opts.init_method == super::init::InitMethod::SurveyTopK {
            return Err(format!(
                "pmmh stage `{}`: --starts-from / `init_mle = \"...\"` and \
                 `init = \"survey_top_k\"` are mutually exclusive — \
                 the former commits every chain to the prior MLE, so any \
                 survey-seeded start would be silently overwritten. Pick one: \
                 drop `init_mle`, or use a non-survey `init`.",
                stage_name));
        }
        vec![base.clone(); n_chains]
    } else if pmmh_opts.init_method == super::init::InitMethod::SurveyTopK {
        // Compute the fit-level cross-check context. Mirrors what the
        // IF2 dispatch site does; see gh#51 §"Validation".
        let model_identity_str = crate::resolve::model_identity_from_ir(&config.model_ir_json);
        let data_spec = fit.data_spec()?;
        let model_obs_names: Vec<String> = config.model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let effective_obs = data_spec.effective_observations(&model_obs_names)?;
        let data_hashes = super::init::compute_data_hashes(&effective_obs)?;
        let estimate_names: Vec<String> = fit.estimate.keys().cloned().collect();
        let fixed_for_ctx = fit.fixed.resolve()?;
        let fixed_hashmap: std::collections::HashMap<String, f64> =
            fixed_for_ctx.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let ctx = super::init::SurveyFitContext {
            model_identity: &model_identity_str,
            data_hashes: &data_hashes,
            fixed: &fixed_hashmap,
            estimate_names: &estimate_names,
        };
        let (chains_opt, result) =
            super::init::resolve_per_chain_starts_from_method(
                &pmmh_opts.init_method,
                pmmh_opts.survey_path.as_deref(),
                pmmh_opts.survey_top_k_n,
                stage_name,
                &config.estimated_params,
                n_chains,
                seed,
                &ctx,
                None,
            ).map_err(|e| format!("pmmh: {}", e))?;
        let chains_specs = chains_opt
            .expect("SurveyTopK must yield per-chain starts");
        survey_top_k_result = result;
        super::init::chain_starts_to_param_vecs(&chains_specs, &base)
    } else if matches!(pmmh_opts.init_method,
        super::init::InitMethod::FromPrior
        | super::init::InitMethod::FromPosterior { .. }
        | super::init::InitMethod::FromMle    { .. }
        | super::init::InitMethod::FromParams { .. })
    {
        // Step 7 warm-start dispatch (gh#83/gh#85). See pgas.rs for
        // the mirror path; same shape, different stage type.
        let resolved_view = super::init::build_resolved_view_for_init(
            &config.model, &base, &config.estimated_params,
        );
        let starts = crate::fit::chain_starts::draw_chain_starts(
            &resolved_view, &pmmh_opts.init_method, n_chains, seed,
        ).map_err(|e| format!("pmmh: --init {}: {}",
            pmmh_opts.init_method, e))?;
        let chains_specs = starts.to_estimated_params(&config.estimated_params);
        super::init::chain_starts_to_param_vecs(&chains_specs, &base)
    } else {
        super::init::build_chain_param_vecs(
            &pmmh_opts.init_method,
            &config.estimated_params,
            &base,
            n_chains,
            seed,
        ).map_err(|e| format!("pmmh: {}", e))?
        .unwrap_or_else(|| vec![base.clone(); n_chains])
    };

    let ll_mean: f64;
    if is_ode_mh {
        // Deterministic ODE marginal likelihood: a single eval at base θ — no
        // replicates, no variance (the ODE skeleton is deterministic). A
        // structural failure aborts (gh#224); a ruled-out θ (−∞) is reported as
        // the initial loglik for the FitState, exactly as PMMH reports its mean.
        let obs_model = ode_obs_model.as_ref()
            .expect("ode_obs_model built on the is_ode_mh path");
        ll_mean = match sim::inference::compute_ode_loglik(
            &config.compiled, obs_model, &ode_obs_times, ode_dt, &base, ode_burnin_dt,
        ) {
            Ok(ll) => ll,
            Err(e) if e.is_structural() =>
                return Err(format!(
                    "mh: structural error during ODE loglik check at base θ: {}", e)),
            Err(_) => f64::NEG_INFINITY,
        };
        eprintln!("  ODE log L = {:.1} (deterministic; no PF variance)", ll_mean);
    } else {
        let logliks: Vec<f64> = (0..20)
            .map(|i| runner::run_quick_pfilter(&config, &base, n_particles, seed + i))
            .collect::<Result<Vec<f64>, _>>()
            .map_err(|e| format!("pmmh: structural error during PF-variance check at base θ: {}", e))?;
        let mean = logliks.iter().sum::<f64>() / logliks.len() as f64;
        let ll_var = logliks.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / (logliks.len() - 1) as f64;
        let ll_sd = ll_var.sqrt();
        ll_mean = mean;

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
    }


    if !force && !resume {
        let state_path = stage_dir.join("fit_state.toml");
        if state_path.exists() {
            eprintln!("\x1b[33mpmmh results already exist in {}. Use --force to re-run or --resume to continue.\x1b[0m",
                stage_dir.display());
            return Ok(());
        }
    }

    std::fs::create_dir_all(stage_dir)
        .map_err(|e| format!("cannot create {}: {}", stage_dir.display(), e))?;

    // Write chain_starts.tsv sidecar for audit (gh#51 v2). Best-effort;
    // failure logs but does not abort the fit. The `chain_starts`
    // vector built above is `Vec<Vec<f64>>` (per-chain full param
    // vectors); the writer accepts the IF2-shaped per-chain
    // EstimatedParam slice instead, so we rebuild the spec view here.
    // For non-survey modes, `survey_top_k_result` is `None` and the
    // writer uses the `<method>:chain-<id>` source convention.
    let per_chain_specs_for_audit: Vec<Vec<EstimatedParam>> = chain_starts.iter()
        .map(|params| config.estimated_params.iter()
            .map(|spec| EstimatedParam {
                initial: params[spec.index], ..spec.clone()
            })
            .collect())
        .collect();
    if let Err(e) = super::init::write_chain_starts_tsv(
        stage_dir,
        &config.estimated_params,
        Some(&per_chain_specs_for_audit),
        n_chains,
        &pmmh_opts.init_method,
        survey_top_k_result.as_ref(),
    ) {
        eprintln!("warning: could not write chain_starts.tsv: {}", e);
    }

    // Resolve priors: fit.toml override → model IR → Flat
    let priors: Vec<Prior> = config.estimated_params.iter()
        .map(|spec| super::runner::resolve_prior(&spec.name, estimate, &config.model).0)
        .collect();

    // Active interventions + events — see same block in pgas.rs.
    crate::util::print_scheduled_actions_summary(&config.model_declared, &config.model);
    crate::util::print_observations_summary(&config.model);

    let dt = config.if2_config.dt;

    // Compute config hash — identifies the statistical problem.
    // Uses the same provenance::fit_stage_hash that the v2 dispatch
    // site uses for cache-hit checks; resume only succeeds when the
    // (model + observations + estimate + fixed + stage_name + Stage
    // variant + seed) tuple is unchanged.
    let fixed_resolved = fit.fixed.resolve()?;
    let data_spec = fit.data_spec()?;
    let config_hash = super::provenance::fit_stage_hash(
        &config.model_ir_json, &data_spec.observations,
        &fit.estimate, &fixed_resolved, &fit.simplex_groups,
        stage_name, stage, seed,
    )?;

    // Load resume states if --resume
    let resume_states: Vec<Option<PMMHResumeState>> = if resume {
        let mut states = Vec::with_capacity(n_chains);
        let mut any_failed = false;
        for chain_id in 0..n_chains {
            let path: PathBuf = stage_dir.join(format!("chain_{}", chain_id + 1))
                .join("resume_state.bin");
            match std::fs::read(&path) {
                Ok(data) => match bincode::deserialize::<PMMHResumeState>(&data) {
                    Ok(state) => {
                        if state.config_hash != config_hash {
                            eprintln!("error: config hash mismatch for chain {} — \
                                cannot resume. Either the model/data/priors changed \
                                since the original run, or this chain predates a camdl \
                                version that changed how a stage's identity is computed \
                                (the 2026-08-23 subtractive-identity change re-keyed \
                                every pgas/pmmh/mh/nuts stage). Re-run from scratch \
                                with --force.",
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
                    eprintln!("error: no resume state file for chain {} ({})",
                        chain_id + 1, path.display());
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

    // One `Reporter` hands out a per-chain `Task` rendered as a coordinated
    // stack. The Reporter honors --progress (Pretty=bars, Plain=throttled
    // `chain N pos/len ll=… acc=…%` log lines, None=silent), so the callback
    // no longer branches on mode.
    let reporter = crate::progress::Reporter::new();
    let bars: Vec<crate::progress::Task> = (0..n_chains)
        .map(|chain_id| reporter.task(n_steps as u64, format!("chain {}", chain_id + 1), "it"))
        .collect();

    // Pre-create chain directories
    for chain_id in 0..n_chains {
        let chain_dir = stage_dir.join(format!("chain_{}", chain_id + 1));
        std::fs::create_dir_all(&chain_dir)
            .map_err(|e| format!("cannot create {}: {}", chain_dir.display(), e))?;
    }

    let t0 = std::time::Instant::now();

    // Run chains in parallel. Each chain yields `Ok(Some(result))`, or
    // `Ok(None)` when its init-eval surfaces a `PFDegenerate` bail (gh#110)
    // — skipped with a BadInit diagnostic and omitted from downstream
    // R̂/ESS/MAP aggregation, surviving chains continue ("skip + continue",
    // so a 6-chain run with one bound-pathological survey_top_k init still
    // gives 5 chains of inference). A `Err` is a structural failure
    // (gh#224): the model/config cannot run, so the whole fit aborts rather
    // than reporting a degenerate posterior — `collect` short-circuits on
    // the first such error.
    let results: Vec<(usize, PMMHResult)> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| -> Result<Option<(usize, PMMHResult)>, String> {
            let chain_seed = crate::util::derive_chain_seed(seed, chain_id);

            // gh#110 init-eval guard. Run a single PF at the chain's
            // starting θ to verify it isn't in the PF-degenerate
            // region (R₀ ≈ 50, σ at the upper bound, etc.). On
            // Err(PFDegenerate) push a BadInit diagnostic and skip
            // this chain. We only fire on the *first* eval — once
            // the chain is past init, PMMH's MH ratio handles
            // -∞ proposals via the existing reject path.
            //
            // Skip the guard on resume: the resumed θ already
            // passed init once; rerunning the eval would waste a
            // PF call and (if the chain had already diverged into
            // a degenerate region during sampling) spuriously
            // mark a working chain as bad.
            if resume_states[chain_id].is_none() && is_ode_mh {
                // ODE-MH init-eval guard: no PF, so the PFDegenerate skip arm
                // does not apply. Evaluate the
                // deterministic loglik once at the chain's start; a structural
                // error aborts the whole fit (gh#224, same hard path as PMMH's
                // structural arm), and any other outcome (Ok, including −∞)
                // proceeds — MH's accept/reject handles an uninformative init.
                let obs_model = ode_obs_model.as_ref()
                    .expect("ode_obs_model built on the is_ode_mh path");
                match sim::inference::compute_ode_loglik(
                    &config.compiled, obs_model, &ode_obs_times, ode_dt,
                    &chain_starts[chain_id], ode_burnin_dt,
                ) {
                    Err(e) if e.is_structural() => {
                        return Err(format!(
                            "chain {} init-eval failed with structural error: {}",
                            chain_id + 1, e));
                    }
                    Ok(ll) if !ll.is_finite() => {
                        // gh#334: a −∞ init means the start θ predicts a
                        // trajectory that cannot explain the data (e.g. an
                        // epidemic that goes extinct, zeroing the modelled
                        // incidence where the data is positive). This is a bad
                        // START in parameter space, not a bad seed — reseeding
                        // would silently change the realized draws and hide the
                        // seed→draws relationship, so we WARN and leave the fix
                        // to the user. With the −∞-escape fix (mh_accept) the
                        // chain CAN still recover if a finite-likelihood region
                        // is within proposal reach; a persistent 0% acceptance
                        // means the start itself is the problem.
                        eprintln!(
                            "  \x1b[33mwarning\x1b[0m: chain {} starts at a θ with \
                             -inf log-likelihood — the start predicts a trajectory that \
                             can't explain the data (e.g. an epidemic that goes extinct \
                             where the data has cases). The chain can only mix if a \
                             finite-likelihood region is reachable from there; if its \
                             acceptance rate stays 0%, propose a better start (a \
                             multi-start `init`, or adjust the start toward a θ that \
                             sustains the epidemic).",
                            chain_id + 1);
                    }
                    _ => {
                        // Ok(finite) or a recoverable Err (ruled-out θ) — MH
                        // proceeds via the standard accept/reject path.
                    }
                }
            } else if resume_states[chain_id].is_none() {
                match runner::run_quick_pfilter_with_dt(
                    &config, &chain_starts[chain_id],
                    n_particles, None, chain_seed,
                ) {
                    Err(e @ sim::error::SimError::PFDegenerate { .. }) => {
                        // A statistically-degenerate init (ESS collapse / all
                        // particles dead) is skipped with a BadInit diagnostic;
                        // the surviving chains continue. (PFIterationBudget is a
                        // deterministic compute-budget bail → structural/fatal,
                        // handled by the structural arm.)
                        let reason = match &e {
                            sim::error::SimError::PFDegenerate { kind, obs_window, elapsed_s } =>
                                format!("{:?} at obs_window={} after {:.2}s",
                                    kind, obs_window, elapsed_s),
                            _ => unreachable!(),
                        };
                        let params: std::collections::BTreeMap<String, f64> =
                            config.estimated_params.iter()
                                .map(|spec| (
                                    spec.name.clone(),
                                    chain_starts[chain_id][spec.index],
                                ))
                                .collect();
                        collector.push(DiagnosticKind::BadInit {
                            chain_id, params, reason: reason.clone(),
                        });
                        eprintln!("  chain {}: \x1b[31m✗ BadInit\x1b[0m — {}",
                            chain_id + 1, reason);
                        // The bar is cleared in the post-loop finish; the skip
                        // is already loud on stderr above.
                        return Ok(None);
                    }
                    Err(other) => {
                        // Non-degeneracy structural error — surface as
                        // a hard failure rather than a skip. These are
                        // config bugs (UnknownCompartment, etc.) that
                        // every chain would hit, so abort the whole fit
                        // (gh#224: a structural error must not be hidden
                        // as a ruled-out θ). `collect` short-circuits.
                        return Err(format!(
                            "chain {} init-eval failed with structural error: {}",
                            chain_id + 1, other));
                    }
                    Ok(_) => {
                        // Finite (or even -inf-but-not-degenerate)
                        // initial loglik — PMMH can proceed. PMMH's
                        // MH ratio handles uninformative inits via
                        // the standard accept/reject path.
                    }
                }
            }

            let pmmh_config = PMMHConfig {
                n_steps,
                n_particles,
                dt,
                proposal_sd: proposal_sd.clone(),
                adapt,
                adapt_start,
                thin,
                burn_in,
                rho,
                n_source_groups: config.compiled.source_groups.len(),
            };

            // Build the loglik evaluator closure for this chain. Both branches
            // return the inference convention (Ok(−∞) = θ ruled out, Err =
            // structural — gh#224). For ODE-MH this is the deterministic
            // `compute_ode_loglik` (the seed is unused — the ODE skeleton is
            // deterministic); otherwise the bootstrap-PF `run_quick_pfilter`.
            // Boxed (lifetime-bounded, `+ '_`) so the two closure types unify
            // into one binding while still borrowing local state — the PF
            // branch borrows `&config` exactly as the prior non-boxed closure
            // did, so the PMMH path is unchanged.
            let eval_loglik: Box<dyn Fn(&[f64], u64) -> Result<f64, sim::error::SimError> + '_> =
                if is_ode_mh {
                    Box::new(|params: &[f64], _seed: u64| -> Result<f64, sim::error::SimError> {
                        let obs_model = ode_obs_model.as_ref()
                            .expect("ode_obs_model built on the is_ode_mh path");
                        match sim::inference::compute_ode_loglik(
                            &config.compiled, obs_model, &ode_obs_times, ode_dt, params, ode_burnin_dt,
                        ) {
                            Ok(ll) => Ok(ll),
                            Err(e) if e.is_structural() => Err(e),
                            Err(_) => Ok(f64::NEG_INFINITY),
                        }
                    })
                } else {
                    Box::new(|params: &[f64], pf_seed: u64| -> Result<f64, sim::error::SimError> {
                        runner::run_quick_pfilter(&config, params, n_particles, pf_seed)
                    })
                };

            // Correlated PF evaluator (when rho is set)
            let process = config.build_process();
            let obs_model_trait = config.build_obs_model();
            let smc_cfg = config.smc_config();
            let eval_correlated: Option<Box<dyn Fn(&[f64], &sim::inference::correlated_pf::PFRandomState) -> Result<f64, sim::error::SimError>>> =
                if pmmh_config.rho.is_some() {
                    Some(Box::new(move |params: &[f64], randoms: &sim::inference::correlated_pf::PFRandomState| -> Result<f64, sim::error::SimError> {
                        // gh#224: structural failures surface; a degenerate or
                        // recoverable correlated-PF run is a ruled-out θ (−∞).
                        match sim::inference::correlated_pf::bootstrap_filter_correlated(
                            &process, &obs_model_trait, params, &smc_cfg, randoms, chain_seed,
                        ) {
                            Ok(r) => Ok(r.log_likelihood),
                            Err(e) if e.is_structural() => Err(e),
                            Err(_) => Ok(f64::NEG_INFINITY),
                        }
                    }))
                } else {
                    None
                };

            let eval_corr_ref: Option<&dyn Fn(&[f64], &sim::inference::correlated_pf::PFRandomState) -> Result<f64, sim::error::SimError>> =
                eval_correlated.as_deref();

            let task = &bars[chain_id];
            let accepted_count = AtomicUsize::new(0);

            // Streaming trace: use TraceWriter with append mode when resuming
            let chain_dir = stage_dir.join(format!("chain_{}", chain_id + 1));
            let _ = std::fs::create_dir_all(&chain_dir);
            let trace_path = chain_dir.join("trace.tsv");
            let trace_path_str = trace_path.to_string_lossy().into_owned();
            let is_resuming = resume_states[chain_id].is_some();
            let param_names: Vec<String> = config.estimated_params.iter()
                .map(|s| s.name.clone()).collect();
            let trace_writer = super::trace_writer::TraceWriter::new(
                &trace_path_str, "step", "log_likelihood", &["accepted"],
                &param_names, is_resuming,
            );

            let progress_cb = |step: usize, loglik: f64, accepted: bool, params: &[f64]| {
                if accepted { accepted_count.fetch_add(1, Ordering::Relaxed); }
                let acc = accepted_count.load(Ordering::Relaxed) as f64 / (step + 1) as f64;

                // Stream trace row to disk. Warm-up (step < burn_in) rows are
                // emitted every step for live burn-in observability (a tailing
                // watcher sees the chain moving during warm-up); the sampling
                // phase keeps its exact burn-in/thin cadence, so the set of rows
                // with `step >= burn_in` is byte-identical to before. The
                // posterior filters warm-up back out: `draws.tsv` on `step >=
                // burn_in` (below), and R̂/ESS/acceptance/MAP off the in-memory
                // post-burn-in `steps` (never this raw trace).
                let is_warmup = step < burn_in;
                let is_sampling_draw =
                    step >= burn_in && (step - burn_in).is_multiple_of(thin);
                if is_warmup || is_sampling_draw {
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

                // Passive bar tick. The callback fires once per step in order,
                // so `inc(1)` tracks position = step+1 exactly. `Task` handles
                // Pretty (redraw) / Plain (throttled `chain N pos/len ll=… acc=…%`
                // line) / None (no-op) — no mode branching here.
                task.set(crate::progress::mcmc(loglik, acc));
                task.inc(1);
            };

            // gh#224: a structural error from the sampling PF aborts the fit
            // (config/model can't run); a ruled-out θ is handled internally as
            // a rejected −∞ proposal, so `run_pmmh` only `Err`s on structural.
            let result = run_pmmh(
                &config.estimated_params, &priors, &chain_starts[chain_id],
                &config.param_names,
                &pmmh_config, &config.observations, eval_loglik.as_ref(), eval_corr_ref, chain_seed,
                Some(&progress_cb), resume_states[chain_id].clone(), config_hash.clone(),
            ).map_err(|e| format!("chain {} failed with structural error: {}", chain_id + 1, e))?;

            // Final metric (MAP ll + acceptance) on the bar; the driver clears
            // it after the par_iter (`Task::finish` consumes, so it can't run
            // on the borrowed Task here). The `acceptance rates:` report below
            // carries the per-chain summary.
            task.set(crate::progress::mcmc(result.map_loglik, result.acceptance_rate));

            // Save resume state for future --resume
            let resume_path = chain_dir.join("resume_state.bin");
            if let Ok(encoded) = bincode::serialize(&result.resume_state) {
                let _ = std::fs::write(&resume_path, encoded);
            }

            Ok(Some((chain_id, result)))
        })
        .collect::<Result<Vec<Option<(usize, PMMHResult)>>, String>>()?
        .into_iter()
        .flatten()
        .collect();

    // Clear all chain bars now that the parallel phase is done (`Task::finish`
    // consumes, so it can't run on the per-chain borrow inside the loop). The
    // `acceptance rates:` report below carries the per-chain summary.
    for t in bars { t.finish(); }

    // gh#110. Skip + continue: surface "ran K of N chains" so the
    // user knows downstream R̂/ESS exclude skipped chains. This
    // line is the load-bearing user-facing signal that bad inits
    // were tolerated rather than silently dropped.
    let n_good_chains = results.len();
    if n_good_chains < n_chains {
        eprintln!("\n\x1b[33mran {} of {} chains\x1b[0m \
                   ({} skipped via BadInit; see diagnostics below)",
            n_good_chains, n_chains, n_chains - n_good_chains);
    }

    let elapsed = t0.elapsed();

    // Traces already written by streaming callback — no post-hoc write needed.

    // Compute diagnostics
    let diagnostics = compute_diagnostics(&results, &config.estimated_params);

    // Report. PMMH's θ-move is random-walk MH, so its band is the random-walk
    // one — taken from `AcceptanceKernel` rather than spelled here, so the
    // coloring, the finding and the message a user reads cannot disagree about
    // where the band starts (gh#299 item 3).
    let kernel = sim::inference::diagnostic::AcceptanceKernel::RandomWalk;
    let (lo, hi) = kernel.healthy_band();
    eprintln!("\nacceptance rates:");
    for (chain_id, result) in &results {
        let status = if result.acceptance_rate < lo {
            "\x1b[31m✗ too low\x1b[0m"
        } else if result.acceptance_rate > hi {
            "\x1b[33m~ high\x1b[0m"
        } else {
            "\x1b[32m✓\x1b[0m"
        };
        eprintln!("  chain {}: {:.1}% {}", chain_id + 1, result.acceptance_rate * 100.0, status);
        if let Some(d) = sim::inference::diagnostic::acceptance_diagnostic(
            result.acceptance_rate, None, kernel)
        {
            collector.push(d);
        }
    }

    if n_chains > 1 {
        // RHAT_REPORT_THRESHOLD is unchanged from the value this stage has
        // always applied; only the STATISTIC it is applied to changed (gh#84).
        eprint!("{}", diagnostics.report(&collector, super::runner::RHAT_REPORT_THRESHOLD));
    }

    // gh#110. All chains skipped via BadInit — no MAP to report.
    // Render diagnostics, persist them, and bail with a clear error
    // rather than panicking on .unwrap() of an empty results vec.
    if results.is_empty() {
        collector.render_to_stderr();
        let diag_path = stage_dir.join("diagnostics.json");
        let _ = collector.write_json(&diag_path.to_string_lossy());
        return Err(format!(
            "pmmh stage `{}`: all {} chains failed init-eval with \
             PFDegenerate. See `diagnostics.json` for per-chain BadInit \
             diagnostics. Common causes: survey_top_k handed pathological \
             bound-pinned points to every chain; check `init` method or \
             widen parameter bounds.",
            stage_name, n_chains));
    }

    // gh#226. Whole-fit backstop: every surviving (non-skipped) chain
    // reached no finite log-likelihood anchor. Each individual `-inf` is
    // a correct "θ ruled out" (so `is_structural` rightly left it alone),
    // but with no finite anchor anywhere the MH chain cannot move
    // (`-inf - (-inf) = NaN`, never accepted) and the run would otherwise
    // write a degenerate posterior (acceptance 0, MAP loglik `-inf`) and
    // exit 0. Init `-inf` is just the special case where the absorbing
    // region starts at step 0, so this single check covers both halves of
    // the issue. Fires ONLY when NOT ONE chain is finite: a mixed run
    // where some inits are ruled out but at least one chain reached a
    // finite loglik still succeeds (that chain sets a finite MAP → the
    // predicate is false → no fire).
    if results.iter().all(|(_, r)| sim::inference::no_finite_anchor(r.map_loglik)) {
        collector.push(DiagnosticKind::InitialLoglikInfinite);
        collector.render_to_stderr();
        let diag_path = stage_dir.join("diagnostics.json");
        let _ = collector.write_json(&diag_path.to_string_lossy());
        return Err(format!(
            "pmmh stage `{}`: all {} surviving chain(s) reached no finite \
             log-likelihood anchor — every evaluated θ scored -inf, so the \
             MH chain never moves (-inf - (-inf) = NaN, never accepted) and \
             the posterior is degenerate. Most often the starting values sit \
             in an impossible region — an MH chain cannot escape a -inf init — \
             so check those first (try `--init lhs` or a different start); less \
             often the data are impossible under this model, or a recoverable \
             error fires at every θ (gh#226). Also check the observation model \
             and parameter bounds; run with --verbosity debug for per-substep \
             diagnostics.",
            stage_name, results.len()));
    }

    // Find MAP across surviving chains
    let (map_chain, map_result) = results.iter()
        .max_by(|a, b| a.1.map_log_posterior.total_cmp(&b.1.map_log_posterior))
        .unwrap();

    // Write summary JSON. Deterministic mh-ODE shares this runner but writes its
    // OWN `mh_summary.json`, not `pmmh_summary.json`.
    let algo = if is_ode_mh {
        crate::run_meta::FitAlgorithm::Mh
    } else {
        crate::run_meta::FitAlgorithm::Pmmh
    };
    write_summary(stage_dir, &results, &config, thin, &diagnostics, algo)?;

    // Write fit_state.toml
    let mut start_values = std::collections::BTreeMap::new();
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

    // Post-fit Richardson dt-convergence check at the MAP (gh#52, gh#227) — only
    // on the deterministic ODE-MH path. Re-evaluates `compute_ode_loglik(θ̂; dt)`
    // (the SAME likelihood the chain scored) on a dt-halving ladder and warns
    // when the MAP is discretization-dependent. PMMH passes `dt_check_opt = None`
    // (its dt-check is the PF-based one wired on the IF2 path). Reuses the
    // obs_model / obs_times / dt built once for the ODE-MH chain evals.
    let dt_check_result = match (is_ode_mh, &dt_check_opt) {
        (true, Some((cfg, strict))) => {
            let obs_model = ode_obs_model.as_deref()
                .expect("ode_obs_model built on the is_ode_mh path");
            let result = super::dt_check::run_richardson_ladder_ode(
                config.compiled.as_ref(),
                obs_model,
                &ode_obs_times,
                &map_result.map_params,
                ode_dt,
                cfg,
                *strict,
            )?;
            super::dt_check::print_terminal_report(&result);
            if matches!(result.verdict, super::dt_check::DtCheckVerdict::Skipped) {
                None
            } else {
                Some(result)
            }
        }
        _ => None,
    };

    let state = FitState {
        stage: stage_name.to_string(),
        seed,
        timestamp: iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik: map_result.map_loglik,
        initial_loglik: ll_mean,
        best_chain: *map_chain,
        n_chains,
        // gh#110. Surface "ran K of N chains" through fit_state so
        // downstream consumers (fit_summary, the book chapters) can
        // print "ran K of N chains" without inferring it from
        // missing chain dirs.
        n_good_chains: if n_good_chains < n_chains { Some(n_good_chains) } else { None },
        start_values,
        rw_sd: std::collections::BTreeMap::new(),
        loglik_type: Some(LoglikType::Marginal),
        acceptance_rate: Some(map_result.acceptance_rate),
        // Bayesian stages don't produce an IF2-style Â table; the
        // refine-gates proposal only gates on scout→refine handoffs.
        tail_chain_agreement: std::collections::BTreeMap::new(),
        ivp_params: Vec::new(),
        chain_logliks: Vec::new(),
        chain_eval_logliks: Vec::new(),
        chain_eval_ses: Vec::new(),
        // Bayesian path — compound gate doesn't apply to PMMH.
        resolved_gate: None,
        resolved_loglik_eval: None,
        // gh#51 v2: chain init provenance. When SurveyTopK was used,
        // emit the full survey hash + top-K via the shared formatter;
        // otherwise render the in-process sampler name verbatim.
        // SurveyTopK is dispatched via the shared
        // `resolve_per_chain_starts_from_method` helper above.
        chain_init_source: Some(super::init::format_chain_init_source(
            &pmmh_opts.init_method, survey_top_k_result.as_ref(),
        )),
        // gh#52, gh#227: deterministic ODE dt-check at the MAP (above); `None`
        // on the PMMH path (PF dt-check is wired on the IF2 path).
        dt_check: dt_check_result,
    };
    state.save(&stage_dir.to_string_lossy())?;

    // Write draws.tsv: complete-M posterior draws (all params, estimated + fixed)
    // Reads the per-chain trace.tsv files (which now also carry warm-up rows),
    // keeps only the post-burn-in tail (`step >= burn_in`), and adds fixed
    // parameter columns.
    {
        use std::io::Write;
        let draws_path = stage_dir.join("draws.tsv");
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&draws_path)
                .map_err(|e| format!("cannot create {}: {}", draws_path.display(), e))?
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
        // gh#322: leading `chain` `draw` key columns (the shared draws loader
        // strips them, so existing readers see param-only rows). PMMH/MH save no
        // latent path, so their fits are `NotSaved` (not forkable) for now and
        // `draw` is just a within-chain row index — no trajectory to join.
        writeln!(f, "chain\tdraw\t{}", all_names.join("\t")).unwrap();

        let fixed_vals: Vec<f64> = fixed_names.iter().map(|name| {
            config.compiled.param_index.get(name.as_str())
                .map(|&idx| config.base_params[idx])
                .unwrap_or(0.0)
        }).collect();

        // Read each chain's trace.tsv and extract param columns
        let mut n_draws = 0usize;
        for chain_id in 0..n_chains {
            let trace_path = stage_dir.join(format!("chain_{}", chain_id + 1))
                .join("trace.tsv");
            if let Ok(content) = std::fs::read_to_string(&trace_path) {
                let mut lines = content.lines();
                let header = lines.next().unwrap_or("");
                let cols: Vec<&str> = header.split('\t').collect();
                // Find column indices for estimated params
                let param_col_indices: Vec<usize> = est_names.iter().map(|name| {
                    cols.iter().position(|c| c == name).unwrap_or(usize::MAX)
                }).collect();
                // The trace now carries warm-up rows too (`step < burn_in`, for
                // live burn-in observability); the posterior is the post-burn-in
                // tail, so filter on the `step` index here. R̂/ESS/MAP read the
                // in-memory post-burn-in `steps`, not this file, so this is the
                // one raw-trace consumer that must exclude warm-up.
                let step_col = cols.iter().position(|c| *c == "step")
                    .expect("PMMH trace.tsv must carry a `step` index column");

                let mut draw_idx = 0usize;
                for line in lines {
                    if line.trim().is_empty() { continue; }
                    let fields: Vec<&str> = line.split('\t').collect();
                    let step: usize = match fields.get(step_col).and_then(|s| s.parse().ok()) {
                        Some(s) => s,
                        None => { eprintln!("warning: trace.tsv row missing step index; skipping"); continue; }
                    };
                    if step < burn_in { continue; }  // warm-up: not a posterior draw
                    let mut vals: Vec<String> = param_col_indices.iter().map(|&col_idx| {
                        if col_idx < fields.len() {
                            fields[col_idx].to_string()
                        } else {
                            eprintln!("warning: trace.tsv missing column at index {}", col_idx);
                            "NaN".to_string()
                        }
                    }).collect();
                    vals.extend(fixed_vals.iter().map(|v| format!("{:.17e}", v)));
                    // 0-based `chain`, matching trajectories.tsv's key convention
                    // (PMMH saves no path today, but keep the key consistent).
                    writeln!(f, "{}\t{}\t{}", chain_id, draw_idx, vals.join("\t")).unwrap();
                    draw_idx += 1;
                    n_draws += 1;
                }
            }
        }
        drop(f);
        eprintln!("  draws.tsv: {} posterior samples (all {} params)", n_draws, all_names.len());
    }

    // Render and persist diagnostics
    collector.render_to_stderr();
    let diag_path = stage_dir.join("diagnostics.json");
    let _ = collector.write_json(&diag_path.to_string_lossy());

    let wall_secs = elapsed.as_secs_f64();
    let total_pf_calls = n_chains * n_steps;
    eprintln!("\npmmh complete in {:.1}s ({} PF evaluations, {:.1}ms/eval): {}/",
        wall_secs, total_pf_calls,
        wall_secs * 1000.0 / total_pf_calls as f64 * n_chains as f64,
        stage_dir.display());
    eprintln!("  MAP loglik: {:.1} (chain {})", map_result.map_loglik, map_chain + 1);

    Ok(())
}

/// Build proposal SDs on the transformed scale.
///
/// v1's [pmmh] section let users point at a separate `proposal_from`
/// directory (independent from `starts_from`); v2's Stage::PMMH carries
/// only `starts_from` (toml key `init_mle`). So we use it for both — if
/// the user wants empirical covariance from scout, they wire that via
/// `init_mle = "scout"` on the PMMH stage.
fn build_proposal_sd(
    config: &FitRunConfig,
    starts_from: Option<&str>,
) -> Result<Vec<f64>, String> {
    if let Some(dir) = starts_from {
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

fn compute_diagnostics(
    results: &[(usize, PMMHResult)],
    estimated_params: &[EstimatedParam],
) -> StageConvergence {
    StageConvergence::compute(estimated_params.iter().map(|spec| {
        let chains: Vec<Vec<f64>> = results.iter()
            .map(|(_, r)| r.steps.iter().map(|s| s.params[spec.index]).collect())
            .collect();
        (spec.name.clone(), chains)
    }))
}

// write_chain_traces removed — streaming callback now handles trace output
// with correct log_posterior and burn-in/thin filtering.

fn write_summary(
    dir: &Path,
    results: &[(usize, PMMHResult)],
    config: &FitRunConfig,
    thin: usize,
    diagnostics: &StageConvergence,
    algo: crate::run_meta::FitAlgorithm,
) -> Result<(), String> {
    let acceptance_rates: Vec<f64> = results.iter().map(|(_, r)| r.acceptance_rate).collect();

    let (map_chain, map_result) = results.iter()
        .max_by(|a, b| a.1.map_log_posterior.total_cmp(&b.1.map_log_posterior))
        .unwrap();

    let map_params: HashMap<String, f64> = config.estimated_params.iter()
        .map(|spec| (spec.name.clone(), map_result.map_params[spec.index]))
        .collect();

    let mut summary = serde_json::json!({
        "stage": "pmmh",
        "n_chains": results.len(),
        "steps_per_chain": results.first().map(|(_, r)| r.n_steps).unwrap_or(0),
        "acceptance_rate": acceptance_rates,
        "map_loglik": map_result.map_loglik,
        "map_chain": map_chain + 1,
        "map_params": map_params,
        // n_samples (kept) × thin = raw sampling iterations → ESS/iteration.
        "thin": thin,
    });
    // Every convergence key comes from one producer, so a statistic cannot be
    // live in this summary and silently absent from pgas's or nuts's.
    summary.as_object_mut().expect("json! built an object")
        .extend(diagnostics.summary_fields());

    let path = dir.join(algo.summary_filename());
    let contents = serde_json::to_string_pretty(&summary)
        .map_err(|e| format!("json error: {}", e))?;
    std::fs::write(&path, contents)
        .map_err(|e| format!("cannot write {}: {}", path.display(), e))
}
