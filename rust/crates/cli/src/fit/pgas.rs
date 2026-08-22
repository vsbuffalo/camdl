//! `camdl fit pgas` — PGAS posterior sampling.
//!
//! Runs PGAS Gibbs sampler chains, each alternating exact parameter
//! updates (θ | X) with conditional SMC trajectory updates (X | θ, y).
//! Outputs per-chain trace files, convergence diagnostics, and summary.

use crate::fit::state::FitState;
use crate::fit::loglik::LoglikType;
use crate::fit::runner::{FitRunConfig, StageConvergence};
use crate::cas::iso8601_utc;
use sim::inference::{
    if2::EstimatedParam,
    pmmh::Prior,
    prior::Density,
    pgas::{PGASConfig, ChainResumeState, run_pgas, PGASSweep, PGASTrajectory, RENEWAL_BINS},
    diagnostic::{DiagnosticCollector, DiagnosticKind},
};
use io::trajectories::{
    Granularity, PosteriorDraw, TrajColumnSpec, TrajManifest, write_trajectories_tsv,
};
use io::progress::{Heartbeat, RunState};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

/// Trace column names for `CSMCDiagnostics::renewal_by_bin` (gh#688), one per
/// tenth of the substep series. The array type is `[&str; RENEWAL_BINS]`, so a
/// change to the bin count that this list does not follow is a compile error
/// rather than a silently mislabelled column.
const RENEWAL_BIN_COLUMNS: [&str; RENEWAL_BINS] = [
    "renewal_b0", "renewal_b1", "renewal_b2", "renewal_b3", "renewal_b4",
    "renewal_b5", "renewal_b6", "renewal_b7", "renewal_b8", "renewal_b9",
];

/// Per-stage knobs extracted from a `Stage::PGAS { ... }` variant by
/// the `camdl fit run` dispatcher and passed verbatim into `run_stage`.
/// Mirrors every PGAS field in `Stage::PGAS` plus burn_in/thin defaults.
#[derive(Debug)]
pub struct PgasStageOpts {
    pub n_chains: usize,
    pub n_particles: usize,
    pub n_sweeps: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub tempering: Vec<f64>,
    pub max_tree_depth: usize,
    pub trajectory_warmup: usize,
    pub csmc_sweeps_per_nuts: usize,
    pub n_trajectories: usize,
    pub dense_mass: bool,
    pub use_nuts: bool,
    pub init_method: super::init::InitMethod,
    /// Survey CAS directory consumed when
    /// `init_method = InitMethod::SurveyTopK` (gh#51 v2). `None`
    /// for other init methods. The dispatcher fills this from the
    /// stage TOML (`survey_path = "..."` on `[stages.X]`) or the CLI
    /// override (`--survey-path`).
    pub survey_path: Option<std::path::PathBuf>,
    /// Top-K count for `init_method = SurveyTopK`. `None` → defaults
    /// to `chains`. v2 enforces `top_k == chains`.
    pub survey_top_k_n: Option<usize>,
}

pub(crate) const DEFAULT_BURN_IN: usize = 2000;
const DEFAULT_THIN: usize = 5;

impl PgasStageOpts {
    /// Build from a `Stage::PGAS { ... }` variant. Errors if `stage` is
    /// not the PGAS variant — caller's responsibility to dispatch.
    pub fn from_stage(stage: &super::config_v2::Stage) -> Result<Self, String> {
        match stage {
            super::config_v2::Stage::PGAS {
                chains, particles, sweeps, burn_in, thin,
                tempering, max_tree_depth, trajectory_warmup,
                csmc_sweeps_per_nuts, n_trajectories,
                dense_mass, use_nuts, init_method,
                survey_path, survey_top_k_n,
                ..
            } => {
                if tempering.is_empty() || (tempering[0] - 1.0).abs() > 1e-9 {
                    return Err(format!(
                        "stage tempering ladder must start with β=1.0 \
                         (cold chain). Got: {:?}", tempering));
                }
                // Every entry must be in (0, 1]. β > 1 concentrates the
                // likelihood (sharper than the posterior); β ≤ 0 inverts
                // it (anti-annealing). Either way the chain converges
                // to the wrong target with no runtime error. Convention
                // is also a non-increasing ladder, but that's not
                // required for correctness — only the (0, 1] range is.
                // See docs/dev/reviews/2026-04-30-correctness.md H4.
                for (i, &beta) in tempering.iter().enumerate() {
                    if !(beta > 0.0 && beta <= 1.0) {
                        return Err(format!(
                            "tempering[{}] = {} is out of range (0, 1]; \
                             every β must be positive and ≤ 1.0. \
                             Got ladder: {:?}", i, beta, tempering));
                    }
                }
                Ok(PgasStageOpts {
                    n_chains: *chains,
                    n_particles: *particles,
                    n_sweeps: *sweeps,
                    burn_in: burn_in.unwrap_or(DEFAULT_BURN_IN),
                    thin: thin.unwrap_or(DEFAULT_THIN),
                    tempering: tempering.clone(),
                    max_tree_depth: *max_tree_depth,
                    trajectory_warmup: *trajectory_warmup,
                    csmc_sweeps_per_nuts: *csmc_sweeps_per_nuts,
                    n_trajectories: *n_trajectories,
                    dense_mass: *dense_mass,
                    use_nuts: *use_nuts,
                    init_method: init_method.clone(),
                    survey_path: survey_path.clone(),
                    survey_top_k_n: *survey_top_k_n,
                })
            }
            other => Err(format!(
                "PgasStageOpts::from_stage: expected Stage::PGAS, got {}",
                other.method_name())),
        }
    }
}

// Per-stage entry point for PGAS — wide because every flag is
// independent at the dispatch site (stage_dir, opts struct, RNG seed,
// --resume / --starts-from). Same pattern as
// `batch::run_one_scenario` and `main::run_simulate`, both of which
// also carry this allow.
#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    fit: &super::config_v2::FitConfigV2,
    stage_name: &str,
    stage: &super::config_v2::Stage,
    stage_dir: &Path,
    pgas_opts: PgasStageOpts,
    seed: u64,
    force: bool,
    resume: bool,
    starts_from: Option<&str>,
) -> Result<(), String> {
    let estimate = &fit.estimate;
    let n_chains = pgas_opts.n_chains;
    let n_sweeps = pgas_opts.n_sweeps;
    let n_particles = pgas_opts.n_particles;
    let burn_in = pgas_opts.burn_in;
    let thin = pgas_opts.thin;
    let n_trajectories = pgas_opts.n_trajectories;
    let use_nuts = pgas_opts.use_nuts;
    let dense_mass = pgas_opts.dense_mass;

    if !force && !resume {
        let state_path = stage_dir.join("fit_state.toml");
        if state_path.exists() {
            eprintln!("\x1b[33mpgas results already exist in {}. Use --force to re-run or --resume to continue.\x1b[0m",
                stage_dir.display());
            return Ok(());
        }
    }

    std::fs::create_dir_all(stage_dir)
        .map_err(|e| format!("cannot create {}: {}", stage_dir.display(), e))?;

    let collector = DiagnosticCollector::new("pgas");

    // Load prior state if --starts-from provided
    let starts_from = starts_from.map(String::from);
    let prior_state = starts_from.as_deref().map(FitState::load).transpose()?;

    // Build FitRunConfig (reuse existing builder). cooling_target_iters
    // is IF2-specific and never read by PGAS — pass 1 as a harmless value.
    let config = FitRunConfig::build(
        fit, prior_state.as_ref(),
        n_chains, n_particles, 1,
        1.0, 1, seed, false,
    )?;

    let dt = config.if2_config.dt;

    // gh#342 P4: the rate/observation/σ² domains are refused inside `run_pgas`
    // (the preflight scans the emitted `DerivEntry` maps for `Unsupported`). The
    // residual the preflight cannot see is a parameter reaching a forcing/table
    // coefficient ONLY through an initial condition: camdl emits no gradient for
    // IC expressions (gh#275), so NUTS would sample against a flat surface. This
    // source-level scan is the only place that classifies it — refuse it here.
    if use_nuts {
        let estimated: std::collections::HashSet<String> =
            config.estimated_params.iter().map(|s| s.name.clone()).collect();
        let offenders = super::coeff_guard::ic_coefficient_only_estimated(&config.model, &estimated);
        if !offenders.is_empty() {
            return Err(super::coeff_guard::nuts_guard_error(&offenders));
        }
    }

    // gh#audit-C4. PGAS produces Bayesian posteriors. Reporting a
    // credible interval as "Bayesian" when the prior is silently
    // improper-uniform (Prior::Flat) means we've reported a
    // likelihood profile rescaled. For polio decision support, that
    // is the worst-case communication failure (per the audit /
    // proposal). Resolve priors with their source labels so we can
    // refuse to run when any estimated parameter has no prior.
    let resolved: Vec<(Prior, &'static str)> = config.estimated_params.iter()
        .map(|spec| super::runner::resolve_prior(&spec.name, estimate, &config.model))
        .collect();
    let missing_priors: Vec<&str> = config.estimated_params.iter()
        .zip(&resolved)
        .filter(|(_, (_, src))| *src == "flat (default)")
        .map(|(spec, _)| spec.name.as_str())
        .collect();
    if !missing_priors.is_empty() {
        return Err(format!(
            "pgas refuses to run with implicit improper-uniform priors. \
             The following estimated parameters have no prior: {}. \
             Add a `[estimate.<name>.prior]` block in fit.toml or a `~` \
             prior in the model. To opt into uniform priors explicitly, \
             use `prior = {{ uniform = {{ lower = ..., upper = ... }} }}`.",
            missing_priors.join(", ")
        ));
    }
    let priors: Vec<Prior> = resolved.into_iter().map(|(p, _)| p).collect();

    // Active interventions + events — makes the scenario/enable default
    // visible before sampling, so a forgotten `scenario = "..."` doesn't
    // hide as "0 of N firing" behind a 6-hour chain.
    crate::util::print_scheduled_actions_summary(&config.model_declared, &config.model);
    crate::util::print_observations_summary(&config.model);

    // Report priors
    let any_non_flat = priors.iter().any(|p| !matches!(p, Prior::Fixed(Density::Flat)));
    if any_non_flat {
        eprintln!("  priors:");
        for (spec, prior) in config.estimated_params.iter().zip(&priors) {
            match prior {
                Prior::Fixed(Density::Flat) => {},
                Prior::Fixed(Density::Uniform { lower, upper }) => {
                    eprintln!("    {:12} Uniform({:.4}, {:.4})", spec.name, lower, upper);
                }
                Prior::Fixed(Density::Normal { mean, sd }) => {
                    eprintln!("    {:12} Normal({:.4}, {:.4})", spec.name, mean, sd);
                }
                Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
                    eprintln!("    {:12} LogNormal(mu={:.4}, sigma={:.4}) → median={:.1}",
                        spec.name, mean, sd, mean.exp());
                }
                Prior::Fixed(Density::HalfNormal { sigma }) => {
                    eprintln!("    {:12} HalfNormal(sigma={:.4})", spec.name, sigma);
                }
                Prior::Fixed(Density::Beta { alpha, beta }) => {
                    let mode = if *alpha > 1.0 && *beta > 1.0 {
                        (alpha - 1.0) / (alpha + beta - 2.0)
                    } else { 0.5 };
                    eprintln!("    {:12} Beta({:.2}, {:.2}) → mode={:.3}",
                        spec.name, alpha, beta, mode);
                }
                Prior::Fixed(Density::Gamma { shape, rate }) => {
                    eprintln!("    {:12} Gamma(shape={:.4}, rate={:.4})",
                        spec.name, shape, rate);
                }
                Prior::Fixed(Density::Exponential { rate }) => {
                    eprintln!("    {:12} Exponential(rate={:.4})", spec.name, rate);
                }
                Prior::Fixed(Density::LogUniform { lower, upper }) => {
                    eprintln!("    {:12} LogUniform({:.4e}, {:.4e})", spec.name, lower, upper);
                }
                Prior::Fixed(Density::TruncatedNormal { mean, sd, lower, upper }) => {
                    eprintln!("    {:12} TruncatedNormal(mean={:.4}, sd={:.4}) on [{:.4}, {:.4}]",
                        spec.name, mean, sd, lower, upper);
                }
                Prior::Hierarchical(d) => {
                    // The full hierarchical metadata (pool_over, named args) lives
                    // in the IR; the runtime Prior carries only the resolved
                    // density shape, so read the display detail from the model.
                    let ir_h = config.model.parameters.iter()
                        .find(|p| p.name == spec.name)
                        .and_then(|p| p.hierarchical());
                    match ir_h {
                        Some(h) => {
                            let parents: Vec<String> = h.args.values()
                                .filter_map(|e| if let ir::expr::Expr::Param(p) = e {
                                    Some(p.param.clone())
                                } else { None })
                                .collect();
                            eprintln!("    {:12} Hierarchical {}(...) | pool_over={} | parents=[{}]",
                                spec.name, h.kind, h.pool_over, parents.join(", "));
                        }
                        None => eprintln!("    {:12} Hierarchical {}(...)", spec.name, d.kind_str()),
                    }
                }
            }
        }
    }

    // Compute config hash — identifies the statistical problem.
    // Changes to model/data/priors/bounds/particles/dt invalidate resume state.
    // Uses provenance::fit_stage_hash, the same hash the v2 dispatch
    // site uses for cache-hit / staleness checks (model + observations
    // + estimate + fixed + stage_name + Stage variant + seed).
    let fixed_resolved = fit.fixed.resolve()?;
    let data_spec = fit.data_spec()?;
    let config_hash = super::provenance::fit_stage_hash(
        &config.model_ir_json, &data_spec.observations,
        &fit.estimate, &fixed_resolved, &fit.simplex_groups,
        stage_name, stage, seed,
    )?;

    // Load resume states if --resume
    let resume_states: Vec<Option<ChainResumeState>> = if resume {
        let mut states = Vec::with_capacity(n_chains);
        let mut any_failed = false;
        for chain_id in 0..n_chains {
            let path: PathBuf = stage_dir.join(format!("chain_{}", chain_id + 1))
                .join("resume_state.bin");
            match std::fs::read(&path) {
                Ok(data) => match bincode::deserialize::<ChainResumeState>(&data) {
                    Ok(state) => {
                        if state.config_hash != config_hash {
                            eprintln!("error: config hash mismatch for chain {} — \
                                model/data/priors have changed since the original run. \
                                Cannot resume. Re-run from scratch with --force.",
                                chain_id + 1);
                            std::process::exit(1);
                        }
                        eprintln!("  chain {}: resuming from sweep {}", chain_id + 1, state.completed_sweeps);
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
            eprintln!("  These are written automatically at the end of every PGAS run.");
            eprintln!("  If the original run was interrupted before saving, use --force to start fresh.");
            std::process::exit(1);
        }
        states
    } else {
        vec![None; n_chains]
    };

    // Generate per-chain starting parameters (gh#42, gh#51 v2).
    // Precedence:
    // 1. `--starts-from` — every chain at the prior MLE; mutually
    //    exclusive with `init = "survey_top_k"` (the former pins every
    //    chain to one point, so any survey-seeded start would be
    //    silently overwritten).
    // 2. `init = "survey_top_k"` — resolved here via the shared helper.
    //    Bayesian seeds are valid because the chain's stationary
    //    distribution is set by the prior, not the start.
    // 3. `init` dispatch on Lhs / Uniform / Single. Default `lhs` gives
    //    stratified posterior coverage.
    let has_starts = prior_state.is_some();
    let mut survey_top_k_result: Option<super::init::SurveyTopKResult> = None;
    let chain_starts: Vec<Vec<f64>> = if has_starts {
        if pgas_opts.init_method == super::init::InitMethod::SurveyTopK {
            return Err(format!(
                "pgas stage `{}`: --starts-from / `init_mle = \"...\"` and \
                 `init = \"survey_top_k\"` are mutually exclusive — \
                 the former commits every chain to the prior MLE, so any \
                 survey-seeded start would be silently overwritten. Pick one: \
                 drop `init_mle`, or use a non-survey `init`.",
                stage_name));
        }
        vec![config.base_params.clone(); n_chains]
    } else if pgas_opts.init_method == super::init::InitMethod::SurveyTopK {
        // Cross-check context. Same construction as IF2 / PMMH.
        let model_identity_str = crate::resolve::model_identity_from_ir(&config.model_ir_json);
        let model_obs_names: Vec<String> = config.model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let effective_obs = data_spec.effective_observations(&model_obs_names)?;
        let data_hashes = super::init::compute_data_hashes(&effective_obs)?;
        let estimate_names: Vec<String> = fit.estimate.keys().cloned().collect();
        let fixed_hashmap: std::collections::HashMap<String, f64> =
            fixed_resolved.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let ctx = super::init::SurveyFitContext {
            model_identity: &model_identity_str,
            data_hashes: &data_hashes,
            fixed: &fixed_hashmap,
            estimate_names: &estimate_names,
        };
        let (chains_opt, result) =
            super::init::resolve_per_chain_starts_from_method(
                &pgas_opts.init_method,
                pgas_opts.survey_path.as_deref(),
                pgas_opts.survey_top_k_n,
                stage_name,
                &config.estimated_params,
                n_chains,
                seed,
                &ctx,
                // SurveyTopK doesn't need ResolvedParameters; that
                // branch fires above. The warm-start variants take a
                // separate dispatch path below (post-step-7).
                None,
            ).map_err(|e| format!("pgas: {}", e))?;
        let chains_specs = chains_opt
            .expect("SurveyTopK must yield per-chain starts");
        survey_top_k_result = result;
        super::init::chain_starts_to_param_vecs(&chains_specs, &config.base_params)
    } else if matches!(pgas_opts.init_method,
        super::init::InitMethod::FromPrior
        | super::init::InitMethod::FromPosterior { .. }
        | super::init::InitMethod::FromMle    { .. }
        | super::init::InitMethod::FromParams { .. })
    {
        // Step 7 warm-start dispatch (gh#83/gh#85). Build a minimal
        // `ResolvedParameters` view from the fit runner config and
        // route through `chain_starts::draw_chain_starts`. Provenance
        // for the resolved value side is already recorded upstream
        // (params_resolver runs in fit/runner.rs:188); this branch
        // owns only chain-start provenance, which step 9 already
        // serializes into `init_provenance.chains[i]`.
        let resolved_view = super::init::build_resolved_view_for_init(
            &config.model, &config.base_params, &config.estimated_params,
        );
        let starts = crate::fit::chain_starts::draw_chain_starts(
            &resolved_view, &pgas_opts.init_method, n_chains, seed,
        ).map_err(|e| format!("pgas: --init {}: {}",
            pgas_opts.init_method, e))?;
        let chains_specs = starts.to_estimated_params(&config.estimated_params);
        super::init::chain_starts_to_param_vecs(&chains_specs, &config.base_params)
    } else {
        super::init::build_chain_param_vecs(
            &pgas_opts.init_method,
            &config.estimated_params,
            &config.base_params,
            n_chains,
            seed,
        ).map_err(|e| format!("pgas: {}", e))?
        .unwrap_or_else(|| vec![config.base_params.clone(); n_chains])
    };

    eprintln!("\npgas: {} chains × {} sweeps × {} particles, burn_in={}, thin={}",
        n_chains, n_sweeps, n_particles, burn_in, thin);
    if has_starts {
        eprintln!("  starting all chains from prior stage (--starts-from)");
    } else {
        eprintln!("  chain starts: init = {} (per-parameter ranges below)",
            pgas_opts.init_method);
        for spec in &config.estimated_params {
            let vals: Vec<f64> = chain_starts.iter().map(|p| p[spec.index]).collect();
            eprintln!("    {:12} [{:.4} .. {:.4}]", spec.name,
                vals.iter().cloned().fold(f64::INFINITY, f64::min),
                vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        }
    }
    eprintln!("  estimated output: {} posterior samples per chain",
        (n_sweeps.saturating_sub(burn_in)) / thin);

    // Pre-create chain directories (must happen before parallel spawn)
    for chain_id in 0..n_chains {
        let chain_dir = stage_dir.join(format!("chain_{}", chain_id + 1));
        std::fs::create_dir_all(&chain_dir)
            .map_err(|e| format!("cannot create {}: {}", chain_dir.display(), e))?;
    }

    // Write chain_starts.tsv sidecar for audit (gh#51 v2). Best-effort;
    // failure logs but does not abort the fit. Rebuild the per-chain
    // EstimatedParam view from the f64 vectors so the writer (which
    // expects the IF2 shape) can label each row with the right
    // `source` (survey:<hash>:rank-N for SurveyTopK, otherwise
    // <method>:chain-<id>).
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
        &pgas_opts.init_method,
        survey_top_k_result.as_ref(),
    ) {
        eprintln!("warning: could not write chain_starts.tsv: {}", e);
    }

    let t0 = std::time::Instant::now();
    let _is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());

    // gh#audit-C7. Per-chain NUTS / tempering diagnostics surfaced from
    // PGASResult. Indexed by chain_id, populated inside the parallel
    // closure via Mutex (only written once per chain). Consumed after
    // the loop to wire DiagnosticKind::DivergentTransitions /
    // MaxTreeDepthHits / LowSwapRate.
    #[derive(Clone, Default)]
    struct ChainNutsDiag {
        n_divergent_post_burn:    usize,
        n_max_treedepth_post_burn: usize,
        swap_acceptance_rates:    Vec<f64>,
    }
    let chain_nuts: std::sync::Mutex<Vec<ChainNutsDiag>> =
        std::sync::Mutex::new(vec![ChainNutsDiag::default(); n_chains]);

    // Per-chain progress bars (mirrors IF2 / PMMH). One `Reporter` hands out a
    // `Task` per chain, rendered as a coordinated stack; the Reporter honors
    // --progress (Pretty=bars, Plain=throttled `chain N pos/len ll=…` log
    // lines, None=silent). The metric is the cold-chain complete-data
    // log-likelihood the sampler already reports each sweep — no new compute.
    let reporter = crate::progress::Reporter::new();
    let bars: Vec<crate::progress::Task> = (0..n_chains)
        .map(|chain_id| reporter.task(n_sweeps as u64, format!("chain {}", chain_id + 1), "sweeps"))
        .collect();

    // gh#278: per-run liveness/progress heartbeat. A background thread writes
    // `progress.json` into the stage dir every 5 s on a FIXED wall-clock timer —
    // independent of sweep cadence (one spatial PGAS sweep can take minutes, so
    // a sweep-boundary heartbeat would be as stale as the trace). Each chain
    // only `bump`s a shared atomic (no I/O), so this cannot affect any fit
    // number. On clean completion we write `Done`; any error/panic drops the
    // heartbeat, leaving the last `Running` to go stale → consumer reads
    // `PresumedDead`.
    let heartbeat = Heartbeat::mcmc(
        stage_dir.to_path_buf(),
        burn_in as u64,
        n_sweeps as u64,
        std::time::Duration::from_secs(5),
    );

    // Posterior-trajectory output metadata, computed once and shared across the
    // per-chain writers (the model is shared, so these are chain-invariant).
    // `model_hash` is the structural model identity (the `# camdl-trajectories`
    // header + manifest `model_hash`); `date_origin` reuses `simulate`'s
    // calendar formatting when the model declares an `origin`.
    let traj_model_hash = crate::resolve::model_identity_from_ir(&config.model_ir_json);
    let traj_date_origin: Option<(String, String)> = config.model.origin.as_ref()
        .map(|o| (o.clone(), config.model.time_unit.clone()));
    let traj_calendar = io::CalendarMeta::from_model(&config.model);

    // Run chains in parallel (each chain is independent: own seed, own
    // trajectory, own RNG). Same pattern as PMMH.
    //
    // Each chain yields `Ok(Some(result))`, or `Ok(None)` when `run_pgas`
    // refuses its start with `NonFiniteChainStart` (gh#607) — skipped with a
    // `BadInit` diagnostic and omitted from every downstream number
    // (draws.tsv, R̂/ESS, the MAP `fit_state`), surviving chains continue. An
    // `Err` is a structural failure: the model/config cannot run, so the whole
    // fit aborts rather than reporting a partial posterior — `collect`
    // short-circuits on the first such error. Mirrors PMMH (`pmmh.rs`) and IF2
    // (`runner.rs`), which do the same for `PFDegenerate`.
    use rayon::prelude::*;
    let all_results: Vec<Result<Option<(usize, Vec<PGASSweep>, Vec<f64>)>, String>> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let chain_seed = crate::util::derive_chain_seed(seed, chain_id);
            let chain_dir = stage_dir.join(format!("chain_{}", chain_id + 1));
            let task = &bars[chain_id];

            let pgas_config = PGASConfig {
                n_particles,
                n_sweeps,
                burn_in,
                thin,
                dt,
                use_nuts,
                dense_mass,
                tempering: pgas_opts.tempering.clone(),
                max_tree_depth: pgas_opts.max_tree_depth,
                trajectory_warmup: pgas_opts.trajectory_warmup,
                csmc_sweeps_per_nuts: pgas_opts.csmc_sweeps_per_nuts,
                // Stage 3: PGAS keeps snap alignment until exact-PGAS recovery
                // evidence lands and resolve_obs_alignment flips the default.
                step_policy: sim::schedule::StepPolicy::Snap,
            };

            // Build multi-stream observation model (evaluates with params at call time)
            let compiled = &*config.compiled;
            let obs_model = config.build_obs_model();

            let observations: Vec<sim::inference::particle_filter::Observation> =
                config.observations.iter()
                    .map(|o| sim::inference::particle_filter::Observation {
                        time: o.time, value: o.value,
                    })
                    .collect();

            // Streaming trace file — append when resuming, create when fresh
            let trace_path = chain_dir.join("trace.tsv");
            let trace_path_str = trace_path.to_string_lossy().into_owned();
            let is_resuming = resume_states[chain_id].is_some();
            let param_names: Vec<String> = config.estimated_params.iter()
                .map(|s| s.name.clone()).collect();
            // gh#688: `renewal_b0 … renewal_b9` carry `trajectory_renewal`
            // resolved in time — bin `b` is renewal over the `b`-th tenth of
            // the substeps. Averaged down the column over post-burn-in sweeps
            // they are the update-rate-against-t plot that Lindsten, Jordan &
            // Schön (2014, JMLR 15:2145-2184, Fig. 1) and Chopin & Singh (2015,
            // Bernoulli 21:1855-1883) both recommend in place of a rule for
            // choosing the particle count. They sit between the aggregate and
            // the ancestor-sampling counters because the three are read
            // together: the profile says WHERE the path is stuck, `as_accept`
            // and `as_proposed` say why.
            let mut trace_columns: Vec<&str> = Vec::with_capacity(RENEWAL_BINS + 11);
            trace_columns.push("trajectory_renewal");
            trace_columns.extend(RENEWAL_BIN_COLUMNS);
            trace_columns.extend(["as_accept", "as_proposed", "transition_ll", "obs_ll",
                  "tree_depth", "n_leapfrog", "step_size", "accept_stat",
                  "n_divergent", "energy"]);
            let trace_writer = super::trace_writer::TraceWriter::new(
                &trace_path_str, "sweep", "log_complete_data_ll",
                &trace_columns,
                &param_names, is_resuming,
            );

            let chain_start = std::time::Instant::now();

            // Trajectory save stride: evenly space n_trajectories across post-burn-in
            let n_post_burnin = n_sweeps.saturating_sub(burn_in);
            let traj_stride = if n_trajectories > 0 && n_post_burnin > 0 {
                (n_post_burnin / n_trajectories).max(1)
            } else {
                usize::MAX // disabled
            };

            // Shared posterior-trajectory output (latent-trajectory-output
            // consolidation, 2026-06-09). The per-substep reference path of each
            // saved sweep projects into the `simulate` `Trajectory` type via the
            // `SubstepRecord → Snapshot` adapter; one tidy/long `trajectories.tsv`
            // (all draws stacked, leading `chain draw time` id columns) plus a
            // `trajectories.json` manifest replaces the per-draw wide files.
            //
            // `inc_<stream>` columns come from the observation model's `FlowSum`
            // projection applied to the substep flows (gh#48 safe path) — never a
            // finite-difference of compartment counts (unsafe under
            // event/balance, gh#264).
            let incidence_streams = obs_model.incidence_streams();
            let incidence_stream_names: Vec<String> =
                incidence_streams.iter().map(|(n, _)| n.clone()).collect();
            let traj_columns =
                TrajColumnSpec::from_model(&config.compiled.model, &incidence_stream_names);
            // Accumulate saved draws; the callback runs once per sweep, in order,
            // within this chain (rayon parallelism is across chains), so a
            // single-threaded RefCell accumulator is sound.
            let saved_draws: RefCell<Vec<PosteriorDraw>> = RefCell::new(Vec::new());

            let progress_cb = |sweep: usize, result: &PGASSweep, traj: &PGASTrajectory| {
                // Stream trace row via shared TraceWriter
                let log_prior: f64 = config.estimated_params.iter().zip(priors.iter())
                    .map(|(spec, prior)| {
                        let natural = result.params[spec.index];
                        let z = spec.to_transformed(natural);
                        prior.log_density(natural, z)
                    })
                    .sum();
                let log_posterior = result.log_complete_data_ll + log_prior;
                let param_vals: Vec<f64> = config.estimated_params.iter()
                    .map(|s| result.params[s.index]).collect();
                let renewal = format!("{:.4}", result.csmc_diag.trajectory_renewal);
                // gh#688: renewal per time bin. `NA` for a bin holding no
                // substep — the same convention as `as_accept` below, and for
                // the same reason: no data is not a renewal of zero.
                let renewal_bins: Vec<String> = result.csmc_diag.renewal_by_bin.iter()
                    .map(|&r| if r.is_finite() { format!("{r:.4}") } else { "NA".to_string() })
                    .collect();
                // gh#607 follow-up: the ancestor-sampling Metropolis acceptance
                // rate, with its denominator alongside. `NA` means the step
                // never ran (no alternative ancestor was admissible), which is
                // a different diagnosis from an acceptance rate of 0.
                let as_rate = result.csmc_diag.as_accept_rate();
                let as_accept_str = if as_rate.is_finite() {
                    format!("{:.4}", as_rate)
                } else {
                    "NA".to_string()
                };
                let as_proposed_str = result.csmc_diag.n_as_proposed.to_string();
                let transition_ll_str = format!("{:.4}", result.transition_ll);
                let obs_ll_str = format!("{:.4}", result.obs_ll);
                // Per-sweep cold-chain NUTS diagnostics (gh#294).
                let nd = &result.nuts;
                let tree_depth_str = nd.tree_depth.to_string();
                let n_leapfrog_str = nd.n_leapfrog.to_string();
                let step_size_str = format!("{:.6}", nd.step_size);
                let accept_stat_str = format!("{:.4}", nd.accept_stat);
                let n_divergent_str = nd.n_divergent.to_string();
                let energy_str = format!("{:.4}", nd.energy);
                let mut extra: Vec<&str> = Vec::with_capacity(RENEWAL_BINS + 11);
                extra.push(&renewal);
                extra.extend(renewal_bins.iter().map(String::as_str));
                extra.extend([
                    as_accept_str.as_str(), as_proposed_str.as_str(),
                    transition_ll_str.as_str(), obs_ll_str.as_str(),
                    tree_depth_str.as_str(), n_leapfrog_str.as_str(), step_size_str.as_str(),
                    accept_stat_str.as_str(), n_divergent_str.as_str(), energy_str.as_str(),
                ]);
                trace_writer.write_row(
                    sweep, result.log_complete_data_ll, log_posterior,
                    &extra,
                    &param_vals,
                );

                // Save posterior trajectory sample. The adapter takes
                // `counts_after` (via `coherent_counts_after`, gh#264) + the
                // per-substep flows, stamps each snapshot at `t0 + dt_substep`,
                // and projects `inc_<stream>` from the substep flows. Drops the
                // density internals (`counts_before`, `gammas`, `dt_substep` as a
                // field). On an incoherent record (an AS-join corruption the
                // adapter can't reconcile) the draw is skipped with a status line
                // rather than silently emitting a backflowing path.
                if sweep >= burn_in && (sweep - burn_in).is_multiple_of(traj_stride) {
                    match traj.to_trajectory(&incidence_streams) {
                        Err(e) => crate::status::step("pgas-traj",
                            format!("skipping trajectory {sweep} (incoherent record): {e}")),
                        Ok((path_traj, incidence)) => {
                            saved_draws.borrow_mut().push(PosteriorDraw {
                                chain: chain_id,
                                draw: sweep,
                                path: path_traj,
                                incidence,
                            });
                        }
                    }
                }

                // Passive bar tick. The callback fires once per sweep in order,
                // so `inc(1)` tracks position = sweep+1 exactly. The metric is
                // the cold-chain complete-data loglik the sampler already
                // reports — fed as `CompleteData` so the live feed reads
                // `ll(complete)=…`, not the bare `ll=…` that means a marginal
                // for every other method (gh#280). `Task` handles Pretty
                // (redraw) / Plain (`chain N pos/len ll(complete)=…`) / None.
                task.set(crate::progress::ll_kind(
                    result.log_complete_data_ll,
                    LoglikType::CompleteData,
                ));
                task.inc(1);

                // gh#278: report progress to the shared heartbeat (monotonic
                // fetch_max across the parallel chains; phase derived from
                // burn_in). Cheap atomic — no I/O on the sweep path.
                heartbeat.bump(sweep as u64);
            };

            let result = match run_pgas(
                compiled,
                &config.estimated_params,
                &priors,
                &chain_starts[chain_id],
                &pgas_config,
                &observations,
                &obs_model,
                chain_seed,
                Some(&progress_cb),
                resume_states[chain_id].clone(),
                config_hash.clone(),
            ) {
                Ok(r) => r,
                // gh#607. The chain's start has zero posterior density and did
                // not recover on its first trajectory update, so it could only
                // have produced `-inf` draws for the whole run. Skip it with a
                // loud `BadInit` diagnostic (collector + stderr) — never
                // silently — and let the survivors finish.
                Err(sim::error::SimError::NonFiniteChainStart {
                    log_posterior, transition, observation, ivp, log_prior,
                }) => {
                    let reason = format!(
                        "initial complete-data log-posterior is {}, still \
                         non-finite after the first trajectory update \
                         (log-likelihood terms: transition {:.4}, observation \
                         {:.4}, ivp {:.4}; log prior {:.4})",
                        log_posterior, transition, observation, ivp, log_prior);
                    // gh#513: report the start THIS chain ran from, not the
                    // configured one. `chain_starts[chain_id]` is the same
                    // vector handed to `run_pgas` above and the same one
                    // `write_chain_starts_tsv` recorded, so the diagnostic and
                    // `chain_starts.tsv` name the same numbers.
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
                Err(e) => {
                    return Err(format!("pgas chain {} error: {}", chain_id + 1, e));
                }
            };

            // Save resume state for future --resume
            let resume_path = chain_dir.join("resume_state.bin");
            if let Ok(encoded) = bincode::serialize(&result.resume_state) {
                let _ = std::fs::write(&resume_path, encoded);
            }

            // Write this chain's posterior latent trajectories: one tidy/long
            // `trajectories.tsv` (all saved draws stacked, leading `chain draw
            // time [date]` id columns) + a `trajectories.json` manifest. Replaces
            // the per-draw `trajectory_NNNNNN.tsv` wide files.
            let draws = saved_draws.into_inner();
            if !draws.is_empty() {
                let date_origin = traj_date_origin.as_ref()
                    .map(|(o, u)| (o.as_str(), u.as_str()));
                let tsv_path = chain_dir.join("trajectories.tsv");
                write_trajectories_tsv(
                    &tsv_path, &draws, &traj_columns, date_origin,
                    &traj_model_hash, "pgas", Granularity::Substep,
                ).map_err(|e| format!("pgas chain {}: {}", chain_id + 1, e))?;

                let manifest = TrajManifest {
                    method: "pgas".to_string(),
                    granularity: Granularity::Substep,
                    n_chains: 1,
                    n_draws: draws.len(),
                    columns: {
                        // Id columns + data columns, in emit order.
                        let mut cols = vec!["chain".to_string(), "draw".to_string(),
                            "time".to_string()];
                        if date_origin.is_some() { cols.push("date".to_string()); }
                        cols.extend(traj_columns.data_column_names());
                        cols
                    },
                    model_hash: traj_model_hash.clone(),
                    // PGAS draws are conditioned smoother paths X|θ,y; their
                    // `inc_<stream>` is conditioned incidence, NOT the
                    // free-forward posterior-predictive a `simulate --obs` run
                    // produces.
                    conditioned: true,
                    // PGAS ancestor sampling mitigates filter-smoother
                    // degeneracy, so no early-time degeneracy caveat (unlike the
                    // PF/PMMH paths a later step adds).
                    degeneracy_caveat: false,
                    n_trajectories,
                    // Best-effort: a sibling forward posterior-predictive obs
                    // file isn't produced by `fit`, so none is recorded here.
                    // (`simulate --obs` on the posterior draws would produce it.)
                    predictive_obs_file: None,
                    calendar: traj_calendar.clone(),
                };
                let _ = manifest.write(&chain_dir.join("trajectories.json"));
            }

            let chain_elapsed = chain_start.elapsed();
            eprintln!("  chain {} done: {:.1}s, acceptance: [{}]",
                chain_id + 1,
                chain_elapsed.as_secs_f64(),
                config.estimated_params.iter().zip(&result.acceptance_rates)
                    .map(|(p, &r)| format!("{}={:.0}%", p.name, r * 100.0))
                    .collect::<Vec<_>>().join(", "));

            // gh#audit-C7. Surface per-chain NUTS / tempering diagnostics so the
            // post-loop diagnostic-collector pass can construct
            // DivergentTransitions / MaxTreeDepthHits / LowSwapRate variants.
            // Mutex contention is negligible (n_chains writes total).
            {
                let mut nd = chain_nuts.lock().unwrap();
                nd[chain_id] = ChainNutsDiag {
                    n_divergent_post_burn:    result.n_divergent_post_burn,
                    n_max_treedepth_post_burn: result.n_max_treedepth_post_burn,
                    swap_acceptance_rates:    result.swap_acceptance_rates.clone(),
                };
            }

            Ok(Some((chain_id, result.sweeps, result.acceptance_rates)))
        })
        .collect();

    // Clear all chain bars now that the parallel phase is done (`Task::finish`
    // consumes, so it can't run on the per-chain borrow inside the loop). Done
    // before the `?` unwrap so a chain error still clears the bars. The
    // `acceptance rates:` report below carries the per-chain summary.
    for t in bars { t.finish(); }

    // Unwrap results (propagate first error), dropping the chains that were
    // refused at their start (gh#607). From here on `all_results` holds only
    // chains that actually sampled, so every downstream number — draws.tsv,
    // R̂/ESS, the MAP `fit_state` — is over survivors alone.
    let all_results: Vec<(usize, Vec<PGASSweep>, Vec<f64>)> = all_results
        .into_iter()
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .flatten()
        .collect();

    // gh#607. Skip + continue: surface "ran K of N chains" so the user knows
    // the downstream R̂/ESS exclude the skipped chains. This line is the
    // load-bearing user-facing signal that a bad start was tolerated rather
    // than silently dropped. Mirrors PMMH (`pmmh.rs`).
    let n_good_chains = all_results.len();
    if n_good_chains < n_chains {
        eprintln!("\n\x1b[33mran {} of {} chains\x1b[0m \
                   ({} skipped via BadInit; see diagnostics below)",
            n_good_chains, n_chains, n_chains - n_good_chains);
    }

    // gh#607. Every chain was refused at its start — there is no posterior to
    // pool. Render and persist the per-chain `BadInit` diagnostics and bail,
    // rather than falling through to an `.expect()` on an empty result set.
    // `InitialLoglikInfinite` rides along because that is exactly the
    // situation, and it is the signal the gh#226 whole-fit backstop (which
    // this refusal now pre-empts for the all-bad-start case) taught consumers
    // to look for.
    if all_results.is_empty() {
        collector.push(DiagnosticKind::InitialLoglikInfinite);
        collector.render_to_stderr();
        let diag_path = stage_dir.join("diagnostics.json");
        let _ = collector.write_json(&diag_path.to_string_lossy());
        return Err(format!(
            "pgas stage `{}`: all {} chain(s) were refused at their starting \
             point — the complete-data log-posterior is non-finite for every \
             one and stayed non-finite through its first trajectory update, so \
             no chain could move. See `diagnostics.json` for the per-chain \
             `bad_init` entries and `chain_starts.tsv` for the starts they \
             name. Most often the starting values sit in an impossible region \
             (try `--init lhs` or a different start); less often the data are \
             impossible under this model — also check the observation model \
             and parameter bounds.",
            stage_name, n_chains));
    }

    let elapsed = t0.elapsed();

    // Compute diagnostics
    let diagnostics = compute_diagnostics(&all_results, &config.estimated_params);

    // Report. The healthy band is KERNEL-specific (gh#631): NUTS targets
    // ~0.8, so applying the random-walk band to it reported every well-tuned
    // NUTS fit as unhealthy — one severity:error per parameter — burying real
    // failures (the ebola F8 stuck chain hid in that noise). The band, the
    // predicate and the message all come from `AcceptanceKernel` (gh#299), and
    // the same `nuts_active` predicate the sampler used picks the kernel. A
    // BLOCK update (identical rate on every parameter) is reported once per
    // chain, not once per parameter.
    let nuts = sim::inference::pgas::nuts_active(use_nuts, config.compiled.as_ref());
    let kernel = if nuts {
        sim::inference::diagnostic::AcceptanceKernel::Nuts
    } else {
        sim::inference::diagnostic::AcceptanceKernel::RandomWalk
    };
    let (lo, hi) = kernel.healthy_band();
    eprintln!("\nacceptance rates{}:", if nuts { " (NUTS block; ~80% is the target)" } else { "" });
    for &(chain_id, _, ref rates) in &all_results {
        let block_update = rates.len() > 1
            && rates.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
        let summary: Vec<String> = config.estimated_params.iter().zip(rates)
            .map(|(p, &r)| {
                let status = if r < lo { "\x1b[31m" }
                    else if r > hi { "\x1b[33m" }
                    else { "\x1b[32m" };
                if !block_update {
                    if let Some(d) = sim::inference::diagnostic::acceptance_diagnostic(
                        r, Some(p.name.clone()), kernel)
                    {
                        collector.push(d);
                    }
                }
                format!("  {}={}{:.0}%\x1b[0m", p.name, status, r * 100.0)
            })
            .collect();
        if block_update {
            if let Some(d) = rates.first()
                .and_then(|&r| sim::inference::diagnostic::acceptance_diagnostic(r, None, kernel))
            {
                collector.push(d);
            }
        }
        eprintln!("  chain {}: {}", chain_id + 1, summary.join(" "));
    }

    if n_chains > 1 {
        // RHAT_REPORT_THRESHOLD is unchanged from the value this stage has
        // always applied; only the STATISTIC it is applied to changed (gh#84).
        eprint!("{}", diagnostics.report(&collector, super::runner::RHAT_REPORT_THRESHOLD));
    }

    // gh#audit-C7 + audit-H4. NUTS / tempering diagnostics surfaced from
    // PGASResult fields. Thresholds:
    //   - DivergentTransitions: ANY post-burn-in divergence (Stan
    //     convention). Matters for posterior validity, not just
    //     adaptation.
    //   - MaxTreeDepthHits: > 5% of post-burn-in sweeps (Stan).
    //   - LowSwapRate: any adjacent-rung pair with rate < 0.10 over
    //     the whole run (audit M18). Indicates the temperature ladder
    //     is too sparse — chains aren't mixing across rungs.
    let n_post_burn = n_sweeps.saturating_sub(burn_in);
    let nuts_diag = chain_nuts.lock().unwrap();
    for chain_id in 0..n_chains {
        let nd = &nuts_diag[chain_id];
        if nd.n_divergent_post_burn > 0 {
            collector.push(DiagnosticKind::DivergentTransitions {
                n_divergent: nd.n_divergent_post_burn,
                n_sweeps:    n_post_burn,
            });
        }
        if n_post_burn > 0 {
            let pct = nd.n_max_treedepth_post_burn as f64 / n_post_burn as f64 * 100.0;
            if pct > 5.0 {
                collector.push(DiagnosticKind::MaxTreeDepthHits {
                    n_hits:    nd.n_max_treedepth_post_burn,
                    n_sweeps:  n_post_burn,
                    pct,
                    max_depth: pgas_opts.max_tree_depth,
                });
            }
        }
        for (i, &rate) in nd.swap_acceptance_rates.iter().enumerate() {
            if rate < 0.10 {
                collector.push(DiagnosticKind::LowSwapRate {
                    rung_i: i,
                    rung_j: i + 1,
                    beta_i: pgas_opts.tempering[i],
                    beta_j: pgas_opts.tempering[i + 1],
                    rate,
                });
            }
        }
    }
    drop(nuts_diag);

    // gh#audit-H4. CSMC diagnostics from each chain's post-burn-in sweeps:
    //   - DegenerateAncestorSampling: pct > 10% (matches the previous
    //     log::warn! threshold at pgas.rs:970, S4 in the proposal).
    //   - LowTrajectoryRenewal: < 10% mean post-burn-in (canonical PGAS
    //     "stuck reference trajectory" signal).
    // Aggregated across post-burn-in sweeps within a chain (per-sweep
    // would generate noise; the aggregate is the actionable signal).
    for (_chain_id, sweeps, _rates) in &all_results {
        if sweeps.is_empty() { continue; }
        let mut total_n_degenerate: usize = 0;
        let mut total_n_substeps:   usize = 0;
        let mut renewal_sum:        f64   = 0.0;
        let mut renewal_n:          usize = 0;
        for sw in sweeps {
            total_n_degenerate += sw.csmc_diag.n_degenerate;
            total_n_substeps   += sw.csmc_diag.n_substeps;
            renewal_sum        += sw.csmc_diag.trajectory_renewal;
            renewal_n          += 1;
        }
        if total_n_substeps > 0 {
            let pct = total_n_degenerate as f64 / total_n_substeps as f64 * 100.0;
            if pct > 10.0 {
                collector.push(DiagnosticKind::DegenerateAncestorSampling {
                    pct,
                    n_degenerate: total_n_degenerate,
                    n_substeps:   total_n_substeps,
                });
            }
        }
        if renewal_n > 0 {
            let mean_renewal = renewal_sum / renewal_n as f64;
            if mean_renewal < 0.10 {
                collector.push(DiagnosticKind::LowTrajectoryRenewal {
                    renewal: mean_renewal,
                });
            }
        }
    }

    // Write summary JSON
    write_summary(stage_dir, &all_results, &config, thin, &diagnostics)?;

    // No-op resume: every chain already reached the target sweep count
    // before this invocation. There are no new sweeps to aggregate
    // and the on-disk fit_state.toml from the prior invocation is
    // still authoritative. Exit cleanly without re-aggregating.
    let any_new_sweeps = all_results.iter().any(|(_, sweeps, _)| !sweeps.is_empty());
    if !any_new_sweeps {
        eprintln!("\npgas: --resume found all chains at the target sweep \
            count. Nothing to do.");
        return Ok(());
    }

    // Write fit_state.toml with best params
    let best_chain = all_results.iter()
        .max_by(|a, b| {
            let best_ll_a = a.1.iter().map(|s| s.log_complete_data_ll)
                .fold(f64::NEG_INFINITY, f64::max);
            let best_ll_b = b.1.iter().map(|s| s.log_complete_data_ll)
                .fold(f64::NEG_INFINITY, f64::max);
            best_ll_a.total_cmp(&best_ll_b)
        })
        .expect("any_new_sweeps guard ensures non-empty");

    let best_sweep = best_chain.1.iter()
        .max_by(|a, b| a.log_complete_data_ll.total_cmp(&b.log_complete_data_ll))
        .expect("any_new_sweeps guard ensures non-empty");

    // gh#226. Whole-fit backstop: the best complete-data log-likelihood
    // across every SURVIVING chain's sweeps is non-finite → not one chain
    // ever reached a finite anchor, so the run would otherwise write a
    // degenerate fit_state (best loglik `-inf`) and exit 0. `best_sweep`
    // is the global maximum, so `no_finite_anchor(best)` is exactly
    // "every chain has no finite anchor"; a single finite sweep anywhere
    // makes it finite and the fit proceeds.
    //
    // The chain-start refusal (gh#607) now catches the special case where the
    // `-inf` region starts at sweep 0, and the `is_empty` guard above handles
    // the all-refused run. This backstop still covers the rest: a chain that
    // starts finite and walks into a `-inf` region it cannot leave.
    if sim::inference::no_finite_anchor(best_sweep.log_complete_data_ll) {
        collector.push(DiagnosticKind::InitialLoglikInfinite);
        collector.render_to_stderr();
        let diag_path = stage_dir.join("diagnostics.json");
        let _ = collector.write_json(&diag_path.to_string_lossy());
        return Err(format!(
            "pgas: all {} chain(s) reached no finite complete-data \
             log-likelihood (best = {}). The reachable surface is `-inf` at \
             every evaluated θ. Most often the starting values sit in an \
             impossible region — check those first (try `--init lhs` or a \
             different start); less often the data are impossible under this \
             model, or a recoverable error fires at every θ (gh#226). Also \
             check the observation model and parameter bounds.",
            n_chains, best_sweep.log_complete_data_ll));
    }

    let mut start_values = std::collections::BTreeMap::new();
    for spec in &config.estimated_params {
        start_values.insert(spec.name.clone(), best_sweep.params[spec.index]);
    }
    for p in &config.model.parameters {
        if !start_values.contains_key(&p.name) {
            if let Some(&idx) = config.compiled.param_index.get(p.name.as_str()) {
                start_values.insert(p.name.clone(), config.base_params[idx]);
            }
        }
    }

    let state = FitState {
        stage: stage_name.to_string(),
        seed,
        timestamp: iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik: best_sweep.log_complete_data_ll,
        initial_loglik: f64::NEG_INFINITY,
        best_chain: best_chain.0,
        n_chains,
        // gh#607: `None` unless a chain was skipped, so a healthy fit's
        // `fit_state.toml` is unchanged. Same convention as PMMH.
        n_good_chains: if n_good_chains < n_chains { Some(n_good_chains) } else { None },
        start_values,
        rw_sd: std::collections::BTreeMap::new(),
        loglik_type: Some(LoglikType::CompleteData),
        acceptance_rate: Some(best_chain.1.iter()
            .map(|s| s.accepted.iter().filter(|&&a| a).count() as f64 / s.accepted.len().max(1) as f64)
            .sum::<f64>() / best_chain.1.len().max(1) as f64),
        // Bayesian stage, no IF2-style Â (chain agreement).
        tail_chain_agreement: std::collections::BTreeMap::new(),
        ivp_params: Vec::new(),
        chain_logliks: Vec::new(),
        chain_eval_logliks: Vec::new(),
        chain_eval_ses: Vec::new(),
        // Bayesian path — compound gate doesn't apply to PGAS.
        resolved_gate: None,
        resolved_loglik_eval: None,
        // gh#51 v2: chain init provenance. When SurveyTopK was used,
        // emit the full survey hash + top-K via the shared formatter;
        // otherwise render the in-process sampler name verbatim.
        // SurveyTopK is dispatched via the shared
        // `resolve_per_chain_starts_from_method` helper above.
        chain_init_source: Some(super::init::format_chain_init_source(
            &pgas_opts.init_method, survey_top_k_result.as_ref(),
        )),
        // gh#52: Richardson dt-check is wired only on IF2 stages in
        // v1 (the inference math is shared but the dispatch site
        // refactor across PGAS/PMMH/NLopt is out of scope here).
        dt_check: None,
    };
    state.save(&stage_dir.to_string_lossy())?;

    // Write draws.tsv: complete-M posterior draws (all params, estimated + fixed)
    // Post-burn-in, thinned draws from all chains combined.
    {
        use std::io::Write;
        let draws_path = stage_dir.join("draws.tsv");
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&draws_path)
                .map_err(|e| format!("cannot create {}: {}", draws_path.display(), e))?
        );

        // Header: all model parameter names (estimated first, then fixed)
        let mut all_names: Vec<String> = config.estimated_params.iter()
            .map(|s| s.name.clone()).collect();
        let fixed_names: Vec<String> = config.model.parameters.iter()
            .filter(|p| !config.estimated_params.iter().any(|e| e.name == p.name))
            .map(|p| p.name.clone())
            .collect();
        all_names.extend(fixed_names.iter().cloned());
        // gh#322: leading `chain` `draw` key columns so each posterior draw is
        // joinable to its smoothed `trajectories.tsv` path (which keys on the
        // same (chain, sweep)). The shared draws loader strips these back out, so
        // every existing reader still sees param-only rows.
        // (Foundation for the keyed-joint (θ, X) output; wired by the join.)
        writeln!(f, "chain\tdraw\t{}", all_names.join("\t")).unwrap();

        // Fixed values (constant across all draws)
        let fixed_vals: Vec<f64> = fixed_names.iter().map(|name| {
            config.compiled.param_index.get(name.as_str())
                .map(|&idx| config.base_params[idx])
                .unwrap_or(0.0)
        }).collect();

        let mut n_draws = 0usize;
        for (chain_id, sweeps, _) in &all_results {
            // `sweeps` is ALREADY the post-burn-in, thinned set — the sim-side
            // recorder applies `burn_in` + `thin` when it builds it
            // (`sim/inference/pgas.rs`: `sweep >= burn_in && (sweep - burn_in) %
            // thin == 0`). Write every retained draw. Re-applying burn_in/thin
            // here (indexing the already-retained list by position) double-
            // filtered the posterior — it dropped the first `burn_in` retained
            // draws (half the cloud at thin=1; ALL of it once thin ≥ the
            // retained count), and desynced `draws.tsv` from the R̂/ESS computed
            // over the full retained set. See
            // docs/dev/incidents/2026-06-28-pgas-draws-double-thinning.md.
            //
            // Each row leads with `(chain, sweep)` — `sweep.sweep` is the draw's
            // true 0-based sweep index (recorded on the draw, not re-derived), so
            // it joins to the same (chain, sweep) in `trajectories.tsv`.
            for sweep in sweeps {
                let mut vals: Vec<String> = config.estimated_params.iter()
                    .map(|spec| format!("{:.17e}", sweep.params[spec.index]))
                    .collect();
                vals.extend(fixed_vals.iter().map(|v| format!("{:.17e}", v)));
                // `chain` is 0-based to MATCH `trajectories.tsv`'s
                // `PosteriorDraw.chain` (also `chain_id`, 0-based) — the join key.
                // (The on-disk `chain_N` dir is 1-based; the in-file key is not.)
                writeln!(f, "{}\t{}\t{}", chain_id, sweep.sweep, vals.join("\t")).unwrap();
                n_draws += 1;
            }
        }
        // Explicit flush: BufWriter swallows write errors on drop, which would
        // silently truncate draws.tsv if the disk filled during the final drain.
        f.flush()
            .map_err(|e| format!("cannot write {}: {}", draws_path.display(), e))?;
        drop(f);
        eprintln!("  draws.tsv: {} posterior samples (all {} params)", n_draws, all_names.len());
    }

    // Render and persist diagnostics
    collector.render_to_stderr();
    let diag_path = stage_dir.join("diagnostics.json");
    let _ = collector.write_json(&diag_path.to_string_lossy());

    let wall_secs = elapsed.as_secs_f64();
    eprintln!("\npgas complete in {:.1}s: {}/", wall_secs, stage_dir.display());
    eprintln!("  best complete-data ll: {:.1} (chain {})",
        best_sweep.log_complete_data_ll, best_chain.0 + 1);

    // gh#278: clean terminal — all output (pgas_summary.json with R̂/ESS,
    // diagnostics.json, draws.tsv) is written above, so the heartbeat's `Done`
    // is the consumer's cue that the final stats are ready to read.
    heartbeat.finish(RunState::Done);

    Ok(())
}

// ── Diagnostics ──────────────────────────────────────────────────

fn compute_diagnostics(
    results: &[(usize, Vec<PGASSweep>, Vec<f64>)],
    estimated_params: &[EstimatedParam],
) -> StageConvergence {
    StageConvergence::compute(estimated_params.iter().map(|spec| {
        let chains: Vec<Vec<f64>> = results.iter()
            .map(|(_, sweeps, _)| sweeps.iter().map(|s| s.params[spec.index]).collect())
            .collect();
        (spec.name.clone(), chains)
    }))
}

fn write_summary(
    dir: &Path,
    results: &[(usize, Vec<PGASSweep>, Vec<f64>)],
    _config: &FitRunConfig,
    thin: usize,
    diagnostics: &StageConvergence,
) -> Result<(), String> {
    let acceptance_rates: Vec<Vec<f64>> = results.iter()
        .map(|(_, _, rates)| rates.clone())
        .collect();

    let summary = serde_json::json!({
        "stage": "pgas",
        "n_chains": results.len(),
        "acceptance_rates": acceptance_rates,
        "rhat": diagnostics.rhat(),
        "rhat_not_reported": diagnostics.rhat_not_reported(),
        "rhat_classic": diagnostics.rhat_classic(),
        "ess": diagnostics.ess_bulk(),
        "ess_tail": diagnostics.ess_tail(),
        "ess_per_chain": diagnostics.ess_per_chain(),
        // Thinning factor: `n_samples` (kept draws) × `thin` = raw sampling
        // iterations, the thinning-invariant denominator for ESS/iteration.
        "thin": thin,
    });

    let path = dir.join(crate::run_meta::FitAlgorithm::Pgas.summary_filename());
    let contents = serde_json::to_string_pretty(&summary)
        .map_err(|e| format!("json error: {}", e))?;
    std::fs::write(&path, contents)
        .map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config_v2::{Stage, StartsFrom};

    fn pgas_stage_with_tempering(tempering: Vec<f64>) -> Stage {
        Stage::PGAS {
            backend: crate::run_meta::InferenceBackend::ChainBinomial,
            chains: 1, particles: 10, sweeps: 10,
            starts_from: StartsFrom::default(),
            init_method: Default::default(),
            survey_path: None,
            survey_top_k_n: None,
            burn_in: Some(2), thin: Some(1),
            tempering,
            max_tree_depth: 10,
            trajectory_warmup: 0,
            csmc_sweeps_per_nuts: 1,
            n_trajectories: 10,
            dense_mass: true,
            use_nuts: true,
        }
    }

    #[test]
    fn tempering_rejects_first_entry_not_one() {
        // First entry MUST be 1.0 (cold chain).
        let stage = pgas_stage_with_tempering(vec![0.7, 0.4]);
        let err = PgasStageOpts::from_stage(&stage).unwrap_err();
        assert!(err.contains("must start with β=1.0"), "got: {}", err);
    }

    #[test]
    fn tempering_rejects_beta_above_one() {
        // β > 1 concentrates likelihood — physically nonsensical.
        let stage = pgas_stage_with_tempering(vec![1.0, 1.5, 0.4]);
        let err = PgasStageOpts::from_stage(&stage).unwrap_err();
        assert!(err.contains("out of range"), "got: {}", err);
        assert!(err.contains("1.5"), "got: {}", err);
    }

    #[test]
    fn tempering_rejects_negative_beta() {
        // β < 0 inverts the likelihood (anti-annealing).
        let stage = pgas_stage_with_tempering(vec![1.0, -0.2]);
        let err = PgasStageOpts::from_stage(&stage).unwrap_err();
        assert!(err.contains("out of range"), "got: {}", err);
    }

    #[test]
    fn tempering_rejects_zero_beta() {
        // β = 0 would scale all log-likelihoods to 0 (uniform), not
        // a valid replica-exchange rung.
        let stage = pgas_stage_with_tempering(vec![1.0, 0.5, 0.0]);
        let err = PgasStageOpts::from_stage(&stage).unwrap_err();
        assert!(err.contains("out of range"), "got: {}", err);
    }

    #[test]
    fn tempering_accepts_well_formed_ladder() {
        // [1.0, 0.7, 0.4, 0.15] — typical 4-rung exchange ladder.
        let stage = pgas_stage_with_tempering(vec![1.0, 0.7, 0.4, 0.15]);
        let opts = PgasStageOpts::from_stage(&stage)
            .expect("well-formed ladder must validate");
        assert_eq!(opts.tempering, vec![1.0, 0.7, 0.4, 0.15]);
    }

    #[test]
    fn tempering_default_single_rung() {
        // Default `[1.0]` (no tempering) must validate.
        let stage = pgas_stage_with_tempering(vec![1.0]);
        let opts = PgasStageOpts::from_stage(&stage)
            .expect("single-rung [1.0] must validate");
        assert_eq!(opts.tempering, vec![1.0]);
    }
}
