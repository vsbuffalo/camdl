//! `camdl fit pmmh` — PMMH posterior sampling.
//!
//! Runs multiple MCMC chains in parallel, each using the bootstrap particle
//! filter as an unbiased likelihood estimator. Outputs per-chain trace files,
//! convergence diagnostics (R̂, ESS), and a summary JSON.

use crate::out::paint::Style;
use crate::chain_index::ChainIndex;
use crate::fit::state::FitState;
use crate::fit::pf_noise;
use crate::fit::report::{ChainWarmup, SamplingCost, SamplingPlan};
use crate::fit::loglik::LoglikType;
use crate::fit::runner::{self, FitRunConfig, MethodConvergence};
use crate::cas::iso8601_utc;
use sim::inference::{
    if2::EstimatedParam,
    pmmh::{run_pmmh, Prior, PMMHConfig, PMMHResult, PMMHResumeState},
    diagnostic::DiagnosticKind,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-method knobs extracted from a `Algorithm::PMMH { ... }` variant by
/// the `camdl fit run` dispatcher and passed verbatim into `run_method`.
/// Mirrors every PMMH field. v1 calls iterations `steps`; v2 calls
/// them `iterations` (we keep the internal `n_steps` name to match
/// `sim::PMMHConfig.n_steps`).
pub struct PmmhMethodOpts {
    pub n_chains: usize,
    pub n_particles: usize,
    pub n_steps: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub adapt: bool,
    pub adapt_start: usize,
}

pub(crate) const DEFAULT_BURN_IN: usize = 5000;
const DEFAULT_THIN: usize = 10;

impl PmmhMethodOpts {
    /// Build from an `Algorithm::PMMH { ... }` or `Algorithm::Mh { ... }`
    /// variant. Errors otherwise — caller's responsibility to dispatch.
    pub fn from_algorithm(algorithm: &super::config_v2::Algorithm) -> Result<Self, String> {
        match algorithm {
            super::config_v2::Algorithm::PMMH {
                chains, particles, iterations, burn_in, thin,
                adapt, adapt_start,
                ..
            } => {
                Ok(PmmhMethodOpts {
                    n_chains: *chains,
                    n_particles: *particles,
                    n_steps: *iterations,
                    burn_in: burn_in.unwrap_or(DEFAULT_BURN_IN),
                    thin: thin.unwrap_or(DEFAULT_THIN),
                    adapt: *adapt,
                    adapt_start: *adapt_start,
                })
            }
            // Deterministic-ODE MH reuses the PMMH machinery via `run_method`'s
            // `is_ode_mh` seam. It carries no `particles` (no PF), so
            // `n_particles` is 0 (unused on the deterministic path).
            super::config_v2::Algorithm::Mh {
                chains, iterations, burn_in, thin,
                adapt, adapt_start,
                ..
            } => {
                Ok(PmmhMethodOpts {
                    n_chains: *chains,
                    n_particles: 0,
                    n_steps: *iterations,
                    burn_in: burn_in.unwrap_or(DEFAULT_BURN_IN),
                    thin: thin.unwrap_or(DEFAULT_THIN),
                    adapt: *adapt,
                    adapt_start: *adapt_start,
                })
            }
            other => Err(format!(
                "PmmhMethodOpts::from_algorithm: expected Algorithm::PMMH or Algorithm::Mh, got {}",
                other.method_name())),
        }
    }
}

// See pgas::run_method for the comment on this allow.
#[allow(clippy::too_many_arguments)]
pub fn run_method(
    fit: &super::config_v2::Problem,
    method: &super::config_v2::Method,
    method_dir: &Path,
    pmmh_opts: PmmhMethodOpts,
    seed: u64,
    force: bool,
    // The leaf `--resume` continues, whose chain files the dispatcher has
    // copied into `method_dir` (`resumed_chain::copy_parent_chains`); `None`
    // for a fresh run.
    resume_base: Option<&Path>,
    // The leaf's resume-compatibility key (`cas::resume_guard`); see
    // `pgas::run_method`.
    resume_guard: &str,
    starts: &super::chain_starts::ResolvedStarts,
    // The base leaf's warm-up record when resuming (`fit::base_warmup`): a
    // resumed chain adapts only over what warm-up its base had left, so its
    // record adds to the base's.
    base_warmup: &[ChainWarmup],
    // Post-fit deterministic ODE dt-check config (gh#52, gh#227). `Some` only on
    // the `mh` (ODE) dispatch; PMMH passes `None` (its dt-check is PF-based and
    // wired on the IF2 path). `--dt-check-strict` is resolved into the config's
    // `threshold_nats` before identity is taken (gh#730), so no flag rides here.
    dt_check_opt: Option<super::config_v2::DtCheckConfig>,
    // The method's liveness/progress heartbeat, owned by the runner
    // (`fit::method_heartbeat`). Its step is one MCMC iteration (gh#900).
    heartbeat: &io::HeartbeatGuard,
    // The command's output handle: the checks the run shows are recorded on
    // it, and its findings go to the collector it holds.
    out: &mut crate::out::Out,
) -> Result<(Vec<ChainWarmup>, Vec<usize>), String> {
    // The PMMH prefer-PGAS-for-long-series caveat banner is emitted by the
    // dispatch chokepoint (`methods::emit_status_banner`), driven by the
    // registry `status_note` so it can't drift from `camdl fit methods`.
    // gh#909: this runner serves `pmmh` and `mh` + `ode`; every label it
    // prints or records is the method that ran, never the runner's name.
    let method_name = method.algorithm.method_name();
    let collector = out.collector();
    let resume = resume_base.is_some();
    let estimate = &fit.estimate;

    let n_chains = pmmh_opts.n_chains;
    let n_steps = pmmh_opts.n_steps;
    let n_particles = pmmh_opts.n_particles;
    let burn_in = pmmh_opts.burn_in;
    let thin = pmmh_opts.thin;
    let adapt = pmmh_opts.adapt;
    let adapt_start = pmmh_opts.adapt_start;

    // Deterministic-ODE Metropolis-Hastings: the method is `Algorithm::Mh`. The MH
    // chain/adaptive-proposal/diagnostics machinery below is shared with PMMH;
    // the ONLY difference is the per-step likelihood evaluation — the
    // deterministic `compute_ode_loglik` instead of the bootstrap particle
    // filter. With no PF the PF-variance preflight is skipped entirely.
    let is_ode_mh = matches!(method.algorithm, super::config_v2::Algorithm::Mh { .. });
    // The class of the log-likelihood the chains' progress reports: a
    // filter's marginal estimate, or the ODE skeleton's.
    let loglik_kind = LoglikType::from(if is_ode_mh {
        crate::run_meta::FitAlgorithm::Mh
    } else {
        crate::run_meta::FitAlgorithm::Pmmh
    });

    // Build FitRunConfig (reuse existing builder). iterations,
    // cooling, cooling_target_iters are IF2-specific and never read
    // by PMMH — pass harmless values.
    let config = FitRunConfig::build(
        fit, Some(&method.algorithm),
        n_chains, n_particles, 1,
        1.0, 1,
        seed, false,
    )?;

    // The registry caveat, with the series length it is about: the data are
    // bound now, so `config.observations` is the series the likelihood sums
    // over. `mh` + `ode` has its caveat said at the dispatch, as every other
    // method does.
    let series_len = config.observations.len();
    if !is_ode_mh {
        super::methods::emit_status_banner(out,
            crate::run_meta::FitAlgorithm::Pmmh, method.algorithm.backend(), Some(series_len));
        if let Some(note) = super::methods::status_note(
            crate::run_meta::FitAlgorithm::Pmmh, method.algorithm.backend())
        {
            out.check(super::report::Check::MethodCaveat {
                method: method_name.to_string(), note: note.to_string(), series_len: Some(series_len),
            });
        }
    }

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
    // One step for both: `FitRunConfig` carries the resolved one (dt_rule).
    let ode_dt: f64 = dt;
    // gh#396 follow-on: coarse warm-up step for the deterministic ODE-MH
    // likelihood. Validated up front (same soundness rules as `nuts`: prevalence
    // only, genuine warm-up window, step LARGER than dt) so a misconfigured fit
    // fails before any chain runs; the per-substep `events`/`balance` refusal is
    // enforced downstream in `run_ode`. `None` ⇒ `ode_dt` (off).
    let ode_burnin_dt: f64 = if is_ode_mh {
        let burnin_opt = match &method.algorithm {
            super::config_v2::Algorithm::Mh { burnin_dt, .. } => *burnin_dt,
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

    // Build proposal SDs
    let proposal_sd = build_proposal_sd(&config);

    // Preflight: likelihood-noise check (skipped for ODE-MH — deterministic,
    // no PF). Measures the spread that governs acceptance and reports it
    // against this run's own target; see `pf_noise`.
    if is_ode_mh {
        out.say("\nODE marginal-likelihood check at base θ (deterministic)...");
    } else {
        out.say(format_args!("\npfilter noise check ({} particles, {} evaluation pairs)...",
            n_particles, pf_noise::NOISE_PAIRS));
    }
    let base = config.base_params.clone();

    // Per-chain starting parameters, through the one seam every runner
    // draws from (`chain_starts::draw_chain_starts`), under the resolved
    // `starts` rule.
    let drawn = super::runner::draw_chain_starts_for(&config, estimate, starts, n_chains, seed)
        .map_err(|e| format!("{method_name}: {e}"))?;
    super::runner::refuse_starts_on_an_end(&config, &drawn).map_err(|e| format!("{method_name}: {e}"))?;
    let chain_starts: Vec<Vec<f64>> = drawn.to_param_vecs(&config.estimated_params, &base);

    let ll_mean: f64;
    // gh#764: the measured spread, carried to the method artifact instead of
    // being printed and dropped. `None` on the ODE-MH path (no filter) and
    // when an evaluation is ruled out (no spread to report).
    let mut pf_noise_check: Option<pf_noise::PfNoiseCheck> = None;
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
        out.say(format_args!("  ODE log L = {:.1} (deterministic; no PF variance)", ll_mean));
    } else {
        // What governs acceptance is the spread of the log-likelihood
        // *difference* between the two θ a Metropolis step compares, so the
        // preflight evaluates pairs: once at the base θ, once at a θ' drawn
        // from the initial proposal, sharing whatever randomness this method's
        // scheme shares. One θ' is drawn and reused across the pairs so the
        // spread is estimator noise and not the curvature of the likelihood
        // surface — see `pf_noise::draw_proposed_theta`.
        let proposed = pf_noise::draw_proposed_theta(
            &base, &config.estimated_params, &proposal_sd, seed);
        let (base_lls, proposed_lls) = pf_noise::measure(
            &config, &base, &proposed, n_particles, pf_noise::NOISE_PAIRS, seed,
        ).map_err(|e| format!(
            "pmmh: structural error during PF-noise check at base θ: {}", e))?;
        ll_mean = base_lls.iter().sum::<f64>() / base_lls.len() as f64;

        match pf_noise::summarize(&base_lls, &proposed_lls, n_particles) {
            Some(check) => {
                out.say_raw(format_args!("{}", pf_noise::report(
                    &check,
                    config.estimated_params.len(),
                    config.observations.len(),
                )));
                out.check(super::report::Check::FilterNoise { check });
                pf_noise_check = Some(check);
            }
            None => {
                // A -inf evaluation is a θ the filter ruled out, not a noise
                // level: there is no spread to take and no ceiling to compute.
                // Say which side it came from, since the two have different
                // fixes — a bad start, versus a proposal whose first move
                // leaves the region the data supports.
                let base_bad = base_lls.iter().any(|l| !l.is_finite());
                let proposed_bad = proposed_lls.iter().any(|l| !l.is_finite());
                let which = match (base_bad, proposed_bad) {
                    (true, true)  => "both the base θ and the proposed θ'",
                    (true, false) => "the base θ",
                    _             => "the proposed θ'",
                };
                out.say(format_args!("  log L̂ mean = {:.1}", ll_mean));
                out.say(format_args!("  {tag} — the filter ruled out {} \
                    (log L̂ = -inf),\n    so there is no spread to take and no acceptance \
                    ceiling to report. The chain\n    still runs; its acceptance rate is then \
                    the only evidence on whether it moves.",
                    which,
                    tag = crate::out::paint::stderr().paint("⚠ no noise measurement", Style::Warn)));
            }
        }
    }


    if !force && !resume {
        let state_path = method_dir.join("fit_state.toml");
        if state_path.exists() {
            out.say(format_args!("{}", crate::out::paint::stderr().paint(&format!(
                "{method_name} results already exist in {}. Use --force to re-run or --resume to continue.",
                method_dir.display()), Style::Warn)));
            return Ok((Vec::new(), Vec::new()));
        }
    }

    std::fs::create_dir_all(method_dir)
        .map_err(|e| format!("cannot create {}: {}", method_dir.display(), e))?;

    // The audit sidecar, captured before any step runs so a watcher can read
    // the starts while the chains are still going, and rewritten from the
    // final record once they finish (gh#887, gh#904). The pre-run write can
    // only say `accepted` — no chain has been refused yet — so the rewrite is
    // what makes the file true, and it is unconditional.
    super::runner::record_chain_starts(out, method_dir, &config, &drawn);
    let retry_accepted: std::sync::Mutex<Vec<Option<super::chain_starts::ChainStart>>> =
        std::sync::Mutex::new(vec![None; n_chains]);
    let retry_rejected: std::sync::Mutex<Vec<super::chain_starts::RejectedStart>> =
        std::sync::Mutex::new(Vec::new());
    let retry_refused: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let can_redraw = super::chain_starts::can_redraw(&starts.rule, n_chains);

    // Resolve priors: fit.toml override → model IR → Flat
    let priors: Vec<Prior> = config.estimated_params.iter()
        .map(|spec| super::runner::resolve_prior(&spec.name, estimate, &config.model).0)
        .collect();

    // Active interventions + events — see same block in pgas.rs.
    crate::util::print_scheduled_actions_summary(out, &config.model_declared, &config.model);
    crate::util::print_observations_summary(out, &config.model);
    for c in super::report::obs_stream_checks(&config.model) {
        out.check(c);
    }

    let dt = config.if2_config.dt;

    // The resume key: this leaf's CAS identity less its extension dimension
    // (`cas::resume_guard`), so resume accepts a chain state exactly when
    // every input but the iteration count is the one the base ran under.
    let config_hash = resume_guard.to_string();

    // Load resume states if --resume
    let resume_states: Vec<Option<PMMHResumeState>> = if resume {
        let mut states = Vec::with_capacity(n_chains);
        let mut any_failed = false;
        for chain in ChainIndex::all(n_chains) {
            let path: PathBuf = chain.dir_in(method_dir).join("resume_state.bin");
            match std::fs::read(&path) {
                Ok(data) => match bincode::deserialize::<PMMHResumeState>(&data) {
                    Ok(state) => {
                        if state.config_hash != config_hash {
                            super::abandon_method(out, format!(
                                "error: config hash mismatch for chain {} — \
                                cannot resume. Either the model/data/priors changed \
                                since the original run, or this chain predates a camdl \
                                version that changed how a method's identity is computed \
                                (the 2026-08-23 subtractive-identity change re-keyed \
                                every pgas/pmmh/mh/nuts method). Re-run from scratch \
                                with --force.",
                                chain));
                        }
                        out.say(format_args!("  chain {}: resuming from step {}", chain, state.completed_steps));
                        if let Some(gap) = sim::inference::types::z_restore_gaps(
                            &state.param_names, &state.transformed, &config.estimated_params)
                        {
                            out.say(gap);
                        }
                        if state.completed_steps >= n_steps {
                            out.say(format_args!("  warning: chain already completed {} steps (requested {}). \
                                       Increase steps in fit.toml to continue.",
                                state.completed_steps, n_steps));
                        }
                        states.push(Some(state));
                    }
                    Err(e) => {
                        out.say(format_args!("error: cannot deserialize resume state for chain {}: {}. \
                            Resume state format may have changed — re-run with --force.", chain, e));
                        any_failed = true;
                        states.push(None);
                    }
                }
                Err(_) => {
                    out.say(format_args!("error: no resume state file for chain {} ({})",
                        chain, path.display()));
                    any_failed = true;
                    states.push(None);
                }
            }
        }
        if any_failed {
            super::abandon_method(out,
                "error: --resume requires resume state files for all chains.\n  \
                 These are written automatically at the end of every PMMH run.\n  \
                 If the original run was interrupted before saving, use --force to \
                 start fresh.".to_string());
        }
        states
    } else {
        vec![None; n_chains]
    };

    // The ODE path runs no particle filter, so it has no particle count.
    let particles = if is_ode_mh { String::new() } else { format!(" × {n_particles} particles") };
    out.say(format_args!("\n{method_name}: {n_chains} chains × {n_steps} steps{particles}, \
               burn_in={burn_in}, thin={thin}, adapt={adapt}"));
    out.say(format_args!("  proposal_sd (transformed): [{}]",
        config.estimated_params.iter().zip(&proposal_sd)
            .map(|(p, &sd)| format!("{}={:.4}", p.name, sd))
            .collect::<Vec<_>>().join(", ")));

    // One `Reporter` hands out a per-chain `Task` rendered as a coordinated
    // stack. The Reporter honors --progress (Pretty=bars, Plain=throttled
    // `chain N pos/len ll=… acc=…%` log lines, None=silent), so the callback
    // no longer branches on mode.
    let reporter = crate::out::progress::Reporter::new();
    let bars: Vec<crate::out::progress::Task> = (0..n_chains)
        .map(|chain_id| reporter.task(n_steps as u64, format!("chain {}", ChainIndex(chain_id)), "it")
            .starting_at(resume_states[chain_id].as_ref().map_or(0, |s| s.completed_steps as u64))
            .recording(out.events(), chain_id))
        .collect();

    // Pre-create chain directories
    for chain in ChainIndex::all(n_chains) {
        let chain_dir = chain.dir_in(method_dir);
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
    // than reporting a degenerate posterior — no further chain starts after
    // the first such error. At most one chain per worker, so a started chain
    // is never displaced by another (gh#821, `chain_fanout`).
    let results: Vec<(usize, PMMHResult)> =
        super::chain_fanout::try_run_chains(n_chains, |chain_id| -> Result<Option<(usize, PMMHResult)>, String> {
            let chain_seed = crate::util::derive_chain_seed(seed, chain_id);
            // gh#887: the start this chain runs from — the draw, or a redraw
            // the init-eval guard below accepted after refusing earlier ones.
            let mut start_vec = chain_starts[chain_id].clone();

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
            // A start where some prior has zero density cannot be repaired by
            // the MH ratio when every proposal is also outside that prior's
            // support (`-inf - -inf` is NaN, never accepted), and the
            // likelihood-only backstop below (gh#226) does not see it. Refuse
            // it here, naming the parameter, as PGAS does
            // (`SimError::NonFiniteChainStart`).
            if resume_states[chain_id].is_none() {
                start_prior_is_finite(&config.estimated_params, &priors,
                                      &config.param_names, &start_vec)
                    .map_err(|e| format!("chain {}: {}", ChainIndex(chain_id), e))?;
            }
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
                            ChainIndex(chain_id), e));
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
                        out.say(format_args!(
                            "  {w}: chain {} starts at a θ with \
                             -inf log-likelihood — the start predicts a trajectory that \
                             can't explain the data (e.g. an epidemic that goes extinct \
                             where the data has cases). The chain can only mix if a \
                             finite-likelihood region is reachable from there; if its \
                             acceptance rate stays 0%, propose a better start (a \
                             multi-start `init`, or adjust the start toward a θ that \
                             sustains the epidemic).",
                            ChainIndex(chain_id),
                            w = crate::out::paint::stderr().paint("warning", Style::Warn)));
                    }
                    _ => {
                        // Ok(finite) or a recoverable Err (ruled-out θ) — MH
                        // proceeds via the standard accept/reject path.
                    }
                }
            } else if resume_states[chain_id].is_none() {
                let mut attempt = 0usize;
                loop { match runner::run_quick_pfilter_with_dt(
                    &config, &start_vec,
                    n_particles, None, chain_seed.wrapping_add(attempt as u64),
                ) {
                    Err(e @ sim::error::SimError::PFDegenerate { .. })
                        if can_redraw && attempt + 1 < super::chain_starts::MAX_START_ATTEMPTS =>
                    {
                        let reason = match &e {
                            sim::error::SimError::PFDegenerate { kind, obs_window, elapsed_s } =>
                                format!("{:?} at obs_window={} after {:.2}s",
                                    kind, obs_window, elapsed_s),
                            _ => unreachable!(),
                        };
                        let values: std::collections::HashMap<String, f64> = config
                            .estimated_params.iter()
                            .map(|spec| (spec.name.clone(), start_vec[spec.index]))
                            .collect();
                        retry_rejected.lock().unwrap().push(super::chain_starts::RejectedStart {
                            chain_id, attempt, values, reason: reason.clone(),
                            ess: runner::ess_at_refusal(&e),
                        });
                        out.say(format_args!(
                            "  chain {}: start {} of {} refused by the filter ({}); drawing another",
                            ChainIndex(chain_id), attempt + 1, super::chain_starts::MAX_START_ATTEMPTS,
                            reason));
                        attempt += 1;
                        let cs = super::runner::redraw_chain_start_for(
                            &config, estimate, starts, n_chains, chain_id, seed, attempt,
                        )?;
                        start_vec = cs.to_param_vec(&config.estimated_params, &config.base_params);
                        retry_accepted.lock().unwrap()[chain_id] = Some(cs);
                        continue;
                    }
                    Err(e @ sim::error::SimError::PFDegenerate { .. }) => {
                        // gh#904: unconditionally. A chain refused on its only
                        // attempt — the whole of the point rules, which do not
                        // redraw — is as refused as one refused after three,
                        // and this is the only channel by which
                        // `chain_starts.tsv` learns the chain never ran.
                        retry_refused.lock().unwrap().push(chain_id);
                        // A statistically-degenerate init (ESS collapse / all
                        // particles dead) is skipped with a BadInit diagnostic;
                        // the surviving chains continue. (PFIterationBudget is a
                        // deterministic compute-budget bail → structural/fatal,
                        // handled by the structural arm.)
                        let tried = if attempt > 0 {
                            format!("none of {} starts drawn under `starts = {}` could be \
                                     scored by the filter; the last: ",
                                attempt + 1, starts.rule.spelled())
                        } else {
                            String::new()
                        };
                        let reason = match &e {
                            sim::error::SimError::PFDegenerate { kind, obs_window, elapsed_s } =>
                                format!("{}{:?} at obs_window={} after {:.2}s",
                                    tried, kind, obs_window, elapsed_s),
                            _ => unreachable!(),
                        };
                        let params: std::collections::BTreeMap<String, f64> =
                            config.estimated_params.iter()
                                .map(|spec| (
                                    spec.name.clone(),
                                    start_vec[spec.index],
                                ))
                                .collect();
                        // A PF degeneracy, not a swarm that lost support at
                        // a named measurement: nothing re-scored an observation
                        // here, so there is no `StreamAttempt` to carry.
                        collector.push(DiagnosticKind::BadInit {
                            // The chain's number, as the stderr line below and
                            // every other artifact name it (gh#990).
                            chain_id, params, reason: reason.clone(),
                            attempts: Vec::new(),
                        });
                        out.say(format_args!("  chain {}: {} — {}",
                            ChainIndex(chain_id),
                            crate::out::paint::stderr().paint("✗ BadInit", Style::Err), reason));
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
                            ChainIndex(chain_id), other));
                    }
                    Ok(_) => {
                        // Finite (or even -inf-but-not-degenerate)
                        // initial loglik — PMMH can proceed. PMMH's
                        // MH ratio handles uninformative inits via
                        // the standard accept/reject path.
                        break;
                    }
                } }
            }

            let pmmh_config = PMMHConfig {
                n_steps,
                n_particles,
                dt,
                proposal_sd: proposal_sd.clone(),
                adapt,
                adapt_start,
                // Warm-up is the discarded prefix: adaptation runs over the
                // burn-in and the kernel is frozen for the draws the fit
                // keeps. This is the Stan arrangement, and it is a policy of
                // this driver rather than a fact about the sampler, which
                // takes the boundary as its own `adapt_stop`. No new
                // configuration key, so no new run-identity input — the
                // boundary moves only when `burn_in` does, and `burn_in` is
                // already part of a method's identity payload.
                //
                // How much of a run this actually freezes is therefore set by
                // `burn_in`, whose default is `DEFAULT_BURN_IN` = 5000 while
                // config validation requires `burn_in < iterations`. A method
                // that leaves it unset and asks for 5,001 iterations adapts
                // over 5,000 of them and freezes for one; the same default at
                // 20,000 iterations freezes three quarters of the run. The
                // freeze is only as useful as the warm-up budget is honest,
                // and whether 5,000 is the right default is a separate
                // question from whether adaptation should stop.
                adapt_stop: burn_in,
                thin,
                burn_in,
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

            let task = &bars[chain_id];
            let accepted_count = AtomicUsize::new(0);

            // Streaming trace: use TraceWriter with append mode when resuming
            let chain_dir = ChainIndex(chain_id).dir_in(method_dir);
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
                            let z = spec.to_transformed(params[spec.index]);
                            prior.log_density_env(&spec.prior_point(z), &env)
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
                task.set(crate::out::progress::mcmc(loglik, loglik_kind, acc));
                task.inc(1);

                // gh#900: and to the run's heartbeat, whose step is one MCMC
                // iteration and whose phase reads `burn_in` off the same
                // boundary this callback does. A monotonic `fetch_max` across
                // the parallel chains — a cheap atomic, no I/O on the step
                // path, so it cannot touch a fit number.
                heartbeat.bump(step as u64);
            };

            // gh#224: a structural error from the sampling PF aborts the fit
            // (config/model can't run); a ruled-out θ is handled internally as
            // a rejected −∞ proposal, so `run_pmmh` only `Err`s on structural.
            let result = run_pmmh(
                &config.estimated_params, &priors, &start_vec,
                &config.param_names,
                &pmmh_config, eval_loglik.as_ref(), chain_seed,
                Some(&progress_cb), resume_states[chain_id].clone(), config_hash.clone(),
            ).map_err(|e| format!("chain {} failed with structural error: {}", ChainIndex(chain_id), e))?;

            // Final metric (MAP ll + acceptance) on the bar; the driver clears
            // it after the fan-out (`Task::finish` consumes, so it can't run
            // on the borrowed Task here). The `acceptance rates:` report below
            // carries the per-chain summary.
            task.set(crate::out::progress::mcmc(result.map_loglik, loglik_kind, result.acceptance_rate));
            // Said now, by the same predicate that pushes the finding after the
            // join, so a line here is exactly a floor finding at the end.
            if let Some(DiagnosticKind::ScaleAtFloor { steps_at_floor, warmup_steps, .. }) =
                scale_at_floor(chain_id, &result)
            {
                task.live_line(&format!(
                    "  chain {}: proposal scale at its floor for {steps_at_floor} of {warmup_steps} \
                     warm-up steps (details at the end)", ChainIndex(chain_id)));
            }

            // Save resume state for future --resume
            let resume_path = chain_dir.join("resume_state.bin");
            if let Ok(encoded) = bincode::serialize(&result.resume_state) {
                let _ = std::fs::write(&resume_path, encoded);
            }

            Ok(Some((chain_id, result)))
        })?
        .into_iter()
        .flatten()
        .collect();

    // Clear all chain bars now that the parallel phase is done (`Task::finish`
    // consumes, so it can't run on the per-chain borrow inside the loop). The
    // `acceptance rates:` report below carries the per-chain summary.
    for t in bars { t.finish(); }

    // gh#887: the sidecar records every start a retry rejected, and the start
    // each chain actually ran from. gh#904: always rewritten, not only when a
    // retry rejected something — a chain refused at its only start leaves
    // `rejected` empty, and the pre-run file would otherwise stand, saying
    // `accepted` for a chain that never ran.
    let drawn = drawn.with_retry_outcome(
        retry_accepted.into_inner().unwrap(),
        retry_rejected.into_inner().unwrap(),
        retry_refused.into_inner().unwrap(),
    );
    super::runner::record_chain_starts(out, method_dir, &config, &drawn);

    // gh#110. Skip + continue: surface "ran K of N chains" so the
    // user knows downstream R̂/ESS exclude skipped chains. This
    // line is the load-bearing user-facing signal that bad inits
    // were tolerated rather than silently dropped.
    let n_good_chains = results.len();
    if n_good_chains < n_chains {
        out.say(format_args!("\n{} \
                   ({} skipped via BadInit; see diagnostics below)",
            crate::out::paint::stderr().paint(&format!("ran {n_good_chains} of {n_chains} chains"), Style::Warn),
            n_chains - n_good_chains));
    }

    let elapsed = t0.elapsed();

    // Traces already written by streaming callback — no post-hoc write needed.

    // What each chain's warm-up did, recorded beside the draws so the
    // condition a stuck chain leaves is readable without its stderr.
    let warmup = chain_warmups(&results, adapt, resume, base_warmup);
    let plan = SamplingPlan {
        n_chains, n_steps, n_particles, burn_in, thin, adapt, series_len,
        proposal_sd: config.estimated_params.iter().zip(&proposal_sd)
            .map(|(p, &sd)| (p.name.clone(), sd))
            .collect(),
    };
    // The steps each chain that ran took in this call: a resumed chain starts
    // where its base stopped, and a refused chain ran none (gh#1012).
    let steps_run: Vec<usize> = results.iter()
        .map(|(chain_id, _)| {
            // A base already at or past `n_steps` runs none (a warning above).
            n_steps.saturating_sub(resume_states[*chain_id].as_ref().map_or(0, |s| s.completed_steps))
        })
        .collect();
    let cost = sampling_cost(elapsed.as_secs_f64(), &steps_run);

    // The step each resumed chain began this call at: its base's completed
    // steps. The rows below it in `chain_N/trace.tsv` are the base's.
    let boundary: Vec<usize> = resume_states.iter()
        .map(|s| s.as_ref().map_or(0, |s| s.completed_steps))
        .collect();
    // A resumed chain's retained draws from its base, read back from the
    // leaf's own trace (`resumed_chain`), so R̂ and ESS are over the whole
    // chain the leaf holds rather than the steps this call ran.
    let param_names: Vec<String> = config.estimated_params.iter().map(|s| s.name.clone()).collect();
    let inherited: Vec<Option<super::resumed_chain::InheritedDraws>> = results.iter()
        .map(|(chain_id, _)| -> Result<_, String> {
            if resume_states[*chain_id].is_none() {
                return Ok(None);
            }
            super::resumed_chain::inherited_draws(
                &ChainIndex(*chain_id).dir_in(method_dir).join("trace.tsv"),
                "step", boundary[*chain_id], burn_in, thin, &param_names,
            ).map(Some)
        })
        .collect::<Result<_, String>>()?;

    // Compute diagnostics
    let diagnostics = compute_diagnostics(&results, &inherited, &config.estimated_params);

    // Report. PMMH's θ-move is random-walk MH, so its band is the random-walk
    // one — taken from `AcceptanceKernel` rather than spelled here, so the
    // coloring, the finding and the message a user reads cannot disagree about
    // where the band starts (gh#299 item 3).
    let kernel = sim::inference::diagnostic::AcceptanceKernel::RandomWalk;
    // The table — each chain's rate, and the scale its draws were proposed
    // under with its warm-up's floor count — is drawn at the end of the run
    // from the recorded rates (`report::render::acceptance_table`).
    for (chain_id, result) in &results {
        if let Some(d) = sim::inference::diagnostic::acceptance_diagnostic(
            Some(*chain_id), result.acceptance_rate, None, kernel)
        {
            out.finding(d);
        }
    }

    if n_chains > 1 {
        // gh#890, as pgas: a point start makes the between-chain statistic
        // uninformative, `fit summary` reports it as not assessed, and this
        // block says the same rather than a number the summary declines.
        let withheld = starts.rule.is_point().then(|| {
            sim::inference::convergence::ConvergenceError::PointStart {
                n_chains, starts: starts.rule.spelled(),
            }
        });
        // RHAT_REPORT_THRESHOLD is unchanged from the value this method has
        // always applied; only the STATISTIC it is applied to changed (gh#84).
        // The table is drawn at the end of the run, from the summary written
        // below (`report::render::rhat_run_table`).
        for kind in diagnostics.rhat_findings(super::runner::RHAT_REPORT_THRESHOLD, withheld.as_ref()) {
            out.finding(kind);
        }
    }

    // What each chain's warm-up says about its draws, after the chains join
    // and in chain order. Judged here from the chain's result, not inside the
    // sampler, so the finding can be recorded and printed once.
    if adapt {
        for (chain_id, r) in &results {
            if let Some(kind) = scale_at_floor(*chain_id, r).or_else(|| scale_far_from_optimum(*chain_id, r)) {
                out.finding(kind);
            }
        }
    }

    // gh#110. All chains skipped via BadInit — no MAP to report.
    // Render diagnostics, persist them, and bail with a clear error
    // rather than panicking on .unwrap() of an empty results vec.
    if results.is_empty() {
        // gh#880, gh#885. The reason and the remedies come from the one
        // string the other two all-chains-refused errors also carry, so they
        // cannot drift apart again.
        return Err(format!(
            "pmmh method `{}`: all {} chains failed init-eval with \
             PFDegenerate — the particle filter could not score any chain's \
             starting parameters. See `diagnostics.json` for the per-chain \
             BadInit entries and `chain_starts.tsv` for the starts they name. \
             {}",
            method_name, n_chains,
            crate::fit::chain_starts::UNSCOREABLE_START_ADVICE));
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
        out.finding(DiagnosticKind::InitialLoglikInfinite);
        return Err(format!(
            "pmmh method `{}`: all {} surviving chain(s) reached no finite \
             log-likelihood anchor — every evaluated θ scored -inf, so the \
             MH chain never moves (-inf - (-inf) = NaN, never accepted) and \
             the posterior is degenerate. Most often the starting values sit \
             in an impossible region — an MH chain cannot escape a -inf init — \
             so check those first (try `--starts lhs`, or `starts = \"lhs\"` \
             under `[method]`, or a different start); less often the data are \
             impossible under this model, or a recoverable \
             error fires at every θ (gh#226). Also check the observation model \
             and parameter bounds; run with --verbosity debug for per-substep \
             diagnostics.",
            method_name, results.len()));
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
    write_summary(method_dir, &results, &config, &diagnostics, algo, &warmup, &plan, &cost)?;

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
        (true, Some(cfg)) => {
            let obs_model = ode_obs_model.as_deref()
                .expect("ode_obs_model built on the is_ode_mh path");
            let result = super::dt_check::run_richardson_ladder_ode(
                config.compiled.as_ref(),
                obs_model,
                &ode_obs_times,
                &map_result.map_params,
                ode_dt,
                cfg,
            )?;
            super::dt_check::print_terminal_report(out, &result);
            if matches!(result.verdict, super::dt_check::DtCheckVerdict::Skipped) {
                None
            } else {
                Some(result)
            }
        }
        _ => None,
    };

    let state = FitState {
        method: method_name.to_string(),
        seed,
        timestamp: iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.into()),
        best_loglik: map_result.map_loglik,
        initial_loglik: ll_mean,
        best_chain: ChainIndex(*map_chain),
        n_chains,
        // gh#110. Surface "ran K of N chains" through fit_state so
        // downstream consumers (fit_summary, the book chapters) can
        // print "ran K of N chains" without inferring it from
        // missing chain dirs.
        n_good_chains: if n_good_chains < n_chains { Some(n_good_chains) } else { None },
        start_values,
        // A filter's marginal for PMMH, the ODE skeleton's for `mh`.
        loglik_type: Some(loglik_kind),
        acceptance_rate: Some(map_result.acceptance_rate),
        // Bayesian methods don't produce an IF2-style Â table; the
        // refine-gates proposal only gates on scout→refine handoffs.
        tail_chain_agreement: std::collections::BTreeMap::new(),
        perturb_only_at_t0_params: Vec::new(),
        chain_logliks: Vec::new(),
        chain_eval_logliks: Vec::new(),
        chain_eval_ids: Vec::new(),
        chain_eval_ses: Vec::new(),
        // Bayesian path — compound gate doesn't apply to PMMH.
        resolved_gate: None,
        resolved_loglik_eval: None,
        // The rule that supplied the starts, as `chain_starts.tsv`'s header
        // spells it, and its kind — what `fit summary` reads to say whether R̂
        // was assessable (gh#871, gh#873, proposal 2026-09-08 §3.4).
        chain_init_source: Some(drawn.rule.spelled()),
        chain_starts_kind: Some(drawn.rule.kind()),
        // gh#52, gh#227: deterministic ODE dt-check at the MAP (above); `None`
        // on the PMMH path (PF dt-check is wired on the IF2 path).
        dt_check: dt_check_result,
        // gh#764: the preflight's measured spread, with the particle and pair
        // counts it was measured at. Without this the one number that explains
        // a stuck pseudo-marginal chain exists only in the run's stderr.
        pf_noise: pf_noise_check,
        optimizer_convergence: None,
    };
    state.save(&method_dir.to_string_lossy())?;

    // Write draws.tsv: complete-M posterior draws (all params, estimated + fixed)
    // Reads the per-chain trace.tsv files (which now also carry warm-up rows),
    // keeps only the post-burn-in tail (`step >= burn_in`), and adds fixed
    // parameter columns.
    {
        use std::io::Write;
        let draws_path = method_dir.join("draws.tsv");
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
        for chain in ChainIndex::all(n_chains) {
            let trace_path = chain.dir_in(method_dir).join("trace.tsv");
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
                        None => { out.say("warning: trace.tsv row missing step index; skipping"); continue; }
                    };
                    if step < burn_in { continue; }  // warm-up: not a posterior draw
                    let mut vals: Vec<String> = param_col_indices.iter().map(|&col_idx| {
                        if col_idx < fields.len() {
                            fields[col_idx].to_string()
                        } else {
                            out.say(format_args!("warning: trace.tsv missing column at index {}", col_idx));
                            "NaN".to_string()
                        }
                    }).collect();
                    vals.extend(fixed_vals.iter().map(|v| format!("{:.17e}", v)));
                    // The chain's number, the same one its directory carries.
                    writeln!(f, "{}\t{}\t{}", chain, draw_idx, vals.join("\t")).unwrap();
                    draw_idx += 1;
                    n_draws += 1;
                }
            }
        }
        drop(f);
        out.say(format_args!("  draws.tsv: {} posterior samples (all {} params)", n_draws, all_names.len()));
    }

    // Persist the findings. The end of the run prints them once, merged,
    // with the cost of sampling and the highest-posterior draw after them
    // (`report::render`'s run-end view, from what is recorded here).
    super::write_diagnostics(&collector, method_dir)?;

    Ok((warmup, if resume { boundary } else { Vec::new() }))
}

/// The `ScaleAtFloor` finding a chain's result raises: its scale sat at the
/// lower bound for some warm-up step. The one predicate both the live line and
/// the recorded finding are decided by.
pub(crate) fn scale_at_floor(chain: usize, r: &PMMHResult) -> Option<DiagnosticKind> {
    (r.steps_at_scale_floor > 0).then_some(DiagnosticKind::ScaleAtFloor {
        chain_id: chain,
        steps_at_floor: r.steps_at_scale_floor,
        warmup_steps: r.warmup_steps,
        target_accept: r.target_accept,
    })
}

/// The `ScaleFarFromOptimum` finding: the chain finished far from λ = 1
/// without reaching the floor. Never raised beside [`scale_at_floor`]'s, which
/// says the same thing more precisely.
fn scale_far_from_optimum(chain: usize, r: &PMMHResult) -> Option<DiagnosticKind> {
    (r.steps_at_scale_floor == 0 && sim::inference::pmmh::scale_is_far_from_optimum(r.final_scale))
        .then_some(DiagnosticKind::ScaleFarFromOptimum { chain_id: chain, final_scale: r.final_scale })
}

/// Build proposal SDs on the transformed scale from `[estimate]`'s
/// `rw_sd`, scaled up for MH jumps: IF2's `rw_sd` is per-perturbation-step,
/// PMMH needs per-proposal (larger). The adaptive scheme (`adapt = true`)
/// takes over from here.
fn build_proposal_sd(config: &FitRunConfig) -> Vec<f64> {
    config.estimated_params.iter().map(|p| {
        p.mh_initial_step()
    }).collect()
}

/// R̂ / ESS over each chain's retained draws: a resumed chain's `inherited`
/// rows from its base (parallel to `results`, in `estimated_params` order),
/// then the draws this call kept.
fn compute_diagnostics(
    results: &[(usize, PMMHResult)],
    inherited: &[Option<super::resumed_chain::InheritedDraws>],
    estimated_params: &[EstimatedParam],
) -> MethodConvergence {
    MethodConvergence::compute(estimated_params.iter().enumerate().map(|(p, spec)| {
        let chains: Vec<Vec<f64>> = results.iter().zip(inherited)
            .map(|((_, r), base)| {
                base.iter().flat_map(|b| b.column(p))
                    .chain(r.steps.iter().map(|s| s.params[spec.index]))
                    .collect()
            })
            .collect();
        (spec.name.clone(), chains)
    }))
}

// write_chain_traces removed — streaming callback now handles trace output
// with correct log_posterior and burn-in/thin filtering.

/// Each chain's warm-up record, in chain order; empty when `adapt = false`.
///
/// A resumed chain's record continues its base's
/// ([`ChainWarmup::continued_from`]). A resumed chain whose base carries no
/// record (written before it existed) gets none: its own call saw only the
/// tail of the warm-up, and reporting that as the whole would understate it.
fn chain_warmups(
    results: &[(usize, PMMHResult)],
    adapt: bool,
    resumed: bool,
    base: &[ChainWarmup],
) -> Vec<ChainWarmup> {
    if !adapt {
        return Vec::new();
    }
    results.iter()
        .filter_map(|(chain, r)| {
            let this_call = ChainWarmup {
                chain: *chain,
                warmup_steps: r.warmup_steps,
                steps_at_scale_floor: r.steps_at_scale_floor,
                final_scale: r.final_scale,
                target_accept: r.target_accept,
            };
            if !resumed {
                return Some(this_call);
            }
            base.iter().find(|b| b.chain == *chain)
                .map(|b| ChainWarmup::continued_from(b, &this_call))
        })
        .collect()
}

/// The sampling cost of this call, which the completion line reports, from
/// the MCMC steps each chain that ran took in it (`steps_run`, one entry per
/// chain): one likelihood evaluation per step, summed over the chains, and
/// wall time per evaluation within one chain, the chains running in parallel.
///
/// The cost is this call's, not the chain's since its first step: a resumed
/// run counts the steps it added, since `wall_secs` is the time of this call
/// alone and a per-evaluation time needs both from the same call (gh#1012).
/// The start check and the MAP re-evaluations are not counted.
fn sampling_cost(wall_secs: f64, steps_run: &[usize]) -> SamplingCost {
    let evaluations: usize = steps_run.iter().sum();
    SamplingCost {
        sampling_seconds: wall_secs,
        evaluations,
        ms_per_evaluation: wall_secs * 1000.0 / evaluations as f64 * steps_run.len() as f64,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    dir: &Path,
    results: &[(usize, PMMHResult)],
    config: &FitRunConfig,
    diagnostics: &MethodConvergence,
    algo: crate::run_meta::FitAlgorithm,
    warmup: &[ChainWarmup],
    plan: &SamplingPlan,
    cost: &SamplingCost,
) -> Result<(), String> {
    let thin = plan.thin;
    let acceptance_rates: Vec<f64> = results.iter().map(|(_, r)| r.acceptance_rate).collect();

    let (map_chain, map_result) = results.iter()
        .max_by(|a, b| a.1.map_log_posterior.total_cmp(&b.1.map_log_posterior))
        .unwrap();

    let map_params: HashMap<String, f64> = config.estimated_params.iter()
        .map(|spec| (spec.name.clone(), map_result.map_params[spec.index]))
        .collect();

    let mut summary = serde_json::json!({
        // gh#728: the tag a consumer keys on to know which statistics the
        // `rhat` and `ess` keys below hold. `serde_json` writes a map in key
        // order, so it lands where the sort puts it rather than first.
        "schema": crate::run_meta::FIT_SUMMARY_SCHEMA,
        // `algo`, not the literal `"pmmh"`: deterministic mh-ODE shares this
        // runner and writes its own `mh_summary.json`, so a hardcoded label
        // put `pmmh` inside a file named `mh`. The key now names the method
        // (gh#901), which makes a wrong value a contradiction rather than a
        // cosmetic one.
        "method": algo.as_str(),
        "n_chains": results.len(),
        "steps_per_chain": results.first().map(|(_, r)| r.n_steps).unwrap_or(0),
        "acceptance_rate": acceptance_rates,
        // The chain each rate belongs to: refused chains leave gaps, so a
        // position in `acceptance_rate` is not a chain number.
        "acceptance_chains": results.iter().map(|(c, _)| *c).collect::<Vec<usize>>(),
        "map_loglik": map_result.map_loglik,
        "map_chain": ChainIndex(*map_chain),
        "map_params": map_params,
        // n_samples (kept) × thin = raw sampling iterations → ESS/iteration.
        "thin": thin,
        "warmup": warmup,
        "plan": plan,
        "cost": cost,
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

/// The run's closing stderr line: which method ran, how long it took, and how
/// many likelihood evaluations that was (gh#909). This runner serves both
/// `pmmh` and `mh` + `ode`, so the method is `method_name`, never a literal,
/// and the evaluations are named by what computed them: a particle filter's
/// estimate for PMMH, the deterministic ODE marginal likelihood for `mh`.
/// The per-evaluation time is wall time per evaluation within one chain, the
/// chains running in parallel.
///
/// The leaf's path is not on this line: the `stored` milestone that closes the
/// run carries it.
pub(crate) fn completion_line(method_name: &str, is_ode_mh: bool, cost: &SamplingCost) -> String {
    let what = if is_ode_mh { "ODE likelihood evaluations" } else { "PF evaluations" };
    format!("{method_name} complete in {:.1}s ({} {what}, {:.1}ms/eval)",
        cost.sampling_seconds, cost.evaluations, cost.ms_per_evaluation)
}

/// `Err` naming every estimated parameter whose prior log-density is not
/// finite at `start` (a start outside the prior's support, or a hierarchical
/// prior whose arguments evaluate out of their domain there).
fn start_prior_is_finite(
    specs: &[EstimatedParam],
    priors: &[Prior],
    names: &[String],
    start: &[f64],
) -> Result<(), String> {
    let env = sim::inference::hierarchical::NamedParams { names, values: start };
    let bad: Vec<String> = specs.iter().zip(priors.iter())
        .filter_map(|(spec, prior)| {
            let z = spec.to_transformed(start[spec.index]);
            let lp = prior.log_density_env(&spec.prior_point(z), &env);
            (!lp.is_finite()).then(|| format!("'{}' = {}", spec.name, start[spec.index]))
        })
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the start has zero prior density (log prior -inf) at {}: an MH chain \
             started there cannot move. Give a start inside every prior's support, \
             or check the hierarchical prior arguments evaluated at this start.",
            bad.join(", ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#909: an `mh` + `ode` run is named `mh`, and counts likelihood
    /// evaluations, not particle-filter ones: there is no filter on that
    /// path. The PMMH line is the negative control: it still says what it
    /// did.
    #[test]
    fn completion_line_names_the_method_and_its_evaluations() {
        let mh = completion_line("mh", true, &sampling_cost(0.3, &[3000]));
        assert!(mh.starts_with("mh complete in 0.3s"), "{mh}");
        assert!(!mh.contains("pmmh") && !mh.contains("PF"), "{mh}");
        assert!(mh.contains("3000 ODE likelihood evaluations"), "{mh}");

        let pmmh = completion_line("pmmh", false, &sampling_cost(2.0, &[500, 500]));
        assert!(pmmh.starts_with("pmmh complete in 2.0s"), "{pmmh}");
        assert!(pmmh.contains("1000 PF evaluations"), "{pmmh}");
        assert!(pmmh.ends_with("ms/eval)"), "the leaf's path is the `stored` line's: {pmmh}");
    }
    /// gh#1012: a resumed call costs the steps it ran, not the chain's length,
    /// and its time per evaluation divides this call's time by them.
    #[test]
    fn a_resumed_call_costs_the_steps_it_ran() {
        let cost = sampling_cost(1.0, &[500, 500, 500, 500]);
        assert_eq!(cost.evaluations, 2000);
        assert_eq!(cost.ms_per_evaluation, 2.0);
    }

    /// A hierarchical prior whose argument is out of its domain at the start
    /// (sigma = s_h with s_h = -1) has zero density there; the guard names the
    /// parameter. The same prior at s_h = 0.5 passes.
    #[test]
    fn start_with_zero_prior_density_is_refused() {
        use sim::inference::types::{Transform, EstimatedParam};
        let expr = |j: &str| serde_json::from_str::<ir::expr::Expr>(j).unwrap();
        let hp = ir::parameter::HierarchicalPrior {
            kind: ir::parameter::HierarchicalKind::LogNormal,
            args: [("mu".to_string(), expr(r#"{"const": -2.0}"#)),
                   ("sigma".to_string(), expr(r#"{"param": "s_h"}"#))]
                .into_iter().collect(),
            pool_over: String::new(),
        };
        let priors = vec![Prior::from_hierarchical_ir(&hp)];
        let specs = vec![EstimatedParam {
            name: "gamma".into(), index: 0, initial: 0.15, rw_sd: 0.1,
            transform: Transform::LowerBounded { lo: 0.0 },
            rw_sd_auto: false, perturb_only_at_t0: false,
        }];
        let names = vec!["gamma".to_string(), "s_h".to_string()];
        let err = start_prior_is_finite(&specs, &priors, &names, &[0.15, -1.0])
            .expect_err("sigma = -1 gives zero density");
        assert!(err.contains("'gamma' = 0.15"), "{err}");
        assert!(start_prior_is_finite(&specs, &priors, &names, &[0.15, 0.5]).is_ok());
    }
}
