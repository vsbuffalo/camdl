//! Per-stage runner for NLopt deterministic MLE — Phase 1 of the
//! ODE-inference proposal
//! (`docs/dev/proposals/2026-05-04-ode-inference-three-phase.md`).
//!
//! Mirrors the shape of `pgas::run_stage` / `pmmh::run_stage`: parses the
//! NLopt-shaped `Stage` variant, builds a `FitRunConfig`, draws LHS-spread
//! per-chain starts, runs each chain's NLopt optimization in parallel via
//! rayon, aggregates the winner, writes per-chain outputs +
//! `fit_state.toml`, and emits a two-leg convergence diagnostic
//! (chain-agreement + decibans-spread) as a stdout verdict line.
//!
//! The optimizer itself lives in `sim::inference::deterministic`; this
//! module is the orchestration layer wiring it to the fit framework.

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use sim::inference::deterministic::{
    optimize_det, NloptAlgorithm, OptResult, OptStatus,
};

use crate::fit::config_v2::{Algorithm, DtCheckConfig, GateConfig, Method, NloptStageConfig, Problem};
use crate::fit::dt_check;
use crate::fit::loglik::LoglikType;
use crate::fit::methods::check_model_capabilities;
use crate::fit::runner::{ode_step_dt, FitRunConfig};
use sim::inference::compute_ode_loglik;
use crate::fit::state::FitState;
use crate::fit::provenance;

/// Run a single NLopt-flavoured method (`Algorithm::NlSbplx`,
/// `Algorithm::NlBobyqa` or `Algorithm::NlLbfgs`). Errors with a clear message
/// if the algorithm is anything else — caller's job to dispatch correctly.
#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    fit: &Problem,
    method: &Method,
    stage_dir: &Path,
    seed: u64,
    starts: &super::chain_starts::ResolvedStarts,
    parent_fit_hash: &str,
    model_identity: &str,
    data_hashes: &[(String, String)],
    dt_check_cfg: &DtCheckConfig,
    // The stage's liveness/progress heartbeat, owned by the runner
    // (`fit::stage_heartbeat`). Its step is one objective evaluation, against
    // the `max_evals` budget (gh#900).
    heartbeat: &io::HeartbeatGuard,
) -> Result<(), String> {
    let stage_name = method.algorithm.method_name();
    let (algorithm, knobs) = extract_nlopt_config(&method.algorithm)?;

    eprintln!(
        "\x1b[33mℹ {} ({}):\x1b[0m deterministic MLE on the ODE-skeleton \
         likelihood.",
        algorithm.as_str(),
        match algorithm {
            NloptAlgorithm::Sbplx => "Subspace simplex; robust to boundary non-smoothness",
            NloptAlgorithm::Bobyqa => "Quadratic trust region; smooth-objective only",
            NloptAlgorithm::Lbfgs =>
                "Quasi-Newton on the forward-sensitivity gradient; differentiable models only",
        }
    );
    eprintln!(
        "  camdl computes p(y|θ, ODE_skeleton) under {algorithm}, not the \
         stochastic-process p(y|θ) IF2/PGAS/PMMH compute. In low-noise \
         regimes the two converge empirically; verify rather than assume.",
        algorithm = algorithm.as_str()
    );

    let n_chains = knobs.chains;
    if n_chains == 0 {
        return Err(format!(
            "stage '{stage_name}': chains must be ≥ 1; got 0"
        ));
    }

    // FitRunConfig is shaped around IF2 / PF (n_particles, n_iterations,
    // cooling). NLopt doesn't use any of those — pass placeholders that
    // produce a valid config without affecting the run. `dt` is read from
    // `model.simulation.dt` inside `compute_ode_loglik`; if2_config.dt is
    // a fallback only.
    let run_config = FitRunConfig::build(
        fit,
        Some(&method.algorithm),
        n_chains,
        /* n_particles */ 1,
        /* n_iterations */ 1,
        /* cooling */ 1.0,
        /* cooling_target_iters */ 1,
        seed,
        // gh#506 / gh#528: the base point carries `[estimate].start`; the
        // `starts` rule decides per chain what is drawn around it.
        /* random_starts */ false,
    )?;

    // Reject models the ODE backend can't represent (e.g. `overdispersed`
    // transitions whose σ² noise the deterministic skeleton ignores).
    // Without this, NLopt happily fits the wrong likelihood — the same
    // silent-fail `camdl simulate --backend ode` already gates against.
    check_model_capabilities(crate::run_meta::InferenceBackend::Ode, &run_config.compiled)?;

    let bounds: Vec<(f64, f64)> = run_config
        .estimated_params
        .iter()
        .map(|p| (p.lower, p.upper))
        .collect();
    let est_indices: Vec<usize> = run_config
        .estimated_params
        .iter()
        .map(|p| p.index)
        .collect();
    let est_names: Vec<String> = run_config
        .estimated_params
        .iter()
        .map(|p| p.name.clone())
        .collect();

    // Every NLopt algorithm here is deterministic, so under a point rule every
    // chain's outcome is identical: run one chain and skip the wasted compute.
    // A spread rule (`lhs` is the natural one here) gives multi-start basin
    // exploration, and the chain-agreement gate then becomes informative.
    let effective_chains = if starts.rule.is_point() { 1 } else { n_chains };
    if effective_chains < n_chains {
        eprintln!(
            "  starts = {} with chains={}: collapsing to 1 chain \
             ({} is deterministic, redundant chains would produce \
             identical output). Set `starts = \"lhs\"` for multi-start \
             basin exploration.",
            starts.rule.spelled(), n_chains, algorithm.as_str()
        );
    }
    let drawn = crate::fit::runner::draw_chain_starts_for(
        &run_config, &fit.estimate, starts, effective_chains, seed,
    )?;
    let chain_starts: Vec<Vec<f64>> =
        drawn.to_param_vecs(&run_config.estimated_params, &run_config.base_params);
    let n_chains = effective_chains;

    std::fs::create_dir_all(stage_dir).map_err(|e| {
        format!("creating {}: {}", stage_dir.display(), e)
    })?;

    let arc_config = Arc::new(run_config);
    let bounds_ref = bounds.clone();
    let est_indices_ref = est_indices.clone();

    // Build the obs model + obs-time vector ONCE for the whole stage.
    // The per-chain closures borrow these via Arc, so per-eval cost is
    // one ODE solve + one obs scoring pass with no reconstruction.
    let obs_model = Arc::new(arc_config.build_obs_model());
    let obs_times: Vec<f64> = arc_config.observations.iter().map(|o| o.time).collect();
    let dt = ode_step_dt(&arc_config);

    let t0 = std::time::Instant::now();
    let mut chain_outcomes: Vec<(usize, ChainOutcome)> = (0..n_chains)
        .into_par_iter()
        .map(|chain_idx| {
            let outcome = run_one_chain(
                algorithm,
                knobs,
                &arc_config.compiled,
                &obs_model,
                &obs_times,
                dt,
                &bounds_ref,
                &est_indices_ref,
                &chain_starts[chain_idx],
                heartbeat,
            );
            (chain_idx, outcome)
        })
        .collect();
    chain_outcomes.sort_by_key(|(i, _)| *i);
    let elapsed = t0.elapsed();

    // Pick winner. NEG_INFINITY logliks (model blew up) sort below
    // anything finite; finite-loglik chains beat them automatically.
    let (winner_idx, winner) = chain_outcomes
        .iter()
        .max_by(|a, b| {
            a.1.loglik
                .partial_cmp(&b.1.loglik)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, c)| (*i, c.clone()))
        .ok_or_else(|| "no chains ran".to_string())?;

    // Per-chain final params dump for inspection.
    write_per_chain_files(stage_dir, &chain_outcomes, &est_names)?;

    // Emit convergence diagnostic before writing fit_state — the stdout
    // verdict tells the user whether to trust the winner.
    let chain_logliks: Vec<f64> =
        chain_outcomes.iter().map(|(_, c)| c.loglik).collect();
    let convergence = check_convergence(
        &chain_outcomes,
        &est_names,
        &bounds,
        &knobs.gate,
        knobs.tolerance,
    );
    print_verdict(
        &convergence,
        elapsed.as_secs_f64(),
        n_chains,
        knobs.gate.decibans_thresh,
    );

    // Post-fit Richardson dt-convergence check at θ̂ (gh#52, gh#227). The
    // deterministic ODE sibling of the IF2/PF dt-check: re-evaluate the same
    // ODE marginal likelihood at the winner on a dt-halving ladder and warn
    // when θ̂'s loglik is still drifting — i.e. the MLE is discretization-
    // dependent. This is the silent-wrong-answer mode where a coarse dt
    // creates a fake basin that synthetic recovery shares and can't detect.
    // Reuses the obs_model / obs_times / dt already built for the chain evals.
    let mut winner_full = arc_config.base_params.clone();
    for (slot, &idx) in est_indices.iter().enumerate() {
        winner_full[idx] = winner.params[slot];
    }
    let dt_check_result = dt_check::run_richardson_ladder_ode(
        arc_config.compiled.as_ref(),
        obs_model.as_ref(),
        &obs_times,
        &winner_full,
        dt,
        dt_check_cfg,
    )
    .unwrap_or_else(|e| super::abandon_stage(heartbeat, format!("error: {e}")));
    dt_check::print_terminal_report(&dt_check_result);

    // Persist the winner's full parameter vector via fit_state.toml so
    // downstream stages (`refine`, `pgas`, `posterior`) can resume from it.
    let mut start_values = std::collections::BTreeMap::new();
    let mut rw_sd = std::collections::HashMap::new();
    for (slot, name) in est_names.iter().enumerate() {
        start_values.insert(name.clone(), winner.params[slot]);
        // Keep the auto-derived rw_sd from the run config so a downstream
        // IF2/PMMH refine doesn't have to re-derive it from scratch.
        if let Some(p) = arc_config
            .estimated_params
            .iter()
            .find(|p| p.name == *name)
        {
            rw_sd.insert(name.clone(), p.rw_sd);
        }
    }
    let perturb_only_at_t0_params: Vec<String> = arc_config
        .estimated_params
        .iter()
        .filter(|p| p.perturb_only_at_t0)
        .map(|p| p.name.clone())
        .collect();

    let fit_state = FitState {
        method: stage_name.to_string(),
        seed,
        timestamp: crate::cas::iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION.to_string()),
        best_loglik: winner.loglik,
        initial_loglik: f64::NAN,
        // gh#912: `winner_idx` is the 0-based chain index; the stored value is
        // the 1-based chain NUMBER, matching the `chain` column of
        // `chain_results.tsv` beside this file.
        best_chain: winner_idx + 1,
        n_chains,
        n_good_chains: Some(
            chain_outcomes
                .iter()
                .filter(|(_, c)| matches!(c.status, OptStatus::Converged(_)))
                .count(),
        ),
        start_values: start_values.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        rw_sd: rw_sd.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        loglik_type: Some(LoglikType::OdeMarginal),
        acceptance_rate: None,
        tail_chain_agreement: convergence.chain_agreement.iter()
            .map(|(k, v)| (k.clone(), *v)).collect(),
        perturb_only_at_t0_params,
        chain_logliks,
        chain_eval_logliks: Vec::new(),
        chain_eval_ids: Vec::new(),
        chain_eval_ses: Vec::new(),
        resolved_gate: Some(knobs.gate.clone()),
        resolved_loglik_eval: None,
        chain_init_source: Some(drawn.rule.spelled()),
        chain_starts_kind: Some(drawn.rule.kind()),
        // gh#52, gh#227: deterministic ODE dt-check at θ̂ (above). Skipped →
        // omit the block, mirroring the IF2 path's legacy semantics.
        dt_check: if matches!(dt_check_result.verdict, dt_check::DtCheckVerdict::Skipped) {
            None
        } else {
            Some(dt_check_result)
        },
        // gh#764: a deterministic optimiser, not a pseudo-marginal sampler.
        pf_noise: None,
    };
    fit_state
        .save(&stage_dir.to_string_lossy())
        .map_err(|e| format!("writing fit_state.toml: {e}"))?;

    // Write mle_params.toml — full parameter vector at the winner with
    // a [provenance] block. `camdl fit summary` walks this file to render
    // per-cell MLE rows.
    let mut all_params = std::collections::BTreeMap::new();
    for (i, name) in arc_config.param_names.iter().enumerate() {
        all_params.insert(name.clone(), arc_config.base_params[i]);
    }
    // Overlay the optimized estimated-param values on top of base_params.
    for (slot, name) in est_names.iter().enumerate() {
        all_params.insert(name.clone(), winner.params[slot]);
    }
    let mle_path = stage_dir.join("mle_params.toml");
    let mle_meta = provenance::MleMetadata {
        input_hash: parent_fit_hash.to_string(),
        model_path: fit.model.camdl.clone(),
        model_identity: model_identity.to_string(),
        data_hashes: data_hashes.to_vec(),
        seed,
        method: stage_name.to_string(),
        best_chain_index: winner_idx,
        // NLopt stages always run on the ODE backend (validated by
        // methods::validate_combo); record it from the inference domain
        // (total `From<InferenceBackend>` into the forward provenance field).
        backend: crate::run_meta::InferenceBackend::Ode.into(),
        dt: ode_step_dt(&arc_config),
        loglik: winner.loglik,
        loglik_sd: 0.0,
        n_particles: 0,
        ess_at_mle: None,
        timestamp: fit_state.timestamp.clone(),
    };
    if let Err(e) = provenance::write_mle_params(
        &mle_path.to_string_lossy(), &all_params, &mle_meta,
    ) {
        eprintln!("warning: writing mle_params.toml failed: {e}");
    }

    Ok(())
}

fn extract_nlopt_config(
    algorithm: &Algorithm,
) -> Result<(NloptAlgorithm, &NloptStageConfig), String> {
    match algorithm {
        Algorithm::NlSbplx(c) => Ok((NloptAlgorithm::Sbplx, c)),
        Algorithm::NlBobyqa(c) => Ok((NloptAlgorithm::Bobyqa, c)),
        Algorithm::NlLbfgs(c) => Ok((NloptAlgorithm::Lbfgs, c)),
        other => Err(format!(
            "nlopt_stage::run_stage: expected nl-sbplx, nl-bobyqa or nl-lbfgs, got {}",
            other.method_name()
        )),
    }
}

#[derive(Clone)]
struct ChainOutcome {
    /// Optimized parameter vector restricted to estimated slots
    /// (in `est_names` order).
    params: Vec<f64>,
    loglik: f64,
    status: OptStatus,
    n_evals: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_one_chain(
    algorithm: NloptAlgorithm,
    knobs: &NloptStageConfig,
    compiled: &Arc<sim::CompiledModel>,
    obs_model: &Arc<sim::inference::MultiStreamObsModel>,
    obs_times: &[f64],
    dt: f64,
    bounds: &[(f64, f64)],
    est_indices: &[usize],
    full_start: &[f64],
    heartbeat: &io::HeartbeatGuard,
) -> ChainOutcome {
    // gh#900: one objective evaluation is one step of the run's progress. It is
    // the only thing that happens often enough to report on a search whose
    // single chain may run for minutes — a per-chain counter would sit at zero
    // for the whole of the usual one-chain Sbplx fit. A relaxed `fetch_max` on
    // a shared atomic, so "furthest any chain has reached" is what the artifact
    // shows; no I/O per evaluation and no RNG consumed.
    let on_eval = |evals: u64| heartbeat.bump(evals);
    let result = optimize_cell(
        algorithm,
        compiled,
        obs_model,
        obs_times,
        dt,
        bounds,
        est_indices,
        full_start,
        knobs.tolerance,
        knobs.max_evals,
        Some(&on_eval),
    );
    match result {
        Ok(r) => ChainOutcome {
            params: r.params,
            loglik: r.loglik,
            status: r.status,
            n_evals: r.n_evals,
        },
        Err(e) => {
            // Either the cell was mis-configured (dimension / bounds) or the
            // gradient could not be taken somewhere along the search. Both are
            // structural: this chain has no optimum to report, so it carries
            // `Failed` and a `-inf` loglik, which loses the winner comparison
            // to any chain that did converge.
            eprintln!("\x1b[31m✗\x1b[0m {}: {e}", algorithm.as_str());
            ChainOutcome {
                params: est_indices.iter().map(|&i| full_start[i]).collect(),
                loglik: f64::NEG_INFINITY,
                status: OptStatus::Failed,
                n_evals: 0,
            }
        }
    }
}

/// One NLopt cell-level optimization — shared by `nlopt_stage::run_stage`
/// (multi-chain MLE for fit.toml stages) and `cli::profile` (per-cell MLE
/// with focal params pinned). The cell is defined by `full_start`: every
/// non-estimated index (focal pins, fixed params, model defaults) is read
/// out of it verbatim; only the indices in `est_indices` are optimized.
///
/// `bounds` are the natural-scale (lower, upper) box for the slots in
/// `est_indices`, in the same order. The closure builds a fresh full
/// parameter vector per evaluation and scores it:
///
/// - **derivative-free** (`nl-sbplx`, `nl-bobyqa`) — NLopt passes no gradient
///   slot, so one `compute_ode_loglik` call gives the loglik (or
///   `f64::NEG_INFINITY` if the model blew up at this θ).
/// - **gradient** (`nl-lbfgs`) — NLopt passes a slot, so the same point is
///   scored by `det_grad`, which returns the loglik AND `∇_θ` from one
///   augmented forward-sensitivity solve. The gradient is written into the
///   slot; the value returned is the same ODE marginal likelihood the
///   derivative-free path computes, off the same trajectory.
///
/// Both pass `burnin_dt = dt`, i.e. no coarse warm-up: the NLopt stage uses
/// the fine step throughout, and value and gradient must be taken on the same
/// trajectory or the reported optimum is not the optimum of the reported
/// likelihood.
///
/// `on_eval` is called with this cell's running evaluation count each time the
/// objective is scored — the seam a caller uses to report search progress
/// (gh#900). It is the only per-evaluation hook there is: the objective closure
/// is built here, so a caller cannot wrap it from outside. `camdl profile`
/// passes `None`; its progress is a grid cell, not an evaluation. One
/// `det_grad` call counts as one evaluation, the same unit as one
/// `compute_ode_loglik` call.
///
/// **When the gradient cannot be taken** — `det_grad` errors, or hands back a
/// value or a gradient component that is not finite — this returns `Err` with
/// the reason, and never an `OptResult`. There is nothing to report at such a
/// point: the derivative-free answer (return `NEG_INFINITY` and let the search
/// steer away) has no gradient analogue, because L-BFGS has no direction to
/// steer in. The first reason is latched, every later evaluation short-circuits
/// to it without paying for another solve, and the latched reason is what comes
/// back — so the chain fails with `det_grad`'s own sentence rather than a bare
/// "failed", and no optimum is reported from a point whose gradient was not
/// computable. The latch is what makes this true; NLopt's own `NLOPT_FAILURE`
/// on a non-finite gradient only makes it prompt.
///
/// This function is the single source of truth for "deterministic-MLE
/// optimization on the ODE skeleton given a focal pin." Both consumers
/// pass an `Arc<MultiStreamObsModel>` and obs_times built once outside
/// the loop, so per-eval cost is one ODE solve + one obs scoring pass.
#[allow(clippy::too_many_arguments)]
pub fn optimize_cell(
    algorithm: NloptAlgorithm,
    compiled: &Arc<sim::CompiledModel>,
    obs_model: &Arc<sim::inference::MultiStreamObsModel>,
    obs_times: &[f64],
    dt: f64,
    bounds: &[(f64, f64)],
    est_indices: &[usize],
    full_start: &[f64],
    tolerance: f64,
    max_evals: usize,
    on_eval: Option<&(dyn Fn(u64) + Sync)>,
) -> Result<OptResult, String> {
    let initial_est: Vec<f64> = est_indices.iter().map(|&i| full_start[i]).collect();

    // The first reason the gradient could not be taken, if any. Read after the
    // optimization finishes; the closure borrows it, and the borrow ends when
    // `optimize_det` returns (it consumes the closure).
    let grad_failure: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    let grad_failure_ref = &grad_failure;

    // Closure-owned mutable state. NLopt's callback signature requires
    // `Fn`; the user_data smuggle inside `optimize_det` lets `objective`
    // be `FnMut`, so we mutate `full_params` in place per call (avoids
    // a Vec alloc per eval).
    let mut full_params = full_start.to_vec();
    let est_indices_local = est_indices.to_vec();
    let compiled_local = Arc::clone(compiled);
    let obs_model_local = Arc::clone(obs_model);
    let obs_times_local = obs_times.to_vec();
    let mut evals: u64 = 0;
    let objective = move |est: &[f64], grad: Option<&mut [f64]>| -> f64 {
        for (slot, &model_idx) in est_indices_local.iter().enumerate() {
            full_params[model_idx] = est[slot];
        }
        evals += 1;
        if let Some(report) = on_eval {
            report(evals);
        }
        let Some(slot) = grad else {
            // Derivative-free: value only.
            return compute_ode_loglik(
                &compiled_local,
                &obs_model_local,
                &obs_times_local,
                dt,
                &full_params,
                dt, // burnin_dt = dt ⇒ coarse burn-in off (fine step throughout)
            )
            .unwrap_or(f64::NEG_INFINITY);
        };
        // Already given up: wind down without another augmented solve.
        if grad_failure_ref.borrow().is_some() {
            slot.fill(f64::NAN);
            return f64::NEG_INFINITY;
        }
        match score_with_gradient(
            &compiled_local,
            &obs_model_local,
            &obs_times_local,
            dt,
            &full_params,
            &est_indices_local,
            slot,
        ) {
            Ok(value) => value,
            Err(reason) => {
                let mut latched = grad_failure_ref.borrow_mut();
                if latched.is_none() {
                    *latched = Some(reason);
                }
                slot.fill(f64::NAN);
                f64::NEG_INFINITY
            }
        }
    };

    let result = optimize_det(
        algorithm,
        &initial_est,
        bounds,
        tolerance,
        max_evals,
        objective,
    );
    if let Some(reason) = grad_failure.into_inner() {
        return Err(reason);
    }
    result
}

/// One gradient-path objective evaluation: score `full_params` on the ODE
/// marginal likelihood with `det_grad` and write `∇_θ` into NLopt's `slot`.
///
/// `slot` is filled only on success, and only with a gradient every component
/// of which is finite beside a finite value — the two come from one augmented
/// forward-sensitivity solve, so a `slot` written here is the derivative of
/// the number returned here. `burnin_dt = dt` (no coarse warm-up), matching
/// the value path in [`optimize_cell`].
///
/// The `Err` is the sentence the chain fails with, so it names the θ it
/// happened at and what to do instead. Two shapes reach it: `det_grad`
/// refusing or erroring, and a solve that ran but produced a non-finite value
/// or gradient. Both mean the same thing to an L-BFGS line search — there is
/// no direction here — which is why neither is softened into a score the
/// search could "steer away" from, the way the derivative-free path softens a
/// blown-up θ into `NEG_INFINITY`.
fn score_with_gradient(
    compiled: &sim::CompiledModel,
    obs_model: &sim::inference::MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    full_params: &[f64],
    est_indices: &[usize],
    slot: &mut [f64],
) -> Result<f64, String> {
    let at = || describe_point(est_indices, compiled, full_params);
    match sim::inference::ode_grad::det_grad(
        compiled, obs_model, obs_times, dt, dt, full_params, est_indices,
    ) {
        Ok((value, gradient)) => {
            if value.is_finite() && gradient.iter().all(|g| g.is_finite()) {
                slot.copy_from_slice(&gradient);
                Ok(value)
            } else {
                Err(format!(
                    "the ODE gradient is not finite at {}: log-likelihood {value}, \
                     gradient {gradient:?}. The deterministic likelihood has no \
                     usable derivative there, so `nl-lbfgs` has no direction to \
                     search in. Narrow the `[estimate]` bounds to exclude the \
                     region, start elsewhere, or use the derivative-free \
                     `algorithm = \"nl-sbplx\"`.",
                    at()
                ))
            }
        }
        Err(e) => Err(format!(
            "the ODE gradient could not be taken at {}: {e}\n  \
             `algorithm = \"nl-sbplx\"` optimizes the same ODE marginal \
             likelihood without a gradient.",
            at()
        )),
    }
}

/// `name = value` for each estimated coordinate, so a gradient failure names
/// the θ it happened at rather than a bare index. Off the hot path — built only
/// when something has already gone wrong.
fn describe_point(
    est_indices: &[usize],
    compiled: &sim::CompiledModel,
    full_params: &[f64],
) -> String {
    let parts: Vec<String> = est_indices
        .iter()
        .map(|&idx| {
            let name = compiled
                .model
                .parameters
                .get(idx)
                .map(|p| p.name.as_str())
                .unwrap_or("?");
            format!("{name} = {}", full_params[idx])
        })
        .collect();
    format!("({})", parts.join(", "))
}

fn write_per_chain_files(
    stage_dir: &Path,
    chain_outcomes: &[(usize, ChainOutcome)],
    est_names: &[String],
) -> Result<(), String> {
    use std::io::Write;
    let path = stage_dir.join("chain_results.tsv");
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("creating {}: {}", path.display(), e))?;
    write!(f, "chain\tloglik\tstatus\tn_evals").map_err(io_err)?;
    for name in est_names {
        write!(f, "\t{name}").map_err(io_err)?;
    }
    writeln!(f).map_err(io_err)?;
    for (chain_idx, c) in chain_outcomes {
        write!(
            f,
            "{}\t{:.6}\t{}\t{}",
            chain_idx + 1,
            c.loglik,
            c.status.as_str(),
            c.n_evals,
        )
        .map_err(io_err)?;
        for v in &c.params {
            write!(f, "\t{v:.10}").map_err(io_err)?;
        }
        writeln!(f).map_err(io_err)?;
    }
    Ok(())
}

fn io_err(e: std::io::Error) -> String {
    format!("io error: {e}")
}

/// Two-leg convergence diagnostic for NLopt stages — generalises IF2's
/// compound gate (chain-agreement + decibans-spread) to deterministic
/// optimizers. See proposal §"Convergence diagnostics for NLopt chains".
struct ConvergenceVerdict {
    /// Per-parameter relative range across converged chains.
    /// `(name, rel_range, abs_range, bound_width)`. Used by the
    /// chain-agreement leg of the gate.
    chain_agreement: std::collections::HashMap<String, f64>,
    /// `max(rel_range) / bound_width` over params — single scalar
    /// summary for the verdict line.
    max_rel_range: f64,
    /// Maximum absolute range over params, in natural units. Used to
    /// distinguish "tight cluster, large bound" from "tight bound, big
    /// optimizer noise".
    max_abs_range: f64,
    /// `max(loglik) - min(loglik)` across converged chains, in nats.
    /// Decibans = nats × NATS_TO_DB; the threshold compare uses
    /// `delta_nats * NATS_TO_DB` against `gate.decibans_thresh`.
    delta_loglik: f64,
    /// `true` iff the configured thresholds were both exceeded.
    chain_agreement_failed: bool,
    decibans_failed: bool,
    /// Number of converged (Success / X/F-tol) chains.
    n_converged: usize,
    /// Number of soft-failed (MaxEvalReached) chains.
    n_maxeval: usize,
    /// Number of hard-failed (Failed) chains.
    n_failed: usize,
}

const NATS_TO_DB: f64 = 4.342944819032518;
/// Per the proposal, threshold the chain-agreement leg fires only when
/// BOTH relative range > 5% bound AND absolute range > 2 × `xtol_rel`-
/// implied numerical floor are violated. The 0.05 placeholder is
/// calibrated against the typhoid diagnostic experiment downstream.
const DET_REL_RANGE_THRESH: f64 = 0.05;
const DET_ABS_RANGE_FACTOR: f64 = 2.0;

fn check_convergence(
    chain_outcomes: &[(usize, ChainOutcome)],
    est_names: &[String],
    bounds: &[(f64, f64)],
    gate: &GateConfig,
    tolerance: f64,
) -> ConvergenceVerdict {
    use std::collections::HashMap;

    let n_converged = chain_outcomes
        .iter()
        .filter(|(_, c)| matches!(c.status, OptStatus::Converged(_)))
        .count();
    let n_maxeval = chain_outcomes
        .iter()
        .filter(|(_, c)| matches!(c.status, OptStatus::MaxEvalReached))
        .count();
    let n_failed = chain_outcomes
        .iter()
        .filter(|(_, c)| matches!(c.status, OptStatus::Failed | OptStatus::MaxTimeReached))
        .count();

    let mut chain_agreement = HashMap::new();
    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut chain_agreement_failed = false;
    if chain_outcomes.len() >= 2 {
        for (slot, name) in est_names.iter().enumerate() {
            let vals: Vec<f64> =
                chain_outcomes.iter().map(|(_, c)| c.params[slot]).collect();
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let abs_range = max - min;
            let bound_width = (bounds[slot].1 - bounds[slot].0).abs().max(1e-300);
            let rel_range = abs_range / bound_width;
            chain_agreement.insert(name.clone(), rel_range);
            if rel_range > max_rel {
                max_rel = rel_range;
            }
            if abs_range > max_abs {
                max_abs = abs_range;
            }
            // Per proposal: refuse only if BOTH legs exceed thresholds —
            // a tight cluster on a wide bound is fine.
            let abs_floor = DET_ABS_RANGE_FACTOR
                * tolerance
                * (vals[0].abs().max(1.0));
            if rel_range > DET_REL_RANGE_THRESH && abs_range > abs_floor {
                chain_agreement_failed = true;
            }
        }
    }

    let logliks: Vec<f64> = chain_outcomes
        .iter()
        .filter(|(_, c)| matches!(c.status, OptStatus::Converged(_)))
        .map(|(_, c)| c.loglik)
        .collect();
    let delta_loglik = if logliks.len() >= 2 {
        let lmax = logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lmin = logliks.iter().cloned().fold(f64::INFINITY, f64::min);
        lmax - lmin
    } else {
        0.0
    };
    let decibans_failed = delta_loglik * NATS_TO_DB > gate.decibans_thresh;

    ConvergenceVerdict {
        chain_agreement,
        max_rel_range: max_rel,
        max_abs_range: max_abs,
        delta_loglik,
        chain_agreement_failed,
        decibans_failed,
        n_converged,
        n_maxeval,
        n_failed,
    }
}

fn print_verdict(
    v: &ConvergenceVerdict,
    wall_secs: f64,
    n_chains: usize,
    decibans_thresh: f64,
) {
    let ok = "\x1b[32m✓\x1b[0m";
    let bad = "\x1b[31m✗\x1b[0m";
    eprintln!();
    eprintln!(
        "  status: {} converged, {} max-eval, {} failed (of {})",
        v.n_converged, v.n_maxeval, v.n_failed, n_chains
    );
    eprintln!(
        "  chain-agreement: rel range = {:.2}% bound | abs range = {:.3e}   {}",
        v.max_rel_range * 100.0,
        v.max_abs_range,
        if v.chain_agreement_failed { bad } else { ok }
    );
    eprintln!(
        "  loglik-eval:     Δ = {:.1} dB / threshold {:.0} dB                {}",
        v.delta_loglik * NATS_TO_DB,
        decibans_thresh,
        if v.decibans_failed { bad } else { ok }
    );
    eprintln!("  wall: {:.2}s ({} chains)", wall_secs, n_chains);
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{Algorithm, GateConfig, NloptStageConfig};
    use crate::run_meta::InferenceBackend;
    use sim::inference::deterministic::SuccessState;

    // ── The ODE-gradient fixture ─────────────────────────────────────
    //
    // The `seir_observations` golden, in the configuration the ODE gradient's
    // own finite-difference oracle uses
    // (`sim::inference::ode_grad::tests::det_grad_matches_finite_difference_*`):
    // emitted `rate_state_grad`, a native `neg_binomial(mean = rho·incidence)`
    // incidence stream, a `prevalence(I)` stream, and an explicit (constant)
    // initial condition so `∂init/∂θ = 0`. That oracle establishes the
    // gradient is RIGHT; the tests here establish it is the gradient that
    // reaches NLopt.

    /// The golden model with the explicit initial condition the oracle uses,
    /// and its scenario values resolved.
    fn seir_gradient_model() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read golden: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse golden: {e}"));
        assert!(
            model.transitions.iter().any(|t| !t.rate_state_grad.0.is_empty()),
            "seir_observations must carry emitted rate_state_grad \
             (run `make update-golden`)"
        );
        // Explicit IC: the ∂init/∂θ seed is zero, which is the configuration
        // the oracle checks. The golden's own parameterized IC would drag
        // `N0`/`I0` into the seed and is a separate case.
        model.initial_conditions = ir::model::InitialConditions::constants([
            ("S".to_string(), 99990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]);
        model.ic_grad = std::collections::HashMap::new();
        model.simulation.t_end = 56.0;
        let defaults = [
            ("beta", 0.3), ("sigma", 0.2), ("gamma", 0.1), ("k", 5.0),
            ("rho", 0.5), ("p_detect", 0.8), ("N0", 100000.0), ("I0", 10.0),
        ];
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                let v = defaults.iter().find(|(n, _)| *n == p.name)
                    .map(|(_, v)| *v).unwrap_or(0.5);
                p.value = p.value.with_value(v);
            }
        }
        model
    }

    /// `(compiled, obs_model, obs_times, params)` — the fixture bound to eight
    /// weekly observations.
    ///
    /// The weekly case counts are a growing epidemic curve of the magnitude
    /// this SEIR produces at the scenario values, so the point is near enough
    /// to the mode for the likelihood and its gradient to be materially
    /// nonzero (asserted at each use). `detection` is the golden's own
    /// `bernoulli(p = p_detect)`, whose likelihood does not read `projected`
    /// at all, so it contributes a constant — a second stream on the axis
    /// without a second gradient chain.
    fn seir_gradient_fixture() -> (
        Arc<sim::CompiledModel>,
        Arc<sim::inference::MultiStreamObsModel>,
        Vec<f64>,
        Vec<f64>,
    ) {
        use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
        use sim::inference::{dense_cells, BoundObs, MultiStreamObsModel};

        let compiled = Arc::new(sim::CompiledModel::new(seir_gradient_model()).unwrap());
        let mut params = vec![0.0; compiled.param_index.len()];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        let per_stream: Vec<Vec<f64>> = vec![
            vec![3.0, 6.0, 13.0, 27.0, 55.0, 110.0, 215.0, 410.0],
            vec![1.0; 8],
        ];

        let projections: Vec<StreamProjection> = compiled
            .model
            .observations
            .iter()
            .map(|om| StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap())
            .collect();
        let t_start = compiled.model.simulation.t_start;
        let specs: Vec<StreamSpec> = compiled
            .model
            .observations
            .iter()
            .enumerate()
            .map(|(si, om)| StreamSpec {
                times: StreamTimes::contiguous_for(
                    &projections[si], t_start, obs_times.clone(),
                )
                .unwrap(),
                projection: projections[si].clone(),
                ir_model: om.clone(),
                observations: dense_cells(per_stream[si].clone()),
                aux: vec![],
            })
            .collect();
        let obs_model = Arc::new(
            MultiStreamObsModel::new(
                BoundObs::bind(t_start, specs).unwrap().0,
                compiled.clone(),
            )
            .unwrap(),
        );
        (compiled, obs_model, obs_times, params)
    }

    /// The gradient NLopt is handed is EXACTLY `det_grad`'s gradient at that
    /// point — bit for bit, not merely close.
    ///
    /// `score_with_gradient` is the function the gradient objective calls once
    /// per evaluation, so what it writes into the slot is what NLopt reads.
    /// Anything between `det_grad` and the slot — a re-scaled gradient, a
    /// gradient taken with a different `burnin_dt`, an `est_indices` order that
    /// drifted from the one the value used — would leave the run converging on
    /// a direction that is not the likelihood's, and every diagnostic
    /// downstream would still look healthy.
    #[test]
    fn the_gradient_nlopt_receives_is_det_grads_gradient() {
        let (compiled, obs_model, obs_times, params) = seir_gradient_fixture();
        let dt = 1.0;
        let est_indices: Vec<usize> = ["beta", "gamma", "k", "rho"]
            .iter()
            .map(|n| compiled.param_index[*n])
            .collect();

        let mut slot = vec![f64::NAN; est_indices.len()];
        let value = score_with_gradient(
            &compiled, &obs_model, &obs_times, dt, &params, &est_indices, &mut slot,
        )
        .expect("the fixture is differentiable");

        let (want_value, want_grad) = sim::inference::ode_grad::det_grad(
            &compiled, &obs_model, &obs_times, dt, dt, &params, &est_indices,
        )
        .expect("det_grad at the same point");

        assert_eq!(value, want_value, "the objective's value must be det_grad's value");
        assert_eq!(
            slot, want_grad,
            "the slot NLopt reads must be det_grad's gradient exactly"
        );
        // Non-vacuity: a fixture whose gradient were all zeros would satisfy
        // the equality above while proving nothing.
        assert!(
            slot.iter().any(|g| g.abs() > 1e-6),
            "the fixture's gradient must be materially nonzero; got {slot:?}"
        );
    }

    /// The whole gradient path, from `optimize_cell` down: L-BFGS improves the
    /// log-likelihood from a displaced start on the same fixture, and reports
    /// having converged. The `est_indices` mapping is exercised here — the
    /// optimizer works in estimated-slot space and the model in
    /// model-parameter space.
    #[test]
    fn optimize_cell_with_lbfgs_improves_the_loglik_from_a_displaced_start() {
        let (compiled, obs_model, obs_times, params) = seir_gradient_fixture();
        let dt = 1.0;
        let est_indices: Vec<usize> = ["beta", "rho"]
            .iter()
            .map(|n| compiled.param_index[*n])
            .collect();
        let bounds = vec![(0.05, 0.5), (0.05, 0.95)];

        let mut start = params.clone();
        start[est_indices[0]] = 0.22; // true 0.3
        start[est_indices[1]] = 0.35; // true 0.5
        let at_start = sim::inference::compute_ode_loglik(
            &compiled, &obs_model, &obs_times, dt, &start, dt,
        )
        .expect("the start is scorable");

        let r = optimize_cell(
            NloptAlgorithm::Lbfgs,
            &compiled, &obs_model, &obs_times, dt,
            &bounds, &est_indices, &start,
            1e-8, 400, None,
        )
        .expect("the fixture is differentiable, so the cell must optimize");
        assert!(
            r.loglik > at_start,
            "L-BFGS must improve on the start: {} vs {at_start}",
            r.loglik
        );
        assert!(r.status.is_converged(), "status: {:?}", r.status);
        for (slot, (lo, hi)) in r.params.iter().zip(&bounds) {
            assert!(
                (*lo..=*hi).contains(slot),
                "the box bounds must be honoured on the gradient algorithm: \
                 {slot} outside [{lo}, {hi}]"
            );
        }
    }

    /// A model the ODE gradient refuses fails the CELL, with the preflight's
    /// own reason, rather than returning an `OptResult` from a search that had
    /// no gradient. Here the refusal is the adaptive integrator (`rk45`), one
    /// of `preflight_gradient_ode`'s cases.
    ///
    /// The derivative-free sibling on the same model is the control: it
    /// optimizes fine, which is what makes the refusal specific to the
    /// gradient path rather than a broken fixture.
    #[test]
    fn a_model_the_gradient_refuses_fails_the_cell_with_the_reason() {
        let (compiled, obs_model, obs_times, params) = seir_gradient_fixture();
        let mut model = (*compiled.model).clone();
        model.simulation.integrator =
            ir::model::Integrator::Rk45 { atol: None, rtol: None };
        let adaptive = Arc::new(sim::CompiledModel::new(model).unwrap());
        let dt = 1.0;
        let est_indices: Vec<usize> = vec![adaptive.param_index["beta"]];
        let bounds = vec![(0.05, 0.5)];

        let err = optimize_cell(
            NloptAlgorithm::Lbfgs,
            &adaptive, &obs_model, &obs_times, dt,
            &bounds, &est_indices, &params,
            1e-6, 50, None,
        )
        .expect_err("an rk45 model has no forward sensitivity to differentiate");
        assert!(
            err.contains("rk4"),
            "the preflight's own reason must survive to the caller; got: {err}"
        );
        assert!(
            err.contains("nl-sbplx"),
            "the message must name the derivative-free alternative; got: {err}"
        );
        assert!(
            err.contains("beta = "),
            "the message must name the point it failed at; got: {err}"
        );

        // Control: the same model optimizes derivative-free.
        assert!(
            optimize_cell(
                NloptAlgorithm::Sbplx,
                &adaptive, &obs_model, &obs_times, dt,
                &bounds, &est_indices, &params,
                1e-6, 50, None,
            )
            .is_ok(),
            "nl-sbplx has no gradient requirement, so the same model must run"
        );
    }

    fn nlopt_config() -> NloptStageConfig {
        NloptStageConfig {
            backend: InferenceBackend::Ode,
            chains: 4,
            tolerance: 1e-6,
            max_evals: 5000,
            gate: GateConfig::default(),
            dt_check: crate::fit::config_v2::DtCheckConfig::default(),
        }
    }

    fn outcome(loglik: f64, params: Vec<f64>, status: OptStatus) -> ChainOutcome {
        ChainOutcome { loglik, params, status, n_evals: 1 }
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("p{i}")).collect()
    }

    // ── extract_nlopt_config ─────────────────────────────────────────

    #[test]
    fn extract_nlopt_config_sbplx_returns_sbplx_algorithm() {
        let stage = Algorithm::NlSbplx(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&stage).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Sbplx);
        assert_eq!(cfg.chains, 4);
        assert_eq!(cfg.tolerance, 1e-6);
    }

    #[test]
    fn extract_nlopt_config_bobyqa_returns_bobyqa_algorithm() {
        let stage = Algorithm::NlBobyqa(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&stage).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Bobyqa);
        assert_eq!(cfg.chains, 4);
    }

    #[test]
    fn extract_nlopt_config_lbfgs_returns_lbfgs_algorithm() {
        let stage = Algorithm::NlLbfgs(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&stage).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Lbfgs);
        assert!(algo.uses_gradient(), "nl-lbfgs is the gradient algorithm");
        assert_eq!(cfg.chains, 4);
    }

    #[test]
    fn extract_nlopt_config_rejects_non_nlopt_stage() {
        // PFilter is the simplest non-nlopt variant to construct.
        let stage = Algorithm::PFilter {
            backend: InferenceBackend::ChainBinomial,
            particles: 100,
            replicates: None,
            record_ancestry: false,
            record_prequential: false,
        };
        let err = extract_nlopt_config(&stage).expect_err("must reject");
        assert!(err.contains("expected nl-sbplx, nl-bobyqa or nl-lbfgs"),
            "error must name the expected variants; got: {err}");
        assert!(err.contains("pfilter"),
            "error must name the actual variant; got: {err}");
    }

    // ── check_convergence two-number gate ────────────────────────────

    #[test]
    fn convergence_passes_when_chains_agree_tightly_and_loglik_clusters() {
        // Two chains converge to nearly-identical params with nearly-
        // identical loglik. Both legs of the gate should pass.
        let chains = vec![
            (0, outcome(-100.0, vec![0.5, 1.0], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-100.1, vec![0.501, 1.001], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let v = check_convergence(&chains, &names(2), &bounds, &GateConfig::default(), 1e-6);
        assert!(!v.chain_agreement_failed,
            "tight agreement on each param must pass the chain-agreement leg; \
             max_rel_range={}, max_abs_range={}", v.max_rel_range, v.max_abs_range);
        assert!(!v.decibans_failed,
            "0.1-nat loglik delta is well under default decibans_thresh");
        assert_eq!(v.n_converged, 2);
        assert_eq!(v.n_maxeval, 0);
        assert_eq!(v.n_failed, 0);
    }

    #[test]
    fn convergence_fails_when_chains_spread_widely_on_wide_bound() {
        // Chains land at opposite ends of a wide bound — both rel- and
        // abs-range exceed thresholds. Two-number gate fails.
        let chains = vec![
            (0, outcome(-100.0, vec![0.05], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-100.5, vec![0.95], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let bounds = vec![(0.0, 1.0)];
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), 1e-6);
        assert!(v.chain_agreement_failed,
            "rel_range = 90% of bound width must fail chain-agreement; \
             max_rel_range = {}", v.max_rel_range);
    }

    #[test]
    fn convergence_passes_when_rel_range_high_but_abs_range_under_optimizer_floor() {
        // Per the proposal's two-number rule: rel_range > 5% bound BUT
        // abs_range < 2 * tolerance * |val| means the spread is within
        // optimizer numerical noise. Should NOT fail the gate (otherwise
        // tight bounds with wide-tolerance optimizers always look bad).
        let chains = vec![
            // 1e-5 spread on a 2e-4 bound is 5% rel, but abs 1e-5 is
            // below 2 * 1e-6 * |1.0| = 2e-6 floor (val = 1.0). Expected:
            // both legs would fail individually but gate passes (both
            // must fail to refuse).
            (0, outcome(-100.0, vec![1.0],     OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-100.0, vec![1.0001],  OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let bounds = vec![(0.99995, 1.00015)];  // 2e-4 wide; spread is 50% of that
        let tolerance = 0.1; // wide tolerance — abs floor = 0.2 * 1.0 = 0.2 nat units
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), tolerance);
        assert!(!v.chain_agreement_failed,
            "rel_range high but abs_range under optimizer numerical floor \
             must pass — proposal's two-number rule (both legs must fail). \
             max_rel_range={}, max_abs_range={}",
            v.max_rel_range, v.max_abs_range);
    }

    #[test]
    fn convergence_fails_decibans_leg_on_large_loglik_spread() {
        // Default decibans_thresh ≈ 30 nats * NATS_TO_DB. Two chains
        // 200 nats apart fail the loglik-spread leg regardless of
        // parameter agreement.
        let chains = vec![
            (0, outcome(  -50.0, vec![0.5], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-2050.0, vec![0.5], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let bounds = vec![(0.0, 1.0)];
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), 1e-6);
        assert!(v.decibans_failed,
            "2000-nat loglik spread must fail the decibans leg; \
             delta_loglik = {}", v.delta_loglik);
    }

    #[test]
    fn convergence_counts_status_categories_correctly() {
        // Mixed outcomes: 2 converged, 1 hit max_evals, 1 hard failed.
        let chains = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-100.1, vec![0.5], OptStatus::Converged(SuccessState::FtolReached))),
            (2, outcome(-110.0, vec![0.4], OptStatus::MaxEvalReached)),
            (3, outcome(f64::NEG_INFINITY, vec![0.5], OptStatus::Failed)),
        ];
        let bounds = vec![(0.0, 1.0)];
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), 1e-6);
        assert_eq!(v.n_converged, 2);
        assert_eq!(v.n_maxeval,   1);
        assert_eq!(v.n_failed,    1);
        // delta_loglik computed only over converged chains (excludes
        // MaxEvalReached and Failed) — so 0.1 nat, well under threshold.
        assert!(v.delta_loglik < 1.0,
            "delta_loglik must use converged chains only; got {}", v.delta_loglik);
    }

    #[test]
    fn convergence_skips_chain_agreement_when_only_one_chain() {
        // Single-chain runs can't compute between-chain spread; the
        // gate should not fire and the agreement map should be empty.
        let chains = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let bounds = vec![(0.0, 1.0)];
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), 1e-6);
        assert!(!v.chain_agreement_failed);
        assert!(v.chain_agreement.is_empty(),
            "single-chain runs should not populate per-param agreement; \
             got {} entries", v.chain_agreement.len());
        assert_eq!(v.delta_loglik, 0.0);
    }
}
