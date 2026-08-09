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

use crate::fit::config_v2::{DtCheckConfig, FitConfigV2, GateConfig, NloptStageConfig, Stage};
use crate::fit::dt_check;
use crate::fit::init::{build_chain_param_vecs, InitMethod};
use crate::fit::loglik::LoglikType;
use crate::fit::methods::check_model_capabilities;
use crate::fit::runner::{ode_step_dt, FitRunConfig};
use sim::inference::compute_ode_loglik;
use crate::fit::state::FitState;
use crate::fit::provenance;

/// Run a single NLopt-flavoured `Stage` (`Stage::NlSbplx` or
/// `Stage::NlBobyqa`). Errors with a clear message if `stage` is anything
/// else — caller's job to dispatch correctly.
#[allow(clippy::too_many_arguments)]
pub fn run_stage(
    fit: &FitConfigV2,
    stage_name: &str,
    stage: &Stage,
    stage_dir: &Path,
    seed: u64,
    starts_from: Option<&str>,
    parent_fit_hash: &str,
    model_identity: &str,
    data_hashes: &[(String, String)],
    dt_check_cfg: &DtCheckConfig,
    dt_check_strict: bool,
) -> Result<(), String> {
    let (algorithm, knobs) = extract_nlopt_config(stage)?;

    eprintln!(
        "\x1b[33mℹ {} ({}):\x1b[0m deterministic MLE on the ODE-skeleton \
         likelihood.",
        algorithm.as_str(),
        match algorithm {
            NloptAlgorithm::Sbplx => "Subspace simplex; robust to boundary non-smoothness",
            NloptAlgorithm::Bobyqa => "Quadratic trust region; smooth-objective only",
        }
    );
    eprintln!(
        "  camdl computes p(y|θ, ODE_skeleton) under {algorithm}, not the \
         stochastic-process p(y|θ) IF2/PGAS/PMMH compute. In low-noise \
         regimes the two converge empirically; verify rather than assume.",
        algorithm = algorithm.as_str()
    );

    let prior_state = starts_from.map(FitState::load).transpose()?;
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
        prior_state.as_ref(),
        n_chains,
        /* n_particles */ 1,
        /* n_iterations */ 1,
        /* cooling */ 1.0,
        /* cooling_target_iters */ 1,
        seed,
        // gh#506: was `prior_state.is_none()`. Under this stage's default
        // `init = "single"` the corruption was masked — `build_chain_param_vecs`
        // returns None there and the chains fall back to `base_params`, which
        // carries `[estimate].start` correctly. It bit `init = "uniform"`,
        // whose chain 0 is documented to keep the seeded start and instead
        // kept the random draw.
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

    // Honour the `init` toml key (rust field: `init_method`) as written.
    // Default for NLopt stages is `single` (every chain starts at the
    // user's seeded values) so the runs land in a finite-likelihood
    // region by construction. For models with tight bounds the user can
    // set `init = "lhs"` to get spread starts; the chain-agreement gate
    // then becomes informative. With Single + chains > 1, every chain's
    // outcome is identical (Sbplx/BOBYQA are deterministic), so we run
    // just one chain and skip the wasted compute.
    let effective_chains = if knobs.init_method == InitMethod::Single {
        1
    } else {
        n_chains
    };
    if effective_chains < n_chains {
        eprintln!(
            "  init=single with chains={}: collapsing to 1 chain \
             (Sbplx/BOBYQA are deterministic, redundant chains would \
             produce identical output). Set `init = \"lhs\"` for \
             multi-start basin exploration.",
            n_chains
        );
    }
    let chain_starts: Vec<Vec<f64>> = build_chain_param_vecs(
        &knobs.init_method,
        &run_config.estimated_params,
        &run_config.base_params,
        effective_chains,
        seed,
    )?
    .unwrap_or_else(|| vec![run_config.base_params.clone(); effective_chains]);
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
        dt_check_strict,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
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
    let ivp_params: Vec<String> = arc_config
        .estimated_params
        .iter()
        .filter(|p| p.ivp)
        .map(|p| p.name.clone())
        .collect();

    let fit_state = FitState {
        stage: stage_name.to_string(),
        seed,
        timestamp: crate::cas::iso8601_utc(std::time::SystemTime::now()),
        input_hash: None,
        camdl_version: Some(crate::version::VERSION.to_string()),
        best_loglik: winner.loglik,
        initial_loglik: f64::NAN,
        best_chain: winner_idx,
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
        ivp_params,
        chain_logliks,
        chain_eval_logliks: Vec::new(),
        chain_eval_ses: Vec::new(),
        resolved_gate: Some(knobs.gate.clone()),
        resolved_loglik_eval: None,
        // gh#51 v3: NLopt SurveyTopK support is deferred (v2 ships on
        // IF2 / PMMH / PGAS only). Record the user-set init_method
        // verbatim; SurveyTopK refuses upstream in
        // build_chain_param_vecs.
        chain_init_source: Some(format!("{}", knobs.init_method)),
        // gh#52, gh#227: deterministic ODE dt-check at θ̂ (above). Skipped →
        // omit the block, mirroring the IF2 path's legacy semantics.
        dt_check: if matches!(dt_check_result.verdict, dt_check::DtCheckVerdict::Skipped) {
            None
        } else {
            Some(dt_check_result)
        },
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
        stage: stage_name.to_string(),
        best_chain: winner_idx,
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
    stage: &Stage,
) -> Result<(NloptAlgorithm, &NloptStageConfig), String> {
    match stage {
        Stage::NlSbplx(c) => Ok((NloptAlgorithm::Sbplx, c)),
        Stage::NlBobyqa(c) => Ok((NloptAlgorithm::Bobyqa, c)),
        other => Err(format!(
            "nlopt_stage::run_stage: expected nl-sbplx or nl-bobyqa, got {}",
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
) -> ChainOutcome {
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
    );
    match result {
        Ok(r) => ChainOutcome {
            params: r.params,
            loglik: r.loglik,
            status: r.status,
            n_evals: r.n_evals,
        },
        Err(e) => {
            eprintln!("nlopt chain failed config: {e}");
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
/// parameter vector per evaluation, calls `compute_ode_loglik`, and
/// returns the loglik (or `f64::NEG_INFINITY` if the model blew up).
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
) -> Result<OptResult, String> {
    let initial_est: Vec<f64> = est_indices.iter().map(|&i| full_start[i]).collect();

    // Closure-owned mutable state. NLopt's callback signature requires
    // `Fn`; the user_data smuggle inside `optimize_det` lets `objective`
    // be `FnMut`, so we mutate `full_params` in place per call (avoids
    // a Vec alloc per eval).
    let mut full_params = full_start.to_vec();
    let est_indices_local = est_indices.to_vec();
    let compiled_local = Arc::clone(compiled);
    let obs_model_local = Arc::clone(obs_model);
    let obs_times_local = obs_times.to_vec();
    let objective = move |est: &[f64]| -> f64 {
        for (slot, &model_idx) in est_indices_local.iter().enumerate() {
            full_params[model_idx] = est[slot];
        }
        compute_ode_loglik(
            &compiled_local,
            &obs_model_local,
            &obs_times_local,
            dt,
            &full_params,
            dt, // burnin_dt = dt ⇒ coarse burn-in off (nlopt uses the fine step)
        )
        .unwrap_or(f64::NEG_INFINITY)
    };

    optimize_det(
        algorithm,
        &initial_est,
        bounds,
        tolerance,
        max_evals,
        objective,
    )
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
    use crate::fit::config_v2::{
        GateConfig, NloptStageConfig, Stage, StartsFrom,
    };
    use crate::fit::init::InitMethod;
    use crate::run_meta::InferenceBackend;
    use sim::inference::deterministic::SuccessState;

    fn nlopt_config() -> NloptStageConfig {
        NloptStageConfig {
            backend: InferenceBackend::Ode,
            chains: 4,
            tolerance: 1e-6,
            max_evals: 5000,
            starts_from: StartsFrom::default(),
            init_method: InitMethod::Single,
            survey_path: None,
            survey_top_k_n: None,
            gate: GateConfig::default(),
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
        let stage = Stage::NlSbplx(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&stage).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Sbplx);
        assert_eq!(cfg.chains, 4);
        assert_eq!(cfg.tolerance, 1e-6);
    }

    #[test]
    fn extract_nlopt_config_bobyqa_returns_bobyqa_algorithm() {
        let stage = Stage::NlBobyqa(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&stage).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Bobyqa);
        assert_eq!(cfg.chains, 4);
    }

    #[test]
    fn extract_nlopt_config_rejects_non_nlopt_stage() {
        // PFilter is the simplest non-nlopt variant to construct.
        let stage = Stage::PFilter {
            backend: InferenceBackend::ChainBinomial,
            particles: 100,
            replicates: None,
            starts_from: StartsFrom::default(),
            record_ancestry: false,
            record_prequential: false,
        };
        let err = extract_nlopt_config(&stage).expect_err("must reject");
        assert!(err.contains("expected nl-sbplx or nl-bobyqa"),
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
