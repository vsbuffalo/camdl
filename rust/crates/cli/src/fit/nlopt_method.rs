//! Per-method runner for NLopt deterministic MLE — Phase 1 of the
//! ODE-inference proposal
//! (`docs/dev/proposals/2026-05-04-ode-inference-three-phase.md`).
//!
//! Mirrors the shape of `pgas::run_method` / `pmmh::run_method`: parses the
//! NLopt-shaped `Method` variant, builds a `FitRunConfig`, draws LHS-spread
//! per-chain starts, runs each chain's NLopt optimization in parallel via
//! rayon, aggregates the winner, writes per-chain outputs +
//! `fit_state.toml`, and emits a two-leg convergence diagnostic
//! (chain-agreement + decibans-spread) as a stdout verdict line.
//!
//! The optimizer itself lives in `sim::inference::deterministic`; this
//! module is the orchestration layer wiring it to the fit framework.

use crate::out::paint::Style;
use std::sync::Arc;

use sim::inference::deterministic::{
    optimize_det, NloptAlgorithm, OptResult, OptStatus,
};

use score::convergence::ChainIndex;
use crate::fit::config_v2::{Algorithm, DtCheckConfig, GateConfig, Method, NloptConfig, Problem};
use crate::fit::dt_check;
use crate::fit::loglik::LoglikType;
use crate::fit::methods::check_model_capabilities;
use crate::fit::runner::FitRunConfig;
use sim::inference::compute_ode_loglik;
use crate::fit::state::FitState;
use crate::fit::provenance;

/// Run a single NLopt-flavoured method (`Algorithm::NlSbplx`,
/// `Algorithm::NlBobyqa` or `Algorithm::NlLbfgs`). Errors with a clear message
/// if the algorithm is anything else — caller's job to dispatch correctly.
#[allow(clippy::too_many_arguments)]
pub fn run_method(
    out: &crate::out::Out,
    fit: &Problem,
    method: &Method,
    files: &dyn super::output::RunWriter,
    seed: u64,
    starts: &super::chain_starts::ResolvedStarts,
    parent_fit_hash: &str,
    model_identity: &str,
    dt_check_cfg: &DtCheckConfig,
    // The method's liveness/progress heartbeat, owned by the runner
    // (`fit::method_heartbeat`). Its step is one objective evaluation, against
    // the `max_evals` budget (gh#900).
    heartbeat: &io::HeartbeatGuard,
) -> Result<(), String> {
    let method_name = method.algorithm.method_name();
    let (algorithm, knobs) = extract_nlopt_config(&method.algorithm)?;

    let head = format!("ℹ {} ({}):",
        algorithm.as_str(),
        match algorithm {
            NloptAlgorithm::Sbplx => "Subspace simplex; robust to boundary non-smoothness",
            NloptAlgorithm::Bobyqa => "Quadratic trust region; smooth-objective only",
            NloptAlgorithm::Lbfgs =>
                "Quasi-Newton on the forward-sensitivity gradient; differentiable models only",
        });
    out.say(format_args!(
        "{} deterministic MLE on the ODE-skeleton likelihood.",
        crate::out::paint::stderr().paint(&head, Style::Warn)
    ));
    out.say(format_args!(
        "  camdl computes p(y|θ, ODE_skeleton) under {algorithm}, not the \
         stochastic-process p(y|θ) IF2/PGAS/PMMH compute. In low-noise \
         regimes the two converge empirically; verify rather than assume.",
        algorithm = algorithm.as_str()
    ));

    let n_chains = knobs.chains;
    if n_chains == 0 {
        return Err(format!(
            "method '{method_name}': chains must be ≥ 1; got 0"
        ));
    }

    // FitRunConfig is shaped around IF2 / PF (n_particles, n_iterations,
    // cooling). NLopt doesn't use any of those — pass placeholders that
    // produce a valid config without affecting the run. `dt` is read from
    // `model.simulation.dt` inside `compute_ode_loglik`; if2_config.dt is
    // a fallback only.
    let run_config = FitRunConfig::build(
        fit,
        Some(&method.algorithm), &super::steps::ResolvedSteps::None,
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
        .map(|p| p.bounds())
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
        out.say(format_args!(
            "  starts = {} with chains={}: collapsing to 1 chain \
             ({} is deterministic, redundant chains would produce \
             identical output). Set `starts = \"lhs\"` for multi-start \
             basin exploration.",
            starts.rule.spelled(), n_chains, algorithm.as_str()
        ));
    }
    let drawn = crate::fit::runner::draw_chain_starts_for(
        &run_config, &fit.estimate, starts, effective_chains, seed,
    )?;
    let chain_starts: Vec<Vec<f64>> =
        drawn.to_param_vecs(&run_config.estimated_params, &run_config.base_params);
    let n_chains = effective_chains;

    let arc_config = Arc::new(run_config);
    let bounds_ref = bounds.clone();
    let est_indices_ref = est_indices.clone();

    // Build the obs model + obs-time vector ONCE for the whole method.
    // The per-chain closures borrow these via Arc, so per-eval cost is
    // one ODE solve + one obs scoring pass with no reconstruction.
    let obs_model = Arc::new(arc_config.build_obs_model());
    let obs_times: Vec<f64> = arc_config.observations.iter().map(|o| o.time).collect();
    // The one dt rule: `FitRunConfig` carries the resolved step.
    let dt = arc_config.if2_config.dt;

    // `chain_results.tsv` is kept current while the chains run: the header
    // now, and one row the moment each chain finishes. A multi-start search
    // can run for an hour with its slowest start holding up every other; until
    // this, nothing about a finished start reached disk before the last one
    // did, and a run killed partway lost every result it had. The rows land
    // in completion order; the file is rewritten in chain order once every
    // chain is in (`write_per_chain_files`, below).
    let live_results = ChainResultsFile::create(files, &est_names)?;
    // `opt_trace.tsv`, the searches' running best, written while they run
    // (`super::opt_trace`). The header goes down now, before any search
    // writes a row.
    let opt_trace = super::opt_trace::OptTraceFile::create(files, &est_names)?;

    let t0 = std::time::Instant::now();
    // At most one chain per worker (gh#821, `chain_fanout`).
    let mut chain_outcomes: Vec<(usize, ChainOutcome)> = super::chain_fanout::run_chains(n_chains, |chain_idx| {
            let outcome = run_one_chain(out,
                chain_idx,
                &opt_trace,
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
            // A failed append costs the live view one row, not the run: the
            // outcome is still returned, and the final rewrite below writes
            // every row (or fails the method if the directory is unwritable).
            if let Err(e) = live_results.append(chain_idx, &outcome) {
                out.say(format_args!(
                    "{} chain {}: could not append to {}: {e}",
                    crate::out::paint::warning_tag(),
                    ChainIndex(chain_idx),
                    live_results.path.display()
                ));
            }
            (chain_idx, outcome)
        });
    drop(live_results);
    drop(opt_trace);
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

    // Per-chain final params dump for inspection, rewritten in chain order
    // over the completion-ordered file the chains appended to. Written before
    // the refusal below, so a refused method still records what each chain
    // reached.
    write_per_chain_files(files, &chain_outcomes, &est_names)?;

    // A chain that FAILED (the gradient could not be taken, or the cell was
    // mis-configured) reports `-inf` without having scored anything, so when
    // every chain failed the cause is the failure, not the starting values. Say
    // the chains' own reason — a refusal the gradient gate deferred to run time
    // (a scenario fit), or a gradient that is not computable at the start —
    // rather than the impossible-region diagnosis below, which is about chains
    // that did evaluate the likelihood.
    if sim::inference::no_finite_anchor(winner.loglik)
        && chain_outcomes.iter().all(|(_, c)| c.failure.is_some())
    {
        let reasons: Vec<&String> = chain_outcomes.iter().filter_map(|(_, c)| c.failure.as_ref()).collect();
        let distinct: std::collections::BTreeSet<&String> = reasons.iter().copied().collect();
        return Err(format!(
            "{method_name}: all {n_chains} chain(s) failed before reaching an optimum, so there \
             is none to report. {}:\n  {}\nPer-chain results: {}.",
            if distinct.len() == 1 { "Every chain gave the same reason".to_string() }
            else { format!("The first chain's reason ({} distinct in all)", distinct.len()) },
            reasons[0].trim_end_matches('.'),
            files.folder().join(CHAIN_RESULTS_FILE).display()
        ));
    }

    // gh#226 / gh#916. Whole-fit backstop, as in the PGAS, PMMH and IF2
    // drivers: the winner is the maximum over every chain, so a non-finite
    // winner means not one chain scored a finite point. Without this the
    // method writes a `-inf` best log-likelihood and an `mle_params.toml` at
    // wherever the search stopped, and exits 0. `optimize_det` reports its
    // internal non-finite floor as `-inf`, which is what makes this reachable.
    if sim::inference::no_finite_anchor(winner.loglik) {
        return Err(format!(
            "{method_name}: all {n_chains} chain(s) reached no finite \
             log-likelihood (best = {}). The ODE likelihood is `-inf` at every \
             θ the search evaluated, so there is no optimum to report. Most \
             often the starting values sit in an impossible region — check \
             those first (try `starts = \"lhs\"` under `[method]`, or a \
             different start); less often the data are impossible under this \
             model (for example a positive count observed where the projected \
             mean is 0 for every θ in the bounds). Also check the observation \
             model and parameter bounds. Per-chain results: {}.",
            winner.loglik,
            files.folder().join(CHAIN_RESULTS_FILE).display()
        ));
    }

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
    print_verdict(out,
        &convergence,
        elapsed.as_secs_f64(),
        n_chains,
        knobs.gate.decibans_thresh,
    );
    if let Some(w) = all_maxeval_warning(&convergence, n_chains, knobs.max_evals) {
        out.say(format_args!("{} {w}", crate::out::paint::warning_tag()));
    }

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
    .unwrap_or_else(|e| super::abandon_method(out, format!("error: {e}")));
    dt_check::print_terminal_report(out, &dt_check_result);

    // Persist the winner's full parameter vector via fit_state.toml so
    // a later fit can start from it.
    let mut start_values = std::collections::BTreeMap::new();
    for (slot, name) in est_names.iter().enumerate() {
        start_values.insert(name.clone(), winner.params[slot]);
    }
    // An optimizer perturbs nothing, so it has no initial-state timing.
    let initial_state_params: Vec<String> = arc_config.steps.initial_state_params();

    let fit_state = FitState {
        method: method_name.to_string(),
        seed,
        input_hash: None,
        camdl_version: Some(crate::version::VERSION_SHORT.to_string()),
        best_loglik: winner.loglik,
        initial_loglik: f64::NAN,
        best_chain: ChainIndex(winner_idx),
        n_chains,
        n_good_chains: Some(
            chain_outcomes
                .iter()
                .filter(|(_, c)| matches!(c.status, OptStatus::Converged(_)))
                .count(),
        ),
        start_values: start_values.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        loglik_type: Some(LoglikType::OdeMarginal),
        acceptance_rate: None,
        tail_chain_agreement: convergence.chain_agreement.iter()
            .map(|(k, v)| (k.clone(), *v)).collect(),
        initial_state_params,
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
        optimizer_convergence: Some(crate::fit::state::OptimizerConvergence {
            n_converged: convergence.n_converged,
            n_maxeval: convergence.n_maxeval,
            n_failed: convergence.n_failed,
            max_rel_range: convergence.max_rel_range,
            max_abs_range: convergence.max_abs_range,
            chain_agreement_failed: convergence.chain_agreement_failed,
            delta_loglik: convergence.delta_loglik,
            decibans_failed: convergence.decibans_failed,
        }),
        chain_loglik_agreement: None,
    };
    fit_state
        .save(files)
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
    let mle_meta = provenance::MleMetadata {
        input_hash: parent_fit_hash.to_string(),
        // The leaf is `<fit folder>/<method>-<h8>/seed_<n>-<h8>`.
        model_identity: model_identity.to_string(),
        seed,
        method: method_name.to_string(),
        best_chain: ChainIndex(winner_idx),
        // NLopt methods always run on the ODE backend (validated by
        // methods::validate_combo); record it from the inference domain
        // (total `From<InferenceBackend>` into the forward provenance field).
        backend: crate::run_meta::InferenceBackend::Ode.into(),
        dt: arc_config.if2_config.dt,
        loglik: winner.loglik,
        loglik_sd: 0.0,
        n_particles: 0,
        ess_at_mle: None,
    };
    if let Err(e) = provenance::mle_params_text(&all_params, &mle_meta)
        .and_then(|text| files.put("mle_params.toml", text.as_bytes()))
    {
        out.say(format_args!("warning: writing mle_params.toml failed: {e}"));
    }

    Ok(())
}

fn extract_nlopt_config(
    algorithm: &Algorithm,
) -> Result<(NloptAlgorithm, &NloptConfig), String> {
    match algorithm {
        Algorithm::NlSbplx(c) => Ok((NloptAlgorithm::Sbplx, c)),
        Algorithm::NlBobyqa(c) => Ok((NloptAlgorithm::Bobyqa, c)),
        Algorithm::NlLbfgs(c) => Ok((NloptAlgorithm::Lbfgs, c)),
        other => Err(format!(
            "nlopt_method::run_method: expected nl-sbplx, nl-bobyqa or nl-lbfgs, got {}",
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
    /// Why the chain produced no optimum (`status == Failed`): the gradient
    /// could not be taken, or the cell was mis-configured. Carried so the
    /// method's own failure names this reason instead of inferring one from
    /// the `-inf` log-likelihood a failed chain reports.
    failure: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_one_chain(
    out: &crate::out::Out,
    chain_idx: usize,
    opt_trace: &super::opt_trace::OptTraceFile,
    algorithm: NloptAlgorithm,
    knobs: &NloptConfig,
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
    //
    // Each evaluation also goes to this search's `opt_trace.tsv` rows, which
    // are written only every few evaluations. Whenever a row is, the same
    // count and best go to the heartbeat's per-search row (`searches`), so the
    // two agree and the heartbeat's shared lock is taken once per trace row,
    // not once per evaluation from every search at once. A row is due at
    // least every 2 s, finer than the heartbeat's 5 s writes. Both read the
    // value after the objective computed it; neither can change it.
    let mut trace = super::opt_trace::SearchTrace::new(opt_trace, chain_idx);
    let mut on_eval = |e: Evaluation<'_>| {
        heartbeat.bump(e.evals);
        if trace.observe(e.evals, e.est, e.loglik) {
            heartbeat.search(chain_idx, e.evals, trace.best_loglik());
        }
    };
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
        Some(&mut on_eval),
    );
    if let Some(evals) = trace.evals() {
        heartbeat.search(chain_idx, evals, trace.best_loglik());
    }
    if let Some(e) = trace.finish() {
        // The trace is a live view: a failed write costs it its rows, not the
        // search its result.
        out.say(format_args!(
            "{} chain {}: could not append to {}: {e}",
            crate::out::paint::warning_tag(),
            ChainIndex(chain_idx),
            opt_trace.path.display()
        ));
    }
    match result {
        Ok(r) => ChainOutcome {
            params: r.params,
            loglik: r.loglik,
            status: r.status,
            n_evals: r.n_evals,
            failure: None,
        },
        Err(e) => {
            // Either the cell was mis-configured (dimension / bounds) or the
            // gradient could not be taken somewhere along the search. Both are
            // structural: this chain has no optimum to report, so it carries
            // `Failed` and a `-inf` loglik, which loses the winner comparison
            // to any chain that did converge.
            out.say(format_args!("{} {}: {e}", crate::out::paint::stderr().paint("✗", Style::Err), algorithm.as_str()));
            ChainOutcome {
                params: est_indices.iter().map(|&i| full_start[i]).collect(),
                loglik: f64::NEG_INFINITY,
                status: OptStatus::Failed,
                n_evals: 0,
                failure: Some(e),
            }
        }
    }
}

/// One objective evaluation, as [`optimize_cell`] reports it to its caller.
pub struct Evaluation<'a> {
    /// The cell's running evaluation count, from 1.
    pub evals: u64,
    /// The point scored: the estimated slots, natural scale, in `est_indices`
    /// order.
    pub est: &'a [f64],
    /// The value the objective returns to NLopt for this point, non-finite
    /// values included (`optimize_det` floors those on its side).
    pub loglik: f64,
}

/// One NLopt cell-level optimization — shared by `nlopt_method::run_method`
/// (multi-chain MLE for fit.toml methods) and `cli::profile` (per-cell MLE
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
/// Both pass `burnin_dt = dt`, i.e. no coarse warm-up: the NLopt method uses
/// the fine step throughout, and value and gradient must be taken on the same
/// trajectory or the reported optimum is not the optimum of the reported
/// likelihood.
///
/// `on_eval` is handed each [`Evaluation`] once the objective has scored it,
/// before the value goes back to NLopt — the seam a caller uses to report
/// search progress (gh#900) and to trace the search (`opt_trace.tsv`). It is
/// the only per-evaluation hook there is: the objective closure is built here,
/// so a caller cannot wrap it from outside. It sees the value and cannot
/// change it. `camdl profile` passes `None`; its progress is a grid cell, not
/// an evaluation, and it writes no trace. One `det_grad` call counts as one
/// evaluation, the same unit as one `compute_ode_loglik` call.
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
    mut on_eval: Option<&mut dyn FnMut(Evaluation<'_>)>,
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
        let value = 'score: {
            let Some(slot) = grad else {
                // Derivative-free: value only.
                break 'score compute_ode_loglik(
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
                break 'score f64::NEG_INFINITY;
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
        if let Some(observe) = on_eval.as_deref_mut() {
            observe(Evaluation { evals, est, loglik: value });
        }
        value
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
                     search in. If the gradient is non-finite at every starting \
                     point, the cause is usually in the model rather than the \
                     parameters: a rate whose derivative is singular where a \
                     compartment is exactly zero (e.g. `X ^ a` with 0 < a < 1 \
                     and X starting at 0). Otherwise, narrow the `[estimate]` \
                     bounds to exclude the region or start elsewhere. The \
                     derivative-free `algorithm = \"nl-sbplx\"` needs no gradient.",
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

/// The per-chain results table, one row per chain.
const CHAIN_RESULTS_FILE: &str = "chain_results.tsv";

/// `chain_results.tsv`'s header line, newline included.
fn chain_results_header(est_names: &[String]) -> String {
    let mut line = String::from("chain\tloglik\tstatus\tn_evals");
    for name in est_names {
        line.push('\t');
        line.push_str(name);
    }
    line.push('\n');
    line
}

/// One chain's `chain_results.tsv` row, newline included. The live append and
/// the final rewrite both format through this, so a row reads the same in
/// either.
fn chain_results_row(chain_idx: usize, c: &ChainOutcome) -> String {
    use std::fmt::Write;
    let mut line = format!(
        "{}\t{:.6}\t{}\t{}",
        ChainIndex(chain_idx),
        c.loglik,
        c.status.as_str(),
        c.n_evals,
    );
    for v in &c.params {
        write!(line, "\t{v:.10}").expect("writing to a String cannot fail");
    }
    line.push('\n');
    line
}

/// `chain_results.tsv` while the chains run: created with its header, then
/// one row appended per chain as that chain finishes, from whichever worker
/// ran it.
///
/// Each row is formatted in full first and handed to the file in a single
/// unbuffered append, so a reader tailing the file sees
/// whole rows only, and a run killed between chains leaves every finished
/// chain's row on disk. [`TraceWriter`](super::trace_writer::TraceWriter) is
/// not used: its fixed `index, loglik, log_posterior` columns are not this
/// table's, and it batches rows behind a `BufWriter`, which is the delay this
/// file exists to remove.
struct ChainResultsFile {
    /// Where the file can be read while it is written.
    path: std::path::PathBuf,
    file: super::output::OutFile,
}

impl ChainResultsFile {
    fn create(files: &dyn super::output::RunWriter, est_names: &[String]) -> Result<Self, String> {
        let file = files.create(CHAIN_RESULTS_FILE)?;
        file.write_all(chain_results_header(est_names).as_bytes())?;
        Ok(ChainResultsFile { path: files.folder().join(CHAIN_RESULTS_FILE), file })
    }

    fn append(&self, chain_idx: usize, c: &ChainOutcome) -> Result<(), String> {
        self.file.write_all(chain_results_row(chain_idx, c).as_bytes())
    }
}

/// The finished `chain_results.tsv`: every chain's row in chain order,
/// replacing the completion-ordered rows whole, so a viewer reading the file
/// during the swap sees either the completion-ordered rows or the sorted ones,
/// never a truncated file.
fn write_per_chain_files(
    files: &dyn super::output::RunWriter,
    chain_outcomes: &[(usize, ChainOutcome)],
    est_names: &[String],
) -> Result<(), String> {
    let mut text = chain_results_header(est_names);
    for (chain_idx, c) in chain_outcomes {
        text.push_str(&chain_results_row(*chain_idx, c));
    }
    files.put(CHAIN_RESULTS_FILE, text.as_bytes())
}

/// Two-leg convergence diagnostic for NLopt methods — generalises IF2's
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

    // Both legs judge the converged chains. A chain that ran out of
    // evaluations or failed stopped where its budget or error did, so its
    // parameters and log-likelihood say nothing about whether the converged
    // chains found one optimum; the status counts above report it, and
    // `OptimizerConvergence::passes` fails a run that has any.
    let converged: Vec<&ChainOutcome> = chain_outcomes
        .iter()
        .filter(|(_, c)| matches!(c.status, OptStatus::Converged(_)))
        .map(|(_, c)| c)
        .collect();
    let mut chain_agreement = HashMap::new();
    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut chain_agreement_failed = false;
    if converged.len() >= 2 {
        for (slot, name) in est_names.iter().enumerate() {
            let vals: Vec<f64> = converged.iter().map(|c| c.params[slot]).collect();
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

    let logliks: Vec<f64> = converged.iter().map(|c| c.loglik).collect();
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

/// The chain-agreement line of the end-of-run verdict. With fewer than two
/// converged chains the leg is not computed, and says so rather than
/// printing a 0.00% range and a ✓.
fn chain_agreement_line(painter: &crate::out::paint::Painter, v: &ConvergenceVerdict) -> String {
    if v.n_converged < 2 {
        return "  chain-agreement: not computed (fewer than two converged chains)".to_string();
    }
    format!(
        "  chain-agreement: rel range = {:.2}% bound | abs range = {:.3e}   {}",
        v.max_rel_range * 100.0,
        v.max_abs_range,
        if v.chain_agreement_failed { painter.paint("✗", Style::Err) } else { painter.paint("✓", Style::Ok) }
    )
}

fn print_verdict(
    out: &crate::out::Out,
    v: &ConvergenceVerdict,
    wall_secs: f64,
    n_chains: usize,
    decibans_thresh: f64,
) {
    let p = crate::out::paint::stderr();
    let ok = p.paint("✓", Style::Ok);
    let bad = p.paint("✗", Style::Err);
    out.say("");
    out.say(format_args!(
        "  status: {} converged, {} max-eval, {} failed (of {})",
        v.n_converged, v.n_maxeval, v.n_failed, n_chains
    ));
    out.say(format_args!("{}", chain_agreement_line(&p, v)));
    out.say(format_args!(
        "  loglik-eval:     Δ = {:.1} dB / threshold {:.0} dB                {}",
        v.delta_loglik * NATS_TO_DB,
        decibans_thresh,
        if v.decibans_failed { bad } else { ok }
    ));
    out.say(format_args!("  wall: {:.2}s ({} chains)", wall_secs, n_chains));
}

/// The warning for a run in which every chain stopped on its evaluation
/// budget. Neither leg of the verdict judges such a run — both read converged
/// chains only, so the loglik line shows Δ = 0 and a ✓ — and the status count
/// is the one place it shows. The winner is then wherever the best chain's
/// budget ran out, which the warning says in words.
fn all_maxeval_warning(v: &ConvergenceVerdict, n_chains: usize, max_evals: usize) -> Option<String> {
    if n_chains == 0 || v.n_maxeval != n_chains {
        return None;
    }
    let every = if n_chains == 1 { "the only chain".to_string() } else { format!("all {n_chains} chains") };
    Some(format!(
        "{every} stopped at max_evals = {max_evals} without meeting the tolerance. \
         No chain converged, so the reported optimum is where the best chain's \
         evaluation budget ran out, not an optimum; both legs of the verdict \
         above read converged chains only, so they judge nothing here. Raise \
         `max_evals` under `[method]` and refit."
    ))
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{Algorithm, GateConfig, NloptConfig};
    use crate::run_meta::InferenceBackend;
    use sim::inference::deterministic::SuccessState;

    // ── The ODE-gradient fixture ─────────────────────────────────────
    //
    // The `seir_observations` golden, in the configuration the ODE gradient's
    // own finite-difference oracle uses
    // (`sim::inference::ode_grad::tests::det_grad_matches_finite_difference_*`):
    // emitted `rate_state_grad`, a native `neg_binomial(mean = rho·incidence)`
    // incidence stream, a state (`projected = I`) stream, and an explicit (constant)
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

    // ── opt_trace.tsv, observed from a real search ───────────────────

    /// Every derivative-free and gradient search on the SEIR fixture, from a
    /// displaced start: `(algorithm, est_indices, bounds, start)`.
    fn traced_cells() -> Vec<(NloptAlgorithm, Vec<usize>, Vec<(f64, f64)>, Vec<f64>)> {
        let (compiled, _, _, params) = seir_gradient_fixture();
        let est: Vec<usize> = ["beta", "rho"].iter().map(|n| compiled.param_index[*n]).collect();
        let mut start = params.clone();
        start[est[0]] = 0.22;
        start[est[1]] = 0.35;
        let bounds = vec![(0.05, 0.5), (0.05, 0.95)];
        [NloptAlgorithm::Sbplx, NloptAlgorithm::Bobyqa, NloptAlgorithm::Lbfgs]
            .into_iter()
            .map(|a| (a, est.clone(), bounds.clone(), start.clone()))
            .collect()
    }

    /// The trace observes the search and does not steer it: with an observer
    /// attached, every algorithm returns the same optimum, bit for bit, after
    /// the same number of evaluations as without one.
    #[test]
    fn an_observed_search_is_bit_identical_to_an_unobserved_one() {
        let (compiled, obs_model, obs_times, _) = seir_gradient_fixture();
        let dir = tempfile::tempdir().unwrap();
        let file = super::super::opt_trace::OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(2)).unwrap();
        for (algorithm, est, bounds, start) in traced_cells() {
            let run = |observe: Option<&mut dyn FnMut(Evaluation<'_>)>| {
                optimize_cell(algorithm, &compiled, &obs_model, &obs_times, 1.0,
                    &bounds, &est, &start, 1e-8, 400, observe)
                .expect("the fixture optimizes")
            };
            let bare = run(None);
            let mut trace = super::super::opt_trace::SearchTrace::new(&file, 0);
            let mut seen = Vec::new();
            let mut observe = |e: Evaluation<'_>| {
                seen.push(e.loglik);
                trace.observe(e.evals, e.est, e.loglik);
            };
            let traced = run(Some(&mut observe));
            assert!(trace.finish().is_none());
            assert_eq!(traced.loglik.to_bits(), bare.loglik.to_bits(), "{algorithm:?}");
            assert_eq!(
                traced.params.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bare.params.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{algorithm:?}"
            );
            assert_eq!((traced.n_evals, seen.len()), (bare.n_evals, bare.n_evals), "{algorithm:?}");
        }
    }

    /// A real search's rows are on disk while it runs — read from inside the
    /// objective, as a reader tailing the file would — and its last row holds
    /// the optimum NLopt reports: the best value and the point that scored it.
    #[test]
    fn a_real_search_traces_live_and_ends_on_its_optimum() {
        let (compiled, obs_model, obs_times, _) = seir_gradient_fixture();
        for (algorithm, est, bounds, start) in traced_cells() {
            let dir = tempfile::tempdir().unwrap();
            let file = super::super::opt_trace::OptTraceFile::create(&crate::fit::output::DirWriter::new(dir.path()), &names(2)).unwrap();
            let path = file.path.clone();
            let mut trace = super::super::opt_trace::SearchTrace::new(&file, 2);
            // What a reader sees mid-search: at eval 2, and at eval 27 when
            // the search gets that far.
            let mut mid_run = std::collections::BTreeMap::new();
            let mut observe = |e: Evaluation<'_>| {
                trace.observe(e.evals, e.est, e.loglik);
                if e.evals == 2 || e.evals == 27 {
                    mid_run.insert(e.evals, std::fs::read_to_string(&path).unwrap());
                }
            };
            let r = optimize_cell(algorithm, &compiled, &obs_model, &obs_times, 1.0,
                &bounds, &est, &start, 1e-8, 400, Some(&mut observe))
            .expect("the fixture optimizes");
            assert!(trace.finish().is_none());
            let evals_at = |text: &String| -> Vec<String> {
                text.lines().skip(1).map(|l| l.split('\t').nth(1).unwrap().to_string()).collect()
            };
            assert_eq!(evals_at(&mid_run[&2]), ["1"], "{algorithm:?}: the first row, mid-search");
            if algorithm != NloptAlgorithm::Lbfgs {
                // L-BFGS converges here in fewer than 27 evaluations.
                assert_eq!(evals_at(&mid_run[&27]), ["1", "26"],
                    "{algorithm:?}: every 25 evaluations, mid-search");
            }

            let text = std::fs::read_to_string(&path).unwrap();
            let last: Vec<&str> = text.lines().last().unwrap().split('\t').collect();
            assert_eq!(last[0], "2");
            assert_eq!(last[1], r.n_evals.to_string(), "{algorithm:?}: the final row is the last eval");
            let best: f64 = last[3].parse().unwrap();
            let point: Vec<f64> = last[4..].iter().map(|v| v.parse().unwrap()).collect();
            if algorithm == NloptAlgorithm::Lbfgs {
                // NLopt's L-BFGS reports its final iterate, not the best point
                // it evaluated: on this fixture the two differ by ~3e-14 nats
                // and ~4e-10 in θ. The trace's best is never below NLopt's.
                assert!(best >= r.loglik && best - r.loglik < 1e-9,
                    "{algorithm:?}: best {best} vs NLopt's {}", r.loglik);
                for (a, b) in point.iter().zip(&r.params) {
                    assert!((a - b).abs() < 1e-6, "{algorithm:?}: {point:?} vs {:?}", r.params);
                }
            } else {
                assert_eq!(best.to_bits(), r.loglik.to_bits(), "{algorithm:?}: best is NLopt's optimum");
                assert_eq!(point, r.params, "{algorithm:?}: the best point is NLopt's");
            }
        }
    }

    fn nlopt_config() -> NloptConfig {
        NloptConfig {
            backend: InferenceBackend::Ode,
            chains: 4,
            tolerance: 1e-6,
            max_evals: 5000,
            gate: GateConfig::default(),
            dt_check: crate::fit::config_v2::DtCheckConfig::default(),
        }
    }

    fn outcome(loglik: f64, params: Vec<f64>, status: OptStatus) -> ChainOutcome {
        ChainOutcome { loglik, params, status, n_evals: 1, failure: None }
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("p{i}")).collect()
    }

    // ── extract_nlopt_config ─────────────────────────────────────────

    #[test]
    fn extract_nlopt_config_sbplx_returns_sbplx_algorithm() {
        let method = Algorithm::NlSbplx(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&method).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Sbplx);
        assert_eq!(cfg.chains, 4);
        assert_eq!(cfg.tolerance, 1e-6);
    }

    #[test]
    fn extract_nlopt_config_bobyqa_returns_bobyqa_algorithm() {
        let method = Algorithm::NlBobyqa(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&method).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Bobyqa);
        assert_eq!(cfg.chains, 4);
    }

    #[test]
    fn extract_nlopt_config_lbfgs_returns_lbfgs_algorithm() {
        let method = Algorithm::NlLbfgs(nlopt_config());
        let (algo, cfg) = extract_nlopt_config(&method).expect("ok");
        assert_eq!(algo, NloptAlgorithm::Lbfgs);
        assert!(algo.uses_gradient(), "nl-lbfgs is the gradient algorithm");
        assert_eq!(cfg.chains, 4);
    }

    #[test]
    fn extract_nlopt_config_rejects_non_nlopt_method() {
        // PFilter is the simplest non-nlopt variant to construct.
        let method = Algorithm::PFilter {
            backend: InferenceBackend::ChainBinomial,
            particles: 100,
            replicates: None,
            record_ancestry: false,
            record_prequential: false,
            binomial: sim::rng::BinomialAlgorithm::Btpe,
        };
        let err = extract_nlopt_config(&method).expect_err("must reject");
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

    /// Both legs judge the same chains: the converged ones. A chain that ran
    /// out of evaluations stopped where its budget did, so its parameters say
    /// nothing about whether the converged chains found one optimum (the
    /// status count already reports it). The decibans leg used converged
    /// chains only and the chain-agreement leg used all of them.
    #[test]
    fn both_legs_judge_the_converged_chains_only() {
        let chains = vec![
            (0, outcome(-100.0, vec![0.50], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-100.1, vec![0.501], OptStatus::Converged(SuccessState::XtolReached))),
            (2, outcome(-400.0, vec![0.95], OptStatus::MaxEvalReached)),
        ];
        let bounds = vec![(0.0, 1.0)];
        let v = check_convergence(&chains, &names(1), &bounds, &GateConfig::default(), 1e-6);
        assert!(!v.chain_agreement_failed && v.max_rel_range < 0.01,
            "the max-eval chain is not part of the agreement: max_rel_range = {}", v.max_rel_range);
        assert!(!v.decibans_failed);
        assert_eq!(v.n_maxeval, 1, "the status count still reports it");
    }

    /// A leg that was not computed is not printed as passing.
    #[test]
    fn an_uncomputed_agreement_leg_says_so() {
        let chains = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let v = check_convergence(&chains, &names(1), &[(0.0, 1.0)], &GateConfig::default(), 1e-6);
        let line = chain_agreement_line(&crate::out::paint::Painter::plain(), &v);
        assert!(line.contains("not computed (fewer than two converged chains)")
            && !line.contains('✓'), "{line}");
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

    // ── chain_results.tsv, live and final ─────────────────────────────

    /// The format `chain_results.tsv` had when it was written once, after the
    /// last chain: header, then every chain's row in chain order. Kept here
    /// verbatim so the shared row formatter is checked against it, not
    /// against itself.
    fn chain_results_as_written_once(chains: &[(usize, ChainOutcome)], est: &[String]) -> String {
        let mut s = String::from("chain\tloglik\tstatus\tn_evals");
        for n in est { s.push_str(&format!("\t{n}")); }
        s.push('\n');
        for (i, c) in chains {
            s.push_str(&format!("{}\t{:.6}\t{}\t{}", ChainIndex(*i), c.loglik, c.status.as_str(), c.n_evals));
            for v in &c.params { s.push_str(&format!("\t{v:.10}")); }
            s.push('\n');
        }
        s
    }

    fn three_chains() -> Vec<(usize, ChainOutcome)> {
        vec![
            (0, outcome(-100.25, vec![0.5, 2.0], OptStatus::Converged(SuccessState::XtolReached))),
            (1, outcome(-110.0, vec![0.4, 1.5], OptStatus::MaxEvalReached)),
            (2, outcome(f64::NEG_INFINITY, vec![0.1, 3.0], OptStatus::Failed)),
        ]
    }

    /// A chain's row is on disk the moment it is appended, in completion
    /// order, and stays there if the run goes no further — the state a run
    /// killed after two of three chains leaves behind.
    #[test]
    fn finished_chains_are_on_disk_before_the_run_ends() {
        let dir = tempfile::tempdir().unwrap();
        let est = names(2);
        let chains = three_chains();
        let live = ChainResultsFile::create(&crate::fit::output::DirWriter::new(dir.path()), &est).unwrap();
        let path = dir.path().join(CHAIN_RESULTS_FILE);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), chain_results_header(&est),
            "the header is on disk before any chain finishes");

        live.append(2, &chains[2].1).unwrap();
        live.append(0, &chains[0].1).unwrap();
        // Read while the writer is still open, as a viewer would.
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "header + the two finished chains: {text:?}");
        assert!(lines[1].starts_with("2\t") && lines[2].starts_with("0\t"),
            "rows in completion order: {text:?}");
        assert!(text.ends_with('\n'), "every row is a complete line");
        drop(live); // the run dies here: no final rewrite
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, text, "the finished rows survive a run that never reaches the end");
    }

    /// The final rewrite puts every row in chain order, byte for byte as the
    /// file was when it was written once at the end, and leaves no sibling.
    #[test]
    fn the_finished_file_is_unchanged_from_the_write_once_format() {
        let dir = tempfile::tempdir().unwrap();
        let est = names(2);
        let chains = three_chains();
        let live = ChainResultsFile::create(&crate::fit::output::DirWriter::new(dir.path()), &est).unwrap();
        for i in [1, 2, 0] {
            live.append(i, &chains[i].1).unwrap();
        }
        drop(live);
        write_per_chain_files(&crate::fit::output::DirWriter::new(dir.path()), &chains, &est).unwrap();
        let text = std::fs::read_to_string(dir.path().join(CHAIN_RESULTS_FILE)).unwrap();
        assert_eq!(text, chain_results_as_written_once(&chains, &est));
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap()
            .map(|e| e.unwrap().file_name()).collect();
        assert_eq!(entries, vec![std::ffi::OsString::from(CHAIN_RESULTS_FILE)],
            "the rename leaves only chain_results.tsv");
    }

    /// Chains finish on different workers: rows appended concurrently are
    /// each whole, and none is lost.
    #[test]
    fn concurrent_appends_keep_every_row_whole() {
        let dir = tempfile::tempdir().unwrap();
        let est = names(6);
        let live = ChainResultsFile::create(&crate::fit::output::DirWriter::new(dir.path()), &est).unwrap();
        const N: usize = 64;
        std::thread::scope(|s| {
            for i in 0..N {
                let live = &live;
                let est = &est;
                s.spawn(move || {
                    let c = outcome(-(i as f64), vec![i as f64 / 7.0; est.len()],
                        OptStatus::Converged(SuccessState::FtolReached));
                    live.append(i, &c).unwrap();
                });
            }
        });
        let text = std::fs::read_to_string(dir.path().join(CHAIN_RESULTS_FILE)).unwrap();
        let rows: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(rows.len(), N);
        let mut seen: Vec<usize> = rows.iter().map(|r| {
            assert_eq!(r.split('\t').count(), 4 + est.len(), "malformed row {r:?}");
            r.split('\t').next().unwrap().parse().unwrap()
        }).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..N).collect::<Vec<_>>());
    }

    // ── every chain out of budget ────────────────────────────────────

    #[test]
    fn every_chain_out_of_budget_is_warned_about() {
        let chains = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::MaxEvalReached)),
            (1, outcome(-101.0, vec![0.6], OptStatus::MaxEvalReached)),
        ];
        let v = check_convergence(&chains, &names(1), &[(0.0, 1.0)], &GateConfig::default(), 1e-6);
        let w = all_maxeval_warning(&v, 2, 5000).expect("all chains hit max_evals");
        assert!(w.contains("all 2 chains") && w.contains("max_evals = 5000"), "{w}");

        let one = all_maxeval_warning(
            &check_convergence(&chains[..1], &names(1), &[(0.0, 1.0)], &GateConfig::default(), 1e-6),
            1, 300).unwrap();
        assert!(one.contains("the only chain"), "{one}");
    }

    #[test]
    fn one_converged_chain_is_no_all_out_of_budget_warning() {
        let chains = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::MaxEvalReached)),
            (1, outcome(-99.0, vec![0.6], OptStatus::Converged(SuccessState::XtolReached))),
        ];
        let v = check_convergence(&chains, &names(1), &[(0.0, 1.0)], &GateConfig::default(), 1e-6);
        assert_eq!(all_maxeval_warning(&v, 2, 5000), None);
        let failed = vec![
            (0, outcome(-100.0, vec![0.5], OptStatus::MaxEvalReached)),
            (1, outcome(f64::NEG_INFINITY, vec![0.6], OptStatus::Failed)),
        ];
        let v = check_convergence(&failed, &names(1), &[(0.0, 1.0)], &GateConfig::default(), 1e-6);
        assert_eq!(all_maxeval_warning(&v, 2, 5000), None,
            "a hard failure is a different finding, reported by the status line");
    }
}
