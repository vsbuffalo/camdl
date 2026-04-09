//! Particle Gibbs with Ancestor Sampling (PGAS) — Bayesian posterior
//! sampling via Gibbs sweeps alternating θ|X (exact MH) and X|θ,y
//! (conditional SMC with ancestor sampling).
//!
//! Lindsten, Jordan & Schön (2014). "Particle Gibbs with ancestor
//! sampling." JMLR 15:2145–2184.
//!
//! PGAS avoids the particle filter variance problem that plagues PMMH:
//! with the full trajectory X known, the complete-data log-likelihood
//! is exact (no estimation noise). The latent trajectory is refreshed
//! via CSMC-AS, which conditions on a reference trajectory and uses
//! ancestor sampling to maintain diversity.

use serde::{Serialize, Deserialize};

use crate::chain_binomial::{StepScratch, step_one, RATE_EPSILON};
use crate::compiled_model::CompiledModel;
use crate::rng::StatefulRng;
use crate::error::SimError;
use crate::inference::obs_loglik::{poisson_logpmf, binom_logpmf};
use crate::inference::particle_filter::Observation;
use crate::inference::resampling::systematic_resample;
use crate::inference::pmmh::Prior;
use crate::inference::if2::{EstimatedParam, Transform};
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::eval_resolved;
use crate::state::{IntState, RealState};

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

/// PGAS configuration.
pub struct PGASConfig {
    pub n_particles: usize,
    pub n_sweeps: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub dt: f64,
    /// Use NUTS (gradient-based) for the θ|X step instead of MH-within-Gibbs.
    /// Requires rate_grad expressions in the IR (compiled with autodiff).
    /// Falls back to MH if gradients are not available.
    pub use_nuts: bool,
    /// Use dense (full covariance) mass matrix for NUTS. Default: true.
    /// Dense handles parameter correlations (e.g., R0-amplitude ridge).
    /// Set false for diagonal-only (handles scale but not correlations).
    pub dense_mass: bool,
    /// Temperature ladder for parallel tempering (replica exchange).
    /// Each entry is a β value in (0, 1]. The first entry MUST be 1.0
    /// (cold chain). Default: `[1.0]` (no tempering, single rung).
    /// Example: `[1.0, 0.7, 0.4, 0.15]` runs 4 temperature rungs.
    /// Only the cold (β=1) rung contributes posterior samples and trace output.
    /// Heated rungs explore a flatter likelihood surface (LL scaled by β)
    /// and exchange with adjacent rungs via Metropolis swap proposals.
    pub tempering: Vec<f64>,
}

/// Per-substep record: minimal information for transition density
/// evaluation and trajectory reconstruction.
#[derive(Clone, Serialize, Deserialize)]
pub struct SubstepRecord {
    /// Compartment counts BEFORE this substep — the exact snapshot that
    /// step_one evaluated propensities from. The density MUST use this
    /// (not the previous substep's post-clamp counts) to avoid the
    /// clamping mismatch where n_exit > n_src_clamped.
    pub counts_before: Vec<i64>,
    /// Compartment counts AFTER this substep (post-clamp, post-intervention).
    /// Used as input to the NEXT substep's step_one.
    pub counts_after: Vec<i64>,
    /// Per-transition flow counts FOR THIS SUBSTEP ONLY.
    pub flows: Vec<u64>,
    /// Gamma multipliers used at this substep (one per overdispersed
    /// source group, in source_groups order). Empty if no overdispersion.
    pub gammas: Vec<f64>,
}

/// Full trajectory stored at substep resolution.
#[derive(Clone, Serialize, Deserialize)]
pub struct PGASTrajectory {
    /// Compartment counts at simulation start (before any substep).
    pub initial_counts: Vec<i64>,
    /// One record per substep, ordered chronologically.
    pub substeps: Vec<SubstepRecord>,
}

/// Mapping from an IVP parameter to the compartment it controls.
/// Used to make the initial state stochastic in CSMC-AS and to add
/// the initial state density to the complete-data LL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IVPMapping {
    /// Index into if2_params / priors vectors.
    pub param_idx: usize,
    /// Index into the model's param vector (if2_params[param_idx].index).
    pub model_param_idx: usize,
    /// Which compartment this IVP controls (local int index).
    pub compartment_idx: usize,
}

/// Diagnostics from one CSMC-AS sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CSMCDiagnostics {
    /// Fraction of traceback substeps from non-reference particles.
    /// Near 0% = path degeneracy (reference never replaced, CSMC broken).
    /// Near 50%+ = healthy trajectory renewal.
    pub trajectory_renewal: f64,
    /// Number of substeps where all ancestor weights were -inf.
    pub n_degenerate: usize,
    /// Total substeps.
    pub n_substeps: usize,
}

/// Result of one Gibbs sweep.
#[derive(Clone, Serialize, Deserialize)]
pub struct PGASSweep {
    pub params: Vec<f64>,
    pub log_complete_data_ll: f64,
    pub accepted: Vec<bool>,
    pub csmc_diag: CSMCDiagnostics,
    pub proposal_sds: Vec<f64>,
}

/// Full PGAS result.
pub struct PGASResult {
    pub sweeps: Vec<PGASSweep>,
    pub final_trajectory: PGASTrajectory,
    pub acceptance_rates: Vec<f64>,
    /// Resume state for chain continuation. Populated at end of every run.
    pub resume_state: ChainResumeState,
}

/// Serializable chain state for `--resume`. Saved to `chain_N/resume_state.bin`
/// via bincode at end of every PGAS run, enabling continuation without
/// re-doing burn-in or mass matrix adaptation.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChainResumeState {
    /// Config hash — only resume if the statistical problem matches.
    pub config_hash: String,
    /// Number of sweeps completed (resume starts from here).
    pub completed_sweeps: usize,
    /// Current parameter values (natural scale, full model param vector).
    pub params: Vec<f64>,
    /// Current transformed parameters (z-scale for NUTS).
    pub transformed: Vec<f64>,
    /// Reference trajectory from the last CSMC sweep.
    pub trajectory: PGASTrajectory,
    /// Adapted mass matrix (NUTS).
    pub mass_matrix: super::nuts::MassMatrix,
    /// Adapted step size (NUTS).
    pub nuts_step_size: f64,
    /// Adapted proposal SDs on log scale (MH-within-Gibbs).
    pub log_proposal_sd: Vec<f64>,
    /// Running acceptance counts per parameter.
    pub total_accepted: Vec<usize>,
    /// Current complete-data log-likelihood.
    pub current_ll: f64,
    /// Estimated parameter names in the same order as `transformed`.
    /// Used to match z-values to the correct parameters on resume,
    /// since HashMap iteration order is non-deterministic.
    /// Empty for legacy states (before this field was added).
    pub param_names: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Transition density
// ═══════════════════════════════════════════════════════════════════

/// Log transition density for ONE substep, mirroring step_one's
/// Euler-multinomial decomposition exactly.
///
/// Evaluates log p(flows | counts_before, params, gammas, t, dt).
///
/// CRITICAL: This must use the SAME rate computation, source grouping,
/// and split ordering as step_one. If this function computes p_split
/// differently from how step_one drew the split, ancestor weights will
/// be wrong and the sampler will degenerate silently.
pub fn log_transition_density_substep(
    model: &CompiledModel,
    counts_before: &[i64],
    flows: &[u64],
    gammas: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
) -> Result<f64, SimError> {
    let n_int = model.int_local_to_global.len();
    let n_tr = model.model.transitions.len();

    // Set up evaluation context (same as step_one)
    let mut int_s = IntState::new(n_int);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = RealState::new(model.real_local_to_global.len());

    let mut propensities = vec![0.0; n_tr];
    eval_propensities(model, &int_s, &real_s, params, t, &mut propensities)?;

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, projected: None,
    };

    // Per-transition: is it deterministic? What's its sigma_sq?
    let mut is_determ = vec![false; n_tr];
    let mut sigma_sq_by_tr: Vec<Option<f64>> = vec![None; n_tr];
    for (i, tr) in model.model.transitions.iter().enumerate() {
        match &tr.draw_method {
            ir::transition::DrawMethod::Deterministic => { is_determ[i] = true; }
            ir::transition::DrawMethod::Overdispersed(_) => {
                sigma_sq_by_tr[i] = Some(eval_resolved(model.resolved.overdispersion[i].as_ref().unwrap(), &ctx));
            }
            _ => {}
        }
    }

    let mut log_p = 0.0;
    let mut handled = vec![false; n_tr];
    let mut gamma_idx = 0;

    // Source-grouped transitions (mirrors step_one's Euler-multinomial)
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts_before[src_local].max(0);
        if n_src == 0 {
            for &tr_idx in group {
                if flows[tr_idx] > 0 { return Ok(f64::NEG_INFINITY); }
                handled[tr_idx] = true;
            }
            continue;
        }

        // Step 1: compute effective per-capita rates (same as step_one)
        let mut probs: Vec<(usize, f64)> = Vec::new();
        let mut total_rate = 0.0_f64;
        let mut gamma_was_used = false;  // track if gamma was actually drawn
        for &tr_idx in group {
            let rate = propensities[tr_idx];
            // Mirror step_one's logic exactly: check rate first, then deterministic.
            if rate <= RATE_EPSILON {
                if flows[tr_idx] > 0 && rate <= 0.0 {
                    // Truly zero rate with nonzero flow — model needs iota.
                    log::warn!(
                        "transition '{}' has rate=0 but flow={}. \
                         Add a seeding term (iota) to the rate expression: \
                         e.g., beta * (I + iota) / N * S.",
                        model.model.transitions[tr_idx].name, flows[tr_idx],
                    );
                    return Ok(f64::NEG_INFINITY);
                } else if flows[tr_idx] > 0 {
                    // Near-zero rate (0 < rate ≤ RATE_EPSILON) with nonzero flow.
                    // Include in multinomial with its tiny rate rather than -inf.
                    let per_capita = rate / n_src as f64;
                    total_rate += per_capita;
                    probs.push((tr_idx, per_capita));
                    continue;
                }
                handled[tr_idx] = true;
                continue;
            }
            if is_determ[tr_idx] {
                // Deterministic transitions are not part of the multinomial.
                // step_one handles them separately (exact count = rate * dt).
                handled[tr_idx] = true;
                continue;
            }
            let per_capita = rate / n_src as f64;
            let effective = if let Some(_sigma_sq) = sigma_sq_by_tr[tr_idx] {
                gamma_was_used = true;
                let g = if gamma_idx < gammas.len() { gammas[gamma_idx] } else { 1.0 };
                per_capita * g
            } else {
                per_capita
            };
            total_rate += effective;
            probs.push((tr_idx, effective));
        }
        // Only advance gamma_idx if the overdispersed transition was actually
        // evaluated (rate > 0). step_one only pushes to gamma_used when the
        // Overdispersed transition has positive rate and enters the split.
        // If it was skipped (rate=0), no gamma was drawn — gamma_idx must NOT
        // advance, or all subsequent groups read the wrong gamma.
        if gamma_was_used {
            gamma_idx += 1;
        }

        if total_rate <= RATE_EPSILON || probs.is_empty() { continue; }

        // Step 2: evaluate total exits density
        let p_total = (1.0 - (-total_rate * dt).exp()).clamp(1e-15, 1.0 - 1e-15);
        let n_exit: u64 = probs.iter().map(|&(tr_idx, _)| flows[tr_idx]).sum();
        let binom_total = binom_logpmf(n_exit, n_src as u64, p_total);

        if !binom_total.is_finite() {
            log::debug!("density: total exits -inf: Binom({}, {}, {:.6e}), src_comp_idx={}",
                n_exit, n_src, p_total, src_local);
            return Ok(f64::NEG_INFINITY);
        }

        log_p += binom_total;

        // Step 3: evaluate split density (mirrors step_one's proportional split)
        let n_competing = probs.len();
        let mut remaining = n_exit;
        let mut rate_remaining = total_rate;
        for (k, &(tr_idx, eff_rate)) in probs.iter().enumerate() {
            handled[tr_idx] = true;
            if k == n_competing - 1 {
                if flows[tr_idx] != remaining {
                    return Ok(f64::NEG_INFINITY);
                }
            } else if remaining > 0 && rate_remaining > 0.0 {
                let p_split = (eff_rate / rate_remaining).clamp(1e-15, 1.0 - 1e-15);
                log_p += binom_logpmf(flows[tr_idx], remaining, p_split);
                remaining -= flows[tr_idx];
                rate_remaining -= eff_rate;
            } else if flows[tr_idx] > 0 {
                return Ok(f64::NEG_INFINITY);
            }
        }
    }

    // Ungrouped / inflow transitions
    for (i, &rate) in propensities.iter().enumerate() {
        if handled[i] || rate <= RATE_EPSILON { continue; }
        let mean = rate * dt;
        if is_determ[i] {
            if flows[i] != mean.round() as u64 {
                return Ok(f64::NEG_INFINITY);
            }
        } else {
            // Poisson density (or overdispersed — approximate as Poisson
            // since ungrouped overdispersed transitions are rare)
            log_p += poisson_logpmf(flows[i] as f64, mean);
        }
    }

    Ok(log_p)
}

/// Complete-data log-likelihood: sum of transition densities + observation
/// densities over the full trajectory.
///
/// log p(y, X | θ) = log p(x₀ | θ)
///                 + Σ_s log p(x_s | x_{s-1}, θ, g_s)
///                 + Σ_k log p(y_k | project(x_{obs_k}), θ)
///
/// The initial state density log p(x₀ | θ) is included for IVP parameters
/// (e.g., S₀ ~ Binom(N₀, s0)). Without it, IVPs are invisible to the MH step.
pub fn complete_data_loglik(
    model: &CompiledModel,
    trajectory: &PGASTrajectory,
    params: &[f64],
    observations: &[Observation],
    dt: f64,
    obs_streams: &[super::types::ObsStreamSpec],
    ivp_mappings: &[IVPMapping],
) -> Result<f64, SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = trajectory.substeps.len();
    let n_tr = model.model.transitions.len();
    let mut log_p = 0.0;

    // Initial state density: log p(x₀ | θ) for IVP-controlled compartments.
    // S₀ ~ Binom(N₀, s0) → log Binom(S₀; N₀, s0) constrains s0.
    // N₀ is the total population of the PATCH containing this compartment,
    // not the global population across all patches. We compute it as the
    // sum of initial counts in the same stratification group.
    if !ivp_mappings.is_empty() {
        // For each IVP compartment, find the patch population by summing
        // all compartments that share the same patch suffix. For unstratified
        // models, this is just the total population.
        let total_pop: i64 = trajectory.initial_counts.iter().sum();
        for ivp in ivp_mappings {
            let count = trajectory.initial_counts[ivp.compartment_idx] as u64;
            let frac = params[ivp.model_param_idx].clamp(1e-10, 1.0 - 1e-10);
            // Determine the per-patch population for this IVP compartment.
            // In a stratified model, only compartments in the same patch
            // contribute to N₀. Detect by matching the patch suffix.
            let comp_name = &model.model.compartments[ivp.compartment_idx].name;
            let patch_suffix = comp_name.rsplit('_').next().unwrap_or("");
            let patch_pop: i64 = if patch_suffix.is_empty() || !comp_name.contains('_') {
                // Unstratified: use total population
                total_pop
            } else {
                // Stratified: sum all compartments with matching patch suffix
                model.model.compartments.iter().enumerate()
                    .filter(|(_, c)| c.name.ends_with(&format!("_{}", patch_suffix)))
                    .map(|(i, _)| trajectory.initial_counts[i])
                    .sum()
            };
            let ivp_ll = binom_logpmf(count, patch_pop as u64, frac);
            if !ivp_ll.is_finite() {
                eprintln!("  IVP density -inf: Binom({}, {}, {:.6e}) for {} (comp={}, patch_pop={})",
                    count, patch_pop, frac,
                    comp_name, ivp.compartment_idx, patch_pop);
            }
            log_p += ivp_ll;
        }
    }

    if !log_p.is_finite() {
        log::debug!("complete_data_loglik: -inf after IVP density (log_p={:.1})", log_p);
        return Ok(f64::NEG_INFINITY);
    }

    // Precompute observation substep indices
    let mut obs_at_substep = std::collections::HashMap::new();
    for (obs_idx, obs) in observations.iter().enumerate() {
        let s = ((obs.time - t_start) / dt).round() as usize;
        if s > 0 { obs_at_substep.insert(s - 1, obs_idx); }
    }

    // Cumulative flows since last observation (for projection)
    let mut cum_flows = vec![0u64; n_tr];

    for s in 0..n_substeps {
        let t = t_start + s as f64 * dt;
        // Use the pre-step snapshot stored in the record — this is the
        // exact state step_one evaluated propensities from.
        let counts_before = &trajectory.substeps[s].counts_before;
        let rec = &trajectory.substeps[s];

        // Transition density
        let td = log_transition_density_substep(
            model, counts_before, &rec.flows, &rec.gammas, params, t, dt,
        )?;
        if !td.is_finite() {
            log::debug!("complete_data_loglik: -inf transition density at substep {} (t={:.1})", s, t);
            return Ok(f64::NEG_INFINITY);
        }
        log_p += td;

        // TODO: gamma multiplier density (log Gamma(g; dt/σ², σ²/dt)) is
        // disabled. The gamma index tracking between step_one and the density
        // doesn't align for models with zero-rate overdispersed transitions.
        // The transition density already constrains σ² through p_total, so
        // this is not blocking. See incident report:
        // docs/dev/incidents/2026-04-07-spatial-pgas-neg-inf.md

        // Accumulate flows
        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }

        // Observation density — joint across all streams
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            let obs_ll = super::types::joint_obs_weight(obs_streams, &cum_flows, obs_idx);
            if !obs_ll.is_finite() {
                log::debug!("complete_data_loglik: obs density -inf at substep {} (obs_idx={})", s, obs_idx);
            }
            log_p += obs_ll;
            if !log_p.is_finite() {
                log::debug!("complete_data_loglik: -inf after obs at substep {} (cumulative)", s);
                return Ok(f64::NEG_INFINITY);
            }
            for f in &mut cum_flows { *f = 0; }
        }
    }

    Ok(log_p)
}

// ═══════════════════════════════════════════════════════════════════
// Forward simulation (initial trajectory)
// ═══════════════════════════════════════════════════════════════════

/// Simulate a forward trajectory recording per-substep detail.
/// Used to initialize the reference trajectory for PGAS.
pub fn simulate_reference(
    model: &CompiledModel,
    params: &[f64],
    t_end: f64,
    dt: f64,
    rng: &mut StatefulRng,
) -> Result<PGASTrajectory, SimError> {
    let (init_int, _) = model.initial_state(params)?;
    let n_tr = model.model.transitions.len();
    let t_start = model.model.simulation.t_start;
    let n_substeps = ((t_end - t_start) / dt).round() as usize;

    let mut counts = init_int.counts.clone();
    let mut scratch = StepScratch::new(model);
    let mut substeps = Vec::with_capacity(n_substeps);

    for s in 0..n_substeps {
        let t = t_start + s as f64 * dt;
        let mut flows = vec![0u64; n_tr];
        scratch.gamma_used.clear();

        let counts_before = counts.clone();
        step_one(model, &mut counts, &mut flows, params, t, dt, rng, &mut scratch)?;

        // Verify: density evaluation of this record won't produce k > n.
        // This catches state/flow mismatches before they cause -inf later.
        if cfg!(debug_assertions) {
            let verify_td = log_transition_density_substep(
                model, &counts_before, &flows, &scratch.gamma_used, params, t, dt,
            );
            if let Ok(td) = verify_td {
                debug_assert!(td.is_finite(),
                    "simulate_reference: density is -inf at substep {} (t={:.1}) \
                     despite matching state. counts_before={:?}, flows={:?}",
                    s, t, &counts_before, &flows);
            }
        }

        substeps.push(SubstepRecord {
            counts_before,
            counts_after: counts.clone(),
            flows,
            gammas: scratch.gamma_used.clone(),
        });
    }

    Ok(PGASTrajectory {
        initial_counts: init_int.counts,
        substeps,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Conditional SMC with Ancestor Sampling (CSMC-AS)
// ═══════════════════════════════════════════════════════════════════

/// Run one CSMC-AS sweep: draw X' ~ p(X | θ, y) conditioned on
/// the reference trajectory.
///
/// Returns a new trajectory + diagnostics.
pub fn csmc_as(
    model: &CompiledModel,
    params: &[f64],
    observations: &[Observation],
    reference: &PGASTrajectory,
    n_particles: usize,
    dt: f64,
    obs_streams: &[super::types::ObsStreamSpec],
    ivp_mappings: &[IVPMapping],
    seed: u64,
) -> Result<(PGASTrajectory, CSMCDiagnostics), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = reference.substeps.len();
    let n_tr = model.model.transitions.len();
    let j_ref = n_particles - 1; // reference particle is the last slot

    // Precompute observation substep indices
    let mut obs_at_substep = std::collections::HashMap::new();
    for (obs_idx, obs) in observations.iter().enumerate() {
        let s = ((obs.time - t_start) / dt).round() as usize;
        if s > 0 { obs_at_substep.insert(s - 1, obs_idx); }
    }

    // Initialize particles with stochastic initial states for IVP compartments.
    // Each free particle draws S₀ ~ Binom(N₀, s0) independently, giving the
    // CSMC diverse initial states to select among. This is what enables
    // posterior sampling of IVP parameters like s0.
    let (init_int, _) = model.initial_state(params)?;
    let total_pop = init_int.counts.iter().sum::<i64>();

    // Precompute per-IVP patch populations (for stratified models, N₀ is the
    // patch population, not the global population).
    let ivp_patch_pops: Vec<i64> = ivp_mappings.iter().map(|ivp| {
        let comp_name = &model.model.compartments[ivp.compartment_idx].name;
        let patch_suffix = comp_name.rsplit('_').next().unwrap_or("");
        if patch_suffix.is_empty() || !comp_name.contains('_') {
            total_pop
        } else {
            model.model.compartments.iter().enumerate()
                .filter(|(_, c)| c.name.ends_with(&format!("_{}", patch_suffix)))
                .map(|(i, _)| init_int.counts[i])
                .sum()
        }
    }).collect();

    // Per-particle RNGs (needed early for stochastic init)
    let mut rngs: Vec<StatefulRng> = (0..n_particles)
        .map(|i| StatefulRng::new(seed ^ (i as u64).wrapping_mul(0x517cc1b727220a95)))
        .collect();

    let mut counts: Vec<Vec<i64>> = (0..n_particles)
        .map(|j| {
            if j == j_ref {
                reference.initial_counts.clone()
            } else {
                let mut c = init_int.counts.clone();
                // Draw stochastic initial state for IVP compartments
                for (k, ivp) in ivp_mappings.iter().enumerate() {
                    let frac = params[ivp.model_param_idx].clamp(1e-10, 1.0 - 1e-10);
                    let patch_n = ivp_patch_pops[k] as u64;
                    c[ivp.compartment_idx] = rngs[j].binomial(patch_n, frac) as i64;
                }
                // Reapply balance constraint if present
                if let Some(ref bal) = model.balance {
                    let bal_val: i64 = total_pop - c.iter().enumerate()
                        .filter(|&(i, _)| i != bal.local_int_idx)
                        .map(|(_, &v)| v)
                        .sum::<i64>();
                    c[bal.local_int_idx] = bal_val;
                }
                c
            }
        })
        .collect();

    // Per-particle per-substep flows (reset each substep)
    let mut substep_flows: Vec<Vec<u64>> = (0..n_particles)
        .map(|_| vec![0u64; n_tr])
        .collect();
    let mut substep_gammas: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| Vec::new())
        .collect();

    // Cumulative flows since last observation (for projection)
    let mut cum_flows: Vec<Vec<u64>> = (0..n_particles)
        .map(|_| vec![0u64; n_tr])
        .collect();

    // Store initial counts per particle BEFORE propagation (for traceback).
    // Needed because free particles have stochastic initial states (Binom draw)
    // that differ from the deterministic initial_state(params).
    let initial_counts_per_particle: Vec<Vec<i64>> = counts.iter()
        .map(|c| c.clone())
        .collect();

    // History for traceback
    let mut history_counts_before: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_counts_after: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_flows: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_substeps);
    let mut history_gammas: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_substeps);
    let mut ancestors: Vec<Vec<usize>> = Vec::with_capacity(n_substeps);

    // Weights (log-space)
    let mut log_weights = vec![0.0f64; n_particles];

    // Resampling RNG (particle RNGs already created above for stochastic init)
    let mut resample_rng = StatefulRng::new(seed.wrapping_add(0xdeadbeef));

    // Per-particle scratch buffers
    let mut scratches: Vec<StepScratch> = (0..n_particles)
        .map(|_| StepScratch::new(model))
        .collect();

    // Previous states (for ancestor sampling: need state before propagation)
    let mut prev_counts: Vec<Vec<i64>> = counts.clone();

    // Diagnostic: count substeps where ancestor sampling is degenerate
    // (no particle can reach the reference state → reference stays self-connected)
    let mut n_degenerate: usize = 0;

    for s in 0..n_substeps {
        let t = t_start + s as f64 * dt;

        // ── 1. Resample free particles (ancestor selection from prev weights) ──
        // On non-observation substeps, weights are uniform → systematic
        // resampling is identity. Skip resampling in that case.
        let substep_ancestors: Vec<usize>;
        let weights_are_uniform = log_weights.iter().all(|&w| (w - log_weights[0]).abs() < 1e-10);

        if weights_are_uniform {
            // Identity: each particle is its own ancestor
            substep_ancestors = (0..n_particles).collect();
        } else {
            // Resample from previous weights
            let indices = systematic_resample(&log_weights, &mut resample_rng);
            // Apply resampling to free particles (not reference)
            let mut new_counts = Vec::with_capacity(n_particles);
            let mut new_cum_flows = Vec::with_capacity(n_particles);
            for j in 0..n_particles {
                if j == j_ref {
                    new_counts.push(counts[j_ref].clone());
                    new_cum_flows.push(cum_flows[j_ref].clone());
                } else {
                    new_counts.push(counts[indices[j]].clone());
                    new_cum_flows.push(cum_flows[indices[j]].clone());
                }
            }
            counts = new_counts;
            cum_flows = new_cum_flows;
            substep_ancestors = indices;
        }

        // Save pre-propagation states for ancestor sampling
        for j in 0..n_particles {
            prev_counts[j].copy_from_slice(&counts[j]);
        }

        // ── 2. Propagate free particles ──
        for j in 0..n_particles {
            if j == j_ref { continue; }
            // Reset substep flows
            for f in &mut substep_flows[j] { *f = 0; }
            scratches[j].gamma_used.clear();

            step_one(
                model, &mut counts[j], &mut substep_flows[j],
                params, t, dt, &mut rngs[j], &mut scratches[j],
            )?;

            substep_gammas[j] = scratches[j].gamma_used.clone();
        }

        // ── 3. Clamp reference particle ──
        let ref_rec = &reference.substeps[s];
        counts[j_ref].copy_from_slice(&ref_rec.counts_after);
        substep_flows[j_ref].copy_from_slice(&ref_rec.flows);
        substep_gammas[j_ref] = ref_rec.gammas.clone();
        // Fix: prev_counts[j_ref] was saved at step 2 from the post-resample
        // state (which could be any particle's state). But ref_rec.flows were
        // drawn from ref_rec.counts_before. The history must pair the correct
        // counts_before with the reference's flows, otherwise the traceback
        // produces Binom(k; n, p) with k > n.
        prev_counts[j_ref].copy_from_slice(&ref_rec.counts_before);

        // ── 4. Ancestor sampling for reference particle ──
        // ã_j = w_{s-1}^j + log f(X_ref_s | x_{s-1}^j, θ, gamma_ref_s)
        // The gamma from the reference is used because we're asking:
        // "given this gamma noise, what's P(reaching ref state from particle j?)"
        {
            let mut ancestor_log_w = vec![f64::NEG_INFINITY; n_particles];
            for j in 0..n_particles {
                let td = log_transition_density_substep(
                    model,
                    &prev_counts[j],
                    &ref_rec.flows,
                    &ref_rec.gammas,
                    params,
                    t,
                    dt,
                )?;
                ancestor_log_w[j] = log_weights[j] + td;
            }

            // Sample from categorical(softmax(ancestor_log_w))
            let max_w = ancestor_log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ref_ancestor = if max_w.is_finite() {
                let weights: Vec<f64> = ancestor_log_w.iter()
                    .map(|&w| (w - max_w).exp())
                    .collect();
                let sum: f64 = weights.iter().sum();
                if sum > 0.0 {
                    let u = resample_rng.uniform() * sum;
                    let mut cum = 0.0;
                    let mut selected = 0;
                    for (j, &w) in weights.iter().enumerate() {
                        cum += w;
                        if cum >= u { selected = j; break; }
                    }
                    selected
                } else {
                    // Degenerate: no particle can reach the reference state.
                    // Keep the reference's own history to maintain internal
                    // consistency (the reference's flows at substep s were
                    // produced from the reference's state at substep s-1).
                    // Picking a random particle here causes splice-point
                    // inconsistencies → -inf in complete_data_loglik.
                    n_degenerate += 1;
                    j_ref
                }
            } else {
                // All ancestor weights are -inf: reference unreachable
                n_degenerate += 1;
                j_ref
            };

            // Record ancestor for reference particle
            let mut step_ancestors = substep_ancestors;
            step_ancestors[j_ref] = ref_ancestor;
            ancestors.push(step_ancestors);
        }

        // Accumulate cumulative flows
        for j in 0..n_particles {
            for (i, &f) in substep_flows[j].iter().enumerate() {
                cum_flows[j][i] += f;
            }
        }

        // ── 5. Compute weights — joint across all streams ──
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            for j in 0..n_particles {
                log_weights[j] = super::types::joint_obs_weight(
                    obs_streams, &cum_flows[j], obs_idx);
            }
            for j in 0..n_particles {
                for f in &mut cum_flows[j] { *f = 0; }
            }
        } else {
            // Non-observation substep: uniform weights
            for w in &mut log_weights { *w = 0.0; }
        }

        // ── 6. Store history ──
        history_counts_before.push(prev_counts.iter().map(|c| c.clone()).collect());
        history_counts_after.push(counts.iter().map(|c| c.clone()).collect());
        history_flows.push(substep_flows.iter().map(|f| f.clone()).collect());
        history_gammas.push(substep_gammas.iter().map(|g| g.clone()).collect());
    }

    // Diagnostic: warn if many substeps had degenerate ancestor sampling
    if n_degenerate > 0 {
        let pct = n_degenerate as f64 / n_substeps as f64 * 100.0;
        if pct > 10.0 {
            log::warn!("CSMC-AS: {}/{} substeps ({:.0}%) had degenerate ancestor sampling — \
                        reference trajectory is too far from particle cloud. \
                        Consider more particles or smaller parameter proposals.",
                        n_degenerate, n_substeps, pct);
        }
    }

    // ── Select final trajectory ──
    // Sample from final weights
    let k = {
        let max_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_w.is_finite() {
            let weights: Vec<f64> = log_weights.iter()
                .map(|&w| (w - max_w).exp())
                .collect();
            let sum: f64 = weights.iter().sum();
            let u = resample_rng.uniform() * sum;
            let mut cum = 0.0;
            let mut selected = 0;
            for (j, &w) in weights.iter().enumerate() {
                cum += w;
                if cum >= u { selected = j; break; }
            }
            selected
        } else {
            j_ref // fallback to reference
        }
    };

    // Trace back through ancestry and compute trajectory renewal
    let mut trajectory_substeps = Vec::with_capacity(n_substeps);
    let mut particle = k;
    let mut n_from_ref = 0usize;
    for s in (0..n_substeps).rev() {
        if particle == j_ref { n_from_ref += 1; }
        trajectory_substeps.push(SubstepRecord {
            counts_before: history_counts_before[s][particle].clone(),
            counts_after: history_counts_after[s][particle].clone(),
            flows: history_flows[s][particle].clone(),
            gammas: history_gammas[s][particle].clone(),
        });
        particle = ancestors[s][particle];
    }
    trajectory_substeps.reverse();

    // Verify: density evaluation of each traceback record is finite.
    if cfg!(debug_assertions) {
        for (s, rec) in trajectory_substeps.iter().enumerate() {
            let t = t_start + s as f64 * dt;
            let verify_td = log_transition_density_substep(
                model, &rec.counts_before, &rec.flows, &rec.gammas, params, t, dt,
            );
            if let Ok(td) = verify_td {
                debug_assert!(td.is_finite(),
                    "csmc_as traceback: density is -inf at substep {} (t={:.1}) \
                     counts_before={:?}, flows={:?}",
                    s, t, &rec.counts_before, &rec.flows);
            }
        }
    }

    let trajectory_renewal = 1.0 - n_from_ref as f64 / n_substeps as f64;

    // Initial counts: use the stored per-particle initial state (which
    // includes stochastic Binom draws for IVP compartments).
    let initial_counts = initial_counts_per_particle[particle].clone();

    let diag = CSMCDiagnostics {
        trajectory_renewal,
        n_degenerate,
        n_substeps,
    };

    Ok((PGASTrajectory {
        initial_counts,
        substeps: trajectory_substeps,
    }, diag))
}

/// Derivative of the transform θ(z) w.r.t. z.
/// dθ/dz for chain rule: d(f(θ))/dz = d(f)/dθ × dθ/dz.
fn transform_deriv(param: &EstimatedParam, z: f64) -> f64 {
    match &param.transform {
        Transform::Log { .. } => z.exp(), // θ = exp(z), dθ/dz = exp(z)
        Transform::Logit { lo, hi } => {
            let p = 1.0 / (1.0 + (-z).exp());
            (hi - lo) * p * (1.0 - p) // θ = lo + (hi-lo)*σ(z), dθ/dz = (hi-lo)*σ(z)*(1-σ(z))
        }
        Transform::None => 1.0,
    }
}

/// Prior log-density AND its gradient on the z (unconstrained) scale.
/// Each variant handles its own chain rule — the caller just sums.
fn prior_log_density_and_grad_z(
    prior: &Prior, param: &EstimatedParam, theta: f64, z: f64,
) -> (f64, f64) {
    match prior {
        Prior::Flat => (0.0, 0.0),
        Prior::Normal { mean, sd } => {
            let lp = -0.5 * ((theta - mean) / sd).powi(2) - sd.ln();
            let dlp_dtheta = -(theta - mean) / (sd * sd);
            let dlp_dz = dlp_dtheta * transform_deriv(param, z);
            (lp, dlp_dz)
        }
        Prior::TransformedNormal { mean, sd } => {
            let lp = -0.5 * ((z - mean) / sd).powi(2) - sd.ln();
            let dlp_dz = -(z - mean) / (sd * sd);
            (lp, dlp_dz)
        }
        Prior::Beta { alpha, beta } => {
            // log Beta(θ; a, b) = (a-1)ln(θ) + (b-1)ln(1-θ) - lnB(a,b)
            // d/dθ = (a-1)/θ - (b-1)/(1-θ)
            // d/dz = d/dθ × dθ/dz
            if theta <= 0.0 || theta >= 1.0 { return (f64::NEG_INFINITY, 0.0); }
            use crate::inference::obs_loglik::lgamma;
            let lp = (alpha - 1.0) * theta.ln() + (beta - 1.0) * (1.0 - theta).ln()
                - (lgamma(*alpha) + lgamma(*beta) - lgamma(alpha + beta));
            let dlp_dtheta = (alpha - 1.0) / theta - (beta - 1.0) / (1.0 - theta);
            let dlp_dz = dlp_dtheta * transform_deriv(param, z);
            (lp, dlp_dz)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main PGAS loop
// ═══════════════════════════════════════════════════════════════════

/// Run the PGAS Gibbs sampler.
///
/// Alternates between:
/// 1. θ | X, y — MH updates using exact complete-data log-likelihood
/// 2. X | θ, y — CSMC-AS to refresh the latent trajectory
///
/// Step 1 evaluates the exact log p(y,X|θ) — no PF, no estimation noise.
/// The surface is sharp (46K transition terms), so proposals are small, but
/// the CSMC-AS in Step 2 shifts the mode by renewing the trajectory X. The
/// Gibbs alternation provides mixing: small θ steps track the shifting mode.
pub fn run_pgas(
    model: &CompiledModel,
    if2_params: &[EstimatedParam],
    priors: &[Prior],
    base_params: &[f64],
    config: &PGASConfig,
    observations: &[Observation],
    obs_streams: &[super::types::ObsStreamSpec],
    seed: u64,
    on_sweep: Option<&dyn Fn(usize, &PGASSweep, &PGASTrajectory)>,
    resume_from: Option<ChainResumeState>,
    config_hash: String,
) -> Result<PGASResult, SimError> {
    let d = if2_params.len();
    assert_eq!(d, priors.len(), "priors must match if2_params length");

    let mut rng = StatefulRng::new(seed);
    let mut current_params = base_params.to_vec();
    let t_end = observations.last().map_or(
        model.model.simulation.t_start,
        |o| o.time,
    );

    // Resume or fresh start
    let start_sweep;
    let trajectory;
    let current_transformed: Vec<f64>;

    // Extract resume adaptation state (consumed separately from trajectory/params)
    let resume_nuts = resume_from.as_ref().map(|s| (
        s.mass_matrix.clone(), s.nuts_step_size,
        s.log_proposal_sd.clone(), s.total_accepted.clone(), s.current_ll,
    ));

    if let Some(state) = resume_from {
        eprintln!("  resuming from sweep {}...", state.completed_sweeps);
        current_params.copy_from_slice(&state.params);
        trajectory = state.trajectory;
        start_sweep = state.completed_sweeps;

        // Restore z-values with name-based reordering.
        // The resume state stores param_names alongside transformed values.
        // HashMap iteration order is non-deterministic, so the current run's
        // if2_params may be in a different order than when the state was saved.
        if !state.param_names.is_empty() && state.param_names.len() == state.transformed.len() {
            // Build name→z lookup from saved state
            let saved_z: std::collections::HashMap<&str, f64> = state.param_names.iter()
                .zip(state.transformed.iter())
                .map(|(name, &z)| (name.as_str(), z))
                .collect();

            current_transformed = if2_params.iter().map(|spec| {
                if let Some(&z) = saved_z.get(spec.name.as_str()) {
                    z
                } else {
                    // Param not in saved state — compute from theta
                    eprintln!("  warning: param '{}' not found in resume state, computing from theta", spec.name);
                    spec.to_transformed(current_params[spec.index])
                }
            }).collect();
        } else {
            // Legacy resume state without param_names — recompute from params.
            // This is the safe fallback: z = to_transformed(theta).
            eprintln!("  warning: resume state lacks param_names — recomputing z from params.");
            current_transformed = if2_params.iter()
                .map(|spec| spec.to_transformed(current_params[spec.index]))
                .collect();
        }

        // Enforce bounds on restored params
        for (i, spec) in if2_params.iter().enumerate() {
            let clamped = spec.from_transformed(current_transformed[i]);
            current_params[spec.index] = clamped;
        }
    } else {
        eprintln!("  initializing reference trajectory...");
        trajectory = simulate_reference(
            model, &current_params, t_end, config.dt, &mut rng,
        )?;
        eprintln!("  reference: {} substeps, initial S={}",
            trajectory.substeps.len(),
            trajectory.initial_counts.get(0).copied().unwrap_or(0));
        current_transformed = if2_params.iter()
            .map(|p| p.to_transformed(current_params[p.index]))
            .collect();
        start_sweep = 0;

        // Sanity check: the trajectory must have finite density at its own params
        // (before IVP mapping, which adds initial state density)
        let sanity_ll = complete_data_loglik(
            model, &trajectory, &current_params, observations,
            config.dt, obs_streams, &[],  // empty IVP mappings
        )?;
        if !sanity_ll.is_finite() {
            eprintln!("  BUG: simulate_reference trajectory has -inf density at own params.");
            eprintln!("  params used:");
            for p in &model.model.parameters {
                if let Some(&idx) = model.param_index.get(p.name.as_str()) {
                    eprintln!("    {} = {}", p.name, current_params[idx]);
                }
            }
        } else {
            eprintln!("  simulate_reference LL sanity check: {:.1} (finite ✓)", sanity_ll);
        }
    }

    // Adaptive proposal SDs via Robbins-Monro stochastic approximation.
    // Each parameter's log(proposal_sd) is nudged after every MH attempt
    // to target 44% acceptance (optimal for 1D MH, Roberts & Rosenthal 2001).
    // The adaptation rate c/√sweep decays to zero, so the proposal stabilizes.
    //
    // Initial scale: (upper - lower) / 10 on the TRANSFORMED scale, giving
    // the chain room to explore broadly during early burn-in. The Robbins-Monro
    // then narrows it to the right scale for each parameter. Starting too
    // small (e.g., rw_sd × 0.1) causes the chain to get stuck near its
    // starting values — the adaptation sees ~44% acceptance (because steps
    // are tiny) and never discovers that larger steps are needed.
    const TARGET_ACCEPTANCE: f64 = 0.44;
    const ADAPT_C: f64 = 2.0; // adaptation speed (higher = faster convergence)
    let adapt_end = config.burn_in; // stop adapting at end of burn-in

    let log_proposal_sd: Vec<f64> = if2_params.iter()
        .map(|p| {
            let lo = p.to_transformed(p.lower.max(1e-10));
            let hi = p.to_transformed(p.upper.min(1e10));
            let range = (hi - lo).abs();
            // 10% of the transformed-scale range: broad enough to explore,
            // Robbins-Monro will shrink to the right scale within ~200 sweeps
            (range / 10.0).max(0.01).ln()
        })
        .collect();

    // Detect IVP parameters: parameters that affect initial_state but not
    // propensities. These get stochastic initial states in CSMC and a
    // Binomial density term in the complete-data LL, enabling posterior
    // sampling through the Gibbs structure.
    let ivp_mappings: Vec<IVPMapping> = {
        let (init_base, _) = model.initial_state(&current_params)?;
        let mut mappings = Vec::new();
        for (i, spec) in if2_params.iter().enumerate() {
            let mut perturbed = current_params.clone();
            let delta = (spec.upper - spec.lower).min(1.0) * 0.01;
            perturbed[spec.index] = (perturbed[spec.index] + delta).min(spec.upper);
            let (init_pert, _) = model.initial_state(&perturbed)?;
            // Find which compartment changed
            for (c, (&base_c, &pert_c)) in init_base.counts.iter()
                .zip(init_pert.counts.iter()).enumerate()
            {
                // Skip balance compartment (it changes as a consequence)
                if model.balance.as_ref().map_or(false, |b| b.local_int_idx == c) {
                    continue;
                }
                if base_c != pert_c {
                    eprintln!("  {} detected as IVP → compartment {} \
                              (stochastic init, Binom density in LL)", spec.name, c);
                    mappings.push(IVPMapping {
                        param_idx: i,
                        model_param_idx: spec.index,
                        compartment_idx: c,
                    });
                    break;
                }
            }
        }
        mappings
    };

    // Initial complete-data log-likelihood (now includes initial state density)
    let current_ll = complete_data_loglik(
        model, &trajectory, &current_params, observations,
        config.dt, obs_streams, &ivp_mappings,
    )?;
    eprintln!("  initial complete-data ll: {:.1}", current_ll);
    if !current_ll.is_finite() {
        eprintln!("  WARNING: initial complete-data LL is -inf at the trajectory's own params.");
        eprintln!("  This indicates a mismatch between step_one and log_transition_density_substep.");
        eprintln!("  Run with CAMDL_TRACE_STEPS=1 for detailed per-substep diagnostics.");
        eprintln!("  Model has {} transitions, {} source groups",
            model.model.transitions.len(),
            model.source_groups.len());
    }

    // Check if gradients are available (compiler emitted rate_grad)
    let has_gradients = config.use_nuts && model.model.transitions.iter()
        .any(|t| !t.rate_grad.is_empty());
    if has_gradients {
        eprintln!("  NUTS enabled (gradient expressions found in IR)");
    }

    // ── Parallel tempering setup ──
    let n_rungs = config.tempering.len().max(1);
    let betas: Vec<f64> = if config.tempering.is_empty() { vec![1.0] } else { config.tempering.clone() };
    assert!((betas[0] - 1.0).abs() < 1e-12, "first tempering rung must be β=1.0 (cold chain)");
    for &b in &betas {
        assert!(b > 0.0 && b <= 1.0, "tempering β values must be in (0, 1], got {}", b);
    }
    if n_rungs > 1 {
        eprintln!("  parallel tempering: {} rungs, β = {:?}", n_rungs, betas);
    }

    // Per-rung state: rung 0 is cold (β=1), higher indices are hotter.
    // All rungs start from the same initial state.
    let mut rung_params: Vec<Vec<f64>> = vec![current_params.clone(); n_rungs];
    let mut rung_transformed: Vec<Vec<f64>> = vec![current_transformed.clone(); n_rungs];
    let mut rung_ll: Vec<f64> = vec![current_ll; n_rungs];
    let mut rung_trajectory: Vec<PGASTrajectory> = vec![trajectory.clone(); n_rungs];

    // NUTS state — restored from resume or initialized fresh (cold rung only,
    // heated rungs start fresh per the spec).
    let (nuts_mass_init, nuts_step_size_init, log_proposal_sd_restored,
         total_accepted_init, current_ll_restored) = if let Some((mass, ss, lpsd, ta, ll)) = resume_nuts {
        (mass, ss, lpsd, ta, Some(ll))
    } else {
        (super::nuts::MassMatrix::identity(d), 0.1, log_proposal_sd, vec![0usize; d], None)
    };

    // Per-rung adaptation state
    let mut rung_nuts_mass: Vec<super::nuts::MassMatrix> = (0..n_rungs)
        .map(|r| if r == 0 { nuts_mass_init.clone() } else { super::nuts::MassMatrix::identity(d) })
        .collect();
    let mut rung_nuts_step_size: Vec<f64> = (0..n_rungs)
        .map(|r| if r == 0 { nuts_step_size_init } else { 0.1 })
        .collect();
    let mut rung_log_proposal_sd: Vec<Vec<f64>> = (0..n_rungs)
        .map(|r| if r == 0 { log_proposal_sd_restored.clone() } else {
            // Heated rungs: same initial proposal SD as cold
            log_proposal_sd_restored.clone()
        })
        .collect();
    let mut rung_total_accepted: Vec<Vec<usize>> = (0..n_rungs)
        .map(|r| if r == 0 { total_accepted_init.clone() } else { vec![0usize; d] })
        .collect();
    let mut rung_nuts_dual_avg: Vec<super::nuts::DualAveraging> = (0..n_rungs)
        .map(|r| super::nuts::DualAveraging::new(rung_nuts_step_size[r], 0.80))
        .collect();

    // Per-rung Welford statistics for mass matrix adaptation
    let mut rung_welford_n: Vec<f64> = vec![0.0; n_rungs];
    let mut rung_welford_mean: Vec<Vec<f64>> = vec![vec![0.0; d]; n_rungs];
    let mut rung_welford_m2: Vec<Vec<f64>> = vec![vec![0.0; d]; n_rungs];
    let mut rung_welford_cov: Vec<Vec<f64>> = vec![vec![0.0; d * d]; n_rungs];

    let mut sweeps = Vec::new();

    // Override cold rung LL if we have a resumed value
    if let Some(ll) = current_ll_restored {
        rung_ll[0] = ll;
    }

    // Swap acceptance tracking (n_rungs - 1 adjacent pairs)
    let mut swap_proposed: Vec<usize> = vec![0; n_rungs.saturating_sub(1)];
    let mut swap_accepted: Vec<usize> = vec![0; n_rungs.saturating_sub(1)];

    if start_sweep >= config.n_sweeps {
        eprintln!("  warning: chain already completed {} sweeps (requested {}). \
                   Increase sweeps in fit.toml to continue.", start_sweep, config.n_sweeps);
    }

    for sweep in start_sweep..config.n_sweeps {
        // Per-rung accepted flags (only cold rung's is used for output)
        let mut rung_accepted: Vec<Vec<bool>> = vec![vec![false; d]; n_rungs];
        // Per-rung CSMC diagnostics (only cold rung's is used for output)
        let mut rung_csmc_diag: Vec<CSMCDiagnostics> = Vec::with_capacity(n_rungs);

        for rung in 0..n_rungs {
            let beta = betas[rung];

            // Current proposal SDs for this rung (MH only)
            let proposal_sd: Vec<f64> = rung_log_proposal_sd[rung].iter()
                .map(|&ls| ls.exp())
                .collect();

            // ── Step 1: Update θ | X, y ──
            // For heated rungs (β < 1), scale LL and its gradient by β.
            // Prior and Jacobian are untempered.
            if has_gradients {
                let param_names: Vec<String> = if2_params.iter().map(|p| p.name.clone()).collect();
                let param_model_indices: Vec<usize> = if2_params.iter().map(|p| p.index).collect();
                let rung_traj = &rung_trajectory[rung];

                let log_prob_and_grad = |z: &[f64]| -> (f64, Vec<f64>) {
                    let mut params = rung_params[rung].clone();
                    for (i, spec) in if2_params.iter().enumerate() {
                        params[spec.index] = spec.from_transformed(z[i]);
                    }

                    let (ll, ll_grad_theta) = match super::pgas_grad::complete_data_loglik_grad(
                        model, rung_traj, &params, observations,
                        config.dt, obs_streams, &ivp_mappings,
                        &param_names, &param_model_indices,
                    ) {
                        Ok(r) => r,
                        Err(_) => return (f64::NEG_INFINITY, vec![0.0; d]),
                    };

                    // Temper: scale LL by β
                    let mut log_p = beta * ll;
                    let mut grad_z = vec![0.0; d];

                    for i in 0..d {
                        let theta = params[if2_params[i].index];
                        let dtheta_dz = transform_deriv(&if2_params[i], z[i]);

                        // LL gradient: chain rule θ → z, scaled by β
                        grad_z[i] += beta * ll_grad_theta[i] * dtheta_dz;

                        // Prior: untempered
                        let (prior_val, prior_grad_z) = prior_log_density_and_grad_z(
                            &priors[i], &if2_params[i], theta, z[i],
                        );
                        log_p += prior_val;
                        grad_z[i] += prior_grad_z;

                        // Jacobian: untempered
                        log_p += if2_params[i].log_jacobian(z[i]);
                        grad_z[i] += if2_params[i].jacobian_grad(z[i]);
                    }

                    (log_p, grad_z)
                };

                let (init_log_p, init_grad) = log_prob_and_grad(&rung_transformed[rung]);

                let nuts_config = super::nuts::NUTSConfig {
                    max_tree_depth: 10,
                    step_size: rung_nuts_step_size[rung],
                    mass_matrix: rung_nuts_mass[rung].clone(),
                };

                let result = super::nuts::nuts_step(
                    &rung_transformed[rung], init_log_p, &init_grad,
                    &nuts_config, &log_prob_and_grad, &mut rng,
                );

                if result.accepted {
                    rung_transformed[rung].copy_from_slice(&result.params);
                    for (i, spec) in if2_params.iter().enumerate() {
                        rung_params[rung][spec.index] = spec.from_transformed(rung_transformed[rung][i]);
                    }
                    for a in &mut rung_accepted[rung] { *a = true; }
                    for t in &mut rung_total_accepted[rung] { *t += 1; }
                }

                // Two-phase adaptation (same schedule as single-rung, per-rung state)
                let mass_adapt_end = (adapt_end as f64 * 0.7) as usize;

                if sweep < mass_adapt_end {
                    rung_nuts_step_size[rung] = rung_nuts_dual_avg[rung].update(result.mean_accept_prob);

                    rung_welford_n[rung] += 1.0;
                    let old_mean = rung_welford_mean[rung].clone();
                    for i in 0..d {
                        let delta = rung_transformed[rung][i] - rung_welford_mean[rung][i];
                        rung_welford_mean[rung][i] += delta / rung_welford_n[rung];
                        let delta2 = rung_transformed[rung][i] - rung_welford_mean[rung][i];
                        rung_welford_m2[rung][i] += delta * delta2;
                    }
                    for i in 0..d {
                        for j in 0..d {
                            rung_welford_cov[rung][i * d + j] +=
                                (rung_transformed[rung][i] - old_mean[i])
                                * (rung_transformed[rung][j] - rung_welford_mean[rung][j]);
                        }
                    }
                } else if sweep == mass_adapt_end {
                    if rung_welford_n[rung] > 10.0 {
                        if config.dense_mass {
                            let mut cov = vec![0.0; d * d];
                            for i in 0..d {
                                for j in 0..d {
                                    cov[i * d + j] = rung_welford_cov[rung][i * d + j] / (rung_welford_n[rung] - 1.0);
                                }
                            }
                            rung_nuts_mass[rung] = super::nuts::MassMatrix::dense_from_covariance(&cov, d);
                            if rung == 0 {
                                eprintln!("  dense mass matrix estimated (sweep {}):", sweep);
                                for (i, spec) in if2_params.iter().enumerate() {
                                    let sd = (cov[i * d + i]).max(1e-10).sqrt();
                                    eprintln!("    {:12} sd={:.6}", spec.name, sd);
                                }
                                eprint!("    correlations:");
                                for i in 0..d {
                                    for j in (i+1)..d {
                                        let r = cov[i * d + j]
                                            / (cov[i * d + i].max(1e-10).sqrt() * cov[j * d + j].max(1e-10).sqrt());
                                        eprint!(" {}-{}={:.2}", &if2_params[i].name[..3.min(if2_params[i].name.len())],
                                            &if2_params[j].name[..3.min(if2_params[j].name.len())], r);
                                    }
                                }
                                eprintln!();
                            }
                        } else {
                            let variances: Vec<f64> = (0..d).map(|i|
                                (rung_welford_m2[rung][i] / (rung_welford_n[rung] - 1.0)).max(1e-10)
                            ).collect();
                            if rung == 0 {
                                eprintln!("  diagonal mass matrix estimated (sweep {}):", sweep);
                                for (i, spec) in if2_params.iter().enumerate() {
                                    eprintln!("    {:12} sd={:.6}", spec.name, variances[i].sqrt());
                                }
                            }
                            rung_nuts_mass[rung] = super::nuts::MassMatrix::diagonal(variances);
                        }
                    }
                    rung_nuts_step_size[rung] = 0.1;
                    rung_nuts_dual_avg[rung] = super::nuts::DualAveraging::new(rung_nuts_step_size[rung], 0.80);
                } else if sweep < adapt_end {
                    rung_nuts_step_size[rung] = rung_nuts_dual_avg[rung].update(result.mean_accept_prob);
                } else if sweep == adapt_end && rung == 0 {
                    rung_nuts_step_size[rung] = rung_nuts_dual_avg[rung].final_step_size();
                    eprintln!("  NUTS fully adapted (sweep {}):", sweep);
                    eprintln!("    final step_size: {:.6}", rung_nuts_step_size[rung]);
                } else if sweep == adapt_end {
                    rung_nuts_step_size[rung] = rung_nuts_dual_avg[rung].final_step_size();
                }
            } else {
                // MH-within-Gibbs: one-at-a-time random walk proposals
                // For heated rungs, scale LL by β in the MH ratio.
                for i in 0..d {
                    let spec = &if2_params[i];
                    let z_old = rung_transformed[rung][i];
                    let z_new = z_old + proposal_sd[i] * rng.normal();
                    let theta_new = spec.from_transformed(z_new);

                    let mut proposed_params = rung_params[rung].clone();
                    proposed_params[spec.index] = theta_new;

                    let proposed_ll = complete_data_loglik(
                        model, &rung_trajectory[rung], &proposed_params, observations,
                        config.dt, obs_streams, &ivp_mappings,
                    )?;

                    let proposed_log_prior_i = priors[i].log_density(theta_new, z_new);
                    let current_log_prior_i = priors[i].log_density(
                        rung_params[rung][spec.index], z_old,
                    );
                    let proposed_log_jac_i = spec.log_jacobian(z_new);
                    let current_log_jac_i = spec.log_jacobian(z_old);

                    // Temper: scale LL difference by β, prior + Jacobian untempered
                    let log_alpha = beta * (proposed_ll - rung_ll[rung])
                                  + (proposed_log_prior_i - current_log_prior_i)
                                  + (proposed_log_jac_i - current_log_jac_i);

                    if log_alpha.is_finite() && rng.uniform().ln() < log_alpha {
                        rung_params[rung][spec.index] = theta_new;
                        rung_transformed[rung][i] = z_new;
                        rung_ll[rung] = proposed_ll;
                        rung_accepted[rung][i] = true;
                        rung_total_accepted[rung][i] += 1;
                    }

                    // Robbins-Monro adaptation (per-rung)
                    if sweep < adapt_end {
                        let gamma_rm = ADAPT_C / (1.0 + sweep as f64).sqrt();
                        let acc_indicator = if rung_accepted[rung][i] { 1.0 } else { 0.0 };
                        rung_log_proposal_sd[rung][i] += gamma_rm * (acc_indicator - TARGET_ACCEPTANCE);
                        rung_log_proposal_sd[rung][i] = rung_log_proposal_sd[rung][i].clamp(-20.0, 5.0);
                    }
                }
            }

            // ── Step 2: Update X | θ, y via CSMC-AS ──
            // CSMC always runs at β=1 — the trajectory must match the data.
            let csmc_seed = seed ^ ((sweep as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
                ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142);
            let (new_trajectory, csmc_diag) = csmc_as(
                model, &rung_params[rung], observations, &rung_trajectory[rung],
                config.n_particles, config.dt, obs_streams,
                &ivp_mappings, csmc_seed,
            )?;
            rung_trajectory[rung] = new_trajectory;

            // Recompute complete-data LL at β=1 (untempered, for swap proposals)
            rung_ll[rung] = complete_data_loglik(
                model, &rung_trajectory[rung], &rung_params[rung], observations,
                config.dt, obs_streams, &ivp_mappings,
            )?;

            rung_csmc_diag.push(csmc_diag);
        } // end rung loop

        // ── Replica exchange: swap adjacent rungs ──
        if n_rungs > 1 {
            // Even-odd scheme: alternate starting parity each sweep
            let pair_start = sweep % 2;
            let mut i = pair_start;
            while i + 1 < n_rungs {
                let j = i + 1;
                swap_proposed[i] += 1;

                // Acceptance: α = min(1, exp((β_i - β_j) * (LL_i - LL_j)))
                // where LL is the UNTEMPERED complete-data log-likelihood.
                let log_alpha = (betas[i] - betas[j]) * (rung_ll[i] - rung_ll[j]);

                if log_alpha >= 0.0 || rng.uniform().ln() < log_alpha {
                    swap_accepted[i] += 1;

                    // Swap all state between rungs i and j
                    rung_params.swap(i, j);
                    rung_transformed.swap(i, j);
                    rung_ll.swap(i, j);
                    rung_trajectory.swap(i, j);
                    rung_accepted.swap(i, j);

                    // Adaptation state swaps WITH the parameters (it belongs
                    // to the parameter state, not the temperature).
                    rung_nuts_mass.swap(i, j);
                    rung_nuts_step_size.swap(i, j);
                    rung_log_proposal_sd.swap(i, j);
                    rung_total_accepted.swap(i, j);
                    rung_nuts_dual_avg.swap(i, j);
                    rung_welford_n.swap(i, j);
                    rung_welford_mean.swap(i, j);
                    rung_welford_m2.swap(i, j);
                    rung_welford_cov.swap(i, j);
                }

                i += 2;
            }
        }

        // ── Cold rung (index 0) output ──
        // Log adapted proposal SDs at end of burn-in (cold rung only)
        if sweep + 1 == adapt_end {
            eprintln!("  proposal SD adapted (end of burn-in):");
            for (i, spec) in if2_params.iter().enumerate() {
                let acc_rate = rung_total_accepted[0][i] as f64 / (sweep + 1) as f64;
                eprintln!("    {:12} sd={:.6} acc={:.0}%",
                    spec.name, rung_log_proposal_sd[0][i].exp(), acc_rate * 100.0);
            }
            eprintln!("  trajectory renewal: {:.1}%", rung_csmc_diag[0].trajectory_renewal * 100.0);

            // Report swap rates at end of burn-in
            if n_rungs > 1 {
                eprintln!("  tempering swap rates:");
                for i in 0..n_rungs - 1 {
                    let rate = if swap_proposed[i] > 0 {
                        swap_accepted[i] as f64 / swap_proposed[i] as f64
                    } else { 0.0 };
                    eprintln!("    B={:.2} <-> B={:.2}: {:.1}%",
                        betas[i], betas[i + 1], rate * 100.0);
                }
            }
        }

        let cold_proposal_sd: Vec<f64> = rung_log_proposal_sd[0].iter()
            .map(|&ls| ls.exp())
            .collect();

        let sweep_result = PGASSweep {
            params: rung_params[0].clone(),
            log_complete_data_ll: rung_ll[0],
            accepted: rung_accepted[0].clone(),
            csmc_diag: rung_csmc_diag[0].clone(),
            proposal_sds: cold_proposal_sd,
        };

        if let Some(cb) = on_sweep {
            cb(sweep, &sweep_result, &rung_trajectory[0]);
        }

        // Record (respecting burn-in and thinning)
        if sweep >= config.burn_in && (sweep - config.burn_in) % config.thin == 0 {
            sweeps.push(sweep_result);
        }
    }

    let acceptance_rates: Vec<f64> = rung_total_accepted[0].iter()
        .map(|&n| n as f64 / config.n_sweeps as f64)
        .collect();

    let resume_state = ChainResumeState {
        config_hash,
        completed_sweeps: config.n_sweeps,
        params: rung_params[0].clone(),
        transformed: rung_transformed[0].clone(),
        param_names: if2_params.iter().map(|p| p.name.clone()).collect(),
        trajectory: rung_trajectory[0].clone(),
        mass_matrix: rung_nuts_mass[0].clone(),
        nuts_step_size: rung_nuts_step_size[0],
        log_proposal_sd: rung_log_proposal_sd[0].clone(),
        total_accepted: rung_total_accepted[0].clone(),
        current_ll: rung_ll[0],
    };

    Ok(PGASResult {
        sweeps,
        final_trajectory: rung_trajectory[0].clone(),
        acceptance_rates,
        resume_state,
    })
}
