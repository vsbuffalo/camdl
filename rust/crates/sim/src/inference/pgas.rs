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
use crate::inference::types::{EstimatedParam, LOG_PROB_FLOOR, RESAMPLE_RNG_STREAM, init_particle_rngs, restore_z_values};
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::eval_resolved;
use crate::state::{IntState, RealState};

// ═══════════════════════════════════════════════════════════════════
// Named constants
// ═══════════════════════════════════════════════════════════════════

/// Fibonacci/Knuth multiplicative hash constant (2^64 / φ). Gives good
/// avalanche properties for (sweep, rung, rep) seed mixing.
const SEED_MIX_KNUTH: u64 = 0x9e3779b97f4a7c15;
/// FNV-1a-derived mixing constant for per-warmup-sweep seed separation.
const SEED_MIX_WARMUP: u64 = 0x517cc1b727220a95;
/// FNV-1a-derived mixing constant for per-rung seed separation.
const SEED_MIX_RUNG: u64 = 0x6c62272e07bb0142;
/// FNV-derived mixing constant for per-CSMC-rep seed separation.
const SEED_MIX_REP: u64 = 0xa2ce44bbfe0cf6d5;

/// Fraction of the adaptation window used for mass-matrix estimation.
/// Stan's dual-averaging schedule uses 0.75 for the equivalent split;
/// 0.70 gives a longer burn-in window before Cholesky updates begin.
const MASS_ADAPT_FRAC: f64 = 0.70;

/// Probability-domain clamp to keep values strictly in (0,1) for
/// numerical stability in log(p) / log(1-p) computations.
/// Distinct from LOG_PROB_FLOOR which is a log-domain floor.
const PROB_CLAMP_EPS: f64 = 1e-15;

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
    /// Maximum NUTS tree depth. Default: 10.
    pub max_tree_depth: usize,
    /// Number of CSMC-only sweeps before parameter updates begin.
    /// During warm-up, the trajectory is refreshed via CSMC-AS but
    /// parameters are held fixed. Default: 0 (no warm-up).
    pub trajectory_warmup: usize,
    /// Number of CSMC trajectory updates per parameter update.
    /// Default: 1. Higher values (e.g., 3-5) improve trajectory
    /// convergence on models with long time series where ancestor
    /// sampling is the bottleneck. Each extra CSMC sweep renovates
    /// more of the trajectory before the next NUTS step.
    pub csmc_sweeps_per_nuts: usize,
}

impl super::traits::InferenceConfig for PGASConfig {
    fn n_particles(&self) -> usize { self.n_particles }
    fn dt(&self) -> f64 { self.dt }
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

/// Decomposed complete-data log-likelihood components.
#[derive(Clone, Debug)]
pub struct LogLikComponents {
    /// Sum of all components.
    pub total: f64,
    /// Sum of per-substep transition densities.
    pub transition: f64,
    /// Sum of observation densities (joint_obs_weight).
    pub observation: f64,
    /// Initial state density (Binomial for IVP params).
    pub ivp: f64,
}

/// Result of one Gibbs sweep.
#[derive(Clone, Serialize, Deserialize)]
pub struct PGASSweep {
    pub params: Vec<f64>,
    pub log_complete_data_ll: f64,
    pub accepted: Vec<bool>,
    pub csmc_diag: CSMCDiagnostics,
    pub proposal_sds: Vec<f64>,
    /// Transition component of the complete-data log-likelihood.
    pub transition_ll: f64,
    /// Observation component of the complete-data log-likelihood.
    pub obs_ll: f64,
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

/// Map from substep index to observation index.
///
/// Built once from observation times and dt, then passed to
/// `complete_data_loglik`, `csmc_as`, and `complete_data_loglik_grad`
/// to avoid rebuilding each call.
pub type ObsAtSubstep = std::collections::HashMap<usize, usize>;

/// Build the substep→observation index mapping.
pub fn build_obs_at_substep(
    observations: &[Observation],
    t_start: f64,
    dt: f64,
) -> ObsAtSubstep {
    let mut map = ObsAtSubstep::new();
    for (obs_idx, obs) in observations.iter().enumerate() {
        let s = ((obs.time - t_start) / dt).round() as usize;
        if s > 0 { map.insert(s - 1, obs_idx); }
    }
    map
}

// ═══════════════════════════════════════════════════════════════════
// Transition density
// ═══════════════════════════════════════════════════════════════════

/// Build the effective-rate list for one source group, advancing `gamma_idx`.
///
/// Returns `(probs, total_rate)` where `probs[k] = (tr_idx, effective_rate)`.
/// The `effective_rate` for an overdispersed transition is `per_capita * g`
/// where `g = gammas[gamma_idx]`; `gamma_idx` is advanced once per
/// overdispersed transition with rate above RATE_EPSILON, in the same order
/// that `step_one` pushes to `gamma_used`.
///
/// Returns `Err(f64::NEG_INFINITY)` when a transition has rate=0 but nonzero
/// flow, which is an impossible state — the density is zero and the caller
/// should propagate NEG_INFINITY immediately.
fn compute_source_group_probs(
    group: &[usize],
    flows: &[u64],
    propensities: &[f64],
    is_determ: &[bool],
    sigma_sq_by_tr: &[Option<f64>],
    gammas: &[f64],
    gamma_idx: &mut usize,
    n_src: i64,
) -> Result<(Vec<(usize, f64)>, f64), f64> {
    let mut probs: Vec<(usize, f64)> = Vec::new();
    let mut total_rate = 0.0_f64;

    for &tr_idx in group {
        let rate = propensities[tr_idx];
        if rate <= RATE_EPSILON {
            if flows[tr_idx] > 0 && rate <= 0.0 {
                log::warn!(
                    "transition index {} has rate=0 but flow={}. \
                     Add a seeding term (iota) to avoid this impossible state.",
                    tr_idx, flows[tr_idx],
                );
                return Err(f64::NEG_INFINITY);
            } else if flows[tr_idx] > 0 {
                // Near-zero rate with nonzero flow: include with tiny rate.
                let per_capita = rate / n_src as f64;
                total_rate += per_capita;
                probs.push((tr_idx, per_capita));
                continue;
            }
            continue;
        }
        if is_determ[tr_idx] { continue; }

        let per_capita = rate / n_src as f64;
        let effective = if sigma_sq_by_tr[tr_idx].is_some() {
            // Consume one gamma per overdispersed transition — same order as step_one.
            let g = if *gamma_idx < gammas.len() { gammas[*gamma_idx] } else { 1.0 };
            *gamma_idx += 1;
            per_capita * g
        } else {
            per_capita
        };
        total_rate += effective;
        probs.push((tr_idx, effective));
    }

    Ok((probs, total_rate))
}

/// Log-density for the total-exits Binomial and the multinomial split.
///
/// Evaluates:
///   log Binom(n_exit; n_src, p_total)
///   + Σ_{k=0}^{K-2} log Binom(flow_k; remaining_k, p_split_k)
///
/// where `p_total = 1 - exp(-total_rate * dt)` and
/// `p_split_k = eff_rate_k / rate_remaining_k`. Returns NEG_INFINITY if
/// the observed counts are incompatible with `probs` (impossible partition).
fn exit_and_split_log_density(
    n_src: i64,
    n_exit: u64,
    total_rate: f64,
    dt: f64,
    probs: &[(usize, f64)],
    flows: &[u64],
    src_local: usize,
) -> f64 {
    let p_total = (1.0 - (-total_rate * dt).exp()).clamp(PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS);
    let binom_total = binom_logpmf(n_exit, n_src as u64, p_total);

    if !binom_total.is_finite() {
        log::debug!("density: total exits -inf: Binom({}, {}, {:.6e}), src_comp_idx={}",
            n_exit, n_src, p_total, src_local);
        return f64::NEG_INFINITY;
    }

    let mut log_p = binom_total;
    let n_competing = probs.len();
    let mut remaining = n_exit;
    let mut rate_remaining = total_rate;

    for (k, &(tr_idx, eff_rate)) in probs.iter().enumerate() {
        if k == n_competing - 1 {
            if flows[tr_idx] != remaining { return f64::NEG_INFINITY; }
        } else if remaining > 0 && rate_remaining > 0.0 {
            let p_split = (eff_rate / rate_remaining).clamp(PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS);
            log_p += binom_logpmf(flows[tr_idx], remaining, p_split);
            remaining -= flows[tr_idx];
            rate_remaining -= eff_rate;
        } else if flows[tr_idx] > 0 {
            return f64::NEG_INFINITY;
        }
    }

    log_p
}

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
        model, int_s: &int_s, real_s: &real_s, params, t, projected: None, int_float_override: None,
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

    // Source-grouped transitions (mirrors step_one's Euler-multinomial).
    // Stage 1: compute effective rates (gamma_idx advances here, same order as step_one).
    // Stage 2: Binomial total-exits + multinomial split densities.
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts_before[src_local].max(0);
        if n_src == 0 {
            for &tr_idx in group {
                if flows[tr_idx] > 0 { return Ok(f64::NEG_INFINITY); }
                handled[tr_idx] = true;
            }
            continue;
        }

        let (probs, total_rate) = match compute_source_group_probs(
            group, flows, &propensities, &is_determ, &sigma_sq_by_tr,
            gammas, &mut gamma_idx, n_src,
        ) {
            Ok(r) => r,
            Err(neg_inf) => return Ok(neg_inf),
        };

        if total_rate <= RATE_EPSILON || probs.is_empty() { continue; }

        let n_exit: u64 = probs.iter().map(|&(tr_idx, _)| flows[tr_idx]).sum();
        let density = exit_and_split_log_density(
            n_src, n_exit, total_rate, dt, &probs, flows, src_local,
        );
        if density == f64::NEG_INFINITY { return Ok(f64::NEG_INFINITY); }
        log_p += density;

        for &(tr_idx, _) in &probs { handled[tr_idx] = true; }
        // Also mark any low-rate/deterministic transitions in the group as handled.
        for &tr_idx in group { handled[tr_idx] = true; }
    }

    // Ungrouped / inflow transitions: Poisson density (or deterministic exact-count check).
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
    _observations: &[Observation],
    dt: f64,
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    ivp_mappings: &[IVPMapping],
    obs_at_substep: &ObsAtSubstep,
) -> Result<LogLikComponents, SimError> {
    let n_substeps = trajectory.substeps.len();
    let n_tr = model.model.transitions.len();
    let mut ivp_ll = 0.0;
    let mut transition_ll = 0.0;
    let mut observation_ll = 0.0;

    // Initial state density: log p(x₀ | θ) for IVP-controlled compartments.
    // S₀ ~ Binom(N₀, s0) → log Binom(S₀; N₀, s0) constrains s0.
    // N₀ is the total population of the PATCH containing this compartment,
    // not the global population across all patches. We compute it as the
    // sum of initial counts in the same stratification group.
    if !ivp_mappings.is_empty() {
        for ivp in ivp_mappings {
            let count = trajectory.initial_counts[ivp.compartment_idx] as u64;
            let frac = params[ivp.model_param_idx].clamp(1e-10, 1.0 - 1e-10);
            let patch_pop = patch_population(model, &trajectory.initial_counts, ivp.compartment_idx);
            let this_ivp_ll = binom_logpmf(count, patch_pop as u64, frac);
            if !this_ivp_ll.is_finite() {
                let comp_name = &model.model.compartments[ivp.compartment_idx].name;
                eprintln!("  IVP density -inf: Binom({}, {}, {:.6e}) for {} (comp={}, patch_pop={})",
                    count, patch_pop, frac,
                    comp_name, ivp.compartment_idx, patch_pop);
            }
            ivp_ll += this_ivp_ll;
        }
    }

    if !ivp_ll.is_finite() {
        log::debug!("complete_data_loglik: -inf after IVP density (ivp_ll={:.1})", ivp_ll);
        return Ok(LogLikComponents {
            total: f64::NEG_INFINITY,
            transition: 0.0,
            observation: 0.0,
            ivp: ivp_ll,
        });
    }

    // Cumulative flows since last observation (for projection)
    let mut cum_flows = vec![0u64; n_tr];
    let t_start = model.model.simulation.t_start;

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
            return Ok(LogLikComponents {
                total: f64::NEG_INFINITY,
                transition: transition_ll + td,
                observation: observation_ll,
                ivp: ivp_ll,
            });
        }
        transition_ll += td;

        // Gamma multiplier density: log Gamma(g; shape, scale) for each
        // gamma recorded at this substep. The gammas are stored in order
        // matching step_one's push order (one per overdispersed transition
        // with rate > 0).
        //
        // Shape = dt / σ², Scale = σ² / dt. Mean = shape * scale = 1.
        // This constrains the gamma multiplier to be near 1 (no overdispersion
        // at high shape) or allow large variation (low shape = high σ²).
        if !rec.gammas.is_empty() {
            // Collect σ² values for overdispersed transitions in source-group order,
            // matching the order step_one pushes to gamma_used.
            let n_int_local = model.int_local_to_global.len();
            let int_s_local = IntState::new(n_int_local);
            let real_s_local = RealState::new(model.real_local_to_global.len());
            let ctx = EvalCtx {
                model, int_s: &int_s_local, real_s: &real_s_local,
                params, t: model.model.simulation.t_start + s as f64 * dt,
                projected: None, int_float_override: None,
            };
            let mut gamma_idx_local = 0;
            for &(src_local, ref group) in &model.source_groups {
                let n_src = rec.counts_before[src_local].max(0);
                if n_src == 0 { continue; }
                // Recompute propensities for rate check
                let mut local_props = vec![0.0; n_tr];
                let _ = eval_propensities(model, &{
                    let mut s = IntState::new(n_int_local);
                    s.counts.copy_from_slice(&rec.counts_before);
                    s
                }, &real_s_local, params, ctx.t, &mut local_props);
                for &tr_idx in group {
                    let rate = local_props[tr_idx];
                    if rate <= RATE_EPSILON { continue; }
                    if let ir::transition::DrawMethod::Deterministic = model.model.transitions[tr_idx].draw_method {
                        continue;
                    }
                    if let Some(ref resolved_od) = model.resolved.overdispersion[tr_idx] {
                        let sigma_sq = eval_resolved(resolved_od, &ctx);
                        if gamma_idx_local < rec.gammas.len() && sigma_sq > 1e-30 {
                            let g = rec.gammas[gamma_idx_local];
                            let shape = dt / sigma_sq;
                            let scale = sigma_sq / dt;
                            // log Gamma(g; shape, scale) = (shape-1)*ln(g) - g/scale
                            //   - shape*ln(scale) - ln(Gamma(shape))
                            let log_gamma_density = (shape - 1.0) * g.max(LOG_PROB_FLOOR).ln()
                                - g / scale
                                - shape * scale.ln()
                                - crate::inference::obs_loglik::lgamma(shape);
                            transition_ll += log_gamma_density;
                        }
                        gamma_idx_local += 1;
                    }
                }
            }
            if gamma_idx_local != rec.gammas.len() {
                log::warn!(
                    "gamma index mismatch at substep {}: tracked {} but trajectory recorded {} gammas",
                    s, gamma_idx_local, rec.gammas.len()
                );
            }
        }

        // Accumulate flows
        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }

        // Observation density — joint across all streams. Snapshot
        // projections read post-step state (after step_one fired any
        // scheduled intervention at t+dt).
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            let obs_ll = obs_model.log_likelihood_from_flows_and_counts(
                &cum_flows, &rec.counts_after, obs_idx, params);
            if !obs_ll.is_finite() {
                log::debug!("complete_data_loglik: obs density -inf at substep {} (obs_idx={})", s, obs_idx);
            }
            observation_ll += obs_ll;
            let total = ivp_ll + transition_ll + observation_ll;
            if !total.is_finite() {
                log::debug!("complete_data_loglik: -inf after obs at substep {} (cumulative)", s);
                return Ok(LogLikComponents {
                    total: f64::NEG_INFINITY,
                    transition: transition_ll,
                    observation: observation_ll,
                    ivp: ivp_ll,
                });
            }
            cum_flows.fill(0);
        }
    }

    Ok(LogLikComponents {
        total: ivp_ll + transition_ll + observation_ll,
        transition: transition_ll,
        observation: observation_ll,
        ivp: ivp_ll,
    })
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
    _observations: &[Observation],
    reference: &PGASTrajectory,
    n_particles: usize,
    dt: f64,
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    ivp_mappings: &[IVPMapping],
    seed: u64,
    obs_at_substep: &ObsAtSubstep,
) -> Result<(PGASTrajectory, CSMCDiagnostics), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = reference.substeps.len();
    let n_tr = model.model.transitions.len();
    let j_ref = n_particles - 1; // reference particle is the last slot

    // Initialize particles with stochastic initial states for IVP compartments.
    // Each free particle draws S₀ ~ Binom(N₀, s0) independently, giving the
    // CSMC diverse initial states to select among. This is what enables
    // posterior sampling of IVP parameters like s0.
    let (init_int, _) = model.initial_state(params)?;
    let total_pop = init_int.counts.iter().sum::<i64>();

    // Precompute per-IVP patch populations (for stratified models, N₀ is the
    // patch population, not the global population).
    let ivp_patch_pops: Vec<i64> = ivp_mappings.iter()
        .map(|ivp| patch_population(model, &init_int.counts, ivp.compartment_idx))
        .collect();

    // Per-particle RNGs via ChaCha8 stream counter (IM1 fix 2026-04-19).
    let mut rngs = init_particle_rngs(seed, n_particles, 0);

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
    let initial_counts_per_particle: Vec<Vec<i64>> = counts.to_vec();

    // History for traceback
    let mut history_counts_before: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_counts_after: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut history_flows: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_substeps);
    let mut history_gammas: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_substeps);
    let mut ancestors: Vec<Vec<usize>> = Vec::with_capacity(n_substeps);

    // Weights (log-space)
    let mut log_weights = vec![0.0f64; n_particles];

    // Resampling RNG — uses a reserved high stream index so it never
    // collides with per-particle streams (which use [0, n_particles)).
    let mut resample_rng = StatefulRng::new_stream(seed, RESAMPLE_RNG_STREAM);

    // Per-particle scratch buffers
    let mut scratches: Vec<StepScratch> = (0..n_particles)
        .map(|_| StepScratch::new(model))
        .collect();

    // Previous states (for ancestor sampling: need state before propagation)
    let mut prev_counts: Vec<Vec<i64>> = counts.clone();

    // Diagnostic: count substeps where ancestor sampling is degenerate
    // (no particle can reach the reference state → reference stays self-connected)
    let mut n_degenerate: usize = 0;

    // Pre-allocated buffer for ancestor sampling weights (reused each substep)
    let mut ancestor_log_w = vec![f64::NEG_INFINITY; n_particles];

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

            std::mem::swap(&mut substep_gammas[j], &mut scratches[j].gamma_used);
        }

        // ── 3. Clamp reference particle ──
        let ref_rec = &reference.substeps[s];
        counts[j_ref].copy_from_slice(&ref_rec.counts_after);
        substep_flows[j_ref].copy_from_slice(&ref_rec.flows);
        substep_gammas[j_ref].clear();
        substep_gammas[j_ref].extend_from_slice(&ref_rec.gammas);
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
        //
        // IM6 in 2026-04-19 inference review: ancestor sampling runs
        // POST-resample here, so `prev_counts[j]` is the state at
        // slot j after resampling (i.e., the pre-resample state of
        // ancestor `indices[j]`), while `log_weights[j]` is the
        // pre-resample weight at ORIGINAL slot j (never reshuffled
        // after resampling at step 1). That pairs a weight from
        // slot-j pre-resample with a state from slot-indices[j]
        // pre-resample — a mismatch.
        //
        // After resampling, per CSMC theory, all slots carry
        // uniform weight 1/N. The correct ancestor weight in the
        // post-resample placement is therefore just the transition
        // density: ã_j ∝ f(X_ref_s | prev_counts[j]). Adding the
        // stale log_weights[j] skews the categorical toward slots
        // whose original weights were high, regardless of whether
        // the ancestor-source state (at slot indices[j]) happens to
        // be a good precursor to X_ref.
        //
        // Fix: drop log_weights[j] from the sum. The current
        // log_weights[j] value is discarded at step 5 anyway (either
        // overwritten by the obs log-likelihood or reset to 0), so
        // removing it here doesn't affect subsequent steps.
        {
            ancestor_log_w.fill(f64::NEG_INFINITY);
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
                // Post-resample slot j carries uniform weight (1/N);
                // the categorical is driven by td alone.
                ancestor_log_w[j] = td;
            }

            // Sample ancestor from categorical(softmax(ancestor_log_w)).
            // Degenerate case (all -inf): keep reference's own history to
            // maintain internal consistency — the reference's flows at
            // substep s were produced from the reference's state at s-1.
            let ref_ancestor = match sample_categorical_log(&ancestor_log_w, &mut resample_rng) {
                Some(j) => j,
                None => { n_degenerate += 1; j_ref }
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
                log_weights[j] = obs_model.log_likelihood_from_flows_and_counts(
                    &cum_flows[j], &counts[j], obs_idx, params);
            }
            for j in 0..n_particles {
                for f in &mut cum_flows[j] { *f = 0; }
            }
        } else {
            // Non-observation substep: uniform weights
            log_weights.fill(0.0);
        }

        // ── 6. Store history ──
        history_counts_before.push(prev_counts.to_vec());
        history_counts_after.push(counts.to_vec());
        history_flows.push(substep_flows.to_vec());
        history_gammas.push(substep_gammas.to_vec());
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
    let k = sample_categorical_log(&log_weights, &mut resample_rng).unwrap_or(j_ref);

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

/// Sample from a categorical distribution parameterized by unnormalized log-weights.
///
/// Applies the log-sum-exp trick for numerical stability: subtracts the max
/// log-weight before exponentiating, then draws from the resulting categorical.
/// Returns `None` if all weights are -inf (degenerate case).
fn sample_categorical_log(log_weights: &[f64], rng: &mut StatefulRng) -> Option<usize> {
    let max_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_w.is_finite() {
        return None;
    }
    let weights: Vec<f64> = log_weights.iter().map(|&w| (w - max_w).exp()).collect();
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    let u = rng.uniform() * sum;
    let mut cum = 0.0;
    for (j, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= u { return Some(j); }
    }
    Some(weights.len() - 1)
}

/// Compute the population of the patch containing `compartment_idx`.
///
/// In a stratified model, compartments in the same patch share a suffix
/// (e.g., `S_patch1`, `I_patch1`). The patch population is the sum of
/// initial counts for all compartments with matching suffix.
/// For unstratified models (no `_` in the name), returns total population.
pub fn patch_population(
    model: &CompiledModel,
    initial_counts: &[i64],
    compartment_idx: usize,
) -> i64 {
    let total: i64 = initial_counts.iter().sum();
    let comp_name = &model.model.compartments[compartment_idx].name;
    let patch_suffix = comp_name.rsplit('_').next().unwrap_or("");
    if patch_suffix.is_empty() || !comp_name.contains('_') {
        total
    } else {
        model.model.compartments.iter().enumerate()
            .filter(|(_, c)| c.name.ends_with(&format!("_{}", patch_suffix)))
            .map(|(i, _)| initial_counts[i])
            .sum()
    }
}

/// Prior log-density AND its gradient on the z (unconstrained) scale.
///
/// Delegates the density computation to `Prior::log_density` and computes
/// only the gradient part here. The chain rule converts d(log prior)/dθ
/// to d/dz via `param.transform_deriv(z)`.
fn prior_log_density_and_grad_z(
    prior: &Prior, param: &EstimatedParam, theta: f64, z: f64,
) -> (f64, f64) {
    let lp = prior.log_density(theta, z);
    let dlp_dz = match prior {
        Prior::Flat => 0.0,
        Prior::Uniform { lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            0.0 // flat density → zero gradient inside support
        }
        Prior::Normal { mean, sd } => {
            let dlp_dtheta = -(theta - mean) / (sd * sd);
            dlp_dtheta * param.transform_deriv(z)
        }
        Prior::TransformedNormal { mean, sd } => {
            -(z - mean) / (sd * sd)
        }
        Prior::HalfNormal { sigma } => {
            if theta < 0.0 { return (lp, 0.0); }
            let dlp_dtheta = -theta / (sigma * sigma);
            dlp_dtheta * param.transform_deriv(z)
        }
        Prior::Beta { alpha, beta } => {
            if theta <= 0.0 || theta >= 1.0 { return (lp, 0.0); }
            let dlp_dtheta = (alpha - 1.0) / theta - (beta - 1.0) / (1.0 - theta);
            dlp_dtheta * param.transform_deriv(z)
        }
        Prior::Gamma { shape, rate } => {
            if theta <= 0.0 { return (lp, 0.0); }
            let dlp_dtheta = (shape - 1.0) / theta - rate;
            dlp_dtheta * param.transform_deriv(z)
        }
        Prior::Exponential { rate } => {
            if theta < 0.0 { return (lp, 0.0); }
            let dlp_dtheta = -rate;
            dlp_dtheta * param.transform_deriv(z)
        }
        // Hierarchical priors need an env-aware density AND gradient to
        // drive NUTS correctly. PGAS+NUTS with hierarchical leaves is
        // tracked as Gate 3b — needs env threaded through this function
        // signature. For Gate 3a (PMMH + hierarchical), PMMH does not
        // call this function. Until 3b lands: model compiles + PMMH
        // works + NUTS on hierarchical coords is disabled-by-infinity.
        Prior::Hierarchical(_) => return (f64::NEG_INFINITY, 0.0),
    };
    (lp, dlp_dz)
}

// ═══════════════════════════════════════════════════════════════════
// Rung state for parallel tempering
// ═══════════════════════════════════════════════════════════════════

/// Per-rung state for parallel tempering. Consolidates the 12+ parallel
/// vectors that were previously maintained separately.
struct RungState {
    params: Vec<f64>,
    transformed: Vec<f64>,
    ll: f64,
    trajectory: PGASTrajectory,
    nuts_mass: super::nuts::MassMatrix,
    nuts_step_size: f64,
    nuts_dual_avg: super::nuts::DualAveraging,
    log_proposal_sd: Vec<f64>,
    total_accepted: Vec<usize>,
    welford_n: f64,
    welford_mean: Vec<f64>,
    welford_m2: Vec<f64>,
    welford_cov: Vec<f64>,
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
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
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

    // Precompute observation→substep mapping once for the entire run.
    let obs_at_substep = build_obs_at_substep(
        observations, model.model.simulation.t_start, config.dt,
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

        current_transformed = restore_z_values(
            &state.param_names, &state.transformed, if2_params, &current_params,
        );

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
            trajectory.initial_counts.first().copied().unwrap_or(0));
        current_transformed = if2_params.iter()
            .map(|p| p.to_transformed(current_params[p.index]))
            .collect();
        start_sweep = 0;

        // Sanity check: the trajectory must have finite density at its own params
        // (before IVP mapping, which adds initial state density)
        let sanity_ll = complete_data_loglik(
            model, &trajectory, &current_params, observations,
            config.dt, obs_model, &[],  // empty IVP mappings
            &obs_at_substep,
        )?.total;
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
                if model.balance.as_ref().is_some_and(|b| b.local_int_idx == c) {
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
        config.dt, obs_model, &ivp_mappings, &obs_at_substep,
    )?.total;
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

    // NUTS state — restored from resume or initialized fresh.
    //
    // Im18 in 2026-04-19 inference review batch 2: only the cold
    // rung's NUTS state (mass matrix, step size, dual averaging,
    // acceptance counts) is persisted in ChainResumeState and
    // restored here. Heated rungs (β < 1) always start with
    // `MassMatrix::identity`, step_size = 0.1, and fresh dual
    // averaging — so every resume re-warms the heated rungs, which
    // wastes sweeps on tempered fits that resume frequently.
    //
    // A full fix requires extending ChainResumeState to hold a
    // Vec<RungNUTSState> and handling back-compat with legacy
    // single-rung resume files. Not done here; when a tempered fit
    // hits the pain point the schema upgrade is straightforward.
    let (nuts_mass_init, nuts_step_size_init, log_proposal_sd_restored,
         total_accepted_init, current_ll_restored) = if let Some((mass, ss, lpsd, ta, ll)) = resume_nuts {
        (mass, ss, lpsd, ta, Some(ll))
    } else {
        (super::nuts::MassMatrix::identity(d), 0.1, log_proposal_sd, vec![0usize; d], None)
    };

    // Per-rung state: rung 0 is cold (β=1), higher indices are hotter.
    let mut rungs: Vec<RungState> = (0..n_rungs).map(|r| {
        let step_size = if r == 0 { nuts_step_size_init } else { 0.1 };
        RungState {
            params: current_params.clone(),
            transformed: current_transformed.clone(),
            ll: current_ll,
            trajectory: trajectory.clone(),
            nuts_mass: if r == 0 { nuts_mass_init.clone() } else { super::nuts::MassMatrix::identity(d) },
            nuts_step_size: step_size,
            nuts_dual_avg: super::nuts::DualAveraging::new(step_size, 0.80),
            log_proposal_sd: log_proposal_sd_restored.clone(),
            total_accepted: if r == 0 { total_accepted_init.clone() } else { vec![0usize; d] },
            welford_n: 0.0,
            welford_mean: vec![0.0; d],
            welford_m2: vec![0.0; d],
            welford_cov: vec![0.0; d * d],
        }
    }).collect();

    let mut sweeps = Vec::new();

    // Override cold rung LL if we have a resumed value
    if let Some(ll) = current_ll_restored {
        rungs[0].ll = ll;
    }

    // Im18: make the heated-rung re-warmup visible in logs.
    // Check the restored NUTS tuple rather than `resume_from` (the
    // latter is partially moved into earlier bindings).
    if current_ll_restored.is_some() && n_rungs > 1 {
        log::info!(
            "pgas resume: restored cold rung NUTS state; heated rungs \
             (β<1) re-warm from defaults each resume. Long-running \
             tempered fits may want to avoid frequent interruption."
        );
    }

    // Swap acceptance tracking (n_rungs - 1 adjacent pairs)
    let mut swap_proposed: Vec<usize> = vec![0; n_rungs.saturating_sub(1)];
    let mut n_max_treedepth: usize = 0;
    let mut n_divergent: usize = 0;
    let mut swap_accepted: Vec<usize> = vec![0; n_rungs.saturating_sub(1)];

    if start_sweep >= config.n_sweeps {
        eprintln!("  warning: chain already completed {} sweeps (requested {}). \
                   Increase sweeps in fit.toml to continue.", start_sweep, config.n_sweeps);
    }

    // Pre-resolve rate_grad indices once for the entire run (avoids O(n_params)
    // string scans per gradient term per substep in the NUTS hot path).
    // model_to_estimated[model_param_idx] = estimated_param_idx, or None if fixed.
    let rate_grads_for_run: Vec<Vec<(usize, crate::resolved_expr::ResolvedExpr)>> = {
        let n_model_params = model.model.parameters.len();
        let mut model_to_estimated: Vec<Option<usize>> = vec![None; n_model_params];
        for (est_idx, spec) in if2_params.iter().enumerate() {
            model_to_estimated[spec.index] = Some(est_idx);
        }
        super::pgas_grad::resolve_rate_grad_for_run(
            &model.resolved.rate_grads_indexed,
            &model_to_estimated,
        )
    };

    // ── Trajectory warm-up: CSMC-only sweeps before parameter updates ──
    if config.trajectory_warmup > 0 && start_sweep == 0 {
        eprintln!("  trajectory warm-up: {} CSMC-only sweeps", config.trajectory_warmup);
        for warmup_sweep in 0..config.trajectory_warmup {
            for rung in 0..n_rungs {
                let csmc_seed = seed ^ ((warmup_sweep as u64).wrapping_mul(SEED_MIX_WARMUP))
                    ^ (rung as u64).wrapping_mul(SEED_MIX_RUNG);
                let (new_traj, _diag) = csmc_as(
                    model, &rungs[rung].params, observations, &rungs[rung].trajectory,
                    config.n_particles, config.dt, obs_model,
                    &ivp_mappings, csmc_seed, &obs_at_substep,
                )?;
                rungs[rung].trajectory = new_traj;
                rungs[rung].ll = complete_data_loglik(
                    model, &rungs[rung].trajectory, &rungs[rung].params, observations,
                    config.dt, obs_model, &ivp_mappings, &obs_at_substep,
                )?.total;
            }
            if warmup_sweep % 10 == 0 {
                eprintln!("  trajectory warm-up {}/{}: cold LL={:.1}",
                    warmup_sweep, config.trajectory_warmup, rungs[0].ll);
            }
        }
        eprintln!("  trajectory warm-up complete: cold LL={:.1}", rungs[0].ll);
    }

    for sweep in start_sweep..config.n_sweeps {
        // Per-rung accepted flags (only cold rung's is used for output)
        let mut rung_accepted: Vec<Vec<bool>> = vec![vec![false; d]; n_rungs];
        // Per-rung CSMC diagnostics (only cold rung's is used for output)
        let mut rung_csmc_diag: Vec<CSMCDiagnostics> = Vec::with_capacity(n_rungs);
        // Cold rung LL components (populated during rung loop)
        let mut cold_transition_ll = 0.0_f64;
        let mut cold_obs_ll = 0.0_f64;

        for rung in 0..n_rungs {
            let beta = betas[rung];

            // Current proposal SDs for this rung (MH only)
            let proposal_sd: Vec<f64> = rungs[rung].log_proposal_sd.iter()
                .map(|&ls| ls.exp())
                .collect();

            // ── Step 1: Update θ | X, y ──
            // For heated rungs (β < 1), scale LL and its gradient by β.
            // Prior and Jacobian are untempered.
            if has_gradients {
                let rung_traj = &rungs[rung].trajectory;

                let log_prob_and_grad = |z: &[f64]| -> (f64, Vec<f64>) {
                    let mut params = rungs[rung].params.clone();
                    for (i, spec) in if2_params.iter().enumerate() {
                        params[spec.index] = spec.from_transformed(z[i]);
                    }

                    let (ll, ll_grad_theta) = match super::pgas_grad::complete_data_loglik_grad(
                        model, rung_traj, &params, observations,
                        config.dt, obs_model, &ivp_mappings,
                        d, &rate_grads_for_run, &obs_at_substep,
                    ) {
                        Ok(r) => r,
                        Err(_) => return (f64::NEG_INFINITY, vec![0.0; d]),
                    };

                    // Temper: scale LL by β
                    let mut log_p = beta * ll;
                    let mut grad_z = vec![0.0; d];

                    for i in 0..d {
                        let theta = params[if2_params[i].index];
                        let dtheta_dz = if2_params[i].transform_deriv(z[i]);

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

                let (init_log_p, init_grad) = log_prob_and_grad(&rungs[rung].transformed);

                let nuts_config = super::nuts::NUTSConfig {
                    max_tree_depth: config.max_tree_depth,
                    step_size: rungs[rung].nuts_step_size,
                    mass_matrix: rungs[rung].nuts_mass.clone(),
                };

                let result = super::nuts::nuts_step(
                    &rungs[rung].transformed, init_log_p, &init_grad,
                    &nuts_config, &log_prob_and_grad, &mut rng,
                );

                if result.accepted {
                    rungs[rung].transformed.copy_from_slice(&result.params);
                    for (i, spec) in if2_params.iter().enumerate() {
                        rungs[rung].params[spec.index] = spec.from_transformed(rungs[rung].transformed[i]);
                    }
                    for a in &mut rung_accepted[rung] { *a = true; }
                    for t in &mut rungs[rung].total_accepted { *t += 1; }
                }
                if rung == 0 {
                    if result.tree_depth >= config.max_tree_depth {
                        n_max_treedepth += 1;
                    }
                    if result.divergent {
                        n_divergent += 1;
                    }
                }

                // Two-phase adaptation (same schedule as single-rung, per-rung state)
                let mass_adapt_end = (adapt_end as f64 * MASS_ADAPT_FRAC) as usize;

                if sweep < mass_adapt_end {
                    rungs[rung].nuts_step_size = rungs[rung].nuts_dual_avg.update(result.mean_accept_prob);

                    rungs[rung].welford_n += 1.0;
                    let old_mean = rungs[rung].welford_mean.clone();
                    for i in 0..d {
                        let delta = rungs[rung].transformed[i] - rungs[rung].welford_mean[i];
                        rungs[rung].welford_mean[i] += delta / rungs[rung].welford_n;
                        let delta2 = rungs[rung].transformed[i] - rungs[rung].welford_mean[i];
                        rungs[rung].welford_m2[i] += delta * delta2;
                    }
                    for i in 0..d {
                        for j in 0..d {
                            rungs[rung].welford_cov[i * d + j] +=
                                (rungs[rung].transformed[i] - old_mean[i])
                                * (rungs[rung].transformed[j] - rungs[rung].welford_mean[j]);
                        }
                    }
                } else if sweep == mass_adapt_end {
                    if rungs[rung].welford_n > 10.0 {
                        if config.dense_mass {
                            let mut cov = vec![0.0; d * d];
                            for i in 0..d {
                                for j in 0..d {
                                    cov[i * d + j] = rungs[rung].welford_cov[i * d + j] / (rungs[rung].welford_n - 1.0);
                                }
                            }
                            rungs[rung].nuts_mass = super::nuts::MassMatrix::dense_from_covariance(&cov, d);
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
                                (rungs[rung].welford_m2[i] / (rungs[rung].welford_n - 1.0)).max(1e-10)
                            ).collect();
                            if rung == 0 {
                                eprintln!("  diagonal mass matrix estimated (sweep {}):", sweep);
                                for (i, spec) in if2_params.iter().enumerate() {
                                    eprintln!("    {:12} sd={:.6}", spec.name, variances[i].sqrt());
                                }
                            }
                            rungs[rung].nuts_mass = super::nuts::MassMatrix::diagonal(variances);
                        }
                    }
                    rungs[rung].nuts_step_size = 0.1;
                    rungs[rung].nuts_dual_avg = super::nuts::DualAveraging::new(rungs[rung].nuts_step_size, 0.80);
                } else if sweep < adapt_end {
                    rungs[rung].nuts_step_size = rungs[rung].nuts_dual_avg.update(result.mean_accept_prob);
                } else if sweep == adapt_end && rung == 0 {
                    rungs[rung].nuts_step_size = rungs[rung].nuts_dual_avg.final_step_size();
                    eprintln!("  NUTS fully adapted (sweep {}):", sweep);
                    eprintln!("    final step_size: {:.6}", rungs[rung].nuts_step_size);
                } else if sweep == adapt_end {
                    rungs[rung].nuts_step_size = rungs[rung].nuts_dual_avg.final_step_size();
                }
            } else {
                // MH-within-Gibbs: one-at-a-time random walk proposals
                // For heated rungs, scale LL by β in the MH ratio.
                for i in 0..d {
                    let spec = &if2_params[i];
                    let z_old = rungs[rung].transformed[i];
                    let z_new = z_old + proposal_sd[i] * rng.normal();
                    let theta_new = spec.from_transformed(z_new);

                    let mut proposed_params = rungs[rung].params.clone();
                    proposed_params[spec.index] = theta_new;

                    let proposed_ll = complete_data_loglik(
                        model, &rungs[rung].trajectory, &proposed_params, observations,
                        config.dt, obs_model, &ivp_mappings, &obs_at_substep,
                    )?.total;

                    let proposed_log_prior_i = priors[i].log_density(theta_new, z_new);
                    let current_log_prior_i = priors[i].log_density(
                        rungs[rung].params[spec.index], z_old,
                    );
                    let proposed_log_jac_i = spec.log_jacobian(z_new);
                    let current_log_jac_i = spec.log_jacobian(z_old);

                    // Temper: scale LL difference by β, prior + Jacobian untempered
                    let log_alpha = beta * (proposed_ll - rungs[rung].ll)
                                  + (proposed_log_prior_i - current_log_prior_i)
                                  + (proposed_log_jac_i - current_log_jac_i);

                    if log_alpha.is_finite() && rng.uniform().ln() < log_alpha {
                        rungs[rung].params[spec.index] = theta_new;
                        rungs[rung].transformed[i] = z_new;
                        rungs[rung].ll = proposed_ll;
                        rung_accepted[rung][i] = true;
                        rungs[rung].total_accepted[i] += 1;
                    }

                    // Robbins-Monro adaptation (per-rung)
                    if sweep < adapt_end {
                        let gamma_rm = ADAPT_C / (1.0 + sweep as f64).sqrt();
                        let acc_indicator = if rung_accepted[rung][i] { 1.0 } else { 0.0 };
                        rungs[rung].log_proposal_sd[i] += gamma_rm * (acc_indicator - TARGET_ACCEPTANCE);
                        rungs[rung].log_proposal_sd[i] = rungs[rung].log_proposal_sd[i].clamp(-20.0, 5.0);
                    }
                }
            }

            // ── Step 2: Update X | θ, y via CSMC-AS ──
            // CSMC always runs at β=1 — the trajectory must match the data.
            // Multiple CSMC sweeps per NUTS step improve trajectory convergence
            // on long time series where ancestor sampling is the bottleneck.
            let mut csmc_diag = CSMCDiagnostics {
                trajectory_renewal: 0.0, n_degenerate: 0, n_substeps: 0,
            };
            for csmc_rep in 0..config.csmc_sweeps_per_nuts {
                let csmc_seed = seed ^ ((sweep as u64 + 1).wrapping_mul(SEED_MIX_KNUTH))
                    ^ (rung as u64).wrapping_mul(SEED_MIX_RUNG)
                    ^ (csmc_rep as u64).wrapping_mul(SEED_MIX_REP);
                let (new_trajectory, diag) = csmc_as(
                    model, &rungs[rung].params, observations, &rungs[rung].trajectory,
                    config.n_particles, config.dt, obs_model,
                    &ivp_mappings, csmc_seed, &obs_at_substep,
                )?;
                rungs[rung].trajectory = new_trajectory;
                csmc_diag = diag;
            }

            // Recompute complete-data LL at β=1 (untempered, for swap proposals)
            let ll_components = complete_data_loglik(
                model, &rungs[rung].trajectory, &rungs[rung].params, observations,
                config.dt, obs_model, &ivp_mappings, &obs_at_substep,
            )?;
            rungs[rung].ll = ll_components.total;

            rung_csmc_diag.push(csmc_diag);

            // Store components for cold rung output
            if rung == 0 {
                cold_transition_ll = ll_components.transition;
                cold_obs_ll = ll_components.observation;
            }
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
                let log_alpha = (betas[i] - betas[j]) * (rungs[i].ll - rungs[j].ll);

                if log_alpha >= 0.0 || rng.uniform().ln() < log_alpha {
                    swap_accepted[i] += 1;

                    // Swap all state between rungs i and j
                    rungs.swap(i, j);
                    rung_accepted.swap(i, j);
                }

                i += 2;
            }
        }

        // ── Cold rung (index 0) output ──
        // Log adapted proposal SDs at end of burn-in (cold rung only)
        if sweep + 1 == adapt_end {
            eprintln!("  proposal SD adapted (end of burn-in):");
            for (i, spec) in if2_params.iter().enumerate() {
                let acc_rate = rungs[0].total_accepted[i] as f64 / (sweep + 1) as f64;
                eprintln!("    {:12} sd={:.6} acc={:.0}%",
                    spec.name, rungs[0].log_proposal_sd[i].exp(), acc_rate * 100.0);
            }
            eprintln!("  trajectory renewal: {:.1}%", rung_csmc_diag[0].trajectory_renewal * 100.0);

            // NUTS diagnostics (Stan-style warnings)
            if has_gradients {
                let pct_maxdepth = n_max_treedepth as f64 / (sweep + 1) as f64 * 100.0;
                if n_max_treedepth > 0 {
                    eprintln!("  WARNING: {}/{} sweeps ({:.0}%) hit max_treedepth={}. \
                        Consider increasing max_treedepth or reparameterizing.",
                        n_max_treedepth, sweep + 1, pct_maxdepth, config.max_tree_depth);
                }
                if n_divergent > 0 {
                    eprintln!("  WARNING: {} divergent transitions during burn-in. \
                        Consider reducing step size or reparameterizing.",
                        n_divergent);
                }
            }

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

        // Periodic swap rate report (every 500 sweeps during sampling)
        if n_rungs > 1 && sweep > 0 && sweep % 500 == 0 {
            let rates: Vec<String> = (0..n_rungs - 1).map(|i| {
                let rate = if swap_proposed[i] > 0 {
                    swap_accepted[i] as f64 / swap_proposed[i] as f64
                } else { 0.0 };
                format!("{:.0}%", rate * 100.0)
            }).collect();
            eprintln!("  sweep {}: swap rates [{}]", sweep, rates.join(", "));
        }

        let cold_proposal_sd: Vec<f64> = rungs[0].log_proposal_sd.iter()
            .map(|&ls| ls.exp())
            .collect();

        let sweep_result = PGASSweep {
            params: rungs[0].params.clone(),
            log_complete_data_ll: rungs[0].ll,
            accepted: rung_accepted[0].clone(),
            csmc_diag: rung_csmc_diag[0].clone(),
            proposal_sds: cold_proposal_sd,
            transition_ll: cold_transition_ll,
            obs_ll: cold_obs_ll,
        };

        if let Some(cb) = on_sweep {
            cb(sweep, &sweep_result, &rungs[0].trajectory);
        }

        // Record (respecting burn-in and thinning)
        if sweep >= config.burn_in && (sweep - config.burn_in).is_multiple_of(config.thin) {
            sweeps.push(sweep_result);
        }
    }

    let acceptance_rates: Vec<f64> = rungs[0].total_accepted.iter()
        .map(|&n| n as f64 / config.n_sweeps as f64)
        .collect();

    let resume_state = ChainResumeState {
        config_hash,
        completed_sweeps: config.n_sweeps,
        params: rungs[0].params.clone(),
        transformed: rungs[0].transformed.clone(),
        param_names: if2_params.iter().map(|p| p.name.clone()).collect(),
        trajectory: rungs[0].trajectory.clone(),
        mass_matrix: rungs[0].nuts_mass.clone(),
        nuts_step_size: rungs[0].nuts_step_size,
        log_proposal_sd: rungs[0].log_proposal_sd.clone(),
        total_accepted: rungs[0].total_accepted.clone(),
        current_ll: rungs[0].ll,
    };

    Ok(PGASResult {
        sweeps,
        final_trajectory: rungs[0].trajectory.clone(),
        acceptance_rates,
        resume_state,
    })
}
