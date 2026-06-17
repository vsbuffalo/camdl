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
use rayon::prelude::*;

use crate::chain_binomial::{StepScratch, step_one, RATE_EPSILON};
use crate::compiled_model::CompiledModel;
use crate::rng::StatefulRng;
use crate::error::SimError;
use crate::inference::obs_loglik::{poisson_logpmf, binom_logpmf};
use crate::inference::particle_filter::Observation;
use crate::inference::resampling::systematic_resample;
use crate::inference::pmmh::Prior;
use crate::inference::types::{EstimatedParam, RESAMPLE_RNG_STREAM, init_particle_rngs, restore_z_values};
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::eval_resolved;
use crate::schedule::{Cursor, Schedule, StepPolicy};
use crate::state::{IntState, RealState};

/// Collect names of every `Param` referenced by an expression tree.
/// Used by the narrowed C1 preflight gate (gh#76 follow-up) to detect
/// estimated parameters reachable through a parametric `DerivedExpr`
/// projection — the one obs-gradient arm that remains uncovered.
fn collect_param_refs(e: &ir::expr::Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        ir::expr::Expr::Param(p) => { out.insert(p.param.clone()); }
        ir::expr::Expr::BinOp(w) => {
            collect_param_refs(&w.bin_op.left,  out);
            collect_param_refs(&w.bin_op.right, out);
        }
        ir::expr::Expr::UnOp(w) => collect_param_refs(&w.un_op.arg, out),
        ir::expr::Expr::Cond(w) => {
            collect_param_refs(&w.cond.pred,  out);
            collect_param_refs(&w.cond.then,  out);
            collect_param_refs(&w.cond.else_, out);
        }
        ir::expr::Expr::PopSum(_) | ir::expr::Expr::Pop(_)
        | ir::expr::Expr::Const(_) | ir::expr::Expr::Time(_)
        | ir::expr::Expr::Dt(_)   | ir::expr::Expr::TimeFunc(_)
        | ir::expr::Expr::Projected(_)
        // A per-observation aux column is data (∂/∂θ = 0): no param to collect.
        | ir::expr::Expr::ObsColumnRef(_) => {}
        ir::expr::Expr::TableLookup(w) => {
            for ix in &w.table_lookup.indices {
                collect_param_refs(ix, out);
            }
        }
        ir::expr::Expr::UncheckedDim(w) => collect_param_refs(&w.unchecked_dim.inner, out),
        // A param reachable only through a Reduce must still be collected, or the
        // gate would let an unsupported obs-param through with a silent zero gradient.
        ir::expr::Expr::Reduce(w) => {
            for t in &w.reduce { collect_param_refs(t, out); }
        }
        // Hoisted bindings are param-free; nothing to collect.
        ir::expr::Expr::BindingRef(_) => {}
    }
}

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
    /// Observation-time alignment for the substep grid (Stage 3).
    /// `Snap` (default): round observation times onto the uniform `dt` grid
    /// — the historical PGAS behavior. `Exact`: tile each observation window
    /// with full-`dt` steps plus a shortened remainder landing exactly on the
    /// obs time (`build_substep_grid`). The CLI keeps this `Snap` until the
    /// exact path's recovery evidence lands and the default is flipped.
    pub step_policy: StepPolicy,
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
    /// Realized start-time of this substep — the time `step_one` froze
    /// propensities at. The single source of truth for the density's time
    /// argument; consumers read this instead of recomputing `t_start + s*dt`.
    /// Under `snap` alignment it equals `t_start + s*dt`; under `exact`
    /// (Stage 3) it is the window-tiled realized time.
    pub t0: f64,
    /// Realized duration of this substep — the magnitude that enters every
    /// density/gradient term (`p = 1 - exp(-rate*dt_substep)`,
    /// `shape = dt_substep/σ²`, …). Under `snap` it equals the run `dt`;
    /// under `exact` (Stage 3) it is the (possibly shortened) tiled step.
    pub dt_substep: f64,
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
    /// gh#audit-C7. NUTS divergent transitions across the full run
    /// (burn-in + sampling). Stan-style diagnostic: any post-burn-in
    /// divergence is a correctness signal worth gating on.
    pub n_divergent_total: usize,
    /// gh#audit-C7. NUTS divergent transitions accumulated only over
    /// post-burn-in sweeps. The Stan-canonical surface — burn-in
    /// divergences are expected during step-size adaptation.
    pub n_divergent_post_burn: usize,
    /// gh#audit-C7. Sweeps that hit max_treedepth across the full run.
    pub n_max_treedepth_total: usize,
    /// gh#audit-C7. Sweeps that hit max_treedepth post-burn-in.
    pub n_max_treedepth_post_burn: usize,
    /// gh#audit-C7 / M18. Per-adjacent-rung swap acceptance rates
    /// (length n_rungs - 1; empty when n_rungs == 1). Adjacent-pair
    /// rate `swap_acceptance_rates[i]` = accepted_{i,i+1} /
    /// proposed_{i,i+1}. Used to wire DiagnosticKind::LowSwapRate
    /// (audit H4): rate < 0.10 on tempered chains is a sign the
    /// temperature ladder is too sparse.
    pub swap_acceptance_rates: Vec<f64>,
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

/// Build the substep→observation index mapping (Snap policy).
///
/// Rejects sub-`dt` observation collisions (M2). Two distinct, strictly-
/// increasing observation times closer together than `dt` round to the same
/// substep index (`interval_steps` is round-to-nearest), so they would collide
/// on the same `ObsAtSubstep` key — and the last-wins `map.insert` would
/// silently drop one observation from the PGAS likelihood, biasing the
/// posterior. The dt-independent increasing-times guard
/// (`validate_obs_times_increasing`) does not catch this, so we detect the
/// collision here, at grid construction, with an actionable message.
pub fn build_obs_at_substep(
    observations: &[Observation],
    t_start: f64,
    dt: f64,
) -> Result<ObsAtSubstep, crate::error::SimError> {
    let mut map = ObsAtSubstep::new();
    // Track which observation last claimed each substep so a collision can
    // name BOTH offending times in the diagnostic.
    let mut claimant: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    for (obs_idx, obs) in observations.iter().enumerate() {
        let s = crate::time::interval_steps(t_start, obs.time, dt);
        if s > 0 {
            if let Some(prev_time) = claimant.insert(s - 1, obs.time) {
                return Err(crate::error::SimError::Validation(format!(
                    "observation times {} and {} are closer than dt = {} and round \
                     to the same substep ({}); under snap obs-alignment they collide \
                     and one observation would be silently dropped from the \
                     likelihood. Use a dt finer than the smallest observation gap, \
                     run with --obs-alignment exact, or remove the closer observation.",
                    prev_time, obs.time, dt, s - 1
                )));
            }
            map.insert(s - 1, obs_idx);
        }
    }
    Ok(map)
}

/// The realized substep grid for one PGAS run: per-substep `(t0, dt_substep)`
/// plus the substep→observation-index map. Built once per run from the
/// observation times, the nominal `dt`, and the alignment policy; the reference
/// trajectory, the CSMC free particles, and the density consumers all tile time
/// against this one grid, so they agree by construction.
#[derive(Clone, Debug, PartialEq)]
pub struct SubstepGrid {
    /// `(t0, dt_substep)` for each substep, chronological. `t0` is computed
    /// drift-free via `Schedule::substep_time` (`window_start + s·dt`, one
    /// multiply — never accumulated), so a time-inhomogeneous rate samples
    /// bounded-error instants.
    pub steps: Vec<(f64, f64)>,
    /// substep index → observation index: the substep whose end coincides with
    /// that observation time (where the likelihood is scored).
    pub obs_at_substep: ObsAtSubstep,
    /// substep index → scheduled-effect-boundary index (into a
    /// [`crate::intervention::TimelineEffects`]): the substep whose end lands on
    /// that scheduled intervention's fire time, where the producer fires it
    /// CURSOR-keyed (gh#216). Empty under `Snap` (effects fire on the `round(t/dt)`
    /// key in the producer's `due_effects`); populated only under `Exact`.
    pub effect_at_substep: ObsAtSubstep,
}

/// Build the substep grid over `[t_start, last_obs]` under the alignment policy.
/// The Exact arm materializes the shared [`Schedule::substeps`] walk — the SAME
/// drift-free inner walk the bootstrap PF / IF2 / correlated-PF iterate (gh#233:
/// one walk, two consumers) — instead of hand-rolling a second tiling. The
/// `Schedule` is the single source of truth for where boundaries fall; the
/// negligible-step floor is the shared `schedule::MIN_STEP_EPS` (was PGAS's own
/// `GRID_STEP_EPS = 1e-12`, unified down).
///
/// * `Snap`: the uniform grid (`t_start + s·dt`, full `dt`) with the obs map from
///   [`build_obs_at_substep`] (obs rounded onto the grid) — the historical PGAS
///   behavior, byte-identical.
/// * `Exact`: loop obs windows over `Schedule::substeps`. Each window yields
///   drift-free `t0 = substep_time(window_start, s)` substeps clipped to its obs;
///   the obs is scored on the window's final clipped substep, effects on the
///   substep the iterator signals. At dt=1.0 (and any window that is an integer
///   multiple of `dt`) this is bit-identical to `Snap`; at non-power-of-2 `dt` the
///   final step of an on-grid window differs from `Snap` by ≤1 ULP — the
///   sanctioned EXACT-stepper drift, bounded to one window (substep-time
///   proposal), in exchange for landing *exactly* on every observation.
pub fn build_substep_grid(
    t_start: f64,
    dt: f64,
    observations: &[Observation],
    effect_times: &[f64],
    policy: StepPolicy,
) -> Result<SubstepGrid, SimError> {
    let last_obs = observations.last().map(|o| o.time).unwrap_or(t_start);
    match policy {
        StepPolicy::Snap => {
            // Snap: effects fire on the `round(t/dt)` key inside the producer's
            // `due_effects`, off this uniform grid — so no effect boundaries are
            // registered and `effect_at_substep` stays empty (byte-identical).
            let n = crate::time::interval_steps(t_start, last_obs, dt);
            let steps = (0..n).map(|s| (t_start + s as f64 * dt, dt)).collect();
            let obs_at_substep = build_obs_at_substep(observations, t_start, dt)?;
            Ok(SubstepGrid { steps, obs_at_substep, effect_at_substep: ObsAtSubstep::new() })
        }
        StepPolicy::Exact => {
            let obs_times: Vec<f64> = observations.iter().map(|o| o.time).collect();
            // gh#216: register the scheduled-effect boundaries so the Exact walk
            // LANDS exactly on each (even an on-grid effect that off-grid obs would
            // otherwise step past), and record which substep fires it so the
            // producer fires CURSOR-keyed. Off-grid effect times are refused
            // upstream (`guard_exact_offgrid_effect_time`).
            let schedule =
                Schedule::new(dt, last_obs, dt, StepPolicy::Exact, Vec::new(), effect_times.to_vec())
                    .with_obs(obs_times);
            // PGAS materializes the SAME inner walk the bootstrap PF / IF2 /
            // correlated-PF iterate (`Schedule::substeps`), instead of hand-rolling
            // a second copy (gh#233 — one walk, two consumers). We loop obs windows
            // and collect each window's drift-free substeps into the flat grid; the
            // obs lands on the window's last substep, effects on the substep the
            // iterator signals via `fired`. The effect cursor carries monotonically
            // across windows (PGAS's single-cursor convention: an effect fires
            // exactly once, on the substep landing on its boundary).
            let mut steps: Vec<(f64, f64)> = Vec::new();
            let mut obs_at_substep = ObsAtSubstep::new();
            let mut effect_at_substep = ObsAtSubstep::new();
            let mut cur = Cursor::default();
            let mut window_start = t_start;
            let mut idx = 0usize;
            while let Some(obs_t) = schedule.obs_time(&cur) {
                let wcur =
                    Cursor { obs_idx: cur.obs_idx, effect_idx: cur.effect_idx, ..Default::default() };
                let mut last_idx = None;
                let mut fired_in_window = 0usize;
                for (t0, step_dt, fired) in schedule.substeps(wcur, window_start) {
                    steps.push((t0, step_dt));
                    if let Some(eff_idx) = fired {
                        let prev = effect_at_substep.insert(idx, eff_idx);
                        debug_assert!(
                            prev.is_none(),
                            "exact grid: substep {idx} claimed twice (effect collision)"
                        );
                        fired_in_window += 1;
                    }
                    last_idx = Some(idx);
                    idx += 1;
                }
                match last_idx {
                    // The obs is scored on the window's last (boundary-clipped)
                    // substep. Coincident obs are rejected upstream, so the key is
                    // fresh; a coincident effect+obs lands on the same substep
                    // (`fired` already recorded it there) — matching the old
                    // single-loop walk.
                    Some(li) => {
                        let prev = obs_at_substep.insert(li, cur.obs_idx);
                        debug_assert!(
                            prev.is_none(),
                            "exact grid: substep {li} claimed twice (obs collision)"
                        );
                    }
                    // A leading window coincident with t_start (obs(0) == t_start)
                    // yields no substep; the old whole-run walk broke here too.
                    None => break,
                }
                window_start = obs_t; // re-anchor at the EXACT obs time
                cur.effect_idx += fired_in_window;
                cur.pass_obs();
            }
            Ok(SubstepGrid { steps, obs_at_substep, effect_at_substep })
        }
    }
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
                // gh#80: this branch fires *correctly* during CSMC ancestor
                // sampling whenever a free particle's pre-step state has
                // n_src=0 (or otherwise zero rate) for a transition that
                // fired in the reference's flow record. The conditional
                // density IS mathematically zero and the particle is
                // legitimately excluded from the ancestor categorical.
                // log::warn! was misleading here — the previous text
                // suggested adding an `iota` term, which is correct only
                // when the *trajectory's own* (counts_before, flows) pair
                // disagrees, not for ancestor-sampling state/flow pairings
                // across particles. Demoted to debug! to keep production
                // logs clean; the math is unchanged.
                log::debug!(
                    "log_transition_density_substep: transition {} has rate=0 \
                     against this counts_before but flow={} in the scored record \
                     — returning -inf (legitimate density of zero, e.g. ancestor \
                     sampling pairing a particle's state with the reference's flows).",
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
    // gh#audit-H3: stable (p, q) primitive with the clamped variant
    // (PGAS hot path needs strict-interior p for the binomial density
    // / NUTS gradient).
    let (p_total, _q) = super::numerics::prob_q_from_rate_dt_clamped(total_rate, dt, 1e-15);
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
            let p_split = (eff_rate / rate_remaining).clamp(1e-15, 1.0 - 1e-15);
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
    eval_propensities(model, &int_s, &real_s, params, t, dt, &mut propensities)?;

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt, projected: None, aux: None, int_float_override: None,
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

    // Cumulative flows since last observation (per-transition tally; UNCHANGED
    // lifecycle). Phase 2a adds the per-Interval-stream persistent `acc` bin,
    // folded once per observation interval and reset per-stream.
    let mut cum_flows = vec![0u64; n_tr];
    let mut acc = vec![0u64; obs_model.n_interval_streams()];
    let t_start = model.model.simulation.t_start;
    // Exact-tiling invariant (debug): the realized (t0, dt_substep) records
    // partition the run contiguously, each duration in (0, dt]. This is the
    // single source of truth the consumers read; it replaces the 2b snap
    // invariant (rec.t0 == t_start+s·dt, rec.dt_substep == dt), which a shortened
    // exact substep violates by design. `dt` is the nominal step (the upper
    // bound). Contiguity catches a producer that mispopulates a record.
    let mut prev_end = t_start;

    for s in 0..n_substeps {
        let rec = &trajectory.substeps[s];
        if cfg!(debug_assertions) {
            debug_assert!(rec.dt_substep > 0.0 && rec.dt_substep <= dt + 1e-9,
                "substep {s}: dt_substep {} not in (0, dt={dt}]", rec.dt_substep);
            debug_assert!((rec.t0 - prev_end).abs() < 1e-9,
                "substep {s}: t0 {} not contiguous with previous end {prev_end}", rec.t0);
            prev_end = rec.t0 + rec.dt_substep;
        }
        let t = rec.t0;
        let dt_s = rec.dt_substep;
        // Use the pre-step snapshot stored in the record — this is the
        // exact state step_one evaluated propensities from.
        let counts_before = &rec.counts_before;

        // Transition density
        let td = log_transition_density_substep(
            model, counts_before, &rec.flows, &rec.gammas, params, t, dt_s,
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
            //
            // Evaluate σ² at the real start-of-step state (`counts_before`),
            // mirroring the three sibling sites: step_one (chain_binomial.rs),
            // log_transition_density_substep (above), and
            // gamma_density_value_and_grad_substep (pgas_grad.rs). σ² is
            // state-independent today — a compile-time guard in
            // compiled_model.rs rejects any overdispersion expression that
            // references compartment state — so this state choice is currently
            // a no-op for the σ² value. Using `counts_before` (not a zeroed
            // scratch) keeps this site byte-identical to its siblings and
            // defensive against any future relaxation of that guard.
            let n_int_local = model.int_local_to_global.len();
            let mut int_s_local = IntState::new(n_int_local);
            int_s_local.counts.copy_from_slice(&rec.counts_before);
            let real_s_local = RealState::new(model.real_local_to_global.len());
            let ctx = EvalCtx {
                model, int_s: &int_s_local, real_s: &real_s_local,
                params, t, dt: dt_s,
                projected: None, aux: None, int_float_override: None,
            };
            let mut gamma_idx_local = 0;
            for &(src_local, ref group) in &model.source_groups {
                let n_src = rec.counts_before[src_local].max(0);
                if n_src == 0 { continue; }
                // Recompute propensities for rate check (same start-of-step state).
                let mut local_props = vec![0.0; n_tr];
                let _ = eval_propensities(model, &int_s_local, &real_s_local,
                    params, ctx.t, dt_s, &mut local_props);
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
                            let shape = dt_s / sigma_sq;
                            let scale = sigma_sq / dt_s;
                            // log Gamma(g; shape, scale). Shared with the gradient
                            // path's energy via one helper so the two agree
                            // f64-exactly (gh#197 / the spine oracle).
                            transition_ll +=
                                crate::inference::obs_loglik::gamma_multiplier_log_density(
                                    shape, scale, g);
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
            // FOLD (Phase 2a): close this interval's per-transition `cum_flows`
            // into each Interval stream's persistent `acc` bin BEFORE scoring;
            // score reads the per-stream `acc`.
            obs_model.fold_into_acc(&cum_flows, &mut acc);
            let obs_ll = obs_model.log_likelihood_from_flows_and_counts(
                &acc, &rec.counts_after, obs_idx, params);
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
            // `cum_flows` blanket-zeroed (unchanged); the per-stream `acc` bins
            // per-stream — only Interval streams scheduled at THIS union index.
            cum_flows.fill(0);
            obs_model.reset_due_acc(obs_idx, &mut acc);
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

/// The scheduled-effect firing plan a PGAS producer fires by (gh#216): `None`
/// selects the Snap `round(t/dt)` whole-batch path; `Some((effect_at_substep,
/// batches))` selects cursor-keyed firing — `effect_at_substep[s]` indexes
/// `batches` (the [`crate::intervention::TimelineEffects`] per-boundary lists).
pub type EffectFiring<'a> = Option<(&'a ObsAtSubstep, &'a [Vec<usize>])>;

/// Fill `out` with the effects firing at the boundary `t_end` for producer
/// substep `s`. `None` (Snap): the whole batch on the `round(t/dt)` key
/// ([`crate::effects::due_effects`]). `Some(..)` (Exact): the effects the
/// timeline landed at this substep, CURSOR-keyed, split by kind via
/// [`crate::effects::split_due_batch`]. PGAS still rejects always-active events
/// under Exact (the residual guard below), so in practice only scheduled
/// interventions reach the `Some` branch here. step_one then applies `out`.
fn fill_producer_batch(
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    t_end: f64,
    grid_dt: f64,
    s: usize,
    firing: EffectFiring<'_>,
    out: &mut crate::schedule::EffectBatch,
) {
    match firing {
        None => crate::effects::due_effects(model, fire_steps, t_end, grid_dt, out),
        Some((effect_at_substep, batches)) => {
            // Exact: every effect is cursor-keyed from the timeline. Split the
            // boundary's batch by kind (events at PROPOSE / interventions at
            // INTERVENE); empty off a boundary. Always-active events under Exact
            // PGAS are still rejected upstream (the residual guard below), so in
            // practice `batches` carries only scheduled interventions here — but
            // routing through the shared `split_due_batch` keeps PGAS on the same
            // firing path as the other cells (no `due_events` round key).
            out.clear();
            if let Some(&eff_idx) = effect_at_substep.get(&s) {
                crate::effects::split_due_batch(model, &batches[eff_idx], out);
            }
        }
    }
}

/// Simulate a forward trajectory recording per-substep detail, on the uniform
/// `dt` grid over `[t_start, t_end]`. Used to initialize the reference trajectory
/// for snap-aligned PGAS and by the gradient/density gates. Thin wrapper over
/// [`simulate_reference_on_grid`] with the uniform grid — byte-identical to the
/// pre-2c loop (`t0 = t_start + s·dt`, `dt_substep = dt`).
pub fn simulate_reference(
    model: &CompiledModel,
    params: &[f64],
    t_end: f64,
    dt: f64,
    rng: &mut StatefulRng,
) -> Result<PGASTrajectory, SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = crate::time::interval_steps(t_start, t_end, dt);
    let grid: Vec<(f64, f64)> = (0..n_substeps).map(|s| (t_start + s as f64 * dt, dt)).collect();
    // Snap (uniform grid): effects fire on the round(t/dt) key in the producer.
    simulate_reference_on_grid(model, params, dt, &grid, None, rng)
}

/// Simulate a forward reference trajectory over an explicit substep grid
/// (`(t0, dt_substep)` per substep, from [`build_substep_grid`]). Each substep
/// freezes propensities at `t0` and advances by `dt_substep`; the realized times
/// are recorded so the density consumers (and CSMC free particles, via the
/// reference) tile against the same grid. `dt` is the nominal step, used only to
/// resolve `fire_steps` (event step indices). The substep loop and RNG draw
/// order are identical to the legacy uniform loop, so a uniform grid produces a
/// byte-identical trajectory.
pub fn simulate_reference_on_grid(
    model: &CompiledModel,
    params: &[f64],
    dt: f64,
    grid: &[(f64, f64)],
    firing: EffectFiring<'_>,
    rng: &mut StatefulRng,
) -> Result<PGASTrajectory, SimError> {
    let (init_int, _) = model.initial_state(params)?;
    let n_tr = model.model.transitions.len();

    // gh#53: resolve fire_steps once at the runtime dt. Used to fill the per-
    // substep effect batch step_one applies (gh#216): the `round(t/dt)` whole
    // batch under Snap, or the `grid_dt`-keyed EVENT half under Exact (scheduled
    // interventions come cursor-keyed from `firing`).
    let fire_steps = model.resolve_fire_steps(dt, params);

    let mut counts = init_int.counts.clone();
    let mut scratch = StepScratch::new(model);
    let mut substeps = Vec::with_capacity(grid.len());
    // KNOWN LIMITATION (docs/dev/incidents/2026-06-07-chain-binomial-stale-
    // real-state.md, §inference scope): PGAS tracks integer counts only — it
    // does not advance the real reservoir (no RK4 step here). We pass a zeroed
    // RealState so rates that couple to a real compartment see 0. For real-free
    // models (n_real == 0) this is empty and byte-identical to before. Fitting
    // real-coupled models on PGAS is part of the separate, larger inference fix.
    let mut real = crate::state::RealState::new(model.real_local_to_global.len());

    for (s, &(t0, dt_s)) in grid.iter().enumerate() {
        let mut flows = vec![0u64; n_tr];
        scratch.gamma_used.clear();

        let counts_before = counts.clone();
        // Populate the due batch step_one applies (gh#216). `dt` is the nominal
        // grid the firing keys on; `dt_s` is the realized (possibly clipped) step.
        fill_producer_batch(model, &fire_steps, t0 + dt_s, dt, s, firing, &mut scratch.effect_batch);
        step_one(model, &mut counts, &mut flows, &mut real, params, t0, dt_s, rng, &mut scratch)?;

        // Verify: density evaluation of this record won't produce k > n.
        // This catches state/flow mismatches before they cause -inf later.
        if cfg!(debug_assertions) {
            let verify_td = log_transition_density_substep(
                model, &counts_before, &flows, &scratch.gamma_used, params, t0, dt_s,
            );
            if let Ok(td) = verify_td {
                debug_assert!(td.is_finite(),
                    "simulate_reference: density is -inf at substep {} (t={:.3}, dt={:.3}) \
                     despite matching state. counts_before={:?}, flows={:?}",
                    s, t0, dt_s, &counts_before, &flows);
            }
        }

        substeps.push(SubstepRecord {
            counts_before,
            counts_after: counts.clone(),
            flows,
            gammas: scratch.gamma_used.clone(),
            t0,
            dt_substep: dt_s,
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
    firing: EffectFiring<'_>,
) -> Result<(PGASTrajectory, CSMCDiagnostics), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = reference.substeps.len();
    let n_tr = model.model.transitions.len();
    let j_ref = n_particles - 1; // reference particle is the last slot

    // gh#53: resolve fire_steps once at the runtime dt for the
    // free-particle propagation step_one calls below.
    let fire_steps = model.resolve_fire_steps(dt, params);

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

    // KNOWN LIMITATION (docs/dev/incidents/2026-06-07-chain-binomial-stale-
    // real-state.md, §inference scope): CSMC free particles track integer
    // counts only — no real reservoir is advanced (no RK4 step in the loop
    // below). Per-particle zeroed RealStates make rates coupling to a real
    // compartment read 0. For real-free models (n_real == 0) these are empty
    // and byte-identical to before. Real-coupled fits need the larger fix.
    let n_real = model.real_local_to_global.len();
    let mut particle_reals: Vec<crate::state::RealState> = (0..n_particles)
        .map(|_| crate::state::RealState::new(n_real))
        .collect();

    // Cumulative flows since last observation (per-transition tally; UNCHANGED
    // lifecycle). Phase 2a adds the per-particle per-Interval-stream persistent
    // `acc` bin, folded once per observation interval and reset per-stream. It
    // travels with the particle at resampling exactly like `cum_flows`.
    let mut cum_flows: Vec<Vec<u64>> = (0..n_particles)
        .map(|_| vec![0u64; n_tr])
        .collect();
    let n_acc = obs_model.n_interval_streams();
    let mut acc: Vec<Vec<u64>> = (0..n_particles)
        .map(|_| vec![0u64; n_acc])
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
        // Tile against the grid carried by the reference trajectory (built once
        // in run_pgas); every particle shares it, so free particles and the
        // reference advance over identical (t0, dt_substep). Under snap these are
        // (t_start + s·dt, dt) — byte-identical to the pre-2c loop.
        let t = reference.substeps[s].t0;
        let step_dt = reference.substeps[s].dt_substep;

        // gh#audit-H8. Cache the pre-resample particle state for
        // ancestor sampling. The previous code saved prev_counts AFTER
        // the resampling shuffle (line 868-871), which categoricalised
        // the ancestor weight over a post-resample-relabelled ensemble
        // rather than the canonical pre-step ensemble. On observation-
        // tight steps with heterogeneous pre-step states (spatial
        // models with very different patch prevalences), the wrong
        // ancestor index could be selected. The IM6 fix at line 925
        // dropped log_weights from the sum to mask part of the issue,
        // but the state mismatch persisted. Capturing the pre-resample
        // counts here closes that loop.
        let prev_counts_for_ancestor: Vec<Vec<i64>> = counts.clone();

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
            // Phase 2a: the per-stream `acc` bins travel with the particle,
            // following EXACTLY the `cum_flows` resampling (reference kept,
            // free particles take their ancestor's bins).
            let mut new_acc = Vec::with_capacity(n_particles);
            for j in 0..n_particles {
                if j == j_ref {
                    new_counts.push(counts[j_ref].clone());
                    new_cum_flows.push(cum_flows[j_ref].clone());
                    new_acc.push(acc[j_ref].clone());
                } else {
                    new_counts.push(counts[indices[j]].clone());
                    new_cum_flows.push(cum_flows[indices[j]].clone());
                    new_acc.push(acc[indices[j]].clone());
                }
            }
            counts = new_counts;
            cum_flows = new_cum_flows;
            acc = new_acc;
            substep_ancestors = indices;
        }

        // Save pre-propagation states for ancestor sampling
        for j in 0..n_particles {
            prev_counts[j].copy_from_slice(&counts[j]);
        }

        // ── 2. Propagate free particles (parallel; gh#209) ──
        // Each particle writes only its own slot and draws from its own RNG
        // stream (`rngs[j]`), so concurrent execution is byte-identical to the
        // serial loop — the same Common-Random-Numbers property PF/IF2/PMMH
        // already rely on. The reference particle (`j_ref`) is clamped below,
        // not propagated. Pinned by the `RAYON_NUM_THREADS` 1-vs-N invariance
        // gate (`tests/gate_pgas_thread_invariance.rs`).
        let prop_results: Vec<Result<(), SimError>> = counts.par_iter_mut()
            .zip(substep_flows.par_iter_mut())
            .zip(particle_reals.par_iter_mut())
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .zip(substep_gammas.par_iter_mut())
            .enumerate()
            .map(|(j, (((((cnt, flows), real), rng), scratch), gammas))| {
                if j == j_ref { return Ok(()); }
                // Reset substep flows
                for f in flows.iter_mut() { *f = 0; }
                scratch.gamma_used.clear();

                // Populate the due batch step_one applies (gh#216): the same firing
                // plan the reference producer used at substep `s`, so free particles
                // and the (clamped) reference fire identically. `t + step_dt` is the
                // boundary; `dt` is the nominal firing-key grid.
                fill_producer_batch(
                    model, &fire_steps, t + step_dt, dt, s, firing,
                    &mut scratch.effect_batch,
                );
                step_one(
                    model, cnt, flows, real,
                    // `step_dt` is the realized substep (clipped under Exact).
                    params, t, step_dt, rng, scratch,
                )?;

                std::mem::swap(gammas, &mut scratch.gamma_used);
                Ok(())
            })
            .collect();
        for r in prop_results { r?; }

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
            // Ancestor weights, in parallel (gh#209). Each slot is an
            // independent transition-density eval over a read-only state; the
            // categorical draw below reads `ancestor_log_w` only after this
            // barrier, so concurrency is byte-identical to the serial loop.
            let ad_results: Vec<Result<(), SimError>> = ancestor_log_w
                .par_iter_mut()
                .enumerate()
                .map(|(j, slot)| {
                    // gh#audit-H8. Use the pre-resample state cache, not
                    // the post-resample prev_counts. CSMC ancestor
                    // sampling is supposed to categoricalise over the
                    // pre-step particle ensemble; the post-resample
                    // prev_counts permutes that ensemble silently.
                    // Reference slot j_ref keeps its corrected
                    // counts_before via prev_counts[j_ref] above.
                    let counts_before_substep = if j == j_ref {
                        &prev_counts[j_ref]  // already corrected to ref_rec.counts_before
                    } else {
                        &prev_counts_for_ancestor[j]
                    };
                    let td = log_transition_density_substep(
                        model,
                        counts_before_substep,
                        &ref_rec.flows,
                        &ref_rec.gammas,
                        params,
                        t,
                        step_dt,
                    )?;
                    // Post-resample slot j carries uniform weight (1/N);
                    // the categorical is driven by td alone.
                    *slot = td;
                    Ok(())
                })
                .collect();
            for r in ad_results { r?; }

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

        // ── 5. Compute weights — joint across all streams (parallel; gh#209) ──
        // Each particle's obs-likelihood is independent; we fold the per-particle
        // cum_flows reset into the same pass. `counts` is read-only here.
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            log_weights.par_iter_mut()
                .zip(cum_flows.par_iter_mut())
                .zip(acc.par_iter_mut())
                .zip(counts.par_iter())
                .for_each(|(((lw, cflows), a), cnt)| {
                    // FOLD (Phase 2a): close this interval's per-transition
                    // `cum_flows` into the per-stream `acc` BEFORE scoring; each
                    // slot is particle-local, so the parallel fold/score/reset is
                    // byte-identical to the serial loop (gh#209 CRN property).
                    obs_model.fold_into_acc(cflows, a);
                    *lw = obs_model.log_likelihood_from_flows_and_counts(
                        a, cnt, obs_idx, params);
                    // `cum_flows` blanket-zeroed; the per-stream `acc` bins
                    // per-stream — only Interval streams scheduled at THIS union
                    // index zero.
                    for f in cflows.iter_mut() { *f = 0; }
                    obs_model.reset_due_acc(obs_idx, a);
                });
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
            // The realized (t0, dt_substep) are grid properties shared by every
            // particle at substep s — read them from the reference, which carries
            // the grid the swarm tiled against. Under snap == (t_start+s·dt, dt).
            t0: reference.substeps[s].t0,
            dt_substep: reference.substeps[s].dt_substep,
        });
        particle = ancestors[s][particle];
    }
    trajectory_substeps.reverse();

    // Verify: each traceback record tiles contiguously (durations in (0, dt])
    // and its density is finite. The exact-tiling invariant — replaces the 2b
    // snap invariant (rec.t0 == t_start+s·dt) a shortened substep would violate.
    if cfg!(debug_assertions) {
        let mut prev_end = t_start;
        for (s, rec) in trajectory_substeps.iter().enumerate() {
            debug_assert!(rec.dt_substep > 0.0 && rec.dt_substep <= dt + 1e-9,
                "traceback substep {s}: dt_substep {} not in (0, dt={dt}]", rec.dt_substep);
            debug_assert!((rec.t0 - prev_end).abs() < 1e-9,
                "traceback substep {s}: t0 {} not contiguous with previous end {prev_end}", rec.t0);
            prev_end = rec.t0 + rec.dt_substep;
            let t = rec.t0;
            let verify_td = log_transition_density_substep(
                model, &rec.counts_before, &rec.flows, &rec.gammas, params, t, rec.dt_substep,
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
            // d/dz of the NATURAL-scale log-normal density log p(θ(z)).
            // log_density returns log N(z; μ, σ) − z (it pre-subtracts the
            // Log Jacobian z), so its z-derivative is −(z−μ)/σ² − 1. The
            // caller adds jacobian_grad = +1, recovering d/dz log N(z) =
            // −(z−μ)/σ². Omitting the −1 here left the NUTS gradient for
            // log_normal priors off by +1 (uncovered: the only FD gradient
            // test used Prior::Flat).
            -(z - mean) / (sd * sd) - 1.0
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
        Prior::LogUniform { lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            // d/dθ[−ln θ − const] = −1/θ; chain to z. With the Log transform
            // this is −1, which the caller's jacobian_grad (+1) cancels → the
            // z-scale density is flat, as it must be.
            let dlp_dtheta = -1.0 / theta;
            dlp_dtheta * param.transform_deriv(z)
        }
        Prior::TruncatedNormal { mean, sd, lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            // The normalizer Z is constant in θ, so only the Gaussian kernel
            // contributes: d/dθ[−0.5((θ−μ)/σ)²] = −(θ−μ)/σ².
            let dlp_dtheta = -(theta - mean) / (sd * sd);
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

    // gh#175: PGAS does not support hierarchical priors. The NUTS gradient
    // for a hierarchical leaf is stubbed to -inf (Gate 3b — see
    // `prior_log_density_and_grad_z`), and the MH fallback's non-env
    // `log_density` is likewise -inf. A hierarchical prior therefore makes
    // the log-posterior -inf everywhere, silently freezing the chain at its
    // starting point (100% divergent, 0% acceptance) rather than erroring —
    // a frozen, warm-started posterior that looks well-mixed. Refuse loudly
    // until Gate 3b lands; PMMH (`algorithm = pmmh`) supports hierarchical
    // priors today.
    if let Some(i) = priors.iter().position(|p| matches!(p, Prior::Hierarchical(_))) {
        let pname = if2_params.get(i).map(|p| p.name.as_str()).unwrap_or("<unknown>");
        return Err(SimError::Validation(format!(
            "PGAS does not support hierarchical priors (parameter '{pname}'): the \
             NUTS gradient for hierarchical leaves is not yet implemented (Gate 3b), \
             so the chain would freeze at its starting point instead of mixing. Use \
             `algorithm = pmmh` for hierarchical models, or give '{pname}' a \
             non-hierarchical prior."
        )));
    }

    let mut rng = StatefulRng::new(seed);
    let mut current_params = base_params.to_vec();
    let t_start = model.model.simulation.t_start;

    // exact-PGAS does not yet support always-active events: their firing keys on
    // round(t/dt) (the fire_steps lookup in effects::due_effects), which a
    // shortened exact substep shifts off the intended step. Refuse loudly rather
    // than silently misfire. (Scheduled non-active interventions ARE applied in
    // the PGAS producer path under both policies — step_one routes them through
    // due_effects -> apply_post_advance; pinned by the
    // gh187_pgas_scheduled_intervention regression test. gh#187's "skipped" claim
    // described pre-refactor code where inject_event_deltas handled only events.)
    if config.step_policy == StepPolicy::Exact
        && model.model.interventions.iter().any(|iv| iv.kind.is_event())
    {
        return Err(SimError::Validation(
            "exact obs-alignment is not yet supported for models with always-active \
             events (their firing keys on round(t/dt), which a shortened substep \
             shifts). Use obs_alignment = \"snap\", or place observations on the dt grid."
                .into(),
        ));
    }

    // gh#216: scheduled interventions fire CURSOR-keyed off the timeline's effect
    // boundaries (registered as `effect_times` in build_substep_grid below), so an
    // off-grid observation re-tiling the Exact grid no longer moves the firing
    // instant. The producer fires the boundary recorded at each substep; the
    // density (which scores records, never fires) is unaffected. Two Exact cases
    // stay unsupported and are refused loudly: a parametric `at [<param>]` schedule
    // (per-particle fire times) and a scheduled fire time off the dt grid (the
    // drift-free walk would need to re-anchor at a within-grid fractional point).
    // Snap is unaffected (no-op guards). Constant across sweeps because
    // AtTimesExpr+Exact is rejected.
    crate::intervention::guard_attimesexpr_exact(model, config.step_policy)?;
    crate::intervention::guard_exact_offgrid_effect_time(
        model, &current_params, t_start, config.dt, config.step_policy,
    )?;
    let scheduled = crate::intervention::timeline_effects(model, &current_params);

    // The realized substep grid (uniform under Snap; window-tiled with shortened
    // remainders under Exact) and its obs→substep + effect→substep maps — the
    // single grid every producer and density consumer tiles against. Under Snap
    // this is byte-identical to the legacy uniform grid + build_obs_at_substep
    // (effect_times only register under Exact, where they re-anchor the walk).
    let grid = build_substep_grid(t_start, config.dt, observations, &scheduled.times, config.step_policy)?;
    let obs_at_substep = grid.obs_at_substep;
    let effect_at_substep = grid.effect_at_substep;
    // The firing plan the producers use: Snap fires on the round(t/dt) key
    // (`None`); Exact fires the cursor-keyed scheduled interventions recorded per
    // substep. EVENTS under Exact are already rejected by the guard above.
    let firing: EffectFiring = match config.step_policy {
        StepPolicy::Snap => None,
        StepPolicy::Exact => Some((&effect_at_substep, scheduled.batches.as_slice())),
    };

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
        trajectory = simulate_reference_on_grid(
            model, &current_params, config.dt, &grid.steps, firing, &mut rng,
        )?;
        eprintln!("  reference: {} substeps, initial S={}",
            trajectory.substeps.len(),
            trajectory.initial_counts.first().copied().unwrap_or(0));
        current_transformed = if2_params.iter()
            .map(|p| p.to_transformed(current_params[p.index]))
            .collect();
        start_sweep = 0;

        // Sanity check: the trajectory must have finite density at its own params
        // (before IVP mapping, which adds initial state density).
        //
        // gh#80: distinguish the failure modes. -inf in the *transition* term
        // is a step_one/density-evaluator disagreement (a real bug). -inf in
        // the *observation* term is "this starting point is incompatible with
        // the data" — common when a chain initialises with a `tau` (or any
        // hard model parameter) outside a feasible region. The original
        // single-line "BUG: simulate_reference trajectory has -inf density at
        // own params" message lumped both together and accused step_one even
        // when the obs term was the cause.
        let sanity = complete_data_loglik(
            model, &trajectory, &current_params, observations,
            config.dt, obs_model, &[],  // empty IVP mappings
            &obs_at_substep,
        )?;
        if !sanity.total.is_finite() {
            let trans_inf = !sanity.transition.is_finite();
            let obs_inf   = !sanity.observation.is_finite();
            if trans_inf {
                eprintln!("  BUG: simulate_reference trajectory has non-finite \
                          *transition* log-density at own params (transition_ll = {}).",
                          sanity.transition);
                eprintln!("  This indicates a mismatch between step_one and \
                           log_transition_density_substep.");
                eprintln!("  Run with CAMDL_TRACE_STEPS=1 for detailed per-substep \
                           diagnostics.");
            }
            if obs_inf {
                eprintln!("  WARNING: simulate_reference predicts observed data with \
                          probability 0 (observation_ll = {}).", sanity.observation);
                eprintln!("  This is the data-vs-model side, NOT a step_one bug — the \
                           predicted trajectory at these starting parameters cannot \
                           explain the observed values.");
                eprintln!("  Common cause: a discrete-event parameter (e.g. `tau`) is \
                           outside the simulation window, so the seeding mechanism \
                           never fires and predicted incidence is 0 while real data has \
                           cases. Adjust starting bounds, or rely on NUTS / MH to \
                           propose into a feasible region.");
            }
            eprintln!("  params used:");
            for p in &model.model.parameters {
                if let Some(&idx) = model.param_index.get(p.name.as_str()) {
                    eprintln!("    {} = {}", p.name, current_params[idx]);
                }
            }
            eprintln!("  components: transition={:.1}, observation={:.1}, ivp={:.1}",
                sanity.transition, sanity.observation, sanity.ivp);
        } else {
            eprintln!("  simulate_reference LL sanity check: {:.1} (finite ✓)", sanity.total);
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
    //
    // gh#80: same split-by-component diagnostic as the sanity check above —
    // distinguish a step_one/density mismatch (transition term) from a
    // data-vs-model incompatibility (observation term).
    let current_components = complete_data_loglik(
        model, &trajectory, &current_params, observations,
        config.dt, obs_model, &ivp_mappings, &obs_at_substep,
    )?;
    let current_ll = current_components.total;
    eprintln!("  initial complete-data ll: {:.1}", current_ll);
    if !current_ll.is_finite() {
        let trans_inf = !current_components.transition.is_finite();
        let obs_inf   = !current_components.observation.is_finite();
        let ivp_inf   = !current_components.ivp.is_finite();
        if trans_inf {
            eprintln!("  WARNING: initial *transition* log-density is non-finite \
                       (transition_ll = {}).", current_components.transition);
            eprintln!("  This indicates a mismatch between step_one and \
                       log_transition_density_substep — run with \
                       CAMDL_TRACE_STEPS=1 for per-substep diagnostics.");
        }
        if obs_inf {
            eprintln!("  WARNING: initial *observation* log-density is non-finite \
                       (observation_ll = {}). The reference trajectory cannot \
                       explain the observed data at these starting parameters; \
                       NUTS / MH will propose into a feasible region if one exists.",
                       current_components.observation);
        }
        if ivp_inf {
            eprintln!("  WARNING: initial *IVP* log-density is non-finite \
                       (ivp_ll = {}) — initial-state Binom is incompatible \
                       with the IVP fraction parameter.",
                       current_components.ivp);
        }
        eprintln!("  components: transition={:.1}, observation={:.1}, ivp={:.1}",
            current_components.transition,
            current_components.observation,
            current_components.ivp);
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
    // gh#audit-C7. Post-burn-in counters (Stan-canonical surface;
    // burn-in counts are expected during step-size adaptation).
    let mut n_max_treedepth_post_burn: usize = 0;
    let mut n_divergent_post_burn: usize = 0;
    let mut swap_accepted: Vec<usize> = vec![0; n_rungs.saturating_sub(1)];

    if start_sweep >= config.n_sweeps {
        eprintln!("  warning: chain already completed {} sweeps (requested {}). \
                   Increase sweeps in fit.toml to continue.", start_sweep, config.n_sweeps);
    }

    // gh#audit-C1 preflight gate, narrowed by gh#20 + gh#76 + the gh#76
    // residual (BetaBinomial gradient). The observation-density gradient
    // now covers every likelihood arm:
    //
    //   • σ² (overdispersion) — wired via `gamma_density_value_and_grad_substep`.
    //   • NegBinomial, Normal, Poisson, Binomial, Bernoulli, BetaBinomial
    //     obs likelihoods — wired via `eval_likelihood_resolved_grad`.
    //
    // One arm remains uncovered (silent-zero gradient):
    //
    //   • `DerivedExpr` obs *projection* that depends on parameters — the
    //     chain-rule term ∂L/∂(projected) · ∂(projected)/∂θ is omitted
    //     (see derivation note 2026-05-25-pgas-obs-grad-derivation.md).
    //
    // That route lands the user in the silent-zero regime gh#76 was filed
    // against. Refuse `if2_params` whose reachability path touches a
    // parametric projection with a clear error.
    {
        use std::collections::HashSet;

        let mut parametric_derived_proj_refs: HashSet<String> = HashSet::new();

        for om in &model.model.observations {
            // DerivedExpr projections that depend on any parameter.
            if let ir::observation::Projection::DerivedExpr(e) = &om.projection {
                collect_param_refs(e, &mut parametric_derived_proj_refs);
            }
        }

        let mut blocked: Vec<String> = Vec::new();
        for spec in if2_params.iter() {
            if parametric_derived_proj_refs.contains(spec.name.as_str()) {
                blocked.push(format!("'{}' (in a parametric DerivedExpr projection)", spec.name));
            }
        }
        if !blocked.is_empty() {
            return Err(crate::error::SimError::Validation(format!(
                "PGAS+NUTS gradient does not cover parametric DerivedExpr obs \
                 projections (gh#76 follow-up). Estimating these parameters with \
                 NUTS would produce silently biased posteriors because the \
                 projection chain-rule term ∂L/∂(projected)·∂(projected)/∂θ is \
                 omitted, so the gradient is identically zero on the affected \
                 coordinate. Blocked parameters: {}. Either fix these parameters \
                 (move from `[estimate.X]` to `[fixed.X]` in fit.toml), switch to \
                 a non-gradient method (IF2, PMMH), or wait for the projection \
                 chain-rule term to land.",
                blocked.join(", ")
            )));
        }
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

    // Inverse map: estimated_to_model[est_idx] = model_param_idx. Used by
    // gh#20 (gamma-density gradient) and gh#76 (obs-density gradient) to
    // thread `eval_resolved_deriv` through σ² and likelihood-arg expressions.
    let estimated_to_model: Vec<usize> = if2_params.iter().map(|spec| spec.index).collect();

    // ── Trajectory warm-up: CSMC-only sweeps before parameter updates ──
    if config.trajectory_warmup > 0 && start_sweep == 0 {
        eprintln!("  trajectory warm-up: {} CSMC-only sweeps", config.trajectory_warmup);
        for warmup_sweep in 0..config.trajectory_warmup {
            for rung in 0..n_rungs {
                let csmc_seed = seed ^ ((warmup_sweep as u64).wrapping_mul(0x517cc1b727220a95))
                    ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142);
                let (new_traj, _diag) = csmc_as(
                    model, &rungs[rung].params, observations, &rungs[rung].trajectory,
                    config.n_particles, config.dt, obs_model,
                    &ivp_mappings, csmc_seed, &obs_at_substep, firing,
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
                        &estimated_to_model,
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
                        if sweep >= config.burn_in {
                            n_max_treedepth_post_burn += 1;
                        }
                    }
                    if result.divergent {
                        n_divergent += 1;
                        if sweep >= config.burn_in {
                            n_divergent_post_burn += 1;
                        }
                    }
                }

                // Two-phase adaptation (same schedule as single-rung, per-rung state)
                let mass_adapt_end = (adapt_end as f64 * 0.7) as usize;

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
                let csmc_seed = seed ^ ((sweep as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
                    ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142)
                    ^ (csmc_rep as u64).wrapping_mul(0xa2ce44bbfe0cf6d5);
                let (new_trajectory, diag) = csmc_as(
                    model, &rungs[rung].params, observations, &rungs[rung].trajectory,
                    config.n_particles, config.dt, obs_model,
                    &ivp_mappings, csmc_seed, &obs_at_substep, firing,
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

    // gh#audit-C7 / M18. Compute swap acceptance rates as a final
    // surface; n_rungs == 1 → empty vec, no diagnostic to fire.
    let swap_acceptance_rates: Vec<f64> = (0..n_rungs.saturating_sub(1))
        .map(|i| if swap_proposed[i] > 0 {
            swap_accepted[i] as f64 / swap_proposed[i] as f64
        } else { 0.0 })
        .collect();

    Ok(PGASResult {
        sweeps,
        final_trajectory: rungs[0].trajectory.clone(),
        acceptance_rates,
        resume_state,
        n_divergent_total: n_divergent,
        n_divergent_post_burn,
        n_max_treedepth_total: n_max_treedepth,
        n_max_treedepth_post_burn,
        swap_acceptance_rates,
    })
}

#[cfg(test)]
mod grid_tests {
    //! Keystone unit tests for [`build_substep_grid`] — the realized-grid + obs-map
    //! contract every exact-PGAS producer tiles against (Stage 3, 2c).
    use super::*;

    fn obs(times: &[f64]) -> Vec<Observation> {
        times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect()
    }

    fn sorted_map(g: &SubstepGrid) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = g.obs_at_substep.iter().map(|(&k, &val)| (k, val)).collect();
        v.sort();
        v
    }

    #[test]
    fn snap_grid_is_the_legacy_uniform_grid() {
        let observations = obs(&[3.0, 7.0, 10.0]);
        let g = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Snap).unwrap();
        let expect: Vec<(f64, f64)> = (0..10).map(|s| (s as f64, 1.0)).collect();
        assert_eq!(g.steps, expect);
        assert_eq!(g.obs_at_substep, build_obs_at_substep(&observations, 0.0, 1.0).unwrap());
        assert_eq!(sorted_map(&g), vec![(2, 0), (6, 1), (9, 2)]);
    }

    #[test]
    fn exact_tiles_off_grid_obs_with_remainder() {
        let observations = obs(&[3.5, 7.0, 10.5]);
        let g = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Exact).unwrap();
        assert_eq!(g.steps.len(), 12);
        let dts: Vec<f64> = g.steps.iter().map(|&(_, d)| d).collect();
        assert_eq!(dts, vec![1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5]);
        // Each window's recorded substep ends exactly on its obs time.
        for (s, obs_t) in [(3usize, 3.5_f64), (7, 7.0), (11, 10.5)] {
            let (t0, d) = g.steps[s];
            assert!((t0 + d - obs_t).abs() < 1e-9, "substep {s} must land on obs {obs_t}");
        }
        assert_eq!(sorted_map(&g), vec![(3, 0), (7, 1), (11, 2)]);
        // 7.0 is on the GLOBAL grid but off the SHIFTED (anchored at 3.5) grid —
        // a window is tiled relative to its own start, so it lands via a remainder.
        assert!(g.steps.iter().any(|&(_, d)| d != 1.0), "off-grid windows must produce shortened substeps");
    }

    #[test]
    fn exact_grid_with_on_grid_effect_and_off_grid_obs() {
        // The effect-re-anchor path the off-grid-obs-only tests don't reach
        // (gh#233: the shared Substeps walk re-anchors at the EXACT effect time).
        // dt=1, obs at [3.0, 7.5], one on-grid effect at 2.0. The full expected
        // grid is hand-computed and was verified bit-identical to the deleted
        // hand-rolled whole-run walk: substeps 0,1,2 reach obs(0)=3.0 (the effect
        // fires on the substep landing on t=2, idx 1); substeps 3..7 reach
        // obs(1)=7.5 with a 0.5 remainder. gate_pgas_density_baseline +
        // gh187_pgas_scheduled_intervention pin the same path end-to-end.
        let observations = obs(&[3.0, 7.5]);
        let g = build_substep_grid(0.0, 1.0, &observations, &[2.0], StepPolicy::Exact).unwrap();
        let dts: Vec<f64> = g.steps.iter().map(|&(_, d)| d).collect();
        assert_eq!(dts, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]);
        let t0s: Vec<f64> = g.steps.iter().map(|&(t0, _)| t0).collect();
        assert_eq!(t0s, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(sorted_map(&g), vec![(2, 0), (7, 1)], "obs land on each window's last substep");
        let mut eff: Vec<(usize, usize)> =
            g.effect_at_substep.iter().map(|(&k, &v)| (k, v)).collect();
        eff.sort();
        assert_eq!(eff, vec![(1, 0)], "effect 0 fires on the substep landing on t=2 (idx 1)");
    }

    #[test]
    fn exact_on_grid_equals_snap_dt_one() {
        // On-grid obs at dt=1.0: Exact and Snap grids are identical.
        let observations = obs(&[3.0, 7.0, 10.0]);
        let snap = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Snap).unwrap();
        let exact = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Exact).unwrap();
        assert_eq!(exact, snap);
    }

    #[test]
    fn exact_on_grid_matches_snap_to_ulp_and_lands_exactly_fractional_dt() {
        // At dt=0.1 the grid SPACING differs from dt in FP, so EXACT (which clips
        // the window's final step via Schedule::substep) and SNAP (literal dt)
        // diverge by ≤1 ULP at each window's last step — the sanctioned
        // EXACT-stepper drift (substep-time proposal). The obs MAP is identical,
        // and EXACT lands exactly on each obs (the property SNAP lacks).
        let observations = obs(&[3.0, 5.0]);
        let snap = build_substep_grid(0.0, 0.1, &observations, &[], StepPolicy::Snap).unwrap();
        let exact = build_substep_grid(0.0, 0.1, &observations, &[], StepPolicy::Exact).unwrap();
        assert_eq!(exact.obs_at_substep, snap.obs_at_substep, "obs map must be identical");
        assert_eq!(exact.steps.len(), snap.steps.len());
        for (i, (&(et, ed), &(st, sd))) in exact.steps.iter().zip(&snap.steps).enumerate() {
            assert!((et - st).abs() <= 1e-12, "t0 differs by > 1 ULP at substep {i}: {et} vs {st}");
            assert!((ed - sd).abs() <= 1e-12, "dt_substep differs by > 1 ULP at substep {i}: {ed} vs {sd}");
        }
        // EXACT lands exactly on each obs (within FP), where SNAP rounds.
        for (&idx, _) in &exact.obs_at_substep {
            let (t0, d) = exact.steps[idx];
            let obs_t = if t0 < 4.0 { 3.0 } else { 5.0 };
            assert!((t0 + d - obs_t).abs() < 1e-9, "exact substep {idx} must land on its obs");
        }
    }

    #[test]
    fn exact_t0_is_drift_free_within_window() {
        // Within a single obs window the t0 is the drift-free Schedule::substep_time
        // value (window_start + s·dt), never an accumulation. The window's final
        // step is the clipped remainder that lands on the obs.
        let observations = obs(&[5.0]);
        let g = build_substep_grid(0.0, 0.1, &observations, &[], StepPolicy::Exact).unwrap();
        let n = g.steps.len();
        for (s, &(t0, d)) in g.steps.iter().enumerate() {
            assert_eq!(t0.to_bits(), (s as f64 * 0.1).to_bits(), "t0 not drift-free at {s}");
            if s + 1 < n {
                assert_eq!(d, 0.1, "interior substep must be a full dt");
            } else {
                // final step: clipped remainder landing on the obs
                assert!(d > 0.0 && d <= 0.1 + 1e-12);
                assert!((t0 + d - 5.0).abs() < 1e-9, "final step must land on the obs");
            }
        }
    }

    #[test]
    fn exact_window_substeps_sum_to_window_length() {
        // Σ dt_substep within each obs window equals the window length, and each
        // t0 is monotone — the relaxed invariant the consumers assert under exact.
        let observations = obs(&[2.5, 6.0, 9.3]);
        let g = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Exact).unwrap();
        let mut prev_end = 0.0;
        for &(t0, d) in &g.steps {
            assert!(t0 >= prev_end - 1e-12, "t0 must be monotone (got {t0} after {prev_end})");
            assert!(d > 0.0 && d <= 1.0 + 1e-12, "0 < dt_substep ≤ dt, got {d}");
            prev_end = t0 + d;
        }
        // The last substep of the run lands on the last obs.
        let (lt, ld) = *g.steps.last().unwrap();
        assert!((lt + ld - 9.3).abs() < 1e-9);
    }

    #[test]
    fn empty_obs_yields_empty_grid() {
        let g = build_substep_grid(0.0, 1.0, &[], &[], StepPolicy::Exact).unwrap();
        assert!(g.steps.is_empty() && g.obs_at_substep.is_empty());
    }

    // ── M2: sub-dt observation collision under Snap ──────────────────────
    //
    // Two DISTINCT, strictly-increasing obs closer than dt round to the same
    // substep index (interval_steps is round-to-nearest), collide on the same
    // `ObsAtSubstep` key, and the last-wins `map.insert` silently drops one
    // from the PGAS likelihood → biased posterior. The increasing-times guard
    // (`validate_obs_times_increasing`) is dt-independent and does NOT catch
    // this. The fix makes grid construction collision-detecting.

    #[test]
    fn snap_sub_dt_colliding_obs_is_rejected_by_build_obs_at_substep() {
        // t=3.0 and t=3.4 at dt=1, t_start=0 both round to substep index 2.
        let observations = obs(&[3.0, 3.4]);
        let result = build_obs_at_substep(&observations, 0.0, 1.0);
        assert!(
            result.is_err(),
            "two distinct obs within dt must be rejected, not silently collapsed"
        );
    }

    #[test]
    fn snap_sub_dt_colliding_obs_is_rejected_by_build_substep_grid() {
        let observations = obs(&[3.0, 3.4]);
        let result = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Snap);
        assert!(
            result.is_err(),
            "Snap grid must reject sub-dt-colliding observation times"
        );
    }

    #[test]
    fn snap_non_colliding_obs_builds_grid_with_both_present() {
        // t=3.0 and t=6.0 at dt=1 land on distinct substeps (2 and 5).
        let observations = obs(&[3.0, 6.0]);
        let g = build_substep_grid(0.0, 1.0, &observations, &[], StepPolicy::Snap)
            .expect("non-colliding obs must build fine");
        assert_eq!(sorted_map(&g), vec![(2, 0), (5, 1)]);
        let map = build_obs_at_substep(&observations, 0.0, 1.0)
            .expect("non-colliding obs must build fine");
        assert_eq!(map.len(), 2, "both observations must be present");
    }
}

#[cfg(test)]
mod prior_grad_tests {
    //! Finite-difference check of the per-parameter NUTS *target* gradient
    //! assembled exactly as `run_pgas`'s `log_prob_and_grad` closure does it:
    //!   value(z)    = prior.log_density(θ, z) + param.log_jacobian(z)
    //!   gradient(z) = prior_grad_z + param.jacobian_grad(z)
    //! where `(_, prior_grad_z) = prior_log_density_and_grad_z(...)`.
    //!
    //! The only existing FD gradient test (`tests/gradient_check.rs`) uses
    //! `Prior::Flat`, so the prior-gradient arms here had no coverage — this
    //! is the gate for them.
    use super::*;
    use crate::inference::types::Transform;

    fn log_param(lo: f64, hi: f64) -> EstimatedParam {
        EstimatedParam {
            name: "p".into(), index: 0, initial: 1.0, rw_sd: 0.1,
            transform: Transform::Log { lo, hi },
            lower: lo, upper: hi, rw_sd_auto: false, ivp: false,
        }
    }

    fn identity_param(lo: f64, hi: f64) -> EstimatedParam {
        EstimatedParam {
            name: "p".into(), index: 0, initial: 0.0, rw_sd: 0.1,
            transform: Transform::None,
            lower: lo, upper: hi, rw_sd_auto: false, ivp: false,
        }
    }

    /// Assemble the per-parameter z-scale target value the NUTS closure sees.
    fn target_value(prior: &Prior, param: &EstimatedParam, z: f64) -> f64 {
        let theta = param.from_transformed(z);
        prior.log_density(theta, z) + param.log_jacobian(z)
    }

    /// Assemble the analytic z-scale gradient the NUTS closure uses.
    fn target_grad(prior: &Prior, param: &EstimatedParam, z: f64) -> f64 {
        let theta = param.from_transformed(z);
        let (_, prior_grad_z) = prior_log_density_and_grad_z(prior, param, theta, z);
        prior_grad_z + param.jacobian_grad(z)
    }

    fn assert_grad_matches_fd(prior: &Prior, param: &EstimatedParam, zs: &[f64]) {
        let eps = 1e-6;
        for &z in zs {
            let fd = (target_value(prior, param, z + eps)
                - target_value(prior, param, z - eps)) / (2.0 * eps);
            let an = target_grad(prior, param, z);
            let rel = if fd.abs() > 1e-6 { (an - fd).abs() / fd.abs() } else { (an - fd).abs() };
            assert!(rel < 1e-4,
                "{:?} @ z={}: analytic grad {} != fd {} (rel {:.2e})",
                prior, z, an, fd, rel);
        }
    }

    #[test]
    fn log_normal_grad_matches_fd() {
        // Regression: the TransformedNormal arm returned -(z-μ)/σ² but the
        // caller adds jacobian_grad = +1 unconditionally and log_density
        // pre-subtracts the -z Jacobian — leaving the gradient off by +1.
        let p = log_param(1e-4, 1e2);
        assert_grad_matches_fd(&Prior::TransformedNormal { mean: 1.0, sd: 0.5 },
            &p, &[-1.0, 0.0, 0.7, 1.5]);
    }

    #[test]
    fn natural_scale_priors_grad_matches_fd() {
        // These arms already follow the natural-density convention; lock them.
        let lp = log_param(1e-4, 1e2);
        assert_grad_matches_fd(&Prior::HalfNormal { sigma: 1.0 }, &lp, &[-1.0, 0.0, 1.0]);
        assert_grad_matches_fd(&Prior::Gamma { shape: 2.0, rate: 1.5 }, &lp, &[-1.0, 0.0, 1.0]);
        assert_grad_matches_fd(&Prior::Exponential { rate: 0.7 }, &lp, &[-1.0, 0.0, 1.0]);
        let ip = identity_param(-5.0, 5.0);
        assert_grad_matches_fd(&Prior::Normal { mean: 0.3, sd: 0.8 }, &ip, &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn log_uniform_grad_matches_fd() {
        // On the Log transform the z-scale density is flat → gradient 0.
        let p = log_param(1e-5, 1e-2);
        let zs = [(1e-4_f64).ln(), (1e-3_f64).ln(), (5e-3_f64).ln()];
        assert_grad_matches_fd(&Prior::LogUniform { lower: 1e-5, upper: 1e-2 }, &p, &zs);
        // And it really is flat (gradient ≈ 0 everywhere interior).
        for &z in &zs {
            assert!(target_grad(&Prior::LogUniform { lower: 1e-5, upper: 1e-2 }, &p, z).abs() < 1e-9);
        }
    }

    #[test]
    fn truncated_normal_grad_matches_fd() {
        // Identity transform, bounds = truncation support.
        let ip = identity_param(0.3, 1.0);
        assert_grad_matches_fd(
            &Prior::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 },
            &ip, &[0.4, 0.7, 0.95]);
        // Logit transform onto [0.3, 1.0] — bounds equal truncation support.
        let lp = EstimatedParam {
            name: "p".into(), index: 0, initial: 0.7, rw_sd: 0.1,
            transform: Transform::Logit { lo: 0.3, hi: 1.0 },
            lower: 0.3, upper: 1.0, rw_sd_auto: false, ivp: false,
        };
        assert_grad_matches_fd(
            &Prior::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 },
            &lp, &[-1.0, 0.0, 1.0]);
    }
}
