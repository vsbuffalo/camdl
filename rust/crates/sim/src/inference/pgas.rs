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
use crate::inference::numerics::BINOM_PROB_EPS;
use crate::inference::particle_filter::Observation;
use crate::inference::resampling::conditional_multinomial_resample;
use crate::inference::pmmh::Prior;
use crate::inference::prior::Density;
use crate::inference::types::{EstimatedParam, RESAMPLE_RNG_STREAM, init_particle_rngs, restore_z_values};

/// Process-noise variance floor below which an overdispersed transition is
/// treated as carrying no gamma multiplier. MUST match between the PGAS
/// log-density (`density`) and its gradient (`pgas_grad`): a divergence
/// desyncs the value path's gamma index from the gradient's, corrupting the
/// trajectory's energy.
pub(crate) const OVERDISP_SIGMA_SQ_FLOOR: f64 = 1e-30;
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::eval_resolved;
use crate::schedule::{Cursor, Schedule, StepPolicy};
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

impl PGASTrajectory {
    /// A directionally-coherent per-substep state path: the compartment counts
    /// at the END of each substep, reconstructed so the path is a single
    /// continuous lineage (gh#264).
    ///
    /// **What this defends against.** A `SubstepRecord` sequence in which
    /// `counts_after[s] != counts_before[s+1]` — the raw `counts_after` sequence
    /// jumps (apparent backflow — e.g. `S` *increasing* in an SEIRD), and the
    /// jump compounds as a fixed offset over the suffix. The CSMC-AS traceback
    /// used to produce exactly that at an ancestor-sampling join, because the
    /// reference slot recorded its OWN pre-state rather than the pre-state of
    /// the ancestor it had been assigned. That is fixed at source (gh#607): the
    /// reference slot is re-anchored on its sampled ancestor, so `csmc_as`
    /// returns a continuous path and this method is an identity on it (pinned by
    /// `tests/csmc_splice_continuity.rs`). It stays as the guard on a corrupt
    /// record from any other producer, and for its negative-count check.
    ///
    /// **Why the net delta is the fix.** The per-substep net delta
    /// `counts_after[s] - counts_before[s]` is exactly the realized state change
    /// at substep `s` (it already includes events/balance/clamping, whatever
    /// `step_one` did). At a join the reference suffix is offset by a constant
    /// `Δ` in both `counts_before` and `counts_after`, so `Δ` **cancels** in the
    /// difference. Chaining these deltas from the first substep's pre-state
    /// therefore yields a path that is continuous, conserves whatever the
    /// per-substep transitions conserve, and is directionally valid by
    /// construction — without needing the model stoichiometry here.
    ///
    /// On a coherent (join-free) trajectory this reproduces `counts_after`
    /// exactly. Returns one state vector per substep. `Err` only on a corrupt
    /// record (mismatched compartment vector lengths).
    pub fn coherent_counts_after(&self) -> Result<Vec<Vec<i64>>, String> {
        let mut out = Vec::with_capacity(self.substeps.len());
        let Some(first) = self.substeps.first() else { return Ok(out) };
        let n_comp = first.counts_before.len();
        let mut state = first.counts_before.clone();
        for (s, rec) in self.substeps.iter().enumerate() {
            if rec.counts_before.len() != n_comp || rec.counts_after.len() != n_comp {
                return Err(format!(
                    "PGASTrajectory::coherent_counts_after: substep {s} has \
                     {}/{} compartment counts, expected {n_comp}",
                    rec.counts_before.len(), rec.counts_after.len()));
            }
            for i in 0..n_comp {
                state[i] += rec.counts_after[i] - rec.counts_before[i];
                // Directionality guard (gh#264): a coherent compartment count can
                // never go negative. If chaining the realized deltas produces one,
                // the record is corrupt — surface it (the writer logs + skips the
                // file) rather than emitting a physically-impossible trajectory.
                if state[i] < 0 {
                    return Err(format!(
                        "PGASTrajectory::coherent_counts_after: compartment {i} went \
                         negative ({}) at substep {s} — incoherent trajectory record",
                        state[i]));
                }
            }
            out.push(state.clone());
        }
        Ok(out)
    }

    /// Project this substep-resolution reference path into the shared
    /// [`crate::state::Trajectory`] output type — the same type `simulate`
    /// produces — plus a per-substep, per-incidence-stream `inc_<stream>`
    /// matrix (`incidence[s][k]`), so one posterior-trajectory writer serves
    /// `simulate` and PGAS alike.
    ///
    /// **Counts.** Each snapshot's integer state is the directionally-coherent
    /// path from [`coherent_counts_after`](Self::coherent_counts_after) (the
    /// net-delta-chained counts, not the raw per-substep `counts_after` — see
    /// that method for why the CSMC-AS join can leave the raw sequence
    /// incoherent, gh#264). The density internals (`counts_before`, `gammas`,
    /// `dt_substep` as a stored field) are dropped: an output trajectory carries
    /// state + flows, not the likelihood machinery.
    ///
    /// **Time.** The first snapshot is the initial-condition row at `t_start`
    /// (`substeps[0].t0`, the path's anchor state `substeps[0].counts_before`,
    /// zeroed flows); each subsequent snapshot is stamped at the substep's
    /// realized END time `t0 + dt_substep` — read from the record, never
    /// recomputed as `t_start + s·dt`, so an off-grid / exact tiling can't
    /// misstamp. Emitting the `t_start` row is what makes the aggregate
    /// `Σ flow_infection == S₀ − S_final` hold in a seeded stratum (gh#270): the
    /// first substep's flow gets its S decrement recorded against the true `S₀`,
    /// not a post-first-substep value.
    ///
    /// **Flows.** The per-substep integer flows ride through as [`Flows::Int`];
    /// the prepended `t_start` row carries zeroed flows (no interval precedes it).
    ///
    /// **Incidence (`inc_<stream>`).** For each incidence stream (the
    /// `(name, flow_indices)` pairs from
    /// [`MultiStreamObsModel::incidence_streams`]), the column value at substep
    /// `s` is `Σ_{i ∈ flow_indices} flows[s][i]` — the model's declared
    /// `FlowSum` projection applied to that substep's flows. This is the gh#48
    /// safe path: it never finite-differences compartment counts (`−ΔS`,
    /// `diff(flow)`), which is unsafe under event/balance interactions (#264).
    /// `incidence` is empty (and the writer emits no `inc_*` columns) when the
    /// model has no incidence streams.
    pub fn to_trajectory(
        &self,
        incidence_streams: &[(String, Vec<usize>)],
    ) -> Result<(crate::state::Trajectory, Vec<Vec<f64>>), String> {
        use crate::state::{Flows, IntState, RealState, Snapshot, Trajectory};

        let coherent = self.coherent_counts_after()?;
        let mut traj = Trajectory::new();
        let mut incidence: Vec<Vec<f64>> = Vec::with_capacity(self.substeps.len() + 1);

        // gh#270: emit the initial-condition row at `t_start` FIRST, so the saved
        // path carries its own starting point. Without it the first written row is
        // the END of substep 0, so that substep's infection flow has no S decrement
        // recorded before it — and the aggregate `Σ flow_infection == S₀ − S_final`
        // then fails by *exactly* `flow_infection[0]` in any stratum seeded with
        // `I₀ > 0` (the seed-stratum residual). The anchor is
        // `substeps[0].counts_before` — the same pre-state `coherent_counts_after`
        // chains its deltas from, so the path stays continuous — stamped at
        // `substeps[0].t0` (the realized `t_start`, read from the record, never
        // recomputed). Flows are zero: no interval precedes `t_start`. This mirrors
        // the forward chain-binomial writer, which emits a `t_start` snapshot with
        // zeroed flows before its substep loop.
        if let Some(first) = self.substeps.first() {
            traj.push(Snapshot {
                t: first.t0,
                int_state: IntState::from_vec(first.counts_before.clone()),
                real_state: RealState::from_vec(Vec::new()),
                flows: Flows::Int(vec![0; first.flows.len()]),
            });
            if !incidence_streams.is_empty() {
                incidence.push(vec![0.0; incidence_streams.len()]);
            }
        }

        for (s, rec) in self.substeps.iter().enumerate() {
            let t = rec.t0 + rec.dt_substep;
            traj.push(Snapshot {
                t,
                int_state: IntState::from_vec(coherent[s].clone()),
                // PGAS runs the chain-binomial backend (integer compartments
                // only); no real compartments to record.
                real_state: RealState::from_vec(Vec::new()),
                flows: Flows::Int(rec.flows.clone()),
            });
            // Per-substep incidence = the FlowSum projection over this substep's
            // flows. Additive across substeps; a downstream consumer cumulates
            // within an observation interval if it wants interval incidence.
            if !incidence_streams.is_empty() {
                incidence.push(
                    incidence_streams.iter()
                        .map(|(_, idxs)| idxs.iter()
                            .map(|&i| rec.flows[i] as f64)
                            .sum::<f64>())
                        .collect(),
                );
            }
        }
        Ok((traj, incidence))
    }
}

/// Number of equal-width time bins `CSMCDiagnostics::renewal_by_bin` resolves
/// renewal into.
///
/// Ten, so that the one number worth quoting on its own — renewal over the
/// early window, where path degeneracy bites first — IS bin 0. It needs no
/// separate accumulator and therefore cannot disagree with the profile.
///
/// Fixed, not proportional to the series length, so the profile is comparable
/// across models and across particle counts: bin `b` is always the fraction of
/// the series from `b/10` to `(b+1)/10`, whatever `n_substeps` is.
pub const RENEWAL_BINS: usize = 10;

/// Accumulator for renewal resolved in time: renewed / total substeps per bin.
///
/// Fixed-size arrays, so recording a substep is a bounds-checked increment with
/// no allocation — it rides along in the traceback loop, which already walks
/// every substep.
#[derive(Clone, Debug)]
pub struct RenewalBins {
    n_substeps: usize,
    renewed: [usize; RENEWAL_BINS],
    total: [usize; RENEWAL_BINS],
}

impl RenewalBins {
    pub fn new(n_substeps: usize) -> Self {
        RenewalBins { n_substeps, renewed: [0; RENEWAL_BINS], total: [0; RENEWAL_BINS] }
    }

    /// Record the traceback's decision at substep `s`: `renewed` iff that
    /// substep was taken from a non-reference particle. Order-independent —
    /// the traceback walks backwards.
    #[inline]
    pub fn record(&mut self, s: usize, renewed: bool) {
        debug_assert!(s < self.n_substeps, "substep {s} outside the series of {}", self.n_substeps);
        let b = (s * RENEWAL_BINS / self.n_substeps).min(RENEWAL_BINS - 1);
        self.total[b] += 1;
        if renewed {
            self.renewed[b] += 1;
        }
    }

    /// Per-bin renewal fraction. A bin holding no substep reads `NaN`, not
    /// `0.0` — the convention [`CSMCDiagnostics::as_accept_rate`] already uses,
    /// and for the same reason: "no substep fell here" and "no substep here was
    /// renewed" are different diagnoses, and collapsing them invents a
    /// degeneracy that was never observed.
    pub fn finish(&self) -> [f64; RENEWAL_BINS] {
        let mut out = [f64::NAN; RENEWAL_BINS];
        for ((slot, &renewed), &total) in out.iter_mut().zip(&self.renewed).zip(&self.total) {
            if total > 0 {
                *slot = renewed as f64 / total as f64;
            }
        }
        out
    }
}

/// Diagnostics from one CSMC-AS sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CSMCDiagnostics {
    /// Fraction of traceback substeps from non-reference particles.
    /// Near 0% = path degeneracy (reference never replaced, CSMC broken).
    /// Near 50%+ = healthy trajectory renewal.
    pub trajectory_renewal: f64,
    /// The same renewal, resolved in time: bin `b` is the fraction of the
    /// substeps in the `b`-th tenth of the series that the traceback took from
    /// a non-reference particle. `NaN` for a bin holding no substep.
    ///
    /// This is the diagnostic the particle-Gibbs literature recommends in place
    /// of a rule for choosing the particle count `N`, of which there is none:
    /// Chopin & Singh (2015, *Bernoulli* 21:1855-1883) prove uniform ergodicity
    /// for particle Gibbs existentially, with no rate in `N`, and Lindsten,
    /// Jordan & Schön (2014, *JMLR* 15:2145-2184) call informative rates in `N`
    /// open. Both recommend instead plotting the update rate of the state xₜ
    /// against t — LJS Figure 1 (PG vs PGAS, N ∈ {5, 20, 100, 1000}, T = 400).
    ///
    /// Averaged over sweeps, this vector IS that plot. The shape is what the
    /// aggregate throws away: under path degeneracy the update rate is ≈0 early
    /// and rises to 1 near the end of the series, because every lineage has
    /// coalesced onto the reference by the time the traceback reaches the early
    /// states — and `trajectory_renewal` scores such a sweep identically to one
    /// renewed uniformly in t. Early-time degeneracy is the failure ancestor
    /// sampling exists to fix, and the early states are where the parameters
    /// governing initial conditions and early dynamics get their information.
    pub renewal_by_bin: [f64; RENEWAL_BINS],
    /// Number of substeps where all ancestor weights were -inf.
    pub n_degenerate: usize,
    /// Substeps where a resampling actually drew a new ancestry — i.e. where
    /// the incoming weights were not all equal. Between observations they are,
    /// so this counts the observation boundaries the sweep consumed, which is
    /// NOT the number of observations: the weights an observation produces are
    /// consumed by the FOLLOWING substep, so a terminal observation sets
    /// weights that only ever feed the final trajectory draw (gh#718).
    pub n_resampled: usize,
    /// Substeps where ancestor sampling was skipped because no resampling drew
    /// an ancestry there. The mixing cost of the gh#718 defect-2 fix: this plus
    /// `n_resampled` is the total substep count, and only the latter offers the
    /// reference a chance to renew.
    pub n_as_skipped_no_resample: usize,
    /// Total substeps.
    pub n_substeps: usize,
    /// Substeps where the Eq.-(17) proposal named an ancestor OTHER than the
    /// reference's current one, so the LJS Eq.-(21) Metropolis step actually
    /// ran. The denominator of the ancestor-sampling acceptance rate.
    ///
    /// Distinguishes "ancestor sampling is proposing moves that get rejected"
    /// from "ancestor sampling never proposes a move at all" — the latter is
    /// the `SpliceGuard` masking every alternative, which looks identical in
    /// `trajectory_renewal` but has a completely different cause.
    pub n_as_proposed: usize,
    /// Of `n_as_proposed`, how many the Metropolis step accepted. Numerator of
    /// the ancestor-sampling acceptance rate.
    pub n_as_accepted: usize,
    /// Of `n_as_proposed`, how many were rejected because the EXACT suffix
    /// ratio was zero-density — a candidate the cheap screened weight admitted
    /// but the full ratio refused.
    ///
    /// Separated from the coin-flip rejections because the two say different
    /// things: an inadmissible rejection is the target's support asserting
    /// itself (correct, and the whole point of gh#607), whereas a large
    /// coin-flip rejection rate at finite ratios means the proposal is simply
    /// badly matched to the target and mixing is paying for it.
    pub n_as_refused_inadmissible: usize,
}

impl CSMCDiagnostics {
    /// Ancestor-sampling Metropolis acceptance rate, or `NaN` when the step
    /// never ran this sweep.
    ///
    /// `NaN` and `0.0` are different diagnoses and must not be collapsed:
    /// `NaN` means no alternative ancestor was ever proposed (the screened
    /// weights left only the reference's own lineage admissible), while `0.0`
    /// means alternatives were proposed every time and the exact ratio refused
    /// all of them. Both flatten `trajectory_renewal`; only the second implicates
    /// the acceptance ratio.
    pub fn as_accept_rate(&self) -> f64 {
        if self.n_as_proposed == 0 {
            f64::NAN
        } else {
            self.n_as_accepted as f64 / self.n_as_proposed as f64
        }
    }
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
    /// `log p(x₀ | θ)` — the density of the trajectory's own initial state
    /// under the laws the model DECLARES (`init { I ~ poisson(rate = I0) }`).
    /// Zero for a model whose `init {}` is entirely deterministic, and zero
    /// there because there is no law, not because the term was dropped.
    ///
    /// Reported on its own in the sweep trace (`initial_state_ll`), not left to
    /// be recovered by subtracting `transition_ll` and `obs_ll` from the total:
    /// a constant component of the target that is only visible by subtraction
    /// is what made gh#719 need trace forensics to find.
    pub initial_state: f64,
}

/// Result of one Gibbs sweep.
#[derive(Clone, Serialize, Deserialize)]
pub struct PGASSweep {
    /// The 0-based sweep index this draw came from. Carried so the persisted
    /// posterior (`draws.tsv`) can key each draw `(chain, sweep)` and join to
    /// the smoothed `trajectories.tsv` (which keys on the same sweep number) —
    /// the `(θ, X)` pairing. Recorded, not re-derived from burn-in/thin, so the
    /// key can't drift from the retention rule.
    /// (gh#322: foundation for the keyed-joint (θ, X) output; wired by the join.)
    pub sweep: usize,
    pub params: Vec<f64>,
    pub log_complete_data_ll: f64,
    pub accepted: Vec<bool>,
    pub csmc_diag: CSMCDiagnostics,
    pub proposal_sds: Vec<f64>,
    /// Transition component of the complete-data log-likelihood.
    pub transition_ll: f64,
    /// Observation component of the complete-data log-likelihood.
    pub obs_ll: f64,
    /// Initial-state component `log p(x₀ | θ)` of the complete-data
    /// log-likelihood — the density of the sweep's own `x₀` under the laws
    /// declared in `init { }`. Zero for a deterministic `init { }`.
    ///
    /// Recorded as its own field for the same reason it gets its own trace
    /// column: a term recoverable only by subtracting the other two from the
    /// total is one nobody looks at until a fit is already wrong (gh#719).
    pub initial_state_ll: f64,
    /// Per-sweep NUTS diagnostics for the cold chain's `θ|X` update (gh#294).
    /// Zero on the non-gradient (random-walk MH) proposal path, which takes no
    /// NUTS step.
    pub nuts: NutsSweepDiag,
}

/// Cold-chain NUTS telemetry recorded once per Gibbs sweep — the standard HMC
/// diagnostic set, surfaced so PGAS geometry/leapfrog cost is observable
/// (gh#294). All from the cold rung's single `nuts_step` per sweep.
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct NutsSweepDiag {
    /// Doublings of the NUTS tree this sweep.
    pub tree_depth: usize,
    /// Leapfrog steps taken this sweep.
    pub n_leapfrog: usize,
    /// Step size used for this sweep's integration (pre-adaptation update).
    pub step_size: f64,
    /// Mean Metropolis acceptance probability across the tree.
    pub accept_stat: f64,
    /// Divergent transitions this sweep (0 or 1 — one NUTS step per sweep).
    pub n_divergent: usize,
    /// Initial Hamiltonian energy `H0` (for E-BFMI).
    pub energy: f64,
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
    let (p_total, _q) = super::numerics::prob_q_from_rate_dt_clamped(total_rate, dt, BINOM_PROB_EPS);
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
            let p_split = (eff_rate / rate_remaining).clamp(BINOM_PROB_EPS, 1.0 - BINOM_PROB_EPS);
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
    // gh#272 LICM: per-eval prologue staged at the PGAS sweep boundary (θ fixed
    // across the conditional filter), threaded in. `None` ⇒ on-demand. MUST match
    // the `params` it was staged from, mirroring step_one's identical use.
    per_eval: Option<&[f64]>,
) -> Result<f64, SimError> {
    let n_int = model.int_local_to_global.len();
    let n_tr = model.model.transitions.len();

    // Set up evaluation context (same as step_one)
    let mut int_s = IntState::new(n_int);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = RealState::new(model.real_local_to_global.len());

    let mut propensities = vec![0.0; n_tr];
    eval_propensities(model, &int_s, &real_s, params, t, dt, per_eval, &mut propensities)?;

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt, projected: None, aux: None, int_float_override: None, per_eval,
    };

    // Per-transition: is it deterministic? What's its sigma_sq?
    let mut is_determ = vec![false; n_tr];
    let mut sigma_sq_by_tr: Vec<Option<f64>> = vec![None; n_tr];
    for (i, tr) in model.model.transitions.iter().enumerate() {
        match &tr.draw_method {
            ir::transition::DrawMethod::Deterministic => { is_determ[i] = true; }
            ir::transition::DrawMethod::Overdispersed { .. } => {
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

        // gh#122: a sole-exit deterministic source member is a POINT MASS.
        // `step_one` records `count = clamp(round(rate*dt), 0, n_src)`, so its
        // density is 0 (log 1) iff the recorded flow matches, else -inf. Mark it
        // handled here (so it is NOT re-scored by the ungrouped Poisson loop
        // below, which lacks the n_src cap) and validate the exact count.
        // `compute_source_group_probs` still skips deterministic members (they
        // never enter the competing-risk split); because a source that mixes a
        // deterministic exit with another exit is rejected upstream, a
        // deterministic member here is always the group's only exit. No-op for
        // deterministic-free groups (byte-identical density).
        for &tr_idx in group {
            if is_determ[tr_idx] {
                let expected = ((propensities[tr_idx] * dt).round() as i64).clamp(0, n_src) as u64;
                if flows[tr_idx] != expected {
                    return Ok(f64::NEG_INFINITY);
                }
                handled[tr_idx] = true;
            }
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

    // gh#607: the recorded gamma multipliers are bound POSITIONALLY, in
    // `step_one`'s push order, and the walk above skips a whole source group
    // when `counts_before` has emptied it and skips a member whose rate has
    // fallen to zero — WITHOUT advancing `gamma_idx`. Evaluating this record
    // against a state that skips differently (an ancestor-sampling candidate,
    // or the constant offset of a splice) silently pairs every later
    // overdispersed group with the wrong multiplier, because the read above
    // falls back to `1.0` past the end of the slice.
    //
    // A state that cannot consume exactly the recorded noise is not a state
    // that could have produced it, so the density is zero. This turns a silent
    // wrong number into a rejection — the ancestor is not selectable, and a
    // splice whose shift changes the term set is refused.
    if gamma_idx != gammas.len() {
        log::debug!(
            "log_transition_density_substep: state consumes {gamma_idx} gamma \
             multipliers but the record holds {} — returning -inf (the recorded \
             noise is not producible at this state).",
            gammas.len());
        return Ok(f64::NEG_INFINITY);
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

/// Log-density of the gamma multipliers recorded at ONE substep, plus how many
/// of them this state consumes.
///
/// `log Gamma(g; dt/σ², σ²/dt)` for each recorded multiplier. Mean
/// `shape · scale = 1`, so the multiplier is constrained near 1 at high shape
/// (no overdispersion) and free to vary at low shape (high σ²).
///
/// **The term set is state-gated, which is why this is a function and not an
/// inlined loop.** A multiplier is scored only for a source group with
/// `n_src > 0` whose member has `rate > RATE_EPSILON` and is a non-deterministic
/// `overdispersed(...)` draw — every gate evaluated at `counts_before`. σ² itself
/// may not reference compartment state (a compile-time guard in
/// `compiled_model.rs` rejects that), so the σ² VALUE is state-independent — but
/// the SET of terms is not. A state offset that empties a source compartment, or
/// drives a rate to zero through a frequency-dependent force of infection, adds
/// or removes a real term of the target. Any consumer comparing two states must
/// therefore evaluate this at both, never assume it cancels (gh#607).
///
/// The return value is the caller's desync check: `gammas` is bound
/// POSITIONALLY in `step_one`'s push order, so a state that consumes a different
/// number has not reproduced the walk that recorded them.
///
/// σ² is evaluated at `counts_before`, mirroring the three sibling sites —
/// `step_one` (chain_binomial.rs), [`log_transition_density_substep`], and
/// `gamma_density_value_and_grad_substep` (pgas_grad.rs).
///
/// Each term is LEFT-FOLDED into `acc` rather than pre-summed and added once.
/// That is load-bearing, not style: the gradient path folds them into its
/// running energy the same way, and `gh#197`'s spine oracle asserts the two
/// agree BIT-exactly — pre-summing opens a ~6e-13 nat gap and trips it.
pub fn fold_gamma_multiplier_log_density_substep(
    model: &CompiledModel,
    counts_before: &[i64],
    gammas: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    per_eval: Option<&[f64]>,
    acc: &mut f64,
) -> usize {
    if gammas.is_empty() {
        return 0;
    }
    let n_tr = model.model.transitions.len();
    let mut int_s_local = IntState::new(model.int_local_to_global.len());
    int_s_local.counts.copy_from_slice(counts_before);
    let real_s_local = RealState::new(model.real_local_to_global.len());
    let ctx = EvalCtx {
        model, int_s: &int_s_local, real_s: &real_s_local,
        params, t, dt,
        projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    let mut consumed = 0usize;
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts_before[src_local].max(0);
        if n_src == 0 { continue; }
        // Recompute propensities for the rate gate (same start-of-step state).
        let mut local_props = vec![0.0; n_tr];
        let _ = eval_propensities(model, &int_s_local, &real_s_local,
            params, ctx.t, dt, per_eval, &mut local_props);
        for &tr_idx in group {
            let rate = local_props[tr_idx];
            if rate <= RATE_EPSILON { continue; }
            if let ir::transition::DrawMethod::Deterministic = model.model.transitions[tr_idx].draw_method {
                continue;
            }
            if let Some(ref resolved_od) = model.resolved.overdispersion[tr_idx] {
                let sigma_sq = eval_resolved(resolved_od, &ctx);
                if consumed < gammas.len() && sigma_sq > OVERDISP_SIGMA_SQ_FLOOR {
                    // Shared with the gradient path's energy via one helper so
                    // the two agree f64-exactly (gh#197 / the spine oracle).
                    *acc += crate::inference::obs_loglik::gamma_multiplier_log_density(
                        dt / sigma_sq, sigma_sq / dt, gammas[consumed]);
                }
                consumed += 1;
            }
        }
    }
    consumed
}

/// Complete-data log-likelihood: sum of transition densities + observation
/// densities over the full trajectory.
///
/// log p(y, X | θ) = log p(x₀ | θ)
///                 + Σ_s log p(x_s | x_{s-1}, θ, g_s)
///                 + Σ_k log p(y_k | project(x_{obs_k}), θ)
///
/// The initial-state density `log p(x₀ | θ)` comes from the laws the model
/// DECLARES (`init { I ~ poisson(rate = I0) }`), through the shared seam
/// `CompiledModel::initial_state_logpdf`. It is zero for a deterministic
/// `init {}`. Nothing is inferred: before the laws landed this term was a
/// Binomial the runtime attached to whichever compartment a finite-difference
/// probe found moving, so two chains of one fit could carry different targets
/// (gh#719).
pub fn complete_data_loglik(
    model: &CompiledModel,
    trajectory: &PGASTrajectory,
    params: &[f64],
    _observations: &[Observation],
    dt: f64,
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    obs_at_substep: &ObsAtSubstep,
) -> Result<LogLikComponents, SimError> {
    let n_substeps = trajectory.substeps.len();
    let n_tr = model.model.transitions.len();
    // gh#272 LICM: stage the per-eval prologue ONCE for this θ (`params` fixed for
    // the whole complete-data evaluation) and thread it into every per-substep
    // density/rate eval below. `None` ⇒ on-demand (LICM off / nothing hoistable).
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, model.model.simulation.t_start, dt);
    let per_eval = per_eval_scratch.as_deref();
    let mut transition_ll = 0.0;
    let mut observation_ll = 0.0;

    // `log p(x₀ | θ)` from the DECLARED initial-state laws, through the same
    // seam the sampler and the gradient use, so the three cannot disagree about
    // which entries are random. CSMC free particles carry integer counts only
    // (the real reservoir is not advanced), so the real half is empty — a real
    // compartment already fails the chain-binomial inference capability check
    // (gh#191) before reaching here.
    let initial_state_ll =
        model.initial_state_logpdf(&trajectory.initial_counts, &[], params)?;

    if !initial_state_ll.is_finite() {
        log::debug!("complete_data_loglik: -inf initial-state density ({initial_state_ll:.1})");
        return Ok(LogLikComponents {
            total: f64::NEG_INFINITY,
            transition: 0.0,
            observation: 0.0,
            initial_state: initial_state_ll,
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
            model, counts_before, &rec.flows, &rec.gammas, params, t, dt_s, per_eval,
        )?;
        if !td.is_finite() {
            log::debug!("complete_data_loglik: -inf transition density at substep {} (t={:.1})", s, t);
            return Ok(LogLikComponents {
                total: f64::NEG_INFINITY,
                transition: transition_ll + td,
                observation: observation_ll,
                initial_state: initial_state_ll,
            });
        }
        transition_ll += td;

        // Gamma multiplier density for each multiplier recorded at this substep
        // — a state-gated term set, so it goes through the one shared walk every
        // consumer uses (see `fold_gamma_multiplier_log_density_substep`).
        let gamma_consumed = fold_gamma_multiplier_log_density_substep(
            model, &rec.counts_before, &rec.gammas, params, t, dt_s, per_eval,
            &mut transition_ll,
        );
        if !rec.gammas.is_empty() && gamma_consumed != rec.gammas.len() {
            // Unreachable via `td` above, which now refuses a state that cannot
            // consume the record. Kept as the divergence detector between that
            // walk and this one.
            log::warn!(
                "gamma index mismatch at substep {}: tracked {} but trajectory recorded {} gammas",
                s, gamma_consumed, rec.gammas.len()
            );
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
            let total = initial_state_ll + transition_ll + observation_ll;
            if !total.is_finite() {
                log::debug!("complete_data_loglik: -inf after obs at substep {} (cumulative)", s);
                return Ok(LogLikComponents {
                    total: f64::NEG_INFINITY,
                    transition: transition_ll,
                    observation: observation_ll,
                    initial_state: initial_state_ll,
                });
            }
            // `cum_flows` blanket-zeroed (unchanged); the per-stream `acc` bins
            // per-stream — only Interval streams scheduled at THIS union index.
            cum_flows.fill(0);
            obs_model.reset_due_acc(obs_idx, &mut acc);
        }
    }

    Ok(LogLikComponents {
        total: initial_state_ll + transition_ll + observation_ll,
        transition: transition_ll,
        observation: observation_ll,
        initial_state: initial_state_ll,
    })
}

/// gh#82. Turn a **candidate** θ's complete-data likelihood outcome into the
/// score the MH ratio consumes.
///
/// [`complete_data_loglik`] already returns `Ok(−∞)` for the ordinary "this θ
/// cannot explain this trajectory / these data" outcomes. It returns `Err` only
/// when the *rate evaluation itself* is unusable at that θ — its single
/// fallible call is `log_transition_density_substep`, whose single fallible
/// call is `eval_propensities`, so the whole reachable error surface here is
/// `NonFiniteParameter`, `NumericalCollapse`, `TableLookup` and
/// `NegativePropensity`. Every one of those is θ-dependent: the same
/// expressions evaluated cleanly at the chain's *current* θ, which is why the
/// sweep is running at all. Propagating one out of `run_pgas` turned a single
/// bad proposal into a dead chain — and, via the CLI's `collect::<Result<…>>()?`
/// over the per-chain rayon loop (`cli/src/fit/pgas.rs`), a dead fit.
///
/// The discriminator is [`SimError::is_structural`] (gh#224) — the same one
/// every other whole-θ evaluation boundary uses (`fit/pmmh.rs`,
/// `fit/runner.rs`, `fit/dt_check.rs`, `profile.rs`), and the same verdict the
/// sibling NUTS branch in [`run_pgas`] already applies to a failed gradient
/// evaluation. A structural failure (`Validation`, `Unknown*`,
/// `ConfigMismatch`, …) fires for *every* θ, so it must surface rather than be
/// laundered into a rejection that leaves a meaningless fit looking successful.
///
/// gh#82 proposes `is_per_particle_recoverable()` here. That answers the
/// *per-particle* question — can a death mask absorb this inside one filter
/// call ([`super::degeneracy::DeathMask`]) — and is strictly narrower: it would
/// leave `NegativePropensity` and `DivisionByZero` killing the chain at a
/// proposed θ, contradicting their own documented classification ("θ-dependent
/// runtime conditions … reject this θ as −∞") and making PGAS disagree with
/// PMMH about the very same θ. Every per-particle-recoverable variant is
/// non-structural (pinned by `recoverable_errors_are_never_structural` in
/// `error.rs`), so the issue's acceptance criteria hold a fortiori.
fn theta_proposal_score(outcome: Result<LogLikComponents, SimError>) -> Result<f64, SimError> {
    match outcome {
        Ok(components) => Ok(components.total),
        Err(e) if e.is_structural() => Err(e),
        Err(e) => {
            // Not silent: the `NonFiniteParameter` diagnostic itself promises
            // "the chain rejects this proposal and continues; if you see
            // thousands of these warnings …", so the rejections have to be
            // observable somewhere.
            log::debug!("pgas: rejecting proposed θ — likelihood evaluation failed: {e}");
            Ok(f64::NEG_INFINITY)
        }
    }
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
    // A reference trajectory is one realization of the process, so its x₀ is a
    // DRAW from the same stream the substep loop below consumes — not the mean.
    // (For a deterministic `init {}` nothing is consumed, so the walk's draw
    // sequence is unchanged from before the laws landed.)
    let (init_int, _) = model.initial_state_draw(params, rng)?;
    let n_tr = model.model.transitions.len();

    // gh#272 LICM: stage the per-eval prologue ONCE for this θ (`params` fixed for
    // the whole reference walk) and thread it into every substep's rate eval.
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, model.model.simulation.t_start, dt);
    let per_eval = per_eval_scratch.as_deref();

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
        step_one(model, &mut counts, &mut flows, &mut real, params, t0, dt_s, per_eval, rng, &mut scratch)?;

        // Verify: density evaluation of this record won't produce k > n.
        // This catches state/flow mismatches before they cause -inf later.
        if cfg!(debug_assertions) {
            let verify_td = log_transition_density_substep(
                model, &counts_before, &flows, &scratch.gamma_used, params, t0, dt_s, per_eval,
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

/// Whether the reference trajectory's remaining substeps survive a CONSTANT
/// compartment offset — the shape every ancestor-sampling splice takes (gh#607).
///
/// **Why a splice is an offset.** Ancestor sampling replaces the reference
/// particle's PREFIX with candidate `j`'s while keeping the reference's own
/// noise — its recorded per-transition `flows` and `gammas` — for substeps
/// `s..T`. camdl's chain-binomial step is `counts_after = counts_before + A·u`
/// (`A` the stoichiometry, `u` the realized flows), so holding `u` fixed and
/// starting from `x_{s-1}^j` instead of the reference's own `x'_{s-1}` shifts
/// EVERY subsequent recorded state by the single constant vector
///
/// ```text
///   Δ_j = x_{s-1}^j − x'_{s-1}.
/// ```
///
/// No dynamics are re-simulated: the recorded per-substep net delta is reused
/// verbatim, and `Δ_j` rides through it unchanged.
///
/// **Why the offset needs screening.** The recorded flows were drawn from the
/// reference's own states, so at `x'_{t-1} + Δ_j` they can be IMPOSSIBLE — a
/// source group whose exits now exceed its (shrunken) occupancy is a
/// `Binom(k; n, p)` with `k > n`, density exactly zero. Selecting such an
/// ancestor would return a trajectory outside the target's support. This guard
/// answers "is `Δ` admissible from substep `s` onward?" in `O(n_compartments)`
/// after one backward pass over the reference, so every candidate can be
/// screened at every substep.
///
/// **What it does NOT certify.** The offset argument assumes the substep's net
/// state change is independent of the state it starts from. That holds for the
/// transition draws (the flows are given) but not for `events {}`, scheduled
/// compartment interventions, or a `balance {}` constraint, all of which
/// recompute counts from the state. Substeps at or after any scheduled effect
/// are therefore refused outright, and under `balance` a population-changing
/// offset is refused; the exact per-substep verification lives in
/// [`splice_log_ratio`].
pub struct SpliceGuard {
    n_comp: usize,
    /// Flat `[s * n_comp + i]`: the minimum over substeps `t ≥ s` of how far
    /// compartment `i` may be shifted DOWN before some recorded flow becomes
    /// impossible or some recorded count goes negative. A splice at `s` with
    /// offset `Δ` clears this test iff `Δ[i] + headroom[s][i] ≥ 0` for all `i`.
    headroom: Vec<i64>,
    /// `true` at `s` when some event or scheduled intervention fires at a
    /// substep `t ≥ s` — where the constant-offset argument does not hold.
    effect_at_or_after: Vec<bool>,
    /// The model rewrites one compartment from a `balance {}` expression every
    /// substep. A canonical `N − Σ others` balance transports a constant offset
    /// only when the offset conserves total population.
    has_balance: bool,
}

impl SpliceGuard {
    /// One backward pass over the reference trajectory. `firing`/`fire_steps`/
    /// `dt` are the same firing plan the producers use, so "does an effect fire
    /// at substep `t`" is answered by the one authority that fires them.
    pub fn from_reference(
        model: &CompiledModel,
        reference: &PGASTrajectory,
        fire_steps: &[std::collections::BTreeSet<i64>],
        dt: f64,
        firing: EffectFiring<'_>,
    ) -> Self {
        let n_comp = model.int_local_to_global.len();
        let n_substeps = reference.substeps.len();

        let mut effect_at_or_after = vec![false; n_substeps];
        let mut batch = crate::schedule::EffectBatch::default();
        for (s, rec) in reference.substeps.iter().enumerate() {
            fill_producer_batch(
                model, fire_steps, rec.t0 + rec.dt_substep, dt, s, firing, &mut batch,
            );
            effect_at_or_after[s] = !batch.is_empty();
        }
        for s in (0..n_substeps.saturating_sub(1)).rev() {
            effect_at_or_after[s] = effect_at_or_after[s] || effect_at_or_after[s + 1];
        }

        let mut headroom = vec![0i64; n_substeps * n_comp];
        let mut running = vec![i64::MAX; n_comp];
        for s in (0..n_substeps).rev() {
            let rec = &reference.substeps[s];
            // A source group's recorded exits must still fit in its occupancy
            // after the shift: `n_exit ≤ n_src + Δ[src]`.
            for &(src_local, ref group) in &model.source_groups {
                let exits: i64 = group.iter().map(|&tr| rec.flows[tr] as i64).sum();
                running[src_local] =
                    running[src_local].min(rec.counts_before[src_local] - exits);
            }
            // Every recorded count must stay non-negative after the shift.
            for i in 0..n_comp {
                running[i] = running[i].min(rec.counts_before[i]).min(rec.counts_after[i]);
            }
            headroom[s * n_comp..(s + 1) * n_comp].copy_from_slice(&running);
        }

        SpliceGuard {
            n_comp,
            headroom,
            effect_at_or_after,
            has_balance: model.balance.is_some(),
        }
    }

    /// Is splicing the reference's suffix from `substep` onto a prefix whose
    /// end-state differs from the reference's by `offset` admissible?
    ///
    /// A zero offset is the reference keeping its own ancestry — always
    /// admissible, and the fallback whenever nothing else is.
    pub fn offset_is_admissible(&self, substep: usize, offset: &[i64]) -> bool {
        if offset.iter().all(|&d| d == 0) {
            return true;
        }
        if self.effect_at_or_after[substep] {
            return false;
        }
        if self.has_balance && offset.iter().sum::<i64>() != 0 {
            return false;
        }
        let base = substep * self.n_comp;
        (0..self.n_comp).all(|i| offset[i].saturating_add(self.headroom[base + i]) >= 0)
    }

    /// Give every candidate whose splice is inadmissible a `−∞` weight, so the
    /// categorical draw cannot select a trajectory outside the target's
    /// support. `ref_recorded_before` is the reference trajectory's OWN
    /// recorded pre-state at this substep — the anchor the offset is measured
    /// against, NOT the reference slot's realized state (which already carries
    /// the offset of an earlier accepted splice).
    ///
    /// `j_ref` is skipped: keeping the current ancestry is the identity move,
    /// and its offset was screened when it was accepted — over a suffix that
    /// starts EARLIER, so its headroom bound is the tighter one.
    pub fn mask_inadmissible(
        &self,
        ancestor_log_w: &mut [f64],
        substep: usize,
        candidate_states: &[Vec<i64>],
        ref_recorded_before: &[i64],
        j_ref: usize,
    ) {
        let mut offset = vec![0i64; self.n_comp];
        for (j, slot) in ancestor_log_w.iter_mut().enumerate() {
            if j == j_ref || !slot.is_finite() {
                continue;
            }
            for i in 0..self.n_comp {
                offset[i] = candidate_states[j][i] - ref_recorded_before[i];
            }
            if !self.offset_is_admissible(substep, &offset) {
                *slot = f64::NEG_INFINITY;
            }
        }
    }
}

/// The reference trajectory's OWN per-substep log-densities — the baseline the
/// ancestor-sampling accept/reject ratio is centred on (gh#607).
///
/// Centring matters numerically, not mathematically: the ratio subtracts two
/// suffix sums of `T − s` terms each, and differencing them term-by-term
/// against a common baseline keeps the answer in single-digit magnitudes
/// instead of cancelling two large negatives. The baseline cancels exactly, so
/// any finite common reference would do; the reference's own densities are the
/// natural choice because they make the "keep the current ancestry" branch
/// evaluate to zero without any arithmetic at all.
///
/// `pub` alongside [`splice_log_ratio`] so the accept/reject ratio — the
/// quantity the kernel's invariance now hinges on — is testable against
/// [`complete_data_loglik`] in isolation, the same reason
/// [`fill_ancestor_log_weights`] is public.
pub struct ReferenceBaseline {
    /// `log f_θ(u'_t | x'_{t-1})` at each substep `t`.
    td: Vec<f64>,
    /// The gamma-multiplier log-density at each substep. Scored EXPLICITLY, not
    /// cancelled: σ² is state-independent but the set of multipliers a state
    /// consumes is not, so a splice's offset can add or remove a term (gh#607).
    gamma: Vec<f64>,
    /// `log g_θ(y_t | ·)` at each substep an observation is due, `0.0`
    /// elsewhere. Accumulated over the reference's own flow bins, exactly as
    /// [`complete_data_loglik`] does.
    obs_ll: Vec<f64>,
    /// Per-transition flow accumulation ENTERING each substep — the prefix of
    /// the interval bin that is still open at that point.
    ///
    /// [`splice_log_ratio`]'s zero-offset short-circuit is valid only if the
    /// candidate's whole walk would reproduce this baseline term for term. At
    /// `Δ = 0` the transition and gamma terms do so by construction, but the
    /// observation closing a straddled bin does not: it scores
    /// `acc_seed + flows`, so a candidate carrying a different partial bin
    /// gets a different total. These are what the seeds must match for the
    /// short-circuit to be sound (gh#720).
    cum_at: Vec<Vec<u64>>,
    /// Per-interval-stream accumulator entering each substep. Companion to
    /// [`Self::cum_at`]; see there.
    acc_at: Vec<Vec<u64>>,
}

impl ReferenceBaseline {
    /// What the reference charged for the observation due at substep `t`
    /// (`0.0` where none is due). Read-only accessor so a test can name the
    /// term a splice's observation delta is measured against.
    pub fn obs_ll_at(&self, t: usize) -> f64 {
        self.obs_ll[t]
    }
}

/// One forward pass over the reference trajectory, mirroring
/// [`complete_data_loglik`]'s fold/score/reset lifecycle so the baseline and
/// the spliced walk are the same computation on different states.
///
/// The gamma-multiplier density is scored here rather than cancelled. σ² may not
/// reference compartment state, but the SET of multipliers a state consumes is
/// gated on that state (`n_src > 0`, `rate > RATE_EPSILON`), so an offset can add
/// or remove a term — see [`fold_gamma_multiplier_log_density_substep`].
pub fn reference_baseline(
    model: &CompiledModel,
    reference: &PGASTrajectory,
    params: &[f64],
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    obs_at_substep: &ObsAtSubstep,
    per_eval: Option<&[f64]>,
) -> Result<ReferenceBaseline, SimError> {
    let n_substeps = reference.substeps.len();
    let n_tr = model.model.transitions.len();
    let mut td = vec![0.0; n_substeps];
    let mut gamma = vec![0.0; n_substeps];
    let mut obs_ll = vec![0.0; n_substeps];
    let mut cum_flows = vec![0u64; n_tr];
    let mut acc = vec![0u64; obs_model.n_interval_streams()];

    let mut cum_at = Vec::with_capacity(n_substeps);
    let mut acc_at = Vec::with_capacity(n_substeps);

    for (t, rec) in reference.substeps.iter().enumerate() {
        // Entering `t`: the state a splice at `t` must match to be a no-op.
        cum_at.push(cum_flows.clone());
        acc_at.push(acc.clone());

        td[t] = log_transition_density_substep(
            model, &rec.counts_before, &rec.flows, &rec.gammas, params,
            rec.t0, rec.dt_substep, per_eval,
        )?;
        fold_gamma_multiplier_log_density_substep(
            model, &rec.counts_before, &rec.gammas, params,
            rec.t0, rec.dt_substep, per_eval, &mut gamma[t],
        );
        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }
        if let Some(&obs_idx) = obs_at_substep.get(&t) {
            obs_model.fold_into_acc(&cum_flows, &mut acc);
            obs_ll[t] = obs_model.log_likelihood_from_flows_and_counts(
                &acc, &rec.counts_after, obs_idx, params);
            cum_flows.fill(0);
            obs_model.reset_due_acc(obs_idx, &mut acc);
        }
    }
    Ok(ReferenceBaseline { td, gamma, obs_ll, cum_at, acc_at })
}

/// `log S_j − log S_ref` for a candidate ancestor whose splice shifts the
/// reference's remaining trajectory by the constant `offset` (gh#607).
///
/// # The ratio this feeds, and what cancels
///
/// Lindsten, Jordan & Schön (2014), JMLR 15:2145–2184. The EXACT
/// ancestor-sampling weight for a non-Markovian model is their Eq. (3),
///
/// ```text
///   w̃^j_{s-1|T} = w^j_{s-1} · γ_T((x^j_{1:s-1}, x'_{s:T})) / γ_{s-1}(x^j_{1:s-1}),
/// ```
///
/// whose ratio expands (their Eq. 22) to the FULL remaining path,
///
/// ```text
///   γ_T/γ_{s-1} = Π_{t=s}^{T} g_θ(y_t | x_{1:t}) f_θ(x_t | x_{1:t-1}).
/// ```
///
/// camdl is non-Markovian in exactly this sense. The reference's suffix is a
/// sequence of realized integer FLOWS, not of states; grafting it onto
/// candidate `j` shifts every subsequent state by `Δ_j`, and the chain-binomial
/// flow density `f_θ(u'_t | x_{t-1})` depends on the state it was drawn from.
/// So the suffix does NOT cancel across candidates, and Eq. (17) — the
/// Markovian collapse to `w^j_{s-1} f_θ(x'_s | x^j_{s-1})` that `csmc_as` draws
/// from — is a proposal, not the target. (Contrast a model whose innovations
/// are state-independent: there the suffix factors are common to all `j` and
/// Eq. (17) is exact.)
///
/// LJS §6.1 sanctions the remedy directly: use the cheap distribution as an MH
/// proposal and let Eq. (21),
///
/// ```text
///   1 ∧ [ w̃^{i'}_{s-1|T} q(N | i') ] / [ w̃^N_{s-1|T} q(i' | N) ],
/// ```
///
/// carry the correction — "The MH accept/reject decision will then compensate
/// for the approximation error caused by the truncation" (§6.2, p. 2166,
/// describing precisely this pairing, at their stated `O(NTℓ + T²)` cost).
///
/// The proposal here is the *independence* kernel `q(· | i) = ρ̂(·)`, the
/// normalized Eq.-(17) weights `csmc_as` already computes (screened by
/// [`SpliceGuard`]). Writing `w̃^j_full = w̃^j_prop · S_j` with
///
/// ```text
///   S_j = Π_{t>s} f_θ(u'_t | x'_{t-1} + Δ_j)
///       · Π_{t≥s} f_γ(g'_t | x'_{t-1} + Δ_j)
///       · Π_{obs t ≥ s} g_θ(y_t | ·(Δ_j)),
/// ```
///
/// Eq. (21) collapses to `α = S_{i'} / S_N`:
///
/// - **`w^j_{s-1}` cancels** — the incoming importance weight appears in both
///   `w̃^j_full` and `w̃^j_prop` for the same `j`.
/// - **The substep-`s` transition factor `f_θ(u'_s | x^j_{s-1})` cancels** —
///   likewise. This is why the transition sum below starts at `t = s+1`.
/// - **The normalizers of `ρ̂` cancel**, which is what an independence proposal
///   buys; nothing here needs `Σ_l w̃^l`.
/// - **Nothing else cancels.** Every transition factor from `s+1` to `T`
///   survives, as does every observation term from `s` on (the proposal carries
///   no `g` factor at all), and the observation at `s` is exactly where a
///   spliced interval bin's hybrid accumulation is charged.
/// - **The gamma-multiplier density does NOT cancel, at any `t ≥ s` including
///   `t = s`.** σ² is state-independent, but the SET of multipliers a state
///   consumes is gated on that state, so an offset can add or remove a term;
///   and the proposal weight carries no multiplier density at all, so the
///   substep-`s` term has nothing to cancel against. See
///   [`fold_gamma_multiplier_log_density_substep`].
///
/// # Arguments
///
/// `cum_seed`/`acc_seed` are the candidate's own partial flow accumulation
/// carried into substep `s` — the prefix half of the first observation bin the
/// splice straddles. `Ok(NEG_INFINITY)` means the splice is impossible (a
/// recorded flow cannot be produced at the shifted state, an observation has
/// zero density, or the `balance {}` rewrite does not transport the offset);
/// the walk stops at the first such term.
#[allow(clippy::too_many_arguments)]
pub fn splice_log_ratio(
    model: &CompiledModel,
    reference: &PGASTrajectory,
    params: &[f64],
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    obs_at_substep: &ObsAtSubstep,
    per_eval: Option<&[f64]>,
    baseline: &ReferenceBaseline,
    substep: usize,
    offset: &[i64],
    cum_seed: &[u64],
    acc_seed: &[u64],
) -> Result<f64, SimError> {
    // Keeping the current ancestry is the identity move: every term is its own
    // baseline. Short-circuit it exactly, without arithmetic.
    //
    // gh#720: a zero compartment offset is NOT sufficient. Two particles can
    // agree on compartment counts and disagree on the partly-filled interval
    // bin they carry into `substep` — routine immediately after a resample
    // duplicates particles — and the observation closing that bin then scores
    // a different total. The identity holds only if the accumulator seeds also
    // match the baseline's own state entering `substep`; otherwise fall
    // through and let the walk charge the difference.
    if offset.iter().all(|&d| d == 0)
        && cum_seed == baseline.cum_at[substep].as_slice()
        && acc_seed == baseline.acc_at[substep].as_slice()
    {
        return Ok(0.0);
    }

    let n_comp = offset.len();
    let mut cum_flows = cum_seed.to_vec();
    let mut acc = acc_seed.to_vec();
    let mut shifted_before = vec![0i64; n_comp];
    let mut shifted_after = vec![0i64; n_comp];
    let real_s = RealState::new(model.real_local_to_global.len());
    let mut total = 0.0;

    for t in substep..reference.substeps.len() {
        let rec = &reference.substeps[t];
        for i in 0..n_comp {
            shifted_before[i] = rec.counts_before[i] + offset[i];
            shifted_after[i] = rec.counts_after[i] + offset[i];
        }

        if t > substep {
            let td = log_transition_density_substep(
                model, &shifted_before, &rec.flows, &rec.gammas, params,
                rec.t0, rec.dt_substep, per_eval,
            )?;
            if !td.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            total += td - baseline.td[t];
        }

        // The gamma multipliers, at EVERY t from `substep` on — including
        // `t == substep`, unlike the transition factor. The proposal weight is
        // `log_weights[j] + log_transition_density_substep(..)`, and that
        // function does not carry the multiplier density, so there is nothing
        // for the substep-`substep` gamma term to cancel against.
        let mut gamma_ll = 0.0;
        let gamma_consumed = fold_gamma_multiplier_log_density_substep(
            model, &shifted_before, &rec.gammas, params,
            rec.t0, rec.dt_substep, per_eval, &mut gamma_ll,
        );
        if gamma_consumed != rec.gammas.len() {
            // The shifted state skips a source group or a zero-rate member that
            // the record's positional gamma binding assumed. Unproducible noise.
            return Ok(f64::NEG_INFINITY);
        }
        total += gamma_ll - baseline.gamma[t];

        // A `balance {}` model rewrites one compartment from an expression over
        // the others every substep, so the constant offset only survives if the
        // expression transports it. Verify rather than assume: the recorded
        // (unshifted) state satisfies this fixed point by construction, so a
        // failure here means the shift is not a path this model can produce.
        if let Some(bal) = &model.balance {
            let mut int_s = IntState::new(n_comp);
            int_s.counts.copy_from_slice(&shifted_after);
            let ctx = EvalCtx {
                model, int_s: &int_s, real_s: &real_s, params,
                t: rec.t0 + rec.dt_substep, dt: rec.dt_substep,
                projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            if (eval_resolved(&bal.expr, &ctx).round() as i64)
                != shifted_after[bal.local_int_idx]
            {
                return Ok(f64::NEG_INFINITY);
            }
        }

        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }
        if let Some(&obs_idx) = obs_at_substep.get(&t) {
            obs_model.fold_into_acc(&cum_flows, &mut acc);
            let ll = obs_model.log_likelihood_from_flows_and_counts(
                &acc, &shifted_after, obs_idx, params);
            if !ll.is_finite() {
                return Ok(f64::NEG_INFINITY);
            }
            total += ll - baseline.obs_ll[t];
            cum_flows.fill(0);
            obs_model.reset_due_acc(obs_idx, &mut acc);
        }
    }
    Ok(total)
}

/// Fill `ancestor_log_w` with the reference particle's ancestor-sampling weights.
///
/// Lindsten, Jordan & Schön (2014), "Particle Gibbs with Ancestor Sampling",
/// JMLR 15:2145–2184, Eq. (3) / Eq. (17): the reference's ancestor index `a^N_s`
/// is drawn with `P(a^N_s = j) ∝ w̃_j`, where
///
/// ```text
///   log w̃_j = log w_{s-1}^j  +  log f_θ(x'_s | x_{s-1}^j).
/// ```
///
/// BOTH factors are load-bearing. `log_weights[j] = log w_{s-1}^j` is the prior
/// probability of ancestor path `j` (the previous substep's importance weight);
/// `td = log f_θ(x'_s | x_{s-1}^j)` is the likelihood of the reference's move
/// from that ancestor's state to its substep-`s` state `x'_s`. The common future
/// factor `p_θ(x'_{s+1:T}, y_{s:T} | x'_s)` is independent of `j` and cancels in
/// the softmax, so it is omitted.
///
/// The candidate states are the PRE-RESAMPLE ensemble
/// (`prev_counts_for_ancestor[j]`, captured before the step-1 resample), so the
/// paired prior weight is `log_weights[j]` — the pre-resample importance weight
/// of the *same* original slot `j` (the resample never reshuffles `log_weights`).
/// Dropping it biases the draw whenever the incoming weights are non-uniform
/// (every substep following an observation) and forfeits the Theorem-1 invariance
/// of the PGAS kernel.
///
/// # Every slot, including the reference's, is scored at its OWN state
///
/// `prev_counts_for_ancestor[j_ref]` is the reference SLOT's realized
/// end-of-`s−1` state, which is **not** `reference.substeps[s].counts_before`
/// once an earlier splice this sweep has re-anchored the slot: it is that value
/// plus the accumulated constant offset `Δ`, and `Δ` persists for the rest of
/// the sweep. Scoring the reference slot at the recorded (unshifted) state
/// instead breaks the one cancellation the accept/reject ratio is built on —
/// [`splice_log_ratio`] starts its transition sum at `s+1` *because* the
/// substep-`s` factor `f_θ(u'_s | x^j_{s-1})` is already in this weight for the
/// same `j`. Evaluate the two at different states and `α` acquires a spurious
/// `f_θ(u'_s | x_ref + Δ) / f_θ(u'_s | x_ref)`, which is not π-invariant
/// (gh#718).
///
/// Extracted from [`csmc_as`] and made `pub` so this weight — the quantity the
/// invariance proof hinges on — is unit-testable in isolation.
#[allow(clippy::too_many_arguments)]
pub fn fill_ancestor_log_weights(
    ancestor_log_w: &mut [f64],
    model: &CompiledModel,
    prev_counts_for_ancestor: &[Vec<i64>],
    ref_flows: &[u64],
    ref_gammas: &[f64],
    log_weights: &[f64],
    params: &[f64],
    t: f64,
    step_dt: f64,
    per_eval: Option<&[f64]>,
) -> Result<(), SimError> {
    // Parallel (gh#209): each slot is an independent transition-density eval over
    // a read-only state; the categorical draw in `csmc_as` reads the buffer only
    // after this barrier, so concurrency is byte-identical to a serial loop.
    let results: Vec<Result<(), SimError>> = ancestor_log_w
        .par_iter_mut()
        .enumerate()
        .map(|(j, slot)| {
            // gh#audit-H8: ancestor states are the pre-resample ensemble.
            let td = log_transition_density_substep(
                model, &prev_counts_for_ancestor[j], ref_flows, ref_gammas, params, t, step_dt,
                per_eval,
            )?;
            // Eq (17): log w̃_j = log w_{s-1}^j + log f_θ(x'_s | x_{s-1}^j).
            *slot = log_weights[j] + td;
            Ok(())
        })
        .collect();
    for r in results {
        r?;
    }
    Ok(())
}

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
    seed: u64,
    obs_at_substep: &ObsAtSubstep,
    firing: EffectFiring<'_>,
) -> Result<(PGASTrajectory, CSMCDiagnostics), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = reference.substeps.len();
    // gh#272 LICM: stage the per-eval prologue ONCE for this sweep (θ = `params`
    // fixed across the conditional filter; NUTS perturbs θ only between sweeps)
    // and thread it into every producer/density eval below. `None` ⇒ on-demand.
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, t_start, dt);
    let per_eval = per_eval_scratch.as_deref();
    let n_tr = model.model.transitions.len();
    let j_ref = n_particles - 1; // reference particle is the last slot

    // gh#53: resolve fire_steps once at the runtime dt for the
    // free-particle propagation step_one calls below.
    let fire_steps = model.resolve_fire_steps(dt, params);

    // gh#607: one backward pass over the reference, so each ancestor-sampling
    // draw below can refuse — in O(n_compartments) — a splice whose constant
    // state offset would make one of the reference's own recorded flows
    // impossible further down the trajectory.
    let splice_guard = SpliceGuard::from_reference(model, reference, &fire_steps, dt, firing);

    // gh#607: the reference's own per-substep densities, so the ancestor
    // accept/reject ratio below is accumulated as a difference against a common
    // baseline rather than as two large cancelling sums.
    let baseline =
        reference_baseline(model, reference, params, obs_model, obs_at_substep, per_eval)?;

    // Per-particle RNGs via ChaCha8 stream counter (IM1 fix 2026-04-19).
    let mut rngs = init_particle_rngs(seed, n_particles, 0);

    // Each free particle draws its own initial state, through the same seam
    // every forward path uses. For a model whose `init {}` declares a law that
    // is a genuine per-particle draw — the spread CSMC selects among, and what
    // makes an initial-state parameter estimable at all. For a deterministic
    // `init {}` the seam consumes nothing from the RNG and every free particle
    // gets the same state, exactly as before.
    let mut counts: Vec<Vec<i64>> = (0..n_particles)
        .map(|j| -> Result<Vec<i64>, SimError> {
            if j == j_ref {
                return Ok(reference.initial_counts.clone());
            }
            let (int_s, real_s) = model.initial_state_draw(params, &mut rngs[j])?;
            let mut c = int_s.counts;
            // Re-apply the balance constraint to the drawn state. The DECLARED
            // expression, evaluated exactly as `lifecycle.rs` evaluates it every
            // substep — not the hardcoded `total_pop − Σothers` this used to
            // assume. Any model whose balance expression is not that got two
            // different initial states from the two paths.
            if let Some(ref bal) = model.balance {
                let int_view = crate::state::IntState::from_vec(c.clone());
                let ctx = crate::propensity::EvalCtx {
                    model, int_s: &int_view, real_s: &real_s, params,
                    t: t_start, dt: 0.0, projected: None, aux: None,
                    int_float_override: None, per_eval,
                };
                c[bal.local_int_idx] =
                    crate::resolved_expr::eval_resolved(&bal.expr, &ctx).round() as i64;
            }
            Ok(c)
        })
        .collect::<Result<Vec<_>, _>>()?;

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
    // that differ from the deterministic initial_state_mean(params).
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
    // gh#718: how often an ancestry was actually drawn, and how often ancestor
    // sampling was consequently skipped. Counters only — no RNG.
    let mut n_resampled: usize = 0;
    let mut n_as_skipped_no_resample: usize = 0;
    // gh#607 follow-up: ancestor-sampling acceptance accounting. Counters only
    // — they consume no RNG, so trajectories stay bit-identical.
    let mut n_as_proposed: usize = 0;
    let mut n_as_accepted: usize = 0;
    let mut n_as_refused_inadmissible: usize = 0;

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
        // gh#607: the interval accumulators must be snapshotted HERE, in the
        // same index space as `prev_counts_for_ancestor`. `ref_ancestor` indexes
        // the PRE-resample ensemble, while `cum_flows`/`acc` are permuted BY the
        // resample below — reading `cum_flows[ref_ancestor]` after it would be
        // an index-space bug that silently pairs one particle's state with
        // another's accumulated flows.
        let prev_cum_flows_for_ancestor: Vec<Vec<u64>> = cum_flows.clone();
        let prev_acc_for_ancestor: Vec<Vec<u64>> = acc.clone();

        // ── 1. Resample free particles (ancestor selection from prev weights) ──
        // Between observations there is no new information, so every weight is
        // equal and resampling would only duplicate particles for nothing. Skip
        // it — and record that we skipped, because ancestor sampling below is
        // only a legal move where an ancestry was actually drawn (gh#718).
        let weights_are_uniform = log_weights.iter().all(|&w| (w - log_weights[0]).abs() < 1e-10);
        let did_resample = !weights_are_uniform;

        let substep_ancestors: Vec<usize> = if !did_resample {
            // Identity: each particle is its own ancestor.
            (0..n_particles).collect()
        } else {
            // gh#718: the CONDITIONAL resample. The reference keeps itself and
            // the other `n-1` slots draw independently from `categorical(W)`
            // over the whole ensemble — including the reference, so its history
            // is inherited as often as its weight warrants. See
            // `conditional_multinomial_resample` for why this is not the
            // systematic scheme the unconditional filters use.
            let indices =
                conditional_multinomial_resample(&log_weights, j_ref, &mut resample_rng);
            let mut new_counts = Vec::with_capacity(n_particles);
            let mut new_cum_flows = Vec::with_capacity(n_particles);
            // Phase 2a: the per-stream `acc` bins travel with the particle,
            // following EXACTLY the `cum_flows` resampling (reference kept,
            // free particles take their ancestor's bins).
            let mut new_acc = Vec::with_capacity(n_particles);
            // No `j == j_ref` arm: `indices[j_ref] == j_ref` by construction, so
            // the reference keeps itself through the ordinary path. The special
            // case used to be load-bearing precisely because the index it
            // overwrote was garbage — which is what hid gh#718.
            for j in 0..n_particles {
                new_counts.push(counts[indices[j]].clone());
                new_cum_flows.push(cum_flows[indices[j]].clone());
                new_acc.push(acc[indices[j]].clone());
            }
            counts = new_counts;
            cum_flows = new_cum_flows;
            acc = new_acc;
            n_resampled += 1;
            indices
        };

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
                    // gh#272 LICM: scratch staged once for this sweep, threaded in.
                    params, t, step_dt, per_eval, rng, scratch,
                )?;

                std::mem::swap(gammas, &mut scratch.gamma_used);
                Ok(())
            })
            .collect();
        for r in prop_results { r?; }

        // ── 3. Clamp the reference particle's NOISE ──
        // The reference contributes its recorded flows and gamma multipliers;
        // the STATE they act on is settled in step 4, because ancestor sampling
        // may re-anchor this slot on a different prefix. `prev_counts[j_ref]`
        // currently holds the reference slot's own realized end-of-(s−1) state
        // (saved at step 2 — the resample never reshuffles the reference slot).
        let ref_rec = &reference.substeps[s];
        substep_flows[j_ref].copy_from_slice(&ref_rec.flows);
        substep_gammas[j_ref].clear();
        substep_gammas[j_ref].extend_from_slice(&ref_rec.gammas);

        // ── 4. Ancestor sampling for reference particle ──
        // Draw the reference's ancestor a^N_s ∝ w̃_j = w_{s-1}^j · f_θ(x'_s |
        // x_{s-1}^j) (Lindsten, Jordan & Schön 2014, Eq. 3/17). The reference's
        // own gamma noise enters the density (given this noise, how likely is
        // reaching x'_s from ancestor j?). `fill_ancestor_log_weights` owns the
        // weight formula; the categorical draw + lineage bookkeeping stay here.
        //
        // gh#718 defect 2: this move is legal ONLY where step 1 actually drew an
        // ancestry. Where it did not, every particle kept its own history, so
        // the ancestry is the identity and moving the reference onto another
        // particle's prefix produces a configuration the resampling step gave
        // probability ZERO — outside the support of the distribution this kernel
        // is supposed to leave invariant. The gate is `did_resample`, a fact
        // about what step 1 DID, deliberately not "does this substep carry an
        // observation": weights can come out equal even at an observation (every
        // particle scoring it identically), and then the resample is skipped and
        // an ancestor move here would be exactly the invalid one. LJS §6 permit
        // performing ancestor sampling only on some substeps; the cost is mixing.
        if !did_resample {
            n_as_skipped_no_resample += 1;
            let mut step_ancestors = substep_ancestors;
            step_ancestors[j_ref] = j_ref;
            ancestors.push(step_ancestors);
            for i in 0..counts[j_ref].len() {
                counts[j_ref][i] =
                    prev_counts[j_ref][i] + (ref_rec.counts_after[i] - ref_rec.counts_before[i]);
            }
        } else {
            fill_ancestor_log_weights(
                &mut ancestor_log_w,
                model,
                &prev_counts_for_ancestor,
                &ref_rec.flows,
                &ref_rec.gammas,
                &log_weights,
                params,
                t,
                step_dt,
                per_eval,
            )?;

            // gh#607: refuse a candidate whose splice would shift the reference's
            // remaining recorded flows onto states that cannot produce them.
            splice_guard.mask_inadmissible(
                &mut ancestor_log_w,
                s,
                &prev_counts_for_ancestor,
                &ref_rec.counts_before,
                j_ref,
            );

            // PROPOSE from categorical(softmax(ancestor_log_w)) — the screened
            // Eq.-(17) weights, used as LJS §6.1's independence proposal `ρ̂`.
            // Degenerate case (all -inf): keep the reference's own history to
            // maintain internal consistency — the reference's flows at
            // substep s were produced from the reference's state at s-1.
            let proposed = match sample_categorical_log(&ancestor_log_w, &mut resample_rng) {
                Some(j) => j,
                None => { n_degenerate += 1; j_ref }
            };

            // ACCEPT/REJECT (LJS Eq. 21). Eq. (17) omits the spliced suffix's
            // dependence on the ancestor, which for camdl's state-dependent flow
            // densities does NOT cancel; the MH step compensates. Everything but
            // the suffix ratio cancels — see `splice_log_ratio`.
            let ref_ancestor = if proposed == j_ref {
                j_ref
            } else {
                n_as_proposed += 1;
                let offset_of = |state: &[i64]| -> Vec<i64> {
                    state.iter().zip(&ref_rec.counts_before).map(|(a, b)| a - b).collect()
                };
                let log_s_prop = splice_log_ratio(
                    model, reference, params, obs_model, obs_at_substep, per_eval,
                    &baseline, s,
                    &offset_of(&prev_counts_for_ancestor[proposed]),
                    &prev_cum_flows_for_ancestor[proposed],
                    &prev_acc_for_ancestor[proposed],
                )?;
                // The current ancestry's own suffix ratio. Exactly zero — and
                // free — until some earlier splice this sweep has already
                // offset the reference slot.
                let log_s_ref = splice_log_ratio(
                    model, reference, params, obs_model, obs_at_substep, per_eval,
                    &baseline, s,
                    &offset_of(&prev_counts[j_ref]),
                    &cum_flows[j_ref],
                    &acc[j_ref],
                )?;
                let accept = if log_s_prop == f64::NEG_INFINITY {
                    n_as_refused_inadmissible += 1;
                    false
                } else if log_s_ref == f64::NEG_INFINITY {
                    // The chain cannot be sitting on a zero-density suffix, but
                    // if it somehow is, any finite proposal is an improvement.
                    true
                } else {
                    let log_alpha = log_s_prop - log_s_ref;
                    log_alpha >= 0.0 || resample_rng.uniform().ln() < log_alpha
                };
                if accept {
                    n_as_accepted += 1;
                    proposed
                } else {
                    j_ref
                }
            };

            // gh#607: RE-ANCHOR. The reference slot descends from `ref_ancestor`
            // now, so its pre-state is that ancestor's end-state — otherwise the
            // traceback stitches a record whose `counts_before` is the
            // reference's own and the returned trajectory JUMPS in state at the
            // splice, a jump `complete_data_loglik` never charges. Applying the
            // reference's recorded NET DELTA to the new pre-state carries the
            // substep's realized change (transitions, and — where
            // `SpliceGuard` admitted the splice — nothing else) onto the new
            // lineage, which is exactly the constant-offset shift.
            if ref_ancestor != j_ref {
                prev_counts[j_ref].copy_from_slice(&prev_counts_for_ancestor[ref_ancestor]);
                // gh#607: the flow accumulators are part of the extended state
                // for an Interval (incidence) stream — the observation closing a
                // bin scores flows summed since the bin opened. The reference
                // slot now descends from `ref_ancestor`, so the bin it is
                // halfway through is that ancestor's, not its own. Without this
                // re-sync the filter weight the slot receives at the next
                // observation scores a bin no trajectory in the ensemble ever
                // walked, and the traceback returns a hybrid interval no weight
                // ever saw. Both snapshots are in the PRE-resample index space
                // `ref_ancestor` lives in.
                cum_flows[j_ref].copy_from_slice(&prev_cum_flows_for_ancestor[ref_ancestor]);
                acc[j_ref].copy_from_slice(&prev_acc_for_ancestor[ref_ancestor]);
            }
            for i in 0..counts[j_ref].len() {
                counts[j_ref][i] =
                    prev_counts[j_ref][i] + (ref_rec.counts_after[i] - ref_rec.counts_before[i]);
            }

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
    // gh#688: the same decision, kept resolved in time. Counters only — no RNG,
    // no effect on the path — and the arrays are stack-allocated, so the loop
    // gains one integer divide and one increment per substep.
    let mut renewal_bins = RenewalBins::new(n_substeps);
    for s in (0..n_substeps).rev() {
        let renewed = particle != j_ref;
        if !renewed { n_from_ref += 1; }
        renewal_bins.record(s, renewed);
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
                model, &rec.counts_before, &rec.flows, &rec.gammas, params, t, rec.dt_substep, per_eval,
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
        renewal_by_bin: renewal_bins.finish(),
        n_degenerate,
        n_resampled,
        n_as_skipped_no_resample,
        n_substeps,
        n_as_proposed,
        n_as_accepted,
        n_as_refused_inadmissible,
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

/// Prior log-density AND its gradient on the z (unconstrained) scale.
///
/// Delegates the density computation to `Prior::log_density` and computes
/// only the gradient part here. The chain rule converts d(log prior)/dθ
/// to d/dz via `param.transform_deriv(z)`. `pub(crate)` so the ODE-NUTS target
/// (`ode_nuts`) reuses the SAME prior-gradient authority PGAS's NUTS target uses,
/// rather than re-deriving it (gh#275 Phase 2).
pub(crate) fn prior_log_density_and_grad_z(
    prior: &Prior, param: &EstimatedParam, theta: f64, z: f64,
) -> (f64, f64) {
    let lp = prior.log_density(theta, z);
    let density = match prior {
        // Hierarchical priors need an env-aware density AND gradient to
        // drive NUTS correctly. PGAS+NUTS with hierarchical leaves is
        // tracked as Gate 3b — needs env threaded through this function
        // signature. For Gate 3a (PMMH + hierarchical), PMMH does not
        // call this function, and `run_pgas` refuses hierarchical priors up
        // front (gh#175), so this arm is a defensive -inf.
        Prior::Hierarchical(_) => return (f64::NEG_INFINITY, 0.0),
        Prior::Fixed(d) => d,
    };
    let dlp_dz = match density {
        Density::Flat => 0.0,
        Density::Uniform { lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            0.0 // flat density → zero gradient inside support
        }
        Density::Normal { mean, sd } => {
            let dlp_dtheta = -(theta - mean) / (sd * sd);
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::TransformedNormal { mean, sd } => {
            // d/dz of the NATURAL-scale log-normal density log p(θ(z)).
            // log_density returns log N(z; μ, σ) − z (it pre-subtracts the
            // Log Jacobian z), so its z-derivative is −(z−μ)/σ² − 1. The
            // caller adds jacobian_grad = +1, recovering d/dz log N(z) =
            // −(z−μ)/σ². Omitting the −1 here left the NUTS gradient for
            // log_normal priors off by +1 (uncovered: the only FD gradient
            // test used a flat prior).
            -(z - mean) / (sd * sd) - 1.0
        }
        Density::HalfNormal { sigma } => {
            if theta < 0.0 { return (lp, 0.0); }
            let dlp_dtheta = -theta / (sigma * sigma);
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::Beta { alpha, beta } => {
            if theta <= 0.0 || theta >= 1.0 { return (lp, 0.0); }
            let dlp_dtheta = (alpha - 1.0) / theta - (beta - 1.0) / (1.0 - theta);
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::Gamma { shape, rate } => {
            if theta <= 0.0 { return (lp, 0.0); }
            let dlp_dtheta = (shape - 1.0) / theta - rate;
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::Exponential { rate } => {
            if theta < 0.0 { return (lp, 0.0); }
            let dlp_dtheta = -rate;
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::LogUniform { lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            // d/dθ[−ln θ − const] = −1/θ; chain to z. With the Log transform
            // this is −1, which the caller's jacobian_grad (+1) cancels → the
            // z-scale density is flat, as it must be.
            let dlp_dtheta = -1.0 / theta;
            dlp_dtheta * param.transform_deriv(z)
        }
        Density::TruncatedNormal { mean, sd, lower, upper } => {
            if theta < *lower || theta > *upper { return (lp, 0.0); }
            // The normalizer Z is constant in θ, so only the Gaussian kernel
            // contributes: d/dθ[−0.5((θ−μ)/σ)²] = −(θ−μ)/σ².
            let dlp_dtheta = -(theta - mean) / (sd * sd);
            dlp_dtheta * param.transform_deriv(z)
        }
    };
    (lp, dlp_dz)
}

// ═══════════════════════════════════════════════════════════════════
// Rung state for parallel tempering
// ═══════════════════════════════════════════════════════════════════

/// Log Metropolis acceptance ratio for a replica-exchange swap between rung `i`
/// (inverse temperature `beta_i`, currently holding a state with untempered
/// log-likelihood `ll_i`) and rung `j`.
///
/// # Derivation
///
/// Rung `k` targets `π_k(x) ∝ L(x)^{β_k} · p(x)`. The proposal exchanges the two
/// rungs' states and is its own reverse, so the Hastings ratio is 1 and the
/// acceptance ratio is just the ratio of joint target densities:
///
/// ```text
/// log(now)   = β_i·ℓ_i + β_j·ℓ_j + log p(x_i) + log p(x_j)
/// log(after) = β_i·ℓ_j + β_j·ℓ_i + log p(x_j) + log p(x_i)
/// ```
///
/// The prior terms are identical — the same two states appear either way, only
/// assigned to different rungs — so they cancel, leaving
///
/// ```text
/// log α = (β_i − β_j)(ℓ_j − ℓ_i)
/// ```
///
/// Read it in words: `i` is the colder rung, so `(β_i − β_j) > 0` and the sign
/// is the sign of `(ℓ_j − ℓ_i)`. **Accept when the hotter rung holds the better
/// state** — i.e. when the free-roaming chain has found somewhere good, move it
/// down to the chain whose draws we report. That is the entire purpose of the
/// ladder.
///
/// # Why this is a named function rather than an inline expression (gh#550)
///
/// It shipped inverted, as `(β_i − β_j)(ℓ_i − ℓ_j)`, from 2026-04-09 to
/// 2026-08-10 — so it rejected exactly the swaps above and accepted their
/// opposites, moving states from the deliberately-flattened rungs INTO the cold
/// one. Every surface signal stayed healthy (finite log-liks, plausible swap
/// rates), and the inline comment restated the same wrong formula, so review saw
/// agreement rather than a contradiction.
///
/// Two things follow, both deliberate here. The comment above **derives** the
/// result instead of restating it, so it can disagree with the code. And the
/// expression is a function so it can be tested without running a fit — the
/// direct reason no test pinned it before.
///
/// The likely origin of the slip: the physics literature states this in
/// energies, `Δ = (β_i − β_j)(E_i − E_j)` (Hukushima & Nemoto 1996, eq. 6), with
/// Boltzmann weight `e^{−βE}`. Ours is `L^β = e^{βℓ}`, so `E = −ℓ`; substituting
/// without the sign flip produces exactly the shipped expression.
///
/// Incident: `docs/dev/incidents/2026-08-10-pgas-tempering-swap-sign.md`.
#[inline]
fn swap_log_alpha(beta_i: f64, beta_j: f64, ll_i: f64, ll_j: f64) -> f64 {
    (beta_i - beta_j) * (ll_j - ll_i)
}

#[cfg(test)]
mod swap_log_alpha_tests {
    use super::swap_log_alpha;
    use crate::inference::mh_accept;
    use crate::rng::StatefulRng;

    /// gh#471. The swap site keeps its own `log_alpha >= 0.0 || u_ln < log_alpha`
    /// form rather than routing through `mh_accept`, because the two consume
    /// RNG differently (`||` short-circuits on a certain accept). They are kept
    /// only because they make the same DECISION — and that equivalence is
    /// CONDITIONAL, which is what this pins.
    ///
    /// The condition is `u ∈ [0, 1)`: `u_ln` is then strictly negative, so
    /// `log_alpha == 0.0` accepts under both forms. `log_alpha == 0.0` is not a
    /// corner here — it is exactly what equal βs or equal log-likelihoods
    /// produce. If `uniform()` ever moved to a convention that can return 1.0,
    /// `u_ln` could reach 0 and the two forms would silently disagree at that
    /// value.
    #[test]
    fn swap_forms_agree_given_half_open_uniform() {
        // The precondition itself, asserted rather than assumed.
        let mut rng = StatefulRng::new(9);
        for _ in 0..10_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u), "uniform() returned {u}, outside [0, 1)");
        }

        let u_lns = [
            f64::NEG_INFINITY,          // u = 0
            (1e-300f64).ln(),
            (0.5f64).ln(),
            (1.0 - f64::EPSILON / 2.0).ln(), // the largest u the RNG can return
        ];
        let alphas = [
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            0.0, -0.0, 50.0, -50.0, 1e-300, -1e-300,
        ];
        for &la in &alphas {
            for &u_ln in &u_lns {
                let swap_form = la >= 0.0 || u_ln < la;
                assert_eq!(swap_form, mh_accept(la, u_ln),
                    "forms disagree at log_alpha={la}, u_ln={u_ln} — the swap \
                     site may no longer keep its own form");
            }
        }
    }

    /// The invariant, not an example: for arbitrary rungs and states, the
    /// returned value must equal the log ratio of the joint product target
    /// computed INDEPENDENTLY from its definition.
    ///
    /// This is the test that would have caught gh#550. A single worked case can
    /// be satisfied by a coincidentally-correct wrong formula; this cannot,
    /// because the oracle is built from `π_k ∝ L^{β_k}·p` rather than from any
    /// rearrangement of the expression under test.
    #[test]
    fn matches_the_product_target_ratio_computed_independently() {
        // Oracle: log Π(after) − log Π(now), with the priors written out and
        // cancelled by construction rather than by algebra.
        fn oracle(beta_i: f64, beta_j: f64, ll_i: f64, ll_j: f64) -> f64 {
            let log_now   = beta_i * ll_i + beta_j * ll_j;
            let log_after = beta_i * ll_j + beta_j * ll_i;
            log_after - log_now
        }

        let betas = [1.0, 0.8, 0.5, 0.25, 0.05];
        let lls   = [-1.0, -12.5, -200.0, -1e4, 0.0, -3.25];
        let mut checked = 0usize;
        for (bi, &beta_i) in betas.iter().enumerate() {
            for &beta_j in betas.iter().skip(bi + 1) {
                for &ll_i in &lls {
                    for &ll_j in &lls {
                        let got = swap_log_alpha(beta_i, beta_j, ll_i, ll_j);
                        let want = oracle(beta_i, beta_j, ll_i, ll_j);
                        assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0),
                            "beta_i={beta_i} beta_j={beta_j} ll_i={ll_i} ll_j={ll_j}: \
                             got {got}, product-target ratio is {want}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 300, "grid barely ran: {checked} comparisons");
    }

    /// The direction the ladder exists for, as a named case: a hot rung holding
    /// a much better state must be accepted with certainty.
    ///
    /// gh#550 shipped the negation, which gave log α = −50 here — a rejection
    /// probability of 1 − 2e−22 for a trade that raises the joint density by a
    /// factor of e^50.
    #[test]
    fn accepts_when_the_hotter_rung_holds_the_better_state() {
        // Cold rung β=1.0 stuck at ll=-200; hot rung β=0.5 found ll=-100.
        let log_alpha = swap_log_alpha(1.0, 0.5, -200.0, -100.0);
        assert!(log_alpha > 0.0,
            "a hot rung holding the better state must be accepted; got {log_alpha}");
        assert!((log_alpha - 50.0).abs() < 1e-9, "expected +50, got {log_alpha}");

        // And the converse: when the COLD rung is already better, the trade
        // would move the worse state into the reported chain, so it must not be
        // automatic. This is the arm the inverted code accepted with certainty.
        let reverse = swap_log_alpha(1.0, 0.5, -100.0, -200.0);
        assert!(reverse < 0.0,
            "swapping a worse state into the cold rung must be penalised; got {reverse}");
    }

    /// Antisymmetry in the rung roles: proposing the same exchange from either
    /// side is the same move, so the ratio must simply negate. Guards against a
    /// future "fix" that special-cases one ordering.
    #[test]
    fn negates_when_the_two_rungs_are_swapped() {
        for (beta_i, beta_j, ll_i, ll_j) in
            [(1.0, 0.5, -200.0, -100.0), (0.9, 0.1, -1.0, -50.0), (1.0, 0.99, 0.0, -0.5)]
        {
            let a = swap_log_alpha(beta_i, beta_j, ll_i, ll_j);
            let b = swap_log_alpha(beta_j, beta_i, ll_j, ll_i);
            assert!((a - b).abs() < 1e-12, "not symmetric under role exchange: {a} vs {b}");
        }
    }

    /// Equal temperatures or equal likelihoods leave nothing to gain, so the
    /// ratio is exactly 0 (accept — the states are interchangeable).
    #[test]
    fn is_zero_when_there_is_nothing_to_gain() {
        assert_eq!(swap_log_alpha(0.7, 0.7, -10.0, -900.0), 0.0);
        assert_eq!(swap_log_alpha(1.0, 0.3, -42.0, -42.0), 0.0);
    }
}

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

/// Whether the θ|X move runs NUTS (vs the MH-within-Gibbs fallback): NUTS is
/// requested AND the compiler emitted gradient expressions. ONE predicate for
/// the sampler and for every diagnostic that keys a healthy band on the
/// kernel (gh#631) — the two must never disagree about which kernel ran.
pub fn nuts_active(use_nuts: bool, model: &CompiledModel) -> bool {
    use_nuts && model.model.transitions.iter().any(|t| !t.rate_grad.is_empty())
}

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
///
/// # Errors
///
/// `SimError::NonFiniteChainStart` (gh#607) when the chain's starting
/// (θ₀, X₀) has zero posterior density AND is still at zero density after the
/// first Gibbs sweep — a chain that cannot move, refused instead of sampled.
/// Callers running several chains should treat it as skip-this-chain, not as a
/// failed fit; `cli/src/fit/pgas.rs` is the reference handling. Everything else
/// is structural or a `Validation` refusal from the preflights above.
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

    // Whether this call begins a chain (vs continues one). The chain-start
    // refusal below applies only to a fresh start: a resumed θ already passed
    // the check once, and re-running it on the restored state would refuse a
    // chain that has been sampling happily. Same reasoning as PMMH's init-eval
    // guard, which is likewise skipped on resume (`cli/src/fit/pmmh.rs`).
    let is_fresh_start = resume_from.is_none();

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
        //
        // This block only EXPLAINS; it does not decide. A non-finite total
        // here reappears in `current_ll` below (same trajectory, same params),
        // where the gh#607 chain-start refusal turns it into a skipped chain.
        let sanity = complete_data_loglik(
            model, &trajectory, &current_params, observations,
            config.dt, obs_model, &obs_at_substep,
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
            eprintln!("  components: transition={:.1}, observation={:.1}, \
                       initial_state={:.1}",
                sanity.transition, sanity.observation, sanity.initial_state);
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

    // Initial complete-data log-likelihood (includes the initial-state density
    // of any DECLARED `init {}` law)
    //
    // gh#80: same split-by-component diagnostic as the sanity check above —
    // distinguish a step_one/density mismatch (transition term) from a
    // data-vs-model incompatibility (observation term).
    let current_components = complete_data_loglik(
        model, &trajectory, &current_params, observations,
        config.dt, obs_model, &obs_at_substep,
    )?;
    let current_ll = current_components.total;
    eprintln!("  initial complete-data ll: {:.1}", current_ll);
    if !current_ll.is_finite() {
        let trans_inf = !current_components.transition.is_finite();
        let obs_inf   = !current_components.observation.is_finite();
        let init_inf  = !current_components.initial_state.is_finite();
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
        if init_inf {
            eprintln!("  WARNING: initial *initial-state* log-density is \
                       non-finite (initial_state_ll = {}) — the reference \
                       trajectory's x0 has zero probability under the declared \
                       `init {{ }}` law at these starting parameters.",
                       current_components.initial_state);
        }
        eprintln!("  components: transition={:.1}, observation={:.1}, \
                   initial_state={:.1}",
            current_components.transition,
            current_components.observation,
            current_components.initial_state);
        eprintln!("  Model has {} transitions, {} source groups",
            model.model.transitions.len(),
            model.source_groups.len());
    }

    // gh#607. A start with zero posterior density is put ON PROBATION here and
    // refused at the end of the first sweep (`start_at_zero_density` below) if
    // it is still there. Seeding every rung with `−∞` and sampling 40 000
    // sweeps anyway is what this replaces.
    //
    // The quantity is the one the chain is actually sampled on: the
    // complete-data log-likelihood plus the log prior at θ₀ — the same
    // definition as the `log_posterior` column the trace writer emits
    // (`cli/src/fit/pgas.rs`), so a refusal names a number the user can find.
    // The transform Jacobian is deliberately excluded, matching that column;
    // `to_transformed` clamps into the declared support, so a start inside its
    // bounds cannot make the Jacobian the non-finite term.
    //
    // WHY PROBATION AND NOT AN IMMEDIATE REFUSAL. `−∞` at (θ₀, X₀) is usually
    // the OBSERVATION term, and that term is a property of the pair, not of θ₀
    // alone: X₀ is one stochastic reference draw, and the X|θ,y move can — and
    // routinely does — replace it with a trajectory that explains the data at
    // the SAME θ₀. Measured on three of this repository's own PGAS fixtures
    // (`fit_predict_e2e`, `contrasts_e2e`, `pgas_resume`): every one starts a
    // chain at `−∞` and every one is finite by the first recorded sweep. An
    // immediate refusal would have killed all three.
    //
    // WHY ONE SWEEP IS THE LINE. After a complete Gibbs sweep the chain has had
    // every move the sampler offers. If it is still at `−∞`, θ can no longer
    // move: the θ|X step scores each proposal against the current X, so with
    // both current and proposed at `−∞` the MH ratio is NaN and rejects, and
    // NUTS is worse — `log p = −∞` gives `h0 = +∞`, so every doubling trips
    // `(h_new − h0).abs() > delta_max` and the tree stops at depth 0
    // (`nuts.rs`). Each later sweep is then an independent retry of the SAME
    // failed X-move at the SAME θ₀. That retry is not impossible, only
    // vanishingly unlikely in practice: the production run that motivated this
    // measured 40 000 consecutive failures, acceptance 0.000 and `n_divergent`
    // 1.000 throughout, with ONE distinct parameter vector across 7 600
    // retained draws.
    //
    // Contrast with PMMH/ODE-MH, which keep their warn-and-continue: their
    // likelihood is MARGINAL in X (the filter re-integrates it at every
    // proposal), so a different θ genuinely re-rolls the observation term and
    // the `+∞` escape is reachable (gh#334, gh#471).
    //
    // Deliberately NOT auto-resampled. A start drawn from the prior that lands
    // on an impossible region is information about the prior; silently
    // redrawing it would hide that (gh#419 rejected auto-quarantine as policy).
    //
    // `Some(..)` carries the START's component breakdown — which term was
    // non-finite is the diagnosis (`observation` = a bad start; `transition` =
    // a step_one/density disagreement, i.e. a bug, gh#80) — so the refusal can
    // report it after the probation sweep.
    let start_at_zero_density: Option<(f64, f64, f64, f64, f64)> = if is_fresh_start {
        let initial_log_prior: f64 = if2_params.iter().zip(priors.iter())
            .map(|(spec, prior)| {
                let theta = current_params[spec.index];
                prior.log_density(theta, spec.to_transformed(theta))
            })
            .sum();
        let initial_log_posterior = current_ll + initial_log_prior;
        if initial_log_posterior.is_finite() {
            None
        } else {
            Some((
                initial_log_posterior,
                current_components.transition,
                current_components.observation,
                current_components.initial_state,
                initial_log_prior,
            ))
        }
    } else {
        None
    };

    // Check if gradients are available (compiler emitted rate_grad)
    let has_gradients = nuts_active(config.use_nuts, model);
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

    // gh#180 P5 — the `DerivEntry::Unsupported` preflight (proposal §4.4).
    //
    // The obs/σ² gradients now ride the compiler-emitted `*_grad` / `sigma_sq_grad`
    // maps, each entry classified `Grad | Unsupported{code}` by the autodiff pass
    // (the projection is inlined before differentiation, so a param reaching an
    // observation THROUGH a parametric `DerivedExpr` projection is already
    // classified in the argument's grad map — a tier-1 case, e.g. `qgam·prevalence`,
    // carries a `Grad` and is ADMITTED here, retiring the old C1 fence).
    //
    // The invariant: a NUTS fit runs only if every estimated parameter reaching an
    // observation — through a projection or any likelihood argument, after
    // projection inlining — is covered by a `Grad`. An `Unsupported{code}` keyed by
    // an estimated param in ANY obs `*_grad` or `sigma_sq_grad` map is refused here,
    // at the `run_pgas` boundary (protecting every caller, not just the CLI
    // `if use_nuts` site), with the human message derived from `code`. This makes
    // `eval_emitted_grad`'s `Unsupported` branch unreachable-by-construction.
    //
    // NUTS is the only θ|X step that consumes the gradient (`has_gradients` gates
    // `complete_data_loglik_grad`, l.~2177); an MH-within-Gibbs (`--no-nuts`) sweep
    // never differentiates, so an Unsupported obs grad cannot bias it — gate on
    // `use_nuts`, matching the CLI `coeff_guard` site.
    if config.use_nuts {
        use std::collections::HashSet;

        let estimated: HashSet<&str> = if2_params.iter().map(|s| s.name.as_str()).collect();

        // The COMMON gradient-coverage scan, shared with the ODE-NUTS gate (§1h):
        // every estimated parameter whose gradient the compiler could not emit for
        // a rate, obs argument, or σ² term, or that reaches a Binomial/BetaBinomial
        // `n`. Lifting it into `gradient_capability` retires the old CLI-only
        // `coeff_guard` for the rate domain, protects every direct `run_pgas` caller
        // (tests, API), and keeps this and the ODE gate from drifting.
        let refused = crate::inference::gradient_capability::scan_unsupported_gradients(
            &model.model,
            &estimated,
        );

        if !refused.is_empty() {
            let details: Vec<String> = refused
                .iter()
                .map(|(p, code)| format!("`{}` {}", p, code.reason_message()))
                .collect();
            return Err(SimError::Validation(format!(
                "PGAS+NUTS cannot estimate parameter(s) whose gradient the compiler \
                 could not emit for a rate, observation, or overdispersion term — NUTS \
                 would sample against an incomplete (silently biased) gradient. Refused: \
                 {}. Estimate these with a gradient-free method (IF2 or PMMH), run PGAS \
                 with --no-nuts, or fix them (`[fixed.X]` in fit.toml).",
                details.join("; ")
            )));
        }
    }

    // Pre-resolve rate_grad indices once for the entire run (avoids O(n_params)
    // string scans per gradient term per substep in the NUTS hot path).
    // model_to_estimated[model_param_idx] = estimated_param_idx, or None if fixed.
    let rate_grads_for_run: Vec<crate::resolved_expr::ResolvedGradMap> = {
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
    // gh#20 (gamma-density gradient) and gh#76/gh#180 (obs-density gradient) to
    // look up the compiler-emitted `∂σ²/∂θ` and `∂arg/∂θ` maps (via the shared
    // `eval_emitted_grad` seam).
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
                    csmc_seed, &obs_at_substep, firing,
                )?;
                rungs[rung].trajectory = new_traj;
                rungs[rung].ll = complete_data_loglik(
                    model, &rungs[rung].trajectory, &rungs[rung].params, observations,
                    config.dt, obs_model, &obs_at_substep,
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
        let mut cold_initial_state_ll = 0.0_f64;
        let mut cold_nuts = NutsSweepDiag::default();

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
                        config.dt, obs_model,
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
                    // Per-sweep cold-chain NUTS telemetry for the trace (gh#294).
                    // `nuts_config.step_size` is the step actually used this
                    // sweep (the dual-averaging update below mutates the rung's
                    // step_size only afterwards).
                    cold_nuts = NutsSweepDiag {
                        tree_depth: result.tree_depth,
                        n_leapfrog: result.n_leapfrog,
                        step_size: nuts_config.step_size,
                        accept_stat: result.mean_accept_prob,
                        n_divergent: usize::from(result.divergent),
                        energy: result.energy,
                    };
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

                    // gh#82: a failed evaluation at a PROPOSED θ is a rejected
                    // proposal (−∞ ⇒ non-finite log α ⇒ the guard below
                    // rejects), not a dead chain. Only a structural failure
                    // still propagates — see `theta_proposal_score`.
                    let proposed_ll = theta_proposal_score(complete_data_loglik(
                        model, &rungs[rung].trajectory, &proposed_params, observations,
                        config.dt, obs_model, &obs_at_substep,
                    ))?;

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

                    // gh#471: `mh_accept`, not an inline `is_finite()` guard.
                    // gh#334 removed that guard from PMMH and never reached
                    // here — `9f99405d` touched only `pmmh.rs`. The guard
                    // uniquely mis-rejects `log_alpha = +∞`, which is the
                    // acceptance ratio of the one move that matters when a
                    // chain sits at `ll = −∞`: a proposal to a finite θ, whose
                    // true Metropolis probability is 1. Rejecting it freezes
                    // the parameter block for the rest of the run.
                    //
                    // That state is still reachable mid-run. The chain-start
                    // refusal above (gh#607) rules out only sweep 0: a rung
                    // that starts finite can still walk into a `−∞` region
                    // later — a CSMC refresh can hand it a trajectory whose
                    // observation term is `−∞` at the current θ — and from
                    // there the `+∞` escape is the only move out.
                    if crate::inference::mh_accept(log_alpha, rng.uniform().ln()) {
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
                trajectory_renewal: 0.0, renewal_by_bin: [f64::NAN; RENEWAL_BINS],
                n_degenerate: 0, n_resampled: 0, n_as_skipped_no_resample: 0, n_substeps: 0,
                n_as_proposed: 0, n_as_accepted: 0, n_as_refused_inadmissible: 0,
            };
            for csmc_rep in 0..config.csmc_sweeps_per_nuts {
                let csmc_seed = seed ^ ((sweep as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
                    ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142)
                    ^ (csmc_rep as u64).wrapping_mul(0xa2ce44bbfe0cf6d5);
                let (new_trajectory, diag) = csmc_as(
                    model, &rungs[rung].params, observations, &rungs[rung].trajectory,
                    config.n_particles, config.dt, obs_model,
                    csmc_seed, &obs_at_substep, firing,
                )?;
                rungs[rung].trajectory = new_trajectory;
                csmc_diag = diag;
            }

            // Recompute complete-data LL at β=1 (untempered, for swap proposals)
            let ll_components = complete_data_loglik(
                model, &rungs[rung].trajectory, &rungs[rung].params, observations,
                config.dt, obs_model, &obs_at_substep,
            )?;
            rungs[rung].ll = ll_components.total;

            rung_csmc_diag.push(csmc_diag);

            // Store components for cold rung output
            if rung == 0 {
                cold_transition_ll = ll_components.transition;
                cold_obs_ll = ll_components.observation;
                cold_initial_state_ll = ll_components.initial_state;
            }
        } // end rung loop

        // gh#607. END OF PROBATION. The chain started at zero posterior density
        // and has now had one complete Gibbs sweep — a θ|X move and the X|θ,y
        // move that is the only one able to rescue such a start. If NO rung
        // reached finite density, refuse the chain here: the reasoning and the
        // measured evidence are at `start_at_zero_density`'s definition.
        //
        // Placed before the swap, the callback, and the draw recorder, so a
        // refused chain leaves no trace row, no retained draw, and no saved
        // trajectory — nothing for a downstream number to pick up.
        //
        // ANY rung finite is enough, not just the cold one: a swap proposal
        // from a `−∞` cold rung to a finite hot rung has `log α =
        // (β_i − β_j)(ℓ_j − (−∞)) = +∞` and is accepted with certainty
        // (`swap_log_alpha`), so the ladder can still pull the cold rung out.
        // With the default single rung this reduces to "the cold rung".
        if let Some((log_posterior, transition, observation, ivp, log_prior)) =
            start_at_zero_density
        {
            if sweep == start_sweep {
                let any_rung_finite = rungs.iter().any(|rung| {
                    let rung_log_prior: f64 = if2_params.iter().zip(priors.iter())
                        .map(|(spec, prior)| {
                            let theta = rung.params[spec.index];
                            prior.log_density(theta, spec.to_transformed(theta))
                        })
                        .sum();
                    (rung.ll + rung_log_prior).is_finite()
                });
                if !any_rung_finite {
                    return Err(SimError::NonFiniteChainStart {
                        log_posterior, transition, observation, ivp, log_prior,
                    });
                }
                eprintln!("  chain recovered from a non-finite start on its first \
                           trajectory update (complete-data ll: {:.1})", rungs[0].ll);
            }
        }

        // ── Replica exchange: swap adjacent rungs ──
        if n_rungs > 1 {
            // Even-odd scheme: alternate starting parity each sweep
            let pair_start = sweep % 2;
            let mut i = pair_start;
            while i + 1 < n_rungs {
                let j = i + 1;
                swap_proposed[i] += 1;

                // gh#550. Derivation in `swap_log_alpha`; the short version is
                // that the sign is the sign of (ℓ_j − ℓ_i), so a swap is
                // accepted when the HOTTER rung holds the better state — which
                // is the whole purpose of the ladder.
                let log_alpha = swap_log_alpha(
                    betas[i], betas[j], rungs[i].ll, rungs[j].ll);

                // Not routed through `mh_accept`, deliberately — but for a
                // narrower reason than an earlier version of this comment
                // claimed, and the difference is worth stating because that
                // claim is what let gh#550 sit here for four months.
                //
                // The two forms make the same DECISION, given that
                // `StatefulRng::uniform` returns `u ∈ [0, 1)`: `u_ln` is then
                // strictly negative, so `log_alpha == 0.0` (common here —
                // it is what equal βs or equal lls produce) accepts under
                // both. That precondition is load-bearing. If `uniform` ever
                // moved to an `OpenClosed01` convention, `u_ln` could reach 0
                // and the two would silently diverge at exactly that value.
                // `swap_forms_agree_given_half_open_uniform` pins it.
                //
                // What differs is RNG CONSUMPTION: `||` short-circuits on a
                // certain accept and draws no uniform, where
                // `mh_accept(…, rng.uniform().ln())` always draws one.
                // Converting would be distribution-neutral and
                // stream-breaking for tempered NUTS runs, which reach this
                // site but not the θ-block accept above. That is the whole of
                // the reason to leave it — NOT any claim that the site is
                // otherwise beyond scrutiny. gh#550 was a sign error sitting
                // two lines above this comment while it asserted the site was
                // fine.
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
            sweep,
            params: rungs[0].params.clone(),
            log_complete_data_ll: rungs[0].ll,
            accepted: rung_accepted[0].clone(),
            csmc_diag: rung_csmc_diag[0].clone(),
            proposal_sds: cold_proposal_sd,
            transition_ll: cold_transition_ll,
            obs_ll: cold_obs_ll,
            initial_state_ll: cold_initial_state_ll,
            nuts: cold_nuts,
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

    /// gh#264: an ancestor-sampling join leaves the raw `counts_after` sequence
    /// discontinuous (`S` jumps up — backflow). `coherent_counts_after` must
    /// chain the offset-free net deltas into a single continuous, monotone,
    /// flow-reconciling path.
    #[test]
    fn coherent_counts_after_removes_as_join_backflow() {
        // 2 compartments [S, E]; one transition S->E (flow index 0).
        let mk = |cb: [i64; 2], ca: [i64; 2], inf: u64| SubstepRecord {
            counts_before: cb.to_vec(),
            counts_after: ca.to_vec(),
            flows: vec![inf],
            gammas: vec![],
            t0: 0.0,
            dt_substep: 1.0,
        };
        // substeps 0,1 coherent; substep 2 is an AS join whose `counts_before`
        // is offset HIGH (the reference's own state from a different prefix),
        // so the RAW counts_after jumps S upward.
        let traj = PGASTrajectory {
            initial_counts: vec![100, 0],
            substeps: vec![
                mk([100, 0], [90, 10], 10),
                mk([90, 10], [85, 15], 5),
                mk([200, 0], [192, 8], 8), // join: counts_before != prev counts_after
            ],
        };

        // The raw counts_after exhibits the bug (S non-monotone: 90, 85, 192).
        let raw_s: Vec<i64> = traj.substeps.iter().map(|r| r.counts_after[0]).collect();
        assert!(raw_s.windows(2).any(|w| w[1] > w[0]),
            "raw counts_after must show the backflow, got {raw_s:?}");

        let coh = traj.coherent_counts_after().unwrap();
        let coh_s: Vec<i64> = coh.iter().map(|st| st[0]).collect();
        assert_eq!(coh_s, vec![90, 85, 77]);
        // Monotone non-increasing (S only leaves via infection).
        assert!(coh_s.windows(2).all(|w| w[1] <= w[0]),
            "coherent S must be monotone non-increasing, got {coh_s:?}");
        // Flow reconciliation: S[s-1] - S[s] == infection flow at s.
        let mut prev = traj.substeps[0].counts_before[0];
        for (s, st) in coh.iter().enumerate() {
            assert_eq!(prev - st[0], traj.substeps[s].flows[0] as i64,
                "substep {s}: S drop must equal infection flow");
            prev = st[0];
        }
        // A join-free trajectory is reproduced exactly.
        let coherent_only = PGASTrajectory {
            initial_counts: vec![100, 0],
            substeps: vec![mk([100, 0], [90, 10], 10), mk([90, 10], [85, 15], 5)],
        };
        let c = coherent_only.coherent_counts_after().unwrap();
        assert_eq!(c, vec![vec![90, 10], vec![85, 15]]);
    }

    /// The `SubstepRecord → Snapshot` adapter must (a) use the coherent
    /// net-delta path for counts, (b) stamp each snapshot at `t0 + dt_substep`,
    /// (c) carry the per-substep flows, and (d) compute `inc_<stream>` as the
    /// `FlowSum` projection of THOSE FLOWS — never a finite-difference of
    /// counts. This last point is the gh#48 / #264 correctness requirement.
    #[test]
    fn to_trajectory_projects_incidence_from_flows_not_count_diff() {
        // 2 compartments [S, I]; one transition S->I (flow index 0) plus a
        // second transition I->R (flow index 1) so the incidence stream sums a
        // SUBSET of flows (index 0 only) — proving it reads the projection's
        // flow set, not a count delta.
        let mk = |cb: [i64; 2], ca: [i64; 2], inf: u64, rec: u64, t0: f64, dt: f64| {
            SubstepRecord {
                counts_before: cb.to_vec(),
                counts_after: ca.to_vec(),
                flows: vec![inf, rec],
                gammas: vec![],
                t0,
                dt_substep: dt,
            }
        };
        let traj = PGASTrajectory {
            initial_counts: vec![100, 0],
            // off-grid times: t0 not a clean s*dt, to catch a recompute-from-s bug.
            substeps: vec![
                mk([100, 0], [90, 10], 10, 0, 0.0, 1.0),
                mk([90, 10], [85, 12], 5, 3, 1.0, 0.5), // dt_substep 0.5 → end 1.5
            ],
        };

        // Incidence stream "cases" sums flow index 0 (infection) only.
        let inc_streams = vec![("cases".to_string(), vec![0usize])];
        let (out, incidence) = traj.to_trajectory(&inc_streams).unwrap();

        // Row 0 is the initial-condition row at t_start (gh#270): the path's
        // anchor state with zeroed flows. The two substep rows follow.
        let coh = traj.coherent_counts_after().unwrap();
        assert_eq!(out.snapshots.len(), 3);
        // (a) initial row = counts_before[0] at t0, zero flows / zero incidence.
        assert_eq!(out.snapshots[0].t, 0.0);
        assert_eq!(out.snapshots[0].int_state.counts, vec![100, 0]);
        assert_eq!(out.snapshots[0].flows.as_int(), &[0, 0]);
        // (a') substep rows' counts == coherent_counts_after.
        for (s, snap) in out.snapshots[1..].iter().enumerate() {
            assert_eq!(snap.int_state.counts, coh[s], "snapshot {s} counts");
        }
        // (b) substep times stamped at t0 + dt_substep.
        assert_eq!(out.snapshots[1].t, 1.0);
        assert_eq!(out.snapshots[2].t, 1.5);
        // (c) flows ride through.
        assert_eq!(out.snapshots[1].flows.as_int(), &[10, 0]);
        assert_eq!(out.snapshots[2].flows.as_int(), &[5, 3]);
        // (d) inc_cases == Σ_{i∈{0}} substep.flows[i] = the infection flow per
        // substep — NOT a count diff (which would be coh S drop: also 10 then 5
        // here, so to disambiguate we use the SUBSET: recovery flow is excluded).
        // The initial row contributes a leading zero, keeping incidence aligned
        // 1:1 with the snapshots.
        assert_eq!(incidence.len(), 3);
        assert_eq!(incidence[0], vec![0.0]);
        assert_eq!(incidence[1], vec![10.0]);
        assert_eq!(incidence[2], vec![5.0]);
        // The projection summed ONLY flow 0; had it summed all flows it would be
        // 10 then 8 (5+3). Confirm it didn't.
        assert_ne!(incidence[2], vec![8.0]);
    }

    /// No incidence streams ⇒ empty incidence sidecar (writer emits no `inc_*`).
    #[test]
    fn to_trajectory_empty_incidence_when_no_streams() {
        let traj = PGASTrajectory {
            initial_counts: vec![100],
            substeps: vec![SubstepRecord {
                counts_before: vec![100],
                counts_after: vec![90],
                flows: vec![10],
                gammas: vec![],
                t0: 0.0,
                dt_substep: 1.0,
            }],
        };
        let (out, incidence) = traj.to_trajectory(&[]).unwrap();
        // One substep ⇒ initial-condition row (t_start) + the substep's end row.
        assert_eq!(out.snapshots.len(), 2);
        assert_eq!(out.snapshots[0].int_state.counts, vec![100]); // S₀
        assert_eq!(out.snapshots[1].int_state.counts, vec![90]);
        assert!(incidence.is_empty());
    }

    // gh#270 RED reproduction: the saved path must include its t_start initial
    // row so the seed stratum reconciles `Σ flow_infection == S₀ − S_final`.
    #[test]
    fn gh270_seed_stratum_flow_reconciles_with_s_depletion() {
        // SEIRD-like seed: S starts at N0-I0; first substep has infection flows
        // (I0 > 0 ⇒ flow_infection[0] > 0). S leaves ONLY via infection.
        let mk = |cb: [i64; 2], ca: [i64; 2], inf: u64| SubstepRecord {
            counts_before: cb.to_vec(),
            counts_after: ca.to_vec(),
            flows: vec![inf],
            gammas: vec![],
            t0: 0.0,
            dt_substep: 1.0,
        };
        let traj = PGASTrajectory {
            initial_counts: vec![1000, 13],
            substeps: vec![
                mk([1000, 13], [994, 19], 6), // first substep: 6 infections
                mk([994, 19], [990, 22], 4),
                mk([990, 22], [988, 23], 2),
            ],
        };
        let (out, _) = traj.to_trajectory(&[]).unwrap();
        let s: Vec<i64> = out.snapshots.iter().map(|sn| sn.int_state.counts[0]).collect();
        let total_flow: i64 = traj.substeps.iter().map(|r| r.flows[0] as i64).sum();
        // The audit identity from gh#270: Σ flow_infection == S₀ − S_final.
        assert_eq!(
            total_flow,
            s[0] - s[s.len() - 1],
            "seed stratum: Σ flow_infection must equal S₀ − S_final (gh#270)"
        );
        // And the path must begin at the true initial S (the t_start row).
        assert_eq!(s[0], 1000, "first written S must be S₀ = initial_counts (t_start row)");
    }

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
mod theta_proposal_score_tests {
    //! gh#82. The θ-proposal boundary: which failures reject a proposal and
    //! which ones tear the chain down.
    //!
    //! These sit here rather than in the integration test because a *structural*
    //! error is unreachable from a built model's rate evaluation — every
    //! name-resolution failure surfaces at `CompiledModel::new`
    //! (`resolved_expr::resolve_expr`: "All name-not-found errors surface here
    //! at model construction time"), so no `.camdl` fixture can drive the
    //! propagate branch through `run_pgas`. Synthesizing the `SimError` is the
    //! only way to exercise it, and leaving it unexercised is how a
    //! "reject everything" regression would land unnoticed.
    use super::*;
    use crate::error::{CollapseKind, NegativeCountCause};

    fn components(total: f64) -> LogLikComponents {
        LogLikComponents { total, transition: total, observation: 0.0, initial_state: 0.0 }
    }

    #[test]
    fn a_successful_evaluation_passes_its_total_through() {
        let score = theta_proposal_score(Ok(components(-12.5))).expect("Ok must not error");
        assert_eq!(score, -12.5);
        // Including the ordinary "this θ is ruled out" outcome, which
        // `complete_data_loglik` already reports as Ok(−∞).
        let score = theta_proposal_score(Ok(components(f64::NEG_INFINITY))).unwrap();
        assert_eq!(score, f64::NEG_INFINITY);
    }

    /// The gh#82 fix: every θ-dependent failure becomes a rejected proposal.
    /// `NonFiniteParameter` is the variant the issue names; `NegativePropensity`
    /// is the one that distinguishes `is_structural()` from the narrower
    /// `is_per_particle_recoverable()` and must reject too, so PGAS and PMMH
    /// agree about the same θ.
    #[test]
    fn theta_dependent_failures_are_rejected_as_neg_infinity() {
        let rejected: Vec<SimError> = vec![
            SimError::NonFiniteParameter { name: "tau".into(), value: f64::NEG_INFINITY, t: -101.0 },
            SimError::NumericalCollapse { kind: CollapseKind::DivByZero, t: 3.0 },
            SimError::NumericalCollapse { kind: CollapseKind::SqrtNegative, t: 3.0 },
            SimError::NegativeCount {
                compartment: "S".into(), attempted_value: -2, t: 4.0,
                cause: NegativeCountCause::BinomialOvershoot,
            },
            SimError::TableLookup("table 'k': index 5 out of bounds [0, 2)".into()),
            SimError::NegativePropensity { transition: "foi".into(), value: -0.5, t: 2.0 },
            SimError::DivisionByZero(7.0),
        ];
        for err in rejected {
            let shown = err.to_string();
            let score = theta_proposal_score(Err(err))
                .unwrap_or_else(|e| panic!("{shown} must reject the proposal, not propagate: {e}"));
            assert_eq!(
                score, f64::NEG_INFINITY,
                "a rejected proposal scores −∞ so log α is non-finite: {shown}",
            );
        }
    }

    /// Negative control. A structural failure fires for EVERY θ — rejecting it
    /// would leave the chain sampling a meaningless posterior and exiting 0.
    /// It must still propagate, verbatim.
    #[test]
    fn structural_failures_still_tear_the_chain_down() {
        let structural: Vec<SimError> = vec![
            SimError::Validation("model cannot run".into()),
            SimError::UnknownParameter("ghost".into()),
            SimError::UnknownCompartment("ghost".into()),
            SimError::ConfigMismatch { expected: "chain_binomial", got: "ode" },
            SimError::NegativeCount {
                compartment: "S".into(), attempted_value: -2, t: 4.0,
                cause: NegativeCountCause::InterventionAddNegative,
            },
        ];
        for err in structural {
            let shown = err.to_string();
            assert!(err.is_structural(), "control must exercise the structural branch: {shown}");
            let out = theta_proposal_score(Err(err));
            assert!(
                out.is_err(),
                "a structural error must propagate out of run_pgas, got Ok({:?}) for {shown}",
                out.ok(),
            );
        }
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
            lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
        }
    }

    fn identity_param(lo: f64, hi: f64) -> EstimatedParam {
        EstimatedParam {
            name: "p".into(), index: 0, initial: 0.0, rw_sd: 0.1,
            transform: Transform::None,
            lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
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
        assert_grad_matches_fd(&Prior::Fixed(Density::TransformedNormal { mean: 1.0, sd: 0.5 }),
            &p, &[-1.0, 0.0, 0.7, 1.5]);
    }

    #[test]
    fn natural_scale_priors_grad_matches_fd() {
        // These arms already follow the natural-density convention; lock them.
        let lp = log_param(1e-4, 1e2);
        assert_grad_matches_fd(&Prior::Fixed(Density::HalfNormal { sigma: 1.0 }), &lp, &[-1.0, 0.0, 1.0]);
        assert_grad_matches_fd(&Prior::Fixed(Density::Gamma { shape: 2.0, rate: 1.5 }), &lp, &[-1.0, 0.0, 1.0]);
        assert_grad_matches_fd(&Prior::Fixed(Density::Exponential { rate: 0.7 }), &lp, &[-1.0, 0.0, 1.0]);
        let ip = identity_param(-5.0, 5.0);
        assert_grad_matches_fd(&Prior::Fixed(Density::Normal { mean: 0.3, sd: 0.8 }), &ip, &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn log_uniform_grad_matches_fd() {
        // On the Log transform the z-scale density is flat → gradient 0.
        let p = log_param(1e-5, 1e-2);
        let zs = [(1e-4_f64).ln(), (1e-3_f64).ln(), (5e-3_f64).ln()];
        assert_grad_matches_fd(&Prior::Fixed(Density::LogUniform { lower: 1e-5, upper: 1e-2 }), &p, &zs);
        // And it really is flat (gradient ≈ 0 everywhere interior).
        for &z in &zs {
            assert!(target_grad(&Prior::Fixed(Density::LogUniform { lower: 1e-5, upper: 1e-2 }), &p, z).abs() < 1e-9);
        }
    }

    #[test]
    fn truncated_normal_grad_matches_fd() {
        // Identity transform, bounds = truncation support.
        let ip = identity_param(0.3, 1.0);
        assert_grad_matches_fd(
            &Prior::Fixed(Density::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 }),
            &ip, &[0.4, 0.7, 0.95]);
        // Logit transform onto [0.3, 1.0] — bounds equal truncation support.
        let lp = EstimatedParam {
            name: "p".into(), index: 0, initial: 0.7, rw_sd: 0.1,
            transform: Transform::Logit { lo: 0.3, hi: 1.0 },
            lower: 0.3, upper: 1.0, rw_sd_auto: false, perturb_only_at_t0: false,
        };
        assert_grad_matches_fd(
            &Prior::Fixed(Density::TruncatedNormal { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 }),
            &lp, &[-1.0, 0.0, 1.0]);
    }
}
