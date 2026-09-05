//! State-space Particle Gibbs with backward simulation (`csmc_bs`) —
//! the experimental trajectory kernel of the 2026-09-02 spike note.
//!
//! One sweep draws `Z' ~ p(Z | θ, Y)` over the STATE path `Z = (X, A)` —
//! compartment counts plus open interval-stream accumulators — with the
//! per-substep innovations (flows) marginalized by the oracle-tested
//! [`state_transition`] density, then reconstructs a complete flow record
//! for the θ-move via [`sample_edge_flows`] (the partially-collapsed Gibbs
//! ordering: the caller must run the θ-move on THIS sweep's reconstruction,
//! never on innovations from an earlier trajectory).
//!
//! Structure per sweep:
//!
//! 1. **Forward conditional filter.** Free particles reuse the ordinary
//!    machinery — `step_one` propagation, per-particle RNG streams,
//!    conditional resampling, observation scoring. The reference slot is
//!    **pinned to the reference state path**: `Z_s[j_ref] = Z*_s` at every
//!    substep, accumulators included — it is never produced by replaying an
//!    innovation record from an ancestor state (the representation change;
//!    see the spike note's reference-slot contract and its test).
//! 2. **Backward stitch.** Draw the final state from the final weights, then
//!    walk s = T−1 … 0 choosing predecessor `j` with weight
//!    `w_{s-1}^j · p(Z_s^chosen | Z_{s-1}^j)` over all N candidates (naive
//!    all-N per the guardrails; the kernel instruments candidate feasibility
//!    and lattice sizes so the spike's proxy numbers are superseded by
//!    in-cloud measurements).
//! 3. **Reconstruction.** Per stitched edge, draw flows from the
//!    lattice-restricted conditional — the same enumeration the backward
//!    density computed — yielding a `PGASTrajectory` whose complete-data
//!    density downstream code can score exactly as today (gammas empty:
//!    the analysis refuses overdispersed models in the prototype class).
//!
//! Accumulator convention (spike note): the `A` component of `Z_s` is the
//! post-fold, PRE-reset per-stream bin at the end of substep `s`; the
//! transition rule is `A_{s+1} = (due_reset(s) ? 0 : A_s) + H·F_{s+1}`, so
//! observations at substep `s` score off `Z_s` and the Markov property holds
//! across bin boundaries.

use rayon::prelude::*;

/// Which latent-trajectory representation the PGAS stage conditions on.
/// `Innovation` is today's kernel family (the retained object carries flows
/// and noise; ancestor sampling splices with the exact suffix correction).
/// `State` conditions on the Markov state path `Z = (X, A)` with innovations
/// marginalized by the state-transition density — the experimental
/// representation of the 2026-09-02 spike note. Selecting `state` changes the
/// sampled draws, so it is identity-bearing: it serializes into the stage
/// payload (the default does not, keeping pre-field payloads byte-identical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryRepresentation {
    #[default]
    Innovation,
    State,
}

/// Which trajectory update runs on that representation. `AncestorSampling`
/// is the LJS move (innovation representation only); `Backward` is
/// backward simulation over stored particle states (state representation
/// only). Unsupported combinations are refused loudly at config validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryKernel {
    #[default]
    AncestorSampling,
    Backward,
}

use crate::chain_binomial::{step_one, StepScratch};
use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::inference::multi_stream_obs::MultiStreamObsModel;
use crate::inference::pgas::{
    fill_producer_batch, sample_categorical_log, EffectFiring, ObsAtSubstep, PGASTrajectory,
    SubstepRecord, RENEWAL_BINS,
};
use crate::inference::resampling::conditional_multinomial_resample;
use crate::inference::state_transition::{
    log_state_transition_density, sample_edge_flows, StateTransitionAnalysis,
};
use crate::inference::types::{init_particle_rngs, RESAMPLE_RNG_STREAM};
use crate::rng::StatefulRng;

/// Per-sweep diagnostics for the backward kernel. Mirrors the innovation
/// kernel's renewal surface (same bins, same slot-identity convention) and
/// carries the in-kernel candidate instrument the spike note requires before
/// the go/no-go experiment is interpreted.
#[derive(Debug, Clone)]
pub struct BsDiagnostics {
    /// Fraction of backward levels where the chosen slot differs from the
    /// reference slot — the same slot-identity renewal `csmc_as` reports.
    pub trajectory_renewal: f64,
    /// Renewal resolved into tenths of the substep series (`NaN` = empty bin).
    pub renewal_by_bin: [f64; RENEWAL_BINS],
    pub n_substeps: usize,
    /// Mean, over backward levels, of the fraction of candidates with a
    /// finite backward weight (feasible edges). The in-cloud version of the
    /// spike's 24.8%-feasible proxy.
    pub candidate_feasible_frac: f64,
    /// Total transition-density lattice terms summed this sweep — the
    /// naive-cost instrument (proxy said ~7×10⁷ at N = 2,400; this is the
    /// real number).
    pub total_lattice_terms: u64,
}

/// One conditional state-space sweep. `reference` is the retained trajectory
/// (flow record) from the previous sweep; only its STATE path (and the
/// accumulator path derived from its flows) conditions this sweep.
#[allow(clippy::too_many_arguments)]
pub fn csmc_bs(
    model: &CompiledModel,
    params: &[f64],
    reference: &PGASTrajectory,
    n_particles: usize,
    dt: f64,
    obs_model: &MultiStreamObsModel,
    seed: u64,
    obs_at_substep: &ObsAtSubstep,
    firing: EffectFiring<'_>,
    analysis: &StateTransitionAnalysis,
    // gh#747: threaded through for the same reason `csmc_as` takes it -- the
    // selection is stamped onto each particle's RNG at construction, so it
    // cannot be lost to work-stealing. Defaulting it here would make
    // `--binomial btrs` a silent no-op on the backward-kernel path, which is
    // the unreachable-selection defect gh#747 closed.
    binomial: crate::rng::BinomialAlgorithm,
) -> Result<(PGASTrajectory, BsDiagnostics), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = reference.substeps.len();
    let n_tr = model.model.transitions.len();
    let n_comp = model.int_local_to_global.len();
    let streams = obs_model.incidence_streams();
    let n_streams = streams.len();
    let j_ref = n_particles - 1;

    let per_eval_scratch = crate::resolved_expr::stage_per_eval(model, params, t_start, dt);
    let per_eval = per_eval_scratch.as_deref();
    let fire_steps = model.resolve_fire_steps(dt, params);

    // ── Which interval streams reset at each substep (probed through the
    // obs model's own reset seam, so the two cannot disagree) ──
    let mut reset_mask: Vec<Vec<bool>> = vec![vec![false; n_streams]; n_substeps];
    for (&s, &obs_idx) in obs_at_substep.iter() {
        let mut probe = vec![1u64; n_streams];
        obs_model.reset_due_acc(obs_idx, &mut probe);
        for k in 0..n_streams {
            reset_mask[s][k] = probe[k] == 0;
        }
    }

    // ── The reference's state path: counts from the record, accumulators by
    // replaying its flows through H under the reset convention ──
    let mut ref_acc_path: Vec<Vec<u64>> = Vec::with_capacity(n_substeps);
    {
        let mut a = vec![0u64; n_streams];
        for s in 0..n_substeps {
            for (k, (_, idxs)) in streams.iter().enumerate() {
                for &j in idxs {
                    a[k] += reference.substeps[s].flows[j];
                }
            }
            ref_acc_path.push(a.clone()); // pre-reset boundary value
            for k in 0..n_streams {
                if reset_mask[s][k] {
                    a[k] = 0;
                }
            }
        }
    }

    // ── Forward conditional filter ──
    let mut rngs = init_particle_rngs(seed, n_particles, 0, binomial);
    let mut counts: Vec<Vec<i64>> = (0..n_particles)
        .map(|j| -> Result<Vec<i64>, SimError> {
            if j == j_ref {
                return Ok(reference.initial_counts.clone());
            }
            crate::inference::pgas::draw_free_particle_initial_state(
                model, params, &mut rngs[j], per_eval,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let initial_counts_per_particle = counts.clone();
    let mut acc: Vec<Vec<u64>> = vec![vec![0u64; n_streams]; n_particles];
    let mut substep_flows: Vec<Vec<u64>> = vec![vec![0u64; n_tr]; n_particles];
    let mut particle_reals: Vec<crate::state::RealState> = (0..n_particles)
        .map(|_| crate::state::RealState::new(model.real_local_to_global.len()))
        .collect();
    let mut scratches: Vec<StepScratch> =
        (0..n_particles).map(|_| StepScratch::new(model)).collect();
    let mut log_weights = vec![0.0f64; n_particles];
    let mut resample_rng = StatefulRng::new_stream(seed, RESAMPLE_RNG_STREAM);

    // Histories the backward pass reads: states, accumulators (pre-reset),
    // and filter weights, per boundary.
    let mut hist_counts: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_substeps);
    let mut hist_acc: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_substeps);
    let mut hist_logw: Vec<Vec<f64>> = Vec::with_capacity(n_substeps);

    for s in 0..n_substeps {
        let t = reference.substeps[s].t0;
        let step_dt = reference.substeps[s].dt_substep;

        // Conditional resample (reference slot keeps itself by construction).
        let uniform = log_weights.iter().all(|&w| (w - log_weights[0]).abs() < 1e-10);
        if !uniform {
            let indices = conditional_multinomial_resample(&log_weights, j_ref, &mut resample_rng);
            let mut new_counts = Vec::with_capacity(n_particles);
            let mut new_acc = Vec::with_capacity(n_particles);
            for j in 0..n_particles {
                new_counts.push(counts[indices[j]].clone());
                new_acc.push(acc[indices[j]].clone());
            }
            counts = new_counts;
            acc = new_acc;
        }

        // Propagate free particles (parallel; byte-identical per-slot RNG
        // streams as the innovation kernel's loop). The reference slot is
        // PINNED below, not propagated.
        let results: Vec<Result<(), SimError>> = counts
            .par_iter_mut()
            .zip(substep_flows.par_iter_mut())
            .zip(particle_reals.par_iter_mut())
            .zip(rngs.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .enumerate()
            .map(|(j, ((((cnt, flows), real), rng), scratch))| {
                if j == j_ref {
                    return Ok(());
                }
                for f in flows.iter_mut() {
                    *f = 0;
                }
                scratch.gamma_used.clear();
                fill_producer_batch(
                    model, &fire_steps, t + step_dt, dt, s, firing, &mut scratch.effect_batch,
                );
                step_one(model, cnt, flows, real, params, t, step_dt, per_eval, rng, scratch)
            })
            .collect();
        for r in results {
            r?;
        }
        // Fold this substep's flows into the free particles' accumulators.
        for j in 0..n_particles {
            if j == j_ref {
                continue;
            }
            for (k, (_, idxs)) in streams.iter().enumerate() {
                for &tr in idxs {
                    acc[j][k] += substep_flows[j][tr];
                }
            }
        }
        // ── Reference slot: STATE-pinned. Z_s[j_ref] = Z*_s, accumulators
        // included; no innovation record is replayed from any ancestor. ──
        counts[j_ref].copy_from_slice(&reference.substeps[s].counts_after);
        acc[j_ref].copy_from_slice(&ref_acc_path[s]);

        // Score observations from Z (pre-reset accumulators + counts).
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            log_weights
                .par_iter_mut()
                .zip(counts.par_iter())
                .zip(acc.par_iter())
                .for_each(|((lw, cnt), a)| {
                    *lw = obs_model.log_likelihood_from_flows_and_counts(a, cnt, obs_idx, params);
                });
        } else {
            log_weights.fill(0.0);
        }

        // History at boundary s: state, pre-reset accumulator, weight.
        hist_counts.push(counts.clone());
        hist_acc.push(acc.clone());
        hist_logw.push(log_weights.clone());

        // Reset due bins for the ongoing recursion (after the boundary value
        // was recorded — the Z convention).
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            for a in acc.iter_mut() {
                obs_model.reset_due_acc(obs_idx, a);
            }
        }
    }

    // ── Backward stitch ──
    let mut chosen: Vec<usize> = vec![0; n_substeps];
    let last = n_substeps - 1;
    chosen[last] = sample_categorical_log(&hist_logw[last], &mut resample_rng).unwrap_or(j_ref);

    let mut n_feasible_sum = 0usize;
    let mut n_candidate_sum = 0usize;
    let mut total_lattice_terms = 0u64;

    let mut bw = vec![f64::NEG_INFINITY; n_particles];
    for s in (1..=last).rev() {
        let k = chosen[s];
        let target_counts = &hist_counts[s][k];
        let target_acc = &hist_acc[s][k];
        let t = reference.substeps[s].t0;
        let step_dt = reference.substeps[s].dt_substep;

        let level: Vec<(f64, u64)> = (0..n_particles)
            .into_par_iter()
            .map(|j| -> Result<(f64, u64), SimError> {
                let base = hist_logw[s - 1][j];
                if base == f64::NEG_INFINITY {
                    return Ok((f64::NEG_INFINITY, 0));
                }
                let before = &hist_counts[s - 1][j];
                let d_counts: Vec<i64> = (0..n_comp)
                    .map(|c| target_counts[c] - before[c])
                    .collect();
                let d_acc: Vec<i64> = (0..n_streams)
                    .map(|kk| {
                        let carried = if reset_mask[s - 1][kk] {
                            0
                        } else {
                            hist_acc[s - 1][j][kk] as i64
                        };
                        target_acc[kk] as i64 - carried
                    })
                    .collect();
                let (td, n_terms) = log_state_transition_density(
                    model, analysis, before, &d_counts, &d_acc, params, t, step_dt, per_eval,
                )?;
                Ok((base + td, n_terms))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (j, &(w, n_terms)) in level.iter().enumerate() {
            bw[j] = w;
            total_lattice_terms += n_terms;
            n_candidate_sum += 1;
            if w > f64::NEG_INFINITY {
                n_feasible_sum += 1;
            }
        }
        chosen[s - 1] = match sample_categorical_log(&bw, &mut resample_rng) {
            Some(j) => j,
            None => k, // no feasible predecessor: keep the current slot's own
        };
    }

    // Backward step to the initial boundary: uniform prior weights (each
    // initial state is an exact draw from the init law).
    let init_choice = {
        let k = chosen[0];
        let target_counts = &hist_counts[0][k];
        let target_acc = &hist_acc[0][k];
        let t = reference.substeps[0].t0;
        let step_dt = reference.substeps[0].dt_substep;
        let level: Vec<(f64, u64)> = (0..n_particles)
            .into_par_iter()
            .map(|j| -> Result<(f64, u64), SimError> {
                let before = &initial_counts_per_particle[j];
                let d_counts: Vec<i64> =
                    (0..n_comp).map(|c| target_counts[c] - before[c]).collect();
                let d_acc: Vec<i64> = (0..n_streams).map(|kk| target_acc[kk] as i64).collect();
                let (td, n_terms) = log_state_transition_density(
                    model, analysis, before, &d_counts, &d_acc, params, t, step_dt, per_eval,
                )?;
                Ok((td, n_terms))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for &(_, n_terms) in &level {
            total_lattice_terms += n_terms;
        }
        n_candidate_sum += n_particles;
        n_feasible_sum += level.iter().filter(|(w, _)| *w > f64::NEG_INFINITY).count();
        let w: Vec<f64> = level.iter().map(|(w, _)| *w).collect();
        sample_categorical_log(&w, &mut resample_rng).unwrap_or(chosen[0])
    };

    // ── Reconstruction: flows for each stitched edge ──
    let mut substeps_out = Vec::with_capacity(n_substeps);
    let mut renewal_bins = crate::inference::pgas::RenewalBins::new(n_substeps);
    let mut n_from_ref = 0usize;
    let initial_counts = initial_counts_per_particle[init_choice].clone();
    let mut prev_counts = initial_counts.clone();
    let mut prev_acc: Vec<u64> = vec![0; n_streams];
    for s in 0..n_substeps {
        let k = chosen[s];
        let renewed = k != j_ref;
        if !renewed {
            n_from_ref += 1;
        }
        renewal_bins.record(s, renewed);

        let after = hist_counts[s][k].clone();
        let a_after = &hist_acc[s][k];
        let t = reference.substeps[s].t0;
        let step_dt = reference.substeps[s].dt_substep;
        let d_counts: Vec<i64> = (0..n_comp).map(|c| after[c] - prev_counts[c]).collect();
        let d_acc: Vec<i64> = (0..n_streams)
            .map(|kk| a_after[kk] as i64 - prev_acc[kk] as i64)
            .collect();
        let flows = sample_edge_flows(
            model, analysis, &prev_counts, &d_counts, &d_acc, params, t, step_dt, per_eval,
            &mut resample_rng,
        )?
        .ok_or_else(|| {
            SimError::Validation(format!(
                "csmc_bs reconstruction: no compatible flows at substep {s} — the \
                 backward stitch selected an edge its own density scored feasible, \
                 so this is a kernel bug, not a model property"
            ))
        })?;
        substeps_out.push(SubstepRecord {
            counts_before: prev_counts.clone(),
            counts_after: after.clone(),
            flows,
            gammas: Vec::new(),
            t0: t,
            dt_substep: step_dt,
        });
        prev_counts = after;
        prev_acc = a_after.clone();
        for kk in 0..n_streams {
            if reset_mask[s][kk] {
                prev_acc[kk] = 0;
            }
        }
    }

    let diag = BsDiagnostics {
        trajectory_renewal: 1.0 - n_from_ref as f64 / n_substeps as f64,
        renewal_by_bin: renewal_bins.finish(),
        n_substeps,
        candidate_feasible_frac: if n_candidate_sum > 0 {
            n_feasible_sum as f64 / n_candidate_sum as f64
        } else {
            f64::NAN
        },
        total_lattice_terms,
    };
    Ok((
        PGASTrajectory { initial_counts, substeps: substeps_out },
        diag,
    ))
}
