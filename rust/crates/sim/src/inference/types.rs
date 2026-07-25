//! Core types for the particle filter and inference algorithms.
//!
//! Flat array layout for cache-friendly resampling — copying a particle's
//! state is one contiguous memcpy, not a pointer chase through Vec<Vec<...>>.
//!
//! Also owns the shared inference infrastructure types — `Transform`,
//! `EstimatedParam` — that all algorithms (IF2, PGAS, PMMH) use. These
//! live here rather than in `if2.rs` so that removing or replacing IF2
//! does not take the shared type contract with it.

use super::traits::Resettable;

// ── Parameter transform ───────────────────────────────────────────────────────

/// The unconstrained-space transform applied to an estimated parameter.
///
/// Matches Stan's lower/upper bounded parameter conventions. Used by IF2,
/// PGAS, and PMMH for all scale-management operations (to/from transformed,
/// Jacobian, gradient chain rule).
#[derive(Clone, Debug)]
pub enum Transform {
    /// Log transform with bounds clamping on the inverse.
    /// Correct for rates, positive quantities, counts.
    /// `from_transformed` saturates `z.exp()` at `[lo, hi]` — out-of-bounds
    /// particles get bad log-likelihood and are resampled away.
    ///
    /// The declared bounds are a hard search box (`docs/camdl-inference-spec.md`
    /// §3.2), so the target they induce is the posterior **truncated** to
    /// `θ ∈ [lo, hi]`: inside, `θ = exp(z)` with log-Jacobian `z`; outside,
    /// `dθ/dz = 0` and the density is zero. [`log_saturated`] is the single
    /// definition of "outside", used by all three derivative accessors.
    Log { lo: f64, hi: f64 },
    /// Scaled logit mapping `[lo, hi]` to `(−∞, +∞)`.
    /// Correct for probabilities. Bounds enforced by the logistic function
    /// (output always in `(0, 1)`). For narrow bounds like `[0.01, 0.10]`
    /// the logit-scaled position can be extreme (|z| > 2), compressing the
    /// effective perturbation range; the preflight diagnostic warns about this.
    Logit { lo: f64, hi: f64 },
    /// No transform. For unconstrained real parameters.
    None,
}

/// One estimated parameter: its name, position in the full model parameter
/// vector, declared transform and bounds, and per-algorithm adaptation state.
///
/// Shared by IF2, PGAS, and PMMH. Constructed by the CLI from the fit config
/// and passed into algorithm entry points.
#[derive(Clone, Debug)]
pub struct EstimatedParam {
    /// Parameter name (for reporting and gradient lookup).
    pub name: String,
    /// Index into the full model `params` array.
    pub index: usize,
    /// Starting value for IF2 (or the current value on resume).
    pub initial: f64,
    /// Random walk standard deviation on the *transformed* scale.
    /// Shrinks by `cooling_fraction` each IF2 iteration.
    pub rw_sd: f64,
    /// Scale transform applied before perturbation / MH proposals.
    pub transform: Transform,
    /// Natural-scale lower bound (used for display and random-start sampling).
    pub lower: f64,
    /// Natural-scale upper bound (used for display and random-start sampling).
    pub upper: f64,
    /// Whether `rw_sd` was auto-computed from the data (for preflight reporting).
    pub rw_sd_auto: bool,
    /// If true, perturb only at t=0 (initial-value parameter: S₀, E₀, I₀ …).
    /// Matches pomp's `ivp()` in `rw.sd`.
    pub ivp: bool,
}

/// Is `z` outside the `Log` transform's declared support `[lo, hi]`?
///
/// The single definition of "the inverse map is saturated here", shared by
/// [`EstimatedParam::log_jacobian`], [`EstimatedParam::jacobian_grad`] and
/// [`EstimatedParam::transform_deriv`] so a derivative can never describe a
/// different map than the one `from_transformed` implements (gh#374: all three
/// returned the derivatives of the *unclamped* `exp`, so a sampler pressed
/// against a bound got a spuriously non-zero gradient instead of a flat one —
/// and, in the MH ratio, a log-Jacobian `z` that grew without bound while the
/// likelihood stayed pinned, rewarding ever-larger `z`).
///
/// The test is in **z-space** (`z ∉ [ln lo, ln hi]`) rather than θ-space
/// (`exp(z) ∉ [lo, hi]`) so that it agrees with `to_transformed`, whose log
/// branch always returns `ln(x.clamp(lo, hi))`. `exp(ln(hi))` rounds one ulp
/// *above* `hi` for many ordinary bounds (10, 100, 0.1, 1e-3, 3, …), so a
/// θ-space test would declare `z = ln(hi)` out-of-support and hand a chain
/// initialised at its upper bound a `−∞` posterior. The two tests differ only
/// within one ulp of the bound.
///
/// Non-finite and degenerate bounds fall out correctly, matching the sides on
/// which `clamp` can actually fire: `hi = +∞ → ln hi = +∞` (never saturated
/// above), `lo = 0 → ln lo = −∞` (never saturated below, and `exp(z) > 0`
/// always), and `lo < 0 → ln lo = NaN`, whose comparison is `false` — again
/// never saturated below, which is right because `exp(z) > 0 > lo`.
fn log_saturated(z: f64, lo: f64, hi: f64) -> bool {
    z < lo.ln() || z > hi.ln()
}

impl EstimatedParam {
    /// Map a natural-scale value to the unconstrained (z) scale.
    pub fn to_transformed(&self, x: f64) -> f64 {
        match &self.transform {
            Transform::Log { lo, hi } => x.clamp(*lo, *hi).max(LOG_PROB_FLOOR).ln(),
            Transform::Logit { lo, hi } => {
                let p = ((x - lo) / (hi - lo)).clamp(PROB_FRACTION_EPS, 1.0 - PROB_FRACTION_EPS);
                (p / (1.0 - p)).ln()
            }
            Transform::None => x,
        }
    }

    /// Map an unconstrained value back to the natural scale.
    pub fn from_transformed(&self, z: f64) -> f64 {
        match &self.transform {
            Transform::Log { lo, hi } => {
                // Clamp to declared bounds — prevents NaN/panic downstream.
                // Out-of-bounds particles get bad loglik and are resampled away.
                z.exp().clamp(*lo, *hi)
            }
            Transform::Logit { lo, hi } => {
                let p = 1.0 / (1.0 + (-z).exp());
                lo + p * (hi - lo)
                // Bounds enforced by construction — no clamp needed.
            }
            Transform::None => z,
        }
    }

    /// log |dθ/dz| for the transform θ = f(z).
    /// Needed for the MH acceptance ratio when proposing on the transformed scale.
    ///
    /// Log-transform:   θ = exp(z)              → log-Jacobian = z
    /// Logit-transform: θ = lo + (hi−lo)·σ(z)   → log-Jacobian = log((hi−lo)·p·(1−p))
    /// No transform:    Jacobian = 1             → log-Jacobian = 0
    ///
    /// Where the `Log` inverse saturates at a declared bound the map is flat,
    /// so `|dθ/dz| = 0` and this is `−∞`: the truncated target has no density
    /// outside its support. Every consumer treats that as a clean rejection —
    /// `mh_accept` (`pmmh.rs`) rejects a move *to* `−∞` and still accepts the
    /// `+∞` escape *from* one, and NUTS scores a non-finite energy as divergent
    /// and refuses to commit a non-finite proposal (`nuts.rs`).
    pub fn log_jacobian(&self, z: f64) -> f64 {
        match &self.transform {
            Transform::Log { lo, hi } if log_saturated(z, *lo, *hi) => f64::NEG_INFINITY,
            Transform::Log { .. } => z,
            Transform::Logit { lo, hi } => {
                let p = 1.0 / (1.0 + (-z).exp());
                ((hi - lo) * p * (1.0 - p)).ln()
            }
            Transform::None => 0.0,
        }
    }

    /// d/dz log|dθ/dz| — derivative of the log-Jacobian w.r.t. z.
    ///
    /// Zero, not the unclamped `+1`, where the `Log` inverse saturates: the
    /// log-Jacobian is constant (`−∞`) there, so it contributes no gradient.
    /// Keeping it finite is what lets the leapfrog integrator carry a usable
    /// momentum into the divergence check instead of a poisoned one.
    pub fn jacobian_grad(&self, z: f64) -> f64 {
        match &self.transform {
            Transform::Log { lo, hi } if log_saturated(z, *lo, *hi) => 0.0,
            Transform::Log { .. } => 1.0,
            Transform::Logit { .. } => {
                let p = 1.0 / (1.0 + (-z).exp());
                1.0 - 2.0 * p
            }
            Transform::None => 0.0,
        }
    }

    /// dθ/dz — derivative of the natural-scale value with respect to z.
    /// Used in the chain rule: d(f(θ))/dz = d(f)/dθ × dθ/dz.
    ///
    /// Zero where the `Log` inverse saturates: θ is pinned at the bound, so
    /// moving z there does not move θ and the data/prior term contributes
    /// nothing to the z-gradient.
    pub fn transform_deriv(&self, z: f64) -> f64 {
        match &self.transform {
            Transform::Log { lo, hi } if log_saturated(z, *lo, *hi) => 0.0,
            Transform::Log { .. } => z.exp(),
            Transform::Logit { lo, hi } => {
                let p = 1.0 / (1.0 + (-z).exp());
                (hi - lo) * p * (1.0 - p)
            }
            Transform::None => 1.0,
        }
    }

    /// Delta method: convert a natural-scale `rw_sd` to the transformed scale.
    /// Matches pomp's convention: the user specifies `rw.sd` on the natural scale.
    pub fn transformed_sd(&self, natural_sd: f64, current_value: f64) -> f64 {
        match &self.transform {
            Transform::Log { .. } => natural_sd / current_value.max(LOG_PROB_FLOOR),
            Transform::Logit { lo, hi } => {
                let range = hi - lo;
                let p = ((current_value - lo) / range).clamp(PROB_FRACTION_EPS, 1.0 - PROB_FRACTION_EPS);
                natural_sd / (range * p * (1.0 - p))
            }
            Transform::None => natural_sd,
        }
    }
}

// ── Numeric constants ─────────────────────────────────────────────────────────

/// Minimum argument for `ln()` in log-weight computations.
///
/// Chosen so that `ln(LOG_PROB_FLOOR) ≈ −690`, well above the
/// underflow threshold for any realistic particle count: even at
/// N = 10_000 particles, a weight of 1e-300 contributes roughly −690 to
/// `log_sum_exp`, which rounds to −∞ for that particle but does not
/// corrupt the normaliser.
///
/// Do NOT reduce below `f64::MIN_POSITIVE` (≈ 5×10⁻³²⁴), which would
/// produce −∞ and defeat the purpose.
pub const LOG_PROB_FLOOR: f64 = 1e-300;

/// Interior clamp keeping an IVP / logit-transform probability strictly inside
/// (0, 1), so the transform and the Binomial(N, p) draw stay finite. Shared by
/// the logit transform here and the PGAS IVP density/gradient (`pgas.rs`,
/// `pgas_grad.rs`) — value and gradient MUST use the same value. Distinct in
/// concept from the correlated-PF base-uniform clamp (`BASE_UNIFORM_EPS`),
/// which shares the magnitude.
pub const PROB_FRACTION_EPS: f64 = 1e-10;

/// Reserved stream index for the per-algorithm resampling RNG.
///
/// Per-particle streams use indices `[0, n_particles)`. This constant
/// is set high enough (2^48) to never collide with any realistic particle
/// count, making it safe to pass to `StatefulRng::new_stream` alongside
/// per-particle streams from the same base seed.
pub const RESAMPLE_RNG_STREAM: u64 = 1u64 << 48;

// ── RNG helpers ───────────────────────────────────────────────────────────────

/// Allocate `n` per-particle RNG streams derived from `seed`.
///
/// `stream_offset` separates particles from different callers or iterations:
/// - Particle filter and PGAS pass `0` (particles are differentiated by
///   index alone).
/// - IF2 passes `(iter as u64) << 32` so each iteration's particle streams
///   are disjoint from all other iterations (top 32 bits = iteration index,
///   bottom 32 bits = particle index).
pub fn init_particle_rngs(
    seed: u64,
    n: usize,
    stream_offset: u64,
) -> Vec<crate::rng::StatefulRng> {
    (0..n)
        .map(|i| crate::rng::StatefulRng::new_stream(seed, stream_offset | (i as u64)))
        .collect()
}

/// Restore the unconstrained z-values from a saved resume state, reordering
/// them to match the current `if2_params` ordering.
///
/// The resume state stores `param_names` alongside `transformed` z-values.
/// Because HashMap iteration order is non-deterministic, the current run's
/// `if2_params` may be in a different order than when the state was saved.
/// Parameters missing from the saved state are recomputed from `current_params`
/// with a warning. If `saved_names` is empty (legacy state before param_names
/// was added), all z-values are recomputed from `current_params`.
pub fn restore_z_values(
    saved_names: &[String],
    saved_z: &[f64],
    if2_params: &[EstimatedParam],
    current_params: &[f64],
) -> Vec<f64> {
    if saved_names.is_empty() || saved_names.len() != saved_z.len() {
        eprintln!("  warning: resume state lacks param_names — recomputing z from params.");
        return if2_params.iter()
            .map(|spec| spec.to_transformed(current_params[spec.index]))
            .collect();
    }

    let saved: std::collections::HashMap<&str, f64> = saved_names.iter()
        .zip(saved_z.iter())
        .map(|(name, &z)| (name.as_str(), z))
        .collect();

    if2_params.iter().map(|spec| {
        if let Some(&z) = saved.get(spec.name.as_str()) {
            z
        } else {
            eprintln!("  warning: param '{}' not found in resume state, computing from theta", spec.name);
            spec.to_transformed(current_params[spec.index])
        }
    }).collect()
}

/// State of one particle: compartment counts + flow accumulators.
#[derive(Clone, Debug)]
pub struct ParticleState {
    /// Integer compartment values (local int indices, same layout as IntState).
    pub counts: Vec<i64>,
    /// Cumulative transition flows since last observation.
    /// Reset after each observation time (used for incidence projections).
    ///
    /// This is the per-TRANSITION, this-interval tally written by `step_one`
    /// (an additive `flows[tr] += count` per substep), blanket-reset by
    /// `reset_flows()` once per observation interval. Its lifecycle is
    /// UNCHANGED by multi-cadence Phase 2a — the forward/substep path, the
    /// correlated-PF resampling sort key (`correlated_pf.rs`), and
    /// `write_final_states` (`pfilter.rs`) all keep reading it exactly as
    /// before.
    pub flow_accumulators: Vec<u64>,
    /// Per-Interval-stream incidence bin (multi-cadence Phase 2a, "Option Z").
    /// One `u64` per incidence (`FlowSum`) observation stream, in the obs
    /// model's `interval_slots` order. PERSISTENT across observation intervals:
    /// folded once per interval from `flow_accumulators` (via
    /// `ObservationModel::fold_into_acc`), read at scoring time as the stream's
    /// already-summed bin, and reset PER-STREAM (only the streams scheduled at
    /// the current union index — `ObservationModel::reset_due_acc`), NOT
    /// blanket-reset by `reset_flows()`.
    ///
    /// Homogeneous (every stream scheduled every interval) ⇒ folded-scored-reset
    /// every interval ⇒ byte-identical to scoring the global accumulator.
    pub acc: Vec<u64>,
}

impl ParticleState {
    /// `n_interval_streams` sizes `acc` (one bin per incidence stream — owned by
    /// the OBS model, not the compiled model). The process model's
    /// `initial_state` does not know it and passes `0`; the FILTER (which holds
    /// the obs model) allocates the swarm states with the real count, since the
    /// filter copies only `init.counts` into the swarm.
    pub fn new(n_compartments: usize, n_transitions: usize, n_interval_streams: usize) -> Self {
        ParticleState {
            counts: vec![0; n_compartments],
            flow_accumulators: vec![0; n_transitions],
            acc: vec![0; n_interval_streams],
        }
    }

    /// Reset flow accumulators to zero (called after each observation).
    /// Zeroes ONLY `flow_accumulators` — the per-transition this-interval tally.
    /// The per-stream `acc` is reset SEPARATELY and per-stream by the obs model
    /// (`reset_due_acc`), because it must survive a sibling stream's union-time.
    pub fn reset_flows(&mut self) {
        self.reset_accumulators();
    }

}

impl Resettable for ParticleState {
    fn reset_accumulators(&mut self) {
        for f in &mut self.flow_accumulators { *f = 0; }
    }
}

impl ParticleState {
    /// Clamp negative compartment values to zero.
    pub fn clamp_nonneg(&mut self) {
        for c in &mut self.counts {
            if *c < 0 { *c = 0; }
        }
    }
}

/// Storage for N particles with log-weights.
pub struct ParticleSwarm {
    pub n_particles: usize,
    pub states: Vec<ParticleState>,
    pub log_weights: Vec<f64>,
}

impl ParticleSwarm {
    pub fn new(
        n_particles: usize,
        n_compartments: usize,
        n_transitions: usize,
        n_interval_streams: usize,
    ) -> Self {
        ParticleSwarm {
            n_particles,
            states: (0..n_particles)
                .map(|_| ParticleState::new(n_compartments, n_transitions, n_interval_streams))
                .collect(),
            log_weights: vec![0.0; n_particles],
        }
    }

    /// Effective sample size: ESS = 1 / Σ(w_normalized²).
    /// Returns N when all weights are equal, 1 when one particle dominates.
    ///
    /// **Degenerate-case contract:** returns 0.0 if the maximum log-weight
    /// is non-finite (every weight is `-∞` or `NaN`-poisoned), or if the
    /// post-shift sum is non-positive or non-finite. ESS=0 signals to the
    /// filter that the particle cloud has collapsed and resampling cannot
    /// produce informative draws — a stronger signal than the
    /// uniform-weight fallback used by `normalize_log_weights`.
    pub fn ess(&self) -> f64 {
        ess_from_log_weights(&self.log_weights)
    }
}

/// Effective sample size from log-weights: `ESS = (Σw)² / Σw²` on max-shifted
/// weights. The single source for [`ParticleSwarm::ess`] and IF2's
/// per-iteration degeneracy watchdog, which holds a `Vec<f64>` of log-weights
/// rather than a `ParticleSwarm`.
///
/// **Degenerate-case contract:** returns 0.0 if the maximum log-weight is
/// non-finite (every weight is `-∞` or `NaN`-poisoned), or if the post-shift
/// sum is non-positive or non-finite — a stronger collapse signal than the
/// uniform-weight fallback used by [`normalize_log_weights`].
pub fn ess_from_log_weights(log_weights: &[f64]) -> f64 {
    let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() { return 0.0; }
    let sum_w: f64 = log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
    let sum_w2: f64 = log_weights.iter().map(|&lw| (2.0 * (lw - max_lw)).exp()).sum();
    if !sum_w.is_finite() || !sum_w2.is_finite() || sum_w2 <= 0.0 { return 0.0; }
    (sum_w * sum_w) / sum_w2
}

/// Numerically stable log-sum-exp.
///
/// Im2 in the 2026-04-19 inference review batch 1: distinguish
/// +∞ vs −∞. If `max = +∞`, at least one entry is +∞ and the
/// result is also +∞ (not −∞ as the old bulk-check produced).
/// If `max = −∞`, every entry is −∞ and the result is −∞.
pub fn log_sum_exp(log_values: &[f64]) -> f64 {
    let max = log_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    if max == f64::INFINITY { return f64::INFINITY; }
    max + log_values.iter().map(|&lv| (lv - max).exp()).sum::<f64>().ln()
}

/// Snyder τ²: the variance of the per-particle incremental log importance
/// weights at one assimilation step, computed over the *finite* (live)
/// particles. Dead particles (`−∞`) are excluded — ESS already flags their
/// collapse, and including them would make the variance infinite.
///
/// This is the high-dimensional-PF degeneracy predictor: the ensemble size
/// needed to avoid weight collapse scales as `exp(τ²/2)` (Snyder, Bengtsson,
/// Bickel & Anderson 2008, *Obstacles to High-Dimensional Particle Filtering*,
/// MWR 136:4629–4640). Returns 0.0 if fewer than two particles carry a finite
/// weight (no spread to measure).
pub fn logw_variance(log_weights: &[f64]) -> f64 {
    let finite: Vec<f64> = log_weights.iter().copied().filter(|w| w.is_finite()).collect();
    let n = finite.len();
    if n < 2 {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    finite.iter().map(|&w| (w - mean) * (w - mean)).sum::<f64>() / n as f64
}

/// Normalize log-weights to a probability vector, with uniform fallback.
///
/// Applies the log-sum-exp trick: subtracts the max log-weight before
/// exponentiating, then divides by the sum. The max-subtraction keeps
/// `(lw - max).exp()` in `[0, 1]` regardless of the absolute scale of
/// `log_weights`, so the sum cannot overflow.
///
/// **Degenerate-case contract:** if the maximum log-weight is non-finite
/// (every weight is `-∞`, or any weight is `NaN` and all weights are NaN,
/// or the slice is empty), or if the post-shift sum is non-positive, the
/// function returns `[1/n, 1/n, ..., 1/n]`. This is the conservative
/// choice: a particle filter or weighted-quantile call with
/// uniform-degenerate weights should treat all particles as equally
/// informative rather than propagate a `NaN` into downstream statistics.
///
/// `f64::max` ignores `NaN` operands when at least one argument is
/// non-NaN, so a slice mixing finite values with `NaN` will pick a
/// finite max — the corresponding `NaN.exp()` then leaks into `sum`,
/// producing `NaN`. The `sum.is_finite() && sum > 0.0` guard catches
/// that case and falls back to uniform.
///
/// Returns `vec![]` only when `log_weights` is empty.
pub fn normalize_log_weights(log_weights: &[f64]) -> Vec<f64> {
    let n = log_weights.len();
    if n == 0 { return Vec::new(); }
    let inv_n = 1.0 / n as f64;
    let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return vec![inv_n; n];
    }
    let raw: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let sum: f64 = raw.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return vec![inv_n; n];
    }
    raw.iter().map(|&w| w / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-12 }

    #[test]
    fn empty_input_returns_empty() {
        assert!(normalize_log_weights(&[]).is_empty());
    }

    #[test]
    fn equal_weights_normalize_to_uniform() {
        let w = normalize_log_weights(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(w.len(), 4);
        for x in &w { assert!(approx_eq(*x, 0.25)); }
    }

    #[test]
    fn extreme_log_weights_do_not_overflow() {
        let w = normalize_log_weights(&[1e9, 1e9 + 2.0, 1e9 + 1.0]);
        let sum: f64 = w.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(w.iter().all(|x| x.is_finite()));
        // largest weight should dominate
        assert!(w[1] > w[2] && w[2] > w[0]);
    }

    #[test]
    fn all_neg_inf_falls_back_to_uniform() {
        let w = normalize_log_weights(&[f64::NEG_INFINITY; 3]);
        for x in &w { assert!(approx_eq(*x, 1.0 / 3.0)); }
    }

    #[test]
    fn all_nan_falls_back_to_uniform() {
        let w = normalize_log_weights(&[f64::NAN, f64::NAN]);
        assert_eq!(w.len(), 2);
        for x in &w { assert!(approx_eq(*x, 0.5),
            "all-NaN must yield uniform, got {:?}", w); }
    }

    #[test]
    fn nan_mixed_with_finite_falls_back_to_uniform() {
        // Once NaN enters the sum, the result is NaN; helper must catch
        // and return uniform rather than propagate poison.
        let w = normalize_log_weights(&[0.0, f64::NAN, -1.0]);
        assert_eq!(w.len(), 3);
        for x in &w { assert!(approx_eq(*x, 1.0 / 3.0),
            "NaN-poisoned slice must fall back to uniform, got {:?}", w); }
    }

    #[test]
    fn pos_inf_max_falls_back_to_uniform() {
        // +∞ is non-finite; helper falls back to uniform rather than
        // attempting to compute (∞ - ∞).exp() = NaN.
        let w = normalize_log_weights(&[0.0, f64::INFINITY, -1.0]);
        for x in &w { assert!(approx_eq(*x, 1.0 / 3.0)); }
    }

    #[test]
    fn logw_variance_uniform_is_zero() {
        // Equal weights → no spread → τ² = 0 (a perfectly healthy filter).
        assert!(approx_eq(logw_variance(&[3.0, 3.0, 3.0, 3.0]), 0.0));
    }

    #[test]
    fn logw_variance_matches_population_variance() {
        // Population variance of {0,2,4,6}: mean 3, Σ(d²)=9+1+1+9=20, /4 = 5.
        assert!(approx_eq(logw_variance(&[0.0, 2.0, 4.0, 6.0]), 5.0));
    }

    #[test]
    fn logw_variance_excludes_dead_particles() {
        // −∞ (dead) particles are dropped; τ² is over the live ensemble only.
        let with_dead = logw_variance(&[0.0, 2.0, f64::NEG_INFINITY, 4.0, 6.0]);
        let live_only = logw_variance(&[0.0, 2.0, 4.0, 6.0]);
        assert!(approx_eq(with_dead, live_only));
        // Fewer than two live particles → no measurable spread.
        assert!(approx_eq(logw_variance(&[f64::NEG_INFINITY, 1.0]), 0.0));
    }

    #[test]
    fn logw_variance_grows_with_obs_informativeness() {
        // The degeneracy direction Snyder predicts: a sharper (more spread)
        // weight set has larger τ² ⇒ larger implied particle count exp(τ²/2).
        let mild = logw_variance(&[-0.1, 0.0, 0.1]);
        let sharp = logw_variance(&[-10.0, 0.0, 10.0]);
        assert!(sharp > mild);
    }

    // ── Log-transform saturation (gh#374) ────────────────────────────────
    //
    // `from_transformed` saturates θ at the declared bound. The three
    // derivative accessors must describe THAT map, not the unclamped `exp`.

    fn log_param(lo: f64, hi: f64) -> EstimatedParam {
        EstimatedParam {
            name: "beta".into(),
            index: 0,
            initial: (lo * hi).sqrt(),
            rw_sd: 0.1,
            transform: Transform::Log { lo, hi },
            lower: lo,
            upper: hi,
            rw_sd_auto: false,
            ivp: false,
        }
    }

    /// Central finite difference of the inverse map — the ground truth
    /// `transform_deriv` is supposed to reproduce.
    fn fd_dtheta_dz(p: &EstimatedParam, z: f64) -> f64 {
        let h = 1e-6;
        (p.from_transformed(z + h) - p.from_transformed(z - h)) / (2.0 * h)
    }

    #[test]
    fn log_transform_deriv_matches_the_map_above_the_upper_bound() {
        let p = log_param(0.01, 10.0);
        let z = 10.0f64.ln() + 1.0;
        assert_eq!(p.from_transformed(z), 10.0,
            "precondition: the inverse map saturates at hi here");
        let fd = fd_dtheta_dz(&p, z);
        assert_eq!(fd, 0.0, "precondition: a saturated map is flat");
        assert!((p.transform_deriv(z) - fd).abs() < 1e-9,
            "dθ/dz must differentiate the map it belongs to: \
             transform_deriv={} vs finite difference={}",
            p.transform_deriv(z), fd);
    }

    #[test]
    fn log_transform_deriv_matches_the_map_below_the_lower_bound() {
        let p = log_param(0.01, 10.0);
        let z = 0.01f64.ln() - 1.0;
        assert_eq!(p.from_transformed(z), 0.01,
            "precondition: the inverse map saturates at lo here");
        let fd = fd_dtheta_dz(&p, z);
        assert_eq!(fd, 0.0, "precondition: a saturated map is flat");
        assert!((p.transform_deriv(z) - fd).abs() < 1e-9,
            "dθ/dz must differentiate the map it belongs to: \
             transform_deriv={} vs finite difference={}",
            p.transform_deriv(z), fd);
    }

    #[test]
    fn log_jacobian_is_neg_inf_where_the_map_saturates() {
        // dθ/dz = 0 ⇒ log|dθ/dz| = −∞: the declared bounds truncate the
        // support, so the change-of-variables density is zero outside it.
        let p = log_param(0.01, 10.0);
        for &z in &[10.0f64.ln() + 1e-3, 10.0f64.ln() + 5.0,
                    0.01f64.ln() - 1e-3, 0.01f64.ln() - 5.0] {
            assert_eq!(p.log_jacobian(z), f64::NEG_INFINITY,
                "z={z} is outside [ln lo, ln hi]; log-Jacobian must be −∞");
        }
    }

    #[test]
    fn jacobian_grad_is_zero_where_the_map_saturates() {
        let p = log_param(0.01, 10.0);
        for &z in &[10.0f64.ln() + 1e-3, 0.01f64.ln() - 1e-3] {
            assert_eq!(p.jacobian_grad(z), 0.0,
                "z={z} is outside the support; the z-gradient must be flat, \
                 not the unclamped +1");
        }
    }

    #[test]
    fn log_transform_derivatives_are_exact_in_the_interior() {
        // The whole interior is unchanged — every fit whose posterior lives
        // inside the declared bounds is bit-for-bit unaffected.
        let p = log_param(0.01, 10.0);
        for &z in &[-4.0, -1.0, 0.0, 1.0, 2.0] {
            assert_eq!(p.transform_deriv(z), z.exp());
            assert_eq!(p.log_jacobian(z), z);
            assert_eq!(p.jacobian_grad(z), 1.0);
            let fd = fd_dtheta_dz(&p, z);
            assert!(((p.transform_deriv(z) - fd) / fd).abs() < 1e-6,
                "interior dθ/dz must match the finite difference");
        }
    }

    #[test]
    fn unbounded_log_params_are_never_saturated() {
        // lo = 0 / hi = +∞ is the no-bounds default: `clamp` cannot fire on
        // either side, so neither may the saturation test.
        let p = log_param(0.0, f64::INFINITY);
        for &z in &[-700.0, -1.0, 0.0, 1.0, 700.0] {
            assert_eq!(p.log_jacobian(z), z);
            assert_eq!(p.jacobian_grad(z), 1.0);
            assert_eq!(p.transform_deriv(z), z.exp());
        }
    }

    #[test]
    fn a_value_at_the_declared_bound_stays_in_support() {
        // `to_transformed` clamps into [lo, hi] and takes ln, so every z the
        // sampler can be INITIALISED at must carry a finite log-Jacobian —
        // otherwise `run_ode_nuts`'s "posterior is not finite at the initial
        // parameters" probe would reject a legal start.
        //
        // This is why the saturation test is in z-space (z ∉ [ln lo, ln hi])
        // rather than θ-space (exp(z) ∉ [lo, hi]): `exp(ln(hi))` rounds one
        // ulp ABOVE hi for hi ∈ {10, 100, 0.1, 1e-3, 3, …}, so a θ-space test
        // would call z = ln(hi) out-of-support.
        for &(lo, hi) in &[(0.01, 10.0), (1e-3, 100.0), (1e-6, 0.1),
                           (0.05, 5.0), (0.1, 3.0), (1.0, 2.0)] {
            let p = log_param(lo, hi);
            for &x in &[lo, hi, hi * 10.0, lo / 10.0, (lo * hi).sqrt()] {
                let z = p.to_transformed(x);
                assert!(p.log_jacobian(z).is_finite(),
                    "bounds [{lo}, {hi}], x={x} → z={z}: a round-tripped \
                     value must stay in support (log-Jacobian was −∞)");
                assert!(p.transform_deriv(z) > 0.0,
                    "bounds [{lo}, {hi}], x={x}: dθ/dz must be positive in support");
                assert_eq!(p.jacobian_grad(z), 1.0);
            }
        }
    }

    #[test]
    fn saturated_z_gives_a_clean_neg_inf_target_not_nan() {
        // What the samplers actually assemble: ll + prior + log-Jacobian,
        // with the gradient chained through `transform_deriv`. A log_normal
        // prior is the sharp case — `Density::TransformedNormal` returns the
        // natural-scale density (it pre-subtracts z) and relies on the caller
        // adding `log_jacobian(z)` back, so the two must not cancel into NaN.
        use crate::inference::pgas::prior_log_density_and_grad_z;
        use crate::inference::prior::{Density, Prior};

        let p = log_param(0.01, 10.0);
        let z = 10.0f64.ln() + 2.0;
        let theta = p.from_transformed(z);
        let prior = Prior::Fixed(Density::TransformedNormal { mean: 0.0, sd: 1.0 });
        let (prior_val, prior_grad_z) = prior_log_density_and_grad_z(&prior, &p, theta, z);

        let ll = -12.5_f64;
        let log_p = ll + prior_val + p.log_jacobian(z);
        let grad_z = prior_grad_z + p.jacobian_grad(z);

        assert!(!log_p.is_nan(), "the out-of-support target must not be NaN");
        assert_eq!(log_p, f64::NEG_INFINITY,
            "outside the declared bounds the target density is zero");
        assert!(grad_z.is_finite(),
            "the gradient must stay finite so the leapfrog momentum is not poisoned");
    }

    #[test]
    fn nuts_rejects_a_leapfrog_step_into_the_saturated_region() {
        // The stability question the −∞ choice raises: does a wall at the
        // bound give a clean rejection, or a NaN/poisoned trajectory? NUTS
        // treats a non-finite energy as divergent and refuses to commit a
        // non-finite proposal, so the answer is: clean rejection, with the
        // divergence surfaced to the user.
        use crate::inference::nuts::{nuts_step, MassMatrix, NUTSConfig};
        use crate::rng::StatefulRng;

        let p = log_param(0.01, 10.0);
        // Standard normal on z, truncated to the transform's support.
        let target = |z: &[f64]| -> (f64, Vec<f64>) {
            if !p.log_jacobian(z[0]).is_finite() {
                return (f64::NEG_INFINITY, vec![0.0]);
            }
            (-0.5 * z[0] * z[0], vec![-z[0]])
        };
        let z0 = vec![10.0f64.ln() - 0.01]; // a hair inside the upper bound
        let (log_p0, grad0) = target(&z0);
        assert!(log_p0.is_finite(), "precondition: the start is in support");

        let cfg = NUTSConfig {
            max_tree_depth: 6,
            step_size: 0.5, // large enough to leap over the wall
            mass_matrix: MassMatrix::Diagonal(vec![1.0]),
        };
        let mut rng = StatefulRng::new(7);
        for _ in 0..50 {
            let r = nuts_step(&z0, log_p0, &grad0, &cfg, &target, &mut rng);
            assert!(r.params[0].is_finite(), "NUTS must never return a NaN position");
            assert!(r.log_posterior.is_finite(),
                "NUTS must never commit a −∞ / NaN log-posterior");
            assert!(p.log_jacobian(r.params[0]).is_finite(),
                "NUTS accepted a position outside the transform's support: z={}",
                r.params[0]);
        }
    }
}
