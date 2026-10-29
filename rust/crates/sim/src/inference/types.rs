//! Core types for the particle filter and inference algorithms.
//!
//! Flat array layout for cache-friendly resampling — copying a particle's
//! state is one contiguous memcpy, not a pointer chase through Vec<Vec<...>>.
//!
//! Also owns the shared inference infrastructure types — `Transform`,
//! `EstimatedParam` — that all algorithms (IF2, PGAS, PMMH) use. These
//! live here rather than in `if2.rs` so that removing or replacing IF2
//! does not take the shared type contract with it.

use super::prior::PriorPoint;
use super::traits::Resettable;

// ── Parameter transform ───────────────────────────────────────────────────────

/// The effective domain of an estimated parameter, as the map θ = g(z) from
/// an unconstrained coordinate z onto it.
///
/// Each variant carries exactly the finite ends it keeps, and g maps the
/// whole real line onto the domain, so no map clamps and no target has a
/// wall: a finite end is z = ±∞. The first four are the constraint
/// transforms of the Stan Reference Manual (v2.40, "Constraint Transforms":
/// "Lower bounded scalar", "Upper bounded scalar", "Lower and upper bounded
/// scalar"); `LogInterval` is the interval map applied to ln θ.
///
/// Built by [`Transform::for_domain`], which chooses the map from which ends
/// are finite and, for a box with 0 < lo, from whether the kind is positive
/// (support-keyed transforms proposal, §4.1 and D11).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    /// (−∞, ∞): θ = z.
    Identity,
    /// [lo, ∞): θ = lo + eᶻ.
    LowerBounded { lo: f64 },
    /// (−∞, hi]: θ = hi − eᶻ.
    UpperBounded { hi: f64 },
    /// [lo, hi]: θ = lo + (hi − lo)·σ(z).
    Interval { lo: f64, hi: f64 },
    /// [lo, hi] with 0 < lo, for a positive kind: ln θ = ln lo + ln(hi/lo)·σ(z).
    LogInterval { lo: f64, hi: f64 },
}

/// A pair of ends that is not an interval [`Transform::for_domain`] can map.
#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    /// NaN, lo ≥ hi, lo = +∞ or hi = −∞.
    NotAnInterval { lo: f64, hi: f64 },
}

impl std::fmt::Display for DomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DomainError::NotAnInterval { lo, hi } => write!(
                f, "[{lo}, {hi}] is not an interval with lo < hi, so no transform maps onto it"),
        }
    }
}

/// The fraction of an interval's width a start exactly on one of its ends is
/// moved inward by: such a start has no finite preimage (z = ±∞), so it is
/// placed at σ(z) = ε, z = ±ln(1/ε − 1) ≈ ±23.03. The same ε is the
/// lower-bounded map's absolute inward move, θ = lo + ε, z = ln ε.
pub const EDGE_START_EPS: f64 = PROB_FRACTION_EPS;

/// The logistic function σ(z) = 1/(1 + e⁻ᶻ), on the branch that does not
/// overflow.
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if z <= 0.0 { let e = z.exp(); e / (1.0 + e) } else { 1.0 / (1.0 + (-z).exp()) }
}

/// softplus(x) = ln(1 + eˣ), evaluated as max(x, 0) + ln1p(e^(−|x|)), which
/// neither overflows nor loses precision.
#[inline]
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// ln(eᵃ + eᵇ) without overflow; `a = −∞` gives `b`.
#[inline]
fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

impl Transform {
    /// The map for the domain with these ends; an infinite end is open.
    /// `positive_kind` is true for `rate`, `positive`, `count`, `duration`.
    pub fn for_domain(lo: f64, hi: f64, positive_kind: bool) -> Result<Self, DomainError> {
        if lo.is_nan() || hi.is_nan() || lo >= hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY {
            return Err(DomainError::NotAnInterval { lo, hi });
        }
        Ok(match (lo.is_finite(), hi.is_finite()) {
            (false, false) => Transform::Identity,
            (true, false) => Transform::LowerBounded { lo },
            (false, true) => Transform::UpperBounded { hi },
            (true, true) if positive_kind && lo > 0.0 => Transform::LogInterval { lo, hi },
            (true, true) => Transform::Interval { lo, hi },
        })
    }

    /// The domain's ends, (−∞, ∞) for `Identity`.
    pub fn bounds(&self) -> (f64, f64) {
        match *self {
            Transform::Identity => (f64::NEG_INFINITY, f64::INFINITY),
            Transform::LowerBounded { lo } => (lo, f64::INFINITY),
            Transform::UpperBounded { hi } => (f64::NEG_INFINITY, hi),
            Transform::Interval { lo, hi } | Transform::LogInterval { lo, hi } => (lo, hi),
        }
    }

    /// Whether start draws and Latin-hypercube points are log-spaced.
    pub fn log_spaced(&self) -> bool {
        matches!(self, Transform::LogInterval { .. })
    }

    /// The map's short name, as the partition record and the launch table
    /// print it.
    pub fn tag(&self) -> &'static str {
        match self {
            Transform::Identity => "identity",
            Transform::LowerBounded { .. } => "lower_bounded",
            Transform::UpperBounded { .. } => "upper_bounded",
            Transform::Interval { .. } => "interval",
            Transform::LogInterval { .. } => "log_interval",
        }
    }

    /// θ = g(z). Total: `±∞` maps to the corresponding end. The interval maps
    /// are evaluated from the nearer end, so the distance to it keeps full
    /// relative precision.
    pub fn inverse(&self, z: f64) -> f64 {
        match *self {
            Transform::Identity => z,
            Transform::LowerBounded { lo } => lo + z.exp(),
            Transform::UpperBounded { hi } => hi - z.exp(),
            Transform::Interval { lo, hi } => {
                let w = hi - lo;
                if z <= 0.0 { lo + w * sigmoid(z) } else { hi - w * sigmoid(-z) }
            }
            Transform::LogInterval { lo, hi } => {
                let l = (hi / lo).ln();
                if z <= 0.0 { lo * (l * sigmoid(z)).exp() } else { hi * (-l * sigmoid(-z)).exp() }
            }
        }
    }

    /// z = g⁻¹(θ). A θ exactly on a finite end has no finite preimage and is
    /// moved inward ([`EDGE_START_EPS`]); a θ outside the domain never reaches
    /// here (it is refused when values are resolved), which a debug build
    /// asserts, and is treated as on the nearer end.
    pub fn forward(&self, theta: f64) -> f64 {
        let (lo, hi) = self.bounds();
        debug_assert!(!(theta < lo || theta > hi),
            "forward map evaluated at {theta}, outside the domain [{lo}, {hi}]");
        let z_edge = (1.0 / EDGE_START_EPS - 1.0).ln();
        match *self {
            Transform::Identity => theta,
            Transform::LowerBounded { lo } => {
                let d = theta - lo;
                if d > 0.0 { d.ln() } else { EDGE_START_EPS.ln() }
            }
            Transform::UpperBounded { hi } => {
                let d = hi - theta;
                if d > 0.0 { d.ln() } else { EDGE_START_EPS.ln() }
            }
            Transform::Interval { lo, hi } => {
                if theta <= lo { return -z_edge; }
                if theta >= hi { return z_edge; }
                (theta - lo).ln() - (hi - theta).ln()
            }
            Transform::LogInterval { lo, hi } => {
                if theta <= lo { return -z_edge; }
                if theta >= hi { return z_edge; }
                // ln θ − ln lo and ln hi − ln θ, each from a difference that
                // keeps its precision near its own end.
                ((theta - lo) / lo).ln_1p().ln() - ((hi - theta) / theta).ln_1p().ln()
            }
        }
    }

    /// Whether `theta` lies on a finite end of the domain, where
    /// [`Transform::forward`] moves it inward.
    pub fn on_an_end(&self, theta: f64) -> bool {
        let (lo, hi) = self.bounds();
        (lo.is_finite() && theta <= lo) || (hi.is_finite() && theta >= hi)
    }

    /// ln |g′(z)|, finite for every finite z.
    pub fn log_jacobian(&self, z: f64) -> f64 {
        match *self {
            Transform::Identity => 0.0,
            Transform::LowerBounded { .. } | Transform::UpperBounded { .. } => z,
            Transform::Interval { lo, hi } => (hi - lo).ln() - softplus(-z) - softplus(z),
            Transform::LogInterval { lo, hi } => {
                let l = (hi / lo).ln();
                lo.ln() + l * sigmoid(z) + l.ln() - softplus(-z) - softplus(z)
            }
        }
    }

    /// d/dz ln |g′(z)|.
    pub fn jacobian_grad(&self, z: f64) -> f64 {
        match *self {
            Transform::Identity => 0.0,
            Transform::LowerBounded { .. } | Transform::UpperBounded { .. } => 1.0,
            Transform::Interval { .. } => -(z / 2.0).tanh(),
            Transform::LogInterval { lo, hi } => {
                let l = (hi / lo).ln();
                l * sigmoid(z) * sigmoid(-z) - (z / 2.0).tanh()
            }
        }
    }

    /// The signed g′(z) = dθ/dz: −eᶻ for `UpperBounded`. Every caller
    /// multiplies by it in a chain rule; none takes its logarithm.
    pub fn deriv(&self, z: f64) -> f64 {
        match *self {
            Transform::Identity => 1.0,
            Transform::LowerBounded { .. } => z.exp(),
            Transform::UpperBounded { .. } => -z.exp(),
            Transform::Interval { .. } | Transform::LogInterval { .. } => self.log_jacobian(z).exp(),
        }
    }

    /// θ and the logarithms a prior needs, computed from z so that they stay
    /// exact where θ rounds to an end (proposal §6).
    pub fn prior_point(&self, z: f64) -> PriorPoint {
        let theta = self.inverse(z);
        let (lo, hi) = self.bounds();
        let log_jac = self.log_jacobian(z);
        // ln(θ − lo) and ln(hi − θ), exact in z where the map provides them.
        let ln_above_lo = match *self {
            Transform::LowerBounded { .. } => z,
            Transform::Interval { lo, hi } => (hi - lo).ln() - softplus(-z),
            Transform::LogInterval { lo, hi } => {
                lo.ln() + ((hi / lo).ln() * sigmoid(z)).exp_m1().ln()
            }
            Transform::Identity | Transform::UpperBounded { .. } => f64::NAN,
        };
        let ln_below_hi = match *self {
            Transform::UpperBounded { .. } => z,
            Transform::Interval { lo, hi } => (hi - lo).ln() - softplus(z),
            Transform::LogInterval { lo, hi } => {
                hi.ln() + (-(-(hi / lo).ln() * sigmoid(-z)).exp_m1()).ln()
            }
            Transform::Identity | Transform::LowerBounded { .. } => f64::NAN,
        };
        let (ln_theta, d_ln_theta_dz) = match *self {
            Transform::LogInterval { lo, hi } => {
                // From the nearer end, so the distance to it in ln θ keeps
                // its precision: ln lo + L·σ(z) loses it near hi.
                let l = (hi / lo).ln();
                let ln_theta = if z <= 0.0 { lo.ln() + l * sigmoid(z) } else { hi.ln() - l * sigmoid(-z) };
                (ln_theta, l * sigmoid(z) * sigmoid(-z))
            }
            _ if lo >= 0.0 => {
                let ln_theta = if lo == 0.0 { ln_above_lo } else { logaddexp(lo.ln(), ln_above_lo) };
                (ln_theta, (log_jac - ln_theta).exp())
            }
            _ => (f64::NAN, f64::NAN),
        };
        let (ln_one_minus, d_ln_one_minus_dz) = if hi <= 1.0 {
            let l1m = if hi == 1.0 { ln_below_hi } else { logaddexp((1.0 - hi).ln(), ln_below_hi) };
            // d ln(1 − θ)/dz = −g′(z)/(1 − θ); g′ > 0 on every map with a
            // finite upper end except `UpperBounded`, whose g′ is −eᶻ.
            let sign = if matches!(self, Transform::UpperBounded { .. }) { 1.0 } else { -1.0 };
            (l1m, sign * (log_jac - l1m).exp())
        } else {
            (f64::NAN, f64::NAN)
        };
        PriorPoint { theta, ln_theta, ln_one_minus, d_ln_theta_dz, d_ln_one_minus_dz }
    }
}

/// One estimated parameter: its name, position in the full model parameter
/// vector, effective domain (as its transform) and per-algorithm adaptation
/// state.
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
    /// Random walk standard deviation, natural scale, converted to z at each
    /// particle's current value by [`EstimatedParam::transformed_sd`].
    /// Shrinks by `cooling_fraction` each IF2 iteration.
    pub rw_sd: f64,
    /// The effective domain, as the map onto it; the bounds are
    /// [`EstimatedParam::bounds`].
    pub transform: Transform,
    /// Whether `rw_sd` was auto-computed from the data (for preflight reporting).
    pub rw_sd_auto: bool,
    /// If true, this parameter is perturbed only at t=0, not at every
    /// observation — the IF2 perturbation schedule for an initial-state
    /// parameter (S₀, E₀, I₀ …), whose whole effect on the trajectory is
    /// spent before the first step. Only IF2 has a perturbation schedule,
    /// so the flag is meaningless (and rejected at config load) under every
    /// other algorithm. Corresponds to pomp's `ivp()` marker in `rw.sd`.
    pub perturb_only_at_t0: bool,
}

impl EstimatedParam {
    /// The effective domain's ends.
    pub fn bounds(&self) -> (f64, f64) {
        self.transform.bounds()
    }

    /// Map a natural-scale value to the unconstrained (z) scale.
    pub fn to_transformed(&self, x: f64) -> f64 {
        self.transform.forward(x)
    }

    /// Map an unconstrained value back to the natural scale.
    pub fn from_transformed(&self, z: f64) -> f64 {
        self.transform.inverse(z)
    }

    /// log |dθ/dz| for the transform θ = g(z): the Jacobian term a sampler
    /// moving in z adds to its log target. Finite for every finite z.
    pub fn log_jacobian(&self, z: f64) -> f64 {
        self.transform.log_jacobian(z)
    }

    /// d/dz log|dθ/dz| — derivative of the log-Jacobian w.r.t. z.
    pub fn jacobian_grad(&self, z: f64) -> f64 {
        self.transform.jacobian_grad(z)
    }

    /// dθ/dz, signed — derivative of the natural-scale value with respect to
    /// z. Used in the chain rule: d(f(θ))/dz = d(f)/dθ × dθ/dz.
    pub fn transform_deriv(&self, z: f64) -> f64 {
        self.transform.deriv(z)
    }

    /// θ and the logarithms a prior needs ([`PriorPoint`]) at the coordinate z.
    pub fn prior_point(&self, z: f64) -> PriorPoint {
        self.transform.prior_point(z)
    }

    /// Delta method: convert a natural-scale `rw_sd` to z at `current_value`.
    pub fn transformed_sd(&self, natural_sd: f64, current_value: f64) -> f64 {
        natural_sd / self.transform_deriv(self.to_transformed(current_value)).abs()
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

/// Interior clamp keeping an IVP probability strictly inside (0, 1), so the
/// Binomial(N, p) draw stays finite; shared by the PGAS IVP density and
/// gradient (`pgas.rs`, `pgas_grad.rs`), which must use the same value. The
/// same ε is [`EDGE_START_EPS`], the interval maps' inward move of a start on
/// an end.
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
    // gh#747: stamped onto every particle's RNG at construction. This is why the
    // choice cannot be lost to work-stealing: it rides on the particle, not on
    // whichever rayon worker happens to pick it up.
    algo: crate::rng::BinomialAlgorithm,
) -> Vec<crate::rng::StatefulRng> {
    (0..n)
        .map(|i| crate::rng::StatefulRng::new_stream(seed, stream_offset | (i as u64))
                     .with_binomial(algo))
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
    /// UNCHANGED by multi-cadence Phase 2a — the forward/substep path and
    /// `write_final_states` (`pfilter.rs`) keep reading it exactly as before.
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
    /// `n_interval_streams` sizes `acc` (one bin per incidence stream, two per
    /// flow-ratio stream — owned by the OBS model, not the compiled model,
    /// which reports it as `n_interval_streams()`). The process model's
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

    // ── Support-keyed transforms (proposal §5, §12.1) ────────────────────

    /// Every map on every domain of §12.1: [0, ∞), [0, 1], [0, 2],
    /// [1e-6, 2.5], [1, 100], [−120, 120], [−120, −60], plus the hand-built
    /// `LowerBounded { 1e-6 }`, `UpperBounded { 30 }` and `{ 0.8 }`, and
    /// `Identity`.
    /// Each finite box is taken both as a positive kind and not, so it is
    /// mapped by `LogInterval` where 0 < lo and by `Interval` always.
    fn maps() -> Vec<(String, Transform)> {
        let inf = f64::INFINITY;
        let mut v = vec![
            ("[0, ∞)".to_string(), Transform::for_domain(0.0, inf, true).unwrap()),
            ("LowerBounded 1e-6".to_string(), Transform::LowerBounded { lo: 1e-6 }),
            ("UpperBounded 30".to_string(), Transform::UpperBounded { hi: 30.0 }),
            // hi ≤ 1, so ln(1 − θ) and its z-derivative (positive here, as g′
            // is −eᶻ) are defined and checked.
            ("UpperBounded 0.8".to_string(), Transform::UpperBounded { hi: 0.8 }),
            ("identity".to_string(), Transform::Identity),
        ];
        for (lo, hi) in [(0.0, 1.0), (0.0, 2.0), (1e-6, 2.5), (1.0, 100.0),
                         (-120.0, 120.0), (-120.0, -60.0)] {
            for positive in [false, true] {
                if let Ok(t) = Transform::for_domain(lo, hi, positive) {
                    v.push((format!("[{lo}, {hi}] {}", t.tag()), t));
                }
            }
        }
        v
    }

    fn ulp(x: f64) -> f64 {
        let x = x.abs();
        if x == 0.0 { f64::MIN_POSITIVE } else { f64::from_bits(x.to_bits() + 1) - x }
    }

    #[test]
    fn for_domain_selects_by_finite_ends_and_kind() {
        let inf = f64::INFINITY;
        assert_eq!(Transform::for_domain(-inf, inf, true), Ok(Transform::Identity));
        assert_eq!(Transform::for_domain(0.0, inf, true), Ok(Transform::LowerBounded { lo: 0.0 }));
        assert_eq!(Transform::for_domain(-inf, 3.0, false), Ok(Transform::UpperBounded { hi: 3.0 }));
        assert_eq!(Transform::for_domain(0.1, 2.0, true), Ok(Transform::LogInterval { lo: 0.1, hi: 2.0 }));
        assert_eq!(Transform::for_domain(0.1, 0.9, false), Ok(Transform::Interval { lo: 0.1, hi: 0.9 }));
        assert_eq!(Transform::for_domain(0.0, 2.0, true), Ok(Transform::Interval { lo: 0.0, hi: 2.0 }));
        assert_eq!(Transform::for_domain(-1.0, 1.0, true), Ok(Transform::Interval { lo: -1.0, hi: 1.0 }));
        for (lo, hi) in [(1.0, 1.0), (2.0, 1.0), (f64::NAN, 1.0), (inf, inf), (-inf, -inf)] {
            assert!(Transform::for_domain(lo, hi, true).is_err(), "[{lo}, {hi}] is not an interval");
        }
        assert!(Transform::LogInterval { lo: 1.0, hi: 2.0 }.log_spaced());
        assert!(!Transform::Interval { lo: 1.0, hi: 2.0 }.log_spaced());
    }

    /// g(g⁻¹(θ)) = θ for interior θ, to 4·max(1, |z|) ulp of θ. The proposal
    /// asks for 4 ulp of the distance to the nearer end; that is not
    /// attainable through a logarithm and an exponential: z carries a
    /// relative rounding error ε, which eᶻ turns into an error |z|·ε in θ's
    /// distance to its end (measured: 12 ulp at θ = 1e-4 on [0, 1], z = −9.2).
    /// Points run from the interior to within 1e-9 of the width of either end.
    #[test]
    fn theta_round_trips() {
        for (label, t) in maps() {
            let (lo, hi) = t.bounds();
            let pts: Vec<f64> = match (lo.is_finite(), hi.is_finite()) {
                (true, true) => [1e-9, 1e-4, 0.1, 0.5, 0.9, 1.0 - 1e-4, 1.0 - 1e-9]
                    .iter().map(|f| lo + f * (hi - lo)).collect(),
                (true, false) => [1e-9, 1e-3, 1.0, 1e3].iter().map(|d| lo + d).collect(),
                (false, true) => [1e-9, 1e-3, 1.0, 1e3].iter().map(|d| hi - d).collect(),
                (false, false) => vec![-1e3, -1.0, 0.0, 2.5, 1e3],
            };
            for x in pts {
                let z = t.forward(x);
                let back = t.inverse(z);
                assert!((back - x).abs() <= 4.0 * z.abs().max(1.0) * ulp(x),
                    "{label}: θ {x} → z {z} → θ {back}");
            }
        }
    }

    /// g⁻¹(g(z)) = z for |z| ≤ 30, to 1e-12 plus the error θ's own rounding
    /// induces in z (4 ulp of θ times |dz/dθ|). Near an end of a box with
    /// lo ≠ 0 that second term dominates, because θ − lo is what rounds.
    #[test]
    fn z_round_trips() {
        for (label, t) in maps() {
            for z in [-30.0, -12.0, -3.0, -0.4, 0.0, 0.7, 3.0, 12.0, 30.0] {
                let x = t.inverse(z);
                let back = t.forward(x);
                let tol = 1e-12 * z.abs().max(1.0) + 4.0 * ulp(x) / t.deriv(z).abs();
                assert!((back - z).abs() <= tol, "{label}: z {z} → θ {x} → z {back} (tol {tol:e})");
            }
        }
    }

    fn fd(f: &dyn Fn(f64) -> f64, z: f64) -> f64 {
        let h = 1e-5;
        (f(z + h) - f(z - h)) / (2.0 * h)
    }

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-300)
    }

    /// `jacobian_grad` and `prior_point`'s two derivatives against central
    /// finite differences of the quantities they differentiate, relative
    /// error below 1e-6 on |z| ≤ 30; those quantities are computed from z,
    /// so the difference is well conditioned everywhere. `transform_deriv`
    /// against the difference of θ itself, where θ is not within 1e-6 of its
    /// magnitude of an end (there θ's rounding swamps a difference of h).
    #[test]
    fn derivatives_match_finite_differences() {
        for (label, t) in maps() {
            let (lo, hi) = t.bounds();
            for z in [-30.0, -12.0, -3.0, -0.4, 0.0, 0.7, 3.0, 12.0, 30.0] {
                let jg = fd(&|z| t.log_jacobian(z), z);
                assert!((t.jacobian_grad(z) - jg).abs() <= 1e-6 * jg.abs().max(1.0),
                    "{label} z={z}: jacobian_grad {} vs fd {jg}", t.jacobian_grad(z));
                let pp = t.prior_point(z);
                if pp.ln_theta.is_finite() {
                    let d = fd(&|z| t.prior_point(z).ln_theta, z);
                    assert!((pp.d_ln_theta_dz - d).abs() <= 1e-6 * d.abs().max(1.0),
                        "{label} z={z}: d ln θ/dz {} vs fd {d}", pp.d_ln_theta_dz);
                }
                if pp.ln_one_minus.is_finite() {
                    let d = fd(&|z| t.prior_point(z).ln_one_minus, z);
                    assert!((pp.d_ln_one_minus_dz - d).abs() <= 1e-6 * d.abs().max(1.0),
                        "{label} z={z}: d ln(1−θ)/dz {} vs fd {d}", pp.d_ln_one_minus_dz);
                }
                let x = t.inverse(z);
                let gap = (x - lo).abs().min((hi - x).abs());
                if gap > 1e-6 * x.abs().max(1.0) {
                    let d = fd(&|z| t.inverse(z), z);
                    assert!(rel_close(t.deriv(z), d, 1e-6),
                        "{label} z={z}: transform_deriv {} vs fd {d}", t.deriv(z));
                }
            }
        }
    }

    /// The log-Jacobian is finite for every |z| ≤ 700, and g(z) is inside the
    /// closed domain at z ∈ {±40, ±800, ±∞}. The logit log-Jacobian before
    /// this change was ln((hi − lo)·p·(1 − p)), −∞ from z = 37 on.
    #[test]
    fn no_map_has_a_wall() {
        for (label, t) in maps() {
            let (lo, hi) = t.bounds();
            let mut z = -700.0;
            while z <= 700.0 {
                assert!(t.log_jacobian(z).is_finite(), "{label}: log_jacobian({z}) not finite");
                z += 0.5;
            }
            for z in [-40.0, 40.0, -800.0, 800.0, f64::NEG_INFINITY, f64::INFINITY] {
                let x = t.inverse(z);
                assert!(x >= lo && x <= hi, "{label}: g({z}) = {x} outside [{lo}, {hi}]");
            }
        }
    }

    /// The log-interval map's ln θ is computed from the nearer end. With
    /// hi = 1, ln hi = 0 exactly, so near hi ln θ is itself the distance to
    /// the end and must carry it to 1e-12 relative: −L·σ(−z), out to z = 30.
    /// Summed from the lower end (ln lo + L·σ(z)) it is the difference of two
    /// numbers near ±L and is off by ~1e-3 relative at z = 30. Nowhere may ln θ
    /// exceed ln hi.
    #[test]
    fn log_interval_ln_theta_is_exact_near_the_upper_end() {
        let (lo, hi) = (0.05_f64, 1.0_f64);
        let t = Transform::LogInterval { lo, hi };
        let l = (hi / lo).ln();
        for z in [5.0, 15.0, 30.0] {
            let got = t.prior_point(z).ln_theta;
            let want = -l * sigmoid(-z);
            assert!((got - want).abs() <= 1e-12 * want.abs(), "z={z}: ln θ = {got:e}, want {want:e}");
        }
        let mut z = -40.0;
        while z <= 40.0 {
            assert!(t.prior_point(z).ln_theta <= 0.0, "z={z}: ln θ above ln hi");
            z += 0.25;
        }
    }

    #[test]
    fn upper_bounded_derivative_is_negative() {
        let t = Transform::UpperBounded { hi: 30.0 };
        for z in [-5.0, 0.0, 3.0] {
            assert!(t.deriv(z) < 0.0);
            assert_eq!(t.log_jacobian(z), z, "log_jacobian is ln|g′| = z");
        }
    }

    /// A start exactly on a finite end has no finite preimage; the forward
    /// map moves it inward to z = ±ln(1/ε − 1) ≈ ±23.03 on the interval maps
    /// and to θ = lo + ε, z = ln ε, on the half-bounded ones.
    #[test]
    fn a_start_on_an_end_maps_to_a_finite_z() {
        let edge = (1.0 / EDGE_START_EPS - 1.0).ln();
        assert!((edge - 23.0258).abs() < 1e-3);
        for (label, t) in maps() {
            let (lo, hi) = t.bounds();
            if lo.is_finite() {
                assert!(t.on_an_end(lo));
                let z = t.forward(lo);
                let want = if hi.is_finite() { -edge } else { EDGE_START_EPS.ln() };
                assert_eq!(z, want, "{label}: start on lo");
            }
            if hi.is_finite() {
                assert!(t.on_an_end(hi));
                let z = t.forward(hi);
                let want = if lo.is_finite() { edge } else { EDGE_START_EPS.ln() };
                assert_eq!(z, want, "{label}: start on hi");
            }
        }
    }

    /// `log_normal` on `LowerBounded { 0 }` (θ = eᶻ) equals, bit for bit, the
    /// value and gradient the z-based formulas gave before priors read a
    /// `PriorPoint`, on z ∈ [−745, 30]. Below −745 θ underflows to 0, which
    /// the old guard scored −∞; the new value is finite and correct there.
    #[test]
    fn log_normal_on_the_half_line_is_bit_identical_to_the_z_formula() {
        use crate::inference::prior::{Density, Prior};
        const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
        let (mu, sd) = (-1.0_f64, 0.5_f64);
        let prior = Prior::Fixed(Density::TransformedNormal { mean: mu, sd });
        let t = Transform::LowerBounded { lo: 0.0 };
        let mut z = -745.0;
        while z <= 30.0 {
            let old_v = {
                let zs = (z - mu) / sd;
                -z - HALF_LN_2PI - sd.ln() - 0.5 * zs * zs
            };
            let old_g = -(z - mu) / (sd * sd) - 1.0;
            let (v, g) = prior.log_density_and_grad_z(&t.prior_point(z), t.deriv(z));
            assert_eq!(v.to_bits(), old_v.to_bits(), "value at z={z}");
            assert_eq!(g.to_bits(), old_g.to_bits(), "gradient at z={z}");
            z += 0.37;
        }
        let z = -800.0;
        assert_eq!(t.inverse(z), 0.0, "precondition: θ underflows");
        let v = prior.log_density(&t.prior_point(z));
        let zs = (z - mu) / sd;
        assert_eq!(v, -z - HALF_LN_2PI - sd.ln() - 0.5 * zs * zs, "finite closed form at z=-800");
    }
}
