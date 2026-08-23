//! Systematic resampling for the particle filter.
//!
//! Lower variance than multinomial resampling, O(N), standard choice
//! for bootstrap filters (Carpenter, Clifford & Fearnhead 1999).

use crate::rng::StatefulRng;
use super::types::normalize_log_weights;

/// Systematic resampling. Returns indices of selected particles.
///
/// `log_weights` are unnormalized log-weights. Internally normalizes
/// via log-sum-exp to avoid overflow.
///
/// Algorithm: one uniform draw U ~ [0, 1/N), then select particle j
/// whenever the cumulative weight crosses U + i/N for i = 0..N-1.
/// This gives exactly N selected particles with probability proportional
/// to exp(log_weight).
pub fn systematic_resample(log_weights: &[f64], rng: &mut StatefulRng) -> Vec<usize> {
    if log_weights.is_empty() { return vec![]; }
    let weights = normalize_log_weights(log_weights);
    systematic_resample_core(&weights, rng.uniform())
}

/// Core systematic-resampling loop over **already-normalized** `weights`, given
/// a base uniform draw `u0 ∈ [0, 1)`: one uniform, evenly-spaced thresholds,
/// returning exactly `weights.len()` selected indices.
///
/// Shared by the rng-driven [`systematic_resample`] and the correlated PF's
/// `systematic_resample_fixed_u` (which supplies a *fixed* `u0` for
/// common-random-numbers coupling across paired runs) so the two selection
/// loops can never drift apart.
pub(crate) fn systematic_resample_core(weights: &[f64], u0: f64) -> Vec<usize> {
    let n = weights.len();
    if n == 0 { return vec![]; }
    let u = u0 / n as f64;
    let mut indices = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    let mut j = 0;
    for i in 0..n {
        let threshold = u + i as f64 / n as f64;
        while j < n - 1 && cumsum + weights[j] < threshold {
            cumsum += weights[j];
            j += 1;
        }
        indices.push(j);
    }
    indices
}

/// Ancestor indices for one step of a **conditional** SMC sweep — the
/// resampling inside particle Gibbs, where slot `reference` holds the reference
/// trajectory and descends from itself.
///
/// Returns one ancestor per slot, with `out[reference] == reference`. Each of
/// the other `n - 1` slots draws **independently** from `categorical(W)` over
/// all `n` particles — the reference included, so the reference's history can
/// be inherited by the free particles exactly as often as its weight warrants.
///
/// # Why multinomial here, when the particle filter uses systematic
///
/// Systematic resampling wastes fewer particles and is the right default for
/// the unconditional filters (`particle_filter`, `if2`), which fill every slot
/// from every pick. It is the wrong tool inside particle Gibbs, for two
/// separate reasons that cost four months of wrong posteriors (gh#718):
///
/// - **Its picks are not separable.** A conditional sweep fills only `n - 1`
///   slots. Taking an `n`-pick systematic draw and dropping the entry at
///   `reference` does not yield the conditional law: the picks are locked to an
///   ordered grid, so dropping one drops a *specific* stratum — the last, which
///   is exactly where the reference lies. Measured at `n = 5` with
///   `w_ref = 0.1`, the reference drew **zero** descendants against the 0.1 it
///   was owed, and the deficit does not shrink with `n`.
/// - **Ancestor sampling needs slot independence.** The accept/reject ratio for
///   an ancestor move is derived assuming the ancestry factorises across slots
///   (Lindsten, Jordan & Schön 2014). Under a scheme whose picks are dependent,
///   the resampling law does not cancel out of that ratio and the move is not a
///   valid Metropolis step. Multinomial makes the cancellation exact.
///
/// Adapting systematic resampling to the conditional case is possible — Chopin
/// & Singh (2015), *On particle Gibbs sampling*, Bernoulli 21(3):1855–1883,
/// give the construction as their Algorithm 4 — but it does not by itself
/// license the ancestor-sampling move above, so it buys nothing here.
///
/// The caller must not perform an ancestor-sampling move on a substep where
/// this function was not called: see the `did_resample` gate in `csmc_as`.
pub fn conditional_multinomial_resample(
    log_weights: &[f64],
    reference: usize,
    rng: &mut StatefulRng,
) -> Vec<usize> {
    let n = log_weights.len();
    assert!(reference < n, "reference slot {reference} out of range for {n} particles");

    // One normalisation and one cumulative pass for the whole draw; each free
    // slot is then a single uniform plus a binary search. `normalize_log_weights`
    // already falls back to uniform on a degenerate weight vector, so the CDF
    // below is always a valid distribution.
    let w = normalize_log_weights(log_weights);
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &x in &w {
        acc += x;
        cdf.push(acc);
    }
    // Pin the tail so floating-point drift cannot leave `u` past the last edge.
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut out = vec![reference; n];
    for slot in 0..n {
        if slot == reference {
            continue;
        }
        let u = rng.uniform();
        out[slot] = cdf.partition_point(|&c| c < u).min(n - 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systematic_resample_uniform_weights() {
        let mut rng = StatefulRng::new(42);
        let log_weights = vec![0.0; 100]; // all equal
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 100);
        // With uniform weights, each particle should be selected exactly once
        let mut counts = vec![0usize; 100];
        for &i in &indices { counts[i] += 1; }
        for &c in &counts {
            assert_eq!(c, 1, "uniform weights should give exactly 1 copy per particle");
        }
    }

    #[test]
    fn test_systematic_resample_degenerate() {
        let mut rng = StatefulRng::new(42);
        // One particle has all the weight
        let mut log_weights = vec![f64::NEG_INFINITY; 10];
        log_weights[3] = 0.0;
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 10);
        // All should select particle 3
        for &i in &indices {
            assert_eq!(i, 3, "degenerate weights should select only particle 3");
        }
    }

    #[test]
    fn test_systematic_resample_proportional() {
        let mut rng = StatefulRng::new(42);
        // Particle 0 has 3x the weight of particle 1
        let log_weights = vec![3.0_f64.ln(), 0.0, 0.0, 0.0];
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 4);
        let count_0 = indices.iter().filter(|&&i| i == 0).count();
        // Particle 0 should get ~2 copies (3/6 * 4 = 2)
        assert!((1..=3).contains(&count_0),
            "particle 0 (weight 3/6) got {} copies out of 4", count_0);
    }

    /// The property gh#718 defect 1 destroyed, stated as directly as it can be:
    /// a free slot picks particle `m` with probability `W_m`, and that includes
    /// the reference particle.
    ///
    /// The previous scheme filled the `n-1` free slots from an `n`-stratum
    /// `systematic_resample` and discarded the entry at the reference slot,
    /// which threw away the last stratum. The reference — laid down last, so
    /// occupying the far right of the resampling line — drew ZERO descendants
    /// here instead of `w_ref`, and every other particle was inflated by
    /// `n/(n-1)`. This test fails catastrophically against that code.
    #[test]
    fn free_slot_ancestors_are_the_weight_vector() {
        const N: usize = 5;
        const REF: usize = N - 1;
        const DRAWS: usize = 400_000;
        // Reference deliberately LIGHT (`w_ref < 1/n`), the case the old scheme
        // excluded outright.
        let w = [0.30_f64, 0.25, 0.20, 0.15, 0.10];
        let log_weights: Vec<f64> = w.iter().map(|x| x.ln()).collect();

        let mut rng = StatefulRng::new(20260823);
        let mut counts = [[0u64; N]; N]; // [slot][ancestor]
        for _ in 0..DRAWS {
            let a = conditional_multinomial_resample(&log_weights, REF, &mut rng);
            assert_eq!(a[REF], REF, "the reference slot must descend from itself");
            for (slot, &anc) in a.iter().enumerate() {
                counts[slot][anc] += 1;
            }
        }

        let mut worst = (0.0_f64, 0usize, 0usize);
        for slot in 0..N {
            if slot == REF {
                continue;
            }
            for m in 0..N {
                let z = (counts[slot][m] as f64 / DRAWS as f64 - w[m])
                    / (w[m] * (1.0 - w[m]) / DRAWS as f64).sqrt();
                if z.abs() > worst.0 {
                    worst = (z.abs(), slot, m);
                }
            }
        }
        assert!(
            worst.0 < 5.0,
            "free-slot ancestor law is not the weight vector: worst |z|={:.2} at slot {} \
             ancestor {} (want {:.4}, got {:.4}). Particle Gibbs is not invariant under a \
             resampling scheme that is not marginally unbiased (gh#718).",
            worst.0, worst.1, worst.2, w[worst.2],
            counts[worst.1][worst.2] as f64 / DRAWS as f64
        );
        // Non-vacuity: the reference must actually be reachable from a free slot,
        // or this test would pass on a scheme that merely never selects it.
        let ref_picks: u64 = (0..N).filter(|&s| s != REF).map(|s| counts[s][REF]).sum();
        assert!(
            ref_picks > 0,
            "no free slot ever selected the reference — the discriminating case is absent"
        );
    }

    /// The conditional draw must be the unconditional scheme's law conditioned
    /// on the reference keeping itself. Oracle: rejection sampling from the
    /// unconditional multinomial draw, which shares no code with the
    /// conditional path's slot skipping.
    #[test]
    fn conditional_multinomial_matches_rejection_sampling() {
        const N: usize = 4;
        const REF: usize = N - 1;
        let w = [0.15_f64, 0.20, 0.25, 0.40];
        let log_weights: Vec<f64> = w.iter().map(|x| x.ln()).collect();
        let target = 200_000usize;

        let tally = |draws: &[Vec<usize>]| {
            let mut m = std::collections::HashMap::<Vec<usize>, u64>::new();
            for d in draws {
                *m.entry(d.clone()).or_insert(0) += 1;
            }
            m
        };
        let draw_one = |rng: &mut StatefulRng| -> usize {
            let u = rng.uniform();
            let mut acc = 0.0;
            for (i, &p) in w.iter().enumerate() {
                acc += p;
                if u < acc {
                    return i;
                }
            }
            N - 1
        };

        let mut rng = StatefulRng::new(11);
        let mut accepted: Vec<Vec<usize>> = Vec::new();
        let mut proposed = 0u64;
        while accepted.len() < target {
            proposed += 1;
            let a: Vec<usize> = (0..N).map(|_| draw_one(&mut rng)).collect();
            if a[REF] == REF {
                accepted.push(a);
            }
        }
        let acc_rate = accepted.len() as f64 / proposed as f64;
        assert!(
            (acc_rate - w[REF]).abs() < 0.01,
            "rejection oracle accepted {acc_rate:.4}, expected w_ref={:.4} — the oracle is \
             not the distribution it claims to be",
            w[REF]
        );

        let mut rng2 = StatefulRng::new(29);
        let direct: Vec<Vec<usize>> = (0..accepted.len())
            .map(|_| conditional_multinomial_resample(&log_weights, REF, &mut rng2))
            .collect();

        let (oracle, got) = (tally(&accepted), tally(&direct));
        let m = accepted.len() as f64;
        let mut keys: Vec<&Vec<usize>> = oracle.keys().chain(got.keys()).collect();
        keys.sort();
        keys.dedup();
        let (mut worst, mut compared) = ((0.0_f64, Vec::new(), 0.0, 0.0), 0usize);
        for k in keys {
            let (po, pg) = (
                *oracle.get(k).unwrap_or(&0) as f64 / m,
                *got.get(k).unwrap_or(&0) as f64 / m,
            );
            if po * m < 25.0 && pg * m < 25.0 {
                continue;
            }
            compared += 1;
            let se = ((po * (1.0 - po) + pg * (1.0 - pg)) / m).sqrt();
            let z = (pg - po) / se;
            if z.abs() > worst.0 {
                worst = (z.abs(), k.clone(), po, pg);
            }
        }
        assert!(compared >= 10, "only {compared} outcomes compared — too vacuous");
        assert!(
            worst.0 < 5.0,
            "the conditional draw is not the unconditional law conditioned on the reference \
             keeping itself: outcome {:?} has oracle probability {:.5} but conditional-scheme \
             probability {:.5} (|z|={:.2})",
            worst.1, worst.2, worst.3, worst.0
        );
    }
}
