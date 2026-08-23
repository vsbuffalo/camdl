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
/// resampling inside particle Gibbs, where slot `reference` is held by the
/// reference trajectory and descends from itself.
///
/// Returns one ancestor per slot, with `out[reference] == reference`.
///
/// # Why this is not [`systematic_resample`] with one slot overwritten
///
/// [`systematic_resample`] lays `n` evenly spaced thresholds across the
/// cumulative weight and returns one ancestor per slot. A conditional sweep has
/// only `n − 1` slots to fill, because the reference keeps itself. Taking the
/// unconditional output and dropping the entry at `reference` drops the LAST
/// threshold, so the free slots span only the first `(n−1+u)/n` of the
/// cumulative range: every particle in the top `1/n` of it loses about one
/// expected offspring, and the reference — last in the weight vector — loses
/// its own. On a 5-particle ensemble with `w_ref = 0.1` the reference receives
/// **zero** descendants where it is owed `4 × 0.1`, and the deficit does not
/// shrink with `n` (~0.6 of one slot at `n = 500`), because a typical particle
/// only expects about one slot to begin with.
///
/// # What correctness requires
///
/// Chopin & Singh (2015) require the *unconditional* resampling distribution to
/// be **marginally unbiased** — the law of a single ancestor `A^n` must assign
/// probability `W^m` to outcome `m` (§5) — which for systematic resampling is
/// obtained by randomly cycling the output. What a conditional sweep must then
/// draw from is that scheme's law **conditioned on the reference keeping
/// itself**; their Algorithm 4 (§5.2) does exactly that, and is implemented
/// here:
///
/// (a) draw `U` conditioned so the reference receives an offspring, with the
///     correct conditional law for how many it receives;
/// (b) run plain systematic resampling at that `U`;
/// (c) cycle the output uniformly over the reference's own copies, placing one
///     of them in the reference slot.
///
/// Note what is NOT claimed: the *conditional* draw's per-slot marginals are
/// not `W` — conditioning changes them. The tests therefore check marginal
/// unbiasedness of the unconditional scheme, and check this function against
/// rejection sampling from it.
///
/// The paper fixes the reference at index 1 and notes the frozen trajectory may
/// be relabelled freely, so this rotates the weight vector to put `reference`
/// first and rotates the result back.
///
/// Reference: Chopin, N. and Singh, S.S. (2015). "On particle Gibbs sampling."
/// *Bernoulli* **21**(3):1855–1883, DOI 10.3150/14-BEJ629; §5 and Algorithm 4
/// in §5.2 (arXiv:1304.1887 reprint repaginates).
pub fn conditional_systematic_resample(
    log_weights: &[f64],
    reference: usize,
    rng: &mut StatefulRng,
) -> Vec<usize> {
    let n = log_weights.len();
    assert!(reference < n, "reference slot {reference} out of range for {n} particles");
    if n == 1 {
        return vec![0];
    }
    let w = normalize_log_weights(log_weights);
    let nf = n as f64;
    // Rotate so the reference is entry 0; `rot` maps rotated → original.
    let rot = |i: usize| (i + reference) % n;
    let wr: Vec<f64> = (0..n).map(|i| w[rot(i)]).collect();

    // (a) The conditioned base uniform.
    let nw1 = nf * wr[0];
    let u0 = if nw1 <= 1.0 {
        rng.uniform() * nw1
    } else {
        let floor = nw1.floor();
        let frac = nw1 - floor;
        if rng.uniform() < frac * (floor + 1.0) / nw1 {
            rng.uniform() * frac
        } else {
            frac + rng.uniform() * (1.0 - frac)
        }
    };

    // (b) Plain systematic selection at that `U` — the same loop the
    // unconditional path uses, so the two can never drift apart.
    let abar = systematic_resample_core(&wr, u0);

    // (c) Cycle uniformly over the reference's own copies, placing one of them
    // in the reference slot. Uniform over copies is what makes this the
    // conditional law of the cycle-randomised scheme rather than of some
    // arbitrary tie-break.
    let copies: Vec<usize> = (0..n).filter(|&i| abar[i] == 0).collect();
    // Step (a) guarantees at least one copy; fall back rather than panic if a
    // degenerate weight vector defeats it.
    let c0 = if copies.is_empty() {
        0
    } else {
        copies[((rng.uniform() * copies.len() as f64) as usize).min(copies.len() - 1)]
    };

    let mut out = vec![0usize; n];
    for slot in 0..n {
        out[rot(slot)] = rot(abar[(c0 + slot) % n]);
    }
    debug_assert_eq!(out[reference], reference, "the reference must descend from itself");
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

    /// The unconditional scheme must be **marginally unbiased**: the law of a
    /// single ancestor is the weight vector itself. Chopin & Singh (2015) §5
    /// require this of the resampling distribution, and obtain it for
    /// systematic resampling by randomly cycling the output (§5.2).
    ///
    /// Note this is a property of the UNCONDITIONAL scheme. The conditional
    /// draw's per-slot marginals are NOT `W` — conditioning on the reference
    /// keeping itself changes them — which is why the conditional scheme is
    /// checked against rejection sampling below instead.
    #[test]
    fn randomized_systematic_is_marginally_unbiased() {
        const N: usize = 5;
        const DRAWS: usize = 400_000;
        let w = [0.30_f64, 0.25, 0.20, 0.15, 0.10];
        let mut rng = StatefulRng::new(20260823);
        let mut counts = [[0u64; N]; N]; // [slot][ancestor]
        for _ in 0..DRAWS {
            let abar = systematic_resample_core(&w, rng.uniform());
            let c0 = ((rng.uniform() * N as f64) as usize).min(N - 1);
            for slot in 0..N {
                counts[slot][abar[(c0 + slot) % N]] += 1;
            }
        }
        let mut worst = (0.0_f64, 0usize, 0usize);
        for slot in 0..N {
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
            "randomly-cycled systematic resampling is not marginally unbiased: \
             worst |z|={:.2} at slot {} ancestor {} (want {:.4}, got {:.4})",
            worst.0, worst.1, worst.2, w[worst.2],
            counts[worst.1][worst.2] as f64 / DRAWS as f64
        );
    }

    /// The conditional scheme must sample the unconditional scheme's law
    /// **conditioned on the reference keeping itself** — that is what makes the
    /// particle Gibbs kernel invariant. Oracle: rejection sampling from the
    /// unconditional randomly-cycled scheme, which is non-circular (it shares
    /// no code path with [`conditional_systematic_resample`]'s conditioning of
    /// `U` or its choice of cycle).
    ///
    /// This is the test gh#718 needed. The previous scheme filled the `n-1`
    /// free slots from an `n`-stratum [`systematic_resample`] and discarded the
    /// entry at the reference slot, which is not the conditional law of
    /// anything: it threw away the top stratum, so the reference — last in the
    /// weight vector — drew ZERO descendants where it is owed its share, and
    /// every other particle was inflated by `n/(n-1)`.
    ///
    /// Three weight vectors, because Algorithm 4 step (a) BRANCHES on
    /// `N·W_ref` and a single fixture leaves branches unexercised. With
    /// `N·W_ref` a whole number the conditioning degenerates to a plain
    /// `U ~ [0,1)`, so a fixture like that cannot tell step (a) from its
    /// absence — mutation-checked.
    fn check_conditional_matches_rejection(w: &[f64], label: &str) {
        const REF: usize = 4;
        let n = w.len();
        assert_eq!(n, 5, "cases are written for n = 5");
        let log_weights: Vec<f64> = w.iter().map(|x| x.ln()).collect();
        let target = 300_000usize;

        let tally = |draws: &[Vec<usize>]| {
            let mut m = std::collections::HashMap::<Vec<usize>, u64>::new();
            for d in draws {
                *m.entry(d.clone()).or_insert(0) += 1;
            }
            m
        };

        // Oracle: unconditional randomly-cycled systematic, keeping the draws in
        // which the reference happened to keep itself.
        let mut rng = StatefulRng::new(11);
        let mut accepted: Vec<Vec<usize>> = Vec::new();
        let mut proposed = 0u64;
        while accepted.len() < target {
            proposed += 1;
            let abar = systematic_resample_core(w, rng.uniform());
            let c0 = ((rng.uniform() * n as f64) as usize).min(n - 1);
            let a: Vec<usize> = (0..n).map(|slot| abar[(c0 + slot) % n]).collect();
            if a[REF] == REF {
                accepted.push(a);
            }
        }
        // Marginal unbiasedness says the acceptance rate is `w[REF]`; if it is
        // not, the oracle is not the distribution it claims to be.
        let acc_rate = accepted.len() as f64 / proposed as f64;
        assert!(
            (acc_rate - w[REF]).abs() < 0.01,
            "{label}: rejection oracle accepted {acc_rate:.4}, expected w_ref={:.4}",
            w[REF]
        );

        let mut rng2 = StatefulRng::new(29);
        let direct: Vec<Vec<usize>> = (0..accepted.len())
            .map(|_| conditional_systematic_resample(&log_weights, REF, &mut rng2))
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
            // Two independent samples of the same size.
            let se = ((po * (1.0 - po) + pg * (1.0 - pg)) / m).sqrt();
            let z = (pg - po) / se;
            if z.abs() > worst.0 {
                worst = (z.abs(), k.clone(), po, pg);
            }
        }
        // A light reference pins `Ā` almost completely, so this case has only a
        // couple of reachable outcomes; require a real distribution rather than
        // a particular count.
        let max_p = oracle.values().map(|&c| c as f64 / m).fold(0.0_f64, f64::max);
        assert!(compared >= 2, "{label}: only {compared} outcomes compared — too vacuous");
        assert!(
            max_p < 0.95,
            "{label}: the oracle is a near point mass (max outcome {max_p:.3}) — nothing to compare"
        );
        assert!(
            worst.0 < 5.0,
            "{label}: the conditional draw is not the unconditional law conditioned on \
             the reference keeping itself: outcome {:?} has rejection-oracle probability \
             {:.5} but conditional-scheme probability {:.5} (|z|={:.2}). Particle Gibbs is \
             not invariant under such a scheme (gh#718).",
            worst.1, worst.2, worst.3, worst.0
        );
    }

    /// `N·W_ref = 0.5 ≤ 1`: the reference is owed less than one copy, so step
    /// (a) must force `U` into the reference's own slice or it gets none.
    #[test]
    fn conditional_systematic_matches_rejection_light_reference() {
        check_conditional_matches_rejection(&[0.25, 0.25, 0.25, 0.15, 0.10], "light reference");
    }

    /// `N·W_ref = 1.75`: a fractional part, so step (a)'s two-branch draw is
    /// live — this is the case that distinguishes it from a plain `U ~ [0,1)`.
    #[test]
    fn conditional_systematic_matches_rejection_fractional_reference() {
        check_conditional_matches_rejection(&[0.15, 0.20, 0.15, 0.15, 0.35], "fractional reference");
    }

    /// `N·W_ref = 2.0`: the reference takes several copies, some of which land
    /// in FREE slots — the case where a free slot can select the reference.
    #[test]
    fn conditional_systematic_matches_rejection_heavy_reference() {
        check_conditional_matches_rejection(&[0.15, 0.15, 0.15, 0.15, 0.40], "heavy reference");
    }
}
