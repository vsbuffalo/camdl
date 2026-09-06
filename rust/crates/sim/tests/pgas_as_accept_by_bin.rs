//! gh#864: the ancestor-sampling acceptance rate resolved by position in the
//! trajectory.
//!
//! `CSMCDiagnostics::as_accept_rate` is one number per sweep, averaged over
//! every substep at which the Eq.-(21) Metropolis step ran. The phenomenon it
//! is quoted for is positional: on the runs this instrument was asked for,
//! path renewal reads ≈0.029 in the earliest tenth of the trajectory and 0.83
//! in the last. The mechanistic claim under test is that grafting the
//! reference's prefix onto another particle gets harder the further back it
//! happens, because more subsequent recorded history has to stay plausible
//! under the spliced ancestor.
//!
//! A sweep-level mean averages across exactly that gradient, so it cannot
//! separate "the claim is right" (a profile falling toward `b0`) from "the
//! claim is wrong" (a flat profile — a proposal mismatched to the target
//! uniformly in time). Those have different remedies.
//!
//! What this file pins is the accumulator's contract: the bins are the same
//! ten `renewal_by_bin` uses, indexed the same way, so the two rows describe
//! the same substeps; and a bin where no move was ever proposed is unmeasured,
//! not an acceptance rate of zero. That second rule is the whole instrument —
//! the bins where the move is never offered are precisely the ones the
//! positional reading is about, and zeroing them would manufacture the falling
//! profile the claim predicts.

use sim::inference::pgas::{PositionBins, RENEWAL_BINS};

/// The bin a substep falls in, recomputed here from the stated rule rather than
/// read off the implementation, so the alignment assertions are not circular.
fn expected_bin(s: usize, n_substeps: usize) -> usize {
    (s * RENEWAL_BINS / n_substeps).min(RENEWAL_BINS - 1)
}

/// The bins holding a measurement, in order.
fn measured(profile: &[f64; RENEWAL_BINS]) -> Vec<usize> {
    profile.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(b, _)| b).collect()
}

/// A bin the Metropolis step never ran in has no data. It is not an acceptance
/// rate of zero, and the difference is the reading the profile exists for:
/// "the ancestor move was never offered here" and "it was offered here and
/// always refused" point at different remedies — the first at the screen and
/// the density that leave nothing to propose, the second at the acceptance
/// ratio.
#[test]
fn a_bin_with_no_proposal_is_unmeasured_not_zero() {
    const N: usize = 100;
    let mut bins = PositionBins::new(N);
    // One proposal, in the last tenth, refused.
    bins.record(N - 1, false);
    let profile = bins.finish();

    assert_eq!(profile[RENEWAL_BINS - 1], 0.0,
        "the bin that was proposed in and refused reads a measured 0.0; \
         profile {profile:?}");
    assert_eq!(measured(&profile), vec![RENEWAL_BINS - 1],
        "and it is the *only* measured bin — the other nine were never proposed \
         in, so they carry no rate at all; profile {profile:?}");
    for b in 0..RENEWAL_BINS - 1 {
        assert!(profile[b].is_nan(),
            "bin {b} had no proposal and must read NaN, not {}; a zero there \
             would report a move that was offered and always refused, which is \
             the opposite diagnosis; profile {profile:?}", profile[b]);
    }
}

/// The distinction stated the other way round, so neither direction can be
/// satisfied by a constant: a measured zero must survive as a zero.
#[test]
fn a_bin_proposed_in_and_never_accepted_reads_zero() {
    const N: usize = 10;
    let mut bins = PositionBins::new(N);
    for s in 0..N {
        bins.record(s, false);
    }
    let profile = bins.finish();
    assert_eq!(measured(&profile).len(), RENEWAL_BINS,
        "every bin was proposed in; profile {profile:?}");
    for (b, &v) in profile.iter().enumerate() {
        assert_eq!(v, 0.0, "bin {b} refused every proposal it saw, got {v}");
    }
}

/// The acceptance bins and the renewal bins are the same bins.
///
/// Both go through [`PositionBins`], which is what makes that true today; this
/// asserts it against the stated rule so a later change that gives one of them
/// its own indexing fails here rather than in a fit, where the two rows would
/// silently describe different substeps and could not be read as a pair.
#[test]
fn a_substep_lands_in_the_same_bin_for_acceptance_as_for_renewal() {
    for n in [1usize, 7, 10, 11, 40, 103, 400, 4001] {
        for s in [0, 1, n / 3, n / 2, n - 1] {
            if s >= n { continue; }
            let mut accept = PositionBins::new(n);
            accept.record(s, true);
            let mut renewal = PositionBins::new(n);
            renewal.record(s, true);
            let (a, r) = (accept.finish(), renewal.finish());
            assert_eq!(measured(&a), measured(&r),
                "n={n}, substep {s}: acceptance landed in bin(s) {:?} and \
                 renewal in {:?}", measured(&a), measured(&r));
            assert_eq!(measured(&a), vec![expected_bin(s, n)],
                "n={n}, substep {s}: the rule puts it in bin {}, the \
                 accumulator in {:?}", expected_bin(s, n), measured(&a));
        }
    }
}

/// A falling profile and a flat profile with the *same* sweep-level rate — the
/// oracle for why the scalar cannot answer the question it is quoted for.
///
/// Both patterns accept 20 of 40 proposals, so `as_accept_rate` scores them
/// identically at 0.50. Resolved by position they are not the same measurement
/// at all: one accepts nothing in the first half of the series and everything
/// in the second, the other accepts half of them everywhere.
#[test]
fn the_sweep_rate_cannot_separate_a_falling_profile_from_a_flat_one() {
    const N: usize = 40;
    let mut falling = PositionBins::new(N);
    let mut flat = PositionBins::new(N);
    let (mut n_falling, mut n_flat) = (0usize, 0usize);
    for s in 0..N {
        // Falling toward b0 in the limit: nothing accepted before the midpoint,
        // everything after it.
        let accept_falling = s >= N / 2;
        // 0.50 everywhere, spread evenly.
        let accept_flat = s % 2 == 0;
        falling.record(s, accept_falling);
        flat.record(s, accept_flat);
        n_falling += usize::from(accept_falling);
        n_flat += usize::from(accept_flat);
    }
    assert_eq!((n_falling, n_flat), (20, 20),
        "the fixture is only an oracle while both patterns carry the same \
         sweep-level acceptance rate");

    let (f, l) = (falling.finish(), flat.finish());
    for b in 0..RENEWAL_BINS / 2 {
        assert_eq!(f[b], 0.0, "the falling profile accepts nothing in bin {b}");
    }
    for b in RENEWAL_BINS / 2..RENEWAL_BINS {
        assert_eq!(f[b], 1.0, "and everything in bin {b}");
    }
    for (b, &v) in l.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-12,
            "the flat profile sits at the sweep rate in every bin; bin {b} \
             read {v}");
    }
    // The separation the scalar throws away, stated as the comparison.
    assert!(f[0] < l[0] && f[RENEWAL_BINS - 1] > l[RENEWAL_BINS - 1],
        "the two profiles must diverge at both ends while scoring the same \
         scalar: falling {f:?}, flat {l:?}");
}

/// The pooled rate is a weighted mean of the bin rates, so it lies between the
/// smallest and the largest of them. The tie between the profile and the scalar
/// it resolves — the same relation `bins_tile_the_series_and_reproduce_the_
/// aggregate` pins for renewal, in the form available without per-bin counts.
#[test]
fn the_pooled_rate_lies_within_the_range_of_the_measured_bins() {
    const N: usize = 137;
    let mut bins = PositionBins::new(N);
    let (mut proposed, mut accepted) = (0usize, 0usize);
    for s in 0..N {
        // Propose at two thirds of the substeps, with an acceptance pattern
        // that varies along the series so the range is not degenerate.
        if s % 3 == 2 { continue; }
        let accept = (s * s) % 11 < s / 20;
        bins.record(s, accept);
        proposed += 1;
        accepted += usize::from(accept);
    }
    let profile = bins.finish();
    let pooled = accepted as f64 / proposed as f64;
    let lo = profile.iter().filter(|v| v.is_finite()).cloned().fold(f64::INFINITY, f64::min);
    let hi = profile.iter().filter(|v| v.is_finite()).cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo > 0.1,
        "the fixture is only informative while the profile actually varies: \
         {profile:?}");
    assert!(lo <= pooled && pooled <= hi,
        "the sweep rate {pooled} must lie inside the bins it averages \
         [{lo}, {hi}]; profile {profile:?}");
}
