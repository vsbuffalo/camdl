//! gh#864: the distribution of the ancestor-sampling acceptance ratio.
//!
//! `log α = log s_prop − log s_ref` is what the Eq.-(21) Metropolis step
//! computes and the coin then consumes. The accept/reject outcome is already
//! counted; the ratio behind it answers a question the outcome cannot, and one
//! the ancestor-weight ESS cannot either — the ESS says how the *proposal*
//! chooses among candidates, this says why the *accept test* refuses the one it
//! chose.
//!
//! Clustered far below zero, the reference's remaining history is hopeless
//! under any other ancestor and no better-informed proposal helps, because the
//! candidates it would pick are refused for the same reason. Spread, with real
//! mass near parity, the move is close to working and a proposal that preferred
//! those candidates would land it. The acceptance rate is the same small number
//! in both, and the remedies differ by weeks of work.
//!
//! What this file pins is why the pair is a median *and* a fraction near parity,
//! and never a mean: on a log ratio with a heavy left tail the mean is a
//! summary of the tail, not of the distribution, and it reports "hopeless" of
//! samples that are mostly at parity.

use sim::inference::pgas::{AcceptRatioTally, LOG_ALPHA_NEAR};

fn tally(log_alphas: &[f64]) -> AcceptRatioTally {
    let mut t = AcceptRatioTally::new();
    for &a in log_alphas {
        t.record(a);
    }
    t
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// The reason the mean is not reported, on a sample where the two disagree
/// about the answer rather than about a digit.
///
/// Seven proposals within half a nat of parity and three at −1000: the coin
/// accepts the seven better than half the time, so this is a move that is
/// landing and a proposal that could land it more often. The mean reads −300 —
/// "hopeless under any other ancestor", the opposite conclusion, and one that
/// sends the reader to restructure the update instead of the proposal.
#[test]
fn the_mean_reports_the_opposite_of_what_the_sample_says() {
    let sample = [-0.5, -0.4, -0.3, -0.5, -0.2, -0.5, -0.4, -1000.0, -1000.0, -1000.0];
    let t = tally(&sample);

    let m = mean(&sample);
    assert!(m < -100.0,
        "fixture premise: the mean must read as hopeless, got {m}");
    // Sorted: [−1000, −1000, −1000, −0.5, −0.5, −0.5, −0.4, −0.4, −0.3, −0.2].
    let median = t.median();
    assert!((median - -0.5).abs() < 1e-12,
        "the median is the average of the two central values, both −0.5, \
         got {median}");
    assert!((t.near_frac() - 0.7).abs() < 1e-12,
        "and seven of the ten sit within one nat of parity, got {}",
        t.near_frac());
    assert_eq!(t.n(), 10, "over all ten proposals");

    // The two readings, stated as the contradiction they are.
    assert!(median > LOG_ALPHA_NEAR && m < LOG_ALPHA_NEAR,
        "the median puts this sample near parity ({median}) and the mean puts \
         it beyond hope ({m}) — this is why the mean is not reported, and \
         restoring one would invert the diagnosis");
}

/// And why the fraction near parity is carried *beside* the median rather than
/// left to be inferred from it.
///
/// The mirror sample: most proposals hopeless, a fifth of them near parity. The
/// median says "hopeless", which is true of the typical proposal and hides that
/// the proposal is already finding winners a fifth of the time — a proposal
/// that preferred them would land the move, and the median alone reads as a
/// target problem rather than a proposal problem.
#[test]
fn the_fraction_near_parity_sees_mass_the_median_cannot() {
    let sample = [-200.0, -190.0, -210.0, -0.5, -0.2, -180.0, -220.0, -195.0,
                  -205.0, -215.0];
    let t = tally(&sample);
    let median = t.median();
    assert!(median < -100.0,
        "the typical proposal is hopeless, and the median says so: {median}");
    assert!((t.near_frac() - 0.2).abs() < 1e-12,
        "while a fifth of the mass sits within one nat of parity, which the \
         median cannot show: {}", t.near_frac());
}

/// The threshold is the one it says it is, and the comparison is strict, so a
/// silent drift in `LOG_ALPHA_NEAR` cannot pass as the same statistic.
#[test]
fn near_is_strictly_above_the_stated_threshold() {
    assert!((LOG_ALPHA_NEAR - -1.0).abs() < 1e-12,
        "the threshold is one nat below parity; changing it changes what \
         `as_logalpha_near` means and must be a deliberate edit");
    let t = tally(&[LOG_ALPHA_NEAR, LOG_ALPHA_NEAR + 1e-9, LOG_ALPHA_NEAR - 1e-9,
                    0.0]);
    assert!((t.near_frac() - 0.5).abs() < 1e-12,
        "the value at the threshold is not near it; two of these four are \
         above it, got {}", t.near_frac());
}

/// A sweep that measured no ratio has no data, not a ratio of zero — and zero
/// is parity, the best reading available, so the difference is not cosmetic.
///
/// The `as_accept_rate` convention, one level down: a sweep can propose moves
/// and still measure nothing here, if every proposal was refused for carrying
/// zero suffix density. `n()` is what tells those apart.
#[test]
fn a_sweep_with_no_finite_ratio_is_no_data_not_parity() {
    let t = AcceptRatioTally::new();
    assert!(t.median().is_nan(),
        "no proposal measured, so no median; got {}", t.median());
    assert!(t.near_frac().is_nan(),
        "and no fraction either; got {}", t.near_frac());
    assert_eq!(t.n(), 0, "and it says the sample is empty");
}

/// The median convention, worked on both parities so an off-by-one in the
/// central index cannot pass.
#[test]
fn the_median_is_the_central_order_statistic() {
    // Odd: the single centre, and deliberately not the mean (−4.0).
    let odd = tally(&[-1.0, -2.0, -9.0]);
    assert!((odd.median() - -2.0).abs() < 1e-12,
        "median of [−9, −2, −1] is −2, got {}", odd.median());
    // Even: the average of the two central values, and again not the mean.
    let even = tally(&[-1.0, -2.0, -4.0, -100.0]);
    assert!((even.median() - -3.0).abs() < 1e-12,
        "median of [−100, −4, −2, −1] is (−4 + −2)/2 = −3, got {}",
        even.median());
    // Order of arrival must not matter.
    let shuffled = tally(&[-100.0, -2.0, -1.0, -4.0]);
    assert_eq!(shuffled.median(), even.median(),
        "the median is over the sample, not over the order it arrived in");
    // A single proposal is a measurement.
    assert!((tally(&[-7.5]).median() - -7.5).abs() < 1e-12);
}

/// A non-finite ratio never reaches the sample, so the median's sort cannot be
/// handed an undefined comparison.
///
/// The accept test's branches make the ordinary case finite, but `NaN` is
/// representable as the difference of two finite densities an upstream defect
/// made nonsensical, and a diagnostic must not be able to abort a fit. The
/// sample size is where such a proposal shows up: it stops accounting for every
/// proposal made.
#[test]
fn a_non_finite_ratio_cannot_enter_the_sample() {
    let mut t = AcceptRatioTally::new();
    t.record(-2.0);
    t.record(f64::NAN);
    t.record(f64::NEG_INFINITY);
    t.record(f64::INFINITY);
    t.record(-4.0);
    assert_eq!(t.n(), 2, "only the two finite ratios are in the sample");
    assert!((t.median() - -3.0).abs() < 1e-12,
        "and the median is over those two, got {}", t.median());
    assert!(t.median().is_finite() && t.near_frac().is_finite(),
        "neither statistic may come back non-finite from a sample that had \
         finite values in it");
}

/// A ratio at or above parity is a proposal the step accepts outright, and it
/// is part of the distribution like any other — the statistic is not clamped to
/// the rejections.
#[test]
fn ratios_at_or_above_parity_are_in_the_sample() {
    let t = tally(&[2.0, 0.0, -0.5, -30.0]);
    assert_eq!(t.n(), 4);
    assert!((t.near_frac() - 0.75).abs() < 1e-12,
        "three of the four are within one nat of parity, got {}", t.near_frac());
    assert!((t.median() - -0.25).abs() < 1e-12,
        "median of [−30, −0.5, 0, 2] is (−0.5 + 0)/2, got {}", t.median());
}
