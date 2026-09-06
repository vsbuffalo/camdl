//! gh#864: the ancestor weights' effective sample size, before and after the
//! `SpliceGuard` mask.
//!
//! The ancestor index is drawn from a categorical over `exp(ancestor_log_w)`.
//! How many entries carry a finite weight (`as_finite_frac`,
//! `as_admissible_frac`) is not how many the draw can reach: one dominant
//! weight makes the categorical effectively a constant, and the ancestor move
//! then cannot renew the reference's prefix however many candidates are
//! nominally admissible.
//!
//! What this file pins is the reason *both* sides of the mask are reported.
//! The guard can lower the candidate count and *raise* the ESS in the same
//! step, by removing a dominant candidate whose splice was backward-infeasible
//! — so a
//! single post-mask number cannot separate "the density concentrates the draw"
//! from "the guard concentrates it", and those have different remedies. The
//! eight-particle example below is that case, and it is the property the whole
//! instrument exists for.

use sim::inference::pgas::AncestorEssMean;

/// Log-weights from normalised weights, with `0.0` spelled as `-inf` — the
/// representation `fill_ancestor_log_weights` produces for a particle that
/// cannot host the reference's step.
fn log_w(weights: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
        .collect()
}

fn n_finite(log_weights: &[f64]) -> usize {
    log_weights.iter().filter(|w| w.is_finite()).count()
}

/// One recorded vector, read back as a mean over one step.
fn ess_of(log_weights: &[f64]) -> f64 {
    let mut m = AncestorEssMean::new();
    m.record(log_weights);
    m.mean()
}

/// The gh#864 example, in the units the issue states it in.
///
/// Pre-mask: 6 finite weights, one of them holding 90% of the categorical mass,
/// so `(Σw)²/Σw² = 1/0.8126 ≈ 1.23` — the draw has one real choice out of six
/// candidates. The guard then refuses that dominant candidate (its splice would
/// strand the reference's later flows), leaving 5 candidates whose mass is
/// spread: ESS ≈ 3.85 out of 5.
///
/// The count fell and the ESS rose more than three-fold. By the counts alone
/// the guard looks harmful; what it did was remove an infeasible candidate that
/// was monopolising the draw.
#[test]
fn the_guard_can_lower_the_count_and_raise_the_ess() {
    let pre = log_w(&[0.90, 0.04, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0]);
    // `mask_inadmissible` writes `-inf` in place; masking the dominant
    // candidate is that write, applied here by hand so the property is stated
    // over the exact vectors rather than over a fixture's incidentals.
    let mut post = pre.clone();
    post[0] = f64::NEG_INFINITY;

    assert_eq!(n_finite(&pre), 6, "the pre-mask example has six finite weights");
    assert_eq!(n_finite(&post), 5, "the guard removed exactly one candidate");

    let ess_pre = ess_of(&pre);
    let ess_post = ess_of(&post);

    // 1 / (0.9² + 0.04² + 0.02² + 0.02² + 0.01² + 0.01²) = 1/0.8126.
    assert!((ess_pre - 1.0 / 0.8126).abs() < 1e-9,
        "pre-mask ESS is 1/Σw² over the six finite weights, got {ess_pre}");
    // Renormalised the survivors are [0.4, 0.2, 0.2, 0.1, 0.1]; 1/0.26 = 3.846.
    assert!((ess_post - 1.0 / 0.26).abs() < 1e-9,
        "post-mask ESS is 1/Σw² over the five survivors, got {ess_post}");

    // The property the two-sided report exists for. Stated as the comparison
    // and not only as the two values, so a future change that collapses the
    // instrument to one number fails here rather than in a fit.
    assert!(n_finite(&post) < n_finite(&pre) && ess_post > ess_pre,
        "the count must fall ({} → {}) while the ESS rises ({ess_pre} → \
         {ess_post}) — this is the case a single post-mask number cannot \
         distinguish from a density that concentrates on its own",
        n_finite(&pre), n_finite(&post));
}

/// The ESS is taken on max-shifted weights, so the weight ranges a
/// 4,800-particle filter visits do not underflow to a plausible small number.
///
/// `ancestor_log_w` holds unnormalised log densities; on a real fit they sit
/// thousands of nats below zero. A naive `(Σw)²/Σw²` over `exp(log_w)` sends
/// every term to exactly 0.0 there (`exp(-746)` is zero in f64, and everything
/// from about -709 down is subnormal) and returns `0/0 = NaN`, or 0.0 after a
/// guard — a number that reads as a collapsed categorical when nothing
/// collapsed. Shifting by the maximum before
/// exponentiating makes the statistic invariant to the offset, which is what
/// this asserts: the same weights, moved 8,000 nats down, read the same ESS.
#[test]
fn the_ess_is_invariant_to_the_log_weight_offset() {
    let base = log_w(&[0.90, 0.04, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0]);
    let shifted: Vec<f64> = base
        .iter()
        .map(|&w| if w.is_finite() { w - 8000.0 } else { w })
        .collect();

    // The naive form, spelled out, to show what is being avoided rather than
    // asserting it in the abstract.
    let naive: f64 = {
        let sum: f64 = shifted.iter().map(|&w| w.exp()).sum();
        let sum2: f64 = shifted.iter().map(|&w| (2.0 * w).exp()).sum();
        (sum * sum) / sum2
    };
    assert!(!naive.is_finite() || naive == 0.0,
        "the unshifted form must be the thing that fails on this input, or the \
         test is not exercising the shift; got {naive}");

    let ess = ess_of(&shifted);
    assert!((ess - ess_of(&base)).abs() < 1e-9,
        "a common offset must not move the ESS: {ess} at −8000 nats against {} \
         at 0", ess_of(&base));
}

/// A step with nothing selectable measured nothing; it is not an ESS of zero.
///
/// The same rule `CSMCDiagnostics::as_accept_rate` applies to a rate over zero
/// proposals, and the reason it matters here is that those steps are the most
/// diagnostic ones: folding them in as zeros would report a concentrated
/// categorical on the sweeps where there was no categorical at all.
#[test]
fn a_step_with_nothing_selectable_is_no_data_not_zero() {
    let dead = vec![f64::NEG_INFINITY; 8];
    let mut m = AncestorEssMean::new();
    m.record(&dead);
    assert!(m.mean().is_nan(),
        "a fully masked step must leave the mean unmeasured, got {}", m.mean());
    assert_eq!(m.n_steps(), 0, "it contributes no denominator either");

    // And it must not dilute a later step that did measure something.
    let live = log_w(&[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]);
    m.record(&live);
    assert!((m.mean() - 4.0).abs() < 1e-12,
        "the mean is over the steps that had something to measure — four equal \
         weights read 4.0, not the 2.0 a folded-in zero would give; got {}",
        m.mean());
    assert_eq!(m.n_steps(), 1);
}

/// One admissible candidate is a measurement of 1, not "no data".
///
/// The distinction matters at the boundary the starvation counter uses:
/// `n_as_starved` counts steps with at most one survivor, and a step with
/// exactly one is a categorical the draw can still resolve — it resolves to the
/// same particle every time, which is exactly what an ESS of 1 says.
#[test]
fn one_surviving_candidate_reads_one() {
    let single = log_w(&[0.0, 0.0, 1.0, 0.0]);
    let mut m = AncestorEssMean::new();
    m.record(&single);
    assert!((m.mean() - 1.0).abs() < 1e-12,
        "a single surviving weight is an ESS of 1, got {}", m.mean());
    assert_eq!(m.n_steps(), 1, "and it is a measured step");
}

/// The sweep-level number is a mean over the steps, so a sweep that is starved
/// half the time reads between the two.
#[test]
fn the_sweep_value_is_the_mean_over_measured_steps() {
    let mut m = AncestorEssMean::new();
    m.record(&log_w(&[0.25, 0.25, 0.25, 0.25]));            // ESS 4
    m.record(&log_w(&[1.0, 0.0, 0.0, 0.0]));                 // ESS 1
    m.record(&vec![f64::NEG_INFINITY; 4]);                   // no data
    assert_eq!(m.n_steps(), 2);
    assert!((m.mean() - 2.5).abs() < 1e-12,
        "mean of 4 and 1 over two measured steps is 2.5, got {}", m.mean());
}
