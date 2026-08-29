//! `poisson_quantile` in the large-λ tail (gh#772), and the gh#362 shape it
//! must not repeat.
//!
//! The correlated PF stores every draw as a standard normal and transforms it
//! at consumption time, so an `init { }` entry written `I ~ poisson(rate = I0)`
//! needs an inverse Poisson CDF. The obvious implementation walks the CDF from
//! `k = 0`, seeded at `P(X = 0) = e^{−λ}`. That seed underflows to exactly `0`
//! at `λ > 745`, after which the walk never accumulates and the function
//! returns its fallback for *every* `u` — the same failure `binomial_quantile`
//! shipped with (gh#362), where it silently over-drained a compartment on this
//! code path with no error and no test catching it. National-scale initial
//! states sit squarely in that regime.
//!
//! `naive_walk_quantile` below is that implementation. It is the reference in
//! the regime where it is correct (`λ` small enough that `e^{−λ}` is a normal
//! number) and the counter-example in the regime where it is not.
//!
//! External oracle: R's `qpois`, which is the same definition — the smallest
//! `k` with `F(k) ≥ u`. The table in `matches_qpois_across_lambda_and_u` was
//! produced by
//!
//! ```text
//! Rscript -e 'for (l in c(0.5,1,3.7,10,50,200,745.5,1000,5000,100000))
//!               for (u in c(1e-12,0.001,0.025,0.25,0.5,0.75,0.975,0.999,1-1e-12))
//!                 cat(sprintf("(%g, %g, %d),\n", l, u, qpois(u, l)))'
//! ```

use sim::inference::correlated_pf::poisson_quantile;

/// The implementation this function must NOT be: accumulate the pmf from
/// `k = 0`, seeded at `e^{−λ}`, and stop when the running CDF reaches `u`.
///
/// Correct while `e^{−λ}` is a normal number. `0` is returned when the walk
/// runs out — which is what happens for every `u` once the seed underflows,
/// and the choice of fallback is not the point: no fallback is right, because
/// the walk carries no information at all in that regime.
fn naive_walk_quantile(lambda: f64, u: f64) -> u64 {
    let mut cdf = 0.0;
    let mut pmf = (-lambda).exp(); // P(X = 0)
    for k in 0..1_000_000u64 {
        cdf += pmf;
        if cdf >= u {
            return k;
        }
        pmf *= lambda / (k + 1) as f64;
    }
    0
}

/// Exact agreement with R's `qpois` across five orders of magnitude in `λ`,
/// up to the crossover at `λ = 10000` where `poisson_quantile` inverts the
/// exact CDF. Includes `λ = 745.5`, just past the `e^{−λ}` underflow edge, and
/// runs out to `u = 1 − 1e-12` in both tails.
#[test]
fn matches_qpois_exactly_below_the_crossover() {
    let near_one = 1.0 - 1e-12;
    let cases: &[(f64, f64, u64)] = &[
        (0.5, 1e-12, 0), (0.5, 0.001, 0), (0.5, 0.025, 0), (0.5, 0.25, 0),
        (0.5, 0.5, 0), (0.5, 0.75, 1), (0.5, 0.975, 2), (0.5, 0.999, 4),
        (0.5, near_one, 11),
        (1.0, 1e-12, 0), (1.0, 0.001, 0), (1.0, 0.025, 0), (1.0, 0.25, 0),
        (1.0, 0.5, 1), (1.0, 0.75, 2), (1.0, 0.975, 3), (1.0, 0.999, 5),
        (1.0, near_one, 14),
        (3.7, 1e-12, 0), (3.7, 0.001, 0), (3.7, 0.025, 1), (3.7, 0.25, 2),
        (3.7, 0.5, 4), (3.7, 0.75, 5), (3.7, 0.975, 8), (3.7, 0.999, 11),
        (3.7, near_one, 24),
        (10.0, 1e-12, 0), (10.0, 0.001, 2), (10.0, 0.025, 4), (10.0, 0.25, 8),
        (10.0, 0.5, 10), (10.0, 0.75, 12), (10.0, 0.975, 17), (10.0, 0.999, 21),
        (10.0, near_one, 39),
        (50.0, 1e-12, 9), (50.0, 0.001, 30), (50.0, 0.025, 37), (50.0, 0.25, 45),
        (50.0, 0.5, 50), (50.0, 0.75, 55), (50.0, 0.975, 64), (50.0, 0.999, 73),
        (50.0, near_one, 107),
        (200.0, 1e-12, 109), (200.0, 0.001, 158), (200.0, 0.025, 173),
        (200.0, 0.25, 190), (200.0, 0.5, 200), (200.0, 0.75, 209),
        (200.0, 0.975, 228), (200.0, 0.999, 245), (200.0, near_one, 307),
        (745.5, 1e-12, 562), (745.5, 0.001, 663), (745.5, 0.025, 692),
        (745.5, 0.25, 727), (745.5, 0.5, 745), (745.5, 0.75, 764),
        (745.5, 0.975, 799), (745.5, 0.999, 831), (745.5, near_one, 945),
        (1000.0, 1e-12, 786), (1000.0, 0.001, 904), (1000.0, 0.025, 938),
        (1000.0, 0.25, 979), (1000.0, 0.5, 1000), (1000.0, 0.75, 1021),
        (1000.0, 0.975, 1062), (1000.0, 0.999, 1099), (1000.0, near_one, 1230),
        (5000.0, 1e-12, 4511), (5000.0, 0.001, 4783), (5000.0, 0.025, 4862),
        (5000.0, 0.25, 4952), (5000.0, 0.5, 5000), (5000.0, 0.75, 5048),
        (5000.0, 0.975, 5139), (5000.0, 0.999, 5220), (5000.0, near_one, 5505),
        (10000.0, 1e-12, 9305), (10000.0, 0.001, 9692), (10000.0, 0.025, 9804),
        (10000.0, 0.25, 9932), (10000.0, 0.5, 10000), (10000.0, 0.75, 10067),
        (10000.0, 0.975, 10196), (10000.0, 0.999, 10310),
        (10000.0, near_one, 10711),
    ];
    for &(lambda, u, want) in cases {
        let got = poisson_quantile(lambda, u);
        assert_eq!(got, want, "poisson_quantile({lambda}, {u}) = {got}, qpois = {want}");
    }
}

/// Above the crossover the exact CDF is no longer trustworthy (the incomplete
/// gamma's power series stops at 1000 terms and needs ~√λ of them near the
/// median), so the Cornish–Fisher expansion is returned instead. This measures
/// the price of that: exact over the body, at most one count at `u = 1 ± 1e-12`
/// up to `λ = 10⁶`, and never more than a hundredth of a standard deviation
/// anywhere — including `λ = 10⁹`, where the worst case is 8 counts in a
/// billion.
///
/// A quantified approximation, in other words, rather than the unquantified
/// silent collapse the naive walk gives at the same λ.
#[test]
fn matches_qpois_within_a_hundredth_of_an_sd_above_the_crossover() {
    let near_one = 1.0 - 1e-12;
    let cases: &[(f64, f64, u64)] = &[
        (100000.0, 1e-12, 97784), (100000.0, 0.001, 99024),
        (100000.0, 0.025, 99381), (100000.0, 0.25, 99787),
        (100000.0, 0.5, 100000), (100000.0, 0.75, 100213),
        (100000.0, 0.975, 100620), (100000.0, 0.999, 100979),
        (100000.0, near_one, 102232),
        (1e6, 1e-12, 992974), (1e6, 0.001, 996911), (1e6, 0.025, 998041),
        (1e6, 0.25, 999325), (1e6, 0.5, 1000000), (1e6, 0.75, 1000674),
        (1e6, 0.975, 1001960), (1e6, 0.999, 1003092), (1e6, near_one, 1007042),
        (1e9, 1e-12, 999777558), (1e9, 0.001, 999902280),
        (1e9, 0.025, 999938021), (1e9, 0.25, 999978671),
        (1e9, 0.5, 1000000000), (1e9, 0.75, 1000021329),
        (1e9, 0.975, 1000061980), (1e9, 0.999, 1000097723),
        (1e9, near_one, 1000222450),
    ];
    let mut worst_counts = 0i64;
    let mut worst_sds = 0.0f64;
    for &(lambda, u, want) in cases {
        let got = poisson_quantile(lambda, u) as i64;
        let d = (got - want as i64).abs();
        let sds = d as f64 / lambda.sqrt();
        assert!(
            sds < 0.01,
            "poisson_quantile({lambda}, {u}) = {got}, qpois = {want} — off by \
             {d} counts = {sds} sd"
        );
        worst_counts = worst_counts.max(d);
        worst_sds = worst_sds.max(sds);
    }
    // Reported so the price is read off a run rather than inferred from the
    // threshold above.
    eprintln!(
        "Cornish-Fisher branch vs qpois: worst {worst_counts} counts, {worst_sds:.2e} sd"
    );
}

/// The gh#362 shape, made explicit: past `λ = 745` the naive walk carries no
/// information and `poisson_quantile` must not be it.
///
/// At `λ = 1000`, `e^{−1000}` is exactly `0.0` in `f64`, so the walk's running
/// CDF stays at zero for every one of its million steps and it returns the same
/// count whatever `u` is. Nothing errors — that is the whole hazard.
#[test]
fn large_lambda_is_not_the_naive_walk() {
    const LAMBDA: f64 = 1000.0;
    assert_eq!((-LAMBDA).exp(), 0.0, "the premise: e^-1000 underflows to zero");

    // The counter-example: every u collapses to one value.
    let naive: Vec<u64> = [0.001, 0.25, 0.5, 0.75, 0.999]
        .iter()
        .map(|&u| naive_walk_quantile(LAMBDA, u))
        .collect();
    assert!(
        naive.iter().all(|&k| k == naive[0]),
        "premise of this test: the naive walk is degenerate at lambda = {LAMBDA}, \
         got {naive:?}"
    );

    // What the function actually does: spread around the mean, matching qpois.
    assert_eq!(poisson_quantile(LAMBDA, 0.001), 904);
    assert_eq!(poisson_quantile(LAMBDA, 0.5), 1000);
    assert_eq!(poisson_quantile(LAMBDA, 0.999), 1099);

    // Stated as the property rather than as three numbers, so a future
    // implementation that regresses to a walk fails here whatever its fallback.
    let lo = poisson_quantile(LAMBDA, 0.01);
    let hi = poisson_quantile(LAMBDA, 0.99);
    assert!(
        lo < hi && (lo as f64) < LAMBDA && (hi as f64) > LAMBDA,
        "the quantile must straddle lambda with real spread, got lo={lo}, hi={hi}"
    );
}

/// Agreement with the term-by-term walk in the regime where the walk IS
/// correct. Without this the large-λ test above could be satisfied by any
/// function that returns spread-out numbers.
#[test]
fn matches_the_exact_walk_where_the_walk_is_valid() {
    for &lambda in &[0.1f64, 0.5, 1.0, 2.5, 7.0, 20.0, 100.0, 700.0] {
        assert!((-lambda).exp() > 0.0, "the walk must be valid at lambda = {lambda}");
        for i in 1..40 {
            let u = i as f64 / 40.0;
            let got = poisson_quantile(lambda, u);
            let want = naive_walk_quantile(lambda, u);
            assert_eq!(got, want, "poisson_quantile({lambda}, {u}) = {got}, walk = {want}");
        }
    }
}

/// Non-decreasing in `u`, which is what the correlated-PF coupling needs: a
/// small perturbation of the pre-drawn normal must move the drawn count a
/// little, in one direction, rather than jump.
#[test]
fn monotone_in_u() {
    for &lambda in &[0.7, 12.0, 900.0, 250000.0] {
        let mut prev = 0u64;
        for i in 0..=200 {
            let u = i as f64 / 200.0;
            let k = poisson_quantile(lambda, u);
            assert!(k >= prev, "non-monotone at lambda={lambda}, u={u}: {k} < prev {prev}");
            prev = k;
        }
    }
}

/// The two branches meet. `λ` moves continuously as an MCMC chain moves, so a
/// step across the crossover must not step the drawn count: the exact answer
/// just below and the Cornish–Fisher answer just above have to agree to within
/// the one count `λ` itself moved by.
#[test]
fn the_exact_and_approximate_branches_agree_at_the_crossover() {
    // Straddling `POISSON_EXACT_CDF_MAX = 10000`.
    for &u in &[1e-9, 0.001, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.999, 1.0 - 1e-9] {
        let below = poisson_quantile(9999.9, u) as i64;
        let above = poisson_quantile(10000.1, u) as i64;
        assert!(
            (below - above).abs() <= 1,
            "the branches disagree at u={u}: exact(9999.9) = {below}, \
             Cornish-Fisher(10000.1) = {above}"
        );
    }
}

/// Degenerate rates draw `0` rather than panicking or looping, matching
/// `StatefulRng::poisson`'s guard.
#[test]
fn non_positive_lambda_draws_zero() {
    for &lambda in &[0.0, -1.0, -1e300, f64::NAN] {
        assert_eq!(poisson_quantile(lambda, 0.5), 0, "lambda = {lambda}");
        assert_eq!(poisson_quantile(lambda, 0.999), 0, "lambda = {lambda}");
    }
}
