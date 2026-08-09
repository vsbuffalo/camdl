use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson, Exp, Gamma, Binomial, StandardNormal};

/// Below `n * min(p, 1-p) < 10`, inverse-transform (BINV) beats the
/// accept/reject scheme. Must match `rand_distr` 0.4.3's `BINV_THRESHOLD`
/// exactly: it selects which of the two branches a draw takes, and taking a
/// different branch than the pre-gh#510 code would change the value AND the
/// number of RNG words consumed.
const BINV_THRESHOLD: f64 = 10.0;

/// Inverse-transform binomial quantile: the smallest `x` with `CDF(x) >= u`,
/// walked with the exact pmf recurrence. `p` must already be the flipped
/// (`<= 0.5`) probability, and `u` the single uniform draw.
///
/// This is `rand_distr` 0.4.3's BINV loop with a termination bound (gh#510).
/// For every input on which that loop terminates, this returns the identical
/// value — the recurrence and the comparison are unchanged.
///
/// **The one judgement call.** The loop can be asked for a quantile above the
/// total pmf mass, because that mass sums to slightly less than 1 in floating
/// point. There is no correct answer — the missing mass is representation
/// error, not probability. Two repairs are available and they distort the
/// distribution by the same (~1e-10) total variation:
///
///   - return `n`, the literal inverse-CDF answer once the walk runs off the
///     end of the support;
///   - return the first `x` whose pmf is no longer representable, i.e. the
///     far tail of the numerically-supported range.
///
/// We take the second. Equal distortion in total variation, but it places the
/// stray mass ~5 orders of magnitude closer to where it belongs: for a source
/// compartment of 5e6 with `n*p = 9`, the tail runs out around x = 295, while
/// `n` would mean "the entire susceptible population left the compartment in
/// one substep" — a catastrophic trajectory from a rounding error. It also
/// terminates in a few hundred iterations rather than `n` of them.
fn binv_inverse_cdf(n: u64, p: f64, u: f64) -> u64 {
    let q = 1.0 - p;
    let s = p / q;
    let a = ((n + 1) as f64) * s;
    let mut r = q.powi(n as i32);
    let mut u = u;
    let mut x: u64 = 0;
    while u > r {
        u -= r;
        x += 1;
        // `a / x - s` is positive for every x <= n, so a non-positive `r` here
        // is underflow: the pmf tail has fallen below the smallest subnormal
        // and every further term is 0.0. That is the state the upstream loop
        // spins in.
        if x > n { return n; }
        r *= a / (x as f64) - s;
        if r <= 0.0 { return x.min(n); }
    }
    x
}

/// Stateful RNG wrapping ChaCha8. Deterministic given seed.
///
/// `Clone` (ChaCha8Rng is `Clone`) is the start-from-state seam's RNG-capture
/// mechanism (gh#322): a head run clones out its final RNG so a resumed tail can
/// restore the exact stream position and reproduce a byte-identical continuation.
#[derive(Clone)]
pub struct StatefulRng(ChaCha8Rng);

impl StatefulRng {
    /// Access the underlying RNG for use with rand_distr distributions.
    pub fn inner_mut(&mut self) -> &mut ChaCha8Rng { &mut self.0 }

    pub fn new(seed: u64) -> Self {
        let seed_bytes = expand_u64_to_seed(seed.wrapping_add(0xdeadbeef_cafebabe));
        StatefulRng(ChaCha8Rng::from_seed(seed_bytes))
    }

    /// Per-stream derivation for embarrassingly parallel paths like
    /// per-particle RNG. Same master seed + different `stream` gives
    /// guaranteed-independent ChaCha8 output streams via the built-in
    /// 64-bit stream counter. IM1 in the 2026-04-19 inference review:
    /// particle_filter.rs and if2.rs previously seeded per-particle
    /// RNGs via `seed ^ i.wrapping_mul(0x517cc...)`, which left
    /// correlated low-bit structure for particles with predictable
    /// index offsets. `set_stream` is the cipher's own documented
    /// mechanism for parallel streams.
    pub fn new_stream(seed: u64, stream: u64) -> Self {
        let seed_bytes = expand_u64_to_seed(seed.wrapping_add(0xdeadbeef_cafebabe));
        let mut rng = ChaCha8Rng::from_seed(seed_bytes);
        rng.set_stream(stream);
        StatefulRng(rng)
    }

    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 { return 0; }
        let lambda = lambda.min(1e15);
        match Poisson::new(lambda) {
            Ok(p) => p.sample(&mut self.0) as u64,
            Err(_) => lambda.round() as u64, // fallback to deterministic
        }
    }

    pub fn exp(&mut self, rate: f64) -> f64 {
        if rate <= 0.0 { return f64::INFINITY; }
        match Exp::new(rate) {
            Ok(e) => e.sample(&mut self.0),
            Err(_) => 1.0 / rate, // fallback to mean
        }
    }

    /// Multiplicative Gamma-Poisson compound (He et al. 2010).
    ///
    /// Draw a unit-mean Gamma multiplier G ~ Gamma(dt/σ², σ²/dt), then
    /// Poisson(mean × G).  E[count] = mean, Var[count] = mean + mean²·σ²/dt.
    /// The dt scaling ensures aggregate noise is invariant to step size:
    /// halving dt halves per-step noise but doubles the number of steps.
    pub fn neg_binomial(&mut self, mean: f64, sigma_sq: f64, dt: f64) -> u64 {
        if mean <= 0.0 || sigma_sq <= 0.0 {
            crate::eval_stats::inc_neg_binomial_pois();
            return self.poisson(mean);
        }
        let shape = dt / sigma_sq;
        // shape < 1e-6 means sigma_sq >> dt: the Gamma is degenerate
        // (nearly all mass at zero, occasional extreme spikes).
        // Fall back to Poisson (no multiplicative noise) rather than
        // producing nonsense draws. IF2 will push sigma_se away from
        // these extreme values via low likelihood.
        if shape < 1e-6 {
            crate::eval_stats::inc_neg_binomial_pois();
            return self.poisson(mean);
        }
        let scale = sigma_sq / dt;
        let g = match Gamma::new(shape, scale) {
            Ok(g) => g.sample(&mut self.0),
            Err(_) => 1.0, // fallback: no overdispersion
        };
        self.poisson(mean * g)
    }

    /// Unit-mean Gamma multiplier for overdispersed rates (He et al. 2010).
    /// G ~ Gamma(dt/σ², σ²/dt), E[G] = 1, Var[G] = σ²/dt.
    /// Used by chain-binomial to noise the rate before probability conversion.
    pub fn gamma_multiplier(&mut self, sigma_sq: f64, dt: f64) -> f64 {
        if sigma_sq <= 0.0 { return 1.0; }
        let shape = dt / sigma_sq;
        // Degenerate guard: Gamma(1e-6, scale) puts >99.9999% of mass at zero.
        // Returning 1.0 (no noise) is the physically correct limit — "no
        // overdispersion." The transition from meaningful noise to deterministic
        // is smooth; any threshold in [1e-10, 1e-3] works identically in practice.
        // This path only triggers for particles with extreme sigma_se values
        // during IF2 exploration — such particles get terrible logliks and are
        // resampled away immediately. The fallback value is irrelevant.
        if shape < 1e-6 { return 1.0; }
        let scale = sigma_sq / dt;
        match Gamma::new(shape, scale) {
            Ok(g) => g.sample(&mut self.0),
            Err(_) => 1.0,
        }
    }

    /// Binomial(n, p) draw. Used by chain-binomial for exact multinomial
    /// competing-risk decomposition (not the Poisson approximation).
    ///
    /// Fallback for invalid inputs: if the rate is so high that p > 1 (everyone
    /// transitions), return n. If p < 0 (shouldn't happen but can from floating
    /// point with extreme parameter perturbations), return 0. These are the
    /// nearest deterministic approximations. In IF2, particles reaching these
    /// guards have extreme parameters, produce -inf logliks, and are resampled
    /// away — the fallback value doesn't affect inference.
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n == 0 || p <= 0.0 { return 0; }
        if p >= 1.0 { return n; }

        // gh#510: own the BINV branch instead of delegating it.
        //
        // `rand_distr` 0.4.3's BINV loop (binomial.rs:114-127) is UNBOUNDED:
        //
        //     let mut r = q.powi(n); let mut u: f64 = rng.gen(); let mut x = 0;
        //     while u > r { u -= r; x += 1; r *= a / (x as f64) - s; }
        //
        // It exits when the cumulative pmf reaches `u`. But `r` follows the
        // exact pmf recurrence and UNDERFLOWS TO 0.0 a few hundred terms into
        // the tail, after which `u -= 0.0` and the loop spins forever — one
        // thread at 100% CPU, no allocation, no progress, no error.
        //
        // It is reachable because the floating-point pmf does not sum to 1:
        // `q = fl(1-p)` carries up to 2^-53 of representation error, which
        // `q.powi(n)` amplifies by a factor of n. Any `u` above that sum never
        // terminates. Measured shortfall (= per-draw hang probability):
        //
        //     n = 1e4   ->  2.0e-13        n = 1e6   ->  2.9e-11
        //     n = 1e5   ->  3.5e-12        n = 5e6   ->  2.6e-10
        //
        // roughly 1.6e-17 * n. This is the population of the SOURCE
        // COMPARTMENT — `chain_binomial::step_one` draws one of these per
        // source group per substep per particle, so a long metapopulation fit
        // does 1e10+ of them and hits it in a matter of hours.
        //
        // The replication below is exact: same branch condition, same
        // symmetry flip, same single `gen::<f64>()` consumed, same recurrence.
        // Every input on which the upstream loop terminates gets the identical
        // value from the identical RNG position, so no trajectory, golden, or
        // run_id moves. See `binv_inverse_cdf` for what happens when it would
        // NOT have terminated.
        let p_flipped = if p <= 0.5 { p } else { 1.0 - p };
        if (n as f64) * p_flipped < BINV_THRESHOLD && n <= (i32::MAX as u64) {
            let u: f64 = rand::Rng::gen(&mut self.0);
            let k = binv_inverse_cdf(n, p_flipped, u);
            return if p_flipped != p { n - k } else { k };
        }

        // BTPE branch (n*p >= 10): a bounded accept/reject scheme, delegated
        // unchanged. `Binomial::new` re-derives the flip internally, so this
        // is the same call the pre-gh#510 code made.
        match Binomial::new(n, p.clamp(0.0, 1.0)) {
            Ok(b) => b.sample(&mut self.0),
            Err(_) => {
                crate::eval_stats::inc_binomial_fallback();
                if p > 0.5 { n } else { 0 }
            }
        }
    }

    /// Standard normal draw N(0, 1). Used for IF2 parameter perturbations.
    pub fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.0)
    }

    /// Uniform [0, 1) — used for Gillespie event selection.
    pub fn uniform(&mut self) -> f64 {
        use rand::Rng;
        self.0.gen()
    }
}

/// Expand a `u64` seed into the 32 bytes ChaCha8 needs, by repeating and
/// mixing with distinct multipliers. The single source of truth: the lineage
/// RNG (`crate::lineage`) calls this too, so its stream's byte layout stays
/// identical to `StatefulRng`'s — previously guaranteed only by a duplicated
/// copy plus a comment.
pub(crate) fn expand_u64_to_seed(v: u64) -> [u8; 32] {
    // Fill 32 bytes from the 8-byte u64 by repeating + mixing
    let b = v.to_le_bytes();
    let b2 = v.wrapping_mul(0x9e3779b97f4a7c15).to_le_bytes();
    let b3 = v.wrapping_mul(0x6c62272e07bb0142).to_le_bytes();
    let b4 = v.wrapping_mul(0xd800000000000000).to_le_bytes();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&b);
    seed[8..16].copy_from_slice(&b2);
    seed[16..24].copy_from_slice(&b3);
    seed[24..32].copy_from_slice(&b4);
    seed
}

#[cfg(test)]
mod binomial_termination_tests {
    use super::*;

    /// `rand_distr` 0.4.3's BINV loop, verbatim and unbounded, capped only so
    /// the test itself cannot hang. `None` means "the upstream loop would have
    /// spun forever here".
    ///
    /// This is the oracle for the differential below: it is the code we are
    /// replacing, so agreement with it IS the byte-identity proof.
    fn upstream_binv(n: u64, p: f64, u: f64, cap: u64) -> Option<u64> {
        let q = 1.0 - p;
        let s = p / q;
        let a = ((n + 1) as f64) * s;
        let mut r = q.powi(n as i32);
        let mut u = u;
        let mut x: u64 = 0;
        let mut steps = 0u64;
        while u > r {
            u -= r;
            x += 1;
            r *= a / (x as f64) - s;
            steps += 1;
            if steps > cap { return None; }
        }
        Some(x)
    }

    /// The whole point: for every input the upstream loop resolves, we return
    /// exactly what it returned. No trajectory, golden, or run_id can move.
    #[test]
    fn binv_matches_upstream_wherever_upstream_terminates() {
        let cases: Vec<(u64, f64)> = vec![
            (10, 0.3), (100, 0.05), (1_000, 5e-5), (1_000, 0.009),
            (10_000, 2e-5), (10_000, 9e-4), (100_000, 9e-5),
            (1_000_000, 1e-6), (1_000_000, 2e-7), (5_000_000, 1.8e-6),
            (2, 0.5), (1, 0.001), (i32::MAX as u64, 1e-9),
        ];
        let mut compared = 0usize;
        let mut skipped = 0usize;
        for (n, p) in cases {
            // Sweep u across the whole unit interval, deliberately including
            // the top edge where the upstream loop diverges.
            for i in 0..=2000u32 {
                let u = (i as f64) / 2000.0;
                let u = if u >= 1.0 { 1.0 - f64::EPSILON / 2.0 } else { u };
                match upstream_binv(n, p, u, 5_000_000) {
                    Some(want) => {
                        assert_eq!(binv_inverse_cdf(n, p, u), want,
                            "divergence at n={n} p={p:e} u={u}: ours={} upstream={want}",
                            binv_inverse_cdf(n, p, u));
                        compared += 1;
                    }
                    None => skipped += 1,
                }
            }
        }
        // Non-vacuity: the sweep must actually have exercised the oracle, and
        // must ALSO have found the divergent region — if `skipped` were 0 the
        // bug would not be reproduced by this grid and the test would be
        // proving nothing about the fix.
        assert!(compared > 20_000, "differential barely ran: {compared} comparisons");
        assert!(skipped > 0,
            "no input in the grid made the upstream loop diverge — this test \
             is not covering the gh#510 case at all");
    }

    /// The inputs that hang `rand_distr` 0.4.3. Verified against the real
    /// crate: `Binomial(5_000_000, 1.8e-6)` and `Binomial(1_000_000, 1e-6)`
    /// spin indefinitely when the uniform sits at the top of its range.
    #[test]
    fn binv_terminates_at_the_top_of_the_uniform_range() {
        // The largest f64 below 1 — what `rand`'s `gen::<f64>()` can return.
        let u_max = 1.0 - f64::EPSILON / 2.0;
        for (n, p) in [(5_000_000u64, 1.8e-6), (1_000_000, 1e-6),
                       (100_000, 9e-5), (100, 0.05)] {
            let k = binv_inverse_cdf(n, p, u_max);
            assert!(k <= n, "n={n} p={p:e}: returned {k} > n");
            // The repair puts the stray mass in the numerically-supported
            // tail, not at n. For n*p <= 10 the tail is O(100), so a return
            // of n would mean the whole compartment emptied in one substep.
            assert!(k < n || n < 1_000,
                "n={n} p={p:e}: returned n={k}, the catastrophic repair this \
                 function exists to avoid");
        }
    }

    /// Guard the branch condition itself. If `BINV_THRESHOLD` or the flip
    /// drifted from `rand_distr`'s, draws would silently switch algorithms and
    /// every chain-binomial trajectory would move.
    #[test]
    fn binv_branch_condition_matches_rand_distr() {
        assert_eq!(BINV_THRESHOLD, 10.0,
            "must equal rand_distr 0.4.3 binomial.rs BINV_THRESHOLD");
        // A p above 0.5 must flip BEFORE the threshold test, exactly as
        // rand_distr does — otherwise n*p is compared on the wrong side of
        // the symmetry and the draw takes the other algorithm.
        //
        // n=100, p=0.99 is the case that separates them: unflipped n*p = 99
        // selects BTPE, flipped n*(1-p) = 1 selects BINV. rand_distr flips
        // first, so BINV is correct.
        let (n, p) = (100u64, 0.99);
        let flipped = if p <= 0.5 { p } else { 1.0 - p };
        assert!((n as f64) * p >= BINV_THRESHOLD,
            "unflipped, this case would take the other branch");
        assert!((n as f64) * flipped < BINV_THRESHOLD,
            "flipped, it is a BINV case — as rand_distr computes it");
        // And the inversion must run: p=0.99 on n=100 sits near 99, not 1.
        let mut rng = StatefulRng::new(7);
        let draws: Vec<u64> = (0..400).map(|_| rng.binomial(n, p)).collect();
        let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
        assert!((mean - 99.0).abs() < 0.4,
            "mean {mean} is not near n*p = 99 — the flip was not inverted");
    }

    /// End-to-end through the public API: the draw returns, and its
    /// distribution is sane (not pinned to 0 or n).
    #[test]
    fn binomial_draws_are_distributed_not_pinned() {
        let mut rng = StatefulRng::new(42);
        let (n, p) = (1_000_000u64, 5e-6); // n*p = 5, the BINV branch
        let draws: Vec<u64> = (0..2000).map(|_| rng.binomial(n, p)).collect();
        let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
        assert!((mean - 5.0).abs() < 0.6,
            "mean {mean} is not near n*p = 5 — the sampler is skewed");
        assert!(draws.iter().any(|&d| d != draws[0]), "every draw identical");
        assert!(draws.iter().all(|&d| d <= n));
        // Symmetric case: p > 0.5 exercises the flip + inversion.
        let hi: Vec<u64> = (0..500).map(|_| rng.binomial(20, 0.9)).collect();
        let hi_mean = hi.iter().sum::<u64>() as f64 / hi.len() as f64;
        assert!((hi_mean - 18.0).abs() < 0.5,
            "mean {hi_mean} is not near n*p = 18 — the p>0.5 flip is wrong");
    }
}
