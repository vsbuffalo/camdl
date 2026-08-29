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
/// We take the second **where it is available**, which is the large-`n` case
/// this function exists for. It places the stray mass ~5 orders of magnitude
/// closer to where it belongs: for a source compartment of 5e6 with `n*p = 9`,
/// the tail runs out around x = 295, while `n` would mean "the entire
/// susceptible population left the compartment in one substep" — a
/// catastrophic trajectory from a rounding error. It also terminates in a few
/// hundred iterations rather than `n` of them.
///
/// gh#530: the first repair is still taken when the SUPPORT ends before the
/// pmf underflows — small `n`, where `x > n` fires first. There the pmf at
/// `x = n` is perfectly representable (`Binomial(100, 0.05)`: `0.05^100 ≈
/// 8e-131`), so there is no "first unrepresentable `x`" to return and `n` is
/// the only defined answer. `binv_inverse_cdf(100, 0.05, 1 - 2^-53) == 100`.
/// Not a defect — but this comment used to describe the second repair as
/// universal, and the test below encodes the exception (`k < n || n < 1_000`)
/// while the prose did not.
fn binv_inverse_cdf(n: u64, p: f64, u: f64) -> u64 {
    let q = 1.0 - p;
    let s = p / q;
    let a = ((n + 1) as f64) * s;
    // The pmf at x = 0, i.e. q^n.
    //
    // Below `i32::MAX` this is `powi`, byte-for-byte what `rand_distr` computes
    // and therefore what every existing trajectory was generated with — do not
    // "improve" it.
    //
    // Above it, `powi` cannot express the exponent at all, which is why
    // `rand_distr` excludes large `n` from BINV and forces it into BTPE, where
    // it panics for small `n·p` (gh#525). `exp(n · ln1p(-p))` is the same
    // quantity computed a way that has an exponent to spare; `ln_1p` rather
    // than `(1-p).ln()` because `p` is necessarily tiny here (`n·p < 10` with
    // `n > 2^31` means `p < 5e-9`), which is exactly where `ln(1-p)` loses its
    // significant digits and `ln1p` does not. No existing draw takes this
    // branch — the inputs that reach it used to abort the process.
    let mut r = if n <= (i32::MAX as u64) {
        q.powi(n as i32)
    } else {
        ((n as f64) * (-p).ln_1p()).exp()
    };
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

    /// NB2 draw with the observation parameterization: mean `mu`, dispersion
    /// `k`, so `Var = mu + mu²/k` and `k → ∞` recovers Poisson(mu). This is the
    /// sampler for `x ~ neg_binomial(mean = .., r = ..)`, and it is the exact
    /// inverse of [`crate::inference::obs_loglik::negbin_logpmf`]'s `(mu, k)`.
    ///
    /// Distinct from [`Self::neg_binomial`] above, which takes `(mean, σ², dt)`
    /// — the He et al. RATE-overdispersion parameterization, whose variance
    /// scales with the integrator step. An initial state is drawn once, not per
    /// step, so it has no `dt` to scale by; passing one parameterization where
    /// the other is meant silently changes the dispersion.
    pub fn neg_binomial_dispersion(&mut self, mean: f64, k: f64) -> u64 {
        if mean <= 0.0 { return 0; }
        // k <= 0 is outside the family; the density returns -inf there, so the
        // nearest well-defined draw is the k → ∞ limit. Non-finite k (NaN or
        // ±inf) takes the same limit — tested for explicitly because `k <= 0.0`
        // is false for NaN, which would otherwise reach `Gamma::new`.
        if !k.is_finite() || k <= 0.0 { return self.poisson(mean); }
        // Unit-mean Gamma(k, 1/k) mixed into a Poisson: E[G] = 1,
        // Var[G] = 1/k, so Var[count] = mu + mu²/k.
        let g = match Gamma::new(k, 1.0 / k) {
            Ok(g) => g.sample(&mut self.0),
            Err(_) => 1.0,
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
        // gh#525: the `n <= i32::MAX` half of `rand_distr`'s predicate is
        // dropped here, deliberately. Upstream carries it because its BINV
        // computes `q.powi(n as i32)` and cannot express a larger exponent —
        // but the effect is to force LARGE n with SMALL n·p into BTPE, which is
        // invalid there: BTPE's triangle radius
        //
        //     p1 = (2.195·sqrt(npq) − 4.6·q).floor() + 0.5
        //
        // goes NEGATIVE once n·p is small, so `p4 <= 0` and the setup panics in
        // `Uniform::new(0., p4)` — an unattributed abort from inside a
        // dependency, mid-fit. `n·p < 10` is precisely the regime BINV exists
        // for, at any n, so it routes there and `binv_inverse_cdf` computes the
        // large-n initial term without `powi`.
        //
        // Reachable from user data, not just exotic models:
        // `obs_model.rs:499` takes its binomial denominator from a data column,
        // so a TSV with a large `n_examined` and a small reporting probability
        // crashed the process.
        //
        // Nothing that previously worked changes: for `n <= i32::MAX` the
        // routing, the recurrence, and the single `gen::<f64>()` are untouched.
        //
        // Nor did the inputs this newly admits work before — but "they aborted"
        // is only part of it, and the smaller part. `p4 <= 0` (the panic) needs
        // `n*p <= 4.35`; the band above it, up to `BINV_THRESHOLD`, did NOT
        // abort. Measured on the pre-change code at n = 4_294_967_296, 200k
        // draws:
        //
        //     n*p = 4.4  ->  mean 4.3414, chi2 2339   (biased: lambda_r -> 0
        //                    degenerates p4 to +/-5e9)
        //     n*p = 5.0  ->  did not return in 6s     (BTPE spinning)
        //     n*p = 6.0  ->  did not return in 6s
        //
        // So this corner previously panicked, hung, or returned a wrong
        // distribution depending on where in it you landed — and the hang is a
        // second instance of the gh#510 class, never separately reported. The
        // BINV values here are sound (chi2 18.6 at n*p = 4.4).
        let p_flipped = if p <= 0.5 { p } else { 1.0 - p };
        if (n as f64) * p_flipped < BINV_THRESHOLD {
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
mod dispersion_domain_tests {
    use super::*;

    /// `k <= 0.0` is false for NaN, so without the explicit non-finite arm a
    /// NaN dispersion reached `Gamma::new`. The k -> inf limit (a plain
    /// Poisson) is the nearest well-defined draw, and it must be finite —
    /// a NaN here becomes a NaN initial state, which is not a diagnosable
    /// failure downstream.
    #[test]
    fn non_finite_dispersion_falls_back_to_the_poisson_limit() {
        let mut rng = StatefulRng::new(42);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let draws: Vec<u64> = (0..64).map(|_| rng.neg_binomial_dispersion(10.0, bad)).collect();
            assert!(draws.iter().any(|&d| d > 0), "k={bad}: all-zero draws suggest the guard bailed");
        }
        // Negative control: a valid k still overdisperses relative to Poisson.
        let var = |xs: &[u64]| {
            let m = xs.iter().sum::<u64>() as f64 / xs.len() as f64;
            xs.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / xs.len() as f64
        };
        let tight: Vec<u64> = (0..4000).map(|_| rng.neg_binomial_dispersion(10.0, f64::NAN)).collect();
        let loose: Vec<u64> = (0..4000).map(|_| rng.neg_binomial_dispersion(10.0, 0.5)).collect();
        assert!(var(&loose) > var(&tight),
            "k=0.5 must be more dispersed than the Poisson limit: {} vs {}", var(&loose), var(&tight));
    }
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

    /// gh#525: `n > i32::MAX` with a small `n·p`. `rand_distr` forces this
    /// into BTPE (its BINV is capped by `powi`'s i32 exponent), where the
    /// triangle radius `p1 = (2.195·sqrt(npq) − 4.6·q).floor() + 0.5` goes
    /// negative and `Uniform::new(0., p4)` panics with `low >= high`.
    ///
    /// The reachable path is a DATA COLUMN: `obs_model.rs:499` takes the
    /// binomial denominator from the user's TSV, so a large `n_examined` with
    /// a small reporting probability aborted the process from inside a
    /// dependency, mid-fit, with no model context.
    #[test]
    fn huge_n_with_small_np_returns_instead_of_panicking() {
        let mut rng = StatefulRng::new(11);
        // n*p = 2.1 — squarely the BINV regime, at an n `powi` cannot reach.
        let n: u64 = 2_147_483_648; // i32::MAX + 1
        let draws: Vec<u64> = (0..2000).map(|_| rng.binomial(n, 1e-9)).collect();
        let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
        assert!((mean - 2.147).abs() < 0.25,
            "mean {mean} is not near n*p = 2.147 — the large-n initial term is wrong");

        // The sharp one. `binv_inverse_cdf` opens with r = (1-p)^n and returns
        // 0 whenever u <= r, so P(X = 0) IS the initial term, directly
        // observable. That makes this the assertion that pins the `powi` fix
        // rather than merely surviving it:
        //
        //   correct  ((n as f64) * (-p).ln_1p()).exp()  ->  0.11678
        //   reverted  q.powi(n as i32), n as i32 = -2^31 ->  8.5633
        //
        // A "pmf" of 8.56 exceeds every u, so EVERY draw returns 0 and the
        // zero-fraction goes to 1.0. The other checks in this test (d <= n,
        // the flip-back) are satisfied by an all-zero sample, which is why
        // four of five assertions here used to survive reverting the fix.
        let zero_frac = draws.iter().filter(|&&d| d == 0).count() as f64 / draws.len() as f64;
        let expect_zero = ((n as f64) * (-1e-9f64).ln_1p()).exp(); // 0.116777…
        assert!((zero_frac - expect_zero).abs() < 0.04,
            "P(X=0) = {zero_frac}, expected {expect_zero:.5} = (1-p)^n. A value \
             near 1.0 means the initial pmf term exceeded 1 — the truncating \
             `n as i32` cast that this fix exists to avoid.");

        // The reporter's exact case. n*p = 2.1e-3, so anything above a couple
        // of events would be extraordinary; `<= 5` is a real bound here, not a
        // restatement of `d <= n`.
        assert!(rng.binomial(2_147_483_648, 1e-12) <= 5);
        // p > 0.5 at huge n exercises the flip AND the flip-back on this path.
        let hi = rng.binomial(n, 1.0 - 1e-9);
        assert!(hi >= n - 40, "flip-back wrong at huge n: {hi} vs n={n}");
    }

    /// gh#530: the differential above proves the WALK replicates `rand_distr`.
    /// Nothing proved the ROUTING did — the branch predicate, the symmetry
    /// flip, the flip-back, and above all the number of RNG words consumed.
    /// Drift in any of those silently moves every chain-binomial trajectory,
    /// and every test in this module would still pass.
    ///
    /// So: sample the same seeded stream through `StatefulRng::binomial` and
    /// through `rand_distr::Binomial` directly, and require both the value and
    /// the resulting stream POSITION to agree. The position is the half that
    /// matters — a wrapper consuming one extra word returns a plausible value
    /// and desynchronises everything drawn afterwards.
    #[test]
    fn binomial_matches_rand_distr_in_value_and_rng_words_consumed() {
        use rand::SeedableRng;
        // Spans both branches: n*p below 10 (BINV) and at/above it (BTPE),
        // p on each side of 0.5 to exercise the flip, and n across the
        // i32::MAX boundary that selects the large-n initial term.
        // Bounded to n <= i32::MAX ON PURPOSE. Above it we deliberately DIVERGE
        // from `rand_distr` — gh#525 routes large n with small n*p to BINV
        // because upstream's BTPE setup computes p4 <= 0 there and panics in
        // `Uniform::new`. Upstream cannot be an oracle where upstream is the
        // bug; including n = 2^31 here panics the test inside rand_distr, which
        // is how this boundary got documented rather than assumed.
        let cases: [(u64, f64); 8] = [
            (10, 0.3), (1_000, 0.005), (1_000_000, 1e-6),   // BINV
            (100, 0.99), (20, 0.9),                          // BINV via the flip
            (1_000, 0.5), (10_000, 0.2), (100, 0.5),         // BTPE
        ];
        let mut compared = 0usize;
        for (n, p) in cases {
            for seed in 0..8u64 {
                let base = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

                let mut ours = StatefulRng(base.clone());
                let got = ours.binomial(n, p);
                let ours_pos = ours.0.get_word_pos();

                let mut theirs = base.clone();
                let want = Binomial::new(n, p).unwrap().sample(&mut theirs);
                let theirs_pos = theirs.get_word_pos();

                assert_eq!(got, want,
                    "value differs at n={n} p={p:e} seed={seed} — the routing or \
                     the flip drifted from rand_distr");
                assert_eq!(ours_pos, theirs_pos,
                    "RNG stream position differs at n={n} p={p:e} seed={seed}: \
                     ours={ours_pos} theirs={theirs_pos}. A different number of \
                     words consumed desynchronises every subsequent draw, so \
                     every trajectory and golden moves.");
                compared += 1;
            }
        }
        // Non-vacuity: the grid must actually have run. A `cases` array
        // silently emptied by a later edit would otherwise pass.
        assert_eq!(compared, 64, "the comparison grid did not run in full");
    }

    /// The neighbouring regime must keep using BTPE, unchanged — `n*p >= 10`
    /// at huge `n` is where BTPE is valid and where every existing draw of
    /// that shape came from.
    #[test]
    fn huge_n_with_large_np_still_routes_to_btpe() {
        let mut rng = StatefulRng::new(13);
        let n: u64 = 10_000_000_000;
        let draws: Vec<u64> = (0..100).map(|_| rng.binomial(n, 0.5)).collect();
        let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
        assert!((mean / (n as f64) - 0.5).abs() < 0.01,
            "mean {mean} is not near n*p = 5e9");
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
