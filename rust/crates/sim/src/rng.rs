use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson, Exp, Gamma, Binomial, StandardNormal};

/// Below `n * min(p, 1-p) < 10`, inverse-transform (BINV) beats the
/// accept/reject scheme. Must match `rand_distr` 0.4.3's `BINV_THRESHOLD`
/// exactly: it selects which of the two branches a draw takes, and taking a
/// different branch than the pre-gh#510 code would change the value AND the
/// number of RNG words consumed.
const BINV_THRESHOLD: f64 = 10.0;

/// Upper bound on `n` for the BTRS route. Above it BTRS de-selects itself and
/// the draw goes to BTPE.
///
/// This is a CORRECTNESS bound, like `BINV_THRESHOLD`, and for the same kind of
/// reason: outside it the hat stops dominating and the sampler returns a wrong
/// distribution silently. `log_bound`'s second term is
/// `(n+1)·ln((n−m+1)/(n−k+1))`, whose argument is `1 + (k−m)/n`; the rounding
/// error of that ratio (`≈ ε`) is multiplied by `n+1`, so the absolute error in
/// the log-acceptance ratio grows as `n·ε`. Past `2^53` the ratio rounds to
/// exactly `1.0`, the term vanishes, and the density is tilted by `e^−(k−m)`.
///
/// Measured `sup V` over the routed hat (must stay `≤ 1`): `0.975` at `n = 1e13`,
/// `1.06` at `1e15`, `43.4` at `1e16`, `2.7e46` at `1e18` — at `n = u64::MAX`
/// with `p = 2e-18` the mean comes out 8.6% low and a 130σ outlier appears.
///
/// `1e12` is set from the analysis, not the first failure: the worst domination
/// margin anywhere in the routed domain is 0.22% (see
/// `domination_margin_at_the_boundary_is_recorded`), so requiring `n·ε` to sit a
/// decade inside it gives `n ≤ 2.2e-4/ε ≈ 1e12`. That is also three decades
/// below the first measured violation. No epidemiological population reaches it
/// — the binomial `n` is a compartment count or a data-column denominator — so
/// nothing legitimate is routed away from BTRS by this bound.
///
/// The narrower repair (`ln_1p` for that term) would extend the valid range
/// rather than fence it, but it changes arithmetic the domination sweep and
/// `log_bound_is_proportional_to_the_exact_pmf` currently certify as correct;
/// fencing first is the conservative order.
const BTRS_MAX_N: u64 = 1_000_000_000_000;

/// Smallest `us` at which BTRS's squeeze may fire. Below it the hat is too far
/// from the density for the fast accept to be sound, so the draw must go to the
/// slow test. Named because the sampler and
/// `btrs_tests::hat_dominates_and_squeeze_is_valid` must agree on it exactly: as
/// two `0.07` literals they could drift, and the sweep would then certify a
/// squeeze boundary the sampler does not use.
const SQUEEZE_US_MIN: f64 = 0.07;

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

/// The Stirling-series tail correction `ln(k!) − ln(√(2πk)·(k/e)^k)`, used by
/// [`btrs_binomial`]'s final acceptance test.
///
/// Exact tabulated values below `k = 10`, where the asymptotic series has not
/// yet converged; the series above. Transcribed from TensorFlow's
/// `random_binomial_op.cc` (`stirling_approx_tail`), Apache-2.0 — the same
/// license as this project. See [`btrs_binomial`] for the full attribution.
fn stirling_approx_tail(k: f64) -> f64 {
    debug_assert!(k >= 0.0 && k.fract() == 0.0, "tail wants a non-negative integer, got {k}");
    /// `ln(k!) − Stirling(k)` for k = 0..=9.
    const TAIL: [f64; 10] = [
        0.081_061_466_795_327_2,
        0.041_340_695_955_409_2,
        0.027_677_925_684_998_3,
        0.020_790_672_103_765_09,
        0.016_644_691_189_821_1,
        0.013_876_128_823_070_7,
        0.011_896_709_945_891_7,
        0.010_411_265_261_972_0,
        0.009_255_462_182_712_73,
        0.008_330_563_433_362_87,
    ];
    if k <= 9.0 {
        return TAIL[k as usize];
    }
    let kp1sq = (k + 1.0) * (k + 1.0);
    (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 / kp1sq) / kp1sq) / (k + 1.0)
}

/// The BTRS hat: every constant derived from `(n, p)`, plus the three functions
/// the sampler and its correctness proof must agree on.
///
/// This is a struct rather than inline arithmetic for one reason: the
/// domination proof in `btrs_tests` has to evaluate **the shipped expressions**.
/// A test that re-derived `log_bound` from the paper would be checking the
/// transcription against itself and would pass on a typo in either copy.
struct BtrsHat {
    count: f64,
    b: f64,
    a: f64,
    c: f64,
    v_r: f64,
    r: f64,
    alpha: f64,
    m: f64,
}

impl BtrsHat {
    fn new(n: u64, p: f64) -> Self {
        let count = n as f64;
        let stddev = (count * p * (1.0 - p)).sqrt();
        let b = 1.15 + 2.53 * stddev;
        Self {
            count,
            b,
            a: -0.0873 + 0.0248 * b + 0.01 * p,
            c: count * p + 0.5,
            v_r: 0.92 - 4.2 / b,
            r: p / (1.0 - p),
            alpha: (2.83 + 5.1 / b) * stddev,
            m: ((count + 1.0) * p).floor(),
        }
    }

    /// The candidate value for a given `u`, via the hat's inverse transform.
    #[inline]
    fn k_of(&self, u: f64, us: f64) -> f64 {
        ((2.0 * self.a / us + self.b) * u + self.c).floor()
    }

    /// `us`, the folded distance from the centre of the `u` interval. One
    /// definition, used by the sampler and by the domination sweep — they were
    /// separate copies, which is how a sweep can end up certifying a hat the
    /// sampler does not actually use.
    #[inline]
    fn us_of(u: f64) -> f64 {
        0.5 - u.abs()
    }

    /// Whether a candidate `k` is in `[0, n]`, tested in f64 BEFORE any integer
    /// cast. Shared with the sweep for the same reason as [`Self::us_of`].
    ///
    /// Note this is deliberately NOT NaN-tolerant — for `k = NaN` both
    /// comparisons are false and it answers "in support". The NaN case is
    /// excluded upstream by `binomial`'s `!p.is_finite()` guard, which is where
    /// it belongs; see the comment there.
    #[inline]
    fn in_support(&self, k: f64) -> bool {
        !(k < 0.0 || k > self.count)
    }

    /// `log` of the density-to-hat ratio at `k`. The slow acceptance test.
    #[inline]
    fn log_bound(&self, k: f64) -> f64 {
        (self.m + 0.5) * ((self.m + 1.0) / (self.r * (self.count - self.m + 1.0))).ln()
            + (self.count + 1.0) * ((self.count - self.m + 1.0) / (self.count - k + 1.0)).ln()
            + (k + 0.5) * (self.r * (self.count - k + 1.0) / (k + 1.0)).ln()
            + stirling_approx_tail(self.m)
            + stirling_approx_tail(self.count - self.m)
            - stirling_approx_tail(k)
            - stirling_approx_tail(self.count - k)
    }

    /// The slow test, in the reference's own algebraic arrangement: accept iff
    /// `ln(v·α / (a/us² + b)) ≤ log_bound(k)`.
    #[inline]
    fn slow_accepts(&self, v: f64, us: f64, k: f64) -> bool {
        (v * self.alpha / (self.a / (us * us) + self.b)).ln() <= self.log_bound(k)
    }

    /// The squeeze: the fast accept, taken where the hat is tight enough to
    /// return `k` without touching the density at all. This is the branch BTRS
    /// wins on.
    ///
    /// A method rather than an inline comparison against `v_r`, for exactly the
    /// reason [`SQUEEZE_US_MIN`] is a named constant. The sampler runs this
    /// comparison and `btrs_tests::hat_dominates_and_squeeze_is_valid` certifies
    /// it; as two copies they can drift, and then the sweep certifies a squeeze
    /// the sampler does not use. Widening the shipped comparison by a factor of
    /// 1.10 is a 4.3% relative pmf error at `(6.3e6, 3.05e-5)` and used to leave
    /// the entire suite green (gh#802).
    #[inline]
    fn squeeze_accepts(&self, v: f64, us: f64) -> bool {
        us >= SQUEEZE_US_MIN && v <= self.v_r
    }

    /// One BTRS proposal: `Some(k)` if the pair `(u, v)` is accepted, `None` if
    /// it must be redrawn. The whole body of [`btrs_binomial`]'s loop, as a
    /// method on the hat.
    ///
    /// Separated for the same reason [`Self::us_of`] and [`Self::in_support`]
    /// are methods, but one level up: the support guard below is a DELIBERATE
    /// DEVIATION from the reference, and in the routed domain it is a no-op —
    /// the hat's geometry keeps every squeeze-region candidate inside `[0, n]`
    /// (it takes `n·p < 1.3` to break that, and the routing predicate requires
    /// `n·p ≥ 10`). So no draw can distinguish the guard's presence, and
    /// deleting it was green under `--release`, where the `debug_assert!` below
    /// is compiled out and the gate never builds this module optimised (gh#802).
    /// A test can hand this method a hat the sampler would never construct;
    /// `btrs_tests::an_out_of_support_candidate_is_redrawn_not_returned` does.
    #[inline(always)]
    fn propose(&self, u: f64, v: f64) -> Option<u64> {
        let us = Self::us_of(u);
        let k = self.k_of(u, us);

        // The reference does this check only AFTER the squeeze, and so returns
        // `k` from the squeeze unchecked, on the strength of the hat's
        // in-support guarantee. Hoisting it is a no-op wherever that guarantee
        // holds — `hat_dominates_and_squeeze_is_valid` asserts it holds across
        // the routed domain — and where it might not, this redraws instead of
        // handing the caller a `k > n` that the `p > 0.5` reflection would turn
        // into a `u64` underflow of ~1.8e19, or a negative `k` that the
        // saturating `as u64` below would turn into a silent zero count.
        if !self.in_support(k) {
            return None;
        }
        let k_int = k as u64;
        debug_assert!(
            (k_int as f64) <= self.count,
            "k={k} escaped the f64 support check at n={}",
            self.count
        );

        if self.squeeze_accepts(v, us) {
            return Some(k_int);
        }
        if self.slow_accepts(v, us, k) {
            return Some(k_int);
        }
        None
    }

    /// The acceptance ratio `V(u)`: the slow test accepts exactly when
    /// `v ≤ V(u)`. **This is what BTRS's exactness is.** The scheme is a valid
    /// rejection sampler iff `V ≤ 1` everywhere (the hat dominates the pmf), and
    /// the squeeze is valid iff `V ≥ v_r` wherever the squeeze can fire. Both are
    /// asserted deterministically in `btrs_tests::hat_dominates_and_squeeze_is_valid`.
    #[cfg(test)]
    fn accept_ratio(&self, us: f64, k: f64) -> f64 {
        self.log_bound(k).exp() * (self.a / (us * us) + self.b) / self.alpha
    }
}

/// Binomial draw by **transformed rejection with squeeze** (BTRS) — Hörmann
/// (1993), *The generation of binomial random variates*, J. Statist. Comput.
/// Simul. 46(1–2):101–110.
///
/// A rejection sampler: the accepted values are distributed `Binomial(n, p)` up
/// to the floating-point accuracy of the acceptance test, whose density is
/// evaluated through [`stirling_approx_tail`]'s table-plus-truncated-series
/// makes, so it is not a regression — but it is why this doc says "up to
/// floating point" and not "exact".
///
/// **Measured, against a 60-digit reference for `ln(k!)`** — an earlier version
/// of this comment said "~1e-15" and "at machine precision", which were wrong by
/// five to six orders of magnitude. The max log-density distortion is `5.4e-11`
/// at `(20, 0.5)`, `3.0e-11` at `(200, 0.05)`, `7.2e-10` at `(6.3e6, 3.05e-5)`
/// and `9.9e-10` at `(8.75e6, 2.2e-5)`; the implied total variation is `~2e-10`,
/// so a χ² would need `1e13`–`1e16` draws to see it. Harmless, but say the real
/// number.
///
/// Two sources, and the second is why `BTRS_MAX_N` exists: the truncated
/// Stirling series is `4e-9` relative at `k = 10`, which floors the small-`n`
/// accuracy; and `(n+1)·ln(ratio)` cancels, contributing `O(n·ε)`. A rigorous
/// `O(n·ε)` total-variation bound is given by *Assessing the Quality of Binomial
/// Samplers* (arXiv 2506.12061, Thm 5.1) — loose (~1e-5 at n = 6.3e6), but its
/// shape, growing in `n`, is confirmed here with constant `≈ ε`.
///
/// **Why this exists.** The BTPE branch it is measured against (`rand_distr`
/// 0.4.3, after Kachitvichyanukul & Schmeiser 1988) was profiled at **38.9% of a
/// PGAS fit** on the province model — half the run once the RNG bytes it consumes
/// are counted (`docs/dev/notes/2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`).
/// BTPE pays ~10 setup constants, two `Uniform` constructions, and — the real
/// cost — a walk from the mode to the sampled value with one f64 division per
/// step, which fires on almost every draw. BTRS uses a single transformed
/// rejection hat, so its setup is ~6 constants and its dominant path is a
/// squeeze that accepts with no logarithm and no division chain.
///
/// **Contract.** `p` must already be the flipped (`≤ 0.5`) probability and
/// `n · p ≥ BINV_THRESHOLD`. Both matter for correctness, not just speed: the
/// hat's domination margin falls to **0.22%** near `n·p = 10` (at `(23, 0.4583)`,
/// `n·p = 10.54`) and goes NEGATIVE by **`n·p ≈ 9.64`** — a gap of 3.6% in `n·p`,
/// not the 30% an earlier version of this comment implied by naming 7. So the
/// routing predicate in [`StatefulRng::binomial`] is what keeps this sampler
/// valid at the bottom, and [`BTRS_MAX_N`] is what keeps it valid at the top.
///
/// Flipping first also keeps `k > n` unreachable at `n > 2^53`, where `n as f64`
/// rounds up and `n − k` could underflow `u64`. Note that is a SEPARATE hazard
/// from the `O(n·ε)` precision loss above, which is the one that actually bites
/// at large `n` and which `BTRS_MAX_N` fences.
///
/// **Source.** Transcribed from TensorFlow's `random_binomial_op.cc` (`btrs`),
/// Apache-2.0 — the same license as this project. This is deliberately the
/// **TensorFlow variant**, not the paper's: TensorFlow Probability's sibling
/// implementation notes that it "deviates from Hormann's BTRS algorithm, as there
/// is a log missing". The variant transcribed here is the one whose hat is
/// verified to dominate by `hat_dominates_and_squeeze_is_valid`; the paper's is
/// not verified here, so do not "restore" the missing log without re-running that
/// proof.
fn btrs_binomial<R: rand::Rng + ?Sized>(n: u64, p: f64, rng: &mut R) -> u64 {
    debug_assert!(p > 0.0 && p <= 0.5, "btrs wants the flipped probability, got {p}");
    debug_assert!((n as f64) * p >= BINV_THRESHOLD, "btrs called below its regime");

    let h = BtrsHat::new(n, p);

    loop {
        // `u ∈ [−0.5, 0.5)`, so `us ∈ (0, 0.5]`. At `u == −0.5` exactly (one draw
        // in 2^53) `us == 0`, `2a/us` is `+∞` and `k` is `−∞` — which
        // [`BtrsHat::propose`]'s support check rejects in f64, BEFORE any integer
        // cast. That order is load-bearing: `(−∞) as u64` saturates to 0 in Rust,
        // so casting first would silently return a zero count instead of
        // redrawing.
        let u: f64 = rng.gen::<f64>() - 0.5;
        let v: f64 = rng.gen();
        if let Some(k) = h.propose(u, v) {
            return k;
        }
    }
}

/// Stateful RNG wrapping ChaCha8. Deterministic given seed.
/// `Clone` (ChaCha8Rng is `Clone`) is the start-from-state seam's RNG-capture
/// mechanism (gh#322): a head run clones out its final RNG so a resumed tail can
/// restore the exact stream position and reproduce a byte-identical continuation.
#[derive(Clone)]
pub struct StatefulRng {
    inner: ChaCha8Rng,
    /// Which accept/reject scheme [`Self::binomial`] uses above
    /// `BINV_THRESHOLD`. Carried ON THE RNG rather than threaded as a
    /// parameter, because the RNG already reaches every draw site — the
    /// chain-binomial step, the initial-state law, quantities, and the
    /// observation sampler — while a parameter would have to be added to
    /// `step_one`'s ten arguments and five other signatures.
    ///
    /// It also makes the choice immune to the hazard that killed the
    /// thread-local: PGAS draws on rayon workers inside a nested `par_iter`,
    /// so a per-thread setting reaches whichever particles that worker happens
    /// to steal. A per-RNG setting is fixed at construction and travels with
    /// the particle.
    algo: BinomialAlgorithm,
}

/// Which accept/reject scheme [`StatefulRng::binomial`] uses above
/// `BINV_THRESHOLD`. Below it, BINV is used regardless — that branch is not
/// part of this choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default,
         serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinomialAlgorithm {
    /// `rand_distr` 0.4.3's BTPE (Kachitvichyanukul & Schmeiser 1988). The
    /// production default, and the oracle the BTRS suite is calibrated against.
    #[default]
    Btpe,
    /// [`btrs_binomial`] — Hörmann (1993). Faster; not bit-compatible with
    /// `Btpe` (a different rejection scheme accepts different draws from the
    /// same stream), so selecting it changes results and is therefore a
    /// deliberate, identity-bearing choice. gh#747.
    Btrs,
}

/// The production default. Selecting the other sampler CHANGES DRAWS, so it
/// arrives through `Stage::PGAS.binomial` — a typed field that enters the run
/// address — and travels on the RNG itself (`StatefulRng::with_binomial`).
///
/// An environment variable was considered and rejected: one that changed draws
/// without entering the run address would serve one sampler's posterior from
/// the other's cache leaf. gh#241 removed `CAMDL_PF_WALLCLOCK_TIMEOUT_S` for
/// the same reason rather than hashing it. See
/// `docs/dev/proposals/2026-08-24-faster-binomial-sampler.md`.
const DEFAULT_BINOMIAL: BinomialAlgorithm = BinomialAlgorithm::Btpe;

impl std::str::FromStr for BinomialAlgorithm {
    type Err = String;
    /// Accepts the same spellings serde does, so a `fit.toml` value and a CLI
    /// flag cannot disagree about what `btrs` means.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "btpe" => Ok(BinomialAlgorithm::Btpe),
            "btrs" => Ok(BinomialAlgorithm::Btrs),
            other  => Err(format!(
                "unknown binomial sampler '{other}' (expected 'btpe' or 'btrs')")),
        }
    }
}

impl std::fmt::Display for BinomialAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { BinomialAlgorithm::Btpe => "btpe",
                                 BinomialAlgorithm::Btrs => "btrs" })
    }
}

impl StatefulRng {
    /// Access the underlying RNG for use with rand_distr distributions.
    pub fn inner_mut(&mut self) -> &mut ChaCha8Rng { &mut self.inner }

    /// Select the binomial sampler this RNG uses. Builder rather than setter so
    /// the choice is fixed where the RNG is built and cannot drift mid-run.
    ///
    /// Selecting `Btrs` CHANGES DRAWS — a different rejection scheme accepts
    /// different values from the same stream — so it must arrive from an input
    /// that enters the run address. `Stage::PGAS.binomial` is that input; see
    /// `docs/dev/proposals/2026-08-24-faster-binomial-sampler.md`.
    pub fn with_binomial(mut self, algo: BinomialAlgorithm) -> Self {
        self.algo = algo;
        self
    }

    /// Which sampler this RNG will use. Exists so "the sampler that ran equals
    /// the one that was hashed" is assertable from outside this module.
    pub fn binomial_algorithm(&self) -> BinomialAlgorithm { self.algo }

    pub fn new(seed: u64) -> Self {
        let seed_bytes = expand_u64_to_seed(seed.wrapping_add(0xdeadbeef_cafebabe));
        StatefulRng { inner: ChaCha8Rng::from_seed(seed_bytes), algo: DEFAULT_BINOMIAL }
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
        StatefulRng { inner: rng, algo: DEFAULT_BINOMIAL }
    }

    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 { return 0; }
        let lambda = lambda.min(1e15);
        match Poisson::new(lambda) {
            Ok(p) => p.sample(&mut self.inner) as u64,
            Err(_) => lambda.round() as u64, // fallback to deterministic
        }
    }

    pub fn exp(&mut self, rate: f64) -> f64 {
        if rate <= 0.0 { return f64::INFINITY; }
        match Exp::new(rate) {
            Ok(e) => e.sample(&mut self.inner),
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
            Ok(g) => g.sample(&mut self.inner),
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
        // nearest well-defined draw is the k → ∞ limit.
        if !(k > 0.0) || !k.is_finite() { return self.poisson(mean); }
        // Unit-mean Gamma(k, 1/k) mixed into a Poisson: E[G] = 1,
        // Var[G] = 1/k, so Var[count] = mu + mu²/k.
        let g = match Gamma::new(k, 1.0 / k) {
            Ok(g) => g.sample(&mut self.inner),
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
            Ok(g) => g.sample(&mut self.inner),
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
        // A NaN `p` passes BOTH guards above — every NaN comparison is false —
        // and then `p_flipped = 1 - NaN` is NaN, `(n as f64) * NaN < THRESHOLD`
        // is false, so it reaches the accept/reject branch with a NaN hat. BTPE
        // absorbs that (`Binomial::new` rejects it and the arm returns 0), but
        // BTRS spins forever: `k` is NaN, the support check `k < 0.0 || k > count`
        // is false for NaN so it does not redraw, and both the squeeze and
        // `slow_accepts` compare against NaN and are false. That is the gh#510
        // hang class — one thread at 100% CPU, no allocation, no progress, no
        // error — which the comment below exists to explain the last instance of.
        // Guard it HERE, with its siblings, rather than inside one sampler: the
        // hazard is the input, not the algorithm, and a `debug_assert!` in
        // `btrs_binomial` is compiled out of exactly the builds that run fits.
        if !p.is_finite() { return 0; }

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
            let u: f64 = rand::Rng::gen(&mut self.inner);
            let k = binv_inverse_cdf(n, p_flipped, u);
            return if p_flipped != p { n - k } else { k };
        }

        // BTPE branch (n*p >= 10): a bounded accept/reject scheme, delegated
        // unchanged. `Binomial::new` re-derives the flip internally, so this
        // is the same call the pre-gh#510 code made.
        // Above `BTRS_MAX_N` the hat stops dominating, so BTRS de-selects itself
        // and the draw falls back to BTPE — the pre-BTRS behaviour, unchanged,
        // including its own huge-`n` fallback. Resolved BEFORE the match so the
        // match stays exhaustive over the enum: a third algorithm must not be
        // able to reach the hot path through a `_` arm.
        let algo = match self.algo {
            BinomialAlgorithm::Btrs if n > BTRS_MAX_N => BinomialAlgorithm::Btpe,
            other => other,
        };
        match algo {
            // BTRS takes the FLIPPED probability and un-flips, so it only ever
            // sees `p <= 0.5` with `n*p >= BINV_THRESHOLD` — the regime its hat
            // is derived for. BTPE keeps the original `p`: `rand_distr` does its
            // own flipping internally, and routing around that would change the
            // draw on the production path.
            BinomialAlgorithm::Btrs => {
                let k = btrs_binomial(n, p_flipped, &mut self.inner);
                if p_flipped != p { n - k } else { k }
            }
            BinomialAlgorithm::Btpe => match Binomial::new(n, p.clamp(0.0, 1.0)) {
                Ok(b) => b.sample(&mut self.inner),
                Err(_) => {
                    crate::eval_stats::inc_binomial_fallback();
                    if p > 0.5 { n } else { 0 }
                }
            },
        }
    }

    /// Standard normal draw N(0, 1). Used for IF2 parameter perturbations.
    pub fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.inner)
    }

    /// Uniform [0, 1) — used for Gillespie event selection.
    pub fn uniform(&mut self) -> f64 {
        use rand::Rng;
        self.inner.gen()
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

                let mut ours = StatefulRng { inner: base.clone(), algo: DEFAULT_BINOMIAL };
                let got = ours.binomial(n, p);
                let ours_pos = ours.inner.get_word_pos();

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

#[cfg(test)]
mod btrs_tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;

    /// Exact binomial pmf in log space, so a 6.3e6-trial cell is as safe as a
    /// 20-trial one (`(1-p)^n` underflows long before `lgamma` does).
    fn log_pmf(n: u64, k: u64, p: f64) -> f64 {
        let (nf, kf) = (n as f64, k as f64);
        numerics::lgamma(nf + 1.0) - numerics::lgamma(kf + 1.0) - numerics::lgamma(nf - kf + 1.0)
            + kf * p.ln()
            + (nf - kf) * (1.0 - p).ln()
    }

    /// Pearson χ² of `draws` against the exact `Binomial(n, p)` pmf, with the
    /// low and high tails POOLED until every cell expects ≥ 5 — the standard
    /// validity condition, without which the statistic is not χ²-distributed
    /// and the test would be measuring its own binning.
    ///
    /// Returns `(chi2, degrees_of_freedom)`.
    fn chi_square(draws: &[u64], n: u64, p: f64) -> (f64, usize) {
        let total = draws.len() as f64;
        let expected: Vec<f64> =
            (0..=n).map(|k| total * log_pmf(n, k, p).exp()).collect();
        let mut observed = vec![0.0f64; (n + 1) as usize];
        for &d in draws {
            observed[d as usize] += 1.0;
        }
        // Walk up from k=0 pooling into the first viable cell, and down from k=n
        // likewise; the interior keeps its own cells.
        let mut lo = 0usize;
        let mut acc = 0.0;
        while lo <= n as usize && acc + expected[lo] < 5.0 {
            acc += expected[lo];
            lo += 1;
        }
        let mut hi = n as usize;
        let mut acc_hi = 0.0;
        while hi > lo && acc_hi + expected[hi] < 5.0 {
            acc_hi += expected[hi];
            hi -= 1;
        }
        let mut cells: Vec<(f64, f64)> = Vec::new();
        let (mut o_lo, mut e_lo) = (0.0, 0.0);
        for k in 0..lo {
            o_lo += observed[k];
            e_lo += expected[k];
        }
        let (mut o_hi, mut e_hi) = (0.0, 0.0);
        for k in (hi + 1)..=(n as usize) {
            o_hi += observed[k];
            e_hi += expected[k];
        }
        if e_lo > 0.0 { cells.push((o_lo + observed[lo], e_lo + expected[lo])); }
        let start = if e_lo > 0.0 { lo + 1 } else { lo };
        for k in start..=hi {
            cells.push((observed[k], expected[k]));
        }
        if e_hi > 0.0 {
            let last = cells.len() - 1;
            cells[last].0 += o_hi;
            cells[last].1 += e_hi;
        }
        let chi2: f64 = cells.iter().map(|&(o, e)| (o - e) * (o - e) / e).sum();
        (chi2, cells.len().saturating_sub(1))
    }

    /// χ² critical value at ≈6σ on the `Normal(df, 2·df)` approximation. Loose
    /// on purpose: every test here uses a FIXED seed, so a pass/fail is
    /// deterministic and cannot flake, and the headroom means an unrelated
    /// change to RNG consumption order does not turn into a red here. That the
    /// looseness has NOT cost the suite its power is not asserted — it is
    /// demonstrated by `chi_square_rejects_a_one_percent_bias` below.
    fn critical(df: usize) -> f64 {
        df as f64 + 6.0 * (2.0 * df as f64).sqrt()
    }

    /// The (n, p) grid. Every cell has `n·min(p, 1−p) ≥ BINV_THRESHOLD`, so
    /// every cell actually reaches the branch under test rather than falling
    /// through to BINV. `(20, 0.5)` and `(40, 0.25)` sit exactly ON the
    /// threshold — the tightest regime for the squeeze's in-support guarantee.
    /// `(500, 0.8)` exercises the `p > 0.5` reflection.
    /// Note the `p` here are the CALLER's, not flipped, and
    /// `chi_square_rejects_a_one_percent_bias` perturbs them by +1%. That
    /// perturbation must not push a cell across `BINV_THRESHOLD`, or the canary
    /// measures BINV instead of the arm under test — which `(20, 0.5)` did:
    /// `p → 0.505` flips to 0.495 and `n·p_flip = 9.9 < 10`. Replaced with
    /// `(24, 0.5)` (`n·p_flip = 11.88` after the bias), keeping a small-`n` cell
    /// without the routing hazard.
    const GRID: &[(u64, f64)] = &[
        (24, 0.5),
        (40, 0.25),
        (100, 0.1),
        (100, 0.5),
        (1000, 0.05),
        (1000, 0.5),
        (500, 0.8),
    ];

    /// Draw under a chosen sampler. The choice rides on the RNG, so there is
    /// no cross-test leakage to guard against — the `AlgoGuard` this used to
    /// need went away with the thread-local it protected.
    fn draw(algo: BinomialAlgorithm, n: u64, p: f64, count: usize, seed: u64) -> Vec<u64> {
        let mut rng = StatefulRng { inner: ChaCha8Rng::seed_from_u64(seed), algo: DEFAULT_BINOMIAL }
            .with_binomial(algo);
        (0..count).map(|_| rng.binomial(n, p)).collect()
    }

    /// `(n, p)` in the routed BTRS domain — `p` ALREADY FLIPPED (`≤ 0.5`) and
    /// `n·p ≥ BINV_THRESHOLD`, i.e. exactly what `btrs_binomial` is ever handed.
    /// Four groups: the `n·p = 10` boundary (thinnest domination margin); the
    /// SPLIT-draw regime `n ≈ 20..200`, which is half of this model's draws and
    /// which the note's `np ≈ 87..192` framing missed; the province total-exit
    /// regimes; and the huge-`n` susceptible draws.
    ///
    /// The last group is ADVERSARIAL, not representative: three cells found by
    /// searching for the tightest `sup V` rather than by reading the model. Each
    /// is the sole witness to a single-constant typo that the other eleven miss
    /// entirely — `m`'s `(n+1)p` losing its `+1` (`sup V = 1.0568` at `(22,
    /// 0.4997)`), `v_r`'s `4.2 → 4.1` (squeeze invalid at `(752, 0.0135)`), and
    /// `alpha`'s `5.1 → 5.0`. Without them the sweep's own docstring claim — that
    /// it catches every single-constant typo — was false. Do not drop a cell
    /// here for looking arbitrary: `(23, 0.4583)` carries the thinnest margin in
    /// the whole routed domain (0.22%) and is why that figure is quotable.
    const DOMAIN: &[(u64, f64)] = &[
        (20, 0.5), (40, 0.25), (100, 0.1), (200, 0.05),
        (100, 0.5), (190, 0.5), (500, 0.2), (1000, 0.5),
        (400, 0.476), (520, 0.295), (780, 0.111), (5000, 0.038),
        (6_300_000, 3.05e-5), (8_750_000, 2.2e-5),
        (22, 0.4997), (752, 0.0135), (23, 0.4583),
    ];

    /// The worst (largest) acceptance ratio `V` over a deterministic lattice in
    /// `u`, and the worst squeeze overshoot: the largest relative distance above
    /// `V` at which [`BtrsHat::squeeze_accepts`] still accepts. `None` means it
    /// never did, which is what squeeze validity is.
    ///
    /// The overshoot is PROBED through the sampler's own predicate rather than
    /// computed as `v_r − V`. The squeeze accepts a half-line in `v`, so "it
    /// never accepts a `v` the slow test rejects" is decided by evaluating the
    /// shipped comparison just above `V`; reading `v_r` off the struct instead
    /// certifies the field while leaving the comparison that uses it untested,
    /// which is how a 10% widening of that comparison stayed green (gh#802).
    ///
    /// **This lattice is a SAMPLE, not a proof**, and its bias has a known
    /// direction: a maximum over a subset of `u` is at most the maximum over all
    /// of it, so it UNDERSTATES `sup V`. Measured at the tightest cell,
    /// `(23, 0.4583)`: 100k points give 0.997496, 10M give 0.997773 — a
    /// shortfall of 2.77e-4 against a true margin of 2.23e-3, i.e. 12.4% of it.
    /// Eightfold headroom, nothing hidden — but a cell whose margin fell below
    /// ~1e-3 would need a finer lattice before this test could be believed about
    /// it.
    fn worst_ratios(h: &BtrsHat) -> (f64, Option<f64>) {
        const STEPS: usize = 100_000;
        /// Relative distances above `V` at which the squeeze must already
        /// reject. The finest rung sets the resolution: the tightest squeeze
        /// margin in `DOMAIN` is `v_r/V = 0.99303` at `(8.75e6, 2.2e-5)`, so
        /// probing at `V·(1 + 1e-12)` detects any widening of the comparison
        /// beyond a factor of 1.0071.
        const OVERSHOOT: &[f64] = &[1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1];

        let mut worst_v = 0.0f64;
        let mut worst_overshoot: Option<f64> = None;
        for i in 0..STEPS {
            let u = -0.5 + (i as f64 + 0.5) / STEPS as f64;
            let us = BtrsHat::us_of(u);
            let k = h.k_of(u, us);
            if !h.in_support(k) {
                continue;
            }
            let v = h.accept_ratio(us, k);
            if v > worst_v {
                worst_v = v;
            }
            for &d in OVERSHOOT {
                if h.squeeze_accepts(v * (1.0 + d), us) {
                    worst_overshoot = Some(worst_overshoot.map_or(d, |w: f64| w.max(d)));
                }
            }
        }
        (worst_v, worst_overshoot)
    }

    /// BTRS must fit the exact pmf — and so must BTPE, on the same grid with the
    /// same statistic. BTPE is the POSITIVE CONTROL: it is the sampler we
    /// already trust, so if it failed here the test would be miscalibrated
    /// rather than the sampler broken, and this assertion is what tells the two
    /// apart.
    #[test]
    fn both_samplers_fit_the_exact_pmf() {
        const N: usize = 200_000;
        for &(n, p) in GRID {
            for algo in [BinomialAlgorithm::Btpe, BinomialAlgorithm::Btrs] {
                let draws = draw(algo, n, p, N, 20_260_824);
                let (chi2, df) = chi_square(&draws, n, p);
                assert!(
                    chi2 < critical(df),
                    "{algo:?} n={n} p={p}: chi2={chi2:.1} exceeds critical={:.1} (df={df})",
                    critical(df)
                );
            }
        }
    }

    /// **The calibration proof — this is what makes the suite above non-vacuous.**
    ///
    /// A χ² test that passes everything is worthless. Feed the same statistic a
    /// sampler biased by 1% in `p` and it must be REJECTED, on every grid cell.
    /// If this test ever starts passing (i.e. the bias goes undetected), the
    /// suite has lost its power and `both_samplers_fit_the_exact_pmf` no longer
    /// means anything.
    #[test]
    fn chi_square_rejects_a_one_percent_bias() {
        const N: usize = 200_000;
        for &(n, p) in GRID {
            // Draw from Binomial(n, 1.01p) but score against Binomial(n, p).
            let draws = draw(BinomialAlgorithm::Btrs, n, (p * 1.01).min(1.0), N, 20_260_824);
            let (chi2, df) = chi_square(&draws, n, p);
            assert!(
                chi2 > critical(df),
                "n={n} p={p}: a 1% bias produced chi2={chi2:.1}, which the critical \
                 value {:.1} (df={df}) FAILED to reject — the suite has lost its power",
                critical(df)
            );
        }
    }

    /// Every draw must land in `[0, n]`. This is the assertion that covers the
    /// squeeze's early return, which accepts `k` WITHOUT a range check on the
    /// strength of the hat's in-support guarantee — a saturating `as u64` would
    /// turn a violation into a silent 0 rather than a panic, so the guarantee is
    /// tested rather than trusted. The on-threshold cells are the tight ones.
    #[test]
    fn draws_stay_in_support_including_the_squeeze_return() {
        for &(n, p) in GRID {
            for algo in [BinomialAlgorithm::Btpe, BinomialAlgorithm::Btrs] {
                for (i, &d) in draw(algo, n, p, 100_000, 7).iter().enumerate() {
                    assert!(d <= n, "{algo:?} n={n} p={p}: draw #{i} = {d} exceeds n");
                }
            }
        }
    }

    /// The regimes the province model actually visits (`np ≈ 87..192` on `n`
    /// from 1e2 to 6.3e6) are far too large for an exact-pmf χ², so they are
    /// checked on the first two moments instead — against a tolerance derived
    /// from the SAMPLING standard error, not an eyeballed epsilon. 5 SE on the
    /// mean; the variance gets a relative band, its own SE being
    /// `σ²·sqrt(2/N)` for a near-normal count.
    #[test]
    fn moments_agree_at_the_province_model_regimes() {
        const N: usize = 200_000;
        // (n, p) reconstructed from the fit: S≈6.3e6 with per-capita ~3e-5;
        // E/I/C compartments in the hundreds with their exit hazards.
        for &(n, p) in &[
            (6_300_000u64, 3.05e-5f64),
            (8_750_000, 2.2e-5),
            (400, 0.476),
            (520, 0.295),
            (780, 0.111),
        ] {
            let (nf, mean_x, var_x) = (n as f64, n as f64 * p, n as f64 * p * (1.0 - p));
            assert!(nf * p.min(1.0 - p) >= BINV_THRESHOLD, "cell must reach the branch");
            let draws = draw(BinomialAlgorithm::Btrs, n, p, N, 99);
            let m: f64 = draws.iter().map(|&d| d as f64).sum::<f64>() / N as f64;
            let v: f64 =
                draws.iter().map(|&d| (d as f64 - m) * (d as f64 - m)).sum::<f64>() / (N as f64 - 1.0);
            let se_mean = (var_x / N as f64).sqrt();
            assert!(
                (m - mean_x).abs() < 5.0 * se_mean,
                "n={n} p={p}: mean {m:.4} vs exact {mean_x:.4}, off by {:.2} SE",
                (m - mean_x).abs() / se_mean
            );
            let se_var = var_x * (2.0 / N as f64).sqrt();
            assert!(
                (v - var_x).abs() < 6.0 * se_var,
                "n={n} p={p}: var {v:.4} vs exact {var_x:.4}, off by {:.2} SE",
                (v - var_x).abs() / se_var
            );
        }
    }

    /// The BINV↔BTRS seam must not be visible in the distribution. Sweep `n·p`
    /// across `BINV_THRESHOLD` and require the mean to track `np` on both sides
    /// — a sampler that was wrong on one side of the branch would show up as a
    /// step here even though each side is individually plausible.
    #[test]
    fn no_discontinuity_across_the_binv_threshold() {
        const N: usize = 100_000;
        // `(n, p)` chosen so `n·p` really lands where the label says. Picking a
        // fixed `p` and rounding `n = np/p` does NOT work: at `p = 0.25` the
        // labels 9.99, 10.0 and 10.01 all round to `n = 40`, so the three cells
        // that are supposed to straddle the seam at 0.01 resolution were one
        // cell measured three times, and the sweep had nothing between 9.5 and
        // 10.0 at all.
        for &(n, p, np) in &[
            (2_000u64, 0.004f64, 8.0f64),
            (1_900, 0.005, 9.5),
            (1_998, 0.005, 9.99),
            (2_000, 0.005, 10.0),
            (2_002, 0.005, 10.01),
            (2_400, 0.005, 12.0),
            (4_000, 0.005, 20.0),
        ] {
            let exact = n as f64 * p;
            assert!(
                (exact - np).abs() < 1e-9,
                "cell (n={n}, p={p}) has n·p = {exact}, not the labelled {np}"
            );
            let draws = draw(BinomialAlgorithm::Btrs, n, p, N, 4242);
            let m: f64 = draws.iter().map(|&d| d as f64).sum::<f64>() / N as f64;
            let se = (exact * (1.0 - p) / N as f64).sqrt();
            assert!(
                (m - exact).abs() < 5.0 * se,
                "np={np} (n={n}): mean {m:.4} vs exact {exact:.4}, {:.2} SE — \
                 a step here means one side of the threshold is wrong",
                (m - exact).abs() / se
            );
        }
    }

    /// The gh#510 / gh#525 inputs — huge `n` with tiny `n·p`, and `p` at the top
    /// of its range — must still terminate and stay in support with BTRS
    /// selected. They route to BINV (`n·p < BINV_THRESHOLD`), so this is a
    /// DISPATCH assertion: selecting BTRS must not drag them onto a hat that was
    /// never derived for them, which is how those two defects presented.
    #[test]
    fn pathological_inputs_still_route_to_binv_under_btrs() {
        let mut rng = StatefulRng { inner: ChaCha8Rng::seed_from_u64(11), algo: DEFAULT_BINOMIAL }
            .with_binomial(BinomialAlgorithm::Btrs);
        for (n, p) in [(u64::MAX / 2, 1e-18f64), (2_147_483_648, 1e-12), (6_300_000, 1e-9)] {
            assert!((n as f64) * p < BINV_THRESHOLD, "precondition: this is the BINV regime");
            for _ in 0..500 {
                let k = rng.binomial(n, p);
                assert!(k <= n, "n={n} p={p}: draw {k} out of support");
            }
        }
        // And the reflected top of the range, which flips to a tiny `p_flipped`.
        assert_eq!(rng.binomial(1000, 1.0), 1000);
        let hi = rng.binomial(1000, 1.0 - 1e-9);
        assert!(hi >= 995, "p→1 should draw near n, got {hi}");
    }

    /// The seam's containment argument, as a test rather than a convention.
    ///
    /// `set_binomial_algorithm` is unconditionally `pub` on a `pub mod` (it
    /// cannot be `#[cfg(test)]`: the bench is a separate crate), so "the
    /// thread-local is the only door" rests entirely on no production code
    /// calling it. That is exactly the kind of claim that rots silently — one
    /// future perf experiment in `cli`, and a release binary draws from BTRS
    /// under a BTPE run address, with nothing in the stored artifact recording
    /// which sampler produced it.
    ///
    /// Asserted as a whitelist of FILES rather than by parsing `#[cfg(test)]`

    /// A non-finite `p` must return, under either sampler.
    ///
    /// NaN passes both of `binomial`'s range guards — every NaN comparison is
    /// false — and then fails to route to BINV for the same reason, so before
    /// the `!p.is_finite()` guard it reached BTRS with a NaN hat and span
    /// FOREVER: the support check `k < 0.0 || k > count` does not reject NaN, and
    /// neither the squeeze nor `slow_accepts` can accept it. That is the gh#510
    /// class — 100% CPU, no allocation, no progress, no error — and in release
    /// the only thing in front of it was a `debug_assert!`, i.e. nothing.
    ///
    /// A hang cannot be asserted on directly; this test fails by never
    /// finishing, which is why the guard is placed with its siblings in
    /// `binomial` rather than inside one sampler.
    #[test]
    fn a_non_finite_p_returns_under_both_samplers() {
        for algo in [BinomialAlgorithm::Btpe, BinomialAlgorithm::Btrs] {
            let mut rng = StatefulRng { inner: ChaCha8Rng::seed_from_u64(7), algo: DEFAULT_BINOMIAL }
                .with_binomial(algo);
            for p in [f64::NAN, -f64::NAN] {
                assert_eq!(rng.binomial(1_000, p), 0, "{algo:?}: NaN p must return 0");
            }
            // ±∞ were already handled by the range guards; pin them so a future
            // reshuffle of the guards cannot quietly drop one.
            assert_eq!(rng.binomial(1_000, f64::INFINITY), 1_000, "{algo:?}: +inf");
            assert_eq!(rng.binomial(1_000, f64::NEG_INFINITY), 0, "{algo:?}: -inf");
        }
    }

    /// Above `BTRS_MAX_N`, selecting BTRS must yield BTPE's draws exactly —
    /// `log_bound`'s `(n+1)·ln(…)` term loses its precision as `n·ε`, so the hat
    /// stops dominating and BTRS would return a biased distribution with no
    /// error (measured: mean 8.6% low at `n = u64::MAX`, with a 130σ outlier).
    ///
    /// Asserted as stream equality against BTPE rather than as a moment, because
    /// the property is DISPATCH: the draw must come from the other sampler, not
    /// merely be plausible. The companion assertion below the bound is what
    /// keeps this from passing vacuously by disabling BTRS everywhere.
    #[test]
    fn btrs_de_selects_itself_above_its_max_n() {
        let n_hi = BTRS_MAX_N + 1;
        let p = 1e-6; // n·p well above BINV_THRESHOLD at both n
        assert!((n_hi as f64) * p >= BINV_THRESHOLD, "precondition: not the BINV regime");
        assert_eq!(
            draw(BinomialAlgorithm::Btrs, n_hi, p, 32, 3),
            draw(BinomialAlgorithm::Btpe, n_hi, p, 32, 3),
            "above BTRS_MAX_N the BTRS selection must fall through to BTPE"
        );
        // NON-VACUITY: at the bound itself BTRS is still live, so the equality
        // above is a routing decision and not a dead arm.
        assert_ne!(
            draw(BinomialAlgorithm::Btrs, BTRS_MAX_N, p, 32, 3),
            draw(BinomialAlgorithm::Btpe, BTRS_MAX_N, p, 32, 3),
            "at BTRS_MAX_N, BTRS must still be the sampler"
        );
    }

    /// The two samplers are NOT bit-compatible, and that must be stated as a
    /// test rather than left as an assumption: they are different rejection
    /// schemes reading the same stream, so they accept different draws. If this
    /// ever passes, one of them is not doing what its name says.
    #[test]
    fn the_two_samplers_are_not_bit_compatible() {
        let a = draw(BinomialAlgorithm::Btpe, 1000, 0.5, 64, 5);
        let b = draw(BinomialAlgorithm::Btrs, 1000, 0.5, 64, 5);
        assert_ne!(a, b, "BTPE and BTRS returned identical streams — check the dispatch");
    }

    /// The sweep tests `accept_ratio`; production runs `slow_accepts`. This is
    /// what keeps that from being a hole.
    ///
    /// They are two algebraic rearrangements of one inequality — `v ≤ V(u)` and
    /// `ln(v·α/(a/us²+b)) ≤ log_bound(k)` — and they are NOT collapsible: the log
    /// form is what production wants (no `exp` of a possibly-large number), and
    /// the ratio form is what makes domination expressible as `V ≤ 1`. So the
    /// duplication stays, and instead it gets checked. Without this, a typo in
    /// the production expression alone — `a/us` for `a/us²`, `v/alpha` for
    /// `v*alpha`, a dropped `.ln()` — would leave
    /// `hat_dominates_and_squeeze_is_valid` perfectly green, because that sweep
    /// never calls the expression the sampler actually uses.
    #[test]
    fn the_two_acceptance_forms_agree() {
        const STEPS: usize = 2_000;
        let mut compared = 0usize;
        for &(n, p) in DOMAIN {
            let h = BtrsHat::new(n, p);
            for i in 0..STEPS {
                let u = -0.5 + (i as f64 + 0.5) / STEPS as f64;
                let us = BtrsHat::us_of(u);
                let k = h.k_of(u, us);
                if !h.in_support(k) {
                    continue;
                }
                let ratio = h.accept_ratio(us, k);
                // Probe both sides of the boundary, and skip a narrow band
                // around it where the two forms may legitimately round apart.
                //
                // The 0.999/1.001 rungs are what give this test its resolution.
                // With only ±0.5 and ±10% probes, the two forms could disagree
                // by any factor `f ∈ (0.909, 1.111)` without a single probe
                // straddling the boundary: a 9% bias in `slow_accepts` passed
                // (total variation 3.5e-3, 3.5% max relative pmf error) and 12%
                // was the first caught (gh#802). At ±0.1% the blind window
                // closes to `f ∈ (0.999, 1.001)`.
                for scale in [0.5f64, 0.9, 0.999, 1.001, 1.1, 2.0] {
                    let v = ratio * scale;
                    if !(v > 0.0 && v.is_finite()) {
                        continue;
                    }
                    if (scale - 1.0).abs() < 1e-9 {
                        continue;
                    }
                    assert_eq!(
                        h.slow_accepts(v, us, k),
                        v <= ratio,
                        "n={n} p={p} u={u} k={k}: the log-form test and the ratio \
                         form disagree at v={v} (V={ratio}) — one of the two \
                         expressions is wrong, and the domination sweep only \
                         checks the ratio form"
                    );
                    compared += 1;
                }
            }
        }
        assert!(
            compared > 50_000,
            "only {compared} comparisons — this test has gone vacuous"
        );
    }

    /// The THIRD exactness condition, and the one the domination sweep is
    /// structurally blind to.
    ///
    /// BTRS returns `k` with probability `exp(log_bound(k))/α` — the hat's
    /// Jacobian cancels the proposal density exactly — so exactness requires
    /// `exp(log_bound(k)) ∝ pmf(k)`, on top of domination and squeeze validity.
    /// `hat_dominates_and_squeeze_is_valid` cannot see this: it forms `V` FROM
    /// `log_bound`, so any k-dependent error that LOWERS `log_bound` keeps
    /// `V ≤ 1` and leaves the sweep green while the distribution is wrong.
    ///
    /// This is not hypothetical. Before this test existed, deleting the
    /// `- stirling_approx_tail(k)` term from `log_bound` — the exact shape of
    /// the deviation TensorFlow Probability documents in its own BTRS ("there is
    /// a log missing") — left the ENTIRE suite green, and so did scaling all ten
    /// `TAIL` entries by 1.10, zeroing them, or a one-digit typo in any of them.
    /// All 13 hand-transcribed constants in `stirling_approx_tail` were untested.
    ///
    /// The reference log-pmf is walked by its own recurrence,
    /// `log f(k+1) − log f(k) = ln((n−k)/(k+1)) + ln(p/(1−p))`, rather than from
    /// `lgamma`: at `n ≈ 9e6` the k-dependent `lgamma` terms are ~1e8, and their
    /// rounding alone would swamp the ~1e-9 property under test.
    #[test]
    fn log_bound_is_proportional_to_the_exact_pmf() {
        for &(n, p) in DOMAIN {
            let h = BtrsHat::new(n, p);

            // The `k` the hat can actually return — the only ones whose density
            // matters — collected off the same lattice the sweep uses.
            const STEPS: usize = 20_000;
            let mut ks: Vec<u64> = (0..STEPS)
                .filter_map(|i| {
                    let u = -0.5 + (i as f64 + 0.5) / STEPS as f64;
                    let k = h.k_of(u, 0.5 - u.abs());
                    (k >= 0.0 && k <= h.count).then_some(k as u64)
                })
                .collect();
            ks.sort_unstable();
            ks.dedup();
            assert!(
                ks.len() >= 3,
                "n={n} p={p}: only {} reachable k — the sweep would be vacuous",
                ks.len()
            );

            let log_ratio = (p / (1.0 - p)).ln();
            let mut log_f = 0.0f64; // exact log-pmf, up to a constant
            let mut k_cur = ks[0];
            let mut worst = (f64::INFINITY, f64::NEG_INFINITY);
            for &k in &ks {
                while k_cur < k {
                    log_f += (((n - k_cur) as f64) / ((k_cur + 1) as f64)).ln() + log_ratio;
                    k_cur += 1;
                }
                let d = h.log_bound(k as f64) - log_f;
                worst = (worst.0.min(d), worst.1.max(d));
            }

            // Constant in `k` means the SPREAD is zero; the offset is the
            // normalisation and carries no information.
            let spread = worst.1 - worst.0;
            assert!(
                spread < 1e-7,
                "n={n} p={p}: log_bound − log_pmf varies by {spread:.3e} over \
                 {} reachable k — exp(log_bound) is not proportional to the pmf, \
                 so BTRS samples the wrong distribution here",
                ks.len()
            );
        }
    }

    /// **This is the correctness proof's first two conditions, and they are the
    /// ones the χ² suite cannot replace.**
    ///
    /// BTRS is a rejection sampler, so exactness needs its hat to DOMINATE the
    /// pmf (`V ≤ 1` everywhere) and its squeeze to be VALID (`V ≥ v_r` wherever
    /// the squeeze can fire — otherwise the fast path accepts draws the density
    /// would have rejected). Both are deterministic properties of the eight
    /// constants: no draws, no seed, nothing to flake. Neither is sufficient
    /// alone: see `log_bound_is_proportional_to_the_exact_pmf` for the third
    /// condition, which this sweep cannot detect a violation of.
    ///
    /// Why a distributional test is not enough: a one-digit transcription error
    /// in `b` (`2.53 → 2.63`) distorts the tails symmetrically about the mode,
    /// leaving the mean bias at **exactly zero** while the distribution is
    /// wrong — so `moments_agree_at_the_province_model_regimes` is structurally
    /// blind to it, and `both_samplers_fit_the_exact_pmf` would need ~10^8 draws
    /// to notice. This sweep catches it in milliseconds, and it evaluates
    /// `BtrsHat`'s own methods, so it checks the SHIPPED arithmetic rather than a
    /// second copy of the formula.
    ///
    /// It catches a single-constant typo in `b`, `a`, `c`, `v_r`, `alpha`, `m`,
    /// the squeeze threshold, and the three `log_bound` shifts — but ONLY with
    /// the three adversarial `DOMAIN` cells present; the model-derived cells
    /// alone miss `m`, `v_r` and `alpha`. It does not test
    /// `stirling_approx_tail` at all. Do not restate this as "catches every
    /// single-constant typo": that claim was made here once and was false.
    #[test]
    fn hat_dominates_and_squeeze_is_valid() {
        for &(n, p) in DOMAIN {
            assert!(
                (n as f64) * p >= BINV_THRESHOLD && p <= 0.5,
                "DOMAIN entry ({n}, {p}) is outside what btrs_binomial is handed"
            );
            let h = BtrsHat::new(n, p);
            let (worst_v, overshoot) = worst_ratios(&h);
            assert!(
                worst_v <= 1.0,
                "n={n} p={p}: hat does NOT dominate (max V = {worst_v:.6} > 1) — \
                 the sampler is not exact here"
            );
            assert!(
                overshoot.is_none(),
                "n={n} p={p}: `squeeze_accepts` took v = V·(1 + {:.0e}), a draw the \
                 slow test rejects — the fast path is biased",
                overshoot.unwrap_or(0.0)
            );
        }
    }

    /// The routing predicate is a CORRECTNESS boundary, not a speed knob, and
    /// this test is what says so out loud.
    ///
    /// The domination margin above is thin — **0.22%** at its worst, not the
    /// "few percent" this comment used to claim — and it goes NEGATIVE by
    /// `n·p ≈ 9.64`, only 3.6% below the threshold. The cell asserted below
    /// (`n·p = 7`) exceeds 1 by just 0.29% on the shipped lattice, so it is a
    /// deliberately marginal witness, not a comfortable one. `BINV_THRESHOLD` is
    /// the only thing
    /// keeping BTRS valid, and anyone who lowers it to buy speed breaks
    /// exactness silently (a χ² would need ~10^12 draws to see the resulting
    /// error). Asserting the hat FAILS here pins that reasoning to a red test.
    #[test]
    fn the_hat_stops_dominating_below_the_routing_threshold() {
        let (worst, _) = worst_ratios(&BtrsHat::new(700, 0.01)); // n·p = 7
        assert!(
            worst > 1.0,
            "expected the hat to FAIL below the threshold (n·p = 7), got max V = \
             {worst:.6}. If this now passes, the domination region is wider than \
             assumed — re-derive it before touching BINV_THRESHOLD."
        );
    }

    /// The support guard at the top of [`BtrsHat::propose`] is a deviation from
    /// the reference, which checks the support only after the squeeze. In the
    /// routed domain the guard is a no-op — that is the point of it — so no
    /// draw can distinguish its presence, and deleting it left the suite GREEN
    /// under `--release`. Five tests went red in debug, but only through the
    /// `debug_assert!` beside it, and the gate never builds this module
    /// optimised, which is the build every fit runs (gh#802).
    ///
    /// So test the guard at the level it is written at: hand `propose` a hat
    /// whose geometry puts a squeeze-region candidate outside `[0, n]` and
    /// require the answer to be "redraw". Without the guard the squeeze returns
    /// that candidate — below zero it becomes 0 through the saturating
    /// `as u64`, and above `n` it becomes a `k > n` that `binomial`'s `p > 0.5`
    /// reflection turns into an `n − k` underflow of ~1.8e19.
    #[test]
    fn an_out_of_support_candidate_is_redrawn_not_returned() {
        // Not a routed hat: `a` is inflated so the candidate leaves `[0, n]`
        // while `us` is still inside the squeeze region. The shipped constants
        // make that impossible above `n·p ≈ 1.3`, and the routing predicate
        // admits nothing below `n·p = 10` — which is why no draw can reach here.
        let h = BtrsHat {
            count: 20.0,
            b: 1.0,
            a: 2.0,
            c: 10.0,
            v_r: 0.5,
            r: 1.0,
            alpha: 1.0,
            m: 10.0,
        };
        let v = 0.1; // below v_r, so the squeeze is what would fire

        for (u, edge) in [(0.42f64, "above n"), (-0.42, "below 0")] {
            let us = BtrsHat::us_of(u);
            let k = h.k_of(u, us);
            // The fixture has to actually pose the question, or this test is
            // vacuous: the candidate must be out of support AND the squeeze
            // must be live at this `us`.
            assert!(
                !h.in_support(k),
                "fixture is wrong: k={k} is in support, so the guard is not \
                 exercised {edge}"
            );
            assert!(
                h.squeeze_accepts(v, us),
                "fixture is wrong: the squeeze does not fire at us={us}, so the \
                 guard is not what decides the outcome {edge}"
            );
            assert_eq!(
                h.propose(u, v),
                None,
                "a candidate {edge} (k={k}) was RETURNED instead of redrawn — \
                 the hoisted support check is what stands between the squeeze \
                 and a saturating `as u64`"
            );
        }
    }

    /// The mirror of `the_hat_stops_dominating_below_the_routing_threshold`, for
    /// the other end. [`BTRS_MAX_N`] is a correctness fence, and until this test
    /// existed nothing pinned it: raising it from 1e12 to 1e15, or to 1e18, left
    /// all 24 tests green while the hat stopped dominating and the sampler
    /// silently returned a wrong distribution.
    /// `btrs_de_selects_itself_above_its_max_n` reads the constant symbolically,
    /// so it follows the mutation rather than catching it.
    ///
    /// Two halves, and both are needed. At the fence the hat must still
    /// dominate — that is what fails if the constant is raised. Above it the hat
    /// must FAIL to dominate — that is what says the fence is doing work rather
    /// than being a decoration, and its `n > BTRS_MAX_N` precondition is what
    /// fails if the constant is raised past these cells instead.
    ///
    /// Measured `sup V` on the shipped lattice, over `p ∈ {1e-6 … 0.5}`:
    /// 0.9951–0.9955 at 1e12, 0.9963–0.9967 at 1e13, 1.102–1.140 at 1e15,
    /// 3.59–8.20 at 1e16, 2.4e75–9.1e81 at 1e18. The crossing is between 1e13
    /// and 1e15; 1e12 sits a decade inside it, per the constant's own docstring.
    ///
    /// **These cells are deliberately NOT in [`DOMAIN`].** Domination is only
    /// the first of the three exactness conditions, and the second —
    /// `log_bound_is_proportional_to_the_exact_pmf` — is already violated here:
    /// its `spread < 1e-7` bar is crossed at `n ≈ 4.6e8` (measured spread
    /// 1.02e-7 there, 2.22e-4 at 1e12). That is the SEPARATE, open defect in
    /// gh#802 — the fence is derived from domination but pmf proportionality
    /// binds ~2000× lower — and choosing between the `ln_1p` repair, a lower
    /// fence, and accepting it is a maintainer decision, not something this test
    /// should pre-empt by turning a DOMAIN cell red.
    #[test]
    fn the_hat_stops_dominating_above_btrs_max_n() {
        // At the fence itself, on both edges of the `p` range the router can
        // hand BTRS at this `n`.
        for &p in &[1e-6f64, 0.5] {
            assert!(
                (BTRS_MAX_N as f64) * p >= BINV_THRESHOLD,
                "precondition: (BTRS_MAX_N, {p}) must reach the BTRS branch"
            );
            let (worst, overshoot) = worst_ratios(&BtrsHat::new(BTRS_MAX_N, p));
            assert!(
                worst <= 1.0,
                "at BTRS_MAX_N = {BTRS_MAX_N} with p={p} the hat does NOT dominate \
                 (max V = {worst:.6} > 1) — the fence is above the range the hat \
                 is valid on, so every draw at the top of the routed domain comes \
                 from the wrong distribution"
            );
            assert!(
                overshoot.is_none(),
                "at BTRS_MAX_N = {BTRS_MAX_N} with p={p} the squeeze accepts \
                 v = V·(1 + {:.0e})",
                overshoot.unwrap_or(0.0)
            );
        }
        // And above it, where the fence exists precisely because it does not.
        for &(n, p) in &[
            (1_000_000_000_000_000u64, 1e-6f64),
            (10_000_000_000_000_000, 1e-6),
            (1_000_000_000_000_000_000, 1e-6),
        ] {
            assert!(
                n > BTRS_MAX_N,
                "n={n} is no longer above BTRS_MAX_N = {BTRS_MAX_N}: the fence has \
                 been raised into the region where the hat is known to fail"
            );
            let (worst, _) = worst_ratios(&BtrsHat::new(n, p));
            assert!(
                worst > 1.0,
                "expected the hat to FAIL above the fence at n={n}, got max V = \
                 {worst:.6}. If it now dominates here, the `(n+1)·ln(…)` precision \
                 loss has been repaired — re-derive BTRS_MAX_N before raising it."
            );
        }
    }

    /// The margin at the boundary is small enough to be worth pinning: if a
    /// future edit erodes it, that shows up here before it shows up as a wrong
    /// posterior.
    #[test]
    fn domination_margin_at_the_boundary_is_recorded() {
        let (worst, _) = worst_ratios(&BtrsHat::new(20, 0.5)); // n·p = 10
        assert!(
            worst > 0.90 && worst <= 1.0,
            "boundary margin moved: max V = {worst:.6}, expected just under 1"
        );
    }
}
