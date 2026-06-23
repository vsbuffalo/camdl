use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson, Exp, Gamma, Binomial, StandardNormal};

/// Stateful RNG wrapping ChaCha8. Deterministic given seed.
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
