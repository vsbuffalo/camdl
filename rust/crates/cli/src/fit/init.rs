//! The numeric builders behind the bare chain-start rules.
//!
//! Three of the [`crate::fit::starts::Spread`] rules are drawn here, each
//! transform-aware via the parameter's [`Transform`]:
//!
//! - **`uniform`** — chain 1 keeps the declared start; chains 2..N draw
//!   uniformly within the natural-scale bounds. Legacy; equivalent to LHS for
//!   `Logit`/`None` parameters but clumps `Log`-typed ones in linear space at
//!   low chain counts. Kept for reproducibility of pre-LHS results.
//! - **`lhs`** — Latin-hypercube stratified sampling. For `Log`-typed
//!   parameters (rates, positive quantities) LHS spans `[ln(lo), ln(hi)]` and
//!   exponentiates back, so one pass covers orders of magnitude rather than
//!   concentrating mass near `hi`; `Logit` and untransformed parameters span
//!   `[lo, hi]` linearly. gh#42: a downstream typhoid fit found 30 LHS chains
//!   reach a basin 80,542 nats better than 8 uniform-random chains, holding
//!   everything else equal.
//! - **`uniform_unconstrained`** — Stan's initialization: an i.i.d. draw
//!   `z ~ Uniform(-2, 2)` on the unconstrained scale, squashed through the
//!   logistic sigmoid and mapped to the natural scale by the same seam LHS
//!   uses, so a start never sits on a bound and the radius is
//!   bounds-independent. The default when a parameter lacks a prior.
//!
//! The dispatcher that chooses among them — and the sourced rules
//! (`from_prior`, `from_posterior`, `from_mle`, `from_params`) — is
//! [`crate::fit::chain_starts::draw_chain_starts`]. Every runner draws its
//! starts through that one seam.

use sim::inference::types::{EstimatedParam, Transform};
use sim::rng::StatefulRng;

use crate::util::derive_chain_seed;

/// Convert per-chain `EstimatedParam` specs into per-chain full
/// parameter vectors. Each chain starts from `base_params`; each
/// `EstimatedParam`-listed index is overwritten with that chain's
/// `initial` value. Shared by the PMMH / PGAS / NUTS / NLopt dispatch sites,
/// which consume `Vec<Vec<f64>>`.
pub fn chain_starts_to_param_vecs(
    chains: &[Vec<EstimatedParam>],
    base_params: &[f64],
) -> Vec<Vec<f64>> {
    chains.iter().map(|chain| {
        let mut params = base_params.to_vec();
        for spec in chain { params[spec.index] = spec.initial; }
        params
    }).collect()
}

/// Per-chain uniform random draw within natural-scale bounds. Chain 0
/// keeps the seeded start (reproducibility); chains 1..N draw fresh.
pub(crate) fn build_uniform_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    (0..n_chains).map(|chain_id| {
        let mut rng = StatefulRng::new(derive_chain_seed(seed, chain_id));
        base.iter().map(|spec| {
            let initial = if chain_id == 0 {
                spec.initial
            } else if spec.lower.is_finite() && spec.upper.is_finite() {
                spec.lower + rng.uniform() * (spec.upper - spec.lower)
            } else {
                spec.initial * (0.5 + rng.uniform())
            };
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Latin-hypercube stratified starts, scale-aware via `Transform`.
///
/// Algorithm (textbook stratified LHS):
/// 1. For each parameter dim d, draw a random permutation π_d of `[0..n_chains)`.
/// 2. For chain k, dim d: `u_{k,d} = (π_d[k] + jitter) / n_chains`, with
///    `jitter ~ Uniform(0, 1)` — a uniform draw within stratum k's cell.
/// 3. Map `u_{k,d}` to natural-scale θ via the parameter's transform:
///    - `Transform::Log` and both bounds positive → exponential mapping
///      `θ = lo · (hi/lo)^u`. Equivalent to LHS in `[ln lo, ln hi]`.
///    - Otherwise (Logit, None, or pathological log bounds) → linear
///      `θ = lo + u · (hi - lo)`.
///
/// Unbounded params (lower or upper non-finite) fall back to a
/// `±50%` jitter around `spec.initial` — same fallback as
/// `build_uniform_chain_starts` for parity. LHS without finite bounds
/// is meaningless; flag with the validator if this matters in practice.
///
/// One RNG seeded from `seed` drives the permutations and the per-stratum
/// jitters, so adding a chain reshuffles every stratum assignment — the
/// price of stratification.
pub(crate) fn build_lhs_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    let n_params = base.len();
    let mut rng = StatefulRng::new(seed ^ 0x1f5_beef_u64);

    // Step 1+2: per-dim permutation, jitter within each stratum.
    // u[chain_id][param_id] is the [0,1] LHS coordinate.
    let mut u: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n_chains];
    for d in 0..n_params {
        let mut perm: Vec<usize> = (0..n_chains).collect();
        // Fisher-Yates using the same RNG (deterministic given seed).
        for i in (1..n_chains).rev() {
            let j = (rng.uniform() * (i as f64 + 1.0)).floor() as usize;
            perm.swap(i, j.min(i));
        }
        for k in 0..n_chains {
            let jitter = rng.uniform();
            u[k][d] = (perm[k] as f64 + jitter) / n_chains as f64;
        }
    }

    // Step 3: map [0,1] LHS coord to natural-scale θ per Transform.
    (0..n_chains).map(|chain_id| {
        base.iter().enumerate().map(|(d, spec)| {
            let initial = lhs_map_to_natural(spec, u[chain_id][d]);
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Stan's default initialization radius on the unconstrained scale: each
/// chain draws `z ~ Uniform(-2, 2)` per parameter (Stan's `init_radius`,
/// mc-stan.org Reference Manual, "Initialization"). Hardcoded for v1.
pub(crate) const STAN_INIT_RADIUS: f64 = 2.0;

/// Stan-style starts: i.i.d. `z ~ Uniform(-R, R)` on the unconstrained
/// scale, squashed to the open unit interval `u = σ(z)` and mapped to
/// natural scale through the same transform-aware seam as LHS
/// ([`lhs_map_to_natural`]).
///
/// Boundary-avoiding (`σ(±2) ≈ (0.119, 0.881)` is a fixed interior band, so
/// starts never sit on a bound — no degenerate `-inf` likelihoods or
/// zero-gradient starts) and scale-invariant (the same radius works whether
/// a parameter is `O(1)` or `O(1e6)`). For a `Logit` parameter it is
/// exactly Stan's `lo + (hi-lo)·σ(z)`; for a `Log` rate it is the
/// camdl-faithful log-scale analog `lo·(hi/lo)^σ(z)`.
///
/// Per-chain RNGs derive from `seed` via `derive_chain_seed` (same as
/// `build_uniform_chain_starts`), so draws are independent across chains
/// and reproducible given the fit seed. Unlike LHS there is no
/// stratification: each `(chain, param)` coordinate is an independent
/// squashed-uniform draw — over-dispersed independent starts are the
/// textbook basis for MCMC convergence diagnostics.
pub(crate) fn build_uniform_unconstrained_chain_starts(
    base: &[EstimatedParam],
    n_chains: usize,
    seed: u64,
) -> Vec<Vec<EstimatedParam>> {
    (0..n_chains).map(|chain_id| {
        let mut rng = StatefulRng::new(derive_chain_seed(seed, chain_id));
        base.iter().map(|spec| {
            // z ~ U(-R, R) on the unconstrained scale; σ(z) lands in the
            // open interior of [0, 1] (σ(±2) ≈ 0.119 / 0.881), so the
            // mapped start never sits on a bound regardless of [lo, hi]
            // width — boundary-avoiding and scale-invariant.
            let z = (rng.uniform() * 2.0 - 1.0) * STAN_INIT_RADIUS;
            let u = 1.0 / (1.0 + (-z).exp());
            let initial = lhs_map_to_natural(spec, u);
            EstimatedParam { initial, ..spec.clone() }
        }).collect()
    }).collect()
}

/// Draw a single Transform-aware value within `[lo, hi]`, suitable for
/// the gh#34 start-fallback path: when an `[estimate]` entry has neither
/// `start =` nor a model-declared parameter `value`, we still need a
/// scalar to seed `model.parameters[i].value` with so that compile +
/// `validate_parameter_values` succeed. Downstream chain init can then
/// perturb from this base.
///
/// "Transform-aware" means: for `Log`-typed parameters with both bounds
/// strictly positive, draw uniformly in *log space* and exponentiate;
/// otherwise draw linearly in `[lo, hi]`. Replaces the legacy
/// bounds-midpoint heuristic (`(lo*hi).sqrt()` or `(lo+hi)/2`), which
/// was geometric-shape-aware via a positive-bounds proxy but ignored
/// the parameter's declared transform and gave the same point at every
/// seed.
///
/// Reproducibility: the per-parameter `u ∈ [0, 1]` is derived from
/// `(seed, param_name)` via a 64-bit hash, so re-running with the same
/// `seed` gives the same start, and two estimate entries with the same
/// bounds at the same seed get *different* draws (their names hash
/// differently). Same seed across runs ⇒ same fallback start; different
/// seeds ⇒ different fallback starts within `[lo, hi]`.
pub fn draw_start_in_bounds(
    lo: f64,
    hi: f64,
    log_scale: bool,
    seed: u64,
    param_name: &str,
) -> f64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    param_name.hash(&mut h);
    // Map the 64-bit hash into u ∈ (0, 1) — open interval, so the
    // log-scale branch's `(hi/lo).powf(u)` never lands exactly on a
    // bound. 53-bit mantissa is plenty.
    let u = ((h.finish() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;

    if log_scale && lo > 0.0 && hi > 0.0 {
        lo * (hi / lo).powf(u)
    } else {
        lo + u * (hi - lo)
    }
}

/// Map an LHS coordinate `u ∈ [0, 1]` to the natural-scale parameter
/// value, respecting the parameter's transform.
fn lhs_map_to_natural(spec: &EstimatedParam, u: f64) -> f64 {
    if !spec.lower.is_finite() || !spec.upper.is_finite() {
        // Unbounded: ±50% jitter around the seeded start. LHS is meaningless
        // here but we don't want to fail — the upstream validator should
        // refuse fits with unbounded estimated params; until that lands,
        // fall back gracefully.
        return spec.initial * (0.5 + u);
    }
    match &spec.transform {
        Transform::Log { .. } if spec.lower > 0.0 && spec.upper > 0.0 => {
            // LHS in log space: θ = lo · (hi/lo)^u
            spec.lower * (spec.upper / spec.lower).powf(u)
        }
        _ => {
            // Linear LHS in [lo, hi]
            spec.lower + u * (spec.upper - spec.lower)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::types::Transform;

    fn ep(name: &str, lower: f64, upper: f64, transform: Transform, initial: f64) -> EstimatedParam {
        EstimatedParam {
            name: name.into(),
            index: 0,
            initial,
            rw_sd: 0.1,
            transform,
            lower,
            upper,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }
    }

    #[test]
    fn lhs_strata_cover_range_uniformly() {
        // 100 chains × 1 param ∈ [0, 1] linear: every decile should
        // contain ~10 starts (LHS guarantee at this resolution).
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let starts = build_lhs_chain_starts(&base, 100, 42);
        let values: Vec<f64> = starts.iter().map(|c| c[0].initial).collect();

        let mut bin_counts = vec![0usize; 10];
        for &v in &values {
            let bin = ((v * 10.0) as usize).min(9);
            bin_counts[bin] += 1;
        }
        // LHS guarantees exactly one sample per stratum at the dim level.
        // With 100 chains and 10 bins, each stratum aligns 10:1 with bins.
        for &c in &bin_counts {
            assert!(c >= 8 && c <= 12,
                "LHS strata uneven: counts = {:?}", bin_counts);
        }
    }

    #[test]
    fn lhs_log_param_spans_orders_of_magnitude() {
        // Log-typed param with bounds [1e-5, 1e-2] should LHS in log space.
        // The geomean of all draws should be near sqrt(1e-5 * 1e-2) = 1e-3.5
        // and the spread should be the full range — not concentrated near 1e-2.
        let base = vec![ep("rate", 1e-5, 1e-2, Transform::Log { lo: 1e-5, hi: 1e-2 }, 1e-3)];
        let starts = build_lhs_chain_starts(&base, 50, 42);
        let values: Vec<f64> = starts.iter().map(|c| c[0].initial).collect();

        // Distribute roughly evenly across each decade.
        let log_vals: Vec<f64> = values.iter().map(|v| v.log10()).collect();
        let mean = log_vals.iter().sum::<f64>() / log_vals.len() as f64;
        // log10(1e-5) = -5, log10(1e-2) = -2, midpoint = -3.5
        assert!((mean - (-3.5)).abs() < 0.3,
            "log-LHS mean = {} (expected ~−3.5)", mean);

        let lo_count = values.iter().filter(|&&v| v < 1e-4).count();
        let hi_count = values.iter().filter(|&&v| v > 1e-3).count();
        // With LHS in log space, mass spreads across decades; uniform
        // (linear) sampling would cluster near 1e-2 with very few < 1e-4.
        assert!(lo_count >= 5 && hi_count >= 5,
            "log-LHS clusters: lo<1e-4={} hi>1e-3={} (linear sampling would skew here)",
            lo_count, hi_count);
    }

    #[test]
    fn lhs_deterministic_given_seed() {
        let base = vec![
            ep("a", 0.0, 1.0, Transform::None, 0.5),
            ep("b", 1e-3, 1.0, Transform::Log { lo: 1e-3, hi: 1.0 }, 0.1),
        ];
        let s1 = build_lhs_chain_starts(&base, 16, 42);
        let s2 = build_lhs_chain_starts(&base, 16, 42);
        for (c1, c2) in s1.iter().zip(s2.iter()) {
            for (p1, p2) in c1.iter().zip(c2.iter()) {
                assert_eq!(p1.initial, p2.initial);
            }
        }
    }

    #[test]
    fn uniform_unconstrained_is_boundary_avoiding() {
        // z ~ U(-2,2) ⇒ σ(z) ∈ (σ(-2), σ(2)) ≈ (0.1192, 0.8808), a fixed
        // interior band independent of the bounds. For a linear param on
        // [lo,hi] every start falls strictly inside the band — never on a
        // bound.
        let (lo, hi) = (0.0_f64, 1.0_f64);
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let band_lo = lo + (hi - lo) * sig(-STAN_INIT_RADIUS);
        let band_hi = lo + (hi - lo) * sig(STAN_INIT_RADIUS);
        let base = vec![ep("p", lo, hi, Transform::None, 0.5)];
        let starts = build_uniform_unconstrained_chain_starts(&base, 200, 7);
        for c in &starts {
            let v = c[0].initial;
            assert!(v > band_lo - 1e-12 && v < band_hi + 1e-12,
                "start {v} escaped the σ(±2) interior band [{band_lo}, {band_hi}]");
            assert!(v > lo && v < hi, "start {v} on/outside bound [{lo}, {hi}]");
        }
    }

    #[test]
    fn uniform_unconstrained_log_param_stays_in_log_interior() {
        // Log param [1e-5, 1e-2]: θ = lo·(hi/lo)^σ(z). log10-interior band
        // is [-5 + 3·σ(-2), -5 + 3·σ(2)] = [-4.64, -2.36] — strictly inside
        // the declared decades.
        let base = vec![ep("rate", 1e-5, 1e-2, Transform::Log { lo: 1e-5, hi: 1e-2 }, 1e-3)];
        let starts = build_uniform_unconstrained_chain_starts(&base, 200, 7);
        for c in &starts {
            let l = c[0].initial.log10();
            assert!(l > -4.65 && l < -2.35,
                "log10(start) = {l} escaped the σ(±2) log-interior [-4.64, -2.36]");
        }
    }

    #[test]
    fn uniform_unconstrained_deterministic_and_iid() {
        let base = vec![
            ep("a", 0.0, 1.0, Transform::None, 0.5),
            ep("b", 1e-3, 1.0, Transform::Log { lo: 1e-3, hi: 1.0 }, 0.1),
        ];
        let s1 = build_uniform_unconstrained_chain_starts(&base, 16, 42);
        let s2 = build_uniform_unconstrained_chain_starts(&base, 16, 42);
        for (c1, c2) in s1.iter().zip(s2.iter()) {
            for (p1, p2) in c1.iter().zip(c2.iter()) {
                assert_eq!(p1.initial, p2.initial);
            }
        }
        // i.i.d. across chains: chain 0 and chain 1 differ on both params.
        assert_ne!(s1[0][0].initial, s1[1][0].initial);
        assert_ne!(s1[0][1].initial, s1[1][1].initial);
    }

    #[test]
    fn uniform_keeps_chain_one_at_the_declared_start() {
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let starts = build_uniform_chain_starts(&base, 4, 42);
        assert_eq!(starts[0][0].initial, 0.5);
        for c in &starts[1..] {
            assert!(c[0].initial >= 0.0 && c[0].initial <= 1.0);
        }
    }

    #[test]
    fn lhs_different_seed_gives_different_draws() {
        let base = vec![ep("a", 0.0, 1.0, Transform::None, 0.5)];
        let s1 = build_lhs_chain_starts(&base, 16, 42);
        let s2 = build_lhs_chain_starts(&base, 16, 43);
        let differs = s1.iter().zip(s2.iter())
            .any(|(c1, c2)| c1[0].initial != c2[0].initial);
        assert!(differs, "LHS with different seeds returned identical draws");
    }

    #[test]
    fn lhs_within_bounds() {
        let base = vec![
            ep("rate",  1e-5, 1.0, Transform::Log   { lo: 1e-5, hi: 1.0 }, 0.01),
            ep("prob",  0.05, 0.95, Transform::Logit { lo: 0.05, hi: 0.95 }, 0.5),
            ep("real", -10.0, 10.0, Transform::None,                          0.0),
        ];
        let starts = build_lhs_chain_starts(&base, 32, 7);
        for chain in &starts {
            for spec in chain {
                assert!(spec.initial >= spec.lower && spec.initial <= spec.upper,
                    "{} out of bounds: {} not in [{}, {}]",
                    spec.name, spec.initial, spec.lower, spec.upper);
            }
        }
    }

    // ── draw_start_in_bounds (gh#34 fallback) ────────────────────────

    #[test]
    fn draw_start_log_scale_lands_inside_positive_bounds() {
        // Log-scale draw across six orders of magnitude: result must
        // be strictly inside (lo, hi) and stay positive.
        let v = draw_start_in_bounds(1e-6, 1.0, true, 42, "beta");
        assert!(v > 1e-6 && v < 1.0, "{} not in (1e-6, 1.0)", v);
        assert!(v.is_finite() && v > 0.0);
    }

    #[test]
    fn draw_start_linear_scale_lands_inside_bounds() {
        // Linear draw on a real-valued parameter (Logit/None analogue):
        // negative-to-positive bounds, no log-scale possible.
        let v = draw_start_in_bounds(-10.0, 10.0, false, 42, "drift");
        assert!(v > -10.0 && v < 10.0, "{} not in (-10, 10)", v);
    }

    #[test]
    fn draw_start_log_falls_back_to_linear_when_lo_nonpositive() {
        // log_scale=true but lo=0 — helper must NOT call powf on zero
        // (would yield 0 always or NaN); falls back to linear.
        let v = draw_start_in_bounds(0.0, 1.0, true, 42, "p");
        assert!(v > 0.0 && v < 1.0, "{} not in (0, 1)", v);
    }

    #[test]
    fn draw_start_deterministic_per_seed_and_name() {
        // Same (seed, name) ⇒ same draw.
        let a = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        assert_eq!(a, b);
    }

    #[test]
    fn draw_start_different_names_give_different_draws() {
        // Two parameters with identical bounds at the same seed must
        // not collide (would defeat the point of the per-name hash).
        let a = draw_start_in_bounds(1e-3, 1.0, true, 7, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 7, "gamma");
        assert_ne!(a, b);
    }

    #[test]
    fn draw_start_different_seeds_give_different_draws() {
        // Reseeding the run shifts the fallback (so users get spread
        // across seed sweeps, unlike the old midpoint heuristic which
        // gave the same point at every seed).
        let a = draw_start_in_bounds(1e-3, 1.0, true, 1, "beta");
        let b = draw_start_in_bounds(1e-3, 1.0, true, 2, "beta");
        assert_ne!(a, b);
    }

    #[test]
    fn draw_start_log_scale_spans_orders_of_magnitude() {
        // Across many seeds, log-scale draws on (1e-6, 1.0) should
        // populate at least three different decade buckets — the prior
        // midpoint would have given 1e-3 at every seed.
        use std::collections::HashSet;
        let mut decades: HashSet<i32> = HashSet::new();
        for seed in 0..64u64 {
            let v = draw_start_in_bounds(1e-6, 1.0, true, seed, "beta");
            decades.insert(v.log10().floor() as i32);
        }
        assert!(decades.len() >= 3,
            "expected ≥3 decades populated across 64 seeds, got {}: {:?}",
            decades.len(), decades);
    }

    #[test]
    fn chain_starts_to_param_vecs_overwrites_estimated_indices() {
        let base_specs = vec![
            ep_with_idx("beta",  0, 0.0, 1.0, Transform::None, 0.5),
            ep_with_idx("gamma", 2, 0.0, 1.0, Transform::None, 0.3),
        ];
        // Two chains, two estimated indices (0 and 2 of a 4-slot vector).
        let chains = vec![
            vec![
                EstimatedParam { initial: 0.1, ..base_specs[0].clone() },
                EstimatedParam { initial: 0.2, ..base_specs[1].clone() },
            ],
            vec![
                EstimatedParam { initial: 0.7, ..base_specs[0].clone() },
                EstimatedParam { initial: 0.8, ..base_specs[1].clone() },
            ],
        ];
        let base_params = vec![999.0, 11.0, 999.0, 22.0];
        let out = chain_starts_to_param_vecs(&chains, &base_params);
        assert_eq!(out.len(), 2);
        // Chain 0: positions 0/2 overwritten, 1/3 untouched.
        assert_eq!(out[0], vec![0.1, 11.0, 0.2, 22.0]);
        assert_eq!(out[1], vec![0.7, 11.0, 0.8, 22.0]);
    }

    fn ep_with_idx(
        name: &str, index: usize, lower: f64, upper: f64,
        transform: Transform, initial: f64,
    ) -> EstimatedParam {
        EstimatedParam {
            name: name.into(),
            index,
            initial,
            rw_sd: 0.1,
            transform,
            lower,
            upper,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }
    }
}
