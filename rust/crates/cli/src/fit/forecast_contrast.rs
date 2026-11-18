//! The difference between two forecast arms (D83).
//!
//! A scenario forecast publishes each arm's full predictive distribution and,
//! per target, the difference of the predictive means
//! `Δ = E[Y_A | D] − E[Y_B | D]` with its Monte Carlo standard error. No band
//! of paired differences `A[s] − B[s]` is published: its spread is set by how
//! the arms' random numbers are coupled, which the separate scenario models do
//! not identify.
//!
//! The arms are coupled: posterior draw `m` runs with the same seed in every
//! arm, each particle with its own random stream, so the arms share the
//! filter up to the origin and particle `i` of draw `m` starts both arms'
//! forecasts from the same state. The draws are not independent of one
//! another, either: they come from Markov chains. The standard error of `Δ`
//! therefore treats neither the particles nor the draws as independent
//! samples. Per draw, the particles are averaged first,
//! `d_m = mean_i A[m, i] − mean_i B[m, i]`, so the shared posterior draw and
//! the several particles per draw collapse into one number per draw; `Δ` is
//! the mean of the `d_m` (equal to the difference of the arms' pooled means,
//! since every draw has the same number of particles). Per chain `c`, with
//! `n_c` of the forecast's draws in chain order, the variance of the chain's
//! mean of `d` is `s²_c / ESS_c`, with `s²_c` the sample variance and `ESS_c`
//! the effective sample size of Geyer's initial-positive-sequence estimator
//! ([`sim::inference::pmmh::mcmc_ess`], the one the fit reports per chain).
//! The chains are independent, so
//! `Var(Δ) = Σ_c (n_c / M)² · s²_c / ESS_c` over the `M` draws.
//!
//! This is Monte Carlo error: how far `Δ` could move were the forecast rerun
//! with other random numbers and other posterior draws. It is not the
//! predictive uncertainty of either arm and not the uncertainty of an
//! intervention's effect.

use std::collections::BTreeMap;

/// The fewest effective draws a chain must carry for its contribution to the
/// standard error to be estimated. Below this the autocorrelation sum behind
/// the ESS rests on too few effective draws to be trusted, and the standard
/// error is withheld rather than understated.
pub(crate) const CONTRAST_MIN_ESS: f64 = 10.0;

/// Why no standard error is given.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NoStandardError {
    /// A chain holds one of the forecast's draws, so its variance cannot be
    /// estimated.
    SingleDrawChain { chain: Option<usize> },
    /// A chain's per-draw differences vary but carry fewer than
    /// [`CONTRAST_MIN_ESS`] effective draws.
    TooFewEffectiveDraws { chain: Option<usize>, ess: f64 },
    /// A per-draw difference is not finite.
    NonFinite,
}

impl NoStandardError {
    pub(crate) fn describe(&self) -> String {
        let name = |c: &Option<usize>| match c {
            Some(c) => format!("chain {c}"),
            None => "the draws".to_string(),
        };
        match self {
            NoStandardError::SingleDrawChain { chain } => format!(
                "{} holds a single forecast draw; raise --n-draws", name(chain)),
            NoStandardError::TooFewEffectiveDraws { chain, ess } => format!(
                "{} carries {ess:.1} effective draws, fewer than {CONTRAST_MIN_ESS}; raise --n-draws",
                name(chain)),
            NoStandardError::NonFinite => "a per-draw difference is not finite".to_string(),
        }
    }
}

/// `Δ` at one target and its Monte Carlo standard error.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeanDifference {
    pub delta: f64,
    pub mcse: Result<f64, NoStandardError>,
    /// `Σ_c ESS_c` over the chains whose differences vary; `M` draws when
    /// none does.
    pub ess: f64,
}

/// Per draw, the mean over its particles of `a` minus that of `b`. Both hold
/// `n_particles` values per draw, draw-major (`s = draw · n_particles + i`).
pub(crate) fn per_draw_differences(a: &[f64], b: &[f64], n_particles: usize) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "two arms of one forecast hold the same number of samples");
    let n = n_particles as f64;
    a.chunks(n_particles).zip(b.chunks(n_particles))
        .map(|(x, y)| x.iter().sum::<f64>() / n - y.iter().sum::<f64>() / n)
        .collect()
}

/// `Δ` and its standard error from the per-draw differences `d`, with
/// `chains[m]` the chain of draw `m` (`None` for a cloud with no chain key,
/// taken as one chain in its row order). Within a chain the draws are taken
/// in the order given, which is the chain's order.
pub(crate) fn paired_mean_difference(d: &[f64], chains: &[Option<usize>]) -> MeanDifference {
    assert_eq!(d.len(), chains.len());
    let m = d.len() as f64;
    let delta = d.iter().sum::<f64>() / m;
    if d.iter().any(|x| !x.is_finite()) {
        return MeanDifference { delta, mcse: Err(NoStandardError::NonFinite), ess: f64::NAN };
    }
    let mut by_chain: BTreeMap<Option<usize>, Vec<f64>> = BTreeMap::new();
    for (&x, &c) in d.iter().zip(chains) {
        by_chain.entry(c).or_default().push(x);
    }
    let mut var = 0.0;
    let mut ess_total = 0.0;
    let mut varying = false;
    for (chain, xs) in &by_chain {
        let n = xs.len() as f64;
        if xs.len() < 2 {
            return MeanDifference { delta, mcse: Err(NoStandardError::SingleDrawChain { chain: *chain }), ess: f64::NAN };
        }
        let mean = xs.iter().sum::<f64>() / n;
        let s2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        if s2 == 0.0 {
            // A chain whose differences are all one value contributes no
            // Monte Carlo variance (two arms identical at this target).
            ess_total += n;
            continue;
        }
        varying = true;
        let ess = sim::inference::pmmh::mcmc_ess(xs);
        if ess.is_nan() || ess < CONTRAST_MIN_ESS {
            return MeanDifference {
                delta, mcse: Err(NoStandardError::TooFewEffectiveDraws { chain: *chain, ess }), ess: f64::NAN,
            };
        }
        ess_total += ess;
        var += (n / m).powi(2) * s2 / ess;
    }
    MeanDifference { delta, mcse: Ok(var.sqrt()), ess: if varying { ess_total } else { m } }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator for the tests (xorshift64* and
    /// Box–Muller), independent of the code under test.
    struct Gen(u64);
    impl Gen {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// The particles are averaged within a draw before anything else, and
    /// `Δ` is the difference of the arms' pooled means.
    #[test]
    fn the_particles_of_a_draw_are_averaged_first() {
        let a = [1.0, 3.0, 10.0, 14.0];
        let b = [0.0, 2.0, 4.0, 4.0];
        assert_eq!(per_draw_differences(&a, &b, 2), vec![1.0, 8.0]);
        let r = paired_mean_difference(&[1.0, 8.0], &[Some(0), Some(0)]);
        assert_eq!(r.delta, (a.iter().sum::<f64>() - b.iter().sum::<f64>()) / 4.0);
    }

    /// Two arms identical at a target have `Δ = 0` and no Monte Carlo error.
    #[test]
    fn identical_arms_differ_by_zero_with_zero_error() {
        let r = paired_mean_difference(&[0.0; 6], &[Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)]);
        assert_eq!((r.delta, r.mcse.clone()), (0.0, Ok(0.0)));
    }

    /// Too little to estimate is said, not understated.
    #[test]
    fn a_standard_error_is_withheld_when_a_chain_cannot_carry_one() {
        let r = paired_mean_difference(&[5.0, 5.0, 3.0], &[Some(0), Some(0), Some(1)]);
        assert_eq!(r.mcse, Err(NoStandardError::SingleDrawChain { chain: Some(1) }));
        let r = paired_mean_difference(&[1.0, 2.0, 3.0, 4.0], &[Some(0); 4]);
        assert!(matches!(r.mcse, Err(NoStandardError::TooFewEffectiveDraws { chain: Some(0), .. })), "{r:?}");
    }

    /// The standard error against an independent measurement of the same
    /// quantity: the spread of `Δ` over many independent repetitions of a
    /// forecast. Each repetition has two chains of 300 draws whose effect
    /// (the per-draw difference of the arms' expected values) follows an
    /// AR(1) process with coefficient 0.6, as a Markov chain's draws do, and
    /// 50 particles per draw per arm whose noise is shared between the arms
    /// in part (coupled streams) and partly not. The reported standard error
    /// averages to the observed standard deviation of `Δ` within 10%.
    /// Treating every particle as an independent sample would understate it
    /// about tenfold, and treating the draws as independent about twofold:
    /// the test tells the right answer from both wrong ones.
    #[test]
    fn the_standard_error_matches_the_spread_of_delta_over_repetitions() {
        let (chains, per_chain, particles, phi) = (2usize, 300usize, 50usize, 0.6f64);
        let mut g = Gen(0x9E37_79B9_7F4A_7C15);
        let (mut deltas, mut ses, mut naive_particles, mut naive_draws) = (vec![], vec![], vec![], vec![]);
        for _ in 0..400 {
            let (mut a, mut b, mut keys) = (vec![], vec![], vec![]);
            for c in 0..chains {
                let mut x: f64 = g.normal() / (1.0 - phi * phi).sqrt();
                for _ in 0..per_chain {
                    x = phi * x + g.normal();
                    let level = 3.0 * g.normal(); // the draw's own level, shared by both arms
                    for _ in 0..particles {
                        let shared = 4.0 * g.normal(); // coupled particle noise
                        a.push(10.0 + level + x + shared + g.normal());
                        b.push(8.0 + level + shared + g.normal());
                    }
                    keys.push(Some(c));
                }
            }
            let d = per_draw_differences(&a, &b, particles);
            let r = paired_mean_difference(&d, &keys);
            deltas.push(r.delta);
            ses.push(r.mcse.unwrap());
            let diffs: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
            naive_particles.push(sd(&diffs) / (diffs.len() as f64).sqrt());
            naive_draws.push(sd(&d) / (d.len() as f64).sqrt());
        }
        let observed = sd(&deltas);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let reported = mean(&ses);
        assert!((reported / observed - 1.0).abs() < 0.10, "reported {reported}, observed {observed}");
        assert!(mean(&naive_particles) < 0.2 * observed, "per-particle SE {}", mean(&naive_particles));
        assert!(mean(&naive_draws) < 0.67 * observed, "per-draw SE {}", mean(&naive_draws));
    }

    fn sd(v: &[f64]) -> f64 {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    }
}
