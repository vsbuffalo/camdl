//! Likelihood-noise preflight for the pseudo-marginal samplers: what the
//! particle filter's log-likelihood spread is, and what acceptance rate that
//! spread allows the proposal adaptation to reach.
//!
//! Fix 3 of `docs/dev/proposals/2026-08-28-pmmh-proposal-adaptation.md`,
//! together with gh#764 (the measured spread is persisted, not printed and
//! dropped).
//!
//! # The quantities
//!
//! - `sigma` — the standard deviation of one estimated log-likelihood
//!   `log L̂` at a fixed parameter vector. It scales as `1/sqrt(N)` in the
//!   particle count, which is why a spread reported without its particle
//!   count is not a number anyone can act on.
//! - `s` — the standard deviation of the *difference*
//!   `log L̂(θ') − log L̂(θ)`. This is what enters the Metropolis ratio, so it
//!   is what governs acceptance. For two independent evaluations at the *same*
//!   noise level `s = sigma·sqrt(2)`; under correlated pseudo-marginal (`rho`
//!   set) the two evaluations share most of their randomness and `s` is
//!   smaller. Neither identity is assumed here — `s` is measured.
//! - `d` — the number of estimated parameters.
//!
//! `s` is generally *not* `sigma·sqrt(2)` even for plain PMMH, and the reason
//! is worth stating because it looks like a defect. The two evaluations of a
//! pair sit at different θ, and the filter's noise level varies across the
//! parameter space: for independent evaluations
//! `s = sqrt(sigma_θ² + sigma_θ'²)`. Measured on a two-parameter SIR with 20
//! daily observations at 200 particles: `sigma` 4.07 at the base θ, `s` 9.40
//! across the step, implying `sigma_θ'` of 8.5 — the initial proposal is a
//! large move and the filter is about twice as noisy where it lands. That is a
//! fact about the run, and a ceiling computed from it errs toward saying the
//! chain cannot accept, which is the safe direction.
//!
//! # The check
//!
//! In the small-step limit a pseudo-marginal random-walk Metropolis accepts at
//! most `2·Φ(−s/2)`: a vanishing step cannot beat a current state whose
//! likelihood estimate came out high by chance. The Robbins–Monro scale
//! adaptation drives toward `0.234 + 0.206/d`, the optimal-scaling target for
//! an *exact* likelihood. When the ceiling sits below that target the
//! recursion has no root and `log λ` falls for the whole run.
//!
//! So the preflight computes both numbers and says which side of the ceiling
//! the run sits on. It does not compare `sigma` against a hand-tuned band.
//!
//! # Three constraints on the measurement
//!
//! Each came out of review of the proposal, and each is load-bearing.
//!
//! 1. **`s` is measured, never derived from `rho`.** The identity
//!    `Var = 2·sigma²·(1 − rho_ll)` holds for `rho_ll`, the correlation of the
//!    log-likelihood *estimates*, which equals the Crank–Nicolson parameter
//!    only if the estimator is a linear Gaussian functional of the auxiliary
//!    variables. On a realistic skewed estimator a Crank–Nicolson parameter of
//!    0.90 induced `rho_ll = 0.81`, and substituting the former understated
//!    `s` by 36%. [`measure`] therefore evaluates each pair and takes
//!    `sd(difference)` directly.
//! 2. **The ceiling assumes `log L̂` is approximately Gaussian, and the error
//!    is one-sided.** Under skew the predicted ceiling is *optimistic*, which
//!    is the wrong direction for a check whose job is to catch collapse. The
//!    approximation is safe once the noise is a sum over many observation
//!    times (measured relative error: 54% at 1 observation, 9% at 5, under 1%
//!    at 20), so [`report`] states the assumption alongside the observation
//!    count instead of presenting the ceiling as unconditional.
//! 3. **A base-point measurement is a best case.** The correlation a
//!    correlated-pseudo-marginal run realises also depends on `θ` moving, and
//!    the achievable correlation degrades with the dimension of the auxiliary
//!    variable (Deligiannidis, Doucet & Pitt 2018, *JRSS-B* 80(5):839–870), so
//!    the second evaluation of each pair is taken at a `θ'` drawn from the
//!    initial proposal rather than at `θ` (see [`draw_proposed_theta`]).

use serde::{Deserialize, Serialize};

use sim::inference::correlated_pf::{self, PFRandomState};
use sim::inference::if2::EstimatedParam;
use sim::rng::StatefulRng;

use super::runner::{self, FitRunConfig};

/// Evaluation pairs the preflight measures.
///
/// A spread of differences needs pairs, so this is `2 × NOISE_PAIRS` filter
/// evaluations — double the single-θ replicate loop it replaces. Twenty is the
/// proposal's recommendation: it puts the standard error of `s` at
/// `1/sqrt(2·19) ≈ 16%`, and fewer pairs trades directly against that.
pub const NOISE_PAIRS: usize = 20;

/// Observation times above which `2·Φ(−s/2)`'s Gaussian assumption is tight.
///
/// The estimator's log-likelihood is a sum of per-window increments, so it
/// approaches normality in the number of observation times. Measured relative
/// error of the ceiling under a realistic skewed estimator: 54% at 1
/// observation, 9% at 5, under 1% at 20 (proposal, fix 3). Below this the
/// ceiling is reported as an upper bound rather than an estimate.
const GAUSSIAN_OBS_FLOOR: usize = 20;

/// Stream offset for the `θ'` draw, so the parameter perturbation cannot
/// collide with any particle-filter seed derived from the same stage seed.
const THETA_PRIME_STREAM: u64 = 0x9E37_79B9_7F4A_7C15;

/// The measured likelihood noise for one pseudo-marginal stage, persisted with
/// the stage artifact (gh#764).
///
/// Everything here is needed to read `sigma` and `s`: the particle count
/// because `sigma ∝ 1/sqrt(N)`, the pair count because it sets the standard
/// error of both spreads, and `rho` because `s` is a property of the scheme
/// and not only of the model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PfNoiseCheck {
    /// Standard deviation of a single `log L̂` at the base θ.
    pub sigma: f64,
    /// Standard deviation of `log L̂(θ') − log L̂(θ)` under this stage's
    /// scheme — correlated when `rho` is set, independent otherwise.
    pub s: f64,
    /// Particles each evaluation used. `sigma` is meaningless without it.
    pub n_particles: usize,
    /// Evaluation pairs behind both spreads.
    pub pairs: usize,
    /// Crank–Nicolson parameter in force, `None` for plain PMMH.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho: Option<f64>,
}

impl PfNoiseCheck {
    /// `2·Φ(−s/2)` — the acceptance rate this spread allows.
    pub fn ceiling(&self) -> f64 {
        acceptance_ceiling(self.s)
    }

    /// One standard error on `s`.
    pub fn s_standard_error(&self) -> f64 {
        spread_standard_error(self.s, self.pairs)
    }
}

/// The acceptance rate a pseudo-marginal random-walk Metropolis tends to as
/// the proposal scale goes to zero: `2·Φ(−s/2)`.
///
/// Equivalently `2·Φ(−sigma/sqrt(2))` for two independent evaluations; the `s`
/// form is the one implemented against because it is scheme-agnostic. Verified
/// by direct Monte Carlo of the noise-only chain (proposal, "Mechanism"):
/// 0.4795 against 0.4805/0.4807 at `sigma = 1`, 0.1573 against 0.1561/0.1555
/// at `sigma = 2`.
pub fn acceptance_ceiling(s: f64) -> f64 {
    if !s.is_finite() || s < 0.0 {
        return f64::NAN;
    }
    2.0 * sim::inference::normal_cdf(-s / 2.0)
}

/// The target the Robbins–Monro scale adaptation drives toward:
/// `0.234 + 0.206/d`.
///
/// This mirrors `AdaptiveProposal::target_accept` in `sim`, which is the code
/// the check is about; the two must agree or the preflight reports a ceiling
/// against a target the sampler does not use.
pub fn target_acceptance(d: usize) -> f64 {
    0.234 + 0.206 / d.max(1) as f64
}

/// The spread at which the ceiling equals the target — above it the
/// Robbins–Monro recursion has no root.
///
/// Solved in closed form from [`acceptance_ceiling`]: `2·Φ(−s/2) = a*` gives
/// `s = −2·Φ⁻¹(a*/2)`. Dividing by `sqrt(2)` recovers the `sigma` crossovers
/// the proposal tabulates (1.565 at `d = 6`, 1.640 at `d = 17`, asymptotically
/// 1.683).
pub fn crossover_s(d: usize) -> f64 {
    -2.0 * sim::inference::normal_quantile(target_acceptance(d) / 2.0)
}

/// One standard error of a standard deviation estimated from `n` values:
/// `s/sqrt(2(n−1))`.
///
/// The large-sample normal result. At the 20 pairs this preflight takes that
/// is 16% of `s` — one standard error, roughly 68% coverage, not a confidence
/// interval.
pub fn spread_standard_error(s: f64, n: usize) -> f64 {
    if n < 2 {
        return f64::NAN;
    }
    s / (2.0 * (n - 1) as f64).sqrt()
}

/// Which side of the acceptance ceiling this run sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeilingVerdict {
    /// The ceiling is above the target: a proposal scale achieving the target
    /// exists, so the adaptation has a root to converge to.
    Reachable,
    /// The ceiling is below the target: no proposal scale reaches it, so
    /// `log λ` falls for the whole run.
    Unreachable,
    /// The interval on `s` straddles the crossover, so the measurement does
    /// not separate the two.
    Unresolved,
}

/// Classify `s` against the crossover, refusing to pick a side when one
/// standard error covers it.
///
/// The band is one standard error rather than the wider 95% read that
/// [`report`] also prints: at 20 pairs a two-standard-error band is ±32% of
/// `s`, which would return `Unresolved` for most realistic fits and say
/// nothing. One standard error asserts a side only when the measurement is at
/// least that far from the crossover, and the printed 95% half-width lets a
/// reader see how much wider the honest interval is.
pub fn ceiling_verdict(s: f64, se: f64, d: usize) -> CeilingVerdict {
    let cross = crossover_s(d);
    if s + se < cross {
        CeilingVerdict::Reachable
    } else if s - se > cross {
        CeilingVerdict::Unreachable
    } else {
        CeilingVerdict::Unresolved
    }
}

/// Reduce paired evaluations to the two spreads.
///
/// `None` when there are fewer than two pairs or any evaluation is
/// non-finite: a θ the filter rules out (`−∞`) has no spread, and reporting a
/// `NaN` as a measurement would be worse than reporting nothing.
pub fn summarize(
    base_lls: &[f64],
    proposed_lls: &[f64],
    n_particles: usize,
    rho: Option<f64>,
) -> Option<PfNoiseCheck> {
    let n = base_lls.len();
    if n < 2 || proposed_lls.len() != n {
        return None;
    }
    if base_lls.iter().chain(proposed_lls).any(|l| !l.is_finite()) {
        return None;
    }
    let diffs: Vec<f64> = base_lls.iter().zip(proposed_lls)
        .map(|(&b, &p)| p - b)
        .collect();
    Some(PfNoiseCheck {
        sigma: sample_sd(base_lls),
        s: sample_sd(&diffs),
        n_particles,
        pairs: n,
        rho,
    })
}

/// Sample standard deviation, `n − 1` in the denominator.
fn sample_sd(xs: &[f64]) -> f64 {
    let n = xs.len();
    debug_assert!(n >= 2, "sample_sd needs at least two values");
    let mean = xs.iter().sum::<f64>() / n as f64;
    (xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// Draw the `θ'` the second evaluation of every pair is taken at.
///
/// This is the chain's own first move: on the transformed scale the step-0
/// proposal is `z' = z + proposal_sd·N(0,1)`, with the Robbins–Monro scale at
/// its starting value `λ = 1` and the Haario shape term not yet learned.
///
/// **One `θ'`, reused across every pair, on purpose.** The pair difference is
/// `log L̂(θ') − log L̂(θ)`; with `θ'` held fixed the true log-likelihood
/// difference is a constant offset, so the spread of the differences is the
/// estimator noise alone — which is the quantity the ceiling is a function of.
/// Redrawing `θ'` per pair would fold the curvature of the log-likelihood
/// surface between θ and its neighbours into `s` and overstate it, and the
/// ceiling would then be a statement about the posterior rather than about the
/// filter. The cost is that the measurement is conditional on the one `θ'`
/// drawn, which is why the seed is deterministic and the pair count is
/// persisted.
pub fn draw_proposed_theta(
    base: &[f64],
    specs: &[EstimatedParam],
    proposal_sd: &[f64],
    seed: u64,
) -> Vec<f64> {
    let mut rng = StatefulRng::new(seed ^ THETA_PRIME_STREAM);
    let mut theta = base.to_vec();
    for (i, spec) in specs.iter().enumerate() {
        let z = spec.to_transformed(base[spec.index]);
        let step = proposal_sd.get(i).copied().unwrap_or(0.0);
        theta[spec.index] = spec.from_transformed(z + step * rng.normal());
    }
    theta
}

/// Evaluate `NOISE_PAIRS` pairs of `log L̂`, under the scheme this stage runs.
///
/// Returns `(base_lls, proposed_lls)`, parallel and in pair order. Errors are
/// the inference convention: a structural error surfaces, a ruled-out θ is
/// `Ok(−∞)` (which [`summarize`] then declines to turn into a spread).
///
/// With `rho` unset this is two independent bootstrap filters per pair, so
/// `s² = sigma_θ² + sigma_θ'²` — equal to `2·sigma²` only where the filter is
/// equally noisy at both θ. With `rho` set the pair shares
/// the pre-drawn randoms through a Crank–Nicolson update, exactly as the chain
/// does, and the per-particle stream seed is held fixed across the pair for
/// the same reason the chain holds it fixed for the whole run. Measuring the
/// plain filter for a correlated run would report noise the run does not have:
/// correlated pseudo-marginal exists to shrink this very spread.
pub fn measure(
    config: &FitRunConfig,
    base: &[f64],
    proposed: &[f64],
    n_particles: usize,
    pairs: usize,
    rho: Option<f64>,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>), sim::error::SimError> {
    match rho {
        None => {
            // Base-θ seeds stay `seed + i`, the seeds this loop used when it
            // measured σ alone, so `initial_loglik` is unchanged for every
            // plain PMMH stage. The θ' evaluations take the block above it.
            let base_lls = (0..pairs)
                .map(|i| runner::run_quick_pfilter(config, base, n_particles, seed + i as u64))
                .collect::<Result<Vec<f64>, _>>()?;
            let proposed_lls = (0..pairs)
                .map(|i| runner::run_quick_pfilter(
                    config, proposed, n_particles, seed + (pairs + i) as u64))
                .collect::<Result<Vec<f64>, _>>()?;
            Ok((base_lls, proposed_lls))
        }
        Some(rho) => {
            let process = config.build_process();
            let obs_model = config.build_obs_model();
            let smc_config = sim::inference::traits::SMCConfig {
                n_particles,
                ..config.smc_config()
            };
            let obs_times: Vec<f64> = config.observations.iter().map(|o| o.time).collect();
            let steps_per_obs = correlated_pf::cpm_steps_per_obs(
                &obs_times, smc_config.t_start, smc_config.dt);
            let n_source_groups = config.compiled.source_groups.len();

            let mut rng = StatefulRng::new(seed);
            let mut base_lls = Vec::with_capacity(pairs);
            let mut proposed_lls = Vec::with_capacity(pairs);
            for _ in 0..pairs {
                let u = PFRandomState::draw_fresh(
                    n_particles, &steps_per_obs, n_source_groups, &mut rng);
                base_lls.push(eval_correlated(
                    &process, &obs_model, base, &smc_config, &u, seed)?);
                let u_prime = u.correlate(rho, &mut rng);
                proposed_lls.push(eval_correlated(
                    &process, &obs_model, proposed, &smc_config, &u_prime, seed)?);
            }
            Ok((base_lls, proposed_lls))
        }
    }
}

/// One correlated-filter evaluation under the inference convention (gh#224):
/// structural errors surface, everything else is a ruled-out θ at `−∞`.
fn eval_correlated(
    process: &sim::inference::ChainBinomialProcess,
    obs_model: &sim::inference::MultiStreamObsModel,
    params: &[f64],
    smc_config: &sim::inference::traits::SMCConfig,
    randoms: &PFRandomState,
    seed: u64,
) -> Result<f64, sim::error::SimError> {
    match correlated_pf::bootstrap_filter_correlated(
        process, obs_model, params, smc_config, randoms, seed,
    ) {
        Ok(r) => Ok(r.log_likelihood),
        Err(e) if e.is_structural() => Err(e),
        Err(_) => Ok(f64::NEG_INFINITY),
    }
}

/// Render the preflight report.
///
/// Split from the printing so the verdict and its wording are testable without
/// capturing stderr, as `preflight_specs` in `runner.rs` is. `n_obs` is the
/// number of observation times the log-likelihood sums over, which is what
/// decides whether the Gaussian assumption behind the ceiling is tight.
pub fn report(check: &PfNoiseCheck, ll_mean: f64, d: usize, n_obs: usize) -> String {
    let se = check.s_standard_error();
    let ceiling = check.ceiling();
    let target = target_acceptance(d);
    let cross = crossover_s(d);
    let verdict = ceiling_verdict(check.s, se, d);

    let scheme = match check.rho {
        Some(rho) => format!("correlated evaluations (rho = {:.2})", rho),
        None => "independent evaluations".to_string(),
    };

    let mut out = String::new();
    out.push_str(&format!(
        "  log L̂ mean = {:.1}, sd σ = {:.2} ({} particles, {} pairs)\n",
        ll_mean, check.sigma, check.n_particles, check.pairs));
    out.push_str(&format!(
        "  log L̂ ratio spread s = {:.2} ±{:.2} (1 SE, ~68%; ±{:.2} for a 95% read)\n",
        check.s, se, 2.0 * se));
    out.push_str(&format!(
        "    at base θ against one θ' from the initial proposal, {}\n", scheme));
    out.push_str(&format!(
        "  acceptance ceiling 2·Φ(-s/2) = {}   target 0.234 + 0.206/d = {:.1}% (d = {})\n",
        percent(ceiling), target * 100.0, d));

    match verdict {
        CeilingVerdict::Unreachable => {
            out.push_str(&format!(
                "  \x1b[33m⚠ ceiling below target\x1b[0m — no proposal scale reaches {:.1}%, \
                 so the Robbins-Monro\n    \
                 scale adaptation has no root and λ falls for the whole run. The chain \
                 under-mixes\n    \
                 however long it runs; s would have to fall below {:.2} to clear the \
                 target.\n    {}\n    \
                 Proceeding anyway — this is a diagnosis, not a refusal.\n",
                target * 100.0, cross, remedy(check)));
        }
        CeilingVerdict::Reachable => {
            out.push_str(&format!(
                "  \x1b[32m✓ ceiling above target\x1b[0m — a proposal scale reaching {:.1}% \
                 exists (s would have to\n    exceed {:.2} for it not to).\n",
                target * 100.0, cross));
            out.push_str(&halving_hint(check, se, cross));
            out.push_str(&gaussian_caveat(n_obs));
        }
        CeilingVerdict::Unresolved => {
            out.push_str(&format!(
                "  \x1b[33m? verdict unresolved\x1b[0m — s = {:.2} ± {:.2} (1 SE) straddles \
                 the crossover s* = {:.2},\n    \
                 where the ceiling equals the target. {} pairs cannot separate the two; \
                 more pairs\n    narrow the interval, more particles move s itself.\n",
                check.s, se, cross, check.pairs));
            out.push_str(&gaussian_caveat(n_obs));
        }
    }
    out
}

/// Format an acceptance rate, without letting rounding read as exactly zero.
///
/// At `s = 9.4` the ceiling is 3e-6, and `0.0%` invites the reader to take it
/// for a formatting artifact rather than the statement it is: this chain
/// essentially cannot accept a small move.
fn percent(p: f64) -> String {
    if p > 0.0 && p < 0.0005 {
        "<0.1%".to_string()
    } else {
        format!("{:.1}%", p * 100.0)
    }
}

/// What actually moves `s` for this stage, named rather than listed.
///
/// A longer chain is not on the list: the ceiling is a property of the
/// estimator, so more steps at the same noise level buy nothing.
fn remedy(check: &PfNoiseCheck) -> String {
    match check.rho {
        Some(rho) => format!(
            "Raise particles (σ ∝ 1/√N) or raise rho above {:.2}; a longer chain does not \
             help.", rho),
        None => "Raise particles (σ ∝ 1/√N), or set `rho` to reuse randomness between \
                 evaluations;\n    a longer chain does not help.".to_string(),
    }
}

/// Whether the run is buying precision the acceptance ceiling does not need.
///
/// Derived rather than banded: for independent evaluations `s = sigma·sqrt(2)`
/// and `sigma ∝ 1/sqrt(N)`, so halving the particle count multiplies `s` by
/// exactly `sqrt(2)`, and the hint fires only when the ceiling would still
/// clear the target there. This replaces the old "sd < 0.5 → halve particles"
/// band, whose threshold was a guess and whose advice was unconnected to the
/// target it was implicitly about.
///
/// Withheld under correlated pseudo-marginal. There `s` depends on the
/// correlation the scheme realises, which itself degrades as the auxiliary
/// variable grows (Deligiannidis, Doucet & Pitt 2018), so `s ∝ 1/sqrt(N)` does
/// not hold and the extrapolation would be a claim we have not measured.
fn halving_hint(check: &PfNoiseCheck, se: f64, cross: f64) -> String {
    if check.rho.is_some() {
        return String::new();
    }
    let s_halved = check.s * std::f64::consts::SQRT_2;
    if s_halved + se * std::f64::consts::SQRT_2 >= cross {
        return String::new();
    }
    format!(
        "    Halving particles to {} would put s at {:.2} (s ∝ 1/√N for independent\n             evaluations), still clear of the crossover — the run is paying for precision\n             the acceptance rate does not need.\n",
        check.n_particles / 2, s_halved)
}

/// The one-sided assumption behind the ceiling, stated where it can mislead.
///
/// Skew in `log L̂` makes `2·Φ(−s/2)` read *high*, so it is the verdicts that
/// clear or nearly clear the target that need the caveat; a run already told
/// its ceiling is below the target is, if anything, in worse shape than the
/// number says.
fn gaussian_caveat(n_obs: usize) -> String {
    if n_obs >= GAUSSIAN_OBS_FLOOR {
        format!(
            "    2·Φ(-s/2) assumes log L̂ is approximately Gaussian; summed over {} \
             observation\n    times that holds to about 1%.\n",
            n_obs)
    } else {
        format!(
            "    2·Φ(-s/2) assumes log L̂ is approximately Gaussian, and this fit sums \
             over only\n    {} observation times. Under skew the formula reads high, so \
             treat the ceiling as\n    an upper bound.\n",
            n_obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ceiling formula is checkable in closed form. These are the values
    /// the proposal tabulates and that its Monte Carlo of the noise-only chain
    /// reproduces, so they pin the formula rather than restating it.
    #[test]
    fn ceiling_matches_the_closed_form_values() {
        // σ = 1 → s = √2; the proposal's Monte Carlo gives 0.4805 / 0.4807.
        assert!((acceptance_ceiling(std::f64::consts::SQRT_2) - 0.479500).abs() < 1e-5,
            "σ = 1: got {}", acceptance_ceiling(std::f64::consts::SQRT_2));
        // σ = 2 → 15.73%, the number in the proposal's "Mechanism" table.
        assert!((acceptance_ceiling(2.0 * std::f64::consts::SQRT_2) - 0.157299).abs() < 1e-5,
            "σ = 2: got {}", acceptance_ceiling(2.0 * std::f64::consts::SQRT_2));
        // σ = 1.812, Sherlock et al.'s optimal noise level → 20.01%.
        assert!((acceptance_ceiling(1.812 * std::f64::consts::SQRT_2) - 0.200089).abs() < 1e-5,
            "σ = 1.812: got {}", acceptance_ceiling(1.812 * std::f64::consts::SQRT_2));
        // A noise-free likelihood accepts without a ceiling.
        assert!((acceptance_ceiling(0.0) - 1.0).abs() < 1e-12);
    }

    /// The crossovers the proposal tabulates, in σ. `crossover_s` works in `s`
    /// because the ceiling does; `s/√2` is the σ the table reports.
    #[test]
    fn crossover_matches_the_proposal_table() {
        for (d, sigma_star) in [(1usize, 1.09), (6, 1.565), (17, 1.640), (50, 1.67)] {
            let got = crossover_s(d) / std::f64::consts::SQRT_2;
            assert!((got - sigma_star).abs() < 5e-3,
                "d = {d}: crossover σ {got:.4}, proposal table {sigma_star}");
        }
        // It asymptotes to 1.683.
        assert!((crossover_s(1_000_000) / std::f64::consts::SQRT_2 - 1.683).abs() < 1e-3);
    }

    /// `crossover_s` inverts `acceptance_ceiling` at the target, so the two
    /// cannot drift: the verdict compares `s` against the crossover while the
    /// report prints the ceiling, and a mismatch would let them disagree.
    #[test]
    fn crossover_inverts_the_ceiling_at_the_target() {
        for d in [1usize, 2, 6, 17, 50, 200] {
            let at_crossover = acceptance_ceiling(crossover_s(d));
            assert!((at_crossover - target_acceptance(d)).abs() < 1e-8,
                "d = {d}: ceiling at crossover {at_crossover}, target {}",
                target_acceptance(d));
        }
    }

    /// The target must be the one the sampler actually adapts toward
    /// (`AdaptiveProposal::target_accept`), or the report compares a measured
    /// ceiling against a number no code uses.
    #[test]
    fn target_matches_the_samplers_own_target() {
        assert!((target_acceptance(1) - 0.44).abs() < 1e-9);
        assert!((target_acceptance(17) - 0.2461176470588).abs() < 1e-9);
        assert!(target_acceptance(1_000_000) > 0.234);
    }

    /// 20 pairs put one standard error at 16% of `s` — the number the report
    /// prints and the band the verdict uses.
    #[test]
    fn standard_error_is_sixteen_percent_at_twenty_pairs() {
        let se = spread_standard_error(1.0, NOISE_PAIRS);
        assert!((se - 0.16222).abs() < 1e-4, "got {se}");
        // Near the d = 17 crossover of σ = 1.64 that is ±0.26 at 1 SE.
        let s = crossover_s(17);
        let se_at_crossover = spread_standard_error(s, NOISE_PAIRS) / std::f64::consts::SQRT_2;
        assert!((se_at_crossover - 0.266).abs() < 5e-3, "got {se_at_crossover}");
    }

    /// Independent evaluations must give `s = σ·√2`; anything else means the
    /// pairing is wrong. Built from a fixed antisymmetric pattern so the
    /// arithmetic is exact rather than sampled.
    #[test]
    fn independent_pairs_at_equal_noise_give_s_equals_sigma_root_two() {
        let base: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let proposed: Vec<f64> = (0..20).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect();
        let c = summarize(&base, &proposed, 1000, None).expect("finite");
        // The two patterns are orthogonal over their 20 values and have equal
        // spread, so the difference has exactly √2 times the spread of either.
        // (Each is ±1 about a zero mean, so the n−1 sample sd is √(20/19).)
        assert!((c.sigma - (20.0f64 / 19.0).sqrt()).abs() < 1e-12, "σ = {}", c.sigma);
        assert!((c.s - c.sigma * std::f64::consts::SQRT_2).abs() < 1e-12,
            "s = {} should be σ√2 = {}", c.s, c.sigma * std::f64::consts::SQRT_2);
    }

    /// Sharing randomness shrinks the spread of the difference below
    /// `σ·√2` — the whole reason the preflight measures under the run's own
    /// scheme rather than under the plain filter.
    #[test]
    fn correlated_pairs_give_a_smaller_s_than_the_plain_filter_would() {
        let base: Vec<f64> = (0..20).map(|i| (i as f64 * 0.7).sin()).collect();
        // Proposed tracks base closely: shared randomness, small independent part.
        let proposed: Vec<f64> = base.iter().enumerate()
            .map(|(i, b)| b + 0.1 * (i as f64 * 1.9).cos())
            .collect();
        let c = summarize(&base, &proposed, 1000, Some(0.9)).expect("finite");
        // The shared part cancels in the difference, so `s` is a fraction of
        // `σ` — not merely below the independent bound `σ√2`, which `s = σ`
        // would also satisfy.
        assert!(c.s < 0.5 * c.sigma,
            "correlated s = {} should be well below σ = {}", c.s, c.sigma);
        assert!(c.s > 0.0, "s = {}", c.s);
        assert_eq!(c.rho, Some(0.9));
    }

    /// A ruled-out θ has no spread. Reporting `NaN` as a measurement would be
    /// worse than reporting nothing, so `summarize` declines.
    #[test]
    fn a_ruled_out_evaluation_yields_no_measurement() {
        let base = vec![-10.0; 5];
        let mut proposed = vec![-11.0; 5];
        proposed[2] = f64::NEG_INFINITY;
        assert!(summarize(&base, &proposed, 100, None).is_none());
        assert!(summarize(&base[..1], &proposed[..1], 100, None).is_none());
    }

    fn check_at(s: f64, sigma: f64, rho: Option<f64>) -> PfNoiseCheck {
        PfNoiseCheck { sigma, s, n_particles: 19_200, pairs: NOISE_PAIRS, rho }
    }

    /// The defect this replaces: `σ = 2` printed a green "PF variance OK
    /// (target: 1-3)". It must now name the ceiling, the target, and the side.
    #[test]
    fn a_sigma_of_two_is_reported_as_below_the_target_not_as_ok() {
        let c = check_at(2.0 * std::f64::consts::SQRT_2, 2.0, None);
        let out = report(&c, -1234.5, 17, 84);
        assert!(!out.contains("OK"), "must not bless the run: {out}");
        assert!(out.contains("2·Φ(-s/2) = 15.7%"), "ceiling missing: {out}");
        assert!(out.contains("24.6%"), "target missing: {out}");
        assert!(out.contains("ceiling below target"), "verdict missing: {out}");
        assert!(out.contains("λ falls for the whole run"), "consequence missing: {out}");
        assert_eq!(ceiling_verdict(c.s, c.s_standard_error(), 17),
            CeilingVerdict::Unreachable);
    }

    /// A quiet filter clears the target and says so, with the Gaussian
    /// assumption attached — that is the branch skew can mislead.
    #[test]
    fn a_quiet_filter_clears_the_target_and_states_the_assumption() {
        let c = check_at(0.5 * std::f64::consts::SQRT_2, 0.5, None);
        let out = report(&c, -1234.5, 17, 84);
        assert!(out.contains("ceiling above target"), "{out}");
        assert!(out.contains("approximately Gaussian"), "{out}");
        assert!(out.contains("84 observation"), "{out}");
        assert_eq!(ceiling_verdict(c.s, c.s_standard_error(), 17), CeilingVerdict::Reachable);
    }

    /// The particle-count hint the old banded check gave, now derived: it
    /// fires only when halving particles would still clear the crossover, and
    /// it is withheld under a scheme whose `s` does not scale as `1/√N`.
    #[test]
    fn the_halving_hint_is_derived_from_the_crossover_not_a_band() {
        // σ = 0.5 → doubling s to 1.41 still clears the d = 17 crossover of 2.32.
        let quiet = check_at(0.5 * std::f64::consts::SQRT_2, 0.5, None);
        let out = report(&quiet, -1.0, 17, 84);
        assert!(out.contains("Halving particles to 9600"), "{out}");
        // σ = 1.2 → halving would put s at 2.40, past the crossover: no hint,
        // even though the run itself clears the target.
        let borderline = check_at(1.2 * std::f64::consts::SQRT_2, 1.2, None);
        let out = report(&borderline, -1.0, 17, 84);
        assert!(out.contains("ceiling above target"), "{out}");
        assert!(!out.contains("Halving particles"), "{out}");
        // Under correlated PMMH `s ∝ 1/√N` is not a claim we have measured.
        let cpm = check_at(0.5, 1.5, Some(0.9));
        let out = report(&cpm, -1.0, 17, 84);
        assert!(!out.contains("Halving particles"), "{out}");
    }

    /// Few observation times make the ceiling an upper bound, and the report
    /// has to say the direction of the error.
    #[test]
    fn a_short_series_downgrades_the_ceiling_to_an_upper_bound() {
        let c = check_at(0.5 * std::f64::consts::SQRT_2, 0.5, None);
        let out = report(&c, -12.0, 17, 4);
        assert!(out.contains("upper bound"), "{out}");
        assert!(out.contains("reads high"), "{out}");
    }

    /// At the crossover the measurement cannot pick a side, and saying so is
    /// the point of reporting the interval at all.
    #[test]
    fn a_spread_at_the_crossover_is_unresolved() {
        let s = crossover_s(17);
        let c = check_at(s, s / std::f64::consts::SQRT_2, None);
        assert_eq!(ceiling_verdict(c.s, c.s_standard_error(), 17), CeilingVerdict::Unresolved);
        let out = report(&c, -12.0, 17, 84);
        assert!(out.contains("unresolved"), "{out}");
        assert!(out.contains("crossover"), "{out}");
        // Both reads are printed and labelled.
        assert!(out.contains("1 SE, ~68%"), "{out}");
        assert!(out.contains("95% read"), "{out}");
    }

    /// A ceiling that rounds to zero must not read as a formatting artifact,
    /// and the remedy named must be one that moves `s` — never a longer chain,
    /// which buys nothing against an estimator-level ceiling.
    #[test]
    fn a_vanishing_ceiling_and_its_remedy_are_stated_plainly() {
        let out = report(&check_at(9.4, 4.07, None), -72.6, 2, 20);
        assert!(out.contains("2·Φ(-s/2) = <0.1%"), "{out}");
        assert!(out.contains("a longer chain does not help"), "{out}");
        assert!(out.contains("set `rho`"), "{out}");
        // Under a correlated scheme the remedy is the one that is still open.
        let cpm = report(&check_at(9.4, 4.07, Some(0.9)), -72.6, 2, 20);
        assert!(cpm.contains("raise rho above 0.90"), "{cpm}");
        assert!(!cpm.contains("set `rho`"), "{cpm}");
    }

    /// The report names the scheme, because `s` is a property of the scheme
    /// and not only of the model.
    #[test]
    fn the_report_names_the_scheme_it_measured_under() {
        let plain = report(&check_at(2.0, 1.41, None), -1.0, 6, 30);
        assert!(plain.contains("independent evaluations"), "{plain}");
        let cpm = report(&check_at(0.6, 1.41, Some(0.9)), -1.0, 6, 30);
        assert!(cpm.contains("correlated evaluations (rho = 0.90)"), "{cpm}");
    }

    /// `θ'` is the chain's own first move: `z + proposal_sd·N(0,1)` on the
    /// transformed scale, deterministic in the seed, and it must actually move
    /// every estimated parameter.
    #[test]
    fn theta_prime_is_one_deterministic_draw_from_the_initial_proposal() {
        use sim::inference::types::Transform;
        let specs = vec![
            EstimatedParam {
                name: "beta".into(), index: 0, initial: 0.4, rw_sd: 0.02,
                transform: Transform::Log { lo: 0.001, hi: 5.0 },
                lower: 0.001, upper: 5.0, rw_sd_auto: false,
                perturb_only_at_t0: false,
            },
            EstimatedParam {
                name: "gamma".into(), index: 1, initial: 0.2, rw_sd: 0.01,
                transform: Transform::Log { lo: 0.01, hi: 1.0 },
                lower: 0.01, upper: 1.0, rw_sd_auto: false,
                perturb_only_at_t0: false,
            },
        ];
        let base = vec![0.4, 0.2, 999.0];
        let sd = vec![0.3, 0.3];
        let a = draw_proposed_theta(&base, &specs, &sd, 7);
        let b = draw_proposed_theta(&base, &specs, &sd, 7);
        assert_eq!(a, b, "the θ' draw must be deterministic in the seed");
        assert_ne!(a, draw_proposed_theta(&base, &specs, &sd, 8));
        assert!(a[0] != base[0] && a[1] != base[1], "both parameters must move: {a:?}");
        assert_eq!(a[2], base[2], "a non-estimated parameter must not move");
        // A zero step is a no-op, which pins that the step is the multiplier.
        assert_eq!(draw_proposed_theta(&base, &specs, &[0.0, 0.0], 7), base);
    }
}
