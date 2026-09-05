//! Rank-normalized convergence diagnostics — Vehtari, Gelman, Simpson,
//! Carpenter & Bürkner (2021), "Rank-normalization, folding, and
//! localization: An improved R̂ for assessing convergence of MCMC",
//! _Bayesian Analysis_ 16(2):667-718, doi:10.1214/20-BA1221.
//!
//! Three statistics, all computed from the same `chains[chain][draw]` layout:
//!
//! * **R̂** — `max(rank-normalized split-R̂, folded rank-normalized split-R̂)`.
//!   Splitting each chain in half catches a chain that drifts across its own
//!   run, which the classic Gelman & Rubin (1992) between-chain-means
//!   statistic cannot see. Rank normalization makes the statistic invariant to
//!   any monotone reparameterization and removes the finite-variance
//!   assumption. Folding (`|x − median(x)|`) catches chains that agree on
//!   location while disagreeing on **scale**.
//! * **bulk-ESS** — effective sample size of the rank-normalized split
//!   chains: how many independent draws the body of the marginal is worth.
//! * **tail-ESS** — the smaller of the effective sample sizes of the 5% and
//!   95% tail indicators. A posterior can mix well in the bulk and badly in
//!   the tail the interval endpoints are read from.
//!
//! Unlike the per-chain Geyer sum in [`crate::inference::pmmh::mcmc_ess`],
//! these use the **between-chain** variance, so they do not overstate the
//! effective N when chains sit in different modes — and they are defined
//! whatever R̂ reads, so nothing has to be suppressed.
//!
//! # Why the layout of this file mirrors an R package
//!
//! Every step here is written to reproduce the R package `posterior`
//! bit-for-bit, including two conventions that are easy to get plausibly
//! wrong: the rank offset is `(r − 3/8) / (S − 2·3/8 + 1)` (note the trailing
//! `+ 1`), and Geyer's truncated estimator keeps `ρ̂₀` in the sum when the
//! very first pair sum is non-positive. `posterior`'s numbers on committed
//! draws are the test oracle (`rust/crates/sim/tests/convergence_oracle.rs`);
//! a deviation here is a bug even when it looks like an improvement.
//!
//! # Particle-MCMC specifics
//!
//! Two properties of camdl's samplers that Stan's usual inputs do not have:
//!
//! * **Exact repeats.** A rejected PMMH proposal repeats θ exactly, and PGAS
//!   repeats it whenever the latent path does not renew. Ranks are therefore
//!   averaged within tied groups; assigning distinct ranks to tied draws would
//!   distort the transform in proportion to the rejection rate.
//! * **Frozen chains.** A chain that never accepted a move has zero variance.
//!   Zero *within-chain* variance is fine — one chain's autocovariances are
//!   then identically zero and the estimator still has the others. Zero
//!   variance across *all* draws is not: R̂ would divide by it. That case is
//!   refused by name ([`ConvergenceError::ConstantDraws`]) rather than
//!   returned as `NaN`/`inf`.

use std::fmt;

/// Why a rank-normalized diagnostic could not be computed for a parameter.
///
/// Every variant names a property of the *input*, **with the numbers**. Callers
/// render these; they must never be collapsed to a bare `NaN`, which reads as a
/// numerical failure and hides which precondition was missed.
///
/// Serialized so the numbers survive the trip to `*_summary.json` and
/// `camdl fit summary` can print the same sentence the stage printed at the
/// time. [`RhatRefusal`] is the lossy classification of the same fact — it
/// answers "is this a sampler pathology"; this type answers "what exactly was
/// wrong with the input".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "refusal", rename_all = "snake_case")]
pub enum ConvergenceError {
    /// R̂ compares chains; one chain has nothing to compare against.
    TooFewChains { n_chains: usize },
    /// Fewer than four draws per chain — below the structural minimum for a
    /// split-half statistic.
    TooFewDraws { n_draws: usize },
    /// The between-chain variance formula uses one draw count for every chain.
    UnequalChainLengths { expected: usize, chain: usize, found: usize },
    /// A draw is `NaN` or `±inf`. Rank normalization has no ordering for these
    /// and would silently propagate `NaN` through every statistic — see gh#607,
    /// a chain that recorded `log_posterior = −inf` for thousands of sweeps.
    ///
    /// `value` is non-finite by construction — that is what the variant is for
    /// — and `serde_json` writes any non-finite `f64` as `null`, which will not
    /// read back. It is carried as its `Display` form (`inf` / `-inf` / `NaN`),
    /// which `f64::from_str` parses, so the round-trip is exact.
    NonFiniteDraw {
        chain: usize,
        draw: usize,
        #[serde(with = "nonfinite_f64")]
        value: f64,
    },
    /// Every draw of every chain is the same value to within
    /// [`DEGENERATE_REL_TOL`] of the parameter's own scale. The total variance
    /// is zero, so R̂'s denominator is zero and the rank transform is constant.
    ConstantDraws { value: f64 },
}

/// `f64` as its `Display` string, for the one field that is non-finite by
/// construction. JSON has no encoding for `inf`/`NaN`, and `serde_json`'s
/// silent `null` would make the field unreadable on the way back.
mod nonfinite_f64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        let s = String::deserialize(d)?;
        s.parse::<f64>().map_err(serde::de::Error::custom)
    }
}

/// Why a parameter has no R̂, and — the part that decides what a fit may
/// claim — **whether that indicates a problem**.
///
/// The two are not the same question and collapsing them is how a fit that
/// could not be assessed came to report `converged: true`. A run given two
/// chains of three draws was never *offered* the shape a between-chain
/// statistic needs; a run whose sampler never accepted a move was, and failed.
/// The first is "not assessed", the second is "did not converge", and neither
/// is "converged".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RhatRefusal {
    /// A draw was `NaN` or `±inf` — gh#607's chain recording
    /// `log_posterior = −inf` for thousands of sweeps.
    NonFiniteDraw,
    /// R̂ itself evaluated non-finite. Every chain is internally constant at a
    /// value the others do not share, so the within-chain variance is zero and
    /// the between-chain variance is not: the 0%-acceptance deadlock.
    NonFiniteRhat,
    /// An **estimated** parameter that never moved. Distinct from a parameter
    /// the model pins, which is not offered to the estimator at all.
    ConstantDraws,
    /// Fewer than two chains.
    TooFewChains,
    /// Fewer than four draws per chain.
    TooFewDraws,
    /// Chains of differing length.
    UnequalChainLengths,
    /// The estimated parameter set could not be determined, so a constant
    /// column cannot be told from a pinned one and no honest classification is
    /// available.
    EstimatedSetUnknown,
}

impl RhatRefusal {
    /// `true` when the refusal is evidence the sampler MISBEHAVED, so a fit
    /// carrying it must not be called converged. `false` when the run was
    /// simply never given the shape the statistic needs — then the honest
    /// report is "not assessed", which is also not "converged".
    pub fn is_pathology(self) -> bool {
        match self {
            Self::NonFiniteDraw | Self::NonFiniteRhat | Self::ConstantDraws => true,
            Self::TooFewChains
            | Self::TooFewDraws
            | Self::UnequalChainLengths
            | Self::EstimatedSetUnknown => false,
        }
    }

    /// One clause naming what happened, for a report.
    pub fn describe(self) -> &'static str {
        match self {
            Self::NonFiniteDraw => "a draw was NaN or infinite",
            Self::NonFiniteRhat => "every chain sat at its own single value",
            Self::ConstantDraws => "the parameter never moved",
            Self::TooFewChains => "fewer than 2 chains",
            Self::TooFewDraws => "fewer than 4 draws per chain",
            Self::UnequalChainLengths => "chains of differing length",
            Self::EstimatedSetUnknown => "the estimated parameter set is unknown",
        }
    }
}

impl ConvergenceError {
    /// How this refusal should be classified for reporting.
    pub fn refusal(&self) -> RhatRefusal {
        match self {
            Self::TooFewChains { .. } => RhatRefusal::TooFewChains,
            Self::TooFewDraws { .. } => RhatRefusal::TooFewDraws,
            Self::UnequalChainLengths { .. } => RhatRefusal::UnequalChainLengths,
            Self::NonFiniteDraw { .. } => RhatRefusal::NonFiniteDraw,
            Self::ConstantDraws { .. } => RhatRefusal::ConstantDraws,
        }
    }
}

impl fmt::Display for ConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewChains { n_chains } => write!(
                f, "R̂ needs at least 2 chains; got {n_chains}"),
            Self::TooFewDraws { n_draws } => write!(
                f, "split-R̂ needs at least 4 draws per chain; got {n_draws}"),
            Self::UnequalChainLengths { expected, chain, found } => write!(
                f, "chain {chain} has {found} draws, expected {expected} \
                    (the between-chain variance uses one draw count)"),
            Self::NonFiniteDraw { chain, draw, value } => write!(
                f, "chain {chain} draw {draw} is {value}; rank normalization \
                    is undefined for non-finite draws"),
            Self::ConstantDraws { value } => write!(
                f, "every draw is {value}: the parameter did not move, so R̂ \
                    has no within-chain variance to divide by"),

        }
    }
}

impl std::error::Error for ConvergenceError {}

/// How close to constant the pooled draws must be before the estimator refuses
/// them, **relative to the parameter's own scale**.
///
/// `posterior` uses an absolute `.Machine$double.eps`, which is the wrong
/// comparison for a parameter measured in cases per year: draws spread over
/// `1e-9` at a mean of `1e6` are constant in every sense that matters and
/// would otherwise produce an R̂ built from rounding noise. This mirrors the
/// `degenerate_w_threshold` already used for the IF2 chain-agreement statistic.
pub const DEGENERATE_REL_TOL: f64 = 1e-12;

/// Draws per chain below which the bulk and tail ESS of
/// [`rank_convergence`] are a CONSTANT rather than a measurement: exactly half
/// the pooled draw count, whatever the draws happen to be.
///
/// The estimator splits every chain in half before estimating the effective
/// sample size, and Geyer's initial-positive-sequence truncation cannot take
/// even its first step unless a half holds more than five draws. Below that
/// the estimated integrated autocorrelation time -- the factor by which
/// within-chain autocorrelation inflates the variance of a chain mean --
/// falls back to its `-1 + 2*rho_0 + rho_0 = 2` default, so ESS comes out as
/// `n_chains * (n_draws / 2)` for every input (integer division: an odd draw
/// count loses its middle draw to the split). Twelve draws per chain is the
/// smallest count whose halves (six) clear the threshold; below six draws a
/// half is under the three the autocovariance needs and ESS is `NaN` instead.
///
/// `posterior::ess_bulk` behaves the same way; this is a property of the
/// reference estimator, not a camdl deviation, and the oracle case
/// `two_chains_short` (2 chains x 8 draws, ESS 8) pins it. The number is
/// therefore not wrong and must not be "corrected" -- but a report that prints
/// it beside real diagnostics has to say that it measures nothing, or a reader
/// will read mixing into a quantity that only counts draws.
pub const MIN_DRAWS_FOR_INFORMATIVE_ESS: usize = 12;

/// The rank-normalized convergence statistics for one parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct RankConvergence {
    /// The headline: `max(rhat_bulk, rhat_folded)`.
    pub rhat: f64,
    /// Rank-normalized split-R̂ — disagreement in **location**.
    pub rhat_bulk: f64,
    /// Rank-normalized split-R̂ of `|x − median(x)|` — disagreement in
    /// **scale**, which `rhat_bulk` cannot see.
    pub rhat_folded: f64,
    /// Bulk effective sample size. `NaN` only when a chain carries fewer than
    /// six draws (each split half then has fewer than the three the
    /// autocovariance estimator needs).
    pub ess_bulk: f64,
    /// Tail effective sample size: `min` over the 5% and 95% indicators.
    /// `NaN` when an indicator is constant — a parameter whose top 5% of draws
    /// are all exactly at a bound has no 95% tail to measure. `posterior`
    /// reports `NA` in the same case.
    pub ess_tail: f64,
    /// `n_chains × n_draws` — the denominator for [`Self::ess_bulk_ratio`].
    pub n_draws_total: usize,
    /// Every chain sat at its own single value: the sampler never accepted a
    /// move. R̂ is `+∞` and ESS is near its floor when this is set; the flag is
    /// what lets a report say *why* instead of printing an infinity and
    /// leaving the reader to infer the cause.
    pub all_chains_frozen: bool,
}

/// Which half of `max(rhat_bulk, rhat_folded)` set the headline R̂ — the
/// answer to *why* R̂ is high, which the headline alone cannot give.
///
/// The two halves differ by exactly one transformation (folding about the
/// median), so the larger one names which kind of between-chain disagreement
/// the statistic is reacting to. That is a decomposition, not a threshold: no
/// cutoff on the gap is proposed or applied here, deliberately — see
/// `docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md`, which
/// defers the lint until the gaps have been observed on a corpus of real fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhatDriver {
    /// `rhat_bulk` is the larger: the chains disagree about **where** the
    /// posterior sits. A chain drifting across its own run reads this way,
    /// because splitting compares each chain's halves.
    Location,
    /// `rhat_folded` is the larger: the chains agree on location and disagree
    /// on **spread**, which `rhat_bulk` cannot see.
    Scale,
}

impl RhatDriver {
    /// The larger half, or `None` when either is undefined (so there is no
    /// comparison to report) — the folded half is undefined whenever
    /// `|x − median(x)|` is constant.
    pub fn of(rhat_bulk: f64, rhat_folded: f64) -> Option<Self> {
        if rhat_bulk.is_nan() || rhat_folded.is_nan() {
            return None;
        }
        Some(if rhat_folded > rhat_bulk { Self::Scale } else { Self::Location })
    }

    /// One clause naming what the larger half means, for a report.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Location => "the chains disagree about where the posterior sits",
            Self::Scale => "the chains agree on location and disagree on spread",
        }
    }

    /// The name of the half, for a line that also prints both numbers.
    pub fn half(self) -> &'static str {
        match self {
            Self::Location => "bulk",
            Self::Scale => "folded",
        }
    }
}

impl RankConvergence {
    /// Which half of `max(rhat_bulk, rhat_folded)` the headline came from.
    pub fn rhat_driver(&self) -> Option<RhatDriver> {
        RhatDriver::of(self.rhat_bulk, self.rhat_folded)
    }

    /// Bulk-ESS as a fraction of the draws it was computed from.
    ///
    /// Report it alongside the ESS itself. Geyer's initial-positive-sequence
    /// truncation destabilizes as the integrated autocorrelation time
    /// approaches the run length: bulk-ESS 11 from 1400 draws per chain means
    /// the estimator summed autocorrelations out to nearly the whole run, and
    /// is then reporting mostly about its own truncation point. The ratio is
    /// what makes that visible; the ESS alone is not.
    pub fn ess_bulk_ratio(&self) -> f64 {
        if self.n_draws_total == 0 {
            return f64::NAN;
        }
        self.ess_bulk / self.n_draws_total as f64
    }
}

/// Compute R̂, bulk-ESS and tail-ESS for one parameter's per-chain draws.
///
/// `chains[c][i]` is chain `c`'s `i`-th retained (post-warm-up, thinned) draw.
/// All chains must have the same length.
pub fn rank_convergence(chains: &[Vec<f64>]) -> Result<RankConvergence, ConvergenceError> {
    let n_chains = chains.len();
    if n_chains < 2 {
        return Err(ConvergenceError::TooFewChains { n_chains });
    }
    let n_draws = chains[0].len();
    for (c, chain) in chains.iter().enumerate() {
        if chain.len() != n_draws {
            return Err(ConvergenceError::UnequalChainLengths {
                expected: n_draws, chain: c, found: chain.len(),
            });
        }
    }
    if n_draws < 4 {
        return Err(ConvergenceError::TooFewDraws { n_draws });
    }
    for (c, chain) in chains.iter().enumerate() {
        for (i, &v) in chain.iter().enumerate() {
            if !v.is_finite() {
                return Err(ConvergenceError::NonFiniteDraw { chain: c, draw: i, value: v });
            }
        }
    }

    let mut pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).expect("draws are finite"));
    let (lo, hi) = (pooled[0], pooled[pooled.len() - 1]);
    let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
    if hi - lo <= DEGENERATE_REL_TOL * mean.abs().max(f64::MIN_POSITIVE) {
        return Err(ConvergenceError::ConstantDraws { value: mean });
    }

    // Every chain internally constant — the 0%-acceptance deadlock — is
    // DETECTED but not refused. With the exact-zero variance above, R̂ comes
    // out as +∞, which is the mathematically correct answer and is what
    // `posterior` reports; ESS stays computable and is still worth having
    // (`posterior` reverted per-chain constancy checking for ESS in #198,
    // as overly conservative). What camdl adds over `posterior` is the
    // REASON, carried alongside the ∞ so the reader is told what to fix
    // rather than left to infer it from an infinity.
    let all_chains_frozen = chains.iter().all(|c| {
        let (lo, hi) = c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
        let scale = (c.iter().sum::<f64>() / c.len() as f64).abs().max(f64::MIN_POSITIVE);
        hi - lo <= DEGENERATE_REL_TOL * scale
    });

    let split = split_chains(chains);
    let rhat_bulk = rhat_basic(&rank_normalize(&split));

    let median = quantile_type7(&pooled, 0.5);
    let folded: Vec<Vec<f64>> = chains.iter()
        .map(|c| c.iter().map(|v| (v - median).abs()).collect())
        .collect();
    let rhat_folded = rhat_basic(&rank_normalize(&split_chains(&folded)));

    let ess_bulk = ess(&rank_normalize(&split));

    // The indicator is formed on the UNSPLIT draws (one quantile for the whole
    // cloud), then split — the order matters, and the reverse would compute a
    // different quantile per half.
    let tail_ess = |p: f64| -> f64 {
        let q = quantile_type7(&pooled, p);
        let ind: Vec<Vec<f64>> = chains.iter()
            .map(|c| c.iter().map(|&v| if v <= q { 1.0 } else { 0.0 }).collect())
            .collect();
        ess(&split_chains(&ind))
    };
    let (e05, e95) = (tail_ess(0.05), tail_ess(0.95));
    // `f64::min` returns the non-NaN operand; R's `min` propagates NA. An
    // undefined 95% indicator must not be papered over by a defined 5% one.
    let ess_tail = if e05.is_nan() || e95.is_nan() { f64::NAN } else { e05.min(e95) };

    // `max(bulk, folded)` with R's semantics: an undefined half makes the
    // headline undefined. `f64::max` returns the non-NaN operand, which would
    // silently publish the bulk value as though the folded check had passed.
    // The folded half is genuinely undefined whenever `|x − median(x)|` is
    // constant — a two-point symmetric marginal, which is exactly what a pair
    // of frozen chains produces. ArviZ has this same latent inconsistency;
    // `posterior` propagates, and `posterior` is what the oracle pins.
    let rhat_headline = if rhat_bulk.is_nan() || rhat_folded.is_nan() {
        f64::NAN
    } else {
        rhat_bulk.max(rhat_folded)
    };

    Ok(RankConvergence {
        all_chains_frozen,
        rhat: rhat_headline,
        rhat_bulk,
        rhat_folded,
        ess_bulk,
        ess_tail,
        n_draws_total: n_chains * n_draws,
    })
}

// ── the pieces ─────────────────────────────────────────────────────────────

/// Split every chain into its first and second half, doubling the chain count.
///
/// An odd draw count drops the middle draw rather than sharing it between the
/// halves — `posterior`'s convention, and the only one that keeps the two
/// halves independent.
fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = chains[0].len();
    if n < 2 {
        return chains.to_vec();
    }
    let head = n / 2;          // floor(n/2)
    let tail_start = n - head; // ceil(n/2) for even n, ceil(n/2)+1 for odd n
    let mut out = Vec::with_capacity(chains.len() * 2);
    for c in chains {
        out.push(c[..head].to_vec());
    }
    for c in chains {
        out.push(c[tail_start..].to_vec());
    }
    out
}

/// Average ranks of every draw across all chains, mapped through the inverse
/// standard normal CDF at the Blom offset `c = 3/8`.
///
/// Ties share their average rank. That is not a nicety here: a PMMH chain's
/// rejections are exact repeats, so tied groups are large and systematic, and
/// breaking them arbitrarily would inject the rejection pattern into the
/// transformed scale.
fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total: usize = chains.iter().map(|c| c.len()).sum();
    // (value, flat index) sorted by value; ties then get the mean of the ranks
    // their group spans.
    let mut flat: Vec<(f64, usize)> = Vec::with_capacity(total);
    for c in chains {
        for &v in c {
            flat.push((v, flat.len()));
        }
    }
    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("draws are finite"));

    let mut ranks = vec![0.0_f64; total];
    let mut i = 0;
    while i < total {
        let mut j = i;
        while j + 1 < total && flat[j + 1].0 == flat[i].0 {
            j += 1;
        }
        // 1-based ranks i+1 ..= j+1, averaged.
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for entry in &flat[i..=j] {
            ranks[entry.1] = avg;
        }
        i = j + 1;
    }

    // Blom: p = (r − 3/8) / (S − 2·3/8 + 1). The trailing `+ 1` is part of
    // `posterior`'s `backtransform_ranks`; dropping it shifts every z and is
    // invisible without an external reference.
    let denom = total as f64 - 2.0 * 0.375 + 1.0;
    let mut out = Vec::with_capacity(chains.len());
    let mut k = 0;
    for c in chains {
        let mut col = Vec::with_capacity(c.len());
        for _ in 0..c.len() {
            col.push(numerics::normal_quantile((ranks[k] - 0.375) / denom));
            k += 1;
        }
        out.push(col);
    }
    out
}

/// Gelman & Rubin's R̂ on the chains exactly as given — no splitting, no rank
/// transform. `rank_convergence` composes it with both.
///
/// `NaN` when the input is constant, which the caller has already refused for
/// the raw draws but which can still arise for a folded/indicator transform.
fn rhat_basic(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    let n = chains[0].len();
    if m < 2 || n < 2 {
        return f64::NAN;
    }
    if is_constant(chains) {
        return f64::NAN;
    }
    let means: Vec<f64> = chains.iter()
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let vars: Vec<f64> = chains.iter().zip(&means)
        .map(|(c, &mu)| {
            // A chain that never moved has variance EXACTLY zero, and R̂'s
            // denominator must be exactly zero so the ratio is +∞.
            //
            // Computing it as Σ(x−μ)²/(n−1) does not give that: μ = Σx/n does
            // not round-trip through the summation, so a constant chain leaves
            // a one-ulp residue around 1e-32. That residue then becomes the
            // denominator, and R̂ comes back as a FINITE ~1e15 whose magnitude
            // is set by the array shape rather than by the chains — the same
            // input at a different draw count gives a different number, and
            // different inputs at the same shape give the identical one.
            // R's `matrixStats::colVars` returns exact zero here; so must this.
            if c.iter().all(|&x| x == c[0]) {
                return 0.0;
            }
            c.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64
        })
        .collect();
    let grand = means.iter().sum::<f64>() / m as f64;
    let var_of_means = means.iter().map(|&mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1) as f64;
    let var_between = n as f64 * var_of_means;
    let var_within = vars.iter().sum::<f64>() / m as f64;
    ((var_between / var_within + n as f64 - 1.0) / n as f64).sqrt()
}

/// Vehtari et al.'s cross-chain effective sample size: Geyer's
/// initial-positive-sequence estimator applied to the autocorrelation
/// *combined across chains* through `var_plus`, then made monotone.
///
/// `NaN` for a constant input or fewer than three draws per chain.
fn ess(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    let n = chains[0].len();
    if n < 3 || is_constant(chains) {
        return f64::NAN;
    }

    // Biased (denominator `n`) autocovariance, averaged over chains, computed
    // lag by lag — Geyer's truncation usually stops far short of `n`.
    let mut cache: Vec<Option<f64>> = vec![None; n];
    let centered: Vec<Vec<f64>> = chains.iter()
        .map(|c| {
            let mu = c.iter().sum::<f64>() / n as f64;
            c.iter().map(|&x| x - mu).collect()
        })
        .collect();
    let mut acov_mean = |lag: usize| -> f64 {
        if let Some(v) = cache[lag] {
            return v;
        }
        let total: f64 = centered.iter()
            .map(|c| c[..n - lag].iter().zip(&c[lag..])
                .map(|(a, b)| a * b).sum::<f64>() / n as f64)
            .sum();
        let v = total / m as f64;
        cache[lag] = Some(v);
        v
    };

    let mean_var = acov_mean(0) * n as f64 / (n - 1) as f64;
    let mut var_plus = mean_var * (n - 1) as f64 / n as f64;
    if m > 1 {
        let means: Vec<f64> = chains.iter()
            .map(|c| c.iter().sum::<f64>() / n as f64)
            .collect();
        let grand = means.iter().sum::<f64>() / m as f64;
        var_plus += means.iter().map(|&mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1) as f64;
    }

    let mut rho = vec![0.0_f64; n];
    let mut t = 0_usize;
    let mut rho_even = 1.0_f64;
    rho[0] = rho_even;
    let mut rho_odd = 1.0 - (mean_var - acov_mean(1)) / var_plus;
    rho[1] = rho_odd;
    while (t as i64) < n as i64 - 5
        && !(rho_even + rho_odd).is_nan()
        && rho_even + rho_odd > 0.0
    {
        t += 2;
        rho_even = 1.0 - (mean_var - acov_mean(t)) / var_plus;
        rho_odd = 1.0 - (mean_var - acov_mean(t + 1)) / var_plus;
        if rho_even + rho_odd >= 0.0 {
            rho[t] = rho_even;
            rho[t + 1] = rho_odd;
        }
    }
    let max_t = t;
    if rho_even > 0.0 {
        rho[max_t] = rho_even;
    }

    // Geyer's initial MONOTONE sequence: the pair sums must not increase.
    let mut t = 0_usize;
    while max_t >= 4 && t <= max_t - 4 {
        t += 2;
        if rho[t] + rho[t + 1] > rho[t - 2] + rho[t - 1] {
            rho[t] = (rho[t - 2] + rho[t - 1]) / 2.0;
            rho[t + 1] = rho[t];
        }
    }

    // Geyer's truncated estimator of the integrated autocorrelation time.
    // When the very first pair sum was non-positive (`max_t == 0`), ρ̂₀ still
    // enters the sum — R's `x[1:0]` selects element 1, which is the behaviour
    // `posterior` inherits and which makes τ̂ = 2 rather than 0 there.
    let sum_head: f64 = if max_t == 0 { rho[0] } else { rho[..max_t].iter().sum() };
    let mut tau = -1.0 + 2.0 * sum_head + rho[max_t];
    let draws = (m * n) as f64;
    // Cap: as τ̂ approaches 1/log10(N) the estimate is dominated by where the
    // truncation happened rather than by the chain.
    let tau_bound = 1.0 / draws.log10();
    if tau < tau_bound {
        tau = tau_bound;
    }
    draws / tau
}

/// True when every value across every chain is identical to machine epsilon.
/// Matches `posterior`'s `is_constant`, and is the guard the transformed
/// (folded, rank, indicator) intermediates need — the scale-relative check in
/// [`rank_convergence`] applies to the raw draws.
fn is_constant(chains: &[Vec<f64>]) -> bool {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in chains {
        for &v in c {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (hi - lo).abs() < f64::EPSILON
}

/// R's default (type 7) sample quantile of an ascending slice.
///
/// The tail indicators and the fold's median are defined against this
/// convention; the several other in-use quantile definitions place the 5%
/// cut on a different draw and change tail-ESS.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    if lo + 1 >= n {
        return sorted[n - 1];
    }
    sorted[lo] + (h - lo as f64) * (sorted[lo + 1] - sorted[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, start: f64, step: f64) -> Vec<f64> {
        (0..n).map(|i| start + step * i as f64).collect()
    }

    /// `MIN_DRAWS_FOR_INFORMATIVE_ESS` is a claim about the estimator, so it
    /// is pinned against the estimator rather than left as a comment.
    ///
    /// Below the threshold, bulk-ESS is exactly half the split draw count for
    /// two datasets that share nothing but their shape -- one strongly
    /// autocorrelated, one strongly antithetic -- which is what "measures
    /// nothing" means. At the threshold the two separate. This is the
    /// mechanism behind
    /// `ESS min (mixed)` reading the same number in every bin of the
    /// latent-path convergence block when a stage saves ten paths per chain.
    #[test]
    fn ess_below_the_informative_threshold_is_half_the_draws_whatever_the_draws() {
        // Deterministic, reproducible, and deliberately unalike: `sticky`
        // repeats each value (near-perfect positive autocorrelation), `mixed`
        // alternates far apart (strongly antithetic).
        let sticky = |c: usize, n: usize| -> Vec<Vec<f64>> {
            (0..c).map(|k| (0..n)
                .map(|i| k as f64 + (i / 2) as f64 * 0.5).collect()).collect()
        };
        let mixed = |c: usize, n: usize| -> Vec<Vec<f64>> {
            (0..c).map(|k| (0..n)
                .map(|i| k as f64 + if i % 2 == 0 { 1.0 } else { -1.0 } + i as f64 * 1e-3)
                .collect()).collect()
        };
        for n in 6..MIN_DRAWS_FOR_INFORMATIVE_ESS {
            let a = rank_convergence(&sticky(8, n)).expect("statistics defined");
            let b = rank_convergence(&mixed(8, n)).expect("statistics defined");
            // The split drops an odd middle draw, so the pinned value is
            // `n_chains * (n_draws / 2)`, not `n_chains * n_draws / 2`.
            let pinned = (8 * (n / 2)) as f64;
            assert_eq!(a.ess_bulk, pinned,
                "{n} draws/chain: bulk-ESS is pinned at half the split draws");
            assert_eq!(b.ess_bulk, pinned,
                "{n} draws/chain: two unlike datasets give the same ESS");
        }
        // At the threshold the truncation runs and the two datasets separate.
        let n = MIN_DRAWS_FOR_INFORMATIVE_ESS;
        let a = rank_convergence(&sticky(8, n)).expect("statistics defined");
        let b = rank_convergence(&mixed(8, n)).expect("statistics defined");
        assert_ne!(a.ess_bulk, b.ess_bulk,
            "at {n} draws/chain ESS must respond to the draws: {} vs {}",
            a.ess_bulk, b.ess_bulk);
    }

    /// Every refusal must survive the trip to `*_summary.json` and back with
    /// its numbers intact, or `camdl fit summary` cannot say more than an
    /// em-dash. `NonFiniteDraw` is the one that needs care: its `value` is
    /// non-finite by construction, and `serde_json` writes any non-finite
    /// `f64` as `null`, which then fails to deserialize as a number. It is
    /// carried as its `Display` form instead.
    #[test]
    fn every_refusal_round_trips_through_json_with_its_numbers() {
        let cases = [
            ConvergenceError::TooFewChains { n_chains: 1 },
            ConvergenceError::TooFewDraws { n_draws: 3 },
            ConvergenceError::UnequalChainLengths { expected: 50, chain: 1, found: 40 },
            ConvergenceError::NonFiniteDraw { chain: 1, draw: 7, value: f64::NEG_INFINITY },
            ConvergenceError::NonFiniteDraw { chain: 0, draw: 0, value: f64::INFINITY },
            ConvergenceError::ConstantDraws { value: 2.5 },
        ];
        for want in cases {
            let text = serde_json::to_string(&want).expect("serializes");
            assert!(!text.contains("null"),
                "a null loses the number the message is made of: {text}");
            let got: ConvergenceError =
                serde_json::from_str(&text).expect("round-trips");
            assert_eq!(got, want, "via {text}");
            // And the sentence a reader sees survives with it.
            assert_eq!(got.to_string(), want.to_string());
        }

        // NaN is not `PartialEq` with itself, so it is checked on the message.
        let nan = ConvergenceError::NonFiniteDraw { chain: 2, draw: 3, value: f64::NAN };
        let text = serde_json::to_string(&nan).expect("serializes");
        let got: ConvergenceError = serde_json::from_str(&text).expect("round-trips");
        assert!(matches!(got, ConvergenceError::NonFiniteDraw { chain: 2, draw: 3, value }
            if value.is_nan()), "got {got:?} via {text}");
    }

    /// The two halves of the headline, and what their order means. No cutoff
    /// on the gap is applied — only which is larger is reported.
    #[test]
    fn the_driver_names_the_larger_half_and_is_undefined_when_one_is() {
        assert_eq!(RhatDriver::of(1.42, 1.00), Some(RhatDriver::Location));
        assert_eq!(RhatDriver::of(1.00, 1.31), Some(RhatDriver::Scale));
        // Ties go to location: `max` took the bulk value, and there is no
        // spread disagreement to report.
        assert_eq!(RhatDriver::of(1.10, 1.10), Some(RhatDriver::Location));
        // An undefined folded half leaves nothing to compare.
        assert_eq!(RhatDriver::of(f64::INFINITY, f64::NAN), None);
        assert_eq!(RhatDriver::of(f64::NAN, 1.0), None);
        // A frozen pair: bulk is a well-defined +inf, so the comparison holds.
        assert_eq!(RhatDriver::of(f64::INFINITY, 1.0), Some(RhatDriver::Location));
    }

    #[test]
    fn refuses_a_single_chain_by_name() {
        let e = rank_convergence(&[ramp(50, 0.0, 0.1)]).unwrap_err();
        assert_eq!(e, ConvergenceError::TooFewChains { n_chains: 1 });
    }

    #[test]
    fn refuses_unequal_chain_lengths_by_name() {
        let e = rank_convergence(&[ramp(50, 0.0, 0.1), ramp(40, 0.0, 0.1)]).unwrap_err();
        assert_eq!(
            e,
            ConvergenceError::UnequalChainLengths { expected: 50, chain: 1, found: 40 }
        );
    }

    #[test]
    fn refuses_too_few_draws_by_name() {
        let e = rank_convergence(&[ramp(3, 0.0, 0.1), ramp(3, 1.0, 0.1)]).unwrap_err();
        assert_eq!(e, ConvergenceError::TooFewDraws { n_draws: 3 });
    }

    /// gh#607: a chain recording `log_posterior = −inf` for thousands of
    /// sweeps must be refused by name, not silently rank-normalized into NaN.
    #[test]
    fn refuses_non_finite_draws_by_name() {
        let mut bad = ramp(50, 0.0, 0.1);
        bad[7] = f64::NEG_INFINITY;
        let e = rank_convergence(&[ramp(50, 0.0, 0.1), bad]).unwrap_err();
        assert_eq!(
            e,
            ConvergenceError::NonFiniteDraw { chain: 1, draw: 7, value: f64::NEG_INFINITY }
        );
        let mut nan = ramp(50, 0.0, 0.1);
        nan[0] = f64::NAN;
        assert!(matches!(
            rank_convergence(&[nan, ramp(50, 0.0, 0.1)]).unwrap_err(),
            ConvergenceError::NonFiniteDraw { chain: 0, draw: 0, .. }
        ));
    }

    /// The degenerate-variance guard is RELATIVE to the parameter's scale: a
    /// spread of 1e-9 around a mean of 1e6 is eleven orders of magnitude below
    /// the value and carries no information, even though it is far above
    /// machine epsilon in absolute terms.
    #[test]
    fn refuses_constant_draws_relative_to_parameter_scale() {
        let flat = vec![vec![2.5_f64; 40], vec![2.5_f64; 40]];
        assert!(matches!(
            rank_convergence(&flat).unwrap_err(),
            ConvergenceError::ConstantDraws { .. }
        ));

        let near_flat: Vec<Vec<f64>> = (0..2)
            .map(|c| (0..40).map(|i| 1.0e6 + 1.0e-9 * ((i + c) % 3) as f64).collect())
            .collect();
        assert!(
            matches!(
                rank_convergence(&near_flat).unwrap_err(),
                ConvergenceError::ConstantDraws { .. }
            ),
            "a 1e-9 spread at scale 1e6 must be refused, not scored"
        );

        // Negative control: the same absolute spread at a scale where it is
        // real information must be scored, not refused.
        let real: Vec<Vec<f64>> = (0..2)
            .map(|c| (0..40).map(|i| 1.0e-9 * ((i * 7 + c * 3) % 11) as f64).collect())
            .collect();
        assert!(rank_convergence(&real).is_ok(),
            "a 1e-9 spread at scale 1e-9 is the whole parameter");
    }

    /// The transform is invariant to any monotone reparameterization — the
    /// property that makes it usable on camdl's bounded and heavy-tailed
    /// marginals. R̂ of `x` must equal R̂ of `exp(x)` exactly.
    #[test]
    fn rank_statistics_are_monotone_invariant() {
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| (0..120).map(|i| {
                let x = i as f64 * 0.37 + c as f64 * 1.1;
                (x.sin() * 0.8 + x.cos() * 0.3) + 0.02 * i as f64
            }).collect())
            .collect();
        let raw = rank_convergence(&chains).expect("scored");
        let warped: Vec<Vec<f64>> = chains.iter()
            .map(|c| c.iter().map(|v| v.exp()).collect())
            .collect();
        let got = rank_convergence(&warped).expect("scored");
        assert!((raw.rhat_bulk - got.rhat_bulk).abs() < 1e-12,
            "bulk R̂ must be invariant: {} vs {}", raw.rhat_bulk, got.rhat_bulk);
        assert!((raw.ess_bulk - got.ess_bulk).abs() < 1e-9,
            "bulk ESS must be invariant: {} vs {}", raw.ess_bulk, got.ess_bulk);
    }

    /// The split convention drops the middle draw of an odd-length chain
    /// rather than putting it in both halves.
    #[test]
    fn split_drops_the_middle_draw_when_the_count_is_odd() {
        let c = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let s = split_chains(&c);
        assert_eq!(s, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
        let even = vec![vec![1.0, 2.0, 3.0, 4.0]];
        assert_eq!(split_chains(&even), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    /// Tied draws — a rejected PMMH proposal repeats θ exactly — share the
    /// mean of the ranks their group spans.
    #[test]
    fn ties_take_the_average_rank() {
        let z = rank_normalize(&[vec![5.0, 1.0, 1.0, 1.0, 9.0]]);
        // Ranks: 1,2,3 tie at 2.0 for the three 1.0s; 5.0 is 4; 9.0 is 5.
        let denom = 5.0 - 0.75 + 1.0;
        let want_tie = numerics::normal_quantile((2.0 - 0.375) / denom);
        assert!((z[0][1] - want_tie).abs() < 1e-12);
        assert!((z[0][2] - want_tie).abs() < 1e-12);
        assert!((z[0][3] - want_tie).abs() < 1e-12);
        assert!(z[0][0] < z[0][4], "5.0 must rank below 9.0");
    }

    /// EVERY chain frozen at its own value — the 0%-acceptance deadlock.
    ///
    /// R̂ is mathematically `+∞` here (within-chain variance exactly zero,
    /// between-chain variance positive), and that is what `posterior` reports.
    /// The trap is that a naive `Σ(x−μ)²/(n−1)` does NOT give `+∞`: `μ` does
    /// not round-trip through the summation, so a constant chain leaves a
    /// ~1e-32 residue and R̂ comes back as a FINITE ~1e15. That number passes
    /// every `is_finite` check, and its magnitude is set by the array shape,
    /// not the chains — measured in ArviZ, five different frozen-value pairs
    /// give the identical `3372237941944279.0` while changing the draw count
    /// changes it.
    ///
    /// So: report the infinity, not the artifact — and carry the REASON, which
    /// is what camdl adds over `posterior`'s bare `Inf`.
    #[test]
    fn all_chains_frozen_reports_infinity_with_a_reason_not_a_shape_artifact() {
        let frozen: Vec<Vec<f64>> = vec![vec![0.239349270; 30], vec![0.438170322; 30]];
        let d = rank_convergence(&frozen).expect("frozen chains are still scored");

        assert!(d.all_chains_frozen, "the cause must be recorded, not just the symptom");
        assert!(!d.rhat.is_finite(),
            "R̂ must not be a finite shape-determined number; got {}", d.rhat);

        // ESS stays computable and is still reported: `posterior` reverted
        // per-chain constancy checking for ESS in #198 as overly conservative.
        assert!(d.ess_bulk.is_finite() && d.ess_bulk > 0.0,
            "ESS is still reported for a frozen parameter; got {}", d.ess_bulk);
        assert!(d.ess_bulk < 10.0,
            "and it is small, which is the honest answer; got {}", d.ess_bulk);

        // The magnitude must not depend on the VALUES the chains are stuck at.
        let other: Vec<Vec<f64>> = vec![vec![1.0; 30], vec![2.0; 30]];
        let e = rank_convergence(&other).expect("scored");
        assert_eq!(d.rhat.is_finite(), e.rhat.is_finite(),
            "two different frozen pairs must not differ in whether R̂ is finite");

        // Negative control: one frozen chain among moving ones is a FINDING,
        // not a frozen fit — `posterior` reports 1.5268 for that shape and so
        // must camdl (pinned against the oracle by `one_stuck_chain`).
        let mut mixed: Vec<Vec<f64>> = (0..3)
            .map(|c| (0..40).map(|i| ((i * 13 + c * 29) % 47) as f64 / 47.0).collect())
            .collect();
        mixed.push(vec![0.37; 40]);
        let m = rank_convergence(&mixed).expect("scored");
        assert!(!m.all_chains_frozen, "one frozen chain is not all of them");
        assert!(m.rhat.is_finite(), "and the statistic is still defined: {}", m.rhat);
    }

    /// R's `max` propagates `NA`; `f64::max` returns the non-NaN operand. When
    /// the folded half is undefined — `|x − median(x)|` constant, which a
    /// two-point symmetric marginal produces — the headline must be undefined
    /// too, not the bulk value smuggled through.
    #[test]
    fn an_undefined_folded_half_makes_the_headline_undefined() {
        let two_point: Vec<Vec<f64>> = vec![vec![0.239349270; 30], vec![0.438170322; 30]];
        let d = rank_convergence(&two_point).expect("scored");
        assert!(d.rhat.is_nan(),
            "posterior returns NA for this shape (folded half constant); got {}", d.rhat);
        assert!(d.rhat_folded.is_nan(), "the folded half is the undefined one");
        assert!(d.rhat_bulk.is_infinite(),
            "while the bulk half is a well-defined +inf; got {}", d.rhat_bulk);
    }

    /// A chain frozen at one value is not a reason to refuse the parameter —
    /// the other chains still carry information, and the frozen chain is
    /// exactly what R̂ should be loud about.
    #[test]
    fn one_frozen_chain_still_scores() {
        let mut chains: Vec<Vec<f64>> = (0..3)
            .map(|c| (0..80).map(|i| ((i * 13 + c * 29) % 47) as f64 / 47.0).collect())
            .collect();
        chains.push(vec![0.37; 80]);
        let d = rank_convergence(&chains).expect("a frozen chain must not refuse the parameter");
        assert!(d.rhat.is_finite() && d.rhat > 1.1,
            "a frozen chain must show up as disagreement, got R̂ = {}", d.rhat);
        assert!(d.ess_bulk.is_finite() && d.ess_bulk > 0.0);
    }
}
