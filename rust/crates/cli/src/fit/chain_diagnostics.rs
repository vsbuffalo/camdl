//! Per-chain log-likelihood comparison for the samplers and for IF2's starts.
//!
//! `fit summary`'s aggregate R̂/ESS answer "did the chains agree?" but not
//! "which chain differs?". This module compares each chain's mean
//! log-likelihood with the pooled mean of the other chains, and scales the
//! difference by its own Monte Carlo uncertainty, so a chain is named only when
//! its difference from the others is larger than the chains' sampling error can
//! explain. It corroborates R̂ and ESS; it is not the convergence verdict.
//!
//! Read-side only. Everything here is computed at `fit summary` time from the
//! committed trace files (or, for IF2, from `fit_state.toml`'s clean-eval
//! record); nothing flows into a `run_id` / CAS key.
//!
//! One shared seam, and one obligation on everyone who feeds it. The scout
//! gate message (`gating::format_decibans_spread_verdict`) and the summary
//! table both score chains through [`assess_chain_logliks`], so "what counts
//! as a chain that differs" is defined in exactly one place. That function
//! scores whatever estimates it is handed: **the caller owns the claim that
//! they are comparable across chains, and that each standard error is the
//! uncertainty of its estimate.** The scout gate hands it IF2 clean-eval
//! marginals `log p(y | θ)` with the standard error of their replicate mean;
//! this module's reader hands it the mean of whatever column
//! [`LoglikType::chain_agreement_column`] nominates, with that mean's Monte
//! Carlo standard error. Anything else — most of all a per-chain quantity
//! evaluated at that chain's own latent path — is not an input.
//!
//! ## Which column, and why not by position (gh#667)
//!
//! Every Bayesian sampler streams its trace through
//! [`super::trace_writer::TraceWriter`], whose fixed layout is
//! `<index> <loglik> log_posterior …`. That made trace column index 1 look
//! like a safe structural invariant to key on, and this module keyed on it.
//! It is not safe: the *name* in that slot is `log_likelihood` for pmmh / mh /
//! nuts — a marginal `log p(y | θ)`, comparable — but `log_complete_data_ll`
//! for PGAS, the joint `log p(y, X | θ)` over the data **and the sampled
//! latent path**, which is not. Two different quantities in one slot, and the
//! substitution was invisible precisely because nothing named the column.
//!
//! Scoring chains on the PGAS value ranks them mostly by the latent-path term
//! `log p(X | θ)`, a density at one sampled path: a θ whose path distribution
//! is more concentrated scores higher on every typical path without fitting
//! the data any better. On the 60,000-sweep fit behind gh#667 the between-
//! chain spread was 522 nats in the path term and 9 nats in the observation
//! term — the flag would have named chains that were sampling correctly.
//!
//! So the reader keys on the column NAME, chosen per sampler by
//! [`LoglikType::chain_agreement_column`]: `obs_ll` (`log p(y | X, θ)`) for
//! PGAS, `log_likelihood` for the marginal samplers. PGAS's own target and its
//! transition term are still read and displayed — see
//! [`CompleteDataMeans`] — they are just never compared.
//!
//! ## The statistic: leave-one-out difference over its Monte Carlo error
//!
//! For chain `i` among the `k` chains that can be assessed,
//!
//! ```text
//!   Z_i = |m_i − m_{−i}| / sqrt(se_i² + se_{−i}²)
//! ```
//!
//! where `m_i` is chain `i`'s mean log-likelihood over its retained draws,
//! `m_{−i}` is the mean over the retained draws of the other chains in the
//! comparison — every assessed chain not already flagged, see below — each
//! chain's mean weighted by its number of draws, `se_i` is the Monte
//! Carlo standard error of `m_i`, and `se_{−i}² = Σ_{j≠i} w_j² se_j²` with
//! `w_j = n_j / Σ_{l≠i} n_l` is the standard error of the pooled mean — the
//! chains are independent, so the variances of a weighted mean add with the
//! weights squared. A chain is flagged when `Z_i` exceeds
//! [`chain_loglik_cutoff`]`(k)`, a Bonferroni two-sided normal cutoff at
//! family-wise `α` = [`CHAIN_LOGLIK_FAMILY_ALPHA`].
//!
//! `se_i = sd_i / sqrt(ESS_i)`, where `sd_i` is the sample standard deviation
//! of chain `i`'s retained log-likelihood trace and `ESS_i` is its effective
//! sample size from [`sim::inference::pmmh::mcmc_ess`] — Geyer's (1992)
//! initial-positive-sequence estimator over the trace's autocorrelation, the
//! per-chain ESS `compute_rhat_ess` reports. The trace's own autocorrelation
//! is what makes this an MCMC standard error rather than an iid one: a sticky
//! chain has few effective draws and a correspondingly wide `se_i`.
//!
//! ### Why this statistic, and not a robust z-score of the chain means
//!
//! The earlier statistic was the Iglewicz–Hoaglin (1993) modified z-score of
//! the chain means, scaled by their median absolute deviation. With four
//! chains that scale is the median of four numbers' deviations, unstable and
//! unrelated to how precisely each mean is known: when the chains agree
//! closely the scale is a few hundredths of a nat and any chain 0.1 nat away
//! scores beyond the cutoff. It flagged a chain on a converged PMMH fit
//! (R̂ 1.002, bulk ESS ≈ 470) for a 0.13-nat difference (gh#1008). The
//! leave-one-out `Z_i` compares a discrepancy with its own uncertainty, so it
//! needs no scale estimated from the chains' spread. No median/MAD guard is
//! kept beside it: its only job was to supply that scale.
//!
//! ### What it can and cannot say
//!
//! A flagged chain's mean log-likelihood differs from the other chains' by
//! more than Monte Carlo error explains. That is consistent with a chain
//! sampling another region of the posterior (a side mode, a ridge along a
//! weakly identified parameter) or one that has not yet mixed; the statistic
//! alone does not say which, and the message never claims either as fact.
//!
//! - **A PMMH trace includes the particle filter's noise.** The recorded
//!   log-likelihood of an accepted state is a particle-filter estimate, and
//!   its error is frozen into the chain until the next acceptance. The trace's
//!   autocorrelation, and so `se_i`, reflects that noise as the chain sees it:
//!   long runs of a lucky overestimate make the chain sticky and widen `se_i`.
//! - **One chain off does not flag the rest.** A chain far from the others
//!   moves the pooled mean every other chain is compared with: with five
//!   chains at −100 and one at −400, each good chain's "others" average −160,
//!   and every chain would score far beyond the cutoff. So the chain with the
//!   largest `Z` is flagged and removed, and the rest are compared again
//!   without it, until no remaining chain exceeds the cutoff (backward
//!   elimination, as in Rosner's (1983) generalized ESD procedure, here with
//!   the cutoff held at its Bonferroni value for the original `k`). The first
//!   step is the test above over all `k` chains, so a fit whose chains all
//!   agree is flagged with probability at most `α`. When two chains remain and
//!   differ, both are flagged: neither is the odd one out.
//! - **Chains that cannot be assessed are said to be.** A chain with no
//!   readable trace, a non-finite mean, or fewer than
//!   [`CHAIN_LOGLIK_MIN_ESS`] effective draws (a chain that never moved has
//!   one) is reported "not assessed", neither passed nor flagged, and is left
//!   out of the other chains' pooled mean. A chain frozen at one state is
//!   caught by the point-mass and `-inf` screens of this module instead.
//! - **IF2 is an optimiser.** Its chains' final log-likelihoods are evaluated
//!   at different θ; the gate compares them with their evaluation noise (the
//!   standard error of the clean-eval replicate mean) to ask whether one start
//!   reached a better optimum than the others beyond that noise.

use std::collections::BTreeMap;
use std::path::Path;

use crate::chain_index::ChainIndex;

use super::loglik::{LoglikType, TRACE_COL_COMPLETE_DATA_LL, TRACE_COL_TRANSITION_LL};

/// The `TraceWriter` layout's third column, present in every sampler's trace.
/// Its presence is the structural check that a file is a trace at all, and it
/// is itself read by the degeneracy screen (gh#608).
const TRACE_COL_LOG_POSTERIOR: &str = "log_posterior";

/// Family-wise false-alarm rate of the per-chain log-likelihood comparison:
/// the probability, over all `k` chains of a well-mixed fit, that any one of
/// them is flagged by Monte Carlo error alone. One in a thousand fits.
pub const CHAIN_LOGLIK_FAMILY_ALPHA: f64 = 0.001;

/// The fewest effective draws a chain's trace must carry for its Monte Carlo
/// standard error to be estimated. Below this the autocorrelation sum behind
/// the ESS rests on a handful of effective draws and the standard error is
/// itself too uncertain to scale a test by, so the chain is not assessed
/// rather than passed or flagged.
pub const CHAIN_LOGLIK_MIN_ESS: f64 = 10.0;

/// The two-sided cutoff on `Z_i` when `k` chains are compared:
/// `z* = Φ⁻¹(1 − α / (2k))` with `α` = [`CHAIN_LOGLIK_FAMILY_ALPHA`].
///
/// Bonferroni over the `k` leave-one-out tests, each two-sided: each test is
/// run at level `α / k`, split `α / (2k)` per tail, so the chance that any
/// chain of a well-mixed fit is flagged is at most `α` whatever the
/// dependence between the tests (they share chains, so they are dependent).
/// `k = 2` gives 3.48, `k = 4` gives 3.66, `k = 8` gives 3.84. Computed as
/// `−Φ⁻¹(α / (2k))`, the lower tail, which keeps full precision where
/// `1 − α / (2k)` would round.
pub fn chain_loglik_cutoff(k: usize) -> f64 {
    -sim::inference::normal_quantile(CHAIN_LOGLIK_FAMILY_ALPHA / (2.0 * k.max(1) as f64))
}

/// Why a chain was neither passed nor flagged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NotAssessed {
    /// The chain has no readable trace, or its mean log-likelihood is not
    /// finite (a trace with `-inf` rows, gh#608).
    NonFiniteMean,
    /// The chain's trace carries fewer than [`CHAIN_LOGLIK_MIN_ESS`]
    /// effective draws, too few to estimate the standard error of its mean.
    TooFewEffectiveDraws { ess: f64 },
    /// No usable standard error was supplied (IF2: `fit_state.toml` records
    /// no clean-eval standard error for the chain, or one that is zero or not
    /// finite).
    NoStandardError,
    /// Fewer than two chains could be assessed, so there are no others to
    /// compare it with.
    TooFewChains,
}

impl NotAssessed {
    /// The reason, as a clause the per-chain table's note prints.
    pub fn describe(&self) -> String {
        match self {
            NotAssessed::NonFiniteMean => "the mean log-likelihood is not a finite number".into(),
            NotAssessed::TooFewEffectiveDraws { ess } => format!(
                "the trace has {ess:.1} effective draws, fewer than the {CHAIN_LOGLIK_MIN_ESS:.0} \
                 needed to estimate the Monte Carlo error of the mean"
            ),
            NotAssessed::NoStandardError => {
                "no usable standard error of the log-likelihood is recorded".into()
            }
            NotAssessed::TooFewChains => "fewer than two chains could be assessed".into(),
        }
    }
}

/// One chain's mean log-likelihood and the standard error of that mean: the
/// input [`assess_chain_logliks`] compares.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLoglikEstimate {
    /// The chain's number, as its `chain_N/` directory carries it (gh#990).
    pub chain: ChainIndex,
    /// The mean log-likelihood. `NaN` when the chain has no readable trace.
    pub mean: f64,
    /// How much the mean counts in the other chains' pooled mean: its number
    /// of retained draws for a sampler, 1 for an IF2 start.
    pub weight: f64,
    /// The standard error of `mean`, or why there is none.
    pub se: Result<f64, NotAssessed>,
}

/// One chain's leave-one-out comparison, when it could be made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainLoglikTest {
    /// `m_i − m_{−i}`: the chain's mean minus the pooled mean of the others,
    /// in nats. Negative when the chain sits below the others.
    pub diff: f64,
    /// `sqrt(se_i² + se_{−i}²)`, the standard error of `diff`.
    pub se: f64,
    /// `|diff| / se`.
    pub z: f64,
    /// The cutoff `z` was compared with, [`chain_loglik_cutoff`] for the
    /// number of chains assessed.
    pub cutoff: f64,
    /// `z > cutoff`.
    pub flagged: bool,
}

/// The outcome for one chain: compared, or why not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainLoglikOutcome {
    Assessed(ChainLoglikTest),
    NotAssessed(NotAssessed),
}

/// One chain's mean log-likelihood and its comparison with the other chains.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLoglikScore {
    /// The chain's number — the one its `chain_N/` directory and every other
    /// artifact carry (gh#990), read from the directory, never from a row's
    /// position (gh#913).
    pub chain: ChainIndex,
    /// The chain's mean log-likelihood. `NaN` when the chain had no readable
    /// trace rows (rendered `—`, never counted as agreeing).
    pub mean_loglik: f64,
    pub outcome: ChainLoglikOutcome,
}

impl ChainLoglikScore {
    /// The comparison, when it was made.
    pub fn test(&self) -> Option<&ChainLoglikTest> {
        match &self.outcome {
            ChainLoglikOutcome::Assessed(t) => Some(t),
            ChainLoglikOutcome::NotAssessed(_) => None,
        }
    }

    /// Whether this chain's difference from the others exceeds the cutoff.
    /// `false` for a chain not assessed — which is not a pass; see
    /// [`ChainLoglikOutcome::NotAssessed`].
    pub fn is_flagged(&self) -> bool {
        self.test().is_some_and(|t| t.flagged)
    }
}

/// The Monte Carlo standard error of the mean of one chain's log-likelihood
/// trace: `sd / sqrt(ESS)`, with `ESS` from
/// [`sim::inference::pmmh::mcmc_ess`] (Geyer's initial-positive-sequence
/// estimator, the per-chain ESS `compute_rhat_ess` reports) and `sd` the
/// sample standard deviation. A trace with a non-finite value has no finite
/// mean; one with fewer than [`CHAIN_LOGLIK_MIN_ESS`] effective draws has no
/// trustworthy standard error.
pub fn mcse_of_mean(series: &[f64]) -> Result<f64, NotAssessed> {
    if series.is_empty() || series.iter().any(|v| !v.is_finite()) {
        return Err(NotAssessed::NonFiniteMean);
    }
    let n = series.len();
    let ess = sim::inference::pmmh::mcmc_ess(series);
    if !ess.is_finite() || ess < CHAIN_LOGLIK_MIN_ESS || n < 2 {
        return Err(NotAssessed::TooFewEffectiveDraws { ess });
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    Ok((var / ess).sqrt())
}

/// Compare each chain's mean log-likelihood with the pooled mean of the
/// others, as the module docs define, in the order given.
///
/// Only chains with a finite mean, a finite positive standard error and a
/// finite positive weight take part; the rest are [`NotAssessed`] and do not
/// enter the others' pooled mean. With fewer than two such chains no chain
/// is assessed.
///
/// Flagged chains are removed one at a time, largest `Z` first, and the rest
/// re-compared without them (see "One chain off does not flag the rest" in
/// the module docs). Each flagged chain carries the comparison it was flagged
/// on; each chain left unflagged carries its comparison with the other
/// unflagged chains.
pub fn assess_chain_logliks(estimates: &[ChainLoglikEstimate]) -> Vec<ChainLoglikScore> {
    let usable = |e: &ChainLoglikEstimate| -> Result<f64, NotAssessed> {
        if !e.mean.is_finite() {
            return Err(NotAssessed::NonFiniteMean);
        }
        let se = e.se?;
        if !(se.is_finite() && se > 0.0) {
            return Err(NotAssessed::NoStandardError);
        }
        if !(e.weight.is_finite() && e.weight > 0.0) {
            return Err(NotAssessed::NonFiniteMean);
        }
        Ok(se)
    };
    let mut outcomes: Vec<ChainLoglikOutcome> = estimates
        .iter()
        .map(|e| match usable(e) {
            Ok(_) => ChainLoglikOutcome::NotAssessed(NotAssessed::TooFewChains),
            Err(why) => ChainLoglikOutcome::NotAssessed(why),
        })
        .collect();
    let mut active: Vec<usize> =
        (0..estimates.len()).filter(|&i| usable(&estimates[i]).is_ok()).collect();
    let k = active.len();
    if k >= 2 {
        let cutoff = chain_loglik_cutoff(k);
        let se_of = |i: usize| usable(&estimates[i]).expect("active chains are usable");
        // Chain `i` against the pooled mean of the other active chains.
        let compare = |i: usize, active: &[usize]| -> ChainLoglikTest {
            let w_total: f64 =
                active.iter().filter(|&&j| j != i).map(|&j| estimates[j].weight).sum();
            let (mut m_rest, mut var_rest) = (0.0, 0.0);
            for &j in active.iter().filter(|&&j| j != i) {
                let w = estimates[j].weight / w_total;
                m_rest += w * estimates[j].mean;
                var_rest += w * w * se_of(j).powi(2);
            }
            let diff = estimates[i].mean - m_rest;
            let se = (se_of(i).powi(2) + var_rest).sqrt();
            let z = diff.abs() / se;
            ChainLoglikTest { diff, se, z, cutoff, flagged: z > cutoff }
        };
        loop {
            let tests: Vec<(usize, ChainLoglikTest)> =
                active.iter().map(|&i| (i, compare(i, &active))).collect();
            let worst = tests
                .iter()
                .max_by(|a, b| a.1.z.partial_cmp(&b.1.z).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .expect("at least two active chains");
            if !worst.1.flagged {
                for (i, t) in tests {
                    outcomes[i] = ChainLoglikOutcome::Assessed(t);
                }
                break;
            }
            outcomes[worst.0] = ChainLoglikOutcome::Assessed(worst.1);
            active.retain(|&i| i != worst.0);
            if active.len() < 2 {
                // The last chain left differs from the one just removed by the
                // same amount, so neither is the odd one out.
                for &i in &active {
                    outcomes[i] = ChainLoglikOutcome::Assessed(worst.1.mirrored());
                }
                break;
            }
        }
    }
    estimates
        .iter()
        .zip(outcomes)
        .map(|(e, outcome)| ChainLoglikScore { chain: e.chain, mean_loglik: e.mean, outcome })
        .collect()
}

impl ChainLoglikTest {
    /// The same comparison seen from the other chain of a pair.
    fn mirrored(self) -> Self {
        ChainLoglikTest { diff: -self.diff, ..self }
    }
}

/// The flagged chains, largest `Z` first. Empty when no chain exceeds the
/// cutoff.
pub fn flagged_chains(scores: &[ChainLoglikScore]) -> Vec<&ChainLoglikScore> {
    let mut flagged: Vec<&ChainLoglikScore> = scores.iter().filter(|s| s.is_flagged()).collect();
    let z = |s: &ChainLoglikScore| s.test().map_or(0.0, |t| t.z);
    flagged.sort_by(|a, b| z(b).partial_cmp(&z(a)).unwrap_or(std::cmp::Ordering::Equal));
    flagged
}

/// What was observed for each flagged chain, one sentence each: the
/// difference in nats, its `Z` and the cutoff. Says nothing about cause.
pub fn describe_flagged(scores: &[ChainLoglikScore], column: &str) -> Vec<String> {
    flagged_chains(scores)
        .into_iter()
        .filter_map(|s| {
            let t = s.test()?;
            Some(format!(
                "chain {}'s mean {column} differs from the other chains' by {:+.2} nats \
                 (Z = {:.1} > {:.2}), beyond what Monte Carlo error explains",
                s.chain, t.diff, t.z, t.cutoff
            ))
        })
        .collect()
}

/// What the per-chain table shows, and what it is allowed to rank on.
///
/// [`scored`](Self::scored) is the only column the chain comparison reads; the
/// rest is display. Splitting them in the type is the point — it is what stops
/// the sampler's own objective from being mistaken for a cross-chain
/// comparison again (gh#667).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLoglikMeans {
    /// The trace column [`scored`](Self::scored) was read from, for labelling
    /// the table and for saying which column is missing when it is.
    pub scored_column: &'static str,
    /// The chains, by the number of their `chain_N/` directory, ascending —
    /// parallel to [`scored`](Self::scored) and to the `complete_data`
    /// columns. Carried so a leaf with a gap labels each row with its own
    /// chain (gh#913).
    pub chain_ids: Vec<ChainIndex>,
    /// Per-chain mean of `scored_column` over the retained draws, parallel to
    /// [`chain_ids`](Self::chain_ids). `NaN` when a chain has no readable rows.
    pub scored: Vec<f64>,
    /// `true` when no discovered trace carries `scored_column` at all — a
    /// method whose chains cannot be compared, which the caller must SAY rather
    /// than render as a table of dashes.
    pub scored_column_absent: bool,
    /// PGAS only: the sampler's own target and its latent-path term, shown
    /// beside the scored column and never ranked on. `None` for the marginal
    /// samplers, where the scored column already IS the target.
    pub complete_data: Option<CompleteDataMeans>,
}

/// PGAS's complete-data target and its latent-path term, per chain — the two
/// quantities gh#667 removed from the ranking but deliberately kept visible.
/// `log_complete_data_ll` is the sampler's own Gibbs target, so a chain that
/// has fallen off the joint support shows up here; `transition_ll` is what
/// makes the diagnosis obvious, because a wide spread there next to a tight
/// spread in `obs_ll` is the entropy effect rather than a fit difference.
#[derive(Debug, Clone, PartialEq)]
pub struct CompleteDataMeans {
    /// Per-chain mean `log p(y, X | θ)` (`log_complete_data_ll`).
    pub complete: Vec<f64>,
    /// Per-chain mean `log p(X | θ)` (`transition_ll`).
    pub transition: Vec<f64>,
}

/// One chain's retained log-likelihood series, by column: `None` where its
/// trace is unreadable or lacks the column.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSeries {
    pub chain: ChainIndex,
    pub scored: Option<Vec<f64>>,
    pub complete: Option<Vec<f64>>,
    pub transition: Option<Vec<f64>>,
}

/// Every chain's retained log-likelihood series, as the traces store them —
/// the data the per-chain means and the outlier scores are computed from.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLoglikSeries {
    pub scored_column: &'static str,
    pub chains: Vec<ChainSeries>,
    /// PGAS's complete-data split is carried ([`CompleteDataMeans`]).
    pub wants_split: bool,
    /// A readable trace exists and none names the scored column.
    pub scored_column_absent: bool,
}

/// Read every `chain_*/trace.tsv` under `method_dir`, keeping each chain's
/// post-burn-in tail, ordered by ascending chain number.
///
/// `kind` is the method's log-likelihood class, which decides *by name* which
/// column the chains may be compared on — see
/// [`LoglikType::chain_agreement_column`] and this module's header for why a
/// column *position* is the wrong key (gh#667).
///
/// Returns `None` when no `chain_*/trace.tsv` files exist (the sampler wrote no
/// per-chain trace — e.g. an optimizer-only method) so the caller can say so
/// rather than silently show an empty table.
///
/// ## Post-burn-in selection
///
/// The post-burn-in kept draws are a suffix of the trace, and `draws.tsv`
/// records exactly how many there are per chain, so we take the **last `K_c`
/// rows** of chain `c`'s trace (`K_c` = that chain's row count in `draws.tsv`):
///
/// - **nuts** — the trace is written post-warmup only, `draws.tsv` has the same
///   `n_samples` rows: last `K_c` == all rows (exact).
/// - **pmmh / mh** — the trace carries a warm-up prefix (`step < burn_in`, one
///   row per step) then the thinned sampling suffix; `draws.tsv` counts the
///   suffix, so the last `K_c` rows are exactly the post-burn-in draws (exact).
/// - **pgas** — the trace carries every sweep (warm-up + all sampling,
///   unthinned); `draws.tsv` counts the thinned post-burn-in draws, so the last
///   `K_c` sweeps are a post-burn-in tail window (`K_c ≤ n_sweeps − burn_in`,
///   so warm-up is never included — a subset of the post-burn-in region when
///   thinning is on).
///
/// When `draws.tsv` is absent or unreadable (a partial / hand-built method) we
/// fall back to the whole trace: without the posterior manifest we cannot strip
/// warm-up, so the mean is noisier but still honest. This never contaminates a
/// well-formed fit.
pub fn read_chain_loglik_series(method_dir: &Path, kind: LoglikType) -> Option<ChainLoglikSeries> {
    let chain_dirs = discover_chain_dirs(method_dir);
    if chain_dirs.is_empty() {
        return None;
    }

    // Per-chain kept-draw counts from draws.tsv, keyed by the chain number —
    // the same number the directory carries (gh#990) — so a chain with no
    // draws pairs with nothing rather than with its neighbour's count.
    let draw_counts = read_draw_counts(&method_dir.join("draws.tsv"));

    let scored_column = kind.chain_agreement_column();
    // The complete-data split is PGAS's alone: the marginal samplers have no
    // latent path to separate out, and their scored column IS their target.
    // Exhaustive, not `!is_marginal()` — a future non-marginal class (an
    // observation-conditional, say) would not thereby acquire PGAS's columns.
    let wants_split = match kind {
        LoglikType::CompleteData => true,
        LoglikType::If2 | LoglikType::Marginal | LoglikType::OdeMarginal => false,
    };
    let names: Vec<&str> = if wants_split {
        vec![scored_column, TRACE_COL_COMPLETE_DATA_LL, TRACE_COL_TRANSITION_LL]
    } else {
        vec![scored_column]
    };

    // "Absent" means a readable trace exists and none of them names the scored
    // column — distinct from "no readable trace at all", which surfaces as NaN
    // means. Starts `false` so an unreadable method never claims the column is
    // missing.
    let mut any_header_read = false;
    let mut any_scored_column = false;
    let mut chains = Vec::with_capacity(chain_dirs.len());
    for (chain, dir) in &chain_dirs {
        let cols = read_trace_cols(&dir.join("trace.tsv"), &names);
        if let Some(cols) = cols.as_ref() {
            any_header_read = true;
            if cols[0].is_some() {
                any_scored_column = true;
            }
        }
        let k = draw_counts.get(chain).copied();
        let tail = |slot: usize| -> Option<Vec<f64>> {
            let all = cols.as_ref()?.get(slot)?.as_ref()?;
            Some(retained(all, k).to_vec())
        };
        chains.push(ChainSeries {
            chain: *chain,
            scored: tail(0),
            complete: if wants_split { tail(1) } else { None },
            transition: if wants_split { tail(2) } else { None },
        });
    }
    Some(ChainLoglikSeries {
        scored_column,
        chains,
        wants_split,
        scored_column_absent: any_header_read && !any_scored_column,
    })
}

/// The per-chain means of [`ChainLoglikSeries`], ordered by chain number.
/// The run and `fit summary` both call this, on the values as the traces
/// store them.
pub fn chain_mean_logliks(series: &ChainLoglikSeries) -> ChainLoglikMeans {
    let mean = |v: &Option<Vec<f64>>| v.as_deref().map_or(f64::NAN, mean_or_nan);
    ChainLoglikMeans {
        scored_column: series.scored_column,
        chain_ids: series.chains.iter().map(|c| c.chain).collect(),
        scored: series.chains.iter().map(|c| mean(&c.scored)).collect(),
        scored_column_absent: series.scored_column_absent,
        complete_data: series.wants_split.then(|| CompleteDataMeans {
            complete: series.chains.iter().map(|c| mean(&c.complete)).collect(),
            transition: series.chains.iter().map(|c| mean(&c.transition)).collect(),
        }),
    }
}

/// Each chain's mean of the scored column with its Monte Carlo standard error
/// ([`mcse_of_mean`]), weighted by its retained draws — the input
/// [`assess_chain_logliks`] takes. A chain whose trace lacks the column has a
/// `NaN` mean and is not assessed.
pub fn chain_loglik_estimates(series: &ChainLoglikSeries) -> Vec<ChainLoglikEstimate> {
    series
        .chains
        .iter()
        .map(|c| match c.scored.as_deref() {
            Some(v) => ChainLoglikEstimate {
                chain: c.chain,
                mean: mean_or_nan(v),
                weight: v.len() as f64,
                se: mcse_of_mean(v),
            },
            None => ChainLoglikEstimate {
                chain: c.chain,
                mean: f64::NAN,
                weight: 0.0,
                se: Err(NotAssessed::NonFiniteMean),
            },
        })
        .collect()
}

/// The per-chain means the table shows, and each chain's comparison with the
/// others, from one read of the traces. `None` when no chain trace exists.
/// The comparison is empty when it cannot be made at all — the scored column
/// is absent from every trace, or fewer than two chains have traces — which
/// the caller says rather than rendering as a table of chains that agree.
pub fn read_chain_loglik_scores(
    method_dir: &Path,
    kind: LoglikType,
) -> Option<(ChainLoglikMeans, Vec<ChainLoglikScore>)> {
    let series = read_chain_loglik_series(method_dir, kind)?;
    let means = chain_mean_logliks(&series);
    let scores = if !means.scored_column_absent && means.scored.len() >= 2 {
        assess_chain_logliks(&chain_loglik_estimates(&series))
    } else {
        Vec::new()
    };
    Some((means, scores))
}

/// The retained tail of one chain's column: the last `k` rows when
/// `draws.tsv` gave a count that fits, else the whole trace (no manifest means
/// we cannot strip warm-up, so the mean is noisier but still honest).
fn retained(all: &[f64], k: Option<usize>) -> &[f64] {
    match k {
        Some(k) if k > 0 && k <= all.len() => &all[all.len() - k..],
        _ => all,
    }
}

/// The mean, or `NaN` for an empty selection — never silently 0.
fn mean_or_nan(selected: &[f64]) -> f64 {
    if selected.is_empty() {
        f64::NAN
    } else {
        selected.iter().sum::<f64>() / selected.len() as f64
    }
}

/// Collect `(chain, dir)` for every `chain_<N>` subdirectory of `method_dir`,
/// in ascending chain order. The number is the chain's label everywhere
/// (gh#990), and it is carried with the directory rather than re-derived from
/// the position in this list, so a leaf with a gap labels each directory as
/// itself (gh#913).
fn discover_chain_dirs(method_dir: &Path) -> Vec<(ChainIndex, std::path::PathBuf)> {
    let mut dirs = Vec::new();
    let Ok(entries) = std::fs::read_dir(method_dir) else {
        return dirs;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(chain) = ChainIndex::parse_dir_name(name) {
            dirs.push((chain, path));
        }
    }
    dirs.sort_by_key(|(c, _)| *c);
    dirs
}

/// Read the NAMED columns of one per-chain trace, row-aligned, as f64 rows.
///
/// `None` when the file is missing/unreadable or does not look like a
/// [`super::trace_writer::TraceWriter`] trace (its header must carry
/// `log_posterior`); that guard is what separates "not a trace" from "a trace
/// without this column". Otherwise one slot per requested name, in order, with
/// `None` for a name the header does not have — so a caller can tell a missing
/// column from an empty one.
///
/// A row is skipped for EVERY column when any *present* requested field is
/// short or unparseable, which keeps the returned vectors aligned with one
/// another (a caller reading `obs_ll` beside `transition_ll` compares the same
/// sweeps).
fn read_trace_cols(trace_path: &Path, names: &[&str]) -> Option<Vec<Option<Vec<f64>>>> {
    let contents = std::fs::read_to_string(trace_path).ok()?;
    let mut lines = contents.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split('\t').collect();
    if !cols.contains(&TRACE_COL_LOG_POSTERIOR) {
        return None;
    }
    let idx: Vec<Option<usize>> = names
        .iter()
        .map(|n| cols.iter().position(|c| c == n))
        .collect();
    let mut out: Vec<Option<Vec<f64>>> =
        idx.iter().map(|i| i.map(|_| Vec::new())).collect();
    let mut row: Vec<f64> = Vec::with_capacity(names.len());
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        row.clear();
        let complete = idx.iter().flatten().all(|&c| {
            match fields.get(c).and_then(|s| s.parse::<f64>().ok()) {
                Some(v) => {
                    row.push(v);
                    true
                }
                None => false,
            }
        });
        if !complete {
            continue;
        }
        for (slot, v) in out.iter_mut().filter_map(|s| s.as_mut()).zip(row.iter()) {
            slot.push(*v);
        }
    }
    Some(out)
}

/// One chain's stuck-state screen (gh#608, ebola F8): over the RETAINED
/// draws (the same last-K_c rule as [`read_chain_loglik_series`]), how many
/// rows record a non-finite log-posterior as the chain's CURRENT state. A
/// Metropolis-within-Gibbs chain whose current state has zero posterior
/// density is incoherent (gh#607) — ANY such retained row marks the chain
/// degenerate, and its draws contaminate `draws.tsv` and every pooled
/// number. Exclusion stays explicit (gh#419): this screen only makes the
/// contamination impossible to miss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainNegInf {
    /// The chain's number, as its directory and the per-chain table carry it.
    pub chain: ChainIndex,
    /// Retained rows whose log-posterior is non-finite.
    pub n_neginf: usize,
    /// Retained rows for this chain.
    pub n_retained: usize,
}

/// One chain's point-mass screen (gh#635, ebola item 1): over `draws.tsv`,
/// how many DISTINCT parameter vectors the chain retained. A chain that
/// never accepted a θ-move keeps exactly one — a point mass at its start,
/// which the −inf screen (gh#608) cannot see when that start has FINITE
/// density. Its draws still enter every pooled number (3 such chains were
/// 37.5% of a pooled cloud downstream, R̂ 5.14, masquerading as a mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainUniqueDraws {
    /// The chain's number, as its directory and the per-chain table carry it.
    pub chain: ChainIndex,
    /// Distinct parameter vectors among the retained draws.
    pub n_unique: usize,
    /// Retained draws for this chain.
    pub n_draws: usize,
}

/// Per-chain distinct-parameter-vector counts from `draws.tsv`, keyed by its
/// `chain` column; vectors are compared as their exact on-disk text (the
/// writer round-trips full precision, so textual equality is value
/// equality). `None` when draws.tsv is missing/unreadable.
pub fn read_chain_unique_draws(method_dir: &Path) -> Option<Vec<ChainUniqueDraws>> {
    let contents = std::fs::read_to_string(method_dir.join("draws.tsv")).ok()?;
    let mut lines = contents.lines();
    let header = lines.next()?;
    let mut cols = header.split('\t');
    if cols.next() != Some("chain") {
        return None;
    }
    use std::collections::{BTreeMap, HashSet};
    let mut per_chain: BTreeMap<usize, (HashSet<&str>, usize)> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let Some((chain_tok, rest)) = line.split_once('\t') else { continue };
        let Ok(c) = chain_tok.parse::<usize>() else { continue };
        // Strip the draw index (column 2): the PARAMS are what repeat.
        let params = rest.split_once('\t').map(|(_, p)| p).unwrap_or(rest);
        let e = per_chain.entry(c).or_default();
        e.0.insert(params);
        e.1 += 1;
    }
    if per_chain.is_empty() {
        return None;
    }
    Some(per_chain.into_iter().map(|(c, (set, n))| ChainUniqueDraws {
        chain: ChainIndex(c),
        n_unique: set.len(),
        n_draws: n,
    }).collect())
}

/// Per-chain [`ChainNegInf`] screen over the retained trace rows. `None`
/// when no chain traces exist (same discovery as the mean-loglik reader).
pub fn read_chain_neginf(method_dir: &Path) -> Option<Vec<ChainNegInf>> {
    let chain_dirs = discover_chain_dirs(method_dir);
    if chain_dirs.is_empty() {
        return None;
    }
    let draw_counts = read_draw_counts(&method_dir.join("draws.tsv"));
    let mut out = Vec::with_capacity(chain_dirs.len());
    for (chain, dir) in &chain_dirs {
        let cols = read_trace_cols(&dir.join("trace.tsv"), &[TRACE_COL_LOG_POSTERIOR]);
        let all: &[f64] = match cols.as_ref().and_then(|c| c[0].as_ref()) {
            Some(v) => v,
            None => &[],
        };
        let selected: &[f64] = match draw_counts.get(chain).copied() {
            Some(k) if k > 0 && k <= all.len() => &all[all.len() - k..],
            _ => all,
        };
        out.push(ChainNegInf {
            chain: *chain,
            n_neginf: selected.iter().filter(|v| !v.is_finite()).count(),
            n_retained: selected.len(),
        });
    }
    Some(out)
}

/// Read `draws.tsv` and return the per-chain kept-draw count, keyed by the
/// `chain` column. Empty when the file is missing or has no `chain` column.
fn read_draw_counts(draws_path: &Path) -> BTreeMap<ChainIndex, usize> {
    let Ok(contents) = std::fs::read_to_string(draws_path) else {
        return BTreeMap::new();
    };
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return BTreeMap::new();
    };
    let cols: Vec<&str> = header.split('\t').collect();
    let Some(chain_col) = cols.iter().position(|c| *c == "chain") else {
        return BTreeMap::new();
    };
    let mut counts: BTreeMap<ChainIndex, usize> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if let Some(id) = fields.get(chain_col).and_then(|s| s.parse::<ChainIndex>().ok()) {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    /// Estimates for contiguous chains counted from 0, one draw's weight each,
    /// every mean known to the same standard error.
    fn estimates(means: &[f64], se: f64) -> Vec<ChainLoglikEstimate> {
        means
            .iter()
            .enumerate()
            .map(|(i, &m)| ChainLoglikEstimate { chain: ChainIndex(i), mean: m, weight: 1.0, se: Ok(se) })
            .collect()
    }

    /// The per-chain means the table shows, as `fit summary` reads them.
    fn means_of(dir: &std::path::Path, kind: LoglikType) -> Option<ChainLoglikMeans> {
        read_chain_loglik_scores(dir, kind).map(|(m, _)| m)
    }

    /// The flagged chains' numbers, largest `Z` first.
    fn flagged(scores: &[ChainLoglikScore]) -> Vec<usize> {
        flagged_chains(scores).iter().map(|s| s.chain.0).collect()
    }

    fn zs(scores: &[ChainLoglikScore]) -> Vec<f64> {
        scores.iter().map(|s| s.test().map_or(f64::NAN, |t| t.z)).collect()
    }

    /// A stationary AR(1) trace of length `n` with lag-one autocorrelation
    /// `phi` and marginal standard deviation `sd`, shifted so its sample mean
    /// is exactly `mean`. Its integrated autocorrelation time is
    /// `(1 + phi) / (1 − phi)`, so its ESS is near `n (1 − phi) / (1 + phi)`.
    fn ar1_trace(seed: u64, n: usize, phi: f64, sd: f64, mean: f64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let innov = sd * (1.0 - phi * phi).sqrt();
        let z0: f64 = StandardNormal.sample(&mut rng);
        let mut x = sd * z0;
        let mut v: Vec<f64> = (0..n)
            .map(|_| {
                let e: f64 = StandardNormal.sample(&mut rng);
                x = phi * x + innov * e;
                x
            })
            .collect();
        let shift = mean - v.iter().sum::<f64>() / n as f64;
        v.iter_mut().for_each(|x| *x += shift);
        v
    }

    fn trace_estimates(traces: &[Vec<f64>]) -> Vec<ChainLoglikEstimate> {
        traces
            .iter()
            .enumerate()
            .map(|(i, v)| ChainLoglikEstimate {
                chain: ChainIndex(i),
                mean: v.iter().sum::<f64>() / v.len() as f64,
                weight: v.len() as f64,
                se: mcse_of_mean(v),
            })
            .collect()
    }

    /// The cutoff is the Bonferroni two-sided normal quantile at family-wise
    /// α = 0.001: Φ⁻¹(1 − 0.001 / (2k)). Reference values from Python's
    /// `statistics.NormalDist().inv_cdf`.
    #[test]
    fn the_cutoff_is_bonferroni_at_one_in_a_thousand() {
        for (k, want) in [(2, 3.480756), (3, 3.587915), (4, 3.662260), (8, 3.836107)] {
            let got = chain_loglik_cutoff(k);
            assert!((got - want).abs() < 1e-5, "k = {k}: {got} vs {want}");
        }
    }

    /// The standard error of a trace's mean follows its autocorrelation: an
    /// iid trace's is close to sd/√n, and an AR(1) trace at φ = 0.9 (ESS ≈
    /// n/19) has one about √19 ≈ 4.4 times larger.
    #[test]
    fn mcse_follows_the_traces_autocorrelation() {
        let n = 20_000;
        let iid = ar1_trace(1, n, 0.0, 2.0, -50.0);
        let se_iid = mcse_of_mean(&iid).unwrap();
        let naive = 2.0 / (n as f64).sqrt();
        assert!((se_iid / naive - 1.0).abs() < 0.1, "iid: {se_iid} vs {naive}");
        let sticky = ar1_trace(2, n, 0.9, 2.0, -50.0);
        let ratio = mcse_of_mean(&sticky).unwrap() / naive;
        assert!((ratio - 19f64.sqrt()).abs() < 0.8, "AR(1) φ = 0.9: ratio {ratio}");
    }

    /// The four retained log-likelihood traces of the gh#1008 fit, by chain.
    fn gh1008_traces() -> Vec<Vec<f64>> {
        let text = include_str!("../../tests/fixtures/gh1008_pmmh_chain_logliks.tsv");
        let mut traces: Vec<Vec<f64>> = vec![Vec::new(); 4];
        for line in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
            let (c, v) = line.split_once('\t').expect("chain<TAB>log_likelihood");
            traces[c.parse::<usize>().unwrap()].push(v.parse().unwrap());
        }
        traces
    }

    /// gh#1008, on the fit itself: four PMMH chains whose mean
    /// log-likelihoods are −59.47, −59.60, −59.44 and −59.48 (R̂ 1.002, bulk
    /// ESS ≈ 470), the traces from `tests/fixtures/`. The 0.13-nat difference
    /// of chain 1 is ordinary Monte Carlo error and nothing is flagged.
    #[test]
    fn gh1008_a_013_nat_difference_on_a_converged_fit_is_not_flagged() {
        let traces = gh1008_traces();
        assert!(traces.iter().all(|t| t.len() == 200), "fixture premise: 200 draws per chain");
        let scores = assess_chain_logliks(&trace_estimates(&traces));
        let means: Vec<f64> = scores.iter().map(|s| s.mean_loglik).collect();
        for (got, want) in means.iter().zip([-59.47, -59.60, -59.44, -59.48]) {
            assert!((got - want).abs() < 0.005, "fixture premise: the issue's means, {means:?}");
        }
        assert!(scores.iter().all(|s| s.test().is_some()), "every chain is assessed: {scores:?}");
        assert!(flagged(&scores).is_empty(), "no chain flagged; Z = {:?}", zs(&scores));
        let t1 = scores[1].test().unwrap();
        assert!((t1.diff - (-0.13)).abs() < 0.01, "chain 1 sits 0.13 nats below the rest: {t1:?}");
        assert!(t1.z < 2.0 && (t1.cutoff - 3.662).abs() < 1e-3, "{t1:?}");
    }

    /// The same fit with chain 1's trace moved 15 nats down — a chain sampling
    /// another region of the likelihood surface. It is flagged, and only it:
    /// set aside, the other three agree.
    #[test]
    fn a_clearly_stuck_chain_is_flagged_and_only_it() {
        let mut traces = gh1008_traces();
        traces[1].iter_mut().for_each(|v| *v -= 15.0);
        let scores = assess_chain_logliks(&trace_estimates(&traces));
        assert_eq!(flagged(&scores), vec![1], "Z = {:?}", zs(&scores));
        let t = scores[1].test().unwrap();
        assert!((t.diff - (-15.13)).abs() < 0.01 && t.z > 30.0, "{t:?}");
        let msg = describe_flagged(&scores, "log_likelihood");
        assert_eq!(msg.len(), 1);
        assert!(msg[0].starts_with("chain 1's mean log_likelihood differs from the other chains' by -15.13 nats"),
            "{msg:?}");
        assert!(msg[0].contains("beyond what Monte Carlo error explains"), "{msg:?}");
    }

    /// A chain whose trace has fewer than ten effective draws is not
    /// assessed — neither flagged nor passed — and is left out of the others'
    /// pooled mean, so the three chains that can be compared still are.
    #[test]
    fn a_chain_with_too_few_effective_draws_is_not_assessed() {
        let mut traces: Vec<Vec<f64>> = [-59.47, -59.44, -59.48]
            .iter()
            .enumerate()
            .map(|(i, &m)| ar1_trace(7 + i as u64, 200, 0.26, 1.95, m))
            .collect();
        // Eight draws far from the rest: a mean that would flag, on too little
        // evidence to say so.
        traces.push(ar1_trace(99, 8, 0.0, 1.95, -90.0));
        let scores = assess_chain_logliks(&trace_estimates(&traces));
        assert!(
            matches!(scores[3].outcome,
                ChainLoglikOutcome::NotAssessed(NotAssessed::TooFewEffectiveDraws { ess }) if ess < 10.0),
            "{:?}", scores[3]
        );
        assert!(!scores[3].is_flagged());
        assert!(scores[..3].iter().all(|s| s.test().is_some_and(|t| !t.flagged)), "{scores:?}");
        // A chain that never moved has one effective draw.
        assert!(matches!(mcse_of_mean(&[-60.0; 50]),
            Err(NotAssessed::TooFewEffectiveDraws { ess }) if ess == 1.0));
    }

    #[test]
    fn flags_a_single_clear_outlier_and_not_the_chains_it_pulls() {
        // Five chains near −100, one stuck far below at −400. Without
        // elimination each good chain's "others" would average −160.
        let scores = assess_chain_logliks(&estimates(&[-100.0, -101.0, -99.0, -100.5, -100.2, -400.0], 0.5));
        assert_eq!(flagged(&scores), vec![5], "Z = {:?}", zs(&scores));
        assert!(scores[..5].iter().all(|s| s.test().is_some()), "the rest are compared among themselves");
    }

    /// Two co-stuck chains among six are both flagged and the four good ones
    /// are not: removing the worst first keeps the pair from masking each
    /// other or condemning the rest.
    #[test]
    fn four_good_two_stuck_flags_both_stuck_chains() {
        let scores = assess_chain_logliks(&estimates(&[-1204.0, -1205.0, -1203.0, -1206.0, -1461.0, -1462.0], 1.0));
        let mut f = flagged(&scores);
        f.sort();
        assert_eq!(f, vec![4, 5], "Z = {:?}", zs(&scores));
    }

    /// Two chains that differ are both flagged: with two there is no odd one
    /// out, and the comparison is the same from either side.
    #[test]
    fn two_chains_that_differ_are_both_flagged() {
        let scores = assess_chain_logliks(&estimates(&[-50.0, -80.0], 1.0));
        let mut f = flagged(&scores);
        f.sort();
        assert_eq!(f, vec![0, 1]);
        let (a, b) = (scores[0].test().unwrap(), scores[1].test().unwrap());
        assert_eq!(a.z, b.z);
        assert_eq!(a.diff, -b.diff);
        assert!(a.diff > 0.0, "chain 0 sits above chain 1: {a:?}");
    }

    #[test]
    fn flags_nothing_when_well_mixed() {
        // Negative control: the chains agree within their standard errors, and
        // every Z is finite and small — not a vacuous "nothing assessed" pass.
        let scores = assess_chain_logliks(&estimates(&[-100.0, -100.3, -99.8, -100.1, -99.9, -100.2], 0.5));
        assert!(flagged(&scores).is_empty());
        assert!(scores.iter().all(|s| s.test().is_some_and(|t| t.z.is_finite() && t.z < 1.5)), "{:?}", zs(&scores));
    }

    #[test]
    fn identical_chains_score_zero() {
        let scores = assess_chain_logliks(&estimates(&[-10.0; 4], 0.5));
        assert!(scores.iter().all(|s| s.test().is_some_and(|t| t.z == 0.0 && !t.flagged)));
    }

    #[test]
    fn a_single_chain_has_nothing_to_be_compared_with() {
        let scores = assess_chain_logliks(&estimates(&[-42.0], 0.5));
        assert_eq!(scores[0].outcome, ChainLoglikOutcome::NotAssessed(NotAssessed::TooFewChains));
    }

    /// A chain with an unreadable trace (NaN mean) is not assessed rather than
    /// read as agreeing, and does not enter the others' pooled mean; the real
    /// outlier among the finite chains is still flagged.
    #[test]
    fn a_non_finite_mean_is_not_assessed_and_not_pooled() {
        let scores = assess_chain_logliks(&estimates(&[-50.0, -51.0, -49.0, f64::NAN, -400.0], 0.5));
        assert_eq!(scores[3].outcome, ChainLoglikOutcome::NotAssessed(NotAssessed::NonFiniteMean));
        assert_eq!(flagged(&scores), vec![4], "Z = {:?}", zs(&scores));
    }

    /// IF2's replicate standard errors: a zero or missing one is "no standard
    /// error", not an infinitely precise estimate.
    #[test]
    fn a_zero_standard_error_is_not_assessed() {
        let mut est = estimates(&[-60.0, -61.0, -300.0], 0.5);
        est[0].se = Ok(0.0);
        let scores = assess_chain_logliks(&est);
        assert_eq!(scores[0].outcome, ChainLoglikOutcome::NotAssessed(NotAssessed::NoStandardError));
        assert_eq!(flagged(&scores), vec![1, 2], "the two remaining chains differ: {:?}", zs(&scores));
    }

    /// The reader: a six-chain method dir (`chain_N/trace.tsv`, one chain far
    /// off) plus a `draws.tsv` manifest, with a warm-up prefix in each trace
    /// that must be stripped. Three retained rows per chain are too few
    /// effective draws to compare on, so every chain reads "not assessed" —
    /// said, not passed.
    #[test]
    fn reader_strips_warmup_and_does_not_assess_three_draws() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_reader");
        std::fs::create_dir_all(&dir).unwrap();

        let warmup = "1\t-900.0\t-905.0\n2\t-880.0\t-885.0\n";
        // Directories count from 0 (every sampler's convention, gh#990).
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            let body = format!("step\tlog_likelihood\tlog_posterior\n{warmup}{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        write_trace(0, "3\t-50.0\t-52.0\n4\t-50.0\t-52.0\n5\t-50.0\t-52.0\n"); // mean -50.0
        write_trace(1, "3\t-50.5\t-52.0\n4\t-50.5\t-52.0\n5\t-50.5\t-52.0\n"); // mean -50.5
        write_trace(2, "3\t-49.5\t-51.0\n4\t-49.5\t-51.0\n5\t-49.5\t-51.0\n"); // mean -49.5
        write_trace(3, "3\t-50.2\t-52.0\n4\t-50.2\t-52.0\n5\t-50.2\t-52.0\n"); // mean -50.2
        write_trace(4, "3\t-49.8\t-51.0\n4\t-49.8\t-51.0\n5\t-49.8\t-51.0\n"); // mean -49.8
        write_trace(5, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // ≈ -300

        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        // A marginal sampler: trace column 1 is `log_likelihood` = log p(y | θ),
        // already comparable across chains, so gh#667 leaves this path alone.
        let (means, scores) = read_chain_loglik_scores(&dir, LoglikType::Marginal)
            .expect("chain traces present");
        assert_eq!(means.scored_column, "log_likelihood");
        assert!(means.complete_data.is_none(), "no latent-path split for a marginal sampler");
        assert_eq!(means.chain_ids, ChainIndex::all(6).collect::<Vec<_>>());
        let m = &means.scored;
        assert_eq!(m.len(), 6);
        // Warm-up (-900/-880) excluded → good chains ≈ -50, not dragged to -890.
        for (i, v) in m.iter().take(5).enumerate() {
            assert!((v - (-50.0)).abs() < 1.0, "chain {i} mean {v} ≈ -50 (warm-up stripped)");
        }
        assert!((m[5] - (-300.0)).abs() < 1.0, "chain 5 mean {} ≈ -300", m[5]);
        assert_eq!(scores.len(), 6);
        assert!(scores.iter().all(|s| matches!(s.outcome,
            ChainLoglikOutcome::NotAssessed(NotAssessed::TooFewEffectiveDraws { .. }))), "{scores:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reader_returns_none_without_chain_dirs() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(means_of(&dir, LoglikType::Marginal).is_none(),
            "no chain_* dirs → None");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A six-chain PGAS method whose complete-data ranking and observation-only
    /// ranking DISAGREE, so the fixture discriminates between them.
    ///
    /// - chain 2 has a hugely concentrated latent path (`transition_ll` −800 vs
    ///   ≈ −3000 elsewhere) and a perfectly ordinary data fit.
    /// - chain 5 has an ordinary latent path and reproduces the data ≈450 nats
    ///   worse than every other chain.
    ///
    /// Compared on `log_complete_data_ll`, chain 2 stands out — a chain that
    /// fits the data as well as any other. Compared on `obs_ll`, chain 5 does.
    fn write_disagreeing_pgas_method(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        // (transition_ll, obs_ll) per chain; complete = transition + obs.
        let chains = [
            (-2832.6, -952.0),
            (-2933.0, -952.8),
            (-800.0, -951.7),    // concentrated path, ordinary data fit
            (-3100.9, -952.9),
            (-3172.6, -952.3),
            (-3000.0, -1400.0),  // ordinary path, BAD data fit
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{i}"));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            let mut body = String::from(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n",
            );
            // Two warm-up rows the last-K_c rule must strip (draws.tsv keeps 3).
            for s in 0..2 {
                body.push_str(&format!("{s}\t-9000\t-9100\t-8000\t-1000\n"));
            }
            for s in 2..5 {
                body.push_str(&format!(
                    "{s}\t{complete}\t{}\t{trans}\t{obs}\n",
                    complete - 1.0
                ));
            }
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        }
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.{}{}\n", c, d));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();
    }


    /// gh#667: a PGAS chain is scored on `obs_ll` = `log p(y | X, θ)` — "does
    /// this chain reproduce the data" — not on `log_complete_data_ll`, whose
    /// latent-path term is a density at one sampled path and rewards a
    /// concentrated path distribution rather than a better fit.
    #[test]
    fn pgas_chains_are_scored_on_obs_ll_not_the_complete_data_target() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667");
        write_disagreeing_pgas_method(&dir);

        let means = means_of(&dir, LoglikType::CompleteData)
            .expect("six chain traces");
        assert_eq!(means.scored_column, "obs_ll",
            "PGAS chains are compared on log p(y | X, θ)");
        assert!(!means.scored_column_absent);
        assert_eq!(means.scored.len(), 6);

        // The scored quantity is obs_ll (warm-up stripped: −1000 never enters).
        assert!((means.scored[5] - (-1400.0)).abs() < 1e-6,
            "chain 5 must be scored on its obs_ll (−1400), got {}", means.scored[5]);
        assert!((means.scored[2] - (-951.7)).abs() < 1e-6,
            "chain 2 must be scored on its obs_ll (−951.7), got {}", means.scored[2]);

        // …and the flag follows it: the chain that fits the data worst. The
        // fixture's three draws per chain carry no standard error, so one is
        // supplied here; the column, not the error model, is under test.
        let on = |column: &[f64]| {
            let est: Vec<ChainLoglikEstimate> = means.chain_ids.iter().zip(column)
                .map(|(&chain, &mean)| ChainLoglikEstimate { chain, mean, weight: 3.0, se: Ok(2.0) })
                .collect();
            assess_chain_logliks(&est)
        };
        let scores = on(&means.scored);
        assert_eq!(flagged(&scores), vec![5], "the badly-fitting chain: {:?}", zs(&scores));

        // The fixture discriminates: on the complete-data column the old
        // reader ranked on, the chain that stands out is the other one. It is
        // still read (kept visible in the table), just never compared.
        let split = means.complete_data.as_ref().expect("PGAS carries the split");
        assert!((split.complete[2] - (-1751.7)).abs() < 1e-6,
            "chain 2's complete-data target is still read: {}", split.complete[2]);
        assert!((split.transition[2] - (-800.0)).abs() < 1e-6,
            "chain 2's transition term is still read: {}", split.transition[2]);
        let cd = on(&split.complete);
        assert_eq!(flagged(&cd).first(), Some(&2),
            "precondition: the complete-data column singles out the concentrated-path chain \
             first, not the badly-fitting one: {:?}", zs(&cd));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: the split shown beside the scored column is the real
    /// decomposition — 522 nats of between-chain spread in `transition_ll`
    /// against 9 in `obs_ll` is what makes "this is an entropy effect, not a
    /// fit difference" legible. Uses the six chains of the real 60,000-sweep
    /// fit in the issue, whose complete-data spread is dominated by the
    /// latent-path term.
    #[test]
    fn transition_and_obs_spreads_are_read_separately() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667_split");
        std::fs::create_dir_all(&dir).unwrap();
        // The issue's table: (transition_ll, obs_ll) for the 6 finite chains.
        let chains = [
            (-2832.6, -952.0), (-2933.0, -952.8), (-2972.0, -951.7),
            (-3100.9, -958.0), (-3172.6, -952.9), (-3354.4, -960.4),
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{i}"));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            std::fs::write(cd.join("trace.tsv"), format!(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
                 0\t{complete}\t{}\t{trans}\t{obs}\n", complete - 1.0)).unwrap();
        }
        let means = means_of(&dir, LoglikType::CompleteData).expect("traces");
        let split = means.complete_data.as_ref().expect("PGAS split");
        let range = |v: &[f64]| {
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        assert!((range(&split.transition) - 521.8).abs() < 0.1,
            "latent-path spread ≈ 522 nats, got {}", range(&split.transition));
        assert!((range(&means.scored) - 8.7).abs() < 0.1,
            "observation spread ≈ 9 nats, got {}", range(&means.scored));
        // One sweep per chain has no Monte Carlo error to compare on.
        let (_, scores) = read_chain_loglik_scores(&dir, LoglikType::CompleteData).unwrap();
        assert!(scores.iter().all(|s| s.test().is_none()), "{scores:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Negative control for gh#667: when one chain is off on the latent path
    /// and also reproduces the data far worse, comparing on `obs_ll` still
    /// names it. Narrowing the compared quantity must not cost the diagnostic
    /// the cases it already caught.
    #[test]
    fn a_chain_bad_on_both_terms_is_still_flagged() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667_agree");
        std::fs::create_dir_all(&dir).unwrap();
        let chains = [
            (-2900.0, -952.0), (-2910.0, -952.8), (-2890.0, -951.7),
            (-2905.0, -952.9), (-2895.0, -952.3), (-3900.0, -1400.0),
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{i}"));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            std::fs::write(cd.join("trace.tsv"), format!(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
                 0\t{complete}\t{}\t{trans}\t{obs}\n", complete - 1.0)).unwrap();
        }
        let means = means_of(&dir, LoglikType::CompleteData).expect("traces");
        let est: Vec<ChainLoglikEstimate> = means.chain_ids.iter().zip(&means.scored)
            .map(|(&chain, &mean)| ChainLoglikEstimate { chain, mean, weight: 1.0, se: Ok(1.0) })
            .collect();
        assert_eq!(flagged(&assess_chain_logliks(&est)), vec![5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: the reader keys on the column NAME. A PGAS trace whose position-1
    /// column is `log_complete_data_ll` must be read on `obs_ll` — the position
    /// is not consulted, so the two quantities cannot be substituted for each
    /// other. The fixture makes them numerically distinguishable on purpose.
    #[test]
    fn reader_keys_on_the_name_so_position_1_cannot_stand_in() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_pgas");
        let cd = dir.join("chain_0");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(
            cd.join("trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
             0\t-77.0\t-79.0\t-55.0\t-22.0\n1\t-77.0\t-79.0\t-55.0\t-22.0\n",
        )
        .unwrap();
        let means = means_of(&dir, LoglikType::CompleteData).expect("one chain");
        assert!((means.scored[0] - (-22.0)).abs() < 1e-9,
            "must score obs_ll (-22), not the position-1 column (-77): {}", means.scored[0]);
        let split = means.complete_data.as_ref().expect("PGAS split");
        assert!((split.complete[0] - (-77.0)).abs() < 1e-9);
        assert!((split.transition[0] - (-55.0)).abs() < 1e-9);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A PGAS trace with no `obs_ll` column cannot be scored at all — the
    /// reader says the column is absent rather than falling back to whatever
    /// sits in position 1, which is exactly the substitution gh#667 removed.
    #[test]
    fn missing_scored_column_is_reported_not_silently_substituted() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_no_obs");
        let cd = dir.join("chain_0");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(
            cd.join("trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\n0\t-77.0\t-79.0\n1\t-77.0\t-79.0\n",
        )
        .unwrap();
        let means = means_of(&dir, LoglikType::CompleteData).expect("one chain");
        assert!(means.scored_column_absent, "the absence must be REPORTED");
        assert!(means.scored[0].is_nan(),
            "no obs_ll → no score; never the complete-data value: {}", means.scored[0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file that is not a `TraceWriter` trace at all (no `log_posterior`)
    /// yields NaN means, and is NOT mistaken for "a trace missing a column".
    #[test]
    fn a_non_trace_file_is_not_a_missing_column() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_not_trace");
        let cd = dir.join("chain_1");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(cd.join("trace.tsv"), "a\tb\tc\n1\t2\t3\n").unwrap();
        let means = means_of(&dir, LoglikType::Marginal).expect("one chain dir");
        assert!(!means.scored_column_absent,
            "an unreadable trace is not evidence about which columns a trace has");
        assert!(means.scored[0].is_nan());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#913: a per-chain mean is labelled with the number parsed from its
    /// `chain_N/` directory, so a leaf with a gap — a chain that died before
    /// writing anything, or a hand-pruned leaf — attributes each mean to its
    /// own chain rather than renumbering the survivors by position.
    #[test]
    fn chain_means_are_labelled_by_directory_number_not_position() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gap");
        std::fs::create_dir_all(&dir).unwrap();
        // chain_1 never ran; chains 0 and 2 did.
        for (c, ll) in [(0usize, -50.0), (2usize, -300.0)] {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            std::fs::write(
                cd.join("trace.tsv"),
                format!("step\tlog_likelihood\tlog_posterior\n1\t{ll}\t{ll}\n2\t{ll}\t{ll}\n"),
            )
            .unwrap();
        }
        std::fs::write(
            dir.join("draws.tsv"),
            "chain\tdraw\tbeta\n0\t0\t0.5\n0\t1\t0.5\n2\t0\t0.5\n2\t1\t0.5\n",
        )
        .unwrap();
        let (means, scores) = read_chain_loglik_scores(&dir, LoglikType::Marginal).expect("traces");
        assert_eq!(means.chain_ids, vec![ChainIndex(0), ChainIndex(2)]);
        let labels: Vec<ChainIndex> = scores.iter().map(|s| s.chain).collect();
        assert_eq!(labels, vec![ChainIndex(0), ChainIndex(2)],
            "each mean carries its own directory's number");
        std::fs::remove_dir_all(&dir).ok();
    }
}
