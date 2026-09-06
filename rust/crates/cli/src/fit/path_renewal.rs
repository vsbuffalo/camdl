//! gh#791: the `path_renewal` block of `pgas_summary.json` — trajectory
//! renewal resolved in time, its two derived numbers, and the
//! ancestor-sampling acceptance rate read beside them.
//!
//! ## Why the aggregate cannot answer this
//!
//! [`CSMCDiagnostics::trajectory_renewal`] is a weighted mean over the bins of
//! [`CSMCDiagnostics::renewal_by_bin`], and its late terms are high in most
//! runs: the traceback's lineages have not yet coalesced by the time it reaches
//! the late states, so the tail of the path renews freely. Those terms hold the
//! mean up. A conditional-SMC genealogy that has coalesced — every lineage
//! traced back to the same ancestor by the time the traceback reaches the EARLY
//! states, so the early path is held at the reference — therefore scores an
//! aggregate that reads healthy.
//!
//! What drives the high late bins is coalescence depth, measured rather than
//! assumed: truncating a run's observation series to its first 40% left the
//! gradient at 0.122 against 0.141 for the full series, so the free segment
//! after the final observation is **not** the mechanism.
//!
//! Measured on an 11-compartment stochastic Ebola model, 103 daily observation
//! times, chain-binomial backend, post-burn-in means across 16 chains:
//!
//! ```text
//!   b0    b1    b2    b3    b4    b5    b6    b7    b8    b9   aggregate
//!   0.03  0.03  0.03  0.03  0.03  0.03  0.53  0.84  0.87  0.98     0.336
//! ```
//!
//! An aggregate of 0.34 reads as "a third of the path renews per sweep". What
//! is happening is that the first sixty percent of the series — the whole
//! initial condition and the early dynamics — changes in 3% of sweeps. The
//! parameters whose likelihood lives in that prefix do not converge; the
//! observation-model parameters, whose likelihood is spread across the whole
//! window, do. Nothing else in the diagnostics moves when the prefix freezes.
//!
//! ## What this module publishes
//!
//! The profile itself, plus two numbers a reader can act on:
//!
//! - **prefix renewal** — the mean over the first half of the bins;
//! - **renewal gradient** — last bin minus first bin, the coalescence
//!   signature: near 0 when renewal is uniform in time, near 1 when the
//!   genealogy has fully coalesced.
//!
//! and the **ancestor-sampling acceptance rate** in the same block, because the
//! two are only legible side by side. On the probe above it was 0.0160 at 4,800
//! particles and 0.0157 at 19,200 — ancestor sampling contributes essentially
//! nothing there and more particles do not change that, which is what you
//! expect for an integer compartment state whose ancestor weight is sharply
//! peaked and often exactly zero on support grounds. Renewal over the early
//! bins meanwhile went from 0.07 to 0.21-0.24 over the same change, so the
//! frozen prefix was particle-limited rather than geometric.
//!
//! ## What is averaged, and over what
//!
//! Each entry of [`PathRenewal::bins`] is the mean **of the per-sweep bin
//! fractions** over every post-burn-in retained sweep of every surviving chain
//! — that is, the mean down the corresponding `renewal_b<n>` column of
//! `trace.tsv`, restricted to the retained sweeps. It is deliberately not a
//! pooled ratio (renewed substeps over total substeps in that bin across
//! sweeps): the two differ whenever bins hold unequal substep counts, and the
//! column mean is the number a reader can reproduce from the trace by hand.
//!
//! A sweep whose bin holds no substep contributes nothing to that bin rather
//! than contributing a zero — the convention `renewal_by_bin` already uses, for
//! the same reason: "no substep fell here" and "no substep here was renewed"
//! are different diagnoses.

use serde::{Deserialize, Serialize};

use sim::inference::diagnostic::DiagnosticKind;
use sim::inference::pgas::{CSMCDiagnostics, LOG_ALPHA_NEAR, RENEWAL_BINS};

/// Leading bins the [`PathRenewal::prefix`] mean spans: the first half of the
/// series, rounded down.
///
/// The first half rather than a hand-picked count, so the number means the same
/// thing under any [`RENEWAL_BINS`]. On the measured runs the frozen region ran
/// to bin 5, so this is a conservative window — it cannot overstate the freeze.
pub const PREFIX_BINS: usize = RENEWAL_BINS / 2;

/// The renewal gradient at or above which the profile is reported as coalesced.
///
/// **An empirical anchor from measured runs, not an operating characteristic.**
/// No operating characteristic for it is available, and none is likely to be: a
/// rule for choosing the particle count `N` in particle Gibbs does not exist
/// either (Chopin & Singh 2015, *Bernoulli* 21:1855-1883, prove uniform
/// ergodicity with no rate in `N`; Lindsten, Jordan & Schön 2014, *JMLR*
/// 15:2145-2184, call informative rates open), which is exactly why both papers
/// recommend reading the update-rate-against-t profile instead of any scalar.
///
/// ## Why the shape and not the level
///
/// `gradient = last_bin − first_bin` needs no notion of a "good" renewal level
/// for a given model, which an absolute prefix bar would and which nobody can
/// currently justify. It also keeps this finding separable from
/// `LowTrajectoryRenewal`: a run renewing poorly but UNIFORMLY in time has a
/// gradient near zero and is a different failure with a different remedy.
///
/// ## Why 0.75 and not the midpoint of the range
///
/// The statistic runs over `[-1, 1]` — ≈0 when renewal is uniform in time, ≈1
/// when the genealogy has fully coalesced — so 0.5 looks like the natural bar.
/// It is not, because on the runs this bar was anchored to the last bin is
/// high, so `gradient ≈ 1 − first_bin` and a bar at the midpoint is an absolute
/// bar on the first bin wearing a disguise. Measured, rather than argued: a
/// 2-chain, 40-particle, 40-substep SIR fit — an ordinary working short run —
/// reads
///
/// ```text
///   b0    b1    b2    b3    b4    b5    b6    b7    b8    b9   gradient
///   0.43  0.31  0.34  0.53  0.38  0.43  0.55  0.84  0.91  0.94     0.51
/// ```
///
/// and a bar at 0.5 reports it. A band that fires on healthy runs buries the
/// real failures underneath it — the gh#631 lesson from this repository's own
/// acceptance-rate diagnostics, where the random-walk band applied to NUTS
/// produced one `severity: error` per parameter per run and hid a genuinely
/// stuck chain in the noise.
///
/// Every run where this failure has actually been diagnosed sits far above
/// that. From gh#791, an 11-compartment stochastic Ebola model, 103 daily
/// observation times, chain-binomial backend, post-burn-in means across chains
/// — four fitted variants and a matched two-point particle-count probe:
///
/// ```text
///   run                        b0     b9   gradient
///   carefix_q070_16c         0.06   0.98      0.92
///   censusnorm_q070_16c      0.03   0.98      0.95
///   ratiosat_q070_16c        0.04   0.98      0.94
///   phasetype_initscale_16c  0.11   0.99      0.88
///   probe,  4,800 particles  0.07   0.98      0.91
///   probe, 19,200 particles  0.21   0.99      0.78
/// ```
///
/// The smallest is 0.78, from the 19,200-particle probe whose early bins had
/// improved to 0.21-0.24 and were still not healthy. So the bar sits at 0.75:
/// below every diagnosed failure, above an ordinary working short fit. That is
/// an anchor on seven measurements, and it is stated as one.
///
/// ## What an independent check found
///
/// Run against 27 PGAS runs on four further model families plus this
/// repository's own `tests/fixtures/polio_afp_es`, the bar holds. Eight
/// converged runs (rank-normalized split R̂ < 1.05, bulk ESS 214-464) read
/// gradient 0.018-0.172; every firing run had R̂ ≥ 1.88; nothing in that cohort
/// landed between 0.172 and 0.758. The anchor generalises beyond the family it
/// was fitted on, and there is no false-positive population in the gap.
///
/// The gap is nonetheless wider than the evidence permits the bar to roam. The
/// working short SIR fit above reads 0.51, which is inside `[0.172, 0.758]` — it
/// was not part of that cohort. So the cohort rules out a bar below 0.172 and
/// above 0.758, and the working run rules out the bottom half of what is left.
/// 0.75 sits near the top of the gap for that reason, not in its middle.
///
/// Two things the same check ruled out as confounds, recorded because they are
/// the first two anyone asks about:
///
/// - **Step size.** Halving `dt` moved the gradient 0.141 → 0.172. The bins are
///   fixed tenths of the substep INDEX, not of wall-clock time, so this is the
///   invariance the fixed-bin choice was made for.
/// - **Series length.** Truncating the observations to the first 40% of the
///   series left it at 0.122 against 0.141. The high last bin is therefore not
///   the free segment after the final observation; it is coalescence depth.
///
/// ## Where `gradient ≈ 1 − first_bin` does not hold, and the blind spot
///
/// That approximation is a property of the anchor family, not of the statistic,
/// and the reasoning above leans on it — so its scope belongs here. Across the
/// 27 runs the LAST bin spans **0.402 to 0.998**, not the 0.98-0.99 of the
/// Ebola table, and the mean absolute discrepancy between the gradient and
/// `1 − first_bin` is **0.146**.
///
/// The consequence is a genuine blind spot, and it is this statistic's own
/// target case: **where the last bin is low, the gradient is bounded below the
/// bar however frozen the prefix is.** Measured — `sirs_T60_N100` has
/// `first_bin = 0.000` and `prefix = 0.001`, which is exactly the failure gh#791
/// exists to catch, and reads gradient 0.402. This finding stays quiet on it.
/// It is caught only because the aggregate is under 0.10 and
/// `LowTrajectoryRenewal` fires. The two rules together cover it; neither does
/// alone, and on a short series it is the aggregate rule doing the work. A
/// reader deserves that up front rather than discovering it.
/// `the_short_series_blind_spot_is_covered_only_by_the_aggregate_rule` pins it.
///
/// Three consequences follow, and are enforced rather than intended:
///
/// - the profile is published unconditionally, in the artifact and in the
///   terminal, so a reader never depends on this number to see the shape;
/// - the finding is `Severity::Warning` at any gradient and gates nothing;
/// - `the_bar_sits_between_the_measured_working_run_and_every_measured_failure`
///   asserts the anchor against the numbers above, so a later change to the
///   value has to move that evidence too.
pub const COALESCENCE_GRADIENT: f64 = 0.75;

/// What [`PathRenewal::bin_span`] says: the sentence a reader needs before
/// `b0 = 0.04` means anything. A constant so the producer and any test
/// asserting the artifact's contract cannot drift.
pub const BIN_SPAN: &str =
    "bin b covers the substeps in the fraction [b/10, (b+1)/10) of the \
     simulated series, a fixed tenth rather than a proportion of the particle \
     count or the observation count, so profiles compare across models and \
     across particle counts; each entry is the mean of the per-sweep bin \
     fractions over the post-burn-in retained sweeps of every surviving chain";

/// The `path_renewal` block of `pgas_summary.json`.
///
/// Every field is derived from `renewal_by_bin` / the ancestor-sampling
/// counters that the sampler already records. Nothing here feeds back into the
/// sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathRenewal {
    /// Bins in the profile — [`RENEWAL_BINS`].
    pub n_bins: usize,
    /// The profile: entry `b` is renewal in the `b`-th tenth of the series,
    /// averaged over sweeps and chains. `null` for a bin no retained sweep ever
    /// recorded a substep in — not `0.0`, which would invent a degeneracy that
    /// was never observed.
    pub bins: Vec<Option<f64>>,
    /// How to read [`Self::bins`], stated in the artifact so a consumer never
    /// has to find it in a code comment.
    pub bin_span: String,
    /// Mean over the first [`Self::n_prefix_bins`] bins — the early window,
    /// where path degeneracy bites first and where the initial condition gets
    /// its information. Averaged over the observed bins among those; `null`
    /// when none of them was observed.
    pub prefix: Option<f64>,
    /// Leading bins [`Self::prefix`] spans. [`PREFIX_BINS`].
    pub n_prefix_bins: usize,
    /// Last bin minus first bin. The coalescence signature — near 0 when
    /// renewal is uniform in time, near 1 when the genealogy has coalesced onto
    /// the reference. `null` when either end bin was not observed.
    pub gradient: Option<f64>,
    /// The aggregate `trajectory_renewal` this profile resolves, over the same
    /// sweeps. Carried so the two are read side by side; it is NOT replaced,
    /// and `trace.tsv` still carries it per sweep.
    pub aggregate: Option<f64>,
    /// Ancestor-sampling Metropolis acceptance rate, pooled over the same
    /// sweeps: `n_as_accepted / n_as_proposed`. `null` when the step never ran
    /// (no alternative ancestor was ever admissible), which is a different
    /// diagnosis from an acceptance rate of 0.
    pub as_accept: Option<f64>,
    /// gh#864. [`Self::as_accept`] resolved into the same ten bins as
    /// [`Self::bins`]: entry `b` is the acceptance rate over the ancestor-
    /// sampling Metropolis steps that ran in the `b`-th tenth of the substep
    /// series. `null` for a bin no retained sweep ever proposed a move in —
    /// not `0.0`, which would report a move that was offered and always
    /// refused.
    ///
    /// Published beside the profile because the pair is the reading: renewal
    /// says where the path stopped moving, this says whether the ancestor move
    /// was ever offered there and whether it landed. A profile that *falls*
    /// toward `b0` says the splice gets harder further back — more subsequent
    /// recorded history has to stay plausible under the new ancestor — while a
    /// *flat* one says the proposal is mismatched uniformly in time. The scalar
    /// [`Self::as_accept`] averages over exactly that difference.
    ///
    /// Aggregated like [`Self::bins`] and *not* like [`Self::as_accept`]: this is
    /// the mean of the per-sweep bin rates, so it is reproducible from the
    /// `as_accept_b<n>` columns of `trace.tsv` by hand and sits on the same
    /// footing as the renewal row above it, while `as_accept` is the pooled
    /// ratio over all proposals. The two therefore need not reconcile exactly —
    /// they weight a sweep's proposals differently — and the pooled scalar
    /// remains the authority on the overall rate.
    pub as_accept_bins: Vec<Option<f64>>,
    /// Substeps at which the Metropolis step actually ran — the denominator of
    /// [`Self::as_accept`].
    pub n_as_proposed: u64,
    /// Of those, the accepted ones.
    pub n_as_accepted: u64,
    /// gh#864. The effective sample size of the ancestor weights before the
    /// `SpliceGuard` mask, pooled over the same sweeps. In particles, not a
    /// fraction: read it against `as_finite_frac × particles` from `trace.tsv`,
    /// the candidates it is an effective count of.
    pub as_ess_pre: SweepMedian,
    /// gh#864. The same after the mask — the effective number of ancestors the
    /// categorical draws from. Read against `as_admissible_frac × particles`.
    ///
    /// Beside [`Self::as_ess_pre`] rather than replacing it: the guard can
    /// lower the candidate count and raise this number in the same step, by
    /// removing a dominant candidate whose splice was backward-infeasible. A
    /// low value here with a high one there is the guard concentrating the
    /// draw; low in both is the density doing it, and the remedies differ.
    pub as_ess_post: SweepMedian,
    /// gh#864. Median over the retained sweeps of each sweep's own median
    /// `log α = log s_prop − log s_ref` — the acceptance ratio behind the
    /// Metropolis decisions, rather than the decisions themselves.
    ///
    /// The reading it settles, which neither the acceptance rate nor the
    /// ancestor-weight ESS can: around −20, the reference's remaining history is
    /// hopeless under any other ancestor, and improving the proposal changes
    /// nothing because the candidates it would pick are refused for the same
    /// reason. Near zero, or far down with [`Self::as_logalpha_near`] well above
    /// zero, the move is close to landing and a proposal that preferred those
    /// candidates would land it.
    pub as_logalpha_median: SweepMedian,
    /// gh#864. Median over the retained sweeps of the fraction of each sweep's
    /// proposals within one nat of parity — how much of the acceptance ratio's
    /// distribution sits close enough that the coin accepts it better than a
    /// third of the time.
    pub as_logalpha_near: SweepMedian,
    /// gh#864. Proposals whose acceptance ratio was a finite number, pooled over
    /// the same sweeps — the sample the two fields above summarise. The gap to
    /// [`Self::n_as_proposed`] is the proposals refused with no ratio at all
    /// (a zero-density spliced suffix), so a `near` fraction is always readable
    /// against how much of the sweep it was measured over.
    pub n_as_logalpha: u64,
    /// Retained post-burn-in sweeps the profile was averaged over, summed
    /// across chains.
    pub n_sweeps: usize,
    /// Surviving chains that contributed at least one sweep.
    pub n_chains: usize,
}

/// One per-sweep ancestor-sampling statistic, pooled over the retained sweeps
/// as a median (gh#864).
///
/// **The median rather than the mean**, for every statistic published this way,
/// and for the same reason each time: these are heavy-tailed quantities where a
/// handful of unrepresentative sweeps moves a mean a long way, and the question
/// being asked is always about the *ordinary* sweep — how many ancestors the
/// categorical can reach on one, how far from accepting its proposals are on
/// one. A mean over the ancestor-weight ESS is pulled up by the few sweeps whose
/// weights happen to spread; a mean over `log α` is pulled down by the hopeless
/// proposals, which is worse, because it then reports "hopeless" of a
/// distribution with real mass near parity.
///
/// The denominator travels with the value because a sweep that measured nothing
/// is absent from the median rather than entered as zero, so `n_sweeps` here can
/// be smaller than [`PathRenewal::n_sweeps`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepMedian {
    /// Median over the sweeps that measured the statistic. `null` when no
    /// retained sweep did, which is a different reading from a measured zero.
    pub median: Option<f64>,
    /// Retained sweeps [`Self::median`] is over.
    pub n_sweeps: usize,
}

/// Accumulator for [`PathRenewal`] over the retained sweeps of every chain.
///
/// A type rather than loose sums at the call site, for the same reason
/// `PositionBins` is one on the sim side: the "skip a bin no substep fell in"
/// rule is then written once, and the profile cannot be updated out of step
/// with the aggregate it resolves.
#[derive(Debug, Clone, Default)]
pub struct PathRenewalAccum {
    bin_sum: [f64; RENEWAL_BINS],
    bin_n: [usize; RENEWAL_BINS],
    /// gh#864: the acceptance profile, accumulated exactly as the renewal
    /// profile above it — a sweep that proposed nothing in bin `b` is skipped
    /// there rather than entered as a zero.
    as_accept_bin_sum: [f64; RENEWAL_BINS],
    as_accept_bin_n: [usize; RENEWAL_BINS],
    aggregate_sum: f64,
    aggregate_n: usize,
    n_sweeps: usize,
    n_chains: usize,
    n_as_proposed: u64,
    n_as_accepted: u64,
    /// gh#864: the per-sweep ancestor-weight ESS, kept rather than summed —
    /// the summary reports a median, which cannot be accumulated in a scalar.
    /// One `f64` per retained sweep per side, which is the same order as the
    /// per-sweep diagnostics already held to compute the profile.
    as_ess_pre: Vec<f64>,
    as_ess_post: Vec<f64>,
    /// gh#864: the per-sweep acceptance-ratio summaries, kept for the same
    /// reason — a median cannot be accumulated in a scalar.
    as_logalpha_median: Vec<f64>,
    as_logalpha_near: Vec<f64>,
    n_as_logalpha: u64,
}

impl PathRenewalAccum {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in one chain's retained post-burn-in sweeps.
    ///
    /// A chain contributing no sweep is not counted in [`PathRenewal::n_chains`]
    /// — it contributed nothing to average, and counting it would report a
    /// denominator the profile does not have.
    pub fn add_chain<'a, I>(&mut self, sweeps: I)
    where
        I: IntoIterator<Item = &'a CSMCDiagnostics>,
    {
        let before = self.n_sweeps;
        for d in sweeps {
            self.add_sweep(d);
        }
        if self.n_sweeps > before {
            self.n_chains += 1;
        }
    }

    /// Fold in one sweep's CSMC diagnostics.
    fn add_sweep(&mut self, d: &CSMCDiagnostics) {
        for (b, &v) in d.renewal_by_bin.iter().enumerate() {
            // Non-finite is "no substep fell in this bin on this sweep". It is
            // skipped, not zeroed: averaging a zero in would report a freeze
            // that was never measured.
            if v.is_finite() {
                self.bin_sum[b] += v;
                self.bin_n[b] += 1;
            }
        }
        // gh#864. Same rule, on the acceptance profile: non-finite is "the
        // Metropolis step never ran in this bin on this sweep", which is not an
        // acceptance rate of zero. Folding those in would report a move that
        // was offered and refused everywhere, on the bins where it was never
        // offered at all — the reading this profile exists to separate.
        for (b, &v) in d.as_accept_by_bin.iter().enumerate() {
            if v.is_finite() {
                self.as_accept_bin_sum[b] += v;
                self.as_accept_bin_n[b] += 1;
            }
        }
        if d.trajectory_renewal.is_finite() {
            self.aggregate_sum += d.trajectory_renewal;
            self.aggregate_n += 1;
        }
        self.n_sweeps += 1;
        self.n_as_proposed += d.n_as_proposed as u64;
        self.n_as_accepted += d.n_as_accepted as u64;
        // gh#864. Non-finite is "this sweep had no ancestor weight to measure"
        // — ancestor sampling off, or no substep drew an ancestry. Dropped, not
        // zeroed, for the same reason a bin holding no substep is: a median
        // taken over invented zeros reports a collapse nothing measured.
        if d.as_ess_pre.is_finite() {
            self.as_ess_pre.push(d.as_ess_pre);
        }
        if d.as_ess_post.is_finite() {
            self.as_ess_post.push(d.as_ess_post);
        }
        // gh#864. Non-finite is "this sweep measured no acceptance ratio" —
        // it proposed nothing, or everything it proposed was refused as
        // zero-density and so carried no ratio at all. Dropped, not zeroed:
        // a `log α` of 0 is parity, the best reading there is, and inventing
        // one for a sweep that measured nothing would report the opposite of
        // what happened.
        if d.as_logalpha_median.is_finite() {
            self.as_logalpha_median.push(d.as_logalpha_median);
        }
        if d.as_logalpha_near.is_finite() {
            self.as_logalpha_near.push(d.as_logalpha_near);
        }
        self.n_as_logalpha += d.n_as_logalpha as u64;
    }

    /// Close the accumulator.
    ///
    /// `None` when no sweep was folded in at all: the block is then omitted
    /// from the summary entirely rather than written as a row of nulls, which a
    /// consumer would read as "measured, found nothing".
    pub fn finish(self) -> Option<PathRenewal> {
        if self.n_sweeps == 0 {
            return None;
        }
        let bins: Vec<Option<f64>> = (0..RENEWAL_BINS)
            .map(|b| (self.bin_n[b] > 0).then(|| self.bin_sum[b] / self.bin_n[b] as f64))
            .collect();
        let as_accept_bins: Vec<Option<f64>> = (0..RENEWAL_BINS)
            .map(|b| (self.as_accept_bin_n[b] > 0)
                .then(|| self.as_accept_bin_sum[b] / self.as_accept_bin_n[b] as f64))
            .collect();
        // Over the OBSERVED bins among the first half. A short series can leave
        // some of them empty (fewer substeps than bins), and averaging over the
        // ones that exist is the honest reading; a consumer that needs to know
        // how many went in can count the non-null entries of `bins[..5]`.
        let prefix = mean_observed(&bins[..PREFIX_BINS]);
        // Both ends, or nothing: a gradient with one end guessed is not the
        // difference it claims to be.
        let gradient = match (bins[RENEWAL_BINS - 1], bins[0]) {
            (Some(last), Some(first)) => Some(last - first),
            _ => None,
        };
        let aggregate = (self.aggregate_n > 0)
            .then(|| self.aggregate_sum / self.aggregate_n as f64);
        let as_accept = (self.n_as_proposed > 0)
            .then(|| self.n_as_accepted as f64 / self.n_as_proposed as f64);
        Some(PathRenewal {
            n_bins: RENEWAL_BINS,
            bins,
            bin_span: BIN_SPAN.to_string(),
            prefix,
            n_prefix_bins: PREFIX_BINS,
            gradient,
            aggregate,
            as_accept,
            as_accept_bins,
            n_as_proposed: self.n_as_proposed,
            n_as_accepted: self.n_as_accepted,
            as_ess_pre: SweepMedian::of(self.as_ess_pre),
            as_ess_post: SweepMedian::of(self.as_ess_post),
            as_logalpha_median: SweepMedian::of(self.as_logalpha_median),
            as_logalpha_near: SweepMedian::of(self.as_logalpha_near),
            n_as_logalpha: self.n_as_logalpha,
            n_sweeps: self.n_sweeps,
            n_chains: self.n_chains,
        })
    }
}

impl SweepMedian {
    /// Close over the per-sweep values collected for one statistic.
    ///
    /// `values` holds only the sweeps that measured it, so the count published
    /// beside the median is the count the median is actually over.
    fn of(mut values: Vec<f64>) -> Self {
        let n_sweeps = values.len();
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite by construction"));
        let median = match n_sweeps {
            0 => None,
            // Even counts average the two central sweeps, the ordinary
            // convention; odd counts take the single centre.
            n if n % 2 == 0 => Some((values[n / 2 - 1] + values[n / 2]) / 2.0),
            n => Some(values[n / 2]),
        };
        SweepMedian { median, n_sweeps }
    }
}

/// Mean of the observed entries, or `None` when none is observed.
fn mean_observed(xs: &[Option<f64>]) -> Option<f64> {
    let (sum, n) = xs.iter().flatten().fold((0.0, 0usize), |(s, n), &x| (s + x, n + 1));
    (n > 0).then(|| sum / n as f64)
}

impl PathRenewal {
    /// The coalescence finding, or `None` when the profile does not show that
    /// shape — or cannot be asked, because one of the two end bins holds no
    /// substep.
    ///
    /// One finding per stage, over the pooled profile, rather than one per
    /// chain: the profile published beside it is the pooled one, and a reader
    /// comparing the message against the artifact must see the same numbers.
    pub fn coalescence_finding(&self) -> Option<DiagnosticKind> {
        let gradient = self.gradient?;
        let first_bin = self.bins[0]?;
        let last_bin = self.bins[self.n_bins - 1]?;
        (gradient >= COALESCENCE_GRADIENT).then_some(DiagnosticKind::PathRenewalCoalesced {
            prefix: self.prefix.unwrap_or(f64::NAN),
            gradient,
            first_bin,
            last_bin,
            aggregate: self.aggregate.unwrap_or(f64::NAN),
            n_bins: self.n_bins,
            n_prefix_bins: self.n_prefix_bins,
        })
    }

    /// The end-of-stage block printed beside the aggregate.
    ///
    /// Prints the whole profile, both derived numbers, the aggregate it
    /// resolves, and the ancestor-sampling acceptance rate — the last of these
    /// adjacent on purpose: "the early path is frozen" and "the ancestor
    /// splice never lands" are only distinguishable when read together.
    pub fn report(&self) -> String {
        let cell = |v: Option<f64>| match v {
            Some(x) => format!("{x:.3}"),
            None => "   NA".to_string(),
        };
        let labels: Vec<String> = (0..self.n_bins).map(|b| format!("   b{b}")).collect();
        let values: Vec<String> = self.bins.iter().map(|&v| cell(v)).collect();
        // gh#864: the acceptance profile is printed directly under the renewal
        // profile, on the same bins, because the pair is the reading — "the
        // prefix does not renew" and "the splice is never accepted there" are
        // one line apart or they are not compared at all.
        let as_accept_values: Vec<String> =
            self.as_accept_bins.iter().map(|&v| cell(v)).collect();
        let mut s = format!(
            "\npath renewal (gh#791; mean over {} retained post-burn-in sweep(s) \
             across {} chain(s)):\n  bin        {}\n  renewal    {}\n  as accept  {}\n",
            self.n_sweeps,
            self.n_chains,
            labels.join(" "),
            values.join(" "),
            as_accept_values.join(" "),
        );
        s.push_str(&format!(
            "  each bin is a fixed tenth of the substep series: b0 is its first \
             tenth, b{} its last;\n  NA is nothing recorded in that bin — no \
             substep, or no ancestor move proposed — and never a measured 0\n",
            self.n_bins - 1,
        ));
        s.push_str(&format!(
            "  prefix (mean b0-b{}): {}   gradient (b{} - b0): {}   \
             aggregate trajectory_renewal: {}\n",
            self.n_prefix_bins - 1,
            cell(self.prefix),
            self.n_bins - 1,
            cell(self.gradient),
            cell(self.aggregate),
        ));
        s.push_str(&format!(
            "  ancestor-sampling acceptance (pooled over proposals): {} \
             ({}/{} proposals)\n",
            match self.as_accept {
                Some(r) => format!("{:.4}", r),
                None => "NA (no alternative ancestor was ever proposed)".to_string(),
            },
            self.n_as_accepted,
            self.n_as_proposed,
        ));
        // gh#864. Beside the acceptance rate because they answer the question
        // the rate leaves open: a low rate at a healthy post-mask ESS is the
        // suffix ratio rejecting real choices, while a low rate at an ESS near
        // 1 is a categorical that had no choice to reject.
        let ess_cell = |e: &SweepMedian| match e.median {
            Some(x) => format!("{x:.2}"),
            None => "NA".to_string(),
        };
        s.push_str(&format!(
            "  ancestor-weight ESS (median): {} before the splice guard ({} \
             sweep(s)), {} after ({} sweep(s))\n  \
             effective particles, not a fraction — read them against \
             as_finite_frac / as_admissible_frac × particles in trace.tsv\n",
            ess_cell(&self.as_ess_pre),
            self.as_ess_pre.n_sweeps,
            ess_cell(&self.as_ess_post),
            self.as_ess_post.n_sweeps,
        ));
        // gh#864. The ratio behind the accept/reject decisions, which the two
        // lines above cannot show: the ESS says how the proposal *chooses*, this
        // says why the accept test *refuses* what it chose.
        s.push_str(&format!(
            "  acceptance ratio log α (median over sweeps): {}   \
             fraction above {LOG_ALPHA_NEAR}: {}\n  \
             over {} proposal(s) with a finite ratio, of {} proposed; the rest \
             carried no ratio (zero-density suffix).\n  \
             Clustered far below 0, no proposal helps; spread with mass near 0, \
             a better-informed proposal would land the move.\n",
            match self.as_logalpha_median.median {
                Some(x) => format!("{x:.3}"),
                None => "NA".to_string(),
            },
            match self.as_logalpha_near.median {
                Some(x) => format!("{x:.4}"),
                None => "NA".to_string(),
            },
            self.n_as_logalpha,
            self.n_as_proposed,
        ));
        s
    }
}

/// The `path_renewal` key of `pgas_summary.json`. Named next to the type so a
/// producer and any reader cannot drift on the spelling.
pub const PATH_RENEWAL_KEY: &str = "path_renewal";

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::pgas::WeightCollapse;

    /// A `CSMCDiagnostics` carrying one sweep's profile and ancestor counters.
    /// Everything else is filled with values this module never reads, so a
    /// test's numbers cannot come from anywhere but the two it sets. The
    /// acceptance profile is left unmeasured (`NaN` throughout) unless a test
    /// sets it — `sweep_as_accept` below is the one that does.
    fn sweep(profile: [f64; RENEWAL_BINS], as_proposed: usize, as_accepted: usize)
        -> CSMCDiagnostics
    {
        // The aggregate the sampler would have recorded for this profile: bins
        // are equal-width in substeps, so the weighted mean over observed bins
        // IS the plain mean of the observed entries.
        let observed: Vec<f64> = profile.iter().copied().filter(|v| v.is_finite()).collect();
        let aggregate = if observed.is_empty() {
            f64::NAN
        } else {
            observed.iter().sum::<f64>() / observed.len() as f64
        };
        CSMCDiagnostics {
            trajectory_renewal: aggregate,
            renewal_by_bin: profile,
            n_degenerate: 0,
            n_resampled: 0,
            n_as_skipped_no_resample: 0,
            n_substeps: 100,
            n_as_proposed: as_proposed,
            n_as_accepted: as_accepted,
            as_accept_by_bin: [f64::NAN; RENEWAL_BINS],
            n_as_refused_inadmissible: 0,
            weight_collapse: WeightCollapse::none(100),
            as_logalpha_median: f64::NAN,
            as_logalpha_near: f64::NAN,
            n_as_logalpha: 0,
            as_finite_frac: f64::NAN,
            as_admissible_frac: f64::NAN,
            as_ess_pre: f64::NAN,
            as_ess_post: f64::NAN,
            n_as_starved: 0,
        }
    }

    fn one_chain(profiles: &[[f64; RENEWAL_BINS]]) -> PathRenewal {
        let diags: Vec<CSMCDiagnostics> =
            profiles.iter().map(|&p| sweep(p, 0, 0)).collect();
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        acc.finish().expect("at least one sweep")
    }

    /// The measured `censusnorm_q070_16c` profile from the issue.
    const COALESCED: [f64; RENEWAL_BINS] =
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.53, 0.84, 0.87, 0.98];

    /// The four fitted Ebola variants and the two-point particle-count probe of
    /// gh#791 — every run where this failure has actually been diagnosed.
    const MEASURED_FAILURES: [(&str, [f64; RENEWAL_BINS]); 6] = [
        ("carefix_q070_16c",
         [0.06, 0.06, 0.07, 0.07, 0.07, 0.07, 0.54, 0.83, 0.86, 0.98]),
        ("censusnorm_q070_16c", COALESCED),
        ("ratiosat_q070_16c",
         [0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.49, 0.76, 0.80, 0.98]),
        ("phasetype_initscale_16c",
         [0.11, 0.27, 0.28, 0.28, 0.28, 0.28, 0.43, 0.95, 0.97, 0.99]),
        ("probe at 4,800 particles",
         [0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.51, 0.77, 0.82, 0.98]),
        ("probe at 19,200 particles",
         [0.21, 0.23, 0.23, 0.23, 0.24, 0.24, 0.67, 0.93, 0.94, 0.99]),
    ];

    /// A profile that renews roughly uniformly in time, with the rise over the
    /// final bins that most runs have — the traceback's lineages have not yet
    /// coalesced when it reaches the late states. A healthy run is not a flat
    /// profile, and a rule that demanded one would fire on every run there is.
    const HEALTHY: [f64; RENEWAL_BINS] =
        [0.55, 0.57, 0.56, 0.58, 0.60, 0.62, 0.66, 0.72, 0.85, 0.95];

    /// Measured, not invented: the post-burn-in profile of a 2-chain,
    /// 40-particle, 40-substep SIR fit — an ordinary working short run, the
    /// negative anchor behind `COALESCENCE_GRADIENT`. Its gradient is 0.51, so
    /// a bar at the midpoint of the statistic's range would report it.
    const MEASURED_WORKING_SHORT_RUN: [f64; RENEWAL_BINS] =
        [0.425, 0.312, 0.338, 0.525, 0.375, 0.425, 0.550, 0.842, 0.908, 0.938];

    /// A smooth monotone ramp with no flat region, of the shape
    /// `tests/fixtures/polio_afp_es` produces (0.06 → 0.31 → 0.53 → … → 0.99).
    /// Interior bins are interpolated; the measured facts pinned against it are
    /// its endpoints, its prefix of 0.449, and that it fires. It fires for a
    /// defensible reason — renewal really is concentrated late — but a 45%
    /// prefix is not a path held at the reference, which is why the message
    /// must not say it is.
    const MEASURED_RAMP_THAT_FIRES: [f64; RENEWAL_BINS] =
        [0.06, 0.31, 0.53, 0.62, 0.725, 0.81, 0.88, 0.93, 0.97, 0.99];

    /// The short-series blind spot, from the independent check:
    /// `sirs_T60_N100` reads `first_bin = 0.000`, `prefix = 0.001` and
    /// `gradient = 0.402` — a genuinely frozen prefix that this finding does
    /// NOT report, because its last bin is low. Those four numbers plus
    /// "aggregate under 0.10" are the measured ones; the interior bins are
    /// chosen to satisfy the aggregate constraint.
    const MEASURED_SHORT_SERIES_BLIND_SPOT: [f64; RENEWAL_BINS] =
        [0.000, 0.000, 0.001, 0.001, 0.003, 0.010, 0.050, 0.120, 0.250, 0.402];

    /// The two derived numbers must be checkable against the bins by hand, or
    /// they can silently drift from the profile they claim to summarise.
    /// Worked here in full: `prefix` is the mean of b0-b4 and NOT of all ten;
    /// `gradient` is b9 − b0 and NOT b0 − b9.
    #[test]
    fn prefix_and_gradient_are_the_stated_functions_of_a_known_profile() {
        let profile: [f64; RENEWAL_BINS] =
            [0.06, 0.06, 0.07, 0.07, 0.07, 0.07, 0.54, 0.83, 0.86, 0.98];
        let pr = one_chain(&[profile]);
        assert_eq!(pr.n_bins, 10);
        assert_eq!(pr.n_prefix_bins, 5, "the prefix is the first HALF of the bins");

        // (0.06 + 0.06 + 0.07 + 0.07 + 0.07) / 5 = 0.33 / 5.
        let prefix = pr.prefix.expect("every leading bin was observed");
        assert!((prefix - 0.066).abs() < 1e-12,
            "prefix must be the mean of b0-b4 = 0.066, got {prefix}");
        // The mean over ALL ten bins is 0.361 — the number `prefix` must NOT be.
        assert!((prefix - 0.361).abs() > 0.2,
            "prefix must not be the mean over every bin");

        // 0.98 − 0.06.
        let gradient = pr.gradient.expect("both end bins were observed");
        assert!((gradient - 0.92).abs() < 1e-12,
            "gradient must be b9 − b0 = 0.92, got {gradient}");
        assert!(gradient > 0.0, "gradient is LAST minus FIRST, not the reverse");

        // And the aggregate the profile resolves rides along unchanged.
        let aggregate = pr.aggregate.expect("recorded on every sweep");
        assert!((aggregate - 0.361).abs() < 1e-12,
            "aggregate must be the sweep's own trajectory_renewal, got {aggregate}");
    }

    /// The defect, asserted from both sides on the measured profile: the
    /// aggregate rule cannot see it, and the gradient rule can.
    #[test]
    fn the_measured_coalesced_profile_is_flagged_where_the_aggregate_is_not() {
        let pr = one_chain(&[COALESCED]);
        let aggregate = pr.aggregate.expect("recorded");
        // The premise: the existing aggregate rule (LowTrajectoryRenewal fires
        // below 0.10) reads this run as healthy. Asserted, not assumed — if the
        // aggregate ever dropped below the bar the flag below would be
        // uninformative about the failure this change exists for.
        assert!(aggregate > 0.10,
            "fixture premise: the aggregate of a coalesced run reads healthy \
             ({aggregate:.3} > 0.10), which is why the profile is needed");
        assert!((aggregate - 0.34).abs() < 0.02,
            "the issue measured ~0.336 for this profile, got {aggregate:.3}");

        let prefix = pr.prefix.expect("observed");
        assert!(prefix < 0.05,
            "the first half of the series renews in under 5% of sweeps: {prefix}");
        let finding = pr.coalescence_finding()
            .expect("a profile rising 0.03 → 0.98 across the series must be reported");
        let msg = finding.render();
        // The message must carry the shape, not just a verdict.
        assert!(msg.contains("3.0%") && msg.contains("98.0%"),
            "the message must name both end bins: {msg}");
        // The aggregate reads 34.0% here — the plain mean of the ten bins,
        // since this fixture's bins are equally weighted. The issue's measured
        // run reported 0.336 for the same profile, the difference being the
        // real run's unequal substep counts per bin.
        assert!(msg.contains("34.0%"),
            "and what the aggregate read instead: {msg}");
        assert_eq!(finding.severity(), sim::inference::diagnostic::Severity::Warning,
            "this is a diagnostic and must never be able to fail a run");
        assert!(finding.hints().iter().any(|h| h.contains("particle")),
            "the message must say what to do; the measured remedy is particles: \
             {:?}", finding.hints());
    }

    /// The negative direction, and the one that matters most: a run renewing
    /// roughly uniformly in time — including the structural rise over the final
    /// bins that every run has — must NOT be reported.
    #[test]
    fn a_healthy_profile_is_not_flagged() {
        let pr = one_chain(&[HEALTHY]);
        let gradient = pr.gradient.expect("observed");
        assert!((gradient - 0.40).abs() < 1e-12, "0.95 − 0.55, got {gradient}");
        assert!(pr.coalescence_finding().is_none(),
            "a profile renewing 0.55 → 0.95 across the series is a working \
             sampler with the usual end-of-series rise, and must draw no \
             finding (gradient {gradient})");
        let prefix = pr.prefix.expect("observed");
        assert!(prefix > 0.5, "sanity: the early bins renew, {prefix}");
    }

    /// The negative direction on a REAL profile rather than a written-down one:
    /// an ordinary working short fit must stay silent. This is the run that
    /// ruled out a bar at the midpoint of the statistic's range — it reads 0.51
    /// — so the assertion is what keeps the bar from drifting back down onto it.
    #[test]
    fn a_measured_working_short_run_is_not_flagged() {
        let pr = one_chain(&[MEASURED_WORKING_SHORT_RUN]);
        let gradient = pr.gradient.expect("observed");
        assert!((gradient - 0.513).abs() < 0.01,
            "fixture premise: this working run reads ~0.51, got {gradient}");
        assert!(pr.coalescence_finding().is_none(),
            "a 2-chain 40-particle 40-substep SIR fit is a working sampler whose \
             early bins renew in ~40% of sweeps; reporting it would bury the runs \
             that renew in 3% (gradient {gradient})");
    }

    /// The finding fires on two different shapes and can distinguish neither,
    /// because the gradient reads only the two end bins. So the message must
    /// describe the shape and hand over the discriminator — it must not assert
    /// a mechanism.
    ///
    /// The case that makes this load-bearing: the repository's own polio
    /// fixture fires with a prefix of 0.449 on a smooth monotone ramp. A
    /// message claiming "the early path is held at the reference" would
    /// contradict the 44.9% printed in the same sentence.
    #[test]
    fn the_message_describes_the_shape_without_asserting_a_cause() {
        for (label, profile) in
            [("coalesced", COALESCED), ("monotone ramp", MEASURED_RAMP_THAT_FIRES)]
        {
            let pr = one_chain(&[profile]);
            let msg = pr.coalescence_finding()
                .unwrap_or_else(|| panic!("{label}: fixture premise — this profile fires"))
                .render();
            assert!(msg.contains("does not say which shape produced"),
                "{label}: the message must say what the gradient cannot resolve: {msg}");
            assert!(msg.contains("monotone ramp"),
                "{label}: and offer the other reading, not just the alarming one: {msg}");
            assert!(!msg.contains("are not being informed"),
                "{label}: the message must not assert a consequence it cannot \
                 establish from two end bins: {msg}");
        }
        // And on the ramp specifically, the prefix the old wording contradicted
        // is printed, so a reader can see the claim would have been false.
        let pr = one_chain(&[MEASURED_RAMP_THAT_FIRES]);
        let prefix = pr.prefix.expect("observed");
        assert!((prefix - 0.449).abs() < 1e-12,
            "fixture premise: the polio fixture's prefix is 0.449, got {prefix}");
        assert!(pr.coalescence_finding().unwrap().render().contains("44.9%"),
            "the prefix must be in the message, beside the shape description");
    }

    /// The blind spot, stated rather than discovered. `gradient ≈ 1 − b0` is a
    /// property of the family the bar was anchored on, not of the statistic:
    /// across the independent check's 27 runs the last bin spans 0.402 to
    /// 0.998. Where it is low the gradient is bounded below the bar no matter
    /// how frozen the prefix is — and `sirs_T60_N100`, at `b0 = 0.000` and
    /// `prefix = 0.001`, is exactly the failure gh#791 exists to catch.
    ///
    /// This finding stays quiet on it. The aggregate rule catches it. Both are
    /// asserted here so the gap is a documented property with a test on it
    /// rather than something a user runs into.
    #[test]
    fn the_short_series_blind_spot_is_covered_only_by_the_aggregate_rule() {
        let pr = one_chain(&[MEASURED_SHORT_SERIES_BLIND_SPOT]);
        assert_eq!(pr.bins[0], Some(0.0), "the first tenth never renews at all");
        let prefix = pr.prefix.expect("observed");
        assert!((prefix - 0.001).abs() < 1e-9,
            "fixture premise: the measured prefix is 0.001, got {prefix}");
        let gradient = pr.gradient.expect("observed");
        assert!((gradient - 0.402).abs() < 1e-9,
            "fixture premise: the measured gradient is 0.402, got {gradient}");

        assert!(pr.coalescence_finding().is_none(),
            "the gradient rule does NOT report a completely frozen prefix when the \
             last bin is low — this is the documented blind spot, not a bug to \
             fix by lowering the bar");
        // The pre-existing aggregate rule is what covers it. `LowTrajectoryRenewal`
        // fires below 0.10 (cli/src/fit/pgas.rs), so this run is reported — by the
        // other rule. Neither rule alone covers both cases.
        let aggregate = pr.aggregate.expect("recorded");
        assert!(aggregate < 0.10,
            "and it is caught only because the aggregate is {aggregate:.4} < 0.10, \
             which is the threshold LowTrajectoryRenewal applies");
    }

    /// The independent check ran the bar against 27 PGAS runs on four further
    /// model families plus this repository's polio fixture. Converged runs
    /// (R̂ < 1.05, ESS 214-464) read 0.018-0.172; every firing run had R̂ ≥ 1.88
    /// and read ≥ 0.758; nothing in that cohort landed in between. The bar must
    /// sit inside the gap.
    ///
    /// The gap is emptier than the evidence overall, though, and this test is
    /// deliberately not the only thing holding the value: the separately
    /// measured working short run reads 0.51, which is INSIDE `[0.172, 0.758]`.
    /// So the cohort's gap alone would permit a bar anywhere above 0.172, and it
    /// is `a_measured_working_short_run_is_not_flagged` that rules out the
    /// bottom of that range. The two anchors are complementary, and 0.75 sits
    /// near the top of the gap for that reason rather than in its middle.
    #[test]
    fn the_bar_lies_inside_the_validation_cohorts_empty_gap() {
        const HIGHEST_CONVERGED: f64 = 0.172;
        const LOWEST_FIRING: f64 = 0.758;
        assert!(COALESCENCE_GRADIENT > HIGHEST_CONVERGED,
            "the bar must clear every converged run in the 27-run cohort \
             (highest {HIGHEST_CONVERGED})");
        assert!(COALESCENCE_GRADIENT < LOWEST_FIRING,
            "and sit below every run that fired (lowest {LOWEST_FIRING})");
        // The cohort gap does not pin the lower end on its own — a working run
        // measured outside that cohort sits in it. Named here so a reader does
        // not mistake this test for the whole justification.
        let working = one_chain(&[MEASURED_WORKING_SHORT_RUN])
            .gradient.expect("observed");
        assert!(working > HIGHEST_CONVERGED && working < LOWEST_FIRING,
            "premise: the measured working run at {working:.3} falls inside the \
             cohort's gap, so the gap alone cannot justify the bar");
    }

    /// The particle-count advice is right about the underlying problem and
    /// misleading about the instrument: raising N moves the gradient the WRONG
    /// way (0.81 → 0.86 → 0.90 measured at N = 100, 400, 1600 with everything
    /// else held fixed) while the aggregate improves. A user who follows the
    /// hint and re-reads the gradient concludes it got worse, so the hints must
    /// say which number to re-read.
    #[test]
    fn the_hints_say_which_number_to_re_read_after_changing_particles() {
        let finding = one_chain(&[COALESCED]).coalescence_finding().expect("fires");
        let hints = finding.hints();
        let re_read = hints.iter()
            .find(|h| h.contains("re-read"))
            .unwrap_or_else(|| panic!(
                "a hint must tell the user what to re-read after changing the \
                 particle count: {hints:?}"));
        assert!(re_read.contains("not the gradient"),
            "and must say the gradient is NOT it: {re_read}");
        assert!(re_read.contains("0.81") && re_read.contains("0.90"),
            "with the measurement that shows why: {re_read}");
    }

    /// The bar is an empirical anchor, and this is the evidence it is anchored
    /// to — asserted, so a later change to `COALESCENCE_GRADIENT` has to move
    /// the measurements with it rather than silently redefining the finding.
    /// Every run where this failure was diagnosed is above the bar; the working
    /// run measured beside them is below it.
    #[test]
    fn the_bar_sits_between_the_measured_working_run_and_every_measured_failure() {
        let gradient_of = |p: [f64; RENEWAL_BINS]| {
            one_chain(&[p]).gradient.expect("both end bins observed")
        };
        let working = gradient_of(MEASURED_WORKING_SHORT_RUN);
        assert!(working < COALESCENCE_GRADIENT,
            "the measured working run reads {working:.3} and must be BELOW the \
             bar {COALESCENCE_GRADIENT}");
        for (name, profile) in MEASURED_FAILURES {
            let g = gradient_of(profile);
            assert!(g >= COALESCENCE_GRADIENT,
                "{name} is a diagnosed coalesced run reading {g:.3}, which must \
                 be at or above the bar {COALESCENCE_GRADIENT}");
            assert!(one_chain(&[profile]).coalescence_finding().is_some(),
                "{name} must be reported");
        }
        // And the margin is not a rounding artefact on either side.
        let worst_failure = MEASURED_FAILURES.iter()
            .map(|&(_, p)| gradient_of(p))
            .fold(f64::INFINITY, f64::min);
        assert!(worst_failure - working > 0.2,
            "the anchor rests on a real separation between the two populations: \
             the mildest diagnosed failure reads {worst_failure:.3}, the working \
             run {working:.3}");
    }

    /// One sweep carrying only the two gh#864 ancestor-weight ESS values. The
    /// profile is the healthy one so nothing else in the block can be read as
    /// coming from these numbers.
    fn sweep_ess(pre: f64, post: f64) -> CSMCDiagnostics {
        let mut d = sweep(HEALTHY, 0, 0);
        d.as_ess_pre = pre;
        d.as_ess_post = post;
        d
    }

    /// gh#864: the pooled number is a median over the sweeps that measured an
    /// ESS, and a sweep that measured none is absent from it rather than
    /// entered as zero.
    ///
    /// The two are distinguishable by construction here: over `[1, 3, 9]` the
    /// median is 3, while folding the unmeasured sweep in as a zero would give
    /// `[0, 1, 3, 9]` and a median of 2. A sweep with no ancestor-sampling step
    /// says nothing about how concentrated the ancestor weights are, and
    /// averaging it in as a collapse invents a reading.
    #[test]
    fn the_ancestor_ess_median_is_over_the_sweeps_that_measured_one() {
        let diags: Vec<CSMCDiagnostics> = vec![
            sweep_ess(1.0, 2.0),
            sweep_ess(3.0, 6.0),
            sweep_ess(f64::NAN, f64::NAN),
            sweep_ess(9.0, 18.0),
        ];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();

        assert_eq!(pr.n_sweeps, 4, "every sweep still counts toward the profile");
        assert_eq!(pr.as_ess_pre.n_sweeps, 3,
            "but the ESS median is over the three sweeps that measured one");
        assert_eq!(pr.as_ess_pre.median, Some(3.0),
            "median of [1, 3, 9] is 3 — a zero folded in for the unmeasured \
             sweep would read 2");
        assert_eq!(pr.as_ess_post.median, Some(6.0), "and the same after the mask");
        assert_eq!(pr.as_ess_post.n_sweeps, 3);
    }

    /// No sweep measured one — ancestor sampling off for the whole stage — and
    /// the median is `null`, not 0.0. The block still publishes, because the
    /// profile and the acceptance rate are still measurements.
    #[test]
    fn a_stage_that_measured_no_ancestor_ess_reports_null_not_zero() {
        let diags: Vec<CSMCDiagnostics> = vec![sweep_ess(f64::NAN, f64::NAN)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();
        assert_eq!(pr.as_ess_pre.median, None);
        assert_eq!(pr.as_ess_post.median, None);
        assert_eq!(pr.as_ess_pre.n_sweeps, 0, "and it says so in its denominator");
        let json = serde_json::to_value(&pr).expect("serializes");
        assert!(json["as_ess_pre"]["median"].is_null(),
            "null in the JSON too — a consumer must not read a collapsed \
             categorical off a stage that measured none: {json}");
    }

    /// The guard can lower the candidate count and raise the ESS, so both sides
    /// of the mask must survive into the artifact and the printed block. A
    /// reader holding only the post-mask number cannot tell the gh#607 screen
    /// concentrating the draw from the density doing it.
    #[test]
    fn both_sides_of_the_mask_reach_the_report_and_the_json() {
        // The gh#864 eight-particle case, at the sweep level: fewer candidates
        // after the guard, a higher ESS among them.
        let diags: Vec<CSMCDiagnostics> = vec![sweep_ess(1.23, 3.85)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();
        let text = pr.report();
        assert!(text.contains("ancestor-weight ESS"),
            "the block must name the statistic:\n{text}");
        assert!(text.contains("1.23") && text.contains("3.85"),
            "both sides are printed, not the post-mask one alone:\n{text}");
        assert!(text.contains("particles"),
            "and it must say the units are particles, since its two \
             `trace.tsv` neighbours are fractions:\n{text}");
        // Adjacent to the acceptance rate: a low rate at a healthy post-mask
        // ESS is the suffix ratio rejecting, a low rate at an ESS near 1 is a
        // categorical with nothing to reject.
        let i_accept = text.find("ancestor-sampling acceptance").expect("acceptance line");
        let i_ess = text.find("ancestor-weight ESS").expect("ess line");
        assert!(i_ess > i_accept && text[i_accept..i_ess].lines().count() <= 2,
            "the ESS must sit beside the acceptance rate it qualifies:\n{text}");

        let json = serde_json::to_value(&pr).expect("serializes");
        assert_eq!(json["as_ess_pre"]["median"].as_f64(), Some(1.23));
        assert_eq!(json["as_ess_post"]["median"].as_f64(), Some(3.85));
        assert_eq!(json["as_ess_post"]["n_sweeps"].as_u64(), Some(1),
            "with the denominator beside it: {json}");
    }

    /// A run that renews poorly but UNIFORMLY in time is a different failure —
    /// the level is wrong, the shape is not — and must not be reported as
    /// coalescence. `LowTrajectoryRenewal` is the finding that fires on the
    /// level, and it still does; keying this one on the shape is what keeps the
    /// two separable.
    #[test]
    fn a_uniformly_low_profile_is_not_reported_as_coalescence() {
        let flat_low: [f64; RENEWAL_BINS] =
            [0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03];
        let pr = one_chain(&[flat_low]);
        assert!(pr.aggregate.expect("recorded") < 0.10,
            "this run IS in trouble, and the aggregate rule reports it");
        assert!(pr.coalescence_finding().is_none(),
            "but not as coalescence: nothing about this profile says the EARLY \
             path specifically is held at the reference");
    }

    /// The profile is the mean down each column, over sweeps and over chains
    /// alike — not the last chain's, not the first sweep's.
    #[test]
    fn the_profile_averages_down_each_column_over_sweeps_and_chains() {
        let a: [f64; RENEWAL_BINS] = [0.10; RENEWAL_BINS];
        let b: [f64; RENEWAL_BINS] = [0.30; RENEWAL_BINS];
        let c: [f64; RENEWAL_BINS] = [0.50; RENEWAL_BINS];
        let d: [f64; RENEWAL_BINS] = [0.70; RENEWAL_BINS];
        let chain0: Vec<CSMCDiagnostics> = [a, b].iter().map(|&p| sweep(p, 10, 1)).collect();
        let chain1: Vec<CSMCDiagnostics> = [c, d].iter().map(|&p| sweep(p, 30, 2)).collect();
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(chain0.iter());
        acc.add_chain(chain1.iter());
        let pr = acc.finish().expect("four sweeps");

        assert_eq!(pr.n_sweeps, 4);
        assert_eq!(pr.n_chains, 2);
        for (i, v) in pr.bins.iter().enumerate() {
            let v = v.expect("every bin observed");
            assert!((v - 0.40).abs() < 1e-12,
                "bin {i} must be (0.1+0.3+0.5+0.7)/4 = 0.40, got {v}");
        }
        assert!((pr.prefix.unwrap() - 0.40).abs() < 1e-12);
        assert!((pr.gradient.unwrap()).abs() < 1e-12, "a flat profile has no gradient");

        // Ancestor sampling pools numerator and denominator across everything,
        // rather than averaging per-sweep rates — a sweep proposing 30 moves
        // and one proposing 10 do not carry equal weight.
        assert_eq!((pr.n_as_proposed, pr.n_as_accepted), (80, 6));
        assert!((pr.as_accept.unwrap() - 6.0 / 80.0).abs() < 1e-12,
            "as_accept is the pooled ratio, got {:?}", pr.as_accept);
    }

    /// A bin no sweep recorded a substep in is `null`, never `0.0`, and it is
    /// skipped by the means rather than dragging them down. A short series —
    /// fewer substeps than bins — is the ordinary case that produces this.
    #[test]
    fn a_bin_holding_no_substep_is_null_and_is_skipped_by_the_summaries() {
        let mut profile = [f64::NAN; RENEWAL_BINS];
        profile[0] = 0.40;
        profile[2] = 0.60;
        profile[9] = 0.80;
        let pr = one_chain(&[profile]);
        assert_eq!(pr.bins[1], None, "an empty bin is null, not 0.0");
        assert_eq!(pr.bins[0], Some(0.40));
        // Mean of the OBSERVED leading bins, 0.40 and 0.60 — not 1.0/5.
        assert!((pr.prefix.unwrap() - 0.50).abs() < 1e-12,
            "prefix must skip the empty bins, got {:?}", pr.prefix);
        assert!((pr.gradient.unwrap() - 0.40).abs() < 1e-12, "0.80 − 0.40");
    }

    /// A gradient needs both ends. When either end bin holds no substep the
    /// gradient is `null` and nothing is reported — a finding built on a
    /// guessed end is a claim the data does not support.
    #[test]
    fn a_missing_end_bin_leaves_the_gradient_undefined_and_reports_nothing() {
        let mut profile = [0.05; RENEWAL_BINS];
        profile[RENEWAL_BINS - 1] = f64::NAN;
        let pr = one_chain(&[profile]);
        assert_eq!(pr.gradient, None);
        assert!(pr.coalescence_finding().is_none(),
            "no last bin, no gradient, no finding");

        let mut profile = [0.05; RENEWAL_BINS];
        profile[RENEWAL_BINS - 1] = 0.99;
        profile[0] = f64::NAN;
        let pr = one_chain(&[profile]);
        assert_eq!(pr.gradient, None, "and the same at the other end");
        assert!(pr.coalescence_finding().is_none());
    }

    /// No sweep, no block — the summary omits the key entirely rather than
    /// carrying a row of nulls a consumer would read as a measurement.
    #[test]
    fn a_stage_with_no_retained_sweep_produces_no_block() {
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(std::iter::empty());
        assert!(acc.clone().finish().is_none());
        assert_eq!(acc.finish().map(|p| p.n_chains), None,
            "and a chain that contributed nothing is not counted as a chain");
    }

    /// The threshold fires at its stated value and not a hair either side, so a
    /// silent drift in `COALESCENCE_GRADIENT` cannot pass as the same rule.
    #[test]
    fn the_reported_gradient_bar_is_the_bar_that_fires() {
        // b0 = 0, so b9 IS the gradient.
        let at = |last: f64| {
            let mut p = [0.0; RENEWAL_BINS];
            p[RENEWAL_BINS - 1] = last;
            one_chain(&[p]).coalescence_finding().is_some()
        };
        assert!(at(COALESCENCE_GRADIENT), "the bar itself is reported");
        assert!(at(COALESCENCE_GRADIENT + 1e-6), "and above it");
        assert!(!at(COALESCENCE_GRADIENT - 1e-6), "and below it is silent");
    }

    /// The printed block must carry the profile, both derived numbers, the
    /// aggregate, and the ancestor-sampling rate — the last ADJACENT to the
    /// profile, which is the whole point of reporting it here (gh#791).
    #[test]
    fn the_report_prints_the_profile_beside_the_aggregate_and_as_accept() {
        let diags: Vec<CSMCDiagnostics> = vec![sweep(COALESCED, 1000, 16)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let text = acc.finish().unwrap().report();
        for label in ["b0", "b9", "prefix", "gradient", "aggregate",
                      "ancestor-sampling acceptance"] {
            assert!(text.contains(label), "report must carry `{label}`:\n{text}");
        }
        assert!(text.contains("0.030") && text.contains("0.980"),
            "the whole profile is printed, not a summary of it:\n{text}");
        assert!(text.contains("tenth"),
            "and it must say what a bin spans — `b0 = 0.03` is uninterpretable \
             without it:\n{text}");
        // Adjacency, asserted rather than eyeballed: the acceptance rate is
        // useless unless it is read against the profile above it. "Same block"
        // is stated two ways — an unbroken run of lines, and a bound on how
        // many — because a line count alone would pass with a blank line
        // between them, which is what separates one block from the next.
        let i_bins = text.find("renewal  ").expect("profile row");
        let i_as = text.find("ancestor-sampling").expect("acceptance line");
        assert!(i_as > i_bins, "the acceptance rate follows the profile:\n{text}");
        // `trim_end` drops the indentation of the acceptance line itself, which
        // is the partial last "line" of the slice and not a blank one.
        let between = text[i_bins..i_as].trim_end();
        assert!(between.lines().all(|l| !l.trim().is_empty()),
            "nothing may break the block between the profile and the acceptance \
             rate:\n{text}");
        assert!(between.lines().count() <= 6,
            "as_accept must sit within the same block as the profile:\n{text}");
    }

    /// A `CSMCDiagnostics` carrying an acceptance profile, on the healthy
    /// renewal profile so nothing else in the block can be read as coming from
    /// it.
    fn sweep_as_accept(as_accept: [f64; RENEWAL_BINS]) -> CSMCDiagnostics {
        let mut d = sweep(HEALTHY, 0, 0);
        d.as_accept_by_bin = as_accept;
        d
    }

    /// gh#864: the acceptance profile is the mean down each column over the
    /// sweeps that proposed a move in that bin, and a sweep that proposed none
    /// there is absent from the mean rather than entered as a zero.
    ///
    /// The two are distinguishable by construction: bin 0 is measured on one
    /// sweep at 0.40 and unmeasured on the other, so the column mean is 0.40
    /// while folding the unmeasured sweep in as a zero would read 0.20.
    #[test]
    fn the_acceptance_profile_averages_only_the_sweeps_that_proposed_a_move() {
        let mut first = [0.10; RENEWAL_BINS];
        first[0] = 0.40;
        let mut second = [0.30; RENEWAL_BINS];
        second[0] = f64::NAN;
        let diags: Vec<CSMCDiagnostics> =
            vec![sweep_as_accept(first), sweep_as_accept(second)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();

        assert_eq!(pr.as_accept_bins.len(), RENEWAL_BINS, "one entry per bin");
        assert_eq!(pr.as_accept_bins[0], Some(0.40),
            "bin 0 was proposed in on one sweep only, so its mean is that \
             sweep's 0.40 — a zero folded in for the other would read 0.20");
        for b in 1..RENEWAL_BINS {
            let v = pr.as_accept_bins[b].expect("both sweeps proposed here");
            assert!((v - 0.20).abs() < 1e-12,
                "bin {b} must be the mean of 0.10 and 0.30, got {v}");
        }
        assert_eq!(pr.n_sweeps, 2, "every sweep still counts toward the profile");
    }

    /// A bin no retained sweep ever proposed a move in is `null`, never `0.0` —
    /// the same rule the renewal row uses, and the one that keeps "the move was
    /// never offered here" separable from "it was offered and always refused".
    /// Those are the two readings the whole profile exists to tell apart.
    #[test]
    fn a_bin_with_no_proposal_is_null_in_the_acceptance_profile_and_the_json() {
        let mut profile = [f64::NAN; RENEWAL_BINS];
        profile[0] = 0.0;
        profile[9] = 0.5;
        let diags: Vec<CSMCDiagnostics> = vec![sweep_as_accept(profile)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();

        assert_eq!(pr.as_accept_bins[0], Some(0.0),
            "b0 was proposed in and never accepted — a measured zero, which is \
             not the same reading as an unproposed bin");
        assert_eq!(pr.as_accept_bins[4], None, "and an unproposed bin is null");

        let json = serde_json::to_value(&pr).expect("serializes");
        let bins = json["as_accept_bins"].as_array().expect("as_accept_bins array");
        assert_eq!(bins.len(), RENEWAL_BINS);
        assert!(bins[4].is_null(),
            "null in the JSON too — a consumer must not read a refused move off \
             a bin the move was never offered in: {json}");
        assert_eq!(bins[0].as_f64(), Some(0.0),
            "while a genuine zero survives as a zero: {json}");
    }

    /// One sweep carrying only the gh#864 acceptance-ratio summaries, on the
    /// healthy renewal profile so nothing else in the block can be read as
    /// coming from them.
    fn sweep_logalpha(median: f64, near: f64, n: usize) -> CSMCDiagnostics {
        let mut d = sweep(HEALTHY, 0, 0);
        d.as_logalpha_median = median;
        d.as_logalpha_near = near;
        d.n_as_logalpha = n;
        d
    }

    /// gh#864: the published ratio summaries are medians over the sweeps that
    /// measured one, and a sweep that measured none is absent rather than
    /// entered as a zero.
    ///
    /// Zero is parity — the best reading a `log α` can carry — so folding an
    /// unmeasured sweep in as one does not merely dilute the number, it reports
    /// the opposite of what happened. Distinguishable by construction: over
    /// `[−30, −6, −2]` the median is −6, while an invented zero would give
    /// `[−30, −6, −2, 0]` and −4.
    #[test]
    fn the_ratio_summaries_are_over_the_sweeps_that_measured_one() {
        let diags: Vec<CSMCDiagnostics> = vec![
            sweep_logalpha(-30.0, 0.10, 20),
            sweep_logalpha(-6.0, 0.20, 15),
            sweep_logalpha(f64::NAN, f64::NAN, 0),
            sweep_logalpha(-2.0, 0.60, 5),
        ];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();

        assert_eq!(pr.n_sweeps, 4, "every sweep still counts toward the profile");
        assert_eq!(pr.as_logalpha_median.n_sweeps, 3,
            "but the ratio median is over the three sweeps that measured one");
        assert_eq!(pr.as_logalpha_median.median, Some(-6.0),
            "median of [−30, −6, −2] is −6 — a zero folded in for the \
             unmeasured sweep would read −4, and a zero is *parity*");
        assert_eq!(pr.as_logalpha_near.median, Some(0.20));
        assert_eq!(pr.n_as_logalpha, 40,
            "and the sample size is pooled over every sweep: 20 + 15 + 0 + 5");
    }

    /// The sample size is published because the fraction near parity is
    /// unreadable without it — `null` for the median, and a pooled count that
    /// is still a measurement, are different facts and both survive into the
    /// JSON.
    #[test]
    fn a_stage_that_measured_no_ratio_reports_null_with_its_denominator() {
        let diags: Vec<CSMCDiagnostics> = vec![sweep_logalpha(f64::NAN, f64::NAN, 0)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let pr = acc.finish().unwrap();
        assert_eq!(pr.as_logalpha_median.median, None);
        assert_eq!(pr.as_logalpha_near.median, None);
        assert_eq!(pr.n_as_logalpha, 0);

        let json = serde_json::to_value(&pr).expect("serializes");
        assert!(json["as_logalpha_median"]["median"].is_null(),
            "null in the JSON — a consumer must not read parity off a stage \
             that measured no ratio at all: {json}");
        assert_eq!(json["n_as_logalpha"].as_u64(), Some(0),
            "with the sample size beside it: {json}");
    }

    /// The printed block carries the ratio beside the outcome counters it
    /// explains, with the sample it is over. A median with no denominator is a
    /// number a reader cannot act on.
    #[test]
    fn the_report_prints_the_ratio_with_the_sample_it_is_over() {
        let mut d = sweep(COALESCED, 1000, 16);
        d.as_logalpha_median = -12.375;
        d.as_logalpha_near = 0.0314;
        d.n_as_logalpha = 640;
        let mut acc = PathRenewalAccum::new();
        acc.add_chain([d].iter());
        let text = acc.finish().unwrap().report();

        assert!(text.contains("acceptance ratio log"),
            "the block must name the statistic:\n{text}");
        assert!(text.contains("-12.375"), "with the median:\n{text}");
        assert!(text.contains("0.0314"), "and the fraction near parity:\n{text}");
        assert!(text.contains("640") && text.contains("1000"),
            "and the sample it is over against the proposals made:\n{text}");
        // The word "mean" is legitimate elsewhere in the block — the profile
        // is one, over sweeps. What must never appear is a mean *of the ratio*.
        let ratio_line = text.lines().find(|l| l.contains("acceptance ratio log"))
            .expect("ratio line");
        assert!(!ratio_line.contains("mean"),
            "the ratio is summarised by a median and never a mean — on a log \
             ratio's left tail the mean reports the opposite of the \
             distribution: {ratio_line}");
        assert!(ratio_line.contains("median"),
            "and it must say which summary it is: {ratio_line}");
    }

    /// The two rows are printed one above the other, on the same bins. A
    /// falling acceptance profile is only legible against the renewal profile
    /// it explains, and a reader who has to scroll between them will not
    /// compare them.
    #[test]
    fn the_acceptance_profile_prints_directly_under_the_renewal_profile() {
        // Falling toward b0: the shape the mechanistic claim predicts — a
        // splice is harder the further back it happens.
        let falling: [f64; RENEWAL_BINS] =
            [0.029, 0.05, 0.09, 0.14, 0.21, 0.30, 0.42, 0.57, 0.71, 0.83];
        let diags: Vec<CSMCDiagnostics> = vec![sweep_as_accept(falling)];
        let mut acc = PathRenewalAccum::new();
        acc.add_chain(diags.iter());
        let text = acc.finish().unwrap().report();

        assert!(text.contains("as accept"), "the row must be labelled:\n{text}");
        assert!(text.contains("0.029") && text.contains("0.830"),
            "both ends of the acceptance profile are printed:\n{text}");
        let lines: Vec<&str> = text.lines().collect();
        let i_renewal = lines.iter().position(|l| l.trim_start().starts_with("renewal"))
            .expect("renewal row");
        let i_accept = lines.iter().position(|l| l.trim_start().starts_with("as accept"))
            .expect("acceptance row");
        assert_eq!(i_accept, i_renewal + 1,
            "the acceptance row must be the line directly after the renewal \
             row, on the same bins:\n{text}");
        // And the bin header they are both read against is directly above.
        assert_eq!(i_renewal, lines.iter().position(|l| l.trim_start().starts_with("bin "))
            .expect("bin header") + 1, "with the bin labels above both:\n{text}");
    }

    /// The artifact states its own bin span. A reader who has only the JSON —
    /// camdl-scope, a notebook — cannot interpret `bins[0] = 0.04` without it,
    /// and today that fact lives only in a source comment.
    #[test]
    fn the_artifact_states_what_a_bin_spans() {
        let pr = one_chain(&[HEALTHY]);
        assert_eq!(pr.bin_span, BIN_SPAN);
        assert!(pr.bin_span.contains("[b/10, (b+1)/10)"),
            "the span must be stated as a fraction of the series: {}", pr.bin_span);
        assert!(pr.bin_span.contains("fixed"),
            "and that it is fixed rather than proportional, which is what makes \
             profiles comparable across models: {}", pr.bin_span);
        let json = serde_json::to_value(&pr).expect("serializes");
        assert!(json["bin_span"].as_str().is_some_and(|s| s.contains("tenth")),
            "and it must survive into the JSON: {json}");
    }

    /// The JSON shape downstream reads: ten entries, `null` for an unobserved
    /// bin, and the derived numbers recomputable from `bins` by a consumer.
    #[test]
    fn the_json_block_carries_ten_entries_with_null_for_an_unobserved_bin() {
        let mut profile = COALESCED;
        profile[4] = f64::NAN;
        let pr = one_chain(&[profile]);
        let json = serde_json::to_value(&pr).expect("serializes");
        let bins = json["bins"].as_array().expect("bins is an array");
        assert_eq!(bins.len(), RENEWAL_BINS);
        assert!(bins[4].is_null(), "an unobserved bin is JSON null: {}", bins[4]);
        assert_eq!(bins[0].as_f64(), Some(0.03));
        assert_eq!(json["n_bins"], serde_json::json!(10));
        assert_eq!(json["n_prefix_bins"], serde_json::json!(5));

        // A consumer must be able to recompute both derived numbers from the
        // published profile. If it cannot, the block is not self-describing.
        let observed: Vec<f64> = bins[..PREFIX_BINS].iter().filter_map(|v| v.as_f64()).collect();
        let prefix = observed.iter().sum::<f64>() / observed.len() as f64;
        assert!((json["prefix"].as_f64().unwrap() - prefix).abs() < 1e-12);
        let gradient = bins[RENEWAL_BINS - 1].as_f64().unwrap() - bins[0].as_f64().unwrap();
        assert!((json["gradient"].as_f64().unwrap() - gradient).abs() < 1e-12);
    }
}
