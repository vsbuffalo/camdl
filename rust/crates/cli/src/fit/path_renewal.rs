//! gh#791: the `path_renewal` block of `pgas_summary.json` — trajectory
//! renewal resolved in time, its two derived numbers, and the
//! ancestor-sampling acceptance rate read beside them.
//!
//! ## Why the aggregate cannot answer this
//!
//! [`CSMCDiagnostics::trajectory_renewal`] is a weighted mean over the bins of
//! [`CSMCDiagnostics::renewal_by_bin`]. Its last term is structurally near 1:
//! the segment of the path after the final observation is resampled freely
//! every sweep, whatever the sampler's health, so that one bin holds the mean
//! up on its own. A conditional-SMC genealogy that has coalesced — every
//! lineage traced back to the same ancestor by the time the traceback reaches
//! the early states, so the early path is held at the reference — therefore
//! scores an aggregate that reads healthy.
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
use sim::inference::pgas::{CSMCDiagnostics, RENEWAL_BINS};

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
/// It is not, because the last bin is high in EVERY run: the segment of the
/// path after the final observation is resampled freely whatever the sampler's
/// health, so `gradient ≈ 1 − first_bin` and a bar at the midpoint is an
/// absolute bar on the first bin wearing a disguise. Measured, rather than
/// argued: a 2-chain, 40-particle, 40-substep SIR fit — an ordinary working
/// short run — reads
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
    /// Substeps at which the Metropolis step actually ran — the denominator of
    /// [`Self::as_accept`].
    pub n_as_proposed: u64,
    /// Of those, the accepted ones.
    pub n_as_accepted: u64,
    /// Retained post-burn-in sweeps the profile was averaged over, summed
    /// across chains.
    pub n_sweeps: usize,
    /// Surviving chains that contributed at least one sweep.
    pub n_chains: usize,
}

/// Accumulator for [`PathRenewal`] over the retained sweeps of every chain.
///
/// A type rather than loose sums at the call site, for the same reason
/// `RenewalBins` is one on the sim side: the "skip a bin no substep fell in"
/// rule is then written once, and the profile cannot be updated out of step
/// with the aggregate it resolves.
#[derive(Debug, Clone, Default)]
pub struct PathRenewalAccum {
    bin_sum: [f64; RENEWAL_BINS],
    bin_n: [usize; RENEWAL_BINS],
    aggregate_sum: f64,
    aggregate_n: usize,
    n_sweeps: usize,
    n_chains: usize,
    n_as_proposed: u64,
    n_as_accepted: u64,
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
        if d.trajectory_renewal.is_finite() {
            self.aggregate_sum += d.trajectory_renewal;
            self.aggregate_n += 1;
        }
        self.n_sweeps += 1;
        self.n_as_proposed += d.n_as_proposed as u64;
        self.n_as_accepted += d.n_as_accepted as u64;
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
            n_as_proposed: self.n_as_proposed,
            n_as_accepted: self.n_as_accepted,
            n_sweeps: self.n_sweeps,
            n_chains: self.n_chains,
        })
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
        let mut s = format!(
            "\npath renewal (gh#791; mean over {} retained post-burn-in sweep(s) \
             across {} chain(s)):\n  bin      {}\n  renewal  {}\n",
            self.n_sweeps,
            self.n_chains,
            labels.join(" "),
            values.join(" "),
        );
        s.push_str(&format!(
            "  each bin is a fixed tenth of the substep series: b0 is its first \
             tenth, b{} its last\n",
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
            "  ancestor-sampling acceptance: {} ({}/{} proposals)\n",
            match self.as_accept {
                Some(r) => format!("{:.4}", r),
                None => "NA (no alternative ancestor was ever proposed)".to_string(),
            },
            self.n_as_accepted,
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
    /// test's numbers cannot come from anywhere but the two it sets.
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
            n_as_refused_inadmissible: 0,
            weight_collapse: WeightCollapse::none(100),
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

    /// A profile that renews roughly uniformly in time, with the structural
    /// rise over the final bins that EVERY run has — the segment after the last
    /// observation is resampled freely whatever the sampler's health. A healthy
    /// run is not a flat profile, and a rule that demanded one would fire on
    /// every run there is.
    const HEALTHY: [f64; RENEWAL_BINS] =
        [0.55, 0.57, 0.56, 0.58, 0.60, 0.62, 0.66, 0.72, 0.85, 0.95];

    /// Measured, not invented: the post-burn-in profile of a 2-chain,
    /// 40-particle, 40-substep SIR fit — an ordinary working short run, the
    /// negative anchor behind `COALESCENCE_GRADIENT`. Its gradient is 0.51, so
    /// a bar at the midpoint of the statistic's range would report it.
    const MEASURED_WORKING_SHORT_RUN: [f64; RENEWAL_BINS] =
        [0.425, 0.312, 0.338, 0.525, 0.375, 0.425, 0.550, 0.842, 0.908, 0.938];

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
        // useless unless it is read against the profile above it.
        let i_bins = text.find("renewal  ").expect("profile row");
        let i_as = text.find("ancestor-sampling").expect("acceptance line");
        assert!(i_as > i_bins && text[i_bins..i_as].lines().count() <= 4,
            "as_accept must sit within the same block as the profile:\n{text}");
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
