//! gh#685: the `filter_ess` block of `pgas_summary.json` and the
//! `filter_ess.tsv` table beside it — the conditional filter's effective
//! sample size at every observation, pooled over the retained sweeps of every
//! chain.
//!
//! ## Why `min_alive` cannot answer this
//!
//! `collapsed_windows` and `min_alive` (gh#783) count particles whose weight
//! is *finite*. A finite weight can be negligible. On a 19,200-particle Ebola
//! fit every sweep reported `min_alive` between 4,357 and 19,166 — nothing
//! collapsed, on that reading — while at one observation the filter's ESS
//! was 2 to 3, sweep after sweep. That observation was a re-issued count
//! floored to zero; the model could reach it with a handful of particles,
//! every other particle scored a density of roughly e⁻²⁵, and the following
//! resample copied those few particles into every slot. Nothing in the trace
//! or the summary said so. A run whose path through one observation is drawn
//! from three particles is not exploring that part of the path, and the
//! diagnostics that read as healthy (renewal, R̂ on the parameters) did not
//! contradict it.
//!
//! ## What this module publishes
//!
//! Per observation, over the retained post-burn-in sweeps: the **mean** ESS
//! and the **minimum**. The mean is the number that carries the diagnosis —
//! a single sweep with a small ESS is noise, the same observation starving
//! every sweep is a data point the model cannot reach. Over observations, the
//! minimum, a low quantile and the median of the mean profile summarise the
//! series in three numbers. The table carries the whole profile per chain, so
//! a consumer can plot ESS against time and match the trough to a row of the
//! data.
//!
//! An observation is *starved* when its mean ESS is below
//! [`STARVED_ESS_FRACTION`] of the particle count. Neither existing threshold
//! fits here: the bootstrap filter's bail floor (`degeneracy::ESS_FLOOR`, an
//! ESS of 2) would have passed the 2.2 above, and the prequential
//! `ESS_COLLAPSE_FRACTION` of a tenth fires on the healthy peak of an
//! epidemic, where an ESS of 4% of N is ordinary for a filter that resamples
//! at every observation. One percent sits between: 192 of 19,200 is not a
//! swarm anyone would call collapsed, and 3 of 19,200 is not one anyone
//! would call healthy. The numbers are printed regardless of the threshold;
//! the threshold only decides whether a finding points at them.

use serde::{Deserialize, Serialize};
use sim::inference::diagnostic::DiagnosticKind;
use sim::inference::pgas::WeightCollapse;
use std::path::Path;

/// The `filter_ess` key of `pgas_summary.json`. Named next to the type so a
/// producer and any reader cannot drift on the spelling.
pub const FILTER_ESS_KEY: &str = "filter_ess";

/// The per-chain, per-observation table written beside the summary.
pub const FILTER_ESS_TSV: &str = "filter_ess.tsv";

/// An observation is starved when its mean filter ESS over the retained
/// sweeps is below this fraction of the particle count. See the module doc
/// for why it is neither the bail floor nor the prequential tenth.
pub const STARVED_ESS_FRACTION: f64 = 0.01;

/// The low quantile of the mean-ESS profile reported over observations.
pub const LOW_QUANTILE: f64 = 0.10;

/// Starved observations carried in the summary block and printed, worst
/// first. The table holds every observation.
pub const WORST_ROWS: usize = 5;

/// The filter ESS at one observation, over a set of sweeps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsFilterEss {
    /// Index into the stage's observation slice — the row of the data.
    pub obs: usize,
    /// The observation's time, in the model's time unit.
    pub time: f64,
    /// Mean ESS over the sweeps that scored this observation.
    pub mean: f64,
    /// Smallest ESS any of those sweeps recorded here.
    pub min: f64,
    /// Sweeps that scored this observation.
    pub n_sweeps: usize,
}

/// One chain's profile — the rows of [`FILTER_ESS_TSV`] for that chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainFilterEss {
    /// Zero-based, as the stage numbers chains internally; written one-based
    /// to match `chain_<n>/`.
    pub chain_id: usize,
    pub by_obs: Vec<ObsFilterEss>,
}

/// The `filter_ess` block of `pgas_summary.json`: the pooled summary and the
/// starved observations, worst first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterEss {
    /// Particles per sweep — the denominator every ESS here is read against.
    pub n_particles: usize,
    /// [`STARVED_ESS_FRACTION`], stated in the artifact.
    pub starved_fraction: f64,
    /// `starved_fraction × n_particles`: the bar a mean ESS is under.
    pub starved_below: f64,
    /// Observations scored by at least one retained sweep.
    pub n_obs: usize,
    /// Of those, the starved ones.
    pub n_starved: usize,
    /// The starved observations, smallest mean ESS first, at most
    /// [`WORST_ROWS`] of them. Empty when nothing is starved: the worst
    /// observation is then [`Self::mean_min`], which is not a finding.
    pub worst: Vec<ObsFilterEss>,
    /// Over observations, the smallest mean ESS.
    pub mean_min: f64,
    /// Over observations, the [`LOW_QUANTILE`] of the mean ESS.
    pub mean_low: f64,
    /// Over observations, the median mean ESS.
    pub mean_median: f64,
    /// [`LOW_QUANTILE`], stated in the artifact.
    pub low_quantile: f64,
    /// Retained post-burn-in sweeps pooled, summed across chains.
    pub n_sweeps: usize,
    /// Surviving chains that contributed at least one sweep.
    pub n_chains: usize,
    /// The table beside this block.
    pub table: String,
}

/// The stage-level result: the summary block plus the profiles it was
/// reduced from, which go to the table rather than the JSON.
#[derive(Debug, Clone)]
pub struct FilterEssStage {
    pub summary: FilterEss,
    /// Pooled over chains — the profile the summary's numbers come from.
    pub pooled: Vec<ObsFilterEss>,
    pub per_chain: Vec<ChainFilterEss>,
}

/// Per-chain running sums, one slot per observation.
#[derive(Debug, Clone)]
struct ChainAccum {
    chain_id: usize,
    sum: Vec<f64>,
    min: Vec<f64>,
    n: Vec<usize>,
    n_sweeps: usize,
}

/// Accumulator for [`FilterEssStage`] over the retained sweeps of every
/// chain.
///
/// A type rather than loose sums at the call site, for the same reason
/// `PathRenewalAccum` is one: the "an unscored observation is skipped, not
/// zeroed" rule is written once, and the per-chain and pooled profiles
/// cannot be reduced out of step with each other.
#[derive(Debug, Clone)]
pub struct FilterEssAccum {
    n_particles: usize,
    obs_times: Vec<f64>,
    chains: Vec<ChainAccum>,
}

impl FilterEssAccum {
    /// `obs_times` is the stage's observation slice, in the order `csmc_as`
    /// indexes it — the same slice every sweep's `ess_by_obs` is indexed by.
    pub fn new(n_particles: usize, obs_times: Vec<f64>) -> Self {
        FilterEssAccum { n_particles, obs_times, chains: Vec::new() }
    }

    /// Fold in one chain's retained post-burn-in sweeps.
    ///
    /// A sweep with an empty profile weighed nothing (`WeightCollapse::none`,
    /// the placeholder for a run with no CSMC sweep) and is skipped. A sweep
    /// whose profile is the wrong length is not the same observation slice,
    /// and is an error rather than a silent skip. A chain contributing no
    /// sweep is not counted in [`FilterEss::n_chains`].
    pub fn add_chain<'a, I>(&mut self, chain_id: usize, sweeps: I) -> Result<(), String>
    where
        I: IntoIterator<Item = &'a WeightCollapse>,
    {
        let n_obs = self.obs_times.len();
        let mut acc = ChainAccum {
            chain_id,
            sum: vec![0.0; n_obs],
            min: vec![f64::INFINITY; n_obs],
            n: vec![0; n_obs],
            n_sweeps: 0,
        };
        for wc in sweeps {
            if wc.ess_by_obs.is_empty() {
                continue;
            }
            if wc.ess_by_obs.len() != n_obs {
                return Err(format!(
                    "filter ESS profile of chain {} has {} entries for {} observations",
                    chain_id + 1,
                    wc.ess_by_obs.len(),
                    n_obs
                ));
            }
            for (i, &e) in wc.ess_by_obs.iter().enumerate() {
                // NaN is "this sweep did not score this observation". Skipped,
                // not zeroed: averaging a zero in would report a starvation
                // that was never measured.
                if e.is_nan() {
                    continue;
                }
                acc.sum[i] += e;
                acc.min[i] = acc.min[i].min(e);
                acc.n[i] += 1;
            }
            acc.n_sweeps += 1;
        }
        if acc.n_sweeps > 0 {
            self.chains.push(acc);
        }
        Ok(())
    }

    /// Close the accumulator.
    ///
    /// `None` when no sweep scored any observation: the block is then omitted
    /// from the summary entirely rather than written as a row of nulls, which
    /// a consumer would read as "measured, found nothing".
    pub fn finish(self) -> Option<FilterEssStage> {
        let n_obs = self.obs_times.len();
        let profile = |sum: &[f64], min: &[f64], n: &[usize]| -> Vec<ObsFilterEss> {
            (0..n_obs)
                .filter(|&i| n[i] > 0)
                .map(|i| ObsFilterEss {
                    obs: i,
                    time: self.obs_times[i],
                    mean: sum[i] / n[i] as f64,
                    min: min[i],
                    n_sweeps: n[i],
                })
                .collect()
        };
        let mut sum = vec![0.0; n_obs];
        let mut min = vec![f64::INFINITY; n_obs];
        let mut n = vec![0usize; n_obs];
        for c in &self.chains {
            for i in 0..n_obs {
                sum[i] += c.sum[i];
                min[i] = min[i].min(c.min[i]);
                n[i] += c.n[i];
            }
        }
        let pooled = profile(&sum, &min, &n);
        if pooled.is_empty() {
            return None;
        }
        let per_chain: Vec<ChainFilterEss> = self
            .chains
            .iter()
            .map(|c| ChainFilterEss { chain_id: c.chain_id, by_obs: profile(&c.sum, &c.min, &c.n) })
            .collect();

        let means: Vec<f64> = pooled.iter().map(|o| o.mean).collect();
        let starved_below = STARVED_ESS_FRACTION * self.n_particles as f64;
        let mut starved: Vec<ObsFilterEss> =
            pooled.iter().filter(|o| o.mean < starved_below).cloned().collect();
        starved.sort_by(|a, b| {
            a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal).then(a.obs.cmp(&b.obs))
        });
        let n_starved = starved.len();
        starved.truncate(WORST_ROWS);
        let summary = FilterEss {
            n_particles: self.n_particles,
            starved_fraction: STARVED_ESS_FRACTION,
            starved_below,
            n_obs: pooled.len(),
            n_starved,
            worst: starved,
            mean_min: means.iter().cloned().fold(f64::INFINITY, f64::min),
            mean_low: crate::quantile::quantile(&means, LOW_QUANTILE),
            mean_median: crate::quantile::quantile(&means, 0.5),
            low_quantile: LOW_QUANTILE,
            n_sweeps: self.chains.iter().map(|c| c.n_sweeps).sum(),
            n_chains: self.chains.len(),
            table: FILTER_ESS_TSV.to_string(),
        };
        Some(FilterEssStage { summary, pooled, per_chain })
    }
}

impl FilterEssStage {
    /// Write [`FILTER_ESS_TSV`]: the pooled profile first (`chain` = `all`),
    /// then one block per chain, one row per observation scored by at least
    /// one of that chain's retained sweeps. Column `obs` is the observation's
    /// index in the stage's observation slice, so a row joins to the data
    /// without a time match.
    pub fn write_tsv(&self, path: &Path) -> Result<(), String> {
        use std::io::Write;
        let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "chain\tobs\ttime\tess_mean\tess_min\tn_sweeps").map_err(|e| e.to_string())?;
        let mut row = |chain: &str, o: &ObsFilterEss| -> Result<(), String> {
            writeln!(
                w,
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{}",
                chain,
                o.obs,
                crate::quantile::fmt_time(o.time),
                o.mean,
                o.min,
                o.n_sweeps
            )
            .map_err(|e| e.to_string())
        };
        for o in &self.pooled {
            row("all", o)?;
        }
        for c in &self.per_chain {
            let label = (c.chain_id + 1).to_string();
            for o in &c.by_obs {
                row(&label, o)?;
            }
        }
        Ok(())
    }
}

impl FilterEss {
    /// The starvation finding, or `None` when no observation is starved.
    ///
    /// One finding per stage, over the pooled profile, rather than one per
    /// chain or per observation: the block published beside it is the pooled
    /// one, and a reader comparing the message against the artifact must see
    /// the same numbers.
    pub fn starvation_finding(&self) -> Option<DiagnosticKind> {
        let worst = self.worst.first()?;
        Some(DiagnosticKind::FilterStarved {
            n_starved: self.n_starved,
            n_obs: self.n_obs,
            n_particles: self.n_particles,
            starved_below: self.starved_below,
            worst_time: worst.time,
            worst_mean: worst.mean,
            worst_min: worst.min,
        })
    }

    /// The end-of-stage block, printed unconditionally: the three numbers
    /// over observations, the starved count against the bar, and the worst
    /// rows when there are any.
    pub fn report(&self) -> String {
        let mut s = format!(
            "\nfilter ESS (gh#685; over {} retained post-burn-in sweep(s) across {} \
             chain(s), {} particles):\n  mean ESS over sweeps, by observation: \
             min {:.1}   {:.0}% quantile {:.1}   median {:.1}\n",
            self.n_sweeps,
            self.n_chains,
            self.n_particles,
            self.mean_min,
            self.low_quantile * 100.0,
            self.mean_low,
            self.mean_median,
        );
        s.push_str(&format!(
            "  {} of {} observations starved (mean ESS below {:.0}, {:.0}% of the \
             particle count)\n",
            self.n_starved,
            self.n_obs,
            self.starved_below,
            self.starved_fraction * 100.0,
        ));
        if !self.worst.is_empty() {
            s.push_str(&format!(
                "    {:>4} {:>10} {:>10} {:>10} {:>8}\n",
                "obs", "time", "ess_mean", "ess_min", "sweeps"
            ));
            for o in &self.worst {
                s.push_str(&format!(
                    "    {:>4} {:>10} {:>10.1} {:>10.1} {:>8}\n",
                    o.obs,
                    crate::quantile::fmt_time(o.time),
                    o.mean,
                    o.min,
                    o.n_sweeps
                ));
            }
            if self.n_starved > self.worst.len() {
                s.push_str(&format!(
                    "    ({} more in {})\n",
                    self.n_starved - self.worst.len(),
                    self.table
                ));
            }
        }
        s.push_str(&format!("  per chain and observation: {}\n", self.table));
        s
    }

    /// The block as `pgas_summary.json` carries it.
    pub fn summary_block(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("every field of FilterEss is finite")
    }

    /// Read the block back from a parsed `pgas_summary.json`. `None` when the
    /// stage wrote none (it predates the block, or no sweep scored anything).
    pub fn read(summary: &serde_json::Value) -> Option<FilterEss> {
        summary.get(FILTER_ESS_KEY).and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::pgas::WeightCollapseTally;

    /// A sweep whose weight vectors give the listed ESS values, one per
    /// observation, built through the sim-side tally so the fixture is the
    /// shape `csmc_as` produces. `k` equal weights have ESS exactly `k`, so
    /// the targets are integers. `None` is an observation this sweep did not
    /// score.
    fn sweep(ess: &[Option<f64>]) -> WeightCollapse {
        let mut t = WeightCollapseTally::new(1000, ess.len());
        for (i, e) in ess.iter().enumerate() {
            if let Some(e) = e {
                assert_eq!(e.fract(), 0.0, "integer ESS targets only");
                t.record(i, i, &vec![0.0; *e as usize]);
            }
        }
        t.finish(false)
    }

    fn stage(n_particles: usize, chains: &[Vec<WeightCollapse>]) -> Option<FilterEssStage> {
        let n_obs = chains.iter().flatten().map(|w| w.ess_by_obs.len()).max().unwrap_or(0);
        let times: Vec<f64> = (0..n_obs).map(|i| 10.0 * i as f64).collect();
        let mut acc = FilterEssAccum::new(n_particles, times);
        for (c, sweeps) in chains.iter().enumerate() {
            acc.add_chain(c, sweeps.iter()).unwrap();
        }
        acc.finish()
    }

    #[test]
    fn pooled_profile_is_mean_and_min_over_every_sweep_of_every_chain() {
        let st = stage(
            1000,
            &[
                vec![sweep(&[Some(1000.0), Some(4.0), Some(900.0)]), sweep(&[Some(800.0), Some(2.0), Some(700.0)])],
                vec![sweep(&[Some(600.0), Some(3.0), Some(500.0)])],
            ],
        )
        .expect("three sweeps scored");
        let s = &st.summary;
        assert_eq!((s.n_sweeps, s.n_chains, s.n_obs), (3, 2, 3));
        assert_eq!(st.pooled.len(), 3);
        let p = &st.pooled[1];
        assert_eq!((p.obs, p.time, p.n_sweeps), (1, 10.0, 3));
        assert!((p.mean - 3.0).abs() < 1e-9, "mean over the three sweeps, got {}", p.mean);
        assert!((p.min - 2.0).abs() < 1e-9, "min over the three sweeps, got {}", p.min);
        assert!((st.pooled[0].mean - 800.0).abs() < 1e-9);
        // The per-chain profiles are the same reduction per chain.
        assert_eq!(st.per_chain.len(), 2);
        assert!((st.per_chain[0].by_obs[1].mean - 3.0).abs() < 1e-9);
        assert!((st.per_chain[1].by_obs[1].mean - 3.0).abs() < 1e-9);
        assert!((st.per_chain[1].by_obs[0].mean - 600.0).abs() < 1e-9);
        // Over observations: means are [800, 3, 700].
        assert!((s.mean_min - 3.0).abs() < 1e-9);
        assert!((s.mean_median - 700.0).abs() < 1e-9);
        assert!((s.mean_low - crate::quantile::quantile(&[800.0, 3.0, 700.0], LOW_QUANTILE)).abs() < 1e-9);
    }

    /// The Ebola shape: `min_alive` reads the full swarm everywhere, and one
    /// observation's mean ESS is 3 of 1000. It is the one starved
    /// observation, the finding names it, and the healthy 4%-of-N trough at
    /// the peak is not a finding.
    #[test]
    fn starvation_is_a_mean_below_one_percent_of_the_particle_count() {
        let st = stage(
            1000,
            &[vec![
                sweep(&[Some(1000.0), Some(40.0), Some(4.0), Some(900.0)]),
                sweep(&[Some(1000.0), Some(40.0), Some(2.0), Some(900.0)]),
            ]],
        )
        .unwrap();
        let s = &st.summary;
        assert_eq!(s.starved_below, 10.0);
        assert_eq!(s.n_starved, 1);
        assert_eq!(s.worst.len(), 1);
        assert_eq!(s.worst[0].obs, 2);
        let d = s.starvation_finding().expect("one observation is starved");
        match d {
            DiagnosticKind::FilterStarved { n_starved, n_obs, n_particles, worst_time, worst_mean, worst_min, .. } => {
                assert_eq!((n_starved, n_obs, n_particles), (1, 4, 1000));
                assert_eq!(worst_time, 20.0);
                assert!((worst_mean - 3.0).abs() < 1e-9);
                assert!((worst_min - 2.0).abs() < 1e-9);
            }
            other => panic!("wrong finding: {other:?}"),
        }
        assert!(s.report().contains("1 of 4 observations starved"));

        // Raise the swarm at that observation to the bar and nothing fires.
        let healthy = stage(1000, &[vec![sweep(&[Some(1000.0), Some(40.0), Some(10.0), Some(900.0)])]]).unwrap();
        assert_eq!(healthy.summary.n_starved, 0);
        assert!(healthy.summary.worst.is_empty());
        assert!(healthy.summary.starvation_finding().is_none());
        assert!(healthy.summary.report().contains("0 of 4 observations starved"));
    }

    /// The worst rows are sorted by mean and capped; the count is not.
    #[test]
    fn worst_rows_are_the_smallest_means_first_and_capped() {
        let ess: Vec<Option<f64>> = (0..8).map(|i| Some(1.0 + i as f64)).collect();
        let st = stage(1000, &[vec![sweep(&ess)]]).unwrap();
        let s = &st.summary;
        assert_eq!(s.n_starved, 8, "every mean is below 10");
        assert_eq!(s.worst.len(), WORST_ROWS);
        let order: Vec<usize> = s.worst.iter().map(|o| o.obs).collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
        assert!(s.report().contains(&format!("({} more in {})", 8 - WORST_ROWS, FILTER_ESS_TSV)));
    }

    /// An observation no sweep scored is absent from the profile, not a zero;
    /// a sweep that weighed nothing is skipped; a chain of such sweeps is not
    /// a contributing chain; a stage of them has no block at all.
    #[test]
    fn unscored_observations_and_empty_sweeps_are_skipped_not_zeroed() {
        let st = stage(
            1000,
            &[
                vec![sweep(&[None, Some(5.0), Some(500.0)]), WeightCollapse::none(1000)],
                vec![WeightCollapse::none(1000)],
            ],
        )
        .unwrap();
        let s = &st.summary;
        assert_eq!((s.n_sweeps, s.n_chains, s.n_obs), (1, 1, 2));
        let obs: Vec<usize> = st.pooled.iter().map(|o| o.obs).collect();
        assert_eq!(obs, vec![1, 2], "observation 0 was never scored");
        assert!((s.mean_min - 5.0).abs() < 1e-9);

        assert!(stage(1000, &[vec![WeightCollapse::none(1000)]]).is_none());
        assert!(stage(1000, &[]).is_none());
    }

    #[test]
    fn a_profile_of_the_wrong_length_is_an_error_not_a_skip() {
        let mut acc = FilterEssAccum::new(1000, vec![0.0, 1.0, 2.0]);
        let wrong = sweep(&[Some(1.0), Some(2.0)]);
        let err = acc.add_chain(0, std::iter::once(&wrong)).unwrap_err();
        assert!(err.contains("2 entries for 3 observations"), "{err}");
    }

    #[test]
    fn summary_block_round_trips_and_the_table_has_one_row_per_chain_and_observation() {
        let st = stage(
            1000,
            &[
                vec![sweep(&[Some(100.0), Some(4.0)])],
                vec![sweep(&[Some(200.0), None])],
            ],
        )
        .unwrap();
        let block = st.summary.summary_block();
        let wrapped = serde_json::json!({ FILTER_ESS_KEY: block });
        let back = FilterEss::read(&wrapped).expect("block reads back");
        assert_eq!(back.n_starved, 1);
        assert_eq!(back.worst, st.summary.worst);
        assert!(FilterEss::read(&serde_json::json!({})).is_none());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILTER_ESS_TSV);
        st.write_tsv(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "chain\tobs\ttime\tess_mean\tess_min\tn_sweeps");
        // Pooled: two observations; chain 1: two; chain 2: one (obs 1 unscored).
        assert_eq!(lines.len(), 1 + 2 + 2 + 1);
        assert_eq!(lines[1], "all\t0\t0\t150.0000\t100.0000\t2");
        assert_eq!(lines[2], "all\t1\t10\t4.0000\t4.0000\t1");
        assert_eq!(lines[3], "1\t0\t0\t100.0000\t100.0000\t1");
        assert_eq!(lines[5], "2\t0\t0\t200.0000\t200.0000\t1");
    }
}
