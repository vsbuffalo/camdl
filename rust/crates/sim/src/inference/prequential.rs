//! Prequential (one-step-ahead out-of-sample) evaluation.
//!
//! See `docs/dev/proposals/2026-04-20-prequential-evaluation.md` for
//! the full design. This module implements Part I: the
//! `PrequentialTrace` struct, proper-scoring-rule kernels (log
//! score, CRPS, PIT), and the summary statistics (elpd, mean CRPS,
//! PIT coverage) callers reach for.
//!
//! PF-side sample emission (the per-step `y_pred_samples` tensor) is
//! wired in `particle_filter.rs`; this module is pure post-processing
//! once the samples are available.
//!
//! Scope: plug-in predictive only (provenance = `PlugIn`). LFO-PSIS,
//! fully-Bayesian, and pseudo-posterior variants are Part II.

use serde::{Serialize, Deserialize};

/// Provenance of the predictive used to compute scores.
///
/// v1 only uses `PlugIn`. The enum is already stable so Part II can
/// add variants without a schema migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Point-estimate (MLE / posterior mean) plug-in predictive.
    /// Proper only when θ is assumed known; overconfident at small n.
    PlugIn,
}

/// A single step's record: observation, predictive samples, and
/// pointwise scores.
///
/// Stored per-step so downstream summaries (total elpd with paired
/// SE, PIT coverage at any level, quantile plots) don't re-run the PF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrequentialStep {
    /// Time index of the assimilated observation.
    pub t: f64,
    /// Observed value at `t`.
    pub y_obs: f64,
    /// Per-particle predictive draws `ỹ^(s) ~ p(y | x_t^(s))`.
    /// Empty if the pipeline elected not to save samples
    /// (`--no-save-samples`). Scalar scores remain valid.
    pub y_pred_samples: Vec<f64>,
    /// log p̂(y_{t+1} | y_{1:t}) = log Σ w^(s) p(y | x^(s)).
    pub log_score: f64,
    /// Continuous Ranked Probability Score (sample estimator).
    pub crps: f64,
    /// Probability integral transform u_t = F̂(y_obs). Should be
    /// Uniform(0, 1) under correct calibration.
    pub pit: f64,
    /// Effective sample size of the filter at this step.
    pub ess: f64,
}

/// Warning attached to a prequential trace — things a reader needs
/// to see before interpreting the summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PrequentialWarning {
    /// ESS dropped below `threshold` at `step_count` scored steps.
    EssCollapse { step_count: usize, threshold: f64 },
    /// `t0` appears lower than the model-class heuristic — scores
    /// may be dominated by initialization variance.
    UnderIdentifiedT0 { t0: usize, heuristic: usize },
    /// The predictive sample array is empty for ≥1 step
    /// (user passed `--no-save-samples`); CRPS recomputed from
    /// log_score+pit cannot be done on these traces.
    SamplesNotSaved,
}

/// The full trace: one entry per scored observation, plus metadata.
///
/// Content-addressed; persisted as JSON alongside the fit artifact
/// and as a human-readable `prequential.tsv`. See §7 of the
/// 2026-04-20 proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrequentialTrace {
    /// Schema version for forward compatibility. Bump on breaking
    /// change; v2 (LFO-PSIS etc.) adds fields as Option so v1 reads
    /// remain valid.
    pub schema_version: u32,
    /// First scored observation index (1-based in y_{1:T}).
    /// Observations y_1 .. y_{t0} initialize the filter and are not
    /// scored.
    pub t0: usize,
    /// How the predictive was constructed.
    pub provenance: Provenance,
    /// Per-step records, length = T - t0.
    pub steps: Vec<PrequentialStep>,
    /// Warnings collected during trace construction.
    pub warnings: Vec<PrequentialWarning>,
}

impl PrequentialTrace {
    /// Total expected log predictive density (elpd_preq).
    pub fn elpd(&self) -> f64 {
        self.steps.iter().map(|s| s.log_score).sum()
    }

    /// Mean CRPS across scored steps.
    pub fn mean_crps(&self) -> f64 {
        if self.steps.is_empty() { return f64::NAN; }
        self.steps.iter().map(|s| s.crps).sum::<f64>() / self.steps.len() as f64
    }

    /// Fraction of observations that fell inside the central
    /// `level`-predictive interval (level ∈ (0, 1)).
    ///
    /// Nominal coverage = `level`. Substantial deviation indicates
    /// miscalibration (plug-in overconfidence is the typical
    /// failure mode).
    pub fn pit_coverage(&self, level: f64) -> f64 {
        if self.steps.is_empty() { return f64::NAN; }
        let half = level / 2.0;
        let lo = 0.5 - half;
        let hi = 0.5 + half;
        let inside = self.steps.iter()
            .filter(|s| s.pit >= lo && s.pit <= hi)
            .count();
        inside as f64 / self.steps.len() as f64
    }

    /// Binned PIT histogram; returns counts of PIT values falling
    /// into each of `bins` equal-width bins on [0, 1].
    pub fn pit_histogram(&self, bins: usize) -> Vec<usize> {
        let mut counts = vec![0usize; bins];
        for s in &self.steps {
            let idx = ((s.pit * bins as f64) as usize).min(bins - 1);
            counts[idx] += 1;
        }
        counts
    }

    /// Number of scored steps (T - t0).
    pub fn n_scored(&self) -> usize { self.steps.len() }
}

// ── Scoring-rule kernels ────────────────────────────────────────────

/// Log-sum-exp-based mixture log-density of the plug-in predictive
/// at the observation.
///
///   log p̂(y | y_{1:t}) = log(Σ w^(s) · p(y | x^(s)))
///
/// Caller provides the per-particle log-likelihoods
/// `log p(y | x^(s))` and the (unnormalized) particle log-weights.
/// Weights are normalized internally.
pub fn log_score_plug_in(log_liks: &[f64], log_weights: &[f64]) -> f64 {
    assert_eq!(log_liks.len(), log_weights.len(),
        "log_liks and log_weights must have the same length");
    if log_liks.is_empty() { return f64::NEG_INFINITY; }

    // log(Σ w^(s) p^(s)) with w normalized:
    //   = logsumexp(log w + log p) − logsumexp(log w).
    let num: Vec<f64> = log_weights.iter().zip(log_liks)
        .map(|(lw, lp)| lw + lp).collect();
    super::types::log_sum_exp(&num) - super::types::log_sum_exp(log_weights)
}

/// Sample-based CRPS via the Hersbach / Laio–Tamea sorted-sample
/// identity:
///
///   ĈRPS = (2/S²) Σ (x_(s) − y) · [S · 1{y < x_(s)} − (s − 1/2)]
///
/// where x_(s) are the samples sorted ascending and s is 1-indexed.
/// O(S log S) via the sort. Equivalent to the naive O(S²) form
///   (1/S)Σ|x^(i) − y| − (1/(2S²))ΣΣ|x^(i) − x^(j)|.
pub fn crps_sample(samples: &[f64], y: f64) -> f64 {
    let s = samples.len();
    if s == 0 { return f64::NAN; }
    if s == 1 { return (samples[0] - y).abs(); }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let s_f = s as f64;
    let mut acc = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let rank = (i + 1) as f64;  // 1-indexed
        let ind = if y < x { 1.0 } else { 0.0 };
        acc += (x - y) * (s_f * ind - (rank - 0.5));
    }
    2.0 * acc / (s_f * s_f)
}

/// Build a `PrequentialTrace` from raw PF recordings and the
/// observation series.
///
/// `recorded` comes from `PFilterResult.prequential` (requires
/// `SMCConfig.record_prequential = true`). `y_obs` is the observation
/// values in the same order as `recorded.obs_times`. `ess_trace`
/// mirrors `PFilterResult.ess_trace`.
///
/// `t0` is the number of leading observations skipped (not scored).
/// Under IC-free inference the first obs is used only to pin x_0;
/// pass `t0 = 1`. Otherwise `t0 = 0`.
///
/// Bootstrap-PF-specific assumption: pre-obs weights are uniform
/// (reset to zero at the end of the previous step), so log-score
/// reduces to `logsumexp(log_liks) − log N`. If this filter ever
/// gains auxiliary weighting, pass weighted log-score here.
pub fn build_trace(
    recorded: &super::particle_filter::PrequentialRecorded,
    y_obs: &[f64],
    ess_trace: &[f64],
    t0: usize,
) -> PrequentialTrace {
    assert_eq!(recorded.obs_times.len(), y_obs.len(),
        "y_obs must align 1:1 with recorded obs_times");
    assert_eq!(recorded.obs_times.len(), ess_trace.len(),
        "ess_trace must align 1:1 with recorded obs_times");

    let mut steps = Vec::with_capacity(recorded.obs_times.len().saturating_sub(t0));
    let mut warnings: Vec<PrequentialWarning> = Vec::new();
    let mut ess_collapse_count = 0usize;
    const ESS_THRESHOLD: f64 = 10.0;

    for idx in t0..recorded.obs_times.len() {
        let log_liks = &recorded.log_liks[idx];
        let samples = &recorded.y_pred_samples[idx];
        let y = y_obs[idx];
        let n = log_liks.len() as f64;

        // Uniform-weight log-score (see docstring).
        let log_score = super::types::log_sum_exp(log_liks) - n.ln();
        let crps = crps_sample(samples, y);
        let pit = pit_sample(samples, y);
        let ess = ess_trace[idx];
        if ess < ESS_THRESHOLD { ess_collapse_count += 1; }

        steps.push(PrequentialStep {
            t: recorded.obs_times[idx],
            y_obs: y,
            y_pred_samples: samples.clone(),
            log_score, crps, pit, ess,
        });
    }

    if ess_collapse_count > 0 {
        warnings.push(PrequentialWarning::EssCollapse {
            step_count: ess_collapse_count,
            threshold: ESS_THRESHOLD,
        });
    }

    PrequentialTrace {
        schema_version: 1,
        t0,
        provenance: Provenance::PlugIn,
        steps,
        warnings,
    }
}

/// Probability integral transform: empirical CDF of the predictive
/// samples evaluated at the observation.
///
/// For continuous predictives this should be Uniform(0, 1) under
/// correct calibration (Dawid 1984; Gneiting-Balabdaoui-Raftery
/// 2007). For discrete observations the point-estimate PIT has
/// stair-step artifacts near integer values — see §12 of the
/// 2026-04-20 proposal for the v2 randomized PIT.
pub fn pit_sample(samples: &[f64], y: f64) -> f64 {
    if samples.is_empty() { return f64::NAN; }
    let n_leq = samples.iter().filter(|&&x| x <= y).count();
    n_leq as f64 / samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn crps_point_mass_equals_abs_error() {
        // With S=1, CRPS reduces to |x - y| (both forms agree at a
        // degenerate predictive). The sorted-sample identity isn't
        // used for S=1 — the shortcut is.
        let c = crps_sample(&[3.0], 5.0);
        assert!(approx_eq(c, 2.0, 1e-12), "got {}", c);
    }

    #[test]
    fn crps_matches_naive_formula() {
        // Compare sorted-sample CRPS against the naive O(S²) form
        // on a small sample.
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 0.5];
        let y = 3.0;

        let s_f = samples.len() as f64;
        let term1: f64 = samples.iter().map(|x: &f64| (x - y).abs()).sum::<f64>() / s_f;
        let term2: f64 = {
            let mut acc = 0.0_f64;
            for a in &samples {
                for b in &samples {
                    acc += (a - b).abs();
                }
            }
            acc / (2.0 * s_f * s_f)
        };
        let naive = term1 - term2;

        let fast = crps_sample(&samples, y);
        assert!(approx_eq(naive, fast, 1e-10),
            "naive = {}, fast = {}", naive, fast);
    }

    #[test]
    fn crps_rewards_sharper_correct_forecast() {
        let y = 5.0;
        // Tight, centered
        let tight: Vec<f64> = (0..100).map(|i| 4.5 + 0.01 * (i as f64)).collect();
        // Diffuse, centered
        let diffuse: Vec<f64> = (0..100).map(|i| 0.0 + 0.1 * (i as f64)).collect();
        let c_tight = crps_sample(&tight, y);
        let c_diffuse = crps_sample(&diffuse, y);
        assert!(c_tight < c_diffuse,
            "sharper forecast should have lower CRPS: tight={}, diffuse={}",
            c_tight, c_diffuse);
    }

    #[test]
    fn log_score_uniform_weights_reduces_to_log_mean_lik() {
        // With uniform weights, log_score = logsumexp(log_liks) - log N
        // = log((1/N)Σ p).
        let log_liks = vec![-1.0, -2.0, -0.5, -3.0];
        let log_weights = vec![0.0; 4];  // uniform (unnormalized)
        let ls = log_score_plug_in(&log_liks, &log_weights);
        let n = log_liks.len() as f64;
        let expected = super::super::types::log_sum_exp(&log_liks) - n.ln();
        assert!(approx_eq(ls, expected, 1e-12), "got {}, expected {}", ls, expected);
    }

    #[test]
    fn log_score_weighted_matches_manual() {
        // Simple two-particle check: log(0.3 · exp(-1) + 0.7 · exp(-2))
        let log_liks = vec![-1.0, -2.0];
        let log_weights = vec![0.3_f64.ln(), 0.7_f64.ln()];
        let ls = log_score_plug_in(&log_liks, &log_weights);
        let expected = (0.3 * (-1.0_f64).exp() + 0.7 * (-2.0_f64).exp()).ln();
        assert!(approx_eq(ls, expected, 1e-10), "got {}, expected {}", ls, expected);
    }

    #[test]
    fn pit_is_uniform_under_correct_forecast() {
        // If samples ~ true distribution and y is a draw from it,
        // PIT at y should be ~ Uniform(0, 1). Use a deterministic
        // large sample vs a middle y.
        let samples: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let p = pit_sample(&samples, 500.0);
        assert!(approx_eq(p, 0.501, 0.01), "got {}", p);  // 501 samples ≤ 500 because 0..=500
    }

    #[test]
    fn pit_coverage_at_perfect_uniform() {
        // A trace whose PIT values span [0, 1] uniformly should have
        // ~level coverage at level. Synthesize 100 evenly-spaced PITs.
        let steps: Vec<PrequentialStep> = (0..100).map(|i| {
            let u = (i as f64 + 0.5) / 100.0;  // 0.005, 0.015, ..., 0.995
            PrequentialStep {
                t: i as f64, y_obs: 0.0, y_pred_samples: vec![],
                log_score: 0.0, crps: 0.0, pit: u, ess: 0.0,
            }
        }).collect();
        let trace = PrequentialTrace {
            schema_version: 1, t0: 0, provenance: Provenance::PlugIn,
            steps, warnings: vec![],
        };
        // 90% interval = PIT in [0.05, 0.95] — 90 of 100 PITs qualify.
        let cov = trace.pit_coverage(0.90);
        assert!(approx_eq(cov, 0.90, 0.01), "got {}", cov);
        // 50% interval = PIT in [0.25, 0.75].
        let cov50 = trace.pit_coverage(0.50);
        assert!(approx_eq(cov50, 0.50, 0.02), "got {}", cov50);
    }

    #[test]
    fn build_trace_from_recorded_aligns_with_kernels() {
        // Hand-rolled PrequentialRecorded with two steps; verify
        // build_trace computes the same log_score / crps / pit as the
        // standalone kernels and forwards ess correctly.
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0],
            log_liks: vec![
                vec![-1.0, -2.0, -0.5, -3.0],
                vec![-0.1, -0.2, -0.3, -0.4],
            ],
            y_pred_samples: vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.5, 5.0, 4.5, 6.0],
            ],
        };
        let y_obs = vec![2.5, 5.2];
        let ess = vec![100.0, 4.0];  // second step below threshold

        let trace = build_trace(&recorded, &y_obs, &ess, 0);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 0);

        // Step 0: log_score = logsumexp(log_liks) - log N
        let expected_ls0 = super::super::types::log_sum_exp(&recorded.log_liks[0])
            - (4.0_f64).ln();
        assert!(approx_eq(trace.steps[0].log_score, expected_ls0, 1e-12));
        // CRPS/PIT agree with kernels.
        assert!(approx_eq(trace.steps[0].crps,
            crps_sample(&recorded.y_pred_samples[0], 2.5), 1e-12));
        assert!(approx_eq(trace.steps[0].pit,
            pit_sample(&recorded.y_pred_samples[0], 2.5), 1e-12));

        // ESS warning fires for the low-ess second step.
        assert_eq!(trace.warnings.len(), 1);
        matches!(trace.warnings[0], PrequentialWarning::EssCollapse { step_count: 1, .. });
    }

    #[test]
    fn build_trace_respects_t0_skip() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0, 3.0],
            log_liks: vec![vec![-1.0; 4]; 3],
            y_pred_samples: vec![vec![0.5, 1.0, 1.5, 2.0]; 3],
        };
        let y_obs = vec![1.25; 3];
        let ess = vec![100.0; 3];

        let trace = build_trace(&recorded, &y_obs, &ess, 1);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 1);
        assert_eq!(trace.steps[0].t, 2.0);
    }

    #[test]
    fn pit_histogram_bins_sum_to_n() {
        let steps: Vec<PrequentialStep> = (0..50).map(|i| PrequentialStep {
            t: i as f64, y_obs: 0.0, y_pred_samples: vec![],
            log_score: 0.0, crps: 0.0, pit: (i as f64) / 50.0, ess: 0.0,
        }).collect();
        let trace = PrequentialTrace {
            schema_version: 1, t0: 0, provenance: Provenance::PlugIn,
            steps, warnings: vec![],
        };
        let hist = trace.pit_histogram(10);
        assert_eq!(hist.iter().sum::<usize>(), 50);
    }
}
