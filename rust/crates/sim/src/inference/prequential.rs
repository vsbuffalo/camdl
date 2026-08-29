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

/// How the parameters were conditioned on the data when scoring — the **second**
/// optimism axis (#295), orthogonal to [`Provenance`] (the plug-in-vs-posterior
/// parameter treatment).
///
/// v1 only uses `InSample`: θ is fit to the *full* series, so every
/// "one-step-ahead" score has already seen the future through θ. The score is
/// then one-step-ahead in `y` but in-sample in `θ` — optimistic in absolute
/// level, not a true out-of-sample forecast score. Honest out-of-sample
/// (`Lfo` / `RollingOrigin`) is #295 Part II; those variants append here without
/// a schema migration when they land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Conditioning {
    /// θ fit to all of `y_{1:T}` → one-step-ahead in `y` but not in `θ`.
    #[default]
    InSample,
}

/// One stream's (district's) score at a single step (gh#269).
///
/// The `--save-prequential` joint score is the cross-stream sum; this
/// breaks it out per stream so a national fit can be diagnosed
/// district-by-district. Computed with the SAME proper-scoring-rule
/// kernels (log score, CRPS, PIT) as the joint, against the stream's own
/// observed value and predictive sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamScore {
    /// Stream name (the bound observation block / district).
    pub stream: String,
    /// This stream's observed value at the step.
    pub y_obs: f64,
    /// Per-particle predictive draws for this stream.
    /// Cleared under `--no-save-samples` (mirrors the joint).
    pub y_pred_samples: Vec<f64>,
    /// log p̂(y^stream | y_{1:t}) for this stream.
    pub log_score: f64,
    /// Per-stream CRPS.
    pub crps: f64,
    /// Per-stream randomized PIT (same construction as the joint `pit`,
    /// with its own uniform draw).
    pub pit: f64,
    /// This stream's plot-ready predictive interval (median + 50%/90% bands)
    /// from its predictive samples. The canonical per-district forecast band.
    /// `#[serde(default)]`: v1 traces (no interval) still deserialize.
    #[serde(default)]
    pub interval: PredInterval,
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
    /// Randomized probability integral transform
    /// u_t = P̂(X < y_obs) + v·P̂(X = y_obs), v ~ Uniform(0, 1)
    /// (`pit_sample_randomized`; seed in the trace's
    /// `pit_randomization_seed`). Uniform(0, 1) under correct
    /// calibration, including for count predictives.
    pub pit: f64,
    /// Effective sample size of the filter at this step.
    pub ess: f64,
    /// Plot-ready predictive interval (median + 50%/90% bands) of the JOINT
    /// predictive samples — the equal-tailed central interval to draw against
    /// the observed point. `#[serde(default)]`: v1 traces still deserialize.
    #[serde(default)]
    pub interval: PredInterval,
    /// Per-stream (per-district) score breakdown (gh#269). One entry per
    /// SCHEDULED, NON-HOLE stream at this step; `Σ per_stream.log_score`
    /// need not equal `log_score` (log-of-sum vs sum-of-logs), but each
    /// uses the same kernels on its own predictive. `#[serde(default)]`:
    /// v1 traces (no per-stream field) still deserialize.
    #[serde(default)]
    pub per_stream: Vec<StreamScore>,
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
    /// How the predictive was constructed — the parameter-treatment axis
    /// (`plug_in` today; `posterior` is #295 Part I.2).
    pub provenance: Provenance,
    /// How the parameters were conditioned on the data — the second optimism
    /// axis (#295). `#[serde(default)]` so a pre-#295 `prequential.json` (no
    /// field) reads as `InSample`, which is factually what every such trace is.
    #[serde(default)]
    pub conditioning: Conditioning,
    /// Per-step records, length = T - t0.
    pub steps: Vec<PrequentialStep>,
    /// Warnings collected during trace construction.
    pub warnings: Vec<PrequentialWarning>,
    /// Seed of the randomized-PIT uniform draws (`pit_sample_randomized`):
    /// one `v` per scored value, drawn from
    /// `StatefulRng::new_stream(seed, PIT_RNG_STREAM)` in step order (joint
    /// first, then each present stream). Recording it makes the trace's PIT
    /// values reproducible. `None` on traces written before the randomized
    /// PIT (gh#629); `#[serde(default)]` so those still deserialize.
    #[serde(default)]
    pub pit_randomization_seed: Option<u64>,
}

impl PrequentialTrace {
    /// Total expected log predictive density (elpd_preq).
    pub fn elpd(&self) -> f64 {
        self.steps.iter().map(|s| s.log_score).sum()
    }

    /// A one-line caveat when the score is optimistic on either axis — plug-in
    /// (under-dispersed: scored at a single θ) and/or in-sample (θ fit to all
    /// the data, so it has seen the future). Returns `None` only when the score
    /// is honest on both axes (out-of-sample posterior). #295: the number must
    /// never be silently read as a true out-of-sample forecast score.
    ///
    /// The in-sample clause names the bias in *differences* as well as in the
    /// level, because a difference is what model comparison reads. The
    /// in-sample optimism is not a shared constant that cancels: it is roughly
    /// the effective number of parameters θ was free to tune against these very
    /// observations (the bias term the AIC/DIC/WAIC corrections estimate), so
    /// the more flexible of two models collects more of it and Δelpd tilts its
    /// way.
    pub fn optimism_caveat(&self) -> Option<String> {
        let plug_in = self.provenance == Provenance::PlugIn;
        let in_sample = self.conditioning == Conditioning::InSample;
        if !plug_in && !in_sample {
            return None;
        }
        let mut why: Vec<&str> = Vec::new();
        if plug_in {
            why.push("plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed)");
        }
        if in_sample {
            why.push(
                "in-sample (θ fit to all the data → optimistic in absolute level, by \
                 roughly the effective number of parameters fit, which does not cancel \
                 in a Δelpd and so biases differences toward the more flexible model)",
            );
        }
        Some(format!(
            "scores are {} — not a leave-future-out forecast score. See LFO (#295).",
            why.join("; and "),
        ))
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
/// `pit_seed` seeds the randomized-PIT uniform draws
/// (`pit_sample_randomized`); callers pass their filter seed, and the
/// trace records it (`pit_randomization_seed`). Draws are consumed in
/// step order — one for the joint score, then one per present stream —
/// so a trace is reproducible from (seed, data).
///
/// Bootstrap-PF-specific assumption: pre-obs weights are uniform
/// (reset to zero at the end of the previous step), so log-score
/// reduces to `logsumexp(log_liks) − log N`. If this filter ever
/// gains auxiliary weighting, pass weighted log-score here.
pub fn build_trace(
    recorded: &super::particle_filter::PrequentialRecorded,
    y_obs: &[f64],
    per_stream_observed: &[Vec<f64>],
    ess_trace: &[f64],
    t0: usize,
    pit_seed: u64,
) -> PrequentialTrace {
    assert_eq!(recorded.obs_times.len(), y_obs.len(),
        "y_obs must align 1:1 with recorded obs_times");
    assert_eq!(recorded.obs_times.len(), per_stream_observed.len(),
        "per_stream_observed must align 1:1 with recorded obs_times");
    assert_eq!(recorded.obs_times.len(), ess_trace.len(),
        "ess_trace must align 1:1 with recorded obs_times");

    let mut steps = Vec::with_capacity(recorded.obs_times.len().saturating_sub(t0));
    let mut warnings: Vec<PrequentialWarning> = Vec::new();
    let mut ess_collapse_count = 0usize;
    const ESS_THRESHOLD: f64 = 10.0;
    let mut pit_rng = crate::rng::StatefulRng::new_stream(pit_seed, PIT_RNG_STREAM);

    for idx in t0..recorded.obs_times.len() {
        let log_liks = &recorded.log_liks[idx];
        let samples = &recorded.y_pred_samples[idx];
        let y = y_obs[idx];
        let n = log_liks.len() as f64;

        // gh#636: presence mask for this step. A hole (NA) or a
        // not-scheduled sibling carries a non-finite observed value; the
        // per-stream loop below already skips those. The JOINT scores must
        // cover the SAME stream set: the recorded joint predictive sums every
        // stream's draw, so at a partial-hole step it would be compared
        // against an observed sum that omits the missing stream — a
        // numerator/denominator mismatch, not a fictitious 0. Recompute the
        // joint predictive over the PRESENT streams only; a step where no
        // stream is present carries no information and is skipped entirely.
        let present: Vec<usize> = (0..recorded.stream_names.len())
            .filter(|&si| per_stream_observed[idx][si].is_finite())
            .collect();
        if present.is_empty() && !recorded.stream_names.is_empty() {
            continue;
        }
        let all_present = present.len() == recorded.stream_names.len();
        let joint_samples: Vec<f64> = if all_present {
            // Hole-free step: the recorded joint is exactly this sum — reuse
            // it so hole-free traces stay byte-identical.
            samples.clone()
        } else {
            let n_particles = recorded.per_stream_samples[idx]
                .first().map(|v| v.len()).unwrap_or(0);
            (0..n_particles)
                .map(|pi| present.iter()
                    .map(|&si| recorded.per_stream_samples[idx][si][pi])
                    .filter(|v| v.is_finite())
                    .sum::<f64>())
                .collect()
        };

        // Uniform-weight log-score (see docstring).
        let log_score = super::types::log_sum_exp(log_liks) - n.ln();
        let crps = crps_sample(&joint_samples, y);
        let pit = pit_sample_randomized(&joint_samples, y, pit_rng.uniform());
        let ess = ess_trace[idx];
        if ess < ESS_THRESHOLD { ess_collapse_count += 1; }

        // gh#269: per-stream breakdown. Skip a stream whose observed value is
        // non-finite (not scheduled at this union index, or a hole) — it has
        // no term at this step. Each scored stream uses the SAME kernels on its
        // own predictive sample + observed value.
        let mut per_stream: Vec<StreamScore> = Vec::new();
        for si in 0..recorded.stream_names.len() {
            let y_s = per_stream_observed[idx][si];
            if !y_s.is_finite() { continue; }
            let ll_s = &recorded.per_stream_log_liks[idx][si];
            let samp_s = &recorded.per_stream_samples[idx][si];
            let n_s = ll_s.len() as f64;
            per_stream.push(StreamScore {
                stream: recorded.stream_names[si].clone(),
                y_obs: y_s,
                y_pred_samples: samp_s.clone(),
                log_score: super::types::log_sum_exp(ll_s) - n_s.ln(),
                crps: crps_sample(samp_s, y_s),
                pit: pit_sample_randomized(samp_s, y_s, pit_rng.uniform()),
                interval: PredInterval::from_samples(samp_s),
            });
        }

        steps.push(PrequentialStep {
            t: recorded.obs_times[idx],
            y_obs: y,
            interval: PredInterval::from_samples(&joint_samples),
            y_pred_samples: joint_samples,
            log_score, crps, pit, ess,
            per_stream,
        });
    }

    if ess_collapse_count > 0 {
        warnings.push(PrequentialWarning::EssCollapse {
            step_count: ess_collapse_count,
            threshold: ESS_THRESHOLD,
        });
    }

    PrequentialTrace {
        schema_version: 2,
        t0,
        // This builder scores a single filter pass at one θ over the full data:
        // plug-in + in-sample. The posterior / LFO producers (#295) stamp their
        // own values.
        provenance: Provenance::PlugIn,
        conditioning: Conditioning::InSample,
        steps,
        warnings,
        pit_randomization_seed: Some(pit_seed),
    }
}

/// RNG stream for the randomized-PIT draws. Disjoint from the filter's
/// per-particle streams (low indices) and `RESAMPLE_RNG_STREAM` (1<<48),
/// so passing the filter seed to `build_trace` cannot correlate the PIT
/// randomization with the filter's own draws.
pub const PIT_RNG_STREAM: u64 = 1u64 << 49;

/// Randomized probability integral transform (PIT) of the predictive
/// samples at the observation (Smith 1985; Brockwell 2007):
///
///   u = P̂(X < y) + v · P̂(X = y),    v ~ Uniform(0, 1)
///
/// where P̂ is the empirical distribution of `samples` and the uniform
/// draw `v` is supplied by the caller — one per scored value.
/// `build_trace` draws them from a seeded stream and records the seed
/// in the trace (`pit_randomization_seed`) for reproducibility.
///
/// Under calibration `u` is Uniform(0, 1) even for discrete
/// predictives — camdl's dominant case (count observation models) —
/// where the naive `P̂(X ≤ y)` form is biased at atoms: all probability
/// mass AT the observed count is counted as "≤ y", shifting PIT upward
/// wherever ties occur, which distorts coverage and uniformity
/// diagnostics (gh#629). For a continuous predictive ties have
/// probability zero and `u` reduces to the empirical CDF as before.
///
/// Czado, Gneiting & Held (2009) is the count-data assessment
/// reference; their own *nonrandomized* PIT is the mean of this
/// quantity over `v` (`v = 0.5` at a sample predictive). The
/// randomized form is chosen so each scored value is itself uniform
/// and every existing consumer (coverage, histogram) reads it
/// unchanged; see §3.3 of the 2026-08-29 honest-predictive-evaluation
/// proposal.
///
/// Tie detection is exact `f64` equality: count predictive draws are
/// integers stored exactly, and for continuous draws equality is
/// measure-zero.
pub fn pit_sample_randomized(samples: &[f64], y: f64, v: f64) -> f64 {
    if samples.is_empty() { return f64::NAN; }
    let n_lt = samples.iter().filter(|&&x| x < y).count();
    let n_eq = samples.iter().filter(|&&x| x == y).count();
    (n_lt as f64 + v * n_eq as f64) / samples.len() as f64
}

/// Equal-tailed empirical quantile of an ALREADY-SORTED ascending slice, by
/// linear interpolation between order statistics (numpy `quantile` / R
/// `type = 7`, the forecast-hub default). `q ∈ [0, 1]`. NaN on empty.
pub fn sample_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    if sorted.len() == 1 { return sorted[0]; }
    let h = (sorted.len() as f64 - 1.0) * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The canonical plot-ready predictive interval: the equal-tailed central
/// interval of the one-step-ahead predictive (observation-scale), as
/// median + 50% (q25–q75) + 90% (q05–q95) bands — the forecast-vs-observed
/// fan chart. These are the standard quantiles every forecast hub reports;
/// the band the observed point is checked against (and what PIT / 90%-coverage
/// are defined against). Discrete count predictives use the same empirical
/// quantiles (the band is a visual envelope; fractional bounds are fine).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PredInterval {
    pub q05: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub q95: f64,
}

impl PredInterval {
    /// Compute the band from predictive samples. NaN-safe: non-finite entries
    /// (e.g. a not-scheduled stream's NaN placeholder) are dropped first.
    /// Empty after filtering ⇒ all-NaN interval.
    pub fn from_samples(samples: &[f64]) -> Self {
        let mut s: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        PredInterval {
            q05: sample_quantile_sorted(&s, 0.05),
            q25: sample_quantile_sorted(&s, 0.25),
            q50: sample_quantile_sorted(&s, 0.50),
            q75: sample_quantile_sorted(&s, 0.75),
            q95: sample_quantile_sorted(&s, 0.95),
        }
    }
}

#[cfg(test)]
mod tests {

    /// gh#636: holes are first-class. At a partial-hole step the JOINT
    /// predictive covers only the present streams (the recorded joint sums
    /// every stream's draw — comparing that against an observed sum missing a
    /// stream was a numerator/denominator mismatch); an all-hole step carries
    /// no information and is omitted; a hole-free step reuses the recorded
    /// joint byte-identically.
    #[test]
    fn build_trace_skips_holes_consistently() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![7.0, 14.0, 21.0],
            log_liks: vec![vec![-2.0, -2.0]; 3],
            // Recorded joint = A + B at every step (the recorder cannot know
            // about holes).
            y_pred_samples: vec![vec![13.0, 15.0]; 3],
            stream_names: vec!["a".into(), "b".into()],
            per_stream_log_liks: vec![vec![vec![-1.0, -1.0], vec![-1.0, -1.0]]; 3],
            per_stream_samples: vec![
                vec![vec![10.0, 12.0], vec![3.0, 3.0]]; 3
            ],
        };
        // Step 0: both present. Step 1: stream b is a hole. Step 2: all holes.
        let per_stream_observed = vec![
            vec![11.0, 3.0],
            vec![11.0, f64::NAN],
            vec![f64::NAN, f64::NAN],
        ];
        let y_obs = vec![14.0, 11.0, 0.0];
        let trace = build_trace(&recorded, &y_obs, &per_stream_observed,
                                &[100.0, 100.0, 100.0], 0, 7);

        assert_eq!(trace.steps.len(), 2, "the all-hole step is omitted");
        // Hole-free step: recorded joint reused verbatim.
        assert_eq!(trace.steps[0].y_pred_samples, vec![13.0, 15.0]);
        assert_eq!(trace.steps[0].per_stream.len(), 2);
        // Partial-hole step: joint = present stream (a) only.
        assert_eq!(trace.steps[1].y_pred_samples, vec![10.0, 12.0],
            "joint predictive covers the present streams only");
        assert_eq!(trace.steps[1].per_stream.len(), 1,
            "the hole stream has no per-stream score");
        assert_eq!(trace.steps[1].per_stream[0].stream, "a");
        // And the joint CRPS at that step is against the present-only sum.
        let expect = crps_sample(&[10.0, 12.0], 11.0);
        assert!((trace.steps[1].crps - expect).abs() < 1e-12);
    }
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn old_prequential_json_without_conditioning_reads_as_in_sample() {
        // #295 back-compat: a pre-#295 trace has no `conditioning` field. It must
        // still deserialize (serde default), reading as the value it factually
        // is — in-sample — never failing to parse.
        let old = r#"{
            "schema_version": 2,
            "t0": 0,
            "provenance": "plug_in",
            "steps": [],
            "warnings": []
        }"#;
        let t: PrequentialTrace = serde_json::from_str(old).expect("old json must still parse");
        assert_eq!(t.conditioning, Conditioning::InSample);
        assert_eq!(t.provenance, Provenance::PlugIn);
        assert_eq!(t.pit_randomization_seed, None,
            "a pre-gh#629 trace has no recorded PIT seed");
    }

    #[test]
    fn plug_in_in_sample_trace_is_flagged_optimistic() {
        // The default builder output (plug-in + in-sample) must carry a caveat
        // on both axes — the #295 "never silently over-read" guarantee.
        let t = PrequentialTrace {
            schema_version: 2, t0: 0,
            provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![], warnings: vec![],
            pit_randomization_seed: None,
        };
        let caveat = t.optimism_caveat().expect("plug-in + in-sample must be flagged");
        assert!(caveat.contains("plug-in"), "names the treatment optimism: {caveat}");
        assert!(caveat.contains("in-sample"), "names the conditioning optimism: {caveat}");
        // The in-sample optimism is not a constant offset that cancels when two
        // models are differenced: it grows with the number of parameters θ was
        // free to tune, so Δelpd tilts toward the more flexible model. A reader
        // told only "optimistic in absolute level" would reasonably assume the
        // level cancels in a difference — which is the quantity `compare`
        // renders.
        assert!(caveat.contains("does not cancel"),
            "says the optimism does not cancel in a difference: {caveat}");
        assert!(caveat.contains("more flexible model"),
            "and which way the difference is biased: {caveat}");
    }

    #[test]
    fn conditioning_serializes_snake_case() {
        assert_eq!(serde_json::to_string(&Conditioning::InSample).unwrap(), r#""in_sample""#);
    }

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
    fn randomized_pit_splits_the_atom() {
        // u = P̂(X < y) + v·P̂(X = y). With samples [1, 2, 2, 3] and y = 2:
        // P̂(X < 2) = 1/4, P̂(X = 2) = 1/2. v = 0 gives the lower CDF limit,
        // v = 1 the naive P̂(X ≤ y) (the old, tie-biased value), v = 0.5 the
        // CGH nonrandomized midpoint.
        let s = [1.0, 2.0, 2.0, 3.0];
        assert!(approx_eq(pit_sample_randomized(&s, 2.0, 0.0), 0.25, 1e-12));
        assert!(approx_eq(pit_sample_randomized(&s, 2.0, 0.5), 0.50, 1e-12));
        assert!(approx_eq(pit_sample_randomized(&s, 2.0, 1.0), 0.75, 1e-12));
        // No tie: v is irrelevant and u is the empirical CDF.
        assert!(approx_eq(pit_sample_randomized(&s, 2.5, 0.0), 0.75, 1e-12));
        assert!(approx_eq(pit_sample_randomized(&s, 2.5, 0.9), 0.75, 1e-12));
        // Empty predictive stays NaN.
        assert!(pit_sample_randomized(&[], 1.0, 0.5).is_nan());
    }

    #[test]
    fn randomized_pit_is_uniform_on_counts_where_naive_is_biased() {
        // The gh#629 defect made observable: draw y and the predictive
        // samples from the SAME count distribution (Poisson(3)), so the
        // forecast is calibrated by construction. The randomized PIT must
        // be uniform; the naive P̂(X ≤ y) (= v = 1) must sit visibly above
        // 0.5 on average, by ~half the mean atom mass Σ p(k)²/2 ≈ 0.06.
        let mut rng = crate::rng::StatefulRng::new_stream(42, 0);
        let n_rep = 2000;
        let s_size = 400;
        let (mut sum_rand, mut sum_naive, mut below_q1) = (0.0, 0.0, 0usize);
        for _ in 0..n_rep {
            let samples: Vec<f64> =
                (0..s_size).map(|_| rng.poisson(3.0) as f64).collect();
            let y = rng.poisson(3.0) as f64;
            let u = pit_sample_randomized(&samples, y, rng.uniform());
            sum_rand += u;
            sum_naive += pit_sample_randomized(&samples, y, 1.0);
            if u < 0.25 { below_q1 += 1; }
        }
        let mean_rand = sum_rand / n_rep as f64;
        let mean_naive = sum_naive / n_rep as f64;
        assert!(approx_eq(mean_rand, 0.5, 0.02),
            "randomized PIT mean should be 0.5, got {mean_rand}");
        assert!(mean_naive - 0.5 > 0.03,
            "naive P(X<=y) PIT should be biased above 0.5 at atoms, got {mean_naive}");
        let frac_q1 = below_q1 as f64 / n_rep as f64;
        assert!(approx_eq(frac_q1, 0.25, 0.03),
            "randomized PIT should put ~25% of mass below 0.25, got {frac_q1}");
    }

    #[test]
    fn build_trace_pit_seed_is_recorded_and_reproducible() {
        // Same seed ⇒ identical PIT draws; a different seed moves the PIT at
        // a tied step; and the seed used is recorded on the trace.
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0],
            log_liks: vec![vec![-1.0; 4]],
            y_pred_samples: vec![vec![1.0, 2.0, 2.0, 3.0]],
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]],
            per_stream_samples: vec![vec![vec![1.0, 2.0, 2.0, 3.0]]],
        };
        let y_obs = vec![2.0];  // ties with two samples → v matters
        let per_stream_observed = vec![vec![2.0]];
        let ess = vec![100.0];
        let a = build_trace(&recorded, &y_obs, &per_stream_observed, &ess, 0, 11);
        let b = build_trace(&recorded, &y_obs, &per_stream_observed, &ess, 0, 11);
        let c = build_trace(&recorded, &y_obs, &per_stream_observed, &ess, 0, 12);
        assert_eq!(a.pit_randomization_seed, Some(11));
        assert_eq!(a.steps[0].pit, b.steps[0].pit, "same seed must reproduce");
        assert_ne!(a.steps[0].pit, c.steps[0].pit,
            "a different seed must move the PIT at a tied step");
        // The randomized PIT stays inside the atom's interval.
        assert!(a.steps[0].pit >= 0.25 && a.steps[0].pit <= 0.75);
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
                interval: PredInterval::default(),
                per_stream: vec![],
            }
        }).collect();
        let trace = PrequentialTrace {
            schema_version: 1, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample, steps, warnings: vec![],
            pit_randomization_seed: None,
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
            // Single-stream recorded: the stream's per-particle log-liks and
            // samples ARE the joint ones (one stream ⇒ joint = the stream).
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![
                vec![vec![-1.0, -2.0, -0.5, -3.0]],
                vec![vec![-0.1, -0.2, -0.3, -0.4]],
            ],
            per_stream_samples: vec![
                vec![vec![1.0, 2.0, 3.0, 4.0]],
                vec![vec![5.5, 5.0, 4.5, 6.0]],
            ],
        };
        let y_obs = vec![2.5, 5.2];
        let per_stream_observed = vec![vec![2.5], vec![5.2]];
        let ess = vec![100.0, 4.0];  // second step below threshold

        let trace = build_trace(&recorded, &y_obs, &per_stream_observed, &ess, 0, 7);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 0);

        // Step 0: log_score = logsumexp(log_liks) - log N
        let expected_ls0 = super::super::types::log_sum_exp(&recorded.log_liks[0])
            - (4.0_f64).ln();
        assert!(approx_eq(trace.steps[0].log_score, expected_ls0, 1e-12));
        // CRPS/PIT agree with kernels.
        assert!(approx_eq(trace.steps[0].crps,
            crps_sample(&recorded.y_pred_samples[0], 2.5), 1e-12));
        // y = 2.5 ties no sample, so the PIT is v-independent and must match
        // the kernel at any v.
        assert!(approx_eq(trace.steps[0].pit,
            pit_sample_randomized(&recorded.y_pred_samples[0], 2.5, 0.0), 1e-12));

        // ESS warning fires for the low-ess second step.
        assert_eq!(trace.warnings.len(), 1);
        matches!(trace.warnings[0], PrequentialWarning::EssCollapse { step_count: 1, .. });

        // gh#269: one stream, so the single per-stream score equals the joint
        // (same per-particle log-liks + samples + observed value).
        assert_eq!(trace.schema_version, 2);
        assert_eq!(trace.steps[0].per_stream.len(), 1);
        assert_eq!(trace.steps[0].per_stream[0].stream, "s0");
        assert!(approx_eq(trace.steps[0].per_stream[0].log_score,
            trace.steps[0].log_score, 1e-12));
        assert!(approx_eq(trace.steps[0].per_stream[0].crps,
            trace.steps[0].crps, 1e-12));
        assert!(approx_eq(trace.steps[0].per_stream[0].pit,
            trace.steps[0].pit, 1e-12));
    }

    #[test]
    fn build_trace_respects_t0_skip() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0, 3.0],
            log_liks: vec![vec![-1.0; 4]; 3],
            y_pred_samples: vec![vec![0.5, 1.0, 1.5, 2.0]; 3],
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]; 3],
            per_stream_samples: vec![vec![vec![0.5, 1.0, 1.5, 2.0]]; 3],
        };
        let y_obs = vec![1.25; 3];
        let per_stream_observed = vec![vec![1.25]; 3];
        let ess = vec![100.0; 3];

        let trace = build_trace(&recorded, &y_obs, &per_stream_observed, &ess, 1, 7);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 1);
        assert_eq!(trace.steps[0].t, 2.0);
    }

    #[test]
    fn pit_histogram_bins_sum_to_n() {
        let steps: Vec<PrequentialStep> = (0..50).map(|i| PrequentialStep {
            t: i as f64, y_obs: 0.0, y_pred_samples: vec![],
            log_score: 0.0, crps: 0.0, pit: (i as f64) / 50.0, ess: 0.0,
            interval: PredInterval::default(),
            per_stream: vec![],
        }).collect();
        let trace = PrequentialTrace {
            schema_version: 1, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample, steps, warnings: vec![],
            pit_randomization_seed: None,
        };
        let hist = trace.pit_histogram(10);
        assert_eq!(hist.iter().sum::<usize>(), 50);
    }

    #[test]
    fn pred_interval_quantiles_are_monotone_and_correct() {
        // numpy/R type-7 quantiles of 0..=100 (n=101): q = value at index 100*q.
        let samples: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let iv = PredInterval::from_samples(&samples);
        assert!(approx_eq(iv.q05, 5.0, 1e-9), "q05={}", iv.q05);
        assert!(approx_eq(iv.q25, 25.0, 1e-9));
        assert!(approx_eq(iv.q50, 50.0, 1e-9));
        assert!(approx_eq(iv.q75, 75.0, 1e-9));
        assert!(approx_eq(iv.q95, 95.0, 1e-9), "q95={}", iv.q95);
        // Monotone band (the plot-ribbon invariant).
        assert!(iv.q05 <= iv.q25 && iv.q25 <= iv.q50 && iv.q50 <= iv.q75 && iv.q75 <= iv.q95);
        // NaN-safe: non-finite entries (not-scheduled stream) are dropped.
        let with_nan = vec![f64::NAN, 1.0, 2.0, 3.0, f64::NAN];
        let iv2 = PredInterval::from_samples(&with_nan);
        assert!(iv2.q50.is_finite() && (1.0..=3.0).contains(&iv2.q50));
        // All-absent ⇒ all-NaN (no fictitious zeros).
        assert!(PredInterval::from_samples(&[f64::NAN]).q50.is_nan());
    }
}
