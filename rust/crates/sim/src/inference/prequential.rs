//! Prequential (one-step-ahead out-of-sample) evaluation.
//!
//! See `docs/dev/proposals/2026-04-20-prequential-evaluation.md` for
//! the full design. This module implements Part I: the
//! `PrequentialTrace` struct, proper-scoring-rule kernels (log
//! score, CRPS, PIT), and the summary statistics (elpd, mean CRPS,
//! PIT coverage) callers reach for.
//!
//! PF-side sample emission (each stream's per-particle predictive draws) is
//! wired in `particle_filter.rs`; this module is pure post-processing
//! once the samples are available.
//!
//! Scope: plug-in predictive only (provenance = `PlugIn`). LFO-PSIS,
//! fully-Bayesian, and pseudo-posterior variants are Part II.

use serde::{Serialize, Deserialize};

pub use super::multi_stream_obs::Coverage;

/// Provenance of the predictive used to compute scores — the
/// parameter-treatment optimism axis (#295), orthogonal to
/// [`Conditioning`]. Externally tagged like `Conditioning`, so `plug_in`
/// stays the bare string existing traces wrote and the struct variant
/// nests under its tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Point-estimate (MLE / posterior mean) plug-in predictive.
    /// Proper only when θ is assumed known; overconfident at small n.
    PlugIn,
    /// Mixture over `n_draws` thinned posterior draws (§3.6 of the
    /// 2026-08-29 proposal): per-step predictive *densities* averaged
    /// over the draws (`logsumexp_m − log M` — averaging per-draw log
    /// scores instead is a different, strictly lower number by Jensen),
    /// and predictive samples pooled across the draws' filters, so
    /// intervals widen to carry parameter uncertainty.
    Posterior { n_draws: usize },
}

/// How the parameters were conditioned on the data when scoring — the **second**
/// optimism axis (#295), orthogonal to [`Provenance`] (the plug-in-vs-posterior
/// parameter treatment).
///
/// Externally tagged (serde's default) — deliberately NOT `tag = "kind"`:
/// v2 traces wrote `"conditioning": "in_sample"` as a bare string
/// (asserted by `conditioning_serializes_snake_case`), and internal
/// tagging would fail to parse a present field of that shape. Externally
/// tagged, `InSample` stays the bare string and the struct variants
/// serialize as `{"hold_out_tail": {...}}`, so v1 (absent field, serde
/// default) and v2 (bare string) traces both keep reading.
///
/// The `Forecast` (no assimilation past the origin) and `Lfo` variants
/// are Stage-5 / follow-up additions (2026-08-29 proposal §3.2); they
/// append here without a schema migration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Conditioning {
    /// θ fit to all of `y_{1:T}` → one-step-ahead in `y` but not in `θ` —
    /// optimistic in absolute level, and biased toward the more flexible
    /// model in differences.
    #[default]
    InSample,
    /// θ from a fit sealed at `train_end` (gh#585): every scored step
    /// satisfies `t > train_end`, and the filter assimilates held-out
    /// observations as it scores them (one-step-ahead mode, §3.7.1).
    /// Honest in θ — the only leak channel the stamp certifies.
    /// `theta_source` names the sealed fit the parameters came from.
    HoldOutTail { train_end: f64, theta_source: String },
}

/// One stream's (district's) score at a single step (gh#269).
///
/// The step's joint log score is the log of the joint predictive density
/// of every stream observed there; this is each stream's own, so a national
/// fit can be diagnosed district-by-district. It is also the only home of
/// accuracy and calibration: CRPS, PIT and the predictive interval are
/// computed here, against the stream's own observed value and predictive
/// sample, and nowhere across streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamScore {
    /// Stream name (the bound observation block / district).
    pub stream: String,
    /// This stream's observed value at the step.
    pub y_obs: f64,
    /// Per-particle predictive draws for this stream.
    /// Cleared under `--no-save-samples`; the scalar scores remain valid.
    pub y_pred_samples: Vec<f64>,
    /// log p̂(y^stream | y_{1:t}) for this stream.
    pub log_score: f64,
    /// Continuous Ranked Probability Score, in the units of `y_obs` — the
    /// fair sample estimator (`crps_sample_fair`; Ferro 2014), unbiased at
    /// finite ensemble size.
    pub crps: f64,
    /// Randomized probability integral transform
    /// u = P̂(X < y_obs) + v·P̂(X = y_obs), v ~ Uniform(0, 1)
    /// (`pit_sample_randomized`; seed in the trace's
    /// `pit_randomization_seed`). Uniform(0, 1) under correct calibration,
    /// including for count predictives.
    pub pit: f64,
    /// This stream's plot-ready predictive interval (median + 50%/90% bands)
    /// from its predictive samples. The canonical per-district forecast band.
    /// `#[serde(default)]`: v1 traces (no interval) still deserialize.
    #[serde(default)]
    pub interval: PredInterval,
    /// What `y_obs` was accumulated over — a declared or inferred period, or
    /// an instant — so two traces can be checked to have scored the same
    /// QUANTITY at this step and not merely at the same time (gh#833;
    /// `compare`'s window gate). `#[serde(default)]`: a trace written before
    /// schema 4 reads as `Unrecorded`, which the gate treats as unverifiable
    /// rather than as agreeing.
    #[serde(default)]
    pub coverage: Coverage,
}

/// A single step's record: the joint log score, the filter's ESS, and each
/// scored stream's own scores.
///
/// Stored per-step so downstream summaries (total elpd with paired SE,
/// per-stream CRPS and PIT coverage, quantile plots) don't re-run the PF.
///
/// The step has no observed value, predictive sample, CRPS, PIT or interval
/// of its own: those are per stream ([`StreamScore`]). Summed across
/// streams they would describe the sum of a case count, a death count and a
/// count derived from a proportion, which no one observed. A trace written
/// with those step-level fields (schemas 1–4) still reads; serde ignores the
/// keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrequentialStep {
    /// Time index of the assimilated observation.
    pub t: f64,
    /// log p̂(y_t | y_{1:t-1}) = log Σ w^(s) p(y_t | x^(s)): the log of the
    /// joint one-step-ahead predictive density of every stream observed at
    /// `t`. Summed over steps it is the elpd.
    pub log_score: f64,
    /// Effective sample size of the filter at this step.
    pub ess: f64,
    /// Per-stream (per-district) score breakdown (gh#269). One entry per
    /// SCHEDULED, NON-HOLE stream at this step; `Σ per_stream.log_score`
    /// need not equal `log_score` (log-of-sum vs sum-of-logs).
    /// `#[serde(default)]`: v1 traces (no per-stream field) still
    /// deserialize, with no stream to summarize.
    #[serde(default)]
    pub per_stream: Vec<StreamScore>,
}

/// Warning attached to a prequential trace — things a reader needs
/// to see before interpreting the summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PrequentialWarning {
    /// ESS dropped below `threshold` at `step_count` scored steps.
    /// `threshold` is the absolute ESS cut applied, derived as
    /// [`ESS_COLLAPSE_FRACTION`] of the particle count.
    EssCollapse { step_count: usize, threshold: f64 },
    /// Scoring starts at the prior: `t0 = 0` and no warm-up precedes the
    /// first observation — its window opens at the run's start — so the
    /// first scored one-step-ahead predictive is issued from the
    /// initial-state distribution with no data assimilated. It scores the
    /// initializer as much as the model, and can dominate a short trace's
    /// elpd. A stream whose first declared period opens after `t_start`
    /// (the warm-up simulated but not scored) does not trip this.
    StartsAtPrior,
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
    /// The scoring boundary as a model TIME (gh#585, Stage 3.2): scored
    /// steps all satisfy `t > score_from`; observations at or before it
    /// were assimilated (the filter reweighted on them) but not scored.
    /// The human-readable time-axis twin of the index `t0` — `t0` remains
    /// the mechanism, this records where the boundary sits on the time
    /// axis. `None` when scoring was not windowed by time (`t0` may still
    /// be nonzero, e.g. IC-free's first-obs pin). `#[serde(default)]` so
    /// older traces deserialize; omitted when absent so a `null` in the
    /// artifact stays what it always meant — a non-finite score leak.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_from: Option<f64>,
    /// Seed of the randomized-PIT uniform draws (`pit_sample_randomized`):
    /// one `v` per scored stream value, drawn from
    /// `StatefulRng::new_stream(seed, PIT_RNG_STREAM)` in step order and,
    /// within a step, in stream order over the present streams. Recording it
    /// makes the trace's PIT values reproducible. (A trace that carries a
    /// step-level `pit` drew one more `v` per step, for it, before the
    /// step's streams.) `None` on traces written before the randomized
    /// PIT (gh#629); `#[serde(default)]` so those still deserialize, and
    /// omitted when absent (see `score_from`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pit_randomization_seed: Option<u64>,
    /// How this trace was computed. `None` on traces written before schema 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<Derivation>,
    /// The leaves a trace re-scored by [`project_onto_stream`] is scored on:
    /// each step's `log_score` is then that leaf's own, not the joint score
    /// over every leaf the fit observed. `None` for a trace as a filter wrote
    /// it. Recorded so a projected trace, if ever serialized, cannot be read
    /// back as a joint one. Omitted when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scored_streams: Option<Vec<String>>,
}

/// The settings that move a trace's scores, recorded by whoever ran the
/// filter (schema 5).
///
/// The log of a particle-filter likelihood estimate is biased downward, by
/// more at fewer particles, and log-mean-exp over replicates reduces that
/// bias; so two traces scored at different `particles` or `replicates` differ
/// partly by the filters. `compare` reads this block to refuse such a pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Derivation {
    /// Particles per filter pass.
    pub particles: usize,
    /// Filter replicates combined into the trace; 1 for a single pass.
    pub replicates: usize,
    /// The integration step the filter ran at.
    pub dt: f64,
    /// The filter seed; replicate `r` ran at `seed + r`.
    pub seed: u64,
    /// Where the parameters the trace was scored at came from.
    pub theta: ThetaSource,
    /// MCMC chains dropped from the posterior before θ was taken (chain
    /// numbers counted from 0, as `draws.tsv` writes them); empty when none.
    pub excluded_chains: Vec<usize>,
    /// Interventions the run switched on by name (`--enable`, or the fit
    /// config's selection).
    pub enabled: Vec<String>,
    /// Interventions the run switched off by name.
    pub disabled: Vec<String>,
    /// Each replicate's total elpd (the sum of its steps' `log_score`), in
    /// replicate order; `replicates` long, one element for a single pass.
    /// Stored so a paired filter-noise estimate can be computed later from
    /// the totals without another schema change.
    pub replicate_elpd: Vec<f64>,
    /// For each stream scored in `per_stream`, each replicate's sum of that
    /// stream's `log_score`, in the same order. These need not sum across
    /// streams to `replicate_elpd`: a step's joint score is the log of a
    /// joint predictive, not a sum of the per-stream logs.
    pub replicate_elpd_by_stream: std::collections::BTreeMap<String, Vec<f64>>,
    /// The filter-noise Monte-Carlo SE of the elpd, from `replicate_elpd`
    /// (the log-mean-exp delta-method SE); `None` for a single pass. Omitted
    /// when absent, so a `null` in a trace keeps its one meaning — a
    /// non-finite value (see `score_from`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mc_se_elpd: Option<f64>,
    /// Waivers applied to this one trace when it was derived, so a reader of
    /// the trace alone can see them. Empty (and omitted) when none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<TraceOverride>,
}

impl Derivation {
    /// The block for one filter pass over `trace`: one replicate, its own
    /// totals, no filter-noise SE. Both single-pass producers
    /// (`camdl pfilter --save-prequential` and `fit run`'s pfilter method)
    /// build it here, so they cannot drift apart.
    pub fn single_pass(
        trace: &PrequentialTrace,
        particles: usize,
        dt: f64,
        seed: u64,
        theta: ThetaSource,
        enabled: Vec<String>,
        disabled: Vec<String>,
    ) -> Self {
        Derivation {
            particles,
            replicates: 1,
            dt,
            seed,
            theta,
            excluded_chains: Vec::new(),
            enabled,
            disabled,
            replicate_elpd: vec![trace.elpd()],
            replicate_elpd_by_stream: trace.elpd_by_stream().into_iter()
                .map(|(k, v)| (k, vec![v]))
                .collect(),
            mc_se_elpd: None,
            overrides: Vec::new(),
        }
    }
}

/// A waiver applied to one trace when it was derived (schema 5). Waivers
/// that only make sense for a pair of traces are not recorded here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceOverride {
    /// `compare --in-sample` scored a fit that declares a holdout, so the
    /// trace is in-sample although the fit sealed a training window.
    InSampleForced,
    /// Derived under `compare --allow-changed-data`: these streams' training
    /// files no longer matched the hashes the fit recorded.
    ChangedData { streams: Vec<String> },
}

/// Where a trace's parameters came from. Externally tagged like
/// [`Provenance`]: unit variants are bare strings, struct variants nest under
/// their tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThetaSource {
    /// Values handed to the filter by its caller — a params file (or
    /// `--param`) given to `camdl pfilter`, or the point a fit config's
    /// `pfilter` method evaluates at — not taken from a fit's posterior or
    /// optimizer result.
    Supplied,
    /// The mean over `n_draws` retained posterior draws.
    PosteriorMean { n_draws: usize },
    /// An optimizer fit's winning point.
    OptimizerWinner,
    /// An evenly thinned mixture of `n_used` of the `n_available` retained
    /// posterior draws.
    PosteriorDraws { n_used: usize, n_available: usize },
}

impl PrequentialTrace {
    /// Total expected log predictive density (elpd_preq).
    pub fn elpd(&self) -> f64 {
        self.steps.iter().map(|s| s.log_score).sum()
    }

    /// Each stream's summed per-stream `log_score` over the scored steps.
    pub fn elpd_by_stream(&self) -> std::collections::BTreeMap<String, f64> {
        let mut out = std::collections::BTreeMap::new();
        for step in &self.steps {
            for ss in &step.per_stream {
                *out.entry(ss.stream.clone()).or_insert(0.0) += ss.log_score;
            }
        }
        out
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

    /// The streams the trace scored, in the order they first appear. Empty
    /// for a trace with no per-stream breakdown (schema 1), which therefore
    /// has no CRPS or calibration figure to report.
    pub fn streams(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for ss in self.steps.iter().flat_map(|s| &s.per_stream) {
            if !out.contains(&ss.stream.as_str()) {
                out.push(&ss.stream);
            }
        }
        out
    }

    /// Each step that scored `stream`, as its time and that stream's score.
    pub fn stream_scores<'a>(&'a self, stream: &'a str)
        -> impl Iterator<Item = (f64, &'a StreamScore)> + 'a
    {
        self.steps.iter().filter_map(move |s| {
            s.per_stream.iter().find(|ss| ss.stream == stream).map(|ss| (s.t, ss))
        })
    }

    /// `stream`'s mean CRPS over the steps that scored it.
    ///
    /// Per stream only: a CRPS is in the units of the observed value, and
    /// across streams there is no one observed value — the sum of a case
    /// count, a death count and a count derived from a proportion is not
    /// something anyone observed.
    pub fn mean_crps(&self, stream: &str) -> StreamMean {
        StreamMean::of(self.stream_scores(stream).map(|(_, ss)| ss.crps))
    }

    /// The share of `stream`'s observations whose PIT fell inside the central
    /// `level` predictive interval (level ∈ (0, 1)), over the steps that
    /// scored it.
    ///
    /// Nominal coverage = `level`. Substantial deviation indicates
    /// miscalibration (plug-in overconfidence is the typical failure mode).
    /// A non-finite PIT (an empty predictive sample) is left out and counted
    /// in `n_nonfinite`, not read as an observation outside the interval.
    pub fn pit_coverage(&self, stream: &str, level: f64) -> StreamMean {
        let (lo, hi) = (0.5 - level / 2.0, 0.5 + level / 2.0);
        StreamMean::of(self.stream_scores(stream).map(|(_, ss)| {
            if ss.pit.is_finite() { f64::from(u8::from(ss.pit >= lo && ss.pit <= hi)) }
            else { f64::NAN }
        }))
    }

    /// Number of scored steps (T - t0).
    pub fn n_scored(&self) -> usize { self.steps.len() }

    /// The `k` worst-scored steps, worst first: the observations the
    /// predictive found most surprising, ranked by the joint `log_score`.
    ///
    /// The elpd is a sum, and a sum hides which terms carry it. One
    /// observation the model cannot reach — a count floored to zero on a
    /// re-issue, a decimal-shifted value — contributes a log score tens of
    /// nats below its neighbours and moves the total by itself, while every
    /// aggregate the trace reports (elpd, mean CRPS, PIT coverage) reads as
    /// ordinary. This is the lookup the aggregates cannot do: which rows of
    /// the data the score is about.
    ///
    /// A non-finite log score (`-inf`: no particle could produce the value;
    /// `NaN`: a scoring failure) is the worst outcome available and sorts
    /// first, `-inf` before `NaN` so a real "unreachable" precedes a defect.
    /// Ties break toward the earlier step. `k` larger than the trace
    /// returns every step.
    pub fn worst_scored(&self, k: usize) -> Vec<&PrequentialStep> {
        let mut idx: Vec<usize> = (0..self.steps.len()).collect();
        idx.sort_by(|&a, &b| {
            worse_first(self.steps[a].log_score, self.steps[b].log_score).then(a.cmp(&b))
        });
        idx.into_iter().take(k).map(|i| &self.steps[i]).collect()
    }
}

/// The trace re-scored on one leaf.
///
/// Each step that scored `leaf` keeps its time and ESS, takes the leaf's own
/// `log_score` as its score, and keeps only the leaf's `per_stream` entry,
/// copied unchanged; a step that did not score the leaf is dropped. Nothing is
/// recomputed from samples and no random number is drawn, so the projected
/// score is exact: the fit's one-step-ahead predictive density for `leaf`
/// given everything its filter had assimilated, every other leaf included.
///
/// What describes the filter pass rather than the scored set is restated for
/// the surviving steps: `StartsAtPrior` stays only if the first step survives,
/// `EssCollapse` is recounted at its recorded threshold, and the `derivation`
/// block's replicate totals and filter-noise SE become the leaf's
/// (`replicate_elpd_by_stream[leaf]`, with the SE recomputed from them by
/// [`super::types::logmeanexp_with_se`]). `scored_streams` records `[leaf]`.
///
/// A sum of these per-leaf scores over several leaves is not the joint score
/// of those leaves (each is a marginal with the other leaves integrated out),
/// so there is deliberately no projection onto a set of leaves.
///
/// Refuses, naming the leaf, a trace with no scored step, one with no
/// per-stream breakdown (schema 1), one that never scored `leaf`, and a
/// derivation block with no replicate totals for `leaf`.
pub fn project_onto_stream(
    trace: &PrequentialTrace,
    leaf: &str,
) -> Result<PrequentialTrace, String> {
    if trace.steps.is_empty() {
        return Err(format!("cannot score '{leaf}' alone: the trace has no scored step"));
    }
    let scored = trace.streams();
    if scored.is_empty() {
        return Err(format!(
            "cannot score '{leaf}' alone: the trace has no per-stream breakdown (a \
             schema 1 prequential.json), so only its joint score exists. Re-derive it."));
    }
    let mut steps: Vec<PrequentialStep> = Vec::new();
    let mut first_survives = false;
    for (k, step) in trace.steps.iter().enumerate() {
        let Some(entry) = step.per_stream.iter().find(|s| s.stream == leaf) else { continue };
        first_survives |= k == 0;
        steps.push(PrequentialStep {
            t: step.t,
            log_score: entry.log_score,
            ess: step.ess,
            per_stream: vec![entry.clone()],
        });
    }
    if steps.is_empty() {
        return Err(format!("the trace does not score '{leaf}'; it scores {}",
            scored.join(", ")));
    }
    let warnings = trace.warnings.iter().filter_map(|w| match w {
        PrequentialWarning::StartsAtPrior => first_survives.then(|| w.clone()),
        PrequentialWarning::EssCollapse { threshold, .. } => {
            let step_count = steps.iter().filter(|s| s.ess < *threshold).count();
            (step_count > 0).then_some(PrequentialWarning::EssCollapse {
                step_count, threshold: *threshold,
            })
        }
        PrequentialWarning::SamplesNotSaved => Some(w.clone()),
    }).collect();
    let derivation = match &trace.derivation {
        None => None,
        Some(d) => {
            let totals = d.replicate_elpd_by_stream.get(leaf).ok_or_else(|| format!(
                "cannot score '{leaf}' alone: the trace's derivation block records no \
                 replicate totals for it, so its filter-noise SE cannot be restated"))?;
            Some(Derivation {
                replicate_elpd: totals.clone(),
                replicate_elpd_by_stream: [(leaf.to_string(), totals.clone())]
                    .into_iter().collect(),
                mc_se_elpd: (totals.len() >= 2)
                    .then(|| super::types::logmeanexp_with_se(totals).1),
                ..d.clone()
            })
        }
    };
    Ok(PrequentialTrace {
        schema_version: trace.schema_version,
        t0: trace.t0,
        provenance: trace.provenance,
        conditioning: trace.conditioning.clone(),
        steps,
        warnings,
        score_from: trace.score_from,
        pit_randomization_seed: trace.pit_randomization_seed,
        derivation,
        scored_streams: Some(vec![leaf.to_string()]),
    })
}

/// One stream's average of a per-observation figure (its CRPS, or whether
/// its PIT fell inside an interval) over the steps that scored the stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamMean {
    /// Steps at which the stream was scored.
    pub n: usize,
    /// Of those, the values that were not finite, which `value` leaves out.
    pub n_nonfinite: usize,
    /// The mean over the finite values; NaN when there are none.
    pub value: f64,
}

impl StreamMean {
    fn of(values: impl Iterator<Item = f64>) -> Self {
        let (mut n, mut n_nonfinite, mut sum) = (0usize, 0usize, 0.0);
        for v in values {
            n += 1;
            if v.is_finite() { sum += v; } else { n_nonfinite += 1; }
        }
        let finite = n - n_nonfinite;
        StreamMean {
            n,
            n_nonfinite,
            value: if finite == 0 { f64::NAN } else { sum / finite as f64 },
        }
    }
}

/// Order two log scores worst first: `-inf`, then `NaN`, then ascending.
/// The one ordering behind [`PrequentialTrace::worst_scored`] and
/// [`PrequentialStep::worst_stream`], so a step and its streams cannot
/// rank a non-finite score differently.
fn worse_first(a: f64, b: f64) -> std::cmp::Ordering {
    let key = |v: f64| -> (u8, f64) {
        if v == f64::NEG_INFINITY {
            (0, 0.0)
        } else if v.is_nan() {
            (1, 0.0)
        } else {
            (2, v)
        }
    };
    let (ka, va) = key(a);
    let (kb, vb) = key(b);
    ka.cmp(&kb).then(va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal))
}

impl PrequentialStep {
    /// The stream this step scored worst, by its own log score — `None`
    /// when the step carries no per-stream breakdown (a single-stream
    /// model, or a v1 trace). Non-finite scores sort first as in
    /// [`PrequentialTrace::worst_scored`]; ties break toward the first
    /// stream. A step has no observed value of its own, so this is the row a
    /// reader can actually look up in the data.
    pub fn worst_stream(&self) -> Option<&StreamScore> {
        self.per_stream.iter().min_by(|a, b| worse_first(a.log_score, b.log_score))
    }
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

/// Sample-based CRPS: the *plain empirical-CDF (edf)* estimator, via the
/// Hersbach / Laio–Tamea sorted-sample identity:
///
///   ĈRPS = (2/S²) Σ (x_(s) − y) · [S · 1{y < x_(s)} − (s − 1/2)]
///
/// where x_(s) are the samples sorted ascending and s is 1-indexed.
/// O(S log S) via the sort. Equivalent to the naive O(S²) form
///   (1/S)Σ|x^(i) − y| − (1/(2S²))ΣΣ|x^(i) − x^(j)|.
///
/// Estimator identity, pinned (gh#628): this is exactly R
/// `scoringRules::crps_sample(method = "edf")` — the CRPS of the
/// empirical distribution of the ensemble, which is biased upward for
/// the underlying predictive's CRPS at finite S (by `E|X−X'|/(2S)`).
/// The trace builder scores with [`crps_sample_fair`] instead; this
/// form is kept as the external-oracle anchor and for explicit
/// finite-ensemble CDF use.
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

/// Sample-based CRPS: the *fair* estimator (Ferro 2014, QJRMS 140(683);
/// pairwise form as written in Zamo & Naveau 2018),
///
///   ĈRPS_fair = (1/S) Σ_s |x_s − y|
///             − (1/(2S(S−1))) Σ_{s≠s'} |x_s − x_s'|,
///
/// which is unbiased for the underlying predictive's CRPS at finite
/// ensemble size — the plain edf form ([`crps_sample`]) carries an
/// `O(1/S)` upward bias that matters when compared traces carry unequal
/// sample counts. This is the estimator the prequential trace scores
/// with (2026-08-29 proposal, Stage 2.2).
///
/// Computed O(S log S): the pairwise sum over the ascending order
/// statistics is `Σ_{s≠s'}|x_s − x_s'| = 2 Σ_k (2k − S − 1) x_(k)`
/// (k 1-indexed). `S = 1` has no pairwise term; the degenerate
/// predictive scores `|x − y|`, as in the edf form.
pub fn crps_sample_fair(samples: &[f64], y: f64) -> f64 {
    let s = samples.len();
    if s == 0 { return f64::NAN; }
    if s == 1 { return (samples[0] - y).abs(); }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let s_f = s as f64;
    let term1 = sorted.iter().map(|x| (x - y).abs()).sum::<f64>() / s_f;
    // Σ_{i<j} (x_(j) − x_(i)) over the sorted samples.
    let gini: f64 = sorted.iter().enumerate()
        .map(|(i, &x)| ((2 * (i + 1)) as f64 - s_f - 1.0) * x)
        .sum();
    term1 - gini / (s_f * (s_f - 1.0))
}

/// Build a `PrequentialTrace` from raw PF recordings and each stream's
/// observed values.
///
/// `recorded` comes from `PFilterResult.prequential` (requires
/// `SMCConfig.record_prequential = true`). `per_stream_observed[k][s]` is
/// stream `s`'s observed value at `recorded.obs_times[k]`, non-finite where
/// the stream is not scheduled or is a hole. `ess_trace` mirrors
/// `PFilterResult.ess_trace`.
///
/// `t0` is the number of leading observations skipped (not scored).
/// Under IC-free inference the first obs is used only to pin x_0;
/// pass `t0 = 1`. Otherwise `t0 = 0`.
///
/// `pit_seed` seeds the randomized-PIT uniform draws
/// (`pit_sample_randomized`); callers pass their filter seed, and the
/// trace records it (`pit_randomization_seed`). Draws are consumed in
/// step order, one per present stream in stream order, so a trace is
/// reproducible from (seed, data).
///
/// `has_warm_up` says whether the first observation's window opens after
/// the run's start (`MultiStreamObsModel::first_observation_opens_after_run_start`).
/// When it does not and `t0 = 0`, the first scored predictive is issued
/// from the initial-state distribution with no data assimilated, and the
/// trace carries a [`PrequentialWarning::StartsAtPrior`].
///
/// Bootstrap-PF-specific assumption: pre-obs weights are uniform
/// (reset to zero at the end of the previous step), so log-score
/// reduces to `logsumexp(log_liks) − log N`. If this filter ever
/// gains auxiliary weighting, pass weighted log-score here.
pub fn build_trace(
    recorded: &super::particle_filter::PrequentialRecorded,
    per_stream_observed: &[Vec<f64>],
    per_stream_coverage: &[Vec<Option<Coverage>>],
    ess_trace: &[f64],
    t0: usize,
    pit_seed: u64,
    has_warm_up: bool,
    score_from: Option<f64>,
) -> PrequentialTrace {
    assert_eq!(recorded.obs_times.len(), per_stream_observed.len(),
        "per_stream_observed must align 1:1 with recorded obs_times");
    assert_eq!(recorded.obs_times.len(), per_stream_coverage.len(),
        "per_stream_coverage must align 1:1 with recorded obs_times");
    assert_eq!(recorded.obs_times.len(), ess_trace.len(),
        "ess_trace must align 1:1 with recorded obs_times");

    let mut steps = Vec::with_capacity(recorded.obs_times.len().saturating_sub(t0));
    let mut warnings: Vec<PrequentialWarning> = Vec::new();
    let mut ess_collapse_count = 0usize;
    let mut ess_threshold_used = 0.0_f64;
    let mut pit_rng = crate::rng::StatefulRng::new_stream(pit_seed, PIT_RNG_STREAM);

    for idx in t0..recorded.obs_times.len() {
        let log_liks = &recorded.log_liks[idx];
        let n = log_liks.len() as f64;

        // gh#636: a hole (NA) or a not-scheduled sibling carries a
        // non-finite observed value and has no term at this step; a step
        // where no stream is present carries no information and is skipped
        // entirely.
        let any_present = (0..recorded.stream_names.len())
            .any(|si| per_stream_observed[idx][si].is_finite());
        if !any_present && !recorded.stream_names.is_empty() {
            continue;
        }

        // Uniform-weight log-score (see docstring).
        let log_score = super::types::log_sum_exp(log_liks) - n.ln();
        let ess = ess_trace[idx];
        // The collapse cue scales with the swarm: an absolute floor (the old
        // 10.0) read N = 10_000 with ESS 50 — a 0.5% survival — as healthy.
        // `n` is this step's particle count, constant across steps in
        // practice, so the recorded threshold is well-defined.
        let ess_threshold = ESS_COLLAPSE_FRACTION * n;
        ess_threshold_used = ess_threshold;
        if ess < ess_threshold { ess_collapse_count += 1; }

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
                crps: crps_sample_fair(samp_s, y_s),
                pit: pit_sample_randomized(samp_s, y_s, pit_rng.uniform()),
                interval: PredInterval::from_samples(samp_s),
                // A finite observed value means the stream is scheduled here,
                // and a scheduled row always has a coverage.
                coverage: per_stream_coverage[idx][si].expect(
                    "a stream with an observed value at this step is scheduled at it"),
            });
        }

        steps.push(PrequentialStep {
            t: recorded.obs_times[idx],
            log_score, ess,
            per_stream,
        });
    }

    if ess_collapse_count > 0 {
        warnings.push(PrequentialWarning::EssCollapse {
            step_count: ess_collapse_count,
            threshold: ess_threshold_used,
        });
    }
    if t0 == 0 && !has_warm_up && !steps.is_empty() {
        warnings.push(PrequentialWarning::StartsAtPrior);
    }

    PrequentialTrace {
        // v3 (gh#585): score_from + pit_randomization_seed + the
        // Conditioning struct variants. v4 (gh#833): per-stream `coverage`.
        // v5: the `derivation` block (filled by the caller). All additions
        // serde-defaulted, so older traces still read; they read as
        // InSample/PlugIn/Unrecorded/no derivation, which is factually what
        // they are.
        schema_version: 5,
        t0,
        // This builder scores a single filter pass at one θ over the full data:
        // plug-in + in-sample. The posterior / LFO producers (#295) stamp their
        // own values.
        provenance: Provenance::PlugIn,
        conditioning: Conditioning::InSample,
        steps,
        warnings,
        score_from,
        pit_randomization_seed: Some(pit_seed),
        // The filter settings are the caller's to record (`pfilter`), not
        // this builder's: it sees the recorded swarm, not how it was run.
        derivation: None,
        // A filter scores every observed leaf jointly.
        scored_streams: None,
    }
}

/// RNG stream for the randomized-PIT draws. Disjoint from the filter's
/// per-particle streams (low indices) and `RESAMPLE_RNG_STREAM` (1<<48),
/// so passing the filter seed to `build_trace` cannot correlate the PIT
/// randomization with the filter's own draws.
pub const PIT_RNG_STREAM: u64 = 1u64 << 49;

/// A scored step's ESS below this fraction of the particle count counts
/// toward the trace's `EssCollapse` warning. A fraction, not an absolute
/// count: at N = 10_000 the old absolute floor of 10 read a 0.5%-survival
/// step (ESS 50) as healthy. 10% is an advisory "the predictive at this
/// step rests on few effective particles" cue — well above the degeneracy
/// watchdog's hard-bail floor (`degeneracy.rs`, ESS ≤ 2 sustained), and
/// below the ~50% rule-of-thumb at which resampling is merely *warranted*.
pub const ESS_COLLAPSE_FRACTION: f64 = 0.1;

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
/// reference. Their *nonrandomized* PIT is not a per-observation
/// value: they average over the randomization to obtain, for each
/// observation, the conditional CDF of `u` given `y` — a ramp across
/// the atom's interval — then aggregate those CDFs over observations
/// and difference the aggregate on a grid into a histogram. `v = 0.5`
/// is the mean of that per-observation distribution, not a drop-in
/// nonrandomized PIT: histogramming `v = 0.5` values gives a picket
/// fence of atom midpoints (on Poisson(3) there are only 15 distinct
/// ones with any mass), not a calibration plot. The randomized form
/// is chosen so each scored value is itself uniform and every
/// existing consumer (coverage, histogram) reads it unchanged; see
/// §3.3 of the 2026-08-29 honest-predictive-evaluation proposal.
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

    /// gh#636: holes are first-class. A hole stream has no score at its
    /// step, the present streams keep theirs, and an all-hole step carries
    /// no information and is omitted.
    #[test]
    fn build_trace_skips_holes_consistently() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![7.0, 14.0, 21.0],
            log_liks: vec![vec![-2.0, -2.0]; 3],
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
        let trace = build_trace(&recorded, &per_stream_observed,
                                &interval_coverage(&recorded.obs_times, 2),
                                &[100.0, 100.0, 100.0], 0, 7, true, None);

        assert_eq!(trace.steps.len(), 2, "the all-hole step is omitted");
        assert_eq!(trace.steps[0].per_stream.len(), 2);
        assert_eq!(trace.steps[1].per_stream.len(), 1,
            "the hole stream has no per-stream score");
        assert_eq!(trace.steps[1].per_stream[0].stream, "a");
        // The present stream is scored against its own observed value.
        let expect = crps_sample_fair(&[10.0, 12.0], 11.0);
        assert!((trace.steps[1].per_stream[0].crps - expect).abs() < 1e-12);
    }
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    /// Every stream an interval closing at its step time, opening at the
    /// previous one (the first at 0) — the `closing_at` shape. Enough for the
    /// tests that are not about coverage.
    fn interval_coverage(obs_times: &[f64], n_streams: usize) -> Vec<Vec<Option<Coverage>>> {
        obs_times.iter().enumerate().map(|(k, &t)| {
            let start = if k == 0 { 0.0 } else { obs_times[k - 1] };
            vec![Some(Coverage::Interval { start, stop: t }); n_streams]
        }).collect()
    }

    /// gh#833: each per-stream score records what its value was accumulated
    /// over, straight from the coverage the obs model supplies — an interval
    /// stream's period, an instant stream's `Instant` — and the record survives
    /// the JSON round trip in the tagged form `compare` reads. A score written
    /// before schema 4 carries no `coverage` key and reads as `Unrecorded`, so
    /// an old trace cannot masquerade as an agreeing one.
    #[test]
    fn build_trace_records_what_each_scored_value_covers() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![7.0, 14.0],
            log_liks: vec![vec![-2.0, -2.0]; 2],
            stream_names: vec!["inc".into(), "prev".into()],
            per_stream_log_liks: vec![vec![vec![-1.0, -1.0], vec![-1.0, -1.0]]; 2],
            per_stream_samples: vec![vec![vec![10.0, 12.0], vec![3.0, 3.0]]; 2],
        };
        // Step 1: the incidence stream is a hole (scheduled, no value).
        let per_stream_observed = vec![vec![11.0, 3.0], vec![f64::NAN, 3.0]];
        let inc0 = Coverage::Interval { start: 0.0, stop: 7.0 };
        let inc1 = Coverage::Interval { start: 7.0, stop: 14.0 };
        let coverage = vec![
            vec![Some(inc0), Some(Coverage::Instant)],
            vec![Some(inc1), Some(Coverage::Instant)],
        ];
        let trace = build_trace(&recorded, &per_stream_observed, &coverage,
                                &[100.0, 100.0], 0, 7, true, None);

        assert_eq!(trace.schema_version, 5,
            "build_trace writes the current schema (coverage arrived in 4)");
        let s0 = &trace.steps[0].per_stream;
        assert_eq!((s0[0].stream.as_str(), s0[0].coverage), ("inc", inc0));
        assert_eq!((s0[1].stream.as_str(), s0[1].coverage), ("prev", Coverage::Instant));
        let s1 = &trace.steps[1].per_stream;
        assert_eq!(s1.len(), 1, "the hole has no score and so no coverage entry");
        assert_eq!((s1[0].stream.as_str(), s1[0].coverage), ("prev", Coverage::Instant));

        let json = serde_json::to_value(&s0[0]).unwrap();
        assert_eq!(json["coverage"],
            serde_json::json!({"kind": "interval", "start": 0.0, "stop": 7.0}),
            "the tagged form on disk: {json}");
        assert_eq!(serde_json::to_value(&s0[1]).unwrap()["coverage"],
            serde_json::json!({"kind": "instant"}));

        // A pre-schema-4 record: no `coverage` key.
        let mut old = serde_json::to_value(&s0[0]).unwrap();
        old.as_object_mut().unwrap().remove("coverage");
        let back: StreamScore = serde_json::from_value(old).unwrap();
        assert_eq!(back.coverage, Coverage::Unrecorded,
            "an old trace says nothing, and must not read as agreeing");
    }

    /// Schema 5: the derivation block survives the JSON round trip, with
    /// `ThetaSource` externally tagged (unit variants bare strings, struct
    /// variants nested under their tag), and a schema-4 trace without the
    /// block still reads, as `None`.
    #[test]
    fn derivation_block_round_trips_and_an_old_trace_reads_without_it() {
        let block = Derivation {
            particles: 300,
            replicates: 5,
            dt: 0.5,
            seed: 7,
            theta: ThetaSource::PosteriorDraws { n_used: 3, n_available: 80 },
            excluded_chains: vec![1, 3],
            enabled: vec!["lockdown".into()],
            disabled: vec![],
            replicate_elpd: vec![-10.0, -10.5, -9.8, -10.1, -10.2],
            replicate_elpd_by_stream: [("cases".to_string(), vec![-4.0, -4.1, -3.9, -4.0, -4.2])]
                .into_iter().collect(),
            mc_se_elpd: Some(0.12),
            overrides: vec![TraceOverride::InSampleForced,
                            TraceOverride::ChangedData { streams: vec!["cases".into()] }],
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["theta"],
            serde_json::json!({"posterior_draws": {"n_used": 3, "n_available": 80}}));
        assert_eq!(json["excluded_chains"], serde_json::json!([1, 3]));
        assert_eq!(json["overrides"],
            serde_json::json!(["in_sample_forced", {"changed_data": {"streams": ["cases"]}}]));
        let back: Derivation = serde_json::from_value(json).unwrap();
        assert_eq!(back, block);
        for (theta, tag) in [(ThetaSource::Supplied, serde_json::json!("supplied")),
                             (ThetaSource::OptimizerWinner, serde_json::json!("optimizer_winner")),
                             (ThetaSource::PosteriorMean { n_draws: 12 },
                              serde_json::json!({"posterior_mean": {"n_draws": 12}}))] {
            assert_eq!(serde_json::to_value(theta).unwrap(), tag);
        }

        let old = r#"{
            "schema_version": 4, "t0": 0, "provenance": "plug_in",
            "conditioning": "in_sample", "steps": [], "warnings": []
        }"#;
        let t: PrequentialTrace = serde_json::from_str(old).expect("a schema-4 trace reads");
        assert_eq!(t.derivation, None, "a schema-4 trace records no derivation");
        let written = serde_json::to_value(&t).unwrap();
        assert!(written.get("derivation").is_none(),
            "an absent block is omitted, not written as null: {written}");
    }

    /// A single pass records its own totals: a one-element `replicate_elpd`
    /// equal to the trace's elpd, one element per stream equal to that
    /// stream's summed score, and no filter-noise SE.
    #[test]
    fn a_single_pass_records_its_own_totals() {
        let score = |stream: &str, ls: f64| StreamScore {
            stream: stream.into(), y_obs: 1.0, y_pred_samples: vec![], log_score: ls,
            crps: 0.0, pit: 0.5, interval: PredInterval::default(), coverage: Coverage::Instant,
        };
        let step = |t: f64, ls: f64, per: Vec<StreamScore>| PrequentialStep {
            t, log_score: ls, ess: 10.0, per_stream: per,
        };
        let trace = PrequentialTrace {
            schema_version: 5, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![
                step(7.0, -3.0, vec![score("north", -1.0), score("south", -1.5)]),
                step(14.0, -2.0, vec![score("north", -0.5)]),
            ],
            warnings: vec![], score_from: None, pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        let d = Derivation::single_pass(&trace, 200, 1.0, 3, ThetaSource::Supplied,
                                        vec![], vec![]);
        assert_eq!(d.replicates, 1);
        assert_eq!(d.replicate_elpd, vec![-5.0]);
        assert_eq!(d.replicate_elpd_by_stream,
            [("north".to_string(), vec![-1.5]), ("south".to_string(), vec![-1.5])]
                .into_iter().collect());
        assert_eq!(d.mc_se_elpd, None);
        let json = serde_json::to_value(&d).unwrap();
        // Absent, not null: in a trace a null means a non-finite score leaked.
        assert!(json.get("mc_se_elpd").is_none(), "{json}");
        assert!(json.get("overrides").is_none(), "no overrides, omitted: {json}");
    }

    /// The step-level fields scored on the summed multi-stream predictive
    /// (`y_obs`, `y_pred_samples`, `crps`, `pit`, `interval`) are not part of a
    /// step. A schema-4 trace that carries them still reads, its per-stream
    /// scores intact and the summed fields ignored, and writing it back drops
    /// them.
    #[test]
    fn a_schema_4_trace_reads_with_its_summed_step_fields_ignored() {
        let old = r#"{
            "schema_version": 4, "t0": 0, "provenance": "plug_in",
            "conditioning": "in_sample", "warnings": [],
            "steps": [{
                "t": 7.0, "y_obs": 15.0, "y_pred_samples": [13.0, 17.0],
                "log_score": -4.5, "crps": 77.7, "pit": 0.5, "ess": 90.0,
                "interval": {"q05": 1.0, "q25": 2.0, "q50": 3.0, "q75": 4.0, "q95": 5.0},
                "per_stream": [
                    {"stream": "cases", "y_obs": 11.0, "y_pred_samples": [10.0, 12.0],
                     "log_score": -2.0, "crps": 0.5, "pit": 0.25,
                     "coverage": {"kind": "interval", "start": 0.0, "stop": 7.0}},
                    {"stream": "deaths", "y_obs": 4.0, "y_pred_samples": [3.0, 5.0],
                     "log_score": -2.5, "crps": 0.75, "pit": 0.75,
                     "coverage": {"kind": "instant"}}
                ]
            }]
        }"#;
        let t: PrequentialTrace = serde_json::from_str(old).expect("a schema-4 trace reads");
        assert_eq!(t.elpd(), -4.5);
        assert_eq!(t.streams(), ["cases", "deaths"]);
        assert_eq!(t.mean_crps("deaths").value, 0.75);
        assert_eq!(t.steps[0].per_stream[0].y_pred_samples, vec![10.0, 12.0]);
        let written = serde_json::to_value(&t).unwrap();
        let step = written["steps"][0].as_object().unwrap();
        for k in ["y_obs", "y_pred_samples", "crps", "pit", "interval"] {
            assert!(!step.contains_key(k), "a step carries no summed `{k}`: {written}");
        }
        assert!(!written.to_string().contains("77.7"), "{written}");
    }

    /// A schema-1 trace has no per-stream breakdown. It still reads and its
    /// elpd still sums, but it names no stream, so it has no CRPS or
    /// calibration figure to report.
    #[test]
    fn a_schema_1_trace_reads_and_has_elpd_but_no_stream_figures() {
        let old = r#"{
            "schema_version": 1, "t0": 1, "provenance": "plug_in", "warnings": [],
            "steps": [
                {"t": 7.0, "y_obs": 3.0, "y_pred_samples": [], "log_score": -1.5,
                 "crps": 0.5, "pit": 0.5, "ess": 100.0},
                {"t": 14.0, "y_obs": 4.0, "y_pred_samples": [], "log_score": -2.5,
                 "crps": 0.5, "pit": 0.5, "ess": 100.0}
            ]
        }"#;
        let t: PrequentialTrace = serde_json::from_str(old).expect("a schema-1 trace reads");
        assert_eq!(t.elpd(), -4.0);
        assert_eq!(t.n_scored(), 2);
        assert!(t.streams().is_empty(), "no per-stream breakdown, so no stream to summarize");
    }

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
        assert_eq!(t.score_from, None,
            "a pre-gh#585 trace has no recorded scoring boundary");
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
            score_from: None,
            pit_randomization_seed: None,
            derivation: None, scored_streams: None,
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
        // The v2 shape: InSample is a BARE string — the reason the enum is
        // externally tagged (a `tag = "kind"` change would fail to parse
        // every existing trace).
        assert_eq!(serde_json::to_string(&Conditioning::InSample).unwrap(), r#""in_sample""#);
        let back: Conditioning = serde_json::from_str(r#""in_sample""#).unwrap();
        assert_eq!(back, Conditioning::InSample);
        // The v3 struct variant (gh#585) nests under its external tag.
        let hot = Conditioning::HoldOutTail {
            train_end: 21.0, theta_source: "fits/sir-ab12cd34".into() };
        let json = serde_json::to_string(&hot).unwrap();
        assert_eq!(json,
            r#"{"hold_out_tail":{"train_end":21.0,"theta_source":"fits/sir-ab12cd34"}}"#);
        let back: Conditioning = serde_json::from_str(&json).unwrap();
        assert_eq!(back, hot);
    }

    #[test]
    fn provenance_serializes_snake_case() {
        // `plug_in` stays the bare string existing traces wrote; the
        // mixture variant nests under its external tag.
        assert_eq!(serde_json::to_string(&Provenance::PlugIn).unwrap(), r#""plug_in""#);
        let post = Provenance::Posterior { n_draws: 64 };
        let json = serde_json::to_string(&post).unwrap();
        assert_eq!(json, r#"{"posterior":{"n_draws":64}}"#);
        let back: Provenance = serde_json::from_str(&json).unwrap();
        assert_eq!(back, post);
    }

    #[test]
    fn posterior_holdout_trace_has_no_optimism_caveat() {
        // Honest on both axes (mixture over the posterior, sealed θ):
        // nothing left to caveat — the #295 endpoint.
        let t = PrequentialTrace {
            schema_version: 3, t0: 3,
            provenance: Provenance::Posterior { n_draws: 64 },
            conditioning: Conditioning::HoldOutTail {
                train_end: 21.0, theta_source: "fits/sir-ab12cd34".into() },
            steps: vec![], warnings: vec![],
            score_from: Some(21.0),
            pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        assert_eq!(t.optimism_caveat(), None,
            "posterior + held-out is honest on both axes");
        // Posterior but in-sample: only the conditioning axis is flagged.
        let t2 = PrequentialTrace {
            conditioning: Conditioning::InSample, ..t
        };
        let caveat = t2.optimism_caveat().expect("in-sample axis still flagged");
        assert!(!caveat.contains("plug-in"), "{caveat}");
        assert!(caveat.contains("in-sample"), "{caveat}");
    }

    #[test]
    fn hold_out_tail_clears_the_in_sample_caveat_axis() {
        // A sealed-θ trace is honest in conditioning; the caveat must name
        // only the remaining plug-in axis.
        let t = PrequentialTrace {
            schema_version: 3, t0: 3,
            provenance: Provenance::PlugIn,
            conditioning: Conditioning::HoldOutTail {
                train_end: 21.0, theta_source: "fits/sir-ab12cd34".into() },
            steps: vec![], warnings: vec![],
            score_from: Some(21.0),
            pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        let caveat = t.optimism_caveat().expect("plug-in axis still flagged");
        assert!(caveat.contains("plug-in"));
        assert!(!caveat.contains("in-sample"),
            "held-out conditioning must not be called in-sample: {caveat}");
    }

    fn scored_step(t: f64, log_score: f64, streams: &[(&str, f64)]) -> PrequentialStep {
        PrequentialStep {
            t, log_score, ess: 10.0,
            per_stream: streams.iter().map(|(name, ls)| StreamScore {
                stream: name.to_string(), y_obs: 0.0, y_pred_samples: vec![],
                log_score: *ls, crps: 0.0, pit: 0.5, interval: PredInterval::default(),
                coverage: Coverage::Unrecorded,
            }).collect(),
        }
    }

    #[test]
    fn worst_scored_ranks_unreachable_then_defective_then_ascending() {
        let t = PrequentialTrace {
            schema_version: 3, t0: 0,
            provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![
                scored_step(1.0, -3.0, &[]),
                scored_step(2.0, -30.0, &[]),
                scored_step(3.0, -1.0, &[]),
                scored_step(4.0, f64::NEG_INFINITY, &[]),
                scored_step(5.0, f64::NAN, &[]),
                scored_step(6.0, -30.0, &[]),
            ],
            warnings: vec![], score_from: None, pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        let order: Vec<f64> = t.worst_scored(10).iter().map(|s| s.t).collect();
        // -inf (no particle reached it) precedes NaN (a scoring defect);
        // the tied -30s keep step order; the best score is last.
        assert_eq!(order, vec![4.0, 5.0, 2.0, 6.0, 1.0, 3.0]);
        let top: Vec<f64> = t.worst_scored(2).iter().map(|s| s.t).collect();
        assert_eq!(top, vec![4.0, 5.0], "k truncates after ranking, not before");
        assert!(PrequentialTrace { steps: vec![], ..t.clone() }.worst_scored(3).is_empty());
    }

    #[test]
    fn worst_stream_is_the_lowest_per_stream_score_or_none() {
        let s = scored_step(1.0, -9.0, &[("a", -2.0), ("b", -7.0), ("c", -7.0)]);
        assert_eq!(s.worst_stream().map(|w| w.stream.as_str()), Some("b"),
            "ties break toward the first stream");
        let s = scored_step(1.0, -9.0, &[("a", -2.0), ("b", f64::NEG_INFINITY)]);
        assert_eq!(s.worst_stream().map(|w| w.stream.as_str()), Some("b"));
        assert!(scored_step(1.0, -9.0, &[]).worst_stream().is_none(),
            "a single-stream or v1 trace has no per-stream breakdown");
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
    fn crps_fair_matches_naive_pairwise_formula() {
        // The O(S log S) order-statistic form against the definitional
        // O(S²) pairwise form.
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
            acc / (2.0 * s_f * (s_f - 1.0))
        };
        let naive = term1 - term2;
        let fast = crps_sample_fair(&samples, y);
        assert!(approx_eq(naive, fast, 1e-10), "naive = {naive}, fast = {fast}");
        // Degenerate predictive: |x − y|, as in the edf form.
        assert!(approx_eq(crps_sample_fair(&[3.0], 5.0), 2.0, 1e-12));
        assert!(crps_sample_fair(&[], 5.0).is_nan());
    }

    #[test]
    fn crps_fair_and_edf_differ_by_the_ensemble_size_factor() {
        // fair = term1 − (S/(S−1))·(term1 − edf): the two estimators share
        // term1 and differ only in the pairwise term's normalization.
        let samples = vec![0.5, 1.0, 2.0, 2.0, 7.5];
        let y = 1.5;
        let s_f = samples.len() as f64;
        let term1: f64 = samples.iter().map(|x: &f64| (x - y).abs()).sum::<f64>() / s_f;
        let edf = crps_sample(&samples, y);
        let fair = crps_sample_fair(&samples, y);
        let expect = term1 - (s_f / (s_f - 1.0)) * (term1 - edf);
        assert!(approx_eq(fair, expect, 1e-10), "fair = {fair}, expect = {expect}");
        assert!(fair < edf, "fair subtracts the larger spread term");
    }

    #[test]
    fn crps_fair_is_unbiased_in_ensemble_size_where_edf_is_not() {
        // Ferro (2014): E[fair CRPS] does not depend on the ensemble size S,
        // while the edf estimator is biased upward by E|X−X'|/(2S). Score a
        // calibrated Poisson(5) forecast with S = 5 and S = 400 ensembles:
        // the fair means must agree; the edf mean at S = 5 must sit visibly
        // above (bias ≈ E|X−X'|/10 ≈ 0.25 here).
        let mut rng = crate::rng::StatefulRng::new_stream(4242, 0);
        let mut draw = |m: usize| -> (Vec<f64>, f64) {
            let s: Vec<f64> = (0..m).map(|_| rng.poisson(5.0) as f64).collect();
            let y = rng.poisson(5.0) as f64;
            (s, y)
        };
        let n_small = 4000;
        let (mut fair_small, mut edf_small) = (0.0, 0.0);
        for _ in 0..n_small {
            let (s, y) = draw(5);
            fair_small += crps_sample_fair(&s, y);
            edf_small += crps_sample(&s, y);
        }
        fair_small /= n_small as f64;
        edf_small /= n_small as f64;
        let n_big = 300;
        let mut fair_big = 0.0;
        for _ in 0..n_big {
            let (s, y) = draw(400);
            fair_big += crps_sample_fair(&s, y);
        }
        fair_big /= n_big as f64;
        assert!(approx_eq(fair_small, fair_big, 0.09),
            "fair estimator must not depend on ensemble size: \
             S=5 mean {fair_small}, S=400 mean {fair_big}");
        assert!(edf_small - fair_small > 0.15,
            "edf at S=5 must show its O(1/S) upward bias: \
             edf {edf_small}, fair {fair_small}");
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
        // midpoint of the atom's interval (the conditional mean, not CGH's
        // nonrandomized PIT).
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
        // 0.5 on average, by half the mean atom mass Σ p(k)²/2: for
        // Poisson(3), Σ p(k)² = 0.1667, so the bias is 0.0833 and the naive
        // mean is 0.5833 (measured 0.5828). The 0.03 threshold below is well
        // inside that.
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
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]],
            per_stream_samples: vec![vec![vec![1.0, 2.0, 2.0, 3.0]]],
        };
        let per_stream_observed = vec![vec![2.0]];  // ties with two samples → v matters
        let cov = interval_coverage(&recorded.obs_times, 1);
        let ess = vec![100.0];
        let a = build_trace(&recorded, &per_stream_observed, &cov, &ess, 0, 11, true, None);
        let b = build_trace(&recorded, &per_stream_observed, &cov, &ess, 0, 11, true, None);
        let c = build_trace(&recorded, &per_stream_observed, &cov, &ess, 0, 12, true, None);
        let pit = |t: &PrequentialTrace| t.steps[0].per_stream[0].pit;
        assert_eq!(a.pit_randomization_seed, Some(11));
        assert_eq!(pit(&a), pit(&b), "same seed must reproduce");
        assert_ne!(pit(&a), pit(&c), "a different seed must move the PIT at a tied step");
        // The randomized PIT stays inside the atom's interval.
        assert!(pit(&a) >= 0.25 && pit(&a) <= 0.75);
        // The draw order: the stream's PIT takes the first uniform of the
        // seeded stream — no draw precedes it for a step-level PIT.
        let v = crate::rng::StatefulRng::new_stream(11, PIT_RNG_STREAM).uniform();
        assert_eq!(pit(&a), pit_sample_randomized(&[1.0, 2.0, 2.0, 3.0], 2.0, v));
    }

    /// A trace whose streams carry the given (CRPS, PIT) at successive steps;
    /// a stream with fewer entries than another is absent from the later steps.
    fn trace_of_streams(streams: &[(&str, &[(f64, f64)])]) -> PrequentialTrace {
        let n = streams.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
        let steps = (0..n).map(|k| {
            let mut step = scored_step(k as f64, -1.0, &[]);
            step.per_stream = streams.iter().filter_map(|(name, v)| v.get(k).map(|&(crps, pit)| {
                StreamScore {
                    stream: name.to_string(), y_obs: 0.0, y_pred_samples: vec![],
                    log_score: -1.0, crps, pit, interval: PredInterval::default(),
                    coverage: Coverage::Unrecorded,
                }
            })).collect();
            step
        }).collect();
        PrequentialTrace {
            schema_version: 5, t0: 0, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample, steps, warnings: vec![],
            score_from: None, pit_randomization_seed: None, derivation: None, scored_streams: None,
        }
    }

    #[test]
    fn pit_coverage_at_perfect_uniform() {
        // A stream whose PIT values span [0, 1] uniformly should have
        // ~level coverage at level. Synthesize 100 evenly-spaced PITs.
        let pits: Vec<(f64, f64)> = (0..100).map(|i| (0.0, (i as f64 + 0.5) / 100.0)).collect();
        let trace = trace_of_streams(&[("cases", &pits)]);
        // 90% interval = PIT in [0.05, 0.95] — 90 of 100 PITs qualify.
        let cov = trace.pit_coverage("cases", 0.90);
        assert_eq!((cov.n, cov.n_nonfinite), (100, 0));
        assert!(approx_eq(cov.value, 0.90, 0.01), "got {:?}", cov);
        // 50% interval = PIT in [0.25, 0.75].
        let cov50 = trace.pit_coverage("cases", 0.50);
        assert!(approx_eq(cov50.value, 0.50, 0.02), "got {:?}", cov50);
    }

    /// D14: calibration is a per-stream fact. A trace whose `cases` stream is
    /// calibrated and whose `deaths` stream is overconfident reports each
    /// stream's coverage on its own; neither masks the other, and the mean
    /// CRPS of each is in its own units, over its own observations.
    #[test]
    fn per_stream_summaries_do_not_mask_one_stream_with_another() {
        let cases: Vec<(f64, f64)> = (0..10).map(|i| (2.0, 0.1 + 0.08 * i as f64)).collect();
        let deaths: Vec<(f64, f64)> = (0..4).map(|i| (0.5 + i as f64, 0.995)).collect();
        let trace = trace_of_streams(&[("cases", &cases), ("deaths", &deaths)]);
        assert_eq!(trace.streams(), ["cases", "deaths"]);
        let c = trace.pit_coverage("cases", 0.90);
        let d = trace.pit_coverage("deaths", 0.90);
        assert_eq!((c.n, c.value), (10, 1.0), "every cases PIT is inside: {c:?}");
        assert_eq!((d.n, d.value), (4, 0.0), "no deaths PIT is inside: {d:?}");
        assert_eq!(trace.mean_crps("cases"), StreamMean { n: 10, n_nonfinite: 0, value: 2.0 });
        assert_eq!(trace.mean_crps("deaths"), StreamMean { n: 4, n_nonfinite: 0, value: 2.0 });
        let none = trace.mean_crps("hospital");
        assert_eq!((none.n, none.n_nonfinite), (0, 0));
        assert!(none.value.is_nan(), "a stream never scored has no mean");
    }

    /// A non-finite PIT or CRPS is left out of the mean and counted, never
    /// read as an observation outside the interval or as a large error.
    #[test]
    fn per_stream_summaries_exclude_and_count_non_finite_values() {
        let v = [(1.0, 0.5), (f64::NAN, f64::NAN), (3.0, 0.99), (f64::INFINITY, 0.5)];
        let trace = trace_of_streams(&[("cases", &v)]);
        let cov = trace.pit_coverage("cases", 0.90);
        assert_eq!((cov.n, cov.n_nonfinite), (4, 1));
        assert!(approx_eq(cov.value, 2.0 / 3.0, 1e-12),
            "two of the three finite PITs are inside, the NaN is not a miss: {cov:?}");
        let crps = trace.mean_crps("cases");
        assert_eq!((crps.n, crps.n_nonfinite), (4, 2));
        assert!(approx_eq(crps.value, 2.0, 1e-12), "{crps:?}");
        let all_nan = trace_of_streams(&[("cases", &[(f64::NAN, f64::NAN)])]);
        assert!(all_nan.mean_crps("cases").value.is_nan());
        assert!(all_nan.pit_coverage("cases", 0.9).value.is_nan());
    }

    #[test]
    fn build_trace_from_recorded_aligns_with_kernels() {
        // Hand-rolled PrequentialRecorded with two steps; verify
        // build_trace computes the same log_score / crps / pit as the
        // standalone kernels and forwards ess correctly. The step carries
        // only the joint log score; CRPS and PIT are the stream's.
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0],
            log_liks: vec![
                vec![-1.0, -2.0, -0.5, -3.0],
                vec![-0.1, -0.2, -0.3, -0.4],
            ],
            // Single-stream recorded: the stream's per-particle log-liks ARE
            // the joint ones (one stream ⇒ joint = the stream).
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
        let per_stream_observed = vec![vec![2.5], vec![5.2]];
        // 4 particles ⇒ collapse threshold 0.1·4 = 0.4; second step below it.
        let ess = vec![100.0, 0.3];

        let trace = build_trace(&recorded, &per_stream_observed,
                                &interval_coverage(&recorded.obs_times, 1), &ess, 0, 7, true, None);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 0);

        // Step 0: log_score = logsumexp(log_liks) - log N
        let expected_ls0 = super::super::types::log_sum_exp(&recorded.log_liks[0])
            - (4.0_f64).ln();
        assert!(approx_eq(trace.steps[0].log_score, expected_ls0, 1e-12));
        // CRPS/PIT agree with kernels.
        let s0 = &trace.steps[0].per_stream[0];
        assert!(approx_eq(s0.crps,
            crps_sample_fair(&recorded.per_stream_samples[0][0], 2.5), 1e-12));
        // y = 2.5 ties no sample, so the PIT is v-independent and must match
        // the kernel at any v.
        assert!(approx_eq(s0.pit,
            pit_sample_randomized(&recorded.per_stream_samples[0][0], 2.5, 0.0), 1e-12));

        // ESS warning fires for the low-ess second step.
        assert_eq!(trace.warnings.len(), 1);
        matches!(trace.warnings[0], PrequentialWarning::EssCollapse { step_count: 1, .. });

        // gh#269: one stream, so the single per-stream log score equals the
        // joint (same per-particle log-liks).
        assert_eq!(trace.schema_version, 5);
        assert_eq!(trace.steps[0].per_stream.len(), 1);
        assert_eq!(s0.stream, "s0");
        assert!(approx_eq(s0.log_score, trace.steps[0].log_score, 1e-12));
    }

    #[test]
    fn ess_collapse_threshold_scales_with_particle_count() {
        // The cue is a FRACTION of N (10%), not the old absolute 10: with 4
        // particles, ESS 4.0 (full survival) must not warn — under the old
        // absolute floor it did — while ESS below 0.4 must, and the warning
        // records the absolute threshold that was applied.
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0],
            log_liks: vec![vec![-1.0; 4]; 2],
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]; 2],
            per_stream_samples: vec![vec![vec![0.5, 1.0, 1.5, 2.0]]; 2],
        };
        let per_stream_observed = vec![vec![1.25]; 2];
        let cov = interval_coverage(&recorded.obs_times, 1);

        let healthy = build_trace(&recorded, &per_stream_observed, &cov,
                                  &[4.0, 4.0], 0, 7, true, None);
        assert!(healthy.warnings.is_empty(),
            "full survival at small N must not warn: {:?}", healthy.warnings);

        let collapsed = build_trace(&recorded, &per_stream_observed, &cov,
                                    &[4.0, 0.3], 0, 7, true, None);
        match collapsed.warnings.as_slice() {
            [PrequentialWarning::EssCollapse { step_count, threshold }] => {
                assert_eq!(*step_count, 1);
                assert!(approx_eq(*threshold, 0.4, 1e-12),
                    "threshold must be ESS_COLLAPSE_FRACTION · N, got {threshold}");
            }
            other => panic!("expected one EssCollapse warning, got {other:?}"),
        }
    }

    #[test]
    fn starts_at_prior_warns_unless_warmup_or_t0() {
        // t0 = 0 with no conditioning window: the first scored predictive is
        // issued from the initial-state distribution — warn. Either a
        // conditioning warm-up or a t0 skip clears it.
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0],
            log_liks: vec![vec![-1.0; 4]; 2],
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]; 2],
            per_stream_samples: vec![vec![vec![0.5, 1.0, 1.5, 2.0]]; 2],
        };
        let obs = vec![vec![1.25]; 2];
        let cov = interval_coverage(&recorded.obs_times, 1);
        let ess = vec![100.0; 2];

        let bare = build_trace(&recorded, &obs, &cov, &ess, 0, 7, false, None);
        assert!(bare.warnings.iter()
            .any(|w| matches!(w, PrequentialWarning::StartsAtPrior)),
            "t0=0 without a warm-up must warn: {:?}", bare.warnings);

        let warmed = build_trace(&recorded, &obs, &cov, &ess, 0, 7, true, None);
        assert!(!warmed.warnings.iter()
            .any(|w| matches!(w, PrequentialWarning::StartsAtPrior)),
            "a conditioning window places the boundary deliberately");

        let skipped = build_trace(&recorded, &obs, &cov, &ess, 1, 7, false, None);
        assert!(!skipped.warnings.iter()
            .any(|w| matches!(w, PrequentialWarning::StartsAtPrior)),
            "t0 >= 1 assimilates before scoring");

        // Serialization: the tagged form downstream renderers read.
        let json = serde_json::to_string(&PrequentialWarning::StartsAtPrior).unwrap();
        assert_eq!(json, r#"{"kind":"starts_at_prior"}"#);
    }

    #[test]
    fn build_trace_respects_t0_skip() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0, 3.0],
            log_liks: vec![vec![-1.0; 4]; 3],
            stream_names: vec!["s0".to_string()],
            per_stream_log_liks: vec![vec![vec![-1.0; 4]]; 3],
            per_stream_samples: vec![vec![vec![0.5, 1.0, 1.5, 2.0]]; 3],
        };
        let per_stream_observed = vec![vec![1.25]; 3];
        let ess = vec![100.0; 3];

        let trace = build_trace(&recorded, &per_stream_observed,
                                &interval_coverage(&recorded.obs_times, 1), &ess, 1, 7, true, None);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.t0, 1);
        assert_eq!(trace.steps[0].t, 2.0);
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

    // ── project_onto_stream ─────────────────────────────────────────────

    /// One stream's score with every field distinct, so a copy that drops or
    /// swaps one shows.
    fn distinct_score(stream: &str, k: f64) -> StreamScore {
        StreamScore {
            stream: stream.into(), y_obs: 10.0 + k, y_pred_samples: vec![9.0 + k, 11.0 + k],
            log_score: -1.0 - k, crps: 0.25 + k, pit: 0.3,
            interval: PredInterval { q05: k, q25: k + 1.0, q50: k + 2.0, q75: k + 3.0, q95: k + 4.0 },
            coverage: Coverage::Interval { start: k, stop: k + 7.0 },
        }
    }

    /// A two-leaf trace: `cases` at 7, 14, 21; `deaths` at 14 and 21 only.
    /// Joint scores differ from every per-leaf score.
    fn cases_and_deaths() -> PrequentialTrace {
        let step = |t: f64, ls: f64, ess: f64, per: Vec<StreamScore>| PrequentialStep {
            t, log_score: ls, ess, per_stream: per,
        };
        PrequentialTrace {
            schema_version: 5, t0: 1, provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![
                step(7.0, -10.0, 95.0, vec![distinct_score("cases", 1.0)]),
                step(14.0, -20.0, 5.0, vec![distinct_score("cases", 2.0), distinct_score("deaths", 3.0)]),
                step(21.0, -30.0, 90.0, vec![distinct_score("deaths", 4.0), distinct_score("cases", 5.0)]),
            ],
            warnings: vec![PrequentialWarning::StartsAtPrior,
                           PrequentialWarning::EssCollapse { step_count: 1, threshold: 10.0 }],
            score_from: Some(3.0), pit_randomization_seed: Some(9),
            derivation: Some(Derivation {
                particles: 100, replicates: 3, dt: 1.0, seed: 4, theta: ThetaSource::Supplied,
                excluded_chains: vec![2], enabled: vec!["x".into()], disabled: vec![],
                replicate_elpd: vec![-60.0, -61.0, -59.0],
                replicate_elpd_by_stream: [
                    ("cases".to_string(), vec![-8.0, -9.5, -7.0]),
                    ("deaths".to_string(), vec![-12.0, -12.5, -11.0]),
                ].into_iter().collect(),
                mc_se_elpd: Some(0.5),
                overrides: vec![TraceOverride::InSampleForced],
            }),
            scored_streams: None,
        }
    }

    /// Each surviving step is the leaf's own score, copied: the step keeps its
    /// time and ESS, its log score becomes the leaf's, and its per-stream
    /// breakdown is the leaf's entry alone, field for field. Steps that did not
    /// score the leaf are dropped.
    #[test]
    fn projection_copies_the_leaf_entry_and_drops_steps_without_it() {
        let trace = cases_and_deaths();
        let p = project_onto_stream(&trace, "deaths").unwrap();
        assert_eq!(p.steps.iter().map(|s| s.t).collect::<Vec<_>>(), vec![14.0, 21.0],
            "the step at 7 observed only cases");
        for (got, (src, k)) in p.steps.iter().zip([(&trace.steps[1], 1), (&trace.steps[2], 0)]) {
            let entry = &src.per_stream[k];
            assert_eq!(entry.stream, "deaths");
            assert_eq!(got.log_score, entry.log_score, "the leaf's own score, not the joint");
            assert_eq!(got.ess, src.ess, "the filter's ESS at that step");
            assert_eq!(got.per_stream.len(), 1);
            assert_eq!(serde_json::to_value(&got.per_stream[0]).unwrap(),
                       serde_json::to_value(entry).unwrap(), "copied unchanged");
        }
        assert_eq!(p.scored_streams, Some(vec!["deaths".to_string()]));
        assert_eq!(p.streams(), ["deaths"], "a single-stream trace");
        assert_eq!(p.elpd(), -4.0 + -5.0, "Σ of deaths' own log scores");
        // Fields that describe the pass are carried over.
        assert_eq!((p.schema_version, p.t0, p.score_from, p.pit_randomization_seed),
                   (5, 1, Some(3.0), Some(9)));
        assert_eq!(p.provenance, trace.provenance);
        assert_eq!(p.conditioning, trace.conditioning);
    }

    /// The derivation block is restated for the leaf: its replicate totals,
    /// those totals as the only per-stream entry, and the filter-noise SE
    /// recomputed from them. Everything else in the block is carried over.
    #[test]
    fn projection_restates_the_derivation_for_the_leaf() {
        let trace = cases_and_deaths();
        let p = project_onto_stream(&trace, "cases").unwrap();
        let d = p.derivation.as_ref().unwrap();
        let totals = vec![-8.0, -9.5, -7.0];
        assert_eq!(d.replicate_elpd, totals);
        assert_eq!(d.replicate_elpd_by_stream,
            [("cases".to_string(), totals.clone())].into_iter().collect());
        let (_, se) = super::super::types::logmeanexp_with_se(&totals);
        assert_eq!(d.mc_se_elpd, Some(se), "the leaf's SE, not the joint 0.5");
        let orig = trace.derivation.as_ref().unwrap();
        assert_eq!((d.particles, d.replicates, d.dt, d.seed, d.theta),
                   (orig.particles, orig.replicates, orig.dt, orig.seed, orig.theta));
        assert_eq!((&d.excluded_chains, &d.enabled, &d.disabled, &d.overrides),
                   (&orig.excluded_chains, &orig.enabled, &orig.disabled, &orig.overrides));

        // One replicate: no SE, as for a single pass.
        let mut one = cases_and_deaths();
        let od = one.derivation.as_mut().unwrap();
        od.replicates = 1;
        od.replicate_elpd = vec![-60.0];
        od.replicate_elpd_by_stream = [("cases".to_string(), vec![-8.0]),
                                       ("deaths".to_string(), vec![-12.0])].into_iter().collect();
        od.mc_se_elpd = None;
        let p = project_onto_stream(&one, "cases").unwrap();
        assert_eq!(p.derivation.as_ref().unwrap().mc_se_elpd, None);
        assert_eq!(p.derivation.as_ref().unwrap().replicate_elpd, vec![-8.0]);

        // A block with no totals for the leaf cannot be restated, and must not
        // be left describing the joint score.
        let mut bad = cases_and_deaths();
        bad.derivation.as_mut().unwrap().replicate_elpd_by_stream.remove("cases");
        let err = project_onto_stream(&bad, "cases").unwrap_err();
        assert!(err.contains("'cases'") && err.contains("replicate"), "{err}");

        // No block: nothing to restate.
        let mut none = cases_and_deaths();
        none.derivation = None;
        assert_eq!(project_onto_stream(&none, "cases").unwrap().derivation, None);
    }

    /// `StartsAtPrior` describes the trace's first step, so it stays only when
    /// that step survives; `EssCollapse` is recounted over the surviving steps
    /// at the recorded threshold, and dropped when none falls below it.
    #[test]
    fn projection_keeps_only_the_warnings_its_steps_still_carry() {
        let trace = cases_and_deaths();
        let cases = project_onto_stream(&trace, "cases").unwrap();
        assert!(cases.warnings.iter().any(|w| matches!(w, PrequentialWarning::StartsAtPrior)),
            "cases keeps the first step: {:?}", cases.warnings);
        assert!(cases.warnings.iter().any(|w| matches!(w,
            PrequentialWarning::EssCollapse { step_count: 1, threshold } if *threshold == 10.0)),
            "{:?}", cases.warnings);

        let deaths = project_onto_stream(&trace, "deaths").unwrap();
        assert!(!deaths.warnings.iter().any(|w| matches!(w, PrequentialWarning::StartsAtPrior)),
            "deaths' first score is at 14, after assimilating 7: {:?}", deaths.warnings);
        assert!(deaths.warnings.iter().any(|w| matches!(w,
            PrequentialWarning::EssCollapse { step_count: 1, .. })), "{:?}", deaths.warnings);

        // The collapsed step (ESS 5 at t = 14) projected away: no collapse left.
        let mut healthy_late = cases_and_deaths();
        healthy_late.steps[1].per_stream.retain(|s| s.stream == "cases");
        let deaths = project_onto_stream(&healthy_late, "deaths").unwrap();
        assert_eq!(deaths.steps.len(), 1);
        assert!(!deaths.warnings.iter().any(|w| matches!(w, PrequentialWarning::EssCollapse { .. })),
            "{:?}", deaths.warnings);

        // A step count recounted, not copied: two collapsed steps in the
        // input, one of them surviving.
        let mut two = cases_and_deaths();
        two.steps[0].ess = 1.0;
        two.warnings[1] = PrequentialWarning::EssCollapse { step_count: 2, threshold: 10.0 };
        let deaths = project_onto_stream(&two, "deaths").unwrap();
        assert!(deaths.warnings.iter().any(|w| matches!(w,
            PrequentialWarning::EssCollapse { step_count: 1, .. })), "{:?}", deaths.warnings);

        // Other warnings are carried over.
        let mut unsaved = cases_and_deaths();
        unsaved.warnings.push(PrequentialWarning::SamplesNotSaved);
        let p = project_onto_stream(&unsaved, "deaths").unwrap();
        assert!(p.warnings.iter().any(|w| matches!(w, PrequentialWarning::SamplesNotSaved)));
    }

    /// The refusals: a leaf the trace never scored, a trace with no per-stream
    /// breakdown, and a trace with no step at all.
    #[test]
    fn projection_refuses_what_it_cannot_project() {
        let err = project_onto_stream(&cases_and_deaths(), "hospital").unwrap_err();
        assert!(err.contains("'hospital'") && err.contains("cases") && err.contains("deaths"),
            "names the leaf and the leaves the trace scored: {err}");

        let mut v1 = cases_and_deaths();
        for s in &mut v1.steps { s.per_stream.clear(); }
        let err = project_onto_stream(&v1, "cases").unwrap_err();
        assert!(err.contains("per-stream breakdown") && err.contains("'cases'"), "{err}");

        let mut empty = cases_and_deaths();
        empty.steps.clear();
        let err = project_onto_stream(&empty, "cases").unwrap_err();
        assert!(err.contains("no scored step") && err.contains("'cases'"), "{err}");
    }

    /// A one-leaf trace's joint score is its leaf's score, so projecting it
    /// onto that leaf reproduces its elpd.
    #[test]
    fn projecting_a_one_leaf_trace_onto_its_leaf_reproduces_its_elpd() {
        let mut t = cases_and_deaths();
        for s in &mut t.steps {
            s.per_stream.retain(|x| x.stream == "cases");
            s.log_score = s.per_stream[0].log_score;
        }
        let p = project_onto_stream(&t, "cases").unwrap();
        assert_eq!(p.elpd(), t.elpd());
        assert_eq!(p.n_scored(), t.n_scored());
    }

    /// `scored_streams` is optional on disk: a trace without it reads as a
    /// joint trace, and a joint trace writes no such key.
    #[test]
    fn a_trace_without_scored_streams_reads_as_a_joint_trace() {
        let old = r#"{
            "schema_version": 5, "t0": 0, "provenance": "plug_in",
            "conditioning": "in_sample", "steps": [], "warnings": []
        }"#;
        let t: PrequentialTrace = serde_json::from_str(old).expect("reads");
        assert_eq!(t.scored_streams, None);
        let written = serde_json::to_value(&t).unwrap();
        assert!(written.get("scored_streams").is_none(), "{written}");
        let p = project_onto_stream(&cases_and_deaths(), "cases").unwrap();
        let back: PrequentialTrace =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back.scored_streams, Some(vec!["cases".to_string()]));
    }
}
