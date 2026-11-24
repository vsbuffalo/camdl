//! Building a prequential trace from one particle-filter pass:
//! [`build_trace`] turns the filter's recorded output into a
//! `score::trace::PrequentialTrace`, drawing the randomized-PIT uniforms
//! from `sim`'s random stream. The ODE skeleton's record (one exact density
//! per step plus observation-model draws, gh#312) is built in
//! `ode_loglik.rs` and goes through the same [`build_trace`]. The trace
//! types, the scoring-rule kernels and the summaries read from a trace are in
//! `score::trace`.

use score::trace::{
    crps_sample_fair, pit_sample_randomized, Conditioning, Coverage, PredInterval,
    PredictiveBasis, PrequentialStep, PrequentialTrace, PrequentialWarning, Provenance,
    StreamScore,
};

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
        let log_score = numerics::log_sum_exp(log_liks) - n.ln();
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
        // A particle the filter killed has density zero in every stream, and
        // its draw is no sample of the predictive.
        let dead = dead_particles(log_liks, &recorded.per_stream_log_liks[idx]);
        for si in 0..recorded.stream_names.len() {
            let y_s = per_stream_observed[idx][si];
            if !y_s.is_finite() { continue; }
            let ll_s: Vec<f64> = recorded.per_stream_log_liks[idx][si].iter().zip(&dead)
                .map(|(&l, &d)| if d { f64::NEG_INFINITY } else { l })
                .collect();
            // A particle's draw is left out only where the record holds one
            // draw per particle (a swarm). The ODE skeleton's record has one
            // support point and many draws from it, and no particle to kill.
            let raw = &recorded.per_stream_samples[idx][si];
            let samp_s: Vec<f64> = if raw.len() == dead.len() {
                raw.iter().zip(&dead).filter(|(_, &d)| !d).map(|(&y, _)| y).collect()
            } else {
                raw.clone()
            };
            let n_s = ll_s.len() as f64;
            per_stream.push(StreamScore {
                stream: recorded.stream_names[si].clone(),
                y_obs: y_s,
                y_pred_samples: samp_s.clone(),
                log_score: numerics::log_sum_exp(&ll_s) - n_s.ln(),
                crps: crps_sample_fair(&samp_s, y_s),
                pit: pit_sample_randomized(&samp_s, y_s, pit_rng.uniform()),
                interval: PredInterval::from_samples(&samp_s),
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
        // v5: the `derivation` block (filled by the caller). v6 (gh#312): the
        // `basis` axis. v7 (gh#1235): a forcing held while the trace was scored
        // is a `ForcingHeld` warning (added by the caller, which holds the
        // policy). All additions serde-defaulted, so older traces still
        // read; they read as InSample/PlugIn/Unrecorded/no derivation/Filtered,
        // which is factually what they are; one below 7 cannot say whether a
        // forcing was held, and `compare` refuses it.
        schema_version: 7,
        t0,
        // This builder scores a single filter pass at one θ over the full data:
        // plug-in + in-sample. The posterior / LFO producers (#295) stamp their
        // own values.
        provenance: Provenance::PlugIn,
        conditioning: Conditioning::InSample,
        // The particle filter's record is the default; the ODE caller, whose
        // record has one deterministic support point, restamps it.
        basis: PredictiveBasis::Filtered,
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

/// Which particles of one stop of a prequential record the filter had
/// killed, from the stop's joint log weights `joint[particle]` and its
/// per-stream log-likelihoods `per_stream[stream][particle]`.
///
/// The filter sets a dead particle's joint log weight to −∞ and still
/// evaluates its per-stream log-likelihoods on its dead state, where they are
/// finite. For a live particle the per-stream terms sum to the joint one, so
/// a live particle with weight −∞ has a −∞ term in some stream. A particle is
/// dead when its joint weight is −∞ and its per-stream terms (a NaN, which an
/// unscheduled stream holds, left out) do not sum to −∞. Its density of every
/// stream's value is zero, and its predictive draw is no sample of the
/// predictive.
pub fn dead_particles(joint: &[f64], per_stream: &[Vec<f64>]) -> Vec<bool> {
    (0..joint.len())
        .map(|i| {
            let sum: f64 = per_stream.iter().filter_map(|s| s.get(i)).filter(|v| !v.is_nan()).sum();
            joint[i] == f64::NEG_INFINITY && sum != f64::NEG_INFINITY
        })
        .collect()
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
            per_stream_held_out_log_liks: Vec::new(),
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
    use score::trace::crps_sample;

    /// A particle the filter killed has a joint log weight of −∞, while its
    /// per-stream log-likelihoods and its predictive draw are read off its
    /// dead state and are finite. It has density zero: with one stream, the
    /// stream's score equals the joint one, and its draw is not a sample.
    /// A live particle whose density of a value is zero is not dead.
    #[test]
    fn a_dead_particle_has_density_zero_in_every_stream() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![7.0],
            log_liks: vec![vec![-1.0, f64::NEG_INFINITY, f64::NEG_INFINITY]],
            stream_names: vec!["a".into()],
            per_stream_held_out_log_liks: Vec::new(),
            per_stream_log_liks: vec![vec![vec![-1.0, -0.5, f64::NEG_INFINITY]]],
            per_stream_samples: vec![vec![vec![3.0, 100.0, 4.0]]],
        };
        assert_eq!(dead_particles(&recorded.log_liks[0], &recorded.per_stream_log_liks[0]), vec![false, true, false]);
        let trace = build_trace(&recorded, &[vec![3.0]], &interval_coverage(&recorded.obs_times, 1), &[1.0], 0, 7, true, None);
        let step = &trace.steps[0];
        assert_eq!(step.per_stream[0].log_score, step.log_score, "one stream: its score is the joint one");
        assert!((step.log_score - ((-1.0_f64).exp() / 3.0).ln()).abs() < 1e-12);
        assert_eq!(step.per_stream[0].y_pred_samples, vec![3.0, 4.0], "the dead particle's draw is not a sample");
    }

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
            per_stream_held_out_log_liks: Vec::new(),
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

        assert_eq!(trace.schema_version, 7,
            "build_trace writes the current schema (coverage arrived in 4, basis in 6, held forcings in 7)");
        assert_eq!(trace.basis, PredictiveBasis::Filtered, "a filter's record is stamped filtered");
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
            per_stream_held_out_log_liks: Vec::new(),
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

    /// The randomized PIT of a two-stream trace, value by value. Uniforms
    /// come from the seeded PIT stream in step order, one per present
    /// stream in stream order: a tied value takes its own uniform, a
    /// non-tied value consumes one but does not depend on it, a hole
    /// consumes none, and there is no step-level draw. The uniforms and
    /// the tied PITs are pinned to their literal values, so a change to
    /// the draw order or to the PIT stream shows here as a changed number.
    #[test]
    fn build_trace_pit_draws_one_uniform_per_present_stream_in_order() {
        let recorded = super::super::particle_filter::PrequentialRecorded {
            obs_times: vec![1.0, 2.0, 3.0],
            log_liks: vec![vec![-1.0; 4]; 3],
            stream_names: vec!["a".to_string(), "b".to_string()],
            per_stream_held_out_log_liks: Vec::new(),
            per_stream_log_liks: vec![vec![vec![-1.0; 4]; 2]; 3],
            per_stream_samples: vec![
                vec![vec![1.0, 2.0, 2.0, 3.0], vec![0.0, 4.0, 4.0, 8.0]],
                vec![vec![7.0, 7.0, 7.0, 7.0], vec![1.0, 1.0, 2.0, 5.0]],
                vec![vec![10.0, 12.0, 14.0, 16.0], vec![3.0, 3.0, 3.0, 3.0]],
            ],
        };
        // Step 1: `a` ties (two of four draws equal 2), `b` does not (5).
        // Step 2: `a` is a hole, `b` ties (two draws equal 1).
        // Step 3: `a` does not tie (13), `b` ties with every draw (3).
        let observed = vec![
            vec![2.0, 5.0],
            vec![f64::NAN, 1.0],
            vec![13.0, 3.0],
        ];
        let cov = interval_coverage(&recorded.obs_times, 2);
        let ess = vec![4.0; 3];
        let build = |seed: u64| build_trace(&recorded, &observed, &cov, &ess, 0, seed, true, None);
        let pits = |t: &PrequentialTrace| -> Vec<(String, f64)> {
            t.steps.iter().flat_map(|s| s.per_stream.iter().map(|p| (p.stream.clone(), p.pit)))
                .collect()
        };

        let trace = build(11);
        let got = pits(&trace);
        let names: Vec<&str> = got.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(names, ["a", "b", "b", "a", "b"], "the hole has no score");

        // Five present values, five uniforms, consumed in this order.
        let mut rng = crate::rng::StatefulRng::new_stream(11, PIT_RNG_STREAM);
        let u: Vec<f64> = (0..5).map(|_| rng.uniform()).collect();
        let pinned_u = [
            0.8732899451957135, 0.8035777362823348, 0.038731235530629515,
            0.7962383186794344, 0.7860387529800408,
        ];
        for (k, (&a, &b)) in u.iter().zip(pinned_u.iter()).enumerate() {
            assert!((a - b).abs() <= 1e-15, "uniform {k} of the seed-11 PIT stream: {a:e} vs {b:e}");
        }
        // u = (#draws below y + v · #draws equal to y) / #draws.
        let want = [
            (1.0 + u[0] * 2.0) / 4.0, // a, step 1: tied, first uniform
            3.0 / 4.0,                // b, step 1: not tied (u[1] consumed, unused)
            (0.0 + u[2] * 2.0) / 4.0, // b, step 2: tied; the hole in `a` drew nothing
            2.0 / 4.0,                // a, step 3: not tied (u[3] consumed, unused)
            (0.0 + u[4] * 4.0) / 4.0, // b, step 3: tied with every draw
        ];
        for (k, ((s, g), w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "value {k} (stream {s}): {g:e} vs {w:e}");
        }
        let pinned_tied =
            [(0, 0.6866449725978567), (2, 0.019365617765314758), (4, 0.7860387529800408)];
        for (k, p) in pinned_tied {
            assert!((got[k].1 - p).abs() <= 1e-15, "tied value {k}: {:e} vs {p:e}", got[k].1);
        }

        // Another seed moves every tied value and no other.
        let other = pits(&build(12));
        for k in [0, 2, 4] {
            assert_ne!(got[k].1, other[k].1, "tied value {k} must depend on the seed");
        }
        for k in [1, 3] {
            assert_eq!(got[k].1.to_bits(), other[k].1.to_bits(),
                "non-tied value {k} is the empirical CDF at any seed");
        }
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
            per_stream_held_out_log_liks: Vec::new(),
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
        let expected_ls0 = numerics::log_sum_exp(&recorded.log_liks[0])
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
        assert_eq!(trace.schema_version, 7);
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
            per_stream_held_out_log_liks: Vec::new(),
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
            per_stream_held_out_log_liks: Vec::new(),
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
            per_stream_held_out_log_liks: Vec::new(),
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
}
