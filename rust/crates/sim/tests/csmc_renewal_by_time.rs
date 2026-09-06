//! gh#688: renewal resolved in time — the diagnostic the particle-Gibbs
//! literature recommends in place of a rule for choosing the particle count.
//!
//! There is no rate in `N`. Chopin & Singh (2015, *Bernoulli* 21:1855-1883)
//! prove uniform ergodicity for particle Gibbs existentially — for any ε there
//! exists an N₀ — with no rate; Lindsten, Jordan & Schön (2014, *JMLR*
//! 15:2145-2184) state in their conclusion that informative rates with an
//! explicit dependence on `N` are open. What both recommend instead is the same
//! diagnostic: **the update rate of the state xₜ plotted against t**. LJS
//! Figure 1 is exactly that plot (PG vs PGAS at N ∈ {5, 20, 100, 1000},
//! T = 400), and its shape is the whole point — under plain particle Gibbs the
//! update rate is ≈0 at small t and rises to 1 near t = T, because every
//! lineage has coalesced onto the reference by the time the traceback reaches
//! the early states. Ancestor sampling exists to flatten that curve.
//!
//! `trajectory_renewal` — the fraction of traceback substeps taken from a
//! non-reference particle, summed over the WHOLE series — cannot see that
//! shape. A sweep renewed only after the midpoint and a sweep renewed
//! uniformly in t score the identical scalar. The first is the failure LJS
//! Figure 1 is drawn to expose, and it is the one that matters: the early
//! states are where the parameters governing initial conditions and early
//! dynamics get their information.
//!
//! These tests pin (1) that the aggregate is blind to that difference and the
//! per-bin vector is not, and (2) that the vector `csmc_as` actually returns is
//! a per-bin count of real per-substep decisions, not the scalar rebroadcast.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, EffectFiring, ObsAtSubstep, PositionBins,
    RENEWAL_BINS,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260820;

// ─────────────────────────── the aggregation itself ───────────────────────────

/// Drive the PRODUCTION accumulator with a per-substep from-reference pattern.
/// `from_ref[s] == true` means the traceback took substep `s` from the
/// reference particle, i.e. that substep was NOT renewed.
fn bins_of(from_ref: &[bool]) -> [f64; RENEWAL_BINS] {
    let mut acc = PositionBins::new(from_ref.len());
    for (s, &r) in from_ref.iter().enumerate() {
        acc.record(s, !r);
    }
    acc.finish()
}

/// The scalar `csmc_as` has always reported, computed the way `csmc_as`
/// computes it: one minus the fraction of substeps taken from the reference.
fn aggregate(from_ref: &[bool]) -> f64 {
    let n_from_ref = from_ref.iter().filter(|&&r| r).count();
    1.0 - n_from_ref as f64 / from_ref.len() as f64
}

/// Bin sizes implied by the binning rule, recomputed here independently of the
/// implementation so the tiling assertions below are not circular.
fn bin_sizes(n_substeps: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; RENEWAL_BINS];
    for s in 0..n_substeps {
        sizes[(s * RENEWAL_BINS / n_substeps).min(RENEWAL_BINS - 1)] += 1;
    }
    sizes
}

/// The oracle. Two sweeps that the scalar scores IDENTICALLY, one of which is
/// the LJS Figure 1 particle-Gibbs pathology and one of which is healthy.
///
/// `T = 400` is LJS Figure 1's series length. The degenerate pattern is the PG
/// curve in the limit — the lineage coalesces onto the reference at the
/// midpoint, so every substep before it is the reference's own and every
/// substep after it is renewed. The healthy pattern renews at the same overall
/// rate but spread evenly in t.
#[test]
fn aggregate_renewal_is_blind_to_early_degeneracy_and_the_bins_are_not() {
    const T: usize = 400;

    let coalesced_at_midpoint: Vec<bool> = (0..T).map(|s| s < T / 2).collect();
    let renewed_evenly: Vec<bool> = (0..T).map(|s| s % 2 == 0).collect();

    // (1) The scalar cannot tell them apart. This is not incidental — it is the
    //     defect, so it is asserted rather than assumed.
    let (agg_bad, agg_ok) = (aggregate(&coalesced_at_midpoint), aggregate(&renewed_evenly));
    assert_eq!(
        agg_bad, agg_ok,
        "fixture is not an oracle unless both patterns carry the same aggregate renewal"
    );
    assert!(
        (agg_bad - 0.5).abs() < 1e-12,
        "both patterns renew half of all substeps; got {agg_bad}"
    );

    let bad = bins_of(&coalesced_at_midpoint);
    let ok = bins_of(&renewed_evenly);

    // (2) Time-resolved, they are not the same series at all. The early window
    //     — the first tenth of the substeps, which with RENEWAL_BINS = 10 is
    //     exactly bin 0 — separates them completely.
    assert_eq!(
        bad[0], 0.0,
        "a sweep whose lineage coalesced onto the reference at the midpoint renews \
         NOTHING in the first tenth of the series; per-bin renewal was {bad:?}"
    );
    assert!(
        (ok[0] - 0.5).abs() < 1e-12,
        "the evenly-renewed sweep renews at its aggregate rate in the first tenth; \
         per-bin renewal was {ok:?}"
    );

    // (3) The whole profile, not just the first bin: the degenerate sweep is a
    //     step function (LJS Fig. 1's PG curve), the healthy one is flat.
    for (b, &r) in bad.iter().enumerate() {
        let expected = if b < RENEWAL_BINS / 2 { 0.0 } else { 1.0 };
        assert_eq!(
            r, expected,
            "bin {b} of the coalesced sweep must be {expected}; profile was {bad:?}"
        );
    }
    for (b, &r) in ok.iter().enumerate() {
        assert!(
            (r - 0.5).abs() < 1e-12,
            "bin {b} of the evenly-renewed sweep must sit at the aggregate 0.5; \
             profile was {ok:?}"
        );
    }
}

/// An empty bin reports `NaN`, not `0.0` — the same convention
/// `CSMCDiagnostics::as_accept_rate` already uses, and for the same reason:
/// "no substep fell in this bin" and "no substep in this bin was renewed" are
/// different diagnoses, and collapsing them invents a degeneracy that was never
/// observed.
#[test]
fn a_bin_with_no_substeps_is_nan_not_zero() {
    // Four substeps over ten bins: six bins are necessarily empty.
    let bins = bins_of(&[true, true, true, true]);
    let n_nan = bins.iter().filter(|r| r.is_nan()).count();
    assert_eq!(
        n_nan,
        RENEWAL_BINS - 4,
        "with 4 substeps and {RENEWAL_BINS} bins, {} bins hold no substep and must read NaN; \
         profile was {bins:?}",
        RENEWAL_BINS - 4
    );
    for (b, &r) in bins.iter().enumerate() {
        assert!(
            r == 0.0 || r.is_nan(),
            "bin {b}: nothing was renewed anywhere, so every non-empty bin is 0.0; got {r}"
        );
    }
}

/// Every substep lands in exactly one bin, and the bins tile the series in
/// order — so the count-weighted mean of the per-bin fractions reproduces the
/// aggregate exactly. This is what lets the two be read side by side.
#[test]
fn bins_tile_the_series_and_reproduce_the_aggregate() {
    for n in [1usize, 7, 8, 10, 11, 80, 399, 400, 1001] {
        // A deterministic but non-uniform pattern, so the reconstruction is not
        // trivially satisfied by a constant.
        let from_ref: Vec<bool> = (0..n).map(|s| (s * s) % 7 < 3).collect();
        let bins = bins_of(&from_ref);
        let sizes = bin_sizes(n);
        assert_eq!(sizes.iter().sum::<usize>(), n, "bins must tile all {n} substeps");

        let renewed: f64 = bins.iter().zip(&sizes)
            .filter(|(r, _)| !r.is_nan())
            .map(|(r, &size)| r * size as f64)
            .sum();
        let expected = aggregate(&from_ref) * n as f64;
        assert!(
            (renewed - expected).abs() < 1e-9,
            "n={n}: per-bin counts sum to {renewed} renewed substeps, aggregate says {expected}; \
             profile was {bins:?}, bin sizes {sizes:?}"
        );
    }
}

// ───────────────────────────── end to end, real model ─────────────────────────

fn poisson_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "weekly_cases".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn model() -> Arc<CompiledModel> {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_overdispersion.ir.json")
        .expect("read sir_overdispersion golden");
    let mut m = ir::from_str(&json).expect("parse");
    m.observations = vec![poisson_obs_block()];
    for p in &mut m.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.name.as_str() {
                "beta" => 0.3,
                "gamma" => 0.1,
                "sigma_se" => 0.1,
                "N0" => 1000.0,
                "I0" => 10.0,
                _ => 0.5,
            };
            p.value = p.value.with_value(v);
        }
    }
    Arc::new(CompiledModel::new(m).expect("compile"))
}

/// One sweep's renewal profile, plus the scalar it must stay consistent with.
struct SweepProfile {
    renewal: f64,
    by_bin: [f64; RENEWAL_BINS],
    n_substeps: usize,
    as_proposed: usize,
    as_accepted: usize,
    /// gh#864: the ancestor-sampling acceptance rate over the same ten bins —
    /// the row the renewal profile is read against.
    as_accept_by_bin: [f64; RENEWAL_BINS],
    /// gh#864: the acceptance *ratio*'s distribution over the sweep's proposals,
    /// and the sample size the two summaries are over.
    as_logalpha_median: f64,
    as_logalpha_near: f64,
    n_as_logalpha: usize,
    as_refused: usize,
}

fn sweeps(n_sweeps: u64, n_particles: usize) -> Vec<SweepProfile> {
    let compiled = model();
    let params = compiled.default_params.clone();
    let t_end = compiled.model.simulation.t_end;

    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, t_end, DT, &mut rng).expect("reference");

    let mut cum: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in reference.substeps.iter().enumerate() {
        cum += rec.flows[0];
        let t = ((s + 1) as f64) * DT;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum as f64 });
            cum = 0;
        }
    }
    assert!(obs.len() >= 3, "need several observation intervals");

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::FlowSum(vec![0]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();

    let obs_at_substep: ObsAtSubstep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs_at_substep");

    (0..n_sweeps)
        .map(|seed| {
            let (_traj, diag) = csmc_as(
                &compiled, &params, &obs, &reference, n_particles, DT, &obs_model,
                SEED + seed, &obs_at_substep, EffectFiring::default(),
                sim::rng::BinomialAlgorithm::Btpe,
                true,
            )
            .expect("csmc_as");
            SweepProfile {
                renewal: diag.trajectory_renewal,
                by_bin: diag.renewal_by_bin,
                n_substeps: diag.n_substeps,
                as_proposed: diag.n_as_proposed,
                as_accepted: diag.n_as_accepted,
                as_accept_by_bin: diag.as_accept_by_bin,
                as_logalpha_median: diag.as_logalpha_median,
                as_logalpha_near: diag.as_logalpha_near,
                n_as_logalpha: diag.n_as_logalpha,
                as_refused: diag.n_as_refused_inadmissible,
            }
        })
        .collect()
}

/// What `csmc_as` returns must be a count of real per-substep decisions.
///
/// Two independent checks, because each catches what the other misses:
///
///  - **Reconstruction.** The per-bin fractions weighted by their bin sizes
///    must sum to the aggregate's renewed-substep count. Catches a vector that
///    is populated from the wrong lineage, or not populated at all.
///  - **Granularity.** Bin `b` is a fraction over the substeps in bin `b`, so
///    it is an exact multiple of `1/size_b`. The aggregate is a multiple of
///    `1/n_substeps`. Rebroadcasting the aggregate into every bin — the exact
///    thing this issue exists to stop — survives reconstruction but violates
///    granularity unless the sweep's aggregate happens to be a multiple of
///    `1/size_b`, which for 80 substeps in bins of 8 is a one-in-ten
///    coincidence per sweep, and would have to hold for EVERY sweep to pass.
#[test]
fn returned_bins_are_per_substep_counts_not_the_scalar_rebroadcast() {
    const SWEEPS: u64 = 12;
    let profiles = sweeps(SWEEPS, 32);
    let mut n_bins_off_the_aggregate = 0usize;

    for (i, p) in profiles.iter().enumerate() {
        let sizes = bin_sizes(p.n_substeps);

        let renewed: f64 = p.by_bin.iter().zip(&sizes)
            .filter(|(r, _)| !r.is_nan())
            .map(|(r, &size)| r * size as f64)
            .sum();
        let expected = p.renewal * p.n_substeps as f64;
        assert!(
            (renewed - expected).abs() < 1e-9,
            "sweep {i}: per-bin renewed substeps sum to {renewed}, but trajectory_renewal \
             {:.6} over {} substeps says {expected}; profile {:?}",
            p.renewal, p.n_substeps, p.by_bin
        );

        for (b, (&r, &size)) in p.by_bin.iter().zip(&sizes).enumerate() {
            if r.is_nan() { continue; }
            let count = r * size as f64;
            assert!(
                (count - count.round()).abs() < 1e-9,
                "sweep {i} bin {b}: renewal {r} over {size} substeps is not a whole number of \
                 substeps ({count}) — the bin is not counting substeps; profile {:?}",
                p.by_bin
            );
            if (r - p.renewal).abs() > 1e-12 {
                n_bins_off_the_aggregate += 1;
            }
        }
    }

    assert!(
        n_bins_off_the_aggregate > 0,
        "across {SWEEPS} sweeps not one bin differed from its sweep's aggregate renewal — \
         the per-bin vector carries no time resolution at all"
    );
}

/// gh#864: what `csmc_as` returns for the acceptance profile must be a record
/// of real per-substep Metropolis decisions, tied to the counters the sweep
/// reports for the same decisions.
///
/// Four ties, each catching what the others miss:
///
///  - **A bin is measured only where a move was proposed.** A sweep proposing
///    `k` moves cannot measure more than `k` bins. Catches a profile recorded
///    at every ancestor-sampling step rather than at the proposals — which
///    would enter "no alternative was proposed here" as a rejection and
///    manufacture a low early profile out of nothing.
///  - **Measured at all iff the step ran at all.** A sweep with no proposal
///    has no measured bin, and one with a proposal has at least one.
///  - **Containment.** The sweep's pooled rate is a weighted mean of its bin
///    rates, so it lies between the smallest and the largest of them. Catches a
///    profile populated from the wrong decision.
///  - **Granularity.** A profile that simply rebroadcast the sweep's scalar
///    into every bin would pass containment; across sweeps at least one bin
///    must differ from its own sweep's rate.
#[test]
fn the_acceptance_profile_is_a_record_of_real_proposals() {
    const SWEEPS: u64 = 12;
    let profiles = sweeps(SWEEPS, 32);
    let (mut any_measured, mut n_off_the_scalar) = (false, 0usize);

    for (i, p) in profiles.iter().enumerate() {
        let measured: Vec<usize> = p.as_accept_by_bin.iter().enumerate()
            .filter(|(_, v)| v.is_finite()).map(|(b, _)| b).collect();
        assert!(measured.len() <= p.as_proposed,
            "sweep {i}: {} bins carry an acceptance rate but only {} moves were \
             proposed all sweep — a bin needs at least one proposal to be \
             measured; profile {:?}",
            measured.len(), p.as_proposed, p.as_accept_by_bin);
        assert_eq!(measured.is_empty(), p.as_proposed == 0,
            "sweep {i}: {} proposals and {} measured bins — a sweep that \
             proposed nothing must measure nothing, and one that proposed \
             something must measure somewhere; profile {:?}",
            p.as_proposed, measured.len(), p.as_accept_by_bin);
        if p.as_proposed == 0 { continue; }
        any_measured = true;

        let pooled = p.as_accepted as f64 / p.as_proposed as f64;
        let (lo, hi) = p.as_accept_by_bin.iter().filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY),
                  |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(lo - 1e-12 <= pooled && pooled <= hi + 1e-12,
            "sweep {i}: the sweep rate {pooled:.4} ({}/{}) must lie inside the \
             bins it averages [{lo}, {hi}]; profile {:?}",
            p.as_accepted, p.as_proposed, p.as_accept_by_bin);
        for &v in p.as_accept_by_bin.iter().filter(|v| v.is_finite()) {
            assert!((0.0..=1.0).contains(&v),
                "sweep {i}: an acceptance rate must lie in [0,1]; profile {:?}",
                p.as_accept_by_bin);
            if (v - pooled).abs() > 1e-12 { n_off_the_scalar += 1; }
        }
    }

    assert!(any_measured,
        "this fixture is only a test of the profile while ancestor sampling \
         proposes moves; no sweep of {SWEEPS} proposed one");
    assert!(n_off_the_scalar > 0,
        "across {SWEEPS} sweeps not one bin differed from its sweep's pooled \
         acceptance rate — the profile carries no positional resolution at all");
}

/// gh#864: the acceptance ratio's distribution, tied to the counters that
/// account for the same proposals.
///
/// The sample is the proposals whose ratio is a finite number — both suffix
/// densities positive, so the coin actually decided. The proposals refused for
/// carrying zero suffix density have no ratio and are counted separately, so
/// the two counts can never together exceed the proposals made. Publishing the
/// sample size is what lets a reader take `as_logalpha_near` at face value: a
/// fraction of 0.6 measured over 5% of a sweep's proposals is a different
/// statement from one measured over all of them.
#[test]
fn the_acceptance_ratio_sample_is_accounted_for_against_the_proposals() {
    const SWEEPS: u64 = 12;
    let profiles = sweeps(SWEEPS, 32);
    let mut any_measured = false;

    for (i, p) in profiles.iter().enumerate() {
        assert!(p.n_as_logalpha <= p.as_proposed,
            "sweep {i}: {} finite ratios recorded over {} proposals — the \
             sample cannot be larger than what was proposed",
            p.n_as_logalpha, p.as_proposed);
        assert!(p.n_as_logalpha + p.as_refused <= p.as_proposed,
            "sweep {i}: {} proposals with a finite ratio plus {} refused as \
             zero-density exceed the {} proposed — every proposal is one or \
             the other (or the zero-density-reference escape, which a chain \
             inside the support cannot take)",
            p.n_as_logalpha, p.as_refused, p.as_proposed);

        let measured = p.n_as_logalpha > 0;
        assert_eq!(p.as_logalpha_median.is_finite(), measured,
            "sweep {i}: the median is a number exactly when the sample is \
             non-empty ({} recorded, median {})",
            p.n_as_logalpha, p.as_logalpha_median);
        assert_eq!(p.as_logalpha_near.is_finite(), measured,
            "sweep {i}: and so is the fraction near parity ({} recorded, \
             near {})", p.n_as_logalpha, p.as_logalpha_near);
        if measured {
            any_measured = true;
            assert!((0.0..=1.0).contains(&p.as_logalpha_near),
                "sweep {i}: a fraction must lie in [0,1], got {}",
                p.as_logalpha_near);
        }
    }

    assert!(any_measured,
        "this fixture is only a test of the ratio while proposals reach the \
         Metropolis step with a finite suffix ratio; none of {SWEEPS} sweeps \
         did");
}

/// Negative control: on a healthy sweep — one where ancestor sampling is
/// proposing and accepting splices — renewal is roughly uniform in t. No bin
/// is starved relative to the series as a whole, which is the flat LJS
/// Figure 1 PGAS curve and the opposite of the pathology above.
///
/// Reported next to the ancestor-sampling acceptance counters: renewal-by-time
/// says WHERE the path is stuck, the acceptance rate says WHY.
#[test]
fn a_healthy_sweep_renews_roughly_uniformly_in_time() {
    const SWEEPS: u64 = 16;
    let profiles = sweeps(SWEEPS, 32);

    let mut bin_sum = [0.0f64; RENEWAL_BINS];
    let mut bin_n = [0usize; RENEWAL_BINS];
    let (mut renewal_sum, mut proposed, mut accepted) = (0.0f64, 0usize, 0usize);
    for p in &profiles {
        for b in 0..RENEWAL_BINS {
            if !p.by_bin[b].is_nan() {
                bin_sum[b] += p.by_bin[b];
                bin_n[b] += 1;
            }
        }
        renewal_sum += p.renewal;
        proposed += p.as_proposed;
        accepted += p.as_accepted;
    }
    let mean_renewal = renewal_sum / profiles.len() as f64;
    let mean_by_bin: Vec<f64> = (0..RENEWAL_BINS)
        .map(|b| bin_sum[b] / bin_n[b].max(1) as f64)
        .collect();

    // gh#864: the acceptance profile over the same bins, printed beside the
    // renewal one because that pairing is how either is read.
    let mut acc_sum = [0.0f64; RENEWAL_BINS];
    let mut acc_n = [0usize; RENEWAL_BINS];
    for p in &profiles {
        for b in 0..RENEWAL_BINS {
            if p.as_accept_by_bin[b].is_finite() {
                acc_sum[b] += p.as_accept_by_bin[b];
                acc_n[b] += 1;
            }
        }
    }
    let as_accept_by_bin: Vec<String> = (0..RENEWAL_BINS)
        .map(|b| if acc_n[b] > 0 {
            format!("{:.3}", acc_sum[b] / acc_n[b] as f64)
        } else {
            "   NA".to_string()
        })
        .collect();

    eprintln!(
        "renewal by time bin ({SWEEPS} sweeps, {} substeps): {:?}",
        profiles[0].n_substeps,
        mean_by_bin.iter().map(|r| format!("{r:.3}")).collect::<Vec<_>>()
    );
    eprintln!("AS acceptance by time bin (sweeps measuring each: {acc_n:?}): {as_accept_by_bin:?}");
    eprintln!("  aggregate renewal {mean_renewal:.3} | AS proposed {proposed} accepted {accepted}");
    // gh#864: and the ratio behind those decisions, over the sweeps that
    // measured one.
    let logalpha: Vec<f64> = profiles.iter()
        .map(|p| p.as_logalpha_median).filter(|v| v.is_finite()).collect();
    let near: Vec<f64> = profiles.iter()
        .map(|p| p.as_logalpha_near).filter(|v| v.is_finite()).collect();
    let n_ratio: usize = profiles.iter().map(|p| p.n_as_logalpha).sum();
    let n_refused: usize = profiles.iter().map(|p| p.as_refused).sum();
    eprintln!(
        "  AS log-alpha: per-sweep medians {:?} | fractions above -1 {:?}",
        logalpha.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>(),
        near.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>(),
    );
    eprintln!("  over {n_ratio} proposals with a finite ratio, {n_refused} refused as zero-density");

    assert!(
        accepted > 0,
        "this fixture is only a negative control while ancestor sampling is working: \
         {proposed} proposals, {accepted} accepted"
    );
    assert!(
        mean_renewal > 0.2,
        "this fixture is only a negative control while the sampler renews: aggregate \
         renewal was {mean_renewal:.3}"
    );
    for (b, &r) in mean_by_bin.iter().enumerate() {
        assert!(
            r > 0.5 * mean_renewal,
            "bin {b} renews at {r:.3}, less than half the series-wide {mean_renewal:.3} — \
             this fixture is no longer uniformly renewed; profile {mean_by_bin:?}"
        );
    }
}
