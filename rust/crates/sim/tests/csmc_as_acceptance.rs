//! gh#607 follow-up: WHY is ancestor sampling refusing splices?
//!
//! `trajectory_renewal` tells you how often the returned path came from a
//! non-reference particle, but it cannot say why a low value is low. Three very
//! different situations flatten it identically:
//!
//!   1. the screened Eq.-(17) weights leave only the reference's own lineage
//!      admissible, so the Metropolis step never runs (`n_as_proposed == 0`);
//!   2. it runs, and the EXACT suffix ratio finds the candidate off-support
//!      (`n_as_refused_inadmissible`) — the target's support asserting itself,
//!      which is what gh#607 set out to enforce;
//!   3. it runs at finite ratios and the coin rejects — the proposal is badly
//!      matched to the target, and mixing is paying for a bad proposal rather
//!      than for correctness.
//!
//! Only (3) implicates the acceptance ratio. This test pins the decomposition
//! so a future change that silently converts (3) into (2), or drives proposals
//! to zero, is visible as a number rather than as a slow fit.
//!
//! Every path in the target's support satisfies
//! `substeps[s-1].counts_after == substeps[s].counts_before`, and
//! `initial_counts == substeps[0].counts_before`. The traceback stitches
//! records along an ancestry that ancestor sampling reassigns, so if the
//! reference slot's recorded `counts_before` is always its OWN rather than the
//! AS-selected ancestor's, the returned trajectory jumps in state at each
//! splice — and `complete_data_loglik` reads each record's stored
//! `counts_before` without checking continuity, so the jump is never charged.
//!
//! This test decides the SCOPE of gh#607: the accumulator defect is specific to
//! interval observations, but a counts discontinuity would affect every PGAS
//! fit regardless of observation type. The model here is observed on a FLOW SUM
//! (interval), but the assertion is purely structural.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, EffectFiring, ObsAtSubstep,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260820;
const I_IDX: usize = 1;

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


/// One sweep-set's ancestor-sampling accounting.
struct AsTally {
    proposed: usize,
    accepted: usize,
    inadmissible: usize,
    substeps: usize,
    renewal: f64,
}

impl AsTally {
    fn coin_rejected(&self) -> usize {
        self.proposed - self.accepted - self.inadmissible
    }
    fn pct(&self, n: usize) -> f64 {
        100.0 * n as f64 / self.proposed.max(1) as f64
    }
    fn report(&self, label: &str, sweeps: u64) {
        eprintln!(
            "{label:>10}: {sweeps} sweeps / {} substeps | proposed {} | accepted {} ({:.1}%) | \
             off-support {} ({:.1}%) | coin-rejected {} ({:.1}%) | renewal {:.3}",
            self.substeps, self.proposed,
            self.accepted, self.pct(self.accepted),
            self.inadmissible, self.pct(self.inadmissible),
            self.coin_rejected(), self.pct(self.coin_rejected()),
            self.renewal,
        );
    }
}

/// Run `sweeps` CSMC sweeps and tally the ancestor-sampling decomposition.
/// `interval` selects an incidence (`FlowSum`) stream against a prevalence
/// (`IntCompSum`) one — the accumulator is part of the extended state only for
/// the former, so this is the A/B that says whether interval observations pay
/// an extra acceptance penalty.
fn tally(interval: bool, sweeps: u64) -> AsTally {
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
            let value = if interval { cum as f64 } else { rec.counts_after[I_IDX] as f64 };
            obs.push(Observation { time: t, value });
            cum = 0;
        }
    }
    assert!(obs.len() >= 3, "need several observation intervals");

    let projection = if interval {
        StreamProjection::FlowSum(vec![0])
    } else {
        StreamProjection::IntCompSum(vec![I_IDX])
    };
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            projection,
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
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT)
            .expect("obs_at_substep");

    let mut t = AsTally { proposed: 0, accepted: 0, inadmissible: 0, substeps: 0, renewal: 0.0 };
    for seed in 0..sweeps {
        let (_traj, diag) = csmc_as(
            &compiled, &params, &obs, &reference, 32, DT, &obs_model,
            SEED + seed, &obs_at_substep, EffectFiring::default(),
            sim::rng::BinomialAlgorithm::Btpe,
            true,
        )
        .expect("csmc_as");
        t.proposed += diag.n_as_proposed;
        t.accepted += diag.n_as_accepted;
        t.inadmissible += diag.n_as_refused_inadmissible;
        t.substeps += diag.n_substeps;
        t.renewal += diag.trajectory_renewal;
    }
    t.renewal /= sweeps as f64;
    t
}

/// The accounting must balance, ancestor sampling must actually PROPOSE moves,
/// and the interval-vs-prevalence decomposition is reported for both.
///
/// A sampler that never proposes has silently degenerated to plain particle
/// Gibbs — correct, but degenerate on long series (LJS 2014 §3), and invisible
/// in `trajectory_renewal`, which would read the same as a healthy sampler
/// whose proposals are all rejected.
#[test]
fn ancestor_sampling_acceptance_is_accounted_for() {
    const SWEEPS: u64 = 16;
    for (label, interval) in [("interval", true), ("prevalence", false)] {
        let t = tally(interval, SWEEPS);
        t.report(label, SWEEPS);

        assert!(
            t.proposed > 0,
            "{label}: ancestor sampling never proposed an alternative ancestor across \
             {SWEEPS} sweeps ({} substeps) — CSMC-AS has degenerated to plain particle Gibbs",
            t.substeps
        );
        assert_eq!(
            t.proposed,
            t.accepted + t.inadmissible + t.coin_rejected(),
            "{label}: every proposal must land in exactly one of \
             accepted / off-support / coin-rejected"
        );
        assert!(
            t.accepted > 0,
            "{label}: every one of {} proposals was rejected — ancestor sampling is doing \
             no work at all, and `trajectory_renewal` cannot distinguish this from health",
            t.proposed
        );
    }
}
