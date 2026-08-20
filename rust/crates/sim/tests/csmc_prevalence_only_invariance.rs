//! gh#607, blast-radius bound: the interval-accumulator re-sync must be INERT
//! for a model with no Interval (incidence) stream.
//!
//! When an ancestor-sampling splice reassigns the reference slot, that slot's
//! per-transition `cum_flows` and per-stream `acc` bins are re-synced to the
//! sampled ancestor's, because a stream that scores flows summed over
//! `(previous observation, this one]` carries that partial sum as part of its
//! state. A model observed only on PREVALENCE (`CurrentPop`, read at the
//! observation instant) has no such bin: `acc` is zero-length and `cum_flows`
//! reaches the likelihood only through `fold_into_acc`, which writes nothing.
//! The re-sync must therefore not move a single trajectory on such a model.
//!
//! Two independent checks, because the structural one alone is nearly
//! tautological and the digest alone cannot say WHY it holds:
//!
//! 1. the seam — the bound observation model reports zero Interval streams, and
//!    folding arbitrary flows into its (empty) accumulator leaves the
//!    likelihood untouched;
//! 2. the digest — a full `csmc_as` sweep over this model reproduces the exact
//!    trajectory measured before the re-sync landed.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, EffectFiring, ObsAtSubstep, PGASTrajectory,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260819;
/// Local index of `I` in `sir_overdispersion` (compartments are `S, I, R`).
const I_IDX: usize = 1;

fn prevalence_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "prevalence".into(),
        source: "prevalence".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "prevalence".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "prevalence".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        // CurrentPop is the Instant (prevalence) kind — no accumulation bin.
        projection: Projection::CurrentPop("I".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn model() -> Arc<CompiledModel> {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_overdispersion.ir.json")
        .expect("read sir_overdispersion golden");
    let mut m = ir::from_str(&json).expect("parse");
    m.observations = vec![prevalence_obs_block()];
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

/// Every integer in the trajectory, in order — `initial_counts` then each
/// substep's `counts_before`, `counts_after` and `flows`. A single changed
/// draw anywhere moves this.
fn digest(traj: &PGASTrajectory) -> u64 {
    let mut h = DefaultHasher::new();
    traj.initial_counts.hash(&mut h);
    for rec in &traj.substeps {
        rec.counts_before.hash(&mut h);
        rec.counts_after.hash(&mut h);
        rec.flows.hash(&mut h);
    }
    h.finish()
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    reference: PGASTrajectory,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
}

fn fixture() -> Fixture {
    let compiled = model();
    let params = compiled.default_params.clone();
    let t_end = compiled.model.simulation.t_end;

    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, t_end, DT, &mut rng).expect("reference");

    // Weekly prevalence readings taken off the reference path.
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in reference.substeps.iter().enumerate() {
        let t = ((s + 1) as f64) * DT;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: rec.counts_after[I_IDX] as f64 });
        }
    }
    assert!(obs.len() >= 3, "need several observations");

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![I_IDX]),
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

    Fixture { compiled, params, reference, obs, obs_model, obs_at_substep }
}

/// (1) The seam: with no Interval stream there is no accumulator to re-sync,
/// and the per-transition flow tally cannot reach the likelihood.
#[test]
fn a_prevalence_only_model_has_no_interval_accumulator() {
    let f = fixture();
    assert_eq!(
        f.obs_model.n_interval_streams(),
        0,
        "a CurrentPop stream is Instant, not Interval — the fixture is wrong if this fires"
    );

    // The re-sync writes `cum_flows[j_ref]`; `fold_into_acc` is the ONLY route
    // from that tally to a likelihood. With no Interval stream it writes
    // nothing, so no flow tally — however corrupted — can move the score.
    let n_tr = f.compiled.model.transitions.len();
    let counts = &f.reference.substeps[10].counts_after;
    let mut acc_zero: Vec<u64> = vec![0; f.obs_model.n_interval_streams()];
    let mut acc_wild: Vec<u64> = vec![0; f.obs_model.n_interval_streams()];
    f.obs_model.fold_into_acc(&vec![0u64; n_tr], &mut acc_zero);
    f.obs_model.fold_into_acc(&vec![u64::MAX / 2; n_tr], &mut acc_wild);
    assert_eq!(acc_zero, acc_wild, "folding flows must be a no-op with no Interval stream");

    let ll_zero =
        f.obs_model.log_likelihood_from_flows_and_counts(&acc_zero, counts, 0, &f.params);
    let ll_wild =
        f.obs_model.log_likelihood_from_flows_and_counts(&acc_wild, counts, 0, &f.params);
    assert!(ll_zero.is_finite(), "negative control: the fixture must score finitely");
    assert_eq!(ll_zero, ll_wild, "prevalence likelihood must not read the flow accumulator");
}

/// (2) The digest: a full CSMC sweep on this model reproduces the trajectory
/// measured immediately before the interval-accumulator re-sync landed
/// (`pgas.rs` at the parent of the gh#607 accumulator commit). A change here is
/// either a regression in that claim or an intended change to the CSMC draw
/// order — never something to re-baseline without deciding which.
///
/// NOT vacuous: the re-sync sits behind `ref_ancestor != j_ref`, and these four
/// sweeps accept **83 splices** over 4 × 80 = 320 ancestor-sampling
/// opportunities (counted by instrumenting that branch on 2026-08-19). The code
/// under test runs 83 times and moves nothing.
#[test]
fn the_interval_accumulator_resync_does_not_move_a_prevalence_only_trajectory() {
    const EXPECTED: [u64; 4] = [
        0xf9ca0a01894e9c7d,
        0x7463433398e370e5,
        0xbc73cf25769a82f2,
        0x92af74721aa5e3c7,
    ];
    let f = fixture();
    let got: Vec<u64> = (0..4u64)
        .map(|seed| {
            let (traj, _diag) = csmc_as(
                &f.compiled,
                &f.params,
                &f.obs,
                &f.reference,
                32,
                DT,
                &f.obs_model,
                &[],
                SEED + seed,
                &f.obs_at_substep,
                EffectFiring::default(),
            )
            .expect("csmc_as");
            digest(&traj)
        })
        .collect();
    assert_eq!(
        got,
        EXPECTED.to_vec(),
        "prevalence-only trajectories moved: got {got:#x?}, expected {EXPECTED:#x?}"
    );
}
