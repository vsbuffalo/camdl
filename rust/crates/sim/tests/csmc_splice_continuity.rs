//! gh#607 scope test: is the trajectory `csmc_as` returns a CONTINUOUS model
//! path?
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
const SEED: u64 = 20260819;

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

/// The oracle for gh#607's red→green. Measured red on 2026-08-19 at b77c47df:
/// 4 discontinuities across 8 CSMC sweeps on a plain SIR. Green once the
/// ancestor-sampling splice re-anchors the reference slot on the ancestor it
/// was assigned.
#[test]
fn csmc_returns_a_continuous_path() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let t_end = compiled.model.simulation.t_end;

    // A reference path, and weekly observations taken from it.
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
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT)
            .expect("obs_at_substep");

    // Many particles + several seeds: ancestor sampling must actually fire and
    // pick something other than the reference for the splice to be exercised.
    let mut jumps = 0usize;
    let mut init_mismatch = 0usize;
    let mut sweeps = 0usize;
    for seed in 0..8u64 {
        let (traj, _diag) = csmc_as(
            &compiled,
            &params,
            &obs,
            &reference,
            32,
            DT,
            &obs_model,
            SEED + seed,
            &obs_at_substep,
            EffectFiring::default(),
        )
        .expect("csmc_as");
        sweeps += 1;

        if traj.initial_counts != traj.substeps[0].counts_before {
            init_mismatch += 1;
        }
        for s in 1..traj.substeps.len() {
            if traj.substeps[s - 1].counts_after != traj.substeps[s].counts_before {
                jumps += 1;
            }
        }
    }

    assert!(sweeps > 0);
    assert_eq!(
        jumps, 0,
        "the returned trajectory must be a CONTINUOUS model path: found {jumps} substeps \
         where counts_after(s-1) != counts_before(s) across {sweeps} CSMC sweeps. A state \
         jump at an ancestor-sampling splice is never charged by complete_data_loglik, so \
         the sampler returns points outside the target's support (gh#607)."
    );
    assert_eq!(
        init_mismatch, 0,
        "initial_counts must equal substeps[0].counts_before: mismatched on \
         {init_mismatch}/{sweeps} sweeps (biases IVP posteriors — gh#607)"
    );
}
