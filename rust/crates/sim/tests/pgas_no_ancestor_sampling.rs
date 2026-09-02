//! `ancestor_sampling = false` must actually disable the move.
//!
//! Plain particle Gibbs without ancestor sampling (Andrieu, Doucet &
//! Holenstein 2010) is a valid kernel — AS is the Lindsten–Jordan–Schön
//! mixing addition — and camdl exposes the switch as a diagnostic control.
//! These tests pin that the flag reaches `csmc_as`: with it off, no ancestor
//! is ever proposed, the reference keeps its own ancestry at every substep,
//! and the starvation instrument reads "no data" rather than zero. If the
//! flag is dropped anywhere between the config and the AS gate, the off-run
//! proposes ancestors like the on-run and these assertions go red.

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
const SEED: u64 = 20260902;
const N_SUBSTEPS: usize = 30;
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
        projection: Projection::CurrentPop("I".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn model() -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = N_SUBSTEPS as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.5,
            "gamma" => 0.1,
            "N0" => 1000.0,
            "I0" => 50.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
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
    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, N_SUBSTEPS as f64 * DT, DT, &mut rng)
        .expect("reference");

    // Observations at every third substep, read off the reference itself so
    // weights differ across particles (forcing resampling — the AS
    // opportunity) without collapsing.
    let obs_substeps: Vec<usize> = (2..N_SUBSTEPS).step_by(3).collect();
    let obs: Vec<Observation> = obs_substeps
        .iter()
        .map(|&s| Observation {
            time: ((s + 1) as f64) * DT,
            value: reference.substeps[s].counts_after[I_IDX] as f64,
        })
        .collect();
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

fn sweep(f: &Fixture, ancestor_sampling: bool, seed: u64) -> sim::inference::pgas::CSMCDiagnostics {
    let (_traj, diag) = csmc_as(
        &f.compiled,
        &f.params,
        &f.obs,
        &f.reference,
        64,
        DT,
        &f.obs_model,
        seed,
        &f.obs_at_substep,
        EffectFiring::default(),
        ancestor_sampling,
    )
    .expect("csmc_as");
    diag
}

#[test]
fn as_off_never_proposes_and_reports_no_data() {
    let f = fixture();
    for seed in [1u64, 2, 3] {
        let diag = sweep(&f, false, seed);
        assert!(diag.n_resampled > 0,
            "the fixture must resample (seed {seed}), or the test is vacuous");
        assert_eq!(diag.n_as_proposed, 0,
            "AS off must never propose an ancestor (seed {seed})");
        assert_eq!(diag.n_as_accepted, 0, "nothing proposed, nothing accepted");
        assert!(diag.as_finite_frac.is_nan() && diag.as_admissible_frac.is_nan(),
            "the starvation instrument must read 'no data' (NaN), not 0.0, \
             when the density pass never ran (seed {seed})");
        assert_eq!(diag.n_as_starved, 0);
    }
}

#[test]
fn as_on_exercises_the_move_on_the_same_fixture() {
    // The positive control: same fixture, AS on. Without this, the off-test
    // could pass on a fixture where AS never fires anyway.
    let f = fixture();
    let fired = [1u64, 2, 3].iter().any(|&seed| {
        let diag = sweep(&f, true, seed);
        diag.n_as_proposed > 0
    });
    assert!(fired,
        "AS on must propose at least once across three seeds on this fixture — \
         if it cannot, the off-test above is not testing the switch");
}
