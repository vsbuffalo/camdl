//! The selected binomial sampler must reach the draws — every one of them.
//!
//! The `binomial` knob travels **on the RNG** from the typed stage field to
//! every draw site (proposal `2026-08-24-faster-binomial-sampler.md` §1). The
//! draws happen on rayon workers inside nested `par_iter`s, where a
//! thread-local set by the configuring thread is invisible — measured at 0 of
//! 4096 particle draws — so the selection is stamped onto each particle's RNG
//! at construction and travels with the particle.
//!
//! That makes the failure mode specific: a construction site that forgets
//! `.with_binomial()` and silently falls back to BTPE. `init_particle_rngs`,
//! `run_pgas`'s reference RNG and `unconditional_smc_pass` are those sites.
//! If any of them drops it, the BTRS run below becomes byte-identical to the
//! BTPE run and the inequality assertions go red.
//!
//! BTPE and BTRS both sample Binomial(n, p) exactly, so "different draws from
//! the same stream" is the *expected* behaviour of two different rejection
//! schemes, not a distributional claim — distributional equivalence is pinned
//! in `sim::rng`'s own suite.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, EffectFiring, ObsAtSubstep, PGASTrajectory,
};
use sim::rng::{BinomialAlgorithm, StatefulRng};

const DT: f64 = 1.0;
const SEED: u64 = 20260901;
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

/// `sir_basic` at a population and prevalence that keep `n·p` above
/// `BINV_THRESHOLD` on the S-exit draw, so the BTPE/BTRS fork is actually
/// exercised (below the threshold both route to BINV and draw identically).
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
            "beta" => 0.6,
            "gamma" => 0.08,
            "N0" => 5000.0,
            "I0" => 400.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

fn reference_with(algo: BinomialAlgorithm) -> PGASTrajectory {
    let compiled = model();
    let params = compiled.default_params.clone();
    // RNG-carrier design: the selection rides on the RNG rather than being
    // threaded as a parameter, so the failure mode this pins is a construction
    // site that FORGETS `.with_binomial()` and silently draws BTPE.
    let mut rng = StatefulRng::new(SEED).with_binomial(algo);
    simulate_reference(&compiled, &params, N_SUBSTEPS as f64 * DT, DT, &mut rng)
        .expect("reference")
}

/// The producer path: `simulate_reference` → `step_one`.
#[test]
fn selected_sampler_reaches_the_reference_producer() {
    let btpe = reference_with(BinomialAlgorithm::Btpe);
    let btpe_again = reference_with(BinomialAlgorithm::Btpe);
    let btrs = reference_with(BinomialAlgorithm::Btrs);

    assert_eq!(
        btpe.substeps.last().unwrap().counts_after,
        btpe_again.substeps.last().unwrap().counts_after,
        "same sampler, same seed must be byte-identical"
    );
    let flows_differ = btpe
        .substeps
        .iter()
        .zip(btrs.substeps.iter())
        .any(|(a, b)| a.flows != b.flows);
    assert!(
        flows_differ,
        "BTRS produced the same trajectory as BTPE from the same stream — the \
         selected sampler is not reaching the propagation draws (the threaded \
         argument is being dropped somewhere between the config and step_one)"
    );
}

/// The CSMC free-particle path: `csmc_as` → parallel `step_one`. This is the
/// path a thread-local could NOT reach (rayon workers), which is why the value
/// threading exists at all.
#[test]
fn selected_sampler_reaches_csmc_free_particles() {
    let compiled = model();
    let params = compiled.default_params.clone();
    let reference = reference_with(BinomialAlgorithm::Btpe);

    let obs_substeps = [9usize, 19, 29];
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

    let sweep = |algo: BinomialAlgorithm| -> PGASTrajectory {
        let (traj, _diag) = csmc_as(
            &compiled,
            &params,
            &obs,
            &reference,
            32,
            DT,
            &obs_model,
            SEED,
            &obs_at_substep,
            EffectFiring::default(),
            algo,
            true,
        )
        .expect("csmc_as");
        traj
    };

    let btpe = sweep(BinomialAlgorithm::Btpe);
    let btpe_again = sweep(BinomialAlgorithm::Btpe);
    let btrs = sweep(BinomialAlgorithm::Btrs);

    let flows =
        |t: &PGASTrajectory| t.substeps.iter().map(|s| s.flows.clone()).collect::<Vec<_>>();
    assert_eq!(flows(&btpe), flows(&btpe_again), "same sampler, same seed: identical sweep");
    assert_ne!(
        flows(&btpe),
        flows(&btrs),
        "BTRS free particles walked exactly BTPE's paths — the sampler \
         selection is not reaching csmc_as's parallel step_one draws"
    );
}
