//! gh#718 defect 2: ancestor sampling may run ONLY where a resampling actually
//! drew a new ancestry.
//!
//! # The property, and the two cases it splits into
//!
//! One `csmc_as` sweep must leave `p(X | θ, y)` invariant. The ancestor-sampling
//! move — detaching the reference trajectory from its own past and re-attaching
//! it to another particle's — is a Metropolis step on the ancestry, and a
//! Metropolis step is only valid on a configuration the surrounding distribution
//! gives positive probability to. So each substep falls into exactly one case:
//!
//! **Case A — an ancestry was drawn.** The incoming weights differ, so
//! `conditional_multinomial_resample` ran: each free slot picked independently
//! from `categorical(W)` over the whole ensemble, the reference included. Every
//! ancestry vector that assigns the reference slot to itself and any particle to
//! any free slot has positive probability, so moving the reference's own entry
//! to some other particle stays inside the support. The ancestor move is legal,
//! and because the picks are independent across slots, the resampling law
//! cancels out of the accept/reject ratio, which is what makes the shipped ratio
//! the right one.
//!
//! **Case B — no ancestry was drawn.** The incoming weights are all equal, so
//! resampling is skipped and every particle keeps its own history: the ancestry
//! is the identity, with probability one. Moving the reference's entry off
//! itself produces a vector of probability ZERO — a configuration the sweep
//! cannot have generated. No Metropolis ratio rescues that; the move is outside
//! the support. So ancestor sampling must not run.
//!
//! There is no third case, and the two are distinguished by exactly one fact:
//! did step 1 draw an ancestry? Not "does this substep carry an observation" —
//! see `an_observation_whose_weights_tie_still_suppresses_the_move` below.
//!
//! # What these tests pin
//!
//! The invariance measurement itself lives in `csmc_exact_invariance`. These are
//! the deterministic complements: they assert the gate's behaviour directly,
//! without a Monte-Carlo argument, so a regression is a hard failure rather than
//! a shifted statistic.

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
const SEED: u64 = 20260823;
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

fn model(t_end: f64) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = t_end;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.35,
            "gamma" => 0.12,
            "N0" => 400.0,
            "I0" => 12.0,
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

/// `obs_values` is keyed by SUBSTEP index; substep `s` spans `(s·dt, (s+1)·dt]`,
/// so an observation attached to substep `s` sits at time `(s+1)·dt`.
fn fixture(n_substeps: usize, obs_substeps: &[usize], value_of: impl Fn(&[i64]) -> f64) -> Fixture {
    let compiled = model(n_substeps as f64 * DT);
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, n_substeps as f64 * DT, DT, &mut rng)
        .expect("reference");
    assert_eq!(reference.substeps.len(), n_substeps, "grid is not what the test assumes");

    let obs: Vec<Observation> = obs_substeps
        .iter()
        .map(|&s| Observation {
            time: ((s + 1) as f64) * DT,
            value: value_of(&reference.substeps[s].counts_after),
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

fn sweep(f: &Fixture, seed: u64) -> (PGASTrajectory, sim::inference::pgas::CSMCDiagnostics) {
    csmc_as(
        &f.compiled,
        &f.params,
        &f.obs,
        &f.reference,
        16,
        DT,
        &f.obs_model,
        seed,
        &f.obs_at_substep,
        EffectFiring::default(),
        sim::rng::BinomialAlgorithm::Btpe,
        true,
    )
    .expect("csmc_as")
}

/// Case B, deterministically: a sweep whose ONLY observation is terminal never
/// draws an ancestry at all, because the weights an observation produces are
/// consumed by the following substep and there is no following substep. So
/// ancestor sampling must never propose anywhere in the sweep.
///
/// This is the strongest available statement of the gate, and it doubles as the
/// edge case in the timing rule below.
///
/// Note what is NOT asserted: that the returned trajectory is the reference.
/// The terminal weights still drive the final trajectory draw, so a free
/// particle can legitimately be selected — that is ordinary conditional-SMC
/// behaviour, not a renewal by ancestor sampling. What must hold is that the
/// returned path is a single unbroken lineage.
#[test]
fn a_sweep_with_only_a_terminal_observation_never_proposes_a_move() {
    const N: usize = 12;
    let f = fixture(N, &[N - 1], |c| c[I_IDX] as f64);
    for seed in 0..8u64 {
        let (traj, d) = sweep(&f, SEED + seed);
        assert_eq!(
            d.n_resampled, 0,
            "the only observation is terminal, so its weights can never drive a resampling"
        );
        assert_eq!(
            d.n_as_skipped_no_resample, d.n_substeps,
            "every substep must have skipped ancestor sampling"
        );
        assert_eq!(d.n_as_proposed, 0, "an ancestor move was proposed with no ancestry drawn");
        assert_eq!(d.n_as_accepted, 0, "an ancestor move was ACCEPTED with no ancestry drawn");

        // The returned path is one particle's own history, start to finish.
        for w in traj.substeps.windows(2) {
            assert_eq!(
                w[0].counts_after, w[1].counts_before,
                "the returned trajectory jumps state — it is not a single lineage"
            );
        }
        assert_eq!(
            traj.initial_counts, traj.substeps[0].counts_before,
            "the returned trajectory does not start where it says it starts"
        );
    }
}

/// The timing rule, stated as a test because it is off by one from the obvious
/// reading: weights produced at the END of substep `s` are consumed by substep
/// `s + 1`. So the number of substeps that resample is the number of
/// observations that have a substep AFTER them — a TERMINAL observation sets
/// weights that only ever feed the final trajectory draw and never a resampling.
#[test]
fn a_terminal_observation_never_drives_a_resampling() {
    // Ten substeps, observations at substeps 3 and 9. Substep 9 is the last, so
    // its weights are terminal.
    let f = fixture(10, &[3, 9], |c| c[I_IDX] as f64);
    let (d_terminal, _) = {
        let (t, d) = sweep(&f, SEED);
        (d, t)
    };
    assert_eq!(
        d_terminal.n_resampled, 1,
        "two observations, but the terminal one cannot drive a resampling — expected 1, \
         got {}. If this reads 2, the weight lifecycle changed and every mixing estimate \
         based on it is off by one.",
        d_terminal.n_resampled
    );
    assert_eq!(d_terminal.n_as_skipped_no_resample, 9);

    // Move the second observation off the end: now BOTH drive a resampling.
    let g = fixture(10, &[3, 7], |c| c[I_IDX] as f64);
    let (d_interior, _) = {
        let (t, d) = sweep(&g, SEED);
        (d, t)
    };
    assert_eq!(
        d_interior.n_resampled, 2,
        "both observations are interior, so both must drive a resampling"
    );
    assert_eq!(d_interior.n_as_skipped_no_resample, 8);
}

/// The gate is a fact about what the resampler DID, not about the observation
/// schedule. An observation whose weights come out exactly equal across the
/// ensemble draws no ancestry, so ancestor sampling must be suppressed there
/// even though the substep carries data.
///
/// Constructed by making the observation carry no information about the state:
/// the reference's own value is used for every particle, and the likelihood is
/// evaluated at a projection that cannot distinguish them because the ensemble
/// has not yet diverged — the substep-0 observation, where every particle still
/// sits on the deterministic initial state.
#[test]
fn an_observation_whose_weights_tie_still_suppresses_the_move() {
    // Observation on substep 0. Every particle starts from the same
    // deterministic initial state and the reference is clamped to its own
    // recorded flows, but the FREE particles have propagated by the time the
    // weight is taken — so this alone would not tie. The tie we can construct
    // deterministically is the one at the START of substep 0, before any
    // observation has been scored: incoming weights are the initialised zeros.
    let f = fixture(6, &[0, 3], |c| c[I_IDX] as f64);
    let (_, d) = sweep(&f, SEED);
    // Substep 0's INCOMING weights are the initialised zeros — all equal — so
    // substep 0 never resamples regardless of what observation it carries.
    // Observations at substeps 0 and 3 therefore drive resampling at substeps 1
    // and 4 only.
    assert_eq!(
        d.n_resampled, 2,
        "two interior observations should drive two resamplings (at the FOLLOWING substeps)"
    );
    assert!(
        d.n_as_skipped_no_resample >= 4,
        "the four substeps with equal incoming weights must all suppress the move, got {}",
        d.n_as_skipped_no_resample
    );
    assert_eq!(
        d.n_resampled + d.n_as_skipped_no_resample,
        d.n_substeps,
        "every substep is in exactly one of the two cases — there is no third"
    );
}

/// Case A must still work: where an ancestry IS drawn, ancestor sampling must
/// run and be capable of moving the reference. Without this, the gate could be
/// "fixed" by disabling ancestor sampling everywhere and every other test here
/// would still pass.
#[test]
fn where_an_ancestry_is_drawn_the_move_is_still_offered() {
    // Observations on most substeps, so most of them resample.
    let f = fixture(12, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], |c| c[I_IDX] as f64);
    let (mut proposed, mut accepted, mut resampled) = (0usize, 0usize, 0usize);
    for seed in 0..8u64 {
        let (_, d) = sweep(&f, SEED + seed);
        proposed += d.n_as_proposed;
        accepted += d.n_as_accepted;
        resampled += d.n_resampled;
    }
    assert!(resampled > 0, "fixture must actually resample or Case A is untested");
    assert!(
        proposed > 0,
        "no ancestor move was proposed anywhere — the gate is suppressing Case A too"
    );
    assert!(
        accepted > 0,
        "moves were proposed ({proposed}) but none accepted; the reference can never renew"
    );
}
