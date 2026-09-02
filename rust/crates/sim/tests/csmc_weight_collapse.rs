//! gh#783: a `csmc_as` sweep in which every particle scored zero observation
//! density must be distinguishable from one that kept the reference because the
//! reference won.
//!
//! # Why the two used to be the same event
//!
//! `csmc_as` ends with a categorical draw over the filter weights. That draw
//! returns `None` when no particle carries a finite weight, and the code took
//! `unwrap_or(j_ref)` — the reference slot's index, which is also what a
//! *successful* draw returns whenever the reference wins. Keeping the reference
//! is a common and correct outcome of the kernel, so the failed search was
//! reported as an ordinary sweep: no error, no diagnostic, no field.
//!
//! # The fixture, and the one thing that differs between its two arms
//!
//! `sir_basic` with a Poisson observation on prevalence, so the observation's
//! rate is the `I` count and `poisson_logpmf(y > 0, lambda = 0)` is `-inf` — the
//! same "a `projected` of exactly 0 against a positive count" that gh#780
//! measured on a real run.
//!
//! Every sweep here runs at `I0 = 0`, so each free particle starts at `I = 0`
//! and stays there: with `I = 0` both the infection rate (`beta*S*I/N`) and the
//! recovery rate (`gamma*I`) are zero, no transition can fire, and the free
//! swarm is frozen with no dependence on the RNG at all. The observed value is
//! a positive constant, so every free particle scores `-inf`.
//!
//! The arms differ in exactly one input: the `I0` the REFERENCE trajectory was
//! simulated under.
//!
//! - `ref_i0 = 12` — the reference carries `I > 0`, scores finitely, and is the
//!   only live particle. The final draw *selects* it. A legitimate
//!   reference-kept sweep.
//! - `ref_i0 = 0` — the reference is frozen at `I = 0` too, so every particle
//!   including the reference scores `-inf`. Total collapse; the final draw has
//!   nothing to select and falls back.
//!
//! Both arms return the reference trajectory, and `assert_indistinguishable_
//! except_for_collapse` pins that they agree on every diagnostic field that
//! existed before this change. `weight_collapse` is the only thing that tells
//! them apart, which is the defect stated as a test.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, CSMCDiagnostics, EffectFiring, ObsAtSubstep,
    PGASTrajectory, WeightCollapseTally,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260828;
const N_PARTICLES: usize = 16;
/// Index of `I` in `sir_basic`'s compartment order (S, I, R).
const I_IDX: usize = 1;
/// The reference's `I0` in the arm where the reference is alive, and the
/// observed count both arms are scored against. One number: the legitimate arm
/// then scores `Poisson(y = 12 | lambda = I_ref)` finitely, and the collapsed
/// arm scores `Poisson(y = 12 | lambda = 0) = -inf`.
const LIVE_REF_I0: f64 = 12.0;
const OBSERVED_COUNT: f64 = 12.0;

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

/// `sir_basic` with the prevalence observation block and `I0 = 0`: the free
/// particles' initial state, and therefore a frozen swarm at `I = 0`.
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
            "I0" => 0.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

fn param_index(compiled: &CompiledModel, name: &str) -> usize {
    compiled
        .model
        .parameters
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no parameter {name}"))
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    reference: PGASTrajectory,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
}

/// `obs_substeps` is keyed by SUBSTEP index; substep `s` spans
/// `(s*dt, (s+1)*dt]`, so an observation attached to substep `s` sits at time
/// `(s+1)*dt`. `ref_i0` is the `I0` the REFERENCE is simulated under, and is
/// the only input that differs between the two arms.
fn fixture(n_substeps: usize, obs: &[(usize, f64)], ref_i0: f64) -> Fixture {
    let compiled = model(n_substeps as f64 * DT);
    // `params` drives the sweep: `I0 = 0`, so every free particle is frozen at
    // `I = 0`. The reference is simulated under `ref_i0`; `I0` enters only the
    // initial condition, so the reference's recorded flows keep the same
    // (finite) transition density under `params` either way.
    let params = compiled.default_params.clone();
    let mut ref_params = params.clone();
    ref_params[param_index(&compiled, "I0")] = ref_i0;

    let mut rng = StatefulRng::new(SEED);
    let reference =
        simulate_reference(&compiled, &ref_params, n_substeps as f64 * DT, DT, &mut rng)
            .expect("reference");
    assert_eq!(reference.substeps.len(), n_substeps, "grid is not what the test assumes");

    let observations: Vec<Observation> = obs
        .iter()
        .map(|&(s, value)| Observation { time: ((s + 1) as f64) * DT, value })
        .collect();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![I_IDX]),
            compiled.model.observations[0].clone(),
            dense_cells(observations.iter().map(|o| o.value).collect()),
            observations.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();
    let obs_at_substep: ObsAtSubstep =
        build_obs_at_substep(&observations, compiled.model.simulation.t_start, DT)
            .expect("obs_at_substep");

    Fixture { compiled, params, reference, obs: observations, obs_model, obs_at_substep }
}

fn sweep(f: &Fixture) -> (PGASTrajectory, CSMCDiagnostics) {
    csmc_as(
        &f.compiled,
        &f.params,
        &f.obs,
        &f.reference,
        N_PARTICLES,
        DT,
        &f.obs_model,
        SEED,
        &f.obs_at_substep,
        EffectFiring::default(),
        sim::rng::BinomialAlgorithm::Btpe,
        true,
    )
    .expect("csmc_as")
}

/// The returned path is the input reference, substep for substep. Both arms
/// must satisfy this — it is what makes them indistinguishable on the output
/// trajectory, and therefore what makes the diagnostic the only signal.
fn assert_is_the_reference(traj: &PGASTrajectory, reference: &PGASTrajectory, label: &str) {
    assert_eq!(traj.initial_counts, reference.initial_counts, "{label}: initial counts");
    assert_eq!(traj.substeps.len(), reference.substeps.len(), "{label}: substep count");
    for (s, (got, want)) in traj.substeps.iter().zip(&reference.substeps).enumerate() {
        assert_eq!(got.counts_before, want.counts_before, "{label}: substep {s} counts_before");
        assert_eq!(got.counts_after, want.counts_after, "{label}: substep {s} counts_after");
        assert_eq!(got.flows, want.flows, "{label}: substep {s} flows");
    }
}

/// Every diagnostic field that existed before gh#783 agrees between the two
/// arms. This is the defect: without `weight_collapse` there is nothing in the
/// result to tell a failed search from a successful one.
fn assert_indistinguishable_except_for_collapse(a: &CSMCDiagnostics, b: &CSMCDiagnostics) {
    assert_eq!(a.trajectory_renewal, b.trajectory_renewal, "trajectory_renewal");
    for (i, (x, y)) in a.renewal_by_bin.iter().zip(&b.renewal_by_bin).enumerate() {
        // A bin holding no substep reads `NaN` in both, which `==` denies.
        assert!(x.is_nan() && y.is_nan() || x == y, "renewal_by_bin[{i}]: {x} vs {y}");
    }
    assert_eq!(a.n_degenerate, b.n_degenerate, "n_degenerate");
    assert_eq!(a.n_resampled, b.n_resampled, "n_resampled");
    assert_eq!(a.n_as_skipped_no_resample, b.n_as_skipped_no_resample, "n_as_skipped_no_resample");
    assert_eq!(a.n_substeps, b.n_substeps, "n_substeps");
    assert_eq!(a.n_as_proposed, b.n_as_proposed, "n_as_proposed");
    assert_eq!(a.n_as_accepted, b.n_as_accepted, "n_as_accepted");
    assert_eq!(a.n_as_refused_inadmissible, b.n_as_refused_inadmissible, "n_as_refused_inadmissible");
}

/// The pair, run side by side so the contrast is one assertion block rather
/// than two tests that could drift apart.
#[test]
fn a_collapsed_sweep_and_a_legitimate_reference_kept_sweep_are_distinguishable() {
    let live = fixture(1, &[(0, OBSERVED_COUNT)], LIVE_REF_I0);
    let dead = fixture(1, &[(0, OBSERVED_COUNT)], 0.0);

    // The fixture must actually realize what it claims: a live reference with
    // `I > 0` (so it can score finitely) and a dead one at `I = 0`.
    assert!(
        live.reference.substeps[0].counts_after[I_IDX] > 0,
        "the live arm's reference must carry infectives, got {:?}",
        live.reference.substeps[0].counts_after
    );
    assert_eq!(
        dead.reference.substeps[0].counts_after[I_IDX], 0,
        "the collapsed arm's reference must be frozen at I = 0",
    );

    let (live_traj, live_diag) = sweep(&live);
    let (dead_traj, dead_diag) = sweep(&dead);

    // Both sweeps returned the reference. This is the outcome that used to be
    // reported identically for both.
    assert_is_the_reference(&live_traj, &live.reference, "legitimate");
    assert_is_the_reference(&dead_traj, &dead.reference, "collapsed");
    assert_eq!(live_diag.trajectory_renewal, 0.0, "the legitimate arm kept the reference");
    assert_eq!(dead_diag.trajectory_renewal, 0.0, "the collapsed arm kept the reference");

    assert_indistinguishable_except_for_collapse(&live_diag, &dead_diag);

    // The legitimate arm: the reference is the ONE live particle, so the draw
    // selected it rather than falling back. `min_alive == 1` is the non-vacuity
    // check — it says the discriminating case is present, not merely that
    // nothing collapsed.
    let live_wc = &live_diag.weight_collapse;
    assert_eq!(live_wc.n_windows, 0, "a legitimate reference-kept sweep must not be flagged");
    assert_eq!(live_wc.first_substep, None, "no window collapsed");
    assert_eq!(live_wc.min_alive, 1, "exactly the reference was alive at the observation");
    assert!(!live_wc.final_draw_fell_back, "the final draw selected the reference, not fell back");
    // gh#685: one finite weight is an ESS of exactly 1, and the profile
    // locates it at the observation that produced it — index 0 of the
    // observation slice the sweep was given.
    assert_eq!(live_wc.ess_by_obs.len(), live.obs.len(), "one slot per observation");
    let (live_at, live_ess) = live_wc.min_ess().expect("one observation was scored");
    assert!((live_ess - 1.0).abs() < 1e-12, "one live particle is an ESS of 1, got {live_ess}");
    assert_eq!(live_at, 0);

    // The collapsed arm: nothing was alive, and the sweep says so.
    let dead_wc = &dead_diag.weight_collapse;
    assert_eq!(dead_wc.n_windows, 1, "the one observation window collapsed");
    assert_eq!(dead_wc.first_substep, Some(0), "and it was substep 0");
    assert_eq!(dead_wc.min_alive, 0, "no particle carried a finite weight");
    assert_eq!(dead_wc.min_ess(), Some((0, 0.0)), "a dead vector is an ESS of 0");
    assert!(dead_wc.final_draw_fell_back, "the final draw had nothing to select");
}

/// gh#685. The ESS sees what the alive count cannot. Every weight here is
/// finite, so `min_alive` reads the full swarm at every window; at the second,
/// one particle carries all of the mass and the following resample would copy
/// it into every slot. The profile reads ~1 there and `min_ess` names that
/// observation. Observation 1 is never scored — the grid dropped it — and
/// stays `NaN`, which `min_ess` skips rather than treating as a minimum.
#[test]
fn ess_profile_locates_a_collapse_that_leaves_every_weight_finite() {
    let mut tally = WeightCollapseTally::new(4, 4);
    tally.record(0, 0, &[0.0, 0.0, 0.0, 0.0]);
    tally.record(3, 2, &[0.0, -30.0, -30.0, -30.0]);
    tally.record(5, 3, &[0.0, -1.0, 0.0, -1.0]);
    let wc = tally.finish(false);

    assert_eq!(wc.n_windows, 0, "no window was dead");
    assert_eq!(wc.min_alive, 4, "every weight was finite at every window");
    assert_eq!(wc.ess_by_obs.len(), 4, "one slot per observation");
    assert_eq!(wc.ess_by_obs[0], 4.0, "uniform weights are the full swarm");
    assert!(wc.ess_by_obs[1].is_nan(), "an unscored observation stays NaN");
    let (worst_at, worst_ess) = wc.min_ess().expect("three windows were scored");
    assert!((worst_ess - 1.0).abs() < 1e-9, "one particle held the mass, got {worst_ess}");
    assert_eq!(worst_at, 2);
}

/// A tie names the earliest observation, and an unscored sweep names none.
#[test]
fn min_ess_breaks_ties_toward_the_earliest_window() {
    let mut tally = WeightCollapseTally::new(3, 3);
    tally.record(2, 1, &[0.0, 0.0, 0.0]);
    tally.record(4, 2, &[0.0, 0.0, 0.0]);
    let wc = tally.finish(false);
    assert_eq!(wc.min_ess(), Some((1, 3.0)));

    let unscored = sim::inference::pgas::WeightCollapse::none(3);
    assert!(unscored.ess_by_obs.is_empty());
    assert!(unscored.min_ess().is_none());

    let never_scored = WeightCollapseTally::new(3, 2).finish(false);
    assert_eq!(never_scored.ess_by_obs.len(), 2);
    assert!(never_scored.min_ess().is_none(), "all-NaN is not a minimum of 0");
}

/// The `unwrap_or(j_ref)` at the final draw cannot be the only detector: a
/// window that collapses anywhere but the last weight-setting substep is
/// consumed by the following resample, which normalises the dead vector to
/// uniform and fabricates a swarm. By the end of the sweep the final draw is
/// perfectly healthy and the collapse has left no trace there.
///
/// Substep 0 is observed against a positive count with every particle at
/// `I = 0` (collapse); substep 2 is observed against 0, which every particle
/// scores as `Poisson(0 | lambda = 0) = 0` (all alive), so the final draw
/// succeeds.
#[test]
fn a_collapse_the_final_draw_never_sees_is_still_recorded() {
    let f = fixture(3, &[(0, OBSERVED_COUNT), (2, 0.0)], 0.0);
    let (_traj, diag) = sweep(&f);

    let wc = &diag.weight_collapse;
    assert!(
        !wc.final_draw_fell_back,
        "the final draw must succeed here, or this test is not exercising the mid-sweep case",
    );
    assert_eq!(wc.n_windows, 1, "the substep-0 window collapsed and must be counted");
    assert_eq!(wc.first_substep, Some(0), "the collapse is located at substep 0");
    assert_eq!(wc.min_alive, 0, "no particle carried a finite weight at that window");
}

/// The aliveness predicate, stated directly. A `NaN` weight is not a particle
/// the categorical draw can return, so it does not count as alive; counting it
/// would report a swarm that is not there.
#[test]
fn n_alive_counts_only_the_finite_weights() {
    assert_eq!(WeightCollapseTally::n_alive(&[0.0, -1.0, -300.0]), 3);
    assert_eq!(WeightCollapseTally::n_alive(&[f64::NEG_INFINITY; 4]), 0);
    assert_eq!(WeightCollapseTally::n_alive(&[f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY]), 1);
    assert_eq!(WeightCollapseTally::n_alive(&[f64::NAN, f64::NEG_INFINITY]), 0);
    assert_eq!(WeightCollapseTally::n_alive(&[f64::NAN, -2.0]), 1);
    assert_eq!(WeightCollapseTally::n_alive(&[]), 0);
}
