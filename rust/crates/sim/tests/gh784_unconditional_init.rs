//! gh#784 step 3a: a PGAS chain's initial reference trajectory `X₀` comes from
//! an ORDINARY UNCONDITIONAL SMC pass at `θ₀`, not from a single forward draw.
//!
//! # The defect this closes
//!
//! Seeding `X₀` by simulating the model forward puts no weight on the
//! observations at all. On an informative model the draw therefore lands, some
//! fraction of the time, where the observation model scores `−∞` — and the
//! fraction depends on `θ₀`, so the chains that survive are a biased sample of
//! the start distribution (gh#780). The conditional sweep was expected to repair
//! such a start, and gh#780 measured that repair failing: conditioned on a
//! reference the data cannot explain, the reference can take essentially the
//! whole normalised weight in an early observation window and the swarm becomes
//! its own descendants.
//!
//! An unconditional filter has no reference to be captured by. It reweights
//! against the data at every observation, so the lineage it returns already
//! explains them as well as `N` draws at `θ₀` can. This is standard
//! particle-Gibbs practice (Andrieu, Doucet & Holenstein 2010, *JRSS-B*
//! 72:269-342, §4.5; Lindsten, Jordan & Schön 2014, *JMLR* 15:2145-2184, §2.3).
//!
//! # The fixture, and why its premise is asserted rather than assumed
//!
//! `sir_basic` at `I₀ = 1`, `β = 0.35`, `γ = 0.12`, `N = 400`: supercritical, so
//! the epidemic takes off about two thirds of the time and dies out the rest.
//! The observed prevalence series is one SURVIVING realization at exactly these
//! parameters — data the model can produce — scored by a Poisson whose rate is
//! the raw projection, so a trajectory with `I = 0` against a positive count is
//! `−∞`, not merely improbable.
//!
//! The chain seed is chosen so its forward draw is one of the extinctions. That
//! is the fixture's whole premise, and
//! [`the_forward_draw_at_this_start_is_impossible`] asserts it directly: if the
//! forward draw ever stops being `−∞` there, every test below would pass for the
//! wrong reason.
//!
//! # What is NOT claimed
//!
//! Nothing here asserts that any `θ` is infeasible. A start whose unconditional
//! pass also fails is an INITIALIZATION failure — a fact about the pass — and
//! [`a_start_no_swarm_can_explain_is_still_refused_and_says_why`] pins that the
//! refusal says so in those words.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::error::{InitFallback, InitSource, SimError};
use sim::inference::dense_cells;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{
    BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec,
};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, run_pgas, simulate_reference,
    simulate_reference_on_grid, EffectFiring, ObsAtSubstep, PGASConfig, PGASTrajectory,
};
use sim::inference::pgas_init::{initial_reference_trajectory, unconditional_smc_pass, UnconditionalPass};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const N_SUBSTEPS: usize = 20;
const I_IDX: usize = 1;
/// Substeps carrying an observation; substep `s` spans `(s·dt, (s+1)·dt]`, so
/// the observation sits at time `(s+1)·dt`.
const OBS_SUBSTEPS: [usize; 4] = [4, 9, 14, 19];
/// The chain seed. Its forward draw goes extinct by substep 14 — asserted, not
/// assumed, by `the_forward_draw_at_this_start_is_impossible`.
const CHAIN_SEED: u64 = 1;
/// The seed whose forward draw supplies the observed series. A surviving
/// realization at the same `θ₀`, so the data are reachable by construction.
const DATA_SEED: u64 = 0;
const N_PARTICLES: usize = 200;

// ── fixture ────────────────────────────────────────────────────────────────

fn prevalence_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    // Rate = the raw projection. No `+ eps` floor: an `I = 0` trajectory against
    // a positive count must be exactly -inf, which is the condition under test.
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

fn model(i0: f64) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = N_SUBSTEPS as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.35,
            "gamma" => 0.12,
            "N0" => 400.0,
            "I0" => i0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
    grid: Vec<(f64, f64)>,
}

/// `i0` seeds the model; `obs_values` are the observed prevalences at
/// `OBS_SUBSTEPS`. `None` takes them from `DATA_SEED`'s own forward draw.
fn fixture(i0: f64, obs_values: Option<[f64; 4]>) -> Fixture {
    let compiled = model(i0);
    let params = compiled.default_params.clone();

    let values: Vec<f64> = match obs_values {
        Some(v) => v.to_vec(),
        None => {
            let mut rng = StatefulRng::new(DATA_SEED);
            let path =
                simulate_reference(&compiled, &params, N_SUBSTEPS as f64 * DT, DT, &mut rng)
                    .expect("data-generating draw");
            OBS_SUBSTEPS
                .iter()
                .map(|&s| path.substeps[s].counts_after[I_IDX] as f64)
                .collect()
        }
    };
    assert!(
        values.iter().all(|&v| v > 0.0),
        "every observed prevalence must be POSITIVE, else an extinct trajectory \
         would score finitely and the fixture would prove nothing; got {values:?}"
    );

    let obs: Vec<Observation> = OBS_SUBSTEPS
        .iter()
        .zip(&values)
        .map(|(&s, &v)| Observation { time: ((s + 1) as f64) * DT, value: v })
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
    let obs_at_substep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs_at_substep");
    // Snap: the uniform grid, which is what `build_substep_grid` produces there.
    let grid: Vec<(f64, f64)> = (0..N_SUBSTEPS).map(|s| (s as f64 * DT, DT)).collect();

    Fixture { compiled, params, obs, obs_model, obs_at_substep, grid }
}

impl Fixture {
    fn density(&self, traj: &PGASTrajectory) -> sim::inference::pgas::LogLikComponents {
        complete_data_loglik(
            &self.compiled,
            traj,
            &self.params,
            &self.obs,
            DT,
            &self.obs_model,
            &self.obs_at_substep,
        )
        .expect("complete_data_loglik")
    }

    fn forward_draw(&self, seed: u64) -> PGASTrajectory {
        let mut rng = StatefulRng::new(seed);
        simulate_reference_on_grid(
            &self.compiled,
            &self.params,
            DT,
            &self.grid,
            EffectFiring::default(),
            &mut rng,
        )
        .expect("forward draw")
    }

    fn init(&self, seed: u64) -> (PGASTrajectory, InitSource) {
        let mut rng = StatefulRng::new(seed);
        initial_reference_trajectory(
            &self.compiled,
            &self.params,
            &self.grid,
            N_PARTICLES,
            DT,
            &self.obs,
            &self.obs_model,
            seed,
            &self.obs_at_substep,
            EffectFiring::default(),
            &mut rng,
        )
        .expect("initial_reference_trajectory")
    }

    fn run(&self, seed: u64) -> Result<sim::inference::pgas::PGASResult, SimError> {
        let beta_idx = self.compiled.param_index["beta"];
        let if2_params = vec![EstimatedParam {
            name: "beta".into(),
            index: beta_idx,
            initial: self.params[beta_idx],
            rw_sd: 0.02,
            transform: Transform::Log { lo: 1e-4, hi: 10.0 },
            lower: 1e-4,
            upper: 10.0,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }];
        let priors = vec![Prior::Fixed(Density::Flat)];
        let config = PGASConfig {
            binomial: sim::rng::BinomialAlgorithm::Btpe,
            n_particles: N_PARTICLES,
            n_sweeps: 2,
            burn_in: 1,
            thin: 1,
            dt: DT,
            use_nuts: false,
            dense_mass: false,
            max_tree_depth: 10,
            tempering: vec![1.0],
            trajectory_warmup: 0,
            csmc_sweeps_per_nuts: 1,
            step_policy: sim::schedule::StepPolicy::Snap,
        };
        run_pgas(
            &self.compiled,
            &if2_params,
            &priors,
            &self.params,
            &config,
            &self.obs,
            &self.obs_model,
            seed,
            None,
            None,
            "gh784".into(),
        )
    }
}

// ── the fixture's premise ──────────────────────────────────────────────────

/// The whole fixture rests on this: at `CHAIN_SEED` the FORWARD draw — the way
/// `X₀` used to be produced — has zero complete-data density, and the term that
/// is non-finite is the OBSERVATION term. A non-finite transition term would be
/// a `step_one`/density disagreement (gh#80), a different bug entirely, and the
/// tests below would then be measuring that instead.
#[test]
fn the_forward_draw_at_this_start_is_impossible() {
    let f = fixture(1.0, None);
    let fwd = f.forward_draw(CHAIN_SEED);

    let extinct_at: Vec<usize> = OBS_SUBSTEPS
        .iter()
        .copied()
        .filter(|&s| fwd.substeps[s].counts_after[I_IDX] == 0)
        .collect();
    assert!(
        !extinct_at.is_empty(),
        "the forward draw at seed {CHAIN_SEED} must go extinct at an observed \
         substep for this fixture to mean anything"
    );

    let ll = f.density(&fwd);
    assert!(
        !ll.total.is_finite(),
        "the forward draw must have ZERO complete-data density; got {}",
        ll.total
    );
    assert!(
        !ll.observation.is_finite(),
        "the non-finite term must be the OBSERVATION term (a bad start), not the \
         transition term (which would be a step_one/density bug, gh#80): \
         transition {}, observation {}",
        ll.transition,
        ll.observation
    );
}

// ── step 3a ────────────────────────────────────────────────────────────────

/// `run_pgas` seeds `X₀` from the unconditional pass, and the chain it starts is
/// at finite density from sweep zero — at a start whose forward draw is
/// impossible.
///
/// # What this test does NOT show, and where the number that matters lives
///
/// It does not show that a chain which the old code REFUSED now runs. At this
/// fixture's scale it could not: measured over 40 chain seeds with pre-gh#784
/// seeding, 14 had an extinct forward draw and **none** of the 40 was refused —
/// the single gh#607 probation sweep repaired every one of them. That is
/// consistent with gh#780's diagnosis rather than against it: the repair fails
/// when the reference can capture the normalised weight in an early window,
/// which needs a long, informative series and a large swarm, not a 20-substep
/// SIR with four observations.
///
/// So the refusal count is measured on the real fit, not here. What is pinned
/// here is the mechanism that changes it: `X₀` comes from the pass, and
/// [`the_initial_reference_has_finite_complete_data_density_and_is_one_lineage`]
/// pins what that buys.
#[test]
fn run_pgas_seeds_x0_from_the_unconditional_pass() {
    let f = fixture(1.0, None);
    let result = match f.run(CHAIN_SEED) {
        Ok(r) => r,
        Err(SimError::NonFiniteChainStart { init, .. }) => panic!(
            "the chain was refused on a start whose UNCONDITIONAL filter can \
             explain the data; X0 came from: {init}"
        ),
        Err(e) => panic!("unexpected error: {e}"),
    };
    assert_eq!(
        result.init_source,
        Some(InitSource::UnconditionalFilter),
        "X0 must come from the unconditional pass, not from the forward draw \
         this fixture has asserted is impossible"
    );
    assert!(
        result.sweeps.iter().all(|s| s.log_complete_data_ll.is_finite()),
        "every sweep of a chain that initialized cleanly must be at finite density"
    );
}

/// The correctness bar for the change: whatever `X₀` PGAS is handed must be a
/// point the target gives positive density. A path the transition or observation
/// density scores as impossible would reintroduce gh#780's failure in a new
/// place.
///
/// Also pinned here: the returned path is a single unbroken lineage. Joining
/// particles across substeps by INDEX rather than by ancestry would produce a
/// path that jumps state at every resample — the mistake `ancestor_trace`'s
/// module doc warns about — and such a path's transition density would be
/// nonsense even where it happened to be finite.
#[test]
fn the_initial_reference_has_finite_complete_data_density_and_is_one_lineage() {
    let f = fixture(1.0, None);
    let (traj, source) = f.init(CHAIN_SEED);
    assert_eq!(source, InitSource::UnconditionalFilter, "{source}");

    let ll = f.density(&traj);
    assert!(
        ll.total.is_finite(),
        "the initial reference must have finite complete-data density: \
         transition {}, observation {}, initial_state {}",
        ll.transition,
        ll.observation,
        ll.initial_state
    );

    assert_eq!(traj.substeps.len(), N_SUBSTEPS, "the path must span the whole grid");
    assert_eq!(
        traj.initial_counts, traj.substeps[0].counts_before,
        "the path does not start where it says it starts"
    );
    for (s, w) in traj.substeps.windows(2).enumerate() {
        assert_eq!(
            w[0].counts_after, w[1].counts_before,
            "the path jumps state between substeps {s} and {}: it is not one lineage",
            s + 1
        );
    }
    for (s, (rec, &(t0, dt))) in traj.substeps.iter().zip(&f.grid).enumerate() {
        assert_eq!((rec.t0, rec.dt_substep), (t0, dt), "substep {s} is off the grid");
    }
}

/// The pass conditions on the data: the lineage it returns tracks the observed
/// prevalence, where the forward draw at the same seed had gone extinct. This is
/// the property that makes the change worth its cost — not merely "finite", but
/// "informed by y".
#[test]
fn the_initial_reference_tracks_the_observed_series() {
    let f = fixture(1.0, None);
    let (traj, _) = f.init(CHAIN_SEED);
    for (k, &s) in OBS_SUBSTEPS.iter().enumerate() {
        let modelled = traj.substeps[s].counts_after[I_IDX] as f64;
        let observed = f.obs[k].value;
        assert!(
            modelled > 0.0,
            "observation {k} sits at prevalence {observed} but the initial \
             reference has I = 0 there"
        );
        // Poisson(rate = I) against `observed`: within a factor of four of the
        // observed count is a loose band that an unconditioned draw fails and a
        // reweighted lineage passes comfortably.
        assert!(
            modelled >= observed / 4.0 && modelled <= observed * 4.0,
            "observation {k}: observed {observed}, initial reference {modelled} \
             — the pass does not look conditioned on the data"
        );
    }
}

// ── the fallback, and what a refusal means after 3a ────────────────────────

/// A start no swarm can explain: `I₀ = 0` makes the infection rate
/// `β·S·I/N` identically zero, so EVERY particle has `I = 0` at every
/// observation while the data are positive. The unconditional pass loses support
/// at the first observation.
///
/// Three things must hold, and they are the design target of gh#784's three
/// statuses:
///
///  1. `X₀` falls back to the forward draw — the pre-gh#784 behaviour, so no
///     chain is worse off than before;
///  2. the fallback is bit-for-bit the draw the old code took off the same RNG,
///     which is what makes a fallback chain's numbers unchanged;
///  3. the failure is reported as an INITIALIZATION failure naming where the
///     swarm lost support, never as a claim that `p(y | θ) = 0`.
#[test]
fn a_start_no_swarm_can_explain_falls_back_to_the_forward_draw_bit_for_bit() {
    let f = fixture(0.0, Some([3.0, 5.0, 13.0, 14.0]));

    // The premise: the swarm genuinely has no support here.
    let pass = unconditional_smc_pass(
        &f.compiled, &f.params, &f.grid, N_PARTICLES, DT, &f.obs_model,
        CHAIN_SEED, &f.obs_at_substep, EffectFiring::default(),
        sim::rng::BinomialAlgorithm::Btpe,
    )
    .expect("unconditional pass");
    match pass {
        UnconditionalPass::NoSupport(InitFallback::SwarmCollapsed { obs_index, .. }) => {
            assert_eq!(obs_index, 0, "support is lost at the FIRST observation here");
        }
        UnconditionalPass::NoSupport(other) => {
            panic!("expected a swarm collapse, got {other}")
        }
        UnconditionalPass::Path(_) => {
            panic!("a swarm with I = 0 everywhere cannot explain a positive count")
        }
    }

    let (traj, source) = f.init(CHAIN_SEED);
    match &source {
        InitSource::ForwardDraw(InitFallback::SwarmCollapsed { obs_index, .. }) => {
            assert_eq!(*obs_index, 0)
        }
        other => panic!("expected the forward-draw fallback, got {other}"),
    }

    // Bit-for-bit the old behaviour: the pass never touches the caller's RNG, so
    // the fallback draw is exactly what a pre-gh#784 chain would have used.
    let old = f.forward_draw(CHAIN_SEED);
    assert_eq!(traj.initial_counts, old.initial_counts);
    assert_eq!(traj.substeps.len(), old.substeps.len());
    for (s, (a, b)) in traj.substeps.iter().zip(&old.substeps).enumerate() {
        assert_eq!(a.counts_before, b.counts_before, "substep {s}");
        assert_eq!(a.counts_after, b.counts_after, "substep {s}");
        assert_eq!(a.flows, b.flows, "substep {s}");
        assert_eq!(a.gammas, b.gammas, "substep {s}");
    }
}

/// The other half: such a chain is STILL refused — gh#784 does not make every
/// start runnable — and the refusal says the initialization failed rather than
/// asserting anything about `θ`.
#[test]
fn a_start_no_swarm_can_explain_is_still_refused_and_says_why() {
    let f = fixture(0.0, Some([3.0, 5.0, 13.0, 14.0]));
    match f.run(CHAIN_SEED) {
        Ok(_) => panic!("a start no trajectory can explain must still be refused"),
        Err(SimError::NonFiniteChainStart { init, .. }) => {
            let fallback = init
                .fallback()
                .unwrap_or_else(|| panic!("the refusal must record the init failure: {init}"));
            assert!(
                matches!(fallback, InitFallback::SwarmCollapsed { obs_index: 0, .. }),
                "expected a swarm collapse at observation 0, got {fallback}"
            );
            let msg = format!("{init}");
            assert!(
                msg.contains("the unconditional pass could not produce a valid path"),
                "the refusal must be phrased as an INITIALIZATION failure: {msg}"
            );
            assert!(
                !msg.contains("infeasible"),
                "an initialization failure must never assert p(y | theta) = 0: {msg}"
            );
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}
