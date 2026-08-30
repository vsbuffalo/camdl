//! gh#780: a PGAS chain start must be refused for what it says about θ₀, not
//! for what one trajectory draw and one sweep's luck said about `(θ₀, X)`.
//!
//! # What went wrong
//!
//! `run_pgas` refuses a fresh start whose complete-data log-posterior is
//! `-inf` and is still `-inf` after one Gibbs sweep (`NonFiniteChainStart`,
//! gh#607). The complete-data density is conditional on a sampled latent
//! trajectory `X`, so a `-inf` there is usually the observation term saying
//! *this draw* predicts zero where the data is positive — a statement about
//! the pair, not about θ₀. The probation sweep is meant to catch that: the
//! `X | θ, y` move gets one attempt to replace the unlucky draw.
//!
//! One sweep is not enough, and the reason is specific to the CONDITIONAL
//! sampler. The fixture below is a closed SIR observed as
//! `incidence(infection) ~ Poisson(projected)` at 12 weekly windows, with θ
//! over-predicting the data by 75%; its reference draw has zero incidence in
//! window 10, where `y = 64`. At 1 000 particles, one sweep from that
//! reference goes:
//!
//! ```text
//!   window   0     1     2     3     4     5     6     7    8    9   10   11
//!   alive  1000  1000  1000  1000  1000  1000  1000   997  847  528  141   46
//!   W_ref  .0000 .0000 .0000 .0000 .0000 1.0000 .0000 .0000 .000 .000 .000 .5000
//! ```
//!
//! - `csmc_as` retains the reference as one particle whatever its weight. At
//!   window 5 the reference takes the whole normalised weight (`W_ref =
//!   1.0000`), because it is the one particle the earlier windows did not
//!   squeeze, so every free slot is resampled onto its state.
//! - From there the swarm is entirely descended from a reference that has
//!   burnt through its susceptibles, and the live-particle count falls away
//!   (847 → 528 → 141 → 46 over windows 8-11).
//! - At the last window the reference holds half the remaining weight, so half
//!   of these sweeps return the reference verbatim — complete-data density
//!   `-inf`, and the chain is refused.
//!
//! At the same θ, particle count and seed the *unconditional* bootstrap filter
//! keeps `1000/1000` particles alive at every window and returns
//! `log p(y | θ) = -132`. The reference particle is the only structural
//! difference between the two samplers. That is the disagreement the
//! downstream report names: camdl's own filter scores a refused start better
//! than a chain that ran to completion.
//!
//! # The criterion
//!
//! `p(y | θ₀) > 0` holds exactly when a positive-density trajectory exists at
//! θ₀, which is the question the refusal is trying to answer. `X` is a
//! nuisance variable the sampler resamples every sweep. So the probation
//! verdict is now checked against the bootstrap filter's marginal before the
//! chain is refused. The check is directional — it can only rescue a start
//! probation would have refused, never refuse one probation admitted.
//!
//! # Acceptance, both directions
//!
//! - `a_start_with_a_finite_marginal_is_not_refused` — the fixture start,
//!   whose complete-data density is `-inf` and whose probation sweep does not
//!   recover, has a finite `log p(y | θ₀)` and must run.
//! - `a_start_whose_filter_finds_no_feasible_trajectory_is_still_refused` — a
//!   θ₀ at which the observation is impossible for every trajectory (`beta =
//!   0`, so incidence is 0 at every window against positive counts) must still
//!   be refused. Its filter bails on ESS collapse, which is the
//!   `PFDegenerate EssCollapsed` half of the acceptance.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{Likelihood, ObservationModel as IrObs, ObservationSchedule, PoissonLikelihood, Projection},
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    error::SimError,
    inference::{
        dense_cells,
        if2::{EstimatedParam, Transform},
        multi_stream_obs::{MultiStreamObsModel, StreamProjection, StreamSpec},
        particle_filter::{bootstrap_filter, Observation},
        pgas::{
            build_obs_at_substep, complete_data_loglik, csmc_as, run_pgas, simulate_reference,
            EffectFiring, PGASConfig, PGASTrajectory,
        },
        pmmh::Prior,
        traits::SMCConfig,
        BoundObs, ChainBinomialProcess,
    },
    rng::StatefulRng,
    schedule::StepPolicy,
};

const DT: f64 = 1.0;
const GAMMA: f64 = 0.10;
const I0: f64 = 5.0;
const N_POP: f64 = 1000.0;
const EVERY: usize = 7;
const N_WINDOWS: usize = 12;
const T_END: f64 = (N_WINDOWS * EVERY) as f64;

/// The β the chain starts at. It over-predicts the data (drawn at β = 0.20)
/// by 75%, which is what makes the conditional sweep collapse onto its
/// reference — see the module header.
const BETA_START: f64 = 0.35;

/// Weekly counts drawn from this model at β = 0.20, γ = 0.10, I₀ = 5, N = 1000
/// (seed 0, the first draw positive in every window), inlined so the fixture
/// does not depend on the simulator to define its own data.
const Y: [f64; N_WINDOWS] = [11.0, 19.0, 18.0, 26.0, 47.0, 76.0, 100.0, 105.0, 107.0, 80.0, 64.0, 32.0];

// ── model ──────────────────────────────────────────────────────────────────

fn p(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn c(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(a), right: Box::new(b) } })
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Div, left: Box::new(a), right: Box::new(b) } })
}
fn param(name: &str, value: f64) -> Parameter {
    Parameter {
        name: name.into(),
        value: ir::parameter::ParamValue::Fixed { value },
        param_kind: None,
        param_dim: None,
    }
}

/// Closed SIR observed as `incidence(infection) ~ Poisson(projected)`.
///
/// Poisson with the bare projection is deliberate: `Poisson(rate = 0)` puts
/// probability exactly 0 on a positive count, which is the ingredient the
/// downstream report names — "a `projected` of exactly 0 against a positive
/// count" — and the only way an observation term reaches `-inf`.
fn model(beta: f64) -> Arc<CompiledModel> {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let m = Model {
        ic_grad: Default::default(),
        name: "gh780_sir".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: div(mul(mul(p("beta"), c("S")), c("I")), n),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
            Transition {
                rate_state_grad: Default::default(),
                name: "recovery".into(),
                stoichiometry: vec![
                    StoichiometryEntry("I".into(), -1),
                    StoichiometryEntry("R".into(), 1),
                ],
                rate: mul(p("gamma"), c("I")),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![IrObs {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ir::observation::ObsColumn {
                    name: "time".into(),
                    role: ir::observation::ColumnRole::Time,
                },
                ir::observation::ObsColumn {
                    name: "cases".into(),
                    role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "cases".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: Projection::CumulativeFlow("infection".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![param("beta", beta), param("gamma", GAMMA)],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), N_POP - I0);
            h.insert("I".into(), I0);
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, T_END]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            time_semantics: "continuous".into(),
            dt: Some(DT),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).expect("fixture model must compile"))
}

fn observations() -> Vec<Observation> {
    (1..=N_WINDOWS)
        .map(|w| Observation { time: (w * EVERY) as f64, value: Y[w - 1] })
        .collect()
}

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let obs = observations();
    MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            ir_model: compiled.model.observations[0].clone(),
            projection: StreamProjection::FlowSum(vec![0]),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap()
}

/// `beta` estimated with a tight random walk; `gamma` held at its value by a
/// degenerate range. Only `beta` moves, and the θ|X block is inert anyway
/// while the chain sits at `-inf`.
fn beta_param() -> EstimatedParam {
    EstimatedParam {
        name: "beta".into(),
        index: 0,
        initial: BETA_START,
        rw_sd: 0.01,
        transform: Transform::Log { lo: 0.01, hi: 2.0 },
        lower: 0.01,
        upper: 2.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

fn pgas_config(n_particles: usize) -> PGASConfig {
    PGASConfig {
        n_particles,
        n_sweeps: 1,
        burn_in: 0,
        thin: 1,
        dt: DT,
        use_nuts: false,
        dense_mass: false,
        tempering: vec![1.0],
        max_tree_depth: 10,
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: StepPolicy::Snap,
    }
}

/// `run_pgas` for one sweep at `seed`; `Err` carries the refusal.
fn run_one_sweep(beta: f64, n_particles: usize, seed: u64) -> Result<(), SimError> {
    let compiled = model(beta);
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    let base = compiled.default_params.clone();
    run_pgas(
        &compiled,
        &[beta_param()],
        &priors,
        &base,
        &pgas_config(n_particles),
        &obs,
        &obs_m,
        seed,
        None,
        None,
        "gh780".into(),
    )
    .map(|_| ())
}

/// `log p(y | θ)` from the plain bootstrap filter — the quantity the refusal
/// is now decided on, and the one the downstream computed by hand.
fn marginal(beta: f64, n_particles: usize, seed: u64) -> Result<f64, SimError> {
    let compiled = model(beta);
    let obs_m = obs_model(&compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let config = SMCConfig {
        n_particles,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter(&process, &obs_m, &compiled.default_params, &config, seed)
        .map(|r| r.log_likelihood)
}

/// The complete-data density of the reference `run_pgas` builds at `seed` —
/// the number the probation criterion reads.
fn start_complete_data(beta: f64, seed: u64) -> sim::inference::pgas::LogLikComponents {
    let compiled = model(beta);
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(seed);
    let traj = simulate_reference(&compiled, &params, T_END, DT, &mut rng).expect("reference");
    let obs_at_substep = build_obs_at_substep(&obs, 0.0, DT).expect("obs_at_substep");
    complete_data_loglik(&compiled, &traj, &params, &obs, DT, &obs_m, &obs_at_substep)
        .expect("score")
}

/// One CSMC-AS sweep from `reference` at `sweep_seed`; `true` if the returned
/// trajectory has finite complete-data density.
fn one_sweep_recovers(reference: &PGASTrajectory, n_particles: usize, sweep_seed: u64) -> bool {
    let compiled = model(BETA_START);
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let params = compiled.default_params.clone();
    let obs_at_substep = build_obs_at_substep(&obs, 0.0, DT).expect("obs_at_substep");
    let (traj, _) = csmc_as(
        &compiled, &params, &obs, reference, n_particles, DT, &obs_m, sweep_seed,
        &obs_at_substep, EffectFiring::default(),
    )
    .expect("csmc_as");
    complete_data_loglik(&compiled, &traj, &params, &obs, DT, &obs_m, &obs_at_substep)
        .expect("score")
        .total
        .is_finite()
}

/// The reference trajectory `run_pgas` draws at `SEED` — the `X₀` the
/// probation criterion is conditional on.
fn fixture_reference() -> PGASTrajectory {
    let compiled = model(BETA_START);
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(SEED);
    simulate_reference(&compiled, &params, T_END, DT, &mut rng).expect("reference")
}

// ── the seed this fixture pins ─────────────────────────────────────────────

/// Chosen from a scan of `run_pgas` seeds: at this one the reference the
/// sampler draws has a zero-incidence observation window (so the start is
/// `-inf`) and the probation sweep does not clear it. Eight of the first forty
/// seeds behave this way at `N_PARTICLES` — the failure is common, not exotic.
const SEED: u64 = 3;
const N_PARTICLES: usize = 1000;

// ── the fixture's own preconditions ────────────────────────────────────────

/// Non-vacuity, part 1. The start is exactly the reported shape: its
/// complete-data density is `-inf` on the OBSERVATION term (a `-inf` on the
/// transition term would be a `step_one`/density disagreement — gh#80, a
/// different finding), and its marginal likelihood is finite. That pair IS the
/// disagreement gh#780 reports.
#[test]
fn the_fixture_start_has_the_reported_shape() {
    let comp = start_complete_data(BETA_START, SEED);
    assert!(
        !comp.observation.is_finite(),
        "the start's observation term must be -inf (its reference draw predicts zero \
         incidence in a window where y > 0); got {}",
        comp.observation
    );
    assert!(
        comp.transition.is_finite(),
        "the start's transition term must be FINITE — a -inf there is a step_one / \
         density disagreement (gh#80), a different finding; got {}",
        comp.transition
    );
    let m = marginal(BETA_START, N_PARTICLES, SEED).expect("filter must run at the fixture start");
    assert!(
        m.is_finite(),
        "the fixture start's log p(y | theta) must be finite — that is the whole \
         disagreement gh#780 reports; got {m}"
    );
    eprintln!(
        "fixture start: complete-data observation {} (transition {:.1}), log p(y|theta) {m:.1}",
        comp.observation, comp.transition
    );
}

/// Non-vacuity, part 2. The single CSMC-AS sweep really does fail from this
/// reference, often — otherwise the acceptance test below would pass on the
/// old criterion too and prove nothing. Measured across sweep seeds rather
/// than pinned to one, so the assertion does not depend on `run_pgas`'s
/// internal seed derivation.
#[test]
fn one_conditional_sweep_often_fails_to_clear_the_start() {
    let reference = fixture_reference();
    const TRIALS: u64 = 24;
    let failures = (0..TRIALS)
        .filter(|s| !one_sweep_recovers(&reference, N_PARTICLES, 90_000 + s))
        .count();
    eprintln!("{failures}/{TRIALS} single CSMC-AS sweeps stay at zero density");
    assert!(
        failures * 4 >= TRIALS as usize,
        "the fixture must exercise the failure: only {failures} of {TRIALS} sweeps \
         stayed at zero density, so a one-sweep probation would usually clear this \
         start and the acceptance test below would be vacuous"
    );
}

/// The mechanism, stated as a test. At the same θ, particle count and seed the
/// unconditional bootstrap filter returns a finite marginal where the
/// conditional sweep returns a zero-density trajectory. The reference particle
/// is the only structural difference between the two samplers.
#[test]
fn the_unconditional_filter_does_not_collapse_where_the_conditional_sweep_does() {
    let m = marginal(BETA_START, N_PARTICLES, SEED).expect("the plain filter must run");
    assert!(
        m.is_finite(),
        "the unconditional filter must find positive-density trajectories at this θ; got {m}"
    );
    let reference = fixture_reference();
    assert!(
        (0..24u64).any(|s| !one_sweep_recovers(&reference, N_PARTICLES, 90_000 + s)),
        "at least one conditional sweep must fail where the plain filter succeeds — \
         otherwise the two do not disagree and there is nothing to fix"
    );
}

// ── acceptance, both directions ────────────────────────────────────────────

/// A start whose marginal likelihood is finite must not be refused, even when
/// its complete-data density is `-inf` and the probation sweep does not clear
/// it. This is the downstream's chain_4, scored 29 nats better by camdl's own
/// filter than a chain that ran to 2 000 sweeps.
#[test]
fn a_start_with_a_finite_marginal_is_not_refused() {
    match run_one_sweep(BETA_START, N_PARTICLES, SEED) {
        Ok(()) => {}
        Err(SimError::NonFiniteChainStart { marginal, .. }) => panic!(
            "the chain was refused ({marginal}), but log p(y | theta_0) is finite at \
             this start — the refusal is reading the complete-data density, which is \
             conditional on one trajectory draw and on one sweep's luck, instead of \
             the marginal, which is about theta alone (gh#780)"
        ),
        Err(e) => panic!("unexpected error: {e}"),
    }
}

/// The other direction, and the one that must not regress. At `beta = 0` there
/// is no infection transition at all, so incidence is 0 in every window against
/// positive counts: no trajectory has positive density, the filter's swarm dies
/// and it bails on ESS collapse, and the chain must still be refused. This is
/// the downstream's chain_3 / chain_13 — the draws `bad_init` exists to drop —
/// and gh#607's `iota = 0` fixture in `cli/tests/pgas_bad_init_skip.rs`.
#[test]
fn a_start_whose_filter_finds_no_feasible_trajectory_is_still_refused() {
    let m = marginal(0.0, N_PARTICLES, SEED);
    assert!(
        matches!(&m, Err(SimError::PFDegenerate { .. })),
        "precondition: at beta = 0 the filter must find no feasible trajectory; got {m:?}"
    );
    match run_one_sweep(0.0, N_PARTICLES, SEED) {
        Err(SimError::NonFiniteChainStart { marginal, .. }) => {
            assert!(
                marginal.contains("bootstrap particle filter"),
                "the refusal must name what the filter said, since that is now what \
                 decided it; got {marginal}"
            );
        }
        Ok(()) => panic!(
            "an impossible start was admitted: at beta = 0 incidence is 0 in every \
             window against positive counts, so no trajectory can explain the data \
             and the chain could only have produced -inf draws (gh#607)"
        ),
        Err(e) => panic!("unexpected error: {e}"),
    }
}
