//! gh#82: a *recoverable* error raised while scoring a **candidate** θ must
//! reject that proposal, not tear the PGAS chain down.
//!
//! `run_pgas`'s MH-within-Gibbs θ|X step scored each proposal with
//! `complete_data_loglik(...)?` (`pgas.rs`, the `else` branch of `if
//! has_gradients`). The `?` propagates every `SimError` out of `run_pgas`, and
//! the CLI chain runner turns that into `pgas chain N error: …`
//! (`cli/src/fit/pgas.rs`: `run_pgas(...).map_err(|e| format!("pgas chain {} \
//! error: {}", …))?`) — so one bad *proposal* killed the whole chain (and, via
//! the `collect::<Result<…>>()?` over the rayon chain loop, the whole fit).
//!
//! Every other whole-θ likelihood evaluation in the codebase routes through
//! `SimError::is_structural()` first (gh#224): `fit/pmmh.rs`, `fit/runner.rs`,
//! `fit/dt_check.rs`, `profile.rs` all read `Err(e) if e.is_structural() =>
//! Err(e), Err(_) => Ok(f64::NEG_INFINITY)`. The sibling NUTS branch inside
//! `run_pgas` itself already returns `−∞` for a failed gradient evaluation. The
//! MH branch was the one θ-proposal site that did not.
//!
//! # Fixture
//!
//! A pure-death model `N → ∅` at rate `mu·N`, with the rate **guarded** so that
//! it is defined only in a pinhole around `k = K0`:
//!
//! ```text
//!   rate = if |k − K0| < 1e-9  then  mu·N  else  sqrt(k − 1e9)
//! ```
//!
//! `sqrt` of a large negative is `NaN`, which `eval_propensities` converts to
//! `SimError::NumericalCollapse` — `is_per_particle_recoverable() == true`,
//! `is_structural() == false`. `k` is estimated with a random-walk proposal SD
//! of ~1.0, so **every** MH proposal for `k` leaves the pinhole and hits the
//! degenerate branch; the probability of a proposal landing back inside it is
//! ~1e-9 per sweep.
//!
//! That is what makes the assertions non-vacuous: `k`'s acceptance rate being
//! exactly 0 means no proposal was ever inside the pinhole, i.e. *every* sweep
//! exercised the recoverable-error path (a proposal that did land in the
//! pinhole would score identically to the current state, `log α = 0`, and be
//! accepted). `mu` is estimated too, is *not* part of the guard, and is
//! proposed after `k` within the same sweep — its non-zero acceptance rate is
//! the proof that the sweep continued past the recovered error rather than
//! merely surviving it.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, CondExpr, CondWrap, Expr, UnOp},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    error::SimError,
    inference::{
        if2::{EstimatedParam, Transform},
        particle_filter::Observation,
        pgas::{build_obs_at_substep, complete_data_loglik, run_pgas, simulate_reference, PGASConfig},
        pmmh::Prior,
        multi_stream_obs::StreamSpec,
        BoundObs, MultiStreamObsModel, dense_cells,
    },
    rng::StatefulRng,
};

/// The only `k` at which the guarded rate is defined.
const K0: f64 = 2.0;
/// Half-width of the pinhole around [`K0`]. Far below the ~1.0 proposal SD, so
/// a proposal landing inside it has probability ~1e-9.
const PINHOLE: f64 = 1e-9;
const MU0: f64 = 0.01;
const N0: f64 = 100.0;
const DT: f64 = 1.0;
const T_END: f64 = 20.0;
const OBS_TIMES: [f64; 4] = [5.0, 10.0, 15.0, 20.0];

const N_SWEEPS: usize = 24;
const N_PARTICLES: usize = 8;
const SEED: u64 = 20260724;

/// Model parameter indices (order of `Model::parameters`).
const MU_IDX: usize = 0;
const K_IDX: usize = 1;
/// Positions in `if2_params` — `k` is proposed first within each MH sweep.
const K_SLOT: usize = 0;
const MU_SLOT: usize = 1;

/// `sqrt(k − 1e9)` — `NaN` for every reachable `k`, which `eval_propensities`
/// surfaces as `SimError::NumericalCollapse` (per-particle recoverable).
fn degenerate_branch() -> Expr {
    Expr::un_op(
        UnOp::Sqrt,
        Expr::bin_op(BinOp::Sub, Expr::param("k"), Expr::const_(1e9)),
    )
}

/// `mu·N` — the well-defined death rate.
fn healthy_branch() -> Expr {
    Expr::bin_op(BinOp::Mul, Expr::param("mu"), Expr::pop("N"))
}

/// `if |k − K0| < PINHOLE then <healthy> else <degenerate>`.
///
/// `Cond` is evaluated lazily (`resolved_expr::eval_resolved`: only the taken
/// branch recurses), so at `k = K0` the degenerate branch is never touched.
fn guarded_rate() -> Expr {
    Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::bin_op(
                BinOp::Lt,
                Expr::un_op(
                    UnOp::Abs,
                    Expr::bin_op(BinOp::Sub, Expr::param("k"), Expr::const_(K0)),
                ),
                Expr::const_(PINHOLE),
            )),
            then: Box::new(healthy_branch()),
            else_: Box::new(degenerate_branch()),
        },
    })
}

/// Pure-death model `N → ∅`. `rate` is supplied so the same scaffolding can
/// build both the guarded fixture and the always-degenerate control.
fn death_model(rate: Expr) -> Arc<CompiledModel> {
    let model = Model {
        ic_grad: Default::default(),
        name: "gh82_guarded_death".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "N".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "death".into(),
            stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
            rate,
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter {
                name: "mu".into(),
                value: ir::parameter::ParamValue::Fixed { value: MU0 },
                param_kind: None,
                param_dim: None,
            },
            Parameter {
                name: "k".into(),
                value: ir::parameter::ParamValue::Fixed { value: K0 },
                param_kind: None,
                param_dim: None,
            },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("N".into(), N0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(OBS_TIMES.to_vec()),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: T_END,
            time_semantics: "continuous".into(),
            dt: Some(DT),
            rng_seed: Some(42),
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
    Arc::new(CompiledModel::new(model).expect("fixture model must compile"))
}

/// Expected deaths per observation interval under the deterministic skeleton,
/// so the Poisson observation density is comfortably finite at the start point.
fn observations() -> Vec<Observation> {
    let mut prev = 0.0;
    OBS_TIMES
        .iter()
        .map(|&t| {
            let survivors = N0 * (-MU0 * t).exp();
            let cum_deaths = N0 - survivors;
            let value = (cum_deaths - prev).round();
            prev = cum_deaths;
            Observation { time: t, value }
        })
        .collect()
}

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let obs = observations();
    MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: sim::inference::multi_stream_obs::StreamProjection::FlowSum(vec![0]),
            ir_model: ir::observation::ObservationModel {
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
                emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                stratum: vec![],
                projection: ir::observation::Projection::CumulativeFlow("death".into()),
                projection_state_grad: Default::default(),
                likelihood: ir::observation::Likelihood::Poisson(
                    ir::observation::PoissonLikelihood {
                        // rate = projected + 0.1 (floor keeps Poisson(0) finite)
                        rate: ir::Diffable::new(Expr::bin_op(
                            BinOp::Add,
                            Expr::Projected(ir::expr::ProjectedExpr { projected: () }),
                            Expr::const_(0.1),
                        )),
                    },
                ),
            },
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

/// `k`: estimated, unconstrained, wide enough that every proposal leaves the
/// pinhole. The initial proposal SD `run_pgas` derives is
/// `(|upper − lower| / 10).max(0.01)` on the transformed scale ⇒ ~1.0 here.
fn k_param() -> EstimatedParam {
    EstimatedParam {
        name: "k".into(),
        index: K_IDX,
        initial: K0,
        rw_sd: 1.0,
        transform: Transform::None,
        lower: 0.0,
        upper: 10.0,
        rw_sd_auto: false,
        ivp: false,
    }
}

/// `mu`: an ordinary estimated parameter, outside the guard. Bounds are tight
/// enough that the derived proposal SD (0.0015) keeps proposals well inside
/// `mu > 0`, so this parameter contributes no errors of its own.
fn mu_param() -> EstimatedParam {
    EstimatedParam {
        name: "mu".into(),
        index: MU_IDX,
        initial: MU0,
        rw_sd: 0.0015,
        transform: Transform::None,
        lower: 0.005,
        upper: 0.02,
        rw_sd_auto: false,
        ivp: false,
    }
}

fn pgas_config() -> PGASConfig {
    PGASConfig {
        n_particles: N_PARTICLES,
        n_sweeps: N_SWEEPS,
        burn_in: 0,
        thin: 1,
        dt: DT,
        use_nuts: false,
        dense_mass: false,
        max_tree_depth: 10,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    }
}

/// Anti-vacuity guard. Drives `complete_data_loglik` — the exact call the MH
/// θ|X step makes — at a `k` outside the pinhole, and pins the error CLASS:
/// per-particle recoverable, and *not* structural. Without this, "the chain
/// completed" could mean the recovery worked OR that nothing ever failed.
#[test]
fn harness_produces_a_recoverable_error_at_a_proposed_theta() {
    let compiled = death_model(guarded_rate());
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let obs_at_substep = build_obs_at_substep(&obs, 0.0, DT).unwrap();

    let mut rng = StatefulRng::new(SEED);
    let params = compiled.default_params.clone();
    let traj = simulate_reference(&compiled, &params, T_END, DT, &mut rng)
        .expect("the reference walk at k = K0 takes the healthy branch");

    // Same trajectory, a proposed k one step outside the pinhole.
    let mut proposed = params.clone();
    proposed[K_IDX] = K0 + 0.5;
    let err = complete_data_loglik(
        &compiled, &traj, &proposed, &obs, DT, &obs_m, &[], &obs_at_substep,
    )
    .expect_err("k outside the pinhole must hit the degenerate rate branch");

    assert!(
        err.is_per_particle_recoverable(),
        "the fixture must produce a RECOVERABLE error (the class gh#82 is about); got {err}"
    );
    assert!(
        !err.is_structural(),
        "a recoverable error must not be structural — that implication is what \
         lets the θ-eval boundary reject it as −∞; got {err}"
    );
    assert!(
        matches!(err, SimError::NumericalCollapse { .. }),
        "expected the degenerate-rate collapse, got {err}"
    );

    // And the same call at the current k is clean — so the error above is a
    // property of the PROPOSAL, not of the fixture as a whole.
    let ok = complete_data_loglik(
        &compiled, &traj, &params, &obs, DT, &obs_m, &[], &obs_at_substep,
    )
    .expect("k = K0 must score cleanly");
    assert!(ok.total.is_finite(), "start point must have finite density, got {}", ok.total);
}

/// gh#82 (the fix). Every MH proposal for `k` raises a per-particle-recoverable
/// `SimError`. The chain must reject those proposals and keep sampling — not
/// terminate.
#[test]
fn recoverable_error_at_a_proposed_theta_does_not_kill_the_chain() {
    let compiled = death_model(guarded_rate());
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let if2_params = vec![k_param(), mu_param()];
    let priors = vec![
        Prior::Fixed(sim::inference::prior::Density::Flat),
        Prior::Fixed(sim::inference::prior::Density::Flat),
    ];
    let base_params = compiled.default_params.clone();

    let result = run_pgas(
        &compiled,
        &if2_params,
        &priors,
        &base_params,
        &pgas_config(),
        &obs,
        &obs_m,
        SEED,
        None,
        None,
        "gh82".into(),
    )
    .expect(
        "a recoverable error at a PROPOSED θ must reject that proposal, not \
         tear the chain down (gh#82)",
    );

    // The chain ran to completion and produced usable output.
    assert_eq!(
        result.sweeps.len(),
        N_SWEEPS,
        "burn_in = 0, thin = 1 ⇒ every sweep is recorded"
    );
    for s in &result.sweeps {
        assert!(
            s.log_complete_data_ll.is_finite(),
            "sweep {} has non-finite complete-data ll {} — the chain must keep a \
             usable state, not drift to −∞",
            s.sweep,
            s.log_complete_data_ll
        );
    }

    // Non-vacuity: `k`'s acceptance rate is exactly 0, so no proposal ever
    // landed inside the pinhole ⇒ EVERY sweep drove a proposal into the
    // degenerate branch and recovered from it.
    assert_eq!(
        result.acceptance_rates[K_SLOT], 0.0,
        "every k proposal must be rejected (each one raises the recoverable \
         error); a non-zero rate means some proposal landed in the pinhole and \
         the test no longer exercises the recovery path"
    );
    for s in &result.sweeps {
        assert_eq!(
            s.params[K_IDX], K0,
            "a rejected proposal must leave k at its current value"
        );
    }

    // The sweep continued PAST the recovered error: `mu` is proposed after `k`
    // in the same MH loop, and it moves.
    assert!(
        result.acceptance_rates[MU_SLOT] > 0.0,
        "mu must still be sampled after k's proposal is rejected; acceptance = {}",
        result.acceptance_rates[MU_SLOT]
    );
    let mu_values: Vec<f64> = result.sweeps.iter().map(|s| s.params[MU_IDX]).collect();
    assert!(
        mu_values.iter().any(|&m| m != MU0),
        "mu must actually move across sweeps, got {mu_values:?}"
    );

    // The recorded likelihoods vary — CSMC renewed the trajectory each sweep,
    // so these are real draws, not a frozen replay of the start point.
    let first = result.sweeps[0].log_complete_data_ll;
    assert!(
        result.sweeps.iter().any(|s| s.log_complete_data_ll != first),
        "the complete-data ll must vary across sweeps (CSMC ran); all equal {first}"
    );
}

/// Control. The fix is scoped to the θ-PROPOSAL site: an error raised anywhere
/// else in `run_pgas` must still terminate the chain. Here the rate is
/// unconditionally degenerate, so the *reference walk* at the start point fails
/// — before any proposal exists to reject.
#[test]
fn degenerate_error_outside_the_proposal_site_still_terminates_the_chain() {
    let compiled = death_model(degenerate_branch());
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let if2_params = vec![k_param(), mu_param()];
    let priors = vec![
        Prior::Fixed(sim::inference::prior::Density::Flat),
        Prior::Fixed(sim::inference::prior::Density::Flat),
    ];
    let base_params = compiled.default_params.clone();

    // `PGASResult` is not `Debug`, so unwrap by hand rather than `expect_err`.
    let err = match run_pgas(
        &compiled,
        &if2_params,
        &priors,
        &base_params,
        &pgas_config(),
        &obs,
        &obs_m,
        SEED,
        None,
        None,
        "gh82-control".into(),
    ) {
        Ok(_) => panic!(
            "a model that cannot run at its OWN start point must still fail loudly — \
             the gh#82 fix must not blanket-swallow errors in run_pgas"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, SimError::NumericalCollapse { .. }),
        "expected the degenerate-rate collapse to propagate verbatim, got {err}"
    );
}
