//! gh#475: a *structural* error raised inside the PGAS-NUTS gradient closure
//! must terminate the chain with that error, not be laundered into `−∞`.
//!
//! `run_pgas`'s NUTS θ|X step hands `nuts::nuts_step` a plain
//! `Fn(&[f64]) -> (f64, Vec<f64>)`. Inside it, every `Err` from
//! `pgas_grad::complete_data_loglik_grad` used to become
//! `(−∞, zeros)` — including structural errors (`SimError::is_structural()`:
//! `UnknownParameter`, `Validation`, …), which fire for *every* θ. NUTS then
//! explored a flat `−∞` landscape, every proposal was rejected, and the fit
//! exited 0 with a posterior frozen at the start point. The gh#82 sibling
//! (MH-within-Gibbs) already routes through `is_structural()`; this file pins
//! both directions at the NUTS site.
//!
//! # Fixture
//!
//! A pure-death model `N → ∅` at rate `mu·N` (with a compiler-shaped
//! `rate_grad`, so NUTS is active), and an initial-state law
//! `N ~ poisson(rate = N0)`. The law's emitted gradient map `∂rate/∂N0` is the
//! injection point: it is evaluated **only** by
//! `CompiledModel::initial_state_logpdf_grad`, which is called only from
//! `complete_data_loglik_grad` — i.e. only inside the NUTS gradient closure.
//! The value path (`complete_data_loglik`, CSMC, the reference walk) never
//! reads it, so everything before the closure runs cleanly and the error
//! surfaces exactly at the site under test. Each test pins that property
//! directly (the anti-vacuity guards below) so "the chain completed" or "the
//! chain failed" cannot be explained by some other site.
//!
//! - Structural: `∂rate/∂N0 = projected`, the observation-projection leaf,
//!   which has no value outside an observation likelihood. Every name in it
//!   resolves, so `CompiledModel::new` builds the model (gh#938 name-checks
//!   initial-state expressions at build, which is why an undeclared parameter
//!   is no longer an injection this test can use); evaluating it on the
//!   initial-state path returns `SimError::Validation` — structural, at every θ.
//! - Recoverable: `∂rate/∂N0 = sqrt(−1)`, which `eval_expr` reports as
//!   `SimError::NumericalCollapse { SqrtNegative }` — not structural.
//!
//! Both errors fire at every θ, so in each case the NUTS closure fails on its
//! very first call (the current point). The two tests differ in the error class
//! and nothing else.

use std::sync::Arc;

use ir::{
    deriv::DerivEntry,
    expr::{BinOp, Expr, UnOp},
    model::{
        Compartment, CompartmentKind, InitCountLaw, InitSpec, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::PoissonLikelihood,
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::{
    compiled_model::CompiledModel,
    error::{CollapseKind, SimError},
    inference::{
        if2::{EstimatedParam, Transform},
        multi_stream_obs::{StreamSpec, StreamTimes},
        particle_filter::Observation,
        pgas::{
            build_obs_at_substep, complete_data_loglik, run_pgas, simulate_reference, PGASConfig,
            PGASResult,
        },
        pmmh::Prior,
        dense_cells, BoundObs, MultiStreamObsModel,
    },
    rng::StatefulRng,
};

const MU0: f64 = 0.01;
/// Non-integer, so `∂/∂rate log Poisson(x; rate) = x/rate − 1` is never zero
/// at an integer draw `x` — the gradient helper skips the chain rule (and
/// with it the injected expression) when that factor is exactly zero.
const N0: f64 = 100.5;
const DT: f64 = 1.0;
const T_END: f64 = 20.0;
const OBS_TIMES: [f64; 4] = [5.0, 10.0, 15.0, 20.0];

const N_SWEEPS: usize = 6;
const N_PARTICLES: usize = 8;
const SEED: u64 = 20260926;

const MU_IDX: usize = 0;
const N0_IDX: usize = 1;

/// Pure-death model whose initial-state law carries `d_rate_d_n0` as its
/// emitted `∂rate/∂N0`.
fn death_model(d_rate_d_n0: Expr) -> Arc<CompiledModel> {
    let mut law_rate = Diffable::new(Expr::param("N0"));
    law_rate.grad.insert("N0".into(), DerivEntry::Grad(d_rate_d_n0));

    let initial = InitialConditions(
        std::iter::once((
            "N".to_string(),
            InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood { rate: law_rate })),
        ))
        .collect(),
    );

    let mut rate_grad = ir::deriv::ParamGradMap::new();
    rate_grad.insert("mu".into(), DerivEntry::Grad(Expr::pop("N")));

    let model = Model {
        ic_grad: Default::default(),
        name: "gh475_death_with_init_law".into(),
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
            rate: Expr::bin_op(BinOp::Mul, Expr::param("mu"), Expr::pop("N")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad,
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
                name: "N0".into(),
                value: ir::parameter::ParamValue::Fixed { value: N0 },
                param_kind: None,
                param_dim: None,
            },
        ],
        initial_conditions: initial,
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

/// `projected` — a leaf that only an observation likelihood can evaluate.
/// Name-clean, so the model builds; on the initial-state path `eval_expr` has
/// no projected value and returns `SimError::Validation` at every θ.
fn structural_grad() -> Expr {
    projected()
}

fn projected() -> Expr {
    Expr::Projected(ir::expr::ProjectedExpr { projected: () })
}

/// The structural error both structural fixtures raise — the verbatim
/// `SimError` the chain must terminate with.
fn is_injected_structural(e: &SimError) -> bool {
    matches!(e, SimError::Validation(m) if m.contains("Projected expression used outside"))
}

/// `if |mu − MU0| < PINHOLE then 1 else projected` — clean at the start point,
/// structural everywhere else. `eval_expr` evaluates `Cond` lazily, so the
/// `projected` branch is only evaluated off the pinhole. This drives the structural
/// failure out of a *leapfrog* evaluation inside `nuts_step`, not out of the
/// closure's first call at the current point — the two are separate checks in
/// `run_pgas`.
fn structural_off_start_grad() -> Expr {
    use ir::expr::{CondExpr, CondWrap};
    Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::bin_op(
                BinOp::Lt,
                Expr::un_op(
                    UnOp::Abs,
                    Expr::bin_op(BinOp::Sub, Expr::param("mu"), Expr::const_(MU0)),
                ),
                Expr::const_(PINHOLE),
            )),
            then: Box::new(Expr::const_(1.0)),
            else_: Box::new(projected()),
        },
    })
}

/// Half-width of the start-point pinhole in [`structural_off_start_grad`]; far
/// below any leapfrog displacement at the initial step size (0.1).
const PINHOLE: f64 = 1e-12;

/// `sqrt(−1)` — a domain error at every θ, reported as `NumericalCollapse`.
fn recoverable_grad() -> Expr {
    Expr::un_op(UnOp::Sqrt, Expr::const_(-1.0))
}

fn observations() -> Vec<Observation> {
    let mut prev = 0.0;
    OBS_TIMES
        .iter()
        .map(|&t| {
            let cum_deaths = N0 - N0 * (-MU0 * t).exp();
            let value = (cum_deaths - prev).round();
            prev = cum_deaths;
            Observation { time: t, value }
        })
        .collect()
}

fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    use sim::inference::multi_stream_obs::StreamProjection;
    let obs = observations();
    let projection = StreamProjection::FlowSum(vec![0]);
    MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![StreamSpec {
            times: StreamTimes::contiguous_for(
                &projection, 0.0, obs.iter().map(|o| o.time).collect(),
            )
            .unwrap(),
            projection,
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
                covers: None,
                projection: ir::observation::Projection::CumulativeFlow("death".into()),
                projection_state_grad: Default::default(),
                likelihood: ir::observation::Likelihood::Poisson(PoissonLikelihood {
                    rate: Diffable::new(Expr::bin_op(
                        BinOp::Add,
                        Expr::Projected(ir::expr::ProjectedExpr { projected: () }),
                        Expr::const_(0.1),
                    )),
                }),
            },
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            aux: vec![],
        }])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap()
}

/// `mu` is the only estimated parameter. `N0` stays fixed: the injected
/// expression is evaluated for every model parameter the law's grad map names,
/// estimated or not, so it fires either way.
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
        perturb_only_at_t0: false,
    }
}

fn nuts_config() -> PGASConfig {
    PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: N_PARTICLES,
        n_sweeps: N_SWEEPS,
        burn_in: 0,
        thin: 1,
        dt: DT,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 6,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    }
}

fn run(compiled: &Arc<CompiledModel>, label: &str) -> Result<PGASResult, SimError> {
    let obs = observations();
    let obs_m = obs_model(compiled);
    run_pgas(
        compiled,
        &[mu_param()],
        &[Prior::Fixed(sim::inference::prior::Density::Flat)],
        &compiled.default_params.clone(),
        &nuts_config(),
        &obs,
        &obs_m,
        SEED,
        None,
        None,
        label.into(),
    )
}

/// Anti-vacuity: the injected error is reachable ONLY through the gradient.
/// The value path (reference walk, `complete_data_loglik`, and the
/// initial-state density itself) is clean at the start point; the
/// initial-state *gradient* returns exactly the error class under test.
fn assert_error_is_gradient_only(compiled: &Arc<CompiledModel>) -> SimError {
    assert!(
        sim::inference::pgas::nuts_active(true, compiled),
        "the fixture must take the NUTS branch of run_pgas"
    );
    let obs = observations();
    let obs_m = obs_model(compiled);
    let obs_at_substep = build_obs_at_substep(&obs, 0.0, DT).unwrap();
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(SEED);
    let traj = simulate_reference(compiled, &params, T_END, DT, &mut rng)
        .expect("the value path must not touch the injected gradient");
    let ll = complete_data_loglik(compiled, &traj, &params, &obs, DT, &obs_m, &obs_at_substep)
        .expect("the value path must not touch the injected gradient");
    assert!(ll.total.is_finite(), "start point must have finite density, got {}", ll.total);
    compiled
        .initial_state_logpdf(&traj.initial_counts, &[], &params)
        .expect("the initial-state density must not touch the injected gradient");
    let x0 = traj.initial_counts[0] as f64;
    assert!(x0 / N0 - 1.0 != 0.0, "chain-rule factor must be non-zero so the grad map is read");
    compiled
        .initial_state_logpdf_grad(&traj.initial_counts, &[], &params)
        .expect_err("the initial-state gradient must evaluate the injected expression")
}

#[test]
fn harness_structural_error_is_gradient_only_and_structural() {
    let compiled = death_model(structural_grad());
    let err = assert_error_is_gradient_only(&compiled);
    assert!(err.is_structural(), "fixture must produce a STRUCTURAL error; got {err}");
    assert!(is_injected_structural(&err), "expected the projected-leaf Validation, got {err}");
}

#[test]
fn harness_recoverable_error_is_gradient_only_and_not_structural() {
    let compiled = death_model(recoverable_grad());
    let err = assert_error_is_gradient_only(&compiled);
    assert!(!err.is_structural(), "fixture must produce a NON-structural error; got {err}");
    assert!(
        matches!(err, SimError::NumericalCollapse { kind: CollapseKind::SqrtNegative, .. }),
        "expected NumericalCollapse(SqrtNegative), got {err}"
    );
}

/// gh#475 (the fix). A structural failure inside the NUTS gradient closure
/// terminates the chain with the underlying `SimError`.
#[test]
fn structural_error_in_nuts_gradient_terminates_the_chain() {
    let compiled = death_model(structural_grad());
    match run(&compiled, "gh475-structural") {
        Ok(result) => panic!(
            "a structural error in the NUTS gradient must terminate the chain, but \
             run_pgas returned Ok with {} sweeps (mu frozen at {:?}) — the silent \
             frozen-posterior failure of gh#475",
            result.sweeps.len(),
            result.sweeps.iter().map(|s| s.params[MU_IDX]).collect::<Vec<_>>(),
        ),
        Err(e) => assert!(
            is_injected_structural(&e),
            "the underlying structural error must propagate verbatim, got {e}"
        ),
    }
}

/// gh#475, the in-trajectory path. The closure's first call (the current
/// point) succeeds; the structural failure arises at a leapfrog point inside
/// `nuts_step`. It must still terminate the chain with the underlying error.
#[test]
fn structural_error_at_a_leapfrog_point_terminates_the_chain() {
    let compiled = death_model(structural_off_start_grad());

    // Anti-vacuity: clean at the start point, structural one step away.
    let mut rng = StatefulRng::new(SEED);
    let params = compiled.default_params.clone();
    let traj = simulate_reference(&compiled, &params, T_END, DT, &mut rng).unwrap();
    compiled
        .initial_state_logpdf_grad(&traj.initial_counts, &[], &params)
        .expect("the gradient must be clean at the start point (mu = MU0)");
    let mut moved = params.clone();
    moved[MU_IDX] = MU0 + 1e-3;
    let err = compiled
        .initial_state_logpdf_grad(&traj.initial_counts, &[], &moved)
        .expect_err("off the pinhole the gradient must hit the projected leaf");
    assert!(err.is_structural(), "expected a structural error off the pinhole, got {err}");

    match run(&compiled, "gh475-structural-leapfrog") {
        Ok(result) => panic!(
            "a structural error at a leapfrog point must terminate the chain, but \
             run_pgas returned Ok with {} sweeps (mu = {:?})",
            result.sweeps.len(),
            result.sweeps.iter().map(|s| s.params[MU_IDX]).collect::<Vec<_>>(),
        ),
        Err(e) => assert!(
            is_injected_structural(&e),
            "the underlying structural error must propagate verbatim, got {e}"
        ),
    }
}

/// Control (unchanged behaviour). A recoverable failure inside the closure is
/// still scored `−∞`: the sampler rejects, the chain keeps running, and `mu`
/// stays at its current value.
#[test]
fn recoverable_error_in_nuts_gradient_rejects_and_the_chain_continues() {
    let compiled = death_model(recoverable_grad());
    let result = match run(&compiled, "gh475-recoverable") {
        Ok(r) => r,
        Err(e) => panic!(
            "a recoverable error in the NUTS gradient must be scored −∞ and rejected, \
             not terminate the chain; got {e}"
        ),
    };
    assert_eq!(result.sweeps.len(), N_SWEEPS, "burn_in = 0, thin = 1 ⇒ every sweep recorded");
    assert_eq!(
        result.acceptance_rates[0], 0.0,
        "every NUTS proposal must be rejected (the closure scores every θ −∞)"
    );
    for s in &result.sweeps {
        assert_eq!(s.params[MU_IDX], MU0, "a rejected proposal must leave mu unchanged");
        assert_eq!(s.params[N0_IDX], N0, "N0 is fixed");
        assert!(
            s.log_complete_data_ll.is_finite(),
            "sweep {}: the chain must keep a usable state, got {}",
            s.sweep,
            s.log_complete_data_ll
        );
    }
}
