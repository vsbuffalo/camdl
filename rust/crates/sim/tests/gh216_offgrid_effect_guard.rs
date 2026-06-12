//! gh#216 STOPGAP GUARD — hard-error the buggy combination loudly until the
//! real fix (cursor-keyed effect firing) lands.
//!
//! See docs/dev/proposals/2026-06-11-spine-effect-firing-consolidation.md §1.4
//! (the re-tiling), §9 phase 1 (the stopgap).
//!
//! The defect: under `StepPolicy::Exact` the inference filters (bootstrap PF,
//! IF2, correlated-PF, PGAS producer) re-anchor the substep grid at each
//! OBSERVATION time (`build_substep_grid` sets `window_start = obs_t`,
//! `pgas.rs:405`). The effect-firing DECISION, however, is made on a SECOND clock
//! — `round(t/dt)` against a precomputed `fire_steps` table. When an OBSERVATION
//! time is OFF the integrator grid, it re-tiles the Exact substep grid so that an
//! intervention — *even one whose fire time is ON the dt grid* — lands on a
//! substep that `round(t/dt)` rounds to the wrong step. That is the gh#216
//! reproduction: an intervention at day 35 (= 5·7, ON the dt=7 grid) fires at
//! day 35 with AFP-only obs, but at day 37 once a biweekly ES stream contributes
//! OFF-grid observation times. So the TRIGGER is OFF-GRID OBSERVATIONS, not
//! off-grid effect times. Until the firing decision is moved onto the timeline
//! (the real fix), reject the combination loudly.
//!
//! Scope of this stopgap (exactly):
//!   1. `Exact` + an intervention/event-bearing model + any OBSERVATION time OFF
//!      the dt grid → hard-error naming the `obs_alignment = "snap"` workaround,
//!      for EACH of PF / IF2 / correlated-PF / PGAS. The intervention's own fire
//!      time being ON-grid does NOT save it — the obs re-tile the grid.
//!   2. ON-grid obs + scheduled intervention → STILL FITS (no over-rejection —
//!      the He-2010 weekly-obs and gh#187 on-grid-intervention cases must keep
//!      working).
//!   3. OFF-grid obs + NO interventions → STILL FITS (proposal B's plain
//!      multi-cadence case — nothing to misfire, must NOT be blocked).

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    intervention::{Action, Intervention, InterventionSchedule, InterventionKind, SetAction},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        obs_loglik::poisson_logpmf,
        correlated_pf::{bootstrap_filter_correlated, PFRandomState},
        if2::{run_if2, EstimatedParam, IF2Config, Transform},
        multi_stream_obs::{dense_cells, BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec},
        particle_filter::{bootstrap_filter, Observation},
        pgas::{run_pgas, PGASConfig},
        pmmh::Prior,
        ChainBinomialProcess,
        traits::{ObservationModel, SMCConfig},
        ParticleState,
    },
    rng::StatefulRng,
    schedule::StepPolicy,
};

struct PoissonPrevalenceObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonPrevalenceObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        poisson_logpmf(self.observations[obs_idx], (state.counts[0] as f64).max(0.1))
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
}

/// Pure-death N model with a single SCHEDULED (`Scenario`, NOT always-active)
/// intervention that `set`s N to 50 at the given `schedule`. The schedule is
/// parameterized so a test can build an ON-grid (e.g. {4.0}) version; the gh#216
/// trigger lives on the OBSERVATION grid, not the intervention grid, so every
/// reproduction below keeps the intervention ON-grid and moves the OBS off-grid.
fn death_model_scenario(schedule: InterventionSchedule) -> CompiledModel {
    death_model_with_kind(schedule, InterventionKind::Scenario)
}

/// As [`death_model_scenario`] but with the intervention `kind` selectable, and
/// with `interventions` empty when `schedule` carries no fire times AND the
/// caller wants a truly intervention-free model (see [`no_intervention_model`]).
fn death_model_with_kind(schedule: InterventionSchedule, kind: InterventionKind) -> CompiledModel {
    build_model(vec![Intervention {
        name: "campaign".into(),
        base_name: None,
        schedule,
        actions: vec![Action::Set(SetAction {
            compartment: "N".into(),
            value: Expr::const_(50.0),
        })],
        kind,
    }])
}

/// A model with NO interventions at all (`interventions: vec![]`), so the gh#216
/// guard's "intervention-bearing?" precondition is false — proposal B's plain
/// multi-cadence case, which must run under Exact with off-grid obs.
fn no_intervention_model() -> CompiledModel {
    build_model(vec![])
}

fn build_model(interventions: Vec<Intervention>) -> CompiledModel {
    let model = Model {
        name: "death_scenario_guard".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![Compartment { name: "N".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            name: "death".into(),
            stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions,
        observations: vec![],
        bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("N".into(), 100.0); m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 10.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 10.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    CompiledModel::new(model).unwrap()
}

fn obs_for(obs_times: &[f64]) -> PoissonPrevalenceObs {
    PoissonPrevalenceObs {
        observations: obs_times.iter().map(|&t| 100.0 * (-0.01 * t).exp()).collect(),
        obs_times: obs_times.to_vec(),
    }
}

fn smc_config(dt: f64) -> SMCConfig {
    SMCConfig {
        n_particles: 50, dt, t_start: 0.0,
        skip_first_obs_from_loglik: false, record_ancestry: false,
        record_prequential: false, pf_wallclock_disabled: false,
    }
}

/// Extract the error message, panicking with `what` if the call succeeded.
/// (The Ok variants — `PFilterResult` / `IF2Result` / `PGASResult` — do not
/// implement `Debug`, so `expect_err` is unavailable.)
fn expect_guard_err<T>(r: Result<T, sim::SimError>, what: &str) -> String {
    match r {
        Ok(_) => panic!("{what}"),
        Err(e) => format!("{e}"),
    }
}

fn assert_names_snap(msg: &str) {
    assert!(
        msg.contains("obs_alignment") || msg.contains("snap"),
        "guard must name the obs_alignment = \"snap\" workaround, got: {msg}"
    );
}

fn if2_config() -> IF2Config {
    IF2Config {
        n_particles: 50, n_iterations: 1, cooling_fraction: 0.5,
        cooling_target_iters: 50, dt: 1.0, t_start: 0.0,
        simplex_groups: vec![], skip_first_obs_from_loglik: false,
        pf_wallclock_disabled: false,
    }
}

fn mu_estimated(compiled: &CompiledModel) -> Vec<EstimatedParam> {
    vec![EstimatedParam {
        name: "mu".into(), index: compiled.param_index["mu"],
        initial: 0.01, rw_sd: 0.05,
        transform: Transform::Log { lo: 1e-4, hi: 1.0 },
        lower: 1e-4, upper: 1.0, rw_sd_auto: false, ivp: false,
    }]
}

fn pgas_config(step_policy: StepPolicy) -> PGASConfig {
    PGASConfig {
        n_particles: 10, n_sweeps: 1, burn_in: 0, thin: 1, dt: 1.0,
        use_nuts: false, dense_mass: false, max_tree_depth: 4,
        tempering: vec![1.0], trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy,
    }
}

fn pgas_observations(obs_times: &[f64]) -> Vec<Observation> {
    obs_times.iter().map(|&t| Observation {
        time: t,
        value: 100.0 * (-0.01 * t).exp(),
    }).collect()
}

/// A single-stream PREVALENCE Poisson observation model on compartment `N`
/// (`IntCompSum([0])` / `CurrentPop`), with cells/times matching `obs_times`.
/// Used so the PGAS PASS cases actually score; for the REJECT cases the guard
/// fires before scoring, but a valid model keeps the test honest either way.
fn pgas_obs_model(compiled: &Arc<CompiledModel>, obs_times: &[f64]) -> MultiStreamObsModel {
    let values: Vec<f64> = obs_times.iter().map(|&t| 100.0 * (-0.01 * t).exp()).collect();
    let spec = StreamSpec {
        projection: StreamProjection::IntCompSum(vec![0]), // N
        ir_model: ir::observation::ObservationModel {
            name: "n_obs".into(),
            source: "n_obs".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "n_obs".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "n_obs".into(),
            emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: ir::observation::Projection::CurrentPop("N".into()),
            likelihood: ir::observation::Likelihood::Poisson(
                ir::observation::PoissonLikelihood {
                    rate: ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                        bin_op: ir::expr::BinOpExpr {
                            op: ir::expr::BinOp::Add,
                            left: Box::new(ir::expr::Expr::Projected(
                                ir::expr::ProjectedExpr { projected: () })),
                            right: Box::new(ir::expr::Expr::Const(
                                ir::expr::ConstExpr { value: 0.1 })),
                        },
                    }),
                }),
        },
        observations: dense_cells(values),
        obs_times: obs_times.to_vec(),
        aux: vec![],
    };
    MultiStreamObsModel::new(
        BoundObs::bind(vec![spec]).unwrap().0,
        compiled.clone(),
    ).unwrap()
}

fn run_pgas_exact(
    compiled: &Arc<CompiledModel>,
    obs_times: &[f64],
    run_id: &str,
) -> Result<(), sim::SimError> {
    let params = compiled.default_params.clone();
    let if2_params = mu_estimated(compiled);
    let priors = vec![Prior::Flat];
    let obs_model = pgas_obs_model(compiled, obs_times);
    let observations = pgas_observations(obs_times);
    let config = pgas_config(StepPolicy::Exact);
    run_pgas(
        compiled, &if2_params, &priors, &params, &config,
        &observations, &obs_model, 1, None, None, run_id.into(),
    ).map(|_| ())
}

// ── 1. THE KEY REPRODUCTION: ON-grid intervention + OFF-grid obs → hard-error ──
//
// This is the case the OLD (intervention-time-keyed) guard MISSED: the
// intervention fires at an on-grid time (day 4 = 4·1), but the obs are off the
// grid (3.5, 7.5), which re-tile the Exact substep grid and misfire the
// intervention. The guard must reject it for every Exact filter.

/// PF: on-grid scheduled intervention (fire at t=4.0) + OFF-grid obs (3.5, 7.5)
/// under the hardcoded Exact bootstrap PF must be rejected.
#[test]
fn pf_rejects_offgrid_obs_with_on_grid_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled, 1.0);
    let obs = obs_for(&[3.5, 7.5]); // OFF the dt=1 grid
    let msg = expect_guard_err(
        bootstrap_filter(&process, &obs, &params, &smc_config(1.0), 42),
        "off-grid obs + on-grid intervention under Exact PF must be rejected",
    );
    assert_names_snap(&msg);
}

/// IF2: same — on-grid intervention + off-grid obs must be rejected.
#[test]
fn if2_rejects_offgrid_obs_with_on_grid_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone(), 1.0);
    let obs = obs_for(&[3.5, 7.5]);
    let if2_params = mu_estimated(&compiled);
    let msg = expect_guard_err(
        run_if2(&process, &obs, &params, &if2_params, &if2_config(), 42),
        "off-grid obs + on-grid intervention under Exact IF2 must be rejected",
    );
    assert_names_snap(&msg);
}

/// Correlated-PF: same — on-grid intervention + off-grid obs must be rejected.
/// The gh#216 guard fires BEFORE the CPM uniform-grid validator, so this is the
/// guard's message (naming snap), not the CPM non-uniformity message.
#[test]
fn correlated_pf_rejects_offgrid_obs_with_on_grid_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone(), 1.0);
    let obs = obs_for(&[3.5, 7.0]); // 3.5 is off-grid → guard fires
    let config = smc_config(1.0);
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(config.n_particles, 2, 4, n_source_groups, &mut rng);
    let msg = expect_guard_err(
        bootstrap_filter_correlated(&process, &obs, &params, &config, &randoms, 42),
        "off-grid obs + on-grid intervention under Exact correlated-PF must be rejected",
    );
    assert_names_snap(&msg);
}

/// PGAS: same — on-grid intervention + off-grid obs must be rejected (widening
/// the existing events-only PGAS guard to also cover the off-grid-obs case).
#[test]
fn pgas_rejects_offgrid_obs_with_on_grid_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    let msg = expect_guard_err(
        run_pgas_exact(&compiled, &[3.5, 7.5], "gh216_pgas_offgrid"),
        "off-grid obs + on-grid intervention under Exact PGAS must be rejected",
    );
    assert_names_snap(&msg);
}

// ── 2. ON-grid obs + intervention: STILL FITS (He-2010 / gh#187 shape) ────────

/// He-2010 shape: ON-grid obs + a scheduled intervention is CORRECT today and
/// must keep running. If the guard rejects this, the obs-grid check is wrong.
#[test]
fn pf_accepts_on_grid_obs_with_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled, 1.0);
    let obs = obs_for(&[4.0, 8.0]); // ON the dt=1 grid
    let res = bootstrap_filter(&process, &obs, &params, &smc_config(1.0), 42)
        .expect("on-grid obs + intervention under Exact must still fit (no over-rejection)");
    assert!(res.log_likelihood.is_finite(),
        "on-grid obs + intervention: loglik should be finite, got {}", res.log_likelihood);
}

/// gh#187 shape (no over-rejection): on-grid obs + on-grid intervention under
/// the Exact PGAS producer must still fit — this is exactly the
/// `gh187_pgas_scheduled_intervention` regression's combination.
#[test]
fn pgas_accepts_on_grid_obs_with_on_grid_intervention() {
    let compiled = Arc::new(death_model_scenario(InterventionSchedule::AtTimes(vec![4.0])));
    run_pgas_exact(&compiled, &[4.0, 8.0], "gh216_pgas_ongrid")
        .expect("on-grid obs + on-grid intervention under Exact PGAS must still fit");
}

// ── 3. OFF-grid obs + NO interventions: STILL FITS (proposal B) ───────────────

/// Proposal B's plain multi-cadence case: a model with NO interventions runs
/// under Exact even with OFF-grid obs. Nothing can misfire, so the guard must
/// NOT block it.
#[test]
fn pf_accepts_offgrid_obs_no_interventions() {
    let compiled = Arc::new(no_intervention_model());
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled, 1.0);
    let obs = obs_for(&[3.5, 7.5]); // OFF-grid obs, but no intervention
    let res = bootstrap_filter(&process, &obs, &params, &smc_config(1.0), 42)
        .expect("no-intervention model must run under Exact with off-grid obs (proposal B)");
    assert!(res.log_likelihood.is_finite());
}

/// Same for PGAS: off-grid obs + no interventions must fit (proposal B).
#[test]
fn pgas_accepts_offgrid_obs_no_interventions() {
    let compiled = Arc::new(no_intervention_model());
    run_pgas_exact(&compiled, &[3.5, 7.5], "gh216_pgas_no_iv")
        .expect("no-intervention model must run under Exact PGAS with off-grid obs (proposal B)");
}

/// SCOPE PIN: a model with an always-active EVENT (not a Scenario) + ON-grid obs
/// is NOT rejected — events are guarded separately by PGAS's events-only guard;
/// PF/IF2/correlated-PF key event firing on the nominal grid. The He-2010
/// cohort-entry case (an always-active event) with on-grid weekly obs runs and
/// agrees with pomp; this stopgap must not break it.
#[test]
fn pf_accepts_on_grid_obs_with_event() {
    let compiled = Arc::new(death_model_with_kind(
        InterventionSchedule::AtTimes(vec![4.0]),
        InterventionKind::Event,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled, 1.0);
    let obs = obs_for(&[4.0, 8.0]); // ON-grid obs
    let res = bootstrap_filter(&process, &obs, &params, &smc_config(1.0), 42)
        .expect("on-grid obs + always-active event under Exact PF must still fit");
    assert!(res.log_likelihood.is_finite());
}

/// SCOPE PIN (load-bearing): an always-active EVENT (NOT a scheduled Scenario) +
/// OFF-grid obs is NOT rejected. Always-active events key their firing on the
/// NOMINAL `grid_dt` (the StepClock fix, spine-v2 §A), so an off-grid obs that
/// clips the final substep does NOT misfire the event — `dt_actual ≠ grid_dt`
/// but the key is `grid_dt`. This guard targets SCHEDULED interventions only;
/// rejecting an off-grid-obs event model would over-reject and contradict
/// `inference_event_misfire_guard::pf_runs_off_grid_obs_with_always_active_event`.
#[test]
fn pf_accepts_offgrid_obs_with_event_only() {
    let compiled = Arc::new(death_model_with_kind(
        InterventionSchedule::AtTimes(vec![4.0]), // event fires at an on-grid time
        InterventionKind::Event,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled, 1.0);
    let obs = obs_for(&[3.5, 7.5]); // OFF-grid obs, but no SCHEDULED intervention
    let res = bootstrap_filter(&process, &obs, &params, &smc_config(1.0), 42)
        .expect("off-grid obs + events-only model under Exact PF must NOT be rejected \
                 (events key on grid_dt; only scheduled interventions misfire)");
    assert!(res.log_likelihood.is_finite());
}
