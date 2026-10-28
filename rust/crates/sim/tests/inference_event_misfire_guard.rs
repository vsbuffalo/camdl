//! Run-correctly tests for always-active events under the `Exact` inference
//! stepping policy, after the `StepClock` fix
//! (docs/dev/proposals/2026-06-07-scheduling-spine-v2.md §A, step 1).
//!
//! The inference filters step EXACTLY to each observation time
//! (`StepPolicy::Exact`). When an observation is off the dt grid, the final
//! substep of that window is shortened: the realized substep length `dt_actual`
//! diverges from the nominal model `grid_dt`. An event's firing instant comes
//! from the run's effect plan (on the run's grid, independent of how the
//! observations clip the substeps), and the filter lands on it. The defect the
//! StepClock fix removed was a firing key computed on the *clipped* `dt_actual`,
//! so the shortened substep landed on the wrong step — the event fired on the
//! wrong step, or not at all, a silent likelihood error.
//!
//! Previously this configuration (off-grid obs + always-active event under
//! Exact) was REJECTED at filter setup by `Schedule::reject_event_misfire`. With
//! the firing key now correctly computed on `grid_dt`, the configuration RUNS,
//! and the event fires at the correct nominal grid step. These tests pin that:
//!
//!   1. The off-grid + event case now runs and returns a finite log-likelihood
//!      (was rejected — the red→green proof at the filter level).
//!   2. The on-grid + event and no-event controls still run unchanged.
//!
//! The firing instant itself, with an off-grid observation re-tiling the
//! substeps, is pinned in `gh216_cursor_firing.rs`
//! (`pf_snaps_an_off_grid_effect_time`, the invariance tests).

use std::collections::HashMap;
use std::sync::Arc;
use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
    intervention::{Action, AddAction, Intervention, InterventionSchedule},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        obs_loglik::poisson_logpmf,
        particle_filter::bootstrap_filter,
        ChainBinomialProcess,
        traits::{ObservationModel, SMCConfig},
        ParticleState,
    },
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

/// Pure-death N with an optional always-active importation event firing at the
/// given integer times (`add N += 1` whenever a substep lands on one of those
/// nominal grid steps). The event makes the model carry an always-active
/// intervention.
fn death_model(event_times: Option<Vec<f64>>) -> CompiledModel {
    let interventions = match event_times {
        Some(times) => vec![Intervention {
            name: "importation".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(times)),
            actions: vec![Action::Add(AddAction {
                compartment: "N".into(),
                count: Expr::const_(1.0),
            })],
            kind: ir::intervention::InterventionKind::Event,
        }],
        None => vec![],
    };
    let model = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "death_event_guard".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![Compartment { name: "N".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
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
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::constants({
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
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    CompiledModel::new(model).unwrap()
}

fn run(with_event: bool, obs_times: Vec<f64>) -> Result<f64, sim::SimError> {
    let event_times = with_event.then(|| (1..=10).map(|k| k as f64).collect());
    let compiled = Arc::new(death_model(event_times));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled);
    let obs_model = PoissonPrevalenceObs {
        observations: obs_times.iter().map(|&t| 100.0 * (-0.01 * t).exp()).collect(),
        obs_times,
    };
    let config = SMCConfig {
        n_particles: 100, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false, record_ancestry: false,
        record_prequential: false, record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter(&process, &obs_model, &params, &config, 42).map(|r| r.log_likelihood)
}

/// With an always-active event, off-grid observations RUN with a finite
/// likelihood (pre-fix this was rejected by `reject_event_misfire`).
///
/// NOTE: a finite likelihood does NOT prove the event fired at the RIGHT time —
/// this model's death rate (`mu = 0.01`) is stochastic, so `M`/`N` can't pin the
/// firing instant. This test asserted only finiteness and so passed green the
/// whole time the gh#216 events arm was MISFIRING (early + double fire). Firing
/// *correctness* is now pinned by the deterministic-`mu=0` tests in
/// `gh216_cursor_firing` (`pf_event_firing_invariant_*` + the proptest
/// `prop_effect_fires_once_at_its_time_for_any_obs_schedule`). This case stays as
/// the does-it-run / no-spurious-rejection check only.
#[test]
fn pf_runs_off_grid_obs_with_always_active_event() {
    let ll = run(true, vec![3.5, 7.5])
        .expect("off-grid obs + always-active event must RUN, not be rejected");
    assert!(ll.is_finite(), "log-likelihood should be finite, got {ll}");
}

/// CONTROL: the same event model with on-grid obs runs (always-active events with
/// on-grid observations are the common importation/seeding fit) — unchanged by
/// the fix (on-grid Exact substeps never clip, so `dt_actual == grid_dt`).
#[test]
fn pf_accepts_on_grid_obs_with_always_active_event() {
    let ll = run(true, vec![4.0, 8.0]).expect("on-grid obs + event must run");
    assert!(ll.is_finite(), "log-likelihood should be finite, got {ll}");
}

/// CONTROL: a model WITHOUT always-active events runs with off-grid obs.
#[test]
fn pf_accepts_off_grid_obs_without_event() {
    let ll = run(false, vec![3.5, 7.5]).expect("off-grid obs without event must run");
    assert!(ll.is_finite(), "log-likelihood should be finite, got {ll}");
}
