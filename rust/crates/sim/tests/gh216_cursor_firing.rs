//! gh#216 FIX: scheduled-intervention firing is decided from the TIMELINE
//! (cursor-keyed off the registered effect boundaries), not from `round(t/dt)`.
//!
//! See docs/dev/proposals/2026-06-11-spine-effect-firing-consolidation.md.
//!
//! The bug (pre-fix, held back by a stopgap): under `StepPolicy::Exact` the
//! inference filters re-anchor the substep grid at every OBSERVATION time, but
//! the firing DECISION for a SCHEDULED intervention was made on a SECOND clock —
//! `round(t/dt)` against a precomputed `fire_steps` table. An OFF-grid observation
//! re-tiled the Exact grid so an otherwise ON-grid intervention's substep landed
//! at a time `round(t/dt)` snapped to the WRONG step — a silent likelihood error
//! whose firing instant moved when the observation streams moved (day 35 → 37).
//!
//! These tests pin the fixed behaviour across PF / IF2 / correlated-PF and the
//! PGAS producer:
//!   - Test 1 (invariance): the intervention's firing INSTANT is invariant to
//!     which observation streams are present — it fires at its own time
//!     regardless of an added off-grid observation. RED before the fix (the
//!     stopgap rejected the combination); GREEN after.
//!   - Test 2 (on-grid bit-identity): on-grid obs + on-grid intervention reproduce
//!     the exact deterministic likelihood (firing lands on the same substep
//!     `round()` did). GREEN throughout — the load-bearing byte-identity gate.
//!   - Test 5 (PGAS producer): the reference producer fires once, at the
//!     registered boundary; value/gradient score those records.
//!   - Test 7 (no over-rejection + the residual guards): on-grid fits, events fit,
//!     intervention-free fits; off-grid effect TIME and `AtTimesExpr`+Exact are
//!     refused loudly.
//!
//! The observable is deterministic by construction: a pure-death model with
//! `mu = 0` (no transition fires, no RNG consumed, every particle identical), so
//! the only state change is the intervention — an absolute transfer `N → M` of 10
//! (NON-idempotent, unlike a `set`, so a mis-timed or double firing is visible in
//! `M`). The bootstrap-PF marginal likelihood then collapses to the exact sum of
//! per-observation Poisson log-PMFs over the deterministic `M` trajectory.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    intervention::{
        AbsoluteTransfer, Action, Intervention, InterventionKind, InterventionSchedule,
    },
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
    inference::{
        correlated_pf::{bootstrap_filter_correlated, PFRandomState},
        if2::{run_if2, EstimatedParam, IF2Config, Transform},
        multi_stream_obs::{dense_cells, BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec},
        obs_loglik::poisson_logpmf,
        particle_filter::{bootstrap_filter, Observation},
        pgas::{build_substep_grid, run_pgas, simulate_reference_on_grid, PGASConfig},
        pmmh::Prior,
        traits::{ObservationModel, SMCConfig},
        ChainBinomialProcess, ParticleState,
    },
    intervention::timeline_effects,
    rng::StatefulRng,
    schedule::StepPolicy,
};

const TRANSFER: f64 = 10.0; // N → M absolute transfer amount per firing
const FIRE_TIME: f64 = 4.0; // on-grid (dt = 1) intervention time

// ── Observation model: Poisson on M (compartment index 1) ───────────────────
// M reveals firing: 0 before, TRANSFER after a single fire, 2·TRANSFER if a
// double-fire occurred. `.max(0.1)` keeps the rate positive at M = 0.
struct PoissonMObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonMObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        poisson_logpmf(self.observations[obs_idx], (state.counts[1] as f64).max(0.1))
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
}

/// Deterministic pure-death model (`mu = 0`) with N, M compartments and a single
/// SCHEDULED absolute transfer `N → M` of `TRANSFER` on `schedule`. A `dummy`
/// parameter (unused in any rate) gives IF2 something to estimate without
/// perturbing the dynamics. `kind` selects scheduled (`Scenario`) vs event.
fn firing_model(schedule: InterventionSchedule, kind: InterventionKind) -> CompiledModel {
    let intervention = Intervention {
        name: "campaign".into(),
        base_name: None,
        fire: ir::intervention::FireSource::Scheduled(schedule),
        actions: vec![Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "N".into(),
            dst: "M".into(),
            count: Expr::const_(TRANSFER),
        })],
        kind,
    };
    build_model(vec![intervention])
}

fn no_intervention_model() -> CompiledModel {
    build_model(vec![])
}

fn build_model(interventions: Vec<Intervention>) -> CompiledModel {
    let model = Model {
        name: "gh216_firing".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "N".into(), kind: CompartmentKind::Integer },
            Compartment { name: "M".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            name: "death".into(),
            stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
            // rate = mu * N, with mu fixed at 0 → no death, fully deterministic.
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "mu".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                },
            }),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions,
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "mu".into(), value: ir::parameter::ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None },
            Parameter { name: "dummy".into(), value: ir::parameter::ParamValue::Fixed { value: 1.0 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("N".into(), 100.0);
            m.insert("M".into(), 0.0);
            m
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
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
    };
    CompiledModel::new(model).unwrap()
}

/// The deterministic `M` trajectory value at observation time `t`: `TRANSFER`
/// once the intervention at `FIRE_TIME` has fired (the substep ending at
/// `FIRE_TIME`, scored at any obs `>= FIRE_TIME`), else 0.
fn m_at(t: f64) -> f64 {
    if t >= FIRE_TIME - 1e-9 { TRANSFER } else { 0.0 }
}

/// The EXACT bootstrap-PF marginal log-likelihood for a deterministic (`mu = 0`)
/// run that fires ONCE at `FIRE_TIME`: every particle is identical, so each
/// observation's increment collapses to `poisson_logpmf(data, M.max(0.1))`.
fn expected_loglik_fired_once(obs_times: &[f64], data: &[f64]) -> f64 {
    obs_times
        .iter()
        .zip(data)
        .map(|(&_t, &y)| poisson_logpmf(y, y.max(0.1)))
        .sum()
}

/// Data = the deterministic `M` trajectory (so `data == M`, and the expected
/// loglik is `Σ poisson_logpmf(M, M.max(0.1))`).
fn obs_on_m(obs_times: &[f64]) -> (PoissonMObs, Vec<f64>) {
    let data: Vec<f64> = obs_times.iter().map(|&t| m_at(t)).collect();
    (PoissonMObs { observations: data.clone(), obs_times: obs_times.to_vec() }, data)
}

fn smc_config() -> SMCConfig {
    SMCConfig {
        n_particles: 64, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false, record_ancestry: false,
        record_prequential: false, max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    }
}

/// Run the bootstrap PF with ancestry recording and return the EXACT integer `M`
/// compartment at each observation (every particle is identical under `mu = 0`,
/// so particle 0 is representative). `M` is an integer count, so this observes
/// the firing instant with no floating-point tolerance: `M` is 0 before the
/// intervention fires and `TRANSFER` once (and only once) it has.
fn pf_m_trajectory(compiled: &Arc<CompiledModel>, obs_times: &[f64]) -> Vec<i64> {
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());
    let (obs, _) = obs_on_m(obs_times);
    let mut config = smc_config();
    config.record_ancestry = true;
    let res = bootstrap_filter(&process, &obs, &params, &config, 42)
        .expect("bootstrap PF must fit");
    let anc = res.ancestry.expect("ancestry recorded");
    (0..anc.obs_times.len()).map(|i| anc.states[i][0][1] as i64).collect()
}

fn expect_err<T>(r: Result<T, sim::SimError>, what: &str) -> String {
    match r {
        Ok(_) => panic!("{what}"),
        Err(e) => format!("{e}"),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1 — gh#216 invariance: the firing instant is invariant to obs streams
// ════════════════════════════════════════════════════════════════════════════

/// PF: adding an OFF-grid observation (3.5) alongside the obs at the intervention
/// time (4) does NOT move the firing. Read directly off the integer `M`
/// compartment: it is 0 up to day 4 and `TRANSFER` from day 4 on — and CRUCIALLY
/// the day-4 value is the same whether or not the off-grid 3.5 stream is present.
/// RED before the fix (the stopgap rejected off-grid obs + a scheduled
/// intervention); GREEN after.
#[test]
fn pf_firing_invariant_to_offgrid_obs_stream() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));

    // AFP-only: on-grid obs at 4 (= fire time) and 8.
    let afp_only = [4.0, 8.0];
    let m_afp = pf_m_trajectory(&compiled, &afp_only);
    assert_eq!(m_afp, vec![TRANSFER as i64, TRANSFER as i64],
        "AFP-only: fired by the day-4 obs, M stays 10");

    // AFP + off-grid ES stream: obs at 3.5 (off-grid), 4, 8. The day-4 value
    // must be UNCHANGED (10), and the 3.5 obs must see M=0 (no early fire).
    let afp_es = [3.5, 4.0, 8.0];
    let m_es = pf_m_trajectory(&compiled, &afp_es);
    assert_eq!(m_es, vec![0, TRANSFER as i64, TRANSFER as i64],
        "AFP+ES: the off-grid 3.5 obs sees M=0, the intervention still fires at day 4 (M=10), \
         and exactly once (final M=10, not 20) — the firing instant is invariant to the obs stream");
}

/// PF, ALWAYS-ACTIVE EVENT arm (gh#216 events; PR#218 review #4). Events keep the
/// `grid_dt` firing key (NOT cursor-keyed — registering event times would re-tile
/// the Exact grid and break events-only byte-identity). That key must still fire
/// the event at its OWN nominal time (day 4) regardless of an off-grid sibling
/// obs: not early at the off-grid obs boundary (3.5 → `round(3.5)=4` collision)
/// and not twice (3.5 AND 4.0). Read off the integer `M`: a misfire shows as
/// M≠0 at the 3.5 obs (early) or final M=20 (double). The scheduled arm above is
/// cursor-keyed; this pins the SEPARATE event firing path.
#[test]
fn pf_event_firing_invariant_to_offgrid_obs_stream() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Event,
    ));

    let on_grid = pf_m_trajectory(&compiled, &[4.0, 8.0]);
    assert_eq!(on_grid, vec![TRANSFER as i64, TRANSFER as i64],
        "event arm, on-grid baseline: fires once at day 4");

    let off_grid = pf_m_trajectory(&compiled, &[3.5, 4.0, 8.0]);
    assert_eq!(off_grid, vec![0, TRANSFER as i64, TRANSFER as i64],
        "event arm: the off-grid 3.5 obs must see M=0 (no early fire at the obs-anchored \
         boundary), the event fires at day 4 (M=10) and exactly once (final M=10, not 20)");
}

/// IF2: same invariance via the (deterministic) perturbed loglik. `dummy` is the
/// estimated parameter — unused in the dynamics, so the trajectory and loglik are
/// independent of the perturbation. RED before the fix (stopgap); GREEN after.
#[test]
fn if2_firing_invariant_to_offgrid_obs_stream() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());

    let off_grid_times = [3.5, 4.0, 8.0];
    let (obs, data) = obs_on_m(&off_grid_times);
    let if2_params = vec![EstimatedParam {
        name: "dummy".into(), index: compiled.param_index["dummy"],
        initial: 1.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 1e-3, hi: 10.0 },
        lower: 1e-3, upper: 10.0, rw_sd_auto: false, ivp: false,
    }];
    let config = IF2Config {
        n_particles: 64, n_iterations: 1, cooling_fraction: 0.5,
        cooling_target_iters: 50, dt: 1.0, t_start: 0.0,
        simplex_groups: vec![], skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let res = run_if2(&process, &obs, &params, &if2_params, &config, 42)
        .expect("off-grid obs + on-grid intervention must now FIT under IF2 (gh#216 fix)");
    // IF2 exposes only the (deterministic) perturbed loglik, not per-obs state.
    // It collapses to Σ poisson_logpmf(M, M.max(0.1)) up to the logsumexp/ln-N
    // rounding path (a few ULPs); a MIS-timed firing would move M at an obs and
    // shift the loglik by many nats, so a tight-but-finite tolerance is decisive.
    let expected = expected_loglik_fired_once(&off_grid_times, &data);
    assert!(
        (res.last_loglik - expected).abs() < 1e-6,
        "IF2: intervention must fire at day 4 regardless of the off-grid 3.5 obs \
         (loglik {} vs fired-at-4 {})", res.last_loglik, expected
    );
}

/// Correlated-PF: the firing is correct under off-grid obs. CPM requires uniform
/// observation windows, so the obs are uniformly spaced (3.5, 7.0 — 4 substeps
/// each, the on-grid intervention at 4 re-tiles within a window without changing
/// the count). The intervention fires exactly once at day 4, so the final `M = 10`
/// (a double-fire would give 20, a missed fire 0). RED before the fix (stopgap);
/// GREEN after.
#[test]
fn correlated_pf_firing_correct_under_offgrid_obs() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());

    let off_grid_times = [3.5, 7.0];
    let (obs, _data) = obs_on_m(&off_grid_times);
    let config = smc_config();
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(config.n_particles, 2, 4, n_source_groups, &mut rng);
    let res = bootstrap_filter_correlated(&process, &obs, &params, &config, &randoms, 42)
        .expect("off-grid obs + on-grid intervention must now FIT under correlated-PF (gh#216 fix)");
    let final_m = res.final_states.as_ref().unwrap()[0].counts[1];
    assert_eq!(
        final_m, TRANSFER as i64,
        "correlated-PF: intervention must fire EXACTLY ONCE at day 4 (final M=10, not 0 or 20)"
    );
}

/// IF2, EVENT arm (gh#216 events / PR#218 #4): the same off-grid firing
/// invariance via the perturbed loglik, with an always-active EVENT instead of a
/// scheduled intervention. Events are now cursor-keyed like interventions
/// (`split_due_batch`), so the event fires at day 4 regardless of the off-grid
/// 3.5 obs — a mis-timed firing would shift `M` at an obs and move the loglik.
#[test]
fn if2_event_firing_invariant_to_offgrid_obs_stream() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Event,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());

    let off_grid_times = [3.5, 4.0, 8.0];
    let (obs, data) = obs_on_m(&off_grid_times);
    let if2_params = vec![EstimatedParam {
        name: "dummy".into(), index: compiled.param_index["dummy"],
        initial: 1.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 1e-3, hi: 10.0 },
        lower: 1e-3, upper: 10.0, rw_sd_auto: false, ivp: false,
    }];
    let config = IF2Config {
        n_particles: 64, n_iterations: 1, cooling_fraction: 0.5,
        cooling_target_iters: 50, dt: 1.0, t_start: 0.0,
        simplex_groups: vec![], skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let res = run_if2(&process, &obs, &params, &if2_params, &config, 42)
        .expect("off-grid obs + always-active event must fit under IF2 (gh#216 events arm)");
    let expected = expected_loglik_fired_once(&off_grid_times, &data);
    assert!(
        (res.last_loglik - expected).abs() < 1e-6,
        "IF2 event: must fire at day 4 regardless of the off-grid 3.5 obs \
         (loglik {} vs fired-at-4 {})", res.last_loglik, expected
    );
}

/// Correlated-PF, EVENT arm: an always-active event fires exactly once at day 4
/// under off-grid (uniform) obs — final `M = 10`, not 0 (missed) or 20 (double).
#[test]
fn correlated_pf_event_firing_correct_under_offgrid_obs() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Event,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());

    let off_grid_times = [3.5, 7.0];
    let (obs, _data) = obs_on_m(&off_grid_times);
    let config = smc_config();
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(config.n_particles, 2, 4, n_source_groups, &mut rng);
    let res = bootstrap_filter_correlated(&process, &obs, &params, &config, &randoms, 42)
        .expect("off-grid obs + always-active event must fit under correlated-PF");
    let final_m = res.final_states.as_ref().unwrap()[0].counts[1];
    assert_eq!(
        final_m, TRANSFER as i64,
        "correlated-PF event: must fire EXACTLY ONCE at day 4 (final M=10, not 0 or 20)"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2 — on-grid bit-identity (the load-bearing gate)
// ════════════════════════════════════════════════════════════════════════════

/// On-grid obs + on-grid intervention: the cursor-keyed firing lands on the SAME
/// substep `round()` did. Observed exactly off the integer `M` compartment: 0 at
/// the day-2 obs (before firing), then `TRANSFER` at day-4/6/8 — fired exactly
/// once, on the day-4 substep. GREEN both before and after the fix (registering
/// `effect_times` does not perturb the on-grid substep grid, so the trajectory —
/// and thus the loglik — is byte-identical; the separate `gate_inference_baseline`
/// / `pgas_exact_tiling` ratchets pin the no-effect grid to the bit).
#[test]
fn pf_on_grid_intervention_is_bit_identical() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));
    let on_grid_times = [2.0, 4.0, 6.0, 8.0]; // all on the dt=1 grid
    let m = pf_m_trajectory(&compiled, &on_grid_times);
    assert_eq!(m, vec![0, TRANSFER as i64, TRANSFER as i64, TRANSFER as i64],
        "on-grid intervention fires exactly once, on the day-4 substep");
}

/// No-intervention model under Exact with off-grid obs runs unperturbed: `M`
/// never moves (stays 0). With no scheduled intervention `effect_times` is empty,
/// so the substep grid is byte-identical to the pre-fix path.
#[test]
fn pf_no_intervention_off_grid_bit_identical() {
    let compiled = Arc::new(no_intervention_model());
    let m = pf_m_trajectory(&compiled, &[3.5, 7.5]);
    assert_eq!(m, vec![0, 0], "no intervention ⇒ M never moves");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5 — PGAS producer fires once, at the registered boundary
// ════════════════════════════════════════════════════════════════════════════

/// The PGAS reference producer, walking the Exact grid built with the
/// scheduled-effect boundary registered, fires the intervention EXACTLY ONCE, at
/// the substep landing on day 4 — even with an off-grid observation (3.5) that
/// re-anchors the drift-free walk. `M` jumps 0 → 10 at that substep and stays 10.
#[test]
fn pgas_producer_fires_once_at_registered_boundary() {
    let compiled = firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    );
    let params = compiled.default_params.clone();
    let observations = pgas_observations(&[3.5, 8.0]); // off-grid 3.5 re-tiles
    let scheduled = timeline_effects(&compiled, &params);
    assert_eq!(scheduled.times, vec![FIRE_TIME], "the scheduled boundary is the intervention time");

    let grid = build_substep_grid(0.0, 1.0, &observations, &scheduled.times, StepPolicy::Exact)
        .expect("build exact grid with the registered effect boundary");
    let firing = Some((&grid.effect_at_substep, scheduled.batches.as_slice()));
    let mut rng = StatefulRng::new(1);
    let traj = simulate_reference_on_grid(&compiled, &params, 1.0, &grid.steps, firing, &mut rng)
        .expect("reference producer must simulate");

    // M (local int index 1) jumps to TRANSFER at the firing substep, and stays.
    let fire_substep = traj.substeps.iter()
        .position(|r| r.counts_after[1] == TRANSFER as i64)
        .expect("the intervention must fire (M reaches 10)");
    let rec = &traj.substeps[fire_substep];
    let landing = rec.t0 + rec.dt_substep;
    assert!((landing - FIRE_TIME).abs() < 1e-9,
        "the producer must fire at day 4, got landing {landing}");
    assert_eq!(traj.substeps.last().unwrap().counts_after[1], TRANSFER as i64,
        "the intervention must fire EXACTLY ONCE (M stays 10, never 20)");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 7 — no over-rejection, plus the residual guards
// ════════════════════════════════════════════════════════════════════════════

/// gh#187 shape: on-grid obs + on-grid scheduled intervention under Exact PGAS
/// must still fit (no over-rejection).
#[test]
fn pgas_on_grid_intervention_still_fits() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));
    run_pgas_exact(&compiled, &[2.0, 4.0, 8.0], "gh216_pgas_ongrid")
        .expect("on-grid obs + on-grid intervention under Exact PGAS must fit");
}

/// PGAS: off-grid obs + on-grid intervention now FITS (the case the stopgap
/// rejected — proposal Test 7).
#[test]
fn pgas_offgrid_obs_on_grid_intervention_now_fits() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![FIRE_TIME]),
        InterventionKind::Scenario,
    ));
    run_pgas_exact(&compiled, &[3.5, 8.0], "gh216_pgas_offgrid")
        .expect("off-grid obs + on-grid intervention under Exact PGAS must now FIT (gh#216 fix)");
}

/// Intervention-free model + off-grid obs fits under Exact PGAS (proposal B).
#[test]
fn pgas_no_intervention_off_grid_fits() {
    let compiled = Arc::new(no_intervention_model());
    run_pgas_exact(&compiled, &[3.5, 7.5], "gh216_pgas_no_iv")
        .expect("no-intervention model under Exact PGAS with off-grid obs must fit");
}

/// RESIDUAL GUARD: a scheduled intervention whose fire time is OFF the dt grid is
/// refused loudly under Exact (deferred generalization, naming the snap migration).
#[test]
fn pf_rejects_off_grid_effect_time() {
    let compiled = Arc::new(firing_model(
        InterventionSchedule::AtTimes(vec![4.5]), // OFF the dt=1 grid
        InterventionKind::Scenario,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled);
    let (obs, _) = obs_on_m(&[2.0, 6.0]); // on-grid obs — only the effect time is off-grid
    let msg = expect_err(
        bootstrap_filter(&process, &obs, &params, &smc_config(), 42),
        "an off-grid scheduled intervention time under Exact must be rejected",
    );
    assert!(msg.to_lowercase().contains("off the dt grid") && (msg.contains("snap") || msg.contains("obs_alignment")),
        "guard must name the off-grid effect time and the snap migration, got: {msg}");
}

/// §3.6 HARD-ERROR: a parametric `at [<param>]` schedule under Exact is refused
/// (one shared `effect_times` cannot represent per-particle fire times).
#[test]
fn if2_rejects_at_times_expr_under_exact() {
    use ir::expr::ParamExpr;
    let compiled = Arc::new(firing_model(
        // `at [t_fire]` — a parametric schedule (gh#69 AtTimesExpr).
        InterventionSchedule::AtTimesExpr(vec![Expr::Param(ParamExpr { param: "dummy".into() })]),
        InterventionKind::Scenario,
    ));
    let params = compiled.default_params.clone();
    let process = ChainBinomialProcess::new(compiled.clone());
    let (obs, _) = obs_on_m(&[2.0, 6.0]);
    let if2_params = vec![EstimatedParam {
        name: "dummy".into(), index: compiled.param_index["dummy"],
        initial: 4.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 1.0, hi: 8.0 },
        lower: 1.0, upper: 8.0, rw_sd_auto: false, ivp: false,
    }];
    let config = IF2Config {
        n_particles: 16, n_iterations: 1, cooling_fraction: 0.5,
        cooling_target_iters: 50, dt: 1.0, t_start: 0.0,
        simplex_groups: vec![], skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let msg = expect_err(
        run_if2(&process, &obs, &params, &if2_params, &config, 42),
        "a parametric at[..] schedule under Exact IF2 must be rejected",
    );
    assert!(msg.contains("parametric") && (msg.contains("snap") || msg.contains("obs_alignment")),
        "guard must name the parametric-schedule limit and the snap migration, got: {msg}");
}

// ── PGAS plumbing (single prevalence stream on M) ───────────────────────────

fn pgas_observations(obs_times: &[f64]) -> Vec<Observation> {
    obs_times.iter().map(|&t| Observation { time: t, value: m_at(t) }).collect()
}

fn pgas_obs_model(compiled: &Arc<CompiledModel>, obs_times: &[f64]) -> MultiStreamObsModel {
    let values: Vec<f64> = obs_times.iter().map(|&t| m_at(t)).collect();
    let spec = StreamSpec {
        projection: StreamProjection::IntCompSum(vec![1]), // M
        ir_model: ir::observation::ObservationModel {
            name: "m_obs".into(),
            source: "m_obs".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "m_obs".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "m_obs".into(),
            emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
            projection: ir::observation::Projection::CurrentPop("M".into()),
            likelihood: ir::observation::Likelihood::Poisson(ir::observation::PoissonLikelihood {
                rate: ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                    bin_op: ir::expr::BinOpExpr {
                        op: ir::expr::BinOp::Add,
                        left: Box::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
                        right: Box::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.1 })),
                    },
                }),
            }),
            stratum: vec![],
        },
        observations: dense_cells(values),
        obs_times: obs_times.to_vec(),
        aux: vec![],
    };
    MultiStreamObsModel::new(BoundObs::bind(vec![spec]).unwrap().0, compiled.clone()).unwrap()
}

fn run_pgas_exact(compiled: &Arc<CompiledModel>, obs_times: &[f64], run_id: &str) -> Result<(), sim::SimError> {
    let params = compiled.default_params.clone();
    let if2_params = vec![EstimatedParam {
        name: "dummy".into(), index: compiled.param_index["dummy"],
        initial: 1.0, rw_sd: 0.1,
        transform: Transform::Log { lo: 1e-3, hi: 10.0 },
        lower: 1e-3, upper: 10.0, rw_sd_auto: false, ivp: false,
    }];
    let priors = vec![Prior::Flat];
    let obs_model = pgas_obs_model(compiled, obs_times);
    let observations = pgas_observations(obs_times);
    let config = PGASConfig {
        n_particles: 10, n_sweeps: 1, burn_in: 0, thin: 1, dt: 1.0,
        use_nuts: false, dense_mass: false, max_tree_depth: 4,
        tempering: vec![1.0], trajectory_warmup: 0, csmc_sweeps_per_nuts: 1,
        step_policy: StepPolicy::Exact,
    };
    run_pgas(compiled, &if2_params, &priors, &params, &config, &observations, &obs_model, 1, None, None, run_id.into())
        .map(|_| ())
}

// ════════════════════════════════════════════════════════════════════════════
// Property: firing-invariance over RANDOM schedules (the gh#216 generalization)
// ════════════════════════════════════════════════════════════════════════════

use proptest::prelude::*;

proptest! {
    /// For ANY on-grid effect time and ANY observation schedule (including
    /// OFF-grid obs), the effect fires EXACTLY ONCE at its own time — invariants
    /// A1 (instant-invariance), A2 (fire-once) and A3 (no spurious fire) at once,
    /// read off the integer `M` trajectory (0 before the fire, `TRANSFER` after;
    /// `2·TRANSFER` would be a double-fire). This is the generated-combination
    /// generalization of the example-based firing tests above: it covers the
    /// (effect_time, obs_times) pairs no hand-written table enumerates — exactly
    /// the class that hid the gh#216 events bug (`obs=[3.5], event@4, dt=1`).
    /// Runs for both kinds, since events and scheduled interventions now share
    /// the one cursor-keyed firing path. See docs/dev/scheduling-spine.md §2.
    #[test]
    fn prop_effect_fires_once_at_its_time_for_any_obs_schedule(
        // On-grid effect time ∈ {1.0 .. 8.0} (dt = 1, window [0, 10]).
        fire_step in 1u32..9,
        // Observation times at n·0.5 ∈ {0.5 .. 9.5} — a MIX of on-grid (integer)
        // and OFF-grid (half) times, the off-grid ones being where round(t/dt)
        // collided. 1–5 distinct, sorted ascending.
        obs_halfsteps in prop::collection::vec(1u32..19, 1..6),
        is_event in any::<bool>(),
    ) {
        let kind = if is_event { InterventionKind::Event } else { InterventionKind::Scenario };
        let fire_time = fire_step as f64;
        let compiled = Arc::new(firing_model(
            InterventionSchedule::AtTimes(vec![fire_time]), kind));

        let mut obs_times: Vec<f64> = obs_halfsteps.iter().map(|&n| n as f64 * 0.5).collect();
        obs_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        obs_times.dedup();

        let m = pf_m_trajectory(&compiled, &obs_times);
        let expected: Vec<i64> = obs_times.iter()
            .map(|&t| if t >= fire_time - 1e-9 { TRANSFER as i64 } else { 0 })
            .collect();
        prop_assert_eq!(
            m, expected,
            "effect@{} (kind={:?}) must fire ONCE at its own time — M=0 before / \
             TRANSFER after — regardless of the obs schedule {:?}",
            fire_time, kind, obs_times
        );
    }
}
