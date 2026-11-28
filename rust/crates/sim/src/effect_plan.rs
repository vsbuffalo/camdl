//! When each scheduled effect fires: the one authority every simulation and
//! inference path derives its firing key from.
//!
//! An *effect* is an `interventions {}` entry or an `events {}` entry; both
//! lower to one IR `Intervention`. Which instant an effect acts at is a function
//! of its declared time, the backend's clock, the run window `[t_start, t_end]`
//! and the run's origin, and nothing else:
//!
//! - [`FireClock::Grid`] — discrete-time dynamics (chain-binomial forward, and
//!   every inference path that runs chain-binomial steps). An effect fires at
//!   the end of a step, on the run's own grid `t_start + k·dt`, with
//!   `k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋` ([`fire_step`]): the nearest step,
//!   ties to the later one. Inside a shortened last step (`t_end` off the grid)
//!   it fires at the nearer of that step's two actual ends, ties to `t_end`.
//! - [`FireClock::Exact`] — continuous-time dynamics (ODE, Gillespie). An
//!   effect fires at its declared instant; fires within
//!   [`EFFECT_EPS`](crate::schedule::EFFECT_EPS) of one another are one instant.
//!
//! On both clocks an effect fires only inside the window `t_start ≤ t ≤ t_end`
//! (with `GRID_TOL·dt` of slack under Grid and `EFFECT_EPS` under Exact), and
//! one at `t_start` applies to the starting state, before the `t_start` row is
//! recorded or scored ([`EffectPlan::at_origin`]) — unless the run resumes a
//! saved state, which is already post-effect ([`Origin::Resumed`]).
//!
//! The plan says *what is due*; applying a batch is
//! [`crate::effects::apply_event_phases`] followed by the stage-4 tail
//! ([`crate::lifecycle::apply_post_advance`] on the integer backends,
//! [`crate::effects::apply_interventions_continuous`] on the ODE). See
//! `docs/dev/proposals/2026-10-04-effect-fire-time-authority.md` §4–§5.

use std::collections::BTreeMap;

use crate::boundary_times::EffectTimes;
use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::schedule::{EFFECT_EPS, GRID_TOL};

/// The backend's natural time, chosen by the dynamics, not by the algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FireClock {
    /// Discrete-time dynamics (chain-binomial forward and every path that runs
    /// chain-binomial steps): fire on the run's grid `t_start + k·dt`.
    Grid { dt: f64 },
    /// Continuous-time dynamics (ODE, Gillespie): fire at the declared instant.
    Exact,
}

/// Whether the run's starting state has seen the effects at `t_start`. An
/// `init {}` state has not; a resumed state (a contrast fork, gh#322, or a
/// forecast from a saved state, gh#641) already has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Init,
    Resumed,
}

/// One effect that a Grid run fires at a different instant from the one it
/// declares, so the modeller is told. Carried on the plan (so a test can read
/// it) and recorded by [`EffectPlan::build`]
/// ([`crate::run_notes::RunNote::EffectSnapped`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SnapNote {
    /// The effect's name.
    pub effect: String,
    /// Its declared fire time (under [`SnapCause::OffGrid`], the earliest of
    /// its declared times that moved).
    pub declared: f64,
    /// The step end it fires at.
    pub fires_at: f64,
    /// Which rule moved it.
    pub cause: SnapCause,
}

/// Why a Grid run fires an effect away from its declared time.
#[derive(Clone, Debug, PartialEq)]
pub enum SnapCause {
    /// The declared time lies off the grid `t_start + k·dt`, and the effect
    /// fires at the nearest step end ([`fire_step`]). One note covers every
    /// declared time of the effect this moved.
    OffGrid {
        t_start: f64,
        dt: f64,
        /// How many of the effect's fire times in the window moved.
        n_moved: usize,
        /// How many of the effect's fire times lie in the window.
        n_fires: usize,
    },
    /// The declared time falls in the run's shortened last step (`t_end` off
    /// the grid), and the effect fires at the nearer of that step's two ends,
    /// not at `nominal`, the grid point the nominal key names, which the last
    /// step never reaches the far side of.
    ShortenedLastStep { nominal: f64 },
}

impl std::fmt::Display for SnapNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            SnapCause::OffGrid { t_start, dt, n_moved, n_fires } => {
                write!(
                    f,
                    "effect `{}` declared at t = {} fires at t = {}: chain-binomial steps on \
                     the grid t_start + k·dt (t_start = {t_start}, dt = {dt}) and applies an \
                     effect at the end of the step nearest its declared time",
                    self.effect, self.declared, self.fires_at,
                )?;
                if *n_moved > 1 {
                    write!(
                        f,
                        " ({n_moved} of its {n_fires} fire times lie off the grid; this is the first)"
                    )?;
                }
                write!(f, ". A dt that divides each of the effect's times from t_start fires it where it is declared.")
            }
            SnapCause::ShortenedLastStep { nominal } => write!(
                f,
                "effect `{}` declared at t = {} falls in the run's shortened last step and \
                 fires at t = {}, the step boundary nearest it (the run ends at {}, off the \
                 dt grid; the nearest grid point, t = {}, is not a step end of this run)",
                self.effect, self.declared, self.fires_at, self.fires_at, nominal,
            ),
        }
    }
}

/// The effects due over one run, by firing instant, built once per run.
#[derive(Clone, Debug, Default)]
pub struct EffectPlan {
    /// Applied to the starting state before the first step and before the
    /// `t_start` row is recorded or scored. Effect indices (into
    /// `model.interventions`), declaration order. Empty under
    /// [`Origin::Resumed`].
    at_origin: Vec<usize>,
    /// Firing instants after `t_start`, ascending, index-aligned with
    /// `batches`. Grid: `t_start + k·dt`, or `t_end` for the last step; Exact:
    /// the declared time.
    times: Vec<f64>,
    /// Grid only (empty under Exact): the step `k` (`1 ≤ k ≤ n_substeps`)
    /// ending at `times[i]`.
    steps: Vec<u64>,
    /// The effects firing at `times[i]`, declaration order.
    batches: Vec<Vec<usize>>,
    /// Effects fired away from their declared time (Grid only).
    notes: Vec<SnapNote>,
}

/// Number of steps a Grid run over `[t_start, t_end]` takes: full steps of `dt`,
/// the last one shortened to land on `t_end` when the window is not a whole
/// number of steps (to within `GRID_TOL`).
pub fn n_substeps(t_start: f64, t_end: f64, dt: f64) -> u64 {
    let span = (t_end - t_start) / dt;
    if span <= GRID_TOL {
        return 0;
    }
    (span - GRID_TOL).ceil() as u64
}

/// The Grid key of the step whose end an effect declared at `t` fires at, or
/// `None` when `t` lies outside the window `[t_start, t_end]` (to within
/// `GRID_TOL·dt`). `k = 0` is the origin.
///
/// Inside the window, `k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋` capped at the
/// run's step count `n`: the nearest step end, a tie going to the later step.
/// When the last step is shortened (`t_end` off the grid), an effect inside it
/// fires at the nearer of its two actual ends, `t_start + (n − 1)·dt` and
/// `t_end`, a tie going to `t_end` — not at the nominal grid point the step
/// never reaches.
pub fn fire_step(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<u64> {
    grid_key(t, t_start, dt, t_end).map(|(k, _)| k)
}

/// The instant a Grid run fires an effect declared at `t`, or `None` when it
/// does not fire: `t_start + k·dt` for `k = fire_step(..)`, or `t_end` for the
/// last step.
pub fn grid_instant(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<f64> {
    let n = n_substeps(t_start, t_end, dt);
    fire_step(t, t_start, dt, t_end).map(|k| step_end(k, n, t_start, dt, t_end))
}

/// The end of step `k` of an `n`-step Grid run.
fn step_end(k: u64, n: u64, t_start: f64, dt: f64, t_end: f64) -> f64 {
    if k == n && n > 0 {
        t_end
    } else {
        t_start + k as f64 * dt
    }
}

/// [`fire_step`] plus whether the shortened-last-step rule, rather than the
/// nominal key, decided it.
fn grid_key(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<(u64, bool)> {
    let u = (t - t_start) / dt;
    let span = (t_end - t_start) / dt;
    if !(-GRID_TOL..=span + GRID_TOL).contains(&u) {
        return None;
    }
    let n = n_substeps(t_start, t_end, dt);
    let nominal = ((u + 0.5 + GRID_TOL).floor().max(0.0) as u64).min(n);
    if n == 0 {
        return Some((0, false));
    }
    let last_start = (n - 1) as f64;
    let last_len = span - last_start;
    if last_len < 1.0 - GRID_TOL && u > last_start {
        // Inside a shortened last step: the nearer of its two ends.
        let to_end = span - u;
        let from_start = u - last_start;
        let k = if to_end <= from_start + GRID_TOL { n } else { n - 1 };
        return Some((k, k != nominal));
    }
    Some((nominal, false))
}

impl EffectPlan {
    /// Build the plan for one run of `model` at `params` over `[t_start, t_end]`
    /// on `clock`. Errors on a non-finite fire time or (Grid) an invalid `dt`.
    pub fn build(
        model: &CompiledModel,
        params: &[f64],
        t_start: f64,
        t_end: f64,
        clock: FireClock,
        origin: Origin,
    ) -> Result<Self, SimError> {
        let fire_times = model.resolve_fire_times(params);
        for times in &fire_times {
            crate::time::validate_fire_times(times)?;
        }
        let mut plan = match clock {
            FireClock::Grid { dt } => {
                crate::time::validate_dt(dt)?;
                Self::build_grid(model, &fire_times, t_start, t_end, dt)
            }
            FireClock::Exact => Self::build_exact(&fire_times, t_start, t_end),
        };
        if origin == Origin::Resumed {
            plan.at_origin.clear();
        }
        for note in &plan.notes {
            crate::run_notes::record(crate::run_notes::RunNote::EffectSnapped(note.clone()));
        }
        Ok(plan)
    }

    fn build_grid(
        model: &CompiledModel,
        fire_times: &[Vec<f64>],
        t_start: f64,
        t_end: f64,
        dt: f64,
    ) -> Self {
        let n = n_substeps(t_start, t_end, dt);
        let mut by_step: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut notes = Vec::new();
        for (iv_idx, times) in fire_times.iter().enumerate() {
            // One intervention fires once per step, however many of its
            // declared times share it (gh#198): a discrete-time model has one
            // boundary per step.
            let mut keys: Vec<u64> = Vec::new();
            let name = &model.model.interventions[iv_idx].name;
            // The earliest declared time the nearest-step rule moved and where
            // it fires, and how many it moved, for the effect's one off-grid note.
            let mut first_moved: Option<(f64, f64)> = None;
            let (mut n_moved, mut n_fires) = (0usize, 0usize);
            for &t in times {
                let Some((k, refined)) = grid_key(t, t_start, dt, t_end) else { continue };
                n_fires += 1;
                let fires_at = step_end(k, n, t_start, dt, t_end);
                if refined {
                    let nominal_k = ((t - t_start) / dt + 0.5 + GRID_TOL).floor().max(0.0) as u64;
                    notes.push(SnapNote {
                        effect: name.clone(),
                        declared: t,
                        fires_at,
                        cause: SnapCause::ShortenedLastStep { nominal: t_start + nominal_k as f64 * dt },
                    });
                } else if ((t - fires_at) / dt).abs() > GRID_TOL {
                    n_moved += 1;
                    if first_moved.is_none_or(|(d, _)| t < d) {
                        first_moved = Some((t, fires_at));
                    }
                }
                if !keys.contains(&k) {
                    keys.push(k);
                }
            }
            if let Some((declared, fires_at)) = first_moved {
                notes.push(SnapNote {
                    effect: name.clone(),
                    declared,
                    fires_at,
                    cause: SnapCause::OffGrid { t_start, dt, n_moved, n_fires },
                });
            }
            for k in keys {
                by_step.entry(k).or_default().push(iv_idx);
            }
        }
        let at_origin = by_step.remove(&0).unwrap_or_default();
        let mut plan = EffectPlan { at_origin, notes, ..Default::default() };
        for (k, batch) in by_step {
            plan.times.push(step_end(k, n, t_start, dt, t_end));
            plan.steps.push(k);
            plan.batches.push(batch);
        }
        plan
    }

    fn build_exact(fire_times: &[Vec<f64>], t_start: f64, t_end: f64) -> Self {
        let mut at_origin = Vec::new();
        let mut pairs: Vec<(f64, usize)> = Vec::new();
        for (iv_idx, times) in fire_times.iter().enumerate() {
            let mut own: Vec<f64> = times
                .iter()
                .copied()
                .filter(|&t| t >= t_start - EFFECT_EPS && t <= t_end + EFFECT_EPS)
                .collect();
            own.sort_by(|a, b| a.total_cmp(b));
            // One fire per instant: times within EFFECT_EPS of the first of a
            // run are that one fire.
            let mut anchor: Option<f64> = None;
            for t in own {
                if anchor.is_some_and(|a| t - a <= EFFECT_EPS) {
                    continue;
                }
                anchor = Some(t);
                if (t - t_start).abs() <= EFFECT_EPS {
                    at_origin.push(iv_idx);
                } else {
                    pairs.push((t, iv_idx));
                }
            }
        }
        // Stable sort: a tie keeps declaration order.
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut plan = EffectPlan { at_origin, ..Default::default() };
        for (t, iv_idx) in pairs {
            match (plan.times.last(), plan.batches.last_mut()) {
                (Some(&anchor), Some(batch)) if t - anchor <= EFFECT_EPS => {
                    batch.push(iv_idx);
                }
                _ => {
                    plan.times.push(t);
                    plan.batches.push(vec![iv_idx]);
                }
            }
        }
        // Consumers require declaration order, but collection only needs
        // time order. Canonicalize each completed batch once, not per append.
        for batch in &mut plan.batches {
            batch.sort_unstable();
            batch.dedup();
        }
        plan
    }

    /// The effects applied to the starting state, declaration order.
    pub fn at_origin(&self) -> &[usize] {
        &self.at_origin
    }

    /// True when nothing fires: no origin batch and no later instant.
    pub fn is_empty(&self) -> bool {
        self.at_origin.is_empty() && self.times.is_empty()
    }

    /// Grid consumers: the batch due at the end of substep `s` (step
    /// `k = s + 1`); empty when nothing fires there or the plan is Exact.
    pub fn batch_ending_substep(&self, s: u64) -> &[usize] {
        match self.steps.binary_search(&(s + 1)) {
            Ok(i) => &self.batches[i],
            Err(_) => &[],
        }
    }

    /// The firing instants after `t_start`, ascending — the boundary list a
    /// [`crate::schedule::Schedule`] lands on, index-aligned with
    /// [`EffectPlan::batch`].
    pub fn effect_times(&self) -> EffectTimes {
        EffectTimes::from_plan_times(self.times.clone())
    }

    /// The firing instants after `t_start`, ascending.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The effects firing at `self.times()[effect_idx]`, declaration order.
    pub fn batch(&self, effect_idx: usize) -> &[usize] {
        &self.batches[effect_idx]
    }

    /// Effects fired away from their declared time, in declaration order.
    pub fn notes(&self) -> &[SnapNote] {
        &self.notes
    }
}

/// Refuse a fit whose data run past the model's simulation end. A fit scores
/// the model the backend simulates over `[t_start, simulation.t_end]` (D2);
/// an observation after `t_end` would be scored against a state `simulate`
/// never produces, with the effects due there unplanned. One check for every
/// inference path: the particle filter, PMMH and IF2 through
/// [`crate::intervention::ExactInferenceTimeline::build`], PGAS at `run_pgas`.
/// `t_end` is the resolved end (`to = last_obs` resolves to the data's end).
pub fn refuse_data_past_end(model: &CompiledModel, last_obs: f64) -> Result<(), SimError> {
    let t_end = model.model.simulation.t_end;
    if last_obs <= t_end + crate::schedule::output_tol(t_end) {
        return Ok(());
    }
    let unit = &model.model.time_unit;
    Err(SimError::Validation(format!(
        "the data run to t={last_obs} {unit} but the model's simulation ends at t={t_end} \
         {unit} (`simulate {{ to }}`); a fit cannot score data past the end of the \
         simulation. Extend `to` to cover the data: `to = {last_obs} '{unit}` does, and so \
         does `to = last_obs` (or `to = last_obs + 8 'weeks`) where the model has no \
         recurring schedule that needs a literal end."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_batch_collection_matches_incremental_scan() {
        let candidates = [-1.0, -EFFECT_EPS, 0.0, EFFECT_EPS, 0.5,
            0.5 + 0.75 * EFFECT_EPS, 0.5 + 1.5 * EFFECT_EPS,
            1.0, 1.0 + EFFECT_EPS, 2.0];
        let mut seed = 42_u64;
        for _ in 0..2000 {
            let mut times = vec![Vec::new(); 8];
            for own in &mut times {
                for _ in 0..12 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    own.push(candidates[(seed >> 32) as usize % candidates.len()]);
                }
            }
            let actual = EffectPlan::build_exact(&times, 0.0, 1.0);
            let expected = build_exact_scan(&times, 0.0, 1.0);
            assert_eq!(actual.at_origin, expected.at_origin);
            assert_eq!(actual.times, expected.times);
            assert_eq!(actual.batches, expected.batches);
        }
    }

    // Previous implementation retained only as an equivalence oracle.
    fn build_exact_scan(fire_times: &[Vec<f64>], t_start: f64, t_end: f64) -> EffectPlan {
        let mut at_origin = Vec::new();
        let mut pairs: Vec<(f64, usize)> = Vec::new();
        for (iv_idx, times) in fire_times.iter().enumerate() {
            let mut own: Vec<f64> = times
                .iter()
                .copied()
                .filter(|&t| t >= t_start - EFFECT_EPS && t <= t_end + EFFECT_EPS)
                .collect();
            own.sort_by(|a, b| a.total_cmp(b));
            // One fire per instant: times within EFFECT_EPS of the first of a
            // run are that one fire.
            let mut anchor: Option<f64> = None;
            for t in own {
                if anchor.is_some_and(|a| t - a <= EFFECT_EPS) {
                    continue;
                }
                anchor = Some(t);
                if (t - t_start).abs() <= EFFECT_EPS {
                    at_origin.push(iv_idx);
                } else {
                    pairs.push((t, iv_idx));
                }
            }
        }
        // Stable sort: a tie keeps declaration order.
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut plan = EffectPlan { at_origin, ..Default::default() };
        for (t, iv_idx) in pairs {
            match (plan.times.last(), plan.batches.last_mut()) {
                (Some(&anchor), Some(batch)) if t - anchor <= EFFECT_EPS => {
                    if !batch.contains(&iv_idx) {
                        batch.push(iv_idx);
                        batch.sort_unstable();
                    }
                }
                _ => {
                    plan.times.push(t);
                    plan.batches.push(vec![iv_idx]);
                }
            }
        }
        plan
    }



    use ir::{
        expr::Expr,
        intervention::{
            Action, AddAction, FireSource, Intervention, InterventionKind, InterventionSchedule,
            RecurringSchedule,
        },
        model::{
            Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
            SimulationConfig,
        },
        parameter::{ParamValue, Parameter},
        transition::{DrawMethod, StoichiometryEntry, Transition},
        Model,
    };

    /// A one-compartment model carrying one `add(A, 1)` effect per entry of
    /// `effects` (`(name, schedule, is_event)`), declared in that order.
    fn model(effects: &[(&str, InterventionSchedule, bool)]) -> CompiledModel {
        let m = Model {
            constants: vec![],
            ic_grad: Default::default(),
            name: "effect_plan_test".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "A".into(), kind: CompartmentKind::Integer },
                Compartment { name: "B".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![Transition {
                rate_state_grad: Default::default(),
                name: "drain".into(),
                stoichiometry: vec![StoichiometryEntry("A".into(), -1), StoichiometryEntry("B".into(), 1)],
                rate: Expr::const_(0.0),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: effects
                .iter()
                .map(|(name, sched, event)| Intervention {
                    name: (*name).into(),
                    base_name: None,
                    fire: FireSource::Scheduled(sched.clone()),
                    actions: vec![Action::Add(AddAction { compartment: "A".into(), count: Expr::const_(1.0), count_grad: Default::default() })],
                    kind: if *event { InterventionKind::Event } else { InterventionKind::Scenario },
                })
                .collect(),
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter {
                name: "p".into(),
                value: ParamValue::Fixed { value: 1.0 },
                param_kind: None,
                param_dim: None,
            }],
            initial_conditions: InitialConditions::constants([("A".into(), 0.0), ("B".into(), 0.0)]),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0]),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 10.0,
                dt: Some(1.0),
                rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None, t_start_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
            quantities: vec![],
            contrasts: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    fn at(times: &[f64]) -> InterventionSchedule {
        InterventionSchedule::AtTimes(times.to_vec())
    }

    fn grid(m: &CompiledModel, t_start: f64, t_end: f64, dt: f64) -> EffectPlan {
        EffectPlan::build(m, &m.default_params, t_start, t_end, FireClock::Grid { dt }, Origin::Init)
            .unwrap()
    }

    fn exact(m: &CompiledModel, t_start: f64, t_end: f64) -> EffectPlan {
        EffectPlan::build(m, &m.default_params, t_start, t_end, FireClock::Exact, Origin::Init).unwrap()
    }

    /// The §2 offset-origin case: `from = 0.5`, `dt = 1`, events at 1, 2, 2.4
    /// and 3.5 fire at 1.5, 2.5, 2.5 and 3.5 on the run's own grid. The old
    /// absolute key `round(t/dt)` dropped the event at 1 and fired 2 and 2.4 at
    /// 1.5.
    #[test]
    fn grid_key_is_relative_to_t_start() {
        let m = model(&[
            ("e1", at(&[1.0]), true),
            ("e2", at(&[2.0]), true),
            ("e24", at(&[2.4]), true),
            ("e35", at(&[3.5]), true),
        ]);
        let p = grid(&m, 0.5, 5.5, 1.0);
        assert!(p.at_origin().is_empty());
        assert_eq!(p.times(), &[1.5, 2.5, 3.5]);
        assert_eq!(p.batch(0), &[0]);
        assert_eq!(p.batch(1), &[1, 2], "2 and 2.4 share the step ending at 2.5");
        assert_eq!(p.batch(2), &[3]);
        assert_eq!(p.batch_ending_substep(0), &[0], "step 1 ends at 1.5");
        assert_eq!(p.batch_ending_substep(1), &[1, 2]);
        assert!(p.batch_ending_substep(3).is_empty());
    }

    /// §5: a tie at a negative time goes to the later step (−0.5 with
    /// `from = −2` fires at 0, not at −1 as `f64::round` gave), and a time
    /// within `GRID_TOL·dt` below a tie goes to the later step too.
    #[test]
    fn grid_ties_go_to_the_later_step() {
        assert_eq!(grid_instant(-0.5, -2.0, 1.0, 5.0), Some(0.0));
        assert_eq!(grid_instant(1.5, 0.0, 1.0, 5.0), Some(2.0));
        assert_eq!(grid_instant(1.5 - 1e-10, 0.0, 1.0, 5.0), Some(2.0));
        assert_eq!(grid_instant(1.49, 0.0, 1.0, 5.0), Some(1.0));
    }

    /// §6.5, `t_end` off the grid (`t_end = 2.5`, `dt = 1`): an effect at 2.45
    /// fires at 2.5 and is noted; one at 2.25 (equidistant) fires at 2.5 and is
    /// noted; one at 2.2 fires at 2.0, the nearest step end, and is noted as
    /// off the grid.
    #[test]
    fn shortened_last_step_snaps_to_the_nearer_end() {
        let m = model(&[
            ("late", at(&[2.45]), false),
            ("tie", at(&[2.25]), false),
            ("early", at(&[2.2]), false),
        ]);
        let p = grid(&m, 0.0, 2.5, 1.0);
        assert_eq!(p.times(), &[2.0, 2.5]);
        assert_eq!(p.batch(0), &[2], "2.2 fires at 2.0");
        assert_eq!(p.batch(1), &[0, 1], "2.45 and 2.25 fire at t_end = 2.5");
        let last = SnapCause::ShortenedLastStep { nominal: 2.0 };
        assert_eq!(
            p.notes(),
            &[
                SnapNote { effect: "late".into(), declared: 2.45, fires_at: 2.5, cause: last.clone() },
                SnapNote { effect: "tie".into(), declared: 2.25, fires_at: 2.5, cause: last },
                SnapNote {
                    effect: "early".into(),
                    declared: 2.2,
                    fires_at: 2.0,
                    cause: SnapCause::OffGrid { t_start: 0.0, dt: 1.0, n_moved: 1, n_fires: 1 },
                },
            ],
            "the rule, not the nominal key, placed 2.45 and 2.25; 2.2 is off the grid",
        );
        // A tie within GRID_TOL·dt still goes to t_end (2.25 − 1e-10 is 2e-10
        // nearer 2 than 2.5 in exact arithmetic).
        assert_eq!(grid_instant(2.25 - 1e-10, 0.0, 1.0, 2.5), Some(2.5));
        assert_eq!(grid_instant(2.25 - 1e-8, 0.0, 1.0, 2.5), Some(2.0), "beyond the tolerance: 2");
        let text = p.notes()[0].to_string();
        assert!(text.contains("late") && text.contains("2.45") && text.contains("2.5"), "{text}");
    }

    /// The window: `t_end` on the grid. An effect at `t_end + dt/2` does not
    /// fire; an effect at `t_end` fires on the last step; a recurring
    /// enumeration's last fire at `t_end + 1e-12` is the fire at `t_end`; an
    /// effect just before `t_start` (beyond the tolerance) does not fire.
    #[test]
    fn grid_window_is_t_start_to_t_end() {
        assert_eq!(fire_step(4.5, 0.0, 1.0, 4.0), None, "t_end + dt/2 does not fire");
        assert_eq!(fire_step(4.3, 0.0, 1.0, 4.0), None, "inside (t_end, t_end + dt/2] does not fire");
        assert_eq!(grid_instant(4.0, 0.0, 1.0, 4.0), Some(4.0));
        assert_eq!(grid_instant(4.0 + 1e-12, 0.0, 1.0, 4.0), Some(4.0));
        assert_eq!(fire_step(1.7, 2.0, 1.0, 6.0), None, "before t_start does not fire");
        assert_eq!(fire_step(2.0 - 1e-12, 2.0, 1.0, 6.0), Some(0), "t_start within tolerance");
        // A recurring block over [0, 6] whose float enumeration overshoots.
        let m = model(&[(
            "every2",
            InterventionSchedule::Recurring(RecurringSchedule { start: 0.0, period: 2.0, end: 6.0, at_day: None }),
            true,
        )]);
        let p = grid(&m, 0.0, 6.0, 1.0);
        assert_eq!(p.at_origin(), &[0], "the default `from = t_start` fire is an origin fire");
        assert_eq!(p.times(), &[2.0, 4.0, 6.0]);
    }

    /// D1 membership: `k = 0` under Grid (declared within half a step after
    /// `t_start`), `|t − t_start| ≤ EFFECT_EPS` under Exact; `Origin::Resumed`
    /// empties the origin batch and nothing else.
    #[test]
    fn origin_membership_per_clock() {
        let m = model(&[("a0", at(&[0.0]), false), ("a04", at(&[0.4]), true), ("a1", at(&[1.0]), false)]);
        let g = grid(&m, 0.0, 5.0, 1.0);
        assert_eq!(g.at_origin(), &[0, 1], "0 and 0.4 both key to step 0");
        assert_eq!(g.times(), &[1.0]);
        let e = exact(&m, 0.0, 5.0);
        assert_eq!(e.at_origin(), &[0], "only the declared t_start is the origin under Exact");
        assert_eq!(e.times(), &[0.4, 1.0]);
        let r = EffectPlan::build(&m, &m.default_params, 0.0, 5.0, FireClock::Grid { dt: 1.0 }, Origin::Resumed)
            .unwrap();
        assert!(r.at_origin().is_empty());
        assert_eq!(r.times(), &[1.0]);
        assert!(!r.is_empty());
    }

    /// Batching follows the clock (gh#198 under Grid only): `at [2.3, 2.4]` at
    /// `dt = 1` fires once under Grid and twice under Exact; under Exact
    /// `at [2.0, 2.0 + 1e-12]` on one intervention is one fire, and two
    /// interventions at one instant are one batch in declaration order.
    #[test]
    fn batching_follows_the_clock() {
        let m = model(&[("pair", at(&[2.3, 2.4]), false)]);
        assert_eq!(grid(&m, 0.0, 5.0, 1.0).times(), &[2.0]);
        assert_eq!(grid(&m, 0.0, 5.0, 1.0).batch(0), &[0], "one fire for the step");
        assert_eq!(exact(&m, 0.0, 5.0).times(), &[2.3, 2.4]);

        let m = model(&[("dup", at(&[2.0, 2.0 + 1e-12]), false)]);
        let e = exact(&m, 0.0, 5.0);
        assert_eq!(e.times(), &[2.0]);
        assert_eq!(e.batch(0), &[0], "one fire, not two");

        // Declared out of time order, within EFFECT_EPS: one batch, declaration order.
        let m = model(&[("first", at(&[3.0 + 1e-11]), false), ("second", at(&[3.0]), true)]);
        let e = exact(&m, 0.0, 5.0);
        assert_eq!(e.times(), &[3.0]);
        assert_eq!(e.batch(0), &[0, 1]);
    }

    /// Exact window: an effect before `t_start` or after `t_end` does not fire.
    #[test]
    fn exact_window_is_t_start_to_t_end() {
        let m = model(&[("before", at(&[1.7]), true), ("after", at(&[4.3]), true), ("in", at(&[3.0]), true)]);
        let e = exact(&m, 2.0, 4.0);
        assert!(e.at_origin().is_empty());
        assert_eq!(e.times(), &[3.0]);
        assert_eq!(e.batch(0), &[2]);
    }

    /// The key scales with the run's own `dt` (the gh#53 fingerprint): day 258
    /// is step 258 at dt = 1, 516 at dt = 0.5 and 1032 at dt = 0.25, and each of
    /// a period's fires keeps its own step.
    #[test]
    fn the_grid_key_scales_with_dt() {
        for (dt, k) in [(1.0, 258), (0.5, 516), (0.25, 1032)] {
            assert_eq!(fire_step(258.0, 0.0, dt, 2000.0), Some(k));
            let steps: Vec<u64> =
                [258.0, 623.25, 988.5].iter().map(|&t| fire_step(t, 0.0, dt, 2000.0).unwrap()).collect();
            assert!(steps.windows(2).all(|w| w[0] < w[1]), "dt={dt}: {steps:?}");
        }
    }

    #[test]
    fn n_substeps_counts_a_shortened_last_step() {
        assert_eq!(n_substeps(0.0, 10.0, 1.0), 10);
        assert_eq!(n_substeps(0.0, 2.5, 1.0), 3);
        assert_eq!(n_substeps(0.0, 0.3, 0.1), 3, "float-rounded whole number of steps");
        assert_eq!(n_substeps(1.0, 1.0, 1.0), 0);
    }

    /// An effect declared off the grid fires at the nearest step end, and the
    /// plan carries one note saying so; an effect on the grid carries none.
    #[test]
    fn an_off_grid_effect_is_noted_and_an_on_grid_one_is_not() {
        let m = model(&[("pulse", at(&[2.5]), false), ("cull", at(&[4.0]), false)]);
        let p = grid(&m, 0.0, 10.0, 1.0);
        assert_eq!(p.times(), &[3.0, 4.0]);
        assert_eq!(p.notes().len(), 1, "{:?}", p.notes());
        let note = &p.notes()[0];
        assert_eq!((note.effect.as_str(), note.declared, note.fires_at), ("pulse", 2.5, 3.0));
        let text = note.to_string();
        assert!(
            text.contains("`pulse`") && text.contains("2.5") && text.contains("t = 3")
                && text.contains("dt = 1") && text.contains("grid"),
            "{text}"
        );
        let run_note = crate::run_notes::RunNote::EffectSnapped(note.clone());
        assert_eq!(run_note.code(), "effect-snapped-to-grid");

        let on = model(&[("cull", at(&[4.0]), false), ("ev", at(&[0.1 * 3.0]), true)]);
        assert!(grid(&on, 0.0, 10.0, 0.1).notes().is_empty(), "float-rounded on-grid times are on the grid");
        assert!(grid(&on, 0.0, 10.0, 1.0).notes().len() == 1, "0.3 is off the dt = 1 grid");
    }

    /// A recurring schedule is one note naming its first snapped time and how
    /// many of its fires were moved, not one note per fire.
    #[test]
    fn a_recurring_off_grid_schedule_is_one_note() {
        let m = model(&[(
            "every25",
            InterventionSchedule::Recurring(RecurringSchedule { start: 0.0, period: 2.5, end: 10.0, at_day: None }),
            true,
        )]);
        let p = grid(&m, 0.0, 10.0, 1.0);
        assert_eq!(p.notes().len(), 1, "{:?}", p.notes());
        let note = &p.notes()[0];
        assert_eq!((note.declared, note.fires_at), (2.5, 3.0));
        let text = note.to_string();
        assert!(text.contains("2 of its 5 fire times"), "{text}");
    }

    /// A parameter-dependent fire time is noted, and the notes of plans built at
    /// different parameter values, as a filter or a sampler builds them, are one
    /// note to the run's record, so it is said once.
    #[test]
    fn a_parametric_off_grid_time_is_said_once() {
        let mut m = model(&[]);
        let mut ir_model = (*m.model).clone();
        ir_model.interventions = vec![Intervention {
            name: "a_parametric_off_grid_time_is_said_once".into(),
            base_name: None,
            fire: FireSource::Scheduled(InterventionSchedule::AtTimesExpr(vec![Expr::param("p")])),
            actions: vec![Action::Add(AddAction {
                compartment: "A".into(),
                count: Expr::const_(1.0),
                count_grad: Default::default(),
            })],
            kind: InterventionKind::Event,
        }];
        m = CompiledModel::new(ir_model).unwrap();
        let mut notes = Vec::new();
        for p in [2.5, 3.5, 4.25, 6.0] {
            let plan = EffectPlan::build(&m, &[p], 0.0, 10.0, FireClock::Grid { dt: 1.0 }, Origin::Init).unwrap();
            assert_eq!(plan.notes().len(), usize::from(p != 6.0), "p = {p}");
            notes.extend(plan.notes().iter().cloned().map(crate::run_notes::RunNote::EffectSnapped));
        }
        assert_eq!(notes.len(), 3);
        assert!(
            notes.windows(2).all(|w| crate::run_notes::same_note(&w[0], &w[1])),
            "one effect at one dt is one note whatever its declared time: {notes:?}"
        );
    }

    #[test]
    fn an_empty_plan_is_empty() {
        let m = model(&[]);
        let p = grid(&m, 0.0, 5.0, 1.0);
        assert!(p.is_empty());
        assert!(p.batch_ending_substep(0).is_empty());
    }
}
