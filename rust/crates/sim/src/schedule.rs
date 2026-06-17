//! The merged timeline spine — one `Schedule` answering "where does the
//! integrator stop next, and what is due there" for every stepping idiom.
//!
//! Today that question is answered three incompatible ways (verified against the
//! backends, 2026-06-05):
//!
//!   - **chain_binomial** (`chain_binomial.rs:170-237`) — steps a FULL `cfg.dt`
//!     every substep (`dt = cfg.dt.min(t_end - t)`, never clipped to an
//!     output/effect time); outputs are emitted post-step at grid times; effects
//!     fire *inside* `step_one` keyed on `resolve_fire_steps(cfg.dt)` (a step
//!     index), NOT on the loop's boundary detection. This is the **snap** policy.
//!   - **ode** (`ode.rs:205-239`) — computes
//!     `next_boundary = min(t_end, out_t, iv_t)` and clips
//!     `dt = cfg.dt.min(next_boundary - t)`, landing EXACTLY on each output/effect
//!     time, where it fires via `apply_interventions_at(t, …, 1e-10)`. This is
//!     the **exact** policy.
//!   - **gillespie** (`gillespie.rs:144-246`) — PROPOSES an exponential time and
//!     clips it back to `min(t_end, out_t, iv_t)`. The **clip** query.
//!
//! This module makes the time→step mapping ONE thing with an explicit policy and
//! an explicit grid, rather than three call-site conventions. The grid is a FIELD
//! (`cfg.dt` for chain/ode, `model.simulation.dt.unwrap_or(1.0)` —
//! gillespie's `iv_resolution_dt`, `gillespie.rs:120` — for gillespie), the
//! single source of truth, so a model never snaps interventions on one grid and
//! observations on another.
//!
//! ## Scope of the byte-identical extraction (Stage 1)
//!
//! Stage 1 unifies the **boundary cursor**: the next-stop time, the substep
//! cadence, and the output-emission walk (the `while output_times[idx] <= t + 1e-12`
//! block duplicated in all four backends). It does NOT move where interventions
//! *fire*: chain_binomial keeps its `fire_steps`-in-`step_one` mechanism, the
//! exact backends keep `apply_interventions_at` at the clipped boundary
//! ("interventions as today"). Unifying the firing mechanism is a behaviour change
//! (it would move chain_binomial's snap-vs-exact divergence — see
//! `tests/fixtures/corner_cases/off_grid_intervention.camdl`) and belongs to the
//! `--obs-alignment` knob (Stage 2), not here. Accordingly, under `Snap` the
//! schedule reports only `Output` boundaries; effects are the backend's business.
//!
//! ## The CRN invariant (the one regression that breaks silently)
//!
//! `Schedule` is immutable and `Sync`; the per-particle [`Cursor`] is `Copy`.
//! [`Schedule::next_boundary`] is a PURE function of `(Schedule, cursor, t)` with
//! no interior mutability, so N particles in a parallel swarm walk an IDENTICALLY
//! ordered boundary sequence — paired-seed / CRN coupling depends on this, and a
//! shared-mutable cursor would corrupt it without failing any all-on-grid golden.
//! Pinned by [`tests::n_cursors_identical_sequence`].

use smallvec::SmallVec;

/// Why a [`TimelineStop`] matters — one stop can carry several reasons (an
/// output time that is ALSO an observation and a scheduled-effect boundary).
/// The driver handles a stop's reasons in one declared canonical order rather
/// than each backend re-deriving "what is due" at the landing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// A trajectory snapshot is due here ([`Schedule::output_times`]).
    Output,
    /// A scheduled effect (intervention / always-active event) fires here
    /// ([`Schedule::effect_times`]).
    ScheduledEffect,
    /// An inference filter scores the likelihood here ([`Schedule::obs_times`]).
    Observation,
    /// The run window terminates here (`t_end`).
    End,
}

/// The next boundary the integrator must land on, and every reason it matters.
/// Returned by [`Schedule::next_stop`]: `t = min(t_end, next_output, next_effect,
/// next_obs)` and `reasons` lists each kind whose own next time equals that `t`
/// (within the schedule's tolerances). The driver consumes the reasons in one
/// canonical order; effect application then reads a known due batch (see
/// [`EffectBatch`]) instead of re-deriving due-ness.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineStop {
    pub t: f64,
    pub reasons: SmallVec<[StopReason; 4]>,
}

/// The interventions/events due to fire at one substep, pre-split by lifecycle
/// stage so application carries a known list instead of re-deriving due-ness.
/// Indices are into `model.model.interventions`, in DECLARATION ORDER (the
/// firing order the discrete backends already used). Filled by
/// [`crate::effects::due_effects`]; consumed by the PROPOSE / INTERVENE stages.
///
/// - `event_idx` — `always_active` interventions (events): fire at PROPOSE,
///   resolved against the start-of-step snapshot and fused with the kernel draw.
/// - `intervention_idx` — scheduled (`!always_active`) interventions: fire at
///   INTERVENE, applied sequentially on the post-advance state.
///
/// Caller-provided + reused across substeps (the inference hot path runs one per
/// particle per substep) — [`EffectBatch::clear`] resets it without freeing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectBatch {
    pub event_idx: SmallVec<[usize; 4]>,
    pub intervention_idx: SmallVec<[usize; 4]>,
}

impl EffectBatch {
    /// Reset both index lists without releasing capacity, for reuse across
    /// substeps.
    pub fn clear(&mut self) {
        self.event_idx.clear();
        self.intervention_idx.clear();
    }

    /// Whether no intervention or event is due (the common off-step case).
    pub fn is_empty(&self) -> bool {
        self.event_idx.is_empty() && self.intervention_idx.is_empty()
    }
}

/// How a fixed-step driver maps off-grid output/effect times onto its `dt` grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPolicy {
    /// chain_binomial: step a full `dt` every substep, never clipping to land on
    /// an output/effect time. Outputs are emitted at grid points; effects are
    /// snapped to a step elsewhere (`resolve_fire_steps`).
    Snap,
    /// ode: clip the step so the integrator lands exactly on the next
    /// output/effect boundary. Also the policy the inference filters run
    /// chain's `step_one` under.
    Exact,
}

/// Per-particle walk position over a shared [`Schedule`]. `Copy` — each particle
/// holds its own; the schedule is never mutated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub output_idx: usize,
    pub effect_idx: usize,
    pub obs_idx: usize,
}

/// The three time tolerances, defined ONCE here (the spine owns time) and reused
/// everywhere — distinguished by MEANING, not just value. Replacing the bare
/// literals the backends used to spell by hand (gh#233). NOTE three same-valued
/// constants that are deliberately NOT these (different axes, do not merge):
/// `chain_binomial::RATE_EPSILON` (a *rate* floor), the `clamp(1e-15, 1-1e-15)`
/// *probability* guard, and `pgas::GRID_STEP_EPS` (a 1e-12 negligible-step floor
/// whose discrepancy with `MIN_STEP_EPS` is a separate, deliberate decision).
///
/// "An output time has been reached" / due: `next_output <= t + OUTPUT_EPS`.
pub const OUTPUT_EPS: f64 = 1e-12;
/// "An effect / observation time has been reached" / due: `next_effect <= t + EFFECT_EPS`.
pub const EFFECT_EPS: f64 = 1e-10;
/// Step FLOOR: a remaining gap this small means the integrator has ARRIVED at the
/// boundary — dispatch, don't step. Distinct in meaning from the two *due* tests
/// above even though numerically smaller. (ode `h_max` arrival, chain loop-break,
/// gillespie RK4-skip guard.)
pub const MIN_STEP_EPS: f64 = 1e-15;

/// Immutable, `Sync`, shared by every particle. Construction sorts the boundary
/// times once; [`Cursor`] walks them. See the module header for the invariant.
#[derive(Clone, Debug)]
pub struct Schedule {
    dt: f64,
    t_end: f64,
    /// The snap grid (`cfg.dt` for chain/tau/ode; `iv_resolution_dt` for
    /// gillespie). Reserved for the Stage-2 `exact` sub-grid tiling; the `Snap`
    /// path emits effects off this grid via the backend's `resolve_fire_steps`.
    grid: f64,
    policy: StepPolicy,
    /// Sorted, ascending. Output snapshot times within `[t_start, t_end]`.
    output_times: Vec<f64>,
    /// Sorted, ascending. Scheduled effect (intervention / always-active event)
    /// boundary times.
    effect_times: Vec<f64>,
    /// Sorted, ascending. Observation times — the boundaries the INFERENCE
    /// drivers step exactly to (where they score the likelihood). A first-class
    /// boundary kind alongside output/effect (the proposal's `Boundary::Observation`),
    /// realized as a parallel list. Empty for the forward backends (`next_obs = ∞`),
    /// so adding it leaves every forward Schedule byte-identical. Populate via
    /// [`Schedule::with_obs`].
    obs_times: Vec<f64>,
}

impl Schedule {
    /// Build from the same inputs the backends compute today: the integrator
    /// `dt`, the run window `[t_start, t_end]`, the snap `grid`, the policy, and
    /// the sorted output / effect time vectors (`get_output_times`,
    /// `all_intervention_times`). Times are assumed already sorted ascending (the
    /// producers guarantee it); debug-asserted. Observation boundaries default to
    /// empty — add them with [`Schedule::with_obs`] for the inference drivers.
    pub fn new(
        dt: f64,
        t_end: f64,
        grid: f64,
        policy: StepPolicy,
        output_times: Vec<f64>,
        effect_times: Vec<f64>,
    ) -> Self {
        debug_assert!(output_times.windows(2).all(|w| w[0] <= w[1]), "output_times not sorted");
        debug_assert!(effect_times.windows(2).all(|w| w[0] <= w[1]), "effect_times not sorted");
        Schedule { dt, t_end, grid, policy, output_times, effect_times, obs_times: Vec::new() }
    }

    /// Attach observation boundaries (sorted ascending). The inference drivers step
    /// EXACTLY to each (where they score); folded into the `substep` boundary min.
    pub fn with_obs(mut self, obs_times: Vec<f64>) -> Self {
        debug_assert!(obs_times.windows(2).all(|w| w[0] <= w[1]), "obs_times not sorted");
        self.obs_times = obs_times;
        self
    }

    fn next_output(&self, cursor: &Cursor) -> f64 {
        self.output_times.get(cursor.output_idx).copied().unwrap_or(f64::INFINITY)
    }

    fn next_effect(&self, cursor: &Cursor) -> f64 {
        self.effect_times.get(cursor.effect_idx).copied().unwrap_or(f64::INFINITY)
    }

    fn next_obs(&self, cursor: &Cursor) -> f64 {
        self.obs_times.get(cursor.obs_idx).copied().unwrap_or(f64::INFINITY)
    }

    /// The substep size to advance from `t`. The SINGLE source of truth for the
    /// time→step mapping, computed the way the original backends did —
    /// `dt.min(boundary - t)`, NOT `(t + dt).min(boundary) - t`. The two are equal
    /// in exact arithmetic but NOT bit-identical for large fractional `t`
    /// (`(t + dt) - t != dt` once `t` is large relative to `dt`). The forward
    /// integer-count draws are insensitive to that ULP, but the chain-binomial
    /// transition density (`shape = dt/σ²`) PGAS evaluates is continuous and
    /// sensitive — so the inference loglik would move. Returning the step size
    /// directly keeps every consumer bit-exact for all `t`.
    ///
    /// PURE in `(self, cursor, t)` — does not mutate the cursor (the backend
    /// advances it via [`Cursor::pass_output`] / [`Cursor::pass_effect`] after
    /// handling what is due, using [`Schedule::output_due_at`] /
    /// [`Schedule::effect_due_at`]). Returns `None` once `t >= t_end`.
    ///
    /// - `Exact`: clip to the next boundary —
    ///   `dt.min(min(t_end, next_output, next_effect) - t)`. A zero/negative
    ///   result means `t` is already AT a boundary (the backend handles it
    ///   without stepping).
    /// - `Snap`: never clip to an output/effect — `dt.min(t_end - t)`.
    pub fn substep(&self, cursor: &Cursor, t: f64) -> Option<f64> {
        // Exactly `dt.min(next_boundary - t)`; see [`Schedule::next_boundary`]
        // for the raw landing target. The two share one boundary computation so
        // the fixed-step substep and the adaptive ODE stepper's `h_max` cannot
        // drift (pinned by `next_boundary_agrees_with_substep`).
        self.next_boundary(cursor, t).map(|boundary| self.dt.min(boundary - t))
    }

    /// The next boundary time the integrator must stop on under this policy,
    /// WITHOUT the per-step `dt` clip — the raw landing target. `Exact`: the min
    /// over `(t_end, next_output, next_effect, next_obs)`. `Snap`: `t_end`
    /// (effects fire off-grid via the backend's `resolve_fire_steps`). `None`
    /// once `t >= t_end` (the cursor has walked past the window).
    ///
    /// [`Schedule::substep`] is exactly `dt.min(next_boundary - t)`. The adaptive
    /// ODE stepper ([`crate::ode`]) instead consumes this RAW distance as its
    /// `h_max` and chooses its own internal sub-step ≤ it (clipping its
    /// controller's natural step to land exactly on the boundary), re-entered
    /// until the boundary is reached. PURE in `(self, cursor, t)`.
    pub fn next_boundary(&self, cursor: &Cursor, t: f64) -> Option<f64> {
        if t >= self.t_end {
            return None;
        }
        Some(match self.policy {
            StepPolicy::Exact => self
                .t_end
                .min(self.next_output(cursor))
                .min(self.next_effect(cursor))
                .min(self.next_obs(cursor)),
            StepPolicy::Snap => self.t_end,
        })
    }

    /// The next boundary the integrator must stop on AND every reason it
    /// matters. `t = min(t_end, next_output, next_effect, next_obs)` for the
    /// cursor; `reasons` lists each kind whose next time equals that `t` within
    /// the schedule's tolerances (so a single stop can be simultaneously
    /// `Output + Observation + ScheduledEffect`, and `End` whenever the min is
    /// `t_end`). Returns `None` once the cursor has walked past `t_end`.
    ///
    /// PURE in `(self, cursor, t)` — does not mutate the cursor; the driver
    /// advances each per-kind cursor (`pass_output` / `pass_effect` / `pass_obs`)
    /// after consuming the corresponding reason. The reason set is built against
    /// the SAME `OUTPUT_EPS` / `EFFECT_EPS` tolerances the `*_due_at` predicates
    /// use, so "is this kind a reason for the stop" agrees with "is this kind due
    /// at the landing."
    ///
    /// Added per the scheduling-spine §B. The backend boundary loops fully adopt
    /// it in Step 3; here it is a correct, unit-tested primitive that the drivers
    /// can drop in where clean.
    pub fn next_stop(&self, cursor: &Cursor, t: f64) -> Option<TimelineStop> {
        if t > self.t_end + OUTPUT_EPS {
            return None;
        }
        let next_out = self.next_output(cursor);
        let next_eff = self.next_effect(cursor);
        let next_obs = self.next_obs(cursor);
        let stop_t = self.t_end.min(next_out).min(next_eff).min(next_obs);

        let mut reasons: SmallVec<[StopReason; 4]> = SmallVec::new();
        // Canonical reason order: Output, ScheduledEffect, Observation, End.
        if next_out <= stop_t + OUTPUT_EPS {
            reasons.push(StopReason::Output);
        }
        if next_eff <= stop_t + EFFECT_EPS {
            reasons.push(StopReason::ScheduledEffect);
        }
        if next_obs <= stop_t + EFFECT_EPS {
            reasons.push(StopReason::Observation);
        }
        if stop_t >= self.t_end - OUTPUT_EPS {
            reasons.push(StopReason::End);
        }
        Some(TimelineStop { t: stop_t, reasons })
    }

    /// Gillespie's query: the process proposes `t_proposed` (drawn exponential)
    /// from the current time `t`; the schedule clips it back to the nearest
    /// boundary if one falls before it, else passes it through. The boundary is
    /// `min(t_end, next_output, next_effect-strictly-after-t)`.
    ///
    /// The `> t` filter on the effect (but NOT the output) is deliberate and
    /// matches the SSA's boundary semantics: an effect exactly at `t` has already
    /// been applied this iteration and must not re-fire, whereas an output exactly
    /// at `t` is still recorded. (Reproduces gillespie.rs's `next_iv` `> t` guard
    /// vs `next_out_t` raw — the asymmetry is observable only when a reaction
    /// lands exactly on a boundary time.) Returns the clipped time and whether it
    /// hit a boundary (vs a reaction firing at `t_proposed`).
    pub fn clip(&self, cursor: &Cursor, t: f64, t_proposed: f64) -> ClipResult {
        let eff = self.effect_time(cursor).filter(|&e| e > t).unwrap_or(f64::INFINITY);
        let boundary = self.t_end.min(self.next_output(cursor)).min(eff);
        if boundary < t_proposed {
            ClipResult { t: boundary, hit_boundary: true }
        } else {
            ClipResult { t: t_proposed, hit_boundary: false }
        }
    }

    /// Whether an output is due at `t` for `cursor` (the `<= t + OUTPUT_EPS` test).
    pub fn output_due_at(&self, cursor: &Cursor, t: f64) -> bool {
        self.next_output(cursor) <= t + OUTPUT_EPS
    }

    /// Whether an effect is due at `t` for `cursor` (the `<= t + EFFECT_EPS` test).
    pub fn effect_due_at(&self, cursor: &Cursor, t: f64) -> bool {
        self.next_effect(cursor) <= t + EFFECT_EPS
    }

    /// The current (next un-emitted) output time for `cursor`, or `None` past the
    /// end. The backend records its snapshot AT this time.
    pub fn output_time(&self, cursor: &Cursor) -> Option<f64> {
        self.output_times.get(cursor.output_idx).copied()
    }

    /// The current (next un-applied) effect time for `cursor`, or `None` past the
    /// end. The backend keeps its own firing-tolerance check against this time
    /// (e.g. the clipped-boundary `(iv - t).abs() < 1e-10` check).
    pub fn effect_time(&self, cursor: &Cursor) -> Option<f64> {
        self.effect_times.get(cursor.effect_idx).copied()
    }

    /// The current (next un-scored) observation time for `cursor`, or `None` past
    /// the end. The inference driver steps exactly to this and scores there.
    pub fn obs_time(&self, cursor: &Cursor) -> Option<f64> {
        self.obs_times.get(cursor.obs_idx).copied()
    }

    /// Whether an observation is due at `t` for `cursor` (`<= t + EFFECT_EPS`,
    /// matching the bootstrap PF's `obs_time - t < 1e-10` step-termination test).
    pub fn obs_due_at(&self, cursor: &Cursor, t: f64) -> bool {
        self.next_obs(cursor) <= t + EFFECT_EPS
    }

    pub fn t_end(&self) -> f64 {
        self.t_end
    }

    pub fn grid(&self) -> f64 {
        self.grid
    }

    /// The start time of the `s`-th substep within a window beginning at
    /// `window_start`: `window_start + s*dt`. The SINGLE source of truth for the
    /// time passed to rate / forcing evaluation. Computed by multiplication (one
    /// rounding, O(1) error) rather than accumulation (`t += dt`, O(s) drift), so
    /// the same model samples time-inhomogeneous forcing at identical times in the
    /// forward simulator and in PGAS — which already uses `t_start + s*dt`. SNAP
    /// steppers anchor `window_start = t_start` (global grid); EXACT steppers
    /// re-anchor `window_start` to each boundary/obs they clip to.
    /// See docs/dev/proposals/2026-06-05-substep-time-sdt-convention.md.
    pub fn substep_time(&self, window_start: f64, s: u64) -> f64 {
        window_start + s as f64 * self.dt
    }

    /// The inner substep walk a fixed-step inference filter performs within ONE
    /// observation window: starting from `t_start`, yield `(t_local, step_dt)` for
    /// each substep up to the boundary `cursor` points at (its current obs). The
    /// step size is [`Schedule::substep`]; `t_local` ACCUMULATES (`t += step_dt`)
    /// — the EXACT-stepper convention the bootstrap PF / IF2 / correlated PF share
    /// (the drift-free `substep_time` variant for these is task #14).
    ///
    /// This is the single shared primitive behind those three inner loops (the
    /// proposal's consolidation seam). Each driver keeps its OWN body over the
    /// iterator — death-on-recoverable-error, pre-drawn-noise injection, the IF2
    /// θ-perturbation — because those genuinely differ; only the walk is shared.
    /// `cursor` is taken by value (`Copy`): the iterator walks within the window
    /// the caller positioned it at and never mutates the caller's cursor, so the
    /// CRN invariant (N particles, identical boundary sequence) is preserved.
    pub fn substeps(&self, cursor: Cursor, t_start: f64) -> Substeps<'_> {
        Substeps { schedule: self, cursor, t: t_start }
    }

    /// The time the observation window ending at `cursor`'s current obs reaches
    /// when walked from `t_start` — i.e. where [`Schedule::substeps`] leaves the
    /// clock. The filters advance their single-threaded reference `t` to the
    /// window end after the parallel per-particle walk; this is that advance,
    /// defined ONCE in terms of the same iterator (so it cannot drift from the
    /// per-particle walk — the divergence hazard of a second hand-rolled copy).
    /// Returns `t_start` for an empty window (no substep due).
    pub fn window_end(&self, cursor: Cursor, t_start: f64) -> f64 {
        self.substeps(cursor, t_start).last().map_or(t_start, |(t0, step_dt, _)| t0 + step_dt)
    }

    /// The effect-cursor position for a window beginning at `t`: the number of
    /// effect boundaries already fired by `t` (those `<= t + EFFECT_EPS`). The
    /// inference filters position each observation window's `Cursor.effect_idx`
    /// here, so the monotone scheduled-effect cursor carries correctly across
    /// windows (effects coincident with a previous window's closing obs counted
    /// as already fired). `effect_times` is sorted, so this is a partition point.
    pub fn effect_idx_at(&self, t: f64) -> usize {
        self.effect_times.partition_point(|&e| e <= t + EFFECT_EPS)
    }

    /// Emit a snapshot at every output time due at or before `until`, advancing
    /// the output cursor. `record(ot)` builds + pushes the per-backend snapshot at
    /// output time `ot` (the only thing that differs across backends — i64 counts
    /// vs ODE's f64 state). The "drain the passed output times" walk all four
    /// forward backends hand-rolled (mid-loop and final flush) is defined ONCE
    /// here. Pass `until = t` for the mid-loop drain (≡ the old
    /// `while output_due_at(cursor, t)`), `until = f64::INFINITY` for the final
    /// flush (≡ the old `while let Some(ot) = output_time(cursor)`).
    pub fn drain_outputs(&self, cursor: &mut Cursor, until: f64, mut record: impl FnMut(f64)) {
        while let Some(ot) = self.output_time(cursor) {
            if ot > until + OUTPUT_EPS {
                break;
            }
            record(ot);
            cursor.pass_output();
        }
    }
}

/// Iterator over one observation window's substeps; see [`Schedule::substeps`].
/// Yields `(t_local, step_dt, fired_effect)`, terminating at the cursor's current
/// observation boundary (`obs_time(cursor) - EFFECT_EPS`), reproducing the
/// `while t_local < obs_time - 1e-10 { … }` loops it replaces. `fired_effect` is
/// `Some(effect_idx)` when this substep LANDS on a scheduled-effect boundary —
/// the caller reads the due batch (e.g. [`crate::intervention::TimelineEffects`]
/// `.batches[effect_idx]`) and fires it CURSOR-keyed (gh#216), instead of the
/// `round(t/dt)` key inside `step_one`.
pub struct Substeps<'a> {
    schedule: &'a Schedule,
    cursor: Cursor,
    t: f64,
}

impl Iterator for Substeps<'_> {
    type Item = (f64, f64, Option<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let obs_time = self.schedule.obs_time(&self.cursor)?;
        if self.t >= obs_time - EFFECT_EPS {
            return None;
        }
        let step_dt = self.schedule.substep(&self.cursor, self.t)?;
        let t0 = self.t;
        self.t += step_dt;
        // Did this substep land on a scheduled-effect boundary? Surface its
        // effect_idx (so the caller fires the due batch) and ADVANCE the effect
        // cursor (`pass_effect`) so the NEXT substep clips past it. Without the
        // advance, once `effect_times` is populated `substep()` would clip every
        // later step to this same boundary and the walk would stall on a run of
        // zero-length substeps (proposal §3.3).
        let fired = if self.schedule.effect_due_at(&self.cursor, self.t) {
            let idx = self.cursor.effect_idx;
            self.cursor.pass_effect();
            Some(idx)
        } else {
            None
        };
        Some((t0, step_dt, fired))
    }
}

impl Cursor {
    /// Advance past the current output (after the backend has recorded it).
    pub fn pass_output(&mut self) {
        self.output_idx += 1;
    }

    /// Advance past the current effect (after the backend has applied it).
    pub fn pass_effect(&mut self) {
        self.effect_idx += 1;
    }

    /// Advance past the current observation (after the inference driver scored it).
    pub fn pass_obs(&mut self) {
        self.obs_idx += 1;
    }
}

/// Result of [`Schedule::clip`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipResult {
    pub t: f64,
    pub hit_boundary: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(dt: f64, t_end: f64, out: Vec<f64>, eff: Vec<f64>) -> Schedule {
        Schedule::new(dt, t_end, dt, StepPolicy::Exact, out, eff)
    }
    fn snap(dt: f64, t_end: f64, out: Vec<f64>, eff: Vec<f64>) -> Schedule {
        Schedule::new(dt, t_end, dt, StepPolicy::Snap, out, eff)
    }

    /// Walk the whole schedule into a sequence of substep sizes (bit-exact),
    /// draining cursors exactly as a backend would.
    fn walk(s: &Schedule) -> Vec<u64> {
        let mut cur = Cursor::default();
        let mut t = 0.0_f64;
        let mut seq = Vec::new();
        let mut guard = 0;
        while let Some(step_dt) = s.substep(&cur, t) {
            seq.push(step_dt.to_bits());
            t += step_dt;
            // Drain what is due at the new t, as a backend does.
            while s.output_due_at(&cur, t) {
                cur.pass_output();
            }
            while s.effect_due_at(&cur, t) {
                cur.pass_effect();
            }
            guard += 1;
            assert!(guard < 100_000, "walk did not terminate");
        }
        seq
    }

    #[test]
    fn exact_clips_to_off_grid_effect() {
        // off_grid_intervention: cull at t=2.5, dt=1, t_end=5. Exact lands on 2.5.
        let s = exact(1.0, 5.0, vec![], vec![2.5]);
        let cur = Cursor::default();
        // First step from 0 → full dt (no boundary inside (0,1]).
        assert_eq!(s.substep(&cur, 0.0).unwrap(), 1.0);
        // From t=2, the 2.5 boundary clips the step to 0.5 (lands on 2.5).
        assert_eq!(s.substep(&cur, 2.0).unwrap(), 0.5, "exact must clip to the off-grid effect");
        assert!(s.effect_due_at(&cur, 2.5), "effect is due at the clipped landing");
    }

    #[test]
    fn snap_steps_full_dt_over_off_grid_effect() {
        // Same model under Snap: the step is NOT clipped to 2.5; effects never
        // surface as schedule boundaries (the backend's step_one fires them).
        let s = snap(1.0, 5.0, vec![], vec![2.5]);
        let cur = Cursor::default();
        assert_eq!(s.substep(&cur, 2.0).unwrap(), 1.0, "snap steps a full dt past the off-grid effect");
    }

    #[test]
    fn exact_and_snap_diverge_on_off_grid() {
        // The corner-case divergence, at the schedule layer: the two policies
        // produce DIFFERENT substep sequences for an off-grid effect.
        let e = walk(&exact(1.0, 5.0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2.5]));
        let n = walk(&snap(1.0, 5.0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2.5]));
        assert_ne!(e, n, "off-grid snap vs exact must differ (the pinned divergence)");
    }

    #[test]
    fn coincident_output_and_effect() {
        // coincident_obs_intervention: output and effect both at t=10.
        let s = exact(1.0, 12.0, vec![10.0], vec![10.0]);
        let cur = Cursor::default();
        assert_eq!(s.substep(&cur, 9.0).unwrap(), 1.0, "steps to land on 10");
        assert!(
            s.output_due_at(&cur, 10.0) && s.effect_due_at(&cur, 10.0),
            "coincident kinds both due at the landing"
        );
    }

    #[test]
    fn fractional_end_clips_last_step() {
        // fractional_output_end: t_end = 80.5, dt = 1. The final step clips to 0.5
        // (lands on 80.5); stepping terminates there.
        let s = exact(1.0, 80.5, vec![], vec![]);
        let cur = Cursor::default();
        assert_eq!(s.substep(&cur, 80.0).unwrap(), 0.5);
        assert!(s.substep(&cur, 80.5).is_none(), "terminates at t_end");
    }

    #[test]
    fn substep_is_bit_exact_dt_min_not_t_to_minus_t() {
        // The robustness property: at large fractional t the substep is
        // dt.min(boundary - t) EXACTLY, NOT the FP-fragile (t+dt).min(boundary) - t.
        // Forward integer draws are insensitive to the ULP, but the PGAS continuous
        // transition density (shape = dt/σ²) is not — so the inference loglik would
        // move. This test pins the fix.
        let dt = 0.1;
        let s = exact(dt, 5000.0, vec![], vec![]);
        let cur = Cursor::default();
        let t = 1095.7275;
        let got = s.substep(&cur, t).unwrap();
        assert_eq!(
            got.to_bits(),
            dt.min(5000.0 - t).to_bits(),
            "substep must be the robust dt.min(boundary - t)"
        );
        let fragile = (t + dt).min(5000.0) - t;
        assert_ne!(got.to_bits(), fragile.to_bits(), "the fragile (t+dt)-t formula differs here");
    }

    #[test]
    fn next_boundary_agrees_with_substep() {
        // The single-source-of-truth invariant: `substep` is exactly
        // `dt.min(next_boundary - t)` for BOTH policies, bit-for-bit, at every
        // walk position. The adaptive ODE stepper consumes `next_boundary` as its
        // raw `h_max`; if it drifted from `substep`, fixed-RK4 (which clips
        // `dt.min(h_max)`) would stop moving the goldens silently.
        for s in [
            exact(1.0, 12.0, vec![0.0, 3.0, 7.0, 12.0], vec![2.5, 9.0]),
            snap(0.7, 13.3, vec![0.0, 2.0, 4.0], vec![3.5]),
            exact(0.1, 5000.0, vec![], vec![]),
        ] {
            let cur = Cursor::default();
            for &t in &[0.0_f64, 0.05, 1.0, 2.4999, 6.999, 1095.7275] {
                match (s.next_boundary(&cur, t), s.substep(&cur, t)) {
                    (Some(b), Some(step)) => assert_eq!(
                        step.to_bits(),
                        s.grid().min(b - t).to_bits(),
                        "substep must equal dt.min(next_boundary - t) at t={t}"
                    ),
                    (None, None) => {} // both past t_end
                    (nb, ss) => panic!("next_boundary/substep disagree on None at t={t}: {nb:?} vs {ss:?}"),
                }
            }
        }
    }

    #[test]
    fn substep_time_is_sdt_drift_free() {
        // window_start + s*dt is bit-exact (one multiply), while accumulating
        // dt over s steps drifts. At dt=0.1, s=10000 the two diverge.
        let dt = 0.1;
        let s = snap(dt, 5000.0, vec![], vec![]);
        let n: u64 = 10_000;
        // s*dt form:
        let sdt = s.substep_time(0.0, n);
        assert_eq!(sdt.to_bits(), (n as f64 * dt).to_bits());
        // accumulation drifts away from it:
        let mut acc = 0.0_f64;
        for _ in 0..n {
            acc += dt;
        }
        assert_ne!(acc.to_bits(), sdt.to_bits(), "accumulation must drift from s*dt by s=10000");
        // and s*dt equals the true grid point exactly (1000.0):
        assert_eq!(sdt, 1000.0);
        // window re-anchoring (EXACT steppers): offset by the window start.
        assert_eq!(s.substep_time(2.5, 3), 2.5 + 3.0 * dt);
    }

    #[test]
    fn obs_boundary_clips_like_an_effect_and_leaves_forward_untouched() {
        // Inference: obs at 7.3, dt=1. The Exact substep clips to land on it,
        // exactly as the bootstrap PF does (dt.min(obs - t)).
        let s = Schedule::new(1.0, 80.0, 1.0, StepPolicy::Exact, vec![], vec![]).with_obs(vec![7.3]);
        let cur = Cursor::default();
        assert_eq!(s.substep(&cur, 0.0).unwrap(), 1.0, "full dt before the obs");
        assert_eq!(s.substep(&cur, 7.0).unwrap(), 7.3 - 7.0, "clips exactly to the obs at 7.3");
        assert!(s.obs_due_at(&cur, 7.3));
        // Forward (no obs): byte-identical — next_obs is ∞, boundary unchanged.
        let f = exact(1.0, 5.0, vec![], vec![2.5]);
        let fo = exact(1.0, 5.0, vec![], vec![2.5]).with_obs(vec![]);
        let cur = Cursor::default();
        assert_eq!(f.substep(&cur, 2.0).unwrap().to_bits(), fo.substep(&cur, 2.0).unwrap().to_bits());
    }

    #[test]
    fn gillespie_clip_passes_through_and_clips() {
        let s = exact(1.0, 100.0, vec![5.0], vec![3.0]);
        let cur = Cursor::default();
        // Proposed reaction before the next boundary (3.0): pass through.
        let r = s.clip(&cur, 0.0, 2.4);
        assert_eq!(r.t, 2.4);
        assert!(!r.hit_boundary);
        // Proposed reaction past the boundary: clip to 3.0.
        let r = s.clip(&cur, 0.0, 3.7);
        assert_eq!(r.t, 3.0);
        assert!(r.hit_boundary);
    }

    #[test]
    fn clip_excludes_effect_exactly_at_t_but_not_output() {
        // The SSA asymmetry: a reaction landing exactly on an effect time (t=3.0,
        // effect at 3.0) must NOT clip back to it (already applied) — the > t
        // filter excludes it, so the proposed 4.0 passes through (next is out=5.0).
        let s = exact(1.0, 100.0, vec![5.0], vec![3.0]);
        let cur = Cursor::default();
        let r = s.clip(&cur, 3.0, 4.0);
        assert_eq!(r.t, 4.0);
        assert!(!r.hit_boundary);
        // An OUTPUT exactly at t is NOT excluded (no > t filter): output at 3.0,
        // t=3.0, proposed 4.0 → clips to 3.0.
        let s2 = exact(1.0, 100.0, vec![3.0], vec![]);
        let r2 = s2.clip(&cur, 3.0, 4.0);
        assert_eq!(r2.t, 3.0);
        assert!(r2.hit_boundary);
    }

    #[test]
    fn next_stop_simultaneous_output_and_observation() {
        // A time that is BOTH an output and an observation boundary: the stop
        // carries both reasons (and ScheduledEffect if an effect also lands).
        let s = Schedule::new(1.0, 12.0, 1.0, StepPolicy::Exact, vec![5.0], vec![5.0])
            .with_obs(vec![5.0]);
        let cur = Cursor::default();
        let stop = s.next_stop(&cur, 4.0).unwrap();
        assert_eq!(stop.t, 5.0);
        assert!(stop.reasons.contains(&StopReason::Output));
        assert!(stop.reasons.contains(&StopReason::Observation));
        assert!(stop.reasons.contains(&StopReason::ScheduledEffect));
        assert!(!stop.reasons.contains(&StopReason::End), "5.0 is not t_end");
        // Reason order is canonical: Output, ScheduledEffect, Observation.
        assert_eq!(
            stop.reasons.as_slice(),
            &[StopReason::Output, StopReason::ScheduledEffect, StopReason::Observation],
        );
    }

    #[test]
    fn next_stop_effect_only() {
        // An off-grid effect with no coincident output/obs: a lone
        // ScheduledEffect stop, landing exactly on the effect time.
        let s = exact(1.0, 5.0, vec![], vec![2.5]);
        let cur = Cursor::default();
        let stop = s.next_stop(&cur, 2.0).unwrap();
        assert_eq!(stop.t, 2.5);
        assert_eq!(stop.reasons.as_slice(), &[StopReason::ScheduledEffect]);
    }

    #[test]
    fn next_stop_end_only() {
        // No outputs/effects/obs remaining before t_end: the next stop is the
        // End boundary, with ONLY the End reason.
        let s = exact(1.0, 5.0, vec![], vec![]);
        let cur = Cursor::default();
        let stop = s.next_stop(&cur, 4.0).unwrap();
        assert_eq!(stop.t, 5.0);
        assert_eq!(stop.reasons.as_slice(), &[StopReason::End]);
    }

    #[test]
    fn next_stop_end_coincides_with_output() {
        // An output exactly at t_end: the stop carries Output AND End.
        let s = exact(1.0, 5.0, vec![5.0], vec![]);
        let cur = Cursor::default();
        let stop = s.next_stop(&cur, 4.0).unwrap();
        assert_eq!(stop.t, 5.0);
        assert_eq!(stop.reasons.as_slice(), &[StopReason::Output, StopReason::End]);
    }

    #[test]
    fn next_stop_none_past_end() {
        // Walked past t_end → no further stop.
        let s = exact(1.0, 5.0, vec![], vec![]);
        let cur = Cursor::default();
        assert!(s.next_stop(&cur, 5.5).is_none(), "no stop once t is past t_end");
        // At exactly t_end the End stop is still reported.
        assert!(s.next_stop(&cur, 5.0).is_some());
    }

    #[test]
    fn next_stop_t_is_the_next_boundary_min() {
        // next_stop.t is exactly min(t_end, next_output, next_effect, next_obs)
        // for the cursor — the same min the Exact substep clips to. A schedule
        // with all four kinds pending: the nearest (the effect at 2.5) wins.
        let s = Schedule::new(1.0, 12.0, 1.0, StepPolicy::Exact, vec![4.0], vec![2.5])
            .with_obs(vec![7.3]);
        let cur = Cursor::default();
        let stop = s.next_stop(&cur, 1.0).unwrap();
        assert_eq!(stop.t, 2.5, "nearest of {{12, 4, 2.5, 7.3}} is the effect at 2.5");
        assert_eq!(stop.reasons.as_slice(), &[StopReason::ScheduledEffect]);
    }

    #[test]
    fn substeps_fire_signal_and_cursor_advance_no_stall() {
        // gh#216: with effect_times populated, the Substeps iterator must surface
        // a fire signal at the effect landing AND advance the effect cursor so the
        // walk progresses (no zero-substep stall, proposal §3.3). dt=1, one obs at
        // 4, one on-grid effect at 2: substeps 0→1→2(fire)→3→4.
        let s = Schedule::new(1.0, 4.0, 1.0, StepPolicy::Exact, vec![], vec![2.0])
            .with_obs(vec![4.0]);
        let cur = Cursor { obs_idx: 0, effect_idx: s.effect_idx_at(0.0), ..Default::default() };
        // `take` bounds the walk so the MUTATION check (omitting `pass_effect` in
        // `next`) fails the length assertion cleanly instead of hanging on an
        // unbounded run of zero-length substeps clipped to the un-passed effect.
        let walk: Vec<(f64, f64, Option<usize>)> = s.substeps(cur, 0.0).take(50).collect();
        // Four substeps, terminating at obs 4 (NOT stalling on the effect at 2).
        assert_eq!(walk.len(), 4, "walk must reach obs 4 in four unit substeps, not stall");
        let fired: Vec<(f64, Option<usize>)> =
            walk.iter().map(|&(t0, dt, f)| (t0 + dt, f)).collect();
        assert_eq!(
            fired,
            vec![(1.0, None), (2.0, Some(0)), (3.0, None), (4.0, None)],
            "the effect fires exactly once, at the substep landing on t=2 (effect_idx 0)"
        );
    }

    #[test]
    fn effect_idx_at_partitions_fired_effects() {
        let s = exact(1.0, 20.0, vec![], vec![3.0, 7.0, 11.0]);
        assert_eq!(s.effect_idx_at(0.0), 0, "no effect fired yet at t=0");
        assert_eq!(s.effect_idx_at(3.0), 1, "effect at 3 counts as fired by t=3 (coincident-obs window)");
        assert_eq!(s.effect_idx_at(6.999), 1);
        assert_eq!(s.effect_idx_at(7.0), 2);
        assert_eq!(s.effect_idx_at(100.0), 3, "all effects fired past the end");
    }

    #[test]
    fn n_cursors_identical_sequence() {
        // THE CRN invariant: N independent cursors over one immutable Schedule
        // walk a byte-identical boundary sequence (next_boundary is pure; the
        // cursor is the only per-particle state).
        let s = exact(0.7, 13.3, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![3.5, 7.1, 11.0]);
        let reference = walk(&s);
        for _ in 0..64 {
            assert_eq!(walk(&s), reference, "per-cursor walk must be byte-identical");
        }
        assert!(!reference.is_empty());
    }

    #[test]
    fn substeps_iterator_matches_the_manual_filter_walk() {
        // The filters' inner loop is `while t_local < obs_time - 1e-10 { step_dt =
        // substep(cur, t_local); …; t_local += step_dt }`. The iterator must yield
        // the byte-identical (t_local, step_dt) sequence, per obs window.
        let s = Schedule::new(1.0, 12.0, 1.0, StepPolicy::Exact, vec![], vec![])
            .with_obs(vec![3.0, 7.3, 12.0]);
        let mut window_start = 0.0;
        for obs_idx in 0..3 {
            let cur = Cursor { obs_idx, ..Default::default() };
            let obs_time = s.obs_time(&cur).unwrap();
            // Manual walk (the loop being replaced).
            let mut manual = Vec::new();
            let mut t = window_start;
            while t < obs_time - 1e-10 {
                let dt = s.substep(&cur, t).unwrap();
                manual.push((t.to_bits(), dt.to_bits()));
                t += dt;
            }
            // Iterator walk (no effect_times here ⇒ fired is always None).
            let iter: Vec<(u64, u64)> = s
                .substeps(cur, window_start)
                .map(|(t0, dt, fired)| {
                    assert!(fired.is_none(), "no effects registered ⇒ no firing");
                    (t0.to_bits(), dt.to_bits())
                })
                .collect();
            assert_eq!(iter, manual, "iterator must reproduce the manual walk for window {obs_idx}");
            assert!(!manual.is_empty());
            // window_end is the catch-up advance: byte-identical to the manual
            // walk's final t (the divergence the 3 filters' re-walk risked).
            let manual_end = t;
            assert_eq!(s.window_end(cur, window_start).to_bits(), manual_end.to_bits(),
                "window_end must equal the manual walk's final t for window {obs_idx}");
            window_start = obs_time;
        }
    }
}
