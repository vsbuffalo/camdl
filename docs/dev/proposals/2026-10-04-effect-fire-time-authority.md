---
date: 2026-10-04
status: accepted — D1 and D2 decided by the maintainer (§5); no open questions
area: simulation timing spine / effect firing / inference correctness
issues: gh#957 (off-grid event), gh#958 (effect at t_start)
related:
  - 2026-06-05-unified-timeline-effect-architecture.md
  - 2026-06-07-scheduling-spine-v2.md
  - 2026-06-11-spine-effect-firing-consolidation.md
  - 2026-06-13-unify-effect-firing-cursor-keyed.md
---

# One authority for when an effect fires

## 1. Simulation and fitting currently define different models

An _effect_ is a scheduled state change: an `interventions {}` entry (off unless
enabled) or an `events {}` entry (always on). Both lower to one IR
`Intervention` and share one schedule grammar (language spec §13.5). Which
instant an effect acts at is decided by three separate pieces of code, and they
disagree for an effect at exactly `t_start`, an effect off the `dt` grid, and
any effect when `t_start` is not a multiple of `dt`. The particle filter (PF)
therefore scores a different model from the one `camdl simulate` generates, and
the forward backends disagree with one another.

The principle: **simulate and fit define the same model, per backend.** One
function computes each effect's firing instant from (declared time, kind,
schedule form, the backend's clock, `t_start`, run origin), and every path
derives its firing key from it. A backend's clock is its natural time:
discrete-time dynamics fire on their step grid, continuous-time dynamics at the
declared instant (D2, §5).

## 2. Three authorities, three answers

The scheduling-spine work (2026-06-05 through 2026-06-13) unified _where the
integrator stops_ and _how a due batch is applied_, but left _which batch is
due_ to two keys, and kept chain-binomial's `round(t/dt)` key as "correct
because Snap stays on the grid" (2026-06-13 §3.4) — true for observation times,
false for effect times.

| Path                                                                                       | Firing key (verified by reading)                                                                                                                                                                        | Effect at `t_start`                                                              | Off-grid effect                                                                                                                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| chain-binomial forward                                                                     | `due_effects` (`effects.rs:314`): `time_to_step(t_end, dt)` = `(t/dt).round()` (`time.rs:132`) against `resolve_fire_steps` (`compiled_model.rs:1345`), at each step end (`chain_binomial.rs:428`)      | never fires: no step ends at `t_start`                                           | snapped to the nearest step of the grid anchored at 0                                                                                                              |
| PGAS (Snap, the only PGAS the CLI accepts: `cli/src/fit/methods.rs:556`)                   | same key via `fill_producer_batch` (`pgas.rs:2105`; called at `:2194`, `:3116`)                                                                                                                         | never fires (inferred, same key)                                                 | snapped (inferred)                                                                                                                                                 |
| ODE forward and value path (`run_ode`, `ode.rs:813`, used by `inference/ode_loglik.rs:52`) | lands on `EffectTimes::from_model` (`boundary_times.rs:129` → `all_intervention_times`, `intervention.rs:209`), then `due_effects` at the landing (`ode.rs:1005`)                                       | fires, but after the `t_start` row is recorded (`ode.rs:961`): row is pre-effect | fires at its time; a second effect whose time rounds to the same absolute step fires early with it, and its own landing is skipped (`schedule.rs:592-600`, gh#198) |
| Gillespie forward                                                                          | as ODE (`gillespie.rs:147`, `:241`, `:310`)                                                                                                                                                             | as ODE                                                                           | as ODE                                                                                                                                                             |
| PF, IF2, PMMH                                                                              | cursor over `timeline_effects` (`intervention.rs:264`), each window starting at `effect_idx_at(t)` = `partition_point(e <= t + EFFECT_EPS)` (`schedule.rs:514`; `particle_filter.rs:349`, `if2.rs:547`) | dropped: counted as already fired                                                | events at their exact time; interventions refused by `guard_exact_offgrid_effect_time` (`intervention.rs:302`), events exempt (`:315`)                             |
| correlated PF                                                                              | as PF (`correlated_pf.rs:654-657`, `:911`)                                                                                                                                                              | dropped (inferred)                                                               | as PF, plus the loud noise-layout refusal when an effect adds a substep (`correlated_pf.rs:744`)                                                                   |
| ODE gradient (`nuts`, `det_grad`)                                                          | refuses any scheduled effect (`inference/gradient_capability.rs:333`, `ode.rs:1296`)                                                                                                                    | refused                                                                          | refused                                                                                                                                                            |

Measured on `main` (15cf347c), `beta = gamma = 0`, `dt = 1`: two interventions
each moving half of S to R at t = 0 and t = 1 leave S = 50 under chain-binomial
and 25 under ODE and Gillespie; events `add(I, 5)` at `[0]` and `[1]` give I = 5
and 10. The exact-marginal PF oracle (gh#201; pilot
`rust/crates/sim/tests/pf_exact_marginal_oracle.rs`, branch
`worktree-agent-a358e57f0a23e5735`) computes `p(y | θ)` exactly on an SIR with
N₀ ≈ 6: an intervention at `t_start` biases the filter by +0.85 nats (z = +363),
and it is unbiased against the model without the intervention; `add(I, 2)` at t
= 2.4 biases it by −0.355 nats (z = −190) against the snapped model, and it is
unbiased against exact-time firing.

Further divergences, measured with the installed `camdl 0.1.0+b276d7e5` (no
firing site changed between b276d7e5 and `main`) or read from code:

- **Recurring schedules lose their first fire under chain-binomial.** A block
  schedule's `from` defaults to `t_start` (spec line 3457) and the enumeration
  includes it (`intervention.rs:98`). `add(I, 1) { every = 2 'days }` over
  `[0, 6]` ends at I = 3 under chain-binomial, 4 under ODE and Gillespie. The
  spec's own `routine_vacc` (lines 3378-3382) and every `date_range(origin, …)`
  list (`ocaml/golden/sirv_anchored_calendar.camdl:47`) are affected.
- **The absolute key misfires effects inside the run when `t_start` is not a
  multiple of `dt`.** `from = 0.5`, `dt = 1`, events at 1, 2, 2.4, 3.5:
  chain-binomial drops the event at 1 (its step, `round(1) = 1`, is matched only
  by `t_end = 0.5 = t_start`), fires 2 and 2.4 at t = 1.5, and fires 3.5
  correctly. The run steps on `t_start + s·dt` (`schedule.rs:473`); the key uses
  a grid anchored at 0. `f64::round` (half away from zero) also breaks ties the
  other way for negative times.
- **ODE and Gillespie record the `t_start` row before the effect**, contrary to
  §13.10 (measured: row t = 0 shows I = 0, the effect appears at t = 1), so an
  observation at `t_start` in an ODE fit scores the pre-effect state.
- **The PF splits what the forward backends merge** (read from code):
  `timeline_effects` groups boundaries by exact equality
  (`intervention.rs:279-287`), so `at [2.3, 2.4]` on one intervention fires
  twice in the PF and once in chain-binomial.
- **The correlated PF and PGAS drop `t_start` effects** (read from code: the
  same `effect_idx_at(t_start)` at `correlated_pf.rs:911`; the same round key in
  PGAS).

## 3. The spec defines firing for one schedule form only

§13.2 gives `NAME : ACTION at [TIME, ...]     # times in model time_unit` (line
3449) and `from = DURATION   start of recurring (default: t_start)` (line 3457),
and says nothing about off-grid times or a fire at `t_start`. The snap rule is
in §13.7, "The `at_day` Schedule", scoped at line 3603 ("For `add` events and
interventions that recur on a specific day within each period"); lines
3613-3614: "Fire times are `at_day + k * period` for the smallest `k` where
`target >= t_start`. The engine fires on the single timestep where
`|t - target| < 0.5 * dt`". Its `>= t_start` admits a fire at `t_start`. §13.10
line 3802: "recorded output is **post-effect**". The IR spec contradicts itself:
chain-binomial "must truncate the current step to the intervention time"
(`compartmental-ir-spec.md:179`) versus "apply at the nearest step boundary
(rounding down)" (`:189`). No text covers `at [...]` off the grid, at `t_start`,
or on a grid not anchored at 0. The PR rewrites §13.2, §13.7 and IR §2.3.1 to
state §5's rule (the language spec by hand; `dprint` must not touch it).

## 4. The design: an `EffectPlan` built once per run

The plan decides _what is due_; the application seams stay (`step_one` via the
`effects` module, `lifecycle::apply_post_advance`,
`effects::apply_boundary_batch_continuous`, `intervention::apply_events_at`),
and the boundary authority stays `Schedule` (gh#233) — exact backends hand it
the plan's times rather than gaining a second stopping rule.

```rust
// rust/crates/sim/src/effect_plan.rs — the spine owns time.

/// The backend's natural time, chosen by the dynamics, not by the algorithm.
pub enum FireClock {
    /// Discrete-time dynamics (chain-binomial forward and every path that runs
    /// chain-binomial steps): fire on the run's grid t_start + k·dt.
    Grid { dt: f64 },
    /// Continuous-time dynamics (ODE, Gillespie): fire at the declared instant.
    Exact,
}

/// An `init {}` state has not seen effects at t_start; a resumed state
/// (contrast fork gh#322, `--init-state` forecast gh#641) already has.
pub enum Origin { Init, Resumed }

pub struct EffectPlan {
    /// Applied to the starting state before the first step and before the
    /// t_start row is recorded or scored. Empty under Origin::Resumed.
    at_origin: Vec<usize>,              // iv indices, declaration order
    /// Firing boundaries after t_start, ascending, index-aligned.
    times: Vec<f64>,                    // Grid: t_start + k·dt; Exact: declared
    steps: Vec<u64>,                    // Grid only: k ≥ 1
    batches: Vec<Vec<usize>>,           // declaration order
}

impl EffectPlan {
    pub fn build(model: &CompiledModel, params: &[f64], t_start: f64,
                 t_end: f64, clock: FireClock, origin: Origin)
        -> Result<Self, SimError>;
    pub fn at_origin(&self) -> &[usize];
    /// Grid consumers: the batch due at the end of substep s (k = s + 1).
    pub fn batch_ending_substep(&self, s: u64) -> &[usize];
    /// Landing times for the Schedule, index-aligned with `batch`.
    pub fn effect_times(&self) -> EffectTimes;
    pub fn batch(&self, effect_idx: usize) -> &[usize];
}

/// The grid key: k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋; None outside
/// [t_start, t_end]. GRID_TOL (intervention.rs:234) moves to schedule.rs.
fn fire_step(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<u64>;
```

Batching follows the clock. **Grid** merges by step: all of one intervention's
fires with the same k fire once (the gh#198 contract, `at [2.3, 2.4]` at
`dt = 1` fires once), because a discrete-time model has one boundary per step
and a second application in one step has no instant to belong to. **Exact**
groups by declared time (within `EFFECT_EPS`, the tolerance `arrive` already
uses for coincidence): each fire acts once at its own instant, so
`at [2.3, 2.4]` fires twice in ODE and Gillespie. The gh#198 merge existed only
to make the exact backends agree with chain-binomial; under D2 they define their
own model, and merging there is what fires an effect early (§2). `t_start`
membership: k = 0 under Grid (declared t within half a step after `t_start`);
`|t − t_start| ≤ EFFECT_EPS` under Exact.

| Path                       | Plan (clock)                                                                                                  | Origin batch                                                               | Due batch                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| chain-binomial forward     | Grid; replaces `chain_binomial.rs:285`; `Resumed` when `resume.start` is set                                  | after `initial_state_draw` (`:276`), before the initial row (`:356`)       | `batch_ending_substep(s)` replaces `due_effects` (`:428`)                                                           |
| PF, IF2, PMMH              | Grid; inside `ExactInferenceTimeline::build` (`intervention.rs:394`), clock from the process (`traits.rs:40`) | after `initial_state_draw` (`particle_filter.rs:155`, `if2.rs:472`)        | the cursor lands on `t_start + k·dt` and fires `batch(idx)`; no `t_start` boundary, so `effect_idx_at(t_start)` = 0 |
| correlated PF              | Grid; replaces `correlated_pf.rs:654-657`                                                                     | after `initial_state_draw_correlated`                                      | as PF                                                                                                               |
| PGAS                       | Grid; replaces `pgas.rs:2174`, `:2864`, `:3963-3966`, `pgas_init.rs:203`                                      | see below                                                                  | `fill_producer_batch` (`pgas.rs:2097`) reads `batch_ending_substep(s)`                                              |
| ODE forward and value path | Exact; replaces `ode.rs:905`, `:1192`                                                                         | after `initial_state_mean` (`ode.rs:841`), before the initial row (`:961`) | `Schedule::exact_forward(…, plan.effect_times())`; `arrive` applies `batch(idx)` (`:1005`)                          |
| Gillespie forward          | Exact; replaces `gillespie.rs:147`                                                                            | after `initial_state_draw` (`:121`), before the initial row (`:180`)       | as ODE (`:241`, `:310`)                                                                                             |
| ODE gradient               | Exact; the refusal (`gradient_capability.rs:333`) reads the plan                                              | —                                                                          | —                                                                                                                   |

**PGAS** (high-risk surface): `initial_counts` stays x₀, the state the `init {}`
law scores (`initial_state_logpdf`, `pgas.rs:1835`); the origin batch is a
deterministic map applied wherever the substep-0 pre-state is built — reference
producer, conditional-SMC free particles (`draw_free_particle_initial_state`),
`pgas_init.rs`, and the complete-data density walk. The PGAS tests (§6) prove no
site is missed.

**Removed:** `effects::due_effects`, `CompiledModel::resolve_fire_steps`,
`time::fire_times_to_steps`, `intervention::timeline_effects`,
`intervention::all_intervention_times`, `guard_exact_offgrid_effect_time` (Grid
times are on the grid by construction), and the step-skip in `Schedule::arrive`
(`schedule.rs:592-600`) with `effect_maps_to_step`.
`validate_recurring_dt_collisions` (`compiled_model.rs:1299`) keys on
`fire_step` and runs for Grid clocks only: under Exact two fires closer than
`dt` are distinct instants, so Gillespie's call through `validate_schedule`
(`gillespie.rs:146`) stops applying it. Unchanged: reactive interventions (fired
by their agenda), the refusal of parametric `at [<param>]` under inference
(`intervention.rs:356`), the ODE-gradient refusal, and the correlated PF's loud
refusal when a grid effect inside a window that starts off the grid adds a
substep (`correlated_pf.rs:744`).

## 5. Decisions

**D1 — An effect at `t_start` applies to the starting state on every path.** The
order is `init {}` (laws included) → inflow events read x₀ → draining events and
`set` → interventions in declaration order → `balance` → the non-negativity
check → the `t_start` row is recorded and any observation at `t_start` scored:
§13.10 without the transition stage, which is what Gillespie already does at a
boundary (`gillespie.rs:239-245`), so the origin batch reuses `apply_events_at`
and `apply_post_advance`. `Origin::Resumed` never fires it, because a forked or
forecast state is already post-effect. Rationale: `at [t_start]` plainly means
it, the spec's enumeration (`target >= t_start`, `from` defaulting to `t_start`)
produces such fires, and ODE and Gillespie already apply them. Refusing at
compile time was rejected: it would break every `every = …` without `from`,
every `date_range(origin, …)` list and the `sirv_anchored_calendar` golden, and
cannot see a runtime `t_start`.

**D2 — Each backend fires in its own natural time.** Paths running
chain-binomial steps (forward, PF, IF2, PMMH, correlated PF, PGAS) fire on the
grid `t_start + fire_step(t)·dt`; ODE and Gillespie (forward, ODE fits, ODE
gradient paths) fire at the declared time. Rationale: the model a fit targets is
the model its backend simulates; a discrete-time process has no state between
steps, so its effects belong to a step, while a continuous-time process has an
exact instant and should not acquire a `dt` dependence it does not otherwise
have. Consequences: the PF family snaps events (fixes gh#957) and accepts
off-grid scheduled interventions by snapping them (the refusal is deleted);
chain-binomial forward changes only by D1 and the relative key; ODE and
Gillespie timing of off-grid effects is unchanged except that a same-step
neighbour no longer fires early. Considered and rejected: (a) snap everywhere —
gives ODE and Gillespie a `dt`-dependent effect instant they do not need; (b)
exact everywhere — requires clipping chain-binomial substeps at effect times (a
new random-number stream for every such model, including He et al. 2010's cohort
pulse at `258 + k·365.25`), effect landings in PGAS, and noise-layout growth in
the correlated PF, and departs from the pomp reference the He 2010 external
cases use (`tests/external/cases/he2010_forward/reference/reference.R:109`,
`fabs(t - floor(t) - 251.0/365.0) < 0.5*dt`); (c) refuse off-grid events under
inference — leaves forward disagreement in place and refuses the He 2010 model
in the PF (its cohort event is off the daily grid three years in four), with no
snap PF to fall back on (`methods.rs:526-531`). (d) is the only option under
which every backend's simulation and fit agree without a new dependence or a new
refusal.

**The grid key is relative to `t_start`, ties to the later step.** This is a bug
fix, not a choice (§2). `⌊x + ½⌋` equals `f64::round` for the non-negative
offsets that can fire, so no run whose `t_start` is a multiple of `dt` changes.
A resumed run keys on its own origin; a contrast fork resumes at an output time
on the original grid, so the grids differ by whole steps (inferred;
`splice_invariant.rs` checks it). Effects outside `[t_start, t_end]` do not
fire.

## 6. Tests

Everything lands with the fix; nothing is `#[ignore]`d on `main`. Each fix gets
a mutation check (revert the source change, keep the test, confirm red, record
the red output in the PR).

1. **The oracle**, carried in from the pilot. Un-ignore
   `off_grid_event_fires_where_the_model_says` and
   `intervention_at_t_start_fires`; delete the two tests that pin today's
   behaviour (`off_grid_event_pf_matches_exact_time_semantics`,
   `intervention_at_t_start_pf_matches_never_firing`). Add, each for PF, IF2 and
   the correlated PF: an off-grid intervention (snapped); an event at `t_start`
   with an observation at `t_start`; `t_start = 0.5`, `dt = 1` with effects at 1
   and 2.4; `at [2.3, 2.4]` on one intervention (fires once).
2. **Forward agreement**, zero-rate fixtures in `tests/fixtures/corner_cases/`
   through `cross_backend_lifecycle_agreement.rs`. Exact agreement of all three
   backends on every row, including the `t_start` row, for: interventions and
   events at `t_start` and on the grid; a recurring schedule with default
   `from`; the offset-origin model. For an off-grid effect: chain-binomial fires
   at `t_start + fire_step(t)·dt`, ODE and Gillespie at `t`. The three gh#198
   tests (`:507`, `:529`, `:555`) are rewritten to the per-backend rule:
   chain-binomial fires once, ODE and Gillespie once per declared time; a new
   test pins that an effect at 2.4 does not fire at an earlier landing at 2.0.
3. **PGAS and resume.** The producer's trajectory carries a `t_start` effect;
   `complete_data_loglik` matches the oracle's complete-data value on a fixed
   path with a `t_start` effect and an init law (x₀ and the post-effect state
   are not conflated); an off-grid event is snapped in the producer; a contrast
   fork at a time with an effect does not fire it twice (`splice_invariant.rs`).
4. **Inversions.** `pf_rejects_off_grid_effect_time`
   (`gh216_cursor_firing.rs:594`) becomes a snap test; the `arrive` step-skip
   unit tests in `schedule.rs` go with the code.

## 7. Identity and goldens

No IR change: no `ir/VERSION` bump, no `*.ir.json` golden moves. Every run
re-keys anyway, because the model digest folds the engine version
`0.1.0+<git hash>` (`runid/src/inputs.rs:163-168, 292`).

No committed baseline moves (inferred from the fixtures' fire times; the PR's
gate run is the verification). The golden trajectory gate clears every effect
before running (`gate_trajectory_baseline.rs:459`). The inference baseline gates
carry no effects (`gate_inference_baseline.rs:124`; no schedule in
`gate_pgas_thread_invariance.rs`, `gate_unconditional_pass_ab.rs`). In the
corner-case gate (`gate_corner_case_baseline.rs`) every effect is on the grid
anchored at `t_start = 0` and none is at `t_start` (`multi_effect_same_time` and
`event_intervention_agree` at 5, `gh70_absorbing_importation` and the culls at
10, `all_lifecycle` importation from 5, `event_drain_fusion` from 1), except
`off_grid_intervention` (2.5): chain-binomial still fires at 3
(`⌊2.5 + ½⌋ = 3`), ODE and Gillespie still at 2.5, so its three hashes
(`:139-141`) stay. If any baseline does move, that is a finding to explain
before re-capturing, and the re-capture is a reviewed commit of its own. The He
2010 PF log-likelihood external case moves (the cohort event now snaps, as in
the pomp reference) and should be re-run before release.

## 8. One PR, in this order

1. Move `GRID_TOL` to `schedule.rs`; add `effect_plan.rs` (`FireClock`,
   `Origin`, `fire_step`, `EffectPlan`), unit-tested on the cases of §2.
   Foundation for this PR, wired by step 2.
2. Forward paths: chain-binomial (Grid), ODE and Gillespie (Exact), origin
   batch, `Origin::Resumed`, the agreement fixtures and gh#198 test rewrites
   (§6.2).
3. PF, IF2, correlated PF: Grid plan; delete `guard_exact_offgrid_effect_time`;
   the oracle (§6.1).
4. PGAS and `pgas_init` (§6.3).
5. Delete the replaced functions and the `arrive` step-skip; restrict the
   recurring-collision check to Grid; rewrite spec §13.2, §13.7 and IR §2.3.1.

Each step ends green on `make test-inference`; the PR lands on a green
`make test`.
