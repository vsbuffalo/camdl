---
date: 2026-10-04
status: accepted — D1–D4 and the window rule decided by the maintainer (D1, D2 on 2026-10-06; D3, D4 and the window rule on 2026-10-24); implementation follows §8, preceded by the application-seam PR (step 0)
area: simulation timing spine / effect firing / inference correctness
issues: gh#957 (off-grid event), gh#958 (effect at t_start); gh#984, gh#985, gh#986 (filed 2026-10-24 from this work)
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
schedule form, the backend's clock, `t_start`, `t_end`, run origin), and every
path derives its firing key from it. A backend's clock is its natural time:
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
| PGAS (Snap, the only PGAS the CLI accepts: `cli/src/fit/methods.rs:556`)                   | same key via `fill_producer_batch` (`pgas.rs:2095`; called at `:2194`, `:2299`, `:3115`, `pgas_init.rs:300`)                                                                                            | never fires (inferred, same key)                                                 | snapped (inferred)                                                                                                                                                 |
| ODE forward and value path (`run_ode`, `ode.rs:813`, used by `inference/ode_loglik.rs:52`) | lands on `EffectTimes::from_model` (`boundary_times.rs:129` → `all_intervention_times`, `intervention.rs:209`), then `due_effects` at the landing (`ode.rs:1005`)                                       | fires, but after the `t_start` row is recorded (`ode.rs:961`): row is pre-effect | fires at its time; a second effect whose time rounds to the same absolute step fires early with it, and its own landing is skipped (`schedule.rs:592-600`, gh#198) |
| Gillespie forward                                                                          | as ODE (`gillespie.rs:147`, `:241`, `:310`)                                                                                                                                                             | as ODE                                                                           | as ODE                                                                                                                                                             |
| PF, IF2, PMMH                                                                              | cursor over `timeline_effects` (`intervention.rs:264`), each window starting at `effect_idx_at(t)` = `partition_point(e <= t + EFFECT_EPS)` (`schedule.rs:514`; `particle_filter.rs:349`, `if2.rs:547`) | dropped: counted as already fired                                                | events at their exact time; interventions refused by `guard_exact_offgrid_effect_time` (`intervention.rs:302`), events exempt (`:315`)                             |
| correlated PF                                                                              | as PF (`correlated_pf.rs:654-657`, `:911`)                                                                                                                                                              | dropped (inferred)                                                               | as PF, plus the loud noise-layout refusal when an effect adds a substep (`correlated_pf.rs:744`)                                                                   |
| ODE gradient (`nuts`, `det_grad`)                                                          | refuses any scheduled effect (`inference/gradient_capability.rs:333`, `ode.rs:1293-1300`)                                                                                                               | refused                                                                          | refused                                                                                                                                                            |

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
- **Effects just outside the run fire on one side each** (measured; zero-rate
  `add(A, 100)` with `A = 50` at `init`). Declared at 1.7 with `from = 2`: ODE
  and Gillespie carry A = 150 from the t = 3 row on, chain-binomial never fires
  it. Declared at 4.3 with `to = 4`: chain-binomial carries A = 150 on the t = 4
  row, ODE and Gillespie never fire it. Both are the absolute key again: 1.7 and
  4.3 round to the steps that end at `t_start` and `t_end`.
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
- **The contrast fork reads the declared time, not the firing instant** (read
  from code). `derive_fork` (`cli/src/fit/contrasts.rs:694-768`) takes the
  earliest toggled fire time from the schedule (`intervention_earliest_fire`,
  `:799`) and forks at the last saved snapshot strictly before it
  (`fork_before`, `:773-779`). The arms resume chain-binomial, which fires an
  intervention declared at 2.4 on the step ending at 2, so with snapshots on the
  integer grid the fork lands on the post-effect snapshot at 2 and both arms
  carry the effect the contrast is meant to isolate.

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
or on a grid not anchored at 0. The PR rewrites §13.2, §13.7, §13.10 and IR
§2.3.1 to state §5's rule (the language spec by hand; `dprint` must not touch
it); the full list of documents is in §8 step 5.

## 4. The design: an `EffectPlan` built once per run

The plan decides _what is due_; the application seams stay (`step_one` via the
`effects` module, `lifecycle::apply_post_advance`,
`effects::apply_boundary_batch_continuous`, `intervention::apply_events_at`),
plus one new origin function (`effects::apply_origin_batch`, D1), and the
boundary authority stays `Schedule` (gh#233) — exact backends hand it the plan's
times rather than gaining a second stopping rule.

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
    times: Vec<f64>,                    // Grid: min(t_start + k·dt, t_end); Exact: declared
    steps: Vec<u64>,                    // Grid only: 1 ≤ k ≤ n_substeps
    batches: Vec<Vec<usize>>,           // declaration order
}

impl EffectPlan {
    pub fn build(model: &CompiledModel, params: &[f64], t_start: f64,
                 t_end: f64, clock: FireClock, origin: Origin)
        -> Result<Self, SimError>;
    pub fn at_origin(&self) -> &[usize];
    /// True when both `at_origin` and `times` are empty.
    pub fn is_empty(&self) -> bool;
    /// Grid consumers: the batch due at the end of substep s (k = s + 1).
    pub fn batch_ending_substep(&self, s: u64) -> &[usize];
    /// Landing times for the Schedule, index-aligned with `batch`.
    pub fn effect_times(&self) -> EffectTimes;
    pub fn batch(&self, effect_idx: usize) -> &[usize];
}

/// The grid key: k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋ for t inside the
/// window; None outside it (see "Window ends"). GRID_TOL (intervention.rs:234)
/// moves to schedule.rs.
fn fire_step(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<u64>;
```

**Window ends.** A Grid run takes `n_substeps` steps of `dt`, the last one
shortened to land on `t_end` when `t_end − t_start` is not a multiple of `dt`
(`schedule.substep` is `dt.min(t_end − t)`, `chain_binomial.rs:398`). The window
test comes first: an effect fires only if `t_start ≤ t ≤ t_end`, with
`GRID_TOL·dt` of slack under Grid and `EFFECT_EPS` (`schedule.rs:165`, the
tolerance every due-test already uses) under Exact. The slack matters because
the recurring enumeration overshoots its bound by float error
(`intervention.rs:93`, `:99`: `t <= rs.end + rs.period·1e-9`), so a last fire
computed as `t_end + 1e-12` is the fire at `t_end`. Inside the window the Grid
key is `k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋` capped at `n_substeps`, and the
instant it names is `min(t_start + k·dt, t_end)`. One refinement for a shortened
last substep, decided 2026-10-24: an effect that falls inside that substep snaps
to the nearer of its two actual boundaries, `t_start + (n_substeps − 1)·dt` and
`t_end`, not to the nominal grid point `t_start + n_substeps·dt` that the step
never reaches. With `t_end = 2.5`, `dt = 1` and an effect at 2.45, the nominal
key gives k = 2 and fires it at 2.0 although 2.5 is 0.05 away; the refinement
fires it at 2.5. The two keys differ only for a time in the second half of a
shortened last substep, so a run whose window is a whole number of steps is
unaffected. Whenever the refinement and not the nominal key decided the step,
`EffectPlan::build` emits a warning on stderr naming the effect, its declared
time and the boundary it snapped to, so a modeller sees the one case where the
run's length rather than the grid chose the instant. Extending `t_end` to the
next grid point was considered and rejected: windows declared in years in a
daily model are the normal case (the He 2010 vignette runs `to = 21 'years`,
7670.0925 days at `dt = 1`), extension would move their final row and every
`final()` quantity, and under the particle filter the last observation defines
`t_end`. No model in the book or the external projects has an effect in a
shortened last substep.

**Amounts.** An action's expression is evaluated at the firing instant with the
backend's step — `cfg.dt` for chain-binomial and ODE, `iv_resolution_dt` for
Gillespie (`gillespie.rs:142`) — at the origin as well. An expression that reads
`t` therefore sees `t_start` at the origin and `t_start + k·dt` under Grid, not
the declared time, because that is when the effect acts.

**Batching follows the clock.** **Grid** merges by step: all of one
intervention's fires with the same k fire once (the gh#198 contract,
`at [2.3, 2.4]` at `dt = 1` fires once), because a discrete-time model has one
boundary per step and a second application in one step has no instant to belong
to. **Exact** groups by declared time within `EFFECT_EPS`: each fire acts once
at its own instant, so `at [2.3, 2.4]` fires twice in ODE and Gillespie, while
one intervention listed twice within `EFFECT_EPS` (`at [2.0, 2.0 + 1e-12]`) is
one fire, and two interventions at one instant are one batch in declaration
order. The gh#198 merge existed only to make the exact backends agree with
chain-binomial; under D2 they define their own model, and merging there is what
fires an effect early (§2). `t_start` membership: k = 0 under Grid (declared t
within half a step after `t_start`); `|t − t_start| ≤ EFFECT_EPS` under Exact.

**Empty origin.** The origin stage is a no-op when `at_origin` is empty. It must
not reach `apply_post_advance` (`lifecycle.rs:50`), which applies `balance`
whether or not the batch is empty (`:69-83`) and would otherwise rewrite the
`t_start` row of every `balance {}` model.

| Path                                       | Plan (clock)                                                                                                                                                                                                                                                                                | Origin batch                                                                                                                                                                        | Due batch                                                                                                                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| chain-binomial forward                     | Grid; replaces `chain_binomial.rs:285`; `Resumed` when `resume.start` is set                                                                                                                                                                                                                | after `initial_state_draw` (`:276`), before the initial row (`:356`)                                                                                                                | `batch_ending_substep(s)` replaces `due_effects` (`:428`)                                                                                                                                                                                       |
| PF, IF2, PMMH                              | Grid, from a new `ProcessModel::fire_clock(dt)` (`traits.rs:40`; `ChainBinomialProcess` is the trait's only implementor and stores no `dt`, so the step arrives per call as it does for `step`), passed into `ExactInferenceTimeline::build` (`intervention.rs:399`), which builds the plan | a new `ProcessModel::apply_origin(state, plan, θ)` after `initial_state_draw`: the PF at the shared θ (`particle_filter.rs:155`), IF2 with each particle's own θ (`if2.rs:468-472`) | the cursor lands on the plan's times and fires `batch(idx)`; no `t_start` boundary, so `effect_idx_at(t_start)` = 0                                                                                                                             |
| correlated PF                              | Grid; replaces `correlated_pf.rs:654-657` (it bypasses `build`, `:679`, and builds the plan itself)                                                                                                                                                                                         | `apply_origin` after `initial_state_draw_correlated`                                                                                                                                | as PF                                                                                                                                                                                                                                           |
| PGAS                                       | Grid; replaces `pgas.rs:2174`, `:2864`, `pgas_init.rs:203`; the Exact firing branch (`pgas.rs:3933-3986`) is deleted (below)                                                                                                                                                                | see below                                                                                                                                                                           | `fill_producer_batch` (`pgas.rs:2095`) reads `batch_ending_substep(s)` for its four callers: the reference walk (`:2194`), the CSMC sweep (`:3115`), `SpliceGuard::from_reference` (`:2299`) and the initialization filter (`pgas_init.rs:300`) |
| ODE forward and value path                 | Exact; replaces `ode.rs:897`, `:905`, `:1189`, `:1192`                                                                                                                                                                                                                                      | after `initial_state_mean` (`ode.rs:841`), before the initial row (`:961`)                                                                                                          | `Schedule::exact_forward(…, plan.effect_times())`; `arrive` applies `batch(idx)` (`:1005`)                                                                                                                                                      |
| Gillespie forward                          | Exact; replaces `gillespie.rs:147`, `:169`                                                                                                                                                                                                                                                  | after `initial_state_draw` (`:121`), before the initial row (`:181`)                                                                                                                | `apply_events_at` (`intervention.rs:158`) takes the batch instead of calling `due_effects` (`:184`); the closures at `gillespie.rs:239`, `:241`, `:308`, `:310` pass `plan.batch(idx)` to it and to `apply_post_advance`                        |
| ODE gradient (`det_grad`, `nuts`)          | Exact; the entry refusal (`gradient_capability.rs:333`) reads the plan and refuses when `at_origin ∪ times` is non-empty                                                                                                                                                                    | — (refused)                                                                                                                                                                         | — (refused; the in-loop refusal through `due_effects` at `ode.rs:1293-1300` goes with `due_effects`)                                                                                                                                            |
| contrast fork (`cli/src/fit/contrasts.rs`) | the resumed arm's clock (Grid); `Origin::Resumed` via `resume.start`                                                                                                                                                                                                                        | — (already post-effect)                                                                                                                                                             | `derive_fork` (`:694-768`) forks strictly before the plan's firing instant of the earliest toggled fire, `t_start + k·dt`, not the declared time (`fork_before`, `:773-779`)                                                                    |

**ODE gradient.** The refusal must read both halves of the plan. A model whose
only effect is at `t_start` has an empty `times` and a non-empty `at_origin`; a
refusal keyed on `times` alone would admit it, and `det_grad` and `nuts` would
then fit the model without its effect — the gradient path applies no origin
batch. That outcome is silent-wrong, so §6.4 pins the refusal on exactly that
model.

**Contrast fork.** The fork must be strictly before the instant the resumed arm
fires at, which under Grid is `t_start + k·dt`, not the declared time (§2, last
bullet). Forking at the declared time with an effect at 2.4 and snapshots on the
integer grid picks the post-effect snapshot at 2, and the contrast then measures
nothing; that is silent-wrong, so §6.4 pins the fork at 1.

**PGAS** (high-risk surface). `initial_counts` stays x₀, the state the `init {}`
law scores (`initial_state_logpdf`, `pgas.rs:1835`); the origin batch is a
deterministic map from x₀ to `substeps[0].counts_before`, applied at three
places and nowhere else. In the reference walk, after `initial_state_draw`
(`:2161`) and before the first `step_one`. In the CSMC sweep and the
initialization filter, after the `initial_counts_per_particle` snapshot
(`pgas.rs:2930`, `pgas_init.rs:214`) and never inside
`draw_free_particle_initial_state` (`:2782`), so the traceback (`:3486`,
`pgas_init.rs:464`) still records x₀. For the reference particle `j_ref` the
pre-state is read, not recomputed: `counts[j_ref]` starts from
`reference.substeps[0].counts_before` (today `:2890` clones
`reference.initial_counts`), so the `prev_counts` bookkeeping (`:2952`, `:3175`,
`:3322`) sees the same post-origin state the reference walk had.
`complete_data_loglik` (`:1803`) gains the debug invariant
`substeps[0].counts_before == origin(initial_counts)` and scores no new term
(D3). `SpliceGuard::from_reference` (`:2299`) measures its headroom on the
recorded substeps, which are post-origin states, so the offset Δ it tests is a
shift between two post-origin states and `initial_counts` is never shifted.
Missing any one of these sites is silent-wrong — the sweep would mix x₀ and its
image as the substep-0 pre-state, the density would score transitions from a
state the producer never used, and the posterior would still print — so §6.3 is
built to catch each of them.

**Removed:** `effects::due_effects`, `CompiledModel::resolve_fire_steps`,
`time::fire_times_to_steps`, `intervention::timeline_effects`,
`intervention::all_intervention_times`, `guard_exact_offgrid_effect_time` (Grid
times are on the grid by construction), the step-skip in `Schedule::arrive`
(`schedule.rs:592-600`) with `effect_maps_to_step`, the `effects.rs:1030` test
shim, and the PGAS Exact firing branch (`pgas.rs:3933-3986`: the
events-under-Exact refusal, the two guards, `timeline_effects` and the `Some`
arm of `EffectFiring`). The CLI cannot reach that branch
(`cli/src/fit/methods.rs:556` refuses `obs_alignment = "exact"` for PGAS), and
what only it consumed goes with it.

`validate_recurring_dt_collisions` (`compiled_model.rs:1299`) takes `t_start`,
keys on `fire_step`, and runs for Grid clocks only: under Exact two fires closer
than `dt` are distinct instants. That removes it from every Exact site —
`validate_schedule` (`:1256`) as called by ODE (`ode.rs:824`, `:1115`) and
Gillespie (`gillespie.rs:146`), and the fit pre-flight
(`cli/src/fit/mod.rs:126-149`), which today runs it for every algorithm and
again for `burnin_dt`; when the algorithm is ODE-backed both steps are Exact
steps and neither can drop a fire. `at_day_fire_collision.rs` is rescoped to
Grid (§6.5).

Unchanged: reactive interventions (fired by their agenda), the refusal of
parametric `at [<param>]` under Exact observation alignment
(`intervention.rs:356`; D3 adds the PGAS case), and the correlated PF's loud
refusal when a grid effect inside a window that starts off the grid adds a
substep (`correlated_pf.rs:744`).

## 5. Decisions

**D1 — An effect at `t_start` applies to the starting state on every path.** The
order is `init {}` (laws included) → inflow events, read from x₀ → draining
events and `set`, read from the post-inflow state (D4) → interventions in
declaration order → `balance` → the non-negativity check → the `t_start` row is
recorded and any observation at `t_start` scored: §13.10 without the transition
stage. One function, `effects::apply_origin_batch`, does it for every backend
from the plan's `at_origin`: the two event phases through `resolve_event_batch`
(the phases `step_one` already runs, `chain_binomial.rs:784-816`, without the
draw between them), then `apply_post_advance` (`lifecycle.rs:50`) for the
interventions, `balance` and the check — and nothing at all when the batch is
empty (§4). Amounts evaluate at `t_start` with the backend's step (§4).
`Origin::Resumed` never fires it, because a forked or forecast state is already
post-effect. Rationale: `at [t_start]` plainly means it, the spec's enumeration
(`target >= t_start`, `from` defaulting to `t_start`) produces such fires, and
ODE and Gillespie already apply them. Refusing at compile time was rejected: it
would break every `every = …` without `from`, every `date_range(origin, …)` list
and the `sirv_anchored_calendar` golden, and cannot see a runtime `t_start`.

**D2 — Each backend fires in its own natural time.** Paths running
chain-binomial steps (forward, PF, IF2, PMMH, correlated PF, PGAS) fire on the
grid `t_start + fire_step(t)·dt`; ODE and Gillespie (forward, ODE fits, ODE
gradient paths) fire at the declared time. Rationale: the model a fit targets is
the model its backend simulates; a discrete-time process has no state between
steps, so its effects belong to a step, while a continuous-time process has an
exact instant and should not acquire a `dt` dependence it does not otherwise
have. Consequences: the PF family snaps events (fixes gh#957) and accepts
off-grid scheduled interventions by snapping them (the refusal is deleted);
chain-binomial forward changes by D1 and by the key and window rule below; ODE
and Gillespie keep the declared instant for every effect inside the window and
change by D1, by the window rule, and by the end of the gh#198 merge (a
same-step neighbour no longer fires early). Considered and rejected: (a) snap
everywhere — gives ODE and Gillespie a `dt`-dependent effect instant they do not
need; (b) exact everywhere — requires clipping chain-binomial substeps at effect
times (a new random-number stream for every such model, including He et al.
2010's cohort pulse at `258 + k·365.25`), effect landings in PGAS, and
noise-layout growth in the correlated PF, and departs from the pomp reference
the He 2010 external cases use
(`tests/external/cases/he2010_forward/reference/reference.R:109`,
`fabs(t - floor(t) - 251.0/365.0) < 0.5*dt`); (c) refuse off-grid events under
inference — leaves forward disagreement in place and refuses the He 2010 model
in the PF (its cohort event is off the daily grid three years in four), with no
snap PF to fall back on (`methods.rs:526-531`). The chosen rule is the only one
under which every backend's simulation and fit agree without a new dependence or
a new refusal.

**The grid key is relative to `t_start`, ties to the later step, and the window
is `[t_start, t_end]`.** A bug fix (§2), and not a behaviour-neutral one.
Today's key is `(t/dt).round()` (`time.rs:132`), half away from zero on a grid
anchored at 0, matched against the end of each step; the new key is
`⌊(t − t_start)/dt + ½ + GRID_TOL⌋` on the run's own grid, inside the window
only. The runs that change, on every Grid path:

- `t_start` not a multiple of `dt`: the §2 case (`from = 0.5`, `dt = 1`, events
  at 1, 2, 2.4 — measured) fires 1 at 1.5 and both 2 and 2.4 at 2.5, instead of
  dropping 1 and firing 2 and 2.4 at 1.5.
- a tie at a negative time: `f64::round` takes −0.5 to −1, so with `from = −2`
  and `dt = 1` an effect at −0.5 fires at −1 today and at 0 after (ties go to
  the later step whatever the sign).
- a time within `GRID_TOL·dt` below a tie: 1.5 − 1e-10 fires at 1 today and at 2
  after.
- the window: an effect in `(t_end, t_end + dt/2]` fires on the last step today
  (measured: `to = 4`, `add(A, 100) at [4.3]` lands on the t = 4 row) and does
  not fire after.
- a shortened last substep: an effect in its second half fires at `t_end`, the
  nearest actual boundary (§4), instead of at the preceding grid point, with a
  warning; no model in the book or the external projects is affected.

Under Exact: an effect in `(t_start − dt/2, t_start)` fires today (measured:
`from = 2`, `add(A, 100) at [1.7]` is in the t = 3 row) and does not fire after;
an effect at `t_start` moves into the `t_start` row (D1); a same-step neighbour
no longer fires early (§2, gh#198). A resumed run keys on its own origin; a
contrast fork resumes at an output time on the original grid, so the grids
differ by whole steps (inferred; `splice_invariant.rs` checks it).

**D3 — Under PGAS, an estimated parameter inside an effect's action or schedule,
or inside a plain `init {}` assignment, is refused in this PR; under IF2, an
estimated parameter inside an effect's schedule is refused too.**
`complete_data_loglik` (`pgas.rs:1803`) scores `log p(x₀ | θ)` from
`trajectory.initial_counts` through `initial_state_logpdf` (`:1835`), then for
each substep the transition density of `rec.flows` given `rec.counts_before`.
The deterministic effect map between substeps — and, after D1, between x₀ and
`substeps[0].counts_before` — appears in no term. With `init { I = 0 }`,
`events { seed : add(I, I0) at [0] }` and `I0` estimated, a proposed `I0′`
changes no term: the complete-data density is flat in `I0`, the θ-move accepts
on the prior ratio alone, and the reported posterior for `I0` is its prior, with
no warning. The same holds today for `add(I, I0) at [5]`, so the bias is
pre-existing; D1 makes the `at [0]` form live too. A parametric `at [<param>]`
schedule whose parameter is estimated is the same flatness through the fire step
rather than the amount, and nothing refuses it under Snap (the guard at
`intervention.rs:356` runs only under Exact, which the CLI never gives PGAS).
This is read from code, not run; the test that would show it is a PGAS fit of
that model, whose `I0` posterior must move off the prior — red today. The
refusal lives at the `run_pgas` boundary (`pgas.rs:3890`), where the estimated
set is known (`if2_params`, the set the NUTS gradient-coverage refusal at
`:4371-4392` already reads): any effect action (`add`, `transfer` by `fraction`
or `count`, `set`) or `at [...]` expression that references an estimated
parameter is refused with the parameter and the effect named, through a
parameter-reference walk over the resolved expressions
(`compiled_model.rs:1960-1975`, `:1988`; the sibling of `references_state`,
`:1994`). `check_model_capabilities` (`methods.rs:710`) cannot host it: it sees
the backend and the model, not the fit's estimated set. Rejected: (b) ship and
file an issue — a posterior that silently equals its prior is the outcome this
repository ranks worst, and the refusal costs one message. Real support (an
effect-map term in the density, or a θ-move that re-walks the stored path under
the proposed θ) is the follow-up in §9. Reactive-intervention triggers are out
of scope for this PR.

Decided by the maintainer on 2026-10-24, with two extensions found while
deciding. First, a plain `init {}` assignment that references an estimated
parameter (`init { I = I0 }` with `I0` estimated) has the same flat density
under PGAS: `initial_state_logpdf` is zero for a deterministic `init`, and the
stored path's x₀ does not move with the proposed `I0′`; the comment at
`pgas.rs:2884` already says a declared law is "what makes an initial-state
parameter estimable at all". The refusal names the init-law spelling
`I ~ poisson(rate = I0)` as the fix. Second, IF2 builds its effect timeline once
per run from `base_params` (`if2.rs:303-320`) while each particle carries its
own θ, so an estimated parameter in an effect's schedule moves nothing: the fire
time is frozen at its starting value for every particle and every iteration, and
the swarm's value for it performs a cooled random walk, exit 0 (gh#984). The
existing guard `guard_attimesexpr_exact` skips events
(`if iv.kind.is_event()
{ continue; }`), which is how the book's seeding idiom
`events { founders_arrive : add(E, n_seed) at [tau] }` reaches IF2 unrefused. It
is correct under PMMH and `pfilter`, which rebuild the timeline at each
likelihood evaluation's θ (`particle_filter.rs:245`), and the book fits it with
PMMH. The `EffectPlan` is likewise built once per filter run and cannot hold
per-particle times, so under IF2 `build` refuses an estimated parameter in any
effect's schedule, events included, with the same message shape; amounts stay
allowed under IF2 because `step` evaluates them with the particle's own θ. The
inference docs say "use PMMH for seed timing" until gh#986's replay θ-move
lands. Tracked as gh#986 (PGAS) and gh#984 (IF2).

**D4 — At the origin, draining events and `set` read the post-inflow state.**
The spec's stage order for the fixed-step backends (§13.10,
`camdl-language-spec.md:3782-3805`) is inflow events from the start-of-step
snapshot → transitions → residual events (draining `transfer`, `set`) on the
post-transition state → interventions → `balance` → non-negativity → record, and
`step_one` implements it: the inflow phase resolves against the snapshot and is
fused with the transition deltas (`chain_binomial.rs:784-798`), then the
residual phase resolves against the resulting `counts` (`:801-816`). With no
transition stage at the origin, that order reads "draining events see the
post-inflow state". Gillespie's boundary path `apply_events_at`
(`intervention.rs:158-200`) instead resolves both phases against the same
pre-event `int_s` and sums the deltas (`:183-199`), so a draining event there
reads the state before the inflow. The two differ on `init { I = 0 }` with
`events { seed : add(I, 10) at [0]  cull : transfer(fraction = 0.5, from = I, to = R) at [0] }`:
post-inflow leaves I = 5, R = 5; pre-inflow leaves I = 10, R = 0. D4 takes
post-inflow: it is the spec's stage order and chain-binomial's, and the origin
stage is §13.10 without the transition stage, so the same model gives the same
`t_start` row on every backend. Rejected: (b) the pre-inflow reading, matching
today's Gillespie boundary path — it would make the origin the one place where a
draining event cannot see an inflow declared beside it. The Gillespie mid-run
boundary keeps its current reading in this PR; that is a separate pre-existing
inconsistency between the backends, listed in §9.

Decided by the maintainer on 2026-10-24, and generalised: the two-phase rule
(inflows from the snapshot, then drains and `set` on the resulting state, then
interventions in declaration order) is the rule at every instant on every
backend, not only at the origin. It is the only candidate under which the
declaration order of events never matters, a drain cannot overdraw (gh#217's
reason), and `set` means set. With `add(X, 10)` and `set X = 5` at one instant,
chain-binomial gives 5 in either order; Gillespie's boundary path resolves both
against the same snapshot and sums the deltas to 15; the ODE's boundary path
(`apply_boundary_batch_continuous`, `effects.rs:531`) reads every amount from a
frozen snapshot but applies them one at a time, so it gives 15 or 5 depending on
declaration order (gh#985; read from code, not run). Rather than add a fourth
application site (`apply_origin_batch`), the implementation lands one
application function with an optional advance hook between the two phases:
chain-binomial fills the hook with its fused multinomial draw; Gillespie, the
ODE and the origin leave it empty; `apply_events_at` and
`apply_boundary_batch_continuous` become its callers. That function is its own
PR before this one (§8, step 0), and `apply_origin_batch` is then that function
with no hook. Spec §13.10 is extended to state the same rule for the continuous
backends' boundaries and for the origin.

## 6. Tests

Everything lands with the fix; nothing is `#[ignore]`d on `main`. Each fix gets
a mutation check (revert the source change, keep the test, confirm red, record
the red output in the PR).

1. **The oracle**, carried in from the pilot
   (`rust/crates/sim/tests/pf_exact_marginal_oracle.rs`, branch
   `worktree-agent-a358e57f0a23e5735`). Un-ignore
   `off_grid_event_fires_where_the_model_says` and
   `intervention_at_t_start_fires`; delete the two tests that pin today's
   behaviour (`off_grid_event_pf_matches_exact_time_semantics`,
   `intervention_at_t_start_pf_matches_never_firing`). The pilot's `acts_at`
   (`:128-134`) snaps events only and lets an intervention act at its declared
   time; under Grid it must snap both, apply the events of an instant before its
   interventions (the §13.10 order), and let one intervention act once per key
   however many of its declared times share it. The oracle drives the bootstrap
   PF; IF2 and the correlated PF do not run through its driver and need their
   own harness against the same exact marginal. Cases, each for the three
   filters: an off-grid intervention (snapped); an event at `t_start` with an
   observation at `t_start`; `t_start = 0.5`, `dt = 1` with effects at 1 and 2.4
   — not 1.5 and 2.5, which lie on the offset grid where today's absolute key
   and the relative key agree, so that pair passes on today's bug;
   `at [2.3, 2.4]` on one intervention (fires once).
2. **Forward agreement**, zero-rate fixtures in `tests/fixtures/corner_cases/`
   (repository root; `gate_corner_case_baseline.rs:34`) through
   `cross_backend_lifecycle_agreement.rs`. Exact agreement of all three backends
   on every row, including the `t_start` row, for: interventions and events at
   `t_start` and on the grid; a recurring schedule with default `from`; the
   offset-origin model, `from = 0.5`, `dt = 1`, with effects at 1 and 2.4 (for
   the reason in §6.1, times on the offset grid would pass today); an inflow and
   a draining event at `t_start` (D4: I = 5, R = 5 on every backend). For an
   off-grid effect: chain-binomial fires at `t_start + fire_step(t)·dt`, ODE and
   Gillespie at `t`. The three gh#198 tests (`:507`, `:529`, `:555`) are
   rewritten to the per-backend rule: chain-binomial fires once, ODE and
   Gillespie once per declared time; a new test pins that an effect at 2.4 does
   not fire at an earlier landing at 2.0. Each new fixture carries its hand
   computation in its header and its captured baseline (§7).
3. **PGAS and resume.** The producer's trajectory carries a `t_start` effect,
   with `initial_counts` = x₀ and `substeps[0].counts_before` its image.
   `complete_data_loglik` matches the oracle's complete-data value on a fixed
   path with a `t_start` effect and an init law (x₀ and the post-effect state
   are not conflated), and its debug invariant trips when the origin map is
   removed from the producer (the mutation check). With one particle, or with
   ancestor sampling off, a CSMC sweep over a model with a `t_start` effect and
   an init law returns the reference trajectory unchanged: a site that applies
   the origin batch twice, or to `j_ref` a second time, or not at all, fails
   this, and without it the error is silent (the sweep runs and the posterior
   prints). An off-grid event is snapped in the producer. A contrast fork at a
   time with an effect does not fire it twice (`splice_invariant.rs`). The D3
   refusal names the parameter and the effect, once for an action and once for a
   schedule.
4. **Inversions and refusals.** `pf_rejects_off_grid_effect_time`
   (`gh216_cursor_firing.rs:594`) becomes a snap test; the `arrive` step-skip
   unit tests in `schedule.rs` go with the code; `det_grad` and `nuts` refuse a
   model whose only effect is at `t_start` (non-empty `at_origin`, empty
   `times`); the contrast fork, on an arm with an intervention at 2.4 and
   snapshots on the integer grid, forks at 1, not 2.
5. **Key and window.** `fire_step` unit tests: the §2 offset case; the negative
   tie and the `GRID_TOL` case of §5; `t_end` on and off the grid (a shortened
   last step fires at `t_end`); an effect at `t_end + dt/2` does not fire; a
   recurring enumeration's last fire at `t_end + 1e-12` fires at `t_end`; under
   Exact, `at [2.0, 2.0 + 1e-12]` on one intervention fires once.
   `at_day_fire_collision.rs` keeps its four cases under Grid and adds that the
   same schedule is accepted under Exact.

## 7. Identity and goldens

No IR change: no `ir/VERSION` bump, no `*.ir.json` golden moves. Every run
re-keys anyway, because the model digest folds the engine version
`0.1.0+<git hash>` (`runid/src/inputs.rs:163-168, 292`).

The runs that change are the ones listed under the grid key in §5, plus D1 and
D4 at the origin. The PR's `make test` run is the verification that no committed
baseline is among them; what follows is the reading the fixtures' fire times
predict. The golden trajectory gate clears every effect before running
(`gate_trajectory_baseline.rs:459`). The inference baseline gates carry no
effects (`gate_inference_baseline.rs:124`; no schedule in
`gate_pgas_thread_invariance.rs`, `gate_unconditional_pass_ab.rs`). In the
corner-case gate (`gate_corner_case_baseline.rs`) every effect is on the grid
anchored at `t_start = 0`, inside the window, and none is at `t_start`
(`multi_effect_same_time` and `event_intervention_agree` at 5,
`gh70_absorbing_importation` and the culls at 10, `all_lifecycle` importation
from 5, `event_drain_fusion` from 1), except `off_grid_intervention` (2.5):
chain-binomial fires at 3 today (`round(2.5) = 3`) and after (`⌊2.5 + ½⌋ = 3`),
ODE and Gillespie at 2.5 on both, so its three hashes (`:139-141`) stay. If any
baseline does move, that is a finding to explain before re-capturing, and the
re-capture is a reviewed commit of its own. The new fixtures of §6.2 need
captured baselines (`CAMDL_CAPTURE_BASELINE=1`,
`gate_corner_case_baseline.rs:216-225`); they land in their own commit, reviewed
against the hand computation in each fixture's header, so a wrong fixture cannot
pin itself. The He 2010 PF log-likelihood external case moves (the cohort event
now snaps, as in the pomp reference) and should be re-run before release.

## 8. One PR, in this order

0. **Application seam, as its own PR before this one** (gh#985, D4). One
   function applies an effect batch to a state in the §13.10 order, with an
   optional advance hook between the inflow phase and the drain phase:
   `step_one` passes its fused multinomial advance; Gillespie and the ODE pass
   none. `apply_events_at` (`intervention.rs:158`) and
   `apply_boundary_batch_continuous` (`effects.rs:531`) are replaced by calls to
   it. Behaviour-preserving except the coincident inflow-plus-drain and
   inflow-plus-`set` cases on Gillespie and the ODE, each pinned red-then-green
   in `cross_backend_lifecycle_agreement.rs`. Spec §13.10 gains the
   continuous-backend sentence. Lands green on `make test` before step 1 starts.
1. **Foundation.** Move `GRID_TOL` to `schedule.rs`; add `effect_plan.rs`
   (`FireClock`, `Origin`, `fire_step`, `EffectPlan`), unit-tested on the §2
   cases and §6.5. `validate_recurring_dt_collisions` takes `t_start` and the
   clock; `at_day_fire_collision.rs` is rescoped to Grid. Wired by step 2.
2. **Forward paths.** Chain-binomial (Grid), ODE and Gillespie (Exact);
   `validate_schedule` skips the collision check under Exact; `apply_events_at`
   takes the batch; `apply_origin_batch` (D1, D4, no-op when empty) and
   `Origin::Resumed` from `resume.start`; the ODE-gradient refusal reads
   `at_origin ∪ times`. The agreement fixtures, with their captured baselines in
   a separate commit, and the gh#198 rewrites (§6.2).
3. **PF, IF2, correlated PF.** `ProcessModel::fire_clock` and `apply_origin`;
   `ExactInferenceTimeline::build` takes the clock and builds the plan; the
   correlated PF builds it directly; under IF2 the build refuses an estimated
   parameter in any effect's schedule, events included (D3, gh#984); the fit
   pre-flight collision check runs only under Grid. The two callers of
   `guard_exact_offgrid_effect_time` here (`intervention.rs:408`,
   `correlated_pf.rs:654`) drop it; the function itself stays until step 5
   because PGAS still calls it (`pgas.rs:3963`). The oracle and the IF2 and
   correlated-PF harnesses (§6.1).
4. **PGAS, `pgas_init` and the contrast fork.** The three origin sites of §4,
   the `j_ref` pre-state, the debug invariant, the D3 refusal at `run_pgas`
   (effect action, effect schedule, plain `init` assignment); the fork at the
   plan's firing instant; §6.3 and §6.4.
5. **Delete and document.** Remove everything under "Removed" in §4, now
   unreferenced by the runtime, and port the nine test files that still call
   those functions — `gh216_cursor_firing.rs` (`timeline_effects`),
   `gh364_if2_per_particle_initial_state.rs`, `if2.rs`,
   `inference_event_misfire_guard.rs`, `intervention_dt_invariance.rs`,
   `interventions.rs`, `snapshot_projections.rs`, `spatial_density.rs` and
   `statistical_distribution.rs` (`due_effects` / `resolve_fire_steps`, the last
   through its `apply_interventions_at` adapter) — onto `EffectPlan`, plus the
   comments that name them (`at_day_fire_collision.rs:5`,
   `cross_backend_lifecycle_agreement.rs:380`, `gate_licm_ab.rs:374`,
   `gh187_pgas_scheduled_intervention.rs:13`). Rewrite the documents that state
   the old rule: language spec §13.2, §13.7 and §13.10 (the origin stage, D1 and
   D4, and the same two-phase rule for the continuous backends' boundaries) by
   hand, and its contrast sentence at line 4446; IR spec §2.3.1; run spec line
   5367 (the fork is before the firing instant) and §14.3 (lines 5957-5981, the
   `round(t/dt)` key); `docs/dev/scheduling-spine.md:81` and `:97-118`;
   `docs/inference.md:1738-1791` (a forecast from a saved state does not re-fire
   an effect at its origin — the state is post-effect, `Origin::Resumed`).

Each step ends green on `make test-inference`; the PR lands on a green
`make test`.

## 9. Follow-up issues to file after the PR lands

Two were filed on 2026-10-24; the third is not yet filed.

1. **gh#986 — the replay θ-move.** The real fix behind D3: recompute each
   substep's pre-state from x₀, the recorded flows and the effect map under the
   proposed θ, then score the recorded flows against it. Effect amounts are
   deterministic (`effects.rs:630`), so no new density term is needed, and a
   path that overdraws under the proposal scores −∞ and is rejected. It covers
   effect amounts, effect times and plain `init` assignments at once. The red
   test is D3's (the `I0` posterior must move off the prior). gh#984 is the IF2
   half: closed by the D3 refusal, reopened only if per-particle timelines are
   ever built.
2. **gh#985 — coincident events at one instant.** Closed by step 0 of §8.
3. **PGAS applies `balance` to x₀ on one path only.**
   `draw_free_particle_initial_state` rewrites the balance target
   (`pgas.rs:2790-2799`); the reference walk takes `initial_state_draw`
   (`:2161`) as is, and `initial_state_draw` (`compiled_model.rs:2262`) applies
   no balance. A `balance {}` model's reference and free particles start from
   different x₀ under the same θ (read from code, not run).
