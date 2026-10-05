---
date: 2026-10-04
status: draft — two decisions for the maintainer (§6); everything else decided inline
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
`Intervention` and share one schedule grammar (language spec §13.5). The
question "at which instant does this effect act" is answered by three separate
pieces of code, and they disagree in three situations: an effect at exactly
`t_start`, an effect whose time is off the `dt` grid, and any effect when
`t_start` is not a multiple of `dt`. A particle filter (PF) fitted to data
therefore scores a different model from the one `camdl simulate` generates from
the same source, and the three forward backends simulate different models from
one another.

The principle this proposal implements: **simulate and fit define the same
model.** One function computes each effect's firing instant from (declared time,
kind, schedule form, `t_start`, `dt`, run origin), and every path — forward
chain-binomial, forward ODE and Gillespie, the ODE value path used by
`mh`/`nl-*`, Exact inference (bootstrap PF, IF2, correlated PF), and PGAS —
derives its firing key from that one answer.

## 2. Three authorities, three answers

The scheduling-spine work (2026-06-05, 2026-06-07, 2026-06-11, 2026-06-13)
unified _where the integrator stops_ and _how a due batch is applied_; it left
_which batch is due_ to two keys and explicitly fenced chain-binomial's
`round(t/dt)` key out as "correct because Snap stays on the grid" (2026-06-13
§3.4). That holds for observation times; it does not hold for effect times.

| Path                                                                                   | Firing key (verified by reading)                                                                                                                                                                                             | Effect at `t_start`                                                                                 | Off-grid effect                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| chain-binomial forward (Snap)                                                          | `due_effects` (`effects.rs:314`) tests `time_to_step(t_end, dt)` = `(t/dt).round()` (`time.rs:132`) against `resolve_fire_steps` (`compiled_model.rs:1345`), called at each step end `t_grid + dt` (`chain_binomial.rs:428`) | never fires: no step ends at `t_start`                                                              | snapped to the nearest absolute step                                                                                                                                                                                                         |
| PGAS (Snap; the only PGAS the CLI accepts, `cli/src/fit/methods.rs:556`)               | same `due_effects` key via `fill_producer_batch` (`pgas.rs:2105`, called at `pgas.rs:2194`, `:3116`)                                                                                                                         | never fires (inferred: same key, substep 0 ends at `t_start + dt`)                                  | snapped (inferred)                                                                                                                                                                                                                           |
| ODE forward and ODE value path (`run_ode`, `ode.rs:813`; `inference/ode_loglik.rs:52`) | lands on `EffectTimes::from_model` (`boundary_times.rs:129` → `all_intervention_times`, `intervention.rs:209`), then `due_effects` at the landing (`ode.rs:1005`)                                                            | fires, but the `t_start` row is recorded first (`ode.rs:961`), so it shows the **pre-effect** state | fires at its exact time; a second effect sharing its absolute step fires early with it, and `arrive` skips its own landing (`schedule.rs:597`, gh#198)                                                                                       |
| Gillespie forward                                                                      | as ODE (`gillespie.rs:147`, `:241`)                                                                                                                                                                                          | as ODE                                                                                              | as ODE                                                                                                                                                                                                                                       |
| PF, IF2, PMMH (Exact)                                                                  | cursor over `timeline_effects` (`intervention.rs:264`); each window's cursor starts at `Schedule::effect_idx_at(t)` = `partition_point(e <= t + EFFECT_EPS)` (`schedule.rs:514`; `particle_filter.rs:349`, `if2.rs:547`)     | dropped: counted as already fired                                                                   | events: exact time; interventions: refused by `guard_exact_offgrid_effect_time` (`intervention.rs:302`), which exempts events (`:315`)                                                                                                       |
| correlated PF (Exact)                                                                  | as PF (`correlated_pf.rs:654-657`, `:911`)                                                                                                                                                                                   | dropped (inferred: same `effect_idx_at(t_start)`)                                                   | interventions refused; an event inside a window whose start is off-grid trips the loud `walked > k` noise-layout refusal (`correlated_pf.rs:744`; inferred for events — the message names interventions, but events re-anchor the same walk) |
| ODE gradient (`nuts`)                                                                  | refuses any non-empty effect batch (`ode.rs:1296`)                                                                                                                                                                           | refused                                                                                             | refused                                                                                                                                                                                                                                      |

Measured on `main` (15cf347c), deterministic (`beta = gamma = 0`, `dt = 1`). Two
interventions moving half of S to R at t = 0 and t = 1: S at t = 1 is 50 under
chain-binomial and 25 under ODE and Gillespie. Events `add(I, 5)` at `[0]` and
`[1]`: I = 5 under chain-binomial, 10 under ODE and Gillespie. The
exact-marginal PF oracle (gh#201; the pilot file
`rust/crates/sim/tests/pf_exact_marginal_oracle.rs` on branch
`worktree-agent-a358e57f0a23e5735`) computes `p(y | θ)` exactly by the forward
algorithm on an SIR with N₀ ≈ 6: with an intervention at `t_start` the filter is
biased by +0.85 nats (z = +363) and is unbiased against the model _without_ the
intervention; with `add(I, 2)` at t = 2.4 the filter is biased by −0.355 nats (z
= −190) against the snapped model and unbiased against exact-time firing. In
both cases the filter is a correct estimator of a different model.

Two further divergences found while writing this, measured with the installed
`camdl 0.1.0+b276d7e5` (the firing code is unchanged between b276d7e5 and
15cf347c: `git log b276d7e5..15cf347c` touches none of the firing sites):

- **Recurring schedules lose their first fire under chain-binomial.** A block
  schedule's `from` defaults to `t_start` (spec line 3457), and
  `intervention_fire_times` includes `from` itself (`intervention.rs:98`).
  `add(I, 1) { every = 2 'days }` over `[0, 6]`: chain-binomial ends at I = 3,
  ODE and Gillespie at I = 4. The spec's own example `routine_vacc` (lines
  3378-3382, `from = 0 'days`) loses its first round under chain-binomial when
  the run starts at 0, and so does every `date_range(origin, …)` list, such as
  the `booster` in `ocaml/golden/sirv_anchored_calendar.camdl:47`.
- **The absolute key misfires effects strictly inside the run when `t_start` is
  not a multiple of `dt`** (question (iii) of the brief). With `from = 0.5`,
  `dt = 1`, events at 1, 2, 2.4 and 3.5: chain-binomial drops the event at 1
  entirely (its step, `round(1) = 1`, is matched only by
  `t_end = 0.5 = t_start`, which no step ends at), fires the events at 2 and 2.4
  at t = 1.5 (half a step and 0.9 days early), and fires 3.5 correctly. ODE and
  Gillespie fire all four at their declared times. The cause is that the step
  index is computed on the grid anchored at 0, not the grid the run steps on
  (`t_start + s·dt`, `schedule.rs:473`). Ties at `.5` add a sign asymmetry:
  `f64::round` rounds half away from zero, so a negative `t_start` breaks ties
  the other way.

## 3. The spec defines firing for one schedule form only

Language spec §13.2 (lines 3444-3458) gives the inline form as
`NAME : ACTION at [TIME, ...]     # times in model time_unit` (line 3449) and
the block form's `from` as "start of recurring (default: t_start)" (line 3457).
It says nothing about off-grid times or about a fire at `t_start`.

The snap rule is in §13.7, "The `at_day` Schedule" (lines 3601-3623), which
opens "For `add` events and interventions that recur on a specific day within
each period" (line 3603) and states (lines 3613-3615): "Fire times are
`at_day + k * period` for the smallest `k` where `target >= t_start`. The engine
fires on the single timestep where `|t - target| < 0.5 * dt`". It applies to
`every … at_day` only. Its enumeration rule `target >= t_start` admits a fire at
exactly `t_start`, which chain-binomial then drops.

§13.10 (lines 3782-3806) fixes the order of effects "when several effects land
at the same time step" and says "recorded output is **post-effect**" (line 3802)
— which ODE and Gillespie violate at `t_start`.

The IR spec contradicts itself in §2.3.1: for chain-binomial "the runtime must
truncate the current step to the intervention time" (`compartmental-ir-spec.md`
line 179), and for discrete time "apply at the nearest step boundary (rounding
down)" (line 189) — "nearest" and "rounding down" are different rules, and the
code implements neither literally (it rounds to nearest, half away from zero).

No spec text defines firing for `at [...]` off the grid, for a fire at
`t_start`, or for a grid not anchored at 0. The implementation PR fixes §13.2,
§13.7 and IR §2.3.1 to state the rule below (by hand — `dprint` must not run on
the language spec).

## 4. The design: an `EffectPlan` built once per run

Types first. The plan replaces `resolve_fire_steps`, `due_effects`,
`timeline_effects`, `all_intervention_times` and
`guard_exact_offgrid_effect_time` as the answer to "which effects fire, and at
which boundary". The application seams stay as they are (`step_one` through the
`effects` module, `lifecycle::apply_post_advance`,
`effects::apply_boundary_batch_continuous`, `intervention::apply_events_at`);
the plan only decides _what is due_. This is the "reach for the existing seam"
rule: the boundary authority (`Schedule`, gh#233) already owns where to stop, so
the plan feeds it `EffectTimes` instead of adding a second stopping rule.

```rust
// rust/crates/sim/src/effect_plan.rs — the spine owns time.

/// Where a run's starting state came from. An `init {}` state has not seen the
/// effects whose fire step is 0; a resumed state (contrast fork, gh#322;
/// `--init-state` forecast, gh#641) already has.
pub enum Origin { Init, Resumed }

/// Every scheduled effect fire of one run, resolved to the boundary the model
/// defines. Built once per run (once per filter call under inference).
pub struct EffectPlan {
    t_start: f64,
    dt: f64,
    /// Applied to the starting state before the first substep and before the
    /// t_start row is recorded or scored. Empty when origin is Resumed.
    at_origin: Vec<usize>,          // iv indices, declaration order
    /// Boundaries k ≥ 1 (time t_start + k·dt), ascending, index-aligned.
    steps: Vec<u64>,
    batches: Vec<Vec<usize>>,       // one entry per intervention per k (gh#198)
}

impl EffectPlan {
    pub fn build(model: &CompiledModel, params: &[f64], t_start: f64,
                 dt: f64, origin: Origin) -> Result<Self, SimError>;
    pub fn at_origin(&self) -> &[usize];
    /// Snap consumers: the batch due at the end of substep index s (k = s + 1).
    /// Integer lookup — no float rounding of a step end.
    pub fn batch_ending_substep(&self, s: u64) -> &[usize];
    /// Exact consumers: landing times t_start + k·dt for the Schedule,
    /// index-aligned with `batch(effect_idx)`.
    pub fn effect_times(&self) -> EffectTimes;
    pub fn batch(&self, effect_idx: usize) -> &[usize];
}

/// The one key: k = ⌊(t − t_start)/dt + ½ + GRID_TOL⌋. None when t < t_start
/// or t > t_end (outside the run). GRID_TOL is the existing on-grid tolerance
/// (intervention.rs:234), moved to schedule.rs with the other time tolerances.
fn fire_step(t: f64, t_start: f64, dt: f64, t_end: f64) -> Option<u64>;
```

This is the representation under recommendation D2 = snap (§6). Under D2 = exact
the plan carries times rather than step indices, and the Snap consumers must
clip a substep to land on them (§6).

Where it plugs in. Every row replaces a firing-key site; none adds a stopping
rule.

| Path                       | Build the plan                                                                 | Origin batch                                                               | Due batch                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| chain-binomial forward     | replaces `chain_binomial.rs:285`; `Origin::Resumed` when `resume.start` is set | after `initial_state_draw` (`:276`), before the initial row (`:356`)       | `batch_ending_substep(s)` replaces `due_effects` at `:428`                                                  |
| ODE forward and value path | replaces `ode.rs:905`, `:1192`                                                 | after `initial_state_mean` (`ode.rs:841`), before the initial row (`:961`) | `Schedule::exact_forward(…, plan.effect_times())`; the `arrive` closure applies `plan.batch(idx)` (`:1005`) |
| Gillespie forward          | replaces `gillespie.rs:147`                                                    | after `initial_state_draw` (`:121`), before the initial row (`:180`)       | as ODE (`:241`, `:310`)                                                                                     |
| PF, IF2, PMMH              | inside `ExactInferenceTimeline::build` (`intervention.rs:394`)                 | after `initial_state_draw` (`particle_filter.rs:155`, `if2.rs:472`)        | `effect_times` contain no `t_start`, so `effect_idx_at(t_start)` is 0 and needs no special case             |
| correlated PF              | replaces `correlated_pf.rs:654-657`                                            | after `initial_state_draw_correlated`                                      | as PF                                                                                                       |
| PGAS                       | replaces `pgas.rs:2174`, `:2864`, `:3963-3966`, `pgas_init.rs:203`             | see below                                                                  | `fill_producer_batch` (`pgas.rs:2097`) reads `batch_ending_substep(s)`                                      |

PGAS needs care (a high-risk surface). The trajectory's `initial_counts` must
stay x₀, the state the declared `init {}` law scores (`initial_state_logpdf`,
`pgas.rs:1835`). The origin batch is a deterministic map applied to x₀ wherever
the substep-0 pre-state is materialised — the reference producer, the
conditional SMC free particles (`draw_free_particle_initial_state`),
`pgas_init.rs`, and the complete-data density walk. The implementer must find
every such site; the PGAS cell of the test matrix (§7) is what proves none was
missed.

Removed once every consumer is wired: `effects::due_effects`,
`CompiledModel::resolve_fire_steps`, `time::fire_times_to_steps`,
`intervention::timeline_effects`, `intervention::all_intervention_times`,
`guard_exact_offgrid_effect_time` (snapped times are on the grid by
construction), and the step-skip loop in `Schedule::arrive`
(`schedule.rs:592-600`) with `effect_maps_to_step` (distinct boundaries are now
distinct steps). `validate_recurring_dt_collisions` keeps its purpose and moves
to `fire_step`.

## 5. Semantics decided inline

**The key is relative to `t_start` (question (iii)).** This is a bug, not a
choice: the absolute key drops or moves effects strictly inside the run (§2),
and the Exact paths already treat "on the grid" as relative to `t_start`
(`intervention.rs:319`). Ties go to the later boundary (`+ ½` then floor), which
equals `f64::round` for the non-negative offsets that can fire, so no model
whose `t_start` is a multiple of `dt` changes. A resumed run keys relative to
its own origin; a contrast fork resumes at an output time, which chain-binomial
emits only on its own grid, so the two grids differ by an integer number of
steps and every fire keeps its boundary (inferred; the splice-invariance test
`splice_invariant.rs` is the check).

**Order at the origin.** `init {}` (laws included) → inflow events read x₀ →
draining events and `set` → interventions, in declaration order → `balance` →
the non-negativity check → the `t_start` row is recorded and any observation at
`t_start` is scored. This is §13.10 with the transition stage removed (an
instant has no transitions) and is what Gillespie already does at a boundary
(`gillespie.rs:239-245`, through `apply_events_at` and `apply_post_advance`), so
the origin batch reuses those two functions. A parameter-dependent amount (an
estimated `fraction`) is evaluated at `t = t_start` like any other effect.

**Effects outside `[t_start, t_end]` do not fire**, as today on every path. A
forecast or fork origin legitimately starts after past effects.

**The gh#198 merge applies uniformly.** Two fires of one intervention that map
to one boundary fire once (`at [2.3, 2.4]` at `dt = 1`). That is the existing
deliberate contract (`at_day_fire_collision.rs:147`); today the PF breaks it by
firing both (inferred: `timeline_effects` groups boundaries by exact time
equality, `intervention.rs:279-287`). Distinct interventions at one boundary
fire in declaration order.

**Unchanged:** reactive interventions (fired by their agenda, not a schedule);
the refusal of parametric `at [<param>]` under Exact (`guard_attimesexpr_exact`,
`intervention.rs:356`; per-particle times are a separate problem); the
ODE-gradient refusal.

## 6. Decisions for the maintainer

**D1 — An effect at `t_start`.** (a) Apply it to the initial state on every
path, in the order of §5. Consequence: chain-binomial, PF, IF2, correlated PF
and PGAS start firing effects they silently drop today, including the first
round of every recurring schedule whose `from` defaults to `t_start`; ODE and
Gillespie change only their `t_start` row (pre-effect → post-effect). (b) Refuse
at compile time, pointing at `init {}`. Consequence: every `every = …` block
without an explicit `from`, every `at_day` schedule whose first target is
`t_start`, and every `date_range(origin, …)` list becomes a compile error
(including the `sirv_anchored_calendar` golden), and the refusal cannot see a
runtime `t_start` override. **Recommendation: (a). Solid** — `at [t_start]`
plainly means it, the spec's enumeration (`target >= t_start`, `from` defaulting
to `t_start`) already produces such fires, and ODE and Gillespie already apply
them.

**D2 — An effect off the `dt` grid.** (a) **Snap everywhere**: every path fires
at `t_start + k·dt` with k from `fire_step`. Chain-binomial and PGAS keep their
current behaviour for grids anchored at a multiple of `dt`; PF, IF2 and the
correlated PF fire snapped events and _accept_ off-grid interventions they
refuse today; ODE and Gillespie move off-grid effects to the grid (at most
`dt/2`), so an effect's instant depends on `dt` on every backend, and
cross-backend agreement holds at a common `dt` (Gillespie's grid is
`model.simulation.dt`, `gillespie.rs:142`). This is the only firing rule the
spec states (§13.7) and the convention of the pomp reference the He et al.
(2010) external cases are validated against
(`tests/external/cases/he2010_forward/reference/reference.R:109`:
`fabs(t - floor(t) - 251.0/365.0) < 0.5*dt`). (b) **Exact everywhere**: every
path fires at the declared time. ODE, Gillespie and the PF are already there for
events; chain-binomial forward must clip a substep at each off-grid effect,
which changes its random-number stream for every such model (He 2010's cohort
pulse at `258 + k·365.25` among them); PGAS's uniform Snap grid must gain effect
landings in `pgas.rs` (high-risk), or refuse off-grid effects until it does; the
correlated PF must size its noise blocks from the walk or keep refusing; the
gh#198 merge is undone. Effect semantics become independent of `dt`. (c)
**Refuse off-grid events under Exact inference**, removing the exemption at
`intervention.rs:315`. Smallest diff, but it leaves forward chain-binomial
disagreeing with ODE and Gillespie, and the PF and IF2 would refuse the He 2010
model (its cohort event is off the daily grid in three years of four), with no
snap PF to fall back on (`methods.rs:526-531`). **Recommendation: (a). Leaning**
— it makes all paths agree with the smallest change to the default backend and
the default Bayesian method, and removes a refusal rather than adding one.
Choose (b) if `dt`-independent effect timing matters more than the cost of
touching chain-binomial stepping and PGAS.

## 7. Test plan

All tests land in the implementation PR with the fix; none is `#[ignore]`d on
`main`. Each fix gets a mutation check: revert the source change, keep the test,
confirm red, and record the red output in the PR.

1. **The oracle.** Carry `pf_exact_marginal_oracle.rs` in. Un-ignore
   `off_grid_event_fires_where_the_model_says` and
   `intervention_at_t_start_fires`. Delete the two localisation tests that pin
   the current behaviour (`off_grid_event_pf_matches_exact_time_semantics`,
   `intervention_at_t_start_pf_matches_never_firing`). Add: an off-grid
   _intervention_ under the PF (refused today); an event at `t_start` with an
   observation at `t_start` (scores post-effect); a `t_start = 0.5`, `dt = 1`
   configuration with effects at 1 and 2.4; and `at [2.3, 2.4]` on one
   intervention (fires once).
2. **Forward backend agreement.** New zero-rate fixtures in
   `tests/fixtures/corner_cases/` run through the Tier B battery of
   `cross_backend_lifecycle_agreement.rs:255`: interventions and events at
   `t_start` and on the grid; a recurring schedule with default `from`; an
   off-grid effect; the offset-origin model of §2. Chain-binomial, ODE and
   Gillespie must agree on every row, including the `t_start` row.
3. **The rest of the matrix.** IF2 and the correlated PF on the oracle
   configurations (both route through the same plan but have their own cursor
   sites, `if2.rs:547`, `correlated_pf.rs:911`); PGAS: the producer's trajectory
   carries the effect at substep 0, and `complete_data_loglik` equals the
   oracle's complete-data value on a fixed path with a `t_start` effect and an
   init law, so x₀ and the post-effect state are not conflated; resume: a
   contrast fork at a time with an effect does not fire it twice
   (`splice_invariant.rs`).
4. **Inversions.** `gh216_cursor_firing.rs:594`
   (`pf_rejects_off_grid_effect_time`) becomes a test that the PF fires the
   snapped time; the `arrive` step-skip unit tests go with the code they test.

## 8. Identity and golden impact

No IR change: no `ir/VERSION` bump and no `*.ir.json` golden moves (both sets
carry fire times as declared; the plan is runtime-only). Run identity re-keys
anyway, because every run's model digest folds the engine version string
`0.1.0+<git hash>` (`runid/src/inputs.rs:163-168, 292`), so no cached run is
served under the old semantics.

The golden trajectory baselines do not move: that gate clears every intervention
and event before simulating (`gate_trajectory_baseline.rs:459`). The corner-case
gate runs with all effects on: under D2 = snap the `off_grid_intervention` ODE
and Gillespie hashes move (the 2.5 cull now fires at 3, matching
chain-binomial), `gate_corner_case_baseline.rs:139,141`; under D2 = exact the
chain-binomial hash moves instead (`:140`). These re-captures are human-loop
changes and are the subject of their own commit. The external He 2010 cases are
not part of `make test`: under D2 = snap the forward case is unchanged
(chain-binomial, `t_start = 0`) and the PF log-likelihood case moves toward the
pomp reference, which snaps; it should be re-run before release.

## 9. One PR, in this order

1. Move `GRID_TOL` to `schedule.rs`; add `effect_plan.rs` with `fire_step` and
   `EffectPlan`, unit-tested against the measured cases of §2. Foundation for
   this PR; wired by step 2.
2. Wire the forward paths (chain-binomial, ODE, Gillespie), including the origin
   batch and `Origin::Resumed`; add the forward agreement fixtures (§7.2).
   Re-capture the corner-case hashes in a separate commit.
3. Wire PF, IF2 and the correlated PF; delete `guard_exact_offgrid_effect_time`;
   carry in the oracle (§7.1).
4. Wire PGAS and `pgas_init`; add the PGAS and resume tests (§7.3).
5. Delete the replaced functions and the `arrive` step skip; fix spec §13.2,
   §13.7 and IR §2.3.1.

Each step ends on a green `make test-inference`; the PR lands on a green
`make test`.
