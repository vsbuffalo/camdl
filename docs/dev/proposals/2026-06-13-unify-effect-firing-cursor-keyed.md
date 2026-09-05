---
date: 2026-06-13
status: implemented — events cursor-keyed (PF/IF2/correlated-PF); PGAS keeps the loud events-under-Exact floor (follow-up)
area: simulation timing spine / effect firing / inference correctness
issue: gh#216 (events arm)
supersedes-scope: the events arm of 2026-06-11-spine-effect-firing-consolidation.md (scheduled interventions already landed)
---

# Collapse the effect-firing fork: events become cursor-keyed like scheduled interventions

## 1. The problem (one sentence)

Always-active **events** fire at the **wrong time** under `StepPolicy::Exact`
inference (bootstrap PF / IF2 / correlated-PF) when an observation lands off the
`dt` grid — silently — because they are the one effect kind still keyed on
`round(t_end/dt)` instead of the timeline cursor.

Reproduction (RED,
`gh216_cursor_firing.rs::pf_event_firing_invariant_to_offgrid_obs_stream`, event
at day 4, obs `[3.5, 4.0, 8.0]`, `dt=1`):

```
M trajectory  left:  [10, 20, 20]    # early fire at 3.5 (round(3.5)=4) + double fire at 4.0
              right: [0, 10, 10]     # correct: fire once, at 4.0
```

This is `gh#216`'s exact signature. The `gh#216` spine fix (`b982f30b`, on this
branch as `ae346eb5`) consolidated **scheduled interventions** onto the timeline
cursor and **deliberately scoped events out** (`effects.rs:284-291`:
_"always-active events keep the grid_dt firing key … out of scope for the gh#216
firing change"_). That left a `Capabilities`-style matrix gap: loud on one cell,
silent-wrong on three.

## 2. Why this is a spine-correctness issue, not a feature bug

The timing spine (`schedule.rs`) is the single substrate every forward backend
and every inference algorithm steps over. A firing-time error here is not one
backend's bug — it is wrong for _all_ inference of any model that has an
`events {}` block and observations off the `dt` grid. **Multi-cadence makes this
the norm:** the union observation axis (this PR) is exactly what produces
off-grid obs, so a polio/measles model with cohort-entry or importation events
fit on AFP+ES data hits it.

## 3. The current seams (types + every firing site)

### 3.1 The type (already unified — no change proposed)

`rust/crates/ir/src/intervention.rs`:

```rust
struct Intervention {                 // ONE struct; events and interventions both lower to it
    name, base_name,
    schedule: InterventionSchedule,   // AtTimes | AtTimesExpr | Recurring
    actions:  Vec<Action>,            // FractionTransfer | AbsoluteTransfer | Set | Add
    kind:     InterventionKind,       // Scenario | Event   (gh#107; extends to Reactive, gh#204)
}
```

The IR type is the right unification (spec §13.5: events = interventions minus
toggleability). **This proposal changes no IR/schema/golden.**

### 3.2 The spine types (`schedule.rs`)

```rust
struct Schedule {                     // immutable, Sync; one per run, shared by all particles
    dt, t_end, grid, policy: StepPolicy,   // Snap | Exact
    output_times: Vec<f64>,           // trajectory snapshots
    effect_times: Vec<f64>,           // intervention/event boundaries   ← the seam
    obs_times:    Vec<f64>,           // inference likelihood/score points
}
struct Cursor { output_idx, effect_idx, obs_idx }   // Copy; per-particle walk
enum  StopReason { Output, ScheduledEffect, Observation, End }
struct EffectBatch { event_idx: SmallVec<usize>, intervention_idx: SmallVec<usize> }
```

`Schedule::substep` (the single time→step mapping): under `Exact` it clips to
`min(t_end, next_output, next_effect, next_obs) − t`; under `Snap` it never
clips to an effect. **So whether an effect time is in `effect_times` decides
whether the integrator LANDS on it** (Exact) — that is the entire mechanism.

The CRN invariant (module header, lines 40-47): `Schedule` is immutable,
`Cursor` is `Copy`, `next_*` are pure, so N particles walk an identical boundary
sequence. **Adding or removing an `effect_times` entry changes that sequence** →
changes the substep count between obs → changes RNG draw order. This is the
byte-identity hazard (§5).

### 3.3 The firing fork (three sites, `is_event()`)

| site                    | file:line                                       | what it does                                                                                                    |
| ----------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| due-ness split          | `effects.rs:267-276` (`due_effects`)            | one `round(t_end/grid_dt)` lookup → firing iv lands in `event_idx` if `is_event`, else `intervention_idx`       |
| event-only round path   | `effects.rs:292-306` (`due_events`)             | the Exact-inference event path: events on the `grid_dt` round key (the **buggy** path)                          |
| cursor-timeline builder | `intervention.rs:243-260` (`scheduled_effects`) | builds `effect_times` for the inference `Schedule` — **`if iv.kind.is_event() { continue }`** (events excluded) |

### 3.4 Forward vs inference — the asymmetry that _is_ the bug

- **Forward** (`intervention.rs:197` `all_intervention_times`): iterates _all_
  interventions (no `is_event` filter) → events ARE in `effect_times`. ode +
  gillespie (Exact) clip-and-land on event times → events fire **cursor-keyed**,
  correctly. chain_binomial (Snap) fires via on-grid `round(t/dt)` — correct
  because Snap stays on the grid.
- **Inference** (`particle_filter.rs:179`, `if2.rs:264`, `correlated_pf.rs:283`,
  `pgas.rs:396`): `effect_times = scheduled.times` (scheduled interventions
  ONLY). Events excluded → fire via `due_events` round-key inside `step_one`, on
  the **obs-anchored, off-grid** Exact grid → misfire.

**The forward Exact backends already do the right thing (cursor-keyed events).
The inference path is the outlier.** B brings inference into line with
ode/gillespie.

## 4. Backend × algorithm clash map (flag before touching)

| path                                      | builds `effect_times` from                | events fire via                                      | apply path                                          | clash to watch                                                             |
| ----------------------------------------- | ----------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------- |
| chain_binomial (Snap, fwd)                | `all_intervention_times` (incl. events)   | `due_effects` round, on-grid                         | `step_one` PROPOSE(event)/INTERVENE(iv)             | none — Snap is on-grid                                                     |
| ode (Exact, fwd)                          | `all_intervention_times` (incl. events)   | cursor land, `due_effects` ±1e-10                    | **`apply_boundary_batch_continuous`** (f64, no RNG) | ODE has its OWN apply path; B must keep it cursor-correct                  |
| gillespie (Exact, fwd)                    | `all_intervention_times` (incl. events)   | cursor land + `apply_events_at`                      | discrete apply + **full propensity recompute**      | propensities recomputed post-effect; B must not change that                |
| **PF / IF2 / correlated-PF (Exact, inf)** | `scheduled_effects` (**events excluded**) | **`due_events` round → MISFIRE**                     | `step_one` (clipped to obs)                         | **the bug; B's target**                                                    |
| PGAS (Exact, inf)                         | `scheduled_effects` (events excluded)     | **hard-rejects events under Exact** (`pgas.rs:1523`) | n/a                                                 | B should let PGAS _accept_ events (remove the rejection) once cursor-keyed |
| PMMH                                      | wraps a PF flavor                         | inherits                                             | inherits                                            | inherits PF's fix automatically                                            |

Three things the design must not break:

1. **ODE's continuous apply** (`apply_boundary_batch_continuous`) — separate
   from the discrete path; events there already land cursor-keyed, must stay so.
2. **Gillespie's post-effect propensity recompute** — a correctness requirement
   (CLAUDE.md / spec §2.3.1); untouched by B.
3. **PGAS's events-under-Exact rejection** — currently a guard; B _removes the
   need for it_ (events become correct under Exact), so PGAS should drop the
   rejection and gain events support. (Verify the PGAS drift-free ancestor walk
   re-anchors at effect landings — `pgas.rs` `build_substep_grid` already
   records `effect_at_substep` for scheduled interventions; events ride the same
   path.)

## 5. The design — before / after

**Core move:** stop excluding events from the inference cursor timeline. One
firing-TIME mechanism (cursor-keyed `effect_times`) for every effect kind, on
every Exact path. Keep the genuine _lifecycle-stage_ distinction (events fuse at
PROPOSE, interventions apply at INTERVENE) — that is a real semantic difference,
not a timing one.

### before

```rust
// intervention.rs — inference timeline EXCLUDES events
pub fn scheduled_effects(model, params) -> ScheduledEffects {
    for (iv_idx, iv) in model.interventions.iter().enumerate() {
        if iv.kind.is_event() { continue; }          // ← events dropped
        for &t in &fire_times[iv_idx] { pairs.push((t, iv_idx)); }
    }
    ...
}

// effects.rs — a SECOND, round-keyed firing path for events
pub fn due_events(model, fire_steps, t_end, grid_dt, out) {
    let step = time_to_step(t_end, grid_dt);          // ← off-grid t_end → wrong step
    for (i, iv) in ... { if iv.kind.is_event() && fire_steps[i].contains(&step) { out.event_idx.push(i); } }
}

// particle_filter.rs / if2.rs / correlated_pf.rs (Exact inference)
let scheduled = scheduled_effects(model, params);     // scheduled interventions only
let schedule = Schedule::new(dt, end, dt, Exact, vec![], scheduled.times).with_obs(obs_times);
// ... step_one then calls due_events(...) internally for events  ← the misfire
```

### after

```rust
// intervention.rs — the inference timeline carries ALL effects (events + interventions),
// exactly as all_intervention_times does for the forward Exact backends. Rename to
// reflect the unified scope; the per-boundary batch still splits event vs intervention
// for the lifecycle stage.
pub fn timeline_effects(model, params) -> ScheduledEffects {   // (was scheduled_effects)
    for (iv_idx, _iv) in model.interventions.iter().enumerate() {
        for &t in &fire_times[iv_idx] { pairs.push((t, iv_idx)); }   // NO is_event filter
    }
    ...   // batches still tag each iv_idx as event vs intervention (kind preserved)
}

// effects.rs — due_events DELETED. due_effects (cursor batch, splits by kind) is the
// ONE due-ness function. The Exact-inference caller populates the batch from the
// timeline cursor at each effect boundary (events land at their own time → no round()).

// particle_filter.rs / if2.rs / correlated_pf.rs (Exact inference)
let effects  = timeline_effects(model, params);        // events + interventions
let schedule = Schedule::new(dt, end, dt, Exact, vec![], effects.times).with_obs(obs_times);
// substep clips to the next effect boundary → integrator LANDS on each event time →
// step_one applies the caller's batch (events at PROPOSE, interventions at INTERVENE).
```

After B, `is_event()` survives only in two legitimate places — the **lifecycle
split** (`event_idx` fuses with the kernel draw at PROPOSE; `intervention_idx`
applies post-advance at INTERVENE) and **scenario activation** (events can't be
toggled). It no longer appears in any firing-TIME computation. The bug class —
"events routed through a different firing-time path that someone forgot to fix"
— becomes unrepresentable: there is one path.

### How "reactive" (gh#204) drops in later

This design leaves the IR `kind` enum and the seam untouched-but-cleaner.
Reactive's distinction is _fire-source_ (a runtime observation-trigger), not
firing-time-mechanism. When it lands it adds a
`FireSource::{ Scheduled,
Reactive(Trigger) }` distinction _above_ the unified
firing mechanism — the trigger computes the next fire time, which then flows
through the same cursor-keyed `effect_times` machinery. No part of B forecloses
that; it makes it easier (one firing path to feed).

## 6. The hard part — byte-identity, and why it points at event-keyed RNG

Adding event times to the inference `effect_times` changes the `Schedule`
boundary sequence. The risk is **on-grid** byte-identity (the inference goldens
/ `he2010_pfilter_loglik{,_sparse}` camdl-vs-pomp values, which must not move):

- **State** is byte-identical on-grid by construction — an event at an integer
  time fires once at that integer either way (round and cursor agree on-grid),
  with the same pre-draw snapshot, the same delta.
- **RNG draw order** is the exposure. Today's coupling is paired-seed CRN: draw
  order is tied to _consumption order_, which is tied to the substep structure.
  Moving event firing from "inside `step_one` via `due_events`" to "caller batch
  via the cursor" must reproduce the exact PROPOSE-stage fusion point and
  consume the same draws in the same order. The gh#216 fix proved this is
  achievable for scheduled interventions (on-grid bit-identity test green); B
  must prove the same for the event/PROPOSE path.

This is exactly where your **event-keyed RNG / separate-streams** white-paper
idea earns its place. Under the current model, _any_ change to the boundary
structure is a byte-identity minefield — B, reactive, and future spine work all
pay this tax. Event-keyed RNG (a stream keyed by `(event identity, substep)`
rather than a single sequential consumption order) **decouples draw order from
substep structure**: adding/moving an effect boundary no longer shifts any other
draw, because each draw's stream is addressed by identity, not position. That
would make B (and reactive, and any future re-tiling) byte-identity-robust _by
construction_.

**Recommendation on sequencing:** do B under the current CRN with a careful
on-grid byte-identity proof (it is achievable — ode/gillespie already register
events as boundaries; the inference path is catching up). Treat event-keyed RNG
as a **separate, larger proposal** that B motivates but does not require. If the
on-grid byte-identity proof for B turns out to be intractable without it (e.g.
the PROPOSE-fusion draw order genuinely cannot be preserved when events become
cursor boundaries), that is the signal to pull event-keyed RNG forward and do it
first. I'd want to write the red→green and _measure_ the golden drift before
deciding — flagging now so it's a conscious fork, not a surprise.

## 7. Test plan (red → green, per cell)

1. **RED already exists:** `pf_event_firing_invariant_to_offgrid_obs_stream`
   (`[10,20,20]` ≠ `[0,10,10]`). Extend `gh216_cursor_firing.rs` to run the full
   firing-invariance suite with `InterventionKind::Event` (it currently runs
   only `Scenario`) across PF / IF2 / correlated-PF.
2. **On-grid byte-identity:** the existing inference goldens +
   `he2010_pfilter_
   loglik{,_sparse}` must show DRIFT 0 and unmoved
   camdl-vs-pomp loglik. This is the gate for "B didn't disturb RNG order
   on-grid."
3. **PGAS gains events:** remove the `pgas.rs:1523` events-under-Exact
   rejection; add a PGAS event-firing-invariance test (value ↔ csmc_as agreement
   with an event present). If the ancestor walk can't yet re-anchor at event
   landings, keep the rejection as the explicit capability floor and say so (no
   silent gap).
4. **Forward unchanged:** ode/gillespie/chain_binomial golden trajectories
   byte-identical (they already register events; B doesn't touch their apply).
5. **Multi-cadence + events:** a fixture with an `events {}` block fit on the
   off-grid union axis recovers params (the integration-level proof that the
   feature this PR adds works with events).

## 8. Scope / non-goals

- **No IR/schema/golden-IR change.** The `Intervention`/`InterventionKind` type
  is correct; B is a runtime firing-path change only.
- **No new trait for events-vs-interventions.** Two kinds differing in one
  activation bit do not warrant a trait; the enum is right-sized. (Revisit only
  when reactive adds the fire-source axis.)
- **Event-keyed RNG is out of scope** — a separate proposal that B motivates.
- **Reactive (gh#204) is out of scope** — B makes it easier to add, no more.
