# The scheduling spine: timeline, cursor, and the invariants every backend × algorithm must satisfy

The timing spine is the correctness foundation for **all** forward simulation
and **all** inference. Every backend (chain-binomial, ODE, Gillespie) and every
inference algorithm (bootstrap PF, IF2, correlated-PF, PGAS, PMMH) walks the
same `Schedule`, so a firing-time or coverage error here is not one backend's
bug — it is wrong for everything. This doc is the reference for how the spine
works and, in §2, the **normative invariant catalog** the property tests verify.

> **Naming note.** The effect type is currently `ir::intervention::Intervention`
> with `kind: InterventionKind { Scenario, Event }` (events and interventions
> are one type — see below). The kind-neutral rename to `Effect`/`EffectKind` is
> tracked in **gh#219**; this doc uses the current names. A separate
> `sim::effects::EffectKind { Intervention, Event }` is an unrelated
> lifecycle-stage trace label (also covered by gh#219).

## 1. The current system

### 1.1 One effect type, two (soon three) kinds

`events {}` and `interventions {}` both lower to a single `Intervention` struct.
The only difference is **activation**, not mechanism (camdl-language-spec
§13.5):

- **Scenario** (`interventions {}`) — off by default; toggled by
  enable/disable/set/scale scenarios.
- **Event** (`events {}`) — always-active; cannot be toggled.
- **Reactive** (gh#204, unbuilt) — fired by a runtime observation trigger.

All three **mutate state** via the same `Action` grammar (`add` / `set` /
`transfer`); they differ in _when/whether_ they fire, not in _what_ they do. The
spine treats them uniformly as **effects** — a state mutation scheduled at a
time.

### 1.2 The spine types (`sim::schedule`)

```
Schedule  (immutable, Sync — ONE per run, shared by every particle)
┌──────────────────────────────────────────────────────────────────────┐
│ dt, t_end, grid, policy: StepPolicy { Snap | Exact }                   │
│ output_times: [0 ···· 10 ···· 20 ··· ]   trajectory snapshots          │
│ effect_times: [    4 ··········· 35 ··· ]  effect boundaries           │
│ obs_times:    [ 3.5 ·· 7.5 ·· 14 ······ ]  inference scores (may be     │
│                                            OFF the dt grid)            │
└──────────────────────────────────────────────────────────────────────┘
        the three lists are pre-sorted ascending once, at construction

Cursor  (Copy — each particle owns one; the Schedule is NEVER mutated)
   { output_idx, effect_idx, obs_idx }     ← three "pop pointers", one per list

StopReason { Output, ScheduledEffect, Observation, End }
TimelineStop { t, reasons: SmallVec<StopReason> }   ← one stop, all its reasons
EffectBatch  { event_idx, intervention_idx }        ← what fires at one boundary,
                                                       pre-split by lifecycle stage
```

### 1.3 The substep walk

`Schedule::substep(cursor, t)` is the single time→step mapping:

- **Exact:** clip to the next boundary —
  `dt.min(min(t_end, next_output, next_effect, next_obs) − t)`. The integrator
  **lands exactly on** every output/effect/obs time. (The inference filters run
  chain-binomial's `step_one` under Exact; ODE and Gillespie are Exact forward.)
- **Snap:** never clip to an output/effect — `dt.min(t_end − t)`. Steps a full
  `dt` and emits/fires at grid points. (Chain-binomial forward.)

`substep` is **pure** in `(Schedule, cursor, t)` — it never mutates the cursor.
The driver advances each per-kind index (`pass_output` / `pass_effect` /
`pass_obs`) after consuming the corresponding reason. The substep returns the
step _size_ (`dt.min(boundary − t)`), not `(t+dt).min(boundary) − t`: the two
are equal in exact arithmetic but not bit-identical for large fractional `t`,
and the PGAS density (`shape = dt/σ²`) is sensitive to that ULP.

`Schedule::clip(cursor, t, t_proposed)` is Gillespie's variant: the SSA proposes
an exponential time and the schedule clips it back to the next boundary.

### 1.4 Effect firing — one cursor-keyed path (post gh#216)

`effect_times` carries **every** effect — events and interventions alike (built
by `intervention::timeline_effects` for inference, `all_intervention_times` for
forward). Under Exact the integrator therefore **lands on each effect time**,
and the driver reads the boundary's batch from the cursor and splits it by kind
via the one shared helper:

```
effects::split_due_batch(model, &timeline.batches[effect_idx], &mut batch)
   → batch.event_idx        (always-active events) — fire at PROPOSE
   → batch.intervention_idx (scheduled interventions) — fire at INTERVENE
```

The **lifecycle stage** is the genuine kind difference the spine keeps:

- **PROPOSE** — events resolve against the start-of-step snapshot and _fuse with
  the kernel draw_ (chain-binomial atomic interleaving).
- **INTERVENE** — scheduled interventions apply on the _post-advance_ state.

Firing _time_ is unified (cursor-keyed for all kinds); only the apply _stage_
forks by kind. There is no `round(t/dt)` firing key on the inference path any
more — that was the gh#216 events bug (an obs-anchored off-grid substep end
rounding onto a colliding `fire_step`). The Snap-forward path keeps
`round(t/dt)` via `due_effects` because Snap stays on the dt grid, where it is
exact.

### 1.5 The apply step is shared substrate; per-backend apply differs

Every path routes firing-_time_ through the spine +
`due_effects`/`split_due_batch`, but the **apply** differs because the state
representation differs — and that is the correct seam (share the timing, keep
the distinct dynamics):

| path                                  | builds `effect_times`                   | fires via                    | apply                                                                     |
| ------------------------------------- | --------------------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| chain-binomial (Snap, fwd)            | `all_intervention_times` (incl. events) | `due_effects` round, on-grid | `step_one` PROPOSE/INTERVENE                                              |
| ODE (Exact, fwd)                      | `all_intervention_times`                | cursor land                  | `apply_boundary_batch_continuous` (f64, no RNG)                           |
| Gillespie (Exact, fwd)                | `all_intervention_times`                | cursor land                  | discrete apply **+ full propensity recompute**                            |
| PF / IF2 / correlated-PF (Exact, inf) | `timeline_effects` (incl. events)       | cursor, `split_due_batch`    | `step_one`                                                                |
| PGAS (Exact, inf)                     | `timeline_effects`                      | cursor, `split_due_batch`    | `step_one`; **rejects events under Exact** (loud floor, gh#219 follow-up) |
| PMMH                                  | wraps a PF flavor                       | inherits                     | inherits                                                                  |

Three things are deliberately backend-specific and must not be "unified" away:
ODE's continuous apply (no RNG), Gillespie's post-effect propensity recompute (a
correctness requirement), and PGAS's events-under-Exact rejection (the loud
capability floor until its ancestor walk re-anchors at event landings).

### 1.6 The CRN invariant

`Schedule` is immutable and `Sync`; the per-particle `Cursor` is `Copy`;
`substep`/`next_stop` are pure. So N particles in a parallel swarm walk an
**identically ordered** boundary sequence — paired-seed / CRN coupling depends
on it, and a shared-mutable cursor would corrupt it without failing any
all-on-grid golden. Pinned by `schedule::tests::n_cursors_identical_sequence`.

## 2. The invariant catalog (normative — what the property tests verify)

These must hold for valid scheduling **across every backend × algorithm cell**.
Most are checked _exactly_ (no statistical tolerance) using a deterministic
`mu=0` model whose non-idempotent `N→M` transfer makes the integer `M`
compartment the firing record: 0 before, `TRANSFER` after one fire, `2·TRANSFER`
on a double-fire — firing observable to the bit.

### A. Firing correctness

- **A1 instant-invariance** — an effect at time T fires at T _regardless_ of
  which other boundaries (obs / output / sibling effects) are present. _(gh#216;
  was broken for events.)_
- **A2 fire-once** — exactly one firing per scheduled occurrence; no double, no
  miss. _(the `M=20` double-fire.)_
- **A3 no spurious fire** — never fires at a non-scheduled time (no early fire
  at an adjacent off-grid obs). _(the `M=10 at t=3.5`.)_
- **A4 declaration order** — co-scheduled effects fire in declaration order.
- **A5 lifecycle stage** — events at PROPOSE (fused with the draw),
  interventions at INTERVENE (post-advance); the kind→stage map is consistent.

### B. Boundary coverage

- **B1 land-on-boundary (Exact)** — the integrator lands exactly on every
  output/effect/obs time.
- **B2 output completeness** — one snapshot per output time.
- **B3 obs completeness** — each obs scored exactly once (inference).
- **B4 no-skip** — no boundary between `t` and the next stop is silently passed.

### C. Byte-identity / CRN

- **C1 on-grid registration is inert** — registering a boundary that coincides
  with a `dt`-multiple does not change the substep sequence or RNG draw order vs
  not registering it. _(what made the gh#216 fix byte-identical.)_
- **C2 CRN coupling** — N particles on the shared `Schedule` walk an identical
  boundary sequence.
- **C3 dt-invariance where claimed** — results stable across `dt` where the
  semantics promise it.

### D. Cross-cell consistency (the matrix)

- **D1 backend firing-time agreement** — chain / ODE / Gillespie agree on _when_
  effects fire (the dynamics differ; the firing instants must not).
- **D2 algorithm firing-time agreement** — PF / IF2 / correlated-PF / PGAS agree
  on firing instants.
- **D3 no-silent-gap** — every (backend × algorithm × feature) cell works or
  fails _loudly_ (a `Capabilities` error), never silently wrong.
- **D4 Snap/Exact equivalence on-grid** — identical results when all boundaries
  lie on the `dt` grid.

### E. Activation semantics

- **E1** — a disabled intervention never fires; an event always fires.
- **E2** _(gh#204, unbuilt)_ — a reactive effect fires iff its trigger holds.

## 3. Where these are tested

The property suite (`proptest`) verifies A1–A4, B1–B4, C1, D4 over generated
schedules at two levels — spine-level (pure `Schedule`/`Cursor`) and
matrix-level (the deterministic `mu=0` model through each filter/backend). C2 is
pinned by `n_cursors_identical_sequence`; D3 by the capability gates; the gh#216
firing arms (A1–A3) by `gh216_cursor_firing`. See `docs/dev/testing.md` for the
suite layout.
