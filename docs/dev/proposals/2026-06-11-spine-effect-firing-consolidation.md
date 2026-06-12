---
date: 2026-06-11
status: proposal — design review (no code yet)
related:
  - 2026-06-07-scheduling-spine-v2.md # the apply seam (§B) landed; this finishes the due-ness key (round → cursor)
  - 2026-06-10-multi-stream-multi-cadence-union-axis.md # B is partly unblocked by this
  - capabilities-system.md # the obs_alignment × algorithm axis this makes uniform
area: inference / scheduling spine / effect firing
issue: gh#216 (interventions fire at the wrong time under Exact obs-alignment)
---

# Finish the scheduling spine: decide effect firing from the timeline, not `round(t/dt)`

## 0. What this is, and the bug it closes

[gh#216](https://github.com/vsbuffalo/camdl/issues/216): a scheduled
intervention or event fires at the **wrong time** under `Exact` obs-alignment in
the bootstrap particle filter, IF2, and correlated-PF — silently, with no error.
Reproduced: with `dt = 7` and an intervention at day 35, an AFP-only fit fires
it at day 35; adding an off-grid sibling observation stream (ES biweekly)
re-tiles the substep grid and the **same** intervention fires at day 37.

The root cause is two disagreeing clocks. The integrator advances on the
**timeline** (`Schedule`), which under `Exact` re-anchors substeps at every
observation time; but the _decision_ of what effect fires at a substep is made
by a **separate** computation — `round(t_end / dt)` against a precomputed
`fire_steps` table, inside `step_one` — that still assumes the global `dt` grid.
When the substep grid moves (because observations moved), `round()` lands the
intervention on the wrong substep.

This proposal closes the gap by **changing where the due-effect decision is
made**: from `round(t/dt)` to the timeline's own effect boundaries (the cursor's
`effect_idx`, `StopReason::ScheduledEffect`). The apply machinery for this
already landed in spine-v2 §B (`apply_effect_batch` consumes a pre-derived batch
and does not re-derive due-ness); only the _source_ of the batch is still
`round()`. The end-state is the plug-in architecture the maintainer's instinct
points at: one timeline spine that owns **all** timing (observations, effects,
outputs), with backends (advance dynamics) and inference algorithms (filter) as
plug-ins over it.

**Required reading:** `2026-06-07-scheduling-spine-v2.md` (§B, §C, the
Sequencing section); `rust/crates/sim/src/schedule.rs`;
`rust/crates/sim/src/effects.rs`; `rust/crates/sim/src/chain_binomial.rs`
(`step_one`); `rust/crates/sim/src/intervention.rs` (`all_intervention_times`,
the `AtTimesExpr` resolution);
`docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md`;
`docs/dev/capabilities-system.md`. High-risk: changes inference-math timing in
code (`step_one`) shared by every backend and every filter.

## 1. What exists today (verified against the code)

### 1.1 The spine and the apply seam — both already landed

`Schedule` owns all three boundary kinds and a closed "what's due" set:

```rust
// schedule.rs
pub struct Schedule { dt, t_end, grid, policy: StepPolicy,
    output_times: Vec<f64>,   // :154
    effect_times: Vec<f64>,   // :157  scheduled intervention / always-active-event boundaries
    obs_times:    Vec<f64> }  // :164  inference scoring boundaries (via with_obs)

pub enum  StopReason  { Output, ScheduledEffect, Observation, End }   // :56  CLOSED, agnostic
pub struct TimelineStop { t: f64, reasons: SmallVec<[StopReason;4]> } // :75
pub struct EffectBatch { event_idx, intervention_idx }               // :94  pre-split by lifecycle
```

`substep()` already clips to `effect_times` under `Exact` (`:231-236`,
`t_end.min(next_output).min(next_effect).min(next_obs)`); the cursor exposes
`pass_effect` (`:443`) and `effect_due_at` (`:313`); `next_stop()` tags a
landing `ScheduledEffect` (`:274`). And the **apply seam landed**:
`apply_interventions_at` is gone, replaced by `apply_effect_batch`, which
consumes a pre-derived `EffectBatch` and does _not_ re-derive due-ness;
`due_effects` (`effects.rs:258`) centralizes the due-check into one function.
Per spine-v2's Sequencing, items 1 (`StepClock`), 2 (the boundary types +
`EffectBatch` + the apply seam), and 4 (drop tau) all landed.

### 1.2 Forward backends — correct, because they register effects

The forward backends build `Schedule` **with** `effect_times` populated (from
`all_intervention_times`). Under `Snap` (chain) firing is decided by
`round(t/dt)` against `fire_steps`, which is correct because Snap never leaves
the grid. Under `Exact` (ODE, gillespie) `substep()` clips to the effect time,
so the landing stands exactly on the effect time and `round(effect_time/dt)`
resolves to the right step — also correct. **Forward is not buggy, including
Exact-forward.**

### 1.3 Inference filters — register only observations; decide firing on a second clock

The inference drivers construct the schedule with **empty** `effect_times`:

```rust
// particle_filter.rs:153, correlated_pf.rs:268, if2.rs:242
Schedule::new(dt, t_end, dt, StepPolicy::Exact, Vec::new(), Vec::new()).with_obs(obs_times)
//                                              ^^^^^^^^^  effect_times = EMPTY
// pgas.rs:375  build_substep_grid builds its own Schedule, also Vec::new() effect_times
```

They advance via the shared `Substeps` iterator (PF/IF2/correlated-PF,
`particle_filter.rs:247`, `if2.rs:420`) or `build_substep_grid` (PGAS,
`pgas.rs:359`), and the firing **decision** is made inside `step_one`, keyed on
`round(t_end / grid_dt)`:

```rust
// step_one (chain_binomial.rs) is the SHARED step function. It takes fire_steps + grid_dt.
// chain_binomial.rs:496   effects::due_effects(model, fire_steps, t + dt, grid_dt, &mut batch)
// effects.rs:266          let current_step = time_to_step(t_end, grid_dt);   // = round(t_end/grid_dt)
// effects.rs:268          if !fire_steps[iv_idx].contains(&current_step) { continue }
```

`step_one` has **no `StepPolicy` parameter** — the same function is called by
the Snap forward backend (`chain_binomial.rs:243`), the PGAS producer
(`pgas.rs:918/1146`), and PF/IF2/correlated-PF
(`chain_binomial_process.rs:106-113`). So firing is _not_ branched on policy; it
is one shared `round()` computation parameterized by `grid_dt`.

### 1.4 The disagreement, made concrete

Under `Exact`, `build_substep_grid` re-anchors the window start at each
observation (`pgas.rs:405`, `window_start = obs_t`; the consequence is
documented at `pgas.rs:2363`: "7.0 is on the GLOBAL grid but off the SHIFTED
(anchored at 3.5) grid"). With `effect_times` empty, the integrator never lands
on an intervention time — it lands on observation times — and
`round(substep_end / dt)` matches whichever `fire_step` the obs-anchored landing
happens to round to. Move the observations and the firing instant moves with
them (day 35 → 37).

### 1.5 The guard is partial

Spine-v2 §B assumed the dangerous combination is rejected by a Tier-1 guard.
That guard exists but is **PGAS-only and always-active-events-only**
(`pgas.rs:1518-1532`, condition
`step_policy == Exact && interventions.any(is_event)`; its recommended
workaround is `obs_alignment = "snap"`). PF/IF2/correlated-PF have **no** guard
and hardcode `Exact`; **scheduled interventions** are unguarded everywhere. So
three of four filters silently misfire today.

## 2. The precise gap

The apply seam (`apply_effect_batch`) and the centralized due-check
(`due_effects`) already landed. What did **not** land is the _due-ness key_:
`due_effects` computes the batch from `round(t_end/grid_dt)` against
`fire_steps` (`effects.rs:266`) instead of reading the cursor's `effect_idx` at
a `ScheduledEffect` landing. The vehicle that spine-v2 §B intended for this —
the §C closure-driver — was **deliberately dropped** as over-consolidation (a
correct call), which left the due-ness key on `round()`. And the inference
filters never populate `effect_times`, so the timeline owns the _observation_
clock for inference but not the _effect_ clock. gh#216 is the symptom.

(Note: spine-v2's prose cited `intervention.rs:133`, `effects.rs:252`,
`effects.rs:382` as the `time_to_step` sites; those line numbers are now stale —
they are a finiteness guard, a doc comment, and a brace respectively. The single
live `round()` firing-key site today is `effects.rs:266`.)

## 3. Design (types-first)

### 3.1 Register effect boundaries on the inference timeline (uniform per filter run)

Every inference `Schedule` / `build_substep_grid` is built **with** the model's
scheduled effect times — `all_intervention_times(model, θ)` — not `Vec::new()`.
The integrator then lands exactly on each intervention time under `Exact`,
regardless of which observation streams are present.

`all_intervention_times(model, θ)` is **θ-dependent** for parametric
`AtTimesExpr` schedules (gh#69). This is fine because, in every method except
IF2, **θ is uniform across the swarm at a single filter run**: PF/correlated-PF
run at one fixed θ; PMMH runs its inner filter at the proposed θ; PGAS's CSMC
step uses one `current_params` per sweep. So `effect_times` is recomputed **once
per filter run** (per sweep for PGAS, per proposal for PMMH) and is well-defined
and uniform. IF2 is the exception (§3.6).

### 3.2 Decide firing at the _caller_, not inside the shared `step_one`

Firing is one `round()` computation inside `step_one`, which is shared across
every backend and filter (§1.3). The fix is **not** a policy branch buried in
`step_one` — it is to lift the due-batch _decision_ out to the caller:

```rust
// before:  step_one(..., grid_dt, ..., fire_steps: &[BTreeSet<i64>])   // decides due-ness internally via round()
// after:   step_one(..., batch: &EffectBatch)                          // applies a batch the caller computed
```

`step_one` becomes "apply this batch" (it already has the apply seam,
`apply_effect_batch`); the **caller** computes the batch:

- **Exact-inference** (PF, IF2, correlated-PF, PGAS producer): cursor-keyed — at
  a substep landing on an effect boundary (`effect_due_at` /
  `StopReason::ScheduledEffect`), read the due `EffectBatch` from the cursor and
  `pass_effect`. The `round()` key is retired here.
- **Snap (forward + inference) and Exact-forward (ODE/gillespie)**:
  `round(t/dt)` against `fire_steps` is **retained** — it is correct there (Snap
  stays on grid; Exact-forward clips to the effect time so the landing rounds
  exactly). Naming this explicitly so a future reader does not "retire round on
  Exact" and break ODE/gillespie.

This keeps the partition **caller-side** (who fills the batch), not a fork
inside the shared step function; `step_one`'s body is unchanged behavior given
the same batch. The seam is the existing one: `apply_effect_batch` already
separates _decide-due_ from _apply_; this proposal only changes the _decide-due
source_ from `round()` to the cursor, for the Exact-inference caller.

### 3.3 The `Substeps` iterator must advance the effect cursor (or it stalls)

The production `Substeps::next` (`schedule.rs:424`) advances only the obs
cursor; it never calls `pass_effect`. So if `effect_times` is populated and the
iterator is reused unchanged, `substep()` clips to the effect time, `effect_idx`
never advances, `next_effect` stays equal to that time, and every later
`substep()` returns `0` — the within-window loop **stalls on zero-length
substeps** (same for `build_substep_grid` and `window_end`). The iterator and
the PGAS grid-builder must: at a landing on an effect boundary, surface a "fire
here" signal and `pass_effect` (advancing `effect_idx`) so the walk progresses.
This mechanism is in scope and must be implemented, not assumed from "substep()
already clips."

### 3.4 PGAS: the producer fires; value and gradient score records (no firing there)

`complete_data_loglik` (MH acceptance, `pgas.rs:659`) and
`complete_data_loglik_grad` (NUTS, `pgas_grad.rs:402`) **do not fire effects** —
they score the transition density over the already-recorded `PGASTrajectory`
(`counts_before/after`, `flows`, `gammas`). Effects fire only in the
**producer** (`simulate_reference_on_grid` / `csmc_as`, via `step_one`,
`pgas.rs:918/1146`). So value and gradient are in lockstep _by construction_ —
both read the same records. The invariant is **producer → records → both
consumers**: the producer must fire at the boundary the shared grid tiles, and
the density must score the records it produced (the `pgas.rs:1594`
producer-vs-density sanity check covers this). Registering `effect_times` once
in the single grid (built at `pgas.rs:1541`, shared by producer and both
consumers) fixes all three at once.

### 3.5 What stays bit-identical (the strict-generalization guard)

- **No interventions** → `effect_times` empty → `next_effect = ∞` → schedule and
  substep grid byte-identical to today; every inference golden unmoved.
- **On-grid interventions** → the integrator already lands at the on-grid effect
  time; the cursor-keyed firing fires at the **same** substep `round()` did
  (`round(on_grid_effect/dt)` hits that step). New path == old path. This is the
  **load-bearing gate** (Test 2), because it is the boundary between
  "byte-identical" and "the fix changed something."
- Behaviour changes **only** for off-grid effect times under `Exact` (the gh#216
  case), where today is wrong.

### 3.6 Parametric `AtTimesExpr` schedules under IF2 — guarded (loud), deferred

IF2 carries **per-particle** parameter vectors (`particle_params`,
`if2.rs:303`). For an `AtTimesExpr` (parametric `at [...]`) schedule the fire
times are evaluated per-particle, so there is no single `effect_times` a shared
immutable `Schedule` can hold. Supporting it would require per-particle
schedules, which breaks the one-shared-`Schedule` CRN invariant (`schedule.rs`
module header — N particles must walk an identically-ordered boundary sequence).
**v1 hard-errors `AtTimesExpr` + IF2 + Exact** with a message naming the
limitation (it is not silently dropped). This is IF2-specific: PGAS/PMMH/PF run
at a uniform θ per filter run (§3.1), so a single recomputed `effect_times` is
exact for them. (Orthogonal, not this proposal: fitting an intervention _time_
parameter under PGAS-NUTS is non-smooth across a substep boundary — the spine-v2
§E `Target=Parameter` / time-invariant-θ deferral, a gradient concern, not the
per-particle-schedule one.)

## 4. Why the scheduling system didn't already solve it

A **connection gap**, not missing machinery. The spine has the effect-boundary
types, `substep()` clips to them, and the apply seam (`apply_effect_batch`) is
in place; the inference drivers simply bypass the spine for the _firing
decision_ (empty `effect_times` + `round()` inside `step_one`). It accreted
naturally: the filters were built around the observation grid, interventions
reused `step_one`'s `fire_steps` mechanism, spine-v2 added the boundary types
and the apply seam, and the §C driver that would have switched the due-ness key
was dropped. No single choice was wrong; the smell is the two clocks, which
diverge only under Exact + off-grid effects — a case the multi-cadence work
makes the norm.

## 5. The plug-in question — and the leak check

The robust end-state is **three orthogonal plug-in points over one shared
spine**: the spine emits stops tagged with the closed, agnostic `StopReason` set
and owns all timing; backends advance dynamics over a substep; algorithms react
to `Observation`/reset stops. This is **not leaky** because the seam already
exists: `apply_effect_batch` separates _decide-due_ from _apply_. This proposal
changes only the decide-due _source_ (round → cursor) for the Exact-inference
caller; it does not push backend/algorithm specifics into the spine, and it does
not resurrect the declined §C driver. The split is "spine decides _when/which_
(the `EffectBatch`), `step_one` applies _how_
(PROPOSE/ADVANCE/INTERVENE/BALANCE)" — which is the architecture already in
place, not a new abstraction. Genuine unsupported cells stay `Capabilities`
hard-errors (`docs/dev/capabilities-system.md`), never spine leaks.

The synthesis: observations binding onto the union spine (proposal B) and
interventions binding onto the same spine (this proposal) are the same operation
— a boundary source with a `StopReason` tag.

## 6. Scope

- **In:** populate `effect_times` in the inference `Schedule` +
  `build_substep_grid` (recomputed per filter run, §3.1); lift the firing
  **decision** to the caller and fire cursor-keyed on the **Exact-inference**
  path (§3.2); the `Substeps` / grid-builder cursor-advance + fire-signal
  (§3.3); PGAS producer fires, consumers score (§3.4); the bit-identity guards
  (§3.5); the `AtTimesExpr + IF2 + Exact` hard-error (§3.6).
- **Retained, named (not retired):** `round(t/dt)` firing on Snap (forward +
  inference) and **Exact-forward** (ODE/gillespie) — correct there; pinned by a
  test (§8.6).
- **Stopgap, ship first (§9 phase 1):** hard-error
  `Exact + off-grid effect times` in all four Exact filters — so there is no
  silent-wrong interim. Scoped to **off-grid** only (on-grid Exact +
  interventions is correct today and must keep working).
- **Out (named):** Snap migration (correct today; uniformity cleanup, not
  needed); always-active **events** under Exact (they fire mid-step at PROPOSE,
  fused with the kernel draw — a separate fix; they stay hard-errored, §10);
  reactive / state-dependent effects (Tier-4 sibling); `Target = Parameter`
  under inference (spine-v2 §E); the §C closure-driver (declined).

## 7. Invariants

- **RNG / paired-seed CRN.** For intervention-bearing off-grid models the
  substep grid changes (it now lands on effect times) → **not** byte-identical
  to today (today is the bug). No-intervention and on-grid-intervention models
  stay byte-identical (§3.5). The cursor stays pure in `(schedule, cursor, t)`
  so N particles still walk an identically-ordered boundary sequence (the
  `schedule.rs` module-header invariant, tested by
  `n_cursors_identical_sequence`).
- **PGAS density + gradient.** Producer fires; value and gradient score records
  and are in lockstep by construction (§3.4); `shape = dt_actual/σ²` unchanged.
- **Capability matrix.** Removes a silent-wrong cell (Exact + off-grid effects
  in PF/IF2/correlated-PF) by making it correct; expresses the IF2 +
  `AtTimesExpr` limit as a loud hard-error (§3.6); changes no other cell.

## 8. Tests (red → green; inference-math — paste red/green in commits)

1. **gh#216 invariance** — one intervention; its firing instant is **invariant
   to which observation streams are present** (AFP-only vs AFP+ES) across PF,
   IF2, correlated-PF, and the PGAS producer. The day-35→37 reproduction
   becomes: fires at 35 regardless of ES.
2. **On-grid bit-identity A/B (the gate, §3.5)** — interventions on the `dt`
   grid: new cursor-keyed firing == old `round()` firing, loglik byte-identical.
   Gates the atomic phases 2+3.
3. **No-intervention bit-identity** — `effect_times` empty → grid
   byte-identical; inference goldens unmoved.
4. **No-stall regression** — an intervention-bearing model does not stall (the
   §3.3 cursor-advance works); mutation check: omitting `pass_effect` reproduces
   the zero-substep stall.
5. **PGAS producer/records** — the producer fires at the registered boundary;
   value and gradient score the produced records (finite-difference
   value-vs-grad as hygiene; the real check is producer-vs-density,
   `pgas.rs:1594`).
6. **Retained paths** — ODE-forward and Snap with an off-grid intervention are
   **unchanged** (round path still correct); pins §3.2's "retained, named."
7. **Stopgap** — before the fix, `Exact + off-grid` interventions hard-error in
   all four filters; **on-grid** Exact + interventions still fits (no
   over-rejection, the gh#187 on-grid case). `AtTimesExpr + IF2 + Exact`
   hard-errors (§3.6).

## 9. Implementation phases

1. **Stopgap guard (S, ship immediately).** Hard-error
   `Exact + off-grid effect
   times` at all four Exact entry points
   (PF/IF2/correlated-PF hardcode Exact → an unconditional check there; PGAS
   widened from events-only to also off-grid scheduled). **Off-grid only** — do
   not reject on-grid fits. Add the `AtTimesExpr
   - IF2` hard-error. Independently landable; Test 7.
2. **Register `effect_times`** in the inference `Schedule` +
   `build_substep_grid`, **and** the §3.3 cursor-advance + fire-signal, **and**
   the §3.2 caller-side firing (retire `round()` on the Exact-inference path) —
   **as one atomic change**. Registering `effect_times` _without_ retiring
   `round()` double-fires (substeps at 35 and 37 both round to step 5 — the
   failure mode of incident `2026-04-17-chain-binomial-double-fire.md`), so
   phases "register" and "retire" cannot be separated for intervention models.
   The stopgap guard stays up through this and is lifted only at the end. Tests
   1, 2, 3, 4, 5, 6.
3. **Docs** — mark spine-v2 §B fully adopted (due-ness key now cursor); update
   `capabilities-system.md` (the `obs_alignment × algorithm` axis); close gh#216
   with the reproduction-as-regression; lift the stopgap guard.

## 10. Relationship to the sibling proposals

- **Partly unblocks B** (`2026-06-10-multi-stream-multi-cadence-union-axis.md`):
  B's review HIGH-1 (the union axis perturbs intervention firing) closes for
  **scheduled interventions** — they now fire at the registered effect boundary
  regardless of observation streams. **Always-active events stay hard-errored
  under Exact** (they fire mid-step at PROPOSE, fused with the kernel draw — a
  separate fix). So B is unblocked for **event-free** multi-cadence models.
  **Action:** confirm whether the polio ES+AFP fixture uses `events {}` (e.g.
  importation pulses); if it does, B needs the event fix too, or an event-free
  fixture for v1.
- **Independent of A** (`2026-06-10-observation-data-entry-dsl.md`): A can
  proceed in parallel from its phase 1.

## 11. References (verified)

- Spine + apply seam: `schedule.rs` — `StopReason` :56, `TimelineStop` :75,
  `EffectBatch` :94, `Schedule` :145, `substep` (clips to effect under Exact)
  :231-239, `effect_due_at` :313, `pass_effect` :443, `Substeps::next`
  (obs-only) :424, CRN invariant module header + `n_cursors_identical_sequence`.
  Apply seam: `apply_effect_batch` (replaces `apply_interventions_at`),
  `due_effects` `effects.rs:258`.
- The live `round()` firing key: `effects.rs:266`
  (`current_step =
  time_to_step(t_end, grid_dt)`), `:268`
  (`fire_steps.contains`). `step_one` is shared and takes `grid_dt` (no
  `StepPolicy`): `chain_binomial.rs:496` (the `due_effects` call), callers
  `chain_binomial.rs:243` (Snap forward), `pgas.rs:918/1146` (producer),
  `chain_binomial_process.rs:106-113` (PF/IF2/corr).
- Inference schedules with empty `effect_times`: `particle_filter.rs:153`,
  `correlated_pf.rs:268`, `if2.rs:242`, `pgas.rs:375`. Re-anchor `pgas.rs:405`,
  shifted-grid test `pgas.rs:2363`, single shared grid `pgas.rs:1541`.
- Partial guard (PGAS-events-only): `pgas.rs:1518-1532`.
- Parametric schedules: `all_intervention_times(model, params)`
  `intervention.rs:195`; `AtTimesExpr` `ir/intervention.rs:27` (gh#69), resolved
  `intervention.rs:65-90`. IF2 per-particle params `if2.rs:303`.
- PGAS density/gradient score records (do not fire): `complete_data_loglik`
  `pgas.rs:659`, `complete_data_loglik_grad` `pgas_grad.rs:402`,
  producer-vs-density check `pgas.rs:1594`.
- Double-fire precedent:
  `docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md`,
  `chain_binomial.rs:237-241`.
- Spine-v2 §B (apply seam + the cursor-keyed due-ness intent), §C (dropped
  driver), §E (`Target=Parameter` deferral):
  `2026-06-07-scheduling-spine-v2.md`.
- gh#216 (the bug + the day-35→37 reproduction).
