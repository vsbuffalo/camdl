# Make `next_stop` the single boundary authority for the exact backends

Date: 2026-06-16 Issue: gh#233 Status: Draft — RFC for the `next_stop` API
change + the backend wiring it enables. Scope: `kind/refactor` (no intended
behaviour change), `status/tricky` (touches the boundary/control-flow of all
forward backends and the inference inner loops).

## Problem

The unified-timeline / scheduling-spine rewrite
(`2026-06-07-scheduling-spine-v2.md`) introduced a "single authority" for _where
does the integrator stop next, and what happens there_: `Schedule`, `Cursor`,
`TimelineStop` / `StopReason`, `EffectBatch`, `next_stop`, `clip`, `substep`,
`substeps`, `drain_outputs`. The primitives the backends actually consume landed
and are load-bearing. The **boundary authority itself did not**.

Verified against `main` (2026-06-16):

```
$ rg -n '\.next_stop\(' rust --type rust | grep -v target | grep -v tests/ | grep -v schedule.rs
rust/crates/sim/src/gillespie.rs:221:  // `Schedule::next_stop`: next_stop takes the effect raw, so an effect
rust/crates/sim/src/gillespie.rs:224:  // next_stop as the single boundary authority requires giving it that
```

`Schedule::next_stop` (`schedule.rs:260`) — the advertised single boundary
authority — has **zero production callers**. The only two references are a
comment in `gillespie.rs` explaining why it is _not_ used. `TimelineStop` and
`StopReason` have **zero** non-test constructions. `next_stop` shipped as a
unit-tested stub and is otherwise dead code.

Worse: the gh#166 ODE/rk45 work (rebased in 2026-06-16) added a **fourth**
boundary primitive — `Schedule::next_boundary` (`schedule.rs:246`), which
returns the boundary _time only_ (raw effect, no reasons) — and wired ODE
through _it_ rather than through `next_stop`. So the boundary loop is now
answered four incompatible ways — `substep` (chain / PGAS), `clip` (gillespie),
`next_boundary` (ode), and `next_stop` (with reasons, unused) — and the one
carrying the reason set, the whole point of the centralization, is the one
nobody calls. `next_boundary` is `next_stop` minus its reasons; ODE re-derives
those reasons by hand right after calling it (`effect_time` / `effect_due_at` /
`output_due_at`, `ode.rs:612,653`). Reaching for a new raw primitive instead of
the existing one-with-reasons is the incomplete centralization continuing, not
closing.

A second piece of the same centralization also shipped half-done. The spine-v2
RFC §B (lines 119–123) specced a **cursor-keyed**
`Schedule::due_effects(cursor, stop) ->
EffectBatch` that reads the due batch
from the cursor _instead of re-deriving due-ness_. What exists is a free
function `crate::effects::due_effects(model, fire_steps, t, grid_dt,
out)`
(`effects.rs:314`) that **re-derives** due-ness by `round(t/dt)` against
`fire_steps`. Every backend calls it with a hand-computed `t`:

```
$ rg -n 'fn due_effects' rust --type rust | grep -v target | grep -v tests
rust/crates/sim/src/effects.rs:314:pub fn due_effects(   # the free, time-keyed re-derivation
# (no `impl Schedule { fn due_effects }` — the cursor-keyed version never landed)
```

So the boundary loop is still answered three incompatible ways, and "consume a
known batch, don't re-derive" is unmet. Because a feature can be correct in one
cell of the backend × path matrix and silently wrong in another, this is the
_incomplete centralization_ that gh#70 (absorbing-state flush + reverse-time
boundary) and gh#208 (sparse propensity clamp) were symptoms of — see
`docs/dev/incidents/2026-06-16-gillespie-silent-wrong-test-sidestep.md`.

### The `> t` filter question — and why the answer is "no", post-rebase

gh#233's task 1 says: give `next_stop` `clip`'s `> t` effect filter so it is
safe as a boundary-advance primitive. The rebased gh#166 work supplies a
decisive counter-example: **`next_boundary` is raw (no `> t` filter) and is
correct.**

```rust
// next_boundary (schedule.rs:246) — RAW effect, and it works:
StepPolicy::Exact => self.t_end.min(self.next_output(cursor))
    .min(self.next_effect(cursor)).min(self.next_obs(cursor)),   // no `> t`

// clip (schedule.rs:298) — filters `> t`:
let eff = self.effect_time(cursor).filter(|&e| e > t).unwrap_or(f64::INFINITY);
```

The two primitives differ because they serve two different consumption patterns:

- **arrive-and-consume** (ODE, gillespie boundary handling): advance `t`
  _toward_ the boundary; when `t` reaches it, the effect cursor _still points at
  the effect_, and the same iteration applies it and `pass_effect`s. Here the
  cursor position — not a `> t` filter — is the "already handled" marker. A raw
  boundary is correct; a `> t` filter would exclude the effect at the instant
  `t` arrives on it and the integrator would step **past it without applying
  it** — silent wrong. This is exactly why `next_boundary` is raw.
- **peek-without-consume** (gillespie's SSA `clip`): boundary-finding and
  effect-application/`pass_effect` are decoupled, and the same effect can be
  re-encountered before it is passed. There the `> t` filter is the workaround
  that prevents clipping back onto an effect already applied this iteration.

The non-termination the incident attributes to a missing `> t` filter is really
a _decoupling_ bug, not a filter bug. In the current gillespie absorbing branch
the infinite loop arises because `next_eff_after_t` is `> t`-filtered (so
`at_iv` is false and the effect is never applied/passed) **while** the boundary
clip returns `t` — the two halves disagree. Couple them —
apply-and-`pass_effect` in one place (`apply_stop`, Layer 2) — and a **raw**
boundary terminates correctly: the `pass_effect` advances the cursor, so the
next `next_stop` sees the next effect (naturally `> t`) and the clock moves.
Traced against the gh#70 scenario (effect at `t_start`, absorbing state) a
raw-`next_stop` + `apply_stop` rewrite terminates; the filter is unnecessary
_and_ would break the arrive-and-consume backends.

> **Not yet test-verified** (inference/sim-correctness surface). This is a
> reasoned code trace; it inverts task 1, so it is the first thing the
> adversarial review must settle (gate angle 1) and the first thing a red→green
> test must pin before any wiring lands.

## Design

Layers, smallest/safest first: (0) name the time tolerances once and reuse them,
so the boundary logic is readable before it is restructured; (1) pin the
`next_stop` contract — **keep it raw, like `next_boundary`**, and
document/enforce the consumption discipline that makes raw correct; (2)
introduce the **one shared boundary-dispatch seam** (`apply_stop`, later
`Walk::arrive`) that couples apply-and-`pass`, which is what actually makes raw
safe and is where gh#70 lived; (2.5) **typed construction** — make
`Schedule::new` private behind mode-named constructors + role-typed boundary
lists + one exact-inference builder, so a schedule cannot be miswired at
construction; (3) a **standing cross-backend gate** (its own section below) as
the net; (4) the **capstone** — lock the raw accessors behind a `Walk` handle so
the authority cannot be bypassed (its own section below). The per-backend
_dynamics_ phase stays per-backend — and for ODE that seam already exists as
`OdeStepper::advance` (gh#166), so Layer 2 is dispatch-only there.

### Layer 0 — name the time tolerances once (byte-identical, lands first)

The boundary code is unreadable partly because its three time tolerances are
spelled as bare numeric literals scattered across the backends — and two of them
the spine _already names_ but keeps private, so the backends re-spell them by
hand. A reader cannot tell `1e-15` (a step floor) from `1e-10` (an effect-due
test) from `1e-12` (an output-due test) without chasing each call site. Promote
all three to one public definition in `schedule.rs` (the spine owns time), with
names that say what the check _means_, and replace every literal:

```rust
// schedule.rs — the single source of truth for time tolerances.
/// An output time is "reached" / due: `next_output <= t + OUTPUT_EPS`.
pub const OUTPUT_EPS: f64 = 1e-12;
/// An effect / observation time is "reached" / due: `next_effect <= t + EFFECT_EPS`.
pub const EFFECT_EPS: f64 = 1e-10;
/// A step shorter than this is zero-length: the integrator has ARRIVED at the
/// boundary, so dispatch instead of stepping. A step FLOOR, not a due-test —
/// distinct in MEANING from the two above even though numerically smaller.
/// (ode `h_max` arrival, chain loop-break, gillespie RK4-skip.)
pub const MIN_STEP_EPS: f64 = 1e-15;
```

Replacements, all byte-identical (literal → same-valued const):

- `ode.rs:608` `h_max <= 1e-15` → `MIN_STEP_EPS`; `:612`, `:653`
  `(iv - t).abs() < 1e-10` → `EFFECT_EPS`.
- `chain_binomial.rs:219` `dt <= 1e-15` → `MIN_STEP_EPS`.
- `gillespie.rs:285`, `:356` `> 1e-15` → `MIN_STEP_EPS`; `:233`, `:292`
  `< 1e-10` → `EFFECT_EPS`.

Two same-valued constants stay **separate** — they are different axes, not
duplicates, and merging them would be a category error: `RATE_EPSILON`
(`chain_binomial.rs:23`, a _rate_ floor) and the `clamp(1e-15, 1 - 1e-15)`
_probability_ guard (`chain_binomial.rs:447`).

**Discrepancy this pass surfaces (NOT byte-identical — reconcile
deliberately):** PGAS already names its negligible-step floor `GRID_STEP_EPS`
(`pgas.rs:347`) — but sets it to `1e-12`, three orders of magnitude coarser than
the forward backends' `1e-15`. A substep between `1e-15` and `1e-12` is kept by
ode/chain/gillespie and dropped by PGAS. Either that is intentional (the
inference grid needs a coarser floor — then document why and keep a distinct
name) or it should be `MIN_STEP_EPS` (a behaviour change with its own
red→green). The naming pass is what makes this visible; today it hides as two
unrelated literals. Flagged for the review, not silently unified.

Layer 0 is a pure readability commit (modulo the flagged `GRID_STEP_EPS`
decision); it lands first so the rest of the diff is legible.

### Layer 1 — the `next_stop` contract (raw)

```rust
pub fn next_stop(&self, cursor: &Cursor, t: f64) -> Option<TimelineStop> {
    if t > self.t_end + OUTPUT_EPS { return None; }      // NOT `>= t_end`: see termination note below
    let next_out = self.next_output(cursor);             // all raw — cursor position is the
    let next_eff = self.next_effect(cursor);             // "already handled" marker, NOT a `> t` filter
    let next_obs = self.next_obs(cursor);
    let stop_t = self.t_end.min(next_out).min(next_eff).min(next_obs);
    // reasons built against the SAME tolerances the *_due_at predicates use ...
}
```

`next_stop` shares its boundary-_min_ with `next_boundary` but **not** its
termination, by design: `next_boundary` returns `None` at `t >= t_end` (ODE
flushes the terminal output in a separate post-loop `drain_outputs`), whereas
`next_stop` returns the **`End` stop at `t == t_end`** so its driver dispatches
the terminal output via `apply_stop` and then breaks — folding the post-loop
flush into the loop. So the agreement property is qualified:
`next_boundary(cur,t) ==
next_stop(cur,t).map(|s| s.t)` **for `t < t_end`**; at
`t == t_end`, `next_boundary` is `None` while `next_stop` is `Some(End)`. The
redundancy still points to deriving one from the other (keep `next_boundary` as
the thin time-only view, `next_stop` as the view-with-reasons-and-the-End-stop),
so there is one boundary-min, not two.

The **correctness obligation moves to the consumer**: a `next_stop` driver MUST
apply-and-`pass` every reason it lands on, in one place, every iteration. That
is `apply_stop` (Layer 2); the contract is documented on `next_stop` and
enforced by the cross-backend gate (Layer 3). Output-at-`t` is still recorded
once then `pass_output`-ed (the
`clip_excludes_effect_exactly_at_t_but_not_output` contract is unaffected — that
is about `clip`, which stays as-is for the SSA).

Three contract points that the current stub does **not** get right and that the
unit tests do **not** cover. Each is a silent-wrong trap and each gets a pinned
test:

1. **Reason _vector_ order is display order, not application order.**
   `next_stop` lists reasons `[Output, ScheduledEffect, Observation, End]`
   (`schedule.rs:269–281`). But the canonical _application_ order at a boundary
   is **effects, then output** — output must record the post-effect state (the
   invariant `cross_backend_lifecycle_agreement.rs` pins). A driver that
   iterates `for r in &stop.reasons` in vector order would record the
   _pre_-effect snapshot. Resolution: the dispatch seam (`apply_stop`, below)
   handles reasons in a fixed application order independent of the vector; the
   vector order is for listing/diagnostics only, and this is documented on
   `TimelineStop`. (Alternative considered: re-order the vector to application
   order. Rejected — the vector is also the public "what mattered here" listing;
   coupling it to application order overloads it.)

2. **Coincident effects at one stop must fire as a batch.** `effect_times`
   (`all_intervention_times`) can list the same time twice (two interventions at
   one time). The dispatch must advance past _all_ coincident effects —
   `while effect_due_at(cursor, t)
   { pass_effect }` — and apply the whole
   batch (`due_effects` already collects all effects at the time key), not a
   single `pass_effect`. The same holds for coincident outputs (drain, not
   record-once).

3. **`End` coinciding with `Output`.** `next_stop` reports `End` whenever
   `stop_t ==
   t_end`, possibly alongside `Output`
   (`next_stop_end_coincides_with_output`). The driver must record the output
   _before_ breaking on `End`, or the terminal snapshot is dropped. Today the
   implicit `drain_outputs(.., f64::INFINITY)` final flush handles this; a
   `next_stop` driver must handle it explicitly.

Ergonomic additions to `TimelineStop`:

```rust
impl TimelineStop {
    pub fn has(&self, r: StopReason) -> bool { self.reasons.contains(&r) }
    pub fn is_end(&self) -> bool { self.has(StopReason::End) }
}
```

### Layer 2 — `apply_stop`: the one shared boundary-dispatch seam

The genuinely-shared, bug-prone piece is the **boundary dispatch**:
effects-then-output ordering, the `while effect_due_at { pass_effect }` cursor
advance, the output drain, the End handling. This is currently re-implemented in
~5 places (ode boundary block + ode post-step block + gillespie absorbing +
gillespie non-absorbing + chain's effect-cursor bookkeeping), and the divergence
between two of those copies _is_ gh#70. `next_stop` makes it one function:

```rust
/// Apply everything due at `stop`, in canonical order (effects → output), advancing
/// the cursor past each consumed boundary. The two genuinely per-backend operations
/// are injected: how an effect batch is applied (discrete counts vs continuous f64),
/// and how a snapshot is built (i64 counts/Flows::Int vs f64 to_states/Flows::Real).
pub fn apply_stop(
    schedule: &Schedule,
    cursor: &mut Cursor,
    stop: &TimelineStop,
    t: f64,
    mut apply_effects: impl FnMut(f64) -> Result<(), SimError>,  // fires the due batch at boundary t
    mut record: impl FnMut(f64),                                 // builds + pushes the snapshot at ot
) -> Result<(), SimError> {
    if stop.has(StopReason::ScheduledEffect) {
        apply_effects(t)?;
        while schedule.effect_due_at(cursor, t) { cursor.pass_effect(); }   // batch: all coincident effects
    }
    schedule.drain_outputs(cursor, t, &mut record);                         // batch: all coincident outputs
    Ok(())
}
```

This does **not** reintroduce the closure-taking _lifecycle driver_ that
spine-v2 §C declined. §C declined unifying the whole fixed-step _integrate_
skeleton (chain's binomial draws vs ode's RK4 — genuinely different dynamics).
`apply_stop` unifies only the boundary _dispatch_ (effect-order +
cursor-advance + output-record), which is the same on every exact backend and is
where gh#70 lived. The natural-seam rule says unify the bug-prone shared
substrate and keep the distinct algorithms distinct: `apply_stop` is the
substrate; `integrate_segment` (the dynamics) stays per-backend.

### Layer 2.5 — typed construction: a schedule cannot be miswired

Layer 4 locks the boundary _accessors_, but a `Schedule` is still
**constructed** from raw parts at all seven call sites:

```rust
Schedule::new(dt, t_end, grid, policy, output_times, effect_times)  // 7× today
```

Every argument after `t_end` is a foot-gun: a caller can pass a snap grid where
a nominal step belongs, select the wrong `StepPolicy`, or — the highest-cost one
— **swap the two `Vec<f64>` boundary lists**, recording snapshots at effect
times and firing interventions at output times. That compiles, runs, and
produces a plausible wrong trajectory. Named arguments do **not** save us: Rust
has no argument labels, so
`exact_forward(dt, t_end, effect_times, output_times)` compiles. The
construction surface needs the same "illegal states unrepresentable" treatment
as the boundary loop.

**Where the type-line falls.** Wrap a value when its distinct instances are
genuinely different _and_ swappable into the same slot; do not wrap when the
"distinct" values are usually the same number. The three **boundary axes**
qualify — different semantics (record / fire / score+reset), all `Vec<f64>`,
adjacent, swap-compiles → silent-wrong. Scalar steps do not — `dt == grid` at
six of seven sites, and finiteness is already checked at entry. So this layer
types the axes, not the scalars.

```rust
// One validated substrate (upgrades today's debug_assert!-only sort check to a real one):
pub struct SortedFiniteTimes(Vec<f64>);   // ctor: reject NaN/inf, sort (total_cmp)

// Three thin typed faces over it — the swap-guard. Producers fold in the current
// free fns (get_output_times / all_intervention_times), so each axis is produced,
// validated, and role-tagged in ONE place:
pub struct OutputTimes(SortedFiniteTimes);  // OutputTimes::from_model(model)
pub struct EffectTimes(SortedFiniteTimes);  // EffectTimes::from_model(model, params)
pub struct ObsTimes(SortedFiniteTimes);     // ObsTimes::from_observations(...)

impl Schedule {
    pub(self) fn new(...) -> Self;          // PRIVATE — the only raw constructor
    pub fn exact_forward(dt, t_end, OutputTimes, EffectTimes) -> Schedule;   // ode
    pub fn snap_forward(dt, t_end, OutputTimes, EffectTimes)  -> Schedule;   // chain
    pub fn ssa_forward(iv_resolution_dt, t_end, OutputTimes, EffectTimes) -> Schedule;  // gillespie
}
```

`SortedFiniteTimes` owns only the facts that are common to every boundary axis:
finite values and a deterministic order. Deduplication / duplicate rejection is
**role-specific**, not baked into the substrate. `EffectTimes` may collapse
coincident boundaries while preserving the batch membership in
`TimelineEffects`; `ObsTimes` should reject or preserve duplicates according to
the observation-model invariant it already enforces; `OutputTimes` follows the
output schedule's existing semantics. Do not hide those decisions behind one
generic "sorted times" policy.

The mode-named constructors bake in the `StepPolicy` (callers never name it) and
fix the slot order; the role wrappers make a swap a **compile error**. All three
return a **plain `Schedule`** — no mode-typed return wrappers (that is O3
typestate, deferred for gillespie's one-cursor/two-modes reason). The wrappers
live **only at the construction boundary** — unwrapped to `Vec<f64>` on the way
in, so `Schedule`'s fields and the per-particle `Cursor` walk are unchanged
(nothing threads through the hot loop or the CRN path; cheap).

**The exact-inference builder.** The exact-inference setup is currently
copy-pasted across all four inference paths — gather `obs_times`, run
`guard_attimesexpr_exact`, run `guard_exact_offgrid_effect_time`, build
`timeline_effects`, choose `sched_t_end`, then
`Schedule::new(… Exact …).with_obs(…)` in each path. A forgotten guard in a
future fifth path is a latent silent-wrong (the gh#187 class). Consolidate to
one builder so **no valid exact-inference timeline exists without the guards**:

```rust
pub struct ExactInferenceTimeline { pub schedule: Schedule, pub effects: TimelineEffects }
impl ExactInferenceTimeline {
    pub fn build(model: Option<&CompiledModel>, params: &[f64],
                 t_start: f64, dt: f64, obs: ObsTimes) -> Result<Self, SimError>;
                 // owns both guards + timeline_effects + Schedule construction
}
```

PF / IF2 / CPM / PGAS receive an already-valid timeline; they cannot drop a
guard.

Both pieces are **byte-identical**: the named constructors call the private
`new`; `SortedFiniteTimes` is a no-op on already-sorted input (only
newly-invalid input is now rejected instead of silently corrupting the walk);
`build` runs the same guards in the same order. Acceptance gates: external
`Schedule::new`, `StepPolicy::`, and `with_obs` counts outside `schedule.rs` and
tests all drop to **0**.

**Non-negotiable** here: private `Schedule::new`, the mode-named constructors,
and `ExactInferenceTimeline::build`. **Kept (cheap):** the role wrappers +
`SortedFiniteTimes`. **Deferred/dropped:** scalar newtypes
(`NominalStep`/`SnapGrid`/`RunEnd`) and mode-typed return wrappers — ceremony
disproportionate to the threat.

## Per-backend wiring

These "after" blocks are design sketches, not compiled. Every one ships behind
its backend's byte-identity baseline plus the cross-backend full-trajectory gate
(Layer 3).

### ODE (`ode.rs:597–685`, post-rebase) — dispatch consolidation, not a rewrite

gh#166 already did most of Layer 2's ODE work. The dynamics phase is extracted
behind the `OdeStepper::advance` seam (`ode.rs:219`) — integrator-agnostic,
takes the raw distance-to-boundary `h_max`, returns `h_taken`, accumulates the
augmented flow internally. So this proposal does **not** touch the integrate
phase; it only collapses the boundary _dispatch_, which gh#166 left as TWO
hand-rolled copies (the `h_max <= 1e-15` arrival block at `ode.rs:608` and the
post-`advance` block at `ode.rs:653`), each re-deriving effect-due with a
`(iv - t).abs() < 1e-10` literal, plus the final
`drain_outputs(.., f64::INFINITY)` flush.

The `1e-15` is the **adaptive-stepper arrival test**, and it stays. With rk45,
`advance` approaches a boundary over several internal steps (`h_taken ≤ h_max`),
so the loop steps, re-enters, and `h_max = boundary - t <= 1e-15` is "have I
arrived?". It is _not_ a due-tolerance (those are `EFFECT_EPS = 1e-10` /
`OUTPUT_EPS = 1e-12`); it is an "effectively-zero f64 time gap". An earlier
draft's `if dt > 1e-15 { integrate; t = stop.t }` was wrong precisely here — it
assumed one integrate reaches the boundary, true only for fixed RK4.

Before: `next_boundary` + `if h_max <= 1e-15 { dispatch-A }` + `advance` +
`{ dispatch-B }` + final flush — dispatch logic in three places.

After — `next_stop` for the reasons, `advance` unchanged, ONE `apply_stop`:

```rust
// initial snapshot at t_start (unchanged) ...
while let Some(stop) = schedule.next_stop(&cursor, t) {
    let h_max = stop.t - t;
    if h_max > 1e-15 {
        t += stepper.advance(model, params, t, h_max, &mut state)?;  // ≤ h_max (rk45 takes many)
        continue;                                                     // re-evaluate; arrive over ≥1 steps
    }
    // h_max <= 1e-15: arrived at the boundary — dispatch once, in canonical order.
    apply_stop(&schedule, &mut cursor, &stop, t,
        |bt| { let mut b = EffectBatch::default();
               crate::effects::due_effects(model, &fire_steps, bt, cfg.dt, &mut b);
               crate::effects::apply_boundary_batch_continuous(
                   model, &b, &mut state.int, &mut state.real, params, bt, cfg.dt) },
        |ot| { let (is, rs) = to_states(&state.int, &state.real);
               traj.push(Snapshot { t: ot, int_state: is, real_state: rs,
                   flows: snapshot_flows(&state.flow) });
               for v in state.flow.iter_mut() { *v = 0.0; } })?;
    if stop.is_end() { break; }
}
```

The `h_max <= 1e-15` re-entry is the load-bearing arrival check for adaptive
stepping; `apply_stop` replaces the two duplicated dispatch blocks and the final
flush with one. Note the `apply_stop` consumer fully satisfies the
raw-`next_stop` discipline (Layer 1): when `t` arrives on an effect, the cursor
still points at it, `apply_stop` applies it and `pass_effect`s — no `> t` filter
needed, and a `> t` filter would have skipped it (the silent-wrong above). This
is the head-to-head that makes "keep `next_stop` raw" concrete: ODE is the
backend the filter would break.

### Gillespie absorbing branch (`gillespie.rs:203–266`)

No proposed reaction ⇒ always take the boundary ⇒ literally `next_stop` +
`apply_stop`. Collapses the hand-rolled `next_eff_after_t` + `clip(∞)` +
`at_iv` + two `while … due_at` loops (gh#70's stranded-cursor site) into the
shared dispatch:

```rust
if lambda_total <= 0.0 {
    let Some(stop) = schedule.next_stop(&cursor, t) else { break };
    t = stop.t;
    apply_stop(&schedule, &mut cursor, &stop, t,
        |bt| { apply_events_at(bt, model, &fire_steps, iv_resolution_dt, &mut int_s, &mut real_s, params)?;
               let mut b = EffectBatch::default();
               crate::effects::due_effects(model, &fire_steps, bt, iv_resolution_dt, &mut b);
               crate::lifecycle::apply_post_advance(model, &b.intervention_idx, &mut int_s, &mut real_s,
                   params, bt - iv_resolution_dt, iv_resolution_dt, None) },
        |ot| { traj.push(Snapshot { t: ot, int_state: int_s.clone(), real_state: real_s.clone(),
                   flows: Flows::Int(current_flows.counts.clone()) }); current_flows.reset(); })?;
    if stop.has(StopReason::ScheduledEffect) {                       // state changed → may leave absorbing
        eval_propensities(model, &int_s, &real_s, params, t, iv_resolution_dt, &mut propensities)?;
        lambda_total = propensities.iter().sum();
    }
    if stop.is_end() { break; }
    continue;
}
```

### Gillespie non-absorbing branch (`gillespie.rs:268–339`) — `clip` stays as the SSA predicate

This is where gh#233's task 2 over-reaches and the natural seam differs from
"route through `next_stop`". The branch _proposes_ an exponential reaction time
`t_next` and must answer "does the reaction fire before the next boundary?" —
`clip`'s job, returning `hit_boundary`. `next_stop` has no notion of a competing
proposed time. So `clip` stays. The duplication to remove is the _dispatch after
the boundary wins_, which becomes the same `apply_stop`:

```rust
let clipped = schedule.clip(&cursor, t, t_next);   // KEEP: "did the reaction beat the boundary?"
if clipped.hit_boundary {
    if n_real > 0 && (clipped.t - t) > 1e-15 { rk4_step(model, &int_s, &mut real_s, params, t, clipped.t - t)?; real_s.clamp_nonneg(); }
    t = clipped.t;
    // `clip` found this boundary from the OLD t (it `> t`-filters the effect);
    // `next_stop` (RAW) called from the NEW t == clipped.t reports that same
    // boundary's reasons, because the cursor has not yet consumed them:
    let stop = schedule.next_stop(&cursor, t).expect("at a boundary < t_end");
    apply_stop(&schedule, &mut cursor, &stop, t, /*apply_effects=*/ ..., /*record=*/ ...)?;
    if stop.has(StopReason::ScheduledEffect) { /* full recompute */ }
    else { /* re-eval time_dep_transitions, as today */ }
    if stop.is_end() { break; }
    continue;
}
// else: reaction fires at t_next — the SSA event path, unchanged.
```

Refined gh#233 task 2: _absorbing branch and ode adopt `next_stop`+`apply_stop`;
the non-absorbing branch keeps `clip` as the reaction-vs-boundary predicate and
shares `apply_stop` for the boundary case._ `clip` and `next_stop` compute the
same boundary-min, but **not unconditionally** — `clip` `> t`-filters the effect
(and ignores obs) while `next_stop` is raw. So
`clip(cur,t,∞).t ==
next_stop(cur,t).t` holds only when `t < t_end` **and the
cursor is clean at `t`** (no unconsumed effect/obs already due at `t`). With a
boundary already due at `t`, raw `next_stop` returns `t` while `clip` looks past
it — that is correct behaviour, not a bug. Pin the property under that
precondition. (The gillespie code above satisfies it: `clip` runs from the
pre-boundary `t`, and `next_stop` runs from the freshly-advanced boundary `t`
whose reasons are still unconsumed.)

### chain_binomial (`chain_binomial.rs:213–289`) — does **not** route through `next_stop`

chain is `StepPolicy::Snap`: it steps a full `dt` every substep (never clips to
an effect/output boundary) and fires effects _inside_ `step_one`, keyed on
`round(t/dt)` — not on the loop's boundary detection. Under `Snap` the schedule
deliberately reports only `Output` boundaries (`substep` returns
`dt.min(t_end - t)`, `schedule.rs:237`). Forcing chain onto `next_stop` would
make it land exactly on effect boundaries — switching it from snap to exact
firing, a **behaviour change** (the `off_grid_intervention` snap-vs-exact
divergence the spine explicitly deferred to a future `--obs-alignment` knob,
`schedule.rs:30–38`). So chain's share of the spine is `substep` +
`substep_time` + `drain_outputs`, which it already uses; gh#233 task 2's "verify
chain substep is the genuinely-shared path" resolves to _yes, and it does not
adopt `next_stop` by design._

chain's real item is gh#233 task 4 — the tolerance mismatch:

```rust
// chain_binomial.rs:275 — cursor bookkeeping after step_one fired the effect:
while schedule.effect_time(&cursor).is_some_and(|iv| iv <= t + cfg.dt * 0.5) { cursor.pass_effect(); }
//                                                          ^^^^^^^^^^^^ half a step (could be days)
const EFFECT_EPS: f64 = 1e-10;   // schedule.rs:140 — what every exact consumer uses
```

`cfg.dt * 0.5` vs `1e-10` are unrelated tolerances. It is safe _today_ because
for snap-chain this is pure cursor bookkeeping (firing already happened in
`step_one` on the `round(t/dt)` key), but a near-coincident effect/output can be
counted "passed" by the `dt*0.5` predicate while `effect_due_at` disagrees. Fix:
advance chain's effect cursor on the _same_ `round(t/dt)` key `step_one` fires
on (`effects::due_effects` already computes it), removing the second, looser
tolerance. Byte-identical for on-grid models; pin with a near-coincident
effect/output fixture.

## PGAS substep walk vs the shared `substeps` iterator (gh#233 task 5)

The bootstrap PF, IF2, and correlated-PF walk each observation window with the
shared iterator (`particle_filter.rs:289`, `if2.rs:451`,
`correlated_pf.rs:413`):

```rust
for (t_local, step_dt, fired) in schedule.substeps(cur, t_start) { ... }
t = schedule.window_end(cur, t);
```

PGAS hand-rolls the same walk in `build_substep_grid`'s Exact arm
(`pgas.rs:409–453`). Precisely: it _consumes_ the spine primitives (`substep`,
`substep_time`, `effect_due_at`, `obs_due_at`, `pass_effect`, `pass_obs`) but
re-implements the **loop** around them — its own `loop` + re-anchor, in place of
the `Substeps` iterator the filters use. (Not "reimplements the primitives" — it
duplicates the _walk_, which is the divergence surface.) These two walks
**must** stay bit-identical on the effect/obs cadence or PGAS's transition
density silently disagrees with the PF's proposal (the gh#187 class). They
cannot be merged mechanically, for one load-bearing reason: the `Substeps`
iterator **accumulates** time (`self.t += step_dt`, `schedule.rs:445`), while
PGAS computes **drift-free** time (`substep_time(window_start,
s)`,
`pgas.rs:410`). The iterator's own docstring flags this — "the drift-free
`substep_time` variant for these is task #14" (`schedule.rs:368`). PGAS _needs_
drift-free because its continuous transition density (`shape = dt/σ²`) is
ULP-sensitive; the PF's integer draws are not.

### Resolution: one walk, two consumers (not a mode flag)

The clean fix is **not** "add a drift-free _mode_ to the iterator" — a mode is a
fork hiding inside one type (the leaky "strategy param" the design philosophy
warns against). The clean fix is structural: there is **one** walk, and PGAS
_materializes_ it instead of re-implementing it. The PF iterates it lazily; PGAS
collects it into the grid it indexes.

For that, the iterator becomes drift-free and yields everything both consumers
need (the effect _and_ obs landing, not just the effect):

```rust
// schedule.rs — ONE walk: drift-free (t0 = substep_time(window_start, s),
// re-anchored at each obs/effect landing), yielding a struct both consumers read.
pub struct Substep { pub t0: f64, pub dt: f64, pub fired_effect: Option<usize>, pub fired_obs: Option<usize> }
impl Iterator for Substeps<'_> { type Item = Substep; /* ... */ }

// PF / IF2 / correlated (lazy — same shape, now reading Substep fields):
for sub in schedule.substeps(cur, t_start) { /* sub.t0, sub.dt, sub.fired_effect */ }

// PGAS (eager — the grid IS the materialized iterator; build_substep_grid's loop DELETED):
let grid: Vec<Substep> = schedule.substeps(cur, t_start).collect();
// obs_at_substep / effect_at_substep fall out of grid[i].fired_obs / fired_effect.
```

Switching the iterator to drift-free also **fixes a latent PF bug**: per
`docs/dev/proposals/2026-06-05-substep-time-sdt-convention.md`, `substep_time`
(one multiply) is the canonical convention and chain + PGAS already use it; the
PF/IF2/correlated `t += step_dt` accumulation _drifts_ at fractional `dt`. So
unifying makes the PF **consistent with the already-canonical convention** — a
correctness alignment, not just a dedup. The cost is re-blessing the
PF/IF2/correlated baselines.

This is the **highest-risk task** (it moves inference loglik) and **needs the
maintainer's scientific sign-off on the re-bless**. The substep-time proposal
claims the shift is ≤1 ULP and bounded per-window (the walk re-anchors at each
obs, so it does not compound across the series) — **not independently verified
here**; verify the magnitude before re-blessing. Lands **last**, behind: the
`pgas_exact_tiling.rs` value+gradient oracle, a direct "new iterator vs the
deleted hand-rolled grid" bit-equality test over a battery of
`(dt, obs, effect)` configurations, and the maintainer's review of the baseline
diff.

Deleting PGAS's hand-rolled walk also removes its direct uses of the raw
accessors (`substep`, `effect_due_at`, `obs_due_at`, `pass_effect`, `pass_obs`)
— which is what makes the capstone lockdown possible (see below).

## Output-flush collapse (gh#233 task 3)

Gillespie records outputs in four places: two via `drain_outputs`
(`gillespie.rs:441`, `:454`) and two hand-rolled `while output_due_at { … }`
loops (absorbing `:252`, non-absorbing boundary `:325`). The `apply_stop` wiring
above routes both hand-rolled loops through `drain_outputs`, leaving
`drain_outputs` the single output-emission seam on every backend. (The absorbing
one is gh#70's stranded-cursor site.)

## Layer 3 — the standing guard (gh#233 task 6)

The cross-backend full-trajectory + time-monotonicity invariant added for gh#70
(`cross_backend_lifecycle_agreement.rs::full_trajectory_no_pre_event_leak_or_time_reversal`)
is the safety net for this whole refactor. Extend it to a **battery ×
multi-seed** harness:

- a set of lifecycle fixtures: coincident event+intervention, off-grid effect,
  fractional `t_end`, output-at-`t_end`, absorbing-then-importation (gh#70),
  multi-effect-coincident;
- each run on chain_binomial / ode (rk4) / gillespie across ≥8 seeds;
- assert (i) full-trajectory agreement where the models are integer-exact, (ii)
  snapshot `t` strictly non-decreasing, (iii) recorded state consistent with its
  claimed time;
- **rk4↔rk45 boundary agreement** (gh#166): on a non-`dt` model, both ODE
  integrators must land on and dispatch the same boundaries (the adaptive
  stepper only changes _how_ the segment is integrated, never _where_ it stops).
  This pins that the `next_stop`/`apply_stop` rewrite is integrator-agnostic,
  and that `advance`'s re-entry contract (`h_taken ≤ h_max`) composes with the
  dispatch.

Any future backend that bypasses the spine and re-derives boundaries by hand is
then caught on day one. This harness is the durable artifact; the point
refactors are downstream of it, and it lands **first**.

## Layer 4 (capstone) — lock the boundary API so it cannot be bypassed

gh#70, gh#208, and the gh#166 `next_boundary` accretion share one root: the
low-level boundary bricks are all `pub`, so a backend (or an agent) can
hand-assemble a boundary loop instead of going through the authority. The
authority is just one more public method in the pile. **Enforcement: make the
bricks private.** Once Layers 0–3 route all four files (ode, gillespie,
chain_binomial, pgas) through the high-level API, the raw accessors have zero
external callers (verified set, 2026-06-16) and become module-private:

| accessor                                           | external call-sites today | after Layers 0–3          |
| -------------------------------------------------- | ------------------------- | ------------------------- |
| `next_boundary`                                    | 1 (ode)                   | 0 → **private / deleted** |
| `effect_time`                                      | 6                         | 0 → **private**           |
| `output_due_at`                                    | 6                         | 0 → **private**           |
| `effect_due_at`                                    | 5                         | 0 → **private**           |
| `obs_due_at`                                       | 1                         | 0 → **private**           |
| `output_time`                                      | 3                         | 0 → **private**           |
| `Cursor::pass_output` / `pass_effect` / `pass_obs` | 6 / 6 / 1                 | 0 → **private**           |

After this, `next_boundary` cannot exist as a separate primitive (no public
`min` to expose) and reasons cannot be re-derived (no public `effect_time` /
`output_due_at`). To bypass the authority an agent must flip a private method to
`pub` — a visible, reviewable one-line diff that reads as "I am bypassing the
scheduler," not a quiet parallel helper. The lock does not _prevent_ bypass; it
makes bypass **loud**.

The recommended realization is a handle type that owns the cursor and borrows
the (immutable, `Sync`) schedule, so the backend never holds a raw `Cursor` at
all:

```rust
pub struct Walk<'s> { sched: &'s Schedule, cur: Cursor }   // Copy (cursor is Copy, sched is &) → CRN-safe
impl<'s> Walk<'s> {
    pub fn new(sched: &'s Schedule) -> Self;                 // policy lives in the Schedule
    pub fn next_stop(&self, t: f64) -> Option<Stop>;         // march peek (ode, gillespie-absorbing) — no mutation
    pub fn arrive(&mut self, t: f64, stop: Stop,
                  effects: impl FnMut(f64) -> Result<(), SimError>,
                  record: impl FnMut(f64)) -> Result<(), SimError>;  // apply effects→output, advance cursor (was apply_stop)
    pub fn clip(&self, t: f64, t_proposed: f64) -> ClipResult;       // SSA query (gillespie)
    pub fn window(self, t_start: f64) -> Substeps<'s>;               // inference window walk (Copy → per-particle)
    pub fn snap_step(&self, t: f64) -> Option<f64>;                  // chain Snap full-dt step
    pub fn drain_outputs(&mut self, until: f64, record: impl FnMut(f64));  // shared output flush
}
```

`Walk::arrive` **subsumes the `apply_stop` decision** (Layer 2): the seam
becomes a method on the handle, with the cursor private inside. `t` stays a
parameter (it is the integrator's clock — backend-owned dynamics; the cursor is
bookkeeping — `Walk`-owned). The two stepping idioms stay honest as **two query
methods** (`next_stop` for march, `clip` for SSA) returning the same `Stop`; the
dynamics (`stepper.advance`, the SSA draw, chain's binomial) never enter `Walk`.
That is the natural seam — encapsulate the shared bug-prone cursor, keep the
distinct algorithms distinct — and the guard against over-abstraction: we do
**not** force the idioms into one driver closure (the IoC design spine-v2 §C
declined).

Lands **last**, after every backend already routes through the high-level
methods, so flipping visibility is a mechanical, compiler-checked commit.

### Open design options for the handle (the review ballot)

The enforcement (raw accessors → private) is settled; the **shape of the
handle** is not. These are the candidates the design review scores against the
criteria below. They are not mutually exclusive in spirit (privacy underlies
all), but the public surface differs:

- **O1 — flat `Walk`.** One struct, ~6 methods (`next_stop`, `arrive`, `clip`,
  `window`, `snap_step`, `drain_outputs`). Simplest to write; a chain backend
  can still _name_ `clip` (cosmetic misuse).
- **O2 — enum-result `Walk` (current recommendation).** 4 methods:
  `advance(t) -> Advance{Integrate|Boundary}`,
  `clip(t, proposed) -> Clip{Reaction|Boundary}`, `arrive`, `drain_outputs`;
  `window` moves off `Walk` to `schedule.substeps()`. Per-idiom behaviour lives
  in two small exhaustive enums; `MIN_STEP_EPS` hides inside `advance`. No
  impossible variants.
- **O3 — typestate views.** `schedule.march() / .ssa() / .snap()` return
  distinct view structs over a shared private core; chain _cannot name_ `clip`
  (misuse unrepresentable). Strongest, ~3 extra tiny types.
- **O4 — privacy-only, no handle.** Keep `&Schedule` + `&mut Cursor`; make the
  raw accessors private; expose `next_stop` / `clip` / `arrive` / `substeps` as
  `Schedule` methods. Least new code; cursor still in backend hands but inert.
- **O5 — IoC driver (foil).** `schedule.drive(|segment| …, |stop| …)` owns the
  loop. Included as the explicit anti-pattern: spine-v2 §C declined it because
  it forces the march and SSA idioms through one closure (leak). Scored to
  confirm _why_ it is wrong, not to adopt.

Criteria (each option scored 1–5):

1. **Bypass-resistance** — does it stop the next-`next_boundary`
   parallel-primitive accretion?
2. **Silent-wrong resistance** — is the shared dispatch (gh#70 site) enforced,
   not optional?
3. **Leak / over-abstraction** — does it force the two idioms together, add
   generic/phantom noise, or create impossible variants?
4. **Runtime fit** — CRN (`Copy`, pure boundary fn, N-particle identical
   sequence) + adaptive rk45 re-entry (`h_taken ≤ h_max`).
5. **Inference fit** — PGAS one-iterator materialize + per-particle `substeps`.
6. **Migration** — can it land incrementally, byte-identical, low blast radius?
7. **Readability** — legible to a non-systems-engineer maintainer.

### Review outcome (4 independent agents, 2026-06-16)

Three evaluators scored this ballot and one explorer designed fresh (before
reading the proposal). They converged hard, against the proposal's own initial
pick:

- **O2 (enum-result) is out — unanimous, on a concrete code-level reason.** All
  four independently found that folding `MIN_STEP_EPS` into an
  `advance(t) -> Advance{Integrate{h_max} | Boundary}` enum **collides with the
  rk45 re-entry contract**: `OdeStepper::advance` takes `h_taken ≤ h_max` over
  many internal controller steps (`ode.rs:444–491`), so the driver needs the
  _raw_ `h_max` to feed the stepper. An `Integrate{h_max}` arm must re-surface
  the very number it claims to hide — self-undermining, and the
  integrate/boundary split is _backend dynamics_, which Layer 2 deliberately
  keeps per-backend.
- **Decision: `next_stop` returns a plain `Stop { t, reasons }`** (no enum); the
  backend computes `h_max = stop.t - t` and keeps the `h_max > MIN_STEP_EPS`
  check in its own loop, next to `stepper.advance` (the seam the integrator
  clock lives in). This is what the Layer-4 sketch and the ODE wiring above
  already show.
- **Target = O1 (flat `Walk`), reached via O4 (privacy-only) as the landing
  path** — not rivals: privatize the bricks + add the shared `arrive` seam first
  (O4, byte-identical, lowest blast radius, inference call sites untouched),
  then wrap the now-inert cursor in the `Walk` handle and finish privatization
  (O1). The enforcement (raw bricks private) is the load-bearing move and is
  shared by both.
- **O3 (typestate views) deferred — with a sharper reason than "more types":**
  gillespie needs _both_ the SSA `clip` query _and_ march-style boundary
  dispatch over **one** cursor (its non-absorbing and absorbing branches), so a
  one-view-per-idiom split fights a single backend (two views sharing a cursor =
  desync hazard). That is itself the over-abstraction criterion 3 guards
  against. Hold O3 as the fallback only if flat-`Walk`'s cosmetic misuse (chain
  _naming_ `clip`) actually bites — pay it on evidence, not preemptively.
- **Add criterion 8 — byte-identity verifiability** (can each migration step be
  A/B'd against current backend hashes): it is the real acceptance gate and was
  buried inside "migration"; O4/O1 score best on it.

All four verified the load-bearing claims against code (zero `next_stop`
callers; `next_boundary` raw + ODE re-derives reasons; the raw-vs-`> t`
inversion sound but still owed its gated red→green; `GRID_STEP_EPS`
1e-12-vs-1e-15 real; task-5 drift real; accessor table exact). No unsupported
claim survived.

## Non-goals (named, not solved here)

These are real smells in adjacent scheduling-ish surfaces, verified present, but
out of scope. gh#233 locks the runtime boundary spine + exact-inference timeline
construction; it does **not** retype the calendar/IR or the CLI emission axes.

- **Calendar/IR stringly-types.** `Model.time_unit: String` (model.rs:163),
  `origin: Option<String>` + `origin_rata_die: Option<i64>` (two independent
  Options — nothing forbids `origin = Some` with `rata_die = None`), and
  `days_per_unit(time_unit: &str)` (caltime.rs:66). Follow-up: a `TimeUnit`
  enum + a `CalendarAnchor` tying `origin` to its rata-die so the inconsistent
  state is unconstructible.
- **Off-grid-warning anchor bug (file as its own tiny cleanup).** The
  dated-loader off-grid warning checks alignment to **zero**
  (`(t / opts.dt).round() * opts.dt`, caltime_load.rs:288) while the collision
  check three blocks up correctly anchors at `t_start`
  (`interval_steps(opts.t_start, t, opts.dt)`, caltime_load.rs:265). So the
  warning mis-fires when `t_start` is not on the zero-grid. Warning-only (no
  wrong numbers), but it is duplicated grid arithmetic outside the spine — a
  small standalone commit, **not** mixed into the `next_stop` wiring.
- **CLI synthetic-obs axis.** `obs_schedule_times` (main.rs:1814) hand-rolls
  `while t <= reg.end + 1e-9 { …; t += reg.step }` — an _accumulating_ loop with
  the same drift hazard as the PF substep walk (task 5). Follow-up: a typed
  emission axis consumed by the synthetic-obs path instead of ad-hoc
  `t += step`.

## Risks and gating

- **High-risk surface** per CLAUDE.md: boundary/control-flow of all forward
  backends and the inference inner loops. Own worktree; byte-identity baselines
  per backend; the Layer-3 gate as the net.
- **No IR / schema / golden change.** Pure internal Rust (`ir/VERSION` stays
  `0.15`, the post-gh#166 schema). Goldens must not move; an `update-golden`
  diff here is a red flag, not an expected output.
- **`apply_stop` is forward-only.** The inference filters keep `substeps` (their
  walk is obs-window-scoped, not boundary-authority-scoped); only PGAS's drift
  dedup touches them.
- **Byte-identity is the acceptance bar**, not "looks equivalent." Each
  backend's existing hash/determinism gates plus the new battery must stay green
  with zero golden drift before the commit lands.

## Pre-implementation gate: adversarial review of the existing spine

`next_stop` has never run in production; its unit tests are the _only_ evidence
it is correct, and gh#233 already found a latent non-termination defect in it.
Making it the authority for every exact backend means a subtle error propagates
everywhere — the same "centralize a wrong thing" failure mode as gh#70/gh#187.
Before implementing, run a focused adversarial pass (independent agents, each
tasked to _break_ one claim against the code, not to confirm it):

1. **raw vs `> t` (the load-bearing one).** The proposal claims `next_stop` must
   stay raw and the `> t` filter would break arrive-and-consume. Try to refute:
   find a raw-`next_stop` + `apply_stop` consumer that fails to terminate or
   double-fires; OR show the `> t` filter does _not_ skip the ODE effect-at-`t`.
   Test both directions against `next_boundary` (the working raw reference).
2. **`apply_stop` order + batching** — can any reason combination record a
   pre-effect snapshot, miss a coincident effect, or drop the terminal output?
3. **adaptive arrival** — with rk45, can `advance` overshoot a boundary
   (`h_taken > h_max`), or leave `t` in a `1e-15 < gap` limbo that never trips
   the arrival branch? Does the `h_max <= 1e-15` re-entry compose with a
   coincident effect+output stop?
4. **`substeps` (accumulate) vs PGAS grid (drift-free)** — construct a
   `(dt, obs, effect)` where the two walks diverge _today_; is the divergence
   already reachable?
5. **CRN / purity** — does `apply_stop` keep `next_stop` a pure function of
   `(schedule, cursor, t)` so N particles still walk an identical boundary
   sequence (`n_cursors_identical_sequence`)?
6. **boundary-primitive agreement, with the right preconditions** —
   `next_boundary(cur,t) == next_stop(cur,t).t` for `t < t_end` (both raw — find
   a disagreement). `clip(cur,t,∞).t == next_stop(cur,t).t` only adds the
   precondition that the cursor is **clean at `t`** (no unconsumed effect/obs
   due at `t`): `clip` `> t`-filters the effect and ignores obs, so a boundary
   already due at `t` makes raw `next_stop` return `t` while `clip` looks past
   it — verify that case is the precondition, not a bug. At `t == t_end`,
   confirm `next_stop` emits `End` where `next_boundary` is `None` (terminal
   output not dropped).

Findings feed back into the contract before code is written.

## Task mapping (gh#233)

- [ ] task 1 (`> t` filter) → **inverted by the gh#166 rebase.** Keep
      `next_stop` raw (like `next_boundary`); correctness comes from
      `apply_stop` coupling apply+`pass`, not from a filter. Layer 1 + the three
      pinned contract tests + the raw-vs-filter red→green. _Documented departure
      from the issue's task 1._
- [ ] task 2 (route backends) → Layer 2: ode adopts `next_stop`+`apply_stop`
      (dispatch only — `OdeStepper::advance` already extracted by gh#166);
      gillespie-absorbing adopts `next_stop`+`apply_stop`;
      gillespie-non-absorbing keeps `clip` + shares `apply_stop`; chain stays
      Snap (does not adopt) — _refinement of the issue's framing, documented
      here._
- [ ] task 3 (output-flush collapse) → falls out of the `apply_stop` wiring.
- [ ] task 4 (chain tolerance) → `round(t/dt)` cursor advance, independent +
      low-risk.
- [ ] task 5 (PGAS dedup) → **one walk, two consumers**: drift-free `Substeps`
      yielding `Substep { t0, dt, fired_effect, fired_obs }`, PF iterates / PGAS
      collects; `build_substep_grid`'s loop deleted; PF baselines re-blessed.
      Highest-risk, lands last, maintainer signs off the re-bless.
- [ ] task 6 (standing guard) → Layer 3; lands **first**.
- [ ] new: Layer 0 (name the time tolerances) → byte-identical readability
      commit + the flagged `GRID_STEP_EPS` reconciliation; lands before any
      wiring so the rest of the diff is legible.
- [ ] new: Layer 2.5 (typed construction) → private `Schedule::new`; mode-named
      constructors (`exact_forward`/`snap_forward`/`ssa_forward`, plain
      `Schedule` returns); role wrappers `OutputTimes`/`EffectTimes`/`ObsTimes`
      over a checked `SortedFiniteTimes`; one `ExactInferenceTimeline::build`
      owning both exact-inference guards. Byte-identical; gates = external
      `Schedule::new` / `StepPolicy::` / `with_obs` counts outside `schedule.rs`
      and tests → 0. Lands before PGAS.
- [ ] new: Layer 4 capstone (lock the boundary API) → raw accessors →
      module-private behind the `Walk` handle; lands **last**, mechanical once
      0–3 route every backend through the high-level methods.
- [ ] new: off-grid-warning anchor (Non-goals) → its own tiny cleanup commit
      (`caltime_load.rs:288` → anchor at `t_start`), not mixed into the wiring.

## Sequencing

**Phasing rule (explicit):** every phase through Layer 4 is **byte-identical /
zero-golden-drift** and gated as such. The **single behaviour-moving phase is
PGAS** (step 9), sequenced **last** and gated on a _quantified_ baseline
movement plus maintainer sign-off before any re-bless. `GRID_STEP_EPS`
unification, if taken, is its own behaviour-moving change with its own red→green
— **not** folded into Layer 0.

1. Adversarial review of the existing spine + this contract.
2. Layer 0 — name the time tolerances (`OUTPUT_EPS` / `EFFECT_EPS` /
   `MIN_STEP_EPS`); byte-identical. `GRID_STEP_EPS` left **untouched** (its
   1e-12-vs-1e-15 unification is a later behaviour decision, not Layer 0). Lands
   first.
3. Layer 3 battery harness (the net).
4. Layer 1 `next_stop` raw contract + the raw-vs-filter red→green + the three
   contract tests + `TimelineStop` helpers.
5. `apply_stop` seam.
6. Layer 2.5 typed construction (private `Schedule::new` + mode constructors +
   role wrappers + `ExactInferenceTimeline::build`); byte-identical; behind the
   net.
7. ODE → gillespie-absorbing → gillespie-non-absorbing dispatch, each behind its
   baseline.
8. chain tolerance (independent).
9. PGAS one-walk dedup (last; inference-loglik risk; maintainer signs the
   re-bless).
10. Layer 4 capstone — `Walk` handle + raw-accessor privatization; raw accessors
    become private.

## Open decisions

**Need the maintainer's eyes (move inference numbers / set direction):**

- **PGAS time convention** — unify the PF/IF2/correlated walk onto drift-free
  `substep_time` and re-bless their baselines? (Recommended; aligns with the
  already-canonical convention. Moves loglik by a claimed ≤1 ULP/window — verify
  before re-blessing.)
- **`GRID_STEP_EPS` 1e-12 vs `MIN_STEP_EPS` 1e-15** — is PGAS's coarser
  negligible-step floor intentional or an accident? (Likely accident → unify to
  `MIN_STEP_EPS`, a behaviour change with its own red→green.)

**Resolved by the review (no longer open):**

- **Handle shape — DECIDED (maintainer, 2026-06-16): O4 then continue to O1.**
  Land privacy + the shared `arrive` seam first (O4, byte-identical, inference
  call sites untouched), then wrap the now-inert cursor in the `Walk` handle and
  finish privatization (O1). O4 is a strict prefix of O1, so each lands as its
  own gated commit. The O2 `advance` enum is dropped (rk45 re-entry); O3
  typestate deferred (gillespie's one-cursor / two-views coupling).
- `next_stop` raw vs `> t`-filtered → **raw**, corroborated independently by all
  four agents (still owes its gated red→green).
- `next_boundary`'s fate → keep as the thin time-only view, derive `next_stop`
  from the shared boundary-min.
- Reason vector order → fixed application order inside `arrive`.
- `apply_stop` shape → `Walk::arrive`.

**Still need the maintainer's eyes (move inference numbers):** the PGAS re-bless
and the `GRID_STEP_EPS` call above.
