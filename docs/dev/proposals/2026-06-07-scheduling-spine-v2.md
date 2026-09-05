# Scheduling spine v2 — timeline tightening

Date: 2026-06-07 Status: Proposed Supersedes: the scheduling-effect topology map
(`2026-06-06-scheduling-effect-topology.md`) as the _target_ architecture — that
document mapped the terrain; this one specifies the remaining reshapes against
the now-landed Tier-1/2 implementation. Out of scope (named siblings): inference
real-state support; reactive interventions (`EffectAgenda`); reactive parameters
(gradient blocker). See "Out of scope".

## Two step lengths a reader must not conflate

A simulator substep has two lengths that are usually equal but mean different
things. The model declares a **nominal** step — `dt`, say 1 day. But the
scheduler often has to **stop short**: if an intervention or observation falls
at _t_ = 2.5 while `dt` = 1, the substep starting at _t_ = 2 is clipped to **0.5
day**, because the integrator must land _exactly_ on the effect/observation
time. So at that substep:

- the **actual** length the integrator advanced is 0.5 — call it `dt_actual`;
- the **nominal grid** the model declared is still 1 — call it `grid_dt`.

They answer different questions, and conflating them is a silent wrong answer:

- _How much did the world change this substep?_ — the stochastic noise that
  accumulated, the transition probability `1 − exp(−rate·dt)`, the gamma
  overdispersion `shape = dt/σ²`, **and the rate evaluation itself** — is over
  the **actual** elapsed 0.5. Everything numerical uses `dt_actual`.
- _Which scheduled step is this?_ — mapping an effect's fire-time onto the model
  grid to decide if it fires now — uses the **nominal** `grid_dt`.

The chain-binomial kernel and the PGAS density **already** do exactly this (eval
and physics both on the clipped `dt_actual`; firing keyed on the base `dt`). The
defect is that the distinction is carried by hand-threaded conventions at each
call site rather than by a type, and one backend — tau-leap — evaluates rates on
the nominal `cfg.dt` while drawing on the clipped `dt`, an inconsistency that
silently changes results for any model whose rate references `dt`. Making the
two lengths a first-class object is the spine of this proposal.

## What already landed (this builds on solid ground)

- **The merged `Schedule` spine** — `substep` / `substeps` / `window_end` /
  `drain_outputs` / `clip`, one time→step mapping with an explicit
  `StepPolicy { Snap, Exact }` (`schedule.rs`).
- **The effect-resolution purity seam** —
  `resolve_intervention`/`resolve_events` (`StateRef` → typed `Int`/`Real`
  deltas) + trivial `apply_effects` (`effects.rs`). This _resolves the i64/f64
  dilemma_ the design review framed as `CountStoreMut`: representation rides the
  delta type, the arithmetic has one home, ODE applies exact f64, events-on-real
  apply. `CountStoreMut` is absent from the code and not coming.
- **The shared post-advance lifecycle tail** — `lifecycle::apply_post_advance`
  already owns INTERVENE + BALANCE + the negative-count check in fixed order for
  chain / tau / gillespie.
- **Tier-1 correctness guards** — finite/negative effect values
  (`finite_action_value`), the off-grid event-misfire guard
  (`schedule::reject_event_misfire`), the real-coupled inference gate (gh#191).

So the spine, the effect layer, and the post-advance tail exist. What remains is
making the _timeline semantics_ first-class instead of convention-held.

## Problem: three things are still held by convention, not types

1. **`dt` is overloaded** (the two lengths above) into one `f64` passed through
   the lifecycle. The discipline "use the clipped `dt` for physics + eval, the
   base `dt` for the firing key" is correct in the code today but
   _hand-maintained per call site_: `effects::resolve_events` computes the
   firing key as `time_to_step(t + dt, dt)` (`effects.rs:252`), and the Exact
   backends pass the base `cfg.dt` into it on purpose (`tau_leap.rs:316`,
   `apply_events_at` `intervention.rs:169`) so the key lands on the grid despite
   a clipped substep. A type would make that structural; a convention can be
   broken by the next edit. (The Tier-1 guard currently _rejects_ the dangerous
   Exact + off-grid + always-active-event combination rather than running it
   correctly — its own doc-comment names `StepClock` as the full fix.)
2. **Due-ness is re-derived after the schedule already decided it.**
   `Schedule::substep` computes where the step stops, but
   `apply_interventions_at` re-discovers "what is due" via `time_to_step(t, dt)`
   against `fire_steps` (`intervention.rs:133`). Two mechanisms answer "what is
   due," and they can disagree under Exact / off-grid / parametric / close
   times.
3. **The within-substep order is only partly structural.** `apply_post_advance`
   shares the INTERVENE + BALANCE tail across chain/tau/gillespie, but the
   PROPOSE stage (`resolve_events`) is still called directly in each backend,
   ODE runs a separate continuous path (`apply_boundary_effects_continuous`),
   and the canonical order is named by the `// → FixedStepLifecycle` comments in
   `lifecycle.rs` rather than enforced by a driver.

## The design (types first)

### A. `StepClock` — name the two step lengths

```rust
struct StepClock {
    t0: f64,         // substep start (== rate/forcing evaluation time)
    t1: f64,         // substep end == the TimelineStop (below)
    dt_actual: f64,  // = t1 - t0. The realized substep. Physics + rate eval (EvalCtx.dt).
    grid_dt: f64,    // the nominal model dt. Fire-key resolution (time_to_step) ONLY.
}
```

The load-bearing decision, stated once: **`EvalCtx.dt = dt_actual`** (rate
evaluation uses the actual elapsed length, consistent with the noise/probability
physics), and **`time_to_step` keys on `grid_dt`** (scheduling uses the nominal
grid). This is what chain and PGAS already do, so threading `StepClock` is
**byte-identical** for the discrete + ODE forward backends and for the
chain-binomial inference kernel — it codifies the existing hand-threaded
discipline. The only code whose numbers were ever on the _other_ convention is
tau-leap (rates/σ² at `cfg.dt`), and that deviation is resolved when tau folds
into chain (D below). These `dt`s are numerical/grid — the calendar→time
conversion (`docs/dates.md`) is upstream and untouched.

### B. `TimelineStop` + `StopReason` — the schedule says where to stop and why

```rust
struct TimelineStop { t: f64, reasons: SmallVec<[StopReason; 4]> }
enum  StopReason  { Output, ScheduledEffect, Observation, End }
```

A single time can be due for several reasons (output + obs + effect + end). The
`Schedule` returns the next `TimelineStop`; the driver handles its reasons in
one declared canonical order. The effect application then consumes a **known due
batch** instead of re-deriving due-ness:

```rust
struct EffectBatch { intervention_idx: SmallVec<[usize; 4]>, event_idx: SmallVec<[usize; 4]> }

impl Schedule {
    fn next_stop(&self, cursor: &Cursor, t: f64) -> Option<TimelineStop>;
    // The due effects at a stop with a ScheduledEffect reason — read from the cursor's
    // effect position, NOT re-derived via time_to_step. Deterministic given the schedule.
    fn due_effects(&self, cursor: &Cursor, stop: &TimelineStop) -> EffectBatch;
}
```

`apply_interventions_at` is replaced by `apply_effect_batch(batch, …)` —
application stops deciding due-ness (it applies a list the schedule handed it)
and does one job. The cursor already holds the effect position (`effect_idx`);
`due_effects` reads it, removing the `time_to_step(t, dt)` re-derivation at
`intervention.rs:133`, `effects.rs:252`, and `effects.rs:382`. (Static schedule
only here; a _reactive_ `due_effects(t, state, params)` that depends on latent
state is the Tier-4 sibling.) Vocabulary, going forward (no churn to existing
prose): `substep`/`interval` = `[t0,t1]`; `timeline stop`/`boundary` = `t1`;
`stop reason` = why `t1` matters; `scheduled effect` = the action due at `t1`.
("Stop" over "Event" — `Event` is already overloaded five ways; "reason" over
"kind" — a stop has several.)

### C. The closure-taking lifecycle driver (D1)

**Decided: dropped.** This unification paid off when _four_ backends shared the
fixed-step skeleton. After D (drop tau-leap) only chain (discrete) and ODE
(continuous f64) remain fixed-step, and the effect-purity seam already factors
their one genuinely-shared, bug-prone part — the representation split — through
the delta types. A two-backend `fixed_step_substep` closure would unify a loop
skeleton that is already short and clear in each, buying a shared indirection
without removing a shared hazard: consolidation past the natural seam. The
`// → FixedStepLifecycle` markers stay as documentation of the shared order;
`apply_post_advance` already owns the bug-prone tail. Retained below as the
record of the considered-and-declined option.

```rust
fixed_step_substep(
    state, clock, due_effects, scratch,
    |snapshot, event_batch, current, scratch| {
        // backend-specific ADVANCE only (the kernel draw). MUST NOT consume RNG before it.
    },
)
```

The driver owns the canonical order — snapshot capture · event PROPOSE from
snapshot · backend ADVANCE · atomic fusion of transition+event deltas ·
scheduled INTERVENE · BALANCE · postcondition checks. It already owns the tail
(`apply_post_advance`); this folds in the PROPOSE call (`resolve_events`) and
routes ODE through the same order (ODE keeps its exact-f64 _apply_, but the
_order_ is shared). The backend closure implements **only** the kernel advance
and **cannot reorder stages**; the driver guarantees the snapshot is captured
before the first RNG draw and that no RNG runs between snapshot and the closure
(the invariant that keeps event PROPOSE — which is RNG-free — order-neutral
w.r.t. the draws). No `FixedStepLifecycle` trait (Gillespie can't honor a
fixed-step advance; the only shared content is the order — exactly when a
closure beats a trait); the `// → FixedStepLifecycle` comments are deleted.
Gillespie keeps its boundary path (it already routes effects through
`apply_post_advance`); it shares effect application without pretending to have a
substep advance. Structure for an invariant currently held by comments;
byte-identical, with a per-backend A/B and an assertion that the closure
receives the pre-draw snapshot.

### D. Drop tau-leap (D3) — delete it; chain's `Exact` policy is the replacement

camdl's "tau-leap" is the **same** Euler-multinomial kernel as chain under the
`Exact` policy — verified (it is not canonical Poisson tau-leaping) — differing
only in two conventions that were arguably bugs relative to chain: rates/σ²
evaluated at `cfg.dt` not the clipped `dt_actual` (`tau_leap.rs:180`), and
transitions skipped at `rate ≤ 0` vs chain's `rate ≤ RATE_EPSILON = 1e-15`
(`tau_leap.rs:229` vs `chain_binomial.rs:23`). Inference never used it (the
filters run chain's `step_one` under Exact). So there is nothing unique to
preserve, and **no equivalence proof is required**: camdl is alpha
(backward-compat is a non-goal), so we do not owe tau's exact numbers. The bar
is **"chain+Exact is correct"**, not **"chain+Exact == tau"** — a strictly
weaker, achievable bar.

The fold is a **pure delete**:

- Delete `TauLeapSim`, `run_tau_leap[_with_observer]`, `TauLeapConfig`,
  `SimConfig::TauLeap`, the `TauLeapSim` export, and the CLI `tau_leap` backend
  arm (no alias — house policy).
- Remove tau's rows from the goldens (`gate_corner_case_baseline`,
  `gate_trajectory_baseline`) and tau-specific tests (the
  `chain_and_tau_byte_identical…` differential oracle, the `tau_leap_*` cases in
  the lifecycle-audit tests). No re-derivation — the backend is gone; the other
  backends' rows are **byte-identical** (the gate proves it). The chain-side
  event fusion the differential oracle pinned is covered by
  `pgas_event_density` + the lifecycle audit.
- **We do NOT add a chain Exact-forward backend now** (deviation from an earlier
  draft of this section). tau and chain compute the _same_ Euler-multinomial
  kernel mathematically — tau's `run_tau_leap` loop only mirrored chain's
  within-substep lifecycle (its own comment: "matches chain_binomial"); it never
  shared chain's `step_one` code. Only tau's run-loop (Exact boundary-clipping)
  was unique, and the Exact _policy_ survives in the inference filters
  (PF/IF2/correlated PF). The fast off-grid-exact _stochastic-forward_ niche is
  covered by gillespie (exact) or a finer chain `dt`; chain+Exact-forward is
  trivially addable later (the kernel + policy already exist) if the forward
  capability is wanted.

No capability is lost: every `Capability` (overdispersion, real-compartments,
balance, lineages) still has a forward backend (chain covers
overdispersion/real/balance/lineages; gillespie real/lineages; ode real). This
leaves the **honest three forward kernels** (chain / ode / gillespie). No
equivalence proof, no t=0-cadence blocker — there is nothing to match.

### E. `Target = Parameter` — the NPI axis (forward half)

`Action` gains a `{ Compartment | Parameter }` target; the resolver gains a
`ParamDelta` peer to `Int/RealDelta` (the `Arena { Int, Real }` dispatch in
`resolve_action` admits an `Arena::Param` additively), and `apply` writes the
parameter arena. **Forward simulation only.** A param effect inside a
forcing/TimeFunc needs the gh#186 fix (params baked at compile) or a
compile-error guard. The inference + reactive halves are deferred: a mid-run
parameter change makes the PGAS/NUTS gradient inconsistent (the time-invariant-θ
assumption; see the effect-purity proposal's "Out of scope"). So: forward
`set`/`scale` of a parameter at a scheduled time, nothing more, this proposal.
This step is additive and could ship independently of A–D.

## Invariants (every reshape must preserve)

- **RNG draw order / paired-seed CRN** — A–C are byte-identical (verified by A/B
  gate, not just a golden pass); the tau fold (D) only deletes a redundant
  backend — the kept backends are unaffected, and tau's goldens re-derive as
  chain+Exact (verified correct, not matched).
- **PGAS complete-data density + gradient** — `shape = dt_actual/σ²` and
  `p = 1−exp(−rate·dt_actual)`: the density's `dt` is `dt_actual`. `StepClock`
  routes `dt_actual` to physics + eval (what the density already uses), so the
  density is unmoved; the producer's source-group draw order is fixed.
- **i64 byte-identity** — the discrete backends stay byte-identical through A–C;
  only the tau→chain fold (D) moves tau's numbers (to chain's, proven/pinned),
  and Target=Parameter (E) is additive.
- **Capability matrix honesty** — three forward kernels (chain / ODE /
  Gillespie) after the fold; inference stays chain-binomial-centred; the
  `ProcessModel`/`DensityProcess` split stays. No backend becomes an inference
  kernel for symmetry.
- **Golden gates** — `gate_trajectory_baseline`, `gate_corner_case_baseline`,
  `gate_pgas_density_baseline`, `gate_inference_baseline`, the lifecycle audit
  set, **plus the two new oracle fixtures below**.

## Sequencing

**Step 0 — the missing oracles (before any reshape).** Before this work there
was _zero_ coverage of a rate expression that references `dt` under clipping:
the `Expr::Dt` node appeared in no fixture, and `gate_pgas_density_baseline`
runs `simulate_reference` at a fixed `dt` (uniform substeps,
`dt_actual == grid_dt` always) so it never clips. The plan called for (a) a
corner fixture whose rate references `dt`, scored under Exact clipping; and (b)
an Exact-clipped IF2/PF baseline.

- **(a) — landed.** `tests/fixtures/corner_cases/dt_rate.camdl` (infection
  hazard `… ·
  dt/tau`, an `Expr::Dt` node) + `gate_dt_rate_exact_clip.rs`.
  Under `StepPolicy::Exact` with off-grid obs it produces genuinely shortened
  substeps and pins, two ways:
  - _propensity level (the isolated guard)_ — the shared evaluator scales the
    infection rate **exactly** as `dt_actual/grid_dt` (0.5 on a 0.5 substep)
    while the dt-free recovery rate is bit-identical. Mutation-checked: freezing
    `Expr::Dt → 1.0` makes the ratio 1.0 and turns this red. This is the direct
    oracle for `StepClock`'s `EvalCtx.dt = dt_actual` on the rate-eval path.
  - _integration / consumer-consistency_ — the full producer → records →
    `complete_data_loglik` pipeline runs on the `Expr::Dt` model, stays finite,
    and scores it from the realized `(t0, dt_substep)` records (== realized
    recompute, ≠ the uniform `s·dt` reconstruction). NB this Δ is kernel-`dt` +
    `t0` + rate-`dt` combined, _not_ `Expr::Dt`-isolated (frozen `Expr::Dt`
    leaves it green) — it guards "reads realized records," the propensity arm
    guards the rate eval.
- **(b) — largely covered, smaller remainder.** The clipped PGAS density _and
  gradient_ are already oracle'd by `pgas_exact_tiling.rs` (shortened-substep
  value/gradient on `seir_vaccine_seasonal`); a dedicated Exact-clipped IF2/PF
  _numeric_ baseline is a lower- priority nicety, not a StepClock blocker.

Then, each behind a byte-identical A/B gate:

1. **`StepClock`** — ✅ landed. `EvalCtx.dt = dt_actual`, `time_to_step` keys on
   `grid_dt`. Byte-identical (codifies chain/PGAS); the Step-0(a) oracle
   confirms it on the dt-rate path. The Tier-1 off-grid guard is deleted (the
   rejected model class is now correct).
2. **`TimelineStop` / `StopReason` + `EffectBatch`** — ✅ landed. The schedule
   returns the next stop + the due batch; `apply_effect_batch` replaces
   `apply_interventions_at`'s due-derivation.
3. **Closure driver (D1)** — ❌ dropped (see §C). With tau gone only chain + ODE
   remain fixed-step; a two-backend `fixed_step_substep` would consolidate past
   the natural seam.
4. **Drop tau (D3)** — ✅ landed (`761c812`). Pure delete:
   `TauLeapSim`/config/CLI arm + tau golden rows + tau tests. Byte-identical for
   the surviving backends.
5. **Target=Parameter forward (E)** — ⬜ deferred (additive; ships
   independently, after the observation system).

Steps 1–2 are byte-identical (Step-0(a) + existing gates are the guard). 3 is
declined. 4 is a pure delete — byte-identical for the surviving backends (only
tau's own golden rows go); its one identity ripple is that runid's `Backend`
enum re-indexes when `TauLeap` is removed, so chain/ode run_ids shift going
forward (Gillespie, index 0, is unchanged) — alpha-acceptable, no literal is
pinned. 5 is additive.

## Out of scope (named siblings, not this proposal)

- **Inference real-state support** — carry + RK4-advance the real reservoir in
  `ParticleState` across PF/IF2/PMMH/PGAS, so real-coupled models can be _fit_
  (lifts the Tier-1 inference gate). CRN-sensitive (the reservoir joins the
  particle) and the PGAS density may need to account for it. Research-y; its own
  proposal.
- **Reactive interventions (`EffectAgenda`)** —
  `due_effects(t, state, params) → EffectBatch` where the agenda depends on
  _latent state_ (resampling clones it, PGAS ancestor-tracing accounts for it,
  CRN breaks). The static `EffectBatch` here is the precursor; the reactive
  `AgendaScope` classification is Tier 4.
- **Reactive parameters** — blocked on the gradient time-invariance assumption
  (effect-purity proposal's "Out of scope").
