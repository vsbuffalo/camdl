---
date: 2026-06-06
status: proposal — design map / architecture
area: simulation engine / inference / observation layer / DSL
related:
  - 2026-06-05-unified-timeline-effect-architecture.md
  - 2026-06-05-substep-time-sdt-convention.md
  - 2026-06-06-observation-system.md
  - 2026-05-14-reactive-interventions-and-evsi.md
  - 2026-06-06-unified-timeline-consolidation-review.md
  - ../notes/2026-06-06-backend-rationalization.md
issue: TBD
---

# Scheduling-effect topology: the seams between time, process, effect, and observation

> **Status (2026-06-07):** under active implementation (Tier 0 landed; see the
> [tiered TODO](../lifecycle-consolidation-todo.md)). Refined by an external
> [design review](../reviews/2026-06-07-lifecycle-design-review.md): the
> lifecycle is a closure-taking driver (no trait), the i64/f64 seam is a
> `CountStoreMut` state-view, tau-leap folds by extracting one shared kernel,
> and the 6-layer framing tightens into a 4-seam target (`Schedule/Clock` ·
> `EffectAgenda` · `Lifecycle` · `Kernel`). A **v2 proposal superseding this
> one** is written when the timeline-tightening tier begins; until then this
> remains the design map, read alongside the review and the TODO.

## Executive summary

**The problem.** A simulation is one object — a latent state advancing through
time — but the timed things that touch it (observations, interventions, events,
balance, and obs-to-`dt` matching) are expressed through scattered, hand-rolled
surfaces across eight loops (four forward backends, four inference algorithms).
That scattering is the bug surface. This document maps where those loops
genuinely share structure, where they diverge, and — at each divergence — _what
is binding underneath_, so we consolidate to the natural seam and stop there
rather than building a leaky monolithic scheduler.

**The thesis.** "Unified timeline" is _two_ separable consolidations, and only
one landed. The **timing spine** (Layer 0 — when does the integrator stop, how
big is the step, how do obs snap or match) was extracted into one `Schedule` and
is byte-identical where it matters. The **effect-order lifecycle** (Layer 2 —
within one step, in what order do effects apply and what state does each read)
was _named_ but **never enforced**: two of four backends still run the opposite
order. That is the one place a cross-backend _correctness_ divergence currently
lives, and it is the headline work that remains.

**The map — six layers; the seam falls differently in each:**

| Layer                    | What it is                                                                         | Shared across the eight?                      | Seam verdict                                           |
| ------------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------ |
| **0. Time substrate**    | `Schedule` boundary cursor: where does time stop next, what is due there           | **all eight**                                 | fully shared — landed, sound                           |
| **1. Kernel**            | the state-advance that consumes randomness                                         | **none** (3 distinct after retiring tau-leap) | distinct by nature — the one legitimately special part |
| **2. Substep lifecycle** | within-step order: propose → advance → intervene → balance → observe → reset       | all backends (Gillespie at `clip` boundaries) | **shareable, not yet built — the real gap**            |
| **3. Effects**           | the typed things on the timeline (observe / event / intervene / constrain / reset) | types yes; _application_ partly               | types shareable; application breaks at ODE             |
| **4. Observation**       | scoring + reconciling obs times with `dt`                                          | scoring fully; reconciliation by policy       | scoring shared; reset + collision guard not built      |
| **5. Drivers**           | the eight loops themselves                                                         | no — distinct bodies                          | distinct by nature — the iterator is the seam          |
| **6. Gates**             | capability / alignment admission checks                                            | no — three axes                               | three honest gates, not one                            |

**Backends — keep three, retire one.** tau-leap's kernel is byte-for-byte
chain-binomial's (`tau_leap.rs:172` "Match chain-binomial's Euler-multinomial");
its only differentiator (exact off-grid boundaries) is now a `StepPolicy`. Full
reasoning:
[backend-rationalization note](../notes/2026-06-06-backend-rationalization.md).

| Backend            | Kernel                                      | Distinct math?     | Inference?             | Unique value                                    | Verdict                                       |
| ------------------ | ------------------------------------------- | ------------------ | ---------------------- | ----------------------------------------------- | --------------------------------------------- |
| **chain-binomial** | Euler-multinomial (binomial competing-exit) | reference          | **yes** (the only one) | production engine + `balance`                   | **keep — it's everything**                    |
| **tau-leap**       | _same_ Euler-multinomial                    | **no — identical** | no                     | "exact boundaries" = now a `StepPolicy`         | **retire → fold into chain-binomial `Exact`** |
| **ODE**            | RK4, deterministic, `f64`                   | yes                | reserved (trajmatch)   | deterministic limit, large-`N`, smooth gradient | keep                                          |
| **Gillespie**      | exact SSA, event-driven, `i64`              | yes                | no                     | exact small-count dynamics + validation oracle  | keep                                          |

| Shared abstraction                  | Unifies across      | Holdout       | Binding reason                                                                     |
| ----------------------------------- | ------------------- | ------------- | ---------------------------------------------------------------------------------- |
| Time substrate (`Schedule`)         | all                 | —             | landed                                                                             |
| **Lifecycle order/apply (`Stage`)** | **all backends**    | —             | pure fn of `(snapshot, current, effects)`; Gillespie calls it at a `clip` boundary |
| Substep **cadence**                 | chain-binomial, ODE | **Gillespie** | event-driven — no "substep" unit; `clip`, not `substep`                            |
| Effect **application** (`Action`)   | i64 backends        | **ODE**       | `Action` is `i64`-typed, ODE is `f64` → the `{IntDelta\|RealDelta}` apply-seam     |
| Balance (`Constrain`)               | chain-binomial      | the rest      | residual-compartment needs a substep + integer end-state                           |
| Scoring (`log_likelihood`)          | all 4 algorithms    | —             | landed                                                                             |

The lifecycle **order/apply unifies across all backends** (Gillespie included,
at a `clip` boundary). The only thing that does _not_ unify is the substep
**cadence** — Gillespie's single, honest holdout.

**Build now, this round (each step gated):**

- **Layer-2 shared lifecycle apply** — lift `apply_effects_in_lifecycle_order`
  out of the per-backend step functions; route all backends through it. This
  kills the `M1` event/intervention inversion class for _every_ backend at once.
- **Layer-3 `{IntDelta | RealDelta | ParamDelta}` apply-seam** — collapses the
  single-endpoint `Set`/`Add` arms of the two `Action` interpreters, ends ODE's
  `f64→i64→f64` quantization, **and** adds the
  `Target = Compartment | Parameter` axis (the NPI / vector-control /
  reactive-`β` unlock) in the same IR change. (Honest scope: `Transfer` is a
  2-endpoint action and `ParamDelta` writes a different destination, so those
  stay distinct arms; and the param axis carries a `gh#186` prerequisite — see
  Layer 3.)
- **Retire tau-leap** by folding it into `chain-binomial + Exact` — _not_ a free
  StepPolicy flip: it requires unifying the off-grid firing tolerance (`dt*0.5`
  vs `1e-10`), a `balance + Exact` decision (no guard exists today), and the
  `M1` canonicalization first (so "byte-identical" is against a re-baselined
  tau-leap, plus a no-orphaned-capability check). With those, the equivalence
  proof doubles as the StepPolicy/lifecycle validation gate.
- First-class `TemporalKind`; the runtime sub-`dt` collision guard; the
  per-stream `ResetWindow`. Keep the **thin** `Schedule` (do _not_ add the heavy
  ADT) and the **three** separate gates (not one).

**Forward-compatible (named now, built later):**

- **Reactive interventions** = a `Sense` stage + an augmented policy register,
  fitting the architecture with no new subsystem — but **forward + PF-family
  only; PGAS is gated out** (its ancestor sampling cannot condition on
  path-dependent policy state). Fitting reactive campaigns is a non-goal, stated
  to the user as a clean capability error.
- **The observation system** binds at exactly one mapping (`BoundObs` stream →
  one `Observe` effect + one per-stream `ResetWindow`); its data layer proceeds
  in parallel.

**Four live correctness gaps to close before calling this "consolidated":** `M1`
(lifecycle inversion, currently _blessed_ by per-backend baselines), `M2`
(strictly-increasing sub-`dt` obs times silently drop a likelihood term), `M3`
(global flow-reset corrupts multi-cadence streams), `M6` (correlated-PF silently
falls back to fresh RNG, decorrelating the estimator).

## What this document is

A simulation in camdl is one object — a latent state advancing through time —
touched by several kinds of timed thing: observations read off it, interventions
and cohort entries written onto it, conservation constraints imposed on it, and
a likelihood scored against it. Eight top-level loops walk that object: four
forward backends (chain-binomial, tau-leap, ODE, Gillespie) and four inference
algorithms (bootstrap particle filter, IF2, PMMH, PGAS). Historically each of
the eight hand-rolled its own timeline arithmetic, its own effect-application
order, and its own observation-matching rule. That scattering is the bug surface
this work exists to shrink.

This document is the **topology**: a map of where those eight loops genuinely
share structure, where they genuinely diverge, and — at each divergence — _what
is binding underneath_ that prevents a larger shared abstraction. It is the
design map the implementation work and the open review items (`M1`–`M6`,
`m1`–`m8` in the
[consolidation review](../reviews/2026-06-06-unified-timeline-consolidation-review.md))
sit under. It covers every scheduling-relevant operation in every backend and
every fitting algorithm, states the coarse set of types and how they interact,
and draws the seams as algebraic data types (ADTs) so the architecture is
visible in the type structure rather than buried in eight control-flow loops.

The governing principle is **consolidate to the natural seam, not past it**. We
are not building a monolithic scheduler. A single "effect engine" that every
backend funnels through would be a leaky abstraction — its toggles would
multiply until it re-encoded the very differences it claimed to erase. The goal
is to find the seam where the genuinely-shared substrate ends and the
genuinely-distinct algorithm begins, unify below it, and stop.

Acronyms on first use: **CRN** common random numbers (paired-seed coupling);
**SSA** stochastic simulation algorithm (Gillespie's exact event-driven method);
**SMC** sequential Monte Carlo; **PF** particle filter; **PGAS** particle Gibbs
with ancestor sampling; **IF2** iterated filtering (Ionides et al. 2015);
**PMMH** particle marginal Metropolis–Hastings; **CPM** correlated
pseudo-marginal (the correlated-PF variant PMMH uses); **ADT** algebraic data
type; **IR** intermediate representation; **FD** finite difference; **ULP** unit
in the last place.

## The organizing axis: dt-dependence

Everything below sorts on one property — whether a backend's _result_ depends on
the integrator step size `dt`.

- **dt-dependent (stochastic, fixed-step): chain-binomial, tau-leap.** State is
  `i64` counts. A step over interval `h` draws `Binom(n, 1−e^{−λh})` (and
  Poisson / negative-binomial for ungrouped transitions), **freezing the rate
  `λ` at the start-of-step value**. For a state-dependent rate (`λ_SI = βI/N`),
  two steps of `h/2` re-evaluate `λ` at the midpoint — a finer, more accurate
  approximation. The realized trajectory distribution is therefore a function of
  _where the step boundaries fall_. Both `dt` and `dt/2` converge to the exact
  process as `dt → 0` (the rate-freezing difference is `O(dt)`); neither is
  "wrong."
- **dt-independent, deterministic: ODE.** State is `f64`. RK4 integrates to
  whatever time you ask; no noise, no discretization-of-randomness error.
  Landing on an off-grid time is free.
- **dt-independent, exact: Gillespie.** State is `i64`. Event-driven SSA: draw
  an exponential waiting time, fire one reaction, repeat. No discretization
  error — indifferent to where you stop. (Caveat: dt-independent only for
  time-_homogeneous_ rates; `gh#95` is the current inhomogeneous-rate bias, so
  do not lean on "Gillespie lands exactly for free" as a clean invariant for
  seasonal models.)

The consequence threads through every layer: **landing exactly on an off-grid
boundary changes the result only for the dt-dependent backends.** This is why
the snap-versus-exact observation-matching choice (Layer 4) is a real behaviour
knob for chain-binomial / tau-leap and a no-op for Gillespie / ODE, and why
migrating PGAS to non-uniform substeps is the single genuinely delicate piece in
the whole effort.

---

## Layer 0 — the time substrate (`Schedule`). Fully shared; landed.

This is the consolidation that worked, and it is the right one. `Schedule`
(`sim/src/schedule.rs`) is a **thin immutable boundary cursor** over three
parallel sorted time vectors, plus a step policy:

```rust
pub enum StepPolicy { Snap, Exact }          // schedule.rs:50

pub struct Schedule {                         // schedule.rs:79
    dt: f64,
    t_end: f64,
    grid: f64,                                // the snap grid (see m1 — currently dead)
    policy: StepPolicy,
    output_times: Vec<f64>,                   // forward snapshot times
    effect_times: Vec<f64>,                   // intervention / event boundary times
    obs_times: Vec<f64>,                      // inference scoring times (empty for forward)
}

pub struct Cursor { output_idx, effect_idx, obs_idx: usize }   // Copy — schedule.rs:63
```

It answers the timeline question two ways, because the spine genuinely forks:

- **Fixed-step _ask_** — `substep(cursor, t) -> Option<f64>`, the next step size
  (`schedule.rs:162`). The bit-exact rule is `dt.min(boundary − t)`, never
  `(t+dt).min(boundary) − t` — the two agree in exact arithmetic but differ by a
  ULP for large fractional `t` (`(t+dt) − t ≠ dt`), and the continuous PGAS
  density (`shape = dt/σ²`) is sensitive to that ULP even though integer draws
  are not (pinned: `substep_is_bit_exact_dt_min_not_t_to_minus_t`,
  schedule.rs:439). Under `Exact` the boundary is
  `min(t_end, next_output, next_effect, next_obs)`; under `Snap` it is just
  `t_end` (effects fire elsewhere — see Layer 2).
- **Event-driven _propose_** — `clip(cursor, t, t_proposed) -> ClipResult`
  (`schedule.rs:189`). Gillespie draws an exponential `t_proposed`; the schedule
  clips it back to the nearest earlier boundary or passes it through. The `> t`
  filter on effects (but not outputs) reproduces the SSA boundary semantics: an
  effect exactly at `t` has already fired this iteration and must not re-fire.

Derived primitives built once and shared by the inference filters:

- `substep_time(window_start, s) = window_start + s·dt` (`schedule.rs:251`) —
  the drift-free substep start time, one multiply (`O(1)` rounding) versus
  accumulation (`t += dt`, `O(s)` drift). This is the only value that reaches a
  time-dependent rate, so it is the only thing the
  [`s·dt` convention](2026-06-05-substep-time-sdt-convention.md) changes.
- `substeps(cursor, t_start) -> Iterator<(t_local, step_dt)>`
  (`schedule.rs:269`) — the inner per-observation-window walk, terminating at
  the cursor's obs boundary. **The single shared primitive behind the bootstrap
  PF, IF2, and correlated-PF inner loops** (formerly four hand-rolled
  `while t_local < obs_time` copies).
- `window_end(cursor, t_start)` (`schedule.rs:280`) — where that walk leaves the
  clock, defined _in terms of the same iterator_ so a filter's single-threaded
  catch-up advance cannot drift from its per-particle walk.
- `drain_outputs(cursor, until, record)` (`schedule.rs:292`) — the
  output-emission walk all four forward backends shared.

**The CRN invariant — the one regression that breaks silently.** `Schedule` is
`Sync` and immutable; `Cursor` is `Copy`. `substep` / `next_boundary` are pure
in `(Schedule, cursor, t)` with no interior mutability, so N particles in a
parallel swarm walk an identically-ordered boundary sequence — paired-seed
coupling depends on this, and a shared-mutable cursor would corrupt it without
failing any on-grid golden. This is a _type-enforced_ property (`Cursor: Copy`,
`Schedule` immutable), which is why it holds; the test that purports to check it
(`n_cursors_identical_sequence`, schedule.rs:530) actually only proves `walk` is
deterministic (review `m5` — strengthen or downgrade its claim).

**Why this seam is right.** Time-advance arithmetic — boundary detection, step
clipping, output draining, the `dt.min(boundary−t)` ULP rule — is _identical_
across all eight loops and is exactly where off-by-a-step and drift bugs lived.
It shares cleanly because it touches no state and consumes no randomness. The
landed `Schedule` is far thinner than the predecessor proposal's sketched
`Boundary`/`Trigger`/`Stage`/`Effect`/`EffectCaps` type system — and that
thinness is correct. The proposal over-specified the types; the code right-sized
to a boundary cursor over parallel vectors. **Do not "fix" this by adding the
heavy ADT.**

One piece of Layer 0 did _not_ land: the `grid` field is **dead** (read only by
its own getter at `schedule.rs:239`; zero call sites — verified). It was meant
to be the single source of truth every `time_to_step` / `resolve_fire_steps`
reads, sealing the "interventions and observations snap to the same grid"
guarantee. Today interventions still snap via each backend's own
`resolve_fire_steps(cfg.dt)`, and Gillespie via
`iv_resolution_dt = model.simulation.dt.unwrap_or(1.0)` (`gillespie.rs:113`) — a
grid the integrator never walks, defaulting to `1.0` when `dt` is unset. In
practice Gillespie uses that one value consistently for both its `Schedule` and
its `fire_steps`, so interventions and outputs do _not_ snap to different grids
today; the hazard is the silent `unwrap_or(1.0)`. Either wire the field (kill
the `unwrap_or`) or delete it (review `m1`).

---

## Layer 1 — the kernel. Distinct by nature; correctly not unified.

The kernel is the one legitimately special component: the state-advance that
moves the true latent state forward, consuming randomness in a fixed order (the
property CRN and the PGAS `gamma_used` / `binomial_z` hooks depend on). Its
specialness is real — the alloc-free hot loop, the draw ordering, and (for PGAS)
the transition density and its derivative.

| Backend        | State                                                      | One "step"                                                                                                  | RNG draws (in order)                                                 | Time-advance query |
| -------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------ |
| chain-binomial | `i64` counts (+ optional `f64` real via RK4)               | Euler-multinomial: per source group `Binom(n, 1−e^{−Σr·dt})` split proportionally; Poisson/NegBin ungrouped | gamma multiplier → total-exit binomial → split binomials → ungrouped | `substep` (Snap)   |
| tau-leap       | `i64` (+ optional `f64` real)                              | same Euler-multinomial kernel as chain-binomial                                                             | same order                                                           | `substep` (Exact)  |
| ODE            | **all `f64`** (`int_vals`, `real_vals`)                    | RK4 over the combined vector; `int_vals.max(0.0)` post-step                                                 | **none** (deterministic; `_seed` ignored)                            | `substep` (Exact)  |
| Gillespie      | `i64` (+ optional `f64` real, PDMP via RK4 between events) | one reaction: exponential wait, select transition ∝ propensity, apply ±1 stoichiometry                      | exponential `u1` → transition-select `u2`                            | **`clip`**         |

Two trait hierarchies express this, and both are kept:

```
ProcessModel : Send + Sync                     // traits.rs — the inference kernel
  type State : Clone + Send + Resettable
  fn step(&self, &mut State, θ, t, dt, rng, scratch)     // alloc-free hot loop
DensityProcess : ProcessModel                  // PGAS / gradient only
  fn log_transition_density(...)               // chain-binomial only, by design
```

`ProcessModel`/`DensityProcess` are implemented by exactly **one** type,
`ChainBinomialProcess` — inference is chain-binomial-only, and that is a
deliberate scope, not an omission. The transition math is _not_ duplicated
between forward and inference: both call the same kernel `step_one`
(`chain_binomial.rs:269`); the forward `Simulate` impls and the inference
`ProcessModel::step` differ only in the loop around it and an allocation
contract (the inference inner loop threads a reusable `Scratch` and must not
allocate; the forward driver allocates freely).

**Why this seam is right.** A `step(dt)` contract subsumes chain-binomial,
tau-leap, and ODE (RK4 sub-stepping inside one `dt`, ignoring `rng` because it
is deterministic). It does _not_ subsume Gillespie — that is the first hard
abstraction boundary, and it is binding, not incidental. See Layer 2.

**Where `forcing {}` lives.** Seasonal forcing and covariate tables (the
`TimeFunc` surface) are _not_ timeline effects — they are time-varying inputs
the kernel reads _inside_ rate evaluation, here at Layer 1, not points on the
schedule. A forcing never mutates, reads, or constrains state; it just makes a
rate a function of `t`. So it has no `Stage` and no entry in the effect taxonomy
(Layer 3) — it belongs to the kernel's rate construction. One sharp consequence,
load-bearing for the parameter-mutation axis (Layer 3): a `TimeFunc`/forcing
caches its parameters at compile time (`compiled_model.rs:686`; `eval_time_func`
takes only `t`, `propensity.rs:340`), so a param consumed through a forcing is
_frozen_ — un-estimable and un-mutable at runtime (`gh#186`). That freeze is the
reason the `Target = Parameter` axis must guard forcing-consumed params (below).

---

## Layer 2 — the substep lifecycle. The gap.

This is the layer the predecessor proposal made its centerpiece and the layer
that did not get built. It is also the layer where a real cross-backend
_correctness_ divergence lives today.

### The canonical order

The within-substep order is a first-class object — the analogue of SLiM's
published tick/generation cycle, whose defining virtue is that a modeller can
reason precisely about _when_ their script runs relative to reproduction and
selection (Haller & Messer 2019, _MBE_ 36:632). camdl's intended substep
lifecycle, matching chain-binomial's `step_one` as it stands:

```
┌─ start of substep: snapshot x_t ────────────────────────────────────┐
│  1. PROPOSE    transition draws (rates frozen at x_t)               │
│                event deltas (computed from the x_t SNAPSHOT)        │
│  2. ADVANCE    apply transition + event deltas ATOMICALLY → x'      │  fused
│  3. INTERVENE  apply scheduled interventions on x' (CURRENT state)  │  &mut
│  4. BALANCE    enforce conservation (last; target exempt)           │  &mut
│  5. OBSERVE    read projection of post-effect state; score / emit   │  &state
│  6. RESET      if an Interval obs fired: zero THAT stream's flows   │  &mut
└─ end of substep: x_{t+dt} ──────────────────────────────────────────┘
```

Two structural facts the order bakes in. **Events read the snapshot** — the
event delta is _computed_ from `x_t` and applied _atomically_ with the
transition draws (a single fused stage, not `Transition < Event`), because the
delta cannot be applied after the draws without losing atomicity.
**Interventions read the current state** — they are a genuine `&mut` on `x'`,
after the fused advance. These two facts are the entire reason "event" and
"intervention" are different types and not one effect with a `stage` flag (Layer
3).

### What the backends do — and the two residuals (updated post-M1)

The lifecycle has two divergence axes — the _order_ (event-vs-intervention) and
the _read-source_ (does an event read the start-of-step snapshot, or
post-transition state). `M1` (`ec4e7d1`) closed the first; the second remains
and is step 2's target.

| Axis                         | chain-binomial                                                                    | tau-leap                                                     | ODE                                                 | Gillespie                                     | status                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------- |
| event vs intervene **order** | event → intervene                                                                 | event → intervene                                            | event → intervene                                   | event → intervene                             | **fixed (M1)** — all backends now event-first, reading pre-intervention state |
| event delta **read-source**  | **snapshot** (pre-transition; `inject_event_deltas` from `scratch.int_s`, cb:446) | **post-transition** (`apply_events_at` on the stepped state) | **post-transition**                                 | n/a (no coincident flow at a `clip` boundary) | **the step-2 residual**                                                       |
| balance                      | last; target exempt from neg-check (cb:490, 524)                                  | N/A (capability not declared)                                | N/A                                                 | N/A                                           | unchanged                                                                     |
| intervention state typing    | `i64` (native)                                                                    | `i64` (native)                                               | **`f64→i64→f64` round-trip** (`to_states`, ode:151) | `i64` (native)                                | step-3 apply-seam                                                             |

So the _order_ is now canonical across all backends, pinned by a cross-backend
agreement test (`cross_backend_lifecycle_agreement.rs`, terminal counts
identical). The remaining **event-vs-transition read-source residual**:
chain-binomial computes event deltas from the pre-transition snapshot and fuses
them with the transition draws, while tau/ode still read the post-transition
state — so a _value-dependent_ event coincident with _live transition flow_ in
the same compartment diverges by `O(dt·rate)`. The existing fixture cannot see
it (`event_intervention_agree` sets the transition rate to zero, and
`all_lifecycle` uses a constant `add`, value-independent of the compartment).
Step 2 closes the residual for the i64 fixed-step backends (tau-leap;
chain-binomial is already the reference) and adds the strong fixture that
exposes it; ODE's fusion waits on the step-3 apply-seam (it is the one `f64`
backend, and a fused i64-snapshot event would re-introduce the mid-trajectory
quantization the ODE backend exists to avoid); Gillespie has no coincident-flow
residual (its events fire at `clip` boundaries with no in-flight transition).

### What is binding — the seam analysis

This is the heart of the topology. Can the canonical lifecycle be shared across
all four backends? Answer, per backend, with the binding constraint named:

- **chain-binomial — yes; it _is_ the reference.** Already implements the order.
- **tau-leap — yes; the divergence is an accident, not a constraint.** The
  kernel is the same fixed-step integer draw as chain-binomial. Nothing about it
  forces interventions before events or events to read live state; the order is
  a pure call-site accident at `tau_leap.rs:128–130`. Canonicalizing is a safe,
  targeted rewrite plus a re-baseline. (It also lacks the BALANCE capability by
  declaration, not by kernel constraint.)
- **ODE — order yes; _effect application_ breaks at the type level.** The order
  inversion is the same accident. But ODE holds `f64` state and the `Action`
  interpreter is `i64`-typed, so it round-trips
  `f64 → round → i64 → apply → f64` at every intervention boundary (`to_states`,
  ode:150). This quantizes continuous
  `Set`/`Add`/`AbsoluteTransfer`/`FractionTransfer`. **The binding constraint is
  the integer-typed `Action` ADT, not the lifecycle order** — see Layer 3.
- **Gillespie — genuinely binding; it cannot share the _substep_ lifecycle.**
  Three things bind: (1) it advances by a drawn continuous waiting time, not
  `dt`, so "substep" is not a unit; (2) it fires _one_ reaction per iteration,
  so the fused PROPOSE→ADVANCE atomicity has no meaning; (3) **after any state
  mutation it must recompute all propensities and draw a fresh exponential** (it
  cannot carry remaining exponential time across a mutation — the proposed
  `t_next` is _discarded_ on a boundary hit, gil:212–234). This is exactly why
  it uses `clip`, not `substep`. What Gillespie _does_ share is the effect
  _order at boundaries_ (intervene → event → observe) and the effect _types_ —
  it just applies them when `clip` lands on a boundary, not on a substep grid.

**The seam, stated precisely.** The lifecycle splits into two parts with
different shareability:

1. The **effect-application order and read-source** —
   `Stage::Advance(fused) <
   Intervene < Balance < Observe < Reset`, "events
   read snapshot, interventions read current." This is a pure function of
   `(current_state, effects)` with no `dt` and no kernel knowledge, **shareable
   across all four backends**. Two refinements the signature must respect: (a)
   "events read snapshot" is a _fixed-step_ concern — only
   chain-binomial/tau-leap/ode have a start-of-step snapshot distinct from the
   current state; Gillespie has no fused advance, so on its `clip` boundary the
   snapshot _is_ the current state and the shared apply takes `current` only
   (the `snapshot` distinction lives in the fixed-step caller, which passes its
   frozen copy). (b) The apply is a state→state function, but Gillespie must
   follow it with a **propensity recompute** (it cannot carry exponential time
   across a mutation) — that recompute is a named _caller post-condition_, not
   part of the pure apply. So the extractable
   `apply_effects_in_lifecycle_order(current, effects)` is shared; the
   snapshot-read and the post-apply recompute are caller obligations. It does
   not exist yet and is the actual remaining Layer-2 work.
2. The **substep cadence** — "do this every `dt`." This is shared by the three
   fixed-step backends and _not_ by Gillespie (event-driven). Gillespie binds.

The predecessor proposal extracted neither the order nor a shared apply; it
extracted only the _timing_ (Layer 0) and left effect application inside each
backend's step function, where the three copies drifted into the inversion. The
fix is to lift `(1)` into one shared, hand-computed-fixture-tested function and
route all four backends through it; `(2)` stays per-kernel because Gillespie
binds.

---

## Layer 3 — the effects. Types shareable; application breaks at ODE.

The things on the timeline differ along three axes the design should make
**types, not conventions**, because each is where a generic "effect" leaks:

1. **Relation to state** — read (observation), mutate (intervention/event), or
   constrain (balance). The read/write split is type-enforceable.
2. **Read-source** — snapshot (event) versus current (intervention). Forced by
   the fusion, above.
3. **Lifecycle stage** — the `Stage` sort key from Layer 2.

### The proposed effect ADTs (right-sized)

These are a design sketch, _not yet built_. The code today keeps the IR types
(`Intervention`/`Action`/`InterventionSchedule`/`BalanceSpec`) and the two
hand-rolled apply functions; this is the typed seam they should map onto:

```rust
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage { Advance, Intervene, Balance, Observe, Reset }
//   txn+event FUSED = Advance  <  Intervene  <  Balance  <  Observe  <  Reset

/// READ — observation. Gets &State, never &mut. Stage::Observe.
pub struct Observe { trigger: Trigger, projection: StreamProjection,
    kind: TemporalKind, likelihood: ResolvedLikelihood }
    // fn project(&self, &State, t) -> f64

/// EVENT — Stage::Advance. A fused delta CONTRIBUTOR; delta computed from the
/// start-of-step snapshot, applied atomically with the transition draws.
pub struct Event { trigger: Trigger /* EverySubstep */, actions: Vec<Action> }
    // fn propose_delta(&self, snapshot: &State, t) -> Deltas      (NOT apply(&mut))

/// INTERVENTION — Stage::Intervene. Genuine &mut on the post-advance current state.
pub struct Intervene { trigger: Trigger, actions: Vec<Action> }
    // fn apply(&self, &mut State, t)

/// CONSTRAIN — balance. Stage::Balance, last among writes; target exempt.
pub struct Constrain { target: usize, expr: ResolvedExpr }
    // fn enforce(&self, &mut State, t)

/// RESET — Stage::Reset. Per-stream flow-accumulator window close, keyed to the
/// firing stream's flow indices (NOT global) — the not-yet-built per-stream fix.
pub struct ResetWindow { flow_indices: Vec<usize> }
    // fn reset(&self, &mut State)
```

The event/intervention split is _forced_, not stylistic:
`propose_delta(snapshot)` versus `apply(&mut current)` is the read-source
distinction made a signature. An `apply(&mut State)` on an event could not be
atomic with the transition draws. This is why a single
`Effect { stage, actions }` with a `stage` discriminator would leak — it would
erase whether the actions read the snapshot or the current state, the exact
distinction the inversion bug turns on.

### The integer-typing break — the deepest abstraction boundary

The IR `Action` ADT (`ir/src/intervention.rs:59`) is f64-valued in its
expressions but **integer-typed at application**:

```rust
pub enum Action {                            // magnitude is always an Expr (f64)
    FractionTransfer { src, dst, fraction: Expr },
    AbsoluteTransfer { src, dst, count: Expr },
    Set { compartment, value: Expr },
    Add { compartment, count: Expr },
}
```

Every apply path quantizes to `i64`: `Set → value.round() as i64`,
`Add → count.round() as i64`, `FractionTransfer → (n·frac).floor() as i64`,
`AbsoluteTransfer → (count.round() as i64).min(n)`. And there are **two parallel
interpreters** that duplicate all four arms:

- `apply_intervention` (`sim/intervention.rs:257`) — scheduled interventions.
  Reads _live_ state, has an `f64` fall-through branch _for compartments that
  resolve via `global_to_real`_.
- `inject_event_deltas` (`sim/intervention.rs:116`) — events. Reads the
  _snapshot_, emits snapshot-relative deltas, and has **no real-compartment
  branch at all** — an event on a real compartment is silently quantized or
  _dropped_.

The break, stated as binding: **ODE is the one `f64`-state backend, and the
`Action` interpreter is `i64`-first.** So ODE round-trips `f64→i64→f64` at every
intervention firing (review `m7`), quantizing the continuous state it exists to
integrate smoothly; and an _event_ on a real ODE compartment falls through the
missing branch. The f64 path is a bolted-on `else if` in one of the two
interpreters. To share the INTERVENE stage across all four backends cleanly, the
fix is to lift the int-vs-real dispatch out of every action arm into **one typed
seam** — an `apply` over a `{ IntDelta | RealDelta }` result — collapsing the
`round`/`floor` sites and making the int/real decision once. That is the
consolidation Layer 3 actually needs; it is orthogonal to the lifecycle order.

**Honest scope of the seam.** The dispatch is not a flat 3-arm enum; it is
`{action} × {int, real, param} × {1-endpoint, 2-endpoint}`. `Set`/`Add` are
single-endpoint and consolidate cleanly. `Transfer` is _2-endpoint_ and does not
produce one delta — it produces a pair, and the pair can straddle the int/real
boundary: a `FractionTransfer` with an `i64` `src` and an `f64` `dst` has **no
code path today** (both interpreters require both endpoints in the same vector,
`intervention.rs:285-295`) and is a silent no-op — a pre-existing hole the seam
rework must close, not introduce. And `ParamDelta` (the parameter axis, below)
writes the parameter vector — exempt from balance and the negative-count guard —
so it shares the _evaluate-Expr-to-f64_ step but not the destination structure
of the count arms. So the consolidation is real for the single-endpoint count
actions and honest about keeping `Transfer` (two endpoints) and `Param`
(different destination) as distinct arms — reuse-with-distinction, not a flat
fold.

### Balance is not portable — and that is correct

`balance {}` is **chain-binomial-only by design** (`Capabilities::BALANCE`,
declared only by chain-binomial; `lib.rs`). The residual-compartment semantics —
"the compartment that absorbs whatever transitions and events left over,
enforced after them, exempt from the negative-count check" — presuppose a
_substep_ with a defined end-of-step state. Gillespie has no substep; ODE
conserves algebraically. Dispatching a `balance {}` model to them is a hard
capability error before simulation starts, not a silent drop. This is a genuine
non-portability with a binding reason, and the correct design records it as a
capability flag rather than faking a shared `Constrain` that three backends
would reject anyway.

### The hash-position contract — the freedom budget for ADT redesign

`ir/src/observation.rs:13` documents that `Projection` variant _indices_ are
positional and permanent: the `run_id` hash (`runid::ir_hash`) tags variants by
declaration order, so inserting a variant earlier churns every stored `run_id`.
For the externally-tagged enums (`Action`, `InterventionSchedule`,
`Likelihood`), the hash binds the variant _tag string_ and _field names_, not a
positional integer. Consequence for any redesign: **appending variants and
adding `#[serde(default)]` fields is free; renaming a variant or field, or
reshaping `Action`'s magnitude representation in a way that changes the JSON,
churns every stored `run_id`.** This is the budget. The `{IntDelta|RealDelta}`
seam above is a _runtime_ apply change, not an IR-shape change, so it is free
under this contract. (CAS run-identity reminder: typed run-input structs
auto-re-key when a field is added, so the budget is about IR serialization, not
the keying.)

---

## Layer 4 — the observation layer. Scoring shared; reconciliation by policy.

This is the layer that sits on top and reconciles observed data with integrator
steps — Vince's question: _what if observations are not aligned with `dt`, and
how do we handle it uniformly across all backends?_

### The scoring seam — fully shared

One method scores everywhere:
`ObservationModel::log_likelihood(&state, obs_idx,
params) -> f64`
(`traits.rs`), called by bootstrap PF, IF2, and (via the PF it wraps) PMMH. PGAS
scores through `MultiStreamObsModel::log_likelihood_from_flows_and_counts` on
the concrete type, because it carries flat
`(counts: Vec<i64>, cum_flows: Vec<u64>)` rather than a `ParticleState`. Both
entry points **converge on one summation loop** (`gh#139`): the trait method
unpacks `ParticleState` and delegates to the flat one. This seam is right and
landed.

### The projection ADT — and the missing `TemporalKind`

```rust
pub enum StreamProjection {                  // multi_stream_obs.rs:71
    FlowSum(Vec<usize>),    // reads FLOWS accumulated over the window — INTERVAL (incidence)
    IntCompSum(Vec<usize>), // reads COUNTS at the obs instant   — INSTANT (prevalence)
    Expr(ResolvedExpr),     // arbitrary expr over counts at the instant — INSTANT
}
// resets_after_observation() == matches!(self, FlowSum(_))      // ONLY FlowSum resets
```

The Interval-versus-Instant distinction — the crux of "obs not aligned with
`dt`" — is encoded _implicitly_ as a boolean predicate on the projection
variant, not as a first-class type. There is **no `TemporalKind` ADT**
(verified: zero matches in the crate). This is a finding: making
`TemporalKind { Interval, Instant }` first-class (carried on `Observe`) is the
clean way to express the two reconciliation rules, rather than a `matches!` on
the projection. Note the two variants do **not** exhaust the temporal-read space
— case-based surveillance with a reporting delay (AFP paralysis-to-confirmation,
a convolution kernel over past incidence) needs a third, `Convolved`, which the
current projections cannot express. That is an observation-model extension, not
a timeline concern, so it is out of scope here — but `TemporalKind` should be
introduced as an _open_ ADT (a future `Convolved` arm is additive under the
hash-position contract), not presented as a closed two-variant type.

### How obs reconcile with `dt` — the two rules, one policy

- **Instant (prevalence)** reads `counts` at the _substep end_. Correct only if
  a substep boundary lands on `t_obs`; otherwise prevalence is read at the wrong
  time.
- **Interval (incidence)** reads `cum_flows`, the running sum since the last
  reset, over the window `(t_prev_obs, t_obs]`. Correct only if the window
  boundary (where the accumulator zeroes) coincides with `t_obs`.

Both reduce to one question — _does a boundary land on `t_obs`?_ — answered
uniformly, across all backends, by `StepPolicy` (Layer 0):

- **`Exact`** shortens the final substep to land on `t_obs` (re-anchoring the
  window grid). Used by bootstrap PF / IF2 / plain PMMH. Lossless timing; for
  dt-dependent backends it changes the result (finer rate-freezing) and is the
  more accurate choice.
- **`Snap`** rounds `t_obs` onto the `dt` grid. Used by PGAS. Reproducible,
  uniform grid, no per-substep bookkeeping — but two sub-`dt` obs collide.

This is exactly Vince's intuition that obs-matching "should have a certain way
to snap or match, regardless of the backend." It does, and it lives in the right
place (the `Schedule`, via `with_obs` + `StepPolicy`), so threading it changes
every caller at once.

### The snap/exact support matrix — three classes, not two

The alignment is **not** uniformly available; the binding reasons are
per-algorithm:

| Algorithm                | Exact (off-grid)            | Snap  | Binding constraint                                                                                                                           |
| ------------------------ | --------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| bootstrap PF             | yes (any obs)               | (n/i) | holds no per-window substep-count assumption                                                                                                 |
| IF2                      | yes (any obs)               | (n/i) | same                                                                                                                                         |
| PMMH (plain, ρ unset)    | yes (any obs)               | (n/i) | _is_ the bootstrap PF                                                                                                                        |
| PMMH (correlated, ρ set) | **on-grid uniform only**    | (n/i) | pre-drawn-noise array sized by scalar `steps_per_obs`, indexed `i·steps_per_obs + substep`; off-grid overruns into the next particle's block |
| PGAS                     | not yet (`Exact` arm gated) | yes   | density/CSMC/traceback need a materialized indexable grid; exact-PGAS is the deferred increment                                              |

So the gate must classify correlated-PMMH as _exact-on-grid-or-error_, never
lump it with the bootstrap PF.

### Two correctness hazards on this surface — both confirmed, both live

- **Sub-`dt` collision drop (review `M2`).** `build_obs_at_substep`
  (`pgas.rs:280`) maps each obs to a substep via `interval_steps`
  (round-to-nearest) and `map.insert(s, obs_idx)` — last-wins. Two _distinct,
  strictly-increasing_ sub-`dt`-separated times (obs at 3.0 and 3.4, `dt=1`)
  both round to the same substep; one observation **silently drops from the
  likelihood** → wrong posterior. The only guard,
  `validate_obs_times_increasing` (`gh#188`), rejects `t[i+1] ≤ t[i]` only —
  strictly-increasing-but-colliding passes it. Needs a runtime collision guard,
  red-first.
- **Silent CPM decorrelation (review `M6`).** The correlated-PF noise-injection
  guard `if noise_idx < gamma_row.len()` (`correlated_pf.rs:333`) is a **silent
  skip to fresh RNG**, not a hard error — it decorrelates the estimator without
  a diagnostic. Unreachable in valid on-grid runs given the upstream
  uniform-spacing gate, but that gate has an `n_obs > 2` hole and a `dt·0.5`
  slack. Should be a `debug_assert!` / hard error per "never silently accept
  invalid input."

### The flow-reset gap — global, not per-stream (review `M3`)

After an Interval observation, the flow accumulators zero to start the next
window. This reset is **global** — it zeroes _all_ flows for all streams
(`particle_filter.rs` ~414; `pgas.rs` ~801). It is safe _today_ only because
`MultiStreamObsModel::new` rejects heterogeneous schedules (every stream must
observe at every obs time; `multi_stream_obs.rs:338`). The "Im5 canary" comment
in `particle_filter.rs` documents this and names the failure: lift the
single-schedule guard for a multi-cadence model (weekly cases + monthly deaths)
without making the reset per-stream, and each weekly `cases` observation zeroes
the `deaths` accumulator too — the monthly stream then sees only the last week's
flows, under-counting monthly incidence ~4× and scoring it against the
full-month observed count. The `ResetWindow{ flow_indices }` keyed to the firing
stream (Layer 3) is the fix; it was **not built**, and it belongs to _this_
work, not the data loader — see the handoff below.

### Handoff to the observation system — what binds where

The next push after this one is the observation system (the data-loading and
sparse-cell layer,
[`2026-06-06-observation-system.md`](2026-06-06-observation-system.md), which
supersedes the pre-spine
[`2026-06-05-observation-data-binding.md`](2026-06-05-observation-data-binding.md)).
It lands cleanly only if the seam between "timeline reconciliation" and "data
binding" is drawn here, not blurred. The split:

- **This work owns the _temporal_ layer** — the reconciliation of obs times with
  `dt`. Concretely it ships: the `Observe` effect (read-only, `&State`); the
  per-stream `ResetWindow` (Layer 3, the `M3` fix — the loader cannot own a
  per-substep state write); first-class `TemporalKind { Interval, Instant }`
  carried on `Observe` (Layer 4, replacing the `resets_after_observation()`
  predicate); the `StepPolicy { Snap, Exact }` off-grid reconciliation (Layer
  0); and the runtime sub-`dt` collision guard (`M2`). Everything that is a
  _per-substep state read/write or a time→step mapping_ lives here.
- **The observation system owns the _data_ layer** — `obsdata::bind` →
  `BoundObs` (a model-shaped object with typed `Option` holes), the cardinality
  map, the `Counted{value, denom}` per-survey denominator, the NaN/finiteness
  guard, the column typing, and `camdl check-data`. Everything that is _parsing
  untyped rows into model-shaped cells_ lives there. None of it touches a
  substep.

The two meet at exactly one mechanical mapping: each `BoundObs` stream becomes
one `Observe` effect (its `TemporalKind` choosing `FlowSum` vs
`IntCompSum`/`Expr`, its `Option` cells deciding which `obs_idx` contribute a
likelihood term), and each `Interval` stream contributes one `ResetWindow` keyed
to its own flow indices. So the per-stream reset is not a thing the loader
"re-homes" — it is a `ResetWindow` this work builds, that the loader's
`BoundObs` _populates_. Correcting that division of labour is the load-bearing
reconciliation, and it is why the per-stream reset must land here (it is a
`Stage::Reset` write, not a parse step). The data layer is independent of the
timeline and can proceed in parallel; the union-axis + `Option`-cell scoring
change waits on the per-stream `ResetWindow` this work ships.

Two consequences of the union axis to state, because both are easy to miss.
**First, a cross-stream trajectory coupling under `Exact`:** the schedule steps
to _every_ union obs time, so adding a second stream (say monthly deaths)
inserts extra integrator stops into a model that previously had only weekly
cases — and for a dt-dependent backend those extra stops change the
rate-freezing granularity, so the _cases_ trajectory is no longer invariant to
which other streams are present. Both alignments converge as `dt → 0`, and
`Snap` is unaffected, but "the cases likelihood depends only on the cases data"
is false under `Exact` + a union axis; this should be documented, not
discovered. **Second, the per-cell denominator:** the obs system's
`ObsCell::Counted{value, denom}` (the per-survey Binomial/BetaBinomial `n`, the
malaria case) must reach the log-pmf, but the scoring seam this work owns is
`log_likelihood(state, obs_idx, params)` — there is no per-`obs_idx` denominator
channel today. Threading `denom` per cell is a _scoring-seam_ change, owned here
(Layer 4), not something the loader can do alone; name it as a seam-signature
extension so the obs system doesn't assume it's free.

---

## Layer 5 — the drivers. Distinct bodies; the iterator is the seam.

The eight loops share the spine (Layer 0) and the scoring seam (Layer 4) and
keep their own bodies, because the bodies differ for real:

```
bootstrap PF    for (t,dt) in substeps { step; recoverable err → mark DEAD }   + systematic resample
correlated PF   for (t,dt) in substeps { inject PreDrawn noise; step }          + sorted/correlated resample
IF2             the bootstrap body, WRAPPED in for iter { perturb θ; cool }      + joint state+param resample
PGAS sweep      for s in grid         { conditional step }   + ANCESTOR resample + density + record
```

PGAS is the one that does _not_ use the `substeps` iterator — it consumes the
`Schedule` once at grid-build time and tiles against a frozen, indexable
`SubstepGrid`, because its density, conditional-SMC, ancestor weights, and
traceback all need a materialized `(t0, dt_substep)` array, not a streamed
iterator. The density/gradient consumers correctly read the realized `rec.t0` /
`rec.dt_substep` off each record (all eight sites verified — `pgas.rs:693`,
`:1021`, `:1245`; `pgas_grad.rs:410`), never recomputing `s·dt`; the only
surviving `t_start + s·dt` sites are uniform-grid _builders_ (the single source
of truth).

**Why folding the bodies would leak.** The death-policy (mark-dead vs
propagate), the pre-drawn-noise injection, the IF2 θ-perturbation/cooling, and
PGAS's conditioning/ancestor/density/record are _real_ differences. Absorbing
them into a `run_filter(strategy, noise, conditional, death_policy, …)` is the
five-toggle god-function the design philosophy warns against — the toggles would
re-encode the very distinctions the unification claimed to remove. The honest
seam is: the shared **iterator** (the spine) + the shared **trait spine**
(`ProcessModel`/`ObservationModel`/`DensityProcess`) + the shared **helpers**
(`systematic_resample`, `log_sum_exp`); four distinct bodies above it. This is
"a family of reused functions that can't merge further without leaking is the
right design," applied. (A thin `run_filter` wrapping the _outer_ obs-loop for
the three non-conditional filters is a candidate, but only if it composes
_without_ death/noise toggles — a later call, not committed here.)

---

## Layer 6 — the gates. Three axes; three honest gates, not one.

The predecessor proposal wanted to "consolidate the two scattered capability
gates into one seam." This is wrong, and the review (`M4`) is right: there are
**three genuinely different axes**, and collapsing them is the leak.

1. **`resolve_obs_alignment`** (`fit/methods.rs:350`) — axis: _(algorithm ×
   obs_alignment × correlated × on-grid)_. Pure timeline reconciliation: no
   model, no capabilities. Arms: if2/pfilter `snap → error`; pgas
   `exact → not-implemented`; correlated-PMMH `off-grid → error`. This is the
   new, clean gate.
2. **`check_model_capabilities`** (`fit/methods.rs:402`) — axis: _(model ×
   backend)_ for inference.
   `chain_binomial ⊇ {OVERDISPERSION, REAL_COMPARTMENTS}`,
   `ode ⊇ {REAL_COMPARTMENTS}`.
3. **forward gate** (`util.rs:1699`) — axis: _(model × backend)_ for forward
   simulation, scanning `required_capabilities()` against
   `backend.capabilities()`.

(A fourth surface, `validate_combo` / `METHODS`, gates _(algorithm × backend)_
structure.) These are different questions — timeline-reconciliation vs
model-feature-vs-backend vs forward-vs-inference. A single gate taking
`(algorithm, backend, model, alignment, on_grid)` and dispatching on all five is
the anti-pattern. **Keep them separate.** The only real duplication is the
capability-set _definitions_ (hard-coded in `methods.rs` vs read from
`backend.capabilities()` in `util.rs`) — those could share one source without
merging the call sites. The proposal text claiming "one seam" should be
corrected to "three gates, three axes."

---

## The binding-constraint map — where each larger abstraction stops, and why

The single table this whole document builds to. For every candidate shared
abstraction: how far it shares, where it breaks, and what is binding underneath.

| Candidate abstraction                                            | Shares across                                    | Breaks at             | Binding constraint                                    | Verdict                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------ | --------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| **Time substrate** (`Schedule` boundary cursor)                  | all 8 drivers                                    | —                     | none (touches no state, no RNG)                       | **shared — landed**                                      |
| **Obs-matching policy** (`Snap`\|`Exact`)                        | all fixed-step + inference                       | Gillespie             | event-driven kernel _proposes_ time → uses `clip`     | shared for fixed-step; Gillespie via `clip`              |
| **Substep cadence** ("every `dt`")                               | chain, tau, ODE                                  | Gillespie             | exact-SSA: continuous waiting time, no substep unit   | binds at Gillespie                                       |
| **Lifecycle order + read-source** (`Stage`, snapshot-vs-current) | **all 4** (even Gillespie, at `clip` boundaries) | —                     | none — pure fn of `(snapshot, current, effects)`      | **shareable — NOT BUILT (the gap)**                      |
| **Effect types** (Observe/Event/Intervene/Constrain/Reset)       | all                                              | —                     | none — types are clean                                | shareable — not built (IR types kept, hand-rolled apply) |
| **Effect application** (`Action` interpreter)                    | i64 backends                                     | ODE                   | `Action` is `i64`-typed; ODE is `f64`                 | **breaks — needs `{IntDelta\|RealDelta}` seam**          |
| **Balance** (`Constrain`)                                        | chain-binomial only                              | tau/ode/gillespie     | residual-compartment needs a substep + integer counts | **not portable by design** (capability flag)             |
| **Scoring** (`log_likelihood`)                                   | all 4 algorithms                                 | —                     | none                                                  | **shared — landed (gh#139)**                             |
| **Flow reset**                                                   | all (single-schedule)                            | multi-cadence streams | global reset zeroes every accumulator                 | **breaks — needs per-stream `ResetWindow`** (not built)  |
| **Filter loop body**                                             | —                                                | each algorithm        | death / noise / θ-perturbation / conditioning differ  | **not shared by design** (iterator is the seam)          |
| **Admission gates**                                              | —                                                | three axes            | alignment ≠ model-capability ≠ forward/inference      | **three gates, not one**                                 |

Read top to bottom, this is the typology: a fully-shared time substrate; a
shareable-but-unbuilt lifecycle order; an effect-application layer that breaks
at the integer/float boundary; a constraint that is honestly
chain-binomial-only; a shared scoring seam; a reset that breaks at
multi-cadence; and distinct algorithm bodies that should stay distinct. Each
"breaks at" cell names a _binding_ reason — a real property of a kernel or a
type — not an incidental implementation choice. Where the reason is "none" and
the verdict is "not built," that is remaining work; where the reason is a kernel
or type property, that is a seam to respect, not erase.

---

## Forward-compatibility: the closed loop and the mutation taxonomy

This architecture earns its keep only if the work _not_ in this push lands
cleanly on top of it. Two future consumers stress it hardest — the **observation
system** (the data layer, next; binding covered in the Layer-4 handoff) and
**reactive interventions** (the closed loop, after that;
[`2026-05-14-reactive-interventions-and-evsi.md`](2026-05-14-reactive-interventions-and-evsi.md)).
Reactive interventions are the sharper test, and stress-testing the seams
against them now is how we keep this round from painting the next one into a
corner. The same pass surfaces a domain-completeness question: are the _mutation
operations_ in the effect taxonomy enough for the polio/malaria models that
drive camdl?

### The closed loop — trajectory → observation → mutate trajectory

Every effect in Layers 2–4 flows one direction: the process advances, then
effects read or write, then the substep ends. A **reactive intervention** (ring
vaccination on case detection, ORV on a cholera threshold, polio SIA after AFP
detection, IRS when prevalence crosses a line) adds the one edge the
one-directional lifecycle does not yet express: a _read_ of an observed quantity
that _fires a write_, within the same simulation. That is closed-loop control,
and it is the right thing to test the taxonomy against because it touches every
layer at once.

It folds in cleanly, and the fold tells us the shape is right rather than
forcing a new subsystem. **A reactive trigger is an `Observe` + a threshold + an
enqueue.** The quantity it triggers on — weekly incidence, current prevalence —
_is a `StreamProjection`_, the same Interval/Instant read Layer 4 already
computes. So the read half reuses the observation projection machinery, the
write half reuses the intervention machinery (Layer 3), and the trigger is the
bridge. Reactive is not a parallel feature; it is the existing `Observe` and
`Intervene` wired head-to-tail. This is also why it belongs in _this_ topology:
the loop runs through the observation layer the obs system builds and back down
into the effect layer this work builds.

**The lifecycle gains one stage — `Sense` — between `Observe` and `Reset`:**

```
Advance < Intervene < Balance < Observe < Sense < Reset
                                   │        │       │
                          read+update    read     zero
                          observed-hist  +enqueue  flows
```

`Sense` must precede `Reset` because an incidence trigger reads the same
`FlowSum` window an `Interval` observation does, _before_ it is zeroed. It
reads; it never mutates compartments directly — on a threshold crossing it
**enqueues** a mutation at `t + after` (the implementation lag), and that
deferred mutation is consumed by a later substep's `Intervene` using the _same_
`Action` machinery as a scheduled intervention. With `after = 0` it fires at the
next substep boundary, one step of latency — which keeps the lifecycle an
acyclic read→write DAG (no read→write→read within one substep) and is arguably
more honest (you cannot act on information instantaneously). **Load-bearing
rule: the enqueued fire time must land on an existing substep boundary**
(`after` measured in grid steps, snapped onto the schedule's grid), _not_ an
arbitrary off-grid time. This is what lets the deferred queue live in
per-particle state while the `Schedule` stays immutable and shared — if a
path-dependent `after` could insert a _new_ boundary, different particles would
want different boundary sequences and the CRN invariant (N particles, identical
boundary walk) would break. Pinning firings to existing boundaries keeps the
queue a state read, not a schedule mutation.

The trigger predicate and the enqueued action are **independent in locus**: the
read can be a projection of patch `i` and the mutation can target patch `j`.
That is the canonical cross-patch reactive pattern — surveillance detects in one
district, the response (SIA / ORV) deploys in neighbours — the headline cVDPV2
and cholera use case. The architecture supports it for free (the register and
the action both see the global augmented state); the only thing to do is _not_
build a patch-local `Sense → Intervene` shortcut that would have to be
re-plumbed.

**The state is augmented with a policy register, and the loop is just extra
read/write edges over it.** Define the augmented state as
`(compartments, flow_accumulators, policy_register, observed_history)`, where
the register holds, per reactive effect, its `fired` flags, its deferred-action
queue, and its `t_last_fired` / `times_fired` bookkeeping. Each lifecycle stage
touches a defined slice:

| Stage     | reads                                             | writes                                |
| --------- | ------------------------------------------------- | ------------------------------------- |
| Advance   | compartments, **register** (for decay rates)      | compartments, flow_accumulators       |
| Intervene | compartments, register (deferred queue)           | compartments, register (dequeue)      |
| Balance   | compartments                                      | compartments                          |
| Observe   | compartments, flow_accumulators                   | observed_history                      |
| Sense     | compartments, flow_accumulators, observed_history | register (enqueue / fired / cooldown) |
| Reset     | —                                                 | flow_accumulators (zero)              |

The whole closed loop is those two `register` columns. Nothing in the kernel or
the schedule changes; the process advances the augmented state, and the register
is written only by effects and read by the process. This is the move that keeps
"the process stays special" honest — the rate evaluator now reads effect history
(for ITN/IRS decay: `itn_eff = e^{−λ(t − itn.t_last_fired)}`, the
malaria-feature case), but only through the register, which is part of the
Markov state.

**Fitting a reactive campaign is a non-goal, and the obstruction is specific to
PGAS — we state the boundary rather than mis-fit silently.** The hard case is
not the firing _density_ (the reactive proposal worried about factoring it; that
part is benign — given the trajectory, a latent-state firing is a deterministic
δ with mass 1). The hard case is PGAS's **ancestor sampling**. PGAS conditions
on a reference trajectory and, at each step, grafts the reference's _future_
onto a free particle's _past_, weighting by the transition density between them.
A reactive intervention with any memory — `once`, `cooldown`, `after`, or a
decay read of `t_last_fired` — makes the latent state _path-dependent_: the
`policy_register` is a function of the whole firing history. The precise
obstruction is that the register adds a **deterministic (δ) component** to the
augmented transition: `register_s` is a deterministic function of
`(register_{s-1}, state)`, so the augmented transition has no density
_dominating_ the reference's register path, and conditional-SMC ancestor
sampling has nothing to reweight against — the AS weight between "a free
particle whose register diverges from the reference" and "the reference just
after a firing" is zero. **Crucially, augmenting the particle state with the
register does _not_ fix this** — it is the standard failure of CSMC-AS on
partly-deterministic kernels (Lindsten/Jordan/Schön 2014 assume a dominating
transition density; a δ-component violates that), not an artifact of forgetting
to carry the register. A purely _memoryless_ trigger (`once = false`, no
cooldown/lag/decay) has no register, collapses to an ordinary state-dependent
kernel, and PGAS already handles it (the kernel is state-dependent via
`λ_SI = βI/N`); but every campaign that matters — one-shot SIAs, cooldown-gated
ORV, decaying ITN — carries memory. So **PGAS + a real reactive campaign is
unsupported**, and the right move is a clean capability error, not a silent
wrong posterior. (This is a design-time argument; the precise memoryless-enough
boundary should be pinned with a recovery test if reactive ever targets PGAS.
Note also `gh#187`: the PGAS producer path does not yet apply ordinary
_scheduled_ interventions correctly, so the reactive design should not be
finalized against PGAS while the non-reactive scheduled-intervention semantics
there are unsettled.)

The cost is narrow, because the _forward_ and _filtering_ paths have **no
grafting step**. Forward simulation propagates one trajectory; the bootstrap
particle filter, IF2, and PMMH propagate independent particles, each carrying
its own register — no reference future is welded onto a foreign past. So
reactive interventions are fully supported for **forward simulation and
PF-family inference (bootstrap PF, IF2, PMMH)**, and **PGAS rejects them** with:
_"reactive interventions are not supported under PGAS — ancestor sampling cannot
condition on path-dependent policy state; use `algorithm = pmmh | if2`, or
remove the reactive block."_ That is the honest boundary, and it is exactly the
kind of limitation to surface to the user rather than paper over.

This also shrinks the path-A/path-B distinction to where it actually bites —
_given a pinned semantics for `observed()`_. **In a fit, `observed(stream)`
reads the recorded datum `y_t`** (shared across particles, fixed), so the
trigger fires identically for every particle: it degenerates to a deterministic,
data-derived scheduled intervention — clean for the PF family, and (no register)
even PGAS-compatible. This must be pinned, because the predecessor reactive
proposal sketches `observed()` as a _fresh noisy redraw_
`ỹ ~ p(y | projection(x_t), θ)` — under _that_ definition the read depends on
the particle's own latent state plus fresh noise, firing becomes per-particle
and path-dependent, and the PGAS obstruction above returns. So the normative
choice: **`observed()` in a fit = the recorded datum, not a redraw.** The two
paths then differ only in **forward / EVSI** simulation, where path B draws
`y ~ p(y | projection)` to feed the trigger and path A reads the true state.
Path B is thus a forward/EVSI implementation detail, with one hook: the
`Observe` stage maintains an `observed_history` buffer (the most recent observed
value per stream), named in the obs system's scope so it lands for free. It is
**not** a fitting blocker; the fitting blocker is PGAS, and PGAS is out
regardless of path.

**Per-backend binding is the same analysis as the lifecycle.** The `Sense`
read + enqueue is pure and shareable across all backends. What binds is _when
within a step the threshold is crossed_: chain-binomial / tau-leap fire at the
substep boundary (natural — the reactive proposal makes them the initial
backends); ODE needs RK4 root-finding to locate the crossing time within a step
(standard ODE event detection); Gillespie needs a synthetic boundary at the
crossing (it has no substep). This is the _same_ binding constraint the
lifecycle table already records — Gillespie's event-driven kernel and ODE's
continuous state — so reactive does not introduce a new abstraction break; it
inherits the ones we already mapped.

One CRN caveat to state, because the register touches two _different_
invariants. The **swarm** CRN invariant (N particles, one immutable `Schedule`,
identical boundary walk) is preserved: the register lives in per-particle
`ParticleState`, the same class as `flow_accumulators`, and never touches the
shared `Schedule` (given the fire-on-existing-boundary rule above). But the
**paired-scenario** coupling (an `enable`/`disable` pair byte-identical
pre-intervention) is _not_ — a reactive firing that mutates state in one
scenario and not its pair changes subsequent propensities, hence the draw
sequence, hence the trajectories diverge. That is inherent to reactive control
(paired-seed CRN is not event-keyed) and acceptable, but it should be named:
reactive firing breaks paired-scenario byte-identity, not swarm CRN.

Net: reactive interventions fit the architecture without a new subsystem — one
`Sense` stage, one augmented-state register, one `observed_history` hook in the
obs layer, the existing per-backend binding, and a `REACTIVE` capability that
PGAS does not declare. The thing to do _now_, so it lands cleanly later, is to
(1) make `Stage` extensible to `Sense` (the proposed enum is a total order;
inserting `Sense` before `Reset` is additive), (2) reserve the
`Trigger::StateCondition` / `ObservedCondition` variants the topology already
sketches, (3) write the obs system's `observed_history` exposure into its scope,
and (4) treat reactive as a _forward + PF-family_ feature gated out of PGAS —
the same capability-flag pattern as `BALANCE`.

### The mutation taxonomy — what polio/malaria need, and the one real gap

The effect taxonomy's _write_ operations are
`Action ∈ {FractionTransfer,
AbsoluteTransfer, Set, Add}` over **compartments**,
plus balance. Walking the real operations a polio or malaria model performs on
the timeline:

- **SIA / mass vaccination, MDA, ring vaccination, culling, ORV** — `Transfer`
  S→V or I→treated, `Set`/`Add` for seeding/depopulation. **Covered.**
- **Importation / re-introduction / variant seeding** (a spark of infection,
  cVDPV2 emergence, imported malaria) — `Event` `Add` into the relevant
  compartment. **Covered.**
- **Pulsed demographic aging** — discrete annual cohort advancement
  (school-entry aging in measles/polio age-structured models, "everyone ages on
  a birthday"). This _is_ a timeline effect: a scheduled `Transfer` across age
  strata. **Covered** by the `Event`/`Intervene` transfer, but worth naming as a
  first-class pattern because it is ubiquitous and its lifecycle placement (does
  aging fire before or after transmission this step?) is a real modelling
  decision the `Stage` order now makes explicit. (Continuous births and deaths
  are **ordinary transitions** — a source-less or population-proportional inflow
  into the youngest susceptible class, an outflow from each compartment — and
  are already expressible in the model graph; they are not timeline effects and
  need nothing here. So the composite age-structured pattern that measles/polio
  actually use — newborns enter with maternal immunity (continuous birth
  inflow), wane M→S (a transition), and age annually (a pulsed `Event` transfer)
  — is fully covered: the first two are transitions, the third is a timeline
  effect. Only metapopulation movement stays out — that is a transition-graph
  coupling, a separate axis.)
- **ITN/IRS deployment with decay, post-campaign waning** — `Transfer` to a
  protected compartment, plus a rate that reads `t_last_fired` / `times_fired`.
  **Covered** once the policy-register read (the `InterventionState` `Expr`
  variant) lands — which the closed-loop augmented state already provides.
- **Parameter mutations (NPIs, vector control, reporting changes) — the one real
  gap, and we close it in this round.** A lockdown that drops `β`, a school
  closure that swaps the contact matrix, a vector-control round that lowers the
  biting rate, a surveillance ramp-up that raises the reporting rate `ρ` — these
  mutate a **parameter**, not a compartment, and the current taxonomy cannot
  express them. See the dedicated subsection below; the recommendation is to
  build it here, bundled with the Layer-3 apply-seam rework.

Everything else is covered or is a known DSL-ergonomics item (the `gh#171`
stratum-subset binder, so an SIA can target `age < 5` without enumerating
strata). So the taxonomy is complete for compartment mutations and one axis
short for parameter mutations.

### Parameter mutations — the design, and why to build it now

Today an `Action` writes a **compartment**: `FractionTransfer` /
`AbsoluteTransfer` move counts between compartments, `Set` / `Add` overwrite or
increment a count. There is no way to write a **parameter**. The consequences
are concrete and load-bearing for the exact models that drive camdl:

- a time-varying `β` (a lockdown, a behaviour change, a seasonal NPI) can only
  be smuggled in as a `forcing {}` table, which mixes a _policy_ into what
  should be a _covariate_ and cannot be made θ-dependent;
- a **reactive** `β` change — "drop transmission 40% when weekly cases exceed a
  threshold," the single most common NPI in COVID/measles/polio response
  analysis — **cannot be expressed at all**, because the reactive `Sense` stage
  can only enqueue a compartment mutation;
- malaria **vector control** (IRS/larviciding lowering the biting or emergence
  rate) and polio **routine-immunization** rate changes are parameter mutations,
  not compartment transfers.

This is `gh#50`'s windowed `set(param)`. The fix is one axis on the mutation
type — a **target** that is either a compartment or a parameter:

```rust
enum Target { Compartment(usize), Parameter(usize) }

struct Mutate {
    trigger: Trigger,                 // AtTimes (scheduled) | StateCondition (reactive)
    stage:   Stage,                   // Intervene
    target:  Target,                  // NEW axis; today implicitly Compartment
    action:  ParamOp,                 // Set | Add | Scale  (Transfer is compartment-only)
}
```

`Set(p, v)` overwrites `params[p] = v`; `Scale(p, f)` does `params[p] *= f` (the
natural NPI verb — "β drops to 60%"); `Add(p, d)` shifts it. `Transfer` stays
compartment-only (you cannot move "mass" between two scalars meaningfully). A
contact-matrix swap is a set of `Scale`/`Set` over the matrix's parameter
entries (or, once `gh#171`-style table addressing lands, one `Set` over a table
parameter).

**Why this is mostly clean — and the one structural exception that is a hard
prerequisite.** For a parameter read _directly_ by a rate (compiled to
`ResolvedExpr::Param`, evaluated as `ctx.params[idx]` every substep,
`resolved_expr.rs:254`), a mutation is a `Stage::Intervene` write to the
parameter vector and changes nothing structural — it does not touch the
stochastic draws (params are _read_ by rate construction, never drawn), the
schedule, the lifecycle order, or the kernel. It is a wider `Action` and a wider
apply step — close to the `{IntDelta | RealDelta}` apply-seam, with a third arm
`ParamDelta` (which writes a different destination — the param vector, exempt
from balance and the negative-count guard — so it is an honest distinct arm, not
a free fold).

**The exception is load-bearing: a parameter read through a `forcing {}` table
or a seasonal `TimeFunc` is baked into a cache at compile time**
(`compiled_model.rs:686` builds the cache from `default_params`;
`eval_time_func` takes no params, `propensity.rs:340`). A runtime
`params[idx] *= f` is therefore **silently inert** on that path — which is
exactly the seasonal-`β` / forcing-`β` NPI the axis is sold on, and which is the
open `gh#186` frozen-`TimeFunc`-param bug. The claim "the rate evaluator already
reads `EvalCtx.params` every substep" holds for the direct-`Param` path and is
**false** for the `TimeFunc` / forcing path.

**Decision for this round: ship the direct-`Param` path; guard the forcing path;
defer the `gh#186` fix.** Concretely:

- A mutation of a parameter read _directly_ by rates (`ResolvedExpr::Param`)
  ships now — scheduled and reactive `set`/`scale`/`add`, the full NPI /
  vector-control surface for non-seasonal models.
- A mutation whose target is consumed by _any_ `forcing {}` / `TimeFunc` is a
  **compile-time hard error** ("parameter `β` is read through a seasonal/forcing
  table, whose value is fixed at compile time; mutating it at runtime would be
  silently ignored — `gh#186`"). This is a new diagnostic over the model's
  frozen-cache param-refs; it does **not** touch the forcing machinery (forcing
  stays baked-at-compile as today). It converts the silent no-op into a clear
  refusal.
- The real `gh#186` fix — making forcing-consumed params live (re-resolve the
  `TimeFunc` / forcing caches, which _does_ break the "caches are immutable"
  assumption) — is deferred. Until it lands, seasonal-`β` NPIs are expressed by
  mutating a _direct_ multiplier (`rate = β · npi(t) · …` with `npi` a
  directly-read param), not by mutating the seasonal amplitude.

So nothing in the forcing layer changes this round; the axis is honest (works,
or errors — never silently no-ops), and `gh#186` is named as the gate to lifting
the guard later.

**The DSL surface** mirrors the existing intervention/reactive blocks —
scheduled and reactive both fall out of the same `Mutate` over a `Parameter`
target:

```camdl
interventions {
  lockdown : set(beta, 0.6 * beta_baseline) at 80 'days   # scheduled NPI
  reopen   : set(beta, beta_baseline)        at 140 'days
}

reactive_interventions {
  emergency_npi : trigger(when = weekly_cases > 500)
                  action  = scale(beta, 0.5)               # reactive NPI
                  cooldown = 30 'days
}
```

**The inference story — and the gradient caveat (broader than one action).** For
forward simulation and PF-family inference (bootstrap PF, IF2, PMMH) a parameter
mutation is trivial: mutate the param vector at the boundary; the filter
propagates as usual; nothing about the likelihood changes shape. The caveat is
the PGAS/NUTS **gradient**, and it is _not_ limited to `Set`. The PGAS gradient
holds the trajectory fixed and computes
`Σ_s ∂/∂θ log p(flows_s | counts_before_s, θ)` by plugging the live `params`
vector into the compiler-emitted symbolic `rate_grad` (`autodiff.ml`), which
treats each estimated param as a _free variable_. If a scheduled mutation
overwrites (`Set`), shifts (`Add`), or scales (`Scale`) an **estimated** param
mid-run, the symbolic chain rule is corrupted: `rate_grad` has no representation
of "this param's value was replaced/scaled by an intervention at `t = s*`," so
downstream of the firing it returns the wrong derivative (`Set` → should be 0,
returns ≠ 0; `Scale` → misses the factor `f`). So **any** scheduled mutation of
an estimated param breaks the gradient, not just `Set`. The correct gate:
**reject any scheduled `Set`/`Add`/`Scale` of an estimated parameter under
PGAS-gradient (NUTS) methods**, and settle the boundary with an FD-gradient
recovery test before relaxing it. Two cases stay clean: mutating a
_non-estimated_ param (a fixed baseline), and a `Scale(p, f)` where `p` is fixed
and the estimated factor `f` enters the rate through the _normal expression
path_ (then the gradient flows through `f` legitimately). For the reactive case
no separate gate is needed — reactive is already out of PGAS entirely.

So parameter mutations are not a deferred "future seam" — they are a pure
widening of the mutation type that the Layer-3 rework should absorb, they unlock
NPIs / vector control / reactive-β for forward sim and PF-family fitting
immediately, and the only inference subtlety (discontinuous `Set` of an
estimated param under PGAS gradient) is the one the architecture already knows
how to gate.

---

## Status against the code — landed, not-yet, fiction-to-correct

So the next proposal/implementation sharpens against reality, not the
predecessor proposals' aspirations. Verified against the branch
(`feature/unified-timeline`, `824fca4`):

**Landed and sound.**

- `Schedule` spine routed through all four forward backends and all four
  filters; the `substeps` iterator and `window_end` shared by the PF family;
  `drain_outputs` shared by the forward backends. Byte-identical where it
  matters.
- The `dt.min(boundary − t)` ULP-robustness fix and the `s·dt` drift-free
  convention, with the eight PGAS density/gradient sites reading the realized
  record.
- The `(algorithm × obs_alignment)` gate (`resolve_obs_alignment`) —
  validation-only; PGAS `step_policy` hard-pinned to `Snap`.
- Scoring unified on one summation loop (`gh#139`).
- Gillespie's post-mutation propensity recompute (the SSA obligation) — correct.

**Not yet built (real remaining work).**

- The shared lifecycle order + apply (`Layer 2`); tau-leap/ode/gillespie still
  invert event-vs-intervene and read post-intervention state. No cross-backend
  agreement test (`M1`).
- The `{IntDelta|RealDelta}` apply seam; ODE still round-trips `f64→i64→f64`
  (`m7`).
- The per-stream `ResetWindow`; the reset is still global (`M3`).
- The runtime sub-`dt` collision guard (`M2`).
- The CPM silent-fallback hardening (`M6`).
- The canonical-lifecycle user-facing doc/figure (`M5`).
- `TemporalKind` as a first-class type (currently a `matches!` predicate).
- The `Target = Parameter` axis (direct-`Param` path) + its compile-time guard
  against mutating a forcing-consumed param; the `gh#186` cache fix that would
  lift the guard.
- The `balance + Exact` decision (no guard exists today) — a prerequisite of the
  tau-leap fold and of running chain-binomial under `Exact` with a `balance {}`
  model.
- The per-cell `Counted` denominator channel on the scoring seam (the malaria
  case).

**Fiction in the predecessor proposals to correct (doc fixes).**

- "One canonical lifecycle, enforced" — not enforced; two backends invert it.
- "Consolidate the two capability gates into one seam" — three honest gates,
  keep separate (`M4`).
- "`ResetWindow` / per-stream reset re-homes to the timeline work" — it did not;
  obs-data owns it (`M3`).
- The heavy `Boundary`/`Trigger`/`Stage`/`Effect`/`EffectCaps` type system — the
  code right-sized to a thin boundary cursor; the heavy ADT is not the target.
  Keep the thin `Schedule`; add only the effect ADTs that earn their place
  (`Observe`/`Event`/ `Intervene`/`Constrain`/`ResetWindow`, `TemporalKind`),
  and only where they replace a hand-rolled hazard.

**Pre-existing minors** (not this effort's regressions): dead `grid` field
(`m1`), dead effect-cursor bookkeeping in chain-binomial (`m2`), dead
`_tolerance` parameter threaded through eight call sites (`m3`), two builders
bypassing `substep_time` (`m4`), the weak CRN test (`m5`), no `--obs-alignment`
CLI flag (`m6`).

---

## Position on the consolidation effort

Asked directly: **I agree with consolidating, and the seam choices that landed
are right — but the framing needs one correction, and the centerpiece claim was
over-sold.**

What is right and should not be touched: the thin `Schedule` (not the heavy
ADT), the shared `substeps` iterator with distinct filter bodies, the
chain-binomial-only balance, the three separate gates, the conservative
PGAS-stays-`Snap` default. These are all "consolidate the substrate, stop at the
seam" done correctly. The review's verdict — the spine extraction is sound and
byte-identical where it matters — holds.

The correction: **"unified timeline" bundled two separable consolidations, and
the proposals' centerpiece was the one that did not land.** The _timing_ spine
(Layer 0) is genuinely done and genuinely shared. The _effect-order lifecycle_
(Layer 2) — the "SLiM tick cycle" the proposal led with — is _not_ enforced; it
is the one place a cross-backend correctness divergence still lives (the
event/intervention inversion), and the per-backend baselines actively bless that
divergence. The honest re-shaping is to **separate the two axes**: declare the
timing spine complete, and make the lifecycle-order extraction (a pure
`apply_effects_in_lifecycle_order` over `(snapshot, current, effects)`, routed
through all four backends, pinned by a _hand-computed cross-backend agreement
fixture_) the actual remaining headline work — together with the
`{IntDelta|RealDelta}` apply seam that the same work needs to stop ODE
quantizing.

I do _not_ think the effort is misdirected or that the prior work should be
unwound. The substrate consolidation is exactly the right lever for the bug
surface, and it landed. What is needed is to name the unfinished half precisely
(so obs-data is not sharpened against fiction), respect the three binding
constraints the topology surfaces (Gillespie's event-driven kernel, the `i64`
`Action`/`f64` ODE break, the chain-binomial-only balance), and close the four
live correctness gaps (`M1`–`M3`, `M6`) before calling it consolidated.

## Recommended sequencing

Ordered so each step is gated and nothing sharpens against fiction:

1. **Correctness gates first.** ✅ _Landed (`M2` `6c80e62`, `M1` `ec4e7d1`)._
   The sub-`dt` collision runtime guard (`M2`, red-first — reject two obs that
   round to one substep). And the `M1` canonicalization, which splits into two
   halves: (a) the **event-vs-intervention order** — events fire before
   interventions, reading the pre-intervention state, in tau/ode/gillespie —
   _landed here_, with an ordering-agreement fixture (an event coincident with
   an intervention that reads the event's compartment, all four backends
   agreeing); (b) the deeper **event-vs- transition fusion** (events read the
   _pre-transition_ snapshot and fuse with the draws, as chain-binomial does) is
   an O(dt) residual _deferred to step 2_, where the shared apply delivers it
   uniformly. The CPM hard-error (`M6`) lands next.
2. **The Layer-2 extraction.** Lift
   `apply_effects_in_lifecycle_order(current,
   effects)` out of the four step
   functions into one shared, fixture-tested function; route all backends
   through it (events-read-snapshot and the post-apply propensity-recompute are
   caller obligations). This delivers the **event-vs- transition fusion**
   deferred from step 1, and earns the **strong cross-backend agreement
   fixture** (a value-dependent event coincident with live transition flow _and_
   an intervention, all backends identical) that the ordering-only fixture
   cannot reach. This is the lifecycle the proposal promised and the bug surface
   it targeted. **Retire tau-leap here**: once chain-binomial runs the shared
   apply under `StepPolicy::Exact`, prove `chain-binomial + Exact == tau-leap`
   byte-for-byte on the corpus, repoint goldens, and delete the backend (the
   equivalence proof doubles as the StepPolicy/lifecycle validation gate — see
   the
   [backend-rationalization note](../notes/2026-06-06-backend-rationalization.md)).
3. **The Layer-3 apply seam — including the parameter-mutation `Target` axis.**
   Collapse the two `Action` interpreters (`apply_intervention` /
   `inject_event_deltas`) and their eight quantization sites into one `apply`
   over `{IntDelta | RealDelta | ParamDelta}`, eliminating ODE's round-trip and
   the events-on-real-compartments drop, **and** adding the
   `Target = Compartment |
   Parameter` axis (the NPI / vector-control /
   reactive-β unlock) in the same rework — one IR change, one golden
   regeneration. Gate **any** `Set`/`Add`/`Scale` of an _estimated_ param under
   PGAS gradient (not just `Set`), and make the `gh#186` frozen-`TimeFunc` fix a
   prerequisite for mutating any forcing-consumed param. **Build the per-stream
   `ResetWindow` here too** — it is a `Stage::Reset` write co-located with the
   apply-seam, and it is the hard dependency of the observation system push that
   comes next (do not leave it in the deferred tier).
4. **Doc reconciliation.** Ship the canonical-lifecycle figure to the language
   spec / user-features (`M5`); correct the gate-consolidation and ResetWindow
   claims in the predecessor proposals (`M3`, `M4`); add `TemporalKind`.
5. **Then** the genuinely-deferred increment on its evidence gate: exact-PGAS
   (the external-oracle battery — pomp cross-check, Richardson `dt`-ladder, FD
   gradient, posterior non-drift — gated on `gh#175`).
6. Minor cleanup (`m1`–`m6`): wire or delete the dead `grid`, drop the dead
   effect-cursor and `_tolerance`, route the two builders through
   `substep_time`, strengthen the CRN test, add the `--obs-alignment` flag.

The cross-cutting requirement under all of it: **at least one cross-backend
agreement invariant** — two backends, same result on a coincident-effect model
where they legitimately should agree. Every gate today pins each backend
independently, so a divergence gets blessed rather than caught — which is
precisely what undercuts the "one surface → fewer bugs" thesis the whole effort
rests on.
