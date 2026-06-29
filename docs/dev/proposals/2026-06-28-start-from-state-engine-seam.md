# Start-from-state engine seam — resuming a forward simulation from an injected mid-run state

Status: spec — implement against this. Relates-to:
`2026-06-25-counterfactual-contrasts.md` (the conditioned fork that needs this —
its prerequisite #2), `2026-06-28-keyed-joint-param-trajectory-output.md` (the
`(θ, X)` output that supplies the state to inject). High-risk:
inference-adjacent engine change — read the full `chain_binomial` substep loop
before editing any part of it, and gate every change on the splice invariant.

## Problem

The forward engine always builds its initial state from the model at the start
time. In `chain_binomial::run_chain_binomial_with_observer`:

```rust
let (mut int_s, mut real_s) = model.initial_state(params)?;   // chain_binomial.rs:172
...
let mut t = cfg.t_start;                                        // :211
let schedule = Schedule::snap_forward(cfg.dt, cfg.t_end,        // :201
                  OutputTimes::from_model(model)?, EffectTimes::from_model(model, params)?);
let mut cursor = Cursor::default();                             // :207
```

There is no way to say "begin at time T\* with _this_ compartment state and
simulate forward to `t_end`." The config carries `t_start`/`t_end`/`dt` but no
state. This blocks the conditioned counterfactual fork: that fork must branch
both arms from the fit's inferred latent state X_i(T\*) at the intervention time
(`2026-06-25-counterfactual-contrasts.md`), and there is nowhere to inject it.

## Goal

A seam that runs a forward simulation from an injected `(state, T*)` and resumes
the substep loop at T\* — producing a trajectory over `[T*, t_end]` that is
**identical to the tail of a continuous run that reached T\* the long way**. The
seam is the engine half; the `(θ, X)` output supplies the state, and `compare`
(contrasts) is the only caller in v1.

## The correctness oracle: the splice invariant

The change is only safe if it has a byte-exact oracle. It does — a continuous
run can be **spliced** at any grid time T\* and the two halves must reassemble:

```
run(t0 → t_end, seed)   ≡   run(t0 → T*, seed)  ++  resume(state@T*, rng@T*, T* → t_end)
```

i.e. inject the continuous run's _exact_ compartment state AND its RNG state at
T\*, resume, and get a byte-identical continuation. This is the strongest
possible test (the splice is transparent), and it pins every hazard below at
once: get the cursor/clock/flows wrong and the spliced tail diverges from the
continuous tail on the first substep.

Two consumers, two RNG policies:

- **Splice-invariant test:** inject state **+ RNG state** → byte-identical
  continuation (the oracle).
- **Contrasts (the real use):** inject state **+ a fresh per-arm seed** — each
  arm re-rolls its own forward noise from X_i(T\*) (CRN at the fork; the
  post-fork streams desynchronize by design, see the contrasts doc). The
  RNG-restore path is test-only.

## The hazards the seam must re-seat (why it is high-risk)

Resuming at T\* is not "set `t = T*`." Four pieces of run state are keyed to the
start and must be re-seated, or effects double-fire / forcings evaluate at the
wrong time / incidence is mis-attributed:

1. **The schedule cursor.** `Cursor::default()` starts before the first
   boundary. Resumed at T\*, every output/effect boundary `< T*` must be treated
   as already passed (NOT re-fired), and a boundary _exactly at_ T\* must fire
   once. This is the gh#216 firing hazard and the
   `2026-04-17-chain-binomial-double-fire.md` class — the engine fires
   interventions inside `step_one` keyed on `fire_steps`, so re-seating must
   reconcile the cursor and the fire-step indexing together.
2. **The substep clock.** Rate/forcing is evaluated at the drift-free grid time
   `t_grid = schedule.substep_time(cfg.t_start, s)` (`chain_binomial.rs:249`;
   `2026-06-05-substep-time-sdt-convention.md`). With `t_start = T*` and `s = 0`
   the grid time is `T* + s·dt`, so a _calendar-time_ forcing evaluates at the
   correct absolute time — but only if T\* lands on the grid (v1 requires it;
   off-grid is a located rejection, decision below).
3. **The flow accumulators.** Incidence is accumulated per output interval and
   zeroed at each emitted snapshot (`current_flows.reset()`,
   `chain_binomial.rs:227`). A resumed run must start with **zeroed** flows at
   T\*, so post-fork incidence is measured from T\* and never carries a pre-fork
   tail (the same per-observer reset discipline as the inference projections).
4. **`fire_steps` are ABSOLUTE — do not re-index.** Intervention fire steps are
   `round(t_fire/dt)`, anchored at time 0, not `t_start` (`time.rs`;
   `model.resolve_fire_steps(cfg.dt, params)`, `:180`). With `t_start = T*` and
   `s = 0` the substep clock `t_grid` stays absolute, so the existing fire steps
   already fire at the right absolute steps and pre-T\* ones auto-vanish
   (`current_step` only increases). Re-indexing them would _break_ firing — the
   re-seat is purely the cursor + clock + flows, NOT `fire_steps`.
5. **The fork intervention at exactly T\* (silent-wrong if missed).**
   Interventions fire on the substep _ending_ at their time (`due_effects` is
   tested at `t_grid + dt`); the resumed loop's first substep (`s = 0`,
   `t_grid = T*`) tests effects due at `T* + dt`, so an intervention scheduled
   _at_ T\* is never tested → arm B ≡ arm A → "cases averted" = 0, silently.
   Decision (below): a state-modifying intervention at T\* is applied to the
   injected `X_i(T*)` _before_ the loop; this is the only fired-at-T\* path, and
   it is what makes the fork instant faithful.
6. **The reactive subsystem is unrepresentable in `(state, rng)`.**
   `ReactiveAgenda` carries mid-run state a resumed `(int_s, real_s, rng)`
   cannot reproduce — `obs_history` (windowed-trigger lookback),
   `last_fired`/`times_fired` (`once`/`cooldown` gating), a `pending` effect
   heap, partial `interval_flows`, and a second RNG stream (`obs_rng`).
   `chain_binomial` **grants** `REACTIVE_INTERVENTIONS` on the forward path, so
   reactive models run here. Decision (below): v1 **capability-gates reactive
   (and attached observers) out at the seam** with a located error — silence
   would be a matrix gap (gh#187 class).

## Sketch of the surface

The injected state rides in the config (or a sibling entry point), kept `None`
on every existing path so non-spliced runs are byte-identical:

```rust
struct StartState {
    t_star:   f64,                 // the resume time (a grid instant in v1)
    int_s:    IntState,            // injected compartment counts at T*
    real_s:   RealState,           // injected real-compartment values at T*
    rng:      Option<RngState>,    // Some → splice-invariant test; None → fresh-seed (contrasts)
}
// run_chain_binomial_with_observer gains an `Option<&StartState>`; None ⇒ today's
// `model.initial_state(params)` + `t = cfg.t_start` + `Cursor::default()`, unchanged.
```

When `Some`, in order: reject the run if the model uses reactive interventions
or attached observers (located capability error, hazard #6); reject an off-grid
T\* (located error, decision below); skip `initial_state` and seed
`int_s`/`real_s` from the injection; apply any caller-supplied state-modifying
intervention scheduled _at_ T\* to that seeded state (hazard #5); set `t = T*`
via a cursor that _starts_ at T\* (`Schedule`/`Cursor` resume constructor, not a
loop fast-forward); zero the flow accumulators; and (if `rng.is_some()`) restore
the RNG state instead of seeding fresh. `fire_steps` are left absolute (hazard
#4). Every other path passes `None` and is byte-identical to today.

## Decisions (v1)

All resolved — this section is what the implementer follows.

- **Validation oracle — the splice invariant.** The seam is correct iff a resume
  with the continuous run's _restored RNG state_ reproduces the continuous tail.
  Byte-identical for integer-dt; agrees-to-ULP for fractional-dt
  time-inhomogeneous forcing (re-anchoring the drift-free clock at
  `window_start = T*` diverges by ULPs there). The strongest available oracle,
  and what makes a high-risk engine change safe to land.
- **`Option<&StartState>`, `None` on every existing path.** Non-spliced runs are
  provably byte-identical (same discipline as `--sweep`/`DesignCoords::none()`
  and the per-eval-staging no-op guards). The per-eval scratch is a non-hazard
  (`stage_per_eval` reads no `Time`/`Dt` — verified inert), so it needs no
  re-seat.
- **Re-seat cursor + clock + flows together; leave `fire_steps` absolute.** The
  gh#216 / double-fire history is exactly the cost of re-seating one and not the
  others. `fire_steps` are `round(t_fire/dt)` anchored at time 0 (hazard #4) —
  already correct under an absolute `t_grid`; re-indexing them would break
  firing.
- **Fork intervention at T\* → apply to the injected state before the loop.** A
  state-modifying intervention scheduled _at_ T\* is applied to `X_i(T*)` before
  resuming (hazard #5); dynamics/parameter changes take effect via the arm's
  scenario from T\* forward. This is the only fired-at-T\* path and makes the
  fork instant faithful — a naive resume would silently never fire it.
- **Reactive interventions + attached observers → capability-gated out.** Their
  mid-run `ReactiveAgenda` state (`obs_history`, `last_fired`/`times_fired`, the
  `pending` heap, partial `interval_flows`, the `obs_rng` stream) cannot be
  reconstructed from `(int_s, real_s, rng)` (hazard #6). v1 rejects such a model
  at the seam with a located capability error rather than forking it silently
  wrong; capturing the agenda is a post-v1 extension.
- **T\* must be on the output-emit grid.** Flow accumulators reset only at
  output emits, so the spliced tail matches the continuous tail only when T\* is
  an output-emit time. v1 requires T\* on that grid and rejects an off-grid T\*
  with a located error — never a silent snap to a neighbour. The `++` splice
  resolves the duplicated/disagreeing T\* boundary row by the resumed run's
  initial-row convention (the resumed run emits its seeded state as its first
  row; the continuous run's accumulated T\* row is the join point).
- **Cursor re-seat via the gh#233 boundary authority.** Re-seat with a
  `Schedule`/`Cursor` constructor that _starts_ at T\* (testable in isolation),
  not by advancing a default cursor in an O(boundaries) loop. Reuse the existing
  "boundary authority" seam rather than adding a parallel one.
- **ODE forks by recomputation, not injection.** A deterministic backend's
  X(T\*) is the θ-determined state, so an ODE arm re-integrates from `t_start`
  to T\* (the contrasts doc's `LatentPath::Deterministic` path) and never enters
  this seam. The state-injection seam is **chain_binomial-only in v1**;
  Gillespie is not an inference backend and is out of scope.
- **Identity-neutral.** A spliced run is a derived/forward artifact: the
  injected state is an input to a forward sim, hashed like any other
  param/scenario input, not folded into a fit hash.

## Validation

The splice invariant is the gate; every test below is a splice-or-reject check,
not a tolerance dial:

1. **Splice byte-identity (integer-dt).** A continuous `run(t0 → t_end, seed)`
   equals `run(t0 → T*) ++ resume(state@T*, rng@T*, T* → t_end)` byte-for-byte
   on a chain-binomial model with interventions straddling T\*. The headline
   correctness test.
2. **Real compartments + balance.** Repeat (1) on a model with real compartments
   (RK4) and a `balance {}` constraint, confirming the injected real state needs
   no extra reconciliation (balance is applied per substep, so it
   self-corrects).
3. **Reactive / attached-observer rejection.** A reactive model entering the
   seam produces the located capability error, not a silent fork (gh#187-class
   guard).
4. **Off-grid T\* rejection.** A T\* between output emits produces the located
   error, never a silent snap.
5. **No-op identity.** The whole existing forward/inference suite is
   byte-identical with the seam present and `StartState = None` (goldens +
   `ir/expected` unmoved).
