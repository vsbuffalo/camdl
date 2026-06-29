# Start-from-state engine seam — resuming a forward simulation from an injected mid-run state

Status: draft (a captured design, not yet a spec to implement against — open
questions are intentional). Relates-to: `2026-06-25-counterfactual-contrasts.md`
(the conditioned fork that needs this — its prerequisite #2),
`2026-06-28-keyed-joint-param-trajectory-output.md` (the `(θ, X)` output that
supplies the state to inject). High-risk: inference-adjacent engine change.

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
   correct absolute time — but only if T\* lands on the dt grid. Off-grid T\* is
   an open question (below).
3. **The flow accumulators.** Incidence is accumulated per output interval and
   zeroed at each emitted snapshot (`current_flows.reset()`,
   `chain_binomial.rs:227`). A resumed run must start with **zeroed** flows at
   T\*, so post-fork incidence is measured from T\* and never carries a pre-fork
   tail (the same per-observer reset discipline as the inference projections).
4. **`fire_steps`.** Intervention fire steps are resolved as integer step
   indices from `t_start` (`model.resolve_fire_steps(cfg.dt, params)`, `:180`).
   Re-anchoring `t_start` to T\* re-indexes them; an intervention scheduled
   _before_ T\* must not reappear, and the fork intervention at T\* must land on
   step 0 (or its correct offset).

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

When `Some`: skip `initial_state`, seed `int_s`/`real_s` from the injection, set
`t = T*`, fast-forward the cursor past every boundary `< T*`, zero the flow
accumulators, and (if `rng.is_some()`) restore the RNG state instead of seeding
fresh.

## Open questions (intentional — this is a draft)

- **Off-grid T\*.** v1 should require T\* on the dt grid (the substep clock and
  `fire_steps` both assume it). Is that acceptable for the fork instant
  (`origin + 20 'weeks` with a weekly dt — yes; a daily-dt model forking at a
  weekly instant — also yes; an arbitrary `date(...)` off the grid — reject with
  a located error)? Decide the grid-snapping policy explicitly.
- **Cursor re-seat: fast-forward vs a `Schedule::resume_at(T*)`.** Re-seating by
  advancing a default cursor in a loop is O(boundaries) and easy to get wrong; a
  dedicated `Schedule`/`Cursor` constructor that _starts_ at T\* is cleaner and
  testable in isolation. Reach for the existing `Cursor`/`Schedule` seam
  (gh#233's "boundary authority") rather than a parallel one.
- **Real compartments + balance.** RK4 for real compartments and the `balance`
  constraint both run inside the substep; confirm an injected real state needs
  no extra reconciliation (balance is applied per substep, so it should
  self-correct — verify with the splice invariant on a model that has both).
- **Backends beyond chain_binomial.** ODE is deterministic given θ, so its
  "resume" is just re-integrating from X(T\*) = the θ-determined state — does
  the seam unify the two, or is ODE handled by the deterministic-fork path in
  the contrasts doc (`NotFilterable::Deterministic`)? Gillespie is not an
  inference backend, so out of scope.
- **Identity.** A spliced run is a derived/forward artifact, not a re-keyed fit
  leaf — confirm it stays identity-neutral (the injected state is an input to a
  forward sim, hashed like any other param/scenario input, not folded into a fit
  hash).

## Decisions recorded

- The seam is validated by the **splice invariant** (resume-with-restored-RNG ≡
  continuous tail, byte-identical) — the strongest available oracle, and the
  thing that makes a high-risk engine change safe to land.
- The injection is `Option<&StartState>` kept `None` on every existing path, so
  non-spliced runs are provably byte-identical (the same discipline as the
  `--sweep`/`DesignCoords::none()` and per-eval-staging no-op guards).
- Cursor/clock/flows/`fire_steps` are re-seated together, not piecemeal — the
  gh#216 / double-fire history is exactly the cost of re-seating one and not the
  others.
- v1 requires T\* on the dt grid; off-grid is a located rejection, not a silent
  snap.
