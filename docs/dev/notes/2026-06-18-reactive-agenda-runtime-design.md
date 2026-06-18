# Reactive agenda runtime (PR2) — forward chain-binomial design

Date: 2026-06-18 Project: camdl (gh#204) Tags: reactive-interventions,
chain-binomial, agenda, forward-sim

## Context / question

PR1 landed the DSL/IR for reactive interventions and a hard capability rejection
(`REACTIVE_INTERVENTIONS`, granted by no backend). PR2 makes **forward
chain-binomial** actually run them: at each observation time, evaluate each
reactive policy's `when` predicate against the observed data, and — if it fires
— enqueue its effect to be applied `after` a lag, through the existing
`effects.rs` resolver. Scope is `SharedExogenous` only (no particle-local, no
inference; those are PR4/PR5).

This note is the types-first design, grounded in the current chain-binomial
runtime (verified 2026-06-18, citations inline).

## The load-bearing finding: forward sim computes no observations today

`run_chain_binomial_with_observer` (`chain_binomial.rs:145–299`) steps `dt`,
draws transitions in `step_one`, accumulates flows into `current_flows`, and
drains output snapshots. **It never evaluates an observation projection** —
`incidence(...)`, `CurrentPop`, etc. are computed only on the _inference_ path
(the particle filter's measurement model). The `Schedule.obs_times` field is
empty for forward Snap schedules (`schedule.rs:183–195`); observation emit times
live only in the IR (`model.observations[i].emit_schedule`).

Consequence: a trigger `sum_observed(weekly_cases, window = 28 'days)` has no
value to read in forward sim today. **PR2 must add, at each emit time, both the
projection eval AND a realized observation draw** (`sample_obs_resolved` on a
dedicated RNG stream — see the `observed(...)` section). This is the bulk of
PR2's new surface; the agenda/firing half reuses existing machinery.

**Constraint (do NOT reuse `current_flows`).** `current_flows`
(`chain_binomial.rs`) is reset on the **output** drain — it accumulates flow
over the _output_ interval, NOT the _observation_ interval. Observation streams
carry their own `emit_schedule` (e.g. `every 7 'days`), which need not coincide
with output times, and an interval projection (`incidence(...)`) must read the
flow **over the observation interval** (since that stream's last emit). Reading
`current_flows` at an emit time would silently give the since-last-output flow —
wrong whenever the cadences differ. PR2 therefore needs **a per-observation-
stream interval accumulator, reset at each stream's emit time** (the same
interval semantics the inference measurement model already applies via its
per-observation flow reset — `TemporalKind::Interval`). This belongs in the
shared observation evaluator (decision #3) so forward reactive and inference use
one accumulator definition, never two.

Fallback if per-stream accumulators prove intractable for PR2: a phase-1
validation that every reactive-trigger stream's emit schedule **aligns with
output times**, and an explicit rejection (clear diagnostic) of any schedule
that does not — never a silent fall-through to output accumulation. Strongly
prefer the per-stream accumulator.

## What `observed(...)` reads — the REALIZED observation draw, on a separate RNG

Decision (locked): **`observed(stream)` reads the realized random observation
`Y ~ p(· | state, params)`, NOT the mean `E[Y|state,params]` and NOT the raw
projection.** Drawn via `sample_obs_resolved` (sibling of
`eval_obs_mean_resolved`, `inference/obs_model.rs`) at each emit time.

**Why realized, not the mean.** The trigger feeds a nonlinear threshold
functional `fire = 1{Y ≥ c}`. With the realized draw the policy fires with
probability `P(Y ≥ c)`, a ramp whose steepness is set by `Var(Y)` — surveillance
noise — which is precisely what makes native EVSI nonzero (better surveillance
shifts the response distribution). The mean-plugin fires iff `1{E[Y] ≥ c}`, a
hard step with no dependence on `Var(Y)`: it deletes the surveillance-detection
stochasticity, so it is a _different_ (improper) model, not a smoothed one. No
Rao-Blackwell argument saves the mean — `E[1{Y≥c}] = P(Y≥c) ≠ 1{E[Y]≥c}`
(Jensen, not RB), and `Y` is a structural random input to the generative
process, not an estimator being conditioned. The correct "integrate out `Y`"
object `E_Y[outcome|X]` needs the full noise model — which Monte Carlo realizes
by drawing `Y`.

**Why CRN and the equivalence oracle still hold — the separate RNG stream.** The
obs draws run on a **dedicated RNG stream** (precedent: the lineage observer's
own stream, `chain_binomial.rs`), independent of the dynamics ChaCha stream.
Therefore:

- the dynamics draws are consumed identically with or without a reactive policy
  ⇒ paired-seed / CRN coupling is preserved;
- the equivalence oracle is byte-identical in the dynamics — a reactive run
  (seed S) firing at realized `T`, and the same model with the policy replaced
  by a scheduled intervention `at [T]` (seed S), share the dynamics stream
  consumed identically (the reactive run's obs draws are on the _separate_
  stream; the scheduled run touches neither), and apply the same effect at `T`
  via the same resolver ⇒ identical trajectory.

`observed(s)` = the current interval's realized draw;
`sum_observed(s, window =
D)` = the sum of realized draws whose emit times fall
in `(t-D, t]`. The agenda keeps a per-stream ring of `(emit_time, Y)` and folds
the reducer over the window. The raw projection (true incidence) and
`latent(...)` (raw state) are differently-named future primitives, NOT
`observed`. `rolling(...)` is **out of PR2 scope** (PR1 rejects method syntax;
PR2 is `observed` + `sum_observed` only).

## Types

```rust
// sim/src/inference/.. NO — forward; sim/src/reactive.rs (new)

/// One reactive policy's runtime state across a single forward run.
struct PolicyState {
    last_fired: Option<f64>,   // for `cooldown`
    times_fired: u32,          // for `once` and readback (future)
}

/// A future effect discovered at an observation time, due at `t = trigger + after`.
struct PendingEffect {
    t: f64,
    policy_idx: usize,         // index into the reactive policies
    seq: u64,                  // stable tie-break: enqueue order
}

/// The forward agenda: a min-heap of pending effects + per-policy state +
/// per-stream realized-observation history for the trigger reducers.
struct ReactiveAgenda {
    pending: BinaryHeap<Reverse<PendingEffect>>,
    policies: Vec<PolicyState>,
    obs_history: Vec<Vec<(f64, f64)>>,   // per stream: (emit_time, realized Y)
    seq: u64,
}
```

`ReactiveAgenda` lives in the backend run state (NOT in `Schedule`, which is
immutable and shared — `schedule.rs` CRN invariant). It is per-run, single
forward trajectory. The realized obs draws are made on a **dedicated RNG
stream** (a `StatefulRng` seeded deterministically from the run seed, separate
from the dynamics RNG — same pattern as the lineage observer), held in the
backend run state alongside the agenda so the dynamics ChaCha stream is consumed
identically with or without a reactive policy.

## Where it hooks into the chain-binomial loop

Two hooks, both in `run_chain_binomial_with_observer`'s substep loop:

1. **Merge due reactive effects into the scheduled due-batch — each substep.**
   Decision (locked): a reactive firing is NOT an arbitrary extra mutation; it
   goes through the **same due-batch lifecycle as a scheduled intervention**, so
   it gets `apply_intervention_effects`, post-advance semantics, balance, and
   the negative-count checks for free. chain is Snap: scheduled effects are
   collected by `due_effects(...) → effect_batch.intervention_idx` then applied
   in `step_one`'s INTERVENE stage via `apply_post_advance`
   (`chain_binomial.rs:245`, `602–611`). A reactive policy already has an
   `iv_idx` in `model.interventions` (kind = Scenario, fire = Reactive), so the
   agenda's contribution each substep is: **append the `iv_idx` of every policy
   whose pending fire time is due at this step to
   `effect_batch.intervention_idx`.** It then flows through the identical path —
   one lifecycle, no fork. Mark `last_fired` / `times_fired` after.

   The "due at this step" key matches the scheduled effects' `round(t/dt)` key
   (`time_to_step(fire_time, grid_dt) == current_step`) so reactive and
   scheduled firings that land on the same step compose deterministically.

   **`after = 0` (post-observation immediate).** A trigger evaluates at obs time
   `t` AFTER this step's INTERVENE stage already ran (lifecycle below), so a
   `t + 0` effect cannot fire at the same step — by construction it fires at the
   NEXT step's INTERVENE. Define `after = 0` as exactly that: it cannot affect
   the observation/output at `t`, only the next interval. If the keying proves
   ambiguous to implement cleanly in PR2, **reject `after = 0` with a clear
   diagnostic** ("zero lag not yet supported on the reactive runtime; use
   `after > 0`") rather than shipping ambiguous semantics.

2. **Evaluate triggers + enqueue — at each observation emit time.** After the
   process has advanced to the emit time, evaluate the projection (new
   forward-sim eval) and **draw the realized obs `Y` via `sample_obs_resolved`
   on the dedicated obs RNG stream**; push `(emit_time, Y)` to
   `obs_history[stream]`; then for each policy, evaluate `when` over the history
   (the `TriggerExpr` interpreter: `observed`/`sum_observed` reducers vs
   `Const`/`Param` threshold, combined by and/or/not). If true AND not disabled
   by `once` (times_fired == 0) AND past `cooldown`
   (`t - last_fired ≥ cooldown`): enqueue
   `PendingEffect { t: emit_time + after, policy_idx, seq }` and write a
   `reactive_log.tsv` row.

Lifecycle order at a shared timestamp (proposal's Lifecycle Rule, realized):
advance → events → scheduled interventions → balance → output → **compute
projection** (over that stream's obs interval) → **draw the realized obs `Y`
(obs RNG stream)** → **reset that stream's interval accumulator** → **evaluate
reactive policies (read `Y`, before the next interval)** → enqueue strictly
after the boundary. The enqueued effect (lag ≥ 0) never affects the `Y` read at
its own trigger time. The obs draws are on a stream independent of the dynamics
ChaCha RNG (CRN invariant preserved).

## Capability

`ChainBinomialSim::capabilities()` (`chain_binomial.rs:113`) gains
`| REACTIVE_INTERVENTIONS` — forward chain-binomial only. Gillespie/ODE
(`Simulate::capabilities`) keep withholding it (PR3), and the inference table in
`fit/methods.rs::check_model_capabilities` keeps withholding it (PR4). So a
reactive model now _runs forward on chain-binomial_ and still hard-errors
everywhere else. The PR1 fit-path / gillespie / ode rejection tests stay green
by construction.

## reactive_log.tsv — a DECLARED CAS artifact (no optional-artifact smell)

One row per firing:
`trigger_time  policy  trigger_value  threshold  fire_time
action` (the
proposal's columns).

**Constraint (CAS identity).** Whenever a model has an active reactive policy,
`reactive_log.tsv` is a **declared output artifact of the run** — always written
into the CAS leaf, part of its exact artifact set. It must NOT be "present on a
fresh run, maybe absent on a cache hit": that optional-artifact smell breaks
cache-hit reconstruction and exact-set lookup (cf. the `--save-prequential` /
holes guard). The CAS leaf is the system of record. `--reactive-log PATH` is a
**mirror** of the canonical log (exactly as `-o` mirrors the trajectory —
`cli/tests/...` "the CAS leaf is the system of record; the trajectory is the
`-o` mirror"), never the source of truth, and its presence/absence does not
change what the run produces. A model with no active reactive policy declares no
log (the artifact set is conditioned on the model, deterministically, so it is
still exact).

## Tests (PR2 behavior goldens)

Tiny deterministic models, expected values eyeball-able; commit source +
expected `reactive_log.tsv`.

1. **Lag** — obs crosses at t=28, after=21 → fires at t=49; S drops / V rises at
   49, not 28.
2. **Cooldown** — crossings at 14/21/28/60, cooldown=30 → fires at first
   eligible, suppresses the middle, fires again after the cooldown.
3. **Once** — multiple crossings, once=true → exactly one firing.
4. **Equivalence oracle (primary gate)** — a reactive run (seed S) fires at its
   realized T; the same model with the policy replaced by a scheduled
   intervention `at [T]` (seed S) produces a **byte-identical** trajectory. The
   semantic proof the agenda applies the effect exactly like a scheduled one.
   Byte-identity holds because the obs draws are on a SEPARATE RNG stream, so
   the dynamics stream is consumed identically in both runs.
5. **Default-off** — same model with/without `--enable sia`: baseline no firing;
   enabled fires.
6. **Reporting scale (`rho ≠ 1`)** — with
   `weekly_cases ~ poisson(rate = rho *
   projected)` and `rho ≠ 1`, the
   trigger sees the realized rho-scaled report, not raw incidence and not its
   mean. Pin via a fixed seed: assert the trigger's `observed` value equals
   `sample_obs_resolved` on the same projection (a draw from the
   `rho·projected`-mean distribution), and that a threshold set between
   `projected` and `rho·projected` fires iff rho-scaling is applied.

## Resolved decisions (locked)

1. **`observed` = the realized obs draw `Y`** (not `E[Y]`, not raw projection),
   sampled via `sample_obs_resolved` on a **dedicated obs RNG stream** —
   surveillance stochasticity is the point (EVSI), and the separate stream keeps
   dynamics CRN and the equivalence oracle byte-identical. Raw projection /
   `latent` are differently-named future primitives.
2. **Reactive firings merge into the scheduled due-batch lifecycle** (append the
   policy `iv_idx` to `effect_batch.intervention_idx`), so they get
   `apply_intervention_effects` + post-advance + balance + negative-count checks
   identically — one lifecycle, no fork. `after = 0` = post-observation
   immediate (fires next interval); reject with a clear diagnostic if the keying
   can't be made unambiguous in PR2.
3. **One shared observation evaluator** feeds inference scoring, synthetic obs,
   AND reactive triggers: the **per-observation-stream interval accumulator**
   (reset at each stream's emit time — NOT `current_flows`, which is
   output-tied), the projection eval (compute `projected` over that interval),
   and the resolved-likelihood sampler (`sample_obs_resolved`) / scorer. Tests
   assert `time`, `projected`, aux columns, and reporting-rate params are
   handled consistently across the three consumers — and that
   observation-interval flow is read over the _obs_ cadence, not the output
   cadence.
4. **`reactive_log.tsv` is a declared CAS artifact** when reactive policies are
   active; `--reactive-log` mirrors it; no optional-on-cache-hit semantics.

## Next

Sequencing (locked): **hold all runtime slices until #252 merges**, then branch
PR2 fresh off `main` (not stacked on the unmerged schema/golden break) and
implement in small gated commits:

1. The **`TriggerExpr` interpreter** — pure, unit-tested against a mock obs
   history (`observed`=latest, `sum_observed`=windowed sum, and/or/not,
   threshold).
2. The **shared observation evaluator**: the per-obs-stream interval accumulator
   (reset at emit times), projection eval over that interval, and
   `sample_obs_resolved` — used by inference, synthetic obs, and reactive. If
   the per-stream accumulator can't land in PR2, instead add the
   schedule-alignment validation + explicit rejection (constraint 1 fallback) —
   never silent output accumulation.
3. The **dedicated obs RNG stream** + emit-time realized draw in the
   chain-binomial loop.
4. The **`ReactiveAgenda`** + the two loop hooks (due-batch merge; evaluate/
   enqueue) + the capability grant.
5. **`reactive_log.tsv`** as a declared CAS artifact + `--reactive-log` mirror.
6. The **six behavior goldens** — equivalence oracle and `rho ≠ 1`
   reporting-scale as the gates.
