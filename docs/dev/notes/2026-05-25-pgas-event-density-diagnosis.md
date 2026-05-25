# gh#80 — PGAS density evaluator vs deterministic events: diagnosis

Date: 2026-05-25
Project: camdl
Tags: pgas, events, density, inference

## Classification

This is **doc-vs-code**: the gh#80 issue text claims a structural
mismatch between `step_one` and `log_transition_density_substep`. The
actual code is consistent. The reported symptoms come from two
unrelated UX issues that share a misleading diagnostic message.

## Verification

The issue's mechanism claim is:

> The deterministic E-jump has no associated rate; the evaluator
> computes `density(flow=n_seed | rate=0) = -inf`.

This implies the event delta shows up in the `flows` array. **It does
not.** Verified by reading every site that mutates `flows` in
`chain_binomial.rs::step_one`:

```
$ rg 'flows\[' rust/crates/sim/src/chain_binomial.rs
chain_binomial.rs:396:            flows[tr_idx] += count;    # stochastic transition split
chain_binomial.rs:413:        flows[i] += count;             # inflow / ungrouped Poisson
```

Both writes are stochastic-transition draws. The event path runs
through `inject_event_deltas` (chain_binomial.rs:417-420), which writes
to `scratch.pending_deltas` — a `(local_idx, delta)` list that is
applied directly to the `counts` slice at chain_binomial.rs:422-425.
Events never touch `flows`.

### What `counts_before` reflects in the trajectory

In `simulate_reference`:

```rust
let counts_before = counts.clone();    // pgas.rs:715, before step_one
step_one(model, &mut counts, &mut flows, params, t, dt, rng, &mut scratch, &fire_steps)?;
substeps.push(SubstepRecord {
    counts_before,                      // pgas.rs:733
    counts_after: counts.clone(),       // pgas.rs:734
    flows,
    gammas: scratch.gamma_used.clone(),
});
```

`counts_before` is captured **before** `step_one` runs, so it is the
pre-substep, pre-event, pre-stochastic-transition state. `counts_after`
is the post-substep, post-event, post-stochastic-transition state.

### Ordering inside `step_one`

```rust
scratch.int_s.counts.copy_from_slice(counts);   // line 274 — snapshot pre-step state
eval_propensities(...);                          // line 282 — rates from pre-step state
// ... draws computed from those pre-step rates, recording into `flows`
crate::intervention::inject_event_deltas(        // line 417 — events evaluated against snapshot
    model, fire_steps, &scratch.int_s, ...,
    &mut scratch.pending_deltas,
)?;
// line 422-425: apply transition deltas AND event deltas atomically.
```

So within a substep that contains an event firing:

* Rates are computed against the pre-event (= start-of-step) state.
* Stochastic transition counts are drawn from those rates.
* The event delta is added to `pending_deltas` from the same snapshot.
* All deltas — transitions and events — are applied atomically.

This matches pomp's chain-binomial convention (the comment at
chain_binomial.rs:115-126 calls this out).

### What `log_transition_density_substep` expects

It evaluates `p(flows | counts_before, params, t, dt)` by recomputing
rates from `counts_before` and feeding them through the same
Binomial-multinomial decomposition `step_one` used. Because
`counts_before` in the trajectory IS the pre-step (pre-event) state,
this is consistent with how `step_one` drew the flows. **The density
is mathematically the right one.**

## Why the proposed fix would break correctness

The gh#80 proposal:

> Apply any deterministic events firing at this boundary to
> `counts_before` → `counts_after_event`. Evaluate rates from
> `counts_after_event`.

At the event substep with `add(E, 5)`:
* Pre-event state: E=0 → progression rate `σ·E = 0`.
* Post-event state: E=5 → progression rate `σ·5 > 0`.

`step_one` drew progression flow from the **pre-event** rate of 0 → so
the recorded flow is 0. If the density evaluator computes rates from
the post-event state, it would score `density(flow=0 | rate = σ·5·dt)`
= Poisson(0; σ·5·dt) < 1. That is a *negative* log-density penalising
the very outcome the simulator was forced to produce.

I.e., the proposed fix introduces a *new* density/simulator
disagreement at exactly the substep that the issue claims has one.

## Where the two warnings actually come from

### "BUG: simulate_reference trajectory has -inf density at own params"

Reproduced on the WA seed-timing model with one of two random-start
chains (`tau = -72.83`) using
`/Users/vsb/projects/work/camdl-book/guide/fitting/seed-timing/fits/smoke_synth_pgas.toml`.
With `RUST_LOG=sim=debug`:

```
DEBUG sim::inference::pgas] complete_data_loglik: obs density -inf at substep 53 (obs_idx=27)
DEBUG sim::inference::pgas] complete_data_loglik: -inf after obs at substep 53 (cumulative)
  BUG: simulate_reference trajectory has -inf density at own params.
  params used:
    ...
    tau = -72.8270034787366
```

`tau = -72.83` is **before** `t_start = -34`, so the `add(E, n_seed) at
[tau]` event never fires within the simulation window — E stays 0, no
infections, predicted incidence is 0 for every observed day. Real data
has `cases ≥ 1` from `t = -7` onwards, and NegBin(mean=0) is -∞ for
any non-zero observation. The `-inf` is therefore in the *obs density*
term of `complete_data_loglik`, **not** in any `log_transition_density_substep`
call.

The "BUG: simulate_reference trajectory has -inf density at own params"
message at pgas.rs:1317 prints regardless of which term went -∞, and
the follow-up sentence ("indicates a mismatch between step_one and
log_transition_density_substep", pgas.rs:1398) is incorrect when the
obs term is the cause. This is a diagnostic bug, not an inference bug.

### "transition index 0 has rate=0 but flow=N"

Fires from `pgas::compute_source_group_probs` at pgas.rs:315. The
condition is "rate ≤ RATE_EPSILON and flows[tr] > 0 with rate ≤ 0".
This is structurally impossible for a trajectory scored at the params
that **produced** it — `step_one` only draws flow > 0 from rate >
RATE_EPSILON, so the trajectory's own state/flow pair always agrees.
The warning therefore fires only when `log_transition_density_substep`
is called with a `counts_before` that came from a **different
particle** than the `flows` argument. The two CSMC callers that do
this are:

* CSMC ancestor sampling (pgas.rs:984-996): pairs each free particle's
  `prev_counts_for_ancestor[j]` with the reference's
  `ref_rec.flows`/`ref_rec.gammas` to score the reference's flows
  against each candidate ancestor's pre-step state.
* CSMC traceback verification under `cfg!(debug_assertions)`
  (pgas.rs:1075-1083): re-scores the reconstructed trajectory at the
  same params — this path *cannot* trigger the warning because each
  substep's `counts_before`/`flows` come from the same particle.

So the warning is exclusively from ancestor sampling. Correct
behaviour: a free particle whose state `n_src = 0` for the source
compartment of a transition that **did** fire in the reference cannot
have been an ancestor of the reference — the conditional density is
mathematically zero (`log = -∞`). In seed-timing models with sparse
early counts the ancestor sampler legitimately encounters this for
many free particles every sweep, hence the spam.

The math is correct; the diagnostic is just verbose.

## What needs to actually change

1. **The diagnostic at pgas.rs:1317-1403** — distinguish "transition
   density -∞" from "obs density -∞" before printing the mismatch
   accusation. Without that split, a user with a poorly-initialised
   `tau` blames `step_one` when the actual signal is "your starting
   parameters are incompatible with the observed data."
2. **The "rate=0 but flow=N" warning at pgas.rs:315** — it is correct
   behaviour from `csmc_as` ancestor sampling, not a model
   pathology. Demote to `debug!` so the production log is not
   flooded; the maths is identical (we still return `-∞` and the
   particle is excluded from the ancestor categorical).
3. **No event-density code changes are warranted.** The current
   `step_one` / `log_transition_density_substep` pair is consistent,
   and the gh#80 proposed fix would *introduce* a mismatch where
   none exists today.

## Repro commands

* PASSES on current code (transition density at the event substep is
  finite by construction):
  `cargo test -p sim --test pgas_event_density pgas_simulate_reference_finite_density_on_event_model`
* PASSES on current code (SEIR seed-into-E variant):
  `cargo test -p sim --test pgas_event_density pgas_simulate_reference_finite_density_on_seir_event_model`
* RED on current code (full PGAS fit on seed-timing model, surfaces
  the obs-density -∞ misdiagnosis): the smoke_synth_pgas.toml fit
  from camdl-book at tau<-34 chain.

## Coordination with the gh#20+gh#76 base branch

This finding does not affect any term in
`pgas_grad::complete_data_loglik_grad`. The transition-density
gradient (gh#76) operates from the same `counts_before` the
non-gradient density uses; both are consistent with `step_one` today.
The obs-density gradient (gh#76) and gamma-density gradient (gh#20)
are unaffected by event handling. Existing `gradient_check_*` tests
remain valid.
