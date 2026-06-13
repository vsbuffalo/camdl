---
date: 2026-06-10
status: Phase 1 landed (2ea5f44c — bind() union axis + at_union); Phase 2 (per-stream reset) in design
related:
  - 2026-06-10-observation-data-entry-dsl.md # the data-layer companion (proposal A): the DSL surface this consumes
  - 2026-06-09-burnin-conditioning-window.md # the per-stream first-bin boundary generalizes condition_from
area: inference / multi-cadence observation axis / per-stream flow reset
issue: gh#171 (the multi-cadence remainder of the sparse-observation lift)
---

# Multi-stream multi-cadence: the union observation axis + per-stream reset

## 0. What this is

This is the **inference-side** lift: let `camdl fit` / `pfilter` / `profile`
consume observation streams on **different schedules** — polio AFP (acute
flaccid paralysis, monthly) plus ES (environmental surveillance, biweekly) is
the motivating case. `camdl simulate` already produces multi-cadence data (one
TSV per stream); only the inference path rejects it.

The **data-layer surface** (`columns { }`, `from <label>`, `~`, `[p in dim]`
indexing) is the sibling proposal A
(`2026-06-10-observation-data-entry-dsl.md`). The **survey-denominator surface**
(a binomial `n = <col>` denominator carried as a declared aux column — proposal
A's Stage 2, not a special cell variant) is a separate additive follow-up (§9).
This proposal is the union time axis and the per-stream flow reset that sit
underneath both.

**Required reading before implementing:**

- this document end-to-end;
- `rust/crates/sim/src/inference/multi_stream_obs.rs` — the `BoundObs` / `bind`
  / scoring types;
- the **six per-observation flow-reset sites** (re-grounded after Phase 1 + the
  gh#216 spine fix): `particle_filter.rs:448`, `if2.rs:587`,
  `correlated_pf.rs:550`, `pgas.rs:884` (value), `pgas.rs:1335` (`csmc_as`), and
  the gradient path `pgas_grad.rs:456`; the **Im5 canary** comment at
  `particle_filter.rs:436` (the `flow_accumulators` blanket-reset note). Note
  `if2.rs:338` is a separate **per-iteration** re-init (it zeroes accumulators
  when re-seeding particles each IF2 iteration), **not** a per-observation reset
  — it stays blanket (§3.4);
- `pgas.rs::build_obs_at_substep` / `SubstepGrid`; `schedule.rs` (`Schedule`,
  `with_obs`); `time.rs::time_to_step` (the one f64→step conversion);
- `docs/camdl-inference-spec.md` §3 (observation model).

High-risk: this touches inference math across every filter regardless of how
mechanical a given edit looks.

## 1. The problem: an asymmetric gap

A model may declare observation streams on different cadences — the DSL and IR
already allow it (surface per proposal A):

```camdl
observations {
  afp[p in patch] from afp_data {
    columns       { time : time, patch : dim, cases : count }
    projected     = incidence(paralysis[patch = p])
    cases         ~ neg_binomial(mean = rho * projected, r = k)
    emit_schedule = every 30 'days     # simulate-only (proposal A §2.5)
  }
  es[p in patch] from es_data {
    columns       { time : time, patch : dim, conc : real }
    projected     = prevalence(I_shed[patch = p])
    conc          ~ normal(mean = lambda * projected, sd = sigma)
    emit_schedule = every 14 'days     # simulate-only
  }
}
```

Both streams are **per-patch stratified** — one scored cell per patch, not a
cross-strata sum — so neither trips proposal A's aggregation gate.

`camdl simulate` **handles this**: it writes one TSV per stream (`--obs-dir` /
`--obs-only-dir`), each on its own time axis, and hard-errors if you ask for a
single wide `--obs` file (different cadences cannot share one time column),
naming the directory escape hatch (`acceptance_obs_only_dir.rs`).

`camdl fit` / `pfilter` / `profile` **reject it.** The inference loaders require
every stream to share **identical** observation times. The six original
rejection sites — the **`bind`** one is now **resolved** (Phase 1, `2ea5f44c`:
`bind` merges to the union); the five CLI/`survey` guards remain (Phase 2 drops
the in-scope three atomically with the per-stream reset):

| site                           | message                                               | reached by    | status                 |
| ------------------------------ | ----------------------------------------------------- | ------------- | ---------------------- |
| `multi_stream_obs.rs` (`bind`) | (was "obs_times that differ from stream 0")           | the substrate | **resolved (Phase 1)** |
| `fit/runner.rs:397`            | "All streams must have identical observation times."  | `fit`         | Phase 2                |
| `pfilter.rs:260`               | "All streams must share identical observation times." | `pfilter`     | Phase 2                |
| `profile.rs:527`               | "must share identical observation times."             | `profile`     | Phase 2                |
| `survey.rs:762`                | "all streams must share identical schedules."         | `survey`      | §9 follow-up           |
| `survey.rs:897`                | "all streams must share identical schedules."         | `survey`      | §9 follow-up           |

So a modeller can _simulate_ a polio AFP+ES model and **cannot fit the result
back**. That asymmetry is the gap this proposal closes. The rejection is the
correct **no-silent-gaps** behaviour for machinery that does not exist — not a
permanent stance; the give-away is `BoundObs`'s own comment, "the single shared
observation axis (homogeneous across streams **today**)."

This lift covers `fit` / `pfilter` / `profile` and the shared `bind`. **`survey`
stays loud-rejecting** — it _does_ call `bind` (`survey.rs:410`), but its own
**hand-rolled identical-schedule guards** (`survey.rs:762`/`:897`) fire first
and collapse every stream to stream-0 times, so it never feeds `bind` a real
union; routing it properly through the multi-cadence path is a separate
follow-up (§9). It is named here so the matrix is honestly enumerated, not
silently excluded.

## 2. The architecture before this lift (why one `obs_idx` worked)

This section describes the **pre-multi-cadence baseline**. Phase 1 (`2ea5f44c`)
has since added `BoundStream.at_union` and made `bind` merge to the union;
`values` was renamed `cells`. The flow-source description below — one global
per-transition `cum_flows` with a blanket reset — is what Phase 2 changes.

```rust
struct BoundStream { ir_model, projection, values: Vec<Option<ObsCell>> }
pub struct BoundObs { times: Vec<f64>, streams: Vec<BoundStream> }   // ONE shared axis
```

`bind(streams)` checked every stream's `obs_times` equals stream 0's (an
**exact** `Vec<f64>` comparison), then **collapsed** them to one `times`; the
invariant `values.len() == times.len()` held for every stream. Scoring was keyed
by a **single** `obs_idx`:

```rust
fn log_likelihood_from_flows_and_counts(&self, cum_flows, counts, obs_idx, params) -> f64 {
    let t = self.obs_times[obs_idx];                          // one time
    (0..streams).map(|si| match streams[si].observations[obs_idx] {   // SAME idx, every stream
        None          => 0.0,                                 // hole: skip the term
        Some(Scalar(v)) => score(project(si, cum_flows, counts, t), v),
    }).sum()
}
```

The flow accumulator `cum_flows: Vec<u64>` is **one global, per-transition**
tally; a stream's `FlowSum(indices)` projection reads `Σ cum_flows[i]`
(`multi_stream_obs.rs:613`). After every observation substep the filter loop
**blanket-resets** it (`reset_flows()` → zeroes all of `cum_flows`). The
substep→obs map (`build_obs_at_substep`) keys on one shared schedule.

One `obs_idx` works because it indexes **all three at once** — the time, every
stream's cell, and the (blanket) reset — which is sound _only_ when the axis is
shared.

### The Im5 canary already scoped the fix

The blanket reset carries a comment (`particle_filter.rs:401-413`, "Im5",
2026-04-19 inference review):

> resets ALL flow accumulators indiscriminately… Safe because: (a) prevalence
> streams don't consume flows; (b) disjoint FlowSum subsets don't share
> accumulator indices; **(c) overlapping subsets both reset to zero anyway.**
> **If a future feature ever stores "flow since the most recent per-stream
> observation" at different cadences per stream, this reset needs to become
> per-flow and indexed by which stream last observed. Keep this comment as the
> canary.**

Condition **(c) is exactly what breaks** under multi-cadence: overlapping flows
reset together today only because the shared axis resets _every_ stream at
_every_ observation time. With AFP and ES on different cadences, an **ES-only
union-time must not reset AFP's incidence bin** — but the blanket
`reset_flows()` would.

## 3. Design (types-first)

### 3.1 The union axis

`BoundObs.times` becomes the **union** of all streams' schedules. Each stream
is, at each union-index, in one of three states — and "not scheduled" is the new
one:

| stream state at union-time `t` | likelihood term | incidence reset     |
| ------------------------------ | --------------- | ------------------- |
| scheduled, observed value      | scored          | yes (if `Interval`) |
| scheduled, **hole** (`None`)   | omitted         | yes (if `Interval`) |
| **not scheduled**              | omitted         | **no**              |

(`Interval` = an incidence stream, which accumulates flow between observations;
the contrast is `Instant` = a prevalence stream, which reads state at the
instant and has no accumulator, so never resets — see `TemporalKind`.)

Each stream keeps **its own** schedule authoritative and derives membership
against the union:

```rust
struct BoundStream {
    ir_model, projection,
    obs_times: Vec<f64>,             // THIS stream's own schedule (authoritative)
    cells:     Vec<Option<ObsCell>>, // values, len == obs_times.len()
    at_union:  Vec<Option<usize>>,   // per union-index: Some(local_idx) if scheduled here, else None
}
pub struct BoundObs { times: Vec<f64> /* union */, streams: Vec<BoundStream> }
```

A stream resets on its **own** grid, never on a sibling's union-time. The
classic case — stream A on rows {1, 3, 8}, sibling B at 5, union {1, 3, 5, 8} —
is unambiguous: 5 ∉ A's grid, so A does not reset at 5; A's bin closes at 8 over
`(3, 8]`. For an irregular stream the grid _is_ its rows; a sibling's time
simply is not in it. Membership is exactly `at_union`: scheduled (member) vs
not-scheduled (non-member), holes being members whose value is `None`.

#### First-bin boundary, per stream (no hard error for late starters)

For incidence, an observation at time `t` on a stream with cadence Δ (its
`every` interval) is scored against the flow over `(t − Δ, t]`. The stream's
**first** observation is no different — its first scored bin opens at

> **`max(t_start, first_obs_s − every_s)`** — one of that stream's own cadences
> back, but never before the simulation begins.

This is per-stream and independent of siblings:

- **He-style** (`t_start = 0`, first obs day 7, weekly) → `max(0, 0) = 0`, first
  bin `(0, 7]`.
- **A late ES stream** (first obs day 365, biweekly, AFP running since day 0) →
  `max(t_start, 351) = 351`, first ES bin `(351, 365]` — **one ES interval, no
  error.** Flow before 351 is discarded for ES; AFP keeps accumulating on its
  own 30-day schedule.
- **An early warm-up** (`t_start = −365`, weekly, first obs 0) →
  `max(−365, −7) =
  −7`; the year of warm-up flow is discarded.

Mechanically this is a per-stream **leading reset** — the burn-in mechanism
(`2026-06-09-burnin-conditioning-window.md`), inserted at
`first_obs_s − every_s` on each incidence stream's own grid whenever that
exceeds `t_start`. So a late-starting stream is a normal surveillance situation,
not a hard error.

For an **irregular** stream (no `every`, hence no Δ), the first-bin width is not
implied by a cadence; the author states it with the existing surface,
`condition_from = first_obs − <duration>`. The default for irregular streams in
the absence of that is the conservative `(t_start, first_obs]` (the author's own
simulation window), which they can narrow explicitly. **Caveat (verified):**
`condition_from` is wired on the **`fit` path only** today — it lives on
`FitRunArgs` and is consumed in `fit/runner.rs`; `pfilter`/`profile` have no
such surface. And it is a single **global** boundary (per-stream
`condition_from` is a §9 follow-up), so using it to narrow one irregular stream
shifts the leading boundary for every stream. So the irregular-stream fallback
is fit-path-only until `condition_from` extends to `pfilter`/`profile` — which
§7 test 4's "pfilter/profile accept multi-cadence" claim must scope to the
regular-cadence (`every`-driven) case, or name the `condition_from` extension as
a prerequisite.

This default makes the gh#134 wide-first-bin condition unreachable for incidence
**by construction**, which means the `W329` wide-first-bin guard becomes vacuous
for incidence — to be retired or repurposed (e.g. to _report_ how much warm-up
is being discarded per stream). The single-stream behaviour change is real and
toward correctness: a single-stream model with an _early_ `t_start` gets the
correct one-cadence first bin instead of the wide one. Because this touches the
just-shipped burn-in default, the implementation must call the `W329`
disposition out explicitly rather than flip it silently.

### 3.2 Identity and merging are on integer steps — no tolerance

> **Phase 1 note:** the landed merge (`2ea5f44c`) uses **exact-f64** identity,
> not step indices — sound today because one deterministic parser gives byte-
> identical f64 for the same calendar instant across streams (all streams share
> the model's `origin`/`time_unit`), and sub-`dt` collisions are still caught
> downstream by `build_obs_at_substep`. The step-index merging + `1e-9` deletion
> below is the Phase-2/3 target, not yet shipped.

Observation times are not arbitrary floats. They enter through **one**
deterministic parser as `rata_die` **integer day-counts** (`caltime.rs:59`;
`t = (rata_die(date) − rata_die(origin)) / days_per_unit`), and the simulation
grid maps a time to a step with the single conversion
`time_to_step(t, dt) = (t / dt).round()` (`time.rs:80`). So observation
**identity is fundamentally integer**:

- in the default **snap** alignment, two observations are the same iff they map
  to the same integer step index; the union axis is the sorted-unique set of
  step indices;
- in **exact** alignment, observations sit at their parsed values; two are the
  same iff their parsed values are equal — and one deterministic parser gives
  one value per calendar instant, so equality is exact.

Either way, **no fuzzy tolerance is required.** The `1e-9` comparison in the
three CLI loaders (`runner.rs`, `pfilter.rs`, `profile.rs`) is defensive cruft
from comparing raw `f64` time vectors; once merging is on step indices it is
**deleted, not reconciled**. (Distinct concern, untouched: the schedule's tiny
`OUTPUT_ε` / `EFFECT_ε` boundary tolerances guard `f64` drift in the substep
time-stepping loop — accumulating `dt` over thousands of substeps — not
observation identity.)

### 3.3 `bind()` merges instead of rejecting, and the CLI plumbing follows

The in-scope rejection sites collapse to **one** lift in `bind()` — the
substrate `fit` / `pfilter` / `profile` and the model layer all route through:

- compute `times = sorted_unique(⋃ streams.obs_times)` on step indices (§3.2);
- for each stream, build `at_union` by matching its times into `times`;
- keep the existing per-stream checks (non-empty, strictly-increasing — gh#188);
- remove the homogeneous-schedule rejection.

Downstream invariants change from "every stream's `values.len() == times.len()`"
to "every stream's `cells.len() == obs_times.len()`, and
`at_union.len() ==
times.len()`."

**The CLI plumbing above `bind` must change too — it is not optional.** Today
the command layer does **not** feed each stream its own schedule into `bind`: it
builds one canonical time list from **stream 0** and reuses it for every stream
(`runner.rs:373` "Canonical observations (from first stream)"; `build_obs_model`
sets `obs_times = self.observations` for every `StreamSpec`; `pfilter.rs:217`
likewise). Under multi-cadence that pins stream 1's values to stream 0's dates
and breaks `cells.len() == obs_times.len()` before `bind` ever sees the real
schedule. So:

- `ObsStream` gains a per-stream `obs_times` (today derived from its own `data`,
  which _is_ per-stream — so this is wiring, not new data);
- `build_obs_model` passes each stream's own times, not the stream-0 canonical
  list;
- the **canonical time vector becomes the union**, computed from all streams,
  everywhere it feeds the first-bin/burn-in reset, the obs-alignment gate,
  `n_obs`, and the single-stream output paths (trace / prequential).

Dropping the identical-times _checks_ (§1) is necessary but not sufficient — the
union has to reach those consumers, or they silently keep using stream 0.

### 3.4 Per-stream reset (the crux): own accumulator, own grid

Per the Im5 canary, the reset becomes "flow since **this stream's** last
observation":

- each `Interval` (incidence) stream carries **its own single-`u64`
  accumulator** — the running sum of the flows it projects. A `FlowSum`
  projection is a plain sum over transition indices (`eval_stream_projection`,
  `multi_stream_obs.rs:320`: `idxs.iter().map(|&i| flows[i]).sum()`), so the
  per-bin quantity collapses to **one counter per stream**, not a per-transition
  vector. Advanced every substep with the dynamics;
- at a union-time where the stream is scheduled (member of its grid — value or
  hole), it is scored against its accumulator, then **its accumulator resets to
  0**;
- at a union-time where it is not scheduled (a sibling's time), it does nothing
  — its accumulator keeps running toward its next scheduled time.

This **engineers out** the overflow and resampling-desync hazards rather than
bounding them. Each accumulator holds only one inter-observation interval's flow
(the same per-bin bound as today's single accumulator — population ×
substeps-per-bin), so there is no never-reset counter to overflow; and it is
part of the particle state, copied with the particle at resampling
(`pgas.rs:1117`, `particle_filter.rs:381`), so there is no separate baseline
that can fall out of sync under ancestor swaps. (A rejected alternative — one
never-reset global counter plus per-stream baselines you subtract — computes
identical bins but reintroduces both the `u64` overflow on long national runs
and the baseline-desync under ancestor resampling.)

The blanket per-observation reset is replaced by `reset_due_flows(...)` — reset
exactly the incidence streams scheduled at the current union-index — at **all
six per-observation reset sites** (re-grounded after multi-cadence Phase 1 + the
gh#216 spine fix, which moved these): `particle_filter.rs:448`, `if2.rs:587`,
`correlated_pf.rs:550`, `pgas.rs:884` (value path `cum_flows.fill(0)`),
`pgas.rs:1335` (`csmc_as` per-particle `for f in &mut
cum_flows[j] { *f = 0 }`),
and `pgas_grad.rs:456` (gradient). `TemporalKind` gates it (only `Interval`
streams accumulate and reset). For the homogeneous case (every stream scheduled
at every union-index) this reproduces today's blanket reset exactly — the
**bit-identical-homogeneous test** (§7) is the guard. `if2.rs:338` stays
blanket: it is a per-iteration re-init, not a per-observation reset, and turning
it into `reset_due_flows` would leave stale flow in particles across iterations.
**There is a seventh** per-observation blanket reset outside the inference
filters — ODE-inference's `cum_flows.fill(0)` (`fit/runner.rs:760`), which
scores through the same seam; it is enumerated in §3.6 and must convert in
lockstep.

**Two reset forms, one shared piece.** `particle_filter` / `if2` /
`correlated_pf` reset a `&mut ParticleState`; PGAS resets a bare `Vec<u64>` /
`Vec<Vec<u64>>` flow buffer (it carries no `ParticleState`). So
`reset_due_flows` is two small functions — one over `ParticleState`, one over
the raw buffer — sharing the genuinely-common substrate: the membership
computation (`at_union[union_idx]` → which streams' flow indices are due). This
is the existing natural seam the codebase already lives with (`log_likelihood`
delegates to `log_likelihood_from_flows_and_counts`).

**`step_one` writes into a caller-owned buffer; the three families accumulate
differently.** The substep kernel `step_one` (`chain_binomial.rs:333`) does
`flows[tr_idx] += count` (`chain_binomial.rs:479`/`:496`) — an _additive write
into a buffer the caller owns and the caller chooses when to zero_. The current
disciplines differ per path, and the design works precisely because the buffer
is caller-owned:

- **forward sim** zeroes per substep — `flows.fill(0)` immediately _before_
  `step_one` (`chain_binomial.rs:244`), so its `flows` is a per-substep delta
  (it feeds the trajectory's separate `Trajectory.flows` snapshot, not a
  cumulative tally — see §3.6);
- **PF / IF2 / correlated-PF** hand `step_one` `state.flow_accumulators`
  directly and never zero between substeps (`chain_binomial_process.rs:114`,
  `correlated_pf.rs:492`) — the accumulator _is_ the cross-substep tally,
  blanket-reset only at the obs boundary;
- **PGAS** already folds a delta: `csmc_as` zeroes `substep_flows` then folds
  into `cum_flows[j]` (`pgas.rs:1216`/`:1324`); value/grad fold the recorded
  `rec.flows` delta (`pgas.rs:861`, `pgas_grad.rs:438`).

So Phase 2 does **not** change `step_one`. It introduces a per-substep-zeroed
**scratch delta** (a per-transition `Vec<u64>` in `StepScratch`) for the
PF-family (which today has no delta) and folds it into each `Interval` stream's
running sum: `for s in interval_streams: acc[s] += Σ_{i ∈ flows(s)} delta[i]`;
the PGAS family already has the delta and needs only its fold _target_
retargeted from per-transition `cum_flows` to per-stream `acc`.
`ParticleState.flow_accumulators: Vec<u64>` (per-transition, `types.rs:240`) is
**replaced** by `acc: Vec<u64>` (one per `Interval` stream); PGAS's bare
`cum_flows: Vec<u64>` / `Vec<Vec<u64>>` become per-stream the same way.
Resampling copies the per-stream `acc` with the particle exactly where it copies
`flow_accumulators` today (`particle_filter.rs:416`, `if2.rs:580`,
`pgas.rs:1196`).

**The accumulator shape comes from the obs model, not the transition count.**
Today `ParticleState::new(n_compartments, n_transitions)` (`types.rs:244`) and
every swarm allocation size on `n_transitions`. Per-stream `acc` is sized by
`n_interval_streams` — a number the **obs model** owns, not the compiled model.
So the filter constructor must take `n_interval_streams` (and per-stream flow
index sets) from `MultiStreamObsModel`, and `reset_due_flows` becomes an
**obs-model method** that knows which `acc` slot each `Interval` stream owns and
which are due at a union index (`at_union[union_idx].is_some()`). This is a real
type-contract change between the filters and the obs model — the filters today
treat the obs model as a black box (`obs_model.log_likelihood(state, …)`); the
seam (`n_interval_streams()`, `interval_flow_indices(s)`, `fold_substep`,
`reset_due`) is the load-bearing part, not the six reset calls.

**Storage.** With the single-`u64`-per-stream accumulator, the added
per-particle storage is **`O(particles × n_interval_streams)`** — one counter
per incidence stream, far smaller than today's `vec![0; n_transitions]` (which
stores a counter for _every_ model transition, including ones no stream
projects). The earlier per-index sketch (`O(particles × Σ_s |flows(s)|)`) is
unnecessary: `FlowSum` is a sum, so the per-bin value is one number, not a
per-transition vector.

**Per-substep cost — a real shift, not just storage.** Today the projection sum
runs **once per observation** (deferred to score time: `eval_stream_projection`
sums `idxs` out of the global accumulator, `multi_stream_obs.rs:320`). The
per-stream design moves that `O(Σ_s |flows(s)|)` sum to the **fold, which runs
every substep** — with ~weekly obs on a daily `dt` that is ≈7× more often,
multiplied by the **overlap** factor (a flow projected by `m` streams is summed
`m` times; a national rollup over per-LGA flows, `|flows(rollup)| = 774`, is the
worst case). So the asymptotic comparison is `O(n_obs × Σ|flows|)` today vs
`O(n_substeps × Σ|flows|)` after. §7's scaling test (test 8) must bound the
_time_ axis, not just memory; if a dense national rollup dominates, consider a
third "deferred" projection mode for non-overlapping rollup streams (keep the
once-per-obs read) rather than folding every substep.

**Honest scope: this is not "six reset calls."** The per-stream accumulator
changes `ParticleState` (replaces `flow_accumulators` with per-stream `acc`),
adds the per-substep fold + the `StepScratch` delta buffer, forks the projection
read (§3.5), changes the scoring-seam signature (§3.5), and reworks the
`mean`/`sample` emission paths. It is the bulk of the work; the reset call sites
are the small part.

### 3.5 Scoring and emission generalize

- The union-index skip and hole-omission already **landed in Phase 1**: the four
  scoring seams (`log_likelihood_from_flows_and_counts`, the gradient path,
  `sample`, `mean`) resolve the union index through `at_union[union_idx]` and
  omit not-scheduled / hole terms. Phase 2's change is the **flow source**.
- **The projection read forks.** Today every incidence stream projects from one
  global per-transition slice via `eval_stream_projection`'s `FlowSum` arm
  (`Σ_{i ∈ idxs} flows[i]`, `multi_stream_obs.rs:320`). After Phase 2, the
  **inference** scoring reads each stream's **own accumulator directly** (it is
  already the summed incidence — no `idxs` fold at score time), so
  `log_likelihood_from_flows_and_counts` / `_grad` no longer take a global
  `cum_flows: &[u64]`; they read the per-stream `acc`. The **forward / CLI
  synthetic-obs** path (`main.rs::project_all_obs_times`) keeps
  `eval_stream_projection`'s `Σ`-over-idxs on the global per-transition buffer
  with its "delta between consecutive obs times" convention — it has no
  per-stream accumulators and does not reset per cadence. This fork is the real
  scoring-seam surgery; `prevalence` (`Instant`) is unaffected (it projects
  state directly, never touches flows).
- `sample` and `mean` project incidence from the same per-stream accumulators —
  the identical rework, not just the loglik path.
- `build_obs_at_substep` / `SubstepGrid` map `substep → union_idx` over the
  union axis; `at_union` then selects who is due. The snap/exact alignment and
  collision diagnostics are unchanged — they already operate on a time list (now
  the union).

**Three PGAS legs in lockstep, not two.** PGAS has **three** separate
accumulate+reset implementations over the one shared `SubstepGrid`, and all
three must move to per-stream `reset_due_flows` together:

- the **value** objective `complete_data_loglik` (drives MH acceptance) — fold
  `rec.flows`, reset `cum_flows.fill(0)` (`pgas.rs:861`/`:884`);
- the **gradient** objective `complete_data_loglik_grad` (drives the NUTS
  leapfrog) — fold `rec.flows`, reset (`pgas_grad.rs:438`/`:456`);
- the **`csmc_as`** sweep that **produces the reference trajectory** the value
  path then scores — per-particle `Vec<Vec<u64>>`, fold `substep_flows`, reset
  (`pgas.rs:1324`/`:1335`).

Value↔gradient is the obvious one (test 6's finite-difference mutation check).
But `csmc_as` is the **producer**: if it stays blanket while the value path goes
per-stream, the conditioned trajectory is binned one way and scored another — a
silent bias _worse_ than value↔grad, because it feeds both. Test 6 must add a
**value↔csmc_as agreement** check (a multi-cadence CSMC reference trajectory's
incidence bins, as produced, equal the value path's bins on re-scoring), not
only value↔grad.

### 3.6 Every reader of the flow accumulator (verified enumeration)

Replacing `ParticleState.flow_accumulators` (and the PGAS bare `cum_flows`) with
per-stream `acc` touches more than the scoring seam. Each reader below was
verified against the code; each needs an explicit disposition in Phase 2 — a
missed one is a silent-wrong (a wrong likelihood, or a perturbed but
plausible-looking result):

| reader                                                                       | what it reads                                                                                                             | disposition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scoring seams (loglik / grad / `sample` / `mean`)                            | global `cum_flows` via `eval_stream_projection` `FlowSum`                                                                 | **read per-stream `acc`** (the fork). `mean`/`sample` go through `project_stream_with_params` → `eval_stream_projection` — the **same** helper the forward path keeps — so the fork is a branch _inside_ that helper (read `acc` for inference incidence streams), not a clean signature swap. The gh#48 pre-reset ancestor capture (`particle_filter.rs:371`) reads at the same instant and must read `acc`.                                                                                                                               |
| **ODE-inference `compute_ode_loglik`** (`fit/runner.rs:728`/`:748`/`:760`)   | its **own** hand-rolled global `cum_flows: Vec<u64>`, blanket-reset `fill(0)` at `:760`, scored through the **same seam** | **THE seventh reset site**, on a path with **no `ParticleState`/no `acc`**. In-scope (`fit` ODE-MLE + `profile`). The §3.4 six-site list omits it; the seam-signature change orphans it. It must convert to per-stream `acc` + `reset_due_flows` **in the same commit** that drops the CLI guards, or ODE-inference becomes the gh#187-class silently-mis-scoring cell. (Alternatively: capability-gate ODE-MLE/profile to homogeneous-only and say so — but that contradicts §7 test 4's requirement that `profile` accept multi-cadence.) |
| **correlated-PF resampling sort key** (`correlated_pf.rs:520`)               | `Σ` over **all** per-transition `flow_accumulators`, as the CRN sort key                                                  | replacing per-transition storage with per-stream `acc` **changes this sum → changes resampling order → changes correlated-PF / PMMH / profile output**. It is a _heuristic_ key (correctness preserved either way), but the **homogeneous-bit-identical claim is false here** unless decided deliberately. Pin the decision; the bit-identical test (test 3) must cover correlated-PF, not just the bootstrap PF.                                                                                                                           |
| **`write_final_states`** (`pfilter.rs:1359`, `--save-final-state`)           | per-transition `flow_<transition>` columns from `flow_accumulators`                                                       | after the field becomes per-stream there is no per-transition cumulative buffer to dump — re-derive the columns from the `StepScratch` delta, or redefine/drop the column set. Explicit.                                                                                                                                                                                                                                                                                                                                                    |
| forward / CLI synthetic-obs (`main.rs::project_all_obs_times`)               | `Trajectory.flows` (a **separate** `FlowVec` on `Trajectory`, `state.rs:107`)                                             | **untouched** — it never reads `ParticleState`. The §3.5 fork is safe on this side because it is a _different buffer_, not because it shares a global one.                                                                                                                                                                                                                                                                                                                                                                                  |
| transition-density + its gradient (`pgas.rs:597`/`:775`, `pgas_grad.rs:415`) | per-substep `rec.flows` (distinct from the cumulative `cum_flows`)                                                        | **untouched** — no coupling to the accumulator.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

**The NaN sentinel must become a real mask before the gate drops.** Phase 1's
`sample`/`mean` return `f64::NAN` for a not-scheduled stream (to preserve the
output-vector shape), and the prediction/prequential reductions sum the
per-stream vector (`particle_filter.rs:302`/`:308`/`:338`/`:371`) — so a NaN
poisons the ribbon, CRPS, and PIT. This is unreachable today (the CLI guards
keep every path homogeneous), but the moment Phase 2 drops those guards it is a
live silent-wrong. Replace the NaN sentinel with a real per-stream "scheduled
here?" mask at the consumer seam (an absent cell is structurally skipped, not a
number), and add a heterogeneous prequential/paths test as the guard.

## 4. What stays loud (capability honesty)

- **Sub-`dt` distinct cadences that collide on the grid** — two union-times that
  snap to the same substep still collide; the existing `build_obs_at_substep`
  collision error fires (now naming union-times). Lift via
  `--obs-alignment exact` exactly as today.
- **Prevalence-only multi-cadence** is trivially supported (no flows, no reset);
  the work is entirely about the incidence (`Interval`) reset.
- **`survey`** stays loud-rejecting (§1); lifting its separate loader is a §9
  follow-up.

## 5. Why this is well-scoped — but not "localized"

In its favour: the simulate side is already done (per-stream `--obs-dir` files);
the data format is settled (`[data.observations] <source> = file`, keyed by the
`from <label>` source per proposal A); the hole machinery is reused
("scheduled-but-missing"), with only "not-scheduled" new; and the in-scope
rejections (`fit` / `pfilter` / `profile` / `bind`) collapse to one `bind()`
lift.

Against any "trivial" framing: the per-stream reset is a **real refactor** of
inference-math code — `ParticleState`, the per-substep accumulation loop, the
scoring-seam signature, `mean`/`sample`, and the CLI canonical-grid plumbing —
not a swap of six reset calls (§3.3–3.5). It is well-scoped; it is not small.

## 6. The fixture: spatial polio AFP + ES

Extend `ocaml/golden/polio_spatial_5.camdl` (gravity-coupled patches; the
`importation[p,q] @ kappa * W[p,q] * S[p] * I[q]/N[q]` pattern) to **2–3
patches**. That base model has **no shedding compartment, no paralysis flow, and
no `observations {}` block** today, so the fixture **adds**: an `I_shed`
compartment, a `paralysis` flow, and two stream blocks (proposal A's surface) at
different cadences:

- **`afp[p in patch]`** — `incidence(paralysis[patch = p])`, **monthly**
  (`emit_schedule = every 30 'days`), low / zero-heavy counts (paralysis is a
  small fraction of infection), `neg_binomial`. Per-patch stratified (one cell
  per patch — not a cross-strata sum). Exercises incidence reset + low-mean NB +
  holes.
- **`es[p in patch]`** — `prevalence(I_shed[patch = p])`, **biweekly**
  (`emit_schedule = every 14 'days`), `poisson` for v1. (When the denominator
  aux column lands — proposal A Stage 2 — `es` upgrades to binomial positivity
  (`n = tested`); the denominator follow-up meets here.)

A mixed **incidence + prevalence at different cadences** is the hardest case and
the one the union axis must get right. Synthetic data via
`camdl simulate … --obs-dir` (one TSV per stream at its own cadence) from known
params; the fit recovers them. The fixture lives in `tests/fixtures/` (model)
plus a `fit.toml` using `[synthetic]` (`true_params` + `sim_seeds`) so the test
is self-contained and recover-known-params.

## 7. Tests (red → green; this is inference math — paste red/green in commits)

1. **`union_axis_per_stream_reset`** (deterministic, like
   `sparse_holes_reset.rs`) — 2 streams, AFP `every = 30`, ES `every = 14`,
   fixed seed, drainless dynamics so counts are RNG-independent. Assert AFP's
   scored bin equals the flow over its **30-day** span (not the 14-day union
   step), and an **ES-only union-time does not reset AFP's accumulator**
   (mutation check: forcing a blanket reset makes AFP's bin too small → test
   fails). The canary condition, made executable.
2. **`bind_merges_heterogeneous_schedules`** — `bind()` on two different-cadence
   `StreamSpec`s returns `Ok` with `times` = the union and correct per-stream
   `at_union`; the old "identical times" rejection is gone; a 3rd, prevalence
   stream at a 3rd cadence also binds.
3. **`homogeneous_is_bit_identical`** — all streams on one cadence: union axis
   == shared axis, per-stream reset == blanket reset, loglik byte-identical to
   today. The regression guard.
4. **End-to-end fit** — the polio AFP+ES fixture: `camdl fit run` (IF2 scout,
   optionally PGAS) on `[synthetic]` data recovers `beta`, coupling `kappa`, and
   the reporting params within tolerance. **`pfilter`, `fit`, AND `profile`**
   all accept the multi-cadence per-stream files — proving the `bind()` lift
   _and_ the CLI plumbing (§3.3) reach every in-scope path.
5. **Single-patch reduction cross-check** — one patch, AFP+ES, against a hand /
   pomp-style computation of each stream's bin, anchoring the mechanism where an
   oracle is tractable (no oracle exists for spatial multi-cadence).
6. **Three-leg PGAS consistency** — the §3.4 per-stream reset reaches the
   gradient path (`pgas_grad.rs:456`) **and** the `csmc_as` producer
   (`pgas.rs:1335`) in lockstep with the value path (`pgas.rs:884`). (a)
   finite-difference value-vs-grad on a genuinely **multi-cadence** fixture
   (homogeneous would not separate the reset policies), including a near-`k = n`
   boundary point; mutation check: leaving `pgas_grad.rs:456` blanket while the
   value path is per-stream fails the finite-difference. (b) **value↔csmc_as
   agreement**: a multi-cadence CSMC reference trajectory's incidence bins, as
   produced, equal the value path's bins on re-scoring; mutation check: leaving
   `pgas.rs:1335` blanket biases the conditioned trajectory.
7. **Late-starting stream** — ES first observation a year after AFP's: the first
   ES bin is `(first_obs − 14, first_obs]`, **not** `(t_start, first_obs]`, and
   the fit does **not** hard-error. Pins the §3.1 first-bin rule.
8. **Scaling** — an **overlapping** configuration (per-LGA incidence + a
   national rollup over the same flows) at the 774-LGA national scale: assert
   per-particle accumulator memory **and** per-substep accumulation time stay
   within a bound. The disjoint 774-LGA case alone understates the storage (no
   overlap multiplier) and would let an overlap blow-up through.

## 8. Implementation phases

1. **`bind()` merge + scoring substrate** (§3.1–3.2) — **LANDED `2ea5f44c`.**
   `BoundStream.at_union`, union `times` on exact-f64 identity, the "must equal
   stream 0" rejection removed; the four scoring seams resolve the union index
   through `at_union`. Tests: `bind_merges_heterogeneous_schedules` (the
   inverted reject test) + the homogeneous happy-path `at_union` assertion; the
   existing suite is the bit-identical-homogeneous guard. **Deliberately NOT in
   Phase 1:** the CLI loaders (`runner.rs`, `pfilter.rs`, `profile.rs`) **keep**
   their identical-times guards, so heterogeneous fits stay loud-rejected
   END-TO-END. Removing the CLI checks without the per-stream reset (Phase 2)
   would open a silent-wrong window — the blanket reset corrupting incidence
   bins across cadences. The merge is exercised by the unit test only.
2. **Per-stream reset + open the gate** (§3.3–3.5) — the highest-risk piece.
   Replace `ParticleState.flow_accumulators` with per-stream `acc`; the
   `StepScratch` delta buffer + the per-substep fold (`step_one` unchanged); the
   projection-read fork; `reset_due_flows` (two forms) at the six sites incl.
   `pgas_grad.rs:456` in lockstep with `pgas.rs:884`; the scoring-seam signature
   and `mean`/`sample` rework. **Atomically** with the reset: the CLI plumbing
   (§3.3 — feed each stream its own times, union becomes canonical, **drop** the
   identical-times guards), so heterogeneous opens end-to-end exactly when it
   becomes correct. Tests 1, 3, 6.
3. **Per-stream first bin** (§3.1) — the leading reset at
   `max(t_start, first_obs_s − every_s)`; irregular via `condition_from`; the
   `W329` disposition (retire/repurpose for incidence, called out explicitly).
   Test 7.
4. **Fixture + end-to-end** (§6) — the polio model (added `I_shed` / `paralysis`
   / observations), `[synthetic]` fit.toml. Tests 4, 5, 8.
5. **Docs** — `camdl-inference-spec.md` §3 (union axis + per-stream reset),
   `fit-toml.md` (heterogeneous cadences now fit), retire the "must share
   identical times" language, and record `survey` as a deliberate loud-reject.

## 9. Out of scope (named, not forgotten)

- **Survey denominators** — a binomial `n = <col>` denominator carried as a
  declared aux column (proposal A Stage 2), on top of this union-axis
  `BoundObs`; not a special cell variant. ES-as-binomial-positivity is its
  natural first consumer.
- **`survey` multi-cadence** — its loader is separate from `bind` (§1); a
  follow-up that routes it through the same lift.
- **Explicit per-stream `condition_from` in the observation block** — the
  automatic `max(t_start, first_obs_s − every_s)` default (§3.1) covers the
  common case; an explicit per-stream override is a small follow-up if a model
  needs a conditioning window other than one cadence.
- **`W329` on the `pfilter` path** — the first-window guard is fit-path-only;
  orthogonal to this lift.

## 10. References

- The Im5 canary: `rust/crates/sim/src/inference/particle_filter.rs:401-413`
  (2026-04-19 inference review) — predicts this feature and scopes the reset
  fix.
- The rejection sites (re-grounded; the `bind` one resolved in Phase 1):
  `fit/runner.rs:397`, `pfilter.rs:260`, `profile.rs:527`, `survey.rs:762`,
  `survey.rs:897`.
- Scoring seam (current): `eval_stream_projection` `FlowSum` arm
  `multi_stream_obs.rs:320`; `log_likelihood_from_flows_and_counts` / `_grad` /
  `sample` / `mean` all resolve `at_union[obs_idx]` (Phase 1).
- Per-observation reset sites (→ `reset_due_flows`, re-grounded after Phase 1 +
  gh#216): `particle_filter.rs:448`, `if2.rs:587`, `correlated_pf.rs:550`,
  `pgas.rs:884` (value), `pgas.rs:1335` (csmc_as), `pgas_grad.rs:456`
  (gradient), and **`fit/runner.rs:760`** (ODE-inference — the seventh, §3.6).
  Per-iteration re-init (stays blanket): `if2.rs:338`. Resampling copy of
  accumulators: `pgas.rs:1196`, `particle_filter.rs:416`, `if2.rs:580`.
  Non-scoring readers of `flow_accumulators` (§3.6): `correlated_pf.rs:520`
  (sort key), `pfilter.rs:1359` (`--save-final-state`).
- The CLI canonical-grid collapse: `fit/runner.rs:373` / `build_obs_model`,
  `pfilter.rs:217`.
- Integer time representation: `rata_die` in `ir/src/caltime.rs:59`;
  `time_to_step` in `sim/src/time.rs:80`. Substep mapping:
  `pgas.rs::build_obs_at_substep`, `SubstepGrid`. Spine seam: `schedule.rs`
  (`Schedule`, `with_obs`).
- The simulate/fit asymmetry: `acceptance_obs_only_dir.rs`.
- Fixture base: `ocaml/golden/polio_spatial_5.camdl` (gravity coupling).
- The sparse/hole machinery this reuses: `2026-06-06-observation-system.md`.
- Sibling data-layer proposal: `2026-06-10-observation-data-entry-dsl.md`.
- The per-stream first-bin boundary generalizes:
  `2026-06-09-burnin-conditioning-window.md`.
