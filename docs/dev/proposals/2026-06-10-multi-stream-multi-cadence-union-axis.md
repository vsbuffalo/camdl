---
date: 2026-06-10
status: Phases 1-3 + fixture landed; heterogeneous fits run end-to-end. Remaining Phase 5 (docs) + 6 (Z->X profiling)
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

#### First-bin boundary, per stream — EXPLICIT, not inferred (shipped Phase 3)

> **Design reversal (decided 2026-06-13).** An earlier draft of this section
> _inferred_ each incidence stream's first-bin boundary automatically as
> `max(t_start, first_obs_s − Δ_s)` (Δ_s = the stream's cadence). That was
> dropped: the particle filter is fragile to a mis-placed first bin (a wide bin
> makes the predicted obs an integral no particle matches → ESS collapse), and
> an _inferred_ boundary fails exactly on the irregular/sparse data this lift
> targets — and fails **silently** (the modeller never chose the window). For
> public-health software that is the wrong default. Conditioning is therefore
> **explicit**: the modeller states it, the filter never guesses.

For incidence, observation `t` is scored against the flow over `(t − Δ, t]`; the
**first** observation is the only one whose left edge isn't a prior obs. The
modeller sets that edge with `condition_from` (per-stream, below); when set, the
leading span `[t_start, cond_from_s)` warms up (simulated, not scored) and the
first scored bin is `(cond_from_s, first_obs_s]`. The boundary is realised as a
per-stream **leading reset-only hole** at `cond_from_s` (the burn-in mechanism,
`2026-06-09-burnin-conditioning-window.md`, generalised per stream — it rides
the Phase-2a hole+reset seam, no new mechanism).

**`condition_from` is a default + per-stream shadows** (a serde-`untagged`
`All(String) | PerStream(table)`):

```toml
condition_from = "first_obs - 1 week" # All: the default for every stream
# — or —
[condition_from] # PerStream: default + shadows
default = "first_obs - 1 week"
es = "first_obs - 2 weeks" # shadows the `es` stream only
```

Resolution per stream `s`: `shadows[s]` → else `default`/`All` → else **none**.
Non-exhaustive (a stream with neither falls to none). The `default` key is
reserved; an unknown shadow label is a hard error (typo-safety).

**The W329 guard is the per-stream enforcer** (it was _not_ repurposed to a
report — that earlier idea is superseded by the explicit-required decision). For
each incidence stream that resolves to **none**, W329 runs against that stream's
_own_ times: if its first window is anomalously wide relative to its own modal
cadence (`first_obs_s − t_start ≫ Δ_s`, the K=5 criterion), it is a **hard
error** naming the fix — `condition_from.<label> = first_obs - <duration>`. The
modal gap is used only to _detect_ the anomaly, never to _set_ a boundary. A
stream whose first window is ~one cadence (He-2010, the polio fixture's
t=0-anchored AFP/ES) is silent and needs no `condition_from`. A prevalence
(`Instant`) stream is exempt from the hard error (free-running drift the first
datum corrects) — soft-warn only.

**Scope (verified):** `condition_from` + the W329 enforcer are **`fit`-path
only** today (`FitRunArgs` / `fit/runner.rs`); `pfilter`/`profile` are
unchanged. Extending the surface to them is a §9 follow-up; §7 test 4's
"pfilter/profile accept multi-cadence" is the regular-cadence case (first window
≈ one cadence, no conditioning needed).

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

### 3.4 Per-stream reset (the crux): a persistent per-stream bin, folded once per interval

> **Shipped as "Option Z" (`87897e61`).** Two equivalent designs were possible.
> **Z** (shipped): keep the per-transition `flow_accumulators` **unchanged** and
> add a persistent per-stream bin folded **once per observation interval**.
> **X** (the original sketch below the fold): remove `flow_accumulators`, fold
> every substep. They compute identical numbers; Z is far less invasive (no
> `step_one` change, the two non-scoring readers of `flow_accumulators` stay
> bit-identical, no per-substep cost amplification). X remains a possible later
> **memory** optimization (it shrinks the per-transition buffer) — deferred to a
> profiling pass after the whole multi-cadence push lands (§8).

Per the Im5 canary, the reset becomes "flow since **this stream's** last
observation", via a persistent per-`Interval`-stream bin `acc`:

- `ParticleState` **keeps** `flow_accumulators: Vec<u64>` (per-transition,
  this-interval tally) **unchanged** — `step_one` writes it
  (`flows[tr] += count` per substep), `reset_flows()` blanket-zeroes it once per
  observation, exactly as before. It **adds** `acc: Vec<u64>` — one persistent
  `u64` per `Interval` (`FlowSum`) stream.
- **once per observation interval**, after the substeps and before scoring, the
  filter folds `acc[k] += Σ_{i ∈ FlowSum(k)} flow_accumulators[i]`
  (`fold_into_acc`); `FlowSum` is a plain sum so the per-bin quantity is one
  counter per stream;
- scoring reads `acc[k]` directly (the already-summed bin); then
  `flow_accumulators` blanket-zeroes (unchanged) and `acc[k]` zeroes **only for
  the streams scheduled at this union index** (`reset_due_acc`, gated on
  `at_union[union_idx].is_some()`). A stream not scheduled here keeps its
  running bin toward its own next observation.

Worked: AFP(30 d) + ES(14 d) sharing transition T. At union times 14, 28 (ES
due, AFP not) the fold adds T's flow to both bins, ES scores+resets, AFP carries
forward; at 30 (AFP due) AFP scores its full `(0, 30]` bin and resets. Each
stream gets its own window.

This **engineers out** the overflow and resampling-desync hazards rather than
bounding them. `flow_accumulators` is bounded per-interval (blanket-reset) as
before; `acc` holds one inter-observation bin and is **part of particle state,
copied with the particle at resampling** (`particle_filter.rs:439`,
`pgas.rs:1218` mirroring `cum_flows` exactly), so there is no separate baseline
that can fall out of sync under ancestor swaps. (A rejected alternative — one
never-reset global counter plus per-stream baselines you subtract — computes
identical bins but reintroduces both the `u64` overflow on long national runs
and the baseline-desync under ancestor resampling.)

`reset_due_acc(union_idx, acc)` — zero exactly the incidence streams scheduled
at the current union-index — fires at **all seven per-observation reset sites**,
each **alongside** the unchanged blanket `flow_accumulators` reset:
`particle_filter.rs`, `if2.rs`, `correlated_pf.rs`, `pgas.rs` value + `csmc_as`
(per-particle), `pgas_grad.rs` (gradient), and ODE-inference
(`fit/runner.rs::compute_ode_loglik` — the seventh, scoring through the same
seam). `stream_to_slot` gates it (only `Interval`/`FlowSum` streams own an `acc`
slot). For the homogeneous case (every stream scheduled at every union-index)
every `acc` zeroes every interval ⇒ byte-identical to today's blanket reset —
the **bit-identical-homogeneous test** plus the camdl-vs-pomp He-2010 **value
oracle** (§7) are the guards. `if2.rs` per-ITERATION re-init stays blanket
(zeroes both `flow_accumulators` and `acc`): it re-seeds particles each IF2
iteration, so per-stream reset there would leave stale flow across iterations.

**The seam, not the seven reset calls, is the load-bearing part.** `acc` is
sized by `n_interval_streams` — a number the **obs model** owns, not the
compiled model. So the work is three `ObservationModel` methods (added as
**default no-ops** so `NullObsModel` and non-incidence mocks stay vacuous, and
overridden on `MultiStreamObsModel`):

- `n_interval_streams()` — how long each particle's `acc` is; the filters size
  the swarm's `acc` from it (the process model passes `0`, since it does not own
  the obs model);
- `fold_into_acc(flow_accumulators, acc)` —
  `acc[k] += Σ_{i∈FlowSum(k)}
  flow_accumulators[i]`, once per interval,
  **never zeroes `flow_accumulators`**;
- `reset_due_acc(union_idx, acc)` — zero `acc[k]` iff the k-th stream is
  scheduled at `union_idx` (`at_union[union_idx].is_some()`).

`stream_to_slot` (`Some(k)` for the k-th `FlowSum` stream, `None` for
prevalence) maps streams to `acc` slots; the scoring fork reads `acc[k]` for
`Some(k)` and projects from `counts` for `None` (§3.5).

**`step_one` is untouched; the fold is once per observation, not per substep.**
The kernel's `flows[tr] += count` (`chain_binomial.rs:479`/`:496`) and the
per-obs blanket `flow_accumulators` reset are exactly as before. The fold reads
that this-interval tally **once at the obs boundary**, so the `O(Σ|flows|)` sum
runs **once per observation — the same cost as today's deferred projection**, no
per-substep amplification. (The original "Option X" sketch removed
`flow_accumulators` and folded a `StepScratch` delta every substep — `≈7×` more
folds plus a per-substep overlap multiplier; Z avoids that entirely.)

**`flow_accumulators`'s lifecycle is identical to before** — which is _why_ its
two non-scoring readers, the correlated-PF resampling sort key
(`correlated_pf.rs:520`) and `write_final_states` (`pfilter.rs:1359`), stay
**bit-identical and untouched** (§3.6). Under Option X both would have broken
(the per-transition buffer goes away); Z dissolves those.

**Storage.** `acc` is `O(particles × n_interval_streams)` — tiny — added **on
top of** the unchanged `flow_accumulators` (`O(particles × n_transitions)`).
Keeping both is slightly redundant in the homogeneous case (`acc` carries
nothing across intervals there). That redundancy — and forgoing X's shrink of
the per-transition buffer — is the deliberate, **reversible** cost of Z: the two
designs compute identical numbers, so X stays available as a later memory
optimization gated on profiling (§8), not bundled into the correctness change.

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

### 3.6 Every reader of the flow accumulator (verified, as shipped)

Option Z **keeps** `ParticleState.flow_accumulators` (and the PGAS `cum_flows`)
unchanged and **adds** the per-stream `acc`. So the only reader that _changes_
is the scoring seam (it reads `acc`); every other reader of `flow_accumulators`
keeps working untouched. Each was verified against the shipped code
(`87897e61`):

| reader                                                                                                    | what it reads                                     | disposition (shipped)                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| scoring seams (loglik / grad / `sample` / `mean`, + the gh#48 pre-reset capture `particle_filter.rs:371`) | per-stream `acc`                                  | **changed**: a new `project_stream_from_acc` reads `acc[k]` for an `Interval` stream, projects from `counts` for an `Instant` stream (the old `project_stream_with_params` is deleted). The four seams take `acc`; the filter folds `flow_accumulators → acc` before scoring, so the gh#48 capture reads the folded bin. |
| ODE-inference `compute_ode_loglik` (`fit/runner.rs`)                                                      | its own `cum_flows`, scored through the same seam | **converted** — the seventh reset site: folds `cum_flows → acc` before scoring, `reset_due_acc` after; `cum_flows.fill(0)` unchanged.                                                                                                                                                                                    |
| correlated-PF resampling sort key (`correlated_pf.rs:520`)                                                | `Σ` over `flow_accumulators` (the CRN key)        | **untouched** — `flow_accumulators` is unchanged under Z, so the sort key is bit-identical (correlated-PF / PMMH / profile output unchanged). This is the consumer Option X would have broken; Z dissolves it.                                                                                                           |
| `write_final_states` (`pfilter.rs:1359`, `--save-final-state`)                                            | per-transition `flow_<transition>` columns        | **untouched** — the per-transition buffer stays, so the dump is unchanged (`pfilter.rs` was not edited).                                                                                                                                                                                                                 |
| forward / CLI synthetic-obs (`main.rs::project_all_obs_times`)                                            | `Trajectory.flows` (a separate buffer)            | untouched — never reads `ParticleState`.                                                                                                                                                                                                                                                                                 |
| transition-density + its gradient (`pgas.rs`/`pgas_grad.rs`)                                              | per-substep `rec.flows`                           | untouched — no coupling to the accumulator.                                                                                                                                                                                                                                                                              |

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
   bins across cadences. The merge is exercised by the unit test only. 2a.
   **Per-stream reset** (§3.4–3.6, Option Z) — **LANDED `87897e61`.** Keep
   `flow_accumulators` unchanged; add persistent per-stream `acc`; the obs-model
   seam (`n_interval_streams`/`fold_into_acc`/`reset_due_acc`, default-no-ops);
   the scoring fork (`project_stream_from_acc`); fold once per interval +
   `reset_due_acc` at all seven sites incl. ODE, with the three PGAS legs in
   lockstep. **CLI guards retained** — heterogeneous still loud-rejected
   end-to-end; validated by the directly-bound `per_stream_reset` canary (+ its
   mutation guard), the bit-identical suite, and the camdl-vs-pomp He-2010 value
   oracle (DRIFT 0). Tests 1, 3.

   **2b. Open the gate** (§3.3) — **LANDED `bc31545b`.** Dropped the three CLI
   identical-times guards (`runner.rs`, `pfilter.rs`, `profile.rs`), made the
   canonical `observations` the union, fed each stream its own times to `bind`,
   and replaced the `sample`/`mean` not-scheduled `f64::NAN` sentinel with a
   `.filter(is_finite)` mask at the prequential/prediction consumers. The polio
   fixture's `synthetic_fit_recovers_params` is un-ignored (a real IF2-scout
   heterogeneous fit recovers R0/rho) and `binding_both_cadences_…` is inverted
   to `…_now_fits`. (`survey` stays loud-rejecting — §9.) Tests 4, 5.
2. **Per-stream first bin — EXPLICIT** (§3.1) — **LANDED.** No automatic
   inference (reversed): `condition_from` is a default+shadow type
   (`All | PerStream`); the W329 guard is the per-stream HARD-ERROR enforcer
   naming `condition_from.<label>` for a wide-window incidence stream with no
   conditioning; the boundary, when given, is a per-stream leading reset-only
   hole. He-2010 + the polio fixture (windows ≈ one cadence) need none and are
   unchanged. Fit-path only. Test 7.
3. **Fixture + end-to-end** (§6) — **the model + synthetic data + forward-sim
   smoke LANDED `2a04da45`** (`tests/fixtures/polio_afp_es_2patch.camdl` +
   `polio_afp_es_multicadence.rs`); the recover-known-params fit is `#[ignore]`d
   until 2b opens the gate. Tests 4, 5, 8 (the fit-recovery + scaling) complete
   with 2b.
4. **Docs** — `camdl-inference-spec.md` §3 (union axis + per-stream reset),
   `fit-toml.md` (heterogeneous cadences now fit), retire the "must share
   identical times" language, and record `survey` as a deliberate loud-reject.
5. **Z→X profiling (after the whole push lands)** — profile the per-particle
   `flow_accumulators` memory at the 774-LGA national scale. If it dominates,
   collapse Option Z → X (drop `flow_accumulators`, fold a `StepScratch` delta
   per substep) as a _separate, profiled_ optimization PR — taking on the
   `step_one` change + the correlated-PF-sort-key / `write_final_states` rework
   deliberately, with the feature already proven correct. Z and X compute
   identical numbers, so this is a reversible representation swap, not a
   re-design.

## 9. Out of scope (named, not forgotten)

- **Survey denominators** — a binomial `n = <col>` denominator carried as a
  declared aux column (proposal A Stage 2), on top of this union-axis
  `BoundObs`; not a special cell variant. ES-as-binomial-positivity is its
  natural first consumer.
- **`survey` multi-cadence** — its loader is separate from `bind` (§1); a
  follow-up that routes it through the same lift.
- **Per-stream `condition_from`** — **LANDED in Phase 3** (the `All | PerStream`
  default+shadow type, §3.1). What remains out of scope: extending the
  `condition_from` surface + the W329 enforcer to the **`pfilter`/`profile`**
  paths (fit-path-only today).

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
