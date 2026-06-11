# Multi-stream multi-cadence: the union observation axis + per-observer reset

- **Status:** Draft — design review. No code yet. Scope is the **inference**
  side only: let `camdl fit` / `pfilter` / `profile` consume observation streams
  on **different schedules** (e.g. polio AFP monthly + ES biweekly). The
  `simulate` side already produces multi-cadence data (one TSV per stream).
- **Issue:** the remaining gap from the sparse-observation lift (gh#171); file a
  dedicated multi-cadence issue on landing.
- **Supersedes / relates to:** completes the observation-system work
  (`2026-06-06-observation-system.md`, the sparse/hole machinery this reuses).
  The **denominator surface** (`Counted{value,denom}` for survey positivity) is
  a _separate, additive_ follow-up — a new `ObsCell` variant on top of the
  union-axis `BoundObs` — and is explicitly out of scope here (§9).
- **Required reading before implementing:** this document end-to-end;
  `rust/crates/sim/src/inference/multi_stream_obs.rs` (the `BoundObs` / `bind` /
  scoring types); the six flow-reset sites (`particle_filter.rs:415`,
  `if2.rs:319/559`, `correlated_pf.rs:521`, `pgas.rs:843/1250`) and the **Im5
  canary** comment at `particle_filter.rs:~401`; `pgas.rs::build_obs_at_substep`
  / `SubstepGrid`; `docs/camdl-inference-spec.md` §3 (observation model). This
  touches inference math across every filter — high-risk regardless of how
  mechanical it looks.

## 1. The problem: an asymmetric gap

A model may declare observation streams on different cadences — the DSL and IR
already allow it:

```camdl
observations {
  afp[p in patch] : {
    columns   { time : time, patch : dim, cases : count }
    projected = incidence(paralysis[p])  every = 30 'days
    cases ~ neg_binomial(mean = rho * projected, r = k)
  }
  es[p in patch] : {
    columns   { time : time, patch : dim, conc : real }
    projected = prevalence(I_shed[p])    every = 14 'days
    conc ~ normal(mean = lambda * projected, sd = sigma)
  }
}
```

(Surface per the sibling proposal A: `columns { }`, `[p in dim]` indexing, `~`.)

`camdl simulate` **handles this**: it writes **one TSV per stream** (`--obs-dir`
/ `--obs-only-dir`), each on its own time axis, and it hard-errors if you ask
for a single wide `--obs` file (different cadences cannot share one time column)
— naming the directory escape hatch (`acceptance_obs_only_dir.rs`).

`camdl fit` / `pfilter` / `profile` **reject it.** The inference loaders require
every stream to share **identical** observation times, at four sites:

| site                               | message                                               |
| ---------------------------------- | ----------------------------------------------------- |
| `fit/runner.rs:355`                | "All streams must have identical observation times."  |
| `pfilter.rs:208`                   | "All streams must share identical observation times." |
| `profile.rs:516`                   | "must share identical observation times."             |
| `multi_stream_obs.rs:439` (`bind`) | "obs_times that differ from stream 0"                 |

So a modeller can _simulate_ a polio AFP+ES model and **cannot fit the result
back**. That asymmetry is the gap this proposal closes.

The rejection is the correct **no-silent-gaps** behaviour for machinery that
does not exist — it is not a permanent design stance. The give-away is in the
code: `BoundObs`'s comment reads "the single shared observation axis
(homogeneous across streams **today**)."

## 2. Current architecture (why one `obs_idx` works today)

```rust
struct BoundStream { ir_model, projection, values: Vec<Option<ObsCell>> }
pub struct BoundObs { times: Vec<f64>, streams: Vec<BoundStream> }   // ONE shared axis
```

`bind(streams)` checks every stream's `obs_times` equals stream 0's, then
**collapses** them to one `times`; the invariant `values.len() == times.len()`
holds for every stream. Scoring is keyed by a **single** `obs_idx`:

```rust
fn log_likelihood_from_flows_and_counts(&self, cum_flows, counts, obs_idx, params) -> f64 {
    let t = self.obs_times[obs_idx];                  // one time
    (0..streams).map(|si| match streams[si].observations[obs_idx] {   // SAME idx, every stream
        None => 0.0,                                  // hole: skip the term
        Some(Scalar(v)) => score(project(si, cum_flows, counts, t), v),
    }).sum()
}
```

The flow accumulator `cum_flows: Vec<u64>` is **global, per-transition**;
`FlowSum(indices)` projects `Σ cum_flows[i]`. After every obs substep the loop
**blanket-resets** it (`reset_flows()` → zeroes all of `cum_flows`). The
substep→obs map is `build_obs_at_substep` (`substep → obs_idx`, one shared
schedule).

One `obs_idx` works because it indexes **all three** at once — the time, every
stream's cell, and the (blanket) reset — which is sound _only_ when the axis is
shared.

### The Im5 canary already scoped the fix

The blanket reset carries a comment (`particle_filter.rs:~401`, "Im5",
2026-04-19 inference review) that predicted this feature:

> resets ALL flow accumulators indiscriminately… Safe because: (a) prevalence
> streams don't consume flows; (b) disjoint FlowSum subsets don't share
> accumulator indices; **(c) overlapping subsets both reset to zero anyway.**
> **If a future feature ever stores "flow since the most recent per-stream
> observation" at different cadences per stream, this reset needs to become
> per-flow and indexed by which stream last observed. Keep this comment as the
> canary.**

Condition **(c) is exactly what breaks** under multi-cadence: overlapping flows
reset together today only because the shared axis resets _every_ stream at
_every_ obs time. With AFP and ES on different cadences, an **ES-only union-time
must not reset AFP's incidence bin** — but the blanket `reset_flows()` would.

## 3. Design (types-first)

### 3.1 The union axis

`BoundObs.times` becomes the **union** of all streams' schedules (sorted,
deduped). Each stream is, at each union-index, either **scheduled** (with a
cell: observed value or a hole) or **not scheduled**. The "not scheduled" state
is _new_ and distinct from a hole:

| stream state at union-time `t` | likelihood term | incidence reset     |
| ------------------------------ | --------------- | ------------------- |
| scheduled, observed value      | scored          | yes (if `Interval`) |
| scheduled, **hole** (`None`)   | omitted         | yes (if `Interval`) |
| **not scheduled**              | omitted         | **no**              |

The hole machinery (`Vec<Option<ObsCell>>`) is reused; "not scheduled" is the
new third state. The cleanest representation keeps each stream's own schedule
authoritative and derives membership against the union:

```rust
struct BoundStream {
    ir_model, projection,
    obs_times: Vec<f64>,            // THIS stream's own schedule (authoritative)
    cells:     Vec<Option<ObsCell>>,// values, len == obs_times.len()
    // derived at bind: for each union-index, Some(local_idx) if scheduled here, else None
    at_union:  Vec<Option<usize>>,  // len == BoundObs.times.len()
}
pub struct BoundObs { times: Vec<f64> /* union */, streams: Vec<BoundStream> }
```

`StreamSpec` is unchanged (`{ projection, ir_model, observations, obs_times }`);
the per-stream `obs_times` it already carries simply stop having to be
identical.

**Each stream resets on its own observation grid — not on the union.** A
stream's incidence bins are bounded by **its own observation times**: its
cadence grid for a regular stream (`every = 7 'days`, including
scheduled-but-missing `NA` holes), or simply its present rows for an irregular
stream. The reset fires when the stream is scheduled at a union-time (a member
of its own grid) and **does not fire at a sibling's union-time that the stream's
grid does not contain**. So the classic case — stream A on rows {1, 3, 8},
sibling B at 5, union {1, 3, 5, 8} — is unambiguous: 5 ∉ A's grid, so A does not
reset at 5; A's bin closes at 8 over (3, 8]. No "declared schedule separate from
the values" is needed — for an irregular stream the grid _is_ its rows, and a
sibling's time simply is not in it. (Prevalence streams have no accumulator, so
no reset, so this never arises for them.) Membership is exactly the per-stream
`at_union` map: scheduled (member) vs not-scheduled (non-member), with holes
being members whose value is `None`.

### 3.2 `bind()` merges instead of rejecting (the shared substrate)

The four rejection sites collapse to **one** lift, in `bind()` — the substrate
every path (`fit`/`pfilter`/`profile`/model-layer) routes through, so all four
are fixed at once (the "no silent gaps / shared substrate" rule):

- compute `times = sorted_unique(⋃ streams.obs_times)`;
- for each stream, build `at_union` by matching its `obs_times` into `times`
  (tolerance `1e-9`, the existing equality tol);
- keep the existing per-stream checks (non-empty, strictly-increasing — gh#188);
- the homogeneous-schedule rejection is **removed**.

Downstream invariants change from "every stream's `values.len() == times.len()`"
to "every stream's `cells.len() == obs_times.len()`, and
`at_union.len() ==
times.len()`."

### 3.3 Per-observer reset (the crux) — per-stream accumulators reset on each stream's own grid

Per the Im5 canary, the reset becomes "flow since **this stream's** last
observation" — literally a per-incidence-stream accumulator that resets on the
stream's own grid (§3.1):

- each `Interval` (incidence) stream `s` carries **its own** flow accumulator
  over the flows it projects (summed by its `FlowSum` indices), advanced every
  substep with the dynamics;
- at a union-time where `s` is scheduled (a member of its grid — value or hole),
  `s` is scored against its accumulator, then **`s`'s accumulator resets to 0**;
- at a union-time where `s` is not scheduled (a sibling's time), `s` does
  nothing — its accumulator keeps running toward `s`'s next scheduled time.

This is the bounded, lock-step-safe representation, and it **engineers the
over/underflow risk out** rather than bounding it. Each accumulator holds only
one inter-observation interval's flow (the same bound as today's single
accumulator — population × substeps-per-bin), so there is **no never-reset
counter to overflow**; it is part of the particle state and is copied with the
rest of the particle at resampling (`pgas.rs:1117`, `particle_filter.rs:381`),
so there is **no separate baseline that can fall out of sync under ancestor
swaps**. Two incidence streams sharing a flow are independent because each has
its own accumulator over that flow. The cost is storage
`O(incidence-streams × their-flows × particles)` — a real scaling axis that §7
gates as bounded at the 774-LGA national model.

(A rejected alternative — "one monotonic, never-reset counter plus per-stream
baselines you subtract" — computes the identical bins, but a never-reset `u64`
can overflow on a long national run, and the baseline must be carried in
lockstep with the particle through ancestor resampling or it underflows. The
per-stream accumulator above avoids both by construction, which is why it is
preferred over the subtraction trick.)

The blanket `reset_flows()` at the six sites is replaced by
`reset_due_flows(obs_model, state, union_idx)` — reset exactly the incidence
streams scheduled at `union_idx`. `TemporalKind` is the gate (only `Interval`
streams accumulate and reset). For the homogeneous case (all streams scheduled
at every union-index) this reproduces today's blanket reset exactly — the
**bit-identical-homogeneous test** (§7) is the guard.

> **Gradient note.** Each stream's accumulator is a scalar readout fed to the
> likelihood args (upstream of `d logL / d projected`), and the reset fires at
> the same scheduled substeps in value and gradient paths. So
> `complete_data_loglik_grad` is structurally unaffected — but the §7 test must
> include a near-`k = n` boundary point, where a binding `value ≤ n` cap makes
> the value `-Inf` while the gradient is 0 (an inconsistency NUTS must not be
> misled by).

### 3.4 Scoring + substep mapping generalize

- `log_likelihood_from_flows_and_counts(cum_flows, counts, union_idx, params)`:
  iterate streams, skip any not scheduled at `union_idx`
  (`at_union[union_idx]
  == None`), else project (incidence from the stream's
  own accumulator; prevalence direct) and score the cell (hole → omit).
- `build_obs_at_substep` / `SubstepGrid` map `substep → union_idx` over the
  union axis; the per-stream `at_union` then selects who is due. The snap/exact
  alignment and collision diagnostics are unchanged (they already operate on a
  time list — now the union list).

## 4. What stays loud (capability honesty)

- **Sub-`dt` distinct cadences that collide on the grid** — two union-times
  closer than `dt` under snap alignment still collide; the existing
  `build_obs_at_substep` collision error fires (now naming union-times). Lift
  via `--obs-alignment exact` exactly as today.
- **Prevalence-only multi-cadence** is trivially supported (no flows, no reset);
  the work is entirely about the incidence (`Interval`) reset.

## 5. Why this is well-scoped, not a god-rewrite

- The **simulate side is done** (per-stream `--obs-dir` files); only inference
  changes.
- The **data format is settled** (`[data.observations] stream = file`, one TSV
  per stream — already multi-file).
- The **hole machinery is reused** ("scheduled-but-missing"); only
  "not-scheduled" is new.
- The **four rejections collapse to one lift** in `bind()`.
- The **reset change is localized**: one new `reset_due_flows` replacing the
  blanket reset at six sites, gated by `TemporalKind`.

## 6. The fixture: spatial polio AFP + ES

Adapt `ocaml/golden/polio_spatial_5.camdl` (gravity-coupled patches; the
`importation[p,q] @ kappa * W[p,q] * S[p] * I[q]/N[q]` pattern) down to **2–3
patches**, with two streams at **different cadences**:

- **`afp`** — `incidence(paralysis)`, **monthly** (`every = 30 'days`), low /
  zero-heavy counts (paralysis is a small fraction of infection),
  `neg_binomial`. Exercises incidence reset + low-mean NB + holes.
- **`es`** — `prevalence(I_shed)` (a shedding compartment), **biweekly**
  (`every = 14 'days`), `poisson` for v1. (When `Counted` lands, `es` upgrades
  to binomial positivity — the denominator follow-up meets here.)

A mixed **incidence + prevalence at different cadences** is the hardest case and
the one the union axis must get right.

Synthetic data via `camdl simulate … --obs-dir` (one TSV per stream at its own
cadence) from known params; the fit recovers them. Fixture lives in
`tests/fixtures/` (model) + a `fit.toml` using `[synthetic]` (`true_params` +
`sim_seeds`) so the test is self-contained and recover-known-params.

## 7. Tests (red → green; this is inference math — paste red/green in commits)

1. **`union_axis_per_observer_reset`** (deterministic, like
   `sparse_holes_reset.rs`) — 2 streams, AFP `every=30`, ES `every=14`, fixed
   seed, drainless dynamics so counts are RNG-independent. Assert: AFP's scored
   bin equals the flow over its **30-day** span (not the 14-day union step), and
   an **ES-only union-time does not reset AFP's accumulator** (mutation check:
   forcing a blanket reset makes AFP's bin too small → test fails). This is the
   canary condition, made executable.
2. **`bind_merges_heterogeneous_schedules`** — `bind()` on two different-cadence
   `StreamSpec`s returns `Ok` with `times` = the union and correct per-stream
   `at_union`; the old "identical times" rejection is gone. A 3rd stream that is
   prevalence at a 3rd cadence also binds.
3. **`homogeneous_is_bit_identical`** — all streams on one cadence: union axis
   == shared axis, per-stream reset == blanket reset, loglik byte-identical to
   today. Guards the regression.
4. **End-to-end fit** — the polio AFP+ES fixture: `camdl fit run` (IF2 scout,
   optionally PGAS) on `[synthetic]` data recovers `beta`, coupling `kappa`, and
   the reporting params within tolerance. Both `pfilter` and `fit` accept the
   multi-cadence per-stream files (proving the `bind()` lift reaches every
   path).
5. **Single-patch reduction cross-check** — one patch, AFP+ES, against a hand /
   pomp-style computation of each stream's bin, to anchor the mechanism where an
   oracle is tractable (no oracle exists for spatial multi-cadence).
6. **PGAS gradient consistency** — the §3.3 per-stream reset reaches
   `complete_data_loglik_grad`; finite-difference value-vs-grad on a
   multi-cadence fixture, **including a near-`k = n` boundary point** (§3.3
   gradient note); also pays down the deferred §6.6 test from the burn-in work.

## 8. Implementation phases

1. **Types + `bind()` merge** (§3.1, §3.2) — `BoundStream.at_union`, union
   `times`, remove the homogeneous rejection; the three CLI loaders
   (`runner.rs`, `pfilter.rs`, `profile.rs`) drop their identical-times checks
   (now `bind()` owns it). Tests 2, 3.
2. **Per-observer reset** (§3.3) — per-stream flow accumulators +
   `reset_due_flows` at the six sites; scoring/substep generalization (§3.4).
   Tests 1, 3, 6. Highest-risk; PGAS value + gradient.
3. **Fixture + end-to-end** (§6) — the polio model, `[synthetic]` fit.toml,
   tests 4, 5.
4. **Docs** — `camdl-inference-spec.md` §3 (the union axis + per-observer
   reset), `fit-toml.md` (multi-file per-stream data is already documented; note
   heterogeneous cadences now fit), and retire the "must share identical times"
   language wherever it appears.

## 9. Out of scope (named, not forgotten)

- **`Counted{value,denom}` / survey denominators** — an additive `ObsCell`
  variant on top of this union-axis `BoundObs`; a separate proposal. ES-as-
  binomial-positivity is its natural first consumer.
- **Per-stream conditioning boundaries** — `condition_from` is currently one
  global boundary (`2026-06-09-burnin-conditioning-window.md` §6.10). With a
  union axis, a per-stream boundary becomes a leading reset-only entry on that
  stream's own schedule; a follow-up.
- **`W329` on the `pfilter` path** — the first-window guard is fit-path-only; a
  separate follow-up (orthogonal to this).

## 10. References

- The Im5 canary: `rust/crates/sim/src/inference/particle_filter.rs:~401`
  (2026-04-19 inference review) — predicts this feature and scopes the reset
  fix.
- The four identical-times rejections: `fit/runner.rs:355`, `pfilter.rs:208`,
  `profile.rs:516`, `multi_stream_obs.rs:439`.
- The simulate/fit asymmetry: `acceptance_obs_only_dir.rs` (simulate writes
  per-stream files; `--obs` single-file multi-cadence hard-errors).
- Reset sites: `particle_filter.rs:415`, `if2.rs:319/559`,
  `correlated_pf.rs:521`, `pgas.rs:843/1250`.
- Substep mapping: `pgas.rs::build_obs_at_substep`, `SubstepGrid`.
- Fixture base: `ocaml/golden/polio_spatial_5.camdl` (gravity coupling),
  `seir_observations.camdl` / `surveillance_likelihoods.camdl` (multi-stream
  different-cadence observation DSL).
- The sparse/hole machinery this reuses: `2026-06-06-observation-system.md`.
