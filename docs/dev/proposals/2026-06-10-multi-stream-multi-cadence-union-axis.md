# Multi-stream multi-cadence: the union observation axis + per-observer reset

- **Status:** Draft — design review. No code yet. Scope is the **inference**
  side only: let `camdl fit` / `pfilter` / `profile` consume observation streams
  on **different schedules** (e.g. polio AFP monthly + ES biweekly). The
  `simulate` side already produces multi-cadence data (one TSV per stream).
- **Issue:** the remaining gap from the sparse-observation lift. gh — (file an
  issue on landing).
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
  afp : { projected = incidence(paralysis)  every = 30 'days  likelihood = neg_binomial(mean = rho * projected, r = k) }
  es  : { projected = prevalence(I_shed)     every = 14 'days  likelihood = poisson(rate = lambda * projected) }
}
```

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

**The reset keys on a _declared_ schedule, not value presence.** For a stream
with a declared cadence (`every = 7 'days`), "scheduled at `t`" is defined by
the cadence, independent of whether a value is present — so a missing value
there is a _hole_ (resets) and a union-time the cadence does not include is _not
scheduled_ (no reset). The only ambiguous case is a stream whose schedule is
**data-defined** (irregular, "whenever it reports"): a union-time injected by a
_sibling_ stream falls in a gap that is genuinely neither, and hole-vs-not-
scheduled give different incidence bins with nothing in the data to
disambiguate. The rule that removes the ambiguity: an **incidence (`Interval`)**
stream's reset keys on its **declared opportunity schedule** — its `every`, or,
for a genuinely irregular stream, an explicitly declared schedule (proposal A
§4) distinct from its present values — **never** the present-cell union.
(Prevalence streams have no reset, so this does not arise for them.) This also
dissolves the "exclude all-hole union times" tension: a reset is keyed on the
stream's own schedule and fires whether or not the union axis happens to carry
that instant, so excluding a union-time with no present cells can never drop a
scheduled reset.

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

### 3.3 Per-observer reset (the crux) — baseline subtraction

Per the Im5 canary, the reset becomes "per-flow, indexed by which stream last
observed." The recommended representation is **baseline subtraction**, which is
correct even when two incidence streams share a flow index:

- the substep loop keeps **one monotonic** `cum_flows: Vec<u64>` (accumulates
  all transitions; **never blanket-reset**);
- each `Interval` (incidence) stream `s` carries a scalar **baseline** `b_s` =
  `Σ_{i∈s.flows} cum_flows[i]` as of `s`'s last scored observation;
- `s`'s projected incidence over its current bin =
  `(Σ_{i∈s.flows} cum_flows[i]) − b_s`;
- "reset `s`" = `b_s ← Σ_{i∈s.flows} cum_flows[i]` (re-baseline) — fired
  **only** when `s` is scheduled at this union-time.

Two incidence streams sharing flow `i` are independent: each subtracts its own
baseline; re-baselining one never touches the other. This dissolves the (b)/(c)
fragility the canary flagged. Storage is one `u64` per incidence stream per
particle (negligible vs the state vector). `Instant` (prevalence) streams carry
no baseline and never reset (they read state at their scheduled instant).

The blanket `reset_flows()` at the six sites is replaced by
`reset_due_baselines(obs_model, state, union_idx)` — re-baseline exactly the
incidence streams scheduled at `union_idx`. `TemporalKind` is the gate (only
`Interval` streams have baselines). For the homogeneous case (all streams
scheduled at every union-index) this reproduces today's semantics exactly — the
**bit-identical-homogeneous test** (§7) is the guard.

> **Gradient note.** `projected` stays a scalar fed to the likelihood args; the
> baseline only changes how that scalar is computed (upstream of
> `d logL / d projected`). So `complete_data_loglik_grad` is structurally
> unaffected — same per-stream re-baseline at the same scheduled substeps.

### 3.4 Scoring + substep mapping generalize

- `log_likelihood_from_flows_and_counts(cum_flows, counts, union_idx, params)`:
  iterate streams, skip any not scheduled at `union_idx`
  (`at_union[union_idx]
  == None`), else project (incidence via baseline
  subtraction; prevalence direct) and score the cell (hole → omit).
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
- The **reset change is localized**: one new `reset_due_baselines` replacing the
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
   an **ES-only union-time does not re-baseline AFP** (mutation check: forcing a
   blanket reset makes AFP's bin too small → test fails). This is the canary
   condition, made executable.
2. **`bind_merges_heterogeneous_schedules`** — `bind()` on two different-cadence
   `StreamSpec`s returns `Ok` with `times` = the union and correct per-stream
   `at_union`; the old "identical times" rejection is gone. A 3rd stream that is
   prevalence at a 3rd cadence also binds.
3. **`homogeneous_is_bit_identical`** — all streams on one cadence: union axis
   == shared axis, baseline reset == blanket reset, loglik byte-identical to
   today. Guards the regression.
4. **End-to-end fit** — the polio AFP+ES fixture: `camdl fit run` (IF2 scout,
   optionally PGAS) on `[synthetic]` data recovers `beta`, coupling `kappa`, and
   the reporting params within tolerance. Both `pfilter` and `fit` accept the
   multi-cadence per-stream files (proving the `bind()` lift reaches every
   path).
5. **Single-patch reduction cross-check** — one patch, AFP+ES, against a hand /
   pomp-style computation of each stream's bin, to anchor the mechanism where an
   oracle is tractable (no oracle exists for spatial multi-cadence).
6. **PGAS gradient consistency** — the §3.3 baseline re-baseline reaches
   `complete_data_loglik_grad`; finite-difference value-vs-grad on a
   multi-cadence fixture (also pays down the deferred §6.6 test from the burn-in
   work).

## 8. Implementation phases

1. **Types + `bind()` merge** (§3.1, §3.2) — `BoundStream.at_union`, union
   `times`, remove the homogeneous rejection; the three CLI loaders
   (`runner.rs`, `pfilter.rs`, `profile.rs`) drop their identical-times checks
   (now `bind()` owns it). Tests 2, 3.
2. **Per-observer reset** (§3.3) — per-stream baselines + `reset_due_baselines`
   at the six sites; scoring/substep generalization (§3.4). Tests 1, 3, 6.
   Highest-risk; PGAS value + gradient.
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
