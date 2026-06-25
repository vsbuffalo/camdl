# Generated quantities

Status: **Proposed** — implementable as specified. No `ir/VERSION` bump, no
golden regeneration (quantities are a non-identity reporting section).

Splits the quantities half out of the combined
`2026-06-24-generated-quantities-and-counterfactuals.md` (now superseded); the
counterfactual half is `2026-06-25-counterfactual-contrasts.md`. Also supersedes
`2026-06-04-output-trajectory-customization.md` Phase 2.

## Summary

A camdl user cannot yet ask the model to _report a derived quantity_ —
cumulative incidence, attack rate, peak prevalence, time to peak. Today the only
way to compute a function of state is to smuggle it in as a _scored_ observation
stream (the expander requires a likelihood, `E266`), which forces the author to
pretend a reported quantity is data.

This adds a `quantities {}` block: named reductions of a trajectory to reported
summaries. A quantity is the **non-scored twin of an observation** — it reuses
the projection machinery, minus the likelihood — and is valuable across the
whole workflow, not just post-hoc:

- on the **observed data** → summary statistics (peak timing, final size) to
  elicit/adjust priors _before_ fitting;
- on **prior-predictive** simulations → prior-predictive checks;
- on **posterior** draws → reporting (attack rate, peak, cumulative incidence,
  with uncertainty bands).

Nothing here touches the inference kernels, and the shared `Expr` type stays
closed.

## The seam: define vs serialize

camdl already separates _defining_ a computed-from-state quantity from
_serializing_ it. An `ObservationModel` is computed from state and written to a
file, yet it is **defined** in `observations {}` (`model.rs:180`) while
`output {}` only carries the serialization switch
(`output { observations = true }`). A generated quantity is the non-scored twin
of an `ObservationModel`, so it gets its own definition block beside
`observations {}`, not a sub-block of `output {}`.

## Stocks, not flow accumulators

The common cumulative quantities are an **absorbing stock** read directly off
state: `total_deaths = final(D)` (`D` absorbing), `(N0 - S)` (cumulative
incidence when `S` is monotone). A compartment count is a lifetime running stock
(`CurrentPop`, never reset). v1 admits only the non-flow projections; a lifetime
_flow_ accumulator (`cumulative(flow)`, for flows captured by no stock — SIRS
waning, reinfections) is a named follow-up. This deliberately avoids the
observation `CumulativeFlow` projection, which accumulates over a reporting
interval and **resets on the cadence** (`observation.rs:32`) — the opposite of a
lifetime total; reusing it by ambient context would be an illegal second meaning
of one IR value (`temporal_kind()` is derived-single-source by design,
`observation.rs:23`).

## Types

```rust
/// The non-scored twin of an ObservationModel: a derived function of state, no
/// likelihood. A named reduction of a trajectory to a reported summary.
pub struct Quantity {
    pub name:       String,
    pub strata:     Vec<StratumKey>,        // empty = whole-pop; reuses obs strata
    pub projection: StockProjection,        // the three non-flow variants only
    pub reduce:     Option<TemporalReduce>, // None = a series; Some = a scalar
}
```

### `StockProjection` — flow projections are unrepresentable, not runtime-checked

```rust
/// The subset of Projection a v1 quantity may use — no flow accumulators, so the
/// reset/differencing hazard cannot arise. Constructed from a Projection by a
/// fallible parse that rejects the two flow variants at the boundary.
pub enum StockProjection {
    CurrentPop(String),
    CurrentPopSum(Vec<String>),
    DerivedExpr(Expr),       // arithmetic over state: I/N, (N0 - S)/N0
}
```

A `Quantity` carrying a `StockProjection` _cannot_ hold `CumulativeFlow` — the
restriction is in the type, not a check (the "make illegal states
unrepresentable" rule, against reusing the full 5-variant `Projection` and
validating at runtime).

### `TemporalReduce` — result kind is typed, not underdetermined

A flat reduction enum would let `argmax` (returns a _time_) and `max` (returns a
_value_) share one shape, leaving the output dimension a function of the runtime
variant. Split so the result kind is in the type:

```rust
pub enum TemporalReduce {
    Value(ValueReduce),    // result has dim(series):  final, max, min, mean
    Time(TimeReduce),      // result has dim T:         time_of_max, time_of_min
    Integral,              // result has dim(series)·T: area under the curve
}
pub enum ValueReduce { Final, Max, Min, Mean }
pub enum TimeReduce  { ArgMax, ArgMin }
```

`None` reduce → a **series** (one value per output time). `Some` → a **scalar**
(the series reduced over time). This is the third reduction axis (over _time_),
intentionally distinct from `ObsReducer` (`intervention.rs:90`, folds a trigger
window) and the n-ary `Reduce` (`expr.rs`, sums over _strata_ at one instant) —
neither is temporal.

**Resolution.** v1 reductions fold over the **output-cadence snapshots**
`fit
predict` already produces. A `peak`/`time_of_max` between output times can
be missed at a coarse `every`, so a fine default cadence is documented for
peak-sensitive quantities, and a resolution-honest **substep fold is a named
fast-follow** (a streaming reducer in the forward step loop, orthogonal to
everything else). Reading the _endpoint_ (`final`, stocks, totals) is
cadence-invariant and exact regardless.

### Model — a non-identity section (no re-key)

```rust
pub struct Model {
    // …existing…
    pub quantities: Vec<Quantity>,   // skip_serializing_if empty
}
```

**Run-identity:** quantities do not change the simulation or the fit — they are
derived reports. So the field is **excluded from `Model::hash_into`** (the
hand-written run-id walk, `ir_hash.rs`), exactly the control the hand-written
hash exists to give. Combined with `skip_serializing_if` empty, an existing
model is **byte-identical** in both its serialized IR and its `run_id` — so **no
`ir/VERSION` bump and no golden regeneration**. A model that adds quantities
keeps the same fit `run_id`; the quantities re-key only the _predict-time_
output that reports them.

## DSL surface

```
compartments { S, E, I, R, D }

quantities {
  prevalence       = I / N                  # series   (DerivedExpr)
  attack_rate      = final((N0 - S) / N0)   # scalar   (stock-derived)
  total_deaths     = final(D)               # scalar   (absorbing stock)
  peak_prevalence  = max(I / N)             # scalar   (value reduction)
  time_to_peak     = time_of_max(I)         # scalar   (time reduction → a time)
  person_days_inf  = integral(I)            # scalar   (dim P·T)
}
```

Stratified, reusing the observation stratum form:

```
quantities {
  prevalence[p in patch]  = I[p] / N[p]
  attack_rate[p in patch] = final((N0[p] - S[p]) / N0[p])
}
```

Grammar additions: a `quantities {}` top-level block (one new keyword + a
`name = expr` body, the shape of `ode_list`); the reductions `final`, `max`,
`min`, `mean`, `time_of_max`, `time_of_min`, `integral` as **reserved names**
legal only inside `quantities {}`. They lex as ordinary `EFuncCall`, so a
reduction name used in a transition rate parses and is rejected at expansion
with a **dedicated diagnostic** (`E2xx`: "temporal reduction `max` is only valid
in `quantities {}`; a rate is evaluated per substep"), not the generic `E100`;
reserved against collision with a `forcing {}` function of the same name.

## Evaluation — three trajectory sources

A quantity is evaluated by replaying/reading a trajectory and folding its
reduction.

- **`camdl fit predict <fit>`** → over each posterior draw, banded into
  `quantities/<name>.tsv` (the reporting use). Reuses the #298 draw-replay loop.
- **`camdl simulate --draws <prior>`** → over prior-predictive draws → prior
  checks.
- **`camdl simulate`** at fixed params / **over the observed data** → a point
  summary (data summary statistics for prior elicitation). Reducing the observed
  data uses the same reduction over an observed series rather than a simulated
  one — the pre-conditioning path.

`fit predict` with an empty `quantities {}` writes no quantity files (not an
error); quantities band over **all** posterior draws.

## Expressivity — worked surveillance questions

Two v1 idioms carry most real questions: `time_of_max` of a step function yields
any **threshold-crossing time** (the step is 0 before the crossing, 1 after, and
`time_of_max` returns the first time the max is hit), and the **band over draws
is the distribution**. Against eleven polio-surveillance questions:

**Direct v1** (`i_thresh` a declared `count` param; `I_total` the patch sum):

```
quantities {
  peak_month     = time_of_max(I_total)                                   # Time reduction → a date when anchored
  takeoff_time   = time_of_max(if I_total > i_thresh then 1.0 else 0.0)   # first threshold crossing
  outbreak_size  = final(N0 - S)                                          # final size; the band IS the size distribution
  peak_time[p in patch] = time_of_max(I[p])                              # per-patch timing (raw material below)
}
```

— so peak month, epidemic takeoff timing, the outbreak-size distribution, and
per-patch timing are clean v1. Every "when does X first happen" is
`time_of_max(if … then 1 else 0)`.

**v1 endpoints + one downstream step.** Silent-circulation duration,
ES-before-AFP lead time, and extinction probability each reduce to v1 quantities
plus a trivial post-step on the banded TSVs:

```
quantities {
  introduction_time = time_of_max(if I_total > 0 then 1.0 else 0.0)   # silent circulation =
  detection_time    = time_of_max(if afp     > 0 then 1.0 else 0.0)   #   detection − introduction
  es_first  = time_of_max(if es  > 0 then 1.0 else 0.0)               # ES lead = afp_first − es_first
  afp_first = time_of_max(if afp > 0 then 1.0 else 0.0)
  peak_prev = max(I_total)                                            # P(extinct) = fraction with peak_prev < threshold
}
```

The differences / tail-fractions are one-liners on the output today; inline once
reduction arithmetic lands (roadmap below).

**Cross-patch statistics.** Spatial ordering and synchrony give the per-patch
`peak_time[p]` in v1; the ordering (argsort over patches) and the spread
(variance over patches) are reductions over **strata**, not time — downstream
today, a new cross-stratum axis as a follow-up.

**New machinery.** Number of distinct waves and inter-wave gaps need **peak
detection** (count local maxima above a prominence); persistence after SIAs
needs a **windowed** quantity (`max(I) over [last_sia, end]`). Genuine
follow-ups.

The v1 primitives cover ~7 of the eleven fully or as banded raw-material; the
misses cluster into exactly the follow-ups below, ordered by value.

## Dimensional checking

Each quantity's projection expression runs through the existing `dimcheck` (a
lone quantity needs no cross-quantity comparison, so no stored dimension is
required — that is a contrast concern, deferred to the counterfactual proposal).
The reduction's output dimension is derived: `Value` preserves `dim(series)`;
`Time` (`time_of_max`) yields `T`; `Integral` yields `dim(series)·T` (the
`dim·T` algebra already exists, `dimcheck.ml:17`). A series whose expression is
dimensionally ill-formed is rejected as today.

## Output format

TSV, **one file per quantity** (`quantities/<name>.tsv`), **long/tidy keyed by
stratum level**, bands reusing `fit predict`'s `q05 q25 q50 q75 q95` over draws.
One-per is required: a _series_ quantity has a `time` column, a _scalar_ does
not.

```
# series quantity, stratified     quantities/prevalence.tsv
time   patch    n_draws  q05 q25 q50 q75 q95

# scalar quantity, stratified     quantities/total_deaths.tsv   (no time axis)
patch  n_draws  q05 q25 q50 q75 q95
```

A `quantities.json` manifest lists each entry's name, shape (series|scalar),
strata, reduction, and **unit**. Resolved spec points: `time_of_max` returns the
**first** argmax on ties; a `Time` reduction's value is a duration-from-origin
in an unanchored model and is rendered as a **date** in an anchored model (the
manifest records which); a quantity name that collides with a
compartment/param/observation is a duplicate-name error.

## Staging and follow-up roadmap

**v1.** The `quantities {}` block, `StockProjection`, typed `TemporalReduce`,
whole-horizon output-cadence reductions, evaluation over data / prior-predictive
/ posterior, the manifest. No re-key.

Follow-ups, ordered by how many real questions each unlocks (from the
walkthrough):

1. **Reduction arithmetic** — difference/ratio of two reductions and `Cond` on a
   reduced scalar. Unlocks silent-circulation duration, ES-AFP lead, and inline
   extinction indicators. Cheapest, highest leverage — do first.
2. **Cross-stratum reductions** — rank / variance / correlation over strata (a
   new axis complementing over-time). Unlocks spatial ordering and synchrony.
3. **Waveform reductions** — peak counting with a prominence threshold. Unlocks
   number-of-waves and inter-wave gaps. Genuinely new, non-trivial.
4. **Substep recorder** — resolution-honest `max` / `time_of_max` / `integral`
   (orthogonal; sharpens every timing/peak quantity).
5. **Windowed quantities** — `over [X, Y]` with instant endpoints. Unlocks
   after-SIA persistence; shared with the counterfactual proposal.
6. **Flow `cumulative`** + flow arithmetic; `at_time(series, t)`.

## Decisions recorded

- Separate `quantities {}` block (the define/serialize seam).
- Restricted `StockProjection` (flow projections unrepresentable in v1).
- Reduction result kind typed (`Value`/`Time`/`Integral`).
- Quantities are a **non-identity reporting section** — no `run_id` re-key, no
  golden regeneration.
- Reductions are quantity-scoped, enforced by a dedicated diagnostic.
- Output-cadence reductions in v1; substep recorder a fast-follow.
