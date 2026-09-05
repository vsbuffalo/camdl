# Observation time as a sum type

- **Status:** Draft
- **Issue:** gh#833
- **Supersedes:** `2026-09-04-explicit-observation-windows.md`
- **Area:** IR (`ObservationModel`, observation rows), runtime
  (`multi_stream_obs`, `particle_filter`), DSL surface (`observations {}`), data
  format, trajectory and observation output
- **Required reading:** `docs/dates.md` §"The calendar axis" — this proposal
  uses its vocabulary and does not re-derive it; `docs/camdl-data-spec.md`;
  `rust/crates/sim/src/inference/multi_stream_obs.rs`

## Summary

An observation carries one scalar time and no statement of what it means. That
single fact is the root of a family of defects: a dated row is read one day
earlier than a surveillance file intends, a count covering several days cannot
say so, a gapped window has nothing to lower onto, and `simulate`'s own output
cannot be read back.

This proposal replaces the scalar with a sum type — an observation's time is
either an **instant** or an **interval**, and which one is decided by the
projection rather than by a convention. The DSL gains a way to state a stream's
period, the data format gains the columns to carry it, and the output side emits
the same shape it reads.

The design principle is the point, not a side effect: **make the illegal state
unrepresentable.** Every ambiguity below exists because a type permits two
meanings for one value. None of them needs a rule once the type stops allowing
it.

## Problem

### One scalar, two meanings

The runtime's observation row is

```rust
pub struct Observation { pub time: f64, pub value: f64 }
```

and the engine scores it over `(previous, this]`. So a row's time is an
**interval endpoint** for an incidence stream and an **instant** for a
prevalence stream, with nothing in the type to distinguish them.

Drawn on the axis from `docs/dates.md`, with `origin = date("2026-07-01")`:

```
internal t     0         1         2         3
               |---------|---------|---------|
date()       07-01     07-02     07-03     07-04

civil day    [ 1 Jul   )[ 2 Jul   )[ 3 Jul  )

camdl reads  "07-02" as (0,1]  = 1 Jul
a file means "07-02" is [1,2)  = 2 Jul
                        ^^^^^ one unit apart
```

Every one of the following is that same missing distinction, surfacing somewhere
else.

### What it costs, concretely

**A dated file is read a day early.** Verified: a burst of transmission confined
to `[0,1)` — declared numerically, no dates involved — is reported on the row
labelled `2026-07-02`.

```
t         0         1         2
          |---------|---------|
burst     [==ON=====)  off
output              83209     990
row label          "07-02"  "07-03"
```

Invisible while every date in a model is a data row, because the whole series
shifts together. Visible the moment a _point_ also carries a date: an
intervention at `date("2026-07-04")` affects `(3,4]`, while the row a file
labels `"07-04"` is scored over `(2,3]`.

**A multi-day count cannot be stated.** The only encoding that widens a window
is deleting the intervening row, which `docs/camdl-data-spec.md` separately
forbids. A downstream pipeline resolved this by deleting rows anyway and
guarding the deletions with a register of its own — correctness living in a
build script because the format could not hold it.

**`origin` cannot repair it.** It translates every coordinate identically, so
the span between two dated rows is invariant:

```
origin = 2026-07-01   row "07-02" -> (0,1] -> 1 Jul
origin = 2026-06-30   row "07-02" -> (1,2] -> 1 Jul
```

**A window cannot be widened.** An interior reset-only boundary already exists —
an `NA` row is exactly that, and closes an incidence bin without scoring it
(`multi_stream_obs.rs`: "a hole still closes the fixed incidence bin on
schedule"). Measured: removing an interior `NA` at `t=2` changes the `t=3` bin
from `(2,3]` to `(1,3]` and the log-likelihood from −40.86 to −148.83. So
_narrowing_ is expressible today. What is not is **widening** — merging rows —
because the only construction that widens a window is deleting the intervening
row, which the data spec forbids.

**Output has the same defect with no data file involved.** A trajectory row
carries a state reading and a flow reading under one date column; the column is
correct for the first and misleading for the second.

## Design

### The type, first

The sum sits at the **stream**, not the row. A stream is uniformly one kind —
its projection decides — and a per-row discriminant would sit in the particle
filter's inner loop for no information gain.

```rust
/// When a stream's observations happen. The variant is fixed by the stream's
/// projection: `Interval` for `incidence(...)` and other accumulating
/// projections, `Instant` for `prevalence(...)` and state reads.
pub enum StreamTimes {
    /// Read the state at each instant. One boundary per observation.
    Instants(Vec<f64>),
    /// Accumulate over each half-open period. Two boundaries per observation:
    /// reset at `start`, score at `stop`.
    Intervals(Vec<Period>),
}

/// A half-open observation period, `[start, stop)`. Constructed only through
/// `Period::new`, which is the single place the invariants hold.
pub struct Period { start: f64, stop: f64 }

impl Period {
    /// The only constructor. `stop > start` and both finite, or an error
    /// naming the stream and the row.
    pub fn new(start: f64, stop: f64) -> Result<Self, ObsError> { … }
    pub fn start(&self) -> f64 { self.start }
    pub fn stop(&self)  -> f64 { self.stop }
    pub fn width(&self) -> f64 { self.stop - self.start }
}
```

Three things become **unrepresentable** rather than validated:

- An interval observation with no stated span. There is no `Period` without both
  endpoints.
- A zero- or negative-width window. `Period::new` is the only way in.
- A window on a stream that reads an instant. `Instants` has no `Period` to put
  one in.

And one becomes _derivable_ rather than conventional: whether a row's label is
its opening or its closing boundary is no longer a question, because the row
carries both.

This is the same move as `Projection::temporal_kind`, which the IR already
documents as _"a derived classification of `Projection`, never a stored field —
an independently-stored kind could only ever disagree with the projection and
would be an illegal state to validate against."_ `StreamTimes` is that
classification made load-bearing: the kind now selects the representation
instead of annotating it.

### Why this is the right shape

The evidence is that a single type change dissolves a family of
unrelated-looking defects rather than addressing them one at a time. The
dated-row offset, the multi-day count, the gapped window, the un-round-trippable
output and the `origin` non-fix were all filed separately and reasoned about
separately. None of them needs a rule once an observation's time says what it
is.

That is the test a design should pass: not that it handles the cases, but that
the cases stop being distinct.

### The DSL surface

A stream states its period. The primitive is an explicit pair; the rest are
named forms that lower to it.

```camdl
# Instant stream — unchanged, and now unable to carry a window.
seroprev {
  columns   { time : time, n_sero : count, seroprev : count }
  projected = R / N
  seroprev  ~ binomial(n = n_sero, p = projected)
}

# Interval stream, per-row windows from the file.
cases_ituri {
  columns     { onset_from : window_start, onset_stop : window_stop,
                cases_ituri : count }
  # `columns {}` requires exactly one `: time` column today (E275). That
  # constraint GENERALIZES rather than relaxes: a stream declares exactly one
  # temporal anchor, which is either a `time` column or a window pair. The
  # fit time source for a windowed stream is `window_stop`, so every
  # downstream "observation time" consumer keeps the position it has today.
  projected   = incidence(confirm[ituri])
  cases_ituri ~ neg_binomial(mean = p_report * projected, r = k)
}

# Interval stream, uniform windows from a labelling rule.
cases {
  columns   { time : time, cases : count }
  covers    = day(time)                 # row D covers [D, D+1)
  projected = incidence(infection)
  cases     ~ poisson(rate = projected)
}
```

| form                                   | the period a row covers      |
| -------------------------------------- | ---------------------------- |
| `window_start` + `window_stop` columns | exactly `[start, stop)`      |
| `covers = day(t)`                      | `[t, t + 1 day)`             |
| `covers = starting_on(t, Δ)`           | `[t, t + Δ)`                 |
| `covers = ending_on(t, Δ)`             | `[t + 1 day − Δ, t + 1 day)` |

`ending_on` is the one that earns a named form: _"week ending 11 July"_ means 11
July is the **last included day**, so the span is `[5 Jul, 12 Jul)`. That
off-by-one is what the constructor exists to hide.

There is no form that spells today's implicit `(previous row, this row]`. A
compatibility form would preserve exactly the behaviour this proposal
establishes to be wrong, and pre-1.0 policy forbids shims. The consequence is
stated rather than softened: a uniform daily stream migrates in one line to
`covers = day(time)` with byte-identical results, and an irregular stream's
likelihood moves, because it was scoring windows nobody wrote down.

**An interval stream must state its period.** There is no default, because no
default is right: `2026-07-08` is `[7 Jul, 8 Jul)` in a daily file,
`[8 Jul, 15 Jul)` in a week-starting one, and `[2 Jul, 9 Jul)` in a week-ending
one. A rule that guesses is silently wrong for someone. The error names the four
forms above.

### Data shapes

The bridge between the type and what people are handed:

| stream kind          | data shape                                             | typical source                                          |
| -------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| `Instants`           | one time column                                        | census, serosurvey, bed occupancy                       |
| `Intervals`, uniform | one time column + `covers = day/starting_on/ending_on` | daily case counts, ISO or MMWR weeks                    |
| `Intervals`, per-row | `window_start` + `window_stop` columns                 | irregular reporting, merged rows, suspended publication |

The third row is the case with no expression today and the one that motivated
this. The second is the common case and stays a one-line declaration.

### Output emits what the loader reads

The same distinction, on the way out. A state column is an instant reading; a
flow column is a period reading; they stop sharing one label.

```
# instant columns
t     date        S     I
5     2026-07-06  9895  73

# period columns
t_start  t_stop  date_start   date_stop    flow_infection
4        5       2026-07-05   2026-07-06   35.8401
```

This is not tidiness. It is what closes the round trip: `simulate --obs-dir`
emits the columns the loader accepts, so synthetic data carries its windows
instead of having them re-inferred. gh#830 and gh#831 both reduce to it.

It also removes the standing readability trap — a flow labelled only by its
closing boundary reads as an instant, and every reader has to be told otherwise.

### What this means for the synthetic-data round trip

The design-preserving `simulate` work (gh#831) needs synthetic data that carries
the _real_ observation design — times, `NA` placement, covariate columns,
windows — so that a self-consistency fit is scored against the same design as
the real one.

Under today's representation a window is implicit in row spacing, so a rule of
the form "reproduce every column and redraw only the scored values" cannot
preserve it: a file whose row covers five days because publication was suspended
yields synthetic data whose row covers one, and the test ends up comparing two
different observation designs.

With a `Period` on the row, that rule becomes correct by construction. This is a
consequence of the design rather than independent evidence for it — the same
author reasoned about both — but it is the concrete reason gh#831 cannot be
built cleanly on the current representation.

### What moves in the runtime

- **The union axis** (`BoundObs::bind`) merges boundaries rather than times. An
  `Intervals` stream contributes two per observation.
- **`reset_due_acc`** splits: reset at a `start`, score at a `stop`. The
  existing leading reset-only hole (`runner.rs`, `condition_from`) generalizes
  to an interior one rather than a parallel mechanism being added.
- **`Schedule { obs_times, Cursor { obs_idx } }`** — `obs_idx` is the union
  index every filter loop passes to both the reset and the score. Two boundaries
  per observation severs its 1:1 correspondence with observations. This is the
  largest single change and was previously unnamed.
- **`reset_due_acc_real` and `reset_due_acc_real_blocks`** — the ODE and
  gradient-block siblings, 11 call sites, six on surfaces `CLAUDE.md` names
  high-risk. The gradient accumulator must close on the same schedule as the
  value accumulator, so reset and score diverging for the first time is a
  correctness question, not a refactor.
- **`ObsTimes`** (`boundary_times.rs`) — its `reject_non_increasing` must become
  non-decreasing, since contiguous windows produce the boundary list `0,1,1,2`.
- **`if2::Observation`** — a second, live `{time, value}` row type used by PMMH
  and constructed by `fit predict`. It is a compat shim in an alpha that forbids
  them; delete it before this arc starts rather than migrating two types.
- **`t` inside the likelihood expression** — `eval_likelihood_resolved` passes
  an observation time into expressions that may reference it (a reporting ramp).
  On a five-day window neither endpoint is right and the honest answer is an
  integral. Refuse a `t`-referencing likelihood on a multi-step period until
  that is settled.
- **Consumers keyed on a scalar `o.time`** — likelihood evaluation, three
  duplicate-detection sites, `--score-from`, `compare`'s fairness gate,
  `fit predict`'s forecast grid, `check_first_interval_window` (W329, whose
  entire premise is inferring the window), `fit/dt_check.rs`,
  `output_schema.rs`. Roughly 68 `.time` sites across 29 files. Both boundaries
  are `f64`, so a wrong pick compiles — the migration needs a `Boundary` newtype
  or an action-list representation, not case-by-case judgement.
- **`compare` gains a window-equality gate.** Two fits of one file differing
  only in declared period have identical observation times and different
  likelihoods; without the gate `paired_delta` reports that as a
  model-comparison result.

### Rules the type does not carry

`Vec<Period>` has no cross-element invariant, so these are expressible and must
be ruled on rather than assumed away. The type removes three per-value illegal
states; these are the relational ones:

| state                                                       | rule                                                                                                                            |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| periods overlap within a stream                             | error — the span would be scored twice                                                                                          |
| periods out of order                                        | error; today's guards are scalar and differ in strictness across three sites                                                    |
| a period narrower than `dt`                                 | error, as today for observations, but it can now fire on a boundary the user never wrote                                        |
| a period starting before `t_start` or `condition_from`      | error naming both                                                                                                               |
| an unintended gap under a uniform form                      | a missing row under `covers = day(t)` is an **error**; a gap stated by per-row columns is legal and discards the uncovered flow |
| `condition_from` on a stream that declared its first period | error — a stated window must not be silently truncated                                                                          |

The last two are the ones that decide behaviour rather than tidiness. A missing
row under a uniform declaration is ambiguous — it could be a hole or a merge —
and today's widening is _correct_ for a merge. Forcing that case onto per-row
columns is the point: the merged row is exactly what the format could not state.

### Naming

`period` as a field sits beside `Period` the type, "period" as used in
`docs/dates.md`, and `sum_observed(stream, window = D)` already in the language.
Four things, three words. The field is named **`covers`** to keep `period` for
the concept and the type:

```camdl
cases { covers = day(time) }
```

`docs/dates.md` names the future output columns `period_start`/`period_stop`;
this proposal emits `t_start`/`t_stop`/`date_start`/`date_stop`. Settle on the
`dates.md` names before either ships.

## IR, run identity, goldens

`ObservationModel` gains the period declaration; the observation row
representation changes shape. **`ir/VERSION` bumps** — the human-loop
confirmation required by `CLAUDE.md` has been given for this change. It touches
54 files under `ocaml/golden/` and 18 under `ir/golden/`, and `ir/golden/` is
not regenerated by `make update-golden`, so it needs a hand path.

The period enters run identity: it changes what the likelihood scores, so two
fits differing only in it must not share an address. It is **not** guarded to
hash only when non-empty — the declaration changes what the likelihood scores,
so the field re-keys every model and orphans stored runs. `--resume` breaks
across the bump, since the fit config hash lives in `resume_state.bin`.

If other re-keying changes are pending, land them in one bump.

## Testing

1. **Red first.** A daily incidence stream declaring `covers = day(time)` with a
   one-day hole is accepted today and silently scores a two-day window. After:
   refused, naming both times and pointing at the per-row form, which is where a
   genuinely merged row belongs.
2. A uniform daily stream declaring `covers = day(time)` reproduces today's
   trajectory and log-likelihood **byte-identically**. This is the
   no-change-for-the-common-case claim and it is asserted, not assumed. An
   irregular stream is expected to move, and the test records by how much.
3. A row with `window_start`/`window_stop` spanning two days scores its count
   against two days of flow. Oracle: a hand-computed likelihood at fixed
   parameters.
4. A declared gap discards the flow in the uncovered span, checked against a run
   where that span carries known flow.
5. `Period::new` rejects zero and negative width; the type admits no other path.
6. Window columns on an `Instants` stream do not compile.
7. Round trip: `simulate --obs-dir` on a windowed model emits a file the same
   model reads back with identical periods and an identical likelihood.
8. `compare` refuses to pair a migrated fit against an unmigrated one.
9. Output: a flow row's `[t_start, t_stop)` matches the interval its value was
   accumulated over, on all three backends.

## Phasing

0. **Clear the ground.** Delete the duplicate `if2::Observation`; fix gh#837
   (`modal_value`), on which W329 rests. Independent of everything below.
1. **`Period` and `StreamTimes` in the IR and runtime.** The bump and the
   goldens land here. This is where `reset_due_acc` splits and `Schedule`'s
   `obs_idx` stops corresponding 1:1 with observations — the correctness step.
2. **The DSL surface** — `window_start`/`window_stop` column kinds and the
   `covers` declaration, with the required-declaration rule and its diagnostics.
   Every model in the repo migrates here; nothing compiles until it does.
3. **`compare`'s window gate**, no later than 2 in wall-clock: declared periods
   become possible at 2, and the gate is what stops a migrated/unmigrated pair
   reading as a model-comparison result.
4. **Output columns**, closing the round trip; unblocks gh#830 and gh#831. The
   `fit predict` grid follows.

Steps 0 and 2 are wide and mechanical. Step 1 is narrow and is the one where a
wrong choice is a silently wrong gradient rather than a red test.

## Relationship to other work

- **gh#847** — a `Duration` silently becoming an `Instant` in `simulate { to }`
  is the same defect one layer up: one dimension, two meanings, no type to
  separate them. Not fixed here, because closing it breaks every anchored model
  that writes `to = 40 'days`. Worth solving with the same instinct.
- **gh#839** — sub-day `--dates` output the loader rejects. The output change
  above should settle the representation rather than leaving a private suffix.
- **gh#836** — negative increments and merge decisions. This proposal lets a
  merge decision be _written down_; it does not decide what to merge.

## Alternatives considered

**Shift the loader by one day.** Makes right-labelled files align with no new
syntax. Rejected: it is a guess that is silently wrong for week-starting files,
and it moves every existing dated fit without anyone writing anything.

**Keep the scalar and add a width field.** Cannot express a gap, and leaves the
zero-width and wrong-stream states representable — the validation burden stays.

**A per-row sum type.** Correct but puts a discriminant in the filter's inner
loop for information that is constant per stream.

**Document the convention and stop.** The convention cannot be documented
correctly, because there is no single correct one — the same date column means
three different spans depending on the source.
