# Dates and calendar time

Epidemic models meet several different kinds of calendar time, and they are not
interchangeable. A model may contain:

- an **instant** — a vaccination campaign begins on 5 July; a hospital census is
  taken at a particular moment; a seeding event occurs at a particular time;
- a **flow over a period** — 17 cases occurred on 5 July; 83 deaths were
  reported for the week ending Saturday 11 July; a total covers 5–11 July;
- a **duration** — a mean infectious period of 7 days, a latent period of 5.2
  days;
- a **calendar displacement** — six calendar months after 31 January, which is
  not the same thing as 182.6 days later;
- and, cutting across all of the above, **which event a date refers to** —
  symptom onset, specimen collection, laboratory confirmation, notification,
  admission, death.

That last one is the axis most often overlooked. A compartmental model generates
latent and observable events; a surveillance system may timestamp a different
stage of the same process — onset, specimen collection, confirmation,
notification, admission, death. When the modelled event and the recorded
timestamp differ, the intervening delay is part of the observation process.

Ignoring it misaligns the series in time. Whether it also _biases_ inference
depends on the delay: a stationary delay distribution shifts the curve without
changing its exponential growth rate, while a delay that varies over time or
interacts with epidemic dynamics can bias growth, intervention effects and
forecasts. Where it matters it is modelled (`via`, `lag`, a convolution), not
absorbed.

Nothing in a `2026-07-05` cell says which of these it is. Two counts bearing
that label, from two sources, routinely mean different spans keyed to different
events. The model file has to say, because the data will not.

camdl therefore keeps these concepts distinct at the user boundary and lowers
them into a deliberately simpler internal representation.

<!-- figure: dates-overview -->

```
CALENDAR TIME: what you write
             07-01     07-02     07-03     07-04     07-05
an instant     |---------|---------•---------|---------|     date("2026-07-03")
a period                 [-------------------)               interval(start, stop)

    one translation, when the model is compiled and when data is loaded:
    t = (day(T) − day(origin)) / days per time_unit
    here origin = 2026-07-01 and time_unit = days, so 1 July is 0

INTERNAL TIME: all the engine uses
               0         1         2         3         4
state X_t      •---------•---------•---------•---------•     a value at each t
flow             (0, 1]    (1, 2]    (2, 3]    (3, 4]       summed over (t_prev, t]
effect at t                                  ◆               acts from t onward
```

<!-- /figure -->

The split is deliberate:

> **The internal time system is designed for clean, reliable interval operations
> inside the engine. The external system is designed to express the temporal
> conventions epidemiologists actually bring to a model.**

The two need not share a surface convention. They need one precise, centralized,
tested translation between them.

## The calendar axis

Everything on this page is drawn on one number line. It is worth building it
before anything uses it.

**A date is a day-number.** camdl converts an ISO date to a day number, a
running count of days on the proleptic-Gregorian calendar (the Gregorian
calendar extended backwards), so consecutive dates are always consecutive
integers, with no exceptions for month lengths or leap years. The code calls it
`rata_die`. Unlike the standard rata die, which counts from 1 January of year 1,
camdl counts from an arbitrary fixed day, and only the difference between two
day numbers is ever used:

```
2026-07-01  ->  n
2026-07-02  ->  n + 1
2026-07-03  ->  n + 2
2026-08-01  ->  n + 31
```

**`origin` fixes which day-number is internal zero.** With
`origin = date("2026-07-01")` and `time_unit = 'days`:

<!-- figure: dates-axis -->

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
date()       07-01     07-02     07-03     07-04     07-05
```

<!-- /figure -->

**`date()` returns a boundary, not a day.** Each label above marks a _point_ on
the line — the instant at which that civil day begins. It has zero width.

**A civil day is the interval between two boundaries:**

<!-- figure: dates-civil-days -->

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
civil day    [ 1 Jul   )[ 2 Jul   )[ 3 Jul  )[ 4 Jul  )
             [0,1)      [1,2)      [2,3)     [3,4)
```

<!-- /figure -->

So `1 July` is `[date("2026-07-01"), date("2026-07-02"))`. Two instants make a
period; one instant is not a period.

**The engine accumulates flow between boundaries**, closed on the right:

<!-- figure: dates-flow-buckets -->

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
bucket at 1    (---------]                              = (0,1]
bucket at 2              (---------]                    = (1,2]
bucket at 3                        (---------]          = (2,3]
```

<!-- /figure -->

`(0,1]` and `[0,1)` span the same 24 hours, so the bucket closing at `1` holds
the flow of `1 Jul`. That correspondence is the hinge of everything below.

**An instant scheduled on a boundary belongs to the period that boundary
opens**, because the period it closes is already complete:

<!-- figure: dates-instant-on-boundary -->

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
                                             •  at [date("2026-07-04")] = 3
bucket at 3                        (---------]     already accumulated
bucket at 4                                  (---------]  first one affected
```

<!-- /figure -->

Four terms, then, and they are not interchangeable: a **day-number** (an
integer), an **instant** (a point, what `date()` returns), a **civil day** (an
interval between two instants), and a **bucket** (what the engine accumulates
over, `(previous, this]`).

## Three concepts

**Instant** — a point on the time axis, zero duration. `date("2026-07-05")` is
an Instant.

**Duration** — a span with no absolute location: `7 'days`, `5.2 'days`,
`6 'months`.

**Period** — an interval bounded by two Instants, written half-open,
`[start, stop)`.

`Instant − Instant` is a Duration. `Instant + Duration` is an Instant.
`Duration + Duration` is a Duration. An arbitrary Duration does not silently
become an Instant merely because both carry dimension `[T]`.

### `date()` is an instant, and it is the start of its day

```camdl
date("2026-07-05")
```

denotes the calendar boundary at which 5 July **begins**. It has zero width. It
does not denote the whole day; the civil day is
`[date("2026-07-05"), date("2026-07-06"))`, built from two Instants.

This rule is invariant. A `date()` does not mean one thing in an intervention
and another in an observation.

Every position in the language reads literally under it:

```camdl
origin = date("2026-07-01")            # t = 0 at the start of 1 July
at [date("2026-07-05")]                # a campaign that begins on 5 July
simulate { from = date("2026-07-01")   # simulation covers 1 July onward
           to   = date("2026-07-10") } # [1 July, 10 July) — nine days
```

The alternative — treating the instant labelled 5 July as the _end_ of that day
— would make `simulate { from = origin }` begin at the end of the origin date
and never simulate it, and would force a campaign conducted on 5 July to be
written `date("2026-07-04")`. Neither is defensible in a model file.

### `[start, stop)` outside, `(start, end]` inside

Both conventions are right for their side, and they are chosen for different
reasons.

**Internally**, `(previous boundary, current boundary]` is the natural
convention for a counting process: an increment is `N(t) − N(s)`, the events in
`(s, t]`. It matches the discrete recurrence exactly — `X_t` is the state at
`t`, and `F_t` is the flow that carried the system from `X_{t-1}` to `X_t`.
Changing it would buy nothing and cost that correspondence.

**Externally**, `[start, stop)` is the calendar and programming convention:
`[2026-07-05, 2026-07-12)` is exactly seven days, 5–11 July; adjacent periods
tile as `[a, b)`, `[b, c)`; `stop − start` is the duration; and an event at
`start` belongs to the period it opens.

They are **not the same set** — they differ at their endpoints. For an
absolutely continuous flow, changing endpoint membership does not change the
integral, so either bracket gives the same number. For a jump process or a
discrete backend, endpoint membership _can_ matter — a counting measure is made
of point masses at event times — and the equivalence comes from explicit
ordering rather than from measure:

<!-- figure: dates-boundary-order -->

```
boundary s
    apply interventions and events scheduled at s
    |
    generate the transitions belonging to [s, e)
    |
boundary e
    read and reset the accumulator
    apply anything scheduled at e to the NEXT period
```

<!-- /figure -->

The lowering rule, stated once:

> **A public flow period `[s, e)` is scored in the internal bucket ending at
> `e`.**

And the invariant worth memorising:

> **State is indexed by the boundary where it exists. Flow is indexed externally
> by the period in which it occurred, and internally by the boundary at which it
> materialises.**

### One row carries both readings

A trajectory or observation row holds quantities of both kinds, and they refer
to different things:

| column                                                        | what the row's time means                                                                                                    |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| compartment state (`S`, `I`), `prevalence(of = …, among = …)` | the value **at** that instant                                                                                                |
| trajectory flow (`flow_*`)                                    | the total over `[t_start, t_stop)`, which the row states beside `t` — `t_stop` is `t`, `t_start` the previous row's          |
| observation `incidence(...)`                                  | the total over the period the stream's `covers` assigns to the label — `[D−Δ, D)` under `closing_at`, `[D, D+1)` under `day` |

```tsv dates-example=trajectory-rows model=daily_trajectory cmd="simulate --dates --backend ode" excerpt
t  t_start  t_stop  date        date_start  date_stop   S      I    flow_infection
4  3        4       2026-07-05  2026-07-04  2026-07-05  9914   73    34.3868
5  4        5       2026-07-06  2026-07-05  2026-07-06  9858  120    56.1248
6  5        6       2026-07-07  2026-07-06  2026-07-07  9767  195    91.0209
```

The flow column is what happened _between_ two rows; `S` and `I` are counts _at_
each one. This is not a wrinkle to fix — it is what stocks and flows are, and
the row says so: `flow_infection = 91.0209` on the row dated 7 July is the flow
over `[6 July, 7 July)` — the infections of **6 July** — and `date_start` /
`date_stop` state exactly that, so a reader never has to infer it from the row
above. The initial-condition row, having no interval before it, writes the empty
`[t, t)` and zero flows. If the output schedule omits the run's start there is
no such row, and the first row's `t_start` is the run's start.

### What a data row means is not `date()`'s job

Parsing `2026-07-05` from a file establishes an Instant. It does **not**
establish whether the value is a count over 5 July, a week ending on the 5th, or
a census taken that morning. Those are properties of the observation stream, and
the stream states them: an incidence stream declares what each row covers —
`covers = day(time)`, `starting_on(time, Δ)`, `ending_on(time, Δ)`,
`closing_at(time, Δ)`, or `window_start`/`window_stop` columns — and is refused
without one (E350); an instant stream declares nothing. The forms are specified
in [`camdl-data-spec.md`](camdl-data-spec.md), "What a row covers", and the
language spec §12.1.1.

The declaration exists because no single reading is right. Under
`closing_at(time, 1 'days)` a row dated `D` is scored over `(D−1, D]` — the day
**ending** at `D`, the civil day before its label — which is what a file written
by `simulate --obs` or derived from pomp means. A file labelled the way most
surveillance systems label — by the day the count describes — means `[D, D+1)`,
and under the closing reading it is scored one day early. On the axis above, the
mismatch is one unit:

<!-- figure: dates-closing-at-offset -->

```
                    (0,1]      (1,2]     (2,3]
closing_at scores  "07-02"    "07-03"   "07-04"      row label -> bucket
spans               1 Jul      2 Jul     3 Jul

a day-labelled file means  "07-02" is [1,2) = 2 Jul
closing_at reads           "07-02" as (0,1] = 1 Jul
                                      ^^^^^ one unit apart
```

<!-- /figure -->

**Changing `origin` does not fix it.** `origin` shifts every date coordinate by
the same amount, so the span between two dated rows is invariant under it — a
row labelled `D` under `closing_at` spans the civil day before `D` whatever
origin you pick:

<!-- figure: dates-origin-shift -->

```
origin = 2026-07-01   row "07-02" -> (0,1] -> 1 Jul
origin = 2026-06-30   row "07-02" -> (1,2] -> 1 Jul
origin = 2026-06-25   row "07-02" -> (6,7] -> 1 Jul
```

<!-- /figure -->

**When it is visible.** If every date in the model is a data row, the offset
shifts everything together and changes no fitted number — the epidemic is
described a day early, which matters against anything external but not to the
fit. It becomes visible the moment a _point_ also carries a date, because an
intervention at `date("2026-07-04")` correctly affects `(3,4] = 4 Jul` while the
row a file labels `"07-04"` is scored over `(2,3] = 3 Jul`. The two disagree by
one bucket.

**The fix is a declaration, not a default.** A date column cannot say what its
rows cover — `2026-07-08` is `[8 Jul, 9 Jul)` in a daily file labelled by the
day it describes, `[7 Jul, 8 Jul)` in one labelled by the closing boundary,
`[8 Jul, 15 Jul)` in a week-starting file, and `[2 Jul, 9 Jul)` in a week-ending
one. No rule gets all four right, so the stream states it: `covers = day(time)`
for the day-labelled file, and the row is then scored on the day its label
names. Say which kind of file you have; do not shift the time column, or
`origin`, to make a form fit.

**Which rows train under a holdout cutoff.** A fit config's
`holdout_after = "2026-07-25"` is the instant 25 July begins, and a row trains
if the _end_ of what it measures is at or before it. A `day(time)` row labelled
`2026-07-24` covers 24 July, ends at the start of the 25th and trains; the row
labelled `2026-07-25` covers the 25th and is held out, which the key's name
alone could hide, so the fit prints each stream's first held-out row. A
`closing_at` row ends at its label, so the row labelled `2026-07-25` trains. A
week-starting row labelled `2026-07-21` covers 21–27 July, ends after the cutoff
and is held out, although its label is before it. `last_obs` in a model anchor
(`simulate { to = last_obs + 2 'weeks }`) is likewise the end of the latest
observed training row, not its label: for a `day(time)` stream it is one day
after the last training row's date.

## Anchored vs unanchored models

camdl models come in two modes, distinguished by a single bit: whether
`origin = date(...)` is declared at the top level.

- **Anchored** — `origin` is declared. Internal time `t` maps to a real calendar
  date via `time_unit`; `t = 0` is `origin`. Dates flow at I/O (data columns,
  `--dates` output, `date()` literals in DSL constant positions). Anchored mode
  is what the seed-timing chapter's COVID-WA and Hagelloch fits use.
- **Unanchored** — no `origin`. The internal axis is abstract; `t = 0` has no
  calendar meaning. Time positions are bare numbers in the model's `time_unit`.
  SBC, synthetic-indexed time, textbook SIR, and the dacca cholera SIRS models
  (where `t = month-number from 1 Jan
  1891` is documented as an _informal_
  anchor but not declared as `origin`) are unanchored.

The two modes share most of the language. Three constructs behave differently
across them — see the **anchor-only primitives** and the **constant-day axis
rule** below. The vocabulary mirrors
[`docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md)
§1 (also referred to in some older docs as "calendar-anchored" /
"indexed-time").

## Exact vs Calendar duration kinds

Inside the `[T]` dimension, durations carry a one-bit refinement distinguishing
two kinds — a static, type-level property tracked by the dimensional checker:

- **Exact** — a duration that's a _constant_ number of axis units, invertible
  under translation. Comes from `'days` and `'weeks` unit literals, from
  `Instant − Instant`, and from references to `duration`-kind parameters or
  `[T]`-annotated parameters.
- **Calendar** — a duration spelled with a month or year unit. Translating an
  instant by one is _ambiguous user intent_: does the modeller mean an affine
  span of about 30.44 days per month, or calendar arithmetic that preserves
  year/month/day and clamps at month ends? camdl requires that choice to be
  explicit. (The affine translation is perfectly invertible; the
  non-invertibility lives in the calendar arithmetic — see W327.) Comes only
  from `'months` and `'years` unit literals (and propagates upward through
  arithmetic — the LUB on the lattice `Exact <: Calendar`).

**The principle:** `Calendar` arises _only_ from `'months`/`'years` literals;
everything else is `Exact`. A `Calendar`-classified duration **cannot translate
an `Instant`** in anchored mode — that's a hard error (**E321**) with the hint
pointing at `add_calendar_months` for calendar-exact stepping or an explicit
`'days` literal for an affine offset.

**Date positions.** Some places in a model hold a date rather than a length:
`simulate { from, to }`, a scenario's `simulate { to }`, an `at [...]` entry, a
recurring schedule's `from` and `to`, an output's `trajectories { at }` and an
observation's `emit_schedule = at [...]`, `value_at`'s time, a forcing's
`breakpoints`, a bspline `window` and an interpolated forcing's `times`, the
bounds of an `instant` parameter and any value given to one (a scenario's `set`,
a prior's `mu`, `mean`, `lower` or `upper`; a prior's scale, such as `sigma`, is
a length), and the value compared with model time `t` or with `t ± d`. Under an
origin a bare duration there denotes `origin + D`, so `to = 2 'years` means
`origin + 2 'years`, and any `'months` or `'years` literal there is **E321**.
The hint names the explicit form, computed from your own base and count:

| refused under an origin                 | write instead                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `simulate { to = 2 'years }`            | `to = add_calendar_years(origin, 2)` (or `730 'days`)                                                         |
| `simulate { to = 18 'months }`          | `to = add_calendar_months(origin, 18)` (or `548 'days`)                                                       |
| `simulate { to = 1.5 'years }`          | `to = add_calendar_months(origin, 18)`                                                                        |
| `at [date("2020-03-01") - 1 'months]`   | `at [add_calendar_months(date("2020-03-01"), -1)]`                                                            |
| `breakpoints = [6 'months]`             | `breakpoints = [add_calendar_months(origin, 6)]` (or `183 'days`)                                             |
| `tau : instant in [0 'days, 6 'months]` | `tau : instant in [0 'days, 183 'days]`                                                                       |
| `if t >= 6 'months then … else …`       | `if t >= add_calendar_months(origin, 6) then … else …`                                                        |
| `if t - 1 'months > tau then … else …`  | `if t - 30 'days > tau then … else …` (no calendar form exists: `t`'s date is known only when the model runs) |

The same refusal from a base known only at run time (`t`, an `instant`
parameter, `last_obs` or `first_obs`) says why no calendar form exists there and
offers the span in days. A calendar duration as a cadence (`every = 1 'months`)
is **E322**, whose hint names `date_range(<start>, <end>, calendar_months = 1)`
in an `at` list. A **length** keeps the average month by design: `dt`, a
forcing's `period`, `step` and `lag`, a sinusoidal `phase`, a `via` mean, the
bounds of a `duration` parameter, and every unit literal in an ordinary
expression. Without an origin none of this applies: there is no calendar, and
every month literal is the average.

See the proposal's
[§3](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md#3-the-two-rules)
for the full statement, including the LUB propagation rule and the
classifier-from-leaves invariant that keeps a `'months`-spelled parameter
_bound_ from contaminating uses of the parameter.

## Constant-day axis rule (anchored mode)

`time_unit = 'months` and `time_unit = 'years` are **forbidden when
`origin = date(...)` is declared** — that's **E320**. Average-length months and
years would make the date↔number conversion drift by an accumulating residual
under repeated conversion, which silently mis-aligns rendered dates from the
calendar. Switch to `time_unit =
'days` (or `'weeks`); affine duration literals
(`5 'months` as a length) and rate literals (`0.2 'per_month`) continue to work,
because the expander converts them at compile time. A parameter has no scale of
its own — its value is always in the model's `time_unit` — so a rate parameter's
value must then be supplied per day.

The unanchored case is unaffected: `time_unit = 'months` with no `origin` is
fine (the dacca configuration). The constant-day rule only fires when an
`origin` is present, because that's the only situation where the axis scale
becomes a _conversion factor_.

## Stepping a date by calendar months/years

Two primitive functions, available in DSL constant positions in **anchored mode
only**:

```camdl
add_calendar_months(d, n)   # step Instant d by n calendar months
add_calendar_years(d, n)    # step Instant d by n calendar years
```

`d` is any compile-time-constant Instant — a `date(...)` literal, the reserved
`origin` identifier, or a nested `add_calendar_*` call. `n` is a compile-time
integer (positive or negative).

These are the **only** way to advance a date by calendar months or years in the
language: a month or year literal in a date position is refused (E321), and its
hint names the call to write. They do real `(year, month, day)` arithmetic via
the proleptic-Gregorian calendar and never touch the `30.4369` average-month
factor.

**Month-end clamping** is canonical: if the source day-of-month doesn't exist in
the target month, the result clamps to that month's last day.

```camdl
add_calendar_months(date("2020-01-31"), 1)   # = date("2020-02-29")  (leap)
add_calendar_months(date("2021-01-31"), 1)   # = date("2021-02-28")
add_calendar_years(date("2020-02-29"), 1)    # = date("2021-02-28")
```

Clamping makes calendar-month stepping **non-invertible**:
`(d + 1 month) − 1 month` is in general _not_ `d`. A W327 warning fires on the
literal nested round-trip `add_calendar_months(add_calendar_months(d, n), -n)`
to flag the assumption (single syntactic shape; let-separated equivalents don't
trigger it).

In unanchored mode, a call to `add_calendar_months` or `add_calendar_years` is
**E327** with a hint pointing at adding `origin = date(...)` or — if the user
wanted an affine offset on the abstract axis — using a duration literal like
`30 'days`.

## `origin` as a referenceable identifier

In anchored mode, `origin` is a **reserved read-only identifier** of type
`Instant`, usable wherever a DSL constant-position Instant is accepted:

```camdl
add_calendar_months(origin, 6)               # 6 calendar months after origin
origin + 90 'days                             # affine 90-day offset
at [origin, origin + 30 'days]                # intervention schedule
let landmark = origin + 90 'days
simulate { from = origin
           to = add_calendar_years(origin, 5) }
```

**Instant arithmetic is accepted in every date position.**
`simulate { from,
to }`, a scenario's `to` and an output or emit `at` list fold
it at compile time as an `at [...]` list does: `to = origin + 90 'days` is 90,
and `to = date("2020-03-01") + 7 'days` is 67 from a 2020-01-01 origin.
`to = origin + 5 'years` is **E321**; write `add_calendar_years(origin, 5)`.

**An Instant moves by a duration.** `date(b) - date(a)` is an Exact duration and
`origin + 30 'days` is an Instant. On a definite Instant — one built from
`date(...)`, `origin`, `add_calendar_*`, `instant`-kind parameters and `let`s of
those — six forms are **E360**: `date(a) + date(b)`; a `sum(...)` of instants;
`2 * date(a)` (and `/`, `^`, negation); `7 'days - date(a)`; an offset with no
unit, such as `date(a) + 5` or `date(a) + 5 * 2`, which would be read silently
in `time_unit` (write `5 'days`); and a conditional whose branches are an
instant and a duration, `if c then date(a) else 5 'days`. Expressions that are
not definite Instants are not checked: `k * (t0 - t)` with model time `t`, and
`max(date(a), date(b))`, compile.

`origin` is a compile-time constant, not a runtime value, so it cannot appear
inside rate expressions or any compartment-state context — only in DSL constant
positions.

In unanchored mode, a reference to `origin` is **E327** (same family as
`add_calendar_*` in unanchored mode).

## Generating a list of dates with `date_range`

An intervention that repeats (a campaign every quarter) would otherwise need its
dates written out one by one. `date_range(...)` generates them at compile time.
In any `at` list (an intervention's or event's `at [...]`, an output's
`trajectories { at = [...] }`, an observation's `emit_schedule = at [...]`) it
splices its entries into the list in place. It needs an anchored model: `origin`
as `start` and a calendar cadence are both **E327** without one, and a
`date(...)` start is **E220**, as a bare `date(...)` is. Write an `origin`
before using any form.

A call has a `start`, a way to say where the list stops, and one cadence.

- **`start`** is the first positional argument: a `date(...)` literal, `origin`,
  or an `add_calendar_*` call. It is always the first entry.
- **The stop** is either an `end` instant (the second positional argument) or
  `count = N`. Give exactly one.
- **The cadence** is exactly one of `every = D`, `calendar_months = N` or
  `calendar_years = N`. `every` takes a duration in `'days` or `'weeks`;
  calendar steps take a positive whole number.

```camdl
date_range(date("2020-01-01"), count = 3, every = 7 'days)
    # 2020-01-01  2020-01-08  2020-01-15  2020-01-22

date_range(origin, count = 3, calendar_months = 3)
    # origin = 2020-01-01:  2020-01-01  2020-04-01  2020-07-01  2020-10-01

date_range(date("2020-01-01"), date("2020-10-01"), calendar_months = 3)
    # the same four dates, stopping at `end`
```

**`count` is the number of steps, so `count = N` gives N + 1 entries.** The
`start` is entry 0 and the entries after it are steps 1 to N. The first example
above has `count = 3` and four dates.

**A calendar entry is counted from `start`, not from the entry before it.**
Entry _i_ of a `calendar_months = k` range is
`add_calendar_months(start, i * k)`. This matters when `start` falls on a day
that some months lack. Stepping from the previous entry would clamp to 29
February and then stay on the 29th; counting from `start` returns to the 31st
whenever the month has one (see the clamping rule above):

```camdl
date_range(date("2020-01-31"), count = 4, calendar_months = 1)
    # 2020-01-31  2020-02-29  2020-03-31  2020-04-30  2020-05-31
date_range(date("2020-02-29"), count = 4, calendar_years = 1)
    # 2020-02-29  2021-02-28  2022-02-28  2023-02-28  2024-02-29
```

The start-and-end form gives the same entries as the count form would, cut off
at `end`: it keeps every entry that is on or before `end` (strictly before it
with `inclusive_end = false`, below). An `every` cadence steps a fixed number of
days, so for it the question of where to count from does not arise.

**An `end` that is not on the cadence is not added.** The range stops at the
last entry on or before `end`, and the compiler warns with **W328** that the
list does not reach `end`:

```camdl
date_range(date("2020-01-01"), date("2020-02-01"), every = 2 'weeks)
    # 2020-01-01  2020-01-15  2020-01-29, plus W328: `end = 2020-02-01` does not
    # land on a cadence boundary from `start = 2020-01-01`
```

If you wanted `2020-02-01` in the list, write it next to the call:
`at [date_range(...), date("2020-02-01")]`.

`inclusive_end = false` stops the list at the last boundary strictly before
`end`: an `end` on the cadence is dropped, and W328 is not raised for an `end`
off it. With `end = start` the list is empty.

```camdl
date_range(date("2020-01-01"), date("2020-01-29"), every = 7 'days, inclusive_end = false)
    # 2020-01-01  2020-01-08  2020-01-15  2020-01-22
```

The errors are **E328** for an argument that does not fit the shapes above
(missing `start`, no cadence or two, both `end` and `count`, `every` in
`'months`, an unknown keyword) and **E329** for a cadence that is zero or
negative or for `count < 1`. In an unanchored model, a calendar cadence is
**E327**; use `every` there.

A complete model that uses it:

```camdl
time_unit = 'days
origin    = date("2020-01-01")

compartments { S, V }
parameters { N0 : count }

transitions { leak : S --> V @ 0.0 'per_day * S }

interventions {
  # A booster every quarter for a year: origin plus three quarterly steps.
  booster : transfer(from = S, to = V, fraction = 0.1)
    at [date_range(origin, count = 3, calendar_months = 3)]
}

init { S = N0 }

simulate { from = origin  to = add_calendar_years(origin, 2) }
```

## The one rule

**Calendar structure is erased at the compiler and data-loader boundary.** The
numerical engine operates only on internal numeric time; two places translate,
and everything below them is a plain `f64`:

<!-- figure: dates-one-rule -->

```
ISO date  ──parse_time_cell──▶   internal time (f64)   ──date-renderer──▶  ISO date
 (data in, date() literal)          (the whole engine)                     (output)
```

<!-- /figure -->

An **instant** is a calendar point, written ISO 8601 `YYYY-MM-DD`. `origin` is
the instant mapped to internal time `t = 0`. Then, for any instant `T`:

```
t = (rata_die(T) − rata_die(origin)) / D(time_unit)        # internal time
T = origin + t · D(time_unit) days                          # the inverse
```

`rata_die` is the proleptic-Gregorian day number; `D(time_unit)` is
days-per-unit (`days=1, weeks=7, months=365.2425/12, years=365.2425`). `t` may
be **negative** (a date before the origin — e.g. a seed time before the first
observation) and fractional under non-day units. A **bare number** in any time
position is _already_ internal time; a **date** is converted by the rule above.
The conversion is identical in the OCaml compiler and the Rust runtime (one
shared `rata_die`), so a `date()` literal and the same date in a data file agree
exactly.

## Weekly and coarser cadences

The same rules, and this is where they most often bite.

A weekly row's period is what its stream's `covers` declaration says. Under
`closing_at(time, 7 'days)`, the form `simulate --obs` writes, a row labelled
`2026-07-08` is scored over the seven days ending at its label, 1–7 July — it
contains nothing from the 8th.

Worse, the grid's **phase**, the weekday its rows fall on, is not set by any
calendar rule. A generated grid counts its steps from the start of the
simulation, `simulate { from }`. With `origin = date("2026-07-01")`, a
Wednesday, and `from = origin`, every weekly row falls on a Wednesday:

```tsv dates-example=weekly-wednesday model=weekly_from_origin cmd="simulate --obs --dates"
time  date        cases
7     2026-07-08    88
14    2026-07-15  1356
```

Keep the same `origin` but start the simulation two days earlier, on Monday 29
June (`from = date("2026-06-29")`, which is `t = -2`), and every weekly row
falls on a Monday instead:

```tsv dates-example=weekly-monday model=weekly_from_monday cmd="simulate --obs --dates"
time  date
5     2026-07-06
12    2026-07-13
19    2026-07-20
```

`origin` played no part in that shift. It only names the calendar date of
`t = 0`, and it moves nothing in time: changing it relabels every instant with a
different date, but leaves each row exactly where it was relative to the start
of the simulation. The grid counts from `from` because a simulation's first
observation counts from its own start: the first weekly row closes the first
seven simulated days, whatever date `t = 0` happens to carry.

An ISO week always ends on a Sunday and an MMWR week on a Saturday.

The remedy differs by where the boundaries come from, and conflating the two is
a trap:

- **Generated boundaries** — a numeric cadence (`emit_schedule = every 7 'days`)
  or `--emit-every` — count from `simulate { from }`, so `from` sets the phase:
  a seven-day grid falls on the weekday of `from`. To end each generated week on
  a Saturday, start the simulation on a Sunday; each row is then dated the
  Sunday that follows its week. Choosing `origin` does not set the phase.
- **Explicitly dated observations** — a file carrying `2026-07-08`, `2026-07-15`
  — already fix their own boundaries. `origin` changes only their numeric
  coordinates, never the periods between them, so it cannot repair a phase or
  label mismatch in a dated file.

And even with the phase right, a boundary date cannot carry "week ending Sunday"
as a _label_; that is what `covers = ending_on(time, 7 'days)` states on the
stream. Do not match a camdl weekly row to an ISO or MMWR week by its date
alone.

Named week constructors (`iso_week(2026, 28)`, `mmwr_week(2026, 28)`) would fix
a week's phase from the standard rather than from the start of the simulation.
They do not exist; `ending_on` states what a week-ending row spans, but its
phase still comes from the labels in the file.

## Interventions and events are instants

An intervention scheduled `at [date("2026-07-05")]` fires **at** the instant at
which 5 July begins. It therefore affects transmission during 5 July and every
period after it. There is no translation: a campaign conducted on 5 July is
written `date("2026-07-05")`.

Two consequences follow from the boundary ordering, and both are visible in
output:

**An intervention dated `D` changes the state columns on the row dated `D`, and
the flow columns on the following row.** Run the same model with and without the
intervention: `flow_infection` on the row dated 5 July is bit-identical between
them, while `S` on that row already shows the change. This holds on all three
backends.

**The flow it first affects — 5 July's — appears on the row labelled
`2026-07-06`**, because a flow row is labelled by the boundary at which its
bucket closes. That is the labelling gap described above, not an intervention
problem: the intervention is where you put it, and the _output row naming_ is
what shifts.

## Writing dates in a model

```camdl
time_unit = 'days
origin    = date("2020-02-24")        # t = 0 is this calendar day

parameters {
  tau   : instant                      # an absolute time → renders as a date
  delay : duration                     # a relative span → renders as a span
  beta  : rate
}

simulate {
  from = date("2020-01-21")            # = t -34 (34 days before the origin)
  to   = date("2020-06-27")
}
```

- **`date("YYYY-MM-DD")`** is usable anywhere a constant is expected (`origin`,
  `simulate { from/to }`, scheduled event/intervention times). It compiles to
  the internal-time number. A `date()` without a top-level `origin` is an error
  (**E220**).
- **`instant` and `duration` parameter kinds** carry dimension `[T]`, so the
  dimensional checker now covers time:
  - `rate + instant` (or `rate + duration`) is a dimension mismatch (**E302**) —
    a real modeling bug caught at compile time.
  - `rate * duration` is dimensionless (a valid per-event probability factor).
  - An `instant` renders as a **date** against `origin`; a `duration` renders as
    a **span** (no origin needed). Both are origin-relative where it matters: if
    you move `origin`, declare your time anchors as `instant`/`duration` so they
    move with it (a bare `real` anchor will _not_).
  - An `instant` may take a **negative** lower bound — e.g. a seed time that
    falls _before_ the origin: `tau : instant in [-40, 120]`. (Negative bounds
    are preserved for `instant`/`real` kinds; `rate`/`positive`/`count` remain
    non-negative by their nature.)

## Loading dated data

The `--data` time column accepts **either** numeric internal time **or** ISO
dates — detected automatically per column:

| column looks like           | treated as                           | needs `origin`? |
| --------------------------- | ------------------------------------ | --------------- |
| `0, 1, 2, …` (any `f64`)    | internal time, used directly         | no              |
| `2020-03-15, …` (ISO dates) | converted via `origin` + `time_unit` | yes             |
| mixed numeric + date        | hard error                           | —               |

```bash
# dated column, auto-detected and converted via the model's origin
camdl fit run fit.toml                 # model declares origin = date("2020-02-24")
camdl pfilter model.camdl --data cases_dated.tsv ...
```

- **Numeric data is unchanged** — indexed day-numbers `0,1,2` (or fractional
  times) take the existing path and need no `origin`. Adding `origin` to a model
  never reinterprets a numeric column.
- **`--time-format`** forces the interpretation (e.g. to reject a packed integer
  like `20200315` that would otherwise parse as a number). The accepted values
  are `auto` (the default), `numeric`, `date`, and `internal-days` — an alias
  for `numeric` that reads as an assertion rather than a coercion. Note that
  `--help` currently lists only the first three.
- **`--time-col NAME`** selects the time column by name.
- A dated column with no `origin` in the model → clear error.
- **An observation period is never moved to fit the step grid.** Converted times
  need not land on the integrator's grid `t_start + k·dt`, measured from the
  run's start `simulate.from`. What happens then is a property of the method,
  not a setting. `pfilter`, `if2` and `pmmh` step exactly to every period's
  start and end, with a shortened step where one falls between grid times.
  `pgas` steps only on its grid, so it refuses a run in which any period starts
  or ends off it, naming the stream, the row, its period and the two grid times
  either side; on-grid means within `1e-9` steps of a grid time. The ODE methods
  read their trajectory at its output times and refuse a period boundary that
  has no output time. Interventions and events fire on the run's own grid,
  `k = ⌊(t − t_start)/dt + ½⌋` on every chain-binomial path (language spec
  §13.2). The `[config] obs_alignment` key that used to name a choice here is
  removed; a fit config that carries it is refused.

## Getting dates back out

- **`camdl simulate --dates`** adds a calendar `date` column (the inverse map)
  alongside the canonical numeric `t` in trajectory and observation output
  (single-file and `--obs-dir`), and in a trajectory `date_start`/`date_stop`
  alongside `t_start`/`t_stop`, the period each row's flows cover. Without
  `--dates`, output is byte-identical to before. Requires `origin`. A whole-day
  timepoint renders as a bare `YYYY-MM-DD`; a **sub-day** timepoint (a
  fractional snapshot step under a sub-day `dt`, the hot-epidemic regime)
  renders the floor date with the fractional day appended as a `+<frac>d` delta
  — e.g. `t = 0.25` under `'days` from a `2020-02-28` origin is
  `2020-02-28+0.25d`. This keeps the column one-to-one with the timepoint:
  distinct sub-day rows get distinct labels rather than silently coalescing onto
  the same date (gh#108). The suffix is a fractional-day delta, deliberately
  _not_ the `YYYY-MM-DDTHH:MM` datetime form (datetimes are out of scope — see
  "Not supported (yet)" below); a consumer grouping on the date column can split
  on `+` to recover the calendar day. The fraction is written to six decimals
  when that reads back to `t` exactly (quarter and half days), and otherwise in
  full, as the shortest decimal that does: an hourly grid's `t = 1/24` is
  `2020-02-28+0.041666666666666664d`, not `+0.041667d`, which would read back as
  a different time.
- **Every stored artifact carries a `calendar` block**
  `{"days_per_unit", "origin", "time_unit"}`: a simulation leaf's `run.json`,
  `fit.meta.json`, and the predictive, observed, quantities and trajectory
  sidecar manifests. A consumer dates a numeric `t` as
  `origin + t · days_per_unit` days, without the model source; `origin` is
  `null` for a model without one.
- **Prequential output and the worst-scored table**
  (`camdl pfilter
  --save-prequential`, `fit run`'s pfilter method) add a
  `date` column after `t` when the model has an origin, written the same way.
- **`camdl fit summary`** renders `instant`-kind estimands as dates when the
  model has an `origin` (e.g. `tau = 23.0  (2020-02-13)`); `duration` estimands
  render as spans. Numeric `t` stays the canonical, diff-stable value. A point
  estimate is a single value (never a column joined on), so its date annotation
  rounds to the nearest whole day for readability.

**Every date camdl writes reads back to the time it was written from.** The
`--data` loader accepts the `+<frac>d` suffix, so a `--dates` column loads as a
dated time column and each cell lands on the exact model time it was rendered
from (gh#839). Under `'days` this holds for every time. Under `'weeks` a date is
read as days / 7, and a time that is not such a quotient of any floating-point
number of days has no exact days offset: it reads back to the adjacent
floating-point number. A malformed suffix (`+1.5d`, `+0d`) is refused as camdl's
own sub-day form, by name.

## International / multi-source data

A bare ISO date is a **civil-calendar label**, not an absolute instant — its day
number is computed straight from `(Y, M, D)`, with no timezone. So:

- A trailing zone designator (`Z`, `+06:00`, `-03:00`, `+05:45`) is a **hard
  error** (E223), naming the offset and the cell, in both a `date()` literal and
  a `--data` time column. camdl does not model time zones, so an offset is
  information it cannot represent; accepting the cell and deleting the offset
  would silently change what the data says. Strip the offset upstream and supply
  the civil date you mean — `2020-03-15+06:00` becomes `2020-03-15`. Pooling
  surveillance from many countries then aligns by civil date onto one axis.
- This is _timezone-independent_, not "assume UTC": two modelers in any two
  zones who write `2020-03-15` get the same internal time. (Whether two
  locations' outbreaks _started_ at the same time is a modeling question —
  per-location seed times `τ_i` — not a calendar one.)
- camdl trusts the civil date as written; if an upstream export mislabeled a
  late-night local event to the wrong civil day, that is an upstream concern no
  date policy can recover.

## Migrating an unanchored monthly model to anchored

> Reference: this section describes the surface introduced by the typed-time
> proposal
> ([`docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md)).
> Phases 1 (rules) and 2 (calendar primitives) have shipped:
> `time_unit = 'months` with `origin` declared is **E320**; the hint below
> points at the migration.

If you have an existing model with `time_unit = 'months` (or `'years`) and no
`origin` — the dacca SIRS shape, for example — and you decide you want calendar
dates, the migration is a few lines, with one trap worth flagging up front.

**The trap.** When you switch `time_unit` from `'months` to `'days`, every
_bare-numeric_ time position in your model silently changes meaning to the new
axis. `simulate { to = 600 }` was 600 months; after the switch it's 600 days.
`dt = 0.05` was 0.05 months (≈ 1.5 days); after the switch it's 0.05 days (≈ 72
minutes). Observation data columns and intervention schedules shift the same
way.

**What survives unchanged.** Any _typed_ position — any value that carries its
own unit literal — converts correctly. `1 'months` as a length keeps working
(it's an affine span of ≈ 30.44 days regardless of axis), and so does a rate
literal such as `0.2 'per_month`. A month or year literal in a date position is
refused under the origin (E321), so `to = 600 'months` is rewritten, as option 1
below says.

**What does not.** A parameter has no scale of its own: its value is always read
in the model's `time_unit`. After the switch, the value you supply for a rate
parameter `beta` is per day, so a per-month value must be divided by ≈ 30.44 —
the compiler cannot do it for you, because the value arrives at fit or run time.

**Before** (unanchored monthly):

```camdl
time_unit = 'months
# no origin

parameters {
  beta  : rate        # per month
  gamma : rate        # per month
  ...
}

let latent = 1 'months

simulate {
  from = 0           # month 0
  to   = 600         # 600 months
  dt   = 0.05        # 0.05 months ≈ 1.5 days
}

interventions {
  sia : transfer(...) at [180, 540]   # months 180 and 540
}
```

**After** (anchored, daily axis, calendar I/O):

```camdl
time_unit = 'days
origin    = date("1891-01-01")

parameters {
  beta  : rate        # now per day: supply the per-month value / 30.44
  gamma : rate        # likewise
  ...
}

let latent = 1 'months        # unchanged — affine 30.44 days

simulate {
  from = date("1891-01-01")   # was 0; now explicit instant
  to   = date("1940-12-01")   # was 600; now explicit instant
  dt   = 1.5                  # was 0.05; same physical step in new axis
                              # (or write `dt = 0.05 'months` to stay unit-aware)
}

interventions {
  sia : transfer(...) at [date("1906-01-15"), date("1936-06-30")]
                              # were 180 and 540; now explicit dates
}
```

**Three options for each bare-numeric site.** At each `simulate { from/to/dt }`,
`at [...]` schedule, and `--data` time column, you have a choice:

1. **Write the calendar step**: `to = add_calendar_months(origin, 600)`, the
   date 600 calendar months after the origin (1941-01-01 from 1891-01-01).
   Smallest diff from the original that keeps its meaning. `to = 600 'months` is
   refused under an origin (E321), and its hint names this form. For an
   intervention list, `at [add_calendar_months(origin, 180)]`.
2. **Use a date literal**: `to = date("1940-12-01")`. Explicit calendar instant.
   Best for new code and for anything a reader needs to verify against a
   calendar.
3. **Manually convert and stay bare-numeric**: `to = 18260`. Legal but the next
   reader can't see what `18260` means without the conversion context, and
   `--time-format numeric` may be needed on the data file's time column if a
   packed integer would otherwise parse as a date. Acceptable for terse
   regression-test fixtures; avoid in models meant to be read.

For new models, prefer dates. For migrations, option 1 (the calendar step) is
the smallest diff.

## Why `'months` is fine in some places and forbidden in others

A natural confusion reading the rules above: `5 'months` works in a table value
or a `duration` parameter's bound, `0.2 'per_month` works in any rate
expression, but `date + 6 'months` and `simulate { to = 6 'months }` are hard
errors. That looks contradictory. It isn't.

The principle: **a calendar month is unambiguous as a _length_ and unambiguous
as a _rate denominator_. It is ambiguous only as a _step from a date_.** A
length is "≈ 30.44 days," period — fine as a table value, parameter bound, or
multiplicand in a rate expression. A rate denominator is "per ≈ 30.44 days" —
fine, the expander converts to per-axis-unit. Neither length nor rate ever
translates an instant, so neither produces a calendar-vs-affine question. Only
`date + 6 'months` does — does it mean "the 24th of the month six months later"
(calendar-exact, clamped) or "182 days later" (affine)? — and that's where the
rule fires.

Stated in one line: **Rule 1 isn't "calendar units are dangerous," it's "you
can't add a calendar duration to a date."** A date position adds its value to
the origin, so a month there is a step from a date too. Rates aren't durations;
lengths and table values don't get added to dates. They all stay legal.

## Seasonal forcing: the same days every year, or this year's dates

A forcing that switches with the calendar, such as school terms raising contact
rates, can mean two different things, and camdl has a form for each. Choose by
what the model claims about the calendar, not by convenience.

**The same days every year.** The classic models of childhood infections
idealise the school year as fixed days of the year, repeated. He, Ionides and
King's measles model ([He et al. 2010](https://doi.org/10.1098/rsif.2009.0151),
_J R Soc Interface_ 7:271–283), which the repository reproduces in
`tests/external/cases/he2010_*`, writes the UK terms as days 7–100, 115–199,
252–300 and 308–356 of each year. Real term dates are set locally and move from
year to year, as Easter does, so this is a deliberate approximation, and it is
the published convention. Write it as a `periodic` forcing with `on` ranges:

```camdl
forcing {
  school : periodic 'ratio {
    period = 1 'years
    step   = 1 'days
    on     = [7 'days : 100 'days, 115 'days : 199 'days,
              252 'days : 300 'days, 308 'days : 356 'days]
  }
}
```

**These actual dates.** When the model should follow this year's real term
dates, or a holiday whose date moves (Easter, Lunar New Year), the forcing is
not periodic. It is a step function on dates, and a `piecewise` forcing takes
dates as its breakpoints directly:

```camdl
origin = date("2020-01-01")

forcing {
  school : piecewise 'ratio {
    breakpoints = [date("2020-01-06"), date("2020-04-03"),
                   date("2020-04-20"), date("2020-07-22")]
    values      = [0, 1, 0, 1, 0]
  }
}
```

The breakpoints compile to days 5, 93, 110 and 203: 6 January, 3 April, 20 April
and 22 July 2020, exactly.

For a fit over several years, the term dates belong in a file, one row per
change, read by an `interpolated` forcing. `method = constant` holds each row's
value until the next row's date, which is the same step function:

```tsv
date	school
2020-01-01	0
2020-01-06	1
2020-04-03	0
2020-04-20	1
2020-07-22	0
2020-09-02	1
2020-12-18	0
2020-12-31	0
```

```camdl
origin = date("2020-01-01")

forcing {
  school : interpolated 'ratio {
    data      = "school_terms.tsv"
    time_col  = "date"
    value_col = "school"
    method    = constant        # hold each value until the next row
  }
}
```

Simulated, this switches on days 5, 110 and 245 (6 January, 20 April, 2
September) and off on days 93, 203 and 352 (3 April, 22 July, 18 December), on
the same steps as the `piecewise` form above. The file must cover the whole run:
a forcing read from a file is an error past its last row, unless the run is
given `--allow-forcing-extrapolation`. That is why the file ends with a row on
the run's last day.

A smooth seasonal curve (`sinusoidal`) and a measured covariate such as rainfall
(an `interpolated` forcing read from a data file) are the other two common
seasonal forms. Neither switches on fixed days, so the choice below does not
arise for them.

**Which year length.** In a `periodic` forcing the `on` days are counted from
the start of each period, so the period's length decides whether those days stay
on the calendar. `period = 1 'years` is 365.2425 days, the average Gregorian
year, and stays within about a day of 1 January indefinitely.
`period = 365 'days` loses a quarter-day each year, so its terms creep later
through the calendar:

| `period`              | start of period, days after 1 January, after 10 years | after 100 years | range over 400 years |
| --------------------- | ----------------------------------------------------- | --------------- | -------------------- |
| `1 'years` (365.2425) | +0.6                                                  | −0.3            | −0.9 to +1.3         |
| `365.25 'days`        | +0.5                                                  | −1.0            | −3.0 to +0.8         |
| `365 'days`           | +3                                                    | +24             | grows without bound  |

The He et al. model uses `365.25 'days`, the Julian year, which is within three
days of the calendar over four centuries. For a model fitted to a few decades,
`1 'years` and `365.25 'days` are indistinguishable; `365 'days` is not, and
should be written only when a 365-day cycle is what the model means.

## Not supported (yet)

- **Times of day** (`2020-03-15T13:30`) in dates/data, and **sub-day
  `time_unit`s** (`'hours`/`'minutes`/`'seconds`). A datetime form is rejected
  with a clear error. Fast/hot epidemics don't need these — they need a smaller
  integrator step `dt` (in `'days`), which is independent of the time unit.

## Known implementation deviations

The temporal model above is normative. These are places the current release
behaves differently, each tracked:

| deviation                                                                                                                                            | issue  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| The CAS leaf's `traj.tsv` carries no `date` column; its `run.json` carries the `calendar` block that maps `t` to dates.                              | gh#838 |
| Observation-alignment errors on a dated file report bare model time and suggest numeric fixes.                                                       | gh#842 |
| `--time-format internal-days` is accepted as an alias for `numeric`, but the name is misleading when `time_unit` is not days, and `--help` omits it. | —      |

## Reference

- **Conversion / parsing:** `rust/crates/ir/src/caltime.rs` (Rust runtime),
  `ocaml/lib/compiler/expander.ml` `days_of_date`/`parse_date_to_float` (compile
  time). The two are pinned to agree.
- **IR fields:** `Model.origin` (the ISO string) and `Model.origin_rata_die`
  (the compiler-derived integer day number the runtime reads). IR schema ≥ 0.6.
- **`time_unit` vs `dt` vs cadence:** `time_unit` is the axis unit; `dt` is the
  integrator step (set by the dynamics, not the data); observation cadence is a
  property of the data. See `docs/camdl-run-spec.md`.
- **Where `dt` lives:** `dt` is a model knob — write it in the model's
  `simulate { dt = … }` block, where it sits next to `from`/`to` and is
  unit-aware (`dt = 0.05 'months`). Models are genuinely sensitive to the
  discretization step, so it belongs in the model, not buried in a CLI flag. The
  `--dt` flag is the _override_: it wins over the model's `dt` for one run,
  which is what you want for a sensitivity sweep or Richardson- extrapolation
  diagnostic. With neither set, the step defaults to 1 (`time_unit`). Omit `dt`
  from the model only when you intend the run to pick it.
- **Design rationale and test plan:**
  `docs/dev/proposals/archive/post-alpha/2026-05-22-calendar-time.md`.
