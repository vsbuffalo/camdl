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

## The calendar axis

Everything on this page is drawn on one number line. It is worth building it
before anything uses it.

**A date is a day-number.** camdl converts an ISO date to its
proleptic-Gregorian day number — its _rata die_ — so consecutive dates are
always consecutive integers, with no exceptions for month lengths or leap years:

```
2026-07-01  ->  739433
2026-07-02  ->  739434
2026-07-03  ->  739435
```

**`origin` fixes which day-number is internal zero.** With
`origin = date("2026-07-01")` and `time_unit = 'days`:

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
date()       07-01     07-02     07-03     07-04     07-05
```

**`date()` returns a boundary, not a day.** Each label above marks a _point_ on
the line — the instant at which that civil day begins. It has zero width.

**A civil day is the interval between two boundaries:**

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
civil day    [ 1 Jul   )[ 2 Jul   )[ 3 Jul  )[ 4 Jul  )
             [0,1)      [1,2)      [2,3)     [3,4)
```

So `1 July` is `[date("2026-07-01"), date("2026-07-02"))`. Two instants make a
period; one instant is not a period.

**The engine accumulates flow between boundaries**, closed on the right:

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
bucket at 1    (---------]                              = (0,1]
bucket at 2              (---------]                    = (1,2]
bucket at 3                        (---------]          = (2,3]
```

`(0,1]` and `[0,1)` span the same 24 hours, so the bucket closing at `1` holds
the flow of `1 Jul`. That correspondence is the hinge of everything below.

**An instant scheduled on a boundary belongs to the period that boundary
opens**, because the period it closes is already complete:

```
internal t     0         1         2         3         4
               |---------|---------|---------|---------|
                                             •  at [date("2026-07-04")] = 3
bucket at 3                        (---------]     already accumulated
bucket at 4                                  (---------]  first one affected
```

Four terms, then, and they are not interchangeable: a **day-number** (an
integer), an **instant** (a point, what `date()` returns), a **civil day** (an
interval between two instants), and a **bucket** (what the engine accumulates
over, `(previous, this]`).

## The system in one picture

```
USER-FACING CALENDAR MODEL          (normative; see "Known deviations")
--------------------------
date("2026-07-05")            an Instant; zero width; the boundary at which
                              5 July begins

                     |                                  |
              date("2026-07-05")                 date("2026-07-06")
                     v                                  v
                     [------------ 5 July -------------)

interval(start, stop)         [start, stop)         ← the period primitive
                              PLANNED — no period constructor ships today;
                              see "Known implementation deviations"

                              |
                              |  one centralized lowering
                              v

CAMDL ENGINE
------------
state  X_t                    the value AT boundary t
flow bucket ending at t       accumulated over (t_prev, t]
effect scheduled at t         changes state at t, and dynamics after t
internal time                 continuous f64 in the model's time_unit
```

The split is deliberate:

> **The internal time system is designed for clean, reliable interval operations
> inside the engine. The external system is designed to express the temporal
> conventions epidemiologists actually bring to a model.**

The two need not share a surface convention. They need one precise, centralized,
tested translation between them.

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

| column                                          | what the row's time means                              |
| ----------------------------------------------- | ------------------------------------------------------ |
| compartment state (`S`, `I`), `prevalence(...)` | the value **at** that instant                          |
| flow (`flow_*`), `incidence(...)`               | the total **over the interval ending at** that instant |

```
t  date        S      I    flow_infection
4  2026-07-05  9931   49    24.2009
5  2026-07-06  9895   73    35.8401
6  2026-07-07  9842  108    52.8828
```

The flow column is what happened _between_ two rows; `S` and `I` are counts _at_
each one. This is not a wrinkle to fix — it is what stocks and flows are.

**Read the flow row carefully.** A flow row is labelled by the boundary at which
its bucket _closes_, so `flow_infection = 52.8828` on the row dated 7 July is
the flow over `[6 July, 7 July)` — the infections of **6 July**, not the 7th.
This is the most counterintuitive consequence of the convention, and it stands
until flow output carries its own `period_start` / `period_stop`.

### What a data row means is not `date()`'s job

Parsing `2026-07-05` from a file establishes an Instant. It does **not**
establish whether the value is a count over 5 July, a week ending on the 5th, or
a census taken that morning. Those are properties of the observation stream.

**Today camdl has no vocabulary for that**, so a dated observation inherits the
Instant directly: a row at time `t` is scored over `(t_prev, t]`, which means a
daily row dated `D` is scored against the day **ending** at `D` — the civil day
before its label. A file labelled the way most surveillance systems label — by
the day the count describes — is therefore read one day early.

That is a real gap, not a convention. Giving a stream a way to state its period
— so a row can say it covers `[D, D+1)` and be scored in the bucket ending at
`D+1` — is the subject of the observation-window and input-format work (gh#833).
On the axis above, the mismatch is one unit:

```
                 (0,1]      (1,2]     (2,3]
camdl scores    "07-02"    "07-03"   "07-04"      row label -> bucket
spans            1 Jul      2 Jul     3 Jul

a file means    "07-02" is [1,2) = 2 Jul
camdl reads     "07-02" as (0,1] = 1 Jul
                           ^^^^^ one unit apart
```

**Changing `origin` does not fix it.** `origin` shifts every date coordinate by
the same amount, so the span between two dated rows is invariant under it — a
row labelled `D` spans the civil day before `D` whatever origin you pick:

```
origin = 2026-07-01   row "07-02" -> (0,1] -> 1 Jul
origin = 2026-06-30   row "07-02" -> (1,2] -> 1 Jul
origin = 2026-06-25   row "07-02" -> (6,7] -> 1 Jul
```

**When it is visible.** If every date in the model is a data row, the offset
shifts everything together and changes no fitted number — the epidemic is
described a day early, which matters against anything external but not to the
fit. It becomes visible the moment a _point_ also carries a date, because an
intervention at `date("2026-07-04")` correctly affects `(3,4] = 4 Jul` while the
row a file labels `"07-04"` is scored over `(2,3] = 3 Jul`. The two disagree by
one bucket.

**The fix is a declaration, not a default.** A date column cannot say what its
rows cover — `2026-07-08` is `[7 Jul, 8 Jul)` in a daily file, `[8 Jul, 15 Jul)`
in a week-starting file, and `[2 Jul, 9 Jul)` in a week-ending one. No rule gets
all three right, so the stream has to state it. That is gh#833, and it is a
runtime change as well as a format one: an observation is a single scalar time
today (`Observation { time, value }`), with one accumulator reset at that time,
so a window with a gap has nothing to lower onto.

Until then: if the model has no dated instants, nothing is wrong beyond the
labelling. If it does, adjust the _instant_ by one day rather than rewriting the
data file — one visible line in the model instead of a silent transformation of
the input.

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
`'days` literal for an affine offset. See the proposal's
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
'days` (or `'weeks`); per-month rate parameters
(`beta : rate
'per_month`) and affine duration literals (`5 'months` as a
length) continue to work, because the expander converts them at compile time.

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
language. They do real `(year, month, day)` arithmetic via the
proleptic-Gregorian calendar and never touch the `30.4369` average-month factor.

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

**`simulate { to = }` does not accept arithmetic**, unlike `let` and `at [...]`:
`to = origin + 90 'days` is **E401** ("expected a constant expression"), and
`to = origin + 5 'years` is additionally **E321** (a Calendar duration cannot
translate an Instant). Use `add_calendar_years` / `add_calendar_months`, or a
bare number. The asymmetry is tracked in gh#844.

`origin` is a compile-time constant, not a runtime value, so it cannot appear
inside rate expressions or any compartment-state context — only in DSL constant
positions.

In unanchored mode, a reference to `origin` is **E327** (same family as
`add_calendar_*` in unanchored mode).

## The one rule

**Calendar structure is erased at the compiler and data-loader boundary.** The
numerical engine operates only on internal numeric time; two places translate,
and everything below them is a plain `f64`:

```
ISO date  ──parse_time_cell──▶   internal time (f64)   ──date-renderer──▶  ISO date
 (data in, date() literal)          (the whole engine)                     (output)
```

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

A weekly row is scored over the seven days ending at its instant. Under the
loader's current behaviour that means a row labelled `2026-07-08` is scored
against 1–7 July — it contains nothing from the 8th.

Worse, the grid's **phase** is set by `origin`, not by a weekday. With
`origin = date("2026-07-01")`, a Wednesday, every weekly row falls on a
Wednesday:

```
t   date        cases
7   2026-07-08  7012
14  2026-07-15  6998
```

An ISO week always ends on a Sunday and an MMWR week on a Saturday.

The remedy differs by where the boundaries come from, and conflating the two is
a trap:

- **Generated boundaries** — a numeric cadence, or `--emit-every` — take their
  weekday from `origin`, so choosing `origin` does set the phase.
- **Explicitly dated observations** — a file carrying `2026-07-08`, `2026-07-15`
  — already fix their own boundaries. `origin` changes only their numeric
  coordinates, never the periods between them, so it cannot repair a phase or
  label mismatch in a dated file.

And even with the phase right, a boundary date cannot carry "week ending Sunday"
as a _label_; that is what `period_ending_on` and `mmwr_week` are for. Do not
match a camdl weekly row to an ISO or MMWR week by its date alone.

Named week constructors (`iso_week(2026, 28)`, `mmwr_week(2026, 28)`) would fix
a week's phase from the standard rather than from `origin`. They do not exist;
they are part of the input-format work (gh#833).

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
- **Off-grid times are governed by `obs_alignment`, not by a silent snap.**
  Converted times need not land on the integrator step `dt`. What happens then
  is a declared, per-algorithm choice (`fit.toml [backend]`, alongside `dt`):
  `exact` steps to the observation time with a shortened final substep, `snap`
  rounds onto the `dt` grid. `pfilter`, `if2` and plain `pmmh` default to
  `exact` and reject `snap`; `pgas` uses a uniform grid and therefore `snap`,
  and rejects `exact` naming the algorithms that support it; correlated `pmmh`
  requires on-grid observations and errors off-grid. Nothing falls back
  silently. Where snapping applies the rule is nearest — `round(t / dt)` — with
  on-grid meaning within `1e-9`, and interventions and events map to steps
  through the same `round(t / dt)`, so a snapped observation and an intervention
  at the same instant land on the same step. The one hard error common to all is
  two distinct observations mapping to the _same_ step (obs spacing `< dt`) — a
  real data/`dt` mismatch.

## Getting dates back out

- **`camdl simulate --dates`** adds a calendar `date` column (the inverse map)
  alongside the canonical numeric `t` in trajectory and observation output
  (single-file and `--obs-dir`). Without `--dates`, output is byte-identical to
  before. Requires `origin`. A whole-day timepoint renders as a bare
  `YYYY-MM-DD`; a **sub-day** timepoint (a fractional snapshot step under a
  sub-day `dt`, the hot-epidemic regime) renders the floor date with the
  fractional day appended as a `+<frac>d` delta — e.g. `t = 0.25` under `'days`
  from a `2020-02-28` origin is `2020-02-28+0.25d`. This keeps the column
  one-to-one with the timepoint: distinct sub-day rows get distinct labels
  rather than silently coalescing onto the same date (gh#108). The suffix is a
  fractional-day delta, deliberately _not_ the `YYYY-MM-DDTHH:MM` datetime form
  (datetimes are out of scope — see "Not supported (yet)" below); a consumer
  grouping on the date column can split on `+` to recover the calendar day.
- **`camdl fit summary`** renders `instant`-kind estimands as dates when the
  model has an `origin` (e.g. `tau = 23.0  (2020-02-13)`); `duration` estimands
  render as spans. Numeric `t` stays the canonical, diff-stable value. A point
  estimate is a single value (never a column joined on), so its date annotation
  rounds to the nearest whole day for readability.

**camdl cannot read this column back.** The `+0.25d` suffix is emitted by
`--dates` and rejected by the `--data` loader, with an error that misattributes
the cause to time-of-day support (gh#839). Under a sub-day `dt`, treat `--dates`
output as for reading, not as input; the numeric `t` column round-trips
correctly.

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

**What survives unchanged.** Any _typed_ position — anything that carries its
own unit literal — converts correctly. `beta : positive 'per_month` keeps
working (the expander converts to per-axis-unit). `1 'months` as a duration
value keeps working (it's an affine span of ≈ 30.44 days regardless of axis).

**Before** (unanchored monthly):

```camdl
time_unit = 'months
# no origin

parameters {
  beta  : positive 'per_month
  gamma : positive 'per_month
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
  beta  : positive 'per_month # unchanged — expander converts to per-day
  gamma : positive 'per_month
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

1. **Annotate with a unit literal**: `to = 600 'months`. The expander converts
   to the new axis (≈ 18260 days). Smallest diff from the original. Reads as
   "600 calendar-average months from origin" — affine; for calendar-exact
   end-of-50-years use a date.
2. **Use a date literal**: `to = date("1940-12-01")`. Explicit calendar instant.
   Best for new code and for anything a reader needs to verify against a
   calendar.
3. **Manually convert and stay bare-numeric**: `to = 18260`. Legal but the next
   reader can't see what `18260` means without the conversion context, and
   `--time-format numeric` may be needed on the data file's time column if a
   packed integer would otherwise parse as a date. Acceptable for terse
   regression-test fixtures; avoid in models meant to be read.

For new models, prefer dates and unit literals. For migrations, option 1 (the
unit-literal annotation) is the smallest diff.

## Why `'months` is fine in some places and forbidden in others

A natural confusion reading the rules above: `5 'months` works in a table value
or a parameter bound, `beta : positive 'per_month` works everywhere, but
`date + 6 'months` is a hard error. That looks contradictory. It isn't.

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
can't add a calendar duration to a date."** Rates aren't durations; bounds and
table values don't get added to dates. They all stay legal.

## Not supported (yet)

- **Times of day** (`2020-03-15T13:30`) in dates/data, and **sub-day
  `time_unit`s** (`'hours`/`'minutes`/`'seconds`). A datetime form is rejected
  with a clear error. Fast/hot epidemics don't need these — they need a smaller
  integrator step `dt` (in `'days`), which is independent of the time unit.

## Known implementation deviations

The temporal model above is normative. These are places the current release
behaves differently, each tracked:

| deviation                                                                                                                                                                                                                                        | issue  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| No period constructors. An observation cannot state the interval it covers, so a dated row is scored over the day _ending_ at its label.                                                                                                         | gh#833 |
| `--dates` emits `YYYY-MM-DD+0.25d` for sub-day times, which camdl's own `--data` loader rejects, with an error naming the wrong cause.                                                                                                           | gh#839 |
| A duration literal in an instant position (`to = 600 'months`) is accepted and read as an origin-relative coordinate, contradicting the Instant/Duration rule stated above. Retained for migration compatibility; do not write it in new models. | gh#847 |
| The CAS leaf carries no `date` column or calendar metadata, so a stored run cannot be mapped back to dates without its source model.                                                                                                             | gh#838 |
| Observation-alignment errors on a dated file report bare model time and suggest numeric fixes.                                                                                                                                                   | gh#842 |
| `--time-format internal-days` is accepted as an alias for `numeric`, but the name is misleading when `time_unit` is not days, and `--help` omits it.                                                                                             | —      |

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
