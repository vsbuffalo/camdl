# Dates and calendar time

camdl's internal time is a **continuous number** in units of the model's
`time_unit`, measured from an `origin`. Real epidemic data is dated. This page
is the single reference for how calendar dates relate to internal time across
the DSL, the data loader, and output — so you can point camdl at a dated file
and read results back as dates, without pre-converting anything by hand.

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
- **Calendar** — a duration whose magnitude is derived from the affine
  month/year constant (≈ 30.4369 days / month, ≈ 365.2425 days / year) and so
  _isn't_ invertible when applied to a date. Comes only from `'months` and
  `'years` unit literals (and propagates upward through arithmetic — the LUB on
  the lattice `Exact <: Calendar`).

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
add_calendar_months(origin, 6)            # 6 calendar months after origin
origin + 90 'days                          # affine 90-day offset
simulate { from = origin to = origin + 5 'years }
at [origin, origin + 30 'days]             # intervention schedule
let landmark = origin + 90 'days
```

`origin` is a compile-time constant, not a runtime value, so it cannot appear
inside rate expressions or any compartment-state context — only in DSL constant
positions.

In unanchored mode, a reference to `origin` is **E327** (same family as
`add_calendar_*` in unanchored mode).

## The one rule

Dates live **only at the I/O boundary**. Two places translate; everything below
them is a plain `f64`:

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
- **`--time-format numeric|date`** forces the interpretation (e.g. to reject a
  packed integer like `20200315` that would otherwise parse as a number); the
  default is `auto`.
- **`--time-col NAME`** selects the time column by name.
- A dated column with no `origin` in the model → clear error.
- **Off-grid times warn, they don't fail.** Converted times need not land on the
  integrator step `dt`; the solver snaps within `dt` (Gillespie is exact). The
  one hard error is two distinct observations mapping to the _same_ step (obs
  spacing `< dt`) — a real data/`dt` mismatch. Under `'months`/`'years`,
  calendar data never lands on integers (average-length months); this warns, and
  is fine on a continuous axis — use `'days` if you want integer-aligned monthly
  points.

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

## International / multi-source data

A bare ISO date is a **civil-calendar label**, not an absolute instant — its day
number is computed straight from `(Y, M, D)`, with no timezone. So:

- A trailing zone designator (`Z`, `+06:00`, `-03:00`, `+05:45`) is a **hard
  error**, naming the offset and the cell, in both a `date()` literal and a
  `--data` time column. camdl does not model time zones, so an offset is
  information it cannot represent; accepting the cell and deleting the offset
  would silently change what the data says. Strip the offset upstream and supply
  the civil date you mean — `2020-03-15+06:00` becomes `2020-03-15`.
- This is _timezone-independent_, not "assume UTC": two modelers in any two
  zones who write `2020-03-15` get the same internal time, and pooling
  surveillance from many countries aligns by civil date onto one axis. (Whether
  two locations' outbreaks _started_ at the same time is a modeling question —
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
   `--time-format internal-days` is required on the data file's numeric time
   column. Acceptable for terse regression-test fixtures; avoid in models meant
   to be read.

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
