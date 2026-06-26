# camdl DSL cheatsheet

Orientation doc — what the DSL supports, in one place, with pointers to the
normative sources. **Not the spec.** When this disagrees with
[`docs/camdl-language-spec.md`](camdl-language-spec.md), the spec wins.

This file exists because the DSL surface is large enough that an agent or new
contributor working from memory or from a single proposal often misses features
the language already provides, then reinvents them poorly. Read this first when
proposing DSL changes.

> **Status conventions.** This cheatsheet reflects the language as of typed-time
> Phases 1 and 2 (the
> [`2026-05-22-typed-time-and-dsl-ergonomics.md`](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md)
> proposal): the Exact/Calendar split, `add_calendar_months/_years`,
> `date_range`, the constant-day axis rule, and the anchored-mode diagnostics
> (E320–E323, E327–E329, W324–W328) have all shipped.

## Time and units

**Time unit declaration** sets the model's internal axis:

```camdl
time_unit = 'days        # also: 'weeks, 'months, 'years
```

All durations and rates normalise to this unit at compile time. The runtime only
sees `f64` values in this unit — `time_unit` itself isn't read in the dynamics
path (only at I/O boundaries for date conversion).

**Numeric literals carry units** — prefix-apostrophe syntax:

```camdl
5 'days       # duration, dimension [T]
14 'days
2 'weeks
0.5 'years
0.1 'per_day  # rate, dimension [T⁻¹]
0.02 'per_year
1.0 'ratio    # dimensionless multiplier (sinusoidal forcings, etc.)
100 'count    # raw population count
```

Available unit literals (defined in `ocaml/lib/compiler/lexer.mll`):

| Class      | Literals                                           | Dimension |
| ---------- | -------------------------------------------------- | --------- |
| Duration   | `'days`, `'weeks`, `'months`, `'years`             | `[T]`     |
| Rate       | `'per_day`, `'per_week`, `'per_month`, `'per_year` | `[T⁻¹]`   |
| Count      | `'count`                                           | `[P]`     |
| Multiplier | `'ratio`                                           | `[1]`     |

**Conversion table** (`days_per_unit` in `rust/crates/ir/src/caltime.rs` and the
mirror in `expander.ml`, proleptic-Gregorian throughout):

```
1 'day   = 1                 day
1 'week  = 7                 days
1 'month = 365.2425 / 12     days  ≈ 30.4369
1 'year  = 365.2425          days
```

**Mixed-unit arithmetic** works through the dimensional checker:

```camdl
5 'days + 3 'days        # = 8 'days
1 / (14 'days)           # = rate (1/time)
0.1 'per_day * 5 'days   # = 0.5 (dimensionless)
5 'days + 0.1 'per_day   # ERROR E302: cannot add time and rate
```

## Dimensional information — three tiers

For full background see `docs/camdl-language-spec.md` §2.3.

| Tier                  | Syntax                                                                    | Carries                        | Use when                                        |
| --------------------- | ------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------- |
| 1. Kind keyword       | `rate`, `probability`, `count`, `positive`, `real`, `instant`, `duration` | dimension (inferred from kind) | parameter declarations — the 99% case           |
| 2. Bracket annotation | `[T]`, `[T^-1]`, `[P]`, `[P/T]`, `[1]`                                    | dimension only                 | kind is under-determined (`real`/`positive`)    |
| 3. Unit literal       | `'days`, `'per_day`, `'count`, `'ratio`, …                                | dimension _and_ scale          | concrete numeric values with a real-world scale |

Tiers are complementary, not redundant — tier 3 carries _scale_, the others
don't. A parameter from a prior or a `--fixed-file` / `--params` TOML lives at
tier 1 or 2 (scale is implicit in the model's `time_unit`); a literal like
`5 'years` lives at tier 3.

## Parameter kinds

```camdl
parameters {
  beta     : rate                          # [T⁻¹], log transform for inference
  rho      : probability                   # [1] bounded [0,1], logit transform
  R0       : positive in [1.0, 20.0]       # >0, log transform, with bounds
  N0       : count                         # [P], integer ≥ 0
  alpha    : real                          # unconstrained
  tau      : instant in [date("2020-01-01"), date("2020-04-30")]
                                            # [T] absolute time, renders as date
  delta    : duration in [1 'days, 60 'days]
                                            # [T] span, renders as span
}
```

`instant` and `duration` are the time-typed kinds; see
[`docs/dates.md`](dates.md) for full date semantics.

## Dates and calendar arithmetic

**`date("YYYY-MM-DD")`** in DSL constant positions converts to internal time via
`origin`:

```camdl
origin = date("2020-02-24")

simulate {
  from = date("2020-01-21")    # = t − 34 days from origin (in time_unit)
  to   = date("2020-06-27")
}
```

`date(...)` is accepted in **every absolute-time position**, not just
`simulate.from/to`: the `at = [...]` schedule of an `interventions {}` or
`events {}` entry takes dates too, each resolving to its internal offset via
`origin`. Prefer it to a bare number anywhere the time is a calendar instant — a
bare numeric there under a date `origin` warns **W324** (`simulate`) / **W325**
(`at`-schedules); a `date(...)` is the legible form that silences it.

```camdl
interventions {
  vacc : transfer(from = S, to = V, fraction = 0.1)
         at [date("2020-03-01"), date("2020-06-01")]   # no W325 — dates are explicit
}
```

Without a top-level `origin`, `date(...)` is **E220**.

**Anchored vs unanchored** models — see [`docs/dates.md`](dates.md) for the full
reference:

- **Anchored**: declares `origin`. Internal axis maps to real calendar dates.
  Must use `time_unit = 'days` or `'weeks` (constant-day rule;
  `time_unit = 'months/'years` with `origin` is **E320**).
- **Unanchored**: no `origin`. Internal axis is abstract; bare numbers. Any
  `time_unit` is fine including `'months`/`'years`. SBC, synthetic, textbook SIR
  live here; so do the dacca SIRS models.

**Anchor-only primitives** — these require `origin` to be declared:

| Construct                                     | Anchor-only? | If used unanchored                |
| --------------------------------------------- | ------------ | --------------------------------- |
| `date("YYYY-MM-DD")`                          | yes          | E220                              |
| `origin` (identifier)                         | yes          | E327                              |
| `add_calendar_months(d, n)`                   | yes          | E327                              |
| `add_calendar_years(d, n)`                    | yes          | E327                              |
| `date_range(..., calendar_months/_years = N)` | yes          | E327                              |
| `instant`-kind param (rendering)              | yes          | works as `[T]`; no date rendering |
| `5 'months`, `5 'years`                       | **no**       | legal — affine span               |
| `0.087 'per_month`                            | **no**       | legal — affine rate               |
| `time_unit = 'months`/`'years`                | **no**       | legal (E320 only fires anchored)  |

The bottom rows are _calendar-named affine constructs_, not anchor-only — the
dacca SIRS configuration (unanchored, monthly axis, per-month rates, month-span
durations) is all of those, and it remains fully legal.

**Exact vs Calendar duration kinds.** Duration unit literals carry a one-bit
refinement on `[T]`: `'days`/`'weeks` are **Exact**; `'months`/`'years` are
**Calendar**. The refinement propagates by LUB through arithmetic. In anchored
mode, `Instant ± Calendar`-duration is **E321** with a hint at `add_calendar_*`
(calendar-exact) or a `'days` literal (affine offset). Parameter references with
`[T]` dimension are always Exact — a `'months`-spelled _bound_ on a
`duration`-kind parameter is a length, not a step-from-a-date, and never
contaminates uses.

**Calendar arithmetic primitives** (anchored mode, DSL constant positions only):

```camdl
add_calendar_months(d, n)    # Instant × Int → Instant
add_calendar_years(d, n)     # Instant × Int → Instant
date_range(start, end, calendar_months = 3)
                              # calendar-aligned breakpoint list
date_range(start, end, every = 7 'days)
                              # affine cadence
```

`d` is any compile-time-constant Instant (`date(...)`, `origin`, or a nested
`add_calendar_*` call). Month-end clamping is canonical and **non-invertible** —
`(d + 1 month) − 1 month ≠ d` in general; a W327 warns on the literal nested
round-trip shape.

## Periodic forcings — already calendar-friendly

```camdl
forcing {
  school : periodic 'ratio {
    period = 365.25 'days
    step   = 1 'days
    on     = [7:100, 115:199, 252:300, 308:356]
  }

  reporting_dow : periodic 'ratio {
    period = 7 'days
    values = [1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]
  }
}
```

Every forcing declaration carries a **tier-3 unit literal** between the kind
keyword and the block (`sinusoidal 'ratio`, `interpolated 'count`, etc.). This
is required per GH #8.

## Generated quantities — `quantities {}`

Report derived summaries of a run — the non-scored twin of an observation (no
likelihood, never re-keys the fit). Each entry is `name [idx]? = body`:

```
quantities {
  prevalence      = I / N                        # series (one value per output time)
  attack_rate     = final((N0 - S) / N0)         # scalar
  peak_prevalence = max(I / N)                    # value reduction
  time_to_peak    = time_of_max(I)                # → a time (a date in an anchored model)
  takeoff         = first_above(I_total, i_thr)   # threshold crossing
  fadeout         = last_above(I_total, 0)
  outbreak_dur    = fadeout - takeoff             # reduction arithmetic over scalars
  person_days     = integral(I)                   # ∫ over time (dim P·T)
  peak_time[p in patch] = time_of_max(I[p])      # stratified
}
```

Reductions (valid **only** inside `quantities {}`): `final`, `max`, `min`,
`mean`, `count_above|below(x, thr)`, `time_of_max|min`,
`first|last_above|below(x, thr)`, `integral`. A quantity with no reduction is a
**series**; one with a reduction is a **scalar**. `max(a, b)` / `min(a, b)` stay
the binary operators everywhere — only a **unary** `max(x)` in a quantity is the
peak reduction. Reduction arithmetic (`a - b`) combines earlier **scalar**
quantities. (`total`/`sum` and an `observations.<stream>` source are not in v1.)

Output: one `quantities/<name>.tsv` per quantity (banded `q05…q95` over draws) +
a `quantities.json` manifest — from `fit predict` (in the fit segment) and
`simulate --quantities-out <dir>` (a point `value` for a single run). A `time`
reduction that never crosses is **right-censored** (reported via
`n_value`/`n_censored`, not faked). Quantities are non-identity: adding a
`quantities {}` block never changes a model's `run_id`.

## Common diagnostics

The compiler issues E-codes with source locations and fix-hints.

| Code | Class    | Typical trigger                                                                            |
| ---- | -------- | ------------------------------------------------------------------------------------------ |
| E100 | naming   | parameter name shadows reserved (`t`, etc.)                                                |
| E203 | indexing | named-index references wrong dimension                                                     |
| E220 | date     | `date(...)` without `origin` declared                                                      |
| E300 | dim      | transition rate not P·T⁻¹                                                                  |
| E301 | dim      | non-dimensionless argument to `exp`/`log`                                                  |
| E302 | dim      | addition/subtraction of mismatched dimensions                                              |
| E303 | dim      | parameter used with inconsistent dimensions                                                |
| E304 | dim      | `sqrt` of odd-exponent dimension                                                           |
| E305 | dim      | balance expression must have dimension P                                                   |
| E306 | dim      | ODE derivative must have dimension P·T⁻¹                                                   |
| E308 | dim      | overdispersion σ² must be dimensionless                                                    |
| E320 | time     | `time_unit = 'months/'years` with `origin` declared                                        |
| E321 | time     | `Instant ± Calendar`-duration (`date(...) + 6 'months`)                                    |
| E322 | time     | calendar cadence in anchored recurring schedule (`every = 1 'months`)                      |
| E323 | time     | bare-numeric `on=[...]` in anchored periodic forcing                                       |
| E327 | time     | `add_calendar_*` / `origin` / calendar-cadence `date_range` in unanchored model            |
| E328 | time     | argument-shape error in `add_calendar_*` / `date_range`                                    |
| E329 | time     | zero/negative cadence or `count < 1` in `date_range`                                       |
| W301 | forcing  | periodic range not aligned to step size                                                    |
| W324 | time     | bare-numeric `simulate.from/to/dt` in anchored mode                                        |
| W325 | time     | bare-numeric `at [k, ...]` schedule in anchored mode                                       |
| W326 | time     | numeric `--data` time column under `origin` (use `--time-format internal-days` to silence) |
| W327 | time     | literal nested `add_calendar_*` round-trip (non-invertible)                                |
| W328 | time     | `date_range` `end` doesn't land on a cadence boundary                                      |

## Where things live

- **Lexer (tokens, unit literals):** `ocaml/lib/compiler/lexer.mll`
- **Parser (grammar):** `ocaml/lib/compiler/parser.mly`
- **AST:** `ocaml/lib/compiler/ast.ml`
- **Expander (stratification + IR emission):** `ocaml/lib/compiler/expander.ml`
- **Dimensional checker:** `ocaml/lib/compiler/dimcheck.ml`
- **IR types (Rust):** `rust/crates/ir/src/`
- **Calendar conversion (Rust):** `rust/crates/ir/src/caltime.rs`
- **IR schema (the OCaml↔Rust contract):** `ir/schema.json`
- **Language spec (authoritative):** `docs/camdl-language-spec.md`
- **User-feature tour:** `docs/user-features.md`
- **Calendar reference:** `docs/dates.md`

## Pitfalls that have actually bitten us

These are real failure modes that have produced incident reports. Read them
before assuming the language doesn't do something — it usually does.

- **Reinventing existing surface.** The DSL already has `5 'days`,
  `0.087 'per_month`, range syntax `[7:100]`, the dimensional checker, and
  `instant`/`duration` parameter kinds. Before adding a new duration / rate /
  cadence / unit construct, grep this cheatsheet and the language spec.
- **Cross-language constants need a single source of truth.** Anything that has
  to agree across OCaml and Rust either lives in one place and is read by both,
  or lives in two places with a test pinning them to match — never two
  hand-maintained copies. The shared proleptic-Gregorian `rata_die` algorithm
  (mirrored in `rust/crates/ir/src/caltime.rs` and
  `ocaml/lib/compiler/expander.ml`, pinned by `days_per_unit` returning
  identical Gregorian-average constants on both sides) is the model to follow.
- **Calendar months aren't durations.** "+1 month" depends on its input _and_ on
  the year: `date("2021-01-31") + 1 month = date("2021-02-28")` but
  `date("2020-01-31") + 1 month = date("2020-02-29")` (leap). It's an instant
  operation, not a translation. The language enforces this through the
  Exact/Calendar refinement on `[T]` (E321), with
  `add_calendar_months`/`add_calendar_years` as the only correct
  calendar-stepping path. See [`docs/dates.md`](dates.md).
- **Hard errors over warnings.** When a construct is silently ambiguous, the
  compiler hard-errors with a fix-hint rather than warning. This is CLAUDE.md
  policy and the typed-time proposal's acceptance criterion (now shipped).

## Recent and incoming changes

- **`quantities {}`** (2026-06-25) — generated quantities: named, non-scored
  reductions of a run (peak, time-to-peak, attack rate, integral, …) reported as
  `quantities/<name>.tsv` from `fit predict` / `simulate --quantities-out`. New
  reserved word: `quantities`; the reduction names (`final`, `time_of_max`,
  `first_above`, `integral`, …) are valid only inside the block (using one in a
  rate is **E290**). See the section above.
- **`reactive_interventions {}`** (gh#204) — state/observation-triggered
  policies:
  `name : when sum_observed(stream, window = D) >= thr { action = transfer(..);
  after = ..; once = ..; cooldown = .. }`.
  The `when` predicate reads observed data via `observed(stream)` /
  `sum_observed(stream, window = ..)` (never latent state — using `observed()`
  in a rate is **E278**). New reserved words: `reactive_interventions`, `when`,
  `action`. Forward **chain-binomial** runs the agenda (firings recorded in
  `reactive_log.tsv`); inference and Gillespie/ODE stop an active policy with a
  `REACTIVE_INTERVENTIONS` capability error. See spec §13.9.

For things this cheatsheet may lag on, check:

- `docs/dev/proposals/` for in-flight design.
- `docs/dev/incidents/` for known bugs and their resolutions.
- `git log -- ocaml/lib/compiler/lexer.mll ocaml/lib/compiler/parser.mly` for
  actual grammar changes.
