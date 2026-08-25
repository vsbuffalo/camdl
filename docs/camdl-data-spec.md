# camdl Data Model

How external data enters a `.camdl` model, how CSV columns map to DSL objects,
and how dimensions are derived from data.

---

## Long-format data

All external data in camdl is **long-format**: one row per observation, columns
are either **index columns** (keys into dimensions) or **value columns** (the
numbers the model uses).

Each row is one fact. Each column has one role. This is the tidy data principle
applied to compartmental models.

**Example: population by district**

```tsv
patch	population
kano_dala	485000
borno_maiduguri	345000
borno_gwoza	78000
```

One index column (`patch`), one value column (`population`). Each row says: "the
population of this patch is this number."

**Example: spatial adjacency**

```tsv
src	dst	weight
kano_dala	borno_maiduguri	0.10
kano_dala	kano_fagge	0.15
borno_maiduguri	kano_dala	0.10
```

Two index columns (`src`, `dst`), one value column (`weight`). Pairs not listed
have weight 0 (sparse — most patches aren't adjacent).

**Example: SIA campaign records**

```tsv
patch	day	coverage
kano_dala	180	0.82
kano_dala	365	0.75
kano_dala	540	0.71
borno_maiduguri	182	0.68
borno_maiduguri	370	0.55
borno_gwoza	190	0.45
```

Two index columns (`patch`, `day`), one value column (`coverage`). Kano has 3
rows (3 campaigns), Gwoza has 1. Patches with no campaigns have no rows —
coverage is 0 by default (no event).

**The pattern:** every table is `(index₁, index₂, ...) → value`. The type
signature in the DSL declares which columns are indices (by naming dimensions)
and which are values (everything remaining).

---

## The type signature is the mapping

A table declaration maps CSV structure to DSL objects:

```camdl
name : dim₁ × dim₂ × ... = read("file.tsv")
```

The dimensions map **positionally** to index columns in the CSV. The remaining
column(s) are the value(s). A CSV with `n_dims + 1` columns has exactly one
value column (the last).

```camdl
tables {
  pop : patch = read("data/pop.tsv")
}
```

The type says 1 dimension (`patch`). The CSV has 2 columns. Positional mapping:

```
Column 1 (patch)      → patch dimension index
Column 2 (population) → table value

kano_dala   485000    → pop[kano_dala] = 485000
borno_gwoza  78000    → pop[borno_gwoza] = 78000
```

For a 2D table:

```camdl
tables {
  adj : patch × patch = read("data/adj.tsv", default = 0.0)
}
```

The type says 2 dimensions. The CSV has 3 columns:

```camdl
Column 1 (src)    → first patch dimension
Column 2 (dst)    → second patch dimension
Column 3 (weight) → table value

kano_dala   borno_maiduguri   0.10  → adj[kano_dala, borno_maiduguri] = 0.10
```

Column names in the CSV are for human readability. The compiler uses positional
mapping from the type signature only. (It does require a header row and skips
it.)

---

## `dimensions {}`: where dimension levels come from

A dimension is a named set of **levels** — the valid values for that index.
`age` has levels `{under5, over5}`. `patch` has levels
`{kano_dala, borno_maiduguri, ...}`.

Levels come from the `dimensions {}` block, which is declared at the top level
of the file. Two forms are supported:

**Inline** — small structural dimensions:

```camdl
dimensions {
  age = [under5, over5]
}
```

**Data-derived** — large or data-driven dimensions read from a file column:

```camdl
dimensions {
  patch = read("data/pop.tsv", column = "patch")
}
```

The `read(file, column = "col")` form reads the named column, collects unique
values in first-occurrence order, and those become the levels of the dimension.
All tables referencing `patch` validate against these derived levels.

After declaring dimensions, `stratify(by = X)` applies them to compartments:

```camdl
dimensions {
  age   = [under5, over5]
  patch = read("data/pop.tsv", column = "patch")
}

stratify(by = age)
stratify(by = patch)

tables {
  pop : patch = read("data/pop.tsv")
  adj : patch × patch = read("data/adj.tsv", default = 0.0)
}
```

### Rules

**1. Each dimension is defined exactly once.**

```camdl
# OK: patch defined once, referenced by two tables
dimensions { patch = read("data/pop.tsv", column = "patch") }
tables {
  pop : patch = read("data/pop.tsv")
  adj : patch × patch = read("data/adj.tsv", default = 0.0)
}

# ERROR: "dimension 'patch' declared twice in dimensions block"
dimensions {
  patch = read("data/pop.tsv", column = "patch")
  patch = read("data/pop2.tsv", column = "patch")
}
```

**2. Validation is exhaustive and strict.**

When a table references a known dimension, the compiler checks the CSV column
values against the dimension's known levels:

- CSV value not in dimension → **error**:
  `"'kano_dala_north' in
  column 1 of adj.tsv is not a valid 'patch' level"`
- Dimension level not in CSV → depends on `default`:
  - With `default = 0.0` → missing cells filled with default (sparse)
  - Without `default` → **error**:
    `"patch level 'borno_abadam' has
    no entry in adj.tsv"` (dense — every
    level must appear)

**3. Dimension levels are exhaustively validated.**

If `stratify(by = patch)` appears, the levels from the `dimensions {}` block
become the compartment expansion set. The compiler ensures every level produces
a valid identifier (no spaces, no special characters beyond `_`).

**4. Bare dimension names must already be declared.**

Unknown bare name in a type signature → **error**:
`"unknown dimension
'ptach' — did you mean 'patch'?"` (with Levenshtein
suggestion). This catches typos that implicit derivation would silently accept.

**5. Inline `[...]` and `read(...)` are mutually exclusive for the same dim.**

```camdl
# OK: inline levels
dimensions { age = [under5, over5] }

# OK: data-derived levels
dimensions { patch = read("data/pop.tsv", column = "patch") }

# ERROR: "dimension 'age' declared with both inline levels and read()"
dimensions { age = [under5, over5] }
# (cannot also have age = read(...) in the same block)
```

---

## Two ways to specify dimensions

| Source           | Declaration                                                           | Stratifies compartments? | Levels from |
| ---------------- | --------------------------------------------------------------------- | ------------------------ | ----------- |
| Inline           | `dimensions { age = [under5, over5] }` + `stratify(by = age)`         | yes                      | inline list |
| Data, stratified | `dimensions { patch = read(f, column="c") }` + `stratify(by = patch)` | yes                      | CSV column  |
| Data, index-only | `dimensions { sia_time = read(f, column="c") }` (no `stratify`)       | no                       | CSV column  |

All three produce the same thing internally: a named set of levels. They differ
in: (a) whether the dimension stratifies compartments, and (b) where the levels
come from.

`stratify(by = X)` declares the role: "this dimension applies to compartments."
The `dimensions {}` block provides the membership. Role and membership are
orthogonal — you can declare a dimension from data without stratifying (e.g.,
`sia_time = read(...)` with no `stratify`), or stratify with inline levels.

---

## Multiple value columns

When a CSV has more columns than `n_dims + 1`, there are multiple value columns.
List multiple table names on the left of `:`:

```camdl
tables {
  pop, sex_ratio : patch = read("data/demographics.tsv")
}
```

```tsv
patch	population	sex_ratio
kano_dala	485000	0.51
borno_maiduguri	345000	0.49
```

One index column, two value columns. Creates two tables with the same index:
`pop[kano_dala] = 485000`, `sex_ratio[kano_dala] = 0.51`.

Value columns map positionally to the names on the left of `:`. The compiler
validates: number of names must match number of non-index columns.

**All values are `f64`.** Table values are always numeric. Dimension levels are
strings (or numeric-coercible strings). These are different things: levels are
the index space, values are what the table stores.

---

## Sparsity and defaults

`default = 0.0` fills index combinations with no CSV row:

```camdl
dimensions {
  sia_time = read("data/sia.tsv", column = "day")
}

tables {
  adj     : patch × patch          = read("data/adj.tsv", default = 0.0)
  sia_cov : patch × sia_time       = read("data/sia.tsv", default = 0.0)
}
```

| Table                    | What default means                      |
| ------------------------ | --------------------------------------- |
| `adj`, default = 0.0     | Non-adjacent patches have zero coupling |
| `sia_cov`, default = 0.0 | No campaign event → zero coverage       |

Without `default`, the table is **dense**: every index combination MUST have a
row. Missing rows → compile error. Dense is the right default for things like
population — every patch must have a value.

**Future: `default = missing`.** For observation data with genuinely absent
measurements. The inference engine conditions on present values and skips
missing ones. Same table structure, different default semantics. Not needed for
forward simulation.

---

## SIA campaigns: the complete pattern

This is where the data model earns its keep. Campaign data is ragged (different
patches get different numbers of rounds), indexed by time (which is continuous
in the model), and has per-event values (coverage).

The data:

```tsv
patch	day	coverage
kano_dala	180	0.82
kano_dala	365	0.75
kano_dala	540	0.71
borno_maiduguri	182	0.68
borno_maiduguri	370	0.55
borno_gwoza	190	0.45
```

The model:

```camdl
dimensions {
  sia_time = read("data/sia.tsv", column = "day")
}

tables {
  sia_cov : patch × sia_time = read("data/sia.tsv", default = 0.0)
}

interventions {
  sia[p in patch, t in sia_time] : transfer(
    fraction = vacc_eff * sia_cov[p, t],
    from = S[under5, p], to = V[under5, p],
    at = t
  ) where sia_cov[p, t] > 0
}
```

**What the compiler does:**

1. Reads `sia.tsv`. Column "day" → `sia_time` levels (unique values:
   `[180, 182, 190, 365, 370, 540]`). Column 1 → `patch` (known, validated).
   Column 3 → value.

2. Builds `sia_cov` as sparse `238 × 6` table. Most cells are 0.0.

3. Expands `sia[p in patch, t in sia_time]`: Cartesian product of 238 × 6 =
   1,428 candidate interventions.

4. Applies `where sia_cov[p, t] > 0`: evaluates at compile time (table values
   are known). ~1,422 zero-coverage cells eliminated. 6 survive.

5. Each surviving intervention:
   - `sia_kano_dala_180`: at = 180.0, fraction = vacc_eff × 0.82
   - `sia_kano_dala_365`: at = 365.0, fraction = vacc_eff × 0.75
   - etc.

6. IR contains 6 flat interventions. No dimension metadata, no table references.
   Concrete names, times, expressions.

**Numeric dimension levels.** `sia_time` has levels `[180, 182, 190, ...]`.
These are numeric strings from the CSV. They coerce to `f64` in expression
contexts like `at = t`. The compiler tracks which dimensions have all-numeric
levels and allows this coercion. Non-numeric levels (`at = child`) would be a
type error.

**Two campaign data sources.** If routine and outbreak campaigns are separate:

```camdl
dimensions {
  routine_time  = read("data/routine.tsv",  column = "day")
  outbreak_time = read("data/outbreak.tsv", column = "day")
}

tables {
  routine_cov  : patch × routine_time  = read("data/routine.tsv",  default = 0.0)
  outbreak_cov : patch × outbreak_time = read("data/outbreak.tsv", default = 0.0)
}

interventions {
  sia_routine[p in patch, t in routine_time] : transfer(
    fraction = vacc_eff * routine_cov[p, t],
    from = S[under5, p], to = V[under5, p],
    at = t
  ) where routine_cov[p, t] > 0

  sia_outbreak[p in patch, t in outbreak_time] : transfer(
    fraction = vacc_eff * outbreak_cov[p, t],
    from = S[under5, p], to = V[under5, p],
    at = t
  ) where outbreak_cov[p, t] > 0
}

scenarios {
  routine_only  { enable = [sia_routine] }
  full_response { enable = [sia_routine, sia_outbreak] }
}
```

Two dimensions, two tables, two intervention families. No union, no
intersection, no ambiguity. The scenario system composes them.

---

## Time: `time_unit` and the interpolation boundary

**What `time_unit` does today:** normalizes unit literals at compile time.
`5 'years` with `time_unit = 'days` becomes `5 × 365.25 = 1826.25`. All rate
expressions and schedule times are in the declared time unit after compilation.
The runtime operates in a single unit — no conversions at simulation time.

**The convention:** all _numeric_ time values in CSVs are in the model's
`time_unit`. Campaign day 180 means day 180. Climate week 26 means... day 26? No
— the user converts. If `time_unit = 'days` and the CSV has weekly data, the
time column should be `7, 14, 21, ...` (days), not `1, 2, 3, ...` (weeks). The
DSL doesn't do unit conversion on numeric CSV columns.

**ISO-date time columns, however, _are_ converted.** If the time column holds
ISO dates (`2020-03-15, …`) instead of numbers, camdl auto-detects this and
converts each to internal time via the model's `origin` + `time_unit` — so dated
surveillance data loads directly, no hand-anchoring. Numeric columns stay
untouched (the convention above). See [`docs/dates.md`](dates.md) for the full
rules (auto-detection, `--time-format`, timezones, output rendering).

**Where this matters for `at = t`:** when `sia_time` has levels
`[180, 365, 540]`, `at = t` produces fire times `180.0, 365.0, 540.0` in the
model's time unit (days). The runtime schedules interventions at exactly these
times. No interpolation needed — discrete events at exact times.

**Where interpolation IS needed: continuous time-varying covariates.**

Climate data, seasonal contact rates, rainfall — these vary continuously but are
measured at discrete intervals. The table stores
`patch × climate_week → temperature`, but the Gillespie algorithm asks "what's
the temperature at t = 14.73 days?"

This is NOT a table lookup problem — it's a time function problem. Tables are
compile-time lookups. Time-varying covariates need runtime interpolation.

**Per-stratum data-driven forcing: an indexed interpolated time function.**

A forcing indexed by a dimension reads each stratum's own series from a
**long-format** file — one row per `(stratum, time)` — and interpolates it at
runtime. The file has a stratum-id column (`key_col`), a time column, and a
value column:

```
# data/temperature.tsv
patch     week   temp
kano      0      27.1
kano      28     29.8
katsina   0      26.4
katsina   28     28.9
```

```camdl
forcing {
  temperature[p in patch] : interpolated 'ratio {
    data      = "data/temperature.tsv"
    key_col   = "patch"      # the column identifying the stratum
    time_col  = "week"       # numeric time in the model's time_unit, or an ISO date
    value_col = "temp"
    method    = linear   # bare enum: linear | constant | spline
  }
}

transitions {
  infection[p in patch] : S[p] --> I[p]
    @ beta * (1 + alpha * temperature[p]) * S[p] * I[p] / N[p]
}
```

This expands to one interpolated forcing per patch — `temperature_kano`,
`temperature_katsina`, … — each carrying only that patch's rows. At runtime
`temperature[kano]` interpolates the Kano series at the current time `t`. The
stratum a row belongs to is its `key_col` value, matched to the forcing's index
level; the binder name (`p`) and the column name (`patch`) need not coincide,
and a level with no matching rows is a compile error (**E227**), not a silent
zero-forcing.

This is a natural extension, not a new concept: time functions already do
runtime interpolation, and indexing one by a dimension is the same
`[p in patch]` pattern used everywhere else.

**Two data sources.** The form above reads its own long-format file
(`data = …`). Alternatively the series can come from a table already in the
central `tables {}` block — the same road population, contact matrices, and
adjacency travel — with `table =` and `time_dim =`:

```camdl
tables {
  temp_data : patch × climate_week = read("data/temperature.tsv")
}

forcing {
  temperature[p in patch] : interpolated 'ratio {
    table    = temp_data     # the source table
    time_dim = climate_week  # which dimension is the time axis (its levels are the knot times)
    method   = linear
  }
}
```

The forcing's own index (`[p in patch]`) binds the stratum; `time_dim` names the
time axis; every table dimension must be **either** indexed by the forcing
**or** the `time_dim`. A dimension left unaccounted for is a compile error
(**E229**) that names it — never a silent guess:

```
# clim : patch × season × week, but the forcing only indexes patch —
temperature[p in patch] : interpolated 'ratio { table = clim  time_dim = week }
# E229: table 'clim' has dimension 'season' that is neither indexed by the
#   forcing nor the time axis — index it ([p in patch, s in season]) or
#   aggregate it out.
```

Use `data =` for a forcing whose data is a standalone long-format file; use
`table =` when the data is (or should be) a `tables {}` matrix — reused
elsewhere in the model, or when the forcing is stratified over more than one
dimension (a single `data =` file has one key column, whereas `table =` composes
to `[p in patch, s in season]` + `time_dim = week`). The knots are sorted by
their numeric time, so the dimension's declaration order does not matter.

**For the immediate Nigeria model:** per-patch sinusoidal approximation via
parameters works:

```camdl
forcing {
  seasonal = sinusoidal(amplitude = 1.0, period = 365.25 'days,
    phase = phi_season, baseline = 1.0)
}

transitions {
  infection[p in patch] : S[p] --> I[p]
    @ beta[p] * seasonal * S[p] * I[p] / N[p]
}
```

Shared waveform, per-patch R0 absorbs amplitude differences. Good enough for the
forward generative model.

---

## Stress test: every epi data type

| Data type                      | Type signature                 | Works?                                    |
| ------------------------------ | ------------------------------ | ----------------------------------------- |
| Population                     | `patch → f64`                  | ✅                                        |
| Population × age               | `patch × age → f64`            | ✅                                        |
| Contact matrix                 | `age × age → f64`              | ✅                                        |
| Spatial adjacency (sparse)     | `patch × patch → f64`          | ✅ `default = 0.0`                        |
| SIA campaigns (ragged)         | `patch × sia_time → f64`       | ✅ `where > 0` filter                     |
| Demographics (multi-value)     | `patch → (f64, f64)`           | ✅ `pop, sex_ratio : patch = read(...)`   |
| Seroprevalence                 | `patch × age → f64`            | ✅                                        |
| Environmental covariates       | `patch → f64`                  | ✅                                        |
| Routine immunization           | `patch × age × ri_month → f64` | ✅ 3D table                               |
| Detection probability          | `patch → f64`                  | ✅                                        |
| Genetic distances              | `strain × strain → f64`        | ✅                                        |
| Climate / time-varying spatial | `patch × week → f64`           | ✅ indexed forcing (long-format `data =`) |
| Case data (for fitting)        | observation data               | separate pipeline (observations block)    |

Per-patch time-varying covariates load as an indexed interpolated forcing (a
long-format file, one series per stratum — see above). Everything else works
with `dims → scalar` tables, the `dimensions {}` block, and the existing
`[i in dim]` iteration.

---

## File format rules

**Extension determines separator:**

- `.csv` → comma-separated
- `.tsv` → tab-separated
- Other → error: `"unrecognized extension '.txt'; use .csv or .tsv"`

**Validation:**

- Compiler checks that the detected separator is actually present
- `.csv` file with tabs but no commas → error:
  `"file 'data.csv' has
  .csv extension but appears tab-separated; rename to .tsv"`
- First row is always a header (required)
- No-header detection: if first row parses as all-numeric when subsequent rows
  are also all-numeric, warn:
  `"first row of
  'data.tsv' looks like data, not a header; add a header row"`
- Comment lines: off by default. `read(..., comment = "#")` enables skipping
  lines starting with `#`

---

## Missing observations: `NA` is a hole, not a zero

In an **observation** file, the token `NA` in a value column marks a **hole**: a
time at which the stream was scheduled but no value was observed.

A hole is not an observed zero, and the difference is load-bearing:

- **A hole contributes no likelihood term.** It is marginalised out — the filter
  neither rewards nor penalises the model for what it did not see. An observed
  `0`, by contrast, is a real datum saying "we looked and counted none," and the
  model is scored against it.
- **A hole still closes an incidence bin.** For a stream whose `projected` is an
  `incidence(...)`, the accumulator resets on the observation SCHEDULE, not on
  value presence (the same convention as pomp's `accumvars`). So a row with `NA`
  ends the interval at its own time, and the NEXT row is scored over the span
  since that hole — not since the last row carrying a number.

That second property is the reason to write `NA` rather than delete a row. If a
row is dropped from the file entirely, the following row's exposure window
silently widens to span the gap, and its count is then compared against more
modelled flow than it represents. Keeping the row as `NA` preserves the window;
deleting it changes the meaning of the row after it.

Both forms are therefore legitimate and mean different things:

| you write              | the model sees                                   |
| ---------------------- | ------------------------------------------------ |
| `2026-07-14  NA`       | scheduled, unobserved — no term, interval closes |
| `2026-07-14  0`        | observed zero — scored as a zero count           |
| (no row at 2026-07-14) | not scheduled — the next row's interval spans it |

**A warning for downstream consumers.** `NA` is not a null token in every tool.
In polars it does not parse as null by default, so one `NA` turns a numeric
value column into a string column; a plotting library then treats those strings
as categorical and can draw them at the axis baseline — holes rendering as
_observed zeros_ in a figure that looks entirely plausible. Read observation
files with `NA` declared as the null value (in polars, `null_values="NA"`).

### Building the file: emit the schedule, not the usable subset

The rule above is stated from the reader's side. From the _writer's_ side it
inverts into one instruction, and it is the instruction most often got wrong:

> **Emit a row for every scheduled time. Make the unusable ones `NA`. Never
> filter them out.**

A pipeline that drops bad days is doing the natural thing — a filter is how you
remove data you cannot trust. But for an `incidence(...)` stream a filter does
not remove a day; it _merges_ that day into the next observation's window, while
the surviving row's count still covers only its own day.

**What that costs, from a real case.** A daily lab stream whose unusable days
were filtered rather than nulled left the retained rows up to **13 days apart**.
Each was scored against 13 days of accumulated modelled flow and one day of
specimens — a 13x inflation of `projected/tests`, which **refused 23 of 24
chains at initialisation**. That refusal is correct: a modelled flow above the
specimen count is impossible, not merely a poor fit. But nothing in the
diagnostic mentions the filter, and the pipeline that caused it looks like
ordinary data cleaning.

Re-indexing onto the full span is the fix, and it is three lines:

```python
span = pl.date_range(out["time"].min(), out["time"].max(),
                     interval="1d", eager=True).alias("time")
full = pl.DataFrame({"time": span}).join(out, on="time", how="left")
full.write_csv(path, separator="\t", null_value="NA")
```

Every scored bin is then exactly one cadence wide, and the days you could not
use contribute no observation at all.

**Proportion streams need a denominator on the unscored rows too.** For a stream
scored as `k / n`, fill `n` with `1` — or any positive value — where the
observation is `NA`. The value is never read, because the observation is a hole;
the fill exists so a `0` denominator cannot reach a division. Leaving it null or
zero makes the file's validity depend on how the projection happens to be
written.

**The count you should check.** After building, assert that the number of rows
equals the number of scheduled times, and that the number of _non-null_ rows
equals the number of usable observations. Those two numbers differing by exactly
the number of holes is the property this section exists to protect; if the row
count equals the usable count, the filter is still there.

---

## Complete Nigeria model with data model

```camdl
time_unit = 'days

compartments { S, E, I, R, V }

dimensions {
  age      = [under5, over5]
  patch    = read("data/demographics.tsv", column = "patch")
  sia_time = read("data/sia.tsv", column = "day")
}

stratify(by = age)
stratify(by = patch)

parameters {
  sigma    : rate        in [0.001, 1.0]
  gamma    : rate        in [0.001, 1.0]
  omega    : rate        in [0.0001, 0.1]
  kappa    : rate        in [0.0, 0.5]
  vacc_eff : probability in [0.0, 1.0]
  I0       : count       in [1, 10000]
  R0[patch] : positive   in [0.5, 15.0]
}

let beta[p in patch] = R0[p] * gamma
let N[a in age, p in patch] = S[a, p] + E[a, p] + I[a, p] + R[a, p] + V[a, p]

tables {
  # patch levels derived from demographics.tsv (238 LGAs)
  pop, init_sus : patch                = read("data/demographics.tsv")

  adj      : patch × patch             = read("data/adj.tsv", default = 0.0)
  age_frac : age                       = [0.18, 0.82]
  sia_cov  : patch × sia_time          = read("data/sia.tsv", default = 0.0)
}

transitions {
  infection[a in age, p in patch] : S[a, p] --> E[a, p]
    @ beta[p] * S[a, p] * I[a, p] / N[a, p]

  importation[a in age, p in patch, q in patch] : S[a, p] --> E[a, p]
    @ kappa * adj[p, q] * S[a, p] * I[a, q] / N[a, q]
    where p != q

  progression[a in age, p in patch] : E[a, p] --> I[a, p]  @ sigma * E[a, p]
  recovery[a in age, p in patch]    : I[a, p] --> R[a, p]  @ gamma * I[a, p]
  waning[a in age, p in patch]      : V[a, p] --> S[a, p]  @ omega * V[a, p]
}

interventions {
  sia[p in patch, t in sia_time] : transfer(
    fraction = vacc_eff * sia_cov[p, t],
    from = S[under5, p], to = V[under5, p],
    at = t
  ) where sia_cov[p, t] > 0
}

init {
  S[a in age, p in patch] = pop[p] * init_sus[p] * age_frac[a]
  I[under5, borno_damboa] = I0
}

simulate {
  from = 0 'days
  to   = 730 'days
}

scenarios {
  baseline {
    label = "no SIA"
  }
  with_sia {
    label = "with SIA"
    enable = [sia]
  }
}
```

---

## Summary

| Concept                     | Mechanism                                | Example                                    |
| --------------------------- | ---------------------------------------- | ------------------------------------------ |
| Small structural dimension  | `dimensions { X = [...] }`               | `age = [under5, over5]`                    |
| Large data-driven dimension | `dimensions { X = read(f, column="c") }` | `patch = read("pop.tsv", column="patch")`  |
| Index-only dimension        | `dimensions { X = ... }` (no stratify)   | `sia_time = read("sia.tsv", column="day")` |
| Dimension validation        | bare `X` in type signature               | adj references known `patch`               |
| Table loading               | `read("file.tsv")`                       | positional column → dimension mapping      |
| Sparse tables               | `default = 0.0`                          | missing index combinations → default       |
| Multiple values             | `a, b : dims = read(...)`                | two tables from one CSV                    |
| Numeric level coercion      | `at = t` where `t` iterates numeric dim  | campaign times as floats                   |
| Compile-time filtering      | `where expr > 0`                         | skip zero-coverage events                  |
| Time-varying spatial        | indexed forcing, long-format `data =`    | `temperature[p in patch]`                  |

**One loader, one table type, one iteration syntax.** All external data is
`dims → scalar`. All iteration is `[i in dim]`. Dimensions are declared in the
`dimensions {}` block, validated everywhere. No union, no intersection, no
implicit derivation. Every mapping from CSV to model is explicit and traceable.
