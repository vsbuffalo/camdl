# The camdl Language Specification

**Version:** 0.3-draft **Date:** 2026-03-16

_camdl (Compartmental Model Description Language) is a domain-specific language
for specifying stochastic compartmental models. A `.camdl` file defines model
structure. Parameter values, inference configuration, and scenario selection are
supplied externally._

---

## 1. Design Principles

**Primitives first.** The language defines a small set of composable primitives.
Convenience sugar is documented as expanding to primitives; users can always
write the explicit form. All design effort focuses on getting a minimal,
extensible, composable, non-blocking, flexible set of core primitives right.
Sugar is never added before the primitives it replaces are solid.

**Explicit over terse.** Named keywords everywhere. No hidden multiplication, no
auto-localization, no implicit scope rules. Every rate expression is a total
propensity — the compiler never silently multiplies by a population count. If a
rate is per-capita, the user writes the `* Pop` factor explicitly.

**Model ≠ parameterization.** The `.camdl` file defines M (parameter space) and
C (configuration). Parameter values come from external TOML files, CLI flags, or
inference engines. The seed is always a CLI argument. This follows the grammar
of model parameters (Buffalo 2026): the model is structurally stable across all
analyses — forward simulation, calibration, scenario comparison, and forecasting
all use the same `.camdl` file with different external configuration. This
separation means the model file can be committed to git, shared in paper
supplements, and reviewed independently of any particular parameter values or
analysis choices.

**Typed and checked.** Index dimensions, table shapes, compartment arities,
parameter domains, and unit dimensions are compiler-checked with clear error
messages. The compiler tracks which dimension each index variable belongs to and
rejects mismatches at compile time, not simulation time.

**No auto-localization.** After stratification, bare compartment names always
refer to the global total (sum over all strata). `S` means "all susceptibles."
`S[child]` means "susceptible children." The compiler never guesses which
stratum you meant. Stratification rules (coupling sugar) handle the
transformation from global to per-stratum formulas mechanically; the user writes
the base model with global names and specifies how dimensions interact.

### 1.1 Syntax Conventions

```
:    structural definition (what something IS)
=    value binding (what something EQUALS)
@    rate expression (how fast, always total propensity)
-->  flow direction
#    comment
{ }  block grouping
[ ]  index access and list literals
( )  function arguments
'    unit literal prefix ('days, 'years)
```

---

## 2. Time Unit and Dimensional Types

```
time_unit = 'days
```

All rates and durations are normalized to this unit at compile time.

### 2.1 Unit Literals

Unit literals are distinguished from identifiers by the `'` prefix:

```
# Duration (dimension: time)
5 'years
14 'days
2 'weeks
0.5 'years

# Rate (dimension: 1/time)
0.1 'per_day
0.02 'per_year
```

Supported units: `'days`, `'weeks`, `'months`, `'years`, `'per_day`,
`'per_week`, `'per_month`, `'per_year`.

Conversions: 1 'week = 7 'days, 1 'month = 30.4375 'days (365.25/12), 1 'year =
365.25 'days. The compiler uses exact rational arithmetic.

### 2.2 Dimensional Type System

Durations and rates are **distinct types**. The compiler tracks dimensions
through expressions and rejects mismatches:

```
'days     — dimension: time
'per_day  — dimension: 1/time (rate)
```

Valid operations:

```
5 'days + 3 'days         → 8 'days (time + time = time)
1 / (14 'days)            → rate (1/time)
0.1 'per_day * 100        → 10 'per_day (rate × scalar = rate)
0.1 'per_day * 5 'days    → 0.5 (rate × time = dimensionless) ✓
```

Invalid operations:

```
5 'days + 0.1 'per_day    → ERROR: cannot add time and rate
5 'days * 3 'days         → ERROR: time² has no meaning in this system
```

Mixed-unit values in tables that aren't compatible are compile errors.
Dimensionless zero (`0.0`) is compatible with any unit context.

### 2.2.1 Transition Rate Dimensional Analysis

The compiler checks that every transition rate expression has dimension
**P·T⁻¹** (population per unit time). This catches the most common modeling
bug: writing a per-capita rate where a total propensity is needed.

The checker infers dimensions from:
- Compartment references (`Pop`, `PopSum`) → dimension **P**
- `Time` → dimension **T**
- Parameter types: `rate` → **T⁻¹**, `probability` → **1**, `count` → **P**
- Unit literals: `'days` → **T**, `'per_day` → **T⁻¹**
- Arithmetic rules: multiplication adds exponents, division subtracts

```
# ✓ Correct: beta:T⁻¹ * S:P * I:P / N:P = P·T⁻¹
infection : S --> I @ beta * S * I / N

# ✗ Error E300: beta:T⁻¹ * I:P / N:P = T⁻¹ (missing S)
infection : S --> I @ beta * I / N
#   error[E300]: transition 'infection' rate has wrong dimension
#     expected: P·T⁻¹ (population-level rate)
#     got:      T⁻¹ (per-capita rate)
```

Parameters with `kind = positive` or `kind = real` have unknown dimension —
the checker infers it from context. If inference is ambiguous, use a `[dim]`
annotation (see §4.1.1). If a parameter is used inconsistently across
transitions, the compiler emits E303.

Additional checks:
- **E301**: argument to `exp()` / `log()` must be dimensionless
- **E302**: addition/subtraction of mismatched dimensions
- **E304**: `sqrt()` of odd-exponent dimension
- **E305**: balance expression must have dimension P
- **E306**: ODE derivative must have dimension P·T⁻¹
- **E308**: overdispersion σ² must be dimensionless

Disable with `--no-dim-check` if a false positive is encountered (and file
a bug).

### 2.3 Date Literals

The `date("YYYY-MM-DD")` expression converts an ISO 8601 date to a float offset
from the model's declared `origin` date, in the model's `time_unit`:

```
origin = date("2019-01-01")   # top-level declaration (optional)

simulate {
  to = date("2021-06-30")     # 912 days from origin (in time_unit = 'days)
}
```

`date(...)` uses proleptic Gregorian calendar arithmetic. The result is exact
(integer day count) when `time_unit = 'days`, and divided by the appropriate
factor for other units (e.g., 365.25 for years).

**E220.** Using `date(...)` without a top-level `origin = date(...)` declaration
is a compile error:

```
# ERROR E220: date("2021-06-30") used but no 'origin' declared
simulate { to = date("2021-06-30") }
```

The `origin` value is stored in the IR (`"origin": "2019-01-01"`). It does not
affect simulation dynamics — it is purely a coordinate reference for converting
calendar dates to simulation time.

### 2.4 Table Unit Annotations

Tables carry a single unit for all values:

```
tables {
  fertility  : age 'per_day   = [0.0, 0.02]
  age_dur    : age 'years     = [5, 60]
  C_age      : age × age      = [[12.0, 4.0], [4.0, 8.0]]  # dimensionless
}
```

`fertility : age 'per_day` means "every value in this table is in units of
'per_day." The compiler normalizes to the model time unit. Dimensionless tables
(contact matrices, weights) have no unit annotation.

---

## 3. Compartments

```
compartments { S, E, I, R }
```

Each is an integer-valued population count. For continuous state:

```
compartments {
  S, I, R
  W : real       # continuous-valued (environmental reservoir)
}
```

After stratification, compartments gain index dimensions (see §5). Access is
always via explicit indexing: `S[child]`, `S[child, female]`, or bare `S` (= sum
over all strata).

---

## 4. Parameters

```
parameters {
  beta     : rate
  gamma    : rate
  sigma    : rate
  mu       : rate
  rho      : probability
  k        : positive
  N0       : count
  I0       : count
}
```

Parameters are **declared** here. Default values may optionally be specified in
the model file. Concrete values for inference are supplied externally via CLI
flags or inference engines.

### 4.1 Parameter Types

```
rate        : ≥ 0, dimension 1/time. Default transform: log.
probability : ∈ [0, 1], dimensionless. Default transform: logit.
positive    : > 0, dimensionless. Default transform: log.
count       : integer ≥ 0.
real        : unconstrained (default if omitted).
```

Types enable: validation of supplied values, default inference transforms,
and dimensional analysis of rate expressions.

### 4.1.1 Dimension Annotations

When a parameter's type doesn't fully determine its dimension (e.g.,
`positive` could be a rate, a count, or dimensionless), you can add an
explicit dimension annotation:

```
parameters {
  beta      : rate                    # dimension T⁻¹ (inferred from type)
  gamma     : rate                    # T⁻¹
  amplitude : real [1]                # explicitly dimensionless
  mu        : positive [1/T]          # per-capita rate
  coupling  : positive [P/T]          # population-level rate
  R0        : positive [1]            # dimensionless
}
```

The annotation goes in square brackets after the type, before optional
bounds. Supported dimension literals:

| Annotation  | Dimension              | Domain name              |
| ----------- | ---------------------- | ------------------------ |
| `[1]`       | dimensionless          | probability, ratio, R₀   |
| `[P]`       | population             | count                    |
| `[T]`       | time                   | duration                 |
| `[1/T]`     | T⁻¹                   | per-capita rate          |
| `[T^-1]`    | T⁻¹ (alternate)       | per-capita rate          |
| `[P/T]`     | P·T⁻¹                 | population-level rate    |
| `[P*T^-1]`  | P·T⁻¹ (alternate)     | population-level rate    |

When present, the annotation overrides type-based inference. If the
annotation conflicts with how the parameter is used in rate expressions,
the compiler emits a dimension error.

Annotations are optional. The compiler infers dimensions from context for
most models — annotations are only needed when inference is ambiguous
(reported as info I300).

### 4.2 External parameter values

Parameter values are **never** specified inside `.camdl` files. The model file
declares names and types only; concrete values are supplied at runtime:

```bash
# Single flat TOML file
camdl-sim simulate model.ir.json --params base.toml

# Layered overrides (later files win)
camdl-sim simulate model.ir.json --params base.toml --params patch.toml

# Single value override
camdl-sim simulate model.ir.json --param gamma=0.1

# Per-stratum override (indexed params)
camdl-sim simulate model.ir.json --param-vec R0=r0_posterior.tsv
```

The TOML format supports both flat and sectioned forms (see §22).

### 4.3 Indexed Parameters

Parameters may be declared with a single dimension index, creating one scalar
parameter per stratum:

```
parameters {
  gamma    : rate
  N[patch] : positive   # expands to N_urban, N_rural, ...
  R0[patch]: positive   # expands to R0_urban, R0_rural, ...
}
```

The index must refer to a declared `stratify` dimension. In expressions, indexed
parameters are accessed with `[index]`:

```
let beta[p in patch] = R0[p] * gamma   # R0[p] → Param("R0_urban") etc.
```

**Index namespace rule.** Inside `[...]` on a parameter reference, the compiler
checks only:

1. The current substitution environment (bound index variables like `p`)
2. The literal dimension values (e.g., `R0[urban]` → `Param("R0_urban")`)

Let bindings and other parameters are never checked in index position.
`R0[urban]` always means the stratum value `urban`, even if a let binding named
`urban` exists.

**Shadowing warning W103.** The compiler emits W103 when a let binding name
matches a stratum value in any dimension:

```
let urban = 1.0   # W103: let binding 'urban' shadows stratum value 'urban'
                  #   in dimension 'patch'. This is allowed but consider renaming.
```

**Consistent indexed syntax everywhere.** The `N0[urban]` form works in all
contexts where a parameter name can appear: expressions (§9.6), init blocks
(§16.2), and scenario set/scale blocks (§18.1). The compiler always mangles to
`N0_urban` in the IR.

**IR representation.** Indexed parameter declarations expand to flat scalar
parameters:

```
N[patch] : positive  →  { name: "N_urban",  value: null }
                         { name: "N_rural",  value: null }
```

**Runtime override.** Use `--param-vec PREFIX=FILE` to supply per-stratum values
at runtime (see §22):

```bash
camdl-sim simulate model.ir.json --param-vec R0=/tmp/r0_posterior.tsv
```

### 4.4 Parameter Bounds

An optional `in [lo, hi]` clause constrains the parameter's valid range:

```
parameters {
  R0       : positive in [1.0, 20.0]    # scalar with bounds
  rho      : probability in [0.0, 1.0]  # redundant but explicit
  R0[patch]: positive in [0.5, 15.0]    # all strata get same bounds
  gamma    : rate                        # unbounded (beyond type constraint)
}
```

Bounds are **optional** and apply to all expanded scalar parameters for indexed
declarations. They are stored in the IR:

```json
{ "name": "R0", "value": null, "bounds": [1.0, 20.0], ... }
```

Bounds are used by inference engines to constrain sampling or optimization; the
forward simulator does not enforce them at runtime. The compiler does not
validate that supplied values lie within bounds — that is the inference engine's
responsibility.

Type constraints still apply independently of bounds: a `positive` parameter
with `in [1.0, 20.0]` is implicitly also constrained to `> 0`.

---

## 5. Index Dimensions and Stratification

Dimension levels are declared in a `dimensions {}` block. Levels can be inline
or read from a data file:

```
dimensions {
  age   = [child, adult]
  sex   = [female, male]
  patch = read("data/lga_pop.tsv", column = "patch")   # levels from data column
}
stratify(by = age)
stratify(by = sex)
stratify(by = patch)
```

Each `stratify` declaration applies a dimension to **all** compartments by
default. Partial stratification restricts to specific compartments:

```
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])
```

After this, S/E/I have dimensions `[age, sex]` but R has `[age, sex, immunity]`.

### 5.1 Indexing Rules

**Positional indexing** (declaration order of stratify blocks):

```
S[child]                # first dimension = age
S[child, female]        # age, then sex
S                       # bare = sum over ALL strata (always global)
S[child]                # if S has [age, sex]: sum over sex for age=child
```

**Named indexing** (explicit dimension labels, any order):

```
S[age = child]                    # equivalent to S[child]
S[sex = female, age = child]      # order doesn't matter
S[patch = p1]                     # sum over age, specific patch
incidence(infection[patch = p])   # sum over age, specific patch
```

Named indexing is useful when a compartment or transition has multiple
dimensions and you want to index a non-first dimension. The compiler resolves
named indices to positional and validates dimension membership.

Positional and named indexing can be mixed: `S[child, sex = female]` is valid
(first positional = age, second named = sex). But for clarity, use one style
consistently.

**Omitting a dimension sums over it.** The compiler knows each compartment's
arity and checks every access.

**In rate expressions** (right of `@`): omitting dimensions is valid — it
produces a sum (a scalar read). `R[a]` when R has `[age, immunity]` means
`R[a, natural] + R[a, vaccine]`.

**In stoichiometry** (left of `@`, source/destination of `-->`): **all
dimensions of the compartment must be specified.** You cannot write into a
marginal — the compiler must know exactly which cell gains or loses an
individual.

```
# ERROR: R has [age, immunity] but only [age] specified in destination
recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]

# CORRECT: specify where recovered individuals go
recovery[a in age] : I[a] --> R[a, natural]  @ gamma * I[a]
```

This rule ensures partial stratification forces the modeler to make explicit
routing decisions — which is exactly the point of partial stratification.

### 5.2 Index Variables

Index variables are bound by transition indices or `sum`:

```
[i in age]              # binds i to iterate over age values
sum(j in age, expr)     # binds j, sums expr over age values
```

The `in dim` clause makes the dimension explicit. The compiler tracks which
dimension each variable belongs to.

### 5.3 Partial Stratification in Expressions

When compartments have different dimensions due to `only = [...]`, bare names
and indexed access follow the same rules — but the compiler resolves them
per-compartment based on each compartment's actual arity.

Example: `E` has `[age, latent_stage]`, `S` has `[age]`.

```
S + E                # both are global sums: PopSum(all S) + PopSum(all E). Valid.
S[a] + E[a]          # S in age=a + E in age=a (summed over latent_stage). Valid.
S[a] + E[a, e1]      # S in age=a + E in age=a and stage=e1. Valid.
S[a, e1]             # ERROR: S has no latent_stage dimension.
```

The omitted-dimension-sums rule applies per-compartment: `E[a]` sums over
`latent_stage` because `E` has that dimension, while `S[a]` is fully resolved
because `S` has only `[age]`. The compiler tracks each compartment's dimensions
independently.

---

## 6. Tables

```
tables {
  C_age      : age × age          = [[12.0, 4.0], [4.0, 8.0]]
  B_sex      : sex × sex          = [[0.0, beta_mf], [beta_fm, 0.0]]
  mu_age     : age 'per_day       = [0.0000685, 0.0000411]
  fertility  : age 'per_day       = [0.0, 0.02]
  age_dur    : age 'years         = [5, 60]

  # File-based data (long format)
  kernel     : patch × patch      = read("data/spatial_kernel.tsv")
  distances  : patch × patch      = read("data/lga_dist.tsv", default = 0.0)

  # Patch population (levels were declared in dimensions block)
  pop        : patch              = read("data/lga_pop.tsv")
}
```

### 6.1 Dimension and Unit Annotations

**Required** in v0.1. The `: dim × dim` annotation enables:

- **Shape validation.** `C_age : age × age` with 2 age values → must be 2×2.
- **Index type checking.** `C_age[i, j]` requires both `i : age` and `j : age`.
  Using `C_age[i, s]` where `s : sex` is a compile error.
- **Documentation.** The annotation tells you what each axis means.

The optional unit annotation (e.g., `: age 'per_day`) specifies the unit for all
values. The compiler normalizes to the model time unit and checks dimensional
consistency when table values appear in expressions.

Multi-dimensional: `: age × sex × risk` for 3D tables. Inline via nested
brackets. For large tables, use `read(...)` (see §6.2).

### 6.2 Loading from Files: `read`

All file-based tables use **long format** (one row per observation, index
columns then value column):

```
# data/lga_pop.tsv
patch           pop
kano_dala       485000
borno_maiduguri 345000
borno_gwoza      78000
```

```
tables {
  pop : patch = read("data/lga_pop.tsv")
}
```

The type signature declares how many index columns there are (one per dimension
listed). The remaining column(s) are value(s). Column names in the file are for
human readability — the compiler uses **positional mapping** from the type
signature.

**Extension determines separator:** `.tsv` → tab, `.csv` → comma, anything else
→ compile error. The first row is always a required header.

**Sparse tables** use `default = value` to fill index combinations missing from
the file:

```
tables {
  distances : patch × patch = read("data/lga_dist.tsv", default = 0.0)
}
```

```
# data/lga_dist.tsv — only nonzero pairs listed
src             dst             distance
kano_dala       borno_maiduguri 245.3
kano_dala       kano_fagge      18.1
borno_maiduguri kano_dala       245.3
```

Index values are the actual level names (not integer positions). The compiler
validates each value against the known dimension levels and errors on typos.
Without `default`, every index combination must have a row (dense check).

### 6.3 Data-Derived Dimension Levels: `dimensions { dim = read(...) }`

For large models (hundreds of patches), listing levels inline is impractical.
Use `read(...)` in the `dimensions {}` block to derive dimension membership from
a data file column:

```
dimensions {
  patch = read("data/lga_pop.tsv", column = "patch")
}

stratify(by = patch)

tables {
  pop : patch = read("data/lga_pop.tsv")
}
```

The `read(file, column = "col")` form reads the named column, collects unique
values in first-occurrence order, and those become the levels of the dimension.
All tables referencing `patch` validate against these derived levels.

**Rules:**

- Each dimension is defined exactly once. Two `patch = [...]` entries → compile
  error.
- Inline `[...]` and `read(...)` are mutually exclusive for the same dimension.
- A `stratify(by = X)` whose dimension `X` was declared via `read(...)` must be
  present; levels come entirely from the file.
- Bare dimension names in type signatures (`pop : patch = ...`) validate against
  the known levels; typos → error with Levenshtein suggestion.

### 6.4 Multi-Value Columns

When a file has more columns than `n_dims + 1`, list multiple table names on the
left of `:`:

```
tables {
  pop, init_sus : patch = read("data/demographics.tsv")
}
```

```
# data/demographics.tsv
patch           pop     init_sus
kano_dala       485000  0.88
borno_maiduguri 345000  0.91
borno_gwoza      78000  0.79
```

Creates two tables with the same index: `pop[kano_dala] = 485000`,
`init_sus[kano_dala] = 0.88`. Value columns map positionally to the names on the
left. Name count must match non-index column count.

### 6.5 External Table Loading

External tables are loaded at compile time and inlined into the IR. The IR is
self-contained — no file references at runtime. For very large tables (>1M
entries), binary IR format (msgpack) is recommended over JSON.

### 6.6 Parameterized Table Entries

Inline table values can be parameter names or arithmetic expressions, not just
numeric literals:

```
tables {
  B_sex : sex × sex = [[0.0,     beta_mf],
                        [beta_fm, 0.0    ]]
}
```

Here `beta_mf` and `beta_fm` are parameters. In the IR, these entries are stored
as `Param("beta_mf")` expression nodes, not resolved floats. The table is fully
resolved only when parameter values are supplied at simulation time. This
enables inference over contact matrix entries.

Tables mixing literals and parameter expressions are valid:
`[[0.0, beta_mf], ...]` has a constant zero and a parameter reference in the
same row.

---

## 7. Forcing

Named time-dependent forcing functions, usable in rate expressions. Four
built-in types cover real-world needs:

- `sinusoidal` — smooth seasonal forcing
- `periodic` — repeating step function (day-of-week, month-of-year effects)
- `piecewise` — non-repeating step function (policy changes, campaign windows)
- `interpolated` — data-driven time series (empirical covariates)

```
forcing {
  seasonal = sinusoidal(
    amplitude = alpha,        # can reference parameters (for inference)
    period    = 365.25 'days,
    phase     = phi_season,   # convention: time from t=0 to peak, in model time_unit
    baseline  = 1.0
  )

  lockdown = piecewise(
    breakpoints = [60 'days, 120 'days],
    values      = [1.0, 0.3, 1.0]
  )

  pop_trend = interpolated(
    times  = population.time,   # reference to data block
    values = population.total,
    method = "cubic_spline",    # "linear" | "cubic_spline" | "pchip"
    knots  = "natural"          # "natural" | "clamped" | "not_a_knot"
  )

  reporting_dow = periodic(
    period = 7 'days,
    values = [1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]
  )

  # Range-based periodic: specify active ranges instead of listing values.
  # step = bin width; on = list of lo:hi ranges where the value is 1.0.
  # Bins outside the ranges are 0.0. The compiler generates the values array.
  school : periodic {
    period = 365.25 'days
    step   = 1 'days
    on     = [7:100, 115:199, 252:300, 308:356]
  }
}
```

Forcing functions compile to `TimeFunc` nodes in the IR. Their arguments can
reference parameters (e.g., `amplitude = alpha`), enabling inference over
function characteristics (e.g., inferring seasonal amplitude).

The `periodic` type supports two forms:

- **Values form:** `values = [1.0, 0.0, 1.0, ...]` — explicit list. Bin width =
  period / len(values). Use for general periodic patterns.
- **Range form:** `step = 1 'days` + `on = [7:100, 115:199]` — binary on/off
  with range literals. The compiler generates the values array. Use for
  calendars with known active periods (school terms, work weeks, campaign
  windows).

Forcing functions are used in rate expressions by name or with explicit `(t)`:

```
transitions {
  infection : S --> I  @ beta * school(t) * S * I / N
  #                          ^^^^^^^^ forcing function reference
}
```

Both `school` and `school(t)` are valid; `school(42)` or other non-`t` arguments
produce an error.

`periodic` is primarily useful in v0.2 reporting pipelines for day-of-week
effects on case reporting.

### Seasonal forcing and dimensional analysis

There are two common parameterizations for seasonal forcing in compartmental
models, and they differ dimensionally:

1. **camdl style (sinusoidal IS the rate):** The `sinusoidal` construct produces
   a value with the same dimension as the `baseline` parameter. When the
   baseline has dimension T^-1 (i.e., when it references a `rate` parameter),
   the forcing function itself carries the rate dimension:
   ```
   infection : S --> I  @ seasonal * S * I / N
   ```
   Here `seasonal` already has dimension T^-1, so the full rate expression is
   P * T^-1 as required.

2. **pomp style (sinusoidal is a dimensionless multiplier):** In some
   frameworks, the sinusoidal function is a pure multiplier around 1.0
   (dimensionless), and the rate parameter appears separately:
   ```
   infection : S --> I  @ beta * (1 + amplitude * sin(...)) * S * I / N
   ```
   Here `beta` carries the T^-1 dimension and the sinusoidal part is
   dimensionless.

Both patterns are valid. camdl's `sinusoidal(baseline = beta, ...)` produces a
value whose dimension matches `baseline`, so it naturally follows pattern (1).
To use pattern (2), set `baseline = 1.0` (dimensionless) and multiply by the
rate parameter explicitly. The dimensional analysis checker handles both
correctly.

---

## 8. Let Bindings

`let` declarations are **top-level** — they appear between blocks, never inside
a block. They are resolved after the full file is parsed (order does not matter
between `let` and other declarations).

```
let N = S + E + I + R
let N_local[a in age, p in patch] = S[a,p] + E[a,p] + I[a,p] + R[a,p]
let foi[a in age, p in patch] = sum(b in age, C_age[a,b] * I[b,p] / N_local[b,p])
```

### 8.1 Scope Rules

**Bare names are always global.** `let N = S + E + I + R` means the total across
ALL strata. After age stratification, N is still the global total. No
auto-localization, ever.

**Indexed let bindings** define computed quantities over dimensions:

```
let N_local[a in age] = S[a] + E[a] + I[a] + R[a]   # per-age-group total
let mig[i in patch, j in patch] = theta * pop[j] / (distance[i,j] ^ 2)
```

The dimension annotation is inferred from the index bindings. `N_local` has type
`: age`, `mig` has type `: patch × patch`.

**Let binding names must be unique.** Two bindings with the same name but
different index signatures (e.g., `let N_local[a in age]` and
`let N_local[a in age, s in sex]`) are a compile error — no overloading by
arity. Use distinct names: `N_age`, `N_age_sex`.

### 8.2 Indexed Let vs Sum

These are two different operations with a common binding syntax:

`let f[i in age] = expr` **defines a family of values** — one per index value.
It is a function from age-index to value. `f[child]` evaluates `expr` with
`i = child`; `f[adult]` evaluates `expr` with `i = adult`.

`sum(i in age, expr)` **reduces** — it evaluates `expr` for all values of `i`
and adds the results, producing a scalar.

They compose: `sum(i in age, f[i]) = f[child] + f[adult]`.

```
# Define per-stratum totals
let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

# N_local[child] = S[child] + E[child] + I[child] + R[child]
# N_local[adult] = S[adult] + E[adult] + I[adult] + R[adult]

# Sum them to get global total
# sum(a in age, N_local[a]) = N_local[child] + N_local[adult] = N
```

### 8.3 No Localization, No Magic

The mixing formula in the base model uses global N:

```
let N = S + I + R
infection : S --> I  @ beta * S * I / N    # global N, no indices
```

After stratification, the compiler transforms this based on the coupling rules
(see §10). The user writes the base model with global names. The coupling rules
produce the correct indexed formulas in the IR.

For explicit per-stratum FOI, the user writes the indexed form directly:

```
infection[a in age] : S[a] --> I[a]
  @ beta * S[a] * sum(b in age, C_age[a,b] * I[b] / N_local[b])
```

Both paths produce the same IR. The first uses sugar (§10); the second is the
primitive.

### 8.4 Typed Let Bindings

A `let` binding may carry a type annotation from the same set of kinds used for
parameters: `rate`, `probability`, `positive`, `count`, `real`.

```
let iota : count = 1e-6
let obs_floor : count = 0.01
let mu_annual : rate = 0.0002 'per_year
```

When a typed `let` has a constant body (`EConst`, `EUnit`, or their negation),
the compiler emits it as a fixed-value parameter in the IR with `param_kind` set
and `value` populated. This means the dimensional analysis checker sees the
declared dimension rather than treating the constant as dimensionless.

Without the type annotation, a bare constant like `1e-6` is dimensionless and
adding it to a population compartment (`I + 1e-6`) triggers E302. The typed form
(`I + iota` where `iota : count`) resolves this because `iota` carries the
population dimension.

For non-constant bodies (e.g., `let N = S + I + R`), the type annotation is
accepted syntactically but the binding is still inlined as usual. The dimension
of such expressions is inferred from their structure.

---

## 9. Transitions

The core dynamics. Every transition has a name, stoichiometry, and rate.

### 9.1 Syntax

```
# Transfer: source --> destination
infection[a in age] : S[a] --> E[a]  @ beta * S[a] * I[a] / N_local[a]

# Inflow (exogenous): no source compartment
birth[p in patch] : --> S[child, p]
  @ mu * sum(a in age, N_local[a, p])

# Outflow: no destination
death_S[a in age, p in patch] : S[a,p] -->  @ mu_age[a] * S[a,p]

# Block form for additional properties
infection_water : S --> I {
  rate = S * beta_W * W / (K + W)
  tag  = "waterborne"
}
```

**Block form properties:**

- `rate` (required): the total propensity expression
- `tag` (optional): a string label that compiles to the IR `metadata` field.
  Used for output filtering, visualization grouping, and documentation. Has no
  effect on simulation dynamics.

**Inflows** (`-->` with nothing on the left) model individuals entering the
system from outside: births, importation, immigration. There is no source
compartment — the rate expression says how fast new individuals appear.
Stoichiometry: `[(destination, +1)]`.

**Importation** is an inflow of infected individuals from an external source,
typically at a constant or data-driven rate unrelated to the model's own state.
Unlike births (which depend on the existing population), importation represents
exogenous exposure — cases entering the modeled region from elsewhere:

```
# Constant importation rate (exogenous FOI)
importation[a in age, p in patch] : --> I[a, p]
  @ import_rate * age_weights[a] * patch_weights[p]
```

### 9.2 Indexed Transitions

```
transition_name[i in dim1, j in dim2, ...] : from --> to  @ rate
```

The `[i in dim]` clause binds index variables. The compiler generates one
concrete IR transition per combination of index values. Dimensionality is known
at compile time: `|dim1| × |dim2| × ...` transitions.

### 9.3 Guard Clauses (`where`)

The `where` clause filters which index combinations generate transitions:

```
# Migration: exclude self-loops
migrate[c in compartments, a in age, src in patch, dst in patch]
  : c[a,src] --> c[a,dst]
  @ mig[dst,src] * c[a,src]
  where src != dst

# Only adults reproduce
birth_from[a in age, p in patch] : --> S[child, p]
  @ fertility[a] * N_local[a, p]
  where a != child

# Compound guard
transfer[a in age, src in patch, dst in patch] : S[a,src] --> S[a,dst]
  @ rate * S[a,src]
  where src != dst and a == adult
```

**Guard grammar:**

```
guard := index_var '!=' index_val_or_var
       | index_var '==' index_val_or_var
       | guard 'and' guard
       | guard 'or' guard
       | '(' guard ')'
```

Guards reference **index variables only** (not parameters or compartments). They
are evaluated at **compile time** — the compiler instantiates all index
combinations, evaluates the guard for each, and emits IR transitions only for
combinations where the guard is true. The IR has no concept of guards.

Guards compose with all iteration forms: regular `[i in dim]`, `consecutive`,
and `c in compartments`.

### 9.4 Consecutive Pair Iterator

The `consecutive(dim)` binding yields adjacent pairs from an ordered dimension:

```
aging[c in compartments, (a, a_next) in consecutive(age), p in patch]
  : c[a, p] --> c[a_next, p]
  @ (1 / age_dur[a]) * c[a, p]
```

For `age = [age_0_5, age_5_15, age_15_50, age_50_65, age_65p]`, this generates
four transitions per compartment per patch: `age_0_5→age_5_15`,
`age_5_15→age_15_50`, `age_15_50→age_50_65`, `age_50_65→age_65p`. The last
stratum has no outgoing aging transition.

This is a general-purpose primitive for any sequential transfer along an ordered
dimension. It also handles **Erlang sub-staging** for non-exponential waiting
times:

```
# Erlang-3 latent period: E passes through 3 sub-stages
dimensions { erlang_E = [e1, e2, e3] }
stratify(by = erlang_E, only = [E])

progression[(s, s_next) in consecutive(erlang_E)]
  : E[s] --> E[s_next]
  @ 3 * sigma * E[s]       # k * sigma for Erlang-k

# Final sub-stage transitions to I
progression_final : E[e3] --> I
  @ 3 * sigma * E[e3]
```

This gives an Erlang(k=3, rate=sigma) distributed latent period. The mean is the
same as exponential (1/sigma), but the variance is reduced by factor k,
producing a more peaked distribution — closer to real disease progression.

### 9.5 Compartment Iteration

The `c in compartments` binding iterates over compartment names:

```
# Death for all compartments
death[c in compartments, a in age, p in patch] : c[a,p] -->
  @ mu * c[a,p]

# Migration for all compartments
migrate[c in compartments, a in age, src in patch, dst in patch]
  : c[a,src] --> c[a,dst]
  @ mig[dst,src] * c[a,src]
  where src != dst
```

**`compartments` means integer compartments only** (the safe default). Real-
valued compartments (like environmental reservoirs `W : real`) are excluded
because population-level operations (death, migration) don't apply to continuous
state.

**Partial stratification and `c in compartments`.** When compartments have
different arities (e.g., R has `[age, patch, immunity]` but S has
`[age, patch]`), the compiler **expands over all omitted dimensions**. For
`death[c in compartments, a in age, p in patch] : c[a,p] --> @ mu * c[a,p]`:

- For S (dims: [age, patch]): generates
  `death_S[a, p] : S[a,p] --> @ mu * S[a,p]`
- For R (dims: [age, patch, immunity]): generates **separate transitions per
  immunity value**:
  `death_R[a, p, natural] : R[a,p,natural] --> @ mu * R[a,p,natural]` and
  `death_R[a, p, vaccine] : R[a,p,vaccine] --> @ mu * R[a,p,vaccine]`

This is correct: the stoichiometry rule (§5.1) requires all dimensions to be
specified for source/destination. The `c in compartments` iterator automatically
fills in omitted dimensions by iterating over them. The user writes `c[a,p]` and
the compiler expands to the correct full-arity transitions for each compartment.

### 9.6 Rate Expressions

The `@` rate is always the **total propensity** — the absolute event rate. No
hidden per-capita multiplication. If you want per-capita semantics, write the
population factor explicitly:

```
death_S[a in age] : S[a] -->  @ mu * S[a]     # mu per capita, explicit * S[a]
recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]  # gamma per capita, explicit * I[a]
```

### 9.7 Expression Grammar

**Operator precedence** (highest to lowest):

```
Precedence  Operators        Associativity
─────────────────────────────────────────
1 (highest) ()  f()  x[]     —
2           - (unary)        right
3           ^                right
4           * /              left
5           + -              left
6           == != < > <= >=  non-associative
7 (lowest)  if/then/else     right
```

Standard mathematical convention: `a + b * c` parses as `a + (b * c)`.
Exponentiation is right-associative: `a ^ b ^ c` = `a ^ (b ^ c)`. Comparisons
cannot be chained: `a < b < c` is a parse error (use `a < b and b < c` in
`where` guards).

**Full grammar:**

```
expr := expr '+' expr | expr '-' expr
      | expr '*' expr | expr '/' expr
      | expr '^' expr
      | '-' expr
      | IDENT                             # parameter, compartment, let binding, function
      | FLOAT | FLOAT UNIT                # literal, optionally with unit
      | IDENT '[' index (',' index)* ']'  # index access (positional or named)
      | sum '(' IDENT 'in' IDENT ',' expr ')'  # summation
      | IDENT '(' kwargs ')'              # function call
      | 'if' expr 'then' expr 'else' expr
      | expr '==' expr | expr '!=' expr | expr '<' expr | expr '>' expr
      | expr '<=' expr | expr '>=' expr
      | '(' expr ')'

index := expr                             # positional: S[child]
       | IDENT '=' expr                   # named: S[age = child]
```

Comparison operators are available for `where` guards and summary expressions.
`sum` is a keyword, not a user-definable function.

**Built-in math functions.** These are recognized by the compiler as
function-call syntax and produce IR expression nodes (not forcing functions):

| Function | Arity | Result |
|----------|-------|--------|
| `exp(x)` | 1 | e^x |
| `log(x)` | 1 | Natural logarithm (ln). Returns -∞ for x ≤ 0. |
| `sqrt(x)` | 1 | Square root. Returns 0 for x < 0. |
| `abs(x)` | 1 | Absolute value |
| `floor(x)` | 1 | Floor (round toward -∞) |
| `ceil(x)` | 1 | Ceiling (round toward +∞) |
| `mod(a, b)` | 2 | Euclidean remainder (always non-negative). Returns 0 for b = 0. |

Example:

```
let day_of_year = mod(t, 365.25)
let pop_decay = N0 * exp(-mu * t)
let is_pulse = (day_of_year > 250.0) * (day_of_year < 252.0)
```

**Rate wrappers.** Two compiler-recognized forms modify how event counts
are drawn for a transition. They are NOT general-purpose functions — they
wrap the entire rate expression and are extracted by the compiler during
expansion.

| Wrapper | Syntax | Effect |
|---------|--------|--------|
| `overdispersed(rate, σ²)` | `@ overdispersed(beta * S * I / N, sigma_se)` | Gamma-Poisson (NegBinomial) draws. Var = mean + mean²·σ²/dt. |
| `deterministic(rate)` | `@ deterministic(mu * N)` | Rounded integer: nearbyint(rate × dt). No stochastic noise. |

These are documented in §9.9 (overdispersion) and are compatible with
tau-leap and chain-binomial backends. Gillespie and ODE reject models
with `overdispersed()` transitions.

**Compile-time vs runtime `if/else`.** The `if/then/else` expression has two
evaluation modes depending on context:

- **In `let` bindings with index variables**: if the condition involves only
  index variables and constants, it is evaluated at **compile time**. The
  compiler instantiates one value per index combination and evaluates the
  condition for each. Example:
  ```
  let mig[i in patch, j in patch] =
    if i == j then 0.0 else theta * pop[j] / (distance[i,j] ^ 2)
  ```
  For each `(i, j)` pair, the compiler evaluates `i == j` and produces either
  `Const(0.0)` or the gravity expression in the IR. No runtime `Cond` node.

- **In rate expressions referencing compartment state**: the condition is
  evaluated at **runtime** and compiles to an IR `Cond` node. Example:
  ```
  @ if I > 0 then beta * S * I / N else 0.0
  ```
  This becomes `Cond(Pop("I"), <rate_expr>, Const(0.0))` in the IR.

Names are resolved in order: **compartments → parameters → let bindings →
forcing → tables**. The compiler reports an error if a name exists in multiple
namespaces. User names cannot shadow reserved identifiers (see §15).

### 9.8 Extra-Demographic Stochasticity (`overdispersed`)

Demographic stochasticity (Poisson event draws) scales as 1/√N and is negligible
for large populations. Extra-demographic stochasticity models rate-level noise —
correlated fluctuations in contact rates, weather effects, superspreading — that
doesn't scale away with population size (He et al. 2010).

The `overdispersed(rate_expr, σ²_SE)` function wraps a rate expression with
Gamma-distributed multiplicative noise:

```
infection : S --> I  @ overdispersed(beta * S * I / N, sigma_se)
recovery  : I --> R  @ gamma * I
```

The first argument is the base rate (a standard propensity expression). The
second is σ²_SE, the variance of the Gamma noise multiplier (which has mean 1).
The resulting event count distribution is NegBinomial — the Poisson-Gamma
compound.

`overdispersed` is syntactically a function call in expression position. The
compiler extracts it during expansion: the inner rate goes to `transition.rate`,
the variance goes to `transition.overdispersion` in the IR. Transitions without
`overdispersed` have `overdispersion: null` — standard Poisson draws.

**Backend compatibility.** Overdispersion is incompatible with Gillespie SSA
(which assumes deterministic rates between events) and meaningless for ODE
(deterministic). It is supported by tau-leap (NegBinomial replaces Poisson
draws) and chain-binomial (same). The runtime enforces this: requesting
`--backend gillespie` for a model with `overdispersed` transitions produces a
hard error with a hint to use `--backend tau_leap`.

**Composability.** Each transition independently chooses whether to be
overdispersed, and with what variance:

```
infection : S --> I  @ overdispersed(beta * S * I / N, sigma_inf)
recovery  : I --> R  @ overdispersed(gamma * I, sigma_rec)
waning    : R --> S  @ omega * R   # no extra noise
```

---

## 10. Coupling Sugar (Shorthand for Stratified Transmission)

> **Status: not yet implemented.** The coupling sugar described in this section
> is a design specification for a planned feature. The compiler does not
> currently support `coupling[dim = M]` syntax. Use the fully explicit indexed
> transition form instead (see §9).

### 10.1 Why Coupling Sugar Exists

Writing the full indexed transmission formula is the primitive — it's always
correct and always available. But for models with multiple stratification
dimensions, the formula gets long:

```
# Primitive: fully explicit age × sex structured transmission
infection[a in age, s in sex] : S[a,s] --> E[a,s]
  @ beta * S[a,s] * sum(b in age, sum(t in sex,
      C_age[a,b] * B_sex[s,t] * I[b,t]
        / sum(c in compartments, c[b,t])
    ))
```

The coupling sugar lets the user write the base (un-stratified) model and
declare how each dimension interacts:

```
# Sugar: base model + coupling declarations
infection : S --> E @ beta * S * I / N {
  coupling[age = C_age]
  coupling[sex = B_sex]
}
```

Both produce the **same IR**. The sugar is pure convenience — the spec documents
exactly what it expands to, and the user can always write the primitive form
instead.

### 10.2 Expansion Rules

The expansion of `coupling[dim = M]` transforms the base transmission rate as
follows. Starting from `@ beta * S * I / N`:

1. The compiler adds index variables for each coupling dimension
2. `S` becomes `S[i]` (localized to the transition's stratum)
3. `I / N` becomes `sum(j in dim, M[i,j] * I[j] / N_j)` where `N_j` is the total
   population in stratum `j`
4. `N_j` is auto-generated: `sum(c in compartments, c[j])` — the compiler always
   knows the total population per stratum without any user-defined binding

Multiple `coupling` lines nest the sums:

```
# coupling[age = C_age], coupling[sex = B_sex] expands to:
infection[a in age, s in sex] : S[a,s] --> E[a,s]
  @ beta * S[a,s] * sum(b in age, sum(t in sex,
      C_age[a,b] * B_sex[s,t] * I[b,t]
        / sum(c in compartments, c[b,t])
    ))
```

The denominator `sum(c in compartments, c[b,t])` is generated automatically. It
equals the total population of stratum `(age=b, sex=t)` across all compartments.
No user-defined `N_local` binding is required — the sugar is fully
self-contained.

### 10.3 What the Matrices Mean

All coupling structures are expressed through the same mechanism — a rate matrix
`M[i,j]` weighting contact between strata i and j:

| Matrix structure  | Effect                                  | Example             |
| ----------------- | --------------------------------------- | ------------------- |
| Dense             | General mixing                          | Age contact matrix  |
| Off-diagonal only | Directed (no within-group transmission) | STI sex-structured  |
| Identity          | Within-stratum only                     | Same as no coupling |
| All ones          | Homogeneous mixing                      | No structure        |

```
tables {
  # Dense: general age mixing
  C_age : age × age = [[12.0, 4.0], [4.0, 8.0]]

  # Off-diagonal: directed STI transmission (female ↔ male only)
  B_sex : sex × sex = [[0.0, beta_mf], [beta_fm, 0.0]]
}
```

There is no separate `directed` or `mixing` keyword — they are all matrices. The
matrix structure determines the coupling semantics. This is the right primitive:
one concept (rate matrix), many structures.

### 10.4 Multi-Strain Models

Multi-strain models are complex enough that coupling sugar is not provided. Use
the primitive indexed transition form instead.

The key structural insight: in a multi-strain compartmental model, **S is a
shared pool** — a susceptible person isn't "susceptible to wild-type," they're
just susceptible. The strain dimension belongs on E, I, R (tracking which strain
you're infected with / recovered from), not on S.

```
compartments { S, E, I, R }

dimensions {
  age    = [child, adult]
  strain = [wt, delta]
}

stratify(by = age)
stratify(by = strain, only = [E, I, R])

tables {
  C_age    : age × age       = [[12.0, 4.0], [4.0, 8.0]]
  # X[w,v] = cross-protection against strain v from recovery from strain w
  # X[wt,wt] = 1.0 (same-strain immunity), X[wt,delta] = 0.3 (partial)
  X_strain : strain × strain = [[1.0, 0.3], [0.3, 1.0]]
}

let N_local[a in age] = S[a] + sum(v in strain, E[a,v] + I[a,v] + R[a,v])

transitions {
  # Infection draws from the shared S pool into a specific strain
  # Cross-immunity reduces susceptibility based on recovered fractions
  infection[a in age, v in strain] : S[a] --> E[a, v]
    @ beta * S[a]
      * sum(b in age, C_age[a,b] * I[b, v] / N_local[b])
      * (1 - sum(w in strain, X_strain[w, v] * R[a, w]) / N_local[a])

  progression[a in age, v in strain] : E[a,v] --> I[a,v]
    @ sigma * E[a,v]

  recovery[a in age, v in strain] : I[a,v] --> R[a,v]
    @ gamma * I[a,v]
}
```

The cross-immunity factor `(1 - sum(w in strain, X[w,v] * R[a,w]) / N_local[a])`
is a population-level mean-field approximation: it reduces the infection rate
for strain `v` based on the fraction of the population recovered from each
strain `w`, weighted by cross-protection `X[w,v]`. When no one has recovered
(all in S), the factor is 1.0 (no reduction). As more people recover from strain
`w`, susceptibility to strain `v` decreases proportionally.

This is the standard approximation for compartmental multi-strain models. Exact
individual-level immunity tracking requires an ABM.

**Negativity guard.** The cross-immunity factor can go negative if
`sum(w, X[w,v] * R[a,w]) > N_local[a]` — possible with large cross-protection
values and high recovery fractions. For well-specified matrices with
`X[w,v] ∈ [0,1]` and proper population fractions this does not occur, but for
safety the rate expression should clamp:
`max(0.0, 1 - sum(w in strain, X_strain[w,v] * R[a,w]) / N_local[a])`.

---

## 11. ODE Block

> **Not yet implemented (v0.2).** The `ode { }` block is parsed but currently
> discarded by the expander. Real-valued compartments (`W : real`) and ODE
> evolution are planned for v0.2.

For real-valued compartments:

```
ode {
  W = xi * I - delta * W      # dW/dt = xi * I - delta * W
}
```

Left side = compartment name, right side = time derivative. Creates a
piecewise-deterministic Markov process (PDMP): stochastic events for integer
compartments, ODE evolution for real compartments between events.

---

## 12. Data

> **Not yet implemented (v0.2).** The `data { }` block is not yet supported by
> the parser or expander. External data loading and observation targets backed
> by data files are planned for v0.2.

External data for features (always loaded) and observation targets.

```
data {
  population = read_csv("data/nga_pop.csv") {
    time      = column("year", unit = 'years, origin = "2000-01-01")
    total     = column("total_pop")
    pop_child = column("pop_0_5")
  }

  cases = read_csv("data/afp_cases.csv") {
    time   = column("epi_week", unit = 'weeks, origin = "2019-01-01")
    values = column("afp_cases")
  }

  # Inline for testing
  test_cases = inline {
    time   = [7, 14, 21, 28, 35, 42]
    values = [0, 3, 12, 45, 102, 89]
  }
}
```

The compiler distinguishes features from observations by usage: if a data column
appears in a rate expression or function, it's a feature (always loaded). If it
appears in an `observations` block, it's an observation target.

Feature data is inlined into the IR as `Interpolated` time functions or tables
at compile time. The IR remains self-contained.

---

## 13. Observations

```
observations {
  weekly_cases : {
    projected  = incidence(infection)
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }

  detection : {
    projected  = prevalence(I)
    every      = 14 'days
    likelihood = bernoulli(p = p_detect)
  }
}
```

Syntax notes: the observation name is followed by `: {` (colon required).
`likelihood` uses `= KIND(...)` (equals sign, function-call form with named
arguments separated by commas).

### 13.1 Projections

```
incidence(transition)                    cumulative flow since last observation
incidence(transition[stratum])           stratum-specific flow (positional)
incidence(transition[patch = p])         named indexing (sums over other dims)
prevalence(compartment)                  current population
prevalence(compartment[age = child])     named index on compartment
```

Named indexing (§5.1) is particularly useful here because transitions and
compartments may have multiple dimensions. `incidence(infection[patch = p])`
sums over age for a specific patch — without named indexing, positional
`infection[p]` would incorrectly index the first dimension (age).

Inside a likelihood expression, the keyword `projected` refers to the evaluated
projection value for that observation.

### 13.2 Likelihood Families

```
neg_binomial(mean = EXPR, r = EXPR)            overdispersed counts
poisson(rate = EXPR)                           Poisson counts
normal(mean = EXPR, sd = EXPR)                 continuous
binomial(n = EXPR, p = EXPR)                   bounded counts
beta_binomial(n = EXPR, alpha = EXPR, beta = EXPR)
bernoulli(p = EXPR)                            binary outcome
```

### 13.3 Indexed Observations

```
observations {
  cases_by_patch[p in patch] : {
    projected  = incidence(infection[patch = p])
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }
}
```

Generates one observation stream per patch.

### 13.4 Sampling vs Scoring

> **v0.1 status**: the `observations {}` block is compiled and included in the
> IR. In the current Rust backend, the projection and likelihood are not yet
> evaluated at runtime and no observations output file is written. This is
> scheduled for v0.2 (inference support). The syntax is final.

In forward simulation (v0.2+): runtime evaluates projection, **samples** from
likelihood → synthetic data. In inference (v0.2+): runtime **scores** observed
data against likelihood → log p(y|θ).

Monthly incidence can be obtained natively by setting `every = 30 'days` (or
`every = 1 'months` once time-unit arithmetic is implemented).

---

## 14. Interventions

Deterministic state modifications at scheduled times. **Inactive by default.**
Enabled via scenarios or CLI.

```
interventions {
  sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180, 545]

  routine_vacc : transfer(fraction = vacc_rate, from = S, to = V) {
    every = 30 'days
    from  = 0 'days
    until = 2 'years
  }

  importation_pulse : set(I[child, p1], value = I[child, p1] + 10) at [90]
}
```

### 14.1 Actions

```
transfer(fraction = EXPR, from = COMP, to = COMP)   # move fraction
transfer(count = EXPR, from = COMP, to = COMP)       # move count
set(COMP, value = EXPR)                               # override value
```

`transfer` is atomic: `delta = floor(source * fraction)` computed from
pre-intervention state, then `source -= delta, dest += delta` applied together.

**Stratified compartments in actions.** `transfer(from = S, to = V)` with bare
compartment names expands over all strata (see §26.10). `set` with a bare
compartment name on a stratified compartment is a **compile error** — the
compiler cannot guess what value to assign to each stratum. Use explicit
indexing: `set(I[child, p1], value = ...)`. Named indexing is supported:
`set(I[age = child, patch = p1], value = ...)`.

### 14.2 Scheduling

**Inline `at` form** (specific times, most common):

```
NAME : ACTION at [TIME, ...]     # times in model time_unit
```

**Block form** (recurring or complex schedules):

```
NAME : ACTION {
  every = DURATION             recurring interval
  from  = DURATION             start of recurring (default: t_start)
  until = DURATION             end of recurring (default: t_end)
}
```

### 14.3 Indexed Interventions

An intervention can be declared with an **index binder**, creating a **family**
of interventions — one per stratum — in a single line:

```
interventions {
  # Declares sia_north, sia_south, sia_east (one per patch)
  sia[p in patch] : transfer(fraction = vacc_eff * sia_cov, from = S[p], to = V[p]) at [180, 545]
}
```

Syntax: `NAME[INDEX_VAR in DIMENSION] : ACTION at [TIME, ...]` (or a `{ ... }`
block for recurring schedules)

The expanded members share a **`base_name`** (the unindexed name, `"sia"`
above). In scenario `enable`/`disable` lists, passing `"sia"` resolves to all
members whose `base_name` is `"sia"` — no need to enumerate them individually
(see §18).

Individual members can still be addressed by their expanded name (`"sia_north"`)
when fine-grained control is needed.

**Per-patch timing from a table.** When SIA rounds happen on different dates per
patch, store the schedule in a table and reference it in the `at` list:

```
tables {
  sia_day : patch × round = read("data/sia_schedule.tsv")
}

interventions {
  sia[p in patch] : transfer(fraction = vacc_eff * sia_cov, from = S[p], to = V[p])
    at [sia_day[p, 0], sia_day[p, 1]]
}
```

The index variable `p` is in scope inside the schedule block, so `sia_day[p, 0]`
resolves to the correct row at compile time — each expanded intervention gets
its own concrete timestamp.

### 14.4 Activation

Interventions are off by default. Enable via scenarios or CLI:

```bash
camdl simulate model.camdl --enable sia_round_1 --seed 42
```

---

## 14.5 Events

Events are always-active scheduled state modifications. They share the
same action grammar and scheduling as interventions but fire
unconditionally — they cannot be disabled via scenarios.

Use events for structural demographic processes (cohort entry, seasonal
migration, importation seeding). Use interventions for policy choices
(SIA campaigns, school closures).

```
events {
  cohort_entry : add(S, cohort * birthrate(t) * pop(t))
    every 365.25 'days at_day 251

  importation : add(I, 10) at [30]
}
```

Events support the same features as interventions: indexed events,
`where` guards, recurring schedules, and all action types.

### 14.6 The `add` Action

```
add(COMPARTMENT, EXPR)
```

Adds `round(EXPR)` individuals to COMPARTMENT. Accepts negative values
(outflow). If the result makes the compartment negative, a warning is
emitted but the simulation continues — in a particle filter, the
particle gets a bad trajectory and is resampled away.

### 14.7 The `at_day` Schedule

For events and interventions that recur on a specific day within each
period:

```
NAME : ACTION every PERIOD at_day DAY
```

`at_day` is the absolute phase within the period, measured from `t = 0`.
Fire times are `at_day + k * period` for the smallest `k` where
`target >= t_start`. The engine fires on the single timestep where
`|t - target| < 0.5 * dt`, guaranteeing exactly one fire per period
regardless of `dt` or fractional-period drift.

Example: `every 365.25 'days at_day 251` fires on day 251 of each year.
If simulation starts at `t = 100`, the first fire is at `t = 251` (not
`t = 351`).

This replaces manual `mod(t, period)` arithmetic, which silently
double-fires when `dt` does not evenly divide the period.

---

## 14.8 Balance Constraint

Forces one compartment to satisfy a population conservation constraint
at every substep. After all transitions, clamps, events, and
interventions apply, the target compartment is overwritten:

```
balance {
  R = pop(t) - S - E - I
}
```

This matches pomp's `R = nearbyint(pop) - S - E - I` pattern for models
where the population trajectory is externally specified and the birth/death
rates don't exactly reproduce it. The balance target is excluded from
the non-negativity clamp — a negative value signals a broken model.

Events that inject people without a source (e.g., `add(S, 20000)`) will
increase the compartment total. The balance compartment absorbs this by
decreasing to maintain the constraint.

---

## 15. Timepoints and Reserved Identifiers

> **Partially implemented.** The `timepoints { }` block is parsed but the
> declared timepoint values are currently discarded by the expander and not
> available in expressions. Full timepoint support is planned for v0.2. The
> built-in reserved identifiers `t_start` and `t_end` are always available
> regardless.

```
timepoints {
  midpoint     = 1 'year
  intervention = 180 'days
}
```

### 15.1 Built-in Timepoints

`t_start` and `t_end` are **reserved identifiers** automatically defined from
the `simulate` block. They are always available in summary expressions:

```
at(R, t_end)        # value of R at simulation end
at(I, midpoint)     # value of I at declared midpoint
at(N, t_start)      # value of N at simulation start
```

If `simulate` is absent (e.g., during `camdl check`), `t_start` and `t_end` are
undefined. Expressions referencing them produce a compile warning: "t_end
referenced but no simulate block present."

### 15.2 Reserved Identifiers

The following names cannot be used as parameter, compartment, table, let
binding, or index dimension names:

```
t_start          # simulation start time (from simulate block)
t_end            # simulation end time (from simulate block)
compartments     # the set of integer compartment names (for iteration)
sum              # summation keyword
consecutive      # pair iteration keyword
```

The compiler errors if a user declaration shadows a reserved name:

```
ERROR: 't_end' is a reserved identifier and cannot be used as a
  parameter name.
```

---

## 16. Initial Conditions

### 16.1 Un-Stratified Models

```
init {
  S = N0 - I0
  I = I0
}
```

Unlisted compartments default to 0. Expressions can reference parameters.

### 16.2 Stratified Models

When compartments have index dimensions, **bare names are a compile error.** The
compiler cannot guess how to distribute a total across strata.

```
# ERROR: S has dimensions [age, patch], must specify strata
init {
  S = N0 - I0
}

# CORRECT: explicit per-stratum values
init {
  S[child, p1] = 100000
  S[adult, p1] = 200000
  I[child, p1] = I0
}
```

**Indexed parameter references** work in init RHS expressions. If `N0[patch]`
is an indexed parameter, both the mangled form and the indexed form are accepted:

```
init {
  S[urban] = N0[urban] - I0   # indexed syntax — preferred
  S[rural] = N0_rural         # mangled form — still accepted
}
```

Named indexing works in init:

```
init {
  S[age = child, patch = p1] = 100000
}
```

Unlisted stratum combinations default to 0. For a 774-patch model, only the
patches mentioned in init are nonzero — the rest start empty. This is common for
initialization from a single-patch seeding event.

### 16.3 Init from Tables

For large spatial models where per-stratum populations come from a CSV, declare
a table (§6) and reference it directly in init expressions:

```
tables {
  N0 : patch = read("data/population.tsv")
}

parameters {
  I0 : count in [1, 1000]
}

init {
  S[p in patch] = N0[p] - I0
  I[p in patch] = I0
}
```

The index binder `[p in patch]` generates one init entry per patch. `N0[p]`
performs a table lookup at compile time — each expanded entry gets its own
concrete value. Parameter references (e.g. `I0`) remain as IR-level expressions
evaluated at runtime.

This is fully supported in v0.1. The `distribute(total, weights = table)`
aggregate helper for proportional allocation is future sugar (v0.2).

---

## 17. Output

Each sub-block produces a **separate file** with its own schema. Different
output types have different shapes — they are never kludged into one TSV.

```
output {
  trajectories {
    every = 1 'day
    quantities {
      total_I    = I
      prevalence = I / N
    }
    format = parquet
  }

  flows {
    every = 7 'days
    quantities {
      weekly_infections = incidence(infection)
    }
    format = parquet
  }

  summary {
    peak_I      = max(I)
    total_cases = cumulative(infection)
    final_size  = at(R, t_end) / at(N, t_end)
    extinct     = at(I, t_end) == 0
    format      = tsv
  }

  synthetic {    # synthetic observations (forward sim only)
    format = tsv
  }
}
```

### 17.1 Output Files

```
trajectories.parquet  # time × named quantities (one row per output time)
flows.parquet         # time × named flow quantities
summary.tsv           # one row per run, scalar columns
synthetic.tsv         # time × stream × projected × observed
metadata.json         # run provenance (see §20)
```

### 17.2 IR Mapping

**Trajectories and flows** are IR-level outputs. The IR `output` section
specifies two schedules: one for state snapshots (trajectories) and one for flow
counts (flows). The runtime writes both directly during simulation. Named
quantities in `trajectories { quantities { ... } }` are compiled to IR
expressions evaluated at each output time. Named flow quantities reference
`CumulativeFlow` projections in the IR.

**Summary** is computed **post-simulation** by the CLI from trajectory and flow
data. The runtime does not compute summaries — it writes the trajectory, and the
CLI reads it back to evaluate summary expressions. For large spatial models, the
CLI processes the trajectory in streaming fashion to avoid loading the full
parquet into memory.

**Synthetic observations** are generated by the runtime's `sample_observations`
method using the observation model definitions.

### 17.3 Summary Functions

```
max(expr)                  maximum value of expr over all output times
min(expr)                  minimum value
cumulative(transition)     total firings over entire simulation
at(expr, timepoint)        value of expr at specific time
time_when(pred)            first time predicate becomes true
```

---

## 18. Scenarios

Patch-based modifications to the baseline. Baseline is the identity patch — the
model as defined, no modifications.

```
scenarios {
  baseline {
    label = "no SIA (baseline)"
  }

  with_sia {
    label  = "with SIA — all patches"
    enable = [sia]
  }

  high_coverage {
    enable = [sia]
    set    = { sia_cov = 0.95 }
  }

  more_transmissible {
    scale = { beta = 1.5 }
  }

  combined {
    compose = [with_sia, more_transmissible]
  }
}
```

### 18.1 Patch Operations

```
label   = STRING                   human-readable name for the scenario
enable  = [INTERVENTION, ...]      turn on interventions
disable = [INTERVENTION, ...]      turn off interventions
set     = { PARAM = EXPR, ... }    override parameter values
scale   = { PARAM = FACTOR, ... }  multiply (compiler checks domain validity)
compose = [SCENARIO, ...]          apply patches in sequence
```

**Indexed parameter syntax in set/scale.** For indexed parameters declared as
`N0[patch]`, the `set` and `scale` blocks accept either the mangled name or the
indexed form:

```
set = {
  N0[urban] = 100000    # indexed — preferred, mirrors declaration syntax
  N0_rural  = 50000     # mangled — still accepted
}
```

The compiler mangles `N0[urban]` to `N0_urban` in the IR. Multi-dimensional
indices are supported: `amp[urban, child]` mangles to `amp_urban_child`.

**Family-based enable resolution.** `enable` entries are matched against
intervention `base_name` as well as exact names. If `"sia"` is the `base_name`
of an indexed family `sia[p in patch]`, writing `enable = [sia]` activates all
238 members at once. Individual members can still be addressed by their expanded
name (e.g., `"sia_borno_damboa"`) when fine-grained control is needed.

The compiler warns on non-commutative compositions (overlapping write sets).
`scale` on a `probability` parameter that would exceed [0,1] is a **compile
error** — the user must handle clamping explicitly via `set` with an
`if/then/else` expression. No implicit clamping.

### 18.2 Scenario Inheritance — `extends`

A scenario can inherit from another via `extends = <parent_name>`, which is
**compile-time sugar**: the child is resolved as the parent with the child's
fields layered on top. The IR keeps its flat preset shape — downstream
consumers see no trace of inheritance.

```
scenarios {
  baseline {
    label = "No intervention"
    set = { beta = 0.4, gamma = 0.15, N0 = 10_000, I0 = 10 }
  }
  with_vacc {
    extends = baseline
    label   = "Vaccination at day 15"
    enable  = [vaccination]
  }
  warmer {
    extends = baseline
    set     = { beta = beta * 1.5 }    # references parent's resolved value
  }
}
```

**Merge rules (per field):**

| Field                     | Behavior                                                   |
|---------------------------|-------------------------------------------------------------|
| `set`, `scale`            | Child keys override parent keys on collision; union otherwise. |
| `enable`, `disable`, `compose` | Parent + child, deduped preserving first-seen order.  |
| `label`, `simulate.to`    | Child overrides parent when present.                       |

**Expression scope.** Child `set` expressions are evaluated *after* parent's
`set` is resolved — so `set = { beta = beta * 1.5 }` in a child reads the
parent's resolved `beta`. There's no default-at-declaration path in camdl;
the name must resolve to a concrete upstream value or the compiler errors.

::: {.callout-warning}
**`enable`/`disable`/`compose` append parent's list to the child's.** A
child writing `enable = [masking]` under a parent with `enable = [vaccination]`
gets `[vaccination, masking]`, **not** just `[masking]`. To remove a parent's
intervention in a child, use `disable`. The compiler emits **W310** whenever
this merge actually changes the child's declared list, so the surprise is
observable rather than silent.
:::

**Diagnostics:**

- **E25x** — cycle in `extends` chain (includes the full chain in the message).
- **E25y** — unknown parent scenario (suggests the closest name by edit distance).
- **E25z** — chain depth > 5; treat as a code smell and factor common ancestors,
  or request multi-parent composition as a future feature.
- **W310** — append-dedup of parent's enable/disable/compose changed the
  resolved list (see callout above).

### 18.3 Scenario Expression Scope

Inside `set = { PARAM = EXPR }`, the RHS expression can reference:

- The parameter's **current value** (its name refers to the pre-patch value)
- Other parameters (their pre-patch values)
- Literal constants

Compartment state, time, and other scenario settings are NOT in scope — scenario
patches are static transformations of parameter values, not runtime-dependent
operations.

### 18.4 External Experiment Files

For multi-model analysis, a separate experiment file:

```
experiment("Nigeria SIA evaluation") {
  model  = "models/seir_nigeria.camdl"
  params = "params/fitted_2024.toml"

  scenarios {
    with_sia {
      enable = [sia_round_1]
    }
    delayed_sia {
      enable = [sia_round_1]
      set    = { sia_time = 365 'days }
    }
  }

  compare {
    pairs = [
      (baseline, with_sia),
      (baseline, delayed_sia)
    ]
    seeds = 1 to 1000    # range syntax: generates integers 1, 2, ..., 1000
  }

  # Cross-scenario derived quantities
  output {
    summary {
      cases_averted      = baseline.total_cases - scenario.total_cases
      relative_reduction = cases_averted / baseline.total_cases
    }
  }
}
```

### 18.5 Compare Block Semantics

The `compare` block drives paired scenario simulation with matched
seeds:

- `pairs` lists 2-tuples of `(reference_scenario, test_scenario)`. The keyword
  `baseline` refers to the identity patch (no scenario modifications).
- `seeds = N to M` is range syntax generating integers N, N+1, ..., M.
- For each pair and each seed, both scenarios are simulated with the same seed.
  Because the runtime uses a stateful PRNG, pre-divergence coupling holds only
  when both runs consume the RNG in the same order.

Inside the experiment's `output.summary`, two special names are available:

- `baseline.QUANTITY` — summary value from the reference scenario
- `scenario.QUANTITY` — summary value from the test scenario

These are only valid inside experiment `compare` output blocks. `QUANTITY` must
match a name declared in the model's `output { summary { ... } }` block (e.g.,
`total_cases`, `peak_I`).

---

## 19. Simulation Configuration

```
simulate {
  from = 0 'days
  to   = 2 'years
}
```

Seed is always external (CLI `--seed`), never in the model file.

---

## 20. Content-Addressable Output

Outputs are stored in a two-level content-addressable hierarchy inside an
`output_dir` you choose:

```
{output_dir}/
  manifest.json
  model.ir.json
  geo/boundaries.geojson          (if geo= specified in experiment)
  runs/
    {sim_hash_8}/                 # model + base params + backend + dt
      {scenario_slug}-{scen_hash_8}/   # scenario overrides
        seed_{seed}/              # individual run
          traj.tsv
          run.json
```

Example with two scenarios after a base-param change:

```
runs/3a7f2c1d/baseline-00000000/seed_1/
runs/3a7f2c1d/with_sia-f9e2b047/seed_1/
# after tweaking with_sia only — baseline reused:
runs/3a7f2c1d/with_sia-d4e2a391/seed_1/
runs/3a7f2c1d/baseline-00000000/seed_1/   ← untouched, cached
# after changing base params — nothing reused:
runs/cc8b1a90/baseline-00000000/seed_1/
```

### 20.1 Hash Computation

```
model_hash = sha256(IR JSON bytes)                         # full 64-char hex

sim_hash   = sha256(model_hash + canonical_base_params
                    + backend + dt + tool_version)         # 64-char hex
                                                           # first 8 used in dir name

scen_hash  = sha256(sorted(enable) + sorted(disable)
                    + canonical_scen_params)               # 64-char hex
                                                           # first 8 used in dir name

scenario_slug = scenario name lowercased,
                non-[a-z0-9_] replaced with _
seed_dir      = seed_{N}   (verbatim u64, no zero-padding)
```

A scenario with no overrides, enables, or disables always produces
`scen_hash = sha256("")` → `00000000` prefix, visually identifying it as the
unmodified baseline.

`scen_hash` covers only the _delta_ (scenario overrides, enable, disable). Base
params and model structure are captured in `sim_hash`. Renaming a scenario
without changing its definition preserves the hash, so cached runs are reused.

**Structural content** included in `model_hash` (via IR JSON):

```
compartments         # state variable declarations
stratify             # index dimension declarations
parameters           # names and types only (not values)
tables               # dimension annotations + inline values
let                  # all let bindings
transitions          # all transition declarations
interventions        # intervention definitions (including base_name)
init                 # initial condition expressions
time_unit            # canonical time unit
```

**Excluded** from `model_hash`:

```
simulate             # time range is analysis-specific
scenarios            # counterfactual modifications, not structural
data (file paths)    # external file paths change across machines
```

### 20.2 Cache Reuse Matrix

| What changed                       | sim_hash  | scen_hash         | Reuse               |
| ---------------------------------- | --------- | ----------------- | ------------------- |
| IR / model                         | changes   | —                 | none                |
| base params                        | changes   | —                 | none                |
| backend or dt                      | changes   | —                 | none                |
| scenario A's enable/disable/params | unchanged | A changes, B same | B's runs reused     |
| add more seeds                     | unchanged | unchanged         | all existing reused |
| rename a scenario                  | unchanged | unchanged         | reused (same sim)   |

### 20.3 Manifest

`manifest.json` at the output root lists every completed run:

```json
{
  "runs": [
    {
      "scenario": "baseline",
      "seed": 1,
      "run_path": "3a7f2c1d/baseline-00000000/seed_1"
    },
    {
      "scenario": "with_sia",
      "seed": 1,
      "run_path": "3a7f2c1d/with_sia-f9e2b047/seed_1"
    }
  ]
}
```

The web app constructs trajectory URLs as `GET /runs/{run_path}/traj.tsv`.

### 20.4 Caching

Same inputs → same hashes → run directory already exists → skip simulation. Pass
`--force` to re-run and overwrite existing results.

---

## 21. Parameter Files

### 21.1 Values (v0.1)

```toml
# params.toml
beta = 0.3
gamma = 0.1
sigma = 0.2
mu = 0.0000548
rho = 0.4
k = 5.0
N0 = 1000000
I0 = 10
```

### 21.2 Priors

Priors are declared with `~` syntax directly on parameters in the `.camdl`
file — they are beliefs about parameters and belong with the declaration:

```
parameters {
    beta  : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
    gamma : rate in [0.05, 1.0] ~ half_normal(sigma = 0.3)
    rho   : probability in [0.001, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
    N0    : count in [100, 1_000_000]  # no prior — must be supplied via
                                       # --params, --scenario, or [fixed] in fit.toml
}
```

The `~` reads "distributed as" and is always optional. Parameters without
a prior can still be fixed at a known value via `--params` files.

**Supported distributions**:

| Distribution    | Syntax                                    | Parameters            |
|-----------------|-------------------------------------------|-----------------------|
| `uniform`       | `~ uniform(lower = L, upper = U)`         | bounds                |
| `normal`        | `~ normal(mu = M, sigma = S)`             | mean, sd (natural)    |
| `log_normal`    | `~ log_normal(mu = M, sigma = S)`         | log-scale mu, sigma   |
| `half_normal`   | `~ half_normal(sigma = S)`                | sd of underlying normal|
| `beta`          | `~ beta(alpha = A, beta = B)`             | shape parameters      |
| `gamma`         | `~ gamma(shape = K, rate = R)`            | shape, rate (NOT scale) |
| `exponential`   | `~ exponential(rate = R)`                 | rate = 1/mean         |

All arguments are keyword (named), never positional. All arguments must be
compile-time constants.

**Parameterization conventions** (these are load-bearing):

- `log_normal(mu, sigma)`: parameters are on the **log scale**.
  `log(X) ~ Normal(mu, sigma)`. Median of X is `exp(mu)`.
- `half_normal(sigma)`: sigma is the SD of the underlying (unfolded) normal.
- `gamma(shape, rate)`: rate parameterization (`E[X] = shape/rate`).

Priors in the model are the primary source; `fit.toml [estimate]` priors
override them for sensitivity analysis. See the run spec §12 for the full
precedence chain.

### 21.3 Views (v0.2+)

```toml
# view.toml — implements V from the parameter grammar
[view]
free = ["beta", "gamma", "rho", "I0"]
```

Free parameters are varied by the inference engine; all other parameters are
held fixed at their values from `params.toml`. Views are only relevant for
`camdl fit` (v0.2+) — they have no effect on forward simulation.

### 21.4 Relationship to the Parameter Grammar

The parameter grammar (Buffalo 2026) defines the formal framework for
partitioning and manipulating model inputs. camdl implements each concept:

| Grammar concept          | camdl implementation                        |
| ------------------------ | ------------------------------------------- |
| **M** (parameter space)  | `parameters { }` block — all tuneable knobs |
| **C** (configuration)    | Model structure + `simulate` + `output`     |
| **S** (seed)             | CLI `--seed`, never in model file           |
| **Point m ∈ M**          | `params.toml`                               |
| **Scenario σ**           | `scenarios { }` — patch operations          |
| **Baseline σ₀**          | Identity patch — model as defined           |
| **View V**               | `view.toml` — free vs fixed                 |
| **Prior π(m)**           | `~` syntax on parameter declaration         |
| **Transform T_V**        | Per-parameter `transform` (v0.2+)           |
| **Reparameterization R** | Future: `reparam.toml`                      |
| **Sim(m, c, s) → y**     | `camdl simulate`                            |
| **Sim_σ,V,T(z, s) → y**  | `camdl fit` (v0.2+)                         |

The downward chain from inference coordinates to simulation output:

```
z ∈ Z_V     inference engine proposes a vector
  │ T_V⁻¹   back-transform (exp, expit)
  ▼
p ∈ P_V     free parameter values
  │ κ_V     fill in fixed values
  ▼
m ∈ M       complete parameter set
  │ σ       apply scenario patch
  ▼
(m', c')    patched parameters + configuration
  │ Sim
  ▼
y ∈ Y       trajectory, observations
```

Every arrow is defined by external configuration. The `.camdl` file defines the
structural skeleton; the parameter grammar fills in the rest.

---

## 22. CLI

The unified `camdl` command routes to two backends: **`camdlc`** (OCaml
compiler) for compilation/inspection and **`camdl-sim`** (Rust) for simulation,
experiments, and inference. All commands accept `.camdl` files directly
(auto-compiled via `camdlc`).

### 22.1 Compilation and Inspection

```bash
camdl compile MODEL.camdl              # compile to IR JSON (stdout)
camdl check   MODEL.camdl              # validate structure (no output)
camdl inspect MODEL.camdl [OPTIONS]    # inspect compartments, transitions, etc.
```

### 22.2 Simulation

```bash
camdl simulate MODEL --params P.toml --seed 42 [OPTIONS]

Options:
  --backend    gillespie|tau_leap|chain_binomial|ode  (default: chain_binomial)
  --dt         DT         step size for tau_leap / chain_binomial / ode
  --seed       N          RNG seed (default: 1)
  --scenario   NAME       select a named scenario
  --enable     NAME       enable an intervention (ad-hoc; mutually exclusive with --scenario)
  --disable    NAME       disable an intervention
  --param      NAME=VALUE override a parameter value
  --param-vec  PREFIX=FILE override indexed params from a keyed TSV
  --table      NAME=FILE  supply a runtime external() table
  --params     FILE.toml  load parameter values (repeatable, later overrides earlier)
```

Output is TSV to stdout: `t`, one column per compartment, `flow_<name>` per
transition.

### 22.3 Expression Evaluation

```bash
camdl eval MODEL --params P.toml --expr "school,seas,R0" --from 0 --to 365 --every 1
camdl eval MODEL --params P.toml --expr "school" --at 0,100,200,300
```

Evaluates time-dependent expressions (forcing functions, parameters, math
expressions) at a time grid without simulation. Expressions that reference
compartment state produce an error.

### 22.4 Batch simulation

```bash
camdl simulate batch    BATCH.toml [--parallel N] [--force] [--dry-run]
camdl simulate status   BATCH.toml
camdl list              [RESULTS_DIR]          # browse cached runs
camdl show  <short-hash>
camdl cat   <short-hash> [--stream NAME]
```

Batch parameter sweeps, scenario comparisons, and posterior-predictive
checks. Sensitivity analysis (Sobol indices) is out of scope for the
CLI; compute it with R's `sensitivity` package or Python's `SALib` on
the batch output. See the Run Specification (`camdl-run-spec.md` §5)
for details.

### 22.5 Inference

```bash
# Particle filter — log-likelihood estimation
camdl pfilter MODEL --params P.toml --data cases.tsv \
    --particles 5000 --dt 1 --seed 42 \
    --flow recovery --obs-model discretized_normal --tol 1e-18 \
    --trace

# Iterated filtering — explicit rw_sd (the list IS the partition)
camdl if2 MODEL --params P.toml --data cases.tsv \
    --rw-sd "R0=5,sigma=0.01,gamma=0.01" \
    --particles 2000 --iterations 100 --cooling 0.95 \
    --chains 4 --regime scout --ivp "S0,I0" \
    --flow recovery --obs-model discretized_normal

# Auto rw_sd from parameter bounds (--fixed excludes non-estimable params)
camdl if2 MODEL --params P.toml --data cases.tsv \
    --rw-sd auto --fixed "N0,mu,k" \
    --regime scout --flow recovery

# Profile likelihood — parameter identifiability
camdl profile MODEL --params P.toml --data cases.tsv \
    --focal R0 --grid "10,20,30,40,50,60,70" \
    --rw-sd "sigma=0.01,gamma=0.01" \
    --particles 500 --iterations 30 --starts 3 --parallel 8

# 2D profile
camdl profile MODEL --focal alpha,gamma \
    --grid-alpha "0.85,0.90,0.95" --grid-gamma "0.06,0.08,0.10" ...
```

**`--flow NAME`**: Which transition's cumulative flow to project for the
observation model. Must match what the data measures (e.g., `recovery` for
case notifications that count recoveries, not infections).

**`--obs-model`**: `negbin` (default) or `discretized_normal` (He et al.
heteroscedastic variance).

**`--rw-sd`**: Perturbation scale per parameter. Three modes:
- Explicit: `--rw-sd "R0=5,sigma=0.01"` — the list IS the partition.
  Parameters not listed are held fixed. No `--fixed` needed.
- Auto: `--rw-sd auto` — heuristic from parameter bounds (`(hi-lo)/6`
  on transformed scale). Use `--fixed "N0,mu,k"` to exclude params.
- Mixed: `--rw-sd "R0=5,sigma=auto"` — explicit where you know, auto
  where you don't.

**`--regime`**: IF2 presets — `scout` (8 chains, 500 particles,
cooling=0.5), `refine` (4 chains, 1000 particles, cooling=0.95),
`validate` (4 chains, 5000 particles, cooling=0.95).

**`--ivp "S0,I0"`**: Initial value parameters, perturbed only at t=0.

**`--fixed "N0,mu"`**: Exclude parameters from estimation (with
`--rw-sd auto`). Also available on `camdl profile`.

### 22.6 Fit Workflow

```bash
camdl fit run    fit.toml [--stage NAME] [--seed N] [--force] [--sweep "PARAM=V1,V2,..."]
camdl fit status fit.toml
```

Driven by `fit.toml` with `[estimate]`, `[fixed]`, `[data]`, and
one or more `[stages.NAME]` blocks. Stages are named by the user
(by convention `scout`, `refine`, `validate`) and chain via each
stage's optional `starts_from` field. `--stage NAME` runs a single
stage; `--sweep` takes a Cartesian product over parameter grids
and, when a cell fails the convergence gate, records the failure
in `sweep_failures.tsv` and continues rather than halting. See
`docs/camdl-inference-spec.md`.

**Pfilter replicates:**

```bash
camdl pfilter MODEL --params P.toml --data d.tsv \
    --replicates 100 --output logliks.tsv
```

Runs N independent particle filters, outputs `seed\tloglik` TSV.

**Pfilter trace** (observation-space predictions):

```bash
camdl pfilter MODEL --params P.toml --data d.tsv --trace diag.tsv
```

Columns: `time ll_increment ESS obs_mean obs_q05 obs_q50 obs_q95
state_mean state_q05 state_q50 state_q95 observed`. `obs_*` includes
observation noise; `state_*` is process uncertainty only.

### 22.7 Data Utilities

```bash
# Split a TSV at a time threshold for train/holdout validation
camdl data split data/cases.tsv --at-time 5474
# → data/cases_train.tsv, data/cases_holdout.tsv

# Explicit output paths
camdl data split data/cases.tsv --at-time 5474 \
    --train data/train.tsv --holdout data/holdout.tsv
```

### 22.8 Particle State Export

```bash
camdl pfilter MODEL --params P.toml --data train.tsv \
    --particles 5000 --save-final-state final_particles.tsv
```

### 22.9 Value of Information

```bash
camdl voi run voi.toml
```

### 22.10 Web Server

```bash
camdl serve [--port 8080] [DIR]
```

Serves experiment output directories over HTTP for the web editor.

---

## 23. Worked Examples

These examples progress from trivial to complex, showing how primitives compose.
Each shows the DSL source and key points about what the compiler generates.

### 23.1 Bare SIR (Simplest Possible Model)

```
time_unit = 'days

compartments { S, I, R }
let N = S + I + R

parameters {
  beta  : rate
  gamma : rate
  N0    : count
  I0    : count
}

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

simulate {
  from = 0 'days
  to   = 120 'days
}
```

10 lines of model structure. No stratification, no demography, no observations.
The compiler generates 2 IR transitions with flat rate expressions. This is the
minimal golden test model.

### 23.2 SIR with Demography (Explicit Transitions)

```
time_unit = 'days

compartments { S, I, R }
let N = S + I + R

parameters {
  beta  : rate
  gamma : rate
  mu    : rate
  N0    : count
  I0    : count
}

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I

  # Demography: explicit, no sugar
  birth   : --> S      @ mu * N
  death_S : S -->      @ mu * S
  death_I : I -->      @ mu * I
  death_R : R -->      @ mu * R
}

init {
  S = N0 - I0
  I = I0
}

simulate {
  from = 0 'days
  to   = 5 'years
}
```

6 transitions total. Every rate is a total propensity — `death_S` rate is
`mu * S` (per-capita rate times population count, explicit). Birth is an inflow
at rate `mu * N` (population-dependent, balances deaths in expectation).

### 23.3 SEIR with Age Mixing (Introducing Stratification)

Two versions shown: the **primitive** form (explicit indexed transitions) and
the **coupling sugar** form. Both produce identical IR.

**Primitive form:**

```
time_unit = 'days

compartments { S, E, I, R }

dimensions { age = [child, adult] }
stratify(by = age)

let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

parameters {
  beta   : rate
  sigma  : rate
  gamma  : rate
}

tables {
  C_age : age × age = [[12.0, 4.0], [4.0, 8.0]]
}

transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, C_age[a,b] * I[b] / N_local[b])

  progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
  recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
}
```

**Coupling sugar form** (not yet implemented — identical IR output when available):

```
time_unit = 'days

compartments { S, E, I, R }
let N = S + E + I + R

dimensions { age = [child, adult] }
stratify(by = age)

parameters {
  beta   : rate
  sigma  : rate
  gamma  : rate
}

tables {
  C_age : age × age = [[12.0, 4.0], [4.0, 8.0]]
}

transitions {
  infection : S --> E @ beta * S * I / N {
    coupling[age = C_age]
  }
  progression : E --> I  @ sigma * E
  recovery    : I --> R  @ gamma * I
}
```

The sugar version has no index variables, no `sum`, no `N_local`. The
`coupling[age = C_age]` declaration tells the compiler to transform `S * I / N`
into the per-stratum formula with contact-matrix-weighted summation. Progression
and recovery are automatically replicated within each stratum (default behavior
when no coupling is declared).

### 23.4 STI with Directed Transmission (Off-Diagonal Matrix)

```
time_unit = 'days

compartments { S, I, R }

dimensions { sex = [female, male] }
stratify(by = sex)

let N_local[s in sex] = S[s] + I[s] + R[s]

parameters {
  beta_mf : rate     # male-to-female transmission
  beta_fm : rate     # female-to-male transmission
  gamma   : rate
}

tables {
  # Off-diagonal: females are only infected BY males and vice versa
  B_sex : sex × sex = [[0.0,     beta_mf],
                        [beta_fm, 0.0    ]]
}

transitions {
  infection[s in sex] : S[s] --> I[s]
    @ S[s] * sum(t in sex, B_sex[s,t] * I[t] / N_local[t])

  recovery[s in sex] : I[s] --> R[s]  @ gamma * I[s]
}
```

The zero diagonal in `B_sex` means no within-sex transmission. The
`sum(t in sex, ...)` sums over both sexes, but the zero entries eliminate
same-sex terms. `infection_female` rate becomes
`S[female] * beta_mf * I[male] / N_local[male]`. No special `directed` keyword
needed — the matrix structure does all the work.

### 23.5 Cholera with Environmental Reservoir (Real Compartment + ODE) _(planned v0.2)_

```
time_unit = 'days

compartments {
  S, I, R
  W : real             # bacteria concentration in water
}

let N = S + I + R

parameters {
  beta_W : positive    # waterborne transmission coefficient
  beta_I : rate        # person-to-person transmission rate
  gamma  : rate
  xi     : positive    # shedding rate
  delta  : rate        # environmental decay rate
  K      : positive    # half-saturation constant
}

transitions {
  infection : S --> I
    @ S * (beta_W * W / (K + W) + beta_I * I / N)

  recovery  : I --> R  @ gamma * I
}

ode {
  W = xi * I - delta * W
}
```

`W : real` is continuous-valued — not a population count. The `ode` block gives
`dW/dt`. Between stochastic events (infections, recoveries), W evolves
deterministically. This is a piecewise-deterministic Markov process (PDMP). `W`
appears in the infection rate via the dose-response term `beta_W * W / (K + W)`
— coupling the continuous and discrete dynamics.

Note: `c in compartments` would NOT iterate over `W` (integer compartments only
by default).

### 23.6 Five-Age-Group Model with Consecutive Aging

```
time_unit = 'days

compartments { S, I, R }

dimensions { age = [age_0_5, age_5_15, age_15_50, age_50_65, age_65p] }
stratify(by = age)

parameters {
  beta  : rate
  gamma : rate
  mu    : rate
}

tables {
  C_age   : age × age 'per_day = read("data/polymod_5x5.tsv")
  age_dur : age 'years          = [5, 10, 35, 15, 20]
  mu_age  : age 'per_day        = [0.00008, 0.00002, 0.00003, 0.0001, 0.0005]
}

let N_local[a in age] = S[a] + I[a] + R[a]

transitions {
  infection[a in age] : S[a] --> I[a]
    @ beta * S[a] * sum(b in age, C_age[a,b] * I[b] / N_local[b])

  recovery[a in age] : I[a] --> R[a]
    @ gamma * I[a]

  # Aging: consecutive pairs generate 4 transitions per compartment
  aging[c in compartments, (a, a_next) in consecutive(age)]
    : c[a] --> c[a_next]
    @ (1 / age_dur[a]) * c[a]

  # Death: age-specific, all compartments
  death[c in compartments, a in age] : c[a] -->
    @ mu_age[a] * c[a]

  # Birth: into youngest age group
  birth : --> S[age_0_5]
    @ mu * sum(a in age, N_local[a])
}
```

`consecutive(age)` generates pairs: `(age_0_5, age_5_15)`,
`(age_5_15, age_15_50)`, `(age_15_50, age_50_65)`, `(age_50_65, age_65p)`. With
3 compartments, this produces 3 × 4 = 12 aging transitions. The last age group
(`age_65p`) has no outgoing aging — individuals stay until death.

Total transitions: 5 infections + 5 recoveries + 12 aging + 15 deaths + 1 birth
= 38.

### 23.7 Erlang Sub-Staging (Non-Exponential Waiting Times)

```
time_unit = 'days

compartments { S, E, I, R }

# E passes through 3 sub-stages for Erlang-distributed latent period
dimensions { latent_stage = [e1, e2, e3] }
stratify(by = latent_stage, only = [E])

parameters {
  beta  : rate
  sigma : rate    # mean latent period = 1/sigma (same as exponential)
  gamma : rate
}

transitions {
  infection : S --> E[e1]  @ beta * S * I / (S + E + I + R)

  # Progression through Erlang stages
  latent[(s, s_next) in consecutive(latent_stage)]
    : E[s] --> E[s_next]
    @ 3 * sigma * E[s]          # k * sigma for Erlang-k

  # Final stage exits to I
  onset : E[e3] --> I  @ 3 * sigma * E[e3]

  recovery : I --> R  @ gamma * I
}
```

The Erlang-3 latent period has the same mean (1/sigma) as exponential but
reduced variance (variance = 1/(k·sigma²)). The distribution is more peaked,
closer to real disease progression. Note: `infection` destination is `E[e1]` —
entering the first sub-stage. Partial stratification (`only = [E]`) means S, I,
R don't have the `latent_stage` dimension.

---

## 24. Full Example: Spatial Age-Structured SEIR

```
time_unit = 'days

compartments { S, E, I, R, V }
let N = S + E + I + R + V

## ── Index dimensions ───────────────────────────────────

dimensions {
  age   = [child, adult]
  patch = read("data/lga_pop.tsv", column = "patch")  # levels from data
}

stratify(by = age)
stratify(by = patch)

## ── Parameters ─────────────────────────────────────────

parameters {
  beta       : rate
  sigma      : rate
  gamma      : rate
  mu         : rate
  theta      : positive       # gravity model scale
  alpha      : probability    # seasonal amplitude
  phi_season : real           # seasonal phase (days from t=0 to peak)
  rho        : probability
  k          : positive
  I0         : count
  vacc_frac  : probability
  import_rate : rate
}

## ── Tables ─────────────────────────────────────────────

tables {
  C_age     : age × age          = [[12.0, 4.0], [4.0, 8.0]]
  mu_age    : age 'per_day       = [0.0000685, 0.0000411]
  fertility : age 'per_day       = [0.0, 0.02]
  age_dur   : age 'years         = [5, 60]
  pop       : patch               = read("data/lga_pop.tsv")
  distance  : patch × patch      = read("data/lga_dist.tsv", default = 0.0)
}

## ── Computed quantities ────────────────────────────────

let N_local[a in age, p in patch] = S[a,p] + E[a,p] + I[a,p] + R[a,p] + V[a,p]

let mig[i in patch, j in patch] =
  if i == j then 0.0
  else theta * pop[j] / (distance[i,j] ^ 2)

## ── Functions ──────────────────────────────────────────

forcing {
  seasonal = sinusoidal(
    amplitude = alpha,
    period    = 365.25 'days,
    phase     = phi_season,
    baseline  = 1.0
  )
}

## ── Transitions ────────────────────────────────────────

transitions {
  # Infection: age mixing, spatial coupling via gravity kernel
  infection[a in age, p in patch] : S[a,p] --> E[a,p]
    @ beta * seasonal * S[a,p]
      * sum(b in age, sum(q in patch,
          C_age[a,b] * mig[p,q] * I[b,q] / N_local[b,q]
        ))

  # Progression and recovery
  progression[a in age, p in patch] : E[a,p] --> I[a,p]
    @ sigma * E[a,p]

  recovery[a in age, p in patch] : I[a,p] --> R[a,p]
    @ gamma * I[a,p]

  # Importation: exogenous inflow, distributed across age/patch
  importation[a in age, p in patch] : --> I[a, p]
    @ import_rate * pop[p] / sum(q in patch, pop[q])

  # Death: all integer compartments, age-specific rate
  death[c in compartments, a in age, p in patch] : c[a,p] -->
    @ mu_age[a] * c[a,p]

  # Aging: consecutive pair transfer (child → adult)
  aging[c in compartments, p in patch] : c[child, p] --> c[adult, p]
    @ (1 / age_dur[child]) * c[child, p]

  # Migration: all compartments across patches, no self-loops
  migrate[c in compartments, a in age, src in patch, dst in patch]
    : c[a,src] --> c[a,dst]
    @ mig[dst,src] * c[a,src]
    where src != dst

  # Birth: fertility-weighted by age
  birth[p in patch] : --> S[child, p]
    @ sum(a in age, fertility[a] * N_local[a, p])
}

## ── Interventions ──────────────────────────────────────

interventions {
  sia_round_1 : transfer(fraction = vacc_frac, from = S, to = V) at [180, 545]
}

## ── Data ───────────────────────────────────────────────

data {
  cases = read_csv("data/nigeria_afp.csv") {
    time   = column("epi_week", unit = 'weeks, origin = "2019-01-01")
    values = column("afp_cases")
  }
}

## ── Observations ───────────────────────────────────────

observations {
  weekly_cases : {
    projected  = incidence(infection)
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }
}

## ── Init ───────────────────────────────────────────────
# Minimal test initialization — single patch seeding.
# For a full 774-patch model, use per-patch init from a table
# or the distribute() function (v0.2).

init {
  S[child, p1]  = 100000
  S[adult, p1]  = 200000
  I[child, p1]  = I0
  # All other compartments/strata default to 0
}

## ── Output ─────────────────────────────────────────────

output {
  trajectories {
    every = 7 'days
    quantities {
      total_I    = I
      total_S    = S
      prevalence = I / N
    }
    format = parquet
  }

  flows {
    every = 7 'days
    quantities {
      weekly_infections = incidence(infection)
    }
    format = parquet
  }

  summary {
    peak_I      = max(I)
    total_cases = cumulative(infection)
    final_size  = at(R, t_end) / at(N, t_end)
    format      = tsv
  }

  synthetic {
    format = tsv
  }
}

## ── Simulation ─────────────────────────────────────────

simulate {
  from = 0 'days
  to   = 2 'years
}

## ── Scenarios ──────────────────────────────────────────

scenarios {
  with_sia {
    enable = [sia_round_1]
  }
}
```

---

## 25. Compilation Pipeline

```
.camdl file  →  [Parser]     →  AST
params.toml  →  [Loader]     ─┐
data files   →  [Loader]     ─┤
                               ▼
                          [Expander]   →  Expanded IR
                               │
                          [Validator]  →  type/dimension checks
                               │
                          [Serializer] →  model.ir.json
                               │
                          [Rust Runtime] → output files
```

**Parser** (Menhir): `.camdl` text → AST. ~60 grammar productions.

### 25.1 File-Level Grammar

A `.camdl` file is a sequence of declarations. Order does not matter — all
declarations are collected first, then resolved (forward references are valid).

```
file := declaration*

declaration :=
  | time_unit_decl                    # time_unit = 'days
  | origin_decl                       # origin = date("YYYY-MM-DD")
  | compartments_block                # compartments { ... }
  | parameters_block                  # parameters { ... }
  | dimensions_block                  # dimensions { dim = [...] | read(...) }
  | tables_block                      # tables { ... }
  | forcing_block                   # forcing { ... }
  | transitions_block                 # transitions { ... }
  | observations_block                # observations { ... }
  | interventions_block               # interventions { ... }
  | data_block                        # data { ... }
  | ode_block                         # ode { ... }
  | output_block                      # output { ... }
  | timepoints_block                  # timepoints { ... }
  | init_block                        # init { ... }
  | simulate_block                    # simulate { ... }
  | scenarios_block                   # scenarios { ... }
  | stratify_decl                     # stratify(by = ..., ...)
  | let_binding                       # let NAME = EXPR
```

**Mandatory** for a runnable model: `compartments`, `transitions`, `init`,
`simulate`, and `time_unit`. Everything else is optional.

**Mandatory** for `camdl check` (validation only): `compartments` and
`parameters`. No `simulate` or `init` required.

**Expander** (OCaml): indexed transitions → flat IR transitions, coupling sugar
→ explicit sums, `c in compartments` → per-compartment transitions,
`consecutive` → adjacent pair transitions, `where` → compile-time filtering, let
bindings → inlined expressions, unit normalization.

**Validator**: compartment arity checking, table dimension checking, index
variable scoping, parameter reference resolution, dimensional analysis.

**Serializer**: expanded IR → JSON (v0.1) or msgpack (large models).

**Runtime** (Rust): deserializes IR, evaluates propensities, simulates, writes
output. Knows nothing about the DSL — sees only flat compartments, transitions,
and expression ASTs.

---

## 26. Expansion Rules (DSL → IR Mapping)

Every DSL construct compiles to specific IR structures. This section documents
the mapping for each construct — the contract between the OCaml frontend and the
Rust backend.

### 26.1 Let Bindings

```
# DSL:
let N = S + E + I + R

# IR: everywhere N appears, inline the expression tree:
BinOp(Add, BinOp(Add, BinOp(Add, Pop("S"), Pop("E")), Pop("I")), Pop("R"))
```

After stratification, bare `S` in the let body becomes
`PopSum(["S_child", "S_adult"])`. N is always the global total.

### 26.2 Indexed Transitions

```
# DSL:
recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]
# with age = [child, adult]

# IR: two concrete transitions:
{ name: "recovery_child",
  stoichiometry: [("I_child", -1), ("R_child", 1)],
  rate: BinOp(Mul, Param("gamma"), Pop("I_child")),
  event_key: "recovery_child:{firing_index}" }

{ name: "recovery_adult",
  stoichiometry: [("I_adult", -1), ("R_adult", 1)],
  rate: BinOp(Mul, Param("gamma"), Pop("I_adult")),
  event_key: "recovery_adult:{firing_index}" }
```

### 26.3 Inflows

```
# DSL:
birth[p in patch] : --> S[child, p]
  @ mu * sum(a in age, N_local[a, p])

# IR (for each patch value, e.g., p1):
{ name: "birth_p1",
  stoichiometry: [("S_child_p1", 1)],
  rate: BinOp(Mul, Param("mu"),
    PopSum(["S_child_p1","E_child_p1",...,"R_adult_p1"])),
  event_key: "birth_p1:{firing_index}" }
```

`sum(a in age, N_local[a, p])` expands to the sum of all compartments in patch p
across all age groups — the compiler generates the `PopSum` from the known
compartment list and index bindings.

### 26.4 Projections

```
# DSL:
incidence(infection)           # un-indexed: sum of all expanded flows

# IR (with age stratification):
BinOp(Add,
  CumulativeFlow("infection_child"),
  CumulativeFlow("infection_adult"))

# DSL:
incidence(infection[child])    # indexed: specific stratum

# IR:
CumulativeFlow("infection_child")

# DSL:
prevalence(R)                  # bare: global total

# IR:
PopSum(["R_child", "R_adult"])
```

### 26.5 Interventions

```
# DSL:
sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180]

# IR (with age = [child, adult]):
# Intervention at t=180, two actions (one per stratum):
{ time: 180.0,
  actions: [
    FractionTransfer("S_child", "V_child", 0.80),
    FractionTransfer("S_adult", "V_adult", 0.80)
  ] }

# Each FractionTransfer is atomic:
#   delta = floor(Pop("S_child") * 0.80)
#   S_child -= delta
#   V_child += delta
# Delta computed from pre-intervention state.
```

### 26.6 Coupling Sugar

```
# DSL:
infection : S --> E @ beta * S * I / N {
  coupling[age = C_age]
}
# with age = [child, adult]

# Expands to primitive form:
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * sum(b in age,
      C_age[a,b] * I[b] / sum(c in compartments, c[b]))

# Which then expands to IR:
{ name: "infection_child",
  stoichiometry: [("S_child", -1), ("E_child", 1)],
  rate: BinOp(Mul, Param("beta"),
    BinOp(Mul, Pop("S_child"),
      BinOp(Add,
        BinOp(Mul, TableLookup("C_age", 0),
          BinOp(Div, Pop("I_child"),
            PopSum(["S_child","E_child","I_child","R_child"]))),
        BinOp(Mul, TableLookup("C_age", 1),
          BinOp(Div, Pop("I_adult"),
            PopSum(["S_adult","E_adult","I_adult","R_adult"])))))) }
```

The auto-generated denominator `sum(c in compartments, c[b])` becomes `PopSum`
of all compartments in stratum `b`.

### 26.7 Consecutive Pairs

```
# DSL:
aging[c in compartments, (a, a_next) in consecutive(age)]
  : c[a] --> c[a_next]
  @ (1 / age_dur[a]) * c[a]
# with age = [age_0_5, age_5_15, age_15_50], compartments = [S, I, R]

# Compiler generates pairs: (age_0_5, age_5_15), (age_5_15, age_15_50)
# × compartments: S, I, R
# = 2 pairs × 3 compartments = 6 IR transitions:

{ name: "aging_S_age_0_5",
  stoichiometry: [("S_age_0_5", -1), ("S_age_5_15", 1)],
  rate: BinOp(Mul,
    BinOp(Div, Const(1.0), TableLookup("age_dur", 0)),
    Pop("S_age_0_5")),
  event_key: "aging_S_age_0_5:{firing_index}" }

{ name: "aging_S_age_5_15",
  stoichiometry: [("S_age_5_15", -1), ("S_age_15_50", 1)],
  rate: BinOp(Mul,
    BinOp(Div, Const(1.0), TableLookup("age_dur", 1)),
    Pop("S_age_5_15")),
  event_key: "aging_S_age_5_15:{firing_index}" }

# ... (same pattern for I and R)
```

Both `a` and `a_next` are available in the rate expression. The last stratum
(`age_15_50` in this example) has no pair — no transition is generated.

### 26.8 Guard Clauses (`where`)

Guards are evaluated at compile time. The compiler instantiates all index
combinations, evaluates the guard, and **omits** transitions where the guard is
false. The IR has no concept of guards.

```
# DSL:
migrate[src in patch, dst in patch] : S[src] --> S[dst]
  @ mig[dst,src] * S[src]
  where src != dst
# with patch = [p1, p2, p3]

# Compiler evaluates:
#   (p1, p1): src == dst → SKIP
#   (p1, p2): src != dst → EMIT
#   (p1, p3): src != dst → EMIT
#   (p2, p1): src != dst → EMIT
#   (p2, p2): src == dst → SKIP
#   ...
# Result: 6 transitions (not 9)
```

### 26.9 Compartment Iteration (`c in compartments`)

The compiler expands `c in compartments` by substituting each integer
compartment name. When a compartment has more dimensions than the index
signature provides, the compiler **iterates over the omitted dimensions** to
satisfy the stoichiometry rule (§5.1).

```
# DSL:
death[c in compartments, a in age] : c[a] -->  @ mu * c[a]
# with compartments = [S, I, R] (R has extra immunity dimension)
# S dims: [age], R dims: [age, immunity]

# For S: straightforward
{ name: "death_S_child", stoichiometry: [("S_child", -1)],
  rate: BinOp(Mul, Param("mu"), Pop("S_child")) }
{ name: "death_S_adult", stoichiometry: [("S_adult", -1)],
  rate: BinOp(Mul, Param("mu"), Pop("S_adult")) }

# For R: compiler fills omitted immunity dimension, generating per-immunity:
{ name: "death_R_child_natural", stoichiometry: [("R_child_natural", -1)],
  rate: BinOp(Mul, Param("mu"), Pop("R_child_natural")) }
{ name: "death_R_child_vaccine", stoichiometry: [("R_child_vaccine", -1)],
  rate: BinOp(Mul, Param("mu"), Pop("R_child_vaccine")) }
# ... (adult × natural, adult × vaccine)
```

### 26.10 Interventions (All Dimensions)

Interventions on stratified compartments expand over **all** dimensions:

```
# DSL:
sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180]
# S and V have dimensions [age, patch] (2 × 774)

# IR: one FractionTransfer per (age × patch) = 1548 atomic transfers
{ time: 180.0,
  actions: [
    FractionTransfer("S_child_p1", "V_child_p1", 0.80),
    FractionTransfer("S_child_p2", "V_child_p2", 0.80),
    ...
    FractionTransfer("S_adult_p774", "V_adult_p774", 0.80)
  ] }
```

Each `FractionTransfer` is atomic: `delta = floor(source * fraction)` from
pre-intervention state, then `source -= delta, dest += delta`.

---

## 27. Errors and Validation

The compiler produces clear, domain-specific error messages. Errors are caught
at compile time, not simulation time.

### 27.0 Diagnostic Codes

Diagnostics carry a numeric code for programmatic consumption (e.g.,
`--json-errors` mode):

| Code | Kind    | Description                                                                                     |
| ---- | ------- | ----------------------------------------------------------------------------------------------- |
| E100 | Error   | Unknown index value in `[...]` — not a bound variable and not a member of any dimension         |
| E200 | Error   | Undeclared compartment or parameter referenced in expression                                    |
| E201 | Error   | Duplicate declaration (two parameters named `beta`, etc.)                                       |
| E202 | Error   | Wrong number of indices for compartment                                                         |
| E203 | Error   | Index belongs to wrong dimension (e.g., `C_age[i, s]` where `s : sex`)                          |
| E204 | Error   | Partial-stratification stoichiometry: destination compartment dimensions incompletely specified |
| W002 | Warning | Zero-firing transition (emitted at simulation time by `camdl-sim`)                              |
| W103 | Warning | Let binding name shadows a stratum value in some dimension                                      |

Diagnostics can be emitted as structured JSON by passing `--json-errors` to
`camdlc`:

```bash
camdlc check model.camdl --json-errors 2>errors.json
```

### 27.1 Dimension Errors

```
# Wrong number of indices
recovery[a in age] : I[a, s] --> R[a]  @ gamma * I[a, s]
# ERROR at line 42: I has dimensions [age] but was indexed with
#   [age, ???]. 's' is not bound — did you mean to add 's in sex'
#   to the transition index?

# Wrong dimension type for table
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * sum(j in sex, C_age[a, j] * I[j] / N[j])
# ERROR at line 45: C_age is declared as age × age, but index 2
#   ('j') is bound to 'sex' (via 'j in sex'). Did you mean
#   'j in age'?
```

### 27.2 Unbound Variables

```
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * I[a, s] / N
# ERROR at line 45: 's' is used in I[a, s] but is not bound.
#   I has dimensions [age, sex]. Bind 's' with 'sum(s in sex, ...)'
#   or add 's in sex' to the transition index.
```

### 27.3 Partial Stratification Stoichiometry

```
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]
# ERROR at line 55: R has dimensions [age, immunity] but destination
#   R[a] only specifies [age]. All dimensions of a destination
#   compartment must be specified in stoichiometry.
#   Did you mean: R[a, natural] or R[a, vaccine]?
```

### 27.4 Dimension Does Not Exist

```
recovery[a in age, r in habitat] : I[a, r] --> R[a, r]  @ gamma * I[a, r]
# ERROR at line 50: 'habitat' is not a declared dimension.
#   Declared dimensions: age, sex, patch.
```

### 27.5 Compartment Doesn't Have Dimension

```
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

waning[a in age] : I[a, natural] --> S[a]  @ wane * I[a, natural]
# ERROR at line 55: I does not have dimension 'immunity'.
#   I has dimensions: [age]. Only R has dimension 'immunity'.
#   Did you mean R[a, natural]?
```

### 27.6 Unit Errors

```
transitions {
  recovery : I --> R  @ gamma + I
  # where gamma : rate ('per_day) and I : count
}
# ERROR at line 33: cannot add rate (1/time) and count (dimensionless).
#   Did you mean 'gamma * I'?
```

### 27.7 Parameter Domain Errors

Checked when parameter values are supplied (not at model compile time):

```
# params.toml: rho = 1.5
# ERROR: parameter 'rho' is declared as probability (∈ [0, 1])
#   but supplied value is 1.5.
```

### 27.8 Scenario Validation

```
scenarios {
  high_coverage {
    scale = { beta = 1.5, beta = 2.0 }
  }
}
# ERROR: parameter 'beta' appears twice in scale operation.

scenarios {
  combined {
    compose = [variant, closure]
  }
}
# WARNING: scenarios 'variant' and 'closure' both modify parameter
#   'beta'. Composition is non-commutative; the result depends on
#   order. 'variant' is applied first, then 'closure'.
```

### 27.9 Self-Loop Detection

```
migrate[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]  @ mig[dst, src] * c[src]
# WARNING: transition 'migrate' generates self-loops where
#   src == dst. Self-loops waste computation (Gillespie fires
#   them but state doesn't change). Add 'where src != dst' to
#   filter, or ensure mig[i,i] = 0 for all i.
```

### 27.10 Name Resolution

Names are resolved in order: compartments → parameters → let bindings → forcing
→ tables. The compiler reports errors for:

- **Shadowing reserved identifiers**: `t_start`, `t_end`, `compartments`, etc.
- **Duplicate declarations**: two parameters named `beta`, two compartments
  named `S`.
- **Ambiguous references**: a name exists in multiple namespaces (e.g., a
  parameter and a compartment both named `N`). The compiler errors rather than
  guessing.

### 27.11 Compiler Reporting

For every model, `camdl check` reports:

```
Model: seir_age_seasonal
  Compartments: 5 base × 2 age × 774 patch = 7740 expanded
  Transitions: 8 base → 47,892 expanded
  Parameters: 13 declared
  Tables: 6 (2 external)
  Observations: 1 stream
  Interventions: 1 (inactive by default)
  Estimated Gillespie event types: 47,892
```

This gives the user a quick sanity check on model size before simulation.

---

## 28. Primitive Summary

The language is built on these composable primitives:

```
# State
compartments { NAME, ... }           integer-valued populations
compartments { NAME : real }         continuous-valued state

# Dimensions
dimensions { DIM = [...] }           declare dimension levels (inline)
dimensions { DIM = read(FILE, column = "COL") }  dimension levels from file
stratify(by = DIM)                   apply dimension to all compartments
stratify(by = DIM, only = [COMP, ...])  partial stratification

# Indexing
NAME[val]                            concrete stratum access
NAME[var]                            index variable access
NAME                                 bare = sum over all strata
[var in dim]                         bind index variable to dimension
sum(var in dim, expr)                sum over dimension

# Iteration
[c in compartments]                  iterate over integer compartment names
(a, a_next) in consecutive(dim)      iterate over adjacent pairs

# Transitions
NAME[indices] : SRC --> DST @ RATE   transfer with indexed stoichiometry
NAME[indices] : --> DST @ RATE       inflow (birth, importation)
NAME[indices] : SRC --> @ RATE       outflow (death)
... where PRED                       guard clause (compile-time filtering)

# Rate wrappers
@ overdispersed(RATE, σ²)            Gamma-Poisson (NegBinomial) draws
@ deterministic(RATE)                nearbyint(rate × dt), no stochasticity

# Data
table : dim × dim unit = [...]       typed, shape-checked, unit-annotated
let name[indices] = expr              computed quantity (family of values)

# Math functions
exp(x), log(x), sqrt(x)             standard math (unary)
abs(x), floor(x), ceil(x)           rounding and absolute value
mod(a, b)                            Euclidean remainder (binary)

# Time
t                                    current simulation time
at(expr, timepoint)                  value at specific time
max(expr), cumulative(trans), ...    summary functions over trajectories

# Forcing (time-dependent functions)
forcing { NAME : sinusoidal { ... } }     smooth seasonal
forcing { NAME : periodic { ... } }       repeating step (values or on = [lo:hi, ...])
forcing { NAME : piecewise { ... } }      non-repeating step
forcing { NAME : interpolated { ... } }   data-driven (linear or spline)

# Reserved identifiers
t, t_start, t_end, compartments, sum, consecutive
```
